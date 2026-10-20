"""C2 variant — flat dual-stream g2stack for kv_iters==2 uniform-noncausal.

Standalone Gate-0 development file (Stage 3 iter5).  gqa.py is NOT modified.

Provenance (audit: audit_flat_kv2_iter5.py, diff: diff_iter5_kv2.txt):
  * ARITHMETIC母体: shipped `_gqa_kernel_prefill_g2stack` (gqa.py:2472-2969,
    ns=2/ci=2 batch form) — the SM/AO/FIN op sequences, fill values, neg_sm
    h*2+cur / sumexp_is,r_factors kv*2+h slot semantics are inherited op-for-op
    (index remap only: batch i -> kvpar, new task-parity dim for acc_o/sumexp).
  * STRUCTURE范本: shipped `_gqa_kernel_prefill_g2stack_flat` (gqa.py:2982-3423,
    kv==1 flattened stream) — macro emission pattern, alloc_var epilogue
    binding, init/destroy token layout, buffer allocation, ws/L1/L0 geometry.

Mechanism (design doc: c2_design_iter5.md §B):
  Flattened per-core stream u = 2*t + kv (U = 2*mc), task_of(u)=my_start+u//2,
  kv_of(u)=u%2 (UNIFORM: noncausal => kv_useful==2 for every task; causal is
  factory-rejected).  Deletes the g2stack per-task batch drain/refill and the
  batch-grain phase alternation (R4 measured: C-id2 4.21us/task waiting the SM
  batch, V-id4 4.06us/task waiting the G2 batch) by running the PROVEN kv==1
  flat steady state across task boundaries at u granularity:
    C step (pair-unrolled, G1-first per iter4 hard constraint):
        G1(t,kv1) -> G2(t,kv0) -> Qload(t+1) -> G1(t+1,kv0) -> G2(t,kv1)
    V beat (pair-unrolled AO -> [FIN] -> SM(+2)):
        AO(t,kv0) -> SM(t+1,kv0) -> AO(t,kv1) -> FIN(t) -> SM(t+1,kv1)
  Gating is inherited VERBATIM from the shipped kv==1 flat kernel (iter4
  framework findings 1-3: ns=2 ws slots, SINGLE-token release semaphores,
  FIFO effective gates SM(u)<-G1(u) / G1(u)<-SM(u-1) / SM(u)<-G2(u-1) /
  G2(u)<-AO(u-1) / AO(u)<-G2(u)): NO ring deepening, NO step reorder, NO new
  cross ids (0-5; 6-10 spare untouched), NO token-count change.

  V-side state-slot liveness (c2_design_iter5.md §B.4, re-derived for kv==2,
  NOT copied from kv==1): acc_o/sumexp are TASK-parity doubled (idx (t%2)*2+h)
  because AO(t,kv0)->AO(t,kv1) is a REAL intra-task rescale chain (kv==1's
  u-parity == task-parity degeneracy no longer holds); sumexp_is/r_factors keep
  the SM(u) write -> AO(u) read -> SM(u+2) overwrite window on the KV-parity
  slot (idx (u%2)*2+h, == g2stack's i*2+h); neg_sm keeps g2stack's h*2+cur
  semantics with a WHOLE-BUFFER cap refill at the head of SM(t,kv0) (exactly
  g2stack's per-task fill position/value; safe: previous task's readers
  finished >= 1 beat earlier, V-serial).  Per-task acc_o/sumexp fills at the
  AO(t,kv0) head (iter2 NaN-leak lesson — fills are NEVER removed).

  Q protocol (the one new LOCAL handshake — single-token form; the first
  Gate-0 attempt used a double set/wait pair and DEADLOCKED at the prologue
  Qload, so local-event token stacking is treated as unsupported):
    G1(*,kv0): wait MTE2->MTE1 full (Qload landed); copy q_l1->l0a[0]; NO set
    G1(*,kv1): copy q_l1->l0a[1] (content resident — no Qload issues between
               the two reads, C-stream serial issue); set MTE1->MTE2 free —
               the SINGLE free token: SetFlag drains the in-order MTE1 queue,
               so it implies BOTH q_l1 reads of the task completed
    Qload(t+1): wait MTE1->MTE2 free x1; copy Q; set MTE2->MTE1 full x1
    init: free x1; destroy: wait free x1 — counts and pending depth (<=1)
    IDENTICAL to the shipped flat kv1 Q protocol.
  Counts (free dir): sets 2+2U... see the flag-count audit table in
  c2_design_iter5.md §B.3 (balanced; pending depth <= 2 local, <= 3 cross,
  <= 6 safety bound).

Arithmetic sequence per (task, kv, h) is IDENTICAL to the shipped g2stack
ns=2 (pure scheduling change) => outputs must be BIT-IDENTICAL (Gate 0
criterion; non-bit-identical = bug, never a tolerance question).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tilelang
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout

from gqa import P28_EXPERT_PASS_CONFIGS, P28_NUM_CORES, _BIG_NEG  # noqa: E402


@tilelang.jit(out_idx=[4], workspace_idx=[5, 6, 7], pass_configs=P28_EXPERT_PASS_CONFIGS)
def _gqa_kernel_prefill_g2stack_flat_kv2(
    B,
    S_padded,
    S_kv_padded,
    sm_scale,
    is_causal,
    N_q=8,
    N_kv=2,
    D=128,
    dtype="float16",
    block_M=64,
    skv_minus_s=0,
    use_mask=True,
    block_N=128,
):
    """Flat cross-task 4-phase pipelined g2stack (kv_iters==2 uniform specialization)."""
    accum_dtype = "float"

    # max-after-mask neg_sm init cap (verbatim g2stack semantics)
    neg_sm_cap = -sm_scale * _BIG_NEG * 0.5

    G = N_q // N_kv
    G_batches = G // 2
    q_blocks = S_padded // block_M
    kv_iters = S_kv_padded // block_N
    block_num = B * N_kv * G_batches * q_blocks

    # ---- factory gate (host-side, static) — C2 uniform-noncausal subclass ----
    assert D == 128 and block_N == 128 and block_M == 64, "flat-kv2: D==block_N==128, bm==64"
    assert dtype in ("float16", "bfloat16")
    assert kv_iters == 2, "flat-kv2: kv_iters==2 only (S_kv_padded == 2*block_N)"
    assert not is_causal, "flat-kv2: UNIFORM subclass only — noncausal (kv_useful==2 for every task)"
    assert G % 2 == 0, "flat-kv2 requires even G (stacked head pairs)"
    assert block_num >= 2 * P28_NUM_CORES, "flat-kv2 requires >= 2 tasks per core"
    # kv_useful == 2 for EVERY task by construction (noncausal: skip_limit =
    # 1<<30 -> kv_u_raw huge -> clamp to kv_iters == 2); no causal block-skip
    # logic exists in this kernel.  The in-block additive mask (pad columns)
    # still applies when use_mask.

    half = block_M // 2  # per-vid row strip within one head's segment
    M_stack = 2 * block_M  # stacked GEMM M dim (2 Q heads)

    q_tasks = block_num // P28_NUM_CORES
    r_tasks = block_num % P28_NUM_CORES

    # Cross-core semaphores (C <-> V) — verbatim g2stack/flat allocation
    SEM_WS1_C2V = 0  # ws_s ready:      C(FIX) -> V
    SEM_WS1_V2C = 1  # ws_s consumed:   V(MTE2) -> C
    SEM_WS2_V2C = 2  # ws_p ready:      V(MTE3) -> C
    SEM_WS2_C2V = 3  # ws_p consumed:   C(MTE2) -> V
    SEM_WS3_C2V = 4  # ws_o ready:      C(FIX) -> V
    SEM_WS3_V2C = 5  # ws_o consumed:   V(MTE2) -> C

    # Local directed event ids (per pipe pair) — verbatim g2stack/flat allocation
    SIG_K_L1 = 0  # MTE2 <-> MTE1 (k_l1[side] ownership, slots 0/1)
    SIG_V_L1 = 2  # MTE2 <-> MTE1 (v_l1[side] ownership, slots 0/1)
    SIG_P_L1 = 4  # MTE2 <-> MTE1 (p_l1 ownership)
    SIG_Q_L1 = 5  # MTE2 <-> MTE1 (q_l1 ownership; kv2: single free-token, set by the kv1 reader)
    SIG_L0AB = 0  # MTE1 <-> M (l0a/l0b slots 0/1)
    SIG_L0C = 0  # M <-> FIX (l0c slots 0/1)
    SIG_IO_UB = 0  # MTE2 <-> V (io_buf slots 0/1)
    SIG_MASK = 2  # MTE2 <-> V (mask_v ready/free)
    SIG_S_HALF = 0  # V <-> MTE3 (acc_s_half ownership)

    def task_range(cid_val):
        start = cid_val * q_tasks + T.if_then_else(cid_val < r_tasks, cid_val, r_tasks)
        count = q_tasks + T.if_then_else(cid_val < r_tasks, 1, 0)
        return start, count

    def _ids(tid):
        """Task index decomposition for a flat-stream task id (TIR expr)."""
        bx = tid % q_blocks
        gb = (tid // q_blocks) % G_batches
        kg = (tid // (q_blocks * G_batches)) % N_kv
        bz = tid // (q_blocks * G_batches * N_kv)
        hq0 = kg * G + gb * 2
        return bx, gb, kg, bz, hq0

    # ==================== C-side emission macros ====================

    @T.macro
    def emit_qload(x_t, Q, q_l1, my_start):
        """MTE2: Q rows of task (my_start+x_t) stacked into q_l1.

        kv2 SINGLE free-token protocol (the first attempt used a double
        set/wait pair and DEADLOCKED at the prologue Qload — local event
        ids evidently do NOT stack tokens like the cross sems do; every
        shipped local protocol is strict depth-1 ping-pong): the ONE
        MTE1->MTE2 free token is set by the task's SECOND reader G1(*,kv1);
        SetFlag drains the in-order MTE1 queue to that point, so the token
        implies BOTH q_l1 reads (kv0 + kv1) completed.  Depth <= 1 on every
        local id, identical in shape to the shipped flat-kv1 Q protocol.
        """
        bx, gb, kg, bz, hq0 = _ids(my_start + x_t)
        T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
        T.copy(Q[bz, hq0, bx * block_M : (bx + 1) * block_M, :], q_l1[0:block_M, :])
        T.copy(
            Q[bz, hq0 + 1, bx * block_M : (bx + 1) * block_M, :],
            q_l1[block_M:M_stack, :],
        )
        T.set_flag("MTE2", "MTE1", SIG_Q_L1)

    @T.macro
    def emit_g1(x_t, kvpar, K, V, ws_s, q_l1, k_l1_0, k_l1_1, v_l1_0, v_l1_1, l0a, l0b, l0c, cid, my_start):
        """GEMM1(task my_start+x_t, kv block kvpar): K/V GM->L1 (V prefetched
        for this task's G2(kvpar)), q/k -> l0[kvpar], mma -> l0c[kvpar] ->
        FIX -> ws_s[cid, kvpar].  kvpar is a PYTHON LITERAL (static slots,
        static flag ids, static GM column windows)."""
        bx, gb, kg, bz, hq0 = _ids(my_start + x_t)
        T.wait_cross_flag(SEM_WS1_V2C)
        # K(x_t, kvpar) load (consumed by this block's MTE1 below)
        T.wait_flag("MTE1", "MTE2", SIG_K_L1 + kvpar)
        if kvpar == 0:
            T.copy(K[bz, kg, 0:block_N, :], k_l1_0)
        else:
            T.copy(K[bz, kg, block_N : 2 * block_N, :], k_l1_1)
        T.set_flag("MTE2", "MTE1", SIG_K_L1 + kvpar)
        # V(x_t, kvpar) prefetch for G2(kvpar) of the SAME task (issued 1-4
        # emissions later): slot kvpar is free — the previous same-parity G2's
        # MTE1 read released it (SIG_V_L1+kvpar wait)
        T.wait_flag("MTE1", "MTE2", SIG_V_L1 + kvpar)
        if kvpar == 0:
            T.copy(V[bz, kg, 0:block_N, :], v_l1_0)
        else:
            T.copy(V[bz, kg, block_N : 2 * block_N, :], v_l1_1)
        T.set_flag("MTE2", "MTE1", SIG_V_L1 + kvpar)
        # MTE1: q_l1 -> l0a[kvpar].  kv0 waits the Qload full token (fresh
        # load); kv1 SKIPS the wait — same resident content, no Qload issues
        # between the two reads (C-stream serial issue order).  Only kv1's
        # read RELEASES the single free token: SetFlag drains the in-order
        # MTE1 queue to that point, so the token implies BOTH reads of the
        # task completed (no local-flag token stacking — the first Gate-0
        # attempt's double set/wait deadlocked at the prologue Qload).
        T.wait_flag("M", "MTE1", SIG_L0AB + kvpar)
        if kvpar == 0:
            T.wait_flag("MTE2", "MTE1", SIG_Q_L1)
        T.copy(q_l1, l0a[kvpar, :, :])
        if kvpar == 1:
            T.set_flag("MTE1", "MTE2", SIG_Q_L1)
        T.wait_flag("MTE2", "MTE1", SIG_K_L1 + kvpar)
        if kvpar == 0:
            T.copy(k_l1_0, l0b[0, :, :], transpose=True)
        else:
            T.copy(k_l1_1, l0b[1, :, :], transpose=True)
        T.set_flag("MTE1", "MTE2", SIG_K_L1 + kvpar)
        T.set_flag("MTE1", "M", SIG_L0AB + kvpar)
        # M
        T.wait_flag("MTE1", "M", SIG_L0AB + kvpar)
        T.wait_flag("FIX", "M", SIG_L0C + kvpar)
        T.mma(l0a[kvpar, :, :], l0b[kvpar, :, :], l0c[kvpar, :, :], init=True)
        T.set_flag("M", "MTE1", SIG_L0AB + kvpar)
        T.set_flag("M", "FIX", SIG_L0C + kvpar)
        # FIX
        T.wait_flag("M", "FIX", SIG_L0C + kvpar)
        T.copy(l0c[kvpar, :, :], ws_s[cid, kvpar, :, :])
        T.set_flag("FIX", "M", SIG_L0C + kvpar)
        T.set_cross_flag("FIX", SEM_WS1_C2V)

    @T.macro
    def emit_g2(kvpar, ws_p, ws_o, v_l1_0, v_l1_1, p_l1, l0a, l0b, l0c, cid):
        """GEMM2(kvpar): ws_p[cid,kvpar] -> p_l1 -> l0a[kvpar],
        v_l1[kvpar] -> l0b[kvpar] (prefetched by this task's G1(kvpar)),
        mma -> l0c[kvpar] -> FIX -> ws_o[cid,kvpar].  Task-independent
        (all addresses are cid/slot based) — no x_t needed."""
        T.wait_cross_flag(SEM_WS3_V2C)
        T.wait_flag("MTE1", "MTE2", SIG_P_L1)
        T.wait_cross_flag(SEM_WS2_V2C)
        T.copy(ws_p[cid, kvpar, :, :], p_l1)
        T.set_flag("MTE2", "MTE1", SIG_P_L1)
        T.wait_flag("MTE2", "MTE1", SIG_V_L1 + kvpar)
        T.wait_flag("M", "MTE1", SIG_L0AB + kvpar)
        if kvpar == 0:
            T.copy(v_l1_0, l0b[0, :, :])
        else:
            T.copy(v_l1_1, l0b[1, :, :])
        T.set_flag("MTE1", "MTE2", SIG_V_L1 + kvpar)
        T.wait_flag("MTE2", "MTE1", SIG_P_L1)
        T.copy(p_l1, l0a[kvpar, :, :])
        T.set_flag("MTE1", "MTE2", SIG_P_L1)
        T.set_flag("MTE1", "M", SIG_L0AB + kvpar)
        T.wait_flag("MTE1", "M", SIG_L0AB + kvpar)
        T.wait_flag("FIX", "M", SIG_L0C + kvpar)
        T.mma(l0a[kvpar, :, :], l0b[kvpar, :, :], l0c[kvpar, :, :], init=True)
        T.set_flag("M", "MTE1", SIG_L0AB + kvpar)
        T.set_flag("M", "FIX", SIG_L0C + kvpar)
        T.wait_flag("M", "FIX", SIG_L0C + kvpar)
        T.copy(l0c[kvpar, :, :], ws_o[cid, kvpar, :, :])
        T.set_flag("FIX", "M", SIG_L0C + kvpar)
        T.set_cross_flag("FIX", SEM_WS3_C2V)
        T.set_cross_flag("MTE2", SEM_WS2_C2V)

    # ==================== V-side emission macros ====================

    @T.macro
    def emit_sm(x_t, kvpar, first_kv, Mask, ws_s, ws_p, io_buf, work_ub, buf_2d, acc_s_half, mask_v, neg_sm, sumexp_is, r_factors, cid, vid, my_start):
        """Softmax(task my_start+x_t, kv block kvpar): ws_s[cid,kvpar] ->
        online-softmax -> ws_p[cid,kvpar].

        g2stack phase-1 body verbatim with i -> kvpar (STATIC), mask column
        window idx*block_N -> kvpar*block_N (same value: idx = k*ns+i = kv at
        num_outer==1).  State written: neg_sm[h*2+kvpar] (prv h*2+(1-kvpar)),
        sumexp_is[kvpar*2+h], r_factors[kvpar*2+h].  first_kv (kv0) refills
        neg_sm to the cap — EXACTLY g2stack's per-task head fill (position:
        head of this macro; previous task's last neg_sm readers ran >= 1 beat
        earlier in the V-serial stream; cur slot is fully overwritten by
        reduce_max before any read).
        """
        bx, gb, kg, bz, hq0 = _ids(my_start + x_t)
        T.wait_cross_flag(SEM_WS2_C2V)
        T.wait_cross_flag(SEM_WS1_C2V)
        if first_kv:
            T.tile.fill(neg_sm, neg_sm_cap)
        for h in T.serial(2):
            row0 = h * block_M + vid * half
            io_side = h

            T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)
            T.copy(ws_s[cid, kvpar, row0 : row0 + half, :], io_buf[io_side, :, :])
            T.set_flag("MTE2", "V", SIG_IO_UB + io_side)

            if use_mask:
                T.set_flag("V", "MTE2", SIG_MASK)
                T.wait_flag("V", "MTE2", SIG_MASK)
                T.copy(
                    Mask[
                        bz,
                        bx * block_M + vid * half : bx * block_M + vid * half + half,
                        kvpar * block_N : (kvpar + 1) * block_N,
                    ],
                    mask_v,
                )
                T.set_flag("MTE2", "V", SIG_MASK)

            T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)
            T.copy(io_buf[io_side, :, :], work_ub)
            T.set_flag("V", "MTE2", SIG_IO_UB + io_side)

            if use_mask:
                T.wait_flag("MTE2", "V", SIG_MASK)
                T.tile.add(work_ub, work_ub, mask_v)
            T.reduce_max(work_ub, neg_sm[h * 2 + kvpar, :, :], dim=-1)
            T.tile.mul(neg_sm[h * 2 + kvpar, :, :], neg_sm[h * 2 + kvpar, :, :], -sm_scale)
            T.tile.min(neg_sm[h * 2 + kvpar, :, :], neg_sm[h * 2 + kvpar, :, :], neg_sm[h * 2 + (1 - kvpar), :, :])

            T.tile.broadcast(buf_2d, neg_sm[h * 2 + kvpar, :, :])
            T.tile.axpy(buf_2d, work_ub, sm_scale)
            T.tile.exp(work_ub, buf_2d)

            T.reduce_sum(work_ub, sumexp_is[kvpar * 2 + h, :, :], dim=-1)
            T.tile.sub(r_factors[kvpar * 2 + h, :, :], neg_sm[h * 2 + kvpar, :, :], neg_sm[h * 2 + (1 - kvpar), :, :])

            T.wait_flag("MTE3", "V", SIG_S_HALF)
            T.copy(work_ub, acc_s_half)
            T.set_flag("V", "MTE3", SIG_S_HALF)

            T.wait_flag("V", "MTE3", SIG_S_HALF)
            T.copy(acc_s_half, ws_p[cid, kvpar, row0 : row0 + half, :])
            T.set_flag("MTE3", "V", SIG_S_HALF)
        T.set_cross_flag("MTE3", SEM_WS2_V2C)
        T.set_cross_flag("MTE2", SEM_WS1_V2C)

    @T.macro
    def emit_ao(x_t, kvpar, first_kv, ws_o, io_buf, work_ub, buf_2d, acc_o, sumexp, sumexp_is, r_factors, cid, vid):
        """O-accumulation(task-offset x_t, kv block kvpar): rescale by
        r_factors[kvpar*2+h], add the ws_o[cid,kvpar] chunk.

        g2stack phase-2 body verbatim with i -> kvpar (STATIC) and the
        acc_o/sumexp row-state slot h -> (t%2)*2+h (TASK parity — the intra
        task AO(kv0)->AO(kv1) rescale chain must land in the SAME slot;
        u-parity would split it, see c2_design_iter5.md §B.4).  first_kv
        (kv0) does the per-task zero fills at the AO head ([S3-iter2-B fix]
        pattern — fills are NEVER removed: fill-free self-reset leaks NaN
        across tasks; position is V-order-safe, the previous same-parity
        readers AO(t-2,*)/FIN(t-2) ran 3+ beats earlier).  Arithmetic
        sequence per (t,kv,h) then matches shipped g2stack exactly
        (fill -> exp r -> sumexp*r -> +is -> acc_o*r -> +ws_o).
        """
        tpar = x_t % 2
        T.wait_cross_flag(SEM_WS3_C2V)
        for h in T.serial(2):
            row0 = h * block_M + vid * half
            if first_kv:
                T.tile.fill(acc_o[tpar * 2 + h, :, :], 0.0)
                T.tile.fill(sumexp[tpar * 2 + h, :, :], 0.0)
            T.tile.exp(r_factors[kvpar * 2 + h, :, :], r_factors[kvpar * 2 + h, :, :])
            T.tile.mul(sumexp[tpar * 2 + h, :, :], sumexp[tpar * 2 + h, :, :], r_factors[kvpar * 2 + h, :, :])
            T.tile.add(sumexp[tpar * 2 + h, :, :], sumexp[tpar * 2 + h, :, :], sumexp_is[kvpar * 2 + h, :, :])
            T.tile.broadcast(buf_2d, r_factors[kvpar * 2 + h, :, :])
            T.tile.mul(acc_o[tpar * 2 + h, :, :], acc_o[tpar * 2 + h, :, :], buf_2d)

            io_side = h
            T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)
            T.copy(ws_o[cid, kvpar, row0 : row0 + half, :], io_buf[io_side, :, :])
            T.set_flag("MTE2", "V", SIG_IO_UB + io_side)

            T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)
            T.copy(io_buf[io_side, :, :], work_ub)
            T.set_flag("V", "MTE2", SIG_IO_UB + io_side)

            T.tile.add(acc_o[tpar * 2 + h, :, :], acc_o[tpar * 2 + h, :, :], work_ub)
        T.set_cross_flag("MTE2", SEM_WS3_V2C)

    @T.macro
    def emit_fin(x_t, Output, acc_o, sumexp, buf_2d, acc_s_half, vid, my_start):
        """Final normalize + contiguous per-(h, vid) output box write for
        task (my_start+x_t).  Verbatim g2stack finalize with the acc_o/sumexp
        slot h -> (x_t%2)*2+h (task parity, matching emit_ao)."""
        tpar = x_t % 2
        bx, gb, kg, bz, hq0 = _ids(my_start + x_t)
        for h in T.serial(2):
            T.tile.add(sumexp[tpar * 2 + h, :, :], sumexp[tpar * 2 + h, :, :], 1e-30)
            T.tile.broadcast(buf_2d, sumexp[tpar * 2 + h, :, :])
            T.tile.div(acc_o[tpar * 2 + h, :, :], acc_o[tpar * 2 + h, :, :], buf_2d)
            T.wait_flag("MTE3", "V", SIG_S_HALF)
            T.copy(acc_o[tpar * 2 + h, :, :], acc_s_half)
            T.set_flag("V", "MTE3", SIG_S_HALF)
            T.wait_flag("V", "MTE3", SIG_S_HALF)
            T.copy(
                acc_s_half,
                Output[bz, hq0 + h, bx * block_M + vid * half : bx * block_M + vid * half + half, :],
            )
            T.set_flag("MTE3", "V", SIG_S_HALF)

    @T.prim_func
    def main(
        Q: T.Tensor([B, N_q, S_padded, D], dtype),  # type: ignore  BHSD
        K: T.Tensor([B, N_kv, S_kv_padded, D], dtype),  # type: ignore  BHSD
        V: T.Tensor([B, N_kv, S_kv_padded, D], dtype),  # type: ignore  BHSD
        Mask: T.Tensor([B, S_padded, S_kv_padded], accum_dtype),  # type: ignore
        Output: T.Tensor([B, N_q, S_padded, D], dtype),  # type: ignore  BHSD
        ws_s: T.Tensor([P28_NUM_CORES, 2, M_stack, block_N], accum_dtype),  # type: ignore
        ws_p: T.Tensor([P28_NUM_CORES, 2, M_stack, block_N], dtype),  # type: ignore
        ws_o: T.Tensor([P28_NUM_CORES, 2, M_stack, D], accum_dtype),  # type: ignore
    ):
        with T.Kernel(P28_NUM_CORES, is_npu=True) as (cid, vid):
            # ===== C-side buffers (verbatim g2stack / flat kv1) =====
            q_l1 = T.alloc_L1([M_stack, D], dtype)
            k_l1_0 = T.alloc_L1([block_N, D], dtype)
            k_l1_1 = T.alloc_L1([block_N, D], dtype)
            v_l1_0 = T.alloc_L1([block_N, D], dtype)
            v_l1_1 = T.alloc_L1([block_N, D], dtype)
            p_l1 = T.alloc_L1([M_stack, block_N], dtype)

            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1_0: make_nz_layout(k_l1_0),
                    k_l1_1: make_nz_layout(k_l1_1),
                    p_l1: make_zn_layout(p_l1),
                    v_l1_0: make_zn_layout(v_l1_0),
                    v_l1_1: make_zn_layout(v_l1_1),
                }
            )

            l0a = T.alloc_L0A([2, M_stack, D], dtype)
            l0b = T.alloc_L0B([2, D, block_N], dtype)
            l0c = T.alloc_L0C([2, M_stack, block_N], accum_dtype)

            # ===== V-side buffers (per vid) — SAME shapes as flat kv1 =====
            # acc_o/sumexp: TASK-parity doubled (idx (t%2)*2+h) — kv2's intra
            # task chain lives in one slot across 2 u; sumexp_is/r_factors:
            # KV-parity (idx (u%2)*2+h, == g2stack i*2+h); neg_sm: g2stack
            # h*2+cur (cur = kvpar).  Footprint identical to flat kv1.
            io_buf = T.alloc_ub([2, half, block_N], accum_dtype)
            work_ub = T.alloc_ub([half, block_N], accum_dtype)
            buf_2d = T.alloc_ub([half, block_N], accum_dtype)
            acc_s_half = T.alloc_ub([half, block_N], dtype)
            acc_o = T.alloc_ub([4, half, D], accum_dtype)  # idx (t%2)*2 + h
            mask_v = T.alloc_ub([half, block_N], accum_dtype)
            neg_sm = T.alloc_ub([4, half, 1], accum_dtype)  # idx h*2 + cur (g2stack semantics)
            sumexp = T.alloc_ub([4, half, 1], accum_dtype)  # idx (t%2)*2 + h
            sumexp_is = T.alloc_ub([4, half, 1], accum_dtype)  # idx (u%2)*2 + h
            r_factors = T.alloc_ub([4, half, 1], accum_dtype)  # idx (u%2)*2 + h

            my_start, my_count = task_range(cid)

            # Epilogue task-offset scalar (my_count embeds the task_range
            # T.if_then_else remainder term -> must be bound through an
            # alloc_var at a STATEMENT position before entering macro arg /
            # tile-index positions — iter2 framework finding 3, verbatim
            # flat-kv1 xc1/xv1 pattern).  Per-scope pair: written AND read
            # inside one scope only (CV-split safe).
            xc_t = T.alloc_var("int32")
            xv_t = T.alloc_var("int32")

            # ============================ C scope ============================
            with T.Scope("C"):
                # init: pretend consumers already released everything
                # (kv2: TWO Q free tokens — Qload's double-wait protocol)
                T.set_cross_flag("MTE2", SEM_WS2_C2V)
                T.set_flag("MTE1", "MTE2", SIG_K_L1)
                T.set_flag("MTE1", "MTE2", SIG_K_L1 + 1)
                T.set_flag("MTE1", "MTE2", SIG_P_L1)
                T.set_flag("MTE1", "MTE2", SIG_V_L1)
                T.set_flag("MTE1", "MTE2", SIG_V_L1 + 1)
                T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                T.set_flag("M", "MTE1", SIG_L0AB)
                T.set_flag("M", "MTE1", SIG_L0AB + 1)
                T.set_flag("FIX", "M", SIG_L0C)
                T.set_flag("FIX", "M", SIG_L0C + 1)

                # ---- prologue: Qload(t0), G1(t0,kv0)   (my_count >= 2 by gate) ----
                emit_qload(0, Q, q_l1, my_start)
                emit_g1(0, 0, K, V, ws_s, q_l1, k_l1_0, k_l1_1, v_l1_0, v_l1_1, l0a, l0b, l0c, cid, my_start)

                # ---- steady pair-steps tt = 0 .. my_count-2:
                #      G1(tt,kv1) -> G2(tt,kv0) -> Qload(tt+1)
                #      -> G1(tt+1,kv0) -> G2(tt,kv1)
                #      (u order: 2tt+1, 2tt, [q], 2tt+2, 2tt+1 — G1-first,
                #       Qload at the task boundary only) ----
                for tt in T.serial(my_count - 1):
                    emit_g1(tt, 1, K, V, ws_s, q_l1, k_l1_0, k_l1_1, v_l1_0, v_l1_1, l0a, l0b, l0c, cid, my_start)
                    emit_g2(0, ws_p, ws_o, v_l1_0, v_l1_1, p_l1, l0a, l0b, l0c, cid)
                    emit_qload(tt + 1, Q, q_l1, my_start)
                    emit_g1(tt + 1, 0, K, V, ws_s, q_l1, k_l1_0, k_l1_1, v_l1_0, v_l1_1, l0a, l0b, l0c, cid, my_start)
                    emit_g2(1, ws_p, ws_o, v_l1_0, v_l1_1, p_l1, l0a, l0b, l0c, cid)

                # ---- epilogue: G1(t_last,kv1), G2(t_last,kv0), G2(t_last,kv1) ----
                xc_t = my_count - 1
                emit_g1(xc_t, 1, K, V, ws_s, q_l1, k_l1_0, k_l1_1, v_l1_0, v_l1_1, l0a, l0b, l0c, cid, my_start)
                emit_g2(0, ws_p, ws_o, v_l1_0, v_l1_1, p_l1, l0a, l0b, l0c, cid)
                emit_g2(1, ws_p, ws_o, v_l1_0, v_l1_1, p_l1, l0a, l0b, l0c, cid)

                # destroy: consume outstanding init-direction flags
                # (residuals identical to shipped flat kv1: +1 per released id)
                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                T.wait_flag("MTE1", "MTE2", SIG_K_L1 + 1)
                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                T.wait_flag("MTE1", "MTE2", SIG_V_L1 + 1)
                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                T.wait_flag("M", "MTE1", SIG_L0AB)
                T.wait_flag("M", "MTE1", SIG_L0AB + 1)
                T.wait_flag("FIX", "M", SIG_L0C)
                T.wait_flag("FIX", "M", SIG_L0C + 1)

            # ============================ V scope ============================
            with T.Scope("V"):
                # init: pretend producers already released everything +
                # one-shot static state init (flat-kv1 pattern; the per-task
                # neg_sm cap refill lives in emit_sm(first_kv), the per-task
                # acc_o/sumexp fills in emit_ao(first_kv))
                T.set_cross_flag("MTE2", SEM_WS1_V2C)
                T.set_cross_flag("MTE2", SEM_WS3_V2C)
                T.set_flag("V", "MTE2", SIG_IO_UB)
                T.set_flag("V", "MTE2", SIG_IO_UB + 1)
                T.set_flag("MTE3", "V", SIG_S_HALF)
                T.tile.fill(acc_o, 0.0)
                T.tile.fill(sumexp, 0.0)
                T.tile.fill(neg_sm, neg_sm_cap)

                # ---- prologue beats: SM(t0,kv0), SM(t0,kv1) ----
                emit_sm(0, 0, True, Mask, ws_s, ws_p, io_buf, work_ub, buf_2d, acc_s_half, mask_v, neg_sm, sumexp_is, r_factors, cid, vid, my_start)
                emit_sm(0, 1, False, Mask, ws_s, ws_p, io_buf, work_ub, buf_2d, acc_s_half, mask_v, neg_sm, sumexp_is, r_factors, cid, vid, my_start)

                # ---- steady pair-beats tt = 0 .. my_count-2:
                #      AO(tt,kv0) -> SM(tt+1,kv0) -> AO(tt,kv1) -> FIN(tt)
                #      -> SM(tt+1,kv1)
                #      (u order: 2tt, 2tt+2, 2tt+1, [fin t], 2tt+3 — the
                #       kv1-flat AO(jj)->FIN(jj)->SM(jj+2) offset generalized;
                #       FIN only at task tails, next task's kv0 softmax runs
                #       AHEAD of the current task's FIN) ----
                for tt in T.serial(my_count - 1):
                    emit_ao(tt, 0, True, ws_o, io_buf, work_ub, buf_2d, acc_o, sumexp, sumexp_is, r_factors, cid, vid)
                    emit_sm(tt + 1, 0, True, Mask, ws_s, ws_p, io_buf, work_ub, buf_2d, acc_s_half, mask_v, neg_sm, sumexp_is, r_factors, cid, vid, my_start)
                    emit_ao(tt, 1, False, ws_o, io_buf, work_ub, buf_2d, acc_o, sumexp, sumexp_is, r_factors, cid, vid)
                    emit_fin(tt, Output, acc_o, sumexp, buf_2d, acc_s_half, vid, my_start)
                    emit_sm(tt + 1, 1, False, Mask, ws_s, ws_p, io_buf, work_ub, buf_2d, acc_s_half, mask_v, neg_sm, sumexp_is, r_factors, cid, vid, my_start)

                # ---- epilogue beats ----
                xv_t = my_count - 1
                emit_ao(xv_t, 0, True, ws_o, io_buf, work_ub, buf_2d, acc_o, sumexp, sumexp_is, r_factors, cid, vid)
                emit_ao(xv_t, 1, False, ws_o, io_buf, work_ub, buf_2d, acc_o, sumexp, sumexp_is, r_factors, cid, vid)
                emit_fin(xv_t, Output, acc_o, sumexp, buf_2d, acc_s_half, vid, my_start)

                # destroy: consume outstanding init-direction flags
                T.wait_flag("V", "MTE2", SIG_IO_UB)
                T.wait_flag("V", "MTE2", SIG_IO_UB + 1)
                T.wait_flag("MTE3", "V", SIG_S_HALF)

    return main
