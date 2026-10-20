"""Flat cross-task pipelined g2stack variant — Stage 3 iter2, project B.

Standalone development file (does NOT modify examples/gqa/gqa.py; imports its
constants only).  Implements the blueprint of
examples/gqa/perf_tuning/flat_pipeline_analysis.md §5.2 with the kv_iters==1
specialization (§5.1):

  * Stream flattening: with kv_iters==1 the flattened (task, kv) index u
    degenerates to the per-core task offset x (0..my_count-1), so the "flat
    token stream" is the task sequence itself.
  * C scope steady step (loop t):  G1(t+1)  ->  G2(t)  ->  Qload(t+2)
      G1(x): K/V GM->L1 (V(x) prefetched one step ahead of its G2(x) use),
             q_l1->l0a[x%2], k->l0b[x%2], mma, FIX l0c->ws_s[cid,x%2].
      G2(s): ws_p[cid,s%2]->p_l1, v_l1[s%2]->l0b, p->l0a, mma,
             FIX l0c->ws_o[cid,s%2].
  * V scope steady beat (loop j):  AO(j-3) -> FIN(j-3) -> SM(j-1)
    (peeled form: prologue SM(0),SM(1); loop jj: AO(jj),FIN(jj),SM(jj+2);
    epilogue AO/FIN(mc-2), AO/FIN(mc-1)).
    offset-3... concretely AO(x) runs 2 beats after SM(x+1)... the SM->G2->AO
    hardware chain gets >= 1 full beat (~2.9us) of slack: zero-stall steady
    state (offset-1/2 would expose the ~0.8us G2 latency every beat).
  * ws slot parity = x%2 (2 slots = ns_flat=2): G1(x) writes ws_s[x%2] while
    SM(x-1) reads ws_s[(x-1)%2]; G2(x) writes ws_o[x%2] two beats before
    AO(x) reads it; slot reuse is gated by the counting semaphores below.
  * State buffers acc_o/sumexp/sumexp_is/r_factors are task-parity double-
    buffered ([4,half,D] / [2,half,1] / [4,half,1] / [4,half,1]); neg_sm
    stays single ([4,half,1], idx h*2+cur, prv slot cap-init'd once) because
    the prv slots are read-only and cur slots are fully overwritten per SM.
  * State init: V-init fills neg_sm=cap once (prv slots are read-only);
    acc_o/sumexp parity slots are zero-filled at the START of each AO(x)
    (per task) — an earlier fill-free variant relied on r_factors underflow
    (finite*0==0) and was bit-identical for finite inputs, but NaN/Inf
    special-value inputs leaked state across same-parity tasks
    (NaN*0==NaN; boundary_sparse_special nan_pos_mismatch=512).  The
    restored per-task fill is V-order-safe (previous same-parity reader
    FIN(x-2) ran two beats earlier) and reproduces the shipped g2stack
    arithmetic sequence exactly (fill -> *r -> +ws_o) => bit-identical
    outputs on finite inputs, identical NaN isolation on special inputs.
  * Single counting cross-semaphores (ids 0-5, same allocation as g2stack):
    the merged pacing stall (G1(x) wait WS1_V2C == "SM(x-1) consumed") is
    beat-aligned with G2(x-1)'s WS2_V2C wait, so parity-split semaphores are
    unnecessary.
  * Q prefetch: single q_l1 slot (local event ids 6/7 are hardware-reserved —
    CANN SetFlag/WaitFlag(ISASI) constraint; the MTE1<->MTE2 id space 0-5 is
    fully allocated).  Qload(x+2) is issued after G2(x) inside loop step x
    and lands ~2 beats before G1(x+2)'s MTE1 read — fully hidden.
  * Loop peeling instead of runtime guards: every flag-bearing block runs
    unconditionally (my_count>=2 guaranteed by the factory gate
    block_num >= 2*NUM_CORES); the only runtime `if`s are the proven
    buffer-select (k_l1_0/1, v_l1_0/1, l0b[0]/[1]) and runtime-index
    slice copies — both verbatim g2stack patterns.
  * Blocks are emitted via @T.macro (framework mechanism, precedent:
    examples/elementwise/elementwise_add_pipeline.py uses T.macro with
    T.set_flag/T.wait_flag bodies).  All prim_func-local entities (tensors,
    buffers, cid/vid/my_start) are passed as EXPLICIT macro args (call-site
    value substitution); factory constants (SIG_*/SEM_*/sizes/_ids) resolve
    through the macro definition-site closure (hygienic default).

Cross/local flag count audit (mc = my_count, per core; init pre-sets mirror
g2stack; "residual +1" matches the shipped kernel's drain pattern):

  cross WS1_V2C: V sets 1+mc (init + per SM);  C waits mc (per G1)   -> +1 residual (as shipped)
  cross WS1_C2V: C sets mc (per G1);           V waits mc (per SM)   -> balanced
  cross WS2_V2C: V sets mc (per SM);           C waits (mc-1)+1 (loop+epilogue G2) -> balanced
  cross WS2_C2V: C sets 1+mc (init + per G2);  V waits mc (per SM)   -> +1 residual (as shipped)
  cross WS3_C2V: C sets mc (per G2);           V waits mc (per AO)   -> balanced
  cross WS3_V2C: V sets 1+mc (init + per AO);  C waits mc (per G2)   -> +1 residual (as shipped)
  local Q: (MTE2->MTE1) sets 1+(mc-1)+... prologue 1 + in-loop qloads mc-1;
           waits mc (one per G1) -> balanced.  (MTE1->MTE2) init 1 + release
           mc (one per G1); waits: prologue 1 + in-loop mc-1 + drain 1 = mc+1
           -> balanced.
  local K/V/P/L0AB/L0C/IO/MASK/S_HALF: set/wait pairs per use, init pre-sets
  consumed by first uses, end-of-scope drains consume residuals — identical
  pairing structure to g2stack (verbatim block bodies).

Deadlock argument (steady state, beat j spans [T_j, T_j+P), P = V busy ~2.9us):
  SM(x) ends ~T_{x+2} (SM(x) runs at the tail of V beat x+1).  C step x
  (issuing G1(x+1), G2(x), Qload(x+2)) unblocks at SM(x) end == T_{x+2}:
  ws_s(x+1) ready ~T_{x+2}+0.7us — SM(x+1) starts ~T_{x+2}+1.3us (tail of
  beat x+2) => ready.  ws_o(x) ready ~T_{x+2}+1.0..1.6us — AO(x) runs at
  beat x+3 head = T_{x+2}+2P => >1us slack.  Q(x+2) lands ~T_{x+2}+0.5us —
  G1(x+2)'s MTE1 q read happens at ~T_{x+3}+0.1us => ready.  Prologue beats
  have no backward dependencies (V init fills are static; SM(0)/SM(1) wait
  only on G1(0)/G1(1); C prologue waits only init pre-sets).  Every wait
  resolves from a strictly-earlier beat's set => no cycle.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tilelang
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout

from gqa import P28_EXPERT_PASS_CONFIGS, P28_NUM_CORES, _BIG_NEG  # noqa: E402


@tilelang.jit(out_idx=[4], workspace_idx=[5, 6, 7], pass_configs=P28_EXPERT_PASS_CONFIGS)
def _gqa_kernel_prefill_g2stack_flat(
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
    """Flat cross-task 4-phase pipelined g2stack (kv_iters==1 specialization)."""
    accum_dtype = "float"

    # max-after-mask neg_sm init cap (verbatim g2stack semantics)
    neg_sm_cap = -sm_scale * _BIG_NEG * 0.5

    G = N_q // N_kv
    G_batches = G // 2
    q_blocks = S_padded // block_M
    kv_iters = S_kv_padded // block_N
    block_num = B * N_kv * G_batches * q_blocks

    # ---- factory gate (host-side, static) ----
    assert D == 128 and block_N == 128 and block_M == 64, "flat variant: D==block_N==128, bm==64"
    assert dtype in ("float16", "bfloat16")
    assert kv_iters == 1, "flat variant: kv_iters==1 only (S_kv_padded == block_N)"
    assert G % 2 == 0, "flat variant requires even G (stacked head pairs)"
    assert block_num >= 2 * P28_NUM_CORES, "flat variant requires >= 2 tasks per core"
    assert (not is_causal) or skv_minus_s >= 0, "causal requires S_kv >= S"
    # kv_useful == 1 for EVERY task by construction:
    #   bound = (bx+1)*block_M + skip_limit >= block_M > 0  (skip_limit >= 0
    #   causal / 1<<30 non-causal) -> ceil(bound/block_N) >= 1 -> no causal
    #   block-skip logic is needed (the in-block additive mask still applies).

    half = block_M // 2  # per-vid row strip within one head's segment
    M_stack = 2 * block_M  # stacked GEMM M dim (2 Q heads)

    q_tasks = block_num // P28_NUM_CORES
    r_tasks = block_num % P28_NUM_CORES

    # Cross-core semaphores (C <-> V) — verbatim g2stack allocation
    SEM_WS1_C2V = 0  # ws_s ready:      C(FIX) -> V
    SEM_WS1_V2C = 1  # ws_s consumed:   V(MTE2) -> C
    SEM_WS2_V2C = 2  # ws_p ready:      V(MTE3) -> C
    SEM_WS2_C2V = 3  # ws_p consumed:   C(MTE2) -> V
    SEM_WS3_C2V = 4  # ws_o ready:      C(FIX) -> V
    SEM_WS3_V2C = 5  # ws_o consumed:   V(MTE2) -> C

    # Local directed event ids (per pipe pair) — verbatim g2stack allocation
    SIG_K_L1 = 0  # MTE2 <-> MTE1 (k_l1[side] ownership, slots 0/1)
    SIG_V_L1 = 2  # MTE2 <-> MTE1 (v_l1[side] ownership, slots 0/1)
    SIG_P_L1 = 4  # MTE2 <-> MTE1 (p_l1 ownership)
    SIG_Q_L1 = 5  # MTE2 <-> MTE1 (q_l1 ownership)
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
    def emit_qload(x, Q, q_l1, my_start):
        """MTE2: Q rows of task (my_start+x) stacked into q_l1."""
        bx, gb, kg, bz, hq0 = _ids(my_start + x)
        T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
        T.copy(Q[bz, hq0, bx * block_M : (bx + 1) * block_M, :], q_l1[0:block_M, :])
        T.copy(
            Q[bz, hq0 + 1, bx * block_M : (bx + 1) * block_M, :],
            q_l1[block_M:M_stack, :],
        )
        T.set_flag("MTE2", "MTE1", SIG_Q_L1)

    @T.macro
    def emit_g1(x, K, V, ws_s, q_l1, k_l1_0, k_l1_1, v_l1_0, v_l1_1, l0a, l0b, l0c, cid, my_start):
        """GEMM1(x): K/V GM->L1 (V prefetched for G2(x)), q/k -> l0,
        mma -> l0c[x%2] -> FIX -> ws_s[cid, x%2]."""
        side = x % 2
        bx, gb, kg, bz, hq0 = _ids(my_start + x)
        T.wait_cross_flag(SEM_WS1_V2C)
        # K(x) load (consumed by this block's MTE1 below)
        T.wait_flag("MTE1", "MTE2", SIG_K_L1 + side)
        if side == 0:
            T.copy(K[bz, kg, 0:block_N, :], k_l1_0)
        else:
            T.copy(K[bz, kg, 0:block_N, :], k_l1_1)
        T.set_flag("MTE2", "MTE1", SIG_K_L1 + side)
        # V(x) prefetch for G2(x) (issued at step x+1): slot x%2 is free —
        # G2(x-2)'s MTE1 read released it (SIG_V_L1+side wait)
        T.wait_flag("MTE1", "MTE2", SIG_V_L1 + side)
        if side == 0:
            T.copy(V[bz, kg, 0:block_N, :], v_l1_0)
        else:
            T.copy(V[bz, kg, 0:block_N, :], v_l1_1)
        T.set_flag("MTE2", "MTE1", SIG_V_L1 + side)
        # MTE1: q_l1 -> l0a[side] (Q(x) landed in a previous step), then
        # release q_l1 for the Qload(x+2) prefetch
        T.wait_flag("M", "MTE1", SIG_L0AB + side)
        T.wait_flag("MTE2", "MTE1", SIG_Q_L1)
        T.copy(q_l1, l0a[side, :, :])
        T.set_flag("MTE1", "MTE2", SIG_Q_L1)
        T.wait_flag("MTE2", "MTE1", SIG_K_L1 + side)
        if side == 0:
            T.copy(k_l1_0, l0b[0, :, :], transpose=True)
        else:
            T.copy(k_l1_1, l0b[1, :, :], transpose=True)
        T.set_flag("MTE1", "MTE2", SIG_K_L1 + side)
        T.set_flag("MTE1", "M", SIG_L0AB + side)
        # M
        T.wait_flag("MTE1", "M", SIG_L0AB + side)
        T.wait_flag("FIX", "M", SIG_L0C + side)
        T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
        T.set_flag("M", "MTE1", SIG_L0AB + side)
        T.set_flag("M", "FIX", SIG_L0C + side)
        # FIX
        T.wait_flag("M", "FIX", SIG_L0C + side)
        T.copy(l0c[side, :, :], ws_s[cid, side, :, :])
        T.set_flag("FIX", "M", SIG_L0C + side)
        T.set_cross_flag("FIX", SEM_WS1_C2V)

    @T.macro
    def emit_g2(s, ws_p, ws_o, v_l1_0, v_l1_1, p_l1, l0a, l0b, l0c, cid):
        """GEMM2(s): ws_p[cid,s%2] -> p_l1 -> l0a, v_l1[s%2] -> l0b
        (prefetched at step s), mma -> l0c[s%2] -> FIX -> ws_o."""
        ps = s % 2
        T.wait_cross_flag(SEM_WS3_V2C)
        T.wait_flag("MTE1", "MTE2", SIG_P_L1)
        T.wait_cross_flag(SEM_WS2_V2C)
        T.copy(ws_p[cid, ps, :, :], p_l1)
        T.set_flag("MTE2", "MTE1", SIG_P_L1)
        T.wait_flag("MTE2", "MTE1", SIG_V_L1 + ps)
        T.wait_flag("M", "MTE1", SIG_L0AB + ps)
        if ps == 0:
            T.copy(v_l1_0, l0b[0, :, :])
        else:
            T.copy(v_l1_1, l0b[1, :, :])
        T.set_flag("MTE1", "MTE2", SIG_V_L1 + ps)
        T.wait_flag("MTE2", "MTE1", SIG_P_L1)
        T.copy(p_l1, l0a[ps, :, :])
        T.set_flag("MTE1", "MTE2", SIG_P_L1)
        T.set_flag("MTE1", "M", SIG_L0AB + ps)
        T.wait_flag("MTE1", "M", SIG_L0AB + ps)
        T.wait_flag("FIX", "M", SIG_L0C + ps)
        T.mma(l0a[ps, :, :], l0b[ps, :, :], l0c[ps, :, :], init=True)
        T.set_flag("M", "MTE1", SIG_L0AB + ps)
        T.set_flag("M", "FIX", SIG_L0C + ps)
        T.wait_flag("M", "FIX", SIG_L0C + ps)
        T.copy(l0c[ps, :, :], ws_o[cid, ps, :, :])
        T.set_flag("FIX", "M", SIG_L0C + ps)
        T.set_cross_flag("FIX", SEM_WS3_C2V)
        T.set_cross_flag("MTE2", SEM_WS2_C2V)

    # ==================== V-side emission macros ====================

    @T.macro
    def emit_sm(x, Mask, ws_s, ws_p, io_buf, work_ub, buf_2d, acc_s_half, mask_v, neg_sm, sumexp_is, r_factors, cid, vid, my_start):
        """Softmax(x): ws_s[cid,x%2] -> online-softmax -> ws_p[cid,x%2].

        State written: neg_sm[h*2] (cur; prv=h*2+1 read-only cap),
        sumexp_is[par*2+h], r_factors[par*2+h].  Verbatim g2stack phase-1
        body with the kv index degenerates to kv_iters==1 (cur=0, prv=1).
        """
        par = x % 2
        bx, gb, kg, bz, hq0 = _ids(my_start + x)
        T.wait_cross_flag(SEM_WS2_C2V)
        T.wait_cross_flag(SEM_WS1_C2V)
        for h in T.serial(2):
            row0 = h * block_M + vid * half
            io_side = h

            T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)
            T.copy(ws_s[cid, par, row0 : row0 + half, :], io_buf[io_side, :, :])
            T.set_flag("MTE2", "V", SIG_IO_UB + io_side)

            if use_mask:
                T.set_flag("V", "MTE2", SIG_MASK)
                T.wait_flag("V", "MTE2", SIG_MASK)
                T.copy(
                    Mask[
                        bz,
                        bx * block_M + vid * half : bx * block_M + vid * half + half,
                        0:block_N,
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
            T.reduce_max(work_ub, neg_sm[h * 2, :, :], dim=-1)
            T.tile.mul(neg_sm[h * 2, :, :], neg_sm[h * 2, :, :], -sm_scale)
            T.tile.min(neg_sm[h * 2, :, :], neg_sm[h * 2, :, :], neg_sm[h * 2 + 1, :, :])

            T.tile.broadcast(buf_2d, neg_sm[h * 2, :, :])
            T.tile.axpy(buf_2d, work_ub, sm_scale)
            T.tile.exp(work_ub, buf_2d)

            T.reduce_sum(work_ub, sumexp_is[par * 2 + h, :, :], dim=-1)
            T.tile.sub(r_factors[par * 2 + h, :, :], neg_sm[h * 2, :, :], neg_sm[h * 2 + 1, :, :])

            T.wait_flag("MTE3", "V", SIG_S_HALF)
            T.copy(work_ub, acc_s_half)
            T.set_flag("V", "MTE3", SIG_S_HALF)

            T.wait_flag("V", "MTE3", SIG_S_HALF)
            T.copy(acc_s_half, ws_p[cid, par, row0 : row0 + half, :])
            T.set_flag("MTE3", "V", SIG_S_HALF)
        T.set_cross_flag("MTE3", SEM_WS2_V2C)
        T.set_cross_flag("MTE2", SEM_WS1_V2C)

    @T.macro
    def emit_ao(x, ws_o, io_buf, work_ub, buf_2d, acc_o, sumexp, sumexp_is, r_factors, cid, vid):
        """O-accumulation(x): rescale by r_factors, add the ws_o chunk.

        Verbatim g2stack phase-2 body with i -> par (task-parity state).
        """
        par = x % 2
        T.wait_cross_flag(SEM_WS3_C2V)
        for h in T.serial(2):
            row0 = h * block_M + vid * half
            # [S3-iter2-B fix] Per-task zero-init of the parity state slots.
            # The fill-free self-reset (r_factors underflow * previous value)
            # is bit-identical ONLY for finite prev values; special-value
            # inputs (boundary_sparse_special) leave NaN/Inf in the slot and
            # NaN*0 == NaN would leak state across same-parity tasks.  Fill
            # position is V-order-safe: the previous same-parity reader
            # FIN(x-2) ran two beats earlier.  Arithmetic sequence then
            # matches the shipped g2stack exactly (fill -> *r -> +ws_o).
            T.tile.fill(acc_o[par * 2 + h, :, :], 0.0)
            T.tile.fill(sumexp[par * 2 + h, :, :], 0.0)
            T.tile.exp(r_factors[par * 2 + h, :, :], r_factors[par * 2 + h, :, :])
            T.tile.mul(sumexp[par * 2 + h, :, :], sumexp[par * 2 + h, :, :], r_factors[par * 2 + h, :, :])
            T.tile.add(sumexp[par * 2 + h, :, :], sumexp[par * 2 + h, :, :], sumexp_is[par * 2 + h, :, :])
            T.tile.broadcast(buf_2d, r_factors[par * 2 + h, :, :])
            T.tile.mul(acc_o[par * 2 + h, :, :], acc_o[par * 2 + h, :, :], buf_2d)

            io_side = h
            T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)
            T.copy(ws_o[cid, par, row0 : row0 + half, :], io_buf[io_side, :, :])
            T.set_flag("MTE2", "V", SIG_IO_UB + io_side)

            T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)
            T.copy(io_buf[io_side, :, :], work_ub)
            T.set_flag("V", "MTE2", SIG_IO_UB + io_side)

            T.tile.add(acc_o[par * 2 + h, :, :], acc_o[par * 2 + h, :, :], work_ub)
        T.set_cross_flag("MTE2", SEM_WS3_V2C)

    @T.macro
    def emit_fin(x, Output, acc_o, sumexp, buf_2d, acc_s_half, vid, my_start):
        """Final normalize + contiguous per-(h, vid) output box write for
        task (my_start+x).  Verbatim g2stack finalize."""
        par = x % 2
        bx, gb, kg, bz, hq0 = _ids(my_start + x)
        for h in T.serial(2):
            T.tile.add(sumexp[par * 2 + h, :, :], sumexp[par * 2 + h, :, :], 1e-30)
            T.tile.broadcast(buf_2d, sumexp[par * 2 + h, :, :])
            T.tile.div(acc_o[par * 2 + h, :, :], acc_o[par * 2 + h, :, :], buf_2d)
            T.wait_flag("MTE3", "V", SIG_S_HALF)
            T.copy(acc_o[par * 2 + h, :, :], acc_s_half)
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
            # ===== C-side buffers (verbatim g2stack) =====
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

            # ===== V-side buffers (per vid) — parity-doubled task state =====
            io_buf = T.alloc_ub([2, half, block_N], accum_dtype)
            work_ub = T.alloc_ub([half, block_N], accum_dtype)
            buf_2d = T.alloc_ub([half, block_N], accum_dtype)
            acc_s_half = T.alloc_ub([half, block_N], dtype)
            acc_o = T.alloc_ub([4, half, D], accum_dtype)  # idx par*2 + h
            mask_v = T.alloc_ub([half, block_N], accum_dtype)
            neg_sm = T.alloc_ub([4, half, 1], accum_dtype)  # idx h*2 + cur (single set)
            sumexp = T.alloc_ub([4, half, 1], accum_dtype)  # idx par*2 + h
            sumexp_is = T.alloc_ub([4, half, 1], accum_dtype)  # idx par*2 + h
            r_factors = T.alloc_ub([4, half, 1], accum_dtype)  # idx par*2 + h

            my_start, my_count = task_range(cid)

            # Epilogue task-offset scalars.  my_count embeds a
            # T.if_then_else (task_range remainder term) which lowers to the
            # multi-statement `condval` pattern; passing `my_count - k`
            # directly as a macro arg inlines that pattern INTO tile-op index
            # / SetFlag-id expression positions where C++ cannot host
            # statements (bisheng: "unexpected type name 'int32_t'").  Binding
            # through alloc_var scalars materializes the condval exactly once
            # at a statement position (the shipped g2stack's per-iteration
            # task_id binding pattern) and every downstream use becomes a
            # plain variable load.  Per-scope pairs: written AND read inside
            # one scope only (CV-split safe, same as one-sided buffers).
            xc1 = T.alloc_var("int32")
            xc2 = T.alloc_var("int32")
            xv1 = T.alloc_var("int32")
            xv2 = T.alloc_var("int32")

            # ============================ C scope ============================
            with T.Scope("C"):
                # init: pretend consumers already released everything
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

                # ---- prologue: Q(0), G1(0), Q(1)   (my_count >= 2 by gate) ----
                emit_qload(0, Q, q_l1, my_start)
                emit_g1(0, K, V, ws_s, q_l1, k_l1_0, k_l1_1, v_l1_0, v_l1_1, l0a, l0b, l0c, cid, my_start)
                emit_qload(1, Q, q_l1, my_start)

                # ---- steady steps t = 0 .. my_count-3:
                #      G1(t+1) -> G2(t) -> Qload(t+2) ----
                for t in T.serial(my_count - 2):
                    emit_g1(t + 1, K, V, ws_s, q_l1, k_l1_0, k_l1_1, v_l1_0, v_l1_1, l0a, l0b, l0c, cid, my_start)
                    emit_g2(t, ws_p, ws_o, v_l1_0, v_l1_1, p_l1, l0a, l0b, l0c, cid)
                    emit_qload(t + 2, Q, q_l1, my_start)

                # ---- epilogue: G2(mc-2), G1(mc-1), G2(mc-1) ----
                xc1 = my_count - 2
                xc2 = my_count - 1
                emit_g2(xc1, ws_p, ws_o, v_l1_0, v_l1_1, p_l1, l0a, l0b, l0c, cid)
                emit_g1(xc2, K, V, ws_s, q_l1, k_l1_0, k_l1_1, v_l1_0, v_l1_1, l0a, l0b, l0c, cid, my_start)
                emit_g2(xc2, ws_p, ws_o, v_l1_0, v_l1_1, p_l1, l0a, l0b, l0c, cid)

                # destroy: consume outstanding init-direction flags
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
                # one-shot static state init (replaces the per-task fills —
                # see the module docstring; r_factors underflow-to-0 reset)
                T.set_cross_flag("MTE2", SEM_WS1_V2C)
                T.set_cross_flag("MTE2", SEM_WS3_V2C)
                T.set_flag("V", "MTE2", SIG_IO_UB)
                T.set_flag("V", "MTE2", SIG_IO_UB + 1)
                T.set_flag("MTE3", "V", SIG_S_HALF)
                T.tile.fill(acc_o, 0.0)
                T.tile.fill(sumexp, 0.0)
                T.tile.fill(neg_sm, neg_sm_cap)

                # ---- prologue beats: SM(0), SM(1) ----
                emit_sm(0, Mask, ws_s, ws_p, io_buf, work_ub, buf_2d, acc_s_half, mask_v, neg_sm, sumexp_is, r_factors, cid, vid, my_start)
                emit_sm(1, Mask, ws_s, ws_p, io_buf, work_ub, buf_2d, acc_s_half, mask_v, neg_sm, sumexp_is, r_factors, cid, vid, my_start)

                # ---- steady beats jj = 0 .. my_count-3:
                #      AO(jj) -> FIN(jj) -> SM(jj+2) ----
                for jj in T.serial(my_count - 2):
                    emit_ao(jj, ws_o, io_buf, work_ub, buf_2d, acc_o, sumexp, sumexp_is, r_factors, cid, vid)
                    emit_fin(jj, Output, acc_o, sumexp, buf_2d, acc_s_half, vid, my_start)
                    emit_sm(jj + 2, Mask, ws_s, ws_p, io_buf, work_ub, buf_2d, acc_s_half, mask_v, neg_sm, sumexp_is, r_factors, cid, vid, my_start)

                # ---- epilogue beats ----
                xv1 = my_count - 2
                xv2 = my_count - 1
                emit_ao(xv1, ws_o, io_buf, work_ub, buf_2d, acc_o, sumexp, sumexp_is, r_factors, cid, vid)
                emit_fin(xv1, Output, acc_o, sumexp, buf_2d, acc_s_half, vid, my_start)
                emit_ao(xv2, ws_o, io_buf, work_ub, buf_2d, acc_o, sumexp, sumexp_is, r_factors, cid, vid)
                emit_fin(xv2, Output, acc_o, sumexp, buf_2d, acc_s_half, vid, my_start)

                # destroy: consume outstanding init-direction flags
                T.wait_flag("V", "MTE2", SIG_IO_UB)
                T.wait_flag("V", "MTE2", SIG_IO_UB + 1)
                T.wait_flag("MTE3", "V", SIG_S_HALF)

    return main


def build_flat_kernel(B, S, S_kv, sm_scale, is_causal, N_q, N_kv, D, dtype_str):
    """Host-side factory mirroring gqa_expert's prefill prep for the gate.

    Returns (kernel_fn, S_padded, S_kv_padded, use_mask).  Raises if the
    shape is outside the flat gate (caller must fall back to the shipped
    dispatch).
    """
    block_M, block_N = 64, 128
    S_padded = ((S + block_M - 1) // block_M) * block_M
    S_kv_padded = ((S_kv + block_N - 1) // block_N) * block_N
    kv_iters = S_kv_padded // block_N
    G = N_q // N_kv
    q_blocks = S_padded // block_M
    block_num = B * N_kv * (G // 2) * q_blocks
    use_mask = bool(is_causal or (S_kv_padded != S_kv))
    eligible = (
        S > 4
        and D == 128
        and dtype_str in ("float16", "bfloat16")
        and kv_iters == 1
        and ((not is_causal) or (S_kv >= S))
        and G % 2 == 0
        and block_num >= 2 * P28_NUM_CORES
    )
    if not eligible:
        raise ValueError(
            f"shape outside flat gate: B={B} S={S} S_kv={S_kv} Nq={N_q} Nkv={N_kv} D={D} "
            f"causal={is_causal} (stacked={block_num}, kv_iters={kv_iters})"
        )
    kernel_fn = _gqa_kernel_prefill_g2stack_flat(
        B,
        S_padded,
        S_kv_padded,
        sm_scale,
        is_causal,
        N_q=N_q,
        N_kv=N_kv,
        D=D,
        dtype=dtype_str,
        block_M=block_M,
        skv_minus_s=S_kv - S,
        use_mask=use_mask,
        block_N=block_N,
    )
    return kernel_fn, S_padded, S_kv_padded, use_mask
