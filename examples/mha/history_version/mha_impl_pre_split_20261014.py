"""MHA (Multi-Head Attention) forward, BSHD layout, Expert mode.

    y = softmax(Q @ K^T * s) @ V,   s = scaleValue if scaleValue > 0 else 1/sqrt(D)

is_causal masks scores[..., i, j] to -inf for j > i + (S_kv - S): the mask is
bottom-right aligned. All-masked rows are reachable only out-of-domain and yield
0 through the sumexp + 1e-30 guard. Q/K/V are fp16 or bf16; the score mainline
and the O accumulator are fp32.

Architecture. The host selects one of three mutually exclusive and exhaustive
C-scope forms from (S, D, S_kv) alone, so a generated kernel holds exactly one.
Form A (S > 16, D <= 128) is a T.mma full pipeline: L0A/L0B/L0C double slots with
manual side = i % 2 ping-pong, ZN/NZ annotated L1 operands and fine-grained
per-pipe-pair event flags. When block_N == D the GEMM2 L0 pool aliases the GEMM1
pool, and in the single-KV-block sub-case the FIX pipe writes Output straight
from L0C with a runtime row extent, because O = P_hat @ V is already final there.
Form B-mma (S > 16, D == 256) is a manual K/N-split T.mma pipeline: GEMM1 splits
K into two KC = 128 chunks chained by mma(init=True -> False), GEMM2 splits N
into two NC = 128 chunks that accumulate into a narrow L0C pool and are each
written to their own workspace_3 column segment. The split exists because
gemm_v0 drains the M pipe at every step at this width, and because an mma
targeting an L0C column sub-region is rejected by the cube unit. Form B gemm_v0
(S <= 16, all D) is the decode fallback: T.gemm_v0 manages L0A/L0B internally
through the kL0Size tiers 32/64/128, and its self-balanced prologue/epilogue pipe
flags cover every segment boundary, so no coarse barrier is needed.

The V scope is shared by all three forms through emit_v, which is a @T.macro and
must stay one. TVMScript resolves T.* frames at parse time, so a plain Python
emitter called from inside a prim_func has its T.* emissions silently dropped:
the kernel still compiles, but the generated body is missing and the aicore hangs
on an orphaned wait.

Pipeline shape. C and V run NUM_STAGES = 8 deep batches and exchange data through
three explicit GM workspaces indexed [core, stage]: workspace_1 = fp32 scores,
workspace_2 = P, workspace_3 = fp32 PV. Inline comments abbreviate these as
w1/w2/w3, and a "colseg" write is a column-segment store. T.copy(L0C -> UB) and
T.copy(UB -> L1) have no device implementation, so that GM round trip is the only
physical C/V route. T.Pipelined is unavailable in this mode; the manual
side = i % 2 slot rotation expresses the pipeline instead. Each AIV owns
half_M = block_M // 2 rows (the vid row split) and writes back
min(half_M, max(0, valid_m - vid * half_M)) rows under a runtime guard.

Multi-KV-block prefill walks a flat (task, kv) slot stream so a slot batch can
cross task boundaries without draining. Under causal masking eff_kv_iters varies
per task at runtime, so the flat index -> (task, kv) mapping is a scalar walk
state machine that C and V compute identically, never via a workspace.

Precision invariants. These must stay bit-identical to the golden reference; do
not reorder or refold any of them.
1. Score double-rounding grid: fp32 QK result -> round to dtype (r16(qk)) ->
   multiply by s in fp32 -> round to dtype again (r16(scores)).
2. Single-KV-block specialization: P is divided by (sumexp + 1e-30) BEFORE the
   dtype rounding, matching the golden one-shot softmax; phase 2 then skips the
   rescale and the final division.
3. sumexp + 1e-30 on every final normalization, so an all-masked row gives 0
   instead of 0/0.
4. Mask chain semantics: a pre-exp -inf select built from a 2D compare against
   hoisted, task-invariant row thresholds, with fully visible KV blocks skipped
   at runtime by a block-uniform condition.

Sync model. Every event id is statically budgeted: init pre-sets the tokens a
first consumer would otherwise wait for, destroy consumes exactly the outstanding
ones, and each iteration is net zero. Cross-core semaphores are strictly 1:1
Set/Wait. A conditional DMA must carry ALL its flags inside the same condition --
a copy guarded while its flag is not (or the reverse) corrupts the run.

Constraints. D in {64, 128, 256}; fp16/bf16; contiguous BSHD; 1 <= B <= 128,
1 <= S <= 2048, 1 <= S_kv <= 4096, 1 <= N <= 64; is_causal requires S <= S_kv.
Every trip count is a compile-time constant or derived from one, never from tensor
data. The 32 KB L0B ping-pong budget and the 511.75 KB L1 budget are what fix the
block_N and gemm1_kL0 tiers. MTE2_MTE1 event ids 6 and 7 are unusable in
static-tensor code. The kv_db decode path (S <= 16, D == 256, more than one KV
block) runs on stock tilelang-ascend: its T.gemm_v0 calls keep the default
prologue, whose whole-MTE2-pipe drain makes every operand enqueued before the
call resident, so that path needs no framework-side patch.

DESIGN.md in this directory derives the tiling tiers, the event-id budget and
the precision gates that test_mha.py checks.

Run this file directly for a self-checking example against a torch reference:
    python examples/mha/mha.py                    # fp16, B=2 S=S_kv=128 N=8 D=64
    python examples/mha/mha.py --causal --D 128   # --help lists every option
"""

import argparse

import tilelang
import torch
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,  # manual T.Scope("C"/"V")
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,  # manual SEM 0-5
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,  # manual SIG flag matrix
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,  # explicit alloc_* declarations
}

NUM_STAGES = 8  # C/V pipeline depth in slots; must be even (double-slot rotation)
CROSS_INTERVAL = 1  # slots per cross-core semaphore Set/Wait pair


def _cube_core_num() -> int:
    """Physical cube-core count (host metadata query; touches no tensor data)."""
    try:
        import torch_npu

        return int(torch_npu.npu.get_device_properties(0).cube_core_num)
    except Exception:  # noqa: BLE001 - conservative 910B-class fallback
        return 20


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)
def mha_kernel(
    B,
    S,
    S_kv,
    N,
    D,
    dtype="float16",
    scaleValue=-1.0,
    is_causal=False,
):
    accum_dtype = "float"
    assert NUM_STAGES % 2 == 0, "NUM_STAGES must be even: the io_buf/L0 slots rotate in pairs"
    # ---- Host-side config derivation: pure Python, touches no tensor data ----
    sm_scale = scaleValue if scaleValue > 0 else (1.0 / D) ** 0.5
    # block_M / block_N / gemm1_kL0 three-tier system. Every tier is the largest
    # value that still fits its on-chip budget (the 32 KB L0B ping-pong and the
    # L1 capacity); the predicates below encode those budget solutions.
    block_M = 16 if S <= 16 else (32 if D == 256 else 64)
    if S <= 16:
        if D == 256 and S_kv > 256:
            # kv_db dual-side double-buffer domain. bN=128 keeps L1 inside
            # q8 + k[2]x64 + v[2]x64 + p4 = 268 KB of the 511.75 KB budget, and
            # that headroom is what makes two k_l1 and two v_l1 slots possible.
            # The second V slot is in turn what allows V(idx+1) to be issued
            # BEFORE the gemm call instead of after it returns: with a single
            # v_l1 slot the epilogue release order forbids a pre-call issue at
            # bN=256. Only the multi-block domain (S_kv > 256) takes this tier;
            # the single-block domain keeps the tiers below.
            block_N = 128
        else:
            # bN=512 (D<=128, S_kv>256): GEMM1 transpose_B B-tile 512*kL0(32)*2B =
            # 32768 == the 4-step ping-pong L0B budget; D=256 excluded by L1 (536KB).
            block_N = 512 if (D <= 128 and S_kv > 256) else (256 if S_kv > 128 else 128)
    else:
        block_N = 128
    gemm1_kL0 = (
        32
        if (S <= 16 and D <= 128 and S_kv > 256)  # bN=512: 512*32*2 = 32768 <= 32KB
        # bN=256, D=256 single-block tier (128 < S_kv <= 256): 256*64*2 =
        # 32768 <= 32KB. The kv_db multi-block domain (S_kv > 256) uses bN=128
        # and falls through to the 128 tier: 128*128*2 = 32768 <= 32KB. Two
        # ping-pong steps are kept on purpose -- kL0=256 would collapse the
        # GEMM1 K loop to a single step, serialising MTE1 against M with no
        # overlap, while leaving the total k-step count per core unchanged.
        else (64 if (S <= 16 and D == 256 and S_kv > 128 and S_kv <= 256) else 128)
    )
    # ---- Expert-mode dispatch predicates ----
    form_a = (S > 16) and (D <= 128)  # Form A: T.mma full pipeline
    # ---- Three-way dispatch: mutually exclusive and exhaustive over (S, D) ----
    form_b_mma = (S > 16) and (D == 256)  # Form B-mma: manual K/N-split mma pipeline
    # else (S <= 16): Form B gemm_v0 decode fallback, all D
    KC = 128  # Form B-mma k-chunk width (compile-time constant; D=256 -> 2 chunks)
    NC = 128  # Form B-mma n-chunk width (compile-time constant; D=256 -> 2 chunks)
    if form_b_mma:
        assert D % KC == 0 and D % NC == 0, "K/N-split requires D divisible by the KC/NC chunk width"
    half_M = block_M // 2  # rows owned by one AIV (vid row split): 8/16/32
    causal_offset = S_kv - S  # bottom-right alignment (>= 0 when causal, asserted)
    need_bound_mask = S_kv % block_N != 0  # tail-column clamp (compile-time)
    need_mask = is_causal or need_bound_mask
    # Unified threshold offset: causal -> bottom-right offset; non-causal -> big
    # constant so T.min(row + offset_eff, S_kv-1) collapses to S_kv-1.
    offset_eff = causal_offset if is_causal else 2**30
    s_blocks = (S + block_M - 1) // block_M
    kv_iters = (S_kv + block_N - 1) // block_N  # compile-time constant
    single_kv_block = kv_iters == 1  # i.e. S_kv <= block_N
    # ---- Flat (task, kv) slot-stream domain: the generic multi-KV-block prefill
    # of both mma forms. Iterating slots over a flat (task, kv) index instead of
    # nested [task][kv-block] loops lets one slot batch cross task boundaries
    # without draining the pipeline. The single-KV-block branch (which batches
    # tasks instead) and the S <= 16 decode fallback use their own loops. This
    # is a parse-time closure flag: emit_v reads it to pick its V-side branch
    # and both C-scope generic branches read it too, so all three must agree.
    flat_v = (S > 16) and not single_kv_block
    # ---- Decode D=256 K/V dual-side L1 double-buffer domain ----
    # gemm_v0 fallback only (S <= 16), D == 256, more than one KV block. L1
    # footprint q8 + k[2]x64 + v[2]x64 + p4 = 268 KB of the 511.75 KB budget.
    # Sync = the stock gemm_v0 default prologue alone: every call self-issues
    # SetFlag/WaitFlag<MTE2_MTE1>(0), draining the whole MTE2 pipe, so all
    # operands enqueued before the call (Q, K(idx), the K(idx+1)/V(idx+1)
    # prefetches, V(idx), P) are L1-resident before its MTE1 reads, and slot
    # reuse is ordered by the epilogue's SetFlag<MTE1_MTE2>. This path must
    # therefore carry ZERO user MTE2_MTE1 flags: a dangling user token on id 0
    # could be consumed stale by a later default-prologue wait. C-side intra-
    # core scheduling only: the V scope, the semaphore matrix, the single-
    # block path and the D <= 128 domains are untouched.
    kv_db = (S <= 16) and (D == 256) and not single_kv_block
    block_num = B * N * s_blocks
    core_num = min(block_num, _cube_core_num())  # never launch more cores than tasks
    q_tasks = block_num // core_num  # base tasks per core
    r_tasks = block_num % core_num  # first r_tasks cores take one extra

    # ---- Cross-core semaphore ids (AIC <-> 2 x AIV group, mode=2) ----
    SEM_WS1_C2V = 0  # w1 scores ready: Cube -> Vector
    SEM_WS1_V2C = 1  # w1 consumed: Vector -> Cube
    SEM_WS2_V2C = 2  # w2 softmax P ready: Vector -> Cube
    SEM_WS2_C2V = 3  # w2 consumed: Cube -> Vector
    SEM_WS3_C2V = 4  # w3 PV ready: Cube -> Vector
    SEM_WS3_V2C = 5  # w3 consumed: Vector -> Cube

    # ---- C-side intra-core event ids, one space per directed pipe pair (Form A) ----
    # MTE2 <-> MTE1
    SIG_K_L1 = 0
    SIG_P_L1 = 1
    SIG_V_L1 = 2
    SIG_Q_L1 = 3
    # MTE1 <-> M
    SIG_L0AB = 0  # double slots 0/1 (Form A)
    # MTE1 <-> M, the Form B-mma fine-grained tokens. Ids 0-5 stay within the
    # 8-id hardware limit. This prim_func contains no T.gemm_v0 call, so these
    # ids cannot collide with gemm_v0's internal L0AB event id 0.
    SIG_L0KC = 0  # l0b_pool chunk 0/1; GEMM1 kc and GEMM2 nc share this stream
    SIG_L0AO = 2  # l0a_o side 0/1 (GEMM2 A ping-pong)
    SIG_L0AS = 4  # l0a_s chunk 0/1 (GEMM1 A, task/slot-level preload)
    # M <-> FIX
    SIG_L0C = 0  # double slots 0/1

    # ---- V-side intra-core event ids (per AIV) ----
    # MTE2 <-> V
    SIG_IO_UB = 0  # io_buf (w1) double slots 0/1
    SIG_IO3 = 2  # io_buf3 (w3) double slots 2/3 (only when D != block_N)
    # MTE3 <-> V
    SIG_S_HALF = 0  # acc_s_half / out_half staging ownership

    # workspace_3 staging and rescale scratch: a shape-forced buffer choice.
    # D == block_N -> alias io_buf / buf_2d (identical shapes, reused in disjoint
    # sequential V-pipe windows, token-balanced on shared flag ids);
    # D != block_N -> dedicated buffers, because the shapes then differ.
    io3_is_alias = block_N == D
    sig_io3 = SIG_IO_UB if io3_is_alias else SIG_IO3

    # ==================== Shared V-scope emitter (@T.macro) ====================
    @T.macro
    def emit_v(
        io_buf,
        io3,
        work_ub,
        buf_2d,
        rt_2d,
        r_2d,
        mask_2d,
        col_pos,
        row_thresh,
        rr_ub,
        acc_s_half,
        acc_o,
        out_half,
        m_i,
        m_i_prev,
        sumexp,
        r_factors,
        sumexp_is,
        Output,
        workspace_1,
        workspace_2,
        workspace_3,
        cid,
        vid,
        my_start,
        my_count,
        sig_io3_base,
    ):
        # init: pre-set the tokens whose consumers would otherwise wait first
        T.set_cross_flag("MTE2", SEM_WS1_V2C)
        T.set_cross_flag("MTE2", SEM_WS3_V2C)
        T.set_flag("V", "MTE2", SIG_IO_UB)
        T.set_flag("V", "MTE2", SIG_IO_UB + 1)
        if not io3_is_alias:
            T.set_flag("V", "MTE2", sig_io3_base)
            T.set_flag("V", "MTE2", sig_io3_base + 1)
        T.set_flag("MTE3", "V", SIG_S_HALF)

        if single_kv_block:
            # ---- Task-batched single-KV-block V path (both mma forms) ----
            # kv_iters == 1, so one protocol round batches NUM_STAGES *tasks*:
            # slot i <-> task my_start + kb*NUM_STAGES + i. The per-slot op
            # sequence is identical to the generic per-task flow. num_tb and
            # batch_tasks MUST be derived from exactly the same formulas as the
            # C-side mirror, because the two scopes advance their flag pairing in
            # lockstep and any divergence deadlocks.
            # The branch serves Form A and Form B alike: the Form A C scope has
            # its own matching `if single_kv_block:` task batch, so the V side
            # needs no form predicate here.
            num_tb = T.ceildiv(my_count, NUM_STAGES)
            for kb in T.serial(num_tb):
                _remaining_t = my_count - kb * NUM_STAGES
                batch_tasks = T.if_then_else(_remaining_t < NUM_STAGES, _remaining_t, NUM_STAGES)

                # ---- Phase 1 batch: per-task softmax -> w2 slots ----
                T.wait_cross_flag(SEM_WS2_C2V)
                for i in T.serial(batch_tasks):
                    task = my_start + kb * NUM_STAGES + i
                    bx = task % s_blocks
                    s0 = bx * block_M  # loop-body scope: TVMScript bans cross-block defs
                    io_side = i % 2
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(m_i, -(2**30))
                    if need_mask:
                        T.tile.arith_progression(row_thresh, s0 + vid * half_M + offset_eff, 1, half_M)
                        T.tile.min(row_thresh, row_thresh, T.cast(S_kv - 1, accum_dtype))

                    # w1[cid, i, vid half] -> io_buf -> work_ub (fp32 mainline)
                    T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)
                    if i % CROSS_INTERVAL == 0:
                        T.wait_cross_flag(SEM_WS1_C2V)
                    T.copy(
                        workspace_1[cid, i, vid * half_M : vid * half_M + half_M, :],
                        io_buf[io_side, :, :],
                    )
                    T.set_flag("MTE2", "V", SIG_IO_UB + io_side)
                    T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)
                    T.copy(io_buf[io_side, :, :], work_ub)
                    T.set_flag("V", "MTE2", SIG_IO_UB + io_side)

                    T.copy(m_i, m_i_prev)  # snapshot the previous max BEFORE reduce
                    # Golden score grid (precision invariant 1): this exact op order
                    T.copy(work_ub, rr_ub)  # r16(qk)
                    T.copy(rr_ub, work_ub)
                    T.tile.mul(work_ub, work_ub, sm_scale)  # * s in fp32
                    T.copy(work_ub, rr_ub)  # r16(scores) = golden grid
                    T.copy(rr_ub, work_ub)

                    if need_mask:  # noqa: SIM102 - parse-time outer / runtime inner
                        # Single block => idx == 0. The condition is uniform over
                        # the s0 block, so both vids take the same branch: flag
                        # pairing stays balanced and this is a pure-V segment.
                        if block_N - 1 > T.min(s0 + offset_eff, S_kv - 1):
                            T.tile.arith_progression(col_pos, 0, 1, block_N)
                            T.tile.broadcast(buf_2d, col_pos, axis=0)  # [bN] -> [hM,bN]
                            T.tile.broadcast(rt_2d, row_thresh, axis=1)  # [hM] -> [hM,bN]
                            T.tile.compare(mask_2d, buf_2d, rt_2d, "LE")
                            T.tile.select(
                                work_ub,
                                mask_2d,
                                work_ub,
                                -T.infinity(accum_dtype),
                                "VSEL_TENSOR_SCALAR_MODE",
                            )

                    T.reduce_max(work_ub, m_i, dim=-1)
                    T.tile.max(m_i, m_i, m_i_prev)
                    T.tile.sub(r_factors[i, :, :], m_i_prev, m_i)
                    T.tile.broadcast(buf_2d, m_i, axis=1)
                    T.tile.sub(work_ub, work_ub, buf_2d)
                    T.tile.exp(work_ub, work_ub)  # P = exp(s - m_new) in [0, 1]
                    T.reduce_sum(work_ub, sumexp_is[i, :, :], dim=-1)

                    # Pre-normalization specialization (precision invariant 2):
                    # divide by sumexp BEFORE the dtype rounding, which is what
                    # reproduces the golden one-shot softmax grid.
                    T.tile.add(sumexp, sumexp_is[i, :, :], 1e-30)
                    T.tile.broadcast(buf_2d, sumexp, axis=1)
                    T.tile.div(work_ub, work_ub, buf_2d)

                    # P -> w2 slot i (SIG_S_HALF ownership closed loop)
                    T.wait_flag("MTE3", "V", SIG_S_HALF)
                    T.copy(work_ub, acc_s_half)  # fp32 -> dtype (P native grid)
                    T.set_flag("V", "MTE3", SIG_S_HALF)
                    T.wait_flag("V", "MTE3", SIG_S_HALF)
                    T.copy(
                        acc_s_half,
                        workspace_2[cid, i, vid * half_M : vid * half_M + half_M, :],
                    )
                    T.set_flag("MTE3", "V", SIG_S_HALF)
                    if (i + 1) % CROSS_INTERVAL == 0 or i == batch_tasks - 1:
                        T.set_cross_flag("MTE3", SEM_WS2_V2C)
                T.set_cross_flag("MTE2", SEM_WS1_V2C)

                # ---- Phase 2 batch: per-task workspace_3 slot -> acc_o -> writeback.
                # Form B only. Under the pre-normalization specialization
                # O = P_hat @ V is already final, so the Form A single-block C
                # scope writes Output directly from l0c_o and neither this phase
                # nor the workspace_3 semaphore round exists there.
                # `form_a` is a parse-time Python bool captured by the closure, so
                # the branch is resolved before lowering (same mechanism as
                # `if not io3_is_alias:` in the init block above).
                # The init pre-set of SEM_WS3_V2C is deliberately kept even on the
                # Form A path: it is a dangling-but-harmless token, exactly like
                # the by-design dangling SEM_WS1_V2C +1, because cross-core
                # semaphore state is per-launch and the C side never waits on it.
                if not form_a:
                    for i in T.serial(batch_tasks):
                        task = my_start + kb * NUM_STAGES + i
                        bx = task % s_blocks
                        by = task // s_blocks % N
                        bz = task // s_blocks // N % B
                        s0 = bx * block_M
                        valid_m = T.min(block_M, S - s0)
                        rows_vid = T.min(half_M, T.max(0, valid_m - vid * half_M))
                        io_side = i % 2
                        T.wait_flag("V", "MTE2", sig_io3_base + io_side)
                        if i % CROSS_INTERVAL == 0:
                            T.wait_cross_flag(SEM_WS3_C2V)
                        T.copy(
                            workspace_3[cid, i, vid * half_M : vid * half_M + half_M, :],
                            io3[io_side, :, :],
                        )
                        T.set_flag("MTE2", "V", sig_io3_base + io_side)
                        T.wait_flag("MTE2", "V", sig_io3_base + io_side)
                        T.tile.fill(acc_o, 0.0)
                        T.tile.add(acc_o, acc_o, io3[io_side, :, :])
                        T.set_flag("V", "MTE2", sig_io3_base + io_side)
                        # writeback inside the SIG_S_HALF closed loop + rows_vid guard
                        T.wait_flag("MTE3", "V", SIG_S_HALF)
                        T.copy(acc_o, out_half)  # fp32 -> dtype
                        T.set_flag("V", "MTE3", SIG_S_HALF)
                        T.wait_flag("V", "MTE3", SIG_S_HALF)
                        if rows_vid > 0:
                            T.copy(
                                out_half[0:rows_vid, :],
                                Output[bz, s0 + vid * half_M : s0 + vid * half_M + rows_vid, by, :],
                            )
                        T.set_flag("MTE3", "V", SIG_S_HALF)
                    T.set_cross_flag("MTE2", SEM_WS3_V2C)

        elif flat_v:
            # ---- Flat (task, kv) slot stream, V side: the generic multi-block
            # prefill of both mma forms.
            # Task-state resets MUST sit in phase 2's kv == 0 slot, not phase 1's:
            # within a round phase 1 runs to completion before phase 2 starts, so
            # resetting acc_o/sumexp in phase 1 would destroy the PREVIOUS task's
            # in-flight accumulation that is still waiting for its phase 2 tail.
            # m_i is the exception -- it is phase-1 exclusive, and reduce_max
            # overwrite semantics plus the m_i_prev chain make fill -> copy
            # equivalent to the generic task-head sequence.
            # Delta versus the single-block path: under causal masking
            # eff_kv_iters is runtime-nonuniform per task, so the flat index ->
            # (task, kv) mapping is a sequential walk state machine (T.alloc_var,
            # O(1) amortised per slot plus a one-time O(my_count) total_flat
            # walk). C and V compute it IDENTICALLY from the same
            # my_start/my_count and the same causal formula; it is never
            # transmitted through a workspace. The per-slot op sequence is the
            # generic one verbatim -- only batch membership changes, so task
            # boundaries do not drain the pipeline.
            total_flat = T.alloc_var("int32", init=0)
            for t in T.serial(my_count):
                _tw = my_start + t
                _s0w = (_tw % s_blocks) * block_M
                _effw = T.floordiv(T.min(_s0w + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                total_flat += _effw
            num_fb = T.ceildiv(total_flat, NUM_STAGES)
            tl_head = T.alloc_var("int32", init=0)  # persistent batch-head walk state
            kv_head = T.alloc_var("int32", init=0)
            for fb in T.serial(num_fb):
                _rem_f = total_flat - fb * NUM_STAGES
                batch_iters = T.if_then_else(_rem_f < NUM_STAGES, _rem_f, NUM_STAGES)

                # ---- Phase 1: batched online softmax (w1 -> P -> w2), flat slots ----
                T.wait_cross_flag(SEM_WS2_C2V)
                tl_p1 = T.alloc_var("int32", init=tl_head)
                kv_p1 = T.alloc_var("int32", init=kv_head)
                for i in T.serial(batch_iters):
                    io_side = i % 2
                    task = my_start + tl_p1
                    bx = task % s_blocks
                    s0 = bx * block_M
                    eff = T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                    idx = kv_p1
                    # w1[cid, i, vid half] -> io_buf -> work_ub (fp32 mainline)
                    T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)
                    if i % CROSS_INTERVAL == 0:
                        T.wait_cross_flag(SEM_WS1_C2V)
                    T.copy(
                        workspace_1[cid, i, vid * half_M : vid * half_M + half_M, :],
                        io_buf[io_side, :, :],
                    )
                    T.set_flag("MTE2", "V", SIG_IO_UB + io_side)
                    T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)
                    T.copy(io_buf[io_side, :, :], work_ub)
                    T.set_flag("V", "MTE2", SIG_IO_UB + io_side)

                    if kv_p1 == 0:
                        # Task-start resets: phase-1 EXCLUSIVE state only (m_i, see
                        # the note above; acc_o/sumexp are reset in phase 2's
                        # kv == 0 slot) plus the row_thresh recompute. The
                        # thresholds are task-invariant and this is a pure-V
                        # segment carrying no flags.
                        T.tile.fill(m_i, -(2**30))
                        if need_mask:
                            T.tile.arith_progression(row_thresh, s0 + vid * half_M + offset_eff, 1, half_M)
                            T.tile.min(row_thresh, row_thresh, T.cast(S_kv - 1, accum_dtype))

                    T.copy(m_i, m_i_prev)  # snapshot the previous max BEFORE reduce
                    # Golden score grid (precision invariant 1): this exact op order
                    T.copy(work_ub, rr_ub)  # r16(qk)
                    T.copy(rr_ub, work_ub)
                    T.tile.mul(work_ub, work_ub, sm_scale)  # * s in fp32
                    T.copy(work_ub, rr_ub)  # r16(scores) = golden grid
                    T.copy(rr_ub, work_ub)

                    if need_mask:  # noqa: SIM102 - parse-time outer / runtime inner
                        # Runtime skip for fully visible blocks (precision
                        # invariant 4). The condition is uniform over the s0 block,
                        # so both vids take the same branch: flag pairing stays
                        # balanced and either way this is a pure-V segment.
                        if idx * block_N + block_N - 1 > T.min(s0 + offset_eff, S_kv - 1):
                            j0 = idx * block_N
                            T.tile.arith_progression(col_pos, j0, 1, block_N)
                            T.tile.broadcast(buf_2d, col_pos, axis=0)  # [bN] -> [hM,bN]
                            T.tile.broadcast(rt_2d, row_thresh, axis=1)  # [hM] -> [hM,bN]
                            T.tile.compare(mask_2d, buf_2d, rt_2d, "LE")
                            T.tile.select(
                                work_ub,
                                mask_2d,
                                work_ub,
                                -T.infinity(accum_dtype),
                                "VSEL_TENSOR_SCALAR_MODE",
                            )

                    T.reduce_max(work_ub, m_i, dim=-1)
                    T.tile.max(m_i, m_i, m_i_prev)
                    # Deferred rescale: store the max difference now and exp it in
                    # phase 2. Arithmetically identical to exp-ing per iteration,
                    # and matching the golden grid depends on that identity.
                    T.tile.sub(r_factors[i, :, :], m_i_prev, m_i)
                    T.tile.broadcast(buf_2d, m_i, axis=1)
                    T.tile.sub(work_ub, work_ub, buf_2d)
                    T.tile.exp(work_ub, work_ub)  # P = exp(s - m_new) in [0, 1]
                    T.reduce_sum(work_ub, sumexp_is[i, :, :], dim=-1)

                    # P -> workspace_2 (inside the SIG_S_HALF ownership closed loop)
                    T.wait_flag("MTE3", "V", SIG_S_HALF)
                    T.copy(work_ub, acc_s_half)  # fp32 -> dtype (P native grid)
                    T.set_flag("V", "MTE3", SIG_S_HALF)
                    T.wait_flag("V", "MTE3", SIG_S_HALF)
                    T.copy(
                        acc_s_half,
                        workspace_2[cid, i, vid * half_M : vid * half_M + half_M, :],
                    )
                    T.set_flag("MTE3", "V", SIG_S_HALF)
                    if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                        T.set_cross_flag("MTE3", SEM_WS2_V2C)
                    # walk advance (pure scalar; eff = THIS slot's task)
                    kv_p1 += 1
                    if kv_p1 == eff:
                        kv_p1 = 0
                        tl_p1 += 1
                T.set_cross_flag("MTE2", SEM_WS1_V2C)

                # ---- Phase 2: O accumulation batch (w3 -> acc_o), flat slots ----
                tl_p2 = T.alloc_var("int32", init=tl_head)
                kv_p2 = T.alloc_var("int32", init=kv_head)
                for i in T.serial(batch_iters):
                    io_side = i % 2
                    task = my_start + tl_p2
                    bx = task % s_blocks
                    by = task // s_blocks % N
                    bz = task // s_blocks // N % B
                    s0 = bx * block_M
                    valid_m = T.min(block_M, S - s0)
                    rows_vid = T.min(half_M, T.max(0, valid_m - vid * half_M))
                    eff = T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                    if kv_p2 == 0:
                        # Task-start resets: the generic task-head fills migrated
                        # to the flat stream -- same first-consumption point, same
                        # fill values. This MUST be phase 2: a phase 1 reset would
                        # destroy the previous task's in-flight acc_o/sumexp, which
                        # the tail below still consumes.
                        T.tile.fill(acc_o, 0.0)
                        T.tile.fill(sumexp, 0.0)
                    T.tile.exp(r_factors[i, :, :], r_factors[i, :, :])  # r_i
                    T.tile.mul(sumexp, sumexp, r_factors[i, :, :])
                    T.tile.add(sumexp, sumexp, sumexp_is[i, :, :])
                    T.tile.broadcast(r_2d, r_factors[i, :, :], axis=1)  # [hM,1]->[hM,D]
                    T.tile.mul(acc_o, acc_o, r_2d)
                    T.wait_flag("V", "MTE2", sig_io3_base + io_side)
                    if i % CROSS_INTERVAL == 0:
                        T.wait_cross_flag(SEM_WS3_C2V)
                    T.copy(
                        workspace_3[cid, i, vid * half_M : vid * half_M + half_M, :],
                        io3[io_side, :, :],
                    )
                    T.set_flag("MTE2", "V", sig_io3_base + io_side)
                    T.wait_flag("MTE2", "V", sig_io3_base + io_side)
                    T.tile.add(acc_o, acc_o, io3[io_side, :, :])
                    T.set_flag("V", "MTE2", sig_io3_base + io_side)
                    if kv_p2 == eff - 1:
                        # ---- Task tail: final normalization and writeback ride the
                        # task's LAST flat slot. The whole SIG_S_HALF closed loop
                        # sits inside the runtime `if`, so its waits and sets enter
                        # and exit together. The condition is vid-independent --
                        # tl/kv/eff walk identically on both vids -- so both AIVs
                        # branch the same way. ----
                        T.tile.add(sumexp, sumexp, 1e-30)  # all-masked-row guard
                        T.tile.broadcast(r_2d, sumexp, axis=1)
                        T.tile.div(acc_o, acc_o, r_2d)
                        T.wait_flag("MTE3", "V", SIG_S_HALF)
                        T.copy(acc_o, out_half)  # fp32 -> dtype
                        T.set_flag("V", "MTE3", SIG_S_HALF)
                        T.wait_flag("V", "MTE3", SIG_S_HALF)
                        if rows_vid > 0:
                            T.copy(
                                out_half[0:rows_vid, :],
                                Output[bz, s0 + vid * half_M : s0 + vid * half_M + rows_vid, by, :],
                            )
                        T.set_flag("MTE3", "V", SIG_S_HALF)
                    # walk advance
                    kv_p2 += 1
                    if kv_p2 == eff:
                        kv_p2 = 0
                        tl_p2 += 1
                T.set_cross_flag("MTE2", SEM_WS3_V2C)
                # batch-head state advance (the phase 2 walk ended at the next head)
                tl_head = tl_p2
                kv_head = kv_p2

        else:
            for t in T.serial(my_count):
                task = my_start + t
                bx = task % s_blocks
                by = task // s_blocks % N
                bz = task // s_blocks // N % B
                s0 = bx * block_M
                valid_m = T.min(block_M, S - s0)
                # Causal KV runtime trim. MUST match the C-scope formula verbatim:
                # both scopes derive their trip counts from it and advance their
                # flag pairing in lockstep.
                eff_kv_iters = T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                eff_num_outer = T.ceildiv(eff_kv_iters, NUM_STAGES)
                rows_vid = T.min(half_M, T.max(0, valid_m - vid * half_M))

                T.tile.fill(acc_o, 0.0)
                T.tile.fill(sumexp, 0.0)
                T.tile.fill(m_i, -(2**30))
                if need_mask:
                    # task-invariant row thresholds (hoisted; vid half-region start):
                    # thresh_r = min(s0 + vid*hM + r + offset_eff, S_kv-1)
                    T.tile.arith_progression(row_thresh, s0 + vid * half_M + offset_eff, 1, half_M)
                    T.tile.min(row_thresh, row_thresh, T.cast(S_kv - 1, accum_dtype))

                for k in T.serial(eff_num_outer):
                    _remaining = eff_kv_iters - k * NUM_STAGES
                    batch_iters = T.if_then_else(_remaining < NUM_STAGES, _remaining, NUM_STAGES)

                    # ---- Phase 1: batched online softmax (w1 -> P -> w2) ----
                    T.wait_cross_flag(SEM_WS2_C2V)
                    for i in T.serial(batch_iters):
                        io_side = i % 2
                        idx = k * NUM_STAGES + i
                        # w1[cid, i, vid half] -> io_buf -> work_ub (fp32 mainline)
                        T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)
                        if i % CROSS_INTERVAL == 0:
                            T.wait_cross_flag(SEM_WS1_C2V)
                        T.copy(
                            workspace_1[cid, i, vid * half_M : vid * half_M + half_M, :],
                            io_buf[io_side, :, :],
                        )
                        T.set_flag("MTE2", "V", SIG_IO_UB + io_side)
                        T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)
                        T.copy(io_buf[io_side, :, :], work_ub)
                        T.set_flag("V", "MTE2", SIG_IO_UB + io_side)

                        T.copy(m_i, m_i_prev)  # snapshot the previous max BEFORE reduce
                        # Golden score grid (precision invariant 1): round16(qk) ->
                        # * s in fp32 -> round16(scores). This chain order is part
                        # of the golden grid and must not be reordered.
                        # rr_ub is the dedicated round-trip scratch and must NEVER
                        # alias acc_s_half, which is the MTE3 cross-core copy source
                        # under the SIG_S_HALF ownership loop.
                        T.copy(work_ub, rr_ub)  # r16(qk)
                        T.copy(rr_ub, work_ub)
                        T.tile.mul(work_ub, work_ub, sm_scale)  # * s in fp32
                        T.copy(work_ub, rr_ub)  # r16(scores) = golden grid
                        T.copy(rr_ub, work_ub)

                        if need_mask:  # noqa: SIM102 - the two-level split is deliberate:
                            # the outer `if` is PARSE-TIME (a Python bool selecting the
                            # kernel variant) while the inner `if` is a RUNTIME TIR
                            # condition. Folding them into `need_mask and <TIR expr>`
                            # would hand a BoolOp AST node to the TVMScript parser.
                            # Runtime skip for fully visible blocks (precision
                            # invariant 4): the condition is uniform over the s0
                            # block, so both vids take the same branch and flag
                            # pairing stays balanced either way.
                            if idx * block_N + block_N - 1 > T.min(s0 + offset_eff, S_kv - 1):
                                j0 = idx * block_N
                                T.tile.arith_progression(col_pos, j0, 1, block_N)
                                T.tile.broadcast(buf_2d, col_pos, axis=0)  # [bN] -> [hM,bN]
                                T.tile.broadcast(rt_2d, row_thresh, axis=1)  # [hM] -> [hM,bN]
                                T.tile.compare(mask_2d, buf_2d, rt_2d, "LE")
                                T.tile.select(
                                    work_ub,
                                    mask_2d,
                                    work_ub,
                                    -T.infinity(accum_dtype),
                                    "VSEL_TENSOR_SCALAR_MODE",
                                )

                        T.reduce_max(work_ub, m_i, dim=-1)
                        T.tile.max(m_i, m_i, m_i_prev)
                        # Deferred rescale: store the max difference now and exp it
                        # in phase 2. Arithmetically identical to exp-ing per
                        # iteration, and the golden grid depends on that identity.
                        T.tile.sub(r_factors[i, :, :], m_i_prev, m_i)
                        T.tile.broadcast(buf_2d, m_i, axis=1)
                        T.tile.sub(work_ub, work_ub, buf_2d)
                        T.tile.exp(work_ub, work_ub)  # P = exp(s - m_new) in [0, 1]
                        T.reduce_sum(work_ub, sumexp_is[i, :, :], dim=-1)

                        if single_kv_block:
                            # Pre-normalization specialization (precision invariant
                            # 2): P_hat = P/(sumexp+1e-30) BEFORE the dtype
                            # rounding, which is the golden one-shot softmax grid.
                            # Phase 2 then skips the rescale and the final division.
                            T.tile.add(sumexp, sumexp_is[0, :, :], 1e-30)
                            T.tile.broadcast(buf_2d, sumexp, axis=1)
                            T.tile.div(work_ub, work_ub, buf_2d)

                        # P -> workspace_2 (SIG_S_HALF ownership closed loop)
                        T.wait_flag("MTE3", "V", SIG_S_HALF)
                        T.copy(work_ub, acc_s_half)  # fp32 -> dtype (P native grid)
                        T.set_flag("V", "MTE3", SIG_S_HALF)
                        T.wait_flag("V", "MTE3", SIG_S_HALF)
                        T.copy(
                            acc_s_half,
                            workspace_2[cid, i, vid * half_M : vid * half_M + half_M, :],
                        )
                        T.set_flag("MTE3", "V", SIG_S_HALF)
                        if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                            T.set_cross_flag("MTE3", SEM_WS2_V2C)
                    T.set_cross_flag("MTE2", SEM_WS1_V2C)

                    # ---- Phase 2: O accumulation batch (w3 -> acc_o) ----
                    for i in T.serial(batch_iters):
                        if not single_kv_block:
                            T.tile.exp(r_factors[i, :, :], r_factors[i, :, :])  # r_i
                            T.tile.mul(sumexp, sumexp, r_factors[i, :, :])
                            T.tile.add(sumexp, sumexp, sumexp_is[i, :, :])
                            T.tile.broadcast(r_2d, r_factors[i, :, :], axis=1)  # [hM,1]->[hM,D]
                            T.tile.mul(acc_o, acc_o, r_2d)
                        io_side = i % 2
                        T.wait_flag("V", "MTE2", sig_io3_base + io_side)
                        if i % CROSS_INTERVAL == 0:
                            T.wait_cross_flag(SEM_WS3_C2V)
                        T.copy(
                            workspace_3[cid, i, vid * half_M : vid * half_M + half_M, :],
                            io3[io_side, :, :],
                        )
                        T.set_flag("MTE2", "V", sig_io3_base + io_side)
                        T.wait_flag("MTE2", "V", sig_io3_base + io_side)
                        T.tile.add(acc_o, acc_o, io3[io_side, :, :])
                        T.set_flag("V", "MTE2", sig_io3_base + io_side)
                    T.set_cross_flag("MTE2", SEM_WS3_V2C)

                # ---- task tail: final normalization (generic path) ----
                if not single_kv_block:
                    T.tile.add(sumexp, sumexp, 1e-30)  # all-masked-row guard
                    T.tile.broadcast(r_2d, sumexp, axis=1)
                    T.tile.div(acc_o, acc_o, r_2d)
                # Writeback inside the SIG_S_HALF closed loop, under the rows_vid
                # runtime guard. The final store must stay inside the ownership
                # loop: out_half is released to MTE3 by the loop's closing set.
                T.wait_flag("MTE3", "V", SIG_S_HALF)
                T.copy(acc_o, out_half)  # fp32 -> dtype
                T.set_flag("V", "MTE3", SIG_S_HALF)
                T.wait_flag("V", "MTE3", SIG_S_HALF)
                if rows_vid > 0:
                    T.copy(
                        out_half[0:rows_vid, :],
                        Output[bz, s0 + vid * half_M : s0 + vid * half_M + rows_vid, by, :],
                    )
                T.set_flag("MTE3", "V", SIG_S_HALF)

        # destroy: consume exactly the tokens pre-set by the init block
        T.wait_flag("V", "MTE2", SIG_IO_UB)
        T.wait_flag("V", "MTE2", SIG_IO_UB + 1)
        if not io3_is_alias:
            T.wait_flag("V", "MTE2", sig_io3_base)
            T.wait_flag("V", "MTE2", sig_io3_base + 1)
        T.wait_flag("MTE3", "V", SIG_S_HALF)

    # ==================== The Form B-mma GEMM1/GEMM2 bodies are INLINE-EXPANDED
    # at their four call sites (single-block / generic x GEMM1 / GEMM2). Do not
    # factor them into a plain-Python closure emitter: TVMScript resolves T.*
    # frames at parse time, so such an emitter's emissions are silently dropped.
    # The kernel still compiles, but the generated C scope then holds only the
    # caller-level instructions and the aicore hangs on an orphaned
    # WaitFlag<M_FIX>. emit_v is a @T.macro precisely because a macro survives
    # parse-time scoping; an inline body needs no such wrapper. ==============
    # ==================== V-side buffer allocation is shared by all forms but
    # must be emitted inside each prim_func (TVMScript scoping). Form dispatch is
    # a Python-level branch that builds one self-contained prim_func. ========
    if form_a:

        @T.prim_func
        def main(
            Q: T.Tensor([B, S, N, D], dtype),  # type: ignore
            K: T.Tensor([B, S_kv, N, D], dtype),  # type: ignore
            V: T.Tensor([B, S_kv, N, D], dtype),  # type: ignore
            Output: T.Tensor([B, S, N, D], dtype),  # type: ignore
            workspace_1: T.Tensor([core_num, NUM_STAGES, block_M, block_N], accum_dtype),  # type: ignore
            workspace_2: T.Tensor([core_num, NUM_STAGES, block_M, block_N], dtype),  # type: ignore
            workspace_3: T.Tensor([core_num, NUM_STAGES, block_M, D], accum_dtype),  # type: ignore
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                # ---- C-side L1, ZN/NZ annotated for the mma operands ----
                q_l1 = T.alloc_L1([block_M, D], dtype)
                k_l1 = T.alloc_L1([block_N, D], dtype)
                v_l1 = T.alloc_L1([block_N, D], dtype)
                p_l1 = T.alloc_L1([block_M, block_N], dtype)
                T.annotate_layout(
                    {
                        q_l1: make_zn_layout(q_l1),
                        k_l1: make_nz_layout(k_l1),
                        p_l1: make_zn_layout(p_l1),
                        v_l1: make_zn_layout(v_l1),
                    }
                )
                # ---- L0 double-slot pools: D == block_N shares one pool, D == 64 splits ----
                l0a_s = T.alloc_L0A([2, block_M, D], dtype)
                l0b_s = T.alloc_L0B([2, D, block_N], dtype)
                l0c_s = T.alloc_L0C([2, block_M, block_N], accum_dtype)
                l0a_o = l0a_s if block_N == D else T.alloc_L0A([2, block_M, block_N], dtype)
                l0b_o = l0b_s if block_N == D else T.alloc_L0B([2, block_N, D], dtype)
                l0c_o = l0c_s if block_N == D else T.alloc_L0C([2, block_M, D], accum_dtype)
                # ---- V-side UB (vid half rows) ----
                io_buf = T.alloc_ub([2, half_M, block_N], accum_dtype)
                io3 = io_buf if io3_is_alias else T.alloc_ub([2, half_M, D], accum_dtype)
                work_ub = T.alloc_ub([half_M, block_N], accum_dtype)
                buf_2d = T.alloc_ub([half_M, block_N], accum_dtype)
                rt_2d = T.alloc_ub([half_M, block_N], accum_dtype)
                r_2d = buf_2d if block_N == D else T.alloc_ub([half_M, D], accum_dtype)
                mask_2d = T.alloc_ub([half_M * block_N // 8], "uint8")
                col_pos = T.alloc_ub([block_N], accum_dtype)
                row_thresh = T.alloc_ub([half_M], accum_dtype)
                rr_ub = T.alloc_ub([half_M, block_N], dtype)
                acc_s_half = T.alloc_ub([half_M, block_N], dtype)
                acc_o = T.alloc_ub([half_M, D], accum_dtype)
                out_half = T.alloc_ub([half_M, D], dtype)
                m_i = T.alloc_ub([half_M, 1], accum_dtype)
                m_i_prev = T.alloc_ub([half_M, 1], accum_dtype)
                sumexp = T.alloc_ub([half_M, 1], accum_dtype)
                r_factors = T.alloc_ub([NUM_STAGES, half_M, 1], accum_dtype)
                sumexp_is = T.alloc_ub([NUM_STAGES, half_M, 1], accum_dtype)

                my_start = cid * q_tasks + T.if_then_else(cid < r_tasks, cid, r_tasks)
                my_count = q_tasks + T.if_then_else(cid < r_tasks, 1, 0)

                # ==================== Cube core (Form A: mma full pipeline) ====================
                with T.Scope("C"):
                    # init: pre-set the tokens the first consumers would otherwise wait for
                    T.set_cross_flag("MTE2", SEM_WS2_C2V)
                    T.set_flag("MTE1", "MTE2", SIG_K_L1)
                    T.set_flag("MTE1", "MTE2", SIG_P_L1)
                    T.set_flag("MTE1", "MTE2", SIG_V_L1)
                    T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                    T.set_flag("M", "MTE1", SIG_L0AB)
                    T.set_flag("M", "MTE1", SIG_L0AB + 1)
                    T.set_flag("FIX", "M", SIG_L0C)
                    T.set_flag("FIX", "M", SIG_L0C + 1)

                    if single_kv_block:
                        # ---- Task-batched single-KV-block C path (Form A) ----
                        # Mirror of the V-side batched branch, carrying the Form A
                        # mma pipeline body (GEMM1/GEMM2 batches) over verbatim
                        # from the generic per-task flow: slot i <-> task
                        # my_start + kb*NUM_STAGES + i. num_tb and batch_tasks come
                        # from the SAME formulas as the V mirror, because the two
                        # scopes advance their flag pairing in lockstep.
                        # Why batch tasks here: with kv_iters == 1 both
                        # eff_num_outer and batch_iters collapse to 1, so the
                        # NUM_STAGES pipeline and the io_buf[2] ping-pong it
                        # feeds never have a second slot in flight -- every task
                        # would run a strictly serial C-QK -> V-softmax -> C-PV
                        # -> V-epilogue chain with no C/V overlap. T.Pipelined
                        # is unavailable here, so batching the TASK dimension is
                        # how the overlap is expressed manually. Two Form-A-only
                        # deltas, both pure Q scheduling, no arithmetic change:
                        #   (a) the Q GM->L1 load moves INSIDE the slot loop,
                        #       because each task has its own s0, and
                        #   (b) q_l1 -> l0a_s[side] is issued for EVERY slot (the
                        #       generic body guards it with `if i < 2` because Q
                        #       is kv-invariant there; here Q changes per slot).
                        # SIG_Q_L1 token balance: per slot, wait(MTE1->MTE2) +
                        # set(MTE2->MTE1) + wait(MTE2->MTE1) + set(MTE1->MTE2)
                        # right after the single MTE1 read of q_l1 = net 0, the
                        # same accounting as the generic task level, so the
                        # task-end return at the bottom of the else branch is
                        # deliberately NOT emitted here.
                        num_tb = T.ceildiv(my_count, NUM_STAGES)
                        for kb in T.serial(num_tb):
                            _remaining_t = my_count - kb * NUM_STAGES
                            batch_tasks = T.if_then_else(_remaining_t < NUM_STAGES, _remaining_t, NUM_STAGES)
                            valid_n = min(block_N, S_kv)  # single block: idx == 0

                            # --- GEMM1 batch: per-task scores -> w1 slot i ---
                            T.wait_cross_flag(SEM_WS1_V2C)
                            for i in T.serial(batch_tasks):
                                side = i % 2
                                task = my_start + kb * NUM_STAGES + i
                                bx = task % s_blocks
                                by = task // s_blocks % N
                                bz = task // s_blocks // N % B
                                s0 = bx * block_M
                                valid_m = T.min(block_M, S - s0)
                                # Q: GM -> L1 (per-slot residency; BSHD row-stride)
                                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                                T.copy(Q[bz, s0 : s0 + valid_m, by, :], q_l1)
                                T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                                T.wait_flag("MTE2", "MTE1", SIG_Q_L1)
                                # K: GM -> L1 (single block => j0 == 0)
                                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                                T.copy(K[bz, 0:valid_n, by, :], k_l1)
                                T.set_flag("MTE2", "MTE1", SIG_K_L1)
                                # Q: L1 -> L0A (EVERY slot: task-dependent Q)
                                T.wait_flag("M", "MTE1", SIG_L0AB + side)
                                T.copy(q_l1, l0a_s[side, :, :])
                                T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                                # K: L1 -> L0B transposed (NZ)
                                T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                                T.copy(k_l1, l0b_s[side, :, :], transpose=True)
                                T.set_flag("MTE1", "MTE2", SIG_K_L1)
                                T.set_flag("MTE1", "M", SIG_L0AB + side)
                                # mma: Q K^T -> L0C
                                T.wait_flag("MTE1", "M", SIG_L0AB + side)
                                T.wait_flag("FIX", "M", SIG_L0C + side)
                                T.mma(l0a_s[side, :, :], l0b_s[side, :, :], l0c_s[side, :, :], init=True)
                                T.set_flag("M", "MTE1", SIG_L0AB + side)
                                T.set_flag("M", "FIX", SIG_L0C + side)
                                # L0C -> w1 (FIX)
                                T.wait_flag("M", "FIX", SIG_L0C + side)
                                T.copy(l0c_s[side, :, :], workspace_1[cid, i, :, :])
                                T.set_flag("FIX", "M", SIG_L0C + side)
                                if (i + 1) % CROSS_INTERVAL == 0 or i == batch_tasks - 1:
                                    T.set_cross_flag("FIX", SEM_WS1_C2V)

                            # --- GEMM2 batch: per-task P (w2 slot i) @ V -> Output
                            # --- GEMM2 batch: per-task P (workspace_2 slot i) @ V,
                            # written straight to Output by the FIX pipe. Under the
                            # pre-normalization specialization O = P_hat @ V is
                            # already FINAL, so the fp32 workspace_3 GM round trip,
                            # the V-side phase-2 acc_o pass and the workspace_3
                            # semaphore pair are all unnecessary and are omitted.
                            # The fixpipe fp32 -> dtype RNE cast is bit-identical to
                            # the V-side cast it replaces, and the L0C -> BSHD 4D
                            # store honours a runtime row extent (including a tail
                            # valid_m < block_M, bf16 and D=128).
                            # The C side writes ALL valid_m rows at once; there is
                            # no vid row split on this path.
                            for i in T.serial(batch_tasks):
                                side = i % 2
                                task = my_start + kb * NUM_STAGES + i
                                bx = task % s_blocks
                                by = task // s_blocks % N
                                bz = task // s_blocks // N % B
                                s0 = bx * block_M
                                valid_m = T.min(block_M, S - s0)
                                # V: GM -> L1
                                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                                T.copy(V[bz, 0:valid_n, by, :], v_l1)
                                T.set_flag("MTE2", "MTE1", SIG_V_L1)
                                # P: w2 -> p_l1
                                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                                if i % CROSS_INTERVAL == 0:
                                    T.wait_cross_flag(SEM_WS2_V2C)
                                T.copy(workspace_2[cid, i, :, :], p_l1)
                                T.set_flag("MTE2", "MTE1", SIG_P_L1)
                                # V: L1 -> L0B (no transpose for PV)
                                T.wait_flag("MTE2", "MTE1", SIG_V_L1)
                                T.wait_flag("M", "MTE1", SIG_L0AB + side)
                                T.copy(v_l1, l0b_o[side, :, :])
                                T.set_flag("MTE1", "MTE2", SIG_V_L1)
                                # P: L1 -> L0A
                                T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                                T.copy(p_l1, l0a_o[side, :, :])
                                T.set_flag("MTE1", "MTE2", SIG_P_L1)
                                T.set_flag("MTE1", "M", SIG_L0AB + side)
                                # mma: P V -> L0C
                                T.wait_flag("MTE1", "M", SIG_L0AB + side)
                                T.wait_flag("FIX", "M", SIG_L0C + side)
                                T.mma(l0a_o[side, :, :], l0b_o[side, :, :], l0c_o[side, :, :], init=True)
                                T.set_flag("M", "MTE1", SIG_L0AB + side)
                                T.set_flag("M", "FIX", SIG_L0C + side)
                                # L0C -> Output (FIX direct write, runtime row
                                # extent; SIG_L0C slot ownership unchanged)
                                T.wait_flag("M", "FIX", SIG_L0C + side)
                                T.copy(l0c_o[side, :, :], Output[bz, s0 : s0 + valid_m, by, :])
                                T.set_flag("FIX", "M", SIG_L0C + side)

                            T.set_cross_flag("MTE2", SEM_WS2_C2V)

                    else:
                        # ---- Flat (task, kv) continuous slot stream: the Form A
                        # generic multi-KV-block domain. Mirror of the Form B-mma
                        # flat branch, with TWO Form-A-specific deltas, both pure
                        # Q/l0a_s scheduling with no arithmetic change:
                        # (a) the l0a_s reload condition is `i < 2 or kv < 2`.
                        #     When D == block_N the shared pool means l0a_o aliases
                        #     l0a_s and every GEMM2 batch overwrites it, so
                        #     batch-head slots MUST reload Q (this preserves the
                        #     original `if i < 2` semantics); in addition a flat
                        #     mid-batch task switch needs the new task's first two
                        #     slots to reload their sides, since Q is task-invariant
                        #     and kv >= 2 slots reuse the resident copy.
                        # (b) the q_l1 return (SIG_Q_L1 MTE1->MTE2 set) moves to the
                        #     task's LAST slot (kv == eff-1), not kv == 1. Tasks that
                        #     span batch boundaries re-read q_l1 at later batch-head
                        #     slots (delta-a reloads); releasing earlier would let
                        #     the NEXT task's kv0 MTE2 load overwrite q_l1 while
                        #     those reads are still queued on MTE1. M-pipe/MTE1
                        #     in-order execution from the last slot covers every
                        #     q_l1 read of the task. Per-task accounting is
                        #     unchanged: one full SIG_Q_L1 round (kv == 0, with the
                        #     DMA and all its flags inside one condition) plus one
                        #     return (kv == eff-1) = net 0, init/destroy untouched.
                        # The walk state machine, the batch skeleton and the V
                        # mirror are shared with the Form B-mma flat branch, and
                        # C/V derive every bound from the same formulas. ----
                        total_flat = T.alloc_var("int32", init=0)
                        for t in T.serial(my_count):
                            _tw = my_start + t
                            _s0w = (_tw % s_blocks) * block_M
                            _effw = T.floordiv(T.min(_s0w + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                            total_flat += _effw
                        num_fb = T.ceildiv(total_flat, NUM_STAGES)
                        tl_head = T.alloc_var("int32", init=0)  # persistent batch-head walk state
                        kv_head = T.alloc_var("int32", init=0)
                        for fb in T.serial(num_fb):
                            _rem_f = total_flat - fb * NUM_STAGES
                            batch_iters = T.if_then_else(_rem_f < NUM_STAGES, _rem_f, NUM_STAGES)

                            # --- GEMM1 batch: flat slots -> w1 ---
                            T.wait_cross_flag(SEM_WS1_V2C)
                            tl_g1 = T.alloc_var("int32", init=tl_head)
                            kv_g1 = T.alloc_var("int32", init=kv_head)
                            for i in T.serial(batch_iters):
                                side = i % 2
                                task = my_start + tl_g1
                                bx = task % s_blocks
                                by = task // s_blocks % N
                                bz = task // s_blocks // N % B
                                s0 = bx * block_M
                                valid_m = T.min(block_M, S - s0)
                                eff_kv_iters = (
                                    T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                                )
                                idx = kv_g1
                                j0 = idx * block_N
                                valid_n = T.min(block_N, S_kv - j0)
                                if kv_g1 == 0:
                                    # ---- task switch: Q GM->L1 (full SIG_Q_L1
                                    # MTE2 round inside the condition; the l0a_s
                                    # reload rides this slot's L0AB round below) ----
                                    T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                                    T.copy(Q[bz, s0 : s0 + valid_m, by, :], q_l1)
                                    T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                                    T.wait_flag("MTE2", "MTE1", SIG_Q_L1)
                                # K: GM -> L1 (BSHD row-stride, runtime extent)
                                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                                T.copy(K[bz, j0 : j0 + valid_n, by, :], k_l1)
                                T.set_flag("MTE2", "MTE1", SIG_K_L1)
                                # Q: L1 -> L0A (delta-a: batch-head OR task-head sides)
                                T.wait_flag("M", "MTE1", SIG_L0AB + side)
                                if i < 2 or kv_g1 < 2:
                                    T.copy(q_l1, l0a_s[side, :, :])
                                # K: L1 -> L0B transposed (NZ)
                                T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                                T.copy(k_l1, l0b_s[side, :, :], transpose=True)
                                T.set_flag("MTE1", "MTE2", SIG_K_L1)
                                if kv_g1 == eff_kv_iters - 1:
                                    # ---- task-end q_l1 return (delta-b: LAST
                                    # slot; MTE1 in-order covers every q_l1 read
                                    # of the task incl. later batch-head reloads) ----
                                    T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                                T.set_flag("MTE1", "M", SIG_L0AB + side)
                                # mma: Q K^T -> L0C
                                T.wait_flag("MTE1", "M", SIG_L0AB + side)
                                T.wait_flag("FIX", "M", SIG_L0C + side)
                                T.mma(l0a_s[side, :, :], l0b_s[side, :, :], l0c_s[side, :, :], init=True)
                                T.set_flag("M", "MTE1", SIG_L0AB + side)
                                T.set_flag("M", "FIX", SIG_L0C + side)
                                # L0C -> w1 (FIX)
                                T.wait_flag("M", "FIX", SIG_L0C + side)
                                T.copy(l0c_s[side, :, :], workspace_1[cid, i, :, :])
                                T.set_flag("FIX", "M", SIG_L0C + side)
                                if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                                    T.set_cross_flag("FIX", SEM_WS1_C2V)
                                # walk advance (pure scalar, no DMA/flags)
                                kv_g1 += 1
                                if kv_g1 == eff_kv_iters:
                                    kv_g1 = 0
                                    tl_g1 += 1

                            # --- GEMM2 batch: consume w2 (P), produce PV -> w3 ---
                            T.wait_cross_flag(SEM_WS3_V2C)
                            tl_g2 = T.alloc_var("int32", init=tl_head)
                            kv_g2 = T.alloc_var("int32", init=kv_head)
                            for i in T.serial(batch_iters):
                                side = i % 2
                                task = my_start + tl_g2
                                bx = task % s_blocks
                                by = task // s_blocks % N
                                bz = task // s_blocks // N % B
                                s0 = bx * block_M
                                eff_kv_iters = (
                                    T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                                )
                                idx = kv_g2
                                j0 = idx * block_N
                                valid_n = T.min(block_N, S_kv - j0)
                                # V: GM -> L1
                                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                                T.copy(V[bz, j0 : j0 + valid_n, by, :], v_l1)
                                T.set_flag("MTE2", "MTE1", SIG_V_L1)
                                # P: w2 -> p_l1
                                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                                if i % CROSS_INTERVAL == 0:
                                    T.wait_cross_flag(SEM_WS2_V2C)
                                T.copy(workspace_2[cid, i, :, :], p_l1)
                                T.set_flag("MTE2", "MTE1", SIG_P_L1)
                                # V: L1 -> L0B (no transpose for PV)
                                T.wait_flag("MTE2", "MTE1", SIG_V_L1)
                                T.wait_flag("M", "MTE1", SIG_L0AB + side)
                                T.copy(v_l1, l0b_o[side, :, :])
                                T.set_flag("MTE1", "MTE2", SIG_V_L1)
                                # P: L1 -> L0A
                                T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                                T.copy(p_l1, l0a_o[side, :, :])
                                T.set_flag("MTE1", "MTE2", SIG_P_L1)
                                T.set_flag("MTE1", "M", SIG_L0AB + side)
                                # mma: P V -> L0C
                                T.wait_flag("MTE1", "M", SIG_L0AB + side)
                                T.wait_flag("FIX", "M", SIG_L0C + side)
                                T.mma(l0a_o[side, :, :], l0b_o[side, :, :], l0c_o[side, :, :], init=True)
                                T.set_flag("M", "MTE1", SIG_L0AB + side)
                                T.set_flag("M", "FIX", SIG_L0C + side)
                                # L0C -> w3 (FIX)
                                T.wait_flag("M", "FIX", SIG_L0C + side)
                                T.copy(l0c_o[side, :, :], workspace_3[cid, i, :, :])
                                T.set_flag("FIX", "M", SIG_L0C + side)
                                if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                                    T.set_cross_flag("FIX", SEM_WS3_C2V)
                                # walk advance (pure scalar, no DMA/flags)
                                kv_g2 += 1
                                if kv_g2 == eff_kv_iters:
                                    kv_g2 = 0
                                    tl_g2 += 1

                            T.set_cross_flag("MTE2", SEM_WS2_C2V)
                            # batch-head state advance (GEMM2 walk ended at the next head)
                            tl_head = tl_g2
                            kv_head = kv_g2

                    # destroy: consume outstanding init-direction flags
                    T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                    T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                    T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                    T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                    T.wait_flag("M", "MTE1", SIG_L0AB)
                    T.wait_flag("M", "MTE1", SIG_L0AB + 1)
                    T.wait_flag("FIX", "M", SIG_L0C)
                    T.wait_flag("FIX", "M", SIG_L0C + 1)

                # ==================== Vector core (shared emitter) ====================
                with T.Scope("V"):
                    emit_v(
                        io_buf,
                        io3,
                        work_ub,
                        buf_2d,
                        rt_2d,
                        r_2d,
                        mask_2d,
                        col_pos,
                        row_thresh,
                        rr_ub,
                        acc_s_half,
                        acc_o,
                        out_half,
                        m_i,
                        m_i_prev,
                        sumexp,
                        r_factors,
                        sumexp_is,
                        Output,
                        workspace_1,
                        workspace_2,
                        workspace_3,
                        cid,
                        vid,
                        my_start,
                        my_count,
                        sig_io3,
                    )

        return main

    elif form_b_mma:
        # ==================== Form B-mma (prefill S > 16 and D = 256): a manual
        # K/N-split T.mma pipeline. gemm_v0 drains the M pipe with a per-step
        # PipeBarrier<PIPE_M>, which at this width leaves the scalar pipe waiting
        # on MTE1 for most of the wall time. Chunk-level token flow instead
        # overlaps three pipes: MTE1(c+1) || M(c) || FIX(c-1).
        # The batch protocol is the one the other forms use too (task batch when
        # single_kv_block, flat stream when multi-block). GEMM2 uses a narrow
        # l0c_o pool plus a per-chunk column-segment FIX write, because an mma
        # targeting an L0C column sub-region is rejected by the cube unit
        # ("CUBE instruction parameter is invalid").
        # Token accounting: init pre-sets 13, destroy waits 12, every round is
        # self-paired, no token is left dangling and no DMA is conditional without
        # its flags. FORBIDDEN in this prim_func: T.gemm_v0 (it would collide on
        # internal event id 0), T.barrier_all, threads=, and T.Pipelined. ======
        @T.prim_func
        def main(
            Q: T.Tensor([B, S, N, D], dtype),  # type: ignore
            K: T.Tensor([B, S_kv, N, D], dtype),  # type: ignore
            V: T.Tensor([B, S_kv, N, D], dtype),  # type: ignore
            Output: T.Tensor([B, S, N, D], dtype),  # type: ignore
            workspace_1: T.Tensor([core_num, NUM_STAGES, block_M, block_N], accum_dtype),  # type: ignore
            workspace_2: T.Tensor([core_num, NUM_STAGES, block_M, block_N], dtype),  # type: ignore
            workspace_3: T.Tensor([core_num, NUM_STAGES, block_M, D], accum_dtype),  # type: ignore
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                # ---- C-side L1. k_l1 is ZN-annotated here, NOT NZ, and the
                # distinction is load-bearing. The copy_gm_to_l1 template always
                # writes canonical zN (LayoutL1 is hard-tagged layout::zN), while a
                # column-segment transpose read computes its source offset as
                # layout_map.Forward(region_min). Under an nZ annotation that yields
                # the nZ-formula offset (2048 for column 128 of a [128, 256] tile),
                # which does not match the physical zN fill; the correct base is
                # 16384, i.e. the zN formula, which is also the nZ(256, 128) tile-1
                # base. Full-buffer reads start at offset 0 and are layout-agnostic,
                # which is why Form A's NZ annotation on k_l1 is and stays correct
                # there. Annotating ZN explicitly gives the k column segments the
                # same Forward formula as the q/v column segments.
                q_l1 = T.alloc_L1([block_M, D], dtype)
                k_l1 = T.alloc_L1([block_N, D], dtype)
                v_l1 = T.alloc_L1([block_N, D], dtype)
                p_l1 = T.alloc_L1([block_M, block_N], dtype)
                T.annotate_layout(
                    {
                        q_l1: make_zn_layout(q_l1),
                        k_l1: make_zn_layout(k_l1),
                        p_l1: make_zn_layout(p_l1),
                        v_l1: make_zn_layout(v_l1),
                    }
                )
                # ---- Explicit L0 pools. Budget at KC=NC=128, block_M=32,
                # block_N=128: L0A 32 KB, L0B 64 KB (the shared pool sits exactly at
                # its ceiling), L0C 64 KB for the narrow PV pool. ----
                l0a_s = T.alloc_L0A([2, block_M, KC], dtype)  # GEMM1 A: Q k-chunks (task-resident)
                l0a_o = T.alloc_L0A([2, block_M, block_N], dtype)  # GEMM2 A: P, side ping-pong
                l0b_pool = T.alloc_L0B([2, KC, block_N], dtype)  # SHARED: GEMM1 K^T kc / GEMM2 V nc
                l0c_s = T.alloc_L0C([2, block_M, block_N], accum_dtype)  # scores: K-split chain target
                l0c_o = T.alloc_L0C([2, block_M, NC], accum_dtype)  # PV: narrow pool, slot=chunk
                # ---- V-side UB: identical in all three forms (emit_v is shared) ----
                io_buf = T.alloc_ub([2, half_M, block_N], accum_dtype)
                io3 = io_buf if io3_is_alias else T.alloc_ub([2, half_M, D], accum_dtype)
                work_ub = T.alloc_ub([half_M, block_N], accum_dtype)
                buf_2d = T.alloc_ub([half_M, block_N], accum_dtype)
                rt_2d = T.alloc_ub([half_M, block_N], accum_dtype)
                r_2d = buf_2d if block_N == D else T.alloc_ub([half_M, D], accum_dtype)
                mask_2d = T.alloc_ub([half_M * block_N // 8], "uint8")
                col_pos = T.alloc_ub([block_N], accum_dtype)
                row_thresh = T.alloc_ub([half_M], accum_dtype)
                rr_ub = T.alloc_ub([half_M, block_N], dtype)
                acc_s_half = T.alloc_ub([half_M, block_N], dtype)
                acc_o = T.alloc_ub([half_M, D], accum_dtype)
                out_half = T.alloc_ub([half_M, D], dtype)
                m_i = T.alloc_ub([half_M, 1], accum_dtype)
                m_i_prev = T.alloc_ub([half_M, 1], accum_dtype)
                sumexp = T.alloc_ub([half_M, 1], accum_dtype)
                r_factors = T.alloc_ub([NUM_STAGES, half_M, 1], accum_dtype)
                sumexp_is = T.alloc_ub([NUM_STAGES, half_M, 1], accum_dtype)

                my_start = cid * q_tasks + T.if_then_else(cid < r_tasks, cid, r_tasks)
                my_count = q_tasks + T.if_then_else(cid < r_tasks, 1, 0)

                # ==================== Cube core (Form B-mma) ====================
                with T.Scope("C"):
                    # init: 13 tokens pre-set = 1 semaphore + 4 L1 + 2 L0KC +
                    # 2 L0AO + 2 L0AS + 2 L0C; destroy waits 12 of them.
                    T.set_cross_flag("MTE2", SEM_WS2_C2V)
                    T.set_flag("MTE1", "MTE2", SIG_K_L1)
                    T.set_flag("MTE1", "MTE2", SIG_P_L1)
                    T.set_flag("MTE1", "MTE2", SIG_V_L1)
                    T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                    T.set_flag("M", "MTE1", SIG_L0KC)
                    T.set_flag("M", "MTE1", SIG_L0KC + 1)
                    T.set_flag("M", "MTE1", SIG_L0AO)
                    T.set_flag("M", "MTE1", SIG_L0AO + 1)
                    T.set_flag("M", "MTE1", SIG_L0AS)
                    T.set_flag("M", "MTE1", SIG_L0AS + 1)
                    T.set_flag("FIX", "M", SIG_L0C)
                    T.set_flag("FIX", "M", SIG_L0C + 1)

                    if single_kv_block:
                        # ---- Task-batched single-KV-block C path: the same
                        # protocol mirror as the other forms, with this form's mma
                        # body. Slot i <-> task my_start + kb*NUM_STAGES + i;
                        # num_tb/batch_tasks come from the SAME formulas as the V
                        # mirror, so flag pairing stays in lockstep. ----
                        num_tb = T.ceildiv(my_count, NUM_STAGES)
                        for kb in T.serial(num_tb):
                            _remaining_t = my_count - kb * NUM_STAGES
                            batch_tasks = T.if_then_else(_remaining_t < NUM_STAGES, _remaining_t, NUM_STAGES)
                            valid_n = min(block_N, S_kv)  # single block: idx == 0

                            # --- GEMM1 batch: per-task scores -> w1 slot i ---
                            T.wait_cross_flag(SEM_WS1_V2C)
                            for i in T.serial(batch_tasks):
                                side = i % 2
                                task = my_start + kb * NUM_STAGES + i
                                bx = task % s_blocks
                                by = task // s_blocks % N
                                bz = task // s_blocks // N % B
                                s0 = bx * block_M
                                valid_m = T.min(block_M, S - s0)
                                # Q: GM -> L1, per slot (Q is task-dependent here)
                                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                                T.copy(Q[bz, s0 : s0 + valid_m, by, :], q_l1)
                                T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                                T.wait_flag("MTE2", "MTE1", SIG_Q_L1)
                                # K: GM -> L1 (single block => j0 == 0)
                                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                                T.copy(K[bz, 0:valid_n, by, :], k_l1)
                                T.set_flag("MTE2", "MTE1", SIG_K_L1)
                                # Q: L1 column segments -> L0A, EVERY slot, since Q
                                # is task-dependent here. One SIG_L0AS round per
                                # slot: the trailing M waits move the M-pipe block
                                # to a point where no M work is in flight, so they
                                # cost nothing, and the task-internal mma reads of
                                # l0a_s are covered by M-pipe in-order execution and
                                # therefore carry no SIG_L0AS at all.
                                T.wait_flag("M", "MTE1", SIG_L0AS)
                                T.copy(q_l1[:, 0:KC], l0a_s[0, :, :])
                                T.wait_flag("M", "MTE1", SIG_L0AS + 1)
                                T.copy(q_l1[:, KC : 2 * KC], l0a_s[1, :, :])
                                T.set_flag("MTE1", "M", SIG_L0AS)
                                T.set_flag("MTE1", "M", SIG_L0AS + 1)
                                T.wait_flag("MTE1", "M", SIG_L0AS)
                                T.wait_flag("MTE1", "M", SIG_L0AS + 1)
                                T.set_flag("MTE1", "MTE2", SIG_Q_L1)  # q_l1 MTE1 read done
                                # K-split mma chain: chunk c=0 with init=True, then
                                # chunk c=1 with init=False so the fp32 L0C
                                # accumulation continues in ascending K order.
                                T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                                # ---- k-chunk c=0 (K columns 0:KC) ----
                                T.wait_flag("M", "MTE1", SIG_L0KC)  # pool chunk 0 released
                                T.copy(k_l1[:, 0:KC], l0b_pool[0, :, :], transpose=True)
                                T.set_flag("MTE1", "M", SIG_L0KC)
                                T.wait_flag("MTE1", "M", SIG_L0KC)
                                T.wait_flag("FIX", "M", SIG_L0C + side)  # l0c_s slot writable
                                T.mma(l0a_s[0, :, :], l0b_pool[0, :, :], l0c_s[side, :, :], init=True)
                                T.set_flag("M", "MTE1", SIG_L0KC)  # pool chunk 0 released
                                # ---- k-chunk c=1 (K columns KC:2*KC, chained) ----
                                T.wait_flag("M", "MTE1", SIG_L0KC + 1)
                                T.copy(k_l1[:, KC : 2 * KC], l0b_pool[1, :, :], transpose=True)
                                T.set_flag("MTE1", "M", SIG_L0KC + 1)
                                T.wait_flag("MTE1", "M", SIG_L0KC + 1)
                                T.mma(l0a_s[1, :, :], l0b_pool[1, :, :], l0c_s[side, :, :], init=False)
                                T.set_flag("M", "MTE1", SIG_L0KC + 1)
                                T.set_flag("M", "FIX", SIG_L0C + side)  # scores full block ready
                                T.set_flag("MTE1", "MTE2", SIG_K_L1)  # k_l1 read done (after c1 load)
                                # L0C -> w1 (FIX full block)
                                T.wait_flag("M", "FIX", SIG_L0C + side)
                                T.copy(l0c_s[side, :, :], workspace_1[cid, i, :, :])
                                T.set_flag("FIX", "M", SIG_L0C + side)
                                # Slot-end l0a_s return. By M-pipe order this slot's
                                # last GEMM1 mma has already read l0a_s and GEMM2
                                # never reads it, so the next slot's head wait
                                # consumes the token: net 0 per slot.
                                T.set_flag("M", "MTE1", SIG_L0AS)
                                T.set_flag("M", "MTE1", SIG_L0AS + 1)
                                if (i + 1) % CROSS_INTERVAL == 0 or i == batch_tasks - 1:
                                    T.set_cross_flag("FIX", SEM_WS1_C2V)

                            # --- GEMM2 batch: per-task P (workspace_2 slot i) @ V ->
                            # workspace_3 slot i. This form keeps workspace_3 and
                            # the V-side phase 2, matching the emit_v
                            # `if not form_a:` gate, so the V scope is shared
                            # unchanged with the gemm_v0 form. ---
                            T.wait_cross_flag(SEM_WS3_V2C)
                            for i in T.serial(batch_tasks):
                                side = i % 2
                                task = my_start + kb * NUM_STAGES + i
                                by = task // s_blocks % N
                                bz = task // s_blocks // N % B
                                # V: GM -> L1
                                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                                T.copy(V[bz, 0:valid_n, by, :], v_l1)
                                T.set_flag("MTE2", "MTE1", SIG_V_L1)
                                # P: w2 -> p_l1
                                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                                if i % CROSS_INTERVAL == 0:
                                    T.wait_cross_flag(SEM_WS2_V2C)
                                T.copy(workspace_2[cid, i, :, :], p_l1)
                                T.set_flag("MTE2", "MTE1", SIG_P_L1)
                                # P: L1 -> L0A (side ping-pong)
                                T.wait_flag("M", "MTE1", SIG_L0AO + side)
                                T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                                T.copy(p_l1, l0a_o[side, :, :])
                                T.set_flag("MTE1", "MTE2", SIG_P_L1)
                                T.set_flag("MTE1", "M", SIG_L0AO + side)
                                T.wait_flag("MTE2", "MTE1", SIG_V_L1)
                                # N-split mma plus a per-chunk column-segment FIX
                                # write. Each chunk accumulates into its own narrow
                                # l0c_o slot with init=True, so the chunks are
                                # column-independent and need no chaining.
                                # ---- n-chunk c=0 (V cols 0:NC -> w3 cols 0:NC) ----
                                T.wait_flag("M", "MTE1", SIG_L0KC)
                                T.copy(v_l1[:, 0:NC], l0b_pool[0, :, :])  # direct, no transpose
                                T.set_flag("MTE1", "M", SIG_L0KC)
                                T.wait_flag("MTE1", "M", SIG_L0KC)
                                T.wait_flag("MTE1", "M", SIG_L0AO + side)  # p ready
                                T.wait_flag("FIX", "M", SIG_L0C)  # narrow slot 0 writable
                                T.mma(l0a_o[side, :, :], l0b_pool[0, :, :], l0c_o[0, :, :], init=True)
                                T.set_flag("M", "MTE1", SIG_L0KC)
                                T.set_flag("M", "FIX", SIG_L0C)  # chunk 0 ready
                                T.wait_flag("M", "FIX", SIG_L0C)
                                T.copy(l0c_o[0, :, :], workspace_3[cid, i, :, 0:NC])  # colseg FIX
                                T.set_flag("FIX", "M", SIG_L0C)
                                # ---- n-chunk c=1 (V cols NC:2*NC -> w3 cols NC:2*NC) ----
                                T.wait_flag("M", "MTE1", SIG_L0KC + 1)
                                T.copy(v_l1[:, NC : 2 * NC], l0b_pool[1, :, :])
                                T.set_flag("MTE1", "M", SIG_L0KC + 1)
                                T.wait_flag("MTE1", "M", SIG_L0KC + 1)
                                T.wait_flag("FIX", "M", SIG_L0C + 1)  # narrow slot 1 writable
                                T.mma(l0a_o[side, :, :], l0b_pool[1, :, :], l0c_o[1, :, :], init=True)
                                T.set_flag("M", "MTE1", SIG_L0KC + 1)
                                T.set_flag("M", "FIX", SIG_L0C + 1)
                                T.set_flag("M", "MTE1", SIG_L0AO + side)  # l0a_o both chunks read
                                T.wait_flag("M", "FIX", SIG_L0C + 1)
                                T.copy(l0c_o[1, :, :], workspace_3[cid, i, :, NC : 2 * NC])
                                T.set_flag("FIX", "M", SIG_L0C + 1)
                                T.set_flag("MTE1", "MTE2", SIG_V_L1)  # v_l1 read done (both chunks)
                                if (i + 1) % CROSS_INTERVAL == 0 or i == batch_tasks - 1:
                                    # after BOTH colseg FIX writes (FIX pipe is in-order)
                                    T.set_cross_flag("FIX", SEM_WS3_C2V)

                            T.set_cross_flag("MTE2", SEM_WS2_C2V)

                    else:
                        # ---- Flat (task, kv) continuous slot stream: this form's
                        # generic multi-KV-block domain. Replaces per-task nested
                        # [task][outer-batch] loops so a slot batch can cross task
                        # boundaries without draining. Structure mirrors the
                        # emit_v flat branch VERBATIM: a one-time O(my_count)
                        # scalar walk builds total_flat = sum(eff_kv(t)); slots
                        # are batched in groups of NUM_STAGES; a per-slot walk
                        # state (T.alloc_var) maps slot -> (task, kv). Causal
                        # masking makes eff_kv runtime-nonuniform per task, which
                        # is what forces the walk -- a compile-time-uniform mapping
                        # only works for decode.
                        # Task switch (kv == 0): Q GM->L1 plus the l0a_s preload.
                        # The FULL SIG_Q_L1 and SIG_L0AS rounds both live INSIDE
                        # the runtime `if`, so their waits and sets enter and exit
                        # together and the per-task net stays 0 -- the task-granular
                        # accounting moved to switch granularity. A conditional DMA
                        # must always carry all of its flags in the same condition.
                        # Task end (kv == eff-1): the L0AS return sets, placed by
                        # M-pipe order after this task's last GEMM1 mma read.
                        # The GEMM1 batch still fully precedes the GEMM2 batch,
                        # because l0b_pool is shared between the two phases and
                        # must not interleave at iteration level. Per-slot bodies
                        # are the generic ones verbatim. ----
                        total_flat = T.alloc_var("int32", init=0)
                        for t in T.serial(my_count):
                            _tw = my_start + t
                            _s0w = (_tw % s_blocks) * block_M
                            _effw = T.floordiv(T.min(_s0w + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                            total_flat += _effw
                        num_fb = T.ceildiv(total_flat, NUM_STAGES)
                        tl_head = T.alloc_var("int32", init=0)  # persistent batch-head walk state
                        kv_head = T.alloc_var("int32", init=0)
                        for fb in T.serial(num_fb):
                            _rem_f = total_flat - fb * NUM_STAGES
                            batch_iters = T.if_then_else(_rem_f < NUM_STAGES, _rem_f, NUM_STAGES)

                            # --- GEMM1 batch: flat slots -> w1 ---
                            T.wait_cross_flag(SEM_WS1_V2C)
                            tl_g1 = T.alloc_var("int32", init=tl_head)
                            kv_g1 = T.alloc_var("int32", init=kv_head)
                            for i in T.serial(batch_iters):
                                side = i % 2
                                task = my_start + tl_g1
                                bx = task % s_blocks
                                by = task // s_blocks % N
                                bz = task // s_blocks // N % B
                                s0 = bx * block_M
                                valid_m = T.min(block_M, S - s0)
                                eff_kv_iters = (
                                    T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                                )
                                idx = kv_g1
                                j0 = idx * block_N
                                valid_n = T.min(block_N, S_kv - j0)
                                if kv_g1 == 0:
                                    # ---- Task switch: Q load (SIG_Q_L1 round) plus
                                    # the l0a_s preload (SIG_L0AS round). Both rounds
                                    # sit entirely inside this condition, so every
                                    # wait is matched by its set on the same
                                    # execution path. ----
                                    T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                                    T.copy(Q[bz, s0 : s0 + valid_m, by, :], q_l1)
                                    T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                                    T.wait_flag("MTE2", "MTE1", SIG_Q_L1)
                                    T.wait_flag("M", "MTE1", SIG_L0AS)
                                    T.copy(q_l1[:, 0:KC], l0a_s[0, :, :])
                                    T.wait_flag("M", "MTE1", SIG_L0AS + 1)
                                    T.copy(q_l1[:, KC : 2 * KC], l0a_s[1, :, :])
                                    T.set_flag("MTE1", "M", SIG_L0AS)
                                    T.set_flag("MTE1", "M", SIG_L0AS + 1)
                                    T.wait_flag("MTE1", "M", SIG_L0AS)
                                    T.wait_flag("MTE1", "M", SIG_L0AS + 1)
                                    T.set_flag("MTE1", "MTE2", SIG_Q_L1)  # q_l1 MTE1 read done
                                # K: GM -> L1 (BSHD row-stride, runtime extent)
                                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                                T.copy(K[bz, j0 : j0 + valid_n, by, :], k_l1)
                                T.set_flag("MTE2", "MTE1", SIG_K_L1)
                                T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                                # K-split mma chain: c=0 init=True -> c=1
                                # init=False (ascending-K fp32 accumulation).
                                # ---- k-chunk c=0 (K columns 0:KC) ----
                                T.wait_flag("M", "MTE1", SIG_L0KC)  # pool chunk 0 released
                                T.copy(k_l1[:, 0:KC], l0b_pool[0, :, :], transpose=True)
                                T.set_flag("MTE1", "M", SIG_L0KC)
                                T.wait_flag("MTE1", "M", SIG_L0KC)
                                T.wait_flag("FIX", "M", SIG_L0C + side)  # l0c_s slot writable
                                T.mma(l0a_s[0, :, :], l0b_pool[0, :, :], l0c_s[side, :, :], init=True)
                                T.set_flag("M", "MTE1", SIG_L0KC)  # pool chunk 0 released
                                # ---- k-chunk c=1 (K columns KC:2*KC, chained) ----
                                T.wait_flag("M", "MTE1", SIG_L0KC + 1)
                                T.copy(k_l1[:, KC : 2 * KC], l0b_pool[1, :, :], transpose=True)
                                T.set_flag("MTE1", "M", SIG_L0KC + 1)
                                T.wait_flag("MTE1", "M", SIG_L0KC + 1)
                                T.mma(l0a_s[1, :, :], l0b_pool[1, :, :], l0c_s[side, :, :], init=False)
                                T.set_flag("M", "MTE1", SIG_L0KC + 1)
                                T.set_flag("M", "FIX", SIG_L0C + side)  # scores full block ready
                                T.set_flag("MTE1", "MTE2", SIG_K_L1)  # k_l1 read done
                                # L0C -> w1 (FIX full block)
                                T.wait_flag("M", "FIX", SIG_L0C + side)
                                T.copy(l0c_s[side, :, :], workspace_1[cid, i, :, :])
                                T.set_flag("FIX", "M", SIG_L0C + side)
                                if kv_g1 == eff_kv_iters - 1:
                                    # ---- Task end: the l0a_s return. By M-pipe
                                    # order this task's last GEMM1 mma has already
                                    # read l0a_s and GEMM2 never reads it, so the
                                    # next switch's head wait consumes the token:
                                    # one round per task. ----
                                    T.set_flag("M", "MTE1", SIG_L0AS)
                                    T.set_flag("M", "MTE1", SIG_L0AS + 1)
                                if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                                    T.set_cross_flag("FIX", SEM_WS1_C2V)
                                # walk advance (pure scalar, no DMA/flags)
                                kv_g1 += 1
                                if kv_g1 == eff_kv_iters:
                                    kv_g1 = 0
                                    tl_g1 += 1

                            # --- GEMM2 batch: consume w2 (P), produce PV -> w3 ---
                            T.wait_cross_flag(SEM_WS3_V2C)
                            tl_g2 = T.alloc_var("int32", init=tl_head)
                            kv_g2 = T.alloc_var("int32", init=kv_head)
                            for i in T.serial(batch_iters):
                                side = i % 2
                                task = my_start + tl_g2
                                bx = task % s_blocks
                                by = task // s_blocks % N
                                bz = task // s_blocks // N % B
                                s0 = bx * block_M
                                eff_kv_iters = (
                                    T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                                )
                                idx = kv_g2
                                j0 = idx * block_N
                                valid_n = T.min(block_N, S_kv - j0)
                                # V: GM -> L1
                                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                                T.copy(V[bz, j0 : j0 + valid_n, by, :], v_l1)
                                T.set_flag("MTE2", "MTE1", SIG_V_L1)
                                # P: w2 -> p_l1
                                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                                if i % CROSS_INTERVAL == 0:
                                    T.wait_cross_flag(SEM_WS2_V2C)
                                T.copy(workspace_2[cid, i, :, :], p_l1)
                                T.set_flag("MTE2", "MTE1", SIG_P_L1)
                                # P: L1 -> L0A (side ping-pong)
                                T.wait_flag("M", "MTE1", SIG_L0AO + side)
                                T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                                T.copy(p_l1, l0a_o[side, :, :])
                                T.set_flag("MTE1", "MTE2", SIG_P_L1)
                                T.set_flag("MTE1", "M", SIG_L0AO + side)
                                T.wait_flag("MTE2", "MTE1", SIG_V_L1)
                                # N-split mma plus a per-chunk column-segment FIX
                                # write. Each chunk accumulates into its own narrow
                                # l0c_o slot with init=True, so the chunks are
                                # column-independent and need no chaining.
                                # ---- n-chunk c=0 (V cols 0:NC -> w3 cols 0:NC) ----
                                T.wait_flag("M", "MTE1", SIG_L0KC)
                                T.copy(v_l1[:, 0:NC], l0b_pool[0, :, :])  # direct, no transpose
                                T.set_flag("MTE1", "M", SIG_L0KC)
                                T.wait_flag("MTE1", "M", SIG_L0KC)
                                T.wait_flag("MTE1", "M", SIG_L0AO + side)  # p ready
                                T.wait_flag("FIX", "M", SIG_L0C)  # narrow slot 0 writable
                                T.mma(l0a_o[side, :, :], l0b_pool[0, :, :], l0c_o[0, :, :], init=True)
                                T.set_flag("M", "MTE1", SIG_L0KC)
                                T.set_flag("M", "FIX", SIG_L0C)  # chunk 0 ready
                                T.wait_flag("M", "FIX", SIG_L0C)
                                T.copy(l0c_o[0, :, :], workspace_3[cid, i, :, 0:NC])  # colseg FIX
                                T.set_flag("FIX", "M", SIG_L0C)
                                # ---- n-chunk c=1 (V cols NC:2*NC -> w3 cols NC:2*NC) ----
                                T.wait_flag("M", "MTE1", SIG_L0KC + 1)
                                T.copy(v_l1[:, NC : 2 * NC], l0b_pool[1, :, :])
                                T.set_flag("MTE1", "M", SIG_L0KC + 1)
                                T.wait_flag("MTE1", "M", SIG_L0KC + 1)
                                T.wait_flag("FIX", "M", SIG_L0C + 1)  # narrow slot 1 writable
                                T.mma(l0a_o[side, :, :], l0b_pool[1, :, :], l0c_o[1, :, :], init=True)
                                T.set_flag("M", "MTE1", SIG_L0KC + 1)
                                T.set_flag("M", "FIX", SIG_L0C + 1)
                                T.set_flag("M", "MTE1", SIG_L0AO + side)  # l0a_o both chunks read
                                T.wait_flag("M", "FIX", SIG_L0C + 1)
                                T.copy(l0c_o[1, :, :], workspace_3[cid, i, :, NC : 2 * NC])
                                T.set_flag("FIX", "M", SIG_L0C + 1)
                                T.set_flag("MTE1", "MTE2", SIG_V_L1)  # v_l1 read done
                                if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                                    # after BOTH colseg FIX writes (FIX pipe is in-order)
                                    T.set_cross_flag("FIX", SEM_WS3_C2V)
                                # walk advance (pure scalar, no DMA/flags)
                                kv_g2 += 1
                                if kv_g2 == eff_kv_iters:
                                    kv_g2 = 0
                                    tl_g2 += 1

                            T.set_cross_flag("MTE2", SEM_WS2_C2V)
                            # batch-head state advance (GEMM2 walk ended at the next head)
                            tl_head = tl_g2
                            kv_head = kv_g2

                    # destroy: symmetric wait x12 for the tokens pre-set by init
                    T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                    T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                    T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                    T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                    T.wait_flag("M", "MTE1", SIG_L0KC)
                    T.wait_flag("M", "MTE1", SIG_L0KC + 1)
                    T.wait_flag("M", "MTE1", SIG_L0AO)
                    T.wait_flag("M", "MTE1", SIG_L0AO + 1)
                    T.wait_flag("M", "MTE1", SIG_L0AS)
                    T.wait_flag("M", "MTE1", SIG_L0AS + 1)
                    T.wait_flag("FIX", "M", SIG_L0C)
                    T.wait_flag("FIX", "M", SIG_L0C + 1)

                # ==================== Vector core (shared emitter) ====================
                with T.Scope("V"):
                    emit_v(
                        io_buf,
                        io3,
                        work_ub,
                        buf_2d,
                        rt_2d,
                        r_2d,
                        mask_2d,
                        col_pos,
                        row_thresh,
                        rr_ub,
                        acc_s_half,
                        acc_o,
                        out_half,
                        m_i,
                        m_i_prev,
                        sumexp,
                        r_factors,
                        sumexp_is,
                        Output,
                        workspace_1,
                        workspace_2,
                        workspace_3,
                        cid,
                        vid,
                        my_start,
                        my_count,
                        sig_io3,
                    )

        return main

    else:

        @T.prim_func
        def main(
            Q: T.Tensor([B, S, N, D], dtype),  # type: ignore
            K: T.Tensor([B, S_kv, N, D], dtype),  # type: ignore
            V: T.Tensor([B, S_kv, N, D], dtype),  # type: ignore
            Output: T.Tensor([B, S, N, D], dtype),  # type: ignore
            workspace_1: T.Tensor([core_num, NUM_STAGES, block_M, block_N], accum_dtype),  # type: ignore
            workspace_2: T.Tensor([core_num, NUM_STAGES, block_M, block_N], dtype),  # type: ignore
            workspace_3: T.Tensor([core_num, NUM_STAGES, block_M, D], accum_dtype),  # type: ignore
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                # ---- C-side L1 (no annotation: gemm_v0 handles L0 internally) ----
                q_l1 = T.alloc_L1([block_M, D], dtype)
                # K-side double slot, kv_db domain only. A 3D L1 buffer with no
                # layout annotation plus a runtime slice behaves exactly like a
                # serial sequence of 2D buffers, which is what double buffering
                # needs here.
                k_l1 = T.alloc_L1([2, block_N, D], dtype) if kv_db else T.alloc_L1([block_N, D], dtype)
                # V-side double slot, kv_db only; the same 3D no-annotation plus
                # runtime-slice mechanism as k_l1.
                v_l1 = T.alloc_L1([2, block_N, D], dtype) if kv_db else T.alloc_L1([block_N, D], dtype)
                p_l1 = T.alloc_L1([block_M, block_N], dtype)
                # ---- L0C single slots (gemm_v0 manages L0A/L0B by kL0Size) ----
                acc_s_l0c = T.alloc_L0C([block_M, block_N], accum_dtype)
                acc_o_l0c = T.alloc_L0C([block_M, D], accum_dtype)
                # ---- V-side UB: identical to Form A (emit_v is shared) ----
                io_buf = T.alloc_ub([2, half_M, block_N], accum_dtype)
                io3 = io_buf if io3_is_alias else T.alloc_ub([2, half_M, D], accum_dtype)
                work_ub = T.alloc_ub([half_M, block_N], accum_dtype)
                buf_2d = T.alloc_ub([half_M, block_N], accum_dtype)
                rt_2d = T.alloc_ub([half_M, block_N], accum_dtype)
                r_2d = buf_2d if block_N == D else T.alloc_ub([half_M, D], accum_dtype)
                mask_2d = T.alloc_ub([half_M * block_N // 8], "uint8")
                col_pos = T.alloc_ub([block_N], accum_dtype)
                row_thresh = T.alloc_ub([half_M], accum_dtype)
                rr_ub = T.alloc_ub([half_M, block_N], dtype)
                acc_s_half = T.alloc_ub([half_M, block_N], dtype)
                acc_o = T.alloc_ub([half_M, D], accum_dtype)
                out_half = T.alloc_ub([half_M, D], dtype)
                m_i = T.alloc_ub([half_M, 1], accum_dtype)
                m_i_prev = T.alloc_ub([half_M, 1], accum_dtype)
                sumexp = T.alloc_ub([half_M, 1], accum_dtype)
                r_factors = T.alloc_ub([NUM_STAGES, half_M, 1], accum_dtype)
                sumexp_is = T.alloc_ub([NUM_STAGES, half_M, 1], accum_dtype)

                my_start = cid * q_tasks + T.if_then_else(cid < r_tasks, cid, r_tasks)
                my_count = q_tasks + T.if_then_else(cid < r_tasks, 1, 0)

                # ================= Cube core (Form B: gemm_v0, barrier-free) =================
                with T.Scope("C"):
                    # init: pretend the workspace_2 slots are already released by
                    # the consumer, because the V side's first phase-1 batch waits
                    # on SEM_WS2_C2V. No SIG_* tokens are pre-set: this form uses
                    # no intra-core user flags at all.
                    T.set_cross_flag("MTE2", SEM_WS2_C2V)
                    if single_kv_block:
                        # ---- Task-batched single-KV-block C path ----
                        # One protocol round batches NUM_STAGES TASKS, since
                        # kv_iters == 1 means every task is exactly one KV block:
                        # slot i <-> task my_start + kb*NUM_STAGES + i. The per-slot
                        # ops are identical to the generic flow, and the V mirror
                        # derives num_tb/batch_tasks from the SAME my_start/my_count
                        # formulas so the pairing stays in lockstep.
                        num_tb = T.ceildiv(my_count, NUM_STAGES)
                        for kb in T.serial(num_tb):
                            _remaining_t = my_count - kb * NUM_STAGES
                            batch_tasks = T.if_then_else(_remaining_t < NUM_STAGES, _remaining_t, NUM_STAGES)
                            valid_n = min(block_N, S_kv)  # single block: idx == 0

                            # --- GEMM1 batch: per-task scores -> w1 slot i ---
                            T.wait_cross_flag(SEM_WS1_V2C)
                            for i in T.serial(batch_tasks):
                                task = my_start + kb * NUM_STAGES + i
                                bx = task % s_blocks
                                by = task // s_blocks % N
                                bz = task // s_blocks // N % B
                                s0 = bx * block_M
                                valid_m = T.min(block_M, S - s0)
                                # Barrier-free ordering. The gemm_v0 prologue
                                # orders prior MTE2 (Q/K loads) before its MTE1
                                # reads and prior FIX (workspace writes) before
                                # its mma; the epilogue orders MTE1 before later
                                # MTE2 (L1 slot reuse) and mma before later FIX
                                # (the L0C read). That covers every intra-core
                                # segment boundary, so no barrier_all is needed.
                                # Same-pipe ops are in-order by hardware; cross-
                                # core hazards stay on the semaphore protocol.
                                T.copy(Q[bz, s0 : s0 + valid_m, by, :], q_l1)
                                T.copy(K[bz, 0:valid_n, by, :], k_l1)
                                T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True, kL0Size=gemm1_kL0)
                                T.copy(acc_s_l0c, workspace_1[cid, i, :, :])
                                if (i + 1) % CROSS_INTERVAL == 0 or i == batch_tasks - 1:
                                    T.set_cross_flag("FIX", SEM_WS1_C2V)

                            # --- GEMM2 batch: per-task P (w2 slot i) @ V -> w3 slot i ---
                            T.wait_cross_flag(SEM_WS3_V2C)
                            for i in T.serial(batch_tasks):
                                task = my_start + kb * NUM_STAGES + i
                                by = task // s_blocks % N
                                bz = task // s_blocks // N % B
                                T.copy(V[bz, 0:valid_n, by, :], v_l1)
                                if i % CROSS_INTERVAL == 0:
                                    T.wait_cross_flag(SEM_WS2_V2C)
                                T.copy(workspace_2[cid, i, :, :], p_l1)
                                T.gemm_v0(p_l1, v_l1, acc_o_l0c, init=True)
                                T.copy(acc_o_l0c, workspace_3[cid, i, :, :])
                                if (i + 1) % CROSS_INTERVAL == 0 or i == batch_tasks - 1:
                                    T.set_cross_flag("FIX", SEM_WS3_C2V)

                            T.set_cross_flag("MTE2", SEM_WS2_C2V)
                    elif kv_db:
                        # ---- kv_db: dual-slot K/V prefetch, stock gemm_v0
                        # default-prologue sync. Every gemm_v0 call self-issues
                        # SetFlag/WaitFlag<MTE2_MTE1>(0): the wait drains the
                        # WHOLE MTE2 queue, so every operand enqueued before
                        # the call is L1-resident before the call's MTE1 reads.
                        # No user flag is needed, and none may be added here --
                        # a user SetFlag<MTE2_MTE1>(0) would leave a dangling
                        # token that a later default prologue wait could
                        # consume stale, starting the reads before the copies
                        # retired. The pre-call prefetch positions below are
                        # kept as-is; under the full drain a prefetch simply
                        # retires inside the consuming call's prologue (it
                        # joins that call's critical path instead of streaming
                        # through it).
                        #
                        # K slots use side = idx % 2 with idx = k*NUM_STAGES + i.
                        # NUM_STAGES is even, so a round/task head slot is always
                        # side 0. Issue sites, chosen so that
                        # #prefetch == #consumed == sum(eff_kv_iters) -- no slot
                        # is loaded twice or read before it is loaded:
                        #   task-0 head (t == 0): Q(0), then K_0 into slot 0.
                        #   K pre-call prefetch, iteration idx: K_{idx+1} into
                        #      slot (idx+1)%2. The guard idx+1 < eff_kv_iters is
                        #      GLOBAL, so it crosses round boundaries and no
                        #      separate round-end site is needed.
                        #   task-end priming (t+1 < my_count): Q(t+1) THEN K'_0,
                        #      in that order, so both operands of GEMM1'(0) sit
                        #      in the MTE2 queue ahead of the next task's first
                        #      call, whose drain makes them resident; slot 0.
                        # GEMM1 consumes k_l1[idx % 2, :, :] as a runtime slice
                        # at a single call site -- the same pattern GEMM2 uses
                        # for its V operand.
                        #
                        # V side: v_l1 has two slots, so V(idx+1) -- slot
                        # (idx+1)%2, last read by gemm2(idx-1) -- is issuable at
                        # iteration idx's pre-call position: after gemm2(idx-1)
                        # returned (its epilogue MTE1_MTE2 release frees the
                        # slot). Lookahead is exactly one slot; V(idx+2) would
                        # race gemm2(idx)'s own slot. V sites:
                        #   round-head V priming at the GEMM1-batch tail
                        #      (i == batch_iters-1): the round's head V into
                        #      STATIC slot 0 (NUM_STAGES even =>
                        #      (k*NUM_STAGES)%2 == 0), so needClear is true --
                        #      the base case of the finite-stale induction
                        #      below. Placed after ALL of this round's K loads,
                        #      keeping issue order == consume order.
                        #   V pre-call prefetch, GEMM2 iteration idx: V_{idx+1}
                        #      into slot (idx+1)%2. The runtime offset makes
                        #      needClear false, so tail rows may retain stale
                        #      data; that is safe by induction, because every
                        #      slot's FIRST load is a full tile (slot 0 via the
                        #      round-head clear, slot 1 via V(idx0+1) with
                        #      valid_n == block_N since S_kv > 256 here) and the
                        #      matching P columns are exactly 0, so
                        #      0 * finite = 0 with no NaN path. The guard
                        #      i+1 < batch_iters is LOCAL: round-head V comes
                        #      from the GEMM1-batch tail, never twice.
                        #   GEMM2 iteration idx: the SEM_WS2 wait and the P copy
                        #      keep their original positions; the call's default
                        #      drain covers V(idx) and P(idx), with
                        #      B = v_l1[idx%2,:,:] as a runtime slice.
                        # Slot release needs no extra flags: the gemm epilogue's
                        # SetFlag<MTE1_MTE2> orders every MTE2 issued after a
                        # call behind that call's MTE1 reads, so the K(idx+1)
                        # and V(idx+1) prefetches -- which overwrite the slot
                        # gemm(idx-1) read and are issued after that call
                        # returned -- are correctly ordered. For Q(t+1) at the
                        # task end, q_l1 was released by task t's last GEMM1
                        # epilogue, and the GEMM1 phase precedes the GEMM2
                        # phase. Semaphore protocol positions are unchanged.
                        # The task-end priming sits after the task's last
                        # set_cross_flag("MTE2", SEM_WS2_C2V), where the MTE2
                        # queue is empty (the last P copy is 8 KB and the
                        # round's K/V loads retired during the GEMM2 phase), so
                        # w2 release is not delayed.
                        for t in T.serial(my_count):
                            task = my_start + t
                            bx = task % s_blocks
                            by = task // s_blocks % N
                            bz = task // s_blocks // N % B
                            s0 = bx * block_M
                            valid_m = T.min(block_M, S - s0)
                            eff_kv_iters = (
                                T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                            )
                            eff_num_outer = T.ceildiv(eff_kv_iters, NUM_STAGES)

                            if t == 0:
                                # Task-0 head priming: Q and K_0 enter the MTE2
                                # queue before the first GEMM1(0) call, whose
                                # default prologue drain makes both resident.
                                # For t > 0, Q(t) and K_0(t) were already primed
                                # at task t-1's end, hence the t == 0 guard: no
                                # double load.
                                hvalid = T.min(block_N, S_kv)
                                T.copy(Q[bz, s0 : s0 + valid_m, by, :], q_l1)
                                T.copy(K[bz, 0:hvalid, by, :], k_l1[0, :, :])

                            for k in T.serial(eff_num_outer):
                                _remaining = eff_kv_iters - k * NUM_STAGES
                                batch_iters = T.if_then_else(_remaining < NUM_STAGES, _remaining, NUM_STAGES)

                                # --- GEMM1 batch: scores -> w1 ---
                                T.wait_cross_flag(SEM_WS1_V2C)
                                for i in T.serial(batch_iters):
                                    idx = k * NUM_STAGES + i
                                    # PRE-CALL prefetch of K(idx+1). Position
                                    # unchanged; under the default full drain
                                    # the copy retires inside gemm1(idx)'s
                                    # prologue, and gemm1(idx+1) finds its
                                    # operand slot already resident.
                                    if idx + 1 < eff_kv_iters:
                                        nidx = idx + 1
                                        nj0 = nidx * block_N
                                        nvalid = T.min(block_N, S_kv - nj0)
                                        T.copy(K[bz, nj0 : nj0 + nvalid, by, :], k_l1[nidx % 2, :, :])
                                    # GEMM1 consume: single call site with the
                                    # K slot as a runtime slice. Both operands
                                    # Q and K(idx) were enqueued before this
                                    # call, so its default prologue drain makes
                                    # them resident.
                                    T.gemm_v0(q_l1, k_l1[idx % 2, :, :], acc_s_l0c, transpose_B=True, init=True, kL0Size=gemm1_kL0)
                                    T.copy(acc_s_l0c, workspace_1[cid, i, :, :])
                                    if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                                        T.set_cross_flag("FIX", SEM_WS1_C2V)
                                    if i == batch_iters - 1:
                                        # Round-head V priming at the GEMM1
                                        # batch tail. FIFO order puts it after
                                        # every K issue of this round, and
                                        # GEMM2 iteration 0 consumes it.
                                        # STATIC slot 0 (NUM_STAGES even =>
                                        # (k*NUM_STAGES)%2 == 0), so needClear
                                        # is true: the stale-free base case of
                                        # the induction.
                                        vj0 = k * NUM_STAGES * block_N
                                        vvalid = T.min(block_N, S_kv - vj0)
                                        T.copy(V[bz, vj0 : vj0 + vvalid, by, :], v_l1[0, :, :])

                                # --- GEMM2 batch: consume w2 (P), produce PV -> w3 ---
                                T.wait_cross_flag(SEM_WS3_V2C)
                                for i in T.serial(batch_iters):
                                    idx = k * NUM_STAGES + i
                                    if i % CROSS_INTERVAL == 0:
                                        T.wait_cross_flag(SEM_WS2_V2C)
                                    # The P copy stays at its original
                                    # position; the call's default drain covers
                                    # both V(idx) and P(idx).
                                    T.copy(workspace_2[cid, i, :, :], p_l1)
                                    # PRE-CALL V(idx+1) prefetch. Dual slot
                                    # (idx+1)%2 was released by gemm2(idx-1)'s
                                    # return; under the default full drain the
                                    # copy retires inside gemm2(idx)'s prologue.
                                    # The guard is LOCAL because the round-head
                                    # V comes from the GEMM1-batch tail and
                                    # must not load twice.
                                    if i + 1 < batch_iters:
                                        nvidx = idx + 1
                                        nvj0 = nvidx * block_N
                                        nvvalid = T.min(block_N, S_kv - nvj0)
                                        T.copy(V[bz, nvj0 : nvj0 + nvvalid, by, :], v_l1[nvidx % 2, :, :])
                                    T.gemm_v0(p_l1, v_l1[idx % 2, :, :], acc_o_l0c, init=True)
                                    T.copy(acc_o_l0c, workspace_3[cid, i, :, :])
                                    if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                                        T.set_cross_flag("FIX", SEM_WS3_C2V)

                                T.set_cross_flag("MTE2", SEM_WS2_C2V)

                            if t + 1 < my_count:
                                # Task-end priming, placed after the task's last
                                # set of WS2_C2V. FIFO order mandates Q(t+1) BEFORE
                                # K'_0 so that both GEMM1'(0) operands sit in the
                                # MTE2 queue ahead of the next task's first call,
                                # whose default drain makes them resident. The
                                # bz'/by'/extent values are re-derived from
                                # task+1 by pure scalar arithmetic.
                                ntask = task + 1
                                nbx = ntask % s_blocks
                                nby = ntask // s_blocks % N
                                nbz = ntask // s_blocks // N % B
                                ns0 = nbx * block_M
                                nvalid_m = T.min(block_M, S - ns0)
                                tvalid = T.min(block_N, S_kv)
                                T.copy(Q[nbz, ns0 : ns0 + nvalid_m, nby, :], q_l1)
                                T.copy(K[nbz, 0:tvalid, nby, :], k_l1[0, :, :])
                    else:
                        for t in T.serial(my_count):
                            task = my_start + t
                            bx = task % s_blocks
                            by = task // s_blocks % N
                            bz = task // s_blocks // N % B
                            s0 = bx * block_M
                            valid_m = T.min(block_M, S - s0)
                            eff_kv_iters = (
                                T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1 if is_causal else kv_iters
                            )
                            eff_num_outer = T.ceildiv(eff_kv_iters, NUM_STAGES)

                            # Barrier-free ordering: the first gemm_v0 prologue of
                            # the task drains this MTE2 Q load before its own MTE1
                            # reads, and q_l1 reuse across tasks is ordered by the
                            # previous gemm's epilogue (MTE1 -> MTE2).
                            T.copy(Q[bz, s0 : s0 + valid_m, by, :], q_l1)

                            for k in T.serial(eff_num_outer):
                                _remaining = eff_kv_iters - k * NUM_STAGES
                                batch_iters = T.if_then_else(_remaining < NUM_STAGES, _remaining, NUM_STAGES)

                                # --- GEMM1 batch: scores -> w1 ---
                                T.wait_cross_flag(SEM_WS1_V2C)
                                for i in T.serial(batch_iters):
                                    idx = k * NUM_STAGES + i
                                    j0 = idx * block_N
                                    valid_n = T.min(block_N, S_kv - j0)
                                    T.copy(K[bz, j0 : j0 + valid_n, by, :], k_l1)
                                    T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True, kL0Size=gemm1_kL0)
                                    T.copy(acc_s_l0c, workspace_1[cid, i, :, :])
                                    if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                                        T.set_cross_flag("FIX", SEM_WS1_C2V)

                                # --- GEMM2 batch: consume w2 (P), produce PV -> w3 ---
                                T.wait_cross_flag(SEM_WS3_V2C)
                                for i in T.serial(batch_iters):
                                    idx = k * NUM_STAGES + i
                                    j0 = idx * block_N
                                    valid_n = T.min(block_N, S_kv - j0)
                                    T.copy(V[bz, j0 : j0 + valid_n, by, :], v_l1)
                                    if i % CROSS_INTERVAL == 0:
                                        T.wait_cross_flag(SEM_WS2_V2C)
                                    T.copy(workspace_2[cid, i, :, :], p_l1)
                                    T.gemm_v0(p_l1, v_l1, acc_o_l0c, init=True)
                                    T.copy(acc_o_l0c, workspace_3[cid, i, :, :])
                                    if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                                        T.set_cross_flag("FIX", SEM_WS3_C2V)

                                T.set_cross_flag("MTE2", SEM_WS2_C2V)

                # ==================== Vector core (shared emitter) ====================
                with T.Scope("V"):
                    emit_v(
                        io_buf,
                        io3,
                        work_ub,
                        buf_2d,
                        rt_2d,
                        r_2d,
                        mask_2d,
                        col_pos,
                        row_thresh,
                        rr_ub,
                        acc_s_half,
                        acc_o,
                        out_half,
                        m_i,
                        m_i_prev,
                        sumexp,
                        r_factors,
                        sumexp_is,
                        Output,
                        workspace_1,
                        workspace_2,
                        workspace_3,
                        cid,
                        vid,
                        my_start,
                        my_count,
                        sig_io3,
                    )

        return main


def mha(query, key, value, scaleValue=-1.0, is_causal=False):
    """Host wrapper: metadata validation + JIT dispatch only (zero tensor ops).

    Args:
        query: [B, S, N, D] float16/bfloat16, contiguous, finite values.
        key: [B, S_kv, N, D] same dtype; B/N/D must match query.
        value: [B, S_kv, N, D] same as key.
        scaleValue: <= 0 -> auto 1/sqrt(D); > 0 -> used as-is (JIT constant).
        is_causal: bottom-right aligned causal mask; requires S <= S_kv.

    Returns:
        [B, S, N, D] output, same dtype as query, contiguous (framework-allocated).
    """
    assert query.dtype == key.dtype == value.dtype, "Q/K/V dtypes must match"
    assert query.dtype in (torch.float16, torch.bfloat16), f"unsupported dtype {query.dtype}; only float16/bfloat16"
    assert query.is_contiguous() and key.is_contiguous() and value.is_contiguous(), "Q/K/V must be contiguous BSHD tensors"
    B, S, N, D = query.shape
    S_kv = key.shape[1]
    assert tuple(key.shape) == (B, S_kv, N, D), "key shape must be (B, S_kv, N, D)"
    assert tuple(value.shape) == (B, S_kv, N, D), "value shape must equal key shape"
    assert D in (64, 128, 256), f"D must be in {{64, 128, 256}}, got {D}"
    assert 1 <= B <= 128, f"B out of range [1, 128]: {B}"
    assert 1 <= S <= 2048, f"S out of range [1, 2048]: {S}"
    assert 1 <= S_kv <= 4096, f"S_kv out of range [1, 4096]: {S_kv}"
    assert 1 <= N <= 64, f"N out of range [1, 64]: {N}"
    if is_causal:
        assert S_kv >= S, "is_causal=True requires S <= S_kv (else all-masked rows)"
    kernel = mha_kernel(
        B=B,
        S=S,
        S_kv=S_kv,
        N=N,
        D=D,
        dtype=str(query.dtype).removeprefix("torch."),
        scaleValue=float(scaleValue),
        is_causal=bool(is_causal),
    )
    return kernel(query, key, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-checking MHA forward example (BSHD layout).")
    parser.add_argument("--B", type=int, default=2, help="batch size, 1..128")
    parser.add_argument("--S", type=int, default=128, help="query length, 1..2048; S <= 16 selects the decode path")
    parser.add_argument("--S-kv", dest="S_kv", type=int, default=128, help="key/value length, 1..4096")
    parser.add_argument("--N", type=int, default=8, help="head count, 1..64")
    parser.add_argument("--D", type=int, default=64, choices=(64, 128, 256), help="head dim")
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16"), help="Q/K/V dtype")
    parser.add_argument("--scale", type=float, default=-1.0, help="softmax scale; <= 0 selects 1/sqrt(D)")
    parser.add_argument("--causal", action="store_true", help="apply the bottom-right aligned causal mask")
    args = parser.parse_args()

    def ref_mha(q, k, v, scale, causal):
        """Reference on the kernel's own rounding grid: QK^T -> *scale -> softmax -> @V,
        computed in the input dtype over [B, N, S, S_kv] scores."""
        S, S_kv = q.shape[1], k.shape[1]
        scores = torch.matmul(q.transpose(1, 2), k.transpose(1, 2).transpose(-2, -1)) * scale
        if causal:
            i = torch.arange(S).unsqueeze(-1)
            j = torch.arange(S_kv).unsqueeze(0)
            scores = scores.masked_fill(j > i + (S_kv - S), float("-inf"))
        # A fully masked row would leave softmax with 0/0; force its weights to 0.
        row_max = scores.max(dim=-1, keepdim=True).values
        weights = torch.nn.functional.softmax(scores, dim=-1)
        weights = torch.where(torch.isinf(row_max) & (row_max < 0), torch.zeros_like(weights), weights)
        return torch.matmul(weights, v.transpose(1, 2)).transpose(1, 2).contiguous()

    scale = args.scale if args.scale > 0 else 1.0 / args.D**0.5
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    query = torch.randn(args.B, args.S, args.N, args.D, dtype=dtype)
    key = torch.randn(args.B, args.S_kv, args.N, args.D, dtype=dtype)
    value = torch.randn(args.B, args.S_kv, args.N, args.D, dtype=dtype)

    out = mha(query.to("npu"), key.to("npu"), value.to("npu"), scaleValue=args.scale, is_causal=args.causal)
    torch.npu.synchronize()
    ref = ref_mha(query, key, value, scale, args.causal)

    # The mixed-tolerance gate test_mha.py uses: at least 99% of the elements
    # inside atol + rtol*|ref|, and no element past the dtype's absolute ceiling.
    atol, rtol, max_abs_limit = {"float16": (2**-14, 2**-9, 1e-1), "bfloat16": (2**-10, 2**-6, 1e0)}[args.dtype]
    out_f, ref_f = out.detach().cpu().float(), ref.float()
    abs_err = (out_f - ref_f).abs()
    matched = (abs_err <= atol + rtol * ref_f.abs()).float().mean().item()
    max_abs = abs_err.max().item()

    print(f"MHA B={args.B} S={args.S} S_kv={args.S_kv} N={args.N} D={args.D} dtype={args.dtype} causal={args.causal} scale={scale:.6f}")
    print(f"matched={matched:.5f} (gate >= 0.99)   max_abs_err={max_abs:.3e} (gate <= {max_abs_limit:g})")
    assert matched >= 0.99, f"only {matched:.5f} of elements are within tolerance (gate 0.99)"
    assert max_abs <= max_abs_limit, f"max_abs_err {max_abs:.3e} exceeds the {max_abs_limit:g} ceiling"
    print("Test Passed!")
