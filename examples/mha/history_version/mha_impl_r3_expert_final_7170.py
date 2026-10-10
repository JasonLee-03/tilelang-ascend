"""MHA (Multi-Head Attention) forward, BSHD layout, Expert mode (DESIGN rev3).

y = softmax(Q @ K^T * s) @ V,   s = scaleValue if scaleValue > 0 else 1/sqrt(D)
is_causal: scores[..., i, j] = -inf for j > i + (S_kv - S)  (bottom-right aligned)
All-masked rows (only reachable out-of-domain) yield 0 via the sumexp + 1e-30 guard.

rev3 Expert implementation (2026-09-19; design = DESIGN.md rev3, semantic
baseline = history_version/mha_impl_developer_final_6505.py, cann-bench 65.05):
- Expert frame: `T.Kernel(core_num, is_npu=True) as (cid, vid)` — NO threads
  parameter (§2.2-b/§8.6-1); vid extent 2 is the structural blockIdx.y binding
  (cv_1_2). Four pass_configs all False (§4.7): manual T.Scope("C"/"V"),
  manual set_flag/wait_flag + set_cross_flag/wait_cross_flag, explicit
  alloc_L1/alloc_ub/alloc_L0A/L0B/L0C, explicit GM workspace x3
  (workspace_idx=[4,5,6], [cid, stage] private indexing).
- Dual C-scope forms (host predicate dispatch, §5.2/§6.1), V-scope shared
  via a @T.macro emitter (two self-contained prim_funcs built by a Python-level
  branch — TVMScript parse-time scoping, rev2-A1 lesson; plain-Python emitters
  cannot host T.serial frames, @T.macro can — probed):
    Form A (prefill S>16 & D<=128): T.mma + L0A/L0B/L0C double slots
      (side=i%2 manual ping-pong) + ZN/NZ annotate_layout + fine-grained
      pipe-pair flags (mha_sink_fwd_bhsd GEMM1/GEMM2 pattern).
      P2 (D==bN==128): GEMM2 shares the GEMM1 L0 pool (same shapes,
      mha_sink isomorph); P1 (D=64): separate l0a_o/l0b_o/l0c_o pool.
      single_kv_block: task-batched like Form B (#R5-1 session-5) — see the
      NUM_STAGES bullet below. PLUS (#R5-2 session-6): O direct write —
      GEMM2's FIX copies l0c_o straight into Output[bz, s0:s0+valid_m, by, :]
      (runtime row slice; gate probe 8/8 int-exact BITEXACT), deleting the
      w3 GM round-trip, the V phase-2 acc_o pass and the WS3 SEM pair; the
      fixpipe fp32->dtype RNE cast is bit-identical to the removed V-side
      vcast (#R3-3 probe 11/11). Form B keeps w3 + phase-2 (parse-time
      `if not form_a:` gate -> Form B IR byte-identical).
    Form B (decode S<=16 all D + prefill D=256): T.gemm_v0 (internal L0
      management, kL0Size three-tier inherited: 32/64/128), barrier-free
      segment ordering (#R4-1 session-4: gemm_v0's self-balanced
      prologue/epilogue pipe flags — MTE2_MTE1/FIX_M in, MTE1_MTE2/M_FIX
      out, common.h L1305-L1371 — cover every segment-boundary hazard; the
      rev3 barrier_all coarse sync was the redundant-global-sync
      antipattern: AIC pipe busy-sum ~95% with zero overlap, msprof op
      evidence case9).
- NUM_STAGES=8 batched CV pipeline: outer T.serial(eff_num_outer) + inner
  T.serial(batch_iters) runtime bounds, identical formulas on C/V scopes
  (§7.2-5 flag-pairing hard constraint); CROSS_INTERVAL=1 cross-core
  semaphore batching (gqa_fwd_varlen iter3 values). T.Pipelined is BANNED
  (§2.2-d/R9 guardrail); manual side=i%2 ping-pong expresses the pipeline.
  Batch dimension: kv-block when kv_iters>1; TASK when single_kv_block
  (kv_iters==1 would otherwise collapse batch_iters to 1 and leave the
  io_buf[2]/L0 double slots with nothing to ping-pong — every task ran a
  strictly serial C-QK -> V-softmax -> C-PV -> V-epi chain). #R3-1 added the
  task batch for Form B; #R5-1 (session-5) added the Form A C-scope mirror and
  dropped the `not form_a` half of the emit_v dispatch gate, so BOTH forms now
  batch. Form A slot-level deltas vs its generic body: the Q GM->L1 load moves
  inside the slot loop and q_l1->l0a_s[side] is issued every slot (Q is
  task-dependent), with the SIG_Q_L1 return moved to right after that single
  MTE1 read (per-slot token balance identical to the generic per-task one).
- vid row split: each AIV handles half_M = block_M//2 rows; writeback
  rows_vid = min(half_M, max(0, valid_m - vid*half_M)) with a runtime
  `if rows_vid > 0` guard (§5.4). Deferred-rescale batched softmax
  (r_factors stored in phase 1, exp in phase 2 — bit-identical to the
  per-iteration Developer order, §9.3.3-6).
- Precision red lines inherited verbatim (§3.2.1 mapping, 12/12 items):
  score double-rounding golden grid (w1 fp32 mainline -> V-side r16(qk) ->
  *s fp32 -> r16(scores)), single-KV-block pre-normalization specialization
  (kv_iters==1), sumexp+1e-30 all-masked-row guard, bottom-right-aligned
  pre-exp -inf mask chain (2D compare/select + row_thresh task-level hoist +
  fully-visible runtime skip), causal KV runtime trim (eff_kv_iters), Fixed
  Core (core_num device query + balanced task loop), bM/bN/gemm1_kL0
  three-tier system (Developer session-1/2 measured hard-limit solutions),
  BSHD row-stride DMA (zero host tensor ops, §4.8), valid_m/valid_n runtime
  tail extents with need_clear zeroing.
- Implementation-level buffer adaptations vs the §4.3 table (shape algebra
  forced, semantics unchanged; recorded in debug_log.md rev3-attempt1):
    io_buf3 [2,hM,D] fp32: dedicated w3 staging — ws3 vid-half slices are
      [hM,D] while io_buf slots are [hM,bN]; D!=bN in 5/6 config classes.
      Aliased to io_buf (same flag ids, token-balanced) when D==bN (P2).
    r_2d [hM,D] fp32: phase-2 rescale / final-div broadcast scratch on the
      acc_o column shape (Developer r_2d analog); buf_2d [hM,bN] cannot
      serve when D!=bN. Aliased to buf_2d when D==bN (both pure-V scratch,
      sequential V-pipe windows — §8.6-2 applies to cross-core copy
      destinations only).
    rr_ub [hM,bN] dtype: double-rounding round-trip scratch, kept SEPARATE
      from acc_s_half — acc_s_half is the MTE3 cross-core copy source under
      the SIG_S_HALF ownership loop; using it as round-trip scratch outside
      the loop would race the previous iteration's MTE3 read (§8.6-2 alias
      ban).
- Virtual-channel ban (§8.6-3/R12): no T.copy(L0C->UB) / T.copy(UB->L1)
  anywhere; CV exchange is the explicit 3-workspace GM path only (the sole
  physical route under this framework, §8.2 source evidence).
- Probes (scratch, /tmp/opencode/rev3_probes/): R10 Expert BSHD family
  16/16 bit-exact (Form A annotate ZN/NZ + mma + transpose + fine flags /
  Form B gemm_v0 + barrier_all; S=1/30/64/90/128 + divisible; need_clear
  zeros via host ws readback; vid-half writeback + rows_vid=0 guard via
  padded-sentinel Out; cross-task ws reuse; transpose-sensitive identity K);
  R15 mask-chain V-scope isolation — see debug_log.md.
"""

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

NUM_STAGES = 8  # CV pipeline depth (even constraint asserted; Stage-3 tunable 4-16)
CROSS_INTERVAL = 1  # cross-core sync interval (gqa iter3 verified; Stage-3 tunable 2-4)


def _cube_core_num() -> int:
    """Physical cube-core count (HOST METADATA query, zero tensor data touched;
    precedent: examples/deepseek_v4/lightning_indexer.py:14-21)."""
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
    assert NUM_STAGES % 2 == 0, "NUM_STAGES must be even (io_buf/L0 double-slot rotation, §8.6-7)"
    # ---- Host-side pure-Python config derivation (no tensor data touched, §4.8) ----
    sm_scale = scaleValue if scaleValue > 0 else (1.0 / D) ** 0.5
    # Tile three-tier system: verbatim inheritance of the Developer final state
    # (mha.py:120-179 probe-verified hard-limit solutions; see its docstring and
    # perf_tuning/optimization_log.md for the full evidence index).
    block_M = 16 if S <= 16 else (32 if D == 256 else 64)
    if S <= 16:
        # bN=512 (D<=128, S_kv>256): GEMM1 transpose_B B-tile 512*kL0(32)*2B =
        # 32768 == the 4-step ping-pong L0B budget; D=256 excluded by L1 (536KB).
        block_N = 512 if (D <= 128 and S_kv > 256) else (256 if S_kv > 128 else 128)
    else:
        block_N = 128
    gemm1_kL0 = (
        32
        if (S <= 16 and D <= 128 and S_kv > 256)  # bN=512: 512*32*2 = 32768 <= 32KB
        else (64 if (S <= 16 and D == 256 and S_kv > 128) else 128)  # bN=256 D=256 / rest
    )
    # ---- Expert additions (§5.2) ----
    form_a = (S > 16) and (D <= 128)  # Form A: mma full pipeline; else Form B: gemm_v0
    half_M = block_M // 2  # vid half rows (structural cv_1_2: 8/16/32)
    causal_offset = S_kv - S  # bottom-right alignment (>= 0 when causal, asserted)
    need_bound_mask = S_kv % block_N != 0  # tail-column clamp (compile-time)
    need_mask = is_causal or need_bound_mask
    # Unified threshold offset: causal -> bottom-right offset; non-causal -> big
    # constant so T.min(row + offset_eff, S_kv-1) collapses to S_kv-1.
    offset_eff = causal_offset if is_causal else 2**30
    s_blocks = (S + block_M - 1) // block_M
    kv_iters = (S_kv + block_N - 1) // block_N  # compile-time constant
    single_kv_block = kv_iters == 1  # i.e. S_kv <= block_N (predicate-move inherited)
    block_num = B * N * s_blocks
    core_num = min(block_num, _cube_core_num())  # Fixed Core (inherited #5)
    q_tasks = block_num // core_num  # base tasks per core
    r_tasks = block_num % core_num  # first r_tasks cores take one extra

    # ---- Cross-core semaphore IDs (AIC <-> 2xAIV group, mode=2; §7.2-1) ----
    SEM_WS1_C2V = 0  # w1 scores ready: Cube -> Vector
    SEM_WS1_V2C = 1  # w1 consumed: Vector -> Cube
    SEM_WS2_V2C = 2  # w2 softmax P ready: Vector -> Cube
    SEM_WS2_C2V = 3  # w2 consumed: Cube -> Vector
    SEM_WS3_C2V = 4  # w3 PV ready: Cube -> Vector
    SEM_WS3_V2C = 5  # w3 consumed: Vector -> Cube

    # ---- C-side intra-core event IDs (per directed pipe pair; Form A; §7.2-2) ----
    # MTE2 <-> MTE1
    SIG_K_L1 = 0
    SIG_P_L1 = 1
    SIG_V_L1 = 2
    SIG_Q_L1 = 3
    # MTE1 <-> M
    SIG_L0AB = 0  # double slots 0/1
    # M <-> FIX
    SIG_L0C = 0  # double slots 0/1

    # ---- V-side intra-core event IDs (per AIV; §7.2-3) ----
    # MTE2 <-> V
    SIG_IO_UB = 0  # io_buf (w1) double slots 0/1
    SIG_IO3 = 2  # io_buf3 (w3) double slots 2/3 (only when D != block_N)
    # MTE3 <-> V
    SIG_S_HALF = 0  # acc_s_half / out_half staging ownership

    # w3 staging + rescale scratch: shape-forced adaptation (§4.3 note; docstring).
    # D == block_N -> alias io_buf / buf_2d (same shapes; pure sequential reuse,
    # token-balanced with shared flag ids); D != block_N -> dedicated buffers.
    io3_is_alias = block_N == D
    sig_io3 = SIG_IO_UB if io3_is_alias else SIG_IO3

    # ==================== Shared V-scope emitter (@T.macro; §6.1) ====================
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
        # init (§7.2-4, mha_sink:379-386 isomorph): pretend workspaces released
        T.set_cross_flag("MTE2", SEM_WS1_V2C)
        T.set_cross_flag("MTE2", SEM_WS3_V2C)
        T.set_flag("V", "MTE2", SIG_IO_UB)
        T.set_flag("V", "MTE2", SIG_IO_UB + 1)
        if not io3_is_alias:
            T.set_flag("V", "MTE2", sig_io3_base)
            T.set_flag("V", "MTE2", sig_io3_base + 1)
        T.set_flag("MTE3", "V", SIG_S_HALF)

        if single_kv_block:
            # ---- #R3-1/#R5-1 task-batched single-KV-block V path (Form A + B) ----
            # Slot i <-> task my_start + kb*NUM_STAGES + i; per-slot op sequence
            # identical to the generic per-task flow (bit-exact); num_tb/
            # batch_tasks from the SAME formulas as the C mirror (§7.2-5).
            # #R5-1: the `and not form_a` gate is dropped — [IMPL-#R3-1]④ kept
            # Form A on the generic path ONLY because the C-side batch mirror did
            # not exist yet (protocol mismatch -> deadlock), not because of a
            # measured-benefit verdict (final_report.md:239 was predictive). The
            # Form A C mirror now exists (the `if single_kv_block:` branch of the
            # Form A C-scope task loop), so both forms share this path.
            # `single_kv_block and not form_a` == `single_kv_block` whenever
            # form_a is False => Form B IR is provably unchanged.
            num_tb = T.ceildiv(my_count, NUM_STAGES)
            for kb in T.serial(num_tb):
                _remaining_t = my_count - kb * NUM_STAGES
                batch_tasks = T.if_then_else(_remaining_t < NUM_STAGES, _remaining_t, NUM_STAGES)

                # ---- Phase 1 batch: per-task softmax -> w2 slots ----
                T.wait_cross_flag(SEM_WS2_C2V)
                for i in T.serial(batch_tasks):
                    task = my_start + kb * NUM_STAGES + i
                    bx = task % s_blocks
                    s0 = bx * block_M  # loop-body scope (rev2-A1: cross-block defs banned)
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

                    T.copy(m_i, m_i_prev)  # history max BEFORE reduce (§10.2 lesson)
                    # golden score-grid alignment (§3.2.1-⑦): identical op order
                    T.copy(work_ub, rr_ub)  # r16(qk)
                    T.copy(rr_ub, work_ub)
                    T.tile.mul(work_ub, work_ub, sm_scale)  # * s in fp32
                    T.copy(work_ub, rr_ub)  # r16(scores) = golden grid
                    T.copy(rr_ub, work_ub)

                    if need_mask:  # noqa: SIM102 - parse-time outer / runtime inner (rev2-A1)
                        # single block => idx == 0; s0-block-uniform -> both vids
                        # take the same branch (flag pairing safe; pure-V segment).
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

                    # pre-normalization specialization (§3.2.1-⑤): P_hat before
                    # the dtype rounding = golden one-shot softmax grid.
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

                # ---- Phase 2 batch: per-task w3 slot -> acc_o -> writeback.
                # Form B ONLY (#R5-2 session-6): Form A single-block C-scope now
                # FIX-writes Output directly from l0c_o (O = P_hat @ V is final
                # under the pre-normalization specialization), so this phase and
                # the WS3 SEM round are deleted there. `not form_a` is a
                # parse-time Python bool (closure) -> Form B IR byte-identical
                # (same mechanism as `if not io3_is_alias:` in init above).
                # V-init's SEM_WS3_V2C pre-set token stays: dangling-but-
                # harmless exactly like the by-design dangling WS1_V2C +1
                # (cross-core semaphore state is per-launch; C never waits).
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

        else:
            for t in T.serial(my_count):
                task = my_start + t
                bx = task % s_blocks
                by = task // s_blocks % N
                bz = task // s_blocks // N % B
                s0 = bx * block_M
                valid_m = T.min(block_M, S - s0)
                # Causal KV runtime trim — MUST match the C-scope formula verbatim
                # (§7.2-5/R11 flag-pairing hard constraint; mha_sink:394-400).
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

                        T.copy(m_i, m_i_prev)  # history max BEFORE reduce (§10.2 lesson)
                        # golden score-grid alignment (§3.2.1-⑦): round16(qk) -> *s fp32
                        # -> round16(scores); chain order = Developer mha.py:304-308.
                        # rr_ub = dedicated round-trip scratch (NEVER acc_s_half —
                        # MTE3 cross-core source, §8.6-2 alias ban).
                        T.copy(work_ub, rr_ub)  # r16(qk)
                        T.copy(rr_ub, work_ub)
                        T.tile.mul(work_ub, work_ub, sm_scale)  # * s in fp32
                        T.copy(work_ub, rr_ub)  # r16(scores) = golden grid
                        T.copy(rr_ub, work_ub)

                        if need_mask:  # noqa: SIM102 — outer if is PARSE-TIME (Python
                            # bool variant switch, §5.6); inner if is a RUNTIME TIR
                            # condition. Combining them into `need_mask and <TIR expr>`
                            # would hand a BoolOp AST to the TVMScript parser (unverified)
                            # — the two-level structure is intentional (rev2-A1 inherited).
                            # runtime skip for fully-visible blocks (§3.2.1-④): the
                            # condition is s0-block-uniform -> both vids take the same
                            # branch (flag pairing safe; pure-V segment either way).
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
                        # deferred rescale: store the difference, exp in phase 2
                        # (bit-identical to per-iteration exp, §9.3.3-6)
                        T.tile.sub(r_factors[i, :, :], m_i_prev, m_i)
                        T.tile.broadcast(buf_2d, m_i, axis=1)
                        T.tile.sub(work_ub, work_ub, buf_2d)
                        T.tile.exp(work_ub, work_ub)  # P = exp(s - m_new) in [0, 1]
                        T.reduce_sum(work_ub, sumexp_is[i, :, :], dim=-1)

                        if single_kv_block:
                            # pre-normalization specialization (§3.2.1-⑤): P_hat =
                            # P/(sumexp+1e-30) BEFORE the dtype rounding = golden
                            # one-shot softmax grid; phase 2 skips rescale/final div.
                            T.tile.add(sumexp, sumexp_is[0, :, :], 1e-30)
                            T.tile.broadcast(buf_2d, sumexp, axis=1)
                            T.tile.div(work_ub, work_ub, buf_2d)

                        # P -> w2 (SIG_S_HALF ownership closed loop, mha_sink:517-528)
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
                # writeback inside the SIG_S_HALF closed loop + rows_vid runtime
                # guard (§5.4 vid row-split rule; best-practices §6: final store
                # must be inside the ownership loop).
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

        # destroy (§7.2-4): consume outstanding init-direction tokens
        T.wait_flag("V", "MTE2", SIG_IO_UB)
        T.wait_flag("V", "MTE2", SIG_IO_UB + 1)
        if not io3_is_alias:
            T.wait_flag("V", "MTE2", sig_io3_base)
            T.wait_flag("V", "MTE2", sig_io3_base + 1)
        T.wait_flag("MTE3", "V", SIG_S_HALF)

    # ==================== V-side buffer allocation is form-shared; emit inside
    # each prim_func (TVMScript scoping). Form dispatch = Python-level branch
    # building two self-contained prim_funcs (§6.1 note). ====================
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
                # ---- C-side L1 (ZN/NZ annotated for mma operands, §3.2) ----
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
                # ---- L0 double-slot pools (§4.5; P2 shares one pool, P1 splits) ----
                l0a_s = T.alloc_L0A([2, block_M, D], dtype)
                l0b_s = T.alloc_L0B([2, D, block_N], dtype)
                l0c_s = T.alloc_L0C([2, block_M, block_N], accum_dtype)
                l0a_o = l0a_s if block_N == D else T.alloc_L0A([2, block_M, block_N], dtype)
                l0b_o = l0b_s if block_N == D else T.alloc_L0B([2, block_N, D], dtype)
                l0c_o = l0c_s if block_N == D else T.alloc_L0C([2, block_M, D], accum_dtype)
                # ---- V-side UB (vid half rows; §4.3 + adaptations) ----
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
                    # init: pretend consumers already released (§7.2-4)
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
                        # ---- #R5-1 task-batched single-KV-block C path (Form A) ----
                        # Mirror of the Form B batched branch below, with the Form A
                        # mma pipeline body (GEMM1/GEMM2 batches) carried over verbatim
                        # from the generic per-task flow: slot i <-> task
                        # my_start + kb*NUM_STAGES + i. num_tb/batch_tasks from the SAME
                        # formulas as the V mirror (§7.2-5 flag-pairing hard constraint).
                        # Rationale: kv_iters==1 collapses eff_num_outer to 1 and
                        # batch_iters to 1, so the NUM_STAGES pipeline (and the io_buf[2]
                        # ping-pong it feeds) never had a second slot in flight — every
                        # task ran a strictly serial C-QK -> V-softmax -> C-PV -> V-epi
                        # chain (camodel case1: C FLOWCTRL 83.2% of wall, V MTE/VECTOR
                        # overlap 5.3%). Batching the TASK dimension is the in-rules
                        # Expert expression of CV overlap (T.Pipelined is banned,
                        # §2.2-d/R9; performance-antipatterns "AIC/AIV no CV overlap").
                        # Two Form-A-specific deltas vs the generic body (both are Q
                        # scheduling, zero arithmetic change):
                        #   (a) the Q GM->L1 load moves INSIDE the slot loop (each task
                        #       has its own s0), and
                        #   (b) q_l1 -> l0a_s[side] is issued for EVERY slot (generic
                        #       guards it with `if i < 2` because Q is kv-invariant
                        #       there; here Q changes per slot).
                        # SIG_Q_L1 token balance: per slot wait(MTE1->MTE2) +
                        # set(MTE2->MTE1) + wait(MTE2->MTE1) + set(MTE1->MTE2) right
                        # after the single MTE1 read of q_l1 = net 0, identical to the
                        # generic task-level accounting — so the task-end return at the
                        # bottom of the else branch is NOT emitted here.
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
                            # direct FIX write (#R5-2 session-6: single-block O =
                            # P_hat @ V is FINAL (pre-normalization specialization,
                            # §3.2.1-⑤) so the w3 fp32 GM round-trip, the V
                            # phase-2 acc_o pass and the WS3 SEM pair are all
                            # removable.
                            # fixpipe fp32->dtype RNE == V-side vcast (probe #R3-3
                            # 11/11 BITEXACT); L0C->BSHD 4D runtime-row-slice FIX
                            # gate probe 8/8 (probe_r52_l0c_bhsd, int-exact
                            # BITEXACT incl. tail valid_m< bM / bf16 / D=128).
                            # C writes ALL valid_m rows at once (no vid split).
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

                            # Q load (task-level residency, SIG_Q_L1 bidirectional)
                            T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                            T.copy(Q[bz, s0 : s0 + valid_m, by, :], q_l1)
                            T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                            T.wait_flag("MTE2", "MTE1", SIG_Q_L1)

                            for k in T.serial(eff_num_outer):
                                _remaining = eff_kv_iters - k * NUM_STAGES
                                batch_iters = T.if_then_else(_remaining < NUM_STAGES, _remaining, NUM_STAGES)

                                # --- GEMM1 batch: scores -> w1 ---
                                T.wait_cross_flag(SEM_WS1_V2C)
                                for i in T.serial(batch_iters):
                                    side = i % 2
                                    idx = k * NUM_STAGES + i
                                    j0 = idx * block_N
                                    valid_n = T.min(block_N, S_kv - j0)
                                    # K: GM -> L1 (BSHD row-stride, runtime extent)
                                    T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                                    T.copy(K[bz, j0 : j0 + valid_n, by, :], k_l1)
                                    T.set_flag("MTE2", "MTE1", SIG_K_L1)
                                    # Q: L1 -> L0A (first 2 iters of each batch, then reused)
                                    T.wait_flag("M", "MTE1", SIG_L0AB + side)
                                    if i < 2:
                                        T.copy(q_l1, l0a_s[side, :, :])
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
                                    if (i + 1) % CROSS_INTERVAL == 0 or i == batch_iters - 1:
                                        T.set_cross_flag("FIX", SEM_WS1_C2V)

                                # --- GEMM2 batch: consume w2 (P), produce PV -> w3 ---
                                T.wait_cross_flag(SEM_WS3_V2C)
                                for i in T.serial(batch_iters):
                                    side = i % 2
                                    idx = k * NUM_STAGES + i
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

                                T.set_cross_flag("MTE2", SEM_WS2_C2V)

                            # MTE1 no longer reads q_l1; return it before the next task.
                            T.set_flag("MTE1", "MTE2", SIG_Q_L1)

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
                k_l1 = T.alloc_L1([block_N, D], dtype)
                v_l1 = T.alloc_L1([block_N, D], dtype)
                p_l1 = T.alloc_L1([block_M, block_N], dtype)
                # ---- L0C single slots (gemm_v0 manages L0A/L0B by kL0Size) ----
                acc_s_l0c = T.alloc_L0C([block_M, block_N], accum_dtype)
                acc_o_l0c = T.alloc_L0C([block_M, D], accum_dtype)
                # ---- V-side UB (identical to Form A; §4.3 + adaptations) ----
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

                # ==================== Cube core (Form B: gemm_v0, barrier-free #R4-1) ====================
                with T.Scope("C"):
                    # init (§7.2-4): pretend the w2 slots are already released by
                    # the consumer — V's first phase-1 batch waits SEM_WS2_C2V.
                    # (No SIG_* inits: Form B has no intra-core user flags, §7.2-2.)
                    T.set_cross_flag("MTE2", SEM_WS2_C2V)
                    if single_kv_block:
                        # ---- #R3-1 task-batched single-KV-block C path ----
                        # One protocol round batches NUM_STAGES TASKS (kv_iters==1
                        # => every task is exactly one KV block): slot i <-> task
                        # my_start + kb*NUM_STAGES + i. Per-slot ops identical to
                        # the generic flow; V mirror derives num_tb/batch_tasks
                        # from the SAME my_start/my_count formulas (§7.2-5).
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
                                # #R4-1 barrier-free: gemm_v0 prologue orders prior
                                # MTE2 (Q/K loads) before its MTE1 reads and prior FIX
                                # (w1/w3 writes) before its mma; epilogue orders MTE1
                                # before subsequent MTE2 (L1 slot reuse) and mma before
                                # subsequent FIX (L0C readout). Same-pipe ops are
                                # in-order; cross-core hazards stay on the SEM protocol
                                # (set/wait positions unchanged). See S4-Step 2.
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

                            # #R4-1 barrier-free: the first gemm_v0 prologue of the task
                            # drains this MTE2 Q load before its MTE1 reads; q_l1 reuse
                            # across tasks is ordered by the previous gemm epilogue
                            # (MTE1 -> MTE2). See S4-Step 2 hazard coverage.
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
