"""MHA (Multi-Head Attention) forward, BSHD layout, Developer mode.

y = softmax(Q @ K^T * s) @ V,   s = scaleValue if scaleValue > 0 else 1/sqrt(D)
is_causal: scores[..., i, j] = -inf for j > i + (S_kv - S)  (bottom-right aligned)
All-masked rows (only reachable out-of-domain) yield 0 via the sumexp + 1e-30 guard.

Implementation notes (see DESIGN.md):
- Developer mode: AUTO_CV_COMBINE / AUTO_CV_SYNC / AUTO_SYNC / MEMORY_PLANNING
  all enabled; no workspace_idx, no vid offsets, no manual sync in source.
- threads=1 (cv_1_1) instead of the design's threads=2 — verified deviation:
  with threads=2 the AscendVidReduction pass mis-handles GM<->UB copies whose
  GM region is 4D BSHD ([1, rows, 1, D]): the row extent sits at dim 1 while
  the pass expects it at dim (gm_dims - ub_dims) = 2 (the unit head dim), so
  the extent is left unhalved while the UB buffer IS halved, emitting
  maskShapeM = block_M against a block_M/2-row slot (2x out-of-bounds rows,
  silent cross-task corruption; observed non-deterministically). The same pass
  also ICHECK-crashes on runtime valid extents (ExtentIsEqualOne) and halves
  the 1D column-dim mask buffers (col_pos/mask_row). threads=1 skips the pass
  entirely and the design-original forms (runtime valid_m/valid_n slices, 1D
  col_pos/mask_row, vid-free thresholds) are bit-exact on S=1/30/64/90 probes.
- T.serial KV loop instead of T.Pipelined(num_stages=2): threads=1 +
  T.Pipelined fails codegen ("Cannot find pre-allocated address for buffer:
  k_l1" — double-buffered L1 operands are not registered in address_map under
  cv_1_1; repro isolated). T.serial + threads=1 compiles and runs; kv_iters
  stays a compile-time bound, tail semantics unchanged. KV prefetch overlap is
  lost (perf sentinels §9.4 gate the impact; manual double-buffering is a
  Stage-3 candidate).
- BSHD row-stride DMA: slice column dim = innermost D (contiguous rows,
  row gap = (N-1)*D elements); no host-side transposes/padding (DESIGN §4.8).
"""

import tilelang
import torch
from tilelang import language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


def _cube_core_num() -> int:
    """Physical cube-core count (HOST METADATA query, zero tensor data touched;
    precedent: examples/deepseek_v4/lightning_indexer.py:14-21)."""
    try:
        import torch_npu

        return int(torch_npu.npu.get_device_properties(0).cube_core_num)
    except Exception:  # noqa: BLE001 - conservative 910B-class fallback
        return 20


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
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
    # Host-side pure-Python config derivation (no tensor data touched, DESIGN §4.8)
    sm_scale = scaleValue if scaleValue > 0 else (1.0 / D) ** 0.5
    # Decode narrow-block specialization (Stage-3 #6, flash_attn_optimize §9(b)):
    # S<=16 -> block_M=16 cuts padded-row waste from 63/64 to <=15/16 and cuts
    # the per-iteration vector work ~4x (every score/output tile op scales with
    # block_M); the per-block CV workspace round-trips shrink 4x too. Restricted
    # to S<=16: for larger S a narrow tile would multiply s_blocks and thus the
    # per-block K/V re-reads/task overhead (anti-win on prefill shapes, where
    # the rows are real work, not padding). M=16 = cube fractal minimum.
    block_M = 16 if S <= 16 else (32 if D == 256 else 64)
    block_N = 64
    causal_offset = S_kv - S  # bottom-right alignment (>= 0 when causal, asserted)
    need_bound_mask = S_kv % block_N != 0  # tail-column clamp (compile-time)
    need_mask = is_causal or need_bound_mask
    # Unified threshold offset: causal -> bottom-right offset; non-causal -> big
    # constant so T.min(row + offset_eff, S_kv - 1) collapses to S_kv - 1.
    offset_eff = causal_offset if is_causal else 2**30
    s_blocks = (S + block_M - 1) // block_M
    kv_iters = (S_kv + block_N - 1) // block_N  # compile-time loop bound
    block_num = B * N * s_blocks
    # Fixed-Core launch (Stage-3 #5, optimization-guide §2.9 + antipatterns
    # 关注项A + flash_attn_optimize §5 + cube_path §4.1): block_num >> physical
    # cores amplifies per-block task dispatch and TPipe/workspace init; launch
    # min(block_num, cores) cores, each looping over a contiguous task chunk.
    # bx-fastest decomposition -> every chunk spans whole bx cycles -> balanced
    # under the #1 causal trim. Workspace becomes core-indexed (case7 footprint
    # 114MB -> ~1.1MB, L2-resident). Degenerate-safe: block_num <= cores ->
    # core_num = block_num, one task per core (identical to the old grid).
    core_num = min(block_num, _cube_core_num())
    q_tasks = block_num // core_num  # base tasks per core
    r_tasks = block_num % core_num  # first r_tasks cores take one extra
    # Single-KV-block specialization: sumexp is final inside the (only) iteration,
    # so P can be normalized BEFORE the dtype rounding -> P_hat = round16(P/sumexp)
    # matches the golden's softmax output grid exactly (one-shot softmax path).
    single_kv_block = kv_iters == 1

    @T.prim_func
    def main(
        Q: T.Tensor([B, S, N, D], dtype),  # type: ignore
        K: T.Tensor([B, S_kv, N, D], dtype),  # type: ignore
        V: T.Tensor([B, S_kv, N, D], dtype),  # type: ignore
        Output: T.Tensor([B, S, N, D], dtype),  # type: ignore
    ):
        with T.Kernel(core_num, threads=1, is_npu=True) as (cid):
            # Fixed-Core (Stage-3 #5): balanced static task distribution
            # (flash_attn_optimize §5 / mha_sink_fwd:236-238 form): the first
            # r_tasks cores take q_tasks+1 tasks, the rest q_tasks. Count-based
            # T.serial(my_count) never inverts (empty = 0 iters, no tail guard);
            # the nested runtime task loop under AUTO_CV_COMBINE replicates with
            # identical trip counts on AIC/AIV (cross-flag pairing preserved).
            # Buffers/TPipe init happen ONCE per core; the CV workspace becomes
            # core-indexed (framework-derived from the launch grid).
            my_start = cid * q_tasks + T.if_then_else(cid < r_tasks, cid, r_tasks)
            my_count = q_tasks + T.if_then_else(cid < r_tasks, 1, 0)

            q_l1 = T.alloc_shared([block_M, D], dtype)
            k_l1 = T.alloc_shared([block_N, D], dtype)
            v_l1 = T.alloc_shared([block_N, D], dtype)
            acc_s_l1 = T.alloc_shared([block_M, block_N], dtype)
            acc_s_l0c = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_o_l0c = T.alloc_fragment([block_M, D], accum_dtype)
            acc_s_ub = T.alloc_shared([block_M, block_N], accum_dtype)
            acc_s_half = T.alloc_shared([block_M, block_N], dtype)
            acc_o = T.alloc_shared([block_M, D], accum_dtype)
            acc_o_ub = T.alloc_shared([block_M, D], accum_dtype)
            acc_o_half = T.alloc_shared([block_M, D], dtype)
            m_i = T.alloc_shared([block_M], accum_dtype)
            m_i_prev = T.alloc_shared([block_M], accum_dtype)
            sumexp = T.alloc_shared([block_M], accum_dtype)
            sumexp_i_ub = T.alloc_shared([block_M], accum_dtype)
            # Dedicated broadcast-2D scratch buffers (Stage-3 #2). MUST NOT alias
            # the cross-core L0C->UB copy destinations (acc_s_ub/acc_o_ub): the
            # AUTO passes order V ops against the flag-anchored copy by buffer
            # dependency, which silently SANK the rescale Mul below the copy when
            # acc_o_ub doubled as scratch (measured: multi-block corruption; the
            # generated stream showed copy(acc_o_ub)<-w3 BEFORE Mul(acc_o,acc_o_ub)).
            m_i_2d = T.alloc_shared([block_M, block_N], accum_dtype)  # sub/div [bM,bN]
            r_2d = T.alloc_shared([block_M, D], accum_dtype)  # rescale/div [bM,D]
            # mask-chain buffers (Stage-3 #3, 2D form): allocated unconditionally
            # (TVMScript scoping would hide them if allocated inside `if need_mask:`).
            # row_thresh [bM] is task-invariant (hoisted before the KV loop);
            # rt_2d is its per-iteration [bM,bN] broadcast (transient); mask_2d is
            # the bit-packed compare output (1 bit/elem, HISA block_sparse_mqa
            # precedent [n//8] uint8). col_2d reuses m_i_2d (pure-V scratch, dead
            # until the sub broadcast; NEVER alias cross-core copy destinations).
            col_pos = T.alloc_shared([block_N], accum_dtype)
            row_thresh = T.alloc_shared([block_M], accum_dtype)
            rt_2d = T.alloc_shared([block_M, block_N], accum_dtype)
            mask_2d = T.alloc_shared([block_M * block_N // 8], "uint8")

            for t in T.serial(my_count):
                task = my_start + t
                bx = task % s_blocks  # seq block (fastest)
                by = task // s_blocks % N  # head
                bz = task // s_blocks // N % B  # batch
                s0 = bx * block_M
                valid_m = T.min(block_M, S - s0)  # runtime tail rows
                # Causal KV runtime trim (Stage-3 #1, DESIGN §10.3-①): the last
                # row of this s_block sees columns <= s0+block_M-1+causal_offset
                # (bottom-right aligned, clamped to S_kv-1); KV blocks beyond
                # that are fully masked and contribute exactly 0
                # (exp(-inf)=0, r=exp(0)=1, 0@V=0) -> skipping is value-
                # identical. Formula = mha_sink_fwd_bhsd.py:268-271; runtime
                # T.serial bound verified threads=1 Developer (sink_bwd:323-324).
                # Non-causal keeps the compile-time bound (parse-time ternary).
                eff_kv_iters = (
                    T.floordiv(T.min(s0 + block_M - 1 + causal_offset, S_kv - 1), block_N) + 1
                    if is_causal
                    else kv_iters
                )

                T.tile.fill(acc_o, 0.0)
                T.tile.fill(sumexp, 0.0)
                T.tile.fill(m_i, -(2**30))
                if need_mask:
                    # task-invariant row thresholds (hoisted, Stage-3 #3):
                    # thresh_r = min(s0 + r + offset_eff, S_kv-1); non-causal uses
                    # offset_eff=2^30 -> min collapses to S_kv-1 (bound clamp only).
                    T.tile.arith_progression(row_thresh, s0 + offset_eff, 1, block_M)
                    T.tile.min(row_thresh, row_thresh, T.cast(S_kv - 1, accum_dtype))
                # BSHD row-stride slice; tail rows auto-zeroed by copy_gm_to_l1
                T.copy(Q[bz, s0 : s0 + valid_m, by, :], q_l1)

                for k in T.serial(0, eff_kv_iters):
                    j0 = k * block_N
                    valid_n = T.min(block_N, S_kv - j0)  # runtime tail cols
                    T.copy(K[bz, j0 : j0 + valid_n, by, :], k_l1)
                    T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                    # Stage-3 #7: L0C -> UB lands DIRECTLY in acc_s_ub (the old
                    # fill(acc_s_ub,0)+add(acc_s_ub,acc_s_ub_) was the identity
                    # 0+x; acc_s_ub_ retired, freeing 16KB UB for #3). +-0 corner
                    # (0+(-0)=+0 vs copy keeps -0) is unobservable: the first
                    # consumer is the round16/exp path where exp(-0-m)==exp(+0-m).
                    T.copy(acc_s_l0c, acc_s_ub)  # L0C -> UB (compiler-routed)

                    T.copy(m_i, m_i_prev)  # history max BEFORE reduce (DESIGN §10.2)
                    # golden score-grid alignment (DESIGN §3.5.2): torch computes
                    # round16(qk_fp32) first (fp16 matmul output), then *scale in
                    # fp32 -> round16. Mirror both roundings exactly:
                    T.copy(acc_s_ub, acc_s_half)  # round16(qk)
                    T.copy(acc_s_half, acc_s_ub)  # back to fp32
                    T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)  # * s in fp32
                    T.copy(acc_s_ub, acc_s_half)  # round16(scores) = golden grid
                    T.copy(acc_s_half, acc_s_ub)

                    if need_mask:  # compile-time variant switch (4 variants, §5.6)
                        # 2D mask chain (Stage-3 #3, sink_bwd:177-196 precedent):
                        # visible iff col <= min(row + offset_eff, S_kv-1). One
                        # arith_progression + 2 broadcasts + one [bM,bN] compare +
                        # one 2D select replace the 64x(CompareScalar+Select+2
                        # barriers) row loop. col_2d staged in m_i_2d scratch.
                        # Runtime skip for FULLY-VISIBLE blocks (Stage-3 #4,
                        # mha_sink_fwd:502 conditional precedent): block k needs
                        # the mask iff its LAST column exceeds the smallest row
                        # threshold of this s_block (row s0; row_thresh is
                        # nondecreasing in r). Pure scalar condition, no UB
                        # scalar read. Under the #1 trim this leaves exactly the
                        # diagonal block (causal) and/or the tail block (bound
                        # clamp) masked; skipping a fully-visible mask is
                        # bit-identical (compare all-true -> select = identity).
                        if k * block_N + block_N - 1 > T.min(s0 + offset_eff, S_kv - 1):
                            T.tile.arith_progression(col_pos, j0, 1, block_N)
                            T.tile.broadcast(m_i_2d, col_pos, axis=0)  # [bN] -> [bM,bN]
                            T.tile.broadcast(rt_2d, row_thresh, axis=1)  # [bM] -> [bM,bN]
                            T.tile.compare(mask_2d, m_i_2d, rt_2d, "LE")
                            T.tile.select(
                                acc_s_ub,
                                mask_2d,
                                acc_s_ub,
                                -T.infinity(accum_dtype),
                                "VSEL_TENSOR_SCALAR_MODE",
                            )

                    T.reduce_max(acc_s_ub, m_i, dim=-1)
                    T.tile.max(m_i, m_i, m_i_prev)
                    T.tile.sub(m_i_prev, m_i_prev, m_i)
                    T.tile.exp(m_i_prev, m_i_prev)  # r = exp(m_old - m_new)

                    # broadcast 2D form (Stage-3 #2, sink_bwd:200-201 precedent):
                    # [bM] -> [bM,bN] replication + single 2D sub replaces the
                    # 64x(GetValue+PIPE_ALL+micro-Adds) row loop. Dedicated m_i_2d
                    # scratch (see allocation note: no aliasing with cross-core
                    # copy destinations).
                    T.tile.broadcast(m_i_2d, m_i, axis=1)
                    T.tile.sub(acc_s_ub, acc_s_ub, m_i_2d)
                    T.tile.exp(acc_s_ub, acc_s_ub)  # P = exp(s - m_new) in [0, 1]
                    T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                    T.tile.mul(sumexp, sumexp, m_i_prev)
                    T.tile.add(sumexp, sumexp, sumexp_i_ub)

                    if single_kv_block:
                        # normalize P before rounding (golden P_hat grid); the
                        # +1e-30 guard keeps all-masked rows at 0 (no NaN)
                        T.tile.add(sumexp, sumexp, 1e-30)
                        # broadcast 2D form (Stage-3 #2): [bM] -> [bM,bN]
                        T.tile.broadcast(m_i_2d, sumexp, axis=1)
                        T.tile.div(acc_s_ub, acc_s_ub, m_i_2d)

                    T.copy(acc_s_ub, acc_s_half)  # P -> input dtype (golden grid)
                    T.copy(acc_s_half, acc_s_l1)  # UB -> L1 (GEMM2 operand A)

                    T.copy(V[bz, j0 : j0 + valid_n, by, :], v_l1)
                    T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                    # rescale acc_o by r=exp(m_old-m_new) BEFORE the L0C->UB copy
                    # (broadcast 2D form, Stage-3 #2, sink_bwd:205-209 ordering).
                    # Dedicated r_2d scratch: acc_o_ub MUST stay free because it is
                    # the cross-core L0C->UB copy destination below (the AUTO pass
                    # hoists that copy right after its CrossCoreWaitFlag, clobbering
                    # any scratch value staged in acc_o_ub -> measured corruption).
                    T.tile.broadcast(r_2d, m_i_prev, axis=1)  # [bM] -> [bM,D]
                    T.tile.mul(acc_o, acc_o, r_2d)
                    T.copy(acc_o_l0c, acc_o_ub)  # L0C -> UB (compiler-routed)
                    T.tile.add(acc_o, acc_o, acc_o_ub)

                if not single_kv_block:
                    # final normalization; +1e-30 = all-masked-row guard (0/1e-30 -> 0)
                    T.tile.add(sumexp, sumexp, 1e-30)
                    # broadcast 2D form (Stage-3 #2): [bM] -> [bM,D]
                    T.tile.broadcast(r_2d, sumexp, axis=1)
                    T.tile.div(acc_o, acc_o, r_2d)

                T.copy(acc_o, acc_o_half)
                # write back valid_m rows only (no OOB, no neighbour clobbering)
                T.copy(acc_o_half[0:valid_m, :], Output[bz, s0 : s0 + valid_m, by, :])

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
