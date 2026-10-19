"""R4 fused sampling chain (V01-a).

Two TileLang kernels + host wrapper:
  K1 r4_topk_stats  : single-pass chunked scan -> online-softmax stats (m, s),
                      synthesized threshold tau (theorem T1/T2), degrade guard valid.
  K2 r4_mask_sample : second pass -> masked (un-renormalized) probs w_out[B, V].
  r4_fused_sampling : host wrapper (pre-checks C3/C6/C7 + K1 + K2).

Programming model: Developer (pure Vector) + Ascend-specific T.tile.topk hybrid.
See DESIGN.md sections 2/3/5/7. No golden / no tests / no __main__ here.

Implementation notes (validated on Ascend910 board, see debug_log.md):
  * All scalars live in [1,1]/[1] alloc_shared buffers; every arithmetic step uses
    T.tile.* (scalar T.exp/T.log/T.max do not codegen in a pure-vector kernel, and
    T.alloc_var rebinding yields wrong values).
  * Never T.copy between 1-element UB buffers (4 B < 32 B DMA alignment -> aicore
    fault); running max uses T.tile.max in place.
  * Tail block uses an exactly-sized [1,TAIL] buffer + FULL reduce (the narrow-reduce
    real_shape path is capped at 64 fp32 elements, codegen_ascend.cc:2145).
  * top-k source is a dedicated rank-1 buffer (max_actual_num = sum(shape) trap).
"""

import tilelang
from tilelang import language as T

# Developer / pure-Vector pass_configs (raw-string form, cf. examples/xllm_kernels/utils.py).
R4_PASS_CONFIGS = {
    "tl.ascend_auto_sync": True,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cross_core_sync": True,
    "tl.ascend_auto_cv_combine": True,
}

VEC_NUM = 2  # 2 Vector cores per Cube core (fused_gdn_gating.py:26)
TOP_K_ALL = 1 << 30  # sglang/srt/sampling/sampling_params.py:40 (NOT -1)

_ASCEND_VEC_CORE_NUM_KEYS = ("vector_core_num", "aiv_core_num", "vec_core_num")


def detect_vec_core_num(default_vec_core_num: int = 48) -> int:
    """Runtime vector-core probe (never hardcode; Ascend910_9362 measures 40)."""
    try:
        import torch

        if hasattr(torch, "npu") and torch.npu.is_available():
            props = torch.npu.get_device_properties(torch.npu.current_device())
            for key in _ASCEND_VEC_CORE_NUM_KEYS:
                value = getattr(props, key, None)
                if isinstance(value, int) and value > 0:
                    return value
    except Exception:
        pass
    return default_vec_core_num


def _aligned_topk(K: int, elems_per_block: int = 8) -> int:
    """topk dst size = ceil(2K / elems_per_block) * elems_per_block (fp32 -> 8)."""
    return ((2 * K + elems_per_block - 1) // elems_per_block) * elems_per_block


@tilelang.jit(out_idx=[4, 5, 6, 7], pass_configs=R4_PASS_CONFIGS)
def r4_topk_stats(B: int, V: int, K_MAX: int, BV: int, NEED_MIN_P: bool):
    """K1: online-softmax stats + hierarchical top-K merge + threshold/guard.

    Outputs (auto-allocated via out_idx): m_out[B], s_out[B], tau_out[B] (fp32),
    valid[B] (int32).
    """
    n_blk_full = V // BV
    TAIL = V - n_blk_full * BV
    b_num = T.ceildiv(B, VEC_NUM)
    AT = _aligned_topk(K_MAX)  # 512 for K_MAX=256, fp32

    @T.prim_func
    def main(
        logits: T.Tensor([B, V], "float32"),  # type: ignore  # in
        top_ks: T.Tensor([B], "int32"),  # type: ignore  # in
        top_ps: T.Tensor([B], "float32"),  # type: ignore  # in
        min_ps: T.Tensor([B], "float32"),  # type: ignore  # in
        m_out: T.Tensor([B], "float32"),  # type: ignore  # out
        s_out: T.Tensor([B], "float32"),  # type: ignore  # out
        tau_out: T.Tensor([B], "float32"),  # type: ignore  # out
        valid: T.Tensor([B], "int32"),  # type: ignore  # out
    ):
        with T.Kernel(b_num, is_npu=True) as (cid, vid):
            row = cid * VEC_NUM + vid
            if row < B:
                # ---- online-softmax scalars ([1,1] buffers; T.tile.* arithmetic) ----
                m_blk = T.alloc_shared([1, 1], "float32")
                m_run = T.alloc_shared([1, 1], "float32")
                m_old = T.alloc_shared([1, 1], "float32")
                s_blk = T.alloc_shared([1, 1], "float32")
                s_run = T.alloc_shared([1, 1], "float32")
                tmp = T.alloc_shared([1, 1], "float32")
                # ---- top-k merge buffers (rank-1 topk source; N2) ----
                l_dst = T.alloc_shared([AT], "float32")
                l_val = T.alloc_shared([K_MAX], "float32")
                merged = T.alloc_shared([2 * K_MAX], "float32")
                m_dst = T.alloc_shared([AT], "float32")
                run_v = T.alloc_shared([K_MAX], "float32")
                # ---- epilogue buffers ----
                rv2 = T.alloc_shared([1, K_MAX], "float32")
                q2 = T.alloc_shared([1, K_MAX], "float32")
                mk2 = T.alloc_shared([1, K_MAX], "float32")
                sk2 = T.alloc_shared([1, K_MAX], "float32")
                tie_m = T.alloc_shared([K_MAX], "float32")
                k_i = T.alloc_shared([1], "int32")
                p_i = T.alloc_shared([1, 1], "float32")
                mu_i = T.alloc_shared([1, 1], "float32")
                muc = T.alloc_shared([1, 1], "float32")
                lnmu = T.alloc_shared([1, 1], "float32")
                tau_k = T.alloc_shared([1], "float32")
                tau_p = T.alloc_shared([1], "float32")
                tau_u = T.alloc_shared([1, 1], "float32")
                tau = T.alloc_shared([1], "float32")
                prefix = T.alloc_shared([1, 1], "float32")
                n_gt = T.alloc_shared([1], "float32")
                n_eq = T.alloc_shared([1], "float32")
                okp = T.alloc_shared([1], "int32")
                tieok = T.alloc_shared([1], "int32")
                vflag = T.alloc_shared([1], "int32")

                # Touch every scalar-write-only buffer with a Vector op (scope inference).
                T.tile.fill(tau_k, 0.0)
                T.tile.fill(tau_p, 0.0)
                T.tile.fill(prefix, 0.0)
                T.tile.fill(tau_u, 0.0)
                T.tile.fill(okp, 0)
                T.tile.fill(tieok, 0)
                T.tile.fill(vflag, 0)
                T.tile.fill(tie_m, 0.0)

                # ---- per-row scalar params (GM -> UB, 1 element each) ----
                T.copy(top_ks[row : row + 1], k_i)
                T.copy(top_ps[row : row + 1], p_i[0, 0:1])
                T.copy(min_ps[row : row + 1], mu_i[0, 0:1])

                T.tile.fill(m_run, -T.infinity("float32"))
                T.tile.fill(s_run, 0.0)
                T.tile.fill(run_v, -T.infinity("float32"))  # N4: -inf before first merge

                # ======== Pass 1 main loop: online softmax + hierarchical top-k merge ========
                # The BV-sized buffers are allocated only when there IS at least one full
                # block. When V < BV (n_blk_full == 0) the whole row is handled by the tail
                # path below; allocating them unconditionally would leave them dead, and a
                # dead alloc_shared buffer has no pre-allocated address -> codegen fails
                # ("Cannot find pre-allocated address for buffer", codegen_ascend.cc:828).
                if n_blk_full > 0:
                    z_ub = T.alloc_shared([1, BV], "float32")
                    e_ub = T.alloc_shared([1, BV], "float32")
                    m2d = T.alloc_shared([1, BV], "float32")
                    zt_ub = T.alloc_shared([BV], "float32")
                    for b in T.serial(n_blk_full):
                        T.copy(logits[row, b * BV], z_ub)
                        # (a) online softmax (T.tile.* only; no 1-elem UB->UB copy)
                        T.reduce_max(z_ub, m_blk, dim=-1)
                        T.tile.max(m_old, m_run, m_blk)  # m_old = m_new = max(m_run, blk)
                        T.tile.sub(tmp, m_run, m_old)  # m_prev - m_new
                        T.tile.exp(tmp, tmp)
                        T.tile.mul(s_run, s_run, tmp)  # s_run *= exp(m_prev - m_new)
                        T.tile.max(m_run, m_run, m_blk)  # m_run = m_new (in place)
                        T.tile.broadcast(m2d, m_run)  # m2d[1,BV] = m_new
                        T.tile.sub(e_ub, z_ub, m2d)
                        T.tile.exp(e_ub, e_ub)
                        T.reduce_sum(e_ub, s_blk, dim=-1)
                        T.tile.add(s_run, s_run, s_blk)
                        # (b) block top-K_MAX + cross-block merge
                        T.copy(z_ub[0, :BV], zt_ub)  # rank-2 -> rank-1 (N2)
                        T.tile.topk(l_dst, zt_ub, K_MAX, BV)
                        T.tile.gather_mask(l_val, l_dst, "P0101")  # even slots = values
                        T.copy(run_v, merged[:K_MAX])
                        T.copy(l_val, merged[K_MAX:])
                        T.tile.topk(m_dst, merged, K_MAX, 2 * K_MAX)
                        T.tile.gather_mask(run_v, m_dst, "P0101")

                # ======== tail block: exactly-sized buffers -> FULL reduce (V-2 fallback) ========
                if TAIL > 0:
                    zt = T.alloc_shared([1, TAIL], "float32")
                    et = T.alloc_shared([1, TAIL], "float32")
                    mt = T.alloc_shared([1, TAIL], "float32")
                    ztt = T.alloc_shared([TAIL], "float32")
                    T.copy(logits[row, n_blk_full * BV], zt)
                    T.reduce_max(zt, m_blk, dim=-1)
                    T.tile.max(m_old, m_run, m_blk)
                    T.tile.sub(tmp, m_run, m_old)
                    T.tile.exp(tmp, tmp)
                    T.tile.mul(s_run, s_run, tmp)
                    T.tile.max(m_run, m_run, m_blk)
                    T.tile.broadcast(mt, m_run)
                    T.tile.sub(et, zt, mt)
                    T.tile.exp(et, et)
                    T.reduce_sum(et, s_blk, dim=-1)
                    T.tile.add(s_run, s_run, s_blk)
                    T.copy(zt[0, :TAIL], ztt)
                    T.tile.topk(l_dst, ztt, K_MAX, TAIL)
                    T.tile.gather_mask(l_val, l_dst, "P0101")
                    T.copy(run_v, merged[:K_MAX])
                    T.copy(l_val, merged[K_MAX:])
                    T.tile.topk(m_dst, merged, K_MAX, 2 * K_MAX)
                    T.tile.gather_mask(run_v, m_dst, "P0101")

                # ======== epilogue: three thresholds (T1/T2) + tie detect + guard ========
                # q2[l] = exp(run_v[l] - m) / s  (vectorized; scalar T.exp does not codegen)
                T.copy(run_v, rv2[0, :])
                T.tile.broadcast(mk2, m_run, axis=1)
                T.tile.sub(q2, rv2, mk2)
                T.tile.exp(q2, q2)
                T.tile.broadcast(sk2, s_run, axis=1)
                T.tile.div(q2, q2, sk2)

                # tau_k = run_v[k_i-1] (guarded dynamic index; cross_entropy:113-115 style)
                T.tile.fill(tau_k, -T.infinity("float32"))
                for _ in T.serial(1):
                    if k_i[0] >= 1 and k_i[0] <= K_MAX:
                        tau_k[0] = run_v[k_i[0] - 1]

                # tau_u (theorem T2): mu>0 -> m + ln(mu); else -inf
                T.tile.fill(tau_u, -T.infinity("float32"))
                if NEED_MIN_P:
                    T.tile.max(muc, mu_i, 1e-30)  # clamp to avoid ln(0)
                    T.tile.ln(lnmu, muc)
                    T.tile.add(tau_u, m_run, lnmu)
                    for _ in T.serial(1):
                        if mu_i[0, 0] <= 0.0:
                            tau_u[0, 0] = -T.infinity("float32")

                # tau_p: strict prefix sum over descending run_v (monotone guard, no break)
                T.tile.fill(tau_p, -T.infinity("float32"))
                T.tile.fill(prefix, 0.0)
                for l in T.serial(K_MAX):
                    with T.If(prefix[0, 0] <= p_i[0, 0]):  # noqa: SIM117
                        with T.Then():
                            tau_p[0] = run_v[l]
                            prefix[0, 0] = prefix[0, 0] + q2[0, l]

                # ok_p: nucleus determined within K_MAX (prefix crossed p) or p>=1.0
                for _ in T.serial(1):
                    okp[0] = T.if_then_else(p_i[0, 0] >= 1.0, 1, T.if_then_else(prefix[0, 0] > p_i[0, 0], 1, 0))

                # D-TIE: n_gt / n_eq on the K_MAX list (vectorized)
                T.tile.fill(tie_m, 0.0)
                for l in T.Parallel(K_MAX):
                    tie_m[l] = T.if_then_else(run_v[l] > tau_k[0], 1.0, 0.0)
                T.reduce_sum(tie_m, n_gt, dim=-1)
                T.tile.fill(tie_m, 0.0)
                for l in T.Parallel(K_MAX):
                    tie_m[l] = T.if_then_else(run_v[l] == tau_k[0], 1.0, 0.0)
                T.reduce_sum(tie_m, n_eq, dim=-1)

                # tau = max(tau_k, tau_p, tau_u)  (theorem T1)
                T.tile.max(tau, tau_k, tau_p)
                T.tile.max(tau, tau, tau_u[0, 0:1])

                for _ in T.serial(1):
                    # D-TIE. n_gt/n_eq are counted over the merged top-K_MAX list, so
                    #   sum >  k_i      -> threshold keeps more than k_i -> degrade
                    #   sum <  K_MAX    -> list not saturated: every tie at tau_k was
                    #                      captured, so sum == k_i is exact -> accept
                    #   sum == K_MAX    -> list saturated; only accept when no tie was
                    #                      observed at tau_k (n_eq <= 1). This keeps the
                    #                      k_i == K_MAX fast path alive for continuous
                    #                      logits (DESIGN.md 3.4 point 2) while still
                    #                      degrading on tie-saturated rows (all-equal).
                    # Known gap: a tie pair straddling ranks K_MAX-1/K_MAX is invisible
                    # here (would need z_sort[K_MAX]); see debug_log.md.
                    tieok[0] = T.if_then_else(
                        tau[0] > tau_k[0],
                        1,
                        T.if_then_else(
                            n_gt[0] + n_eq[0] > T.Cast("float32", k_i[0]),
                            0,
                            T.if_then_else(
                                n_gt[0] + n_eq[0] < T.Cast("float32", K_MAX),
                                1,
                                T.if_then_else(n_eq[0] <= 1.0, 1, 0),
                            ),
                        ),
                    )
                    vflag[0] = T.if_then_else(
                        (k_i[0] >= 1 and k_i[0] <= K_MAX),
                        T.if_then_else(okp[0] == 1 and tieok[0] == 1, 1, 0),
                        0,
                    )

                T.copy(m_run[0, 0:1], m_out[row : row + 1])
                T.copy(s_run[0, 0:1], s_out[row : row + 1])
                T.copy(tau, tau_out[row : row + 1])
                T.copy(vflag, valid[row : row + 1])

    return main


@tilelang.jit(out_idx=[4], pass_configs=R4_PASS_CONFIGS)
def r4_mask_sample(B: int, V: int, BV: int):
    """K2: second pass -> w_out[B,V] = (z >= tau) ? exp(z-m)/s : 0 (no renorm)."""
    n_blk = T.ceildiv(V, BV)
    b_num = T.ceildiv(B, VEC_NUM)

    @T.prim_func
    def main(
        logits: T.Tensor([B, V], "float32"),  # type: ignore  # in
        m_in: T.Tensor([B], "float32"),  # type: ignore  # in
        s_in: T.Tensor([B], "float32"),  # type: ignore  # in
        tau_in: T.Tensor([B], "float32"),  # type: ignore  # in
        w_out: T.Tensor([B, V], "float32"),  # type: ignore  # out
    ):
        with T.Kernel(b_num, is_npu=True) as (cid, vid):
            row = cid * VEC_NUM + vid
            if row < B:
                z_ub = T.alloc_shared([1, BV], "float32")
                e_ub = T.alloc_shared([1, BV], "float32")
                w_ub = T.alloc_shared([1, BV], "float32")
                m2d = T.alloc_shared([1, BV], "float32")
                s2d = T.alloc_shared([1, BV], "float32")
                t2d = T.alloc_shared([1, BV], "float32")
                mask = T.alloc_shared([1, BV // 8], "uint8")
                m_v = T.alloc_shared([1, 1], "float32")
                s_v = T.alloc_shared([1, 1], "float32")
                t_v = T.alloc_shared([1, 1], "float32")

                T.copy(m_in[row : row + 1], m_v[0, 0:1])
                T.copy(s_in[row : row + 1], s_v[0, 0:1])
                T.copy(tau_in[row : row + 1], t_v[0, 0:1])

                for b in T.serial(n_blk):
                    # gm2ub clamps the tail; pad -inf so exp(pad-m)=0 (store re-clamps anyway)
                    T.copy(logits[row, b * BV], z_ub, pad_value=-T.infinity("float32"))
                    T.tile.broadcast(m2d, m_v)
                    T.tile.sub(e_ub, z_ub, m2d)
                    T.tile.exp(e_ub, e_ub)
                    T.tile.broadcast(s2d, s_v)
                    T.tile.div(e_ub, e_ub, s2d)
                    T.tile.broadcast(t2d, t_v)
                    T.tile.compare(mask, z_ub, t2d, "GE")
                    T.tile.select(w_ub, mask, e_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                    T.copy(w_ub, w_out[row, b * BV])  # ub2gm auto re-clamp (C2)

    return main


def r4_fused_sampling(
    logits,
    top_ks,
    top_ps,
    min_ps,
    need_min_p_sampling,
    K_MAX=256,
    BV=4096,
):
    """Host wrapper: pre-checks (C3/C6/C7) + K1 + K2.

    Returns (m_out, s_out, tau_out, valid, w_out). torch.multinomial(w_out, 1) is the
    user-retained downstream sampler (V01-a, DEV-1); not invoked here.
    """
    import torch

    B, V = logits.shape
    # C6: B == 0 -> host intercept, do not launch (zero-sized grid aborts).
    if B == 0:
        dev = logits.device
        empty_w = torch.empty((0, V), dtype=torch.float32, device=dev)
        empty_b = torch.empty((0,), dtype=torch.float32, device=dev)
        empty_i = torch.empty((0,), dtype=torch.int32, device=dev)
        return empty_b, empty_b, empty_b, empty_i, empty_w
    # C7: V too small for top-K_MAX.
    if V < K_MAX:
        raise NotImplementedError(f"R4: V={V} < K_MAX={K_MAX}; fall back to torch path")
    # C3: topk needs K_MAX <= actual_num on every block incl. tail.
    tail = V % BV
    if tail != 0 and tail < K_MAX:
        raise NotImplementedError(f"R4: tail block {tail} < K_MAX={K_MAX} (need V%BV==0 or V%BV>=K_MAX); fall back to torch path")
    if logits.dtype != torch.float32:
        raise NotImplementedError(f"R4 V01-a supports float32 only; got {logits.dtype}")

    k1 = r4_topk_stats(B, V, K_MAX, BV, bool(need_min_p_sampling))
    m_out, s_out, tau_out, valid = k1(logits, top_ks, top_ps, min_ps)
    k2 = r4_mask_sample(B, V, BV)
    w_out = k2(logits, m_out, s_out, tau_out)
    return m_out, s_out, tau_out, valid, w_out
