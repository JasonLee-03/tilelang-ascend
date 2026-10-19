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
  * Tail block uses an exactly-sized [1,TAIL] buffer + FULL reduce for the softmax
    statistics (the narrow-reduce real_shape path is capped at 64 fp32 elements,
    codegen_ascend.cc:2145).
  * top-k source is a dedicated rank-1 buffer (max_actual_num = sum(shape) trap).
  * The tail top-k source must be padded to a multiple of 32 ELEMENTS with -inf (see
    TAIL_PAD in r4_topk_stats). Sort32 consumes ceil(count/32)*32 elements and
    tl::ascend::Sort() repairs the remainder with a MASKED AscendC::Duplicate that does
    not take effect on dav_c220 -- without our own pad those trailing slots keep stale
    UB and get sorted into the result; and a UB->UB DataCopy of a non-32B-multiple count
    silently TRUNCATES to a multiple of 8 fp32, dropping real tail elements. Both are
    avoided by staging through a [1, TAIL_PAD] buffer that the GM->UB copy -inf-pads
    itself (pad_value), then hopping exactly TAIL_PAD elements (V01-b rev2 fix).
"""

import ctypes

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


# ---- V01-b rev1: counter-based avalanche hash (lowbias32) constants ----
# (DESIGN_v01b.md section 0.0.2 / 3.3). int32 scalars are derived from the canonical hex
# via ctypes to avoid decimal-transcription errors.
# NOTE (Stage 2 fix): DESIGN_v01b.md section 3.3 lists HASH_M2 = -2073151861, but that decimal
# is 0x846E368B; the lowbias32 multiplier is M2 = 0x846CA68B -> int32 = -2073254261. The golden
# (r4_ref.py) uses the hex 0x846CA68B (correct); the kernel must match it (validated V-b1).
def _i32(c: int) -> int:
    """Python int constant -> its signed int32 kernel-scalar equivalent (mod 2^32)."""
    return ctypes.c_int32(c & 0xFFFFFFFF).value


PRIME_W = _i32(0x9E3779B1)  # -1640531535: seed mixer (Knuth multiplicative); kernel const, not a JIT key
HASH_M1 = _i32(0x7FEB352D)  # 2146121005: lowbias32 multiplier 1 (positive int32)
HASH_M2 = _i32(0x846CA68B)  # -2073254261: lowbias32 multiplier 2
INV_2_32 = 2.3283064365386963e-10  # 2^-32 (fp32-exact power of two) -- D-UMAP
U_MIN = 5.9604644775390625e-08  # 2^-24 (fp32-exact) -- D-CLAMP lower
U_MAX = 0.9999999403953552  # 1-2^-24 (fp32-exact, largest fp32 < 1.0) -- D-CLAMP upper

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
    # Sort32 consumes ceil(count/32)*32 elements of the top-k source. tl::ascend::Sort()
    # pads the remainder itself with a MASKED AscendC::Duplicate, which does not take
    # effect on dav_c220 -> the trailing elements keep stale UB and get sorted into the
    # result (a huge stale float wins rank 0, shifting the whole merged list by one and
    # poisoning tau). Hand it an already-32-element-aligned count instead, so Sort()'s
    # padCount == 0 and the broken masked path is never taken; we pre-fill the pad with
    # -inf ourselves (an aligned full-buffer Duplicate, which does work).
    TAIL_PAD = ((TAIL + 31) // 32) * 32
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
                # S3-#1: vectorized tie counting (replaces 2x 256-iter scalar loops that
                # codegen'd to GetValue/SetValue + per-iter PipeBarrier<PIPE_ALL>).
                tie_mask = T.alloc_shared([K_MAX // 8], "uint8")  # bit-packed compare mask
                tie_one = T.alloc_shared([K_MAX], "float32")  # constant 1.0 tensor for select
                tie_tkbc = T.alloc_shared([K_MAX], "float32")  # tau_k broadcast to [K_MAX] (tensor-tensor compare)
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
                T.tile.fill(tie_one, 1.0)  # S3-#1: constant for tie select

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
                    # S3R2-#4: z_ub is rank-1 [BV] so it can serve BOTH as the GM->UB
                    # destination and as the top-k source. topk's sizing uses
                    # max_actual_num = sum(src.shape) (ascend_tile.py:635-645), so a
                    # rank-2 [1,BV] source would give 1+BV and over-read; rank-1 [BV]
                    # is exact. This removes the old per-block UB->UB hop into a
                    # dedicated zt_ub[BV] (16 KB UB + one V->V copy per block).
                    # Full-width rank-1 reduce stays on the NORMAL codegen path: the
                    # narrow-reduce branch is gated on a physical_row arg that the
                    # front end only attaches when row width != logical N
                    # (reduce_ascend.py:337-353 -> codegen_ascend.cc:2098), and here
                    # both are BV. Same shape K2g already uses (r4_gumbel_sample).
                    z_ub = T.alloc_shared([BV], "float32")  # GM->UB dst + top-k src
                    e_ub = T.alloc_shared([1, BV], "float32")
                    m2d = T.alloc_shared([1, BV], "float32")
                    for b in T.serial(n_blk_full):
                        T.copy(logits[row, b * BV], z_ub)
                        # (a) online softmax (T.tile.* only; no 1-elem UB->UB copy)
                        # rank-1 source reduces to an empty/[1] shape, so m_blk is
                        # passed as the region m_blk[0, 0:1] (reduce_ascend.py:130-172
                        # accepts it via _shape_can_embed_expected_shape).
                        T.reduce_max(z_ub, m_blk[0, 0:1], dim=-1)
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
                        # S3R2-#4: top-k reads z_ub directly (it is rank-1 [BV] now), so
                        # the per-block UB->UB hop is gone. z_ub's last read is the
                        # T.tile.sub above and it is overwritten by the GM->UB copy at
                        # the top of the next iteration, so topk's in-place -inf padding
                        # (N3) cannot poison anything. actual_num == BV == buffer size
                        # -> padCount == 0, so Sort()'s masked-Duplicate remainder path
                        # (broken on dav_c220) is never taken; BV % 32 == 0 also keeps
                        # every UB->UB byte count 32B-aligned by construction.
                        T.tile.topk(l_dst, z_ub, K_MAX, BV)
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
                    # top-k source: staged through a 32-element-aligned rank-2 buffer that
                    # the GM->UB copy -inf-pads itself (K2g's proven pad_value pattern), then
                    # hopped to a rank-1 buffer in exactly TAIL_PAD elements (a 32 B multiple,
                    # so AscendC::DataCopy cannot truncate it). See TAIL_PAD above.
                    ztp = T.alloc_shared([1, TAIL_PAD], "float32")
                    ztt = T.alloc_shared([TAIL_PAD], "float32")
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
                    T.copy(logits[row, n_blk_full * BV], ztp, pad_value=-T.infinity("float32"))
                    T.copy(ztp[0, :TAIL_PAD], ztt)
                    T.tile.topk(l_dst, ztt, K_MAX, TAIL_PAD)
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

                # D-TIE: n_gt / n_eq on the K_MAX list (S3-#1: vector compare+select+reduce).
                # Bitwise-identical to the prior `T.Parallel(K_MAX)` scalar fill: that fill
                # codegen'd to 256-iter GetValue/SetValue loops with a per-iter PipeBarrier<PIPE_ALL>
                # (K1_topk_stats.cc L169-193). fp32 GT/EQ are exact and the 0/1 counts (<=256) sum
                # exactly regardless of reduce order, so n_gt/n_eq (and tieok/vflag) are unchanged.
                # tau_k is broadcast to a [K_MAX] tensor so both compares use the tensor-tensor path
                # (two compare_scalar on the same tau_k[0] collide on a generated 'tau_k_scalar' var).
                T.tile.broadcast(tie_tkbc, tau_k, axis=0)  # [1] -> [K_MAX]
                T.tile.compare(tie_mask, run_v, tie_tkbc, "GT")  # run_v[l] > tau_k
                T.tile.select(tie_m, tie_mask, tie_one, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                T.reduce_sum(tie_m, n_gt, dim=-1)
                T.tile.compare(tie_mask, run_v, tie_tkbc, "EQ")  # run_v[l] == tau_k
                T.tile.select(tie_m, tie_mask, tie_one, 0.0, "VSEL_TENSOR_SCALAR_MODE")
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


@tilelang.jit(out_idx=[3], pass_configs=R4_PASS_CONFIGS)
def r4_gumbel_sample(B: int, V: int, BV: int):
    """K2 Gumbel path (V01-b rev1, FUSE_GUMBEL=True): z-domain Gumbel-max argmax -> tok_out[B] int32.

    Theorem T3 (DESIGN_v01b.md 1.3): y_i = argmax_{j in S_i}[z_ij + G_ij], S_i = {j : z_ij >= tau_i}.
    Does NOT read m_in/s_in (T3 removes normalization); reads logits + tau_in (keep mask) +
    rand (per-row hash seed seed_i, D-RAND Design C). Gumbel noise (rev1): counter-based avalanche
    hash state = lowbias32(seed_i*PRIME_W + j) -> u = clamp(state*2^-32 +0.5, 2^-24, 1-2^-24)
    -> G = -ln(-ln(u)). Block argmax via T.tile.topk(K=1); cross-block strict-GT merge =>
    smallest-index tie-break (== torch.argmax first, D-TIE-ARGMAX).

    rev1 RNG realization on dav-2201 (validated V-b1 bitwise vs golden_hash, debug_log.md):
      * int32 AscendC::And/Or are int16-only -> TileLang int32 bitwise_and consumes count as int16
        and processes only count/2 int32 ("halving"). Fix: OVER-ALLOCATE the and-operands
        (state_ub/h_ub/tmp_ub) to int32[2*BV] with real data in [0:BV); count=2*BV then covers
        all BV int32. int32 xor/not don't compile; scalar-and unregistered -> XOR is emulated as
        x^t = x + t - 2*(x&t) (add/and/mul/sub, all int32-native and full-count).
      * int32 ShiftRight is ARITHMETIC -> logical x>>>s via shift-trick a - ((a>>(32-s))<<(32-s)),
        a = x>>s (rshift/lshift/sub only; no mask-and needed).
      * HASH_M2 = 0x846CA68B -> int32 -2073254261 (DESIGN_v01b.md 3.3 decimal -2073151861 is a
        transcription error = 0x846E368B; golden r4_ref.py uses the correct hex).
    Gumbel/argmax/merge/tail unchanged from rev0 (V-b2..V-b6 validated). All scalars in [1]/[1,1]
    alloc_shared + T.tile.* (no scalar T.log/T.exp, no 1-elem UB->UB copy); no T.reduce_* (argmax
    via topk) so the V01-a narrow-reduce <=256B constraint is not triggered; tail shares full path.
    """
    n_blk = T.ceildiv(V, BV)  # static (>=1 always; no T.serial(0))
    b_num = T.ceildiv(B, VEC_NUM)
    AT1 = _aligned_topk(1)  # = 8 (K=1, fp32)
    BV2 = 2 * BV  # over-allocation: int16-native And processes count/2 int32, so [2*BV] covers BV

    @T.prim_func
    def main(
        logits: T.Tensor([B, V], "float32"),  # type: ignore  # in
        tau_in: T.Tensor([B], "float32"),  # type: ignore  # in (from K1)
        rand: T.Tensor([B], "int32"),  # type: ignore  # in (per-row hash seed seed_i; D-RAND)
        tok_out: T.Tensor([B], "int32"),  # type: ignore  # out
    ):
        with T.Kernel(b_num, is_npu=True) as (cid, vid):
            row = cid * VEC_NUM + vid
            if row < B:
                # ---- all alloc_shared (pure Vector -> shared.ub); rank-1 [BV] topk source (N2) ----
                z_ub = T.alloc_shared([BV], "float32")  # logits block -> in-place cand
                # hash buffers OVER-ALLOCATED [2*BV] (int16-native And halving workaround); real data [0:BV)
                state_ub = T.alloc_shared([BV2], "int32")  # hash working value x (in-place avalanche)
                h_ub = T.alloc_shared([BV2], "int32")  # logical-rshift result t (=x>>>s); then a
                tmp_ub = T.alloc_shared([BV2], "int32")  # shift-trick temp; then p = x&t
                g_ub = T.alloc_shared([BV], "float32")  # Gumbel noise (cast dst + in-place chain)
                mask = T.alloc_shared([BV // 8], "uint8")  # keep mask (compare GE); reused as EQ mask (S3-#2)
                # S3-#2: block argmax via reduce_max + reduce_min-of-index (replaces topk(K=1) full
                # sort(4096)). idx = loop-invariant local index vector; g_ub (dead after z+=G) is
                # reused as the masked-index buffer, mask (dead after keep-select) as the EQ mask.
                idx = T.alloc_shared([BV], "float32")  # [0..BV-1] local indices (fp32 exact, BV<2^24)
                blk_max = T.alloc_shared([1], "float32")  # per-block max value (reduce_max)
                blk_argmax = T.alloc_shared([1], "float32")  # per-block smallest argmax index (reduce_min)
                t_v = T.alloc_shared([1, 1], "float32")  # tau_i
                seed_v = T.alloc_shared([1], "int32")  # seed_i
                seedP_v = T.alloc_shared([1], "int32")  # seed_i * PRIME_W (per row, once)
                best_val = T.alloc_shared([1], "float32")  # running argmax value
                best_idx = T.alloc_shared([1], "float32")  # running argmax global index (<2^24 exact)
                tok_i32 = T.alloc_shared([1], "int32")  # output scalar

                # Touch every scalar buffer with a Vector op (scope inference; debug_log.md finding 3).
                T.tile.fill(best_val, -T.infinity("float32"))
                T.tile.fill(best_idx, 0.0)
                T.tile.fill(tok_i32, 0)
                T.tile.fill(seedP_v, 0)
                T.copy(tau_in[row : row + 1], t_v[0, 0:1])  # GM->UB 1-elem (normal)
                T.copy(rand[row : row + 1], seed_v)
                T.tile.mul(seedP_v, seed_v, PRIME_W)  # G0: seedP = seed_i*PRIME_W (int32 wrap)
                T.tile.arith_progression(idx, 0.0, 1.0, BV)  # S3-#2: idx[j]=j (hoisted, loop-invariant)

                for b in T.serial(n_blk):
                    # (0) load logits block (tail auto-clamp + pad -inf; keep mask excludes pad)
                    T.copy(logits[row, b * BV], z_ub, pad_value=-T.infinity("float32"))
                    # (a) rev1 avalanche hash: x = seed_i*PRIME_W + global j, then lowbias32.
                    #     Over-alloc [2*BV]: arith_progression fills [0:2*BV); real data is [0:BV).
                    T.tile.fill(state_ub, 0)
                    T.tile.arith_progression(state_ub, seedP_v[0] + b * BV, 1, BV2)
                    # lowbias32 = 3x[x ^= x>>>s; x *= M]; >>> via shift-trick (rshift/lshift/sub),
                    # ^ via x+t-2*(x&t) with the over-allocated int32 And (covers [0:BV) at count 2*BV).
                    # round 1: s=16, *M1
                    T.tile.bitwise_rshift(h_ub, state_ub, 16)  # a = x>>16 (arithmetic)
                    T.tile.bitwise_rshift(tmp_ub, h_ub, 16)  # a>>16
                    T.tile.bitwise_lshift(tmp_ub, tmp_ub, 16)  # (a>>16)<<16
                    T.tile.sub(h_ub, h_ub, tmp_ub)  # h = x>>>16 (logical) = t
                    T.tile.bitwise_and(tmp_ub, state_ub, h_ub)  # tmp = x & t (over-alloc And)
                    T.tile.add(state_ub, state_ub, h_ub)  # x = x + t
                    T.tile.mul(tmp_ub, tmp_ub, 2)  # tmp = 2*(x&t)
                    T.tile.sub(state_ub, state_ub, tmp_ub)  # x = x^t (emulated XOR)
                    T.tile.mul(state_ub, state_ub, HASH_M1)  # x *= M1 (int32 wrap)
                    # round 2: s=15, *M2
                    T.tile.bitwise_rshift(h_ub, state_ub, 15)
                    T.tile.bitwise_rshift(tmp_ub, h_ub, 17)
                    T.tile.bitwise_lshift(tmp_ub, tmp_ub, 17)
                    T.tile.sub(h_ub, h_ub, tmp_ub)
                    T.tile.bitwise_and(tmp_ub, state_ub, h_ub)
                    T.tile.add(state_ub, state_ub, h_ub)
                    T.tile.mul(tmp_ub, tmp_ub, 2)
                    T.tile.sub(state_ub, state_ub, tmp_ub)
                    T.tile.mul(state_ub, state_ub, HASH_M2)  # x *= M2 (0x846CA68B)
                    # round 3: s=16, no mul
                    T.tile.bitwise_rshift(h_ub, state_ub, 16)
                    T.tile.bitwise_rshift(tmp_ub, h_ub, 16)
                    T.tile.bitwise_lshift(tmp_ub, tmp_ub, 16)
                    T.tile.sub(h_ub, h_ub, tmp_ub)
                    T.tile.bitwise_and(tmp_ub, state_ub, h_ub)
                    T.tile.add(state_ub, state_ub, h_ub)
                    T.tile.mul(tmp_ub, tmp_ub, 2)
                    T.tile.sub(state_ub, state_ub, tmp_ub)
                    # (b) Gumbel: u = clamp(state*2^-32+0.5), G = -ln(-ln(u)); cast real half [0:BV)
                    T.tile.cast(g_ub, state_ub, "CAST_RINT", BV)  # int32[0:BV) -> fp32
                    T.tile.mul(g_ub, g_ub, INV_2_32)
                    T.tile.add(g_ub, g_ub, 0.5)
                    T.tile.clamp(g_ub, g_ub, U_MIN, U_MAX, BV)  # u in [2^-24, 1-2^-24]
                    T.tile.ln(g_ub, g_ub)  # ln(u)  (u in (0,1) -> negative)
                    T.tile.mul(g_ub, g_ub, -1.0)  # -ln(u) (positive)
                    T.tile.ln(g_ub, g_ub)  # ln(-ln(u))
                    T.tile.mul(g_ub, g_ub, -1.0)  # G = -ln(-ln(u))
                    # (c) keep mask + z-domain candidate (theorem T3); compare BEFORE add uses raw z
                    T.tile.compare(mask, z_ub, t_v[0, 0], "GE")  # scalar src1 -> compare_scalar (V-b5)
                    T.tile.add(z_ub, z_ub, g_ub)  # z + G (in-place -> cand)
                    T.tile.select(z_ub, mask, z_ub, -T.infinity("float32"), "VSEL_TENSOR_SCALAR_MODE")
                    # (d) block argmax-with-index (S3-#2: reduce_max + reduce_min-of-index; replaces
                    #     topk(K=1) full sort(4096) that only extracted 1 max). Bitwise-exact vs prior:
                    #     max is exact (no rounding); reduce_min over {j : z==max} yields the SMALLEST
                    #     index == torch.argmax-first == the topk(K=1) tie behavior validated in V-b4.
                    #     Full-width [BV] reduce (normal path, not the <=256B narrow-reduce constraint).
                    T.reduce_max(z_ub, blk_max, dim=-1)  # block max value
                    T.tile.compare(mask, z_ub, blk_max[0], "EQ")  # positions == max (reuse mask buf)
                    T.tile.select(g_ub, mask, idx, float(BV), "VSEL_TENSOR_SCALAR_MODE")  # idx where max else BV(sentinel)
                    T.reduce_min(g_ub, blk_argmax, dim=-1)  # smallest max-index (sentinel BV > any index)
                    # (e) cross-block merge (strict GT -> tie keeps earlier block = smaller index)
                    with T.If(blk_max[0] > best_val[0]):  # noqa: SIM117
                        with T.Then():
                            best_val[0] = blk_max[0]
                            best_idx[0] = blk_argmax[0] + b * BV  # local -> global (fp32 exact)
                # write GM (best_idx fp32 -> int32; 1-elem UB->GM copy is normal)
                T.tile.cast(tok_i32, best_idx, "CAST_ROUND", 1)
                T.copy(tok_i32, tok_out[row : row + 1])

    return main


def r4_fused_sampling(
    logits,
    top_ks,
    top_ps,
    min_ps,
    need_min_p_sampling,
    K_MAX=256,
    BV=4096,
    rand=None,
    FUSE_GUMBEL=False,
):
    """Host wrapper: pre-checks (C3/C6/C7) + K1 + K2 (dispatched by FUSE_GUMBEL).

    FUSE_GUMBEL=False (V01-a, default): K1 + K2 r4_mask_sample -> returns
        (m_out, s_out, tau_out, valid, w_out). torch.multinomial(w_out, 1) is the
        user-retained downstream sampler (DEV-1); not invoked here. UNCHANGED.
    FUSE_GUMBEL=True (V01-b): K1 + K2 r4_gumbel_sample -> returns
        (m_out, s_out, tau_out, valid, tok_out). tok_out[B] int32 is the sampled token id
        (Gumbel-max fused; eliminates host torch.multinomial). Requires rand[B] int32
        (per-row avalanche-hash seed seed_i). Rows with valid==0 are the host's fallback signal.

    NOTE: rand/FUSE_GUMBEL are appended AFTER (K_MAX, BV) -- not before K_MAX as in the
    DESIGN_v01b.md 3.3 sketch -- to preserve V01-a positional calls
    r4_fused_sampling(logits, ks, ps, mus, need_min_p, K_MAX, BV) (R-K1-REGRESSION).
    V01-b callers pass rand=/FUSE_GUMBEL= by keyword. Logical param set matches proto.yaml.
    """
    import torch

    B, V = logits.shape
    # C6: B == 0 -> host intercept, do not launch (zero-sized grid aborts).
    if B == 0:
        dev = logits.device
        empty_b = torch.empty((0,), dtype=torch.float32, device=dev)
        empty_i = torch.empty((0,), dtype=torch.int32, device=dev)
        if FUSE_GUMBEL:
            return empty_b, empty_b, empty_b, empty_i, empty_i  # (..., tok_out empty)
        empty_w = torch.empty((0, V), dtype=torch.float32, device=dev)
        return empty_b, empty_b, empty_b, empty_i, empty_w
    # C7: V too small for top-K_MAX.
    if V < K_MAX:
        raise NotImplementedError(f"R4: V={V} < K_MAX={K_MAX}; fall back to torch path")
    # C3: topk needs K_MAX <= actual_num on every block incl. tail.
    tail = V % BV
    if tail != 0 and tail < K_MAX:
        raise NotImplementedError(f"R4: tail block {tail} < K_MAX={K_MAX} (need V%BV==0 or V%BV>=K_MAX); fall back to torch path")
    if logits.dtype != torch.float32:
        raise NotImplementedError(f"R4 supports float32 only; got {logits.dtype}")

    k1 = r4_topk_stats(B, V, K_MAX, BV, bool(need_min_p_sampling))
    m_out, s_out, tau_out, valid = k1(logits, top_ks, top_ps, min_ps)
    if not FUSE_GUMBEL:
        # V01-a path (UNCHANGED): K2 r4_mask_sample -> w_out; host retains torch.multinomial.
        k2 = r4_mask_sample(B, V, BV)
        w_out = k2(logits, m_out, s_out, tau_out)
        return m_out, s_out, tau_out, valid, w_out
    # V01-b path: K2 r4_gumbel_sample -> tok_out (Gumbel-max fused; no multinomial).
    if rand is None:
        raise ValueError("R4 V01-b FUSE_GUMBEL=True requires rand[B] int32 (per-row avalanche-hash seed)")
    if rand.dtype != torch.int32:
        raise NotImplementedError(f"R4 V01-b rand must be int32; got {rand.dtype}")
    k2g = r4_gumbel_sample(B, V, BV)
    tok_out = k2g(logits, tau_out, rand)
    return m_out, s_out, tau_out, valid, tok_out
