#!/usr/bin/python3
# coding=utf-8
"""Round-2 item-2 probe: block-level threshold pruning (point-5 branch form + overhead).

Defines four K1 variants so that part (b) of the feasibility study (Sort/topk+merge vs
online-softmax cost split) and section-2 point 5 (can a block-internal data-dependent
branch actually SKIP the Sort on Ascend?) can be answered with evidence:

  * k1_real       : verbatim call of the delivered r4_topk_stats (baseline).
  * k1_nosort     : same, minus the per-block topk+merge  -> Delta(real-nosort) = topk+merge cost.
  * k1_nosoftmax  : same, minus the per-block online-softmax -> additivity cross-check.
  * k1_pruned     : topk+merge wrapped in `with T.If(block_max >= tau_k_run)` (STRICT-LT prune).
                    Compiled ONLY to inspect the emitted C++ (point 5); also timed to price the
                    predicate overhead that a 0%-prune workload would still have to pay.

The three probe bodies are copies of the delivered K1 with exactly one region removed, so
everything else (buffers, epilogue, tail path, pass_configs) is identical by construction.
"""

import tilelang
from tilelang import language as T

R4_PASS_CONFIGS = {
    "tl.ascend_auto_sync": True,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cv_combine": True,
    "tl.ascend_auto_cross_core_sync": True,
}
VEC_NUM = 2


def _aligned_topk(K: int, elems_per_block: int = 8) -> int:
    return ((2 * K + elems_per_block - 1) // elems_per_block) * elems_per_block


# -------------------------------------------------------------------------- nosort
@tilelang.jit(out_idx=[4, 5, 6, 7], pass_configs=R4_PASS_CONFIGS)
def k1_pruned(B: int, V: int, K_MAX: int, BV: int, NEED_MIN_P: bool):
    """K1 with the per-block topk+merge guarded by a data-dependent block-level threshold.

    prune  <=>  block_max < tau_k_run   (STRICT LT -- see feasibility_study.md)
    keep   <=>  block_max >= tau_k_run  (the form actually written, so no T.Else is needed)
    tau_k_run = run_v[K_MAX-1] = the K_MAX-th largest of all blocks seen so far.
    """
    n_blk_full = V // BV
    TAIL = V - n_blk_full * BV
    TAIL_PAD = ((TAIL + 31) // 32) * 32
    b_num = T.ceildiv(B, VEC_NUM)
    AT = _aligned_topk(K_MAX)

    @T.prim_func
    def main(
        logits: T.Tensor([B, V], "float32"),  # type: ignore
        top_ks: T.Tensor([B], "int32"),  # type: ignore
        top_ps: T.Tensor([B], "float32"),  # type: ignore
        min_ps: T.Tensor([B], "float32"),  # type: ignore
        m_out: T.Tensor([B], "float32"),  # type: ignore
        s_out: T.Tensor([B], "float32"),  # type: ignore
        tau_out: T.Tensor([B], "float32"),  # type: ignore
        valid: T.Tensor([B], "int32"),  # type: ignore
    ):
        with T.Kernel(b_num, is_npu=True) as (cid, vid):
            row = cid * VEC_NUM + vid
            if row < B:
                m_blk = T.alloc_shared([1, 1], "float32")
                m_run = T.alloc_shared([1, 1], "float32")
                m_old = T.alloc_shared([1, 1], "float32")
                s_blk = T.alloc_shared([1, 1], "float32")
                s_run = T.alloc_shared([1, 1], "float32")
                tmp = T.alloc_shared([1, 1], "float32")
                l_dst = T.alloc_shared([AT], "float32")
                l_val = T.alloc_shared([K_MAX], "float32")
                merged = T.alloc_shared([2 * K_MAX], "float32")
                m_dst = T.alloc_shared([AT], "float32")
                run_v = T.alloc_shared([K_MAX], "float32")
                rv2 = T.alloc_shared([1, K_MAX], "float32")
                q2 = T.alloc_shared([1, K_MAX], "float32")
                mk2 = T.alloc_shared([1, K_MAX], "float32")
                sk2 = T.alloc_shared([1, K_MAX], "float32")
                tie_m = T.alloc_shared([K_MAX], "float32")
                tie_mask = T.alloc_shared([K_MAX // 8], "uint8")
                tie_one = T.alloc_shared([K_MAX], "float32")
                tie_tkbc = T.alloc_shared([K_MAX], "float32")
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

                T.tile.fill(tau_k, 0.0)
                T.tile.fill(tau_p, 0.0)
                T.tile.fill(prefix, 0.0)
                T.tile.fill(tau_u, 0.0)
                T.tile.fill(okp, 0)
                T.tile.fill(tieok, 0)
                T.tile.fill(vflag, 0)
                T.tile.fill(tie_m, 0.0)
                T.tile.fill(tie_one, 1.0)
                # running top-K threshold (K_MAX-th largest of everything seen so far)
                tau_run = T.alloc_shared([1, 1], "float32")
                T.tile.fill(tau_run, -T.infinity("float32"))

                T.copy(top_ks[row : row + 1], k_i)
                T.copy(top_ps[row : row + 1], p_i[0, 0:1])
                T.copy(min_ps[row : row + 1], mu_i[0, 0:1])

                T.tile.fill(m_run, -T.infinity("float32"))
                T.tile.fill(s_run, 0.0)
                T.tile.fill(run_v, -T.infinity("float32"))

                if n_blk_full > 0:
                    z_ub = T.alloc_shared([BV], "float32")
                    e_ub = T.alloc_shared([1, BV], "float32")
                    m2d = T.alloc_shared([1, BV], "float32")
                    for b in T.serial(n_blk_full):
                        T.copy(logits[row, b * BV], z_ub)
                        T.reduce_max(z_ub, m_blk[0, 0:1], dim=-1)
                        T.tile.max(m_old, m_run, m_blk)
                        T.tile.sub(tmp, m_run, m_old)
                        T.tile.exp(tmp, tmp)
                        T.tile.mul(s_run, s_run, tmp)
                        T.tile.max(m_run, m_run, m_blk)
                        T.tile.broadcast(m2d, m_run)
                        T.tile.sub(e_ub, z_ub, m2d)
                        T.tile.exp(e_ub, e_ub)
                        T.reduce_sum(e_ub, s_blk, dim=-1)
                        T.tile.add(s_run, s_run, s_blk)
                        # (b) block top-K_MAX + cross-block merge, threshold-pruned.
                        #     m_blk already holds block_max from the reduce_max above, so the
                        #     predicate itself costs no extra reduction.
                        with T.If(m_blk[0, 0] >= tau_run[0, 0]):  # keep <=> NOT(block_max < tau)
                            with T.Then():
                                T.tile.topk(l_dst, z_ub, K_MAX, BV)
                                T.tile.gather_mask(l_val, l_dst, "P0101")
                                T.copy(run_v, merged[:K_MAX])
                                T.copy(l_val, merged[K_MAX:])
                                T.tile.topk(m_dst, merged, K_MAX, 2 * K_MAX)
                                T.tile.gather_mask(run_v, m_dst, "P0101")
                                tau_run[0, 0] = run_v[K_MAX - 1]  # monotone non-decreasing

                if TAIL > 0:
                    zt = T.alloc_shared([1, TAIL], "float32")
                    et = T.alloc_shared([1, TAIL], "float32")
                    mt = T.alloc_shared([1, TAIL], "float32")
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

                T.copy(run_v, rv2[0, :])
                T.tile.broadcast(mk2, m_run, axis=1)
                T.tile.sub(q2, rv2, mk2)
                T.tile.exp(q2, q2)
                T.tile.broadcast(sk2, s_run, axis=1)
                T.tile.div(q2, q2, sk2)

                T.tile.fill(tau_k, -T.infinity("float32"))
                for _ in T.serial(1):
                    if k_i[0] >= 1 and k_i[0] <= K_MAX:
                        tau_k[0] = run_v[k_i[0] - 1]

                T.tile.fill(tau_u, -T.infinity("float32"))
                if NEED_MIN_P:
                    T.tile.max(muc, mu_i, 1e-30)
                    T.tile.ln(lnmu, muc)
                    T.tile.add(tau_u, m_run, lnmu)
                    for _ in T.serial(1):
                        if mu_i[0, 0] <= 0.0:
                            tau_u[0, 0] = -T.infinity("float32")

                T.tile.fill(tau_p, -T.infinity("float32"))
                T.tile.fill(prefix, 0.0)
                for l in T.serial(K_MAX):
                    with T.If(prefix[0, 0] <= p_i[0, 0]):  # noqa: SIM117
                        with T.Then():
                            tau_p[0] = run_v[l]
                            prefix[0, 0] = prefix[0, 0] + q2[0, l]

                for _ in T.serial(1):
                    okp[0] = T.if_then_else(p_i[0, 0] >= 1.0, 1, T.if_then_else(prefix[0, 0] > p_i[0, 0], 1, 0))

                T.tile.broadcast(tie_tkbc, tau_k, axis=0)
                T.tile.compare(tie_mask, run_v, tie_tkbc, "GT")
                T.tile.select(tie_m, tie_mask, tie_one, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                T.reduce_sum(tie_m, n_gt, dim=-1)
                T.tile.compare(tie_mask, run_v, tie_tkbc, "EQ")
                T.tile.select(tie_m, tie_mask, tie_one, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                T.reduce_sum(tie_m, n_eq, dim=-1)

                T.tile.max(tau, tau_k, tau_p)
                T.tile.max(tau, tau, tau_u[0, 0:1])

                for _ in T.serial(1):
                    tieok[0] = T.if_then_else(
                        tau[0] > tau_k[0], 1,
                        T.if_then_else(
                            n_gt[0] + n_eq[0] > T.Cast("float32", k_i[0]), 0,
                            T.if_then_else(n_gt[0] + n_eq[0] < T.Cast("float32", K_MAX), 1,
                                           T.if_then_else(n_eq[0] <= 1.0, 1, 0))))
                    vflag[0] = T.if_then_else(
                        (k_i[0] >= 1 and k_i[0] <= K_MAX),
                        T.if_then_else(okp[0] == 1 and tieok[0] == 1, 1, 0), 0)

                T.copy(m_run[0, 0:1], m_out[row : row + 1])
                T.copy(s_run[0, 0:1], s_out[row : row + 1])
                T.copy(tau, tau_out[row : row + 1])
                T.copy(vflag, valid[row : row + 1])

    return main
