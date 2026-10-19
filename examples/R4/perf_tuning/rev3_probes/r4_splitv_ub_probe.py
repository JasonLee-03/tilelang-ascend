"""Stage-1 UB/compile feasibility probe for the R4 split-V redesign.

Defines PROTOTYPE kernels (never written into examples/R4/) mirroring the proposed
rev3 architecture, then compiles them with get_kernel_source() and measures UB via
the same method as perf_tuning/round2_bv_probe/ub_probe.py (parse the memory
planner's ascend_ub.GetWithOffset<T>(n, off); total = max(off + n*sizeof(T))).

Nothing here is a deliverable; it exists so DESIGN.md rev3 can quote MEASURED
(not modelled) UB numbers, per the round2 lesson that the analytic model under-counts.
"""
import re
import sys

sys.path.insert(0, '/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4')

import tilelang
from tilelang import language as T
from r4_fused_sampling import (R4_PASS_CONFIGS, VEC_NUM, PRIME_W, HASH_M1, HASH_M2,
                               INV_2_32, U_MIN, U_MAX, _aligned_topk)

_SZ = {'float': 4, 'int': 4, 'uint8_t': 1, 'int32_t': 4, 'float16_t': 2,
       'uint16_t': 2, 'int16_t': 2, 'uint32_t': 4, 'int8_t': 1, 'half': 2}
_PAT = re.compile(r'ascend_ub\.GetWithOffset<\s*([A-Za-z0-9_:]+)\s*>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)')
UB_LIMIT = 196352


def ub_usage(src):
    det = []
    for m in _PAT.finditer(src):
        dt, n, off = m.group(1), int(m.group(2)), int(m.group(3))
        key = dt.split('::')[-1]
        esz = _SZ.get(key)
        if esz is None:
            raise SystemExit(f'unknown dtype {dt}')
        det.append((off, n * esz, dt, n))
    if not det:
        raise SystemExit('no GetWithOffset -- parse failure')
    region = re.search(r'InitBuffer\(ascend_ub,\s*(\d+)\)', src)
    return max(o + s for o, s, _, _ in det), len(det), region.group(1) if region else None


# ---------------------------------------------------------------- split geometry
def split_geom(V, BV, b_num, cube=20):
    """Python-level (compile-time) split geometry -- exactly what the host wrapper does."""
    n_blk_full = V // BV
    TAIL = V - n_blk_full * BV
    n_blk_total = n_blk_full + (1 if TAIL > 0 else 0)
    S_cap = cube // b_num                      # stay inside ONE wave
    S = max(1, min(S_cap, n_blk_total))
    n_blk_split = -(-n_blk_total // S)         # ceil
    S = -(-n_blk_total // n_blk_split)         # shrink so no split is empty
    n_blk_split = -(-n_blk_total // S)
    return dict(n_blk_full=n_blk_full, TAIL=TAIL, TAIL_PAD=((TAIL + 31) // 32) * 32,
                n_blk_total=n_blk_total, S=S, n_blk_split=n_blk_split,
                b_num=b_num, grid=b_num * S)


# =============================================== K1A: topk-only, split-parallel
@tilelang.jit(pass_configs=R4_PASS_CONFIGS)
def k1a_topk_split(B: int, V: int, K_MAX: int, BV: int, S: int, NBS: int):
    """Per (row, split): hierarchical top-K over the split's blocks -> pv[row*S+s, :K_MAX]."""
    n_blk_full = V // BV
    TAIL = V - n_blk_full * BV
    TAIL_PAD = ((TAIL + 31) // 32) * 32
    b_num = T.ceildiv(B, VEC_NUM)
    AT = _aligned_topk(K_MAX)
    grid = b_num * S

    @T.prim_func
    def main(
        logits: T.Tensor([B, V], "float32"),   # type: ignore
        pv: T.Tensor([B * S, K_MAX], "float32"),  # type: ignore  # out (caller-alloc)
    ):
        with T.Kernel(grid, is_npu=True) as (cid, vid):
            task = cid * VEC_NUM + vid
            row = task // S
            sid = task % S
            if row < B:
                l_dst = T.alloc_shared([AT], "float32")
                l_val = T.alloc_shared([K_MAX], "float32")
                merged = T.alloc_shared([2 * K_MAX], "float32")
                m_dst = T.alloc_shared([AT], "float32")
                run_v = T.alloc_shared([K_MAX], "float32")
                T.tile.fill(run_v, -T.infinity("float32"))

                if n_blk_full > 0:
                    z_ub = T.alloc_shared([BV], "float32")
                    for lb in T.serial(NBS):
                        b = sid * NBS + lb
                        if b < n_blk_full:
                            T.copy(logits[row, b * BV], z_ub)
                            T.tile.topk(l_dst, z_ub, K_MAX, BV)
                            T.tile.gather_mask(l_val, l_dst, "P0101")
                            T.copy(run_v, merged[:K_MAX])
                            T.copy(l_val, merged[K_MAX:])
                            T.tile.topk(m_dst, merged, K_MAX, 2 * K_MAX)
                            T.tile.gather_mask(run_v, m_dst, "P0101")

                if TAIL > 0:
                    ztp = T.alloc_shared([1, TAIL_PAD], "float32")
                    ztt = T.alloc_shared([TAIL_PAD], "float32")
                    for lb in T.serial(NBS):
                        b = sid * NBS + lb
                        if b == n_blk_full:
                            T.copy(logits[row, n_blk_full * BV], ztp,
                                   pad_value=-T.infinity("float32"))
                            T.copy(ztp[0, :TAIL_PAD], ztt)
                            T.tile.topk(l_dst, ztt, K_MAX, TAIL_PAD)
                            T.tile.gather_mask(l_val, l_dst, "P0101")
                            T.copy(run_v, merged[:K_MAX])
                            T.copy(l_val, merged[K_MAX:])
                            T.tile.topk(m_dst, merged, K_MAX, 2 * K_MAX)
                            T.tile.gather_mask(run_v, m_dst, "P0101")

                T.copy(run_v, pv[row * S + sid, :])
    return main


# =========================== K1B: softmax scan (UNCHANGED) + cross-split merge + epilogue
@tilelang.jit(out_idx=[5, 6, 7, 8], pass_configs=R4_PASS_CONFIGS)
def k1b_softmax_merge(B: int, V: int, K_MAX: int, BV: int, S: int, NEED_MIN_P: bool):
    n_blk_full = V // BV
    TAIL = V - n_blk_full * BV
    TAIL_PAD = ((TAIL + 31) // 32) * 32
    b_num = T.ceildiv(B, VEC_NUM)
    AT = _aligned_topk(K_MAX)

    @T.prim_func
    def main(
        logits: T.Tensor([B, V], "float32"),     # type: ignore
        top_ks: T.Tensor([B], "int32"),          # type: ignore
        top_ps: T.Tensor([B], "float32"),        # type: ignore
        min_ps: T.Tensor([B], "float32"),        # type: ignore
        pv: T.Tensor([B * S, K_MAX], "float32"),  # type: ignore  # in (from K1A)
        m_out: T.Tensor([B], "float32"),         # type: ignore
        s_out: T.Tensor([B], "float32"),         # type: ignore
        tau_out: T.Tensor([B], "float32"),       # type: ignore
        valid: T.Tensor([B], "int32"),           # type: ignore
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
                merged = T.alloc_shared([2 * K_MAX], "float32")
                m_dst = T.alloc_shared([AT], "float32")
                run_v = T.alloc_shared([K_MAX], "float32")
                pvb = T.alloc_shared([K_MAX], "float32")
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
                T.copy(top_ks[row: row + 1], k_i)
                T.copy(top_ps[row: row + 1], p_i[0, 0:1])
                T.copy(min_ps[row: row + 1], mu_i[0, 0:1])
                T.tile.fill(m_run, -T.infinity("float32"))
                T.tile.fill(s_run, 0.0)
                T.tile.fill(run_v, -T.infinity("float32"))

                # ---- softmax side: BYTE-IDENTICAL op sequence to the delivered kernel ----
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
                if TAIL > 0:
                    zt = T.alloc_shared([1, TAIL], "float32")
                    et = T.alloc_shared([1, TAIL], "float32")
                    mt = T.alloc_shared([1, TAIL], "float32")
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

                # ---- cross-split top-K merge, split-id order (S-1 merges of 2*K_MAX) ----
                for s in T.serial(S):
                    T.copy(pv[row * S + s, :], pvb)
                    T.copy(run_v, merged[:K_MAX])
                    T.copy(pvb, merged[K_MAX:])
                    T.tile.topk(m_dst, merged, K_MAX, 2 * K_MAX)
                    T.tile.gather_mask(run_v, m_dst, "P0101")

                # ---- epilogue: UNCHANGED from the delivered kernel ----
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
                    okp[0] = T.if_then_else(p_i[0, 0] >= 1.0, 1,
                                            T.if_then_else(prefix[0, 0] > p_i[0, 0], 1, 0))
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
                T.copy(m_run[0, 0:1], m_out[row: row + 1])
                T.copy(s_run[0, 0:1], s_out[row: row + 1])
                T.copy(tau, tau_out[row: row + 1])
                T.copy(vflag, valid[row: row + 1])
    return main


# =============================================== K2gA: gumbel argmax, split-parallel
@tilelang.jit(pass_configs=R4_PASS_CONFIGS)
def k2ga_gumbel_split(B: int, V: int, BV: int, S: int, NBS: int):
    n_blk = T.ceildiv(V, BV)
    b_num = T.ceildiv(B, VEC_NUM)
    BV2 = 2 * BV
    grid = b_num * S

    @T.prim_func
    def main(
        logits: T.Tensor([B, V], "float32"),   # type: ignore
        tau_in: T.Tensor([B], "float32"),      # type: ignore
        rand: T.Tensor([B], "int32"),          # type: ignore
        pv_val: T.Tensor([B * S], "float32"),  # type: ignore  # out
        pv_idx: T.Tensor([B * S], "float32"),  # type: ignore  # out
    ):
        with T.Kernel(grid, is_npu=True) as (cid, vid):
            task = cid * VEC_NUM + vid
            row = task // S
            sid = task % S
            if row < B:
                z_ub = T.alloc_shared([BV], "float32")
                state_ub = T.alloc_shared([BV2], "int32")
                h_ub = T.alloc_shared([BV2], "int32")
                tmp_ub = T.alloc_shared([BV2], "int32")
                g_ub = T.alloc_shared([BV], "float32")
                mask = T.alloc_shared([BV // 8], "uint8")
                idx = T.alloc_shared([BV], "float32")
                blk_max = T.alloc_shared([1], "float32")
                blk_argmax = T.alloc_shared([1], "float32")
                t_v = T.alloc_shared([1, 1], "float32")
                seed_v = T.alloc_shared([1], "int32")
                seedP_v = T.alloc_shared([1], "int32")
                best_val = T.alloc_shared([1], "float32")
                best_idx = T.alloc_shared([1], "float32")

                T.tile.fill(best_val, -T.infinity("float32"))
                T.tile.fill(best_idx, 0.0)
                T.tile.fill(seedP_v, 0)
                T.copy(tau_in[row: row + 1], t_v[0, 0:1])
                T.copy(rand[row: row + 1], seed_v)
                T.tile.mul(seedP_v, seed_v, PRIME_W)
                T.tile.arith_progression(idx, 0.0, 1.0, BV)

                for lb in T.serial(NBS):
                    b = sid * NBS + lb
                    if b < n_blk:
                        T.copy(logits[row, b * BV], z_ub, pad_value=-T.infinity("float32"))
                        T.tile.fill(state_ub, 0)
                        T.tile.arith_progression(state_ub, seedP_v[0] + b * BV, 1, BV2)
                        T.tile.bitwise_rshift(h_ub, state_ub, 16)
                        T.tile.bitwise_rshift(tmp_ub, h_ub, 16)
                        T.tile.bitwise_lshift(tmp_ub, tmp_ub, 16)
                        T.tile.sub(h_ub, h_ub, tmp_ub)
                        T.tile.bitwise_and(tmp_ub, state_ub, h_ub)
                        T.tile.add(state_ub, state_ub, h_ub)
                        T.tile.mul(tmp_ub, tmp_ub, 2)
                        T.tile.sub(state_ub, state_ub, tmp_ub)
                        T.tile.mul(state_ub, state_ub, HASH_M1)
                        T.tile.bitwise_rshift(h_ub, state_ub, 15)
                        T.tile.bitwise_rshift(tmp_ub, h_ub, 17)
                        T.tile.bitwise_lshift(tmp_ub, tmp_ub, 17)
                        T.tile.sub(h_ub, h_ub, tmp_ub)
                        T.tile.bitwise_and(tmp_ub, state_ub, h_ub)
                        T.tile.add(state_ub, state_ub, h_ub)
                        T.tile.mul(tmp_ub, tmp_ub, 2)
                        T.tile.sub(state_ub, state_ub, tmp_ub)
                        T.tile.mul(state_ub, state_ub, HASH_M2)
                        T.tile.bitwise_rshift(h_ub, state_ub, 16)
                        T.tile.bitwise_rshift(tmp_ub, h_ub, 16)
                        T.tile.bitwise_lshift(tmp_ub, tmp_ub, 16)
                        T.tile.sub(h_ub, h_ub, tmp_ub)
                        T.tile.bitwise_and(tmp_ub, state_ub, h_ub)
                        T.tile.add(state_ub, state_ub, h_ub)
                        T.tile.mul(tmp_ub, tmp_ub, 2)
                        T.tile.sub(state_ub, state_ub, tmp_ub)
                        T.tile.cast(g_ub, state_ub, "CAST_RINT", BV)
                        T.tile.mul(g_ub, g_ub, INV_2_32)
                        T.tile.add(g_ub, g_ub, 0.5)
                        T.tile.clamp(g_ub, g_ub, U_MIN, U_MAX, BV)
                        T.tile.ln(g_ub, g_ub)
                        T.tile.mul(g_ub, g_ub, -1.0)
                        T.tile.ln(g_ub, g_ub)
                        T.tile.mul(g_ub, g_ub, -1.0)
                        T.tile.compare(mask, z_ub, t_v[0, 0], "GE")
                        T.tile.add(z_ub, z_ub, g_ub)
                        T.tile.select(z_ub, mask, z_ub, -T.infinity("float32"),
                                      "VSEL_TENSOR_SCALAR_MODE")
                        T.reduce_max(z_ub, blk_max, dim=-1)
                        T.tile.compare(mask, z_ub, blk_max[0], "EQ")
                        T.tile.select(g_ub, mask, idx, float(BV), "VSEL_TENSOR_SCALAR_MODE")
                        T.reduce_min(g_ub, blk_argmax, dim=-1)
                        with T.If(blk_max[0] > best_val[0]):  # noqa: SIM117
                            with T.Then():
                                best_val[0] = blk_max[0]
                                best_idx[0] = blk_argmax[0] + b * BV
                T.copy(best_val, pv_val[row * S + sid: row * S + sid + 1])
                T.copy(best_idx, pv_idx[row * S + sid: row * S + sid + 1])
    return main


# =============================================== K2gB: cross-split argmax merge
@tilelang.jit(out_idx=[3], pass_configs=R4_PASS_CONFIGS)
def k2gb_argmax_merge(B: int, S: int):
    b_num = T.ceildiv(B, VEC_NUM)

    @T.prim_func
    def main(
        pv_val: T.Tensor([B * S], "float32"),  # type: ignore
        pv_idx: T.Tensor([B * S], "float32"),  # type: ignore
        dummy: T.Tensor([B], "float32"),       # type: ignore  (unused; keeps signature stable)
        tok_out: T.Tensor([B], "int32"),       # type: ignore
    ):
        with T.Kernel(b_num, is_npu=True) as (cid, vid):
            row = cid * VEC_NUM + vid
            if row < B:
                bv = T.alloc_shared([1], "float32")
                bi = T.alloc_shared([1], "float32")
                best_val = T.alloc_shared([1], "float32")
                best_idx = T.alloc_shared([1], "float32")
                tok_i32 = T.alloc_shared([1], "int32")
                T.tile.fill(best_val, -T.infinity("float32"))
                T.tile.fill(best_idx, 0.0)
                T.tile.fill(tok_i32, 0)
                for s in T.serial(S):
                    T.copy(pv_val[row * S + s: row * S + s + 1], bv)
                    T.copy(pv_idx[row * S + s: row * S + s + 1], bi)
                    with T.If(bv[0] > best_val[0]):  # noqa: SIM117
                        with T.Then():
                            best_val[0] = bv[0]
                            best_idx[0] = bi[0]
                T.tile.cast(tok_i32, best_idx, "CAST_ROUND", 1)
                T.copy(tok_i32, tok_out[row: row + 1])
    return main


def probe(label, fn, args):
    try:
        src = fn(*args).get_kernel_source()
    except Exception as e:                                        # noqa: BLE001
        print(f'{label:34s} COMPILE-FAIL  {type(e).__name__}: {str(e)[:300]}')
        return None
    tot, nb, region = ub_usage(src)
    reg = int(region) if region else UB_LIMIT
    flag = 'OK ' if tot <= reg else 'OVER'
    print(f'{label:34s} UB={tot:7d}/{reg} ({100.0*tot/reg:5.2f}%) '
          f'head={reg-tot:7d} bufs={nb:3d} {flag}')
    return tot


if __name__ == '__main__':
    CASES = [  # (case, B, V, K_MAX, BV)
        (7, 4, 152064, 256, 4096), (17, 16, 152377, 256, 4096),
        (15, 64, 152064, 256, 4096), (16, 128, 32768, 256, 4096),
        (12, 16, 32768, 256, 4096), (3, 8, 16384, 256, 4096),
        (8, 4, 16384, 128, 2048), (13, 6, 2304, 128, 2048),
        (5, 7, 4409, 256, 4096), (14, 32, 1024, 256, 4096),
        (18, 64, 65536, 256, 4096), (1, 2, 256, 256, 4096),
    ]
    print(f'UB_LIMIT={UB_LIMIT}')
    worst = {'K1A': 0, 'K1B': 0, 'K2gA': 0, 'K2gB': 0}
    for (cid, B, V, K, BV) in CASES:
        b_num = -(-B // VEC_NUM)
        g = split_geom(V, BV, b_num)
        S, NBS = g['S'], g['n_blk_split']
        print(f"\n--- case{cid} B={B} V={V} K={K} BV={BV} | b_num={b_num} "
              f"n_blk_full={g['n_blk_full']} TAIL={g['TAIL']} TAIL_PAD={g['TAIL_PAD']} "
              f"n_blk_total={g['n_blk_total']} -> S={S} NBS={NBS} grid={g['grid']} "
              f"waves={-(-g['grid']//20)}")
        r = probe(f'  case{cid} K1A(topk split)', k1a_topk_split, (B, V, K, BV, S, NBS))
        if r: worst['K1A'] = max(worst['K1A'], r)
        r = probe(f'  case{cid} K1B(softmax+mrg+epi)', k1b_softmax_merge, (B, V, K, BV, S, False))
        if r: worst['K1B'] = max(worst['K1B'], r)
        r = probe(f'  case{cid} K2gA(gumbel split)', k2ga_gumbel_split, (B, V, BV, S, NBS))
        if r: worst['K2gA'] = max(worst['K2gA'], r)
        r = probe(f'  case{cid} K2gB(argmax merge)', k2gb_argmax_merge, (B, S))
        if r: worst['K2gB'] = max(worst['K2gB'], r)
    print('\n==== WORST UB ACROSS PROBED CASES ====')
    for k, v in worst.items():
        print(f'  {k:5s} {v:7d} / {UB_LIMIT}  ({100.0*v/UB_LIMIT:5.2f}%)  head={UB_LIMIT-v}')
