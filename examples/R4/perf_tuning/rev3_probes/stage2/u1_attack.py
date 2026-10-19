"""U1 ATTACK (DESIGN.md section 14.6 U1 / section 10.2 V-S1 + V-S2): the single highest
risk of rev3.

U1 = "does tl::ascend::TopK<float> always emit EXACTLY the same descending value sequence
for the same value multiset?" rev3's merge-tree SHAPE differs from the delivered kernel
(delivered: n_blk_total x topk(2K) left-leaning chain; rev3: per-split chain + S merges in
K1B). If the hardware Sort32/MrgSort is order-sensitive on equal values or on -inf, Tier-2
bitwise is broken -- and because K2g's bitwise TRANSITIVELY depends on K1's tau (16.4),
a U1 failure takes down tau_out, valid, w_out AND tok_out at once.

Run order follows section 10.3's debug ladder so a D1 failure is already triaged:
  E1 (S=1 forced, white-box)  -> isolates "is the DECOMPOSITION correct" (K1A/K1B split of
                                 duties + pv round trip) from "is S>1 merging correct"
  A3 (NBS=1, S=4)             -> smallest non-trivial cross-split merge
  A2/D1 (S=10, grid=20)       -> the real U1 test: largest merge-tree shape difference

Every comparison is torch.equal on the raw BIT PATTERN (fp32 viewed as int32), so +-0, -inf
and NaN patterns all have to match -- not a tolerance check.
"""
import sys
import time

sys.path.insert(0, "/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

from r4_fused_sampling import (  # noqa: E402
    VEC_NUM,
    _r4_split_plan,
    r4_fused_sampling,
    r4_gumbel_sample,
    r4_gumbel_sample_splitA,
    r4_gumbel_sample_splitB,
    r4_mask_sample,
    r4_topk_stats,
    r4_topk_stats_splitA,
    r4_topk_stats_splitB,
)
from r4_ref import golden_gumbel_sample, golden_r4  # noqa: E402

LOGIT_SCALE = 8.0  # MANDATORY recipe (section 9.2.1)
FAILS = []
N_BITWISE = 0
N_BITWISE_OK = 0
N_TIER1 = 0
N_TIER1_OK = 0


def bits(t):
    """fp32 -> raw bit pattern as int32 (so +-0 / -inf / NaN patterns compare exactly)."""
    t = t.detach().cpu()
    return t.view(torch.int32) if t.dtype == torch.float32 else t


def bitwise(tag, name, actual, baseline):
    global N_BITWISE, N_BITWISE_OK
    N_BITWISE += 1
    ok = (actual.shape == baseline.shape and actual.dtype == baseline.dtype
          and bool(torch.equal(bits(actual), bits(baseline))))
    N_BITWISE_OK += ok
    extra = ""
    if not ok:
        a, b = bits(actual), bits(baseline)
        if a.shape == b.shape:
            ne = (a != b).sum().item()
            extra = f" mismatched_elems={ne}/{a.numel()}"
            idx = (a != b).nonzero()
            if idx.numel():
                i = tuple(int(x) for x in idx[0])
                extra += f" first@{i}: actual={float(actual[i]) if actual.dtype==torch.float32 else int(actual[i]):.9g} base={float(baseline[i]) if baseline.dtype==torch.float32 else int(baseline[i]):.9g}"
        else:
            extra = f" shape {tuple(actual.shape)} vs {tuple(baseline.shape)}"
        FAILS.append(f"[BITWISE_FAIL] {tag} {name}{extra}")
    print(f"[BITWISE_{'PASS' if ok else 'FAIL'}] {tag} {name} shape={tuple(actual.shape)} "
          f"dtype={str(actual.dtype).replace('torch.','')} numel={actual.numel()}{extra}")
    return ok


def tier1(tag, name, actual, golden, dtype):
    """Tier-1 mixed-tolerance double gate (section 9.3)."""
    global N_TIER1, N_TIER1_OK
    N_TIER1 += 1
    a, g = actual.detach().cpu().float(), golden.detach().cpu().float()
    if dtype == "int32":
        mism = int((actual.detach().cpu() != golden.detach().cpu()).sum().item())
        ok, ratio, mx = mism == 0, 1.0 - mism / max(a.numel(), 1), (0.0 if mism == 0 else float("inf"))
    else:
        atol, rtol, lim, req = 2**-16, 2**-10, 1e-2, 0.99
        nan_bad = int((torch.isnan(a) != torch.isnan(g)).sum().item())
        inf_bad = int((torch.isposinf(a) != torch.isposinf(g)).sum().item()) + \
            int((torch.isneginf(a) != torch.isneginf(g)).sum().item())
        m = torch.isfinite(a) & torch.isfinite(g)
        if int(m.sum().item()) == 0:
            ok, ratio, mx = (nan_bad + inf_bad) == 0, 1.0, 0.0
        else:
            err = (a[m] - g[m]).abs()
            ratio = float((err <= (atol + rtol * g[m].abs())).sum().item()) / max(a.numel(), 1)
            mx = float(err.max().item())
            ok = ratio >= req and mx <= lim and (nan_bad + inf_bad) == 0
    N_TIER1_OK += ok
    if not ok:
        FAILS.append(f"[PRECISION_FAIL] {tag} {name} ratio={ratio:.6f} max={mx:.4e}")
    print(f"[PRECISION_{'PASS' if ok else 'FAIL'}] {tag} {name} tier1 ratio={ratio:.6f} max_abs={mx:.4e}")
    return ok


def make_inputs(B, V, k, p, mu):
    torch.manual_seed(0)
    logits = torch.randn(B, V, dtype=torch.float32) * LOGIT_SCALE
    top_ks = torch.full((B,), k, dtype=torch.int32)
    top_ps = torch.full((B,), p, dtype=torch.float32)
    min_ps = torch.full((B,), mu, dtype=torch.float32)
    rand = torch.arange(B, dtype=torch.int32) * 7919 + 13
    return logits, top_ks, top_ps, min_ps, rand


# ------------------------------------------------------------------ baseline (frozen kernels)
def run_current(B, V, K_MAX, BV, nmp, logits, top_ks, top_ps, min_ps, rand, fuse):
    """Delivered path: K1 -> (K2m | K2g). The 3 frozen kernels, called directly."""
    m, s, tau, valid = r4_topk_stats(B, V, K_MAX, BV, nmp)(
        logits.npu(), top_ks.npu(), top_ps.npu(), min_ps.npu())
    torch.npu.synchronize()
    if not fuse:
        w = r4_mask_sample(B, V, BV)(logits.npu(), m, s, tau)
        torch.npu.synchronize()
        return [x.cpu() for x in (m, s, tau, valid, w)]
    tok = r4_gumbel_sample(B, V, BV)(logits.npu(), tau, rand.npu())
    torch.npu.synchronize()
    return [x.cpu() for x in (m, s, tau, valid, tok)]


# ------------------------------------------------------------------ white-box split path
def run_split_whitebox(B, V, K_MAX, BV, nmp, logits, top_ks, top_ps, min_ps, rand, fuse,
                       S, NBS, S_TAIL):
    """Split path with an EXPLICIT (S, NBS, S_TAIL) -- used by E1 to force the S=1 degenerate."""
    lg = logits.npu()
    pv = torch.empty((B * S, K_MAX), dtype=torch.float32, device=lg.device)
    r4_topk_stats_splitA(B, V, K_MAX, BV, S, NBS, S_TAIL)(lg, pv)
    m, s, tau, valid = r4_topk_stats_splitB(B, V, K_MAX, BV, S, nmp)(
        lg, top_ks.npu(), top_ps.npu(), min_ps.npu(), pv)
    torch.npu.synchronize()
    if not fuse:
        w = r4_mask_sample(B, V, BV)(lg, m, s, tau)
        torch.npu.synchronize()
        return [x.cpu() for x in (m, s, tau, valid, w)], pv.cpu()
    pv_v = torch.empty((B * S,), dtype=torch.float32, device=lg.device)
    pv_i = torch.empty((B * S,), dtype=torch.float32, device=lg.device)
    r4_gumbel_sample_splitA(B, V, BV, S, NBS)(lg, tau, rand.npu(), pv_v, pv_i)
    tok = r4_gumbel_sample_splitB(B, S)(pv_v, pv_i)
    torch.npu.synchronize()
    return [x.cpu() for x in (m, s, tau, valid, tok)], pv.cpu()


def one(tag, B, V, K_MAX, BV, k, p, mu, nmp, force_S=None):
    """Full dual-path comparison for one shape x both tasks."""
    logits, top_ks, top_ps, min_ps, rand = make_inputs(B, V, k, p, mu)
    use_split, S, NBS, S_TAIL = _r4_split_plan(B, V, K_MAX, BV)
    if force_S is not None:
        S = force_S
        nbf = V // BV
        TAIL = V - nbf * BV
        nbt = nbf + (1 if TAIL > 0 else 0)
        NBS = -(-nbt // S)
        S_TAIL = nbf // NBS if TAIL > 0 else -1
    bn = -(-B // VEC_NUM)
    print("\n" + "=" * 96)
    print(f"[{tag}] B={B} V={V} K_MAX={K_MAX} BV={BV} k={k} p={p} mu={mu} NEED_MIN_P={nmp}")
    print(f"       host plan: use_split={use_split} S={S} NBS={NBS} S_TAIL={S_TAIL} "
          f"grid_A={bn*S} grid_B={bn}"
          + ("  [WHITE-BOX FORCED S]" if force_S is not None else ""))
    print("=" * 96)
    for fuse in (False, True):
        task = "fused(K2g)" if fuse else "mask(K2m)"
        t0 = time.time()
        base = run_current(B, V, K_MAX, BV, nmp, logits, top_ks, top_ps, min_ps, rand, fuse)
        t_base = time.time() - t0
        t0 = time.time()
        got, pv = run_split_whitebox(B, V, K_MAX, BV, nmp, logits, top_ks, top_ps, min_ps,
                                     rand, fuse, S, NBS, S_TAIL)
        t_split = time.time() - t0
        names = ("m_out", "s_out", "tau_out", "valid", "tok_out" if fuse else "w_out")
        print(f"  -- task={task}: current={t_base:.1f}s split={t_split:.1f}s (incl. JIT) "
              f"pv_shape={tuple(pv.shape)}")
        # Tier-2 bitwise vs the FROZEN kernels -- this is the U1 verdict
        for nm, a, b in zip(names, got, base):
            bitwise(tag, f"{task}/{nm}", a, b)
        # Tier-1 vs golden
        if fuse:
            m_g, s_g, tau_g, valid_g, w_g = golden_r4(logits, top_ks, top_ps, min_ps, nmp, K_MAX)
            g = [m_g, s_g, tau_g, valid_g, golden_gumbel_sample(logits, tau_g, rand)]
        else:
            g = list(golden_r4(logits, top_ks, top_ps, min_ps, nmp, K_MAX))
        dt = ("float32", "float32", "float32", "int32", "int32" if fuse else "float32")
        for nm, a, gg, d in zip(names, got, g, dt):
            tier1(tag, f"{task}/{nm}", a, gg, d)
        vg = g[3]
        all1 = bool((vg == 1).all().item())
        print(f"  -- valid_ref all-1 = {all1} (section 9.2.1 recipe check)"
              + ("" if all1 else "  *** GUARD FIRED: L0 would false-fail, check LOGIT_SCALE ***"))
        if not all1:
            FAILS.append(f"[RECIPE] {tag} {task}: valid_ref not all 1")


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(f"U1 / V-S1 / V-S2 ATTACK  (torch={torch.__version__}, npu={torch.npu.is_available()})")
    # ---- E1: S=1 degenerate, white-box. Isolates decomposition from cross-split merging.
    if only in ("all", "e1"):
        one("L0-E1_S1_degenerate", 2, 16384, 256, 4096, 40, 0.95, 0.0, False, force_S=1)
    # ---- A3: NBS=1 (each split owns exactly 1 block), S=4. Smallest real cross-split merge.
    if only in ("all", "a3"):
        one("L0-A3_split_s4_oneblk", 8, 16384, 256, 4096, 40, 0.95, 0.05, True)
    # ---- A2 == D1 shape: S=10 (max), grid=20 (exactly one full wave), n_blk_full=37 >> 2,
    #      uneven load [4x9, 2], aligned tail 512. THE U1 VERDICT.
    if only in ("all", "d1"):
        one("L0-D1_dualpath_bitwise", 4, 152064, 256, 4096, 40, 0.95, 0.05, True)

    print("\n" + "=" * 96)
    print(f"U1 SUMMARY: Tier-2 bitwise {N_BITWISE_OK}/{N_BITWISE} PASS   "
          f"Tier-1 golden {N_TIER1_OK}/{N_TIER1} PASS   failures={len(FAILS)}")
    for f in FAILS:
        print("   ", f)
    u1 = N_BITWISE_OK == N_BITWISE
    verdict = "CONFIRMED SAFE" if u1 else \
        "*** BROKEN *** -> apply the section 14.6 fallback (single topk(K_MAX, S*K_MAX) in K1B)"
    print(f"\n>>> U1 (topk hardware determinism across merge-tree shapes): {verdict}")
    print(">>> V-S1 (dual-path bitwise):", "PASS" if u1 else "FAIL")
    print("U1 VERDICT:", "[BITWISE_PASS]" if u1 else "[BITWISE_FAIL]")
    sys.exit(0 if u1 else 1)
