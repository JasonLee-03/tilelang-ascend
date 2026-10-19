"""Root-cause the 6 new BOUNDARY_WARN cases (V-S4 whole-split -inf recipes).

Every one of them reported tier2_bitwise=5/5, i.e. the split path is BIT-IDENTICAL to the
three frozen kernels -- so whatever is happening is NOT a rev3 regression. This probe pins
down (a) what the values actually are, (b) whether it is NaN, (c) whether it reproduces on
the CURRENT path with a shape that never splits (proving path-independence), and (d) whether
the D-GUARD degrades safely (valid==0 -> host falls back) instead of returning wrong tokens.
"""
import sys

sys.path.insert(0, "/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4")
import torch
import torch_npu  # noqa: F401

from r4_fused_sampling import _r4_split_plan, r4_fused_sampling, r4_topk_stats
from r4_ref import golden_r4

LOGIT_SCALE = 8.0


def probe(tag, B, V, kmax, bv, n_ninf_blocks, k=40, p=0.95, mu=0.05, nmp=True):
    torch.manual_seed(0)
    logits = torch.randn(B, V, dtype=torch.float32) * LOGIT_SCALE
    if n_ninf_blocks > 0:
        logits[:, : n_ninf_blocks * bv] = float("-inf")
    use_split, S, NBS, ST = _r4_split_plan(B, V, kmax, bv)
    top_ks = torch.full((B,), k, dtype=torch.int32)
    top_ps = torch.full((B,), p, dtype=torch.float32)
    min_ps = torch.full((B,), mu, dtype=torch.float32)

    print(f"\n{'='*100}")
    print(f"[{tag}] B={B} V={V} K_MAX={kmax} BV={bv} n_-inf_leading_blocks={n_ninf_blocks} "
          f"(-inf frac={float(torch.isneginf(logits).float().mean()):.3f})")
    print(f"        host plan: use_split={use_split} S={S} NBS={NBS} S_TAIL={ST} "
          f"-> path={'SPLIT' if use_split else 'CURRENT'}")
    print(f"        row_all_-inf={bool(torch.isneginf(logits).all(dim=-1).any())}")

    # golden (CPU float64)
    gm, gs, gtau, gvalid, gw = golden_r4(logits, top_ks, top_ps, min_ps, nmp, kmax)
    # kernel via the host wrapper (routes by plan)
    m, s, tau, valid, w = r4_fused_sampling(logits.npu(), top_ks.npu(), top_ps.npu(),
                                            min_ps.npu(), nmp, kmax, bv)
    torch.npu.synchronize()
    m, s, tau, valid, w = m.cpu(), s.cpu(), tau.cpu(), valid.cpu(), w.cpu()
    # kernel via the FROZEN K1 directly, bypassing the dispatch entirely -> path-independent proof
    fm, fs, ftau, fvalid = r4_topk_stats(B, V, kmax, bv, nmp)(
        logits.npu(), top_ks.npu(), top_ps.npu(), min_ps.npu())
    torch.npu.synchronize()
    fm, fs, ftau, fvalid = fm.cpu(), fs.cpu(), ftau.cpu(), fvalid.cpu()

    def show(nm, t):
        t = t.detach().cpu()
        fin = torch.isfinite(t.float()) if t.dtype == torch.float32 else torch.ones_like(t, dtype=torch.bool)
        print(f"        {nm:12s} nan={int(torch.isnan(t.float()).sum()) if t.dtype==torch.float32 else 0} "
              f"inf={int(torch.isinf(t.float()).sum()) if t.dtype==torch.float32 else 0} "
              f"finite={int(fin.sum())}/{t.numel()} first4={[round(float(x),5) if torch.isfinite(torch.tensor(float(x))) else str(float(x)) for x in t.flatten()[:4]]}")

    print("     GOLDEN (CPU float64):")
    for nm, t in (("m", gm), ("s", gs), ("tau", gtau), ("valid", gvalid)):
        show(nm, t)
    print("     KERNEL via host wrapper (path=%s):" % ("SPLIT" if use_split else "CURRENT"))
    for nm, t in (("m", m), ("s", s), ("tau", tau), ("valid", valid)):
        show(nm, t)
    print("     KERNEL via FROZEN r4_topk_stats called DIRECTLY (dispatch bypassed):")
    for nm, t in (("m", fm), ("s", fs), ("tau", ftau), ("valid", fvalid)):
        show(nm, t)

    same = all(bool(torch.equal(a.view(torch.int32) if a.dtype == torch.float32 else a,
                                b.view(torch.int32) if b.dtype == torch.float32 else b))
               for a, b in ((m, fm), (s, fs), (tau, ftau), (valid, fvalid)))
    print(f"     >>> wrapper output == frozen-K1-direct output (bitwise): {same}")
    print(f"     >>> golden s finite: {bool(torch.isfinite(gs).all())}   "
          f"kernel s NaN: {bool(torch.isnan(s).any())}")
    print(f"     >>> golden valid: {sorted(set(gvalid.tolist()))}   kernel valid: {sorted(set(valid.tolist()))}")
    guard_fired = bool((valid == 0).all().item())
    print(f"     >>> D-GUARD fired on every row (valid all-0 -> host would fall back): {guard_fired}")
    nan_s = bool(torch.isnan(s).any())
    print(f"     VERDICT: {'s_run went NaN via -inf - (-inf)' if nan_s else 'no NaN'}; "
          f"{'SAFE degrade (valid=0)' if guard_fired else '*** UNSILENT WRONG VALUE ***'}; "
          f"{'path-independent (frozen kernel identical)' if same else '*** PATH-DEPENDENT ***'}")
    return dict(nan_s=nan_s, guard_fired=guard_fired, path_independent=same)


if __name__ == "__main__":
    res = {}
    # The 3 recipes that WARNed (whole leading split(s) are -inf), both on a SPLIT shape...
    res["split0_neginf_SPLITPATH"] = probe("V-S4 split0", 4, 20793, 256, 4096, 1)
    res["split0to4_neginf_SPLITPATH"] = probe("V-S4 split0..4", 4, 20793, 256, 4096, 5)
    res["splitmid_neginf_SPLITPATH"] = probe("V-S4 split0..1", 8, 16384, 256, 4096, 2)
    # ...and the SAME hazard on a shape that NEVER splits (b_num>=20 -> S=1, current path),
    # which proves it is a property of the delivered online-softmax scan, not of rev3.
    res["leading_block_neginf_CURRENTPATH"] = probe("control CURRENT path", 64, 16384, 256, 4096, 1)
    # control: sparse -inf (no whole block) must stay clean
    print(f"\n{'='*100}")
    torch.manual_seed(0)
    lg = torch.randn(4, 20793, dtype=torch.float32) * LOGIT_SCALE
    lg[0, :50] = float("-inf")
    lg[2, 100] = float("-inf")
    ks = torch.full((4,), 40, dtype=torch.int32); ps = torch.full((4,), 0.95, dtype=torch.float32)
    mus = torch.full((4,), 0.05, dtype=torch.float32)
    m, s, tau, valid, w = r4_fused_sampling(lg.npu(), ks.npu(), ps.npu(), mus.npu(), True, 256, 4096)
    torch.npu.synchronize()
    gm, gs, gtau, gvalid, gw = golden_r4(lg, ks, ps, mus, True, 256)
    print(f"[control sparse -inf] kernel s NaN={bool(torch.isnan(s.cpu()).any())} "
          f"valid={sorted(set(valid.cpu().tolist()))} golden valid={sorted(set(gvalid.tolist()))} "
          f"tau_max_err={float((tau.cpu()-gtau).abs().max()):.3e} "
          f"valid_match={bool(torch.equal(valid.cpu(), gvalid))}")
    print("\n" + "=" * 100)
    print("SUMMARY")
    for k, v in res.items():
        print(f"  {k:38s} {v}")
    allpi = all(v["path_independent"] for v in res.values())
    allguard = all(v["guard_fired"] for v in res.values())
    print(f"\n  all path-independent (wrapper == frozen-K1-direct, bitwise): {allpi}")
    print(f"  all degrade safely via D-GUARD (valid all-0):               {allguard}")
