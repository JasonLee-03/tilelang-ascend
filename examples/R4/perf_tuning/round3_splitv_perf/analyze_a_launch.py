"""Round-3 item (2): close U2 / V-S3 by MEASURING a_launch, then analyse the guard.

`a_launch` (the fixed cost of one extra kernel launch) is the ONLY parameter of the DESIGN.md
section 12.4 cost model that was never measured; the delivered guard therefore uses the
conservative default `_SPLIT_A_LAUNCH_US = 8.0`, which Stage 2 did not relax.

Why this matters and what口径 is used
------------------------------------
cann-bench's `elapsed_us` is the SUM of the per-step `main_kernel` `Duration(us)` values from
msprof kernel_details.csv (round 2 verified K1_us + K2_us == elapsed_us exactly; ksplit_fine.py
re-verifies it for all 18 cases x 2 tasks in this round). Inter-kernel gaps are NOT part of that
sum. So the `a_launch` that the guard must model is the *device-side* duration floor of a launch,
NOT the host launch latency. Both are reported below, but only the former feeds the guard.

Three independent estimators
----------------------------
E1 (model-free floor, PRIMARY): rev3's `r4_gumbel_sample_splitB(B, S)` (K2gB) has UB=132 B and its
    entire useful work is folding S <= 10 partial (m, s, idx) triples per row. Its measured
    Duration is therefore launch + a tiny amount of real work => an UPPER BOUND on a_launch.
E2 (least-squares decomposition of K2gB): fit Duration = a0 + a1*b_num + a2*S over the 6 split
    fused cases; the intercept a0 separates the launch floor from the two work terms.
E3 (section 12.4 model residual): solve the piecewise model for a_launch on K1A / K1B / K2gA using
    the section 12.2 calibrated constants. Reported for completeness AND to document where the
    model does not fit -- it is not used as the primary answer.

Then: the guard-relaxation sensitivity required by the task -- which cases would NEWLY take the
split path if `_SPLIT_A_LAUNCH_US` were lowered to the measured value. Analysis only; the guard is
NOT modified here (that needs Orchestrator approval).

Usage: analyze_a_launch.py [out.json]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R4 = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (R4, os.path.join(R4, "bench", "hap_thw_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r4_fused_sampling as M  # noqa: E402
from r4_thw_common import load_cases  # noqa: E402

out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "a_launch_analysis.json")

CASES = {c["case_id"]: c for c in load_cases("r4_fused_sampling", verbose=False)}
CUBE = M.detect_cube_core_num()
VEC_NUM = M.VEC_NUM
KS = {k: json.load(open(os.path.join(HERE, f"ksplit_cand_{t}.json")))["per_case"]
      for k, t in (("fused", "fused"), ("mask", "mask"))}

print("=" * 100)
print("E1/E2  a_launch from K2gB -- the smallest complete kernel in the rev3 chain (UB=132 B)")
print("=" * 100)
print(f"{'case':>4} {'B':>4} {'b_num':>5} {'S':>3} {'K2gB blk':>8} {'K2gB_us':>8} "
      f"{'K1A_us':>8} {'K1B_us':>8} {'K2gA_us':>8} {'total_us':>9}")
pts = []
for ci in sorted(int(k) for k in KS["fused"]):
    d = KS["fused"][str(ci)]
    if d.get("path") != "SPLIT":
        continue
    pk = d["per_kernel_us"]
    print(f"{ci:>4} {d['B']:>4} {d['b_num']:>5} {d['S']:>3} {d['blocks'][3]:>8} "
          f"{pk['K2gB']:>8.3f} {pk['K1A']:>8.3f} {pk['K1B']:>8.3f} {pk['K2gA']:>8.3f} "
          f"{d['total_us']:>9.3f}")
    pts.append((d["b_num"], d["S"], pk["K2gB"]))

k2gb = [p[2] for p in pts]
e1_min, e1_med, e1_max = min(k2gb), sorted(k2gb)[len(k2gb) // 2], max(k2gb)
print(f"\n  E1 K2gB Duration over the 6 split fused cases: min={e1_min:.3f} us  "
      f"median={e1_med:.3f} us  max={e1_max:.3f} us")
print(f"  => a_launch < {e1_min:.3f} us  (K2gB does nonzero fold work, so the floor is BELOW its min)")
print(f"  => a conservative planning value is a_launch <= {e1_max:.3f} us "
      f"= {100 * e1_max / M._SPLIT_A_LAUNCH_US:.1f}% of the guard's default "
      f"{M._SPLIT_A_LAUNCH_US} us")

# ---- E2: least-squares fit  K2gB = a0 + a1*b_num + a2*S  (normal equations, no numpy needed) ----
A = [[1.0, float(b), float(s)] for b, s, _ in pts]
y = [float(d) for _, _, d in pts]
n, p = len(A), 3
AtA = [[sum(A[k][i] * A[k][j] for k in range(n)) for j in range(p)] for i in range(p)]
Aty = [sum(A[k][i] * y[k] for k in range(n)) for i in range(p)]
# gaussian elimination
Mx = [row[:] + [Aty[i]] for i, row in enumerate(AtA)]
for c in range(p):
    piv = max(range(c, p), key=lambda r: abs(Mx[r][c]))
    Mx[c], Mx[piv] = Mx[piv], Mx[c]
    for r in range(p):
        if r != c and Mx[c][c] != 0:
            f = Mx[r][c] / Mx[c][c]
            for cc in range(c, p + 1):
                Mx[r][cc] -= f * Mx[c][cc]
coef = [Mx[i][p] / Mx[i][i] if Mx[i][i] else float("nan") for i in range(p)]
resid = [y[k] - sum(coef[i] * A[k][i] for i in range(p)) for k in range(n)]
print(f"\n  E2 least-squares fit  K2gB_us = a0 + a1*b_num + a2*S   (n={n} points, 3 params)")
print(f"     a0 (launch floor intercept) = {coef[0]:+.3f} us")
print(f"     a1 (per b_num)              = {coef[1]:+.3f} us")
print(f"     a2 (per split S)            = {coef[2]:+.3f} us")
print(f"     max |residual| = {max(abs(r) for r in resid):.3f} us  "
      f"rms = {(sum(r * r for r in resid) / n) ** 0.5:.3f} us")
print(f"     NOTE: 6 points / 3 params is a thin fit; a0 is an EXTRAPOLATION to b_num=0,S=0 and")
print(f"           is reported as an order-of-magnitude cross-check on E1, not as the answer.")

# ---- E3: section 12.4 model residual for a_launch, per split kernel ----
print("\n" + "=" * 100)
print("E3  section 12.4 piecewise-model residual for a_launch (documents where the model mis-fits)")
print("=" * 100)
T_SORT = M._SPLIT_T_SORT_US
T_MRG = M._SPLIT_T_MERGE_US
F256, P4096 = 21.63, 1.802          # section 12.2 calibrated constants
print(f"{'task':>5} {'case':>4} {'kern':>5} {'meas_us':>8} {'model_nonlaunch_us':>19} "
      f"{'implied a_launch':>17}")
e3 = []
for task in ("fused", "mask"):
    for ci in sorted(int(k) for k in KS[task]):
        d = KS[task][str(ci)]
        if d.get("path") != "SPLIT":
            continue
        B, V, K, BV, S, NBS = d["B"], d["V"], d["K_MAX"], d["BV"], d["S"], d["NBS"]
        bn, nbf = d["b_num"], V // BV
        TAIL = V - nbf * BV
        maxblk = min(nbf, NBS)
        t = T_SORT.get((BV, K), M._SPLIT_T_SORT_DEFAULT)
        tm = T_MRG.get(K, M._SPLIT_T_MERGE_DEFAULT)
        wA = -(-bn * S // CUBE)
        wB = -(-bn // CUBE)
        pk = d["per_kernel_us"]
        # K1A: waves(b_num*S) * [a + maxblk*t + (tail sort cost if S_TAIL owns the tail)]
        nl_A = wA * maxblk * t
        # K1B: waves(b_num) * [a + F + n_blk_full*p + S*t_mrg]   (tail softmax cost not modelled)
        nl_B = wB * (F256 + nbf * P4096 + S * tm)
        for nm, meas, nl, w in (("K1A", pk["K1A"], nl_A, wA), ("K1B", pk["K1B"], nl_B, wB)):
            a = (meas - nl) / w if w else float("nan")
            e3.append((task, ci, nm, meas, nl, a))
            print(f"{task:>5} {ci:>4} {nm:>5} {meas:>8.3f} {nl:>19.3f} {a:>+17.3f}")
neg = [x for x in e3 if x[5] < 0]
pos = [x for x in e3 if x[5] >= 0]
print(f"\n  E3 spread: {len(e3)} residuals, {len(neg)} NEGATIVE (model over-predicts the non-launch "
      f"part), {len(pos)} non-negative")
if pos:
    print(f"           non-negative range [{min(x[5] for x in pos):+.3f}, "
          f"{max(x[5] for x in pos):+.3f}] us")
print("  => E3 is INCONSISTENT: section 12.2's F(256)=21.63 us/row was calibrated on the CURRENT")
print("     single-kernel K1 and does not transfer to K1B (which no longer runs the per-block Sort).")
print("     E3 is therefore reported only as evidence that the model's F/p split is not separable")
print("     at this granularity. The DEFENSIBLE answer is E1: a_launch < "
      f"{e1_min:.3f} us, plan with <= {e1_max:.3f} us.")

# ---- guard-relaxation sensitivity ----
print("\n" + "=" * 100)
print("GUARD-RELAXATION SENSITIVITY (analysis only -- the guard is NOT modified here)")
print("=" * 100)


def plan_with(a_launch, B, V, K, BV, cube=CUBE):
    """Byte-faithful re-implementation of _r4_split_plan with a_launch as a free parameter."""
    if B <= 0 or V <= 0:
        return False, 1, 1, -1, None
    b_num = -(-B // VEC_NUM)
    n_blk_full = V // BV
    TAIL = V - n_blk_full * BV
    n_blk_total = n_blk_full + (1 if TAIL > 0 else 0)
    if n_blk_total <= 0:
        return False, 1, 1, -1, None
    S_cap = max(0, cube // b_num)
    S = max(1, min(S_cap, n_blk_total))
    NBS = -(-n_blk_total // S)
    S = -(-n_blk_total // NBS)
    NBS = -(-n_blk_total // S)
    S_TAIL = n_blk_full // NBS if TAIL > 0 else -1
    use_split, dbg = False, None
    if S >= 2 and n_blk_full >= 2:
        maxblk_full = max(min(n_blk_full, max(0, (s + 1) * NBS - s * NBS)) for s in range(S))
        t_sort = T_SORT.get((BV, K), M._SPLIT_T_SORT_DEFAULT)
        t_mrg = T_MRG.get(K, M._SPLIT_T_MERGE_DEFAULT)
        pred_saving = (n_blk_full - maxblk_full) * t_sort
        pred_cost = a_launch + S * t_mrg
        use_split = pred_saving > M._SPLIT_SAFETY * pred_cost
        dbg = dict(S=S, NBS=NBS, maxblk_full=maxblk_full, t_sort=t_sort, t_mrg=t_mrg,
                   pred_saving=round(pred_saving, 4), pred_cost=round(pred_cost, 4),
                   threshold=round(M._SPLIT_SAFETY * pred_cost, 4),
                   margin=round(pred_saving - M._SPLIT_SAFETY * pred_cost, 4),
                   structural_ok=True)
    else:
        # report the TRUE structural reason: which of the two terms bound S below 2
        if S < 2:
            S_cap = max(0, cube // b_num)
            binding = (f"S_cap=cube//b_num={cube}//{b_num}={S_cap}"
                       if S_cap <= n_blk_total else f"n_blk_total={n_blk_total}")
            why = (f"S<2: S=min(S_cap={S_cap}, n_blk_total={n_blk_total})={S}; "
                   f"binding term = {binding}")
        else:
            why = f"n_blk_full={n_blk_full} < 2 (S={S} is >= 2, so the cost guard is not reached)"
        dbg = dict(S=S, NBS=NBS, structural_ok=False, why=why,
                   S_cap=max(0, cube // b_num), n_blk_total=n_blk_total,
                   n_blk_full=n_blk_full, TAIL=TAIL)
    return use_split, S, NBS, S_TAIL, dbg


# self-check: the re-implementation must reproduce the DELIVERED dispatch at the shipped default
mism = []
for ci in sorted(CASES):
    c = CASES[ci]
    got = plan_with(M._SPLIT_A_LAUNCH_US, c["B"], c["V"], c["K_MAX"], c["BV"])[:4]
    want = M._r4_split_plan(c["B"], c["V"], c["K_MAX"], c["BV"])
    if got != want:
        mism.append((ci, got, want))
print(f"  self-check: re-implementation == delivered _r4_split_plan at "
      f"_SPLIT_A_LAUNCH_US={M._SPLIT_A_LAUNCH_US} for "
      f"{len(CASES) - len(mism)}/{len(CASES)} cases"
      + (f"  MISMATCHES={mism}" if mism else "  (exact)"))

SCAN = [8.0, 4.0, 3.20, 2.74, 1.54, 1.0, 0.5, 0.0]
base_set = {ci for ci in sorted(CASES)
            if M._r4_split_plan(CASES[ci]["B"], CASES[ci]["V"], CASES[ci]["K_MAX"],
                                CASES[ci]["BV"])[0]}
print(f"\n  delivered split set at a_launch={M._SPLIT_A_LAUNCH_US}: {sorted(base_set)}")
print(f"\n{'a_launch':>9} | {'split set':<28} {'n':>2} | newly added")
flip = {}
for a in SCAN:
    s = {ci for ci in sorted(CASES)
         if plan_with(a, CASES[ci]["B"], CASES[ci]["V"], CASES[ci]["K_MAX"], CASES[ci]["BV"])[0]}
    added = sorted(s - base_set)
    flip[a] = added
    tag = "  <-- delivered default" if a == M._SPLIT_A_LAUNCH_US else ""
    print(f"{a:>9.2f} | {str(sorted(s)):<28} {len(s):>2} | {added if added else '-'}{tag}")

# why each non-split case cannot flip, and the margin for those that can
print(f"\n  per-case guard detail at the two candidate relaxations (structural blockers first):")
print(f"{'case':>4} {'B':>4} {'V':>7} {'b_num':>5} {'nbf':>4} {'S':>3} {'struct':>7} | "
      f"{'saving':>8} {'thr@8.0':>8} {'thr@1.54':>9} {'thr@3.20':>9} | verdict")
detail = {}
for ci in sorted(CASES):
    c = CASES[ci]
    B, V, K, BV = c["B"], c["V"], c["K_MAX"], c["BV"]
    u8, S8, N8, T8, d8 = plan_with(8.0, B, V, K, BV)
    u154, _, _, _, d154 = plan_with(1.54, B, V, K, BV)
    u320, _, _, _, d320 = plan_with(3.20, B, V, K, BV)
    bn = -(-B // VEC_NUM)
    nbf = V // BV
    if not d8["structural_ok"]:
        verdict = f"BLOCKED structurally ({d8['why']}) -- cannot flip at any a_launch"
        sav = thr8 = thr154 = thr320 = float("nan")
    else:
        sav = d8["pred_saving"]
        thr8, thr154, thr320 = d8["threshold"], d154["threshold"], d320["threshold"]
        if u8:
            verdict = "already split"
        elif u154 and u320:
            verdict = f"FLIPS at both 1.54 and 3.20 (margin@1.54={d154['margin']:+.3f} us)"
        elif u154:
            verdict = f"flips only at 1.54 (margin@3.20={d320['margin']:+.3f} us)"
        else:
            verdict = f"stays current (margin@1.54={d154['margin']:+.3f} us)"
    detail[ci] = dict(B=B, V=V, b_num=bn, n_blk_full=nbf, S=S8, structural_ok=d8["structural_ok"],
                      blocker=None if d8["structural_ok"] else d8.get("why"),
                      pred_saving=sav, threshold_a8=thr8, threshold_a320=thr320,
                      threshold_a154=thr154, split_at_8=bool(u8), split_at_320=bool(u320),
                      split_at_154=bool(u154), verdict=verdict)
    print(f"{ci:>4} {B:>4} {V:>7} {bn:>5} {nbf:>4} {S8:>3} {str(d8['structural_ok']):>7} | "
          f"{sav:>8.3f} {thr8:>8.3f} {thr154:>9.3f} {thr320:>9.3f} | {verdict}")

json.dump(dict(
    _meta=dict(description="a_launch measured from msprof (round-3 cand arm) + guard-relaxation "
                           "sensitivity. No Python/torch timing.",
               kernel_sha256=M.__file__, cube_core_num=CUBE, vec_num=VEC_NUM,
               guard_default_a_launch_us=M._SPLIT_A_LAUNCH_US,
               guard_safety=M._SPLIT_SAFETY),
    E1_k2gb_floor_us=dict(min=e1_min, median=e1_med, max=e1_max,
                          conclusion=f"a_launch < {e1_min:.3f} us; plan with <= {e1_max:.3f} us"),
    E1_points=[dict(b_num=b, S=s, k2gb_us=d) for b, s, d in pts],
    E2_fit=dict(a0_intercept_us=coef[0], a1_per_b_num=coef[1], a2_per_S=coef[2],
                max_abs_resid_us=max(abs(r) for r in resid)),
    E3_model_residuals=[dict(task=t, case=c, kernel=k, meas_us=m, model_nonlaunch_us=nl,
                             implied_a_launch_us=a) for t, c, k, m, nl, a in e3],
    guard_selfcheck_mismatches=mism,
    delivered_split_set=sorted(base_set),
    flip_scan={str(a): flip[a] for a in SCAN},
    per_case_guard_detail={str(k): v for k, v in detail.items()},
), open(out_path, "w"), indent=1)
print(f"\nwrote {out_path}")
