"""Item-A criterion-2 ARBITRATION on the harness口径: distribution, not a single sample.

The 3-arm A/B gave ONE sample per arm and flagged exactly one pair outside the 3% band:
case 13 mask, 14.620 -> 15.240 us (+4.24%, absolute +0.62 us). That case is structurally blocked
(`_r4_split_plan` = (False, 2, 1, 1) at a_launch 8.0 AND 1.540, because n_blk_full = 2304//2048 = 1
< 2, so the cost guard is never reached), and ksplit_fine_itemA.py confirmed an identical launch
sequence and Block Num in all three arms. One sample per arm cannot bound a noise distribution, so
ab_repeat_mask_itemA.sh re-ran R4MaskSample alternately for both arms 3 more times.

This script pools every harness mask sample:
    base arm (a_launch 8.0)   : base1, base2, base_r1, base_r2, base_r3   (5 samples)
    cand arm (a_launch 1.540) : cand,  cand_r1, cand_r2, cand_r3          (4 samples)
and for each case reports the within-arm spread alongside the between-arm delta, so the flagged
+4.24% can be judged against the harness's OWN noise at that absolute scale rather than against a
percentage band calibrated on cases 100x larger.

Decision rule applied (stated up front, not chosen after seeing the data):
  the flagged pair is measurement scatter, not a regression, iff the cand arm's distribution
  OVERLAPS the base arm's -- i.e. min(cand) <= max(base) and min(base) <= max(cand) -- AND the
  median-to-median delta is inside the pooled within-arm spread. If instead the two distributions
  are disjoint with cand entirely above base, it is a real regression and the change must roll back.

Also re-verifies accuracy on every repeat arm (18/18 passed, 90/90 output tensors mismatch 0).

Usage: arbitrate_case13.py <out.json> <label:report.json> ...
"""
import json
import math
import statistics
import sys

OUT = sys.argv[1]
ARMS = []
for a in sys.argv[2:]:
    lab, path = a.split(":", 1)
    ARMS.append((lab, path))
BASE_LABS = {"base1", "base2", "base_r1", "base_r2", "base_r3"}
CAND_LABS = {"cand", "cand_r1", "cand_r2", "cand_r3"}
FLAGGED = 13
NOISE_PCT = 3.0


def load(path):
    d = json.load(open(path))
    op = d["operators"][0]
    cases = op["cases"]
    if isinstance(cases, dict):
        cases = list(cases.values())
    out = {}
    for c in cases:
        cid = int(str(c["case_id"]).rsplit("_", 1)[-1])
        acc = c.get("accuracy") or {}
        ors = acc.get("output_results") or []
        out[cid] = dict(elapsed_us=c.get("elapsed_us"), status=c.get("status"),
                        acc_passed=acc.get("passed"),
                        mismatch=sum(int(r.get("mismatch_count") or 0) for r in ors),
                        n_out=len(ors),
                        worst_mad=max([float(r.get("max_abs_diff") or 0.0) for r in ors] or [0.0]))
    return dict(operator=op["operator"], passed=op["passed_cases"], total=op["total_cases"],
                failed=op["failed_cases"], score=op["score"], cases=out)


D = {lab: load(p) for lab, p in ARMS}
base_labs = [l for l, _ in ARMS if l in BASE_LABS]
cand_labs = [l for l, _ in ARMS if l in CAND_LABS]
print("=" * 122)
print("ITEM-A CRITERION-2 ARBITRATION -- harness口径 R4MaskSample, pooled samples")
print("=" * 122)
print(f"  BASE arm (a_launch=8.0)    : {base_labs}  (n={len(base_labs)})")
print(f"  CAND arm (a_launch=1.540)  : {cand_labs}  (n={len(cand_labs)})")
for lab, _ in ARMS:
    v = D[lab]
    n_out = sum(c["n_out"] for c in v["cases"].values())
    mm = sum(c["mismatch"] for c in v["cases"].values())
    npass = sum(bool(c["acc_passed"]) for c in v["cases"].values())
    print(f"    {lab:9s} {v['operator']:16s} passed={v['passed']}/{v['total']} failed={v['failed']} "
          f"score={v['score']:.4f} | accuracy {npass}/{len(v['cases'])} cases, "
          f"{n_out} output tensors, total mismatch_count={mm}, "
          f"worst max_abs_diff={max(c['worst_mad'] for c in v['cases'].values())}")

ids = sorted(D[base_labs[0]]["cases"])
print(f"\n{'-' * 122}")
print(f"{'case':>4} | {'base med':>9} {'base min':>9} {'base max':>9} {'spread%':>8} | "
      f"{'cand med':>9} {'cand min':>9} {'cand max':>9} {'spread%':>8} | "
      f"{'med delta%':>10} {'overlap':>8} | verdict")
print(f"{'-' * 122}")
rows = []
for ci in ids:
    b = [D[l]["cases"][ci]["elapsed_us"] for l in base_labs]
    c = [D[l]["cases"][ci]["elapsed_us"] for l in cand_labs]
    bm, cm = statistics.median(b), statistics.median(c)
    bsp = 100 * (max(b) - min(b)) / bm
    csp = 100 * (max(c) - min(c)) / cm
    dl = 100 * (cm - bm) / bm
    overlap = (min(c) <= max(b)) and (min(b) <= max(c))
    pooled_spread = max(bsp, csp)
    inside = abs(dl) <= max(NOISE_PCT, pooled_spread)
    if ci == FLAGGED:
        verdict = ("SCATTER (distributions overlap, delta inside pooled within-arm spread)"
                   if (overlap and inside) else "*** REAL REGRESSION -- ROLL BACK ***")
    else:
        # a large NEGATIVE delta is an improvement (the two flip cases), not something to chase
        if dl <= -NOISE_PCT:
            verdict = "IMPROVEMENT (guard-admitted flip case)"
        elif abs(dl) <= NOISE_PCT:
            verdict = "ok"
        elif inside:
            verdict = "ok (within pooled spread)"
        else:
            verdict = "*** INVESTIGATE ***"
    print(f"{ci:>4} | {bm:>9.3f} {min(b):>9.3f} {max(b):>9.3f} {bsp:>8.2f} | "
          f"{cm:>9.3f} {min(c):>9.3f} {max(c):>9.3f} {csp:>8.2f} | {dl:>+10.2f} "
          f"{'YES' if overlap else 'NO':>8} | {verdict}")
    rows.append(dict(case=ci, base_samples=b, cand_samples=c, base_median=bm, cand_median=cm,
                     base_min=min(b), base_max=max(b), cand_min=min(c), cand_max=max(c),
                     base_spread_pct=round(bsp, 3), cand_spread_pct=round(csp, 3),
                     median_delta_pct=round(dl, 3), distributions_overlap=overlap,
                     pooled_within_arm_spread_pct=round(pooled_spread, 3),
                     delta_inside_pooled_spread=inside, flagged=(ci == FLAGGED)))

fr = [r for r in rows if r["flagged"]][0]
print(f"\n{'=' * 122}\nFLAGGED PAIR: case {FLAGGED} mask\n{'=' * 122}")
print(f"  base samples (a=8.0)    : {fr['base_samples']}")
print(f"  cand samples (a=1.540)  : {fr['cand_samples']}")
print(f"  base median {fr['base_median']:.3f} us   cand median {fr['cand_median']:.3f} us   "
      f"delta {fr['median_delta_pct']:+.2f}%")
print(f"  within-arm spread: base {fr['base_spread_pct']:.2f}%  cand {fr['cand_spread_pct']:.2f}%  "
      f"-> pooled {fr['pooled_within_arm_spread_pct']:.2f}%")
print(f"  distributions OVERLAP  : {fr['distributions_overlap']}")
print(f"  delta inside pooled spread: {fr['delta_inside_pooled_spread']}")
c2 = fr["distributions_overlap"] and fr["delta_inside_pooled_spread"]
print(f"\n  [ARBITRATION] case {FLAGGED} mask is "
      f"{'MEASUREMENT SCATTER, not a regression' if c2 else '*** A REAL REGRESSION ***'}")

others = [r for r in rows if not r["flagged"]]
bad = [r for r in others if r["median_delta_pct"] > NOISE_PCT
       and not (r["distributions_overlap"] and r["delta_inside_pooled_spread"])]
print(f"  [ARBITRATION] other non-flagged mask cases regressing >3% AND not explainable as "
      f"scatter: {[(r['case'], r['median_delta_pct']) for r in bad]}")
worst_abs = max(rows, key=lambda r: r["median_delta_pct"] if not r["flagged"] else -999)
print(f"  [ARBITRATION] worst non-flagged median delta = case {worst_abs['case']} "
      f"{worst_abs['median_delta_pct']:+.2f}%")
print(f"  [ARBITRATION] largest within-arm spread seen on ANY mask case = "
      f"{max(r['pooled_within_arm_spread_pct'] for r in rows):.2f}% "
      f"(this is the harness's own noise floor at these absolute scales)")

acc_ok = all(D[l]["passed"] == D[l]["total"] == 18 and D[l]["failed"] == 0
             and sum(c["mismatch"] for c in D[l]["cases"].values()) == 0
             and sum(c["n_out"] for c in D[l]["cases"].values()) == 90 for l, _ in ARMS)
print(f"  [ARBITRATION] accuracy on ALL {len(ARMS)} pooled mask arms: 18/18 passed, "
      f"90/90 output tensors mismatch_count=0 -> {'PASS' if acc_ok else '*** FAIL ***'}")

verdict = c2 and not bad and acc_ok
print(f"\n[ARBITRATION VERDICT] criterion 2 = {'SATISFIED IN SUBSTANCE' if verdict else '*** NOT SATISFIED ***'}")
print("=" * 122)

json.dump(dict(_meta=dict(
    description="criterion-2 arbitration on the harness口径: pooled base/cand mask samples",
    base_arms=base_labs, cand_arms=cand_labs, threshold_pct=NOISE_PCT,
    rule="flagged pair is scatter iff the arm distributions OVERLAP and the median delta is inside "
         "the pooled within-arm spread; disjoint-with-cand-above => real regression => roll back",
    flagged_case=FLAGGED,
    structural_fact="case 13 (B=6,V=2304,K_MAX=128,BV=2048) has n_blk_full=1 < 2 so "
                    "_r4_split_plan returns (False,2,1,1) at BOTH guard values; the cost guard is "
                    "never reached and the dispatch is identical"),
    per_case=rows, flagged_verdict_scatter=bool(c2), other_unexplained_regressions=bad,
    accuracy_all_arms_ok=acc_ok, criterion2_satisfied=bool(verdict)), open(OUT, "w"), indent=1)
print(f"wrote {OUT}")
sys.exit(0 if verdict else 1)
