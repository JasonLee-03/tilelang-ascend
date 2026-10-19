"""Item-A adoption analysis: 3-arm same-session A/B on the guard relaxation.

Arms (all rev3 code; they differ ONLY in _SPLIT_A_LAUNCH_US):
  base1 = a764e117  a_launch 8.0    (shipped conservative placeholder)
  cand  = c59a83bc  a_launch 1.540  (measured E1 floor)  <-- arm under test
  base2 = a764e117  identical code to base1 => PURE DRIFT CONTROL

Every timing number comes from the cann-bench msprof report `elapsed_us` (warmup=3 repeat=5,
median of per-step Duration sums). NO Python/torch timing is used anywhere (hard constraint 5).

The Orchestrator ruled that the 36-case geomean uplift (+2.82% projected) is DILUTED -- only 2 of 18
cases change -- so the 3% threshold is applied at the PER-CASE level instead:

  ADOPT  <=>  (1) case 6 AND case 11 each speed up > 3% on BOTH tasks (fused and mask),
              (2) none of the other 34 (task, case) pairs regresses > 3%,
              (3) the drift control (base2 vs base1, byte-identical code) is reported alongside so
                  a "gain" inside the drift band is not mistaken for an effect.

Also emits the requirement-4 accuracy census: 36/36 accuracy.passed and 180/180 output tensors
with mismatch_count == 0, per output tensor, for the CAND arm.

Usage: cmp_itemA.py <base1_fused> <base1_mask> <cand_fused> <cand_mask> \
                    <base2_fused> <base2_mask> [out.json]
"""
import json
import math
import sys

FLIP_CASES = {6, 11}          # the only two the cost guard can admit (preflight P1)
NOISE_PCT = 3.0               # per-case adoption threshold set by the Orchestrator
# Round-3 measured device-time geomeans vs the PRE-rev3 base (19e51e8d). Item A does not re-run
# that arm, so the 36-case geomean-vs-pre-rev3 is composed from the measured cand/base1 ratio and
# round 3's measured anchor. Labelled as composed, never as a fresh measurement.
R3_ANCHOR = {"fused": 1.2796, "mask": 1.1840, "combined": 1.2309}


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
        out[cid] = dict(
            elapsed_us=c.get("elapsed_us"), status=c.get("status"),
            speedup=c.get("speedup"),
            acc_passed=acc.get("passed"),
            n_outputs=len(ors),
            mismatch_total=sum(int(r.get("mismatch_count") or 0) for r in ors),
            worst_mismatch=max([int(r.get("mismatch_count") or 0) for r in ors] or [0]),
            worst_max_abs_diff=max([float(r.get("max_abs_diff") or 0.0) for r in ors] or [0.0]),
            outputs=[dict(name=r.get("name"), dtype=r.get("dtype"), passed=r.get("passed"),
                          mismatch_count=int(r.get("mismatch_count") or 0),
                          total_count=int(r.get("total_count") or 0),
                          max_abs_diff=float(r.get("max_abs_diff") or 0.0)) for r in ors],
        )
    return dict(operator=op["operator"], score=op["score"], passed=op["passed_cases"],
                total=op["total_cases"], failed=op["failed_cases"],
                avg_speedup=op.get("avg_speedup"), cases=out)


def gm(vals):
    vals = [v for v in vals if v and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")


p = sys.argv[1:7]
if len(p) < 6:
    sys.exit(__doc__)
out_path = sys.argv[7] if len(sys.argv) > 7 else None
A = {k: load(x) for k, x in zip(("b1f", "b1m", "cf", "cm", "b2f", "b2m"), p)}
ARMS = {"fused": ("b1f", "cf", "b2f"), "mask": ("b1m", "cm", "b2m")}

print("=" * 118)
print("ITEM-A 3-ARM SAME-SESSION A/B  --  base1(a=8.0) -> cand(a=1.540) -> base2(a=8.0, drift ctrl)")
print("=" * 118)
for k, v in A.items():
    print(f"  {k:4s} {v['operator']:18s} passed={v['passed']}/{v['total']} failed={v['failed']} "
          f"score={v['score']}")

rows, per_task = [], {}
for task, (kb, kc, kd) in ARMS.items():
    b, c, d = A[kb]["cases"], A[kc]["cases"], A[kd]["cases"]
    ids = sorted(set(b) & set(c) & set(d))
    print(f"\n{'-' * 118}")
    print(f"TASK = {task}   ({len(ids)} cases)")
    print(f"{'-' * 118}")
    print(f"{'case':>4} | {'base1_us':>9} {'cand_us':>9} {'base2_us':>9} | "
          f"{'cand/base1':>10} {'delta%':>8} | {'DRIFT b2/b1':>11} {'delta%':>8} | "
          f"{'drift-corr':>10} | verdict")
    xr, xd, xdrift = [], [], []
    for ci in ids:
        ub, uc, ud = b[ci]["elapsed_us"], c[ci]["elapsed_us"], d[ci]["elapsed_us"]
        r = ub / uc if uc else float("nan")
        dl = 100.0 * (uc - ub) / ub if ub else float("nan")
        dr = 100.0 * (ud - ub) / ub if ub else float("nan")
        corr = dl - dr
        xr.append(r); xd.append(dl); xdrift.append(dr)
        if ci in FLIP_CASES:
            verdict = ("ADOPT-GAIN" if dl < -NOISE_PCT else
                       ("*** NO GAIN ***" if abs(dl) <= NOISE_PCT else "*** REGRESSION ***"))
        else:
            verdict = ("ok (must stay ~1.000x: same code path)" if abs(dl) <= NOISE_PCT
                       else "*** UNEXPECTED: >3% move on an unchanged path ***")
        print(f"{ci:>4} | {ub:>9.3f} {uc:>9.3f} {ud:>9.3f} | {r:>10.4f} {dl:>+8.2f} | "
              f"{ud / ub if ub else float('nan'):>11.4f} {dr:>+8.2f} | {corr:>+10.2f} | {verdict}")
        rows.append(dict(task=task, case=ci, base1_us=ub, cand_us=uc, base2_us=ud,
                         ratio_cand_base1=r, delta_pct=dl, drift_pct=dr, drift_corrected_pct=corr,
                         flips=(ci in FLIP_CASES),
                         base1_status=b[ci]["status"], cand_status=c[ci]["status"]))
    print(f"  geomean speedup cand/base1 = {gm(xr):.4f}x   "
          f"geomean drift b2/b1 = {gm([ (b[i]['elapsed_us']/d[i]['elapsed_us']) for i in ids]):.4f}x")
    per_task[task] = dict(geomean_cand_over_base1=gm(xr),
                          geomean_drift=gm([(b[i]["elapsed_us"] / d[i]["elapsed_us"]) for i in ids]),
                          n_cases=len(ids))

# ------------------------------------------------------------------ adoption criteria 1 + 2
print(f"\n{'=' * 118}\nADOPTION CRITERIA (Orchestrator ruling: per-case level, 3% threshold)\n{'=' * 118}")
crit1 = {}
for ci in sorted(FLIP_CASES):
    for task in ("fused", "mask"):
        r = [x for x in rows if x["case"] == ci and x["task"] == task][0]
        gain_pct = -r["delta_pct"]
        ok = gain_pct > NOISE_PCT
        crit1[f"case{ci}_{task}"] = dict(case=ci, task=task,
                                         gain_pct=round(gain_pct, 3),
                                         speedup_x=round(r["ratio_cand_base1"], 5),
                                         drift_pct=round(r["drift_pct"], 3),
                                         drift_corrected_gain_pct=round(-r["drift_corrected_pct"], 3),
                                         over_3pct=ok)
        print(f"  (1) case {ci:>2} {task:5s}: {r['base1_us']:.3f} -> {r['cand_us']:.3f} us  "
              f"speedup={r['ratio_cand_base1']:.4f}x  gain={gain_pct:+.2f}%  "
              f"(drift ctrl {r['drift_pct']:+.2f}%, drift-corrected gain "
              f"{-r['drift_corrected_pct']:+.2f}%)  >3%: {'YES' if ok else '*** NO ***'}")
c1_ok = all(v["over_3pct"] for v in crit1.values())

others = [x for x in rows if not x["flips"]]
worst = max(others, key=lambda x: x["delta_pct"])
n_bad = [x for x in others if x["delta_pct"] > NOISE_PCT]
print(f"\n  (2) the other {len(others)} (task, case) pairs: worst regression = "
      f"case {worst['case']} {worst['task']} {worst['delta_pct']:+.2f}% "
      f"({worst['base1_us']:.3f} -> {worst['cand_us']:.3f} us)")
print(f"      pairs regressing > {NOISE_PCT}% = {len(n_bad)}"
      + (f"  -> {[(x['task'], x['case'], round(x['delta_pct'], 2)) for x in n_bad]}" if n_bad else ""))
absmax = max(others, key=lambda x: abs(x["delta_pct"]))
print(f"      largest |move| on an unchanged path = case {absmax['case']} {absmax['task']} "
      f"{absmax['delta_pct']:+.2f}%  (drift control on the same pair "
      f"{absmax['drift_pct']:+.2f}%)")
c2_ok = len(n_bad) == 0
drift_absmax = max(abs(x["drift_pct"]) for x in rows)
print(f"      drift-control band |base2-base1| max = {drift_absmax:.2f}% "
      f"(round 2 measured 0.3-1.7% on a byte-identical arm)")

# ------------------------------------------------------------------ geomeans
allr = [x["ratio_cand_base1"] for x in rows]
print(f"\n{'=' * 118}\nGEOMEANS\n{'=' * 118}")
print(f"  cand/base1  36-case combined geomean = {gm(allr):.4f}x")
for task in ("fused", "mask"):
    print(f"  cand/base1  {task:5s} 18-case geomean     = {per_task[task]['geomean_cand_over_base1']:.4f}x")
flip_r = [x["ratio_cand_base1"] for x in rows if x["flips"]]
print(f"  cand/base1  the 4 flipping measurements  = {gm(flip_r):.4f}x  "
      f"(individual: {[round(v, 4) for v in flip_r]})")
composed = {}
print(f"\n  COMPOSED with round 3's measured pre-rev3 anchor (NOT a fresh measurement of that arm):")
for task in ("fused", "mask"):
    v = R3_ANCHOR[task] * per_task[task]["geomean_cand_over_base1"]
    composed[task] = v
    print(f"    {task:5s}: {R3_ANCHOR[task]:.4f} x {per_task[task]['geomean_cand_over_base1']:.4f} "
          f"= {v:.4f}x   (round 3 shipped {R3_ANCHOR[task]:.4f}x, "
          f"projected {R3_ANCHOR[task] * (1.0333 if task == 'fused' else 1.0232):.4f}x)")
composed["combined"] = R3_ANCHOR["combined"] * gm(allr)
print(f"    combined: {R3_ANCHOR['combined']:.4f} x {gm(allr):.4f} = {composed['combined']:.4f}x  "
      f"(round 3 shipped {R3_ANCHOR['combined']:.4f}x, projected 1.2656x)")
print(f"    uplift vs the shipped 1.2309x = "
      f"{100.0 * (composed['combined'] / R3_ANCHOR['combined'] - 1.0):+.2f}%")

# ------------------------------------------------------------------ accuracy census (cand arm)
print(f"\n{'=' * 118}\nREQUIREMENT 4 -- cann-bench accuracy census on the CAND arm\n{'=' * 118}")
n_case_pass = n_case_tot = n_out = n_out_bad = 0
worst_mm, worst_mad = 0, 0.0
bad_list = []
for task, (_kb, kc, _kd) in ARMS.items():
    for ci, c in sorted(A[kc]["cases"].items()):
        n_case_tot += 1
        n_case_pass += bool(c["acc_passed"])
        for o in c["outputs"]:
            n_out += 1
            worst_mm = max(worst_mm, o["mismatch_count"])
            worst_mad = max(worst_mad, o["max_abs_diff"])
            if o["mismatch_count"] != 0 or not o["passed"]:
                n_out_bad += 1
                bad_list.append((task, ci, o["name"], o["mismatch_count"], o["passed"]))
        if c["status"] != "success":
            bad_list.append((task, ci, "<status>", c["status"], False))
print(f"  cases accuracy.passed = {n_case_pass}/{n_case_tot}   (required 36/36)")
print(f"  output tensors        = {n_out}   with mismatch_count != 0 or passed=False: {n_out_bad}"
      f"   (required {n_out}/{n_out} clean)")
print(f"  worst mismatch_count  = {worst_mm}   worst max_abs_diff = {worst_mad}")
if bad_list:
    print(f"  *** PROBLEMS: {bad_list}")
c4_ok = (n_case_pass == n_case_tot == 36 and n_out == 180 and n_out_bad == 0)
print(f"  criterion 4 = {'PASS' if c4_ok else '*** FAIL ***'}")

verdict = c1_ok and c2_ok and c4_ok
print(f"\n{'=' * 118}")
print(f"[ITEM-A VERDICT] criterion1(case6/11 both tasks >3%)={'PASS' if c1_ok else 'FAIL'}  "
      f"criterion2(no other pair regresses >3%)={'PASS' if c2_ok else 'FAIL'}  "
      f"criterion4(36/36 + 180/180 mismatch 0)={'PASS' if c4_ok else 'FAIL'}")
print(f"[ITEM-A VERDICT] OVERALL (perf+accuracy part) = {'ADOPT' if verdict else '*** DO NOT ADOPT ***'}")
print("=" * 118)

if out_path:
    json.dump(dict(_meta=dict(
        arms=dict(base1="a764e117 a_launch=8.0", cand="c59a83bc a_launch=1.540",
                  base2="a764e117 a_launch=8.0 (drift control)"),
        timing_source="cann-bench msprof report elapsed_us (warmup=3 repeat=5, median of per-step "
                      "Duration sums). No Python/torch timing.",
        threshold_pct=NOISE_PCT, flip_cases=sorted(FLIP_CASES),
        r3_anchor=R3_ANCHOR,
        composed_note="geomean-vs-pre-rev3 is COMPOSED from the measured cand/base1 ratio and "
                      "round 3's measured anchor; the pre-rev3 arm was NOT re-run in item A."),
        reports={k: dict(operator=v["operator"], score=v["score"], passed=v["passed"],
                         total=v["total"], failed=v["failed"]) for k, v in A.items()},
        per_pair=rows, per_task=per_task, criterion1=crit1, criterion1_ok=c1_ok,
        criterion2_ok=c2_ok, criterion2_worst=dict(task=worst["task"], case=worst["case"],
                                                   delta_pct=worst["delta_pct"]),
        criterion2_n_over_threshold=len(n_bad),
        drift_band_max_abs_pct=drift_absmax,
        geomean_cand_over_base1=dict(combined=gm(allr), **{t: per_task[t]["geomean_cand_over_base1"]
                                                            for t in per_task}),
        geomean_composed_vs_prerev3=composed,
        accuracy=dict(cases_passed=n_case_pass, cases_total=n_case_tot, outputs=n_out,
                      outputs_bad=n_out_bad, worst_mismatch=worst_mm,
                      worst_max_abs_diff=worst_mad, problems=bad_list, ok=c4_ok),
        verdict_adopt=bool(verdict)),
        open(out_path, "w"), indent=1)
    print(f"wrote {out_path}")
