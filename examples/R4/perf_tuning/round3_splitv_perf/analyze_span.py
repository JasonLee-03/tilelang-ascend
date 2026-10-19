"""Round-3 honesty check: the SCORED口径 (sum of device Durations) vs the WALL-CLOCK SPAN口径.

cann-bench scores `elapsed_us` = SUM of the per-step `main_kernel` `Duration(us)` values. Round 2
verified that identity and ksplit_fine.py re-verifies it here for all 18 cases x 2 tasks (the
per-case `total_us` reproduces the report's `elapsed_us` exactly). That sum EXCLUDES the idle time
between launches.

While reading the inter-kernel gaps for the a_launch analysis it turned out that those gaps are
LARGE and ASYMMETRIC between the two arms: the split path issues 3 (mask) or 4 (fused) launches
instead of 2, and each launch is separated by ~300-650 us of host-side time. This script measures
both口径 side by side so the report can state plainly what rev3 does and does not improve.

  scored_us  = median over the 5 active steps of  SUM(Duration of main_kernel rows in the step)
  span_us    = median over the 5 active steps of  (last main_kernel end - first main_kernel start)
  gap_us     = span_us - scored_us   (host-side idle between launches, inside the step)

No conclusion in this round is drawn from span_us; it is reported because omitting it would make
the rev3 result look better end-to-end than it is.

Usage: analyze_span.py <base1_fused.csv> <cand_fused.csv> <base1_mask.csv> <cand_mask.csv> [out]
"""
import csv
import json
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
R4 = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (R4, os.path.join(R4, "bench", "hap_thw_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from r4_thw_common import load_cases  # noqa: E402

STEPS_PER_CASE, WARMUP = 8, 3
out_path = sys.argv[5] if len(sys.argv) > 5 else os.path.join(HERE, "span_vs_scored.json")
ARMS = [("base1_fused", sys.argv[1], "fused"), ("cand_fused", sys.argv[2], "fused"),
        ("base1_mask", sys.argv[3], "mask"), ("cand_mask", sys.argv[4], "mask")]


def analyse(path, task):
    cases = {c["case_id"]: c for c in
             load_cases("r4_mask_sample" if task == "mask" else "r4_fused_sampling", verbose=False)}
    by = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["Name"].strip() != "main_kernel":
            continue
        by[int(r["Step Id"].strip())].append(
            (float(r["Start Time(us)"].strip()), float(r["Duration(us)"])))
    out = {}
    for step in sorted(by):
        ci, rep = step // STEPS_PER_CASE + 1, step % STEPS_PER_CASE
        if ci not in cases or rep < WARMUP:
            continue
        seq = sorted(by[step])
        scored = sum(d for _, d in seq)
        span = max(s + d for s, d in seq) - seq[0][0]
        out.setdefault(ci, []).append((scored, span, len(seq)))
    return {ci: dict(scored_us=round(statistics.median(x[0] for x in v), 3),
                     span_us=round(statistics.median(x[1] for x in v), 3),
                     gap_us=round(statistics.median(x[1] - x[0] for x in v), 3),
                     n_launch=v[0][2], n_steps=len(v))
            for ci, v in out.items()}


D = {lab: analyse(p, t) for lab, p, t in ARMS}
res = {}
print("=" * 118)
print("SCORED口径 (sum of device Durations == cann-bench elapsed_us)  vs  WALL-CLOCK SPAN口径")
print("=" * 118)
for task, bl, cl in (("fused", "base1_fused", "cand_fused"), ("mask", "base1_mask", "cand_mask")):
    b, c = D[bl], D[cl]
    print(f"\n########## {task} ##########")
    print(f"{'case':>4} | {'#ln b':>5} {'#ln c':>5} | {'scored_b':>9} {'scored_c':>9} {'x':>6} | "
          f"{'span_b':>9} {'span_c':>9} {'x':>6} | {'gap_b':>8} {'gap_c':>8}")
    sp_sc, sp_sp = [], []
    for ci in sorted(b):
        rb, rc = b[ci], c[ci]
        xsc = rb["scored_us"] / rc["scored_us"]
        xsp = rb["span_us"] / rc["span_us"]
        sp_sc.append(xsc)
        sp_sp.append(xsp)
        print(f"{ci:>4} | {rb['n_launch']:>5} {rc['n_launch']:>5} | {rb['scored_us']:>9.2f} "
              f"{rc['scored_us']:>9.2f} {xsc:>6.3f} | {rb['span_us']:>9.2f} {rc['span_us']:>9.2f} "
              f"{xsp:>6.3f} | {rb['gap_us']:>8.2f} {rc['gap_us']:>8.2f}")

    def _gm(v):
        import math
        return math.exp(sum(math.log(x) for x in v) / len(v))
    split = [ci for ci in sorted(b) if b[ci]["n_launch"] != c[ci]["n_launch"]]
    curr = [ci for ci in sorted(b) if ci not in split]
    print(f"  geomean SCORED speedup: all={_gm(sp_sc):.4f}x  "
          f"launch-count-changed({split})={_gm([b[i]['scored_us'] / c[i]['scored_us'] for i in split]):.4f}x  "
          f"unchanged={_gm([b[i]['scored_us'] / c[i]['scored_us'] for i in curr]):.4f}x")
    print(f"  geomean SPAN   speedup: all={_gm(sp_sp):.4f}x  "
          f"launch-count-changed={_gm([b[i]['span_us'] / c[i]['span_us'] for i in split]):.4f}x  "
          f"unchanged={_gm([b[i]['span_us'] / c[i]['span_us'] for i in curr]):.4f}x")
    print(f"  host gap per launch (base arm, cases whose first kernel is SHORT): "
          f"{statistics.median([b[i]['gap_us'] / max(1, b[i]['n_launch'] - 1) for i in curr]):.1f} us")
    res[task] = dict(
        per_case={str(ci): dict(base=b[ci], cand=c[ci],
                                scored_speedup_x=round(b[ci]["scored_us"] / c[ci]["scored_us"], 5),
                                span_speedup_x=round(b[ci]["span_us"] / c[ci]["span_us"], 5))
                  for ci in sorted(b)},
        geomean_scored_all=_gm(sp_sc), geomean_span_all=_gm(sp_sp),
        geomean_scored_launchchanged=_gm([b[i]["scored_us"] / c[i]["scored_us"] for i in split]),
        geomean_span_launchchanged=_gm([b[i]["span_us"] / c[i]["span_us"] for i in split]),
        geomean_scored_unchanged=_gm([b[i]["scored_us"] / c[i]["scored_us"] for i in curr]),
        geomean_span_unchanged=_gm([b[i]["span_us"] / c[i]["span_us"] for i in curr]),
        launch_changed_cases=split)

print("\n" + "=" * 118)
print("INTERPRETATION (stated plainly, no spin)")
print("=" * 118)
print("""
  * The BASE arm already pays ~300-650 us of host-side time per launch for every case whose first
    kernel is SHORTER than that overhead. It is hidden only when a kernel runs long enough for the
    host to queue the next one asynchronously (base case7/15/16/17/18: gap ~3 us because K1 alone
    runs 390-860 us).
  * rev3 splits one long kernel into several SHORT ones, so on the split cases the overhead that
    was previously hidden becomes exposed: the SCORED device time drops sharply (case7 fused
    728->179 us) while the SPAN grows, because 4 launches x ~300-540 us of host time now dominate.
  * Therefore: rev3's win is REAL in device-time / core-utilisation terms (which is what cann-bench
    scores, what the roofline attainment measures, and what a captured ACL graph would execute),
    but it is NOT visible -- and reverses -- in eager-mode end-to-end wall clock under this
    harness, where Python/ACL launch preparation costs ~2 orders of magnitude more than the
    kernels themselves.
  * This is a HOST-side software-overhead finding, not a device regression: no kernel got slower.
    It is reported because the task requires that rev3 not be made to look better than it is.
""")
json.dump(dict(_meta=dict(
    description="scored (sum of Durations) vs wall-clock span, both arms, all 36 measurements",
    scored_definition="median over 5 active steps of SUM(main_kernel Duration(us)) in the step; "
                      "identical to cann-bench elapsed_us (verified: ksplit total_us == report)",
    span_definition="median over 5 active steps of (last main_kernel end - first main_kernel start)",
    used_for_conclusions=False,
    reason="reported for honesty; the scored口径 is the contractual metric and the span is "
           "dominated by eager-mode host launch preparation"), results=res),
    open(out_path, "w"), indent=1)
print(f"wrote {out_path}")
