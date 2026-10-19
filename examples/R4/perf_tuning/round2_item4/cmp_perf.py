"""Compare two cann-bench performance report pairs (baseline vs candidate).

Usage: cmp_perf.py <base_fused.json> <base_mask.json> <cand_fused.json> <cand_mask.json> [out.json]

Emits a per-case table (elapsed_us, delta%, accuracy mismatch/max_abs_diff) plus
geomean speedup and worst-case regression, and the adoption verdict per the
task's criteria (a)/(b) with a 3% noise threshold.
"""
import json
import math
import sys


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
        worst_mismatch = 0
        worst_maxabs = 0.0
        per_out = {}
        for r in (acc.get("output_results") or []):
            per_out[r["name"]] = dict(mismatch=r.get("mismatch_count"),
                                      total=r.get("total_count"),
                                      max_abs_diff=r.get("max_abs_diff"),
                                      max_diff=r.get("max_diff"),
                                      passed=r.get("passed"))
            worst_mismatch = max(worst_mismatch, r.get("mismatch_count") or 0)
            v = r.get("max_abs_diff")
            worst_maxabs = max(worst_maxabs, float(v) if v is not None else 0.0)
        out[cid] = dict(elapsed_us=c.get("elapsed_us"), status=c.get("status"),
                        acc_passed=acc.get("passed"), mismatch=worst_mismatch,
                        max_abs_diff=worst_maxabs, per_out=per_out)
    return dict(operator=op["operator"], score=op["score"],
                passed=op["passed_cases"], total=op["total_cases"],
                failed=op["failed_cases"], cases=out)


def geomean(vals):
    vals = [v for v in vals if v and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")


base_f, base_m, cand_f, cand_m = (load(p) for p in sys.argv[1:5])
out_path = sys.argv[5] if len(sys.argv) > 5 else None

rows = {}
summary = {}
for task, b, c in (("r4_fused_sampling", base_f, cand_f), ("r4_mask_sample", base_m, cand_m)):
    print(f"\n================ {task} ================")
    print(f"score: base={b['score']:.6f}  cand={c['score']:.6f}  "
          f"delta={100*(c['score']-b['score'])/b['score']:+.3f}%   "
          f"cases {c['passed']}/{c['total']} fail={c['failed']}")
    print(f"{'case':>4} {'base_us':>10} {'cand_us':>10} {'delta%':>8} {'acc':>5} "
          f"{'mism':>5} {'maxabs':>10}")
    ratios, deltas, trows = [], [], []
    for cid in sorted(b["cases"]):
        bb, cc = b["cases"][cid], c["cases"][cid]
        e0, e1 = bb["elapsed_us"], cc["elapsed_us"]
        d = 100.0 * (e1 - e0) / e0 if e0 else float("nan")
        deltas.append(d)
        ratios.append(e1 / e0)
        ok = cc["acc_passed"]
        print(f"{cid:>4} {e0:>10.2f} {e1:>10.2f} {d:>+8.2f} {str(ok):>5} "
              f"{cc['mismatch']:>5} {cc['max_abs_diff']:>10.3g}")
        trows.append(dict(case=cid, base_us=e0, cand_us=e1, delta_pct=round(d, 4),
                          acc_passed=ok, mismatch=cc["mismatch"],
                          max_abs_diff=cc["max_abs_diff"], per_out=cc["per_out"]))
    g = geomean(ratios)
    # elapsed-time geomean ratio: <1 means candidate FASTER
    speed_pct = 100.0 * (1.0 / g - 1.0)      # throughput-style gain
    print(f"  geomean(cand/base elapsed) = {g:.6f}  ->  "
          f"{'speedup' if g < 1 else 'slowdown'} {abs(speed_pct):.3f}%")
    print(f"  worst regression (max delta%) = {max(deltas):+.3f}%   "
          f"best = {min(deltas):+.3f}%   n_regress>3% = {sum(1 for d in deltas if d > 3.0)}")
    summary[task] = dict(base_score=b["score"], cand_score=c["score"],
                         geomean_elapsed_ratio=g, speedup_pct=speed_pct,
                         worst_delta_pct=max(deltas), best_delta_pct=min(deltas),
                         n_regress_gt3=sum(1 for d in deltas if d > 3.0),
                         n_acc_fail=sum(1 for r in trows if not r["acc_passed"]),
                         total_mismatch=sum(r["mismatch"] for r in trows),
                         cases=trows)
    rows[task] = trows

# ---- adoption verdict ----
NOISE = 3.0
verdicts = {}
for task, s in summary.items():
    a = (s["speedup_pct"] > NOISE) and (s["worst_delta_pct"] <= NOISE) and (s["n_acc_fail"] == 0)
    b_ = (abs(s["speedup_pct"]) <= NOISE) and (s["worst_delta_pct"] <= NOISE) and (s["n_acc_fail"] == 0)
    verdicts[task] = dict(criterion_a=a, criterion_b_neutral=b_)
print("\n================ VERDICT (per task) ================")
for t, v in verdicts.items():
    print(f"{t:22s} (a) perf-win={v['criterion_a']}   (b) perf-neutral={v['criterion_b_neutral']}")

if out_path:
    json.dump(dict(summary=summary, verdicts=verdicts, noise_pct=NOISE),
              open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")
