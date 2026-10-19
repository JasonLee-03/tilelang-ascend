"""Round-3 A/B comparison: BASE(19e51e8d) vs CAND(a764e117), same session, 3 passes.

Reads the cann-bench msprof performance reports produced by ab_run.sh and emits:
  * per-case elapsed_us for base1 / cand / base2 and the deltas,
  * the DRIFT CONTROL (base2 vs base1 -- byte-identical code, so any nonzero delta is session
    drift, not effect; round 2 #4 measured 0.3%-1.7% of this),
  * the SPLIT / NON-SPLIT partition required by the task:
      - split cases {3,4,7,8,12,17} carry the whole rev3 effect and are judged individually,
      - the other 12 run BYTE-IDENTICAL code and must therefore come out at exactly 1.000x
        inside the drift band; anything outside it is unexpected coupling and is flagged,
  * geomeans: 36-case combined, fused-18, mask-18, and split-6 per task,
  * a side-by-side against the DESIGN.md section 12.4 model predictions (all of which were
    model estimates, never measured) including the a_launch sensitivity band.

All timings come from msprof kernel_details.csv via the cann-bench harness (warmup=3 repeat=5,
median of per-step sums). NO Python/torch timing is used (hard constraint 2).

Usage: cmp_round3.py <base1_fused> <base1_mask> <cand_fused> <cand_mask> \
                     <base2_fused> <base2_mask> [out.json]
"""
import json
import math
import sys

SPLIT_CASES = {3, 4, 7, 8, 12, 17}
NOISE_PCT = 3.0

# DESIGN.md section 12.4, the a_launch = 3.0 us column: (mask_x, fused_x) predicted speedup.
MODEL_X = {
    1: (1.000, 1.000), 2: (1.000, 1.000), 3: (1.32, 1.49), 4: (1.31, 1.52),
    5: (1.000, 1.000), 6: (1.000, 1.000), 7: (2.57, 3.89), 8: (1.57, 1.87),
    9: (1.000, 1.000), 10: (1.000, 1.000), 11: (1.000, 1.000), 12: (1.31, 1.44),
    13: (1.000, 1.000), 14: (1.000, 1.000), 15: (1.000, 1.000), 16: (1.000, 1.000),
    17: (1.54, 1.72), 18: (1.000, 1.000),
}
# section 12.4 sensitivity band on the geomeans and on the best case (case 7), a_launch 0..8 us.
MODEL_BAND = {
    "mask_geomean": (1.107, 1.196), "fused_geomean": (1.151, 1.292),
    "mask_case7": (2.43, 2.66), "fused_case7": (3.52, 4.15),
    "mask_split6": (1.554, 1.554), "fused_split6": (1.855, 1.855),  # quoted at a_launch=3 only
}


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
        mm = max([r.get("mismatch_count") or 0 for r in (acc.get("output_results") or [])] or [0])
        out[cid] = dict(elapsed_us=c.get("elapsed_us"), status=c.get("status"),
                        acc_passed=acc.get("passed"), mismatch=mm)
    return dict(operator=op["operator"], score=op["score"], passed=op["passed_cases"],
                total=op["total_cases"], failed=op["failed_cases"], cases=out)


def gm(vals):
    vals = [v for v in vals if v and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")


p = sys.argv[1:7]
out_path = sys.argv[7] if len(sys.argv) > 7 else None
A = {k: load(x) for k, x in zip(("b1f", "b1m", "cf", "cm", "b2f", "b2m"), p)}

result = {"_meta": {
    "description": "R4 Stage-3 round-3 same-session A/B: BASE 19e51e8d (pre-rev3) vs "
                   "CAND a764e117 (rev3 split-V). base2 is a re-measurement of BASE and serves "
                   "as the session-drift control (identical code).",
    "data_source": "cann-bench run_evaluation.sh -> msprof kernel_details.csv "
                   "(warmup=3 repeat=5, median of per-step sums). No Python/torch timing.",
    "base_kernel_sha256": "19e51e8d8ee126b920cf48c915d863a9be3a349e2c68cf728d35d482b48bcc57",
    "cand_kernel_sha256": "a764e117ce150d8677096c2e00bde3db3ebc3ec16aba63e4c613d8dffdfbd117",
    "reports": dict(zip(("base1_fused", "base1_mask", "cand_fused", "cand_mask",
                         "base2_fused", "base2_mask"), p)),
    "split_cases": sorted(SPLIT_CASES), "noise_threshold_pct": NOISE_PCT,
}}

for task, kb1, kc, kb2 in (("fused", "b1f", "cf", "b2f"), ("mask", "b1m", "cm", "b2m")):
    b1, c, b2 = A[kb1], A[kc], A[kb2]
    mi = 0 if task == "mask" else 1     # MODEL_X tuples are (mask_x, fused_x)
    print(f"\n{'=' * 100}\n{task}   base1={A[kb1]['operator']}  score base1={b1['score']:.6f} "
          f"cand={c['score']:.6f} base2={b2['score']:.6f}\n{'=' * 100}")
    print(f"{'case':>4} {'path':>7} | {'base1_us':>9} {'cand_us':>9} {'base2_us':>9} | "
          f"{'c/b1 %':>8} {'b2/b1 %':>8} {'driftadj%':>9} | {'meas_x':>7} {'model_x':>7} "
          f"{'m_err%':>7} | {'acc':>5} {'mism':>4}")
    rows = []
    for cid in sorted(b1["cases"]):
        e0, e1, e2 = (b1["cases"][cid]["elapsed_us"], c["cases"][cid]["elapsed_us"],
                      b2["cases"][cid]["elapsed_us"])
        d_cand = 100.0 * (e1 - e0) / e0
        d_drift = 100.0 * (e2 - e0) / e0
        # drift-adjusted: compare cand against the MIDPOINT of the two identical base passes
        e0m = 0.5 * (e0 + e2)
        d_adj = 100.0 * (e1 - e0m) / e0m
        meas_x = e0 / e1 if e1 else float("nan")          # >1 means candidate FASTER
        mx = MODEL_X[cid][mi]
        m_err = 100.0 * (meas_x - mx) / mx if mx else float("nan")
        rows.append(dict(case=cid, path="SPLIT" if cid in SPLIT_CASES else "CURRENT",
                         base1_us=e0, cand_us=e1, base2_us=e2,
                         delta_cand_pct=round(d_cand, 4), delta_drift_pct=round(d_drift, 4),
                         delta_driftadj_pct=round(d_adj, 4), measured_speedup_x=round(meas_x, 5),
                         model_speedup_x=mx, model_err_pct=round(m_err, 2),
                         acc_passed=c["cases"][cid]["acc_passed"],
                         mismatch=c["cases"][cid]["mismatch"]))
        print(f"{cid:>4} {'SPLIT' if cid in SPLIT_CASES else 'curr':>7} | {e0:>9.2f} {e1:>9.2f} "
              f"{e2:>9.2f} | {d_cand:>+8.2f} {d_drift:>+8.2f} {d_adj:>+9.2f} | "
              f"{meas_x:>7.3f} {mx:>7.3f} {m_err:>+7.1f} | "
              f"{str(c['cases'][cid]['acc_passed']):>5} {c['cases'][cid]['mismatch']:>4}")

    sp = [r for r in rows if r["path"] == "SPLIT"]
    cu = [r for r in rows if r["path"] == "CURRENT"]

    def stats(sel, key):
        return (gm([r[key] for r in sel]), max(r["delta_cand_pct"] for r in sel),
                min(r["delta_cand_pct"] for r in sel))

    g_all = gm([r["cand_us"] / r["base1_us"] for r in rows])
    g_sp = gm([r["cand_us"] / r["base1_us"] for r in sp]) if sp else float("nan")
    g_cu = gm([r["cand_us"] / r["base1_us"] for r in cu])
    g_sp_adj = gm([r["cand_us"] / (0.5 * (r["base1_us"] + r["base2_us"])) for r in sp]) if sp else float("nan")
    g_cu_adj = gm([r["cand_us"] / (0.5 * (r["base1_us"] + r["base2_us"])) for r in cu])
    g_drift = gm([r["base2_us"] / r["base1_us"] for r in rows])
    x_all, x_sp, x_cu = 1 / g_all, 1 / g_sp, 1 / g_cu

    print(f"\n  -- {task} aggregates --")
    print(f"  geomean speedup ALL 18       = {x_all:.4f}x   (drift-adjusted {1 / gm([r['cand_us'] / (0.5 * (r['base1_us'] + r['base2_us'])) for r in rows]):.4f}x)")
    print(f"  geomean speedup SPLIT 6      = {x_sp:.4f}x   (drift-adjusted {1 / g_sp_adj:.4f}x)"
          f"   [model section 12.4 @a_launch=3us: {MODEL_BAND[task + '_split6'][0]:.3f}x]")
    print(f"  geomean speedup NON-SPLIT 12 = {x_cu:.4f}x   (drift-adjusted {1 / g_cu_adj:.4f}x)"
          f"   [MUST be 1.000x -- byte-identical code]")
    print(f"  DRIFT CONTROL geomean(base2/base1) = {g_drift:.6f} "
          f"({100 * (g_drift - 1):+.3f}%)  worst single-case drift "
          f"{max(r['delta_drift_pct'] for r in rows):+.3f}%  "
          f"best {min(r['delta_drift_pct'] for r in rows):+.3f}%")
    print(f"  NON-SPLIT worst delta = {stats(cu, 'delta_cand_pct')[1]:+.3f}%  "
          f"best = {stats(cu, 'delta_cand_pct')[2]:+.3f}%  "
          f"n outside +/-{NOISE_PCT}% = {sum(1 for r in cu if abs(r['delta_cand_pct']) > NOISE_PCT)}")
    # ---- NON-SPLIT three-way consistency: is the residual drift or real coupling? ----
    # The 12 non-split cases run BYTE-IDENTICAL device code in both arms, so all three pairwise
    # geomeans below must agree inside the drift band. If cand/base1 and cand/base2 are the same
    # size as base2/base1 (the identical-code control), the residual IS the drift.
    g_c_b1 = gm([r["cand_us"] / r["base1_us"] for r in cu])
    g_c_b2 = gm([r["cand_us"] / r["base2_us"] for r in cu])
    g_b2_b1 = gm([r["base2_us"] / r["base1_us"] for r in cu])
    print(f"  NON-SPLIT pairwise geomean ratios (identical device code in both arms):")
    print(f"      cand /base1 = {g_c_b1:.6f} ({100 * (g_c_b1 - 1):+.3f}%)")
    print(f"      cand /base2 = {g_c_b2:.6f} ({100 * (g_c_b2 - 1):+.3f}%)")
    print(f"      base2/base1 = {g_b2_b1:.6f} ({100 * (g_b2_b1 - 1):+.3f}%)  <-- IDENTICAL-CODE CONTROL")
    resid = abs(100 * (g_c_b2 - 1)) - abs(100 * (g_b2_b1 - 1))
    print(f"      |cand/base2| - |control| = {resid:+.3f} pp  ->  "
          f"{'residual is within the identical-code drift magnitude' if abs(resid) < 1.0 else '*** RESIDUAL EXCEEDS DRIFT -- INVESTIGATE COUPLING ***'}")
    if sp:
        print(f"  SPLIT     worst delta = {stats(sp, 'delta_cand_pct')[1]:+.3f}%  "
              f"best = {stats(sp, 'delta_cand_pct')[2]:+.3f}%  "
              f"n regress>{NOISE_PCT}% = {sum(1 for r in sp if r['delta_cand_pct'] > NOISE_PCT)}")
    print(f"  acc: {sum(1 for r in rows if r['acc_passed'])}/{len(rows)} passed, "
          f"total mismatch_count = {sum(r['mismatch'] for r in rows)}, "
          f"failed_cases = {c['failed']}")
    lo, hi = MODEL_BAND[task + "_geomean"]
    print(f"  MODEL CHECK geomean: measured {x_all:.4f}x vs section 12.4 band [{lo:.3f}x, {hi:.3f}x] "
          f"-> {'INSIDE' if lo <= x_all <= hi else ('BELOW' if x_all < lo else 'ABOVE')} band")
    c7 = [r for r in rows if r["case"] == 7][0]
    lo7, hi7 = MODEL_BAND[task + "_case7"]
    print(f"  MODEL CHECK case7  : measured {c7['measured_speedup_x']:.4f}x vs band "
          f"[{lo7:.2f}x, {hi7:.2f}x] -> "
          f"{'INSIDE' if lo7 <= c7['measured_speedup_x'] <= hi7 else ('BELOW' if c7['measured_speedup_x'] < lo7 else 'ABOVE')}")

    result[task] = dict(operator=c["operator"], score_base1=b1["score"], score_cand=c["score"],
                        score_base2=b2["score"], cases=rows,
                        geomean_speedup_all=x_all, geomean_speedup_split6=x_sp,
                        geomean_speedup_nonsplit12=x_cu,
                        geomean_speedup_split6_driftadj=1 / g_sp_adj,
                        geomean_speedup_nonsplit12_driftadj=1 / g_cu_adj,
                        drift_control_geomean_ratio=g_drift,
                        drift_worst_case_pct=max(r["delta_drift_pct"] for r in rows),
                        nonsplit_worst_delta_pct=stats(cu, "delta_cand_pct")[1],
                        nonsplit_best_delta_pct=stats(cu, "delta_cand_pct")[2],
                        nonsplit_n_outside_noise=sum(1 for r in cu if abs(r["delta_cand_pct"]) > NOISE_PCT),
                        nonsplit_geomean_cand_over_base1=g_c_b1,
                        nonsplit_geomean_cand_over_base2=g_c_b2,
                        nonsplit_geomean_control_base2_over_base1=g_b2_b1,
                        split_worst_delta_pct=stats(sp, "delta_cand_pct")[1] if sp else None,
                        split_best_delta_pct=stats(sp, "delta_cand_pct")[2] if sp else None,
                        split_n_regress_gt_noise=sum(1 for r in sp if r["delta_cand_pct"] > NOISE_PCT) if sp else 0,
                        total_mismatch=sum(r["mismatch"] for r in rows),
                        n_acc_fail=sum(1 for r in rows if not r["acc_passed"]),
                        model_band_geomean=[lo, hi], model_band_case7=[lo7, hi7],
                        model_geomean_verdict=("INSIDE" if lo <= x_all <= hi
                                               else ("BELOW" if x_all < lo else "ABOVE")),
                        model_case7_verdict=("INSIDE" if lo7 <= c7["measured_speedup_x"] <= hi7
                                             else ("BELOW" if c7["measured_speedup_x"] < lo7 else "ABOVE")))

# ---- 36-case combined ----
allr = [r for r in result["fused"]["cases"]] + [r for r in result["mask"]["cases"]]
g36 = gm([r["cand_us"] / r["base1_us"] for r in allr])
g36s = gm([r["cand_us"] / r["base1_us"] for r in allr if r["path"] == "SPLIT"])
g36c = gm([r["cand_us"] / r["base1_us"] for r in allr if r["path"] == "CURRENT"])
g36d = gm([r["base2_us"] / r["base1_us"] for r in allr])
print(f"\n{'=' * 100}\n36-CASE COMBINED (fused 18 + mask 18)\n{'=' * 100}")
print(f"  geomean speedup 36           = {1 / g36:.4f}x")
print(f"  geomean speedup SPLIT 12     = {1 / g36s:.4f}x   (6 cases x 2 tasks)")
print(f"  geomean speedup NON-SPLIT 24 = {1 / g36c:.4f}x   [MUST be 1.000x]")
print(f"  DRIFT CONTROL 36             = {100 * (g36d - 1):+.3f}%")
print(f"  acc: mismatch total = {sum(r['mismatch'] for r in allr)}, "
      f"n_acc_fail = {sum(1 for r in allr if not r['acc_passed'])}")
result["combined36"] = dict(geomean_speedup=1 / g36, geomean_speedup_split12=1 / g36s,
                            geomean_speedup_nonsplit24=1 / g36c,
                            drift_control_pct=100 * (g36d - 1),
                            total_mismatch=sum(r["mismatch"] for r in allr),
                            n_acc_fail=sum(1 for r in allr if not r["acc_passed"]),
                            worst_nonsplit_delta_pct=max(r["delta_cand_pct"] for r in allr if r["path"] == "CURRENT"),
                            best_nonsplit_delta_pct=min(r["delta_cand_pct"] for r in allr if r["path"] == "CURRENT"),
                            worst_split_delta_pct=max(r["delta_cand_pct"] for r in allr if r["path"] == "SPLIT"),
                            best_split_delta_pct=min(r["delta_cand_pct"] for r in allr if r["path"] == "SPLIT"))

if out_path:
    json.dump(result, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")
