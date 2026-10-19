#!/usr/bin/python3
# coding=utf-8
"""Fold the item-2 ceiling onto the 36-case (18 x 2 tasks) geomean.

    ceiling_case = prune_rate_case x topk_share_of_K1 x K1_share_of_task
    ceiling_36   = 1/geomean(1 - ceiling_case) - 1     (geomean of per-case speedups)

Two numbers are reported:
  * ACTUAL         -- prune_rate measured on the real bench inputs (prune_rate.py).
  * COUNTERFACTUAL -- prune_rate forced to 1.0, i.e. the best a PERFECT ORACLE pruning
                      scheme could do. Only the 6 cases whose topk_share was probed with
                      msprof get a number; the rest are "not probed" (no extrapolation).

K1_share_of_task comes from perf_tuning/round2_item4/k1split_{fused,mask}_cand.json = the
DELIVERED kernel (19e51e8d) measured with msprof and position-split into K1/K2. That is a
previous session, so absolute us carry the known ~0.3-1.7% session drift; the K1/K2 RATIO
used here is intra-session, so the drift largely cancels.

Usage: ceiling_fold.py <out.json>
"""
import json
import math
import sys

R4 = "/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4"
PROBE = json.load(open("/tmp/opencode/r4_item2/probe_perf.json"))
PRUNE = json.load(open(R4 + "/perf_tuning/round2_item2/prune_rate_result.json"))
K1SPLIT = {t: json.load(open(f"/tmp/opencode/r4_item4/k1split_{t}_cand.json"))["per_case"]
           for t in ("fused", "mask")}

TOPK_SHARE = {c["case"]: c["topk_merge_share_of_K1"] for c in PROBE["cases"]}
PRUNE_RATE = {c["case_id"]: (c["prune_rate"] or 0.0) for c in PRUNE["bench_cases"]}
CASES = sorted(PRUNE_RATE)


def geomean_speedup_pct(gains):
    gains = [g for g in gains if g is not None]
    if not gains:
        return None
    ratios = [1.0 - g for g in gains]
    return 100.0 * (1.0 / math.exp(sum(math.log(r) for r in ratios) / len(ratios)) - 1.0)


def main():
    out = {"per_case": [],
           "k1share_source": "round2_item4/k1split_*_cand.json (delivered 19e51e8d, msprof)"}
    hdr = (f"{'case':>4} {'prune':>6} {'topk_sh':>8} | "
           f"{'K1sh_fus':>9} {'act_fus':>8} {'cf_fus':>8} | "
           f"{'K1sh_msk':>9} {'act_msk':>8} {'cf_msk':>8}")
    print(hdr)
    print("-" * len(hdr))
    actual_all, cf_probed = [], []
    for ci in CASES:
        pr = PRUNE_RATE[ci]
        ts = TOPK_SHARE.get(ci)
        rec = {"case": ci, "prune_rate": pr, "topk_share_probed": ts}
        cells = []
        for task in ("fused", "mask"):
            k1 = K1SPLIT[task][str(ci)]["k1_us"]
            k2 = K1SPLIT[task][str(ci)]["k2_us"]
            share = k1 / (k1 + k2)
            act = pr * (ts or 0.0) * share
            cf = (ts * share) if ts is not None else None
            rec[f"k1_share_{task}"] = round(share, 6)
            rec[f"k1_us_{task}"] = k1
            rec[f"k2_us_{task}"] = k2
            rec[f"actual_ceiling_{task}"] = round(act, 8)
            rec[f"counterfactual_ceiling_{task}"] = (round(cf, 8) if cf is not None else None)
            actual_all.append(act)      # one entry per (case, task) -> 36 total
            cf_probed.append(cf)
            cells.append((share, act, cf))
        out["per_case"].append(rec)
        (s0, a0, c0), (s1, a1, c1) = cells
        print(f"{ci:>4} {pr:>6.3f} {('n/p' if ts is None else format(ts, '.4f')):>8} | "
              f"{s0:>9.4f} {a0*100:>7.4f}% "
              f"{('n/p' if c0 is None else format(c0*100, '.3f') + '%'):>8} | "
              f"{s1:>9.4f} {a1*100:>7.4f}% "
              f"{('n/p' if c1 is None else format(c1*100, '.3f') + '%'):>8}")

    g_act = geomean_speedup_pct(actual_all)
    g_cf = geomean_speedup_pct(cf_probed)
    out["n_36case_entries_actual"] = len(actual_all)
    out["geomean_36case_actual_pct"] = g_act
    out["geomean_counterfactual_probed_pct"] = g_cf
    out["n_probed_cases"] = len(TOPK_SHARE)
    out["probed_cases"] = sorted(TOPK_SHARE)
    print(f"\nACTUAL 36-case geomean ceiling            = {g_act:.6f}%   (gate: must exceed 3%)")
    print(f"COUNTERFACTUAL (100% prune), {len(TOPK_SHARE)} probed cases = {g_cf:.3f}%")
    print(f"VERDICT: {'GO' if (g_act or 0) > 3.0 else 'NO-GO'}")
    json.dump(out, open(sys.argv[1], "w"), indent=2)
    print(f"wrote {sys.argv[1]}")


if __name__ == "__main__":
    main()
