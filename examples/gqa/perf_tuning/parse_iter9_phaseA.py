"""Iter9 Phase-A parser — BASE-arm tout/e2e census on the wave-2 domains.

Scans msprof_op_iter9_PA_{shape}_base_s{n} (layout B: kernel_kernel:0 =
transpose_qkv, main_kernel:0 = main, kernel_kernel:1 = transpose_out) and
reports per shape:
  e2e   = SUM of TaskDur over all instances (B-matrix口径, iter7/8 identical)
  tout  = transpose_out TaskDur + Block Dim  (the wave-2 benefit pool)
  main  = main kernel TaskDur                (the strided-write cost base)
  tqkv  = transpose_qkv TaskDur              (untouched by wave-2)
  tout share of e2e  =>  marginal-gain estimate for the go/no-go verdict
Artifact filter (impossible PipeUtilization rows) via parse_msprof_iter6.
Freq/rc audit lines included.

Run: python3 examples/gqa/perf_tuning/parse_iter9_phaseA.py
"""

import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parse_msprof_iter6 import find_csvs, parse_kernel  # noqa: E402

SHAPES = ["r1", "c5", "c19", "c9", "c8", "c10", "c6"]
DOMAIN = {
    "r1": "per_head", "c5": "per_head", "c6": "per_head",
    "c19": "prefill_expert",
    "c8": "g2stack", "c9": "g2stack", "c10": "g2stack",
}


def parse_dir(base):
    out = {}
    for label, kinds in find_csvs(base).items():
        r = parse_kernel(kinds)
        if r:
            out[label] = r
    return out


def main():
    rows = {}
    dropped_total = 0
    freq_bad = []
    for shape in SHAPES:
        per = []
        for d in sorted(glob.glob(os.path.join(HERE, f"msprof_op_iter9_PA_{shape}_base_s*"))):
            rec = parse_dir(d)
            if not rec:
                print(f"[WARN] empty parse: {d}")
                continue
            e2e = sum(v["dur_us"] for v in rec.values())
            tout = rec.get("kernel_kernel:1")
            main_i = rec.get("main_kernel:0")
            tqkv = rec.get("kernel_kernel:0")
            for v in rec.values():
                dropped_total += v.get("rows_dropped", 0)
                f = v.get("freq", "")
                if f and f != "1650/1650":
                    freq_bad.append((d, v.get("op_name"), f))
            per.append({
                "dir": os.path.basename(d),
                "e2e": e2e,
                "tout": tout["dur_us"] if tout else None,
                "tout_blk": tout.get("block_dim") if tout else None,
                "main": main_i["dur_us"] if main_i else None,
                "tqkv": tqkv["dur_us"] if tqkv else None,
                "n_inst": len(rec),
            })
        rows[shape] = per

    print(f"{'shape':6s} {'domain':15s} {'e2e us (samples)':28s} {'tout us (blk)':22s} "
          f"{'main us':22s} {'tqkv us':16s} tout%e2e")
    summary = {}
    for shape in SHAPES:
        per = rows[shape]
        if not per:
            continue
        e2es = [p["e2e"] for p in per]
        tous = [p["tout"] for p in per if p["tout"] is not None]
        mains = [p["main"] for p in per if p["main"] is not None]
        tqkvs = [p["tqkv"] for p in per if p["tqkv"] is not None]
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
        share = 100 * mean(tous) / mean(e2es) if tous else 0.0
        summary[shape] = {
            "domain": DOMAIN[shape],
            "e2e_samples": [round(x, 2) for x in e2es],
            "e2e_mean": round(mean(e2es), 2),
            "tout_samples": [round(x, 2) for x in tous],
            "tout_mean": round(mean(tous), 2),
            "tout_blk": per[0]["tout_blk"],
            "main_samples": [round(x, 2) for x in mains],
            "main_mean": round(mean(mains), 2),
            "tqkv_mean": round(mean(tqkvs), 2),
            "tout_share_pct": round(share, 1),
        }
        s = summary[shape]
        tout_txt = "/".join(f"{x:.2f}" for x in tous) + f" (blk{per[0]['tout_blk']})"
        print(f"{shape:6s} {DOMAIN[shape]:15s} "
              f"{('/'.join(f'{x:.2f}' for x in e2es)):28s} "
              f"{tout_txt:22s} "
              f"{('/'.join(f'{x:.2f}' for x in mains)):22s} "
              f"{('/'.join(f'{x:.2f}' for x in tqkvs)):16s} {share:5.1f}%")

    print(f"\nartifact rows dropped total: {dropped_total}")
    print(f"non-rated-freq instances: {freq_bad if freq_bad else 'NONE (all 1650/1650)'}")

    import json
    with open(os.path.join(HERE, "phaseA_iter9_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("WROTE phaseA_iter9_summary.json")


if __name__ == "__main__":
    main()
