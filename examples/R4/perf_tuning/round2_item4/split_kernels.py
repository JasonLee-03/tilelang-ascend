"""Split msprof kernel_details.csv into per-kernel-name timings, replicating the
cann-bench measurement口径: group by Step Id, sum durations per kernel name within
a step, then take the MEDIAN across steps.

Usage: split_kernels.py <csv> [<csv> ...]
"""
import csv
import statistics
import sys
from collections import defaultdict

for path in sys.argv[1:]:
    rows = list(csv.DictReader(open(path)))
    # per (step, kernel-name) summed duration
    per_step = defaultdict(lambda: defaultdict(float))
    for r in rows:
        step = r["Step Id"].strip()
        name = r["Name"].strip()
        try:
            dur = float(r["Duration(us)"])
        except (ValueError, KeyError):
            continue
        per_step[step][name] += dur
    # median across steps, per kernel name
    agg = defaultdict(list)
    for step, d in per_step.items():
        for name, dur in d.items():
            agg[name].append(dur)
    med = {n: statistics.median(v) for n, v in agg.items()}
    total = sum(med.values())
    print(f"\n===== {path.split('/')[-5] if len(path.split('/')) > 5 else path} =====")
    print(f"  steps={len(per_step)}  distinct kernels={len(med)}  SUM-of-medians={total:.3f} us")
    for n, v in sorted(med.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * v / total if total else 0.0
        short = n if len(n) < 70 else n[:67] + "..."
        print(f"    {v:10.3f} us  {pct:5.1f}%  {short}")
