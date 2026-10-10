"""Parse msprof timeline output for bench_mha.py runs (tooling, NOT a deliverable).

Caliber (validated against cann-bench case1 anchor: op_statistic Avg(main_kernel)
= 46.925us == cann-bench T_cand 46.9us):
  per-call device time = Task Duration(us) of the single main_kernel MIX_AIC
  row per launch (mix task bundles AIC+AIV); case value = MEDIAN over the
  measured calls (first `warmup` launches dropped).

Usage:
  python examples/mha/perf_tuning/parse_prof.py <prof_output_dir> \
      --case case7 --warmup 5 --iters 20 [--json out.json]
"""

import argparse
import csv
import glob
import json
import os
import statistics
import sys


def _latest(pattern):
    c = glob.glob(pattern)
    return max(c, key=os.path.getmtime) if c else None


def find_outputs(prof_dir):
    op_summary = _latest(os.path.join(prof_dir, "PROF_*", "mindstudio_profiler_output", "op_summary_*.csv"))
    if op_summary is None:
        op_summary = _latest(os.path.join(prof_dir, "**", "op_summary_*.csv"))
    op_stat = None
    if op_summary:
        op_stat = _latest(os.path.join(os.path.dirname(op_summary), "op_statistic_*.csv"))
    return op_summary, op_stat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prof_dir")
    p.add_argument("--case", required=True)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--kernel-filter", default="main_kernel")
    p.add_argument("--json", default=None)
    args = p.parse_args()

    op_summary, op_stat = find_outputs(args.prof_dir)
    if not op_summary:
        print(f"[PARSE_FAIL] no op_summary_*.csv under {args.prof_dir}")
        sys.exit(2)

    durs = []
    task_types = {}
    with open(op_summary, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            op = (row.get("Op Name") or "").strip()
            if args.kernel_filter not in op:
                continue
            try:
                d = float((row.get("Task Duration(us)") or "0").strip())
            except ValueError:
                continue
            tt = (row.get("Task Type") or "").strip()
            task_types[tt] = task_types.get(tt, 0) + 1
            durs.append(d)

    n_calls = args.warmup + args.iters
    if not durs:
        print(f"[PARSE_FAIL] {op_summary}: 0 '{args.kernel_filter}' rows")
        sys.exit(2)
    if len(durs) != n_calls:
        print(f"[PARSE_WARN] {len(durs)} main_kernel launches != expected {n_calls} ({args.case})")

    measured = durs[args.warmup :] if len(durs) > args.warmup else durs
    stat_row = None
    if op_stat and os.path.isfile(op_stat):
        with open(op_stat, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if args.kernel_filter in (row.get("OP Type") or ""):
                    stat_row = {
                        "count": row.get("Count"),
                        "avg_us": row.get("Avg Time(us)"),
                        "min_us": row.get("Min Time(us)"),
                        "max_us": row.get("Max Time(us)"),
                    }
                    break

    result = {
        "case": args.case,
        "source": op_summary,
        "launches_total": len(durs),
        "task_types": task_types,
        "warmup": args.warmup,
        "iters": args.iters,
        "per_call_us": {
            "median": round(statistics.median(measured), 3),
            "mean": round(statistics.fmean(measured), 3),
            "min": round(min(measured), 3),
            "max": round(max(measured), 3),
            "stdev": round(statistics.stdev(measured), 3) if len(measured) > 1 else 0.0,
        },
        "op_statistic_all_launches": stat_row,
    }
    out = json.dumps(result, indent=2)
    print(out)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(out + "\n")


if __name__ == "__main__":
    main()
