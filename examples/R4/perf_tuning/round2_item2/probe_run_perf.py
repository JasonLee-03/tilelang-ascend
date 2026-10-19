#!/usr/bin/python3
# coding=utf-8
"""Part (b) of the item-2 feasibility study: measured cost split of one K1 block.

Profiles, per case, three K1 variants ALONE (no K2), with the cann-bench profiler recipe
(torch_npu.profiler, ProfilerLevel.Level1, ExportType.Text, schedule(wait=0, warmup=3,
active=5, repeat=1)) and reads `Duration(us)` for `main_kernel` from kernel_details.csv,
taking the median over the 5 active steps -- the same primitive and the same statistic the
harness uses (perf_eval.py:88-115, base/perf_strategy.py:_median).

    topk+merge cost  = T(real) - T(nosort)      <- the only part pruning can ever save
    predicate cost   = T(pruned) - T(real)      <- what a 0%-prune workload still pays
    softmax cost     = T(nosort) - (epilogue + tail + GM load); not separately probed.

Only three variants are profiled (real / nosort / pruned); no `nosoftmax` probe was built,
so the softmax figure is a residual, not an independent measurement.

CAVEAT (documented, not hidden): the harness also launches a CannBenchCacheClean kernel
before each measured step; this probe does not. Absolute microseconds may therefore read
slightly optimistic versus the harness, but every variant is profiled under IDENTICAL
conditions, so the RATIOS (which are all the ceiling formula uses) are unaffected.

Usage: probe_run_perf.py <out.json> [case_id ...]
"""
import csv
import glob
import json
import logging
import os
import shutil
import statistics
import sys
import tempfile

import torch
import torch_npu  # noqa: F401

R4 = "/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4"
HERE = os.path.join(R4, "perf_tuning/round2_item2")
sys.path.insert(0, R4)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(R4, "perf_tuning/round2_bv_probe"))

from bv_static_analysis import CASES  # noqa: E402
from r4_fused_sampling import r4_topk_stats  # noqa: E402
from probe_k1_nosort import k1_nosort  # noqa: E402
from probe_k1_pruned import k1_pruned  # noqa: E402

WARMUP, REPEAT = 3, 5
CASE_BY_ID = {c[0]: c for c in CASES}
DEV = "npu:0"


def profile_k1(fn, args, prof_dir):
    """Run warmup+repeat under torch_npu.profiler; return median main_kernel Duration(us)."""
    os.makedirs(prof_dir, exist_ok=True)
    for name in ("", "torch", "torch_npu", "torch_npu.profiler", "ascend", "profiler"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.ERROR)
        lg.handlers = []
        lg.addHandler(logging.NullHandler())

    torch.npu.synchronize()
    fn(*args)                      # pre-flight: compile + first launch OUTSIDE the window
    torch.npu.synchronize()

    exp = torch_npu.profiler._ExperimentalConfig(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    )
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=WARMUP, active=REPEAT, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir),
        record_shapes=False, profile_memory=False, with_stack=False,
        experimental_config=exp,
    ) as prof:
        for _ in range(WARMUP + REPEAT):
            fn(*args)
            prof.step()
    torch.npu.synchronize()
    try:
        from torch_npu.profiler.analysis.prof_common_func._multi_process_pool import MultiProcessPool
        MultiProcessPool().close_pool(wait=True)
    except Exception:
        pass

    csvs = glob.glob(os.path.join(prof_dir, "**", "kernel_details.csv"), recursive=True)
    if not csvs:
        raise RuntimeError(f"no kernel_details.csv under {prof_dir}")
    durs = []
    with open(max(csvs, key=os.path.getmtime), newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("Name") or "").strip() == "main_kernel":
                durs.append(float((r.get("Duration(us)") or "0").strip()))
    if not durs:
        raise RuntimeError("no main_kernel rows")
    return round(statistics.median(durs), 3), len(durs)


def make_inputs(B, V, vr):
    """Bench-faithful inputs: uniform over the case's declared logits range."""
    torch.manual_seed(0)
    lo, hi = float(vr[0]), float(vr[1])
    z = (torch.rand(B, V, dtype=torch.float64) * (hi - lo) + lo).to(torch.float32)
    ks = torch.full((B,), 40, dtype=torch.int32)
    ps = torch.full((B,), 1.0, dtype=torch.float32)
    mus = torch.zeros((B,), dtype=torch.float32)
    return [t.to(DEV) for t in (z, ks, ps, mus)]


def main():
    out_path = sys.argv[1]
    want = [int(x) for x in sys.argv[2:]] or [2, 3, 7, 8, 15, 16]
    import csv as _csv
    ranges = {}
    with open(os.path.join(
            "/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/.agents/skills/cann-bench",
            "bench_lab/tilelang_ascend_bench/r4_fused_sampling/cases.csv"), newline="") as f:
        for row in _csv.DictReader(f):
            ranges[int(row["case_id"])] = json.loads(row["value_range"])[0]

    results = []
    for cid in want:
        _, B, V, K, BV, nmp = CASE_BY_ID[cid]
        args = make_inputs(B, V, ranges[cid])
        rec = {"case": cid, "B": B, "V": V, "K_MAX": K, "BV": BV,
               "n_blk_full": V // BV, "TAIL": V % BV}
        for label, factory in (("real", r4_topk_stats), ("nosort", k1_nosort),
                               ("pruned", k1_pruned)):
            fn = factory(B, V, K, BV, nmp)
            pd = tempfile.mkdtemp(prefix=f"prof_c{cid}_{label}_", dir="/tmp/opencode/r4_item2")
            us, n = profile_k1(fn, args, pd)
            rec[label + "_us"] = us
            rec[label + "_rows"] = n
            print(f"case{cid:>3} {label:<7} {us:>10.2f} us   (rows={n})", flush=True)
            shutil.rmtree(pd, ignore_errors=True)
        r, ns, pr = rec["real_us"], rec["nosort_us"], rec["pruned_us"]
        rec["topk_merge_us"] = round(r - ns, 3)
        rec["topk_merge_share_of_K1"] = round((r - ns) / r, 6) if r else None
        rec["pruned_minus_real_us"] = round(pr - r, 3)
        rec["pruned_overhead_pct"] = round(100.0 * (pr - r) / r, 4) if r else None
        nb = rec["n_blk_full"]
        rec["topk_merge_us_per_block"] = (round((r - ns) / nb, 4) if nb else None)
        results.append(rec)
        print(f"        -> topk+merge = {rec['topk_merge_us']:.2f} us "
              f"({100*rec['topk_merge_share_of_K1']:.2f}% of K1); "
              f"pruned-real = {rec['pruned_minus_real_us']:+.2f} us "
              f"({rec['pruned_overhead_pct']:+.3f}%)", flush=True)
        del args

    with open(out_path, "w") as f:
        json.dump({"warmup": WARMUP, "repeat": REPEAT, "cases": results}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
