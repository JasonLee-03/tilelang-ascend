"""Round-3 item (2) follow-up: EMPIRICAL cost/benefit of relaxing the split guard.

analyze_a_launch.py showed that lowering `_SPLIT_A_LAUNCH_US` from the shipped conservative 8.0 us
to the measured floor can only ever admit TWO more cases -- 6 and 11 -- because the other ten
non-split cases are blocked STRUCTURALLY (S<2 or n_blk_full<2), not by the cost guard. The split set
saturates at 8 cases for any a_launch <= 2.74 us.

This probe measures what those two cases would actually do, instead of estimating it:
  arm A = delivered guard (a_launch 8.0) -> both cases take the CURRENT 2-launch path
  arm B = relaxed guard  (a_launch 1.54) -> both cases take the SPLIT path (S=3 / S=2)

THE DELIVERED KERNEL FILE IS NOT MODIFIED. `_SPLIT_A_LAUNCH_US` is monkey-patched on the imported
module object inside this process only, so the on-disk sha256 is untouched (verified before/after).

Measurement口径 matches cann-bench: torch_npu.profiler ProfilerLevel.Level1, schedule(wait=0,
warmup=3, active=5, repeat=1), read `main_kernel` `Duration(us)` from kernel_details.csv, SUM the
rows inside each Step Id, then take the MEDIAN over the 5 active steps. No Python/torch timing.

CAVEAT (same one round 2 documented, restated here): the harness also launches a CannBenchCacheClean
kernel before each measured step and this probe does not, so absolute microseconds may read slightly
optimistic versus the harness. Both arms are profiled under IDENTICAL conditions, so the RATIO --
which is the only thing this probe is used for -- is unaffected.

The probe also checks arm A vs arm B BITWISE on all 5 outputs. rev3's split path is bitwise by
construction for the shapes it currently serves, but S=3/NBS=1 (case 6) and S=2/NBS=1 (case 11) are
specializations the delivered static-assertion table does not cover, so equivalence must be checked
rather than assumed before anyone relaxes the guard.

Usage: probe_guard_relax.py [out.json]
"""
import csv
import glob
import hashlib
import json
import logging
import os
import shutil
import statistics
import sys
import tempfile
from collections import defaultdict

import torch
import torch_npu  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
R4 = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (R4, os.path.join(R4, "bench", "hap_thw_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r4_fused_sampling as M  # noqa: E402

out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "guard_relax_probe.json")
KERNEL = os.path.join(R4, "r4_fused_sampling.py")
WARMUP, REPEAT = 3, 5
DEV = "npu:0"
A_SHIPPED = 8.0
A_MEASURED = 1.54          # E1 floor from analyze_a_launch.py (K2gB min Duration)
FLIP_CASES = [6, 11]       # the ONLY two the cost guard can ever admit
SHAPES = {6: dict(B=8, V=10240, K_MAX=256, BV=4096, nmp=False),
          11: dict(B=8, V=8192, K_MAX=256, BV=4096, nmp=False)}


def sha():
    return hashlib.sha256(open(KERNEL, "rb").read()).hexdigest()


def quiet():
    for name in ("", "torch", "torch_npu", "torch_npu.profiler", "ascend", "profiler"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.ERROR)
        lg.handlers = []
        lg.addHandler(logging.NullHandler())


def profile_wrapper(fn, prof_dir, n_launch):
    """Median over the active steps of the SUM of main_kernel Durations inside each step.

    The harness CSV carries a `Step Id` column; this probe's profiler export does not, so steps are
    recovered by chunking the launch-ordered `main_kernel` rows into groups of `n_launch` (the
    launch count is known exactly from the arm's plan: 2 for CURRENT, 3 for split-mask, 4 for
    split-fused). The chunking is ASSERTED to divide evenly, so a launch-count surprise raises
    instead of silently mis-attributing durations.
    """
    os.makedirs(prof_dir, exist_ok=True)
    quiet()
    torch.npu.synchronize()
    fn()                                   # pre-flight: JIT compile + first launch OUTSIDE window
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
            fn()
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
    rows = []
    with open(max(csvs, key=os.path.getmtime), newline="") as f:
        rd = csv.DictReader(f)
        has_step = "Step Id" in (rd.fieldnames or [])
        for r in rd:
            if (r.get("Name") or "").strip() != "main_kernel":
                continue
            rows.append((float((r.get("Duration(us)") or "0").strip()),
                         int(r.get("Block Num") or 0),
                         float((r.get("Start Time(us)") or "0").strip()),
                         int(r["Step Id"].strip()) if has_step else None))
    if not rows:
        raise RuntimeError("no main_kernel rows")
    if has_step:
        per_step = defaultdict(list)
        for d, b, s, st in rows:
            per_step[st].append((d, b, s))
        groups = [sorted(per_step[k], key=lambda t: t[2]) for k in sorted(per_step)]
    else:
        if len(rows) % n_launch != 0:
            raise RuntimeError(f"launch-count mismatch: {len(rows)} main_kernel rows is not a "
                               f"multiple of the planned n_launch={n_launch}")
        rows.sort(key=lambda t: t[2])                     # launch order by Start Time
        groups = [[(d, b, s) for d, b, s, _ in rows[i:i + n_launch]]
                  for i in range(0, len(rows), n_launch)]
    sums = [sum(d for d, _, _ in g) for g in groups]
    blocks = [[b for _, b, _ in g] for g in groups]
    if len(sums) < REPEAT:
        raise RuntimeError(f"only {len(sums)} steps recovered, expected >= {REPEAT}")
    return (round(statistics.median(sums), 3), len(sums), blocks[-1],
            round(min(sums), 3), round(max(sums), 3))


def bits(t):
    t = t.detach().cpu()
    return t.view(torch.int32) if t.dtype == torch.float32 else t


def make_inputs(B, V):
    torch.manual_seed(0)
    z = torch.randn(B, V, dtype=torch.float32) * 8.0
    ks = torch.full((B,), 40, dtype=torch.int32)
    ps = torch.full((B,), 0.95, dtype=torch.float32)
    mus = torch.zeros((B,), dtype=torch.float32)
    rnd = torch.arange(B, dtype=torch.int32) * 7919 + 13
    return [t.to(DEV) for t in (z, ks, ps, mus, rnd)]


sha_before = sha()
print(f"[probe] delivered kernel sha256 BEFORE = {sha_before}")
print(f"[probe] arms: A = shipped a_launch {A_SHIPPED} us (CURRENT path), "
      f"B = measured a_launch {A_MEASURED} us (SPLIT path)")
print(f"[probe] cases probed = {FLIP_CASES} (the only two the cost guard can ever admit)\n")

results = []
for cid in FLIP_CASES:
    sp = SHAPES[cid]
    B, V, K, BV, nmp = sp["B"], sp["V"], sp["K_MAX"], sp["BV"], sp["nmp"]
    z, ks, ps, mus, rnd = make_inputs(B, V)
    for task in ("fused", "mask"):
        rec = dict(case=cid, task=task, B=B, V=V, K_MAX=K, BV=BV, need_min_p=nmp)
        outs = {}
        for arm, a in (("A_shipped", A_SHIPPED), ("B_relaxed", A_MEASURED)):
            M._SPLIT_A_LAUNCH_US = a
            plan = M._r4_split_plan(B, V, K, BV)
            rec[arm] = dict(a_launch_us=a, use_split=plan[0], S=plan[1], NBS=plan[2],
                            S_TAIL=plan[3])
            fuse = task == "fused"

            def call(fuse=fuse):
                return M.r4_fused_sampling(z, ks, ps, mus, nmp, K, BV,
                                           rand=(rnd if fuse else None), FUSE_GUMBEL=fuse)

            outs[arm] = call()
            torch.npu.synchronize()
            n_launch = (2 if not plan[0] else (3 if task == "mask" else 4))
            pd = tempfile.mkdtemp(prefix=f"prof_c{cid}_{task}_{arm}_", dir="/tmp/opencode")
            us, nsteps, blk, mn, mx = profile_wrapper(call, pd, n_launch)
            shutil.rmtree(pd, ignore_errors=True)
            rec[arm].update(total_us=us, n_steps=nsteps, block_num=blk,
                            min_step_us=mn, max_step_us=mx, n_launch=len(blk))
            print(f"  case{cid:>2} {task:5s} arm {arm}: a_launch={a:>5.2f} "
                  f"use_split={str(plan[0]):>5} S={plan[1]:>2} NBS={plan[2]:>2} "
                  f"ST={plan[3]:>2} | launches={len(blk)} BlockNum={blk} | "
                  f"total={us:>9.3f} us  (steps={nsteps}, min={mn:.3f}, max={mx:.3f})", flush=True)
        # bitwise equivalence between the two arms
        names = ("m_out", "s_out", "tau_out", "valid", "tok_out" if task == "fused" else "w_out")
        eq = {}
        for nm, x, y in zip(names, outs["A_shipped"], outs["B_relaxed"]):
            eq[nm] = bool(torch.equal(bits(x), bits(y)))
        rec["bitwise_A_vs_B"] = eq
        rec["bitwise_all_equal"] = all(eq.values())
        a_us, b_us = rec["A_shipped"]["total_us"], rec["B_relaxed"]["total_us"]
        rec["speedup_x"] = round(a_us / b_us, 5) if b_us else None
        rec["delta_pct"] = round(100.0 * (b_us - a_us) / a_us, 4) if a_us else None
        print(f"        -> A(current)={a_us:.3f} us   B(split)={b_us:.3f} us   "
              f"speedup={rec['speedup_x']:.4f}x  delta={rec['delta_pct']:+.3f}%   "
              f"bitwise A==B: {rec['bitwise_A_vs_B']} "
              f"{'ALL EQUAL' if rec['bitwise_all_equal'] else '*** DIFFERS ***'}\n", flush=True)
        results.append(rec)
        del outs
    del z, ks, ps, mus, rnd

M._SPLIT_A_LAUNCH_US = A_SHIPPED          # restore the module constant
sha_after = sha()
print(f"[probe] delivered kernel sha256 AFTER  = {sha_after}")
print(f"[probe] kernel file UNMODIFIED = {sha_before == sha_after}")

json.dump(dict(_meta=dict(
    description="Empirical cost/benefit of relaxing _SPLIT_A_LAUNCH_US from the shipped 8.0 us to "
                "the measured floor 1.54 us, for the only two cases the cost guard can admit.",
    measurement="torch_npu.profiler Level1 msprof kernel_details.csv, warmup=3 active=5, "
                "median of per-step SUMS of main_kernel Duration(us). No Python/torch timing.",
    caveat="no CannBenchCacheClean kernel in this probe (the harness launches one per step), so "
           "absolute us may read slightly optimistic; both arms identical => ratio unaffected",
    a_launch_shipped_us=A_SHIPPED, a_launch_measured_us=A_MEASURED,
    kernel_sha256_before=sha_before, kernel_sha256_after=sha_after,
    kernel_file_unmodified=(sha_before == sha_after),
    module_constant_restored=(M._SPLIT_A_LAUNCH_US == A_SHIPPED)), results=results),
    open(out_path, "w"), indent=1)
print(f"[probe] wrote {out_path}")
