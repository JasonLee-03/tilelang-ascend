"""Item-A NULL PROBE: is the case-13-mask +4.24% a real effect or run-to-run scatter?

WHY THIS PROBE EXISTS
---------------------
The 3-arm A/B flagged exactly one pair outside the 3% band: case 13 mask, 14.620 -> 15.240 us
(+4.24%), while the byte-identical drift control read 14.720 us (+0.68%). Case 13 is
(B=6, V=2304, K_MAX=128, BV=2048); `_r4_split_plan` returns use_split=False for it at BOTH
a_launch=8.0 and a_launch=1.540 because it is blocked STRUCTURALLY (n_blk_full = 2304//2048 = 1 < 2,
so the cost guard is never even reached). ksplit_fine_itemA.py independently confirmed the launch
sequence and Block Num are identical in all three arms.

So there is NO causal path from the constant to this measurement. But "no causal path" is an
argument, not a measurement, and the adoption criterion is stated as a number. This probe settles it
by measurement: it interleaves the two guard values A/B/A/B/... on the flagged pair plus controls
and reports the DISTRIBUTION of the delta. For a structurally-blocked case the two arms are the same
dispatch, so any nonzero delta is pure noise -- which gives the noise floor AT THAT ABSOLUTE SCALE
directly, instead of borrowing a percentage band calibrated on much larger cases.

口径 is identical to round 3 / the A/B: torch_npu.profiler Level1, msprof kernel_details.csv,
schedule(wait=0, warmup=3, active=5, repeat=1), per-step SUM of main_kernel Duration(us), MEDIAN
over the 5 active steps. No Python/torch timing is used for any device number (hard constraint 5).

The delivered kernel file is NEVER modified: `_SPLIT_A_LAUNCH_US` is patched on the imported module
object inside this process only, and the file sha256 is checked before and after (the round-3
probe_guard_relax.py precedent).

Usage: probe_null_case13.py [rounds] [out.json]
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
from r4_thw_common import load_cases  # noqa: E402

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "null_probe_case13.json")
KERNEL = os.path.join(R4, "r4_fused_sampling.py")
WARMUP, REPEAT = 3, 5
DEV = "npu:0"
A_BASE, A_CAND = 8.0, 1.540

CASES = {c["case_id"]: c for c in load_cases("r4_fused_sampling", verbose=False)}
# the flagged pair first, then controls: case 1 mask (2nd smallest), case 13 fused (SAME shape as the
# flagged pair, different task), case 14 mask (similar size), and cases 6/11 mask (the real effect --
# these must land FAR outside the noise floor the probe measures on the null pairs).
PROBES = [(13, "mask"), (13, "fused"), (1, "mask"), (14, "mask"), (2, "mask"),
          (6, "mask"), (11, "mask"), (6, "fused"), (11, "fused")]


def sha():
    return hashlib.sha256(open(KERNEL, "rb").read()).hexdigest()


def quiet():
    for name in ("", "torch", "torch_npu", "torch_npu.profiler", "ascend", "profiler"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.ERROR)
        lg.handlers = []
        lg.addHandler(logging.NullHandler())


def profile_wrapper(fn, prof_dir, n_launch):
    """Median over the active steps of the SUM of main_kernel Durations inside each step."""
    os.makedirs(prof_dir, exist_ok=True)
    quiet()
    torch.npu.synchronize()
    fn()                                   # pre-flight: JIT + first launch OUTSIDE the window
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
            raise RuntimeError(f"launch-count mismatch: {len(rows)} rows not a multiple of "
                               f"n_launch={n_launch}")
        rows.sort(key=lambda t: t[2])
        groups = [[(d, b, s) for d, b, s, _ in rows[i:i + n_launch]]
                  for i in range(0, len(rows), n_launch)]
    sums = [sum(d for d, _, _ in g) for g in groups]
    blocks = [[b for _, b, _ in g] for g in groups]
    if len(sums) < REPEAT:
        raise RuntimeError(f"only {len(sums)} steps recovered, expected >= {REPEAT}")
    return (round(statistics.median(sums), 3), blocks[-1],
            round(min(sums), 3), round(max(sums), 3), len(sums))


def make_inputs(B, V):
    torch.manual_seed(0)
    z = torch.randn(B, V, dtype=torch.float32) * 8.0
    ks = torch.full((B,), 40, dtype=torch.int32)
    ps = torch.full((B,), 0.95, dtype=torch.float32)
    mus = torch.zeros((B,), dtype=torch.float32)
    rnd = torch.arange(B, dtype=torch.int32) * 7919 + 13
    return [t.to(DEV) for t in (z, ks, ps, mus, rnd)]


sha0 = sha()
print(f"[null] kernel sha256 BEFORE = {sha0}")
print(f"[null] arms: A = a_launch {A_BASE} (shipped), B = a_launch {A_CAND} (candidate)")
print(f"[null] interleaved rounds = {ROUNDS}  (A then B inside each round, per probe)")
print(f"[null] probes = {PROBES}\n")

# plan census: prove which probes are NULL (identical dispatch in both arms)
print(f"{'probe':<12} {'B':>4} {'V':>7} {'K':>4} {'BV':>5} | {'plan@8.0':>22} "
      f"{'plan@1.54':>22} | kind")
census = {}
for ci, task in PROBES:
    c = CASES[ci]
    B, V, K, BV = c["B"], c["V"], c["K_MAX"], c["BV"]
    M._SPLIT_A_LAUNCH_US = A_BASE
    pa = M._r4_split_plan(B, V, K, BV)
    M._SPLIT_A_LAUNCH_US = A_CAND
    pb = M._r4_split_plan(B, V, K, BV)
    null = (pa == pb)
    census[(ci, task)] = dict(B=B, V=V, K_MAX=K, BV=BV, plan_base=list(pa), plan_cand=list(pb),
                              null_probe=null,
                              n_launch_base=(2 if not pa[0] else (3 if task == "mask" else 4)),
                              n_launch_cand=(2 if not pb[0] else (3 if task == "mask" else 4)))
    print(f"case{ci:<3}{task:<5} {B:>4} {V:>7} {K:>4} {BV:>5} | {str(pa):>22} {str(pb):>22} | "
          f"{'NULL (identical dispatch -> pure noise probe)' if null else 'FLIPS (real effect)'}")

results = defaultdict(lambda: {"A": [], "B": []})
print()
for rd in range(1, ROUNDS + 1):
    for ci, task in PROBES:
        c = CASES[ci]
        B, V, K, BV = c["B"], c["V"], c["K_MAX"], c["BV"]
        nmp = bool(c.get("need_min_p", False))
        z, ks, ps, mus, rnd = make_inputs(B, V)
        fuse = task == "fused"
        for arm, a in (("A", A_BASE), ("B", A_CAND)):
            M._SPLIT_A_LAUNCH_US = a
            plan = M._r4_split_plan(B, V, K, BV)

            def call(fuse=fuse):
                return M.r4_fused_sampling(z, ks, ps, mus, nmp, K, BV,
                                           rand=(rnd if fuse else None), FUSE_GUMBEL=fuse)

            nl = 2 if not plan[0] else (3 if task == "mask" else 4)
            pd = tempfile.mkdtemp(prefix=f"null_c{ci}_{task}_{arm}_r{rd}_", dir="/tmp/opencode")
            us, blk, mn, mx, ns = profile_wrapper(call, pd, nl)
            shutil.rmtree(pd, ignore_errors=True)
            results[(ci, task)][arm].append(dict(round=rd, total_us=us, block_num=blk,
                                                 min_step_us=mn, max_step_us=mx, n_steps=ns,
                                                 plan=list(plan)))
            print(f"  r{rd} case{ci:>2} {task:5s} arm {arm} (a={a:<5}): {us:>9.3f} us  "
                  f"launches={len(blk)} blocks={blk} (steps={ns} min={mn:.3f} max={mx:.3f})",
                  flush=True)
        del z, ks, ps, mus, rnd

M._SPLIT_A_LAUNCH_US = A_BASE
sha1 = sha()

print(f"\n{'=' * 112}")
print(f"{'probe':<12} {'kind':<6} | {'A med':>9} {'A min':>8} {'A max':>8} | {'B med':>9} "
      f"{'B min':>8} {'B max':>8} | {'B/A med':>8} {'delta%':>8} | per-round delta% spread")
print(f"{'=' * 112}")
summary = {}
for ci, task in PROBES:
    r = results[(ci, task)]
    av = [x["total_us"] for x in r["A"]]
    bv = [x["total_us"] for x in r["B"]]
    per = [100.0 * (b - a) / a for a, b in zip(av, bv)]
    am, bm = statistics.median(av), statistics.median(bv)
    kind = "NULL" if census[(ci, task)]["null_probe"] else "FLIP"
    summary[f"case{ci}_{task}"] = dict(
        kind=kind, n_rounds=len(av), A_us=av, B_us=bv,
        A_median=am, A_min=min(av), A_max=max(av),
        B_median=bm, B_min=min(bv), B_max=max(bv),
        median_ratio=round(am / bm, 5), median_delta_pct=round(100 * (bm - am) / am, 3),
        per_round_delta_pct=[round(x, 3) for x in per],
        per_round_delta_min=round(min(per), 3), per_round_delta_max=round(max(per), 3),
        per_round_delta_stdev=round(statistics.stdev(per), 3) if len(per) > 1 else None,
        blocks_A=r["A"][-1]["block_num"], blocks_B=r["B"][-1]["block_num"])
    print(f"case{ci:<3}{task:<5} {kind:<6} | {am:>9.3f} {min(av):>8.3f} {max(av):>8.3f} | "
          f"{bm:>9.3f} {min(bv):>8.3f} {max(bv):>8.3f} | {am / bm:>8.4f} "
          f"{100 * (bm - am) / am:>+8.2f} | {[round(x, 2) for x in per]}")

nulls = [v for v in summary.values() if v["kind"] == "NULL"]
flips = [v for v in summary.values() if v["kind"] == "FLIP"]
if nulls:
    worst_null = max(max(abs(x) for x in v["per_round_delta_pct"]) for v in nulls)
    print(f"\n[null] NULL-probe per-round |delta| envelope = {worst_null:.2f}%  "
          f"({len(nulls)} null probes x {ROUNDS} interleaved rounds)")
    print(f"[null] => on a structurally-blocked shape, interleaving the two guard values moves the "
          f"measured device time by up to {worst_null:.2f}% purely from run-to-run scatter.")
if flips:
    best_flip = min(v["median_delta_pct"] for v in flips)
    print(f"[null] FLIP-probe best (least negative) median delta = {best_flip:+.2f}%  "
          f"-- the real effect must be far outside the null envelope above.")

print(f"\n[null] kernel sha256 AFTER = {sha1}   UNMODIFIED = {sha0 == sha1}")
print(f"[null] module constant restored to {M._SPLIT_A_LAUNCH_US}")

json.dump(dict(_meta=dict(
    description="Interleaved A/B null probe: does the case-13-mask +4.24% reproduce, or is it "
                "run-to-run scatter on a structurally-blocked (identical-dispatch) shape?",
    measurement="torch_npu.profiler Level1 msprof kernel_details.csv, warmup=3 active=5, median of "
                "per-step SUMS of main_kernel Duration(us). No Python/torch timing.",
    a_base=A_BASE, a_cand=A_CAND, rounds=ROUNDS,
    kernel_sha256_before=sha0, kernel_sha256_after=sha1,
    kernel_file_unmodified=(sha0 == sha1),
    caveat="no CannBenchCacheClean kernel in this probe (the harness launches one per step), so "
           "absolute us may read slightly optimistic; both arms identical => ratios unaffected"),
    census={f"case{k[0]}_{k[1]}": v for k, v in census.items()},
    summary=summary,
    null_envelope_pct=max((max(abs(x) for x in v["per_round_delta_pct"]) for v in nulls),
                          default=None)), open(OUT, "w"), indent=1)
print(f"[null] wrote {OUT}")
