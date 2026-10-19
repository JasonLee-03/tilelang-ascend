"""PROBE 7 -- FAITHFUL reproduction of the cann-bench batched profiling step loop.

WHY: probe3's L5 measured ~537 us/launch for case1, but the round-3 harness CSVs imply
~381-404 us/launch. probe3 never called `prof.step()` and used `active=10**8`, so the
profiler's step machinery and buffer flushing did not match the harness. This probe
reproduces the harness loop faithfully and then validates itself against the CSV it produces:

  harness loop (perf_eval.py:_profile + _profile_batch + op_runner.py:_DeferredProfileCall):
    * ONE profiler session for the whole batch, schedule(wait=0, warmup=0, active=STEPS, repeat=1)
    * `include_host=True` -> activities=[CPU, NPU]                       (perf_eval.py:541)
    * guards entered ONCE around the whole session                       (op_runner.py:186-188)
    * per step: [_clear_cache() -> op + torch.npu.synchronize()] then
                `with record_function("CannBenchBatchStep/case=i/step=j"): fn()`
                then `_synchronize_profile_step()` then `prof.step()`    (perf_eval.py:277-289)
    * ProfilerLevel.Level1, ExportType.Text, AiCoreNone, record_shapes=False,
      profile_memory=False, with_stack=False

VALIDATION: after the run, this probe parses its OWN kernel_details.csv with the same
identity used on the round-3 CSVs (H_i = gap_i + d_{i-1}, host-bound launches only) and
reports whether the reproduced H matches the round-3 harness H. If it does, the C1/C2 split
from the ladder below is anchored to the real harness rather than to an approximation.

Then it re-runs the ladder with `prof.step()` present, so C1 (msprof) and C2 (guards) are
separated under faithful conditions.

READ-ONLY w.r.t. examples/R4/r4_fused_sampling.py. Usage: probe7_faithful_harness.py
"""
import argparse
import contextlib
import csv as _csv
import hashlib
import json
import os
import shutil
import statistics
import sys
import tempfile
from collections import defaultdict

from probe_common import HERE, R4, Seg, dump, env_fingerprint, npu_sync, perf_counter_ns, summarize

import torch
import r4_fused_sampling as R4M  # noqa: E402

CANNBENCH_SRC = os.path.abspath(os.path.join(R4, "..", "..", ".agents", "skills", "cann-bench", "src"))
if CANNBENCH_SRC not in sys.path:
    sys.path.insert(0, CANNBENCH_SRC)

SHAPES = {1: (2, 256, 256, 4096, False), 3: (8, 16384, 256, 4096, False),
          13: (6, 2304, 128, 2048, False), 7: (4, 152064, 256, 4096, False)}
WARMUP_SKIP = 3          # harness: warmup=3, repeat=5 -> 8 steps/case, first 3 discarded
H_GAP_VALID_US = 50.0


def make_inputs(B, V, K_MAX, dev):
    g = torch.Generator().manual_seed(0)
    return dict(logits=torch.randn((B, V), dtype=torch.float32, generator=g).npu(),
                top_ks=torch.full((B,), K_MAX, dtype=torch.int32, device=dev),
                top_ps=torch.ones((B,), dtype=torch.float32, device=dev),
                min_ps=torch.zeros((B,), dtype=torch.float32, device=dev),
                rand=torch.arange(B, dtype=torch.int32, device=dev))


def guard_stack(level):
    st = contextlib.ExitStack()
    if level >= 1:
        from kernel_eval.security.torch_op_guard import TorchOpGuard
        st.enter_context(TorchOpGuard(mode="block"))
    if level >= 2:
        from kernel_eval.security.device_residency_guard import DeviceResidencyGuard
        st.enter_context(DeviceResidencyGuard(mode="block"))
    return st


def prof_session(level, root, total_steps):
    """Faithful: schedule(wait=0, warmup=0, active=total_steps, repeat=1)."""
    if level < 4:
        return contextlib.nullcontext(), None
    import torch_npu
    os.makedirs(root, exist_ok=True)
    ec = torch_npu.profiler._ExperimentalConfig(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone)
    acts = [torch_npu.profiler.ProfilerActivity.NPU]
    if level >= 5:
        acts.insert(0, torch_npu.profiler.ProfilerActivity.CPU)
    p = torch_npu.profiler.profile(
        activities=acts,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=total_steps, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(root),
        record_shapes=False, profile_memory=False, with_stack=False, experimental_config=ec)
    return p, root


def find_csv(root):
    if not root or not os.path.isdir(root):
        return None
    for dp, _dn, fn in os.walk(root):
        if dp.endswith("ASCEND_PROFILER_OUTPUT") and "kernel_details.csv" in fn:
            return os.path.join(dp, "kernel_details.csv")
    return None


def implied_H_from_csv(path, steps_per_case):
    """Same identity as reconcile_with_round3_csv.py: H_i = gap_i + d_{i-1}, host-bound only."""
    by = defaultdict(list)
    for r in _csv.DictReader(open(path)):
        if r["Name"].strip() != "main_kernel":
            continue
        by[int(r["Step Id"].strip())].append(
            (float(r["Start Time(us)"].strip()), float(r["Duration(us)"])))
    Hs, spans, scored, gaps = [], [], [], []
    for step, rows in by.items():
        if step % steps_per_case < WARMUP_SKIP:
            continue
        seq = sorted(rows)
        scored.append(sum(d for _, d in seq))
        spans.append(max(s + d for s, d in seq) - seq[0][0])
        for i in range(1, len(seq)):
            gap = seq[i][0] - (seq[i - 1][0] + seq[i - 1][1])
            if gap >= H_GAP_VALID_US:
                Hs.append(gap + seq[i - 1][1])
                gaps.append(gap)
    if not Hs:
        return None
    return dict(n_H=len(Hs), H_median_us=round(statistics.median(Hs), 3),
                H_min_us=round(min(Hs), 3), H_max_us=round(max(Hs), 3),
                gap_median_us=round(statistics.median(gaps), 3),
                span_median_us=round(statistics.median(spans), 3),
                scored_median_us=round(statistics.median(scored), 3), n_steps=len(spans))


def run_case(ci, level, iters, dev, prof_root, emulate_cache_clean):
    B, V, K_MAX, BV, nmp = SHAPES[ci]
    inp = make_inputs(B, V, K_MAX, dev)
    kw = dict(K_MAX=K_MAX, BV=BV, rand=inp["rand"], FUSE_GUMBEL=True)
    args = (inp["logits"], inp["top_ks"], inp["top_ps"], inp["min_ps"])
    for _ in range(6):
        R4M.r4_fused_sampling(*args, nmp, **kw)
    npu_sync()
    use_split, S, NBS, S_TAIL = R4M._r4_split_plan(B, V, K_MAX, BV)
    n_launch = 4 if use_split else 2

    # a stand-in for cann_bench_cache_clean (L2 flush) -- a vector op + sync, as the harness does
    flush_buf = torch.randn(1024 * 1024, device=dev)

    steps = WARMUP_SKIP + iters
    root = os.path.join(prof_root, f"c{ci}_L{level}") if level >= 4 else None
    sess, _ = prof_session(level, root, steps)
    seg = Seg()
    with guard_stack(level) as _st, sess as prof:
        for j in range(steps):
            if emulate_cache_clean:
                torch.max(flush_buf)          # harness uses cann_bench_cache_clean; torch.max is
                torch.npu.synchronize()       # the documented pre-V3 equivalent (perf_eval.py:257)
            rf = (torch.autograd.profiler.record_function(f"CannBenchBatchStep/case={ci}/step={j}")
                  if level >= 3 else contextlib.nullcontext())
            with rf:
                if j >= WARMUP_SKIP:
                    t0 = perf_counter_ns()
                    R4M.r4_fused_sampling(*args, nmp, **kw)
                    t1 = perf_counter_ns()
                    seg.add("fn_host_us", t1 - t0)
                else:
                    R4M.r4_fused_sampling(*args, nmp, **kw)
            torch.npu.synchronize()           # _synchronize_profile_step
            if prof is not None:
                prof.step()                   # <-- the call probe3 omitted
    r = summarize(f"case{ci} L{level} faithful (prof.step() called, cache-clean emulated)", seg)
    out = dict(case=ci, level=level, n_launch=n_launch, use_split=use_split,
               fn_host=r["segments"]["fn_host_us"],
               per_launch_us=round(r["segments"]["fn_host_us"]["med_us"] / n_launch, 3),
               csv=None)
    if root:
        # the profiler writes on exit; give it a moment then locate the CSV
        p = find_csv(root)
        if p:
            out["csv"] = dict(path=p, **implied_H_from_csv(p, steps))
            print(f"    CSV validation: implied H median = {out['csv']['H_median_us']:.2f} us "
                  f"(n={out['csv']['n_H']})  span={out['csv']['span_median_us']:.2f}  "
                  f"scored={out['csv']['scored_median_us']:.2f}")
        else:
            print("    CSV validation: kernel_details.csv NOT FOUND")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5, help="measured steps per case (harness uses 5)")
    ap.add_argument("--cases", default="1,3,13,7")
    ap.add_argument("--out", default="probe7_faithful_harness.json")
    a = ap.parse_args()
    dev = torch.device("npu")
    sel = [int(x) for x in a.cases.split(",") if x.strip()]
    fp = env_fingerprint()
    fp["delivered_kernel_sha256"] = hashlib.sha256(
        open(os.path.join(R4, "r4_fused_sampling.py"), "rb").read()).hexdigest()
    print("env:", json.dumps(fp, indent=1))
    res = {"_meta": dict(
        description="faithful reproduction of the cann-bench batched step loop (incl. "
                    "prof.step()), self-validated against the kernel_details.csv it produces",
        measurement="perf_counter_ns for host time; CSV-implied H from device timestamps",
        iters=a.iters, env=fp)}

    prof_root = tempfile.mkdtemp(prefix="probe7_prof_", dir=HERE)
    try:
        ladder = {}
        for ci in sel:
            ladder[ci] = {}
            for level in (0, 1, 2, 3, 5):
                print(f"\n--- case{ci} level L{level} ---")
                r = run_case(ci, level, a.iters, dev, prof_root, emulate_cache_clean=(level >= 3))
                ladder[ci][level] = r
                print(f"    fn_host={r['fn_host']['med_us']:.2f}us  "
                      f"per_launch={r['per_launch_us']:.2f}us")
        res["ladder_faithful"] = ladder

        print("\n" + "=" * 112)
        print("FAITHFUL LADDER -- per-launch host cost (us), cache-clean emulated from L3 up")
        print("=" * 112)
        lv = (0, 1, 2, 3, 5)
        print(f"{'case':>5}{'#ln':>4}" + "".join(f"{('L'+str(x)):>11}" for x in lv)
              + f"{'L3/L0':>8}{'L5/L0':>8}{'L5/L3':>8}")
        tab = {}
        for ci in sel:
            row = [ladder[ci][x]["per_launch_us"] for x in lv]
            tab[ci] = dict(zip([f"L{x}" for x in lv], row))
            tab[ci]["n_launch"] = ladder[ci][0]["n_launch"]
            print(f"{ci:>5}{ladder[ci][0]['n_launch']:>4}" + "".join(f"{v:>11.2f}" for v in row)
                  + f"{row[3]/row[0]:>8.3f}{row[4]/row[0]:>8.3f}{row[4]/row[3]:>8.3f}")
        res["ladder_table"] = tab

        print("\n" + "=" * 112)
        print("SELF-VALIDATION: does the reproduced CSV match the round-3 harness CSV?")
        print("=" * 112)
        rc = json.load(open(os.path.join(HERE, "reconcile_with_round3_csv.json")))
        r3 = rc.get("per_case_fused", {})
        s3p = os.path.join(R4, "perf_tuning", "round3_splitv_perf", "span_vs_scored.json")
        s3 = json.load(open(s3p))["results"]["fused"]["per_case"] if os.path.exists(s3p) else {}
        val = {}
        print(f"{'case':>5} | {'H: r3':>8}{'p7':>8}{'ratio':>7} | {'gap/ln: r3':>11}{'p7':>8}"
              f"{'ratio':>7} | {'span: r3':>9}{'p7':>9}{'ratio':>7} | {'scored: r3':>11}{'p7':>8}{'ratio':>7}")
        for ci in sel:
            c = ladder[ci].get(5, {}).get("csv")
            k = str(ci)
            h3 = (r3.get(k) or {}).get("H_cand_med_us")
            g3 = (r3.get(k) or {}).get("gap_cand_med_us")
            s3c = ((s3.get(k) or {}).get("cand") or {})
            sp3, sc3 = s3c.get("span_us"), s3c.get("scored_us")
            if not (c and h3):
                print(f"{ci:>5} | (no CSV or no round-3 H)")
                continue
            ratios = [c["H_median_us"] / h3]
            line = f"{ci:>5} | {h3:>8.2f}{c['H_median_us']:>8.2f}{ratios[0]:>7.3f} |"
            if g3:
                rg = c["gap_median_us"] / g3
                ratios.append(rg)
                line += f" {g3:>10.2f}{c['gap_median_us']:>8.2f}{rg:>7.3f} |"
            else:
                line += f" {'-':>10}{'-':>8}{'-':>7} |"
            if sp3:
                rs = c["span_median_us"] / sp3
                ratios.append(rs)
                line += f" {sp3:>8.2f}{c['span_median_us']:>9.2f}{rs:>7.3f} |"
            else:
                line += f" {'-':>8}{'-':>9}{'-':>7} |"
            if sc3:
                rc_ = c["scored_median_us"] / sc3
                ratios.append(rc_)
                line += f" {sc3:>10.2f}{c['scored_median_us']:>8.2f}{rc_:>7.3f}"
            else:
                line += f" {'-':>10}{'-':>8}{'-':>7}"
            print(line)
            val[ci] = dict(round3=dict(H_cand_us=h3, gap_cand_us=g3, span_us=sp3, scored_us=sc3),
                           probe7_L5=dict(H_us=c["H_median_us"], gap_us=c["gap_median_us"],
                                          span_us=c["span_median_us"],
                                          scored_us=c["scored_median_us"], n_H=c["n_H"]),
                           ratios=dict(H=round(c["H_median_us"] / h3, 4),
                                       gap=round(c["gap_median_us"] / g3, 4) if g3 else None,
                                       span=round(c["span_median_us"] / sp3, 4) if sp3 else None,
                                       scored=round(c["scored_median_us"] / sc3, 4) if sc3 else None))
        res["self_validation_vs_round3"] = val
        if val:
            rr = [v["ratios"]["H"] for v in val.values()]
            rs = [v["ratios"]["scored"] for v in val.values() if v["ratios"]["scored"]]
            print(f"\n  probe7/round3 ratio  H: median={statistics.median(rr):.3f} "
                  f"range=[{min(rr):.3f}, {max(rr):.3f}] n={len(rr)}")
            if rs:
                print(f"  probe7/round3 ratio  scored: median={statistics.median(rs):.4f} "
                      f"range=[{min(rs):.4f}, {max(rs):.4f}] n={len(rs)}")
            print("  A ratio near 1.00 means the reproduction is faithful and the L0->L5 ladder")
            print("  can be read as the real C1/C2 split. A ratio >1 means probe7 still")
            print("  over-instruments; the ladder then gives an UPPER BOUND on C1+C2.")
            print("  NOTE the `scored` column is the strongest check: it is a pure DEVICE-time")
            print("  quantity taken from msprof in both runs, so it is independent of every")
            print("  host-side measurement in this investigation.")
            res["fidelity_ratio_median_H"] = round(statistics.median(rr), 4)
            if rs:
                res["fidelity_ratio_median_scored"] = round(statistics.median(rs), 5)
    finally:
        shutil.rmtree(prof_root, ignore_errors=True)
    dump(a.out, res)


if __name__ == "__main__":
    main()
