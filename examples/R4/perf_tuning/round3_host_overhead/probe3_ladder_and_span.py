"""PROBE 3 -- the instrumentation ladder + SPAN-without-msprof + base-vs-cand per-launch cost.

Answers three of the task's required outputs:

(1) ATTRIBUTION LADDER (C1 + C2). The cann-bench batched profiling path stacks FIVE layers
    of instrumentation on top of a bare launch. Source references:
      L1 TorchFunctionMode  -- security/torch_op_guard.py:299-312, every torch call goes
                               through `_qualified_name(func)` (getattr x3 + split + dict)
      L2 TorchDispatchMode  -- security/device_residency_guard.py:__enter__, intercepts at
                               the aten dispatcher level (lower, and normally costlier, than L1)
      L3 record_function    -- eval/perf_eval.py:517 `with torch.autograd.profiler.record_function(...)`
                               wrapped around EVERY step by _BatchDispatcher.__call__
      L4 msprof NPU         -- eval/perf_eval.py:_profile activities=[NPU]  (per-case path)
      L5 msprof CPU+NPU     -- eval/perf_eval.py:541 `include_host=True` from _profile_batch,
                               i.e. EXACTLY the config that produced round-3's _batched CSVs
    This probe measures per-launch host cost at each cumulative level, so C1 (L4+L5) and
    C2 (L1+L2+L3) are separated from the bare cost (L0).

(2) SPAN WITHOUT msprof, via torch.npu.Event. Round 3's span_us/gap_us came from
    msprof kernel_details.csv Start Time. Here the same span is measured on the device
    timeline with NPU events and NO msprof, so the SPAN speedup can be re-estimated for a
    deployment that is not being profiled.

(3) base arm (2 launches) vs cand arm (4 launches) SINGLE-launch host cost -- did rev3 make
    the host side heavier per launch, or only longer? The delivered wrapper auto-dispatches,
    so both paths are driven explicitly through the module's public factories (read-only).

READ-ONLY w.r.t. examples/R4/r4_fused_sampling.py. Usage: probe3_ladder_and_span.py [--iters N]
"""
import argparse
import contextlib
import hashlib
import json
import os
import shutil
import statistics
import sys
import tempfile
import time

from probe_common import HERE, R4, Seg, dump, env_fingerprint, npu_sync, perf_counter_ns, summarize

import torch

sys.path.insert(0, os.path.join(R4, "..", "..", ".agents", "skills", "cann-bench", "src"))
import r4_fused_sampling as R4M  # noqa: E402

CANNBENCH_SRC = os.path.abspath(os.path.join(R4, "..", "..", ".agents", "skills", "cann-bench", "src"))
if CANNBENCH_SRC not in sys.path:
    sys.path.insert(0, CANNBENCH_SRC)

# (B, V, K_MAX, BV, need_min_p) -- case1 = smallest / no split; case3 = split (4 launches fused)
SHAPES = {
    1:  (2,  256,   256, 4096, False),
    3:  (8,  16384, 256, 4096, False),
    13: (6,  2304,  128, 2048, False),
    7:  (4,  152064, 256, 4096, False),
}


def make_inputs(B, V, K_MAX, dev):
    g = torch.Generator().manual_seed(0)
    logits = torch.randn((B, V), dtype=torch.float32, generator=g).npu()
    top_ks = torch.full((B,), K_MAX, dtype=torch.int32, device=dev)
    top_ps = torch.ones((B,), dtype=torch.float32, device=dev)
    min_ps = torch.zeros((B,), dtype=torch.float32, device=dev)
    rand = torch.arange(B, dtype=torch.int32, device=dev)
    npu_sync()
    return dict(logits=logits, top_ks=top_ks, top_ps=top_ps, min_ps=min_ps, rand=rand)


# --------------------------------------------------------------- guard contexts ---
def guards(level):
    """Return the cumulative context stack for ladder level `level` (0..5)."""
    ctxs = []
    if level >= 1:
        from kernel_eval.security.torch_op_guard import TorchOpGuard
        ctxs.append(TorchOpGuard(mode="block"))
    if level >= 2:
        from kernel_eval.security.device_residency_guard import DeviceResidencyGuard
        ctxs.append(DeviceResidencyGuard(mode="block"))
    return contextlib.ExitStack(), ctxs


class RF:
    """L3: per-step record_function exactly as _BatchDispatcher does."""

    def __init__(self, on):
        self.on = on

    def __call__(self, name):
        if not self.on:
            return contextlib.nullcontext()
        return torch.autograd.profiler.record_function(name)


def prof_ctx(level, root):
    if level < 4:
        return contextlib.nullcontext()
    import torch_npu
    os.makedirs(root, exist_ok=True)
    ec = torch_npu.profiler._ExperimentalConfig(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone)
    acts = [torch_npu.profiler.ProfilerActivity.NPU]
    if level >= 5:
        acts.insert(0, torch_npu.profiler.ProfilerActivity.CPU)
    return torch_npu.profiler.profile(
        activities=acts,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=10 ** 8, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(root),
        record_shapes=False, profile_memory=False, with_stack=False, experimental_config=ec)


LEVEL_NAMES = {
    0: "L0 bare (no profiler, no guards)",
    1: "L1 +TorchFunctionMode (TorchOpGuard)",
    2: "L2 +TorchDispatchMode (DeviceResidencyGuard)",
    3: "L3 +record_function per step",
    4: "L4 +msprof activities=[NPU]",
    5: "L5 +msprof activities=[CPU,NPU]  == HARNESS (_profile_batch include_host=True)",
}


# ------------------------------------------------------------------ the ladder ---
def run_ladder(iters, prof_root):
    dev = torch.device("npu")
    out = {}
    for ci, (B, V, K_MAX, BV, nmp) in SHAPES.items():
        inp = make_inputs(B, V, K_MAX, dev)
        # warm/compile outside timing
        for _ in range(4):
            R4M.r4_fused_sampling(inp["logits"], inp["top_ks"], inp["top_ps"], inp["min_ps"],
                                  nmp, K_MAX=K_MAX, BV=BV, rand=inp["rand"], FUSE_GUMBEL=True)
        npu_sync()
        use_split, S, NBS, S_TAIL = R4M._r4_split_plan(B, V, K_MAX, BV)
        n_launch = 4 if use_split else 2

        per_level = {}
        for level in range(6):
            rf = RF(level >= 3)
            stack, ctxs = guards(level)
            seg = Seg()
            try:
                with stack:
                    for c in ctxs:
                        stack.enter_context(c)
                    with prof_ctx(level, os.path.join(prof_root, f"c{ci}_L{level}")):
                        for _ in range(3):
                            with rf("CannBenchBatchStep/probe"):
                                R4M.r4_fused_sampling(
                                    inp["logits"], inp["top_ks"], inp["top_ps"], inp["min_ps"],
                                    nmp, K_MAX=K_MAX, BV=BV, rand=inp["rand"], FUSE_GUMBEL=True)
                            npu_sync()
                        for _ in range(iters):
                            npu_sync()
                            with rf(f"CannBenchBatchStep/case={ci}/step=0"):
                                t0 = perf_counter_ns()
                                R4M.r4_fused_sampling(
                                    inp["logits"], inp["top_ks"], inp["top_ps"], inp["min_ps"],
                                    nmp, K_MAX=K_MAX, BV=BV, rand=inp["rand"], FUSE_GUMBEL=True)
                                t1 = perf_counter_ns()
                            seg.add("fn_host_us", t1 - t0)
                            npu_sync()
            finally:
                pass
            r = summarize(f"case{ci} (B={B},V={V}, {n_launch} launches) {LEVEL_NAMES[level]}", seg)
            per_level[level] = dict(
                stats=r["segments"]["fn_host_us"],
                fn_host_us=r["segments"]["fn_host_us"]["med_us"],
                per_launch_us=round(r["segments"]["fn_host_us"]["med_us"] / n_launch, 3))
            print(f"    => fn_host={per_level[level]['fn_host_us']:.2f}us  "
                  f"per_launch={per_level[level]['per_launch_us']:.2f}us")
        out[ci] = dict(B=B, V=V, K_MAX=K_MAX, BV=BV, use_split=use_split, S=S,
                       n_launch=n_launch, levels=per_level)
    return out


# ------------------------------------------------ SPAN with NPU events, no msprof ---
def measure_span_events(ci, inp, nmp, K_MAX, BV, fuse, iters, dev, use_msprof_level=0,
                        prof_root=None, label=""):
    """Device-timeline span via torch.npu.Event, WITHOUT msprof (level 0).

    span_ev = elapsed(e_before_first_launch -> e_after_last_launch) on the DEVICE timeline.
    Also returns the event-derived sum of per-kernel spans so the event overhead is visible.
    """
    E = torch.npu.Event
    spans, kspans = [], []
    for _ in range(iters):
        npu_sync()
        e0 = E(enable_timing=True)
        e1 = E(enable_timing=True)
        e0.record()
        R4M.r4_fused_sampling(inp["logits"], inp["top_ks"], inp["top_ps"], inp["min_ps"],
                              nmp, K_MAX=K_MAX, BV=BV, rand=inp["rand"], FUSE_GUMBEL=fuse)
        e1.record()
        npu_sync()
        spans.append(e0.elapsed_time(e1) * 1000.0)  # ms -> us
    spans.sort()
    return dict(n=iters, min_us=round(spans[0], 3), med_us=round(statistics.median(spans), 3),
                p90_us=round(spans[int(0.9 * (len(spans) - 1))], 3), max_us=round(spans[-1], 3),
                label=label)


# ------------------------------------- base(2-launch) vs cand(4-launch) per launch ---
def base_vs_cand_per_launch(iters, dev):
    """Item 3: is the SINGLE-launch host cost the same in both arms?

    Drives each path explicitly through the module's public factories so that the same
    shape can be measured in the base (delivered, 2-launch) form and the cand (rev3 split,
    4-launch) form regardless of what _r4_split_plan would auto-select.
    """
    res = {}
    for ci, (B, V, K_MAX, BV, nmp) in SHAPES.items():
        inp = make_inputs(B, V, K_MAX, dev)
        logits, ks, ps, ms, rand = (inp["logits"], inp["top_ks"], inp["top_ps"],
                                    inp["min_ps"], inp["rand"])
        use_split, S, NBS, S_TAIL = R4M._r4_split_plan(B, V, K_MAX, BV)
        if not use_split:
            # force the split plan for comparison purposes only (probe-local, not delivered)
            b_num = -(-B // R4M.VEC_NUM)
            n_blk_full = V // BV
            TAIL = V - n_blk_full * BV
            n_blk_total = n_blk_full + (1 if TAIL > 0 else 0)
            cube = R4M.detect_cube_core_num()
            S2 = max(2, min(max(1, cube // b_num), n_blk_total))
            NBS2 = -(-n_blk_total // S2)
            S2 = -(-n_blk_total // NBS2)
            NBS2 = -(-n_blk_total // S2)
            S_TAIL2 = n_blk_full // NBS2 if TAIL > 0 else -1
        else:
            S2, NBS2, S_TAIL2 = S, NBS, S_TAIL
        if S2 < 2 or (V // BV) < 2:
            print(f"  case{ci}: cannot force a >=2 split (S={S2}); skipping cand arm")
            res[ci] = dict(skipped=True, S_forced=S2)
            continue

        m_out = torch.empty(B, dtype=torch.float32, device=dev)
        s_out = torch.empty(B, dtype=torch.float32, device=dev)
        tau = torch.empty(B, dtype=torch.float32, device=dev)
        pv = torch.empty((B * S2, K_MAX), dtype=torch.float32, device=dev)
        pvv = torch.empty((B * S2,), dtype=torch.float32, device=dev)
        pvi = torch.empty((B * S2,), dtype=torch.float32, device=dev)

        base = [
            ("jit_K1", lambda: R4M.r4_topk_stats(B, V, K_MAX, BV, nmp)),
            ("launch_K1", None), ("jit_K2g", lambda: R4M.r4_gumbel_sample(B, V, BV)),
            ("launch_K2g", None),
        ]
        # Pre-resolve the JIT objects ONCE (cache is warm in the harness too); we time the
        # jit-lookup separately so it cannot hide inside the launch number.
        k1 = R4M.r4_topk_stats(B, V, K_MAX, BV, nmp)
        k2g = R4M.r4_gumbel_sample(B, V, BV)
        k1a = R4M.r4_topk_stats_splitA(B, V, K_MAX, BV, S2, NBS2, S_TAIL2)
        k1b = R4M.r4_topk_stats_splitB(B, V, K_MAX, BV, S2, nmp)
        k2ga = R4M.r4_gumbel_sample_splitA(B, V, BV, S2, NBS2)
        k2gb = R4M.r4_gumbel_sample_splitB(B, S2)

        def arm_base(seg):
            t = perf_counter_ns(); _ = R4M.r4_topk_stats(B, V, K_MAX, BV, nmp); seg.add("jit_k1", perf_counter_ns() - t)
            t = perf_counter_ns(); r = k1(logits, ks, ps, ms); seg.add("L1_K1", perf_counter_ns() - t)
            t = perf_counter_ns(); _ = R4M.r4_gumbel_sample(B, V, BV); seg.add("jit_k2g", perf_counter_ns() - t)
            t = perf_counter_ns(); k2g(logits, r[2], rand); seg.add("L2_K2g", perf_counter_ns() - t)

        def arm_cand(seg):
            t = perf_counter_ns(); _ = R4M.r4_topk_stats_splitA(B, V, K_MAX, BV, S2, NBS2, S_TAIL2); seg.add("jit_k1a", perf_counter_ns() - t)
            t = perf_counter_ns(); k1a(logits, pv); seg.add("L1_K1A", perf_counter_ns() - t)
            t = perf_counter_ns(); _ = R4M.r4_topk_stats_splitB(B, V, K_MAX, BV, S2, nmp); seg.add("jit_k1b", perf_counter_ns() - t)
            t = perf_counter_ns(); r = k1b(logits, ks, ps, ms, pv); seg.add("L2_K1B", perf_counter_ns() - t)
            t = perf_counter_ns(); _ = R4M.r4_gumbel_sample_splitA(B, V, BV, S2, NBS2); seg.add("jit_k2ga", perf_counter_ns() - t)
            t = perf_counter_ns(); k2ga(logits, r[2], rand, pvv, pvi); seg.add("L3_K2gA", perf_counter_ns() - t)
            t = perf_counter_ns(); _ = R4M.r4_gumbel_sample_splitB(B, S2); seg.add("jit_k2gb", perf_counter_ns() - t)
            t = perf_counter_ns(); k2gb(pvv, pvi); seg.add("L4_K2gB", perf_counter_ns() - t)

        out_ci = dict(B=B, V=V, S_forced=S2, NBS=NBS2, S_TAIL=S_TAIL2,
                      auto_use_split=use_split, arms={})
        for aname, fn, nl in (("base_2launch", arm_base, 2), ("cand_4launch", arm_cand, 4)):
            for _ in range(4):
                fn(Seg())
            npu_sync()
            seg = Seg()
            for _ in range(iters):
                npu_sync(); fn(seg); npu_sync()
            keys = [k for k in sorted(seg.stats()) if k.startswith("L")]
            jitk = [k for k in sorted(seg.stats()) if k.startswith("jit")]
            r = summarize(f"case{ci} {aname}", seg)
            launch_sum = sum(seg.med(k) for k in keys)
            jit_sum = sum(seg.med(k) for k in jitk)
            out_ci["arms"][aname] = dict(
                per_kernel={k: seg.med(k) for k in keys},
                per_jit_lookup={k: seg.med(k) for k in jitk},
                launch_sum_us=round(launch_sum, 3), jit_lookup_sum_us=round(jit_sum, 3),
                n_launch=nl,
                per_launch_us=round(launch_sum / nl, 3),
                per_launch_incl_jit_us=round((launch_sum + jit_sum) / nl, 3),
                full=r)
            print(f"    {aname}: launch_sum={launch_sum:.2f}us / {nl} = "
                  f"{launch_sum/nl:.2f}us per launch (jit lookups {jit_sum:.2f}us extra)")
        b = out_ci["arms"]["base_2launch"]["per_launch_us"]
        c = out_ci["arms"]["cand_4launch"]["per_launch_us"]
        out_ci["per_launch_delta_cand_minus_base_us"] = round(c - b, 3)
        out_ci["per_launch_ratio_cand_over_base"] = round(c / b, 4) if b else None
        print(f"    ==> per-launch: base={b:.2f}us cand={c:.2f}us  delta={c-b:+.2f}us  ratio={c/b:.3f}")
        res[ci] = out_ci
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--cases", default="1,3,13,7")
    ap.add_argument("--out", default="probe3_ladder_and_span.json")
    a = ap.parse_args()
    global SHAPES
    _sel = [int(x) for x in a.cases.split(",") if x.strip()]
    SHAPES = {k: v for k, v in SHAPES.items() if k in _sel}
    dev = torch.device("npu")
    fp = env_fingerprint()
    fp["delivered_kernel_sha256"] = hashlib.sha256(
        open(os.path.join(R4, "r4_fused_sampling.py"), "rb").read()).hexdigest()
    print("env:", json.dumps(fp, indent=1))
    res = {"_meta": dict(
        description="instrumentation ladder L0..L5, NPU-event span without msprof, "
                    "base(2-launch) vs cand(4-launch) per-launch host cost",
        measurement="perf_counter_ns around HOST calls (async launches) + torch.npu.Event "
                    "for device-timeline span. NOT kernel durations.",
        iters=a.iters, env=fp, level_names=LEVEL_NAMES)}

    prof_root = tempfile.mkdtemp(prefix="probe3_prof_", dir=HERE)
    try:
        print("\n" + "#" * 100 + "\n# PART A: instrumentation ladder\n" + "#" * 100)
        res["ladder"] = run_ladder(a.iters, prof_root)

        print("\n" + "#" * 100 + "\n# PART B: SPAN via NPU events, NO msprof\n" + "#" * 100)
        res["span_events"] = {}
        for ci, (B, V, K_MAX, BV, nmp) in SHAPES.items():
            inp = make_inputs(B, V, K_MAX, dev)
            for _ in range(4):
                R4M.r4_fused_sampling(inp["logits"], inp["top_ks"], inp["top_ps"], inp["min_ps"],
                                      nmp, K_MAX=K_MAX, BV=BV, rand=inp["rand"], FUSE_GUMBEL=True)
            npu_sync()
            s = measure_span_events(ci, inp, nmp, K_MAX, BV, True, a.iters, dev)
            res["span_events"][ci] = s
            print(f"  case{ci:<3} span_npu_event(no msprof) = med {s['med_us']:.2f} us "
                  f"[min {s['min_us']:.2f} max {s['max_us']:.2f}] n={s['n']}")

        print("\n" + "#" * 100 + "\n# PART C: base(2-launch) vs cand(4-launch) per-launch\n" + "#" * 100)
        res["base_vs_cand"] = base_vs_cand_per_launch(a.iters, dev)
    finally:
        shutil.rmtree(prof_root, ignore_errors=True)

    # --------------------------------------------------------------- ladder table ---
    print("\n" + "=" * 118)
    print("LADDER SUMMARY -- fn() host time (us) and per-launch host cost (us), cumulative levels")
    print("=" * 118)
    print(f"{'level':<62} " + " ".join(f"{'case'+str(c):>12}" for c in res["ladder"]))
    tab = {}
    for level in range(6):
        row = []
        for ci in res["ladder"]:
            v = res["ladder"][ci]["levels"][str(level)] if str(level) in res["ladder"][ci]["levels"] \
                else res["ladder"][ci]["levels"][level]
            row.append(v["per_launch_us"])
        tab[level] = row
        print(f"{LEVEL_NAMES[level]:<62} " + " ".join(f"{x:>12.2f}" for x in row))
    print("\n  (values are us PER LAUNCH; case3 has 4 launches, the others 2)")
    res["ladder_summary_per_launch_us"] = {str(k): v for k, v in tab.items()}

    l0 = {ci: res["ladder"][ci]["levels"][0]["per_launch_us"] for ci in res["ladder"]}
    l5 = {ci: res["ladder"][ci]["levels"][5]["per_launch_us"] for ci in res["ladder"]}
    print("\n  msprof-only inflation (L5/L0, i.e. with guards already on at both ends):")
    l3 = {ci: res["ladder"][ci]["levels"][3]["per_launch_us"] for ci in res["ladder"]}
    for ci in res["ladder"]:
        print(f"    case{ci:<3} L0={l0[ci]:7.2f}  L3={l3[ci]:7.2f}  L5={l5[ci]:7.2f}   "
              f"L3/L0={l3[ci]/l0[ci]:.3f}  L5/L0={l5[ci]/l0[ci]:.3f}  L5/L3={l5[ci]/l3[ci]:.3f}")
    res["inflation"] = {str(ci): dict(L0=l0[ci], L3=l3[ci], L5=l5[ci],
                                      L3_over_L0=round(l3[ci] / l0[ci], 4),
                                      L5_over_L0=round(l5[ci] / l0[ci], 4),
                                      L5_over_L3=round(l5[ci] / l3[ci], 4)) for ci in res["ladder"]}
    dump(a.out, res)


if __name__ == "__main__":
    main()
