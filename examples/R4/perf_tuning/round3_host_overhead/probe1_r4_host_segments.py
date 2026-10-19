"""PROBE 1 -- decompose the R4 host wrapper into timed segments, WITH and WITHOUT msprof.

Answers:
  C1 (msprof instrumentation)  -- same measurement loop under 3 profiler configs:
                                  "none" | "npu_only" | "cpu_npu"(= what the round-3
                                  _batched CSVs actually used, include_host=True)
  C3 (our host wrapper)        -- per-segment: pre-checks / _r4_split_plan / torch.empty
                                  workspace / jit-cache lookup / launch submit
  C5 (ACL launch itself)       -- the `launch_*` segments after C3 is subtracted

READ-ONLY: imports examples/R4/r4_fused_sampling.py and calls its public factories.
It does NOT modify the delivered kernel. The wrapper logic below is a REPLICA written
only so that segment boundaries can be timed; probe1 cross-checks the replica against
the real `r4_fused_sampling()` end-to-end host time so the replica cannot silently drift.

Usage: probe1_r4_host_segments.py [--iters N] [--cases 1,3,13] [--out name.json]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

from probe_common import (HERE, R4, Seg, dump, env_fingerprint, npu_sync,
                          perf_counter_ns, summarize)

import torch  # noqa: E402

import r4_fused_sampling as R4M  # noqa: E402  (the delivered kernel, read-only)

CASES = {
    #  case_id: (B, V, K_MAX, BV, need_min_p, why)
    1:  (2,   256,    256, 4096, False, "smallest; base arm K1=24.48us -> round3 gap=458.75us"),
    3:  (8,   16384,  256, 4096, False, "split path, 4 launches, round3 gap_c=1329.77us"),
    13: (6,   2304,   128, 2048, False, "94% of wall clock was host (23.84us device / 416.75us wall)"),
    7:  (4,   152064, 256, 4096, False, "long kernels; base gap=3.00us -> the H-d1 identity check"),
}


# --------------------------------------------------------------------- inputs ---
def make_inputs(B, V, K_MAX, need_min_p, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn((B, V), dtype=torch.float32, generator=g).npu()
    top_ks = torch.full((B,), K_MAX, dtype=torch.int32).npu()
    top_ps = torch.ones((B,), dtype=torch.float32).npu()
    min_ps = torch.zeros((B,), dtype=torch.float32).npu()
    rand = torch.arange(B, dtype=torch.int32).npu()
    npu_sync()
    return logits, top_ks, top_ps, min_ps, rand


# ------------------------------------------------- segmented REPLICA of wrapper ---
def wrapper_segments(logits, top_ks, top_ps, min_ps, rand, need_min_p, K_MAX, BV,
                     FUSE_GUMBEL, seg: Seg):
    """Faithful replica of r4_fused_sampling() with a perf_counter around each segment."""
    pc = perf_counter_ns
    t = pc()

    # ---- segment: pre-checks (C6/C7/C3/dtype) -----------------------------------
    B, V = logits.shape
    _ = (B == 0)
    _ = (V < K_MAX)
    tail = V % BV
    _ = (tail != 0 and tail < K_MAX)
    _ = (logits.dtype != torch.float32)
    seg.add("01_prechecks", pc() - t)

    # ---- segment: _r4_split_plan (pure python int math) ---------------------------
    t = pc()
    use_split, S, NBS, S_TAIL = R4M._r4_split_plan(B, V, K_MAX, BV)
    seg.add("02_split_plan", pc() - t)

    # ---- segment: GM workspace torch.empty (rev3 ADDITION, split path only) -------
    t = pc()
    pv = None
    if use_split:
        pv = torch.empty((B * S, K_MAX), dtype=torch.float32, device=logits.device)
    seg.add("03_alloc_workspace", pc() - t)

    # ---- segment: jit factory lookup for K1 / K1A --------------------------------
    t = pc()
    if use_split:
        k1a = R4M.r4_topk_stats_splitA(B, V, K_MAX, BV, S, NBS, S_TAIL)
    else:
        k1 = R4M.r4_topk_stats(B, V, K_MAX, BV, bool(need_min_p))
    seg.add("04_jit_lookup_k1", pc() - t)

    # ---- segment: LAUNCH K1 (or K1A) ---------------------------------------------
    t = pc()
    if use_split:
        k1a(logits, pv)
    else:
        m_out, s_out, tau_out, valid = k1(logits, top_ks, top_ps, min_ps)
    seg.add("05_launch_k1", pc() - t)

    # ---- segment: jit lookup + LAUNCH K1B (rev3 addition) -------------------------
    if use_split:
        t = pc()
        k1b = R4M.r4_topk_stats_splitB(B, V, K_MAX, BV, S, bool(need_min_p))
        seg.add("06_jit_lookup_k1b", pc() - t)
        t = pc()
        m_out, s_out, tau_out, valid = k1b(logits, top_ks, top_ps, min_ps, pv)
        seg.add("07_launch_k1b", pc() - t)

    # ---- segment: K2 (mask -> w_out[B,V]  |  gumbel split -> tok_out) -------------
    if not FUSE_GUMBEL:
        t = pc()
        k2 = R4M.r4_mask_sample(B, V, BV)
        seg.add("08_jit_lookup_k2", pc() - t)
        t = pc()
        w_out = k2(logits, m_out, s_out, tau_out)
        seg.add("09_launch_k2", pc() - t)
        return m_out, s_out, tau_out, valid, w_out

    t = pc()
    if use_split:
        k2ga = R4M.r4_gumbel_sample_splitA(B, V, BV, S, NBS)
    else:
        k2g = R4M.r4_gumbel_sample(B, V, BV)
    seg.add("08_jit_lookup_k2", pc() - t)

    if use_split:
        t = pc()
        pv_val = torch.empty((B * S,), dtype=torch.float32, device=logits.device)
        pv_idx = torch.empty((B * S,), dtype=torch.float32, device=logits.device)
        seg.add("09_alloc_k2ws", pc() - t)
        t = pc()
        k2ga(logits, tau_out, rand, pv_val, pv_idx)
        seg.add("10_launch_k2gA", pc() - t)
        t = pc()
        k2gb = R4M.r4_gumbel_sample_splitB(B, S)
        seg.add("11_jit_lookup_k2gB", pc() - t)
        t = pc()
        tok_out = k2gb(pv_val, pv_idx)
        seg.add("12_launch_k2gB", pc() - t)
        return m_out, s_out, tau_out, valid, tok_out

    t = pc()
    tok_out = k2g(logits, tau_out, rand)
    seg.add("10_launch_k2g", pc() - t)
    return m_out, s_out, tau_out, valid, tok_out


# ------------------------------------------------------------ profiler configs ---
def profiler_ctx(mode: str, prof_root: str):
    """mode: 'none' | 'npu_only' | 'cpu_npu'.

    'cpu_npu' reproduces EXACTLY what produced the round-3 _batched CSVs:
    perf_eval.py:_profile_batch -> self._profile(..., include_host=True) ->
    activities.insert(0, ProfilerActivity.CPU), plus a per-step record_function range.
    """
    import contextlib
    if mode == "none":
        return contextlib.nullcontext(), False
    import torch_npu
    os.makedirs(prof_root, exist_ok=True)
    ec = torch_npu.profiler._ExperimentalConfig(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    )
    acts = [torch_npu.profiler.ProfilerActivity.NPU]
    if mode == "cpu_npu":
        acts.insert(0, torch_npu.profiler.ProfilerActivity.CPU)
    p = torch_npu.profiler.profile(
        activities=acts,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=10 ** 9, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_root),
        record_shapes=False, profile_memory=False, with_stack=False,
        experimental_config=ec,
    )
    return p, (mode == "cpu_npu")


# ------------------------------------------------------------------- one config ---
def run_mode(mode, cases, iters, fuse_gumbel, prof_root):
    results = {}
    ctx, wrap_rf = profiler_ctx(mode, prof_root)
    with ctx:
        for ci in cases:
            B, V, K_MAX, BV, nmp, why = CASES[ci]
            inp = make_inputs(B, V, K_MAX, nmp)

            # ---- warm up (compile + cache), OUTSIDE timing ----------------------
            for _ in range(5):
                R4M.r4_fused_sampling(*inp[:4], nmp, K_MAX=K_MAX, BV=BV, rand=inp[4],
                                      FUSE_GUMBEL=fuse_gumbel)
                wrapper_segments(*inp, nmp, K_MAX, BV, fuse_gumbel, Seg())
            npu_sync()

            # ---- (a) DELIVERED wrapper, host time per call, sync each iter ------
            seg_real = Seg()
            # ---- (b) REPLICA segments -------------------------------------------
            seg_rep = Seg()
            for _ in range(iters):
                npu_sync()
                t0 = perf_counter_ns()
                R4M.r4_fused_sampling(*inp[:4], nmp, K_MAX=K_MAX, BV=BV, rand=inp[4],
                                      FUSE_GUMBEL=fuse_gumbel)
                t1 = perf_counter_ns()
                seg_real.add("delivered_host_async_us", t1 - t0)

                wrapper_segments(*inp, nmp, K_MAX, BV, fuse_gumbel, seg_rep)
                npu_sync()

            # ---- (c) fully SERIAL per-call (sync inside the timed region) -------
            seg_ser = Seg()
            for _ in range(iters):
                npu_sync()
                t0 = perf_counter_ns()
                R4M.r4_fused_sampling(*inp[:4], nmp, K_MAX=K_MAX, BV=BV, rand=inp[4],
                                      FUSE_GUMBEL=fuse_gumbel)
                npu_sync()
                t1 = perf_counter_ns()
                seg_ser.add("delivered_host_PLUS_device_us", t1 - t0)

            use_split, S, NBS, S_TAIL = R4M._r4_split_plan(B, V, K_MAX, BV)
            n_launch = (4 if fuse_gumbel else 3) if use_split else 2
            keys = sorted(k for k in seg_rep.stats() if not k.startswith("_"))
            r_rep = summarize(f"[{mode}] case{ci} REPLICA segments (fuse_gumbel={fuse_gumbel})",
                              seg_rep, keys)
            r_real = summarize(f"[{mode}] case{ci} DELIVERED wrapper", seg_real)
            r_ser = summarize(f"[{mode}] case{ci} DELIVERED serial(host+device)", seg_ser)

            host_async = seg_real.med("delivered_host_async_us")
            results[ci] = dict(
                B=B, V=V, K_MAX=K_MAX, BV=BV, need_min_p=nmp, why=why,
                use_split=use_split, S=S, NBS=NBS, S_TAIL=S_TAIL, n_launch=n_launch,
                replica=r_rep, delivered=r_real, serial=r_ser,
                derived=dict(
                    host_async_us=round(host_async, 3),
                    host_async_per_launch_us=round(host_async / n_launch, 3),
                    serial_us=round(seg_ser.med("delivered_host_PLUS_device_us"), 3),
                    # serial - host_async == device time as seen by a serialized caller
                    implied_device_us=round(seg_ser.med("delivered_host_PLUS_device_us")
                                            - host_async, 3),
                    replica_sum_us=r_rep["sum_median_us"],
                    replica_vs_delivered_ratio=round(r_rep["sum_median_us"] / host_async, 4)
                    if host_async else None,
                ),
            )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--cases", default="1,3,13,7")
    ap.add_argument("--out", default="probe1_host_segments.json")
    a = ap.parse_args()
    cases = [int(x) for x in a.cases.split(",") if x.strip()]

    fp = env_fingerprint()
    print("env:", fp)
    import hashlib
    kp = os.path.join(R4, "r4_fused_sampling.py")
    fp["delivered_kernel_sha256"] = hashlib.sha256(open(kp, "rb").read()).hexdigest()
    print("delivered kernel sha256:", fp["delivered_kernel_sha256"])

    prof_root = tempfile.mkdtemp(prefix="probe1_prof_", dir=HERE)
    out = {"_meta": dict(
        description="R4 host wrapper segment decomposition under 3 msprof configs",
        measurement="time.perf_counter_ns() around HOST calls only; launches are async so "
                    "this is host submission cost, NOT kernel duration (see probe_common.py)",
        iters=a.iters, cases=cases, env=fp,
        probe_modes={
            "none": "no profiler at all",
            "npu_only": "torch_npu.profiler activities=[NPU] (harness per-case path)",
            "cpu_npu": "activities=[CPU,NPU] + per-step record_function (harness _batched path "
                       "== the config that produced the round-3 span_vs_scored CSVs)",
        })}
    try:
        for fuse in (True, False):
            for mode in ("none", "npu_only", "cpu_npu"):
                print("\n" + "=" * 100)
                print(f"### FUSE_GUMBEL={fuse}  msprof={mode}")
                print("=" * 100)
                out.setdefault("fused" if fuse else "mask", {})[mode] = run_mode(
                    mode, cases, a.iters, fuse, os.path.join(prof_root, f"{fuse}_{mode}"))
    finally:
        shutil.rmtree(prof_root, ignore_errors=True)

    # ------------------------------------------------ cross-mode comparison table ---
    print("\n" + "=" * 100)
    print("C1 ISOLATION: delivered-wrapper HOST async time per call, by msprof config")
    print("=" * 100)
    cmp_rows = {}
    for task in ("fused", "mask"):
        print(f"\n##### {task} #####")
        print(f"{'case':>5} {'#ln':>4} {'none':>12} {'npu_only':>12} {'cpu_npu':>12} "
              f"{'npu/none':>9} {'cpunpu/none':>12} {'per-launch(cpu_npu)':>20}")
        for ci in cases:
            v = {m: out[task][m][ci]["derived"]["host_async_us"] for m in ("none", "npu_only", "cpu_npu")}
            nl = out[task]["none"][ci]["n_launch"]
            print(f"{ci:>5} {nl:>4} {v['none']:>12.2f} {v['npu_only']:>12.2f} {v['cpu_npu']:>12.2f} "
                  f"{v['npu_only']/v['none']:>9.3f} {v['cpu_npu']/v['none']:>12.3f} "
                  f"{v['cpu_npu']/nl:>20.2f}")
            cmp_rows[f"{task}_case{ci}"] = dict(n_launch=nl, **v,
                                                ratio_npu_over_none=round(v['npu_only'] / v['none'], 4),
                                                ratio_cpunpu_over_none=round(v['cpu_npu'] / v['none'], 4),
                                                per_launch_cpu_npu_us=round(v['cpu_npu'] / nl, 3),
                                                per_launch_none_us=round(v['none'] / nl, 3))
    out["_c1_isolation"] = cmp_rows
    dump(a.out, out)


if __name__ == "__main__":
    main()
