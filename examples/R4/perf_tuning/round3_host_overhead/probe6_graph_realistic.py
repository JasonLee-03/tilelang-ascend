"""PROBE 6 -- the REALISTIC ACL-graph deployment pattern: static input buffers + copy-in + replay.

probe4 captured with fixed input tensors and replayed them, which proves capture works and is
bitwise-identical. But that is not by itself a deployable pattern, and it would be actively
wrong inside cann-bench:

  * A captured graph bakes in the DEVICE ADDRESSES of every tensor it touched.
  * cann-bench deliberately defeats address-based caching: `CallInputPool` /
    `InputPool` (src/kernel_eval/eval/input_pool.py) pre-clones `warmup+repeat` input sets and
    `get_next()` rotates them so that "每次调用 data_ptr 不同" (its own comment), and
    `_DeferredProfileCall.__call__` (op_runner.py:101-107) feeds a different set every step.
  * So a naive replay under the harness would read STALE addresses -> wrong results, and using
    a graph to survive that rotation would be an anti-cheat violation, not an optimisation.

The standard correct pattern is: keep STATIC input buffers, copy the caller's tensors into them,
then replay. This probe measures that pattern end-to-end (copy + replay) and, critically,
verifies CORRECTNESS WHEN THE INPUT *VALUES* CHANGE -- a replay that only reproduces the
capture-time answer would pass probe4 and fail here.

READ-ONLY w.r.t. examples/R4/r4_fused_sampling.py. Usage: probe6_graph_realistic.py [--iters N]
"""
import argparse
import hashlib
import json
import os
import shutil
import statistics
import sys
import tempfile

from probe_common import HERE, R4, dump, env_fingerprint, npu_sync, perf_counter_ns

import torch
import r4_fused_sampling as R4M  # noqa: E402

CANNBENCH_SRC = os.path.abspath(os.path.join(R4, "..", "..", ".agents", "skills", "cann-bench", "src"))
if CANNBENCH_SRC not in sys.path:
    sys.path.insert(0, CANNBENCH_SRC)

SHAPES = {1: (2, 256, 256, 4096, False), 3: (8, 16384, 256, 4096, False),
          13: (6, 2304, 128, 2048, False), 7: (4, 152064, 256, 4096, False)}


def stat(v):
    v = sorted(v)
    return dict(n=len(v), min_us=round(v[0], 3), med_us=round(statistics.median(v), 3),
                p90_us=round(v[int(0.9 * (len(v) - 1))], 3), max_us=round(v[-1], 3))


def harness_stack(level):
    import contextlib
    stack = contextlib.ExitStack()
    if level >= 1:
        from kernel_eval.security.torch_op_guard import TorchOpGuard
        stack.enter_context(TorchOpGuard(mode="block"))
    if level >= 2:
        from kernel_eval.security.device_residency_guard import DeviceResidencyGuard
        stack.enter_context(DeviceResidencyGuard(mode="block"))
    return stack


def prof_ctx(level, root):
    import contextlib
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


def make_set(B, V, K_MAX, dev, seed):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn((B, V), dtype=torch.float32, generator=g).npu()
    return dict(logits=logits,
                top_ks=torch.randint(1, K_MAX + 1, (B,), generator=g,
                                     dtype=torch.int32).npu(),
                top_ps=torch.rand((B,), generator=g).npu().float() * 0.5 + 0.5,
                min_ps=torch.zeros((B,), dtype=torch.float32, device=dev),
                rand=torch.randint(0, 2 ** 30, (B,), generator=g, dtype=torch.int32).npu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--cases", default="1,3,13,7")
    ap.add_argument("--nvals", type=int, default=5, help="distinct input value sets for the "
                                                         "correctness-under-changing-inputs check")
    ap.add_argument("--out", default="probe6_graph_realistic.json")
    a = ap.parse_args()
    dev = torch.device("npu")
    sel = [int(x) for x in a.cases.split(",") if x.strip()]
    fp = env_fingerprint()
    fp["delivered_kernel_sha256"] = hashlib.sha256(
        open(os.path.join(R4, "r4_fused_sampling.py"), "rb").read()).hexdigest()
    print("env:", json.dumps(fp, indent=1))
    res = {"_meta": dict(
        description="realistic ACL-graph deployment: static input buffers + D2D copy-in + "
                    "replay; correctness verified across DISTINCT input VALUE sets",
        why="a captured graph bakes in device ADDRESSES; cann-bench rotates inputs per step "
            "(input_pool.py), so naive replay would read stale addresses. This probe measures "
            "the correct copy-in+replay pattern instead.",
        measurement="perf_counter_ns host-side; serial figures include device time and are "
                    "NOT kernel-duration claims",
        iters=a.iters, n_value_sets=a.nvals, env=fp)}

    prof_root = tempfile.mkdtemp(prefix="probe6_prof_", dir=HERE)
    try:
        for ci in sel:
            B, V, K_MAX, BV, nmp = SHAPES[ci]
            use_split, S, NBS, S_TAIL = R4M._r4_split_plan(B, V, K_MAX, BV)
            n_launch = 4 if use_split else 2
            print("\n" + "=" * 104)
            print(f"case{ci}  B={B} V={V} K_MAX={K_MAX} BV={BV} use_split={use_split} "
                  f"S={S} -> {n_launch} launches")
            print("=" * 104)

            # static buffers that the graph will permanently reference
            static = make_set(B, V, K_MAX, dev, seed=0)
            kw = dict(K_MAX=K_MAX, BV=BV, FUSE_GUMBEL=True)

            def call_graph_inputs():
                return R4M.r4_fused_sampling(static["logits"], static["top_ks"],
                                             static["top_ps"], static["min_ps"], nmp,
                                             rand=static["rand"], **kw)

            for _ in range(6):
                call_graph_inputs()
            npu_sync()
            rec = dict(B=B, V=V, use_split=use_split, S=S, n_launch=n_launch)
            try:
                g = torch.npu.NPUGraph()
                with torch.npu.graph(g):
                    cap_out = call_graph_inputs()
                npu_sync()
                rec["capture_ok"] = True
            except Exception as e:
                rec["capture_ok"] = False
                rec["capture_error"] = f"{type(e).__name__}: {e}"
                print(f"  CAPTURE FAILED: {type(e).__name__}: {str(e)[:300]}")
                res[ci] = rec
                continue

            # ---- CORRECTNESS across DISTINCT input VALUE sets -----------------------
            print("  correctness across distinct input value sets (copy-in + replay vs eager):")
            all_ok, detail = True, []
            for k in range(a.nvals):
                fresh = make_set(B, V, K_MAX, dev, seed=1000 + k)
                # eager reference on the fresh values
                ref = R4M.r4_fused_sampling(fresh["logits"], fresh["top_ks"], fresh["top_ps"],
                                            fresh["min_ps"], nmp, rand=fresh["rand"], **kw)
                npu_sync()
                ref_c = [t.clone() for t in ref]
                # graph path: copy fresh values into the STATIC buffers, then replay
                for key in static:
                    static[key].copy_(fresh[key])
                npu_sync()
                g.replay()
                npu_sync()
                same = [bool(torch.equal(x.cpu(), y.cpu())) for x, y in zip(cap_out, ref_c)]
                ok = all(same)
                all_ok &= ok
                detail.append(dict(value_set=k, per_output_bitwise=same, all_ok=ok))
                print(f"    value set {k}: bitwise==eager {ok}  {same}")
            rec["correctness_all_value_sets_bitwise"] = all_ok
            rec["correctness_detail"] = detail

            # ---- COST: copy-in + replay, vs eager -----------------------------------
            fresh = make_set(B, V, K_MAX, dev, seed=7)
            keys = list(static)

            def copy_in():
                for key in keys:
                    static[key].copy_(fresh[key])

            for level, tag in ((0, "L0_bare"), (5, "L5_harness")):
                with harness_stack(level), prof_ctx(level, os.path.join(prof_root, f"{ci}_{level}")):
                    # copy-in only
                    vc = []
                    for _ in range(a.iters):
                        npu_sync(); t = perf_counter_ns(); copy_in(); t1 = perf_counter_ns()
                        vc.append((t1 - t) / 1000.0); npu_sync()
                    # copy-in + replay, async host
                    va = []
                    for _ in range(a.iters):
                        npu_sync()
                        t = perf_counter_ns(); copy_in(); g.replay(); t1 = perf_counter_ns()
                        va.append((t1 - t) / 1000.0)
                        npu_sync()
                    # copy-in + replay, serial (host + device)
                    vs = []
                    for _ in range(a.iters):
                        npu_sync()
                        t = perf_counter_ns(); copy_in(); g.replay(); npu_sync(); t1 = perf_counter_ns()
                        vs.append((t1 - t) / 1000.0)
                    # eager reference under the same instrumentation
                    ve = []
                    for _ in range(a.iters):
                        npu_sync()
                        t = perf_counter_ns()
                        R4M.r4_fused_sampling(fresh["logits"], fresh["top_ks"], fresh["top_ps"],
                                              fresh["min_ps"], nmp, rand=fresh["rand"], **kw)
                        t1 = perf_counter_ns()
                        ve.append((t1 - t) / 1000.0)
                        npu_sync()
                    ves = []
                    for _ in range(a.iters):
                        npu_sync()
                        t = perf_counter_ns()
                        R4M.r4_fused_sampling(fresh["logits"], fresh["top_ks"], fresh["top_ps"],
                                              fresh["min_ps"], nmp, rand=fresh["rand"], **kw)
                        npu_sync(); t1 = perf_counter_ns()
                        ves.append((t1 - t) / 1000.0)
                rec[tag] = dict(copy_in_only=stat(vc), graph_host_async=stat(va),
                                graph_serial=stat(vs), eager_host_async=stat(ve),
                                eager_serial=stat(ves))
                r = rec[tag]
                print(f"  {tag:<11} copy_in={r['copy_in_only']['med_us']:8.2f}us  "
                      f"copy+replay async={r['graph_host_async']['med_us']:8.2f}us  "
                      f"serial={r['graph_serial']['med_us']:8.2f}us")
                print(f"              eager async={r['eager_host_async']['med_us']:8.2f}us  "
                      f"eager serial={r['eager_serial']['med_us']:8.2f}us")
                print(f"              => host-async speedup {r['eager_host_async']['med_us']/r['graph_host_async']['med_us']:.2f}x ; "
                      f"serial speedup {r['eager_serial']['med_us']/r['graph_serial']['med_us']:.2f}x")
            res[ci] = rec
    finally:
        shutil.rmtree(prof_root, ignore_errors=True)

    # ------------------------------------------------------------------- summary ---
    print("\n" + "=" * 104)
    print("SUMMARY -- realistic graph deployment (copy-in + replay) vs eager, host async us/call")
    print("=" * 104)
    print(f"{'case':>5} {'#ln':>4} {'bitwise ok':>11} {'eager L0':>10} {'graph L0':>10} {'x':>7} "
          f"{'eager L5':>10} {'graph L5':>10} {'x':>7}")
    tab = {}
    for ci in sel:
        r = res.get(str(ci), res.get(ci))
        if not r or not r.get("capture_ok"):
            continue
        e0 = r["L0_bare"]["eager_host_async"]["med_us"]
        g0 = r["L0_bare"]["graph_host_async"]["med_us"]
        e5 = r["L5_harness"]["eager_host_async"]["med_us"]
        g5 = r["L5_harness"]["graph_host_async"]["med_us"]
        print(f"{ci:>5} {r['n_launch']:>4} {str(r['correctness_all_value_sets_bitwise']):>11} "
              f"{e0:>10.2f} {g0:>10.2f} {e0/g0:>7.2f} {e5:>10.2f} {g5:>10.2f} {e5/g5:>7.2f}")
        tab[ci] = dict(n_launch=r["n_launch"],
                       bitwise_ok=r["correctness_all_value_sets_bitwise"],
                       eager_L0=e0, graph_L0=g0, speedup_L0=round(e0 / g0, 3),
                       eager_L5=e5, graph_L5=g5, speedup_L5=round(e5 / g5, 3),
                       copy_in_L0_us=r["L0_bare"]["copy_in_only"]["med_us"],
                       graph_serial_L0_us=r["L0_bare"]["graph_serial"]["med_us"],
                       eager_serial_L0_us=r["L0_bare"]["eager_serial"]["med_us"],
                       serial_speedup_L0=round(r["L0_bare"]["eager_serial"]["med_us"]
                                               / r["L0_bare"]["graph_serial"]["med_us"], 3))
    res["summary"] = tab
    dump(a.out, res)


if __name__ == "__main__":
    main()
