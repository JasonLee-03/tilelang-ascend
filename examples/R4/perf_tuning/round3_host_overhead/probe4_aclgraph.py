"""PROBE 4 -- ACL graph capture: feasibility, correctness, replay cost, and the
marginal per-launch cost under capture (which adjudicates DESIGN.md 16.2 option A).

WHY THIS PROBE EXISTS
---------------------
DESIGN.md 16.2.2 rejected option A (4-pass decomposition: K1 -> Pass1/2/3/4, i.e. 5 launches
instead of 2) partly because `a_launch` was unmeasured (risk R-LAUNCH). Round 3 then measured
`a_launch < 1.54 us` -- but that is a DEVICE-side number (the duration intercept of the smallest
complete kernel, K2gB). This investigation measured the EAGER HOST-side cost of one launch at
~196-206 us bare and ~464-537 us under the cann-bench harness. The two point in OPPOSITE
directions for option A, so only a capture measurement can adjudicate.

API EVIDENCE (checked, not guessed):
  * torch_npu/npu/__init__.py:111-113 exports "NPUGraph", "graph", "graph_pool_handle"
  * torch_npu/npu/graphs.py:327 class NPUGraph(torch_npu._C._NPUGraph) with capture_begin /
    capture_end / replay (lines 342 / 361 / 372)
  * torch_npu/npu/graphs.py:464 class graph -- context manager; __enter__ (518) syncs, empties
    cache, switches to an internal SIDE STREAM and calls capture_begin; __exit__ (532) calls
    capture_end. Because it makes its own side stream current, tilelang's
    `torch.npu.current_stream().npu_stream` (cython_wrapper.pyx:86-89) picks up the CAPTURE
    stream, which is why tilelang kernels are capturable.
  * repo precedent: examples/aclgraph/rms_rope_aclgraph.py:270-275 captures TWO tilelang
    kernels into one graph and replays it; examples/aclgraph/README.md states the purpose is
    "reducing kernel launch overhead".

READ-ONLY w.r.t. examples/R4/r4_fused_sampling.py. Usage: probe4_aclgraph.py [--iters N]
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
import r4_fused_sampling as R4M  # noqa: E402

CANNBENCH_SRC = os.path.abspath(os.path.join(R4, "..", "..", ".agents", "skills", "cann-bench", "src"))
if CANNBENCH_SRC not in sys.path:
    sys.path.insert(0, CANNBENCH_SRC)

SHAPES = {
    1:  (2,  256,    256, 4096, False),
    3:  (8,  16384,  256, 4096, False),
    13: (6,  2304,   128, 2048, False),
    7:  (4,  152064, 256, 4096, False),
}


def make_inputs(B, V, K_MAX, dev):
    g = torch.Generator().manual_seed(0)
    logits = torch.randn((B, V), dtype=torch.float32, generator=g).npu()
    return dict(logits=logits,
                top_ks=torch.full((B,), K_MAX, dtype=torch.int32, device=dev),
                top_ps=torch.ones((B,), dtype=torch.float32, device=dev),
                min_ps=torch.zeros((B,), dtype=torch.float32, device=dev),
                rand=torch.arange(B, dtype=torch.int32, device=dev))


def harness_stack(level):
    """Level 0 = bare; level 5 = the full cann-bench instrumentation stack (probe3's L5)."""
    stack = contextlib.ExitStack()
    if level >= 1:
        from kernel_eval.security.torch_op_guard import TorchOpGuard
        stack.enter_context(TorchOpGuard(mode="block"))
    if level >= 2:
        from kernel_eval.security.device_residency_guard import DeviceResidencyGuard
        stack.enter_context(DeviceResidencyGuard(mode="block"))
    return stack


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


def stat(v):
    v = sorted(v)
    return dict(n=len(v), min_us=round(v[0], 3), med_us=round(statistics.median(v), 3),
                p90_us=round(v[int(0.9 * (len(v) - 1))], 3), max_us=round(v[-1], 3))


# ------------------------------------------------------------------- chains -------
def build_chains(B, V, K_MAX, BV, nmp, S, NBS, S_TAIL, inp, dev):
    """Explicit 2-launch (delivered/base) and 4-launch (rev3 split) chains, read-only."""
    logits, ks, ps, ms, rand = (inp["logits"], inp["top_ks"], inp["top_ps"],
                                inp["min_ps"], inp["rand"])
    pv = torch.empty((B * S, K_MAX), dtype=torch.float32, device=dev)
    pvv = torch.empty((B * S,), dtype=torch.float32, device=dev)
    pvi = torch.empty((B * S,), dtype=torch.float32, device=dev)
    k1 = R4M.r4_topk_stats(B, V, K_MAX, BV, nmp)
    k2g = R4M.r4_gumbel_sample(B, V, BV)
    k1a = R4M.r4_topk_stats_splitA(B, V, K_MAX, BV, S, NBS, S_TAIL)
    k1b = R4M.r4_topk_stats_splitB(B, V, K_MAX, BV, S, nmp)
    k2ga = R4M.r4_gumbel_sample_splitA(B, V, BV, S, NBS)
    k2gb = R4M.r4_gumbel_sample_splitB(B, S)

    def chain2():
        m, s, tau, valid = k1(logits, ks, ps, ms)
        tok = k2g(logits, tau, rand)
        return m, s, tau, valid, tok

    def chain4():
        k1a(logits, pv)
        m, s, tau, valid = k1b(logits, ks, ps, ms, pv)
        k2ga(logits, tau, rand, pvv, pvi)
        tok = k2gb(pvv, pvi)
        return m, s, tau, valid, tok

    return chain2, chain4


def measure_replay(g, iters, level, prof_root, tag):
    """Host cost of g.replay(): async submit, serial (sync both sides), back-to-back, span."""
    out = {}
    with harness_stack(level) as _st, prof_ctx(level, prof_root):
        # async submit cost
        v = []
        for _ in range(iters):
            npu_sync()
            t = perf_counter_ns(); g.replay(); t1 = perf_counter_ns()
            v.append((t1 - t) / 1000.0)
            npu_sync()
        out["replay_host_async"] = stat(v)

        # serial: sync, replay, sync  -> host submit + device execution
        v = []
        for _ in range(iters):
            npu_sync()
            t = perf_counter_ns(); g.replay(); npu_sync(); t1 = perf_counter_ns()
            v.append((t1 - t) / 1000.0)
        out["replay_serial_host_plus_device"] = stat(v)

        # back-to-back throughput (deployment-relevant steady state)
        npu_sync()
        t = perf_counter_ns()
        for _ in range(iters):
            g.replay()
        npu_sync()
        t1 = perf_counter_ns()
        out["backtoback_per_replay_us"] = round((t1 - t) / 1000.0 / iters, 3)

        # device-timeline span of one replay, via NPU events
        v = []
        for _ in range(iters):
            npu_sync()
            e0 = torch.npu.Event(enable_timing=True); e1 = torch.npu.Event(enable_timing=True)
            e0.record(); g.replay(); e1.record()
            npu_sync()
            v.append(e0.elapsed_time(e1) * 1000.0)
        out["replay_span_events"] = stat(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--cases", default="1,3,13,7")
    ap.add_argument("--out", default="probe4_aclgraph.json")
    a = ap.parse_args()
    dev = torch.device("npu")
    sel = [int(x) for x in a.cases.split(",") if x.strip()]

    fp = env_fingerprint()
    fp["delivered_kernel_sha256"] = hashlib.sha256(
        open(os.path.join(R4, "r4_fused_sampling.py"), "rb").read()).hexdigest()
    print("env:", json.dumps(fp, indent=1))

    res = {"_meta": dict(
        description="ACL graph capture feasibility / correctness / replay cost / marginal "
                    "per-launch cost under capture",
        api_evidence="torch_npu/npu/__init__.py:111-113; npu/graphs.py:327,342,361,372,464,518,532; "
                     "repo precedent examples/aclgraph/rms_rope_aclgraph.py:270-275",
        measurement="perf_counter_ns for HOST replay cost; torch.npu.Event for device-timeline "
                    "span. replay_serial is host+device and is NOT a kernel-duration claim.",
        iters=a.iters, env=fp)}

    # ---------------------------------------------------------- A. API existence ---
    print("=" * 100)
    print("A. torch_npu ACL-graph API surface (runtime check)")
    print("=" * 100)
    api = {}
    for name in ("NPUGraph", "graph", "graph_pool_handle", "make_graphed_callables"):
        ok = hasattr(torch.npu, name)
        api[name] = ok
        print(f"  torch.npu.{name:<24} present={ok}")
    res["api_surface"] = api
    if not (api.get("NPUGraph") and api.get("graph")):
        res["verdict"] = "UNSUPPORTED: torch.npu.NPUGraph/graph missing"
        print("\n" + res["verdict"])
        dump(a.out, res)
        return

    prof_root = tempfile.mkdtemp(prefix="probe4_prof_", dir=HERE)
    try:
        # ------------------------------------- B. capture the DELIVERED wrapper ------
        print("\n" + "=" * 100)
        print("B. Capture the DELIVERED r4_fused_sampling() wrapper end-to-end")
        print("=" * 100)
        res["wrapper_capture"] = {}
        for ci in sel:
            B, V, K_MAX, BV, nmp = SHAPES[ci]
            inp = make_inputs(B, V, K_MAX, dev)
            args = (inp["logits"], inp["top_ks"], inp["top_ps"], inp["min_ps"])
            kw = dict(K_MAX=K_MAX, BV=BV, rand=inp["rand"], FUSE_GUMBEL=True)
            use_split, S, NBS, S_TAIL = R4M._r4_split_plan(B, V, K_MAX, BV)
            n_launch = 4 if use_split else 2

            # warm up on the default stream so all JIT compilation happens BEFORE capture
            eager_out = None
            for i in range(6):
                eager_out = R4M.r4_fused_sampling(*args, nmp, **kw)
            npu_sync()
            eager_ref = [t.clone() for t in eager_out]

            rec = dict(B=B, V=V, use_split=use_split, S=S, n_launch=n_launch)
            try:
                g = torch.npu.NPUGraph()
                t = perf_counter_ns()
                with torch.npu.graph(g):
                    cap_out = R4M.r4_fused_sampling(*args, nmp, **kw)
                npu_sync()
                rec["capture_ok"] = True
                rec["capture_wall_ms"] = round((perf_counter_ns() - t) / 1e6, 3)

                # correctness: replay then compare bitwise with the eager reference
                g.replay(); npu_sync()
                match = []
                for j, (c, r) in enumerate(zip(cap_out, eager_ref)):
                    same = bool(torch.equal(c.cpu(), r.cpu()))
                    match.append(same)
                rec["bitwise_equal_to_eager"] = all(match)
                rec["per_output_bitwise"] = match
                print(f"  case{ci:<3} ({n_launch} launches): capture OK in "
                      f"{rec['capture_wall_ms']:.1f} ms; bitwise==eager: {rec['bitwise_equal_to_eager']} "
                      f"{match}")

                # replay cost, bare and under the full harness stack
                for lvl, tag in ((0, "L0_bare"), (5, "L5_harness")):
                    rec[tag] = measure_replay(g, a.iters, lvl,
                                              os.path.join(prof_root, f"w{ci}_{lvl}"), tag)
                    r = rec[tag]
                    print(f"      {tag:<11} replay_host_async med={r['replay_host_async']['med_us']:8.2f} us"
                          f"  serial={r['replay_serial_host_plus_device']['med_us']:8.2f} us"
                          f"  backtoback={r['backtoback_per_replay_us']:8.2f} us"
                          f"  span_ev={r['replay_span_events']['med_us']:8.2f} us")
                    print(f"                  -> per-launch equiv (async/{n_launch}) = "
                          f"{r['replay_host_async']['med_us']/n_launch:.2f} us")
            except Exception as e:
                rec["capture_ok"] = False
                rec["capture_error"] = f"{type(e).__name__}: {e}"
                print(f"  case{ci:<3}: CAPTURE FAILED -> {type(e).__name__}: {str(e)[:300]}")
            res["wrapper_capture"][ci] = rec

        # ------------------------- C. marginal per-launch cost UNDER CAPTURE ---------
        print("\n" + "=" * 100)
        print("C. Marginal per-launch host cost under capture: 2-launch vs 4-launch chain")
        print("   (this is the number DESIGN 16.2 option A needs: what does ONE MORE")
        print("    kernel cost once the chain is captured?)")
        print("=" * 100)
        res["marginal"] = {}
        for ci in (3, 7):
            if ci not in sel:
                continue
            B, V, K_MAX, BV, nmp = SHAPES[ci]
            inp = make_inputs(B, V, K_MAX, dev)
            use_split, S, NBS, S_TAIL = R4M._r4_split_plan(B, V, K_MAX, BV)
            chain2, chain4 = build_chains(B, V, K_MAX, BV, nmp, S, NBS, S_TAIL, inp, dev)
            rec = dict(B=B, V=V, S=S)
            for nm, fn, nl in (("chain2", chain2, 2), ("chain4", chain4, 4)):
                for _ in range(6):
                    fn()
                npu_sync()

                # ---- EAGER reference for the same chain (host async + serial) ----
                va, vs = [], []
                for _ in range(a.iters):
                    npu_sync()
                    t = perf_counter_ns(); fn(); t1 = perf_counter_ns()
                    va.append((t1 - t) / 1000.0)
                    npu_sync()
                for _ in range(a.iters):
                    npu_sync()
                    t = perf_counter_ns(); fn(); npu_sync(); t1 = perf_counter_ns()
                    vs.append((t1 - t) / 1000.0)
                eager = dict(n_launch=nl, host_async=stat(va), serial=stat(vs))
                print(f"  case{ci} {nm} ({nl} ln) EAGER : host_async="
                      f"{eager['host_async']['med_us']:.2f}us  serial="
                      f"{eager['serial']['med_us']:.2f}us")
                try:
                    g = torch.npu.NPUGraph()
                    with torch.npu.graph(g):
                        fn()
                    npu_sync()
                    m0 = measure_replay(g, a.iters, 0, os.path.join(prof_root, f"m{ci}{nm}0"), nm)
                    m5 = measure_replay(g, a.iters, 5, os.path.join(prof_root, f"m{ci}{nm}5"), nm)
                    rec[nm] = dict(n_launch=nl, eager=eager, L0=m0, L5=m5, ok=True)
                    print(f"  case{ci} {nm} ({nl} ln) CAPTURED L0: replay_async="
                          f"{m0['replay_host_async']['med_us']:.2f}us "
                          f"serial={m0['replay_serial_host_plus_device']['med_us']:.2f}us "
                          f"b2b={m0['backtoback_per_replay_us']:.2f}us | "
                          f"L5 replay_async={m5['replay_host_async']['med_us']:.2f}us "
                          f"serial={m5['replay_serial_host_plus_device']['med_us']:.2f}us")
                except Exception as e:
                    rec[nm] = dict(n_launch=nl, eager=eager, ok=False,
                                   error=f"{type(e).__name__}: {e}")
                    print(f"  case{ci} {nm}: CAPTURE FAILED {type(e).__name__}: {str(e)[:200]}")
            if rec.get("chain2", {}).get("ok") and rec.get("chain4", {}).get("ok"):
                a2 = rec["chain2"]["L0"]["replay_host_async"]["med_us"]
                a4 = rec["chain4"]["L0"]["replay_host_async"]["med_us"]
                s2 = rec["chain2"]["L0"]["replay_serial_host_plus_device"]["med_us"]
                s4 = rec["chain4"]["L0"]["replay_serial_host_plus_device"]["med_us"]
                e2 = rec["chain2"]["eager"]["serial"]["med_us"]
                e4 = rec["chain4"]["eager"]["serial"]["med_us"]
                b2 = rec["chain2"]["L0"]["backtoback_per_replay_us"]
                b4 = rec["chain4"]["L0"]["backtoback_per_replay_us"]
                eb2 = rec["chain2"]["eager"]["host_async"]["med_us"]
                eb4 = rec["chain4"]["eager"]["host_async"]["med_us"]
                rec["marginal_host_per_extra_launch_L0_us"] = round((a4 - a2) / 2, 3)
                rec["marginal_serial_per_extra_launch_L0_us"] = round((s4 - s2) / 2, 3)
                rec["HEAD_TO_HEAD"] = dict(
                    eager_serial_chain2_us=e2, eager_serial_chain4_us=e4,
                    eager_serial_chain4_over_chain2=round(e4 / e2, 4),
                    captured_serial_chain2_us=s2, captured_serial_chain4_us=s4,
                    captured_serial_chain4_over_chain2=round(s4 / s2, 4),
                    eager_b2b_chain2_us=eb2, eager_b2b_chain4_us=eb4,
                    captured_b2b_chain2_us=b2, captured_b2b_chain4_us=b4,
                    captured_b2b_chain4_over_chain2=round(b4 / b2, 4))
                print(f"  ==> case{ci} MARGINAL host cost of ONE extra launch under capture: "
                      f"async {(a4-a2)/2:+.2f} us, serial {(s4-s2)/2:+.2f} us "
                      f"(compare EAGER: ~196-206 us bare / ~464-537 us in harness)")
                print(f"  ==> case{ci} HEAD-TO-HEAD chain4/chain2 ratio: EAGER serial "
                      f"{e4/e2:.3f}x (>1 = rev3 WORSE) | CAPTURED serial {s4/s2:.3f}x "
                      f"(<1 = rev3 BETTER) | CAPTURED back-to-back {b4/b2:.3f}x")
            res["marginal"][ci] = rec

        # --------------------------- D. eager vs captured, side by side --------------
        print("\n" + "=" * 100)
        print("D. EAGER vs CAPTURED, per-launch host cost (L0 bare / L5 harness)")
        print("=" * 100)
        print(f"{'case':>5} {'#ln':>4} {'eager L0/ln':>12} {'cap L0/ln':>11} {'ratio':>7} "
              f"{'eager L5/ln':>12} {'cap L5/ln':>11} {'ratio':>7}")
        try:
            lad = json.load(open(os.path.join(HERE, "probe3_ladder_and_span.json")))["ladder"]
        except Exception:
            lad = {}
        cmp_tab = {}
        for ci in sel:
            w = res["wrapper_capture"].get(ci, {})
            lk = lad.get(str(ci), lad.get(ci)) if lad else None
            if not w.get("capture_ok") or not lk:
                print(f"{ci:>5} {'-':>4}   (skipped: capture_ok={w.get('capture_ok')} "
                      f"ladder_present={bool(lk)})")
                continue
            nl = w["n_launch"]
            lv = lk["levels"]
            e0 = lv.get("0", lv.get(0))["per_launch_us"]
            e5 = lv.get("5", lv.get(5))["per_launch_us"]
            c0 = w["L0_bare"]["replay_host_async"]["med_us"] / nl
            c5 = w["L5_harness"]["replay_host_async"]["med_us"] / nl
            print(f"{ci:>5} {nl:>4} {e0:>12.2f} {c0:>11.2f} {c0/e0:>7.3f} {e5:>12.2f} {c5:>11.2f} {c5/e5:>7.3f}")
            cmp_tab[ci] = dict(n_launch=nl, eager_L0_per_launch=e0, cap_L0_per_launch=round(c0, 3),
                               ratio_L0=round(c0 / e0, 4), eager_L5_per_launch=e5,
                               cap_L5_per_launch=round(c5, 3), ratio_L5=round(c5 / e5, 4))
        res["eager_vs_captured"] = cmp_tab
    finally:
        shutil.rmtree(prof_root, ignore_errors=True)
    dump(a.out, res)


if __name__ == "__main__":
    main()
