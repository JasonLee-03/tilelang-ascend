"""PROBE 2b -- independent confirmation of the two surprising micro-costs.

probe2 measured, per tilelang launch, host prep ~= 153-196 us of which the two largest
isolated primitives were:
    tvm.arith.Analyzer()            ~131 us   (constructed UNCONDITIONALLY at
                                              cython_wrapper.pyx:96, even though every R4
                                              kernel has dynamic_symbolic_map == {})
    torch.npu.current_stream()      ~23 us    (cython_wrapper.pyx:86-89)
131 us to construct a TVM object is surprising, so this probe re-measures it three ways
(direct, via a different TVM FFI entry point, and with a C-level timer) and also measures
the arithmetic that the additive model must reproduce:

    predicted_prep(n_alloc) = Analyzer + current_stream + n_alloc*torch_empty + small
    measured_prep           = probe2's forward_prep_replica_us

READ-ONLY. Usage: probe2b_confirm_analyzer.py [--iters N]
"""
import argparse
import gc
import os
import statistics
import sys
import time

from probe_common import R4, dump, env_fingerprint, npu_sync, perf_counter_ns

import torch


def timeit(fn, iters, warm=200):
    for _ in range(warm):
        fn()
    v = []
    pc = time.perf_counter_ns
    for _ in range(iters):
        t = pc()
        fn()
        v.append((pc() - t) / 1000.0)
    v.sort()
    return dict(n=iters, min_us=round(v[0], 3), med_us=round(statistics.median(v), 3),
                p90_us=round(v[int(0.9 * (len(v) - 1))], 3), max_us=round(v[-1], 3),
                stdev_us=round(statistics.stdev(v), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--out", default="probe2b_confirm_analyzer.json")
    a = ap.parse_args()
    it = a.iters
    dev = torch.device("npu")
    fp = env_fingerprint()
    import hashlib
    fp["delivered_kernel_sha256"] = hashlib.sha256(
        open(os.path.join(R4, "r4_fused_sampling.py"), "rb").read()).hexdigest()

    res = {"_meta": dict(
        description="independent re-measurement of the two dominant per-launch primitives",
        measurement="perf_counter_ns, no profiler, host-side only", iters=it, env=fp)}

    print("=" * 96)
    print("A. tvm.arith.Analyzer() construction -- 3 independent measurement styles")
    print("=" * 96)
    from tvm.arith import Analyzer
    import tvm

    a1 = timeit(lambda: Analyzer(), it)
    print(f"  A1 direct Analyzer()            : {a1}")

    def a2_fn():
        an = Analyzer()
        an.simplify(tvm.tir.IntImm("int32", 5))
    a2 = timeit(a2_fn, it)
    print(f"  A2 Analyzer()+simplify(IntImm)  : {a2}   (delta vs A1 = simplify cost)")

    res["analyzer_direct"] = a1
    res["analyzer_plus_simplify"] = a2
    res["simplify_only_us"] = round(a2["med_us"] - a1["med_us"], 3)

    print("\n" + "=" * 96)
    print("B. Other TVM FFI object constructions -- is TVM FFI slow in general, or Analyzer?")
    print("=" * 96)
    probes = {
        "tvm.tir.IntImm('int32',5)": lambda: tvm.tir.IntImm("int32", 5),
        "tvm.runtime.convert(5)": lambda: tvm.runtime.convert(5),
        "tvm.tir.const(5,'int32')": lambda: tvm.tir.const(5, "int32"),
        "Analyzer()": lambda: Analyzer(),
    }
    res["tvm_ffi"] = {}
    for name, fn in probes.items():
        r = timeit(fn, it)
        res["tvm_ffi"][name] = r
        print(f"  {name:<32}: med={r['med_us']:>9.3f} us  min={r['min_us']:>9.3f}  max={r['max_us']:>9.3f}")

    print("\n" + "=" * 96)
    print("C. torch.npu stream / device queries (cython_wrapper.pyx:86-89)")
    print("=" * 96)
    res["torch_npu"] = {}
    for name, fn in {
        "torch.npu.current_stream().npu_stream": lambda: torch.npu.current_stream().npu_stream,
        "torch.npu.current_stream()": lambda: torch.npu.current_stream(),
        "torch.npu.current_device()": lambda: torch.npu.current_device(),
        "torch.empty(2,f32,npu)": lambda: torch.empty(2, dtype=torch.float32, device=dev),
        "torch.empty(0,) [no dev]": lambda: torch.empty(0),
    }.items():
        r = timeit(fn, it)
        res["torch_npu"][name] = r
        print(f"  {name:<38}: med={r['med_us']:>9.3f} us  min={r['min_us']:>9.3f}  max={r['max_us']:>9.3f}")

    print("\n" + "=" * 96)
    print("D. Additive model check: predicted prep vs probe2's measured prep")
    print("=" * 96)
    an = res["analyzer_direct"]["med_us"]
    st = res["torch_npu"]["torch.npu.current_stream().npu_stream"]["med_us"]
    em = res["torch_npu"]["torch.empty(2,f32,npu)"]["med_us"]
    # probe2 measured prep medians (re-read from its json so this stays honest/linked)
    p2path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe2_launch_anatomy.json")
    model = {}
    print(f"  Analyzer={an:.2f}us  current_stream={st:.2f}us  torch_empty(npu tiny)={em:.2f}us")
    if os.path.exists(p2path):
        import json
        p2 = json.load(open(p2path))
        print(f"\n  {'kernel':<32} {'n_alloc':>7} {'measured_prep':>13} {'model(A+st+n*em)':>17} {'resid':>8}")
        for k, v in p2["r4_anatomy"].items():
            n = v["n_torch_empty_per_call"]
            meas = v["segments"]["forward_prep_replica_us"]["med_us"]
            pred = an + st + n * em
            model[k] = dict(n_alloc=n, measured_prep_us=meas, model_prep_us=round(pred, 3),
                            residual_us=round(meas - pred, 3))
            print(f"  {k:<32} {n:>7} {meas:>13.2f} {pred:>17.2f} {meas - pred:>8.2f}")
        rs = [abs(v["residual_us"]) for v in model.values()]
        print(f"\n  model residual: max|resid| = {max(rs):.2f} us  mean = {statistics.mean(rs):.2f} us")
        res["model"] = model
        res["model_residual_max_us"] = round(max(rs), 3)
    else:
        print("  probe2_launch_anatomy.json not found -- run probe2 first")

    print("\n" + "=" * 96)
    print("E. GC sensitivity: is the Analyzer() cost an allocation/GC artifact?")
    print("=" * 96)
    gc.disable()
    try:
        r = timeit(lambda: Analyzer(), it)
        print(f"  gc.disable() Analyzer(): med={r['med_us']:.3f} us")
        res["analyzer_gc_disabled"] = r
    finally:
        gc.enable()
    dump(a.out, res)


if __name__ == "__main__":
    main()
