"""PROBE 2 -- decompose ONE tilelang launch into its host primitives + C4 control.

Probe 1 localised ~98% of the R4 host time INSIDE the launch call (`k1(...)`), i.e. inside
tilelang's `CythonKernelWrapper.forward` (jit/adapter/cython/cython_wrapper.pyx:75-198).
`forward`'s `lib` attribute is a `cdef` member so it cannot be monkey-patched; instead this
probe REPLICATES every host step of `forward` in pure Python, times the replica, and times
the real `forward`, so that

    T_libcall  ~=  T_forward_real  -  T_forward_prep_replica

Steps replicated verbatim from cython_wrapper.pyx:75-198 (static-shape case, which R4 is:
`dynamic_symbolic_map == {}` for every R4 kernel, verified at runtime below):
  1. `torch.npu.current_stream().npu_stream`                       (line 86-89)
  2. `Analyzer()` construction                                     (line 96, UNCONDITIONAL)
  3. shape resolution loop over params                              (lines 102-127)
  4. `torch.empty(*shape, dtype, device)` for EACH out_idx param    (line 128)
  5. contiguity check + `ctypes.c_void_p(data_ptr())` per tensor     (lines 141-160)
  6. buffer_device_map verification loop                            (lines 163-175)
  7. buffer_dtype_map verification loop                             (lines 177-182)
  8. `self.lib.call(*call_args)`                                    (line 197)  <- the ACL launch

C4 CONTROLS (the "is this R4-specific?" cut):
  * minimal tilelang elementwise add (examples/elementwise/elementwise_add.py, out_idx=[-1])
  * a tilelang kernel with NO out_idx (0 allocations) built from the same source
  * a native torch_npu op (`torch.add`) as the runtime floor

READ-ONLY w.r.t. the delivered kernel. Usage: probe2_launch_anatomy.py [--iters N]

NOTE: this module deliberately does NOT use `from __future__ import annotations`. PEP 563
would turn the `A: T.Tensor((M, N), dtype)` parameter annotations of the control kernels
into plain strings, and TVMScript then fails with
"Arg(0: runtime.String, 1: runtime.Object) ... expected Object but got str".
(Confirmed by bisect: the identical control compiles with and without that import.)
"""
import argparse
import ctypes
import os
import statistics
import sys
import time

from probe_common import HERE, R4, Seg, dump, env_fingerprint, npu_sync, perf_counter_ns, summarize

import torch  # noqa: E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

import r4_fused_sampling as R4M  # noqa: E402


# ------------------------------------------------------- replica of forward prep ---
def forward_prep_replica(cw, adapter, inputs, sink):
    """Python replica of cython_wrapper.forward() lines 86-193 (everything but lib.call).

    Returns nothing; side effects are pushed into `sink` so the allocated tensors stay
    alive exactly as they would inside the real forward.
    """
    result_idx = adapter.result_idx
    workspace_idx = adapter.workspace_idx
    auto_gm_idx = adapter.auto_gm_idx
    params = adapter.params

    # (1) stream lookup -- forward lines 86-89
    stream = torch.npu.current_stream().npu_stream

    # (2) Analyzer -- forward line 96 (unconditional even when dyn map is empty)
    from tvm.arith import Analyzer
    analyzer = Analyzer()
    sym_val_by_name = {}
    for key, (ref_tensor_idx, ref_shape_idx) in adapter.dynamic_symbolic_map.items():
        sym_val_by_name[str(key)] = int(inputs[ref_tensor_idx].shape[ref_shape_idx])

    # (3)+(4) shape resolution + torch.empty for outputs -- forward lines 102-140
    tensor_list = []
    ins_idx = 0
    for i in range(len(params)):
        if i in result_idx or i in workspace_idx:
            dtype = params[i].dtype
            shape = []
            for s in params[i].shape:
                if isinstance(s, int):
                    shape.append(s)
                else:
                    from tvm import tir
                    from tvm.tir import stmt_functor
                    vmap = {}
                    for v in tir.analysis.undefined_vars(s):
                        vmap[v] = tir.IntImm(v.dtype, sym_val_by_name[str(v)])
                    shape.append(int(analyzer.simplify(stmt_functor.substitute(s, vmap))))
            device = inputs[0].device if len(inputs) > 0 else torch.npu.current_device()
            tensor = torch.empty(*shape, dtype=dtype, device=device)
        elif i in auto_gm_idx:
            device = inputs[0].device if len(inputs) > 0 else torch.npu.current_device()
            tensor = torch.empty(*params[i].shape, dtype=params[i].dtype, device=device)
        else:
            tensor = inputs[ins_idx]
            ins_idx += 1
        tensor_list.append(tensor)
    sink.extend(tensor_list)

    # (5) contiguity + ctypes packing -- forward lines 142-160
    call_args = []
    for i in range(len(tensor_list)):
        tensor = tensor_list[i]
        if isinstance(tensor, torch.Tensor):
            if not tensor.is_contiguous():
                raise ValueError(f"Input tensor at index {i} must be contiguous")
            call_args.append(ctypes.c_void_p(tensor.data_ptr()))
        else:
            call_args.append(tensor)

    # (6) buffer_device_map -- forward lines 163-175
    t0type = tensor_list[0].device.type if isinstance(tensor_list[0], torch.Tensor) else None
    for _param, (buffer_idx, device) in adapter.buffer_device_map.items():
        if isinstance(tensor_list[buffer_idx], torch.Tensor):
            td = tensor_list[buffer_idx].device
            if (t0type != device.type or (td.index is not None and device.index is not None
                                          and td.index != device.index)):
                raise ValueError("device mismatch")

    # (7) buffer_dtype_map -- forward lines 177-182
    for _param, (buffer_idx, torch_dtype) in adapter.buffer_dtype_map.items():
        if isinstance(tensor_list[buffer_idx], torch.Tensor):
            if tensor_list[buffer_idx].dtype != torch_dtype:
                raise ValueError("dtype mismatch")

    # dynamic symbolics + stream (forward lines 190-195)
    for _, (buffer_idx, shape_idx) in adapter.dynamic_symbolic_map.items():
        call_args.append(ctypes.c_int64(inputs[buffer_idx].shape[shape_idx]))
    call_args.append(ctypes.c_void_p(stream))
    return call_args


# --------------------------------------------------------------- micro primitives ---
def micro_primitives(iters, dev):
    seg = Seg()
    sink = []
    from tvm.arith import Analyzer

    for _ in range(200):
        torch.npu.current_stream().npu_stream
        Analyzer()
        torch.empty(2, dtype=torch.float32, device=dev)
    npu_sync()

    for _ in range(iters):
        t = perf_counter_ns(); torch.npu.current_stream().npu_stream; seg.add("p1_current_stream", perf_counter_ns() - t)
        t = perf_counter_ns(); Analyzer(); seg.add("p2_Analyzer_ctor", perf_counter_ns() - t)
        t = perf_counter_ns(); sink.append(torch.empty(2, dtype=torch.float32, device=dev)); seg.add("p3_torch_empty_tiny_npu", perf_counter_ns() - t)
        t = perf_counter_ns(); sink.append(torch.empty(2 * 256, dtype=torch.float32, device=dev)); seg.add("p3b_torch_empty_512f_npu", perf_counter_ns() - t)
        t = perf_counter_ns(); sink.append(torch.empty(2, dtype=torch.float32, device="cpu")); seg.add("p3c_torch_empty_tiny_CPU", perf_counter_ns() - t)
        a = torch.randn(4, device=dev)
        t = perf_counter_ns(); ctypes.c_void_p(a.data_ptr()); seg.add("p4_ctypes_void_p", perf_counter_ns() - t)
        t = perf_counter_ns(); _ = a.is_contiguous(); seg.add("p5_is_contiguous", perf_counter_ns() - t)
        t = perf_counter_ns(); _ = a.dtype; _ = a.device; _ = a.device.type; seg.add("p6_attr_access", perf_counter_ns() - t)
        if len(sink) > 4000:
            sink.clear()
    npu_sync()
    return summarize("micro primitives (isolated, no profiler)", seg)


# ------------------------------------------------------------------- C4 controls ---
@tilelang.jit(out_idx=[-1])
def vec_add_ctl(M, N, block_M, block_N, dtype="float"):
    """Verbatim structure of examples/elementwise/elementwise_add.py (the minimal control)."""
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)
                T.barrier_all()
                T.tile.add(c_ub, a_ub, b_ub)
                T.barrier_all()
                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])
    return main


@tilelang.jit
def vec_add_noout(M, N, block_M, block_N, dtype="float"):
    """Same kernel, NO out_idx -> 0 torch.empty inside forward. Isolates alloc cost."""
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)
                T.barrier_all()
                T.tile.add(c_ub, a_ub, b_ub)
                T.barrier_all()
                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])
    return main


def control_launches(iters, dev):
    """Per-launch host cost for: minimal tilelang (1 alloc), tilelang (0 allocs), torch native."""
    res = {}
    M = N = 256
    a = torch.randn(M, N, device=dev)
    b = torch.randn(M, N, device=dev)
    c = torch.empty(M, N, device=dev)

    f_out = vec_add_ctl(M, N, 128, 128)
    f_noout = vec_add_noout(M, N, 128, 128)
    sink = []

    for _ in range(20):
        sink.append(f_out(a, b)); f_noout(a, b, c); torch.add(a, b)
    npu_sync(); sink.clear()

    for label, fn, nargs in (
        ("tilelang_minimal_out_idx_1alloc", lambda: f_out(a, b), 1),
        ("tilelang_minimal_NO_out_idx_0alloc", lambda: f_noout(a, b, c), 0),
        ("torch_native_add_npu", lambda: torch.add(a, b), 1),
    ):
        seg = Seg()
        for _ in range(iters):
            npu_sync()
            t = perf_counter_ns(); fn(); t1 = perf_counter_ns()
            seg.add("host_async_us", t1 - t)
            npu_sync()
        # also the no-sync back-to-back rate
        seg2 = Seg()
        t = perf_counter_ns()
        for _ in range(iters):
            sink.append(fn())
        tot = (perf_counter_ns() - t) / 1000.0
        npu_sync()
        seg2.add("backtoback_us", tot / iters * 1000.0)
        r = summarize(f"C4 control: {label}", seg)
        r["backtoback_per_call_us"] = round(tot / iters, 3)
        r["n_torch_empty_per_call"] = nargs
        res[label] = r
        if len(sink) > 2000:
            sink.clear()
    return res


# ------------------------------------------------ R4 launch anatomy (prep vs libcall) ---
def r4_launch_anatomy(iters, dev):
    """For each R4 kernel object: real forward total vs replica prep => implied lib.call."""
    B, V, K_MAX, BV = 2, 256, 256, 4096
    logits = torch.randn((B, V), dtype=torch.float32, device=dev)
    top_ks = torch.full((B,), K_MAX, dtype=torch.int32, device=dev)
    top_ps = torch.ones((B,), dtype=torch.float32, device=dev)
    min_ps = torch.zeros((B,), dtype=torch.float32, device=dev)
    rand = torch.arange(B, dtype=torch.int32, device=dev)
    pv = torch.empty((B * 1, K_MAX), dtype=torch.float32, device=dev)
    pv1 = torch.empty((B * 1,), dtype=torch.float32, device=dev)
    pv2 = torch.empty((B * 1,), dtype=torch.float32, device=dev)
    m_out = torch.empty(B, dtype=torch.float32, device=dev)
    s_out = torch.empty(B, dtype=torch.float32, device=dev)
    tau = torch.empty(B, dtype=torch.float32, device=dev)

    targets = {
        "K1_r4_topk_stats(out4)": (R4M.r4_topk_stats(B, V, K_MAX, BV, False),
                                   (logits, top_ks, top_ps, min_ps)),
        "K2m_r4_mask_sample(out1)": (R4M.r4_mask_sample(B, V, BV), (logits, m_out, s_out, tau)),
        "K2g_r4_gumbel(out1)": (R4M.r4_gumbel_sample(B, V, BV), (logits, tau, rand)),
        "K1A_split(out0)": (R4M.r4_topk_stats_splitA(B, V, K_MAX, BV, 1, 1, -1), (logits, pv)),
        "K1B_split(out4)": (R4M.r4_topk_stats_splitB(B, V, K_MAX, BV, 1, False),
                            (logits, top_ks, top_ps, min_ps, pv)),
        "K2gA_split(out0)": (R4M.r4_gumbel_sample_splitA(B, V, BV, 1, 1),
                             (logits, tau, rand, pv1, pv2)),
        "K2gB_split(out1)": (R4M.r4_gumbel_sample_splitB(B, 1), (pv1, pv2)),
    }

    out = {}
    sink = []
    for name, (jk, args) in targets.items():
        ad = jk.adapter
        cw = ad.cython_wrapper
        n_alloc = len(ad.result_idx) + len(ad.workspace_idx) + len(ad.auto_gm_idx)
        assert not ad.dynamic_symbolic_map, f"{name} has dynamic symbolics; replica invalid"
        for _ in range(10):
            cw.forward(list(args)); forward_prep_replica(cw, ad, list(args), sink)
        npu_sync(); sink.clear()

        seg = Seg()
        for _ in range(iters):
            npu_sync()
            t = perf_counter_ns(); cw.forward(list(args)); t1 = perf_counter_ns()
            seg.add("forward_real_us", t1 - t)
            t = perf_counter_ns(); forward_prep_replica(cw, ad, list(args), sink); t2 = perf_counter_ns()
            seg.add("forward_prep_replica_us", t2 - t)
            t = perf_counter_ns(); _ = list(args); t3 = perf_counter_ns()
            seg.add("list_args_only_us", t3 - t)
            npu_sync()
            if len(sink) > 3000:
                sink.clear()

        r = summarize(f"R4 launch anatomy: {name}", seg)
        real = seg.med("forward_real_us")
        prep = seg.med("forward_prep_replica_us")
        la = seg.med("list_args_only_us")
        r.update(n_torch_empty_per_call=n_alloc, result_idx=list(ad.result_idx),
                 n_params=len(ad.params),
                 implied_libcall_us=round(real - prep, 3),
                 prep_minus_listargs_us=round(prep - la, 3),
                 prep_pct_of_forward=round(100.0 * prep / real, 2) if real else None)
        print(f"    -> n_alloc={n_alloc}  forward_real={real:.2f}us  prep_replica={prep:.2f}us  "
              f"IMPLIED lib.call={real - prep:.2f}us  ({100*(real-prep)/real:.1f}% of forward)")
        out[name] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--out", default="probe2_launch_anatomy.json")
    a = ap.parse_args()
    dev = torch.device("npu")
    fp = env_fingerprint()
    import hashlib
    fp["delivered_kernel_sha256"] = hashlib.sha256(
        open(os.path.join(R4, "r4_fused_sampling.py"), "rb").read()).hexdigest()
    print("env:", fp)

    res = {"_meta": dict(
        description="anatomy of ONE tilelang launch: forward prep replica vs real => lib.call; "
                    "plus C4 controls (minimal tilelang / 0-alloc tilelang / torch native)",
        measurement="perf_counter_ns around HOST calls, no profiler; NOT kernel durations",
        iters=a.iters, env=fp)}

    print("\n" + "#" * 100 + "\n# PART A: micro primitives\n" + "#" * 100)
    res["micro"] = micro_primitives(a.iters * 5, dev)

    print("\n" + "#" * 100 + "\n# PART B: C4 controls -- is ~370us R4-specific?\n" + "#" * 100)
    res["controls"] = control_launches(a.iters, dev)

    print("\n" + "#" * 100 + "\n# PART C: R4 per-kernel launch anatomy\n" + "#" * 100)
    res["r4_anatomy"] = r4_launch_anatomy(a.iters, dev)

    # ------------------------------------------------------------- verdict table ---
    print("\n" + "=" * 100)
    print("VERDICT TABLE (all host us/launch, NO profiler)")
    print("=" * 100)
    print(f"{'subject':<40} {'#alloc':>7} {'forward/call':>13} {'prep':>9} {'lib.call':>10}")
    for k, v in res["controls"].items():
        print(f"{k:<40} {v['n_torch_empty_per_call']:>7} "
              f"{v['segments']['host_async_us']['med_us']:>13.2f} {'-':>9} {'-':>10}")
    for k, v in res["r4_anatomy"].items():
        print(f"{k:<40} {v['n_torch_empty_per_call']:>7} "
              f"{v['segments']['forward_real_us']['med_us']:>13.2f} "
              f"{v['segments']['forward_prep_replica_us']['med_us']:>9.2f} "
              f"{v['implied_libcall_us']:>10.2f}")
    dump(a.out, res)


if __name__ == "__main__":
    main()
