"""PROBE 5 -- quantify each CANDIDATE MITIGATION (evaluate only; nothing is implemented).

Part A  execution_backend: `@tilelang.jit(..., execution_backend=)` accepts
        "cython" (default, jit/__init__.py:36) | "ctypes" | "dlpack". cython's forward is the
        Python-heavy path probe2 dissected. Does a different backend cost less host time?
        Measured on the minimal control kernel (the R4 decorator is NOT touched).
Part B  framework ceiling: probe2 showed per-launch prep = Analyzer() + current_stream() +
        n_alloc*torch.empty + per-param Python loops. `Analyzer()` (cython_wrapper.pyx:96) is
        constructed UNCONDITIONALLY although every R4 kernel has dynamic_symbolic_map == {}.
        This measures the replica prep with (i) Analyzer made lazy and (ii) the stream cached,
        to bound what a tilelang-side fix would buy.
Part C  wrapper-level savings (our code): memoising `_r4_split_plan`, reusing the GM
        workspace instead of `torch.empty` per call, and short-circuiting the pre-checks.
        Each is measured as the DELTA it would remove from one fn() call.
Part D  ACL graph, restated from probe4 for a single comparison table.

READ-ONLY w.r.t. examples/R4/r4_fused_sampling.py. Usage: probe5_mitigations.py [--iters N]
"""
import argparse
import hashlib
import json
import os
import statistics
import time

from probe_common import HERE, R4, Seg, dump, env_fingerprint, npu_sync, perf_counter_ns, summarize

import torch
import tilelang
import tilelang.language as T

import r4_fused_sampling as R4M  # noqa: E402

SHAPES = {1: (2, 256, 256, 4096, False), 3: (8, 16384, 256, 4096, False),
          13: (6, 2304, 128, 2048, False), 7: (4, 152064, 256, 4096, False)}


def stat(v):
    v = sorted(v)
    return dict(n=len(v), min_us=round(v[0], 3), med_us=round(statistics.median(v), 3),
                max_us=round(v[-1], 3))


def timeit(fn, iters, warm=50, sync_each=True):
    for _ in range(warm):
        fn()
    if sync_each:
        npu_sync()
    v = []
    pc = perf_counter_ns
    for _ in range(iters):
        t = pc(); fn(); v.append((pc() - t) / 1000.0)
    return stat(v)


# --------------------------------------------------------- Part A: backends --------
def make_control(backend):
    @tilelang.jit(out_idx=[-1], execution_backend=backend)
    def vec_add_bk(M, N, block_M, block_N, dtype="float"):
        m_num = M // block_M
        n_num = N // block_N
        VEC_NUM = 2

        @T.prim_func
        def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype),
                 C: T.Tensor((M, N), dtype)):
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
    return vec_add_bk


def part_a(iters, dev):
    print("=" * 100)
    print("PART A -- execution_backend host cost (minimal control kernel, 256x256 add)")
    print("=" * 100)
    a = torch.randn(256, 256, device=dev)
    b = torch.randn(256, 256, device=dev)
    res = {}
    for bk in ("cython", "ctypes", "dlpack"):
        try:
            t0 = time.perf_counter()
            f = make_control(bk)(256, 256, 128, 128)
            build_s = time.perf_counter() - t0
            out = f(a, b)
            npu_sync()
            ok = bool(torch.allclose(out.cpu(), (a + b).cpu(), atol=1e-2, rtol=1e-2))
            r = timeit(lambda: f(a, b), iters)
            r.update(backend=bk, build_s=round(build_s, 2), correct=ok,
                     adapter=type(f.adapter).__name__)
            print(f"  {bk:<8} adapter={r['adapter']:<22} per-launch host = "
                  f"{r['med_us']:8.2f} us  [min {r['min_us']:.2f} max {r['max_us']:.2f}]  "
                  f"correct={ok}  build={build_s:.1f}s")
            res[bk] = r
        except Exception as e:
            print(f"  {bk:<8} FAILED: {type(e).__name__}: {str(e)[:220]}")
            res[bk] = dict(backend=bk, error=f"{type(e).__name__}: {str(e)[:400]}")
    base = res.get("cython", {}).get("med_us")
    for bk, r in res.items():
        if base and "med_us" in r:
            r["vs_cython_ratio"] = round(r["med_us"] / base, 4)
            r["vs_cython_saving_us"] = round(base - r["med_us"], 3)
    return res


# --------------------------------------------- Part B: framework ceiling -----------
def forward_prep(ad, inputs, sink, use_analyzer=True, cache_stream=None):
    """Replica of cython_wrapper.forward prep, with switchable Analyzer / stream handling."""
    import ctypes as _ct
    result_idx, workspace_idx, auto_gm_idx, params = (
        ad.result_idx, ad.workspace_idx, ad.auto_gm_idx, ad.params)
    stream = cache_stream if cache_stream is not None else torch.npu.current_stream().npu_stream
    if use_analyzer:
        from tvm.arith import Analyzer
        analyzer = Analyzer()
    tensor_list, ins_idx = [], 0
    for i in range(len(params)):
        if i in result_idx or i in workspace_idx or i in auto_gm_idx:
            p = params[i]
            device = inputs[0].device if inputs else torch.npu.current_device()
            tensor_list.append(torch.empty(*[int(s) for s in p.shape], dtype=p.dtype, device=device))
        else:
            tensor_list.append(inputs[ins_idx]); ins_idx += 1
    sink.extend(tensor_list)
    call_args = []
    for i, t_ in enumerate(tensor_list):
        if isinstance(t_, torch.Tensor):
            if not t_.is_contiguous():
                raise ValueError("non-contiguous")
            call_args.append(_ct.c_void_p(t_.data_ptr()))
    for _p, (bi, dv) in ad.buffer_device_map.items():
        if isinstance(tensor_list[bi], torch.Tensor):
            td = tensor_list[bi].device
            if tensor_list[0].device.type != dv.type or (td.index is not None
                                                         and dv.index is not None
                                                         and td.index != dv.index):
                raise ValueError("dev")
    for _p, (bi, dt) in ad.buffer_dtype_map.items():
        if isinstance(tensor_list[bi], torch.Tensor) and tensor_list[bi].dtype != dt:
            raise ValueError("dtype")
    call_args.append(_ct.c_void_p(stream))
    return call_args


def part_b(iters, dev):
    print("\n" + "=" * 100)
    print("PART B -- framework ceiling: what a tilelang-side fix would buy per launch")
    print("  (replica prep, current = Analyzer+stream+allocs+loops; variants remove terms)")
    print("=" * 100)
    B, V, K_MAX, BV = 2, 256, 256, 4096
    logits = torch.randn((B, V), dtype=torch.float32, device=dev)
    ks = torch.full((B,), K_MAX, dtype=torch.int32, device=dev)
    ps = torch.ones((B,), dtype=torch.float32, device=dev)
    ms = torch.zeros((B,), dtype=torch.float32, device=dev)
    targets = {"K1(4 allocs)": R4M.r4_topk_stats(B, V, K_MAX, BV, False),
               "K1A(0 allocs)": R4M.r4_topk_stats_splitA(B, V, K_MAX, BV, 1, 1, -1)}
    args = {"K1(4 allocs)": (logits, ks, ps, ms),
            "K1A(0 allocs)": (logits, torch.empty((B, K_MAX), dtype=torch.float32, device=dev))}
    sink = []
    cached_stream = torch.npu.current_stream().npu_stream
    res = {}
    print(f"  {'kernel':<16} {'variant':<34} {'prep med us':>12} {'saving vs current':>19}")
    for name, jk in targets.items():
        ad = jk.adapter
        ins = list(args[name])
        variants = {
            "current (Analyzer + stream lookup)": lambda: forward_prep(ad, ins, sink, True, None),
            "lazy Analyzer (skip when dyn map empty)": lambda: forward_prep(ad, ins, sink, False, None),
            "cached stream only": lambda: forward_prep(ad, ins, sink, True, cached_stream),
            "lazy Analyzer + cached stream": lambda: forward_prep(ad, ins, sink, False, cached_stream),
        }
        cur = None
        rec = {}
        for vn, fn in variants.items():
            r = timeit(fn, iters, warm=100, sync_each=False)
            if cur is None:
                cur = r["med_us"]
            saving = cur - r["med_us"]
            rec[vn] = dict(**r, saving_vs_current_us=round(saving, 3),
                           saving_pct=round(100 * saving / cur, 2) if cur else 0)
            print(f"  {name:<16} {vn:<34} {r['med_us']:>12.2f} {saving:>+18.2f}")
            if len(sink) > 5000:
                sink.clear()
        # the real forward, for reference
        cw = ad.cython_wrapper
        rr = timeit(lambda: cw.forward(ins), iters, warm=30)
        rec["_real_forward_us"] = rr["med_us"]
        rec["_implied_libcall_us"] = round(rr["med_us"] - cur, 3)
        print(f"  {name:<16} {'[real cw.forward, for reference]':<34} {rr['med_us']:>12.2f} "
              f"-> implied lib.call {rr['med_us']-cur:+.2f}")
        res[name] = rec
    return res


# ------------------------------------------- Part C: wrapper-level savings ---------
def part_c(iters, dev):
    print("\n" + "=" * 100)
    print("PART C -- wrapper-level (OUR code) savings, per fn() call")
    print("=" * 100)
    res = {}
    print(f"  {'case':>5} {'item':<38} {'as-is us':>10} {'cached us':>10} {'saving us':>10}")
    for ci, (B, V, K_MAX, BV, nmp) in SHAPES.items():
        use_split, S, NBS, S_TAIL = R4M._r4_split_plan(B, V, K_MAX, BV)
        logits = torch.randn((B, V), dtype=torch.float32, device=dev)
        memo = {}
        r_plan = timeit(lambda: R4M._r4_split_plan(B, V, K_MAX, BV), iters * 4, warm=200, sync_each=False)

        def cached_plan():
            k = (B, V, K_MAX, BV)
            if k in memo:
                return memo[k]
            memo[k] = R4M._r4_split_plan(B, V, K_MAX, BV)
            return memo[k]
        r_plan_c = timeit(cached_plan, iters * 4, warm=200, sync_each=False)

        ws_shape = (B * max(S, 1), K_MAX)
        r_ws = timeit(lambda: torch.empty(ws_shape, dtype=torch.float32, device=dev),
                      iters * 4, warm=200, sync_each=False)
        held = torch.empty(ws_shape, dtype=torch.float32, device=dev)
        r_ws_c = timeit(lambda: held, iters * 4, warm=200, sync_each=False)

        def prechecks():
            _ = (B == 0); _ = (V < K_MAX); t = V % BV
            _ = (t != 0 and t < K_MAX); _ = (logits.dtype != torch.float32)
        r_pre = timeit(prechecks, iters * 4, warm=200, sync_each=False)
        flag = {(B, V, logits.dtype): True}
        r_pre_c = timeit(lambda: flag.get((B, V, logits.dtype)), iters * 4, warm=200, sync_each=False)

        items = [("_r4_split_plan -> memoised", r_plan, r_plan_c),
                 ("GM workspace torch.empty -> reused", r_ws, r_ws_c),
                 ("pre-checks -> shape/dtype memoised", r_pre, r_pre_c)]
        rec = dict(B=B, V=V, use_split=use_split, S=S, items={})
        tot = 0.0
        for nm, a_, b_ in items:
            sv = a_["med_us"] - b_["med_us"]
            # the workspace alloc only happens on the split path
            applies = (nm.startswith("GM workspace") and use_split) or not nm.startswith("GM workspace")
            eff = sv if applies else 0.0
            tot += eff
            rec["items"][nm] = dict(as_is_us=a_["med_us"], cached_us=b_["med_us"],
                                    saving_us=round(sv, 3), applies_to_this_case=applies,
                                    effective_saving_us=round(eff, 3))
            print(f"  {ci:>5} {nm:<38} {a_['med_us']:>10.2f} {b_['med_us']:>10.2f} "
                  f"{eff:>+10.2f}{'' if applies else '  (n/a: not split path)'}")
        rec["total_saving_us"] = round(tot, 3)
        res[ci] = rec
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out", default="probe5_mitigations.json")
    a = ap.parse_args()
    dev = torch.device("npu")
    fp = env_fingerprint()
    fp["delivered_kernel_sha256"] = hashlib.sha256(
        open(os.path.join(R4, "r4_fused_sampling.py"), "rb").read()).hexdigest()
    print("env:", json.dumps(fp, indent=1))
    res = {"_meta": dict(
        description="quantified evaluation of candidate mitigations (NOT implemented)",
        measurement="perf_counter_ns, host-side, no profiler unless stated", iters=a.iters, env=fp)}
    res["A_execution_backend"] = part_a(min(a.iters, 100), dev)
    res["B_framework_ceiling"] = part_b(a.iters, dev)
    res["C_wrapper_savings"] = part_c(a.iters, dev)

    # ------------------------------------------------------------- summary table ---
    print("\n" + "=" * 112)
    print("MITIGATION SUMMARY (expected saving per fn() call, host side, bare/L0 conditions)")
    print("=" * 112)
    p1p = os.path.join(HERE, "probe1_host_segments.json")
    p4p = os.path.join(HERE, "probe4_aclgraph.json")
    p1 = json.load(open(p1p)) if os.path.exists(p1p) else None
    p4 = json.load(open(p4p)) if os.path.exists(p4p) else None
    rows = []
    for ci in SHAPES:
        if not p1:
            continue
        v = p1["fused"]["none"].get(str(ci)) or p1["fused"]["none"].get(ci)
        fn_host = v["derived"]["host_async_us"]
        nl = v["n_launch"]
        wrap_sv = res["C_wrapper_savings"].get(ci, {}).get("total_saving_us", 0.0)
        cap = None
        if p4:
            w = (p4.get("wrapper_capture", {}).get(str(ci))
                 or p4.get("wrapper_capture", {}).get(ci))
            if w and w.get("capture_ok"):
                cap = w["L0_bare"]["replay_host_async"]["med_us"]
        rows.append(dict(case=ci, n_launch=nl, eager_fn_host_us=fn_host,
                         C_wrapper_saving_us=wrap_sv,
                         C_after_us=round(fn_host - wrap_sv, 3),
                         C_pct=round(100 * wrap_sv / fn_host, 2),
                         aclgraph_replay_host_us=cap,
                         aclgraph_pct_of_eager=round(100 * cap / fn_host, 2) if cap else None,
                         aclgraph_saving_us=round(fn_host - cap, 3) if cap else None))
    print(f"{'case':>5} {'#ln':>4} {'eager fn_host':>14} {'C: wrapper cache':>17} {'after C':>10} "
          f"{'C %':>6} {'D: aclgraph replay':>19} {'D %of eager':>12} {'D saving':>10}")
    for r in rows:
        print(f"{r['case']:>5} {r['n_launch']:>4} {r['eager_fn_host_us']:>14.2f} "
              f"{r['C_wrapper_saving_us']:>17.2f} {r['C_after_us']:>10.2f} {r['C_pct']:>5.1f}% "
              f"{str(r['aclgraph_replay_host_us']):>19} {str(r['aclgraph_pct_of_eager']):>12} "
              f"{str(r['aclgraph_saving_us']):>10}")
    res["summary"] = rows
    bk = res["A_execution_backend"]
    if "cython" in bk and "med_us" in bk["cython"]:
        print("\n  Part A (execution_backend, minimal control):")
        for k, v in bk.items():
            if "med_us" in v:
                print(f"    {k:<8} {v['med_us']:8.2f} us/launch   vs cython "
                      f"{v.get('vs_cython_ratio','-')}")
            else:
                print(f"    {k:<8} unavailable: {v.get('error','?')[:110]}")
    print("\n  Part B (framework ceiling, per launch, prep only):")
    for k, v in res["B_framework_ceiling"].items():
        cur = v["current (Analyzer + stream lookup)"]["med_us"]
        opt = v["lazy Analyzer + cached stream"]["med_us"]
        print(f"    {k:<16} prep {cur:7.2f} -> {opt:7.2f} us  (saving {cur-opt:+.2f} us, "
              f"{100*(cur-opt)/cur:.1f}%)  implied lib.call {v['_implied_libcall_us']:.2f} us")
    dump(a.out, res)


if __name__ == "__main__":
    main()
