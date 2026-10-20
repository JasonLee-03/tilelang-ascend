#!/usr/bin/env python3
"""[H5 / attempt-3 step 6] performance spotcheck: case8/case10 main_kernel
vs the ATTEMPT-2 baseline (<3% regression gate) + one NON-ALIGNED case e2e
(H2 moved its padding + output-crop into tilelang kernels — old path did
4-6 torch d2d dispatches; new path should be equal-or-faster).

Cases:
  case8-fp16:  B4 S1024 Skv1024 Nq32  Nkv8 D128 causal fp16  (g2stack)
  case10-bf16: B4 S512  Skv512  Nq128 Nkv8 D128 causal bf16  (g2stack)
  nonalign:    B2 S124  Skv252  Nq32  Nkv8 D128 causal fp16  (e2e only —
               kernel-only through the factory would take the torch-pad
               fallback on BOTH sides; the bench interface is the metric)

Usage:
  unset PYTHONPATH; export ACL_OP_INIT_MODE=1
  python examples/gqa/eval_fix_verify/h5_perf_spotcheck.py
"""
import importlib.util
import os
import sys
import time

import torch
import torch_npu  # noqa: F401

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "examples", "gqa"))

WARMUP = 3
ITERS = 20


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bench_fn(fn, args=(), kwargs=None):
    kwargs = kwargs or {}
    for _ in range(WARMUP):
        fn(*args, **kwargs)
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        fn(*args, **kwargs)
    torch.npu.synchronize()
    return (time.time() - t0) / ITERS * 1e6  # us


def main():
    base = _load(os.path.join(_REPO, "examples", "gqa", "history_version",
                              "gqa_impl_eval_fix_attempt2.py"), "gqa_base")
    cur = _load(os.path.join(_REPO, "examples", "gqa", "gqa.py"), "gqa_cur")

    torch.manual_seed(7)
    results = []
    cases = [
        ("case8-fp16", 4, 1024, 1024, 32, 8, 128, True, torch.float16),
        ("case10-bf16", 4, 512, 512, 128, 8, 128, True, torch.bfloat16),
        ("nonalign-fp16", 2, 124, 252, 32, 8, 128, True, torch.float16),
    ]
    for tag, B, S, Skv, Nq, Nkv, D, causal, dt in cases:
        q = ((torch.rand(B, S, Nq, D) * 2 - 1) * 0.5).to(dt)
        k = ((torch.rand(B, Skv, Nkv, D) * 2 - 1) * 0.5).to(dt)
        v = ((torch.rand(B, Skv, Nkv, D) * 2 - 1) * 0.5).to(dt)
        q_n, k_n, v_n = q.npu(), k.npu(), v.npu()

        # main_kernel (factory _run, BHSD inputs — the eval's kernel metric)
        if tag != "nonalign-fp16":
            for name, mod in (("base", base), ("cur", cur)):
                fn = mod.gqa(B=B, S=S, S_kv=Skv, sm_scale=-1.0,
                             is_causal=causal, N_q=Nq, N_kv=Nkv, D=D,
                             dtype=str(dt).replace("torch.", ""))
                qh = q_n.transpose(1, 2).contiguous()
                kh = k_n.transpose(1, 2).contiguous()
                vh = v_n.transpose(1, 2).contiguous()
                mask = torch.zeros(B, S, Skv, dtype=torch.float32).npu()
                t = bench_fn(fn, (qh, kh, vh, mask))
                results.append((tag, f"kernel/{name}", t))

        # e2e (bench interface gqa(query=...))
        for name, mod in (("base", base), ("cur", cur)):
            t = bench_fn(mod.gqa, (), dict(query=q_n, key=k_n, value=v_n,
                                           scaleValue=-1.0, is_causal=causal))
            results.append((tag, f"e2e/{name}", t))

    print(f"{'case':<14} {'metric':<10} {'base us':>10} {'cur us':>10} {'ratio':>7}")
    by = {}
    for tag, metric, t in results:
        by.setdefault((tag, metric.split("/")[0]), {})[metric.split("/")[1]] = t
    regressions = []
    for (tag, kind), d in by.items():
        if "base" in d and "cur" in d:
            ratio = d["cur"] / d["base"]
            flag = ""
            if kind == "kernel" and ratio > 1.03:
                flag = " <-- REGRESSION"
                regressions.append((tag, kind, ratio))
            print(f"{tag:<14} {kind:<10} {d['base']:>10.1f} {d['cur']:>10.1f} "
                  f"{ratio:>7.3f}{flag}")
    if regressions:
        print(f"\nKERNEL REGRESSIONS >3%: {regressions}")
        sys.exit(1)
    print("\nNo main_kernel regression >3% (e2e is host-noise dominated, "
          "informational)")


if __name__ == "__main__":
    main()
