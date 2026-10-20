"""F6: performance spot-check — case 8 & case 10, baseline vs fixed.

Task spec: main_kernel time on
  - case 8:  B4 S1024 Skv1024 Nq32  Nkv8 D128 causal fp16
  - case 10: B4 S512  Skv512  Nq128 Nkv8 D128 causal bf16
must not regress more than 3% after the eval fixes ([F4] adds one
T.tile.max per KV iter + one T.tile.add at normalize in the g2/g2stack
kernels; [F3] adds a dead [1, R] ub_row alloc in the aligned transpose
path).

Two measurements per case:
  a) kernel-only: the factory _run(...) call (BHSD inputs, aligned -> no
     padding; includes the mask hand-off but no host transposes)
  b) end-to-end: the cann-bench gqa(query=...) entry (transposes included)

Usage:
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python examples/gqa/eval_fix_verify/f6_perf_spotcheck.py
"""

import importlib.util
import os
import sys
import time

import torch
import torch_npu  # noqa: F401

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "examples", "gqa"))

CASES = [
    ("case8-fp16", 4, 1024, 1024, 32, 8, 128, True, torch.float16),
    ("case10-bf16", 4, 512, 512, 128, 8, 128, True, torch.bfloat16),
]
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
                              "gqa_impl_eval_fix_attempt1.py"), "gqa_base")
    fixed = _load(os.path.join(_REPO, "examples", "gqa", "gqa.py"), "gqa_fixed")

    print(f"{'case':<12} {'metric':<10} {'baseline us':>12} {'fixed us':>12} "
          f"{'ratio':>7}")
    regressions = []
    for (tag, B, S, S_kv, N_q, N_kv, D, causal, dtype) in CASES:
        torch.manual_seed(7)
        q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
        k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
        v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
        scale = -1.0

        # --- (a) kernel-only via the factory interface ---
        res = {}
        for name, mod in (("base", base), ("fixed", fixed)):
            run = mod.gqa(B=B, S=S, S_kv=S_kv, sm_scale=scale,
                          is_causal=causal, N_q=N_q, N_kv=N_kv, D=D,
                          dtype="float16" if dtype == torch.float16
                          else "bfloat16")
            # BHSD inputs + a real causal mask (the aligned causal contract)
            qh = q.transpose(1, 2).contiguous()
            kh = k.transpose(1, 2).contiguous()
            vh = v.transpose(1, 2).contiguous()
            mask = mod._build_causal_mask(B, S, S_kv, q.device)
            res[name] = bench_fn(run, (qh, kh, vh, mask))
        ratio = res["fixed"] / res["base"]
        flag = " <-- REGRESSION" if ratio > 1.03 else ""
        print(f"{tag:<12} {'kernel':<10} {res['base']:>12.1f} "
              f"{res['fixed']:>12.1f} {ratio:>7.3f}{flag}")
        if ratio > 1.03:
            regressions.append((tag, "kernel", ratio))

        # --- (b) end-to-end cann-bench entry ---
        res = {}
        for name, mod in (("base", base), ("fixed", fixed)):
            kwargs = dict(query=q, key=k, value=v, scaleValue=scale,
                          is_causal=causal)
            res[name] = bench_fn(mod.gqa, (), kwargs)
        ratio = res["fixed"] / res["base"]
        flag = " <-- REGRESSION" if ratio > 1.03 else ""
        print(f"{tag:<12} {'e2e':<10} {res['base']:>12.1f} "
              f"{res['fixed']:>12.1f} {ratio:>7.3f}{flag}")
        if ratio > 1.03:
            regressions.append((tag, "e2e", ratio))

    if regressions:
        print(f"\nPERF REGRESSIONS (>3%): {regressions}")
        sys.exit(1)
    print("\n[PERF PASS] no >3% regression on case8/case10")


if __name__ == "__main__":
    main()
