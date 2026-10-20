#!/usr/bin/env python3
"""Post-fix smoke test: dispatch fallback chain + G1/G2/G3 fixes.

Verifies:
  1. base shapes still route to their PRIMARY kernels (chain first-hit).
  2. D=512 prefill compiles via the L0C-budget ladder (BM=32).
  3. D=256 non-causal prefill compiles (per-head pipeline now serial).
  4. Non-aligned padding (G1 zero pool) works on fp16/bf16.
  5. Output correctness vs golden on all of the above.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import torch  # noqa: E402
import gqa as gqa_mod  # noqa: E402

BENCH = "/mnt/workspace/cann/20260910_gqa_V02/tilelang-ascend/.agents/skills/cann-bench"
sys.path.insert(0, os.path.join(BENCH, "src"))
from kernel_eval.utils.compare import compare_tensors  # noqa: E402

sys.path.insert(0, os.path.join(BENCH, "tasks", "level4", "gqa"))
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "gqa_golden", os.path.join(BENCH, "tasks", "level4", "gqa", "golden.py"))
gqa_golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gqa_golden)

NPU = "npu:0"


def check(tag, B, S, S_kv, N_q, N_kv, D, dtype, causal, vrange=(-1, 1)):
    q = (torch.rand(B, S, N_q, D, dtype=torch.float32) * 2 - 1)
    k = (torch.rand(B, S_kv, N_kv, D, dtype=torch.float32) * 2 - 1)
    v = (torch.rand(B, S_kv, N_kv, D, dtype=torch.float32) * 2 - 1)
    if vrange != (-1, 1):
        lo, hi = vrange
        q = q * ((hi - lo) / 2) + (hi + lo) / 2
        k = k * ((hi - lo) / 2) + (hi + lo) / 2
        v = v * ((hi - lo) / 2) + (hi + lo) / 2
    td = getattr(torch, dtype)
    q, k, v = q.to(td).to(NPU), k.to(td).to(NPU), v.to(td).to(NPU)

    out = gqa_mod.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
    torch.npu.synchronize()
    # eval-faithful: fp64 CPU golden + native (same-precision CPU) reference
    ref = gqa_golden.gqa(q.double().cpu(), k.double().cpu(), v.double().cpu(),
                         scaleValue=-1.0, is_causal=causal)
    native = gqa_golden.gqa(q.cpu(), k.cpu(), v.cpu(),
                            scaleValue=-1.0, is_causal=causal)

    res = compare_tensors(output=out.cpu(), golden=ref, dtype=dtype,
                          native_output=native)
    print(f"[{'PASS' if res.passed else 'FAIL'}] {tag} "
          f"(mere={res.mere:.2e} mare={res.mare:.2e} maxdiff={res.max_diff:.2e})",
          flush=True)
    return res.passed


def main():
    results = []
    # 1. base aligned shapes (primary kernels, incl. g2stack/prefill_expert/
    #    per-head/dense-expert coverage)
    results.append(check("base1 fp16 B2S128Nq32Nkv8 causal", 2, 128, 128, 32, 8, 128, "float16", True))
    results.append(check("base4 bf16 B4S256Nq64Nkv8 noncausal", 4, 256, 256, 64, 8, 128, "bfloat16", False))
    results.append(check("base6 bf16 B2S256Nq128Nkv8 D256 causal", 2, 256, 256, 128, 8, 256, "bfloat16", True))
    results.append(check("base11 fp16 B16S1Skv2048 decode", 16, 1, 2048, 32, 8, 128, "float16", False))
    results.append(check("base15 fp16 B60S1Skv1024 D256 decode", 60, 1, 1024, 32, 8, 256, "float16", False))
    results.append(check("base19 MHA bf16 B2S128Nq32Nkv32 causal", 2, 128, 128, 32, 32, 128, "bfloat16", True))
    # 2. G3: D=512 prefill (L0C ladder -> BM=32) + D=256 non-causal (serial)
    results.append(check("D512 fp16 B2S256Nq32Nkv8 causal", 2, 256, 256, 32, 8, 512, "float16", True))
    results.append(check("D512 fp16 B2S256Nq32Nkv8 noncausal", 2, 256, 256, 32, 8, 512, "float16", False))
    results.append(check("D256 fp16 B2S252Skv256 Nq128 noncausal (was UB-overflow)",
                         2, 252, 256, 128, 8, 256, "float16", False))
    # 3. G1: non-aligned padding both dtypes
    results.append(check("nonaligned fp16 B2S124Skv124Nq32 causal", 2, 124, 124, 32, 8, 128, "float16", True))
    results.append(check("nonaligned bf16 B2S127Skv125Nq48 noncausal", 2, 127, 125, 48, 8, 128, "bfloat16", False))
    results.append(check("nonaligned decode fp16 B16S1Skv2044", 16, 1, 2044, 32, 8, 128, "float16", False))
    # 4. G2: full-range causal fp16 (the case-67 class)
    results.append(check("fullrange fp16 B2S128Nq32Nkv8 causal", 2, 128, 128, 32, 8, 128,
                         "float16", True, vrange=(-65504, 65504)))
    results.append(check("fullrange bf16 B2S128Nq32Nkv8 causal [-65504,65504]",
                         2, 128, 128, 32, 8, 128, "bfloat16", True, vrange=(-65504, 65504)))
    print(f"\n=== {sum(results)}/{len(results)} PASS ===", flush=True)


if __name__ == "__main__":
    main()
