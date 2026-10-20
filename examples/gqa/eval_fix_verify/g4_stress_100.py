#!/usr/bin/env python3
"""[G4 / eval cases 84-100 — rc=-11 cascade] Repeat-100 stress test.

Case 84 (the suspected trigger, baseline ~364us — between S128 and S256
classes) started a 17-case SIGSEGV cascade in the eval.  This script
repeats non-aligned + special-value combinations of that magnitude
100x each on ONE device, checking for crashes, exceptions, and
output corruption (compare against the first iteration's output).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import torch  # noqa: E402
import torch_npu  # noqa: F401
import gqa as gqa_mod  # noqa: E402

DEVICE = os.environ.get("STRESS_DEVICE", "npu:0")
NPU = DEVICE
REPEAT = 100


def gen(shape, mode, seed):
    g = torch.Generator().manual_seed(seed)
    r = torch.rand(shape, dtype=torch.float64, generator=g)
    if mode == "fullrange":
        return (r * 131008.0 - 65504.0).clamp(-65504, 65504).half()
    if mode == "inf":
        t = (r * 2 - 1).half().float().flatten()
        n = max(1, len(t) // 20)
        t[:n] = float("-inf")
        t[-n:] = float("inf")
        return t.reshape(shape).half()
    if mode == "nan":
        t = (r * 2 - 1).half().float()
        m = torch.rand(shape, generator=g) < 0.5
        t[m] = float("nan")
        return t.half()
    return (r * 2 - 1).half()


def nanaware_equal(a, b):
    """Bitwise-stable comparison that treats NaN==NaN and ±0 distinctly."""
    return (torch.equal(a, b)
            or (torch.equal(torch.isnan(a), torch.isnan(b))
                and torch.equal(a[~torch.isnan(a)], b[~torch.isnan(b)])))


def stress(tag, B, S, S_kv, N_q, N_kv, D, causal, mode):
    q = gen((B, S, N_q, D), mode, 1).to(NPU)
    k = gen((B, S_kv, N_kv, D), mode, 2).to(NPU)
    v = gen((B, S_kv, N_kv, D), mode, 3).to(NPU)
    try:
        ref = gqa_mod.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
        torch.npu.synchronize()
        ref = ref.clone()
        for i in range(REPEAT):
            out = gqa_mod.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
            torch.npu.synchronize()
            if not nanaware_equal(out, ref):
                both_nan = torch.isnan(out.float()) & torch.isnan(ref.float())
                fin = ~both_nan
                maxdiff = float((out.float()[fin] - ref.float()[fin]).abs().max()) if fin.any() else 0.0
                print(f"[FAIL] {tag} iter {i}: output differs from iter 0 "
                      f"(maxdiff={maxdiff:.3e})", flush=True)
                return False
        print(f"[PASS] {tag}: {REPEAT} repeats bitwise-stable", flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {tag}: {type(e).__name__}: {str(e)[-200:]}", flush=True)
        return False


def main():
    results = []
    # ~364us-class non-aligned + special value combos (S128-S256 magnitude)
    results.append(stress("B2S124Skv124 fullrange causal", 2, 124, 124, 32, 8, 128, True, "fullrange"))
    results.append(stress("B2S252Skv252 fullrange causal D256", 2, 252, 252, 128, 8, 256, True, "fullrange"))
    results.append(stress("B2S124Skv124 inf noncausal", 2, 124, 124, 48, 4, 128, False, "inf"))
    results.append(stress("B2S252Skv252 nan causal D256 bf16-like", 2, 252, 252, 128, 8, 256, True, "nan"))
    results.append(stress("B60S2Skv2044 decode-causal fullrange", 60, 2, 2044, 32, 8, 128, True, "fullrange"))
    print(f"\n=== {sum(results)}/{len(results)} stress PASS ===", flush=True)


if __name__ == "__main__":
    main()
