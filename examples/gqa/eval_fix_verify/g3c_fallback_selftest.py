#!/usr/bin/env python3
"""[G3 / step 3c] Dispatch compile-fallback chain self-test.

Monkey-patches kernel factories to force compile-phase failures and checks
that the dispatch falls through to the next candidate with CORRECT output:

  1. Force _gqa_kernel_prefill_g2stack to raise -> expect prefill_expert.
  2. Force g2stack + prefill_expert to raise -> expect per-head ladder.
  3. Force g2stack + prefill_expert + per-head(BM=64) to raise -> expect
     per-head BM=32 (correct S_padded handling for the smaller block).
  4. Force dense_expert to raise -> expect AUTO dense on the decode path.

Verifies the chosen kernel produces output matching golden in every case.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import torch  # noqa: E402
import gqa as gqa_mod  # noqa: E402
import importlib.util  # noqa: E402

BENCH = "/mnt/workspace/cann/20260910_gqa_V02/tilelang-ascend/.agents/skills/cann-bench"
_spec = importlib.util.spec_from_file_location(
    "gqa_golden", os.path.join(BENCH, "tasks", "level4", "gqa", "golden.py"))
gqa_golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gqa_golden)

NPU = "npu:0"


def run_and_check(tag, B, S, S_kv, N_q, N_kv, D, causal):
    torch.manual_seed(7)
    td = torch.float16
    q = (torch.rand(B, S, N_q, D) * 2 - 1).to(td).to(NPU)
    k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(td).to(NPU)
    v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(td).to(NPU)
    try:
        out = gqa_mod.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
        torch.npu.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {tag}: {type(e).__name__}: {str(e)[-200:]}", flush=True)
        return False
    ref = gqa_golden.gqa(q.double().cpu(), k.double().cpu(), v.double().cpu(),
                         scaleValue=-1.0, is_causal=causal).float()
    of = out.float().cpu()
    viol = ((of - ref).abs() > 2e-3 + 1e-2 * ref.abs()).sum().item()
    ok = viol == 0 and not torch.isnan(of).any()
    print(f"[{'PASS' if ok else 'FAIL'}] {tag} (viol={viol})", flush=True)
    return ok


def force_fail(mod, name):
    """Patch factory `name` on module `mod` to always raise (compile-phase)."""
    def boom(*args, **kwargs):
        raise RuntimeError(f"FORCED COMPILE FAILURE: {name}")
    setattr(mod, name, boom)
    return name


def main():
    results = []

    # ---- 1. g2stack forced-fail -> prefill_expert (base4-class shape) ----
    saved = gqa_mod._gqa_kernel_prefill_g2stack
    force_fail(gqa_mod, "_gqa_kernel_prefill_g2stack")
    gqa_mod._KERNEL_BUILD_CACHE.clear()
    gqa_mod._COMPILED_CACHE.clear()
    results.append(run_and_check("fallback->prefill_expert (g2stack forced)",
                                 4, 256, 256, 64, 8, 128, False))
    gqa_mod._gqa_kernel_prefill_g2stack = saved

    # ---- 2. g2stack + prefill_expert forced -> per-head BM64 ----
    saved2 = gqa_mod._gqa_kernel_prefill_expert
    force_fail(gqa_mod, "_gqa_kernel_prefill_g2stack")
    force_fail(gqa_mod, "_gqa_kernel_prefill_expert")
    gqa_mod._KERNEL_BUILD_CACHE.clear()
    gqa_mod._COMPILED_CACHE.clear()
    results.append(run_and_check("fallback->per_head_bm64 (both experts forced)",
                                 4, 256, 256, 64, 8, 128, False))
    gqa_mod._gqa_kernel_prefill_g2stack = saved
    gqa_mod._gqa_kernel_prefill_expert = saved2

    # ---- 3. + per-head forced -> BM=32 with correct S_padded ----
    saved3 = gqa_mod._gqa_kernel_expert
    real_expert = saved3

    def boom_bm64(B, S_padded, S_kv_padded, sm_scale, is_causal, **kw):
        if kw.get("block_M", 32) == 64:
            raise RuntimeError("FORCED COMPILE FAILURE: per_head_bm64")
        return real_expert(B, S_padded, S_kv_padded, sm_scale, is_causal, **kw)

    force_fail(gqa_mod, "_gqa_kernel_prefill_g2stack")
    force_fail(gqa_mod, "_gqa_kernel_prefill_expert")
    gqa_mod._gqa_kernel_expert = boom_bm64
    gqa_mod._KERNEL_BUILD_CACHE.clear()
    gqa_mod._COMPILED_CACHE.clear()
    # S=100 -> S_padded@BM64=128 vs @BM32=128 (same); use S=70 -> 128 vs 96
    results.append(run_and_check("fallback->per_head_bm32 (S=70 row-pad changes)",
                                 2, 70, 128, 32, 8, 128, False))
    gqa_mod._gqa_kernel_prefill_g2stack = saved
    gqa_mod._gqa_kernel_prefill_expert = saved2
    gqa_mod._gqa_kernel_expert = saved3

    # ---- 4. decode: dense_expert forced -> AUTO dense ----
    saved4 = gqa_mod._gqa_kernel_dense_expert
    force_fail(gqa_mod, "_gqa_kernel_dense_expert")
    gqa_mod._KERNEL_BUILD_CACHE.clear()
    gqa_mod._COMPILED_CACHE.clear()
    results.append(run_and_check("fallback->dense_auto (dense_expert forced)",
                                 16, 1, 2048, 32, 8, 128, False))
    gqa_mod._gqa_kernel_dense_expert = saved4
    gqa_mod._KERNEL_BUILD_CACHE.clear()
    gqa_mod._COMPILED_CACHE.clear()

    print(f"\n=== {sum(results)}/{len(results)} PASS ===", flush=True)


if __name__ == "__main__":
    main()
