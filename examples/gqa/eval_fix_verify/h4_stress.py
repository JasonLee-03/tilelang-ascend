#!/usr/bin/env python3
"""[H4 / eval 20260910_173445 cases 84-100 — SIGSEGV cascade] crash stress.

Forces the dispatch compile-fallback chain into its DEGRADED forms (per-head
BM32 / BM16 via monkey-patched primary factories, decode AUTO dense via a
forced dense_expert failure — same technique as g3c_fallback_selftest) and
hammers each x {non-aligned + special-value} shape 100x, checking:

  (a) no exception / no crash (the round-3 rc=-11 signature),
  (b) iteration-0 numeric correctness vs the fp64 golden (NaN-position
      match included),
  (c) bit-stability across all 100 repeats (the G4 race signature was
      varying partial-NaN outputs on identical inputs).

Results are reported honestly (PASS/FAIL per combo).

Usage:
  unset PYTHONPATH; export ACL_OP_INIT_MODE=1
  python examples/gqa/eval_fix_verify/h4_stress.py
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BENCH = os.path.join(REPO, ".agents", "skills", "cann-bench")
sys.path.insert(0, os.path.join(REPO, "examples", "gqa"))
sys.path.insert(0, os.path.join(BENCH, "src"))

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
from kernel_eval.data.data_generator import DataGenerator  # noqa: E402
import gqa as g  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "gqa_golden", os.path.join(BENCH, "tasks", "level4", "gqa", "golden.py"))
gqa_golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gqa_golden)

NPU = "npu:0"
REPEATS = 100

# YAML-parsed eval semantics: cases.yaml spells [.nan, .nan] / [-.inf, .inf]
# which the YAML loader delivers as FLOAT nan/-inf/inf (not strings)
VR = {
    "fullrange": [-65504, 65504],
    "nan": [float("nan"), float("nan")],
    "inf": [float("-inf"), float("inf")],
}

# (degrade-level, tag, B, S, Skv, Nq, Nkv, D, causal, mode)
COMBOS = [
    ("bm32", "bm32-D128-fullrange", 2, 124, 252, 32, 8, 128, True, "fullrange"),
    ("bm32", "bm32-D128-nan", 2, 124, 252, 32, 8, 128, False, "nan"),
    ("bm32", "bm32-D128-inf", 2, 124, 252, 32, 8, 128, True, "inf"),
    ("bm32", "bm32-D256-fullrange", 2, 132, 252, 16, 8, 256, True, "fullrange"),
    ("bm16", "bm16-D128-fullrange", 2, 124, 252, 32, 8, 128, True, "fullrange"),
    ("bm16", "bm16-D128-nan", 2, 124, 252, 32, 8, 128, False, "nan"),
    ("bm16", "bm16-D512-fullrange", 2, 124, 252, 8, 4, 512, False, "fullrange"),
    ("dense_auto", "dense-auto-decode-fullrange", 60, 2, 2044, 32, 8, 128,
     True, "fullrange"),
    ("dense_auto", "dense-auto-decode-nan", 60, 2, 2044, 32, 8, 128,
     False, "nan"),
]


def gen(shape, dtype_str, mode, seed):
    dg = DataGenerator(input_dist="uniform")
    gen_ = torch.Generator().manual_seed(seed)
    return dg.generate_input_tensor(
        list(shape), dtype_str, VR[mode], generator=gen_)


def degrade(level):
    """Patch factories so the chain degrades; returns the saved originals."""
    saved = {
        "g2stack": g._gqa_kernel_prefill_g2stack,
        "prefill_expert": g._gqa_kernel_prefill_expert,
        "per_head": g._gqa_kernel_expert,
        "dense_expert": g._gqa_kernel_dense_expert,
        "dense_expert_d256": g._gqa_kernel_dense_expert_d256,
    }
    real_per_head = saved["per_head"]

    def boom(name):
        def _boom(*a, **kw):
            raise RuntimeError(f"FORCED COMPILE FAILURE: {name}")
        return _boom

    def per_head_gate(block_ms):
        def _gate(B, sp, skvp, sm, c, **kw):
            if kw.get("block_M", 32) in block_ms:
                raise RuntimeError(
                    f"FORCED COMPILE FAILURE: per_head bm{kw.get('block_M')}")
            return real_per_head(B, sp, skvp, sm, c, **kw)
        return _gate

    if level in ("bm32", "bm16"):
        g._gqa_kernel_prefill_g2stack = boom("g2stack")
        g._gqa_kernel_prefill_expert = boom("prefill_expert")
        g._gqa_kernel_expert = per_head_gate({64} if level == "bm32"
                                             else {64, 32})
    elif level == "dense_auto":
        g._gqa_kernel_dense_expert = boom("dense_expert")
        g._gqa_kernel_dense_expert_d256 = boom("dense_expert_d256")
    g._KERNEL_BUILD_CACHE.clear()
    g._COMPILED_CACHE.clear()
    return saved


def restore(saved):
    g._gqa_kernel_prefill_g2stack = saved["g2stack"]
    g._gqa_kernel_prefill_expert = saved["prefill_expert"]
    g._gqa_kernel_expert = saved["per_head"]
    g._gqa_kernel_dense_expert = saved["dense_expert"]
    g._gqa_kernel_dense_expert_d256 = saved["dense_expert_d256"]
    g._KERNEL_BUILD_CACHE.clear()
    g._COMPILED_CACHE.clear()


def check_numeric(out, ref, native, dtype_str, tag):
    """Iteration-0 correctness: NaN-position equality + tolerance."""
    nan_out = torch.isnan(out.float().cpu())
    nan_ref = torch.isnan(ref.float())
    if not bool(torch.all(nan_out == nan_ref)):
        print(f"[FAIL] {tag}: NaN-position mismatch "
              f"(out={int(nan_out.sum())} ref={int(nan_ref.sum())})", flush=True)
        return False
    of = out.double().cpu()
    rf = ref.double()
    viol = ((of - rf).abs() > 5e-3 + 5e-2 * rf.abs()).sum().item()
    if viol > 0:
        print(f"[FAIL] {tag}: {viol} tolerance violations", flush=True)
        return False
    return True


def run_combo(level, tag, B, S, Skv, Nq, Nkv, D, causal, mode):
    dtype_str = "float16"
    q = gen((B, S, Nq, D), dtype_str, mode, 31)
    k = gen((B, Skv, Nkv, D), dtype_str, mode, 32)
    v = gen((B, Skv, Nkv, D), dtype_str, mode, 33)
    ref = gqa_golden.gqa(q.double(), k.double(), v.double(), -1.0, causal)
    q_npu, k_npu, v_npu = q.to(NPU), k.to(NPU), v.to(NPU)

    saved = degrade(level)
    try:
        outs = []
        for it in range(REPEATS):
            try:
                out = g.gqa(query=q_npu, key=k_npu, value=v_npu,
                            scaleValue=-1.0, is_causal=causal)
                torch.npu.synchronize()
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL] {tag}: iter {it} raised "
                      f"{type(e).__name__}: {str(e)[-150:]}", flush=True)
                return False
            if it == 0:
                if out.shape != (B, S, Nq, D) or not out.is_contiguous():
                    print(f"[FAIL] {tag}: bad output shape/layout {out.shape}",
                          flush=True)
                    return False
                if not check_numeric(out, ref, None, dtype_str, tag):
                    return False
            outs.append(out.view(torch.int16).cpu().clone())
        base = outs[0]
        unstable = [i for i in range(1, REPEATS)
                    if not torch.equal(outs[i], base)]
        if unstable:
            print(f"[FAIL] {tag}: bit-instability at iters "
                  f"{unstable[:5]}(+{len(unstable) - 5})", flush=True)
            return False
        print(f"[PASS] {tag}: {REPEATS}x stable, numerics OK", flush=True)
        return True
    finally:
        restore(saved)


def main():
    results = []
    for combo in COMBOS:
        results.append(run_combo(*combo))
    print(f"\n=== H4 stress: {sum(results)}/{len(results)} PASS ===",
          flush=True)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
