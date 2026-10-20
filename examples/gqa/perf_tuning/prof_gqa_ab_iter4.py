"""Iter4 three-way same-session A/B probe: BASE (shipped flat) / C1 (ns4) / C3 (g2first).

Single-shot run for msprof op collection (mirrors prof_gqa_ab_iter3.py:
EXACTLY ONE main_kernel launch, no warm-up, no golden compare).

--arm ns4|g2first monkey-patches gqa._gqa_kernel_prefill_g2stack_flat with the
variant factory from perf_tuning/gqa_flat_{ns4,g2first}_variant.py BEFORE the
dispatch runs.  The patch is instrumented: the probe asserts (a) the variant
factory was actually invoked and (b) the resolved spec is still
"g2stack_flat" — so a silent compile-fallback to g2stack can never be
mis-measured as the arm.  Host-side pipeline (transpose_qkv, mask build,
transpose_out) is IDENTICAL between arms; gqa.py / test_gqa.py are NOT
modified.

Usage (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:$LD_LIBRARY_PATH
    msprof op --kernel-name=main_kernel --aic-metrics=PipeUtilization \
        --output=<dir> python3 examples/gqa/perf_tuning/prof_gqa_ab_iter4.py \
        --arm ns4 --shape r3
"""

import argparse
import importlib
import os
import sys

import torch
import torch_npu  # noqa: F401  (registers the npu backend)

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_DIR = os.path.dirname(HERE)
sys.path.insert(0, GQA_DIR)
sys.path.insert(0, HERE)

import gqa as G  # noqa: E402

SHAPES = {
    # name: (B, S, S_kv, N_q, N_kv, D, causal, dtype)
    "r2": (2, 128, 128, 32, 8, 128, True, "fp16"),    # bench c1-like, mc=4/3
    "r3": (2, 128, 128, 128, 8, 128, True, "bf16"),   # bench c3-like, mc=13/12
    "r6": (2, 128, 128, 48, 4, 128, False, "fp16"),   # bench c2-like, mc=5/4
}

VARIANT_MODULES = {"ns4": "gqa_flat_ns4_variant", "g2first": "gqa_flat_g2first_variant"}


def main():
    ap = argparse.ArgumentParser(description="Single-shot GQA three-way A/B run for msprof (iter4)")
    ap.add_argument("--arm", required=True, choices=["base", "ns4", "g2first"])
    ap.add_argument("--shape", required=True, choices=list(SHAPES))
    args = ap.parse_args()

    B, S, S_kv, N_q, N_kv, D, causal, dtag = SHAPES[args.shape]
    dtype = torch.float16 if dtag == "fp16" else torch.bfloat16

    invoked = []
    if args.arm != "base":
        mod = importlib.import_module(VARIANT_MODULES[args.arm])
        variant_factory = mod._gqa_kernel_prefill_g2stack_flat

        def patched(*a, **k):
            invoked.append(True)
            return variant_factory(*a, **k)

        G._gqa_kernel_prefill_g2stack_flat = patched

    torch.manual_seed(42)
    q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
    k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
    v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()

    print(
        f"[PROF_RUN] arm={args.arm} shape={args.shape} B={B} S={S} S_kv={S_kv} N_q={N_q} "
        f"N_kv={N_kv} D={D} causal={causal} dtype={dtag}",
        flush=True,
    )

    # The single profiled launch: transpose_qkv -> main_kernel -> transpose_out.
    out = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
    torch.npu.synchronize()

    chosen = [val[2] for val in G._KERNEL_BUILD_CACHE.values()]
    if args.arm != "base":
        assert invoked, f"[AB-PROBE] arm={args.arm}: variant factory was NEVER invoked"
        assert chosen == ["g2stack_flat"], (
            f"[AB-PROBE] arm={args.arm}: dispatch resolved to {chosen} — variant build "
            "failed and fell back; this sample is INVALID"
        )
    print(
        f"[PROF_DONE] arm={args.arm} chosen={chosen} variant_invoked={bool(invoked)} "
        f"out.shape={tuple(out.shape)} out.dtype={out.dtype}",
        flush=True,
    )


if __name__ == "__main__":
    main()
