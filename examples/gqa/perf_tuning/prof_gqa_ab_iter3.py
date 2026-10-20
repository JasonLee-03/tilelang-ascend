"""Iter3 R6 same-session A/B probe: flat (shipped) vs forced g2stack fallback.

Single-shot run for msprof op collection (mirrors prof_gqa_pipeline.py:
EXACTLY ONE main_kernel launch, no warm-up, no golden compare).

--force g2stack monkey-patches gqa._gqa_kernel_prefill_g2stack_flat to raise
at BUILD time, so _resolve_kernel_chain (gqa.py:3621) falls through to the
next spec ("g2stack", gqa.py:4007) — the exact compile-fallback path the
shipped dispatch keeps for flat.  Host-side pipeline (transpose_qkv, mask
build, transpose_out) is IDENTICAL between arms; only the main kernel
choice differs.  gqa.py / test_gqa.py are NOT modified.

The chosen kernel name is printed from _KERNEL_BUILD_CACHE for audit.

Usage (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:$LD_LIBRARY_PATH
    msprof op --kernel-name=main_kernel --aic-metrics=PipeUtilization \
        --output=<dir> python3 examples/gqa/perf_tuning/prof_gqa_ab_iter3.py \
        --batch 2 --nq 48 --nkv 4 --no-causal [--force g2stack]
"""

import argparse
import os
import sys

import torch
import torch_npu  # noqa: F401  (registers the npu backend)

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_DIR = os.path.dirname(HERE)
sys.path.insert(0, GQA_DIR)

import gqa as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Single-shot GQA A/B run for msprof (iter3 R6)")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=128, help="query sequence length S")
    ap.add_argument("--seq-kv", type=int, default=128, help="key/value sequence length S_kv")
    ap.add_argument("--nq", type=int, default=48, help="query head count N_q")
    ap.add_argument("--nkv", type=int, default=4, help="kv head count N_kv")
    ap.add_argument("--dim", type=int, default=128, help="head dim D")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--no-causal", action="store_true", help="run non-causal (default: causal)")
    ap.add_argument(
        "--force",
        default="none",
        choices=["none", "g2stack"],
        help="g2stack: disable the flat kernel at build time so the dispatch "
        "falls through to the g2stack compile-fallback candidate",
    )
    args = ap.parse_args()

    if args.force == "g2stack":

        def _flat_disabled(*a, **k):
            raise RuntimeError("[AB-PROBE] flat kernel force-disabled; falling back to g2stack")

        G._gqa_kernel_prefill_g2stack_flat = _flat_disabled

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    causal = not args.no_causal
    B, S, S_kv = args.batch, args.seq, args.seq_kv
    N_q, N_kv, D = args.nq, args.nkv, args.dim

    torch.manual_seed(42)
    q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
    k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
    v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()

    print(
        f"[PROF_RUN] force={args.force} B={B} S={S} S_kv={S_kv} N_q={N_q} N_kv={N_kv} D={D} "
        f"causal={causal} dtype={args.dtype}",
        flush=True,
    )

    # The single profiled launch: transpose_qkv -> main_kernel -> transpose_out.
    out = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
    torch.npu.synchronize()

    chosen = [val[2] for val in G._KERNEL_BUILD_CACHE.values()]
    print(f"[PROF_DONE] chosen={chosen} out.shape={tuple(out.shape)} out.dtype={out.dtype}", flush=True)


if __name__ == "__main__":
    main()
