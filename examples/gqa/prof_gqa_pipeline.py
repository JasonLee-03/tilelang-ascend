"""Minimal single-shot GQA run for msprof pipeline-trace (流水图) collection.

Runs the public BSND API of examples/gqa/gqa.py EXACTLY ONCE so that the
msprof simulator (camodel) only has to simulate one main attention kernel
launch. Do NOT add warm-up loops or golden comparisons here — every extra
launch multiplies the (slow) simulation time.

Default shape: B=1 S=128 S_kv=128 N_q=8 N_kv=2 D=128 causal=True fp16
— structurally identical to L0 case c1 of test_gqa.py (G = N_q/N_kv = 4,
D=128, block-aligned prefill path, expert main_kernel), scaled down for a
feasible simulator turnaround. Override via CLI flags if needed.

Pipeline-trace collection (from the repo root):

    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    msprof op simulator --soc-version=Ascend910_9362 \
        --kernel-name=main_kernel \
        --output=examples/gqa/msprof_sim \
        python3 examples/gqa/prof_gqa_pipeline.py

Hardware performance collection (real NPU, op mode):

    msprof op --kernel-name=main_kernel \
        --output=examples/gqa/msprof_op \
        python3 examples/gqa/prof_gqa_pipeline.py

Sanity run without profiling:

    python3 examples/gqa/prof_gqa_pipeline.py

NOTE: same environment contract as test_gqa.py — the repo root must NOT be
on PYTHONPATH (the in-repo tilelang package is not built and would shadow
the installed site-packages tilelang used by the verified runs).
"""

import argparse
import os
import sys

import torch
import torch_npu  # noqa: F401  (registers the npu backend)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gqa import gqa  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Single-shot GQA run for msprof trace collection")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=128, help="query sequence length S")
    ap.add_argument("--seq-kv", type=int, default=128, help="key/value sequence length S_kv")
    ap.add_argument("--nq", type=int, default=8, help="query head count N_q")
    ap.add_argument("--nkv", type=int, default=2, help="kv head count N_kv")
    ap.add_argument("--dim", type=int, default=128, help="head dim D")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--no-causal", action="store_true", help="run non-causal (default: causal)")
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    causal = not args.no_causal
    B, S, S_kv = args.batch, args.seq, args.seq_kv
    N_q, N_kv, D = args.nq, args.nkv, args.dim

    torch.manual_seed(42)
    q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
    k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
    v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()

    print(
        f"[PROF_RUN] B={B} S={S} S_kv={S_kv} N_q={N_q} N_kv={N_kv} D={D} "
        f"causal={causal} dtype={args.dtype}",
        flush=True,
    )

    # The single profiled launch: transpose_qkv -> main_kernel -> transpose_out.
    out = gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
    torch.npu.synchronize()

    print(f"[PROF_DONE] out.shape={tuple(out.shape)} out.dtype={out.dtype}", flush=True)


if __name__ == "__main__":
    main()
