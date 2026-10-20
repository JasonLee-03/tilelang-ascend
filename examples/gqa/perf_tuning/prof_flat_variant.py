"""Single-shot flat-variant run for msprof (B1 gate) — mirrors
prof_gqa_pipeline.py conventions: EXACTLY ONE main_kernel launch, no
warm-up loops, no golden comparison (every extra launch multiplies
profiling/simulation time).

The shipped-kernel counterpart measurement uses examples/gqa/prof_gqa_pipeline.py
(same shapes) — both report main_kernel Task Duration under msprof op.

Usage (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    msprof op --kernel-name=main_kernel --aic-metrics=PipeUtilization \
        --output=<dir> python3 examples/gqa/perf_tuning/prof_flat_variant.py \
        --batch 2 --nq 128 --nkv 8 --dtype bf16
"""

import argparse
import os
import sys

import torch
import torch_npu  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_DIR = os.path.dirname(HERE)
sys.path.insert(0, GQA_DIR)
sys.path.insert(0, HERE)

from gqa import _BIG_NEG, _dummy_mask, transpose_qkv  # noqa: E402
from gqa_flat_variant import build_flat_kernel  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Single-shot flat-variant run for msprof")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--seq-kv", type=int, default=128)
    ap.add_argument("--nq", type=int, default=32)
    ap.add_argument("--nkv", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--no-causal", action="store_true")
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
        f"[PROF_RUN] FLAT B={B} S={S} S_kv={S_kv} N_q={N_q} N_kv={N_kv} D={D} "
        f"causal={causal} dtype={args.dtype}",
        flush=True,
    )

    S_pad = ((S + 63) // 64) * 64
    Skv_pad = ((S_kv + 127) // 128) * 128
    qb, kb, vb = transpose_qkv(q, k, v, S_out=S_pad, S_kv_out=Skv_pad)
    assert qb is not None, "transpose_qkv fallback — unexpected on NPU"

    dtype_str = "float16" if dtype == torch.float16 else "bfloat16"
    sm_scale = 1.0 / (D**0.5)
    kernel_fn, _, _, use_mask = build_flat_kernel(
        B, S, S_kv, sm_scale, causal, N_q, N_kv, D, dtype_str
    )
    if use_mask and causal:
        mask = torch.full((S_pad, Skv_pad), _BIG_NEG, dtype=torch.float32)
        i = torch.arange(S).unsqueeze(-1)
        j = torch.arange(S_kv).unsqueeze(0)
        mask[:S, :S_kv][j <= (i + (S_kv - S))] = 0.0
        mask = mask.unsqueeze(0).expand(B, S_pad, Skv_pad).contiguous().to(q.device)
    elif use_mask:
        mask = torch.full((S_pad, Skv_pad), _BIG_NEG, dtype=torch.float32)
        mask[:, :S_kv] = 0.0
        mask = mask.unsqueeze(0).expand(B, S_pad, Skv_pad).contiguous().to(q.device)
    else:
        mask = _dummy_mask(q.device)

    # The single profiled launch (transpose_qkv above is filtered out by
    # --kernel-name=main_kernel; the flat kernel's prim_func is `main`).
    out = kernel_fn(qb, kb, vb, mask)
    torch.npu.synchronize()
    print(f"[PROF_DONE] out.shape={tuple(out.shape)} out.dtype={out.dtype}", flush=True)


if __name__ == "__main__":
    main()
