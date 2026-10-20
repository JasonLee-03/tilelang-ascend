#!/usr/bin/env python3
"""Isolate the segfault from g3_l0c_enum.py: per-head _gqa_kernel_expert,
D=256, non-causal, use_mask=False, kv_iters=2 -> use_pipeline=True
(Pipelined 2 stages, threads=1).  Config: B2 S252(->256) Skv256 Nq128 Nkv8.
Run: unset PYTHONPATH; export ACL_OP_INIT_MODE=1; python g3b_crash_iso.py [variant]
variants: pipe (default: S252/Skv256 noncausal) | causal (S252/Skv252 causal)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import torch  # noqa: E402
import gqa as gqa_mod  # noqa: E402

variant = sys.argv[1] if len(sys.argv) > 1 else "pipe"

if variant == "pipe":
    B, S, S_kv, N_q, N_kv, D, causal = 2, 252, 256, 128, 8, 256, False
else:
    B, S, S_kv, N_q, N_kv, D, causal = 2, 252, 252, 128, 8, 256, True

block_M, block_N = 64, 128
S_padded = ((S + block_M - 1) // block_M) * block_M
S_kv_padded = ((S_kv + block_N - 1) // block_N) * block_N
use_mask = bool(causal or (S_kv_padded != S_kv))
kv_iters = S_kv_padded // block_N
print(f"variant={variant} S_padded={S_padded} S_kv_padded={S_kv_padded} "
      f"use_mask={use_mask} kv_iters={kv_iters} D={D}", flush=True)

print("compiling kernel (per-head _gqa_kernel_expert)...", flush=True)
fn = gqa_mod._gqa_kernel_expert(
    B, S_padded, S_kv_padded, 1.0 / (D ** 0.5), causal,
    N_q=N_q, N_kv=N_kv, D=D, dtype="float16", block_M=block_M,
    skv_minus_s=S_kv - S, use_mask=use_mask, block_N=block_N,
)
print("compile OK — building tensors", flush=True)
q = torch.randn(B, N_q, S_padded, D, dtype=torch.float16).to("npu:0")
k = torch.randn(B, N_kv, S_kv_padded, D, dtype=torch.float16).to("npu:0")
v = torch.randn(B, N_kv, S_kv_padded, D, dtype=torch.float16).to("npu:0")
m = torch.zeros(B, S_padded, S_kv_padded, dtype=torch.float32).to("npu:0")
print("launching...", flush=True)
out = fn(q, k, v, m)
torch.npu.synchronize()
print(f"run OK — finite={bool(torch.isfinite(out.float()).all().item())}", flush=True)
