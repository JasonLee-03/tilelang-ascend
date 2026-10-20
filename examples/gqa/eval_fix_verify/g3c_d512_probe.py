#!/usr/bin/env python3
"""Test per-head _gqa_kernel_expert compile for D=512 (acc_o_l0c [2,64,512]
f32 == 262144 == the eval gqa_56 'required' number) and D=256 fallback
variants (BM=32; serial non-pipelined)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import torch  # noqa: E402
import gqa as gqa_mod  # noqa: E402


def probe(tag, B, S, S_kv, N_q, N_kv, D, dtype, causal, block_M):
    block_N = 128
    S_padded = ((S + block_M - 1) // block_M) * block_M
    S_kv_padded = ((S_kv + block_N - 1) // block_N) * block_N
    use_mask = bool(causal or (S_kv_padded != S_kv))
    try:
        fn = gqa_mod._gqa_kernel_expert(
            B, S_padded, S_kv_padded, 1.0 / (D ** 0.5), causal,
            N_q=N_q, N_kv=N_kv, D=D, dtype=dtype, block_M=block_M,
            skv_minus_s=S_kv - S, use_mask=use_mask, block_N=block_N,
        )
        q = torch.randn(B, N_q, S_padded, D, dtype=getattr(torch, dtype)).to("npu:0")
        k = torch.randn(B, N_kv, S_kv_padded, D, dtype=getattr(torch, dtype)).to("npu:0")
        v = torch.randn(B, N_kv, S_kv_padded, D, dtype=getattr(torch, dtype)).to("npu:0")
        m = torch.zeros(B, S_padded, S_kv_padded, dtype=torch.float32).to("npu:0")
        out = fn(q, k, v, m)
        torch.npu.synchronize()
        print(f"[OK ] {tag}", flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {tag}: {type(e).__name__}: {str(e)[-300:]}", flush=True)
        return False


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "d512"):
        probe("D512 BM64 causal    B2 S256 Skv256 Nq32", 2, 256, 256, 32, 8, 512, "float16", True, 64)
    if which in ("all", "d51232"):
        probe("D512 BM32 causal    B2 S256 Skv256 Nq32", 2, 256, 256, 32, 8, 512, "float16", True, 32)
    if which in ("all", "d256nc"):
        # the local UB-exhaustion config, with BM=32 + (patched) no-pipeline
        probe("D256 BM32 noncausal B2 S252 Skv256 Nq128", 2, 252, 256, 128, 8, 256, "float16", False, 32)
    if which in ("all", "d256c"):
        probe("D256 BM32 causal    B2 S252 Skv252 Nq128", 2, 252, 252, 128, 8, 256, "float16", True, 32)
