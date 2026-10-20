#!/usr/bin/env python3
"""Isolate base15: B60 S1 Skv1024 D256 decode (dense_expert_d256, W5 reorder)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import torch  # noqa: E402
import gqa as gqa_mod  # noqa: E402

NPU = "npu:0"
B, S, S_kv, N_q, N_kv, D = 60, 1, 1024, 32, 8, 256
torch.manual_seed(3)
q = (torch.rand(B, S, N_q, D) * 2 - 1).half().to(NPU)
k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).half().to(NPU)
v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).half().to(NPU)
print("inputs ready; calling gqa() (non-causal)...", flush=True)
out = gqa_mod.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=False)
torch.npu.synchronize()
print(f"non-causal OK finite={bool(torch.isfinite(out.float()).all())}", flush=True)

torch.manual_seed(4)
q = (torch.rand(B, S, N_q, D) * 2 - 1).half().to(NPU)
k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).half().to(NPU)
v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).half().to(NPU)
print("calling gqa() (causal, mask path)...", flush=True)
out = gqa_mod.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=True)
torch.npu.synchronize()
print(f"causal OK finite={bool(torch.isfinite(out.float()).all())}", flush=True)
