"""Diagnose: which call site trips the contiguous() guard on pad_S>0 shapes."""

import os
import sys
import traceback

import torch
import torch_npu  # noqa: F401

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "examples", "gqa"))

from gqa import gqa  # noqa: E402

_orig_contig = torch.Tensor.contiguous


def diag_contiguous(self, *args, **kwargs):
    if str(self.device).startswith("npu") and not self.is_contiguous():
        print("=== .contiguous() on non-contiguous NPU tensor "
              f"shape={tuple(self.shape)} stride={self.stride()} ===")
        traceback.print_stack()
    return _orig_contig(self, *args, **kwargs)


torch.Tensor.contiguous = diag_contiguous

torch.manual_seed(11)
B, S, S_kv, N_q, N_kv, D = 2, 124, 252, 32, 8, 128
q = (torch.rand(B, S, N_q, D) * 2 - 1).half().npu()
k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).half().npu()
v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).half().npu()
out = gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=False)
print("output:", tuple(out.shape), out.dtype, "contiguous:", out.is_contiguous())
