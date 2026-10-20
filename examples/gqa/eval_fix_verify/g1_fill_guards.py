"""[G1 / eval cann_final_eval_20260910_071341 cases 71/72/73/81/82] Fill
binary guards.

The eval SoC is missing the fp16/bf16 aclnnInplaceFillScalar (FillAiCore)
binary (error 561103); fp32 Fill IS available (causal Mask_full passed both
rounds).  This script monkey-patches torch.full / torch.zeros / torch.Tensor
.zero_ / torch.Tensor.fill_ to RAISE if they would dispatch a Fill/Zeros
kernel for fp16/bf16 on an NPU tensor, then runs the padding shapes of the
failure classes end-to-end through gqa() and checks the results vs golden.

fp32 fills remain allowed (Mask_full construction).

Usage:
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python examples/gqa/eval_fix_verify/g1_fill_guards.py
"""

import contextlib
import os
import sys

import torch
import torch_npu  # noqa: F401

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "examples", "gqa"))
sys.path.insert(0, os.path.join(_REPO, ".agents", "skills", "cann-bench",
                                "tasks", "level4", "gqa"))

from gqa import gqa  # noqa: E402
from golden import gqa as gqa_golden  # noqa: E402

_HALF_DTYPES = (torch.float16, torch.bfloat16)


def _is_npu(x):
    try:
        return str(x.device).startswith("npu")
    except AttributeError:
        return False


def _violates(dtype, device):
    return dtype in _HALF_DTYPES and str(device).startswith("npu")


@contextlib.contextmanager
def fill_guards():
    orig_full = torch.full
    orig_zeros = torch.zeros
    orig_zero_ = torch.Tensor.zero_
    orig_fill_ = torch.Tensor.fill_

    def guarded_full(*args, **kwargs):
        if args and isinstance(args[0], (tuple, list)):
            dtype = kwargs.get("dtype")
            if dtype is None:
                dtype = args[2] if len(args) > 2 else torch.get_default_dtype()
            device = kwargs.get("device")
            if _violates(dtype, device if device is not None else "cpu"):
                raise RuntimeError(
                    "GUARD: torch.full(fp16/bf16, device=npu) — FillScalar "
                    "binary missing on eval SoC (561103)")
        return orig_full(*args, **kwargs)

    def guarded_zeros(*args, **kwargs):
        dtype = kwargs.get("dtype", torch.get_default_dtype())
        device = kwargs.get("device")
        if device is None:
            for a in args:
                if _is_npu(a):
                    device = a
                    break
        if _violates(dtype, device if device is not None else "cpu"):
            raise RuntimeError(
                "GUARD: torch.zeros(fp16/bf16, device=npu) — ZerosLike "
                "binary missing on eval SoC (561103)")
        return orig_zeros(*args, **kwargs)

    def guarded_zero_(self):
        if _is_npu(self) and self.dtype in _HALF_DTYPES:
            raise RuntimeError("GUARD: .zero_() on NPU fp16/bf16 tensor")
        return orig_zero_(self)

    def guarded_fill_(self, value):
        if _is_npu(self) and self.dtype in _HALF_DTYPES:
            raise RuntimeError("GUARD: .fill_() on NPU fp16/bf16 tensor")
        return orig_fill_(self, value)

    torch.full = guarded_full
    torch.zeros = guarded_zeros
    torch.Tensor.zero_ = guarded_zero_
    torch.Tensor.fill_ = guarded_fill_
    try:
        yield
    finally:
        torch.full = orig_full
        torch.zeros = orig_zeros
        torch.Tensor.zero_ = orig_zero_
        torch.Tensor.fill_ = orig_fill_


# The G1 failure classes: non-aligned prefill/decode padding in both dtypes
CASES = [
    # 73-class: S%64 != 0 -> Q padding (prefill)
    ("g1-73-fp16", 2, 124, 128, 32, 8, 128, False, torch.float16),
    ("g1-73-bf16", 2, 124, 128, 32, 8, 128, False, torch.bfloat16),
    # 71/72/81/82-class: Skv%128 != 0 -> K/V padding + wrapper mask
    ("g1-81-fp16", 2, 128, 124, 32, 8, 128, True, torch.float16),
    ("g1-82-bf16", 2, 128, 124, 48, 4, 128, False, torch.bfloat16),
    # decode K/V padding (both dtypes, causal + non-causal)
    ("g1-decode-fp16", 16, 1, 2044, 32, 8, 128, False, torch.float16),
    ("g1-decode-bf16", 16, 1, 2044, 96, 8, 128, False, torch.bfloat16),
    ("g1-decode-causal", 60, 2, 2044, 32, 8, 128, True, torch.float16),
    # D=512 (ladder padding) + large non-aligned prefill
    ("g1-d512-fp16", 2, 252, 252, 32, 8, 512, True, torch.float16),
]


def main():
    n_pass = 0
    for (tag, B, S, S_kv, N_q, N_kv, D, causal, dtype) in CASES:
        torch.manual_seed(21)
        q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
        k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
        v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
        try:
            with fill_guards():
                out = gqa(query=q, key=k, value=v, scaleValue=-1.0,
                          is_causal=causal)
        except RuntimeError as e:
            if str(e).startswith("GUARD"):
                print(f"[{tag}] FAIL: {e}", flush=True)
                continue
            raise
        ref = gqa_golden(q, k, v, -1.0, causal)
        of, rf = out.float().cpu(), ref.float().cpu()
        diff = (of - rf).abs()
        max_diff = diff.max().item()
        atol = 2e-3 if dtype == torch.float16 else 2e-2
        viol = (diff > atol + 1e-2 * rf.abs()).sum().item()
        n_nan = torch.isnan(of).sum().item()
        ok = viol == 0 and n_nan == 0
        print(f"[{tag}] {'PASS' if ok else 'FAIL'} max_diff={max_diff:.3e} "
              f"tolviol={viol} nan={n_nan}", flush=True)
        n_pass += ok
    print(f"\nG1 fill-guard result: {n_pass}/{len(CASES)} PASS")
    if n_pass != len(CASES):
        sys.exit(1)
    print("[PRECISION_PASS] no fp16/bf16 Fill/ZerosLike dispatched on any "
          "padding path; outputs match golden")


if __name__ == "__main__":
    main()
