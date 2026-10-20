"""F1/F2 verification: non-aligned eval paths must not invoke the missing
NPU binaries (ZerosLike / ConcatD / Cat_1_SliceAiCore / aclnn Transpose).

The eval SoC (CANN 9.1.0 / Ascend910_9362) is missing these aclnn binaries
(error 561103).  This script monkey-patches the corresponding torch entry
points to RAISE if they would dispatch on an NPU tensor, then runs the
shapes of the four A2 failure classes (71/72/73/81) plus the A1 transpose
class end-to-end through gqa() and checks the results against golden.

Guards (active ONLY around the gqa() call, restored for the golden):
  - torch.zeros(device=npu)          -> ZerosLike
  - torch.cat on NPU tensors         -> ConcatD / Cat_1_SliceAiCore
  - non-contiguous NPU .contiguous() -> aclnn Transpose materialization

Usage:
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python examples/gqa/eval_fix_verify/f12_op_guards.py
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


def _is_npu(x):
    try:
        return str(x.device).startswith("npu")
    except AttributeError:
        return False


def _is_permuted(t):
    """True if the layout is dimension-PERMUTED (aclnn Transpose territory).

    A crop-only view (e.g. x[:, :, :S, :]) keeps strictly decreasing strides
    and materializes via aclnnInplaceCopy -> the ViewCopy device kernel —
    the SAME binary the eval-verified slice-assignment pattern dispatches
    (verified: strings in libtorch_npu.so show copy_ -> aclnnInplaceCopy ->
    contiguous_d_ViewCopy; only permuted layouts route to aclnnTranspose,
    which is the missing binary).  A permuted layout (e.g. transpose(1,2))
    breaks the decreasing-stride order.
    """
    pairs = [(abs(t.stride()[i]), t.shape[i]) for i in range(t.dim())
             if t.shape[i] > 1]
    for i in range(len(pairs) - 1):
        if pairs[i][0] <= pairs[i + 1][0]:
            return True
    return False


@contextlib.contextmanager
def npu_op_guards():
    orig_zeros = torch.zeros
    orig_cat = torch.cat
    orig_contig = torch.Tensor.contiguous

    def guarded_zeros(*args, **kwargs):
        device = kwargs.get("device")
        if device is None:
            for a in args:
                if _is_npu(a):
                    device = a
                    break
        if device is not None and str(device).startswith("npu"):
            raise RuntimeError(
                "GUARD: torch.zeros on NPU (ZerosLike binary missing on "
                "eval SoC)")
        return orig_zeros(*args, **kwargs)

    def guarded_cat(tensors, *args, **kwargs):
        if any(_is_npu(t) for t in tensors):
            raise RuntimeError(
                "GUARD: torch.cat on NPU tensors (ConcatD binary missing on "
                "eval SoC)")
        return orig_cat(tensors, *args, **kwargs)

    def guarded_contiguous(self, *args, **kwargs):
        if _is_npu(self) and not self.is_contiguous():
            if _is_permuted(self):
                raise RuntimeError(
                    "GUARD: .contiguous() materializing a PERMUTED NPU "
                    "tensor (aclnn Transpose binary missing on eval SoC); "
                    f"shape={tuple(self.shape)} stride={self.stride()}")
            # crop-only materialization: aclnnInplaceCopy/ViewCopy — same
            # verified binary as slice assignment, allowed.
        return orig_contig(self, *args, **kwargs)

    torch.zeros = guarded_zeros
    torch.cat = guarded_cat
    torch.Tensor.contiguous = guarded_contiguous
    try:
        yield
    finally:
        torch.zeros = orig_zeros
        torch.cat = orig_cat
        torch.Tensor.contiguous = orig_contig


# (tag, B, S, S_kv, N_q, N_kv, D, causal, dtype, which failure class)
CASES = [
    # A2 71/72 class: prefill non-causal + Skv%128 != 0 -> wrapper zeros (F1)
    ("a2-71-fp16", 2, 128, 124, 32, 8, 128, False, torch.float16),
    ("a2-72-bf16", 2, 128, 124, 48, 4, 128, False, torch.bfloat16),
    # A2 73 class: prefill S%64 != 0 -> Q padding (F2)
    ("a2-73-fp16", 2, 124, 128, 32, 8, 128, False, torch.float16),
    # A2 81 class: prefill Skv%128 != 0 -> K/V padding (F2)
    ("a2-81-fp16", 2, 128, 124, 32, 8, 128, True, torch.float16),
    # decode + non-aligned Skv -> decode _run K/V padding (F2)
    ("a2-decode-fp16", 16, 1, 2044, 32, 8, 128, False, torch.float16),
    ("a2-decode-bf16", 16, 1, 2044, 96, 8, 128, False, torch.bfloat16),
    # decode causal + non-aligned Skv (case 16/17 class)
    ("a2-decode-causal", 60, 2, 2044, 32, 8, 128, True, torch.float16),
    # A1 31/50/56/57 class: non-divisible transpose -> tail mode (F3)
    ("a1-fp16", 2, 124, 252, 32, 8, 128, False, torch.float16),
    ("a1-bf16", 2, 124, 252, 32, 8, 128, False, torch.bfloat16),
    ("a1-large-fp16", 4, 1020, 1020, 32, 8, 128, True, torch.float16),
]


def main():
    n_pass = 0
    for (tag, B, S, S_kv, N_q, N_kv, D, causal, dtype) in CASES:
        torch.manual_seed(11)
        q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
        k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
        v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
        try:
            with npu_op_guards():
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
        atol = 1e-3 if dtype == torch.float16 else 2e-2
        viol = (diff > atol + 1e-2 * rf.abs()).sum().item()
        n_nan = torch.isnan(of).sum().item()
        n_masked = max(0, S - S_kv) if causal else 0
        mmax = of[:, :n_masked].abs().max().item() if n_masked else 0.0
        ok = (viol == 0 and n_nan == 0 and mmax == 0.0
              and max_diff <= (2e-3 if dtype == torch.float16 else 2e-2))
        print(f"[{tag}] {'PASS' if ok else 'FAIL'} max_diff={max_diff:.3e} "
              f"tolviol={viol} nan={n_nan} masked_absmax={mmax:.1e}",
              flush=True)
        n_pass += ok
    print(f"\nF1/F2 op-guard result: {n_pass}/{len(CASES)} PASS")
    if n_pass != len(CASES):
        sys.exit(1)
    print("[PRECISION_PASS] no ZerosLike/ConcatD/Transpose dispatched on "
          "non-aligned paths; outputs match golden")


if __name__ == "__main__":
    main()
