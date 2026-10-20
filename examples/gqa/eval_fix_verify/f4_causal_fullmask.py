"""F4 verification: causal S > S_kv fully-masked rows must output 0 (golden F217).

Eval failure class B (case 67): causal with S > S_kv makes the first
(S - S_kv) query rows fully masked.  The kernel's causal-skip leaves their
sumexp == 0 -> 0/0 = NaN, and (g2 kernel, W5 max-after-mask) the non-skipped
mixed-block path cancels the additive mask in the max shift.  Golden
(F217) explicitly outputs 0 for fully-masked rows.

Fix under test:
  - [F4]  final normalize divisor: sumexp + 1e-30  (0/1e-30 == 0)
  - [F4b] g2 kernel running-max clamp at 0 (masked cols exp(-2^30) -> 0)

Repro shape (per orchestrator): B=2, S=256, S_kv=128, N_q=32, N_kv=8, D=128,
causal.  Dispatches to _gqa_kernel_prefill_g2 (causal S_kv < S -> not
expert-eligible).  Rows 0..127 live in fully-skipped blocks (bx 0,1); rows
128..191 live in mixed block bx=2 (tests the [F4b] clamp path); rows
192..255 in block bx=3.

Usage:
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python examples/gqa/eval_fix_verify/f4_causal_fullmask.py
"""

import os
import sys

import torch
import torch_npu  # noqa: F401  (registers the npu backend)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_GQA_DIR = os.path.join(_REPO, "examples", "gqa")
_GOLDEN_DIR = os.path.join(
    _REPO, ".agents", "skills", "cann-bench", "tasks", "level4", "gqa")
sys.path.insert(0, _GQA_DIR)
sys.path.insert(0, _GOLDEN_DIR)

from gqa import gqa  # noqa: E402
from golden import gqa as gqa_golden  # noqa: E402

torch.manual_seed(42)
DEV = "npu:0"


def run_case(B, S, S_kv, N_q, N_kv, D, dtype, causal, tag):
    q = (torch.rand(B, S, N_q, D, dtype=torch.float32) * 2 - 1).to(dtype).to(DEV)
    k = (torch.rand(B, S_kv, N_kv, D, dtype=torch.float32) * 2 - 1).to(dtype).to(DEV)
    v = (torch.rand(B, S_kv, N_kv, D, dtype=torch.float32) * 2 - 1).to(dtype).to(DEV)

    out = gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
    ref = gqa_golden(q, k, v, -1.0, causal)

    assert out.shape == ref.shape, f"{tag}: shape {out.shape} != {ref.shape}"
    assert out.dtype == ref.dtype, f"{tag}: dtype {out.dtype} != {ref.dtype}"

    n_nan = torch.isnan(out.float()).sum().item()
    n_inf = torch.isinf(out.float()).sum().item()

    out_f = out.float().cpu()
    ref_f = ref.float().cpu()

    # Golden F217: rows i < S - S_kv are fully masked -> output 0.
    n_masked_rows = max(0, S - S_kv) if causal else 0
    if n_masked_rows > 0:
        masked = out_f[:, :n_masked_rows, :, :]
        masked_max = masked.abs().max().item()
    else:
        masked_max = 0.0

    # Visible-region comparison (rows >= S - S_kv)
    if n_masked_rows > 0:
        ov = out_f[:, n_masked_rows:, :, :]
        rv = ref_f[:, n_masked_rows:, :, :]
    else:
        ov, rv = out_f, ref_f
    diff = (ov - rv).abs()
    max_diff = diff.max().item()

    ok_nan = (n_nan == 0) and (n_inf == 0)
    ok_masked = masked_max == 0.0
    # Mixed tolerance per task spec (fp16 atol=1e-2/rtol=1e-2 class; bf16
    # atol=2e-2).  The pre-existing fp16 kernel-vs-golden magnitude for this
    # case class is 7.3e-4 (verified bit-identical vs the baseline in
    # f4b_clamp_isolation.py), so atol=1e-3 + rtol=1e-2 catches real errors
    # with ample margin over ordinary fp16 rounding.
    atol = 1e-3 if dtype == torch.float16 else 4e-3
    rtol = 1e-2
    viol = (diff > atol + rtol * rv.abs()).sum().item()
    ok_visible = (viol == 0) and max_diff <= 10 * atol

    status = "PASS" if (ok_nan and ok_masked and ok_visible) else "FAIL"
    print(f"[{tag}] {status}: nan={n_nan} inf={n_inf} "
          f"masked_rows_absmax={masked_max:.3e} (expect 0) "
          f"visible_max_diff={max_diff:.3e} tolviol={viol} "
          f"n_masked_rows={n_masked_rows}")
    return status == "PASS"


def main():
    results = []
    # Primary repro (orchestrator spec): full-skip blocks + mixed block
    results.append(run_case(2, 256, 128, 32, 8, 128, torch.float16, True,
                            "f4-repro-fp16"))
    results.append(run_case(2, 256, 128, 32, 8, 128, torch.bfloat16, True,
                            "f4-repro-bf16"))
    # Mixed-block-only variant: S - S_kv = 4 (not a multiple of block_M=64)
    # -> fully-masked rows inside a NON-skipped block ([F4b] clamp path)
    results.append(run_case(2, 132, 128, 32, 8, 128, torch.float16, True,
                            "f4-mixed-fp16"))
    results.append(run_case(2, 132, 128, 32, 8, 128, torch.bfloat16, True,
                            "f4-mixed-bf16"))
    # Per-head kernel path (G=1): causal S > S_kv, mask applied after shift
    results.append(run_case(2, 192, 128, 8, 8, 128, torch.float16, True,
                            "f4-perhead-G1-fp16"))
    # Non-causal sanity (no masked rows)
    results.append(run_case(2, 256, 128, 32, 8, 128, torch.float16, False,
                            "f4-noncausal-fp16"))

    n_pass = sum(results)
    print(f"\nF4 result: {n_pass}/{len(results)} PASS")
    if n_pass != len(results):
        sys.exit(1)
    print("[PRECISION_PASS] F4 causal fully-masked-row semantics verified")


if __name__ == "__main__":
    main()
