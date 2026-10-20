"""F6: full accuracy regression — cases.csv base shapes x dtypes x variants.

Per the task spec:
  1. All 20 base shapes from cases.csv x {fp16, bfloat16} (causal per csv).
  2. Non-aligned variants: (S-4, S_kv), (S, S_kv-4), (S-4, S_kv-4) per base
     shape (skipping dimensions that would go <= 0) — simulates the eval's
     non-aligned cases (which exercise the F1/F2/F3 code paths).
  3. causal S > S_kv semantics (golden F217): fully-masked rows output 0.

Acceptance: no NaN/Inf; elementwise |a-b| <= atol + rtol*|b|
(fp16 atol=1e-3/rtol=1e-2, bf16 atol=2e-2/rtol=1e-2 per task spec — the
baseline itself shows 1-2 ULP bf16 rounding up to 7.8e-3, bit-identical
before/after the fixes per f6_bf16_ab.py); causal fully-masked rows
exactly 0; max_diff at the pre-existing magnitude (fp16 <= 2e-3).

Usage:
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python examples/gqa/eval_fix_verify/f6_accuracy_regression.py \
        [--quick]   # base shapes only, single dtype subset
"""

import os
import sys
import time

import torch
import torch_npu  # noqa: F401

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "examples", "gqa"))
sys.path.insert(0, os.path.join(_REPO, ".agents", "skills", "cann-bench",
                                "tasks", "level4", "gqa"))

from gqa import gqa  # noqa: E402
from golden import gqa as gqa_golden  # noqa: E402

# (case_id, B, S, S_kv, N_q, N_kv, D, causal, scale) — from cases.csv
CASES = [
    (1, 2, 128, 128, 32, 8, 128, True, -1.0),
    (2, 2, 128, 128, 48, 4, 128, False, -1.0),
    (3, 2, 128, 128, 128, 8, 128, True, -1.0),
    (4, 4, 256, 256, 64, 8, 128, False, -1.0),
    (5, 2, 128, 128, 32, 8, 256, True, -1.0),
    (6, 2, 256, 256, 128, 8, 256, True, -1.0),
    (7, 2, 128, 256, 32, 8, 128, False, -1.0),
    (8, 4, 1024, 1024, 32, 8, 128, True, -1.0),
    (9, 4, 512, 512, 64, 8, 128, False, -1.0),
    (10, 4, 512, 512, 128, 8, 128, True, -1.0),
    (11, 16, 1, 2048, 32, 8, 128, False, -1.0),
    (12, 16, 1, 2048, 96, 8, 128, False, -1.0),
    (13, 60, 1, 2048, 32, 8, 128, False, -1.0),
    (14, 128, 1, 512, 128, 8, 128, False, -1.0),
    (15, 60, 1, 1024, 32, 8, 256, False, -1.0),
    (16, 60, 2, 2048, 32, 8, 128, True, -1.0),
    (17, 128, 4, 512, 128, 8, 128, True, -1.0),
    (18, 2, 128, 128, 32, 1, 128, False, -1.0),
    (19, 2, 128, 128, 32, 32, 128, True, -1.0),
    (20, 2, 128, 128, 32, 8, 128, True, 0.08838),
]

QUICK = "--quick" in sys.argv
DTYPES = [torch.float16] if QUICK else [torch.float16, torch.bfloat16]


def variants(case_id, B, S, S_kv, N_q, N_kv, D, causal, scale):
    """Base + non-aligned variants (S-4 / S_kv-4 flavors)."""
    out = [("base", S, S_kv)]
    if not QUICK:
        if S - 4 >= 1:
            out.append(("s-4", S - 4, S_kv))
        if S_kv - 4 >= 1:
            out.append(("skv-4", S, S_kv - 4))
        if S - 4 >= 1 and S_kv - 4 >= 1:
            out.append(("both-4", S - 4, S_kv - 4))
    return out


def check_one(B, S, S_kv, N_q, N_kv, D, causal, scale, dtype, tag):
    torch.manual_seed(42 + B + S + S_kv)
    q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
    k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
    v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()

    t0 = time.time()
    out = gqa(query=q, key=k, value=v, scaleValue=scale, is_causal=causal)
    ref = gqa_golden(q, k, v, scale, causal)
    dt = time.time() - t0

    problems = []
    if out.shape != ref.shape:
        return False, [f"shape {tuple(out.shape)} != {tuple(ref.shape)}"]
    of = out.float().cpu()
    rf = ref.float().cpu()

    n_nan = torch.isnan(of).sum().item()
    n_inf = torch.isinf(of).sum().item()
    if n_nan or n_inf:
        problems.append(f"nan={n_nan} inf={n_inf}")

    # Golden F217: causal fully-masked rows (i < S - S_kv) output 0
    n_masked = max(0, S - S_kv) if causal else 0
    if n_masked:
        mmax = of[:, :n_masked].abs().max().item()
        if mmax != 0.0:
            problems.append(f"masked_rows_absmax={mmax:.3e}!=0")

    diff = (of - rf).abs()
    max_diff = diff.max().item()
    # Task-spec tolerances: fp16 atol=1e-2/rtol=1e-2 class (kept tighter at
    # atol=1e-3 — all observed fp16 max_diff <= 7.3e-4, the S>S_kv class);
    # bf16 atol=2e-2 (task spec — the baseline itself shows 1-2 ULP
    # bf16 rounding up to 7.8e-3 on c5/c6-class shapes, bit-identical
    # before/after the fixes per f6_bf16_ab.py).
    atol = 1e-3 if dtype == torch.float16 else 2e-2
    rtol = 1e-2
    viol = (diff > atol + rtol * rf.abs()).sum().item()
    if viol:
        problems.append(f"tolviol={viol}")
    cap = 2e-3 if dtype == torch.float16 else 2e-2
    if max_diff > cap:
        problems.append(f"max_diff={max_diff:.3e}>{cap:.0e}")

    status = "PASS" if not problems else "FAIL"
    print(f"[{tag}] {status} max_diff={max_diff:.3e} "
          f"({time.time() - t0:.0f}s incl golden)"
          + (f"  PROBLEMS: {'; '.join(problems)}" if problems else ""),
          flush=True)
    return status == "PASS", problems


def main():
    n_pass = n_total = 0
    failures = []
    t_start = time.time()
    for (cid, B, S, S_kv, N_q, N_kv, D, causal, scale) in CASES:
        for dtype in DTYPES:
            dtag = "fp16" if dtype == torch.float16 else "bf16"
            for (vtag, vs, vskv) in variants(cid, B, S, S_kv, N_q, N_kv, D,
                                             causal, scale):
                tag = f"c{cid}-{dtag}-{vtag}"
                ok, probs = check_one(B, vs, vskv, N_q, N_kv, D, causal,
                                      scale, dtype, tag)
                n_total += 1
                if ok:
                    n_pass += 1
                else:
                    failures.append((tag, probs))

    print(f"\nF6 accuracy: {n_pass}/{n_total} PASS "
          f"({time.time() - t_start:.0f}s)")
    if failures:
        print("FAILURES:")
        for tag, probs in failures:
            print(f"  {tag}: {probs}")
        sys.exit(1)
    print("[PRECISION_PASS] F6 full accuracy regression")


if __name__ == "__main__":
    main()
