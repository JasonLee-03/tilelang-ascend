"""F6.6: 80-case sequential simulation — process-level robustness check.

Mimics the eval structure: 20 base shapes x {fp16, bf16} x {aligned,
non-aligned} = 80 gqa() calls executed SEQUENTIALLY in ONE process (the
non-aligned variant uses S-4 / S_kv-4 where the dim allows).  Checks that
no call raises, hangs, returns NaN/Inf, or corrupts subsequent calls —
the local expectation is that the D-class cascading SIGSEGV (cases
82-100, environmental in the eval sandbox) does NOT reproduce here.

Usage:
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python examples/gqa/eval_fix_verify/f6_sequential_80.py
"""

import os
import sys
import time
import traceback

import torch
import torch_npu  # noqa: F401

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "examples", "gqa"))

from gqa import gqa  # noqa: E402

# (B, S, S_kv, N_q, N_kv, D, causal, scale) — from cases.csv
BASES = [
    (2, 128, 128, 32, 8, 128, True, -1.0),
    (2, 128, 128, 48, 4, 128, False, -1.0),
    (2, 128, 128, 128, 8, 128, True, -1.0),
    (4, 256, 256, 64, 8, 128, False, -1.0),
    (2, 128, 128, 32, 8, 256, True, -1.0),
    (2, 256, 256, 128, 8, 256, True, -1.0),
    (2, 128, 256, 32, 8, 128, False, -1.0),
    (4, 1024, 1024, 32, 8, 128, True, -1.0),
    (4, 512, 512, 64, 8, 128, False, -1.0),
    (4, 512, 512, 128, 8, 128, True, -1.0),
    (16, 1, 2048, 32, 8, 128, False, -1.0),
    (16, 1, 2048, 96, 8, 128, False, -1.0),
    (60, 1, 2048, 32, 8, 128, False, -1.0),
    (128, 1, 512, 128, 8, 128, False, -1.0),
    (60, 1, 1024, 32, 8, 256, False, -1.0),
    (60, 2, 2048, 32, 8, 128, True, -1.0),
    (128, 4, 512, 128, 8, 128, True, -1.0),
    (2, 128, 128, 32, 1, 128, False, -1.0),
    (2, 128, 128, 32, 32, 128, True, -1.0),
    (2, 128, 128, 32, 8, 128, True, 0.08838),
]


def nonaligned(S, S_kv):
    s = S - 4 if S - 4 >= 1 else S
    skv = S_kv - 4 if S_kv - 4 >= 1 else S_kv
    return s, skv


def main():
    n_ok = 0
    n_cases = 0
    t0 = time.time()
    for dtype in (torch.float16, torch.bfloat16):
        for (B, S, S_kv, N_q, N_kv, D, causal, scale) in BASES:
            for aligned in (True, False):
                vs, vskv = (S, S_kv) if aligned else nonaligned(S, S_kv)
                tag = (f"{'al' if aligned else 'na'}-"
                       f"{'fp16' if dtype == torch.float16 else 'bf16'}-"
                       f"B{B}S{vs}K{vskv}")
                n_cases += 1
                try:
                    torch.manual_seed(100 + n_cases)
                    q = (torch.rand(B, vs, N_q, D) * 2 - 1).to(dtype).npu()
                    k = (torch.rand(B, vskv, N_kv, D) * 2 - 1).to(dtype).npu()
                    v = (torch.rand(B, vskv, N_kv, D) * 2 - 1).to(dtype).npu()
                    out = gqa(query=q, key=k, value=v, scaleValue=scale,
                              is_causal=causal)
                    of = out.float().cpu()
                    bad = torch.isnan(of).sum().item() + \
                        torch.isinf(of).sum().item()
                    if bad:
                        print(f"[{tag}] FAIL: {bad} NaN/Inf elements",
                              flush=True)
                        continue
                    # causal fully-masked rows must be exactly 0
                    nm = max(0, vs - vskv) if causal else 0
                    if nm and of[:, :nm].abs().max().item() != 0.0:
                        print(f"[{tag}] FAIL: masked rows nonzero", flush=True)
                        continue
                    n_ok += 1
                    print(f"[{tag}] ok ({time.time() - t0:.0f}s elapsed)",
                          flush=True)
                except Exception:  # noqa: BLE001
                    print(f"[{tag}] EXCEPTION:", flush=True)
                    traceback.print_exc()
    print(f"\nSequential 80-case simulation: {n_ok}/{n_cases} clean "
          f"({time.time() - t0:.0f}s)")
    if n_ok != n_cases:
        sys.exit(1)
    print("[PRECISION_PASS] 80-case sequential simulation — no crashes, "
          "no NaN, no cascade")


if __name__ == "__main__":
    main()
