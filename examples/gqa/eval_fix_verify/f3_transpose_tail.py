"""F3 verification: transpose_qkv tail mode for non-divisible S / S_kv.

Eval failure class A1 (cases 31/50/56/57): when S or S_kv is not divisible
by a shared whole-block bm, transpose_qkv used to return None and the
wrapper fell back to aclnn Transpose — missing binary on the eval SoC
(error 561103).

Fix under test: tail mode (per-row guarded copies, fixed bm) inside
_transpose_qkv + dispatch fallback.  Transpose is pure data movement, so
the results must be BIT-EXACT vs torch .transpose(1, 2).contiguous().

Covers:
  - V2 tail (S >= 256 or Nq <= 64, no divisible shared bm)  -> universal tail
  - V1 + KV tail (S < 256, Nq > 64, Skv non-divisible)
  - Nq % q_bn != 0 (V1 Q segment unusable)                  -> universal tail
  - aligned shapes keep the whole-block path (bit-exact, unchanged)

Usage:
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python examples/gqa/eval_fix_verify/f3_transpose_tail.py
"""

import os
import sys

import torch
import torch_npu  # noqa: F401

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "examples", "gqa"))

from gqa import transpose_qkv  # noqa: E402

torch.manual_seed(42)
DEV = "npu:0"


def run_case(B, S, Skv, Nq, Nkv, D, dtype, tag):
    q = (torch.rand(B, S, Nq, D) * 2 - 1).to(dtype).to(DEV)
    k = (torch.rand(B, Skv, Nkv, D) * 2 - 1).to(dtype).to(DEV)
    v = (torch.rand(B, Skv, Nkv, D) * 2 - 1).to(dtype).to(DEV)

    res = transpose_qkv(q, k, v)
    if res is None:
        print(f"[{tag}] FAIL: transpose_qkv returned None (would fall back "
              f"to aclnn Transpose)")
        return False
    qy, ky, vy = res

    q_ref = q.transpose(1, 2).contiguous()
    k_ref = k.transpose(1, 2).contiguous()
    v_ref = v.transpose(1, 2).contiguous()

    ok = True
    for name, a, b in (("Q", qy, q_ref), ("K", ky, k_ref), ("V", vy, v_ref)):
        if a.dtype != b.dtype or a.shape != b.shape:
            print(f"[{tag}] FAIL {name}: shape/dtype mismatch "
                  f"{tuple(a.shape)}/{a.dtype} vs {tuple(b.shape)}/{b.dtype}")
            ok = False
            continue
        same = torch.equal(a, b)  # bit-exact
        if not same:
            n_diff = (a != b).sum().item()
            print(f"[{tag}] FAIL {name}: {n_diff}/{a.numel()} elements differ "
                  f"(max_abs={(a.float() - b.float()).abs().max().item():.3e})")
            ok = False
    if ok:
        print(f"[{tag}] PASS: bit-exact Q/K/V transpose "
              f"(B{B} S{S} Skv{Skv} Nq{Nq} Nkv{Nkv} D{D} {str(dtype).split('.')[-1]})")
    return ok


def main():
    results = []
    # --- V2 tail: orchestrator repro shape (S/Skv not 16-multiples) ---
    results.append(run_case(2, 250, 246, 32, 8, 128, torch.float16,
                            "f3-v2tail-fp16"))
    results.append(run_case(2, 250, 246, 32, 8, 128, torch.bfloat16,
                            "f3-v2tail-bf16"))
    # --- case-8 variant class: large S == Skv, not divisible ---
    results.append(run_case(4, 1020, 1020, 32, 8, 128, torch.float16,
                            "f3-v2tail-large"))
    # --- case-7 variant class: S=124, Skv=252 (A1 shape) ---
    results.append(run_case(2, 124, 252, 32, 8, 128, torch.float16,
                            "f3-v2tail-a1shape"))
    # --- V1 + KV tail: S<256, Nq>64, Skv=400 with D=256 (kv search fails) ---
    results.append(run_case(2, 128, 400, 128, 8, 256, torch.float16,
                            "f3-v1-kvtail"))
    # --- universal tail via Nq % q_bn != 0 (Nq=192, q_bn=128) ---
    results.append(run_case(2, 128, 128, 192, 8, 128, torch.float16,
                            "f3-nq-nondiv"))
    # --- aligned paths unchanged (whole-block) ---
    results.append(run_case(2, 256, 256, 32, 8, 128, torch.float16,
                            "f3-aligned-v2"))
    results.append(run_case(2, 128, 128, 128, 8, 128, torch.float16,
                            "f3-aligned-v1"))
    results.append(run_case(2, 128, 256, 128, 8, 128, torch.float16,
                            "f3-aligned-v1-kv"))

    n_pass = sum(results)
    print(f"\nF3 result: {n_pass}/{len(results)} PASS")
    if n_pass != len(results):
        sys.exit(1)
    print("[PRECISION_PASS] F3 transpose tail mode bit-exact")


if __name__ == "__main__":
    main()
