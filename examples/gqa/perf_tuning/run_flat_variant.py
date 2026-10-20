"""B0 gate runner — flat g2stack variant precision verification (Stage 3 iter2).

Standalone: does NOT modify examples/gqa/gqa.py / test_gqa.py; imports the
gqa module read-only for helpers (transpose_qkv / transpose_out / _ref_gqa /
_dummy_mask) and the shipped dispatch entry gqa() as the reference kernel.

Semantics mirror test_gqa.py f6 conventions:
  * seed = 42 + B + S + S_kv, inputs uniform [-1,1] in the test dtype
  * golden = _ref_gqa in the INPUT dtype on NPU (f6-verified setup)
  * tolerance: fp16 atol=1e-3 / bf16 atol=2e-2, rtol=1e-2, elementwise
    |out-ref| <= atol + rtol*|ref|
Additionally compares the variant against the SHIPPED kernel output
(gqa() dispatch -> g2stack on all gate shapes) — expected bit-identical
(same per-task math order; cross-task reordering does not touch numerics).

Run (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python examples/gqa/perf_tuning/run_flat_variant.py [--case NAME]
"""

import argparse
import os
import sys
import traceback

import torch
import torch_npu  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_DIR = os.path.dirname(HERE)
sys.path.insert(0, GQA_DIR)
sys.path.insert(0, HERE)

from gqa import _BIG_NEG, _dummy_mask, _ref_gqa, gqa, transpose_out, transpose_qkv  # noqa: E402
from gqa_flat_variant import build_flat_kernel  # noqa: E402

# (name, B, S, S_kv, N_q, N_kv, D, causal, dtype)
CASES = [
    ("r2_causal_fp16", 2, 128, 128, 32, 8, 128, True, torch.float16),
    ("r2_noncausal_fp16", 2, 128, 128, 32, 8, 128, False, torch.float16),
    ("c3like_causal_bf16", 2, 128, 128, 128, 8, 128, True, torch.bfloat16),
    ("c2like_noncausal_fp16", 2, 128, 128, 48, 4, 128, False, torch.float16),
    ("mc2_edge_causal_fp16", 5, 128, 128, 8, 2, 128, True, torch.float16),
    ("mc2_edge_noncausal_bf16", 5, 128, 128, 8, 2, 128, False, torch.bfloat16),
    ("mixed_mc23_causal_fp16", 3, 128, 128, 16, 4, 128, True, torch.float16),
]


def _padded_causal_mask(B, S, S_kv, S_pad, Skv_pad, device):
    """[B, S_pad, Skv_pad] fp32 additive mask — same semantics as
    _build_causal_mask + _prefill_run_padded's Mask_full (pad rows/cols
    fully masked with _BIG_NEG). Built on CPU + single H2D (avoids the
    missing ZerosLike/FillScalar device binaries — same rationale as gqa.py).
    """
    mask = torch.full((S_pad, Skv_pad), _BIG_NEG, dtype=torch.float32)
    i = torch.arange(S).unsqueeze(-1)
    j = torch.arange(S_kv).unsqueeze(0)
    visible = j <= (i + (S_kv - S))
    mask[:S, :S_kv][visible] = 0.0
    return mask.unsqueeze(0).expand(B, S_pad, Skv_pad).contiguous().to(device)


def run_case(name, B, S, S_kv, N_q, N_kv, D, causal, dtype):
    dtag = "fp16" if dtype == torch.float16 else "bf16"
    tag = f"{name} B={B} S={S} S_kv={S_kv} N_q={N_q} N_kv={N_kv} D={D} causal={causal} {dtag}"
    try:
        torch.manual_seed(42 + B + S + S_kv)
        q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
        k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
        v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
        scale = -1.0
        sm_scale = 1.0 / (D**0.5)

        # ---- reference: shipped dispatch (g2stack on all gate shapes) ----
        out_ref = gqa(query=q, key=k, value=v, scaleValue=scale, is_causal=causal)
        torch.npu.synchronize()

        # ---- golden: input dtype on NPU (f6-verified suite semantics) ----
        golden = _ref_gqa(q, k, v, scale, causal)
        torch.npu.synchronize()

        # ---- variant prep: same BHSD-padded path as the gqa() adapter ----
        S_pad = ((S + 63) // 64) * 64
        Skv_pad = ((S_kv + 127) // 128) * 128
        res = transpose_qkv(q, k, v, S_out=S_pad, S_kv_out=Skv_pad)
        assert res is not None, "transpose_qkv fallback (None) — unexpected on NPU"
        qb, kb, vb = res
        dtype_str = "float16" if dtype == torch.float16 else "bfloat16"
        kernel_fn, _, _, use_mask = build_flat_kernel(
            B, S, S_kv, sm_scale, causal, N_q, N_kv, D, dtype_str
        )
        if use_mask:
            mask = _padded_causal_mask(B, S, S_kv, S_pad, Skv_pad, q.device) if causal else None
            if mask is None:
                # non-causal + column padding: all-zero real cols, _BIG_NEG pads
                mask = torch.full((S_pad, Skv_pad), _BIG_NEG, dtype=torch.float32)
                mask[:, :S_kv] = 0.0
                mask = mask.unsqueeze(0).expand(B, S_pad, Skv_pad).contiguous().to(q.device)
        else:
            mask = _dummy_mask(q.device)
        out_bhsd = kernel_fn(qb, kb, vb, mask)
        torch.npu.synchronize()
        out_var = transpose_out(out_bhsd)
        torch.npu.synchronize()

        # ---- compare: variant vs golden (suite tolerance) ----
        of = out_var.float().cpu()
        gf = golden.float().cpu()
        atol = 1e-3 if dtype == torch.float16 else 2e-2
        rtol = 1e-2
        diff = (of - gf).abs()
        max_diff = float(diff.max())
        viol = int((diff > atol + rtol * gf.abs()).sum())
        # NaN/Inf screen
        bad = int((~torch.isfinite(of)).sum())

        # ---- compare: variant vs shipped kernel (expect bit-identical) ----
        rf = out_ref.float().cpu()
        max_diff_ref = float((of - rf).abs().max())
        bit_identical = bool(torch.equal(out_var, out_ref))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        print(f"[B0_FAIL] {tag} error={e}", flush=True)
        return False
    ok = viol == 0 and bad == 0
    status = "B0_PASS" if ok else "B0_FAIL"
    print(
        f"[{status}] {tag} golden_max_diff={max_diff:.3e} viol={viol} nonfinite={bad} "
        f"vs_shipped_max_diff={max_diff_ref:.3e} bit_identical={bit_identical}",
        flush=True,
    )
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None, help="run a single case by name")
    args = ap.parse_args()
    cases = CASES if args.case is None else [c for c in CASES if c[0] == args.case]
    if not cases:
        print(f"unknown case {args.case}; available: {[c[0] for c in CASES]}")
        sys.exit(2)
    results = [run_case(*c) for c in cases]
    n_ok = sum(results)
    print(f"\n[B0_SUMMARY] {n_ok}/{len(results)} cases passed", flush=True)
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
