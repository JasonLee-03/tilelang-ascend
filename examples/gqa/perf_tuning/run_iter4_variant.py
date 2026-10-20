"""Gate 0 runner — iter4 flat variants (C1 ns_flat=4 / C3 G2-first) precision.

Standalone: does NOT modify examples/gqa/gqa.py or test_gqa.py.  Imports the
gqa module read-only (transpose_qkv / transpose_out / _ref_gqa / _dummy_mask /
_BIG_NEG and the shipped gqa() dispatch as the reference) and builds the
selected variant kernel from perf_tuning/gqa_flat_{ns4,g2first}_variant.py.

Criteria (both are pure scheduling / slot-depth changes — arithmetic sequence
per task is identical to the shipped flat kernel):
  1. variant output BIT-IDENTICAL to shipped gqa() output (torch.equal; for
     the sparse NaN/Inf case: NaN positions equal, Inf positions+signs equal,
     finite entries bit-equal);
  2. variant within suite tolerance vs the input-dtype NPU golden
     (fp16 atol=1e-3 / bf16 atol=2e-2, rtol=1e-2 — test_gqa f6 semantics).
A non-bit-identical finite-input result is treated as a BUG (locate, do not
relax tolerances).

Run (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python3 examples/gqa/perf_tuning/run_iter4_variant.py --arm ns4
    python3 examples/gqa/perf_tuning/run_iter4_variant.py --arm g2first [--case NAME]
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

from gqa import (  # noqa: E402
    P28_NUM_CORES,
    _BIG_NEG,
    _dummy_mask,
    _ref_gqa,
    gqa,
    transpose_out,
    transpose_qkv,
)

VARIANT_MODULES = {
    "ns4": "gqa_flat_ns4_variant",
    "g2first": "gqa_flat_g2first_variant",
}

# (name, B, S, S_kv, N_q, N_kv, D, causal, dtype)
CASES = [
    # --- the three in-gate A/B shapes ---
    ("r3_c3like_causal_bf16", 2, 128, 128, 128, 8, 128, True, torch.bfloat16),
    ("r2_c1like_causal_fp16", 2, 128, 128, 32, 8, 128, True, torch.float16),
    ("r6_c2like_noncausal_fp16", 2, 128, 128, 48, 4, 128, False, torch.float16),
    # --- edge / structural coverage (iter2 B0 set) ---
    ("r2_noncausal_fp16", 2, 128, 128, 32, 8, 128, False, torch.float16),
    ("mc2_edge_causal_fp16", 5, 128, 128, 8, 2, 128, True, torch.float16),
    ("mc2_edge_noncausal_bf16", 5, 128, 128, 8, 2, 128, False, torch.bfloat16),
    ("mixed_mc23_causal_fp16", 3, 128, 128, 16, 4, 128, True, torch.float16),
    # --- new iter4 coverage: S_kv non-aligned (pad cols, use_mask noncausal),
    #     causal S_kv > S (skv_minus_s > 0), both in-gate ---
    ("padcol_noncausal_fp16", 2, 128, 100, 32, 8, 128, False, torch.float16),
    ("causal_skvgts_fp16", 4, 64, 128, 32, 8, 128, True, torch.float16),
]

# Boundary sparse-special case (iter2 fill-removal NaN-leak lesson): finite
# base with position-sensitive NaN/Inf injections — same injection plan as
# test_gqa.py boundary_sparse_special, on the R2 shape (noncausal fp16,
# in-gate so BOTH shipped and variant take the flat path).
BOUNDARY_CASE = ("boundary_sparse_r2like", 2, 128, 128, 32, 8, 128, False, torch.float16)


def _padded_causal_mask(B, S, S_kv, S_pad, Skv_pad, device):
    """[B, S_pad, Skv_pad] fp32 additive mask — _build_causal_mask semantics
    (right-aligned: j > i + (S_kv - S) masked) + pad rows/cols _BIG_NEG,
    matching _prefill_run_padded's Mask_full.  CPU build + single H2D."""
    mask = torch.full((S_pad, Skv_pad), _BIG_NEG, dtype=torch.float32)
    i = torch.arange(S).unsqueeze(-1)
    j = torch.arange(S_kv).unsqueeze(0)
    visible = j <= (i + (S_kv - S))
    mask[:S, :S_kv][visible] = 0.0
    return mask.unsqueeze(0).expand(B, S_pad, Skv_pad).contiguous().to(device)


def _padcol_mask(B, S, S_kv, S_pad, Skv_pad, device):
    """Noncausal + S_kv pad columns: real cols 0, pad cols _BIG_NEG
    (gqa() adapter builds zeros[B,S,S_kv]; _prefill_run_padded embeds it in
    a _BIG_NEG Mask_full).  S is assumed block-aligned here (S == S_pad)."""
    assert S == S_pad
    mask = torch.full((S_pad, Skv_pad), _BIG_NEG, dtype=torch.float32)
    mask[:, :S_kv] = 0.0
    return mask.unsqueeze(0).expand(B, S_pad, Skv_pad).contiguous().to(device)


def _build_variant(arm, B, S, S_kv, sm_scale, causal, N_q, N_kv, D, dtype_str):
    import importlib

    mod = importlib.import_module(VARIANT_MODULES[arm])
    factory = mod._gqa_kernel_prefill_g2stack_flat
    block_M, block_N = 64, 128
    S_pad = ((S + block_M - 1) // block_M) * block_M
    Skv_pad = ((S_kv + block_N - 1) // block_N) * block_N
    kv_iters = Skv_pad // block_N
    G = N_q // N_kv
    block_num = B * N_kv * (G // 2) * (S_pad // block_M)
    eligible = (
        D == 128
        and dtype_str in ("float16", "bfloat16")
        and kv_iters == 1
        and ((not causal) or (S_kv >= S))
        and G % 2 == 0
        and block_num >= 2 * P28_NUM_CORES
    )
    if not eligible:
        raise ValueError(f"case outside flat gate: B={B} S={S} S_kv={S_kv} Nq={N_q} Nkv={N_kv}")
    use_mask = bool(causal or (Skv_pad != S_kv))
    kernel_fn = factory(
        B, S_pad, Skv_pad, sm_scale, causal,
        N_q=N_q, N_kv=N_kv, D=D, dtype=dtype_str, block_M=block_M,
        skv_minus_s=S_kv - S, use_mask=use_mask, block_N=block_N,
    )
    return kernel_fn, S_pad, Skv_pad, use_mask


def _inputs(name, B, S, S_kv, N_q, N_kv, D, causal, dtype):
    torch.manual_seed(42 + B + S + S_kv)
    q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
    k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
    v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
    if name.startswith("boundary_sparse"):
        # test_gqa boundary_sparse_special injection plan (G=4: kv head h
        # serves q heads 4h..4h+3):
        #   q[0, 5, 3, 0]  = NaN  -> out[0, 5, 3, :]   all NaN
        #   v[0, 20, 1, 7] = +Inf -> out[0, :, 4:8, 7] all +Inf
        #   v[1, 30, 5, 9] = -Inf -> out[1, :, 20:24, 9] all -Inf
        q[0, 5, 3, 0] = float("nan")
        v[0, 20, 1, 7] = float("inf")
        v[1, 30, 5, 9] = float("-inf")
    return q, k, v


def _special_struct_equal(a, b):
    """NaN/Inf-aware bitwise comparison: NaN positions equal, Inf positions
    and signs equal, finite entries bit-equal."""
    nan_a, nan_b = torch.isnan(a), torch.isnan(b)
    if not bool((nan_a == nan_b).all()):
        return False, f"nan_pos_mismatch={int((nan_a != nan_b).sum())}"
    inf_a, inf_b = torch.isinf(a), torch.isinf(b)
    if not bool((inf_a == inf_b).all()):
        return False, f"inf_pos_mismatch={int((inf_a != inf_b).sum())}"
    both_inf = inf_a & inf_b
    if bool(both_inf.any()) and not bool((torch.sign(a[both_inf]) == torch.sign(b[both_inf])).all()):
        return False, "inf_sign_mismatch"
    finite = ~nan_a & ~inf_a
    if not bool(torch.equal(a[finite], b[finite])):
        n = int((a[finite] != b[finite]).sum())
        return False, f"finite_bit_mismatch={n}"
    return True, "ok"


def run_case(arm, case, boundary=False):
    name, B, S, S_kv, N_q, N_kv, D, causal, dtype = case
    dtag = "fp16" if dtype == torch.float16 else "bf16"
    tag = f"[{arm}] {name} B={B} S={S} S_kv={S_kv} Nq={N_q} Nkv={N_kv} causal={causal} {dtag}"
    try:
        q, k, v = _inputs(name, B, S, S_kv, N_q, N_kv, D, causal, dtype)
        scale = -1.0
        sm_scale = 1.0 / (D**0.5)

        # ---- reference: shipped dispatch (flat on all in-gate shapes) ----
        out_ref = gqa(query=q, key=k, value=v, scaleValue=scale, is_causal=causal)
        torch.npu.synchronize()

        # ---- variant: same BHSD-padded path as the gqa() adapter ----
        S_pad = ((S + 63) // 64) * 64
        Skv_pad = ((S_kv + 127) // 128) * 128
        res = transpose_qkv(q, k, v, S_out=S_pad, S_kv_out=Skv_pad)
        assert res is not None, "transpose_qkv fallback (None) — unexpected on NPU"
        qb, kb, vb = res
        dtype_str = "float16" if dtype == torch.float16 else "bfloat16"
        kernel_fn, _, _, use_mask = _build_variant(
            arm, B, S, S_kv, sm_scale, causal, N_q, N_kv, D, dtype_str
        )
        if use_mask:
            if causal:
                mask = _padded_causal_mask(B, S, S_kv, S_pad, Skv_pad, q.device)
            else:
                mask = _padcol_mask(B, S, S_kv, S_pad, Skv_pad, q.device)
        else:
            mask = _dummy_mask(q.device)
        out_bhsd = kernel_fn(qb, kb, vb, mask)
        torch.npu.synchronize()
        out_var = transpose_out(out_bhsd)
        torch.npu.synchronize()

        # ---- criterion 1: bit-identical vs shipped ----
        if boundary:
            bit_identical, why = _special_struct_equal(out_var, out_ref)
        else:
            bit_identical = bool(torch.equal(out_var, out_ref))
            why = "ok" if bit_identical else f"max_diff={float((out_var.float() - out_ref.float()).abs().max()):.3e}"

        # ---- criterion 2: suite tolerance vs golden (finite cases only) ----
        if boundary:
            max_diff, viol, bad = float("nan"), -1, -1  # golden compare = Gate 1 suite job
        else:
            golden = _ref_gqa(q, k, v, scale, causal)
            torch.npu.synchronize()
            of = out_var.float().cpu()
            gf = golden.float().cpu()
            atol = 1e-3 if dtype == torch.float16 else 2e-2
            rtol = 1e-2
            diff = (of - gf).abs()
            max_diff = float(diff.max())
            viol = int((diff > atol + rtol * gf.abs()).sum())
            bad = int((~torch.isfinite(of)).sum())
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        print(f"[G0_FAIL] {tag} error={e}", flush=True)
        return False
    if boundary:
        ok = bit_identical
        print(f"[{'G0_PASS' if ok else 'G0_FAIL'}] {tag} special_struct_vs_shipped={why}", flush=True)
        return ok
    ok = bit_identical and viol == 0 and bad == 0
    print(
        f"[{'G0_PASS' if ok else 'G0_FAIL'}] {tag} bit_identical={bit_identical}({why}) "
        f"golden_max_diff={max_diff:.3e} viol={viol} nonfinite={bad}",
        flush=True,
    )
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(VARIANT_MODULES))
    ap.add_argument("--case", default=None)
    args = ap.parse_args()
    cases = CASES if args.case is None else [c for c in CASES if c[0] == args.case]
    if args.case is None:
        cases = cases + [BOUNDARY_CASE]
    if not cases:
        print(f"unknown case {args.case}")
        sys.exit(2)
    results = [run_case(args.arm, c, boundary=c[0].startswith("boundary")) for c in cases]
    n_ok = sum(results)
    print(f"\n[G0_SUMMARY] arm={args.arm} {n_ok}/{len(results)} cases passed", flush=True)
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
