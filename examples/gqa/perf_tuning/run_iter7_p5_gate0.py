"""Gate 0 runner — iter7 P5 (flat_kv2 neg_sm cap-refill narrowing) precision.

Standalone: does NOT modify examples/gqa/gqa.py or test_gqa.py.  Imports the
gqa module read-only; the variant kernel is generated from the SHIPPED
flat_kv2 block (gen_variant_iter7_p5.py, diff_iter7_p5.txt).

Reference arm = the SHIPPED dispatch (g2stack_flat_kv2 — the iter5-adopted
kernel; note this differs from iter5's runner whose reference was g2stack).
The P5 edit is a bijective slot remap + dead-store fill narrowing (pure
scheduling/addressing, zero arithmetic change) => outputs must be
BIT-IDENTICAL to shipped flat_kv2.  Special-value discipline (iter2 lesson):
NaN positions equal, Inf positions+signs equal, finite entries bit-equal —
the iter5 dual-kv-block injection case AND a fullrange (fp16 [-65504,
65504]) case exercise the real rescale chain carrying specials.

Three layers (iter5 pattern):
  1. DIRECT-BUILD bit-identity: variant factory built directly, adapter path
     (transpose_qkv -> variant -> transpose_out).
  2. DISPATCH-REDIRECT: monkeypatch G._gqa_kernel_prefill_g2stack_flat_kv2
     -> variant; must be INVOKED, chosen == ["g2stack_flat_kv2"], zero
     fallback, bit-identical.
  3. GATE ISOLATION (causal kv2 + stacked<40): with the SAME patch installed
     the flat_kv2 spec is never built (gate fails before the lambda) —
     variant NEVER invoked, chosen == ["g2stack"], outputs bit-identical.
  Golden tolerance (viol=0; fp16 atol 1e-3 / bf16 2e-2, rtol 1e-2) checked
  on all finite-input cases.

Run (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python3 examples/gqa/perf_tuning/run_iter7_p5_gate0.py [--case NAME]
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

import gqa as G  # noqa: E402
from gqa import (  # noqa: E402
    P28_NUM_CORES,
    _BIG_NEG,
    _dummy_mask,
    _ref_gqa,
    transpose_out,
    transpose_qkv,
)

import gqa_flat_kv2_p5_variant  # noqa: E402

VARIANT_FACTORY = gqa_flat_kv2_p5_variant._gqa_kernel_prefill_g2stack_flat_kv2


def _clear_caches():
    G._KERNEL_BUILD_CACHE.clear()
    G._COMPILED_CACHE.clear()


# (name, B, S, S_kv, N_q, N_kv, D, causal, dtype) — all inside the flat_kv2 gate
CASES = [
    ("r4_c4like_bf16", 4, 256, 256, 64, 8, 128, False, torch.bfloat16),     # bench c4, PRIMARY A/B shape
    ("c7like_bf16", 2, 128, 256, 32, 8, 128, False, torch.bfloat16),        # bench c7, secondary A/B shape
    ("mc2_edge_noncausal_fp16", 5, 128, 256, 8, 2, 128, False, torch.float16),   # stacked=40 gate edge, mc=2/0
    ("mixed_mc32_noncausal_fp16", 3, 128, 256, 16, 4, 128, False, torch.float16),  # stacked=48, mc=3/2 (r=8)
    ("mixed_mc76_noncausal_fp16", 2, 256, 256, 32, 8, 128, False, torch.float16),  # stacked=128, mc=7/6 (r=8)
    ("padcol_noncausal_fp16", 2, 128, 200, 32, 8, 128, False, torch.float16),      # Skv pad cols (use_mask)
    ("padrow_noncausal_fp16", 2, 100, 256, 32, 8, 128, False, torch.float16),      # S pad rows (no mask)
]

# Boundary sparse-special case (iter5 verbatim): c7-like (G=4) with injections
# into BOTH kv blocks so the real kv0->kv1 rescale chain carries specials:
#   q[0, 5, 3, 0]   = NaN   -> out[0, 5, 3, :]    all NaN (scores NaN both kv)
#   v[0, 20, 1, 7]  = +Inf  -> out[0, :, 4:8, 7]  all +Inf (kv block 0)
#   v[1, 200, 5, 9] = -Inf  -> out[1, :, 20:24, 9] all -Inf (kv block 1)
BOUNDARY_CASE = ("boundary_sparse_c7like", 2, 128, 256, 32, 8, 128, False, torch.bfloat16)

# Fullrange case (task-book requirement): fp16 uniform over [-65504, 65504] —
# near-argmax softmax, |scores| up to ~5e10 in fp32: the neg_sm running max
# and the cap semantics (neg_sm_cap = -sm_scale*_BIG_NEG/2 ~ 4.4e28) are
# exercised at the dtype extremes (test_gqa boundary_fullrange semantics).
FULLRANGE_CASE = ("fullrange_r4like_fp16", 4, 256, 256, 64, 8, 128, False, torch.float16)

# Gate-isolation cases: redirect patch installed, variant must NOT fire.
ISO_CASES = [
    ("iso_causal_kv2_g2stack", 4, 256, 256, 64, 8, 128, True, torch.bfloat16),   # causal -> shipped g2stack
    ("iso_smallstack_g2stack", 1, 128, 256, 32, 8, 128, False, torch.float16),   # stacked=32 in [20,40)
]


def _padded_causal_mask(B, S, S_kv, S_pad, Skv_pad, device):
    mask = torch.full((S_pad, Skv_pad), _BIG_NEG, dtype=torch.float32)
    i = torch.arange(S).unsqueeze(-1)
    j = torch.arange(S_kv).unsqueeze(0)
    visible = j <= (i + (S_kv - S))
    mask[:S, :S_kv][visible] = 0.0
    return mask.unsqueeze(0).expand(B, S_pad, Skv_pad).contiguous().to(device)


def _padcol_mask(B, S, S_kv, S_pad, Skv_pad, device):
    assert S == S_pad
    mask = torch.full((S_pad, Skv_pad), _BIG_NEG, dtype=torch.float32)
    mask[:, :S_kv] = 0.0
    return mask.unsqueeze(0).expand(B, S_pad, Skv_pad).contiguous().to(device)


def make_redirect(invoked):
    """Monkeypatch the flat_kv2 spec builder itself: the dispatch gate
    (kv==2 & noncausal & stacked>=40) is evaluated BEFORE the lambda is
    built, so gate-failing shapes never reach the patched attribute."""
    orig = G._gqa_kernel_prefill_g2stack_flat_kv2

    def patched(B, S_padded, S_kv_padded, sm_scale, is_causal, N_q=8, N_kv=2, D=128,
                dtype="float16", block_M=64, skv_minus_s=0, use_mask=True, block_N=128):
        invoked.append(True)
        return VARIANT_FACTORY(B, S_padded, S_kv_padded, sm_scale, is_causal,
                               N_q=N_q, N_kv=N_kv, D=D, dtype=dtype, block_M=block_M,
                               skv_minus_s=skv_minus_s, use_mask=use_mask, block_N=block_N)

    return patched, orig


def _inputs(name, B, S, S_kv, N_q, N_kv, D, causal, dtype):
    if name.startswith("fullrange"):
        g = torch.Generator().manual_seed(3101)
        lo, hi = -65504.0, 65504.0

        def gen(shape, seed_off):
            gg = torch.Generator().manual_seed(3101 + seed_off)
            t = torch.rand(shape, dtype=torch.float64, generator=gg) * (hi - lo) + lo
            return t.to(dtype).npu()

        del g
        return gen((B, S, N_q, D), 0), gen((B, S_kv, N_kv, D), 1), gen((B, S_kv, N_kv, D), 2)
    torch.manual_seed(42 + B + S + S_kv + N_q)
    q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
    k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
    v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
    if name.startswith("boundary_sparse"):
        q[0, 5, 3, 0] = float("nan")
        v[0, 20, 1, 7] = float("inf")
        v[1, 200, 5, 9] = float("-inf")
    return q, k, v


def _special_struct_equal(a, b):
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


def _run_variant_direct(q, k, v, B, S, S_kv, N_q, N_kv, D, causal, dtype):
    """transpose_qkv -> variant kernel -> transpose_out (gqa() adapter path)."""
    S_pad = ((S + 63) // 64) * 64
    Skv_pad = ((S_kv + 127) // 128) * 128
    kv_iters = Skv_pad // 128
    Gg = N_q // N_kv
    block_num = B * N_kv * (Gg // 2) * (S_pad // 64)
    eligible = (D == 128 and kv_iters == 2 and not causal and Gg % 2 == 0
                and block_num >= 2 * P28_NUM_CORES)
    if not eligible:
        raise ValueError(f"case outside flat-kv2 gate: B={B} S={S} S_kv={S_kv} Nq={N_q} Nkv={N_kv}")
    res = transpose_qkv(q, k, v, S_out=S_pad, S_kv_out=Skv_pad)
    assert res is not None, "transpose_qkv fallback (None) — unexpected on NPU"
    qb, kb, vb = res
    dtype_str = "float16" if dtype == torch.float16 else "bfloat16"
    sm_scale = 1.0 / (D**0.5)
    use_mask = bool(causal or (Skv_pad != S_kv))
    kernel_fn = VARIANT_FACTORY(B, S_pad, Skv_pad, sm_scale, causal,
                                N_q=N_q, N_kv=N_kv, D=D, dtype=dtype_str, block_M=64,
                                skv_minus_s=S_kv - S, use_mask=use_mask, block_N=128)
    if use_mask:
        mask = _padcol_mask(B, S, S_kv, S_pad, Skv_pad, q.device) if S == S_pad else \
            _padded_causal_mask(B, S, S_kv, S_pad, Skv_pad, q.device)
    else:
        mask = _dummy_mask(q.device)
    out_bhsd = kernel_fn(qb, kb, vb, mask)
    torch.npu.synchronize()
    out = transpose_out(out_bhsd, s_real=S) if S != S_pad else transpose_out(out_bhsd)
    torch.npu.synchronize()
    return out


def _golden_check(out, q, k, v, causal, dtype):
    golden = _ref_gqa(q, k, v, -1.0, causal)
    torch.npu.synchronize()
    of = out.float().cpu()
    gf = golden.float().cpu()
    atol = 1e-3 if dtype == torch.float16 else 2e-2
    rtol = 1e-2
    finite = torch.isfinite(of) & torch.isfinite(gf)
    diff = (of - gf).abs()
    max_diff = float(diff[finite].max()) if bool(finite.any()) else float("nan")
    viol = int((diff[finite] > atol + rtol * gf.abs()[finite]).sum())
    return max_diff, viol


def run_case(case, boundary=False):
    name, B, S, S_kv, N_q, N_kv, D, causal, dtype = case
    dtag = "fp16" if dtype == torch.float16 else "bf16"
    tag = f"{name} B={B} S={S} S_kv={S_kv} Nq={N_q} Nkv={N_kv} causal={causal} {dtag}"
    try:
        q, k, v = _inputs(name, B, S, S_kv, N_q, N_kv, D, causal, dtype)

        # ---- reference: shipped dispatch (flat_kv2 on all these shapes) ----
        _clear_caches()
        out_ref = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
        torch.npu.synchronize()
        chosen_ref = [val[2] for val in G._KERNEL_BUILD_CACHE.values()]
        assert chosen_ref == ["g2stack_flat_kv2"], \
            f"reference dispatch = {chosen_ref}, expected ['g2stack_flat_kv2']"

        # ---- layer 1: direct-build variant ----
        out_var = _run_variant_direct(q, k, v, B, S, S_kv, N_q, N_kv, D, causal, dtype)

        # ---- layer 2: dispatch-redirect ----
        invoked = []
        patched, orig = make_redirect(invoked)
        G._gqa_kernel_prefill_g2stack_flat_kv2 = patched
        try:
            _clear_caches()
            out_disp = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
            torch.npu.synchronize()
            chosen = [val[2] for val in G._KERNEL_BUILD_CACHE.values()]
        finally:
            G._gqa_kernel_prefill_g2stack_flat_kv2 = orig
            _clear_caches()

        if boundary:
            eq1, why1 = _special_struct_equal(out_var, out_ref)
            eq2, why2 = _special_struct_equal(out_disp, out_ref)
            max_diff, viol = float("nan"), -1
        else:
            eq1 = bool(torch.equal(out_var, out_ref))
            why1 = "ok" if eq1 else f"max_diff={float((out_var.float() - out_ref.float()).abs().max()):.3e}"
            eq2 = bool(torch.equal(out_disp, out_ref))
            why2 = "ok" if eq2 else "MISMATCH"
            max_diff, viol = _golden_check(out_var, q, k, v, causal, dtype)
        disp_ok = bool(invoked) and chosen == ["g2stack_flat_kv2"] and eq2
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        print(f"[G0_FAIL] {tag} error={e}", flush=True)
        return False
    if boundary:
        ok = eq1 and disp_ok
        print(f"[{'G0_PASS' if ok else 'G0_FAIL'}] {tag} struct(direct)={why1} "
              f"struct(dispatch)={why2} invoked={bool(invoked)} chosen={chosen}", flush=True)
        return ok
    ok = eq1 and eq2 and disp_ok and viol == 0
    print(
        f"[{'G0_PASS' if ok else 'G0_FAIL'}] {tag} bit_direct={eq1}({why1}) "
        f"bit_dispatch={eq2}({why2}) invoked={bool(invoked)} chosen={chosen} "
        f"golden_max_diff={max_diff:.3e} viol={viol}",
        flush=True,
    )
    return ok


def run_iso(case):
    name, B, S, S_kv, N_q, N_kv, D, causal, dtype = case
    tag = f"{name} B={B} S={S} S_kv={S_kv} Nq={N_q} Nkv={N_kv} causal={causal}"
    try:
        torch.manual_seed(42 + B + S + S_kv + N_q)
        q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
        k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
        v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
        invoked = []
        patched, orig = make_redirect(invoked)
        G._gqa_kernel_prefill_g2stack_flat_kv2 = patched
        try:
            _clear_caches()
            out_p = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
            torch.npu.synchronize()
            chosen = [val[2] for val in G._KERNEL_BUILD_CACHE.values()]
        finally:
            G._gqa_kernel_prefill_g2stack_flat_kv2 = orig
        _clear_caches()
        out_s = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
        torch.npu.synchronize()
        _clear_caches()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        print(f"[G0_FAIL] {tag} error={e}", flush=True)
        return False
    iso_ok = (not invoked) and chosen == ["g2stack"]
    bit_ok = bool(torch.equal(out_p, out_s))
    ok = iso_ok and bit_ok
    print(
        f"[{'G0_PASS' if ok else 'G0_FAIL'}] {tag} variant_invoked={bool(invoked)}(must be False) "
        f"chosen={chosen} patched_vs_plain_bit_identical={bit_ok}",
        flush=True,
    )
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None)
    args = ap.parse_args()
    results = []
    for c in CASES:
        if args.case and c[0] != args.case:
            continue
        results.append(run_case(c))
    for c, bnd in ((BOUNDARY_CASE, True), (FULLRANGE_CASE, True)):
        if args.case and c[0] != args.case:
            continue
        results.append(run_case(c, boundary=bnd))
    for c in ISO_CASES:
        if args.case and c[0] != args.case:
            continue
        results.append(run_iso(c))
    n_ok = sum(results)
    print(f"\n[G0_SUMMARY] P5: {n_ok}/{len(results)} cases passed", flush=True)
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
