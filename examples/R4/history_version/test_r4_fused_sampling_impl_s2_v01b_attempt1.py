"""R4 fused sampling chain -- layered tests: L0/L1/L2/Boundary + main(--level).

Run (from examples/R4, so the script dir is sys.path[0] for sibling imports and the
repo root is NOT on the path -> the installed tilelang wheel is used):

    cd examples/R4 && python test_r4_fused_sampling.py --level l0

Precision standard (DESIGN.md section 9.3, mixed tolerance double-gate):
    float32: atol=2**-16, rtol=2**-10, max_abs_error_limit=1e-2, required_matched_ratio=0.99
    int32  : exact (0/0/0/1.0)

L0 data recipe (DESIGN.md section 9.2.1, MANDATORY): randn * LOGIT_SCALE(8.0). Any other
scale makes the nucleus exceed K_MAX=256 -> guard fires -> valid=0 -> false L0 failure
(NOT a kernel bug; check the recipe first).
"""

import argparse
import os
import sys
import time

import torch

# Sibling imports: the script dir is sys.path[0] when run as `python test_...py` from
# examples/R4. Insert explicitly too, so it works from any cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r4_fused_sampling import (  # noqa: E402
    r4_fused_sampling,
    r4_mask_sample,  # noqa: F401  (public kernel API, exercised via r4_fused_sampling)
    r4_topk_stats,  # noqa: F401  (public kernel API, exercised via r4_fused_sampling)
)
from r4_ref import golden_r4  # noqa: E402

LOGIT_SCALE = 8.0  # MANDATORY (DESIGN.md section 9.2.1)
K_MAX = 256
BV = 4096

# Operator category (DESIGN.md section 9.0): pure-Vector Fusion. Declared explicitly so
# coverage_check.py does not fall back to heuristics. Fusion => no dimension is exemptable.
COVERAGE_CATEGORY = "Fusion"


# ========== Precision standard (mixed tolerance, see tilelang-op-test-design/references/precision-standard.md) ==========
def get_precision(dtype):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio)."""
    fp_table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
        "hifloat32": (2**-16, 2**-10, 1e-2, 0.99),
        "float8_e4m3": (2**-4, 2**-2, 1e0, 0.99),
        "float8_e5m2": (2**-3, 2**-1, 1e-1, 0.99),
    }
    if dtype in {"int8", "int16", "int32", "int64", "uint8"}:
        return (0.0, 0.0, 0.0, 1.0)
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    """Mixed-tolerance double gate -> (passed, matched_ratio, max_abs_error)."""
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a, g = actual.detach().cpu(), golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:  # integer exact match
        mism = (a != g).sum().item()
        return mism == 0, 1.0 - mism / max(a.numel(), 1), (0.0 if mism == 0 else float("inf"))
    a, g = a.float(), g.float()
    # Position-sensitive: NaN / +Inf / -Inf masks must match exactly, then compare finite.
    nan_a, nan_g = torch.isnan(a), torch.isnan(g)
    pinf_a, pinf_g = torch.isposinf(a), torch.isposinf(g)
    ninf_a, ninf_g = torch.isneginf(a), torch.isneginf(g)
    special_mismatch = (nan_a != nan_g) | (pinf_a != pinf_g) | (ninf_a != ninf_g)
    both_finite = torch.isfinite(a) & torch.isfinite(g)
    m = both_finite
    n_special = int(special_mismatch.sum().item())
    if m.sum().item() == 0:
        return n_special == 0, (1.0 if n_special == 0 else 0.0), (0.0 if n_special == 0 else float("inf"))
    abs_err = (a[m] - g[m]).abs()
    matched = abs_err <= (atol + rtol * g[m].abs())
    ratio = matched.float().mean().item()
    max_abs = abs_err.max().item()
    # special-position mismatches count against the ratio budget
    total = a.numel()
    ratio = (matched.sum().item()) / total if total else 1.0
    passed = (ratio >= required_ratio) and (max_abs <= max_abs_limit) and (n_special == 0)
    return passed, ratio, max_abs


# ========== L0 test cases (DESIGN.md section 9.2.2, 8 cases) ==========
# (name, B, V, k, p, mu, need_min_p)
L0_CASES = [
    ("l0_min_single_block", 2, 4096, 40, 1.0, 0.0, False),
    ("l0_multiblock_topk", 8, 16384, 40, 1.0, 0.0, False),
    ("l0_topp_nucleus", 8, 16384, 40, 0.95, 0.0, False),
    ("l0_minp_T2", 8, 16384, 40, 0.95, 0.05, True),
    ("l0_prod_v_tail", 16, 152064, 40, 0.95, 0.05, True),
    ("l0_full_occupancy", 40, 152064, 20, 0.9, 0.1, True),
    ("l0_k_eq_1", 2, 16384, 1, 1.0, 0.0, False),
    ("l0_k_eq_kmax", 2, 16384, 256, 1.0, 0.0, False),
]


def _make_inputs(B, V, k, p, mu):
    torch.manual_seed(0)
    logits = torch.randn(B, V, dtype=torch.float32) * LOGIT_SCALE
    top_ks = torch.full((B,), k, dtype=torch.int32)
    top_ps = torch.full((B,), p, dtype=torch.float32)
    min_ps = torch.full((B,), mu, dtype=torch.float32)
    return logits, top_ks, top_ps, min_ps


def _run_case(name, B, V, k, p, mu, need_min_p, level_tag):
    """Run one case, compare all 5 outputs. Return (ok, detail_str)."""
    logits, top_ks, top_ps, min_ps = _make_inputs(B, V, k, p, mu)
    t0 = time.time()
    lg = logits.npu()
    m_out, s_out, tau_out, valid, w_out = r4_fused_sampling(lg, top_ks.npu(), top_ps.npu(), min_ps.npu(), need_min_p, K_MAX, BV)
    torch.npu.synchronize()
    m_out, s_out, tau_out, valid, w_out = (m_out.cpu(), s_out.cpu(), tau_out.cpu(), valid.cpu(), w_out.cpu())
    run_s = time.time() - t0
    m_ref, s_ref, tau_ref, valid_ref, w_ref = golden_r4(logits, top_ks, top_ps, min_ps, need_min_p, K_MAX)

    results = {}
    results["m_out"] = check_precision(m_out, m_ref, "float32")
    results["s_out"] = check_precision(s_out, s_ref, "float32")
    results["tau_out"] = check_precision(tau_out, tau_ref, "float32")
    results["valid"] = check_precision(valid, valid_ref, "int32")
    results["w_out"] = check_precision(w_out, w_ref, "float32")

    # valid_ref must be all-1 under the LOGIT_SCALE=8.0 recipe (guard must not fire).
    valid_ref_all1 = bool((valid_ref == 1).all().item())
    ok = all(r[0] for r in results.values()) and valid_ref_all1

    detail = (
        f"{name} B={B} V={V} k={k} p={p} mu={mu} min_p={need_min_p} ({run_s:.1f}s) "
        + " ".join(f"{o}[ratio={results[o][1]:.4f},max={results[o][2]:.2e}]" for o in ("m_out", "s_out", "tau_out", "valid", "w_out"))
        + ("" if valid_ref_all1 else " !!valid_ref_not_all_1(check LOGIT_SCALE recipe)")
    )
    return ok, detail


def test_r4_fused_sampling_l0():
    """L0 gate: 8 cases x 5 outputs = 40 comparisons (DESIGN.md section 9.2)."""
    print("=" * 80)
    print("[L0] R4 gate test -- 8 cases x 5 outputs (LOGIT_SCALE=8.0)")
    print("=" * 80)
    all_ok = True
    for case in L0_CASES:
        name = case[0]
        try:
            ok, detail = _run_case(*case, level_tag="L0")
            tag = "PASS" if ok else "FAIL"
            print(f"[PRECISION_{tag}] {detail}")
            all_ok &= ok
        except Exception as e:  # noqa: BLE001
            import traceback

            print(f"[PRECISION_FAIL] {name}: {type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc()
            all_ok = False
    return all_ok


# ========== shared runner: run kernel + golden, compare all 5 outputs ==========
def _compare_outputs(name, B, V, kmax, bv, k, p, mu, nmp, logits, expect_valid=None):
    """Run r4_fused_sampling + golden_r4, compare all 5 outputs.

    expect_valid: None -> require valid_ref all-1 (fast path must be trusted);
                  0 -> require valid all-0 on BOTH sides (guard must fire);
                  1 -> require valid all-1 on both sides.
    Returns (ok, detail_str, results_dict).
    """
    top_ks = torch.full((B,), k, dtype=torch.int32)
    top_ps = torch.full((B,), p, dtype=torch.float32)
    min_ps = torch.full((B,), mu, dtype=torch.float32)
    t0 = time.time()
    m_out, s_out, tau_out, valid, w_out = r4_fused_sampling(logits.npu(), top_ks.npu(), top_ps.npu(), min_ps.npu(), nmp, kmax, bv)
    torch.npu.synchronize()
    m_out, s_out, tau_out, valid, w_out = (m_out.cpu(), s_out.cpu(), tau_out.cpu(), valid.cpu(), w_out.cpu())
    run_s = time.time() - t0
    m_ref, s_ref, tau_ref, valid_ref, w_ref = golden_r4(logits, top_ks, top_ps, min_ps, nmp, kmax)
    res = {
        "m_out": check_precision(m_out, m_ref, "float32"),
        "s_out": check_precision(s_out, s_ref, "float32"),
        "tau_out": check_precision(tau_out, tau_ref, "float32"),
        "valid": check_precision(valid, valid_ref, "int32"),
        "w_out": check_precision(w_out, w_ref, "float32"),
    }
    ok = all(r[0] for r in res.values())
    note = ""
    if expect_valid is None:
        if not bool((valid_ref == 1).all().item()):
            ok = False
            note = " !!valid_ref_not_all_1"
    elif expect_valid == 0 and not (bool((valid_ref == 0).all().item()) and bool((valid == 0).all().item())):
        ok = False
        note = f" !!guard_not_fired(ref0={int((valid_ref == 0).sum())},out0={int((valid == 0).sum())}/{B})"
    detail = (
        f"{name} B={B} V={V} K_MAX={kmax} BV={bv} k={k} p={p} mu={mu} min_p={nmp} "
        f"({run_s:.1f}s) " + " ".join(f"{o}[r={res[o][1]:.4f},m={res[o][2]:.1e}]" for o in res) + note
    )
    return ok, detail, res


# ========== L1 functional cases (scenario B). tags = coverage dimension IDs ==========
# (name, B, V, K_MAX, BV, k, p, mu, need_min_p, scale, offset, tags)
# NOTE: top_k/top_p/min_p are RUNTIME tensor values (not JIT keys), so param coverage
# is cheap; only (B, V, K_MAX, BV, NEED_MIN_P) trigger recompiles (DESIGN.md R-JIT).
# Every case was CPU-pre-verified to yield valid_ref all-1 (fast path trusted).
L1_CASES = [
    (
        "l1_aligned_base",
        8,
        16384,
        256,
        4096,
        40,
        0.95,
        0.0,
        False,
        8.0,
        0.0,
        ["D-DTYPE-fp32", "D-DTYPE-int32", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-top_k", "D-PARAM-top_p"],
    ),
    ("l1_aligned_p1", 8, 16384, 256, 4096, 40, 1.0, 0.0, False, 8.0, 0.0, ["D-PARAM-top_p"]),
    (
        "l1_minp_on",
        8,
        16384,
        256,
        4096,
        40,
        0.95,
        0.05,
        True,
        8.0,
        0.0,
        ["D-PARAM-NEED_MIN_P", "D-PARAM-need_min_p_sampling", "D-PARAM-min_p"],
    ),
    ("l1_minp_strong", 8, 16384, 256, 4096, 60, 0.9, 0.2, True, 8.0, 0.0, ["D-PARAM-min_p", "D-PARAM-top_k", "D-PARAM-top_p"]),
    ("l1_valrange_S", 4, 4096, 256, 4096, 40, 0.95, 0.0, False, 5.0, 0.0, ["D-VALRANGE-S"]),
    ("l1_valrange_L", 4, 16384, 256, 4096, 40, 0.95, 0.0, False, 1000.0, 0.0, ["D-VALRANGE-L"]),
    ("l1_valrange_L_huge", 4, 4096, 256, 4096, 20, 0.9, 0.0, False, 1e4, 0.0, ["D-VALRANGE-L", "D-PARAM-top_k"]),
    ("l1_valrange_ASYM", 4, 4096, 256, 4096, 40, 0.95, 0.0, False, 6.0, -4.0, ["D-VALRANGE-ASYM"]),
    # B=5 -> B % VEC_NUM(2) == 1: batch-axis tail remainder 1 (row guard).
    ("l1_tail1_oddB", 5, 16384, 256, 4096, 40, 0.95, 0.0, False, 8.0, 0.0, ["D-SHAPE-TAIL-1"]),
    # V=10240 -> V % BV == 2048 == BV//2: vocab-axis mid tail.
    ("l1_tailmid_V", 8, 10240, 256, 4096, 40, 0.95, 0.0, False, 8.0, 0.0, ["D-SHAPE-TAIL-MID"]),
    # V=4409 prime (V % BV == 313 >= K_MAX, C3 ok); B=7 prime.
    ("l1_prime_V_B", 7, 4409, 256, 4096, 40, 0.95, 0.0, False, 8.0, 0.0, ["D-SHAPE-PRIME"]),
    ("l1_edge_B1", 1, 4096, 256, 4096, 40, 0.95, 0.0, False, 8.0, 0.0, ["D-SHAPE-EDGE"]),
    # V=K_MAX=256 -> n_blk_full=0, only the tail block runs (degenerate minimum V).
    ("l1_edge_V_eq_KMAX", 2, 256, 256, 4096, 40, 0.95, 0.0, False, 8.0, 0.0, ["D-SHAPE-EDGE"]),
    ("l1_kmax_128", 4, 16384, 128, 4096, 100, 0.95, 0.0, False, 8.0, 0.0, ["D-PARAM-K_MAX", "D-PARAM-top_k"]),
    ("l1_kmax_128_keq", 4, 16384, 128, 4096, 128, 1.0, 0.0, False, 8.0, 0.0, ["D-PARAM-K_MAX", "D-PARAM-top_k", "D-PARAM-top_p"]),
    ("l1_bv_2048", 4, 16384, 256, 2048, 40, 0.95, 0.0, False, 8.0, 0.0, ["D-PARAM-BV"]),
    ("l1_k_eq_1", 4, 4096, 256, 4096, 1, 1.0, 0.0, False, 8.0, 0.0, ["D-PARAM-top_k", "D-PARAM-top_p"]),
    ("l1_prod_v_minp", 4, 152064, 256, 4096, 30, 0.95, 0.1, True, 8.0, 0.0, ["D-PARAM-min_p", "D-PARAM-top_k", "D-VALRANGE-M"]),
]


def test_r4_fused_sampling_l1():
    """L1 functional tests: parameter/shape/value-range coverage (blocking)."""
    print("=" * 80)
    print(f"[L1] R4 functional tests -- {len(L1_CASES)} cases")
    print("=" * 80)
    all_ok = True
    for name, B, V, kmax, bv, k, p, mu, nmp, scale, offset, _tags in L1_CASES:
        try:
            torch.manual_seed(0)
            logits = torch.randn(B, V, dtype=torch.float32) * scale + offset
            ok, detail, _ = _compare_outputs(name, B, V, kmax, bv, k, p, mu, nmp, logits)
            print(f"[PRECISION_{'PASS' if ok else 'FAIL'}] {detail}")
            all_ok &= ok
        except Exception as e:  # noqa: BLE001
            print(f"[PRECISION_FAIL] l1 {name}: {type(e).__name__}: {str(e)[:200]}")
            all_ok = False
    return all_ok


# ========== L2 negative tests (non-blocking): illegal input must be rejected ==========
def _run_exception(name, fn, tags):
    """Illegal input must be rejected: raises -> [BOUNDARY_PASS]; silent -> [BOUNDARY_WARN]."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print(f"[BOUNDARY_PASS] l2 {name}: correctly rejected ({type(e).__name__}) {tags}")
        return
    print(f"[BOUNDARY_WARN] l2 {name}: illegal input silently accepted {tags}")


def _run_guard(name, B, V, kmax, bv, k, p, mu, nmp, scale, offset, tags):
    """Out-of-fast-path input must be rejected via valid==0 (host then falls back)."""
    try:
        torch.manual_seed(0)
        logits = torch.randn(B, V, dtype=torch.float32) * scale + offset
        _m, _s, _t, valid, _w = r4_fused_sampling(
            logits.npu(),
            torch.full((B,), k, dtype=torch.int32).npu(),
            torch.full((B,), p, dtype=torch.float32).npu(),
            torch.full((B,), mu, dtype=torch.float32).npu(),
            nmp,
            kmax,
            bv,
        )
        torch.npu.synchronize()
        valid = valid.cpu()
        if bool((valid == 0).all().item()):
            print(f"[BOUNDARY_PASS] l2 {name}: guard fired, valid all-0 {tags}")
        else:
            print(f"[BOUNDARY_WARN] l2 {name}: guard did NOT fire (valid ones={int((valid == 1).sum())}/{B}) {tags}")
    except Exception as e:  # noqa: BLE001
        print(f"[BOUNDARY_WARN] l2 {name}: raised instead of guarding ({type(e).__name__}: {str(e)[:100]}) {tags}")


def test_r4_fused_sampling_l2():
    """L2 negative tests: unsupported dtype / illegal shape / out-of-fast-path params."""
    print("=" * 80)
    print("[L2] R4 negative tests (non-blocking)")
    print("=" * 80)
    base = torch.randn(4, 4096, dtype=torch.float32) * 8.0
    ks = torch.full((4,), 40, dtype=torch.int32)
    ps = torch.full((4,), 0.95, dtype=torch.float32)
    mus = torch.full((4,), 0.0, dtype=torch.float32)

    # -- host pre-check must raise (C7 / C3 / dtype) --
    _run_exception(
        "exc_dtype_bf16",
        lambda: r4_fused_sampling(base.to(torch.bfloat16).npu(), ks.npu(), ps.npu(), mus.npu(), False, 256, 4096),
        ["D-EXC-DTYPE"],
    )
    _run_exception(
        "exc_dtype_fp16",
        lambda: r4_fused_sampling(base.to(torch.float16).npu(), ks.npu(), ps.npu(), mus.npu(), False, 256, 4096),
        ["D-EXC-DTYPE"],
    )
    _run_exception(
        "exc_V_lt_KMAX",
        lambda: r4_fused_sampling(
            torch.randn(2, 128, dtype=torch.float32).npu(),
            torch.full((2,), 40, dtype=torch.int32).npu(),
            torch.full((2,), 0.95, dtype=torch.float32).npu(),
            torch.full((2,), 0.0, dtype=torch.float32).npu(),
            False,
            256,
            4096,
        ),
        ["D-EXC-SHAPE"],
    )
    # V=4097 -> V % BV == 1 < K_MAX: tail block could not hold K_MAX (C3).
    _run_exception(
        "exc_tail_lt_KMAX",
        lambda: r4_fused_sampling(
            torch.randn(2, 4097, dtype=torch.float32).npu(),
            torch.full((2,), 40, dtype=torch.int32).npu(),
            torch.full((2,), 0.95, dtype=torch.float32).npu(),
            torch.full((2,), 0.0, dtype=torch.float32).npu(),
            False,
            256,
            4096,
        ),
        ["D-EXC-SHAPE"],
    )

    # -- out-of-fast-path sampling params: guard must fire (valid all-0) --
    _run_guard("guard_k_TOP_K_ALL", 4, 4096, 256, 4096, 1 << 30, 0.95, 0.0, False, 8.0, 0.0, ["D-PARAM-top_k"])
    _run_guard("guard_k_zero", 4, 4096, 256, 4096, 0, 0.95, 0.0, False, 8.0, 0.0, ["D-PARAM-top_k"])
    _run_guard("guard_k_gt_KMAX", 4, 4096, 256, 4096, 257, 0.95, 0.0, False, 8.0, 0.0, ["D-PARAM-top_k"])
    # Flat distribution (scale 1.0): nucleus >> K_MAX -> ok_p=0 (DESIGN.md R-KMAX).
    _run_guard("guard_flat_dist", 4, 16384, 256, 4096, 40, 0.95, 0.0, False, 1.0, 0.0, ["D-VALRANGE-S"])

    # -- B == 0: host intercept (C6), must not launch a zero-sized grid --
    try:
        outs = r4_fused_sampling(
            torch.empty(0, 4096, dtype=torch.float32).npu(),
            torch.empty(0, dtype=torch.int32).npu(),
            torch.empty(0, dtype=torch.float32).npu(),
            torch.empty(0, dtype=torch.float32).npu(),
            False,
            256,
            4096,
        )
        if all(o.numel() == 0 for o in outs):
            print('[BOUNDARY_PASS] l2 guard_empty_B0: host intercept returned empty tensors ["D-EXC-SHAPE"]')
        else:
            print('[BOUNDARY_WARN] l2 guard_empty_B0: non-empty output for B=0 ["D-EXC-SHAPE"]')
    except Exception as e:  # noqa: BLE001
        print(f'[BOUNDARY_WARN] l2 guard_empty_B0: raised ({type(e).__name__}) ["D-EXC-SHAPE"]')


# ========== Boundary tests (non-blocking): legal special values, precision compared ==========
def _run_boundary(name, dtype, fn, tags):
    """Legal special values -> compare by the precision standard; fail => WARN."""
    try:
        out, ref = fn()
        passed, ratio, max_abs = check_precision(out, ref, dtype)
        print(
            f"[BOUNDARY_{'PASS' if passed else 'WARN'}] boundary {name} dtype={dtype} "
            f"matched_ratio={ratio:.4f} max_abs={max_abs:.3e} {tags}"
        )
    except Exception as e:  # noqa: BLE001
        print(f"[BOUNDARY_WARN] boundary {name} dtype={dtype}: {type(e).__name__}: {str(e)[:120]} {tags}")


def _bnd_case(name, logits, k, p, mu, nmp, out_key, dtype, tags, kmax=256, bv=4096):
    """Build a Boundary lambda: run kernel+golden, return the chosen output pair."""
    B, V = logits.shape

    def fn():
        top_ks = torch.full((B,), k, dtype=torch.int32)
        top_ps = torch.full((B,), p, dtype=torch.float32)
        min_ps = torch.full((B,), mu, dtype=torch.float32)
        m_o, s_o, t_o, v_o, w_o = r4_fused_sampling(logits.npu(), top_ks.npu(), top_ps.npu(), min_ps.npu(), nmp, kmax, bv)
        torch.npu.synchronize()
        got = {"m_out": m_o, "s_out": s_o, "tau_out": t_o, "valid": v_o, "w_out": w_o}[out_key].cpu()
        m_r, s_r, t_r, v_r, w_r = golden_r4(logits, top_ks, top_ps, min_ps, nmp, kmax)
        exp = {"m_out": m_r, "s_out": s_r, "tau_out": t_r, "valid": v_r, "w_out": w_r}[out_key]
        return got, exp

    _run_boundary(name, dtype, fn, tags)


def test_r4_fused_sampling_boundary():
    """Boundary: sparse -inf / fp32 bounds / all-zero / all-equal / sparse NaN.

    Per DESIGN.md 9.3 the input contract allows -inf (upstream masking) but NOT NaN
    (upstream sanitize_nan_logits), so NaN is expected to WARN (undefined semantics).
    Degenerate rows (all-zero / all-equal) must fire the guard -> valid all-0.
    """
    print("=" * 80)
    print("[BOUNDARY] R4 special-value tests (non-blocking)")
    print("=" * 80)
    torch.manual_seed(0)
    B, V = 4, 4096

    # Finite values + SPARSE -inf (never an all -inf row): DESIGN.md 9.3 position contract.
    lg_inf = torch.randn(B, V, dtype=torch.float32) * 8.0
    lg_inf[0, :50] = float("-inf")
    lg_inf[2, 100] = float("-inf")
    for key in ("m_out", "s_out", "tau_out", "valid", "w_out"):
        _bnd_case(
            f"bnd_sparse_neginf_{key}", lg_inf, 40, 0.95, 0.0, False, key, "int32" if key == "valid" else "float32", ["D-SPECIAL-INF"]
        )

    # fp32 / exp boundary magnitudes (online softmax keeps exp(z-m) <= 1, so stable).
    lg_bnd = torch.randn(B, V, dtype=torch.float32) * 8.0
    lg_bnd[0, 0] = 88.0
    lg_bnd[0, 1] = -88.0
    lg_bnd[1, 5] = 1e4
    lg_bnd[1, 6] = -1e4
    for key in ("m_out", "s_out", "tau_out", "valid", "w_out"):
        _bnd_case(f"bnd_dbound_{key}", lg_bnd, 40, 0.95, 0.0, False, key, "int32" if key == "valid" else "float32", ["D-SPECIAL-DBOUND"])

    # All-zero and all-equal rows: uniform / fully tied -> guard must fire (valid all-0).
    _bnd_case("bnd_all_zero_valid", torch.zeros(B, V, dtype=torch.float32), 40, 0.95, 0.0, False, "valid", "int32", ["D-SPECIAL-ZERO"])
    _bnd_case(
        "bnd_all_equal_valid", torch.full((B, V), 3.5, dtype=torch.float32), 40, 0.95, 0.0, False, "valid", "int32", ["D-SPECIAL-ZERO"]
    )

    # Sparse NaN: unsupported input (DESIGN.md 5.7) -> semantics undefined, WARN expected.
    lg_nan = torch.randn(B, V, dtype=torch.float32) * 8.0
    lg_nan[1, 7] = float("nan")
    lg_nan[3, 11] = float("nan")
    _bnd_case("bnd_sparse_nan_w_out", lg_nan, 40, 0.95, 0.0, False, "w_out", "float32", ["D-SPECIAL-NAN"])


# ========== Coverage summary (coverage_check.py reads tags above + these) ==========
# Fusion category => every dimension is mandatory (exempt set is empty), so COVERAGE_NA
# must stay empty: all 24 dimensions are hit by real cases above.
COVERAGE_MANIFEST = {
    "D-DTYPE-fp32": 1,
    "D-DTYPE-int32": 1,
    "D-SHAPE-ALIGNED": 1,
    "D-SHAPE-TAIL-1": 1,
    "D-SHAPE-TAIL-MID": 1,
    "D-SHAPE-PRIME": 1,
    "D-SHAPE-EDGE": 2,
    "D-VALRANGE-S": 2,
    "D-VALRANGE-M": 2,
    "D-VALRANGE-L": 2,
    "D-VALRANGE-ASYM": 1,
    "D-PARAM-K_MAX": 2,
    "D-PARAM-BV": 1,
    "D-PARAM-NEED_MIN_P": 1,
    "D-PARAM-need_min_p_sampling": 1,
    "D-PARAM-top_k": 6,
    "D-PARAM-top_p": 4,
    "D-PARAM-min_p": 3,
    "D-SPECIAL-INF": 5,
    "D-SPECIAL-NAN": 1,
    "D-SPECIAL-ZERO": 2,
    "D-SPECIAL-DBOUND": 5,
    "D-EXC-DTYPE": 2,
    "D-EXC-SHAPE": 3,
}
COVERAGE_NA = {}


# ========== main: --level dispatch + exit code ==========
def main():
    parser = argparse.ArgumentParser(description="R4 fused sampling layered tests")
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()

    # NOTE: DESIGN.md section 11.5.6 (R-JIT mitigation) -- do NOT call
    # tilelang.disable_cache()/clear_cache() here; B is part of the JIT key and a
    # persistent TILELANG_CACHE_DIR avoids recompilation storms. Set the env var instead:
    #   export TILELANG_CACHE_DIR=/tmp/opencode/r4_tl_cache
    torch.manual_seed(0)

    blocking_ok = True  # only L0/L1 gate the exit code
    if args.level in ("l0", "all"):
        blocking_ok &= test_r4_fused_sampling_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_r4_fused_sampling_l1()
    if args.level in ("l2", "all"):
        test_r4_fused_sampling_l2()  # negative: non-blocking
    if args.level in ("boundary", "all"):
        test_r4_fused_sampling_boundary()  # special values: non-blocking

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
