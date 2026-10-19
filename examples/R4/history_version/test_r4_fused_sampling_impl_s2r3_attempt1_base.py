"""R4 fused sampling chain -- layered tests: L0/L1/L2/Boundary/Stability + main(--level).

Run (from examples/R4, so the script dir is sys.path[0] for sibling imports and the
repo root is NOT on the path -> the installed tilelang wheel is used):

    cd examples/R4 && python test_r4_fused_sampling.py --level l0

Levels:
    l0 / l1        blocking (precision gate)
    stability      blocking repeat-execution gate (N>=20 bit-identical launches on the
                   high-risk shapes + a tail-adversary sentinel sweep); part of `all`
    l2 / boundary  non-blocking (negative + special-value); reported as BOUNDARY_PASS/WARN
    stress         optional deep stress (UB poisoner + 200 reps); NOT part of `all`
    all            l0 + l1 + stability + l2 + boundary

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
import tilelang  # noqa: E402
from tilelang import language as T  # noqa: E402

from r4_fused_sampling import (  # noqa: E402
    R4_PASS_CONFIGS,
    VEC_NUM,
    HASH_M1,
    HASH_M2,
    PRIME_W,
    r4_fused_sampling,
    r4_gumbel_sample,  # noqa: F401  (V01-b K2 Gumbel kernel, exercised via r4_fused_sampling)
    r4_mask_sample,  # noqa: F401  (public kernel API, exercised via r4_fused_sampling)
    r4_topk_stats,  # noqa: F401  (public kernel API, exercised via r4_fused_sampling)
)
from r4_ref import (  # noqa: E402
    golden_gumbel_sample,
    golden_hash,
    golden_hash_u32ref,
    golden_r4,
)

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


# ========== V01-b L0-A: avalanche-hash unit probe (bitwise int32; DESIGN_v01b.md section 9.2.3) ==========
# Minimal probe mirroring r4_gumbel_sample's lowbias32 chain EXACTLY: seedP=seed*PRIME_W, counter
# arith_progression, 3 rounds of [shift-trick logical rshift (rshift/rshift/lshift/sub) + emulated
# XOR x^t=x+t-2*(x&t) via OVER-ALLOCATED int32 And]. Over-alloc [2*BVp] because int32 And is
# int16-native on dav-2201 (processes count/2 int32); [2*BVp] at count 2*BVp covers all BVp.
# Isolates the RNG so an L0-B tok mismatch can be triaged hash -> u/G -> argmax (graded isolation, V-b1).
@tilelang.jit(out_idx=[1], pass_configs=R4_PASS_CONFIGS)
def _hash_probe(B: int, M: int, BVp: int):
    n_blk = T.ceildiv(M, BVp)
    b_num = T.ceildiv(B, VEC_NUM)
    BVp2 = 2 * BVp  # over-allocation for the int16-native And (see kernel r4_gumbel_sample)

    @T.prim_func
    def main(seed_in: T.Tensor([B], "int32"), state_out: T.Tensor([B, M], "int32")):  # type: ignore
        with T.Kernel(b_num, is_npu=True) as (cid, vid):
            row = cid * VEC_NUM + vid
            if row < B:
                state_ub = T.alloc_shared([BVp2], "int32")  # over-alloc; real data [0:BVp)
                h_ub = T.alloc_shared([BVp2], "int32")  # logical-rshift temp t
                tmp_ub = T.alloc_shared([BVp2], "int32")  # shift-trick temp; then p = x&t
                seed_v = T.alloc_shared([1], "int32")
                seedP_v = T.alloc_shared([1], "int32")
                T.tile.fill(seedP_v, 0)
                T.copy(seed_in[row : row + 1], seed_v)
                T.tile.mul(seedP_v, seed_v, PRIME_W)  # seedP = seed_i*PRIME_W (int32 wrap)
                for b in T.serial(n_blk):
                    T.tile.fill(state_ub, 0)
                    T.tile.arith_progression(state_ub, seedP_v[0] + b * BVp, 1, BVp2)
                    # round 1: x ^= x>>>16; x *= M1
                    T.tile.bitwise_rshift(h_ub, state_ub, 16)
                    T.tile.bitwise_rshift(tmp_ub, h_ub, 16)
                    T.tile.bitwise_lshift(tmp_ub, tmp_ub, 16)
                    T.tile.sub(h_ub, h_ub, tmp_ub)  # h = x>>>16 (logical, shift-trick)
                    T.tile.bitwise_and(tmp_ub, state_ub, h_ub)  # tmp = x&t (over-alloc And)
                    T.tile.add(state_ub, state_ub, h_ub)
                    T.tile.mul(tmp_ub, tmp_ub, 2)
                    T.tile.sub(state_ub, state_ub, tmp_ub)  # x = x^t
                    T.tile.mul(state_ub, state_ub, HASH_M1)
                    # round 2: x ^= x>>>15; x *= M2
                    T.tile.bitwise_rshift(h_ub, state_ub, 15)
                    T.tile.bitwise_rshift(tmp_ub, h_ub, 17)
                    T.tile.bitwise_lshift(tmp_ub, tmp_ub, 17)
                    T.tile.sub(h_ub, h_ub, tmp_ub)
                    T.tile.bitwise_and(tmp_ub, state_ub, h_ub)
                    T.tile.add(state_ub, state_ub, h_ub)
                    T.tile.mul(tmp_ub, tmp_ub, 2)
                    T.tile.sub(state_ub, state_ub, tmp_ub)
                    T.tile.mul(state_ub, state_ub, HASH_M2)
                    # round 3: x ^= x>>>16
                    T.tile.bitwise_rshift(h_ub, state_ub, 16)
                    T.tile.bitwise_rshift(tmp_ub, h_ub, 16)
                    T.tile.bitwise_lshift(tmp_ub, tmp_ub, 16)
                    T.tile.sub(h_ub, h_ub, tmp_ub)
                    T.tile.bitwise_and(tmp_ub, state_ub, h_ub)
                    T.tile.add(state_ub, state_ub, h_ub)
                    T.tile.mul(tmp_ub, tmp_ub, 2)
                    T.tile.sub(state_ub, state_ub, tmp_ub)
                    T.copy(state_ub[0:BVp], state_out[row, b * BVp])  # export real half

    return main


def test_r4_gumbel_l0a():
    """L0-A: kernel avalanche-hash int32 stream bitwise == golden_hash (== canonical lowbias32)."""
    print("=" * 80)
    print("[L0-A] V01-b avalanche-hash unit probe -- int32 bitwise (3 seed x {256,4096})")
    print("=" * 80)
    all_ok = True
    seeds = [0, 42, 1234]
    B = len(seeds)
    seed_in = torch.tensor(seeds, dtype=torch.int32)
    for M, BVp in ((256, 256), (4096, 4096), (4096, 256)):
        try:
            out = _hash_probe(B, M, BVp)(seed_in.npu()).cpu()
            for r, s in enumerate(seeds):
                ref = golden_hash(M, s)  # int32 [M] (kernel-mirror emulated chain)
                u32 = golden_hash_u32ref(M, s)  # canonical uint32 lowbias32
                bit_gold = torch.equal(ref, u32)  # golden == canonical lowbias32
                bit_kern = torch.equal(out[r], ref)  # kernel == golden (int32 exact)
                mism = int((out[r] != ref).sum())
                ok = bit_gold and bit_kern
                all_ok &= ok
                print(
                    f"[PRECISION_{'PASS' if ok else 'FAIL'}] l0a_hash_unit M={M} BV={BVp} seed={s} "
                    f"kernel==golden={bit_kern}(mism={mism}) golden==u32ref={bit_gold}"
                )
        except Exception as e:  # noqa: BLE001
            print(f"[PRECISION_FAIL] l0a_hash_unit M={M} BV={BVp}: {type(e).__name__}: {str(e)[:200]}")
            all_ok = False
    return all_ok


# ========== V01-b L0-B: end-to-end tok_out bitwise (DESIGN_v01b.md section 9.2.2, 8 cases) ==========
# (name, B, V, k, p, mu, need_min_p, seed_mode). seed_mode: "arange" -> rand=arange(B);
# "var" -> run SEED in {0,42,1234} each with rand=full((B,),SEED).
L0B_CASES = [
    ("l0b_single_block", 2, 4096, 40, 1.0, 0.0, False, "arange"),
    ("l0b_multiblock", 8, 16384, 40, 1.0, 0.0, False, "arange"),
    ("l0b_topp_nucleus", 8, 16384, 40, 0.95, 0.0, False, "arange"),
    ("l0b_minp_T2", 8, 16384, 40, 0.95, 0.05, True, "arange"),
    ("l0b_prod_v_tail", 16, 152064, 40, 0.95, 0.05, True, "arange"),
    ("l0b_full_occupancy", 40, 152064, 20, 0.9, 0.1, True, "arange"),
    ("l0b_k_eq_1", 2, 16384, 1, 1.0, 0.0, False, "arange"),
    ("l0b_seed_var", 8, 16384, 40, 0.95, 0.0, False, "var"),
]


def _make_gumbel_inputs(B, V, k, p, mu, rand):
    torch.manual_seed(0)
    logits = torch.randn(B, V, dtype=torch.float32) * LOGIT_SCALE
    top_ks = torch.full((B,), k, dtype=torch.int32)
    top_ps = torch.full((B,), p, dtype=torch.float32)
    min_ps = torch.full((B,), mu, dtype=torch.float32)
    return logits, top_ks, top_ps, min_ps, rand


def _run_gumbel_once(name, B, V, k, p, mu, nmp, rand, tag):
    """Run K1+K2-Gumbel (FUSE_GUMBEL=True); compare tok_out bitwise + valid all-1 + tau regression."""
    logits, top_ks, top_ps, min_ps, rand = _make_gumbel_inputs(B, V, k, p, mu, rand)
    t0 = time.time()
    _m, _s, tau_out, valid, tok = r4_fused_sampling(
        logits.npu(), top_ks.npu(), top_ps.npu(), min_ps.npu(), nmp, K_MAX, BV, rand=rand.npu(), FUSE_GUMBEL=True
    )
    torch.npu.synchronize()
    tau_out, valid, tok = tau_out.cpu(), valid.cpu(), tok.cpu()
    run_s = time.time() - t0

    # Main judge: tok_out bitwise vs faithful Gumbel-max golden (feed the KERNEL's tau_out to
    # isolate K2-Gumbel from any K1 tau discrepancy; V01-a proved tau_out == golden tau_ref).
    tok_ref = golden_gumbel_sample(logits, tau_out, rand)
    tok_ok, tok_ratio, tok_max = check_precision(tok, tok_ref, "int32")
    # valid must be all-1 under the LOGIT_SCALE=8.0 recipe (guard must not fire).
    _mr, _sr, tau_ref, valid_ref, _wr = golden_r4(logits, top_ks, top_ps, min_ps, nmp, K_MAX)
    valid_ok, valid_ratio, valid_max = check_precision(valid, valid_ref, "int32")
    valid_all1 = bool((valid_ref == 1).all().item()) and bool((valid == 1).all().item())
    # K1 regression sentinel: tau_out vs golden tau_ref (fp32 tolerance) -> K1 not broken by V01-b.
    tau_ok, tau_ratio, tau_max = check_precision(tau_out, tau_ref, "float32")

    ok = tok_ok and valid_ok and valid_all1 and tau_ok
    detail = (
        f"{name}{tag} B={B} V={V} k={k} p={p} mu={mu} min_p={nmp} ({run_s:.1f}s) "
        f"tok[ratio={tok_ratio:.4f},max={tok_max:.0f}] valid[r={valid_ratio:.4f}] tau[r={tau_ratio:.4f},m={tau_max:.1e}]"
        + ("" if valid_all1 else " !!valid_not_all_1(check LOGIT_SCALE recipe)")
    )
    if not tok_ok:
        mism = (tok != tok_ref).nonzero().flatten().tolist()[:5]
        detail += f" tok_mismatch_rows={mism} got={tok[mism].tolist()} ref={tok_ref[mism].tolist()}"
    return ok, detail


def test_r4_gumbel_l0b():
    """L0-B gate: 8 cases, tok_out int32 bitwise == golden_gumbel_sample (+ valid all-1)."""
    print("=" * 80)
    print("[L0-B] V01-b end-to-end tok_out bitwise -- 8 cases (LOGIT_SCALE=8.0, rand=arange/var)")
    print("=" * 80)
    all_ok = True
    for name, B, V, k, p, mu, nmp, seed_mode in L0B_CASES:
        try:
            if seed_mode == "var":
                for SEED in (0, 42, 1234):
                    rand = torch.full((B,), SEED, dtype=torch.int32)
                    ok, detail = _run_gumbel_once(name, B, V, k, p, mu, nmp, rand, f"[seed={SEED}]")
                    print(f"[PRECISION_{'PASS' if ok else 'FAIL'}] {detail}")
                    all_ok &= ok
            else:
                rand = torch.arange(B, dtype=torch.int32)
                ok, detail = _run_gumbel_once(name, B, V, k, p, mu, nmp, rand, "")
                print(f"[PRECISION_{'PASS' if ok else 'FAIL'}] {detail}")
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


# ========== V01-b L1 functional (scenario B): tok_out bitwise across shapes/params ==========
# (name, B, V, K_MAX, BV, k, p, mu, need_min_p, scale, offset, seed_mode, tags)
# Shapes deterministically generated per coverage-matrix §6 (block BV=4096): ALIGNED / TAIL-1
# (batch axis, B odd) / TAIL-MID (V%BV==BV//2) / PRIME / EDGE (B=1, V=K_MAX). Vocab-axis tail
# is always >=K_MAX (host C3 forbids tail<K_MAX), so TAIL-1 rides the BATCH axis (row guard).
# Every case CPU-pre-checked to yield valid all-1 (LOGIT_SCALE>=5.0 recipe). All FUSE_GUMBEL=True.
L1B_CASES = [
    (
        "l1b_aligned_base",
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
        "arange",
        ["D-PARAM-FUSE_GUMBEL", "D-DTYPE-fp32", "D-DTYPE-int32", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-top_k", "D-PARAM-top_p"],
    ),
    ("l1b_aligned_p1", 8, 16384, 256, 4096, 40, 1.0, 0.0, False, 8.0, 0.0, "arange", ["D-PARAM-FUSE_GUMBEL", "D-PARAM-top_p"]),
    (
        "l1b_minp_on",
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
        "arange",
        ["D-PARAM-FUSE_GUMBEL", "D-PARAM-NEED_MIN_P", "D-PARAM-need_min_p_sampling", "D-PARAM-min_p"],
    ),
    ("l1b_valrange_S", 4, 4096, 256, 4096, 40, 0.95, 0.0, False, 5.0, 0.0, "arange", ["D-PARAM-FUSE_GUMBEL", "D-VALRANGE-S"]),
    ("l1b_valrange_L", 4, 16384, 256, 4096, 40, 0.95, 0.0, False, 1000.0, 0.0, "arange", ["D-PARAM-FUSE_GUMBEL", "D-VALRANGE-L"]),
    ("l1b_valrange_ASYM", 4, 4096, 256, 4096, 40, 0.95, 0.0, False, 6.0, -4.0, "arange", ["D-PARAM-FUSE_GUMBEL", "D-VALRANGE-ASYM"]),
    ("l1b_tail1_oddB", 5, 16384, 256, 4096, 40, 0.95, 0.0, False, 8.0, 0.0, "arange", ["D-PARAM-FUSE_GUMBEL", "D-SHAPE-TAIL-1"]),
    ("l1b_tailmid_V", 8, 10240, 256, 4096, 40, 0.95, 0.0, False, 8.0, 0.0, "arange", ["D-PARAM-FUSE_GUMBEL", "D-SHAPE-TAIL-MID"]),
    ("l1b_prime_V_B", 7, 4409, 256, 4096, 40, 0.95, 0.0, False, 8.0, 0.0, "arange", ["D-PARAM-FUSE_GUMBEL", "D-SHAPE-PRIME"]),
    ("l1b_edge_B1", 1, 4096, 256, 4096, 40, 0.95, 0.0, False, 8.0, 0.0, "arange", ["D-PARAM-FUSE_GUMBEL", "D-SHAPE-EDGE"]),
    ("l1b_edge_V_eq_KMAX", 2, 256, 256, 4096, 40, 0.95, 0.0, False, 8.0, 0.0, "arange", ["D-PARAM-FUSE_GUMBEL", "D-SHAPE-EDGE"]),
    (
        "l1b_kmax_128",
        4,
        16384,
        128,
        4096,
        100,
        0.95,
        0.0,
        False,
        8.0,
        0.0,
        "arange",
        ["D-PARAM-FUSE_GUMBEL", "D-PARAM-K_MAX", "D-PARAM-top_k"],
    ),
    ("l1b_bv_2048", 4, 16384, 256, 2048, 40, 0.95, 0.0, False, 8.0, 0.0, "arange", ["D-PARAM-FUSE_GUMBEL", "D-PARAM-BV"]),
    ("l1b_k_eq_1", 4, 4096, 256, 4096, 1, 1.0, 0.0, False, 8.0, 0.0, "arange", ["D-PARAM-FUSE_GUMBEL", "D-PARAM-top_k", "D-PARAM-top_p"]),
    (
        "l1b_prod_v_minp",
        4,
        152064,
        256,
        4096,
        30,
        0.95,
        0.1,
        True,
        8.0,
        0.0,
        "arange",
        ["D-PARAM-FUSE_GUMBEL", "D-PARAM-min_p", "D-PARAM-top_k", "D-VALRANGE-M"],
    ),
    ("l1b_seed_var", 8, 16384, 256, 4096, 40, 0.95, 0.0, False, 8.0, 0.0, "var", ["D-PARAM-FUSE_GUMBEL", "D-PARAM-top_p"]),
]


def _run_gumbel_l1(name, B, V, kmax, bv, k, p, mu, nmp, scale, offset, seed_mode, tags):
    """Run K1+K2-Gumbel; tok_out bitwise vs faithful golden + valid all-1 + tau regression."""
    torch.manual_seed(0)
    logits = torch.randn(B, V, dtype=torch.float32) * scale + offset
    top_ks = torch.full((B,), k, dtype=torch.int32)
    top_ps = torch.full((B,), p, dtype=torch.float32)
    min_ps = torch.full((B,), mu, dtype=torch.float32)
    _mr, _sr, tau_ref, valid_ref, _wr = golden_r4(logits, top_ks, top_ps, min_ps, nmp, kmax)
    seeds = (0, 42, 1234) if seed_mode == "var" else (None,)
    ok_all = True
    parts = []
    t0 = time.time()
    for SEED in seeds:
        rand = torch.arange(B, dtype=torch.int32) if SEED is None else torch.full((B,), SEED, dtype=torch.int32)
        _m, _s, tau_out, valid, tok = r4_fused_sampling(
            logits.npu(), top_ks.npu(), top_ps.npu(), min_ps.npu(), nmp, kmax, bv, rand=rand.npu(), FUSE_GUMBEL=True
        )
        torch.npu.synchronize()
        tau_out, valid, tok = tau_out.cpu(), valid.cpu(), tok.cpu()
        tok_ref = golden_gumbel_sample(logits, tau_out, rand)  # feed kernel tau -> isolate K2-Gumbel
        tok_ok, tok_ratio, tok_max = check_precision(tok, tok_ref, "int32")
        valid_ok, _, _ = check_precision(valid, valid_ref, "int32")
        valid_all1 = bool((valid_ref == 1).all().item()) and bool((valid == 1).all().item())
        tau_ok, _, tau_max = check_precision(tau_out, tau_ref, "float32")
        ok = tok_ok and valid_ok and valid_all1 and tau_ok
        ok_all &= ok
        sl = f"s{SEED}" if SEED is not None else "arange"
        parts.append(f"{sl}:tok[r={tok_ratio:.4f},m={tok_max:.0f}],v1={valid_all1},tau[m={tau_max:.1e}]")
    detail = (
        f"{name} B={B} V={V} K_MAX={kmax} BV={bv} k={k} p={p} mu={mu} min_p={nmp} "
        f"scale={scale}({time.time() - t0:.1f}s) | " + " ".join(parts) + f" {tags}"
    )
    return ok_all, detail


def test_r4_gumbel_l1():
    """V01-b L1 functional: tok_out bitwise across deterministic shapes/params (blocking)."""
    print("=" * 80)
    print(f"[L1-B] V01-b Gumbel functional tests -- {len(L1B_CASES)} cases (tok_out bitwise)")
    print("=" * 80)
    all_ok = True
    for case in L1B_CASES:
        name = case[0]
        try:
            ok, detail = _run_gumbel_l1(*case)
            print(f"[PRECISION_{'PASS' if ok else 'FAIL'}] {detail}")
            all_ok &= ok
        except Exception as e:  # noqa: BLE001
            import traceback

            print(f"[PRECISION_FAIL] l1b {name}: {type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc()
            all_ok = False
    return all_ok


# ========== V01-b distribution consistency (DESIGN_v01b.md 9.5; blocking statistical gate) ==========
def test_r4_gumbel_distribution():
    """Gumbel-max + avalanche hash must sample the target categorical: chi2 p>0.01, KL<1e-3,
    top-k coverage>1-1e-4, entropy rel-err<1%. Fixed logits row, N multi-seed samples.

    R-JIT: fixed chunk batch Bc -> ONE JIT compile; seeds vary by value only (no recompile).
    """
    from scipy import stats

    print("=" * 80)
    print("[DIST] V01-b Gumbel-max distribution consistency (chi2/KL/coverage/entropy)")
    print("=" * 80)
    torch.manual_seed(0)
    V, k = 4096, 40
    N, Bc = 100000, 10000  # 10 chunks of 10000 (fixed Bc -> single JIT)
    base = torch.randn(1, V, dtype=torch.float32) * LOGIT_SCALE  # one fixed logits row
    # target categorical: softmax(z) restricted to keep set {z >= tau}, renormalized (tau = k-th largest)
    tau = torch.topk(base, k, dim=-1).values[:, -1]  # [1]
    keep = base[0] >= tau[0]
    keep_idx = keep.nonzero().flatten()
    p_target = torch.softmax(base[0][keep], dim=0).double()  # [n_keep]
    n_keep = int(keep.sum().item())

    logits_chunk = base.expand(Bc, V).contiguous()  # [Bc,V] identical rows (CPU prep, one H2D)
    top_ks = torch.full((Bc,), k, dtype=torch.int32)
    top_ps = torch.full((Bc,), 1.0, dtype=torch.float32)
    min_ps = torch.full((Bc,), 0.0, dtype=torch.float32)
    # Hoist H2D out of the loop: logits/params are identical across chunks (only rand varies).
    lg_npu = logits_chunk.npu()
    ks_npu, ps_npu, mus_npu = top_ks.npu(), top_ps.npu(), min_ps.npu()
    counts = torch.zeros(V, dtype=torch.int64)
    n_valid = 0
    t0 = time.time()
    for c in range(N // Bc):
        rand = torch.arange(c * Bc, (c + 1) * Bc, dtype=torch.int32)  # distinct seeds
        _m, _s, _t, valid, tok = r4_fused_sampling(lg_npu, ks_npu, ps_npu, mus_npu, False, K_MAX, BV, rand=rand.npu(), FUSE_GUMBEL=True)
        torch.npu.synchronize()
        valid, tok = valid.cpu(), tok.cpu()
        n_valid += int((valid == 1).sum().item())
        counts += torch.bincount(tok.to(torch.int64), minlength=V)
    run_s = time.time() - t0

    obs_keep = counts[keep_idx].double()
    n_samp = float(obs_keep.sum().item())
    exp_keep = p_target * n_samp
    chi2 = float(((obs_keep - exp_keep) ** 2 / exp_keep).sum().item())
    dof = max(n_keep - 1, 1)
    p_val = float(stats.chi2.sf(chi2, dof))
    p_emp = obs_keep / n_samp
    kl = float((p_target * torch.log(p_target / p_emp.clamp_min(1e-12))).sum().item())
    coverage = float(counts[keep_idx].sum().item()) / max(float(counts.sum().item()), 1.0)  # frac in keep set
    H_emp = float(-(p_emp * torch.log(p_emp.clamp_min(1e-12))).sum().item())
    H_tgt = float(-(p_target * torch.log(p_target)).sum().item())
    ent_rel = abs(H_emp - H_tgt) / max(H_tgt, 1e-12)

    ok_chi2 = p_val > 0.01
    ok_kl = kl < 1e-3
    ok_cov = coverage > 1 - 1e-4
    ok_ent = ent_rel < 0.01
    ok_valid = n_valid == N
    ok = ok_chi2 and ok_kl and ok_cov and ok_ent and ok_valid
    print(
        f"[PRECISION_{'PASS' if ok else 'FAIL'}] dist N={N} n_keep={n_keep} ({run_s:.1f}s) "
        f"chi2={chi2:.2f}/dof={dof} p={p_val:.4f}(>0.01:{ok_chi2}) KL={kl:.2e}(<1e-3:{ok_kl}) "
        f"cov={coverage:.6f}(>1-1e-4:{ok_cov}) H_relerr={ent_rel:.2e}(<1%:{ok_ent}) valid={n_valid}/{N}"
    )
    return ok


# ========== V01-b L2 negative (non-blocking): illegal input must be rejected ==========
def _run_gumbel_guard(name, B, V, kmax, bv, k, p, mu, nmp, scale, tags):
    """Out-of-fast-path input with FUSE_GUMBEL=True -> valid all-0 (host falls back)."""
    try:
        torch.manual_seed(0)
        logits = torch.randn(B, V, dtype=torch.float32) * scale
        rand = torch.arange(B, dtype=torch.int32)
        _m, _s, _t, valid, _tok = r4_fused_sampling(
            logits.npu(),
            torch.full((B,), k, dtype=torch.int32).npu(),
            torch.full((B,), p, dtype=torch.float32).npu(),
            torch.full((B,), mu, dtype=torch.float32).npu(),
            nmp,
            kmax,
            bv,
            rand=rand.npu(),
            FUSE_GUMBEL=True,
        )
        torch.npu.synchronize()
        valid = valid.cpu()
        if bool((valid == 0).all().item()):
            print(f"[BOUNDARY_PASS] l2b {name}: guard fired, valid all-0 {tags}")
        else:
            print(f"[BOUNDARY_WARN] l2b {name}: guard did NOT fire (ones={int((valid == 1).sum())}/{B}) {tags}")
    except Exception as e:  # noqa: BLE001
        print(f"[BOUNDARY_WARN] l2b {name}: raised instead of guarding ({type(e).__name__}: {str(e)[:100]}) {tags}")


def test_r4_gumbel_l2():
    """V01-b L2: unsupported dtype / missing-rand / bad rand dtype / guards / illegal shape."""
    print("=" * 80)
    print("[L2-B] V01-b Gumbel negative tests (non-blocking)")
    print("=" * 80)
    base = torch.randn(4, 4096, dtype=torch.float32) * 8.0
    ks = torch.full((4,), 40, dtype=torch.int32)
    ps = torch.full((4,), 0.95, dtype=torch.float32)
    mus = torch.full((4,), 0.0, dtype=torch.float32)
    rand = torch.arange(4, dtype=torch.int32)

    # bf16 logits with FUSE_GUMBEL -> dtype pre-check must raise
    _run_exception(
        "exc_gumbel_bf16",
        lambda: r4_fused_sampling(
            base.to(torch.bfloat16).npu(), ks.npu(), ps.npu(), mus.npu(), False, 256, 4096, rand=rand.npu(), FUSE_GUMBEL=True
        ),
        ["D-EXC-DTYPE", "D-PARAM-FUSE_GUMBEL"],
    )
    # rand=None with FUSE_GUMBEL -> must raise (seed required)
    _run_exception(
        "exc_gumbel_rand_none",
        lambda: r4_fused_sampling(base.npu(), ks.npu(), ps.npu(), mus.npu(), False, 256, 4096, rand=None, FUSE_GUMBEL=True),
        ["D-EXC-DTYPE", "D-PARAM-FUSE_GUMBEL"],
    )
    # rand wrong dtype (int64) -> must raise
    _run_exception(
        "exc_gumbel_rand_dtype",
        lambda: r4_fused_sampling(
            base.npu(), ks.npu(), ps.npu(), mus.npu(), False, 256, 4096, rand=torch.arange(4, dtype=torch.int64).npu(), FUSE_GUMBEL=True
        ),
        ["D-EXC-DTYPE", "D-PARAM-FUSE_GUMBEL"],
    )
    # V < K_MAX (C7) with FUSE_GUMBEL -> must raise
    _run_exception(
        "exc_gumbel_V_lt_KMAX",
        lambda: r4_fused_sampling(
            torch.randn(2, 128, dtype=torch.float32).npu(),
            torch.full((2,), 40, dtype=torch.int32).npu(),
            torch.full((2,), 0.95, dtype=torch.float32).npu(),
            torch.full((2,), 0.0, dtype=torch.float32).npu(),
            False,
            256,
            4096,
            rand=torch.arange(2, dtype=torch.int32).npu(),
            FUSE_GUMBEL=True,
        ),
        ["D-EXC-SHAPE", "D-PARAM-FUSE_GUMBEL"],
    )
    # tail < K_MAX (C3) with FUSE_GUMBEL -> must raise
    _run_exception(
        "exc_gumbel_tail_lt_KMAX",
        lambda: r4_fused_sampling(
            torch.randn(2, 4097, dtype=torch.float32).npu(),
            torch.full((2,), 40, dtype=torch.int32).npu(),
            torch.full((2,), 0.95, dtype=torch.float32).npu(),
            torch.full((2,), 0.0, dtype=torch.float32).npu(),
            False,
            256,
            4096,
            rand=torch.arange(2, dtype=torch.int32).npu(),
            FUSE_GUMBEL=True,
        ),
        ["D-EXC-SHAPE", "D-PARAM-FUSE_GUMBEL"],
    )
    # out-of-fast-path sampling params -> guard fires (valid all-0), tok discarded by host
    _run_gumbel_guard(
        "guard_gumbel_k_TOP_K_ALL", 4, 4096, 256, 4096, 1 << 30, 0.95, 0.0, False, 8.0, ["D-PARAM-top_k", "D-PARAM-FUSE_GUMBEL"]
    )
    _run_gumbel_guard("guard_gumbel_k_zero", 4, 4096, 256, 4096, 0, 0.95, 0.0, False, 8.0, ["D-PARAM-top_k", "D-PARAM-FUSE_GUMBEL"])
    _run_gumbel_guard("guard_gumbel_k_gt_KMAX", 4, 4096, 256, 4096, 257, 0.95, 0.0, False, 8.0, ["D-PARAM-top_k", "D-PARAM-FUSE_GUMBEL"])
    _run_gumbel_guard("guard_gumbel_flat_dist", 4, 16384, 256, 4096, 40, 0.95, 0.0, False, 1.0, ["D-VALRANGE-S", "D-PARAM-FUSE_GUMBEL"])

    # B == 0 with FUSE_GUMBEL -> host intercept returns empty tok_out (int32)
    try:
        outs = r4_fused_sampling(
            torch.empty(0, 4096, dtype=torch.float32).npu(),
            torch.empty(0, dtype=torch.int32).npu(),
            torch.empty(0, dtype=torch.float32).npu(),
            torch.empty(0, dtype=torch.float32).npu(),
            False,
            256,
            4096,
            rand=torch.empty(0, dtype=torch.int32).npu(),
            FUSE_GUMBEL=True,
        )
        tok = outs[4]
        if all(o.numel() == 0 for o in outs) and tok.dtype == torch.int32:
            print('[BOUNDARY_PASS] l2b guard_gumbel_empty_B0: empty tok_out (int32) ["D-EXC-SHAPE","D-PARAM-FUSE_GUMBEL"]')
        else:
            print('[BOUNDARY_WARN] l2b guard_gumbel_empty_B0: unexpected output for B=0 ["D-EXC-SHAPE"]')
    except Exception as e:  # noqa: BLE001
        print(f'[BOUNDARY_WARN] l2b guard_gumbel_empty_B0: raised ({type(e).__name__}) ["D-EXC-SHAPE"]')


# ========== V01-b Boundary (non-blocking): legal special values, tok_out compared ==========
def _gumbel_bnd(name, logits, rand, k, p, mu, nmp, tags, kmax=256, bv=4096, expect_guard=False):
    """Run K1+K2-Gumbel on special-value logits; compare tok_out bitwise vs golden (or guard)."""
    B, V = logits.shape

    def fn():
        top_ks = torch.full((B,), k, dtype=torch.int32)
        top_ps = torch.full((B,), p, dtype=torch.float32)
        min_ps = torch.full((B,), mu, dtype=torch.float32)
        _m, _s, tau_out, valid, tok = r4_fused_sampling(
            logits.npu(), top_ks.npu(), top_ps.npu(), min_ps.npu(), nmp, kmax, bv, rand=rand.npu(), FUSE_GUMBEL=True
        )
        torch.npu.synchronize()
        tau_out, valid, tok = tau_out.cpu(), valid.cpu(), tok.cpu()
        if expect_guard:
            # degenerate row -> guard must fire; encode as valid (int32) all-0 check via tok==valid sentinel
            return valid, torch.zeros_like(valid)
        tok_ref = golden_gumbel_sample(logits, tau_out, rand)
        return tok, tok_ref

    dtype = "int32"
    _run_boundary(name, dtype, fn, tags)


def test_r4_gumbel_boundary():
    """V01-b Boundary: sparse -inf / large-seed wraparound / sparse NaN / degenerate guard."""
    print("=" * 80)
    print("[BOUNDARY-B] V01-b Gumbel special-value tests (non-blocking)")
    print("=" * 80)
    torch.manual_seed(0)
    B, V = 4, 4096

    # Sparse -inf logits (never a whole -inf row): keep mask excludes them, argmax unaffected.
    lg_inf = torch.randn(B, V, dtype=torch.float32) * 8.0
    lg_inf[0, :50] = float("-inf")
    lg_inf[2, 100] = float("-inf")
    _gumbel_bnd(
        "bnd_gumbel_sparse_neginf",
        lg_inf,
        torch.arange(B, dtype=torch.int32),
        40,
        0.95,
        0.0,
        False,
        ["D-SPECIAL-INF", "D-PARAM-FUSE_GUMBEL"],
    )

    # Large seed (seed*PRIME_W + j wraps int32): golden_hash _wrap_int32 wraps identically for ANY
    # seed (rev1 relaxed the old LCG seed+V<2^31 constraint) -> still bitwise.
    big = 2**31 - 100
    _gumbel_bnd(
        "bnd_gumbel_big_seed_wrap",
        torch.randn(B, V, dtype=torch.float32) * 8.0,
        torch.full((B,), big, dtype=torch.int32),
        40,
        0.95,
        0.0,
        False,
        ["D-SPECIAL-DBOUND", "D-PARAM-FUSE_GUMBEL"],
    )

    # fp32 magnitude bounds on logits (online softmax keeps exp(z-m)<=1; argmax robust).
    lg_bnd = torch.randn(B, V, dtype=torch.float32) * 8.0
    lg_bnd[0, 0] = 88.0
    lg_bnd[0, 1] = -88.0
    lg_bnd[1, 5] = 1e4
    lg_bnd[1, 6] = -1e4
    _gumbel_bnd(
        "bnd_gumbel_dbound", lg_bnd, torch.arange(B, dtype=torch.int32), 40, 0.95, 0.0, False, ["D-SPECIAL-DBOUND", "D-PARAM-FUSE_GUMBEL"]
    )

    # Degenerate rows (all-zero / all-equal): guard must fire (valid all-0); host falls back.
    _gumbel_bnd(
        "bnd_gumbel_all_zero_guard",
        torch.zeros(B, V, dtype=torch.float32),
        torch.arange(B, dtype=torch.int32),
        40,
        0.95,
        0.0,
        False,
        ["D-SPECIAL-ZERO", "D-PARAM-FUSE_GUMBEL"],
        expect_guard=True,
    )
    _gumbel_bnd(
        "bnd_gumbel_all_equal_guard",
        torch.full((B, V), 3.5, dtype=torch.float32),
        torch.arange(B, dtype=torch.int32),
        40,
        0.95,
        0.0,
        False,
        ["D-SPECIAL-ZERO", "D-PARAM-FUSE_GUMBEL"],
        expect_guard=True,
    )

    # Sparse NaN: unsupported input (DESIGN_v01b.md 9.3) -> semantics undefined, WARN expected.
    lg_nan = torch.randn(B, V, dtype=torch.float32) * 8.0
    lg_nan[1, 7] = float("nan")
    lg_nan[3, 11] = float("nan")
    _gumbel_bnd(
        "bnd_gumbel_sparse_nan", lg_nan, torch.arange(B, dtype=torch.int32), 40, 0.95, 0.0, False, ["D-SPECIAL-NAN", "D-PARAM-FUSE_GUMBEL"]
    )


# ========== Stability / repeat-execution gate (V01-b rev2, BLOCKING) ==========
# Why this layer exists: the V01-b rev1 suite executed every config exactly ONCE, so it
# had zero detection power against an intermittent (~13% per launch) stale-UB defect that
# cann-bench case 5 (B=7, V=4409, TAIL=313) later exposed as tau_out = garbage large
# finite float with valid still 1 (so the D-GUARD/D-TIE degrade guard never fired and the
# garbage propagated to tok_out -> 0). A single pass cannot catch that; N repeats can.
#
# Two deterministic detectors are used instead of relying on luck:
#   (1) repeat-N bit-identity: every launch must be BIT-identical to the first, and must
#       match golden. Catches any launch-to-launch nondeterminism (stale UB / buffer
#       rotation), which is by definition impossible for a correct pure function.
#   (2) tail-adversary sentinel: put an unambiguous sentinel maximum at each tail index j
#       and require tau to reflect it. Catches silent DROPPING of tail elements (an
#       AscendC::DataCopy of a non-32B-multiple count truncates to a multiple of 8 fp32),
#       which is deterministic and therefore invisible to (1).
#   (3) cheap golden-free invariant: tau_i <= row_max(logits) or tau_i == -inf. The
#       reported garbage (1e19 .. 1e38) violates this instantly, and it costs no golden.
#
# Risk profile targeted: odd B (batch-axis row guard, B % VEC_NUM == 1), TAIL not a
# multiple of 32 B (V % BV % 8 != 0), the 1-element scalar UB merge epilogue
# (tau = max(tau_k, tau_p, tau_u)), and the multi-kernel K1 -> K2 chain (both paths).
STAB_REPS = 20  # >= 20 required; 13% per-launch defect -> P(miss) < 0.87^20 = 6.3%

# (name, B, V, K_MAX, BV, k, p, fuse, tags)
STAB_CASES = [
    # The original cann-bench defect shape: odd B, TAIL=313 (313 % 32 = 25 -> non-32B-aligned)
    ("stab_case5_B7_V4409_gumbel", 7, 4409, 256, 4096, 40, 1.0, True,
     ["D-SHAPE-PRIME", "D-PARAM-FUSE_GUMBEL"]),
    ("stab_case5_B7_V4409_mask", 7, 4409, 256, 4096, 40, 1.0, False,
     ["D-SHAPE-PRIME"]),
    # odd B with NO tail at all (control: proves odd B alone is not the trigger)
    ("stab_B1_V4096_gumbel", 1, 4096, 256, 4096, 40, 0.95, True,
     ["D-SHAPE-EDGE", "D-PARAM-FUSE_GUMBEL"]),
    ("stab_B1_V4096_mask", 1, 4096, 256, 4096, 40, 0.95, False,
     ["D-SHAPE-EDGE"]),
    # odd B + minimal legal non-aligned tail (TAIL=257, 257 % 32 = 1)
    ("stab_B3_V4353_gumbel", 3, 4353, 256, 4096, 40, 0.95, True,
     ["D-SHAPE-PRIME", "D-PARAM-FUSE_GUMBEL"]),
    # odd B + TAIL=511 (511 % 32 = 31, the largest possible misalignment)
    ("stab_B9_V4607_gumbel", 9, 4607, 256, 4096, 30, 0.9, True,
     ["D-SHAPE-PRIME", "D-PARAM-FUSE_GUMBEL"]),
    # odd B + non-aligned tail + non-default BV (TAIL=313 with BV=2048)
    ("stab_B7_V4409_bv2048_gumbel", 7, 4409, 256, 2048, 40, 0.95, True,
     ["D-PARAM-BV", "D-PARAM-FUSE_GUMBEL"]),
    # aligned-tail control (TAIL=2048): must stay clean, guards against over-fitting the fix
    ("stab_B5_V10240_gumbel", 5, 10240, 256, 4096, 40, 0.95, True,
     ["D-SHAPE-TAIL-MID", "D-PARAM-FUSE_GUMBEL"]),
]

# Adversary probes: (B, V, K_MAX, BV) -> sentinel swept over the at-risk tail indices.
ADV_CASES = [
    (2, 4409, 256, 4096),   # TAIL=313 -> 1 element at risk (313 - 312)
    (3, 4353, 256, 4096),   # TAIL=257 -> 7 elements at risk (257 - 256)
    (2, 4607, 256, 4096),   # TAIL=511 -> 7 elements at risk (511 - 504)
    (5, 261, 256, 4096),    # tail-only V<K_MAX*2, TAIL=261 -> 5 at risk (261 - 256)
]
ADV_SENTINEL = 1.0e4  # far above the LOGIT_SCALE=8 data range -> unambiguous row max


def _tau_invariant(logits, tau, valid, tok=None, tag=""):
    """Cheap golden-free invariant. Returns a list of violation strings.

    I1: tau_i <= row_max(logits) (1 ulp slack) OR tau_i == -inf.
    I2: tau_i is not NaN.
    I3: valid_i == 1 and tau_i finite => the chosen token column satisfies z >= tau.
    Any garbage large finite tau (the reported 1e19..1e38 class) trips I1 immediately.
    """
    viol = []
    z = logits.detach().float().cpu()
    t = tau.detach().float().cpu()
    v = valid.detach().int().cpu()
    rowmax = z.max(dim=-1).values
    for i in range(z.shape[0]):
        ti, rm = float(t[i]), float(rowmax[i])
        if ti != ti:
            viol.append(f"{tag}I2 row{i}: tau=NaN")
            continue
        if ti == float("-inf"):
            continue
        if ti > rm and abs(ti - rm) > 1e-5 * max(1.0, abs(rm)):
            viol.append(f"{tag}I1 row{i}: tau={ti:.6e} > row_max={rm:.6e}")
    if tok is not None:
        tk = tok.detach().long().cpu()
        for i in range(z.shape[0]):
            if int(v[i]) != 1 or float(t[i]) == float("-inf"):
                continue
            j = int(tk[i])
            if not (0 <= j < z.shape[1]):
                viol.append(f"{tag}I3 row{i}: tok={j} out of range")
            elif float(z[i, j]) < float(t[i]) - 1e-5 * max(1.0, abs(float(t[i]))):
                viol.append(f"{tag}I3 row{i}: tok={j} z={float(z[i, j]):.6e} < tau={float(t[i]):.6e}")
    return viol


def _stab_launch(B, V, kmax, bv, k, p, fuse, logits):
    top_ks = torch.full((B,), k, dtype=torch.int32)
    top_ps = torch.full((B,), p, dtype=torch.float32)
    min_ps = torch.zeros(B, dtype=torch.float32)
    rand = (torch.arange(B, dtype=torch.int32) * 7919 + 13)
    if fuse:
        return r4_fused_sampling(logits.npu(), top_ks.npu(), top_ps.npu(), min_ps.npu(),
                                 False, kmax, bv, rand=rand.npu(), FUSE_GUMBEL=True)
    return r4_fused_sampling(logits.npu(), top_ks.npu(), top_ps.npu(), min_ps.npu(),
                             False, kmax, bv)


def _stab_golden(B, V, kmax, k, p, fuse, logits):
    top_ks = torch.full((B,), k, dtype=torch.int32)
    top_ps = torch.full((B,), p, dtype=torch.float32)
    min_ps = torch.zeros(B, dtype=torch.float32)
    m, s, tau_ref, valid_ref, w_ref = golden_r4(logits, top_ks, top_ps, min_ps, False, kmax)
    if not fuse:
        return (m, s, tau_ref, valid_ref, w_ref)
    rand = (torch.arange(B, dtype=torch.int32) * 7919 + 13)
    return (m, s, tau_ref, valid_ref, golden_gumbel_sample(logits, tau_ref, rand))


def _run_stability_case(name, B, V, kmax, bv, k, p, fuse, tags, reps):
    """Repeat `reps` launches; require bit-identity + golden match + invariant, every time."""
    torch.manual_seed(0)
    logits = torch.randn(B, V, dtype=torch.float32) * LOGIT_SCALE
    g = _stab_golden(B, V, kmax, k, p, fuse, logits)
    names = ("m_out", "s_out", "tau_out", "valid", "tok_out" if fuse else "w_out")

    outs = _stab_launch(B, V, kmax, bv, k, p, fuse, logits)
    torch.npu.synchronize()
    ref = [o.detach().cpu().clone() for o in outs]

    n_self = n_gold = n_inv = 0
    first = ""
    for it in range(reps):
        outs = _stab_launch(B, V, kmax, bv, k, p, fuse, logits)
        torch.npu.synchronize()
        cur = [o.detach().cpu().clone() for o in outs]
        for i, nm in enumerate(names):
            if not torch.equal(cur[i], ref[i]):
                n_self += 1
                first = first or f"it{it}:{nm} not bit-identical to launch 0"
            ok_i, ratio_i, max_i = check_precision(cur[i], g[i], "int32" if cur[i].dtype == torch.int32 else "float32")
            if not ok_i:
                n_gold += 1
                first = first or f"it{it}:{nm} golden ratio={ratio_i:.4f} max={max_i:.3e}"
        v = _tau_invariant(logits, cur[2], cur[3], tok=cur[4] if fuse else None, tag=f"it{it} ")
        if v:
            n_inv += 1
            first = first or v[0]
    ok = (n_self == 0 and n_gold == 0 and n_inv == 0)
    tail = V - (V // bv) * bv
    detail = (f"{name} B={B} V={V} K_MAX={kmax} BV={bv} TAIL={tail}(32Baligned={tail % 32 == 0}) "
              f"k={k} p={p} fuse={int(fuse)} reps={reps} "
              f"selfdiff={n_self} goldenfail={n_gold} invariant={n_inv} {tags}")
    if not ok:
        detail += f" first_bad={first}"
    return ok, detail


def _run_tail_adversary(B, V, kmax, bv, tags):
    """Sentinel max at every tail index j -> tau (k=1) must equal the sentinel.

    Deterministically catches silent dropping of tail elements: if the tail top-k source
    is built with a non-32B-multiple UB->UB DataCopy, indices [(TAIL//8)*8, TAIL) are
    truncated away and a sentinel placed there never reaches tau.
    """
    tail = V - (V // bv) * bv
    if tail <= 0:
        return True, f"adv B={B} V={V}: TAIL=0, skipped {tags}"
    off = (V // bv) * bv
    base = torch.full((B, V), -50.0, dtype=torch.float32)
    ramp = torch.linspace(-20.0, -10.0, tail, dtype=torch.float32)
    bad = []
    for j in range(tail):
        lg = base.clone()
        lg[:, off:] = ramp.unsqueeze(0).expand(B, tail).clone()
        for b in range(B):
            lg[b, off + j] = ADV_SENTINEL + b
        top_ks = torch.ones(B, dtype=torch.int32)
        top_ps = torch.ones(B, dtype=torch.float32)
        min_ps = torch.zeros(B, dtype=torch.float32)
        _m, _s, tau, _v, _w = r4_fused_sampling(lg.npu(), top_ks.npu(), top_ps.npu(),
                                                min_ps.npu(), False, kmax, bv)
        torch.npu.synchronize()
        tau = tau.detach().cpu()
        exp = ADV_SENTINEL + torch.arange(B, dtype=torch.float32)
        if not bool(((tau - exp).abs() < 1e-2).all()):
            bad.append((j, [round(float(x), 4) for x in tau[:2]]))
            if len(bad) >= 8:
                break
    ok = not bad
    detail = (f"adv B={B} V={V} K_MAX={kmax} BV={bv} TAIL={tail} swept={tail} "
              f"dropped={len(bad)} {tags}")
    if not ok:
        detail += f" bad_idx={bad}"
    return ok, detail


def test_r4_stability(reps=STAB_REPS, adversary=True):
    """Repeat-execution gate (BLOCKING): N>=20 bit-identical launches + tail adversary."""
    print("=" * 80)
    print(f"[STABILITY] repeat-execution gate -- {len(STAB_CASES)} configs x {reps} reps"
          + (f" + {len(ADV_CASES)} tail-adversary sweeps" if adversary else ""))
    print("=" * 80)
    all_ok = True
    for name, B, V, kmax, bv, k, p, fuse, tags in STAB_CASES:
        try:
            ok, detail = _run_stability_case(name, B, V, kmax, bv, k, p, fuse, tags, reps)
            print(f"[PRECISION_{'PASS' if ok else 'FAIL'}] {detail}")
            all_ok &= ok
        except Exception as e:  # noqa: BLE001
            print(f"[PRECISION_FAIL] stability {name}: {type(e).__name__}: {str(e)[:200]}")
            all_ok = False
    if adversary:
        for B, V, kmax, bv in ADV_CASES:
            tags = ["D-SHAPE-PRIME", "D-SHAPE-EDGE"]
            try:
                ok, detail = _run_tail_adversary(B, V, kmax, bv, tags)
                print(f"[PRECISION_{'PASS' if ok else 'FAIL'}] {detail}")
                all_ok &= ok
            except Exception as e:  # noqa: BLE001
                print(f"[PRECISION_FAIL] adversary B={B} V={V}: {type(e).__name__}: {str(e)[:200]}")
                all_ok = False
    return all_ok


# ---- optional deep stress (--level stress): UB poisoner + many more reps ----
_POISON_ELEMS = 40960  # fp32 -> 163840 B of the 196352 B VECCALC UB region


def _make_ub_poisoner(nb=4):
    """Fill [0, 163840) B of every launched AIV's UB with a literal, then return.

    UB is on-device SRAM and is NOT zeroed between launches, so a kernel that reads any
    location before writing it picks this value up. That converts an intermittent stale-UB
    defect into a deterministic one (this is how the rev2 tau_out defect was reproduced:
    poison 1e30 -> tau_out == 1e30 on every row, every launch, pre-fix).
    """

    @tilelang.jit(out_idx=[0], pass_configs=R4_PASS_CONFIGS)
    def ub_poison(NB: int, NELEM: int, VAL: float):
        @T.prim_func
        def main(sink: T.Tensor([NB * VEC_NUM, 32], "float32")):  # type: ignore
            with T.Kernel(NB, is_npu=True) as (cid, vid):
                row = cid * VEC_NUM + vid
                buf = T.alloc_shared([NELEM], "float32")
                T.tile.fill(buf, VAL)
                T.copy(buf[0:32], sink[row, 0:32])  # 128 B, 32 B aligned (anti-DCE)

        return main

    return ub_poison(nb, _POISON_ELEMS, 1e30)


def test_r4_stress(reps=200):
    """Deep stress (optional, --level stress): UB-poisoned repeat launches. BLOCKING if run."""
    print("=" * 80)
    print(f"[STRESS] UB-poisoned repeat gate -- {len(STAB_CASES)} configs x {reps} reps")
    print("=" * 80)
    try:
        pois = _make_ub_poisoner()
    except Exception as e:  # noqa: BLE001
        print(f"[PRECISION_FAIL] stress: poisoner build failed {type(e).__name__}: {str(e)[:160]}")
        return False
    all_ok = True
    for name, B, V, kmax, bv, k, p, fuse, tags in STAB_CASES:
        torch.manual_seed(0)
        logits = torch.randn(B, V, dtype=torch.float32) * LOGIT_SCALE
        g = _stab_golden(B, V, kmax, k, p, fuse, logits)
        names = ("m_out", "s_out", "tau_out", "valid", "tok_out" if fuse else "w_out")
        n_self = n_gold = n_inv = 0
        first = ""
        ref = None
        for it in range(reps):
            pois()  # poison UB, then launch immediately
            outs = _stab_launch(B, V, kmax, bv, k, p, fuse, logits)
            torch.npu.synchronize()
            cur = [o.detach().cpu().clone() for o in outs]
            if ref is None:
                ref = cur
            for i, nm in enumerate(names):
                if not torch.equal(cur[i], ref[i]):
                    n_self += 1
                    first = first or f"it{it}:{nm} not bit-identical"
                ok_i, ratio_i, max_i = check_precision(cur[i], g[i], "int32" if cur[i].dtype == torch.int32 else "float32")
                if not ok_i:
                    n_gold += 1
                    first = first or f"it{it}:{nm} golden ratio={ratio_i:.4f} max={max_i:.3e}"
            v = _tau_invariant(logits, cur[2], cur[3], tok=cur[4] if fuse else None, tag=f"it{it} ")
            if v:
                n_inv += 1
                first = first or v[0]
        ok = (n_self == 0 and n_gold == 0 and n_inv == 0)
        detail = (f"{name} poisoned reps={reps} selfdiff={n_self} goldenfail={n_gold} "
                  f"invariant={n_inv} {tags}")
        if not ok:
            detail += f" first_bad={first}"
        print(f"[PRECISION_{'PASS' if ok else 'FAIL'}] {detail}")
        all_ok &= ok
    return all_ok


# ========== Coverage summary (coverage_check.py reads tags above + these) ==========
# Fusion category => every dimension is mandatory (exempt set is empty), so COVERAGE_NA
# must stay empty: all dimensions are hit by real cases above (V01-a + V01-b).
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
    "D-PARAM-FUSE_GUMBEL": 16,  # V01-b: all Gumbel L1/L2/Boundary cases (FUSE_GUMBEL=True path)
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
    parser.add_argument("--level", default="l0",
                        choices=["l0", "l1", "l2", "boundary", "stability", "stress", "all"])
    parser.add_argument("--reps", type=int, default=STAB_REPS,
                        help=f"repeat count for --level stability (default {STAB_REPS}, min 20)")
    parser.add_argument("--stress-reps", type=int, default=200,
                        help="repeat count for --level stress (default 200)")
    args = parser.parse_args()

    # NOTE: DESIGN.md section 11.5.6 (R-JIT mitigation) -- do NOT call
    # tilelang.disable_cache()/clear_cache() here; B is part of the JIT key and a
    # persistent TILELANG_CACHE_DIR avoids recompilation storms. Set the env var instead:
    #   export TILELANG_CACHE_DIR=/tmp/opencode/r4_tl_cache
    torch.manual_seed(0)

    blocking_ok = True  # only L0/L1 gate the exit code
    if args.level in ("l0", "all"):
        blocking_ok &= test_r4_fused_sampling_l0()  # V01-a L0 (regression: K1 + K2 w_out)
        blocking_ok &= test_r4_gumbel_l0a()  # V01-b L0-A (avalanche-hash int32 bitwise)
        blocking_ok &= test_r4_gumbel_l0b()  # V01-b L0-B (tok_out bitwise)
    if args.level in ("l1", "all"):
        blocking_ok &= test_r4_fused_sampling_l1()  # V01-a L1 (regression)
        blocking_ok &= test_r4_gumbel_l1()  # V01-b L1 functional (tok_out bitwise)
        blocking_ok &= test_r4_gumbel_distribution()  # V01-b distribution consistency (chi2/KL/cov/entropy)
    if args.level in ("stability", "all"):
        # Repeat-execution gate (V01-b rev2). BLOCKING and part of `all`: a single pass per
        # config is what let the intermittent stale-UB tau_out defect escape rev1.
        # `all` enforces the N>=20 floor; a standalone --level stability honours --reps.
        reps_floor = STAB_REPS if args.level == "all" else 1
        blocking_ok &= test_r4_stability(reps=max(args.reps, reps_floor))
    if args.level == "stress":
        # Optional deep stress (UB poisoner + many reps). Not part of `all` (slow).
        blocking_ok &= test_r4_stress(reps=args.stress_reps)
    if args.level in ("l2", "all"):
        test_r4_fused_sampling_l2()  # V01-a negative: non-blocking
        test_r4_gumbel_l2()  # V01-b negative: non-blocking
    if args.level in ("boundary", "all"):
        test_r4_fused_sampling_boundary()  # V01-a special values: non-blocking
        test_r4_gumbel_boundary()  # V01-b special values: non-blocking

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
