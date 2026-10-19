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
    static         rev3 compile-time gate only (section 9.2.5 assertions 1-4, no NPU); also
                   runs first inside `l0` / `all`
    all            l0 + l1 + stability + l2 + boundary

Precision standard (DESIGN.md section 9.3, mixed tolerance double-gate):
    float32: atol=2**-16, rtol=2**-10, max_abs_error_limit=1e-2, required_matched_ratio=0.99
    int32  : exact (0/0/0/1.0)

L0 data recipe (DESIGN.md section 9.2.1, MANDATORY): randn * LOGIT_SCALE(8.0). Any other
scale makes the nucleus exceed K_MAX=256 -> guard fires -> valid=0 -> false L0 failure
(NOT a kernel bug; check the recipe first).

rev3 (split-V) adds a TWO-TIER acceptance criterion (DESIGN.md section 9.2.2) on top of the
tolerance gate above, plus its own layers:
    Tier 1  vs golden_r4 / golden_gumbel_sample -- the section 9.3 mixed-tolerance double gate
    Tier 2  vs the THREE FROZEN kernels on the same input -- torch.equal on the raw BIT
            PATTERN (fp32 viewed as int32), i.e. ZERO ulp of drift is allowed. Reported as
            [BITWISE_PASS] / [BITWISE_FAIL]. Tier 1 alone cannot catch a reduction-order bug
            (E-S4): a 1-ulp drift stays inside tolerance yet flips tau at a tau_p prefix-sum
            boundary, which then flips valid / tok_out -- both int32 EXACT-match gates.
    test_r4_rev3_static()  section 9.2.5 assertions 1-4 (pass_configs, frozen-kernel sha256,
                           measured UB, padCount==0 / UB->UB 32B / static loop bounds)
    test_r4_rev3_l0()      section 9.2.3's 15-case matrix x 2 tasks = 30 runs
    STAB_CASES / ADV_CASES gained split-path configs (V-S5 / V-S6) because when rev3 landed every
                           legacy config had n_blk_full <= 2 and was vetoed by D-SPLIT-GUARD, i.e.
                           the legacy stability + tail-adversary gates covered ZERO lines of the
                           split path (section 16.8). Both suites now FAIL LOUDLY if that regress.
                           [Stage-3 round-3 item A] _SPLIT_A_LAUNCH_US moved from the conservative
                           8.0 us placeholder to the MEASURED 1.540 us, which additionally routes 2
                           of the 8 legacy STAB configs (both n_blk_full == 2, i.e. vetoed by the
                           COST guard rather than structurally) onto the split path, and adds
                           cb6/cb11 -- the two contract cases that relaxation newly admits -- to
                           REV3_L1_CASES, STAB_CASES and _REV3_STATIC_SHAPES.
"""

import argparse
import os
import re
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
    _r4_split_plan,  # rev3 host dispatch plan -> (use_split, S, NBS, S_TAIL)
    detect_cube_core_num,  # rev3: runtime-probed core count (never hardcode 20)
    r4_fused_sampling,
    r4_gumbel_sample,  # noqa: F401  (V01-b K2 Gumbel kernel, exercised via r4_fused_sampling)
    r4_gumbel_sample_splitA,  # rev3 K2gA (split-parallel Gumbel argmax)
    r4_gumbel_sample_splitB,  # rev3 K2gB (cross-split strict-GT fold)
    r4_mask_sample,  # noqa: F401  (public kernel API, exercised via r4_fused_sampling)
    r4_topk_stats,  # noqa: F401  (public kernel API, exercised via r4_fused_sampling)
    r4_topk_stats_splitA,  # rev3 K1A (split-parallel hierarchical top-k merge)
    r4_topk_stats_splitB,  # rev3 K1B (serial softmax scan + cross-split merge + epilogue)
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


# =========================================================================================
# rev3 split-V layer (DESIGN.md rev3 sections 9.2.2 / 9.2.3 / 9.2.5 / 13 / 14).
#
# WHY THIS LAYER EXISTS: rev3 adds a SECOND implementation path (K1A->K1B->[K2m | K2gA->K2gB])
# selected on the host by _r4_split_plan(). The delivered path must stay byte-identical in
# OUTPUT, so the acceptance criterion is TWO-tiered (section 9.2.2):
#   Tier 1 -- vs golden_r4 / golden_gumbel_sample, the section 9.3 mixed-tolerance double gate
#   Tier 2 -- vs the THREE FROZEN kernels on the SAME input, torch.equal on the raw BIT PATTERN
# Tier 1 alone cannot detect a reduction-ORDER bug (E-S4): a 1-ulp drift stays inside the fp32
# tolerance yet flips tau at a tau_p prefix-sum boundary, which then flips valid / tok_out --
# both int32 EXACT-match gates. Tier 2 is therefore the primary rev3 gate (section 9.0 point 3).
#
# Section 16.8 blind spot this layer closes: when rev3 landed, every pre-existing STAB_CASES /
# ADV_CASES config had n_blk_full in {0,1,2} and was therefore vetoed by D-SPLIT-GUARD, so the old
# stability and tail-adversary gates never executed ONE LINE of the split path no matter how many
# reps they ran. Split-path STAB and ADV configs are added further below (V-S5 / V-S6).
# [Stage-3 round-3 item A] The veto is no longer universal on the STAB side: _SPLIT_A_LAUNCH_US is
# now the MEASURED 1.540 us instead of the 8.0 us placeholder, and the two legacy configs with
# n_blk_full == 2 (stab_B7_V4409_bv2048_gumbel, stab_B5_V10240_gumbel) were vetoed by the COST
# guard alone, so they now route to the split path. The other 6 legacy configs are blocked
# STRUCTURALLY (n_blk_total == 1 or n_blk_full < 2) and cannot move at any guard value. ADV_CASES
# is unaffected: all 8 of its shapes have n_blk_full <= 1.
# =========================================================================================

# ---- section 9.2.5 static assertion 2 baseline: sha256 of each FROZEN kernel's source
# segment (decorator line .. `return main`), extracted via ast. These are the rev2 delivered
# values (whole-file sha256 19e51e8d...); user constraint 3 requires them to never change.
REV3_FROZEN_SHA256 = {
    "r4_topk_stats": "d39090631598f07d1007917b310216b2f650a4ab4c3e7dd6b60b3e63a3a94831",
    "r4_mask_sample": "f787d1ad05a9fc26913b98de34f3a48ee67f730530ad6add8761082ef187a3d1",
    "r4_gumbel_sample": "ca90de8d0f6f4f69f366f380009f43915aa3c9f5b90430d683be239916683889",
}
REV3_NEW_KERNELS = (
    "r4_topk_stats_splitA",
    "r4_topk_stats_splitB",
    "r4_gumbel_sample_splitA",
    "r4_gumbel_sample_splitB",
)
REV3_UB_LIMIT = 196352  # generated InitBuffer(ascend_ub, N); section 4.5
_KERNEL_SRC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "r4_fused_sampling.py")

# ---- section 9.2.3 L0 matrix (15 cases). NOTHING here may be dropped (section 9.0 point 4).
# (name, B, V, K_MAX, BV, k, p, mu, need_min_p, expect_use_split, expect_S, force_S, tags)
#   expect_use_split / expect_S : structural assertions (section 9.2.5)
#   force_S                     : not None -> white-box call K1A/K1B with this S (E1 only; the
#                                 host plan would pick S=4 for that shape)
REV3_L0_CASES = [
    # A group -- split path basic function (4)
    ("l0_A1_split_s2_aligned", 16, 32768, 256, 4096, 20, 0.9, 0.1, True, True, 2, None,
     ["D-PATH-SPLIT", "D-SPLIT-S-MIN", "D-BITWISE-DUALPATH", "D-SHAPE-ALIGNED", "D-VALRANGE-M"]),
    ("l0_A2_split_smax_prodV", 4, 152064, 256, 4096, 40, 0.95, 0.05, True, True, 10, None,
     ["D-PATH-SPLIT", "D-SPLIT-S-MAX", "D-SPLIT-UNEVEN", "D-SPLIT-TAILOWNER", "D-BITWISE-DUALPATH",
      "D-VALRANGE-L", "D-SHAPE-TAIL-MID"]),
    ("l0_A3_split_s4_oneblk", 8, 16384, 256, 4096, 40, 0.95, 0.05, True, True, 4, None,
     ["D-PATH-SPLIT", "D-SPLIT-NBS1", "D-BITWISE-DUALPATH", "D-SHAPE-ALIGNED"]),
    ("l0_A4_split_k128_bv2048", 4, 16384, 128, 2048, 40, 0.95, 0.0, False, True, 8, None,
     ["D-PATH-SPLIT", "D-SPLIT-NBS1", "D-BITWISE-DUALPATH", "D-PARAM-K_MAX", "D-PARAM-BV",
      "D-SHAPE-ALIGNED"]),
    # B group -- TAIL % 32 != 0 under the split path (5); regression face of defects A/B.
    # TAIL spectrum {257,313,511,261,825} -> %32 remainders {1,25,31,5,25}: min, the rev2
    # defect value, and max. All five are mandatory (section 9.2.3 B-group note).
    ("l0_B1_split_tail257", 4, 20737, 256, 4096, 40, 0.95, 0.05, True, True, 6, None,
     ["D-PATH-SPLIT", "D-SPLIT-NBS1", "D-SPLIT-TAILOWNER", "D-BITWISE-DUALPATH", "D-SHAPE-PRIME"]),
    ("l0_B2_split_tail313", 4, 20793, 256, 4096, 40, 0.95, 0.05, True, True, 6, None,
     ["D-PATH-SPLIT", "D-SPLIT-NBS1", "D-SPLIT-TAILOWNER", "D-BITWISE-DUALPATH", "D-SHAPE-PRIME"]),
    ("l0_B3_split_tail511", 4, 20991, 256, 4096, 40, 0.95, 0.05, True, True, 6, None,
     ["D-PATH-SPLIT", "D-SPLIT-NBS1", "D-SPLIT-TAILOWNER", "D-BITWISE-DUALPATH", "D-SHAPE-PRIME"]),
    ("l0_B4_split_tail261", 4, 20741, 256, 4096, 40, 0.95, 0.05, True, True, 6, None,
     ["D-PATH-SPLIT", "D-SPLIT-NBS1", "D-SPLIT-TAILOWNER", "D-BITWISE-DUALPATH", "D-SHAPE-PRIME"]),
    ("l0_B5_split_tail825", 16, 152377, 256, 4096, 40, 0.95, 0.05, True, True, 2, None,
     ["D-PATH-SPLIT", "D-SPLIT-S-MIN", "D-SPLIT-TAILOWNER", "D-BITWISE-DUALPATH",
      "D-VALRANGE-L", "D-SHAPE-PRIME"]),
    # C group -- dispatch boundary (3). C1/C2 form the SHARP b_num 10 -> 11 edge that the
    # section 0.2 tightening of constraint 3 hinges on; C3 is constraint 3's literal edge.
    ("l0_C1_dispatch_bn10_split", 20, 32768, 256, 4096, 20, 0.9, 0.1, True, True, 2, None,
     ["D-PATH-SPLIT", "D-SPLIT-S-MIN", "D-BITWISE-DUALPATH", "D-SHAPE-ALIGNED", "D-VALRANGE-M"]),
    ("l0_C2_dispatch_bn11_current", 22, 32768, 256, 4096, 20, 0.9, 0.1, True, False, 1, None,
     ["D-PATH-CURRENT", "D-SHAPE-ALIGNED", "D-VALRANGE-M"]),
    ("l0_C3_dispatch_bn20_current", 40, 32768, 256, 4096, 20, 0.9, 0.1, True, False, 1, None,
     ["D-PATH-CURRENT", "D-SHAPE-ALIGNED", "D-VALRANGE-M"]),
    # D group -- dual-path bitwise, the direct executable form of constraint 2 (1)
    ("l0_D1_dualpath_bitwise", 4, 152064, 256, 4096, 40, 0.95, 0.05, True, True, 10, None,
     ["D-PATH-SPLIT", "D-PATH-CURRENT", "D-BITWISE-DUALPATH", "D-SPLIT-S-MAX", "D-VALRANGE-L"]),
    # E group -- S=1 degenerate, white-box (1). Decouples "is the DECOMPOSITION right" from
    # "is S>1 merging right": E1 pass + A2 fail => cross-split merge; E1 fail => decomposition.
    ("l0_E1_split_S1_degenerate", 2, 16384, 256, 4096, 40, 0.95, 0.0, False, True, 1, 1,
     ["D-PATH-SPLIT", "D-PATH-CURRENT", "D-BITWISE-DUALPATH", "D-SPLIT-DEGEN-S1", "D-SHAPE-ALIGNED"]),
    # F group -- theorem P3 back-substitution shrink (1). Naive S=min(S_cap,n_blk_total)=4
    # would give per-split [2,2,1,0] with an EMPTY split 3; the shrink yields S=3 -> [2,2,1].
    # This is the regression sentinel for section 5.8.2 step 4 (E-S6).
    ("l0_F1_split_shrink_noempty", 10, 20480, 256, 4096, 40, 0.95, 0.05, True, True, 3, None,
     ["D-PATH-SPLIT", "D-SPLIT-SHRINK", "D-SPLIT-UNEVEN", "D-BITWISE-DUALPATH", "D-SHAPE-ALIGNED"]),
]


def _rev3_bits(t):
    """fp32 -> raw bit pattern viewed as int32 so +-0 / -inf / NaN patterns compare exactly."""
    t = t.detach().cpu()
    return t.view(torch.int32) if t.dtype == torch.float32 else t


def _rev3_assert_bitwise(tag, name, actual, baseline):
    """Tier-2 (section 9.2.2): the 5 outputs must be BIT-identical to the frozen kernels."""
    ok = (actual.shape == baseline.shape and actual.dtype == baseline.dtype
          and bool(torch.equal(_rev3_bits(actual), _rev3_bits(baseline))))
    extra = ""
    if not ok:
        a, b = _rev3_bits(actual), _rev3_bits(baseline)
        if a.shape == b.shape:
            ne = int((a != b).sum().item())
            extra = f" mismatched={ne}/{a.numel()}"
            idx = (a != b).nonzero()
            if idx.numel():
                i = tuple(int(x) for x in idx[0])
                extra += f" first@{i} got={float(actual[i]):.9g} base={float(baseline[i]):.9g}"
        else:
            extra = f" shape {tuple(actual.shape)} vs {tuple(baseline.shape)}"
    print(f"[BITWISE_{'PASS' if ok else 'FAIL'}] {tag} {name} shape={tuple(actual.shape)} "
          f"dtype={str(actual.dtype).replace('torch.', '')} numel={actual.numel()}{extra}")
    return ok


def _rev3_make_inputs(B, V, k, p, mu):
    """The MANDATORY section 9.2.1 recipe: manual_seed(0) + randn * LOGIT_SCALE(8.0)."""
    torch.manual_seed(0)
    logits = torch.randn(B, V, dtype=torch.float32) * LOGIT_SCALE
    top_ks = torch.full((B,), k, dtype=torch.int32)
    top_ps = torch.full((B,), p, dtype=torch.float32)
    min_ps = torch.full((B,), mu, dtype=torch.float32)
    rand = torch.arange(B, dtype=torch.int32) * 7919 + 13  # deterministic per-row hash seed
    return logits, top_ks, top_ps, min_ps, rand


def _rev3_run_current(B, V, kmax, bv, nmp, logits, top_ks, top_ps, min_ps, rand, fuse):
    """BASELINE side: the three FROZEN kernels called directly (K1 -> K2m | K2g)."""
    lg = logits.npu()
    m, s, tau, valid = r4_topk_stats(B, V, kmax, bv, nmp)(lg, top_ks.npu(), top_ps.npu(), min_ps.npu())
    torch.npu.synchronize()
    if not fuse:
        w = r4_mask_sample(B, V, bv)(lg, m, s, tau)
        torch.npu.synchronize()
        return [x.cpu() for x in (m, s, tau, valid, w)]
    tok = r4_gumbel_sample(B, V, bv)(lg, tau, rand.npu())
    torch.npu.synchronize()
    return [x.cpu() for x in (m, s, tau, valid, tok)]


def _rev3_run_split(B, V, kmax, bv, nmp, logits, top_ks, top_ps, min_ps, rand, fuse,
                    S=None, NBS=None, S_TAIL=None):
    """SPLIT side. S=None -> go through the host wrapper (exercises the real dispatch).
    S given -> white-box: call K1A/K1B/K2gA/K2gB directly with a forced plan (E1)."""
    lg = logits.npu()
    if S is None:
        out = r4_fused_sampling(lg, top_ks.npu(), top_ps.npu(), min_ps.npu(), nmp, kmax, bv,
                                rand=rand.npu(), FUSE_GUMBEL=fuse)
        torch.npu.synchronize()
        return [x.cpu() for x in out]
    pv = torch.empty((B * S, kmax), dtype=torch.float32, device=lg.device)
    r4_topk_stats_splitA(B, V, kmax, bv, S, NBS, S_TAIL)(lg, pv)
    m, s, tau, valid = r4_topk_stats_splitB(B, V, kmax, bv, S, nmp)(
        lg, top_ks.npu(), top_ps.npu(), min_ps.npu(), pv)
    torch.npu.synchronize()
    if not fuse:
        w = r4_mask_sample(B, V, bv)(lg, m, s, tau)  # K2m is NOT split (constraint 4)
        torch.npu.synchronize()
        return [x.cpu() for x in (m, s, tau, valid, w.cpu())]
    pv_v = torch.empty((B * S,), dtype=torch.float32, device=lg.device)
    pv_i = torch.empty((B * S,), dtype=torch.float32, device=lg.device)
    r4_gumbel_sample_splitA(B, V, bv, S, NBS)(lg, tau, rand.npu(), pv_v, pv_i)
    tok = r4_gumbel_sample_splitB(B, S)(pv_v, pv_i)
    torch.npu.synchronize()
    return [x.cpu() for x in (m, s, tau, valid, tok)]


def _rev3_plan_for(B, V, kmax, bv, force_S=None):
    """(use_split, S, NBS, S_TAIL) for a case; force_S re-derives NBS/S_TAIL white-box."""
    use_split, S, NBS, S_TAIL = _r4_split_plan(B, V, kmax, bv)
    if force_S is not None:
        S = int(force_S)
        nbf = V // bv
        tail = V - nbf * bv
        nbt = nbf + (1 if tail > 0 else 0)
        NBS = -(-nbt // S)
        S_TAIL = nbf // NBS if tail > 0 else -1
        use_split = True  # white-box: we drive the split kernels regardless of the guard
    return use_split, S, NBS, S_TAIL


def _rev3_run_l0_case(case):
    """One L0 case x BOTH tasks. Returns (ok, n_bitwise, n_tier1, detail)."""
    (name, B, V, kmax, bv, k, p, mu, nmp, exp_split, exp_S, force_S, tags) = case
    logits, top_ks, top_ps, min_ps, rand = _rev3_make_inputs(B, V, k, p, mu)
    use_split, S, NBS, S_TAIL = _rev3_plan_for(B, V, kmax, bv, force_S)
    nbf = V // bv
    tail = V - nbf * bv
    nbt = nbf + (1 if tail > 0 else 0)
    per = [max(0, min(nbt, (s + 1) * NBS) - s * NBS) for s in range(S)]
    bn = -(-B // VEC_NUM)

    # ---- structural assertions (section 9.2.5) ----
    struct_ok, struct_notes = True, []
    if use_split != exp_split:
        struct_ok = False
        struct_notes.append(f"use_split={use_split} expected {exp_split}")
    if exp_S != S:  # (exp_S on the left: ruff SIM300 treats the all-caps `S` as a constant)
        struct_ok = False
        struct_notes.append(f"S={S} expected {exp_S}")
    if exp_split and min(per or [1]) < 1:  # theorem P3 / E-S6: no empty split
        struct_ok = False
        struct_notes.append(f"EMPTY SPLIT per={per}")
    if not struct_ok:
        print(f"[PRECISION_FAIL] {name} STRUCTURAL {'; '.join(struct_notes)} "
              f"(B={B} V={V} K_MAX={kmax} BV={bv} S={S} NBS={NBS} S_TAIL={S_TAIL} per={per}) {tags}")

    all_ok = struct_ok
    n_bw = n_bw_ok = n_t1 = n_t1_ok = 0
    for fuse in (False, True):
        task = "fused" if fuse else "mask"
        t0 = time.time()
        got = _rev3_run_split(B, V, kmax, bv, nmp, logits, top_ks, top_ps, min_ps, rand, fuse,
                              S=S, NBS=NBS, S_TAIL=S_TAIL) if force_S is not None else \
            _rev3_run_split(B, V, kmax, bv, nmp, logits, top_ks, top_ps, min_ps, rand, fuse)
        base = _rev3_run_current(B, V, kmax, bv, nmp, logits, top_ks, top_ps, min_ps, rand, fuse)
        run_s = time.time() - t0
        names = ("m_out", "s_out", "tau_out", "valid", "tok_out" if fuse else "w_out")

        # Tier 2 -- only meaningful when the split path actually ran (for a current-path case
        # both sides ARE the frozen kernels, so it is trivially true and carries no signal;
        # section 9.2.2 "Tier 2 applicability").
        if exp_split:
            for nm, a, b in zip(names, got, base):
                n_bw += 1
                n_bw_ok += _rev3_assert_bitwise(name, f"{task}/{nm}", a, b)

        # Tier 1 -- always
        if fuse:
            gm, gs, gtau, gvalid, _gw = golden_r4(logits, top_ks, top_ps, min_ps, nmp, kmax)
            gold = [gm, gs, gtau, gvalid, golden_gumbel_sample(logits, gtau, rand)]
        else:
            gold = list(golden_r4(logits, top_ks, top_ps, min_ps, nmp, kmax))
        dts = ("float32", "float32", "float32", "int32", "int32" if fuse else "float32")
        t1 = []
        for nm, a, g, d in zip(names, got, gold, dts):
            n_t1 += 1
            passed, ratio, mx = check_precision(a, g, d)
            n_t1_ok += passed
            t1.append(f"{nm}[r={ratio:.4f},m={mx:.1e}]")
            if not passed:
                print(f"[PRECISION_FAIL] {name} {task}/{nm} tier1 ratio={ratio:.6f} max={mx:.4e} {tags}")
        # section 9.2.1 recipe guard: valid_ref must be all-1 or the whole case is a false fail
        vref_all1 = bool((gold[3] == 1).all().item())
        if not vref_all1:
            t1.append("!!valid_ref_not_all_1(check_LOGIT_SCALE_recipe)")
        all_ok &= vref_all1
        all_ok &= (n_t1_ok == n_t1)
        if exp_split:
            all_ok &= (n_bw_ok == n_bw)
        print(f"[PRECISION_{'PASS' if (vref_all1 and n_t1_ok == n_t1 and (n_bw_ok == n_bw or not exp_split)) else 'FAIL'}] "
              f"{name} {task} B={B} V={V} K_MAX={kmax} BV={bv} k={k} p={p} mu={mu} min_p={nmp} "
              f"({run_s:.1f}s) path={'SPLIT' if exp_split else 'CURRENT'} S={S} NBS={NBS} S_TAIL={S_TAIL} "
              f"grid_A={bn * S if exp_split else bn} per={per} tier1=[{' '.join(t1)}] "
              f"tier2_bitwise={n_bw_ok}/{n_bw} {tags}")
    return all_ok, n_bw, n_bw_ok, n_t1, n_t1_ok, \
        f"S={S} NBS={NBS} S_TAIL={S_TAIL} per={per} path={'SPLIT' if exp_split else 'CURRENT'}"


def test_r4_rev3_l0():
    """rev3 L0 gate: the section 9.2.3 15-case matrix x 2 tasks = 30 runs, TWO-tier judged.

    15 shapes x 2 tasks (mask / fused). Every split-path case is judged by Tier 1 (mixed
    tolerance vs golden) AND Tier 2 (bit-identical vs the frozen kernels). Current-path cases
    (C2/C3) are judged by Tier 1 + a structural `use_split == False` assertion, because there
    Tier 2 is trivially true (section 9.2.2).
    """
    print("=" * 80)
    print(f"[L0-REV3] split-V gate -- {len(REV3_L0_CASES)} cases x 2 tasks "
          f"(Tier-1 golden + Tier-2 bitwise; LOGIT_SCALE={LOGIT_SCALE})")
    print("=" * 80)
    all_ok = True
    tot_bw = tot_bw_ok = tot_t1 = tot_t1_ok = 0
    n_split = 0
    for case in REV3_L0_CASES:
        try:
            ok, n_bw, n_bw_ok, n_t1, n_t1_ok, info = _rev3_run_l0_case(case)
            tot_bw += n_bw
            tot_bw_ok += n_bw_ok
            tot_t1 += n_t1
            tot_t1_ok += n_t1_ok
            n_split += int(case[9])
            all_ok &= ok
        except Exception as e:  # noqa: BLE001
            import traceback

            print(f"[PRECISION_FAIL] {case[0]}: {type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc()
            all_ok = False
    print(f"[L0-REV3] SUMMARY cases={len(REV3_L0_CASES)} (split-path={n_split}, "
          f"current-path={len(REV3_L0_CASES) - n_split}) runs={2 * len(REV3_L0_CASES)} "
          f"tier1={tot_t1_ok}/{tot_t1} tier2_bitwise={tot_bw_ok}/{tot_bw}")
    print(f"[BITWISE_{'PASS' if tot_bw_ok == tot_bw else 'FAIL'}] [L0-REV3] all Tier-2 "
          f"dual-path comparisons ({tot_bw_ok}/{tot_bw})")
    return all_ok


# ---- section 9.2.5 static assertions 1..4 (compile-time only; no NPU execution needed) ----
_REV3_SZ = {"float": 4, "int": 4, "uint8_t": 1, "int32_t": 4, "float16_t": 2,
            "uint16_t": 2, "int16_t": 2, "uint32_t": 4, "int8_t": 1, "half": 2}
_REV3_PAT_UB = re.compile(r"ascend_ub\.GetWithOffset<\s*([A-Za-z0-9_:]+)\s*>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)")
_REV3_PAT_TOPK = re.compile(r"TopK<\w+>\([^;]*?,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")
_REV3_PAT_U2U = re.compile(r"copy_ub_to_ub<[^,]+,[^,]+,\s*(\d+)>")
_REV3_PAT_FOR = re.compile(r"for\s*\(\s*int32_t\s+(\w+)\s*=\s*[^;]+?;\s*\1\s*<\s*([^;]+?)\s*;")
_REV3_PAT_REGION = re.compile(r"InitBuffer\(ascend_ub,\s*(\d+)\)")
# Representative specializations for assertions 3/4: both contract specializations
# (K_MAX,BV) in {(256,4096),(128,2048)}, aligned + non-aligned tail, NBS=1 + NBS>1,
# S=2 .. S=10, and the theorem-P3 shrink shape.
_REV3_STATIC_SHAPES = (
    ("A3_K256_BV4096_NBS1", 8, 16384, 256, 4096),
    ("A4_K128_BV2048", 4, 16384, 128, 2048),
    ("B2_TAIL313", 4, 20793, 256, 4096),
    ("A2_S10_grid20", 4, 152064, 256, 4096),
    ("F1_shrink_S3", 10, 20480, 256, 4096),
    ("B5_TAIL825_S2", 16, 152377, 256, 4096),
    # ---- Stage-3 round-3 item A: the two geometries the MEASURED guard newly admits ----
    # _SPLIT_A_LAUNCH_US went from the conservative 8.0 us placeholder to the measured 1.540 us,
    # which routes cann-bench case 6 (S=3/NBS=1/S_TAIL=2, TAIL=2048 mid tail) and case 11
    # (S=2/NBS=1/S_TAIL=-1, aligned) onto the split path. Before this change those two geometries
    # were only ever checked by an OUT-OF-BAND Stage-2 probe, so the delivered gate did not own
    # the "UB fits + padCount==0 + 32B ub2ub + no dynamic bounds" property for shapes the contract
    # really dispatches here. Adding them makes the gate own it (38 -> 48 STATIC_PASS lines).
    ("cb6_S3_NBS1_STAIL2", 8, 10240, 256, 4096),
    ("cb11_S2_NBS1_aligned", 8, 8192, 256, 4096),
)


def _rev3_ub_usage(src):
    """Measured UB = max(off + n*sizeof(T)) over every ascend_ub.GetWithOffset<T>(n, off).

    This is exactly what ascend_memory_planning.cc:582-585 computes for its DLOG line, i.e.
    MEASUREMENT rather than the analytic model (section 4.5: the model is only an upper bound
    once liveness reuse kicks in, so feasibility must be decided by measurement).
    """
    det = []
    for m in _REV3_PAT_UB.finditer(src):
        esz = _REV3_SZ.get(m.group(1).split("::")[-1])
        if esz is None:
            raise ValueError(f"unknown dtype {m.group(1)}")
        det.append((int(m.group(3)), int(m.group(2)) * esz))
    if not det:
        raise ValueError("no GetWithOffset found -- parse failure")
    reg = _REV3_PAT_REGION.search(src)
    return max(o + s for o, s in det), len(det), (int(reg.group(1)) if reg else REV3_UB_LIMIT)


def _rev3_frozen_segments():
    """{kernel_name: (start_line, end_line, sha256)} for the three frozen kernels, via ast."""
    import ast as _ast
    import hashlib as _hashlib

    src = open(_KERNEL_SRC_PATH, encoding="utf-8").read()
    lines = src.split("\n")
    out = {}
    for node in _ast.parse(src).body:
        if isinstance(node, _ast.FunctionDef) and node.name in REV3_FROZEN_SHA256:
            start = node.lineno - 1
            if node.decorator_list:
                start = min(start, min(d.lineno for d in node.decorator_list) - 1)
                while start > 0 and not lines[start].lstrip().startswith("@"):
                    start -= 1
            txt = "\n".join(lines[start:node.end_lineno])
            out[node.name] = (start + 1, node.end_lineno, _hashlib.sha256(txt.encode()).hexdigest())
    return out


def _rev3_decorator_pass_configs():
    """{new_kernel_name: ast-unparsed pass_configs expression} -- assertion 1 (source level)."""
    import ast as _ast

    src = open(_KERNEL_SRC_PATH, encoding="utf-8").read()
    out = {}
    for node in _ast.parse(src).body:
        if not isinstance(node, _ast.FunctionDef) or node.name not in REV3_NEW_KERNELS:
            continue
        for d in node.decorator_list:
            if isinstance(d, _ast.Call) and getattr(d.func, "attr", getattr(d.func, "id", None)) == "jit":
                for kw in d.keywords:
                    if kw.arg == "pass_configs":
                        out[node.name] = _ast.unparse(kw.value)
    return out


def test_r4_rev3_static():
    """Section 9.2.5 static assertions 1-4. Runs BEFORE the on-device L0 cases.

    1. the 4 new kernels' @tilelang.jit pass_configs == R4_PASS_CONFIGS, which contains
       tl.ascend_memory_planning=True (R-MEM: the sibling path at ascend_memory_planning.cc:598
       is check_overflow=false, i.e. a SILENT UB overflow; only this config makes it LOG(FATAL))
    2. the 3 frozen kernels' source segments are byte-identical to the rev2 delivery
    3. measured UB <= the generated region limit, for every new kernel x representative shape
    4. every generated TopK call site has padCount == 0 AND every copy_ub_to_ub byte count is
       a multiple of 32 (defects A and B structurally unreachable) -- plus: zero dynamic loop
       bounds (section 3.5.1).
    """
    print("=" * 80)
    print("[STATIC-REV3] section 9.2.5 assertions 1-4 (compile-time; no NPU execution)")
    print("=" * 80)
    ok = True

    # ---- assertion 1 ----
    pcs = _rev3_decorator_pass_configs()
    a1 = len(pcs) == len(REV3_NEW_KERNELS) and all(v == "R4_PASS_CONFIGS" for v in pcs.values())
    a1 &= R4_PASS_CONFIGS.get("tl.ascend_memory_planning") is True
    for kn in REV3_NEW_KERNELS:
        print(f"[STATIC_{'PASS' if pcs.get(kn) == 'R4_PASS_CONFIGS' else 'FAIL'}] assert-1 "
              f"{kn} pass_configs={pcs.get(kn)!r}")
    print(f"[STATIC_{'PASS' if R4_PASS_CONFIGS.get('tl.ascend_memory_planning') is True else 'FAIL'}] "
          f"assert-1 R4_PASS_CONFIGS['tl.ascend_memory_planning']="
          f"{R4_PASS_CONFIGS.get('tl.ascend_memory_planning')} (R-MEM safety net)")
    print(f"[PRECISION_{'PASS' if a1 else 'FAIL'}] assert-1 all 4 new kernels carry "
          f"pass_configs=R4_PASS_CONFIGS with ascend_memory_planning=True")
    ok &= a1

    # ---- assertion 2 ----
    segs = _rev3_frozen_segments()
    a2 = len(segs) == 3
    for kn, want in REV3_FROZEN_SHA256.items():
        got = segs.get(kn, (None, None, None))[2]
        same = got == want
        a2 &= same
        ln = segs.get(kn, ("?", "?", None))
        print(f"[STATIC_{'PASS' if same else 'FAIL'}] assert-2 frozen {kn} L{ln[0]}-{ln[1]} "
              f"sha256={str(got)[:16]}... {'== rev2 delivery' if same else f'*** != {want[:16]}... ***'}")
    print(f"[PRECISION_{'PASS' if a2 else 'FAIL'}] assert-2 the 3 delivered kernels are "
          f"byte-identical (user constraint 3)")
    ok &= a2

    # ---- assertions 3 + 4 (need generated source; get_kernel_source() needs no NPU device) ----
    worst = {}
    a3 = a4 = True
    n_src = 0
    for label, B, V, kmax, bv in _REV3_STATIC_SHAPES:
        _u, S, NBS, S_TAIL = _r4_split_plan(B, V, kmax, bv)
        if S < 2:  # geometrically unsplittable -> the split kernels are not used for it
            S = max(2, S)
            nbt = -(-V // bv)
            NBS = -(-nbt // S)
            S = -(-nbt // NBS)
            NBS = -(-nbt // S)
            nbf = V // bv
            tl_ = V - nbf * bv
            S_TAIL = nbf // NBS if tl_ > 0 else -1
        jobs = (("K1A", r4_topk_stats_splitA, (B, V, kmax, bv, S, NBS, S_TAIL)),
                ("K1B", r4_topk_stats_splitB, (B, V, kmax, bv, S, False)),
                ("K1B", r4_topk_stats_splitB, (B, V, kmax, bv, S, True)),
                ("K2gA", r4_gumbel_sample_splitA, (B, V, bv, S, NBS)),
                ("K2gB", r4_gumbel_sample_splitB, (B, S)))
        for kn, fn, args in jobs:
            try:
                src = fn(*args).get_kernel_source()
            except BaseException as e:  # noqa: BLE001  (LOG(FATAL) surfaces as SystemExit)
                print(f"[STATIC_FAIL] assert-3/4 {label} {kn} COMPILE-FAIL "
                      f"{type(e).__name__}: {str(e)[:160]}")
                a3 = a4 = False
                continue
            n_src += 1
            tot, nbuf, region = _rev3_ub_usage(src)
            topk = [(int(x), int(y), int(z)) for x, y, z in _REV3_PAT_TOPK.findall(src)]
            bad_pad = [t for t in topk if t[1] * 32 - t[2] != 0]
            u2u = sorted({int(c) for c in _REV3_PAT_U2U.findall(src)})
            bad_u2u = [c for c in u2u if (c * 4) % 32 != 0]
            dyn = [(v, hi) for v, hi in _REV3_PAT_FOR.findall(src) if not re.fullmatch(r"-?\d+", hi.strip())]
            if kn not in worst or tot > worst[kn][0]:
                worst[kn] = (tot, region, label)
            if tot > region:
                a3 = False
            if bad_pad or bad_u2u or dyn:
                a4 = False
            print(f"[STATIC_{'PASS' if (tot <= region and not bad_pad and not bad_u2u and not dyn) else 'FAIL'}] "
                  f"assert-3/4 {label:22s} {kn:5s} S={S} NBS={NBS} UB={tot}/{region} "
                  f"({100.0 * tot / region:.2f}%) bufs={nbuf} topk={sorted(set(topk))} "
                  f"padCount_all0={not bad_pad} ub2ub={u2u} all32B={not bad_u2u} dyn_bounds={dyn}")
    for kn in sorted(worst):
        tot, region, label = worst[kn]
        print(f"[STATIC_INFO] worst measured UB {kn:5s} = {tot}/{region} "
              f"({100.0 * tot / region:.2f}%) headroom={region - tot} at {label}")
    print(f"[PRECISION_{'PASS' if a3 else 'FAIL'}] assert-3 measured UB <= region for all "
          f"{n_src} generated sources (limit {REV3_UB_LIMIT})")
    print(f"[PRECISION_{'PASS' if a4 else 'FAIL'}] assert-4 every TopK padCount==0, every "
          f"copy_ub_to_ub 32B-aligned, zero dynamic loop bounds")
    ok &= a3 and a4
    return ok


# ---- rev3 scenario-B extension: split-path L1 / L2 / Boundary coverage -------------------
# WHY: the pre-existing L1/L2/Boundary suites were written for a single implementation path.
# Many of their shapes DO happen to route to the split path now (e.g. l1_aligned_base
# B=8,V=16384 -> S=4; guard_flat_dist B=4,V=16384 -> S=4), but nothing asserted it, and three
# rev3-specific risks had no coverage at all:
#   * V-S4 -- "a whole split's V-interval is -inf" (DESIGN.md 9.3 rev3 addendum / 10.2): K1B
#     merges S per-split top-K lists whose -inf padding could in principle meet a -inf m_run.
#   * Tier-2 bitwise at L1 -- the tolerance gate cannot see a reduction-order bug (E-S4).
#   * L2 rejection on a split-path shape -- proving the dispatch runs AFTER the C3/C6/C7/dtype
#     pre-checks, so no illegal input can slip through the new path (DESIGN.md 5.7).
# Every shape below was CPU-pre-verified with golden_r4 to yield valid_ref all-1 (9.2.1).
# (name, B, V, K_MAX, BV, k, p, mu, need_min_p, scale, offset, tags)
REV3_L1_CASES = [
    # k=1 -> tau_k = run_v[0]: the cross-split merge must get rank 0 exactly right.
    ("l1r3_split_k1", 8, 16384, 256, 4096, 1, 1.0, 0.0, False, 8.0, 0.0,
     ["D-PATH-SPLIT", "D-BITWISE-DUALPATH", "D-PARAM-top_k", "D-PARAM-top_p", "D-SPLIT-NBS1"]),
    # k == K_MAX -> the D-TIE saturated-list fast path on the split route.
    ("l1r3_split_k_eq_kmax", 8, 16384, 256, 4096, 256, 1.0, 0.0, False, 8.0, 0.0,
     ["D-PATH-SPLIT", "D-BITWISE-DUALPATH", "D-PARAM-top_k", "D-PARAM-top_p", "D-SPLIT-NBS1"]),
    # strong min-p: tau_u = m + ln(mu) binds; exercises the T2 branch under split.
    ("l1r3_split_minp_strong", 8, 16384, 256, 4096, 60, 0.9, 0.2, True, 8.0, 0.0,
     ["D-PATH-SPLIT", "D-BITWISE-DUALPATH", "D-PARAM-min_p", "D-PARAM-NEED_MIN_P",
      "D-PARAM-need_min_p_sampling", "D-PARAM-top_k", "D-PARAM-top_p"]),
    # second contract specialization (K_MAX=128, BV=2048) with k == K_MAX.
    ("l1r3_split_k128bv2048_keq", 4, 16384, 128, 2048, 128, 1.0, 0.0, False, 8.0, 0.0,
     ["D-PATH-SPLIT", "D-BITWISE-DUALPATH", "D-PARAM-K_MAX", "D-PARAM-BV", "D-PARAM-top_k",
      "D-SPLIT-NBS1"]),
    # S=2 with NBS>1 (each split owns 4 blocks) -- the in-split merge chain, not just cross-split.
    ("l1r3_split_s2_b16", 16, 32768, 256, 4096, 20, 0.9, 0.1, True, 8.0, 0.0,
     ["D-PATH-SPLIT", "D-BITWISE-DUALPATH", "D-SPLIT-S-MIN", "D-PARAM-min_p", "D-VALRANGE-M"]),
    # odd B on the split path: b_num=3 -> grid_A=12, so the row guard (row < B) is live in K1A.
    ("l1r3_split_oddB5", 5, 16384, 256, 4096, 40, 0.95, 0.0, False, 8.0, 0.0,
     ["D-PATH-SPLIT", "D-BITWISE-DUALPATH", "D-SHAPE-TAIL-1", "D-SPLIT-NBS1"]),
    # BV=2048 with K_MAX=256: a (BV,K_MAX) pairing outside the two calibrated cost points, so
    # the guard falls back to the default t_est -- the dispatch must still be correct.
    ("l1r3_split_bv2048_k256", 4, 16384, 256, 2048, 40, 0.95, 0.0, False, 8.0, 0.0,
     ["D-PATH-SPLIT", "D-BITWISE-DUALPATH", "D-PARAM-BV", "D-SPLIT-NBS1"]),
    # theorem-P3 shrink shape with k=1 (uneven per-split [2,2,1]).
    ("l1r3_split_shrink_k1", 10, 20480, 256, 4096, 1, 1.0, 0.0, False, 8.0, 0.0,
     ["D-PATH-SPLIT", "D-BITWISE-DUALPATH", "D-SPLIT-SHRINK", "D-SPLIT-UNEVEN", "D-PARAM-top_k"]),
    # value ranges on the split path (large / small / asymmetric magnitudes).
    ("l1r3_split_valrange_L", 4, 16384, 256, 4096, 40, 0.95, 0.0, False, 1000.0, 0.0,
     ["D-PATH-SPLIT", "D-BITWISE-DUALPATH", "D-VALRANGE-L"]),
    ("l1r3_split_valrange_S", 4, 16384, 256, 4096, 40, 0.95, 0.0, False, 5.0, 0.0,
     ["D-PATH-SPLIT", "D-BITWISE-DUALPATH", "D-VALRANGE-S"]),
    ("l1r3_split_valrange_ASYM", 8, 16384, 256, 4096, 40, 0.95, 0.0, False, 6.0, -4.0,
     ["D-PATH-SPLIT", "D-BITWISE-DUALPATH", "D-VALRANGE-ASYM"]),
    # non-aligned tail (313) with a non-zero S_TAIL, min-p on.
    ("l1r3_split_prime_tail313", 4, 20793, 256, 4096, 20, 0.9, 0.05, True, 8.0, 0.0,
     ["D-PATH-SPLIT", "D-BITWISE-DUALPATH", "D-SHAPE-PRIME", "D-SPLIT-TAILOWNER",
      "D-PARAM-min_p", "D-PARAM-top_k"]),
    # K_MAX=128 with BV=4096 (only K_MAX varies): AT = _aligned_topk(128) = 256.
    ("l1r3_split_kmax128_bv4096", 4, 16384, 128, 4096, 100, 0.95, 0.0, False, 8.0, 0.0,
     ["D-PATH-SPLIT", "D-BITWISE-DUALPATH", "D-PARAM-K_MAX", "D-PARAM-top_k"]),
    # ---- Stage-3 round-3 item A: RUNTIME split-path coverage for the two newly admitted cases ----
    # Relaxing _SPLIT_A_LAUNCH_US to the measured 1.540 us routes cann-bench case 6 and case 11
    # onto the split path. Until this change NEITHER shape had any runtime split-path precision
    # coverage at all: case 11's shape (8, 8192) appeared in NO test table, and case 6's shape
    # (8, 10240) appeared only as l1_tailmid_V / l1b_tailmid_V, which judge against the golden
    # (Tier 1) but never against the frozen kernels (Tier 2) -- and Tier 1 alone cannot see a
    # reduction-order bug (E-S4). Without these two entries the guard relaxation would move two
    # contract cases into a code path with no dual-path bitwise gate. Both are judged by
    # _rev3_dual_compare with expect_split=True + blocking=True, so they run BOTH tasks (mask and
    # fused), compare Tier 1 vs golden_r4/golden_gumbel_sample AND Tier 2 bit-pattern vs the three
    # frozen kernels, and FAIL LOUDLY if a future guard change ever routes them back to the
    # delivered path. Both shapes were CPU-pre-verified under the section 9.2.1 recipe
    # (manual_seed(0) + randn*8.0) to yield valid_ref all-1 with no golden NaN
    # (perf_tuning/round3_itemA_guard/preflight.json, P2).
    # cb6: S=3, NBS=1, S_TAIL=2 -> TAIL=2048 (V % BV == BV//2) owned by the LAST split, so the
    #      tail-owning split carries 0 full blocks while splits 0/1 carry 1 each: an unbalanced
    #      full-block load with a non-zero S_TAIL, a combination no pre-existing L1 case had.
    ("l1r3_split_cb6_B8_V10240", 8, 10240, 256, 4096, 40, 0.95, 0.0, False, 8.0, 0.0,
     ["D-PATH-SPLIT", "D-BITWISE-DUALPATH", "D-SHAPE-TAIL-MID", "D-SPLIT-NBS1",
      "D-SPLIT-TAILOWNER", "D-PARAM-top_k", "D-PARAM-top_p"]),
    # cb11: S=2, NBS=1, S_TAIL=-1 -> perfectly aligned (V % BV == 0), the minimum splittable S,
    #      and the smallest V of any split-path case: the tightest geometry the guard admits.
    ("l1r3_split_cb11_B8_V8192", 8, 8192, 256, 4096, 40, 0.95, 0.0, False, 8.0, 0.0,
     ["D-PATH-SPLIT", "D-BITWISE-DUALPATH", "D-SHAPE-ALIGNED", "D-SPLIT-S-MIN", "D-SPLIT-NBS1",
      "D-PARAM-top_k", "D-PARAM-top_p"]),
]

# V-S4 (DESIGN.md 9.3 rev3 addendum / 10.2): a WHOLE split's V-interval forced to -inf.
# n_ninf_splits=0 means the sparse -inf control (never a whole split, never a whole row).
# Section 9.3 forbids an all -inf ROW as the only special-value case (it hides merge/index
# errors), and every recipe below was CPU-checked to leave at least one finite value per row
# and to produce NO NaN in golden.
# (name, B, V, K_MAX, BV, k, p, mu, need_min_p, n_ninf_splits, tags)
REV3_BND_CASES = [
    ("bnd_split0_all_neginf", 4, 20793, 256, 4096, 40, 0.95, 0.05, True, 1,
     ["D-PATH-SPLIT", "D-SPECIAL-INF", "D-BITWISE-DUALPATH", "D-SPLIT-NBS1", "D-SPLIT-TAILOWNER"]),
    ("bnd_split0to4_all_neginf", 4, 20793, 256, 4096, 40, 0.95, 0.05, True, 5,
     ["D-PATH-SPLIT", "D-SPECIAL-INF", "D-BITWISE-DUALPATH", "D-SPLIT-NBS1", "D-SPLIT-TAILOWNER"]),
    ("bnd_splitmid_all_neginf", 8, 16384, 256, 4096, 40, 0.95, 0.05, True, 2,
     ["D-PATH-SPLIT", "D-SPECIAL-INF", "D-BITWISE-DUALPATH", "D-SPLIT-NBS1"]),
    ("bnd_sparse_neginf_split", 4, 20793, 256, 4096, 40, 0.95, 0.05, True, 0,
     ["D-PATH-SPLIT", "D-SPECIAL-INF", "D-BITWISE-DUALPATH", "D-SPLIT-TAILOWNER"]),
]


def _rev3_bnd_logits(B, V, bv, nbs, n_ninf_splits):
    """Build a V-S4 recipe: the first n_ninf_splits*NBS*BV columns are -inf (whole splits)."""
    torch.manual_seed(0)
    logits = torch.randn(B, V, dtype=torch.float32) * LOGIT_SCALE
    if n_ninf_splits > 0:
        logits[:, : n_ninf_splits * nbs * bv] = float("-inf")
    else:  # sparse control: finite values + sparse -inf, never a whole split or row
        logits[0, :50] = float("-inf")
        logits[2, 100] = float("-inf")
    return logits


def _rev3_dual_compare(name, B, V, kmax, bv, k, p, mu, nmp, logits, tags,
                       level="l1", expect_split=True, blocking=True,
                       expect_guard_degrade=False):
    """rev3 dual-path runner for the L1 / Boundary extensions (scenario B).

    For BOTH tasks (mask / fused) it runs the split path (through the host wrapper, so the
    real dispatch is exercised) and the delivered path (the three frozen kernels directly),
    then judges Tier 1 (vs golden, section 9.3 mixed tolerance) and Tier 2 (bit-pattern
    equality between the two paths, section 9.2.2). Returns ok (bool).

    blocking=True  -> prints [PRECISION_*] (counts toward the exit code)
    blocking=False -> prints [BOUNDARY_*]  (non-blocking, per the layer contract)

    expect_guard_degrade=True documents a KNOWN, PRE-EXISTING deviation (see V-S4 below):
    Tier 1 will miss, so the case reports [BOUNDARY_WARN] for the golden comparison, and the
    two properties rev3 is actually responsible for are asserted SEPARATELY and reported on
    their own line -- (i) the two paths are still bit-identical, (ii) the D-GUARD degrades
    safely (valid all-0 on both paths, so the host falls back instead of returning a wrong
    token). Neither property is weakened by allowing the Tier-1 WARN.
    """
    P, B_ = ("PRECISION", "BITWISE") if blocking else ("BOUNDARY", "BOUNDARY")
    top_ks = torch.full((B,), k, dtype=torch.int32)
    top_ps = torch.full((B,), p, dtype=torch.float32)
    min_ps = torch.full((B,), mu, dtype=torch.float32)
    rand = torch.arange(B, dtype=torch.int32) * 7919 + 13
    use_split, S, NBS, S_TAIL = _r4_split_plan(B, V, kmax, bv)
    if expect_split and not use_split:
        # The point of these cases is split-path coverage; if a guard change ever routes them
        # to the delivered path the coverage silently vanishes -> say so loudly.
        print(f"[{P}_{'FAIL' if blocking else 'WARN'}] {level} {name}: expected the SPLIT path "
              f"but _r4_split_plan returned use_split=False (S={S}) -- split coverage lost {tags}")
    ok = True
    for fuse in (False, True):
        task = "fused" if fuse else "mask"
        t0 = time.time()
        got = _rev3_run_split(B, V, kmax, bv, nmp, logits, top_ks, top_ps, min_ps, rand, fuse)
        base = _rev3_run_current(B, V, kmax, bv, nmp, logits, top_ks, top_ps, min_ps, rand, fuse)
        run_s = time.time() - t0
        names = ("m_out", "s_out", "tau_out", "valid", "tok_out" if fuse else "w_out")
        if fuse:
            gm, gs, gtau, gvalid, _gw = golden_r4(logits, top_ks, top_ps, min_ps, nmp, kmax)
            gold = [gm, gs, gtau, gvalid, golden_gumbel_sample(logits, gtau, rand)]
        else:
            gold = list(golden_r4(logits, top_ks, top_ps, min_ps, nmp, kmax))
        dts = ("float32", "float32", "float32", "int32", "int32" if fuse else "float32")
        t1, n_t1_ok = [], 0
        for nm, a, g, d in zip(names, got, gold, dts):
            passed, ratio, mx = check_precision(a, g, d)
            n_t1_ok += passed
            t1.append(f"{nm}[r={ratio:.4f},m={mx:.1e}]")
        n_bw = n_bw_ok = 0
        for nm, a, b in zip(names, got, base):
            n_bw += 1
            same = (a.shape == b.shape and a.dtype == b.dtype
                    and bool(torch.equal(_rev3_bits(a), _rev3_bits(b))))
            n_bw_ok += same
            if not same:
                print(f"[{B_}_FAIL] {name} {task}/{nm}: split path differs from the frozen "
                      f"kernels at the bit level (mismatched="
                      f"{int((_rev3_bits(a) != _rev3_bits(b)).sum().item())}/{a.numel()}) {tags}")
        tier1_ok = n_t1_ok == len(names)
        tier2_ok = n_bw_ok == n_bw
        ok &= tier1_ok and tier2_ok
        tag = "PASS" if (tier1_ok and tier2_ok) else ("FAIL" if blocking else "WARN")
        note = ""
        if expect_guard_degrade and not tier1_ok:
            note = ("  <- EXPECTED pre-existing degrade, NOT a rev3 regression (see V-S4 note)")
        print(f"[{P}_{tag}] {level} {name} {task} B={B} V={V} K_MAX={kmax} BV={bv} k={k} p={p} "
              f"mu={mu} min_p={nmp} ({run_s:.1f}s) path={'SPLIT' if use_split else 'CURRENT'} "
              f"S={S} NBS={NBS} S_TAIL={S_TAIL} tier1=[{' '.join(t1)}] "
              f"tier2_bitwise={n_bw_ok}/{n_bw} {tags}{note}")

        if expect_guard_degrade:
            # The two properties rev3 OWNS on this input, asserted independently of Tier 1.
            v_split, v_base = got[3], base[3]
            deg_split = bool((v_split == 0).all().item())
            deg_base = bool((v_base == 0).all().item())
            nan_split = bool(torch.isnan(got[1]).any().item())
            inv_ok = tier2_ok and deg_split and deg_base
            print(f"[BOUNDARY_{'PASS' if inv_ok else 'WARN'}] {level} {name} {task} "
                  f"rev3-invariants: dual_path_bitwise={n_bw_ok}/{n_bw} "
                  f"guard_degrade_split(valid all-0)={deg_split} "
                  f"guard_degrade_current(valid all-0)={deg_base} s_out_is_nan={nan_split} "
                  f"(both paths) {tags}")
            if not inv_ok:
                print(f"[BOUNDARY_WARN] {level} {name} {task}: a rev3 invariant broke -- "
                      f"either the paths diverged (tier2={n_bw_ok}/{n_bw}) or the D-GUARD did "
                      f"not degrade on BOTH paths (split={deg_split} current={deg_base}); this "
                      f"would mean rev3 changed the behaviour instead of preserving it {tags}")
            ok &= inv_ok
    return ok


def test_r4_rev3_l1():
    """rev3 L1 extension: split-path functional coverage, Tier-1 + Tier-2 judged (blocking).

    Every shape in REV3_L1_CASES (15 as of Stage-3 round-3 item A) was CPU-pre-verified with
    golden_r4 to yield valid_ref all-1 (section 9.2.1) AND to satisfy use_split == True, so this
    layer really executes K1A/K1B/K2gA/K2gB. Coverage added on top of the legacy L1: k=1 and
    k==K_MAX on the split route (rank-0 and the D-TIE saturated-list fast path), both contract
    specializations, odd B with a live row guard in K1A, the theorem-P3 shrink shape, three value
    ranges, a non-aligned tail with S_TAIL=5, a (BV,K_MAX) pairing outside the two calibrated cost
    points, and -- added together with the measured _SPLIT_A_LAUNCH_US relaxation -- the two
    cann-bench geometries that relaxation newly admits (case 6 S=3/NBS=1/S_TAIL=2 and case 11
    S=2/NBS=1/S_TAIL=-1), which previously had NO runtime dual-path coverage.
    """
    print("=" * 80)
    print(f"[L1-REV3] split-V functional tests -- {len(REV3_L1_CASES)} cases x 2 tasks")
    print("=" * 80)
    all_ok = True
    for (name, B, V, kmax, bv, k, p, mu, nmp, scale, offset, tags) in REV3_L1_CASES:
        try:
            torch.manual_seed(0)
            logits = torch.randn(B, V, dtype=torch.float32) * scale + offset
            all_ok &= _rev3_dual_compare(name, B, V, kmax, bv, k, p, mu, nmp, logits, tags,
                                         level="l1", expect_split=True, blocking=True)
        except Exception as e:  # noqa: BLE001
            import traceback

            print(f"[PRECISION_FAIL] l1 {name}: {type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc()
            all_ok = False
    return all_ok


def test_r4_rev3_boundary():
    """rev3 Boundary extension (V-S4, non-blocking): whole-split -inf + sparse -inf control.

    Verifies the cross-split merge in K1B does not manufacture NaN out of the -inf padding
    (run_v's -inf initial value, and an entire split whose candidates are all -inf) and that
    the split path stays BIT-identical to the delivered path on these legal special values.

    ⚠️ MEASURED FINDING (V-S4, this Stage) -- section 9.3's先验 was HALF right, and the half
    that was wrong is worth stating precisely:
      * The design predicted the CROSS-SPLIT TOP-K MERGE is safe with -inf. **CONFIRMED**:
        m_out and run_v/tau stay finite, and every one of these cases is bit-identical
        between the two paths (tier2_bitwise = 5/5).
      * The design did NOT anticipate a NaN coming from the ONLINE-SOFTMAX RESCALING instead.
        When a LEADING WHOLE BLOCK is entirely -inf, m_run is still -inf and m_blk is -inf, so
        `m_old = max(m_run, m_blk) = -inf` and `tmp = m_run - m_old = -inf - (-inf) = NaN`,
        then `s_run *= exp(NaN)` poisons s for the rest of the row. q2 = exp(run_v-m)/NaN = NaN
        -> prefix = NaN -> tau_p freezes at run_v[0] (= m) and okp = 0 -> valid = 0.
      * **This is a PRE-EXISTING property of the delivered rev2 kernel, not a rev3 regression.**
        Proven three ways: (i) tier2_bitwise = 5/5 on every case; (ii) a CONTROL shape that
        never splits (B=64, V=16384 -> use_split=False, S=1) reproduces the identical NaN;
        (iii) calling the frozen r4_topk_stats DIRECTLY, bypassing the dispatch, is bit-identical
        to the wrapper. Root-cause probe: /tmp/opencode/r4_s2r3/vs4_rootcause.py.
      * **It degrades SAFELY**: valid == 0 on every row, which is the designed fallback signal,
        so the host never consumes a wrong tau/token. A sparse -inf control (no whole block)
        stays completely clean (valid all-1, tau error 0).
      * Fixing it is OUT OF SCOPE for rev3 and would violate two hard constraints at once:
        user constraint 3 (the three delivered kernels must stay byte-identical) and user
        constraint 2 (K1B's softmax scan must stay line-for-line identical, which is what makes
        m/s bitwise constructive). It is therefore reported, not patched.
    The [BOUNDARY_WARN] lines below are the honest Tier-1 verdict against golden; the paired
    [BOUNDARY_PASS] rev3-invariants lines are the verdict on what rev3 is responsible for.
    """
    print("=" * 80)
    print(f"[BOUNDARY-REV3] V-S4 split-path special values -- {len(REV3_BND_CASES)} cases x 2 tasks")
    print("=" * 80)
    for (name, B, V, kmax, bv, k, p, mu, nmp, n_ninf, tags) in REV3_BND_CASES:
        try:
            _u, S, NBS, _st = _r4_split_plan(B, V, kmax, bv)
            logits = _rev3_bnd_logits(B, V, bv, NBS, n_ninf)
            ninf = float(torch.isneginf(logits).float().mean().item())
            rowall = bool(torch.isneginf(logits).all(dim=-1).any().item())
            # n_ninf > 0 puts at least one WHOLE leading block at -inf -> the documented
            # pre-existing s_run NaN degrade fires; n_ninf == 0 is the sparse control.
            print(f"[BOUNDARY_INFO] {name}: S={S} NBS={NBS} whole_-inf_splits={n_ninf} "
                  f"-inf_frac={ninf:.3f} any_row_all_-inf={rowall} "
                  f"expect_pre_existing_degrade={n_ninf > 0} {tags}")
            _rev3_dual_compare(name, B, V, kmax, bv, k, p, mu, nmp, logits, tags,
                               level="boundary", expect_split=True, blocking=False,
                               expect_guard_degrade=(n_ninf > 0))
        except Exception as e:  # noqa: BLE001
            print(f"[BOUNDARY_WARN] boundary {name}: {type(e).__name__}: {str(e)[:160]} {tags}")


def test_r4_rev3_l2():
    """rev3 L2 extension (non-blocking): illegal input must STILL be rejected when the shape
    would take the split path. The dispatch runs AFTER the C3/C6/C7/dtype pre-checks
    (DESIGN.md 5.7), so the new path must not become a way around them; and the D-GUARD
    degrade path must fire identically on the split route.
    """
    print("=" * 80)
    print("[L2-REV3] split-path negative tests (non-blocking)")
    print("=" * 80)
    # (8, 16384, 256, 4096) -> use_split True (S=4); (4, 20793, ...) -> S=6, S_TAIL=5.
    sb, sv, sk, sbv = 8, 16384, 256, 4096
    ub, us_, unbs, _ust = _r4_split_plan(sb, sv, sk, sbv)
    print(f"[BOUNDARY_INFO] l2 split-path probe shape B={sb} V={sv} K_MAX={sk} BV={sbv} -> "
          f"use_split={ub} S={us_} NBS={unbs}")
    base = torch.randn(sb, sv, dtype=torch.float32) * 8.0
    ks = torch.full((sb,), 40, dtype=torch.int32)
    ps = torch.full((sb,), 0.95, dtype=torch.float32)
    mus = torch.full((sb,), 0.0, dtype=torch.float32)
    t_split = ["D-PATH-SPLIT", "D-EXC-DTYPE"]
    t_shape = ["D-PATH-SPLIT", "D-EXC-SHAPE"]
    # dtype rejection on a split-path shape (bf16 / fp16 are out of contract).
    _run_exception("rev3_exc_dtype_bf16_splitpath",
                   lambda: r4_fused_sampling(base.to(torch.bfloat16).npu(), ks.npu(), ps.npu(),
                                             mus.npu(), False, sk, sbv), t_split)
    _run_exception("rev3_exc_dtype_fp16_splitpath",
                   lambda: r4_fused_sampling(base.to(torch.float16).npu(), ks.npu(), ps.npu(),
                                             mus.npu(), False, sk, sbv), t_split)
    # C3 violation (tail 1 < K_MAX) on an otherwise split-eligible geometry.
    _run_exception("rev3_exc_tail_lt_KMAX_splitpath",
                   lambda: r4_fused_sampling(
                       torch.randn(8, 16385, dtype=torch.float32).npu(),
                       torch.full((8,), 40, dtype=torch.int32).npu(),
                       torch.full((8,), 0.95, dtype=torch.float32).npu(),
                       torch.full((8,), 0.0, dtype=torch.float32).npu(), False, sk, sbv),
                   t_shape)
    # C7 violation (V < K_MAX): cannot be split, must still raise (regression sentinel).
    _run_exception("rev3_exc_V_lt_KMAX",
                   lambda: r4_fused_sampling(
                       torch.randn(8, 128, dtype=torch.float32).npu(),
                       torch.full((8,), 40, dtype=torch.int32).npu(),
                       torch.full((8,), 0.95, dtype=torch.float32).npu(),
                       torch.full((8,), 0.0, dtype=torch.float32).npu(), False, sk, sbv),
                   ["D-EXC-SHAPE"])
    # D-GUARD degrade must fire on the SPLIT route too (valid all-0), both tasks.
    for nm, kk, pp, scale in (("rev3_guard_k_TOP_K_ALL_splitpath", 1 << 30, 0.95, 8.0),
                              ("rev3_guard_k_zero_splitpath", 0, 0.95, 8.0),
                              ("rev3_guard_k_gt_KMAX_splitpath", 257, 0.95, 8.0),
                              ("rev3_guard_flat_dist_splitpath", 40, 0.95, 1.0)):
        _run_guard(nm, sb, sv, sk, sbv, kk, pp, 0.0, False, scale, 0.0,
                   ["D-PATH-SPLIT", "D-PARAM-top_k"])
    # Same on the fused (Gumbel) task: valid must come back all-0 there too. _run_guard only
    # drives the mask task, so the fused variant goes through _run_gumbel_guard (FUSE_GUMBEL=True),
    # which is what actually exercises K1A/K1B/K2gA/K2gB under a fired guard.
    _run_gumbel_guard("rev3_guard_flat_dist_splitpath_fused", sb, sv, sk, sbv, 40, 0.95, 0.0,
                      False, 1.0, ["D-PATH-SPLIT", "D-PARAM-FUSE_GUMBEL", "D-VALRANGE-S"])
    _run_gumbel_guard("rev3_guard_k_gt_KMAX_splitpath_fused", sb, sv, sk, sbv, 257, 0.95, 0.0,
                      False, 8.0, ["D-PATH-SPLIT", "D-PARAM-FUSE_GUMBEL", "D-PARAM-top_k"])



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
    # [Stage-3 round-3 item A] PATH CHANGED: this is cann-bench case 6's shape. n_blk_full=2,
    # n_blk_total=3 -> S=3/NBS=1/S_TAIL=2, and with _SPLIT_A_LAUNCH_US at the measured 1.540 us the
    # guard's pred_saving 8.902 us clears 1.5*(1.540 + 3*0.974) = 6.693 us, so it now routes to the
    # SPLIT path (the shipped 8.0 us placeholder gave a 16.383 us threshold and kept it CURRENT).
    # Nothing is weakened: the Tier-1 golden comparison below is identical on either path, and the
    # Tier-2 dual-path bitwise gate for this exact shape is l1r3_split_cb6_B8_V10240 in
    # REV3_L1_CASES. D-PATH-SPLIT/D-SPLIT-* tags added so the routing is asserted, not incidental.
    ("l1_tailmid_V", 8, 10240, 256, 4096, 40, 0.95, 0.0, False, 8.0, 0.0,
     ["D-PATH-SPLIT", "D-SPLIT-NBS1", "D-SPLIT-TAILOWNER", "D-SHAPE-TAIL-MID"]),
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
    # [Stage-3 round-3 item A] PATH CHANGED: same shape as l1_tailmid_V (cann-bench case 6), so
    # S=3/NBS=1/S_TAIL=2 now clears the measured guard and routes to the SPLIT path. This is the
    # FUSE_GUMBEL=True variant, i.e. it exercises the full split chain K1A -> K1B -> K2gA -> K2gB
    # and judges tok_out bitwise against golden_gumbel_sample. Tier-2 dual-path bitwise coverage for
    # this shape lives in REV3_L1_CASES/l1r3_split_cb6_B8_V10240.
    ("l1b_tailmid_V", 8, 10240, 256, 4096, 40, 0.95, 0.0, False, 8.0, 0.0, "arange",
     ["D-PATH-SPLIT", "D-SPLIT-NBS1", "D-SPLIT-TAILOWNER", "D-PARAM-FUSE_GUMBEL",
      "D-SHAPE-TAIL-MID"]),
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
    # [Stage-3 round-3 item A] PATH CHANGED: n_blk_full=2, n_blk_total=3 -> S=3/NBS=1/S_TAIL=2,
    # and the guard's pred_saving 8.902 us now clears 1.5*(1.540 + 3*0.974) = 6.693 us, so this
    # config routes to the SPLIT path (it was CURRENT under the shipped 8.0 us placeholder, whose
    # threshold was 16.383 us). Its assertion is UNCHANGED and NOT weakened -- N>=20 bit-identical
    # launches + golden match + the tau invariant are required on either path -- and it now also
    # exercises (BV=2048, K_MAX=256) on the split route at S=3, a geometry the S=8 case
    # l1r3_split_bv2048_k256 did not reach. D-PATH-SPLIT/D-SPLIT-* tags added to say so.
    ("stab_B7_V4409_bv2048_gumbel", 7, 4409, 256, 2048, 40, 0.95, True,
     ["D-PATH-SPLIT", "D-SPLIT-NBS1", "D-SPLIT-TAILOWNER", "D-PARAM-BV", "D-PARAM-FUSE_GUMBEL"]),
    # 32B-aligned-tail control (TAIL=2048, 2048 % 32 == 0): must stay clean, guards against
    # over-fitting the rev2 fix to the non-aligned remainders {1, 25, 31, 5}.
    # [Stage-3 round-3 item A] PATH CHANGED for the same reason and with the same numbers as above
    # (n_blk_full=2 -> S=3/NBS=1/S_TAIL=2, pred_saving 8.902 > 6.693). The control role is
    # PRESERVED -- it still feeds TAIL=2048 through the tail-padding path and still demands 20
    # bit-identical launches -- but it now does so on the SPLIT route, so it is no longer a
    # CURRENT-path control. The CURRENT-path aligned/odd-B controls that remain are
    # stab_case5_B7_V4409_{gumbel,mask}, stab_B1_V4096_{gumbel,mask}, stab_B3_V4353_gumbel and
    # stab_B9_V4607_gumbel (6 of the 8 legacy configs; all are blocked STRUCTURALLY by
    # n_blk_total == 1 or n_blk_full < 2, so no guard value can ever move them).
    ("stab_B5_V10240_gumbel", 5, 10240, 256, 4096, 40, 0.95, True,
     ["D-PATH-SPLIT", "D-SPLIT-NBS1", "D-SPLIT-TAILOWNER", "D-SHAPE-TAIL-MID",
      "D-PARAM-FUSE_GUMBEL"]),
    # ---- rev3 SPLIT-PATH stability configs (V-S5; DESIGN.md section 9.2.4 + 16.8) ----
    # WHY: when rev3 landed, every legacy config above had n_blk_full in {0,1,2} and was vetoed by
    # D-SPLIT-GUARD, so the 8 legacy configs executed ZERO lines of the split path no matter how
    # many reps they ran (section 16.8: "现有 STAB 全部走现行路径，对 split 路径零覆盖"). The
    # shapes below all satisfy use_split == True, so this gate plus --level stress (which iterates
    # STAB_CASES) is what applies the UB poisoner and the N>=20 bit-identity requirement to
    # K1A / K1B / K2gA / K2gB.
    # [Stage-3 round-3 item A] The blanket claim "every config above is vetoed" is no longer true:
    # relaxing _SPLIT_A_LAUNCH_US to the measured 1.540 us moves 2 of the 8 legacy configs
    # (stab_B7_V4409_bv2048_gumbel, stab_B5_V10240_gumbel -- both n_blk_full == 2, i.e. vetoed by
    # the COST guard rather than structurally) onto the split path, so split-path STAB coverage is
    # now 6 configs, not 4. The 6 remaining legacy configs are still blocked structurally.
    # A2: S=10 (max), grid=20 = exactly one full wave, n_blk_full=37 >> 2, uneven [4x9, 2],
    #     S_TAIL=9, aligned tail 512. The heaviest split shape (cann-bench case7).
    ("stab_split_A2_B4_V152064_gumbel", 4, 152064, 256, 4096, 40, 0.95, True,
     ["D-PATH-SPLIT", "D-SPLIT-S-MAX", "D-SPLIT-UNEVEN", "D-SPLIT-TAILOWNER", "D-PARAM-FUSE_GUMBEL"]),
    ("stab_split_A2_B4_V152064_mask", 4, 152064, 256, 4096, 40, 0.95, False,
     ["D-PATH-SPLIT", "D-SPLIT-S-MAX", "D-SPLIT-UNEVEN", "D-SPLIT-TAILOWNER"]),
    # B5: large n_blk_full (37) + NON-aligned large tail (825 -> TAIL_PAD 832) + small S (2)
    #     + S_TAIL=1, so the tail-owning split also carries 18 full blocks (heaviest load).
    ("stab_split_B5_B16_V152377_gumbel", 16, 152377, 256, 4096, 40, 0.95, True,
     ["D-PATH-SPLIT", "D-SPLIT-S-MIN", "D-SPLIT-TAILOWNER", "D-SHAPE-PRIME", "D-PARAM-FUSE_GUMBEL"]),
    # F1: theorem-P3 shrink shape (naive S=4 -> per [2,2,1,0] with an EMPTY split; shrink gives
    #     S=3 -> [2,2,1]). Under UB poisoning this is the sentinel for E-S6: an empty split
    #     would leave a pv row unwritten and K1B would read torch.empty garbage, which shows up
    #     exactly like the rev2 defect A symptom (intermittent huge tau_out).
    ("stab_split_F1_B10_V20480_gumbel", 10, 20480, 256, 4096, 40, 0.95, True,
     ["D-PATH-SPLIT", "D-SPLIT-SHRINK", "D-SPLIT-UNEVEN", "D-PARAM-FUSE_GUMBEL"]),
    # ---- Stage-3 round-3 item A: stability configs for the two NEWLY ADMITTED contract cases ----
    # The measured guard routes cann-bench case 6 (8, 10240) and case 11 (8, 8192) onto the split
    # path. Neither shape had ANY repeat-execution coverage before, and the UB poisoner is the only
    # RUNTIME detector of a stale-UB slot being consumed as a real candidate (padCount == 0 is
    # proven at compile time only -- this is how the rev2 defect A was reproduced). Both tasks are
    # covered per shape because the fused task additionally exercises K2gA/K2gB while the mask task
    # does not. Shapes CPU-pre-verified under the section 9.2.1 recipe (preflight.json P2:
    # valid_ref all-1, no golden NaN).
    # cb6: S=3/NBS=1/S_TAIL=2, TAIL=2048 -> the tail-owning split (sid 2) carries 0 FULL blocks
    #      while splits 0/1 carry 1 each, and grid_A = b_num*S = 4*3 = 12 <= cube_core_num 20, so
    #      theorem P2's one-wave requirement still holds.
    ("stab_split_cb6_B8_V10240_gumbel", 8, 10240, 256, 4096, 40, 0.95, True,
     ["D-PATH-SPLIT", "D-SPLIT-NBS1", "D-SPLIT-TAILOWNER", "D-SHAPE-TAIL-MID",
      "D-PARAM-FUSE_GUMBEL"]),
    ("stab_split_cb6_B8_V10240_mask", 8, 10240, 256, 4096, 40, 0.95, False,
     ["D-PATH-SPLIT", "D-SPLIT-NBS1", "D-SPLIT-TAILOWNER", "D-SHAPE-TAIL-MID"]),
    # cb11: S=2/NBS=1/S_TAIL=-1, perfectly aligned, the SMALLEST V and the minimum splittable S of
    #       any split-path config -> the tightest geometry the relaxed guard admits.
    ("stab_split_cb11_B8_V8192_gumbel", 8, 8192, 256, 4096, 40, 0.95, True,
     ["D-PATH-SPLIT", "D-SPLIT-S-MIN", "D-SPLIT-NBS1", "D-SHAPE-ALIGNED", "D-PARAM-FUSE_GUMBEL"]),
    ("stab_split_cb11_B8_V8192_mask", 8, 8192, 256, 4096, 40, 0.95, False,
     ["D-PATH-SPLIT", "D-SPLIT-S-MIN", "D-SPLIT-NBS1", "D-SHAPE-ALIGNED"]),
]

# Adversary probes: (B, V, K_MAX, BV) -> sentinel swept over the at-risk tail indices.
ADV_CASES = [
    (2, 4409, 256, 4096),   # TAIL=313 -> 1 element at risk (313 - 312)
    (3, 4353, 256, 4096),   # TAIL=257 -> 7 elements at risk (257 - 256)
    (2, 4607, 256, 4096),   # TAIL=511 -> 7 elements at risk (511 - 504)
    (5, 261, 256, 4096),    # tail-only V<K_MAX*2, TAIL=261 -> 5 at risk (261 - 256)
    # ---- rev3 SPLIT-PATH tail adversaries (V-S6; DESIGN.md sections 9.2.4 / 16.8) ----
    # All four legacy shapes above have n_blk_full <= 1 -> vetoed by D-SPLIT-GUARD -> the
    # sentinel sweep never verified that the tail block is handled by the CORRECT split.
    # Under rev3 the tail belongs to the compile-time split S_TAIL, so a wrong S_TAIL (or a
    # truncated TAIL_PAD hop) would silently drop tail elements; these four shapes all take
    # the split path (B=4 -> b_num=2 -> S_cap=10, n_blk_full=5, S=6, NBS=1, S_TAIL=5) and
    # reproduce the L0-B TAIL spectrum {%32 = 1, 25, 31, 5}.
    (4, 20737, 256, 4096),  # split path, TAIL=257 (%32=1)  -> 7 elements at risk
    (4, 20793, 256, 4096),  # split path, TAIL=313 (%32=25) -> 1 element at risk (rev2 defect)
    (4, 20991, 256, 4096),  # split path, TAIL=511 (%32=31) -> 7 elements at risk
    (4, 20741, 256, 4096),  # split path, TAIL=261 (%32=5)  -> 5 elements at risk
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
    # rev3: record WHICH path this sweep actually exercised, so a future D-SPLIT-GUARD change
    # cannot silently turn a "split-path adversary" into a current-path one (section 16.8).
    _us, _S, _NBS, _ST = _r4_split_plan(B, V, kmax, bv)
    path = f"SPLIT(S={_S},NBS={_NBS},S_TAIL={_ST})" if _us else f"CURRENT(S={_S})"
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
              f"path={path} dropped={len(bad)} {tags}")
    if not ok:
        detail += f" bad_idx={bad}"
    return ok, detail


def _rev3_count_split_configs():
    """How many STAB / ADV configs actually take the split path (section 16.8 blind-spot gate).

    Returns (n_stab_split, n_adv_split). Both must be >= 1: if a future D-SPLIT-GUARD change
    made every repeat-execution and every tail-adversary config route to the delivered path,
    the gates would still print PASS while covering ZERO lines of K1A/K1B/K2gA/K2gB -- exactly
    the silent blindness rev3 is required to remove. Failing loudly here is the fix.
    """
    n_stab = sum(1 for (_n, B, V, kmax, bv, _k, _p, _f, _t) in STAB_CASES
                 if _r4_split_plan(B, V, kmax, bv)[0])
    n_adv = sum(1 for (B, V, kmax, bv) in ADV_CASES if _r4_split_plan(B, V, kmax, bv)[0])
    return n_stab, n_adv


def test_r4_stability(reps=STAB_REPS, adversary=True):
    """Repeat-execution gate (BLOCKING): N>=20 bit-identical launches + tail adversary."""
    print("=" * 80)
    print(f"[STABILITY] repeat-execution gate -- {len(STAB_CASES)} configs x {reps} reps"
          + (f" + {len(ADV_CASES)} tail-adversary sweeps" if adversary else ""))
    print("=" * 80)
    all_ok = True
    # rev3 section 16.8 blind-spot gate: this suite must actually REACH the split path.
    n_stab_split, n_adv_split = _rev3_count_split_configs()
    cube = detect_cube_core_num()
    print(f"[STABILITY] cube_core_num={cube} (probed) | split-path configs: "
          f"STAB={n_stab_split}/{len(STAB_CASES)} ADV={n_adv_split}/{len(ADV_CASES)}")
    for nm, B, V, kmax, bv, _k, _p, _f, _t in STAB_CASES:
        us, S, NBS, ST = _r4_split_plan(B, V, kmax, bv)
        if us:
            print(f"[STABILITY]   split-path STAB {nm}: S={S} NBS={NBS} S_TAIL={ST} "
                  f"grid_A={-(-B // VEC_NUM) * S}")
    cov_ok = n_stab_split >= 1 and (n_adv_split >= 1 if adversary else True)
    print(f"[PRECISION_{'PASS' if cov_ok else 'FAIL'}] stability split-path coverage: "
          f"STAB_split={n_stab_split} ADV_split={n_adv_split} (both must be >=1, "
          f"otherwise this gate covers zero lines of K1A/K1B/K2gA/K2gB -- section 16.8)")
    all_ok &= cov_ok
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
    # V-S5 (DESIGN.md sections 4.9 / 10.2): the poisoner MUST be applied to K1A/K1B/K2gA/K2gB.
    # padCount==0 is proven at COMPILE time only; poisoning UB with 1e30 and launching
    # immediately is the sole RUNTIME detector of a stale-UB slot being consumed as a real
    # candidate (this is how the rev2 defect A was reproduced). Split-path STAB configs are
    # what makes this reach the four new kernels -- assert they are present, do not assume.
    n_stab_split, _n_adv = _rev3_count_split_configs()
    split_names = [nm for (nm, B, V, kmax, bv, _k, _p, _f, _t) in STAB_CASES
                   if _r4_split_plan(B, V, kmax, bv)[0]]
    print(f"[STRESS] poison elems={_POISON_ELEMS} ({_POISON_ELEMS * 4} B of the "
          f"{REV3_UB_LIMIT} B UB region) | split-path configs poisoned: {n_stab_split} {split_names}")
    try:
        pois = _make_ub_poisoner()
    except Exception as e:  # noqa: BLE001
        print(f"[PRECISION_FAIL] stress: poisoner build failed {type(e).__name__}: {str(e)[:160]}")
        return False
    all_ok = n_stab_split >= 1
    if n_stab_split < 1:
        print("[PRECISION_FAIL] stress V-S5: NO split-path STAB config -> the UB poisoner would "
              "never reach K1A/K1B/K2gA/K2gB (defect A undetectable at runtime)")
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
    # ---- rev3 split-V dimensions (DESIGN.md section 9.2.4 tag table) ----
    # These are EXTRA dimensions: proto.yaml is unchanged, so the coverage gate's REQUIRED
    # set (category Fusion + proto dtype/attr derived) stays at 25 and must stay 25/0/0.
    # COVERAGE_NA stays empty because Fusion exempts nothing.
    # use_split == True (A1-A4, B1-B5, C1, D1, E1, F1 + split STAB/ADV). Stage-3 round-3 item A:
    # the measured _SPLIT_A_LAUNCH_US also routes cb6/cb11 (new REV3_L1 + STAB + static entries) and
    # the 2 cost-guard-vetoed legacy STAB configs here, so this is a FLOOR, not an exact count.
    # D-PATH-CURRENT stays at 4: no case lost current-path coverage (C2/C3 are blocked
    # structurally, and the 6 remaining legacy STAB configs cannot move at any guard value).
    "D-PATH-SPLIT": 12,
    "D-PATH-CURRENT": 4,         # use_split == False (C2, C3 + D1/E1 baseline side)
    "D-SPLIT-S-MAX": 3,          # S == S_cap and grid == CUBE_CORE_NUM (A2, D1, split STAB A2)
    "D-SPLIT-S-MIN": 4,          # S == 2 (A1, B5, C1, split STAB B5)
    "D-SPLIT-NBS1": 6,           # NBS == 1, each split owns exactly 1 block (A3, A4, B1-B4)
    "D-SPLIT-SHRINK": 2,         # section 5.8.2 step 4 back-substitution shrink (F1 + split STAB F1)
    "D-SPLIT-UNEVEN": 4,         # max(per) != min(per): unbalanced split load (A2, F1, + STAB)
    "D-SPLIT-TAILOWNER": 8,      # S_TAIL > 0: the tail block is not owned by split 0
    "D-SPLIT-DEGEN-S1": 1,       # white-box forced S=1 degenerate (E1)
    "D-BITWISE-DUALPATH": 13,    # Tier-2 dual-path bit-pattern comparison (section 9.2.2)
}
COVERAGE_NA = {}


# ========== main: --level dispatch + exit code ==========
def main():
    parser = argparse.ArgumentParser(description="R4 fused sampling layered tests")
    parser.add_argument("--level", default="l0",
                        choices=["l0", "l1", "l2", "boundary", "stability", "stress", "static", "all"])
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

    if args.level == "static":
        # rev3 compile-time gate only (no NPU execution): section 9.2.5 assertions 1-4.
        sys.exit(0 if test_r4_rev3_static() else 1)

    blocking_ok = True  # only L0/L1 gate the exit code
    if args.level in ("l0", "all"):
        # rev3 section 11.4 step 1: the STATIC gate runs FIRST and stops the run on failure,
        # so a pass_configs / frozen-kernel / UB / padCount regression never burns NPU time.
        blocking_ok &= test_r4_rev3_static()
        blocking_ok &= test_r4_fused_sampling_l0()  # V01-a L0 (regression: K1 + K2 w_out)
        blocking_ok &= test_r4_gumbel_l0a()  # V01-b L0-A (avalanche-hash int32 bitwise)
        blocking_ok &= test_r4_gumbel_l0b()  # V01-b L0-B (tok_out bitwise)
        # rev3 section 9.2.3: the 15-case split-V matrix, Tier-1 + Tier-2 judged.
        blocking_ok &= test_r4_rev3_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_r4_fused_sampling_l1()  # V01-a L1 (regression)
        blocking_ok &= test_r4_gumbel_l1()  # V01-b L1 functional (tok_out bitwise)
        blocking_ok &= test_r4_gumbel_distribution()  # V01-b distribution consistency (chi2/KL/cov/entropy)
        blocking_ok &= test_r4_rev3_l1()  # rev3 L1 (split-path functional, Tier-1 + Tier-2)
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
        test_r4_rev3_l2()  # rev3 negative on split-path shapes: non-blocking
    if args.level in ("boundary", "all"):
        test_r4_fused_sampling_boundary()  # V01-a special values: non-blocking
        test_r4_gumbel_boundary()  # V01-b special values: non-blocking
        test_r4_rev3_boundary()  # rev3 V-S4 whole-split -inf: non-blocking

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
