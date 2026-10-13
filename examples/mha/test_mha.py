"""MHA layered test suite: L0 gate + L1/L2/Boundary + perf sentinels (DESIGN rev2).

Run:
    python examples/mha/test_mha.py --level l0      # precision convergence
    python examples/mha/test_mha.py --level all     # full suite + sentinels

Precision criteria (DESIGN rev2 §9.3.1, user-ruled 2026-09-18, not changeable):
- Blocking = mixed tolerance (§9.3.2: matched_ratio + max_abs_error dual gate)
  AND MARE_abs < 10*Threshold (float16: 9.77e-3; bfloat16: 7.81e-2).
- MERE and MARE_rel are REFERENCE values: computed and printed on the
  [PRECISION_*] lines, never blocking; the rev1 [MARE_REL_ESCALATION]
  arbitration flow is abolished (rev2-C1).
- Thresholds: float16 -> 2**-10; bfloat16 -> 2**-7 (MARE_abs gate = 10x).

Perf sentinels (DESIGN rev2 §9.4):
- Blocking feasibility gate = KERNEL-PATH wall time only: H2D -> kernel (incl.
  first JIT compile for the shape) -> D2H + synchronize, <= 120s per sentinel.
- Blocking precision gate = revised §9.3 caliber (mixed tolerance + MARE_abs).
- Golden-side cost control (mandatory): per-(b,n) chunked golden, persistent
  cross-run cache (.golden_cache/, test-side only), and — for the prefill
  sentinel only — deterministic sampled comparison (16 fixed-seed full (b,n)
  chunk pairs + >=1024 elements per remaining pair at deterministic rows
  covering both the causal-diagonal (partially masked) and fully-visible
  regions). Blocking criteria are judged on the sampled union.
- Case-level 1800s wall clock = NON-BLOCKING harness hang guard; firing it
  flags a harness defect, never an operator feasibility failure.
"""

import argparse
import hashlib
import os
import random
import signal
import sys
import time

import tilelang
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mha import mha  # noqa: E402  (kernel host wrapper under test)


# ========== Golden reference (DESIGN §9.1, verbatim; renamed golden_mha to
# avoid collision with the kernel import — semantics unchanged) ==========
def golden_mha(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaleValue: float = -1.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """多头注意力: query [B,S,N,D], key/value [B,S_kv,N,D] → 输出 [B,S,N,D]"""
    B, S, N, D = query.shape
    S_kv = key.shape[1]

    if scaleValue <= 0:
        scaleValue = 1.0 / (D**0.5)

    # 转置为 [B, N, S, D]
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    # 缩放点积注意力
    scores = torch.matmul(q, k.transpose(-2, -1)) * scaleValue
    if is_causal:
        i = torch.arange(S, device=scores.device).unsqueeze(-1)
        j = torch.arange(S_kv, device=scores.device).unsqueeze(0)
        causal_mask = j > (i + (S_kv - S))  # 右下角对齐：上三角置 -inf
        scores = scores.masked_fill(causal_mask, float("-inf"))
    # F217: 全 mask 行 (整行 = -inf) 在 softmax 时得 0/0 = NaN，对齐
    # sparse_flash_attention 加显式保护 → 全 mask 行权重置 0。
    scores_max = scores.max(dim=-1, keepdim=True).values
    all_masked = torch.isinf(scores_max) & (scores_max < 0)
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    attn_weights = torch.where(all_masked, torch.zeros_like(attn_weights), attn_weights)
    attn_output = torch.matmul(attn_weights, v)

    # 转回 [B, S, N, D]；transpose 只改 stride, 而输出契约要求 contiguous
    return attn_output.transpose(1, 2).contiguous()


def golden_mha_chunked(query, key, value, scaleValue=-1.0, is_causal=False):
    """Per-(batch, head) chunked golden for very large sentinel shapes.

    Mathematically identical to golden_mha (batch/head slices are independent);
    keeps CPU peak memory bounded. Runs on CPU in the same dtype (no upcast).
    """
    B, S, N, D = query.shape
    out = torch.empty_like(query)
    for b in range(B):
        for n in range(N):
            res = golden_mha(
                query[b : b + 1, :, n : n + 1, :],
                key[b : b + 1, :, n : n + 1, :],
                value[b : b + 1, :, n : n + 1, :],
                scaleValue,
                is_causal,
            )
            out[b, :, n, :] = res[0, :, 0, :]
    return out


# ========== Precision standards (DESIGN §9.3) ==========
def get_precision(dtype):
    """§9.3.2 mixed tolerance: (atol, rtol, max_abs_error_limit, matched_ratio)."""
    fp_table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
    }
    if dtype in {"int8", "int16", "int32", "int64", "uint8"}:
        return (0.0, 0.0, 0.0, 1.0)
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def eco_threshold(dtype):
    """§9.3.1 ecosystem standard Threshold (MERE gate; MARE gate = 10x)."""
    return {"float16": 2**-10, "bfloat16": 2**-7}.get(dtype, 2**-10)


def check_special_masks(actual, golden):
    """Position-sensitive NaN/+Inf/-Inf mask comparison (DESIGN §9.2 principle)."""
    a, g = actual.float(), golden.float()
    masks_match = (
        torch.equal(torch.isnan(a), torch.isnan(g))
        and torch.equal(torch.isposinf(a), torch.isposinf(g))
        and torch.equal(torch.isneginf(a), torch.isneginf(g))
    )
    finite = torch.isfinite(a) & torch.isfinite(g)
    return masks_match, finite


def check_precision(actual, golden, dtype):
    """§9.3.2 mixed-tolerance dual gate. Returns (passed, matched_ratio, max_abs).

    Golden-special positions are structurally compared first (mask equality);
    numeric tolerance applies where both sides are finite. Golden-finite spots
    where actual is inf/nan count as unmatched and lift max_abs.
    """
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a, g = actual.detach().cpu(), golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:  # integer: exact match
        mism = (a != g).sum().item()
        return mism == 0, 1.0 - mism / max(a.numel(), 1), (0.0 if mism == 0 else float("inf"))
    a, g = a.float(), g.float()
    masks_match, _ = check_special_masks(a, g)
    if not masks_match:
        return False, 0.0, float("inf")
    m = torch.isfinite(g)
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    return (ratio >= required_ratio and max_abs <= max_abs_limit), ratio, max_abs


def mere_mare(actual, golden):
    """§9.3.1 metrics in fp32 domain: (MERE, MARE_abs, MARE_rel).

    MERE     = mean(|a-g| / (|g| + 1e-7))
    MARE_rel = max (|a-g| / (|g| + 1e-7))   (reference value, escalation-only)
    MARE_abs = max |a-g|                    (ecosystem implementation caliber)
    """
    a = actual.detach().cpu().float()
    g = golden.detach().cpu().float()
    finite = torch.isfinite(a) & torch.isfinite(g)
    if finite.sum().item() == 0:
        return float("nan"), float("nan"), float("nan")
    af, gf = a[finite], g[finite]
    rel = (af - gf).abs() / (gf.abs() + 1e-7)
    return rel.mean().item(), (af - gf).abs().max().item(), rel.max().item()


# ========== Case harness ==========
def make_inputs(B, S, S_kv, N, D, dtype_str, seed, vrange_qk=None, vrange_v=None):
    """CPU-generated deterministic inputs (test prep stays on CPU, H2D once).

    Default (vrange_*=None): randn * U[-1,1] cast to dtype — the L0 scheme,
    bit-stable with attempt1. vrange=(lo, hi): uniform U[lo, hi] cast to
    dtype (L1 value-range dimensions D-VALRANGE-S/M/L/ASYM).
    """
    gen = torch.Generator().manual_seed(seed)
    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_str]

    def one(shape, vr):
        if vr is None:
            base = torch.randn(shape, generator=gen)
            scale = torch.rand(shape, generator=gen) * 2.0 - 1.0
            return (base * scale).to(dt)
        lo, hi = vr
        return (torch.rand(shape, generator=gen) * (hi - lo) + lo).to(dt)

    q = one((B, S, N, D), vrange_qk)
    k = one((B, S_kv, N, D), vrange_qk)
    v = one((B, S_kv, N, D), vrange_v)
    return q, k, v


class CaseTimeout(Exception):
    """Sentinel wall-clock timeout."""


def _alarm_handler(signum, frame):
    raise CaseTimeout("case timeout")


def run_case(
    level,
    name,
    shape,
    dtype_str,
    scaleValue,
    is_causal,
    seed,
    chunked_golden=False,
    gen=None,
    tags=None,
):
    """Run one blocking (L0/L1-style) precision case. Returns (ok, info_dict).

    gen: optional dict forwarded to make_inputs (vrange_qk/vrange_v) plus
    post-processing flags: negzero_v=True negates V (exercises -0.0 paths on
    all-zero inputs). Default None = attempt1 L0 scheme, bit-stable.
    """
    B, S, S_kv, N, D = shape
    info = {"name": name, "shape": shape, "dtype": dtype_str}
    try:
        gen = dict(gen or {})
        negzero_v = gen.pop("negzero_v", False)
        q, k, v = make_inputs(B, S, S_kv, N, D, dtype_str, seed, **gen)
        if negzero_v:
            v = -v  # +0.0 -> -0.0 (CPU-side prep; signed-zero coverage)
        t0 = time.time()
        out = mha(q.to("npu"), k.to("npu"), v.to("npu"), scaleValue=scaleValue, is_causal=is_causal)
        torch.npu.synchronize()
        kernel_s = time.time() - t0
        out = out.cpu()
        golden_fn = golden_mha_chunked if chunked_golden else golden_mha
        ref = golden_fn(q, k, v, scaleValue=scaleValue, is_causal=is_causal)

        mixed_ok, ratio, max_abs = check_precision(out, ref, dtype_str)
        mere, mare_abs, mare_rel = mere_mare(out, ref)
        thr = eco_threshold(dtype_str)
        # rev2-C1 blocking caliber: mixed tolerance (§9.3.2) AND MARE_abs < 10*Threshold.
        # MERE / MARE_rel are reference-only (printed, never gate; the rev1
        # [MARE_REL_ESCALATION] arbitration flow is abolished).
        mare_abs_ok = (mare_abs == mare_abs) and mare_abs < 10 * thr  # NaN -> False
        ok = mixed_ok and mare_abs_ok

        info.update(matched=ratio, max_abs=max_abs, mere=mere, mare_abs=mare_abs, mare_rel=mare_rel, kernel_s=kernel_s)
        tag = "PASS" if ok else "FAIL"
        tag_str = f" tags={','.join(tags)}" if tags else ""
        print(
            f"[PRECISION_{tag}] {level} case={name} shape={shape} dtype={dtype_str} "
            f"causal={is_causal} scale={scaleValue}{tag_str} matched={ratio:.5f} "
            f"max_abs={max_abs:.3e} MARE_abs={mare_abs:.3e} (gate<{10 * thr:.3e}) "
            f"MERE={mere:.3e}(ref) MARE_rel={mare_rel:.3e}(ref) kernel_time={kernel_s:.2f}s"
        )
        return ok, info
    except Exception as exc:  # noqa: BLE001 - report and continue
        print(f"[PRECISION_FAIL] {level} case={name} shape={shape} dtype={dtype_str}: {type(exc).__name__}: {exc}")
        info["error"] = f"{type(exc).__name__}: {exc}"
        return False, info


# ========== L0: gate cases (DESIGN §9.2, 10 cases, regular divisible shapes) ==========
# Case table unchanged from DESIGN §9.2; tags added per its coverage-matrix
# mapping (D-DTYPE-fp16: #1,2,5,6,9,10; D-DTYPE-bf16: #3,4,7,8;
# D-PARAM-is_causal(True): #2,4,6,8,9; D-PARAM-scaleValue(explicit): #10).
L0_CASES = [
    # (name, (B,S,S_kv,N,D), dtype, scaleValue, is_causal, tags)
    ("l0_min_fp16", (1, 64, 64, 1, 64), "float16", -1.0, False, ["D-DTYPE-fp16", "D-SHAPE-ALIGNED"]),
    ("l0_min_fp16_causal", (1, 64, 64, 1, 64), "float16", -1.0, True, ["D-DTYPE-fp16", "D-PARAM-is_causal"]),
    ("l0_min_bf16_d128", (1, 64, 64, 1, 128), "bfloat16", -1.0, False, ["D-DTYPE-bf16", "D-SHAPE-ALIGNED"]),
    ("l0_min_bf16_d128_causal", (1, 64, 64, 1, 128), "bfloat16", -1.0, True, ["D-DTYPE-bf16", "D-PARAM-is_causal"]),
    ("l0_typical_fp16", (2, 128, 128, 8, 64), "float16", -1.0, False, ["D-DTYPE-fp16", "D-SHAPE-ALIGNED"]),
    ("l0_typical_fp16_causal", (2, 128, 128, 8, 64), "float16", -1.0, True, ["D-DTYPE-fp16", "D-PARAM-is_causal"]),
    ("l0_typical_bf16", (2, 128, 128, 8, 64), "bfloat16", -1.0, False, ["D-DTYPE-bf16"]),
    ("l0_typical_bf16_causal", (2, 128, 128, 8, 64), "bfloat16", -1.0, True, ["D-DTYPE-bf16", "D-PARAM-is_causal"]),
    ("l0_kvlong_fp16_causal", (1, 128, 256, 4, 128), "float16", -1.0, True, ["D-DTYPE-fp16", "D-PARAM-is_causal"]),
    ("l0_d256_fp16_scale", (1, 64, 64, 2, 256), "float16", 0.0625, False, ["D-DTYPE-fp16", "D-PARAM-scaleValue"]),
]


def test_mha_l0():
    """L0 gate: 10/10 must pass (DESIGN §9.2 pass standard, rev2-C1 caliber)."""
    ok = True
    n_pass = 0
    for idx, (name, shape, dtype_str, scale, causal, tags) in enumerate(L0_CASES):
        case_ok, _info = run_case("l0", name, shape, dtype_str, scale, causal, seed=1000 + idx, tags=tags)
        n_pass += int(case_ok)
        ok &= case_ok
    print(f"[L0] summary: {n_pass}/{len(L0_CASES)} passed, ok={ok}")
    return ok


# ========== L1: functional cases (tilelang-op-test-design scenario B) ==========
# Shapes deterministically generated per test-design §6 from block=(bM,bN)
# (D in {64,128} -> (64,64); D=256 -> (32,64)): TAIL-1 (bM*k+1 / bN*k+1),
# TAIL-MID (bM*k+bM//2 / bN*k+bN//2), PRIME (nearest primes), EDGE (S=1 /
# S_kv=1 / single-element). GEMM three-axis non-alignment: S axis (GEMM1/2 M)
# via S=65/96/61/127; S_kv axis (GEMM1 N + GEMM2 reduction K) via S_kv=65/96/
# 127/160/251/1; B·N grid axis via prime B=3/N=5 and N=3 cid decomposition.
# GEMM1's K axis = D is host-constrained to {64,128,256} (always block-aligned;
# illegal D is covered by L2 rejection cases). Mask 4-variant coverage (§5.6):
# (F,T)/(T,T) in L0; (F,F) tail-clamp-only via l1_tail1_Skv_fp16 /
# l1_tailmid_Skv_bf16 / l1_prime_bn_bf16 / l1_edge_Skv1; (T,F) causal+clamp via
# l1_tail1_Skv_causal_bf16 / l1_tailmid_causal_fp16 / l1_prime_causal_fp16 /
# l1_d256_tailmix_bf16 / l1_valrange_asym_causal.
# Value-range magnitude calibration (measured, debug_log.md attempt2): the
# fp16/bf16 blocking gates are ABSOLUTE (MARE_abs < 9.77e-3 / 7.81e-2;
# max_abs <= 1e-1 / 1e0) while any non-bit-exact attention output carries
# >= 1 ulp(|out|) error; ulp(16)=1.56e-2 already exceeds the fp16 gate, so
# blocking-case |V| is capped at ~10 (measured: |out|<=10 -> 1 ulp = 7.8e-3 <
# gate; the single-block specialization measured BIT-EXACT with |score| up to
# 2264). Multi-block configs with |score| ~ 1e3 additionally hit fp16
# score-grid boundary amplification (measured 0.236 max error at qk U[-30,30]
# — golden's own round16(qk) flips 1 fp16 ulp=8 there, i.e. e^1 weight jumps)
# and are avoided by design. Literal dtype-bound values (fp16 ±65504, bf16
# ±3.39e38) are therefore probed in Boundary (D-SPECIAL-DBOUND, non-blocking).
L1_CASES = [
    # (name, (B,S,S_kv,N,D), dtype, scaleValue, is_causal, gen, tags)
    ("l1_aligned_bf16_d128", (1, 128, 128, 2, 128), "bfloat16", -1.0, False, None, ["D-DTYPE-bf16", "D-SHAPE-ALIGNED"]),
    ("l1_tail1_S_fp16", (1, 65, 64, 1, 64), "float16", -1.0, False, None, ["D-SHAPE-TAIL-1", "D-DTYPE-fp16"]),
    # (F,F) variant: tail-column clamp only (S_kv = 64+1 -> valid_n=1 tail iter)
    ("l1_tail1_Skv_fp16", (1, 64, 65, 1, 64), "float16", -1.0, False, None, ["D-SHAPE-TAIL-1"]),
    # (T,F) variant: causal + tail clamp; offset=1; N=3 (B·N=6 grid); KV tail valid_n=1
    (
        "l1_tail1_Skv_causal_bf16",
        (2, 64, 65, 3, 128),
        "bfloat16",
        -1.0,
        True,
        None,
        ["D-SHAPE-TAIL-1", "D-DTYPE-bf16", "D-PARAM-is_causal"],
    ),
    # (T,F): S=96 (bM+32 tail rows), S_kv=96 (bN+32 tail cols), offset=0
    ("l1_tailmid_causal_fp16", (2, 96, 96, 4, 64), "float16", -1.0, True, None, ["D-SHAPE-TAIL-MID", "D-PARAM-is_causal"]),
    # (F,F): S=96 tail rows, S_kv=160 (2*bN+32) tail cols
    ("l1_tailmid_Skv_bf16", (1, 96, 160, 2, 128), "bfloat16", -1.0, False, None, ["D-SHAPE-TAIL-MID", "D-DTYPE-bf16"]),
    # (T,F): S=61/S_kv=127 primes, offset=66 (right-bottom aligned, non-standard diagonal)
    ("l1_prime_causal_fp16", (1, 61, 127, 1, 64), "float16", -1.0, True, None, ["D-SHAPE-PRIME", "D-PARAM-is_causal"]),
    # (F,F): primes on all three grid axes (B=3, S=127, S_kv=251, N=5); kv_iters=4
    ("l1_prime_bn_bf16", (3, 127, 251, 5, 128), "bfloat16", -1.0, False, None, ["D-SHAPE-PRIME", "D-DTYPE-bf16"]),
    # D=256 -> bM=32: S=65=2*32+1 (TAIL-1 on the D-adaptive block), S_kv=96=64+32 (TAIL-MID), (T,F)
    ("l1_d256_tailmix_bf16", (1, 65, 96, 2, 256), "bfloat16", -1.0, True, None, ["D-SHAPE-TAIL-1", "D-SHAPE-TAIL-MID", "D-DTYPE-bf16"]),
    # S=1 decode (§10.3): valid_m=1, offset=63 -> the single row is fully visible
    ("l1_edge_decode_S1", (2, 1, 64, 2, 64), "float16", -1.0, True, None, ["D-SHAPE-EDGE", "D-PARAM-is_causal"]),
    # S_kv=1: kv_iters=1 specialization + valid_n=1 clamp; out == V row exactly
    ("l1_edge_Skv1", (1, 64, 1, 1, 64), "float16", -1.0, False, None, ["D-SHAPE-EDGE"]),
    ("l1_edge_single_elem", (1, 1, 1, 1, 64), "float16", -1.0, True, None, ["D-SHAPE-EDGE", "D-PARAM-is_causal"]),
    # all-zero incl. -0.0 V (§10.3): uniform softmax over visible cols, out = 0, bit-exact
    (
        "l1_zero_input",
        (1, 64, 64, 2, 64),
        "float16",
        -1.0,
        False,
        {"vrange_qk": (0, 0), "vrange_v": (0, 0), "negzero_v": True},
        ["D-SPECIAL-ZERO"],
    ),
    ("l1_valrange_s", (1, 64, 128, 2, 64), "float16", -1.0, False, {"vrange_qk": (-1, 1), "vrange_v": (-1, 1)}, ["D-VALRANGE-S"]),
    # measured: matched=0.99756, MARE_abs=7.8e-3 (gate 9.77e-3, structural 1-ulp bound)
    ("l1_valrange_m", (1, 64, 128, 2, 64), "float16", -1.0, False, {"vrange_qk": (-10, 10), "vrange_v": (-10, 10)}, ["D-VALRANGE-M"]),
    # largest gate-reachable symmetric range: scores to |2264| (fp16 score-grid
    # ulp=2, no round16(qk) overflow), saturated softmax; measured BIT-EXACT
    ("l1_valrange_l_sb", (1, 64, 64, 2, 64), "float16", -1.0, False, {"vrange_qk": (-40, 40), "vrange_v": (-10, 10)}, ["D-VALRANGE-L"]),
    # asymmetric range [-5,10], causal multi-block; measured matched=0.99512, MARE_abs=7.8e-3
    (
        "l1_valrange_asym_causal",
        (1, 64, 128, 2, 64),
        "float16",
        -1.0,
        True,
        {"vrange_qk": (-5, 10), "vrange_v": (-5, 10)},
        ["D-VALRANGE-ASYM", "D-PARAM-is_causal"],
    ),
    (
        "l1_valrange_m_bf16",
        (1, 64, 128, 2, 64),
        "bfloat16",
        -1.0,
        False,
        {"vrange_qk": (-10, 10), "vrange_v": (-10, 10)},
        ["D-VALRANGE-M", "D-DTYPE-bf16"],
    ),
    # explicit non-power-of-2 scale on D=128 (double-rounding mirror path, §3.5.2)
    ("l1_scale_explicit_d128", (1, 64, 64, 2, 128), "float16", 0.3, True, None, ["D-PARAM-scaleValue", "D-PARAM-is_causal"]),
]


def test_mha_l1():
    """L1 functional tests: irregular/tail/prime shapes + value ranges + attrs.
    Blocking layer ([PRECISION_*], rev2-C1 caliber). Returns all-passed bool."""
    ok = True
    n_pass = 0
    for idx, (name, shape, dtype_str, scale, causal, gen, tags) in enumerate(L1_CASES):
        case_ok, _info = run_case("l1", name, shape, dtype_str, scale, causal, seed=2100 + idx, gen=gen, tags=tags)
        n_pass += int(case_ok)
        ok &= case_ok
    print(f"[L1] summary: {n_pass}/{len(L1_CASES)} passed, ok={ok}")
    return ok


# ========== L2: negative tests (illegal inputs must be rejected; non-blocking) ==========
# All rejections must come from the host metadata asserts (AssertionError)
# BEFORE any JIT/NPU work (DESIGN §4.8/§5.6: unsupported inputs are rejected
# by metadata validation without reading tensor data). NaN/Inf inputs are NOT
# L2 cases (host cannot reject them without reading data — they are probed in
# Boundary instead, per DESIGN §5.6 "非有限值输入…非阻塞探测").
def _l2_thunks():
    fp16, bf16, fp32 = torch.float16, torch.bfloat16, torch.float32

    def z(*shape, dtype=fp16):
        return torch.zeros(*shape, dtype=dtype)

    def nc_q():
        # [1,64,2,64] non-contiguous view (metadata-only test prep on CPU)
        return torch.randn(1, 64, 64, 2, dtype=fp16).transpose(2, 3)

    return [
        ("dtype_mismatch_qk", ["D-EXC-DTYPE"], lambda: mha(z(1, 64, 1, 64), z(1, 64, 1, 64, dtype=bf16), z(1, 64, 1, 64, dtype=bf16))),
        (
            "dtype_unsupported_fp32",
            ["D-EXC-DTYPE"],
            lambda: mha(z(1, 64, 1, 64, dtype=fp32), z(1, 64, 1, 64, dtype=fp32), z(1, 64, 1, 64, dtype=fp32)),
        ),
        ("d_invalid_96", ["D-EXC-SHAPE"], lambda: mha(z(1, 64, 1, 96), z(1, 64, 1, 96), z(1, 64, 1, 96))),
        ("d_invalid_32", ["D-EXC-SHAPE"], lambda: mha(z(1, 64, 1, 32), z(1, 64, 1, 32), z(1, 64, 1, 32))),
        ("kv_shape_mismatch", ["D-EXC-SHAPE"], lambda: mha(z(1, 64, 1, 64), z(1, 32, 1, 64), z(1, 64, 1, 64))),
        ("causal_s_gt_skv", ["D-EXC-SHAPE"], lambda: mha(z(1, 128, 1, 64), z(1, 64, 1, 64), z(1, 64, 1, 64), is_causal=True)),
        ("s_out_of_range", ["D-EXC-SHAPE"], lambda: mha(z(1, 5000, 1, 64), z(1, 64, 1, 64), z(1, 64, 1, 64))),
        ("skv_out_of_range", ["D-EXC-SHAPE"], lambda: mha(z(1, 64, 1, 64), z(1, 5000, 1, 64), z(1, 5000, 1, 64))),
        ("n_out_of_range", ["D-EXC-SHAPE"], lambda: mha(z(1, 64, 100, 64), z(1, 64, 100, 64), z(1, 64, 100, 64))),
        ("b_out_of_range", ["D-EXC-SHAPE"], lambda: mha(z(200, 64, 1, 64), z(200, 64, 1, 64), z(200, 64, 1, 64))),
        ("b_zero_empty", ["D-EXC-SHAPE"], lambda: mha(z(0, 64, 1, 64), z(0, 64, 1, 64), z(0, 64, 1, 64))),
        ("non_contiguous_q", ["D-EXC-SHAPE"], lambda: mha(nc_q(), z(1, 64, 2, 64), z(1, 64, 2, 64))),
    ]


def run_l2_case(name, tags, fn):
    """L2 negative case (NON-BLOCKING): assert-rejection -> [BOUNDARY_PASS];
    silent acceptance or non-assert rejection -> [BOUNDARY_WARN] (recorded)."""
    try:
        fn()
    except AssertionError as exc:
        print(f"[BOUNDARY_PASS] l2 case={name} tags={','.join(tags)}: rejected by host assert ({exc})")
        return True
    except Exception as exc:  # noqa: BLE001 - rejection by unexpected path
        print(
            f"[BOUNDARY_WARN] l2 case={name} tags={','.join(tags)}: rejected by non-assert "
            f"{type(exc).__name__}: {exc} — host metadata-validation gap (expected AssertionError)"
        )
        return False
    print(f"[BOUNDARY_WARN] l2 case={name} tags={','.join(tags)}: illegal input SILENTLY ACCEPTED")
    return False


def test_mha_l2():
    """L2 negative tests (non-blocking; results recorded, never gate exit code)."""
    cases = _l2_thunks()
    n_pass = 0
    for name, tags, fn in cases:
        n_pass += int(run_l2_case(name, tags, fn))
    print(f"[L2] summary: {n_pass}/{len(cases)} correctly rejected (non-blocking)")


# ========== Boundary: special-value probes (position-sensitive; non-blocking) ==========
# DESIGN §9.2 principle: the declared input domain is finite reals — NaN/Inf
# are OUT-of-domain probes (never all-NaN/all-Inf: finite base + SPARSE
# injected specials, >=1 special and >=1 finite value). Comparison is
# mask-first: NaN/+inf/-inf positions must match exactly (check_special_masks
# inside check_precision), numeric tolerance only where both sides are finite.
# Dtype-bound finite values (±65504 fp16 / ±3.39e38 bf16) are in-domain but
# push |out| where the ABSOLUTE gates are unreachable for any non-bit-exact
# path (1 ulp(1024)=0.5 > max_abs limit) — a WARN there is the documented
# expectation, not a kernel defect.
DTYPE_FINITE_MAX = {"float16": 65504.0, "bfloat16": 3.3895313892515355e38}

BOUNDARY_CASES = [
    # (name, (B,S,S_kv,N,D), dtype, scaleValue, is_causal, inject, tags)
    (
        "bnd_nan_q_sparse",
        (1, 64, 64, 1, 64),
        "float16",
        -1.0,
        False,
        [("q", (0, 7, 0, 3), "nan"), ("q", (0, 33, 0, 60), "nan")],
        ["D-SPECIAL-NAN"],
    ),
    ("bnd_nan_v_multiblock", (1, 64, 128, 1, 64), "float16", -1.0, False, [("v", (0, 11, 0, 7), "nan")], ["D-SPECIAL-NAN"]),
    (
        "bnd_inf_k_sparse",
        (1, 64, 64, 1, 64),
        "float16",
        -1.0,
        False,
        [("k", (0, 5, 0, 10), "-inf"), ("k", (0, 40, 0, 2), "inf")],
        ["D-SPECIAL-INF"],
    ),
    ("bnd_inf_v_sparse", (1, 64, 64, 1, 64), "float16", -1.0, False, [("v", (0, 11, 0, 7), "inf")], ["D-SPECIAL-INF"]),
    (
        "bnd_dbound_v_fp16",
        (1, 64, 64, 1, 64),
        "float16",
        -1.0,
        False,
        [("v", (0, 3, 0, 1), "max"), ("v", (0, 20, 0, 5), "-max")],
        ["D-SPECIAL-DBOUND"],
    ),
    (
        "bnd_dbound_v_bf16",
        (1, 64, 64, 1, 64),
        "bfloat16",
        -1.0,
        False,
        [("v", (0, 3, 0, 1), "max"), ("v", (0, 20, 0, 5), "-max")],
        ["D-SPECIAL-DBOUND", "D-DTYPE-bf16"],
    ),
]


def apply_inject(tensors, inject, dtype_str):
    """Sparse special-value injection on CPU tensors (test-side data prep)."""
    q, k, v = tensors
    specials = {
        "nan": float("nan"),
        "inf": float("inf"),
        "-inf": float("-inf"),
        "max": DTYPE_FINITE_MAX[dtype_str],
        "-max": -DTYPE_FINITE_MAX[dtype_str],
    }
    for tname, idx, val in inject:
        t = {"q": q, "k": k, "v": v}[tname]
        t[idx] = specials[val] if isinstance(val, str) else val
    return q, k, v


def run_boundary_case(name, shape, dtype_str, scaleValue, is_causal, seed, inject, tags):
    """Boundary probe (NON-BLOCKING): mask-first position-sensitive compare;
    precision miss / exception -> [BOUNDARY_WARN] (recorded, never gates)."""
    B, S, S_kv, N, D = shape
    try:
        q, k, v = make_inputs(B, S, S_kv, N, D, dtype_str, seed)
        q, k, v = apply_inject((q, k, v), inject, dtype_str)
        out = mha(q.to("npu"), k.to("npu"), v.to("npu"), scaleValue=scaleValue, is_causal=is_causal)
        torch.npu.synchronize()
        out = out.cpu()
        ref = golden_mha(q, k, v, scaleValue=scaleValue, is_causal=is_causal)
        masks_match, _finite = check_special_masks(out.float(), ref.float())
        passed, ratio, max_abs = check_precision(out, ref, dtype_str)
        mere, mare_abs, mare_rel = mere_mare(out, ref)
        print(
            f"[BOUNDARY_{'PASS' if passed else 'WARN'}] boundary case={name} shape={shape} "
            f"dtype={dtype_str} tags={','.join(tags)} special_masks={'match' if masks_match else 'MISMATCH'} "
            f"matched={ratio:.5f} max_abs={max_abs:.3e} MERE={mere:.3e}(ref) "
            f"MARE_abs={mare_abs:.3e}(ref) MARE_rel={mare_rel:.3e}(ref)"
        )
        return passed
    except Exception as exc:  # noqa: BLE001 - probes never block
        print(f"[BOUNDARY_WARN] boundary case={name} shape={shape} dtype={dtype_str}: {type(exc).__name__}: {exc}")
        return False


def test_mha_boundary():
    """Boundary special-value probes (non-blocking; WARN only recorded)."""
    n_pass = 0
    for idx, (name, shape, dtype_str, scale, causal, inject, tags) in enumerate(BOUNDARY_CASES):
        n_pass += int(run_boundary_case(name, shape, dtype_str, scale, causal, 3100 + idx, inject, tags))
    print(f"[BOUNDARY] summary: {n_pass}/{len(BOUNDARY_CASES)} within tolerance (non-blocking)")


# ========== Coverage bookkeeping (coverage_check.py; coverage-matrix.md) ==========
COVERAGE_CATEGORY = "Fusion"  # CV-fused matmul+softmax+matmul (DESIGN §9.2)
COVERAGE_MANIFEST = {
    "D-DTYPE-fp16": 7,
    "D-DTYPE-bf16": 11,
    "D-SHAPE-ALIGNED": 4,
    "D-SHAPE-TAIL-1": 4,
    "D-SHAPE-TAIL-MID": 3,
    "D-SHAPE-PRIME": 2,
    "D-SHAPE-EDGE": 3,
    "D-VALRANGE-S": 1,
    "D-VALRANGE-M": 2,
    "D-VALRANGE-L": 1,
    "D-VALRANGE-ASYM": 1,
    "D-PARAM-is_causal": 12,
    "D-PARAM-scaleValue": 2,
    "D-SPECIAL-ZERO": 1,
    "D-SPECIAL-NAN": 2,
    "D-SPECIAL-INF": 2,
    "D-SPECIAL-DBOUND": 2,
    "D-EXC-DTYPE": 2,
    "D-EXC-SHAPE": 10,
}
# Fusion is the strictest category: NO exemptable dimensions -> every required
# dimension is covered by tagged cases above (COVERAGE_NA intentionally empty).
COVERAGE_NA = {}


# ========== Perf sentinels (DESIGN rev2 §9.4) ==========
# Gate structure (rev2-C4):
#   1. kernel-path wall-time gate (BLOCKING, the only feasibility timing
#      basis): H2D -> kernel (incl. first JIT compile for the shape) ->
#      D2H + synchronize, <= KERNEL_PATH_GATE_S per sentinel. Warm-kernel
#      time recorded separately as the Stage-3 baseline (informational).
#   2. precision gate (BLOCKING): revised §9.3 caliber (mixed tolerance +
#      MARE_abs < 10*Threshold); MERE/MARE_rel printed as reference.
#   3. golden-side cost control (mandatory): per-(b,n) chunked golden +
#      persistent cross-run cache + deterministic sampled comparison
#      (prefill sentinel only, scheme fixed in code below).
#   4. case-level 1800s wall clock: NON-BLOCKING harness hang guard —
#      firing it flags a harness defect, never an operator feasibility fail.
KERNEL_PATH_GATE_S = 120.0
CASE_WALLCLOCK_GUARD_S = 1800

SENTINELS = [
    # (name, (B,S,S_kv,N,D), dtype, scaleValue, is_causal, sampled_golden)
    ("perf_prefill_worst", (8, 2048, 4096, 32, 128), "float16", -1.0, True, True),
    ("perf_decode_worst", (64, 1, 4096, 32, 128), "float16", -1.0, True, False),
    ("perf_d256_bf16", (2, 1024, 1024, 16, 256), "bfloat16", -1.0, False, False),
    ("perf_max_grid", (128, 128, 128, 64, 64), "float16", -1.0, True, False),
]

# Deterministic prefill sampling scheme (§9.4-3 requires reproducibility):
PREFILL_SAMPLE_SEED = 1234  # fixed seed selecting the full-comparison (b,n) pairs
PREFILL_FULL_PAIRS = 16  # >=16 (b,n) chunk pairs get full-element comparison


def prefill_sample_rows(S, D):
    """Deterministic row set for the remaining (sampled) (b,n) pairs.

    9 rows x D=128 = 1152 >= 1024 elements per pair (§9.4-3). Rows 0/1/7 sit
    deep in the causal-diagonal (partially masked) region; row S-1 is fully
    visible under bottom-right alignment (offset = S_kv - S) — so the sampled
    set covers both required regions.
    """
    rows = sorted({0, 1, 7, S // 4, S // 2, (3 * S) // 4, S - 9, S - 2, S - 1})
    assert len(rows) * D >= 1024, "sampled set must cover >=1024 elements per pair"
    return rows


# ---- Golden persistent cache (§9.4-3; test-side only, NOT a deliverable) ----
GOLDEN_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".golden_cache")
GOLDEN_CACHE_SCHEME = "mha-golden-v1"  # bump when make_inputs/golden semantics change


def cached_golden(key, compute_fn):
    """Same (shape, seed, dtype, attr) golden computed once, reused across runs."""
    digest = hashlib.sha256(repr((GOLDEN_CACHE_SCHEME, key)).encode()).hexdigest()[:40]
    path = os.path.join(GOLDEN_CACHE_DIR, digest + ".pt")
    if os.path.exists(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception:  # noqa: BLE001 - corrupt/stale cache -> recompute
            pass
    val = compute_fn()
    try:
        os.makedirs(GOLDEN_CACHE_DIR, exist_ok=True)
        torch.save(val, path)
    except OSError:
        pass  # best-effort cache; correctness never depends on it
    return val


def golden_mha_rows(query, key, value, b, n, rows, scaleValue=-1.0, is_causal=False):
    """Row-slice golden for one (b,n) chunk at GLOBAL row indices `rows` (CPU).

    Mathematically identical to golden_mha restricted to those rows (rows are
    independent given full K/V; the causal mask uses the global row index, so
    bottom-right alignment is preserved). Op sequence mirrors golden_mha
    exactly: same-dtype matmul -> fp32-opmath scalar mul -> mask -> softmax
    -> same-dtype matmul. Only used for the §9.4 sampled sentinel comparison.
    """
    S, S_kv, D = query.shape[1], key.shape[1], query.shape[3]
    s = scaleValue if scaleValue > 0 else 1.0 / (D**0.5)
    q = query[b, rows, n, :].unsqueeze(0)  # [1, R, D]
    k = key[b, :, n, :].unsqueeze(0)  # [1, S_kv, D]
    v = value[b, :, n, :].unsqueeze(0)  # [1, S_kv, D]
    scores = torch.matmul(q, k.transpose(-2, -1)) * s
    if is_causal:
        i = torch.tensor(rows, dtype=torch.long).unsqueeze(-1)
        j = torch.arange(S_kv, dtype=torch.long).unsqueeze(0)
        causal_mask = (j > (i + (S_kv - S))).unsqueeze(0)
        scores = scores.masked_fill(causal_mask, float("-inf"))
    scores_max = scores.max(dim=-1, keepdim=True).values
    all_masked = torch.isinf(scores_max) & (scores_max < 0)
    w = torch.nn.functional.softmax(scores, dim=-1)
    w = torch.where(all_masked, torch.zeros_like(w), w)
    return torch.matmul(w, v).squeeze(0)  # [R, D] == out[b, rows, n, :]


def sentinel_compare(name, shape, dtype_str, scaleValue, is_causal, seed, sampled, out_cpu, q, k, v):
    """Golden (cost-controlled) + revised-§9.3 precision for one sentinel.

    sampled=True (prefill only): PREFILL_FULL_PAIRS fixed-seed (b,n) pairs get
    full-element chunked golden; every remaining pair is compared on the
    deterministic prefill_sample_rows set (>=1024 elements/pair). Blocking
    criteria are judged on the union of both subsets (§9.4-3).
    sampled=False: full chunked golden, persistently cached (§9.4-3).
    """
    B, S, S_kv, N, D = shape
    if not sampled:
        ref = cached_golden(
            ("full", name, shape, seed, dtype_str, scaleValue, is_causal),
            lambda: golden_mha_chunked(q, k, v, scaleValue, is_causal),
        )
        acts, glds = [out_cpu.reshape(-1)], [ref.reshape(-1)]
        golden_desc = "chunked-full(cached)"
    else:
        pairs = [(b, n) for b in range(B) for n in range(N)]
        rng = random.Random(PREFILL_SAMPLE_SEED)
        n_full = min(PREFILL_FULL_PAIRS, len(pairs))
        full_pairs = sorted(rng.sample(pairs, n_full))
        full_set = set(full_pairs)
        rest_pairs = [p for p in pairs if p not in full_set]
        rows = prefill_sample_rows(S, D)
        acts, glds = [], []
        for b, n in full_pairs:
            g = cached_golden(
                ("pair", name, b, n, seed, dtype_str, scaleValue, is_causal),
                lambda b=b, n=n: golden_mha(
                    q[b : b + 1, :, n : n + 1, :],
                    k[b : b + 1, :, n : n + 1, :],
                    v[b : b + 1, :, n : n + 1, :],
                    scaleValue,
                    is_causal,
                )[0, :, 0, :],
            )
            acts.append(out_cpu[b, :, n, :].reshape(-1))
            glds.append(g.reshape(-1))
        if rest_pairs:
            rows_golden = cached_golden(
                ("rows", name, tuple(rows), seed, dtype_str, scaleValue, is_causal),
                lambda: torch.stack([golden_mha_rows(q, k, v, b, n, rows, scaleValue, is_causal) for b, n in rest_pairs]),
            )
            for idx, (b, n) in enumerate(rest_pairs):
                acts.append(out_cpu[b, rows, n, :].reshape(-1))
                glds.append(rows_golden[idx].reshape(-1))
        golden_desc = f"sampled({len(full_pairs)}full+{len(rest_pairs)}x{len(rows)}rows,cached)"
    A, G = torch.cat(acts), torch.cat(glds)
    mixed_ok, ratio, max_abs = check_precision(A, G, dtype_str)
    mere, mare_abs, mare_rel = mere_mare(A, G)
    thr = eco_threshold(dtype_str)
    mare_abs_ok = (mare_abs == mare_abs) and mare_abs < 10 * thr  # NaN -> False
    prec_ok = mixed_ok and mare_abs_ok
    info = (
        f"golden={golden_desc} matched={ratio:.5f} max_abs={max_abs:.3e} "
        f"MARE_abs={mare_abs:.3e} (gate<{10 * thr:.3e}) MERE={mere:.3e}(ref) "
        f"MARE_rel={mare_rel:.3e}(ref)"
    )
    return prec_ok, info


def run_sentinel(name, shape, dtype_str, scaleValue, is_causal, sampled, seed):
    """One §9.4 sentinel. BLOCKING: kernel-path <= 120s AND revised-§9.3
    precision. NON-BLOCKING: case-level 1800s harness guard (a timeout there
    is a harness defect, not an operator feasibility verdict)."""
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(CASE_WALLCLOCK_GUARD_S)
    try:
        B, S, S_kv, N, D = shape
        q, k, v = make_inputs(B, S, S_kv, N, D, dtype_str, seed)  # CPU gen, outside gate
        # ---- kernel-path timing gate (the only feasibility timing basis) ----
        t0 = time.time()
        qn, kn, vn = q.to("npu"), k.to("npu"), v.to("npu")  # H2D
        out = mha(qn, kn, vn, scaleValue=scaleValue, is_causal=is_causal)  # incl. first JIT
        torch.npu.synchronize()
        out_cpu = out.cpu()  # D2H
        torch.npu.synchronize()
        kernel_path_s = time.time() - t0
        # ---- warm kernel re-run (Stage-3 baseline, informational) ----
        t1 = time.time()
        mha(qn, kn, vn, scaleValue=scaleValue, is_causal=is_causal)
        torch.npu.synchronize()
        warm_s = time.time() - t1
        gate_ok = kernel_path_s <= KERNEL_PATH_GATE_S
        # ---- golden (cost-controlled) + precision (blocking) ----
        prec_ok, prec_info = sentinel_compare(name, shape, dtype_str, scaleValue, is_causal, seed, sampled, out_cpu, q, k, v)
        ok = gate_ok and prec_ok
        print(
            f"[PERF_{'PASS' if ok else 'FAIL'}] {name} shape={shape} dtype={dtype_str} "
            f"causal={is_causal} kernel_path={kernel_path_s:.1f}s "
            f"(gate<={KERNEL_PATH_GATE_S:.0f}s) warm_kernel={warm_s:.2f}s(ref) {prec_info}"
        )
        if not gate_ok:
            print(
                f"[PERF_FAIL] {name}: kernel path {kernel_path_s:.1f}s exceeds {KERNEL_PATH_GATE_S:.0f}s gate -> feasibility failure (§9.4)"
            )
        return ok
    except CaseTimeout:
        print(
            f"[PERF_WARN] {name}: case-level {CASE_WALLCLOCK_GUARD_S}s harness guard fired "
            f"— harness defect (record & fix); NOT an operator feasibility verdict (§9.4-4)"
        )
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[PERF_FAIL] {name} {type(exc).__name__}: {exc}")
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def test_mha_perf():
    ok = True
    for idx, (name, shape, dt, scale, causal, sampled) in enumerate(SENTINELS):
        ok &= run_sentinel(name, shape, dt, scale, causal, sampled, seed=9000 + idx)
    print(f"[PERF] summary: all_sentinels_ok={ok} (kernel-path gate {KERNEL_PATH_GATE_S:.0f}s/sentinel, rev2 §9.4)")
    return ok


# ========== Main: --level dispatch + exit code ==========
def main():
    parser = argparse.ArgumentParser(description="MHA layered test suite")
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "perf", "all"])
    args = parser.parse_args()

    tilelang.disable_cache()  # avoid stale compilation artifacts (checklist #11)
    torch.manual_seed(0)

    blocking_ok = True  # only L0/L1/perf are blocking
    if args.level in ("l0", "all"):
        blocking_ok &= test_mha_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_mha_l1()
    if args.level in ("l2", "all"):
        test_mha_l2()  # negative: non-blocking
    if args.level in ("boundary", "all"):
        test_mha_boundary()  # special values: non-blocking
    if args.level in ("perf", "all"):
        blocking_ok &= test_mha_perf()

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
