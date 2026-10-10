"""MHA layered test suite: L0 gate + L1/L2/Boundary (stubs, expanded by
tilelang-op-test-design scenario B after L0 passes) + perf sentinels (DESIGN §9.4).

Run:
    python examples/mha/test_mha.py --level l0      # precision convergence
    python examples/mha/test_mha.py --level all     # full suite + sentinels

Precision criteria (DESIGN §9.3, orchestrator-approved execution caliber):
- Blocking = mixed tolerance (§9.3.2: matched_ratio + max_abs_error dual gate)
  AND MERE < Threshold AND MARE_abs < 10*Threshold (§9.3.1 eco standard).
- MARE_rel is computed and printed as a reference value; if everything else
  passes and only MARE_rel exceeds the gate (near-zero golden elements), the
  case prints [MARE_REL_ESCALATION] for orchestrator arbitration (not a silent
  standard change, not a direct FAIL).
- Thresholds: float16 -> 2**-10 (MERE) / 2**-10*10 (MARE); bfloat16 -> 2**-7.
"""

import argparse
import os
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
def make_inputs(B, S, S_kv, N, D, dtype_str, seed):
    """CPU-generated deterministic inputs: randn * U[-1,1], then cast to dtype."""
    gen = torch.Generator().manual_seed(seed)
    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_str]

    def one(shape):
        base = torch.randn(shape, generator=gen)
        scale = torch.rand(shape, generator=gen) * 2.0 - 1.0
        return (base * scale).to(dt)

    return one((B, S, N, D)), one((B, S_kv, N, D)), one((B, S_kv, N, D))


class CaseTimeout(Exception):
    """Sentinel wall-clock timeout."""


def _alarm_handler(signum, frame):
    raise CaseTimeout("case timeout")


def run_case(level, name, shape, dtype_str, scaleValue, is_causal, seed, chunked_golden=False):
    """Run one blocking (L0/L1-style) precision case. Returns (ok, info_dict)."""
    B, S, S_kv, N, D = shape
    info = {"name": name, "shape": shape, "dtype": dtype_str}
    try:
        q, k, v = make_inputs(B, S, S_kv, N, D, dtype_str, seed)
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
        eco_ok = (mere == mere) and mere < thr and mare_abs < 10 * thr  # NaN -> False
        mare_rel_ok = (mare_rel == mare_rel) and mare_rel < 10 * thr
        ok = mixed_ok and eco_ok

        info.update(matched=ratio, max_abs=max_abs, mere=mere, mare_abs=mare_abs, mare_rel=mare_rel, kernel_s=kernel_s)
        tag = "PASS" if ok else "FAIL"
        print(
            f"[PRECISION_{tag}] {level} case={name} shape={shape} dtype={dtype_str} "
            f"causal={is_causal} scale={scaleValue} matched={ratio:.5f} "
            f"max_abs={max_abs:.3e} MERE={mere:.3e} MARE_abs={mare_abs:.3e} "
            f"MARE_rel={mare_rel:.3e} kernel_time={kernel_s:.2f}s"
        )
        if ok and not mare_rel_ok:
            print(
                f"[MARE_REL_ESCALATION] {level} case={name}: mixed+MERE+MARE_abs pass, "
                f"only MARE_rel={mare_rel:.3e} >= {10 * thr:.3e} (near-zero golden "
                f"elements); escalated to orchestrator arbitration per DESIGN §9.3.1"
            )
        return ok, info
    except Exception as exc:  # noqa: BLE001 - report and continue
        print(f"[PRECISION_FAIL] {level} case={name} shape={shape} dtype={dtype_str}: {type(exc).__name__}: {exc}")
        info["error"] = f"{type(exc).__name__}: {exc}"
        return False, info


# ========== L0: gate cases (DESIGN §9.2, 10 cases, regular divisible shapes) ==========
L0_CASES = [
    # (name, (B,S,S_kv,N,D), dtype, scaleValue, is_causal)
    ("l0_min_fp16", (1, 64, 64, 1, 64), "float16", -1.0, False),
    ("l0_min_fp16_causal", (1, 64, 64, 1, 64), "float16", -1.0, True),
    ("l0_min_bf16_d128", (1, 64, 64, 1, 128), "bfloat16", -1.0, False),
    ("l0_min_bf16_d128_causal", (1, 64, 64, 1, 128), "bfloat16", -1.0, True),
    ("l0_typical_fp16", (2, 128, 128, 8, 64), "float16", -1.0, False),
    ("l0_typical_fp16_causal", (2, 128, 128, 8, 64), "float16", -1.0, True),
    ("l0_typical_bf16", (2, 128, 128, 8, 64), "bfloat16", -1.0, False),
    ("l0_typical_bf16_causal", (2, 128, 128, 8, 64), "bfloat16", -1.0, True),
    ("l0_kvlong_fp16_causal", (1, 128, 256, 4, 128), "float16", -1.0, True),
    ("l0_d256_fp16_scale", (1, 64, 64, 2, 256), "float16", 0.0625, False),
]


def test_mha_l0():
    """L0 gate: 10/10 must pass (DESIGN §9.2 pass standard)."""
    ok = True
    n_pass = 0
    for idx, (name, shape, dtype_str, scale, causal) in enumerate(L0_CASES):
        case_ok, _info = run_case("l0", name, shape, dtype_str, scale, causal, seed=1000 + idx)
        n_pass += int(case_ok)
        ok &= case_ok
    print(f"[L0] summary: {n_pass}/{len(L0_CASES)} passed, ok={ok}")
    return ok


# ========== L1/L2/Boundary: stubs (filled by tilelang-op-test-design scenario B) ==========
def test_mha_l1():
    """L1 functional tests (irregular/tail/prime shapes) — stub."""
    print("[L1] not expanded yet — run tilelang-op-test-design (scenario B)")
    return True


def test_mha_l2():
    """L2 negative tests (illegal inputs must be rejected) — stub."""
    print("[L2] not expanded yet — run tilelang-op-test-design (scenario B)")


def test_mha_boundary():
    """Boundary special-value tests (position-sensitive per DESIGN §9.2) — stub."""
    print("[BOUNDARY] not expanded yet — run tilelang-op-test-design (scenario B)")


# ========== Perf sentinels (DESIGN §9.4, mandatory, blocking on timeout) ==========
SENTINELS = [
    # (name, (B,S,S_kv,N,D), dtype, scaleValue, is_causal, timeout_s)
    ("perf_prefill_worst", (8, 2048, 4096, 32, 128), "float16", -1.0, True, 600),
    ("perf_decode_worst", (64, 1, 4096, 32, 128), "float16", -1.0, True, 300),
    ("perf_d256_bf16", (2, 1024, 1024, 16, 256), "bfloat16", -1.0, False, 300),
    ("perf_max_grid", (128, 128, 128, 64, 64), "float16", -1.0, True, 600),
]


def run_sentinel(name, shape, dtype_str, scaleValue, is_causal, timeout_s, seed):
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_s)
    t0 = time.time()
    try:
        ok, info = run_case("perf", name, shape, dtype_str, scaleValue, is_causal, seed=seed, chunked_golden=True)
        elapsed = time.time() - t0
        print(f"[PERF_{'PASS' if ok else 'FAIL'}] {name} elapsed={elapsed:.1f}s (timeout={timeout_s}s)")
        return ok
    except CaseTimeout:
        print(f"[PERF_FAIL] {name} TIMEOUT after {timeout_s}s — feasibility gate failed (DESIGN §9.4)")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"[PERF_FAIL] {name} {type(exc).__name__}: {exc}")
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def test_mha_perf():
    ok = True
    for idx, (name, shape, dt, scale, causal, timeout_s) in enumerate(SENTINELS):
        ok &= run_sentinel(name, shape, dt, scale, causal, timeout_s, seed=9000 + idx)
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
