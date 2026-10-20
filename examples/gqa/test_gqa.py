"""GQA (Grouped Query Attention) forward — layered accuracy test suite.

Exercises the public BSND API of examples/gqa/gqa.py:

    gqa(query=Q, key=K, value=V, scaleValue=..., is_causal=...)

with query [B, S, N_q, D] and key/value [B, S_kv, N_kv, D] in float16 /
bfloat16, returning [B, S, N_q, D].

The golden and the case matrix reuse the semantics of the locally verified
eval-regression suites (see examples/gqa/eval_fix_verify/ and
examples/gqa/debug_log.md):
  - f6_accuracy_regression.py (132/132 PASS): input-dtype golden running on
    NPU, seeds torch.manual_seed(42 + B + S + S_kv), inputs uniform in
    [-1, 1], mixed tolerance (fp16 atol=1e-3 / bf16 atol=2e-2, rtol=1e-2,
    zero tolerance violations) plus a max_abs_error cap, and causal
    fully-masked rows (S > S_kv) required to output exactly 0 (golden F217).
  - g2_special_values.py: fp64 CPU oracle golden for special values with
    structural NaN/Inf comparison (position / sign), fp16 full-range
    inputs, and NaN/Inf propagation semantics.

Test levels (--level, default l0):
  l0        representative aligned shapes (cases.csv base shapes) — both
            dtypes, both dispatch paths (prefill S > 4 / decode 1 <= S <= 4),
            causal / non-causal, D = 128 / 256, S != S_kv, explicit
            scaleValue. Blocking.
  l1        irregular shapes (S / S_kv non-128-aligned tails) and degenerate
            group sizes (MQA N_kv=1, MHA N_kv=N_q). Blocking.
  boundary  special values — large magnitudes, fp16 full range, sparse
            NaN/Inf propagation, causal fully-masked rows. Non-blocking
            ([BOUNDARY_WARN] does not affect the exit code).
  all       all of the above.

Only L0/L1 failures affect the exit code; Boundary failures are recorded
as [BOUNDARY_WARN] and do not block. CI compatibility: the final line is
"Test Passed!" when the blocking levels pass.

Usage (from the repo root — same invocation as eval_fix_verify/f6_accuracy_regression.py):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python examples/gqa/test_gqa.py --level all

NOTE: the repo root must NOT be on PYTHONPATH here — the in-repo tilelang
package is not built (no TVM) and would shadow the installed site-packages
tilelang used by the verified eval-regression runs. When the in-repo
tilelang is built, `source set_env.sh && python examples/gqa/test_gqa.py`
works as well.
"""

import argparse
import os
import sys
import traceback

import torch
import torch_npu  # noqa: F401  (registers the npu backend)

import tilelang

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gqa import gqa  # noqa: E402


# ===========================================================================
# Golden function — faithful replica of the cann-bench eval golden
# (.agents/skills/cann-bench/tasks/level4/gqa/golden.py), the reference the
# eval harness compares against. Runs in the input tensors' dtype/device:
# pass fp16/bf16 NPU tensors for the regular path (f6 semantics), or
# .double().cpu() tensors for the special-value fp64-oracle path (G2).
# ===========================================================================


def ref_gqa(query, key, value, scaleValue=-1.0, is_causal=False):
    """PyTorch reference for GQA forward (grouped query attention).

    Semantics (must match the kernel contract):
      - GQA group folding: query head n_kv * G + g shares KV head n_kv
        (G = N_q // N_kv). K/V are NOT materialized to N_q heads; the group
        dim is folded into the matmul M dim (numerically equivalent, much
        lower peak memory).
      - scaleValue <= 0 -> 1/sqrt(D).
      - is_causal: right-bottom-aligned causal mask — score (i, j) with
        j > i + (S_kv - S) is set to -inf before the softmax.
      - Fully-masked rows (causal, S > S_kv — golden F217): the softmax of
        an all--inf row is NaN; it is replaced by 0 so fully-masked rows
        output exactly 0.
      - NaN/Inf inputs propagate unsanitized (matches the kernel's
        fp64-semantics verification in eval_fix_verify/g2_special_values.py).

    Args:
        query: [B, S, N_q, D]
        key:   [B, S_kv, N_kv, D]
        value: [B, S_kv, N_kv, D]

    Returns:
        output: [B, S, N_q, D] (contiguous)
    """
    B, S, N_q, D = query.shape
    S_kv = key.shape[1]
    N_kv = key.shape[2]

    if scaleValue <= 0:
        scaleValue = 1.0 / (D**0.5)

    G = N_q // N_kv
    q = query.reshape(B, S, N_kv, G, D).permute(0, 2, 3, 1, 4).reshape(B, N_kv, G * S, D)
    k = key.permute(0, 2, 1, 3)  # [B, N_kv, S_kv, D]
    v = value.permute(0, 2, 1, 3)  # [B, N_kv, S_kv, D]

    scores = torch.matmul(q, k.transpose(-2, -1)) * scaleValue
    scores = scores.reshape(B, N_kv, G, S, S_kv)
    if is_causal:
        i = torch.arange(S, device=scores.device).unsqueeze(-1)
        j = torch.arange(S_kv, device=scores.device).unsqueeze(0)
        causal_mask = j > (i + (S_kv - S))  # right-bottom aligned
        scores = scores.masked_fill(causal_mask, float("-inf"))
    # F217: fully-masked rows (all -inf) softmax to 0/0 = NaN -> output 0.
    scores_max = scores.max(dim=-1, keepdim=True).values
    all_masked = torch.isinf(scores_max) & (scores_max < 0)
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    attn_weights = torch.where(all_masked, torch.zeros_like(attn_weights), attn_weights)
    attn_weights = attn_weights.reshape(B, N_kv, G * S, S_kv)
    attn_output = torch.matmul(attn_weights, v)  # [B, N_kv, G*S, D]

    return attn_output.reshape(B, N_kv, G, S, D).permute(0, 3, 1, 2, 4).reshape(B, S, N_q, D).contiguous()


# ===========================================================================
# Comparison helpers
# ===========================================================================


def _compare_regular(out, ref, dtype, causal, S, S_kv):
    """f6-verified comparison: mixed tolerance + cap + structural checks.

    Returns (max_diff, problems); an empty problems list means the case
    passed. Checks (all must hold, zero tolerance violations allowed):
      - shape / dtype / contiguity (output contract, issue #146)
      - no NaN / Inf in the output (finite-value inputs must stay finite)
      - causal fully-masked rows (S > S_kv) output exactly 0 (F217)
      - elementwise |out - ref| <= atol + rtol * |ref|
      - max_abs_error <= cap
    """
    if out.shape != ref.shape:
        return float("nan"), [f"shape {tuple(out.shape)} != {tuple(ref.shape)}"]
    problems = []
    if out.dtype != ref.dtype:
        problems.append(f"dtype {out.dtype} != {ref.dtype}")
    if not out.is_contiguous():
        problems.append("output is not contiguous")

    of = out.float().cpu()
    rf = ref.float().cpu()

    n_nan = int(torch.isnan(of).sum())
    n_inf = int(torch.isinf(of).sum())
    if n_nan or n_inf:
        problems.append(f"nan={n_nan} inf={n_inf}")

    # Golden F217: causal fully-masked rows (i < S - S_kv) output exactly 0.
    n_masked = max(0, S - S_kv) if causal else 0
    if n_masked:
        mmax = of[:, :n_masked].abs().max().item()
        if mmax != 0.0:
            problems.append(f"masked_rows_absmax={mmax:.3e}!=0")

    diff = (of - rf).abs()
    max_diff = diff.max().item()
    # f6 tolerances: fp16 atol=1e-3 / bf16 atol=2e-2 (task-spec; the bf16
    # baseline itself shows 1-2 ULP rounding up to 7.8e-3), rtol=1e-2.
    atol = 1e-3 if dtype == torch.float16 else 2e-2
    rtol = 1e-2
    viol = int((diff > atol + rtol * rf.abs()).sum())
    if viol:
        problems.append(f"tolviol={viol}")
    cap = 2e-3 if dtype == torch.float16 else 2e-2
    if max_diff > cap:
        problems.append(f"max_diff={max_diff:.3e}>{cap:.0e}")
    return max_diff, problems


def _compare_special(out, golden64, dtype, val_scale=1.0):
    """Special-value comparison vs the fp64 CPU oracle (G2 semantics).

    Mirrors the eval compare_tensors core: the golden is truncated to the
    output dtype before comparing; NaN positions must match exactly; Inf
    positions must match by sign (one-side Inf is saturated to the dtype
    max finite value and continues through the finite comparison); finite
    positions use the f6 mixed tolerance with a magnitude-scaled cap.

    Returns (max_diff, problems).
    """
    ref = golden64.to(dtype)  # fp64 oracle -> output dtype (eval semantics)
    a = out.to(device="cpu", dtype=torch.float64)
    g = ref.double()
    problems = []

    nan_a, nan_g = torch.isnan(a), torch.isnan(g)
    if not bool((nan_a == nan_g).all()):
        problems.append(f"nan_pos_mismatch={int((nan_a != nan_g).sum())}")

    inf_a, inf_g = torch.isinf(a), torch.isinf(g)
    both_inf = inf_a & inf_g
    if bool(both_inf.any()):
        sign_a = torch.sign(a[both_inf])
        sign_g = torch.sign(g[both_inf])
        if not bool((sign_a == sign_g).all()):
            problems.append("inf_sign_mismatch")
    one_inf = inf_a != inf_g
    if bool(one_inf.any()):
        # Saturate one-side Inf to the dtype max finite value (eval
        # inf_sat semantics) and keep the positions in the comparison.
        max_finite = float(torch.finfo(dtype).max)
        a = torch.where(one_inf, torch.sign(a) * max_finite, a)
        g = torch.where(one_inf, torch.sign(g) * max_finite, g)

    mask = ~(nan_a | nan_g | both_inf)
    max_diff = 0.0
    if bool(mask.any()):
        diff = (a - g).abs()
        tol = (1e-3 if dtype == torch.float16 else 2e-2) + 1e-2 * g.abs()
        viol = int((diff[mask] > tol[mask]).sum())
        if viol:
            problems.append(f"tolviol={viol}")
        max_diff = diff[mask].max().item()
        cap = (2e-3 if dtype == torch.float16 else 2e-2) * max(1.0, val_scale)
        if max_diff > cap:
            problems.append(f"max_diff={max_diff:.3e}>{cap:.3e}")
    return max_diff, problems


# ===========================================================================
# Input generators
# ===========================================================================


def _gen_uniform(shape, dtype, scale, seed):
    """Uniform [-scale, scale] CPU tensor (scale=1 -> the eval value_range
    [-1, 1] path). Deterministic per-tensor torch.Generator seed."""
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(shape, dtype=torch.float32, generator=g) * 2 - 1
    return (t * scale).to(dtype)


def _gen_uniform3(B, S, S_kv, N_q, N_kv, D, dtype, scale, seed):
    """(q, k, v) triple of uniform [-scale, scale] CPU tensors."""
    q = _gen_uniform((B, S, N_q, D), dtype, scale, seed)
    k = _gen_uniform((B, S_kv, N_kv, D), dtype, scale, seed + 1)
    v = _gen_uniform((B, S_kv, N_kv, D), dtype, scale, seed + 2)
    return q, k, v


def _gen_fullrange(shape, dtype, seed):
    """Uniform over the full fp16 representable range [-65504, 65504] —
    replica of the eval data generator's special-value "fullrange" variant
    (f64 rand + clamp to the dtype bounds)."""
    g = torch.Generator().manual_seed(seed)
    min_val, max_val = -65504.0, 65504.0
    t = torch.rand(shape, dtype=torch.float64, generator=g) * (max_val - min_val) + min_val
    finfo = torch.finfo(dtype)
    return torch.clamp(t, finfo.min, finfo.max).to(dtype)


# ===========================================================================
# Case runners
# ===========================================================================


def _run_regular(name, B, S, S_kv, N_q, N_kv, D, causal, scale, dtype, blocking):
    """Run one finite-value case with the f6 input/seed/golden/tolerance
    semantics. Prints the level marker; returns True iff the case passed.

    blocking=True (L0/L1) -> [PRECISION_PASS] / [PRECISION_FAIL]
    blocking=False (Boundary) -> [BOUNDARY_PASS] / [BOUNDARY_WARN]
    """
    dtag = "fp16" if dtype == torch.float16 else "bf16"
    tag = f"{name} B={B} S={S} S_kv={S_kv} N_q={N_q} N_kv={N_kv} D={D} causal={causal} scale={scale} {dtag}"
    try:
        torch.manual_seed(42 + B + S + S_kv)  # f6 seed semantics
        q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
        k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
        v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()

        out = gqa(query=q, key=k, value=v, scaleValue=scale, is_causal=causal)
        torch.npu.synchronize()
        # Golden in the input dtype on NPU — the f6-verified setup.
        ref = ref_gqa(q, k, v, scale, causal)
        torch.npu.synchronize()

        max_diff, problems = _compare_regular(out, ref, dtype, causal, S, S_kv)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        marker = "PRECISION_FAIL" if blocking else "BOUNDARY_WARN"
        print(f"[{marker}] {tag} error={e}", flush=True)
        return False
    if problems:
        marker = "PRECISION_FAIL" if blocking else "BOUNDARY_WARN"
        print(f"[{marker}] {tag} max_diff={max_diff:.3e} PROBLEMS: {'; '.join(problems)}", flush=True)
        return False
    marker = "PRECISION_PASS" if blocking else "BOUNDARY_PASS"
    print(f"[{marker}] {tag} max_diff={max_diff:.3e}", flush=True)
    return True


def _run_fp64_case(name, B, S, S_kv, N_q, N_kv, D, causal, scale, dtype, gen_qkv, val_scale=1.0):
    """Run one special-value case against the fp64 CPU oracle golden.

    gen_qkv: zero-arg callable returning (q, k, v) CPU tensors in `dtype`.
    Non-blocking: prints [BOUNDARY_PASS] / [BOUNDARY_WARN].
    """
    dtag = "fp16" if dtype == torch.float16 else "bf16"
    tag = f"{name} B={B} S={S} S_kv={S_kv} N_q={N_q} N_kv={N_kv} D={D} causal={causal} scale={scale} {dtag}"
    try:
        q, k, v = gen_qkv()
        out = gqa(query=q.npu(), key=k.npu(), value=v.npu(), scaleValue=scale, is_causal=causal)
        torch.npu.synchronize()
        # fp64 CPU oracle golden (G2 fp64_cpu strategy).
        golden64 = ref_gqa(q.double(), k.double(), v.double(), scale, causal)
        max_diff, problems = _compare_special(out, golden64, dtype, val_scale)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        print(f"[BOUNDARY_WARN] {tag} error={e}", flush=True)
        return
    if problems:
        print(f"[BOUNDARY_WARN] {tag} max_diff={max_diff:.3e} PROBLEMS: {'; '.join(problems)}", flush=True)
    else:
        print(f"[BOUNDARY_PASS] {tag} max_diff={max_diff:.3e}", flush=True)


# ===========================================================================
# L0 gate tests — representative aligned shapes (cases.csv base shapes)
# ===========================================================================


def test_gqa_l0():
    """L0 gate tests: regular shapes, for precision convergence. Blocking.

    Covers both dtypes (fp16/bf16), both dispatch paths (prefill S > 4 and
    decode 1 <= S <= 4), causal / non-causal, D = 128 / 256, S < S_kv
    (cross-attention), and an explicit scaleValue. Base shapes are taken
    from the eval cases.csv (f6 CASES ids noted per config).
    """
    # (name, B, S, S_kv, N_q, N_kv, D, causal, scale, dtype)
    configs = [
        ("l0_prefill_causal_fp16", 2, 128, 128, 32, 8, 128, True, -1.0, torch.float16),  # c1 G4
        ("l0_prefill_noncausal_fp16", 2, 128, 128, 48, 4, 128, False, -1.0, torch.float16),  # c2 G12
        ("l0_prefill_g16_causal_bf16", 2, 128, 128, 128, 8, 128, True, -1.0, torch.bfloat16),  # c3 G16
        ("l0_prefill_noncausal_bf16", 4, 256, 256, 64, 8, 128, False, -1.0, torch.bfloat16),  # c4 multi-block
        ("l0_prefill_d256_causal_fp16", 2, 128, 128, 32, 8, 256, True, -1.0, torch.float16),  # c5 D=256
        ("l0_cross_noncausal_bf16", 2, 128, 256, 32, 8, 128, False, -1.0, torch.bfloat16),  # c7 S<S_kv
        ("l0_decode_noncausal_fp16", 16, 1, 2048, 32, 8, 128, False, -1.0, torch.float16),  # c11 decode
        ("l0_decode_causal_bf16", 60, 2, 2048, 32, 8, 128, True, -1.0, torch.bfloat16),  # c16 MTP decode
        ("l0_explicit_scale_causal_bf16", 2, 128, 128, 32, 8, 128, True, 0.08838, torch.bfloat16),  # c20
    ]

    ok = True
    for name, B, S, S_kv, N_q, N_kv, D, causal, scale, dtype in configs:
        ok &= _run_regular(name, B, S, S_kv, N_q, N_kv, D, causal, scale, dtype, blocking=True)
    return ok


# ===========================================================================
# L1 functional tests — irregular shapes + degenerate group sizes
# ===========================================================================


def test_gqa_l1():
    """L1 functional tests: irregular shapes, tail blocks, degenerate GQA.
    Blocking.

    Non-aligned variants follow f6's (S-4, S_kv-4) matrix, exercising the
    transpose tail path (F3), the non-aligned wrapper mask (F1), the
    zero-padding path (F2), and causal fully-masked rows (F4/F217). MQA
    (N_kv=1) and MHA (N_kv=N_q) cover the degenerate group sizes.

    The decode non-causal large-pad-ratio cases (l1_decode_skv100 /
    l1_decode_skv132) close the coverage blind spot behind the attempt-5
    fix: a non-causal decode with S_kv non-128-aligned zero-pads the KV
    columns, and those pad columns MUST be masked with _BIG_NEG or their
    score-0 weight inflates the softmax denominator (systematic output
    shrink proportional to pad ratio). The pre-existing tail case used
    Skv=2044 (pad 0.2% -> inside tolerance, defect masked); Skv=100
    (pad 21.9%) and Skv=132 (pad 48.4%) make the deviation ~5e-2 / ~1.3e-1
    when unmasked, so a regression is caught instead of hidden.
    """
    # (name, B, S, S_kv, N_q, N_kv, D, causal, scale, dtype)
    configs = [
        ("l1_s_tail_causal_fp16", 2, 124, 128, 32, 8, 128, True, -1.0, torch.float16),  # c1 S-4
        ("l1_skv_tail_causal_fp16", 2, 128, 124, 32, 8, 128, True, -1.0, torch.float16),  # c1 S_kv-4
        ("l1_both_tail_noncausal_bf16", 4, 252, 252, 64, 8, 128, False, -1.0, torch.bfloat16),  # c4 both-4
        ("l1_decode_skv_tail_fp16", 16, 1, 2044, 32, 8, 128, False, -1.0, torch.float16),  # c11 S_kv-4 decode (pad 0.2%)
        ("l1_decode_skv100_noncausal_fp16", 16, 1, 100, 32, 8, 128, False, -1.0, torch.float16),  # decode pad 21.9%
        ("l1_decode_skv132_noncausal_fp16", 16, 1, 132, 32, 8, 128, False, -1.0, torch.float16),  # decode pad 48.4%
        ("l1_decode_skv132_noncausal_bf16", 8, 2, 132, 32, 8, 128, False, -1.0, torch.bfloat16),  # decode pad 48.4% bf16
        ("l1_mqa_noncausal_fp16", 2, 128, 128, 32, 1, 128, False, -1.0, torch.float16),  # c18 MQA
        ("l1_mha_causal_bf16", 2, 128, 128, 32, 32, 128, True, -1.0, torch.bfloat16),  # c19 MHA
    ]

    ok = True
    for name, B, S, S_kv, N_q, N_kv, D, causal, scale, dtype in configs:
        ok &= _run_regular(name, B, S, S_kv, N_q, N_kv, D, causal, scale, dtype, blocking=True)
    return ok


# ===========================================================================
# Boundary tests — special values, non-blocking
# ===========================================================================


def test_gqa_boundary():
    """Boundary / special-value tests. Non-blocking: [BOUNDARY_PASS/WARN].

    Semantics from eval_fix_verify/g2_special_values.py (fp64 CPU oracle,
    structural NaN/Inf comparison, fp16 full-range inputs) and the F4/F217
    fully-masked-row contract.
    """
    B, S, S_kv, N_q, N_kv, D = 2, 128, 128, 32, 8, 128
    fp16 = torch.float16

    # 1) Large magnitudes: 8x the eval value range, still comfortably
    #    representable in fp16 scores — the full softmax averaging path
    #    at scale (vs the near-argmax behavior of the full range below).
    def gen_large():
        return _gen_uniform3(B, S, S_kv, N_q, N_kv, D, fp16, scale=8.0, seed=2101)

    _run_fp64_case("boundary_large_value", B, S, S_kv, N_q, N_kv, D, True, -1.0, fp16, gen_large, val_scale=8.0)

    # 2) fp16 full range [-65504, 65504] — the eval's special-value variant
    #    range: near-argmax softmax with outputs up to the dtype bound.
    def gen_fullrange():
        q = _gen_fullrange((B, S, N_q, D), fp16, 2111)
        k = _gen_fullrange((B, S_kv, N_kv, D), fp16, 2112)
        v = _gen_fullrange((B, S_kv, N_kv, D), fp16, 2113)
        return q, k, v

    _run_fp64_case("boundary_fullrange", B, S, S_kv, N_q, N_kv, D, False, -1.0, fp16, gen_fullrange, val_scale=65504.0)

    # 3) Sparse special values — position-sensitive NaN/Inf propagation on a
    #    finite base (sharper than the eval's 50%-NaN mode). With G =
    #    N_q // N_kv = 4, q-head n_kv*G + g shares KV head n_kv, so the
    #    injections land on deterministic output slices:
    #      q[0, 5, 3, 0]   = NaN  -> out[0, 5, 3, :]     all NaN
    #      v[0, 20, 1, 7]  = +Inf -> out[0, :, 4:8, 7]   all +Inf
    #      v[1, 30, 5, 9]  = -Inf -> out[1, :, 20:24, 9] all -Inf
    def gen_sparse():
        q, k, v = _gen_uniform3(B, S, S_kv, N_q, N_kv, D, fp16, scale=1.0, seed=2121)
        q[0, 5, 3, 0] = float("nan")
        v[0, 20, 1, 7] = float("inf")
        v[1, 30, 5, 9] = float("-inf")
        return q, k, v

    _run_fp64_case("boundary_sparse_special", B, S, S_kv, N_q, N_kv, D, False, -1.0, fp16, gen_sparse)

    # 4) Causal fully-masked rows: S > S_kv makes the first S - S_kv query
    #    rows fully masked; they must output exactly 0 (golden F217 / fix
    #    F4). Visible rows use the regular f6 comparison. Non-blocking.
    _run_regular("boundary_full_masked_rows", 2, 128, 64, 32, 8, 128, True, -1.0, fp16, blocking=False)

    # 5) Decode (S<=4) + non-causal + non-128-aligned S_kv with special
    #    values — the attempt-5 fix masks the zero-filled KV pad columns
    #    with _BIG_NEG.  These cases confirm the additive _BIG_NEG pad mask
    #    does NOT fight NaN/Inf/full-range semantics: the fp64 CPU oracle is
    #    reproduced structurally (NaN positions exact, Inf by sign) and the
    #    finite positions stay inside tolerance.  Decode shape, non-causal,
    #    Skv=300 (pad 21.9%).
    dB, dS, dSkv, dNq, dNkv, dD = 2, 1, 300, 32, 8, 128

    def gen_decode_fullrange():
        q = _gen_fullrange((dB, dS, dNq, dD), fp16, 2141)
        k = _gen_fullrange((dB, dSkv, dNkv, dD), fp16, 2142)
        v = _gen_fullrange((dB, dSkv, dNkv, dD), fp16, 2143)
        return q, k, v

    _run_fp64_case(
        "boundary_decode_noncausal_fullrange",
        dB,
        dS,
        dSkv,
        dNq,
        dNkv,
        dD,
        False,
        -1.0,
        fp16,
        gen_decode_fullrange,
        val_scale=65504.0,
    )

    # Sparse NaN/Inf on the decode non-causal unaligned path. With G = 4,
    # q-head n_kv*G + g shares KV head n_kv:
    #   q[0,0,3,0]=NaN   -> out[0,0,3,:]   all NaN
    #   v[0,50,1,7]=+Inf -> out[0,0,4:8,7] all +Inf (weight at col 50 > 0)
    #   v[1,299,0,9]=-Inf-> out[1,0,0:4,9] all -Inf (col 299 = LAST real col)
    def gen_decode_sparse():
        q, k, v = _gen_uniform3(dB, dS, dSkv, dNq, dNkv, dD, fp16, scale=1.0, seed=2151)
        q[0, 0, 3, 0] = float("nan")
        v[0, 50, 1, 7] = float("inf")
        v[1, dSkv - 1, 0, 9] = float("-inf")
        return q, k, v

    _run_fp64_case(
        "boundary_decode_noncausal_sparse_special",
        dB,
        dS,
        dSkv,
        dNq,
        dNkv,
        dD,
        False,
        -1.0,
        fp16,
        gen_decode_sparse,
    )


# ===========================================================================
# Main entry
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(description="GQA forward (BSND) layered accuracy tests (Ascend NPU)")
    parser.add_argument(
        "--level",
        default="l0",
        choices=["l0", "l1", "boundary", "all"],
        help="Test level to run (default: l0)",
    )
    args = parser.parse_args()

    tilelang.disable_cache()

    blocking_ok = True  # Only L0/L1 count toward blocking

    if args.level in ("l0", "all"):
        blocking_ok &= test_gqa_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_gqa_l1()
    if args.level in ("boundary", "all"):
        test_gqa_boundary()

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
