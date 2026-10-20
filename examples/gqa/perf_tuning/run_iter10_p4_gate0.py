"""Gate 0 runner — iter10 P4 (transpose_qkv elimination, flat kv1/kv2 BSND
strided direct reads).

Standalone: does NOT modify examples/gqa/gqa.py or test_gqa.py.  Both arms run
through the PUBLIC gqa() API in ONE process, caches cleared per arm (iter5
finding 2).  SERIAL execution only (Fixed-Core-20 expert kernels, device
serialization discipline).

  BASE = shipped chain: transpose_qkv (BSND -> padded BHSD, ONE launch) ->
         flat main kernel (BHSD reads) -> BSND output (wave-1 out_bsnd, tout
         already eliminated).
  ARM  = P4 variant chain: the two flat factories replaced by
         perf_tuning/gqa_p4_inbsnd_iter10_variant.py (5 audited line-scoped
         edit sites each, gen_variant_iter10_p4.py) with in_bsnd=True
         injected, and transpose_qkv conditionally PASSED THROUGH (the
         monkeypatch equivalent of the integration's capability flag +
         adapter reorder).  Host gate: S % 64 == 0 AND S_kv % 128 == 0 AND
         the replicated flat-dispatch predicate (post-hoc verified against
         the resolved `chosen`).
         PER-HEAD / g2stack / prefill_expert factories are NEVER patched
         (first-wave scope); their shapes appear below as isolation cases.

Criteria (PREREGISTERED in candidate_iter10_P4.json — bit-identity, NOT a
relaxed tolerance: transpose_qkv on the aligned domain is a pure
rearrangement, zero arithmetic change):

  L1 covered  : ARM final output BIT-IDENTICAL to BASE (raw-bit view(int16),
                iter8 finding 3) + golden viol_arm == viol_base == 0 (f6
                policy; boundary_sparse: NaN/Inf positions+signs+finite bits
                equal; fullrange: golden-invariance criterion, iter7/8
                precedent) + variant INVOKED + tqkv SKIPPED in ARM / RAN in
                BASE + chosen == expected flat spec (both arms)
  L3 isolation: variant NOT invoked, tqkv behavior IDENTICAL across arms,
                outputs BIT-IDENTICAL — incl. per-head (excluded domain),
                g2stack, prefill_expert, decode (no tqkv at all), the
                non-aligned-S flat-hit gate case and the padcol-Skv200
                flat-hit gate case (S_kv alignment gate independence).

Run (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python3 examples/gqa/perf_tuning/run_iter10_p4_gate0.py [--only SUBSTR]
"""

import argparse
import hashlib
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
import gqa_p4_inbsnd_iter10_variant as VAR  # noqa: E402

RTOL = 1e-2
EXPECT_MD5 = "606b07ff474ec42b8244f0831327d75b"
P4_TARGETS = ("_gqa_kernel_prefill_g2stack_flat", "_gqa_kernel_prefill_g2stack_flat_kv2")
P4_CHOSEN = ("g2stack_flat", "g2stack_flat_kv2")


def _atol(dtype):
    return 1e-3 if dtype == torch.float16 else 2e-2


def _rawbit(t):
    return t.view(torch.int16)


def p4_gate(B, S, S_kv, N_q, N_kv, D, dtype, causal):
    """Pure-host replication of gqa_expert's flat dispatch + the P4 alignment
    gate.  Mirrors gqa.py:4620-4740 (use_expert_prefill / use_g2_stack /
    g2stack_flat / g2stack_flat_kv2 specs) — post-hoc verified against the
    resolved `chosen` in every case, so a predicate drift fails loudly."""
    if S % 64 != 0 or S_kv % 128 != 0:
        return False
    if D != 128 or dtype not in (torch.float16, torch.bfloat16):
        return False
    if N_q % N_kv != 0:
        return False
    G_ = N_q // N_kv
    kv_iters = S_kv // 128
    q_blocks = S // 64
    use_expert_prefill = (
        (kv_iters == 1 or (kv_iters >= 2 and kv_iters % 2 == 0))
        and ((not causal) or (S_kv >= S))
        and (B * N_q * q_blocks) >= 20
    )
    if not use_expert_prefill:
        return False
    use_g2_stack = (G_ % 2 == 0) and (B * N_kv * (G_ // 2) * q_blocks) >= 20
    if not use_g2_stack:
        return False
    flat_block_num = B * N_kv * (G_ // 2) * q_blocks
    if kv_iters == 1 and flat_block_num >= 40:
        return True
    if kv_iters == 2 and not causal and flat_block_num >= 40:
        return True
    return False


# --------------------------------------------------------------------------
# L1 — covered domain (flat kv1/kv2, aligned S AND aligned S_kv)
# (label, B, S, S_kv, N_q, N_kv, D, causal, dtype, expected_chosen)
COVERED = [
    # ---- flat kv1 ----
    ("c1_like_kv1_causal", 2, 128, 128, 32, 8, 128, True, torch.float16, "g2stack_flat"),
    ("c2_like_kv1_noncausal", 2, 128, 128, 48, 4, 128, False, torch.float16, "g2stack_flat"),
    ("c20_like_kv1_bf16_causal", 2, 128, 128, 32, 8, 128, True, torch.bfloat16, "g2stack_flat"),
    ("c18_like_mqa_nkv1", 2, 128, 128, 32, 1, 128, False, torch.float16, "g2stack_flat"),
    ("c3_like_kv1_nq128_bf16", 2, 128, 128, 128, 8, 128, True, torch.bfloat16, "g2stack_flat"),
    ("mc2_boundary_stacked40", 1, 128, 128, 40, 2, 128, True, torch.float16, "g2stack_flat"),
    ("mixed_mc_8tasks", 2, 128, 128, 24, 1, 128, False, torch.float16, "g2stack_flat"),
    # ---- flat kv2 ----
    ("c4_like_kv2_bf16", 4, 256, 256, 64, 8, 128, False, torch.bfloat16, "g2stack_flat_kv2"),
    ("c7_like_kv2_cross", 2, 128, 256, 32, 8, 128, False, torch.bfloat16, "g2stack_flat_kv2"),
    ("kv2_mixed_mc_46", 1, 128, 256, 46, 1, 128, False, torch.float16, "g2stack_flat_kv2"),
    # ---- special values (iter5/8/9 precedent set) ----
    ("boundary_sparse_kv1", 2, 128, 128, 32, 8, 128, True, torch.float16, "g2stack_flat"),
    ("boundary_sparse_kv2", 4, 256, 256, 64, 8, 128, False, torch.bfloat16, "g2stack_flat_kv2"),
    ("fullrange_kv1", 2, 128, 128, 48, 4, 128, False, torch.float16, "g2stack_flat"),
    ("fullrange_kv2", 4, 256, 256, 64, 8, 128, False, torch.bfloat16, "g2stack_flat_kv2"),
]

# --------------------------------------------------------------------------
# L3 — out-of-domain isolation (BIT-IDENTICAL and untouched)
# (label, B, S, S_kv, Nq, Nkv, D, causal, dtype, expected_chosen_prefix,
#  tqkv_expected)  tqkv_expected: "run" | "absent"
OUT_OF_DOMAIN = [
    # per-head AUTO (excluded domain — never patched)
    ("r1_perhead_bm64", 1, 128, 128, 8, 2, 128, True, torch.float16, "per_head_bm64", "run"),
    ("c5_d256_perhead", 2, 128, 128, 32, 8, 256, True, torch.float16, "per_head_bm64", "run"),
    ("odd_kv3_perhead", 2, 128, 300, 32, 8, 128, False, torch.float16, "per_head_bm64", "run"),
    # expert families OUT of the first wave (never patched; tout already
    # skipped by wave-2 out_bsnd — tqkv must STILL RUN in both arms)
    ("c19_prexpert_mha", 2, 128, 128, 32, 32, 128, True, torch.bfloat16, "prefill_expert", "run"),
    ("c9_like_g2stack_kv4", 4, 512, 512, 64, 8, 128, False, torch.bfloat16, "g2stack", "run"),
    ("kv2_causal_g2stack", 2, 128, 256, 32, 8, 128, True, torch.float16, "g2stack", "run"),
    # decode — BSND direct supply, no tqkv at all
    ("r5_decode_expert", 16, 1, 2048, 32, 8, 128, False, torch.float16, "dense_expert", "absent"),
    ("decode_auto_small", 2, 2, 512, 8, 2, 128, False, torch.float16, "dense_auto", "absent"),
    # GATE ISOLATION: flat-dispatch HIT but non-aligned S => in_bsnd gate
    # excludes => tqkv RUNS (pads rows) + tout(s_real) RUNS (S non-aligned =>
    # bsnd_eff False => BHSD + transpose_out)
    ("s100_flat_unaligned", 2, 100, 128, 32, 8, 128, True, torch.float16, "g2stack_flat", "run"),
    # GATE ISOLATION: flat kv2 dispatch HIT, S aligned but S_kv=200
    # non-aligned => K/V zero-pad columns needed => tqkv RUNS (wave-1
    # out_bsnd still skips tout — S aligned)
    ("padcol_skv200_flatkv2", 2, 128, 200, 32, 8, 128, False, torch.float16, "g2stack_flat_kv2", "run"),
    # GATE ISOLATION: g2stack with non-aligned S (iter9 l1_s_tail analog)
    ("l1_s_tail_s130_g2stack", 2, 130, 130, 32, 8, 128, True, torch.float16, "g2stack", "run"),
]


def _clear():
    G._KERNEL_BUILD_CACHE.clear()
    G._COMPILED_CACHE.clear()


def _inputs(label, B, S, S_kv, N_q, N_kv, D, dtype):
    if label.startswith("fullrange"):
        lo, hi = -65504.0, 65504.0

        def gen(shape, off):
            gg = torch.Generator().manual_seed(3101 + off)
            t = torch.rand(shape, dtype=torch.float64, generator=gg) * (hi - lo) + lo
            return t.to(dtype).npu()

        return gen((B, S, N_q, D), 0), gen((B, S_kv, N_kv, D), 1), gen((B, S_kv, N_kv, D), 2)
    torch.manual_seed(42 + B + S + S_kv + N_q)
    q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
    k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
    v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
    if label.startswith("boundary_sparse"):
        # iter5 verbatim injection set, extended to both kv blocks for the
        # kv2 shape (v indices land in kv-block 0 and 1 when S_kv=256):
        q[0, 5, 3, 0] = float("nan")
        v[0, 20, 1, 7] = float("inf")
        v[1, min(200, S_kv - 1), 5, 9] = float("-inf")
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
        return False, f"finite_bit_mismatch={int((a[finite] != b[finite]).sum())}"
    return True, "ok"


class Arm:
    """Installs/removes the P4 monkeypatches; records dispatch evidence.
    Only the two flat factories are ever patched (first-wave scope)."""

    def __init__(self):
        self.invoked = []
        self.tqkv_calls = []
        self._saved = None
        self.causal = False
        self.enable_variant = True

    def install(self, shape, causal, enable_variant=True):
        """shape = (B, S, S_kv, N_q, N_kv, D, dtype).

        HARNESS-DEFECT FIX (disclosed, criteria untouched — iter7/8/9 harness
        fix precedent): the factory patch and the transpose_qkv pass-through
        MUST be driven by the SAME gate predicate.  The first version patched
        the factories unconditionally while the pass-through was gated, so on
        non-aligned flat-hit shapes (s100/padcol_skv200) the ARM compiled an
        in_bsnd kernel but fed it BHSD tensors (tilelang does not validate
        input shapes) => silent corruption caught by the isolation cases.
        The integration avoids this structurally: a single `in_bsnd_eff`
        host variable drives BOTH the spec-builder kwarg and the adapter's
        capability-flag skip (iter8 out_bsnd pattern).
        """
        self.invoked = []
        self.tqkv_calls = []
        self.causal = causal
        self.enable_variant = enable_variant
        B, S, S_kv, N_q, N_kv, D, dtype = shape
        self.gated = enable_variant and p4_gate(B, S, S_kv, N_q, N_kv, D, dtype, causal)
        self._saved = tuple(getattr(G, n) for n in P4_TARGETS) + (G.transpose_qkv,)
        if self.gated:
            for name in P4_TARGETS:
                var_fn = getattr(VAR, name)

                def wrap(*a, _n=name, _f=var_fn, **kw):
                    kw["in_bsnd"] = True  # the integration's in_bsnd_eff spec pass-through
                    self.invoked.append(_n)
                    return _f(*a, **kw)

                setattr(G, name, wrap)

        orig_tqkv = self._saved[-1]
        arm = self

        def tqkv_cond(query, key, value, S_out=None, S_kv_out=None):
            if arm.gated:
                # P4 pass-through: the aligned-domain transpose_qkv is a pure
                # rearrangement; the in_bsnd kernel reads these exact tensors.
                # (The gate was evaluated ONCE in install() from the case's
                # declared shape — same predicate, same variables.)
                S = query.shape[1]
                Skv = key.shape[1]
                assert S_out in (None, S) and S_kv_out in (None, Skv), \
                    f"pass-through requires no padding, got S_out={S_out} S_kv_out={S_kv_out}"
                arm.tqkv_calls.append(("SKIP", tuple(query.shape)))
                return (query, key, value)
            arm.tqkv_calls.append(("RUN", tuple(query.shape)))
            return orig_tqkv(query, key, value, S_out=S_out, S_kv_out=S_kv_out)

        G.transpose_qkv = tqkv_cond

    def remove(self):
        if self._saved is not None:
            for name, orig in zip(P4_TARGETS, self._saved[:-1]):
                setattr(G, name, orig)
            G.transpose_qkv = self._saved[-1]
            self._saved = None


def _run_arm(q, k, v, causal, shape, arm=None, enable_variant=True):
    _clear()
    if arm is not None:
        arm.install(shape, causal, enable_variant=enable_variant)
    try:
        out = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
        torch.npu.synchronize()
        chosen = [val[2] for val in G._KERNEL_BUILD_CACHE.values()]
        tqkv_calls = list(arm.tqkv_calls) if arm is not None else None
        inv = list(arm.invoked) if arm is not None else None
    finally:
        if arm is not None:
            arm.remove()
        _clear()
    return out, chosen, tqkv_calls, inv


def _golden(q, k, v, causal, out, dtype):
    ref = G._ref_gqa(q, k, v, -1.0, causal)
    torch.npu.synchronize()
    got = out.cpu().float()
    rf = ref.cpu().float()
    finite = torch.isfinite(rf) & torch.isfinite(got)
    diff = (got - rf).abs()
    atol = _atol(dtype)
    viol = int((diff[finite] > atol + RTOL * rf.abs()[finite]).sum())
    return (viol, int(torch.isnan(got).sum()), int(torch.isnan(rf).sum()), int(torch.isinf(got).sum()),
            (float(diff[finite].max()) if bool(finite.any()) else float("nan")))


def run_covered(case):
    label, B, S, S_kv, N_q, N_kv, D, causal, dtype, expect = case
    q, k, v = _inputs(label, B, S, S_kv, N_q, N_kv, D, dtype)
    special = label.startswith("boundary_sparse")
    fullrange = label.startswith("fullrange")

    # BASE arm: patches installed but variant DISABLED (identical code path
    # shape, tqkv RUN asserted below) — mirrors iter9's enable_variant=False
    shape = (B, S, S_kv, N_q, N_kv, D, dtype)
    arm_b = Arm()
    out_b, chosen_b, tqkv_b, inv_b = _run_arm(q, k, v, causal, shape, arm=arm_b, enable_variant=False)
    arm = Arm()
    out_a, chosen_a, tqkv_a, inv = _run_arm(q, k, v, causal, shape, arm=arm)

    checks = {}
    checks["bit_identical"] = bool(torch.equal(_rawbit(out_b.cpu()), _rawbit(out_a.cpu())))
    viol_b, nan_b, nanref_b, inf_b, md_b = _golden(q, k, v, causal, out_b, dtype)
    viol_a, nan_a, nanref_a, inf_a, md_a = _golden(q, k, v, causal, out_a, dtype)
    if special:
        eq, why = _special_struct_equal(out_a.cpu().float(), out_b.cpu().float())
        checks["special_struct"] = eq
        checks["golden"] = (nan_a == nan_b == nanref_a == nanref_b) and viol_a == viol_b
        gmsg = (f"nan_arm={nan_a} nan_base={nan_b} nan_ref={nanref_a} inf_arm={inf_a} "
                f"viol_arm={viol_a} viol_base={viol_b} special={why}")
    elif fullrange:
        checks["golden_invariant"] = (viol_a == viol_b and nan_a == nan_b and inf_a == inf_b)
        gmsg = (f"[fullrange: golden SATURATED nan_ref={nanref_a} — absolute tol N/A] "
                f"viol_arm={viol_a} viol_base={viol_b} nan_arm={nan_a} nan_base={nan_b} "
                f"inf_arm={inf_a}")
    else:
        checks["golden"] = (viol_a == 0 and viol_b == 0 and nan_a == 0 and inf_a == 0)
        checks["golden_invariant"] = (viol_a == viol_b)
        gmsg = f"viol_arm={viol_a} viol_base={viol_b} nan={nan_a} max_diff={md_a:.3e}"
    # dispatch + capability evidence
    checks["variant_invoked"] = bool(inv) and all(x in P4_TARGETS for x in inv)
    checks["chosen"] = bool(chosen_a) and chosen_a[0] == expect and chosen_b[0] == expect
    checks["tqkv_skipped_arm"] = all(c[0] == "SKIP" for c in tqkv_a) and len(tqkv_a) == 1
    checks["tqkv_ran_base"] = all(c[0] == "RUN" for c in tqkv_b) and len(tqkv_b) == 1
    checks["gate_predicate"] = p4_gate(B, S, S_kv, N_q, N_kv, D, dtype, causal)

    ok = all(checks.values())
    print(f"[G0-L1] {label}: {'PASS' if ok else 'FAIL'} bit={checks['bit_identical']} "
          f"invoked={checks['variant_invoked']} chosen={chosen_a[:1]} "
          f"tqkv_arm={[c[0] for c in tqkv_a]} tqkv_base={[c[0] for c in tqkv_b]} | {gmsg}",
          flush=True)
    if not ok:
        print(f"        checks={checks}", flush=True)
    return ok


def run_isolation(case):
    label, B, S, S_kv, N_q, N_kv, D, causal, dtype, expect, tqkv_exp = case
    q, k, v = _inputs(label, B, S, S_kv, N_q, N_kv, D, dtype)

    shape = (B, S, S_kv, N_q, N_kv, D, dtype)
    arm_b = Arm()
    out_b, chosen_b, tqkv_b, inv_b = _run_arm(q, k, v, causal, shape, arm=arm_b, enable_variant=False)
    arm = Arm()
    out_a, chosen_a, tqkv_a, inv = _run_arm(q, k, v, causal, shape, arm=arm)

    checks = {}
    checks["bit_identical"] = bool(torch.equal(_rawbit(out_b.cpu()), _rawbit(out_a.cpu())))
    checks["variant_not_invoked"] = not inv and not inv_b
    checks["chosen_same"] = chosen_a == chosen_b and bool(chosen_a) and chosen_a[0].startswith(expect)
    if tqkv_exp == "absent":
        checks["tqkv_absent_both"] = not tqkv_a and not tqkv_b
    else:
        checks["tqkv_ran_both"] = (
            len(tqkv_a) == 1 and tqkv_a[0][0] == "RUN"
            and len(tqkv_b) == 1 and tqkv_b[0][0] == "RUN"
        )
    # golden sanity on both arms (unchanged policy)
    viol_b, nan_b, _, inf_b, _ = _golden(q, k, v, causal, out_b, dtype)
    viol_a, nan_a, _, inf_a, _ = _golden(q, k, v, causal, out_a, dtype)
    checks["golden_invariant"] = viol_a == viol_b and nan_a == nan_b and inf_a == inf_b

    ok = all(checks.values())
    print(f"[G0-L3] {label}: {'PASS' if ok else 'FAIL'} bit={checks['bit_identical']} "
          f"chosen={chosen_a[:1]} tqkv_arm={[c[0] for c in tqkv_a]} inv={inv} "
          f"viol_arm={viol_a} viol_base={viol_b}", flush=True)
    if not ok:
        print(f"        checks={checks}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    md5 = hashlib.md5(open(os.path.join(GQA_DIR, "gqa.py"), "rb").read()).hexdigest()
    assert md5 == EXPECT_MD5, f"gqa.py drifted: {md5} != {EXPECT_MD5}"
    tmd5 = hashlib.md5(open(os.path.join(GQA_DIR, "test_gqa.py"), "rb").read()).hexdigest()
    assert tmd5 == "3e2d08881cb7b6f36d5c89b2fd37a146", f"test_gqa.py drifted: {tmd5}"

    npass = nfail = 0
    failures = []
    covered = [c for c in COVERED if not args.only or args.only in c[0]]
    ood = [c for c in OUT_OF_DOMAIN if not args.only or args.only in c[0]]
    for case in covered:
        try:
            ok = run_covered(case)
        except Exception:
            traceback.print_exc()
            ok = False
        npass, nfail = (npass + 1, nfail) if ok else (npass, nfail + 1)
        if not ok:
            failures.append(case[0])
    for case in ood:
        try:
            ok = run_isolation(case)
        except Exception:
            traceback.print_exc()
            ok = False
        npass, nfail = (npass + 1, nfail) if ok else (npass, nfail + 1)
        if not ok:
            failures.append(case[0])
    print(f"\n===== GATE 0 SUMMARY: {npass} PASS / {nfail} FAIL =====")
    if failures:
        print("failed:", failures)
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
