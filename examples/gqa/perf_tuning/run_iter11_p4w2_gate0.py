"""Gate 0 runner — iter11 P4 wave-2 (transpose_qkv elimination extended to
g2stack + prefill_expert BSND strided direct reads).

Standalone: does NOT modify examples/gqa/gqa.py or test_gqa.py.  Both arms run
through the PUBLIC gqa() API in ONE process, caches cleared per arm (iter5
finding 2).  SERIAL execution only (Fixed-Core-20 expert kernels, device
serialization discipline).

  BASE = shipped chain: transpose_qkv (BSND -> padded BHSD, ONE launch) ->
         g2stack/prefill_expert main kernel (BHSD reads) -> BSND output
         (iter9 out_bsnd, tout already eliminated on aligned S).
  ARM  = W2 variant chain: the two factories replaced by
         perf_tuning/gqa_p4w2_inbsnd_iter11_variant.py (5 audited line-scoped
         edit sites each, gen_variant_iter11_p4w2.py) with in_bsnd=True
         injected, and transpose_qkv conditionally PASSED THROUGH under the
         SAME gate predicate (p4w2_gate: dual-aligned AND resolved to
         g2stack/prefill_expert — the iter10 harness-defect lesson: ONE
         predicate drives both the factory patch and the pass-through,
         mirroring the integration's single in_bsnd_eff variable).
         PER-HEAD / flat kv1/kv2 / decode factories are NEVER patched; their
         shapes appear below as isolation cases (wave-1 flat cases must show
         tqkv ABSENT in BOTH arms — the shipped adapter already skips it
         there; that is the "wave-1 undisturbed" evidence).

Criteria (PREREGISTERED in candidate_iter11_P4w2.json — bit-identity, NOT a
relaxed tolerance):

  L1 covered  : ARM final output BIT-IDENTICAL to BASE (raw-bit view(int16),
                iter8 finding 3) + golden viol_arm == viol_base == 0 (f6
                policy; boundary_sparse: NaN/Inf positions+signs+finite bits
                equal; fullrange: golden-invariance criterion, iter7/8
                precedent) + variant INVOKED + tqkv SKIPPED in ARM / RAN in
                BASE + chosen == expected spec (both arms)
  L3 isolation: variant NOT invoked, tqkv behavior IDENTICAL across arms,
                outputs BIT-IDENTICAL, chosen identical — incl. per-head
                (excluded), wave-1 flat (tqkv ABSENT both arms), decode (no
                tqkv at all), and the DOMAIN-BOUNDARY cases: non-aligned-S
                g2stack/prexpert hits (tqkv RUN + tout RUN both arms),
                padcol-Skv400 g2stack hit (S aligned, S_kv non-aligned =>
                tqkv RUN tail-mode, tout SKIP), odd-kv per-head fallback.

Run (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python3 examples/gqa/perf_tuning/run_iter11_p4w2_gate0.py [--only SUBSTR]
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
import gqa_p4w2_inbsnd_iter11_variant as VAR  # noqa: E402

RTOL = 1e-2
EXPECT_MD5 = "f59b561212e60a50b78a51671999ef7c"
W2_TARGETS = ("_gqa_kernel_prefill_g2stack", "_gqa_kernel_prefill_expert")
W2_CHOSEN = ("g2stack", "prefill_expert")


def _atol(dtype):
    return 1e-3 if dtype == torch.float16 else 2e-2


def _rawbit(t):
    return t.view(torch.int16)


def p4w2_gate(B, S, S_kv, N_q, N_kv, D, dtype, causal):
    """Pure-host replication of gqa_expert's prefill dispatch (flat variants
    FIRST, then g2stack, then prefill_expert) + the P4 dual-alignment gate.
    Mirrors gqa.py:4798-5002 (use_expert_prefill / use_g2_stack / flat kv1/kv2
    precedence / g2stack / prefill_expert specs) — post-hoc verified against
    the resolved `chosen` in every case, so a predicate drift fails loudly.
    True ONLY when the resolution lands on g2stack or prefill_expert WITH both
    row dims aligned (the wave-2 in_bsnd domain)."""
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
    if use_g2_stack:
        flat_block_num = B * N_kv * (G_ // 2) * q_blocks
        # flat variants dispatch AHEAD of g2stack (wave-1 territory — NOT
        # re-gated here; their in_bsnd path is already shipped)
        if kv_iters == 1 and flat_block_num >= 40:
            return False
        if kv_iters == 2 and not causal and flat_block_num >= 40:
            return False
        return True  # -> g2stack (wave-2 target)
    return True  # -> prefill_expert (wave-2 target)


# --------------------------------------------------------------------------
# L1 — covered domain (g2stack / prefill_expert, aligned S AND aligned S_kv)
# (label, B, S, S_kv, N_q, N_kv, D, causal, dtype, expected_chosen)
COVERED = [
    # ---- g2stack (bench-exact c8/c9/c10 + structural边界 carriers) ----
    ("c8_like_kv8_causal", 4, 1024, 1024, 32, 8, 128, True, torch.float16, "g2stack"),
    ("c9_like_kv4_noncausal", 4, 512, 512, 64, 8, 128, False, torch.bfloat16, "g2stack"),
    ("c10_like_kv4_causal_nq128", 4, 512, 512, 128, 8, 128, True, torch.bfloat16, "g2stack"),
    ("kv2_causal_g2stack", 2, 128, 256, 32, 8, 128, True, torch.float16, "g2stack"),
    # dispatch boundary (stacked==20 < 40 => g2stack not flat) + N_kv=1
    # contiguous-degenerate K/V strides + ns=1 parity dead-branch form
    ("ns1_stacked20_mqa", 1, 128, 128, 20, 1, 128, True, torch.float16, "g2stack"),
    # mixed mc (block_num 32 % 20 = 12 => r_tasks>0 both-parity task slots)
    ("small_mixed_mc_causal", 1, 256, 256, 16, 4, 128, True, torch.float16, "g2stack"),
    # ---- prefill_expert ----
    ("c19_like_mha_causal", 2, 128, 128, 32, 32, 128, True, torch.bfloat16, "prefill_expert"),
    ("mha_noncausal_nomask", 2, 128, 128, 32, 32, 128, False, torch.float16, "prefill_expert"),
    ("g3_odd_kv2_causal", 2, 256, 256, 24, 8, 128, True, torch.float16, "prefill_expert"),
    ("g3_kv4_causal_bf16", 2, 256, 512, 24, 8, 128, True, torch.bfloat16, "prefill_expert"),
    # ---- special values (iter5/8/9/10 precedent set; causal kv2 keeps the
    # g2stack case out of the flat_kv2 dispatch) ----
    ("boundary_sparse_g2stack", 2, 128, 256, 32, 8, 128, True, torch.float16, "g2stack"),
    ("boundary_sparse_prexpert", 2, 128, 128, 32, 32, 128, True, torch.bfloat16, "prefill_expert"),
    ("fullrange_g2stack", 2, 128, 256, 32, 8, 128, True, torch.float16, "g2stack"),
    ("fullrange_prexpert", 2, 128, 128, 32, 32, 128, True, torch.bfloat16, "prefill_expert"),
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
    # WAVE-1 FLAT (adopted in_bsnd domain — tqkv must be ABSENT in BOTH arms;
    # this is the "flat wave-1 undisturbed" evidence; flat factories unpatched)
    ("r2_flat_wave1", 2, 128, 128, 32, 8, 128, True, torch.float16, "g2stack_flat", "absent"),
    ("r4_flatkv2_wave1", 4, 256, 256, 64, 8, 128, False, torch.bfloat16, "g2stack_flat_kv2", "absent"),
    # decode — BSND direct supply, no tqkv at all
    ("r5_decode_expert", 16, 1, 2048, 32, 8, 128, False, torch.float16, "dense_expert", "absent"),
    ("decode_auto_small", 2, 2, 512, 8, 2, 128, False, torch.float16, "dense_auto", "absent"),
    # DOMAIN BOUNDARY: g2stack dispatch HIT but non-aligned S => in_bsnd gate
    # excludes => tqkv RUNS (pads rows) + tout RUNS (bsnd_eff False)
    ("s130_g2stack_unaligned", 2, 130, 130, 32, 8, 128, True, torch.float16, "g2stack", "run"),
    # DOMAIN BOUNDARY: prefill_expert hit, non-aligned S
    ("s100e_prexpert_unaligned", 2, 100, 128, 32, 32, 128, True, torch.bfloat16, "prefill_expert", "run"),
    # DOMAIN BOUNDARY: g2stack hit, S ALIGNED but S_kv=400 non-aligned =>
    # K/V zero-pad columns needed => tqkv RUNS (tail mode) while tout SKIPS
    # (bsnd_eff True — the S-aligned iter9 out_bsnd path stays live)
    ("padcol_skv400_g2stack", 2, 256, 400, 32, 8, 128, True, torch.float16, "g2stack", "run"),
]


def _clear():
    G._KERNEL_BUILD_CACHE.clear()
    G._COMPILED_CACHE.clear()


def _inputs(label, B, S, S_kv, N_q, N_kv, D, dtype):
    if label.startswith("fullrange"):
        lo, hi = -65504.0, 65504.0

        def gen(shape, off):
            gg = torch.Generator().manual_seed(3111 + off)
            t = torch.rand(shape, dtype=torch.float64, generator=gg) * (hi - lo) + lo
            return t.to(dtype).npu()

        return gen((B, S, N_q, D), 0), gen((B, S_kv, N_kv, D), 1), gen((B, S_kv, N_kv, D), 2)
    torch.manual_seed(42 + B + S + S_kv + N_q)
    q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
    k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
    v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
    if label.startswith("boundary_sparse"):
        # iter5 verbatim injection set, both kv blocks for the kv2 shape
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
    """Installs/removes the W2 monkeypatches; records dispatch evidence.
    Only the two wave-2 factories are ever patched; ONE gate predicate drives
    BOTH the factory patch and the tqkv pass-through (iter10 lesson)."""

    def __init__(self):
        self.invoked = []
        self.tqkv_calls = []
        self._saved = None
        self.gated = False

    def install(self, shape, causal, enable_variant=True):
        self.invoked = []
        self.tqkv_calls = []
        self.enable_variant = enable_variant
        B, S, S_kv, N_q, N_kv, D, dtype = shape
        self.gated = enable_variant and p4w2_gate(B, S, S_kv, N_q, N_kv, D, dtype, causal)
        self._saved = tuple(getattr(G, n) for n in W2_TARGETS) + (G.transpose_qkv,)
        if self.gated:
            for name in W2_TARGETS:
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
                # W2 pass-through: the aligned-domain transpose_qkv is a pure
                # rearrangement; the in_bsnd kernel reads these exact tensors.
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
            for name, orig in zip(W2_TARGETS, self._saved[:-1]):
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
        gmsg = (f"[fullrange: golden-invariance criterion] viol_arm={viol_a} viol_base={viol_b} "
                f"nan_arm={nan_a} nan_base={nan_b} nan_ref={nanref_a} inf_arm={inf_a}")
    else:
        checks["golden"] = (viol_a == 0 and viol_b == 0 and nan_a == 0 and inf_a == 0)
        checks["golden_invariant"] = (viol_a == viol_b)
        gmsg = f"viol_arm={viol_a} viol_base={viol_b} nan={nan_a} max_diff={md_a:.3e}"
    # dispatch + capability evidence
    checks["variant_invoked"] = bool(inv) and all(x in W2_TARGETS for x in inv)
    checks["chosen"] = bool(chosen_a) and chosen_a[0] == expect and chosen_b[0] == expect
    checks["tqkv_skipped_arm"] = all(c[0] == "SKIP" for c in tqkv_a) and len(tqkv_a) == 1
    checks["tqkv_ran_base"] = all(c[0] == "RUN" for c in tqkv_b) and len(tqkv_b) == 1
    checks["gate_predicate"] = p4w2_gate(B, S, S_kv, N_q, N_kv, D, dtype, causal)

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
        # decode: never called; wave-1 flat: shipped adapter in_bsnd skip —
        # ABSENT IN BOTH ARMS is the "wave-1 undisturbed" evidence
        checks["tqkv_absent_both"] = not tqkv_a and not tqkv_b
    else:
        checks["tqkv_ran_both"] = (
            len(tqkv_a) == 1 and tqkv_a[0][0] == "RUN"
            and len(tqkv_b) == 1 and tqkv_b[0][0] == "RUN"
        )
    # gate predicate must be False for every isolation case (domain boundary)
    checks["gate_false"] = not p4w2_gate(B, S, S_kv, N_q, N_kv, D, dtype, causal)
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
