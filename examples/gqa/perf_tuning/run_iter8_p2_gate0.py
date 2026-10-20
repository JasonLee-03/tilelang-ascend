"""Gate 0 runner — iter8 P2 (main kernel BSND direct write, transpose_out skipped).

Standalone: does NOT modify examples/gqa/gqa.py or test_gqa.py.  Both arms run
through the PUBLIC gqa() API in ONE process, caches cleared per arm (iter5
finding 2: _KERNEL_BUILD_CACHE + adapter _COMPILED_CACHE must both be cleared
or the pre-patch kernel silently survives).

  BASE = shipped chain: main kernel writes BHSD [B,Nq,S_pad,D] -> transpose_out
         (_transpose_v1, P1-retiled plan) -> [B,S,Nq,D]
  ARM  = P2 variant chain: main kernel writes BSND [B,S_pad,Nq,D] directly
         (perf_tuning/gqa_flat_bsnd_iter8_variant.py, 2 line-scoped edits per
         kernel, audited by gen_variant_iter8_p2.py) -> transpose_out SKIPPED
         (conditional pass-through monkeypatch == the integration's
         `if getattr(kernel_fn,'out_bsnd',False): return out`).
         The wave-1 host gate (S % 64 == 0, i.e. S_padded == S => no crop) is
         applied by this harness exactly where the integration will apply it.

Criteria (PREREGISTERED in candidate_iter8_P2.json — bit-identity, NOT a
relaxed tolerance: P2 is a pure data-path re-partition with zero arithmetic
change, so anything but bit-identity is a bug):

  L1 covered domain : ARM final output BIT-IDENTICAL to BASE + golden viol=0
                      (special-value cases: NaN positions / Inf positions+signs
                      / finite bits all equal — iter5 injection set) +
                      variant factory INVOKED + chosen == expected +
                      transpose_out SKIPPED in ARM / RAN in BASE
  L2 kernel-level   : ARM main output (BSND) == transpose_out(BASE main output
                      (BHSD)) BIT-IDENTICAL — isolates the kernel change from
                      the host path
  L3 out-of-domain  : variant factory NOT invoked, chosen identical to BASE,
                      transpose_out RAN (or absent on decode), output
                      BIT-IDENTICAL to BASE — incl. the non-aligned-S gate
                      isolation case (S=100 resolves to g2stack_flat but MUST
                      stay on the BHSD + transpose_out(s_real) path)

Run (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python3 examples/gqa/perf_tuning/run_iter8_p2_gate0.py [--layer 1|2|3]
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
import gqa_flat_bsnd_iter8_variant as VAR  # noqa: E402

_BIG_NEG = G._BIG_NEG
RTOL = 1e-2


def _atol(dtype):
    """test_gqa.py _compare_regular policy (f6-verified): fp16 1e-3 / bf16 2e-2
    — the bf16 baseline itself shows 1-2 ULP rounding up to 7.8e-3."""
    return 1e-3 if dtype == torch.float16 else 2e-2


def _rawbit(t):
    """NaN-safe bit-level view.  torch.equal uses IEEE == so ANY NaN position
    makes it False even when the payloads are identical — measured on the
    boundary_sparse cases (both arms emit 0x7fff at every differing position;
    triage_iter8_nan.py: BASE-run1 vs BASE-run2 shows the same artifact, so it
    is a comparison-tool property, not a P2 difference).  Raw-int compare is
    the rigorous bit-identity test used throughout this runner."""
    return t.view(torch.int16)

# --------------------------------------------------------------------------
# L1/L2 — covered domain (chosen in {g2stack_flat, g2stack_flat_kv2}, S%64==0)
# (label, B, S, S_kv, N_q, N_kv, D, causal, dtype, expected_chosen)
COVERED = [
    ("r2_flatkv1_causal_fp16", 2, 128, 128, 32, 8, 128, True, torch.float16, "g2stack_flat"),
    ("r6_flatkv1_noncausal_fp16", 2, 128, 128, 48, 4, 128, False, torch.float16, "g2stack_flat"),
    ("r3_flatkv1_causal_bf16", 2, 128, 128, 128, 8, 128, True, torch.bfloat16, "g2stack_flat"),
    ("c18_mqa_flatkv1_fp16", 2, 128, 128, 32, 1, 128, False, torch.float16, "g2stack_flat"),
    ("r4_flatkv2_bf16", 4, 256, 256, 64, 8, 128, False, torch.bfloat16, "g2stack_flat_kv2"),
    ("c7_flatkv2_cross_bf16", 2, 128, 256, 32, 8, 128, False, torch.bfloat16, "g2stack_flat_kv2"),
    ("padcol_skv200_flatkv2_fp16", 2, 128, 200, 32, 8, 128, False, torch.float16, "g2stack_flat_kv2"),
    ("mc2_edge_flatkv2_fp16", 5, 128, 256, 8, 2, 128, False, torch.float16, "g2stack_flat_kv2"),
    ("mixed_mc32_flatkv2_fp16", 3, 128, 256, 16, 4, 128, False, torch.float16, "g2stack_flat_kv2"),
    ("mixed_mc76_flatkv2_fp16", 2, 256, 256, 32, 8, 128, False, torch.float16, "g2stack_flat_kv2"),
    ("causal_kv1_bigG_bf16", 2, 128, 128, 64, 4, 128, True, torch.bfloat16, "g2stack_flat"),
    # ---- special-value / extreme-range cases (iter5 injection set) ----
    ("boundary_sparse_flatkv2_bf16", 2, 128, 256, 32, 8, 128, False, torch.bfloat16, "g2stack_flat_kv2"),
    ("boundary_sparse_flatkv1_fp16", 2, 128, 128, 32, 8, 128, True, torch.float16, "g2stack_flat"),
    ("fullrange_flatkv1_fp16", 2, 128, 128, 32, 8, 128, True, torch.float16, "g2stack_flat"),
    ("fullrange_flatkv2_fp16", 4, 256, 256, 64, 8, 128, False, torch.float16, "g2stack_flat_kv2"),
]

# --------------------------------------------------------------------------
# L3 — out-of-domain isolation (must be BIT-IDENTICAL and untouched)
OUT_OF_DOMAIN = [
    # label, B, S, S_kv, Nq, Nkv, D, causal, dtype, expected_chosen, tout_expected
    ("r1_perhead_bm64", 1, 128, 128, 8, 2, 128, True, torch.float16, "per_head_bm64", True),
    ("r5_decode_expert", 16, 1, 2048, 32, 8, 128, False, torch.float16, "dense_expert", False),
    ("c5_d256_perhead", 2, 128, 128, 32, 8, 256, True, torch.float16, "per_head_bm64", True),
    ("c19_mha_g1_prefill_expert", 2, 128, 128, 32, 32, 128, True, torch.bfloat16, "prefill_expert", True),
    ("c9_like_g2stack_kv4", 4, 512, 512, 64, 8, 128, False, torch.bfloat16, "g2stack", True),
    # GATE ISOLATION: flat-eligible but S%64 != 0 => must stay BHSD + tout(s_real)
    # S_kv=256 => kv_iters=2 => the flat_kv2 gate; S=100 is NOT 64-aligned so
    # the P2 host gate must exclude it (stays BHSD + transpose_out(s_real))
    ("padrow_s100_flatgate_excluded", 2, 100, 256, 32, 8, 128, False, torch.float16, "g2stack_flat_kv2", True),
    ("l1_s_tail_s130_g2stack", 2, 130, 130, 32, 8, 128, True, torch.float16, "g2stack", True),
    ("decode_auto_small", 2, 2, 512, 8, 2, 128, False, torch.float16, "dense_auto", False),
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
        # iter5 verbatim injection set (both kv blocks for the kv2 shapes):
        #   q NaN -> a whole output row NaN;  v +Inf (kv block 0) / -Inf
        #   (kv block 1) -> whole head-group columns Inf with signs
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
    """Installs/removes the P2 monkeypatches; records dispatch evidence."""

    def __init__(self):
        self.invoked = []
        self.tout_calls = []
        self.main_out = None
        self._saved = None

    # -- host gate: identical to the integration's `bsnd_ok = (S_padded == S)`
    def install(self, S, enable_variant=True):
        self.invoked = []
        self.tout_calls = []
        self.main_out = None
        aligned = (S % 64 == 0) and enable_variant
        orig = (G._gqa_kernel_prefill_g2stack_flat, G._gqa_kernel_prefill_g2stack_flat_kv2,
                G.transpose_out)
        self._saved = orig
        self.aligned = aligned
        if aligned:
            def wrap_kv1(*a, **kw):
                self.invoked.append("g2stack_flat")
                return VAR._gqa_kernel_prefill_g2stack_flat(*a, **kw)

            def wrap_kv2(*a, **kw):
                self.invoked.append("g2stack_flat_kv2")
                return VAR._gqa_kernel_prefill_g2stack_flat_kv2(*a, **kw)

            G._gqa_kernel_prefill_g2stack_flat = wrap_kv1
            G._gqa_kernel_prefill_g2stack_flat_kv2 = wrap_kv2

        orig_tout = orig[2]
        arm = self

        def tout_cond(out_bhsd, s_real=None):
            if arm.invoked:
                # BSND already in place: the covered domain is aligned by gate
                assert s_real is None, f"covered domain must be aligned, got s_real={s_real}"
                assert out_bhsd.dim() == 4, f"expected BSND 4-D, got {tuple(out_bhsd.shape)}"
                arm.tout_calls.append(("SKIP", tuple(out_bhsd.shape)))
                arm.main_out = out_bhsd
                return out_bhsd
            arm.tout_calls.append(("RUN", tuple(out_bhsd.shape), s_real))
            arm.main_out = out_bhsd
            return orig_tout(out_bhsd, s_real=s_real)

        G.transpose_out = tout_cond

    def remove(self):
        if self._saved is not None:
            (G._gqa_kernel_prefill_g2stack_flat, G._gqa_kernel_prefill_g2stack_flat_kv2,
             G.transpose_out) = self._saved
            self._saved = None


def _run_arm(q, k, v, causal, arm=None, S=None, enable_variant=True):
    _clear()
    if arm is not None:
        arm.install(S, enable_variant=enable_variant)
    try:
        out = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
        torch.npu.synchronize()
        chosen = [val[2] for val in G._KERNEL_BUILD_CACHE.values()]
        main_out = arm.main_out if arm is not None else None
        calls = list(arm.tout_calls) if arm is not None else None
        inv = list(arm.invoked) if arm is not None else None
    finally:
        if arm is not None:
            arm.remove()
        _clear()
    return out, chosen, main_out, calls, inv


def _golden(q, k, v, causal, out, dtype):
    """Inline same-source golden in the test_gqa.py convention: ref_gqa on the
    NPU tensors IN THEIR INPUT DTYPE (test_gqa.py:287 'Golden in the input
    dtype on NPU — the f6-verified setup') + dtype-aware atol (1e-3 fp16 /
    2e-2 bf16) + rtol 1e-2, zero violations allowed."""
    ref = G._ref_gqa(q, k, v, -1.0, causal)
    torch.npu.synchronize()
    got = out.cpu().float()
    rf = ref.cpu().float()
    finite = torch.isfinite(rf) & torch.isfinite(got)
    diff = (got - rf).abs()
    atol = _atol(dtype)
    viol = int((diff[finite] > atol + RTOL * rf.abs()[finite]).sum())
    n_nan_got = int(torch.isnan(got).sum())
    n_nan_ref = int(torch.isnan(rf).sum())
    n_inf_got = int(torch.isinf(got).sum())
    return viol, n_nan_got, n_nan_ref, n_inf_got, (float(diff[finite].max()) if bool(finite.any()) else float("nan"))


def run_covered(case, layer2=True):
    label, B, S, S_kv, N_q, N_kv, D, causal, dtype, expect = case
    q, k, v = _inputs(label, B, S, S_kv, N_q, N_kv, D, dtype)
    special = label.startswith("boundary_sparse")
    fullrange = label.startswith("fullrange")

    arm_b = Arm()
    out_b, chosen_b, main_b, calls_b, inv_b = _run_arm(q, k, v, causal, arm=arm_b, S=S,
                                                       enable_variant=False)
    arm = Arm()
    out_a, chosen_a, main_a, calls_a, inv = _run_arm(q, k, v, causal, arm=arm, S=S)

    checks = {}
    checks["bit_identical"] = bool(torch.equal(_rawbit(out_b.cpu()), _rawbit(out_a.cpu())))
    viol_b, nan_b, nanref_b, inf_b, md_b = _golden(q, k, v, causal, out_b, dtype)
    viol_a, nan_a, nanref_a, inf_a, md_a = _golden(q, k, v, causal, out_a, dtype)
    if special:
        eq, why = _special_struct_equal(out_a.cpu().float(), out_b.cpu().float())
        checks["special_struct"] = eq
        # special values propagate unsanitized (module contract): the golden
        # carries the same NaN count and neither arm may differ from it
        checks["golden"] = (nan_a == nan_b == nanref_a == nanref_b) and viol_a == viol_b
        gmsg = (f"nan_arm={nan_a} nan_base={nan_b} nan_ref={nanref_a} inf_arm={inf_a} "
                f"viol_arm={viol_a} viol_base={viol_b} special={why}")
    elif fullrange:
        # +-65504 fp16 inputs: the fp16 golden itself saturates (128 terms of
        # ~65504^2 = 5.5e11 >> fp16 max => Inf => NaN in the softmax), so the
        # absolute tolerance check is not meaningful — measured nan_ref ==
        # EVERY element.  Criterion = the iter7 precedent ("fullrange fp16
        # +-65504 structural equality"): raw-bit identity between arms +
        # golden invariance (arm and base deviate from the saturated golden
        # identically).  Disclosed, not silently dropped.
        checks["golden_invariant"] = (viol_a == viol_b and nan_a == nan_b and inf_a == inf_b)
        gmsg = (f"[fullrange: golden SATURATED nan_ref={nanref_a} — absolute tol N/A] "
                f"viol_arm={viol_a} viol_base={viol_b} nan_arm={nan_a} nan_base={nan_b} "
                f"inf_arm={inf_a}")
    else:
        checks["golden"] = (viol_a == 0 and viol_b == 0 and nan_a == 0 and inf_a == 0)
        checks["golden_invariant"] = (viol_a == viol_b)
        gmsg = f"viol_arm={viol_a} viol_base={viol_b} nan={nan_a} max_diff={md_a:.3e}"
    checks["variant_invoked"] = bool(inv) and all(x.startswith("g2stack_flat") for x in inv)
    checks["chosen_ok"] = bool(chosen_a) and chosen_a[0] == expect and chosen_a == chosen_b
    checks["tout_skipped"] = any(c[0] == "SKIP" for c in calls_a) and not any(c[0] == "RUN" for c in calls_a)
    checks["tout_ran_base"] = any(c[0] == "RUN" for c in calls_b) and not inv_b
    checks["shape_ok"] = tuple(out_a.shape) == (B, S, N_q, D) and out_a.is_contiguous()
    if layer2:
        # kernel-level: ARM main (BSND) == transpose_out(BASE main (BHSD))
        t_b = G.transpose_out(main_b)
        checks["L2_kernel_bit"] = bool(torch.equal(_rawbit(t_b.cpu()), _rawbit(main_a.cpu())))
        checks["L2_layout"] = (tuple(main_b.shape) == (B, N_q, S, D)
                               and tuple(main_a.shape) == (B, S, N_q, D))
    ok = all(checks.values())
    bad = [k_ for k_, v_ in checks.items() if not v_]
    print(
        f"[{'G0_PASS' if ok else 'G0_FAIL'}] L1 {label} B={B} S={S} Skv={S_kv} Nq={N_q} Nkv={N_kv} "
        f"D={D} causal={causal} {str(dtype)[6:]} chosen={chosen_a} invoked={inv} "
        f"tout_arm={[c[0] for c in calls_a]} bit={checks['bit_identical']} {gmsg}"
        + (f" L2kernel={checks.get('L2_kernel_bit')}" if layer2 else "")
        + (f" FAILED={bad}" if bad else ""),
        flush=True,
    )
    return ok


def run_isolation(case):
    label, B, S, S_kv, N_q, N_kv, D, causal, dtype, expect, tout_exp = case
    q, k, v = _inputs(label, B, S, S_kv, N_q, N_kv, D, dtype)
    out_b, chosen_b, _, _, _ = _run_arm(q, k, v, causal)
    arm = Arm()
    out_a, chosen_a, main_a, calls_a, inv = _run_arm(q, k, v, causal, arm=arm, S=S)

    aligned = (S % 64 == 0)
    checks = {}
    checks["bit_identical"] = bool(torch.equal(_rawbit(out_b.cpu()), _rawbit(out_a.cpu())))
    checks["variant_not_invoked"] = (not inv) if not (aligned and expect.startswith("g2stack_flat")) else bool(inv)
    checks["chosen_identical"] = chosen_a == chosen_b and bool(chosen_a) and chosen_a[0] == expect
    if tout_exp:
        checks["tout_ran"] = any(c[0] == "RUN" for c in calls_a)
    else:
        checks["no_tout"] = len(calls_a) == 0
    viol, n_nan, _, n_inf, md = _golden(q, k, v, causal, out_a, dtype)
    checks["golden"] = viol == 0 and n_nan == 0 and n_inf == 0
    ok = all(checks.values())
    bad = [k_ for k_, v_ in checks.items() if not v_]
    print(
        f"[{'G0_PASS' if ok else 'G0_FAIL'}] L3 {label} B={B} S={S} Skv={S_kv} Nq={N_q} D={D} "
        f"causal={causal} chosen={chosen_a} invoked={inv} aligned={aligned} "
        f"tout={[c[0] for c in calls_a]} bit={checks['bit_identical']} viol={viol} "
        f"max_diff={md:.3e}" + (f" FAILED={bad}" if bad else ""),
        flush=True,
    )
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default=None, choices=["1", "2", "3"])
    ap.add_argument("--only", default=None, help="substring filter on the case label")
    args = ap.parse_args()
    results = []
    try:
        if args.layer in (None, "1", "2"):
            for c in COVERED:
                if args.only and args.only not in c[0]:
                    continue
                results.append(("L1:" + c[0], run_covered(c, layer2=(args.layer in (None, "2")))))
        if args.layer in (None, "3"):
            for c in OUT_OF_DOMAIN:
                if args.only and args.only not in c[0]:
                    continue
                results.append(("L3:" + c[0], run_isolation(c)))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        print(f"[G0_FAIL] error={type(e).__name__}: {e}", flush=True)
        results.append(("exception", False))
    n_ok = sum(1 for _, r in results if r)
    print(f"\n[G0_SUMMARY] P2: {n_ok}/{len(results)} checks passed", flush=True)
    for name, r in results:
        if not r:
            print(f"  FAILED: {name}", flush=True)
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
