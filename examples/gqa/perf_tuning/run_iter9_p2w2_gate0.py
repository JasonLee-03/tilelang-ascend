"""Gate 0 runner — iter9 P2 wave-2 (prefill_expert + g2stack BSND direct write).

Standalone: does NOT modify examples/gqa/gqa.py or test_gqa.py.  Both arms run
through the PUBLIC gqa() API in ONE process, caches cleared per arm (iter5
finding 2).  SERIAL execution only (task #5 — Fixed-Core-20 expert kernels).

  BASE = shipped chain: main kernel writes BHSD [B,Nq,S_pad,D] -> transpose_out
         (_transpose_v1, P1-retiled plan) -> [B,S,Nq,D]
  ARM  = wave-2 variant chain: `_gqa_kernel_prefill_expert` /
         `_gqa_kernel_prefill_g2stack` replaced by
         perf_tuning/gqa_w2_bsnd_iter9_variant.py (2 line-scoped edits each,
         audited by gen_variant_iter9_p2w2.py) -> transpose_out SKIPPED
         (conditional pass-through == the integration's capability flag).
         Host gate: S % 64 == 0 (aligned => S_padded == S => no crop).
         THE PER-HEAD FACTORY IS NEVER PATCHED (domain excluded — AUTO
         threads=2 strided-write mis-lowering, bit_probe_iter9.log); its
         shapes appear below as out-of-domain isolation cases.

Criteria (PREREGISTERED in candidate_iter9_P2w2.json — bit-identity, NOT a
relaxed tolerance: pure data-path re-partition, zero arithmetic change):

  L1 covered  : ARM final output BIT-IDENTICAL to BASE (raw-bit view(int16),
                iter8 finding 3) + golden viol_arm == viol_base == 0 (f6
                policy; boundary_sparse: NaN/Inf positions+signs+finite bits
                equal; fullrange: golden-invariance criterion, iter7/8
                precedent) + variant INVOKED + chosen == expected +
                transpose_out SKIPPED in ARM / RAN in BASE
  L2 kernel   : transpose_out(BASE main BHSD) == ARM main BSND raw-bit
  L3 isolation: variant NOT invoked, chosen identical, tout RAN (or ABSENT
                for wave-1/decode), output BIT-IDENTICAL — incl. per-head
                (excluded domain), wave-1 flat (shipped BSND), decode, and
                the non-aligned-S gate cases of BOTH covered domains

Run (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python3 examples/gqa/perf_tuning/run_iter9_p2w2_gate0.py [--layer 1|2|3] [--only SUBSTR]
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
import gqa_w2_bsnd_iter9_variant as VAR  # noqa: E402

RTOL = 1e-2
W2_TARGETS = ("_gqa_kernel_prefill_expert", "_gqa_kernel_prefill_g2stack")


def _atol(dtype):
    """test_gqa.py _compare_regular policy (f6-verified): fp16 1e-3 / bf16 2e-2."""
    return 1e-3 if dtype == torch.float16 else 2e-2


def _rawbit(t):
    """NaN-safe bit-level view (iter8 finding 3)."""
    return t.view(torch.int16)


# --------------------------------------------------------------------------
# L1/L2 — covered domains, aligned S
# (label, B, S, S_kv, N_q, N_kv, D, causal, dtype, expected_chosen)
COVERED = [
    # ---- prefill_expert domain ----
    ("c19_mha_g1_ns1", 2, 128, 128, 32, 32, 128, True, torch.bfloat16, "prefill_expert"),
    ("prexpert_g3_kv2_noncausal", 2, 128, 256, 30, 10, 128, False, torch.float16, "prefill_expert"),
    ("prexpert_g3_kv4_causal_varlen", 2, 256, 512, 30, 10, 128, True, torch.bfloat16, "prefill_expert"),
    ("prexpert_padcol_skv200", 2, 128, 200, 30, 10, 128, False, torch.float16, "prefill_expert"),
    ("boundary_sparse_prexpert_bf16", 2, 128, 128, 32, 32, 128, True, torch.bfloat16, "prefill_expert"),
    ("fullrange_prexpert_g3_fp16", 2, 128, 128, 30, 10, 128, True, torch.float16, "prefill_expert"),
    # ---- g2stack domain ----
    ("c9_like_kv4_noncausal", 4, 512, 512, 64, 8, 128, False, torch.bfloat16, "g2stack"),
    ("c8_like_kv8_causal_varlen", 4, 1024, 1024, 32, 8, 128, True, torch.float16, "g2stack"),
    ("c10_like_kv4_causal_varlen", 4, 512, 512, 128, 8, 128, True, torch.bfloat16, "g2stack"),
    ("g2stack_kv2_causal_varlen", 2, 128, 256, 32, 8, 128, True, torch.float16, "g2stack"),
    ("g2stack_ns1_stacked20", 1, 128, 128, 20, 2, 128, True, torch.float16, "g2stack"),
    ("g2stack_padcol_skv200_causal", 4, 128, 200, 64, 8, 128, True, torch.float16, "g2stack"),
    ("boundary_sparse_g2stack_kv4causal", 2, 256, 512, 32, 8, 128, True, torch.float16, "g2stack"),
    ("fullrange_g2stack_kv4causal", 2, 256, 512, 32, 8, 128, True, torch.float16, "g2stack"),
]

# --------------------------------------------------------------------------
# L3 — out-of-domain isolation (must be BIT-IDENTICAL and untouched)
# (label, B, S, S_kv, Nq, Nkv, D, causal, dtype, expected_chosen, tout_expected)
# tout_expected: "run" | "absent"
OUT_OF_DOMAIN = [
    # per-head: the EXCLUDED domain — factory never patched, tout must run
    ("r1_perhead_bm64", 1, 128, 128, 8, 2, 128, True, torch.float16, "per_head_bm64", "run"),
    ("c5_d256_perhead", 2, 128, 128, 32, 8, 256, True, torch.float16, "per_head_bm64", "run"),
    ("odd_kv3_perhead", 2, 128, 300, 32, 8, 128, False, torch.float16, "per_head_bm64", "run"),
    # wave-1 shipped BSND domains — tout absent BOTH arms, patch must not disturb
    ("r2_flatkv1_wave1", 2, 128, 128, 32, 8, 128, True, torch.float16, "g2stack_flat", "absent"),
    ("r4_flatkv2_wave1", 4, 256, 256, 64, 8, 128, False, torch.bfloat16, "g2stack_flat_kv2", "absent"),
    # decode — BSND direct, no tout
    ("r5_decode_expert", 16, 1, 2048, 32, 8, 128, False, torch.float16, "dense_expert", "absent"),
    ("decode_auto_small", 2, 2, 512, 8, 2, 128, False, torch.float16, "dense_auto", "absent"),
    # GATE ISOLATION: covered-domain kernels chosen BUT S non-aligned =>
    # the wave-2 host gate must exclude them (tout(s_real) keeps running)
    ("s100e_prexpert_unaligned", 2, 100, 128, 32, 32, 128, True, torch.bfloat16, "prefill_expert", "run"),
    ("s100g_g2stack_unaligned", 4, 100, 512, 64, 8, 128, False, torch.bfloat16, "g2stack", "run"),
    ("s100p_perhead_unaligned", 1, 100, 100, 8, 2, 128, True, torch.float16, "per_head_bm64", "run"),
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
        # iter5 verbatim injection set (multi-kv-block shapes):
        #   q NaN -> a whole output row NaN;  v +Inf (early kv block) /
        #   -Inf (later kv block) -> whole head-group columns Inf with signs
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
    """Installs/removes the wave-2 monkeypatches; records dispatch evidence.
    The per-head factory is NEVER patched (excluded domain)."""

    def __init__(self):
        self.invoked = []
        self.tout_calls = []
        self.main_out = None
        self._saved = None

    def install(self, S, enable_variant=True):
        self.invoked = []
        self.tout_calls = []
        self.main_out = None
        aligned = (S % 64 == 0) and enable_variant
        self._saved = tuple(getattr(G, n) for n in W2_TARGETS) + (G.transpose_out,)
        self.aligned = aligned
        if aligned:
            for name in W2_TARGETS:
                var_fn = getattr(VAR, name)

                def wrap(*a, _n=name, _f=var_fn, **kw):
                    self.invoked.append(_n)
                    return _f(*a, **kw)

                setattr(G, name, wrap)

        orig_tout = self._saved[-1]
        arm = self

        def tout_cond(out_bhsd, s_real=None):
            if arm.invoked:
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
            for name, orig in zip(W2_TARGETS, self._saved[:-1]):
                setattr(G, name, orig)
            G.transpose_out = self._saved[-1]
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
    """test_gqa.py convention: ref_gqa on the NPU tensors IN THEIR INPUT DTYPE
    + dtype-aware atol + rtol 1e-2 (f6-verified policy, iter8 finding 4)."""
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
    return (viol, n_nan_got, n_nan_ref, n_inf_got,
            (float(diff[finite].max()) if bool(finite.any()) else float("nan")))


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
        checks["golden"] = (nan_a == nan_b == nanref_a == nanref_b) and viol_a == viol_b
        gmsg = (f"nan_arm={nan_a} nan_base={nan_b} nan_ref={nanref_a} inf_arm={inf_a} "
                f"viol_arm={viol_a} viol_base={viol_b} special={why}")
    elif fullrange:
        # +-65504 fp16: the golden itself saturates (iter8 precedent) —
        # criterion = raw-bit identity between arms + golden invariance.
        checks["golden_invariant"] = (viol_a == viol_b and nan_a == nan_b and inf_a == inf_b)
        gmsg = (f"[fullrange: golden SATURATED nan_ref={nanref_a} — absolute tol N/A] "
                f"viol_arm={viol_a} viol_base={viol_b} nan_arm={nan_a} nan_base={nan_b} "
                f"inf_arm={inf_a}")
    else:
        checks["golden"] = (viol_a == 0 and viol_b == 0 and nan_a == 0 and inf_a == 0)
        checks["golden_invariant"] = (viol_a == viol_b)
        gmsg = f"viol_arm={viol_a} viol_base={viol_b} nan={nan_a} max_diff={md_a:.3e}"
    checks["variant_invoked"] = bool(inv) and all(x in W2_TARGETS for x in inv)
    checks["chosen_ok"] = bool(chosen_a) and chosen_a[0] == expect and chosen_a == chosen_b
    checks["tout_skipped"] = any(c[0] == "SKIP" for c in calls_a) and not any(c[0] == "RUN" for c in calls_a)
    checks["tout_ran_base"] = any(c[0] == "RUN" for c in calls_b) and not inv_b
    checks["shape_ok"] = tuple(out_a.shape) == (B, S, N_q, D) and out_a.is_contiguous()
    if layer2:
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
    out_b, chosen_b, _, calls_b, inv_b = _run_arm(q, k, v, causal, arm=Arm(), S=S,
                                                  enable_variant=False)
    arm = Arm()
    out_a, chosen_a, main_a, calls_a, inv = _run_arm(q, k, v, causal, arm=arm, S=S)

    checks = {}
    checks["bit_identical"] = bool(torch.equal(_rawbit(out_b.cpu()), _rawbit(out_a.cpu())))
    checks["variant_not_invoked"] = not inv and not inv_b
    checks["chosen_identical"] = chosen_a == chosen_b and bool(chosen_a) and chosen_a[0] == expect
    if tout_exp == "run":
        checks["tout_ran"] = any(c[0] == "RUN" for c in calls_a)
    else:
        checks["no_tout"] = len(calls_a) == 0 and len(calls_b) == 0
    viol, n_nan, _, n_inf, md = _golden(q, k, v, causal, out_a, dtype)
    checks["golden"] = viol == 0 and n_nan == 0 and n_inf == 0
    ok = all(checks.values())
    bad = [k_ for k_, v_ in checks.items() if not v_]
    print(
        f"[{'G0_PASS' if ok else 'G0_FAIL'}] L3 {label} B={B} S={S} Skv={S_kv} Nq={N_q} D={D} "
        f"causal={causal} chosen={chosen_a} invoked={inv} aligned={S % 64 == 0} "
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
    print(f"\n[G0_SUMMARY] P2-wave2: {n_ok}/{len(results)} checks passed", flush=True)
    for name, r in results:
        if not r:
            print(f"  FAILED: {name}", flush=True)
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
