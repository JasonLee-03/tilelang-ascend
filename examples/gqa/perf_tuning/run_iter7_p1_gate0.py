"""Gate 0 runner — iter7 P1 (transpose_out retile): bit-identity + plan audit.

Standalone: does NOT modify examples/gqa/gqa.py or test_gqa.py.  Three layers:

  1. KERNEL-LEVEL bit-identity: for each shape, build _transpose_v1 with the
     LEGACY plan and with the P1 retiled plan, run both on the same input and
     compare against each other AND a CPU torch reference.  The retile is a
     pure re-partition of the same byte set (every output element written
     exactly once) => bit-identical is the criterion (non-identical = BUG).
     Covers padded mode (n_pad > real S) and the plan-IDENTICAL shapes
     (regression guard: same plan must produce the same kernel result).
  2. E2E bit-identity through the PUBLIC gqa() API: plain vs G.out_plan
     monkeypatched to out_plan_p1 (both caches cleared; patch-invoked flag
     asserted).  Covers every dispatch domain that launches transpose_out
     (per-head, flat kv1, flat kv2, g2stack, D=256, padded-S) plus decode
     (transpose_out NOT invoked there — output must be untouched).
  3. PLAN AUDIT: predicted plans re-asserted in-process — starved shapes
     retile, healthy shapes (R4/c6/c8/c9/c10, L1 tails) stay BYTE-IDENTICAL
     to the legacy plan, D*2%32!=0 keeps returning None.

Run (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python3 examples/gqa/perf_tuning/run_iter7_p1_gate0.py [--layer 1|2|3]
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
from p1_retile_iter7 import legacy_plan, out_plan_p1  # noqa: E402

# (label, B, Nq, S(real rows), D, dtype, n_pad|None) — plan pairs from the
# offline predictor (p1_retile_iter7.py table, this session)
DIRECT_CASES = [
    ("r1_starved", 1, 8, 128, 128, torch.float16, None),        # (8,128)->(1,32)
    ("r2_like", 2, 32, 128, 128, torch.float16, None),          # (8,128)->(1,32)
    ("r3_like", 2, 128, 128, 128, torch.bfloat16, None),        # (16,128)->(1,128)
    ("r6_like", 2, 48, 128, 128, torch.float16, None),          # (16,128)->(1,64)
    ("r4_identical", 4, 64, 256, 128, torch.bfloat16, None),    # plan IDENTICAL guard
    ("d256_c5like", 2, 32, 128, 256, torch.float16, None),      # (8,128)->(1,32)
    ("padded_s100", 1, 8, 100, 128, torch.float16, 128),        # s_real=100: (8,4)->(1,4)
    ("padded_s124", 2, 32, 124, 128, torch.float16, 128),       # plan IDENTICAL (8,4)
    ("s24_small", 1, 8, 24, 128, torch.float16, None),          # (8,8)->(1,8)
    ("healthy_c8", 4, 32, 1024, 128, torch.float16, None),      # plan IDENTICAL (8,128)
]

# (label, B, S, S_kv, N_q, N_kv, D, causal, dtype) — public-API e2e
E2E_CASES = [
    ("r1_perhead", 1, 128, 128, 8, 2, 128, True, torch.float16),
    ("r2_flatkv1", 2, 128, 128, 32, 8, 128, True, torch.float16),
    ("r4_flatkv2", 4, 256, 256, 64, 8, 128, False, torch.bfloat16),
    ("r5_decode", 16, 1, 2048, 32, 8, 128, False, torch.float16),   # NO transpose_out
    ("r6_g2sflat", 2, 128, 128, 48, 4, 128, False, torch.float16),
    ("c7_cross", 2, 128, 256, 32, 8, 128, False, torch.bfloat16),
    ("c5_d256", 2, 128, 128, 32, 8, 256, True, torch.float16),
    ("padrow_s100", 2, 100, 256, 32, 8, 128, False, torch.float16),  # s_real mode
    ("healthy_c8", 4, 1024, 1024, 32, 8, 128, True, torch.float16),
]

# plan-identity audit shapes (B, Nq, S, D) — must return the legacy plan
IDENTICAL_PLANS = [
    (4, 64, 256, 128),    # R4/c4
    (2, 128, 256, 256),   # c6
    (4, 32, 1024, 128),   # c8
    (4, 64, 512, 128),    # c9
    (4, 128, 512, 128),   # c10
    (2, 32, 124, 128),    # l1_s_tail (s_real)
    (4, 64, 252, 128),    # l1_both_tail (s_real)
    (2, 32, 100, 128),    # suite padrow (s_real)
    (16, 1, 2048, 128),   # decode-ish rows (never called, still identical)
    (1, 8, 100, 128),     # suite_small — v2: launch-bound, legacy kept
    (1, 8, 24, 128),      # suite_s24 — v2: launch-bound, legacy kept
]
# retiled shapes (B, Nq, S, D) -> expected V2 plan (offline predictor output)
RETILED_PLANS = {
    (1, 8, 128, 128): (1, 32),
    (2, 32, 128, 128): (2, 128),
    (2, 128, 128, 128): (8, 128),
    (2, 48, 128, 128): (4, 128),
    (2, 32, 128, 256): (2, 128),
}


def _dtype_str(t):
    return "float16" if t == torch.float16 else "bfloat16"


def run_direct(case):
    label, B, Nq, S, D, dtype, n_pad = case
    leg = legacy_plan(B, Nq, S, D)
    p1 = out_plan_p1(B, Nq, S, D)
    assert leg is not None and p1 is not None
    rows = n_pad if n_pad is not None else S
    torch.manual_seed(1707)
    x4 = torch.rand(B, Nq, rows, D, dtype=torch.float32) * 2 - 1
    x = x4.to(dtype).reshape(B * Nq, rows, D).npu()
    ref = x4[:, :, :S, :].transpose(1, 2).reshape(B * S, Nq * D).to(dtype)

    outs = {}
    for tag, plan in (("legacy", leg), ("p1", p1)):
        bm, bn = plan[1], plan[2]
        fn = G._transpose_v1(B, Nq, S, D, bm, bn, _dtype_str(dtype), n_pad=n_pad)
        y = fn(x)
        torch.npu.synchronize()
        outs[tag] = y.reshape(B * S, Nq * D).cpu()
    bit_ab = bool(torch.equal(outs["legacy"], outs["p1"]))
    bit_ref_l = bool(torch.equal(outs["legacy"], ref))
    bit_ref_p = bool(torch.equal(outs["p1"], ref))
    changed = leg[1:] != p1[1:]
    ok = bit_ab and bit_ref_l and bit_ref_p
    print(
        f"[{'G0_PASS' if ok else 'G0_FAIL'}] direct {label} B={B} Nq={Nq} S={S} D={D} "
        f"n_pad={n_pad} legacy={leg[1:]} p1={p1[1:]} retiled={changed} "
        f"bit(legacy==p1)={bit_ab} bit(legacy==ref)={bit_ref_l} bit(p1==ref)={bit_ref_p}",
        flush=True,
    )
    return ok


def run_e2e(case):
    label, B, S, S_kv, N_q, N_kv, D, causal, dtype = case
    torch.manual_seed(42 + B + S + S_kv + N_q)
    q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
    k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
    v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()

    G._KERNEL_BUILD_CACHE.clear()
    G._COMPILED_CACHE.clear()
    out_base = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
    torch.npu.synchronize()
    chosen_base = [val[2] for val in G._KERNEL_BUILD_CACHE.values()]

    invoked = []
    orig_plan = G.out_plan

    def patched_plan(Bx, Nqx, Sx, Dx):
        r = out_plan_p1(Bx, Nqx, Sx, Dx)
        invoked.append(((Bx, Nqx, Sx, Dx), r))
        return r

    G.out_plan = patched_plan
    try:
        G._KERNEL_BUILD_CACHE.clear()
        G._COMPILED_CACHE.clear()
        out_arm = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
        torch.npu.synchronize()
        chosen_arm = [val[2] for val in G._KERNEL_BUILD_CACHE.values()]
    finally:
        G.out_plan = orig_plan
        G._KERNEL_BUILD_CACHE.clear()
        G._COMPILED_CACHE.clear()

    bit = bool(torch.equal(out_base, out_arm))
    decode = S <= 4
    if decode:
        plan_ok = (not invoked)  # transpose_out must NOT be reached on decode
        plans = "decode:no-out_plan-call"
    else:
        plan_ok = bool(invoked)
        plans = ";".join(f"{k_}->{v_}" for k_, v_ in invoked)
    ok = bit and plan_ok and chosen_base == chosen_arm
    print(
        f"[{'G0_PASS' if ok else 'G0_FAIL'}] e2e {label} B={B} S={S} Skv={S_kv} Nq={N_q} "
        f"D={D} causal={causal} chosen={chosen_arm} out_plan_invoked={len(invoked)} "
        f"bit_identical={bit} plans[{plans}]",
        flush=True,
    )
    return ok


def run_plans():
    ok = True
    for shape in IDENTICAL_PLANS:
        leg, p1 = legacy_plan(*shape), out_plan_p1(*shape)
        same = leg == p1
        ok &= same
        print(f"[{'G0_PASS' if same else 'G0_FAIL'}] plan-identity {shape} legacy={leg} p1={p1}", flush=True)
    for shape, want in RETILED_PLANS.items():
        leg, p1 = legacy_plan(*shape), out_plan_p1(*shape)
        good = p1 == ("custom",) + want and leg != p1
        ok &= good
        print(f"[{'G0_PASS' if good else 'G0_FAIL'}] plan-retile {shape} legacy={leg[1:]} "
              f"p1={p1[1:]} expected={want}", flush=True)
    none_ok = out_plan_p1(2, 32, 128, 88) is None and legacy_plan(2, 32, 128, 88) is None
    print(f"[{'G0_PASS' if none_ok else 'G0_FAIL'}] plan-None D=88 unchanged", flush=True)
    return ok and none_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default=None, choices=["1", "2", "3"])
    args = ap.parse_args()
    results = []
    try:
        if args.layer in (None, "3"):
            results.append(run_plans())
        if args.layer in (None, "1"):
            for c in DIRECT_CASES:
                results.append(run_direct(c))
        if args.layer in (None, "2"):
            for c in E2E_CASES:
                results.append(run_e2e(c))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        print(f"[G0_FAIL] error={e}", flush=True)
        results.append(False)
    n_ok = sum(results)
    print(f"\n[G0_SUMMARY] P1: {n_ok}/{len(results)} checks passed", flush=True)
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
