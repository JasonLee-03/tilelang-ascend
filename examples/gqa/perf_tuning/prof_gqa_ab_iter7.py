"""Iter7 multi-arm same-session A/B probe: base / p1 / p5.

Single-shot run for msprof collection (mirrors prof_gqa_ab_iter4/5.py:
EXACTLY ONE gqa() call = the full kernel chain, no warm-up, no golden
compare inside the profiled process).

Arms:
  base : shipped gqa.py as-is (md5 bd470e072d0a48faeac1e786306932e8), no
         patches — asserts the resolved dispatch chain per shape.
  p1   : host-only out_plan patch (p1_retile_iter7.out_plan_p1) — retiles
         the transpose_out task grid on STARVED shapes only (healthy grids
         return the byte-identical legacy plan).  Asserts (a) the patch was
         invoked on prefill / NEVER invoked on decode (r5, no
         transpose_out), and (b) the resolved plan matches the offline
         predictor (retiled on r1/r2/r3/r6, IDENTICAL on r4) — a silent
         no-op or an unexpected r4 change invalidates the sample.  The
         kernel body (_transpose_v1) is untouched; main kernel untouched.
  p5   : kernel-variant patch (gqa_flat_kv2_p5_variant, generated from the
         shipped flat_kv2 block — neg_sm cap-refill narrowed to the prv
         slots).  Asserts the variant factory was invoked and chosen ==
         ["g2stack_flat_kv2"] on r4/c7 (never invoked elsewhere).

BOTH dispatch caches (_KERNEL_BUILD_CACHE + adapter _COMPILED_CACHE) are
cleared so patches are guaranteed to take effect.  gqa.py / test_gqa.py are
NOT modified by anything in this iteration (Gate 3 discipline).

Usage (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:$LD_LIBRARY_PATH
    # P1 B-matrix (all-kernel chain):
    msprof op --launch-count=3 --aic-metrics=PipeUtilization,Memory,L2Cache \
        --output=<dir> python3 examples/gqa/perf_tuning/prof_gqa_ab_iter7.py \
        --arm p1 --shape r1
    # P5 main-kernel matrix:
    msprof op --kernel-name=main_kernel --aic-metrics=PipeUtilization \
        --output=<dir> python3 examples/gqa/perf_tuning/prof_gqa_ab_iter7.py \
        --arm p5 --shape r4
"""

import argparse
import os
import sys

import torch
import torch_npu  # noqa: F401  (registers the npu backend)

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_DIR = os.path.dirname(HERE)
sys.path.insert(0, GQA_DIR)
sys.path.insert(0, HERE)

import gqa as G  # noqa: E402

SHAPES = {
    # name: (B, S, S_kv, N_q, N_kv, D, causal, dtype, expected_domain)
    "r1": (1, 128, 128, 8, 2, 128, True, "fp16", "per_head"),
    "r2": (2, 128, 128, 32, 8, 128, True, "fp16", "g2stack_flat"),
    "r3": (2, 128, 128, 128, 8, 128, True, "bf16", "g2stack_flat"),
    "r4": (4, 256, 256, 64, 8, 128, False, "bf16", "g2stack_flat_kv2"),
    "r5": (16, 1, 2048, 32, 8, 128, False, "fp16", "dense_expert"),
    "r6": (2, 128, 128, 48, 4, 128, False, "fp16", "g2stack_flat"),
    "c7": (2, 128, 256, 32, 8, 128, False, "bf16", "g2stack_flat_kv2"),
}

# offline predictor results (p1_retile_iter7.py V2 table — pinned here so a
# silent rule change invalidates the sample instead of being mis-measured)
P1_EXPECT_PLAN = {
    # shape: (legacy (bm,bn), p1-V2 (bm,bn)) on the REAL output rows
    "r1": ((8, 128), (1, 32)),
    "r2": ((8, 128), (2, 128)),
    "r3": ((16, 128), (8, 128)),
    "r4": ((16, 128), (16, 128)),   # IDENTICAL — hysteresis keeps the adequate grid
    "r6": ((16, 128), (4, 128)),
    "c7": ((8, 128), (2, 128)),
}


def main():
    ap = argparse.ArgumentParser(description="Single-shot GQA multi-arm A/B run for msprof (iter7)")
    ap.add_argument("--arm", required=True, choices=["base", "p1", "p5"])
    ap.add_argument("--shape", required=True, choices=list(SHAPES))
    args = ap.parse_args()

    B, S, S_kv, N_q, N_kv, D, causal, dtag, expect = SHAPES[args.shape]
    dtype = torch.float16 if dtag == "fp16" else torch.bfloat16
    decode = S <= 4

    invoked = []
    plans = []

    if args.arm == "p1":
        from p1_retile_iter7 import legacy_plan, out_plan_p1

        orig_plan = G.out_plan

        def patched(Bx, Nqx, Sx, Dx):
            r = out_plan_p1(Bx, Nqx, Sx, Dx)
            invoked.append((Bx, Nqx, Sx, Dx))
            plans.append(r[1:] if r else None)
            return r

        G.out_plan = patched
        # pinned plan audit (host-side, before the run)
        leg = legacy_plan(B, N_q, S, D)
        p1p = out_plan_p1(B, N_q, S, D)
        if not decode:
            exp_leg, exp_p1 = P1_EXPECT_PLAN[args.shape]
            assert leg[1:] == exp_leg, f"legacy plan {leg} != pinned {exp_leg}"
            assert p1p[1:] == exp_p1, f"p1 plan {p1p} != pinned {exp_p1}"
        print(f"[P1_PLAN] shape={args.shape} legacy={leg[1:] if leg else None} "
              f"p1={p1p[1:] if p1p else None}", flush=True)

    if args.arm == "p5":
        assert args.shape in ("r4", "c7"), "p5 arm is judged on r4/c7 only"
        import gqa_flat_kv2_p5_variant

        variant_factory = gqa_flat_kv2_p5_variant._gqa_kernel_prefill_g2stack_flat_kv2
        orig = G._gqa_kernel_prefill_g2stack_flat_kv2

        def patched_kv2(B, S_padded, S_kv_padded, sm_scale, is_causal, N_q=8, N_kv=2, D=128,
                        dtype="float16", block_M=64, skv_minus_s=0, use_mask=True, block_N=128):
            invoked.append(True)
            return variant_factory(B, S_padded, S_kv_padded, sm_scale, is_causal,
                                   N_q=N_q, N_kv=N_kv, D=D, dtype=dtype, block_M=block_M,
                                   skv_minus_s=skv_minus_s, use_mask=use_mask, block_N=block_N)

        G._gqa_kernel_prefill_g2stack_flat_kv2 = patched_kv2

    G._KERNEL_BUILD_CACHE.clear()
    G._COMPILED_CACHE.clear()

    torch.manual_seed(42)
    q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
    k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
    v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()

    print(
        f"[PROF_RUN] arm={args.arm} shape={args.shape} B={B} S={S} S_kv={S_kv} N_q={N_q} "
        f"N_kv={N_kv} D={D} causal={causal} dtype={dtag}",
        flush=True,
    )

    # The single profiled launch chain: transpose_qkv -> main -> transpose_out
    # (prefill) or main only (decode, BSND direct).
    out = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
    torch.npu.synchronize()

    chosen = [val[2] for val in G._KERNEL_BUILD_CACHE.values()]
    if args.arm == "p1":
        if decode:
            assert not invoked, "[AB-PROBE] p1+decode: out_plan invoked — impossible path"
        else:
            assert invoked, "[AB-PROBE] p1: out_plan patch NEVER invoked"
    if args.arm == "p5":
        assert invoked, "[AB-PROBE] p5: variant factory NEVER invoked"
    if args.arm == "base":
        assert not invoked, "[AB-PROBE] base: contamination"
    if not chosen or not chosen[0].startswith(expect):
        # hard assert would kill the msprof export on mismatch — warn first,
        # then still fail: an unattributable sample must never be measured
        print(f"[WARN] dispatch mismatch: chosen={chosen} expected~={expect}", flush=True)
    assert chosen and chosen[0].startswith(expect), \
        f"[AB-PROBE] arm={args.arm}: dispatch resolved to {chosen}, expected ~{expect} — INVALID"
    print(
        f"[PROF_DONE] arm={args.arm} shape={args.shape} chosen={chosen} "
        f"patch_invoked={len(invoked)} plans={plans} out.shape={tuple(out.shape)} "
        f"out.dtype={out.dtype}",
        flush=True,
    )


if __name__ == "__main__":
    main()
