"""Iter8 same-session A/B probe: base (shipped chain) vs p2 (BSND direct write).

Single-shot run for msprof collection (mirrors prof_gqa_ab_iter4/5/7.py:
EXACTLY ONE gqa() call = the full kernel chain, no warm-up, no golden compare
inside the profiled process).

Arms:
  base : shipped gqa.py as-is (md5 4b4e453e4b8f2922baefbecb85912494), no
         patches — main kernel writes BHSD, transpose_out runs (P1-retiled
         plan).  Asserts the resolved dispatch chain per shape.
  p2   : P2 variant chain — the two flat kernels are replaced by
         perf_tuning/gqa_flat_bsnd_iter8_variant.py (2 audited line-edits
         each: Output declared BSND + FIN box strided) and transpose_out is
         conditionally SKIPPED when a BSND kernel produced the output (the
         monkeypatch equivalent of the integration's
         `if getattr(kernel_fn,'out_bsnd',False): return out`).
         Wave-1 host gate applied here exactly as the integration will:
         `S % 64 == 0` (aligned => S_padded == S => no row crop needed).
         Asserts (a) the variant factory WAS invoked on covered shapes and
         NEVER on out-of-domain shapes (r1 per-head, r5 decode), (b) chosen
         spec unchanged vs the expected domain, (c) transpose_out actually
         skipped on covered shapes / actually run elsewhere.

BOTH dispatch caches (_KERNEL_BUILD_CACHE + adapter _COMPILED_CACHE) are
cleared so patches are guaranteed to take effect (iter5 finding 2).
gqa.py / test_gqa.py are NOT modified by anything in this iteration until
Gate 3 (discipline #5).

Usage (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:$LD_LIBRARY_PATH
    msprof op --launch-count=3 --aic-metrics=PipeUtilization,Memory,L2Cache \
        --output=<dir> python3 examples/gqa/perf_tuning/prof_gqa_ab_iter8.py \
        --arm p2 --shape r2
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
    # name: (B, S, S_kv, N_q, N_kv, D, causal, dtype, expected_domain, p2_covered)
    "r1": (1, 128, 128, 8, 2, 128, True, "fp16", "per_head", False),
    "r2": (2, 128, 128, 32, 8, 128, True, "fp16", "g2stack_flat", True),
    "r3": (2, 128, 128, 128, 8, 128, True, "bf16", "g2stack_flat", True),
    "r4": (4, 256, 256, 64, 8, 128, False, "bf16", "g2stack_flat_kv2", True),
    "r5": (16, 1, 2048, 32, 8, 128, False, "fp16", "dense_expert", False),
    "r6": (2, 128, 128, 48, 4, 128, False, "fp16", "g2stack_flat", True),
    "c7": (2, 128, 256, 32, 8, 128, False, "bf16", "g2stack_flat_kv2", True),
}


def main():
    ap = argparse.ArgumentParser(description="Single-shot GQA A/B run for msprof (iter8 P2)")
    ap.add_argument("--arm", required=True, choices=["base", "p2"])
    ap.add_argument("--shape", required=True, choices=list(SHAPES))
    args = ap.parse_args()

    B, S, S_kv, N_q, N_kv, D, causal, dtag, expect, covered = SHAPES[args.shape]
    dtype = torch.float16 if dtag == "fp16" else torch.bfloat16
    decode = S <= 4
    aligned = (S % 64 == 0)

    invoked = []
    tout_calls = []
    state = {"bsnd": False}

    if args.arm == "p2":
        import gqa_flat_bsnd_iter8_variant as VAR

        orig_kv1 = G._gqa_kernel_prefill_g2stack_flat
        orig_kv2 = G._gqa_kernel_prefill_g2stack_flat_kv2
        orig_tout = G.transpose_out

        # wave-1 host gate: only aligned-S shapes get the BSND kernel
        if aligned:
            def wrap_kv1(*a, **kw):
                invoked.append("g2stack_flat")
                return VAR._gqa_kernel_prefill_g2stack_flat(*a, **kw)

            def wrap_kv2(*a, **kw):
                invoked.append("g2stack_flat_kv2")
                return VAR._gqa_kernel_prefill_g2stack_flat_kv2(*a, **kw)

            G._gqa_kernel_prefill_g2stack_flat = wrap_kv1
            G._gqa_kernel_prefill_g2stack_flat_kv2 = wrap_kv2

        def tout_cond(out_bhsd, s_real=None):
            if invoked:
                state["bsnd"] = True
                assert s_real is None, f"[AB-PROBE] covered domain must be aligned, s_real={s_real}"
                tout_calls.append(("SKIP", tuple(out_bhsd.shape)))
                return out_bhsd
            tout_calls.append(("RUN", tuple(out_bhsd.shape)))
            return orig_tout(out_bhsd, s_real=s_real)

        G.transpose_out = tout_cond

    G._KERNEL_BUILD_CACHE.clear()
    G._COMPILED_CACHE.clear()

    torch.manual_seed(42)
    q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
    k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
    v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()

    print(
        f"[PROF_RUN] arm={args.arm} shape={args.shape} B={B} S={S} S_kv={S_kv} N_q={N_q} "
        f"N_kv={N_kv} D={D} causal={causal} dtype={dtag} covered={covered} aligned={aligned}",
        flush=True,
    )

    out = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
    torch.npu.synchronize()

    chosen = [val[2] for val in G._KERNEL_BUILD_CACHE.values()]

    # ---- probe assertions: an unattributable sample must never be measured ----
    if args.arm == "p2":
        if covered:
            assert invoked, f"[AB-PROBE] p2 covered shape {args.shape}: variant NEVER invoked"
            assert state["bsnd"], f"[AB-PROBE] p2 {args.shape}: transpose_out was NOT skipped"
            assert all(c[0] == "SKIP" for c in tout_calls), f"[AB-PROBE] tout calls {tout_calls}"
        else:
            assert not invoked, (
                f"[AB-PROBE] p2 OUT-OF-DOMAIN shape {args.shape}: variant invoked ({invoked}) "
                "— the wave-1 gate leaked")
            if not decode:
                assert any(c[0] == "RUN" for c in tout_calls), f"[AB-PROBE] tout not run: {tout_calls}"
            else:
                assert not tout_calls, f"[AB-PROBE] decode must not call transpose_out: {tout_calls}"
    else:
        assert not invoked, "[AB-PROBE] base arm contaminated by the variant"
    if not chosen or not chosen[0].startswith(expect):
        print(f"[WARN] dispatch mismatch: chosen={chosen} expected~={expect}", flush=True)
    assert chosen and chosen[0].startswith(expect), \
        f"[AB-PROBE] arm={args.arm}: dispatch resolved to {chosen}, expected ~{expect} — INVALID"
    assert tuple(out.shape) == (B, S, N_q, D), f"[AB-PROBE] out shape {tuple(out.shape)}"
    assert out.is_contiguous(), "[AB-PROBE] out not contiguous"

    print(
        f"[PROF_DONE] arm={args.arm} shape={args.shape} chosen={chosen} "
        f"variant_invoked={invoked} tout={[c[0] for c in tout_calls]} "
        f"out.shape={tuple(out.shape)} out.dtype={out.dtype}",
        flush=True,
    )


if __name__ == "__main__":
    main()
