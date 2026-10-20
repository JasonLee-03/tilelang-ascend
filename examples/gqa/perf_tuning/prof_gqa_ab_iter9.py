"""Iter9 same-session A/B probe: base (shipped chain) vs w2 (P2 wave-2 BSND
direct write on the prefill_expert + g2stack transpose_out domains).

Single-shot run for msprof collection (mirrors prof_gqa_ab_iter8.py:
EXACTLY ONE gqa() call = the full kernel chain, no warm-up, no golden compare
inside the profiled process).

Arms:
  base : shipped gqa.py as-is (md5 3aa9cfa2625cdce96a62ce967785fe85, iter8
         adopted), no patches.  Wave-1 domains already write BSND (shipped);
         wave-2 domains (prefill_expert / g2stack) write BHSD and
         transpose_out runs (P1-retiled plan).  Asserts the resolved dispatch.
  w2   : P2 wave-2 variant chain — the TWO expert-mode target factories are
         replaced by perf_tuning/gqa_w2_bsnd_iter9_variant.py (2 audited
         line-edits each: Output declared BSND + FIN box strided) and
         transpose_out is conditionally SKIPPED when a variant produced the
         output (the monkeypatch equivalent of the integration's capability
         flag).  The per-head AUTO kernel `_gqa_kernel_expert` is EXCLUDED
         from wave-2 (threads=2 strided-write mis-lowering — see the variant
         module header / bit_probe_iter9.log), so per-head shapes (r1) are
         untouched guards here.  Host gate applied exactly as the integration
         will: `S % 64 == 0` (aligned => S_padded == S => no row crop).
         Asserts (a) the variant factory WAS invoked on covered shapes and
         NEVER on out-of-domain shapes, (b) chosen spec unchanged vs the
         expected domain, (c) transpose_out actually skipped on covered
         shapes / actually run elsewhere.

BOTH dispatch caches (_KERNEL_BUILD_CACHE + adapter _COMPILED_CACHE) are
cleared so patches are guaranteed to take effect (iter5 finding 2).
gqa.py / test_gqa.py are NOT modified by anything in this iteration until
Gate 3 (discipline #6).

SERIAL-ONLY discipline (task-book #5): every kernel family here except the
per-head AUTO ladder is blockDim=20 Fixed-Core with cross-core semaphores —
two such kernels co-resident on the 20-core chip can deadlock (eval attempt-1
case15 aicore timeout precedent).  Never run two collections concurrently.

Usage (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:$LD_LIBRARY_PATH
    msprof op --launch-count=3 --aic-metrics=PipeUtilization,Memory,L2Cache \
        --output=<dir> python3 examples/gqa/perf_tuning/prof_gqa_ab_iter9.py \
        --arm base --shape r1
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
    # name: (B, S, S_kv, N_q, N_kv, D, causal, dtype, expected_chosen_prefix, role)
    # role: w2        = wave-2 covered (variant must be invoked, tout SKIPPED)
    #       w1        = wave-1 shipped guard (variant NOT invoked; tout ABSENT —
    #                   the shipped flat kernel already writes BSND)
    #       decode    = no tout in either arm
    #       unaligned = non-aligned S guard (variant NOT invoked; tout RUNS both arms)
    #       perhead_untouched = aligned per-head guard — the per-head AUTO
    #                   kernel is EXCLUDED from wave-2 (threads=2 strided-write
    #                   mis-lowering, bit_probe_iter9.log), so its factory is
    #                   never patched: tout RUNS in both arms, arms identical
    # ---- wave-2 target domains (expert manual-flag kernels only) ----
    "c19": (2, 128, 128, 32, 32, 128, True, "bf16", "prefill_expert", "w2"),
    "c8": (4, 1024, 1024, 32, 8, 128, True, "fp16", "g2stack", "w2"),
    "c9": (4, 512, 512, 64, 8, 128, False, "bf16", "g2stack", "w2"),
    "c10": (4, 512, 512, 128, 8, 128, True, "bf16", "g2stack", "w2"),
    # ---- out-of-domain guards ----
    "r1": (1, 128, 128, 8, 2, 128, True, "fp16", "per_head_bm", "perhead_untouched"),
    "r2": (2, 128, 128, 32, 8, 128, True, "fp16", "g2stack_flat", "w1"),
    "r5": (16, 1, 2048, 32, 8, 128, False, "fp16", "dense_expert", "decode"),
    "s100e": (2, 100, 128, 32, 32, 128, True, "bf16", "prefill_expert", "unaligned"),
    "s100g": (4, 100, 512, 64, 8, 128, False, "bf16", "g2stack", "unaligned"),
    "s100p": (1, 100, 100, 8, 2, 128, True, "fp16", "per_head_bm", "unaligned"),
}

# the wave-2 target factories (module attr name).  The per-head AUTO kernel
# `_gqa_kernel_expert` is DELIBERATELY ABSENT — excluded by the Phase-B
# verdict (AUTO threads=2 strided-write mis-lowering; see the variant module
# header + bit_probe_iter9.log).
W2_TARGETS = (
    "_gqa_kernel_prefill_expert",
    "_gqa_kernel_prefill_g2stack",
)


def main():
    ap = argparse.ArgumentParser(description="Single-shot GQA A/B run for msprof (iter9 P2 wave-2)")
    ap.add_argument("--arm", required=True, choices=["base", "w2"])
    ap.add_argument("--shape", required=True, choices=list(SHAPES))
    args = ap.parse_args()

    B, S, S_kv, N_q, N_kv, D, causal, dtag, expect, role = SHAPES[args.shape]
    covered = role == "w2"
    dtype = torch.float16 if dtag == "fp16" else torch.bfloat16
    decode = S <= 4
    aligned = (S % 64 == 0)
    if role == "unaligned":
        assert not aligned, f"[AB-PROBE] role=unaligned but S={S} is aligned"
    if covered:
        assert aligned, f"[AB-PROBE] role=w2 requires aligned S, got S={S}"

    invoked = []
    tout_calls = []
    state = {"bsnd": False}

    if args.arm == "w2":
        import gqa_w2_bsnd_iter9_variant as VAR

        orig_tout = G.transpose_out
        saved = {}

        # wave-2 host gate: only aligned-S shapes get the BSND kernels
        if aligned:
            for name in W2_TARGETS:
                orig = getattr(G, name)
                saved[name] = orig
                var_fn = getattr(VAR, name)

                def wrap(*a, _n=name, _f=var_fn, **kw):
                    invoked.append(_n)
                    return _f(*a, **kw)

                setattr(G, name, wrap)

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
        f"N_kv={N_kv} D={D} causal={causal} dtype={dtag} role={role} aligned={aligned}",
        flush=True,
    )

    out = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
    torch.npu.synchronize()

    chosen = [val[2] for val in G._KERNEL_BUILD_CACHE.values()]

    # ---- probe assertions: an unattributable sample must never be measured ----
    if args.arm == "w2":
        if covered:
            assert invoked, f"[AB-PROBE] w2 covered shape {args.shape}: variant NEVER invoked"
            assert state["bsnd"], f"[AB-PROBE] w2 {args.shape}: transpose_out was NOT skipped"
            assert all(c[0] == "SKIP" for c in tout_calls), f"[AB-PROBE] tout calls {tout_calls}"
        else:
            assert not invoked, (
                f"[AB-PROBE] w2 OUT-OF-DOMAIN shape {args.shape} (role={role}): variant invoked "
                f"({invoked}) — the wave-2 gate leaked")
            if role in ("unaligned", "perhead_untouched"):
                # per-head excluded / non-aligned S: transpose_out MUST still run
                assert any(c[0] == "RUN" for c in tout_calls), f"[AB-PROBE] tout not run: {tout_calls}"
            else:
                # w1 (shipped BSND — the adapter returns before transpose_out)
                # and decode (BSND direct) must NOT reach transpose_out at all
                assert not tout_calls, f"[AB-PROBE] {role} must not call transpose_out: {tout_calls}"
        # restore (single-shot process, but keep the discipline)
        for name, orig in saved.items():
            setattr(G, name, orig)
        G.transpose_out = orig_tout
    else:
        assert not invoked, "[AB-PROBE] base arm contaminated by the variant"
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
