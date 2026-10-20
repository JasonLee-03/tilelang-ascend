"""Iter10 same-session A/B probe: base (shipped chain) vs p4 (transpose_qkv
eliminated on the flat kv1/kv2 aligned domain — main kernel reads BSND
strided GM directly).

Single-shot run for msprof collection (mirrors prof_gqa_ab_iter9.py:
EXACTLY ONE gqa() call = the full kernel chain, no warm-up, no golden compare
inside the profiled process).

Arms:
  base : shipped gqa.py as-is (md5 606b07ff474ec42b8244f0831327d75b, iter9
         adopted), no patches.  Covered shapes (flat kv1/kv2, aligned):
         chain = transpose_qkv + main (tout already eliminated by wave-1
         out_bsnd).  Guards: r1 per-head = tqkv + main + tout; c19/c8 =
         tqkv + main (wave-2); r5 decode = main only.
  p4   : P4 variant chain — the TWO flat factories replaced by
         perf_tuning/gqa_p4_inbsnd_iter10_variant.py (5 audited line-scoped
         edit sites each) with in_bsnd=True injected, and transpose_qkv
         conditionally PASSED THROUGH under the SAME gate predicate
         (p4_gate: aligned S AND aligned S_kv AND flat dispatch — the
         Gate-0 harness-fix pattern: one predicate drives both the factory
         patch and the pass-through, mirroring the integration's single
         in_bsnd_eff variable).  Covered-shape chain = main ONLY.
         PER-HEAD / g2stack / prefill_expert factories are NEVER patched
         (first-wave scope); their shapes are untouched guards here.
         Asserts (a) the variant factory WAS invoked on covered shapes and
         NEVER on guards, (b) chosen spec unchanged, (c) transpose_qkv
         actually skipped on covered shapes / actually run elsewhere.

BOTH dispatch caches (_KERNEL_BUILD_CACHE + adapter _COMPILED_CACHE) are
cleared so patches are guaranteed to take effect (iter5 finding 2).
gqa.py / test_gqa.py are NOT modified by anything in this iteration until
Gate 3 (discipline #6).

SERIAL-ONLY discipline (task-book #5): every kernel family here except the
per-head AUTO ladder is blockDim=20 Fixed-Core with cross-core semaphores —
never run two collections concurrently.

Usage (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:$LD_LIBRARY_PATH
    msprof op --launch-count=3 --aic-metrics=PipeUtilization,Memory,L2Cache \
        --output=<dir> python3 examples/gqa/perf_tuning/prof_gqa_ab_iter10.py \
        --arm base --shape r2
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
    # role: p4        = covered (variant invoked, tqkv SKIPPED)
    #       perhead   = per-head guard (untouched; tqkv + main + tout both arms)
    #       decode    = decode guard (no tqkv either arm)
    #       prexpert  = prefill_expert guard (untouched; tqkv + main both arms)
    #       g2stack   = g2stack guard (untouched; tqkv + main both arms)
    # ---- P4 covered: flat kv1 (primary r2, secondary r6/r3) ----
    "r2": (2, 128, 128, 32, 8, 128, True, "fp16", "g2stack_flat", "p4"),
    "r6": (2, 128, 128, 48, 4, 128, False, "fp16", "g2stack_flat", "p4"),
    "r3": (2, 128, 128, 128, 8, 128, True, "bf16", "g2stack_flat", "p4"),
    # ---- P4 covered: flat kv2 (primary r4, secondary c7) ----
    "r4": (4, 256, 256, 64, 8, 128, False, "bf16", "g2stack_flat_kv2", "p4"),
    "c7": (2, 128, 256, 32, 8, 128, False, "bf16", "g2stack_flat_kv2", "p4"),
    # ---- guards ----
    "r1": (1, 128, 128, 8, 2, 128, True, "fp16", "per_head_bm", "perhead"),
    "r5": (16, 1, 2048, 32, 8, 128, False, "fp16", "dense_expert", "decode"),
    "c19": (2, 128, 128, 32, 32, 128, True, "bf16", "prefill_expert", "prexpert"),
    "c8": (4, 1024, 1024, 32, 8, 128, True, "fp16", "g2stack", "g2stack"),
    # ---- Gate-3 fallback addition: flat kv2 dispatch HIT but S_kv=200
    # non-aligned => the in_bsnd gate excludes it; tqkv MUST still run on
    # the INTEGRATED file (zero monkeypatch) ----
    "skv200": (2, 128, 200, 32, 8, 128, False, "fp16", "g2stack_flat_kv2", "gate_ungated"),
}

P4_TARGETS = (
    "_gqa_kernel_prefill_g2stack_flat",
    "_gqa_kernel_prefill_g2stack_flat_kv2",
)


def p4_gate(B, S, S_kv, N_q, N_kv, D, dtype, causal):
    """Pure-host replication of gqa_expert's flat dispatch + P4 alignment gate
    (SAME function as run_iter10_p4_gate0.py — one predicate drives both the
    factory patch and the transpose_qkv pass-through)."""
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


def main():
    ap = argparse.ArgumentParser(description="Single-shot GQA A/B run for msprof (iter10 P4)")
    ap.add_argument("--arm", required=True, choices=["base", "p4"])
    ap.add_argument("--shape", required=True, choices=list(SHAPES))
    args = ap.parse_args()

    B, S, S_kv, N_q, N_kv, D, causal, dtag, expect, role = SHAPES[args.shape]
    covered = role == "p4"
    dtype = torch.float16 if dtag == "fp16" else torch.bfloat16
    gated = covered and p4_gate(B, S, S_kv, N_q, N_kv, D, dtype, causal)
    if covered:
        assert gated, f"[AB-PROBE] covered shape {args.shape} fails its own gate predicate"

    invoked = []
    tqkv_calls = []

    if args.arm == "p4":
        import gqa_p4_inbsnd_iter10_variant as VAR

        orig_tqkv = G.transpose_qkv
        saved = {}
        # ONE gate drives both patches (Gate-0 harness-fix pattern; the
        # integration's single in_bsnd_eff variable is the structural form)
        if gated:
            for name in P4_TARGETS:
                orig = getattr(G, name)
                saved[name] = orig
                var_fn = getattr(VAR, name)

                def wrap(*a, _n=name, _f=var_fn, **kw):
                    kw["in_bsnd"] = True
                    invoked.append(_n)
                    return _f(*a, **kw)

                setattr(G, name, wrap)

        def tqkv_cond(query, key, value, S_out=None, S_kv_out=None):
            if gated:
                Sq, Skv = query.shape[1], key.shape[1]
                assert S_out in (None, Sq) and S_kv_out in (None, Skv), \
                    f"[AB-PROBE] pass-through requires no padding: {S_out}/{S_kv_out}"
                tqkv_calls.append(("SKIP", tuple(query.shape)))
                return (query, key, value)
            tqkv_calls.append(("RUN", tuple(query.shape)))
            return orig_tqkv(query, key, value, S_out=S_out, S_kv_out=S_kv_out)

        G.transpose_qkv = tqkv_cond

    G._KERNEL_BUILD_CACHE.clear()
    G._COMPILED_CACHE.clear()

    torch.manual_seed(42)
    q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
    k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
    v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()

    print(
        f"[PROF_RUN] arm={args.arm} shape={args.shape} B={B} S={S} S_kv={S_kv} N_q={N_q} "
        f"N_kv={N_kv} D={D} causal={causal} dtype={dtag} role={role} gated={gated}",
        flush=True,
    )

    out = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
    torch.npu.synchronize()

    chosen = [val[2] for val in G._KERNEL_BUILD_CACHE.values()]

    # ---- probe assertions: an unattributable sample must never be measured ----
    if args.arm == "p4":
        if covered:
            assert invoked, f"[AB-PROBE] p4 covered shape {args.shape}: variant NEVER invoked"
            assert all(c[0] == "SKIP" for c in tqkv_calls) and len(tqkv_calls) == 1, \
                f"[AB-PROBE] tqkv calls {tqkv_calls} — expected exactly one SKIP"
        else:
            assert not invoked, (
                f"[AB-PROBE] p4 GUARD shape {args.shape} (role={role}): variant invoked "
                f"({invoked}) — the P4 gate leaked")
            if role == "decode":
                assert not tqkv_calls, f"[AB-PROBE] decode must not call tqkv: {tqkv_calls}"
            else:
                assert len(tqkv_calls) == 1 and tqkv_calls[0][0] == "RUN", \
                    f"[AB-PROBE] guard tqkv behavior changed: {tqkv_calls}"
        for name, orig in saved.items():
            setattr(G, name, orig)
        G.transpose_qkv = orig_tqkv
    else:
        assert not invoked, "[AB-PROBE] base arm contaminated by the variant"
    assert chosen and chosen[0].startswith(expect), \
        f"[AB-PROBE] arm={args.arm}: dispatch resolved to {chosen}, expected ~{expect} — INVALID"
    assert tuple(out.shape) == (B, S, N_q, D), f"[AB-PROBE] out shape {tuple(out.shape)}"
    assert out.is_contiguous(), "[AB-PROBE] out not contiguous"

    print(
        f"[PROF_DONE] arm={args.arm} shape={args.shape} chosen={chosen} "
        f"variant_invoked={invoked} tqkv={[c[0] for c in tqkv_calls]} "
        f"out.shape={tuple(out.shape)} out.dtype={out.dtype}",
        flush=True,
    )


if __name__ == "__main__":
    main()
