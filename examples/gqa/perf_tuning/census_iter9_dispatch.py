"""Iter9 P2-wave2 coverage census — pure-host replication of gqa_expert's
prefill dispatch predicates (NO compilation, no NPU) + the wave-1 gate (shipped)
+ the PROPOSED wave-2 gates (per domain).

Purpose (Phase A scope): for every bench case (cann-bench level4 gqa
cases.yaml) and every perf workload (r1..r6/c7 + wave-2 candidates), state
  (a) which main kernel the chain resolves to (first-compiling spec; chain
      order source of truth: gqa.py:4660-4884),
  (b) whether transpose_out runs (decode = BSND direct, no tout),
  (c) wave-1 coverage (SHIPPED): chosen in {g2stack_flat, g2stack_flat_kv2}
      AND S % 64 == 0,
  (d) wave-2 domain classification (PROPOSED):
        per_head        chosen startswith "per_head_bm"   AND S % 64 == 0
        prefill_expert  chosen == "prefill_expert"        AND S % 64 == 0
        g2stack         chosen == "g2stack"               AND S % 64 == 0
        out             everything else (non-aligned S keeps transpose_out
                        (s_real=...) — the iter8 crop tradeoff is unchanged;
                        decode is already BSND; "g2" last-resort fallback
                        stays BHSD by design)

Line-anchored to gqa.py md5 3aa9cfa2625cdce96a62ce967785fe85 (iter8 adopted)
so predicate drift fails loudly instead of silently mis-scoping.

Run: python3 examples/gqa/perf_tuning/census_iter9_dispatch.py
"""

import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_DIR = os.path.dirname(HERE)
GQA_PY = os.path.join(GQA_DIR, "gqa.py")
EXPECT_MD5 = "3aa9cfa2625cdce96a62ce967785fe85"
P28_NUM_CORES = 20

# ---- line-anchored drift guard on the dispatch predicates -----------------
ANCHORS = {
    4317: '_P2_BSND_KERNELS = ("g2stack_flat", "g2stack_flat_kv2")',
    4364: "block_N = 128",
    4365: "S_kv_padded = ((S_kv + block_N - 1) // block_N) * block_N",
    4623: "S_padded = ((S + block_M - 1) // block_M) * block_M",
    4626: "use_expert_prefill = (",
    4629: "and (kv_iters == 1 or (kv_iters >= 2 and kv_iters % 2 == 0))",
    4631: "and (B * N_q * q_blocks) >= P28_NUM_CORES",
    4644: "use_g2_stack = use_expert_prefill and (G % 2) == 0",
    4654: "bsnd_eff = bool(out_bsnd) and S_padded == S",
    4681: "if kv_iters == 1 and flat_block_num >= 2 * P28_NUM_CORES:",
    4727: "if kv_iters == 2 and not is_causal and flat_block_num >= 2 * P28_NUM_CORES:",
    4923: "_run.out_bsnd = bool(bsnd_eff) and chosen in _P2_BSND_KERNELS",
    5854: 'if getattr(kernel_fn, "out_bsnd", False):',
    5857: "return transpose_out(out_padded, s_real=S_bs)",
}


def check_anchors():
    md5 = hashlib.md5(open(GQA_PY, "rb").read()).hexdigest()
    assert md5 == EXPECT_MD5, f"gqa.py drifted: {md5} != {EXPECT_MD5}"
    lines = open(GQA_PY).read().splitlines()
    for ln, pat in ANCHORS.items():
        assert pat in lines[ln - 1], f"anchor drift at gqa.py:{ln}: {lines[ln-1]!r} !~ {pat!r}"
    return md5


def perhead_ladder(D, block_N=128):
    return [bm for bm in (64, 32, 16) if (2 * bm * block_N + 2 * bm * D) * 4 <= 196608]


def dispatch(B, S, S_kv, N_q, N_kv, D, is_causal, dtype="float16"):
    """Return (chosen_first_spec, S_padded, tout_runs, wave1, wave2_domain).

    chosen_first = the FIRST spec in chain order (what production resolves to
    when every factory compiles — the shipped behavior on this platform;
    compile fallbacks only reorder toward BHSD-safe kernels, and the wave-1
    capability flag already reverts automatically then).
    """
    G = N_q // N_kv
    block_M = 64
    block_N = 128
    S_padded = ((S + block_M - 1) // block_M) * block_M
    S_kv_padded = ((S_kv + block_N - 1) // block_N) * block_N
    kv_iters = S_kv_padded // block_N
    q_blocks = S_padded // block_M

    if S <= 4:  # decode — BSND direct, no tout, out of wave scope
        return ("decode", S, False, False, "out(decode)")

    use_expert_prefill = (
        D == 128
        and dtype in ("float16", "bfloat16")
        and (kv_iters == 1 or (kv_iters >= 2 and kv_iters % 2 == 0))
        and ((not is_causal) or (S_kv >= S))
        and (B * N_q * q_blocks) >= P28_NUM_CORES
    )
    use_g2_stack = (
        use_expert_prefill
        and (G % 2) == 0
        and (B * N_kv * (G // 2) * q_blocks) >= P28_NUM_CORES
    )

    chosen = None
    if use_g2_stack:
        flat_block_num = B * N_kv * (G // 2) * q_blocks
        if kv_iters == 1 and flat_block_num >= 2 * P28_NUM_CORES:
            chosen = "g2stack_flat"
        elif kv_iters == 2 and not is_causal and flat_block_num >= 2 * P28_NUM_CORES:
            chosen = "g2stack_flat_kv2"
        else:
            chosen = "g2stack"
    elif use_expert_prefill:
        chosen = "prefill_expert"
    else:
        ladder = perhead_ladder(D, block_N)
        chosen = f"per_head_bm{ladder[0]}" if ladder else "g2"

    aligned = S_padded == S  # S % 64 == 0
    wave1 = chosen in ("g2stack_flat", "g2stack_flat_kv2") and aligned
    if not aligned:
        dom = "out(non-aligned-S)"
    elif chosen.startswith("per_head_bm"):
        dom = "per_head"
    elif chosen == "prefill_expert":
        dom = "prefill_expert"
    elif chosen == "g2stack":
        dom = "g2stack"
    elif wave1:
        dom = "wave1(shipped)"
    else:
        dom = "out"
    tout_runs = not wave1  # prefill: tout runs unless wave-1 eliminated it
    return (chosen, S_padded, tout_runs, wave1, dom)


# (label, B, S, S_kv, N_q, N_kv, D, causal, dtype)
BENCH = [
    ("c1", 2, 128, 128, 32, 8, 128, True, "float16"),
    ("c2", 2, 128, 128, 48, 4, 128, False, "float16"),
    ("c3", 2, 128, 128, 128, 8, 128, True, "float16"),
    ("c4", 4, 256, 256, 64, 8, 128, False, "bfloat16"),
    ("c5", 2, 128, 128, 32, 8, 256, True, "float16"),
    ("c6", 2, 256, 256, 128, 8, 256, True, "bfloat16"),
    ("c7", 2, 128, 256, 32, 8, 128, False, "bfloat16"),
    ("c8", 4, 1024, 1024, 32, 8, 128, True, "float16"),
    ("c9", 4, 512, 512, 64, 8, 128, False, "bfloat16"),
    ("c10", 4, 512, 512, 128, 8, 128, True, "bfloat16"),
    ("c11", 16, 1, 2048, 32, 8, 128, False, "float16"),
    ("c12", 16, 1, 2048, 96, 8, 128, False, "bfloat16"),
    ("c13", 60, 1, 2048, 32, 8, 128, False, "float16"),
    ("c14", 128, 1, 512, 128, 8, 128, False, "float16"),
    ("c15", 60, 1, 1024, 32, 8, 256, False, "float16"),
    ("c16", 60, 2, 2048, 32, 8, 128, True, "float16"),
    ("c17", 128, 4, 512, 128, 8, 128, True, "float16"),
    ("c18", 2, 128, 128, 32, 1, 128, False, "float16"),
    ("c19", 2, 128, 128, 32, 32, 128, True, "bfloat16"),
    ("c20", 2, 128, 128, 32, 8, 128, True, "bfloat16"),
]
PERF = [
    ("r1", 1, 128, 128, 8, 2, 128, True, "float16"),
    ("r2", 2, 128, 128, 32, 8, 128, True, "float16"),
    ("r3", 2, 128, 128, 128, 8, 128, True, "bfloat16"),
    ("r4", 4, 256, 256, 64, 8, 128, False, "bfloat16"),
    ("r5", 16, 1, 2048, 32, 8, 128, False, "float16"),
    ("r6", 2, 128, 128, 48, 4, 128, False, "float16"),
]
# extra isolation / gate-edge shapes (Gate 0 L3 set, iter9)
EXTRA = [
    ("padrow_s100_flat", 2, 100, 256, 32, 8, 128, False, "float16"),
    ("l1_s_tail_s130", 2, 130, 130, 32, 8, 128, True, "float16"),
    ("padrow_s100_perhead", 1, 100, 100, 8, 2, 128, True, "float16"),
    ("padrow_s100_g2stack", 4, 100, 512, 64, 8, 128, False, "bfloat16"),
    ("padrow_s100_prexpert", 2, 100, 128, 32, 32, 128, True, "bfloat16"),
    ("stacked20_g2stack_ns1", 1, 128, 128, 20, 2, 128, True, "float16"),
    ("odd_kv3_perhead", 2, 128, 300, 32, 8, 128, False, "float16"),
    ("causal_skvlt_s_perhead", 2, 256, 128, 32, 8, 128, True, "float16"),
    ("decode_auto_small", 2, 2, 512, 8, 2, 128, False, "float16"),
]


def main():
    md5 = check_anchors()
    print(f"gqa.py md5 {md5} — anchors OK ({len(ANCHORS)} sites)\n")
    counts = {}
    for group, cases in (("BENCH", BENCH), ("PERF", PERF), ("EXTRA", EXTRA)):
        print(f"== {group} ==")
        print(f"{'case':24s} {'chosen':16s} {'Spad':>5s} {'tout':>5s} {'wave1':>5s} wave2_domain")
        for label, B, S, S_kv, N_q, N_kv, D, causal, dt in cases:
            chosen, spad, tout, w1, dom = dispatch(B, S, S_kv, N_q, N_kv, D, causal, dt)
            counts[dom] = counts.get(dom, 0) + 1
            print(f"{label:24s} {chosen:16s} {spad:5d} {str(tout):>5s} {str(w1):>5s} {dom}")
        print()
    print("domain census:", counts)
    bench_only = {}
    for label, B, S, S_kv, N_q, N_kv, D, causal, dt in BENCH:
        _, _, _, _, dom = dispatch(B, S, S_kv, N_q, N_kv, D, causal, dt)
        bench_only[dom] = bench_only.get(dom, 0) + 1
    print("bench-only census:", bench_only)


if __name__ == "__main__":
    main()
