"""Iter8 P2 coverage census — pure-host replication of gqa_expert's prefill
dispatch predicates (NO compilation, no NPU) + the P2 wave-1 gate.

Purpose (Phase A scope界定): for every bench case (cann-bench level4 gqa
cases.yaml) and every perf work load (r1..r6/c7), state
  (a) which main kernel the chain resolves to (first-compiling spec; the
      chain order is the source of truth in gqa.py:4570-4789),
  (b) whether transpose_out runs at all (decode = BSND direct, no tout),
  (c) whether the P2 wave-1 gate covers it:
          chosen in {g2stack_flat, g2stack_flat_kv2}  AND  S % 64 == 0
      (S%64==0 <=> S_padded == S <=> the BSND output needs NO row crop;
       non-aligned S would need Out[:, :S].contiguous() = a fp16/bf16
       ViewCopy d2d whose binary is missing on some CANN 9.x packages,
       gqa.py:5486-5489 — that is why the padded domain stays on
       transpose_out(s_real=...) in wave 1).

The predicate replication is line-anchored to gqa.py (md5 asserted) so a
drift in the dispatch gates fails loudly instead of silently mis-scoping.

Run: python3 examples/gqa/perf_tuning/census_iter8_dispatch.py
"""

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_DIR = os.path.dirname(HERE)
GQA_PY = os.path.join(GQA_DIR, "gqa.py")
EXPECT_MD5 = "4b4e453e4b8f2922baefbecb85912494"
P28_NUM_CORES = 20

# ---- line-anchored drift guard on the dispatch predicates -----------------
ANCHORS = {
    4543: "use_expert_prefill = (",
    4561: "use_g2_stack = use_expert_prefill and (G % 2) == 0",
    4588: "if kv_iters == 1 and flat_block_num >= 2 * P28_NUM_CORES:",
    4633: "if kv_iters == 2 and not is_causal and flat_block_num >= 2 * P28_NUM_CORES:",
    4540: "S_padded = ((S + block_M - 1) // block_M) * block_M",
    4546: "and (kv_iters == 1 or (kv_iters >= 2 and kv_iters % 2 == 0))",
    4548: "and (B * N_q * q_blocks) >= P28_NUM_CORES",
    5741: "return transpose_out(out_padded, s_real=S_bs)",
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
    """Return (chain_spec_order, chosen_first, S_padded, tout_runs, p2_covered)."""
    G = N_q // N_kv
    block_N = 128
    S_kv_padded = ((S_kv + block_N - 1) // block_N) * block_N
    use_mask = bool(is_causal or (S_kv_padded != S_kv))
    if 1 <= S <= 4:
        kv_iters = S_kv_padded // block_N
        G_inner = 4 if (S <= 2 and G >= 4 and D == 128) else min(G, 2)  # _dense_tile_params
        specs = []
        use_expert = (D == 128 and dtype in ("float16", "bfloat16") and kv_iters >= 4
                      and kv_iters % 2 == 0 and (G % G_inner) == 0
                      and (B * N_kv * (G // G_inner)) >= P28_NUM_CORES)
        use_expert_d256 = (D == 256 and dtype in ("float16", "bfloat16")
                           and (S_kv_padded // 64) >= 4 and (G % G_inner) == 0
                           and (B * N_kv * (G // G_inner)) >= P28_NUM_CORES)
        if use_expert:
            specs.append("dense_expert")
        if use_expert_d256:
            specs.append("dense_expert_d256")
        specs.append("dense_auto")
        return specs, specs[0], None, False, False, use_mask

    block_M = 64
    S_padded = ((S + block_M - 1) // block_M) * block_M
    kv_iters = S_kv_padded // block_N
    q_blocks = S_padded // block_M
    use_expert_prefill = (
        D == 128
        and dtype in ("float16", "bfloat16")
        and (kv_iters == 1 or (kv_iters >= 2 and kv_iters % 2 == 0))
        and ((not is_causal) or (S_kv >= S))
        and (B * N_q * q_blocks) >= P28_NUM_CORES
    )
    use_g2_stack = (use_expert_prefill and (G % 2) == 0
                    and (B * N_kv * (G // 2) * q_blocks) >= P28_NUM_CORES)
    specs = []
    if use_g2_stack:
        flat_block_num = B * N_kv * (G // 2) * q_blocks
        if kv_iters == 1 and flat_block_num >= 2 * P28_NUM_CORES:
            specs.append("g2stack_flat")
        if kv_iters == 2 and not is_causal and flat_block_num >= 2 * P28_NUM_CORES:
            specs.append("g2stack_flat_kv2")
        specs.append("g2stack")
    if use_expert_prefill:
        specs.append("prefill_expert")
    for bm in perhead_ladder(D, block_N):
        specs.append(f"per_head_bm{bm}")
    if G >= 2 and D <= 128:
        specs.append("g2")
    chosen = specs[0] if specs else "g2"
    p2_covered = chosen in ("g2stack_flat", "g2stack_flat_kv2") and S % 64 == 0
    return specs, chosen, S_padded, True, p2_covered, use_mask


BENCH = [
    # case_id, B, S, S_kv, Nq, Nkv, D, causal, dtype
    (1, 2, 128, 128, 32, 8, 128, True, "float16"),
    (2, 2, 128, 128, 48, 4, 128, False, "float16"),
    (3, 2, 128, 128, 128, 8, 128, True, "bfloat16"),
    (4, 4, 256, 256, 64, 8, 128, False, "bfloat16"),
    (5, 2, 128, 128, 32, 8, 256, True, "float16"),
    (6, 2, 256, 256, 128, 8, 256, True, "bfloat16"),
    (7, 2, 128, 256, 32, 8, 128, False, "bfloat16"),
    (8, 4, 1024, 1024, 32, 8, 128, True, "float16"),
    (9, 4, 512, 512, 64, 8, 128, False, "bfloat16"),
    (10, 4, 512, 512, 128, 8, 128, True, "bfloat16"),
    (11, 16, 1, 2048, 32, 8, 128, False, "float16"),
    (12, 16, 1, 2048, 96, 8, 128, False, "bfloat16"),
    (13, 60, 1, 2048, 32, 8, 128, False, "float16"),
    (14, 128, 1, 512, 128, 8, 128, False, "float16"),
    (15, 60, 1, 1024, 32, 8, 256, False, "float16"),
    (16, 60, 2, 2048, 32, 8, 128, True, "float16"),
    (17, 128, 4, 512, 128, 8, 128, True, "float16"),
    (18, 2, 128, 128, 32, 1, 128, False, "float16"),
    (19, 2, 128, 128, 32, 32, 128, True, "bfloat16"),
    (20, 2, 128, 128, 32, 8, 128, True, "bfloat16"),
]
PERF = [
    ("r1", 1, 128, 128, 8, 2, 128, True, "float16"),
    ("r2", 2, 128, 128, 32, 8, 128, True, "float16"),
    ("r3", 2, 128, 128, 128, 8, 128, True, "bfloat16"),
    ("r4", 4, 256, 256, 64, 8, 128, False, "bfloat16"),
    ("r5", 16, 1, 2048, 32, 8, 128, False, "float16"),
    ("r6", 2, 128, 128, 48, 4, 128, False, "float16"),
    ("c7", 2, 128, 256, 32, 8, 128, False, "bfloat16"),
]
# extra non-aligned / out-of-domain shapes used by Gate 0 isolation checks
EXTRA = [
    ("padrow_s100", 2, 100, 128, 32, 8, 128, True, "float16"),
    ("padcol_skv200", 2, 128, 200, 32, 8, 128, False, "float16"),
    ("l1_s_tail_s130", 2, 130, 130, 32, 8, 128, True, "float16"),
    ("d256_perhead", 2, 128, 128, 32, 8, 256, True, "float16"),
    ("mha_g1", 2, 128, 128, 32, 32, 128, True, "bfloat16"),
    ("kv8_g2stack", 4, 1024, 1024, 32, 8, 128, True, "float16"),
]


def main():
    md5 = check_anchors()
    print(f"gqa.py md5 {md5} — dispatch anchors verified ({len(ANCHORS)} lines)\n")
    rows = []
    for tag, group in (("bench", BENCH), ("perf", PERF), ("extra", EXTRA)):
        for item in group:
            if tag == "bench":
                cid, B, S, Skv, Nq, Nkv, D, ca, dt = item
                name = f"c{cid}"
            else:
                name, B, S, Skv, Nq, Nkv, D, ca, dt = item
            specs, chosen, sp, tout, cov, um = dispatch(B, S, Skv, Nq, Nkv, D, ca, dt)
            rows.append((tag, name, B, S, Skv, Nq, Nkv, D, ca, dt, chosen, sp, tout, cov, um, specs))
    hdr = (f"{'grp':6} {'name':14} {'B':>3} {'S':>5} {'Skv':>5} {'Nq':>4} {'Nkv':>4} {'D':>4} "
           f"{'cau':>4} {'dt':>8} | {'chosen':18} {'S_pad':>6} {'tout':>5} {'P2cov':>6} {'mask':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        (tag, name, B, S, Skv, Nq, Nkv, D, ca, dt, chosen, sp, tout, cov, um, specs) = r
        print(f"{tag:6} {name:14} {B:>3} {S:>5} {Skv:>5} {Nq:>4} {Nkv:>4} {D:>4} "
              f"{str(ca)[:1]:>4} {dt[5:]:>8} | {chosen:18} {str(sp):>6} "
              f"{('yes' if tout else 'no'):>5} {('YES' if cov else '-'):>6} {('yes' if um else 'no'):>5}")
    print()
    cov_bench = [r[1] for r in rows if r[0] == "bench" and r[13]]
    cov_perf = [r[1] for r in rows if r[0] == "perf" and r[13]]
    tout_bench = [r[1] for r in rows if r[0] == "bench" and r[12]]
    print(f"P2 wave-1 coverage (chosen in {{g2stack_flat, g2stack_flat_kv2}} AND S%64==0):")
    print(f"  bench: {len(cov_bench)}/{len(BENCH)} cases -> {cov_bench}")
    print(f"  perf : {cov_perf}")
    print(f"  bench cases that run transpose_out at all: {len(tout_bench)}/{len(BENCH)} -> {tout_bench}")
    print(f"  NOT covered but tout-bearing: "
          f"{[r[1] for r in rows if r[0]=='bench' and r[12] and not r[13]]}")
    print()
    print("Chain order (first spec = chosen unless it fails to compile):")
    for r in rows:
        if r[0] != "bench" or r[1] in ("c1", "c4", "c5", "c7", "c8", "c11", "c18", "c19"):
            print(f"  {r[1]:6} {r[10]:18} <- {r[15]}")


if __name__ == "__main__":
    main()
