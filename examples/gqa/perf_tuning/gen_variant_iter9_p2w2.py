"""Iter9 P2-wave2 variant generator — prefill_expert / g2stack with BSND
direct-write Output (the iter8 wave-1 mechanism extended to the two
REMAINING VIABLE transpose_out domains).

PER-HEAD `_gqa_kernel_expert` WAS DROPPED after the Phase-B verdict: the AUTO
threads=2 (serial/causal) form mis-lowers the strided BSND write
(AscendVidReduction keeps the contiguous vid offset while emitting the
unsplit blockCount — codegen_audit_iter9.log + bit_probe_iter9.log: r1/c5
BITFAIL).  All bench per-head cases (r1/c5/c6) are causal => threads=2, so
the domain yields nothing viable this round; its BLOCKS/SITES entries are
RETAINED below as line-anchored drift guards (load_blocks asserts them) and
the pre-verdict diff is preserved in
diff_iter9_p2w2_with_perhead_pre_verdict.txt.

Extracts the TWO SHIPPED kernel blocks from examples/gqa/gqa.py
(md5-asserted, line-anchored) and applies the minimal P2 edit — 2 lines per
kernel — so each variant is provably "shipped kernel + tiny diff"
(gen_variant_iter8_p2.py pattern):

    perf_tuning/gqa_w2_bsnd_iter9_variant.py  +  diff_iter9_p2w2.txt

Mechanism (identical to wave-1, p2_bsnd_design_iter8.md §3):
  * `main`'s Output declaration changes from BHSD [B, N_q, S_padded, D] to
    BSND [B, S_padded, N_q, D].
  * The output write box changes from `Output[bz, hq, rows, :]` (contiguous,
    row stride D) to `Output[bz, rows, hq, :]` (strided, row stride N_q*D).
    Element (b,hq,s,d) lands at ((b*S+s)*N_q+hq)*D+d == exactly what
    _transpose_v1 (transpose_out) wrote (algebraic identity, wave-1 §3.3).
  * NOTHING else changes.  For the two EXPERT kernels (prefill_expert,
    g2stack) every set/wait flag line is byte-identical (audited) — the FIN
    write stays inside its shipped SIG_S_HALF V<->MTE3 pair.  For the
    per-head AUTO kernel there are no manual flag lines at all (auto-sync);
    the audit instead pins the schedule-relevant host lines (use_pipeline /
    pipe_stages / kernel_threads / skip_limit / T.Kernel launch form) and the
    generated-code audit (codegen_audit_iter9.py) verifies the AUTO codegen
    emits the same instruction count with only dstStride changed.
  * Coverage gate (host side, applied by the harness/integration, NOT by the
    kernel): S % 64 == 0 so S_padded == S and the row window never leaves the
    real-S range => no crop, no masked writes (wave-1 design §3.4 — the row
    windows here are [bx*BM, (bx+1)*BM) (per-head) and
    [bx*BM + vid*half, +half) (expert kernels), both structurally inside
    [0, S_padded) == [0, S)).

Run:  python3 examples/gqa/perf_tuning/gen_variant_iter9_p2w2.py
"""

import difflib
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_DIR = os.path.dirname(HERE)
GQA_PY = os.path.join(GQA_DIR, "gqa.py")

EXPECT_MD5 = "3aa9cfa2625cdce96a62ce967785fe85"

# 1-based inclusive line ranges of the shipped factory blocks (decorator..return main)
BLOCKS = {
    "perhead": (57, 300, "_gqa_kernel_expert"),
    "prexpert": (1997, 2468, "_gqa_kernel_prefill_expert"),
    "g2stack": (2471, 2969, "_gqa_kernel_prefill_g2stack"),
}

# pinned edit sites (absolute gqa.py line numbers)
EDITS = {
    "perhead": {"decl": 162, "write": 297},
    "prexpert": {"decl": 2120, "write": 2459},
    "g2stack": {"decl": 2603, "write": 2960},
}

# per-kernel site text (decl comments differ: the per-head kernel says
# "BHSD layout", the expert kernels say "BHSD"; write indentation differs)
SITES = {
    "perhead": {
        "decl_old": "        Output: T.Tensor([B, N_q, S_padded, D], dtype),  # type: ignore  BHSD layout\n",
        "decl_new": (
            "        # [S3-iter9-P2w2] BSND direct write: the output box below is\n"
            "        # strided (row stride N_q*D) instead of contiguous, so the host\n"
            "        # skips transpose_out entirely (wave-1 mechanism, iter8).\n"
            "        Output: T.Tensor([B, S_padded, N_q, D], dtype),  # type: ignore  BSND layout\n"
        ),
        "write_old": "                Output[bz, hq, bx * block_M : (bx + 1) * block_M, :],\n",
        "write_new": (
            "                # [S3-iter9-P2w2] BSND box (bz, rows, hq, :) — same element\n"
            "                # set, strided rows; one DataCopyPad, count unchanged.\n"
            "                Output[bz, bx * block_M : (bx + 1) * block_M, hq, :],\n"
        ),
    },
    "prexpert": {
        "decl_old": "        Output: T.Tensor([B, N_q, S_padded, D], dtype),  # type: ignore  BHSD\n",
        "decl_new": (
            "        # [S3-iter9-P2w2] BSND direct write: the FIN box below is strided\n"
            "        # (row stride N_q*D) instead of contiguous, so the host skips\n"
            "        # transpose_out entirely (wave-1 mechanism, iter8).\n"
            "        Output: T.Tensor([B, S_padded, N_q, D], dtype),  # type: ignore  BSND\n"
        ),
        "write_old": "                        Output[bz, hq, bx * block_M + vid * half : bx * block_M + vid * half + half, :],\n",
        "write_new": (
            "                        # [S3-iter9-P2w2] BSND box (bz, rows, hq, :) — same\n"
            "                        # element set, strided rows; one DataCopyPad inside\n"
            "                        # the shipped SIG_S_HALF pair, count unchanged.\n"
            "                        Output[bz, bx * block_M + vid * half : bx * block_M + vid * half + half, hq, :],\n"
        ),
    },
    "g2stack": {
        "decl_old": "        Output: T.Tensor([B, N_q, S_padded, D], dtype),  # type: ignore  BHSD\n",
        "decl_new": (
            "        # [S3-iter9-P2w2] BSND direct write: the FIN boxes below are strided\n"
            "        # (row stride N_q*D) instead of contiguous, so the host skips\n"
            "        # transpose_out entirely (wave-1 mechanism, iter8).\n"
            "        Output: T.Tensor([B, S_padded, N_q, D], dtype),  # type: ignore  BSND\n"
        ),
        "write_old": "                            Output[bz, hq0 + h, bx * block_M + vid * half : bx * block_M + vid * half + half, :],\n",
        "write_new": (
            "                            # [S3-iter9-P2w2] BSND box (bz, rows, hq, :) — same\n"
            "                            # element set, strided rows; one DataCopyPad inside\n"
            "                            # the shipped SIG_S_HALF pair, count unchanged.\n"
            "                            Output[bz, bx * block_M + vid * half : bx * block_M + vid * half + half, hq0 + h, :],\n"
        ),
    },
}

HEADER = '''"""P2 wave-2 variant — prefill_expert / g2stack with BSND direct-write Output.

Standalone Gate-0/Gate-2 development file (Stage 3 iter9).  Generated by
gen_variant_iter9_p2w2.py from the SHIPPED
`_gqa_kernel_prefill_expert` (gqa.py:1997-2468) and
`_gqa_kernel_prefill_g2stack` (gqa.py:2471-2969) blocks of
examples/gqa/gqa.py (md5 3aa9cfa2625cdce96a62ce967785fe85) via 2 line-scoped
edits per kernel — see diff_iter9_p2w2.txt in this directory for the complete
unified diff (the audit artifact).  gqa.py is NOT modified by this file;
constants are imported read-only.

PER-HEAD `_gqa_kernel_expert` IS DELIBERATELY ABSENT FROM THIS MODULE
(iter9 Phase-B verdict — structural infeasibility, NOT a budget cut):
the AUTO-mode threads=2 (serial/causal) form mis-lowers the strided BSND
output write.  AscendVidReduction splits the contiguous BHSD write correctly
(one `copy_ub_to_gm<half,128,32>(..., 128, 32, 128)` per vid, blockCount=32,
dst offset vid*half*D) but for the strided BSND box it emits the UNSPLIT
blockCount=block_M (`(..., 1024, 64, 128)`, blockCount=64, dstStride=1792B)
WHILE KEEPING the contiguous-layout vid offset (vid*4096): both vids then
write overlapping 64-row windows 4 rows apart => corruption.  Measured:
codegen_audit_iter9.log (descriptor diff) + bit_probe_iter9.log (r1/c5
BITFAIL maxabsdiff 5.9e4/NaN; the threads=1 PIPELINED form is BITOK but no
bench per-head case lands there — r1/c5/c6 are all causal => threads=2).
Any attempt to getattr(VAR, "_gqa_kernel_expert") MUST fail loudly.

Mechanism (identical to the ADOPTED iter8 wave-1, p2_bsnd_design_iter8.md):
Output declared BSND [B, S_padded, N_q, D]; the FIN write box becomes
`Output[bz, rows, hq, :]` — a strided MTE3 box (row stride N_q*D) carrying
exactly the element set transpose_out's _transpose_v1 used to move, so the
host skips the transpose_out launch on the covered domain (aligned S:
S_padded == S, no crop, no masked rows).  Both kernels here are EXPERT
manual-flag kernels: the FIN write stays inside its shipped SIG_S_HALF
V<->MTE3 flag pair, every flag line byte-identical (audited), box
[half=32, D=128] exactly as wave-1; the generated code keeps blockCount=32
per vid with the correctly recomputed BSND vid offset (vid*half*N_q*D) and
dstStride=(N_q*D-D)*sizeof(T) — verified in codegen_audit_iter9.log for the
c19 (prefill_expert ns=1) and c9/c8 (g2stack ns=4/ns=8) configurations.

Arithmetic is UNTOUCHED => final outputs must be BIT-IDENTICAL to the
shipped main+transpose_out chain (Gate 0 criterion; special values: NaN
positions equal, Inf positions+signs equal, finite bits equal — raw-bit
compare, iter8 harness finding 3).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tilelang
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout

from gqa import P28_EXPERT_PASS_CONFIGS, P28_NUM_CORES, _BIG_NEG  # noqa: E402,F401


'''


def load_blocks():
    md5 = hashlib.md5(open(GQA_PY, "rb").read()).hexdigest()
    assert md5 == EXPECT_MD5, f"gqa.py drifted: md5 {md5} != {EXPECT_MD5}"
    lines = open(GQA_PY).read().splitlines(keepends=True)
    out = {}
    for tag, (start, end, fname) in BLOCKS.items():
        block = lines[start - 1 : end]
        # boundary assertions (fail loudly on drift)
        assert block[0].startswith("@tilelang.jit("), block[0]
        assert any(ln.startswith(f"def {fname}(") for ln in block[:12]), fname
        assert block[-1].rstrip() == "    return main", block[-1]
        # edit-site pins
        s = SITES[tag]
        d = EDITS[tag]["decl"] - start
        w = EDITS[tag]["write"] - start
        assert block[d] == s["decl_old"], f"{tag} decl site drifted:\n{block[d]!r}"
        assert block[w] == s["write_old"], f"{tag} write site drifted:\n{block[w]!r}"
        # uniqueness of the pinned patterns inside the block
        whole = "".join(block)
        assert whole.count(s["decl_old"]) == 1, f"{tag}: {whole.count(s['decl_old'])} decl sites"
        assert whole.count(s["write_old"]) == 1, f"{tag}: {whole.count(s['write_old'])} write sites"
        out[tag] = block
    return out


def make_w2(block, tag):
    lines = list(block)
    start = BLOCKS[tag][0]
    s = SITES[tag]
    lines[EDITS[tag]["decl"] - start] = s["decl_new"]
    lines[EDITS[tag]["write"] - start] = s["write_new"]
    return lines


def audit(variant_lines, block, tag):
    """Static re-audit: the variant differs from shipped in EXACTLY the 2 pinned sites."""
    text = "".join(variant_lines)
    ship = "".join(block)
    s = SITES[tag]

    # (1) the two edits landed
    assert text.count("Output: T.Tensor([B, S_padded, N_q, D], dtype)") == 1, "decl edit missing"
    assert "Output: T.Tensor([B, N_q, S_padded, D], dtype)" not in text, "old BHSD decl survives"
    wnew_code = [ln for ln in s["write_new"].split("\n") if ln.strip() and not ln.strip().startswith("#")][-1]
    assert text.count(wnew_code.strip()) == 1, "write edit missing"
    assert s["write_old"].strip() not in text, "old BHSD write survives"

    # (2) flag protocol byte-identical (expert kernels; empty == empty on the
    # per-head AUTO kernel, which has no manual flag lines)
    def flaglines(st):
        return [ln for ln in st.splitlines()
                if any(k in ln for k in ("set_flag", "wait_flag", "set_cross_flag", "wait_cross_flag"))]

    assert flaglines(ship) == flaglines(text), "flag protocol lines diverged"
    if tag != "perhead":
        assert len(flaglines(ship)) > 0, "expert kernel without flag lines?"
        for k in ("SEM_WS1_C2V", "SEM_WS1_V2C", "SEM_WS2_V2C", "SEM_WS2_C2V", "SEM_WS3_C2V",
                  "SEM_WS3_V2C", "SIG_K_L1", "SIG_V_L1", "SIG_P_L1", "SIG_Q_L1", "SIG_L0AB",
                  "SIG_L0C", "SIG_IO_UB", "SIG_MASK", "SIG_S_HALF"):
            assert ship.count(k + " = ") == text.count(k + " = "), k

    # (3) memory geometry identical
    for pat in ("T.alloc_shared(", "T.alloc_L1(", "T.alloc_L0A(", "T.alloc_L0B(", "T.alloc_L0C(",
                "ws_s: T.Tensor", "ws_p: T.Tensor", "ws_o: T.Tensor",
                "Q: T.Tensor", "K: T.Tensor", "V: T.Tensor", "Mask: T.Tensor",
                "make_zn_layout", "make_nz_layout", "T.annotate_layout"):
        assert ship.count(pat) == text.count(pat), pat

    # (3b) per-head AUTO schedule pins (no manual protocol to byte-compare;
    # these host lines drive the AUTO codegen's launch form and pipelining,
    # so pinning them + the generated-code audit closes the schedule surface)
    if tag == "perhead":
        for pat in ("use_pipeline = ", "pipe_stages = ", "kernel_threads = ", "skip_limit = ",
                    "T.Kernel(block_num, threads=kernel_threads, is_npu=True)",
                    "T.Pipelined(kv_iters, num_stages=pipe_stages)", "m_init_floor = "):
            assert ship.count(pat) == text.count(pat), f"perhead schedule pin drifted: {pat}"

    # (4) arithmetic untouched: every removed line must be one of the 2 pinned
    # originals and every added line must be a comment or one of the 2 pinned
    # replacements.  Site order is file order: in all three iter9 kernels the
    # decl site precedes the write site (the opposite of the iter8 flat
    # kernels, whose emit_fin macro sits above the main signature).
    decl_first = EDITS[tag]["decl"] < EDITS[tag]["write"]
    exp_removed = [s["decl_old"], s["write_old"]] if decl_first else [s["write_old"], s["decl_old"]]
    diff = list(difflib.unified_diff(ship.splitlines(), text.splitlines(), lineterm=""))
    removed = [ln[1:] + "\n" for ln in diff if ln.startswith("-") and not ln.startswith("---")]
    added = [ln[1:] + "\n" for ln in diff if ln.startswith("+") and not ln.startswith("+++")]
    assert removed == exp_removed, f"unexpected removed lines: {removed}"
    first_new, second_new = (s["decl_new"], s["write_new"]) if decl_first else (s["write_new"], s["decl_new"])
    exp_added = [ln + "\n" for ln in first_new.split("\n") if ln] + \
                [ln + "\n" for ln in second_new.split("\n") if ln]
    assert added == exp_added, f"unexpected added lines:\n{added}\nvs\n{exp_added}"

    # code-only projection (comments stripped) must differ in exactly 2 lines
    def code_only(st):
        out = []
        for ln in st.splitlines():
            stp = ln.strip()
            if not stp or stp.startswith("#"):
                continue
            out.append(ln)
        return out

    cs, cv = code_only(ship), code_only(text)
    assert len(cs) == len(cv), f"code line count {len(cs)} != {len(cv)}"
    changed = [(a, b) for a, b in zip(cs, cv) if a != b]
    assert len(changed) == 2, f"expected 2 changed code lines, got {changed}"

    def last_code(st):
        return [ln for ln in st.split("\n") if ln.strip() and not ln.strip().startswith("#")][-1].strip()

    first_old, second_old = (s["decl_old"], s["write_old"]) if decl_first else [s["write_old"], s["decl_old"]]
    assert changed[0][0].strip() == first_old.strip(), changed[0][0]
    assert changed[0][1].strip() == last_code(first_new), changed[0][1]
    assert changed[1][0].strip() == second_old.strip(), changed[1][0]
    assert changed[1][1].strip() == last_code(second_new), changed[1][1]

    # (5) task decomposition / schedule untouched
    for pat in ("def task_range", "T.serial(", "T.Pipelined(", "T.Kernel("):
        assert ship.count(pat) == text.count(pat), pat

    # (6) diff hunk census: exactly 2 hunks (decl + write)
    hunks = [ln for ln in diff if ln.startswith("@@")]
    assert len(hunks) == 2, f"unexpected diff spread ({len(hunks)} hunks): {hunks}"
    return diff


def main():
    # per-head EXCLUDED (iter9 Phase-B verdict): the AUTO threads=2 form
    # mis-lowers the strided BSND write — AscendVidReduction keeps the
    # contiguous-layout vid offset (vid*half*D) while emitting the UNSPLIT
    # blockCount=block_M, so both vids write overlapping 64-row windows
    # (codegen_audit_iter9.py: blockCount 32->64 + stale vid*4096;
    # bit_probe_iter9.log: r1/c5 BITFAIL maxabsdiff 5.9e4/NaN, threads=1
    # pipelined form BITOK).  Every bench per-head case (r1/c5/c6) is causal
    # => threads=2 => broken.  Evidence: diff_iter9_p2w2_with_perhead_pre_verdict.txt,
    # codegen_iter9_perhead_*_{base,w2}.cu.  The variant module below carries
    # ONLY the two expert manual-flag kernels (mechanism proven by wave-1).
    blocks = load_blocks()
    all_diff = []
    parts = []
    for tag in ("prexpert", "g2stack"):
        variant = make_w2(blocks[tag], tag)
        diff = audit(variant, blocks[tag], tag)
        all_diff.append(f"### {tag} — {BLOCKS[tag][2]} (gqa.py:{BLOCKS[tag][0]}-{BLOCKS[tag][1]})")
        all_diff.extend(diff)
        all_diff.append("")
        parts.append("".join(variant))
        print(f"[{tag}] audit OK: 2 edits, flag/schedule lines identical, geometry identical, "
              f"code-only diff exactly 2 lines, 2 diff hunks")
    dst = os.path.join(HERE, "gqa_w2_bsnd_iter9_variant.py")
    with open(dst, "w") as f:
        f.write(HEADER)
        f.write("\n\n".join(parts))
    with open(os.path.join(HERE, "diff_iter9_p2w2.txt"), "w") as f:
        f.write("\n".join(all_diff) + "\n")
    import ast

    ast.parse(open(dst).read())
    print(f"WROTE {dst}")
    print(f"WROTE {os.path.join(HERE, 'diff_iter9_p2w2.txt')} ({len(all_diff)} lines)")


if __name__ == "__main__":
    main()
