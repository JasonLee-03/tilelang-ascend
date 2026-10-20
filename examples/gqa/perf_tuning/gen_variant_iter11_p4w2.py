"""Iter11 P4 wave-2 variant generator — g2stack + prefill_expert with BSND
strided INPUT reads (transpose_qkv elimination, wave-2 of the iter10 P4
mechanism).

Extracts the TWO SHIPPED factory blocks from examples/gqa/gqa.py
(md5-asserted f59b5612..., line-anchored) and applies the minimal wave-2 edit
— 5 line-scoped edit sites per kernel, the IDENTICAL mechanism already ADOPTED
for the two flat kernels in iter10 — so each variant is provably "shipped
kernel + tiny diff" (gen_variant_iter8/9/10 pattern):

    perf_tuning/gqa_p4w2_inbsnd_iter11_variant.py  +  diff_iter11_p4w2.txt

Mechanism (see p4_wave2_design_iter11.md for the per-box algebra):
  * Q/K/V declarations switch from BHSD [B, N, S_pad, D] to BSND
    [B, S_pad, N, D] when in_bsnd=True (factory kwarg, default False —
    shipped codegen byte-identical when off).
  * Load boxes switch from `X[bz, head, rows, :]` (contiguous, row stride D)
    to `X[bz, rows, head, :]` (strided, row stride N*D).  Lowering:
    compute_strideN folds the singleton head dim into realSrcN — ONE
    copy_gm_to_l1 per box, count unchanged (probe_iter10_P4.md §1).
  * g2stack's stacked-M Q pair: the M_stack dim is DST-side only (static q_l1
    slices, byte-identical across arms); each box has the same per-box source
    extents [1, BM, 1, D] as the wave-1 flat emit_qload — the algebraically
    identical two-box form is already adopted+audited (design §2.1).
  * The K/V sites sit inside RUNTIME `if i % 2 == 0:` parity branches with the
    runtime idx window — the target BSND form is LITERALLY the decode
    dense_expert production pattern (gqa.py:1228-1231/:1265-1268).
  * NOTHING else changes: every set/wait flag line byte-identical (audited),
    slots/depths/step order untouched (iter4 pacing-is-load-bearing), the
    iter9 out_bsnd FIN mechanism untouched, UB/L1/L0/ws geometry
    byte-identical, arithmetic untouched.

Coverage gate (host side, applied by harness/integration, NOT by the kernel):
S % 64 == 0 AND S_kv % 128 == 0 (wave-1 dual-alignment gate, unchanged).

PER-HEAD / AUTO domains remain DELIBERATELY ABSENT (iter9 finding 1 + iter10
finding 5: AUTO vid-reduction strided mis-lowering demonstrated on writes,
untested on reads — conservative exclusion).  The flat kv1/kv2 kernels are
wave-1 territory and are NOT touched here.

Run:  python3 examples/gqa/perf_tuning/gen_variant_iter11_p4w2.py
"""

import difflib
import hashlib
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_DIR = os.path.dirname(HERE)
GQA_PY = os.path.join(GQA_DIR, "gqa.py")

EXPECT_MD5 = "f59b561212e60a50b78a51671999ef7c"

# 1-based inclusive line ranges of the shipped factory blocks (decorator..return main)
BLOCKS = {
    "prexpert": (1997, 2500, "_gqa_kernel_prefill_expert"),
    "g2stack": (2503, 3028, "_gqa_kernel_prefill_g2stack"),
}

# pinned edit sites (absolute gqa.py line numbers; decl/qload/k/v are ranges).
# SITE ORDER in these kernels: sig < decl < qload < k < v  (the flat kernels'
# order was sig < qload < k < v < decl — decl moved ahead of the load sites).
EDITS = {
    "prexpert": {"sig": 2014, "decl": (2129, 2135), "qload": (2207, 2210), "k": (2234, 2243), "v": (2277, 2286)},
    "g2stack": {"sig": 2520, "decl": (2637, 2643), "qload": (2722, 2729), "k": (2750, 2759), "v": (2793, 2802)},
}

_SIG_OLD = "    out_bsnd=False,\n"
_SIG_NEW = "    out_bsnd=False,\n    in_bsnd=False,\n"

# ---- decl: out_shape line .. V declaration line (identical text both kernels) ----
_DECL_OLD = (
    "    out_shape = [B, S_padded, N_q, D] if out_bsnd else [B, N_q, S_padded, D]\n"
    "\n"
    "    @T.prim_func\n"
    "    def main(\n"
    "        Q: T.Tensor([B, N_q, S_padded, D], dtype),  # type: ignore  BHSD\n"
    "        K: T.Tensor([B, N_kv, S_kv_padded, D], dtype),  # type: ignore  BHSD\n"
    "        V: T.Tensor([B, N_kv, S_kv_padded, D], dtype),  # type: ignore  BHSD\n"
)
_DECL_NEW = (
    "    out_shape = [B, S_padded, N_q, D] if out_bsnd else [B, N_q, S_padded, D]\n"
    "\n"
    "    # [S3-iter11-P4w2] Input layout: BSND when the host skips transpose_qkv\n"
    "    # (caller-gated on S_padded == S AND S_kv_padded == S_kv — aligned\n"
    "    # shapes only, so the strided load windows never leave the real tensor\n"
    "    # and no zero-fill pad semantics are lost), else the shipped BHSD.\n"
    "    qkv_q_shape = [B, S_padded, N_q, D] if in_bsnd else [B, N_q, S_padded, D]\n"
    "    qkv_kv_shape = [B, S_kv_padded, N_kv, D] if in_bsnd else [B, N_kv, S_kv_padded, D]\n"
    "\n"
    "    @T.prim_func\n"
    "    def main(\n"
    "        Q: T.Tensor(qkv_q_shape, dtype),  # type: ignore  BSND if in_bsnd else BHSD\n"
    "        K: T.Tensor(qkv_kv_shape, dtype),  # type: ignore  ditto\n"
    "        V: T.Tensor(qkv_kv_shape, dtype),  # type: ignore  ditto\n"
)

# ---- qload: prexpert single box ----
_QLOAD_OLD_PE = (
    "                    T.copy(\n"
    "                        Q[bz, hq, bx * block_M : (bx + 1) * block_M, :],\n"
    "                        q_l1,\n"
    "                    )\n"
)
_QLOAD_NEW_PE = (
    "                    if in_bsnd:\n"
    "                        # [S3-iter11-P4w2] BSND strided Q box (bz, rows, hq, :)\n"
    "                        # — same element set as the BHSD box, row stride\n"
    "                        # N_q*D; ONE copy_gm_to_l1, count unchanged (decode\n"
    "                        # dense_expert precedent; wave-1 flat emit_qload is\n"
    "                        # the same algebra, p4_wave2_design_iter11.md §3).\n"
    "                        T.copy(\n"
    "                            Q[bz, bx * block_M : (bx + 1) * block_M, hq, :],\n"
    "                            q_l1,\n"
    "                        )\n"
    "                    else:\n"
    "                        T.copy(\n"
    "                            Q[bz, hq, bx * block_M : (bx + 1) * block_M, :],\n"
    "                            q_l1,\n"
    "                        )\n"
)

# ---- qload: g2stack stacked pair (dst slices byte-identical across arms) ----
_QLOAD_OLD_G2 = (
    "                    T.copy(\n"
    "                        Q[bz, hq0, bx * block_M : (bx + 1) * block_M, :],\n"
    "                        q_l1[0:block_M, :],\n"
    "                    )\n"
    "                    T.copy(\n"
    "                        Q[bz, hq0 + 1, bx * block_M : (bx + 1) * block_M, :],\n"
    "                        q_l1[block_M:M_stack, :],\n"
    "                    )\n"
)
_QLOAD_NEW_G2 = (
    "                    if in_bsnd:\n"
    "                        # [S3-iter11-P4w2] BSND strided Q boxes (bz, rows,\n"
    "                        # hq, :) — same element sets, row stride N_q*D; ONE\n"
    "                        # copy_gm_to_l1 per box, count unchanged.  The\n"
    "                        # stacked M dim is dst-side only (static q_l1 slices,\n"
    "                        # byte-identical across arms); each box's source\n"
    "                        # extents [1, BM, 1, D] fold to realSrcN = N_q*D —\n"
    "                        # the wave-1 flat emit_qload is the identical\n"
    "                        # two-box form (p4_wave2_design_iter11.md §2.1).\n"
    "                        T.copy(\n"
    "                            Q[bz, bx * block_M : (bx + 1) * block_M, hq0, :],\n"
    "                            q_l1[0:block_M, :],\n"
    "                        )\n"
    "                        T.copy(\n"
    "                            Q[bz, bx * block_M : (bx + 1) * block_M, hq0 + 1, :],\n"
    "                            q_l1[block_M:M_stack, :],\n"
    "                        )\n"
    "                    else:\n"
    "                        T.copy(\n"
    "                            Q[bz, hq0, bx * block_M : (bx + 1) * block_M, :],\n"
    "                            q_l1[0:block_M, :],\n"
    "                        )\n"
    "                        T.copy(\n"
    "                            Q[bz, hq0 + 1, bx * block_M : (bx + 1) * block_M, :],\n"
    "                            q_l1[block_M:M_stack, :],\n"
    "                        )\n"
)


def _kv_old(op, l1):
    return (
        "                            if i % 2 == 0:\n"
        "                                T.copy(\n"
        f"                                    {op}[bz, kg, idx * block_N : (idx + 1) * block_N, :],\n"
        f"                                    {l1}_0,\n"
        "                                )\n"
        "                            else:\n"
        "                                T.copy(\n"
        f"                                    {op}[bz, kg, idx * block_N : (idx + 1) * block_N, :],\n"
        f"                                    {l1}_1,\n"
        "                                )\n"
    )


def _kv_new(op, l1, comment):
    head = (
        "                            if i % 2 == 0:\n"
        "                                if in_bsnd:\n"
    )
    if comment:
        head += (
            f"                                    # [S3-iter11-P4w2] BSND strided box\n"
            f"                                    # (bz, kv-rows, kg, :), row stride\n"
            f"                                    # N_kv*D — decode dense_expert\n"
            f"                                    # {op}-load production precedent\n"
            f"                                    # (gqa.py:{comment}, identical\n"
            f"                                    # runtime-idx parity form).\n"
            )
    return (
        head
        + "                                    T.copy(\n"
        f"                                        {op}[bz, idx * block_N : (idx + 1) * block_N, kg, :],\n"
        f"                                        {l1}_0,\n"
        "                                    )\n"
        "                                else:\n"
        "                                    T.copy(\n"
        f"                                        {op}[bz, kg, idx * block_N : (idx + 1) * block_N, :],\n"
        f"                                        {l1}_0,\n"
        "                                    )\n"
        "                            else:\n"
        "                                if in_bsnd:\n"
        "                                    T.copy(\n"
        f"                                        {op}[bz, idx * block_N : (idx + 1) * block_N, kg, :],\n"
        f"                                        {l1}_1,\n"
        "                                    )\n"
        "                                else:\n"
        "                                    T.copy(\n"
        f"                                        {op}[bz, kg, idx * block_N : (idx + 1) * block_N, :],\n"
        f"                                        {l1}_1,\n"
        "                                    )\n"
    )


_K_OLD = _kv_old("K", "k_l1")
_K_NEW = _kv_new("K", "k_l1", "1229")
_V_OLD = _kv_old("V", "v_l1")
_V_NEW = _kv_new("V", "v_l1", "1266")

SITES = {
    "prexpert": {"sig": (_SIG_OLD, _SIG_NEW), "decl": (_DECL_OLD, _DECL_NEW),
                 "qload": (_QLOAD_OLD_PE, _QLOAD_NEW_PE), "k": (_K_OLD, _K_NEW), "v": (_V_OLD, _V_NEW)},
    "g2stack": {"sig": (_SIG_OLD, _SIG_NEW), "decl": (_DECL_OLD, _DECL_NEW),
                "qload": (_QLOAD_OLD_G2, _QLOAD_NEW_G2), "k": (_K_OLD, _K_NEW), "v": (_V_OLD, _V_NEW)},
}

HEADER = '''"""P4 wave-2 variant — g2stack + prefill_expert with BSND strided INPUT reads
(transpose_qkv elimination).

Standalone precheck/Gate-0/Gate-2 development file (Stage 3 iter11).  Generated
by gen_variant_iter11_p4w2.py from the SHIPPED
`_gqa_kernel_prefill_expert` (gqa.py:1997-2500) and
`_gqa_kernel_prefill_g2stack` (gqa.py:2503-3028) blocks of
examples/gqa/gqa.py (md5 f59b561212e60a50b78a51671999ef7c) via 5 line-scoped
edit sites per kernel — see diff_iter11_p4w2.txt in this directory for the
complete unified diff (the audit artifact).  gqa.py is NOT modified by this
file; constants are imported read-only.

Mechanism: the ADOPTED iter10 wave-1 in_bsnd pattern (flat kv1/kv2) extended
to the two ns-batched expert kernels — in_bsnd=True declares Q/K/V as the
caller's BSND tensors and switches the load boxes from `X[bz, head, rows, :]`
(contiguous, row stride D) to `X[bz, rows, head, :]` (strided, row stride
N*D).  compute_strideN folds the singleton head dim into copy_gm_to_l1's
realSrcN; ND->zN/zn is the MTE2 on-the-fly transform (probe_iter10_P4.md §1-2).
The K/V target form (runtime parity branch + runtime idx window + BSND strided
box) is LITERALLY the decode dense_expert production pattern
(gqa.py:1228-1231/:1265-1268).  g2stack's stacked-M Q pair keeps the M_stack
dim dst-side only — per-box source algebra identical to the adopted wave-1
flat emit_qload (p4_wave2_design_iter11.md §2.1).

Bit-identity argument: on the ALIGNED domain (S%64==0, S_kv%128==0 — the host
gate) transpose_qkv is a pure rearrangement with NO zero-fill and NO value
change; element (b, head, s, d) of its BHSD output == element (b, s, head, d)
of its BSND input.  The variant reads the identical element set from the
identical source addresses into identical L1 zn/nz tiles => every downstream
mma/softmax/FIN input is bit-identical => final outputs BIT-IDENTICAL to the
shipped transpose_qkv+main chain (Gate 0 criterion; raw-bit compare, iter8
finding 3).

Untouched (audited): every set/wait flag line byte-identical; slot depths,
step order, token counts, causal runtime kv_useful bounds, use_mask branches
(iter4 pacing-is-load-bearing); UB/L1/L0/ws geometry; arithmetic; the iter9
out_bsnd FIN write.  in_bsnd=False reproduces the shipped codegen
byte-for-byte (the else branches keep the shipped lines verbatim).

PER-HEAD / AUTO domains stay excluded (iter9 finding 1 / iter10 finding 5).
The flat kv1/kv2 kernels are wave-1 territory — untouched here.
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
        # edit-site pins: line numbers AND exact text AND uniqueness
        whole = "".join(block)
        for site, (lo, hi) in _site_ranges(tag).items():
            old = SITES[tag][site][0]
            seg = "".join(lines[lo - 1 : hi])
            assert seg == old, f"{tag}.{site} site drifted at {lo}-{hi}:\n{seg!r}\nvs\n{old!r}"
            assert whole.count(old) == 1, f"{tag}.{site}: {whole.count(old)} occurrences in block"
        out[tag] = block
    return out


def _site_ranges(tag):
    e = EDITS[tag]
    return {
        "sig": (e["sig"], e["sig"]),
        "decl": e["decl"],
        "qload": e["qload"],
        "k": e["k"],
        "v": e["v"],
    }


def make_w2(block, tag):
    text = "".join(block)
    for site in ("sig", "decl", "qload", "k", "v"):
        old, new = SITES[tag][site]
        assert text.count(old) == 1, f"{tag}.{site}: {text.count(old)} occurrences"
        text = text.replace(old, new)
    return text.splitlines(keepends=True)


def audit(variant_lines, block, tag):
    """Static re-audit: the variant differs from shipped ONLY in the 5 pinned
    sites; flags/geometry/arithmetic byte-identical (wave-1 audit verbatim,
    site order adapted: sig < decl < qload < k < v)."""
    text = "".join(variant_lines)
    ship = "".join(block)

    # (1) all five edits landed
    assert text.count("    in_bsnd=False,\n") == 1, "sig edit missing"
    assert text.count("qkv_q_shape = [B, S_padded, N_q, D] if in_bsnd") == 1, "decl edit missing"
    assert text.count("Q: T.Tensor(qkv_q_shape, dtype)") == 1, "Q decl edit missing"
    assert "Q: T.Tensor([B, N_q, S_padded, D], dtype)" not in text, "old BHSD Q decl survives"
    if tag == "g2stack":
        for pat_new in ("Q[bz, bx * block_M : (bx + 1) * block_M, hq0, :]",
                        "Q[bz, bx * block_M : (bx + 1) * block_M, hq0 + 1, :]"):
            assert pat_new in text, f"new strided stacked Q box missing: {pat_new}"
    else:
        assert "Q[bz, bx * block_M : (bx + 1) * block_M, hq, :]" in text, "new strided Q box missing"
    for pat_new in ("K[bz, idx * block_N : (idx + 1) * block_N, kg, :]",
                    "V[bz, idx * block_N : (idx + 1) * block_N, kg, :]"):
        assert pat_new in text, f"new strided box missing: {pat_new}"
    # shipped BHSD boxes preserved verbatim inside else branches
    for pat_old in ("K[bz, kg, idx * block_N : (idx + 1) * block_N, :]",
                    "V[bz, kg, idx * block_N : (idx + 1) * block_N, :]"):
        assert text.count(pat_old) == 2, f"shipped BHSD box not preserved 2x (parity branches): {pat_old}"

    # (2) flag protocol byte-identical
    def flaglines(st):
        return [ln for ln in st.splitlines()
                if any(k in ln for k in ("set_flag", "wait_flag", "set_cross_flag", "wait_cross_flag"))]

    assert flaglines(ship) == flaglines(text), "flag protocol lines diverged"
    assert len(flaglines(ship)) > 0, "expert kernel without flag lines?"
    for k in ("SEM_WS1_C2V", "SEM_WS1_V2C", "SEM_WS2_V2C", "SEM_WS2_C2V", "SEM_WS3_C2V",
              "SEM_WS3_V2C", "SIG_K_L1", "SIG_V_L1", "SIG_P_L1", "SIG_Q_L1", "SIG_L0AB",
              "SIG_L0C", "SIG_IO_UB", "SIG_MASK", "SIG_S_HALF"):
        assert ship.count(k + " = ") == text.count(k + " = "), k

    # (3) memory geometry identical (L1/L0/UB allocs, layouts, ws + Output decls)
    for pat in ("T.alloc_L1(", "T.alloc_ub(", "T.alloc_L0A(", "T.alloc_L0B(", "T.alloc_L0C(",
                "ws_s: T.Tensor", "ws_p: T.Tensor", "ws_o: T.Tensor",
                "Mask: T.Tensor", "Output: T.Tensor(out_shape",
                "make_zn_layout", "make_nz_layout", "T.annotate_layout"):
        assert ship.count(pat) == text.count(pat), pat

    # (4) exact-diff projection: every removed line must belong to the 5
    # pinned ORIGINAL site blocks and every added line to the 5 pinned
    # REPLACEMENT blocks (difflib subset projection; reconstruction of the
    # variant from (ship, diff) is exact by difflib semantics).
    diff = list(difflib.unified_diff(ship.splitlines(), text.splitlines(), lineterm=""))
    removed = [ln[1:] + "\n" for ln in diff if ln.startswith("-") and not ln.startswith("---")]
    added = [ln[1:] + "\n" for ln in diff if ln.startswith("+") and not ln.startswith("+++")]
    order = ["sig", "decl", "qload", "k", "v"]  # file order in BOTH kernels here
    e = EDITS[tag]
    assert e["sig"] < e["decl"][0] < e["qload"][0] < e["k"][0] < e["v"][0], "site order drifted"
    old_pool, new_pool = [], []
    for site in order:
        old, new = SITES[tag][site]
        old_pool.extend(old.splitlines(keepends=True))
        new_pool.extend(new.splitlines(keepends=True))
    old_c, new_c = Counter(old_pool), Counter(new_pool)
    rem_c, add_c = Counter(removed), Counter(added)
    for ln, n in rem_c.items():
        assert old_c[ln] >= n, f"removed line not from a pinned site ({n}x): {ln!r}"
    for ln, n in add_c.items():
        assert new_c[ln] >= n, f"added line not from a pinned site ({n}x): {ln!r}"
    for must in ("    in_bsnd=False,\n",):
        assert add_c[must] == 1, f"expected insertion missing: {must!r}"
    # the shipped box lines survive ONLY inside else branches (re-indented),
    # so the ORIGINAL indentation forms must show up as removed
    for must in ("                                T.copy(\n",):
        assert rem_c[must] >= 4, f"expected K/V removals missing: {must!r} ({rem_c[must]})"
    assert rem_c["        Q: T.Tensor([B, N_q, S_padded, D], dtype),  # type: ignore  BHSD\n"] >= 1

    # (5) task decomposition / schedule / arithmetic untouched
    for pat in ("def task_range", "T.serial(", "T.Kernel(", "T.mma(",
                "T.reduce_max", "T.reduce_sum", "T.tile.", "T.if_then_else(",
                "kv_useful", "batch_iters", "use_mask", "out_bsnd"):
        assert ship.count(pat) == text.count(pat), pat

    # (6) diff hunk census: 5 sites, all separated by >6 unchanged lines in
    # these kernels (sig..decl ~115, decl..qload ~78, qload..k ~24, k..v ~34)
    # => expect exactly 5 hunks (6 tolerated if difflib context merges oddly)
    hunks = [ln for ln in diff if ln.startswith("@@")]
    assert len(hunks) in (5, 6), f"unexpected diff spread ({len(hunks)} hunks): {hunks}"
    return diff


def main():
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
        print(f"[{tag}] audit OK: 5 edit sites, flag lines byte-identical, geometry identical, "
              f"exact-diff projection clean, {len([l for l in diff if l.startswith(chr(64)*2)])} diff hunks")
    dst = os.path.join(HERE, "gqa_p4w2_inbsnd_iter11_variant.py")
    with open(dst, "w") as f:
        f.write(HEADER)
        f.write("\n\n".join(parts))
    with open(os.path.join(HERE, "diff_iter11_p4w2.txt"), "w") as f:
        f.write("\n".join(all_diff) + "\n")
    import ast

    ast.parse(open(dst).read())
    print(f"WROTE {dst}")
    print(f"WROTE {os.path.join(HERE, 'diff_iter11_p4w2.txt')} ({len(all_diff)} lines)")


if __name__ == "__main__":
    main()
