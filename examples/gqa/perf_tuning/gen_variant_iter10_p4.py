"""Iter10 P4 variant generator — flat kv1/kv2 with BSND strided INPUT reads
(transpose_qkv elimination probe / implementation).

Extracts the TWO SHIPPED flat kernel factory blocks from examples/gqa/gqa.py
(md5-asserted, line-anchored) and applies the minimal P4 edit — 5 line-scoped
edit sites per kernel — so each variant is provably "shipped kernel + tiny
diff" (gen_variant_iter8/9 pattern):

    perf_tuning/gqa_p4_inbsnd_iter10_variant.py  +  diff_iter10_p4.txt

Mechanism (mirror of the ADOPTED iter8/9 BSND OUTPUT mechanism, now on the
INPUT side; precedent = the decode dense_expert kernel, gqa.py:1229/:1266,
which has read K/V via strided BSND boxes straight into nz/zn L1 IN
PRODUCTION since before Stage 3):
  * Q/K/V declarations switch from BHSD [B, N, S_pad, D] to BSND
    [B, S_pad, N, D] when in_bsnd=True (factory kwarg, default False —
    shipped codegen byte-identical when off, wave-1 out_shape pattern).
  * The load boxes switch from `K[bz, kg, rows, :]` (contiguous, row stride
    D) to `K[bz, rows, kg, :]` (strided, row stride N_kv*D).  Same element
    set (transpose_qkv is a pure rearrangement on the aligned domain:
    S_padded==S and S_kv_padded==S_kv, no zero-fill) => bit-identical L1
    content => bit-identical final output.
  * Lowering path (verified in source, probe_iter10_P4.md §1):
    src/op/ascend.cc compute_strideN folds the trailing singleton head dim
    into realSrcN (= D*N_kv / D*N_q) -> copy_gm_to_l1<T,dstM,dstN>(...,
    realSrcN, ...) -> Catlass TileCopyTla ND->zN fractal via MTE2 (the SAME
    helper the decode kernel's strided K/V boxes use).
  * NOTHING else changes: every set/wait flag line byte-identical (audited),
    slots/depths/step order untouched (iter4 pacing-is-load-bearing),
    UB/L1/L0/ws geometry byte-identical, arithmetic untouched.

Coverage gate (host side, applied by harness/integration, NOT by the
kernel): S % 64 == 0 AND S_kv % 128 == 0, so the kernel reads exactly the
caller's BSND tensors — no zero-pad rows/cols are needed (the transpose_qkv
pad semantics stay on the shipped path for non-aligned shapes).

PER-HEAD / AUTO domains are DELIBERATELY ABSENT (iter9 Phase-B verdict:
AUTO vid-reduction mis-lowers strided boxes on the threads=2 form — the
defect was demonstrated for UB->GM writes; GM->L1 reads under AUTO vid
splitting are UNTESTED and conservatively excluded, same domain treatment).
g2stack / prefill_expert (wave-2 output domains) are NOT in this first wave
either (scope discipline: <=2 kernel families, the two flat factories share
one emit-macro structure).

Run:  python3 examples/gqa/perf_tuning/gen_variant_iter10_p4.py
"""

import difflib
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_DIR = os.path.dirname(HERE)
GQA_PY = os.path.join(GQA_DIR, "gqa.py")

EXPECT_MD5 = "606b07ff474ec42b8244f0831327d75b"

# 1-based inclusive line ranges of the shipped factory blocks (decorator..return main)
BLOCKS = {
    "flatkv1": (3040, 3530, "_gqa_kernel_prefill_g2stack_flat"),
    "flatkv2": (3547, 4143, "_gqa_kernel_prefill_g2stack_flat_kv2"),
}

# pinned edit sites (absolute gqa.py line numbers; qload/k/v/decl are ranges)
EDITS = {
    "flatkv1": {"sig": 3055, "qload": (3138, 3142), "k": (3154, 3157), "v": (3162, 3165), "decl": (3363, 3369)},
    "flatkv2": {"sig": 3562, "qload": (3648, 3652), "k": (3665, 3668), "v": (3674, 3677), "decl": (3912, 3918)},
}

_SIG_OLD = "    out_bsnd=False,\n"
_SIG_NEW = "    out_bsnd=False,\n    in_bsnd=False,\n"

# ---- qload: 5 shipped lines -> if in_bsnd / else (shipped verbatim) ----
_QLOAD_OLD = (
    "        T.copy(Q[bz, hq0, bx * block_M : (bx + 1) * block_M, :], q_l1[0:block_M, :])\n"
    "        T.copy(\n"
    "            Q[bz, hq0 + 1, bx * block_M : (bx + 1) * block_M, :],\n"
    "            q_l1[block_M:M_stack, :],\n"
    "        )\n"
)
_QLOAD_NEW = (
    "        if in_bsnd:\n"
    "            # [S3-iter10-P4] BSND strided Q boxes (bz, rows, hq, :) — same\n"
    "            # element set as the BHSD boxes, row stride N_q*D; ONE\n"
    "            # copy_gm_to_l1 per box, count unchanged (decode dense_expert\n"
    "            # K/V-load precedent, gqa.py:1229/:1266).\n"
    "            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, hq0, :], q_l1[0:block_M, :])\n"
    "            T.copy(\n"
    "                Q[bz, bx * block_M : (bx + 1) * block_M, hq0 + 1, :],\n"
    "                q_l1[block_M:M_stack, :],\n"
    "            )\n"
    "        else:\n"
    "            T.copy(Q[bz, hq0, bx * block_M : (bx + 1) * block_M, :], q_l1[0:block_M, :])\n"
    "            T.copy(\n"
    "                Q[bz, hq0 + 1, bx * block_M : (bx + 1) * block_M, :],\n"
    "                q_l1[block_M:M_stack, :],\n"
    "            )\n"
)

# ---- K/V boxes: kv1 (side branches, both windows 0:block_N) ----
_K_OLD_KV1 = (
    "        if side == 0:\n"
    "            T.copy(K[bz, kg, 0:block_N, :], k_l1_0)\n"
    "        else:\n"
    "            T.copy(K[bz, kg, 0:block_N, :], k_l1_1)\n"
)
_K_NEW_KV1 = (
    "        if side == 0:\n"
    "            if in_bsnd:\n"
    "                # [S3-iter10-P4] BSND strided box (bz, kv-rows, kg, :), row\n"
    "                # stride N_kv*D — decode dense_expert K-load precedent.\n"
    "                T.copy(K[bz, 0:block_N, kg, :], k_l1_0)\n"
    "            else:\n"
    "                T.copy(K[bz, kg, 0:block_N, :], k_l1_0)\n"
    "        else:\n"
    "            if in_bsnd:\n"
    "                T.copy(K[bz, 0:block_N, kg, :], k_l1_1)\n"
    "            else:\n"
    "                T.copy(K[bz, kg, 0:block_N, :], k_l1_1)\n"
)
_V_OLD_KV1 = (
    "        if side == 0:\n"
    "            T.copy(V[bz, kg, 0:block_N, :], v_l1_0)\n"
    "        else:\n"
    "            T.copy(V[bz, kg, 0:block_N, :], v_l1_1)\n"
)
_V_NEW_KV1 = (
    "        if side == 0:\n"
    "            if in_bsnd:\n"
    "                # [S3-iter10-P4] BSND strided box (bz, kv-rows, kg, :), row\n"
    "                # stride N_kv*D — decode dense_expert V-load precedent.\n"
    "                T.copy(V[bz, 0:block_N, kg, :], v_l1_0)\n"
    "            else:\n"
    "                T.copy(V[bz, kg, 0:block_N, :], v_l1_0)\n"
    "        else:\n"
    "            if in_bsnd:\n"
    "                T.copy(V[bz, 0:block_N, kg, :], v_l1_1)\n"
    "            else:\n"
    "                T.copy(V[bz, kg, 0:block_N, :], v_l1_1)\n"
)

# ---- K/V boxes: kv2 (kvpar branches, window depends on parity) ----
_K_OLD_KV2 = (
    "        if kvpar == 0:\n"
    "            T.copy(K[bz, kg, 0:block_N, :], k_l1_0)\n"
    "        else:\n"
    "            T.copy(K[bz, kg, block_N : 2 * block_N, :], k_l1_1)\n"
)
_K_NEW_KV2 = (
    "        if kvpar == 0:\n"
    "            if in_bsnd:\n"
    "                # [S3-iter10-P4] BSND strided box (bz, kv-rows, kg, :), row\n"
    "                # stride N_kv*D — decode dense_expert K-load precedent.\n"
    "                T.copy(K[bz, 0:block_N, kg, :], k_l1_0)\n"
    "            else:\n"
    "                T.copy(K[bz, kg, 0:block_N, :], k_l1_0)\n"
    "        else:\n"
    "            if in_bsnd:\n"
    "                T.copy(K[bz, block_N : 2 * block_N, kg, :], k_l1_1)\n"
    "            else:\n"
    "                T.copy(K[bz, kg, block_N : 2 * block_N, :], k_l1_1)\n"
)
_V_OLD_KV2 = (
    "        if kvpar == 0:\n"
    "            T.copy(V[bz, kg, 0:block_N, :], v_l1_0)\n"
    "        else:\n"
    "            T.copy(V[bz, kg, block_N : 2 * block_N, :], v_l1_1)\n"
)
_V_NEW_KV2 = (
    "        if kvpar == 0:\n"
    "            if in_bsnd:\n"
    "                # [S3-iter10-P4] BSND strided box (bz, kv-rows, kg, :), row\n"
    "                # stride N_kv*D — decode dense_expert V-load precedent.\n"
    "                T.copy(V[bz, 0:block_N, kg, :], v_l1_0)\n"
    "            else:\n"
    "                T.copy(V[bz, kg, 0:block_N, :], v_l1_0)\n"
    "        else:\n"
    "            if in_bsnd:\n"
    "                T.copy(V[bz, block_N : 2 * block_N, kg, :], v_l1_1)\n"
    "            else:\n"
    "                T.copy(V[bz, kg, block_N : 2 * block_N, :], v_l1_1)\n"
)

# ---- decl: out_shape line .. V declaration line ----
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
    "    # [S3-iter10-P4] Input layout: BSND when the host skips transpose_qkv\n"
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

SITES = {
    "flatkv1": {"sig": (_SIG_OLD, _SIG_NEW), "qload": (_QLOAD_OLD, _QLOAD_NEW),
                "k": (_K_OLD_KV1, _K_NEW_KV1), "v": (_V_OLD_KV1, _V_NEW_KV1),
                "decl": (_DECL_OLD, _DECL_NEW)},
    "flatkv2": {"sig": (_SIG_OLD, _SIG_NEW), "qload": (_QLOAD_OLD, _QLOAD_NEW),
                "k": (_K_OLD_KV2, _K_NEW_KV2), "v": (_V_OLD_KV2, _V_NEW_KV2),
                "decl": (_DECL_OLD, _DECL_NEW)},
}

HEADER = '''"""P4 variant — flat kv1/kv2 with BSND strided INPUT reads (transpose_qkv
elimination).

Standalone probe/Gate-0/Gate-2 development file (Stage 3 iter10).  Generated
by gen_variant_iter10_p4.py from the SHIPPED
`_gqa_kernel_prefill_g2stack_flat` (gqa.py:3040-3530) and
`_gqa_kernel_prefill_g2stack_flat_kv2` (gqa.py:3547-4143) blocks of
examples/gqa/gqa.py (md5 606b07ff474ec42b8244f0831327d75b) via 5 line-scoped
edit sites per kernel — see diff_iter10_p4.txt in this directory for the
complete unified diff (the audit artifact).  gqa.py is NOT modified by this
file; constants are imported read-only.

Mechanism (input-side mirror of the ADOPTED iter8/9 BSND output mechanism):
in_bsnd=True declares Q/K/V as the caller's BSND tensors and switches the
three load macros' boxes from `X[bz, head, rows, :]` (contiguous, row stride
D) to `X[bz, rows, head, :]` (strided, row stride N*D).  The lowering folds
the singleton head dim into copy_gm_to_l1's realSrcN (src/op/ascend.cc
compute_strideN) and the Catlass MTE2 TileCopy performs the ND->zN fractal
rearrangement — the SAME production path as the decode dense_expert kernel's
strided BSND K/V loads (gqa.py:1229/:1266, measured at 87-94% of the MTE2
roof, roofline_analysis_iter6.md §6.2).

Bit-identity argument: transpose_qkv on the ALIGNED domain (S%64==0,
S_kv%128==0 — the host gate) is a pure rearrangement with NO zero-fill and
NO value change; element (b, head, s, d) of its BHSD output == element
(b, s, head, d) of its BSND input.  The variant reads the identical element
set from the identical source addresses, into identical L1 zn/nz tiles, so
every downstream mma/softmax/FIN input is bit-identical => final outputs are
BIT-IDENTICAL to the shipped transpose_qkv+main chain (Gate 0 criterion;
special values via raw-bit compare, iter8 harness finding 3).

Untouched (audited): every set/wait flag line byte-identical; slot depths,
step order, token counts (iter4 pacing-is-load-bearing); UB/L1/L0/ws
geometry; arithmetic; the out_bsnd FIN write (shipped wave-1 mechanism).
in_bsnd=False reproduces the shipped codegen byte-for-byte (the else
branches keep the shipped lines verbatim).

PER-HEAD / AUTO domains are excluded (iter9 Phase-B verdict — AUTO
vid-reduction mis-lowers strided boxes; GM->L1 reads under AUTO vid
splitting untested, conservatively out of scope).  g2stack / prefill_expert
stay on the shipped BHSD-input path this wave (scope discipline).
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
        for site, (lo, hi) in _site_ranges(tag, start).items():
            old = SITES[tag][site][0]
            seg = "".join(lines[lo - 1 : hi])
            assert seg == old, f"{tag}.{site} site drifted at {lo}-{hi}:\n{seg!r}\nvs\n{old!r}"
            assert whole.count(old) == 1, f"{tag}.{site}: {whole.count(old)} occurrences in block"
        out[tag] = block
    return out


def _site_ranges(tag, start):
    e = EDITS[tag]
    return {
        "sig": (e["sig"], e["sig"]),
        "qload": e["qload"],
        "k": e["k"],
        "v": e["v"],
        "decl": e["decl"],
    }


def make_p4(block, tag):
    text = "".join(block)
    for site in ("sig", "qload", "k", "v", "decl"):
        old, new = SITES[tag][site]
        assert text.count(old) == 1, f"{tag}.{site}: {text.count(old)} occurrences"
        text = text.replace(old, new)
    return text.splitlines(keepends=True)


def audit(variant_lines, block, tag):
    """Static re-audit: the variant differs from shipped ONLY in the 5 pinned
    sites; flags/geometry/arithmetic byte-identical."""
    text = "".join(variant_lines)
    ship = "".join(block)

    # (1) all five edits landed
    assert text.count("    in_bsnd=False,\n") == 1, "sig edit missing"
    assert text.count("qkv_q_shape = [B, S_padded, N_q, D] if in_bsnd") == 1, "decl edit missing"
    assert text.count("Q: T.Tensor(qkv_q_shape, dtype)") == 1, "Q decl edit missing"
    assert "Q: T.Tensor([B, N_q, S_padded, D], dtype)" not in text, "old BHSD Q decl survives"
    for pat_new in ("T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, hq0, :]",
                    "T.copy(K[bz, 0:block_N, kg, :]",
                    "T.copy(V[bz, 0:block_N, kg, :]"):
        assert pat_new in text, f"new strided box missing: {pat_new}"
    if tag == "flatkv2":
        assert "T.copy(K[bz, block_N : 2 * block_N, kg, :], k_l1_1)" in text, "kv2 K1 box missing"
        assert "T.copy(V[bz, block_N : 2 * block_N, kg, :], v_l1_1)" in text, "kv2 V1 box missing"

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
    # REPLACEMENT blocks (difflib keeps common lines as context, so this is
    # a subset projection, not a sequence equality — reconstruction of the
    # variant from (ship, diff) is exact by difflib semantics).
    diff = list(difflib.unified_diff(ship.splitlines(), text.splitlines(), lineterm=""))
    removed = [ln[1:] + "\n" for ln in diff if ln.startswith("-") and not ln.startswith("---")]
    added = [ln[1:] + "\n" for ln in diff if ln.startswith("+") and not ln.startswith("+++")]
    # site order is file order in BOTH kernels: sig < qload < k < v < decl
    order = ["sig", "qload", "k", "v", "decl"]
    e = EDITS[tag]
    assert e["sig"] < e["qload"][0] < e["k"][0] < e["v"][0] < e["decl"][0], "site order drifted"
    old_pool, new_pool = [], []
    for site in order:
        old, new = SITES[tag][site]
        old_pool.extend(old.splitlines(keepends=True))
        new_pool.extend(new.splitlines(keepends=True))
    from collections import Counter

    old_c, new_c = Counter(old_pool), Counter(new_pool)
    rem_c, add_c = Counter(removed), Counter(added)
    for ln, n in rem_c.items():
        assert old_c[ln] >= n, f"removed line not from a pinned site ({n}x): {ln!r}"
    for ln, n in add_c.items():
        assert new_c[ln] >= n, f"added line not from a pinned site ({n}x): {ln!r}"
    # every shipped T.copy load line and every BHSD decl line must appear in
    # the removed-or-context accounting: specifically the strided-form NEW
    # code lines must be present exactly once each in added
    for must in ("    in_bsnd=False,\n",):
        assert add_c[must] == 1, f"expected insertion missing: {must!r}"
    # the shipped box lines survive ONLY inside else branches (re-indented),
    # so the ORIGINAL 12-space forms must show up as removed
    for must in ("            T.copy(K[bz, kg, 0:block_N, :], k_l1_0)\n",
                 "            T.copy(V[bz, kg, 0:block_N, :], v_l1_0)\n",
                 "        T.copy(Q[bz, hq0, bx * block_M : (bx + 1) * block_M, :], q_l1[0:block_M, :])\n",
                 "        Q: T.Tensor([B, N_q, S_padded, D], dtype),  # type: ignore  BHSD\n"):
        assert rem_c[must] >= 1, f"expected removal missing: {must!r}"

    # (5) task decomposition / schedule / arithmetic untouched
    for pat in ("def task_range", "T.serial(", "T.Kernel(", "T.mma(", "emit_qload(",
                "emit_g1(", "emit_g2(", "T.reduce_max", "T.reduce_sum", "T.tile."):
        assert ship.count(pat) == text.count(pat), pat

    # (6) diff hunk census: 5 edit sites, but the k/v sites sit 4-5 lines
    # apart (flag+comment interleave) so difflib's 3-line context MERGES
    # them into one hunk => expect 4 hunks (sig, qload, k+v, decl).
    hunks = [ln for ln in diff if ln.startswith("@@")]
    assert len(hunks) in (4, 5), f"unexpected diff spread ({len(hunks)} hunks): {hunks}"
    return diff


def main():
    blocks = load_blocks()
    all_diff = []
    parts = []
    for tag in ("flatkv1", "flatkv2"):
        variant = make_p4(blocks[tag], tag)
        diff = audit(variant, blocks[tag], tag)
        all_diff.append(f"### {tag} — {BLOCKS[tag][2]} (gqa.py:{BLOCKS[tag][0]}-{BLOCKS[tag][1]})")
        all_diff.extend(diff)
        all_diff.append("")
        parts.append("".join(variant))
        print(f"[{tag}] audit OK: 5 edit sites, flag lines byte-identical, geometry identical, "
              f"exact-diff projection clean, {len([l for l in diff if l.startswith(chr(64)*2)])} diff hunks")
    dst = os.path.join(HERE, "gqa_p4_inbsnd_iter10_variant.py")
    with open(dst, "w") as f:
        f.write(HEADER)
        f.write("\n\n".join(parts))
    with open(os.path.join(HERE, "diff_iter10_p4.txt"), "w") as f:
        f.write("\n".join(all_diff) + "\n")
    import ast

    ast.parse(open(dst).read())
    print(f"WROTE {dst}")
    print(f"WROTE {os.path.join(HERE, 'diff_iter10_p4.txt')} ({len(all_diff)} lines)")


if __name__ == "__main__":
    main()
