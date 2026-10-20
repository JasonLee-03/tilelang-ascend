"""Iter9 P2-wave2 Gate-3 integration — extend the BSND direct-write to the
prefill_expert + g2stack domains in gqa.py.

Line-anchored, assertion-guarded edit script (integrate_iter8_p2.py pattern):
every edit site's ORIGINAL content is asserted before replacement, edits are
applied bottom-up so earlier line numbers stay valid, and the result is
AST-checked + diffed + post-condition-counted.  gqa.py md5 must equal the
shipped baseline 3aa9cfa2625cdce96a62ce967785fe85 on entry.

Integration surface (minimal; nothing else in the file changes):
  1. `_gqa_kernel_prefill_expert` / `_gqa_kernel_prefill_g2stack`: new kwarg
     `out_bsnd=False`; `out_shape` host variable; Output declared with it;
     the FIN write box becomes an `if out_bsnd:` branch (BSND strided box)
     with the shipped BHSD box preserved VERBATIM in the `else` — so
     out_bsnd=False traces to byte-identical generated code.
  2. `gqa_expert`: the g2stack + prefill_expert spec builders pass
     `out_bsnd=bsnd_eff` (the wave-1 alignment gate variable, unchanged);
     `_P2_BSND_KERNELS` grows to 4 entries (the capability flag
     `_run.out_bsnd = bsnd_eff and chosen in _P2_BSND_KERNELS`, the chain_key
     bsnd_eff component, `_prefill_run_padded`'s out_bsnd branch and the
     gqa() adapter's capability check are ALL wave-1 machinery, reused
     untouched).
  3. Comment/docstring accuracy updates at 4 wave-1 comment sites that would
     otherwise go stale (adapter, _run.out_bsnd, bsnd_eff gate, gqa_expert
     docstring) — comment-only, zero code semantics.
  NOT touched: the per-head AUTO ladder `_gqa_kernel_expert` (EXCLUDED —
  threads=2 strided-write mis-lowering, p2_wave2_design_iter9.md §2), the g2
  last-resort fallback, transpose_out / out_plan / _retile_out_plan (P1),
  decode, test_gqa.py.

Run (repo root):
    python3 examples/gqa/perf_tuning/integrate_iter9_p2w2.py [--dry]
"""

import ast
import difflib
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_DIR = os.path.dirname(HERE)
GQA_PY = os.path.join(GQA_DIR, "gqa.py")
EXPECT_MD5 = "3aa9cfa2625cdce96a62ce967785fe85"

# ---------------------------------------------------------------- kernel side --
SIG_OLD = "    cross_interval=2,\n"
SIG_NEW = "    cross_interval=2,\n    out_bsnd=False,\n"

DOC_PX_OLD = '    row segment one contiguous box (runtime-base/static-extent copy).\n    """\n'
DOC_PX_NEW = '''    row segment one contiguous box (runtime-base/static-extent copy).

    out_bsnd ([S3-iter9-P2w2]): declare Output BSND [B, S_padded, N_q, D] and
    write the FIN box strided (row stride N_q*D) instead of the BHSD
    contiguous box, so the host can skip the transpose_out kernel entirely —
    the iter8 wave-1 mechanism extended to this kernel (identical SIG_S_HALF
    pair around the write, identical single DataCopyPad per box with the
    per-vid blockCount, only dstStride differs).  Callers MUST only request it
    when S_padded == S (row-aligned): the BSND tensor then IS the
    [B, S, N_q, D] result and no crop is materialized.  Arithmetic and the
    flag protocol are untouched => final outputs are BIT-IDENTICAL to the
    BHSD + transpose_out chain (Gate 0: run_iter9_p2w2_gate0.py).
    """
'''

DOC_G2_OLD = "    (runtime-base/static-extent copy).\n    \"\"\"\n"
DOC_G2_NEW = '''    (runtime-base/static-extent copy).

    out_bsnd: see _gqa_kernel_prefill_expert ([S3-iter9-P2w2], identical
    mechanism and gate — BSND direct write for the two per-(h, vid) FIN
    boxes, transpose_out eliminated on the aligned-S domain).
    """
'''

OUT_SHAPE_LINE = """    # [S3-iter9-P2w2] Output layout: BSND [B, S_padded, N_q, D] when the host
    # skips transpose_out (caller-gated on S_padded == S), else the shipped
    # BHSD [B, N_q, S_padded, D].
    out_shape = [B, S_padded, N_q, D] if out_bsnd else [B, N_q, S_padded, D]

"""

DECL_OLD = "        Output: T.Tensor([B, N_q, S_padded, D], dtype),  # type: ignore  BHSD\n"
DECL_NEW = "        Output: T.Tensor(out_shape, dtype),  # type: ignore  BSND if out_bsnd else BHSD\n"

FIN_PX_OLD = """                    T.copy(
                        acc_s_half,
                        Output[bz, hq, bx * block_M + vid * half : bx * block_M + vid * half + half, :],
                    )
"""
FIN_PX_NEW = """                    if out_bsnd:
                        # [S3-iter9-P2w2] BSND strided box: the SAME element set
                        # as the BHSD box in the else branch, written straight
                        # into the caller's [B, S, N_q, D] layout so the
                        # transpose_out launch is eliminated.  Both forms lower
                        # to ONE copy_ub_to_gm/DataCopyPad (blockCount=half,
                        # blockLen=D*sizeof(T)); only dstStride differs
                        # (N_q*D - D vs 0) — the instruction count and the
                        # SIG_S_HALF flag pair around it are unchanged (wave-1
                        # mechanism; codegen_audit_iter9.log prexpert_c19).
                        T.copy(
                            acc_s_half,
                            Output[bz, bx * block_M + vid * half : bx * block_M + vid * half + half, hq, :],
                        )
                    else:
                        T.copy(
                            acc_s_half,
                            Output[bz, hq, bx * block_M + vid * half : bx * block_M + vid * half + half, :],
                        )
"""

FIN_G2_OLD = """                        T.copy(
                            acc_s_half,
                            Output[bz, hq0 + h, bx * block_M + vid * half : bx * block_M + vid * half + half, :],
                        )
"""
FIN_G2_NEW = """                        if out_bsnd:
                            # [S3-iter9-P2w2] BSND strided box: the SAME
                            # element set as the BHSD box in the else branch,
                            # written straight into the caller's
                            # [B, S, N_q, D] layout so the transpose_out
                            # launch is eliminated.  Both forms lower to ONE
                            # copy_ub_to_gm/DataCopyPad per (h, vid)
                            # (blockCount=half, blockLen=D*sizeof(T)); only
                            # dstStride differs (N_q*D - D vs 0) — the
                            # instruction count and the SIG_S_HALF flag pair
                            # around it are unchanged (wave-1 mechanism;
                            # codegen_audit_iter9.log g2stack_c8/c9).
                            T.copy(
                                acc_s_half,
                                Output[bz, bx * block_M + vid * half : bx * block_M + vid * half + half, hq0 + h, :],
                            )
                        else:
                            T.copy(
                                acc_s_half,
                                Output[bz, hq0 + h, bx * block_M + vid * half : bx * block_M + vid * half + half, :],
                            )
"""

# ------------------------------------------------------------------ host side --
P2K_OLD = """# [S3-iter8-P2] the only prefill kernels that can write their output BSND
# directly (out_bsnd=True), eliminating the transpose_out launch.  Everything
# else keeps the BHSD contract + transpose_out (with iter7's P1 retiled plan).
_P2_BSND_KERNELS = ("g2stack_flat", "g2stack_flat_kv2")
"""
P2K_NEW = """# [S3-iter8-P2][S3-iter9-P2w2] the prefill kernels that can write their output
# BSND directly (out_bsnd=True), eliminating the transpose_out launch.  Wave-1
# (iter8): the two flat kernels.  Wave-2 (iter9): the ns-batched g2stack and
# the per-head prefill_expert — both manual-flag expert kernels carrying the
# identical FIN mechanism.  The per-head AUTO ladder (_gqa_kernel_expert) is
# EXCLUDED: its threads=2 serial form mis-lowers strided BSND writes (the AUTO
# vid-reduction keeps the contiguous-layout vid offset while emitting the
# unsplit blockCount — perf_tuning/p2_wave2_design_iter9.md §2, empirically
# bit-failed), as is the last-resort g2 kernel (never converted).  Everything
# else keeps the BHSD contract + transpose_out (with iter7's P1 retiled plan).
_P2_BSND_KERNELS = ("g2stack_flat", "g2stack_flat_kv2", "g2stack", "prefill_expert")
"""

SPEC_OLD = "                        cross_interval=2,\n"
SPEC_NEW = "                        cross_interval=2,\n                        out_bsnd=bsnd_eff,\n"

BSNDEFF_CMT_OLD = """    # in-kernel with zero d2d.  Only the two flat kernels implement it; the
    # capability actually granted is re-checked against the resolved spec below.
"""
BSNDEFF_CMT_NEW = """    # in-kernel with zero d2d.  The BSND-capable kernels are _P2_BSND_KERNELS
    # (wave-1: the two flat kernels; wave-2: g2stack + prefill_expert — the
    # per-head AUTO ladder is EXCLUDED, see the note at _P2_BSND_KERNELS); the
    # capability actually granted is re-checked against the resolved spec below.
"""

EXPERT_DOC_A_OLD = """    out_bsnd=True ([S3-iter8-P2], prefill only) asks the row-aligned flat
    domain to write its output BSND directly and reports the capability on the
    returned callable (`_run.out_bsnd`), letting the caller skip the
"""
EXPERT_DOC_A_NEW = """    out_bsnd=True ([S3-iter8-P2] flat wave-1 / [S3-iter9-P2w2] g2stack +
    prefill_expert, prefill only) asks the row-aligned BSND-capable domain to
    write its output BSND directly and reports the capability on the
    returned callable (`_run.out_bsnd`), letting the caller skip the
"""

EXPERT_DOC_B_OLD = """    The request is honoured ONLY when S_padded == S (no row crop needed) AND
    the resolved kernel is one of the two flat kernels; every other shape
    keeps the BHSD contract above byte-identically.
"""
EXPERT_DOC_B_NEW = """    The request is honoured ONLY when S_padded == S (no row crop needed) AND
    the resolved kernel is in _P2_BSND_KERNELS; every other shape (per-head
    AUTO ladder, g2 fallback, non-aligned S) keeps the BHSD contract above
    byte-identically.
"""

RUNFLAG_CMT_OLD = """    # [S3-iter8-P2] capability flag consumed by the gqa() adapter: True only
    # when the resolved kernel really writes BSND.  A compile fallback down the
    # chain (g2stack / prefill_expert / per-head) leaves it False, so the
    # adapter keeps calling transpose_out and the BHSD contract holds.
"""
RUNFLAG_CMT_NEW = """    # [S3-iter8-P2] capability flag consumed by the gqa() adapter: True only
    # when the resolved kernel really writes BSND.  A compile fallback down
    # the chain (to the per-head AUTO ladder or g2 — neither writes BSND)
    # leaves it False, so the adapter keeps calling transpose_out and the
    # BHSD contract holds.
"""

ADAPTER_CMT_OLD = """    # [S3-iter8-P2] BSND direct-write domain (row-aligned flat kv1/kv2): the
    # main kernel already produced the contiguous [B, S, N_q, D] result, so the
    # transpose_out launch is eliminated — measured e2e -11..-20% on this domain
    # (perf_tuning/candidate_iter8_P2.json).  Every other domain (per-head,
    # prefill_expert, g2stack, non-aligned S) falls through unchanged, and
    # decode never reaches here.
"""
ADAPTER_CMT_NEW = """    # [S3-iter8-P2][S3-iter9-P2w2] BSND direct-write domain (row-aligned:
    # flat kv1/kv2 since iter8, g2stack + prefill_expert since iter9): the
    # main kernel already produced the contiguous [B, S, N_q, D] result, so
    # the transpose_out launch is eliminated — measured e2e -11..-20% on the
    # wave-1 domain (perf_tuning/candidate_iter8_P2.json).  Every other
    # domain (per-head AUTO ladder — excluded, see _P2_BSND_KERNELS —
    # non-aligned S, g2 fallback) falls through unchanged, and decode never
    # reaches here.
"""


def build_ops(lines):
    """[(start_0based, end_0based_exclusive, expect_text, new_text)] — every
    site pinned by verified line number against md5 3aa9cfa2..., applied
    bottom-up."""

    def op(lineno_1based, nlines, expect, new):
        start = lineno_1based - 1
        got = "".join(lines[start : start + nlines])
        assert got == expect, (
            f"site drift at line {lineno_1based}:\n--- got ---\n{got}--- want ---\n{expect}"
        )
        return (start, start + nlines, expect, new)

    def insert_before(lineno_1based, anchor, new):
        start = lineno_1based - 1
        assert lines[start] == anchor, f"anchor drift at {lineno_1based}: {lines[start]!r}"
        return (start, start, "", new)

    ops = [
        # ---- host side, bottom-up ----
        op(5848, 6, ADAPTER_CMT_OLD, ADAPTER_CMT_NEW),
        op(4919, 4, RUNFLAG_CMT_OLD, RUNFLAG_CMT_NEW),
        op(4805, 1, SPEC_OLD, SPEC_NEW),          # prefill_expert spec builder
        op(4772, 1, SPEC_OLD, SPEC_NEW),          # g2stack spec builder
        op(4652, 2, BSNDEFF_CMT_OLD, BSNDEFF_CMT_NEW),
        op(4339, 3, EXPERT_DOC_B_OLD, EXPERT_DOC_B_NEW),
        op(4330, 3, EXPERT_DOC_A_OLD, EXPERT_DOC_A_NEW),
        op(4314, 4, P2K_OLD, P2K_NEW),
        # ---- g2stack kernel ----
        op(2958, 4, FIN_G2_OLD, FIN_G2_NEW),
        op(2603, 1, DECL_OLD, DECL_NEW),
        insert_before(2597, "    @T.prim_func\n", OUT_SHAPE_LINE),
        op(2537, 2, DOC_G2_OLD, DOC_G2_NEW),      # docstring close at 2538
        op(2487, 1, SIG_OLD, SIG_NEW),
        # ---- prefill_expert kernel ----
        op(2457, 4, FIN_PX_OLD, FIN_PX_NEW),
        op(2120, 1, DECL_OLD, DECL_NEW),
        insert_before(2114, "    @T.prim_func\n", OUT_SHAPE_LINE),
        op(2055, 2, DOC_PX_OLD, DOC_PX_NEW),      # docstring close at 2056
        op(2013, 1, SIG_OLD, SIG_NEW),
    ]
    return ops


def main():
    dry = "--dry" in sys.argv
    md5 = hashlib.md5(open(GQA_PY, "rb").read()).hexdigest()
    assert md5 == EXPECT_MD5, f"gqa.py md5 {md5} != shipped baseline {EXPECT_MD5}"
    orig_text = open(GQA_PY).read()
    lines = orig_text.splitlines(keepends=True)
    ops = build_ops(lines)  # every site asserted inside build_ops
    for start, end, _expect, new in sorted(ops, key=lambda o: -o[0]):
        lines[start:end] = new.splitlines(keepends=True)
    text = "".join(lines)
    ast.parse(text)

    # ---- post-conditions ----
    assert text.count("Output: T.Tensor(out_shape, dtype)") == 4  # wave-1 x2 + wave-2 x2
    # BSND strided boxes (if-branches): flat kv1 + flat kv2 + g2stack (hq0+h form)
    assert text.count(
        "Output[bz, bx * block_M + vid * half : bx * block_M + vid * half + half, hq0 + h, :],"
    ) == 3
    # BHSD boxes preserved verbatim (else-branches): flat kv1 + flat kv2 + g2stack
    assert text.count(
        "Output[bz, hq0 + h, bx * block_M + vid * half : bx * block_M + vid * half + half, :],"
    ) == 3
    assert text.count(
        "Output[bz, bx * block_M + vid * half : bx * block_M + vid * half + half, hq, :],"
    ) == 1  # prexpert if-branch
    assert text.count(
        "Output[bz, hq, bx * block_M + vid * half : bx * block_M + vid * half + half, :],"
    ) == 1  # prexpert else-branch
    # per-head kernel UNTOUCHED (excluded domain)
    assert text.count("Output[bz, hq, bx * block_M : (bx + 1) * block_M, :],") == 1
    assert text.count(
        "Output: T.Tensor([B, N_q, S_padded, D], dtype),  # type: ignore  BHSD layout"
    ) == 2  # per-head + g2 fallback decls (the two untouched "BHSD layout" kernels)
    assert "_gqa_kernel_expert" in text and "out_bsnd" not in "".join(
        text.split("def _gqa_kernel_expert(")[1].split("@tilelang.jit")[0]
    ), "per-head kernel must not gain out_bsnd"
    # g2 last-resort fallback untouched
    assert "out_bsnd" not in "".join(
        text.split("def _gqa_kernel_prefill_g2(")[1].split("def _dense_tile_params")[0]
    ), "g2 fallback must not gain out_bsnd"
    # _P2_BSND_KERNELS: 1 def + 1 capability use + 4 doc/comment cross-refs
    # (gqa_expert docstring, bsnd_eff gate note, adapter note x2 in bsnd_eff note)
    assert text.count("_P2_BSND_KERNELS") == 6, text.count("_P2_BSND_KERNELS")
    # out_bsnd=bsnd_eff: wave-1 flat kv1 + flat kv2 spec builders + the
    # _prefill_run_padded call, plus the 2 new wave-2 spec builders = 5
    assert text.count("out_bsnd=bsnd_eff,") == 5, text.count("out_bsnd=bsnd_eff,")
    assert text.count("    out_bsnd=False,\n") == 4  # 4 factory signatures
    longest = max((len(ln.rstrip("\n")) for ln in text.splitlines()), default=0)
    print(f"edit sites: {len(ops)}   longest line: {longest} (ruff line-length 140)")
    assert longest <= 140, "line-length violation"

    diff = list(difflib.unified_diff(orig_text.splitlines(), text.splitlines(),
                                     "gqa.py (pre-P2w2)", "gqa.py (post-P2w2)", lineterm="", n=3))
    with open(os.path.join(HERE, "diff_iter9_p2w2_integration.txt"), "w") as f:
        f.write("\n".join(diff) + "\n")
    added = sum(1 for d in diff if d.startswith("+") and not d.startswith("+++"))
    removed = sum(1 for d in diff if d.startswith("-") and not d.startswith("---"))
    hunks = sum(1 for d in diff if d.startswith("@@"))
    print(f"diff: +{added}/-{removed} lines in {hunks} hunks -> diff_iter9_p2w2_integration.txt")
    if dry:
        print("[DRY] no write")
        return
    with open(GQA_PY, "w") as f:
        f.write(text)
    new_md5 = hashlib.md5(open(GQA_PY, "rb").read()).hexdigest()
    print(f"WROTE {GQA_PY}\n  md5 {md5} -> {new_md5}")


if __name__ == "__main__":
    main()
