"""Iter11 P4 wave-2 Gate-3 integration — transpose_qkv elimination for the
g2stack + prefill_expert dual-aligned domains in gqa.py (main kernels read
BSND strided GM directly).

Line-anchored, assertion-guarded edit script (integrate_iter8/9/10 pattern):
every edit site's ORIGINAL content is asserted before replacement, edits are
applied bottom-up so earlier line numbers stay valid, and the result is
AST-checked + diffed + post-condition-counted.  gqa.py md5 must equal the
shipped baseline f59b561212e60a50b78a51671999ef7c on entry.

Kernel-side site texts are IMPORTED from gen_variant_iter11_p4w2.py (single
source of truth — the integrated edits are byte-identical to the Gate-0/2
validated variant edits).

Integration surface (minimal; nothing else in the file changes):
  1. The two target factories (`_gqa_kernel_prefill_expert`,
     `_gqa_kernel_prefill_g2stack`): new kwarg `in_bsnd=False`;
     `qkv_q_shape`/`qkv_kv_shape` host variables; Q/K/V declared with them;
     the Q/K/V load boxes become `if in_bsnd:` branches (BSND strided boxes)
     with the shipped BHSD boxes preserved VERBATIM in the `else` — so
     in_bsnd=False traces to byte-identical generated code.  Docstring
     in_bsnd paragraphs added.
  2. `gqa_expert`: the g2stack + prefill_expert spec builders pass
     `in_bsnd=in_bsnd_eff`; `_P4_BSND_KERNELS` extended to 4 entries.
     NOTHING else: in_bsnd_eff gate, chain_key, _run.in_bsnd, adapter
     resolve-then-supply machinery are all wave-1 infrastructure and stay
     byte-identical (only comments refreshed for accuracy).
  3. Comment/docstring accuracy updates at the touched host sites.
  NOT touched: flat kv1/kv2 kernels (wave-1), per-head AUTO ladder, g2,
  decode, transpose_qkv itself, transpose_out / out_plan / _retile_out_plan
  (P1), out_bsnd machinery (iter8/9), _prefill_run_padded, test_gqa.py.

Partial adoption: --domains {both,g2stack,prexpert} (default both) — a
failing domain's sites are skipped and its name stays out of the tuple.

W2_MEASURED below is filled from the Gate-2 results BEFORE running (the
comment it feeds is accuracy-only; criteria live in candidate_iter11_P4w2.json).

Run (repo root):
    python3 examples/gqa/perf_tuning/integrate_iter11_p4w2.py [--dry] [--domains D]
"""

import ast
import difflib
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_DIR = os.path.dirname(HERE)
GQA_PY = os.path.join(GQA_DIR, "gqa.py")
EXPECT_MD5 = "f59b561212e60a50b78a51671999ef7c"

sys.path.insert(0, HERE)
import gen_variant_iter11_p4w2 as GV  # noqa: E402  (site texts: single source of truth)

# Filled from Gate-2 measured results before the real run (comment accuracy).
W2_MEASURED = "-33.4% (c19) / -5.4..-7.2% (c8/c9/c10)"

# ---------------------------------------------------------------- kernel side --
# Site texts imported from the generator (byte-identical to the validated variant)
_SIG_OLD, _SIG_NEW = GV._SIG_OLD, GV._SIG_NEW
_DECL_OLD, _DECL_NEW = GV._DECL_OLD, GV._DECL_NEW
_QLOAD_OLD_PE, _QLOAD_NEW_PE = GV._QLOAD_OLD_PE, GV._QLOAD_NEW_PE
_QLOAD_OLD_G2, _QLOAD_NEW_G2 = GV._QLOAD_OLD_G2, GV._QLOAD_NEW_G2
_K_OLD, _K_NEW = GV._K_OLD, GV._K_NEW
_V_OLD, _V_NEW = GV._V_OLD, GV._V_NEW

DOC_PE_INSERT = """
    in_bsnd ([S3-iter11-P4w2]): declare Q/K/V BSND ([B, S_padded, N_q, D] /
    [B, S_kv_padded, N_kv, D]) and read the Q/K/V load boxes strided (row
    stride N*D) instead of the BHSD contiguous boxes, so the host can skip
    the transpose_qkv kernel entirely — the iter10 wave-1 mechanism (flat
    kv1/kv2) extended to this kernel.  Callers MUST only request it when
    S_padded == S AND S_kv_padded == S_kv (both row dims aligned — the
    transpose's zero-fill pad semantics are then empty by construction).
    Each box stays ONE copy_gm_to_l1 (realSrcN folds the singleton head dim;
    the K/V form — runtime parity branch + runtime idx window + BSND strided
    box — is literally the decode dense_expert production pattern,
    gqa.py:1229/:1266); instruction count and every flag line unchanged.
    Arithmetic untouched => outputs are BIT-IDENTICAL to the transpose_qkv +
    BHSD chain (Gate 0: run_iter11_p4w2_gate0.py).
"""

DOC_G2_INSERT = """
    in_bsnd: see _gqa_kernel_prefill_expert ([S3-iter11-P4w2], identical
    mechanism and dual-alignment gate — BSND strided direct reads,
    transpose_qkv eliminated).  The stacked-M Q pair keeps the M_stack dim
    dst-side only (static q_l1 slices, byte-identical across arms): each
    box's source extents fold to realSrcN = N_q*D exactly as in the wave-1
    flat emit_qload (p4_wave2_design_iter11.md §2.1).
"""

# ------------------------------------------------------------------ host side --
def p4k_block(domains):
    names = '"g2stack_flat", "g2stack_flat_kv2"'
    if "g2stack" in domains:
        names += ', "g2stack"'
    if "prexpert" in domains:
        names += ', "prefill_expert"'
    wave2_note = (
        "# Wave-2 (iter11): the ns-batched\n"
        "# g2stack and the per-head prefill_expert — the same 5-site mechanism (their\n"
        "# K/V runtime-parity strided form is the decode dense_expert production\n"
        "# pattern).\n"
        if len(domains) == 2 else
        "# Wave-2 (iter11, PARTIAL adoption — see\n"
        "# candidate_iter11_P4w2.json): " + ", ".join(sorted(domains)) + " only.\n"
    )
    return (
        "# [S3-iter10-P4][S3-iter11-P4w2] the prefill kernels that can read their Q/K/V\n"
        "# inputs BSND-strided directly (in_bsnd=True), eliminating the transpose_qkv\n"
        "# launch.  All are expert manual-flag kernels whose MTE2 load boxes are\n"
        "# emitted verbatim from the source (no AUTO vid-splitting — the iter9\n"
        "# AscendVidReduction strided-box defect applies to AUTO lowering; the GM->L1\n"
        "# read side under AUTO is untested, so the per-head ladder stays EXCLUDED).\n"
        "# Wave-1 (iter10): the two flat kernels.\n"
        + wave2_note +
        "# Everything else keeps the BHSD contract + transpose_qkv.\n"
        f"_P4_BSND_KERNELS = ({names})\n"
    )


P4K_OLD = '''# [S3-iter10-P4] the only prefill kernels that can read their Q/K/V inputs
# BSND-strided directly (in_bsnd=True), eliminating the transpose_qkv launch.
# Both are expert manual-flag kernels whose MTE2 load boxes are emitted
# verbatim from the source (no AUTO vid-splitting — the iter9
# AscendVidReduction strided-box defect applies to AUTO lowering; the GM->L1
# read side under AUTO is untested, so the per-head ladder is EXCLUDED, as
# are g2stack / prefill_expert this wave — first-wave scope discipline, their
# conversion is the same 5-site mechanism).  Everything else keeps the BHSD
# contract + transpose_qkv.
_P4_BSND_KERNELS = ("g2stack_flat", "g2stack_flat_kv2")
'''

EXPERT_DOC_OLD = """    in_bsnd=True ([S3-iter10-P4], prefill only) additionally asks the
    dual-aligned flat kv1/kv2 domain to READ its Q/K/V inputs BSND-strided
    directly (transpose_qkv eliminated), reported on `_run.in_bsnd`.
    Honoured ONLY when S_padded == S AND S_kv_padded == S_kv AND the resolved
    kernel is in _P4_BSND_KERNELS; the caller contract then becomes
    `run(Q_bsnd, K_bsnd, V_bsnd, Mask_b, crop=False)` with the SAME alignment
    guarantee.  Every other shape keeps the BHSD input contract above
    byte-identically (in_bsnd=False generates the shipped code).
"""
EXPERT_DOC_NEW = """    in_bsnd=True ([S3-iter10-P4] flat wave-1 / [S3-iter11-P4w2] g2stack +
    prefill_expert, prefill only) additionally asks the dual-aligned expert
    prefill domain to READ its Q/K/V inputs BSND-strided directly
    (transpose_qkv eliminated), reported on `_run.in_bsnd`.  Honoured ONLY
    when S_padded == S AND S_kv_padded == S_kv AND the resolved kernel is in
    _P4_BSND_KERNELS; the caller contract then becomes
    `run(Q_bsnd, K_bsnd, V_bsnd, Mask_b, crop=False)` with the SAME alignment
    guarantee.  Every other shape keeps the BHSD input contract above
    byte-identically (in_bsnd=False generates the shipped code).
"""

BSNDEFF_CMT_OLD = """    # _P4_BSND_KERNELS (the two flat expert kernels); the capability actually
    # granted is re-checked against the resolved spec below.
"""
BSNDEFF_CMT_NEW = """    # _P4_BSND_KERNELS (the four expert prefill kernels: flat kv1/kv2 +
    # g2stack + prefill_expert); the capability actually granted is
    # re-checked against the resolved spec below.
"""

SPEC_OLD = "                        out_bsnd=bsnd_eff,\n"
SPEC_NEW = "                        out_bsnd=bsnd_eff,\n                        in_bsnd=in_bsnd_eff,\n"

RUNFLAG_CMT_OLD = """    # transpose_qkv and the BHSD contract holds.  in_bsnd => out_bsnd (the
    # gate is a strict subset and _P4_BSND_KERNELS ⊂ _P2_BSND_KERNELS).
"""
RUNFLAG_CMT_NEW = """    # transpose_qkv and the BHSD contract holds.  in_bsnd => out_bsnd (the
    # gate is a subset and _P4_BSND_KERNELS ⊆ _P2_BSND_KERNELS — equal since
    # [S3-iter11-P4w2]).
"""

ADAPTER_REQ_OLD = """            # [S3-iter10-P4] ask for the BSND direct-read domain (flat
            # kv1/kv2 only; gqa_expert applies the dual-alignment gate and
            # reports the capability on _run.in_bsnd).
"""
ADAPTER_REQ_NEW = """            # [S3-iter10-P4][S3-iter11-P4w2] ask for the BSND direct-read
            # domain (all four expert prefill kernels; gqa_expert applies the
            # dual-alignment gate and reports the capability on _run.in_bsnd).
"""

ADAPTER_SKIP_OLD = """            # [S3-iter10-P4] BSND direct-read domain (S % 64 == 0 AND
            # S_kv % 128 == 0 AND the resolved kernel is flat kv1/kv2): the
            # main kernel reads the caller's BSND tensors via strided MTE2
            # boxes (the decode dense_expert production pattern), so the
            # transpose_qkv launch is eliminated — measured e2e -13..-25% on
            # this domain (perf_tuning/candidate_iter10_P4.json).  The
            # variable names keep the shipped q_bhsd spelling; on this branch
"""
ADAPTER_SKIP_NEW = """            # [S3-iter10-P4][S3-iter11-P4w2] BSND direct-read domain (S % 64
            # == 0 AND S_kv % 128 == 0 AND the resolved kernel is one of the
            # four BSND-read experts — flat kv1/kv2 since iter10, g2stack +
            # prefill_expert since iter11): the main kernel reads the caller's
            # BSND tensors via strided MTE2 boxes (the decode dense_expert
            # production pattern), so the transpose_qkv launch is eliminated —
            # measured e2e -13..-27% wave-1 (candidate_iter10_P4.json),
            # """ + W2_MEASURED + """ wave-2 (candidate_iter11_P4w2.json).  The
            # variable names keep the shipped q_bhsd spelling; on this branch
"""


def build_ops(lines, domains):
    """[(start_0based, end_0based_exclusive, expect_text, new_text)] — every
    site pinned by verified line number against md5 f59b5612..., applied
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

    nDocPe = len(DOC_PE_INSERT.splitlines(keepends=True))
    nP4k = len(P4K_OLD.splitlines(keepends=True))
    nExDoc = len(EXPERT_DOC_OLD.splitlines(keepends=True))
    nDecl = len(_DECL_OLD.splitlines(keepends=True))
    nQPe = len(_QLOAD_OLD_PE.splitlines(keepends=True))
    nQg2 = len(_QLOAD_OLD_G2.splitlines(keepends=True))
    nK = len(_K_OLD.splitlines(keepends=True))

    ops = [
        # ---- adapter comment accuracy (bottom-up) ----
        op(6007, 7, ADAPTER_SKIP_OLD, ADAPTER_SKIP_NEW),
        op(5998, 3, ADAPTER_REQ_OLD, ADAPTER_REQ_NEW),
        # ---- gqa_expert host machinery ----
        op(5123, 2, RUNFLAG_CMT_OLD, RUNFLAG_CMT_NEW),
    ]
    if "prexpert" in domains:
        ops.append(op(4997, 1, SPEC_OLD, SPEC_NEW))          # prefill_expert spec builder
    if "g2stack" in domains:
        ops.append(op(4963, 1, SPEC_OLD, SPEC_NEW))          # g2stack spec builder
    ops += [
        op(4840, 2, BSNDEFF_CMT_OLD, BSNDEFF_CMT_NEW),
        op(4510, 8, EXPERT_DOC_OLD, EXPERT_DOC_NEW),
        op(4473, nP4k, P4K_OLD, p4k_block(domains)),
    ]
    if "g2stack" in domains:
        ops += [
            # ---- g2stack kernel ----
            op(2793, nK, _V_OLD, _V_NEW),
            op(2750, nK, _K_OLD, _K_NEW),
            op(2722, nQg2, _QLOAD_OLD_G2, _QLOAD_NEW_G2),
            op(2637, nDecl, _DECL_OLD, _DECL_NEW),
            insert_before(2575, '    """\n', DOC_G2_INSERT),
            op(2520, 1, _SIG_OLD, _SIG_NEW),
        ]
    if "prexpert" in domains:
        ops += [
            # ---- prefill_expert kernel ----
            op(2277, nK, _V_OLD, _V_NEW),
            op(2234, nK, _K_OLD, _K_NEW),
            op(2207, nQPe, _QLOAD_OLD_PE, _QLOAD_NEW_PE),
            op(2129, nDecl, _DECL_OLD, _DECL_NEW),
            insert_before(2068, '    """\n', DOC_PE_INSERT),
            op(2014, 1, _SIG_OLD, _SIG_NEW),
        ]
    return ops


def flaglines(st):
    return [ln for ln in st.splitlines()
            if any(k in ln for k in ("set_flag", "wait_flag", "set_cross_flag", "wait_cross_flag"))]


def main():
    dry = "--dry" in sys.argv
    domains = ("both", "g2stack", "prexpert")
    dom = "both"
    for a in sys.argv[1:]:
        if a.startswith("--domains"):
            dom = a.split("=", 1)[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]
    assert dom in domains, f"--domains must be one of {domains}"
    dlist = ["g2stack", "prexpert"] if dom == "both" else [dom]
    assert W2_MEASURED != "TBD-BEFORE-RUN" or dry, \
        "fill W2_MEASURED from the Gate-2 results before the real run"

    md5 = hashlib.md5(open(GQA_PY, "rb").read()).hexdigest()
    assert md5 == EXPECT_MD5, f"gqa.py md5 {md5} != shipped baseline {EXPECT_MD5}"
    orig_text = open(GQA_PY).read()
    lines = orig_text.splitlines(keepends=True)
    ops = build_ops(lines, dlist)  # every site asserted inside build_ops
    spans = sorted((s, e) for s, e, _, _ in ops)
    for i in range(1, len(spans)):
        assert spans[i][0] >= spans[i - 1][1], f"overlapping sites: {spans[i-1]} {spans[i]}"
    for start, end, _expect, new in sorted(ops, key=lambda o: -o[0]):
        lines[start:end] = new.splitlines(keepends=True)
    text = "".join(lines)
    ast.parse(text)

    nK_dom = len(dlist)
    # ---- post-conditions ----
    assert text.count("    in_bsnd=False,\n") == 2 + nK_dom          # 2 flat (wave-1) + wave-2 kernels
    assert text.count("qkv_q_shape = [B, S_padded, N_q, D] if in_bsnd") == 2 + nK_dom
    assert text.count("Q: T.Tensor(qkv_q_shape, dtype)") == 2 + nK_dom
    assert text.count("Q: T.Tensor([B, N_q, S_padded, D], dtype),  # type: ignore  BHSD\n") == 2 - nK_dom
    assert text.count("in_bsnd=in_bsnd_eff,") == 2 + nK_dom           # spec builders
    # new strided multi-line K/V boxes: +2 per kernel per operand (parity branches);
    # the ",\n" suffix excludes the decode kernels' single-line BSND forms (count 0 pre-edit)
    assert text.count("K[bz, idx * block_N : (idx + 1) * block_N, kg, :],\n") == 2 * nK_dom
    assert text.count("V[bz, idx * block_N : (idx + 1) * block_N, kg, :],\n") == 2 * nK_dom
    # shipped BHSD runtime-idx boxes preserved verbatim (else branches, re-indented): 4 total
    assert text.count("K[bz, kg, idx * block_N : (idx + 1) * block_N, :],") == 4
    assert text.count("V[bz, kg, idx * block_N : (idx + 1) * block_N, :],") == 4
    # host machinery
    names = '"g2stack_flat", "g2stack_flat_kv2"' + ('' if "g2stack" not in dlist else ', "g2stack"') \
        + ('' if "prexpert" not in dlist else ', "prefill_expert"')
    assert text.count(f"_P4_BSND_KERNELS = ({names})") == 1
    assert text.count("in_bsnd_eff = bool(in_bsnd) and S_padded == S and S_kv_padded == S_kv") == 1
    assert text.count("_run.in_bsnd = bool(in_bsnd_eff) and chosen in _P4_BSND_KERNELS") == 1
    assert text.count("sm_scale, bsnd_eff, in_bsnd_eff)") == 1        # chain_key unchanged
    # adapter machinery UNCHANGED (wave-1 structure; only comments refreshed)
    assert text.count("q_bhsd, k_bhsd, v_bhsd = q, k_t, v_t") == 1
    assert text.count("in_bsnd=True,\n") == 1
    assert text.count('if getattr(kernel_fn, "in_bsnd", False):') == 1
    assert text.count("res = transpose_qkv(q, k_t, v_t, S_out=S_out, S_kv_out=S_kv_out)") == 1
    assert text.count("cache_key = (B_bs,") == 1
    # untouched surfaces
    assert text.count("_P2_BSND_KERNELS = (") == 1
    assert '("g2stack_flat", "g2stack_flat_kv2", "g2stack", "prefill_expert")' in text  # _P2 unchanged
    assert "def _transpose_qkv(" in text and "def transpose_out(" in text
    assert text.count("_gqa_kernel_expert(") >= 2                     # per-head ladder intact
    # GLOBAL flag-line invariance (whole file — no flag line added/removed/moved)
    assert flaglines(orig_text) == flaglines(text), "global flag protocol lines diverged"
    # line-length discipline (repo ruff line-length = 140)
    longest = max((len(ln), i + 1) for i, ln in enumerate(text.splitlines()))
    assert longest[0] <= 140, f"line {longest[1]} is {longest[0]} chars (> 140)"

    diff = list(difflib.unified_diff(orig_text.splitlines(), text.splitlines(),
                                     fromfile="gqa.py (f59b5612)", tofile=f"gqa.py (iter11 P4w2 {dom})",
                                     lineterm="", n=3))
    hunks = [ln for ln in diff if ln.startswith("@@")]
    adds = sum(1 for ln in diff if ln.startswith("+") and not ln.startswith("+++"))
    dels = sum(1 for ln in diff if ln.startswith("-") and not ln.startswith("---"))
    print(f"domains={dlist}  edit sites: {len(ops)}  hunks: {len(hunks)}  +{adds}/-{dels} lines  "
          f"longest line: {longest[0]} (at {longest[1]})")
    with open(os.path.join(HERE, f"diff_iter11_p4w2_integration_{dom}.txt"), "w") as f:
        f.write("\n".join(diff) + "\n")
    if dry:
        print("[dry] not writing gqa.py")
        return
    with open(GQA_PY, "w") as f:
        f.write(text)
    new_md5 = hashlib.md5(text.encode()).hexdigest()
    print(f"WROTE {GQA_PY}")
    print(f"new md5: {new_md5}")


if __name__ == "__main__":
    main()
