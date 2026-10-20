"""Iter8 P2 Gate-3 integration — apply the BSND direct-write to gqa.py.

Line-anchored, assertion-guarded edit script (the same discipline as
gen_variant_iter8_p2.py): every edit site's ORIGINAL content is asserted
before replacement, edits are applied bottom-up so earlier line numbers stay
valid, and the result is AST-checked + diffed.  gqa.py md5 must equal the
shipped baseline 4b4e453e4b8f2922baefbecb85912494 on entry.

Integration surface (minimal; nothing else in the file changes):
  1. `_gqa_kernel_prefill_g2stack_flat` / `_kv2`: new kwarg `out_bsnd=False`;
     `out_shape` host variable; Output declared with it; emit_fin's write box
     becomes an `if out_bsnd:` branch (BSND strided box) with the shipped BHSD
     box preserved verbatim in the `else` — so `out_bsnd=False` traces to
     byte-identical generated code.
  2. `_prefill_run_padded`: new kwarg `out_bsnd=False`; the crop branch gets a
     BSND variant (dim-1 rows, asserted S_padded == S so it is a no-op view).
  3. `gqa_expert`: new kwarg `out_bsnd=False`; `bsnd_eff = out_bsnd and
     S_padded == S` (the wave-1 alignment gate); the two flat spec builders
     pass it; chain_key carries it (so a factory caller and the adapter never
     share a cache entry with different layouts); `_run.out_bsnd` capability
     flag = bsnd_eff AND the resolved spec really is one of the two flat
     kernels (a compile fallback to g2stack/per-head leaves it False => the
     adapter keeps calling transpose_out).
  4. `gqa()` adapter: requests out_bsnd=True and returns the BSND tensor
     directly when `kernel_fn.out_bsnd` — the transpose_out launch is
     eliminated.  transpose_out / out_plan / _retile_out_plan (iter7 P1) are
     NOT modified and stay live for every out-of-domain shape.

Run (repo root):
    python3 examples/gqa/perf_tuning/integrate_iter8_p2.py [--dry]
"""

import ast
import difflib
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_DIR = os.path.dirname(HERE)
GQA_PY = os.path.join(GQA_DIR, "gqa.py")
EXPECT_MD5 = "4b4e453e4b8f2922baefbecb85912494"

FIN_OLD = """            T.copy(
                acc_s_half,
                Output[bz, hq0 + h, bx * block_M + vid * half : bx * block_M + vid * half + half, :],
            )
"""
FIN_NEW = """            if out_bsnd:
                # [S3-iter8-P2] BSND strided box: the SAME element set as the
                # BHSD box in the else branch, written straight into the
                # caller's [B, S, N_q, D] layout so the transpose_out launch is
                # eliminated.  Both forms lower to ONE copy_ub_to_gm/DataCopyPad
                # (blockCount=half, blockLen=D*2B); only dstStride differs
                # (N_q*D - D vs 0), so the instruction count and the V/MTE3
                # flag pair around it are unchanged — measured strided cost
                # 1.07-1.17x per box (perf_tuning/probe_iter8_P2_bsnd.py,
                # p2_bsnd_design_iter8.md §1.2/§4).
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

OUT_SHAPE_LINE = """    # [S3-iter8-P2] Output layout: BSND [B, S_padded, N_q, D] when the host
    # skips transpose_out (caller-gated on S_padded == S), else the shipped
    # BHSD [B, N_q, S_padded, D].
    out_shape = [B, S_padded, N_q, D] if out_bsnd else [B, N_q, S_padded, D]

"""

DECL_OLD = "        Output: T.Tensor([B, N_q, S_padded, D], dtype),  # type: ignore  BHSD\n"
DECL_NEW = "        Output: T.Tensor(out_shape, dtype),  # type: ignore  BSND if out_bsnd else BHSD\n"

DOC_KV1_OLD = '    """Flat cross-task 4-phase pipelined g2stack (kv_iters==1 specialization)."""\n'
DOC_KV1_NEW = '''    """Flat cross-task 4-phase pipelined g2stack (kv_iters==1 specialization).

    out_bsnd ([S3-iter8-P2]): declare Output BSND [B, S_padded, N_q, D] and
    write emit_fin's box strided (row stride N_q*D) instead of the BHSD
    contiguous box, so the host can skip the transpose_out kernel entirely.
    Callers MUST only request it when S_padded == S (row-aligned): the BSND
    tensor then IS the [B, S, N_q, D] result and no crop is materialized (a
    fp16/bf16 ViewCopy d2d whose binary is missing on some CANN 9.x SoC
    packages).  Arithmetic, flag protocol, slots and UB/L1/L0/ws geometry are
    untouched => final outputs are BIT-IDENTICAL to the BHSD + transpose_out
    chain (Gate 0: 23/23, perf_tuning/run_iter8_p2_gate0.py).
    """
'''
DOC_KV2_OLD = '    """Flat cross-task 4-phase pipelined g2stack (kv_iters==2 uniform specialization)."""\n'
DOC_KV2_NEW = '''    """Flat cross-task 4-phase pipelined g2stack (kv_iters==2 uniform specialization).

    out_bsnd: see _gqa_kernel_prefill_g2stack_flat ([S3-iter8-P2], identical
    mechanism and gate — BSND direct write, transpose_out eliminated).
    """
'''

SIG_OLD = "    block_N=128,\n):\n"
SIG_NEW = "    block_N=128,\n    out_bsnd=False,\n):\n"

# ---------------------------------------------------------------- host side --
RUNPAD_SIG_OLD = (
    "def _prefill_run_padded(kernel_fn, Q, K, V, Mask, B, S, S_kv, N_q, N_kv, D, S_padded, "
    "S_kv_padded, use_mask, crop=True):\n"
)
RUNPAD_SIG_NEW = (
    "def _prefill_run_padded(kernel_fn, Q, K, V, Mask, B, S, S_kv, N_q, N_kv, D, S_padded, "
    "S_kv_padded, use_mask, crop=True, out_bsnd=False):\n"
)
RUNPAD_TAIL_OLD = """    if not crop:
        return Out_padded
    return Out_padded[:, :, :S, :]
"""
RUNPAD_TAIL_NEW = """    if not crop:
        return Out_padded
    if out_bsnd:
        # [S3-iter8-P2] BSND [B, S_padded, N_q, D]: the row dim is 1.  The
        # host gate (gqa_expert's bsnd_eff) guarantees S_padded == S, so this
        # is a no-op view of a contiguous tensor — no crop copy is ever
        # materialized (fp16/bf16 ViewCopy d2d binary, 561103).
        assert S_padded == S, "out_bsnd requires a row-aligned S (S_padded == S)"
        return Out_padded[:, :S, :, :]
    return Out_padded[:, :, :S, :]
"""

EXPERT_SIG_OLD = 'def gqa_expert(B, S, S_kv, sm_scale, is_causal, N_q=8, N_kv=2, D=128, dtype="float16"):\n'
EXPERT_SIG_NEW = (
    'def gqa_expert(B, S, S_kv, sm_scale, is_causal, N_q=8, N_kv=2, D=128, dtype="float16", out_bsnd=False):\n'
)
EXPERT_DOC_OLD = '''        S >  4 (prefill, BHSD — caller transposes):
            out_bhsd = gqa_expert(...)( Q_bhsd, K_bhsd, V_bhsd, Mask_b )
'''
EXPERT_DOC_NEW = '''        S >  4 (prefill, BHSD — caller transposes):
            out_bhsd = gqa_expert(...)( Q_bhsd, K_bhsd, V_bhsd, Mask_b )

    out_bsnd=True ([S3-iter8-P2], prefill only) asks the row-aligned flat
    domain to write its output BSND directly and reports the capability on the
    returned callable (`_run.out_bsnd`), letting the caller skip the
    transpose_out launch:

        run = gqa_expert(..., out_bsnd=True)
        out = run(Q_bhsd, K_bhsd, V_bhsd, Mask_b, crop=False)
        out = out if run.out_bsnd else transpose_out(out)   # [B, S, N_q, D]

    The request is honoured ONLY when S_padded == S (no row crop needed) AND
    the resolved kernel is one of the two flat kernels; every other shape
    keeps the BHSD contract above byte-identically.
'''
G2STACK_ANCHOR = (
    "    use_g2_stack = use_expert_prefill and (G % 2) == 0 and (B * N_kv * (G // 2) * q_blocks) "
    ">= P28_NUM_CORES\n"
)
BSND_EFF_NEW = G2STACK_ANCHOR + """
    # [S3-iter8-P2] BSND direct-write eligibility (host gate).  The caller must
    # ask for it AND the shape must be row-aligned, so S_padded == S and the
    # kernel's BSND output tensor IS the final [B, S, N_q, D] result: no crop
    # copy (fp16/bf16 ViewCopy d2d, binary missing on some CANN 9.x SoC
    # packages, 561103) and no transpose_out launch.  Non-aligned S keeps the
    # BHSD + transpose_out(s_real=...) path, whose n_pad mode already crops
    # in-kernel with zero d2d.  Only the two flat kernels implement it; the
    # capability actually granted is re-checked against the resolved spec below.
    bsnd_eff = bool(out_bsnd) and S_padded == S
"""
CHAIN_KEY_OLD = '    chain_key = ("prefill", B, S, S_kv, N_q, N_kv, D, dtype, is_causal, sm_scale)\n'
CHAIN_KEY_NEW = (
    '    # [S3-iter8-P2] bsnd_eff in the key: a factory caller (BHSD contract) and\n'
    '    # the adapter (BSND request) must never share one cached chain.\n'
    '    chain_key = ("prefill", B, S, S_kv, N_q, N_kv, D, dtype, is_causal, sm_scale, bsnd_eff)\n'
)
RUN_TAIL_OLD = "    return _run\n"
RUN_TAIL_NEW = """    # [S3-iter8-P2] capability flag consumed by the gqa() adapter: True only
    # when the resolved kernel really writes BSND.  A compile fallback down the
    # chain (g2stack / prefill_expert / per-head) leaves it False, so the
    # adapter keeps calling transpose_out and the BHSD contract holds.
    _run.out_bsnd = bool(bsnd_eff) and chosen in _P2_BSND_KERNELS

    return _run
"""
P2_KERNELS_CONST = '''

# [S3-iter8-P2] the only prefill kernels that can write their output BSND
# directly (out_bsnd=True), eliminating the transpose_out launch.  Everything
# else keeps the BHSD contract + transpose_out (with iter7's P1 retiled plan).
_P2_BSND_KERNELS = ("g2stack_flat", "g2stack_flat_kv2")


'''
ADAPTER_BUILD_OLD = "            dtype=dtype_str,\n        )\n"
ADAPTER_BUILD_NEW = "            dtype=dtype_str,\n            # [S3-iter8-P2] ask for the BSND direct-write domain (gqa_expert\n            # applies the alignment gate and reports the real capability on\n            # _run.out_bsnd; out-of-domain shapes silently keep BHSD).\n            out_bsnd=True,\n        )\n"
ADAPTER_TAIL_OLD = """    out_padded = kernel_fn(q_bhsd, k_bhsd, v_bhsd, mask_npu, crop=False)
    if S_out > S_bs:
        return transpose_out(out_padded, s_real=S_bs)
    return transpose_out(out_padded)
"""
ADAPTER_TAIL_NEW = """    out_padded = kernel_fn(q_bhsd, k_bhsd, v_bhsd, mask_npu, crop=False)
    # [S3-iter8-P2] BSND direct-write domain (row-aligned flat kv1/kv2): the
    # main kernel already produced the contiguous [B, S, N_q, D] result, so the
    # transpose_out launch is eliminated — measured e2e -11..-20% on this domain
    # (perf_tuning/candidate_iter8_P2.json).  Every other domain (per-head,
    # prefill_expert, g2stack, non-aligned S) falls through unchanged, and
    # decode never reaches here.
    if getattr(kernel_fn, "out_bsnd", False):
        return out_padded
    if S_out > S_bs:
        return transpose_out(out_padded, s_real=S_bs)
    return transpose_out(out_padded)
"""
SPEC_KWARG_OLD = "                            block_N=block_N,\n                        ),\n"
SPEC_KWARG_NEW = "                            block_N=block_N,\n                            out_bsnd=bsnd_eff,\n                        ),\n"
RUNPAD_CALL_OLD = "            crop=crop,\n        )\n"
RUNPAD_CALL_NEW = "            crop=crop,\n            out_bsnd=bsnd_eff,\n        )\n"


def build_ops(lines):
    """Return [(start_idx, end_idx_exclusive, expect_text, new_text)] 0-based."""
    ops = []

    def find(expect, start=0, occ=1):
        """Locate the occ-th occurrence of expect (a full-line block)."""
        idx = -1
        for _ in range(occ):
            idx = lines.index(expect, idx + 1) if idx >= 0 or _ == 0 else lines.index(expect, idx + 1)
            if idx < 0:
                break
        return idx

    def find_block(text, start=0, occ=1):
        el = text.splitlines(keepends=True)
        n = len(el)
        hits = []
        i = start
        while True:
            j = lines.index(el[0], i) if el[0] in lines[i:] else -1
            if j < 0:
                break
            if lines[j : j + n] == el:
                hits.append(j)
            i = j + 1
        assert len(hits) >= occ, f"expected >={occ} occurrences of {text[:60]!r}, found {len(hits)}"
        return hits[occ - 1], j + n

    # ---- kernel-side (2 kernels x 4 sites) ----
    for tag, doc_old, doc_new in (("kv1", DOC_KV1_OLD, DOC_KV1_NEW), ("kv2", DOC_KV2_OLD, DOC_KV2_NEW)):
        # signature: `    block_N=128,\n):\n` immediately followed by the docstring
        di = lines.index(doc_old)
        assert lines[di - 1] == "):\n" and lines[di - 2] == "    block_N=128,\n", (lines[di - 2], lines[di - 1])
        ops.append((di - 2, di, SIG_OLD, SIG_NEW))
        ops.append((di, di + 1, doc_old, doc_new))
        # emit_fin write box: the FIRST FIN_OLD occurrence after the docstring
        fi = None
        i = di
        while fi is None:
            j = lines.index("            T.copy(\n", i)
            if "".join(lines[j : j + 4]) == FIN_OLD:
                fi = j
            i = j + 1
        ops.append((fi, fi + 4, FIN_OLD, FIN_NEW))
        # out_shape insert immediately before `    @T.prim_func` following FIN
        pi = lines.index("    @T.prim_func\n", fi)
        assert lines[pi - 1] == "\n", repr(lines[pi - 1])
        ops.append((pi, pi, "", OUT_SHAPE_LINE))
        # Output declaration (6 lines below @T.prim_func: def main( + Q,K,V,Mask)
        dcli = lines.index(DECL_OLD, pi)
        assert dcli - pi == 6, f"{tag}: decl offset {dcli - pi}"
        ops.append((dcli, dcli + 1, DECL_OLD, DECL_NEW))

    # ---- host side ----
    i = lines.index(RUNPAD_SIG_OLD)
    ops.append((i, i + 1, RUNPAD_SIG_OLD, RUNPAD_SIG_NEW))
    i, _n = find_block(RUNPAD_TAIL_OLD)
    ops.append((i, i + 3, RUNPAD_TAIL_OLD, RUNPAD_TAIL_NEW))

    i = lines.index(EXPERT_SIG_OLD)
    ops.append((i, i + 1, EXPERT_SIG_OLD, EXPERT_SIG_NEW))
    i, _n = find_block(EXPERT_DOC_OLD)
    ops.append((i, i + 2, EXPERT_DOC_OLD, EXPERT_DOC_NEW))
    # module constant just above `def gqa_expert(`
    ge = lines.index(EXPERT_SIG_OLD)
    assert lines[ge - 1] == "\n" and lines[ge - 2] == "\n", (lines[ge - 2], lines[ge - 1])
    ops.append((ge - 2, ge, "\n\n", P2_KERNELS_CONST))

    i = lines.index(G2STACK_ANCHOR)
    ops.append((i, i + 1, G2STACK_ANCHOR, BSND_EFF_NEW))

    # the two flat spec builders (kv1 first, then kv2) — both end with the same
    # `block_N=block_N,\n                        ),\n` pattern
    for occ in (1, 2):
        i, _n = find_block(SPEC_KWARG_OLD, occ=occ)
        ops.append((i, i + 2, SPEC_KWARG_OLD, SPEC_KWARG_NEW))

    i = lines.index(CHAIN_KEY_OLD)
    ops.append((i, i + 1, CHAIN_KEY_OLD, CHAIN_KEY_NEW))

    i, _n = find_block(RUNPAD_CALL_OLD)
    ops.append((i, i + 2, RUNPAD_CALL_OLD, RUNPAD_CALL_NEW))

    # `    return _run` inside gqa_expert's prefill branch (the one right after
    # the _prefill_run_padded call)
    i = lines.index(RUN_TAIL_OLD, _n)
    ops.append((i, i + 1, RUN_TAIL_OLD, RUN_TAIL_NEW))

    i, _n = find_block(ADAPTER_BUILD_OLD)
    ops.append((i, i + 2, ADAPTER_BUILD_OLD, ADAPTER_BUILD_NEW))
    i, _n = find_block(ADAPTER_TAIL_OLD)
    ops.append((i, i + 4, ADAPTER_TAIL_OLD, ADAPTER_TAIL_NEW))
    return ops


def main():
    dry = "--dry" in sys.argv
    md5 = hashlib.md5(open(GQA_PY, "rb").read()).hexdigest()
    assert md5 == EXPECT_MD5, f"gqa.py md5 {md5} != shipped baseline {EXPECT_MD5}"
    lines = open(GQA_PY).read().splitlines(keepends=True)
    ops = build_ops(lines)
    # verify every site, then apply bottom-up
    for start, end, expect, _new in ops:
        got = "".join(lines[start:end])
        assert got == expect, f"site drift at line {start + 1}:\n--- got ---\n{got}--- want ---\n{expect}"
    for start, end, _expect, new in sorted(ops, key=lambda o: -o[0]):
        nl = new.splitlines(keepends=True)
        lines[start:end] = nl
    text = "".join(lines)
    ast.parse(text)
    diff = list(difflib.unified_diff(open(GQA_PY).read().splitlines(), text.splitlines(),
                                     "gqa.py (pre-P2)", "gqa.py (post-P2)", lineterm="", n=3))
    with open(os.path.join(HERE, "diff_iter8_p2_integration.txt"), "w") as f:
        f.write("\n".join(diff) + "\n")
    added = sum(1 for d in diff if d.startswith("+") and not d.startswith("+++"))
    removed = sum(1 for d in diff if d.startswith("-") and not d.startswith("---"))
    hunks = sum(1 for d in diff if d.startswith("@@"))
    print(f"edit sites: {len(ops)}   diff: +{added}/-{removed} lines in {hunks} hunks")
    # post-conditions on the new text
    assert text.count("out_bsnd") >= 20, text.count("out_bsnd")
    assert text.count("_P2_BSND_KERNELS") == 2, text.count("_P2_BSND_KERNELS")
    assert text.count("Output: T.Tensor(out_shape, dtype)") == 2
    assert text.count("Output[bz, bx * block_M + vid * half : bx * block_M + vid * half + half, hq0 + h, :],") == 2
    # 3 = the untouched g2stack kernel + the two new `else:` branches
    assert text.count("Output[bz, hq0 + h, bx * block_M + vid * half : bx * block_M + vid * half + half, :],") == 3
    assert text.count("bsnd_eff") >= 6
    longest = max((len(ln) for ln in text.splitlines()), default=0)
    print(f"longest line: {longest} (ruff line-length 140)")
    assert longest <= 140, "line-length violation"
    if dry:
        print("[DRY] no write")
        return
    with open(GQA_PY, "w") as f:
        f.write(text)
    new_md5 = hashlib.md5(open(GQA_PY, "rb").read()).hexdigest()
    print(f"WROTE {GQA_PY}\n  md5 {md5} -> {new_md5}")


if __name__ == "__main__":
    main()
