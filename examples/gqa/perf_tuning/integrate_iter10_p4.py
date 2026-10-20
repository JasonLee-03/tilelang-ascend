"""Iter10 P4 Gate-3 integration — transpose_qkv elimination for the flat
kv1/kv2 aligned domain in gqa.py (main kernel reads BSND strided GM directly).

Line-anchored, assertion-guarded edit script (integrate_iter8/9 pattern):
every edit site's ORIGINAL content is asserted before replacement, edits are
applied bottom-up so earlier line numbers stay valid, and the result is
AST-checked + diffed + post-condition-counted.  gqa.py md5 must equal the
shipped baseline 606b07ff474ec42b8244f0831327d75b on entry.

Integration surface (minimal; nothing else in the file changes):
  1. The two flat factories (`_gqa_kernel_prefill_g2stack_flat`,
     `_gqa_kernel_prefill_g2stack_flat_kv2`): new kwarg `in_bsnd=False`;
     `qkv_q_shape`/`qkv_kv_shape` host variables; Q/K/V declared with them;
     the emit_qload / emit_g1 K / emit_g1 V boxes become `if in_bsnd:`
     branches (BSND strided boxes) with the shipped BHSD boxes preserved
     VERBATIM in the `else` — so in_bsnd=False traces to byte-identical
     generated code (same pattern as wave-1's out_bsnd).
  2. `gqa_expert`: new kwarg `in_bsnd=False`; `in_bsnd_eff` gate (BOTH row
     dims aligned: S_padded == S AND S_kv_padded == S_kv); the two flat spec
     builders pass `in_bsnd=in_bsnd_eff`; chain_key carries in_bsnd_eff;
     `_P4_BSND_KERNELS` constant; capability flag `_run.in_bsnd`.
  3. `gqa()` adapter: kernel resolution MOVED ABOVE the input transpose
     (pure host reordering — gqa_expert depends only on shapes; NPU launch
     order unchanged), gqa_expert asked with in_bsnd=True, and when
     `kernel_fn.in_bsnd` the transpose_qkv launch is SKIPPED (the BSND
     originals are passed through — zero d2d, 561103 discipline).  The
     S_out variable only exists on the else branch, and the later
     `if S_out > S_bs:` line is reachable ONLY when out_bsnd is False, which
     implies in_bsnd False (the in_bsnd domain is a strict subset: it
     requires the same S alignment plus S_kv alignment, and
     _P4_BSND_KERNELS ⊂ _P2_BSND_KERNELS) => no NameError path.
  4. Docstring/comment accuracy updates at the touched sites.
  NOT touched: per-head AUTO ladder, g2stack, prefill_expert, g2, decode,
  transpose_qkv itself (still runs on every non-gated shape), transpose_out /
  out_plan / _retile_out_plan (P1), _prefill_run_padded (aligned domain never
  enters its pad branch), test_gqa.py.

Run (repo root):
    python3 examples/gqa/perf_tuning/integrate_iter10_p4.py [--dry]
"""

import ast
import difflib
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_DIR = os.path.dirname(HERE)
GQA_PY = os.path.join(GQA_DIR, "gqa.py")
EXPECT_MD5 = "606b07ff474ec42b8244f0831327d75b"

# ---------------------------------------------------------------- kernel side --
SIG_OLD = "    out_bsnd=False,\n"
SIG_NEW = "    out_bsnd=False,\n    in_bsnd=False,\n"

DOC_KV1_OLD = """    chain (Gate 0: 23/23, perf_tuning/run_iter8_p2_gate0.py).
    \"\"\"
"""
DOC_KV1_NEW = """    chain (Gate 0: 23/23, perf_tuning/run_iter8_p2_gate0.py).

    in_bsnd ([S3-iter10-P4]): declare Q/K/V BSND ([B, S_padded, N_q, D] /
    [B, S_kv_padded, N_kv, D]) and read the emit_qload / emit_g1 load boxes
    strided (row stride N*D) instead of the BHSD contiguous boxes, so the
    host can skip the transpose_qkv kernel entirely.  Callers MUST only
    request it when S_padded == S AND S_kv_padded == S_kv (both row dims
    aligned — the transpose's zero-fill pad semantics are then empty by
    construction).  Each box stays ONE copy_gm_to_l1 (realSrcN folds the
    singleton head dim, src/op/ascend.cc compute_strideN; the decode
    dense_expert K/V loads are the production precedent) — instruction count
    and every flag line unchanged.  Arithmetic untouched => outputs are
    BIT-IDENTICAL to the transpose_qkv + BHSD chain (Gate 0: 25/25,
    perf_tuning/run_iter10_p4_gate0.py).
    \"\"\"
"""

DOC_KV2_OLD = """    out_bsnd: see _gqa_kernel_prefill_g2stack_flat ([S3-iter8-P2], identical
    mechanism and gate — BSND direct write, transpose_out eliminated).
    \"\"\"
"""
DOC_KV2_NEW = """    out_bsnd: see _gqa_kernel_prefill_g2stack_flat ([S3-iter8-P2], identical
    mechanism and gate — BSND direct write, transpose_out eliminated).

    in_bsnd: see _gqa_kernel_prefill_g2stack_flat ([S3-iter10-P4], identical
    mechanism and dual-alignment gate — BSND strided direct reads,
    transpose_qkv eliminated).
    \"\"\"
"""

# qload/K/V/decl edit texts — IDENTICAL to gen_variant_iter10_p4.py SITES
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

# ------------------------------------------------------------------ host side --
P4K_INSERT = '''# [S3-iter10-P4] the only prefill kernels that can read their Q/K/V inputs
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

EXPERT_SIG_OLD = "def gqa_expert(B, S, S_kv, sm_scale, is_causal, N_q=8, N_kv=2, D=128, dtype=\"float16\", out_bsnd=False):\n"
EXPERT_SIG_NEW = "def gqa_expert(B, S, S_kv, sm_scale, is_causal, N_q=8, N_kv=2, D=128, dtype=\"float16\", out_bsnd=False, in_bsnd=False):\n"

BSNDEFF_OLD = "    bsnd_eff = bool(out_bsnd) and S_padded == S\n"
BSNDEFF_NEW = """    bsnd_eff = bool(out_bsnd) and S_padded == S

    # [S3-iter10-P4] BSND direct-READ eligibility (host gate): the caller must
    # ask for it AND the shape must be aligned on BOTH row dims, so
    # S_padded == S and S_kv_padded == S_kv — the strided load windows then
    # read exactly the caller's BSND tensors and transpose_qkv's zero-fill pad
    # semantics are never needed (non-aligned shapes keep the transpose, which
    # also builds the padded K/V columns).  The capable kernels are
    # _P4_BSND_KERNELS (the two flat expert kernels); the capability actually
    # granted is re-checked against the resolved spec below.
    in_bsnd_eff = bool(in_bsnd) and S_padded == S and S_kv_padded == S_kv
"""

SPEC_OLD = "                            out_bsnd=bsnd_eff,\n"
SPEC_NEW = "                            out_bsnd=bsnd_eff,\n                            in_bsnd=in_bsnd_eff,\n"

CHAINKEY_OLD = '    chain_key = ("prefill", B, S, S_kv, N_q, N_kv, D, dtype, is_causal, sm_scale, bsnd_eff)\n'
CHAINKEY_NEW = (
    "    # [S3-iter10-P4] in_bsnd_eff in the key as well: a factory caller (BHSD\n"
    "    # inputs) and the adapter (BSND request) must never share one cached chain.\n"
    '    chain_key = ("prefill", B, S, S_kv, N_q, N_kv, D, dtype, is_causal, sm_scale, bsnd_eff, in_bsnd_eff)\n'
)

RUNFLAG_OLD = """    _run.out_bsnd = bool(bsnd_eff) and chosen in _P2_BSND_KERNELS

    return _run
"""
RUNFLAG_NEW = """    _run.out_bsnd = bool(bsnd_eff) and chosen in _P2_BSND_KERNELS

    # [S3-iter10-P4] capability flag consumed by the gqa() adapter: True only
    # when the resolved kernel really reads BSND inputs.  A compile fallback
    # down the chain leaves it False, so the adapter keeps calling
    # transpose_qkv and the BHSD contract holds.  in_bsnd => out_bsnd (the
    # gate is a strict subset and _P4_BSND_KERNELS ⊂ _P2_BSND_KERNELS).
    _run.in_bsnd = bool(in_bsnd_eff) and chosen in _P4_BSND_KERNELS

    return _run
"""

ADAPTER_A1_OLD = """    decode_direct = 1 <= S_bs <= 4

    if not decode_direct:
        # Block-aligned pad targets for the transpose outputs — MUST
        # match gqa_expert's prefill S_padded (block_M=64) / S_kv_padded
        # (block_N=128) so the padded BHSD tensors the transpose emits are
        # exactly what the chosen kernel declares (every chain candidate is
        # built with the unified S_padded).  Pad rows are zero-filled
        # in-kernel: zero torch device-side copy ops on the fp16/bf16
        # non-aligned path (ViewCopyAiCore binary missing on some CANN 9.x
        # SoC packages, 561103).
        S_out = ((S_bs + 63) // 64) * 64
        S_kv_out = ((S_kv_bs + 127) // 128) * 128
        res = transpose_qkv(q, k_t, v_t, S_out=S_out, S_kv_out=S_kv_out)
        if res is not None:
            q_bhsd, k_bhsd, v_bhsd = res
        else:
            q_bhsd = q.transpose(1, 2).contiguous()
            k_bhsd = k_t.transpose(1, 2).contiguous()
            v_bhsd = v_t.transpose(1, 2).contiguous()
"""
ADAPTER_A1_NEW = """    decode_direct = 1 <= S_bs <= 4

    # [S3-iter10-P4] Resolve/compile the kernel BEFORE the input transpose:
    # the in_bsnd capability flag decides whether transpose_qkv runs at all.
    # Pure host-side reordering — gqa_expert depends only on shapes, and the
    # NPU launch order below is unchanged (transpose (when needed) -> mask ->
    # main).  (Moved verbatim from below the mask block; the only content
    # change is the added in_bsnd=True request.)
    cache_key = (B_bs, S_bs, S_kv_bs, N_q_bs, N_kv_bs, D_bs, dtype_str, is_causal, sm_scale_val)
    if cache_key not in _COMPILED_CACHE:
        _COMPILED_CACHE[cache_key] = gqa_expert(
            B=B_bs,
            S=S_bs,
            S_kv=S_kv_bs,
            sm_scale=sm_scale_val,
            is_causal=is_causal,
            N_q=N_q_bs,
            N_kv=N_kv_bs,
            D=D_bs,
            dtype=dtype_str,
            # [S3-iter8-P2] ask for the BSND direct-write domain (gqa_expert
            # applies the alignment gate and reports the real capability on
            # _run.out_bsnd; out-of-domain shapes silently keep BHSD).
            out_bsnd=True,
            # [S3-iter10-P4] ask for the BSND direct-read domain (flat
            # kv1/kv2 only; gqa_expert applies the dual-alignment gate and
            # reports the capability on _run.in_bsnd).
            in_bsnd=True,
        )
    kernel_fn = _COMPILED_CACHE[cache_key]

    if not decode_direct:
        if getattr(kernel_fn, "in_bsnd", False):
            # [S3-iter10-P4] BSND direct-read domain (S % 64 == 0 AND
            # S_kv % 128 == 0 AND the resolved kernel is flat kv1/kv2): the
            # main kernel reads the caller's BSND tensors via strided MTE2
            # boxes (the decode dense_expert production pattern), so the
            # transpose_qkv launch is eliminated — measured e2e -13..-25% on
            # this domain (perf_tuning/candidate_iter10_P4.json).  The
            # variable names keep the shipped q_bhsd spelling; on this branch
            # they hold the BSND ORIGINALS (no d2d copy is ever materialized
            # — 561103 discipline).  S_out/S_kv_out are deliberately NOT
            # defined here: the later `if S_out > S_bs:` line is reachable
            # only when out_bsnd is False, which implies in_bsnd False
            # (strict-subset gate), i.e. the else branch ran.
            q_bhsd, k_bhsd, v_bhsd = q, k_t, v_t
        else:
            # Block-aligned pad targets for the transpose outputs — MUST
            # match gqa_expert's prefill S_padded (block_M=64) / S_kv_padded
            # (block_N=128) so the padded BHSD tensors the transpose emits are
            # exactly what the chosen kernel declares (every chain candidate is
            # built with the unified S_padded).  Pad rows are zero-filled
            # in-kernel: zero torch device-side copy ops on the fp16/bf16
            # non-aligned path (ViewCopyAiCore binary missing on some CANN 9.x
            # SoC packages, 561103).
            S_out = ((S_bs + 63) // 64) * 64
            S_kv_out = ((S_kv_bs + 127) // 128) * 128
            res = transpose_qkv(q, k_t, v_t, S_out=S_out, S_kv_out=S_kv_out)
            if res is not None:
                q_bhsd, k_bhsd, v_bhsd = res
            else:
                q_bhsd = q.transpose(1, 2).contiguous()
                k_bhsd = k_t.transpose(1, 2).contiguous()
                v_bhsd = v_t.transpose(1, 2).contiguous()
"""

ADAPTER_A2_OLD = """    cache_key = (B_bs, S_bs, S_kv_bs, N_q_bs, N_kv_bs, D_bs, dtype_str, is_causal, sm_scale_val)
    if cache_key not in _COMPILED_CACHE:
        _COMPILED_CACHE[cache_key] = gqa_expert(
            B=B_bs,
            S=S_bs,
            S_kv=S_kv_bs,
            sm_scale=sm_scale_val,
            is_causal=is_causal,
            N_q=N_q_bs,
            N_kv=N_kv_bs,
            D=D_bs,
            dtype=dtype_str,
            # [S3-iter8-P2] ask for the BSND direct-write domain (gqa_expert
            # applies the alignment gate and reports the real capability on
            # _run.out_bsnd; out-of-domain shapes silently keep BHSD).
            out_bsnd=True,
        )
    kernel_fn = _COMPILED_CACHE[cache_key]
"""
ADAPTER_A2_NEW = """    # [S3-iter10-P4] (kernel resolution moved ABOVE the input transpose —
    # the in_bsnd capability gate needs kernel_fn before transpose_qkv runs.)
"""

EXPERT_DOC_OLD = """    The request is honoured ONLY when S_padded == S (no row crop needed) AND
    the resolved kernel is in _P2_BSND_KERNELS; every other shape (per-head
    AUTO ladder, g2 fallback, non-aligned S) keeps the BHSD contract above
    byte-identically.
"""
EXPERT_DOC_NEW = """    The request is honoured ONLY when S_padded == S (no row crop needed) AND
    the resolved kernel is in _P2_BSND_KERNELS; every other shape (per-head
    AUTO ladder, g2 fallback, non-aligned S) keeps the BHSD contract above
    byte-identically.

    in_bsnd=True ([S3-iter10-P4], prefill only) additionally asks the
    dual-aligned flat kv1/kv2 domain to READ its Q/K/V inputs BSND-strided
    directly (transpose_qkv eliminated), reported on `_run.in_bsnd`.
    Honoured ONLY when S_padded == S AND S_kv_padded == S_kv AND the resolved
    kernel is in _P4_BSND_KERNELS; the caller contract then becomes
    `run(Q_bsnd, K_bsnd, V_bsnd, Mask_b, crop=False)` with the SAME alignment
    guarantee.  Every other shape keeps the BHSD input contract above
    byte-identically (in_bsnd=False generates the shipped code).
"""


def build_ops(lines):
    """[(start_0based, end_0based_exclusive, expect_text, new_text)] — every
    site pinned by verified line number against md5 606b07ff..., applied
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

    nA1 = len(ADAPTER_A1_OLD.splitlines(keepends=True))
    nA2 = len(ADAPTER_A2_OLD.splitlines(keepends=True))
    nDocKv1 = len(DOC_KV1_OLD.splitlines(keepends=True))
    nDocKv2 = len(DOC_KV2_OLD.splitlines(keepends=True))
    nDocEx = len(EXPERT_DOC_OLD.splitlines(keepends=True))
    nQload = len(_QLOAD_OLD.splitlines(keepends=True))
    nK = len(_K_OLD_KV1.splitlines(keepends=True))
    nDecl = len(_DECL_OLD.splitlines(keepends=True))
    nRun = len(RUNFLAG_OLD.splitlines(keepends=True))
    nBsnd = len(BSNDEFF_NEW.splitlines(keepends=True))  # old is 1 line

    ops = [
        # ---- adapter (bottom-up) ----
        op(5894, nA2, ADAPTER_A2_OLD, ADAPTER_A2_NEW),
        op(5845, nA1, ADAPTER_A1_OLD, ADAPTER_A1_NEW),
        # ---- gqa_expert host machinery ----
        op(4996, nRun, RUNFLAG_OLD, RUNFLAG_NEW),
        op(4957, 1, CHAINKEY_OLD, CHAINKEY_NEW),
        op(4816, 1, SPEC_OLD, SPEC_NEW),               # flat kv2 spec builder
        op(4770, 1, SPEC_OLD, SPEC_NEW),               # flat kv1 spec builder
        op(4724, 1, BSNDEFF_OLD, BSNDEFF_NEW),
        op(4406, nDocEx, EXPERT_DOC_OLD, EXPERT_DOC_NEW),
        op(4386, 1, EXPERT_SIG_OLD, EXPERT_SIG_NEW),
        insert_before(4386, EXPERT_SIG_OLD, P4K_INSERT),
        # ---- flat kv2 kernel ----
        op(3912, nDecl, _DECL_OLD, _DECL_NEW),
        op(3674, nK, _V_OLD_KV2, _V_NEW_KV2),
        op(3665, nK, _K_OLD_KV2, _K_NEW_KV2),
        op(3648, nQload, _QLOAD_OLD, _QLOAD_NEW),
        op(3566, nDocKv2, DOC_KV2_OLD, DOC_KV2_NEW),
        op(3562, 1, SIG_OLD, SIG_NEW),
        # ---- flat kv1 kernel ----
        op(3363, nDecl, _DECL_OLD, _DECL_NEW),
        op(3162, nK, _V_OLD_KV1, _V_NEW_KV1),
        op(3154, nK, _K_OLD_KV1, _K_NEW_KV1),
        op(3138, nQload, _QLOAD_OLD, _QLOAD_NEW),
        op(3067, nDocKv1, DOC_KV1_OLD, DOC_KV1_NEW),
        op(3055, 1, SIG_OLD, SIG_NEW),
    ]
    return ops


def main():
    dry = "--dry" in sys.argv
    md5 = hashlib.md5(open(GQA_PY, "rb").read()).hexdigest()
    assert md5 == EXPECT_MD5, f"gqa.py md5 {md5} != shipped baseline {EXPECT_MD5}"
    orig_text = open(GQA_PY).read()
    lines = orig_text.splitlines(keepends=True)
    ops = build_ops(lines)  # every site asserted inside build_ops
    # sites must not overlap
    spans = sorted((s, e) for s, e, _, _ in ops)
    for i in range(1, len(spans)):
        assert spans[i][0] >= spans[i - 1][1], f"overlapping sites: {spans[i-1]} {spans[i]}"
    for start, end, _expect, new in sorted(ops, key=lambda o: -o[0]):
        lines[start:end] = new.splitlines(keepends=True)
    text = "".join(lines)
    ast.parse(text)

    # ---- post-conditions ----
    assert text.count("    in_bsnd=False,\n") == 2                     # two factory sigs
    assert text.count("qkv_q_shape = [B, S_padded, N_q, D] if in_bsnd") == 2
    assert text.count("Q: T.Tensor(qkv_q_shape, dtype)") == 2
    # prefill_expert + g2stack keep their BHSD Q decls (first-wave scope);
    # the two flat kernels' decls were switched to qkv_q_shape
    assert text.count("Q: T.Tensor([B, N_q, S_padded, D], dtype),  # type: ignore  BHSD\n") == 2
    assert text.count("T.copy(K[bz, 0:block_N, kg, :], k_l1_0)") == 2  # kv1 + kv2
    assert text.count("T.copy(K[bz, block_N : 2 * block_N, kg, :], k_l1_1)") == 1
    assert text.count("T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, hq0, :]") == 2
    # shipped BHSD boxes preserved verbatim in else branches
    assert text.count("T.copy(K[bz, kg, 0:block_N, :], k_l1_0)") == 2
    assert text.count("T.copy(Q[bz, hq0, bx * block_M : (bx + 1) * block_M, :], q_l1[0:block_M, :])") == 2  # shipped box preserved in each else branch
    # host machinery
    assert text.count("_P4_BSND_KERNELS") == 5                          # def + docstring + 2 comments + capability line
    assert text.count("in_bsnd_eff = bool(in_bsnd) and S_padded == S and S_kv_padded == S_kv") == 1
    assert text.count("in_bsnd=in_bsnd_eff,") == 2                      # two spec builders
    assert text.count("_run.in_bsnd = bool(in_bsnd_eff) and chosen in _P4_BSND_KERNELS") == 1
    assert text.count("sm_scale, bsnd_eff, in_bsnd_eff)") == 1          # chain_key
    assert text.count("def gqa_expert(B, S, S_kv, sm_scale, is_causal, N_q=8, N_kv=2, D=128, dtype=\"float16\", out_bsnd=False, in_bsnd=False):") == 1
    # adapter
    assert text.count("q_bhsd, k_bhsd, v_bhsd = q, k_t, v_t") == 1
    assert text.count("in_bsnd=True,\n") == 1
    assert text.count("if getattr(kernel_fn, \"in_bsnd\", False):") == 1
    assert text.count("res = transpose_qkv(q, k_t, v_t, S_out=S_out, S_kv_out=S_kv_out)") == 1  # else branch intact
    assert text.count("cache_key = (B_bs,") == 1                        # resolution moved, not duplicated
    # untouched surfaces
    assert text.count("_P2_BSND_KERNELS = (") == 1
    assert "def _transpose_qkv(" in text and "def transpose_out(" in text
    assert text.count("_gqa_kernel_expert(") >= 2                       # per-head factory + ladder call intact
    # line-length discipline (repo ruff line-length = 140)
    longest = max((len(ln), i + 1) for i, ln in enumerate(text.splitlines()))
    assert longest[0] <= 140, f"line {longest[1]} is {longest[0]} chars (> 140)"

    diff = list(difflib.unified_diff(orig_text.splitlines(), text.splitlines(),
                                     fromfile="gqa.py (606b07ff)", tofile="gqa.py (iter10 P4)",
                                     lineterm="", n=3))
    hunks = [ln for ln in diff if ln.startswith("@@")]
    adds = sum(1 for ln in diff if ln.startswith("+") and not ln.startswith("+++"))
    dels = sum(1 for ln in diff if ln.startswith("-") and not ln.startswith("---"))
    print(f"edit sites: {len(ops)}  hunks: {len(hunks)}  +{adds}/-{dels} lines  "
          f"longest line: {longest[0]} (at {longest[1]})")
    with open(os.path.join(HERE, "diff_iter10_p4_integration.txt"), "w") as f:
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
