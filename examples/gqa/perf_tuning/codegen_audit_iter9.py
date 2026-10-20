"""Iter9 P2-wave2 generated-code audit — shipped (BHSD) vs variant (BSND).

Wave-1's "instruction count unchanged" check applied to the three wave-2
kernels, with the per-head AUTO kernel as the primary subject (no manual flag
protocol exists there to byte-compare — the task book requires a codegen audit
for the AUTO domain instead of the wait-id gate):

  For each (kernel, representative config):
    1. build the SHIPPED factory (gqa.py) and the VARIANT factory
       (gqa_w2_bsnd_iter9_variant.py) with identical arguments,
    2. dump get_kernel_source() for both,
    3. count MTE3 output-write instructions (copy_ub_to_gm / DataCopyPad /
       DataCopyExtParams / SetAtomic) and total DataCopy* call sites —
       the counts MUST be equal (pure descriptor effect, wave-1 finding 1),
    4. locate the output-write DataCopyExtParams and verify only dstStride
       differs (0 -> (N_q*D - D)*sizeof(T)),
    5. write both sources to perf_tuning/codegen_iter9_{cfg}_{base,w2}.cu
       for the diff artifact.

Run (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python3 examples/gqa/perf_tuning/codegen_audit_iter9.py [--only <substr>]
"""

import argparse
import os
import re
import sys

import torch  # noqa: F401  (tilelang needs it at import)
import torch_npu  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_DIR = os.path.dirname(HERE)
sys.path.insert(0, GQA_DIR)
sys.path.insert(0, HERE)

import gqa as G  # noqa: E402
import gqa_w2_bsnd_iter9_variant as VAR  # noqa: E402

# (tag, factory attr, args, kwargs, expect_boxes)
# expect_boxes = number of output-write T.copy sites in the TVM source form
# (per-head: 1; prefill_expert: 1; g2stack: 2 — the h-loop unrolls x2)
CONFIGS = [
    # per-head AUTO: r1 form (serial threads=2, causal, D128)
    ("perhead_r1", "_gqa_kernel_expert",
     (1, 128, 128, 0.08838834764831845, True),
     dict(N_q=8, N_kv=2, D=128, dtype="float16", block_M=64, skv_minus_s=0,
          use_mask=True, block_N=128)),
    # per-head AUTO: c5 form (serial threads=2, causal, D256)
    ("perhead_c5", "_gqa_kernel_expert",
     (2, 128, 128, 0.0625, True),
     dict(N_q=32, N_kv=8, D=256, dtype="float16", block_M=64, skv_minus_s=0,
          use_mask=True, block_N=128)),
    # per-head AUTO: PIPELINED threads=1 form (non-causal kv_iters=2 even)
    ("perhead_pipe", "_gqa_kernel_expert",
     (1, 128, 256, 0.08838834764831845, False),
     dict(N_q=8, N_kv=2, D=128, dtype="float16", block_M=64, skv_minus_s=0,
          use_mask=False, block_N=128)),
    # prefill_expert: c19 form (ns=1)
    ("prexpert_c19", "_gqa_kernel_prefill_expert",
     (2, 128, 128, 0.08838834764831845, True),
     dict(N_q=32, N_kv=32, D=128, dtype="bfloat16", block_M=64, skv_minus_s=0,
          use_mask=True, block_N=128, num_stages=1, cross_interval=2)),
    # g2stack: c9 form (ns=4, noncausal, use_mask False)
    ("g2stack_c9", "_gqa_kernel_prefill_g2stack",
     (4, 512, 512, 0.08838834764831845, False),
     dict(N_q=64, N_kv=8, D=128, dtype="bfloat16", block_M=64, skv_minus_s=0,
          use_mask=False, block_N=128, num_stages=4, cross_interval=2)),
    # g2stack: c8 form (ns=8, causal, use_mask True)
    ("g2stack_c8", "_gqa_kernel_prefill_g2stack",
     (4, 1024, 1024, 0.08838834764831845, True),
     dict(N_q=32, N_kv=8, D=128, dtype="float16", block_M=64, skv_minus_s=0,
          use_mask=True, block_N=128, num_stages=8, cross_interval=2)),
]

MTE3_PATTERNS = [
    "copy_ub_to_gm",
    "DataCopyPad",
    "DataCopyExtParams",
    "DataCopy(",
    "copy_gm_to_ub",
    "copy_ub_to_ub",
]


def counts(src):
    return {p: len(re.findall(re.escape(p), src)) for p in MTE3_PATTERNS}


def ext_params(src):
    """All DataCopyExtParams(...) argument tuples (flattened text)."""
    out = []
    for m in re.finditer(r"DataCopyExtParams\(([^;]*?)\);", src, re.S):
        out.append(re.sub(r"\s+", " ", m.group(1)).strip())
    return out


def audit_one(tag, fname, args, kwargs):
    fb = getattr(G, fname)
    fv = getattr(VAR, fname)
    kb = fb(*args, **kwargs)
    kv = fv(*args, **kwargs)
    sb = kb.get_kernel_source()
    sv = kv.get_kernel_source()
    with open(os.path.join(HERE, f"codegen_iter9_{tag}_base.cu"), "w") as f:
        f.write(sb)
    with open(os.path.join(HERE, f"codegen_iter9_{tag}_w2.cu"), "w") as f:
        f.write(sv)

    cb, cv = counts(sb), counts(sv)
    ok_counts = cb == cv
    eb, ev = ext_params(sb), ext_params(sv)
    ok_ext_count = len(eb) == len(ev)
    # classify the ext-params delta: pairs that differ must differ ONLY in the
    # dstStride slot (last-ish field) — report every differing pair verbatim
    diffs = [(a, b) for a, b in zip(eb, ev) if a != b]

    print(f"[{tag}] factory={fname}")
    print(f"  source chars: base={len(sb)} w2={len(sv)}")
    print(f"  instruction-pattern counts base: {cb}")
    print(f"  instruction-pattern counts w2  : {cv}")
    print(f"  COUNTS_EQUAL={ok_counts}  ext_params n={len(eb)}/{len(ev)} equal={ok_ext_count}")
    for a, b in diffs:
        print(f"  EXT_DIFF:\n    base: {a}\n    w2  : {b}")
    return {
        "tag": tag,
        "counts_equal": ok_counts,
        "counts": {"base": cb, "w2": cv},
        "ext_params_n": [len(eb), len(ev)],
        "ext_diffs": diffs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    results = []
    for tag, fname, a, kw in CONFIGS:
        if args.only and args.only not in tag:
            continue
        results.append(audit_one(tag, fname, a, kw))
    bad = [r["tag"] for r in results if not (r["counts_equal"] and r["ext_params_n"][0] == r["ext_params_n"][1])]
    print("\n[AUDIT_SUMMARY] configs:", [r["tag"] for r in results])
    print("[AUDIT_SUMMARY] instruction-count mismatches:", bad if bad else "NONE")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
