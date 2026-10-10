"""Stage-3 perf bench harness — REFERENCE state, ALL 20 cases (tooling only, NOT a deliverable).

Loads history_version/mha_impl_r4_perf7.py (r8-2 adopted state 5d466b5f; IR == ee5bf9c9
session-start baseline for every domain benched here: Form A single/multi + decode +
fbm single-block — only fbm generic case9/4 differ, and those use msprof_r8_base proper).

Usage:
  msprof op --kernel-name="main_kernel" --output=<prof_dir> \
      python examples/mha/perf_tuning/bench_mha.py --case case7 [--warmup 5] [--iters 20]

Timing is NOT taken from this script (msprof kernel time is the only perf
basis); it only prints a launch-count marker for the parser to group rows.
Case shapes/attrs are transcribed from cann-bench tasks/level4/mha/cases.yaml.
"""

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # examples/mha
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("mha_v7", os.path.join(os.path.dirname(_HERE), "history_version", "mha_impl_r4_perf7.py"))
_m = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_m)
mha = _m.mha  # noqa: E402

# (B, S, S_kv, N, D), dtype, scaleValue, is_causal, value_range  — cases.yaml
CASES = {
    # S-float16-B2S128N8D64-prefill-causal
    "case1": ((2, 128, 128, 8, 64), "float16", -1.0, True, (-1, 1)),
    # L-float16-B4S1024N32D128-long-prefill-causal (worst gap: 16762us vs 432.8us)
    "case7": ((4, 1024, 1024, 32, 128), "float16", -1.0, True, (-1, 1)),
    # M-float16-B1S1Skv2048N32D128-decode-single-batch
    "case10": ((1, 1, 2048, 32, 128), "float16", -1.0, False, (-1, 1)),
    # L-float16-B128S1Skv128N32D128-decode-large-batch
    "case14": ((128, 1, 128, 32, 128), "float16", -1.0, False, (-1, 1)),
    # S-float16-B2S128N32D128-zero-input (non-causal, all-zero)
    "case20": ((2, 128, 128, 32, 128), "float16", -1.0, False, (0, 0)),
    # extra diagnostics (not in the §12 representative set, opt-in):
    # L-bfloat16-B4S1024N32D128-very-long-prefill-causal (case8 twin of case7)
    "case8": ((4, 1024, 1024, 32, 128), "bfloat16", -1.0, True, (-1, 1)),
    # L-float16-B60Sq2Skv512N32D128-MTP-large-batch-causal (S=2 causal)
    "case16": ((60, 2, 512, 32, 128), "float16", -1.0, True, (-1, 1)),
    # L-float16-B2S512N32D128-prefill-causal bf16 (case3, mid prefill)
    "case3": ((2, 512, 512, 32, 128), "bfloat16", -1.0, True, (-1, 1)),
    # session-2 #10b: D=256 decode beneficiaries + prefill D=256 guard
    # L-float16-B60S1Skv512N16D256-decode-large-batch-D256
    "case15": ((60, 1, 512, 16, 256), "float16", -1.0, False, (-1, 1)),
    # L-float16-B60Sq2Skv512N16D256-MTP-large-batch-D256
    "case18": ((60, 2, 512, 16, 256), "float16", -1.0, False, (-1, 1)),
    # S-float16-B2S256N8D256-D256-causal (prefill guard)
    "case4": ((2, 256, 256, 8, 256), "float16", -1.0, True, (-1, 1)),
    # session-3: L-float16-B4S1024N16D256-long-prefill-D256-causal (Form B P3, 0.24x M1 gap)
    "case9": ((4, 1024, 1024, 16, 256), "float16", -1.0, True, (-1, 1)),
    # session-3: L-bfloat16-B4S1024N32D128 twin (case8 = bf16 case7, Form A P2 guard)
    # session-3: L-float16-B16S1Skv1024N32D128-decode (case11, Form B D1 mid decode)
    "case11": ((16, 1, 1024, 32, 128), "float16", -1.0, False, (-1, 1)),
    # session-5 (#R5-1 Form A single-block task batching targets, cases.yaml):
    # S-bfloat16-B2Sq512Skv128N16D128-cross (Form A single_kv_block, 12.8 task/core)
    "case6": ((2, 512, 128, 16, 128), "bfloat16", -1.0, False, (-1, 1)),
    # S-float16-B2S128N32D128-explicit-scale-causal (Form A single_kv_block)
    "case19": ((2, 128, 128, 32, 128), "float16", 0.08838, True, (-1, 1)),
    # session-5 guards: Form A multi-kv generic path (IR must stay byte-identical)
    # M-bfloat16-B4S256N16D64-prefill-causal (P1 separate L0 pool, kv_iters=2)
    "case2": ((4, 256, 256, 16, 64), "bfloat16", -1.0, True, (-1, 1)),
    # S-float16-B2Sq128Skv256N12D64-cross (P1, kv_iters=2, non-causal bound mask)
    "case5": ((2, 128, 256, 12, 64), "float16", -1.0, False, (-1, 1)),
    # session-8 Step C: full-20-case joint wall (shapes transcribed verbatim from bench_mha.py)
    # L-bfloat16-B32S1Skv512N32D128-decode (B-gv0 fallback domain)
    "case12": ((32, 1, 512, 32, 128), "bfloat16", -1.0, False, (-1, 1)),
    # L-float16-B60S1Skv512N32D128-decode-large-batch (B-gv0 fallback domain)
    "case13": ((60, 1, 512, 32, 128), "float16", -1.0, False, (-1, 1)),
    # L-bfloat16-B96Sq2Skv256N32D128-MTP-causal (B-gv0 fallback domain, S=2)
    "case17": ((96, 2, 256, 32, 128), "bfloat16", -1.0, True, (-1, 1)),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True, choices=sorted(CASES))
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    (B, S, S_kv, N, D), dtype_s, scale, causal, (lo, hi) = CASES[args.case]
    dt = getattr(torch, dtype_s)
    if lo == 0 and hi == 0:
        q = torch.zeros(B, S, N, D, dtype=dt)
        k = torch.zeros(B, S_kv, N, D, dtype=dt)
        v = torch.zeros(B, S_kv, N, D, dtype=dt)
    else:
        # CPU gen + single H2D (memcpy, not AI-core kernels -> no CSV pollution)
        q = torch.empty(B, S, N, D, dtype=dt).uniform_(lo, hi)
        k = torch.empty(B, S_kv, N, D, dtype=dt).uniform_(lo, hi)
        v = torch.empty(B, S_kv, N, D, dtype=dt).uniform_(lo, hi)
    q, k, v = q.to("npu"), k.to("npu"), v.to("npu")
    torch.npu.synchronize()

    # warmup launches (JIT compile happens on the first call, host-side)
    for _ in range(args.warmup):
        out = mha(q, k, v, scaleValue=scale, is_causal=causal)
    torch.npu.synchronize()
    print(f"[BENCH] {args.case} warmup_done n={args.warmup}")

    for _ in range(args.iters):
        out = mha(q, k, v, scaleValue=scale, is_causal=causal)
    torch.npu.synchronize()
    print(f"[BENCH] {args.case} iters_done n={args.iters} out_sum={float(out.float().sum()):.4f}")


if __name__ == "__main__":
    main()
