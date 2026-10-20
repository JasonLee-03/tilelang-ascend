"""Iter6 roofline analysis probe: single-shot GQA run + dispatch verification.

Analysis-only (Stage 3 iter6): runs the PUBLIC gqa() API EXACTLY ONCE per
invocation (transpose_qkv -> main_kernel -> transpose_out on prefill;
BSND-direct main_kernel on decode), then prints the resolved dispatch chain
so every msprof sample is attributable to a verified kernel domain.

NO monkey-patching, NO variant loading — the shipped gqa.py is measured
as-is (md5 bd470e072d0a48faeac1e786306932e8).  gqa.py / test_gqa.py are
NOT modified by anything in this iteration.

Shape table (iter3 six-condition matrix, R4 now dispatches to the iter5
g2stack_flat_kv2 kernel):
    r1: B1 S128 Skv128 Nq8  Nkv2 causal   fp16  -> per_head_bm64
    r2: B2 S128 Skv128 Nq32 Nkv8 causal   fp16  -> g2stack_flat      (bench c1-like)
    r3: B2 S128 Skv128 Nq128 Nkv8 causal  bf16  -> g2stack_flat      (bench c3-like)
    r4: B4 S256 Skv256 Nq64 Nkv8 noncausal bf16 -> g2stack_flat_kv2  (bench c4-like)
    r5: B16 S1  Skv2048 Nq32 Nkv8 noncausal fp16 -> dense_expert     (bench c11-like)
    r6: B2 S128 Skv128 Nq48 Nkv4 noncausal fp16 -> g2stack_flat      (bench c2-like)

Usage (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:$LD_LIBRARY_PATH
    msprof op --kernel-name=main_kernel --aic-metrics=<groups> \
        --output=<dir> python3 examples/gqa/perf_tuning/prof_gqa_iter6.py --shape r3
    # all-kernel chain capture (aux separation):
    msprof op --launch-count=3 --aic-metrics=<groups> \
        --output=<dir> python3 examples/gqa/perf_tuning/prof_gqa_iter6.py --shape r3
"""

import argparse
import os
import sys

import torch
import torch_npu  # noqa: F401  (registers the npu backend)

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_DIR = os.path.dirname(HERE)
sys.path.insert(0, GQA_DIR)

import gqa as G  # noqa: E402

SHAPES = {
    # name: (B, S, S_kv, N_q, N_kv, D, causal, dtype, expected_domain)
    "r1": (1, 128, 128, 8, 2, 128, True, "fp16", "per_head"),
    "r2": (2, 128, 128, 32, 8, 128, True, "fp16", "g2stack_flat"),
    "r3": (2, 128, 128, 128, 8, 128, True, "bf16", "g2stack_flat"),
    "r4": (4, 256, 256, 64, 8, 128, False, "bf16", "g2stack_flat_kv2"),
    "r5": (16, 1, 2048, 32, 8, 128, False, "fp16", "dense_expert"),
    "r6": (2, 128, 128, 48, 4, 128, False, "fp16", "g2stack_flat"),
}


def main():
    ap = argparse.ArgumentParser(description="Single-shot GQA run for msprof (iter6 roofline)")
    ap.add_argument("--shape", required=True, choices=list(SHAPES))
    args = ap.parse_args()

    B, S, S_kv, N_q, N_kv, D, causal, dtag, expect = SHAPES[args.shape]
    dtype = torch.float16 if dtag == "fp16" else torch.bfloat16

    torch.manual_seed(42)
    q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
    k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
    v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()

    print(
        f"[PROF_RUN] shape={args.shape} B={B} S={S} S_kv={S_kv} N_q={N_q} N_kv={N_kv} "
        f"D={D} causal={causal} dtype={dtag}",
        flush=True,
    )

    out = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
    torch.npu.synchronize()

    chosen = [val[2] for val in G._KERNEL_BUILD_CACHE.values()]
    print(
        f"[PROF_DONE] shape={args.shape} chosen={chosen} expected={expect} "
        f"out.shape={tuple(out.shape)} out.dtype={out.dtype}",
        flush=True,
    )
    # soft domain check (hard assert would kill the msprof export on mismatch;
    # the parse step flags any sample whose chosen != expected)
    if not chosen or chosen[0] != expect:
        print(f"[WARN] dispatch mismatch: chosen={chosen} expected={expect}", flush=True)


if __name__ == "__main__":
    main()
