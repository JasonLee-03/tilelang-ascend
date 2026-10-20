"""Iter5 Gate-3 fallback re-check probe — R1 (per-head) / R3 (flat kv1).

Native dispatch through the MODIFIED gqa.py (no monkeypatch): these two shapes
are OUT of the C2 gate, so their chosen spec must be unchanged from the iter3
anchors (per_head_bm64 / g2stack_flat) and their Task Duration must stay inside
the cross-session noise band.  Single-shot run for msprof op collection
(EXACTLY ONE main_kernel launch, no warm-up, no golden compare).

Usage (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:$LD_LIBRARY_PATH
    msprof op --kernel-name=main_kernel --aic-metrics=PipeUtilization \
        --output=<dir> python3 examples/gqa/perf_tuning/prof_gqa_regress_iter5.py --shape r1
"""

import argparse
import os
import sys

import torch
import torch_npu  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_DIR = os.path.dirname(HERE)
sys.path.insert(0, GQA_DIR)

import gqa as G  # noqa: E402

# name: (B, S, S_kv, N_q, N_kv, D, causal, dtype, expected_chosen, iter3_anchor_us)
SHAPES = {
    "r1": (1, 128, 128, 8, 2, 128, True, "fp16", "per_head_bm64", 10.83),
    "r3": (2, 128, 128, 128, 8, 128, True, "bf16", "g2stack_flat", 61.09),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True, choices=list(SHAPES))
    args = ap.parse_args()
    B, S, S_kv, N_q, N_kv, D, causal, dtag, expect, anchor = SHAPES[args.shape]
    dtype = torch.float16 if dtag == "fp16" else torch.bfloat16

    G._KERNEL_BUILD_CACHE.clear()
    G._COMPILED_CACHE.clear()
    torch.manual_seed(42)
    q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
    k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
    v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()

    print(f"[REG_RUN] shape={args.shape} B={B} S={S} S_kv={S_kv} N_q={N_q} N_kv={N_kv} causal={causal} {dtag}", flush=True)
    out = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
    torch.npu.synchronize()

    chosen = [val[2] for val in G._KERNEL_BUILD_CACHE.values()]
    assert chosen == [expect], f"[REG-PROBE] shape={args.shape}: dispatch = {chosen}, expected [{expect}] — OUT-OF-GATE ARM MOVED"
    print(f"[REG_DONE] shape={args.shape} chosen={chosen} iter3_anchor={anchor}us out.shape={tuple(out.shape)}", flush=True)


if __name__ == "__main__":
    main()
