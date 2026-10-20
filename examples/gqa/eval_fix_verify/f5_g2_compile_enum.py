"""F5: enumerate _gqa_kernel_prefill_g2 compile combinations (compile-only).

Eval failure class C (case 68): _gqa_kernel_prefill_g2 hit a bisheng C++
compile failure on some small shape in the CANN 9.1.0 / Ascend910_9362 eval
environment (no compiler stderr in the eval log).  This script enumerates
the g2 parameter space on the LOCAL CANN 9.2 toolchain to try to reproduce.

Parameters are derived exactly as the gqa_expert dispatch would:
    S_padded    = ceil(S / 64) * 64
    S_kv_padded = ceil(S_kv / 128) * 128
    use_mask    = is_causal or (S_kv_padded != S_kv)
    skv_minus_s = S_kv - S
    block_M=64, block_N=128

Usage:
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python examples/gqa/eval_fix_verify/f5_g2_compile_enum.py [--phase N]
"""

import os
import sys
import time
import traceback

import torch  # noqa: F401
import torch_npu  # noqa: F401

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "examples", "gqa"))

from gqa import _gqa_kernel_prefill_g2, _gqa_kernel_expert  # noqa: E402

PHASE = 1
if "--phase" in sys.argv:
    PHASE = int(sys.argv[sys.argv.index("--phase") + 1])

# phase 6 enumerates the per-head kernel instead (the F5 fallback target)
KERNEL_FN = _gqa_kernel_prefill_g2
if PHASE == 6:
    KERNEL_FN = _gqa_kernel_expert


def compile_one(B, S, S_kv, N_q, N_kv, D, dtype, is_causal):
    S_padded = ((S + 63) // 64) * 64
    S_kv_padded = ((S_kv + 127) // 128) * 128
    use_mask = bool(is_causal or (S_kv_padded != S_kv))
    sm_scale = 1.0 / (D ** 0.5)
    t0 = time.time()
    kernel = KERNEL_FN(
        B, S_padded, S_kv_padded, sm_scale, is_causal,
        N_q=N_q, N_kv=N_kv, D=D, dtype=dtype, block_M=64,
        skv_minus_s=S_kv - S, use_mask=use_mask, block_N=128,
    )
    dt = time.time() - t0
    assert kernel is not None
    return dt


def run_grid(cases, label):
    n_fail = 0
    t_start = time.time()
    for i, (B, S, S_kv, N_q, N_kv, D, dtype, causal) in enumerate(cases):
        desc = (f"B{B} S{S} Skv{S_kv} Nq{N_q} Nkv{N_kv} D{D} {dtype} "
                f"causal={causal}")
        try:
            dt = compile_one(B, S, S_kv, N_q, N_kv, D, dtype, causal)
            print(f"[{label} {i+1}/{len(cases)}] OK   {desc} ({dt:.1f}s)",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"[{label} {i+1}/{len(cases)}] FAIL {desc}: "
                  f"{type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    print(f"[{label}] done: {len(cases) - n_fail}/{len(cases)} compiled, "
          f"{n_fail} FAILED, {time.time() - t_start:.0f}s total", flush=True)
    return n_fail


def main():
    S_LIST = [8, 16, 32, 48, 64, 80, 96, 128]
    SKV_LIST = [128, 192, 256, 320, 384, 512]

    if PHASE == 1:
        # High suspicion (orchestrator hint): N_kv=1 (G=32) and odd kv_iters
        cases = []
        for S in S_LIST:
            for S_kv in (320, 384):  # -> S_kv_padded 384 -> kv_iters=3 (odd)
                for causal in (True, False):
                    cases.append((2, S, S_kv, 32, 1, 128, "float16", causal))
        for S in S_LIST:
            for S_kv in (320, 384):
                for causal in (True, False):
                    cases.append((2, S, S_kv, 32, 1, 128, "bfloat16", causal))
        run_grid(cases, "phase1")
    elif PHASE == 2:
        # Broad: all N_kv, G odd (tail_possible), odd kv, causal/non-causal
        cases = []
        for (N_q, N_kv) in ((32, 1), (32, 2), (32, 4), (32, 8), (24, 8), (48, 4)):
            for S in S_LIST:
                for S_kv in (128, 320, 256):
                    for causal in (True, False):
                        cases.append((2, S, S_kv, N_q, N_kv, 128, "float16",
                                      causal))
        run_grid(cases, "phase2")
    elif PHASE == 3:
        # causal S > S_kv (skip predicate fires) + tiny instance counts
        cases = []
        for (B, S, S_kv) in ((2, 192, 128), (2, 256, 128), (2, 256, 192),
                             (1, 96, 64), (1, 64, 64), (2, 320, 256),
                             (2, 512, 384), (2, 640, 512)):
            for (N_q, N_kv) in ((32, 8), (24, 8), (32, 1)):
                for dtype in ("float16", "bfloat16"):
                    cases.append((B, S, S_kv, N_q, N_kv, 128, dtype, True))
        run_grid(cases, "phase3")
    elif PHASE == 4:
        # bf16 broad + remaining S/S_kv grid (fp16)
        cases = []
        for (N_q, N_kv) in ((32, 8), (24, 8)):
            for S in S_LIST:
                for S_kv in SKV_LIST:
                    cases.append((2, S, S_kv, N_q, N_kv, 128, "float16",
                                  True))
                    cases.append((2, S, S_kv, N_q, N_kv, 128, "bfloat16",
                                  False))
        run_grid(cases, "phase4")
    elif PHASE == 5:
        # +4-style variants: odd q_blocks (S=132 -> S_padded=192 -> q=3)
        # and odd kv_iters (Skv=260 -> 384 -> kv=3, 1028 -> 1152 -> kv=9)
        # — combinations never exercised by the aligned bench shapes.
        cases = []
        # (S+4, Skv+4) on S==Skv bases: q_blocks in {3, 5, 9, 17}
        for (S, Skv) in ((132, 132), (260, 260), (516, 516), (1028, 1028)):
            for causal in (True, False):
                for dtype in ("float16", "bfloat16"):
                    cases.append((2, S, Skv, 32, 8, 128, dtype, causal))
                    cases.append((2, S, Skv, 32, 1, 128, dtype, causal))
        # cross-attention +4 (base-7-like): (132, 260), (132, 516)
        for (S, Skv) in ((132, 260), (132, 516)):
            for dtype in ("float16", "bfloat16"):
                cases.append((2, S, Skv, 32, 8, 128, dtype, False))
        # tiny-S prefill (decode bases with S+4): odd kv via Skv+4
        for (S, Skv) in ((5, 2052), (5, 1028), (6, 2052), (8, 516)):
            for (N_q, N_kv) in ((32, 8), (96, 8), (128, 8)):
                for causal in (True, False):
                    cases.append((16 if S < 8 else 4, S, Skv, N_q, N_kv, 128,
                                  "float16", causal))
        # G odd (tail_possible) with odd q_blocks
        for causal in (True, False):
            cases.append((2, 132, 132, 24, 8, 128, "float16", causal))
            cases.append((2, 132, 132, 24, 8, 128, "bfloat16", causal))
        run_grid(cases, "phase5")
    elif PHASE == 6:
        # [F5 fallback target] same representative grid through the per-head
        # kernel (_gqa_kernel_expert) — verifies the dispatch fallback compiles
        # on every combination the g2 path currently serves.
        cases = []
        for (S, Skv) in ((132, 132), (260, 260), (516, 516), (1028, 1028),
                         (132, 260), (192, 128), (256, 128), (124, 252),
                         (1020, 1020), (128, 124)):
            for causal in (True, False):
                for dtype in ("float16", "bfloat16"):
                    cases.append((2, S, Skv, 32, 8, 128, dtype, causal))
        for (S, Skv) in ((5, 2052), (8, 516), (48, 320)):
            for causal in (True, False):
                cases.append((4, S, Skv, 32, 8, 128, "float16", causal))
        for (N_q, N_kv) in ((24, 8), (32, 1), (48, 4)):
            for causal in (True, False):
                cases.append((2, 132, 132, N_q, N_kv, 128, "float16", causal))
                cases.append((2, 260, 260, N_q, N_kv, 128, "bfloat16", causal))
        run_grid(cases, "phase6")
    else:
        print(f"unknown phase {PHASE}")
        sys.exit(2)


if __name__ == "__main__":
    main()
