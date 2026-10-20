"""Iter5 two-arm same-session A/B probe: BASE (shipped g2stack ns=2) / C2 (flat kv2 variant).

Single-shot run for msprof op collection (mirrors prof_gqa_ab_iter4.py:
EXACTLY ONE main_kernel launch, no warm-up, no golden compare).

--arm c2 monkey-patches gqa._gqa_kernel_prefill_g2stack with a Gate-3-style
REDIRECT: when the builder is invoked with (num_stages==2 & kv_iters==2 &
noncausal & D128 & even G & stacked >= 2*NUM_CORES) it forwards to the
variant factory from perf_tuning/gqa_flat_kv2_variant.py; every other
dispatch is byte-identical to shipped.  The probe asserts (a) the variant
factory was actually invoked (c2 arm) / NEVER invoked (base arm) and (b) the
resolved spec is "g2stack" — a silent compile-fallback cannot be
mis-measured as an arm.  BOTH caches (_KERNEL_BUILD_CACHE + adapter
_COMPILED_CACHE) are cleared so the patched builder is guaranteed to run.
Host-side pipeline (transpose_qkv, transpose_out) is IDENTICAL between
arms; gqa.py / test_gqa.py are NOT modified.

Usage (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:$LD_LIBRARY_PATH
    msprof op --kernel-name=main_kernel --aic-metrics=PipeUtilization \
        --output=<dir> python3 examples/gqa/perf_tuning/prof_gqa_ab_iter5.py \
        --arm c2 --shape r4
"""

import argparse
import os
import sys

import torch
import torch_npu  # noqa: F401  (registers the npu backend)

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_DIR = os.path.dirname(HERE)
sys.path.insert(0, GQA_DIR)
sys.path.insert(0, HERE)

import gqa as G  # noqa: E402

SHAPES = {
    # name: (B, S, S_kv, N_q, N_kv, D, causal, dtype)
    "r4": (4, 256, 256, 64, 8, 128, False, "bf16"),  # bench c4, PRIMARY judgement shape
    "c7": (2, 128, 256, 32, 8, 128, False, "bf16"),  # bench c7, secondary judgement shape
}


def main():
    ap = argparse.ArgumentParser(description="Single-shot GQA two-arm A/B run for msprof (iter5)")
    ap.add_argument("--arm", required=True, choices=["base", "c2"])
    ap.add_argument("--shape", required=True, choices=list(SHAPES))
    args = ap.parse_args()

    B, S, S_kv, N_q, N_kv, D, causal, dtag = SHAPES[args.shape]
    dtype = torch.float16 if dtag == "fp16" else torch.bfloat16

    invoked = []
    if args.arm == "c2":
        import gqa_flat_kv2_variant

        variant_factory = gqa_flat_kv2_variant._gqa_kernel_prefill_g2stack_flat_kv2
        orig = G._gqa_kernel_prefill_g2stack

        def patched(B_, S_padded, S_kv_padded, sm_scale, is_causal, N_q=8, N_kv=2, D=128,
                    dtype="float16", block_M=64, skv_minus_s=0, use_mask=True, block_N=128,
                    num_stages=4, cross_interval=2):
            Gg = N_q // N_kv
            q_blocks = S_padded // block_M
            block_num = B_ * N_kv * (Gg // 2) * q_blocks
            kv_iters = S_kv_padded // block_N
            if (num_stages == 2 and kv_iters == 2 and not is_causal and D == 128
                    and dtype in ("float16", "bfloat16") and Gg % 2 == 0
                    and block_num >= 2 * G.P28_NUM_CORES):
                invoked.append(True)
                return variant_factory(B_, S_padded, S_kv_padded, sm_scale, is_causal,
                                       N_q=N_q, N_kv=N_kv, D=D, dtype=dtype, block_M=block_M,
                                       skv_minus_s=skv_minus_s, use_mask=use_mask, block_N=block_N)
            return orig(B_, S_padded, S_kv_padded, sm_scale, is_causal, N_q=N_q, N_kv=N_kv, D=D,
                        dtype=dtype, block_M=block_M, skv_minus_s=skv_minus_s, use_mask=use_mask,
                        block_N=block_N, num_stages=num_stages, cross_interval=cross_interval)

        G._gqa_kernel_prefill_g2stack = patched

    G._KERNEL_BUILD_CACHE.clear()
    G._COMPILED_CACHE.clear()

    torch.manual_seed(42)
    q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
    k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
    v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()

    print(
        f"[PROF_RUN] arm={args.arm} shape={args.shape} B={B} S={S} S_kv={S_kv} N_q={N_q} "
        f"N_kv={N_kv} D={D} causal={causal} dtype={dtag}",
        flush=True,
    )

    # The single profiled launch: transpose_qkv -> main_kernel -> transpose_out.
    out = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
    torch.npu.synchronize()

    chosen = [val[2] for val in G._KERNEL_BUILD_CACHE.values()]
    if args.arm == "c2":
        assert invoked, f"[AB-PROBE] arm=c2: variant factory was NEVER invoked"
    else:
        assert not invoked, f"[AB-PROBE] arm=base: variant factory invoked — contamination"
    assert chosen == ["g2stack"], (
        f"[AB-PROBE] arm={args.arm}: dispatch resolved to {chosen} — expected ['g2stack']; "
        "this sample is INVALID"
    )
    print(
        f"[PROF_DONE] arm={args.arm} chosen={chosen} variant_invoked={bool(invoked)} "
        f"out.shape={tuple(out.shape)} out.dtype={out.dtype}",
        flush=True,
    )


if __name__ == "__main__":
    main()
