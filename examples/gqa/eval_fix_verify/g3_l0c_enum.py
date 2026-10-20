#!/usr/bin/env python3
"""[G3 / eval cann_final_eval_20260910_071341 case 56] L0C compile-failure
enumeration for the per-head _gqa_kernel_expert.

Eval symptom: AscendMemoryPlanning "Memory allocation failed for: acc_o_l0c
required: 262144, new memory available: 0".  Suspect shape class: D=256
non-aligned variant of base6 (B2 S256 Nq128 Nkv8 D256), or possibly D=512.

This probe compiles the per-head kernel factory for padded shapes covering:
  - D=128 / D=256 / D=512, aligned and non-aligned padded S/S_kv,
  - causal / non-causal, use_mask True/False, fp16/bf16
and reports compile OK / FAIL with the exception text.

Run: unset PYTHONPATH; export ACL_OP_INIT_MODE=1; python g3_l0c_enum.py
"""
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))  # examples/gqa

import torch  # noqa: E402
import gqa as gqa_mod  # noqa: E402

NPU = "npu:0" if torch.npu.is_available() else "cpu"
print(f"device={NPU} torch={torch.__version__}")


def try_compile(tag, B, S, S_kv, N_q, N_kv, D, dtype, is_causal):
    """Compile the per-head kernel exactly as the gqa_expert fallback path
    would (block_M=64, block_N=128, S padded to 64, S_kv padded to 128)."""
    block_M, block_N = 64, 128
    S_padded = ((S + block_M - 1) // block_M) * block_M
    S_kv_padded = ((S_kv + block_N - 1) // block_N) * block_N
    use_mask = bool(is_causal or (S_kv_padded != S_kv))
    try:
        fn = gqa_mod._gqa_kernel_expert(
            B, S_padded, S_kv_padded, 1.0 / (D ** 0.5), is_causal,
            N_q=N_q, N_kv=N_kv, D=D, dtype=dtype, block_M=block_M,
            skv_minus_s=S_kv - S, use_mask=use_mask, block_N=block_N,
        )
        # force actual compilation by calling once with real tensors
        q = torch.randn(B, N_q, S_padded, D, dtype=getattr(torch, dtype)).to(NPU)
        k = torch.randn(B, N_kv, S_kv_padded, D, dtype=getattr(torch, dtype)).to(NPU)
        v = torch.randn(B, N_kv, S_kv_padded, D, dtype=getattr(torch, dtype)).to(NPU)
        m = torch.zeros(B, S_padded, S_kv_padded, dtype=torch.float32).to(NPU)
        out = fn(q, k, v, m)
        torch.npu.synchronize()
        ok = bool(torch.isfinite(out.float()).all().item())
        print(f"[OK ] {tag:52s} out_finite={ok}")
        return True
    except Exception as e:  # noqa: BLE001
        msg = str(e).replace("\n", " | ")[:300]
        print(f"[FAIL] {tag:52s} {type(e).__name__}: {msg}")
        return False


def main():
    results = {}
    # --- D=256 non-aligned variants (base6 class: B2 S256 Nq128 Nkv8 D256) ---
    for (S, S_kv) in [(252, 252), (252, 256), (256, 252), (260, 260),
                      (1020, 1020), (1028, 2044), (2044, 2052)]:
        for causal in (True, False):
            for dt in ("float16", "bfloat16"):
                tag = f"D256 B2 S{S} Skv{S_kv} Nq128 Nkv8 {'causal' if causal else 'noncausal'} {dt}"
                results[tag] = try_compile(tag, 2, S, S_kv, 128, 8, 256, dt, causal)

    # --- D=512 probes (acc_o_l0c [2,64,512] f32 == 262144 exactly) ---
    for (S, S_kv) in [(256, 256), (252, 252), (512, 512)]:
        for causal in (True, False):
            tag = f"D512 B2 S{S} Skv{S_kv} Nq32 Nkv8 {'causal' if causal else 'noncausal'} fp16"
            results[tag] = try_compile(tag, 2, S, S_kv, 32, 8, 512, "float16", causal)

    # --- D=128 large non-aligned sanity ---
    for (S, S_kv) in [(1020, 1020), (1028, 2044)]:
        tag = f"D128 B2 S{S} Skv{S_kv} Nq32 Nkv8 noncausal fp16"
        results[tag] = try_compile(tag, 2, S, S_kv, 32, 8, 128, "float16", False)

    n_ok = sum(results.values())
    print(f"\n=== {n_ok}/{len(results)} compiled OK ===")
    fails = [t for t, ok in results.items() if not ok]
    if fails:
        print("FAIL list:")
        for t in fails:
            print("  -", t)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "g3_l0c_enum.log"), "w") as f:
        for t, ok in results.items():
            f.write(f"{'OK  ' if ok else 'FAIL'} {t}\n")


if __name__ == "__main__":
    main()
