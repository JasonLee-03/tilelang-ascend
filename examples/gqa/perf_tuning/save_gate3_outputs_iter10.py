"""Gate-3 evidence: save gqa() outputs + transpose_qkv census + dispatch
census, PRE (shipped backup gqa_impl_iter10_before.py loaded under an alias)
vs POST (the integrated gqa.py) — iter10 P4 (transpose_qkv elimination on the
flat kv1/kv2 dual-aligned domain).

P4 changes, on the newly-covered domain only (chosen in {g2stack_flat,
g2stack_flat_kv2} AND S % 64 == 0 AND S_kv % 128 == 0):
  * the flat kernels' Q/K/V declarations BHSD -> BSND and the emit_qload /
    emit_g1 K/V load boxes to the strided form (realSrcN = N*D);
  * the adapter resolves the kernel FIRST and skips transpose_qkv entirely
    when `_run.in_bsnd` is True.
Everything else (per-head AUTO ladder, g2stack, prefill_expert, decode,
non-aligned S / non-aligned S_kv, transpose_qkv itself, transpose_out/P1,
wave-1/wave-2 out_bsnd machinery) is untouched.  Therefore:
  * chosen dispatch spec per shape: IDENTICAL pre vs post;
  * FINAL gqa() output: BIT-IDENTICAL (raw-bit, NaN-safe) pre vs post for
    EVERY shape — transpose_qkv on the aligned domain is a pure
    rearrangement; the strided loads read the identical element set;
  * tqkv census: covered shapes RAN in pre, SKIPPED in post; every other
    shape behaves identically in both.

Usage (repo root, unset PYTHONPATH, ACL_OP_INIT_MODE=1):
    python3 examples/gqa/perf_tuning/save_gate3_outputs_iter10.py pre|post
Writes perf_tuning/gate3_iter10_outputs_{tag}.pt   (final outputs, cpu)
       perf_tuning/gate3_iter10_chosen_{tag}.json  (chosen + tqkv census + md5)
"""

import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_DIR = os.path.dirname(HERE)
sys.path.insert(0, GQA_DIR)

import torch
import torch_npu  # noqa: F401

TAG = sys.argv[1]
assert TAG in ("pre", "post"), "tag must be pre|post"

if TAG == "pre":
    src = os.path.join(HERE, "gqa_impl_iter10_before.py")
    spec = importlib.util.spec_from_file_location("gqa_pre_iter10", src)
    G = importlib.util.module_from_spec(spec)
    sys.modules["gqa_pre_iter10"] = G
    spec.loader.exec_module(G)
else:
    src = os.path.join(GQA_DIR, "gqa.py")
    import gqa as G

MOD_MD5 = hashlib.md5(open(src, "rb").read()).hexdigest()
print(f"[save-{TAG}] module {src} md5 {MOD_MD5}")

# (name, B, S, S_kv, N_q, N_kv, D, causal, dtype, expected_chosen, category)
#   category: "p4"  = newly covered by THIS integration (tqkv pre RAN -> post SKIPPED)
#             "out" = out-of-domain (untouched paths)
SHAPES = [
    # ---- P4 newly covered ----
    ("r2_flat_kv1_causal", 2, 128, 128, 32, 8, 128, True, torch.float16, "g2stack_flat", "p4"),
    ("r6_flat_kv1_mqa4", 2, 128, 128, 48, 4, 128, False, torch.float16, "g2stack_flat", "p4"),
    ("r4_flat_kv2_bf16", 4, 256, 256, 64, 8, 128, False, torch.bfloat16, "g2stack_flat_kv2", "p4"),
    ("c7_flat_kv2_cross", 2, 128, 256, 32, 8, 128, False, torch.bfloat16, "g2stack_flat_kv2", "p4"),
    ("mqa_nkv1_flat_kv1", 2, 128, 128, 32, 1, 128, False, torch.float16, "g2stack_flat", "p4"),
    # ---- out-of-domain: per-head AUTO ladder (EXCLUDED domain) ----
    ("r1_perhead", 1, 128, 128, 8, 2, 128, True, torch.float16, "per_head_bm64", "out"),
    ("d256_perhead", 2, 128, 128, 32, 8, 256, True, torch.float16, "per_head_bm64", "out"),
    # ---- out-of-domain: expert families NOT in the first wave ----
    ("c19_prexpert_mha", 2, 128, 128, 32, 32, 128, True, torch.bfloat16, "prefill_expert", "out"),
    ("c9_g2stack_kv4", 4, 512, 512, 64, 8, 128, False, torch.bfloat16, "g2stack", "out"),
    # ---- out-of-domain: GATE ISOLATION (flat hit but non-aligned S / S_kv) ----
    ("padrow_s100_flatkv1", 2, 100, 128, 32, 8, 128, True, torch.float16, "g2stack_flat", "out"),
    ("padcol_skv200_flatkv2", 2, 128, 200, 32, 8, 128, False, torch.float16, "g2stack_flat_kv2", "out"),
    # ---- decode (BSND direct already, no tqkv) ----
    ("r5_decode_expert", 16, 1, 2048, 32, 8, 128, False, torch.float16, "dense_expert", "out"),
    ("decode_auto", 4, 2, 256, 16, 8, 128, False, torch.float16, "dense_auto", "out"),
]


def main():
    outs = {}
    census = {"module_md5": MOD_MD5, "shapes": {}}

    orig_tqkv = G.transpose_qkv
    tqkv_calls = []

    def tqkv_hook(query, key, value, S_out=None, S_kv_out=None):
        tqkv_calls.append((tuple(query.shape), S_out, S_kv_out))
        return orig_tqkv(query, key, value, S_out=S_out, S_kv_out=S_kv_out)

    G.transpose_qkv = tqkv_hook
    try:
        for (name, B, S, S_kv, N_q, N_kv, D, causal, dtype, expect, cat) in SHAPES:
            G._KERNEL_BUILD_CACHE.clear()
            if hasattr(G, "_COMPILED_CACHE"):
                G._COMPILED_CACHE.clear()
            tqkv_calls.clear()
            torch.manual_seed(1234 + B + S + N_q)
            q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
            k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
            v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
            out = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
            torch.npu.synchronize()
            outs[name] = out.cpu().view(torch.int16)
            chosen = [val[2] for val in G._KERNEL_BUILD_CACHE.values()]
            assert chosen and chosen[0].startswith(expect), \
                f"{name}: chosen {chosen} != ~{expect}"
            census["shapes"][name] = {
                "chosen": chosen, "cat": cat,
                "tqkv_calls": [list(map(str, c)) for c in tqkv_calls],
                "in_bsnd": None,
            }
            print(f"[save-{TAG}] {name}: chosen={chosen} tqkv={len(tqkv_calls)} "
                  f"out={tuple(out.shape)}", flush=True)
    finally:
        G.transpose_qkv = orig_tqkv

    torch.save(outs, os.path.join(HERE, f"gate3_iter10_outputs_{TAG}.pt"))
    with open(os.path.join(HERE, f"gate3_iter10_chosen_{TAG}.json"), "w") as f:
        json.dump(census, f, indent=1)
    print(f"[save-{TAG}] WROTE outputs + chosen census ({len(outs)} shapes)")


if __name__ == "__main__":
    main()
