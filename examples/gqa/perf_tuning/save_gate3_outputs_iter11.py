"""Gate-3 evidence: save gqa() outputs + transpose_qkv census + dispatch
census, PRE (shipped backup gqa_impl_iter11_before.py loaded under an alias)
vs POST (the integrated gqa.py) — iter11 P4 wave-2 (transpose_qkv elimination
extended to the g2stack + prefill_expert dual-aligned domains).

Wave-2 changes, on the newly-covered domain only (chosen in {g2stack,
prefill_expert} AND S % 64 == 0 AND S_kv % 128 == 0):
  * the two kernels' Q/K/V declarations BHSD -> BSND and the Q/K/V load boxes
    to the strided form (realSrcN = N*D);
  * the shipped adapter machinery (wave-1, unchanged) resolves the kernel
    first and skips transpose_qkv when `_run.in_bsnd` is True — the extended
    _P4_BSND_KERNELS list now grants the capability to these two kernels.
Everything else (flat kv1/kv2 wave-1 domain, per-head AUTO ladder, decode,
non-aligned S / non-aligned S_kv, transpose_qkv itself, transpose_out/P1,
out_bsnd machinery) is untouched.  Therefore:
  * chosen dispatch spec per shape: IDENTICAL pre vs post;
  * FINAL gqa() output: BIT-IDENTICAL (raw-bit, NaN-safe) pre vs post for
    EVERY shape;
  * tqkv census: covered shapes RAN in pre, SKIPPED in post; every other
    shape behaves identically in both (wave-1 flat shapes: SKIPPED in BOTH —
    that is the "wave-1 undisturbed" evidence).

Usage (repo root, unset PYTHONPATH, ACL_OP_INIT_MODE=1):
    python3 examples/gqa/perf_tuning/save_gate3_outputs_iter11.py pre|post
Writes perf_tuning/gate3_iter11_outputs_{tag}.pt   (final outputs, cpu)
       perf_tuning/gate3_iter11_chosen_{tag}.json  (chosen + tqkv census + md5)
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
    src = os.path.join(HERE, "gqa_impl_iter11_before.py")
    spec = importlib.util.spec_from_file_location("gqa_pre_iter11", src)
    G = importlib.util.module_from_spec(spec)
    sys.modules["gqa_pre_iter11"] = G
    spec.loader.exec_module(G)
else:
    src = os.path.join(GQA_DIR, "gqa.py")
    import gqa as G

MOD_MD5 = hashlib.md5(open(src, "rb").read()).hexdigest()
print(f"[save-{TAG}] module {src} md5 {MOD_MD5}")

# (name, B, S, S_kv, N_q, N_kv, D, causal, dtype, expected_chosen, category)
#   category: "p4w2" = newly covered by THIS integration (tqkv pre RAN -> post SKIPPED)
#             "out"  = out-of-domain (untouched paths; census identical pre/post)
SHAPES = [
    # ---- wave-2 newly covered: prefill_expert ----
    ("c19_prexpert_mha", 2, 128, 128, 32, 32, 128, True, torch.bfloat16, "prefill_expert", "p4w2"),
    ("g3_prexpert_kv2_causal", 2, 256, 256, 24, 8, 128, True, torch.float16, "prefill_expert", "p4w2"),
    # ---- wave-2 newly covered: g2stack ----
    ("c8_g2stack_kv8_causal", 4, 1024, 1024, 32, 8, 128, True, torch.float16, "g2stack", "p4w2"),
    ("c9_g2stack_kv4", 4, 512, 512, 64, 8, 128, False, torch.bfloat16, "g2stack", "p4w2"),
    ("c10_g2stack_nq128", 4, 512, 512, 128, 8, 128, True, torch.bfloat16, "g2stack", "p4w2"),
    ("kv2_causal_g2stack", 2, 128, 256, 32, 8, 128, True, torch.float16, "g2stack", "p4w2"),
    # ---- out-of-domain: WAVE-1 flat (tqkv SKIPPED in BOTH — undisturbed) ----
    ("r2_flat_wave1", 2, 128, 128, 32, 8, 128, True, torch.float16, "g2stack_flat", "out"),
    ("r4_flatkv2_wave1", 4, 256, 256, 64, 8, 128, False, torch.bfloat16, "g2stack_flat_kv2", "out"),
    # ---- out-of-domain: per-head AUTO ladder (EXCLUDED domain) ----
    ("r1_perhead", 1, 128, 128, 8, 2, 128, True, torch.float16, "per_head_bm64", "out"),
    ("d256_perhead", 2, 128, 128, 32, 8, 256, True, torch.float16, "per_head_bm64", "out"),
    # ---- out-of-domain: GATE ISOLATION (dispatch hit, non-aligned dims) ----
    ("padrow_s100_flatkv1", 2, 100, 128, 32, 8, 128, True, torch.float16, "g2stack_flat", "out"),
    ("padcol_skv200_flatkv2", 2, 128, 200, 32, 8, 128, False, torch.float16, "g2stack_flat_kv2", "out"),
    ("s130_g2stack_unaligned", 2, 130, 130, 32, 8, 128, True, torch.float16, "g2stack", "out"),
    ("padcol_skv400_g2stack", 2, 256, 400, 32, 8, 128, True, torch.float16, "g2stack", "out"),
    ("s100e_prexpert_unaligned", 2, 100, 128, 32, 32, 128, True, torch.bfloat16, "prefill_expert", "out"),
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
            }
            print(f"[save-{TAG}] {name}: chosen={chosen} tqkv={len(tqkv_calls)} "
                  f"out={tuple(out.shape)}", flush=True)
    finally:
        G.transpose_qkv = orig_tqkv

    torch.save(outs, os.path.join(HERE, f"gate3_iter11_outputs_{TAG}.pt"))
    with open(os.path.join(HERE, f"gate3_iter11_chosen_{TAG}.json"), "w") as f:
        json.dump(census, f, indent=1)
    print(f"[save-{TAG}] WROTE outputs + chosen census ({len(outs)} shapes)")


if __name__ == "__main__":
    main()
