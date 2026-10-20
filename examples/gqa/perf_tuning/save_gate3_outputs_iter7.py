"""Gate-3 evidence: save gqa() outputs + main-kernel outputs + plans (iter7 P1).

P1 only changes transpose_out's host-side (bm, bn) plan on STARVED grids;
main kernels and decode are untouched.  Therefore after the gqa.py edit:
  * chosen dispatch spec per shape: must be IDENTICAL pre vs post;
  * MAIN-kernel output (transpose_out's input tensor, captured via a
    transpose_out hook): must be BIT-IDENTICAL pre vs post for EVERY shape;
  * final gqa() output: must be BIT-IDENTICAL pre vs post for EVERY shape
    (retile is a pure re-partition of the same byte set — verified at
    Gate 0 on kernel level and e2e);
  * the out_plan census must show the retiled plans exactly on the starved
    shapes and legacy plans everywhere else.

Shapes (iter5 gate-3 set extended with the P1-relevant padded/D256 arms):
  r1_perhead         (1,128,128,8,2,causal,fp16)     per_head, tout retiled (1,32)
  r2_flat_kv1        (2,128,128,32,8,causal,fp16)    g2stack_flat, tout retiled (2,128)
  r3_flat_kv1        (2,128,128,128,8,causal,bf16)   g2stack_flat, tout retiled (8,128)
  r6_flat_kv1        (2,128,128,48,4,noncausal,fp16) g2stack_flat, tout retiled (4,128)
  r5_decode_expert   (16,1,2048,32,8,noncausal,fp16) dense_expert, NO transpose_out
  decode_auto        (4,2,256,16,8,noncausal,fp16)   dense_auto, NO transpose_out
  causal_kv2_g2s     (4,256,256,64,8,causal,bf16)    g2stack ns=2, tout retiled? (16,128)-identical
  smallstack_kv2_g2s (1,128,256,32,8,noncausal,fp16) g2stack ns=2 (stacked 32<40)
  c4like_kv2         (4,256,256,64,8,noncausal,bf16) flat_kv2, plan IDENTICAL (16,128)
  c7like_kv2         (2,128,256,32,8,noncausal,bf16) flat_kv2, tout retiled (2,128)
  d256_perhead       (2,128,128,32,8,256,causal,fp16)  D=256 ladder, tout retiled (2,128)
  padrow_s100        (2,100,256,32,8,noncausal,fp16) padded s_real path, plan (8,4) identical
  l1_s_tail          (2,124,128,32,8,causal,fp16)    padded s_real=124, plan (8,4) identical

Usage (repo root, unset PYTHONPATH, ACL_OP_INIT_MODE=1):
    python3 examples/gqa/perf_tuning/save_gate3_outputs_iter7.py pre|post
Writes perf_tuning/gate3_iter7_outputs_{tag}.pt   (final + main outputs, cpu)
       perf_tuning/gate3_iter7_chosen_{tag}.json  (chosen specs + plan census + md5)
"""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch
import torch_npu  # noqa: F401

import gqa as G

SHAPES = [
    ("r1_perhead", 1, 128, 128, 8, 2, 128, True, torch.float16),
    ("r2_flat_kv1", 2, 128, 128, 32, 8, 128, True, torch.float16),
    ("r3_flat_kv1", 2, 128, 128, 128, 8, 128, True, torch.bfloat16),
    ("r6_flat_kv1", 2, 128, 128, 48, 4, 128, False, torch.float16),
    ("r5_decode_expert", 16, 1, 2048, 32, 8, 128, False, torch.float16),
    ("decode_auto", 4, 2, 256, 16, 8, 128, False, torch.float16),
    ("causal_kv2_g2s", 4, 256, 256, 64, 8, 128, True, torch.bfloat16),
    ("smallstack_kv2_g2s", 1, 128, 256, 32, 8, 128, False, torch.float16),
    ("c4like_kv2", 4, 256, 256, 64, 8, 128, False, torch.bfloat16),
    ("c7like_kv2", 2, 128, 256, 32, 8, 128, False, torch.bfloat16),
    ("d256_perhead", 2, 128, 128, 32, 8, 256, True, torch.float16),
    ("padrow_s100", 2, 100, 256, 32, 8, 128, False, torch.float16),
    ("l1_s_tail", 2, 124, 128, 32, 8, 128, True, torch.float16),
]


def main():
    tag = sys.argv[1]
    assert tag in ("pre", "post")
    finals, mains, chosen, plans = {}, {}, {}, {}

    # hook transpose_out to capture the MAIN-kernel output (its input tensor)
    orig_tout = G.transpose_out
    cur_main = {}

    def hooked(out_bhsd, s_real=None):
        cur_main["t"] = out_bhsd.detach().clone()
        cur_main["s_real"] = s_real
        return orig_tout(out_bhsd, s_real=s_real)

    orig_plan = G.out_plan
    cur_plans = []

    def hooked_plan(B, Nq, S, D):
        r = orig_plan(B, Nq, S, D)
        cur_plans.append(((B, Nq, S, D), r[1:] if r else None))
        return r

    G.transpose_out = hooked
    G.out_plan = hooked_plan
    try:
        for name, B, S, Skv, Nq, Nkv, D, causal, dt in SHAPES:
            torch.manual_seed(1234 + B + S + Nq)
            q = (torch.rand(B, S, Nq, D) * 2 - 1).to(dt).npu()
            k = (torch.rand(B, Skv, Nkv, D) * 2 - 1).to(dt).npu()
            v = (torch.rand(B, Skv, Nkv, D) * 2 - 1).to(dt).npu()
            G._KERNEL_BUILD_CACHE.clear()
            G._COMPILED_CACHE.clear()
            cur_main.clear()
            cur_plans.clear()
            o = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
            torch.npu.synchronize()
            finals[name] = o.cpu()
            if "t" in cur_main:
                mains[name] = cur_main["t"].cpu()
            chosen[name] = [val[2] for val in G._KERNEL_BUILD_CACHE.values()]
            plans[name] = [(list(k_), (list(v_) if v_ else None)) for k_, v_ in cur_plans]
            print(f"saved {name:22s} chosen={chosen[name]} plans={plans[name]} "
                  f"main_out={'yes' if name in mains else 'no(decode)'}", flush=True)
    finally:
        G.transpose_out = orig_tout
        G.out_plan = orig_plan

    payload = {
        "tag": tag,
        "finals": finals,
        "mains": mains,
    }
    torch.save(payload, os.path.join(HERE, f"gate3_iter7_outputs_{tag}.pt"))
    meta = {
        "tag": tag,
        "gqa_py_md5": hashlib.md5(
            open(os.path.join(os.path.dirname(HERE), "gqa.py"), "rb").read()).hexdigest(),
        "chosen": chosen,
        "plans": plans,
    }
    with open(os.path.join(HERE, f"gate3_iter7_chosen_{tag}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[SAVE_DONE] tag={tag} shapes={len(finals)} mains={len(mains)} md5={meta['gqa_py_md5']}")


if __name__ == "__main__":
    main()
