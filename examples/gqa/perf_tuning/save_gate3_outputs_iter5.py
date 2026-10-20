"""Gate-3 evidence: save gqa() outputs + chosen dispatch spec per shape (iter5).

Shapes:
  OUT-OF-GATE (dispatch + outputs must stay bit-identical after the gqa.py edit):
    r1_perhead        (1,128,128,8,2,causal,fp16)    -> per_head ladder (16 inst < 20)
    r2_flat_kv1       (2,128,128,32,8,causal,fp16)   -> g2stack_flat (kv==1)
    r3_flat_kv1       (2,128,128,128,8,causal,bf16)  -> g2stack_flat (kv==1)
    r6_flat_kv1       (2,128,128,48,4,noncausal,fp16)-> g2stack_flat (kv==1)
    r5_decode_expert  (16,1,2048,32,8,noncausal,fp16)-> dense_expert
    decode_auto       (4,2,256,16,8,noncausal,fp16)  -> dense_auto (kv=2 < 4)
    causal_kv2_g2s    (4,256,256,64,8,causal,bf16)   -> g2stack ns=2 (C2 gate excludes causal)
    smallstack_kv2_g2s(1,128,256,32,8,noncausal,fp16)-> g2stack ns=2 (stacked 32 < 40)
  IN-GATE (outputs must ALSO stay bit-identical — pure scheduling change;
           chosen spec must move g2stack -> g2stack_flat_kv2 after the edit):
    c4like_kv2        (4,256,256,64,8,noncausal,bf16)
    c7like_kv2        (2,128,256,32,8,noncausal,bf16)

Usage (repo root, unset PYTHONPATH, ACL_OP_INIT_MODE=1):
    python3 examples/gqa/perf_tuning/save_gate3_outputs_iter5.py pre|post
Writes perf_tuning/gate3_outputs_{tag}.pt  (dict name -> cpu tensor) and
perf_tuning/gate3_chosen_{tag}.json       (dict name -> chosen spec list)
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

IN_GATE = {"c4like_kv2", "c7like_kv2"}

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
]


def main():
    tag = sys.argv[1]
    assert tag in ("pre", "post")
    out, chosen = {}, {}
    for name, B, S, Skv, Nq, Nkv, D, causal, dt in SHAPES:
        torch.manual_seed(1234 + B + S + Nq)
        q = (torch.rand(B, S, Nq, D) * 2 - 1).to(dt).npu()
        k = (torch.rand(B, Skv, Nkv, D) * 2 - 1).to(dt).npu()
        v = (torch.rand(B, Skv, Nkv, D) * 2 - 1).to(dt).npu()
        G._KERNEL_BUILD_CACHE.clear()
        G._COMPILED_CACHE.clear()
        o = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
        torch.npu.synchronize()
        out[name] = o.cpu()
        chosen[name] = [val[2] for val in G._KERNEL_BUILD_CACHE.values()]
        print(f"saved {name:22s} chosen={chosen[name]} shape={tuple(o.shape)}", flush=True)
    # Record BOTH dispatch arms (shipped gqa.py) and the independent variant
    # path so the post run can be checked against each: out-of-gate shapes
    # must stay bit-identical to the shipped arm; in-gate shapes must stay
    # bit-identical too (pure scheduling change) while their chosen spec moves
    # g2stack -> g2stack_flat_kv2.
    variant_out, shipped_out = {}, {}
    for name, v in out.items():
        (variant_out if name in IN_GATE else shipped_out)[name] = v
    payload = {
        "tag": tag,
        "gqa_py_md5": hashlib.md5(open(os.path.join(os.path.dirname(HERE), "gqa.py"), "rb").read()).hexdigest(),
        "shipped_md5": "cbce74410fee5f615489d0391b3ea620",
        "chosen": chosen,
        "shipped_arm": shipped_out,
        "in_gate_arm": variant_out,
    }
    torch.save(payload, os.path.join(HERE, f"gate3_outputs_{tag}.pt"))
    with open(os.path.join(HERE, f"gate3_chosen_{tag}.json"), "w") as f:
        json.dump({"gqa_py_md5": payload["gqa_py_md5"], "chosen": chosen}, f, indent=1)
    print("WROTE", f"gate3_outputs_{tag}.pt (md5={payload['gqa_py_md5']}) + gate3_chosen_{tag}.json")


if __name__ == "__main__":
    main()
