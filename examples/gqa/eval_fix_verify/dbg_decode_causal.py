#!/usr/bin/env python3
"""Debug decode-causal fullrange: which rows/positions deviate from golden."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import torch  # noqa: E402
import gqa as gqa_mod  # noqa: E402
import importlib.util  # noqa: E402

BENCH = "/mnt/workspace/cann/20260910_gqa_V02/tilelang-ascend/.agents/skills/cann-bench"
_spec = importlib.util.spec_from_file_location(
    "gqa_golden", os.path.join(BENCH, "tasks", "level4", "gqa", "golden.py"))
gqa_golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gqa_golden)

NPU = "npu:0"
B, S, S_kv, N_q, N_kv, D = 60, 2, 2048, 32, 8, 128
g = torch.Generator().manual_seed(1234)


def gen(shape):
    r = torch.rand(shape, dtype=torch.float64, generator=g)
    return (r * 131008.0 - 65504.0).clamp(-65504, 65504).half()


q = gen((B, S, N_q, D)).to(NPU)
k = gen((B, S_kv, N_kv, D)).to(NPU)
v = gen((B, S_kv, N_kv, D)).to(NPU)

out = gqa_mod.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=True)
torch.npu.synchronize()
ref = gqa_golden.gqa(q.double().cpu(), k.double().cpu(), v.double().cpu(),
                     scaleValue=-1.0, is_causal=True)

o, r = out.float().cpu(), ref.float()
rel = ((o - r).abs() / (r.abs() + 1e-7))
bad = (rel > 1e-2).nonzero()
print("elements with rel>1e-2:", bad.shape[0], "of", r.numel())
if bad.shape[0]:
    # aggregate per (b, s, hq)
    keys = {}
    for idx in bad.tolist():
        b, s, hq, d = idx
        keys.setdefault((b, s, hq), []).append(d)
    print("affected (b,s,hq) rows:", len(keys))
    for key in list(keys)[:8]:
        b, s, hq = key
        ds = keys[key]
        print(f"  b={b} s={s} hq={hq}: {len(ds)} bad dims, out={o[b, s, hq, ds[0]]:.4f} "
              f"ref={r[b, s, hq, ds[0]]:.4f}")
    # for the first bad row: examine the score distribution + weights
    b, s, hq = list(keys)[0]
    kg = hq // 4
    qv = q[b, s, hq].double().cpu()
    kv = k[b, :, kg].double().cpu()
    scores = (qv @ kv.T) / (D ** 0.5)
    vis = scores > torch.finfo(torch.float32).min
    top = scores.topk(5)
    print(f"row (b={b},s={s},hq={hq}) kg={kg} top scores: {top.values.tolist()}")
    print(f"top cols: {top.indices.tolist()}")
    # causal visibility: j <= s + (S_kv - S)
    for j in top.indices.tolist():
        print(f"  col {j}: causal-visible={j <= s + (S_kv - S)}")
    # what's the golden weight vs kernel contribution at the bad dim
    d0 = keys[(b, s, hq)][0]
    print(f"bad dim d={d0}: out={o[b, s, hq, d0]:.5e} ref={r[b, s, hq, d0]:.5e} "
          f"V[top0,d]={v[b, top.indices[0], kg, d0].float():.5e}")
