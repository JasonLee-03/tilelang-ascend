#!/usr/bin/env python3
"""Debug the fullrange nondeterminism: N consecutive identical calls, which
rows differ, and whether the pattern stabilizes."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import torch  # noqa: E402
import torch_npu  # noqa: F401
import gqa as gqa_mod  # noqa: E402

NPU = os.environ.get("STRESS_DEVICE", "npu:0")


def gen(shape, seed):
    g = torch.Generator().manual_seed(seed)
    r = torch.rand(shape, dtype=torch.float64, generator=g)
    return (r * 131008.0 - 65504.0).clamp(-65504, 65504).half()


def main():
    B, S, S_kv, N_q, N_kv, D = 2, 124, 124, 32, 8, 128
    q = gen((B, S, N_q, D), 1).to(NPU)
    k = gen((B, S_kv, N_kv, D), 2).to(NPU)
    v = gen((B, S_kv, N_kv, D), 3).to(NPU)

    outs = []
    for i in range(6):
        o = gqa_mod.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=True)
        torch.npu.synchronize()
        outs.append(o.float().cpu())
    base = outs[0]
    for i, o in enumerate(outs[1:], 1):
        d = (o - base).abs()
        ndiff = int((d > 0).sum())
        print(f"call {i} vs call 0: differing elements={ndiff} maxdiff={float(d.max()):.4e}")
    # which rows differ (call 0 vs call 1)
    d01 = (outs[1] - base).abs()
    bad = (d01.amax(dim=-1) > 0).nonzero()
    print("differing (b,s,hq):", bad.shape[0])
    for idx in bad[:6].tolist():
        b, s, hq = idx
        print(f"  b={b} s={s} hq={hq}: out0={base[b, s, hq, :4].tolist()}")
        print(f"               out1={outs[1][b, s, hq, :4].tolist()}")
    # stability among calls 1..5
    stable = all(torch.equal(outs[i], outs[1]) for i in range(2, 6))
    print("calls 1-5 mutually identical:", stable)


if __name__ == "__main__":
    main()
