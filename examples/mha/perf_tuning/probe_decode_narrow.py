"""Scratch probe (NOT a deliverable): #6 decode narrow-block equivalence check.

Compares the CURRENT mha.py (block_M=16 for S<=16) against the post-#3 backup
history_version/mha_impl_s3_perf4.py (block_M=64) on decode/MTP shapes.
Expectation: row 0..S-1 outputs are computed by an identical per-row op
sequence (rows are independent; same column reduce width; same roundings) ->
BITWISE equal outputs. Any mismatch = narrowing changed semantics -> reject #6.
"""

import importlib.util
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_OP_DIR = os.path.dirname(_HERE)
sys.path.insert(0, _OP_DIR)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


import mha as cur  # noqa: E402  (current kernel under test)

old = _load(os.path.join(_OP_DIR, "history_version", "mha_impl_s3_perf4.py"), "mha_perf4")

# (B,S,S_kv,N,D, causal) — decode S=1, MTP S=2, all D, tail+aligned S_kv,
# causal+non-causal; shapes mirror cann-bench case10-18 semantics.
SHAPES = [
    (1, 1, 2048, 32, 128, False),   # case10 exact
    (16, 1, 1024, 32, 128, False),  # case11-like
    (32, 1, 512, 32, 128, False),   # case12-like
    (8, 1, 128, 32, 128, False),    # case14-like (B trimmed for CPU-golden cost)
    (4, 1, 512, 16, 256, False),    # case15-like D=256
    (4, 2, 512, 32, 128, True),     # case16-like MTP causal
    (2, 2, 256, 8, 64, True),       # MTP causal D=64
    (1, 1, 64, 1, 64, True),        # minimal causal single-block
    (1, 1, 65, 1, 64, False),       # S_kv tail (bound mask variant)
    (1, 2, 192, 2, 128, True),      # S=2 causal tail S_kv=192 (offset=190)
    (1, 16, 128, 2, 64, False),     # S=16 boundary of the branch
    (1, 16, 128, 2, 64, True),      # S=16 causal
]

fail = 0
for B, S, S_kv, N, D, causal in SHAPES:
    g = torch.Generator().manual_seed(777)
    dt = torch.float16
    q = ((torch.randn(B, S, N, D, generator=g) * torch.rand(B, S, N, D, generator=g) * 2 - 0).to(dt)).to("npu")
    k = ((torch.randn(B, S_kv, N, D, generator=g) * torch.rand(B, S_kv, N, D, generator=g) * 2 - 0).to(dt)).to("npu")
    v = ((torch.randn(B, S_kv, N, D, generator=g) * torch.rand(B, S_kv, N, D, generator=g) * 2 - 0).to(dt)).to("npu")
    o_new = cur.mha(q, k, v, scaleValue=-1.0, is_causal=causal)
    o_old = old.mha(q, k, v, scaleValue=-1.0, is_causal=causal)
    torch.npu.synchronize()
    bitexact = torch.equal(o_new.cpu(), o_old.cpu())
    maxd = (o_new.float().cpu() - o_old.float().cpu()).abs().max().item()
    tag = "BITEXACT" if bitexact else f"DIFF max={maxd:.3e}"
    print(f"[{'OK' if bitexact else 'FAIL'}] B{B} S{S} Skv{S_kv} N{N} D{D} causal={causal}: {tag}")
    fail += int(not bitexact)

# S=17 must NOT take the narrow branch (sanity: still compiles/equals old path)
g = torch.Generator().manual_seed(99)
q = torch.randn(1, 17, 2, 64, generator=g).half().to("npu")
k = torch.randn(1, 17, 2, 64, generator=g).half().to("npu")
v = torch.randn(1, 17, 2, 64, generator=g).half().to("npu")
o_new = cur.mha(q, k, v, is_causal=True)
o_old = old.mha(q, k, v, is_causal=True)
torch.npu.synchronize()
be = torch.equal(o_new.cpu(), o_old.cpu())
print(f"[{'OK' if be else 'FAIL'}] S=17 (block_M=64 path unchanged): {'BITEXACT' if be else 'DIFF'}")
fail += int(not be)

print("PROBE_RESULT:", "PASS" if fail == 0 else f"FAIL({fail})")
sys.exit(1 if fail else 0)
