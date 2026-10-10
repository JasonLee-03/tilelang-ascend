"""Scratch probe (NOT a deliverable): #5 Fixed-Core equivalence check.

Compares CURRENT mha.py (FC: core_num=min(block_num,20), contiguous task
chunks) against history_version/mha_impl_s3_perf5.py (post-#6, per-block
grid). Every task computes the same (bz,by,bx) tile with identical op
sequences -> outputs must be BITWISE equal. Mismatch = FC changed semantics
(task mapping / state reset / workspace reuse) -> reject #5.

Edge coverage: block_num < 20 (core_num degenerates to block_num),
block_num not divisible by core_num (empty-range cores), causal trim inside
nested runtime loops, prefill + decode + MTP + D variants.
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


import mha as cur  # noqa: E402

old = _load(os.path.join(_OP_DIR, "history_version", "mha_impl_s3_perf5.py"), "mha_perf5")

# (B,S,S_kv,N,D,causal) with block_num = B*N*ceil(S/bM) annotated
SHAPES = [
    (1, 64, 64, 1, 64, False),     # block_num=1  (core_num=1)
    (1, 64, 64, 1, 64, True),      # block_num=1 causal
    (2, 128, 128, 8, 64, False),   # block_num=32 (>20, rem 12)
    (2, 128, 128, 8, 64, True),    # block_num=32 causal trim x FC
    (3, 128, 128, 5, 64, True),    # block_num=30 (odd grid, rem 10)
    (1, 127, 127, 3, 128, True),   # block_num=6, S/S_kv tails + causal offset=0
    (2, 65, 192, 2, 64, False),    # block_num=8, S tail rows + S_kv tail cols
    (4, 1024, 1024, 4, 128, True), # block_num=256 prefill-like (case7 trimmed N)
    (1, 1, 2048, 32, 128, False),  # block_num=32 decode (case10-like)
    (8, 1, 128, 32, 128, False),   # block_num=256 decode narrow (case14-like)
    (4, 2, 512, 32, 128, True),    # block_num=128 MTP causal
    (2, 1024, 1024, 2, 256, True), # block_num=64 D=256 bM=32 prefill causal
]

fail = 0
for B, S, S_kv, N, D, causal in SHAPES:
    g = torch.Generator().manual_seed(4242)
    dt = torch.float16
    q = (torch.randn(B, S, N, D, generator=g) * (torch.rand(B, S, N, D, generator=g) * 2 - 1)).to(dt).to("npu")
    k = (torch.randn(B, S_kv, N, D, generator=g) * (torch.rand(B, S_kv, N, D, generator=g) * 2 - 1)).to(dt).to("npu")
    v = (torch.randn(B, S_kv, N, D, generator=g) * (torch.rand(B, S_kv, N, D, generator=g) * 2 - 1)).to(dt).to("npu")
    o_new = cur.mha(q, k, v, scaleValue=-1.0, is_causal=causal)
    o_old = old.mha(q, k, v, scaleValue=-1.0, is_causal=causal)
    torch.npu.synchronize()
    be = torch.equal(o_new.cpu(), o_old.cpu())
    maxd = (o_new.float().cpu() - o_old.float().cpu()).abs().max().item()
    print(f"[{'OK' if be else 'FAIL'}] B{B} S{S} Skv{S_kv} N{N} D{D} causal={causal}: {'BITEXACT' if be else f'DIFF max={maxd:.3e}'}")
    fail += int(not be)

print("PROBE_RESULT:", "PASS" if fail == 0 else f"FAIL({fail})")
sys.exit(1 if fail else 0)
