"""Scratch probe (NOT a deliverable): #8d decode-branch block_N=128 checks.

1. PREFILL shapes (S>16): code path untouched -> outputs must be BITWISE equal
   to perf7 (post-#4, bN=64 everywhere).
2. DECODE shapes (S<=16): bN 64->128 changes the online-softmax intermediate
   rounding sequence -> NOT bitwise; compared against perf7 (closeness) and
   against the CPU golden with the L0 blocking caliber (mixed tolerance +
   MARE_abs < 10*Threshold, test_mha.py functions reused).
"""

import importlib.util
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_OP_DIR = os.path.dirname(_HERE)
sys.path.insert(0, _OP_DIR)
sys.path.insert(0, _HERE)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


import mha as cur  # noqa: E402

old = _load(os.path.join(_OP_DIR, "history_version", "mha_impl_s3_perf7.py"), "mha_perf7")
tm = _load(os.path.join(_OP_DIR, "test_mha.py"), "test_mha_mod")

PREFILL = [  # (B,S,S_kv,N,D,causal) — must be BITEXACT vs perf7
    (2, 128, 128, 8, 64, True),
    (1, 64, 65, 1, 64, False),
    (2, 1024, 1024, 2, 128, True),
    (1, 65, 96, 2, 256, True),
]
DECODE = [  # S<=16 — tolerance-checked vs golden + closeness vs perf7
    (1, 1, 2048, 32, 128, False),   # case10 exact
    (8, 1, 128, 32, 128, False),    # case14-like: S_kv=128 -> single_kv_block now
    (4, 1, 64, 8, 64, False),       # S_kv=64 single-block before & after
    (4, 2, 512, 32, 128, True),     # case16-like MTP causal
    (2, 2, 256, 4, 256, True),      # MTP D=256 causal
    (1, 1, 192, 2, 64, False),      # S_kv=192: %128!=0 -> new tail-clamp variant
    (1, 1, 129, 1, 128, False),     # S_kv=129 tail-1 with bN=128
    (2, 16, 512, 2, 128, True),     # S=16 causal
]

fail = 0
print("== PREFILL (must be BITEXACT vs perf7) ==")
for B, S, S_kv, N, D, causal in PREFILL:
    g = torch.Generator().manual_seed(31337)
    dt = torch.float16
    q = (torch.randn(B, S, N, D, generator=g) * (torch.rand(B, S, N, D, generator=g) * 2 - 1)).to(dt).to("npu")
    k = (torch.randn(B, S_kv, N, D, generator=g) * (torch.rand(B, S_kv, N, D, generator=g) * 2 - 1)).to(dt).to("npu")
    v = (torch.randn(B, S_kv, N, D, generator=g) * (torch.rand(B, S_kv, N, D, generator=g) * 2 - 1)).to(dt).to("npu")
    o_new = cur.mha(q, k, v, scaleValue=-1.0, is_causal=causal)
    o_old = old.mha(q, k, v, scaleValue=-1.0, is_causal=causal)
    torch.npu.synchronize()
    be = torch.equal(o_new.cpu(), o_old.cpu())
    print(f"[{'OK' if be else 'FAIL'}] B{B} S{S} Skv{S_kv} N{N} D{D} causal={causal}: {'BITEXACT' if be else 'DIFF'}")
    fail += int(not be)

print("== DECODE (golden blocking caliber + closeness vs perf7) ==")
for B, S, S_kv, N, D, causal in DECODE:
    g = torch.Generator().manual_seed(4242)
    dt = torch.float16
    q = (torch.randn(B, S, N, D, generator=g) * (torch.rand(B, S, N, D, generator=g) * 2 - 1)).to(dt)
    k = (torch.randn(B, S_kv, N, D, generator=g) * (torch.rand(B, S_kv, N, D, generator=g) * 2 - 1)).to(dt)
    v = (torch.randn(B, S_kv, N, D, generator=g) * (torch.rand(B, S_kv, N, D, generator=g) * 2 - 1)).to(dt)
    o_new = cur.mha(q.to("npu"), k.to("npu"), v.to("npu"), scaleValue=-1.0, is_causal=causal)
    o_old = old.mha(q.to("npu"), k.to("npu"), v.to("npu"), scaleValue=-1.0, is_causal=causal)
    torch.npu.synchronize()
    o_new_c, o_old_c = o_new.cpu(), o_old.cpu()
    ref = tm.golden_mha(q, k, v, scaleValue=-1.0, is_causal=causal)
    mixed_ok, ratio, max_abs = tm.check_precision(o_new_c, ref, "float16")
    mere, mare_abs, mare_rel = tm.mere_mare(o_new_c, ref)
    thr = tm.eco_threshold("float16")
    gate_ok = mixed_ok and (mare_abs == mare_abs) and mare_abs < 10 * thr
    delta_old = (o_new_c.float() - o_old_c.float()).abs().max().item()
    print(
        f"[{'OK' if gate_ok else 'FAIL'}] B{B} S{S} Skv{S_kv} N{N} D{D} causal={causal}: "
        f"matched={ratio:.5f} max_abs={max_abs:.3e} MARE_abs={mare_abs:.3e}(gate<{10*thr:.3e}) |Δvs_perf7|max={delta_old:.3e}"
    )
    fail += int(not gate_ok)

print("PROBE_RESULT:", "PASS" if fail == 0 else f"FAIL({fail})")
sys.exit(1 if fail else 0)
