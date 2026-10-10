"""Scratch probe (NOT a deliverable): #10 decode-branch bN 256/512 adaptive.

Stage A: compile patched kernel (scratch copy in /tmp, mha.py untouched) for
         decode configs; read back actual UB/L1/L0C bytes from generated
         GetWithOffset lines; fail loudly on overflow.
Stage B: numerics —
  - UNCHANGED-path shapes (prefill + S_kv<=128 decode): BITEXACT vs perf9.
  - widened decode shapes: golden blocking caliber (mixed tol + MARE_abs gate,
    test_mha.py functions reused) + closeness vs perf9.
"""

import importlib.util
import os
import re
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_OP_DIR = os.path.dirname(_HERE)
sys.path.insert(0, _OP_DIR)
sys.path.insert(0, _HERE)

_DTYPE_BYTES = {"float": 4, "half": 2, "uint8_t": 1, "int8_t": 1, "int32_t": 4, "uint32_t": 4, "bfloat16_t": 2}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_patched():
    src = open(os.path.join(_OP_DIR, "mha.py"), encoding="utf-8").read()
    old = "    block_N = 128 if S <= 16 else 64"
    new = "    block_N = 256 if (S <= 16 and D <= 128 and S_kv > 128) else (128 if S <= 16 else 64)"
    assert old in src, "patch anchor not found"
    out = "/tmp/opencode/mha_bNwide.py"
    open(out, "w", encoding="utf-8").write(src.replace(old, new))
    return _load(out, "mha_bNwide")


def _res_bytes(cc):
    """max(offset+size*dtype) per on-chip space from GetWithOffset lines."""
    acc = {"ascend_ub": 0, "ascend_l1": 0, "ascend_l0c": 0}
    for m in re.finditer(r"ascend_(ub|l1|l0c)\.GetWithOffset<(\w+)>\((\d+),\s*(\d+)\)", cc):
        space, dt, n, off = m.group(0), m.group(2), int(m.group(3)), int(m.group(4))
        key = "ascend_" + m.group(1)
        acc[key] = max(acc[key], off + n * _DTYPE_BYTES[dt])
    return acc


STAGE_A = [  # (B,S,S_kv,N,D,causal,dtype,expect_bN)
    (1, 1, 2048, 32, 128, False, "float16", 256),   # case10: 16->8 iters
    (4, 2, 512, 32, 128, True, "float16", 256),     # case16-like causal: 4->2 iters
    (8, 2, 256, 32, 128, True, "bfloat16", 256),    # case17 exact (bf16) single-block
    (16, 1, 1024, 32, 128, False, "float16", 256),  # case11: 8->4 iters
    (32, 1, 512, 32, 128, False, "float16", 256),   # case12/13: 4->2 iters
    (1, 1, 4096, 2, 128, False, "float16", 256),    # domain max Skv
    (1, 1, 192, 2, 64, False, "float16", 256),      # D=64 bound-mask single
    (1, 1, 129, 1, 128, False, "float16", 256),     # bound single vs cur 2-iter
    (2, 16, 512, 2, 128, True, "float16", 256),     # S=16 causal
    (8, 1, 128, 32, 128, False, "float16", 128),    # case14-like UNCHANGED
    (4, 1, 512, 16, 256, False, "float16", 128),    # case15-like D=256 UNCHANGED
]

BITEXACT = [  # unchanged paths vs perf9
    (2, 128, 128, 8, 64, True, "float16"),
    (2, 1024, 1024, 2, 128, True, "float16"),
    (8, 1, 128, 32, 128, False, "float16"),
    (4, 1, 64, 8, 64, False, "float16"),
    (4, 1, 512, 16, 256, False, "float16"),   # D=256 decode stays bN=128
    (4, 2, 512, 16, 256, False, "float16"),   # D=256 S=2 stays bN=128
]

TOLERANCE = [  # widened decode: golden gate + closeness vs perf9
    (1, 1, 2048, 32, 128, False, "float16"),   # case10 8-iter multi-block
    (16, 1, 1024, 32, 128, False, "float16"),  # case11 4-iter
    (32, 1, 512, 32, 128, False, "float16"),   # case12/13 2-iter
    (4, 2, 512, 32, 128, True, "float16"),     # case16 2-iter causal
    (8, 2, 256, 32, 128, True, "bfloat16"),    # case17 single-block causal bf16
    (1, 1, 192, 2, 64, False, "float16"),      # bound-mask single
    (1, 1, 129, 1, 128, False, "float16"),     # bN=256 bound single vs cur 2-iter
    (2, 16, 512, 2, 128, True, "float16"),     # S=16 causal 2-iter
    (1, 1, 4096, 2, 128, False, "float16"),    # domain max 16-iter
    (2, 2, 1000, 4, 128, True, "float16"),     # causal odd Skv: trim+bound+4-iter
]

def _inputs(B, S, S_kv, N, D, dt_s, seed):
    g = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dt_s)
    mk = lambda r, c: (torch.randn(r, c, generator=g) * (torch.rand(r, c, generator=g) * 2 - 1)).to(dt)  # noqa: E731
    q = mk(B * S * N, D).reshape(B, S, N, D)
    k = mk(B * S_kv * N, D).reshape(B, S_kv, N, D)
    v = mk(B * S_kv * N, D).reshape(B, S_kv, N, D)
    return q, k, v


def main():
    fail = 0
    cur = _make_patched()
    old = _load(os.path.join(_OP_DIR, "history_version", "mha_impl_s3_perf9.py"), "mha_perf9")
    tm = _load(os.path.join(_OP_DIR, "test_mha.py"), "test_mha_mod")

    print("== STAGE A: compile + resource readout (limits UB 196352 / L1 524032 / L0C 131072) ==")
    for B, S, S_kv, N, D, causal, dt_s, exp_bN in STAGE_A:
        try:
            kern = cur.mha_kernel(B=B, S=S, S_kv=S_kv, N=N, D=D, dtype=dt_s, scaleValue=-1.0, is_causal=causal)
            cc = kern.get_kernel_source()
            res = _res_bytes(cc)
            ok = res["ascend_ub"] <= 196352 and res["ascend_l1"] <= 524032 and res["ascend_l0c"] <= 131072
            print(
                f"[{'OK' if ok else 'OVER'}] B{B} S{S} Skv{S_kv} N{N} D{D} c={causal} {dt_s} bN={exp_bN}: "
                f"UB={res['ascend_ub']/1024:.1f}KB L1={res['ascend_l1']/1024:.1f}KB L0C={res['ascend_l0c']/1024:.1f}KB"
            )
            fail += int(not ok)
        except Exception as e:  # noqa: BLE001
            print(f"[COMPILE_FAIL] B{B} S{S} Skv{S_kv} N{N} D{D} c={causal} {dt_s}: {type(e).__name__}: {str(e)[:300]}")
            fail += 1

    print("== STAGE B1: UNCHANGED paths must be BITEXACT vs perf9 ==")
    for B, S, S_kv, N, D, causal, dt_s in BITEXACT:
        q, k, v = _inputs(B, S, S_kv, N, D, dt_s, 31337)
        qn, kn, vn = q.to("npu"), k.to("npu"), v.to("npu")
        o_new = cur.mha(qn, kn, vn, scaleValue=-1.0, is_causal=causal)
        o_old = old.mha(qn, kn, vn, scaleValue=-1.0, is_causal=causal)
        torch.npu.synchronize()
        be = torch.equal(o_new.cpu(), o_old.cpu())
        print(f"[{'OK' if be else 'FAIL'}] B{B} S{S} Skv{S_kv} N{N} D{D} c={causal} {dt_s}: {'BITEXACT' if be else 'DIFF'}")
        fail += int(not be)

    print("== STAGE B2: widened decode — golden blocking caliber + closeness vs perf9 ==")
    for B, S, S_kv, N, D, causal, dt_s in TOLERANCE:
        q, k, v = _inputs(B, S, S_kv, N, D, dt_s, 4242)
        qn, kn, vn = q.to("npu"), k.to("npu"), v.to("npu")
        o_new = cur.mha(qn, kn, vn, scaleValue=-1.0, is_causal=causal)
        o_old = old.mha(qn, kn, vn, scaleValue=-1.0, is_causal=causal)
        torch.npu.synchronize()
        o_new_c, o_old_c = o_new.cpu(), o_old.cpu()
        ref = tm.golden_mha(q, k, v, scaleValue=-1.0, is_causal=causal)
        mixed_ok, ratio, max_abs = tm.check_precision(o_new_c, ref, dt_s)
        mere, mare_abs, mare_rel = tm.mere_mare(o_new_c, ref)
        thr = tm.eco_threshold(dt_s)
        gate_ok = mixed_ok and (mare_abs == mare_abs) and mare_abs < 10 * thr
        d_old = (o_new_c.float() - o_old_c.float()).abs().max().item()
        be = torch.equal(o_new_c, o_old_c)
        print(
            f"[{'OK' if gate_ok else 'FAIL'}] B{B} S{S} Skv{S_kv} N{N} D{D} c={causal} {dt_s}: "
            f"matched={ratio:.5f} max_abs={max_abs:.3e} MARE_abs={mare_abs:.3e}(gate<{10*thr:.3e}) "
            f"|Δvs_perf9|max={d_old:.3e}{' BITEXACT' if be else ''}"
        )
        fail += int(not gate_ok)

    print("PROBE_RESULT:", "PASS" if fail == 0 else f"FAIL({fail})")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
