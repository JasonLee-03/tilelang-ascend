"""Scratch probe (NOT a deliverable): #13 decode bN=512 (D<=128, S_kv>256).

GEMM1 transpose_B L0B: nTile=512 x kL0Size=32 x 2B = 32768 == 4-step
ping-pong budget (common.h:1275-1278). GEMM2 keeps default kL0Size=128
(N-split path, B tile 128x128x2 = 32768 <= 32KB). D=256 excluded (L1).
old = current mha.py (post-#11). Decode Skv<=256 / D=256 / prefill must
stay BITEXACT; widened shapes use the blocking caliber.
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
    old1 = "    block_N = 256 if (S <= 16 and S_kv > 128) else 128"
    new1 = (
        "    if S <= 16:\n"
        "        block_N = 512 if (D <= 128 and S_kv > 256) else (256 if S_kv > 128 else 128)\n"
        "    else:\n"
        "        block_N = 128"
    )
    old2 = "    gemm1_kL0 = 64 if (S <= 16 and D == 256 and S_kv > 128) else 128"
    new2 = (
        "    gemm1_kL0 = (\n"
        "        32 if (S <= 16 and D <= 128 and S_kv > 256)\n"
        "        else (64 if (S <= 16 and D == 256 and S_kv > 128) else 128)\n"
        "    )"
    )
    assert old1 in src and old2 in src, "patch anchors not found"
    out = "/tmp/opencode/mha_bN512.py"
    open(out, "w", encoding="utf-8").write(src.replace(old1, new1).replace(old2, new2))
    return _load(out, "mha_bN512")


def _res_bytes(cc):
    acc = {"ub": 0, "l1": 0, "l0c": 0}
    for m in re.finditer(r"ascend_(ub|l1|l0c)\.GetWithOffset<(\w+)>\((\d+),\s*(\d+)\)", cc):
        acc[m.group(1)] = max(acc[m.group(1)], int(m.group(4)) + int(m.group(3)) * _DTYPE_BYTES[m.group(2)])
    return acc


STAGE_A = [
    (1, 1, 2048, 32, 128, False, "float16"),   # case10 -> 4 iters
    (4, 2, 512, 32, 128, True, "float16"),     # case16-like -> single causal
    (32, 1, 512, 32, 128, False, "float16"),   # case12/13-like -> single
    (16, 1, 1024, 32, 128, False, "float16"),  # case11-like -> 2 iters
    (1, 1, 4096, 2, 128, False, "float16"),    # domain max -> 8 iters
    (1, 1, 1024, 2, 64, False, "float16"),     # D=64 -> bN=512
    (1, 1, 300, 2, 64, False, "float16"),      # D=64 single + bound
    (1, 1, 513, 1, 128, False, "float16"),     # bound, 2 iters
    (2, 2, 1000, 4, 128, True, "float16"),     # odd causal trim
]

BITEXACT = [  # regions the patch must not touch
    (8, 1, 128, 32, 128, False, "float16"),    # decode Skv=128 -> bN=128
    (8, 2, 256, 32, 128, True, "float16"),     # decode Skv=256 -> bN=256
    (1, 1, 192, 2, 64, False, "float16"),      # decode -> bN=256
    (4, 1, 512, 16, 256, False, "float16"),    # D=256 decode -> bN=256
    (2, 128, 128, 8, 64, True, "float16"),     # prefill
    (2, 1024, 1024, 2, 128, True, "float16"),  # prefill
]

TOLERANCE = [
    (1, 1, 2048, 32, 128, False, "float16"),
    (16, 1, 1024, 32, 128, False, "float16"),
    (32, 1, 512, 32, 128, False, "float16"),
    (4, 2, 512, 32, 128, True, "float16"),
    (2, 16, 512, 2, 128, True, "float16"),
    (1, 1, 4096, 2, 128, False, "float16"),
    (2, 2, 1000, 4, 128, True, "float16"),
    (1, 1, 513, 1, 128, False, "float16"),
    (1, 1, 1024, 2, 64, False, "float16"),
    (1, 1, 300, 2, 64, False, "float16"),
    (4, 1, 768, 8, 128, False, "float16"),     # 2 iters + bound (768%512!=0)
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
    old = _load(os.path.join(_OP_DIR, "mha.py"), "mha_post11")
    tm = _load(os.path.join(_OP_DIR, "test_mha.py"), "test_mha_mod")

    print("== STAGE A: compile + resource readout (UB 196352 / L1 524032 / L0C 131072) ==")
    for B, S, S_kv, N, D, causal, dt_s in STAGE_A:
        try:
            kern = cur.mha_kernel(B=B, S=S, S_kv=S_kv, N=N, D=D, dtype=dt_s, scaleValue=-1.0, is_causal=causal)
            cc = kern.get_kernel_source()
            r = _res_bytes(cc)
            kl0 = sorted(set(re.findall(r"gemm_v0<[^>]+>", cc)))
            ok = r["ub"] <= 196352 and r["l1"] <= 524032 and r["l0c"] <= 131072
            print(
                f"[{'OK' if ok else 'OVER'}] B{B} S{S} Skv{S_kv} N{N} D{D} c={causal}: "
                f"UB={r['ub']/1024:.1f}KB L1={r['l1']/1024:.1f}KB L0C={r['l0c']/1024:.1f}KB | " + "; ".join(kl0)
            )
            fail += int(not ok)
        except Exception as e:  # noqa: BLE001
            print(f"[COMPILE_FAIL] B{B} S{S} Skv{S_kv} N{N} D{D} c={causal}: {type(e).__name__}: {str(e)[:400]}")
            fail += 1

    print("== STAGE B1: UNCHANGED regions BITEXACT vs post-#11 ==")
    for B, S, S_kv, N, D, causal, dt_s in BITEXACT:
        q, k, v = _inputs(B, S, S_kv, N, D, dt_s, 31337)
        qn, kn, vn = q.to("npu"), k.to("npu"), v.to("npu")
        o_new = cur.mha(qn, kn, vn, scaleValue=-1.0, is_causal=causal)
        o_old = old.mha(qn, kn, vn, scaleValue=-1.0, is_causal=causal)
        torch.npu.synchronize()
        be = torch.equal(o_new.cpu(), o_old.cpu())
        print(f"[{'OK' if be else 'FAIL'}] B{B} S{S} Skv{S_kv} N{N} D{D} c={causal}: {'BITEXACT' if be else 'DIFF'}")
        fail += int(not be)

    print("== STAGE B2: widened bN=512 — golden gate + closeness ==")
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
        be_g = torch.equal(o_new_c, ref.to(o_new_c.dtype))
        print(
            f"[{'OK' if gate_ok else 'FAIL'}] B{B} S{S} Skv{S_kv} N{N} D{D} c={causal}: "
            f"matched={ratio:.5f} max_abs={max_abs:.3e} MARE_abs={mare_abs:.3e}(gate<{10*thr:.3e}) "
            f"|Δvs_old|max={d_old:.3e}{' GOLDEN-BITEXACT' if be_g else ''}"
        )
        fail += int(not gate_ok)

    print("PROBE_RESULT:", "PASS" if fail == 0 else f"FAIL({fail})")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
