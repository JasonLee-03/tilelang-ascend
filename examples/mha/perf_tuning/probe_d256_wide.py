"""Scratch probe (NOT a deliverable): #10b D=256 decode bN=256 via GEMM1 kL0Size=64.

old = current mha.py (post-#10a). Patch: widen the decode bN=256 predicate to
all D, and pass kL0Size=64 to GEMM1 only for the new domain (S<=16, D==256,
S_kv>128) so the transpose_B B-tile (256x64x2B = 32KB) fits the 4-step
ping-pong L0B budget (common.h:1275-1278).
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
    old1 = "    block_N = 256 if (S <= 16 and D <= 128 and S_kv > 128) else (128 if S <= 16 else 64)"
    new1 = "    block_N = 256 if (S <= 16 and S_kv > 128) else (128 if S <= 16 else 64)"
    old2 = "T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)"
    new2 = ("T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True, "
            "kL0Size=(64 if (S <= 16 and D == 256 and S_kv > 128) else 128))")
    assert old1 in src and old2 in src, "patch anchors not found"
    out = "/tmp/opencode/mha_d256wide.py"
    open(out, "w", encoding="utf-8").write(src.replace(old1, new1).replace(old2, new2))
    return _load(out, "mha_d256wide")


def _res_bytes(cc):
    acc = {"ascend_ub": 0, "ascend_l1": 0, "ascend_l0c": 0}
    for m in re.finditer(r"ascend_(ub|l1|l0c)\.GetWithOffset<(\w+)>\((\d+),\s*(\d+)\)", cc):
        dt, n, off = m.group(2), int(m.group(3)), int(m.group(4))
        key = "ascend_" + m.group(1)
        acc[key] = max(acc[key], off + n * _DTYPE_BYTES[dt])
    return acc


STAGE_A = [
    (4, 1, 512, 16, 256, False, "float16"),   # case15-like
    (4, 2, 512, 16, 256, False, "float16"),   # case18-like
    (2, 2, 512, 4, 256, True, "float16"),     # D=256 causal MTP
    (1, 1, 1024, 2, 256, False, "float16"),   # 4-iter
    (1, 1, 300, 2, 256, False, "float16"),    # single-block + bound mask
]

BITEXACT = [  # regions the patch must not touch
    (2, 256, 256, 2, 256, True, "float16"),   # prefill D=256 (bN=64)
    (2, 1, 128, 4, 256, False, "float16"),    # decode D=256 Skv<=128 (bN=128)
    (1, 1, 128, 2, 256, False, "float16"),    # decode D=256 Skv=128 (bN=128)
    (1, 1, 2048, 32, 128, False, "float16"),  # decode D=128 (already bN=256 via #10a)
    (2, 128, 128, 8, 64, True, "float16"),    # prefill D=64
    (1, 65, 96, 2, 256, True, "float16"),     # prefill D=256 odd tail
]

TOLERANCE = [  # newly widened D=256 decode
    (4, 1, 512, 16, 256, False, "float16"),
    (4, 2, 512, 16, 256, False, "float16"),
    (2, 2, 512, 4, 256, True, "float16"),
    (1, 1, 1024, 2, 256, False, "float16"),
    (1, 1, 300, 2, 256, False, "float16"),
    (2, 1, 256, 4, 256, False, "float16"),    # single-block exact
    (2, 1, 129, 2, 256, False, "float16"),    # single-block bound
    (1, 2, 4096, 2, 256, True, "float16"),    # domain max causal
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
    old = _load(os.path.join(_OP_DIR, "mha.py"), "mha_post10a")
    tm = _load(os.path.join(_OP_DIR, "test_mha.py"), "test_mha_mod")

    print("== STAGE A: compile + resource readout (UB 196352 / L1 524032 / L0C 131072) ==")
    for B, S, S_kv, N, D, causal, dt_s in STAGE_A:
        try:
            kern = cur.mha_kernel(B=B, S=S, S_kv=S_kv, N=N, D=D, dtype=dt_s, scaleValue=-1.0, is_causal=causal)
            cc = kern.get_kernel_source()
            res = _res_bytes(cc)
            kl0 = set(re.findall(r"gemm_v0<[^>]+>", cc))
            ok = res["ascend_ub"] <= 196352 and res["ascend_l1"] <= 524032 and res["ascend_l0c"] <= 131072
            print(
                f"[{'OK' if ok else 'OVER'}] B{B} S{S} Skv{S_kv} N{N} D{D} c={causal}: "
                f"UB={res['ascend_ub']/1024:.1f}KB L1={res['ascend_l1']/1024:.1f}KB L0C={res['ascend_l0c']/1024:.1f}KB | "
                + "; ".join(sorted(kl0))
            )
            fail += int(not ok)
        except Exception as e:  # noqa: BLE001
            print(f"[COMPILE_FAIL] B{B} S{S} Skv{S_kv} N{N} D{D} c={causal}: {type(e).__name__}: {str(e)[:400]}")
            fail += 1

    print("== STAGE B1: UNCHANGED regions must be BITEXACT vs post-#10a ==")
    for B, S, S_kv, N, D, causal, dt_s in BITEXACT:
        q, k, v = _inputs(B, S, S_kv, N, D, dt_s, 31337)
        qn, kn, vn = q.to("npu"), k.to("npu"), v.to("npu")
        o_new = cur.mha(qn, kn, vn, scaleValue=-1.0, is_causal=causal)
        o_old = old.mha(qn, kn, vn, scaleValue=-1.0, is_causal=causal)
        torch.npu.synchronize()
        be = torch.equal(o_new.cpu(), o_old.cpu())
        print(f"[{'OK' if be else 'FAIL'}] B{B} S{S} Skv{S_kv} N{N} D{D} c={causal}: {'BITEXACT' if be else 'DIFF'}")
        fail += int(not be)

    print("== STAGE B2: widened D=256 decode — golden gate + closeness ==")
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
            f"[{'OK' if gate_ok else 'FAIL'}] B{B} S{S} Skv{S_kv} N{N} D{D} c={causal}: "
            f"matched={ratio:.5f} max_abs={max_abs:.3e} MARE_abs={mare_abs:.3e}(gate<{10*thr:.3e}) "
            f"|Δvs_old|max={d_old:.3e}{' BITEXACT' if be else ''}"
        )
        fail += int(not gate_ok)

    print("PROBE_RESULT:", "PASS" if fail == 0 else f"FAIL({fail})")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
