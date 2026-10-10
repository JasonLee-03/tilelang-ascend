"""Scratch probe (NOT a deliverable): #11 prefill bN=128 numerics.

old = current mha.py (post-#10b, prefill bN=64). Patch: prefill block_N
64 -> 128 (decode branch untouched). Prefill multi-block shapes shift the
online-softmax reduction width (64->128) -> NOT bitwise, blocking caliber;
S_kv<=128 shapes become single_kv_block (one-shot pre-normalized P_hat) ->
expected bit-exact-to-golden (higher-precision path, #8d precedent).
Decode shapes must stay BITEXACT vs post-#10b.
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


def _make_patched():
    src = open(os.path.join(_OP_DIR, "mha.py"), encoding="utf-8").read()
    old = "    block_N = 256 if (S <= 16 and S_kv > 128) else (128 if S <= 16 else 64)"
    new = "    block_N = 256 if (S <= 16 and S_kv > 128) else (128 if S <= 16 else 128)"
    assert old in src, "patch anchor not found"
    out = "/tmp/opencode/mha_pf128.py"
    open(out, "w", encoding="utf-8").write(src.replace(old, new))
    return _load(out, "mha_pf128")


DECODE_BITEXACT = [  # decode branch untouched by the patch
    (1, 1, 2048, 32, 128, False, "float16"),
    (8, 1, 128, 32, 128, False, "float16"),
    (4, 2, 512, 16, 256, False, "float16"),
]

SINGLE_BLOCK = [  # prefill S_kv<=128: kv_iters 2->1, expect golden-bit-exact
    (2, 128, 128, 8, 64, True, "float16"),     # case1 exact
    (2, 512, 128, 16, 128, False, "float16"),  # case6-like cross (S>Skv)
    (2, 128, 128, 2, 128, False, "float16"),   # case20-like (zeros below)
    (1, 65, 96, 2, 256, True, "float16"),      # odd tail single + trim
    (1, 127, 127, 1, 64, False, "float16"),    # odd bound-mask single
]

MULTI_BLOCK = [  # reduction width 64->128: blocking caliber
    (2, 256, 256, 4, 64, True, "bfloat16"),    # case2-like bf16
    (2, 512, 512, 2, 128, True, "bfloat16"),   # case3-like bf16
    (2, 128, 256, 12, 64, False, "float16"),   # case5-like cross
    (2, 1024, 1024, 2, 128, True, "float16"),  # case7-like long causal
    (2, 256, 256, 8, 256, True, "float16"),    # case4-like D=256
    (4, 1024, 1024, 2, 256, True, "float16"),  # case9-like D=256 long
    (1, 200, 300, 2, 128, True, "float16"),    # odd causal: trim+bound+3-iter
    (2, 130, 260, 2, 64, False, "float16"),    # odd tail rows + bound
]


def _inputs(B, S, S_kv, N, D, dt_s, seed, zeros=False):
    if zeros:
        z = lambda *sh: torch.zeros(*sh, dtype=getattr(torch, dt_s))  # noqa: E731
        return z(B, S, N, D), z(B, S_kv, N, D), z(B, S_kv, N, D)
    g = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dt_s)
    mk = lambda r, c: (torch.randn(r, c, generator=g) * (torch.rand(r, c, generator=g) * 2 - 1)).to(dt)  # noqa: E731
    q = mk(B * S * N, D).reshape(B, S, N, D)
    k = mk(B * S_kv * N, D).reshape(B, S_kv, N, D)
    v = mk(B * S_kv * N, D).reshape(B, S_kv, N, D)
    return q, k, v


def _run(cur, old, tm, shapes, mode, fail_box):
    for B, S, S_kv, N, D, causal, dt_s in shapes:
        zeros = mode == "single" and S_kv == 128 and N == 2 and D == 128 and not causal
        q, k, v = _inputs(B, S, S_kv, N, D, dt_s, 5150, zeros=zeros)
        qn, kn, vn = q.to("npu"), k.to("npu"), v.to("npu")
        o_new = cur.mha(qn, kn, vn, scaleValue=-1.0, is_causal=causal)
        o_old = old.mha(qn, kn, vn, scaleValue=-1.0, is_causal=causal)
        torch.npu.synchronize()
        o_new_c, o_old_c = o_new.cpu(), o_old.cpu()
        tag = f"B{B} S{S} Skv{S_kv} N{N} D{D} c={causal} {dt_s}{' ZEROS' if zeros else ''}"
        if mode == "bitexact":
            be = torch.equal(o_new_c, o_old_c)
            print(f"[{'OK' if be else 'FAIL'}] {tag}: {'BITEXACT' if be else 'DIFF'}")
            fail_box[0] += int(not be)
            continue
        ref = tm.golden_mha(q, k, v, scaleValue=-1.0, is_causal=causal)
        mixed_ok, ratio, max_abs = tm.check_precision(o_new_c, ref, dt_s)
        mere, mare_abs, mare_rel = tm.mere_mare(o_new_c, ref)
        thr = tm.eco_threshold(dt_s)
        gate_ok = mixed_ok and (mare_abs == mare_abs) and mare_abs < 10 * thr
        d_old = (o_new_c.float() - o_old_c.float()).abs().max().item()
        be_gold = torch.equal(o_new_c, ref.to(o_new_c.dtype))
        extra = " GOLDEN-BITEXACT" if be_gold else ""
        if mode == "single" and not be_gold:
            # INFORMATIONAL: golden-bit-exactness on single-block conversion is
            # a statistical outcome (fp16 output rounding absorbs fp32 ulp diffs
            # except at rounding boundaries), not a gate — test_mha.py blocking
            # caliber is mixed tolerance + MARE_abs (MERE reference-only). The
            # directional improvement is |Δgolden| << |Δvs_old| (closer to
            # golden than the multi-block bN=64 path was).
            extra = f" NOT-GOLDEN-BITEXACT (Δgolden={max_abs:.2e} vs Δold={d_old:.2e})"
        print(
            f"[{'OK' if gate_ok else 'FAIL'}] {tag}: matched={ratio:.5f} max_abs={max_abs:.3e} "
            f"MARE_abs={mare_abs:.3e}(gate<{10*thr:.3e}) |Δvs_old|max={d_old:.3e}{extra}"
        )
        fail_box[0] += int(not gate_ok)


def main():
    fail_box = [0]
    cur = _make_patched()
    old = _load(os.path.join(_OP_DIR, "mha.py"), "mha_post10b")
    tm = _load(os.path.join(_OP_DIR, "test_mha.py"), "test_mha_mod")

    print("== DECODE guards: must be BITEXACT vs post-#10b ==")
    _run(cur, old, tm, DECODE_BITEXACT, "bitexact", fail_box)
    print("== PREFILL single-block conversions: expect GOLDEN-BITEXACT ==")
    _run(cur, old, tm, SINGLE_BLOCK, "single", fail_box)
    print("== PREFILL multi-block: golden blocking caliber ==")
    _run(cur, old, tm, MULTI_BLOCK, "multi", fail_box)

    print("PROBE_RESULT:", "PASS" if fail_box[0] == 0 else f"FAIL({fail_box[0]})")
    sys.exit(1 if fail_box[0] else 0)


if __name__ == "__main__":
    main()
