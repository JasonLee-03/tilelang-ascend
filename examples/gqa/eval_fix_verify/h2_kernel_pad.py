#!/usr/bin/env python3
"""[H2 / eval 20260910_173445 cases 71/72/73/81/82] padding-in-kernel verify.

Three parts (per the attempt-3 H2 acceptance criteria):

  (i) torch dispatch-hook guard: on the NON-ALIGNED fp16/bf16 bench path,
      gqa() must not dispatch ANY torch device-side copy op whose fp16/bf16
      binary is missing on the eval SoC (ViewCopyAiCore slice __setitem__,
      .contiguous() materialization, Fill/Zeros/Cat).  The fp32 mask
      construction (torch.full + slice assign) is EXEMPT (verified available
      on the eval SoC in production on both prior rounds).
  (ii) numeric vs golden (bench checker compare_tensors): non-aligned
      prefill (B2 S124 Skv252 Nq32 Nkv8 D128 fp16/bf16 x causal/non) +
      decode (B60 S1 Skv2044 fp16) + D512 ladder-unification + row-pad-only
      + q_v1+KV-tail shapes.
  (iii) aligned cases BIT-EXACT vs the pre-H2 implementation
      (history_version/gqa_impl_eval_fix_attempt3.py; H1's reduce_max
      clear=False is provably bit-neutral — the pre-fill sentinel is below
      every reachable max — so any diff isolates H2).

Usage (pip tilelang, no set_env.sh):
  unset PYTHONPATH; export ACL_OP_INIT_MODE=1
  python examples/gqa/eval_fix_verify/h2_kernel_pad.py [part]  # part in {guard, numeric, bitexact}
"""
import contextlib
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BENCH = os.path.join(REPO, ".agents", "skills", "cann-bench")
sys.path.insert(0, os.path.join(REPO, "examples", "gqa"))
sys.path.insert(0, os.path.join(BENCH, "src"))

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
from kernel_eval.utils.compare import compare_tensors  # noqa: E402
from kernel_eval.utils.thresholds import get_threshold  # noqa: E402
import gqa as g  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "gqa_golden", os.path.join(BENCH, "tasks", "level4", "gqa", "golden.py"))
gqa_golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gqa_golden)

NPU = "npu:0"
_FP16_BF16 = (torch.float16, torch.bfloat16)


def _is_npu(x):
    try:
        return str(x.device).startswith("npu")
    except AttributeError:
        return False


@contextlib.contextmanager
def h2_d2d_guards():
    """Raise on any fp16/bf16 NPU-side torch copy/fill op (eval-SoC missing
    binaries).  fp32 (the mask construction) and CPU tensors are exempt."""
    orig_setitem = torch.Tensor.__setitem__
    orig_contig = torch.Tensor.contiguous
    orig_full = torch.full
    orig_zeros = torch.zeros
    orig_cat = torch.cat
    orig_copy = torch.Tensor.copy_
    trips = []

    def _bad(what):
        msg = f"GUARD[H2]: {what} on the non-aligned fp16/bf16 NPU path"
        trips.append(msg)
        raise RuntimeError(msg)

    def guarded_setitem(self, key, value):
        if _is_npu(self) and self.dtype in _FP16_BF16:
            _bad(f"slice __setitem__ (dtype={self.dtype}, shape="
                 f"{tuple(self.shape)}) — ViewCopyAiCore")
        return orig_setitem(self, key, value)

    def guarded_contiguous(self, *a, **kw):
        if _is_npu(self) and self.dtype in _FP16_BF16 \
                and not self.is_contiguous():
            _bad(f".contiguous() materialization (shape={tuple(self.shape)}, "
                 f"stride={self.stride()}) — ViewCopy/Transpose")
        return orig_contig(self, *a, **kw)

    def guarded_full(*a, **kw):
        out = orig_full(*a, **kw)
        if _is_npu(out) and out.dtype in _FP16_BF16:
            _bad(f"torch.full(dtype={out.dtype}, device=npu) — FillAiCore")
        return out

    def guarded_zeros(*a, **kw):
        out = orig_zeros(*a, **kw)
        if _is_npu(out) and out.dtype in _FP16_BF16:
            _bad(f"torch.zeros(dtype={out.dtype}, device=npu) — ZerosLike")
        return out

    def guarded_cat(tensors, *a, **kw):
        if any(_is_npu(t) and t.dtype in _FP16_BF16 for t in tensors):
            _bad("torch.cat on fp16/bf16 NPU tensors — ConcatD")
        return orig_cat(tensors, *a, **kw)

    def guarded_copy(self, src, non_blocking=False):
        if _is_npu(self) and self.dtype in _FP16_BF16 and _is_npu(src):
            _bad(f".copy_ d2d (dtype={self.dtype}) — ViewCopyAiCore")
        return orig_copy(self, src, non_blocking=non_blocking)

    torch.Tensor.__setitem__ = guarded_setitem
    torch.Tensor.contiguous = guarded_contiguous
    torch.full = guarded_full
    torch.zeros = guarded_zeros
    torch.cat = guarded_cat
    torch.Tensor.copy_ = guarded_copy
    try:
        yield trips
    finally:
        torch.Tensor.__setitem__ = orig_setitem
        torch.Tensor.contiguous = orig_contig
        torch.full = orig_full
        torch.zeros = orig_zeros
        torch.cat = orig_cat
        torch.Tensor.copy_ = orig_copy


def gen_uniform(shape, dtype, seed, lo=-1.0, hi=1.0):
    g_ = torch.Generator().manual_seed(seed)
    return (torch.rand(shape, dtype=torch.float64, generator=g_)
            * (hi - lo) + lo).clamp(-65504, 65504).to(dtype)


# ---------------------------------------------------------------------------
# (i) guard part
# ---------------------------------------------------------------------------
GUARD_CASES = [
    # (tag, B, S, S_kv, N_q, N_kv, D, causal, dtype) — the 5 round-3 failure
    # classes + the orchestrator's H2 shapes
    ("g-71-fp16-skv", 2, 128, 124, 32, 8, 128, False, torch.float16),
    ("g-72-bf16-skv", 2, 128, 124, 48, 4, 128, False, torch.bfloat16),
    ("g-73-fp16-S", 2, 124, 128, 32, 8, 128, False, torch.float16),
    ("g-81-fp16-causal-skv", 2, 128, 124, 32, 8, 128, True, torch.float16),
    ("g-h2-prefill-fp16", 2, 124, 252, 32, 8, 128, False, torch.float16),
    ("g-h2-prefill-bf16", 2, 124, 252, 32, 8, 128, True, torch.bfloat16),
    ("g-h2-prefill-nq128", 2, 124, 252, 128, 4, 128, True, torch.float16),
    ("g-h2-prefill-d512", 2, 124, 252, 8, 4, 512, False, torch.float16),
    ("g-h2-decode-fp16", 60, 1, 2044, 32, 8, 128, False, torch.float16),
    ("g-h2-decode-causal", 60, 1, 2044, 32, 8, 128, True, torch.float16),
    ("g-h2-decode-bf16", 16, 2, 2044, 32, 8, 128, True, torch.bfloat16),
]


def part_guard():
    n_pass = 0
    for tag, B, S, Skv, Nq, Nkv, D, causal, dt in GUARD_CASES:
        q = gen_uniform((B, S, Nq, D), dt, 7).to(NPU)
        k = gen_uniform((B, Skv, Nkv, D), dt, 8).to(NPU)
        v = gen_uniform((B, Skv, Nkv, D), dt, 9).to(NPU)
        try:
            with h2_d2d_guards():
                out = g.gqa(query=q, key=k, value=v, scaleValue=-1.0,
                            is_causal=causal)
                torch.npu.synchronize()
            assert out.shape == (B, S, Nq, D) and out.is_contiguous()
            print(f"[GUARD-PASS] {tag}", flush=True)
            n_pass += 1
        except RuntimeError as e:
            print(f"[GUARD-FAIL] {tag}: {e}", flush=True)
    print(f"guard: {n_pass}/{len(GUARD_CASES)} PASS", flush=True)
    return n_pass == len(GUARD_CASES)


# ---------------------------------------------------------------------------
# (ii) numeric part (bench checker)
# ---------------------------------------------------------------------------
NUMERIC_CASES = [
    ("n-prefill-fp16-noncausal", 2, 124, 252, 32, 8, 128, False, torch.float16),
    ("n-prefill-fp16-causal", 2, 124, 252, 32, 8, 128, True, torch.float16),
    ("n-prefill-bf16-noncausal", 2, 124, 252, 32, 8, 128, False, torch.bfloat16),
    ("n-prefill-bf16-causal", 2, 124, 252, 32, 8, 128, True, torch.bfloat16),
    ("n-rowpad-only-noncausal", 2, 124, 256, 32, 8, 128, False, torch.float16),
    ("n-prefill-nq128-causal", 2, 124, 252, 128, 4, 128, True, torch.float16),
    ("n-prefill-d512-noncausal", 2, 124, 252, 8, 4, 512, False, torch.float16),
    ("n-prefill-d512-causal", 2, 124, 252, 8, 4, 512, True, torch.float16),
    ("n-decode-fp16-noncausal", 60, 1, 2044, 32, 8, 128, False, torch.float16),
    ("n-decode-fp16-causal", 60, 1, 2044, 32, 8, 128, True, torch.float16),
    ("n-decode-bf16-causal", 16, 2, 2044, 32, 8, 128, True, torch.bfloat16),
]


def part_numeric():
    n_pass = 0
    for tag, B, S, Skv, Nq, Nkv, D, causal, dt in NUMERIC_CASES:
        q = gen_uniform((B, S, Nq, D), dt, 11)
        k = gen_uniform((B, Skv, Nkv, D), dt, 12)
        v = gen_uniform((B, Skv, Nkv, D), dt, 13)
        ref = gqa_golden.gqa(q.double(), k.double(), v.double(), -1.0, causal)
        native = gqa_golden.gqa(q, k, v, -1.0, causal)
        try:
            out = g.gqa(query=q.to(NPU), key=k.to(NPU), value=v.to(NPU),
                        scaleValue=-1.0, is_causal=causal)
            torch.npu.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"[RUNTIME-FAIL] {tag}: {type(e).__name__}: "
                  f"{str(e)[-200:]}", flush=True)
            continue
        dtype_str = str(dt).replace("torch.", "")
        res = compare_tensors(output=out.cpu(), golden=ref, dtype=dtype_str,
                              native_output=native)
        status = "PASS" if res.passed else "FAIL"
        print(f"[{status}] {tag} mere={res.mere:.3e} mare={res.mare:.3e} "
              f"maxdiff={res.max_diff:.3e} thr={get_threshold(dtype_str):.1e}"
              f"{'' if res.passed else ' | ' + (res.error_msg or '')}",
              flush=True)
        n_pass += int(res.passed)
    print(f"numeric: {n_pass}/{len(NUMERIC_CASES)} PASS", flush=True)
    return n_pass == len(NUMERIC_CASES)


# ---------------------------------------------------------------------------
# (iii) bit-exact part (aligned shapes, new vs pre-H2 backup module)
# ---------------------------------------------------------------------------
BITEXACT_CASES = [
    # (tag, B, S, S_kv, N_q, N_kv, D, causal, dtype) — ALL aligned; covers
    # whole-block V2 / q_v1+whole-KV / universal-tail / decode / D256 / D512
    ("b-c8-g2stack", 4, 1024, 1024, 32, 8, 128, True, torch.float16),
    ("b-qv1-wholekv", 2, 192, 1024, 128, 4, 128, False, torch.float16),
    ("b-tail-aligned", 2, 384, 640, 8, 2, 128, True, torch.float16),
    ("b-decode-causal", 16, 1, 2048, 32, 8, 128, True, torch.float16),
    ("b-decode-expert", 60, 2, 2048, 32, 8, 128, False, torch.float16),
    ("b-d256-perhead", 2, 256, 256, 16, 4, 256, True, torch.float16),
    ("b-d512-perhead-bm32", 2, 256, 256, 8, 4, 512, True, torch.float16),
]


def part_bitexact():
    backup_path = os.path.join(
        HERE, "..", "history_version", "gqa_impl_eval_fix_attempt3.py")
    bspec = importlib.util.spec_from_file_location("gqa_pre_h2", backup_path)
    gb = importlib.util.module_from_spec(bspec)
    bspec.loader.exec_module(gb)
    n_pass = 0
    for tag, B, S, Skv, Nq, Nkv, D, causal, dt in BITEXACT_CASES:
        q = gen_uniform((B, S, Nq, D), dt, 21)
        k = gen_uniform((B, Skv, Nkv, D), dt, 22)
        v = gen_uniform((B, Skv, Nkv, D), dt, 23)
        out_old = gb.gqa(query=q.to(NPU), key=k.to(NPU), value=v.to(NPU),
                         scaleValue=-1.0, is_causal=causal)
        out_new = g.gqa(query=q.to(NPU), key=k.to(NPU), value=v.to(NPU),
                        scaleValue=-1.0, is_causal=causal)
        torch.npu.synchronize()
        bit_dt = torch.int16 if dt in _FP16_BF16 else torch.int32
        same = torch.equal(out_old.view(bit_dt).cpu(),
                           out_new.view(bit_dt).cpu())
        n_diff = int((out_old.view(bit_dt).cpu()
                      != out_new.view(bit_dt).cpu()).sum())
        print(f"[{'BITEXACT' if same else 'DIFF'}] {tag}"
              f"{'' if same else f' — {n_diff} elems differ'}", flush=True)
        n_pass += int(same)
    print(f"bitexact: {n_pass}/{len(BITEXACT_CASES)} PASS", flush=True)
    return n_pass == len(BITEXACT_CASES)


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    ok = True
    if part in ("guard", "all"):
        ok &= part_guard()
    if part in ("numeric", "all"):
        ok &= part_numeric()
    if part in ("bitexact", "all"):
        ok &= part_bitexact()
    print(f"[H2] {'ALL PASS' if ok else 'FAILED'} (part={part})", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
