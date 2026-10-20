#!/usr/bin/env python3
"""[G2 / eval case 67 — NaN position mismatch] Special-value reproduction.

Replicates the eval pipeline exactly:
  - input generation per kernel_eval/data/data_generator.py:
      * full-range fp16: uniform [-65504, 65504]  (value_range path, f64 rand)
      * NaN:  rand*2-1 base, 50% deterministic-random positions set to NaN
      * Inf:  rand*2-1 base, flat[:n]=-Inf, flat[-n:]=+Inf  (n = numel//20)
  - golden: tasks/level4/gqa/golden.py at fp64 on CPU (golden_precision=
    fp64_cpu default), inputs cast .double()
  - kernel: examples/gqa/gqa.py gqa(query=..., key=..., value=...) BSND
  - comparison: kernel_eval.utils.compare.compare_tensors (the eval checker)

Usage (repo root NOT in path; bench src on PYTHONPATH):
  unset PYTHONPATH; export ACL_OP_INIT_MODE=1
  PYTHONPATH=<bench>/src python g2_special_values.py [filter]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.abspath(os.path.join(
    HERE, "..", "..", "..", ".agents", "skills", "cann-bench"))
sys.path.insert(0, os.path.join(BENCH, "src"))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))  # examples/gqa

import torch  # noqa: E402
from kernel_eval.utils.compare import compare_tensors  # noqa: E402
from kernel_eval.utils.thresholds import get_threshold  # noqa: E402
import gqa as gqa_mod  # noqa: E402  (examples/gqa/gqa.py)

sys.path.insert(0, os.path.join(BENCH, "tasks", "level4", "gqa"))
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "gqa_golden", os.path.join(BENCH, "tasks", "level4", "gqa", "golden.py"))
gqa_golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gqa_golden)

NPU = "npu:0"


def gen_tensor(shape, dtype, mode, seed):
    """Faithful replica of DataGenerator._gen_float/_gen_special (uniform path)."""
    g = torch.Generator().manual_seed(seed)
    if mode in ("fullrange", "fullrange_bf16max"):
        # eval variant range is the FP16 bound pair applied to either dtype
        min_val, max_val = -65504.0, 65504.0
        if mode == "fullrange_bf16max":
            min_val, max_val = -3.389e38, 3.389e38  # dtype-true bf16 bounds
        rand_f64 = torch.rand(shape, dtype=torch.float64, generator=g)
        t = rand_f64 * (max_val - min_val) + min_val
        finfo = torch.finfo(dtype)
        return torch.clamp(t, finfo.min, finfo.max).to(dtype)
    # special path: rand*2-1 base (fp32) then positions
    rand_base = torch.rand(shape, dtype=torch.float32, generator=g)
    t = (rand_base * 2 - 1).to(dtype)
    if mode == "nan":
        nan_rand = torch.rand(shape, dtype=torch.float32, generator=g)
        t = t.float()  # NaN assignment in fp32 then cast back (matches eval:
        # tensor[nan_mask] = nan on the dtype tensor; NaN survives all casts)
        nan_mask = nan_rand < 0.5
        t[nan_mask] = float("nan")
        return t.to(dtype)
    if mode == "inf":
        flat = t.float().flatten()
        n = max(1, len(flat) // 20)
        flat[:n] = float("-inf")
        flat[-n:] = float("inf")
        return flat.reshape(shape).to(dtype)
    raise ValueError(mode)


def run_case(tag, B, S, S_kv, N_q, N_kv, D, torch_dtype, causal, mode, seed):
    q = gen_tensor((B, S, N_q, D), torch_dtype, mode, seed)
    k = gen_tensor((B, S_kv, N_kv, D), torch_dtype, mode, seed + 1)
    v = gen_tensor((B, S_kv, N_kv, D), torch_dtype, mode, seed + 2)

    # golden: fp64 CPU (fp64_cpu strategy)
    out_golden = gqa_golden.gqa(
        q.double().cpu(), k.double().cpu(), v.double().cpu(),
        scaleValue=-1.0, is_causal=causal)
    # native: same-precision CPU golden (the eval's stage-2 small-value
    # reference — bench_func falls back to golden at original precision)
    out_native = gqa_golden.gqa(
        q.cpu(), k.cpu(), v.cpu(), scaleValue=-1.0, is_causal=causal)

    # kernel: BSND on NPU
    q_npu, k_npu, v_npu = q.to(NPU), k.to(NPU), v.to(NPU)
    try:
        out = gqa_mod.gqa(query=q_npu, key=k_npu, value=v_npu,
                          scaleValue=-1.0, is_causal=causal)
        torch.npu.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"[RUNTIME-FAIL] {tag}: {type(e).__name__}: {str(e)[-200:]}", flush=True)
        return False

    dtype_str = str(torch_dtype).replace("torch.", "")
    res = compare_tensors(output=out.cpu(), golden=out_golden, dtype=dtype_str,
                          native_output=out_native)
    status = "PASS" if res.passed else "FAIL"
    extra = ""
    if not res.passed:
        nan_out = torch.isnan(out.float().cpu())
        nan_g = torch.isnan(out_golden.float())
        extra = (f" | nan_out={int(nan_out.sum())} nan_gold={int(nan_g.sum())}"
                 f" nan_pos_match={bool(torch.all(nan_out == nan_g))}"
                 f" inf_out={int(torch.isinf(out.float().cpu()).sum())}"
                 f" inf_gold={int(torch.isinf(out_golden).sum())}")
    print(f"[{status}] {tag} mere={res.mere:.3e} mare={res.mare:.3e} "
          f"maxdiff={res.max_diff:.3e} thr={get_threshold(dtype_str):.1e}{extra}",
          flush=True)
    return res.passed


def main():
    filt = sys.argv[1] if len(sys.argv) > 1 else ""
    cases = []
    # base1-class prefill shape (g2stack kernel), fp16/bf16 x causal/non x 3 modes
    for dt, dname in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
        for causal in (False, True):
            for mode in ("fullrange", "nan", "inf"):
                cases.append((f"{dname} B2S128Nq32Nkv8D128 "
                              f"{'causal' if causal else 'noncausal'} {mode}",
                              2, 128, 128, 32, 8, 128, dt, causal, mode))
    # MHA per-head path (G=1) — causal fullrange exercises the per-head
    # mask-after-sub order at full magnitude
    for mode in ("fullrange", "nan", "inf"):
        cases.append((f"fp16 B2S128Nq32Nkv32D128 MHA noncausal {mode}",
                      2, 128, 128, 32, 32, 128, torch.float16, False, mode))
        cases.append((f"fp16 B2S128Nq32Nkv32D128 MHA causal {mode}",
                      2, 128, 128, 32, 32, 128, torch.float16, True, mode))
    # decode paths (dense_expert / AUTO dense) — noncausal + causal (c16 class)
    for mode in ("fullrange", "nan", "inf"):
        cases.append((f"fp16 B16S1Skv2048Nq32Nkv8D128 decode {mode}",
                      16, 1, 2048, 32, 8, 128, torch.float16, False, mode))
        cases.append((f"fp16 B60S2Skv2048Nq32Nkv8D128 decode-causal {mode}",
                      60, 2, 2048, 32, 8, 128, torch.float16, True, mode))
    # informational: dtype-true bf16 full range (scores overflow fp32 ->
    # +-Inf/NaN; NOT in the eval variant list [-65504,65504] — documents the
    # known boundary; expected FAIL, excluded from the pass count summary)
    cases.append((f"bf16 B2S128Nq32Nkv8D128 causal fullrange_bf16max",
                  2, 128, 128, 32, 8, 128, torch.bfloat16, True,
                  "fullrange_bf16max"))

    n_pass = 0
    n_total = 0
    for i, (tag, B, S, S_kv, N_q, N_kv, D, dt, causal, mode) in enumerate(cases):
        if filt and filt not in tag:
            continue
        ok = run_case(tag, B, S, S_kv, N_q, N_kv, D, dt, causal, mode, seed=1000 + i * 10)
        if "fullrange_bf16max" in tag:
            print("  (informational only — dtype-true bf16 range is outside "
                  "the eval variant set; not counted)", flush=True)
            continue
        n_pass += ok
        n_total += 1
    print(f"\n=== {n_pass}/{n_total} passed ===", flush=True)


if __name__ == "__main__":
    main()
