#!/usr/bin/env python3
"""[H3 / eval 20260910_173445 case 67 — NaN-position mismatch] matrix repro.

Sweeps {nan, inf, fullrange} x {fp16, bf16} x {G1, G4, G16} x {causal, non}
x {aligned, non-aligned} x {prefill, decode} with the bench's OWN
data_generator (imported from kernel_eval — value semantics identical to
the eval: nan=50% deterministic-random positions, inf=flat[:n]=-inf /
flat[-n:]=+inf with n=numel//20, fullrange=[-65504, 65504]) and the bench's
own compare_tensors checker.  Any cell failing with the case-67 signature
("NaN位置不匹配" / partial NaN mismatch) is the reproduction target.

If no cell reproduces at seed-0, the sweep is repeated at 10 more seeds
(--seeds N) and the result is reported honestly.

Usage:
  unset PYTHONPATH; export ACL_OP_INIT_MODE=1
  python examples/gqa/eval_fix_verify/h3_matrix.py [--seeds 10] [--quick]
"""
import argparse
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
from kernel_eval.data.data_generator import DataGenerator  # noqa: E402
from kernel_eval.utils.compare import compare_tensors  # noqa: E402
from kernel_eval.utils.thresholds import get_threshold  # noqa: E402
import gqa as g  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "gqa_golden", os.path.join(BENCH, "tasks", "level4", "gqa", "golden.py"))
gqa_golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gqa_golden)

NPU = "npu:0"

# value_range triples exactly as the eval cases.yaml spells them
# YAML-parsed eval semantics: cases.yaml spells [.nan, .nan] / [-.inf, .inf]
# which the YAML loader delivers as FLOAT nan/-inf/inf (not strings)
VR = {
    "nan": [float("nan"), float("nan")],
    "inf": [float("-inf"), float("inf")],
    "fullrange": [-65504, 65504],
}

# (tag, N_q, N_kv) — G = N_q // N_kv
GROUPS = [
    ("G1", 32, 32),
    ("G4", 32, 8),
    ("G16", 16, 1),
]

# (tag, path, B, S, S_kv, aligned)
SHAPES = [
    ("prefill-align", "prefill", 2, 128, 128, True),
    ("prefill-nonalign", "prefill", 2, 124, 252, False),
    ("decode-align", "decode", 16, 2, 2048, True),
    ("decode-nonalign", "decode", 16, 2, 2044, False),
]

# small-shape probes (~10us baseline class), run once per (mode, dtype, G)
# at causal + non on the non-aligned geometry
EXTRA_SHAPES = [
    ("prefill-tiny-S8", "prefill", 2, 8, 128, False),
    ("prefill-tiny-S32", "prefill", 2, 32, 252, False),
    ("prefill-S4", "prefill", 2, 5, 132, False),      # smallest prefill
    ("decode-tiny-B2", "decode", 2, 1, 2044, False),
    ("decode-tiny-skv132", "decode", 2, 2, 132, False),
    ("prefill-causal-S-gt-Skv", "prefill", 2, 128, 96, False),  # S > S_kv
]


def gen(shape, dtype_str, mode, seed):
    dg = DataGenerator(input_dist="uniform")
    gen_ = torch.Generator().manual_seed(seed)
    return dg.generate_input_tensor(
        list(shape), dtype_str, VR[mode], generator=gen_)


def run_cell(tag, B, S, Skv, Nq, Nkv, D, torch_dtype, dtype_str, causal,
             mode, seed):
    q = gen((B, S, Nq, D), dtype_str, mode, seed)
    k = gen((B, Skv, Nkv, D), dtype_str, mode, seed + 1)
    v = gen((B, Skv, Nkv, D), dtype_str, mode, seed + 2)
    ref = gqa_golden.gqa(q.double(), k.double(), v.double(), -1.0, causal)
    native = gqa_golden.gqa(q, k, v, -1.0, causal)
    try:
        out = g.gqa(query=q.to(NPU), key=k.to(NPU), value=v.to(NPU),
                    scaleValue=-1.0, is_causal=causal)
        torch.npu.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"[RUNTIME-FAIL] {tag}: {type(e).__name__}: {str(e)[-200:]}",
              flush=True)
        return "runtime_fail", None, False
    res = compare_tensors(output=out.cpu(), golden=ref, dtype=dtype_str,
                          native_output=native)
    extra = ""
    has_sig = False
    if not res.passed:
        nan_out = torch.isnan(out.float().cpu())
        nan_g = torch.isnan(ref.float())
        nan_match = bool(torch.all(nan_out == nan_g))
        extra = (f" | nan_out={int(nan_out.sum())} nan_gold={int(nan_g.sum())}"
                 f" nan_pos_match={nan_match}"
                 f" inf_out={int(torch.isinf(out.float().cpu()).sum())}"
                 f" inf_gold={int(torch.isinf(ref).sum())}")
        if not nan_match:
            extra += " <<< CASE67-SIGNATURE"
            has_sig = True
        print(f"[FAIL] {tag} mere={res.mere:.3e} mare={res.mare:.3e} "
              f"thr={get_threshold(dtype_str):.1e}{extra}"
              f" | {res.error_msg or ''}", flush=True)
        return "fail", res, has_sig
    print(f"[PASS] {tag} mere={res.mere:.3e} mare={res.mare:.3e} "
          f"thr={get_threshold(dtype_str):.1e}", flush=True)
    return "pass", res, has_sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=0,
                    help="extra seeds to rescan when seed-0 finds nothing")
    ap.add_argument("--seed-start", type=int, default=1,
                    help="first rescan seed (resume support)")
    ap.add_argument("--skip-base", action="store_true",
                    help="skip the seed-0 matrix (resume: only the rescan)")
    ap.add_argument("--quick", action="store_true",
                    help="one dtype (fp16), G4 only — smoke the harness")
    args = ap.parse_args()

    dtypes = [(torch.float16, "float16")]
    if not args.quick:
        dtypes.append((torch.bfloat16, "bfloat16"))
    groups = GROUPS if not args.quick else [("G4", 32, 8)]
    shapes = SHAPES if not args.quick else SHAPES[:2]

    tally = {"pass": 0, "fail": 0, "runtime_fail": 0}
    sig_cells = []
    cell_id = 0
    seed_base = 1000
    if args.skip_base:
        print("[H3] skipping seed-0 matrix (resume mode)", flush=True)
    for mode in (() if args.skip_base else ("nan", "inf", "fullrange")):
        for torch_dtype, dtype_str in dtypes:
            for gtag, Nq, Nkv in groups:
                for causal in (False, True):
                    for stag, _p, B, S, Skv, _al in shapes:
                        cell_id += 1
                        tag = (f"{mode}/{dtype_str}/{gtag}/"
                               f"{'causal' if causal else 'non'}/{stag}")
                        st, _res, has_sig = run_cell(
                            tag, B, S, Skv, Nq, Nkv, 128, torch_dtype,
                            dtype_str, causal, mode,
                            seed_base + cell_id * 10)
                        tally[st] += 1
                        if has_sig:
                            sig_cells.append(tag)
    # extra small-shape probes (G4 + fp16 + both causal, all modes)
    if not args.quick and not args.skip_base:
        for mode in ("nan", "inf", "fullrange"):
            for causal in (False, True):
                for stag, _p, B, S, Skv, _al in EXTRA_SHAPES:
                    cell_id += 1
                    tag = (f"{mode}/float16/G4-extra/"
                           f"{'causal' if causal else 'non'}/{stag}")
                    st, _res, has_sig = run_cell(
                        tag, B, S, Skv, 32, 8, 128, torch.float16, "float16",
                        causal, mode, seed_base + cell_id * 10)
                    tally[st] += 1
                    if has_sig:
                        sig_cells.append(tag)
    if not args.skip_base:
        total = sum(tally.values())
        print(f"\n=== matrix seed-0: {tally['pass']}/{total} PASS, "
              f"{tally['fail']} FAIL, {tally['runtime_fail']} RUNTIME-FAIL ===",
              flush=True)

    # --- rescan at extra seeds (deterministic cell-indexed seeds) ---
    if args.seeds > 0:
        print(f"\n=== rescan x{args.seeds} seeds "
              f"(seed {args.seed_start}..{args.seed_start + args.seeds - 1}) ===",
              flush=True)
        for s in range(args.seed_start, args.seed_start + args.seeds):
            cid = 0
            for mode in ("nan", "inf", "fullrange"):
                for torch_dtype, dtype_str in dtypes:
                    for gtag, Nq, Nkv in groups:
                        for causal in (False, True):
                            for stag, _p, B, S, Skv, _al in shapes:
                                cid += 1
                                tag = (f"s{s}/{mode}/{dtype_str}/{gtag}/"
                                       f"{'causal' if causal else 'non'}/"
                                       f"{stag}")
                                st, _r, has_sig = run_cell(
                                    tag, B, S, Skv, Nq, Nkv, 128,
                                    torch_dtype, dtype_str, causal, mode,
                                    seed_base + 100000 * s + cid * 10)
                                if st != "pass":
                                    tally[st] += 1
                                    if has_sig:
                                        sig_cells.append(tag)
            print(f"--- seed {s} done (cumulative: {tally}) ---", flush=True)
    print(f"\n[H3] final tally: {tally}", flush=True)
    if sig_cells:
        print(f"[H3] case67-signature cells: {sig_cells}", flush=True)
    else:
        print("[H3] no case67-signature (NaN-position-mismatch) cell "
              "reproduced locally", flush=True)


if __name__ == "__main__":
    main()
