#!/usr/bin/python3
# coding=utf-8
"""Round-2 item-2 (K1 block-level threshold pruning) -- OFFLINE feasibility, part (a).

Measures, for the 18 cann-bench cases, how many K1 main-loop blocks are actually
prunable under the mechanism proposed by the Orchestrator:

    tau_k_run := K_MAX-th largest value among ALL elements of blocks 0..b-1
                 (== run_v[K_MAX-1] == min(run_v), since run_v is exactly the
                  sorted-desc top-K_MAX list of everything seen so far)
    block b is prunable  <=>  block_max(b) < tau_k_run      (STRICT LT, never LE)

Block 0 can never be pruned: run_v is initialised to -inf, so tau_k_run == -inf and
`block_max < -inf` is false. This reproduces the "first blocks are not prunable"
requirement without a special case.

Inputs are reproduced EXACTLY as the evaluation harness builds them:
  * kernel_eval.benches.cann_loader -> case_id / input_shapes / dtypes / value_ranges / attrs
  * kernel_eval.eval.evaluator      -> case_seed = (eval_seed + sha256(case_id)[:8] % 2^31) % 2^31
  * kernel_eval.data.data_generator -> DataGenerator(input_dist="uniform")
No NPU, no timing. Host-only statistics.
"""

import argparse
import hashlib
import json
import sys

import numpy as np
import torch

CB = "/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/.agents/skills/cann-bench"
sys.path.insert(0, CB + "/src")

from kernel_eval.data.data_generator import DataGenerator  # noqa: E402

TASK_DIR = CB + "/bench_lab/tilelang_ascend_bench"
EVAL_SEED = 0  # cli.py:370 default


def case_seed(case_id_str: str) -> int:
    """evaluator.py:317-322 verbatim."""
    digest = hashlib.sha256(case_id_str.encode("utf-8")).digest()
    deterministic_hash = int.from_bytes(digest[:8], byteorder="big") % (2**31)
    return (EVAL_SEED + deterministic_hash) % (2**31)


def simulate_row(row: np.ndarray, BV: int, K_MAX: int, verify: bool = False) -> dict:
    """Replay the K1 merge chain for one row; count prunable full blocks.

    run_v is kept SORTED DESCENDING and holds exactly the K_MAX largest values seen so
    far, so run_v[K_MAX-1] == min(run_v) == the true K_MAX-th largest (this mirrors the
    kernel, where T.tile.topk yields a descending list). NOTE: np.partition(...)[-K:]
    alone is UNORDERED inside the slice -- indexing it at [K-1] would silently give an
    arbitrary member of the top-K instead of the K-th largest, inflating the prune rate.
    """
    V = row.shape[0]
    n_blk_full = V // BV
    run_v = np.full(K_MAX, -np.inf, dtype=np.float64)  # sorted desc: top-K_MAX seen
    pruned = 0
    # margin diagnostics: how far block_max sits ABOVE the running threshold
    min_margin = np.inf  # min over blocks of (block_max - tau_k_run); >0 => never prunable
    for b in range(n_blk_full):
        blk = row[b * BV : (b + 1) * BV]
        bmax = float(blk.max())
        tau_run = float(run_v[K_MAX - 1])
        if verify:
            seen = row[: b * BV]
            true_tau = (-np.inf if seen.size < K_MAX
                        else float(np.partition(seen, -K_MAX)[-K_MAX:].min()))
            assert tau_run == true_tau, (b, tau_run, true_tau)
        margin = bmax - tau_run
        if b >= 1 and margin < min_margin:
            min_margin = margin
        if bmax < tau_run:  # STRICT LT -- the only admissible prune predicate
            pruned += 1
            continue  # online-softmax only; run_v untouched
        top_blk = np.partition(blk, -K_MAX)[-K_MAX:]
        merged = np.concatenate([run_v, top_blk])
        run_v = np.sort(np.partition(merged, -K_MAX)[-K_MAX:])[::-1]  # keep sorted desc
    return {
        "n_blk_full": n_blk_full,
        "prunable": pruned,
        "min_margin": (None if n_blk_full < 2 else float(min_margin)),
    }


def simulate(logits: np.ndarray, BV: int, K_MAX: int, verify: bool = False) -> dict:
    B = logits.shape[0]
    tot_blk = tot_pr = 0
    per_row = []
    margins = []
    for r in range(B):
        s = simulate_row(logits[r], BV, K_MAX, verify=verify)
        tot_blk += s["n_blk_full"]
        tot_pr += s["prunable"]
        per_row.append(s["prunable"])
        if s["min_margin"] is not None:
            margins.append(s["min_margin"])
    return {
        "B": B,
        "V": int(logits.shape[1]),
        "BV": BV,
        "K_MAX": K_MAX,
        "total_blocks": tot_blk,
        "prunable_blocks": tot_pr,
        "prune_rate": (tot_pr / tot_blk) if tot_blk else None,
        "min_margin_over_rows": (float(min(margins)) if margins else None),
        "rows_with_any_prune": int(sum(1 for p in per_row if p > 0)),
    }


# ---------------------------------------------------------------- sanity check
def self_test():
    """Validate the simulator on inputs whose prune rate is known a priori."""
    rng = np.random.default_rng(1234)
    BV, K_MAX = 4096, 256
    out = {}
    # (1) strictly descending -> after block 0 every later block lies wholly below tau_k_run
    desc = np.sort(rng.uniform(-8, 8, size=BV * 8))[::-1].copy()
    r = simulate_row(desc, BV, K_MAX, verify=True)
    assert r["prunable"] == r["n_blk_full"] - 1 == 7, f"descending broken: {r}"
    out["descending"] = r
    # (2) strictly ascending -> nothing is ever prunable (tau keeps rising)
    asc = np.sort(rng.uniform(-8, 8, size=BV * 8)).copy()
    r = simulate_row(asc, BV, K_MAX, verify=True)
    assert r["prunable"] == 0, f"ascending broken: {r}"
    out["ascending"] = r
    # (3) i.i.d. uniform -> expect 0 (the bench recipe family)
    for trial in range(3):
        iid = rng.uniform(-8, 8, size=BV * 16)
        r = simulate_row(iid, BV, K_MAX, verify=True)
        assert r["prunable"] == 0, f"trial {trial}: unexpected prune on iid: {r}"
        out[f"iid_uniform_trial{trial}"] = r
    # (4) i.i.d. gaussian (the four-gate recipe family) -> expect 0
    for trial in range(3):
        g = torch.randn(16 * BV).numpy().astype(np.float64) * 8.0
        r = simulate_row(g, BV, K_MAX, verify=True)
        assert r["prunable"] == 0, f"gauss trial {trial}: unexpected prune: {r}"
        out[f"iid_gauss_trial{trial}"] = r
    # (5) verify=True itself was exercised above (exact tau_k_run cross-check per block)
    return {k: {"n_blk_full": v["n_blk_full"], "prunable": v["prunable"],
                "min_margin": v["min_margin"]} for k, v in out.items()}


# ------------------------------------------------------------------ bench cases
def load_cases():
    """Parse the delivered cases.csv (authoritative; identical to cases.yaml)."""
    import csv

    out = []
    with open(TASK_DIR + "/r4_fused_sampling/cases.csv", newline="") as f:
        for row in csv.DictReader(f):
            out.append({
                "operator": row["operator"],
                "case_id": int(row["case_id"]),
                "input_shape": json.loads(row["input_shape"]),
                "dtype": json.loads(row["dtype"]),
                "attrs": json.loads(row["attrs"]),
                "value_range": json.loads(row["value_range"]),
            })
    return out


def bench_logits(case) -> np.ndarray:
    """Reproduce the harness-generated logits for one case (float32, as the NPU sees it)."""
    op_dir = "r4_fused_sampling"
    cid_str = f"{op_dir}_{case['case_id']}"
    gen = DataGenerator(input_dist="uniform")
    tensors = gen.generate_input_tensors_from_case(
        input_shapes=case["input_shape"],
        dtypes=case["dtype"],
        value_ranges=case["value_range"],
        seed=case_seed(cid_str),
    )
    return tensors[0].to(torch.float32).cpu().numpy().astype(np.float64)


def gate_logits(B, V, scale=8.0, seed=0) -> np.ndarray:
    """test_r4_fused_sampling.py::_make_inputs recipe: manual_seed(0); randn(B,V)*8.0."""
    torch.manual_seed(seed)
    return (torch.randn(B, V, dtype=torch.float32) * scale).cpu().numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--also-gate", action="store_true",
                    help="also measure the four-gate recipe (randn*8.0) on a few shapes")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the per-block exact tau_k_run cross-check (default: on)")
    args = ap.parse_args()
    verify = not args.no_verify

    report = {"self_test": self_test(), "bench_cases": [], "gate_recipe": [],
              "tau_crosscheck_enabled": verify}

    print("=== self-test (simulator validity) ===")
    print(json.dumps(report["self_test"]))

    print("\n=== (a) prune rate on the 18 cann-bench cases (uniform recipe) ===")
    hdr = f"{'case':>4} {'B':>4} {'V':>7} {'BV':>5} {'K_MAX':>5} {'n_blk':>5} " \
          f"{'tot_blk':>8} {'prunable':>9} {'rate':>8} {'min_margin':>12}"
    print(hdr)
    tot_b = tot_p = 0
    for c in load_cases():
        a = c["attrs"]
        BV, K_MAX = int(a["BV"]), int(a["K_MAX"])
        B, V = c["input_shape"][0]
        lg = bench_logits(c)
        assert lg.shape == (B, V), (lg.shape, (B, V))
        s = simulate(lg, BV, K_MAX, verify=verify)
        s.update({"case_id": c["case_id"], "value_range_logits": c["value_range"][0],
                  "note": "n_blk_full==0 -> tail-only, pruning structurally impossible"
                          if s["total_blocks"] == 0 else ""})
        report["bench_cases"].append(s)
        tot_b += s["total_blocks"]
        tot_p += s["prunable_blocks"]
        mm = s["min_margin_over_rows"]
        rate_s = "n/a" if s["prune_rate"] is None else format(s["prune_rate"], ".6f")
        mm_s = "n/a" if mm is None else format(mm, "+.4f")
        print(f"{c['case_id']:>4} {B:>4} {V:>7} {BV:>5} {K_MAX:>5} {V // BV:>5} "
              f"{s['total_blocks']:>8} {s['prunable_blocks']:>9} {rate_s:>8} {mm_s:>12}")
        del lg

    report["bench_totals"] = {"total_blocks": tot_b, "prunable_blocks": tot_p,
                              "prune_rate": (tot_p / tot_b) if tot_b else None}
    print(f"\nAGGREGATE over 18 cases: prunable {tot_p} / {tot_b} blocks"
          f"  rate={report['bench_totals']['prune_rate']}")

    if args.also_gate:
        print("\n=== (a') prune rate on the four-gate recipe (randn*8.0) ===")
        for (B, V, BV, K_MAX) in [(4, 16384, 4096, 256), (64, 152064, 4096, 256),
                                   (8, 8192, 4096, 256), (6, 2304, 2048, 128),
                                   (4, 4096, 4096, 256), (128, 32768, 4096, 256)]:
            lg = gate_logits(B, V)
            s = simulate(lg, BV, K_MAX, verify=verify)
            s.update({"B": B, "V": V, "BV": BV, "K_MAX": K_MAX, "recipe": "randn*8.0"})
            report["gate_recipe"].append(s)
            print(f"  B={B:<4} V={V:<7} BV={BV:<5} K_MAX={K_MAX:<4} "
                  f"tot={s['total_blocks']:<5} prunable={s['prunable_blocks']:<4} "
                  f"min_margin="
                  f"{('n/a' if s['min_margin_over_rows'] is None else format(s['min_margin_over_rows'],'+.4f'))}")
            del lg

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
