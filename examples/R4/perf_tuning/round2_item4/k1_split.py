"""Position-split msprof kernel_details.csv into K1 / K2 per bench case.

Each profiled step contains 3 rows: CannBenchCacheClean + N x `main_kernel`, where N is the
number of TileLang kernels the host wrapper launched for that (task, case). All TileLang
kernels are emitted under the same `main_kernel` name, so they are separated by POSITION
within the step -- the same split round 1 used.

Cases run sequentially, 8 steps each (warmup=3 + repeat=5), so 144 steps = 18 cases.

=========================== rev3 CONSCIOUS GATE UPDATE ===========================
WHY THIS FILE CHANGED (DESIGN.md section 15.3.1; the orchestrator red line reads: "that gate
MUST be consciously updated, not silently broken").

Before rev3 every step had exactly 2 `main_kernel` rows (K1 then K2) and both had
Block Num == ceil(B/2), so this script cross-checked against that constant. rev3 adds a
split-V path whose launch count AND Block Num sequence both change:

    path    task    rows/step   Block Num per row (launch order)
    current mask    2           [b_num, b_num]                        (K1, K2m)
    current fused   2           [b_num, b_num]                        (K1, K2g)
    split   mask    3           [b_num*S, b_num, b_num]               (K1A, K1B, K2m)
    split   fused   4           [b_num*S, b_num, b_num*S, b_num]      (K1A, K1B, K2gA, K2gB)

The old code did `if len(seq) != 2: bad += 1; continue`, which would have SILENTLY DISCARDED
every split-path step (6 cases x 8 steps = 48 steps, i.e. a third of the data) while still
printing a plausible-looking report -- precisely the "silent failure" the red line forbids.

The 4-step update from section 15.3.1, as implemented here:
  1. Expected row count and Block Num sequence are DERIVED from the host's own dispatch
     function `_r4_split_plan(B, V, K_MAX, BV)` -- imported from the delivered kernel, so
     there is a SINGLE SOURCE OF TRUTH. Change the dispatch and this gate follows automatically;
     it cannot drift out of sync with the implementation. Case shapes (B, V, K_MAX, BV) come
     from `r4_thw_common.load_cases()`, not from a duplicated hardcoded table.
  2. Each case's path and S are printed as FIRST-CLASS output (not implicit in the timings).
  3. The harness-side gate is read-only and owned by another agent, so the deviation is
     recorded in writing in bench/hap_thw_model/README.md and cann_bench_submission/README.md
     together with the full expected Block Num table.
  4. NEW gate: the per-step Block Num SUM must equal the planned sum, and every split-parallel
     kernel's Block Num must be <= CUBE_CORE_NUM -- the RUNTIME verification of theorem P2
     (split-V must stay inside one wave). This catches an edited S_cap formula at runtime and
     is the third layer of R-SGUARD protection.

Usage: k1_split.py <csv> <label> [task]
    task = mask | fused (which r4_ops entry point was profiled). If omitted it is inferred
    from <label> ('mask' -> mask, otherwise fused, matching r4_ops.r4_fused_sampling's
    FUSE_GUMBEL=True default); the inferred value is always printed so it can be checked.
"""
import csv
import json
import os
import statistics
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_R4 = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_R4, os.path.join(_R4, "bench", "hap_thw_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Single source of truth for the dispatch (step 1 of the section 15.3.1 update).
from r4_fused_sampling import VEC_NUM, _r4_split_plan, detect_cube_core_num  # noqa: E402
from r4_thw_common import load_cases  # noqa: E402

STEPS_PER_CASE = 8
WARMUP = 3

path, label = sys.argv[1], sys.argv[2]
TASK = sys.argv[3] if len(sys.argv) > 3 else ("mask" if "mask" in label.lower() else "fused")
if TASK not in ("mask", "fused"):
    raise SystemExit(f"task must be mask|fused, got {TASK!r}")

# ---- case shapes from the model's own table (no duplicated hardcoding) ----
_task_name = "r4_mask_sample" if TASK == "mask" else "r4_fused_sampling"
CASES = {c["case_id"]: c for c in load_cases(_task_name, verbose=False)}
CUBE = detect_cube_core_num()


def plan_for(ci):
    """(rows_expected, blocks_expected, use_split, S, NBS, S_TAIL, b_num) for one bench case."""
    c = CASES[ci]
    B, V, K_MAX, BV = c["B"], c["V"], c["K_MAX"], c["BV"]
    use_split, S, NBS, S_TAIL = _r4_split_plan(B, V, K_MAX, BV)
    bn = -(-B // VEC_NUM)
    if not use_split:
        return 2, [bn, bn], use_split, S, NBS, S_TAIL, bn
    if TASK == "mask":       # K1A, K1B, K2m (K2m is NOT split -- constraint 4)
        return 3, [bn * S, bn, bn], use_split, S, NBS, S_TAIL, bn
    return 4, [bn * S, bn, bn * S, bn], use_split, S, NBS, S_TAIL, bn   # K1A, K1B, K2gA, K2gB


rows = list(csv.DictReader(open(path)))

by_step = defaultdict(list)
for r in rows:
    if r["Name"].strip() != "main_kernel":
        continue
    by_step[int(r["Step Id"].strip())].append(
        (float(r["Duration(us)"]), int(r["Block Num"]), float(r["Start Time(us)"].strip())))

recs = []
bad = 0
bad_detail = []
for step in sorted(by_step):
    seq = sorted(by_step[step], key=lambda t: t[2])      # launch order
    ci = step // STEPS_PER_CASE + 1
    rep = step % STEPS_PER_CASE
    if ci not in CASES:
        bad += 1
        bad_detail.append(f"step{step}: unknown case {ci}")
        continue
    n_exp, blk_exp, use_split, S, NBS, S_TAIL, bn = plan_for(ci)
    # STEP 4 of the update: do NOT silently drop a step whose row count differs. Report it.
    if len(seq) != n_exp:
        bad += 1
        bad_detail.append(f"step{step} case{ci}: {len(seq)} main_kernel rows, expected {n_exp} "
                          f"(path={'SPLIT' if use_split else 'CURRENT'} S={S})")
        continue
    blocks = [b for _, b, _ in seq]
    durs = [d for d, _, _ in seq]
    # K1 side = the topk-stats stage(s); K2 side = the sampling stage(s).
    if use_split:
        k1_us, k2_us = sum(durs[:2]), sum(durs[2:])
    else:
        k1_us, k2_us = durs[0], durs[1]
    blk_ok = blocks == blk_exp
    sum_ok = sum(blocks) == sum(blk_exp)
    # theorem P2 runtime check: every split-PARALLEL kernel must fit in one wave.
    par_idx = (0, 2) if (use_split and TASK == "fused") else ((0,) if use_split else ())
    p2_ok = all(blocks[i] <= CUBE for i in par_idx)
    if not (blk_ok and sum_ok and p2_ok):
        bad += 1
        bad_detail.append(f"step{step} case{ci}: blocks={blocks} expected={blk_exp} "
                          f"blk_ok={blk_ok} sum_ok={sum_ok} p2_ok={p2_ok} (cube={CUBE})")
    recs.append(dict(step=step, case=ci, rep=rep, k1_us=k1_us, k2_us=k2_us,
                     rows=len(seq), path="SPLIT" if use_split else "CURRENT",
                     S=S, NBS=NBS, S_TAIL=S_TAIL,
                     blk=blocks[0], blocks=blocks, blk_expected=blk_exp,
                     blk_ok=blk_ok, blk_sum_ok=sum_ok, blk_p2_ok=p2_ok,
                     blk_ok_all=(blk_ok and sum_ok and p2_ok)))

# median over the 5 REPEAT steps (rep >= warmup), matching the harness口径
per_case = {}
for ci in sorted(CASES):
    sel = [r for r in recs if r["case"] == ci and r["rep"] >= WARMUP]
    n_exp, blk_exp, use_split, S, NBS, S_TAIL, bn = plan_for(ci)
    if not sel:
        per_case[ci] = dict(n=0, path="SPLIT" if use_split else "CURRENT", S=S, NBS=NBS,
                            S_TAIL=S_TAIL, rows_expected=n_exp, blocks_expected=blk_exp,
                            blk_ok=False, note="no repeat steps found")
        continue
    per_case[ci] = dict(
        n=len(sel),
        k1_us=round(statistics.median(r["k1_us"] for r in sel), 3),
        k2_us=round(statistics.median(r["k2_us"] for r in sel), 3),
        total_us=round(statistics.median(r["k1_us"] + r["k2_us"] for r in sel), 3),
        path=sel[0]["path"], S=S, NBS=NBS, S_TAIL=S_TAIL,
        rows_expected=n_exp, blocks_expected=blk_exp,
        blk=sel[0]["blocks"],
        blk_ok=all(r["blk_ok_all"] for r in sel))

# ---- STEP 2 of the update: path and S are first-class output ----
n_split = sum(1 for d in per_case.values() if d["path"] == "SPLIT")
print(f"===== {label} =====  task={TASK}  cube_core_num={CUBE} (probed)")
print(f"steps_attributed={len(recs)} anomaly_rows={bad} "
      f"cases={len(per_case)} split_path={n_split} current_path={len(per_case) - n_split}")
print(f"{'case':>4} {'path':>8} {'S':>3} {'NBS':>4} {'Stail':>6} {'rows':>5} "
      f"{'BlockNum(actual)':>22} {'BlockNum(expected)':>22} {'blk_ok':>7} "
      f"{'K1_us':>10} {'K2_us':>10} {'total_us':>10} {'K1%':>6}")
for ci, d in per_case.items():
    if d["n"] == 0:
        print(f"{ci:>4} {d['path']:>8} {d['S']:>3} {d['NBS']:>4} {d['S_TAIL']:>6} "
              f"{d['rows_expected']:>5} {'-':>22} {str(d['blocks_expected']):>22} "
              f"{'False':>7}  (no repeat steps)")
        continue
    print(f"{ci:>4} {d['path']:>8} {d['S']:>3} {d['NBS']:>4} {d['S_TAIL']:>6} "
          f"{d['rows_expected']:>5} {str(d['blk']):>22} {str(d['blocks_expected']):>22} "
          f"{str(d['blk_ok']):>7} {d['k1_us']:>10.3f} {d['k2_us']:>10.3f} "
          f"{d['total_us']:>10.3f} {100 * d['k1_us'] / d['total_us']:>5.1f}%")
if bad_detail:
    print(f"--- {len(bad_detail)} anomaly detail (first 20) ---")
    for d in bad_detail[:20]:
        print("   ", d)
# theorem P2 summary: the split-parallel kernels must never exceed one wave.
p2_viol = [r["step"] for r in recs if not r["blk_p2_ok"]]
print(f"GATE theorem-P2 (split-parallel Block Num <= cube_core_num={CUBE}): "
      f"{'PASS' if not p2_viol else f'FAIL steps={p2_viol[:10]}'}")
print(f"GATE Block Num sequence == _r4_split_plan-derived expectation: "
      f"{'PASS' if bad == 0 else f'FAIL ({bad} anomalous steps)'}")

_outdir = "/tmp/opencode/r4_item4"
os.makedirs(_outdir, exist_ok=True)
json.dump(dict(label=label, task=TASK, cube_core_num=CUBE, anomaly_rows=bad,
               anomaly_detail=bad_detail, per_case=per_case,
               p2_violations=p2_viol, gate_pass=(bad == 0 and not p2_viol)),
          open(f"{_outdir}/k1split_{label}.json", "w"), indent=1)
sys.exit(0 if (bad == 0 and not p2_viol) else 1)
