"""Fine-grained per-kernel split of an msprof kernel_details.csv (round 3).

round2_item4/k1_split.py sums the K1-side and K2-side rows of a step. This script keeps EVERY row
separate and names it, which is what round 3 needs for three things the coarser split cannot do:

  1. a_launch back-out (task item 2 / U2 / V-S3). rev3's K2gB `r4_gumbel_sample_splitB(B, S)` has
     UB=132 B and does almost nothing, so its msprof Duration is close to a pure per-launch floor.
     K1A/K1B/K2gA durations feed the section 12.4 piecewise model so a_launch can also be FITTED
     rather than only bounded.
  2. The runtime `Block Num` table (task item 4 / U4 / V-S7), including theorem P2
     (every split-PARALLEL kernel must stay inside one wave: Block Num <= cube_core_num) and the
     case-7 fused expectation [20, 2, 20, 2].
  3. Verifying that the 12 NON-split cases launch exactly the same kernel sequence with the same
     Block Num in both arms -- the structural half of the "must be exactly 1.000x" requirement.

The expected row count / Block Num sequence is DERIVED from the delivered host dispatch
`_r4_split_plan` (single source of truth, per DESIGN.md section 15.3.1 step 1); the pre-rev3 BASE
arm has no split API, so every case there is planned as CURRENT.

Also reports the inter-kernel GAP (next Start Time - previous end). The cann-bench `elapsed_us`
口径 is the SUM of per-step kernel Durations, so gaps are NOT part of any performance number in
this round; they are reported only to characterise host launch latency separately.

Usage: ksplit_fine.py <csv> <label> [mask|fused] [out.json]
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

import r4_fused_sampling as M  # noqa: E402
from r4_thw_common import load_cases  # noqa: E402

STEPS_PER_CASE = 8
WARMUP = 3

path, label = sys.argv[1], sys.argv[2]
TASK = sys.argv[3] if len(sys.argv) > 3 else ("mask" if "mask" in label.lower() else "fused")
out_path = sys.argv[4] if len(sys.argv) > 4 else os.path.join(_HERE, f"ksplit_{label}_{TASK}.json")
HAS_SPLIT = hasattr(M, "_r4_split_plan")
# When analysing a BASE-arm CSV the delivered file in place may be the CANDIDATE (whose dispatch
# would wrongly expect 3/4 rows). R4_FORCE_CURRENT=1 pins the plan to the pre-rev3 2-launch path so
# the expectation always matches the arm that produced the CSV.
if os.environ.get("R4_FORCE_CURRENT") == "1":
    HAS_SPLIT = False
# ---- item-A addition (the ONLY change vs round3_splitv_perf/ksplit_fine.py) ----
# In item A BOTH arms are rev3 code with the split API; they differ ONLY in _SPLIT_A_LAUNCH_US
# (base 8.0 vs cand 1.540). ab_all_itemA.sh restores the CANDIDATE at the end, so the in-place file
# carries the RELAXED guard while a base-arm CSV is analysed -- and the relaxed plan would wrongly
# expect 4 launches for cases 6/11 where the base arm actually emitted 2. R4_FORCE_CURRENT=1 is not
# usable here either: it would pin ALL 18 cases to CURRENT, which is wrong for the 6 cases that
# split at 8.0 as well. So pin the constant to the value the arm under analysis really ran with.
if os.environ.get("R4_FORCE_A_LAUNCH"):
    M._SPLIT_A_LAUNCH_US = float(os.environ["R4_FORCE_A_LAUNCH"])
    print(f"[ksplit] R4_FORCE_A_LAUNCH -> _SPLIT_A_LAUNCH_US pinned to "
          f"{M._SPLIT_A_LAUNCH_US} for expectation derivation")
CUBE = M.detect_cube_core_num() if hasattr(M, "detect_cube_core_num") \
    else M.detect_vec_core_num() // M.VEC_NUM
VEC_NUM = M.VEC_NUM

CASES = {c["case_id"]: c for c in
         load_cases("r4_mask_sample" if TASK == "mask" else "r4_fused_sampling", verbose=False)}


def names_for(ci):
    """Kernel names in launch order for one bench case, derived from the host dispatch."""
    c = CASES[ci]
    B, V, K, BV = c["B"], c["V"], c["K_MAX"], c["BV"]
    plan = M._r4_split_plan(B, V, K, BV) if HAS_SPLIT else (False, 1, 1, -1)
    use_split, S, NBS, S_TAIL = plan
    bn = -(-B // VEC_NUM)
    if not use_split:
        return ["K1", "K2g" if TASK == "fused" else "K2m"], [bn, bn], plan, bn
    if TASK == "mask":          # K2m is NOT split (DESIGN.md constraint 4 / section 13.5)
        return ["K1A", "K1B", "K2m"], [bn * S, bn, bn], plan, bn
    return ["K1A", "K1B", "K2gA", "K2gB"], [bn * S, bn, bn * S, bn], plan, bn


rows = list(csv.DictReader(open(path)))
by_step = defaultdict(list)
for r in rows:
    if r["Name"].strip() != "main_kernel":
        continue
    by_step[int(r["Step Id"].strip())].append(
        dict(dur=float(r["Duration(us)"]), blk=int(r["Block Num"]),
             start=float(r["Start Time(us)"].strip())))

recs, bad, bad_detail = [], 0, []
for step in sorted(by_step):
    seq = sorted(by_step[step], key=lambda d: d["start"])
    ci = step // STEPS_PER_CASE + 1
    rep = step % STEPS_PER_CASE
    if ci not in CASES:
        bad += 1
        bad_detail.append(f"step{step}: unknown case {ci}")
        continue
    nm, blk_exp, plan, bn = names_for(ci)
    if len(seq) != len(nm):
        bad += 1
        bad_detail.append(f"step{step} case{ci}: {len(seq)} main_kernel rows, expected {len(nm)} "
                          f"({nm}) plan={plan}")
        continue
    blocks = [d["blk"] for d in seq]
    par_idx = tuple(i for i, n in enumerate(nm) if n.endswith("A"))   # split-PARALLEL kernels
    ok_blk = blocks == blk_exp
    ok_sum = sum(blocks) == sum(blk_exp)
    ok_p2 = all(blocks[i] <= CUBE for i in par_idx)
    if not (ok_blk and ok_sum and ok_p2):
        bad += 1
        bad_detail.append(f"step{step} case{ci}: blocks={blocks} expected={blk_exp} "
                          f"blk_ok={ok_blk} sum_ok={ok_sum} p2_ok={ok_p2} cube={CUBE}")
    gaps = [seq[i + 1]["start"] - (seq[i]["start"] + seq[i]["dur"]) for i in range(len(seq) - 1)]
    recs.append(dict(step=step, case=ci, rep=rep, names=nm, plan=plan, b_num=bn,
                     blocks=blocks, blocks_expected=blk_exp, gaps_us=gaps,
                     ok_blk=ok_blk, ok_sum=ok_sum, ok_p2=ok_p2,
                     per={n: seq[i]["dur"] for i, n in enumerate(nm)},
                     total=sum(d["dur"] for d in seq)))

per_case = {}
for ci in sorted(CASES):
    sel = [r for r in recs if r["case"] == ci and r["rep"] >= WARMUP]
    nm, blk_exp, plan, bn = names_for(ci)
    if not sel:
        per_case[ci] = dict(n=0, names=nm, note="no repeat steps")
        continue
    use_split, S, NBS, S_TAIL = plan
    per = {n: round(statistics.median(r["per"][n] for r in sel), 3) for n in nm}
    gaps = ([round(statistics.median(r["gaps_us"][i] for r in sel), 3) for i in range(len(nm) - 1)]
            if len(nm) > 1 else [])
    per_case[ci] = dict(
        n=len(sel), path="SPLIT" if use_split else "CURRENT", S=S, NBS=NBS, S_TAIL=S_TAIL,
        B=CASES[ci]["B"], V=CASES[ci]["V"], K_MAX=CASES[ci]["K_MAX"], BV=CASES[ci]["BV"],
        b_num=bn, names=nm, per_kernel_us=per,
        total_us=round(statistics.median(r["total"] for r in sel), 3),
        blocks=sel[0]["blocks"], blocks_expected=blk_exp,
        gaps_us=gaps, gap_sum_us=round(sum(gaps), 3),
        ok_blk=all(r["ok_blk"] for r in sel), ok_sum=all(r["ok_sum"] for r in sel),
        ok_p2=all(r["ok_p2"] for r in sel),
        n_launch=len(nm))

print(f"===== {label} =====  task={TASK}  HAS_SPLIT_API={HAS_SPLIT}  cube_core_num={CUBE}")
print(f"steps_attributed={len(recs)} anomaly_rows={bad} cases={len(per_case)} "
      f"split={sum(1 for d in per_case.values() if d.get('path') == 'SPLIT')} "
      f"current={sum(1 for d in per_case.values() if d.get('path') == 'CURRENT')}")
hdr = (f"{'case':>4} {'path':>7} {'S':>3} {'NBS':>4} {'#ln':>3} {'BlockNum(actual)':>20} "
       f"{'BlockNum(exp)':>20} {'ok':>5} | per-kernel Duration(us) | {'total':>9} {'gaps':>9}")
print(hdr)
for ci, d in per_case.items():
    if d["n"] == 0:
        print(f"{ci:>4} {'?':>7} (no repeat steps)")
        continue
    pk = "  ".join(f"{n}={d['per_kernel_us'][n]:.3f}" for n in d["names"])
    ok = d["ok_blk"] and d["ok_sum"] and d["ok_p2"]
    print(f"{ci:>4} {d['path']:>7} {d['S']:>3} {d['NBS']:>4} {d['n_launch']:>3} "
          f"{str(d['blocks']):>20} {str(d['blocks_expected']):>20} {str(ok):>5} | {pk} | "
          f"{d['total_us']:>9.3f} {d['gap_sum_us']:>9.3f}")

p2v = [r["step"] for r in recs if not r["ok_p2"]]
print(f"\nGATE theorem-P2 (split-parallel Block Num <= cube_core_num={CUBE}): "
      f"{'PASS' if not p2v else f'FAIL steps={p2v[:10]}'}")
print(f"GATE Block Num sequence == _r4_split_plan-derived expectation: "
      f"{'PASS' if bad == 0 else f'FAIL ({bad} anomalous steps)'}")
if bad_detail:
    print(f"--- {len(bad_detail)} anomaly detail (first 20) ---")
    for x in bad_detail[:20]:
        print("   ", x)

# ---- launch-floor statistics: the smallest kernels observed ----
floors = defaultdict(list)
for r in recs:
    if r["rep"] < WARMUP:
        continue
    for n, v in r["per"].items():
        floors[n].append(v)
print(f"\n--- per-kernel-name Duration statistics over repeat steps (us) ---")
print(f"{'kernel':>6} {'n':>5} {'min':>9} {'median':>9} {'max':>9}")
for n in ("K1", "K2g", "K2m", "K1A", "K1B", "K2gA", "K2gB"):
    if n in floors:
        v = floors[n]
        print(f"{n:>6} {len(v):>5} {min(v):>9.3f} {statistics.median(v):>9.3f} {max(v):>9.3f}")

json.dump(dict(label=label, task=TASK, csv=path, has_split_api=HAS_SPLIT, cube_core_num=CUBE,
               vec_num=VEC_NUM, anomaly_rows=bad, anomaly_detail=bad_detail,
               per_case={str(k): v for k, v in per_case.items()},
               p2_violations=p2v, gate_pass=(bad == 0 and not p2v),
               kernel_name_stats={n: dict(n=len(v), min=min(v),
                                          median=statistics.median(v), max=max(v))
                                  for n, v in floors.items()}),
          open(out_path, "w"), indent=1)
print(f"\nwrote {out_path}")
sys.exit(0 if (bad == 0 and not p2v) else 1)
