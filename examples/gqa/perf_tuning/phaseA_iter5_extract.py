"""Iter5 Phase A — C2 re-estimation data extraction from EXISTING iter3 msprof data.

PipeUtilization.csv row layout (verified): per block_id one `cube0` row
(aic_* fields + C-side wait ids) and two `vector*` rows (aiv_* fields +
V-side wait ids).  Extracts per-sample:
  * TaskDur + frequency (OpBasicInfo)
  * HOT-AIC block (max aic_time): cube/mte2/fix/scalar busy + C wait ids
  * HOT-AIV row (max aiv_time): vec/mte2/mte3/scalar busy + V wait ids
  * max-across-rows wait columns (iter3 reporting口径)
Groups: R4 g2stack-ns2 (hot core 26 tasks = 52 kv), R6 flat kv==1 (hot core
5 tasks = 5 u), R6 g2stack ns=1 forced (5 tasks).
Derived: exposure decomposition + flat noncausal residual anchor for the
C2 re-estimation (see c2_design_iter5.md Phase A).  Read-only, no NPU runs.

Run: python3 examples/gqa/perf_tuning/phaseA_iter5_extract.py
"""

import csv
import glob
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))

C_WAIT = {
    "C_id1": "aic_scalar_wait_id1_time(us)",
    "C_id2": "aic_scalar_wait_id2_time(us)",
    "C_id5": "aic_scalar_wait_id5_time(us)",
}
V_WAIT = {
    "V_id0": "aiv_scalar_wait_id0_time(us)",
    "V_id3": "aiv_scalar_wait_id3_time(us)",
    "V_id4": "aiv_scalar_wait_id4_time(us)",
}


def find_csv(base, name):
    hits = sorted(glob.glob(os.path.join(base, "OPPROF_*", name)))
    return hits[-1] if hits else None


def fnum(r, k):
    v = r.get(k, "NA")
    return None if v in ("NA", "", None) else float(v)


def parse(base):
    ob = find_csv(base, "OpBasicInfo.csv")
    pu = find_csv(base, "PipeUtilization.csv")
    if ob is None or pu is None:
        return None
    with open(ob) as f:
        rows = list(csv.DictReader(f))
    main = [r for r in rows if r["Op Name"] == "main_kernel"][0]
    dur = float(main["Task Duration(us)"])
    freq = main.get("AICore Frequency(MHz)", main.get("Aicore Frequency(MHz)", "NA"))
    with open(pu) as f:
        prows = list(csv.DictReader(f))

    aic_rows = [r for r in prows if r.get("sub_block_id", "").startswith("cube") and fnum(r, "aic_time(us)") is not None]
    aiv_rows = [r for r in prows if r.get("sub_block_id", "").startswith("vector") and fnum(r, "aiv_time(us)") is not None]
    # sanity filter vs TaskDur (iter4 data-quality lesson)
    bad = [r for r in aic_rows + aiv_rows if max(fnum(r, "aic_time(us)") or 0, fnum(r, "aiv_time(us)") or 0) > 1.5 * dur]
    aic_rows = [r for r in aic_rows if r not in bad]
    aiv_rows = [r for r in aiv_rows if r not in bad]

    hot_c = max(aic_rows, key=lambda r: fnum(r, "aic_time(us)"))
    hot_v = max(aiv_rows, key=lambda r: fnum(r, "aiv_time(us)"))

    out = {
        "dir": os.path.basename(base),
        "dur": dur,
        "freq": freq,
        "dropped": len(bad),
        "hot_aic": fnum(hot_c, "aic_time(us)"),
        "hot_cube": fnum(hot_c, "aic_cube_time(us)"),
        "hot_mte1": fnum(hot_c, "aic_mte1_time(us)"),
        "hot_mte2": fnum(hot_c, "aic_mte2_time(us)"),
        "hot_fix": fnum(hot_c, "aic_fixpipe_time(us)"),
        "hot_aic_scl": fnum(hot_c, "aic_scalar_time(us)"),
        "hot_aiv": fnum(hot_v, "aiv_time(us)"),
        "hot_vec": fnum(hot_v, "aiv_vec_time(us)"),
        "hot_aiv_mte2": fnum(hot_v, "aiv_mte2_time(us)"),
        "hot_aiv_mte3": fnum(hot_v, "aiv_mte3_time(us)"),
        "hot_aiv_scl": fnum(hot_v, "aiv_scalar_time(us)"),
        "hot_wait": {k: fnum(hot_c, c) for k, c in C_WAIT.items()},
    }
    out["hot_wait"].update({k: fnum(hot_v, c) for k, c in V_WAIT.items()})
    out["max_wait"] = {}
    for k, c in list(C_WAIT.items()) + list(V_WAIT.items()):
        vals = [fnum(r, c) for r in prows if fnum(r, c) is not None]
        out["max_wait"][k] = max(vals) if vals else None
    # vec-busy spread across all vector rows (uniformity check)
    vecs = [fnum(r, "aiv_vec_time(us)") for r in aiv_rows if fnum(r, "aiv_vec_time(us)") is not None]
    out["vec_minmax"] = (min(vecs), max(vecs)) if vecs else None
    return out


KEYS = ["dur", "hot_aic", "hot_cube", "hot_mte1", "hot_mte2", "hot_fix", "hot_aic_scl",
        "hot_aiv", "hot_vec", "hot_aiv_mte2", "hot_aiv_mte3", "hot_aiv_scl"]


def show(tag, dirs, per, per_task=None):
    print(f"\n=== {tag} (units/core={per}) ===")
    accs = {}
    for d in dirs:
        o = parse(os.path.join(HERE, d))
        if o is None:
            print(f"  {d}: NO DATA")
            continue
        w = o["hot_wait"]
        print(f"  {d}: TaskDur={o['dur']:.2f} freq={o['freq']} dropped={o['dropped']} vec_range={o['vec_minmax'][0]:.1f}-{o['vec_minmax'][1]:.1f}")
        CKEYS = ["hot_aic", "hot_cube", "hot_mte1", "hot_mte2", "hot_fix", "hot_aic_scl"]
        VKEYS = ["hot_aiv", "hot_vec", "hot_aiv_mte2", "hot_aiv_mte3", "hot_aiv_scl"]
        print("      hot C: " + " ".join(f"{k.replace('hot_','')}={o[k]:.2f}" for k in CKEYS if o[k] is not None))
        print("      hot V: " + " ".join(f"{k.replace('hot_','')}={o[k]:.2f}" for k in VKEYS if o[k] is not None))
        print("      hot wait: " + " ".join(f"{k}={v:.2f}" for k, v in w.items() if v is not None))
        print("      max wait: " + " ".join(f"{k}={v:.2f}" for k, v in o["max_wait"].items() if v is not None))
        for k in KEYS:
            if o[k] is not None:
                accs.setdefault(k, []).append(o[k])
        for k, v in w.items():
            if v is not None:
                accs.setdefault("w_" + k, []).append(v)
    if not accs:
        return
    print("  --- mean over samples (and per-unit) ---")
    for k, v in accs.items():
        m = statistics.mean(v)
        print(f"    {k:14s} = {m:8.3f}   /unit = {m/per:6.3f}" + (f"   /task = {m/per_task:6.3f}" if per_task else ""))


def main():
    show("R4 g2stack ns=2 (c4-like B4 S256 Skv256 Nq64 Nkv8 bf16 noncausal)",
         ["msprof_op_iter3_r4_s1", "msprof_op_iter3_r4_s2", "msprof_op_iter3_r4_s3"], 52, per_task=26)
    show("R6 flat kv==1 (c2-like B2 S128 Nq48 Nkv4 fp16 noncausal)",
         ["msprof_op_iter3_r6_flat_a1", "msprof_op_iter3_r6_flat_a2", "msprof_op_iter3_r6_flat_a3"], 5)
    show("R6 g2stack ns=1 forced (same shape)",
         ["msprof_op_iter3_r6_g2s_b1", "msprof_op_iter3_r6_g2s_b2", "msprof_op_iter3_r6_g2s_b3"], 5)


if __name__ == "__main__":
    main()
