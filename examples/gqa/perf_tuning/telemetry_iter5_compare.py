"""Iter5 Gate-2 telemetry comparison: BASE (g2stack ns2) vs C2 (flat kv2).

Per group (arm x shape, 5 samples): TaskDur mean, hot-core V window, V vec
busy, exposure (window - vec), wait-id columns (per-sample max across rows,
then group mean) + per-kv normalization.  Mechanism questions:
  1. did the batch-alternation exposure (base: C-id2 53% / V-id4 49% of wall)
     disappear in the C2 arm?
  2. did the waits migrate to the pacing/release side (id1/id3 — the kv1-flat
     signature), i.e. is the kernel now pacing-bound like the proven flat?
Read-only over msprof_op_iter5_* dirs.

Run: python3 examples/gqa/perf_tuning/telemetry_iter5_compare.py
"""

import csv
import glob
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))

WAIT = {
    "C_id1": "aic_scalar_wait_id1_time(us)",
    "C_id2": "aic_scalar_wait_id2_time(us)",
    "C_id5": "aic_scalar_wait_id5_time(us)",
    "V_id0": "aiv_scalar_wait_id0_time(us)",
    "V_id3": "aiv_scalar_wait_id3_time(us)",
    "V_id4": "aiv_scalar_wait_id4_time(us)",
}


def fnum(r, k):
    v = r.get(k, "NA")
    return None if v in ("NA", "", None) else float(v)


def parse_one(base):
    ob = sorted(glob.glob(os.path.join(base, "OPPROF_*", "OpBasicInfo.csv")))
    pu = sorted(glob.glob(os.path.join(base, "OPPROF_*", "PipeUtilization.csv")))
    if not ob or not pu:
        return None
    with open(ob[-1]) as f:
        main = [r for r in csv.DictReader(f) if r["Op Name"] == "main_kernel"][0]
    dur = float(main["Task Duration(us)"])
    with open(pu[-1]) as f:
        prows = list(csv.DictReader(f))
    aiv = [r for r in prows if r.get("sub_block_id", "").startswith("vector") and fnum(r, "aiv_time(us)")]
    aic = [r for r in prows if r.get("sub_block_id", "").startswith("cube") and fnum(r, "aic_time(us)")]
    aiv = [r for r in aiv if fnum(r, "aiv_time(us)") <= 1.5 * dur]  # sanity filter (iter4 lesson)
    aic = [r for r in aic if fnum(r, "aic_time(us)") <= 1.5 * dur]
    out = {"dur": dur}
    if aiv:
        hotv = max(aiv, key=lambda r: fnum(r, "aiv_time(us)"))
        out["hot_aiv"] = fnum(hotv, "aiv_time(us)")
        out["hot_vec"] = fnum(hotv, "aiv_vec_time(us)")
    if aic:
        hotc = max(aic, key=lambda r: fnum(r, "aic_time(us)"))
        out["hot_aic"] = fnum(hotc, "aic_time(us)")
        out["hot_cube"] = fnum(hotc, "aic_cube_time(us)")
        out["hot_mte2"] = fnum(hotc, "aic_mte2_time(us)")
    else:
        print(f"  NOTE {os.path.basename(base)}: all cube rows failed the TaskDur sanity "
              "filter (profiler artifact, iter4 lesson) — C-side telemetry skipped for this "
              "sample; TaskDur (adoption metric, OpBasicInfo) unaffected")
    for k, col in WAIT.items():
        src = aic if k.startswith("C") else aiv
        vals = [fnum(r, col) for r in src if fnum(r, col) is not None]
        out[k] = max(vals) if vals else None
    return out


def group(tag, dirs, per_kv):
    accs = {}
    for d in dirs:
        o = parse_one(os.path.join(HERE, d))
        if o is None:
            print(f"  MISSING {d}")
            continue
        for k, v in o.items():
            if v is not None:
                accs.setdefault(k, []).append(v)
    if not accs:
        return None
    m = {k: statistics.mean(v) for k, v in accs.items()}
    print(f"\n=== {tag} (n={len(accs['dur'])}, per hot-core kv={per_kv}) ===")
    print(f"  TaskDur  = {m['dur']:8.2f} us  [{min(accs['dur']):.2f}-{max(accs['dur']):.2f}]  /kv = {m['dur']/per_kv:.3f}")
    if 'hot_aiv' in m and 'hot_vec' in m:
        print(f"  hot_aiv  = {m['hot_aiv']:8.2f}    hot_vec = {m['hot_vec']:7.2f} (/kv {m['hot_vec']/per_kv:.3f})   exposure = {m['hot_aiv']-m['hot_vec']:7.2f} (/kv {(m['hot_aiv']-m['hot_vec'])/per_kv:.3f})")
    if 'hot_aic' in m:
        print(f"  hot_aic  = {m['hot_aic']:8.2f}    hot_cube = {m['hot_cube']:6.2f}   hot_mte2 = {m['hot_mte2']:6.2f}")
    for k in WAIT:
        v = m.get(k)
        if v is not None:
            print(f"  {k:6s} = {v:8.2f} us  (/kv {v/per_kv:.3f}, {v/m['dur']*100:5.1f}% of wall)")
    return m


def main():
    shapes = {"r4": 52, "c7": 8}  # hot-core kv units (26 tasks x2 / 4 tasks x2)
    res = {}
    for shape, per_kv in shapes.items():
        for arm in ("base", "c2"):
            dirs = sorted(glob.glob(os.path.join(HERE, f"msprof_op_iter5_{shape}_{arm}_s*")))
            dirs = [os.path.basename(d) for d in dirs]
            res[(shape, arm)] = group(f"{shape} {arm}", dirs, per_kv)
    print("\n=== deltas (c2 - base) ===")
    for shape in shapes:
        b, c = res[(shape, "base")], res[(shape, "c2")]
        if b and c:
            exp = ""
            if all(k in b for k in ("hot_aiv", "hot_vec")) and all(k in c for k in ("hot_aiv", "hot_vec")):
                exp = f"  exposure/kv {(b['hot_aiv']-b['hot_vec'])/shapes[shape]:.3f} -> {(c['hot_aiv']-c['hot_vec'])/shapes[shape]:.3f}"
            print(f"  {shape}: TaskDur {b['dur']:.2f} -> {c['dur']:.2f} ({(c['dur']-b['dur'])/b['dur']*100:+.2f}%){exp}")
            for k in WAIT:
                if b.get(k) is not None and c.get(k) is not None:
                    print(f"      {k:6s} {b[k]:8.2f} -> {c[k]:8.2f}  ({c[k]-b[k]:+8.2f})")


if __name__ == "__main__":
    main()
