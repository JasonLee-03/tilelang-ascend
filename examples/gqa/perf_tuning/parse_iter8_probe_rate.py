"""Iter8 P2 probe-rate parser — contiguous (bhsd) vs strided (bsnd4d) FIN box
stream.  Reads the probe dirs' OpBasicInfo/PipeUtilization/Memory CSVs and
reports, per shape/arm/sample:

  TaskDur (kernel wall), aiv_time (hot vector window), aiv_mte3_time /
  aiv_mte3_ratio / aiv_mte3_instructions / aiv_mte3_active_bw, UB_to_GM KB,
  and the derived PER-BOX MTE3 cost = aiv_mte3_time / boxes(block0,vid0).

boxes(block0,vid0) = tasks_on_core0 * 2 h * rep, tasks_on_core0 = q_tasks +
(1 if 0 < r_tasks else 0)  (the flat task_range decomposition, gqa.py:3046).

The Phase-A cost model uses the DELTA of per-box MTE3 time (strided minus
contiguous) times the real kernel's boxes-per-vid-per-kernel.

Run: python3 examples/gqa/perf_tuning/parse_iter8_probe_rate.py
"""

import csv
import glob
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

SHAPES = {  # name: (B, S, Nq, Nkv, D, rep)
    "r2": (2, 128, 32, 8, 128, 8),
    "r6": (2, 128, 48, 4, 128, 8),
    "r3": (2, 128, 128, 8, 128, 4),
    "r4": (4, 256, 64, 8, 128, 4),
}
ARMS = ["bhsd", "bsnd4d"]
CORES = 20


def _f(v):
    try:
        if v in ("NA", "", None):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def tasks_core0(B, S, Nq, Nkv):
    G = Nq // Nkv
    n_tasks = B * Nkv * (G // 2) * (S // 64)
    q, r = n_tasks // CORES, n_tasks % CORES
    return q + (1 if 0 < r else 0), n_tasks


def prof_dir(shape, arm, n):
    d = os.path.join(HERE, f"msprof_op_iter8_PR_{shape}_{arm}_s{n}")
    p = sorted(glob.glob(os.path.join(d, "OPPROF_*")))
    return p[-1] if p else None


def read_one(pd):
    r = {}
    with open(os.path.join(pd, "OpBasicInfo.csv")) as f:
        row = list(csv.DictReader(f))[0]
    r["dur_us"] = float(row["Task Duration(us)"])
    r["blk"] = int(row["Block Dim"])
    r["freq"] = f"{row['Current Freq']}/{row['Rated Freq']}"
    with open(os.path.join(pd, "PipeUtilization.csv")) as f:
        prows = list(csv.DictReader(f))
    v0 = [p for p in prows if p["block_id"] == "0" and p["sub_block_id"] == "vector0"]
    allv = [p for p in prows if p["sub_block_id"].startswith("vector")]
    if not v0:
        return None
    v = v0[0]
    r["aiv_time_us"] = _f(v["aiv_time(us)"])
    r["mte3_time_us"] = _f(v["aiv_mte3_time(us)"])
    r["mte3_ratio"] = _f(v["aiv_mte3_ratio"])
    r["mte3_instr"] = None  # instructions live in Memory.csv (filled below)
    r["mte3_bw"] = _f(v["aiv_mte3_active_bw(GB/s)"])
    r["vec_time_us"] = _f(v["aiv_vec_time(us)"])
    r["scalar_wait_us"] = _f(v["aiv_scalar_wait_time(us)"])
    # hottest vector row across blocks (wall-relevant core)
    mt = [_f(p["aiv_mte3_time(us)"]) for p in allv if _f(p["aiv_mte3_time(us)"]) is not None]
    r["mte3_time_max_us"] = max(mt) if mt else None
    with open(os.path.join(pd, "Memory.csv")) as f:
        mrows = list(csv.DictReader(f))
    m0 = [p for p in mrows if p["block_id"] == "0" and p["sub_block_id"] == "vector0"]
    r["mte3_instr"] = _f(m0[0]["aiv_mte3_instructions"]) if m0 else None
    r["mte3_instr_max"] = max([_f(p["aiv_mte3_instructions"]) for p in mrows
                               if _f(p.get("aiv_mte3_instructions")) is not None] or [0])
    r["ub_to_gm_kb"] = sum(v for v in (_f(p.get("UB_to_GM_datas(KB)")) for p in mrows) if v)
    return r


def main():
    out = {}
    print("=" * 128)
    hdr = (f"{'shape':5} {'arm':7} {'s':>2} {'TaskDur':>8} {'aiv_t':>7} {'mte3_t(b0)':>11} "
           f"{'mte3_t(max)':>12} {'mte3_rat':>8} {'instr':>6} {'bw GB/s':>8} {'UB2GM KB':>9} "
           f"{'ns/box(b0)':>11} {'freq':>9}")
    print(hdr)
    print("-" * 128)
    for shp, (B, S, Nq, Nkv, D, rep) in SHAPES.items():
        t0, ntasks = tasks_core0(B, S, Nq, Nkv)
        boxes_b0 = t0 * 2 * rep
        for arm in ARMS:
            for n in (1, 2):
                pd = prof_dir(shp, arm, n)
                if pd is None:
                    print(f"{shp:5} {arm:7} {n:>2}  MISSING")
                    continue
                r = read_one(pd)
                if r is None:
                    print(f"{shp:5} {arm:7} {n:>2}  NO vector0 row")
                    continue
                r["boxes_b0"] = boxes_b0
                r["ns_per_box"] = r["mte3_time_us"] / boxes_b0 * 1000
                out.setdefault((shp, arm), []).append(r)
                print(f"{shp:5} {arm:7} {n:>2} {r['dur_us']:>8.2f} {r['aiv_time_us']:>7.2f} "
                      f"{r['mte3_time_us']:>11.3f} {r['mte3_time_max_us']:>12.3f} "
                      f"{r['mte3_ratio']:>8.3f} {int(r['mte3_instr'] or 0):>6} {r['mte3_bw']:>8.1f} "
                      f"{r['ub_to_gm_kb']:>9.0f} {r['ns_per_box']:>11.1f} {r['freq']:>9}")
    print("=" * 128)
    print(f"ntasks/boxes_b0 per shape: " +
          ", ".join(f"{s}: ntasks={tasks_core0(*SHAPES[s][:4])[1]} "
                    f"boxes_b0(rep)={tasks_core0(*SHAPES[s][:4])[0]*2*SHAPES[s][5]}" for s in SHAPES))
    print()
    print(f"{'shape':6} | {'dur bhsd':>9} {'dur bsnd':>9} {'Δdur us':>8} {'Δdur %':>7} | "
          f"{'ns/box bhsd':>12} {'ns/box bsnd':>12} {'Δns/box':>9} {'ratio':>6} | "
          f"{'mte3_t bhsd':>12} {'mte3_t bsnd':>12} | {'Δscalar_wait':>13}")
    print("-" * 128)
    model = {}
    for shp, (B, S, Nq, Nkv, D, rep) in SHAPES.items():
        if (shp, "bhsd") not in out or (shp, "bsnd4d") not in out:
            continue
        b = {k: statistics.mean([r[k] for r in out[(shp, "bhsd")]]) for k in
             ("dur_us", "mte3_time_us", "ns_per_box", "mte3_bw", "scalar_wait_us", "aiv_time_us")}
        s = {k: statistics.mean([r[k] for r in out[(shp, "bsnd4d")]]) for k in b}
        t0, ntasks = tasks_core0(B, S, Nq, Nkv)
        boxes_real = t0 * 2  # rep=1 == the real kernel's Output box stream per vid
        pred = (s["ns_per_box"] - b["ns_per_box"]) * boxes_real / 1000.0
        model[shp] = dict(pred_delta_us=pred, boxes_real=boxes_real,
                          ns_box_bhsd=b["ns_per_box"], ns_box_bsnd=s["ns_per_box"],
                          dur_bhsd=b["dur_us"], dur_bsnd=s["dur_us"])
        print(f"{shp:6} | {b['dur_us']:>9.2f} {s['dur_us']:>9.2f} {s['dur_us']-b['dur_us']:>+8.2f} "
              f"{100*(s['dur_us']-b['dur_us'])/b['dur_us']:>+6.1f}% | {b['ns_per_box']:>12.1f} "
              f"{s['ns_per_box']:>12.1f} {s['ns_per_box']-b['ns_per_box']:>+9.1f} "
              f"{s['ns_per_box']/b['ns_per_box']:>6.2f} | {b['mte3_time_us']:>12.3f} "
              f"{s['mte3_time_us']:>12.3f} | {s['scalar_wait_us']-b['scalar_wait_us']:>+13.3f}")
    print()
    print("PHASE-A COST MODEL (per-vid MTE3 time delta scaled to the real kernel's box stream):")
    for shp, m in model.items():
        print(f"  {shp}: boxes/vid/kernel={m['boxes_real']}  ns/box bhsd={m['ns_box_bhsd']:.1f} "
              f"bsnd={m['ns_box_bsnd']:.1f}  -> predicted main-kernel MTE3 increment "
              f"{m['pred_delta_us']:+.3f} us (probe wall delta {m['dur_bsnd']-m['dur_bhsd']:+.2f} us "
              f"at rep={SHAPES[shp][5]})")
    import json
    with open(os.path.join(HERE, "probe_iter8_rate_summary.json"), "w") as f:
        json.dump({k: [{kk: vv for kk, vv in r.items()} for r in v] for k, v in
                   {(f"{a}_{b}"): v for (a, b), v in out.items()}.items()}, f, indent=1, default=str)
    print(f"\nwrote {os.path.join(HERE, 'probe_iter8_rate_summary.json')}")


if __name__ == "__main__":
    main()
