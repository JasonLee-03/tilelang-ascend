"""Iter10 Gate-2 A/B aggregator — P4 (transpose_qkv elimination, flat kv1/kv2
BSND strided direct reads).

Reuses parse_msprof_iter6.find_csvs/parse_kernel (impossible-row artifact
filter included).  Layout: msprof_op_iter10_{SEG}_{shape}_{arm}_s{n}/OPPROF_*/
{kernel}/{i}/*.csv with instances in launch order:
  BASE covered (r2/r4/r6/r3/c7): kernel_kernel:0 = transpose_qkv,
      main_kernel:0 = main (tout already eliminated by wave-1 out_bsnd).
  P4 covered: main_kernel:0 ONLY — tqkv ABSENCE is the elimination evidence.
  Guards: r1 = kernel_kernel:0 (tqkv) + main + kernel_kernel:1 (tout);
      c19/c8 = tqkv + main; r5 = main only.
e2e = SUM of TaskDur over all instances in the run.

Verdict lines evaluate the PREREGISTERED criteria in candidate_iter10_P4.json
(locked 2026-10-20T01:05) — they do NOT define them.

Run: python3 examples/gqa/perf_tuning/parse_iter10_ab.py
"""

import glob
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parse_msprof_iter6 import find_csvs, parse_kernel  # noqa: E402

SHAPES = ["r2", "r4", "r6", "r3", "c7", "r1", "r5", "c19", "c8"]
COVERED = ["r2", "r4", "r6", "r3", "c7"]
PRIMARY = ["r2", "r4"]                 # one per first-wave kernel family
GUARDS = ["r1", "r5", "c19", "c8"]
ARMS = ("base", "p4")
SAMPLES = {s: (1, 2, 3) for s in SHAPES}
for s in ("r5", "c19", "c8"):
    SAMPLES[s] = (1, 2)
WAITID = ["aic_scalar_wait_id1_time(us)", "aic_scalar_wait_id2_time(us)",
          "aic_scalar_wait_id5_time(us)", "aiv_scalar_wait_id0_time(us)",
          "aiv_scalar_wait_id3_time(us)", "aiv_scalar_wait_id4_time(us)"]
# Session-health anchors (shipped e2e, most recent same-baseline sessions):
# iter8 Gate2 (r2/r6/r3/r4/c7), iter9 Gate2/phaseA (r1/r5/c19/c8).
# Cross-session reference level ONLY; band: >=20us components +/-3%,
# micro-kernel chains (r1 e2e ~20us, tout-dominated) +/-10% (iter8 finding 7).
ANCHOR_E2E = {"r2": 37.55, "r4": 217.16, "r6": 39.04, "r3": 79.88, "c7": 47.69,
              "r1": 20.67, "r5": 146.07, "c19": 49.86, "c8": 797.67}
ANCHOR_BAND = {s: 3.0 for s in ANCHOR_E2E}
ANCHOR_BAND["r1"] = 10.0
# Predicted net gain central (candidate_iter10_P4.json economics): the full
# eliminated tqkv share; c7 is the inference-level midpoint of 15-23%.
PREDICTED = {"r2": 27.0, "r4": 12.8, "r6": 27.8, "r3": 23.0, "c7": 19.0}
# measured tqkv anchors for the benefit-coverage decomposition (iter6 B-matrix
# r2/r6/r3/r4; c7 measured by THIS census — filled from base-arm rows)
TQKV_ANCHOR = {"r2": 10.02, "r6": 10.85, "r3": 18.38, "r4": 27.77,
               "r1": 4.65, "c19": 16.50, "c8": 51.14}


def parse_dir(base):
    out = {}
    for label, kinds in find_csvs(base).items():
        r = parse_kernel(kinds)
        if r:
            out[label] = r
    return out


def stats(vals):
    m = statistics.mean(vals)
    sp = (max(vals) - min(vals)) / m * 100 if m else 0.0
    return m, sp


def fmt(vals):
    return "[" + ", ".join(f"{v:.2f}" for v in vals) + "]"


def collect():
    data = {}
    for shape in SHAPES:
        for arm in ARMS:
            rows = []
            for n in SAMPLES[shape]:
                hits = sorted(glob.glob(os.path.join(HERE, f"msprof_op_iter10_*_{shape}_{arm}_s{n}")))
                for d in hits:
                    if not os.path.isdir(d):
                        continue
                    inst = parse_dir(d)
                    if not inst:
                        print(f"!! no CSVs {os.path.basename(d)}")
                        continue
                    rec = {"s": n, "dir": os.path.basename(d), "inst": inst,
                           "e2e": sum(r["dur_us"] for r in inst.values()),
                           "dropped": sum(r.get("rows_dropped", 0) for r in inst.values()),
                           "freqs": {r.get("freq") for r in inst.values()}}
                    rows.append(rec)
            data[(shape, arm)] = rows
    return data


def main_inst(rec):
    return rec["inst"].get("main_kernel:0") or rec["inst"].get("main")


def report(data):
    ok_all = True
    print("=" * 118)
    print("0. sample census + frequency + artifact check")
    print("=" * 118)
    for shape in SHAPES:
        for arm in ARMS:
            rows = data[(shape, arm)]
            freqs = set().union(*[r["freqs"] for r in rows]) if rows else set()
            dropped = sum(r["dropped"] for r in rows)
            print(f"{shape:<5}{arm:<6} n={len(rows)}/{len(SAMPLES[shape])} "
                  f"freqs={sorted(str(f) for f in freqs)} artifact_rows_dropped={dropped}")
            if not rows:
                ok_all = False

    print()
    print("=" * 118)
    print("1. e2e (Σ TaskDur over the whole kernel chain) — same-session interleaved A/B")
    print("=" * 118)
    means = {}
    print(f"{'shape':<7}{'arm':<6}{'n':<3}{'e2e mean':<11}{'spread%':<9}{'samples'}")
    for shape in SHAPES:
        for arm in ARMS:
            rows = data[(shape, arm)]
            if not rows:
                continue
            e2e = [r["e2e"] for r in rows]
            m, sp = stats(e2e)
            means[(shape, arm)] = m
            print(f"{shape:<7}{arm:<6}{len(rows):<3}{m:<11.2f}{sp:<9.1f}{fmt(e2e)}")

    print(f"\n{'shape':<7}{'role':<24}{'base e2e':<11}{'p4 e2e':<11}{'delta us':<11}"
          f"{'improve %':<11}{'predicted %':<12}{'note'}")
    verdicts = {}
    for shape in SHAPES:
        if (shape, "base") not in means or (shape, "p4") not in means:
            continue
        b, p = means[(shape, "base")], means[(shape, "p4")]
        imp = (b - p) / b * 100
        role = ("PRIMARY" if shape in PRIMARY else
                "covered-secondary" if shape in COVERED else "out-of-domain guardrail")
        if shape in PRIMARY:
            ok = imp >= 3.0
            note = "PASS >=3%" if ok else "** FAIL <3% **"
        elif shape in COVERED:
            ok = imp > -3.0
            note = "PASS (no regression)" if ok else "** REGRESSION >3% — REJECT **"
        else:
            ok = abs(imp) <= 3.0
            note = "PASS (untouched)" if ok else "** >3% MOVE — INVESTIGATE **"
        verdicts[shape] = (imp, ok, note)
        ok_all = ok_all and ok
        print(f"{shape:<7}{role:<24}{b:<11.2f}{p:<11.2f}{b - p:<+11.2f}{imp:<+11.2f}"
              f"{(f'{PREDICTED[shape]:.1f}' if shape in PREDICTED else '—'):<12}{note}")
        bs = [r["e2e"] for r in data[(shape, "base")]]
        ps = [r["e2e"] for r in data[(shape, "p4")]]
        if bs and ps:
            ov = "non-overlapping" if (min(bs) > max(ps) or min(ps) > max(bs)) else "OVERLAPPING"
            if shape in PRIMARY:
                ok_all = ok_all and (min(bs) > max(ps))
            print(f"{'':7}  distributions: base {fmt(bs)} vs p4 {fmt(ps)} -> {ov}")

    print("\n--- BASE-arm session-health anchors (cross-session reference level) ---")
    for shape in SHAPES:
        if (shape, "base") in means:
            a = ANCHOR_E2E[shape]
            dev = (means[(shape, "base")] - a) / a * 100
            band = ANCHOR_BAND[shape]
            tag = "OK" if abs(dev) <= band else f"** OUT OF ±{band}% BAND — disclose **"
            print(f"{shape:<7} base {means[(shape,'base')]:9.2f} vs anchor {a:9.2f} "
                  f"dev {dev:+6.2f}% (band ±{band}%)  {tag}")

    print()
    print("=" * 118)
    print("2. transpose_qkv attribution (elimination evidence) + main TaskDur (strided-load cost)")
    print("=" * 118)
    print(f"{'shape':<7}{'arm':<6}{'instances':<36}{'tqkv us[spread%]':<20}"
          f"{'tout us':<10}{'main us[spread%]':<22}{'main blk'}")
    tqkv_base = {}
    tqkv_p4 = {}
    main_means = {}
    for shape in SHAPES:
        for arm in ARMS:
            rows = data[(shape, arm)]
            if not rows:
                continue
            labels = sorted({k for r in rows for k in r["inst"]})

            def agg(key):
                rs = [r["inst"].get(key) for r in rows]
                rs = [x for x in rs if x]
                if not rs:
                    return None
                vals = [x["dur_us"] for x in rs]
                m, sp = stats(vals)
                return m, sp, rs[0].get("block_dim")

            a_q = agg("kernel_kernel:0")
            a_t = agg("kernel_kernel:1")
            a_m = agg("main_kernel:0") or agg("main")
            f = lambda a: f"{a[0]:.2f}[{a[1]:.1f}%]" if a else "ABSENT"  # noqa: E731
            # kernel_kernel:0 is tqkv on prefill chains; on r1 it is also tqkv
            # (kernel_kernel:1 = tout there).  Decode (r5) has neither.
            if a_q and arm == "base":
                tqkv_base.setdefault(shape, a_q[0])
            if a_q and arm == "p4":
                tqkv_p4.setdefault(shape, a_q[0])
            if a_m:
                main_means[(shape, arm)] = a_m[0]
            print(f"{shape:<7}{arm:<6}{str(labels)[:34]:<36}{f(a_q):<20}"
                  f"{f(a_t):<10}{f(a_m):<22}{(a_m[2] if a_m else '—')}")

    print("\n--- tqkv elimination on covered shapes (p4 arm must be ABSENT) ---")
    for shape in COVERED:
        absent = shape not in tqkv_p4
        tb = tqkv_base.get(shape, float("nan"))
        print(f"{shape:<7} base tqkv={tb:.2f}us -> p4 "
              f"{'ABSENT (eliminated)' if absent else 'STILL PRESENT ' + format(tqkv_p4[shape], '.2f')}")
        ok_all = ok_all and absent
    print("\n--- guards: tqkv must STILL RUN in both arms (untouched domains) ---")
    for shape in GUARDS:
        if shape == "r5":
            both_absent = "r5" not in tqkv_base and "r5" not in tqkv_p4
            print(f"{shape:<7} decode: tqkv absent both arms = {both_absent}")
            ok_all = ok_all and both_absent
        else:
            ran = shape in tqkv_base and shape in tqkv_p4
            print(f"{shape:<7} tqkv base={tqkv_base.get(shape, float('nan')):.2f} "
                  f"p4={tqkv_p4.get(shape, float('nan')):.2f} (still runs = {ran})")
            ok_all = ok_all and ran

    print("\n--- main-kernel TaskDur ARM vs BASE (P4 criterion: record as-is; "
          "tripwire = Δmain > 50% of eliminated tqkv) ---")
    for shape in SHAPES:
        if (shape, "base") in main_means and (shape, "p4") in main_means:
            mb, mp = main_means[(shape, "base")], main_means[(shape, "p4")]
            d = (mp - mb) / mb * 100
            dm = mp - mb
            tq = tqkv_base.get(shape, 0.0)
            trip = shape in COVERED and dm > 0.5 * tq
            role = "covered" if shape in COVERED else "guard (must be ~0)"
            tag = "** TRIPWIRE: Δmain > 50% tqkv — re-verify economics **" if trip else "ok"
            if shape in GUARDS and abs(d) > 3.0:
                tag = "** guard main moved >3% — INVESTIGATE **"
                ok_all = False
            print(f"{shape:<7} base={mb:9.2f} p4={mp:9.2f} delta={dm:+8.2f}us ({d:+6.2f}%)  "
                  f"tqkv={tq:7.2f}  {tag}  [{role}]")

    print("\n--- e2e net-gain decomposition (benefit vs measured cost) ---")
    for shape in COVERED:
        if (shape, "base") in means and (shape, "p4") in means:
            b, p = means[(shape, "base")], means[(shape, "p4")]
            tq = tqkv_base.get(shape, 0.0)
            dm = main_means.get((shape, "p4"), 0) - main_means.get((shape, "base"), 0)
            print(f"{shape:<7} e2e delta={p - b:+9.2f}us = -tqkv({tq:.2f}) + main_delta({dm:+.2f}) "
                  f"+ residual({(p - b) + tq - dm:+.2f})   [coverage of benefit: "
                  f"{100 * (b - p) / tq if tq else float('nan'):.1f}% of tqkv]")

    print()
    print("=" * 118)
    print("3. wait-id telemetry (main instance) — MANDATORY pacing discriminating gate")
    print("   (both covered kernels are manual-flag expert kernels — iter4-signature check applies)")
    print("=" * 118)
    for shape in COVERED:
        print(f"\n--- {shape} ---")
        agg = {}
        for arm in ARMS:
            sums = {w: [] for w in WAITID}
            for r in data[(shape, arm)]:
                m = main_inst(r)
                if not m or not m.get("waitid"):
                    continue
                for w in WAITID:
                    v = m["waitid"].get(w)
                    if v is not None:
                        sums[w].append(v[2])  # mean across cores
            agg[arm] = {w: (statistics.mean(v) if v else float("nan")) for w, v in sums.items()}
        tq = tqkv_base.get(shape, 0.0)
        tot_b = sum(v for v in agg["base"].values() if v == v)
        tot_p = sum(v for v in agg["p4"].values() if v == v)
        print(f"{'wait id':<34}{'base us':>11}{'p4 us':>11}{'delta us':>11}{'delta %':>10}")
        for w in WAITID:
            b, p = agg["base"][w], agg["p4"][w]
            dp = (p - b) / b * 100 if b else float("nan")
            print(f"{w:<34}{b:>11.2f}{p:>11.2f}{p - b:>+11.2f}{dp:>+9.1f}%")
        print(f"{'SUM(wait ids)':<34}{tot_b:>11.2f}{tot_p:>11.2f}{tot_p - tot_b:>+11.2f}"
              f"{(100 * (tot_p - tot_b) / tot_b if tot_b else float('nan')):>+9.1f}%")
        absorbed = (tot_p - tot_b) > tq
        v3b, v3p = agg["base"]["aiv_scalar_wait_id3_time(us)"], agg["p4"]["aiv_scalar_wait_id3_time(us)"]
        v3_infl = v3b and (v3p - v3b) / v3b > 0.20
        c1b, c1p = agg["base"]["aic_scalar_wait_id1_time(us)"], agg["p4"]["aic_scalar_wait_id1_time(us)"]
        c1_infl = c1b and (c1p - c1b) / c1b > 0.20
        gain = 100 * (means[(shape, "base")] - means[(shape, "p4")]) / means[(shape, "base")] \
            if (shape, "base") in means else float("nan")
        half_pred = gain < 0.5 * PREDICTED[shape]
        if absorbed:
            v = "REJECT — wait inflation exceeds the eliminated tqkv TaskDur (iter4 signature)"
            ok_all = False
        elif (v3_infl or c1_infl) and half_pred:
            v = "DOWNGRADE confidence + disclose — pacing mirror inflates >20% and gain <50% of predicted"
        elif v3_infl or c1_infl:
            v = "disclose — pacing mirror inflates >20% but the gain still >=50% of predicted"
        else:
            v = "PASS — no pacing migration signature"
        print(f"  eliminated tqkv={tq:.2f}us; SUM(wait) delta={tot_p - tot_b:+.2f}us; "
              f"gain={gain:+.1f}% vs predicted {PREDICTED[shape]:.1f}%")
        print(f"  VERDICT: {v}")

    print()
    print("=" * 118)
    print(f"OVERALL: {'ALL PREREGISTERED CRITERIA PASS' if ok_all else 'SOME CRITERION FAILED — see above'}")
    print("=" * 118)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(report(collect()))
