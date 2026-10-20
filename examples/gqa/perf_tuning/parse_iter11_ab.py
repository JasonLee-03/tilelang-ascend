"""Iter11 Gate-2 A/B aggregator — P4 wave-2 (transpose_qkv elimination on the
g2stack + prefill_expert dual-aligned domains).

Reuses parse_msprof_iter6.find_csvs/parse_kernel (impossible-row artifact
filter included).  Layout: msprof_op_iter11_{SEG}_{shape}_{arm}_s{n}/OPPROF_*/
{kernel}/{i}/*.csv with instances in launch order:
  BASE covered (c19/c8/c9/c10): kernel_kernel:0 = transpose_qkv,
      main_kernel:0 = main (tout already eliminated by iter9 out_bsnd).
  W2 covered: main_kernel:0 ONLY — tqkv ABSENCE is the elimination evidence.
  Guards: r2 = main only BOTH arms (wave-1 shipped in_bsnd — tqkv absence on
      r2 is the "wave-1 chain unchanged" evidence); r1 = tqkv + main + tout;
      r5 = main only.
e2e = SUM of TaskDur over all instances in the run.

Verdict lines evaluate the PREREGISTERED criteria in candidate_iter11_P4w2.json
(locked 2026-10-20T12:35) — they do NOT define them:
  primary  : c19 >=3% AND non-overlapping, AND (c8 OR c10) >=3% AND
             non-overlapping — BOTH conjuncts must pass
  secondary: c9 reported as-is; ANY covered-shape e2e regression >3% vetoes
  guards   : r2/r1/r5 — any e2e regression >3% vetoes (>3% "improvement" on a
             guard is impossible-by-construction => INVESTIGATE + disclose)
  main cost: recorded as-is; tripwire Δmain > 50% of eliminated tqkv
  wait-id  : MANDATORY — Σwait inflation > eliminated tqkv => REJECT;
             pacing mirrors C-id2/V-id4 (ns-batch phase-alternation pacers,
             iter3/5 evidence) ±20% band + C-id1/V-id3 release mirrors;
             mirror out-of-band AND gain <50% predicted => downgrade+disclose
  anchors  : BASE-arm reproduction of the recorded shipped ranges; >=20µs
             components ±3%, micro-kernel launch-floor chains (r1) ±10%;
             deviation reported as-is, no goalpost moving

Run: python3 examples/gqa/perf_tuning/parse_iter11_ab.py
"""

import glob
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parse_msprof_iter6 import find_csvs, parse_kernel  # noqa: E402

SHAPES = ["c19", "c8", "c9", "c10", "r2", "r1", "r5"]
COVERED = ["c19", "c8", "c9", "c10"]
SECONDARY = ["c9"]
GUARDS = ["r2", "r1", "r5"]
ARMS = ("base", "w2")
SAMPLES = {s: (1, 2, 3) for s in SHAPES}
SAMPLES["r5"] = (1, 2)
WAITID = ["aic_scalar_wait_id1_time(us)", "aic_scalar_wait_id2_time(us)",
          "aic_scalar_wait_id5_time(us)", "aiv_scalar_wait_id0_time(us)",
          "aiv_scalar_wait_id3_time(us)", "aiv_scalar_wait_id4_time(us)"]
# Session-health anchors (shipped-chain e2e ranges recorded in
# candidate_iter11_P4w2.json / baseline_iter11.json; most recent same-file
# sessions).  Cross-session reference level ONLY; band: >=20us components
# ±3%, micro-kernel launch-floor-dominated chains (r1) ±10% (iter8 finding 7).
ANCHOR_E2E = {
    "c19": (49.86, 50.14),     # iter9 W2 arm / iter10 guard base
    "c8": (797.67, 800.61),    # iter9 W2 arm / iter10 guard base
    "c9": (605.21, 605.21),    # iter9 W2 arm
    "c10": (1071.94, 1071.94), # iter9 W2 arm
    "r2": (26.35, 26.91),      # iter10 Gate2 p4 arm / iter10 G3 fallback (current shipped)
    "r1": (20.17, 20.67),      # iter10 gate2 / iter9
    "r5": (146.07, 146.51),    # iter9/iter10
}
ANCHOR_BAND = {s: 3.0 for s in ANCHOR_E2E}
ANCHOR_BAND["r1"] = 10.0
# Predicted net gain central (candidate_iter11_P4w2.json economics): the full
# eliminated tqkv share (wave-1 realization 92.4-103.3% as the reference band).
PREDICTED = {"c19": 33.0, "c8": 6.4, "c9": 7.3, "c10": 8.4}
# measured tqkv anchors for the benefit-coverage decomposition (iter9 phaseA;
# re-measured in-session from the base-arm rows below)
TQKV_ANCHOR = {"c19": 16.50, "c8": 51.14, "c9": 44.32, "c10": 89.99}


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
                hits = sorted(glob.glob(os.path.join(HERE, f"msprof_op_iter11_*_{shape}_{arm}_s{n}")))
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

    dist_nonoverlap = {}
    print(f"\n{'shape':<7}{'role':<26}{'base e2e':<11}{'w2 e2e':<11}{'delta us':<11}"
          f"{'improve %':<11}{'predicted %':<12}{'note'}")
    verdicts = {}
    for shape in SHAPES:
        if (shape, "base") not in means or (shape, "w2") not in means:
            continue
        b, w = means[(shape, "base")], means[(shape, "w2")]
        imp = (b - w) / b * 100
        bs = [r["e2e"] for r in data[(shape, "base")]]
        ws = [r["e2e"] for r in data[(shape, "w2")]]
        nonov = bool(bs and ws) and (min(bs) > max(ws) or min(ws) > max(bs))
        dist_nonoverlap[shape] = nonov
        if shape in COVERED:
            role = ("PRIMARY(prexpert)" if shape == "c19" else
                    "PRIMARY-ALT(g2stack)" if shape in ("c8", "c10") else "covered-secondary")
        else:
            role = "out-of-domain guardrail"
        if shape in COVERED:
            ok = imp > -3.0  # covered regression >3% vetoes; >=3% gain evaluated in the primary/secondary verdicts below
            note = "no regression" if ok else "** REGRESSION >3% — REJECT **"
        else:
            # PREREGISTERED guard criterion: any e2e REGRESSION >3% vetoes.
            # A >3% "gain" on an untouched guard is impossible by construction
            # (identical code both arms) => session drift/noise: INVESTIGATE +
            # disclose, NOT a criteria veto (criteria define what vetoes).
            ok = imp > -3.0
            if imp <= -3.0:
                note = "** REGRESSION >3% — REJECT **"
            elif imp > 3.0:
                note = ">3% GAIN ON UNTOUCHED GUARD — investigate/discard as drift (disclose)"
            else:
                note = "PASS (untouched)"
        verdicts[shape] = (imp, ok, note)
        ok_all = ok_all and ok
        print(f"{shape:<7}{role:<26}{b:<11.2f}{w:<11.2f}{b - w:<+11.2f}{imp:<+11.2f}"
              f"{(f'{PREDICTED[shape]:.1f}' if shape in PREDICTED else '—'):<12}{note}")
        if bs and ws:
            ov = "non-overlapping" if nonov else "OVERLAPPING"
            print(f"{'':7}  distributions: base {fmt(bs)} vs w2 {fmt(ws)} -> {ov}")

    # ---- PREREGISTERED primary criterion: c19 AND (c8 OR c10) ----
    print("\n--- PRIMARY CRITERION (preregistered): c19 >=3% non-overlapping "
          "AND (c8 OR c10) >=3% non-overlapping — BOTH conjuncts ---")
    def prim_ok(s):
        return s in verdicts and verdicts[s][0] >= 3.0 and dist_nonoverlap.get(s, False)
    c19_ok = prim_ok("c19")
    g2_ok = prim_ok("c8") or prim_ok("c10")
    print(f"c19: improve={verdicts.get('c19', (float('nan'),))[0]:+.2f}% "
          f"non-overlapping={dist_nonoverlap.get('c19')} -> {'PASS' if c19_ok else 'FAIL'}")
    for s in ("c8", "c10"):
        print(f"{s}: improve={verdicts.get(s, (float('nan'),))[0]:+.2f}% "
              f"non-overlapping={dist_nonoverlap.get(s)} -> {'PASS' if prim_ok(s) else 'below bar'}")
    print(f"g2stack representative (c8 OR c10): {'PASS' if g2_ok else 'FAIL'}")
    primary_verdict = c19_ok and g2_ok
    ok_all = ok_all and primary_verdict
    print(f"PRIMARY: {'PASS' if primary_verdict else '** FAIL **'}")
    print(f"SECONDARY (c9, reported as-is): improve={verdicts.get('c9', (float('nan'),))[0]:+.2f}% "
          f"non-overlapping={dist_nonoverlap.get('c9')} (predicted {PREDICTED['c9']:.1f}%)")

    # ---- per-domain partial-adoption recommendation (task-book: partial
    # adoption allowed — integrate ONLY the passing domain(s)) ----
    g2_no_regression = all(verdicts.get(s, (0, True))[1] for s in ("c8", "c9", "c10") if s in verdicts)
    pe_domain_ok = c19_ok and verdicts.get("c19", (0, False))[1]
    g2_domain_ok = g2_ok and g2_no_regression
    print(f"\nDOMAIN VERDICTS: prefill_expert(c19)={'ADOPT-eligible' if pe_domain_ok else 'NOT eligible'}; "
          f"g2stack(c8/c9/c10)={'ADOPT-eligible' if g2_domain_ok else 'NOT eligible'}")

    print("\n--- BASE-arm session-health anchors (cross-session reference level) ---")
    for shape in SHAPES:
        if (shape, "base") in means:
            lo, hi = ANCHOR_E2E[shape]
            m = means[(shape, "base")]
            band = ANCHOR_BAND[shape]
            in_range = lo * (1 - band / 100) <= m <= hi * (1 + band / 100)
            dev_lo, dev_hi = (m - lo) / lo * 100, (m - hi) / hi * 100
            tag = "OK" if in_range else f"** OUT OF anchor-range ±{band}% — disclose as-is **"
            print(f"{shape:<5} base {m:9.2f} vs anchor [{lo:.2f}, {hi:.2f}] "
                  f"dev {dev_lo:+6.2f}%/{dev_hi:+6.2f}% (band ±{band}%)  {tag}")

    print()
    print("=" * 118)
    print("2. transpose_qkv attribution (elimination evidence) + main TaskDur (strided-load cost)")
    print("=" * 118)
    print(f"{'shape':<7}{'arm':<6}{'instances':<36}{'tqkv us[spread%]':<20}"
          f"{'tout us':<10}{'main us[spread%]':<22}{'main blk'}")
    tqkv_base = {}
    tqkv_w2 = {}
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
            # (kernel_kernel:1 = tout there).  Decode (r5) and wave-1 (r2)
            # have neither.
            if a_q and arm == "base":
                tqkv_base.setdefault(shape, a_q[0])
            if a_q and arm == "w2":
                tqkv_w2.setdefault(shape, a_q[0])
            if a_m:
                main_means[(shape, arm)] = a_m[0]
            print(f"{shape:<7}{arm:<6}{str(labels)[:34]:<36}{f(a_q):<20}"
                  f"{f(a_t):<10}{f(a_m):<22}{(a_m[2] if a_m else '—')}")

    print("\n--- tqkv elimination on covered shapes (w2 arm must be ABSENT) ---")
    for shape in COVERED:
        absent = shape not in tqkv_w2
        tb = tqkv_base.get(shape, float("nan"))
        anch = TQKV_ANCHOR[shape]
        dev = (tb - anch) / anch * 100 if tb == tb else float("nan")
        print(f"{shape:<7} base tqkv={tb:.2f}us (anchor {anch:.2f}, dev {dev:+.1f}%) -> w2 "
              f"{'ABSENT (eliminated)' if absent else 'STILL PRESENT ' + format(tqkv_w2[shape], '.2f')}")
        ok_all = ok_all and absent

    print("\n--- guards: chain structure must be UNCHANGED across arms ---")
    for shape in GUARDS:
        if shape == "r5":
            both_absent = "r5" not in tqkv_base and "r5" not in tqkv_w2
            print(f"{shape:<7} decode: tqkv absent both arms = {both_absent}")
            ok_all = ok_all and both_absent
        elif shape == "r2":
            both_absent = "r2" not in tqkv_base and "r2" not in tqkv_w2
            print(f"{shape:<7} wave-1 flat: tqkv ABSENT both arms (shipped in_bsnd "
                  f"unchanged) = {both_absent}")
            ok_all = ok_all and both_absent
        else:
            ran = shape in tqkv_base and shape in tqkv_w2
            print(f"{shape:<7} tqkv base={tqkv_base.get(shape, float('nan')):.2f} "
                  f"w2={tqkv_w2.get(shape, float('nan')):.2f} (still runs = {ran}; tout must "
                  f"also still run — see instances column above)")
            ok_all = ok_all and ran

    print("\n--- main-kernel TaskDur ARM vs BASE (criterion: record as-is; "
          "tripwire = Δmain > 50% of eliminated tqkv) ---")
    for shape in SHAPES:
        if (shape, "base") in main_means and (shape, "w2") in main_means:
            mb, mw = main_means[(shape, "base")], main_means[(shape, "w2")]
            d = (mw - mb) / mb * 100
            dm = mw - mb
            tq = tqkv_base.get(shape, 0.0)
            trip = shape in COVERED and dm > 0.5 * tq
            role = "covered" if shape in COVERED else "guard (must be ~0)"
            tag = "** TRIPWIRE: Δmain > 50% tqkv — re-verify economics **" if trip else "ok"
            if trip:
                ok_all = False
            if shape in GUARDS and abs(d) > 3.0:
                tag = "** guard main moved >3% — INVESTIGATE **"
                ok_all = False
            print(f"{shape:<7} base={mb:9.2f} w2={mw:9.2f} delta={dm:+8.2f}us ({d:+6.2f}%)  "
                  f"tqkv={tq:7.2f}  {tag}  [{role}]")

    print("\n--- e2e net-gain decomposition (benefit vs measured cost) ---")
    for shape in COVERED:
        if (shape, "base") in means and (shape, "w2") in means:
            b, w = means[(shape, "base")], means[(shape, "w2")]
            tq = tqkv_base.get(shape, 0.0)
            dm = main_means.get((shape, "w2"), 0) - main_means.get((shape, "base"), 0)
            print(f"{shape:<7} e2e delta={w - b:+9.2f}us = -tqkv({tq:.2f}) + main_delta({dm:+.2f}) "
                  f"+ residual({(w - b) + tq - dm:+.2f})   [coverage of benefit: "
                  f"{100 * (b - w) / tq if tq else float('nan'):.1f}% of tqkv]")

    print()
    print("=" * 118)
    print("3. wait-id telemetry (main instance) — MANDATORY pacing discriminating gate")
    print("   (both covered kernels are ns-batch manual-flag expert kernels — the batch")
    print("    pacers are C-id2/V-id4 (iter3/5); release mirrors C-id1/V-id3 reported)")
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
        tot_w = sum(v for v in agg["w2"].values() if v == v)
        print(f"{'wait id':<34}{'base us':>11}{'w2 us':>11}{'delta us':>11}{'delta %':>10}")
        for w in WAITID:
            b, p = agg["base"][w], agg["w2"][w]
            dp = (p - b) / b * 100 if b else float("nan")
            print(f"{w:<34}{b:>11.2f}{p:>11.2f}{p - b:>+11.2f}{dp:>+9.1f}%")
        print(f"{'SUM(wait ids)':<34}{tot_b:>11.2f}{tot_w:>11.2f}{tot_w - tot_b:>+11.2f}"
              f"{(100 * (tot_w - tot_b) / tot_b if tot_b else float('nan')):>+9.1f}%")
        absorbed = (tot_w - tot_b) > tq
        # batch pacers (primary mirrors for the ns-batched kernels)
        c2b, c2w = agg["base"]["aic_scalar_wait_id2_time(us)"], agg["w2"]["aic_scalar_wait_id2_time(us)"]
        c2_infl = c2b and (c2w - c2b) / c2b > 0.20
        v4b, v4w = agg["base"]["aiv_scalar_wait_id4_time(us)"], agg["w2"]["aiv_scalar_wait_id4_time(us)"]
        v4_infl = v4b and (v4w - v4b) / v4b > 0.20
        # release mirrors (reference)
        v3b, v3w = agg["base"]["aiv_scalar_wait_id3_time(us)"], agg["w2"]["aiv_scalar_wait_id3_time(us)"]
        v3_infl = v3b and (v3w - v3b) / v3b > 0.20
        c1b, c1w = agg["base"]["aic_scalar_wait_id1_time(us)"], agg["w2"]["aic_scalar_wait_id1_time(us)"]
        c1_infl = c1b and (c1w - c1b) / c1b > 0.20
        gain = 100 * (means[(shape, "base")] - means[(shape, "w2")]) / means[(shape, "base")] \
            if (shape, "base") in means else float("nan")
        half_pred = gain < 0.5 * PREDICTED[shape]
        mirror_infl = bool(c2_infl or v4_infl or v3_infl or c1_infl)
        if absorbed:
            v = "REJECT — wait inflation exceeds the eliminated tqkv TaskDur (iter4 signature)"
            ok_all = False
        elif mirror_infl and half_pred:
            v = "DOWNGRADE confidence + disclose — pacing mirror inflates >20% and gain <50% of predicted"
        elif mirror_infl:
            v = "disclose — pacing mirror inflates >20% but the gain still >=50% of predicted"
        else:
            v = "PASS — no pacing migration signature"
        print(f"  eliminated tqkv={tq:.2f}us; SUM(wait) delta={tot_w - tot_b:+.2f}us; "
              f"gain={gain:+.1f}% vs predicted {PREDICTED[shape]:.1f}%")
        print(f"  mirrors: C-id2 {'INFL>20%' if c2_infl else 'flat'} / V-id4 {'INFL>20%' if v4_infl else 'flat'}"
              f" (batch pacers); C-id1 {'INFL>20%' if c1_infl else 'flat'} / V-id3 "
              f"{'INFL>20%' if v3_infl else 'flat'} (release, reference)")
        print(f"  VERDICT: {v}")

    print()
    print("=" * 118)
    print(f"OVERALL: {'ALL PREREGISTERED CRITERIA PASS' if ok_all else 'SOME CRITERION FAILED — see above'}")
    print("=" * 118)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(report(collect()))
