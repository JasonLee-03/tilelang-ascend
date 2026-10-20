"""Iter8 Gate-2 A/B aggregator — P2 (BSND direct write, transpose_out eliminated).

Reuses parse_msprof_iter6.find_csvs/parse_kernel (impossible-row artifact
filter included).  Layout: msprof_op_iter8_P2_{shape}_{arm}_s{n}/OPPROF_*/
{kernel}/{i}/*.csv with instances in launch order:
  kernel_kernel:0 = transpose_qkv, main_kernel:0 = main,
  kernel_kernel:1 = transpose_out (BASE arm / out-of-domain shapes only —
  its ABSENCE in the P2 arm on covered shapes IS the elimination evidence).
  r5 decode: main only in both arms (BSND direct, no aux).
e2e = SUM of TaskDur over all instances in the run.

Reports (criteria are PREREGISTERED in candidate_iter8_P2.json — the verdict
lines below evaluate them, they do not define them):
  1. e2e per shape/arm: samples, mean, spread; improvement %
  2. transpose_out attribution: present/absent, TaskDur, Block Dim
  3. main-kernel TaskDur ARM vs BASE (direct-write cost guardrail, <= +5%)
  4. wait-id telemetry (main instance) ARM vs BASE — the mandatory pacing
     discriminating gate: SUM(wait) inflation must not exceed the eliminated
     tout TaskDur; V-id3 inflation > 20% with < 50% of predicted gain =>
     confidence downgrade
  5. session health: BASE vs the iter7 anchors (+/-3%)
  6. artifact/log audit: dropped rows, rc, freq, probe asserts from the log

Run: python3 examples/gqa/perf_tuning/parse_iter8_ab.py
"""

import glob
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parse_msprof_iter6 import find_csvs, parse_kernel  # noqa: E402

SHAPES = ["r1", "r2", "r3", "r4", "r5", "r6", "c7"]
COVERED = ["r2", "r3", "r4", "r6", "c7"]
OUT_OF_DOMAIN = ["r1", "r5"]
PRIMARY = ["r2", "r6"]
ARMS = ("base", "p2")
WAITID = ["aic_scalar_wait_id1_time(us)", "aic_scalar_wait_id2_time(us)",
          "aic_scalar_wait_id5_time(us)", "aiv_scalar_wait_id0_time(us)",
          "aiv_scalar_wait_id3_time(us)", "aiv_scalar_wait_id4_time(us)"]
# iter7 V2 (adopted P1) anchors — session health reference
ANCHOR_E2E = {"r1": 20.05, "r2": 43.70, "r3": 98.31, "r4": 246.45, "r5": 146.81, "r6": 49.06}
ANCHOR_TOUT = {"r1": 4.86, "r2": 7.67, "r3": 17.63, "r4": 29.58, "r6": 9.69}
# Phase-A predicted net gain (central) per covered shape, % of e2e
PREDICTED = {"r2": 17.0, "r6": 19.5, "r3": 17.1, "r4": 11.6, "c7": 14.0}


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
            for n in (1, 2, 3):
                d = os.path.join(HERE, f"msprof_op_iter8_P2_{shape}_{arm}_s{n}")
                if not os.path.isdir(d):
                    print(f"!! missing {os.path.basename(d)}")
                    continue
                inst = parse_dir(d)
                if not inst:
                    print(f"!! no CSVs {os.path.basename(d)}")
                    continue
                rec = {"s": n, "inst": inst,
                       "e2e": sum(r["dur_us"] for r in inst.values()),
                       "dropped": sum(r.get("rows_dropped", 0) for r in inst.values()),
                       "freqs": {r.get("freq") for r in inst.values()}}
                rows.append(rec)
            data[(shape, arm)] = rows
    return data


def main_inst(rec):
    return rec["inst"].get("main_kernel:0") or rec["inst"].get("main")


def report(data):
    print("=" * 118)
    print("1. e2e (Σ TaskDur over the whole kernel chain) — same-session interleaved A/B")
    print("=" * 118)
    means = {}
    print(f"{'shape':<6}{'arm':<6}{'n':<3}{'e2e mean':<10}{'spread%':<9}{'samples'}")
    for shape in SHAPES:
        for arm in ARMS:
            rows = data[(shape, arm)]
            if not rows:
                continue
            e2e = [r["e2e"] for r in rows]
            m, sp = stats(e2e)
            means[(shape, arm)] = m
            print(f"{shape:<6}{arm:<6}{len(rows):<3}{m:<10.2f}{sp:<9.1f}{fmt(e2e)}")

    print(f"\n{'shape':<6}{'role':<22}{'base e2e':<10}{'p2 e2e':<10}{'delta us':<11}"
          f"{'improve %':<11}{'predicted %':<12}{'note'}")
    verdicts = {}
    for shape in SHAPES:
        if (shape, "base") not in means or (shape, "p2") not in means:
            continue
        b, p = means[(shape, "base")], means[(shape, "p2")]
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
        print(f"{shape:<6}{role:<22}{b:<10.2f}{p:<10.2f}{b - p:<+11.2f}{imp:<+11.2f}"
              f"{(f'{PREDICTED[shape]:.1f}' if shape in PREDICTED else '—'):<12}{note}")
        # distribution overlap disclosure
        bs = [r["e2e"] for r in data[(shape, "base")]]
        ps = [r["e2e"] for r in data[(shape, "p2")]]
        if bs and ps:
            ov = "non-overlapping" if (min(bs) > max(ps) or min(ps) > max(bs)) else "OVERLAPPING"
            print(f"{'':6}  distributions: base {fmt(bs)} vs p2 {fmt(ps)} -> {ov}")

    print()
    print("=" * 118)
    print("2. transpose_out attribution (elimination evidence) + main TaskDur (direct-write cost)")
    print("=" * 118)
    print(f"{'shape':<6}{'arm':<6}{'instances':<34}{'tout us[spread%]':<20}{'tout blk':<9}"
          f"{'tqkv us':<10}{'main us[spread%]':<20}{'main blk'}")
    tout_base = {}
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

            a_t = agg("kernel_kernel:1")
            a_q = agg("kernel_kernel:0")
            a_m = agg("main_kernel:0") or agg("main")
            f = lambda a: f"{a[0]:.2f}[{a[1]:.1f}%]" if a else "ABSENT"
            if a_t:
                tout_base.setdefault(shape, a_t[0])
            if a_m:
                main_means[(shape, arm)] = a_m[0]
            print(f"{shape:<6}{arm:<6}{str(labels)[:32]:<34}{f(a_t):<20}"
                  f"{(a_t[2] if a_t else '—'):<9}{(f'{a_q[0]:.2f}' if a_q else '—'):<10}"
                  f"{f(a_m):<20}{(a_m[2] if a_m else '—')}")

    print("\n--- main-kernel TaskDur ARM vs BASE (preregistered guardrail: inflation <= +5%) ---")
    main_ok = {}
    for shape in SHAPES:
        if (shape, "base") in main_means and (shape, "p2") in main_means:
            mb, mp = main_means[(shape, "base")], main_means[(shape, "p2")]
            d = (mp - mb) / mb * 100
            ok = d <= 5.0
            main_ok[shape] = (d, ok)
            tag = "OK" if ok else "** >+5% — pacing-damage tripwire **"
            role = "covered" if shape in COVERED else "out-of-domain (must be ~0)"
            print(f"{shape:<6} base={mb:8.2f} p2={mp:8.2f} delta={d:+6.2f}%  {tag}  [{role}]")

    print("\n--- e2e net-gain decomposition (benefit vs measured cost) ---")
    for shape in COVERED:
        if (shape, "base") in means and (shape, "p2") in means:
            b, p = means[(shape, "base")], means[(shape, "p2")]
            tb = tout_base.get(shape, 0.0)
            dm = main_means.get((shape, "p2"), 0) - main_means.get((shape, "base"), 0)
            print(f"{shape:<6} e2e delta={p - b:+8.2f}us = -tout({tb:.2f}) + main_delta({dm:+.2f}) "
                  f"+ residual({(p - b) + tb - dm:+.2f})   [coverage of benefit: "
                  f"{100 * (b - p) / tb if tb else float('nan'):.1f}% of tout]")

    print()
    print("=" * 118)
    print("3. wait-id telemetry (main instance) — MANDATORY pacing discriminating gate")
    print("=" * 118)
    wait_verdict = {}
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
        tb = tout_base.get(shape, 0.0)
        tot_b = sum(v for v in agg["base"].values() if v == v)
        tot_p = sum(v for v in agg["p2"].values() if v == v)
        print(f"{'wait id':<34}{'base us':>10}{'p2 us':>10}{'delta us':>11}{'delta %':>10}")
        for w in WAITID:
            b, p = agg["base"][w], agg["p2"][w]
            dp = (p - b) / b * 100 if b else float("nan")
            print(f"{w:<34}{b:>10.2f}{p:>10.2f}{p - b:>+11.2f}{dp:>+9.1f}%")
        print(f"{'SUM(wait ids)':<34}{tot_b:>10.2f}{tot_p:>10.2f}{tot_p - tot_b:>+11.2f}"
              f"{(100 * (tot_p - tot_b) / tot_b if tot_b else float('nan')):>+9.1f}%")
        # preregistered rules
        absorbed = (tot_p - tot_b) > tb
        v3b, v3p = agg["base"]["aiv_scalar_wait_id3_time(us)"], agg["p2"]["aiv_scalar_wait_id3_time(us)"]
        v3_infl = v3b and (v3p - v3b) / v3b > 0.20
        gain = 100 * (means[(shape, "base")] - means[(shape, "p2")]) / means[(shape, "base")] \
            if (shape, "base") in means else float("nan")
        half_pred = gain < 0.5 * PREDICTED[shape]
        if absorbed:
            v = "REJECT — wait inflation exceeds the eliminated tout TaskDur (iter4 signature)"
        elif v3_infl and half_pred:
            v = "DOWNGRADE confidence + disclose — V-id3 inflates >20% and gain <50% of predicted"
        elif v3_infl:
            v = "disclose — V-id3 inflates >20% but the gain still >=50% of predicted"
        else:
            v = "PASS — no pacing migration signature"
        wait_verdict[shape] = v
        print(f"  eliminated tout={tb:.2f}us; SUM(wait) delta={tot_p - tot_b:+.2f}us; "
              f"gain={gain:+.1f}% vs predicted {PREDICTED[shape]:.1f}%")
        print(f"  VERDICT: {v}")

    print()
    print("=" * 118)
    print("4. session health (BASE vs the iter7 V2 adopted-arm anchors, +/-3%)")
    print("=" * 118)
    health = {}
    for shape, a in ANCHOR_E2E.items():
        if (shape, "base") in means:
            b = means[(shape, "base")]
            d = (b - a) / a * 100
            ok = abs(d) <= 3.0
            health[shape] = ok
            print(f"{shape:<6} base e2e={b:8.2f} anchor={a:8.2f} delta={d:+6.2f}%  "
                  f"{'OK' if ok else '** OUT OF BAND **'}")
    for shape, a in ANCHOR_TOUT.items():
        if shape in tout_base:
            d = (tout_base[shape] - a) / a * 100
            print(f"{shape:<6} base tout={tout_base[shape]:8.2f} anchor={a:8.2f} delta={d:+6.2f}%")

    print()
    print("=" * 118)
    print("5. data-quality audit (artifact filter, frequency, probe asserts)")
    print("=" * 118)
    dropped = 0
    freqs = set()
    for key, rows in data.items():
        for r in rows:
            dropped += r["dropped"]
            freqs |= r["freqs"]
    print(f"impossible-row artifacts dropped by the parser: {dropped}")
    print(f"frequencies observed: {sorted(freqs)}")
    n_dirs = len(glob.glob(os.path.join(HERE, "msprof_op_iter8_P2_*")))
    print(f"msprof dirs present: {n_dirs} (expected {len(SHAPES) * len(ARMS) * 3})")
    log = "/tmp/opencode/iter8_matrix_P2.log"
    if os.path.exists(log):
        txt = open(log).read()
        rcs = re.findall(r"rc=(\d+)", txt)
        print(f"matrix log: {len(rcs)} END lines, non-zero rc = {sum(1 for x in rcs if x != '0')}")
        print(f"  PROF_DONE lines = {txt.count('[PROF_DONE]')}, AB-PROBE assertion failures = "
              f"{txt.count('[AB-PROBE]')}, Tracebacks = {txt.count('Traceback')}")
        skips = txt.count("(exists)")
        print(f"  skipped (pre-existing) dirs = {skips}")
        for ln in txt.splitlines():
            if "[PROF_DONE]" in ln:
                m = re.search(r"arm=(\w+) shape=(\w+).*variant_invoked=(\[[^\]]*\]) tout=(\[[^\]]*\])", ln)
                if m and m.group(2) in ("r2", "r1", "r5") and m.group(1) == "p2":
                    print(f"    {ln.strip()[:150]}")

    print()
    print("=" * 118)
    print("6. PREREGISTERED CRITERION EVALUATION (criteria locked 2026-10-19T16:10 in candidate_iter8_P2.json)")
    print("=" * 118)
    all_ok = True
    for shape in PRIMARY:
        if shape in verdicts:
            imp, ok, note = verdicts[shape]
            print(f"  PRIMARY {shape}: improvement {imp:+.2f}% -> {'PASS' if ok else 'FAIL'} ({note})")
            all_ok &= ok
    for shape in COVERED:
        if shape in PRIMARY:
            continue
        if shape in verdicts:
            imp, ok, note = verdicts[shape]
            print(f"  SECONDARY {shape}: improvement {imp:+.2f}% -> {'PASS' if ok else 'FAIL'} ({note})")
            all_ok &= ok
    for shape in OUT_OF_DOMAIN:
        if shape in verdicts:
            imp, ok, note = verdicts[shape]
            print(f"  OUT-OF-DOMAIN {shape}: {imp:+.2f}% -> {'PASS' if ok else 'FAIL'} ({note})")
            all_ok &= ok
    for shape, (d, ok) in main_ok.items():
        print(f"  MAIN-COST {shape}: {d:+.2f}% -> {'PASS' if ok else 'FAIL'} (limit +5%)")
        all_ok &= ok
    for shape, v in wait_verdict.items():
        ok = not v.startswith(("REJECT",))
        print(f"  WAITID {shape}: {'PASS' if ok else 'FAIL'} — {v}")
        all_ok &= ok
    for shape, ok in health.items():
        if not ok:
            print(f"  SESSION-HEALTH {shape}: FAIL (BASE out of the +/-3% anchor band)")
        all_ok &= ok
    print(f"\n  OVERALL: {'ALL PREREGISTERED CRITERIA PASS' if all_ok else 'AT LEAST ONE CRITERION FAILED'}")
    return all_ok


if __name__ == "__main__":
    d = collect()
    ok = report(d)
    sys.exit(0 if ok else 1)
