"""Iter7 A/B aggregator — P1 (B-matrix e2e chain) + P5 (main kernel).

Reuses parse_msprof_iter6.find_csvs/parse_kernel (artifact filter included).
P1 layout: msprof_op_iter7_P1_{shape}_{arm}_s{n}/OPPROF_*/{kernel}/{i}/*.csv
  instances: kernel_kernel:0 = transpose_qkv, main_kernel:0 = main,
  kernel_kernel:1 = transpose_out (launch order; block-dim cross-checked
  against the plan audit printed in the matrix log). r5 decode: main only.
  e2e = SUM of TaskDur over all instances in the run.
P5 layout: msprof_op_iter7_P5_{shape}_{arm}_s{n}/OPPROF_*/*.csv (main only)
  + wait-id telemetry (PipeUtilization) for the flag-protocol check.

Judgement math is printed but NOT decided here (criteria are preregistered
in candidate_iter7_P1/P5.json).

Run: python3 examples/gqa/perf_tuning/parse_iter7_ab.py
"""

import glob
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parse_msprof_iter6 import find_csvs, parse_kernel  # noqa: E402

P1_SHAPES = ["r1", "r2", "r3", "r4", "r5", "r6"]
P5_SHAPES = ["r4", "c7"]
WAITID = ["aic_scalar_wait_id1_time(us)", "aic_scalar_wait_id2_time(us)",
          "aic_scalar_wait_id5_time(us)", "aiv_scalar_wait_id0_time(us)",
          "aiv_scalar_wait_id3_time(us)", "aiv_scalar_wait_id4_time(us)"]


def parse_dir(base):
    kernels = find_csvs(base)
    out = {}
    for label, kinds in kernels.items():
        r = parse_kernel(kinds)
        if r:
            out[label] = r
    return out


def fmt_samples(vals):
    return "[" + ", ".join(f"{v:.2f}" for v in vals) + "]"


def stats(vals):
    m = statistics.mean(vals)
    spread = (max(vals) - min(vals)) / m * 100 if m else 0.0
    return m, spread


def p1_report(prefix="P1"):
    print("=" * 100)
    print(f"{prefix} — B-matrix e2e (Σ TaskDur over the kernel chain) + transpose_out audit")
    print("=" * 100)
    data = {}   # (shape, arm) -> list of per-sample dicts
    for shape in P1_SHAPES:
        for arm in ("base", "p1"):
            rows = []
            for n in (1, 2, 3):
                base = os.path.join(HERE, f"msprof_op_iter7_{prefix}_{shape}_{arm}_s{n}")
                if not os.path.isdir(base):
                    print(f"!! missing {os.path.basename(base)}")
                    continue
                inst = parse_dir(base)
                if not inst:
                    print(f"!! no CSVs {os.path.basename(base)}")
                    continue
                rec = {"s": n, "inst": inst}
                rec["e2e"] = sum(r["dur_us"] for r in inst.values())
                rows.append(rec)
            data[(shape, arm)] = rows

    print(f"\n{'shape':<6}{'arm':<6}{'n':<3}{'e2e mean':<10}{'spread%':<9}{'samples'}")
    means = {}
    for shape in P1_SHAPES:
        for arm in ("base", "p1"):
            rows = data[(shape, arm)]
            if not rows:
                continue
            e2e = [r["e2e"] for r in rows]
            m, sp = stats(e2e)
            means[(shape, arm)] = m
            print(f"{shape:<6}{arm:<6}{len(rows):<3}{m:<10.2f}{sp:<9.1f}{fmt_samples(e2e)}")

    print(f"\n{'shape':<6}{'base e2e':<10}{'p1 e2e':<10}{'delta us':<10}{'improvement %':<15}verdict-note")
    for shape in P1_SHAPES:
        if (shape, "base") in means and (shape, "p1") in means:
            b, p = means[(shape, "base")], means[(shape, "p1")]
            imp = (b - p) / b * 100
            note = "PRIMARY" if shape in ("r1", "r6") else ("guardrail" if shape != "r5" else "guardrail(decode)")
            print(f"{shape:<6}{b:<10.2f}{p:<10.2f}{b - p:<+10.2f}{imp:<+15.2f}{note}")

    # per-kernel breakdown: tout + main invariance
    print("\n--- transpose_out single-kernel + block dim (P1 attribution) ---")
    print(f"{'shape':<6}{'arm':<6}{'tout us mean[spread%]':<28}{'blk':<6}{'tqkv us':<10}{'main us mean[spread%]':<26}{'main blk'}")
    for shape in P1_SHAPES:
        for arm in ("base", "p1"):
            rows = data[(shape, arm)]
            tout = [r["inst"].get("kernel_kernel:1") for r in rows]
            tqkv = [r["inst"].get("kernel_kernel:0") for r in rows]
            main = [r["inst"].get("main_kernel:0") or r["inst"].get("main") for r in rows]
            def agg(rs):
                rs = [x for x in rs if x]
                if not rs:
                    return None
                vals = [x["dur_us"] for x in rs]
                m, sp = stats(vals)
                return m, sp, rs[0].get("block_dim")
            a_t, a_q, a_m = agg(tout), agg(tqkv), agg(main)
            f = lambda a: f"{a[0]:.2f}[{a[1]:.1f}%]" if a else "—"
            print(f"{shape:<6}{arm:<6}{f(a_t):<28}{(a_t[2] if a_t else '—'):<6}"
                  f"{(f'{a_q[0]:.2f}' if a_q else '—'):<10}{f(a_m):<26}{(a_m[2] if a_m else '—')}")

    # main-invariance check (preregistered guardrail)
    print("\n--- main kernel TaskDur ARM-vs-BASE (preregistered: |delta| <= 3% => untouched) ---")
    for shape in P1_SHAPES:
        mb = [r["inst"].get("main_kernel:0") or r["inst"].get("main") for r in data[(shape, "base")]]
        mp = [r["inst"].get("main_kernel:0") or r["inst"].get("main") for r in data[(shape, "p1")]]
        mb = [x["dur_us"] for x in mb if x]
        mp = [x["dur_us"] for x in mp if x]
        if mb and mp:
            d = (statistics.mean(mp) - statistics.mean(mb)) / statistics.mean(mb) * 100
            flag = "OK" if abs(d) <= 3.0 else "** >3% — INVESTIGATE (leak/session anomaly) **"
            print(f"{shape:<6} base={statistics.mean(mb):8.2f} p1={statistics.mean(mp):8.2f} delta={d:+6.2f}%  {flag}")


def p5_report():
    print("\n" + "=" * 100)
    print("P5 — main kernel TaskDur (flat_kv2 domain) + wait-id telemetry")
    print("=" * 100)
    means = {}
    tele = {}
    for shape in P5_SHAPES:
        for arm in ("base", "p5"):
            durs = []
            waits = {w: [] for w in WAITID}
            freqs = []
            for n in (1, 2, 3):
                base = os.path.join(HERE, f"msprof_op_iter7_P5_{shape}_{arm}_s{n}")
                if not os.path.isdir(base):
                    print(f"!! missing {os.path.basename(base)}")
                    continue
                inst = parse_dir(base)
                r = inst.get("main")
                if not r:
                    print(f"!! no main CSVs {os.path.basename(base)}")
                    continue
                durs.append(r["dur_us"])
                freqs.append(r.get("freq"))
                for w in WAITID:
                    v = r.get("waitid", {}).get(w)
                    if v:
                        waits[w].append(v[2])  # mean across rows
            if durs:
                m, sp = stats(durs)
                means[(shape, arm)] = m
                tele[(shape, arm)] = {w: statistics.mean(v) for w, v in waits.items() if v}
                print(f"{shape:<6}{arm:<6}n={len(durs)} TaskDur mean={m:.2f}us spread={sp:.1f}% "
                      f"min={min(durs):.2f} max={max(durs):.2f} samples={fmt_samples(durs)} "
                      f"freq={set(freqs)} blk={r.get('block_dim')}")

    print(f"\n{'shape':<6}{'base us':<10}{'p5 us':<10}{'delta us':<10}{'improvement %'}")
    for shape in P5_SHAPES:
        if (shape, "base") in means and (shape, "p5") in means:
            b, p = means[(shape, "base")], means[(shape, "p5")]
            print(f"{shape:<6}{b:<10.2f}{p:<10.2f}{b - p:<+10.2f}{(b - p) / b * 100:<+.2f}")

    print("\n--- wait-id telemetry (hot-row mean of per-sample means; flag protocol must NOT migrate) ---")
    for shape in P5_SHAPES:
        tb, tp = tele.get((shape, "base"), {}), tele.get((shape, "p5"), {})
        print(f"{shape}:")
        for w in WAITID:
            if w in tb or w in tp:
                print(f"   {w:<38} base={tb.get(w, float('nan')):8.2f} p5={tp.get(w, float('nan')):8.2f} "
                      f"delta={tp.get(w, 0) - tb.get(w, 0):+8.2f}")


def log_audit():
    print("\n" + "=" * 100)
    print("matrix log audit (rc, dispatch asserts, plan audit lines)")
    print("=" * 100)
    for seg in ("P1", "P1V2", "P5"):
        log = f"/tmp/opencode/iter7_matrix_{seg}.log"
        if not os.path.exists(log):
            print(f"!! {log} missing")
            continue
        txt = open(log).read()
        rc0 = len(re.findall(r"rc=0", txt))
        rcx = re.findall(r"rc=(\d+)", txt)
        bad = [r for r in rcx if r != "0"]
        warns = re.findall(r"\[WARN\].*", txt)
        asserts = re.findall(r"AssertionError.*", txt)
        fallbacks = re.findall(r"\[gqa\]\[fallback\].*", txt)
        plans = re.findall(r"\[P1_PLAN\].*", txt)
        print(f"{seg}: runs rc=0: {rc0}/{len(rcx)}; nonzero rc: {bad}; WARN lines: {len(warns)}; "
              f"AssertionErrors: {len(asserts)}; fallbacks: {len(fallbacks)}")
        for w in warns[:5]:
            print(f"   WARN: {w}")
        for a in asserts[:3]:
            print(f"   ASSERT: {a}")
        for f in fallbacks[:3]:
            print(f"   FALLBACK: {f}")
        from collections import Counter
        print(f"   plan census: {Counter(plans)}" if plans else "   plan census: (none)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "p5":
        p5_report()
    elif len(sys.argv) > 1 and sys.argv[1] == "p1":
        p1_report()
    elif len(sys.argv) > 1 and sys.argv[1] == "p1v2":
        p1_report(prefix="P1V2")
    else:
        p1_report()
        p5_report()
    log_audit()
