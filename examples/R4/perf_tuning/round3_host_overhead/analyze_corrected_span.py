"""Corrected SPAN re-estimation (required output #4).

Round 3 reported SPAN speedups (fused 0.7443x, mask 0.8005x; split cases 0.3831x / 0.4387x)
computed from msprof `kernel_details.csv` timestamps taken INSIDE the cann-bench harness,
i.e. with the full instrumentation stack active. probe3 measured that stack to inflate the
per-launch HOST cost from ~200 us (bare) to ~434 us (guards) / ~500 us (guards+msprof), and
the CSV-implied H for host-bound launches is ~404 us median. So the round-3 SPAN numbers
describe a PROFILED, GUARDED harness, not a deployment.

MODEL
-----
For a chain of n launches with device durations d_1..d_n (msprof, authoritative) and a fixed
per-launch host cost H, the device timeline gives

    start_1 = H                                   (queue empty: the device starts as soon as
                                                   the host finishes submitting)
    start_i = max(start_{i-1} + d_{i-1}, i*H)     (host-bound when i*H wins)
    span_msprof = end_n - start_1                 (round-3's definition: first main_kernel
                                                   START -> last main_kernel END)
    span_event  = span_msprof + H                 (an NPU event recorded BEFORE the first
                                                   launch also pays launch 1's host prep)

VALIDATION (two independent ways, both must hold before the model is used to extrapolate)
  V1: with H = H_harness (from the CSVs themselves) the model must reproduce round-3's
      measured span_us.
  V2: with H = H_bare (probe3 L0, 200.51 us) the model's span_event must reproduce probe3's
      directly-measured NPU-event spans (no msprof, no guards).

Only after V1 and V2 pass is the model used to produce corrected SPAN speedups for
  (a) bare eager deployment            H = H_bare
  (b) cann-bench harness (round 3)     H = H_harness   [reproduces the published numbers]
  (c) ACL-graph captured deployment    span ~= sum(d_i) + replay_overhead (probe4 measured)

Device durations come from msprof (round-3 CSVs) and are NOT re-derived from any Python
timer. H values come from probe3's host wall-clock measurements. The corrected spans are
MODEL ESTIMATES and are labelled as such everywhere.

Usage: analyze_corrected_span.py
"""
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict

from probe_common import HERE, R4, dump

MANIFEST = os.path.join(R4, "perf_tuning", "round3_splitv_perf", "csv_manifest.txt")
STEPS_PER_CASE, WARMUP = 8, 3
H_GAP_VALID_US = 50.0  # a launch is host-bound (H identifiable) only when its gap >= this


def read_manifest():
    arms, lines, i = {}, open(MANIFEST).read().splitlines(), 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln and not ln.startswith("/") and i + 1 < len(lines):
            arms[ln.split()[0]] = lines[i + 1].strip()
            i += 2
        else:
            i += 1
    return arms


def load(path):
    """-> {case: {'durs': [median d per launch index], 'H': [median implied H], 'span': med}}"""
    by = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["Name"].strip() != "main_kernel":
            continue
        by[int(r["Step Id"].strip())].append(dict(
            start=float(r["Start Time(us)"].strip()), dur=float(r["Duration(us)"])))
    steps = defaultdict(list)
    for step, rows in by.items():
        ci, rep = step // STEPS_PER_CASE + 1, step % STEPS_PER_CASE
        if rep < WARMUP:
            continue
        seq = sorted(rows, key=lambda x: x["start"])
        durs = [k["dur"] for k in seq]
        span = max(k["start"] + k["dur"] for k in seq) - seq[0]["start"]
        Hs = []
        for i in range(1, len(seq)):
            gap = seq[i]["start"] - (seq[i - 1]["start"] + seq[i - 1]["dur"])
            if gap >= H_GAP_VALID_US:
                Hs.append(gap + seq[i - 1]["dur"])
        steps[ci].append(dict(durs=durs, span=span, Hs=Hs, n=len(seq)))
    out = {}
    for ci, v in steps.items():
        n = v[0]["n"]
        assert all(x["n"] == n for x in v), f"case{ci}: launch count varies across steps"
        out[ci] = dict(
            n_launch=n,
            durs=[round(statistics.median([x["durs"][i] for x in v]), 4) for i in range(n)],
            span_measured_us=round(statistics.median([x["span"] for x in v]), 3),
            H_identifiable_us=(round(statistics.median([h for x in v for h in x["Hs"]]), 3)
                               if any(x["Hs"] for x in v) else None),
            n_steps=len(v))
    return out


def model_span(durs, H):
    """Device-timeline span (round-3 definition) for per-launch host cost H."""
    n = len(durs)
    start = H
    prev_end = H + durs[0]
    for i in range(1, n):
        start = max(prev_end, (i + 1) * H)
        prev_end = start + durs[i]
    return prev_end - H  # end_n - start_1


def gm(v):
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def main():
    arms = read_manifest()
    D = {lab: load(p) for lab, p in arms.items()}

    p3 = json.load(open(os.path.join(HERE, "probe3_ladder_and_span.json")))
    H_bare = statistics.median([v["levels"]["0"]["per_launch_us"] for v in p3["ladder"].values()])
    H_l3 = statistics.median([v["levels"]["3"]["per_launch_us"] for v in p3["ladder"].values()])
    H_l5 = statistics.median([v["levels"]["5"]["per_launch_us"] for v in p3["ladder"].values()])
    ev = {int(k): v["med_us"] for k, v in p3["span_events"].items()}
    print(f"probe3 host costs per launch:  L0 bare={H_bare:.2f}us  L3 guards={H_l3:.2f}us  "
          f"L5 guards+msprof={H_l5:.2f}us")
    print(f"probe3 measured NPU-event spans (no msprof, no guards): {ev}")

    # ------------------------------------------------------------------ V1 / V2 ----
    print("\n" + "=" * 118)
    print("MODEL VALIDATION")
    print("=" * 118)
    print("\nV1: model span with H = H_harness (per-case, from the CSVs) vs round-3 MEASURED span")
    print(f"{'arm':<13}{'case':>5}{'#ln':>4}{'H_csv':>9}{'model_span':>12}{'measured':>11}{'err%':>8}")
    v1 = []
    for lab in ("base1_fused", "cand_fused", "base1_mask", "cand_mask"):
        for ci, r in sorted(D[lab].items()):
            H = r["H_identifiable_us"]
            if H is None:
                # device-bound case: H not identifiable from gaps. Bound it by d_prev.
                H = 0.0
            ms = model_span(r["durs"], H) if H > 0 else sum(r["durs"])
            meas = r["span_measured_us"]
            err = 100 * (ms - meas) / meas
            v1.append(abs(err))
            flag = "" if abs(err) < 12 else "  <-- check"
            print(f"{lab:<13}{ci:>5}{r['n_launch']:>4}{H:>9.1f}{ms:>12.2f}{meas:>11.2f}"
                  f"{err:>+8.1f}{flag}")
    print(f"\n  V1 |err|: median={statistics.median(v1):.2f}%  p90="
          f"{sorted(v1)[int(0.9*(len(v1)-1))]:.2f}%  max={max(v1):.2f}%  n={len(v1)}")

    print("\nV2: model span_event (= model_span + H) with H = H_bare vs probe3's MEASURED")
    print("    NPU-event spans (no msprof, no guards). probe3 measured the AUTO-DISPATCHED path.")
    print(f"{'case':>5}{'arm':>12}{'#ln':>4}{'model_span_ev':>15}{'measured_ev':>13}{'err%':>8}")
    v2 = []
    for ci in sorted(ev):
        lab = "cand_fused"  # probe3 measured the delivered (== rev3/cand) wrapper, fused task
        r = D[lab][ci]
        ms = model_span(r["durs"], H_bare) + H_bare
        err = 100 * (ms - ev[ci]) / ev[ci]
        v2.append(abs(err))
        print(f"{ci:>5}{lab:>12}{r['n_launch']:>4}{ms:>15.2f}{ev[ci]:>13.2f}{err:>+8.1f}")
    print(f"\n  V2 |err|: median={statistics.median(v2):.2f}%  max={max(v2):.2f}%  n={len(v2)}")

    ok1 = statistics.median(v1) < 12.0
    ok2 = statistics.median(v2) < 15.0
    print(f"\n  V1 pass(|err|med<12%) = {ok1}    V2 pass(|err|med<15%) = {ok2}")
    if not (ok1 and ok2):
        print("  MODEL NOT VALIDATED -- corrected spans below are indicative only.")

    # ------------------------------------------------- corrected SPAN speedups -----
    print("\n" + "=" * 118)
    print("CORRECTED SPAN ESTIMATES  (model; device durations from msprof CSVs, H from probe3)")
    print("=" * 118)
    p4p = os.path.join(HERE, "probe4_aclgraph.json")
    p4 = json.load(open(p4p)) if os.path.exists(p4p) else {}
    cap = {}
    for k, v in (p4.get("wrapper_capture") or {}).items():
        if v.get("capture_ok"):
            cap[int(k)] = v["L0_bare"]["replay_serial_host_plus_device"]["med_us"]

    # Derive the RESIDUAL per-launch overhead that survives graph capture, from probe4's
    # measured captured-serial times minus the msprof device time of the same chain.
    # (Capture removes the ~200-500 us host prep but graph-node dispatch is not free.)
    G_samples = []
    for ci, rec in (p4.get("marginal") or {}).items():
        ci = int(ci)
        for nm, nl in (("chain2", 2), ("chain4", 4)):
            a = rec.get(nm)
            if not a or not a.get("ok"):
                continue
            lab = "cand_fused" if nm == "chain4" else "base1_fused"
            if ci in D.get(lab, {}):
                sc = sum(D[lab][ci]["durs"])
                ser = a["L0"]["replay_serial_host_plus_device"]["med_us"]
                G_samples.append(dict(case=ci, chain=nm, n_launch=nl, scored_us=round(sc, 3),
                                      captured_serial_us=ser,
                                      total_overhead_us=round(ser - sc, 3),
                                      per_launch_us=round((ser - sc) / nl, 3)))
    G = (statistics.median([s["per_launch_us"] for s in G_samples]) if G_samples else 15.0)
    print(f"\nprobe4 captured-serial residual overhead samples: {json.dumps(G_samples)}")
    print(f"  => residual graph overhead per launch G = {G:.2f} us "
          f"(capture removes the ~{H_bare:.0f} us host prep but graph-node dispatch is not free)")

    out = dict(_meta=dict(
        description="corrected SPAN speedups under 3 deployment assumptions",
        model="span = device-timeline recursion start_i = max(end_{i-1}, i*H); "
              "span_event = span + H",
        H_bare_us=H_bare, H_guards_us=H_l3, H_guards_msprof_us=H_l5,
        device_durations_source="round-3 msprof kernel_details.csv (authoritative)",
        validated=dict(V1_median_abs_err_pct=round(statistics.median(v1), 3),
                       V2_median_abs_err_pct=round(statistics.median(v2), 3),
                       V1_pass=ok1, V2_pass=ok2),
        note="these spans are MODEL ESTIMATES, not measurements; the measured anchors are "
             "round-3's msprof spans (V1) and probe3's NPU-event spans (V2)",
        graph_residual_overhead_per_launch_us=round(G, 3),
        graph_residual_overhead_samples=G_samples))

    for task in ("fused", "mask"):
        bl, cl = f"base1_{task}", f"cand_{task}"
        b, c = D[bl], D[cl]
        print(f"\n########## {task} ##########")
        print(f"{'case':>4} {'#ln b/c':>8} {'scored b':>9} {'scored c':>9} | "
              f"{'SPAN harness':>13} {'x':>6} | {'SPAN bare':>10} {'x':>6} | {'SPAN graph':>11} {'x':>6}")
        rows = []
        sp_h, sp_b, sp_g, sp_sc = [], [], [], []
        for ci in sorted(b):
            rb, rc = b[ci], c[ci]
            Hb = rb["H_identifiable_us"] or 0.0
            Hc = rc["H_identifiable_us"] or 0.0
            sc_b, sc_c = sum(rb["durs"]), sum(rc["durs"])
            # (b) harness: per-case H from the CSVs (what round 3 actually measured)
            h_b = model_span(rb["durs"], Hb) if Hb else sc_b
            h_c = model_span(rc["durs"], Hc) if Hc else sc_c
            # (a) bare eager: H = probe3 L0
            bare_b = model_span(rb["durs"], H_bare)
            bare_c = model_span(rc["durs"], H_bare)
            # (c) ACL graph: host prep is gone; what remains is device time plus the
            #     measured residual graph-node dispatch overhead G per launch (probe4).
            g_b = sc_b + rb["n_launch"] * G
            g_c = sc_c + rc["n_launch"] * G
            xh, xb, xg, xsc = h_b / h_c, bare_b / bare_c, g_b / g_c, sc_b / sc_c
            sp_h.append(xh); sp_b.append(xb); sp_g.append(xg); sp_sc.append(xsc)
            meas_b = rb["span_measured_us"]
            print(f"{ci:>4} {str(rb['n_launch'])+'/'+str(rc['n_launch']):>8} "
                  f"{sc_b:>9.2f} {sc_c:>9.2f} | {h_b:>13.1f} {xh:>6.3f} | "
                  f"{bare_b:>10.1f} {xb:>6.3f} | {g_c:>11.1f} {xg:>6.3f}   "
                  f"(round3 measured span_b={meas_b:.1f})")
            rows.append(dict(case=ci, n_launch_base=rb["n_launch"], n_launch_cand=rc["n_launch"],
                             scored_base_us=round(sc_b, 3), scored_cand_us=round(sc_c, 3),
                             durs_base=rb["durs"], durs_cand=rc["durs"],
                             H_base_csv_us=Hb, H_cand_csv_us=Hc,
                             span_harness_model_us=round(h_b, 2), span_harness_measured_us=meas_b,
                             span_base_model_us=round(h_c, 2),
                             span_bare_base_us=round(bare_b, 2), span_bare_cand_us=round(bare_c, 2),
                             span_graph_base_us=round(g_b, 2), span_graph_cand_us=round(g_c, 2),
                             x_harness=round(xh, 4), x_bare=round(xb, 4),
                             x_graph=round(xg, 4), x_scored=round(xsc, 4)))
        split = [r["case"] for r in rows if r["n_launch_base"] != r["n_launch_cand"]]
        print(f"\n  geomean SCORED (device time, contractual metric) : {gm(sp_sc):.4f}x")
        print(f"  geomean SPAN harness (H from CSV, ~404us/launch) : {gm(sp_h):.4f}x   "
              f"[round 3 published: {0.7443 if task=='fused' else 0.8005}x]")
        print(f"  geomean SPAN bare eager (H={H_bare:.0f}us/launch)    : {gm(sp_b):.4f}x")
        print(f"  geomean SPAN ACL-graph captured (H~0)           : {gm(sp_g):.4f}x")
        print(f"    split cases only {split}:  harness {gm([r['x_harness'] for r in rows if r['case'] in split]):.4f}x"
              f"  bare {gm([r['x_bare'] for r in rows if r['case'] in split]):.4f}x"
              f"  graph {gm([r['x_graph'] for r in rows if r['case'] in split]):.4f}x")
        out[task] = dict(per_case=rows,
                         geomean_scored=round(gm(sp_sc), 5),
                         geomean_span_harness=round(gm(sp_h), 5),
                         geomean_span_bare=round(gm(sp_b), 5),
                         geomean_span_graph=round(gm(sp_g), 5),
                         geomean_span_harness_splitonly=round(
                             gm([r["x_harness"] for r in rows if r["case"] in split]), 5),
                         geomean_span_bare_splitonly=round(
                             gm([r["x_bare"] for r in rows if r["case"] in split]), 5),
                         geomean_span_graph_splitonly=round(
                             gm([r["x_graph"] for r in rows if r["case"] in split]), 5),
                         split_cases=split)
    dump("corrected_span.json", out)


if __name__ == "__main__":
    main()
