"""Item-A requirement 5: quantify and DISCLOSE the SPAN consequence for case 6 / case 11.

The guard relaxation makes those two cases go from 2 launches to 4 (fused) / 3 (mask). The SCORED
口径 -- the sum of per-step kernel Durations, which is the contract's performance metric and what
the user ruled to keep -- improves (measured: 1.44x / 1.31x / 1.25x / 1.15x in round 3's probe).
The WALL-CLOCK SPAN does not necessarily, because every extra launch pays a per-launch HOST cost.

Round 3's read-only investigation settled that host cost (attribution_report.md, 381 lines):
  bare eager                    H = 200.5115 us/launch   (probe3 L0)
  cann-bench harness guards     H = 433.9085 us/launch   (probe3 L3)
  harness guards + msprof       H = 499.6380 us/launch   (probe3 L5)
  ACL-graph replay residual       =  15.5880 us/launch   (probe4/probe6)
and established that ~135-158 us of it is tilelang framework dispatch (single largest primitive
Analyzer() = 122.603 us median), ~17-38 us ACL launch residual, 233.40 us harness guards, 65.73 us
msprof instrumentation -- while OUR OWN wrapper is < 2% (C_pct 0.97-1.97%). An empty tilelang kernel
already costs 160.6-180.5 us/launch, so this is neither R4-specific nor wrapper-induced.

MODEL (verbatim from round3_host_overhead/analyze_corrected_span.py, which was validated two ways
before use -- V1 median abs err 0.058% against round-3's measured msprof spans, V2 7.583% against
probe3's directly measured NPU-event spans):

    start_1 = H                                   (queue empty)
    start_i = max(start_{i-1} + d_{i-1}, i*H)     (host-bound when i*H wins)
    span_msprof = end_n - start_1                 (round-3 definition: first START -> last END)
    span_event  = span_msprof + H                 (an NPU event before launch 1 also pays H)
    span_graph  = sum(d_i) + n * graph_residual   (ACL graph: no per-launch host dispatch)

Device durations d_i come from the item-A msprof CSVs via ksplit_fine_itemA.py (authoritative).
The spans below are MODEL ESTIMATES and are labelled as such; they are a DISCLOSURE, not an
adoption criterion -- the user ruled to keep rev3 with device time as the contract口径.

Usage: span_itemA.py [out.json]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
H = {"bare_eager_us": 200.5115, "harness_guards_us": 433.9085,
     "harness_guards_msprof_us": 499.6380}
GRAPH_RESIDUAL_US = 15.5880
FLIP = [6, 11]
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "span_consequence.json")


def span(durs, h):
    """Device-timeline recursion; returns (span_msprof, span_event, n_host_bound)."""
    if not durs:
        return float("nan"), float("nan"), 0
    start, nb = [], 0
    prev_end = 0.0
    for i, d in enumerate(durs, start=1):
        s = h if i == 1 else max(prev_end, i * h)
        if i > 1 and i * h > prev_end:
            nb += 1
        start.append(s)
        prev_end = s + d
    span_msprof = prev_end - start[0]
    return round(span_msprof, 3), round(span_msprof + h, 3), nb


def get(arm, task, ci):
    f = os.path.join(HERE, f"ksplit_{arm}_{task}.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    c = d.get("per_case", {}).get(str(ci))
    if not c:
        return None
    names = c["names"]
    pk = c["per_kernel_us"]
    # per_kernel_us is keyed by kernel NAME; rebuild the launch-ordered duration list. Duplicate
    # names (K1B appears twice in the static gate, never in a launch chain) would be ambiguous, so
    # assert the launch order is recoverable from the name list.
    if len(set(names)) != len(names):
        raise RuntimeError(f"{arm}/{task}/case{ci}: duplicate kernel names {names} -- cannot "
                           f"rebuild the launch order from per_kernel_us")
    durs = [pk[n] for n in names]
    return dict(names=names, durs=durs, total_us=c["total_us"], n_launch=c["n_launch"],
                path=c["path"], S=c["S"], NBS=c["NBS"], S_TAIL=c["S_TAIL"],
                blocks=c["blocks"], ok_blk=c["ok_blk"], ok_sum=c["ok_sum"], ok_p2=c["ok_p2"])


res = {}
print("=" * 116)
print("ITEM-A SPAN CONSEQUENCE for the two newly split cases  (MODEL ESTIMATES, disclosure only)")
print("=" * 116)
print(f"  per-launch host cost H: bare eager {H['bare_eager_us']:.4f} us | harness guards "
      f"{H['harness_guards_us']:.4f} us | guards+msprof {H['harness_guards_msprof_us']:.4f} us")
print(f"  ACL-graph replay residual: {GRAPH_RESIDUAL_US:.4f} us/launch\n")

missing = []
for task in ("fused", "mask"):
    for ci in FLIP:
        b = get("base1", task, ci)
        c = get("cand", task, ci)
        if b is None or c is None:
            missing.append((task, ci))
            continue
        key = f"{task}_case{ci}"
        rec = dict(task=task, case=ci, base=b, cand=c)
        print(f"{'-' * 116}")
        print(f"TASK {task:5s}  case {ci:>2}   launches {b['n_launch']} -> {c['n_launch']}   "
              f"path {b['path']} -> {c['path']} (S={c['S']} NBS={c['NBS']} S_TAIL={c['S_TAIL']})")
        print(f"  base1 (a=8.0)  {b['names']}  durs={b['durs']}  scored_sum={b['total_us']:.3f} us  "
              f"blocks={b['blocks']}")
        print(f"  cand  (a=1.54) {c['names']}  durs={c['durs']}  scored_sum={c['total_us']:.3f} us  "
              f"blocks={c['blocks']}")
        print(f"  SCORED口径 (contract metric) : {b['total_us']:.3f} -> {c['total_us']:.3f} us  "
              f"= {b['total_us'] / c['total_us']:.4f}x  ({100 * (c['total_us'] / b['total_us'] - 1):+.2f}%)")
        rec["scored_x"] = round(b["total_us"] / c["total_us"], 5)
        rec["scored_delta_pct"] = round(100 * (c["total_us"] / b["total_us"] - 1), 3)
        spans = {}
        print(f"  {'assumption':<26} {'H us/launch':>12} | {'span_base':>10} {'span_cand':>10} "
              f"{'SPAN x':>8} {'delta%':>8} | {'event_base':>10} {'event_cand':>10} "
              f"{'EVENT x':>8} | host-bound launches")
        for label, h in H.items():
            sb, eb, nbb = span(b["durs"], h)
            sc, ec, nbc = span(c["durs"], h)
            spans[label] = dict(H_us=h, span_base_us=sb, span_cand_us=sc,
                                span_x=round(sb / sc, 4) if sc else None,
                                span_delta_pct=round(100 * (sc / sb - 1), 2) if sb else None,
                                event_base_us=eb, event_cand_us=ec,
                                event_x=round(eb / ec, 4) if ec else None,
                                host_bound_base=nbb, host_bound_cand=nbc)
            print(f"  {label:<26} {h:>12.4f} | {sb:>10.2f} {sc:>10.2f} {sb / sc:>8.4f} "
                  f"{100 * (sc / sb - 1):>+8.2f} | {eb:>10.2f} {ec:>10.2f} {eb / ec:>8.4f} | "
                  f"{nbb}/{b['n_launch']} -> {nbc}/{c['n_launch']}")
        gb = round(sum(b["durs"]) + b["n_launch"] * GRAPH_RESIDUAL_US, 3)
        gc = round(sum(c["durs"]) + c["n_launch"] * GRAPH_RESIDUAL_US, 3)
        spans["acl_graph"] = dict(residual_us_per_launch=GRAPH_RESIDUAL_US, span_base_us=gb,
                                  span_cand_us=gc, span_x=round(gb / gc, 4),
                                  span_delta_pct=round(100 * (gc / gb - 1), 2))
        print(f"  {'acl_graph (captured)':<26} {GRAPH_RESIDUAL_US:>12.4f} | {gb:>10.2f} {gc:>10.2f} "
              f"{gb / gc:>8.4f} {100 * (gc / gb - 1):>+8.2f} | {'-':>10} {'-':>10} {'-':>8} | "
              f"0 -> 0 (no per-launch host dispatch under replay)")
        rec["spans"] = spans
        res[key] = rec

if missing:
    print(f"\n*** MISSING ksplit inputs for {missing} -- run run_ksplit_itemA.sh first ***")

print(f"\n{'=' * 116}")
print("READING: under BARE EAGER and under the PROFILED HARNESS the extra launches cost more host time")
print("than the device saves, so the SPAN ratio is < 1 even though the SCORED ratio is > 1. Under ACL")
print("GRAPH capture the per-launch host dispatch disappears and the span follows the device time.")
print("This is the SAME tradeoff round 3 documented for the 6 already-split cases (section 9 of")
print("optimization_log_round3.md); item A extends it to 2 more cases. The user ruled to keep rev3")
print("with DEVICE TIME as the contract口径, so this is a disclosure, not a rejection reason. M1 (ACL")
print("graph capture + D2D copy-in) remains the only mitigation that removes the root cause, and it is")
print("explicitly OUT OF SCOPE for this item.")
print("=" * 116)

json.dump(dict(_meta=dict(
    description="SPAN consequence of routing case 6 / case 11 onto the split path (2 -> 4/3 launches)",
    model="start_1=H; start_i=max(end_{i-1}, i*H); span_msprof=end_n-start_1; span_event=span+H; "
          "span_graph=sum(d_i)+n*residual  (verbatim from round3_host_overhead/"
          "analyze_corrected_span.py, V1 0.058% / V2 7.583% validated)",
    H_us=H, graph_residual_us_per_launch=GRAPH_RESIDUAL_US,
    device_durations_source="item-A msprof kernel_details.csv via ksplit_fine_itemA.py",
    status="MODEL ESTIMATES -- disclosure only, NOT an adoption criterion",
    missing=missing), cases=res), open(OUT, "w"), indent=1)
print(f"wrote {OUT}")
