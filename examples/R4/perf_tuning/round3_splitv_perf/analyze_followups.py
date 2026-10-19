"""Round-3 follow-up analysis: (A) roofline attainment before/after rev3, (B) blast radius of a
guard relaxation on the test tables.

(A) ROOFLINE ATTAINMENT. `t_hw_us` is the full-core roofline floor published by
bench/hap_thw_model (unique-bytes / no-reload-factor convention, hap_lib.py:20). Attainment is
`t_hw_us / elapsed_us`. Both arms are scored against the SAME t_hw, so the ratio is directly
comparable -- with one口径 caveat that DESIGN.md section 15.3.2 requires to be stated explicitly:
rev3's split path reads `logits` from GM TWICE (K1A for the Sort side + K1B for the softmax side),
while the model charges it ONCE. The attainment of a split-path case is therefore slightly
OPTIMISTIC relative to a current-path case. The size of that bias is computed here per case from
the model's own `t_read` component instead of being hand-waved.

(B) GUARD-RELAXATION BLAST RADIUS. analyze_a_launch.py showed a relaxation can only flip cann-bench
cases 6 and 11. This part answers the implementation-risk question: how many entries of the
delivered test tables carry a shape that would flip, i.e. how many hardcoded `expect_use_split`
fields would go stale. Pure host arithmetic, nothing is compiled and nothing is modified.

Usage: analyze_followups.py <base1_fused.json> <base1_mask.json> <cand_fused.json> <cand_mask.json>
                            [out.json]
"""
import ast
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R4 = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (R4, os.path.join(R4, "bench", "hap_thw_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r4_fused_sampling as M  # noqa: E402

SPLIT_CASES = {3, 4, 7, 8, 12, 17}
A_SHIPPED, A_RELAXED = 8.0, 1.54
out_path = sys.argv[5] if len(sys.argv) > 5 else os.path.join(HERE, "followups_analysis.json")

GATES = json.load(open(os.path.join(R4, "bench", "hap_thw_model", "gates_report.json")))
THW = {t: {r["case_id"]: r for r in GATES["results"][t]}
       for t in ("r4_fused_sampling", "r4_mask_sample")}


def elapsed(path):
    d = json.load(open(path))
    op = d["operators"][0]
    cs = op["cases"]
    cs = list(cs.values()) if isinstance(cs, dict) else cs
    return {int(str(c["case_id"]).rsplit("_", 1)[-1]): c["elapsed_us"] for c in cs}


BASE = {"r4_fused_sampling": elapsed(sys.argv[1]), "r4_mask_sample": elapsed(sys.argv[2])}
CAND = {"r4_fused_sampling": elapsed(sys.argv[3]), "r4_mask_sample": elapsed(sys.argv[4])}

print("=" * 112)
print("(A) ROOFLINE ATTAINMENT  attain% = 100 * t_hw_us / elapsed_us   (same t_hw for both arms)")
print("=" * 112)
print(f"{'task':>20} {'case':>4} {'path':>7} {'t_hw_us':>8} {'t_read':>8} {'t_vect':>8} | "
      f"{'base_us':>9} {'cand_us':>9} | {'att_base%':>9} {'att_cand%':>9} {'x':>6} | "
      f"{'t_hw_2pass':>10} {'att2p_c%':>9} {'bound2p':>8}")
rows = []
for task in ("r4_fused_sampling", "r4_mask_sample"):
    for ci in sorted(THW[task]):
        g = THW[task][ci]
        thw, tread, tvec = float(g["t_hw_us"]), float(g["t_read"]), float(g["t_vector"])
        tcube, twrite = float(g.get("t_cube", 0.0)), float(g.get("t_write", 0.0))
        e0, e1 = BASE[task][ci], CAND[task][ci]
        a0, a1 = 100.0 * thw / e0, 100.0 * thw / e1
        split = ci in SPLIT_CASES
        # DESIGN.md section 15.3.2口径 caveat, computed instead of hand-waved: the published t_hw
        # charges ONE logits read (hap_lib unique-bytes / no-reload-factor convention) while the
        # rev3 split path reads logits TWICE (K1A for the Sort side, K1B for the softmax side).
        # t_hw_2pass re-derives the floor with that reload charged, exactly as the model would if
        # reloads were in scope: max over the same 4 components, then the same >=1.0 clip.
        raw2 = max(tcube, 2.0 * tread, twrite, tvec)
        # round exactly the way the model publishes t_hw (GATE 3 asserts t_hw == round(max(raw,1.0),4));
        # without this, the stored unrounded t_vector differs from the rounded t_hw_us in the last
        # digits and every split case would be flagged as "changed" by pure representation noise.
        thw2 = round(max(raw2, 1.0), 4) if split else thw
        bound2 = ("vector" if tvec >= 2.0 * tread else "READ") if split else g.get("bottleneck", "?")
        a1_2p = 100.0 * thw2 / e1
        rows.append(dict(task=task, case=ci, path="SPLIT" if split else "CURRENT", t_hw_us=thw,
                         t_read_us=tread, t_vector_us=tvec, t_cube_us=tcube, t_write_us=twrite,
                         base_us=e0, cand_us=e1, attain_base_pct=round(a0, 4),
                         attain_cand_pct=round(a1, 4), attain_ratio=round(a1 / a0, 5),
                         t_hw_2pass_us=round(thw2, 4), attain_cand_2pass_pct=round(a1_2p, 4),
                         bound_under_2pass=bound2,
                         two_pass_changes_floor=bool(split and abs(thw2 - thw) > 1e-4)))
        print(f"{task.replace('r4_', ''):>20} {ci:>4} {'SPLIT' if split else 'curr':>7} "
              f"{thw:>8.4f} {tread:>8.4f} {tvec:>8.4f} | {e0:>9.2f} {e1:>9.2f} | {a0:>9.3f} "
              f"{a1:>9.3f} {a1 / a0:>6.3f} | {thw2:>10.4f} {a1_2p:>9.3f} {bound2:>8}")

def _gm(v):
    import math
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


for task in ("r4_fused_sampling", "r4_mask_sample"):
    sel = [r for r in rows if r["task"] == task]
    sp = [r for r in sel if r["path"] == "SPLIT"]
    cu = [r for r in sel if r["path"] == "CURRENT"]
    print(f"\n  {task}:")
    print(f"    attainment geomean  ALL 18 : base {_gm([r['attain_base_pct'] for r in sel]):.4f}% "
          f"-> cand {_gm([r['attain_cand_pct'] for r in sel]):.4f}%  "
          f"({_gm([r['attain_cand_pct'] for r in sel]) / _gm([r['attain_base_pct'] for r in sel]):.4f}x)")
    print(f"    attainment geomean  SPLIT 6: base {_gm([r['attain_base_pct'] for r in sp]):.4f}% "
          f"-> cand {_gm([r['attain_cand_pct'] for r in sp]):.4f}%  "
          f"({_gm([r['attain_cand_pct'] for r in sp]) / _gm([r['attain_base_pct'] for r in sp]):.4f}x)")
    print(f"    attainment geomean  CURR  12: base {_gm([r['attain_base_pct'] for r in cu]):.4f}% "
          f"-> cand {_gm([r['attain_cand_pct'] for r in cu]):.4f}%")
    ch = [r for r in sp if r["two_pass_changes_floor"]]
    print(f"    2-pass GM-read口径 (section 15.3.2): {len(ch)}/{len(sp)} split cases have their t_hw "
          f"floor CHANGED by charging the second logits read:")
    for r in ch:
        print(f"        case{r['case']:<3} t_hw {r['t_hw_us']:.4f} -> {r['t_hw_2pass_us']:.4f} us "
              f"({r['t_hw_2pass_us'] / r['t_hw_us']:.3f}x), bound {r['bound_under_2pass']}, "
              f"attain_cand {r['attain_cand_pct']:.3f}% -> {r['attain_cand_2pass_pct']:.3f}%")
    if not ch:
        print(f"        none -- every split case stays vector-bound with 2*t_read still below "
              f"t_vector, so the published t_hw is unaffected by the reload.")
    un = [r for r in sp if not r["two_pass_changes_floor"]]
    if un:
        print(f"        unchanged: cases {[r['case'] for r in un]} (still vector-bound under the "
              f"2-pass accounting)")

# ---------------- (B) guard-relaxation blast radius on the test tables ----------------
print("\n" + "=" * 112)
print("(B) GUARD-RELAXATION BLAST RADIUS on the delivered test tables (nothing is modified)")
print("=" * 112)
TEST = os.path.join(R4, "test_r4_fused_sampling.py")
tree = ast.parse(open(TEST, encoding="utf-8").read())
entries = []
for node in tree.body:
    if not isinstance(node, ast.Assign):
        continue
    for tgt in node.targets:
        if not isinstance(tgt, ast.Name) or not tgt.id.isupper():
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            continue
        for el in node.value.elts:
            if not isinstance(el, ast.Tuple) or len(el.elts) < 5:
                continue
            try:
                vals = [ast.literal_eval(x) for x in el.elts[:5]]
            except Exception:
                continue
            name = vals[0]
            nums = vals[1:5]
            if not (isinstance(name, str) and all(isinstance(x, int) for x in nums)):
                continue
            B, V, K, BV = nums
            if not (B > 0 and V > 0 and BV > 0 and K in (128, 256)):
                continue
            entries.append((tgt.id, name, B, V, K, BV))

seen, flips, tot = set(), [], 0
for table, name, B, V, K, BV in entries:
    key = (table, name)
    if key in seen:
        continue
    seen.add(key)
    tot += 1
    u8 = M._r4_split_plan(B, V, K, BV)
    saved = M._SPLIT_A_LAUNCH_US
    M._SPLIT_A_LAUNCH_US = A_RELAXED
    u1 = M._r4_split_plan(B, V, K, BV)
    M._SPLIT_A_LAUNCH_US = saved
    if u1[0] and not u8[0]:
        flips.append(dict(table=table, name=name, B=B, V=V, K_MAX=K, BV=BV,
                          plan_shipped=list(u8), plan_relaxed=list(u1)))
print(f"  test-file case entries with a (B, V, K_MAX, BV) shape scanned : {tot}")
print(f"  entries whose path would FLIP current->split at a_launch={A_RELAXED} : {len(flips)}")
for f in flips:
    print(f"    {f['table']:22s} {f['name']:38s} B={f['B']:<4} V={f['V']:<7} "
          f"K={f['K_MAX']} BV={f['BV']}  S={f['plan_relaxed'][1]} NBS={f['plan_relaxed'][2]} "
          f"S_TAIL={f['plan_relaxed'][3]}")
print(f"\n  KEY STRUCTURAL FACT: `_SPLIT_A_LAUNCH_US` is read ONLY inside the cost guard, AFTER")
print(f"  S/NBS/S_TAIL have been derived. So a relaxation flips the `use_split` BOOLEAN and cannot")
print(f"  change S/NBS/S_TAIL for any shape -> it admits NO new JIT specialization geometry, only")
print(f"  routes existing ones. The cb6/cb11 geometries (S=3/NBS=1/S_TAIL=2 and S=2/NBS=1/S_TAIL=-1)")
print(f"  are already compiled and statically asserted clean in the Stage-2 static gate")
print(f"  (perf_tuning/rev3_probes/stage2/static_gate_all.log: '--- cb6:' / '--- cb11:', FAILURES: 0).")
print(f"  RESIDUAL RISK if the guard is relaxed: {len(flips)} test-table entries carry a hardcoded")
print(f"  `expect_use_split` field that would go stale (False while the code now splits), so the")
print(f"  test file must be updated in the same change -- a test edit that needs Orchestrator approval.")

json.dump(dict(attainment=rows, guard_blast_radius=dict(
    scanned_test_entries=tot, flipping_entries=flips,
    a_launch_shipped=A_SHIPPED, a_launch_relaxed=A_RELAXED)),
    open(out_path, "w"), indent=1)
print(f"\nwrote {out_path}")
