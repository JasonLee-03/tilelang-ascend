"""Item-A requirement 4: post-change `_r4_split_plan` reproducibility self-check.

preflight_itemA.py monkey-patched the module constant to *predict* the effect. This script does the
opposite: it reads the constant AS SHIPPED IN THE FILE (no patching at all) and asserts the
resulting 18-case admission set equals the expected set. It also re-asserts the two properties the
relaxation must not disturb:

  * geometry invariance -- (S, NBS, S_TAIL) for every one of the 18 cases is identical to what the
    shipped 8.0 us guard produced (preflight.json P1 is the reference), because the constant is
    read only inside the cost guard, AFTER step 1-6 derive S/NBS/S_TAIL.
  * theorem P2 -- every split-PARALLEL kernel stays inside one wave: grid_A = b_num * S <=
    cube_core_num (probed, never hardcoded).

Usage: verify_split_plan_itemA.py [expected_set_csv] [out.json]
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R4 = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (R4, os.path.join(R4, "bench", "hap_thw_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r4_fused_sampling as M                        # noqa: E402
from r4_thw_common import load_cases                 # noqa: E402

EXPECTED = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "3,4,6,7,8,11,12,17").split(",")]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "split_plan_reproduction.json")
A_SHIP = 8.0

src = open(os.path.join(R4, "r4_fused_sampling.py"), "rb").read()
sha = hashlib.sha256(src).hexdigest()
print(f"kernel sha256                    = {sha}")
print(f"_SPLIT_A_LAUNCH_US as shipped    = {M._SPLIT_A_LAUNCH_US!r}   (READ FROM THE FILE, "
      f"not patched)")
print(f"_SPLIT_SAFETY as shipped         = {M._SPLIT_SAFETY!r}   (must still be 1.5)")
cube = M.detect_cube_core_num()
print(f"cube_core_num (probed)           = {cube}")
print(f"VEC_NUM                          = {M.VEC_NUM}")

CASES = {c["case_id"]: c for c in load_cases("r4_fused_sampling", verbose=False)}
PREFLIGHT = os.path.join(HERE, "preflight.json")
ref = {}
if os.path.exists(PREFLIGHT):
    for r in json.load(open(PREFLIGHT))["P1_cases"]:
        ref[r["case"]] = r

rows, admitted, geom_ok, p2_ok = [], [], True, True
print(f"\n{'=' * 108}")
print(f"{'case':>4} {'B':>4} {'V':>7} {'K':>4} {'BV':>5} {'b_num':>5} | {'use_split':>9} "
      f"{'S':>3} {'NBS':>4} {'S_TAIL':>6} {'grid_A':>6} {'P2':>4} | {'geom==ship':>10} | note")
print(f"{'=' * 108}")
for ci in sorted(CASES):
    c = CASES[ci]
    B, V, K, BV = c["B"], c["V"], c["K_MAX"], c["BV"]
    use, S, NBS, ST = M._r4_split_plan(B, V, K, BV)          # <- the REAL shipped function
    b_num = -(-B // M.VEC_NUM)
    grid_A = b_num * S if use else b_num
    p2 = (grid_A <= cube) if use else True
    p2_ok &= p2
    same_geom = True
    if ci in ref:
        same_geom = (S, NBS, ST) == (ref[ci]["S"], ref[ci]["NBS"], ref[ci]["S_TAIL"])
        geom_ok &= same_geom
    if use:
        admitted.append(ci)
    note = "SPLIT" if use else "current"
    if ci in (6, 11):
        note += "  <-- newly admitted by this change"
    print(f"{ci:>4} {B:>4} {V:>7} {K:>4} {BV:>5} {b_num:>5} | {str(use):>9} {S:>3} {NBS:>4} "
          f"{ST:>6} {grid_A:>6} {'ok' if p2 else '*** VIOLATION ***':>4} | "
          f"{str(same_geom):>10} | {note}")
    rows.append(dict(case=ci, B=B, V=V, K_MAX=K, BV=BV, b_num=b_num, use_split=use, S=S, NBS=NBS,
                     S_TAIL=ST, grid_A=grid_A, theorem_P2_ok=p2, geometry_matches_shipped=same_geom))

print(f"\n[self-check] admitted set  = {admitted}")
print(f"[self-check] expected set  = {EXPECTED}")
set_ok = admitted == EXPECTED
print(f"[self-check] SET MATCH     = {set_ok}")
print(f"[self-check] geometry (S,NBS,S_TAIL) unchanged vs shipped 8.0 guard for all 18 = {geom_ok}")
print(f"[self-check] theorem P2 (grid_A <= cube_core_num={cube}) holds for every split case = {p2_ok}")
print(f"[self-check] _SPLIT_SAFETY still 1.5 = {M._SPLIT_SAFETY == 1.5}")
verdict = set_ok and geom_ok and p2_ok and M._SPLIT_SAFETY == 1.5
print(f"\n[SELF-CHECK] {'PASS' if verdict else '*** FAIL ***'}")

json.dump(dict(_meta=dict(kernel_sha256=sha, a_launch_us=M._SPLIT_A_LAUNCH_US,
                          safety=M._SPLIT_SAFETY, cube_core_num=cube, VEC_NUM=M.VEC_NUM,
                          note="calls the DELIVERED _r4_split_plan with the constant exactly as "
                               "shipped in the file; nothing is patched"),
               admitted_set=admitted, expected_set=EXPECTED, set_match=set_ok,
               geometry_unchanged_all18=geom_ok, theorem_P2_ok=p2_ok, verdict=bool(verdict),
               cases=rows), open(OUT, "w"), indent=1)
print(f"[self-check] wrote {OUT}")
sys.exit(0 if verdict else 1)
