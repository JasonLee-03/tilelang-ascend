"""Item-A prerequisite scan: which shape-carrying test entries flip when _SPLIT_A_LAUNCH_US moves?

Round 3 reported "84 shape-carrying entries, 4 of which go stale when the guard is relaxed".
This script RE-DERIVES that number instead of trusting it: it walks the test module's AST, picks
up every tuple literal that contains a plausible (B, V, K_MAX, BV) window, evaluates the
delivered _r4_split_plan at the shipped 8.0 us and at a candidate value, and reports every entry
whose use_split verdict differs.

Nothing here touches device or files. Usage: scan_shape_entries.py [a_candidate ...]
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R4 = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, R4)

import r4_fused_sampling as M  # noqa: E402

TEST = os.path.join(R4, "test_r4_fused_sampling.py")
SRC = open(TEST).read()
TREE = ast.parse(SRC)
LINES = SRC.split("\n")

CANDIDATES = [float(a) for a in (sys.argv[1:] or ["2.74", "1.54", "3.20"])]


def plan(a, B, V, K, BV):
    old = M._SPLIT_A_LAUNCH_US
    M._SPLIT_A_LAUNCH_US = a
    try:
        return M._r4_split_plan(B, V, K, BV)
    finally:
        M._SPLIT_A_LAUNCH_US = old


def lit(node):
    """Constant-fold a tuple/list element; None for anything non-literal."""
    try:
        v = ast.literal_eval(node)
        return v if isinstance(v, (int, float, bool)) or v is None else None
    except Exception:
        return None


entries = []
# top-level assignment targets only -> the real case tables
for node in ast.walk(TREE):
    if not isinstance(node, (ast.List, ast.Tuple)):
        continue
    for el in node.elts:
        if not isinstance(el, ast.Tuple) or len(el.elts) < 4:
            continue
        vals = [lit(e) for e in el.elts]
        name = vals[0] if isinstance(vals[0], str) else None
        ints = [(i, v) for i, v in enumerate(vals) if isinstance(v, int) and not isinstance(v, bool)]
        if not ints:
            continue
        # find a (B, V, K_MAX, BV)-shaped window: V>=256, K_MAX in {128,256}, BV in {2048,4096}
        for j in range(len(vals) - 3):
            B, V, K, BV = vals[j], vals[j + 1], vals[j + 2], vals[j + 3]
            if all(isinstance(x, int) and not isinstance(x, bool) for x in (B, V, K, BV)) \
                    and 1 <= B <= 4096 and V >= 256 and K in (64, 128, 256, 512) and BV in (1024, 2048, 4096, 8192):
                entries.append(dict(line=el.lineno, name=name, B=B, V=V, K_MAX=K, BV=BV,
                                    window=j, n_fields=len(vals), raw=vals))
                break
        else:
            # ADV_CASES style: bare (B, V, K_MAX, BV)
            if len(vals) == 4 and all(isinstance(x, int) for x in vals) and vals[1] >= 256:
                entries.append(dict(line=el.lineno, name=name, B=vals[0], V=vals[1],
                                    K_MAX=vals[2], BV=vals[3], window=0, n_fields=4, raw=vals))

entries.sort(key=lambda e: e["line"])
print(f"shape-carrying tuple entries found in test file: {len(entries)}")

base = {id(e): plan(8.0, e["B"], e["V"], e["K_MAX"], e["BV"]) for e in entries}
print(f"\ndelivered (a=8.0) split-admitted entries: "
      f"{sum(1 for e in entries if base[id(e)][0])}/{len(entries)}")

for a in CANDIDATES:
    flips = []
    for e in entries:
        p = plan(a, e["B"], e["V"], e["K_MAX"], e["BV"])
        if p[0] != base[id(e)][0]:
            flips.append((e, base[id(e)], p))
    print(f"\n{'=' * 100}")
    print(f"a_launch = {a} us  ->  {len(flips)} entr{'y' if len(flips) == 1 else 'ies'} FLIP")
    print(f"{'=' * 100}")
    for e, p0, p1 in flips:
        print(f"  L{e['line']:>5}  {str(e['name']):<38} B={e['B']:<4} V={e['V']:<7} "
              f"K={e['K_MAX']:<4} BV={e['BV']:<5} field#{e['window']} of {e['n_fields']}")
        print(f"           a=8.0 : use_split={p0[0]!s:<5} S={p0[1]:<3} NBS={p0[2]:<3} S_TAIL={p0[3]}")
        print(f"           a={a:<5}: use_split={p1[0]!s:<5} S={p1[1]:<3} NBS={p1[2]:<3} S_TAIL={p1[3]}")
        print(f"           line  : {LINES[e['line'] - 1].strip()[:150]}")
