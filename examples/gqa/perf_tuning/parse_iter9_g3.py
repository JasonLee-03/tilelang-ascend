"""Iter9 Gate-3 bottom-line parser (INTEGRATED gqa.py, ZERO monkeypatch).

msprof_op_iter9_G3_{shape}_base_s{1,2}: instance census + TaskDur on the
shipped-post-integration file.  Expectations:
  c19 (prefill_expert, newly covered): instances = kernel_kernel:0 (tqkv) +
      main_kernel:0 ONLY — transpose_out ABSENT; e2e in the Gate-2 w2 band
      (~49.9us)
  c9  (g2stack, newly covered): tout ABSENT; e2e ~605us
  r1  (per-head, EXCLUDED domain): transpose_out STILL PRESENT with Block Dim
      16 (iter7's P1 retile live); e2e ~20.7-21.0us
  r2  (wave-1 flat, unchanged): tout ABSENT (since iter8); e2e ~37us
  r5  (decode): main only, ~146us (anchors 145.89 iter6 / 146.81 iter7 /
      145.36 iter8-G3 / 146.07 this-session Phase-A base)

Run: python3 examples/gqa/perf_tuning/parse_iter9_g3.py
"""

import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parse_msprof_iter6 import find_csvs, parse_kernel  # noqa: E402

# shape -> (role, Gate-2 reference mean us)
G3 = {
    "c19": ("covered-new prefill_expert", 49.86),
    "c9": ("covered-new g2stack", 605.21),
    "r1": ("EXCLUDED per-head (tout must stay)", 20.67),
    "r2": ("wave-1 flat (unchanged)", 37.13),
    "r5": ("decode", 146.07),
}


def parse_dir(base):
    out = {}
    for label, kinds in find_csvs(base).items():
        r = parse_kernel(kinds)
        if r:
            out[label] = r
    return out


def main():
    print("=" * 112)
    print("Gate-3 bottom line on the INTEGRATED gqa.py (arm=base, zero patches)")
    print("=" * 112)
    print(f"{'shape':<6}{'s':<3}{'instances':<44}{'e2e us':>9}{'main us':>9}{'tout us':>9}{'tout blk':>9}{'freq':>10}")
    agg = {}
    for shape in G3:
        for n in (1, 2):
            d = os.path.join(HERE, f"msprof_op_iter9_G3_{shape}_base_s{n}")
            if not os.path.isdir(d):
                print(f"{shape:<6}{n:<3} MISSING")
                continue
            inst = parse_dir(d)
            e2e = sum(r["dur_us"] for r in inst.values())
            m = inst.get("main_kernel:0") or inst.get("main")
            t = inst.get("kernel_kernel:1")
            freq = next(iter(inst.values())).get("freq")
            agg.setdefault(shape, []).append((e2e, m["dur_us"] if m else None,
                                              t["dur_us"] if t else None,
                                              t.get("block_dim") if t else None))
            print(f"{shape:<6}{n:<3}{str(sorted(inst))[:42]:<44}{e2e:>9.2f}"
                  f"{(m['dur_us'] if m else float('nan')):>9.2f}"
                  f"{(t['dur_us'] if t else float('nan')):>9.2f}"
                  f"{(str(t.get('block_dim')) if t else 'ABSENT'):>9}{freq:>10}")
    print()
    print(f"{'shape':<6}{'role':<38}{'e2e mean':>10}{'Gate-2 ref':>12}{'delta %':>9}  tout status")
    ok = True
    for shape, (role, ref) in G3.items():
        if shape not in agg:
            print(f"{shape:<6}{role:<38} MISSING")
            ok = False
            continue
        e2es = [a[0] for a in agg[shape]]
        m = statistics.mean(e2es)
        tout = [a[2] for a in agg[shape] if a[2] is not None]
        tstat = (f"PRESENT {statistics.mean(tout):.2f}us blk={agg[shape][0][3]}" if tout
                 else "ABSENT")
        delta = 100 * (m - ref) / ref
        # expectations: newly covered + wave1 + decode => tout ABSENT;
        # per-head excluded => tout PRESENT with blk 16
        want_absent = shape in ("c19", "c9", "r2", "r5")
        exp_ok = (not tout) if want_absent else (bool(tout) and str(agg[shape][0][3]) == "16")
        band_ok = abs(delta) <= 3.0 or shape == "r1"  # r1 micro-kernel band +/-10%
        if shape == "r1":
            band_ok = abs(delta) <= 10.0
        ok &= exp_ok and band_ok
        print(f"{shape:<6}{role:<38}{m:>10.2f}{ref:>12.2f}{delta:>+8.1f}%  {tstat}  "
              f"{'OK' if (exp_ok and band_ok) else '** CHECK **'}")
    print(f"\n[G3_BOTTOM_LINE] {'ALL EXPECTATIONS MET' if ok else 'MISMATCH — SEE ABOVE'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
