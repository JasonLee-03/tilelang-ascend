"""Iter11 Gate-3 bottom-line parser (INTEGRATED gqa.py, ZERO monkeypatch).

msprof_op_iter11_G3_{shape}_base_s{1,2}: instance census + TaskDur on the
shipped-post-integration file.  Expectations (preregistered,
candidate_iter11_P4w2.json gate3):
  c19  (prefill_expert, newly covered): instances = main_kernel:0 ONLY —
      transpose_qkv ABSENT; e2e in the Gate-2 w2 band (~33.7us)
  c9   (g2stack, newly covered): tqkv ABSENT; e2e ~565us
  skv200 (wave-1 flat gate, S_kv=200 NON-aligned): tqkv STILL PRESENT
      (tail mode, zero-fill columns) + main; e2e recorded (tail-tqkv
      dominated, iter10 reference ~166.8us tqkv alone)
  skv400g (wave-2 g2stack gate, S aligned / S_kv=400 NON-aligned): tqkv
      STILL PRESENT (tail mode) + main (out_bsnd skips tout — S aligned);
      e2e recorded
  r1   (per-head, EXCLUDED domain): tqkv + main + transpose_out with Block
      Dim 16 (iter7's P1 retile live); e2e ~20.3-20.4us
  r2   (wave-1 flat, UNCHANGED): main ONLY; e2e ~26.3-27.0us
  r5   (decode): main only, ~145.5-146.0us

Run: python3 examples/gqa/perf_tuning/parse_iter11_g3.py
"""

import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parse_msprof_iter6 import find_csvs, parse_kernel  # noqa: E402

# shape -> (role, Gate-2 w2-arm/guard reference mean us or None)
G3 = {
    "c19": ("covered-new prexpert (tqkv must be ABSENT)", 33.72),
    "c9": ("covered-new g2stack (tqkv must be ABSENT)", 565.15),
    "skv200": ("wave-1 NON-ALIGNED gate (tqkv must STAY)", None),
    "skv400g": ("wave-2 NON-ALIGNED S_kv gate (tqkv must STAY)", None),
    "r1": ("EXCLUDED per-head (tqkv+tout must stay)", 20.39),
    "r2": ("wave-1 flat UNCHANGED (main only)", 26.66),
    "r5": ("decode", 145.76),
}


def parse_dir(base):
    out = {}
    for label, kinds in find_csvs(base).items():
        r = parse_kernel(kinds)
        if r:
            out[label] = r
    return out


def main():
    print("=" * 118)
    print("Gate-3 bottom line on the INTEGRATED gqa.py (arm=base, zero patches)")
    print("=" * 118)
    print(f"{'shape':<9}{'s':<3}{'instances':<42}{'e2e us':>9}{'tqkv us':>9}{'main us':>9}{'tout us':>9}{'tout blk':>9}{'freq':>10}")
    agg = {}
    ok = True
    for shape in G3:
        for n in (1, 2):
            d = os.path.join(HERE, f"msprof_op_iter11_G3_{shape}_base_s{n}")
            if not os.path.isdir(d):
                print(f"{shape:<9}{n:<3} MISSING")
                ok = False
                continue
            inst = parse_dir(d)
            e2e = sum(r["dur_us"] for r in inst.values())
            m = inst.get("main_kernel:0") or inst.get("main")
            q = inst.get("kernel_kernel:0")
            # r1's chain has TWO aux kernels: kernel_kernel:0 = tqkv,
            # kernel_kernel:1 = tout; covered shapes have main only;
            # skv400g has tqkv + main (tout skipped — S aligned, iter9 out_bsnd).
            t = inst.get("kernel_kernel:1")
            labels = sorted(inst)
            f = lambda r: f"{r['dur_us']:9.2f}" if r else f"{'ABSENT':>9}"  # noqa: E731
            freq = m.get("freq") if m else "?"
            print(f"{shape:<9}{n:<3}{str(labels)[:40]:<42}{e2e:9.2f}{f(q)}{f(m)}{f(t)}"
                  f"{(t.get('block_dim') if t else '—'):>9}{str(freq):>10}")
            agg.setdefault(shape, []).append(
                {"e2e": e2e, "tqkv": q["dur_us"] if q else None, "tout": t["dur_us"] if t else None,
                 "main": m["dur_us"] if m else None, "inst": labels})

    print()
    print("--- role checks ---")
    for shape, (role, ref) in G3.items():
        rows = agg.get(shape, [])
        if not rows:
            print(f"{shape:<9} NO DATA — FAIL")
            ok = False
            continue
        if shape in ("c19", "c9", "r2"):
            q_absent = all(r["tqkv"] is None for r in rows)
            m_only = all(r["inst"] == ["main_kernel:0"] or r["inst"] == ["main"] for r in rows)
            e2e = statistics.mean(r["e2e"] for r in rows)
            band = abs(e2e - ref) / ref * 100 if ref else 0
            good = q_absent and m_only and band <= 10
            ok = ok and good
            print(f"{shape:<9} tqkv ABSENT={q_absent} main-only={m_only} e2e {e2e:.2f} vs "
                  f"ref {ref} ({band:+.1f}%, band 10%) -> {'PASS' if good else '** FAIL **'}  [{role}]")
        elif shape in ("skv200", "skv400g"):
            q_present = all(r["tqkv"] is not None for r in rows)
            t_absent = all(r["tout"] is None for r in rows)  # skv400g: S aligned => tout skipped (iter9)
            ok = ok and q_present
            e2e = statistics.mean(r["e2e"] for r in rows)
            extra = "" if shape == "skv200" else f" tout ABSENT (S aligned, out_bsnd live)={t_absent}"
            print(f"{shape:<9} tqkv PRESENT={q_present} (durs "
                  f"{[round(r['tqkv'],2) for r in rows]}) e2e {e2e:.2f}{extra} -> "
                  f"{'PASS' if q_present else '** FAIL **'}  [{role}]")
        elif shape == "r1":
            q_ok = all(r["tqkv"] is not None for r in rows)
            t_ok = all(r["tout"] is not None for r in rows)
            blk16 = all(r["inst"].count("kernel_kernel:1") == 1 for r in rows)
            e2e = statistics.mean(r["e2e"] for r in rows)
            band = abs(e2e - ref) / ref * 100
            good = q_ok and t_ok and band <= 10
            ok = ok and good
            print(f"{shape:<9} tqkv PRESENT={q_ok} tout PRESENT={t_ok} e2e {e2e:.2f} vs "
                  f"{ref} ({band:+.1f}%) -> {'PASS' if good else '** FAIL **'}  [{role}]")
        elif shape == "r5":
            m_only = all(len(r["inst"]) == 1 for r in rows)
            e2e = statistics.mean(r["e2e"] for r in rows)
            band = abs(e2e - ref) / ref * 100
            good = m_only and band <= 3
            ok = ok and good
            print(f"{shape:<9} main-only={m_only} e2e {e2e:.2f} vs {ref} ({band:+.1f}%, band 3%) -> "
                  f"{'PASS' if good else '** FAIL **'}  [{role}]")

    print()
    print(f"GATE-3 MSPROF BOTTOM LINE: {'PASS' if ok else '** FAIL **'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
