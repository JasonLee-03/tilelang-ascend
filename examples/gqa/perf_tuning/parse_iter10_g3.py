"""Iter10 Gate-3 bottom-line parser (INTEGRATED gqa.py, ZERO monkeypatch).

msprof_op_iter10_G3_{shape}_base_s{1,2}: instance census + TaskDur on the
shipped-post-integration file.  Expectations (preregistered,
candidate_iter10_P4.json gate3):
  r2  (flat kv1, newly covered): instances = main_kernel:0 ONLY —
      transpose_qkv ABSENT; e2e in the Gate-2 p4 band (~26.4us)
  c7  (flat kv2, newly covered): tqkv ABSENT; e2e ~36.2us
  skv200 (flat kv2 dispatch HIT but S_kv=200 NON-aligned): tqkv STILL
      PRESENT (the zero-fill pad columns need it) + main; e2e recorded
      (no Gate-2 reference — new workload; c7-like band expected)
  r1  (per-head, EXCLUDED domain): tqkv + main + transpose_out with Block
      Dim 16 (iter7's P1 retile live); e2e ~20.2-20.7us
  r5  (decode): main only, ~146.5us

Run: python3 examples/gqa/perf_tuning/parse_iter10_g3.py
"""

import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parse_msprof_iter6 import find_csvs, parse_kernel  # noqa: E402

# shape -> (role, Gate-2 p4-arm reference mean us or None)
G3 = {
    "r2": ("covered-new flat kv1 (tqkv must be ABSENT)", 26.35),
    "c7": ("covered-new flat kv2 (tqkv must be ABSENT)", 36.19),
    "skv200": ("NON-ALIGNED gate (tqkv must STAY)", None),
    "r1": ("EXCLUDED per-head (tqkv+tout must stay)", 20.28),
    "r5": ("decode", 146.50),
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
    print(f"{'shape':<8}{'s':<3}{'instances':<42}{'e2e us':>9}{'tqkv us':>9}{'main us':>9}{'tout us':>9}{'tout blk':>9}{'freq':>10}")
    agg = {}
    ok = True
    for shape in G3:
        for n in (1, 2):
            d = os.path.join(HERE, f"msprof_op_iter10_G3_{shape}_base_s{n}")
            if not os.path.isdir(d):
                print(f"{shape:<8}{n:<3} MISSING")
                ok = False
                continue
            inst = parse_dir(d)
            e2e = sum(r["dur_us"] for r in inst.values())
            m = inst.get("main_kernel:0") or inst.get("main")
            q = inst.get("kernel_kernel:0")
            # r1's chain has TWO aux kernels: kernel_kernel:0 = tqkv,
            # kernel_kernel:1 = tout; covered shapes have main only.
            t = inst.get("kernel_kernel:1")
            labels = sorted(inst)
            f = lambda r: f"{r['dur_us']:9.2f}" if r else f"{'ABSENT':>9}"  # noqa: E731
            freq = m.get("freq") if m else "?"
            print(f"{shape:<8}{n:<3}{str(labels)[:40]:<42}{e2e:9.2f}{f(q)}{f(m)}{f(t)}"
                  f"{(t.get('block_dim') if t else '—'):>9}{str(freq):>10}")
            agg.setdefault(shape, []).append(
                {"e2e": e2e, "tqkv": q["dur_us"] if q else None, "tout": t["dur_us"] if t else None,
                 "main": m["dur_us"] if m else None, "inst": labels})

    print()
    print("--- role checks ---")
    for shape, (role, ref) in G3.items():
        rows = agg.get(shape, [])
        if not rows:
            print(f"{shape:<8} NO DATA — FAIL")
            ok = False
            continue
        if shape in ("r2", "c7"):
            q_absent = all(r["tqkv"] is None for r in rows)
            m_only = all(r["inst"] == ["main_kernel:0"] or r["inst"] == ["main"] for r in rows)
            e2e = statistics.mean(r["e2e"] for r in rows)
            band = abs(e2e - ref) / ref * 100 if ref else 0
            good = q_absent and m_only and band <= 10
            ok = ok and good
            print(f"{shape:<8} tqkv ABSENT={q_absent} main-only={m_only} e2e {e2e:.2f} vs "
                  f"Gate2-p4 {ref} ({band:+.1f}%, band 10%) -> {'PASS' if good else '** FAIL **'}  [{role}]")
        elif shape == "skv200":
            q_present = all(r["tqkv"] is not None for r in rows)
            ok = ok and q_present
            e2e = statistics.mean(r["e2e"] for r in rows)
            print(f"{shape:<8} tqkv PRESENT={q_present} (durs "
                  f"{[round(r['tqkv'],2) for r in rows]}) e2e {e2e:.2f} -> "
                  f"{'PASS' if q_present else '** FAIL **'}  [{role}]")
        elif shape == "r1":
            q_ok = all(r["tqkv"] is not None for r in rows)
            t_ok = all(r["tout"] is not None for r in rows)
            e2e = statistics.mean(r["e2e"] for r in rows)
            band = abs(e2e - ref) / ref * 100
            good = q_ok and t_ok and band <= 10
            ok = ok and good
            print(f"{shape:<8} tqkv PRESENT={q_ok} tout PRESENT={t_ok} e2e {e2e:.2f} vs "
                  f"{ref} ({band:+.1f}%) -> {'PASS' if good else '** FAIL **'}  [{role}]")
        elif shape == "r5":
            m_only = all(len(r["inst"]) == 1 for r in rows)
            e2e = statistics.mean(r["e2e"] for r in rows)
            band = abs(e2e - ref) / ref * 100
            good = m_only and band <= 3
            ok = ok and good
            print(f"{shape:<8} main-only={m_only} e2e {e2e:.2f} vs {ref} ({band:+.1f}%, band 3%) -> "
                  f"{'PASS' if good else '** FAIL **'}  [{role}]")

    print()
    print(f"GATE-3 MSPROF BOTTOM LINE: {'PASS' if ok else '** FAIL **'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
