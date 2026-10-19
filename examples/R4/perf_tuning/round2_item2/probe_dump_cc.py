#!/usr/bin/python3
# coding=utf-8
"""Compile the real K1 + the three probes and dump their emitted Ascend C source.

Answers:
  * section-2 point 5: does `with T.If(...)` around T.tile.topk emit a REAL C++ branch
    that can skip the Sort, or is it predicated/masked (which would zero the gain)?
  * part (b): static vector-instruction census of the K1 main loop, split into
    online-softmax vs topk+merge.

Usage: probe_dump_cc.py <out_dir> [case_id ...]
"""
import os
import re
import sys

R4 = "/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4"
HERE = os.path.join(R4, "perf_tuning/round2_item2")
sys.path.insert(0, R4)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(R4, "perf_tuning/round2_bv_probe"))

from bv_static_analysis import CASES  # noqa: E402
from r4_fused_sampling import r4_topk_stats  # noqa: E402
from probe_k1_nosort import k1_nosort  # noqa: E402
from probe_k1_pruned import k1_pruned  # noqa: E402

CASE_BY_ID = {c[0]: c for c in CASES}


def main():
    out_dir = sys.argv[1]
    want = [int(x) for x in sys.argv[2:]] or [15, 8]
    os.makedirs(out_dir, exist_ok=True)
    for cid in want:
        _, B, V, K, BV, nmp = CASE_BY_ID[cid]
        srcs = {
            "real": r4_topk_stats(B, V, K, BV, nmp).get_kernel_source(),
            "nosort": k1_nosort(B, V, K, BV, nmp).get_kernel_source(),
            "pruned": k1_pruned(B, V, K, BV, nmp).get_kernel_source(),
        }
        for name, s in srcs.items():
            p = os.path.join(out_dir, f"K1_case{cid:02d}_{name}.cc")
            open(p, "w").write(s)
            print(f"wrote {p}  ({len(s)} bytes)")

    # ---- point 5: is the topk inside a real branch in the pruned variant? ----
    cid = want[0]
    _, B, V, K, BV, nmp = CASE_BY_ID[cid]
    s = k1_pruned(B, V, K, BV, nmp).get_kernel_source()
    print("\n=== point-5 evidence: emitted code around the pruned topk (case %d) ===" % cid)
    lines = s.splitlines()
    for i, ln in enumerate(lines):
        if "TopK" in ln:
            lo = max(0, i - 6)
            for j in range(lo, min(len(lines), i + 3)):
                print(f"{j+1:>5}: {lines[j]}")
            print("      ...")
            break


if __name__ == "__main__":
    main()
