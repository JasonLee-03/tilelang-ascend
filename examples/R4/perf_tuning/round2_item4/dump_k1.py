"""Compile all 18 bench-case K1 specializations (+ K2m/K2g) and dump:
  * the generated Ascend C source (for an emitted-code diff), and
  * the empirical UB usage parsed from ascend_ub.GetWithOffset<T>(n, off).

Usage: dump_k1.py <out_dir>
Reads the case table from perf_tuning/round2_bv_probe/bv_static_analysis.py (CASES).
"""
import json
import os
import re
import sys
import time

R4 = "/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4"
sys.path.insert(0, R4)
sys.path.insert(0, os.path.join(R4, "perf_tuning/round2_bv_probe"))

from ub_probe import ub_usage                                     # noqa: E402
from bv_static_analysis import CASES, UB_LIMIT, align_up          # noqa: E402
from r4_fused_sampling import (r4_topk_stats, r4_mask_sample,     # noqa: E402
                               r4_gumbel_sample)

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)

# Per-case BV comes from the case fixture (contract attr), NOT a kernel default:
# cases 8 & 13 use BV=2048 (K_MAX=128), everything else BV=4096.
rows = []
for (cid, B, V, K, BV, nmp) in CASES:
    t0 = time.time()
    src = r4_topk_stats(B, V, K, BV, nmp).get_kernel_source()
    dt = time.time() - t0
    tot, nb, det, reg = ub_usage(src)
    open(os.path.join(out_dir, f"K1_case{cid:02d}.cc"), "w").write(src)
    TAIL = V % BV
    rows.append(dict(case=cid, B=B, V=V, K=K, BV=BV, need_min_p=nmp,
                     n_blk_full=V // BV, TAIL=TAIL, TAIL_PAD=align_up(TAIL, 32),
                     ub_k1=tot, ub_region=reg, ub_headroom=reg - tot,
                     ub_pct=round(100.0 * tot / reg, 4), n_buffers=nb,
                     compile_s=round(dt, 2)))
    print(f"case{cid:>2} B={B:<4}V={V:<7}K={K:<4}BV={BV:<5} n_blk_full={V // BV:<3}"
          f" TAIL={TAIL:<5} UB={tot:>7}/{reg} ({100.0 * tot / reg:5.2f}%)"
          f" headroom={reg - tot:>7} bufs={nb:>3} compile={dt:5.1f}s"
          f"{'  *** OVER LIMIT ***' if tot > UB_LIMIT else ''}")

# K2m / K2g are unaffected by item #4 but measured for the record.
u2m = ub_usage(r4_mask_sample(8, 10240, 4096).get_kernel_source())
u2g = ub_usage(r4_gumbel_sample(8, 10240, 4096).get_kernel_source())
u2m_2048 = ub_usage(r4_mask_sample(4, 16384, 2048).get_kernel_source())
u2g_2048 = ub_usage(r4_gumbel_sample(4, 16384, 2048).get_kernel_source())
print(f"\nK2m(BV=4096) UB={u2m[0]}/{u2m[3]}  K2g(BV=4096) UB={u2g[0]}/{u2g[3]}")
print(f"K2m(BV=2048) UB={u2m_2048[0]}/{u2m_2048[3]}  K2g(BV=2048) UB={u2g_2048[0]}/{u2g_2048[3]}")

worst = max(r["ub_k1"] for r in rows)
meta = dict(kernel_sha256=None, ub_limit=UB_LIMIT, rows=rows,
            worst_k1_ub=worst, worst_k1_headroom=UB_LIMIT - worst,
            k2m_4096=u2m[0], k2g_4096=u2g[0],
            k2m_2048=u2m_2048[0], k2g_2048=u2g_2048[0])
json.dump(meta, open(os.path.join(out_dir, "ub_k1_18cases.json"), "w"), indent=1)
print(f"\nWORST K1 UB over 18 cases = {worst}/{UB_LIMIT}  headroom={UB_LIMIT - worst}")
print(f"wrote {out_dir}/ub_k1_18cases.json + {len(rows)} .cc files")
