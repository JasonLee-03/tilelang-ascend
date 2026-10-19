"""K1-isolated A/B: baseline vs candidate, per case, both tasks.

K1 (r4_topk_stats) is the ONLY kernel item #4 touches. K2g/K2m are untouched, so
their numbers serve as a built-in control: any movement there is measurement noise,
which calibrates how to read the K1 column.
"""
import json
import math

CASES = {1: (2, 256, 256, 4096), 2: (1, 4096, 256, 4096), 3: (8, 16384, 256, 4096),
         4: (4, 16384, 256, 4096), 5: (7, 4409, 256, 4096), 6: (8, 10240, 256, 4096),
         7: (4, 152064, 256, 4096), 8: (4, 16384, 128, 2048), 9: (64, 4096, 256, 4096),
         10: (4, 4096, 256, 4096), 11: (8, 8192, 256, 4096), 12: (16, 32768, 256, 4096),
         13: (6, 2304, 128, 2048), 14: (32, 1024, 256, 4096), 15: (64, 152064, 256, 4096),
         16: (128, 32768, 256, 4096), 17: (16, 152377, 256, 4096), 18: (64, 65536, 256, 4096)}

D = {lab: json.load(open(f"/tmp/opencode/r4_item4/k1split_{lab}.json"))["per_case"]
     for lab in ("fused_base", "fused_cand", "mask_base", "mask_cand")}


def geo(vals):
    vals = [v for v in vals if v and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")


for task, bl, cd in (("r4_fused_sampling (K1+K2g)", "fused_base", "fused_cand"),
                     ("r4_mask_sample   (K1+K2m)", "mask_base", "mask_cand")):
    print(f"\n================ {task} ================")
    print(f"{'case':>4} {'B':>4} {'V':>7} {'BV':>5} {'nblk':>5} {'TAIL':>5} | "
          f"{'K1 base':>9} {'K1 cand':>9} {'K1 d%':>7} | {'K2 base':>9} {'K2 cand':>9} {'K2 d%':>7}")
    k1r, k2r, k1r_nblk0, k1r_nblkpos = [], [], [], []
    for ci in sorted(CASES):
        B, V, K, BV = CASES[ci]
        nblk, TAIL = V // BV, V % BV
        b, c = D[bl][str(ci)], D[cd][str(ci)]
        d1 = 100 * (c["k1_us"] - b["k1_us"]) / b["k1_us"]
        d2 = 100 * (c["k2_us"] - b["k2_us"]) / b["k2_us"]
        k1r.append(c["k1_us"] / b["k1_us"]); k2r.append(c["k2_us"] / b["k2_us"])
        (k1r_nblk0 if nblk == 0 else k1r_nblkpos).append(c["k1_us"] / b["k1_us"])
        print(f"{ci:>4} {B:>4} {V:>7} {BV:>5} {nblk:>5} {TAIL:>5} | "
              f"{b['k1_us']:>9.2f} {c['k1_us']:>9.2f} {d1:>+7.2f} | "
              f"{b['k2_us']:>9.2f} {c['k2_us']:>9.2f} {d2:>+7.2f}")
    g1, g2 = geo(k1r), geo(k2r)
    print(f"  K1 geomean(cand/base) = {g1:.6f}  ->  {100*(1/g1-1):+.3f}% "
          f"({'faster' if g1 < 1 else 'slower'})")
    print(f"  K2 geomean(cand/base) = {g2:.6f}  ->  {100*(1/g2-1):+.3f}% "
          f"[CONTROL: K2 code is untouched -> this is the noise floor]")
    if k1r_nblk0:
        print(f"  K1 geomean, n_blk_full==0 cases only = {geo(k1r_nblk0):.6f} "
              f"[CONTROL: emitted code byte-identical -> expect ~1.0]")
    if k1r_nblkpos:
        print(f"  K1 geomean, n_blk_full>0  cases only = {geo(k1r_nblkpos):.6f} "
              f"-> {100*(1/geo(k1r_nblkpos)-1):+.3f}%  [the cases the change actually affects]")
    print(f"  K1 worst regression = {100*(max(k1r)-1):+.3f}%   K1 best = {100*(min(k1r)-1):+.3f}%")
