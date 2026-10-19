#!/usr/bin/python3
# coding=utf-8
"""Seed-INDEPENDENT analytic prune-rate bound for item #2 (part (a), complementary).

The measured prune rate was 0/5478 on the actual bench inputs. This script proves that is
not a seed artifact by computing the exact expected rate for i.i.d. data.

Model (uniform or any continuous i.i.d. logits -- only continuity matters):
  After b full blocks, n = b*BV elements have been seen. tau_k_run is the K_MAX-th largest
  of those n samples, so its CDF value U := F(tau_k_run) follows Beta(a, c) with
      a = n - K_MAX + 1,  c = K_MAX.
  Block b+1 is prunable iff ALL of its BV elements fall below tau_k_run. Conditional on U,
  that probability is U^BV, hence
      P(prune at step b+1) = E[U^BV] = B(a+BV, c) / B(a, c)
                           = prod_{i=0}^{BV-1} (a+i) / (a+c+i)
  computed in log-space for stability. Note a+c = n+1.

  Expected prunable blocks per row  = sum_{b=1}^{N-1} P(b)   (N = n_blk_full)
  Expected prune rate per row       = that / N

Also computes the folded 36-case geomean ceiling under this EXPECTED rate, using the
msprof-measured topk_share where available and the measured K1 task share, so the bound
does not depend on the particular seed the harness happened to draw.

Usage: analytic_prune_bound.py <out.json>
"""
import json
import math
import sys

sys.path.insert(0, "/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4/perf_tuning/round2_item2")

CASES = [  # (case_id, B, V, K_MAX, BV)
    (1, 2, 256, 256, 4096), (2, 1, 4096, 256, 4096), (3, 8, 16384, 256, 4096),
    (4, 4, 16384, 256, 4096), (5, 7, 4409, 256, 4096), (6, 8, 10240, 256, 4096),
    (7, 4, 152064, 256, 4096), (8, 4, 16384, 128, 2048), (9, 64, 4096, 256, 4096),
    (10, 4, 4096, 256, 4096), (11, 8, 8192, 256, 4096), (12, 16, 32768, 256, 4096),
    (13, 6, 2304, 128, 2048), (14, 32, 1024, 256, 4096), (15, 64, 152064, 256, 4096),
    (16, 128, 32768, 256, 4096), (17, 16, 152377, 256, 4096), (18, 64, 65536, 256, 4096),
]


def log_p_prune(n_seen: int, K: int, BV: int) -> float:
    """ln E[U^BV] with U ~ Beta(n_seen-K+1, K)."""
    a = n_seen - K + 1
    if a <= 0:
        return 0.0  # fewer than K samples seen -> tau is -inf -> never prunes
    s = 0.0
    for i in range(BV):
        s += math.log(a + i) - math.log(n_seen + 1 + i)
    return s


def expected_rate(N: int, K: int, BV: int):
    """Expected prunable blocks / total blocks for one row of N full blocks."""
    if N <= 1:
        return 0.0, 0.0, []
    tot, per = 0.0, []
    for b in range(1, N):                 # b blocks already seen -> block index b is tested
        lp = log_p_prune(b * BV, K, BV)
        p = math.exp(lp) if lp > -700 else 0.0
        per.append((b, p))
        tot += p
    return tot / N, tot, per


def main():
    probe = json.load(open("/tmp/opencode/r4_item2/probe_perf.json"))
    ceil = json.load(open("/tmp/opencode/r4_item2/ceiling.json"))
    TOPK = {c["case"]: c["topk_merge_share_of_K1"] for c in probe["cases"]}
    K1SH = {r["case"]: (r["k1_share_fused"], r["k1_share_mask"]) for r in ceil["per_case"]}

    print(f"{'case':>4} {'N':>4} {'K':>4} {'BV':>5} | {'E[pruned]/row':>13} "
          f"{'E[rate]':>10} | {'P(last blk)':>11} | {'analytic ceiling (36-case)':>26}")
    rows, gains = [], []
    for (cid, B, V, K, BV) in CASES:
        N = V // BV
        rate, exp_blocks, per = expected_rate(N, K, BV)
        p_last = per[-1][1] if per else None
        sf, sm = K1SH[cid]
        ts = TOPK.get(cid)
        # ceiling with the ANALYTIC expected rate; topk_share only known for probed cases
        cf = rate * ts * sf if ts is not None else None
        cm = rate * ts * sm if ts is not None else None
        if cf is not None:
            gains += [cf, cm]
        rows.append(dict(case=cid, B=B, V=V, K_MAX=K, BV=BV, n_blk_full=N,
                         expected_pruned_blocks_per_row=exp_blocks,
                         expected_prune_rate=rate,
                         p_prune_last_block=p_last,
                         topk_share_probed=ts,
                         analytic_ceiling_fused=cf, analytic_ceiling_mask=cm))
        print(f"{cid:>4} {N:>4} {K:>4} {BV:>5} | {exp_blocks:>13.3e} {rate:>10.3e} | "
              f"{('n/a' if p_last is None else f'{p_last:.3e}'):>11} | "
              f"{('not probed' if cf is None else f'fused {cf*100:.6f}% / mask {cm*100:.6f}%'):>26}")

    ratios = [1.0 - g for g in gains]
    g36 = 100.0 * (1.0 / math.exp(sum(math.log(r) for r in ratios) / len(ratios)) - 1.0) if ratios else None
    worst = max(rows, key=lambda r: r["expected_prune_rate"])
    print(f"\nmax expected prune rate over the 18 cases = {worst['expected_prune_rate']:.3e} "
          f"(case {worst['case']}, N={worst['n_blk_full']})")
    print(f"analytic 36-case geomean ceiling (probed cases only) = "
          f"{('n/a' if g36 is None else f'{g36:.6f}%')}")
    print("VERDICT: " + ("GO" if (g36 or 0) > 3.0 else "NO-GO"))
    json.dump({"rows": rows, "analytic_geomean_ceiling_pct": g36,
               "max_expected_prune_rate": worst["expected_prune_rate"],
               "max_expected_prune_rate_case": worst["case"]},
              open(sys.argv[1], "w"), indent=2)
    print(f"wrote {sys.argv[1]}")


if __name__ == "__main__":
    main()
