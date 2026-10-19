#!/usr/bin/python3
# coding=utf-8
"""Gate-blindness audit for item #2: can ANY four-gate input ever trigger a prune?

Enumerates every shape the four gates exercise (L0_CASES / L1_CASES / STAB_CASES /
ADV_CASES, parsed straight out of the DELIVERED test file -- which is not modified),
computes n_blk_full = V // BV for each, and replays the K1 merge chain under the
gate data recipe to count prunable blocks.

Why this matters: the adoption rule requires the four gates to stay green. If the prune
predicate never fires on any gate input, then a pruning implementation is UNVERIFIABLE
dead code -- the gates would pass identically whether the prune logic is correct,
inverted, or catastrophically wrong. That is an independent no-go argument on top of
the 0% measured prune rate on the bench distribution.

Usage: gate_blindness.py <out.json>
"""
import ast
import json
import re
import sys

import numpy as np
import torch

HERE = "/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4"
TEST = HERE + "/test_r4_fused_sampling.py"
sys.path.insert(0, HERE + "/perf_tuning/round2_item2")
from prune_rate import simulate  # noqa: E402  (validated simulator, tau cross-checked)

LOGIT_SCALE = 8.0


def grab_list(src: str, name: str):
    m = re.search(rf"^{name} = (\[.*?^\])", src, re.S | re.M)
    assert m, f"{name} not found"
    return ast.literal_eval(m.group(1))


def gate_logits(B, V, scale=LOGIT_SCALE, offset=0.0, seed=0):
    """test file recipe: torch.manual_seed(seed); randn(B,V)*scale (+offset)."""
    torch.manual_seed(seed)
    return (torch.randn(B, V, dtype=torch.float32) * scale + offset).cpu().numpy().astype(np.float64)


def main():
    src = open(TEST).read()
    L0 = grab_list(src, "L0_CASES")
    L1 = grab_list(src, "L1_CASES")
    STAB = grab_list(src, "STAB_CASES")
    ADV = grab_list(src, "ADV_CASES")

    rows = []
    # L0: (name, B, V, k, p, mu, need_min_p); K_MAX/BV are the wrapper defaults 256/4096
    for (nm, B, V, k, p, mu, nmp) in L0:
        rows.append(("L0", nm, B, V, 256, 4096, LOGIT_SCALE, 0.0))
    # L1: (name, B, V, K_MAX, BV, k, p, mu, need_min_p, scale, offset, tags)
    for t in L1:
        rows.append(("L1", t[0], t[1], t[2], t[3], t[4], t[9], t[10]))
    # STAB: (name, B, V, K_MAX, BV, k, p, gumbel, tags)
    for t in STAB:
        rows.append(("STAB", t[0], t[1], t[2], t[3], t[4], LOGIT_SCALE, 0.0))
    # ADV: (B, V, K_MAX, BV)
    for (B, V, K, BV) in ADV:
        rows.append(("ADV", f"adv_B{B}_V{V}", B, V, K, BV, LOGIT_SCALE, 0.0))

    out = {"recipe": "torch.manual_seed(0); randn(B,V)*scale + offset  (test file, unmodified)",
           "groups": {}, "rows": []}
    print(f"{'grp':<5} {'name':<34} {'B':>4} {'V':>7} {'K_MAX':>5} {'BV':>5} {'nblk':>4} "
          f"{'tot_blk':>7} {'prunable':>8} {'min_margin':>10}")
    tot_blk = tot_pr = 0
    per_grp = {}
    for (grp, nm, B, V, K, BV, scale, off) in rows:
        nblk = V // BV
        lg = gate_logits(B, V, scale, off)
        s = simulate(lg, BV, K, verify=True)
        tot_blk += s["total_blocks"]
        tot_pr += s["prunable_blocks"]
        g = per_grp.setdefault(grp, {"configs": 0, "blocks": 0, "prunable": 0, "max_nblk": 0})
        g["configs"] += 1
        g["blocks"] += s["total_blocks"]
        g["prunable"] += s["prunable_blocks"]
        g["max_nblk"] = max(g["max_nblk"], nblk)
        mm = s["min_margin_over_rows"]
        mm_s = "n/a" if mm is None else format(mm, "+.4f")
        print(f"{grp:<5} {nm[:34]:<34} {B:>4} {V:>7} {K:>5} {BV:>5} {nblk:>4} "
              f"{s['total_blocks']:>7} {s['prunable_blocks']:>8} {mm_s:>10}")
        out["rows"].append({"group": grp, "name": nm, "B": B, "V": V, "K_MAX": K, "BV": BV,
                            "n_blk_full": nblk, "scale": scale, "offset": off, **s})
        del lg
    out["groups"] = per_grp
    out["totals"] = {"configs": len(rows), "total_blocks": tot_blk,
                     "prunable_blocks": tot_pr,
                     "prune_rate": (tot_pr / tot_blk) if tot_blk else None}
    print(f"\nTOTAL over {len(rows)} gate configs: prunable {tot_pr} / {tot_blk} blocks")
    print("per group:", json.dumps(per_grp))
    json.dump(out, open(sys.argv[1], "w"), indent=2)
    print(f"wrote {sys.argv[1]}")


if __name__ == "__main__":
    main()
