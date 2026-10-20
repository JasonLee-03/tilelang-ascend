"""Gate-3 comparator — iter10 P4: pre (shipped backup) vs post (integrated).

Criteria (preregistered, candidate_iter10_P4.json gate3):
  * FINAL gqa() output BIT-IDENTICAL (raw-bit int16 view) for EVERY shape;
  * chosen dispatch IDENTICAL for every shape;
  * tqkv census: "p4" shapes RAN in pre (exactly 1 call) and SKIPPED in post
    (0 calls); "out" shapes IDENTICAL call counts pre vs post (non-aligned
    gate cases must STILL run tqkv in post; decode runs none in either).

Run: python3 examples/gqa/perf_tuning/compare_gate3_iter10.py
"""

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))

pre = torch.load(os.path.join(HERE, "gate3_iter10_outputs_pre.pt"), weights_only=False)
post = torch.load(os.path.join(HERE, "gate3_iter10_outputs_post.pt"), weights_only=False)
cpre = json.load(open(os.path.join(HERE, "gate3_iter10_chosen_pre.json")))
cpost = json.load(open(os.path.join(HERE, "gate3_iter10_chosen_post.json")))

assert set(pre) == set(post), "shape sets differ"
nfail = 0
print(f"{'shape':<26}{'cat':<6}{'bit':<7}{'chosen':<9}{'tqkv pre':<10}{'tqkv post':<10}{'verdict'}")
for name in pre:
    bit = bool(torch.equal(pre[name], post[name]))
    ch_same = cpre["shapes"][name]["chosen"] == cpost["shapes"][name]["chosen"]
    n_pre = len(cpre["shapes"][name]["tqkv_calls"])
    n_post = len(cpost["shapes"][name]["tqkv_calls"])
    cat = cpre["shapes"][name]["cat"]
    if cat == "p4":
        census_ok = (n_pre == 1 and n_post == 0)
    else:
        census_ok = (n_pre == n_post)
    ok = bit and ch_same and census_ok
    nfail += 0 if ok else 1
    print(f"{name:<26}{cat:<6}{'OK' if bit else 'FAIL':<7}{'OK' if ch_same else 'FAIL':<9}"
          f"{n_pre:<10}{n_post:<10}{'PASS' if ok else '** FAIL **'}")

n_p4 = sum(1 for s in cpre["shapes"].values() if s["cat"] == "p4")
n_out = sum(1 for s in cpre["shapes"].values() if s["cat"] == "out")
print(f"\nSUMMARY: {len(pre) - nfail}/{len(pre)} PASS "
      f"(p4-covered {n_p4}, out-of-domain {n_out} >= 5 required)")
assert n_out >= 5, "out-of-domain evidence count below the preregistered >=5"
sys.exit(1 if nfail else 0)
