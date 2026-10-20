"""Gate-3 comparator — iter11 P4 wave-2: pre (shipped backup) vs post
(integrated).

Criteria (preregistered, candidate_iter11_P4w2.json gate3):
  * FINAL gqa() output BIT-IDENTICAL (raw-bit int16 view) for EVERY shape;
  * chosen dispatch IDENTICAL for every shape;
  * tqkv census: "p4w2" shapes RAN in pre (exactly 1 call) and SKIPPED in
    post (0 calls); "out" shapes IDENTICAL call counts pre vs post (wave-1
    flat shapes: 0 in BOTH — the undisturbed evidence; non-aligned gate
    cases must STILL run tqkv in post; decode runs none in either).

Run: python3 examples/gqa/perf_tuning/compare_gate3_iter11.py
"""

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))

pre = torch.load(os.path.join(HERE, "gate3_iter11_outputs_pre.pt"), weights_only=False)
post = torch.load(os.path.join(HERE, "gate3_iter11_outputs_post.pt"), weights_only=False)
cpre = json.load(open(os.path.join(HERE, "gate3_iter11_chosen_pre.json")))
cpost = json.load(open(os.path.join(HERE, "gate3_iter11_chosen_post.json")))

assert set(pre) == set(post), "shape sets differ"
nfail = 0
print(f"{'shape':<28}{'cat':<6}{'bit':<7}{'chosen':<9}{'tqkv pre':<10}{'tqkv post':<10}{'verdict'}")
for name in pre:
    bit = bool(torch.equal(pre[name], post[name]))
    ch_same = cpre["shapes"][name]["chosen"] == cpost["shapes"][name]["chosen"]
    n_pre = len(cpre["shapes"][name]["tqkv_calls"])
    n_post = len(cpost["shapes"][name]["tqkv_calls"])
    cat = cpre["shapes"][name]["cat"]
    if cat == "p4w2":
        census_ok = (n_pre == 1 and n_post == 0)
    else:
        census_ok = (n_pre == n_post)
    ok = bit and ch_same and census_ok
    nfail += 0 if ok else 1
    print(f"{name:<28}{cat:<6}{'OK' if bit else 'FAIL':<7}{'OK' if ch_same else 'FAIL':<9}"
          f"{n_pre:<10}{n_post:<10}{'PASS' if ok else '** FAIL **'}")

n_w2 = sum(1 for s in cpre["shapes"].values() if s["cat"] == "p4w2")
n_out = sum(1 for s in cpre["shapes"].values() if s["cat"] == "out")
# wave-1 undisturbed evidence must be present in the out set
for must in ("r2_flat_wave1", "r4_flatkv2_wave1", "r1_perhead", "r5_decode_expert",
             "padcol_skv200_flatkv2", "decode_auto"):
    assert must in cpre["shapes"], f"required out-of-domain evidence missing: {must}"
    assert len(cpre["shapes"][must]["tqkv_calls"]) == len(cpost["shapes"][must]["tqkv_calls"])
print(f"\nSUMMARY: {len(pre) - nfail}/{len(pre)} PASS "
      f"(wave-2 covered {n_w2}, out-of-domain {n_out} >= 5 required)")
assert n_out >= 5, "out-of-domain evidence count below the preregistered >=5"
sys.exit(1 if nfail else 0)
