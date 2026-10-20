"""Gate-3 comparator (iter8 P2): pre vs post integration bit-identity + census.

Criteria (preregistered in candidate_iter8_P2.json, Gate 3):
  1. FINAL gqa() output BIT-IDENTICAL (raw-bit, NaN-safe) pre vs post for
     EVERY shape — covered and out-of-domain alike.
  2. MAIN-kernel output:
       out-of-domain shapes -> BIT-IDENTICAL BHSD tensor pre vs post;
       covered shapes       -> transpose_out(main_pre_BHSD) == main_post_BSND
                               bit-exactly (main_post == final_post there, as
                               transpose_out is skipped by design).
  3. chosen dispatch spec identical pre vs post for every shape.
  4. transpose_out SKIPPED post-integration exactly on the covered shapes and
     still RUNNING (with the identical P1-retiled plan) everywhere else.
  5. gqa.py md5 recorded for both tags (pre must equal the shipped baseline
     4b4e453e4b8f2922baefbecb85912494).

Run (repo root): python3 examples/gqa/perf_tuning/compare_gate3_iter8.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch
import torch_npu  # noqa: F401

import gqa as G
from save_gate3_outputs_iter8 import SHAPES

SHIPPED_MD5 = "4b4e453e4b8f2922baefbecb85912494"


def rawbit(t):
    return t.view(torch.int16)


def main():
    pre = torch.load(os.path.join(HERE, "gate3_iter8_outputs_pre.pt"), weights_only=False)
    post = torch.load(os.path.join(HERE, "gate3_iter8_outputs_post.pt"), weights_only=False)
    mpre = json.load(open(os.path.join(HERE, "gate3_iter8_chosen_pre.json")))
    mpost = json.load(open(os.path.join(HERE, "gate3_iter8_chosen_post.json")))

    print(f"gqa.py md5: pre={mpre['gqa_py_md5']} post={mpost['gqa_py_md5']}")
    results = []
    assert mpre["gqa_py_md5"] == SHIPPED_MD5, "PRE capture was not on the shipped baseline!"

    for name, B, S, Skv, Nq, Nkv, D, causal, dt, expect, cov in SHAPES:
        f_pre, f_post = pre["finals"][name], post["finals"][name]
        fin_bit = bool(torch.equal(rawbit(f_pre), rawbit(f_post)))
        shape_ok = tuple(f_post.shape) == (B, S, Nq, D) and f_post.is_contiguous()
        chosen_ok = mpre["chosen"][name] == mpost["chosen"][name] == [expect]
        # plan census: on COVERED shapes transpose_out is eliminated, so
        # out_plan is never called post-integration => the census must be
        # EMPTY (that emptiness IS part of the elimination evidence).  On
        # every out-of-domain prefill shape the census must be IDENTICAL
        # pre vs post (iter7's P1 retile stays live there).  Decode: empty
        # in both runs.
        if S <= 4:
            plans_ok = mpre["plans"][name] == [] == mpost["plans"][name]
        elif cov:
            plans_ok = (mpost["plans"][name] == [] and len(mpre["plans"][name]) > 0)
        else:
            plans_ok = (mpre["plans"][name] == mpost["plans"][name]
                        and len(mpost["plans"][name]) > 0)

        main_pre = pre["mains"].get(name)
        main_post = post["mains"].get(name)
        if S <= 4:  # decode: no transpose_out in either run
            main_ok, main_note = (main_pre is None and main_post is None), "decode (no tout)"
        elif cov:
            # covered: post skips tout, so the main output IS the final output
            mp = f_post
            assert main_post is None, f"{name}: transpose_out still ran post-integration"
            # main_pre was saved on CPU; transpose_out is a device kernel
            t_pre = G.transpose_out(main_pre.npu().contiguous())
            torch.npu.synchronize()
            main_ok = bool(torch.equal(rawbit(t_pre.cpu()), rawbit(mp)))
            main_note = (f"transpose_out(main_pre{tuple(main_pre.shape)}) == main_post"
                         f"{tuple(mp.shape)} -> {main_ok}")
        else:
            main_ok = (main_pre is not None and main_post is not None
                       and bool(torch.equal(rawbit(main_pre), rawbit(main_post))))
            main_note = (f"main BHSD bit-identical {tuple(main_pre.shape) if main_pre is not None else None}"
                         f" vs {tuple(main_post.shape) if main_post is not None else None}")
        tout_ok = (mpost["tout_skipped"][name] is True) if cov else \
                  (mpost["tout_skipped"][name] is False)
        ok = fin_bit and shape_ok and chosen_ok and plans_ok and main_ok and tout_ok
        results.append((name, ok))
        print(f"[{'G3_PASS' if ok else 'G3_FAIL'}] {name:24s} covered={str(cov):5s} "
              f"final_bit={fin_bit} shape/contig={shape_ok} chosen={chosen_ok} "
              f"plans={plans_ok}(post={mpost['plans'][name]}) tout_skip={'yes' if mpost['tout_skipped'][name] else 'no'}"
              f"(expected {'yes' if cov else 'no'}) -> {tout_ok} | main: {main_note}")

    n_ok = sum(1 for _, r in results if r)
    print(f"\n[G3_SUMMARY] {n_ok}/{len(results)} shapes fully verified")
    for n, r in results:
        if not r:
            print(f"  FAILED: {n}")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
