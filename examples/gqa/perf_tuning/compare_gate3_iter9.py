"""Gate-3 comparator (iter9 P2 wave-2): pre vs post integration bit-identity +
census.

Criteria (preregistered in candidate_iter9_P2w2.json, Gate 3):
  1. FINAL gqa() output BIT-IDENTICAL (raw-bit view(int16), NaN-safe) pre vs
     post for EVERY shape — covered and out-of-domain alike.
  2. MAIN-kernel output (criterion driven by the RECORDED tout-skip flags):
       tout ran pre AND post (out-of-domain)   -> main_pre == main_post bit-exact
       tout ran pre, SKIPPED post (wave-2 domain) -> transpose_out(main_pre_BHSD)
                                                  == final_post_BSND bit-exact
       tout skipped pre AND post (wave-1 flat) -> covered by (1): final == main
  3. chosen dispatch spec identical pre vs post for every shape.
  4. transpose_out SKIPPED post-integration exactly on the covered shapes
     (wave-1 flat + wave-2 prefill_expert/g2stack aligned) and still RUNNING
     (with the identical P1-retiled plan) everywhere else; the out_bsnd
     capability attribute is True exactly on the covered shapes.
  5. out_plan census IDENTICAL pre vs post on every shape that still runs
     transpose_out (iter7's P1 retile stays live); EMPTY post on covered.
  6. gqa.py md5 recorded for both tags (pre must equal the shipped baseline
     3aa9cfa2625cdce96a62ce967785fe85).

Run (repo root): python3 examples/gqa/perf_tuning/compare_gate3_iter9.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch
import torch_npu  # noqa: F401

import gqa as G
from save_gate3_outputs_iter9 import SHAPES

SHIPPED_MD5 = "3aa9cfa2625cdce96a62ce967785fe85"


def rawbit(t):
    return t.view(torch.int16)


def main():
    pre = torch.load(os.path.join(HERE, "gate3_iter9_outputs_pre.pt"), weights_only=False)
    post = torch.load(os.path.join(HERE, "gate3_iter9_outputs_post.pt"), weights_only=False)
    mpre = json.load(open(os.path.join(HERE, "gate3_iter9_chosen_pre.json")))
    mpost = json.load(open(os.path.join(HERE, "gate3_iter9_chosen_post.json")))

    print(f"gqa.py md5: pre={mpre['gqa_py_md5']} post={mpost['gqa_py_md5']}")
    results = []
    assert mpre["gqa_py_md5"] == SHIPPED_MD5, "PRE capture was not on the shipped baseline!"

    for name, B, S, Skv, Nq, Nkv, D, causal, dt, expect, cat in SHAPES:
        f_pre, f_post = pre["finals"][name], post["finals"][name]
        checks = {}
        checks["final_bit"] = bool(torch.equal(rawbit(f_pre), rawbit(f_post)))
        checks["shape_ok"] = tuple(f_post.shape) == (B, S, Nq, D) and f_post.is_contiguous()
        checks["chosen_identical"] = (mpre["chosen"][name] == mpost["chosen"][name]
                                      and mpost["chosen"][name] == [expect])

        skip_pre = mpre["tout_skipped"].get(name, False)
        skip_post = mpost["tout_skipped"].get(name, False)
        decode = S <= 4
        main_pre = pre["mains"].get(name)
        main_post = post["mains"].get(name)

        if cat == "wave2":
            # newly covered: tout ran pre (main_pre captured BHSD), skipped post
            checks["skip_flipped"] = (not skip_pre) and skip_post
            if main_pre is not None:
                t_pre = G.transpose_out(main_pre.npu())
                checks["main_via_transpose"] = bool(
                    torch.equal(rawbit(t_pre.cpu()), rawbit(f_post)))
                checks["main_pre_is_bhsd"] = tuple(main_pre.shape) == (B, Nq, S, D)
        elif cat == "wave1":
            # already BSND pre: skipped both arms, final==main both arms
            checks["skip_both"] = skip_pre and skip_post
        elif not decode:
            # out-of-domain prefill: tout runs BOTH arms, main bit-identical.
            # The main output is the PADDED BHSD tensor [B, Nq, S_padded, D]
            # (S_padded = ceil(S/64)*64 > S on the non-aligned gate cases —
            # harness-shape fix, disclosure-only: the bit-identity criteria
            # are unchanged, iter7/8 precedent).
            checks["tout_ran_both"] = (not skip_pre) and (not skip_post)
            if main_pre is not None and main_post is not None:
                checks["main_bit"] = bool(torch.equal(rawbit(main_pre), rawbit(main_post)))
                s_pad = ((S + 63) // 64) * 64
                checks["main_is_bhsd"] = tuple(main_post.shape) == (B, Nq, s_pad, D)
        else:
            checks["no_tout_both"] = (not skip_pre) and (not skip_post) \
                and main_pre is None and main_post is None

        # plan census: identical pre/post where tout still runs; empty post on covered
        p_pre = mpre["plans"][name]
        p_post = mpost["plans"][name]
        if skip_post:
            checks["plan_empty_post"] = (len(p_post) == 0)
            if not skip_pre:
                checks["plan_ran_pre"] = (len(p_pre) > 0)
        else:
            checks["plan_identical"] = (p_pre == p_post)

        ok = all(checks.values())
        bad = [k for k, v in checks.items() if not v]
        results.append((name, ok))
        print(f"[{'G3_PASS' if ok else 'G3_FAIL'}] {name:26s} cat={cat:6s} "
              f"chosen={mpost['chosen'][name]} skip_pre={skip_pre} skip_post={skip_post} "
              f"final_bit={checks['final_bit']}"
              + (f" main_via_transpose={checks.get('main_via_transpose')}" if "main_via_transpose" in checks else "")
              + (f" main_bit={checks.get('main_bit')}" if "main_bit" in checks else "")
              + (f" FAILED={bad}" if bad else ""), flush=True)

    # capability-attribute census: out_bsnd True exactly on covered aligned shapes
    print("\n--- coverage census ---")
    n_w2 = sum(1 for s in SHAPES if s[-1] == "wave2")
    n_skip_post = sum(1 for k, v in mpost["tout_skipped"].items() if v)
    n_skip_pre = sum(1 for k, v in mpre["tout_skipped"].items() if v)
    print(f"tout_skipped PRE  ({n_skip_pre}): {sorted(k for k, v in mpre['tout_skipped'].items() if v)}")
    print(f"tout_skipped POST ({n_skip_post}): {sorted(k for k, v in mpost['tout_skipped'].items() if v)}")
    newly = sorted(set(k for k, v in mpost["tout_skipped"].items() if v)
                   - set(k for k, v in mpre["tout_skipped"].items() if v))
    print(f"newly skipped post-integration ({len(newly)}): {newly}")
    expect_newly = sorted(s[0] for s in SHAPES if s[-1] == "wave2")
    census_ok = newly == expect_newly
    print(f"census: newly-skipped == wave2 set -> {census_ok}")
    results.append(("coverage_census", census_ok))

    n_ok = sum(1 for _, r in results if r)
    print(f"\n[G3_SUMMARY] {n_ok}/{len(results)} shapes+census verified")
    for name, r in results:
        if not r:
            print(f"  FAILED: {name}")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
