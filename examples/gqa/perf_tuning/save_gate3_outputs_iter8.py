"""Gate-3 evidence: save gqa() outputs + main-kernel outputs + dispatch census
(iter8 P2 — BSND direct write, transpose_out eliminated on the covered domain).

P2 changes, on the COVERED domain only (chosen in {g2stack_flat,
g2stack_flat_kv2} AND S % 64 == 0):
  * the main kernel's Output declaration BHSD [B,Nq,S_pad,D] -> BSND
    [B,S_pad,Nq,D] and the FIN write box to the strided form;
  * the host skips transpose_out entirely.
Everything else (out-of-domain kernels, non-aligned S, decode, the
transpose_out/out_plan/_retile_out_plan code, P1's retile behaviour) is
untouched.  Therefore after the gqa.py edit:
  * chosen dispatch spec per shape: must be IDENTICAL pre vs post;
  * FINAL gqa() output: must be BIT-IDENTICAL (raw-bit, NaN-safe) pre vs post
    for EVERY shape — P2 is a pure data-path re-partition, zero arithmetic
    change, so anything else is a bug;
  * MAIN-kernel output: out-of-domain shapes must be BIT-IDENTICAL (same BHSD
    tensor); covered shapes change layout by design, so the criterion is
    transpose_out(main_pre_BHSD) == main_post_BSND bit-exactly (main_post IS
    the final output there, since transpose_out is skipped);
  * the out_plan census must be unchanged (P1 retile still live for every
    shape that still runs transpose_out).

Capture mechanics: transpose_out is hooked to record its INPUT tensor (the
main kernel's raw output).  On covered shapes in the POST run transpose_out is
not called at all, so the recorded main output is absent — the comparator uses
the final output as main_post there (documented above).

Usage (repo root, unset PYTHONPATH, ACL_OP_INIT_MODE=1):
    python3 examples/gqa/perf_tuning/save_gate3_outputs_iter8.py pre|post
Writes perf_tuning/gate3_iter8_outputs_{tag}.pt   (final + main outputs, cpu)
       perf_tuning/gate3_iter8_chosen_{tag}.json  (chosen specs + plan census
                                                   + tout-skip census + md5)
"""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch
import torch_npu  # noqa: F401

import gqa as G

# (name, B, S, S_kv, N_q, N_kv, D, causal, dtype, expected_chosen, p2_covered)
SHAPES = [
    # ---- covered domain (P2 active post-integration) ----
    ("r2_flat_kv1", 2, 128, 128, 32, 8, 128, True, torch.float16, "g2stack_flat", True),
    ("r3_flat_kv1", 2, 128, 128, 128, 8, 128, True, torch.bfloat16, "g2stack_flat", True),
    ("r6_flat_kv1", 2, 128, 128, 48, 4, 128, False, torch.float16, "g2stack_flat", True),
    ("c18_mqa_flat_kv1", 2, 128, 128, 32, 1, 128, False, torch.float16, "g2stack_flat", True),
    ("r4_flat_kv2", 4, 256, 256, 64, 8, 128, False, torch.bfloat16, "g2stack_flat_kv2", True),
    ("c7like_kv2", 2, 128, 256, 32, 8, 128, False, torch.bfloat16, "g2stack_flat_kv2", True),
    ("padcol_skv200_kv2", 2, 128, 200, 32, 8, 128, False, torch.float16, "g2stack_flat_kv2", True),
    # ---- out-of-domain: per-head / prefill_expert / g2stack / D=256 ----
    ("r1_perhead", 1, 128, 128, 8, 2, 128, True, torch.float16, "per_head_bm64", False),
    ("d256_perhead", 2, 128, 128, 32, 8, 256, True, torch.float16, "per_head_bm64", False),
    ("mha_g1_prefill_expert", 2, 128, 128, 32, 32, 128, True, torch.bfloat16, "prefill_expert", False),
    ("c9like_g2stack_kv4", 4, 512, 512, 64, 8, 128, False, torch.bfloat16, "g2stack", False),
    ("causal_kv2_g2s", 4, 256, 256, 64, 8, 128, True, torch.bfloat16, "g2stack", False),
    # ---- out-of-domain: NON-ALIGNED S inside the flat dispatch (gate test) ----
    ("padrow_s100_flatkv2", 2, 100, 256, 32, 8, 128, False, torch.float16, "g2stack_flat_kv2", False),
    # S=124 (non-aligned) but kv_iters==1 => the flat kv1 dispatch: a SECOND
    # wave-1 gate-isolation case (flat-eligible, alignment gate must exclude it)
    ("l1_s_tail_flatkv1", 2, 124, 128, 32, 8, 128, True, torch.float16, "g2stack_flat", False),
    # ---- decode (no transpose_out at all) ----
    ("r5_decode_expert", 16, 1, 2048, 32, 8, 128, False, torch.float16, "dense_expert", False),
    ("decode_auto", 4, 2, 256, 16, 8, 128, False, torch.float16, "dense_auto", False),
]


def main():
    tag = sys.argv[1]
    assert tag in ("pre", "post")
    finals, mains, chosen, plans, skips = {}, {}, {}, {}, {}

    orig_tout = G.transpose_out
    cur = {}

    def hooked(out_bhsd, s_real=None):
        cur["main"] = out_bhsd.detach().clone()
        cur["s_real"] = s_real
        cur["calls"] = cur.get("calls", 0) + 1
        return orig_tout(out_bhsd, s_real=s_real)

    orig_plan = G.out_plan
    cur_plans = []

    def hooked_plan(Bx, Nqx, Sx, Dx):
        r = orig_plan(Bx, Nqx, Sx, Dx)
        cur_plans.append(((Bx, Nqx, Sx, Dx), r[1:] if r else None))
        return r

    G.transpose_out = hooked
    G.out_plan = hooked_plan
    try:
        for name, B, S, Skv, Nq, Nkv, D, causal, dt, expect, cov in SHAPES:
            torch.manual_seed(1234 + B + S + Nq)
            q = (torch.rand(B, S, Nq, D) * 2 - 1).to(dt).npu()
            k = (torch.rand(B, Skv, Nkv, D) * 2 - 1).to(dt).npu()
            v = (torch.rand(B, Skv, Nkv, D) * 2 - 1).to(dt).npu()
            G._KERNEL_BUILD_CACHE.clear()
            G._COMPILED_CACHE.clear()
            cur.clear()
            cur_plans.clear()
            o = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
            torch.npu.synchronize()
            finals[name] = o.cpu()
            if "main" in cur:
                mains[name] = cur["main"].cpu()
            skips[name] = (cur.get("calls", 0) == 0 and S > 4)  # prefill but no tout => skipped
            chosen[name] = [val[2] for val in G._KERNEL_BUILD_CACHE.values()]
            plans[name] = [(list(k_), (list(v_) if v_ else None)) for k_, v_ in cur_plans]
            ob = getattr(list(G._COMPILED_CACHE.values())[0], "out_bsnd", None) \
                if G._COMPILED_CACHE else None
            print(f"saved {name:24s} chosen={chosen[name]} out_bsnd_attr={ob} "
                  f"tout_calls={cur.get('calls', 0)} plans={plans[name]} "
                  f"final={tuple(o.shape)} main={'yes' if name in mains else 'no'}", flush=True)
            assert chosen[name] and chosen[name][0] == expect, \
                f"{name}: chosen {chosen[name]} != expected {expect}"
    finally:
        G.transpose_out = orig_tout
        G.out_plan = orig_plan

    torch.save({"tag": tag, "finals": finals, "mains": mains},
               os.path.join(HERE, f"gate3_iter8_outputs_{tag}.pt"))
    meta = {
        "tag": tag,
        "gqa_py_md5": hashlib.md5(
            open(os.path.join(os.path.dirname(HERE), "gqa.py"), "rb").read()).hexdigest(),
        "chosen": chosen,
        "plans": plans,
        "tout_skipped": skips,
        "covered_expected": {n: c for n, *_x, c in [(s[0], s[-1]) for s in SHAPES]},
    }
    with open(os.path.join(HERE, f"gate3_iter8_chosen_{tag}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[SAVE_DONE] tag={tag} shapes={len(finals)} mains_captured={len(mains)} "
          f"tout_skipped={[k for k, v in skips.items() if v]} md5={meta['gqa_py_md5']}")


if __name__ == "__main__":
    main()
