"""Gate-3 evidence: save gqa() outputs + main-kernel outputs + dispatch census
(iter9 P2 wave-2 — BSND direct write extended to prefill_expert + g2stack).

Wave-2 changes, on the newly-covered domain only (chosen in {prefill_expert,
g2stack} AND S % 64 == 0):
  * the main kernel's Output declaration BHSD [B,Nq,S_pad,D] -> BSND
    [B,S_pad,Nq,D] and the FIN write box to the strided form;
  * the host skips transpose_out entirely.
Wave-1 flat domains were ALREADY BSND in the shipped baseline (pre) and stay
untouched.  Everything else (per-head AUTO ladder — the EXCLUDED domain,
decode, non-aligned S, transpose_out/out_plan/_retile_out_plan, P1 retile) is
untouched.  Therefore after the gqa.py edit:
  * chosen dispatch spec per shape: IDENTICAL pre vs post;
  * FINAL gqa() output: BIT-IDENTICAL (raw-bit, NaN-safe) pre vs post for EVERY
    shape — pure data-path re-partition, zero arithmetic change;
  * MAIN-kernel output: shapes whose transpose_out runs in BOTH arms ->
    BIT-IDENTICAL; shapes whose transpose_out runs in PRE but is skipped in
    POST (the newly-covered wave-2 domain) -> transpose_out(main_pre_BHSD) ==
    main_post_BSND bit-exactly (main_post == final_post there);
  * the out_plan census must be unchanged wherever transpose_out still runs.

Capture: transpose_out is hooked to record its INPUT (the main kernel's raw
output).  Where transpose_out is skipped the main output is not recorded — the
comparator uses the final output as main_post there (documented above).

Usage (repo root, unset PYTHONPATH, ACL_OP_INIT_MODE=1):
    python3 examples/gqa/perf_tuning/save_gate3_outputs_iter9.py pre|post
Writes perf_tuning/gate3_iter9_outputs_{tag}.pt   (final + main outputs, cpu)
       perf_tuning/gate3_iter9_chosen_{tag}.json  (chosen + plan + skip census + md5)
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

# (name, B, S, S_kv, N_q, N_kv, D, causal, dtype, expected_chosen, category)
#   category: "wave2" = newly covered by THIS integration (BHSD pre -> BSND post)
#             "wave1" = already BSND in the shipped baseline (unchanged)
#             "out"   = out-of-domain (per-head excluded / decode / non-aligned)
SHAPES = [
    # ---- wave-2 newly covered: prefill_expert (aligned S) ----
    ("mha_g1_prefill_expert", 2, 128, 128, 32, 32, 128, True, torch.bfloat16, "prefill_expert", "wave2"),
    ("prexpert_g3_kv2", 2, 128, 256, 30, 10, 128, False, torch.float16, "prefill_expert", "wave2"),
    # ---- wave-2 newly covered: g2stack (aligned S) ----
    ("c9like_g2stack_kv4", 4, 512, 512, 64, 8, 128, False, torch.bfloat16, "g2stack", "wave2"),
    ("causal_kv2_g2s", 4, 256, 256, 64, 8, 128, True, torch.bfloat16, "g2stack", "wave2"),
    ("g2stack_ns1_stacked20", 1, 128, 128, 20, 2, 128, True, torch.float16, "g2stack", "wave2"),
    # ---- wave-1 already covered (must stay unchanged) ----
    ("r2_flat_kv1", 2, 128, 128, 32, 8, 128, True, torch.float16, "g2stack_flat", "wave1"),
    ("r4_flat_kv2", 4, 256, 256, 64, 8, 128, False, torch.bfloat16, "g2stack_flat_kv2", "wave1"),
    # ---- out-of-domain: per-head AUTO ladder (the EXCLUDED domain) ----
    ("r1_perhead", 1, 128, 128, 8, 2, 128, True, torch.float16, "per_head_bm64", "out"),
    ("d256_perhead", 2, 128, 128, 32, 8, 256, True, torch.float16, "per_head_bm64", "out"),
    # ---- out-of-domain: NON-ALIGNED S gate isolation (wave-1 + wave-2 domains) ----
    ("padrow_s100_flatkv2", 2, 100, 256, 32, 8, 128, False, torch.float16, "g2stack_flat_kv2", "out"),
    ("s100_g2stack_unaligned", 4, 100, 512, 64, 8, 128, False, torch.bfloat16, "g2stack", "out"),
    ("l1_s_tail_s130_g2stack", 2, 130, 130, 32, 8, 128, True, torch.float16, "g2stack", "out"),
    # ---- decode (no transpose_out at all) ----
    ("r5_decode_expert", 16, 1, 2048, 32, 8, 128, False, torch.float16, "dense_expert", "out"),
    ("decode_auto", 4, 2, 256, 16, 8, 128, False, torch.float16, "dense_auto", "out"),
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
        for name, B, S, Skv, Nq, Nkv, D, causal, dt, expect, cat in SHAPES:
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
            print(f"saved {name:26s} cat={cat:6s} chosen={chosen[name]} out_bsnd_attr={ob} "
                  f"tout_calls={cur.get('calls', 0)} plans={plans[name]} "
                  f"final={tuple(o.shape)} main={'yes' if name in mains else 'no'}", flush=True)
            assert chosen[name] and chosen[name][0] == expect, \
                f"{name}: chosen {chosen[name]} != expected {expect}"
    finally:
        G.transpose_out = orig_tout
        G.out_plan = orig_plan

    torch.save({"tag": tag, "finals": finals, "mains": mains},
               os.path.join(HERE, f"gate3_iter9_outputs_{tag}.pt"))
    meta = {
        "tag": tag,
        "gqa_py_md5": hashlib.md5(
            open(os.path.join(os.path.dirname(HERE), "gqa.py"), "rb").read()).hexdigest(),
        "chosen": chosen,
        "plans": plans,
        "tout_skipped": skips,
        "category": {s[0]: s[-1] for s in SHAPES},
    }
    with open(os.path.join(HERE, f"gate3_iter9_chosen_{tag}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[SAVE_DONE] tag={tag} shapes={len(finals)} mains_captured={len(mains)} "
          f"tout_skipped={[k for k, v in skips.items() if v]} md5={meta['gqa_py_md5']}")


if __name__ == "__main__":
    main()
