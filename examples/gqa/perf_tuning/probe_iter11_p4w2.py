"""Iter11 P4 wave-2 PRECHECK probe — applicability of the wave-1 conclusions to
g2stack + prefill_expert (lightweight, per the task book: NOT a full probe).

Scope (candidate_iter11_P4w2.json precheck block):
  E1 codegen audit (MANDATORY): 6 configs — both-arm sources dumped and
     compared (copy_gm_to_l1 census per dst buffer, realSrcN flips exactly
     D -> N_q*D (Q) / N_kv*D (K,V), p_l1 ws-load stays 128, realTailM/
     need_clear multiset identical, SetFlag/WaitFlag/DataCopy* whole-source
     counts identical).  Includes the two structural difference-carriers:
       * g2stack stacked-M Q pair (dst-side slices; per-box src algebra must
         match the wave-1 flat emit_qload)
       * runtime `if i % 2 == 0:` parity branches with the runtime idx window
         (the decode dense_expert production form)
       * N_kv=1 MQA degenerate (stride collapses to D — contiguous in BOTH
         arms; algebra must close without a gap)
       * use_mask=False configs (c9-like/MHA-noncausal: mask path eliminated)
  E2 minimal bitwise smoke: 2 small configs (c19-like prexpert causal +
     kv2-causal g2stack varlen) — BASE = the real production chain
     (transpose_qkv -> BHSD -> main, out_bsnd=True) vs W2 = variant factory
     in_bsnd=True + BSND originals.  raw-bit view(int16) + golden f6 policy.
  E3 directional main-only wall (probe-grade, host loop): c8-like + c19-like
     — catastrophic-regression tripwire only; quantitative Δmain is Gate-2
     msprof territory (wave-1 discipline).

Discipline: device SERIAL, no camodel, no fp16/bf16 torch d2d, gqa.py
ZERO touch (md5 asserted).

Run (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python3 examples/gqa/perf_tuning/probe_iter11_p4w2.py [--only SUBSTR] [--skip-e3]
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter

import torch
import torch_npu  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_DIR = os.path.dirname(HERE)
sys.path.insert(0, GQA_DIR)
sys.path.insert(0, HERE)

import gqa as G  # noqa: E402
import gqa_p4w2_inbsnd_iter11_variant as VAR  # noqa: E402

RTOL = 1e-2
EXPECT_MD5 = "f59b561212e60a50b78a51671999ef7c"

# (tag, factory name, B, S, S_kv, N_q, N_kv, D, causal, dtype, do_e2, do_e3)
CASES = [
    # g2stack: bench-exact c8 (ns=8, causal, use_mask=True)
    ("g2stack_c8_like", "_gqa_kernel_prefill_g2stack", 4, 1024, 1024, 32, 8, 128, True, torch.float16, False, True),
    # g2stack: bench-exact c9 (ns=4, NONcausal => use_mask=False)
    ("g2stack_c9_like", "_gqa_kernel_prefill_g2stack", 4, 512, 512, 64, 8, 128, False, torch.bfloat16, False, False),
    # g2stack: N_kv=1 MQA degenerate at the stacked==20 dispatch boundary
    # (ns=1; K/V realSrcN collapses to D — contiguous in BOTH arms)
    ("g2stack_ns1_mqa_nkv1", "_gqa_kernel_prefill_g2stack", 1, 128, 128, 20, 1, 128, True, torch.float16, False, False),
    # g2stack: kv2 causal varlen (small; E2 bitwise carrier)
    ("g2stack_kv2_causal", "_gqa_kernel_prefill_g2stack", 2, 128, 256, 32, 8, 128, True, torch.float16, True, False),
    # prefill_expert: bench-exact c19 (G=1 MHA, ns=1)
    ("prexpert_c19_like", "_gqa_kernel_prefill_expert", 2, 128, 128, 32, 32, 128, True, torch.bfloat16, True, True),
    # prefill_expert: MHA noncausal (use_mask=False branch)
    ("prexpert_mha_noncausal", "_gqa_kernel_prefill_expert", 2, 128, 128, 32, 32, 128, False, torch.float16, False, False),
]


def _atol(dtype):
    return 1e-3 if dtype == torch.float16 else 2e-2


def _rawbit(t):
    return t.view(torch.int16)


def _inputs(B, S, S_kv, N_q, N_kv, D, dtype, seed=1142):
    torch.manual_seed(seed + B + S + N_q)
    q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
    k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
    v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
    return q, k, v


def _mask_for(causal, B, S, S_kv, device):
    if causal:
        m = G._build_causal_mask(B, S, S_kv, device)
        torch.npu.synchronize()
        return m
    return G._dummy_mask(device)


def _build(case, in_bsnd):
    tag, fname, B, S, S_kv, N_q, N_kv, D, causal, dtype, _, _ = case
    S_pad = ((S + 63) // 64) * 64
    Skv_pad = ((S_kv + 127) // 128) * 128
    assert S_pad == S and Skv_pad == S_kv, f"precheck cases must be aligned: {tag}"
    sm_scale = 1.0 / (D**0.5)
    dstr = "float16" if dtype == torch.float16 else "bfloat16"
    use_mask = bool(causal or (Skv_pad != S_kv))
    kv_iters = Skv_pad // 128
    ns = min(kv_iters, 8)
    factory = getattr(VAR if in_bsnd else G, fname)
    kw = dict(N_q=N_q, N_kv=N_kv, D=D, dtype=dstr, block_M=64, skv_minus_s=S_kv - S,
              use_mask=use_mask, block_N=128, num_stages=ns, cross_interval=2, out_bsnd=True)
    if in_bsnd:
        kw["in_bsnd"] = True
    return factory(B, S_pad, Skv_pad, sm_scale, causal, **kw), use_mask, ns


def _golden(q, k, v, causal, out, dtype):
    ref = G._ref_gqa(q, k, v, -1.0, causal)
    torch.npu.synchronize()
    got = out.cpu().float()
    rf = ref.cpu().float()
    finite = torch.isfinite(rf) & torch.isfinite(got)
    diff = (got - rf).abs()
    atol = _atol(dtype)
    viol = int((diff[finite] > atol + RTOL * rf.abs()[finite]).sum())
    return (viol, int(torch.isnan(got).sum()), int(torch.isinf(got).sum()),
            (float(diff[finite].max()) if bool(finite.any()) else float("nan")))


# --------------------------------------------------------------------------
# E1 — generated-code audit (probe_iter10_P4.audit_codegen, adapted: the
# per-dst expected box census differs — g2stack q_l1 2 boxes / prexpert 1)
# --------------------------------------------------------------------------
GM2L1_RE = re.compile(r"copy_gm_to_l1<[^>]+>\([^;]*?;", re.S)


def audit_codegen(tag, kb, kw2, expect_qsrc, expect_kvsrc, expect_boxes):
    sb = kb.get_kernel_source()
    sw2 = kw2.get_kernel_source()
    open(os.path.join(HERE, f"codegen_iter11_p4w2_{tag}_base.cu"), "w").write(sb)
    open(os.path.join(HERE, f"codegen_iter11_p4w2_{tag}_w2.cu"), "w").write(sw2)

    def sites(src):
        out = []
        for m in GM2L1_RE.finditer(src):
            call = m.group(0)
            inner = call[call.index("(") + 1 : call.rindex(")")]
            depth, args, cur = 0, [], ""
            for ch in inner:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                if ch == "," and depth == 0:
                    args.append(cur.strip())
                    cur = ""
                else:
                    cur += ch
            args.append(cur.strip())
            out.append(args)  # dst, src, realSrcN, realTailM, realTailN, need_clear
        return out

    sb_sites, sw2_sites = sites(sb), sites(sw2)
    res = {
        "gm2l1_count_base": len(sb_sites),
        "gm2l1_count_w2": len(sw2_sites),
        "count_equal": len(sb_sites) == len(sw2_sites),
        "realSrcN_base": sorted({a[2] for a in sb_sites}),
        "realSrcN_w2": sorted({a[2] for a in sw2_sites}),
    }
    assert res["count_equal"], f"copy_gm_to_l1 count differs: {res['gm2l1_count_base']} vs {res['gm2l1_count_w2']}"

    def by_dst(site_list):
        d = {}
        for a in site_list:
            buf = a[0].split("[")[0].strip()
            buf = re.sub(r"_\d+$", "", buf)  # k_l1_0/k_l1_1 -> k_l1
            d.setdefault(buf, Counter())[a[2]] += 1
        return d

    db, dw = by_dst(sb_sites), by_dst(sw2_sites)
    res["by_dst_base"] = {k: dict(v) for k, v in db.items()}
    res["by_dst_w2"] = {k: dict(v) for k, v in dw.items()}
    assert set(db) == set(dw), f"dst buffer sets differ: {set(db)} vs {set(dw)}"
    for buf in db:
        assert sum(db[buf].values()) == sum(dw[buf].values()), f"{buf} site count differs"
        assert sum(db[buf].values()) == expect_boxes.get(buf), \
            f"{buf}: {sum(db[buf].values())} boxes, expected {expect_boxes.get(buf)}"
    # BASE: every Q/K/V load contiguous (realSrcN == D == 128)
    for buf, c in db.items():
        assert set(c) == {"128"}, f"BASE {buf} realSrcN not all D: {dict(c)}"
    # W2: q -> N_q*D, k/v -> N_kv*D (N_kv=1 collapses to 128 — same formula),
    # p_l1 (ws_p workspace load) MUST stay 128 in both arms
    assert set(dw.get("q_l1", {})) == {str(expect_qsrc)}, f"q_l1 strides: {dict(dw.get('q_l1', {}))}"
    for buf in ("k_l1", "v_l1"):
        assert set(dw.get(buf, {})) == {str(expect_kvsrc)}, f"{buf} strides: {dict(dw.get(buf, {}))}"
    assert set(dw.get("p_l1", {})) == {"128"}, f"p_l1 (ws_p) must stay contiguous: {dict(dw.get('p_l1', {}))}"
    # realTailM / need_clear invariance per site (box geometry + L1 clear protocol)
    tail_b = sorted((re.sub(r"_\d+$", "", a[0].split("[")[0].strip()), a[3], a[5]) for a in sb_sites)
    tail_w = sorted((re.sub(r"_\d+$", "", a[0].split("[")[0].strip()), a[3], a[5]) for a in sw2_sites)
    assert tail_b == tail_w, f"realTailM/need_clear multiset differs:\n{tail_b}\nvs\n{tail_w}"
    res["tail_needclear_identical"] = True
    res["tail_multiset"] = tail_b
    # whole-source instruction-count invariance (iter8 method)
    for pat in ("copy_gm_to_l1", "copy_l1_to_l0a", "copy_l1_to_l0b", "copy_l0c_to_gm",
                "copy_ub_to_gm", "copy_gm_to_ub", "DataCopyPad", "SetFlag", "WaitFlag"):
        cb, cw = sb.count(pat), sw2.count(pat)
        res[f"count_{pat}"] = [cb, cw]
        assert cb == cw, f"{pat} count differs: base {cb} vs w2 {cw}"
    return res


# --------------------------------------------------------------------------
def run_case(case, skip_e3):
    tag, fname, B, S, S_kv, N_q, N_kv, D, causal, dtype, do_e2, do_e3 = case
    do_e3 = do_e3 and not skip_e3
    kv_iters = S_kv // 128
    ns = min(kv_iters, 8)
    print(f"\n===== {tag} ({fname}, B{B} S{S} Skv{S_kv} Nq{N_q} Nkv{N_kv} D{D} "
          f"causal={causal} {dtype} ns={ns}) =====", flush=True)
    result = {"tag": tag, "ns": ns}

    t0 = time.time()
    kb, use_mask_b, _ = _build(case, in_bsnd=False)
    t1 = time.time()
    kw2, use_mask_w2, _ = _build(case, in_bsnd=True)
    t2 = time.time()
    assert use_mask_b == use_mask_w2
    print(f"[E1] compiled base {t1-t0:.1f}s / w2 {t2-t1:.1f}s (use_mask={use_mask_b})", flush=True)

    # PROBE-EXPECTATION FIX (disclosed, criteria untouched — iter7-10 harness-fix
    # precedent): at ns==1 the batch loop provably runs i==0 only, so the
    # compiler dead-eliminates the parity-else K/V boxes IDENTICALLY IN BOTH
    # ARMS (5 boxes: q x2(g2stack)/x1(prexpert) + k x1 + v x1 + p x1) — shipped
    # pre-existing behavior, verified base==w2.  Expected census is ns-aware.
    kv_boxes = 2 if ns >= 2 else 1
    expect_boxes = ({"q_l1": 2, "k_l1": kv_boxes, "v_l1": kv_boxes, "p_l1": 1} if "g2stack" in fname
                    else {"q_l1": 1, "k_l1": kv_boxes, "v_l1": kv_boxes, "p_l1": 1})
    res = audit_codegen(tag, kb, kw2, expect_qsrc=N_q * D, expect_kvsrc=N_kv * D,
                        expect_boxes=expect_boxes)
    result["e1_codegen"] = res
    print(f"[E1] copy_gm_to_l1 count base={res['gm2l1_count_base']} w2={res['gm2l1_count_w2']} "
          f"(equal, per-dst census {expect_boxes} verified); realSrcN base={res['realSrcN_base']} "
          f"w2={res['realSrcN_w2']} (expect Q={N_q*D}, KV={N_kv*D}); "
          f"tail/need_clear identical; SetFlag/WaitFlag/DataCopy* counts identical", flush=True)

    q, k, v = _inputs(B, S, S_kv, N_q, N_kv, D, dtype)
    mask = _mask_for(causal, B, S, S_kv, q.device)

    if do_e2:
        res_t = G.transpose_qkv(q, k, v, S_out=S, S_kv_out=S_kv)
        assert res_t is not None, "transpose_qkv returned None on an aligned probe shape"
        q_bhsd, k_bhsd, v_bhsd = res_t
        out_b = kb(q_bhsd, k_bhsd, v_bhsd, mask)
        torch.npu.synchronize()
        out_w2 = kw2(q, k, v, mask)
        torch.npu.synchronize()
        bit = bool(torch.equal(_rawbit(out_b.cpu()), _rawbit(out_w2.cpu())))
        viol_b, nan_b, inf_b, md_b = _golden(q, k, v, causal, out_b, dtype)
        viol_w, nan_w, inf_w, md_w = _golden(q, k, v, causal, out_w2, dtype)
        gok = viol_b == 0 and viol_w == 0 and nan_b == 0 and nan_w == 0 and inf_w == 0
        result["e2"] = {"bit_identical": bit, "golden_ok": gok,
                        "viol_base": viol_b, "viol_w2": viol_w,
                        "max_diff_w2": md_w, "out_shape": tuple(out_w2.shape)}
        print(f"[E2] bit_identical={bit} golden viol_base={viol_b} viol_w2={viol_w} "
              f"nan={nan_w} max_diff={md_w:.3e} out_shape={tuple(out_w2.shape)}", flush=True)
    else:
        q_bhsd = k_bhsd = v_bhsd = None

    if do_e3:
        if q_bhsd is None:
            res_t = G.transpose_qkv(q, k, v, S_out=S, S_kv_out=S_kv)
            q_bhsd, k_bhsd, v_bhsd = res_t
            torch.npu.synchronize()
        n = 10

        def main_only_base():
            return kb(q_bhsd, k_bhsd, v_bhsd, mask)

        def main_only_w2():
            return kw2(q, k, v, mask)

        def timeit(fn, reps=n):
            for _ in range(3):
                fn()
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            torch.npu.synchronize()
            return (time.perf_counter() - t0) / reps * 1e6

        import statistics

        m_b, m_w = [], []
        for _ in range(3):
            m_b.append(timeit(main_only_base))
            m_w.append(timeit(main_only_w2))
        mb, mw = statistics.median(m_b), statistics.median(m_w)
        result["e3_wall_us"] = {"main_base": round(mb, 2), "main_w2": round(mw, 2),
                                "main_delta_us": round(mw - mb, 2),
                                "main_delta_pct": round((mw - mb) / mb * 100, 2)}
        print(f"[E3] main-only wall(probe-grade, host-loop ~180us overhead): base {mb:.2f} -> "
              f"w2 {mw:.2f} us (delta {mw-mb:+.2f} = {(mw-mb)/mb*100:+.2f}%) — DIRECTIONAL ONLY",
              flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip-e3", action="store_true")
    args = ap.parse_args()

    md5 = hashlib.md5(open(os.path.join(GQA_DIR, "gqa.py"), "rb").read()).hexdigest()
    assert md5 == EXPECT_MD5, f"gqa.py drifted: {md5}"

    cases = [c for c in CASES if not args.only or args.only in c[0]]
    results = []
    nfail = 0
    for case in cases:
        r = run_case(case, skip_e3=args.skip_e3)
        ok = r["e1_codegen"]["count_equal"] and r.get("e2", {}).get("bit_identical", True) \
            and r.get("e2", {}).get("golden_ok", True)
        r["verdict"] = "PASS" if ok else "FAIL"
        if not ok:
            nfail += 1
        results.append(r)
        print(f"[{case[0]}] verdict={r['verdict']}", flush=True)

    out = os.path.join(HERE, "probe_iter11_p4w2_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"\n===== PRECHECK SUMMARY: {len(results)-nfail}/{len(results)} PASS -> {out}")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
