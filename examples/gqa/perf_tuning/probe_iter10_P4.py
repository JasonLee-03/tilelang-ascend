"""Iter10 P4 probe — transpose_qkv elimination feasibility (E1/E2/E3).

Standalone probe: does NOT modify examples/gqa/gqa.py or test_gqa.py.  Both
arms are built DIRECTLY from the factories (no adapter monkeypatching — the
probe measures the kernel-level mechanism; the host-side skip is Gate-0/2
harness territory).  SERIAL execution only (Fixed-Core-20 expert kernels,
device-serialization discipline).

E1 — compile + generated-code audit (expressiveness + Nz-layout gate):
  BASE arm: shipped `_gqa_kernel_prefill_g2stack_flat{,_kv2}` (gqa.py, md5
            606b07ff...) with BHSD inputs produced by the shipped
            transpose_qkv (the exact production chain).
  P4   arm: variant factory (gqa_p4_inbsnd_iter10_variant.py, audited 5-site
            diff) with in_bsnd=True reading the caller's BSND tensors.
  Audit: count copy_gm_to_l1 call sites (MUST be equal — instruction-count
  invariance, iter8 minimal-perturbation principle) and verify the runtime
  realSrcN argument of every Q/K/V load flips D -> N*D exactly as
  src/op/ascend.cc compute_strideN prescribes (singleton head dim folded
  into the row stride).

E2 — minimal correctness smoke (bit-identity, raw-bit view(int16), iter8
  finding 3): final outputs of the two arms MUST be bit-identical on
  finite / NaN-inf-injected / fullrange cases + golden viol invariance
  (f6 policy: NPU input-dtype golden, fp16 atol 1e-3 / bf16 2e-2, rtol 1e-2).

E3 — directional timing (probe-grade, NOT the adoption criterion):
  interleaved same-session wall timing of the two full chains
  (BASE = transpose_qkv + main; P4 = main only) + main-kernel-only.
  The decision-grade A/B is the Gate-2 msprof B-matrix (e2e = Σ TaskDur).

Run (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python3 examples/gqa/perf_tuning/probe_iter10_P4.py [--only SUBSTR] [--skip-e3]
"""

import argparse
import json
import os
import re
import sys
import time

import torch
import torch_npu  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_DIR = os.path.dirname(HERE)
sys.path.insert(0, GQA_DIR)
sys.path.insert(0, HERE)

import gqa as G  # noqa: E402
import gqa_p4_inbsnd_iter10_variant as VAR  # noqa: E402

RTOL = 1e-2

# (tag, factory name, B, S, S_kv, N_q, N_kv, D, causal, dtype)
CASES = [
    # flat kv1 domain
    ("r2_like_nc", "_gqa_kernel_prefill_g2stack_flat", 2, 128, 128, 48, 4, 128, False, torch.float16),
    ("c1_like_causal", "_gqa_kernel_prefill_g2stack_flat", 2, 128, 128, 32, 8, 128, True, torch.float16),
    # flat kv2 domain
    ("r4_like_kv2", "_gqa_kernel_prefill_g2stack_flat_kv2", 4, 256, 256, 64, 8, 128, False, torch.bfloat16),
]


def _atol(dtype):
    return 1e-3 if dtype == torch.float16 else 2e-2


def _rawbit(t):
    return t.view(torch.int16)


def _inputs(tag, B, S, S_kv, N_q, N_kv, D, dtype):
    if tag.startswith("fullrange"):
        lo, hi = -65504.0, 65504.0

        def gen(shape, off):
            gg = torch.Generator().manual_seed(3101 + off)
            t = torch.rand(shape, dtype=torch.float64, generator=gg) * (hi - lo) + lo
            return t.to(dtype).npu()

        return gen((B, S, N_q, D), 0), gen((B, S_kv, N_kv, D), 1), gen((B, S_kv, N_kv, D), 2)
    torch.manual_seed(42 + B + S + S_kv + N_q)
    q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
    k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
    v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
    if tag.startswith("boundary_sparse"):
        # iter5 verbatim injection set (kv1 shape: single kv block; the NaN/Inf
        # must propagate through scores/weights to the output rows)
        q[0, 5, 3, 0] = float("nan")
        v[0, 20, 1, 7] = float("inf")
        v[1, 100, 5, 9] = float("-inf")
    return q, k, v


def _special_struct_equal(a, b):
    nan_a, nan_b = torch.isnan(a), torch.isnan(b)
    if not bool((nan_a == nan_b).all()):
        return False, f"nan_pos_mismatch={int((nan_a != nan_b).sum())}"
    inf_a, inf_b = torch.isinf(a), torch.isinf(b)
    if not bool((inf_a == inf_b).all()):
        return False, f"inf_pos_mismatch={int((inf_a != inf_b).sum())}"
    both_inf = inf_a & inf_b
    if bool(both_inf.any()) and not bool((torch.sign(a[both_inf]) == torch.sign(b[both_inf])).all()):
        return False, "inf_sign_mismatch"
    finite = ~nan_a & ~inf_a
    if not bool(torch.equal(a[finite], b[finite])):
        return False, f"finite_bit_mismatch={int((a[finite] != b[finite]).sum())}"
    return True, "ok"


def _mask_for(causal, B, S, S_kv, S_pad, Skv_pad, device):
    """Same mask the adapter/_prefill_run_padded supply on the aligned domain."""
    if causal:
        m = G._build_causal_mask(B, S, S_kv, device)
        torch.npu.synchronize()
        return m
    return G._dummy_mask(device)


def _build(case, in_bsnd):
    tag, fname, B, S, S_kv, N_q, N_kv, D, causal, dtype = case
    S_pad = ((S + 63) // 64) * 64
    Skv_pad = ((S_kv + 127) // 128) * 128
    assert S_pad == S and Skv_pad == S_kv, "probe cases must be aligned"
    sm_scale = 1.0 / (D**0.5)
    dstr = "float16" if dtype == torch.float16 else "bfloat16"
    use_mask = bool(causal or (Skv_pad != S_kv))
    factory = getattr(VAR if in_bsnd else G, fname)
    kw = dict(N_q=N_q, N_kv=N_kv, D=D, dtype=dstr, block_M=64, skv_minus_s=S_kv - S,
              use_mask=use_mask, block_N=128, out_bsnd=True)
    if in_bsnd:
        kw["in_bsnd"] = True
    return factory(B, S_pad, Skv_pad, sm_scale, causal, **kw), use_mask


def _golden(q, k, v, causal, out, dtype):
    ref = G._ref_gqa(q, k, v, -1.0, causal)
    torch.npu.synchronize()
    got = out.cpu().float()
    rf = ref.cpu().float()
    finite = torch.isfinite(rf) & torch.isfinite(got)
    diff = (got - rf).abs()
    atol = _atol(dtype)
    viol = int((diff[finite] > atol + RTOL * rf.abs()[finite]).sum())
    return (viol, int(torch.isnan(got).sum()), int(torch.isnan(rf).sum()), int(torch.isinf(got).sum()),
            (float(diff[finite].max()) if bool(finite.any()) else float("nan")))


# --------------------------------------------------------------------------
# E1 — generated-code audit
# --------------------------------------------------------------------------
GM2L1_RE = re.compile(r"copy_gm_to_l1<[^>]+>\([^;]*?;", re.S)


def audit_codegen(tag, kb, kp4, expect_qsrc, expect_kvsrc):
    """Count copy_gm_to_l1 sites and collect their realSrcN (3rd runtime arg).

    Emitted form: copy_gm_to_l1<T, dstM, dstN>(dst[off], src[off], realSrcN,
    realTailM, realTailN, need_clear) — codegen_ascend.cc:2936 (3 extra args)
    + the appended need_clear flag.
    """
    sb = kb.get_kernel_source()
    sp4 = kp4.get_kernel_source()
    open(os.path.join(HERE, f"codegen_iter10_p4_{tag}_base.cu"), "w").write(sb)
    open(os.path.join(HERE, f"codegen_iter10_p4_{tag}_p4.cu"), "w").write(sp4)

    def sites(src):
        out = []
        for m in GM2L1_RE.finditer(src):
            call = m.group(0)
            # split top-level args (no nested parens inside the runtime args
            # for this op form; offsets are plain identifiers/ints)
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
            # args: dst, src, realSrcN, realTailM, realTailN, need_clear
            out.append(args)
        return out

    sb_sites, sp4_sites = sites(sb), sites(sp4)
    res = {
        "gm2l1_count_base": len(sb_sites),
        "gm2l1_count_p4": len(sp4_sites),
        "count_equal": len(sb_sites) == len(sp4_sites),
        "realSrcN_base": sorted({a[2] for a in sb_sites}),
        "realSrcN_p4": sorted({a[2] for a in sp4_sites}),
    }
    assert res["count_equal"], f"copy_gm_to_l1 count differs: {res['gm2l1_count_base']} vs {res['gm2l1_count_p4']}"

    # classify by destination L1 buffer (q_l1 / k_l1_* / v_l1_* / p_l1):
    # the Q/K/V boxes must flip realSrcN D -> N*D; the ws_p -> p_l1 workspace
    # load (also copy_gm_to_l1, contiguous [128,128]) must stay 128 in BOTH.
    from collections import Counter

    def by_dst(site_list):
        d = {}
        for a in site_list:
            buf = a[0].split("[")[0].strip()
            buf = re.sub(r"_\d+$", "", buf)  # k_l1_0/k_l1_1 -> k_l1
            d.setdefault(buf, Counter())[a[2]] += 1
        return d

    db, dp = by_dst(sb_sites), by_dst(sp4_sites)
    res["by_dst_base"] = {k: dict(v) for k, v in db.items()}
    res["by_dst_p4"] = {k: dict(v) for k, v in dp.items()}
    assert set(db) == set(dp), f"dst buffer sets differ: {set(db)} vs {set(dp)}"
    for buf in db:
        assert sum(db[buf].values()) == sum(dp[buf].values()), f"{buf} site count differs"
    # BASE: every load contiguous (realSrcN == D == 128)
    for buf, c in db.items():
        assert set(c) == {"128"}, f"BASE {buf} realSrcN not all D: {dict(c)}"
    # P4: q -> N_q*D, k/v -> N_kv*D, p (ws_p) -> 128 unchanged
    assert set(dp.get("q_l1", {})) == {str(expect_qsrc)}, f"q_l1 strides: {dict(dp.get('q_l1', {}))}"
    for buf in ("k_l1", "v_l1"):
        assert set(dp.get(buf, {})) == {str(expect_kvsrc)}, f"{buf} strides: {dict(dp.get(buf, {}))}"
    assert set(dp.get("p_l1", {})) == {"128"}, f"p_l1 (ws_p) must stay contiguous: {dict(dp.get('p_l1', {}))}"
    # realTailM / need_clear invariance per site (box geometry + L1 clear protocol)
    tail_b = sorted((re.sub(r"_\d+$", "", a[0].split("[")[0].strip()), a[3], a[5]) for a in sb_sites)
    tail_p = sorted((re.sub(r"_\d+$", "", a[0].split("[")[0].strip()), a[3], a[5]) for a in sp4_sites)
    assert tail_b == tail_p, f"realTailM/need_clear multiset differs:\n{tail_b}\nvs\n{tail_p}"
    res["tail_needclear_identical"] = True
    # DataCopy* census (whole-source instruction count invariance, iter8 method)
    for pat in ("copy_gm_to_l1", "copy_l1_to_l0a", "copy_l1_to_l0b", "copy_l0c_to_gm",
                "copy_ub_to_gm", "copy_gm_to_ub", "DataCopyPad", "SetFlag", "WaitFlag"):
        cb, cp = sb.count(pat), sp4.count(pat)
        res[f"count_{pat}"] = [cb, cp]
        assert cb == cp, f"{pat} count differs: base {cb} vs p4 {cp}"
    return res


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def run_case(case, do_e3):
    tag, fname, B, S, S_kv, N_q, N_kv, D, causal, dtype = case
    print(f"\n===== {tag} ({fname}, B{B} S{S} Skv{S_kv} Nq{N_q} Nkv{N_kv} D{D} "
          f"causal={causal} {dtype}) =====", flush=True)
    result = {"tag": tag}

    q, k, v = _inputs(tag, B, S, S_kv, N_q, N_kv, D, dtype)
    mask = _mask_for(causal, B, S, S_kv, S, S_kv, q.device)

    t0 = time.time()
    kb, use_mask_b = _build(case, in_bsnd=False)
    t1 = time.time()
    kp4, use_mask_p4 = _build(case, in_bsnd=True)
    t2 = time.time()
    assert use_mask_b == use_mask_p4
    print(f"[E1] compiled base {t1-t0:.1f}s / p4 {t2-t1:.1f}s", flush=True)

    # ---- E1 codegen audit ----
    res = audit_codegen(tag, kb, kp4, expect_qsrc=N_q * D, expect_kvsrc=N_kv * D)
    result["e1_codegen"] = res
    print(f"[E1] copy_gm_to_l1 count base={res['gm2l1_count_base']} p4={res['gm2l1_count_p4']} "
          f"(equal={res['count_equal']}); realSrcN base={res['realSrcN_base']} "
          f"p4={res['realSrcN_p4']} (expect Q={N_q*D}, KV={N_kv*D})", flush=True)

    # ---- E2 bit-identity smoke ----
    # BASE arm = the exact production chain: transpose_qkv -> BHSD -> main
    res_t = G.transpose_qkv(q, k, v, S_out=S, S_kv_out=S_kv)
    assert res_t is not None, "transpose_qkv returned None on an aligned probe shape"
    q_bhsd, k_bhsd, v_bhsd = res_t
    out_b = kb(q_bhsd, k_bhsd, v_bhsd, mask)
    torch.npu.synchronize()
    # P4 arm = main reads the caller's BSND tensors directly
    out_p4 = kp4(q, k, v, mask)
    torch.npu.synchronize()

    bit = bool(torch.equal(_rawbit(out_b.cpu()), _rawbit(out_p4.cpu())))
    viol_b, nan_b, nanref_b, inf_b, md_b = _golden(q, k, v, causal, out_b, dtype)
    viol_p, nan_p, nanref_p, inf_p, md_p = _golden(q, k, v, causal, out_p4, dtype)
    special = tag.startswith("boundary_sparse")
    fullrange = tag.startswith("fullrange")
    if special:
        seq, why = _special_struct_equal(out_p4.cpu().float(), out_b.cpu().float())
        gok = (nan_p == nan_b == nanref_p == nanref_b) and viol_p == viol_b
        print(f"[E2] special-struct equal={seq} ({why}); golden nan_arm={nan_p} nan_base={nan_b} "
              f"nan_ref={nanref_p} viol_arm={viol_p} viol_base={viol_b}", flush=True)
    elif fullrange:
        gok = (viol_p == viol_b and nan_p == nan_b and inf_p == inf_b)
        print(f"[E2] fullrange golden-invariance viol_arm={viol_p} viol_base={viol_b} "
              f"nan_arm={nan_p} nan_base={nan_b} (golden SATURATED nan_ref={nanref_p})", flush=True)
    else:
        gok = (viol_p == 0 and viol_b == 0 and nan_p == 0 and inf_p == 0)
        print(f"[E2] golden viol_arm={viol_p} viol_base={viol_b} nan={nan_p} "
              f"max_diff={md_p:.3e}", flush=True)
    result["e2"] = {"bit_identical": bit, "golden_ok": gok,
                    "viol_base": viol_b, "viol_p4": viol_p, "out_shape": tuple(out_p4.shape)}
    print(f"[E2] bit_identical={bit} golden_ok={gok} out_shape={tuple(out_p4.shape)}", flush=True)

    # ---- E3 directional timing (probe-grade) ----
    if do_e3:
        n = 30

        def chain_base():
            r = G.transpose_qkv(q, k, v, S_out=S, S_kv_out=S_kv)
            o = kb(r[0], r[1], r[2], mask)
            return o

        def chain_p4():
            return kp4(q, k, v, mask)

        def main_only_base():
            return kb(q_bhsd, k_bhsd, v_bhsd, mask)

        def main_only_p4():
            return kp4(q, k, v, mask)

        def timeit(fn, reps=n):
            for _ in range(5):
                fn()
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            torch.npu.synchronize()
            return (time.perf_counter() - t0) / reps * 1e6

        # interleaved A/B/A/B (same-session discipline, wall-clock probe grade)
        e2e_b, e2e_p, m_b, m_p = [], [], [], []
        for _ in range(3):
            e2e_b.append(timeit(chain_base))
            e2e_p.append(timeit(chain_p4))
            m_b.append(timeit(main_only_base))
            m_p.append(timeit(main_only_p4))
        import statistics

        eb, ep = statistics.median(e2e_b), statistics.median(e2e_p)
        mb, mp = statistics.median(m_b), statistics.median(m_p)
        result["e3_wall_us"] = {
            "e2e_base": round(eb, 2), "e2e_p4": round(ep, 2),
            "e2e_delta_pct": round((eb - ep) / eb * 100, 2),
            "main_base": round(mb, 2), "main_p4": round(mp, 2),
            "main_delta_us": round(mp - mb, 2),
            "main_delta_pct": round((mp - mb) / mb * 100, 2),
        }
        print(f"[E3] wall(probe-grade, host-loop): e2e base {eb:.2f} -> p4 {ep:.2f} us "
              f"({(eb-ep)/eb*100:+.2f}%); main base {mb:.2f} -> p4 {mp:.2f} us "
              f"(delta {mp-mb:+.2f} = {(mp-mb)/mb*100:+.2f}%)", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip-e3", action="store_true")
    args = ap.parse_args()

    md5 = __import__("hashlib").md5(open(os.path.join(GQA_DIR, "gqa.py"), "rb").read()).hexdigest()
    assert md5 == "606b07ff474ec42b8244f0831327d75b", f"gqa.py drifted: {md5}"

    cases = list(CASES)
    # special-value extensions of the two kv1 shapes (iter5/8/9 precedent set)
    cases.append(("boundary_sparse_c1", "_gqa_kernel_prefill_g2stack_flat", 2, 128, 128, 32, 8, 128, True, torch.float16))
    cases.append(("fullrange_r2", "_gqa_kernel_prefill_g2stack_flat", 2, 128, 128, 48, 4, 128, False, torch.float16))
    if args.only:
        cases = [c for c in cases if args.only in c[0]]

    results = []
    nfail = 0
    for case in cases:
        r = run_case(case, do_e3=not args.skip_e3 and not case[0].startswith(("boundary", "fullrange")))
        ok = r["e2"]["bit_identical"] and r["e2"]["golden_ok"] and r["e1_codegen"]["count_equal"]
        r["verdict"] = "PASS" if ok else "FAIL"
        if not ok:
            nfail += 1
        results.append(r)
        print(f"[{case[0]}] verdict={r['verdict']}", flush=True)

    out = os.path.join(HERE, "probe_iter10_P4_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"\n===== PROBE SUMMARY: {len(results)-nfail}/{len(results)} PASS -> {out}")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
