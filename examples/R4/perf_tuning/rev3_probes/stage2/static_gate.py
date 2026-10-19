"""rev3 STATIC gate (DESIGN.md section 11.4 step 1 + section 9.2.5 assertions 1/3/4).

No NPU device needed: get_kernel_source() only compiles to C++ source.

Checks, per new kernel x per shape specialization:
  1. compiles (no LOG(FATAL) / ValueError / InternalError)
  2. measured UB <= the generated region limit (parse ascend_ub.GetWithOffset<T>(n, off))
  3. every TopK<T>(dst, src, tmp, K, repeatTimes, actual_num) has padCount == 0,
     i.e. repeatTimes*32 - actual_num == 0   (defect A structurally unreachable)
  4. every copy_ub_to_ub<..., count> has count*4 % 32 == 0   (defect B structurally unreachable)
  5. every C++ `for` loop has a LITERAL integer bound (zero dynamic loop bounds)
  6. (once, source level) each new kernel's @tilelang.jit decorator carries
     pass_configs=R4_PASS_CONFIGS  -> tl.ascend_memory_planning=True is present (R-MEM)

Usage: static_gate.py [--shapes all|split|smoke] [--out json]
"""
import argparse
import ast
import json
import re
import sys
import time

sys.path.insert(0, "/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4")

from r4_fused_sampling import (  # noqa: E402
    R4_PASS_CONFIGS,
    VEC_NUM,
    _r4_split_plan,
    detect_cube_core_num,
    r4_gumbel_sample_splitA,
    r4_gumbel_sample_splitB,
    r4_topk_stats_splitA,
    r4_topk_stats_splitB,
)

_SZ = {"float": 4, "int": 4, "uint8_t": 1, "int32_t": 4, "float16_t": 2,
       "uint16_t": 2, "int16_t": 2, "uint32_t": 4, "int8_t": 1, "half": 2}
_PAT_UB = re.compile(r"ascend_ub\.GetWithOffset<\s*([A-Za-z0-9_:]+)\s*>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)")
_PAT_TOPK = re.compile(r"TopK<\w+>\([^;]*?,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")
_PAT_U2U = re.compile(r"copy_ub_to_ub<[^,]+,[^,]+,\s*(\d+)>")
_PAT_FOR = re.compile(r"for\s*\(\s*int32_t\s+(\w+)\s*=\s*([^;]+?)\s*;\s*\1\s*<\s*([^;]+?)\s*;")
_PAT_REGION = re.compile(r"InitBuffer\(ascend_ub,\s*(\d+)\)")
UB_LIMIT_FALLBACK = 196352

NEW_KERNELS = ("r4_topk_stats_splitA", "r4_topk_stats_splitB",
               "r4_gumbel_sample_splitA", "r4_gumbel_sample_splitB")
KERNEL_PATH = "/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4/r4_fused_sampling.py"

# (label, B, V, K_MAX, BV, force_S or None)
SHAPES = {
    # ---- L0 section 9.2.3 matrix (15 cases) ----
    "L0-A1": (16, 32768, 256, 4096, None),
    "L0-A2": (4, 152064, 256, 4096, None),
    "L0-A3": (8, 16384, 256, 4096, None),
    "L0-A4": (4, 16384, 128, 2048, None),
    "L0-B1": (4, 20737, 256, 4096, None),
    "L0-B2": (4, 20793, 256, 4096, None),
    "L0-B3": (4, 20991, 256, 4096, None),
    "L0-B4": (4, 20741, 256, 4096, None),
    "L0-B5": (16, 152377, 256, 4096, None),
    "L0-C1": (20, 32768, 256, 4096, None),
    "L0-C2": (22, 32768, 256, 4096, None),
    "L0-C3": (40, 32768, 256, 4096, None),
    "L0-E1": (2, 16384, 256, 4096, 1),      # white-box forced S=1 degenerate
    "L0-F1": (10, 20480, 256, 4096, None),   # theorem P3 shrink sentinel
    # ---- cann-bench split-path cases (section 13.4) ----
    "cb3": (8, 16384, 256, 4096, None),
    "cb4": (4, 16384, 256, 4096, None),
    "cb7": (4, 152064, 256, 4096, None),
    "cb8": (4, 16384, 128, 2048, None),
    "cb12": (16, 32768, 256, 4096, None),
    "cb17": (16, 152377, 256, 4096, None),
    # ---- geometrically splittable but guard-rejected: K1A must still COMPILE them
    #      (L0-E1-style white-box may construct them; DESIGN.md section 4.5 note) ----
    "cb5": (7, 4409, 256, 4096, None),
    "cb6": (8, 10240, 256, 4096, None),
    "cb11": (8, 8192, 256, 4096, None),
    "cb13": (6, 2304, 128, 2048, None),
}
SMOKE = ("L0-A3", "L0-A4", "L0-B2", "L0-E1", "L0-F1")
SPLIT_ONLY = ("L0-A1", "L0-A2", "L0-A3", "L0-A4", "L0-B1", "L0-B2", "L0-B3", "L0-B4",
              "L0-B5", "L0-C1", "L0-E1", "L0-F1", "cb3", "cb4", "cb7", "cb8", "cb12", "cb17")


def ub_usage(src):
    """(total_used_bytes, n_buffers, region_limit) -- same method as round2_bv_probe/ub_probe.py."""
    det = []
    for m in _PAT_UB.finditer(src):
        dt, n, off = m.group(1), int(m.group(2)), int(m.group(3))
        esz = _SZ.get(dt.split("::")[-1])
        if esz is None:
            raise SystemExit(f"unknown dtype {dt}")
        det.append((off, n * esz))
    if not det:
        raise SystemExit("no GetWithOffset -- parse failure")
    reg = _PAT_REGION.search(src)
    return max(o + s for o, s in det), len(det), (int(reg.group(1)) if reg else UB_LIMIT_FALLBACK)


def geom(B, V, K_MAX, BV, force_S=None):
    """Derived split geometry, exactly what the host wrapper passes to the kernels."""
    use_split, S, NBS, S_TAIL = _r4_split_plan(B, V, K_MAX, BV)
    if force_S is not None:  # white-box: override S and re-derive NBS / S_TAIL
        S = int(force_S)
        n_blk_full = V // BV
        TAIL = V - n_blk_full * BV
        n_blk_total = n_blk_full + (1 if TAIL > 0 else 0)
        NBS = -(-n_blk_total // S)
        S_TAIL = n_blk_full // NBS if TAIL > 0 else -1
    return use_split, S, NBS, S_TAIL


def check_source(label, kname, src):
    """Return a dict of the 5 per-source checks (all derived from generated C++)."""
    tot, nbuf, region = ub_usage(src)
    topk = [(int(a), int(b), int(c)) for a, b, c in _PAT_TOPK.findall(src)]
    bad_pad = [(K, r, a) for (K, r, a) in topk if r * 32 - a != 0]
    u2u = sorted({int(c) for c in _PAT_U2U.findall(src)})
    bad_u2u = [c for c in u2u if (c * 4) % 32 != 0]
    fors = [(v, lo, hi) for v, lo, hi in _PAT_FOR.findall(src)]
    dyn = [(v, lo, hi) for (v, lo, hi) in fors if not re.fullmatch(r"-?\d+", hi.strip())]
    return dict(kernel=kname, label=label, ub=tot, ub_region=region, ub_pct=round(100.0 * tot / region, 2),
                nbuf=nbuf, ub_ok=tot <= region,
                topk_calls=sorted(set(topk)), pad_all_zero=not bad_pad, bad_pad=bad_pad,
                u2u_counts=u2u, u2u_all_32B=not bad_u2u, bad_u2u=bad_u2u,
                n_for=len(fors), for_bounds=sorted({int(hi.strip()) for _, _, hi in fors if hi.strip().lstrip('-').isdigit()}),
                dyn_bounds=dyn, static_ok=not dyn)


def assert_pass_configs():
    """Section 9.2.5 static assertion 1 (source/AST level): each new kernel's decorator must
    pass pass_configs=R4_PASS_CONFIGS, and that dict must contain tl.ascend_memory_planning=True."""
    src = open(KERNEL_PATH, encoding="utf-8").read()
    tree = ast.parse(src)
    res, ok_all = {}, True
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in NEW_KERNELS:
            continue
        found = None
        for d in node.decorator_list:
            if not isinstance(d, ast.Call):
                continue
            fn = d.func
            fname = getattr(fn, "attr", getattr(fn, "id", None))
            if fname != "jit":
                continue
            for kw in d.keywords:
                if kw.arg == "pass_configs":
                    found = ast.unparse(kw.value)
            found_outidx = [ast.unparse(kw.value) for kw in d.keywords if kw.arg == "out_idx"]
        res[node.name] = found
        good = found == "R4_PASS_CONFIGS"
        ok_all &= good
        print(f"  [assert-1] {node.name:26s} pass_configs={found!r} "
              f"{'== R4_PASS_CONFIGS OK' if good else '*** MISMATCH ***'}"
              + (f"  out_idx={found_outidx}" if node.name.endswith(("splitB",)) else ""))
    memplan = R4_PASS_CONFIGS.get("tl.ascend_memory_planning")
    print(f"  [assert-1] R4_PASS_CONFIGS['tl.ascend_memory_planning'] = {memplan} "
          f"(R-MEM safety net: ascend_memory_planning.cc:598 sibling path is check_overflow=false)")
    ok_all &= memplan is True
    missing = [k for k in NEW_KERNELS if k not in res]
    if missing:
        print(f"  [assert-1] *** kernels not found in source: {missing}")
        ok_all = False
    return ok_all, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="all", choices=["all", "split", "smoke"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    keys = {"all": tuple(SHAPES), "split": SPLIT_ONLY, "smoke": SMOKE}[args.shapes]

    cube = detect_cube_core_num()
    print("=" * 100)
    print(f"rev3 STATIC GATE   cube_core_num={cube} (probed, not hardcoded)   shapes={len(keys)}")
    print("=" * 100)
    print("[section 9.2.5 assertion 1] pass_configs of the 4 new kernels:")
    pc_ok, pc = assert_pass_configs()

    rows, fails = [], []
    for label in keys:
        B, V, K, BV, force_S = SHAPES[label]
        use_split, S, NBS, S_TAIL = geom(B, V, K, BV, force_S)
        bn = -(-B // VEC_NUM)
        nbf, TAIL = V // BV, V - (V // BV) * BV
        nbt = nbf + (1 if TAIL > 0 else 0)
        per = [max(0, min(nbt, (s + 1) * NBS) - s * NBS) for s in range(S)]
        print(f"\n--- {label}: B={B} V={V} K={K} BV={BV} | b_num={bn} nbf={nbf} TAIL={TAIL} nbt={nbt} "
              f"-> use_split={use_split} S={S} NBS={NBS} S_TAIL={S_TAIL} grid={bn*S} "
              f"waves={-(-bn*S//cube)} per={per}"
              + ("  [FORCED S]" if force_S is not None else ""))
        if min(per) < 1:
            fails.append((label, "EMPTY SPLIT (theorem P3 violated)", per))
            print(f"    *** EMPTY SPLIT: per={per}")
        jobs = [
            ("K1A", r4_topk_stats_splitA, (B, V, K, BV, S, NBS, S_TAIL)),
            ("K1B", r4_topk_stats_splitB, (B, V, K, BV, S, False)),
            ("K1B", r4_topk_stats_splitB, (B, V, K, BV, S, True)),
            ("K2gA", r4_gumbel_sample_splitA, (B, V, BV, S, NBS)),
            ("K2gB", r4_gumbel_sample_splitB, (B, S)),
        ]
        for kn, fn, a in jobs:
            t0 = time.time()
            try:
                src = fn(*a).get_kernel_source()
            except BaseException as e:  # noqa: BLE001  (LOG(FATAL) may raise SystemExit)
                dt = time.time() - t0
                fails.append((label, kn, f"COMPILE-FAIL {type(e).__name__}: {str(e)[:200]}"))
                print(f"    {kn:5s} COMPILE-FAIL ({dt:.1f}s) {type(e).__name__}: {str(e)[:200]}")
                continue
            dt = time.time() - t0
            r = check_source(label, kn, src)
            r["args"] = [str(x) for x in a]
            r["compile_s"] = round(dt, 1)
            rows.append(r)
            ok = r["ub_ok"] and r["pad_all_zero"] and r["u2u_all_32B"] and r["static_ok"]
            if not ok:
                fails.append((label, kn, {k: r[k] for k in ("ub", "ub_region", "bad_pad", "bad_u2u", "dyn_bounds")}))
            print(f"    {kn:5s} UB={r['ub']:7d}/{r['ub_region']} ({r['ub_pct']:5.2f}%) bufs={r['nbuf']:3d} "
                  f"| topk(K,rep,act)={r['topk_calls']} padCount_ALL0={r['pad_all_zero']} "
                  f"| ub2ub={r['u2u_counts']} all32B={r['u2u_all_32B']} "
                  f"| for_bounds={r['for_bounds']} dynamic={r['dyn_bounds']} "
                  f"| {dt:5.1f}s {'OK' if ok else '*** FAIL ***'}")

    n_ok = len(rows)
    worst = {}
    for r in rows:
        k = r["kernel"]
        if k not in worst or r["ub"] > worst[k]["ub"]:
            worst[k] = r
    print("\n" + "=" * 100)
    print(f"COMPILED SOURCES : {n_ok} OK / {n_ok + len([f for f in fails if len(f) == 3 and 'COMPILE' in str(f[2])])} attempted")
    print("WORST MEASURED UB PER NEW KERNEL (section 4.5 cross-check):")
    for k in sorted(worst):
        r = worst[k]
        print(f"  {k:6s} UB={r['ub']:7d}/{r['ub_region']} ({r['ub_pct']:5.2f}%) headroom={r['ub_region']-r['ub']:7d} "
              f"worst at {r['label']} args={r['args']}")
    print(f"\n[section 9.2.5 assert-1] pass_configs all == R4_PASS_CONFIGS : {pc_ok}")
    print(f"[section 9.2.5 assert-3] measured UB <= region for all     : {all(r['ub_ok'] for r in rows)}")
    print(f"[section 9.2.5 assert-4] padCount==0 & ub2ub 32B for all     : "
          f"{all(r['pad_all_zero'] and r['u2u_all_32B'] for r in rows)}")
    print(f"[section 11.4 step1 (5)] zero dynamic loop bounds            : {all(r['static_ok'] for r in rows)}")
    print(f"FAILURES: {len(fails)}")
    for f in fails:
        print("   ", f)
    verdict = pc_ok and not fails and bool(rows)
    print(f"\nSTATIC GATE VERDICT: {'PASS' if verdict else 'FAIL'}")
    if args.out:
        json.dump(dict(pass_configs=pc, rows=rows, fails=[list(map(str, f)) for f in fails],
                       worst={k: v for k, v in worst.items()}, verdict=verdict),
                  open(args.out, "w"), indent=1)
        print(f"json -> {args.out}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
