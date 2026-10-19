"""Round-3 JIT compile-cost disclosure for the 4 NEW rev3 split kernels (hard constraint 7).

Every performance number in this round comes from msprof device Duration, so JIT compile time is
structurally excluded from all of them. This probe exists solely to DISCLOSE that cost separately:
it compiles each distinct rev3 split specialization once against a FRESH (empty) TILELANG_CACHE_DIR
and records the host wall-clock, which is what a first-ever run on a new machine would pay.

The delivered kernel file is only imported, never modified (sha256 checked before/after).

Usage: probe_cold_split_compile.py <fresh_cache_dir> [out.json]
"""
import hashlib
import json
import os
import sys
import time

CACHE = os.path.abspath(sys.argv[1])
HERE = os.path.dirname(os.path.abspath(__file__))
R4 = os.path.abspath(os.path.join(HERE, "..", ".."))
os.environ["TILELANG_CACHE_DIR"] = CACHE          # must be set BEFORE tilelang is imported
os.makedirs(CACHE, exist_ok=True)
for _p in (R4, os.path.join(R4, "bench", "hap_thw_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r4_fused_sampling as M  # noqa: E402
from r4_thw_common import load_cases  # noqa: E402

out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "probe_cold_split_compile.json")
KERNEL = os.path.join(R4, "r4_fused_sampling.py")
sha0 = hashlib.sha256(open(KERNEL, "rb").read()).hexdigest()
CASES = {c["case_id"]: c for c in load_cases("r4_fused_sampling", verbose=False)}
SPLIT = [ci for ci in sorted(CASES)
         if M._r4_split_plan(CASES[ci]["B"], CASES[ci]["V"], CASES[ci]["K_MAX"],
                             CASES[ci]["BV"])[0]]

entries_before = len(os.listdir(CACHE)) if os.path.isdir(CACHE) else 0
print(f"[cold] kernel sha256 = {sha0}")
print(f"[cold] fresh TILELANG_CACHE_DIR = {CACHE} (entries before = {entries_before})")
print(f"[cold] split cases = {SPLIT}\n")

rows, seen = [], set()
for ci in SPLIT:
    c = CASES[ci]
    B, V, K, BV = c["B"], c["V"], c["K_MAX"], c["BV"]
    nmp = bool(c["need_min_p"])
    use_split, S, NBS, ST = M._r4_split_plan(B, V, K, BV)
    jobs = [("r4_topk_stats_splitA", "K1A", (B, V, K, BV, S, NBS, ST)),
            ("r4_topk_stats_splitB", "K1B", (B, V, K, BV, S, nmp)),
            ("r4_gumbel_sample_splitA", "K2gA", (B, V, BV, S, NBS)),
            ("r4_gumbel_sample_splitB", "K2gB", (B, S))]
    for fname, short, args in jobs:
        key = (fname, args)
        if key in seen:
            rows.append(dict(case=ci, kernel=short, fn=fname, args=[str(a) for a in args],
                             reused_specialization=True, compile_s=0.0))
            print(f"  case{ci:>2} {short:5s} args={args}  REUSED (same specialization as an "
                  f"earlier case) -> 0.00 s", flush=True)
            continue
        seen.add(key)
        t0 = time.time()
        getattr(M, fname)(*args)
        dt = round(time.time() - t0, 2)
        rows.append(dict(case=ci, kernel=short, fn=fname, args=[str(a) for a in args],
                         reused_specialization=False, compile_s=dt))
        print(f"  case{ci:>2} {short:5s} args={args}  COLD compile = {dt:6.2f} s", flush=True)

cold = [r for r in rows if not r["reused_specialization"]]
entries_after = len(os.listdir(CACHE)) if os.path.isdir(CACHE) else 0
sha1 = hashlib.sha256(open(KERNEL, "rb").read()).hexdigest()
print(f"\n[cold] distinct specializations compiled = {len(cold)}  "
      f"(of {len(rows)} kernel invocations; {len(rows) - len(cold)} reused an earlier one)")
print(f"[cold] total cold compile wall = {sum(r['compile_s'] for r in cold):.2f} s   "
      f"mean {sum(r['compile_s'] for r in cold) / max(1, len(cold)):.2f} s   "
      f"min {min(r['compile_s'] for r in cold):.2f} s   max {max(r['compile_s'] for r in cold):.2f} s")
print(f"[cold] per-kernel totals: " + "  ".join(
    f"{k}={sum(r['compile_s'] for r in cold if r['kernel'] == k):.1f}s"
    for k in ("K1A", "K1B", "K2gA", "K2gB")))
print(f"[cold] cache entries {entries_before} -> {entries_after} (NEW={entries_after - entries_before})")
print(f"[cold] kernel sha256 after = {sha1}  UNMODIFIED={sha0 == sha1}")

json.dump(dict(_meta=dict(
    purpose="separate disclosure of JIT compile cost for the 4 NEW rev3 split kernels; excluded "
            "from every performance number in round 3 (hard constraint 7)",
    measurement="host wall-clock of the tilelang.jit call against a FRESH TILELANG_CACHE_DIR",
    fresh_cache_dir=CACHE, kernel_sha256_before=sha0, kernel_sha256_after=sha1,
    kernel_file_unmodified=(sha0 == sha1), cube_core_num=M.detect_cube_core_num(),
    split_cases=SPLIT),
    n_distinct_specializations=len(cold), n_invocations=len(rows),
    total_cold_compile_s=round(sum(r["compile_s"] for r in cold), 2),
    mean_cold_compile_s=round(sum(r["compile_s"] for r in cold) / max(1, len(cold)), 2),
    per_kernel_total_s={k: round(sum(r["compile_s"] for r in cold if r["kernel"] == k), 2)
                        for k in ("K1A", "K1B", "K2gA", "K2gB")},
    cache_entries_before=entries_before, cache_entries_after=entries_after,
    rows=rows), open(out_path, "w"), indent=1)
print(f"[cold] wrote {out_path}")
