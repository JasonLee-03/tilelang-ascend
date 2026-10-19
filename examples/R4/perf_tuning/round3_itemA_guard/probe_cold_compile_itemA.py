"""Item-A JIT compile-cost disclosure for the specializations the guard relaxation newly requires.

Hard constraint 7: new JIT compile time must NOT be counted inside any performance number, and must
be disclosed separately. Every performance figure in this item comes from msprof device Duration, so
compile time is structurally excluded from all of them; this probe exists only to state what a
FIRST-EVER run on a cold machine would pay for the newly routed geometries.

It compiles exactly the specs enumerated by preflight_itemA.py P3 (`P3_total_new`) against a FRESH,
empty TILELANG_CACHE_DIR, and records host wall-clock per spec. For contrast it also re-times the
same specs against the WARM default cache, which is what the gates and the A/B actually saw.

The delivered kernel file is only imported, never modified (sha256 checked before and after).

Usage: probe_cold_compile_itemA.py <fresh_cache_dir> [out.json]
"""
import ast
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

out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "cold_compile_itemA.json")
KERNEL = os.path.join(R4, "r4_fused_sampling.py")
sha0 = hashlib.sha256(open(KERNEL, "rb").read()).hexdigest()

# The authoritative list of newly required specs is preflight P3; re-parse it rather than
# hardcoding, so this probe cannot silently drift from the analysis it is disclosing.
pf = json.load(open(os.path.join(HERE, "preflight.json")))
NEW = [ast.literal_eval(s) for s in pf["P3_total_new"]]
FN = {"K1A": M.r4_topk_stats_splitA, "K1B": M.r4_topk_stats_splitB,
      "K2gA": M.r4_gumbel_sample_splitA, "K2gB": M.r4_gumbel_sample_splitB}

print(f"[cold] kernel sha256              = {sha0}")
print(f"[cold] _SPLIT_A_LAUNCH_US         = {M._SPLIT_A_LAUNCH_US}")
print(f"[cold] fresh TILELANG_CACHE_DIR   = {CACHE}")
print(f"[cold] newly required specs       = {len(NEW)} (from preflight.json P3_total_new)\n")

before = len(os.listdir(CACHE)) if os.path.isdir(CACHE) else 0
t0 = time.time()
rows = []
for name, args in NEW:
    ts = time.time()
    src = FN[name](*args).get_kernel_source()
    dt = round(time.time() - ts, 2)
    rows.append(dict(kernel=name, args=list(args), cold_compile_s=dt, src_bytes=len(src)))
    print(f"  [COLD] {name:5s} {str(args):<44} {dt:>7.2f} s   ({len(src)} B of generated source)",
          flush=True)
cold_total = round(time.time() - t0, 2)
after = len(os.listdir(CACHE)) if os.path.isdir(CACHE) else 0

# WARM re-timing: same specs, cache now populated by the pass above.
t1 = time.time()
warm = []
for name, args in NEW:
    ts = time.time()
    FN[name](*args).get_kernel_source()
    warm.append(round(time.time() - ts, 3))
warm_total = round(time.time() - t1, 2)

sha1 = hashlib.sha256(open(KERNEL, "rb").read()).hexdigest()
print(f"\n[cold] COLD total for {len(NEW)} new specs = {cold_total} s "
      f"({round(cold_total / max(1, len(NEW)), 2)} s/spec average)")
print(f"[cold] WARM total for the same specs      = {warm_total} s "
      f"({round(warm_total / max(1, len(NEW)), 3)} s/spec average)")
print(f"[cold] cache entries {before} -> {after} (NEW={after - before})")
print(f"[cold] kernel sha256 after = {sha1}  UNMODIFIED={sha0 == sha1}")
print("[cold] NOTE: none of this is inside any timing window. The A/B arms were pre-warmed "
      "(prewarm_itemA.py) before their profiler windows opened, and the scored口径 is the sum of "
      "msprof device Durations, which excludes host compile time by construction.")

json.dump(dict(_meta=dict(kernel_sha256_before=sha0, kernel_sha256_after=sha1,
                          kernel_unmodified=(sha0 == sha1),
                          a_launch_us=M._SPLIT_A_LAUNCH_US, fresh_cache_dir=CACHE,
                          source_of_spec_list="preflight.json P3_total_new",
                          note="host wall-clock for compilation only; excluded from every "
                               "performance number by construction"),
               n_new_specs=len(NEW), cold_total_s=cold_total,
               cold_avg_s_per_spec=round(cold_total / max(1, len(NEW)), 3),
               warm_total_s=warm_total, cache_entries_before=before, cache_entries_after=after,
               new_cache_entries=after - before, specs=rows, warm_retiming_s=warm),
          open(out_path, "w"), indent=1)
print(f"[cold] wrote {out_path}")
