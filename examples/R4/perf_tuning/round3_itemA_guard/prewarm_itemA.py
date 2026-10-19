"""Round-3 JIT pre-warm + compile-cost probe.

Instantiates every TileLang specialization the A/B eval will need, once per (case, task) over the
full cann-bench 18-case table, so the JIT cache is warm BEFORE any timing window opens (compile
time must never be mixed into a measurement -- hard constraint 7).

Also reports, per specialization, how long the compile took, which is the single-sourced
disclosure for "new JIT specializations + compile cost".

COMPILE-ONLY: no tensors are allocated and NO device timing is taken here. Every number this
script prints is HOST wall-clock for compilation and is reported separately from all performance
figures in this round.

Usage: prewarm.py <label>
"""
import hashlib
import json
import os
import sys
import time

R4 = "/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4"
sys.path.insert(0, R4)
sys.path.insert(0, os.path.join(R4, "bench", "hap_thw_model"))

label = sys.argv[1] if len(sys.argv) > 1 else "unlabeled"

import r4_fused_sampling as M  # noqa: E402
from r4_thw_common import load_cases  # noqa: E402

KERNEL_SHA = hashlib.sha256(open(os.path.join(R4, "r4_fused_sampling.py"), "rb").read()).hexdigest()
CACHE_DIR = os.environ.get("TILELANG_CACHE_DIR", os.path.expanduser("~/.tilelang/cache"))

CASES = {c["case_id"]: c for c in load_cases("r4_fused_sampling", verbose=False)}

# The rev3 split-V API does not exist in the pre-rev3 BASE arm (19e51e8d). Detect it instead of
# assuming, so this one script warms BOTH arms; BASE therefore always plans the CURRENT path.
HAS_SPLIT = hasattr(M, "_r4_split_plan")
CUBE = M.detect_cube_core_num() if HAS_SPLIT else None
print(f"[{label}] kernel_sha256={KERNEL_SHA}  HAS_SPLIT_API={HAS_SPLIT}  cube_core_num={CUBE}",
      flush=True)

before = set(os.listdir(CACHE_DIR)) if os.path.isdir(CACHE_DIR) else set()
t_start = time.time()
rows = []

for task in ("fused", "mask"):
    for ci in sorted(CASES):
        c = CASES[ci]
        B, V, K, BV = c["B"], c["V"], c["K_MAX"], c["BV"]
        nmp = bool(c.get("need_min_p", False))
        use_split, S, NBS, ST = M._r4_split_plan(B, V, K, BV) if HAS_SPLIT else (False, 1, 1, -1)
        if use_split:
            specs = [("K1A", lambda: M.r4_topk_stats_splitA(B, V, K, BV, S, NBS, ST)),
                     ("K1B", lambda: M.r4_topk_stats_splitB(B, V, K, BV, S, nmp))]
            if task == "fused":
                specs += [("K2gA", lambda: M.r4_gumbel_sample_splitA(B, V, BV, S, NBS)),
                          ("K2gB", lambda: M.r4_gumbel_sample_splitB(B, S))]
            else:
                specs += [("K2m", lambda: M.r4_mask_sample(B, V, BV))]
        else:
            specs = [("K1", lambda: M.r4_topk_stats(B, V, K, BV, nmp))]
            specs += [("K2g", lambda: M.r4_gumbel_sample(B, V, BV))] if task == "fused" \
                else [("K2m", lambda: M.r4_mask_sample(B, V, BV))]
        per = {}
        for name, fn in specs:
            ts = time.time()
            fn()
            per[name] = round(time.time() - ts, 2)
        rows.append(dict(task=task, case=ci, B=B, V=V, K_MAX=K, BV=BV, need_min_p=nmp,
                         path="SPLIT" if use_split else "CURRENT", S=S, NBS=NBS, S_TAIL=ST,
                         compile_s=per, total_compile_s=round(sum(per.values()), 2)))
        print(f"[{label}] {task:5s} case{ci:>2} {'SPLIT' if use_split else 'CURR'} S={S:>2} "
              f"NBS={NBS:>2} ST={ST:>2} compile={per} total={sum(per.values()):.2f}s", flush=True)

after = set(os.listdir(CACHE_DIR)) if os.path.isdir(CACHE_DIR) else set()
elapsed = round(time.time() - t_start, 2)
n_cold = sum(1 for r in rows for v in r["compile_s"].values() if v > 1.0)
print(f"\n[{label}] kernel_sha256={KERNEL_SHA}")
print(f"[{label}] cache_dir={CACHE_DIR}  entries {len(before)} -> {len(after)}  "
      f"NEW={len(after - before)}")
print(f"[{label}] specializations={sum(len(r['compile_s']) for r in rows)}  cold(>1s)={n_cold}  "
      f"total_wall={elapsed}s")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"prewarm_{label}.json")
json.dump(dict(label=label, kernel_sha256=KERNEL_SHA, cache_dir=CACHE_DIR, cube_core_num=CUBE,
               entries_before=len(before), entries_after=len(after),
               new_cache_entries=len(after - before), cold_compiles=n_cold,
               total_wall_s=elapsed, rows=rows), open(out, "w"), indent=1)
print(f"[{label}] wrote {out}")
