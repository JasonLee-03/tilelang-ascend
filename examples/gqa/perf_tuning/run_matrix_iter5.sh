#!/bin/bash
# Iter5 Gate 2 — two-arm same-session INTERLEAVED A/B matrix.
# Arms: base (shipped g2stack ns=2) / c2 (flat kv2 variant); shapes r4 (primary)
# + c7 (secondary).  Round-robin with per-round arm rotation so both arms see
# the same drift profile.  Resumable: existing output dirs are skipped.
# Env contract: repo root, unset PYTHONPATH, ACL_OP_INIT_MODE=1,
# sqlite3 lib on LD_LIBRARY_PATH (msprof export — missing it => 0.00 artifact).
# Usage: run_matrix_iter5.sh N_ROUNDS   (default 3)
set -u
cd /mnt/workspace/cann/20260910_gqa_V02/tilelang-ascend
unset PYTHONPATH
export ACL_OP_INIT_MODE=1
export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:${LD_LIBRARY_PATH:-}
OUT=examples/gqa/perf_tuning
LOG=/tmp/opencode/iter5_matrix.log
touch "$LOG"
N=${1:-3}
AB=examples/gqa/perf_tuning/prof_gqa_ab_iter5.py

run() { # shape arm round
  local shape="$1" arm="$2" n="$3"
  local d="$OUT/msprof_op_iter5_${shape}_${arm}_s${n}"
  if [ -d "$d" ]; then echo "SKIP ${shape}_${arm}_s${n} (exists)" >> "$LOG"; return; fi
  echo "=== START ${shape}_${arm}_s${n} $(date +%T) ===" >> "$LOG"
  timeout 900 msprof op --kernel-name=main_kernel --aic-metrics=PipeUtilization \
      --output="$d" python3 $AB --arm "$arm" --shape "$shape" >> "$LOG" 2>&1
  echo "=== END ${shape}_${arm}_s${n} rc=$? $(date +%T) ===" >> "$LOG"
}

for n in $(seq 1 "$N"); do
  for shape in r4 c7; do
    if [ $((n % 2)) -eq 1 ]; then
      run "$shape" base "$n"
      run "$shape" c2 "$n"
    else
      run "$shape" c2 "$n"
      run "$shape" base "$n"
    fi
  done
done

echo "=== MATRIX DONE $(date +%T) ===" >> "$LOG"
npu-smi info >> "$LOG" 2>&1
