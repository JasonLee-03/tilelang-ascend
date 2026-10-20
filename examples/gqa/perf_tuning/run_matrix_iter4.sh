#!/bin/bash
# Iter4 Gate 2 — three-way same-session INTERLEAVED A/B matrix.
# Arms: base (shipped flat) / ns4 (C1) / g2first (C3); shapes r3/r2/r6.
# Round-robin with per-round arm rotation so every arm sees the same
# drift profile. Resumable: existing output dirs are skipped.
# Env contract: repo root, unset PYTHONPATH, ACL_OP_INIT_MODE=1,
# sqlite3 lib on LD_LIBRARY_PATH (msprof export).
# Usage: run_matrix_iter4.sh N_ROUNDS   (default 3)
set -u
cd /mnt/workspace/cann/20260910_gqa_V02/tilelang-ascend
unset PYTHONPATH
export ACL_OP_INIT_MODE=1
export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:$LD_LIBRARY_PATH
OUT=examples/gqa/perf_tuning
LOG=/tmp/opencode/iter4_matrix.log
: > "$LOG"
N=${1:-3}
AB=examples/gqa/perf_tuning/prof_gqa_ab_iter4.py

run() { # shape arm round
  local shape="$1" arm="$2" n="$3"
  local d="$OUT/msprof_op_iter4_${shape}_${arm}_s${n}"
  if [ -d "$d" ]; then echo "SKIP ${shape}_${arm}_s${n} (exists)" >> "$LOG"; return; fi
  echo "=== START ${shape}_${arm}_s${n} $(date +%T) ===" >> "$LOG"
  timeout 900 msprof op --kernel-name=main_kernel --aic-metrics=PipeUtilization \
      --output="$d" python3 $AB --arm "$arm" --shape "$shape" >> "$LOG" 2>&1
  echo "=== END ${shape}_${arm}_s${n} rc=$? $(date +%T) ===" >> "$LOG"
}

ARMS=(base ns4 g2first)
for n in $(seq 1 "$N"); do
  for shape in r3 r2 r6; do
    # rotate arm order per round: round n starts at index (n-1) % 3
    for k in 0 1 2; do
      idx=$(( (n - 1 + k) % 3 ))
      run "$shape" "${ARMS[$idx]}" "$n"
    done
  done
done

echo "=== MATRIX DONE $(date +%T) ===" >> "$LOG"
npu-smi info >> "$LOG" 2>&1
