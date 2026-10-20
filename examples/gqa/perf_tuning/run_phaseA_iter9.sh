#!/bin/bash
# Iter9 Phase-A BASE sweep — tout/e2e/main TaskDur census on the three wave-2
# target domains (+r1 re-anchor).  SERIAL ONLY (task-book #5: Fixed-Core-20
# expert kernels must never co-reside with another collection on this chip).
# Resumable: existing dirs skip.
#
# Env contract: repo root, unset PYTHONPATH, ACL_OP_INIT_MODE=1, sqlite3 lib
# on LD_LIBRARY_PATH before msprof export.
# Usage: run_phaseA_iter9.sh [s1|s2]
set -u
cd /mnt/workspace/cann/20260910_gqa_V02/tilelang-ascend
unset PYTHONPATH
export ACL_OP_INIT_MODE=1
export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:${LD_LIBRARY_PATH:-}
OUT=examples/gqa/perf_tuning
SAMP="${1:-s1}"
LOG=/tmp/opencode/iter9_phaseA_${SAMP}.log
AB=examples/gqa/perf_tuning/prof_gqa_ab_iter9.py
METRICS=PipeUtilization,Memory,L2Cache
mkdir -p /tmp/opencode
touch "$LOG"
echo "=== PHASE-A $SAMP START $(date +%T) ===" >> "$LOG"

run_b() { # shape n
  local d="$OUT/msprof_op_iter9_PA_$1_base_$2"
  if [ -d "$d" ]; then echo "SKIP PA_$1_base_$2 (exists)" >> "$LOG"; return; fi
  echo "=== START PA_$1_base_$2 $(date +%T) ===" >> "$LOG"
  timeout 900 msprof op --launch-count=3 --aic-metrics=$METRICS \
      --output="$d" python3 $AB --arm base --shape "$1" >> "$LOG" 2>&1
  echo "=== END PA_$1_base_$2 rc=$? $(date +%T) ===" >> "$LOG"
}

for s in r1 c5 c19 c9 c8 c10 c6; do
  run_b "$s" "$SAMP"
done
echo "=== PHASE-A $SAMP DONE $(date +%T) ===" >> "$LOG"
