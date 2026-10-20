#!/bin/bash
# Iter9 Gate-2 A/B collection matrix — P2 wave-2 (prefill_expert + g2stack
# BSND direct write, transpose_out eliminated on those domains).
# ONE session, SEQUENTIAL (task #5: never two Fixed-Core-20 expert kernels
# concurrently), resumable (existing dirs skip).  Round-robin with per-round
# arm rotation (iter5/7/8 pattern) so both arms of a pair see the same drift.
#
# Mode B (all-kernel chain, --launch-count=3): e2e = SUM TaskDur over every
# instance (tqkv + main + tout); transpose_out attribution (presence + TaskDur
# + Block Dim); main TaskDur (cost guardrail); wait-id telemetry on main
# (PipeUtilization carries aic/aiv_scalar_wait_id* in this mode — the
# mandatory pacing discriminating gate needs no separate collection).
#
# Samples (preregistered): covered c19/c8/c9/c10 = 3+3; guards r1/r2/r5 = 3+3;
# non-aligned guards s100e/s100g = 2+2.
#
# Env contract: repo root, unset PYTHONPATH, ACL_OP_INIT_MODE=1, sqlite3 lib
# on LD_LIBRARY_PATH before msprof export (missing it => 0.00 artifacts).
# Usage: run_matrix_iter9.sh [SEG]
set -u
cd /mnt/workspace/cann/20260910_gqa_V02/tilelang-ascend
unset PYTHONPATH
export ACL_OP_INIT_MODE=1
export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:${LD_LIBRARY_PATH:-}
OUT=examples/gqa/perf_tuning
SEG="${1:-W2}"
LOG=/tmp/opencode/iter9_matrix_${SEG}.log
AB=examples/gqa/perf_tuning/prof_gqa_ab_iter9.py
METRICS=PipeUtilization,Memory,L2Cache
mkdir -p /tmp/opencode
touch "$LOG"
echo "=== SEGMENT $SEG START $(date +%T) ===" >> "$LOG"

run_b() { # shape arm n
  local d="$OUT/msprof_op_iter9_${SEG}_$1_$2_s$3"
  if [ -d "$d" ]; then echo "SKIP ${SEG}_$1_$2_s$3 (exists)" >> "$LOG"; return; fi
  echo "=== START ${SEG}_$1_$2_s$3 $(date +%T) ===" >> "$LOG"
  timeout 900 msprof op --launch-count=3 --aic-metrics=$METRICS \
      --output="$d" python3 $AB --arm "$2" --shape "$1" >> "$LOG" 2>&1
  echo "=== END ${SEG}_$1_$2_s$3 rc=$? $(date +%T) ===" >> "$LOG"
}

# covered (3+3) + guards r1/r2/r5 (3+3)
MAIN="c19 c8 c9 c10 r1 r2 r5"
# non-aligned guards (2+2)
GUARD2="s100e s100g"

for n in 1 2 3; do
  for s in $MAIN; do
    if [ $((n % 2)) -eq 1 ]; then
      run_b "$s" base "$n"
      run_b "$s" w2 "$n"
    else
      run_b "$s" w2 "$n"
      run_b "$s" base "$n"
    fi
  done
  if [ "$n" -le 2 ]; then
    for s in $GUARD2; do
      if [ $((n % 2)) -eq 1 ]; then
        run_b "$s" base "$n"; run_b "$s" w2 "$n"
      else
        run_b "$s" w2 "$n"; run_b "$s" base "$n"
      fi
    done
  fi
done
echo "=== SEGMENT $SEG DONE $(date +%T) ===" >> "$LOG"
