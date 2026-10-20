#!/bin/bash
# Iter7 A/B collection matrix — P1 (B-mode e2e chain) + P5 (main kernel).
# Segments (ONE session, sequential; resumable — existing dirs skip):
#   P1: shapes r1..r6 x arms {base,p1} x 3 rounds, all-kernel chain
#       (--launch-count=3, metrics AUX) — e2e Σ (tqkv+main+tout TaskDur)
#       + transpose_out single-kernel + block-dim audit
#   P5: shapes r4,c7 x arms {base,p5} x 3 rounds, --kernel-name=main_kernel
#       --aic-metrics=PipeUtilization (TaskDur + wait-id telemetry)
# Round-robin with per-round arm rotation (iter5 pattern): both arms of a
# pair see the same drift profile.
# Env contract: repo root, unset PYTHONPATH, ACL_OP_INIT_MODE=1, sqlite3 lib
# on LD_LIBRARY_PATH (msprof export; missing it => 0.00 artifacts).
# Usage: run_matrix_iter7.sh P1|P5
set -u
cd /mnt/workspace/cann/20260910_gqa_V02/tilelang-ascend
unset PYTHONPATH
export ACL_OP_INIT_MODE=1
export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:${LD_LIBRARY_PATH:-}
OUT=examples/gqa/perf_tuning
SEG="${1:-P1}"
LOG=/tmp/opencode/iter7_matrix_${SEG}.log
AB=examples/gqa/perf_tuning/prof_gqa_ab_iter7.py
METRICS_AUX=PipeUtilization,Memory,L2Cache
touch "$LOG"
echo "=== SEGMENT $SEG START $(date +%T) ===" >> "$LOG"

run_p1() { # shape arm n [tag]
  local tag="${4:-P1}"
  local d="$OUT/msprof_op_iter7_${tag}_$1_$2_s$3"
  if [ -d "$d" ]; then echo "SKIP ${tag}_$1_$2_s$3 (exists)" >> "$LOG"; return; fi
  echo "=== START ${tag}_$1_$2_s$3 $(date +%T) ===" >> "$LOG"
  timeout 900 msprof op --launch-count=3 --aic-metrics=$METRICS_AUX \
      --output="$d" python3 $AB --arm "$2" --shape "$1" >> "$LOG" 2>&1
  echo "=== END ${tag}_$1_$2_s$3 rc=$? $(date +%T) ===" >> "$LOG"
}

run_p5() { # shape arm n
  local d="$OUT/msprof_op_iter7_P5_$1_$2_s$3"
  if [ -d "$d" ]; then echo "SKIP P5_$1_$2_s$3 (exists)" >> "$LOG"; return; fi
  echo "=== START P5_$1_$2_s$3 $(date +%T) ===" >> "$LOG"
  timeout 900 msprof op --kernel-name=main_kernel --aic-metrics=PipeUtilization \
      --output="$d" python3 $AB --arm "$2" --shape "$1" >> "$LOG" 2>&1
  echo "=== END P5_$1_$2_s$3 rc=$? $(date +%T) ===" >> "$LOG"
}

case "$SEG" in
  P1|P1V2)
    for n in 1 2 3; do
      for s in r1 r2 r3 r4 r5 r6; do
        if [ $((n % 2)) -eq 1 ]; then
          run_p1 "$s" base "$n" "$SEG"
          run_p1 "$s" p1 "$n" "$SEG"
        else
          run_p1 "$s" p1 "$n" "$SEG"
          run_p1 "$s" base "$n" "$SEG"
        fi
      done
    done
    ;;
  P5)
    for n in 1 2 3; do
      for s in r4 c7; do
        if [ $((n % 2)) -eq 1 ]; then
          run_p5 "$s" base "$n"
          run_p5 "$s" p5 "$n"
        else
          run_p5 "$s" p5 "$n"
          run_p5 "$s" base "$n"
        fi
      done
    done
    ;;
esac

echo "=== SEGMENT $SEG DONE $(date +%T) ===" >> "$LOG"
npu-smi info >> "$LOG" 2>&1
ps aux | grep -c "[m]sprof" >> "$LOG" 2>&1
