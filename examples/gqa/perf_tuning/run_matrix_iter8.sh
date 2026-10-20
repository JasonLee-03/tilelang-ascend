#!/bin/bash
# Iter8 Gate-2 A/B collection matrix — P2 (BSND direct write, tout eliminated).
# ONE session, sequential, resumable (existing dirs skip).  Round-robin with
# per-round arm rotation (iter5/iter7 pattern) so both arms of a pair see the
# same drift profile.
#
# Mode B (all-kernel chain, --launch-count=3): gives in ONE collection
#   * e2e = SUM of TaskDur over every kernel instance in the chain
#     (transpose_qkv + main + transpose_out for prefill; main only for decode)
#   * transpose_out attribution (presence/absence + TaskDur + Block Dim)
#   * main-kernel TaskDur (the direct-write cost guardrail)
#   * wait-id telemetry for the main instance (PipeUtilization carries the
#     aic_scalar_wait_id{1,2,5} / aiv_scalar_wait_id{0,3,4} columns in this
#     mode — verified against msprof_op_iter7_P1V2_r2_base_s1) => the
#     mandatory pacing discriminating gate needs no separate collection.
#
# Env contract: repo root, unset PYTHONPATH, ACL_OP_INIT_MODE=1, sqlite3 lib
# on LD_LIBRARY_PATH before msprof export (missing it => 0.00 artifacts).
# Usage: run_matrix_iter8.sh [P2]
set -u
cd /mnt/workspace/cann/20260910_gqa_V02/tilelang-ascend
unset PYTHONPATH
export ACL_OP_INIT_MODE=1
export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:${LD_LIBRARY_PATH:-}
OUT=examples/gqa/perf_tuning
SEG="${1:-P2}"
LOG=/tmp/opencode/iter8_matrix_${SEG}.log
AB=examples/gqa/perf_tuning/prof_gqa_ab_iter8.py
METRICS=PipeUtilization,Memory,L2Cache
touch "$LOG"
echo "=== SEGMENT $SEG START $(date +%T) ===" >> "$LOG"

run_b() { # shape arm n
  local d="$OUT/msprof_op_iter8_${SEG}_$1_$2_s$3"
  if [ -d "$d" ]; then echo "SKIP ${SEG}_$1_$2_s$3 (exists)" >> "$LOG"; return; fi
  echo "=== START ${SEG}_$1_$2_s$3 $(date +%T) ===" >> "$LOG"
  timeout 900 msprof op --launch-count=3 --aic-metrics=$METRICS \
      --output="$d" python3 $AB --arm "$2" --shape "$1" >> "$LOG" 2>&1
  echo "=== END ${SEG}_$1_$2_s$3 rc=$? $(date +%T) ===" >> "$LOG"
}

for n in 1 2 3; do
  for s in r1 r2 r3 r4 r5 r6 c7; do
    if [ $((n % 2)) -eq 1 ]; then
      run_b "$s" base "$n"
      run_b "$s" p2 "$n"
    else
      run_b "$s" p2 "$n"
      run_b "$s" base "$n"
    fi
  done
done

echo "=== SEGMENT $SEG DONE $(date +%T) ===" >> "$LOG"
npu-smi info >> "$LOG" 2>&1
ps aux | grep -c "[m]sprof" >> "$LOG" 2>&1
