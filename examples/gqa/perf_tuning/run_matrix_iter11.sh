#!/bin/bash
# Iter11 Gate-2 A/B collection matrix — P4 wave-2 (transpose_qkv elimination
# on the g2stack + prefill_expert dual-aligned domains; main kernels read BSND
# strided GM directly).
# ONE session, SEQUENTIAL (task #2: never two Fixed-Core-20 expert kernels
# concurrently), resumable (existing dirs skip).  Round-robin with per-round
# arm rotation (iter5/7/8/9/10 pattern) so both arms of a pair see the same drift.
#
# Mode B (all-kernel chain, --launch-count=3): e2e = SUM TaskDur over every
# instance.  BASE covered chain = kernel_kernel:0 (transpose_qkv) +
# main_kernel:0; W2 covered chain = main_kernel:0 ONLY (tqkv ABSENCE is the
# elimination evidence).  Guards: r2 = main only BOTH arms (wave-1 shipped
# in_bsnd); r1 = tqkv + main + tout; r5 = main only.  waitid telemetry rides
# on PipeUtilization (the mandatory pacing discriminating gate needs no
# separate collection).
#
# Samples (preregistered, candidate_iter11_P4w2.json): covered c19/c8/c9/c10
# = 3+3; guards r2/r1 = 3+3 (small-shape variance); r5 = 2+2 (iter9/10 cadence).
#
# Env contract: repo root, unset PYTHONPATH, ACL_OP_INIT_MODE=1, sqlite3 lib
# on LD_LIBRARY_PATH before msprof export (missing it => 0.00 artifacts).
# Usage: run_matrix_iter11.sh [SEG]   SEG in {S1,S2,S3,S4,ALL}
set -u
cd /mnt/workspace/cann/20260910_gqa_V02/tilelang-ascend
unset PYTHONPATH
export ACL_OP_INIT_MODE=1
export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:${LD_LIBRARY_PATH:-}
OUT=examples/gqa/perf_tuning
SEG="${1:-ALL}"
LOG=/tmp/opencode/iter11_matrix_${SEG}.log
AB=examples/gqa/perf_tuning/prof_gqa_ab_iter11.py
METRICS=PipeUtilization,Memory,L2Cache
mkdir -p /tmp/opencode
touch "$LOG"
echo "=== SEGMENT $SEG START $(date +%T) ===" >> "$LOG"

run_b() { # shape arm n
  local d="$OUT/msprof_op_iter11_${SEG}_$1_$2_s$3"
  if [ -d "$d" ]; then echo "SKIP ${SEG}_$1_$2_s$3 (exists)" >> "$LOG"; return; fi
  echo "=== START ${SEG}_$1_$2_s$3 $(date +%T) ===" >> "$LOG"
  timeout 1800 msprof op --launch-count=3 --aic-metrics=$METRICS \
      --output="$d" python3 $AB --arm "$2" --shape "$1" >> "$LOG" 2>&1
  echo "=== END ${SEG}_$1_$2_s$3 rc=$? $(date +%T) ===" >> "$LOG"
}

pair3() { # shape — 3+3 with per-round arm rotation
  for n in 1 2 3; do
    if [ $((n % 2)) -eq 1 ]; then run_b "$1" base "$n"; run_b "$1" w2 "$n";
    else run_b "$1" w2 "$n"; run_b "$1" base "$n"; fi
  done
}
pair2() { # shape — 2+2 with rotation
  for n in 1 2; do
    if [ $((n % 2)) -eq 1 ]; then run_b "$1" base "$n"; run_b "$1" w2 "$n";
    else run_b "$1" w2 "$n"; run_b "$1" base "$n"; fi
  done
}

case "$SEG" in
  S1) pair3 c19; pair3 c8 ;;                    # primaries (prexpert + g2stack)
  S2) pair3 c9; pair3 c10 ;;                    # secondaries (g2stack)
  S3) pair3 r2; pair3 r1 ;;                     # small guards (wave-1 + per-head)
  S4) pair2 r5 ;;                               # decode guard
  ALL) pair3 c19; pair3 c8; pair3 c9; pair3 c10; pair3 r2; pair3 r1; pair2 r5 ;;
  *) echo "unknown SEG $SEG"; exit 2 ;;
esac
echo "=== SEGMENT $SEG DONE $(date +%T) ===" >> "$LOG"
