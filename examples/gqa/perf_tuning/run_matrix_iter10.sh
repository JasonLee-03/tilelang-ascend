#!/bin/bash
# Iter10 Gate-2 A/B collection matrix — P4 (transpose_qkv elimination, flat
# kv1/kv2 aligned domain reads BSND strided GM directly).
# ONE session, SEQUENTIAL (task #5: never two Fixed-Core-20 expert kernels
# concurrently), resumable (existing dirs skip).  Round-robin with per-round
# arm rotation (iter5/7/8/9 pattern) so both arms of a pair see the same drift.
#
# Mode B (all-kernel chain, --launch-count=3): e2e = SUM TaskDur over every
# instance.  BASE covered chain = kernel_kernel:0 (transpose_qkv) +
# main_kernel:0; P4 covered chain = main_kernel:0 ONLY (tqkv ABSENCE is the
# elimination evidence).  Guards: r1 = tqkv + main + tout; c19/c8 = tqkv +
# main (wave-2); r5 = main only.  waitid telemetry rides on PipeUtilization
# (the mandatory pacing discriminating gate needs no separate collection).
#
# Samples (preregistered, candidate_iter10_P4.json): primary r2/r4 +
# secondary r6/r3/c7 = 3+3; guard r1 = 3+3 (micro, iter9 precedent);
# guards r5/c19/c8 = 2+2 (iter9 guard cadence).
#
# Env contract: repo root, unset PYTHONPATH, ACL_OP_INIT_MODE=1, sqlite3 lib
# on LD_LIBRARY_PATH before msprof export (missing it => 0.00 artifacts).
# Usage: run_matrix_iter10.sh [SEG]   SEG in {S1,S2,S3,S4,ALL}
set -u
cd /mnt/workspace/cann/20260910_gqa_V02/tilelang-ascend
unset PYTHONPATH
export ACL_OP_INIT_MODE=1
export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:${LD_LIBRARY_PATH:-}
OUT=examples/gqa/perf_tuning
SEG="${1:-ALL}"
LOG=/tmp/opencode/iter10_matrix_${SEG}.log
AB=examples/gqa/perf_tuning/prof_gqa_ab_iter10.py
METRICS=PipeUtilization,Memory,L2Cache
mkdir -p /tmp/opencode
touch "$LOG"
echo "=== SEGMENT $SEG START $(date +%T) ===" >> "$LOG"

run_b() { # shape arm n
  local d="$OUT/msprof_op_iter10_${SEG}_$1_$2_s$3"
  if [ -d "$d" ]; then echo "SKIP ${SEG}_$1_$2_s$3 (exists)" >> "$LOG"; return; fi
  echo "=== START ${SEG}_$1_$2_s$3 $(date +%T) ===" >> "$LOG"
  timeout 1200 msprof op --launch-count=3 --aic-metrics=$METRICS \
      --output="$d" python3 $AB --arm "$2" --shape "$1" >> "$LOG" 2>&1
  echo "=== END ${SEG}_$1_$2_s$3 rc=$? $(date +%T) ===" >> "$LOG"
}

pair3() { # shape — 3+3 with per-round arm rotation
  for n in 1 2 3; do
    if [ $((n % 2)) -eq 1 ]; then run_b "$1" base "$n"; run_b "$1" p4 "$n";
    else run_b "$1" p4 "$n"; run_b "$1" base "$n"; fi
  done
}
pair2() { # shape — 2+2 with rotation
  for n in 1 2; do
    if [ $((n % 2)) -eq 1 ]; then run_b "$1" base "$n"; run_b "$1" p4 "$n";
    else run_b "$1" p4 "$n"; run_b "$1" base "$n"; fi
  done
}

case "$SEG" in
  S1) for s in r2 r4; do pair3 "$s"; done ;;                 # primaries
  S2) for s in r6 r3 c7; do pair3 "$s"; done ;;              # secondaries
  S3) pair3 r1; pair2 r5; pair2 c19 ;;                       # light guards
  S4) pair2 c8 ;;                                            # heavy guard (g2stack ns=8)
  ALL) for s in r2 r4 r6 r3 c7; do pair3 "$s"; done
       pair3 r1; pair2 r5; pair2 c19; pair2 c8 ;;
  *) echo "unknown SEG $SEG"; exit 2 ;;
esac
echo "=== SEGMENT $SEG DONE $(date +%T) ===" >> "$LOG"
