#!/bin/bash
# Iter6 roofline analysis collection matrix (analysis_only — nothing outside
# perf_tuning/ is written; gqa.py / test_gqa.py untouched).
#
# Segments (run in ONE session, sequentially; resumable — existing dirs skip):
#   A: main-kernel six-condition matrix, 3 samples each, FULL roofline metric set
#   B: all-kernel chain capture (aux separation), 2 samples each
#   C: hardware-peak microbenchmarks, 2 samples each
#
# Env contract: repo root, unset PYTHONPATH, ACL_OP_INIT_MODE=1,
# sqlite3 lib on LD_LIBRARY_PATH (msprof export; missing it => 0.00 artifacts).
# Usage: run_matrix_iter6.sh A|B|C
set -u
cd /mnt/workspace/cann/20260910_gqa_V02/tilelang-ascend
unset PYTHONPATH
export ACL_OP_INIT_MODE=1
export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:$LD_LIBRARY_PATH
OUT=examples/gqa/perf_tuning
SEG="${1:-A}"
LOG=/tmp/opencode/iter6_matrix_${SEG}.log
PROF=examples/gqa/perf_tuning/prof_gqa_iter6.py
MB=examples/gqa/perf_tuning/microbench_peak_iter6.py
METRICS_FULL=PipeUtilization,Memory,L2Cache,ArithmeticUtilization,MemoryUB,MemoryL0,ResourceConflictRatio
METRICS_AUX=PipeUtilization,Memory,L2Cache

echo "=== SEGMENT $SEG START $(date +%T) ===" >> "$LOG"

run_main() { # shape n
  local d="$OUT/msprof_op_iter6_A_$1_s$2"
  if [ -d "$d" ]; then echo "SKIP A_$1_s$2 (exists)" >> "$LOG"; return; fi
  echo "=== START A_$1_s$2 $(date +%T) ===" >> "$LOG"
  timeout 900 msprof op --kernel-name=main_kernel --aic-metrics=$METRICS_FULL \
      --output="$d" python3 $PROF --shape "$1" >> "$LOG" 2>&1
  echo "=== END A_$1_s$2 rc=$? $(date +%T) ===" >> "$LOG"
}

run_allk() { # shape n count
  local d="$OUT/msprof_op_iter6_B_$1_allk_s$2"
  if [ -d "$d" ]; then echo "SKIP B_$1_s$2 (exists)" >> "$LOG"; return; fi
  echo "=== START B_$1_s$2 $(date +%T) ===" >> "$LOG"
  timeout 900 msprof op --launch-count="$3" --aic-metrics=$METRICS_AUX \
      --output="$d" python3 $PROF --shape "$1" >> "$LOG" 2>&1
  echo "=== END B_$1_s$2 rc=$? $(date +%T) ===" >> "$LOG"
}

run_mb() { # mode n [kernel-name]
  local d="$OUT/msprof_op_iter6_C_$1_s$2"
  if [ -d "$d" ]; then echo "SKIP C_$1_s$2 (exists)" >> "$LOG"; return; fi
  echo "=== START C_$1_s$2 $(date +%T) ===" >> "$LOG"
  if [ -n "${3:-}" ]; then
    timeout 900 msprof op --kernel-name="$3" --aic-metrics=$METRICS_AUX,ArithmeticUtilization \
        --output="$d" python3 $MB --mode "$1" >> "$LOG" 2>&1
  else
    timeout 900 msprof op --launch-count=1 --aic-metrics=$METRICS_AUX,ArithmeticUtilization \
        --output="$d" python3 $MB --mode "$1" >> "$LOG" 2>&1
  fi
  echo "=== END C_$1_s$2 rc=$? $(date +%T) ===" >> "$LOG"
}

case "$SEG" in
  A)
    for n in 1 2 3; do
      for s in r1 r2 r3 r4 r5 r6; do
        run_main "$s" "$n"
      done
    done
    ;;
  B)
    # prefill chains: transpose_qkv + main + transpose_out = 3 kernels
    for n in 1 2; do
      for s in r1 r2 r3 r4 r6; do
        run_allk "$s" "$n" 3
      done
      # decode chain: main only (Skv=2048 aligned, noncausal -> no pad/mask kernel);
      # count=3 verifies nothing hidden launches (msprof ends at app exit)
      run_allk r5 "$n" 3
    done
    ;;
  C)
    for n in 1 2; do
      run_mb gemm "$n" main_kernel
      run_mb copy "$n" main_kernel
      run_mb copyl2 "$n" main_kernel
      run_mb vec "$n" main_kernel
      run_mb torch_gemm "$n" ""
    done
    ;;
esac

echo "=== SEGMENT $SEG DONE $(date +%T) ===" >> "$LOG"
npu-smi info >> "$LOG" 2>&1
ps aux | grep -c "[m]sprof" >> "$LOG" 2>&1
