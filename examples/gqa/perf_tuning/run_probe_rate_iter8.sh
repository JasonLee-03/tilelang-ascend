#!/bin/bash
# Iter8 P2 probe-rate matrix: MTE3 cost of the FIN box stream, contiguous
# (bhsd, shipped form) vs strided (bsnd4d, P2 form).  Isolated probe kernel
# (probe_iter8_P2_bsnd.py --exp rate): identical V-side work per box in both
# arms, GM read hoisted out, task stream replayed `rep` times => the TaskDur
# delta IS the strided-addressing cost that P2 adds to the main kernel's
# V-side FIN beat.  Input to the Phase-A marginal-benefit model.
# Env contract: repo root, unset PYTHONPATH, ACL_OP_INIT_MODE=1, sqlite3 lib.
# Usage: run_probe_rate_iter8.sh
set -u
cd /mnt/workspace/cann/20260910_gqa_V02/tilelang-ascend
unset PYTHONPATH
export ACL_OP_INIT_MODE=1
export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:${LD_LIBRARY_PATH:-}
OUT=examples/gqa/perf_tuning
LOG=/tmp/opencode/iter8_probe_rate.log
PR=$OUT/probe_iter8_P2_bsnd.py
touch "$LOG"
echo "=== PROBE RATE START $(date +%T) ===" >> "$LOG"

run_rate() { # shape arm rep n
  local d="$OUT/msprof_op_iter8_PR_$1_$2_s$4"
  if [ -d "$d" ]; then echo "SKIP PR_$1_$2_s$4 (exists)" >> "$LOG"; return; fi
  echo "=== START PR_$1_$2_s$4 $(date +%T) ===" >> "$LOG"
  timeout 900 msprof op --kernel-name=main_kernel --aic-metrics=PipeUtilization,Memory \
      --output="$d" python3 $PR --exp rate --arm "$2" --shape "$1" --rep "$3" >> "$LOG" 2>&1
  echo "=== END PR_$1_$2_s$4 rc=$? $(date +%T) ===" >> "$LOG"
}

for n in 1 2; do
  for s in r2 r6 r3 r4; do
    REP=8
    if [ "$s" = "r3" ] || [ "$s" = "r4" ]; then REP=4; fi
    if [ $((n % 2)) -eq 1 ]; then
      run_rate "$s" bhsd "$REP" "$n"
      run_rate "$s" bsnd4d "$REP" "$n"
    else
      run_rate "$s" bsnd4d "$REP" "$n"
      run_rate "$s" bhsd "$REP" "$n"
    fi
  done
done

echo "=== PROBE RATE DONE $(date +%T) ===" >> "$LOG"
npu-smi info >> "$LOG" 2>&1
ps aux | grep -c "[m]sprof" >> "$LOG" 2>&1
