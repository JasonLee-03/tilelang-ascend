#!/bin/bash
# Iter8 Gate-3 pre-integration evidence collection (gqa.py still SHIPPED):
#  (1) r1 diagnostic samples s4-s6 (both arms) — the session-health anchor
#      miss (BASE 20.73us vs the iter7 anchor 20.05 = +3.41%) is on the
#      OUT-OF-DOMAIN, launch-floor-dominated r1 chain (tout 0.5MB, ~5us);
#      iter7's own Gate-3 re-measure of the same integrated file gave tout
#      5.20us vs 4.86us in its A/B session, i.e. this component has a
#      documented cross-session band.  Extra samples characterise it; they are
#      reported SEPARATELY and are NOT merged into the decision numbers.
#  (2) Gate-3 PRE capture (final + main-kernel outputs + dispatch/plan census
#      on 16 shapes) for the post-integration bit-identity comparison.
# Env contract: repo root, unset PYTHONPATH, ACL_OP_INIT_MODE=1, sqlite3 lib.
set -u
cd /mnt/workspace/cann/20260910_gqa_V02/tilelang-ascend
unset PYTHONPATH
export ACL_OP_INIT_MODE=1
export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:${LD_LIBRARY_PATH:-}
OUT=examples/gqa/perf_tuning
LOG=/tmp/opencode/iter8_pre.log
AB=examples/gqa/perf_tuning/prof_gqa_ab_iter8.py
touch "$LOG"
echo "=== PRE STAGE START $(date +%T) md5=$(md5sum examples/gqa/gqa.py | cut -d' ' -f1) ===" >> "$LOG"

for n in 4 5 6; do
  for arm in base p2; do
    d="$OUT/msprof_op_iter8_P2diag_r1_${arm}_s$n"
    if [ -d "$d" ]; then echo "SKIP diag_r1_${arm}_s$n" >> "$LOG"; continue; fi
    echo "=== START diag_r1_${arm}_s$n $(date +%T) ===" >> "$LOG"
    timeout 900 msprof op --launch-count=3 --aic-metrics=PipeUtilization,Memory,L2Cache \
        --output="$d" python3 $AB --arm "$arm" --shape r1 >> "$LOG" 2>&1
    echo "=== END diag_r1_${arm}_s$n rc=$? $(date +%T) ===" >> "$LOG"
  done
done

echo "=== GATE3 PRE CAPTURE $(date +%T) ===" >> "$LOG"
timeout 2400 python3 $OUT/save_gate3_outputs_iter8.py pre >> "$LOG" 2>&1
echo "=== PRE CAPTURE rc=$? $(date +%T) ===" >> "$LOG"
npu-smi info >> "$LOG" 2>&1
ps aux | grep -c "[m]sprof" >> "$LOG" 2>&1
