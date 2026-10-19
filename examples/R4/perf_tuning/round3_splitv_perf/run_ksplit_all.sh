#!/bin/bash
# Round-3: run the fine-grained per-kernel splitter over all 6 msprof kernel_details.csv produced
# by ab_all.sh, and record the CSV<->report manifest so the attribution is auditable.
#
# BASE-arm CSVs are analysed with R4_FORCE_CURRENT=1: the delivered file in place is the CANDIDATE
# (restored by ab_all.sh), whose dispatch would wrongly expect 3/4 launches for the 6 split cases.
# The BASE kernel has no split path at all, so every one of its cases is a 2-launch CURRENT step.
set -uo pipefail
R4=/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4
O=$R4/perf_tuning/round3_splitv_perf
PD=$R4/bench/reports/prof_data/_batched
source "$O/env_r4.sh"

declare -A CSV=(
 [base1_fused]="$PD/r4_fused_sampling/2616692-1791702187548982857/124c18542182426382010d673204c8fc_2616692_20261011150307649_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv"
 [base1_mask]="$PD/r4_mask_sample/2619141-1791702307535917603/124c18542182426382010d673204c8fc_2619141_20261011150507621_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv"
 [cand_fused]="$PD/r4_fused_sampling/2620899-1791702429714686482/124c18542182426382010d673204c8fc_2620899_20261011150709816_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv"
 [cand_mask]="$PD/r4_mask_sample/2623417-1791702546737849487/124c18542182426382010d673204c8fc_2623417_20261011150906824_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv"
 [base2_fused]="$PD/r4_fused_sampling/2625986-1791702673462674172/124c18542182426382010d673204c8fc_2625986_20261011151113575_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv"
 [base2_mask]="$PD/r4_mask_sample/2627859-1791702787872407622/124c18542182426382010d673204c8fc_2627859_20261011151307957_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv"
)

: > "$O/csv_manifest.txt"
RC=0
for K in base1_fused base1_mask cand_fused cand_mask base2_fused base2_mask; do
  F="${CSV[$K]}"
  if [ ! -f "$F" ]; then echo "MISSING CSV for $K: $F"; RC=1; continue; fi
  TASK="${K##*_}"
  ARM="${K%%_*}"
  NSPLIT=$(grep -c . "$F")
  echo "$K  arm=$ARM task=$TASK  csv_lines=$NSPLIT" | tee -a "$O/csv_manifest.txt"
  echo "    $F" >> "$O/csv_manifest.txt"
  if [ "$ARM" = "base1" ] || [ "$ARM" = "base2" ]; then export R4_FORCE_CURRENT=1; else unset R4_FORCE_CURRENT; fi
  python "$O/ksplit_fine.py" "$F" "$ARM" "$TASK" "$O/ksplit_${K}.json" \
     > "$O/ksplit_${K}.txt" 2>&1
  R=$?
  echo "    ksplit rc=$R  -> $O/ksplit_${K}.txt"
  grep -E "^GATE|^steps_attributed" "$O/ksplit_${K}.txt" | sed 's/^/    /'
  [ $R -ne 0 ] && RC=$R
done
unset R4_FORCE_CURRENT
echo "ALL_KSPLIT_RC=$RC"
exit $RC
