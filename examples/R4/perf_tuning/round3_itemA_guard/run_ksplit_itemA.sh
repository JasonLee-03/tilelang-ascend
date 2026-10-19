#!/bin/bash
# ---------------------------------------------------------------------------------------------
# Item-A: run the fine-grained per-kernel splitter over the 6 msprof kernel_details.csv produced by
# ab_all_itemA.sh, and record a CSV<->arm manifest so the attribution is auditable.
#
# CSV IDENTIFICATION: the arms run strictly sequentially (base1 -> cand -> base2), so for each
# operator the three prof dirs created inside this A/B window, ordered by their embedded
# <pid>-<epoch_ns> timestamp, ARE the three arms in order. The selection is then VERIFIED rather
# than trusted: ksplit_fine_itemA.py derives the expected launch count / Block Num sequence from
# _r4_split_plan and asserts it against the CSV, so a mis-assigned arm cannot pass silently.
#
# ARM-AWARE GUARD: unlike round 3, BOTH item-A arms are rev3 code with the split API; they differ
# only in _SPLIT_A_LAUNCH_US. ab_all_itemA.sh restores the CANDIDATE, so base-arm CSVs are analysed
# with R4_FORCE_A_LAUNCH=8.0 and the cand CSV with 1.540 (see the note inside ksplit_fine_itemA.py).
# ---------------------------------------------------------------------------------------------
set -uo pipefail
R4=/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4
O=$R4/perf_tuning/round3_itemA_guard
PD=$R4/bench/reports/prof_data/_batched
# A/B window start (ab_all_itemA.sh launched 2026-10-12T19:37 local); only dirs newer than this
# marker file belong to item A.
MARKER=$O/.ab_window_start
[ -f "$MARKER" ] || touch -d "2026-10-12 19:36:00" "$MARKER"

export PYTHONPATH=/home/developer/Ascend/cann-9.2.0/python/site-packages:/home/developer/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe:
export ACL_OP_INIT_MODE=1
export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:${LD_LIBRARY_PATH:-}
cd "$R4" || exit 9

: > "$O/csv_manifest_itemA.txt"
RC=0
for TASKDIR in r4_fused_sampling r4_mask_sample; do
  TASK=$([ "$TASKDIR" = "r4_mask_sample" ] && echo mask || echo fused)
  mapfile -t DIRS < <(find "$PD/$TASKDIR" -mindepth 1 -maxdepth 1 -type d -newer "$MARKER" \
                      -printf '%T@ %p\n' 2>/dev/null | sort -n | cut -d' ' -f2-)
  echo "[ksplit] task=$TASK  dirs inside the item-A window: ${#DIRS[@]}"
  if [ "${#DIRS[@]}" -lt 3 ]; then
    echo "[ksplit] *** expected 3 arms, found ${#DIRS[@]} -- ABORT task=$TASK ***"; RC=1; continue
  fi
  # the LAST three in chronological order are base1, cand, base2 (an earlier stray dir, if any,
  # would be from a pre-flight run; take the tail so the arm order stays right)
  ARMS=(base1 cand base2)
  for i in 0 1 2; do
    ARM=${ARMS[$i]}
    D=${DIRS[$(( ${#DIRS[@]} - 3 + i ))]}
    CSV=$(find "$D" -name kernel_details.csv | head -1)
    if [ -z "$CSV" ] || [ ! -f "$CSV" ]; then
      echo "[ksplit] *** no kernel_details.csv under $D ***"; RC=1; continue
    fi
    A_LAUNCH=$([ "$ARM" = "cand" ] && echo 1.540 || echo 8.0)
    KEY="${ARM}_${TASK}"
    echo "$KEY  arm=$ARM task=$TASK  a_launch=$A_LAUNCH  csv_lines=$(grep -c . "$CSV")" \
      | tee -a "$O/csv_manifest_itemA.txt"
    echo "    $CSV" >> "$O/csv_manifest_itemA.txt"
    R4_FORCE_A_LAUNCH=$A_LAUNCH python "$O/ksplit_fine_itemA.py" "$CSV" "$ARM" "$TASK" \
      "$O/ksplit_${KEY}.json" > "$O/ksplit_${KEY}.txt" 2>&1
    R=$?
    echo "    ksplit rc=$R (a_launch pinned to $A_LAUNCH) -> ksplit_${KEY}.txt"
    grep -E "^GATE|^steps_attributed|R4_FORCE_A_LAUNCH" "$O/ksplit_${KEY}.txt" | sed 's/^/    /'
    [ $R -ne 0 ] && RC=$R
  done
done
echo "ALL_KSPLIT_ITEMA_RC=$RC"
exit $RC
