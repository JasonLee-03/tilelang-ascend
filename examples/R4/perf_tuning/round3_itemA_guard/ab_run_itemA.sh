#!/bin/bash
# ---------------------------------------------------------------------------------------------
# Round-3 item-A SAME-SESSION 3-arm A/B (task section 2 / adoption criteria 1+2).
#
# Unlike round 3's ab_run.sh, which compared the PRE-rev3 BASE (19e51e8d) against the rev3
# candidate, item A compares two guard values of the SAME rev3 code:
#
#   base1 = a764e117  _SPLIT_A_LAUNCH_US = 8.0    (shipped conservative placeholder)
#   cand  = c59a83bc  _SPLIT_A_LAUNCH_US = 1.540  (measured E1 floor)   <-- arm under test
#   base2 = a764e117  identical code to base1     (pure DRIFT CONTROL)
#
# base2 vs base1 is byte-identical code, so any nonzero delta there is session drift rather than
# effect. Round 2 measured 0.3%-1.7% of it, which is why a cross-session baseline is unusable and
# why the adoption threshold is judged per (task, case) against that band.
#
# Measurement口径 is unchanged from round 1/2/3: cann-bench run_evaluation.sh, msprof
# kernel_details.csv, warmup=3 repeat=5, median of per-step Duration sums. NO Python/torch timing
# is used for any performance number (hard constraint 5).
#
# prewarm_itemA.py runs INSIDE each arm, after the swap, so every JIT specialization that arm needs
# is compiled BEFORE any timing window opens (hard constraint 7: compile cost is disclosed
# separately, never mixed into a measurement).
#
# Usage: ab_run_itemA.sh <base1|cand|base2>
# ---------------------------------------------------------------------------------------------
set -uo pipefail

TL=/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend
R=$TL/.agents/skills/cann-bench
R4=$TL/examples/R4
STAGE=/tmp/opencode/r4_itemA
OUT=$R4/perf_tuning/round3_itemA_guard

PASS="${1:?usage: ab_run_itemA.sh <base1|cand|base2>}"
case "$PASS" in
  base1|base2) SRC=$STAGE/BASE_a764e117_r4_fused_sampling.py
               WANT=a764e117ce150d8677096c2e00bde3db3ebc3ec16aba63e4c613d8dffdfbd117 ;;
  cand)        SRC=$STAGE/CAND_c59a83bc_r4_fused_sampling.py
               WANT=c59a83bcfa5f43365ab921499e2d3a6f9ec54011d861149e50d1beb12079c534 ;;
  *) echo "bad pass '$PASS'"; exit 9 ;;
esac

# ---- swap the delivered kernel to the arm under test, then PROVE the swap by sha256 ----
cp -f "$SRC" "$R4/r4_fused_sampling.py"
GOT=$(sha256sum "$R4/r4_fused_sampling.py" | cut -d' ' -f1)
if [ "$GOT" != "$WANT" ]; then
  echo "[ab_itemA] FATAL sha256 mismatch: got $GOT want $WANT"; exit 8
fi
echo "[ab_itemA] pass=$PASS  r4_fused_sampling.py sha256=$GOT  (verified == $WANT)"

cd "$R" || exit 9
export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:${LD_LIBRARY_PATH:-}   # else elapsed_us reads 0
export ACL_OP_INIT_MODE=1
export PYTHONPATH=/tmp/opencode/r4_eval/shim${PYTHONPATH:+:$PYTHONPATH}
export R4_HTML_SHIM=1
unset TILELANG_CACHE_DIR

# ---- pre-warm THIS arm's specializations; compile time is reported, never timed as perf ----
( cd "$R4" && python "$OUT/prewarm_itemA.py" "$PASS" ) > "$OUT/prewarm_${PASS}.log" 2>&1
echo "[ab_itemA] pass=$PASS prewarm rc=$?  $(grep -E 'specializations=' "$OUT/prewarm_${PASS}.log" | tail -1)"

# Record the report set that exists BEFORE the run so the new ones can be identified exactly.
ls -1 "$R4/bench/reports" > "$OUT/_reports_before_${PASS}.txt"

RC_TOTAL=0
for OP in R4FusedSampling R4MaskSample; do
  echo "################ ${PASS} ${OP} ################"
  S=$(date +%s)
  ./scripts/run_evaluation.sh --bench-name cann \
    --task-dir bench_lab/tilelang_ascend_bench \
    --operator "$OP" \
    --source-dir "$R4/cann_bench_submission" \
    --device-id 0 --processes-per-card 1 --timeout-per-process 1800 \
    --reports-dir "$R4/bench/reports" -v \
    > "$OUT/eval_${PASS}_${OP}.log" 2>&1
  RC=$?
  echo "### ${PASS} ${OP} EXIT_CODE=$RC WALL=$(( $(date +%s) - S ))s"
  [ $RC -ne 0 ] && RC_TOTAL=$RC
done

ls -1 "$R4/bench/reports" > "$OUT/_reports_after_${PASS}.txt"
echo "[ab_itemA] new report files for pass=$PASS:"
comm -13 "$OUT/_reports_before_${PASS}.txt" "$OUT/_reports_after_${PASS}.txt" \
  | tee "$OUT/_reports_new_${PASS}.txt"
exit $RC_TOTAL
