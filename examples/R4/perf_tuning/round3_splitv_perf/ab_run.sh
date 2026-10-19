#!/bin/bash
# ---------------------------------------------------------------------------------------------
# Round-3 SAME-SESSION A/B performance measurement (task section 1(1)).
#
# Measures the pre-rev3 BASE kernel (19e51e8d) and the rev3 CANDIDATE (a764e117) in ONE session,
# back to back, with a third BASE pass bracketing the candidate so that session-level drift can be
# separated from real effect (round 2 #4 measured 0.3%-1.7% drift even for a byte-identical
# control arm, so a cross-session baseline would be unusable).
#
#   pass order:  base1  ->  cand  ->  base2
#   base2 vs base1 == SAME CODE  => pure drift control
#   cand  vs base1 == primary comparison
#
# Measurement口径 is identical to round 1/2: cann-bench run_evaluation.sh, msprof
# kernel_details.csv, warmup=3 repeat=5, median of per-step sums. NO Python/torch timing is used
# anywhere in this round (hard constraint 2).
#
# TILELANG_CACHE_DIR is deliberately NOT set: prewarm.py already warmed the default
# ~/.tilelang/cache for BOTH arms (base: 2/72 cold, cand: 21/90 cold, all frozen-kernel entries
# whose cache keys are shared), so neither arm pays JIT compile inside a timing window.
#
# Usage: ab_run.sh <base1|cand|base2>
# ---------------------------------------------------------------------------------------------
set -uo pipefail

R=/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/.agents/skills/cann-bench
TL=/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend
R4=$TL/examples/R4
STAGE=/tmp/opencode/r4_round3
OUT=$R4/perf_tuning/round3_splitv_perf

PASS="${1:?usage: ab_run.sh <base1|cand|base2>}"
case "$PASS" in
  base1|base2) SRC=$STAGE/BASE_19e51e8d_r4_fused_sampling.py
               WANT=19e51e8d8ee126b920cf48c915d863a9be3a349e2c68cf728d35d482b48bcc57 ;;
  cand)        SRC=$STAGE/CAND_a764e117_r4_fused_sampling.py
               WANT=a764e117ce150d8677096c2e00bde3db3ebc3ec16aba63e4c613d8dffdfbd117 ;;
  *) echo "bad pass '$PASS'"; exit 9 ;;
esac

# ---- swap the delivered kernel to the arm under test, then PROVE the swap by sha256 ----
cp -f "$SRC" "$R4/r4_fused_sampling.py"
GOT=$(sha256sum "$R4/r4_fused_sampling.py" | cut -d' ' -f1)
if [ "$GOT" != "$WANT" ]; then
  echo "[ab_run] FATAL sha256 mismatch: got $GOT want $WANT"; exit 8
fi
echo "[ab_run] pass=$PASS  r4_fused_sampling.py sha256=$GOT  (verified == $WANT)"

cd "$R" || exit 9
export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:${LD_LIBRARY_PATH:-}
export ACL_OP_INIT_MODE=1
export PYTHONPATH=/tmp/opencode/r4_eval/shim${PYTHONPATH:+:$PYTHONPATH}
export R4_HTML_SHIM=1
unset TILELANG_CACHE_DIR

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
echo "[ab_run] new report files for pass=$PASS:"
comm -13 "$OUT/_reports_before_${PASS}.txt" "$OUT/_reports_after_${PASS}.txt" \
  | tee "$OUT/_reports_new_${PASS}.txt"
exit $RC_TOTAL
