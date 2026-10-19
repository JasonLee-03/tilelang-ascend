#!/bin/bash
# ---------------------------------------------------------------------------------------------
# Item-A criterion-2 arbitration: REPEAT the harness口径 measurement on the mask operator,
# alternating arms, to get a DISTRIBUTION instead of one sample per arm.
#
# WHY: the 3-arm A/B flagged exactly one pair outside the 3% band -- case 13 mask,
# 14.620 -> 15.240 us (+4.24%, absolute +0.62 us) -- on a shape whose dispatch is provably
# identical in both arms (`_r4_split_plan` = (False, 2, 1, 1) at a_launch 8.0 AND 1.540; it is
# blocked structurally by n_blk_full = 2304//2048 = 1 < 2, so the cost guard is never reached).
# ksplit_fine_itemA.py independently confirmed the same launch sequence and Block Num in all three
# arms. probe_null_case13.py then measured a 5.18% pure-noise envelope on 5 such identical-dispatch
# shapes under interleaving, and case 13 mask itself read -0.42% there -- but that probe uses a
# slightly different setup (no CannBenchCacheClean), so it cannot arbitrate a criterion that is
# stated in the HARNESS口径.
#
# This script closes that gap on the harness口径 itself: it re-runs R4MaskSample alternately for the
# base and candidate kernels, so case 13 mask (and every other non-split mask case) gets several
# independent samples per arm. One sample per arm cannot bound a noise distribution; several can.
#
# The delivered kernel is ALWAYS restored to the candidate at the end and the restore is proven.
#
# Usage: ab_repeat_mask_itemA.sh <n_pairs>
# ---------------------------------------------------------------------------------------------
set -uo pipefail
TL=/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend
R=$TL/.agents/skills/cann-bench
R4=$TL/examples/R4
STAGE=/tmp/opencode/r4_itemA
OUT=$R4/perf_tuning/round3_itemA_guard
CAND_SHA=c59a83bcfa5f43365ab921499e2d3a6f9ec54011d861149e50d1beb12079c534
N="${1:-2}"

run_arm() {  # <label> <src> <want_sha>
  local LABEL=$1 SRC=$2 WANT=$3
  cp -f "$SRC" "$R4/r4_fused_sampling.py"
  local GOT; GOT=$(sha256sum "$R4/r4_fused_sampling.py" | cut -d' ' -f1)
  if [ "$GOT" != "$WANT" ]; then echo "[repeat] FATAL sha mismatch $GOT != $WANT"; return 8; fi
  echo "[repeat] arm=$LABEL sha256=$GOT (verified)"
  ls -1 "$R4/bench/reports" > "$OUT/_reports_before_${LABEL}.txt"
  ( cd "$R" && ./scripts/run_evaluation.sh --bench-name cann \
      --task-dir bench_lab/tilelang_ascend_bench --operator R4MaskSample \
      --source-dir "$R4/cann_bench_submission" \
      --device-id 0 --processes-per-card 1 --timeout-per-process 1800 \
      --reports-dir "$R4/bench/reports" -v ) > "$OUT/eval_${LABEL}_R4MaskSample.log" 2>&1
  local RC=$?
  ls -1 "$R4/bench/reports" > "$OUT/_reports_after_${LABEL}.txt"
  comm -13 "$OUT/_reports_before_${LABEL}.txt" "$OUT/_reports_after_${LABEL}.txt" \
    > "$OUT/_reports_new_${LABEL}.txt"
  echo "[repeat] arm=$LABEL R4MaskSample EXIT_CODE=$RC report=$(grep -m1 'cann_performance_eval.*json' "$OUT/_reports_new_${LABEL}.txt")"
  return $RC
}

export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:${LD_LIBRARY_PATH:-}
export ACL_OP_INIT_MODE=1
export PYTHONPATH=/tmp/opencode/r4_eval/shim${PYTHONPATH:+:$PYTHONPATH}
export R4_HTML_SHIM=1
unset TILELANG_CACHE_DIR

RC=0
for i in $(seq 1 "$N"); do
  echo "=========== REPEAT PAIR $i/$N start $(date -Is) ==========="
  run_arm "base_r$i"    "$STAGE/BASE_a764e117_r4_fused_sampling.py" a764e117ce150d8677096c2e00bde3db3ebc3ec16aba63e4c613d8dffdfbd117 || RC=$?
  run_arm "cand_r$i"    "$STAGE/CAND_c59a83bc_r4_fused_sampling.py" "$CAND_SHA" || RC=$?
done

# ---- ALWAYS restore the candidate ----
cp -f "$STAGE/CAND_c59a83bc_r4_fused_sampling.py" "$R4/r4_fused_sampling.py"
GOT=$(sha256sum "$R4/r4_fused_sampling.py" | cut -d' ' -f1)
echo "[repeat] RESTORED sha256=$GOT  ok=$([ "$GOT" = "$CAND_SHA" ] && echo YES || echo '*** NO ***')"
cp -f "$R4/r4_fused_sampling.py" "$R4/cann_bench_submission/cann_bench/_r4_kernels.py"
cmp -s "$R4/r4_fused_sampling.py" "$R4/cann_bench_submission/cann_bench/_r4_kernels.py" \
  && echo "[repeat] submission copy cmp-IDENTICAL" || echo "[repeat] *** submission copy DIFFERS ***"
( cd "$R4/cann_bench_submission" && bash build.sh ) > "$OUT/rebuild_wheel_after_repeat.log" 2>&1
echo "[repeat] wheel rebuilt rc=$?"
echo "[repeat] REPEAT DONE $(date -Is) RC=$RC"
exit $RC
