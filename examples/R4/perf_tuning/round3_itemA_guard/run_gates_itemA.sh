#!/bin/bash
# ---------------------------------------------------------------------------------------------
# Round-3 item-A gate runner. Identical shape to round3_splitv_perf/run_gates_round3.sh so the
# counts are directly comparable, but writes into round3_itemA_guard/ and takes a LABEL argument
# so the same script serves the pre-change baseline, the post-change candidate, and the rollback.
#
# Expected counts on a764e117 (Orchestrator-verified, must not go down):
#   static     38 STATIC_PASS / 0 FAIL
#   all       143 PRECISION_PASS / 0 FAIL, 56 BOUNDARY_PASS / 6 WARN, 38 STATIC_PASS
#   stability  21 PASS / 0 FAIL, 8 ADV all dropped=0
#   stress     12/12 selfdiff=0 goldenfail=0 invariant=0
#   coverage   25 PASS / 0 MISS / 0 N/A
#
# Usage: run_gates_itemA.sh <label>
# ---------------------------------------------------------------------------------------------
set -uo pipefail
R4=/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4
TL=/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend
D=$R4/perf_tuning/round3_itemA_guard
COV=$TL/.agents/skills/tilelang-op-test-design/scripts/coverage_check.py
LABEL="${1:?usage: run_gates_itemA.sh <label>}"

# env recipe (task section 5): NEVER `source set_env.sh` from the repo root -- the uncompiled
# tilelang/ source tree would shadow the installed wheel (ModuleNotFoundError: No module named tvm).
export PYTHONPATH=/home/developer/Ascend/cann-9.2.0/python/site-packages:/home/developer/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe:
export ACL_OP_INIT_MODE=1
export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:${LD_LIBRARY_PATH:-}
cd "$R4" || exit 9

sha256sum r4_fused_sampling.py test_r4_fused_sampling.py proto.yaml \
          cann_bench_submission/cann_bench/_r4_kernels.py > "$D/gate_${LABEL}_sha256_before.txt"
cat "$D/gate_${LABEL}_sha256_before.txt"
: > "$D/gate_${LABEL}_exits.txt"

for L in static all stability stress; do
  S=$(date +%s)
  python test_r4_fused_sampling.py --level "$L" > "$D/gate_${LABEL}_${L}.log" 2>&1
  echo "gate_${L}=$? wall=$(( $(date +%s) - S ))s" | tee -a "$D/gate_${LABEL}_exits.txt"
done

python "$COV" test_r4_fused_sampling.py --proto proto.yaml > "$D/gate_${LABEL}_coverage.log" 2>&1
echo "gate_coverage=$?" | tee -a "$D/gate_${LABEL}_exits.txt"

sha256sum r4_fused_sampling.py test_r4_fused_sampling.py proto.yaml \
          cann_bench_submission/cann_bench/_r4_kernels.py > "$D/gate_${LABEL}_sha256_after.txt"
if cmp -s "$D/gate_${LABEL}_sha256_before.txt" "$D/gate_${LABEL}_sha256_after.txt"; then
  echo "sha256_stable_before_after_gates=YES" | tee -a "$D/gate_${LABEL}_exits.txt"
else
  echo "sha256_stable_before_after_gates=*** NO ***" | tee -a "$D/gate_${LABEL}_exits.txt"
  diff "$D/gate_${LABEL}_sha256_before.txt" "$D/gate_${LABEL}_sha256_after.txt"
fi

# ---- count summary, machine-readable ----
{
  echo "label=$LABEL"
  echo "static_STATIC_PASS=$(grep -c '\[STATIC_PASS\]'   "$D/gate_${LABEL}_static.log")"
  echo "static_STATIC_FAIL=$(grep -c '\[STATIC_FAIL\]'   "$D/gate_${LABEL}_static.log")"
  echo "all_PRECISION_PASS=$(grep -c '\[PRECISION_PASS\]' "$D/gate_${LABEL}_all.log")"
  echo "all_PRECISION_FAIL=$(grep -c '\[PRECISION_FAIL\]' "$D/gate_${LABEL}_all.log")"
  echo "all_STATIC_PASS=$(grep -c '\[STATIC_PASS\]'     "$D/gate_${LABEL}_all.log")"
  echo "all_BOUNDARY_PASS=$(grep -c '\[BOUNDARY_PASS\]' "$D/gate_${LABEL}_all.log")"
  echo "all_BOUNDARY_WARN=$(grep -c '\[BOUNDARY_WARN\]' "$D/gate_${LABEL}_all.log")"
  echo "all_BITWISE_PASS=$(grep -c '\[BITWISE_PASS\]'   "$D/gate_${LABEL}_all.log")"
  echo "all_BITWISE_FAIL=$(grep -c '\[BITWISE_FAIL\]'   "$D/gate_${LABEL}_all.log")"
  echo "stab_PRECISION_PASS=$(grep -c '\[PRECISION_PASS\]' "$D/gate_${LABEL}_stability.log")"
  echo "stab_PRECISION_FAIL=$(grep -c '\[PRECISION_FAIL\]' "$D/gate_${LABEL}_stability.log")"
  echo "stress_PRECISION_PASS=$(grep -c '\[PRECISION_PASS\]' "$D/gate_${LABEL}_stress.log")"
  echo "stress_PRECISION_FAIL=$(grep -c '\[PRECISION_FAIL\]' "$D/gate_${LABEL}_stress.log")"
  echo "cov_PASS=$(grep -c '\[PASS\]' "$D/gate_${LABEL}_coverage.log")"
  echo "cov_MISS=$(grep -c '\[MISS\]' "$D/gate_${LABEL}_coverage.log")"
  echo "cov_NA=$(grep -c 'N/A'    "$D/gate_${LABEL}_coverage.log")"
} > "$D/gate_${LABEL}_counts.txt"
cat "$D/gate_${LABEL}_counts.txt"
echo "GATES_DONE label=$LABEL $(date -Is)" | tee -a "$D/gate_${LABEL}_exits.txt"
