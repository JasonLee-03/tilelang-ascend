#!/bin/bash
# Round-3 gate re-run on the FINAL delivered state.
#
# WHY THIS IS RE-RUN EVEN THOUGH THE ORCHESTRATOR ALREADY VERIFIED a764e117: the round-3 A/B
# swapped examples/R4/r4_fused_sampling.py to the pre-rev3 BASE twice and back again, and rebuilt
# the submission wheel three times. The sha256 was verified at every swap and after the restore,
# but the gates are the only thing that proves the *behaviour* is still green in the final state.
# Expected counts (Stage-2 / Orchestrator measurements on this exact sha256, must not go down):
#   static     38 STATIC_PASS / 0 FAIL
#   all       143 PRECISION_PASS / 0 FAIL, 56 BOUNDARY_PASS / 6 WARN, 38 STATIC_PASS
#   stability  21 PASS / 0 FAIL, 8 ADV all dropped=0
#   stress     12/12 selfdiff=0 goldenfail=0 invariant=0
#   coverage   25 PASS / 0 MISS / 0 N/A
#
# cwd = examples/R4 (outside the repo root, per the task section 4 env recipe).
set -uo pipefail
cd /mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4 || exit 9
D=perf_tuning/round3_splitv_perf
COV=../../.agents/skills/tilelang-op-test-design/scripts/coverage_check.py

sha256sum r4_fused_sampling.py test_r4_fused_sampling.py proto.yaml \
          cann_bench_submission/cann_bench/_r4_kernels.py > "$D/gate_sha256_before.txt"
cat "$D/gate_sha256_before.txt"
: > "$D/gate_exits.txt"

for L in static all stability stress; do
  S=$(date +%s)
  python test_r4_fused_sampling.py --level "$L" > "$D/gate_${L}.log" 2>&1
  echo "gate_${L}=$? wall=$(( $(date +%s) - S ))s" | tee -a "$D/gate_exits.txt"
done

python "$COV" test_r4_fused_sampling.py --proto proto.yaml > "$D/gate_coverage.log" 2>&1
echo "gate_coverage=$?" | tee -a "$D/gate_exits.txt"

sha256sum r4_fused_sampling.py test_r4_fused_sampling.py proto.yaml \
          cann_bench_submission/cann_bench/_r4_kernels.py > "$D/gate_sha256_after.txt"
if cmp -s "$D/gate_sha256_before.txt" "$D/gate_sha256_after.txt"; then
  echo "sha256_stable_before_after_gates=YES" | tee -a "$D/gate_exits.txt"
else
  echo "sha256_stable_before_after_gates=*** NO ***" | tee -a "$D/gate_exits.txt"
  diff "$D/gate_sha256_before.txt" "$D/gate_sha256_after.txt"
fi
echo "GATES_DONE $(date -Is)" | tee -a "$D/gate_exits.txt"
