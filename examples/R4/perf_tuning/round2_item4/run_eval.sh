#!/bin/bash
# cann-bench 36-case msprof eval for BOTH R4 tasks (18 cases each).
# Recipe copied verbatim from /tmp/opencode/r4_eval/run_step4_fs_x3.sh (round-1 harness),
# which is the known-good invocation (R4_HTML_SHIM + sqlite3 LD_LIBRARY_PATH prefix).
# Usage: run_eval.sh <label>
R=/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/.agents/skills/cann-bench
TL=/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend
LABEL="${1:?need label}"
cd "$R" || exit 9
export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:$LD_LIBRARY_PATH
export ACL_OP_INIT_MODE=1
export TILELANG_CACHE_DIR=/tmp/opencode/r4_tl_cache
export PYTHONPATH=/tmp/opencode/r4_eval/shim${PYTHONPATH:+:$PYTHONPATH}
export R4_HTML_SHIM=1

sha256sum "$TL/examples/R4/r4_fused_sampling.py" \
          "$TL/examples/R4/cann_bench_submission/cann_bench/_r4_kernels.py"

for OP in R4FusedSampling R4MaskSample; do
  echo "################ ${LABEL} ${OP} ################"
  S=$(date +%s)
  ./scripts/run_evaluation.sh --bench-name cann \
    --task-dir bench_lab/tilelang_ascend_bench \
    --operator "$OP" \
    --source-dir "$TL/examples/R4/cann_bench_submission" \
    --device-id 0 --processes-per-card 1 --timeout-per-process 1800 \
    --reports-dir "$TL/examples/R4/bench/reports" -v
  echo "### ${LABEL} ${OP} EXIT_CODE=$? WALL=$(( $(date +%s) - S ))s"
done
