#!/bin/bash
# Round-3 A/B driver: runs the 3 passes back-to-back in ONE session, then RESTORES the delivered
# candidate kernel and proves the restore by sha256. Launched with setsid so a client-side shell
# timeout cannot kill the measurement mid-pass.
set -uo pipefail
OUT=/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4/perf_tuning/round3_splitv_perf
R4=/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4
CAND_SHA=a764e117ce150d8677096c2e00bde3db3ebc3ec16aba63e4c613d8dffdfbd117

for P in base1 cand base2; do
  echo "=========== PASS $P start $(date -Is) ==========="
  bash "$OUT/ab_run.sh" "$P"
  echo "=========== PASS $P rc=$? end $(date -Is) ==========="
done

# ---- ALWAYS leave the delivered kernel as the rev3 candidate ----
cp -f /tmp/opencode/r4_round3/CAND_a764e117_r4_fused_sampling.py "$R4/r4_fused_sampling.py"
GOT=$(sha256sum "$R4/r4_fused_sampling.py" | cut -d' ' -f1)
echo "[ab_all] RESTORED r4_fused_sampling.py sha256=$GOT"
[ "$GOT" = "$CAND_SHA" ] && echo "[ab_all] restore OK (== candidate $CAND_SHA)" \
                         || echo "[ab_all] *** RESTORE MISMATCH *** want $CAND_SHA"
# submission copy must stay verbatim-synced with the delivered kernel
cp -f "$R4/r4_fused_sampling.py" "$R4/cann_bench_submission/cann_bench/_r4_kernels.py"
echo "[ab_all] submission copy sha256=$(sha256sum "$R4/cann_bench_submission/cann_bench/_r4_kernels.py" | cut -d' ' -f1)"
# the last pass left a BASE wheel in dist/ -> rebuild from the restored candidate
( cd "$R4/cann_bench_submission" && bash build.sh ) > "$OUT/rebuild_wheel_after_ab.log" 2>&1
echo "[ab_all] wheel rebuilt rc=$? sha256=$(sha256sum "$R4"/cann_bench_submission/dist/*.whl | cut -d' ' -f1)"
echo "[ab_all] ALL PASSES DONE $(date -Is)"
