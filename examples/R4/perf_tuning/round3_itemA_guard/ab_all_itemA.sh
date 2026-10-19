#!/bin/bash
# ---------------------------------------------------------------------------------------------
# Round-3 item-A A/B driver: runs the 3 arms back-to-back in ONE session, then RESTORES the
# candidate kernel and proves the restore by sha256 + cmp against the submission copy.
# Launched with setsid so a client-side shell timeout cannot kill the measurement mid-arm.
#
#   arm order:  base1 (a764e117)  ->  cand (c59a83bc)  ->  base2 (a764e117)
#   base2 vs base1 == SAME CODE  => pure drift control
#   cand  vs base1 == primary comparison, judged per (task, case)
# ---------------------------------------------------------------------------------------------
set -uo pipefail
OUT=/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4/perf_tuning/round3_itemA_guard
R4=/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4
CAND_SHA=c59a83bcfa5f43365ab921499e2d3a6f9ec54011d861149e50d1beb12079c534
BASE_SHA=a764e117ce150d8677096c2e00bde3db3ebc3ec16aba63e4c613d8dffdfbd117

for P in base1 cand base2; do
  echo "=========== ARM $P start $(date -Is) ==========="
  bash "$OUT/ab_run_itemA.sh" "$P"
  echo "=========== ARM $P rc=$? end $(date -Is) ==========="
done

# ---- ALWAYS leave the delivered kernel as the item-A candidate ----
cp -f /tmp/opencode/r4_itemA/CAND_c59a83bc_r4_fused_sampling.py "$R4/r4_fused_sampling.py"
GOT=$(sha256sum "$R4/r4_fused_sampling.py" | cut -d' ' -f1)
echo "[ab_all_itemA] RESTORED r4_fused_sampling.py sha256=$GOT"
[ "$GOT" = "$CAND_SHA" ] && echo "[ab_all_itemA] restore OK (== candidate $CAND_SHA)" \
                         || echo "[ab_all_itemA] *** RESTORE MISMATCH *** want $CAND_SHA"
# submission copy must stay verbatim-synced with the delivered kernel (cmp is the proof)
cp -f "$R4/r4_fused_sampling.py" "$R4/cann_bench_submission/cann_bench/_r4_kernels.py"
if cmp -s "$R4/r4_fused_sampling.py" "$R4/cann_bench_submission/cann_bench/_r4_kernels.py"; then
  echo "[ab_all_itemA] submission copy cmp-IDENTICAL to the delivered kernel"
else
  echo "[ab_all_itemA] *** submission copy DIFFERS ***"
fi
echo "[ab_all_itemA] submission copy sha256=$(sha256sum "$R4/cann_bench_submission/cann_bench/_r4_kernels.py" | cut -d' ' -f1)"
# the last arm left a BASE wheel in dist/ -> rebuild from the restored candidate
( cd "$R4/cann_bench_submission" && bash build.sh ) > "$OUT/rebuild_wheel_after_ab.log" 2>&1
echo "[ab_all_itemA] wheel rebuilt rc=$? sha256=$(sha256sum "$R4"/cann_bench_submission/dist/*.whl | cut -d' ' -f1)"
python - <<'EOF'
import zipfile, hashlib, glob
w = sorted(glob.glob("/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4/"
                     "cann_bench_submission/dist/cann_bench-*.whl"))[-1]
d = zipfile.ZipFile(w).read("cann_bench/_r4_kernels.py")
print(f"[ab_all_itemA] _r4_kernels.py INSIDE the rebuilt wheel sha256={hashlib.sha256(d).hexdigest()}")
EOF
echo "[ab_all_itemA] ALL ARMS DONE $(date -Is)"
