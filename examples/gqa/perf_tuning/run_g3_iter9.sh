#!/bin/bash
# Iter9 Gate-3 post-integration msprof bottom line (INTEGRATED gqa.py, ZERO
# monkeypatch): confirm on the shipped-post-integration file that
#   (a) the NEWLY-COVERED domains' chains have NO transpose_out instance any
#       more (c19 prefill_expert, c9 g2stack: transpose_qkv + main only) and
#       their e2e lands in the Gate-2 w2-arm band;
#   (b) the EXCLUDED per-head domain STILL launches transpose_out with the
#       P1-retiled grid (r1: blk=16 => iter7 retile live, wave-2 didn't touch it);
#   (c) wave-1 flat (r2) is unchanged (still no tout, e2e in its band);
#   (d) decode (r5) untouched, reproduces its anchor.
# Env contract: repo root, unset PYTHONPATH, ACL_OP_INIT_MODE=1, sqlite3 lib.
# Usage: run_g3_iter9.sh
set -u
cd /mnt/workspace/cann/20260910_gqa_V02/tilelang-ascend
unset PYTHONPATH
export ACL_OP_INIT_MODE=1
export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:${LD_LIBRARY_PATH:-}
OUT=examples/gqa/perf_tuning
LOG=/tmp/opencode/iter9_g3_msprof.log
AB=examples/gqa/perf_tuning/prof_gqa_ab_iter9.py
mkdir -p /tmp/opencode
touch "$LOG"
echo "=== G3 MSPROF START $(date +%T) md5=$(md5sum examples/gqa/gqa.py | cut -d' ' -f1) ===" >> "$LOG"

run_g3() { # shape n
  local d="$OUT/msprof_op_iter9_G3_$1_base_s$2"
  if [ -d "$d" ]; then echo "SKIP G3_$1_base_s$2 (exists)" >> "$LOG"; return; fi
  echo "=== START G3_$1_base_s$2 $(date +%T) ===" >> "$LOG"
  timeout 900 msprof op --launch-count=3 --aic-metrics=PipeUtilization,Memory,L2Cache \
      --output="$d" python3 $AB --arm base --shape "$1" >> "$LOG" 2>&1
  echo "=== END G3_$1_base_s$2 rc=$? $(date +%T) ===" >> "$LOG"
}

for n in 1 2; do
  run_g3 c19 "$n"   # prefill_expert covered: expect NO tout, e2e ~49.9us (Gate-2 w2 band)
  run_g3 c9 "$n"    # g2stack covered: expect NO tout, e2e ~605us
  run_g3 r1 "$n"    # per-head EXCLUDED: expect tout PRESENT blk=16 (P1 retile live)
  run_g3 r2 "$n"    # wave-1 flat covered (unchanged): expect NO tout
  run_g3 r5 "$n"    # decode: main only, anchor ~146us
done

echo "=== G3 MSPROF DONE $(date +%T) ===" >> "$LOG"
npu-smi info >> "$LOG" 2>&1
echo "residual msprof procs: $(ps aux | grep -c '[m]sprof')" >> "$LOG"
