#!/bin/bash
# Iter8 Gate-3 post-integration msprof bottom line (INTEGRATED gqa.py, no
# monkeypatch): confirm on the shipped file that
#   (a) a COVERED shape's chain has NO transpose_out instance any more
#       (r2: transpose_qkv + main only) and its e2e matches the A/B ARM;
#   (b) an OUT-OF-DOMAIN prefill shape STILL launches transpose_out with the
#       P1-retiled grid (r1: blk=16 => the iter7 retile is live);
#   (c) a NON-ALIGNED in-flat-dispatch shape still uses transpose_out(s_real)
#       (padrow_s100 is not in the probe shape table, so this is covered by
#       the Gate-3 output capture/comparator instead);
#   (d) decode (r5) is untouched and reproduces its anchor (145.89us iter6 /
#       146.81 iter7 / 147.86 iter7-Gate3).
# Env contract: repo root, unset PYTHONPATH, ACL_OP_INIT_MODE=1, sqlite3 lib.
# Usage: run_g3_iter8.sh
set -u
cd /mnt/workspace/cann/20260910_gqa_V02/tilelang-ascend
unset PYTHONPATH
export ACL_OP_INIT_MODE=1
export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:${LD_LIBRARY_PATH:-}
OUT=examples/gqa/perf_tuning
LOG=/tmp/opencode/iter8_g3_msprof.log
AB=examples/gqa/perf_tuning/prof_gqa_ab_iter8.py
touch "$LOG"
echo "=== G3 MSPROF START $(date +%T) md5=$(md5sum examples/gqa/gqa.py | cut -d' ' -f1) ===" >> "$LOG"

# The integrated file must show the P2 behaviour with NO patches at all: run
# the *base* arm (which asserts "not contaminated by the variant") on a
# covered shape.  If the integration is live, tout is absent there too.
run_g3() { # shape arm n
  local d="$OUT/msprof_op_iter8_G3_$1_$2_s$3"
  if [ -d "$d" ]; then echo "SKIP G3_$1_$2_s$3 (exists)" >> "$LOG"; return; fi
  echo "=== START G3_$1_$2_s$3 $(date +%T) ===" >> "$LOG"
  timeout 900 msprof op --launch-count=3 --aic-metrics=PipeUtilization,Memory,L2Cache \
      --output="$d" python3 $AB --arm "$2" --shape "$1" >> "$LOG" 2>&1
  echo "=== END G3_$1_$2_s$3 rc=$? $(date +%T) ===" >> "$LOG"
}

for n in 1 2; do
  run_g3 r2 base "$n"    # covered: expect NO tout instance, e2e ~37us
  run_g3 r1 base "$n"    # out-of-domain per-head: expect tout blk=16 (P1 retile live)
  run_g3 r5 base "$n"    # decode: main only, anchor ~146us
  run_g3 r4 base "$n"    # covered flat_kv2: expect NO tout, e2e ~217us
done

echo "=== G3 MSPROF DONE $(date +%T) ===" >> "$LOG"
npu-smi info >> "$LOG" 2>&1
ps aux | grep -c "[m]sprof" >> "$LOG" 2>&1
