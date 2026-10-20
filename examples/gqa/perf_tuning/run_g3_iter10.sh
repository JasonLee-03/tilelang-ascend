#!/bin/bash
# Iter10 Gate-3 post-integration msprof bottom line (INTEGRATED gqa.py, ZERO
# monkeypatch — prof_gqa_ab_iter10.py's base arm installs no patches, so the
# run exercises the integrated adapter's native in_bsnd path): confirm that
#   (a) the COVERED flat domains' chains have NO transpose_qkv instance any
#       more (r2 flat kv1, c7 flat kv2: main ONLY) and their e2e lands in the
#       Gate-2 p4-arm band (r2 ~26.4us, c7 ~36.2us);
#   (b) the NON-ALIGNED gate domain STILL launches transpose_qkv (skv200:
#       flat kv2 dispatch hit but S_kv=200 => tqkv PRESENT, zero-fill columns);
#   (c) the EXCLUDED per-head domain is unchanged (r1: tqkv + main + tout with
#       the P1-retiled blk=16 grid);
#   (d) decode (r5) untouched, reproduces its anchor (~146us).
# Env contract: repo root, unset PYTHONPATH, ACL_OP_INIT_MODE=1, sqlite3 lib.
# Usage: run_g3_iter10.sh
set -u
cd /mnt/workspace/cann/20260910_gqa_V02/tilelang-ascend
unset PYTHONPATH
export ACL_OP_INIT_MODE=1
export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:${LD_LIBRARY_PATH:-}
OUT=examples/gqa/perf_tuning
LOG=/tmp/opencode/iter10_g3_msprof.log
AB=examples/gqa/perf_tuning/prof_gqa_ab_iter10.py
mkdir -p /tmp/opencode
touch "$LOG"
echo "=== G3 MSPROF START $(date +%T) md5=$(md5sum examples/gqa/gqa.py | cut -d' ' -f1) ===" >> "$LOG"

run_g3() { # shape n
  local d="$OUT/msprof_op_iter10_G3_$1_base_s$2"
  if [ -d "$d" ]; then echo "SKIP G3_$1_base_s$2 (exists)" >> "$LOG"; return; fi
  echo "=== START G3_$1_base_s$2 $(date +%T) ===" >> "$LOG"
  timeout 1200 msprof op --launch-count=3 --aic-metrics=PipeUtilization,Memory,L2Cache \
      --output="$d" python3 $AB --arm base --shape "$1" >> "$LOG" 2>&1
  echo "=== END G3_$1_base_s$2 rc=$? $(date +%T) ===" >> "$LOG"
}

for n in 1 2; do
  run_g3 r2 "$n"      # flat kv1 covered: expect main ONLY, e2e ~26.4us (Gate-2 p4 band)
  run_g3 c7 "$n"      # flat kv2 covered: expect main ONLY, e2e ~36.2us
  run_g3 skv200 "$n"  # flat kv2 hit but S_kv NON-aligned: expect tqkv PRESENT
  run_g3 r1 "$n"      # per-head EXCLUDED: expect tqkv + main + tout blk=16 (P1 live)
  run_g3 r5 "$n"      # decode: main only, anchor ~146us
done

echo "=== G3 MSPROF DONE $(date +%T) ===" >> "$LOG"
npu-smi info >> "$LOG" 2>&1
echo "residual msprof procs: $(ps aux | grep -c '[m]sprof')" >> "$LOG"
