#!/bin/bash
# Iter11 Gate-3 post-integration msprof bottom line (INTEGRATED gqa.py, ZERO
# monkeypatch — prof_gqa_ab_iter11.py's base arm installs no patches, so the
# run exercises the integrated adapter's native in_bsnd path): confirm that
#   (a) the WAVE-2 COVERED domains' chains have NO transpose_qkv instance any
#       more (c19 prexpert, c9 g2stack: main ONLY) and their e2e lands in the
#       Gate-2 w2-arm band (c19 ~33.7us, c9 ~565us);
#   (b) the NON-ALIGNED gate domains STILL launch transpose_qkv (skv200:
#       wave-1 flat gate; skv400g: wave-2 g2stack gate — S_kv=400 => tqkv
#       PRESENT tail-mode, zero-fill columns);
#   (c) the EXCLUDED per-head domain is unchanged (r1: tqkv + main + tout with
#       the P1-retiled blk=16 grid);
#   (d) the WAVE-1 flat domain is unchanged (r2: main ONLY, anchor band);
#   (e) decode (r5) untouched, reproduces its anchor (~146us).
# Env contract: repo root, unset PYTHONPATH, ACL_OP_INIT_MODE=1, sqlite3 lib.
# Usage: run_g3_iter11.sh
set -u
cd /mnt/workspace/cann/20260910_gqa_V02/tilelang-ascend
unset PYTHONPATH
export ACL_OP_INIT_MODE=1
export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:${LD_LIBRARY_PATH:-}
OUT=examples/gqa/perf_tuning
LOG=/tmp/opencode/iter11_g3_msprof.log
AB=examples/gqa/perf_tuning/prof_gqa_ab_iter11.py
mkdir -p /tmp/opencode
touch "$LOG"
echo "=== G3 MSPROF START $(date +%T) md5=$(md5sum examples/gqa/gqa.py | cut -d' ' -f1) ===" >> "$LOG"

run_g3() { # shape n
  local d="$OUT/msprof_op_iter11_G3_$1_base_s$2"
  if [ -d "$d" ]; then echo "SKIP G3_$1_base_s$2 (exists)" >> "$LOG"; return; fi
  echo "=== START G3_$1_base_s$2 $(date +%T) ===" >> "$LOG"
  timeout 1800 msprof op --launch-count=3 --aic-metrics=PipeUtilization,Memory,L2Cache \
      --output="$d" python3 $AB --arm base --shape "$1" >> "$LOG" 2>&1
  echo "=== END G3_$1_base_s$2 rc=$? $(date +%T) ===" >> "$LOG"
}

for n in 1 2; do
  run_g3 c19 "$n"      # prexpert covered: expect main ONLY, e2e ~33.7us (Gate-2 w2 band)
  run_g3 c9 "$n"       # g2stack covered: expect main ONLY, e2e ~565us
  run_g3 skv200 "$n"   # wave-1 flat gate: expect tqkv PRESENT (tail mode)
  run_g3 skv400g "$n"  # wave-2 g2stack gate: expect tqkv PRESENT (tail mode)
  run_g3 r1 "$n"       # per-head EXCLUDED: expect tqkv + main + tout blk=16 (P1 live)
  run_g3 r2 "$n"       # wave-1 flat: expect main ONLY, anchor band (unchanged)
  run_g3 r5 "$n"       # decode: main only, anchor ~146us
done

echo "=== G3 MSPROF DONE $(date +%T) ===" >> "$LOG"
npu-smi info >> "$LOG" 2>&1
echo "residual msprof procs: $(ps aux | grep -c '[m]sprof')" >> "$LOG"
