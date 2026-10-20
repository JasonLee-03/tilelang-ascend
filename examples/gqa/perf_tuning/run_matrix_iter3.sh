#!/bin/bash
# Iter3 T2 measurement matrix — all runs in ONE session (same shell), sequential.
# Env contract: repo root, unset PYTHONPATH, ACL_OP_INIT_MODE=1, sqlite3 lib for msprof export.
set -u
cd /mnt/workspace/cann/20260910_gqa_V02/tilelang-ascend
unset PYTHONPATH
export ACL_OP_INIT_MODE=1
export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:$LD_LIBRARY_PATH
OUT=examples/gqa/perf_tuning
LOG=/tmp/opencode/iter3_matrix.log
: > "$LOG"

run() { # name, script, flags...
  local name="$1"; shift
  local d="$OUT/msprof_op_iter3_${name}"
  if [ -d "$d" ]; then echo "SKIP $name (exists)" >> "$LOG"; return; fi
  echo "=== START $name $(date +%T) ===" >> "$LOG"
  timeout 900 msprof op --kernel-name=main_kernel --aic-metrics=PipeUtilization \
      --output="$d" "$@" >> "$LOG" 2>&1
  echo "=== END $name rc=$? $(date +%T) ===" >> "$LOG"
}

PROF=examples/gqa/prof_gqa_pipeline.py
AB=examples/gqa/perf_tuning/prof_gqa_ab_iter3.py

# R1 default (s1 already collected)
run r1_s2 python3 $PROF
run r1_s3 python3 $PROF
# R2 L0-c1 -> flat
run r2_s1 python3 $PROF --batch 2 --nq 32 --nkv 8
run r2_s2 python3 $PROF --batch 2 --nq 32 --nkv 8
run r2_s3 python3 $PROF --batch 2 --nq 32 --nkv 8
# R3 c3-like -> flat
run r3_s1 python3 $PROF --batch 2 --nq 128 --nkv 8 --dtype bf16
run r3_s2 python3 $PROF --batch 2 --nq 128 --nkv 8 --dtype bf16
run r3_s3 python3 $PROF --batch 2 --nq 128 --nkv 8 --dtype bf16
# R4 c4-like -> g2stack ns=2 (kv=2)
run r4_s1 python3 $PROF --batch 4 --seq 256 --seq-kv 256 --nq 64 --nkv 8 --dtype bf16 --no-causal
run r4_s2 python3 $PROF --batch 4 --seq 256 --seq-kv 256 --nq 64 --nkv 8 --dtype bf16 --no-causal
run r4_s3 python3 $PROF --batch 4 --seq 256 --seq-kv 256 --nq 64 --nkv 8 --dtype bf16 --no-causal
# R5 c11 decode -> dense_expert ns=16 ci=4
run r5_s1 python3 $PROF --batch 16 --seq 1 --seq-kv 2048 --nq 32 --nkv 8 --no-causal
run r5_s2 python3 $PROF --batch 16 --seq 1 --seq-kv 2048 --nq 32 --nkv 8 --no-causal
run r5_s3 python3 $PROF --batch 16 --seq 1 --seq-kv 2048 --nq 32 --nkv 8 --no-causal
# R6 c2-like same-session interleaved A/B: flat (a) vs forced g2stack (b)
run r6_flat_a1 python3 $AB --batch 2 --nq 48 --nkv 4 --no-causal
run r6_g2s_b1  python3 $AB --batch 2 --nq 48 --nkv 4 --no-causal --force g2stack
run r6_flat_a2 python3 $AB --batch 2 --nq 48 --nkv 4 --no-causal
run r6_g2s_b2  python3 $AB --batch 2 --nq 48 --nkv 4 --no-causal --force g2stack
run r6_flat_a3 python3 $AB --batch 2 --nq 48 --nkv 4 --no-causal
run r6_g2s_b3  python3 $AB --batch 2 --nq 48 --nkv 4 --no-causal --force g2stack

echo "=== MATRIX DONE $(date +%T) ===" >> "$LOG"
npu-smi info >> "$LOG" 2>&1
