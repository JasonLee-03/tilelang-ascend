#!/bin/bash
# Round-3 environment recipe (task section 4). Source this, do NOT `source set_env.sh` from the
# repo root: that prepends the repo root to PYTHONPATH and the UNCOMPILED tilelang/ source tree
# shadows the installed wheel -> ModuleNotFoundError: No module named 'tvm'.
#
# NOTE: TILELANG_CACHE_DIR is deliberately NOT exported -> tilelang uses the default
# ~/.tilelang/cache, which Stage 2 already warmed for the rev3 candidate (92 entries newer than
# the rev3 delivery). prewarm.py warms it symmetrically for BOTH A/B arms before any timing run.
R4=/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4
export PYTHONPATH=/home/developer/Ascend/cann-9.2.0/python/site-packages:/home/developer/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe:
export ACL_OP_INIT_MODE=1
export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:${LD_LIBRARY_PATH:-}
cd "$R4" || exit 9
