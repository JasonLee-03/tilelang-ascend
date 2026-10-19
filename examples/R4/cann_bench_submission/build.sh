#!/bin/bash
# ---------------------------------------------------------------------------
# CANN Bench submission build for the R4 fused-sampling TileLang operators.
#
# Single source of truth: the delivered kernel examples/R4/r4_fused_sampling.py
# (one directory up from this submission project). At build time we vendor it
# VERBATIM (byte-for-byte) into cann_bench/_r4_kernels.py, then build a wheel
# named cann_bench. There is no hand-maintained fork of the kernel: rebuild
# after any kernel change and the wheel picks it up automatically.
#
# The harness (kernel_eval/data/package_manager.py) deletes build/, dist/ and
# *.egg-info before invoking `bash build.sh` with cwd=<this dir> and a 600 s
# timeout, then requires rc==0 AND a dist/cann_bench*.whl. This script is also
# self-contained / idempotent so it can be run by hand.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Resolve the delivered kernel relative to THIS script (robust to cwd).
KERNEL_SRC="${SCRIPT_DIR}/../r4_fused_sampling.py"
GEN_PKG_DIR="${SCRIPT_DIR}/cann_bench"
GEN_DST="${GEN_PKG_DIR}/_r4_kernels.py"

if [[ ! -f "${KERNEL_SRC}" ]]; then
    echo "[build.sh] FATAL: kernel source not found: ${KERNEL_SRC}" >&2
    echo "[build.sh] Expected the delivered operator at examples/R4/r4_fused_sampling.py" >&2
    echo "[build.sh] (resolved relative to this script's own directory)." >&2
    exit 1
fi

# Clean stale artifacts + generated module (idempotent; the harness also removes
# build/ dist/ *.egg-info before calling us, but stay self-contained).
rm -rf build dist cann_bench.egg-info "${GEN_DST}"
mkdir -p "${GEN_PKG_DIR}"

# Vendor the kernel VERBATIM. NOTE: only r4_fused_sampling.py is copied -- the
# golden (r4_ref.py) and the test file are NOT part of the submission.
cp -f "${KERNEL_SRC}" "${GEN_DST}"
echo "[build.sh] generated ${GEN_DST#"${SCRIPT_DIR}"/} from ${KERNEL_SRC}"

# Build the wheel (dist/cann_bench-<version>-py3-none-any.whl).
python setup.py bdist_wheel

echo "=== Build complete ==="
ls -la dist/
