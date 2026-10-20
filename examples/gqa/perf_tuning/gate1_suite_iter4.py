"""Gate 1 harness — run the FULL test_gqa.py suite with a variant monkey-patched
into the dispatch, WITHOUT modifying gqa.py / test_gqa.py.

Patches gqa._gqa_kernel_prefill_g2stack_flat with the arm's variant factory,
then executes examples/gqa/test_gqa.py as __main__ (runpy) with --level all.
In-gate shapes inside the suite then build the variant; every other dispatch
path is untouched.  A build failure of the variant would surface as a
"[gqa][fallback] kernel 'g2stack_flat' failed to compile" line in the log —
the log must contain ZERO such lines for the gate to count.

Usage (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python3 examples/gqa/perf_tuning/gate1_suite_iter4.py ns4   > <log> 2>&1
"""

import importlib
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_DIR = os.path.dirname(HERE)
sys.path.insert(0, GQA_DIR)
sys.path.insert(0, HERE)

VARIANT_MODULES = {"ns4": "gqa_flat_ns4_variant", "g2first": "gqa_flat_g2first_variant"}

arm = sys.argv[1]
assert arm in VARIANT_MODULES, f"arm must be one of {list(VARIANT_MODULES)}"

import gqa as G  # noqa: E402

mod = importlib.import_module(VARIANT_MODULES[arm])
G._gqa_kernel_prefill_g2stack_flat = mod._gqa_kernel_prefill_g2stack_flat
print(f"[GATE1] arm={arm} patched gqa._gqa_kernel_prefill_g2stack_flat <- {mod.__file__}", flush=True)

sys.argv = ["test_gqa.py", "--level", "all"]
runpy.run_path(os.path.join(GQA_DIR, "test_gqa.py"), run_name="__main__")
