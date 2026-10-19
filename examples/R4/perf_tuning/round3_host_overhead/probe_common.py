"""Shared helpers for the round-3 HOST-OVERHEAD attribution probes (read-only investigation).

MEASUREMENT BOUNDARY (task section 2 -- READ THIS BEFORE INTERPRETING ANY NUMBER)
----------------------------------------------------------------------------------
Every number produced by the probes in this directory is a HOST-SIDE wall-clock number
obtained with ``time.perf_counter_ns()``. None of them is a kernel duration and none of
them may be used to claim a device-time speed-up.

That boundary is legitimate here for one specific reason: msprof itself is one of the
objects under investigation (candidate C1). The kernel-duration figures that the project
ban on Python timing protects are already established elsewhere, from msprof, in
``../round3_splitv_perf/`` (scored_us == cann-bench elapsed_us). This directory only
measures how long the HOST spends preparing/submitting work.

What ``perf_counter_ns`` around an asynchronous launch actually measures
-----------------------------------------------------------------------
TileLang launches are asynchronous: the call returns as soon as the work is QUEUED, not
when it finishes. So ``perf_counter_ns`` around a launch measures host submission cost,
which is exactly the quantity whose absence shows up as the inter-kernel ``gap_us`` in
``../round3_splitv_perf/span_vs_scored.txt``.

Consistency identity used throughout (and worth stating because it is what makes the
attribution falsifiable rather than a guess):

    gap_us  =  span_us - scored_us  ~=  max(0, H_launch - d_prev)

where ``H_launch`` is the host cost of preparing+submitting one launch and ``d_prev`` is
the device duration of the kernel that was already in flight. Round 3 measured
gap=458.75 us with d_prev=24.48 us  -> H ~= 483 us, and gap=3.00 us with d_prev=425.06 us
-> H ~= 428 us. Two independent steps agreeing on H in [428, 483] us is the evidence that
H is a FIXED host cost, not something that scales with the kernel.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from collections import defaultdict

# ---------------------------------------------------------------- environment ---
# Task section 5 recipe. Deliberately NOT `source set_env.sh` from the repo root: that
# prepends the repo root to PYTHONPATH and the UNCOMPILED tilelang/ source tree shadows
# the installed wheel -> ModuleNotFoundError: No module named 'tvm'.
R4 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPO = os.path.abspath(os.path.join(R4, "..", ".."))
for _p in (R4, os.path.join(R4, "bench", "hap_thw_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

HERE = os.path.dirname(os.path.abspath(__file__))


def perf_counter_ns() -> int:
    return time.perf_counter_ns()


class Seg:
    """Accumulates per-segment host timings over many iterations."""

    def __init__(self) -> None:
        self._d: dict[str, list[float]] = defaultdict(list)

    def add(self, name: str, ns: float) -> None:
        self._d[name].append(ns / 1000.0)  # us

    def stats(self) -> dict:
        out = {}
        for k, v in self._d.items():
            out[k] = dict(
                n=len(v),
                min_us=round(min(v), 3),
                med_us=round(statistics.median(v), 3),
                mean_us=round(statistics.mean(v), 3),
                max_us=round(max(v), 3),
                p90_us=round(sorted(v)[int(0.9 * (len(v) - 1))], 3),
                stdev_us=round(statistics.stdev(v), 3) if len(v) > 1 else 0.0,
            )
        return out

    def med(self, name: str) -> float:
        return statistics.median(self._d[name]) if self._d.get(name) else float("nan")


def summarize(label: str, seg: Seg, keys: list[str] | None = None) -> dict:
    st = seg.stats()
    keys = keys or [k for k in st if k != "_TOTAL"]
    total = sum(st[k]["med_us"] for k in keys if k in st)
    print(f"\n--- {label} ---")
    print(f"{'segment':<28} {'n':>5} {'min':>10} {'median':>10} {'p90':>10} {'max':>10} {'stdev':>9}")
    for k in keys:
        if k not in st:
            continue
        s = st[k]
        print(f"{k:<28} {s['n']:>5} {s['min_us']:>10.2f} {s['med_us']:>10.2f} "
              f"{s['p90_us']:>10.2f} {s['max_us']:>10.2f} {s['stdev_us']:>9.2f}")
    print(f"{'SUM(median)':<28} {'':>5} {'':>10} {total:>10.2f}")
    return dict(label=label, segments=st, sum_median_us=round(total, 3), keys=keys)


def dump(name: str, obj) -> str:
    p = os.path.join(HERE, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=1)
    print(f"\nwrote {p}")
    return p


def npu_sync() -> None:
    import torch
    torch.npu.synchronize()


def env_fingerprint() -> dict:
    """Record exactly what we ran against, so the numbers are reproducible/auditable."""
    import torch
    import torch_npu
    import tilelang
    return dict(
        tilelang=tilelang.__version__,
        tilelang_file=tilelang.__file__,
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        python=sys.version.split()[0],
        device=torch.npu.get_device_name(0),
        cwd=os.getcwd(),
        PYTHONPATH=os.environ.get("PYTHONPATH", ""),
        ACL_OP_INIT_MODE=os.environ.get("ACL_OP_INIT_MODE", ""),
        LD_LIBRARY_PATH_has_sqlite3="/usr/local/sqlite3/lib" in os.environ.get("LD_LIBRARY_PATH", ""),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
