"""Iter3 T1 dispatch verification: which kernel each R1-R6 workload selects.

Read-only w.r.t. gqa.py; clears the build cache between probes and reports
the chosen spec name from _KERNEL_BUILD_CACHE.  Also probes the R6 c2 shape
under --force g2stack semantics (flat disabled) to confirm the fallback arm.

Run (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python3 examples/gqa/perf_tuning/probe_dispatch_iter3.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

import gqa as G  # noqa: E402

# (name, B, S, Skv, Nq, Nkv, causal, dtype)
PROBES = [
    ("R1_default_perhead", 1, 128, 128, 8, 2, True, torch.float16),
    ("R2_l0c1_flat", 2, 128, 128, 32, 8, True, torch.float16),
    ("R3_c3like_flat", 2, 128, 128, 128, 8, True, torch.bfloat16),
    ("R4_c4like_g2stack_ns2", 4, 256, 256, 64, 8, False, torch.bfloat16),
    ("R5_c11_decode_dense", 16, 1, 2048, 32, 8, False, torch.float16),
    ("R6_c2_flat", 2, 128, 128, 48, 4, False, torch.float16),
]


def run_probe(name, B, S, Skv, Nq, Nkv, causal, dt, disable_flat=False):
    saved = None
    if disable_flat:
        saved = G._gqa_kernel_prefill_g2stack_flat

        def _boom(*a, **k):
            raise RuntimeError("[AB-PROBE] flat disabled")

        G._gqa_kernel_prefill_g2stack_flat = _boom
    G._KERNEL_BUILD_CACHE.clear()
    G._COMPILED_CACHE.clear()
    torch.manual_seed(0)
    q = (torch.rand(B, S, Nq, 128) * 2 - 1).to(dt).npu()
    k = (torch.rand(B, Skv, Nkv, 128) * 2 - 1).to(dt).npu()
    v = (torch.rand(B, Skv, Nkv, 128) * 2 - 1).to(dt).npu()
    o = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
    torch.npu.synchronize()
    chosen = [val[2] for val in G._KERNEL_BUILD_CACHE.values()]
    tag = f"{name}[force_g2stack]" if disable_flat else name
    print(f"{tag}: chosen={chosen} out={tuple(o.shape)}", flush=True)
    if saved is not None:
        G._gqa_kernel_prefill_g2stack_flat = saved


for p in PROBES:
    run_probe(*p)
# R6 fallback arm
run_probe("R6_c2_flat", 2, 128, 128, 48, 4, False, torch.float16, disable_flat=True)
print("[PROBE_ALL_DONE]", flush=True)
