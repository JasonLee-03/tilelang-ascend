#!/usr/bin/env python3
"""Minimal debug: base1 shape, inspect where kernel output deviates from golden."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import torch  # noqa: E402
import gqa as gqa_mod  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "gqa_golden", "/mnt/workspace/cann/20260910_gqa_V02/tilelang-ascend/"
    ".agents/skills/cann-bench/tasks/level4/gqa/golden.py")
gqa_golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gqa_golden)

NPU = "npu:0"
B, S, S_kv, N_q, N_kv, D = 2, 128, 128, 32, 8, 128
torch.manual_seed(0)
q = (torch.rand(B, S, N_q, D) * 2 - 1).half().to(NPU)
k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).half().to(NPU)
v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).half().to(NPU)

out = gqa_mod.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=True)
torch.npu.synchronize()
ref = gqa_golden.gqa(q.double().cpu(), k.double().cpu(), v.double().cpu(),
                     scaleValue=-1.0, is_causal=True).half()

o = out.float().cpu()
r = ref.float()
diff = (o - r).abs()
print("max diff:", float(diff.max()), " mean diff:", float(diff.mean()))
# where are the errors concentrated?
per_row = diff.amax(dim=(0, 3))  # [S, N_q]
bad_rows = (per_row > 0.01).nonzero()
print("rows with diff>0.01:", bad_rows.shape[0], "of", S * N_q)
if bad_rows.shape[0]:
    print("first bad (s, hq):", bad_rows[:10].tolist())
    s0, hq0 = bad_rows[0].tolist()
    print("out[s0,hq0,:8]:", o[0, s0, hq0, :8].tolist())
    print("ref[s0,hq0,:8]:", r[0, s0, hq0, :8].tolist())
    # is the output maybe V-mean (unmasked softmax) or shifted?
    # compare against a non-causal golden:
    ref_nc = gqa_golden.gqa(q.double().cpu(), k.double().cpu(), v.double().cpu(),
                            scaleValue=-1.0, is_causal=False).half().float()
    print("out vs NONCAUSAL ref max diff:", float((o - ref_nc).abs().max()))
