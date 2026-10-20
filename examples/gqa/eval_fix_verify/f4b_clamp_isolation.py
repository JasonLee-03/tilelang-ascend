"""Isolate whether the [F4b] m_i clamp changed visible-row numerics.

Runs the SAME causal S>S_kv case through the baseline (history_version)
module and the fixed module, comparing (a) each against golden and (b) the
two kernel outputs against each other, visible rows only.
"""

import importlib.util
import os
import sys

import torch
import torch_npu  # noqa: F401

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, ".agents", "skills", "cann-bench",
                                "tasks", "level4", "gqa"))
from golden import gqa as gqa_golden  # noqa: E402


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    torch.manual_seed(42)
    B, S, S_kv, N_q, N_kv, D = 2, 256, 128, 32, 8, 128
    q = (torch.rand(B, S, N_q, D) * 2 - 1).half().npu()
    k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).half().npu()
    v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).half().npu()
    ref = gqa_golden(q, k, v, -1.0, True).float().cpu()

    base = _load_module(os.path.join(_REPO, "examples", "gqa", "history_version",
                                     "gqa_impl_eval_fix_attempt1.py"), "gqa_base")
    fixed = _load_module(os.path.join(_REPO, "examples", "gqa", "gqa.py"), "gqa_fixed")

    out_b = base.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=True)
    out_f = fixed.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=True)

    ob = out_b.float().cpu()
    of = out_f.float().cpu()
    vis = slice(128, S)  # rows 128.. are visible

    print(f"baseline NaN rows: {torch.isnan(ob[:, :128]).all(dim=-1).all(dim=-1).sum().item()}")
    print(f"fixed    NaN total: {torch.isnan(of).sum().item()}")
    print(f"baseline vs golden (visible): max_diff="
          f"{(ob[:, vis] - ref[:, vis]).abs().max().item():.3e}")
    print(f"fixed    vs golden (visible): max_diff="
          f"{(of[:, vis] - ref[:, vis]).abs().max().item():.3e}")
    print(f"fixed    vs baseline (visible): max_diff="
          f"{(of[:, vis] - ob[:, vis]).abs().max().item():.3e}")
    print(f"fixed    masked rows absmax: {of[:, :128].abs().max().item():.3e}")


if __name__ == "__main__":
    main()
