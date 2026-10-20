"""A/B the 5 marginal bf16 failures: baseline vs fixed, same seeds.

Determines whether the tiny tolviol counts (1-9 elements, max_diff at 1-2
bf16 ULP) are pre-existing baseline behavior or introduced by the eval
fixes (F4 sumexp+1e-30 / F5 dispatch change).
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

# the 5 failing cases: (tag, B, S, S_kv, N_q, N_kv, D, causal, scale)
CASES = [
    ("c3-bf16-skv-4", 2, 128, 124, 128, 8, 128, True, -1.0),
    ("c5-bf16-base", 2, 128, 128, 32, 8, 256, True, -1.0),
    ("c6-bf16-both-4", 2, 252, 252, 128, 8, 256, True, -1.0),
    ("c10-bf16-skv-4", 4, 512, 508, 128, 8, 128, True, -1.0),
    ("c19-bf16-skv-4", 2, 128, 124, 32, 32, 128, True, -1.0),
]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    base = _load(os.path.join(_REPO, "examples", "gqa", "history_version",
                              "gqa_impl_eval_fix_attempt1.py"), "gqa_base")
    fixed = _load(os.path.join(_REPO, "examples", "gqa", "gqa.py"), "gqa_fixed")

    for (tag, B, S, S_kv, N_q, N_kv, D, causal, scale) in CASES:
        torch.manual_seed(42 + B + S + S_kv)
        q = (torch.rand(B, S, N_q, D) * 2 - 1).bfloat16().npu()
        k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).bfloat16().npu()
        v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).bfloat16().npu()

        outs = {}
        for name, mod in (("base", base), ("fixed", fixed)):
            try:
                outs[name] = mod.gqa(query=q, key=k, value=v, scaleValue=scale,
                                     is_causal=causal).float().cpu()
            except Exception as e:  # noqa: BLE001
                outs[name] = None
                print(f"[{tag}] {name} raised: {type(e).__name__}: {e}")
        ref = gqa_golden(q, k, v, scale, causal).float().cpu()

        line = f"[{tag}]"
        for name in ("base", "fixed"):
            o = outs[name]
            if o is None:
                line += f"  {name}: CRASH"
                continue
            diff = (o - ref).abs()
            md = diff.max().item()
            v1 = (diff > 4e-3 + 1e-2 * ref.abs()).sum().item()
            v2 = (diff > 2e-2 + 1e-2 * ref.abs()).sum().item()
            line += f"  {name}: max={md:.3e} viol@4e-3={v1} viol@2e-2={v2}"
        if outs["base"] is not None and outs["fixed"] is not None:
            kv = (outs["fixed"] - outs["base"]).abs().max().item()
            same = torch.equal(outs["base"], outs["fixed"])
            line += f"  fixed-vs-base max={kv:.3e} identical={same}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
