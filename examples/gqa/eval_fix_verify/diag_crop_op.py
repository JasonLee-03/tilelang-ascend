"""Profile (chrome trace): which NPU kernel does the output-crop
.contiguous() dispatch, vs slice-assignment vs permuted transpose?"""

import json
import os
import sys

import torch
import torch_npu  # noqa: F401
from torch_npu.profiler import profile, ProfilerActivity  # noqa: F401

OUT = "/tmp/opencode/traces"
os.makedirs(OUT, exist_ok=True)


def kernel_names(path):
    with open(path) as f:
        data = json.load(f)
    names = {}
    for evt in data.get("traceEvents", []):
        cat = evt.get("cat", "")
        if cat in ("kernel", "gpu_op", "npu_op") or "kernel" in str(cat):
            name = evt.get("name", "?")
            names[name] = names.get(name, 0) + 1
    return names


def run(tag, fn):
    path = os.path.join(OUT, f"{tag}.json")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
        fn()
        torch.npu.synchronize()
    prof.export_chrome_trace(path)
    print(f"--- {tag} ---")
    for name, count in sorted(kernel_names(path).items()):
        print(f"  {name}  x{count}")


def main():
    B, Nq, S_pad, S, D = 2, 32, 128, 124, 128
    padded = torch.randn(B, Nq, S_pad, D, dtype=torch.float16, device="npu")

    run("contig_crop", lambda: padded[:, :, :S, :].contiguous())

    out = torch.empty((B, Nq, S_pad, D), dtype=torch.float16, device="npu")
    run("slice_assign", lambda: out[:, :, :S, :].__setitem__(
        slice(None), padded[:, :, :S, :].detach()))

    out2 = torch.empty((B, Nq, S, D), dtype=torch.float16, device="npu")
    run("empty_assign_strided_src",
        lambda: out2.__setitem__(
            (slice(None), slice(None), slice(None), slice(None)),
            padded[:, :, :S, :].detach()))

    run("permuted_contig", lambda: padded.transpose(1, 2).contiguous())


if __name__ == "__main__":
    main()
