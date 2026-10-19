"""Submission smoke: run the INSTALLED cann_bench wheel and compare bitwise with the dev file.

Proves the vendored submission copy (_r4_kernels.py) actually works end-to-end through the
harness entry points (cann_bench.r4_fused_sampling = FUSE_GUMBEL True, cann_bench.r4_mask_sample
= FUSE_GUMBEL False) and produces bit-identical results to calling the dev module directly --
on both a split-path case and a current-path case. Run from a NEUTRAL cwd (the repo's tilelang/
source dir and the submission's own cann_bench/ both shadow installed packages).
"""
import sys

sys.path.insert(0, "/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4")
import torch

import cann_bench  # the INSTALLED wheel
from r4_fused_sampling import _r4_split_plan
from r4_fused_sampling import r4_fused_sampling as dev_wrapper

LOGIT_SCALE = 8.0
CASES = [
    ("case7_SPLIT", 4, 152064, 256, 4096, 40, 0.95, 0.05, True),
    ("case12_SPLIT", 16, 32768, 256, 4096, 20, 0.9, 0.1, True),
    ("case8_SPLIT_K128", 4, 16384, 128, 2048, 40, 0.95, 0.0, False),
    ("case15_CURRENT", 64, 152064, 256, 4096, 40, 0.95, 0.05, True),
    ("case1_CURRENT", 2, 256, 256, 4096, 40, 0.95, 0.0, False),
]


def bits(t):
    t = t.detach().cpu()
    return t.view(torch.int32) if t.dtype == torch.float32 else t


ok_all = True
print(f"cann_bench installed at: {cann_bench.__file__}")
for (tag, B, V, K, BV, k, p, mu, nmp) in CASES:
    torch.manual_seed(0)
    logits = torch.randn(B, V, dtype=torch.float32) * LOGIT_SCALE
    ks = torch.full((B,), k, dtype=torch.int32)
    ps = torch.full((B,), p, dtype=torch.float32)
    mus = torch.full((B,), mu, dtype=torch.float32)
    rand = torch.arange(B, dtype=torch.int32) * 7919 + 13
    use_split, S, NBS, ST = _r4_split_plan(B, V, K, BV)
    lg, ks_n, ps_n, mus_n, rnd = (x.npu() for x in (logits, ks, ps, mus, rand))

    # fused task: wheel entry point vs dev wrapper
    w_f = cann_bench.r4_fused_sampling(lg, ks_n, ps_n, mus_n, rnd,
                                       need_min_p_sampling=nmp, K_MAX=K, BV=BV)
    d_f = dev_wrapper(lg, ks_n, ps_n, mus_n, nmp, K, BV, rand=rnd, FUSE_GUMBEL=True)
    torch.npu.synchronize()
    # mask task
    w_m = cann_bench.r4_mask_sample(lg, ks_n, ps_n, mus_n,
                                    need_min_p_sampling=nmp, K_MAX=K, BV=BV)
    d_m = dev_wrapper(lg, ks_n, ps_n, mus_n, nmp, K, BV)
    torch.npu.synchronize()

    names = ("m_out", "s_out", "tau_out", "valid", "tok_out")
    res = []
    for nm, a, b in zip(names, w_f, d_f):
        same = bool(torch.equal(bits(a), bits(b)))
        res.append(f"fused/{nm}={'=' if same else 'DIFF'}")
        ok_all &= same
    for nm, a, b in zip(("m_out", "s_out", "tau_out", "valid", "w_out"), w_m, d_m):
        same = bool(torch.equal(bits(a), bits(b)))
        res.append(f"mask/{nm}={'=' if same else 'DIFF'}")
        ok_all &= same
    print(f"  {tag:20s} B={B:3d} V={V:6d} K={K} BV={BV} path={'SPLIT' if use_split else 'CURRENT'} "
          f"S={S} NBS={NBS} S_TAIL={ST} | wheel==dev bitwise: {' '.join(res)}")

print(f"\nSUBMISSION SMOKE: {'PASS -- installed wheel bit-identical to the dev module on all '
                     f'{len(CASES)} cases x 2 tasks x 5 outputs' if ok_all else '*** FAIL ***'}")
sys.exit(0 if ok_all else 1)
