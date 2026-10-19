"""Bitwise output capture for R4 K1+K2, both dispatch paths.

Runs a deterministic recipe (torch.manual_seed(0), randn*8.0 -- the SAME recipe as
test_r4_fused_sampling.py:_make_inputs, LOGIT_SCALE=8.0) over a shape set chosen to
cover every distinct K1 JIT specialization signature (V, K_MAX, BV, NEED_MIN_P) in
the 18 bench cases, plus the tail-adversary shapes.

Dumps the raw bytes of every output to <out_dir>/<tag>__<path>__<name>.bin and a
sha256 manifest. Comparing the manifests before/after an edit gives a BITWISE
verdict (not a tolerance verdict).

Usage: bitwise_capture.py <out_dir>
"""
import hashlib
import json
import os
import sys

import torch

R4 = "/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4"
sys.path.insert(0, R4)
import torch_npu  # noqa: E402,F401  (registers the .npu() device)
from r4_fused_sampling import r4_fused_sampling  # noqa: E402

LOGIT_SCALE = 8.0
out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)

# (tag, B, V, K_MAX, BV, need_min_p)  -- covers every distinct K1 specialization
# among the 18 bench cases + the tail-adversary sentinels from --level stability.
SHAPES = [
    # --- bench-case signatures ---
    ("c01_nblk0_tail256",   2,   256, 256, 4096, False),   # n_blk_full=0 (main path NEVER runs)
    ("c14_nblk0_tail1024", 32,  1024, 256, 4096, False),   # n_blk_full=0
    ("c02_nblk1_tail0",     1,  4096, 256, 4096, False),   # TAIL=0
    ("c09_nblk1_tail0_B64",64,  4096, 256, 4096, False),
    ("c03_nblk4_tail0",     8, 16384, 256, 4096, False),
    ("c12_nblk8_tail0",    16, 32768, 256, 4096, False),
    ("c18_nblk16_tail0",   16, 65536, 256, 4096, True),    # need_min_p=True
    ("c05_tail313",         7,  4409, 256, 4096, False),   # odd prime V
    ("c06_tail2048",        8, 10240, 256, 4096, False),   # WORST UB case
    ("c07_tail512",         4, 152064, 256, 4096, False),
    ("c15_tail512_B64",    16, 152064, 256, 4096, False),  # the 37-full-block case
    ("c17_tail825",         4, 152377, 256, 4096, False),
    ("c10_nblk1_minp",      4,  4096, 256, 4096, True),
    ("c11_nblk2_minp",      8,  8192, 256, 4096, True),
    ("c08_bv2048_tail0",    4, 16384, 128, 2048, False),   # BV=2048 K_MAX=128
    ("c13_bv2048_tail256",  6,  2304, 128, 2048, False),
    # --- tail-adversary sentinels (--level stability) ---
    ("adv_tail313",         4,  4409, 256, 4096, False),
    ("adv_tail257",         4,  4353, 256, 4096, False),
    ("adv_tail511",         4,  4607, 256, 4096, False),
    ("adv_tail261",         4,  4357, 256, 4096, False),
    # --- k edges ---
    ("edge_k1",             2, 16384, 256, 4096, False),
    ("edge_kmax",           2, 16384, 256, 4096, False),
]
# k override for the two edge rows
K_OVERRIDE = {"edge_k1": 1, "edge_kmax": 256}


def make(B, V, k, mu, p):
    torch.manual_seed(0)
    logits = torch.randn(B, V, dtype=torch.float32) * LOGIT_SCALE
    return (logits,
            torch.full((B,), k, dtype=torch.int32),
            torch.full((B,), p, dtype=torch.float32),
            torch.full((B,), mu, dtype=torch.float32))


manifest = {}
for (tag, B, V, KMAX, BVx, nmp) in SHAPES:
    k = K_OVERRIDE.get(tag, 40)
    p, mu = (1.0, 0.0) if not nmp else (0.95, 0.05)
    logits, ks, ps, mus = make(B, V, k, mu, p)
    lg, kn, pn, mn = logits.npu(), ks.npu(), ps.npu(), mus.npu()
    rand = torch.arange(B, dtype=torch.int32)

    # ---- path A: FUSE_GUMBEL=False -> w_out ----
    oA = r4_fused_sampling(lg, kn, pn, mn, nmp, KMAX, BVx)
    torch.npu.synchronize()
    # ---- path B: FUSE_GUMBEL=True -> tok_out ----
    oB = r4_fused_sampling(lg, kn, pn, mn, nmp, KMAX, BVx, rand=rand.npu(), FUSE_GUMBEL=True)
    torch.npu.synchronize()

    names = ["m_out", "s_out", "tau_out", "valid", "w_out"]
    entry = {}
    for path, outs in (("mask", oA), ("gumbel", oB)):
        nm = names if path == "mask" else ["m_out", "s_out", "tau_out", "valid", "tok_out"]
        for n, t in zip(nm, outs):
            b = t.cpu().contiguous().view(torch.uint8).numpy().tobytes() \
                if t.dtype in (torch.float32, torch.int32) else t.cpu().numpy().tobytes()
            h = hashlib.sha256(b).hexdigest()
            fn = f"{tag}__{path}__{n}.bin"
            open(os.path.join(out_dir, fn), "wb").write(b)
            entry[f"{path}.{n}"] = dict(sha256=h, shape=list(t.shape), dtype=str(t.dtype), nbytes=len(b))
    manifest[tag] = dict(B=B, V=V, K_MAX=KMAX, BV=BVx, need_min_p=nmp, k=k, p=p, mu=mu,
                         n_blk_full=V // BVx, TAIL=V - (V // BVx) * BVx, outputs=entry)
    print(f"{tag:22s} B={B:<4}V={V:<7}K={KMAX:<4}BV={BVx:<5}nmp={int(nmp)} "
          f"n_blk_full={V // BVx:<3}TAIL={V - (V // BVx) * BVx:<5} "
          f"tau={entry['mask.tau_out']['sha256'][:16]} tok={entry['gumbel.tok_out']['sha256'][:16]}")
    del lg, kn, pn, mn, oA, oB, logits
    torch.npu.empty_cache()

json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
print(f"\nwrote {out_dir}/manifest.json ({len(manifest)} shapes x 2 paths x 5 outputs)")
