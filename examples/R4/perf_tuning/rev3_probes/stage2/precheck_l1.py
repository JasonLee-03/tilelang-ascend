"""CPU pre-verification of the rev3 scenario-B candidate cases (DESIGN.md section 9.2.1
MANDATORY: every L0/L1 shape must be checked with golden_r4 to yield valid_ref all-1, else
the case false-fails and can be misdiagnosed as a kernel bug or even a [DESIGN_ERROR]).

Also prints the host dispatch plan so only genuinely split-path shapes are used for the
split-path coverage, and checks the V-S4 "whole split is -inf" recipes stay finite-safe.
"""
import sys

sys.path.insert(0, "/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4")
import torch

from r4_fused_sampling import VEC_NUM, _r4_split_plan
from r4_ref import golden_gumbel_sample, golden_r4

# (name, B, V, K_MAX, BV, k, p, mu, nmp, scale, offset)
CAND = [
    ("l1r3_split_k1", 8, 16384, 256, 4096, 1, 1.0, 0.0, False, 8.0, 0.0),
    ("l1r3_split_k_eq_kmax", 8, 16384, 256, 4096, 256, 1.0, 0.0, False, 8.0, 0.0),
    ("l1r3_split_minp_strong", 8, 16384, 256, 4096, 60, 0.9, 0.2, True, 8.0, 0.0),
    ("l1r3_split_k128bv2048_keq", 4, 16384, 128, 2048, 128, 1.0, 0.0, False, 8.0, 0.0),
    ("l1r3_split_s2_b16", 16, 32768, 256, 4096, 20, 0.9, 0.1, True, 8.0, 0.0),
    ("l1r3_split_oddB5", 5, 16384, 256, 4096, 40, 0.95, 0.0, False, 8.0, 0.0),
    ("l1r3_split_bv2048_k256", 4, 16384, 256, 2048, 40, 0.95, 0.0, False, 8.0, 0.0),
    ("l1r3_split_shrink_k1", 10, 20480, 256, 4096, 1, 1.0, 0.0, False, 8.0, 0.0),
    ("l1r3_split_valrange_L", 4, 16384, 256, 4096, 40, 0.95, 0.0, False, 1000.0, 0.0),
    ("l1r3_split_valrange_S", 4, 16384, 256, 4096, 40, 0.95, 0.0, False, 5.0, 0.0),
    ("l1r3_split_valrange_ASYM", 8, 16384, 256, 4096, 40, 0.95, 0.0, False, 6.0, -4.0),
    ("l1r3_split_prime_tail313", 4, 20793, 256, 4096, 20, 0.9, 0.05, True, 8.0, 0.0),
    ("l1r3_split_kmax128_bv4096", 4, 16384, 128, 4096, 100, 0.95, 0.0, False, 8.0, 0.0),
]
# V-S4 boundary recipes: (name, B, V, K_MAX, BV, k, p, mu, nmp, n_neg_inf_splits)
BND = [
    ("bnd_split0_all_neginf", 4, 20793, 256, 4096, 40, 0.95, 0.05, True, 1),
    ("bnd_split0to4_all_neginf", 4, 20793, 256, 4096, 40, 0.95, 0.05, True, 5),
    ("bnd_splitmid_all_neginf", 8, 16384, 256, 4096, 40, 0.95, 0.05, True, 2),
    ("bnd_sparse_neginf_split", 4, 20793, 256, 4096, 40, 0.95, 0.05, True, 0),
]

print("=" * 118)
print("L1 CANDIDATES (must be split-path AND valid_ref all-1)")
print("=" * 118)
good = []
for (name, B, V, kmax, bv, k, p, mu, nmp, scale, offset) in CAND:
    torch.manual_seed(0)
    logits = torch.randn(B, V, dtype=torch.float32) * scale + offset
    top_ks = torch.full((B,), k, dtype=torch.int32)
    top_ps = torch.full((B,), p, dtype=torch.float32)
    min_ps = torch.full((B,), mu, dtype=torch.float32)
    rand = torch.arange(B, dtype=torch.int32) * 7919 + 13
    m, s, tau, valid_ref, w = golden_r4(logits, top_ks, top_ps, min_ps, nmp, kmax)
    tok = golden_gumbel_sample(logits, tau, rand)
    us, S, NBS, ST = _r4_split_plan(B, V, kmax, bv)
    nbf = V // bv
    tail = V - nbf * bv
    nbt = nbf + (1 if tail > 0 else 0)
    per = [max(0, min(nbt, (x + 1) * NBS) - x * NBS) for x in range(S)]
    all1 = bool((valid_ref == 1).all().item())
    tok_ok = bool(((tok >= 0) & (tok < V)).all().item())
    fin = bool(torch.isfinite(m).all() and torch.isfinite(s).all() and (s > 0).all())
    verdict = "USE" if (us and all1 and tok_ok and fin) else "SKIP"
    if verdict == "USE":
        good.append(name)
    print(f"  {verdict:4s} {name:30s} B={B:3d} V={V:6d} K={kmax:3d} BV={bv:4d} k={k:3d} p={p} mu={mu} "
          f"nmp={int(nmp)} scale={scale:<7g} | split={us} S={S} NBS={NBS} S_TAIL={ST} per={per} "
          f"| valid_ref_all1={all1} tok_in_range={tok_ok} m/s_finite={fin}")

print()
print("=" * 118)
print("V-S4 BOUNDARY RECIPES (a whole split's V-interval forced to -inf)")
print("=" * 118)
for (name, B, V, kmax, bv, k, p, mu, nmp, n_ninf) in BND:
    us, S, NBS, ST = _r4_split_plan(B, V, kmax, bv)
    torch.manual_seed(0)
    logits = torch.randn(B, V, dtype=torch.float32) * 8.0
    if n_ninf > 0:
        logits[:, : n_ninf * NBS * bv] = float("-inf")
    else:  # sparse -inf control (never a whole split, never a whole row)
        logits[0, :50] = float("-inf")
        logits[2, 100] = float("-inf")
    ninf_frac = float(torch.isneginf(logits).float().mean().item())
    top_ks = torch.full((B,), k, dtype=torch.int32)
    top_ps = torch.full((B,), p, dtype=torch.float32)
    min_ps = torch.full((B,), mu, dtype=torch.float32)
    rand = torch.arange(B, dtype=torch.int32) * 7919 + 13
    m, s, tau, valid_ref, w = golden_r4(logits, top_ks, top_ps, min_ps, nmp, kmax)
    tok = golden_gumbel_sample(logits, tau, rand)
    nan_ct = int(torch.isnan(m).sum() + torch.isnan(s).sum() + torch.isnan(tau).sum())
    row_all_ninf = bool(torch.isneginf(logits).all(dim=-1).any().item())
    print(f"  {name:30s} split={us} S={S} NBS={NBS} S_TAIL={ST} n_ninf_splits={n_ninf} "
          f"-inf_frac={ninf_frac:.3f} row_all_-inf={row_all_ninf} "
          f"| golden: valid_ref_all1={bool((valid_ref==1).all().item())} "
          f"valid_ref_vals={sorted(set(valid_ref.tolist()))} m_finite={bool(torch.isfinite(m).all())} "
          f"s>0={bool((s>0).all())} tau_finite={bool(torch.isfinite(tau).all())} "
          f"nan_in_golden={nan_ct} tok_sample={tok[:3].tolist()} w_finite={bool(torch.isfinite(w).all())}")

print(f"\nL1 candidates to USE ({len(good)}): {good}")
