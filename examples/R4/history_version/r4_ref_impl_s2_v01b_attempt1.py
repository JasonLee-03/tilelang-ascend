"""Golden reference for R4 fused sampling chain (V01-a).

Masking semantics replicate sglang sampler.py:846-879 (Path 3) line-by-line, plus
reverse-derivation of R4-specific tau (theorem T1) and valid (D-GUARD / D-TIE).
Computed in float64 so golden rounding noise stays well below the fp32 tolerance.

Source of truth (verified): sglang/python/sglang/srt/layers/sampler.py:846-879;
TOP_K_ALL = 1<<30 from sglang/python/sglang/srt/sampling/sampling_params.py:40.
See DESIGN.md section 9.1.
"""

import torch

TOP_K_ALL = 1 << 30  # sglang/srt/sampling/sampling_params.py:40


def golden_r4(logits, top_ks, top_ps, min_ps, need_min_p_sampling, K_MAX=256, top_k_all=TOP_K_ALL):
    """R4 fused sampling chain PyTorch reference.

    Returns: (m_ref, s_ref, tau_ref, valid_ref, w_ref)
             float32 [B], float32 [B], float32 [B], int32 [B], float32 [B,V]
    """
    B, V = logits.shape
    NEG_INF = float("-inf")
    z = logits.detach().to(torch.float64).cpu()
    ks = top_ks.detach().to(torch.int64).cpu().clone()
    ps = top_ps.detach().to(torch.float64).cpu()
    mus = min_ps.detach().to(torch.float64).cpu()

    # ---- Step 2: softmax stats (explicit form of sampler.py:847 torch.softmax) ----
    m_ref = z.max(dim=-1).values  # [B]
    e = torch.exp(z - m_ref.unsqueeze(-1))
    s_ref = e.sum(dim=-1)  # [B] = sum exp(z-m)
    probs = e / s_ref.unsqueeze(-1)  # [B,V] == softmax(z,-1)

    # ---- Step 3: top-k truncation (sampler.py:848,851-856, by rank) ----
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)  # :848
    ks_eff = torch.where(ks == top_k_all, torch.full_like(ks, V), ks)  # :851-852
    rank = torch.arange(V, dtype=torch.int64).unsqueeze(0)  # :853-855

    # RAW (pre-top-k) strict prefix sum, captured before probs_sort is rebound below.
    # The reference applies top-p AFTER top-k, but for every rank l < k the masked
    # strict prefix equals the raw one, so theorem T1's tau_p is the RAW nucleus
    # threshold z_sort[L_raw]. Deriving L from the masked q instead would count the
    # top-k zeroed tail as "kept" whenever sum(top-k probs) <= p, inflating L to V-1
    # and falsely firing the ok_p guard (see debug_log.md, L0 attempt 1).
    probs_sort_raw = probs_sort
    excl_raw = probs_sort_raw.cumsum(dim=-1) - probs_sort_raw
    kept_raw = excl_raw <= ps.unsqueeze(-1)

    q = probs_sort.masked_fill(rank >= ks_eff.unsqueeze(-1), 0.0)  # :856

    # ---- Step 4: top-p (nucleus) truncation (sampler.py:858-860, strict prefix > p) ----
    excl_q = q.cumsum(dim=-1) - q  # :858-859 strict prefix (excl self)
    kept_p = excl_q <= ps.unsqueeze(-1)  # :859 (strict > masks)
    probs_sort = q.masked_fill(~kept_p, 0.0)  # :860

    # ---- Step 5: min-p truncation (sampler.py:862-865, strict < masks) ----
    if need_min_p_sampling:
        thr = probs_sort[:, 0] * mus  # :863 (rank0 kept => max prob)
        probs_sort = probs_sort.masked_fill(probs_sort < thr.unsqueeze(-1), 0.0)  # :864-865

    # ---- Step 6: no renormalization (D-NORENORM; sampler.py:893-894) ----
    w_ref = torch.zeros_like(probs)
    w_ref.scatter_(1, probs_idx, probs_sort)

    # ---- reverse-derive tau_ref (theorem T1: tau = max(tau_k, tau_p, tau_mu)) ----
    z_sort = z.sort(dim=-1, descending=True).values  # [B,V]

    # tau_k: top-k threshold = k_i-th largest logit
    k_ok = (ks_eff >= 1) & (ks_eff <= K_MAX)
    idx_k = (ks_eff.clamp(min=1, max=K_MAX) - 1).unsqueeze(-1)
    tau_k = torch.where(k_ok.unsqueeze(-1), z_sort.gather(1, idx_k), torch.full((B, 1), NEG_INF, dtype=torch.float64)).squeeze(-1)

    # tau_mu (theorem T2): mu<=0 or !need_min_p -> -inf
    mu_active = (mus > 0.0) if need_min_p_sampling else torch.zeros_like(mus, dtype=torch.bool)
    tau_u = torch.where(mu_active, m_ref + torch.log(mus.clamp(min=1e-30)), torch.full((B,), NEG_INF, dtype=torch.float64))

    # tau_p: z value at the last rank L with RAW excl-prefix <= p. This is the same
    # criterion the kernel's monotone-guard scalar loop applies to the top-K_MAX list.
    L = kept_raw.to(torch.int64).sum(dim=-1) - 1  # prefix-monotone => count-1
    L_safe = L.clamp(min=0, max=V - 1).unsqueeze(-1)
    tau_p = torch.where((L >= 0).unsqueeze(-1), z_sort.gather(1, L_safe), torch.full((B, 1), NEG_INF, dtype=torch.float64)).squeeze(-1)

    tau_ref = torch.maximum(torch.maximum(tau_k, tau_p), tau_u)  # theorem T1

    # ---- valid_ref (D-GUARD + D-TIE; one-to-one with kernel criteria) ----
    ok_p = (ps >= 1.0) | (L < K_MAX)
    n_gt = (z > tau_k.unsqueeze(-1)).sum(dim=-1)
    n_eq = (z == tau_k.unsqueeze(-1)).sum(dim=-1)
    tie_ok = (tau_ref > tau_k) | ((n_gt + n_eq) <= ks_eff)
    valid_ref = (k_ok & ok_p & tie_ok).to(torch.int32)

    return (m_ref.to(torch.float32), s_ref.to(torch.float32), tau_ref.to(torch.float32), valid_ref, w_ref.to(torch.float32))


# ============================================================================
# V01-b: Gumbel-max fused-sampling goldens (DESIGN_v01b.md section 9.1).
# golden_r4 above is UNCHANGED (V01-a). These are additive.
# ============================================================================
import ctypes  # noqa: E402

import numpy as np  # noqa: E402

LCG_A = 1103515245  # random_1d.py:20 (classic ANSI-C LCG multiplier)
LCG_C = 12345  # random_1d.py:21
_INV2_32 = 2.0**-32  # 2^-32 (fp32-exact, power of two) -- D-UMAP
_UMIN = 2.0**-24  # 5.9604644775390625e-08 (fp32-exact) -- D-CLAMP lower
_UMAX = 1.0 - 2.0**-24  # 0.9999999403953552 (fp32-exact, largest fp32 < 1.0) -- D-CLAMP upper


def _wrap_int32(x):
    """Wrap an int64 array/scalar into signed int32 (== ctypes.c_int32 semantics).

    (x + 2^31) % 2^32 - 2^31 is a no-op for x already in [-2^31, 2^31), so it is
    safe to apply unconditionally; it makes golden_lcg bitwise-exact vs the kernel's
    int32 arith_progression + mul/add for ANY seed (not only seed+V < 2^31).
    """
    return (x + 2**31) % 2**32 - 2**31


def golden_lcg(M, seed):
    """Exact LCG reference (== random_1d.reference_random_1d(M, seed)); int32 [M].

    state_j = seed + j (mod 2^32, signed); then 3 rounds state = A*state + C (mod 2^32).
    Vectorized in int64 with explicit wrap -> bitwise identical to the ctypes.c_int32
    loop AND to the kernel's int32 arith_progression + T.tile.mul/add (validated V-b1).
    """
    idx = np.arange(M, dtype=np.int64)
    state = _wrap_int32(int(seed) + idx)  # seed + j as signed int32
    for _ in range(3):
        state = _wrap_int32(LCG_A * state + LCG_C)  # int32 wraparound, 3 rounds
    return torch.from_numpy(state.astype(np.int32))


def golden_lcg_reference_ctypes(M, seed):
    """Literal ctypes port of random_1d.reference_random_1d -- cross-check for golden_lcg."""
    out = np.zeros(M, dtype=np.int32)
    for i in range(M):
        s = seed + i
        s = ctypes.c_int32(s * LCG_A + LCG_C).value
        s = ctypes.c_int32(s * LCG_A + LCG_C).value
        s = ctypes.c_int32(s * LCG_A + LCG_C).value
        out[i] = s
    return torch.from_numpy(out)


def golden_lcg_batch(seeds, V):
    """Batched LCG: seeds [B] (int) -> int32 [B, V]; per-row semantics == golden_lcg."""
    seeds_np = np.asarray(seeds, dtype=np.int64).reshape(-1, 1)  # [B,1]
    idx = np.arange(V, dtype=np.int64).reshape(1, -1)  # [1,V]
    state = _wrap_int32(seeds_np + idx)  # [B,V] seed_i + j
    for _ in range(3):
        state = _wrap_int32(LCG_A * state + LCG_C)
    return torch.from_numpy(state.astype(np.int32))


def golden_gumbel_u_G(logits_shape_seeds, V):
    """Return (u, G) fp32 [B,V] mirroring the kernel's Gumbel chain (V-b2/V-b3 diagnostic).

    logits_shape_seeds: [B] int seeds. u = clamp(fp32(state)*2^-32 + 0.5, 2^-24, 1-2^-24);
    G = -ln(-ln(u)). fp32 (mirror kernel), host torch.log (software fp32; may differ ~1 ULP
    from Ascend T.tile.ln -> R-GUMBEL-LN, quantified by V-b2).
    """
    state = golden_lcg_batch(np.asarray(logits_shape_seeds, dtype=np.int64), V).to(torch.float32)
    u = (state * _INV2_32 + 0.5).clamp(_UMIN, _UMAX)  # fp32 affine + clamp (mirror kernel order)
    G = -torch.log(-torch.log(u))  # fp32 two logs (mirror kernel T.tile.ln x2 + neg x2)
    return u, G


def golden_gumbel_sample(logits, tau, rand):
    """Faithful Gumbel-max reference mirroring the kernel's fp32 arithmetic (bitwise main judge).

    Returns tok_ref[B] int32. Operation order mirrors DESIGN_v01b.md section 3.3 exactly:
      LCG(int32) -> cast fp32 -> *2^-32 -> +0.5 -> clamp[2^-24,1-2^-24]
      -> ln -> neg -> ln -> neg (=G) -> cand = where(z>=tau, z+G, -inf) -> argmax(first).
    Gumbel part is fp32 (mirror kernel), NOT float64. tau is K1's tau_out (isolate K2-Gumbel)
    or golden_r4's tau_ref (end-to-end) -- V01-a proved the two are bitwise-consistent.
    argmax tie-break = first (smallest index) == kernel's strict-GT cross-block merge.
    """
    z = logits.detach().to(torch.float32).cpu()
    tau_c = tau.detach().to(torch.float32).cpu()
    rand_np = rand.detach().to(torch.int64).cpu().numpy()
    B, V = z.shape
    state = golden_lcg_batch(rand_np, V).to(torch.float32)  # int32 -> fp32 (mirror kernel cast)
    u = (state * _INV2_32 + 0.5).clamp(_UMIN, _UMAX)  # D-UMAP + D-CLAMP
    G = -torch.log(-torch.log(u))  # Gumbel(0,1)
    cand = torch.where(z >= tau_c.unsqueeze(-1), z + G, torch.full_like(z, float("-inf")))
    return torch.argmax(cand, dim=-1).to(torch.int32)
