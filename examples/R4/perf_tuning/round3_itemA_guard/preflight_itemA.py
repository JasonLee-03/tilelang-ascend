"""Item-A pre-flight (read-only; the kernel is NOT modified here).

Three things must be settled BEFORE `_SPLIT_A_LAUNCH_US` is touched:

  P1  18-case `_r4_split_plan` reproduction table at a sweep of a_launch values, with a
      SELF-CHECK that a=8.0 reproduces the delivered split set {3,4,7,8,12,17} exactly
      (round 3's scan script re-implemented the plan; this one calls the DELIVERED function
      with the module constant monkey-patched in-process, so it cannot diverge by construction).
  P2  CPU golden pre-verification of every shape this change proposes to ADD to the test tables
      (section 9.2.1 recipe: manual_seed(0) + randn*8.0 must yield valid_ref all-1, otherwise the
      new case would fail for a recipe reason rather than a kernel reason).
  P3  JIT specialization inventory: which (kernel, arg-tuple) pairs become newly required.
      Round 3 asserted "no new JIT specialization geometry" -- that is true for the 18 CONTRACT
      cases (S/NBS/S_TAIL are derived BEFORE the guard is read) but NOT automatically true for the
      TEST TABLES, whose entries flip too. This enumerates both, so the compile cost can be
      disclosed separately (hard constraint 7) instead of silently landing inside a timing window.

Usage: preflight_itemA.py <a_new> [out.json]
"""
import hashlib
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R4 = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (R4, os.path.join(R4, "bench", "hap_thw_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r4_fused_sampling as M                                    # noqa: E402
from r4_thw_common import load_cases                             # noqa: E402
from test_r4_fused_sampling import golden_r4, golden_gumbel_sample, LOGIT_SCALE  # noqa: E402

A_NEW = float(sys.argv[1])
A_SHIP = 8.0
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "preflight.json")
KERNEL = os.path.join(R4, "r4_fused_sampling.py")
SWEEP = [8.0, 4.0, 3.99, 3.20, 3.02, 3.01, 2.84, 2.74, 1.54, 1.0, 0.0]

# shapes this change proposes to ADD to the test tables (case 6 = cb6, case 11 = cb11)
NEW_TEST_SHAPES = [
    ("l1r3_split_cb6_B8_V10240", 8, 10240, 256, 4096, 40, 0.95, 0.0, False),
    ("l1r3_split_cb11_B8_V8192", 8, 8192, 256, 4096, 40, 0.95, 0.0, False),
    ("stab_split_cb6_B8_V10240", 8, 10240, 256, 4096, 40, 0.95, None, None),
    ("stab_split_cb11_B8_V8192", 8, 8192, 256, 4096, 40, 0.95, None, None),
]

sha_kernel = hashlib.sha256(open(KERNEL, "rb").read()).hexdigest()
print(f"[preflight] kernel sha256 = {sha_kernel}  (must stay {sha_kernel[:8]}... throughout)")
print(f"[preflight] a_shipped = {A_SHIP}   a_new = {A_NEW}\n")

# ---------------------------------------------------------------- P1: 18-case reproduction
CASES = {c["case_id"]: c for c in load_cases("r4_fused_sampling", verbose=False)}
CASES_M = {c["case_id"]: c for c in load_cases("r4_mask_sample", verbose=False)}
print(f"[P1] contract cases loaded: fused={len(CASES)} mask={len(CASES_M)}")


def plan_at(a, B, V, K, BV):
    old = M._SPLIT_A_LAUNCH_US
    M._SPLIT_A_LAUNCH_US = a
    try:
        return M._r4_split_plan(B, V, K, BV)
    finally:
        M._SPLIT_A_LAUNCH_US = old


def split_set(a):
    return sorted(ci for ci in CASES
                  if plan_at(a, CASES[ci]["B"], CASES[ci]["V"],
                             CASES[ci]["K_MAX"], CASES[ci]["BV"])[0])


DELIVERED_SET = [3, 4, 7, 8, 12, 17]
selfcheck = split_set(A_SHIP)
print(f"[P1] SELF-CHECK a=8.0 -> delivered function returns {selfcheck}")
print(f"[P1] SELF-CHECK {'OK (exact)' if selfcheck == DELIVERED_SET else '*** MISMATCH ***'} "
      f"vs round-3 delivered set {DELIVERED_SET}")

rows = []
print(f"\n{'=' * 104}")
print(f"{'case':>4} {'B':>4} {'V':>7} {'K':>4} {'BV':>5} {'b_num':>5} {'nbf':>4} {'nbt':>4} "
      f"{'S':>3} {'NBS':>4} {'S_T':>4} | {'saving':>8} {'thr@8.0':>8} {'thr@new':>8} | verdict")
print(f"{'=' * 104}")
for ci in sorted(CASES):
    c = CASES[ci]
    B, V, K, BV = c["B"], c["V"], c["K_MAX"], c["BV"]
    p0 = plan_at(A_SHIP, B, V, K, BV)
    p1 = plan_at(A_NEW, B, V, K, BV)
    b_num = -(-B // M.VEC_NUM)
    nbf = V // BV
    nbt = nbf + (1 if V - nbf * BV > 0 else 0)
    S, NBS = p0[1], p0[2]
    maxblk = max(min(nbf, max(0, (s + 1) * NBS - s * NBS)) for s in range(S)) if S >= 1 else 0
    t_sort = M._SPLIT_T_SORT_US.get((BV, K), M._SPLIT_T_SORT_DEFAULT)
    t_mrg = M._SPLIT_T_MERGE_US.get(K, M._SPLIT_T_MERGE_DEFAULT)
    saving = (nbf - maxblk) * t_sort if (S >= 2 and nbf >= 2) else float("nan")
    thr0 = M._SPLIT_SAFETY * (A_SHIP + S * t_mrg) if (S >= 2 and nbf >= 2) else float("nan")
    thr1 = M._SPLIT_SAFETY * (A_NEW + S * t_mrg) if (S >= 2 and nbf >= 2) else float("nan")
    if p0[0] and p1[0]:
        verdict = "already split (unchanged)"
    elif not p0[0] and p1[0]:
        verdict = f"*** FLIPS to SPLIT (margin {saving - thr1:+.3f} us) ***"
    elif p0[0] and not p1[0]:
        verdict = "*** REGRESSES to CURRENT ***"
    else:
        why = (f"S<2 (S_cap={max(0, M.detect_cube_core_num() // b_num)}, n_blk_total={nbt})"
               if S < 2 else f"n_blk_full={nbf}<2")
        verdict = f"blocked structurally: {why}"
    print(f"{ci:>4} {B:>4} {V:>7} {K:>4} {BV:>5} {b_num:>5} {nbf:>4} {nbt:>4} "
          f"{S:>3} {NBS:>4} {p0[3]:>4} | {saving:>8.3f} {thr0:>8.3f} {thr1:>8.3f} | {verdict}")
    rows.append(dict(case=ci, B=B, V=V, K_MAX=K, BV=BV, b_num=b_num, n_blk_full=nbf,
                     n_blk_total=nbt, S=S, NBS=NBS, S_TAIL=p0[3],
                     geom_unchanged=(p0[1:] == p1[1:]),
                     pred_saving=None if saving != saving else round(saving, 4),
                     thr_shipped=None if thr0 != thr0 else round(thr0, 4),
                     thr_new=None if thr1 != thr1 else round(thr1, 4),
                     use_split_shipped=p0[0], use_split_new=p1[0], verdict=verdict))

new_set = split_set(A_NEW)
print(f"\n[P1] split set @8.0 = {selfcheck}  (n={len(selfcheck)})")
print(f"[P1] split set @{A_NEW} = {new_set}  (n={len(new_set)})")
print(f"[P1] newly admitted = {sorted(set(new_set) - set(selfcheck))}")
print(f"[P1] lost            = {sorted(set(selfcheck) - set(new_set))}")
print(f"[P1] geometry (S,NBS,S_TAIL) unchanged for all 18 = "
      f"{all(r['geom_unchanged'] for r in rows)}")

print(f"\n[P1] sweep (delivered function, monkey-patched in-process):")
sweep = {}
for a in SWEEP:
    s = split_set(a)
    sweep[str(a)] = s
    print(f"   a={a:<5} -> n={len(s):<2} {s}")

# ---------------------------------------------------------------- P2: CPU golden pre-verify
print(f"\n{'=' * 104}\n[P2] CPU golden pre-verification of the shapes to be ADDED (section 9.2.1 recipe)\n{'=' * 104}")
p2 = []
for nm, B, V, K, BV, k, p, mu, nmp in NEW_TEST_SHAPES:
    torch.manual_seed(0)
    logits = torch.randn(B, V, dtype=torch.float32) * LOGIT_SCALE
    use_nmp = bool(nmp) if nmp is not None else False
    use_mu = float(mu) if mu is not None else 0.0
    ks = torch.full((B,), k, dtype=torch.int32)
    ps = torch.full((B,), p, dtype=torch.float32)
    mus = torch.full((B,), use_mu, dtype=torch.float32)
    m, s, tau, valid, w = golden_r4(logits, ks, ps, mus, use_nmp, K)
    all1 = bool((valid == 1).all().item())
    nan = bool(torch.isnan(s).any().item() or torch.isnan(tau).any().item())
    plan = plan_at(A_NEW, B, V, K, BV)
    print(f"  {nm:<28} B={B:<3} V={V:<6} K={K} BV={BV:<5} k={k} p={p} mu={use_mu} "
          f"min_p={use_nmp} | valid_all1={all1!s:<5} golden_nan={nan!s:<5} "
          f"finite_w={bool(torch.isfinite(w).all().item())!s:<5} | plan@new={plan}")
    p2.append(dict(name=nm, B=B, V=V, K_MAX=K, BV=BV, k=k, p=p, mu=use_mu,
                   need_min_p=use_nmp, valid_ref_all1=all1, golden_has_nan=nan,
                   plan_at_new=list(plan)))
    if not all1 or nan:
        print(f"    *** {nm} does NOT satisfy the 9.2.1 recipe -- DO NOT add it as-is ***")

# ---------------------------------------------------------------- P3: JIT specialization delta
print(f"\n{'=' * 104}\n[P3] JIT specialization inventory (newly required compile jobs)\n{'=' * 104}")


def specs_for(B, V, K, BV, nmp, fuse, a):
    """The exact jit arg-tuples the host wrapper would compile for one (shape, task)."""
    use, S, NBS, ST = plan_at(a, B, V, K, BV)
    if not use:
        out = {("K1", (B, V, K, BV, nmp))}
        out.add(("K2g", (B, V, BV)) if fuse else ("K2m", (B, V, BV)))
        return out
    out = {("K1A", (B, V, K, BV, S, NBS, ST)),
           ("K1B", (B, V, K, BV, S, nmp))}
    if fuse:
        out |= {("K2gA", (B, V, BV, S, NBS)), ("K2gB", (B, S))}
    else:
        out.add(("K2m", (B, V, BV)))
    return out


def inventory(shapes, a):
    acc = set()
    for (B, V, K, BV, nmp, fuse) in shapes:
        acc |= specs_for(B, V, K, BV, nmp, fuse, a)
    return acc


contract = []
for ci in sorted(CASES):
    c = CASES[ci]
    contract.append((c["B"], c["V"], c["K_MAX"], c["BV"], bool(c.get("need_min_p", False)), True))
for ci in sorted(CASES_M):
    c = CASES_M[ci]
    contract.append((c["B"], c["V"], c["K_MAX"], c["BV"], bool(c.get("need_min_p", False)), False))

inv_c0, inv_c1 = inventory(contract, A_SHIP), inventory(contract, A_NEW)
print(f"  CONTRACT (36 task-case pairs): shipped={len(inv_c0)} specs, new={len(inv_c1)} specs")
print(f"  newly required by the contract : {sorted(inv_c1 - inv_c0)}")
print(f"  dropped by the contract        : {sorted(inv_c0 - inv_c1)}")

# the four test-table entries that flip + the shapes being added
import test_r4_fused_sampling as TF  # noqa: E402

flip_shapes, add_shapes = [], []
for tbl, fuse_opts in ((TF.L1_CASES, (False,)), (TF.L1B_CASES, (True,)),
                       (TF.STAB_CASES, None)):
    for e in tbl:
        if tbl is TF.STAB_CASES:
            nm, B, V, K, BV, k, p, fuse, _t = e
            fl = (B, V, K, BV, False, bool(fuse))
        elif tbl is TF.L1_CASES:
            nm, B, V, K, BV, k, p, mu, nmp, _sc, _of, _t = e
            fl = (B, V, K, BV, bool(nmp), False)
        else:
            nm, B, V, K, BV, k, p, mu, nmp, _sc, _of, _sm, _t = e
            fl = (B, V, K, BV, bool(nmp), True)
        if plan_at(A_SHIP, B, V, K, BV)[0] != plan_at(A_NEW, B, V, K, BV)[0]:
            flip_shapes.append((nm, fl))
for nm, B, V, K, BV, k, p, mu, nmp in NEW_TEST_SHAPES:
    for fuse in ((False, True) if fuse is None else (fuse,)):
        add_shapes.append((nm + ("_gumbel" if fuse else "_mask"),
                           (B, V, K, BV, bool(nmp), bool(fuse))))

inv_t0 = inventory([s for _n, s in flip_shapes], A_SHIP)
inv_t1 = inventory([s for _n, s in flip_shapes], A_NEW)
inv_a1 = inventory([s for _n, s in add_shapes], A_NEW)
print(f"\n  TEST-TABLE entries that FLIP ({len(flip_shapes)}):")
for nm, s in flip_shapes:
    print(f"     {nm:<34} B={s[0]:<3} V={s[1]:<6} K={s[2]} BV={s[3]:<5} fuse={s[5]} "
          f"plan@ship={plan_at(A_SHIP, s[0], s[1], s[2], s[3])} "
          f"plan@new={plan_at(A_NEW, s[0], s[1], s[2], s[3])}")
print(f"  newly required by the flips    : {sorted(inv_t1 - inv_t0)}")
print(f"\n  TEST-TABLE entries being ADDED ({len(add_shapes)}):")
for nm, s in add_shapes:
    print(f"     {nm:<34} B={s[0]:<3} V={s[1]:<6} K={s[2]} BV={s[3]:<5} fuse={s[5]} "
          f"plan@new={plan_at(A_NEW, s[0], s[1], s[2], s[3])}")
print(f"  newly required by the additions: {sorted(inv_a1 - inv_t0 - inv_c0)}")

allnew = sorted((inv_c1 - inv_c0) | (inv_t1 - inv_t0) | (inv_a1 - inv_t0 - inv_c0))
print(f"\n  TOTAL newly required JIT specs = {len(allnew)}")
for t in allnew:
    print(f"     {t[0]:<5} {t[1]}")

json.dump(dict(
    _meta=dict(kernel_sha256=sha_kernel, a_shipped=A_SHIP, a_new=A_NEW,
               selfcheck_ok=(selfcheck == DELIVERED_SET),
               note="P1 calls the DELIVERED _r4_split_plan with the module constant patched "
                    "in-process, so it cannot diverge from the shipped logic by construction."),
    P1_split_set_shipped=selfcheck, P1_split_set_new=new_set,
    P1_newly_admitted=sorted(set(new_set) - set(selfcheck)),
    P1_sweep=sweep, P1_cases=rows,
    P2_golden=p2,
    P3_contract_new=sorted(map(str, inv_c1 - inv_c0)),
    P3_flips=[dict(name=n, shape=list(s)) for n, s in flip_shapes],
    P3_flips_new=sorted(map(str, inv_t1 - inv_t0)),
    P3_additions=[dict(name=n, shape=list(s)) for n, s in add_shapes],
    P3_total_new=sorted(map(str, allnew)),
), open(OUT, "w"), indent=1)
print(f"\n[preflight] wrote {OUT}")
print(f"[preflight] kernel sha256 AFTER (must be unchanged) = "
      f"{hashlib.sha256(open(KERNEL, 'rb').read()).hexdigest()}")
