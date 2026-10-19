"""Scan the whole BV space (multiples of 32) for feasibility across all 18 bench cases."""
import sys
sys.path.insert(0, '/tmp/opencode')
from bv_static_analysis import CASES, UB_LIMIT, ub_k1, ub_k2m, ub_k2g, align_up

print(f'UB limit = {UB_LIMIT} B (src/transform/ascend_memory_planning.cc:35)\n')
print(f'{"BV":>6} {"K1worst":>8} {"K2m":>7} {"K2g":>7} {"UBfail":>7} {"C3fail":>7} {"K>act":>6} {"verdict":>10}')
feasible = []
for BV in range(1024, 8224, 32):
    if BV % 32:
        continue
    k1w = k2mw = k2gw = 0
    ubfail = c3fail = klefail = 0
    for (cid, B, V, K, BVfix, nmp) in CASES:
        u1 = ub_k1(BV, K, V)
        k1w = max(k1w, u1)
        k2mw = max(k2mw, ub_k2m(BV))
        k2gw = max(k2gw, ub_k2g(BV))
        if max(u1, ub_k2m(BV), ub_k2g(BV)) > UB_LIMIT:
            ubfail += 1
        TAIL = V % BV
        if TAIL != 0 and TAIL < K:
            c3fail += 1
        n_blk = V // BV
        TP = align_up(TAIL, 32)
        if (n_blk > 0 and K > BV) or (TAIL > 0 and K > TP) or K > 2 * K:
            klefail += 1
    ok = (ubfail == 0 and c3fail == 0 and klefail == 0)
    if ok:
        feasible.append((BV, k1w, k2mw, k2gw))
    if BV >= 3072 or ok:
        print(f'{BV:6d} {k1w:8d} {k2mw:7d} {k2gw:7d} {ubfail:7d} {c3fail:7d} {klefail:6d} '
              f'{"FEASIBLE" if ok else "infeasible":>10}')

print('\n=== FEASIBLE BV values (all 18 cases, all 3 kernels within UB) ===')
print('  ' + ', '.join(str(b[0]) for b in feasible))
print('\n  max feasible BV =', max(b[0] for b in feasible) if feasible else None)
print('  UB headroom at each feasible BV >= 4096:')
for BV, k1w, k2mw, k2gw in feasible:
    if BV >= 4096:
        w = max(k1w, k2mw, k2gw)
        print(f'    BV={BV:5d} worst={w:7d} headroom={UB_LIMIT-w:7d} ({100.0*w/UB_LIMIT:.2f}% used)')
