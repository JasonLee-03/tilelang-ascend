"""Empirical K1/K2m/K2g UB sweep across all 18 bench cases x candidate BV.

The analytic model (bv_static_analysis.py) is EXACT when the memory planner does no
liveness reuse (validated at BV=2048/4096 and for TAIL==0 at 4608/5120), but it is only
an UPPER BOUND once reuse kicks in (measured 192100 vs predicted 235108 for
K1 BV=4608 V=8192). Feasibility must therefore be decided by MEASUREMENT, not model.
"""
import re
import sys

sys.path.insert(0, '/tmp/opencode')
sys.path.insert(0, '/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4')
from ub_probe import ub_usage          # noqa: E402
from bv_static_analysis import CASES, UB_LIMIT, ub_k1, ub_k2m, ub_k2g, align_up  # noqa: E402
from r4_fused_sampling import r4_topk_stats, r4_mask_sample, r4_gumbel_sample  # noqa: E402

_FATAL = re.compile(r'Memory allocation failed for:.*?available: -?\d+', re.S)


def measure(fn, args):
    """Return (ub_bytes, region, err) -- err None on success."""
    try:
        src = fn(*args).get_kernel_source()
    except Exception as e:                                    # noqa: BLE001
        m = _FATAL.search(str(e))
        return None, None, (m.group(0).replace('\n', ' | ') if m else f'{type(e).__name__}: {str(e)[:200]}')
    tot, nb, det, reg = ub_usage(src)
    return tot, reg, None


BVS = [int(x) for x in sys.argv[1:]] or [4608, 5120]
results = {}
for BV in BVS:
    print(f'\n================ BV = {BV} ================')
    u2m, reg, err = measure(r4_mask_sample, (8, 10240, BV))
    print(f'  K2m (BV-only)      : {u2m if u2m else "FAIL"} / {reg}   model={ub_k2m(BV)}   err={err}')
    u2g, reg2, err2 = measure(r4_gumbel_sample, (8, 10240, BV))
    print(f'  K2g (BV-only)      : {u2g if u2g else "FAIL"} / {reg2}   model={ub_k2g(BV)}   err={err2}')
    rows = []
    for (cid, B, V, K, BVfix, nmp) in CASES:
        u1, reg1, e1 = measure(r4_topk_stats, (B, V, K, BV, nmp))
        TAIL = V % BV
        TP = align_up(TAIL, 32)
        c3 = (TAIL == 0) or (TAIL >= K)
        model = ub_k1(BV, K, V)
        vals = [x for x in (u1, u2m, u2g) if x]
        worst = max(vals) if vals else None
        ok = (u1 is not None and u2m is not None and u2g is not None
              and all(v <= UB_LIMIT for v in (u1, u2m, u2g)) and c3)
        rows.append(dict(case=cid, B=B, V=V, K=K, TAIL=TAIL, TAIL_PAD=TP, c3=c3,
                         ub_k1=u1, ub_model=model, ub_k2m=u2m, ub_k2g=u2g,
                         err=e1 or err or err2, ok=ok))
        print(f'  case{cid:>2} B={B:<4}V={V:<7}K={K:<4} TAIL={TAIL:<5} TAIL_PAD={TP:<5} '
              f'K1={str(u1):>7} (model {model:>7}) K2m={str(u2m):>7} K2g={str(u2g):>7} '
              f'C3={"ok" if c3 else "FAIL"} -> {"FEASIBLE" if ok else "INFEASIBLE"}'
              + (f'  [{e1}]' if e1 else ''))
    results[BV] = dict(k2m=u2m, k2g=u2g, k2m_err=err, k2g_err=err2, rows=rows,
                       n_feasible=sum(1 for r in rows if r['ok']),
                       worst=max([r['ub_k1'] for r in rows if r['ub_k1']] or [0]))
    print(f'  --> BV={BV}: {results[BV]["n_feasible"]}/18 cases feasible; '
          f'worst measured K1 UB = {results[BV]["worst"]}/{UB_LIMIT}')

import json
json.dump({str(k): v for k, v in results.items()}, open('/tmp/opencode/bv_ub_measured.json', 'w'), indent=1)
print('\nwrote /tmp/opencode/bv_ub_measured.json')
