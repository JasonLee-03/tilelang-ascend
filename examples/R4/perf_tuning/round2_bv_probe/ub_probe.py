"""Empirical UB-usage probe: compile a specialization and parse the memory planner's
emitted `ascend_ub.GetWithOffset<T>(n, off)` layout from the generated Ascend C source.

total_used = max(off + n * sizeof(T))  -- exactly what
src/transform/ascend_memory_planning.cc:582-585 computes for the DLOG line
"Scope shared.ub memory usage: X/196352 bytes".
"""
import re
import sys
import time

sys.path.insert(0, '/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4')

from r4_fused_sampling import r4_topk_stats, r4_mask_sample, r4_gumbel_sample  # noqa: E402

_SZ = {'float': 4, 'int': 4, 'uint8_t': 1, 'int32_t': 4, 'float16_t': 2,
       'uint16_t': 2, 'int16_t': 2, 'uint32_t': 4, 'int8_t': 1, 'half': 2}
_PAT = re.compile(r'ascend_ub\.GetWithOffset<\s*([A-Za-z0-9_:]+)\s*>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)')


def ub_usage(src):
    """Return (total_used, n_buffers, detail list, ub_region_size)."""
    detail = []
    for m in _PAT.finditer(src):
        dt, n, off = m.group(1), int(m.group(2)), int(m.group(3))
        key = dt.split('::')[-1]
        esz = _SZ.get(key)
        if esz is None:
            raise SystemExit(f'unknown dtype in GetWithOffset: {dt}')
        detail.append((off, n * esz, dt, n))
    if not detail:
        raise SystemExit('no ascend_ub.GetWithOffset found -- parse failure')
    total = max(o + s for o, s, _, _ in detail)
    m = re.search(r'InitBuffer\(ascend_ub,\s*(\d+)\)', src)
    region = int(m.group(1)) if m else None
    return total, len(detail), sorted(detail), region


def probe(kernel_name, fn, args, label):
    t0 = time.time()
    k = fn(*args)
    src = k.get_kernel_source()
    dt = time.time() - t0
    total, nb, detail, region = ub_usage(src)
    print(f'{label:38s} {kernel_name:16s} UB={total:7d}/{region} '
          f'({100.0*total/region:5.2f}%) headroom={region-total:7d} bufs={nb:3d} '
          f'compile={dt:6.1f}s {"CACHED" if dt < 2.0 else "NEW"}')
    return total, region, dt


if __name__ == '__main__':
    jobs = eval(sys.argv[1]) if len(sys.argv) > 1 else None
    if jobs is None:
        raise SystemExit('usage: ub_probe.py "<list of (kernel,args,label)>"')
    worst = 0
    for kn, args, label in jobs:
        fn = {'K1': r4_topk_stats, 'K2m': r4_mask_sample, 'K2g': r4_gumbel_sample}[kn]
        t, r, _ = probe(kn, fn, tuple(args), label)
        worst = max(worst, t)
        if t > r:
            print(f'   *** OVERFLOW {t} > {r} ***')
    print(f'WORST={worst}')
