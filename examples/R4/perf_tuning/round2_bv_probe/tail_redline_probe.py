"""TAIL red-line evidence table: for every (case, BV) extract from the GENERATED Ascend C
source the real TopK args and UB->UB copy widths.

True emitted signature (verified): TopK<T>(dst, src, tmp, K, repeatTimes, actualCount)
  alignedCount = repeatTimes * 32          (common.h:1712-1713)
  padCount     = alignedCount - actualCount (common.h:1714)
  broken masked-Duplicate path taken iff padCount>0 AND actualCount % 8 != 0 (fp32
  elemPerBlock=8) -- common.h:1731-1755; the masked AscendC::Duplicate does not take
  effect on dav_c220, which is exactly the round-1 TAIL_PAD workaround target.

UB->UB: copy_ub_to_ub<T,T,N>(...) copies N elements; N*4 bytes must be a 32 B multiple
(the silent truncation-to-multiple-of-8 defect B).
"""
import json
import re
import sys

sys.path.insert(0, '/mnt/opencode')
sys.path.insert(0, '/tmp/opencode')
sys.path.insert(0, '/mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4')
from bv_static_analysis import CASES, align_up  # noqa: E402
from r4_fused_sampling import r4_topk_stats  # noqa: E402

TOPK = re.compile(r'TopK<[^>]*>\(([^;]*?)\)\s*;')
UB2UB = re.compile(r'copy_ub_to_ub<[^>]*?,\s*(\d+)\s*>\(')


def analyse_src(src):
    tk = []
    for m in TOPK.finditer(src):
        nums = [int(x) for x in re.findall(r'(?<![\w.])\d+(?![\w.])', m.group(1))]
        K, rep, actual = nums[-3], nums[-2], nums[-1]
        aligned = rep * 32
        pad = aligned - actual
        tk.append(dict(K=K, repeatTimes=rep, actualCount=actual, alignedCount=aligned,
                       padCount=pad, broken_masked_path=bool(pad > 0 and actual % 8 != 0)))
    copies = [int(m.group(1)) for m in UB2UB.finditer(src)]
    return tk, copies


out = {}
all_ok = True
for BV in (4096, 4608, 5120):
    print(f'\n================= BV = {BV} =================')
    per_case = {}
    for (cid, B, V, K, BVfix, nmp) in CASES:
        TAIL = V % BV
        TP = align_up(TAIL, 32)
        nbf = V // BV
        src = r4_topk_stats(B, V, K, BV, nmp).get_kernel_source()
        tk, copies = analyse_src(src)
        pads = sorted({t['padCount'] for t in tk})
        broken = [t for t in tk if t['broken_masked_path']]
        copy_bytes = sorted({n * 4 for n in copies})
        copy_ok = all(b % 32 == 0 for b in copy_bytes)
        c3 = (TAIL == 0) or (TAIL >= K)
        kle = all(t['K'] <= t['actualCount'] for t in tk)
        ok = (pads == [0]) and not broken and copy_ok and c3 and kle and BV % 32 == 0
        all_ok &= ok
        per_case[cid] = dict(B=B, V=V, K_MAX=K, n_blk_full=nbf, TAIL=TAIL, TAIL_PAD=TP,
                             topk_calls=tk, padCounts=pads, broken_masked_path=bool(broken),
                             ub2ub_elem_counts=sorted(set(copies)), ub2ub_bytes=copy_bytes,
                             ub2ub_all_32B_multiple=copy_ok, c3_ok=c3, K_le_actualNum=kle,
                             redline_ok=ok)
        print(f'  case{cid:>2} V={V:<7}K={K:<4} nbf={nbf:<3} TAIL={TAIL:<5} TAIL_PAD={TP:<5} '
              f'TAIL%32={TAIL%32:<3} padCounts={pads} broken={len(broken)} '
              f'ub2ub_bytes={copy_bytes} 32B_ok={copy_ok} C3={"ok" if c3 else "FAIL"} '
              f'K<=act={kle} -> {"RED-LINE OK" if ok else "*** RED-LINE VIOLATION ***"}')
    out[str(BV)] = per_case

print(f'\n=== ALL (case, BV) combos satisfy the TAIL red lines: {all_ok} ===')
json.dump(out, open('/tmp/opencode/tail_redline.json', 'w'), indent=1)
print('wrote /tmp/opencode/tail_redline.json')
