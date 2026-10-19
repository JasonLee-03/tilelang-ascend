"""R4 Stage-3 round-2 item #3: static BV feasibility analysis (no NPU needed).

Derives exact UB accounts for K1/K2m/K2g and the TAIL red-line table.
Formulas were reverse-engineered from the memory planner's emitted
`ascend_ub.GetWithOffset<T>(n, off)` layout in cached generated sources and are
cross-validated empirically by ub_probe.py.
"""
import json

UB_LIMIT = 196352          # src/transform/ascend_memory_planning.cc:35 ASCEND_SHARED_MEM_SIZE


def align_up(x, a=32):
    return ((x + a - 1) // a) * a


def aligned_topk(K, elems_per_block=8):
    return ((2 * K + elems_per_block - 1) // elems_per_block) * elems_per_block


# ---- bench 18-case matrix (cases.yaml/cases.csv attrs) ----
CASES = [
    # (case_id, B, V, K_MAX, BV_fixture, need_min_p)
    (1, 2, 256, 256, 4096, False),
    (2, 1, 4096, 256, 4096, False),
    (3, 8, 16384, 256, 4096, False),
    (4, 4, 16384, 256, 4096, False),
    (5, 7, 4409, 256, 4096, False),
    (6, 8, 10240, 256, 4096, False),
    (7, 4, 152064, 256, 4096, False),
    (8, 4, 16384, 128, 2048, False),
    (9, 64, 4096, 256, 4096, False),
    (10, 4, 4096, 256, 4096, True),
    (11, 8, 8192, 256, 4096, True),
    (12, 16, 32768, 256, 4096, False),
    (13, 6, 2304, 128, 2048, False),
    (14, 32, 1024, 256, 4096, False),
    (15, 64, 152064, 256, 4096, False),
    (16, 128, 32768, 256, 4096, False),
    (17, 16, 152377, 256, 4096, False),
    (18, 64, 65536, 256, 4096, True),
]


def ub_k1(BV, K_MAX, V):
    """K1 r4_topk_stats shared.ub bytes.

    EXACT model, validated against the memory planner's emitted
    `ascend_ub.GetWithOffset<T>(n, off)` layout (max(off+size) == the planner's
    "Scope shared.ub memory usage" figure) at 4 points:
      (BV=4096,K=256,V=10240)->188004  (BV=4096,K=256,V=152377)->163684
      (BV=2048,K=128,V=16384)->73828   (BV=4096,K=256,V=256,n_blk_full=0)->29284
    """
    TAIL = V % BV
    TAIL_PAD = align_up(TAIL, 32)
    n_blk_full = V // BV
    AT = aligned_topk(K_MAX)
    # scalars / small fixed buffers (each AlignUp(size,32)); 1-elem buffers occupy 32 B
    fixed = (7 * 32            # tau_k tau_p prefix tau_u okp tieok vflag
             + 5 * 32          # k_i p_i mu_i m_run s_run
             + 3 * 32          # m_blk m_old tmp
             + 32              # s_blk
             + 32 + 32 + 4)    # n_gt n_eq tau  (= 580 total)
    ksmall = 3 * (4 * K_MAX)   # tie_m, tie_one, run_v  (= 12*K_MAX)
    kmax_bufs = (4 * AT        # l_dst
                 + 4 * K_MAX   # l_val
                 + 4 * 2 * K_MAX  # merged
                 + 4 * AT      # m_dst
                 + 5 * 4 * K_MAX)  # rv2 mk2 q2 sk2 tie_tkbc
    tie_mask = align_up(K_MAX // 8, 32)
    if n_blk_full > 0:
        bv_bufs = 4 * (4 * BV)             # z_ub e_ub m2d zt_ub  (= 16*BV)
        topk_src = align_up(BV, 32)        # block topk src zt_ub[BV] is the widest
    else:
        bv_bufs = 0                        # guarded by `if n_blk_full > 0`
        topk_src = align_up(TAIL_PAD, 32)  # only tail topk (src ztt[TAIL_PAD]) exists
    # single shared tmp arena = max over calls: topk 4*src*4 B, merge topk 4*(2K)*4 B
    arena = max(16 * topk_src, 32 * K_MAX)
    tail_bufs = 3 * align_up(4 * TAIL, 32) + 2 * (4 * TAIL_PAD) if TAIL > 0 else 0
    return fixed + ksmall + kmax_bufs + tie_mask + bv_bufs + arena + tail_bufs


def ub_k2m(BV):
    """K2m r4_mask_sample = 24*BV + AlignUp(BV/8,32) + 96.
    EXACT: validated (BV=4096)->98912, (BV=2048)->49504."""
    return 6 * (4 * BV) + align_up(BV // 8, 32) + 3 * 32


def ub_k2g(BV):
    """K2g r4_gumbel_sample = 36*BV + AlignUp(BV/8,32) + 484.
    z_ub/idx/g_ub 3*(4BV) + state/h/tmp int32[2BV] 3*(8BV) = 36*BV; mask BV/8;
    reduce tmp arena 256 B (cols>=64 -> kVectorRepeatBytes); 8 scalars at 32 B stride
    with the last one charged at raw size 4 -> 7*32+4 = 228.
    EXACT: validated (BV=4096)->148452, (BV=2048)->74468."""
    return 3 * (4 * BV) + 3 * (4 * 2 * BV) + align_up(BV // 8, 32) + 7 * 32 + 4 + 256


def analyse(BV):
    rows = []
    for (cid, B, V, K_MAX, BV_fix, nmp) in CASES:
        TAIL = V % BV
        TAIL_PAD = align_up(TAIL, 32)
        n_blk_full = V // BV
        c3 = (TAIL == 0) or (TAIL >= K_MAX)
        c7 = V >= K_MAX
        # padCount for each topk call: alignedCount = ceil(src_buf_size/32)*32; actual = count arg
        pc_block = align_up(BV, 32) - BV if n_blk_full > 0 else None
        pc_tail = align_up(TAIL_PAD, 32) - TAIL_PAD if TAIL > 0 else None
        pc_merge = align_up(2 * K_MAX, 32) - 2 * K_MAX
        k_le_actual = (K_MAX <= BV if n_blk_full > 0 else True) and (K_MAX <= TAIL_PAD if TAIL > 0 else True) \
            and K_MAX <= 2 * K_MAX
        copies = {}
        if n_blk_full > 0:
            copies['z_ub->zt_ub'] = 4 * BV
        if TAIL > 0:
            copies['ztp->ztt'] = 4 * TAIL_PAD
        copies['run_v->merged[:K]'] = 4 * K_MAX
        copies['l_val->merged[K:]'] = 4 * K_MAX
        copies['run_v->rv2'] = 4 * K_MAX
        copy_ok = all(v % 32 == 0 for v in copies.values())
        u1, u2m, u2g = ub_k1(BV, K_MAX, V), ub_k2m(BV), ub_k2g(BV)
        rows.append(dict(case=cid, B=B, V=V, K_MAX=K_MAX, BV_fixture=BV_fix,
                         n_blk_full=n_blk_full, TAIL=TAIL, TAIL_PAD=TAIL_PAD,
                         padCount_block=pc_block, padCount_tail=pc_tail, padCount_merge=pc_merge,
                         K_le_actual_num=k_le_actual, c3_ok=c3, c7_ok=c7,
                         ub2ub_copies=copies, ub2ub_all_32B_multiple=copy_ok,
                         BV_mod32=(BV % 32 == 0),
                         ub_k1=u1, ub_k2m=u2m, ub_k2g=u2g,
                         ub_max=max(u1, u2m, u2g),
                         ub_fits=max(u1, u2m, u2g) <= UB_LIMIT,
                         binding=('K1' if u1 == max(u1, u2m, u2g) else ('K2m' if u2m == max(u1, u2m, u2g) else 'K2g'))))
    return rows


if __name__ == '__main__':
    # ---- model self-validation against empirically measured UB (generated sources) ----
    MEASURED = [
        ('K1', (4096, 256, 10240), 188004), ('K1', (4096, 256, 152377), 163684),
        ('K1', (2048, 128, 16384), 73828), ('K1', (4096, 256, 256), 29284),
        ('K2m', (4096,), 98912), ('K2m', (2048,), 49504),
        ('K2g', (4096,), 148452), ('K2g', (2048,), 74468),
    ]
    print('=== model self-validation (predicted vs measured from generated source) ===')
    allok = True
    for kn, args, meas in MEASURED:
        pred = {'K1': lambda a: ub_k1(*a), 'K2m': lambda a: ub_k2m(*a),
                'K2g': lambda a: ub_k2g(*a)}[kn](args)
        ok = pred == meas
        allok &= ok
        print(f'  {kn:4s} {str(args):22s} pred={pred:7d} meas={meas:7d} {"OK" if ok else "*** MISMATCH ***"}')
    print(f'  model exact on all {len(MEASURED)} validation points: {allok}\n')
    if not allok:
        raise SystemExit('model not exact -- refusing to draw feasibility conclusions')

    out = {}
    for BV in (4096, 4608, 5120, 6144):
        rows = analyse(BV)
        out[BV] = rows
        worst = max(r['ub_max'] for r in rows)
        nfail_ub = sum(1 for r in rows if not r['ub_fits'])
        nfail_c3 = sum(1 for r in rows if not r['c3_ok'])
        nfail_pc = sum(1 for r in rows
                       if (r['padCount_block'] not in (0, None)) or (r['padCount_tail'] not in (0, None))
                       or r['padCount_merge'] != 0)
        nfail_copy = sum(1 for r in rows if not r['ub2ub_all_32B_multiple'])
        nfail_kle = sum(1 for r in rows if not r['K_le_actual_num'])
        print(f"BV={BV}: worst_UB={worst} (limit {UB_LIMIT}) headroom={UB_LIMIT-worst} | "
              f"UB_overflow_cases={nfail_ub} C3_fail={nfail_c3} padCount!=0={nfail_pc} "
              f"copy32B_fail={nfail_copy} K>actual={nfail_kle}")
        print(f"   per-kernel worst: K1={max(r['ub_k1'] for r in rows)} "
              f"K2m={max(r['ub_k2m'] for r in rows)} K2g={max(r['ub_k2g'] for r in rows)}")
    json.dump({str(k): v for k, v in out.items()}, open('/tmp/opencode/bv_static.json', 'w'), indent=1)
    print("\nwrote /tmp/opencode/bv_static.json")
