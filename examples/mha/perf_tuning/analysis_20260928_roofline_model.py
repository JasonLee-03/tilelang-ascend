#!/usr/bin/env python3
"""MHA rev4 (mha.py ee5bf9c9) theoretical-performance model — v5 FINAL.

Layers:
  A   ideal flash kernel (no GM workspace), best of two tilings, sustained rates
  A*  same at cube issue peak (absolute hw ceiling)
  B   CURRENT architecture+tiling+instruction structure, PERFECT pipelining
  now measured (cann-bench cann_performance_eval_20260922_172117, baseline=FusedInferAttentionScore)

Evidence-sourced constants (citations):
  chip      Ascend910_9362, 20 AIC + 40 AIV, 1650 MHz  [torch_npu.npu.get_device_properties;
            msprof_case9_20260923/op_device/OpBasicInfo.csv Current/Rated Freq=1650]
  L2 168MB  [same props]  UB192K/L1 512K/L0A,B 64K/L0C 128K [TBE get_soc_spec Ascend910B4]
  cube peak 4096 fp16 MAC/cy/core [TBE cube_m/k/n_size = 16,16,16] -> 270.3 TFLOPS (20 cores)
  cube sust 3069 MAC/cy [case9 real: 19.33G padded MAC / (20 x 0.202 cube_ratio x 944.74us) / 1.65GHz;
            camodel instr-level MMAD 149cy per [32,128]x[128,128] = 3518 MAC/cy issue rate]
            -> 202.5 TFLOPS
  HBM       read 1472 / write 1481 / mixed 1250 GB/s [torch_npu 1GB microbench, 2026-09-28:
            max() 0.729ms, fill() 0.725ms, copy 1.718ms; baseline case15 independently implies
            >=1458 GB/s (503MB/345.29us) - cross-validates]
  L2 agg    >= 4.0 TB/s [40MB copy microbench 21us = launch-overhead-contaminated LOWER bound;
            case9 sustains 2230MB/944.74us = 2.36 TB/s aggregate GM access in production]
  per-core pipe sustained-busy rates [case9 real msprof op-mode Memory.csv+PipeUtilization block0]:
    C.MTE2 125.2 GB/s (48% of 264 implied peak) | C.MTE1 191.3 | C.FIX 64.1 (26% of ~244 peak)
    V.MTE2 56.8 | V.MTE3 30.6 GB/s per AIV (group=2 AIV)
    V.vec 1.058 us/(iter*group) @ 12288 fp32 elems (as-is fragmented code, linear elem scaling)
  VEC_PEAK  40 AIV x 64 fp32 lanes x 1.65GHz = 4.22 Telu/s [2048-bit AIV ASSUMPTION; exp on MUFU
            ~1/4 rate; verified non-binding in all 20 cases]
  FLOOR     9 us launch+ramp [calibrated: case20 work~6us vs meas 14.5; case1 work~2 vs 10.3]
  batching  single_kv_block protocol batches NUM_STAGES=8 tasks/round [mha.py:136,1041-1044]
            -> task-boundary ramp amortized per 8-task batch, not per task.
"""
import statistics as st

FREQ = 1.65e9
NAIC = 20
CUBE_PEAK, CUBE_SUST = 4096.0, 3069.0
TF_PEAK = NAIC * CUBE_PEAK * 2 * FREQ / 1e12
TF_SUST = NAIC * CUBE_SUST * 2 * FREQ / 1e12
HBM_RD, HBM_MIX = 1472e9, 1250e9
L2_AGG = 4000e9
VEC_PEAK = 40 * 64 * FREQ
CM2, CM1, CFIX = 125.2e9, 191.3e9, 64.1e9
VM2, VM3 = 56.8e9, 30.6e9
VEC_IT9 = 1.058e-6
L2_SIZE = 168e6
FLOOR = 9.0e-6
NUM_STAGES = 8

CASES = [
    (1,  2, 128, 128,  8,  64, 2, 1,  10.72,  10.0),
    (2,  4, 256, 256, 16,  64, 2, 1,  67.92,  32.8),
    (3,  2, 512, 512, 32, 128, 2, 1, 170.10, 131.1),
    (4,  2, 256, 256,  8, 256, 2, 1,  36.48,  32.8),
    (5,  2, 128, 256, 12,  64, 2, 0,  18.88,  10.5),
    (6,  2, 512, 128, 16, 128, 2, 0,  25.90,  32.8),
    (7,  4,1024,1024, 32, 128, 2, 1, 936.18, 432.83),
    (8,  4,1024,1024, 32, 128, 2, 1, 922.50, 429.63),
    (9,  4,1024,1024, 16, 256, 2, 1, 957.74, 337.63),
    (10, 1,   1,2048, 32, 128, 2, 0,  53.94,  53.44),
    (11,16,   1,1024, 32, 128, 2, 0, 309.28, 285.67),
    (12,32,   1, 512, 32, 128, 2, 0, 306.30, 272.27),
    (13,60,   1, 512, 32, 128, 2, 0, 562.60, 494.09),
    (14,128,  1, 128, 32, 128, 2, 0, 377.38, 527.55),
    (15,60,   1, 512, 16, 256, 2, 0, 504.02, 345.29),
    (16,60,   2, 512, 32, 128, 2, 1, 564.82, 439.39),
    (17,96,   2, 256, 32, 128, 2, 1, 494.66, 564.31),
    (18,60,   2, 512, 16, 256, 2, 0, 504.48, 408.79),
    (19, 2, 128, 128, 32, 128, 2, 1,  23.28,  32.8),
    (20, 2, 128, 128, 32, 128, 2, 0,  15.44,  32.8),
]

def tiling(S, Skv, D):
    bM = 16 if S <= 16 else (32 if D == 256 else 64)
    if S <= 16:
        bN = 512 if (D <= 128 and Skv > 256) else (256 if Skv > 128 else 128)
    else:
        bN = 128
    form = 'A' if (S > 16 and D <= 128) else ('Bmma' if S > 16 else 'Bgv0')
    s_blocks = (S + bM - 1) // bM
    kv_iters = (Skv + bN - 1) // bN
    return bM, bN, form, s_blocks, kv_iters, kv_iters == 1

def ideal_candidates(S, Skv, D):
    """Two ideal-kernel tiling candidates; Layer A takes the min-time one.
    big-tile: bM=128,bN=256 (D<=128) / bM=64 (D=256, UB budget) - cuts KV reload ~2x;
    cur-tile: current bM,bN - better causal padding efficiency (0.89 vs 0.80)."""
    cur_bM, cur_bN, *_ = tiling(S, Skv, D)
    if S <= 16:
        return [(cur_bM, cur_bN)]
    big = (64, 128) if D == 256 else (128, 256)
    cands = [big]
    if (cur_bM, cur_bN) != big:
        cands.append((cur_bM, cur_bN))
    return cands

def causal_area(S, Skv):
    off = Skv - S
    return sum(min(r + 1 + off, Skv) for r in range(S))

def eff_iters(bx, bM, bN, S, Skv, causal):
    kv = (Skv + bN - 1) // bN
    if not causal:
        return kv
    off = Skv - S
    return min(kv, (min(bx * bM + bM - 1 + off, Skv - 1)) // bN + 1)

def hbm_time(rd, wr):
    return (rd + wr) / HBM_MIX if wr / max(rd + wr, 1) > 0.10 else rd / HBM_RD

def layerA(B, S, Skv, H, D, db, causal, bM, bN, tf, vec_ops, uniq_rd, uniq_wr, kv_b):
    sb = (S + bM - 1) // bM
    iters = B * H * sum(eff_iters(bx, bM, bN, S, Skv, causal) for bx in range(sb))
    mac_pad = iters * 2 * bM * bN * D
    reload = iters * 2 * bN * D * db
    t_cube = mac_pad * 2 / (tf * 1e12)
    extra = max(reload - kv_b, 0)
    t_hbm = hbm_time(uniq_rd + (extra * 0.5 if kv_b > L2_SIZE else 0), uniq_wr)
    t_l2 = reload / L2_AGG
    t_mte1 = reload / (CM1 * NAIC)
    t_vec = vec_ops / VEC_PEAK
    t = max(t_cube, t_hbm, t_l2, t_mte1, t_vec, FLOOR)
    bnd = max([('cube', t_cube), ('hbm', t_hbm), ('l2', t_l2), ('mte1', t_mte1),
               ('vec', t_vec), ('floor', FLOOR)], key=lambda x: x[1])[0]
    return t, bnd, iters, mac_pad * 2 / 1e9, reload / 1e6

def model(case):
    cid, B, S, Skv, H, D, db, causal, ours, base = case
    bM, bN, form, s_blocks, kv_iters, single = tiling(S, Skv, D)
    tasks = B * H * s_blocks
    cores = min(tasks, NAIC)
    it_tb = [eff_iters(bx, bM, bN, S, Skv, causal) for bx in range(s_blocks)]
    total_iters = B * H * sum(it_tb)
    q_t, r_t = tasks // cores, tasks % cores
    my_cnt = q_t + (1 if r_t > 0 else 0)
    max_core_iters = sum(it_tb[t % s_blocks] for t in range(my_cnt))
    mac_pad = total_iters * 2 * bM * bN * D
    area = causal_area(S, Skv) if causal else S * Skv
    flops_eff = 4 * B * H * area * D
    q_b = B * S * H * D * db; kv_b = 2 * B * Skv * H * D * db; o_b = q_b
    uniq_rd = q_b + kv_b; uniq_wr = o_b
    kv_reload = total_iters * 2 * bN * D * db
    extra_reload = max(kv_reload - kv_b, 0)
    fa_single = (form == 'A' and single)
    m2_it = (2 * bN * D + bM * bN) * db
    m1_it = m2_it
    fix_it = bM * bN * 4 + (bM * D * db if fa_single else bM * D * 4)
    vm2_it = bM * bN * 4 + (0 if fa_single else bM * D * 4)
    vm3_it = bM * bN * db
    q_task = bM * D * db; o_task = 0 if fa_single else bM * D * db
    gm_tot = total_iters * (m2_it + fix_it + vm2_it + vm3_it) + tasks * (q_task + o_task)
    vec_ops = 7 * B * H * area + 3 * B * S * H * D
    # ---- Layer A: ideal kernel, best tiling, sustained & peak cube ----
    cands = [layerA(B, S, Skv, H, D, db, causal, t[0], t[1], TF_SUST, vec_ops,
                    uniq_rd, uniq_wr, kv_b) for t in ideal_candidates(S, Skv, D)]
    best = min(cands, key=lambda x: x[0])
    tA, boundA, i_iters, i_gf, i_rel = best
    cands_pk = [layerA(B, S, Skv, H, D, db, causal, t[0], t[1], TF_PEAK, vec_ops,
                       uniq_rd, uniq_wr, kv_b) for t in ideal_candidates(S, Skv, D)]
    tApk = min(c[0] for c in cands_pk)
    # ---- Layer B: current arch, perfect pipelining ----
    c_m2 = m2_it / CM2; c_m1 = m1_it / CM1
    c_cub = (2 * bM * bN * D) / (CUBE_SUST * FREQ)
    c_fix = fix_it / CFIX
    v_m2 = vm2_it / (2 * VM2); v_m3 = vm3_it / (2 * VM3)
    v_elems = bM * bN + (0 if fa_single else bM * D)
    v_vec = VEC_IT9 * v_elems / 12288
    if form == 'Bgv0':
        c_pipe = max(c_m2, c_fix, c_cub + c_m1, v_m2, v_m3, v_vec)
        c_pipe_nv = max(c_m2, c_fix, c_cub + c_m1, v_m2, v_m3)
    else:
        c_pipe = max(c_m2, c_m1, c_cub, c_fix, v_m2, v_m3, v_vec)
        c_pipe_nv = max(c_m2, c_m1, c_cub, c_fix, v_m2, v_m3)
    t_task = q_task / CM2 + bM * D * db / CFIX + 2 * c_pipe
    if single:   # task-batched protocol: ramp per 8-task batch
        ramp = -(-my_cnt // NUM_STAGES) * t_task * 0.5
    else:        # generic multi-kv: ramp per task
        ramp = my_cnt * t_task * 0.5
    tB_cores = max_core_iters * c_pipe + ramp
    tB_cores_nv = max_core_iters * c_pipe_nv + ramp
    tB_hbm = hbm_time(uniq_rd + (extra_reload * 0.5 if kv_b > L2_SIZE else 0), uniq_wr)
    tB_l2 = gm_tot / L2_AGG
    tB = max(tB_cores, tB_hbm, tB_l2, FLOOR)
    tB_nv = max(tB_cores_nv, tB_hbm, tB_l2, FLOOR)
    boundB = max([('pipe', tB_cores), ('hbm', tB_hbm), ('l2', tB_l2), ('floor', FLOOR)], key=lambda x: x[1])[0]
    pipebnd = max([('C.mte2', c_m2), ('C.mte1', c_m1), ('C.cube', c_cub), ('C.fix', c_fix),
                   ('V.mte2', v_m2), ('V.mte3', v_m3), ('V.vec', v_vec)], key=lambda x: x[1])[0]
    eff_hbm_now = kv_b / (ours * 1e-6) / 1e9 if S <= 16 else 0
    return dict(cid=cid, form=form, bM=bM, bN=bN, single=single, tasks=tasks, cores=cores,
                iters=total_iters, i_iters=i_iters, imb=max_core_iters / (total_iters / cores),
                gf_pad=mac_pad * 2 / 1e9, gf_ipad=i_gf, gf_eff=flops_eff / 1e9,
                pad_eff=flops_eff / (mac_pad * 2), uniq_mb=(uniq_rd + uniq_wr) / 1e6,
                gm_mb=gm_tot / 1e6, rel_mb=kv_reload / 1e6, i_rel=i_rel,
                tA=tA * 1e6, tApk=tApk * 1e6, boundA=boundA,
                tB=tB * 1e6, tBnv=tB_nv * 1e6, tB_pipe=tB_cores * 1e6,
                tB_hbm=tB_hbm * 1e6, tB_l2=tB_l2 * 1e6,
                cpipe_us=c_pipe * 1e6, pipebnd=pipebnd, boundB=boundB,
                eff_hbm_now=eff_hbm_now, ours=ours, base=base)

res = [model(c) for c in CASES]
print(f"chip: cube peak {TF_PEAK:.0f} / sust {TF_SUST:.0f} TFLOPS fp16; HBM rd {HBM_RD/1e9:.0f} GB/s; "
      f"L2>={L2_AGG/1e12:.0f}TB/s; FLOOR {FLOOR*1e6:.0f}us")
print(f"{'cs':>3} {'form':<5} {'bMxbN':<7} {'iters':>6} {'padE':>5} {'uniqMB':>7} {'relMB':>6} "
      f"{'implMB':>7} | {'tA':>6} {'bA':<5} {'tA*':>6} | {'tB':>6} {'bB':<5} {'pipeB':<7} {'tBnv':>6} "
      f"| {'spA':>5} {'spA*':>5} {'spB':>5} {'now':>5} {'B/now':>5}")
for m in res:
    print(f"{m['cid']:>3} {m['form']:<5} {str(m['bM'])+'x'+str(m['bN']):<7} {m['iters']:>6} "
          f"{m['pad_eff']:>5.2f} {m['uniq_mb']:>7.1f} {m['rel_mb']:>6.0f} {m['gm_mb']:>7.0f} | "
          f"{m['tA']:>6.1f} {m['boundA']:<5} {m['tApk']:>6.1f} | {m['tB']:>6.1f} {m['boundB']:<5} "
          f"{m['pipebnd']:<7} {m['tBnv']:>6.1f} | {m['base']/m['tA']:>5.2f} {m['base']/m['tApk']:>5.2f} "
          f"{m['base']/m['tB']:>5.2f} {m['base']/m['ours']:>5.2f} {m['tB']/m['ours']*100:>4.0f}%")
print()
for tag, key in [('A  ideal kernel, sust cube ', 'tA'), ('A* ideal kernel, peak cube ', 'tApk'),
                 ('B  cur arch, perfect pipe  ', 'tB'), ('B- cur arch, +vec coherent   ', 'tBnv'),
                 ('now measured               ', 'ours')]:
    sp = [m['base'] / m[key] for m in res]
    print(f"mean speedup vs baseline | {tag}: {st.mean(sp):.2f}x  (geomean {st.geometric_mean(sp):.2f}x)")
fam = {'small prefill 1,2,5,6,19,20': [1, 2, 5, 6, 19, 20], 'big prefill 3,7,8': [3, 7, 8],
       'D256 prefill 4,9': [4, 9], 'decode 10-15': [10, 11, 12, 13, 14, 15], 'MTP 16,17,18': [16, 17, 18]}
print()
for name, ids in fam.items():
    ms = [m for m in res if m['cid'] in ids]
    print(f"  {name:<26} A={st.mean([m['base']/m['tA'] for m in ms]):5.2f}x  "
          f"A*={st.mean([m['base']/m['tApk'] for m in ms]):5.2f}x  "
          f"B={st.mean([m['base']/m['tB'] for m in ms]):5.2f}x  "
          f"Bnv={st.mean([m['base']/m['tBnv'] for m in ms]):5.2f}x  "
          f"now={st.mean([m['base']/m['ours'] for m in ms]):5.2f}x")
print()
for m in res:
    print(f"case{m['cid']:<3} A: iters={m['i_iters']:>5} gf={m['gf_ipad']:6.2f} rel={m['i_rel']:5.0f}MB "
          f"bnd={m['boundA']:<5} | B: pipe={m['tB_pipe']:7.1f} hbm={m['tB_hbm']:6.1f} l2={m['tB_l2']:6.1f} "
          f"perIt={m['cpipe_us']:.3f}us({m['pipebnd']}) imb={m['imb']:.2f}"
          + (f" | now effHBM={m['eff_hbm_now']:.0f}GB/s" if m['eff_hbm_now'] else ""))
