"""Iter6 roofline analytical model — FLOPs / bytes / AI per condition.

Pure computation from shape + kernel-structure constants verified against
gqa.py source (task grids, tile sizes, ws dtypes) and cross-checked with
measured Memory.csv path volumes from the iter6 A/B matrices.

Run: python3 examples/gqa/perf_tuning/model_roofline_iter6.py
"""

E = 1024.0  # KB->MB helper below uses 1024-based KB

# Measured anchors (iter6 A matrix, 3-sample mean TaskDur us; Memory.csv KB)
MEAS = {
    # r: (TaskDur_us, GM_to_L1_KB, L0C_to_GM_KB, GM_to_UB_KB, UB_to_GM_KB,
    #     read_main_KB, write_main_KB, l2_rd_hit%, l2_wr_hit%,
    #     aic_rd_hit, aic_rd_miss, aiv_rd_hit, aiv_rd_miss,
    #     aic_wr_hit, aic_wr_miss, aiv_wr_hit, aiv_wr_miss)
    "r1": (10.34, 1536.0, 1024.0, 1536.0, 512.0, 3090.0, 1550.0,
           85.30, 1.54, 9246, 3090, 11840, 544, 16, 2048, 32, 1024),
    "r2": (27.22, 10240.0, 8192.0, 12288.0, 4096.0, 20502.5, 12380.0,
           84.35, 31.42, 40998, 24598, 97360, 1064, 6164, 10240, 1576, 6656),
    "r3": (60.53, 40960.0, 32768.0, 49152.0, 16384.0, 81942.5, 49460.0,
           88.59, 70.33, 188454, 73750, 392272, 1064, 55316, 10240, 13864, 18944),
    "r4": (189.65, 131072.0, 131072.0, 131072.0, 49152.0, 245782.5, 181396.0,
           91.66, 87.36, 753702, 163862, 1048656, 40, 251924, 10240, 63016, 35328),
    "r5": (145.89, 140032.0, 32768.0, 32768.0, 8320.0, 172182.5, 41444.0,
           61.25, 84.09, 581491, 533705, 262224, 40, 55308, 10248, 13846, 2834),
    "r6": (28.02, 15360.0, 12288.0, 12288.0, 6144.0, 24598.5, 18560.0,
           85.40, 48.69, 69671, 28693, 98383, 41, 14356, 10240, 3624, 8704),
}

# Shapes: (B,S,Skv,Nq,Nkv,D, causal, dt_bytes, kernel family)
SHAPES = {
    "r1": (1, 128, 128, 8, 2, 128, True, 2, "per_head_bm64"),
    "r2": (2, 128, 128, 32, 8, 128, True, 2, "g2stack_flat"),
    "r3": (2, 128, 128, 128, 8, 128, True, 2, "g2stack_flat"),
    "r4": (4, 256, 256, 64, 8, 128, False, 2, "g2stack_flat_kv2"),
    "r5": (16, 1, 2048, 32, 8, 128, False, 2, "dense_expert"),
    "r6": (2, 128, 128, 48, 4, 128, False, 2, "g2stack_flat"),
}

# Roofs (sources in report appendix).  Decimal bytes; copy volumes:
#   copy:   40*96*64KiB = 251.66MB per direction (TaskDur 384.95us mean s3/s4)
#   copyl2: 40*64*64KiB = 167.77MB per direction (TaskDur 93.77us mean s3/s4)
CUBE_ROOF_TFLOPS_1650 = 20 * 8192 * 1.65e9 / 1e12   # 270.3 (ini-derived @ measured freq)
CUBE_ROOF_TFLOPS_1500 = 20 * 8192 * 1.50e9 / 1e12   # 245.8 (ini cube_freq)
TORCH_GEMM_TFLOPS = 2 * 4096**3 / 572.42e-6 / 1e12  # empirical cube anchor
HBM_MIXED_TBS = 2 * 251.66e6 / 384.95e-6 / 1e12     # 1.308 TB/s L2-side mixed r+w
HBM_SIDE_TBS = (0.5 * 251.66e6 + 251.66e6) / 384.95e-6 / 1e12  # ~0.98 (rd 50% L2-hit)
L2_MIXED_TBS = 2 * 167.77e6 / 93.77e-6 / 1e12       # 3.578 TB/s L2-resident mixed
MTE2_READ_ROOF_TBS = 20 * 32 * 1.65e9 / 1e12        # 1.056 TB/s spec (ini ddr_rate)
VEC_ADD_TFLOPS = 1.505e9 / 342.91e-6 / 1e12         # measured vec bench


def ceildiv(a, b):
    return (a + b - 1) // b


def prefill_model(name):
    B, S, Skv, Nq, Nkv, D, causal, dtb, fam = SHAPES[name]
    G = Nq // Nkv
    bm, bn = 64, 128
    q_blocks = ceildiv(S, bm)
    kv_iters = ceildiv(Skv, bn)
    kv2 = 2 if fam == "g2stack_flat_kv2" else 1
    tasks = B * Nkv * (G // 2) * q_blocks
    u_total = tasks * kv_iters  # kv_useful == kv_iters (noncausal or S==Skv causal single col-block)
    M, N, K = 128, 128, 128  # stacked tile
    flops_gemm = 2 * (2 * M * N * K) * u_total  # G1+G2 per u
    # softmax vector FLOPs (fp32 tile ops: sub,exp,sum,mul ~4/elem + reduce)
    flops_vec = u_total * M * N * 4
    KB = 1024.0
    q_bytes = tasks * M * D * dtb            # per-task Q stacked tile
    k_bytes = u_total * N * D * dtb          # per-u K tile (reload per u)
    v_bytes = u_total * N * D * dtb
    wsp_l1 = u_total * M * N * 2             # ws_p fp16 -> p_l1
    ws_s_fix = u_total * M * N * 4           # L0C fp32 -> GM
    ws_o_fix = u_total * M * N * 4
    ws_s_ub = u_total * M * N * 4            # GM -> UB readback
    ws_o_ub = u_total * M * N * 4
    mask_ub = u_total * M * N * 4 if causal else 0  # fp32 [128,128]/u... measured 64KB/u
    mask_ub = u_total * 64 * KB if causal else 0    # measured-anchored: 64KB/u
    wsp_ub2gm = u_total * M * N * 2          # ws_p write fp16
    out_bytes = tasks * M * D * dtb          # Output per task
    gm_to_l1 = (q_bytes + k_bytes + v_bytes + wsp_l1) / KB
    l0c_to_gm = (ws_s_fix + ws_o_fix) / KB
    gm_to_ub = (ws_s_ub + ws_o_ub + mask_ub) / KB
    ub_to_gm = (wsp_ub2gm + out_bytes) / KB
    ws_roundtrip = (ws_s_fix + ws_o_fix + ws_s_ub + ws_o_ub + wsp_ub2gm + wsp_l1) / KB
    return dict(name=name, tasks=tasks, u=u_total, kv=kv_iters,
                gflops=flops_gemm / 1e9, gflops_vec=flops_vec / 1e9,
                gm_to_l1=gm_to_l1, l0c_to_gm=l0c_to_gm, gm_to_ub=gm_to_ub,
                ub_to_gm=ub_to_gm, ws_rt=ws_roundtrip, out=out_bytes / KB)


def decode_model():
    B, S, Skv, Nq, Nkv, D, causal, dtb, fam = SHAPES["r5"]
    G = Nq // Nkv  # 4
    tasks = B * Nkv  # G_batches = 1 (G_inner = G = 4)
    kv = ceildiv(Skv, 128)  # 16
    u_total = tasks * kv
    M_tile = 16  # padded mma M — calibrated: L0C_to_GM/GM_to_UB/UB_to_GM all
    # match measured at M_tile=16 (ws [16,128] fp32 8KB/u); M_tile=32 overshoots 2x.
    flops_pad = 2 * (2 * M_tile * 128 * 128) * u_total
    flops_eff = 2 * (2 * Nq * S * Skv * D) * B  # G1+G2 on effective rows
    KB = 1024.0
    kv_bytes = u_total * 128 * D * dtb * 2  # K+V tiles per u (64KB/u)
    q_bytes = tasks * M_tile * D * dtb
    wsp_l1 = u_total * M_tile * 128 * 2     # ws_p fp16 -> p_l1 (4KB/u)
    ws_fix = u_total * 2 * M_tile * 128 * 4  # ws_s + ws_o fp32 L0C->GM (16KB/u)
    ws_ub = ws_fix                          # ws_s + ws_o GM->UB readback
    wsp_ub2gm = u_total * M_tile * 128 * 2  # ws_p UB->GM (4KB/u)
    out = B * Nq * S * D * dtb
    return dict(name="r5", tasks=tasks, u=u_total, kv=kv,
                gflops=flops_eff / 1e9, gflops_pad=flops_pad / 1e9,
                gm_to_l1=(kv_bytes + q_bytes + wsp_l1) / KB,
                l0c_to_gm=ws_fix / KB,
                gm_to_ub=ws_ub / KB,
                ub_to_gm=(wsp_ub2gm + out) / KB, kv_bytes=kv_bytes / KB)


def perhead_model():
    # r1: per-head AUTO bm=64: 16 tasks; L0C->GM = scores+partial fp32 [64,128] x2
    B, S, Skv, Nq, Nkv, D, causal, dtb, fam = SHAPES["r1"]
    tasks = B * Nq * ceildiv(S, 64)  # 16
    flops = tasks * 2 * (2 * 64 * 128 * 128)
    KB = 1024.0
    q = tasks * 64 * D * dtb
    k = tasks * Skv * D * dtb
    v = k
    mask = tasks * 64 * Skv * 4 if causal else 0
    l0c = tasks * 2 * 64 * 128 * 4  # scores + G2 partial fp32 -> GM
    gub = l0c + mask
    out = tasks * 64 * D * dtb
    return dict(name="r1", tasks=tasks, u=tasks, kv=1, gflops=flops / 1e9,
                gm_to_l1=(q + k + v) / KB, l0c_to_gm=l0c / KB,
                gm_to_ub=gub / KB, ub_to_gm=out / KB)


def main():
    print("=" * 100)
    print("ROOFS: cube@1650=%.1f TFLOPS cube@1500=%.1f torch_gemm=%.1f "
          "HBM_mixed=%.3f TB/s L2_mixed=%.3f TB/s MTE2_read_spec=%.3f TB/s vec_add=%.2f TFLOPS"
          % (CUBE_ROOF_TFLOPS_1650, CUBE_ROOF_TFLOPS_1500, TORCH_GEMM_TFLOPS,
             HBM_MIXED_TBS, L2_MIXED_TBS, MTE2_READ_ROOF_TBS, VEC_ADD_TFLOPS))
    print("ridge(HBM)=%.0f FLOP/B  ridge(L2-mixed)=%.0f FLOP/B"
          % (CUBE_ROOF_TFLOPS_1650 * 1e12 / (HBM_MIXED_TBS * 1e12),
             CUBE_ROOF_TFLOPS_1650 * 1e12 / (L2_MIXED_TBS * 1e12)))
    models = {}
    for r in ("r2", "r3", "r4", "r6"):
        models[r] = prefill_model(r)
    models["r1"] = perhead_model()
    models["r5"] = decode_model()

    for r in ("r1", "r2", "r3", "r4", "r5", "r6"):
        m = models[r]
        meas = MEAS[r]
        dur = meas[0]
        gf = m["gflops"]
        tfs = gf * 1e3 / dur  # GFLOP -> TFLOPS over us: gf*1e3/dur
        # bytes: measured path totals (L2-side)
        rd_mb = meas[5] / 1024.0
        wr_mb = meas[6] / 1024.0
        tot_mb = rd_mb + wr_mb
        gbs = tot_mb * 1024 * 1024 / (dur * 1e-6) / 1e9
        ai = gf * 1e9 / (tot_mb * 1024 * 1024)
        # analytical vs measured volumes
        print("-" * 100)
        print(f"{r} [{SHAPES[r][8]}] tasks={m['tasks']} u={m['u']} TaskDur={dur:.2f}us")
        print(f"  FLOPs eff={gf:.3f} GFLOP" + (f" padded={m['gflops_pad']:.2f}" if 'gflops_pad' in m else ""))
        print(f"  measured TFLOPS_eff={tfs:.2f}  ({100*tfs/CUBE_ROOF_TFLOPS_1650:.2f}% of cube@1650)")
        print(f"  L2-side traffic: rd={rd_mb:.1f}MB wr={wr_mb:.1f}MB tot={tot_mb:.1f}MB -> {gbs:.0f} GB/s ({100*gbs/1e3/L2_MIXED_TBS:.1f}% of L2-mixed roof)")
        print(f"  AI={ai:.1f} FLOP/B")
        print("  path KB  analytic vs measured:")
        for k, idx in (("gm_to_l1", 1), ("l0c_to_gm", 2), ("gm_to_ub", 3), ("ub_to_gm", 4)):
            a, mm = m[k], meas[idx]
            print(f"    {k:12s} {a:10.0f} {mm:10.0f}  delta={mm-a:+8.0f} ({100*(mm-a)/max(a,1):+.1f}%)")
        if "ws_rt" in m:
            print(f"    ws roundtrip = {m['ws_rt']/1024.0:.1f}MB = {100*m['ws_rt']/1024.0/tot_mb:.0f}% of L2-side traffic")
        # HBM traffic from L2 counters (reads 128B units; writes calibrated 512B units)
        rh, rm = meas[9] + meas[11], meas[10] + meas[12]
        wh, wm = meas[13] + meas[15], meas[14] + meas[16]
        hbm_rd = rm * 128 / 1e6
        hbm_wr = wm * 512 / 1e6
        print(f"  L2 hit: rd={100*rh/(rh+rm):.1f}% wr={100*wh/(wh+wm):.1f}%  "
              f"HBM est: rd={hbm_rd:.1f}MB wr={hbm_wr:.1f}MB -> {(hbm_rd+hbm_wr)*1e6/1e3/(dur):.0f} GB/s HBM-side")


if __name__ == "__main__":
    main()
