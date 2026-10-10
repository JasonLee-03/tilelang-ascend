# MHA Stage-3 性能调优最终报告

- 日期: 2026-09-18 | perf_session=1 | 迭代上限 10，实际使用 9（8 个优化轮 + 1 收口轮）
- 对象: `examples/mha/mha.py`（MHA forward, BSHD, Developer 模式 threads=1 cv_1_1）
- 目标: DESIGN §12 baseline_compare（cann-bench `tasks/level4/mha`，20 cases，910b2，baseline = CANN FusedInferAttentionScore）
- 工作流: tilelang-perf-optimization skill 5 步（基线采集 → 类型判断 → 优化点识别 Part A/B → 逐项实施 6 子步骤门禁 → 效果验证）

## 1. 最终成果（cann-bench 权威复评 cann_final_eval_20260918_220619）

| 指标 | 调优前 | 调优后 | 变化 |
|------|--------|--------|------|
| 综合得分 | 54.18/100 | **61.76/100** | +7.58 |
| 性能分 | 4.18/50 | **11.76/50** | 2.8x |
| 平均加速比（vs CANN baseline） | 0.13x | **0.42x** | 3.2x |
| geomean 加速比 | 0.104x | 0.369x | 3.5x |
| 通过率 | 20/20 | 20/20 | 保持 |

**全部 20 case 提升 1.84x ~ 6.24x**，无一回退。逐 case（T_old us → T_new us，加速比 vs CANN baseline）：

| case | old us | new us | 提升 | vs baseline | | case | old us | new us | 提升 | vs baseline |
|------|--------|--------|------|-------------|-|------|--------|--------|------|-------------|
| 1 | 46.9 | 14.9 | 3.15x | 0.213→0.671 | | 11 | 2842 | 640.1 | 4.44x | 0.101→0.446 |
| 2 | 534.7 | 129.2 | 4.14x | 0.061→0.254 | | 12 | 2860 | 642.1 | 4.45x | 0.095→0.424 |
| 3 | 2138 | 476.5 | 4.49x | 0.061→0.275 | | 13 | 5249 | 1187.6 | 4.42x | 0.094→0.416 |
| 4 | 179.9 | 66.9 | 2.69x | 0.182→0.490 | | 14 | 2808 | 675.2 | 4.16x | 0.188→**0.781** |
| 5 | 86.7 | 42.2 | 2.05x | 0.121→0.249 | | 15 | 1736 | 751.9 | 2.31x | 0.199→0.459 |
| 6 | 187.5 | 102.1 | 1.84x | 0.175→0.321 | | 16 | 7840 | 1256.1 | **6.24x** | 0.056→0.350 |
| 7 | 16762 | 3233 | **5.18x** | 0.026→0.134 | | 17 | 6325 | 1045.9 | 6.05x | 0.089→0.540 |
| 8 | 16862 | 3300.2 | 5.11x | 0.025→0.130 | | 18 | 1742 | 756.7 | 2.30x | 0.235→0.540 |
| 9 | 9832 | 2782.3 | 3.53x | 0.034→0.121 | | 19 | 152 | 51.3 | 2.96x | 0.216→0.639 |
| 10 | 444 | 103.1 | 4.31x | 0.120→0.518 | | 20 | 104.4 | 55.6 | 1.88x | 0.314→0.590 |

（msprof 代表 shape 全程跟踪表见 §3；cann-bench 口径与本地 msprof 口径互验一致，差 <3%。）

## 2. 瓶颈诊断结论（Step 2/3，msprof op 模式证据）

- 算子类型: **CV 融合混合型**（KERNEL_TYPE_MIX_AIC_1_1，ASCEND_IS_AIC + ASCEND_IS_AIV 双段）。
- 初始瓶颈（case7 基线 profile）: cube 利用率 **2.0%**、AIC scalar 98.5% 在 wait_id1（等 AIV）；AIV vec_time 56.6% + **scalar_time 49.6%**（3×64 `GetValue`+`PIPE_ALL` 行循环 + 64×2 掩码微指令 + 逐 op barrier 的指令下发成本）→ **瓶颈 = vector 指令下发 + 同步，而非算力/带宽**。
- decode 附加瓶颈: valid_m=1 时 block_M=64 行 98% 空转（§10.3 性能注记）。
- 每迭代固定成本（CV workspace GM 往返 w1/w2/w3 + 3 对 cross-flag + AUTO_SYNC barrier 链）在迭代数多时主导 → 减迭代数（trim/bN=128）与减指令数（broadcast 2D/掩码跳过）是两条主线。

## 3. 优化项与逐轮实测（msprof median us；完整门禁记录见 optimization_log.md）

| 轮 | 优化项 | case7 | case1 | case10 | case14 | case20 | 精度 |
|----|--------|-------|-------|--------|--------|--------|------|
| 基线 | — | 16772.5 | 46.60 | 441.4 | 2842.7 | 102.1 | l0 10/10 |
| 1 | #1 causal KV 运行期裁剪 | **8960.6 (-46.6%)** | 45.5 | 442.1 | 2843.6 | 101.7 | l0+l1 ✓ |
| 2 | #2 行循环 broadcast 2D 化（专用 scratch） | **6062.4 (-32.3%)** | **30.1 (-33.8%)** | **217.2 (-50.9%)** | **1665.8 (-41.4%)** | **58.0 (-43.0%)** | 首版 fail→修复→l0+l1 ✓ |
| 3 | #7 Fill+Add→Copy（#3 使能）+ #3 掩码链 2D 化+hoist | **3554.7 (-41.3%)** | **18.3 (-39.2%)** | 217.1 | 1668.5 | 58.5 | l0+l1 ✓ |
| 4 | #6 decode 窄块 block_M=16（S≤16） | 3554.7 | 18.5 | **148.4 (-31.6%)** | **1242.9 (-25.5%)** | 58.2 | probe 13/13 BITEXACT + l0/l1 ✓ |
| 5 | #5 Fixed Core（20 核任务循环） | 3615.9 (+1.7%*) | **15.4 (-17.0%)** | 151.5 (+2.1%*) | **1009.9 (-18.7%)** | **54.1 (-6.9%)** | probe 12/12 BITEXACT + l0/l1 ✓ |
| 6 | #4 全可见迭代掩码运行期跳过 | **3177.1 (-12.1%)** | 14.8 | （无掩码路径，不变） | 1011.3 | 54.3 | probe+l0/l1 ✓ |
| 7 | #8d decode 分支 block_N=128 | 3208.8* | 15.0* | **92.9 (-38.7%)** | **667.2 (-34.0%)** | 54.8* | prefill BITEXACT + decode 阻塞口径过 + l0/l1 ✓ |
| 8 | #9 K/V 手动双缓冲预取 | 不实施（Developer 边界内 codegen 零重叠证据，见 optimization_log 迭代 8） | | | | | n/a |

\* 噪声带内（<3%）漂移；prefill 二进制路径经 probe 证实位级未变。case16（MTP）跟踪：2234.7（iter4 首测）→ 2146.0（#5）→ 1950.1（#4）→ 1247.0（#8d）。

累计（本地 msprof 口径）：case7 **5.2x**、case1 **3.1x**、case10 **4.7x**、case14 **4.3x**、case20 **1.9x**、case16 **6.3x**。

## 4. 精度最终状态（收口全量验证，/tmp/opencode/all_final.log）

- `test_mha.py --level all` exit 0 "Test Passed!"：**L0 10/10 + L1 19/19 + L2 12/12（正确拒绝）+ Boundary 6/6 + 哨兵 4/4 PERF_PASS**（kernel_path 6.1~7.1s ≤ 120s 门；prefill 哨兵 warm kernel 0.27s→0.07s）。
- 阻塞口径（混合容差 + MARE_abs < 10×Threshold）全过；单块特化 case（l0 case4 等）保持位级精确；多块 MARE_abs 余量 ≥10x（哨兵 3.05e-5 vs gate 9.77e-3 ≈ 320x）。
- 精度红线复核：score 双舍入栅格保留（4 次 cast 链未动）、单块预归一化特化保留（并随 #8d 扩展至 S_kv≤128 的 decode）、sumexp+1e-30 保护保留、掩码链语义等价（2D 化 + 运行期跳过均证明为位级/值级恒等）。
- cann-bench 复评 20/20 精度通过（独立口径交叉确认）。
- ruff check / ruff format --check：clean（line-length 140 项目配置）。

## 5. 未实施候选与建议（遗留）

| 候选 | 状态 | 原因 / 建议 |
|------|------|-------------|
| #9 K/V 预取（手动双缓冲） | 评估后不实施 | AUTO_SYNC=True 下 codegen 实测零重叠（Variant A `PipeBarrier<PIPE_ALL>` 全排空；Variant B `M_MTE2/MTE2_M` 成对串行）——与 cube_path §3.1 "gemm_v0 内嵌同步、外部无法 L1 双缓冲"互证。真重叠需 AUTO_SYNC=False+手动 flag（Expert 化，越 DESIGN §7.1/§8.6 边界），预期收益 ~5-10% 不匹配风险。恢复走 DESIGN §10.3-⑥(b)/(c)：框架 codegen address_map 修复（R9）或 VidReduction BSHD 修复（R8）后重评 |
| #8 全局 block_N=128（prefill） | UB 不可行 | bM=64/D=128 需 ~216-242KB > 192KB；仅 decode 分支落地（#8d）。若未来 UB 压力缓解（如 acc_o_half 直转 GM 支持）可重评 |
| Expert 化全面重写（flash_attn_optimize §1-§8：num_stages 批量流水/cross_interval/MMA intrinsic/批量 softmax） | 超出 Stage 3 范围 | case7 距 baseline 仍 7.5x——剩余差距主体是 AUTO_SYNC 逐 op barrier 链 + 每迭代 3 段 GM workspace 往返的结构性同步成本，需 Expert 模式批量流水（批量 softmax + cross_interval 同步）才能逼近；属设计模式变更（DESIGN §2 选型为 Developer），建议作为独立立项由需求方决策 |
| decode 专用重排（heads-as-M / split-KV flash-decoding） | 超出范围 | DESIGN §10.3 明示 v1 不引入；case10 (0.52x)/case13 (0.42x) 与 baseline 的剩余差距主要在此（CANN 用 decode 专用 kernel）；同样属新 kernel 设计 |
| §10.3-④ D=256 bM=64 精简 buffer | 暂缓 | vector-bound 下行数翻倍无净收益；D=256 case（4/9/15/18 已提升 2.3-3.5x）剩余差距同上属结构成本 |

## 6. 产物清单

- kernel: `examples/mha/mha.py`（343 行；变更 = trim 循环界 + broadcast 2D 化×4 + 掩码链 2D 化/hoist/运行期跳过 + decode 窄块 bM=16/bN=128 分支 + Fixed Core 任务循环 + Fill+Add→Copy；host 包装保持 metadata-only，新增 `_cube_core_num()` 设备属性查询）
- 提交副本: `/mnt/workspace/cann/20260917_mha_V01/mha_submission/cann_bench/mha.py` 已同步 + pip 重装（评测走优化版；原始副本备份 /tmp/opencode/mha_submission_ORIGINAL_backup.py）
- `perf_tuning/`: baseline.json、baseline_iter*.json、optimization_log.md（Part A/B + 8 轮门禁记录）、perf_log.md、final_report.md（本文件）、bench_mha.py / parse_prof.py（计时工具）、probe_decode_narrow.py / probe_fc_equiv.py / probe_bN128.py / probe_db_prefetch.py（验证探针）、msprof_baseline|iter1..7/*.json（原始计时证据）
- `history_version/`: mha_impl_s3_perf1..8.py（每轮修改前备份，perf1=原始、perf8=最终）
- 评测报告: `.agents/skills/cann-bench/reports/cann_final_eval_20260918_220619.{json,md,html}`

## 7. 计时口径声明

msprof 为唯一性能依据：本地 = `msprof --output --application` timeline 模式 op_summary Task Duration（MIX_AIC 单任务行/launch）中位数（warmup 5 + iters 20）；口径与 cann-bench KernelDetailsStrategy 对齐（基线时 case1 本地 46.6us vs 锚点 46.9us，差 0.6%）。msprof op 模式（OpBasicInfo/PipeUtilization）用于瓶颈归因。环境：`source set_env.sh && export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:$LD_LIBRARY_PATH`（缺 sqlite3 时 CSV 导出静默失败）。

---

# 会话 2（perf_session=2，2026-09-19，追加预算；内部迭代 6/6 = 上限收口）

> 用户指令：保持 Developer 模式，常规调优深挖剩余性能空间。工作流同会话 1（skill 5 步 + 6 子步骤门禁）；全局迭代编号续接（10-15）。

## S2-1. 最终成果（cann-bench 权威复评 cann_final_eval_20260919_154250）

| 指标 | 会话 1 终态 | 会话 2 终态 | 变化 |
|------|------------|------------|------|
| 综合得分 | 61.76/100 | **65.05/100** | +3.29 |
| 性能分 | 11.76/50 | **15.05/50** | +28% |
| 平均加速比（vs CANN baseline） | 0.42x | **0.55x** | +31% |
| 通过率 | 20/20 | 20/20 | 保持 |

逐 case（T_s1 us → T_s2 us，加速比变化）：

| case | s1 us | s2 us | 提升 | 加速比 | | case | s1 us | s2 us | 提升 | 加速比 |
|------|-------|-------|------|--------|-|------|-------|-------|------|--------|
| 1 | 14.9 | 15.7 | 0.94x* | 0.671→0.64 | | 11 | 640.1 | 372.5 | 1.72x | 0.446→**0.77** |
| 2 | 129.2 | 110.5 | 1.17x | 0.254→0.30 | | 12 | 642.1 | 395.3 | 1.62x | 0.424→0.69 |
| 3 | 476.5 | 356.6 | 1.34x | 0.275→0.37 | | 13 | 1187.6 | 720.3 | 1.65x | 0.416→0.69 |
| 4 | 66.9 | 54.5 | 1.23x | 0.490→0.60 | | 14 | 675.2 | 671.3 | 1.01x | 0.781→0.79 |
| 5 | 42.2 | 29.7 | 1.42x | 0.249→0.35 | | 15 | 751.9 | 584.4 | 1.29x | 0.459→0.59 |
| 6 | 102.1 | 72.8 | 1.40x | 0.321→0.45 | | 16 | 1256.1 | 842.1 | 1.49x | 0.350→0.52 |
| 7 | 3233 | 2254.1 | **1.43x** | 0.134→0.19 | | 17 | 1045.9 | 851.2 | 1.23x | 0.540→0.66 |
| 8 | 3300.2 | 2312.9 | **1.43x** | 0.130→0.19 | | 18 | 756.7 | 581.9 | 1.30x | 0.540→0.70 |
| 9 | 2782.3 | 1936.5 | 1.44x | 0.121→0.17 | | 19 | 51.3 | 48.3 | 1.06x | 0.639→0.68 |
| 10 | 103.1 | 62.6 | **1.65x** | 0.518→**0.85** | | 20 | 55.6 | 40.1 | 1.39x | 0.590→**0.82** |

\* case1（唯一微降，-5.5%）：#11 单块化使对角掩码变宽 [64,128]（wall 仅 15us 的固定开销主导形态；本地 msprof +1.9~2.4% 在 3% 噪声带内，[RESULT-#11] 已如实记录；净收益权衡下保留）。19/20 case 提升，无一超过噪声带的回退。

## S2-2. 会话 2 优化项与逐轮实测（msprof median us；完整门禁记录见 optimization_log.md 会话 2 各节）

| 迭代 | 优化项 | case10 | case16 | case7 | case20 | case15/18 | case3/4 | 精度 |
|------|--------|--------|--------|-------|--------|-----------|---------|------|
| s2 基线 | —（=会话 1 终态复测） | 92.28 | 1266.42 | 3202.04 | 54.67 | — | — | l0 10/10 |
| 10 | #10a decode bN=256（D≤128,Skv>128） | **65.68 (-28.8%)** | **932.84 (-26.3%)** | 3202.04 | 54.81 | — | — | probe+l0/l1 ✓ |
| 11 | #10b D=256 decode bN=256（GEMM1 kL0Size=64） | 65.71 | — | — | — | **583.76/583.94 (-22.7/-22.8%)** | 64.99 | probe+l0/l1 ✓ |
| 12 | #11 prefill bN=128（全 D，零手术） | — | — | **2234.75 (-30.2%)** | **38.26 (-30.2%)** | — | **352.77 (-25.1%) / 52.47 (-19.3%)** | probe+l0/l1 ✓ |
| 13 | #13 decode bN=512（D≤128,Skv>256，GEMM1 kL0Size=32） | **51.04 (-21.8%)** | **853.82 (-8.5%)** | 2260.09 | — | — | — | probe+l0/l1 ✓ |
| 14 | #12 反模式全清单复扫 | 静态项：零新可修复项（11 项逐一核对，记录于 optimization_log 迭代 14） | | | | | | n/a |
| 15 | 收口 | docstring/注释同步 + ruff + smoke + `--level all` + 提交副本同步/pip 重装 + cann-bench 复评 + 本报告 | | | | | | ✓ |

会话 2 累计（本地 msprof）：case10 **-44.7%**（92.28→51.04）/ case16 **-32.6%** / case7 **-29.4%** / case20 **-30.0%** / case3 -25.1% / case4 -19.5% / case15 -22.7% / case18 -22.8% / case14 +0.4%（零变化守卫）/ case1 +2.4%（噪声带，如实记录）。
两会话累计 vs 原始基线：case7 **7.5x**、case10 **8.7x**、case14 **4.2x**、case16 **9.3x**（vs cann-bench T_cand 口径）。

## S2-3. 会话 2 结构性发现（新增硬限知识）

1. **gemm_v0 transpose_B 的 L0B 预算规则**（common.h L1263-1278）：transpose_B 路径 nTile==N 不做 N 分片，B tile = N×kL0Size×sizeof(fp16) 须 ≤ 64KB（kNumSteps=1）或 32KB（多 K-step ping-pong）。→ bN 硬顶：D≤128 kL0Size=128 → 256；D≤128 kL0Size=32 → 512；D=256 kL0Size=64 → 256（bN=512 需 kL0Size=32 且 K=256 → 8-step ping-pong 32KB：512×32×2=32768 亦合法，但被 L1 536KB>511.75KB 独立否决）。
2. **MEMORY_PLANNING 实配远优于手算 naive 和**：prefill bN=128/D=128 手算 227KB（会话 1 据此否决 #8）→ 实配 178.8KB（死亡 scratch rt_2d/acc_s_half/m_i_2d 被合并）。教训：**UB 预算判定必须以实编 GetWithOffset 读数为准**，手算 naive 只能作初筛。最紧配置 = D=64 causal prefill 186.8KB（余量 4.9KB，已在源码注释固化警示）。
3. 单块特化（one-shot 预归一化）覆盖面大幅扩展：decode S_kv≤512 + prefill S_kv≤128 全部落入更高精度路径（probe 多形状 GOLDEN-BITEXACT 或 8x 更接近 golden）。

## S2-4. 精度最终状态（收口全量验证，/tmp/opencode/all_s2_final.log）

- `test_mha.py --level all` exit 0 "Test Passed!"：**L0 10/10 + L1 19/19 + L2 12/12（正确拒绝）+ Boundary 6/6 + 哨兵 4/4 PERF_PASS**（kernel_path 6.1~7.2s ≤ 120s 门；perf_prefill_worst (8,2048,4096,32,128) MARE_abs 3.05e-5 余量 320x；perf_decode_worst (64,1,4096,32,128) causal 覆盖 bN=512+trim+掩码全交互，MARE_abs 3.05e-5）。
- 精度红线四项复核：score 双舍入栅格保留（4×cast 链未动）✓；单块预归一化特化保留且扩展（谓词移动均经 probe 阻塞口径+GOLDEN-BITEXACT 佐证）✓；sumexp+1e-30 保护保留 ✓；掩码链语义等价（bN 参数化原生，(F,F)→(F,T) 迁移 probe 验证）✓。
- 每优化项 probe（BITEXACT 零变化域 + 阻塞口径变化域）+ l0/l1 复验全过；cann-bench 复评 20/20 精度通过（独立口径三确认）。
- ruff check / format：clean。

## S2-5. 遗留问题与建议（会话 2 终态评估）

| 候选 | 状态 | 结论 |
|------|------|------|
| case7/8/9（0.17-0.19x）剩余差距 | **Developer 模式内已触结构边界** | 大 prefill 每迭代成本 ≈ AUTO_SYNC 逐 op barrier 链 + 3 段 CV workspace GM 往返（w1/w2/w3）+ cross-flag 对，均为 AUTO pass 生成、源码不可表达；迭代数已压到硬限（bN=128 prefill/bM=64 UB 186.8KB 顶格）。进一步逼近 baseline 需 Expert 化批量流水（flash_attn §1-§8：num_stages 多缓冲+批量 softmax+cross_interval+细粒度 flag）——模式级变更，独立立项由需求方决策 |
| decode 专用重排（heads-as-M / split-KV） | 证据性放弃（会话 2 Part A2） | heads-as-M 对 MHA proto 数学不适用（每 head 独立 K/V，折叠进 M 需 K 复制 N 份）；split-KV slot-iter 总量守恒、#13 后 case10 packing 低效已被 bN=512 吸收，二段 merge kernel 净收益为负 |
| prefill bM=32+bN=256 | 量化后跳过（#12 记录） | slot-iter 仅 -6.7% 且 KV 重读翻倍、Skv=128 型带 128 padding 列反退化 |
| CV workspace 段数/dtype 削减 | 证据性放弃（Part A2 线索 2） | GM 路由由 AUTO_CV_COMBINE 自动生成不可源码干预；P 已 fp16 最小；L0C→UB 落地缓冲复用=跨核 copy 目标别名=会话 1 #2 实测 corruption |
| K/V 预取双缓冲 | 维持会话 1 结论不重做 | AUTO_SYNC=True 零重叠 codegen 实证（会话 1 #9）；恢复走 DESIGN §10.3-⑥(b)/(c)（R8/R9 框架修复后重评） |
| case1 微降（0.671→0.64x） | 保留（噪声带内） | 净收益权衡：同为 Skv=128 的 case19/20/6 提升 1.06-1.40x；如需求方要求可按 S_kv≤128&&D==64&&causal 回退 bN=64（新增 dispatch 域，不建议） |

**结论：Developer 模式（四 pass_configs 全开、threads=1 cv_1_1）常规调优空间已收口**——tile 尺寸已探至 L0B/L1/UB 三面硬限、指令数已 2D 化、迭代数已裁剪+单块化、核映射已 Fixed Core。剩余差距主体为模式结构性同步成本，属 Expert 化设计变更范畴。

## S2-6. 会话 2 产物清单

- kernel: `examples/mha/mha.py`（414 行；会话 2 变更 = block_N 三档自适应（decode 512/256/128、prefill 128）+ gemm1_kL0 三档（32/64/128）+ docstring/注释同步；host 包装保持 metadata-only 零改动）
- 提交副本: `/mnt/workspace/cann/20260917_mha_V01/mha_submission/cann_bench/mha.py` 已同步 + pip 重装（installed md5 校验一致）
- `perf_tuning/`: baseline_iter10..14.json、optimization_log.md（会话 2 Part A2/B2 + 迭代 10-15 门禁记录）、perf_log.md（Iteration 10-15）、probe_bN_wide.py / probe_d256_wide.py / probe_pf128.py / probe_bN512.py（验证探针）、msprof_s2_base/ msprof_s2_iter10/ msprof_s2_iter11/ msprof_s2_iter12/ msprof_s2_iter13/*.json（原始计时证据）、final_report.md（本节）
- `history_version/`: mha_impl_s3_perf9..12.py（perf9=会话 1 终态、perf10=post-#10a、perf11=post-#10b、perf12=post-#11）
- 评测报告: `.agents/skills/cann-bench/reports/cann_final_eval_20260919_154250.{json,md,html}`

---

# 会话 3（perf_session=3，2026-09-19/20，Expert rev3 轨首轮；迭代 4/10 用毕——中止条款②：连续 3 次无提升）

> 对象：Expert 版 mha.py（rev3-attempt1，cann-bench 67.88/0.68x）。目标：平均加速比逼近 1.0x（M1≥0.75x、M2≥0.9x）。
> 工作流同会话 1/2（skill 5 步 + 6 子步骤门禁）；备份链独立编号 r3_perf{N}。

## S3-1. 最终成果（cann-bench 权威复评 cann_final_eval_20260920_005958）

| 指标 | 会话 3 起点（rev3-attempt1） | 会话 3 终态 | 变化 |
|------|------------------------------|-------------|------|
| 综合得分 | 67.88/100 | **68.50/100** | +0.62 |
| 性能分 | 17.88/50 | **18.50/50** | +0.62 |
| 平均加速比 | 0.68x（精确均值 0.6745） | **0.70x（0.7017）** | +0.027 (+4.0%) |
| 通过率 | 20/20 | 20/20 | 保持 |
| 里程碑 | — | M1 未达（avg 0.70<0.75；case9 0.24<0.4；case7/8 0.466/0.467 ≥0.4 ✅） | — |
| 回退条款 | — | **不触发**（68.50 > 65.05 且 0.70x > 0.55x） | — |

per-case 变化（锚 → 终态，仅列 |Δ|>0.03 与关键项）：

| case | 锚 | 终态 | Δ | 说明 |
|------|-----|------|-----|------|
| **14** | 0.71 | **0.995** | **+0.285** | 唯一回退项修复超额（目标 ≥0.79；本地 msprof 746→531us -28.9%） |
| **17** | 0.69 | **0.916** | **+0.226** | MTP bf16 批处理受益 |
| **16** | 0.58 | **0.678** | **+0.098** | 本地 msprof 763→647us -15.3% |
| **12** | 0.72 | **0.785** | **+0.065** | decode bf16 批处理受益 |
| **13** | 0.71 | **0.766** | **+0.056** | decode 大 B 批处理受益 |
| 20 | 1.04 | 1.001 | -0.039 | Form A 单块（IR 逐字节未变）——评测运行间漂移，如实记录 |
| 1 | 0.78 | 0.742 | -0.038 | 同上（本地 msprof 12.91→12.93us +0.2% 佐证为方差） |
| 19 | 0.89 | 0.867 | -0.023 | 同上（噪声带内） |
| 10 | 0.92 | 0.903 | -0.017 | Form B multi-kv generic（守卫 BITEXACT；本地 msprof 40.5→39.9us 反而改善）——噪声带内持平 |
| 7/8/9 | 0.47/0.47/0.24 | 0.466/0.467/0.240 | ≈0 | 三假设证伪（S3-3），未动 |

**decode 不回退硬约束核验**：case14 ≥0.79 ✅（0.995）/ case20 ≥0.82 ✅（1.001）/ case13 ≥0.69 ✅（0.766）/ case16 ≥0.52 ✅（0.678）/ case10 ≥0.92：0.903（Δ-1.8%，3% 噪声带内持平——路径 BITEXACT + 本地 msprof 改善佐证为评测方差，如实记录）。

## S3-2. 会话 3 优化项与逐轮实测（msprof median us；完整门禁见 optimization_log.md 会话 3 节）

| 迭代 | 优化项 | case14 | case16 | case7 | 其余 | 精度 | 采纳 |
|------|--------|--------|--------|-------|------|------|------|
| r3 基线 | —（=rev3-attempt1） | 746.08 | 763.42 | 928.00 | case1 12.91/case9 1398.35/case10 40.53/case11 349.63/case20 31.30 | l0 10/10+锚 10/10 | — |
| r3-1 | **#R3-1 Form B single_kv_block 任务批处理** | **530.73 (-28.9%)** | **646.84 (-15.3%)** | 927.11 守卫 | case9 -1.3%/case10 -1.6% 守卫 | probe 14/14+l0+锚+压力 ✓ | **✅ YES** |
| r3-2 | #R3-3 w1 dtype 瘦身（fixpipe r16） | 526.65 | 635.11 | **935.99 (+1.0%)** | 7/7 shape 噪声带内 | RNE 探针 11/11 BITEXACT ✓ | ❌ 回滚（无变化） |
| r3-3 | #R3-2 Form A k/v L1 双缓冲 | — | 648.12 守卫 | **931.83 (+0.5%)** | case20 -2.2%/case1 +0.2% | probe 10/10 BITEXACT ✓ | ❌ 回滚（无变化） |
| r3-4 | #R3-4 CROSS_INTERVAL=2（判决实验） | 527.34 | 640.18 | **917.42 (-1.0%)** | 6/6 噪声带内（case9 +2.4%/case10 +2.6% 偏劣） | probe 8/8 BITEXACT ✓ | ❌ 回滚（无变化+§3.2 无收益即退） |
| 收口 | `--level all` + 提交副本同步 + cann-bench 复评 + 本报告 | — | — | — | — | **ALL exit 0**（L0 10/10+L1 19/19+L2 12/12+Boundary 6/6+哨兵 4/4） | — |

## S3-3. 会话 3 结构性发现（三轮证伪收敛的 case7 归因——新增硬知识）

1. **case7（Form A P2 大 prefill）wall≈920-935us 对三类单点手段全部零响应**（每项均位级安全、独立回滚）：①V 侧 vector 指令削减（w1 fp16 省 2 次 [hM,bN] cast + w1 DMA 减半 + FIX 减半 → +1.0%）；②C 侧 K/V GM→L1 装载链重叠（L1 双缓冲批内预取 → +0.5%）；③跨核 SEM 会合频率减半（CROSS_INTERVAL=2 → -1.0% 噪声带）。
2. **推论**：wall 约束 = **每 KV 块迭代的固定结构成本**——两侧 scalar 发射链（AIC scalar 79%/AIV scalar 74-77% busy）+ 三段 ws GM 往返（w1/w2/w3，144KB/iter C 侧 + 80KB/iter V 侧）中不可重叠的部分 × 461 iter/核。busy 指标（V vec 71%、C mte2 51%）均低于 wall，两侧互等（wait_id2 606us/wait_id4 292us）构成有限缓冲下的耦合平衡态——单边优化被另一边吸收。
3. **可复用机制知识**：①fixpipe L0C(fp32)→GM(fp16/bf16) cast 与 V 侧 vcast round-trip **位级等价（RNE）**——11/11 全形态 BITEXACT 实证（未来 dtype 瘦身重试免探针成本）；②3D `[2,bN,D]` L1 + make_nz/zn_layout 前导槽位维透传**可用且位级安全**（10/10 BITEXACT）；③单块任务批处理协议重排（批维=task）位级零漂移（9/9 BITEXACT）——SEM/flag 收支同构变换安全模式。
4. **进一步逼近 1.0x 的剩余手段全部是结构/设计级**（单点调度手段已穷尽）：(a) 迭代数削减——prefill bN=256（L0B 双槽 128KB>64KB 硬限 ✗；单槽牺牲 mma/FIX 乒乓、或 k-split 增 MTE1 指令，收益不确定）；(b) ws 段数削减——CV 三段往返是 Expert 框架下唯一物理路由（§8.2），减段 = 设计变更；(c) case9（0.24x）Form B→A mma 化（§10.3-⑤，K-split+N-split+L0B 共池顶格，工程量大）；(d) #R3-8 扁平 (task,kv) 批处理推广至多块 decode（case10/11/15/18，预期收益同 case14 量级的协议摊薄）。

## S3-4. 精度最终状态（收口全量，/tmp/opencode/r3_all_final.log）

- `--level all` exit 0 "Test Passed!"：**L0 10/10 + L1 19/19 + L2 12/12（正确拒绝）+ Boundary 6/6（0 WARN）+ 哨兵 4/4 PERF_PASS**（kernel_path 6.3s ≤120s 门）。
- 位级诊断锚：**10/10 BITEXACT vs Developer 终态**（每优化项后均复核，全程零失配）；R11 竞态压力 4 配置 ×12 复跑 nondet=0。
- 采纳项 #R3-1 验收：9/9 批处理路径 shape 对旧版 BITEXACT + golden 阻塞口径全过（case14 全尺寸 MARE_abs 1.22e-4 余量 80x；zeros max_abs=0.0）+ 5/5 generic 守卫 BITEXACT。
- 精度红线四项复核：双舍入栅格链逐字保留（r16(qk)→*s fp32→r16(scores)，#R3-3 回滚后落点仍在 V 侧）✅；单块预归一化特化保留且批处理路径同式继承 ✅；sumexp+1e-30 保护保留 ✅；掩码链语义等价（批处理分支 idx≡0 特化 + s0 逐槽重算 row_thresh，探针 causal/bound 变体 BITEXACT）✅。
- cann-bench 复评 20/20 精度通过（独立口径交叉确认）。ruff check/format clean。

## S3-5. 遗留问题与建议（会话 3 终态评估）

| 候选 | 状态 | 结论 / 建议 |
|------|------|-------------|
| case7/8（0.466/0.467x）| **单点手段穷尽（三轮证伪）** | 剩余差距 = 每迭代固定结构成本（S3-3）；候选 (a) bN=256 单槽/k-split 变体——L0B 硬限下的工程权衡，建议独立立项探针先行 |
| case9（0.240x，M1 未达线） | 未动 | Form B P3 barrier 细化（#R3-6，antipattern 冗余全局同步命中：AIC pipe busy 和 ≈95% 零重叠）为**下一会话首位候选**（预期方向明确、gemm_v0 内部 event id 占用需先核对）；彻底解 = §10.3-⑤ Form B→A mma 化（工程量大） |
| #R3-8 扁平 (task,kv) 批处理 | 暂缓 | case14 类收益模型已验证（-28.9%）；推广至 kv_iters≥2 decode（case10/11/15/18）需 V 侧任务边界状态复位/中途写回，复杂度高一档——下一会话次位候选 |
| Form A 单块任务批处理（case1/19/20） | 未实施 | Form A 已有细粒度流水，批处理增量收益预期小于 Form B（其 wall 由流水而非协议主导）；case20 已 ≈1.0x |
| #R3-3 w1 瘦身 / #R3-2 k/v DB | 证据性放弃（本会话） | 零收益实测在案（S3-2）；机制知识入库（S3-3-3），若未来结构变更改变瓶颈构成可低成本重试 |
| case1/19/20 复评 -0.02~-0.04 漂移 | 如实记录 | 三 case 路径 IR 逐字节未变（守卫 BITEXACT + 本地 msprof ±0.2% 佐证）——cann-bench 运行间方差，非代码回退 |

**结论**：会话 3 净收益 = **#R3-1 任务批处理**（综合 67.88→68.50，平均 0.68→0.70x；case14 回退修复至 0.995x、case17 0.916x、case16 0.678x、case12/13 +0.06）。case7/8/9 大 prefill 的 1.0x 差距确认为 Expert 框架下每迭代固定结构成本，单点调优空间收口；M1（0.75x）需 case9 专项（#R3-6/§10.3-⑤）+ #R3-8 推广方有机会，属下一会话/设计级决策范畴。

## S3-6. 会话 3 产物清单

- kernel: `examples/mha/mha.py`（947 行；会话 3 净变更 = #R3-1 Form B single_kv_block 任务批处理双分支（C-scope + emit_v 共享 macro），generic 路径逐字节未动（modulo 缩进），Form A/host/SEM-SIG 矩阵零触碰；md5 1d3a261c9c44c1ad9dd1b30b571717b6）
- 提交副本: `/mnt/workspace/cann/20260917_mha_V01/mha_submission/cann_bench/mha.py` 同步 + 评测脚本编译安装（installed md5 校验一致 1d3a261c）
- `perf_tuning/`: baseline_r3.json、baseline_r3_iter1.json（采纳态基线 + cann-bench 终态数据）、optimization_log.md（会话 3 Part A3/B3 + 迭代 r3-1..r3-4 门禁全记录）、perf_log.md（Iteration r3-1..r3-4）、msprof_r3_base/ msprof_r3_iter1/ msprof_r3_iter2/ msprof_r3_iter3/ msprof_r3_iter4/*.json（原始计时证据）、bench_mha.py（+case9/case11 条目）
- `history_version/`: mha_impl_r3_perf1.py（=rev3-attempt1 基线 d71ba4c3）、r3_perf2.py（=post-#R3-1 采纳态 1d3a261c）、r3_perf3.py（=post-#R3-3 回滚还原态，同 1d3a261c）、r3_perf4.py（=post-#R3-2 回滚还原态，同 1d3a261c）
- 探针（scratch，/tmp/opencode/r3_probes/）：probe_r31_batch.py（14/14）、probe_r33_w1slim.py（11/11 RNE 等价）、probe_r32_kvdb.py（10/10）、probe_r34_ci2.py（8/8）；日志 r3_l0_iter*.log、r3_anchor_iter*.log、r3_all_final.log、r3_cannbench_final.log
- 评测报告: `.agents/skills/cann-bench/reports/cann_final_eval_20260920_005958.{json,md,html}`

---

# 会话 4（perf_session=4，2026-09-20，Expert rev3 轨第二轮；迭代 4/10 用毕——中止条款②：连续 3 次无提升）

> 对象：Expert 版 mha.py（会话 3 终态 = #R3-1 采纳态 1d3a261c，cann-bench 68.50 / 0.7017x）。
> 任务：冲击 M1（平均 ≥0.75x 且 case9 ≥0.4x）；主攻 #R3-6 barrier 细化 + #R3-8 扁平批处理两条移交线索。
> 工作流同会话 1/2/3（skill 5 步 + 6 子步骤门禁）；备份链续编 r3_perf5..8。

## S4-1. 最终成果（cann-bench 权威复评 cann_final_eval_20260920_141424）

| 指标 | 会话 4 起点（=会话 3 终态） | 会话 4 终态 | 变化 |
|------|------------------------------|-------------|------|
| 综合得分 | 68.50/100 | **70.21/100** | **+1.71** |
| 性能分 | 18.50/50 | **20.21/50** | +1.71 |
| 平均加速比 | 0.70x（0.7017） | **0.78x（0.7813）** | **+0.080 (+11.3%)** |
| 通过率 | 20/20 | 20/20 | 保持 |
| 里程碑 M1 | 未达（avg 0.70 / case9 0.24） | **部分达成：avg 0.7813 ≥0.75 ✅；case9 0.324 <0.4 ❌** | avg 线超额 +0.031；case9 线缺口 0.076 |
| 回退条款 | — | **不触发**（70.21 > 65.05 且 0.78x > 0.55x） | — |

per-case 变化（锚 → 终态，全部 20 case）：

| case | s3 | s4 | Δ | 说明 | | case | s3 | s4 | Δ | 说明 |
|------|-----|-----|------|------|-|------|-----|-----|------|------|
| 1 | 0.742 | 0.737 | -0.005 | Form A 未触碰（噪声） | | 11 | 0.824 | **0.925** | +0.101 | decode 多块 generic 受益 |
| 2 | 0.478 | 0.484 | +0.006 | 同上 | | 12 | 0.785 | **0.886** | +0.101 | batched 单块受益 |
| 3 | 0.758 | 0.768 | +0.010 | 同上 | | 13 | 0.766 | **0.883** | +0.117 | batched 单块受益 |
| 4 | 0.732 | **0.846** | +0.114 | Form B prefill D256 受益 | | 14 | 0.995 | **1.394** | **+0.399** | batched 最大受益项 |
| 5 | 0.545 | 0.555 | +0.010 | Form A（噪声） | | 15 | 0.598 | **0.688** | +0.090 | decode D256 generic 受益 |
| 6 | 0.571 | 0.584 | +0.013 | 同上 | | 16 | 0.678 | **0.784** | +0.106 | MTP batched 受益 |
| 7 | 0.466 | 0.464 | -0.002 | Form A 未触碰（噪声带） | | 17 | 0.916 | **1.138** | **+0.222** | MTP bf16 batched 受益 |
| 8 | 0.467 | 0.470 | +0.003 | 同上 | | 18 | 0.703 | **0.811** | +0.108 | MTP D256 generic 受益 |
| 9 | 0.240 | **0.324** | **+0.084** | barrier 移除 -26%（本地）；M1 线未达（三证据链收敛 §S4-3） | | 19 | 0.867 | 0.873 | +0.006 | Form A（噪声） |
| 10 | 0.903 | **0.998** | +0.095 | decode generic 受益 | | 20 | 1.001 | 1.013 | +0.012 | Form A zeros（噪声） |

**decode 不回退硬约束核验**：case10 ≥0.90 ✅（0.998）/ case14 ≥0.99 ✅（1.394）/ case20 ≥0.82 ✅（1.013）/ case13 ≥0.69 ✅（0.883）/ case16 ≥0.52 ✅（0.784）。case7/8/9 只升不降 ✅（7/8 在 ±0.003 评测噪声带、Form A 路径 IR 逐字节未动 + 本地 msprof 守卫 +0.6% 噪声带佐证；9 +0.084）。**14/20 case 提升（+0.08~+0.40），6 case 噪声带内持平，无一回退。**

## S4-2. 会话 4 优化项与逐轮实测（msprof median us；完整门禁见 optimization_log.md 会话 4 节）

| 迭代 | 优化项 | case9 | case14 | case16 | case11 | case10 | case15/18 | 精度 | 采纳 |
|------|--------|-------|--------|--------|--------|--------|-----------|------|------|
| r4 基线 | —（=会话 3 终态；case9 复测 1379.56 与记录 1380.34 一致 Δ0.06%） | 1380.34 | 530.73 | 646.84 | 349.63 | 39.90 | 578/581（锚） | l0 10/10+锚 10/10+压力 ✓ | — |
| r4-1 | **#R4-1 Form B barrier_all 纯移除（15 处）** | **1020.73 (-26.0%)** | **378.16 (-28.7%)** | **565.27 (-12.6%)** | **312.34 (-10.7%)** | **35.34 (-11.4%)** | 504.2/504.5 (-12.8%) | probe 15/15 BITEXACT+l0+锚+压力+l1 ✓ | **✅ YES** |
| r4-2 | #R4-2a prefill D256 GEMM1 kL0=256 单 K-step | **1058.92 (+3.7%)** | — | — | — | — | — | probe 12/12 BITEXACT+l0+锚 ✓ | ❌ 回滚（劣化） |
| r4-3 | #R4-2b prefill D256 k/v L1 双缓冲（预取式零新增 flag） | 1029.77 (+0.9%；op 归因全 pipe 零变化) | — | — | — | — | — | 机制探针+probe 15/15 BITEXACT+l0+锚 ✓ | ❌ 回滚（无变化） |
| r4-4 | #R4-3 扁平 (task,kv) decode 批处理 | — | 377.28 守卫 | 563.14 守卫 | 307.02 (-1.7%) | 35.12 (-0.6%) | 505.2/504.9 (±0.2%) | probe 17/17 BITEXACT+l0+锚+压力 ✓ | ❌ 回滚（无变化） |
| 收口 | `--level all` + 锚/压力终态复跑 + 提交副本同步 + cann-bench 复评 + 本报告 | — | — | — | — | — | — | **ALL exit 0** | — |

会话 4 净变更 = **#R4-1 一项**（Form B C-scope 15 处 `T.barrier_all()` 纯移除；生成码 PipeBarrier<PIPE_ALL> 8→0/份，CrossCoreSetFlag/WaitFlag 协议位置零漂移，C-scope 保持零用户 flag）。本地 msprof 累计（vs r4 基线）：case9 -26.0% / case14 -28.7% / case16 -12.6% / case15/18 -12.8% / case11 -10.7% / case10 -11.4% / case4 族受益（评测 +0.114x）；Form A 守卫 case7 +0.6% / case20 -1.5% / case1 +0.1%（噪声带，IR 未动）。

## S4-3. 会话 4 结构性发现（新增硬知识，rev4 立项核心证据）

1. **gemm_v0 黑盒自同步边界（可复用机制，会话 3 移交前置检查的答案）**：gemm_v0 内部 prologue（`SetFlag/WaitFlag<MTE2_MTE1>(0)` + `<FIX_M>(0)`）与 epilogue（`<MTE1_MTE2>(0)` + `<M_FIX>(0)`，common.h L1305-L1371）构成**调用级自平衡的完整段界序**：前序 MTE2（GM→L1）→ 内部 MTE1、前序 FIX（L0C→ws）→ mma、内部 MTE1 → 后续 MTE2（L1 槽复用）、mma → 后续 FIX（L0C 读出）。四类段界 hazard 全覆盖 ⇒ **gemm_v0 三段式用户代码的 barrier_all 全部冗余，可纯移除且零新增 flag**（event id 0/1 由 gemm_v0 独占自平衡，Form B C-scope 无用户 flag 即无冲突）。量级实证：-26~-29%（Form B 全族）。
2. **case9 剩余 wall 的三证据链收敛（规则内手段穷尽判定）**：barrier 移除后 case9 ≈1020us 的约束 = **gemm_v0 内部每-step 串行链**（MTE1 L1→L0 40-80KB/step + M_MTE1 双槽 ping-pong + 每-step `PipeBarrier<PIPE_M>` 排空前一 mma；scalar_mte1_stall ≈78% wall × 4 step/iter × ~460 iter/核）。①step 数削减（kL0=256 单步）→ **+3.7% 劣化**（kNumSteps=1 关闭 ping-pong，重叠损失 > step 开销节省）；②MTE2 载入隐藏（k/v L1 DB 预取）→ **零响应**（op 归因全 pipe 读数逐位不变——MTE2 45-50% busy 本已被吸收在 stall 阴影下）；③会话 3 三证伪（V 指令/C 装载/SEM 频率）同向。**破局唯一路径 = 手动 mma 流水（Form A 式 L0C 双槽 + 细粒度 flag，无每-step PIPE_M 排空）= §10.3-⑤ Form B→A mma 化 = 设计级变更**（本会话禁实施；证据已足量，见 §S4-5 rev4 建议）。
3. **协议税模型修正（#R3-1 收益不可叠加性）**：barrier_all 移除后，每协议轮 SEM 开销从 ~2-3.6us（barrier 放大态）坍缩至 ~0.2-0.3us——扁平 (task,kv) 批处理把 decode 协议轮削减 3.7-4x（case11 26→7、case15/18 48→12）收益**归零**（6/6 shape 噪声带）。decode 多块 wall = 每槽 MTE2 带宽（case11/15 ≈5.3-5.9us/slot，K+V 256-272KB）。**#R3-1 的 -28.9% 收益模型 = barrier×SEM 复合税，单因子消除后另一因子失效**——批处理类优化的前置检查项：先确认协议税在 wall 中的现存占比。
4. **位级等价新知识入库**：①单 mma k=256 ≡ 链式 2×mma k=128（C0 顺序累加同序，12/12 BITEXACT——kL0Size 上调免位级探针）；②3D 无 annotate L1 [2,bN,D] + gemm_v0 运行期切片 ≡ 2D 串行（15/15 BITEXACT——Form B 系 DB 免机制探针）；③扁平协议重排位级零漂移（17/17，含跨轮任务状态携带/轮中任务切换/phase2 复位时序）。
5. **V 侧任务状态复位的时序约束（扁平化通用教训）**：轮内 phase1 全先于 phase2 执行 ⇒ 任务状态复位（fill）必须放 phase2 的 kv==0 槽（phase1 复位会摧毁前一任务在途状态）；m_i 例外（phase1 独占，reduce_max 覆写语义 + m_i_prev 链保持位级同值）。

## S4-4. 精度最终状态（收口全量，/tmp/opencode/r4_all_final.log）

- `--level all` exit 0 "Test Passed!"：**L0 10/10 + L1 19/19 + L2 12/12（正确拒绝）+ Boundary 6/6 + 哨兵 4/4 PERF_PASS**（kernel_path 6.0-7.4s ≤ 120s 门；prefill_worst (8,2048,4096,32,128) MARE_abs 3.05e-5 余量 320x；decode_worst (64,1,4096,32,128) causal 3.05e-5；d256_bf16 (2,1024,1024,16,256) 4.88e-4；max_grid (128,128,128,64,64) 9.77e-4）。
- 位级诊断锚：**10/10 BITEXACT vs Developer 终态**（终态 e80f7f65 上新鲜复跑；每优化项后均复核，全会话零失配——#R4-1 纯调度变更位级零漂移实证）；R11 竞态压力 4 配置 ×12 复跑 nondet=0（终态复跑）。
- 采纳项 #R4-1 验收：probe_r41 15/15 BITEXACT（Form B batched/generic 全路径矩阵 + Form A 双守卫）+ golden 阻塞口径全过。回滚三项各自 probe 全 BITEXACT（12/12、机制+15/15、17/17）——四项变更全部位级安全，采纳判据纯由性能规则驱动。
- 精度红线四项复核：双舍入栅格链逐字保留（4×cast 链未动，r16 落点 V 侧）✅；单块预归一化特化保留（batched/generic/扁平探针三路径同式继承验证）✅；sumexp+1e-30 保护保留 ✅；掩码链语义等价（#R4-1/#R4-3 条件式 s0=0 代入恒等，探针 causal/bound/tail 变体 BITEXACT）✅。
- cann-bench 复评 20/20 精度通过（独立口径交叉确认）。ruff check/format clean（947 行终态，与起点同行数：15 删 15 增注释抵冲）。

## S4-5. 遗留问题与建议（会话 4 终态评估；M1 后距 M2/1.0x 归因）

| 候选/case | 状态 | 结论 / 建议 |
|------|------|-------------|
| **case9（0.324x，M1 唯一未达线）** | **规则内穷尽（三证据链 §S4-3-2）** | 剩余差距 = gemm_v0 每-step scalar 链（78% wall）。**rev4 立项建议（设计级）：§10.3-⑤ Form B→A mma 化**——手动 K-split/N-split + L0C 双槽 + 细粒度 flag（Form A 已验证模式），消除每-step PipeBarrier<PIPE_M>；工程量要点：D=256 GEMM1 双 k-chunk（init=True/False 链）、GEMM2 双 D-half、l0b_s/l0b_o 共池预算（64KB 顶格 ⚠ 需分池复核）、位级锚失配预案（mma 累加序变化 → 阻塞口径 + 根因分析纪律）。预期：case9 → case7 同族结构（0.46x 级），M1 case9 线可达 |
| case7/8（0.464/0.470x） | 会话 3 结论维持 | Form A 每迭代固定结构成本（scalar 发射链 + 3 段 ws GM 往返）；bN=256 变体（L0B 硬限工程权衡）未启动（中止条款②）——若 rev4 立项可并入 Form A 结构重组一并评估 |
| case2/5/6（0.484/0.555/0.584x） | 未分析（本会话范围外） | Form A 中小 prefill/cross 形态——每迭代固定成本占比高（同 case7 族但 wall 小）；rev4 若做 Form A 结构优化可顺带受益；独立线索：launch/协议固定开销占比需 msprof op 归因（下一会话可低成本补测） |
| #R4-2a kL0=256 / #R4-2b k-v DB / #R4-3 扁平批处理 | 证据性放弃（本会话） | 三项全位级安全 + 零/负收益实测在案；机制知识入库（§S4-3-3/4/5），若未来结构变更改变瓶颈构成可低成本重试（探针全免） |
| M2（0.9x）距离 | 量化 | 0.7813→0.9 需再 +0.119 avg：case9→0.46 贡献 +0.007、case7/8→0.6 贡献 +0.027、case2/5/6→0.75 贡献 +0.026、decode 族（10/11/12/13/15/16/18）各 +0.1 贡献 +0.035——**无任何单点杠杆，全面依赖 rev4 结构级变更**（Form B→A mma 化 + Form A 每迭代成本重构） |
| 1.0x 距离 | 定性 | CANN FusedInferAttentionScore 为高度特化融合 kernel（decode 专用重排 + 单 kernel 无 ws 往返）；Expert 框架 3-ws GM 往返为物理路由（§8.2）——1.0x 全面达成需框架级变更（L0C→UB 直通 / vid 重排），超出算子级调优范畴，建议需求方按 §S4-3 证据评估投入产出 |

**结论**：会话 4 净收益 = **#R4-1 barrier 纯化**（综合 68.50→**70.21**，平均 0.7017→**0.7813x**，14/20 case 提升无一回退；case14 1.394x / case17 1.138x / case10 0.998x / case9 +0.084）。**M1 = 部分达成**（avg 线超额 ✅ / case9 线 0.324 ❌——规则内手段已穷尽，三证据链指向 §10.3-⑤ 设计级 mma 化为唯一破局路径，rev4 立项证据已备齐）。中止条款②触发（连续 3 次无提升），迭代 4/10，采纳 1/回滚 3。

## S4-6. 会话 4 产物清单

- kernel: `examples/mha/mha.py`（947 行；会话 4 净变更 = Form B C-scope 15 处 barrier_all 移除 + docstring/注释同步；md5 **e80f7f652a7123c82c71c0354723755b**；Form A / emit_v / host / SEM-SIG 矩阵逐字节未动——区域 diff 核验）
- 提交副本: `/mnt/workspace/cann/20260917_mha_V01/mha_submission/cann_bench/mha.py` 同步（md5 e80f7f65 一致）
- `perf_tuning/`: baseline_r4.json（起点基线 + case9 复测）、baseline_r4_iter1.json（#R4-1 采纳态）、optimization_log.md（会话 4 Part A4/B4 + 迭代 r4-1..r4-5 门禁全记录）、perf_log.md（Iteration r4-1..r4-4）、msprof_r4_base/（case9/15/18）、msprof_r4_iter1/（8 case）、msprof_r4_iter2/（case9）、msprof_r4_iter3/（case9）、msprof_r4_iter4/（6 case）
- `history_version/`: r3_perf5.py（=改前基线 1d3a261c）、r3_perf6/7/8.py（=各轮改前备份，三者同体 e80f7f65 = 终态）
- 探针（scratch）：/tmp/opencode/r4_probes/probe_r41_barrier.py（15/15）、probe_r42_kl0.py（12/12）、probe_r42b_l1db_mech.py（机制）、probe_r43_kvdb.py（15/15）、probe_r44_flat.py（17/17）；日志 r4_l0_iter1..4、r4_l1_iter1、r4_anchor_iter1..4+final、r4_all_final、r4_cannbench_final.log；op 归因 /tmp/opencode/r4_op_case9{,_db}/；生成码 /tmp/opencode/r4_src/
- 评测报告: `.agents/skills/cann-bench/reports/cann_final_eval_20260920_141424.{json,md,html}`

---

# 会话 5（perf_session=5，2026-09-21，Expert rev3 轨第三轮；迭代 2/软预算 3-4，采纳 1 回滚 1）

> 对象：Expert 版 mha.py（会话 4 终态 = #R4-1 采纳态，md5 e80f7f65，947 行，cann-bench 70.21 / 0.7813x）。
> 触发：**用户基于 `perf_tuning/simulator/case1` camodel 指令级流水图的直接实施指令**——"整个算子 C 核的等待时间明显长于 V 核，V 核计算当前的问题是 MTE 与 V 核计算是串行执行，V 核计算过于碎片化，请进行修改"（此前一次"生成方案并问询"被用户 dismiss，本轮 mode=implement）。
> 验收锚定：**单块 Form A 类 case1/6/19/20**（case6 为主受益者，12.8 task/核）；守卫 = Form A 多块 generic（case7/2/5）+ Form B 全族（case9/10/14/16）。

## S5-1. 用户三点观察的独立量化复核（不盲信转述）

方法：自行重解析会话 4 的 merged `trace.json`（77365 events），busy = 每 pipe `[ts, ts+dur)` 区间**并集**，overlap = MTE2∪MTE3 与 VECTOR 的**交集**（工具 `/tmp/opencode/s5_analysis/trace_stats.py`，副本存 `simulator_r5/`）。

| 用户观察 | 复核判定 | 实测（core0） |
|----------|----------|---------------|
| ① C 核等待明显长于 V 核 | **证实** | C FLOWCTRL 7.726us/9.288us wall = **83.18%**；V FLOWCTRL 5.874us/9.832us = **59.74%**；两侧均以 `WAIT_FLAG_DEVI`(n=6) 为绝对主体 |
| ② MTE 与 VECTOR 串行 | **证实**（本人读数略高于 Orchestrator，结论同） | MTE2∪MTE3 = 2.447us，VECTOR = 3.387us，交集 **0.1297us = MTE 的 5.30% / VECTOR 的 3.83%**（Orchestrator 报 0.077us = 3.1%/2.3%；差异溯源：29 对 ≤25.4ns 的相位边界簿记是否计入。两口径均判"几乎完全串行、io_buf 双槽乒乓未发生"） |
| ③ VECTOR 计算过于碎片化 | **精确证实** | VECTOR 11 段：9.2/11.4/49.2/15.1/54.1/8.6/**1537.8**/113.5/68.1/**1537.8**/67.6 ns；6 段 8.6~54.1ns 微型碎片 + 恰好 2 段 1537.8ns 真实 softmax（每 task 一次） |

根因（复算确认）：case1 = Form A ∧ `single_kv_block`（S=128>16, D=64≤128, S_kv=128≤bN）∧ `block_num=32`/`core_num=20`/`q_tasks=1`/`r_tasks=12` ⇒ cid0-11 各 2 task（**同一 (b,h) 的相邻两个 s_block**）、cid12-19 各 1 task；`eff_num_outer = ceildiv(1,8) = 1` ⇒ `batch_iters = 1` ⇒ **NUM_STAGES=8 批处理流水与 io_buf[2] 双槽彻底空转**，每 task 走"C-QK → V-softmax → C-PV → V-epi"四相位严格串行链，task 之间零重叠（core0 上两次 `wait SEM_WS2_V2C` 分别阻塞 3816ns / 2751ns）。

**两处历史认知修正**：①`final_report:239`"Form A 单块任务批处理未实施"是**预测性否决、从未实测**，真实原因是 `[IMPL-#R3-1]④`"C 侧批镜像未写 ⇒ C/V 协议失配死锁"，V 侧批分支早已存在且被 Form B 五 case 位级验证；②`[#R3-9]`"K/V 跨任务无复用"对 single_kv_block 子类**不成立**（trace 地址级证据：task1/task2 的 K 源地址同为 `0x11351c00`、V 同为 `0x11391e00`）。

## S5-2. 会话 5 优化项与逐轮实测（msprof median us）

| 迭代 | 优化项 | case1 | case6 | case19 | case20 | 守卫 7 case | 精度 | 采纳 |
|------|--------|-------|-------|--------|--------|-------------|------|------|
| r5 基线 | —（=会话 4 终态 e80f7f65；7 项与会话 4 记录同测全在 ±3% 内 ⇒ 环境一致） | 13.03 | 54.08 | 35.86 | 30.71 | case7 926.43 / case2 67.39 / case5 18.44 / case9 1018.49 / case14 377.42 / case10 34.41 / case16 568.37 | l0 10/10+锚 10/10 | — |
| **r5-1** | **#R5-1 Form A single_kv_block 任务批处理** | **11.56 (-11.28%)** | **30.07 (-44.40%)** | **23.38 (-34.80%)** | **18.28 (-40.48%)** | 7/7 全在 ±3% 内（最大 case10 +2.12%）**且 12/12 守卫 IR 逐字节未变** | probe **30/30** + l0/l1/coverage/all + 锚 10/10 + R11 继承 4 配置 **＋新增扩展 7 配置（打在被改路径上）** nondet=0 | **✅ YES** |
| r5-2 | #R5-3 同 (b,h) 相邻槽 K/V L1 驻留（runtime 谓词跳过同字节重读） | — | — | — | — | — | **FAIL（非确定性数据损坏）** | ❌ **回滚**（4 次实验证伪链，见 §S5-4） |
| 收口 | 终态复测 + 全量门禁终态复跑 + IR 级还原核验 + 本报告 | **11.07 (-15.04%)** | **29.66 (-45.16%)** | **22.98 (-35.92%)** | **18.16 (-40.87%)** | 生成码逐字节未变（免重测） | `--level all` exit 0 + 锚 10/10 + 压力 nondet=0（**终态新鲜复跑**） | — |

**跨会话 msprof 方差声明（如实）**：同一 kernel 在 iter1 与收口两次 msprof 会话间最大差 -4.2%（case1 11.56→11.07），**大于会话内 stdev（0.113~0.208）**⇒ 采纳判定以**保守的 iter1 数字**为准；两组读数相对基线均远超 3% 噪声带，采纳结论不受影响。

**加速比外推（估算，非权威）**：按会话 4 cann-bench per-case × (T_base/T_new) ⇒ case1 0.737→~0.831 / case6 0.584→~1.050 / case19 0.873→~1.339 / case20 1.013→~1.702，对 20-case 平均贡献 +0.0857 ⇒ 平均 0.7813→**~0.867（估算）**。⚠ 该外推**偏乐观**（cann-bench 口径含每-launch 固定开销，不随 kernel 时间等比缩放；会话 3/4 均记录过 ±0.02~0.04 运行间漂移）。**权威数字需 cann-bench 复评——本轮未执行，属 Orchestrator 决策项。**

## S5-3. 机制层验证（本轮新增验收件；防 [#R3-4]"只看 wall 数字不看机制"覆辙）

**采集方法已恢复并复现**（会话 4 未留脚本）：`msopprof simulator --aic-metrics=PipeUtilization --dump=true --soc-version=Ascend910_9362 --kernel-name=main_kernel --application="python3 .../bench_mha.py --case case1 --warmup 0 --iters 1"`；恢复依据 = `/tmp/opencode/msprof_sim_case1.log` 的 `Binary Simulation Running, use simulator Ascend910 9362`。**在未修改的会话 4 终态上重采，per-core running time 与原始日志吻合到 0.01us、VECTOR 11 段长度逐一相同** ⇒ 方法学可信（完整命令与陷阱见 `simulator_r5/README.md`：默认模式会在真机跑 DeviceProf 并报 `Addr info is null`，`TimelineDetail` 在 simulator 模式下非法）。

| 用户观察 | 基线（本人重采） | #R5-1 | 判定 |
|----------|------------------|-------|------|
| ① C 核等待长于 V 核 | C 7.726us(85.0%) / V 5.876us(61.0%)；wall C 9.093 / V 9.637 | C **6.564us(82.9%, -15.0%)** / V **3.705us(43.7%, -37.0%)**；wall C **7.919(-12.9%)** / V **8.487(-11.9%)** | 两侧**绝对**等待时间均下降。**如实说明**：C 的等待**占比**仍 ~83% > V 43.7%——该 shape 下 C 的真实工作（CUBE 0.322 + MTE1 0.502 + FIXP 2.108us）本就远小于 V 的 VECTOR 3.368us，C 结构性轻载；缺陷不在占比高，而在这些等待**位于关键依赖链上**，本轮已移除（V 等待 -37%、V 的 VECTOR 占比 35.1%→**39.7%**） |
| ② MTE 与 VECTOR 串行 | 交集 0.1297us = MTE **5.30%** / VECTOR **3.83%** | 交集 **0.5146us（+296.7%）** = MTE **20.93%（×3.95）** / VECTOR **15.28%（×3.99）** | **已解决**：io_buf[2] 双槽乒乓被真正激活。旁证：V MTE2 union +54.6% 而 MTE 总 union 仅 +0.5%（搬运被摊开重叠而非增量）；MTE3 union -38.8%（两次写回在批处理 phase-2 背靠背） |
| ③ VECTOR 碎片化 | 11 段；两段真实 softmax 间隔 **2.684us** | **10 段**；两段 softmax 间隔 **0.582us（-78.3%）** | **部分解决**：空隙内容从"等对侧核 PV+epi 相位"变为"自身必要 DMA（P0 写 187.6ns + w1[1] 读 350.3ns）"。**未达成部分如实记录：微型碎片段数未实质减少**（<60ns 口径 6→6、<100ns 口径 8→8）——那是 flag 管程 copy 模式固有的 BAR/prologue-epilogue 簿记，批处理不消除；[#R3-4] 已判决性证明削减此类指令数不缩减 wall，故未列为本轮目标 |

真实重叠的预期代价（如实记录）：两段 softmax 各自 1537.8→1553.0/1608.1ns（+1.0%/+4.6%）= MTE2 与 VECTOR 争用 UB 带宽的签名；净 wall 仍 -12%。

## S5-4. 会话 5 结构性发现（新增硬知识）

1. **单块 Form A 的批处理收益远大于此前预测**：`final_report:239` 的预测性否决（"Form A 已有细粒度流水，批处理增量收益预期小于 Form B"）**被实测推翻**——case6 **-44.4%**、case20 **-40.5%**、case19 **-34.8%**，量级与 Form B 的 #R3-1（case14 -28.9%）相当甚至更大。**教训固化：以"预期收益小"为由不实施 = 预测性否决，必须标注为未实测，不得写成结论。** 机制解释：Form A 的细粒度 flag 流水只覆盖**单个 task 内部**的 pipe 重叠，对"task 之间的四相位串行链"无能为力；批处理是唯一能删掉这条链的规则内手段（`T.Pipelined` 被 DESIGN §2.2-d/R9 禁用）。
2. **"省略一次冗余 GM→L1 DMA"在手动 pipe-pair flag 协议下不安全（判决性证伪，4 次实验）**：谓词逻辑、生成码条件、flag 位置、L1 无别名、runtime-if 机制、P1/P2 池分类、K/V 哪一侧——**逐项排除后仍非确定性损坏**。恒真守卫 BITEXACT ⇒ 罪在"真的不发那条 DMA"。首要未验证假设与最小复现入口见 `optimization_log.md [RESULT-#R5-3]`。
3. **探针方法学新教训（可复用）**：**"跳过冗余装载"类优化必须用非零数据探针**——case20（全零输入）对该类改动是**假阴性陷阱**（K 恒零 ⇒ 错误跳过完全不可见，曾让 K-only 变体误报 BITEXACT）。
4. **TIR 复合布尔条件写法（已实测，非猜测）**：`T.serial` 槽循环内 `T.And(a, b)` 与 Python `a and b`（**两操作数均为 TIR**）生成完全相同的 TIR `if a and b`；rev2-A1 的 BoolOp 警告只针对"Python bool 混 TIR"。仓内先例 `examples/gemm/example_gemm_intrinsic_persistent.py:76`。
5. **#R5-1 后的新瓶颈读数（camodel core0）**：**C 侧 MTE2 = 3.961us = C wall 的 50.0%**，已成为两侧全部真实工作管中的最大项（单段 3050.8ns = Q1,K1,Q2,K2 背靠背）；其余 C FIXP 2.108us(26.6%) / V VECTOR 3.368us(39.7%) / C CUBE 0.322us(4.1%)。⇒ 下一轮的靶心是 C 侧 GM→L1 装载量，而 #R5-3 正是打这个靶（方向正确、手段受阻）。
6. **多块 Form A 的 K/V 冗余更大但 Stage 3 内不可消除（新发现，如实归档）**：case7/8 的 s_blocks=16 且任务连续分配 ⇒ 连续 16 个 task 共享同一 (bz,by)，每 task 的整条 K/V 块序列（causal 裁剪后 ≤8 块）被重复装载 ⇒ 理论 ~16× 冗余。但消除它需要 **kv-outer / q-task-inner 循环互换**，而 online softmax 的 acc_o 是 per-q-task 的跨 kv 累加状态，互换后需同时持有 16 份 acc_o（UB 预算不允许）⇒ 属 DESIGN 级重构，非 Stage 3 范畴。

## S5-5. 精度最终状态（收口全量，终态 md5 3d56c5dd 上新鲜复跑）

- `--level all` exit 0 "Test Passed!"：**L0 10/10 + L1 19/19 + L2 12/12 正确拒绝 + Boundary 6/6（0 BOUNDARY_WARN）+ 哨兵 4/4 PERF_PASS**（kernel_path 6.5s ≤120s 门；/tmp/opencode/r5_probes/r5_all_iter1.log）。覆盖门禁 `coverage_check.py --proto` = **19 PASS / 0 MISS / 0 N/A**（exit 0）。
- 位级诊断锚：**10/10 BITEXACT vs Developer 终态**（改前态 + 终态各复跑一次，零失配）。
- R11 竞态压力：继承 4 配置 ×12 复跑 nondet=0 **＋ 本轮新增扩展 7 配置 ×12 复跑 nondet=0**（`stress_r51.py`，专打新批分支：batch_tasks=1/2/5/6/7/8、num_tb=1/2 跨轮、P1/P2、双 dtype、causal、双尾块、my_count 非齐次）。**方法学修正入库：继承的压力配置全部落在 generic/Form B 路径、并不覆盖被改路径——压力门必须打在被改路径上。**
- 采纳项 #R5-1 验收：probe **30/30**（19 个批处理路径 shape 对改前 BITEXACT + golden 阻塞口径全过，最差 MARE_abs 1.95e-3 vs gate 7.81e-2 = 40x 余量；11 个守卫 BITEXACT）；PART 0 host-plan 审计确保 batch_tasks 覆盖 {1,2,3,4,5,6,7,8} 奇偶全备、num_tb {1,2} 含 8+5/8+4/8+1 跨轮、P1 分池与 P2 共池、双 dtype、causal/非 causal、S 尾块、S_kv 尾块、双尾块、cross(S≠S_kv)、zeros、显式 scale、大幅值域。
- 精度红线四项复核：双舍入栅格链**逐字未动** ✅；单块预归一化特化保留（批分支 `sumexp_is[i]+1e-30 → broadcast → div` 与 generic 特化同序）✅；sumexp+1e-30 保护保留 ✅；掩码链语义等价（批分支 `block_N-1 > T.min(s0+offset_eff,S_kv-1)` = generic 式在 idx≡0 下的恒等式；`row_thresh` 逐槽按 `s0+vid*half_M` 重算；causal/bound/双尾块变体全 BITEXACT）✅。
- **IR 级守卫（本轮新增硬门）**：16 shape `get_kernel_source()` dump ⇒ **12/12 守卫 shape 生成码逐字节相同**、4/4 目标 shape 按预期改变；基线 dump 与会话 4 `/tmp/opencode/r4_src/FormB_case9_postR41.cc`+`FormB_case14_postR41.cc` md5 一致 ⇒ 起点确为会话 4 终态；docstring 同步经实证 **codegen-neutral**（16/16 逐字节相同）；回滚后 **16/16 逐字节还原**。**该门比 msprof 守卫更强：它把"未触碰路径"从统计论证升级为逐字节证明。**
- ruff check / format：clean（1086 行终态）。

## S5-6. 遗留问题与建议（会话 5 终态评估）

| 候选/case | 状态 | 结论 / 建议 |
|-----------|------|-------------|
| **case1/6/19/20（单块 Form A）** | ✅ **本轮已解决** | #R5-1 采纳：-11.3~-44.4%（保守口径），机制层 MTE↔VECTOR 重叠 ×3.95、inter-softmax 空隙 -78.3% |
| **#R5-3 K/V L1 驻留** | ❌ **证伪回滚（根因未定案）** | 冗余是事实（case6 一个批内 7/8 的 K/V 装载同字节），消除手段（省略 DMA）不安全。**重试前置条件 = 算子外最小复现定案根因**；备选写法 = [#R3-2] 的"wait+set 同进同出"（消费侧 wait 也条件化，令牌仍平衡） |
| **#R5-2 单块 O 直写 FIX(L0C)→Output** | ⏸ **未启动（本轮唯一未试的结构候选）** | 直接砍掉关键链第四相位（V-epi，core0 上 0.93us/task）+ 删 WS3 一对 SEM；先例 `fusedmoe:628` / `lightning_indexer:756-764`；fixpipe fp32→dtype RNE ≡ V 侧 vcast 已由 [#R3-3] 探针 11/11 BITEXACT 实证。**前置门 = "L0C→BSHD 4D GM 运行期行切片"独立最小探针**（仓内先例均为整块或 BHSD） |
| **#R5-4 NUM_STAGES 12/16** | ⏸ 暂缓 | #R5-1 后 case6 为 8+5 两轮批，NUM_STAGES=16 可折成单轮；但**爆炸半径大**（每个 shape 的 IR 都变 ⇒ 失去 IR 逐字节守卫；GM ws 足迹 P2 15.7→31.4MB，R14 L2 驻留风险）⇒ 需全 20 case 复测，非低成本旋钮 |
| case7/8（0.464/0.470x） | 会话 3 结论维持 + 新增归因 | 每迭代固定结构成本；**新增**：K/V ~16× 冗余已证实但需 kv-outer 循环互换（= acc_o 状态 ×16，UB 不允许）⇒ DESIGN 级（§S5-4-6） |
| case9（0.324x，M1 唯一未达线） | 会话 4 结论维持 | 规则内下限（三证据链 §S4-3-2）；破局唯一路径 = §10.3-⑤ Form B→A mma 化（设计级，rev4 建议证据已备齐） |
| cann-bench 权威复评 + 提交副本同步 | ✅ **已由 Orchestrator 补充执行**（见 §S5-8） | 提交副本已同步至终态 md5 3d56c5dd（旧副本 e80f7f65 备份 /tmp/opencode/mha_submission_backup_before_r5_sync.py）；cann-bench 权威复评 **cann_final_eval_20260921_150725**：综合 70.21→**71.40**，平均加速比 0.7813→**0.8720x**，20/20 通过，与本轮本地 msprof 保守估算方向一致且量级更优 |

**结论**：会话 5 净收益 = **#R5-1 Form A single_kv_block 任务批处理**（变更面 = 1 行分派门 + 117 行 C 侧批镜像；V 侧批处理本体零改动，直接复用 Form B 已位级验证的分支）。**用户三点观察中 ② 已解决（重叠 ×3.95）、③ 部分解决（inter-softmax 空隙 -78.3%，微型碎片段数如实报告为未减少）、① 两侧绝对等待时间均下降（C -15.0% / V -37.0%）且 wall -12%，C 的等待占比仍高已给出结构性解释**。4/4 目标 case 提升 11.3~44.4%（保守口径，本地 msprof），7/7 守卫在噪声带内且 12/12 守卫 IR 逐字节未变，精度 6 门全过（含终态新鲜复跑与专打被改路径的扩展压力门）。#R5-3 按证伪归档（4 次实验证据链完整、根因未定案已如实声明）。中止条款②未触发（仅 1 次无提升）；按预算纪律收口，未强行耗尽。**（Orchestrator 补充：上述"保守口径"数字后经 §S5-8 cann-bench 权威复评确认为真实收益，非估算——case1 0.737→0.894x、case6 0.584→1.083x、case19 0.873→1.352x、case20 1.013→1.714x，平均 0.7813→0.8720x。）**

## S5-7. 会话 5 产物清单

- kernel: `examples/mha/mha.py`（**1086 行，md5 3d56c5dd3d8a60d26eed7ade2e404a2b**；会话 5 净变更 = emit_v 分派门 1 行 + Form A C-scope 批镜像分支 + docstring/注释同步；Form A generic 循环 97 行 identical-modulo-indent（ruff 对 `eff_kv_iters` 长行括号重排 1 处）、Form B prim_func / host 包装 / SEM-SIG 矩阵 / init-destroy 逐字节未动）
- `perf_tuning/`: `baseline_r5.json`（起点基线 + 用户三点观察的独立复核 + camodel 方法学复现校验）、`baseline_r5_iter1.json`（#R5-1 采纳态）、`baseline_r5_iter2.json`（#R5-3 回滚记录 + 证伪链）、`optimization_log.md`（会话 5 S5-Step1~4 + Part A5/B5 + 迭代 r5-1/r5-2 门禁全记录）、`perf_log.md`（Iteration r5-1/r5-2/r5-3）、`msprof_r5_base/`（11 case）、`msprof_r5_iter1/`（11 case）、`msprof_r5_final/`（4 目标 case 终态复测）、**`simulator_r5/`**（README = 恢复出的采集命令与陷阱 + case1_base/ + case1_r51/ 的 merged trace 与 core0 三核 trace/instr_exe + `trace_stats.py`/`cmp_mechanism.py` + `r51_mechanism_cmp.txt`）、`bench_mha.py`（+case6/case19/case2/case5 条目，工具非交付物）
- `history_version/`: `mha_impl_r3_perf9.py`（=改前基线 e80f7f65）、`r3_perf10.py`（=#R5-1 采纳态 3d56c5dd）、`r3_perf11.py`（=r5-2 改前，同体 3d56c5dd）、`r3_perf12.py`（=r5-2 回滚还原态，同体 3d56c5dd）
- 探针与证据（scratch）：`/tmp/opencode/r5_probes/`（probe_r51_forma_batch.py 30/30、probe_r53_kvresidency.py FAIL、bisect_r53.py、bisect2_r53.py、diag_r53.py、probe_tir_bool.py、stress_r51.py、anchor_stress_r5.py、dump_src.py、run_msprof.sh；日志 probe_r51.log、probe_r53.log、r5_l0_iter1.log、r5_l1_iter1.log、r5_coverage_iter1.log、r5_all_iter1.log、r5_anchor_iter1/final.log、r5_stress_iter1/final.log、r51_mechanism_cmp.txt、sim_base_case1.log、sim_r51_case1.log、r51_diff.txt；失败变体 mha_r53_both.py / mha_r53_konly_FAILED.py）、`/tmp/opencode/r5_src_{base,new,final,rollback}/`（16 shape 生成码四份快照）、`/tmp/opencode/s5_analysis/`（trace 复核工具与基线读数）
- **本轮子任务未产出、后由 Orchestrator 补充执行（如实声明）**：cann-bench 复评报告 + 提交副本同步（会话 5 子任务刻意留旧，不以未验证分数声称成果；Orchestrator 独立复核全部门禁后执行，见 §S5-8）

## S5-8. cann-bench 权威复评（Orchestrator 补充执行，cann_final_eval_20260921_150725）

会话 5 子任务（perf-tuner）刻意未执行权威复评与提交副本同步，留待 Orchestrator 独立复核门禁后决策。Orchestrator 侧完成的独立验证：① `test_mha.py --level all` 独立复跑 exit 0（L0 10/10 + L1 19/19 + L2 12/12 + Boundary 6/6 + 哨兵 4/4，与子任务报告一致）；② 终态 md5/行数/备份链核对（`mha.py` 3d56c5dd 1086 行，`history_version/mha_impl_r3_perf9..12.py` 存在且 md5 对应）；③ 位级锚/竞态压力/机制对比原始日志抽查（`/tmp/opencode/r5_probes/r5_anchor_final.log`、`r5_stress_final.log`、`perf_tuning/simulator_r5/r51_mechanism_cmp.txt`，数字与子任务报告逐项吻合）。验证通过后同步提交副本并执行权威复评：

- 提交副本：`/mnt/workspace/cann/20260917_mha_V01/mha_submission/cann_bench/mha.py` 同步至终态 md5 **3d56c5dd**（旧副本 e80f7f65 备份 `/tmp/opencode/mha_submission_backup_before_r5_sync.py`），`run_evaluation.sh` 自动完成 build.sh 重打包 + pip 强制重装。
- 复评命令（与 r3/r4 会话同口径）：`run_evaluation.sh --source-dir mha_submission --task-dir tasks/level4/mha --operator MHA --device-id 0 --processes-per-card 1 --timeout-per-process 300 --warmup 3 --repeat 5 --profiler-level Level1 --perf-batch-case-memory-limit-mb 32768 --perf-batch-cases`（原始日志 `/tmp/opencode/r5_cannbench_final.log`）。

| 指标 | 会话 5 起点（=会话 4 终态） | 会话 5 终态 | 变化 |
|------|------------------------------|--------------|------|
| 综合得分 | 70.21/100 | **71.40/100** | **+1.19** |
| 性能分 | 20.21/50 | **21.40/50** | +1.19 |
| 精度分 / 编译运行分 | 30.00/30 + 20.00/20 | 30.00/30 + 20.00/20 | 保持 |
| 平均加速比 | 0.7813x | **0.8721x** | **+0.091 (+11.6%)** |
| 通过率 | 20/20 | 20/20 | 保持 |

per-case 变化（会话 4 终态 → 会话 5 终态，全部 20 case）：

| case | s4 | s5 | Δ | 说明 | | case | s4 | s5 | Δ | 说明 |
|------|-----|-----|------|------|-|------|-----|-----|------|------|
| 1 | 0.737 | **0.894** | **+0.157** | 目标项，#R5-1 直接受益 | | 11 | 0.925 | 0.927 | +0.002 | 噪声带 |
| 2 | 0.484 | 0.495 | +0.011 | 噪声带 | | 12 | 0.886 | 0.880 | -0.006 | 噪声带 |
| 3 | 0.768 | 0.763 | -0.005 | 噪声带 | | 13 | 0.883 | 0.869 | -0.014 | 噪声带（3%内） |
| 4 | 0.846 | 0.846 | 0.000 | 未触碰 | | 14 | 1.394 | 1.393 | -0.001 | decode 不回退约束 ✅（≥0.99） |
| 5 | 0.555 | 0.555 | 0.000 | 未触碰 | | 15 | 0.688 | 0.686 | -0.002 | 噪声带 |
| 6 | 0.584 | **1.083** | **+0.499** | 目标项，本轮最大单项收益，**首次破 1.0x** | | 16 | 0.784 | 0.789 | +0.005 | 噪声带 |
| 7 | 0.464 | 0.463 | -0.001 | 守卫，噪声带（Form A generic 路径未触碰） | | 17 | 1.138 | 1.143 | +0.005 | 噪声带 |
| 8 | 0.470 | 0.468 | -0.002 | 守卫，噪声带 | | 18 | 0.811 | 0.807 | -0.004 | 噪声带 |
| 9 | 0.324 | 0.325 | +0.001 | 守卫，噪声带（M1 case 线仍未达，非本轮范围） | | 19 | 0.873 | **1.352** | **+0.479** | 目标项 |
| 10 | 0.998 | 0.988 | -0.010 | decode 不回退约束 ✅（≥0.90），噪声带 | | 20 | 1.013 | **1.714** | **+0.701** | 目标项，本轮最大绝对提升 |

**核验**：4 个目标 case（1/6/19/20，均属"单块 Form A"类）全部大幅提升，其中 case6/19/20 首次或进一步突破 1.0x（超过 CANN 基线）；16 个非目标 case 全部在 ±0.014（远小于 3% 噪声带对应的绝对阈值）内浮动，无一实质回退；**decode 不回退硬约束全部核验通过**（case10 0.988≥0.90 ✅ / case13 0.869≥0.69 ✅ / case14 1.393≥0.99 ✅ / case16 0.789≥0.52 ✅ / case20 1.714≥0.82 ✅）；case7/8/9 只升不降约束：case9 +0.001 ✅，case7/8 -0.001/-0.002 在评测噪声带内（Form A generic 路径 IR 逐字节未变已由子任务 §S5-5 证明，此处权威复评交叉确认）。**回退条款不触发**（71.40 > 70.21 且 0.8721x > 0.7813x）。

- 评测报告：`.agents/skills/cann-bench/reports/cann_final_eval_20260921_150725.{json,md,html}`

---

# 会话 6（perf_session=6，2026-09-21，Expert rev3 轨第四轮；迭代 3/软预算 3-4，采纳 1 回滚 1 调查归档 1）

> 对象：Expert 版 mha.py（会话 5 终态 = #R5-1 采纳态，md5 3d56c5dd，1086 行，cann-bench 71.40 / 0.8721x）。
> 触发：用户提案「L0/L1 显式多缓冲 + MTE2 预取 Tile i+1 + 20 核手动深流水线」→ 本轮先以 **mode=analysis 只读可行性分析**收口（结论：提案主体已实现/已证伪——四路径形态均非单缓冲退化；L0 双槽已实现且 L0B 64KB 顶格不可加深；L1 双缓冲三实验证伪；"20 核深流水"仓内无逐字出处，候选=已实现的 Fixed Core+NUM_STAGES 批流水 或 rev4 §10.3-⑤ mma 化设计级建议；规则内幸存候选 = #R5-2 O 直写 / 批处理路径 L1 DB 低 EV 变体 / #R5-3 安全重试）→ Orchestrator 采纳排序并授权执行序（Iter-1 #R5-2 硬前置门探针 / Iter-2 幸存变体硬门 camodel / Iter-3 #R5-3 根因调查；禁重做已证伪项、禁 rev4 设计级、禁触 Form B decode）。
> 工作流同会话 3/4/5（skill 5 步 + 6 子步骤门禁 + IR 逐字节守卫 + camodel 机制复采）。

## S6-1. 最终成果（本地 msprof 口径；cann-bench 权威复评 = Orchestrator 轮末执行，本轮未跑）

| 指标 | 会话 6 起点（=会话 5 终态） | 会话 6 终态（#R6-1 采纳 + #R6-2 回滚还原） | 变化 |
|------|------------------------------|--------------------------------------------|------|
| kernel | 3d56c5dd（1086 行） | **bbfcbcaf1e438fa69664b4a7770e6d1c（1113 行）** | +27 行净变更 |
| msprof case1 | 11.26us | **10.06us** | **-10.66%** |
| msprof case6 | 29.91us | **24.51us** | **-18.05%** |
| msprof case19 | 23.46us | **21.53us** | **-8.23%** |
| msprof case20 | 18.49us | **14.32us** | **-22.55%** |
| 守卫 7 case | — | case2/5/7/9/10/14/16 全部 ±1.6% 噪声带内 | 零回退（IR 逐字节未变） |

**加速比外推（估算，非权威，方法同 §S5-2；警告同前——cann-bench 口径含每-launch 固定开销不等比缩放）**：case1 0.894→~1.00 / case6 1.083→~1.32 / case19 1.352→~1.47 / case20 1.714→~2.21，对 20-case 平均贡献 +0.048 ⇒ 平均 0.8721→**~0.920（估算）**。权威数字待 Orchestrator cann-bench 复评。

**decode 不回退硬约束核验（本地口径）**：case10 35.21→35.19 (-0.06%) / case14 377.97→376.28 (-0.45%) / case16 561.41→563.67 (+0.40%) / case20（属目标项，-22.55% 提升）——全部噪声带内或提升；**且 Form B 全域 12/12 守卫 shape 生成码逐字节未变（IR 级证明，强于计时守卫）**。case7/8/9 只升不降：case7 +0.67% / case9 -1.27%（均噪声带，Form A generic 与 Form B IR 未动）。

## S6-2. 会话 6 优化项与逐轮实测（msprof median us；完整门禁见 optimization_log.md 会话 6 节）

| 迭代 | 优化项 | case1 | case6 | case19 | case20 | 守卫 7 case | 精度 | 采纳 |
|------|--------|-------|-------|--------|--------|-------------|------|------|
| r6 基线 | —（=3d56c5dd 新鲜复测，与 §S5-8 记录差 +0.2~+2.1% = 跨会话方差带内 ⇒ 环境一致） | 11.26 | 29.91 | 23.46 | 18.49 | case2 67.29 / case5 18.39 / case7 923.98 / case9 1024.20 / case10 35.21 / case14 377.97 / case16 561.41 | 锚 10/10+压力 ✓ | — |
| **r6-1** | **#R6-1（=归档候选 #R5-2）单块 O 直写 FIX(L0C)→Output** | **10.06 (-10.66%)** | **24.51 (-18.05%)** | **21.53 (-8.23%)** | **14.32 (-22.55%)** | 7/7 ±1.6% 内 **且 12/12 守卫 IR 逐字节未变** | 硬前置门探针 8/8 + probe **30/30** + 锚双新鲜 10/10 + l0/l1/coverage/all + 压力继承 4+扩展 7 nondet=0 | **✅ YES** |
| r6-2 | #R6-2 Form A 单块批处理 q/k/v L1 双槽+批头/批内/批间预取（用户提案幸存变体） | 9.89 (-1.69%) | 24.65 (+0.57%) | 21.93 (+1.86%) | 14.04 (-1.96%) | ±2.5% 内（IR 未变） | probe 30/30 BITEXACT + IR 守卫 12/12 ✓ | ❌ 回滚（性能无变化：4/4 目标噪声带内且方向不一致） |
| r6-3 | #R6-3 #R5-3 根因调查（算子外最小复现，两级升级：单核紧凑源 / 20 核跨步 ND2NZ 源） | — | — | — | — | — | mha.py 零触碰 | 📁 **根因未定案归档**（最小复现两级均不复现 ⇒ 假设"空 MTE2 流 SetFlag×ND2NZ 失序"在最小结构不成立；需带跨核 SEM+V 并发上下文的消融法定案，属独立调查任务） |
| 收口 | 回滚 IR 级还原核验（16/16 逐字节）+ 终态新鲜锚/压力复跑 + 本报告 | — | — | — | — | — | **锚 10/10 + 压力 4/4 PASS（终态新鲜）** | — |

## S6-3. 会话 6 结构性发现（新增硬知识）

1. **L0C→BSHD 4D GM 运行期行切片 FIX 直写 = 可用且位级安全（前置门探针 8/8）**：`copy_l0c_to_gm` 的 realDstN/realTailM/realTailN 运行期形参原生支持（src/op/ascend.cc:531-543 + common.h:190-213 源码查证）；整数精确模式 BITEXACT（尾块 26/30/62 行、fp16/bf16、D=64/128、sentinel 无漏写、邻块无越写）；fixpipe fp32→dtype RNE ≡ V 侧 vcast 再获 19/19 实证（#R3-3 知识的输出侧推广）。**仓内先例空白已补齐**（fusedmoe/lightning_indexer 均为整块或编译期行界）。
2. **单块特化的输出侧结构红利（#R6-1 收益模型）**：预归一化特化使 O = P̂@V 一步终值 ⇒ 第四相位（V-epi）纯搬运可整段删除：省 w3 fp32 GM 写+读往返（每槽 2×[bM,D]fp32）、省 V phase-2 vector/MTE3 段、省 WS3 SEM 对（稳态 cross-core op 15→11）、Output 直写字节减半（fp32 w3 → dtype 直写）。camodel：V wall -13.6%、V FLOWCTRL -46.6%。**代价如实记录：C 核接管 Output FIX 后成为新 pacing 侧（case1 C wall +3.4%）**——输出侧优化把墙移到了 C 侧 FIX/GM 通道。
3. **camodel 与真机的方向背离（#R6-2 判决性知识，方法学级）**：双槽预取使 camodel case6 C wall -19.0%、FIXP union -24.4%（预取聚簇改变其 GM 争用记账），但真机 msprof +0.57% 零响应。camodel FIXP sum（29.8us）> 真机全 wall（24.65us）⇒ 其 fixpipe 绝对时长模型显著慢于真机。**固化纪律：camodel 仅用于"机制是否激活"定性判定（本轮 MTE2 段结构 22→20 段、气泡消除 = 激活证明），禁止用于收益定量；msprof 为唯一性能依据（§7 纪律的判决性再实证）。**
4. **case6 归因补全（analysis 阶段数据缺口）**：#R6-1 态 case6 C 侧 **FIXP 92.38% = pacing 管**（与 case7 95.68% 同构）、V 侧 FLOWCTRL 79.27% ⇒ 单块批处理族的 C wall 已是 fixpipe 吞吐墙；MTE2 预取类（不减字节）对其无杠杆。**#R6-2 的零响应不是实现缺陷（机制已激活 + 位级安全），是字节总量墙的必然**。
5. **「MTE2 预取 / L1 显式多缓冲」提案族的完整实测收口（跨会话证据链闭合）**：Form A generic（#R3-2 +0.5%）/ Form B D256（#R4-2b 全 pipe 逐位不变）/ Form B decode（带宽墙+L1 预算否决）/ **Form A 单块批处理（#R6-2 机制激活+真机零响应）**——四路径形态全部实测证伪或排除。输入侧唯一剩余杠杆 = 冗余字节消除（#R5-3 域）；其最小复现两级未定案（S6-2 r6-3），重试前置门已升级（消融法，需独立调查立项）。
6. **令牌守恒的可复用新模式（#R6-2 机制入库）**：双槽 L1 + 三预取点（批头 kb==0 / 批内 i+1<batch_tasks / 批间 kb+1<num_tb）的记账恒等式 #预取(slot s)=#消费(slot s)=总槽数，init/destroy 每槽 1 令牌（7 vs generic 4）；parse-time 条件化（三元式分配 + SIG 重排 + init/destroy 分支）保 generic/Form B IR 逐字节不变；30/30 BITEXACT——未来若瓶颈构成改变（如 FIX 流量削减后 MTE2 成为新墙）可零探针成本重试。

## S6-4. 精度最终状态（收口全量，终态 md5 bbfcbcaf 上新鲜复跑）

- `--level all` exit 0 "Test Passed!"：**L0 10/10 + L1 19/19 + L2 12/12 正确拒绝 + Boundary 6/6 + 哨兵 4/4 PERF_PASS**（kernel_path 6.0~7.3s ≤ 120s 门；r6_all_iter1.log，跑于 #R6-1 采纳态 = 终态同体）；覆盖门禁 **19 PASS / 0 MISS / 0 N/A**（r6_coverage_iter1.log）。
- 位级诊断锚：**10/10 BITEXACT vs Developer 终态**（改前基线 r6_anchor_base.log + 改后 r6_anchor_iter1.log + **回滚收口终态新鲜复跑 r6_anchor_final.log**，三次全零失配）。
- R11 竞态压力：继承 4 配置 ×12 **nondet=0** + 扩展 7 配置（打在被改路径：batch_tasks {1,2,4,5,6,7,8}、跨轮 8+5、P1/P2、双 dtype、causal、双尾块、my_count 非齐次）×12 **nondet=0**（r6_stress_ext_iter1.log；终态复跑 r6_anchor_final.log 含继承 4 配置 PASS）。
- 采纳项 #R6-1 验收：前置门探针 8/8（算子外独立 kernel，整数精确 BITEXACT）+ probe_r61 **30/30**（变更域 19/19 对改前 BITEXACT + golden 阻塞口径全过，最差 MARE_abs 1.95e-3 vs gate 7.81e-2 = 40x 余量；case20 zeros 位级 0.0；守卫 11/11 BITEXACT）。回滚项 #R6-2 探针 30/30 BITEXACT（位级安全但性能无变化，采纳判据纯由性能规则驱动——#R3-2/#R4-2b 同款纪律）。
- 精度红线四项复核（终态）：双舍入栅格链**逐字未动**（phase-1 内）✅；单块预归一化特化保留（且为 #R6-1 等价性依据）✅；sumexp+1e-30 保护保留 ✅；掩码链语义等价（未触碰）✅。-0.0 边界：#R6-1 删除了 `0.0+w3` 规范化——整数精确/zeros/大幅值域/随机 30 shape 位级比较未触发（域内不可达论证归档 [#R5-2]/analysis §六）。
- IR 级守卫（16 shape）：**12/12 守卫逐字节相同 + 4/4 目标按预期改变**（#R6-1）；#R6-2 回滚后 **16/16 逐字节还原**（diff r6_src_new vs r6_src_rollback 无差异）。ruff check/format clean（1113 行终态）。

## S6-5. 遗留问题与建议（会话 6 终态评估）

| 候选/case | 状态 | 结论 / 建议 |
|-----------|------|-------------|
| **case1/6/19/20（单块 Form A）** | ✅ #R6-1 再削减 -8.2~-22.6% | 输出侧红利已兑现；case6 新墙 = C FIXP 92.4%（w1 fp32 写为主）⇒ **下一杠杆 = w1 dtype 瘦身重试**（#R3-3 曾零响应于 case7 耦合平衡态，但 case6/1/19/20 现处 FIX 吞吐墙下、瓶颈构成已变——RNE 等价 11/11 + 本轮 19/19 再证，探针全免，改动 ~2 行 + ws 形状/dtype 派生；风险 = 位级锚失配预案需按 #R3-3 归档口径复核） |
| **#R6-2 双槽预取** | ❌ 证据性放弃（本轮） | 机制激活+位级安全+真机零响应（字节总量墙）；机制知识入库（S6-3-6），若未来 FIX/w1 流量削减使 MTE2 成为新墙可零探针重试 |
| **#R5-3 K/V 冗余消除** | ⏸ 根因未定案（两级最小复现不复现） | 冗余仍是最大输入侧事实（case6 一个批内 7/8 同字节）；重试前置门升级 = 以 mha.py 为基座的**消融法调查**（删 V-scope/删 SEM/删 GEMM2 逐项降级定位最小致损因子组合），属独立调查任务非调优迭代 |
| **#R5-4 NUM_STAGES 12/16** | ⏸ 暂缓（维持 §S5-6 判定） | case6 折单轮收益 ~1-3%（轮边界停顿）vs 全 shape IR 变化 + ws 足迹×2 + 全量复测成本；#R6-1 后 case6 墙在 FIXP，批深不解决 |
| case9（0.325x）/ case7/8（0.46x） | 维持会话 4/5 结论 | 规则内穷尽；破局 = §10.3-⑤ Form B→A mma 化 / kv-outer 互换 = **rev4 设计级，需用户立项**（analysis 报告 §四-6 证据已备齐） |
| cann-bench 权威复评 + 提交副本同步 | ⬜ **Orchestrator 轮末执行**（沿用会话 5 分工，本轮刻意未跑、未同步） | 终态 md5 bbfcbcaf；备份链/门禁日志全在案可独立复核 |

**结论**：会话 6 净收益 = **#R6-1 单块 O 直写**（4/4 目标 case -8.2~-22.6% 本地 msprof，估算平均 0.8721→~0.92，权威待复评；变更面 = C 批分支 GEMM2 输出段 + emit_v phase-2 parse-time 包裹 + docstring，Form B/generic IR 逐字节未动）。用户提案按 analysis 结论执行完毕：幸存变体 #R6-2 实测收口（回滚归档，提案族四路径全部证伪/排除）、#R5-3 根因调查两级未定案（按授权归档不上 kernel）。中止条款②未触发（1 采纳 + 1 性能无变化回滚 + 1 调查归档）；按软预算纪律收口。

## S6-6. 会话 6 产物清单

- kernel: `examples/mha/mha.py`（**1113 行，md5 bbfcbcaf1e438fa69664b4a7770e6d1c**；会话 6 净变更 = #R6-1：C 侧 Form A 批分支 GEMM2 尾 Output 直写（+bx/s0/valid_m 派生、删 WS3 wait/set）+ emit_v 批分支 phase-2 `if not form_a:` 包裹 + docstring；#R6-2 已 IR 级完整还原）
- `perf_tuning/`: `msprof_r6_base/`（11 case 基线）、`msprof_r6_iter1/`（11 case 采纳态）、`msprof_r6_iter2/`（11 case 回滚项）、**`simulator_r6/`**（README + case1_r61 + case6_r6base + case6_r62 + r61_mechanism_cmp.txt + case6 前后 tracestats）、optimization_log.md（会话 6 S6-Step1~4 + 迭代 r6-1/r6-2/r6-3 门禁全记录）、perf_log.md（Iteration r6-1/r6-2/r6-3 条目见下）、本文件 §S6
- `history_version/`: `mha_impl_r3_perf13.py`（=改前基线 3d56c5dd）、`mha_impl_r3_perf14.py`（=#R6-1 采纳态 = 终态 bbfcbcaf）
- 探针与证据（scratch，/tmp/opencode/r6_probes/）：probe_r52_l0c_bhsd.py（硬前置门 8/8）、probe_r61_o_direct.py（30/30）、probe_r62_kvqdb.py（30/30）、repro_r53.py / repro_r53_v2.py（最小复现两级）、dump_src.py（16 shape IR）、日志 probe_r52.log、probe_r61.log、probe_r62.log、r6_anchor_{base,iter1,final}.log、r6_l0/l1/coverage/all_iter1.log、r6_stress_ext_iter1.log、repro_r53{,_v2}.log、msprof_r6_*.log、sim_r6*.log、smoke_case1_r62.cc；IR 快照 /tmp/opencode/r6_src_{new,iter2,rollback}/
- 评测报告: ✅ **已由 Orchestrator 补充执行**（见 §S6-7）：`.agents/skills/cann-bench/reports/cann_final_eval_20260922_100108.{json,md,html}`

## S6-7. cann-bench 权威复评（Orchestrator 补充执行，cann_final_eval_20260922_100108）

会话 6 子任务（perf-tuner）沿用分工纪律刻意未执行权威复评与提交副本同步，留待 Orchestrator 独立复核门禁后决策。Orchestrator 侧完成的独立验证（全部通过后才同步副本并复评）：

1. **`test_mha.py --level all` 独立复跑 exit 0 "Test Passed!"**（L0 10/10 + L1 19/19 + L2 12/12 正确拒绝 + Boundary 6/6 + 哨兵 4/4 PERF_PASS；`/tmp/opencode/r6_orchestrator_all_rerun.log`，与子任务 r6_all_iter1.log 结论一致）。
2. **终态 md5/备份链核对**：`mha.py` 1113 行 md5 `bbfcbcaf` = `history_version/mha_impl_r3_perf14.py`；`mha_impl_r3_perf13.py` = 改前基线 `3d56c5dd`。
3. **覆盖门禁独立复跑**：`coverage_check.py --proto` = 19 PASS / 0 MISS / 0 N/A。
4. **原始日志抽查**：`r6_anchor_final.log`（位级锚 10/10 BITEXACT + 继承压力 4 配置 nondet=0 PASS）、`simulator_r6/r61_mechanism_cmp.txt`（V wall -13.6% / V FLOWCTRL -46.6% / V MTE3 -62.9% / C wall +3.4% / softmax 段长不变——数字与子任务报告逐项吻合）。
5. **IR 守卫独立复核**：`r6_src_new` vs 会话 5 终态快照 `r5_src_final`：12/12 守卫逐字节一致 + 4/4 目标（case1/6/19/20 单块 Form A）按预期改变；`r6_src_new` vs `r6_src_rollback`：16/16 逐字节一致（#R6-2 回滚 IR 级还原确证）。

**副本同步与复评**：提交副本 `/mnt/workspace/cann/20260917_mha_V01/mha_submission/cann_bench/mha.py` 由旧态 `3d56c5dd`（备份 `/tmp/opencode/mha_submission_backup_before_r6_sync.py`）同步至终态 `bbfcbcaf`；`run_evaluation.sh` 自动完成 build.sh 重打包 + pip 强制重装（安装副本 md5 校验一致 `bbfcbcaf`）。复评命令与会话 3/4/5 同口径（`--device-id 0 --processes-per-card 1 --timeout-per-process 300 --warmup 3 --repeat 5 --profiler-level Level1 --perf-batch-case-memory-limit-mb 32768 --perf-batch-cases`，原始日志 `/tmp/opencode/r6_cannbench_final.log`）。

**权威结果**：综合 **71.40 → 71.70**，性能分 21.40 → **21.70**，平均加速比 **0.8721x → 0.9054x（+3.8%，M2 平均线 0.9x 首次达成 ✅）**，20/20 pass。

| case | S5 | S6 | Δ | 备注 |
|------|----|----|---|------|
| case1 | 0.894 | **0.943** | +0.049 | #R6-1 目标族 |
| case6 | 1.083 | **1.283** | +0.200 | #R6-1 目标族（主受益） |
| case19 | 1.352 | **1.426** | +0.074 | #R6-1 目标族 |
| case20 | 1.714 | **2.079** | +0.365 | #R6-1 目标族 |
| case4 | 0.846 | 0.828 | -0.018 | **IR 守卫逐字节未变集成员**（G_case4_B_p3_kv2）⇒ 内核因果被 IR 级排除，判定测量方差（略超历史 ±0.014 带，如实记录） |
| case5 | 0.555 | 0.538 | -0.017 | 同上（G_case5_A_cross_kv2） |
| 其余 14 case | — | — | ±0.009 内 | 噪声带内零回退 |

**decode 不回退硬约束全达标**：case10 0.983（≥0.90）/ case13 0.886（≥0.69）/ case14 1.394（≥0.99）/ case16 0.785（≥0.52）/ case20 2.079（≥0.82）。**回退条款不触发**（71.70 > 71.40 且 0.9054x > 0.8721x）。case9 0.324x 仍为 M1 case 线（0.4）唯一未达项——维持会话 4/5 结论：规则内穷尽，破局 = rev4 设计级 §10.3-⑤（需用户立项）。

**收尾补记（2026-09-22，用户指令）**：按用户提供的 MindStudio 采集命令示例（`msprof op simulator --soc-version=Ascend910B4`→本环境实际 `msopprof simulator --soc-version=Ascend910_9362`，Orchestrator 直接执行，非调优迭代），在**当前终态 kernel（bbfcbcaf，采集前 md5 已录日志）**上新鲜采集 case9 指令级流水数据：`mindstudio_export/case9_simulator/OPPROF_20260922102656_CMHAXPPYVZSKKAQH/`（simulator/ 60 子核 trace + merged trace.json + dump/ 3 文件；24GB camodel 中间 `*.dump` 日志按 case1/会话 4 既定导出约定清理，清理后 7.6GB；SIM_EXIT=0，耗时 25m19s）；转移用 tarball `mindstudio_export/mha_case9_simulator_20260922.tar.gz`（629MB，248 entries，gzip 完整性校验 OK，root=OPPROF 目录与 case1 约定一致）。数据一致性：case9 Form B D256 路径 IR 在 #R5-1/#R6-1 两轮均逐字节未变（G_case9_B_p3_multi ∈ 两轮 12/12 守卫集）。采集命令/环境/分析导引详见 `mindstudio_export/case9_simulator/README.md`；用途 = rev4 §10.3-⑤ Form B→A mma 化立项评审的流水依据。

---

# 会话 7（perf_session=7，2026-09-22/23，rev4 轨首轮；迭代 2/软预算 3-4 + 1 评估归档，采纳 0 回滚 2）

> 对象：rev4-attempt1 基线 mha.py（1593 行，md5 ee5bf9c9，cann-bench **71.79 / 0.9108x**，cann_final_eval_20260922_172117；三分派：form_a IR 未变 / form_b_mma 新增 mma 流水 / form_b_gemm_v0 decode 兜底禁触）。
> 触发：case9 mma 化后 0.324→0.353（+9%）但远未达设计外推 ~0.46x ⇒ **Step 0 = case9 新鲜归因（首位任务）**，随后按归因执行规则内候选（授权序：①w1 瘦身重试 ②§10.3-⑨ O 直写扩展评估 ③归因驱动新候选）。
> 执行中断说明（如实）：首次执行在 Step 0 采集完成后被外部中断，orchestrator 盘点核实后续跑（kernel 零改动、采集数据复用、未重复采集）。

## S7-1. 最终成果

| 指标 | 会话 7 起点（=rev4 S1） | 会话 7 终态 | 变化 |
|------|--------------------------|-------------|------|
| kernel | ee5bf9c9（1593 行） | **ee5bf9c9（1593 行，两次回滚后 IR 级 16/16 还原核验）** | 零净变更 |
| cann-bench | 71.79 / 0.9108x | （终态=起点，复评由 Orchestrator 轮末统一执行） | — |
| 知识产出 | — | **case9 pacing 结构判定完成（双侧 A/B 排除法）+ 两个候选族证据性关闭 + 会话 8 移交候选完成 scoping** | — |

**如实结论：本会话无性能净收益**——两次迭代（w1 瘦身 / k/v 双槽）精度门禁全过、机制均实测激活，但真机 wall 均零响应（+0.31%/+0.35%），按"性能无变化→回滚"规则回滚。其价值为**判决性归因**：case9 的剩余 wall 结构被双侧 A/B 唯一确定（见 S7-3-2），剩余规则内杠杆收敛至单一候选（flat 批处理）并完成移交 scoping。

## S7-2. 会话 7 优化项与逐轮实测（msprof median us；完整门禁见 optimization_log.md 会话 7 节）

| 迭代 | 优化项 | case9 | case4 | 守卫 10 case | 精度 | 采纳 |
|------|--------|-------|-------|--------------|------|------|
| r7 基线 | —（=ee5bf9c9 同会话采集；IR 未变域与会话 6 记录差 ±2% 内 ⇒ 环境一致） | 944.61 | 35.32 | case6 24.77 / case1 10.25 / case19 21.85 / case20 14.51 / case7 926.60 / case2 65.46 / case5 18.56 / case10 34.98 / case14 376.16 / case16 560.96 | 锚A 9/9+锚B 14/14（Stage-2 态复核） | — |
| r7-1 | #R7-1 w1 dtype 瘦身（form_b_mma 专属 scope，§10.3-① 重试） | 947.50 (+0.31%) | 35.73 (+1.16%) | ±2.5% 带内（IR 13/13 逐字节） | 探针 18/18 BITEXACT + 全门禁 ✓ | ❌ 回滚（无变化） |
| r7-2 | #R7-2 k/v L1 双槽（form_b_mma，§10.3-③ 重试窗口） | 947.96 (+0.35%) | 36.23 (+2.58%) | -3.0~+2.5% 带内（IR 13/13 逐字节） | 探针 18/18 BITEXACT + 全门禁 ✓ | ❌ 回滚（无变化） |
| r7-3 | 剩余候选评估归档（无 kernel 变更） | — | — | — | — | 📁 flat 批处理移交会话 8 |
| 收口 | 回滚 IR 还原核验 ×2（16/16 逐字节）+ 终态新鲜锚复跑 | — | — | — | 锚 10/10 + 压力 PASS | — |

## S7-3. 会话 7 结构性发现（新增硬知识）

1. **S1 收益机制真机实证 + case9 瓶颈迁移定量**（op 模式 old gemm_v0 → new mma 化）：scalar_mte1_stall **739→412us（-44%）** = 每-step PIPE_M 串行链消除的直接签名；wall 1045.7→955.4us（-8.6%，与 cann-bench +9% 一致）；C wait_id2 151→324us（+114%）= C 变快后更早到达 w2 交接点。**未兑现 case7 同族 0.46x 的结构原因**：D256 形态 GM 流量强度 4.25KB/输出行（case7 的 3.4×：K/V [128,256] 各 64KB/iter 配 bM=32）+ PLAN-B colseg FIX 税（事件 2→3/iter，真机 fixp 251→298us）。
2. **case9 pacing 结构判定完成（双侧 A/B 排除法，本会话核心产出）**：r7-1 加速 V-P1（vec -2.8pp/vector_stall -37%）→ C mte1_stall +33us 吸收，wall +0.31%；r7-2 加速 C-G1（**mte1_stall -54% 靶心命中**）→ C wait_id2 +221us（+68%）**1:1 吸收**，wall +0.35%。两侧各有 ≥200us 级等待余量 ⇒ **wall = 四相位每槽交接延迟链**（每槽 4 次跨核 SEM 传播+管道排空的不可重叠段 × 461 iter/核，causal eff_kv 均值 4.5 ⇒ 批深仅 4.5 槽、每 4.5 iter 一次任务边界，NUM_STAGES=8 半空转摊薄不足）。**"耦合平衡吸收律"获得迄今最强定量证据：单侧节省 = 对侧等待增量，1:1**。字节类/速度类/单侧类手段在该结构下数学上无杠杆。
3. **两个候选族证据性关闭**：①w1 dtype 瘦身（§10.3-①）——双 régime 零响应（会话 3 gemm_v0 时代全域 7 shape + 本轮 mma 化后 form_b_mma 域），重试前置 = 某 bench case 真机 pacing 管为 fixp/FIX 字节（当前无一命中：case6 28.8%/case7 30.8%/case9 31.6-33.4%，camodel 的 92-95% FIXP 为模型放大——§S6-3-3 三度实证，本轮 case6 真机 op 首采为判据）；②k/v L1 双缓冲（§10.3-③）——五环证伪链闭合（Form A generic #R3-2 / Form B gemm_v0 #R4-2b / decode 带宽+预算 / Form A 单块 #R6-2 / Form B-mma 本轮），**全路径形态穷尽**。RNE 位级等价知识三次实证（11/11+19/19+12/12）与 3D×ZN×列段 transpose 组合 BITEXACT 探针均入库免重试成本。
4. **§10.3-⑨ O 直写扩展 = bench 收益面为零（论证归档）**：form_b_mma ∧ single_kv_block 在 20 case 无命中（case4 kv_iters=2 / case9 多块 / case15/18 decode 兜底禁触；唯一命中 = L0 #10/哨兵）——DESIGN "case4 族"预期为设计期疏漏，如实修正。
5. **case7 真机 op 首采（守卫域归因补全）**：C wait_id2 = 617us = 70% wall（C 大量余量吸收 V 侧任何节省——#R3-3 会话 3 零响应的机制解释闭环）；V vec 68.3% + vector_stall 258us = case7 的 V softmax 链为最大真实工作管。case7 域规则内手段维持穷尽判定。
6. **rev4 门禁体系全链路首次实战验证**：锚-A 9/9 硬守卫 + 锚-B 14/14（两次改动均 BITEXACT——"锚-B 必失配"预案未触发，RNE/纯调度类等价成立时锚-B 保持诊断力）+ 13 守卫 IR 逐字节 ×2 轮 + rev4 扩展压力 16 项（SIG_L0KC/L0AO/L0AS token 流专打）×2 轮 nondet=0 + disable_cache 全程（会话专属 TILELANG_CACHE_DIR）——**两次高危改动（dtype 载体迁移 / 令牌 id 重排+双槽）均零缺陷通过全门禁后仅因性能规则回滚**，门禁链的缺陷拦截力与回滚还原力（IR 16/16 ×2）双向验证。

## S7-4. 精度最终状态（收口，终态 = ee5bf9c9 = 会话起点，Stage-2 全门禁已在案）

- 本会话终态与 Stage-2 rev4-attempt1 逐字节同体（两次回滚 IR 级还原核验 16/16 逐字节 diff ir_new2 无差异 ×2）⇒ Stage-2 全量精度门禁（[PRECISION_PASS] + orchestrator 二次校验）对终态有效，无失效风险。
- 收口新鲜复跑（本轮终态上）：位级锚 **10/10 BITEXACT** vs Developer 终态 + R11 继承 4 配置压力 nondet=0 PASS（r7_anchor_final5.log，默认缓存；**注**：首跑 r7_anchor_final.log 曾出现 #10 DIFF 1.3e-1——根因为默认缓存中 Stage-2 时代毒条目与当前源同 key 冲突（rev4 已归档"缓存键不覆盖 annotation 级变更"缺口的实际发生学证据），清除后复跑 10/10；bench case9/case4 测量有效性经条目 md5 逐字节核验不受影响，全链证据见 debug_log.md 会话 7 补录与 optimization_log.md 收口补录）。
- 两轮迭代过程中的完整门禁记录（探针/双锚/l0/l1/coverage/all/压力/IR）见 optimization_log.md [RESULT-#R7-1]/[RESULT-#R7-2]——全部通过，回滚判据纯由性能规则驱动（#R3-2/#R4-2b/#R6-2 同款纪律）。

## S7-5. 遗留问题与建议（会话 7 终态评估）

| 候选/case | 状态 | 结论 / 建议 |
|-----------|------|-------------|
| **case9（0.353x，M1 case 线 0.4 未达）** | **pacing 结构判定完成；规则内剩余杠杆唯一化** | **flat (task,kv) 批处理（Form B-mma generic 域）= 会话 8 首位候选**：批深 4.5→8 + 任务边界 102→~58/核（V task-tail 关键路径段削减）；EV -5~-15%（上界依据：C 轮界等待 87us + V task-tail 段）；⚠ 爆炸半径 = emit_v 三形态共享 ⇒ 全 20 case 复测 + 失去 IR 守卫，需独立会话全额预算；实施蓝图 = #R4-3 机制（17/17 BITEXACT 归档）+ §S4-3-5 状态复位时序约束 + 本轮 A/B pacing 判定 |
| w1 瘦身 / k-v L1 DB | ❌ 证据性关闭（双 régime / 五环链） | 重试前置条件已写入 §S7-3-3；机制知识（RNE 等价 ×3、3D×ZN×colseg 探针）入库免成本 |
| §10.3-⑨ O 直写扩展 | 📁 论证归档不实施 | bench 收益面 = 0（§S7-3-4）；L0 #10 域无 wall 意义 |
| CROSS_INTERVAL / w3 plan-a / NUM_STAGES | 不启动 | 禁令类（wait 削减）/ EV≈0+崩溃史 / case9 单轮无意义（维持既有判定） |
| case7/8（0.46x 族） | 维持穷尽判定 + 归因补全 | C wait_id2 617us=70% wall（V 链为最大真实工作管）；kv-outer 维持 §0.5 降级（本轮无推翻其前提的新证据——case7 域未测改动） |
| decode 硬约束 | ✅ 全程 IR 级成立 | 两轮 13/13 守卫逐字节（含 decode ×5）+ 兜底域源码零触碰；msprof 守卫 case10/14/16 全带内 |
| cann-bench 复评 + 提交副本 | ⬜ Orchestrator 轮末统一执行（沿用分工） | 终态 = ee5bf9c9（与复评基线同体，预期分数不变；复评可作为环境漂移校验） |

**结论**：会话 7 = **归因会话**（零净变更、双回滚、判定完备）。case9 的 mma 化收益（+9%）已兑现并确认机制（mte1_stall -44%）；其剩余差距被双侧 A/B 唯一归因至四相位交接延迟链——字节类与速度类手段在该结构下 1:1 被吸收（吸收律定量实证），规则内唯一剩余杠杆（flat 批处理）因共享 emit_v 爆炸半径需独立会话，已完成 scoping 移交。中止条款未触发（连续 2 次无提升 < 3）；按预算纪律收口，未强行耗尽。

## S7-6. 会话 7 产物清单

- kernel: `examples/mha/mha.py`（**1593 行，md5 ee5bf9c93f979b9fc5b111a3c4121888 = 会话起点 = 终态**；两轮改动均已 IR 级完整还原）
- `history_version/`: `mha_impl_r4_perf1.py`（=开工基线 ee5bf9c9）、`mha_impl_r4_perf2.py`（=r7-1/r7-2 改前备份，同体 ee5bf9c9；两轮均回滚至此，无采纳态快照新增）
- `perf_tuning/`: `msprof_r7_base/`（12 case 同会话基线）、`msprof_r7_iter1/`（r7-1）、`msprof_r7_iter2/`（r7-2）、**`simulator_r7/`**（case9_r7s1 = rev4 S1 态 camodel：merged 2.6GB + core0 三核 trace/instr_exe + msopprof.log；对比基线 = simulator/case9 会话 4 旧结构数据）、optimization_log.md（会话 7：[ANALYSIS] Step 0 + S7-Step4 两轮门禁全记录 + S7-Step5 归因综合）、perf_log.md（Iteration r7-1/r7-2/r7-3）、final_report.md §S7（本节）
- 归因数据（scratch）：/tmp/opencode/r7_op_case9/（S1 态 op 归因）、r7_op_case6/ + r7_op_case7/（守卫域真机首采）、r7_op_case9_w1slim/ + r7_op_case9_kvdb/（两轮 A/B）、r7_probes/（probe_r71_w1slim.py 18/18、probe_r72_kvdb_mech.py 机制探针、probe_r72_kvdb.py 18/18、repro 工具、全部日志）、r7_src_{iter1,iter2,rollback,rollback2}/（IR 快照 ×4）
- 失败/回滚变体留存：/tmp/opencode/r7_probes/mha_r71_edit_v1.py（w1 瘦身态 1633 行）、mha_r72_edit_v1.py（k/v 双槽态 1624 行）——均全门禁通过、仅性能规则回滚，未来结构变更后可低成本复活
- 评测报告: ⬜ Orchestrator 轮末统一执行（本轮未跑——分工纪律）

---

# 会话 8（perf_session=8，2026-09-29/30，rev4 轨大 prefill ①+②+③ 联合流水专项；迭代 3/软预算 6 + 1 评估归档 + 收口，采纳 2 回滚 1；含外部中断后恢复处置）

> 对象：rev4 终态 mha.py（1593 行，md5 ee5bf9c9，cann-bench 71.79 / 0.9108x）。
> 任务：① flat(task,kv) 连续令牌流 + ② C 侧 L0B 4 槽旋转 + ③ V 侧批相干宽通道三项联合（原 B 层 626us/0.54x 级目标 = 模型推算，msprof 实测唯一裁决）。
> **中断与恢复（如实）**：前次执行在 r8-3 实施+门禁完成、采纳判定未落档时被外部中断；恢复会话以新上下文接续，Step A 溯源核验（当前 mha.py md5/行数/ast/ruff 与事实基座一致 + 新鲜 dump 16/16 逐字节 = 门禁参照 r8_src_iter3（provenance 成立 ⇒ 门禁日志对当前文件态有效）+ 备份链 perf1-7 md5 核对一致 + probe 两项脚本级伪影对账复核）后**最大化复用已验证工件完成 r8-3 采纳判定、Step C 联合裁决、Step D 条件评估与收口，零已完成工作重做**。

## S8-1. 最终成果

| 指标 | 会话 8 起点（rev4 终态） | 会话 8 终态 | 变化 |
|------|---------------------|-------------|------|
| kernel | ee5bf9c9（1593 行） | **dda3ad08（1926 行）= r8-3 采纳态 2728c96e（1900 行）+ codegen-neutral docstring rev4r8 注记（IR 16/16 逐字节复核）** | ① 全域 flat 落地 |
| msprof 实测（同会话，median） | — | **① 目标域 7/7 收益：case2 -33.08% / case3 -21.60% / case4 -20.66% / case5 -18.13% / case9 -8.09% / case7 -7.36% / case8 -6.26%；守卫 13/13 带内零回退** | 联合态全 20 case |
| cann-bench | 71.79 / 0.9108x | （复评由 Orchestrator 轮末统一执行） | 估算 ~0.94-0.95x 级（**非权威**，固定开销不等比缩放警示 §S5-2） |
| mission 承诺兑现 | ①+②+③ 联合 → B 层 626us（0.54x） | **净产出 = ① 全域落地 + 实测收益；② 证据性关闭（r8-1）、③ 评估归档移交会话 9；0.54x 承诺维持 r8-1 降级口径，本会话全部按实测报告、不引用为已兑现值** | 如实声明 |

## S8-2. 会话 8 优化项与逐轮实测（msprof median us；完整门禁见 optimization_log.md 会话 8 节）

| 迭代 | 优化项 | 目标域实测 | 精度 | 采纳 |
|------|--------|-----------|------|------|
| r8 基线 | —（ee5bf9c9 同会话新鲜采集 msprof_r8_base） | case9 944.74 / case4 35.57 / case7 924.69 / case10 34.86 / case14 377.69 / case16 562.96 | — | — |
| r8-1 | #R8-1 ② L0B 4×16KB 槽旋转（fbm 域，同取向 v4ksp 形态 + load-first + id 矩阵 7/8；附.3 字面双取向形态因 T.view 损坏 + 取向冲突不可表达，探针实证归档） | case9 946.93（**+0.23% 零响应**）；op 判据 (a) FAIL：mte1_stall 434→425（-2% 非显著）且 stall>busy 签名存续 | 门禁全过（探针 20/20 BITEXACT、锚 9/9+14/14、IR 13/13 守卫、压力 4+16×12 nondet=0、l0/l1/coverage/all） | ❌ 回滚（机制判据失败 + 代价净值负；变体留存 mha_r81_edit_v1.py 全门禁通过态） |
| r8-2 | #R8-2 ① flat(task,kv) fbm generic 域（顺序游走状态机映射运行期非均匀 causal eff_kv，C/V 双侧同式；V 任务状态机 §S4-3-5） | **case9 944.74→869.51（-7.96%，mission EV 带兑现）/ case4 35.57→28.20（-20.72%）**；op 靶心：C wait_id5 87.6→2.2（-97.5%）、V wait_id4 -42.6%、真实工作管全平 = 纯结构收益 | 门禁全过（探针 23/23 含 16/16 flat 域 BITEXACT、锚 9/9+14/14、IR 14/16+2/2、压力 4+22×12、l0/l1/coverage/all） | ✅ 采纳（perf6 = 5d466b5f） |
| r8-3 | #R8-3 ① flat form_a 多块域扩展（case7/8/3/2/5；两项 Form A delta：l0a_s 重载条件 + q_l1 归还任务末槽竞态修复——设计期静态推演捕获） | **case7 926.04→856.62（-7.50%）/ case3 167.71→131.48（-21.60%）/ case2 67.11→44.91（-33.08%）/ case5 18.81→15.40（-18.13%）**；op 靶心复现：wait_id5 -95.2%、V wait_id4 -50.3%、cube +0.00% | 门禁全过（探针 15/15 BITEXACT + 守卫 7/7 + 两项脚本级伪影独立复核归档、锚 9/9+14/14、IR 12/16+4/4 · vs 起点 10/16 含 decode×5、压力 4+26×12、l0/l1/coverage/all；**恢复会话 provenance 16/16 核验后判定落档**） | ✅ 采纳（perf8 = 2728c96e） |
| r8-4 | Step C 联合裁决（全 20 case）+ Step D 条件评估 + 收口（评估归档，无 kernel 结构变更） | 联合表：7/7 目标 -6.26%~-33.08%、13/13 守卫零回退（最差 case20 +1.75% = IR 逐字节域 +0.25us 会话方差）；**case8 实测 -6.26%（禁外推条款履行）**；decode 线 case10 +0.86%/13 +0.29%/14 -0.70%/16 +0.05%/20 +1.75% 本地口径无回退信号 | 终态新鲜门禁子集全过（锚-A 9/9 + 锚-B 14/14、继承压力 4×12 nondet=0、l0 10/10、all exit 0：L1 19/19 + L2 12/12 拒绝 + Boundary 6/6 + 哨兵 4/4，PRECISION_FAIL=0）+ 16/16 IR codegen-neutral 复核 | 📁 ③ 移交会话 9 首位候选 |

全 20 case 联合收益汇总表与机制三段对比（wait_id5/wait_id2/wait_id4/mte1_stall/批轮次与批深，base→iter2→iter3，block0 口径）见 optimization_log.md r8-4 节（Step C.2/C.4，含基线来源逐 case 标注：r8_base 本体 6 case + perf7 参考态 14 case——perf7 在被测各域生成码与 ee5bf9c9 逐字节等价，IR 守卫证明在案）。

## S8-3. 会话 8 结构性发现（新增硬知识）

1. **吸收律边界补全 → 跨域定律**：单侧吞吐类手段（②③/w1 瘦身/kv-DB；r7-1/r7-2/r8-1 先例）被 1:1 吸收（单侧节省 = 对侧等待增量）；**交接结构删除类（① flat）不被吸收**——wait_id5/id2/id4 三处等待同降 + 双侧真实工作管零变化 + wall -7~-8%，且在 fbm/Form A 两域同构复现。mission"now→B 差距 63% = 交接延迟链"归因获治疗性验证（轮界成分 wait_id5 -95~97% 兑现）。
2. **flat 收益模型 = f(批深空缺率, 任务边界密度) 定量验证**：收益全序 case2（-33.1%，微批 {4,5,8}）> case3（-21.6%，{1,2,8}）> case4（-20.7%，eff 1..2）> case5（-18.1%，{2,3}）> case9（-8.1%，近满批 {6,7,8}/eff 均值 4.5）> case7（-7.4%）> case8（-6.3%）——微批族叠加"批深合并"项（协议轮 ~6→~2 级），近满批族收敛于"轮界等待删除"项；批轮次 ~103→~58/核、批深 4.5→8 两域同式。模型可外推未测 shape（case8 仍按纪律实测）。
3. **顺序游走状态机 = 运行期非均匀 eff_kv 的可行映射**：无前缀表（max_grid 型 my_count 819 不可标量化）、O(1) 摊还/槽、alloc_var 变异、C/V 4 循环独立重走批头快照（恒等由确定性算术保证，不经 ws 传递 §7.2-5）——#R4-3 decode 蓝图的 prefill 推广完成，BITEXACT + 压力 nondet=0 双裁决。
4. **mte1_stall = 四相位链伴生症状非杠杆（三度确认）**：r8-1 从 L0B 侧攻击 stall -2% wall 零响应；r8-2/r8-3 链收紧后 stall +14.7%/+44.9% 而 wall 反降（生产性 pacing 重分配）。② 复活前置更新 = 需 MTE2 管收缩后 MTE1 成为 pacing 管的新结构证据。
5. **v4ksp 形态入库（未来 4 槽需求免成本）**：T.view/T.reshape 本 TVM 版本损坏（探针实证）+ 16KB 槽下 GEMM1/GEMM2 B-tile 互为转置 ⇒ 附.3 字面双取向形态不可表达；同取向替代（GEMM1 K-split×4 + GEMM2 PV K-split×N-split + V 行列双切片 ZN MTE1→L0B 直载新组合）BITEXACT 20/20 + L0B 64.00KB 顶格单池实编读数 + 事件代价账（MMAD 8/iter 近零成本、MTE1 事件税 +2.1% busy、FIX 3/iter 不变）。
6. **flat 域扩展工程审查单新增项**："池别名重载点 × 任务边界交叉"（Form A P2 批头 q_l1/l0a_s 重载 × 跨批任务竞态——设计期静态推演捕获，零实测缺陷成本；fbm 域因 l0a_s 独立池无批头重载而免疫）。
7. **③ 的 post-flat EV 为真开放问题（会话 9 首位候选的立项依据）**：吸收律先例的因果结构在联合态 case7 上已不成立——#R3-3 的吸收通道（批轮界空缺 slack）被 flat 删除（深 8 连续槽窗）、r7-1 的 pacing 管是 C scalar（95%）而联合态 case7 是 **V-paced**（V.vec 634.39us = 77.5% 最大真实工作管 + C.wait_id2 547.70us = 67% C 侧大余量）；判据 = wall 响应主判 + 预测签名（V.vec 显著降 ∧ C.wait_id2 同步降 ∧ wall 降），若 V.vec 降而 wall 零响应 = 吸收律第四例、③ 族证据性关闭。

## S8-4. 精度最终状态（收口，终态 = dda3ad08）

- 终态构成 = r8-3 采纳态（2728c96e，全门禁在案）+ codegen-neutral docstring rev4r8 注记（r8-1 SELF-CHECK ⑨ 遗留项收口）；**16/16 IR 逐字节复核**（终态新鲜 dump r8_src_final == r8_src_prov == r8_src_iter3，dump_final.log/dump_prov.log）⇒ r8-3 全门禁电池对终态有效。
- 终态新鲜复跑关键子集（r8_cache_finalg 隔离缓存，全过）：锚-A **9/9 BITEXACT** + 锚-B **14/14**（BATTERY_RESULT: PASS）；继承压力 4 配置 ×12 **nondet=0**（STRESS_RESULT: PASS）；l0 **10/10**；`--level all` **exit 0**（L0 10/10 + L1 19/19 + L2 12/12 正确拒绝 + Boundary 6/6 + 哨兵 4/4 PERF_PASS，PRECISION_PASS=29 / FAIL=0）——无漂移。
- decode 不回退硬约束：**IR 级**（两轮守卫 decode×5 逐字节 + 兜底域 C 侧源码零触碰）+ **msprof 级**（case10/13/14/16 同会话 ±0.9% 带内）双成立；bench 口径线（case10≥0.90/13≥0.69/14≥0.99/16≥0.52/20≥0.82，起点在案值全带余量达标）的权威换算判定留 Orchestrator 轮末复评（本地为相对口径，如实标注）。
- 精度红线四项全程保持：双舍入栅格（r16 落点未动）/ 单块预归一特化（分派域收窄零触碰）/ sumexp+1e-30（逐字迁移 flat 任务尾）/ 右下角掩码链（idx→kv 同值代入，探针 causal/bound/tail 变体 BITEXACT 佐证）。

## S8-5. 遗留问题与建议（会话 8 终态评估）

| 候选/case | 状态 | 结论 / 建议 |
|-----------|------|-------------|
| ③ VW=2 宽通道（case7/8 域） | 📁 评估归档 → **会话 9 首位候选** | 方案要素齐备落档 optimization_log [STEP-D-EVAL]-5：窄形宽通道形态 + UB 实编读数（D=128 VW=2 156.75K ✓ / D=256 ✓ / D=64 保 W=1）+ 阻塞口径主门（混合容差 + MARE_abs<10×Threshold，现余量 320×）+ 锚降级方案（#9 与锚-B 多块成员 → 阻塞口径 + 根因纪律，#1-8 硬守卫保留）+ 全 20 case 复测条款（emit_v 变更 IR 保证失效）；EV 真开放（S8-3-7） |
| ② L0B 4 槽复活 | ❌ 维持证据性关闭（复活前置更新） | 联合态新数据再证根因门控未变：case9 MTE2 仍 57.8% 最忙真实工作管（flat 前后 ±0.3%）+ C scalar 95.5% ⇒ L0B 槽深非杠杆，预测 wall≈0；复活前置 = MTE2 管收缩的新结构证据；变体留存零探针成本（mha_r81_edit_v1.py） |
| B 层 626us（0.54x）承诺 | ⬇️ 维持降级（r8-1 口径，本会话实测报告） | 实测 = ① 单项 -6.3~-8.1%（case9/7/8 大 prefill）；剩余差距归因 = MTE2 填充节拍（~1.09us/iter 地板）+ 每槽四相位交接不可删段 + V 侧碎片事件税（③ 攻击面）——入库 |
| case20 +1.75%（守卫带内最高读数） | ✅ 判定会话方差 | IR 逐字节（10/16 守卫集成员）⇒ 内核因果排除；绝对量 +0.25us（14.30→14.55）；r7_base 跨会话交叉 +0.3%；orchestrator 复评如超线可按本条归因核查 |
| cann-bench 复评 + 提交副本同步 | ⬜ Orchestrator 轮末统一执行（分工纪律） | 终态 dda3ad08（1926 行）；估算 ~0.94-0.95x 级（非权威） |

**结论**：会话 8 净产出 = **① flat(task,kv) 全域落地**（fbm generic r8-2 + form_a generic r8-3；单块/decode 域分派谓词收窄零触碰），7/7 目标 case 实测收益 -6.26%~-33.08%、全 20 case 无一回退、精度门禁全过 + 终态新鲜子集复核无漂移。迭代 3/6（采纳 2 / 回滚 1），中止条款未触发（consecutive_no_improvement = 0，r8-2/r8-3 连续采纳）；联合包不完整（② 关闭、③ 移交）按 mission 授权条款如实收口——**"① 全域落地 + 实测收益"为会话 8 净产出，0.54x 级承诺不作为已兑现值报告**。

## S8-6. 会话 8 产物清单

- kernel: `examples/mha/mha.py`（**终态 1926 行，md5 dda3ad0877ba4df74bbd559f6912eeed**；r8-3 采纳态 = 1900 行 md5 2728c96eb2fabbfff279918c61c6c3e0）
- `history_version/`: perf4（r8-1 改前 ee5bf9c9）→ perf5（r8-2 改前 ee5bf9c9）→ perf6（**r8-2 采纳态** 5d466b5f）→ perf7（r8-3 改前 5d466b5f）→ perf8（**r8-3 采纳态** 2728c96e）→ perf9（**会话终态** dda3ad08）
- `perf_tuning/`: `msprof_r8_base/`（6 case ee5bf9c9 本体基线）、`msprof_r8_iter1/`（r8-1）、`msprof_r8_iter2/`（r8-2）、`msprof_r8_iter3/`（**全 20 case 联合态**）、`msprof_r8_iter3ref/`（15 case perf7 参考态 = Form A/decode 域 ee5bf9c9 等价基线）、optimization_log.md（会话 8：ANALYSIS 6 步 + ORDER-PLAN 及数据驱动修订 + [RESULT-#R8-1/2/3] + r8-4 [STEP-D-EVAL]/收口节）、perf_log.md（Iteration r8-1..r8-4）、bench_mha_refall.py（perf7 参考态全 20 case 工具扩展，tooling）、final_report.md §S8（本节）、debug_log.md（会话 8 节：中断与恢复处置全记录）
- 归因数据（scratch）：/tmp/opencode/r8_op_case9_{base,iter1,iter2}/ + r8_op_case7_{ref,iter3}/（op A/B block0 口径）、r8_src_{iter1,iter2,iter3,rollback,prov,final}/（IR 快照 ×6，prov/final = 恢复会话 provenance 与 codegen-neutral 证明）、r8_probes/（探针 probe_r81×2/probe_r82×2/probe_r83 + 压力 stress_r82/r83 + budget_r81 + 全部门禁/测量日志 + run_* 脚本 + **mha_r81_edit_v1.py 回滚变体留存**）、r8_cache_{s1,s2,s3,prov,final,finalg}（会话隔离缓存目录族）
- 评测报告: ⬜ Orchestrator 轮末统一执行（本轮未跑——分工纪律；提交副本 mha_submission/ 未触碰）

---

# 会话 9（perf_session=9，2026-09-30，rev4 轨 case9 瓶颈双路径专项：③ V 宽通道主攻 + bN=256 scoping；迭代 1/软预算 4-6 + 1 scoping 归档 + 收口，采纳 0 回滚 1）

> 对象：会话 8 终态 mha.py（1926 行，md5 dda3ad0877ba4df74bbd559f6912eeed，cann-bench 72.59 / 0.96x，cann_final_eval_20260930_083840）。
> 任务（用户 2026-09-30 决策立项）：路径 1 = ③ V 宽通道主攻（VW=2；Stage 1 form_a D=128 → Stage 2 fbm D=256；判决性判据三签名）+ 路径 2 = fbm prefill bN 128→256 仅 scoping 探针（不改生产 kernel）。
> 开工快照 = history_version/mha_impl_r4_perf10.py；改前备份 = perf11（同体 dda3ad08）。

## S9-1. 最终成果

| 指标 | 会话 9 起点 | 会话 9 终态 | 变化 |
|------|-----------|-------------|------|
| kernel | dda3ad08（1926 行） | **dda3ad08（1926 行）= 起点**（r9-1 回滚，IR 16/16 逐字节还原验证 + l0 10/10 回滚态复验） | **零净变更** |
| ③ V 宽通道（路径 1） | STEP-D-EVAL 移交首位候选（EV 真开放） | **证据性关闭（族级）**：Stage 1 全精度门禁通过后性能判决 3/3 目标域 **+12.4~+16.0% 负响应** → 回滚；三签名判定全 FAIL 反向（V.vec +21.8% 未降反升）；Stage 2 不实施（[STAGE2-RULING] 数据驱动：机制证伪硬件级跨 form + case9 域 EV≤0 预判，r8-1 ② 否决先例同款纪律） | 判决性证伪 |
| bN=256 scoping（路径 2） | 未启动 | **scoping 报告落档 [SCOPING-PATH2-R9]**：资源矩阵 GATE1-5 实编全 PASS（L0B/L0C 双 64.00K 顶格 + L0A 48K + L1 288K + UB 170.75K）；GATE6（L2 命中）/GATE7（FIX 税净值）未实测如实声明（需 rev4-S1 量级完整变体，超 scoping 预算；实测方案入库）；EV 修正 = **-12~-19% 条件化（L2 命中 ≥90% 前提）/ 0~+10%（命中塌缩反转）**；蓝图 5 要点（F5 首选形态）供会话 10 立项 | 移交就绪 |
| cann-bench | 72.59 / 0.96x | （复评由 Orchestrator 轮末统一执行；终态与复评基线同体，预期分数不变——复评可作环境漂移校验） | 不变预期 |

## S9-2. 会话 9 优化项与逐轮实测（msprof median us；完整门禁见 optimization_log.md 会话 9 节）

| 迭代 | 优化项 | 目标域实测 | 精度 | 采纳 |
|------|--------|-----------|------|------|
| r9 基线 | —（dda3ad08 同会话新鲜采集 msprof_r9_base 12 case：case9 869.04/case7 858.26/case8 855.78/case3 131.35/case4 27.92/case2 44.84/case5 15.19/decode 5 case；vs 会话 8 终态最大偏差 +0.7% = 环境一致） | — | — | — |
| r9-1 | #R9-1 ③ VW=2 宽通道（form_a flat D=128：pair 合并宽 [2hM,bN] op + per-half mask + 宽 select + S_HALF 1/pair + 方案 K 宽形状尾槽；三次缓冲方案迭代收敛 A''+R，全程实编裁决） | case7 858.26→**964.77（+12.41%）**/ case8 →**993.06（+16.04%）**/ case3 →**150.20（+14.35%）**——3/3 显著下降；守卫 9/9 带内（case9 -0.03% IR 逐字节零变化） | **门禁全过**（探针 16/16 BITEXACT + 守卫 7/7；锚-A 9/9 含 #9 宽域 BITEXACT（Layer 1 兑现无降级）+ 锚-B 14/14；压力 33 项含 wide 专项 w01-w07 ×12 nondet=0；l0/l1/coverage/all EXIT=0；IR 14/16 守卫逐字节 + 2/2 目标预期变；UB wide 域 172.62K ✓；红线四项保持） | ❌ **回滚**（性能下降规则；md5 还原核验 + IR 16/16 逐字节还原 + l0 10/10 复验；变体留存 mha_r91_edit_v1.py 精度全过态） |
| r9-2 | Stage-3 路径 2 bN=256 scoping（scratch 探针，零生产变更） | n/a（GATE 矩阵实编读数 + EV 修正 + 蓝图落档） | n/a | 📁 归档移交会话 10 |
| r9-3 | Stage-4 收口（回滚验证 + 落档，无 kernel 变更） | 回滚态抽查复测（case10/7/20/9，msprof_r9_rb——case10 候选态 +3.04% 边缘读数的方差澄清） | 回滚态 l0 10/10 + IR 16/16 还原 | — |

## S9-3. 会话 9 结构性发现（新增硬知识）

1. **③ 族判决性证伪 + "V.vec 碎片税"模型实测修正**：宽通道事件税侧签名**全部兑现**（V.v_scalar -11.7% / v_stall -35.6% / v_mte3 -27.9%（S_HALF 闭环减半）/ wait_id0 -44.2% / wait_id4 -51.0% / C.wait_id5 -100% / C.scalar -14.9%）——**但 V.vec 主管道 +21.8% 吞没全部节省并反转 wall（+12.4%）**。修正结论：**碎片事件税主体在 scalar/stall/MTE3/wait 侧（可削），V.vec 634us 主体 = 数据体量×吞吐的真实 vector 执行周期**；宽 [2hM,bN] op 执行周期 > 2× 窄 [hM,bN]（大 repeat 宽化低效）+ always-on mask（wide 路径放弃 runtime-skip 的恒等式代价）直接进 vec 管道。analysis §2.2-③ "vec 49.5% busy 中碎片化税、理想 18%" 模型被实测否定（理想占空不可经宽化达到）。
2. **V-paced 结构减速侧对照实验（因果系数 ≈1 双向确证）**：r7-1/r7-2（加速侧 1:1 吸收）+ 本轮（**减速侧 1:1 放大传导**：V.vec +21.8% → C.wait_id2 +20.1% → wall +12.4%）⇒ case7 类 V-paced 域中 V.vec 时长对 wall 的因果系数 ≈ 1，**四相位耦合环 pacing 管判定方法学闭环**（任何单侧手段的 wall 效应 = 该侧 pacing 管时长变化的 1:1 传导；pacing 管身份由最大真实工作管 + 对侧等待余量判定）。
3. **③ 重试前置（唯一合理形态）**：宽化 vec 增量 = repeat 建立税未被摊薄——**行体量翻倍（bN≥256）时宽化 EV 才可能转正**（单行 repeat 数 ×2 摊薄建立成本 + always-on mask 占比下降）。与路径 2 bN=256 蓝图存在组合可能（会话 10 若立项，scratch 变体顺带复测，零生产风险）。
4. **[N,1] 2D copy 块粒度陷阱（工具级硬知识，探针/实现通用）**：`T.copy(buf[N,1] ↔ GM[N,1])` 生成按行 32B 块读/写（UB 侧行 stride 8 floats）→ 数据采样错乱（"output[r]=rowmax(8r)"假模式）+ GM→UB 方向溢出踩踏邻接 buffer 8×。**规避 = [N,1] 缓冲 DMA 一律 1D 视图（buf[:,0] ↔ GM[N]，生成 (N,1,N) 紧凑拷贝）**。生产 kernel 无暴露面（[hM,1] 微缓冲仅被 V-pipe op/broadcast 消费）。
5. **prim_func 内三条解析层硬约束（实测归档）**：①`None` 赋值 → LetStmt(None) TVM FFI **segfault**（缓冲条件化必须用 alloc/别名/双调用分支，禁 None）；②`T.alloc_var(init=TIR 表达式)` → **"only accept int or float"**（init 仅 int/float/Var；复杂表达式用 init=0 + Assign 二段式 = 可变快照，跨 mutation 使用安全）；③缓冲别名赋值 `m_w = m_i` → 生成码**重命名漂移**（last-binding-wins，io3 别名先例同款）破坏 IR 字节级守卫——**跨块传缓冲用宏实参替换（双调用点，实参文本 = 生成码名）或 V-scope 内同块 alloc（方案 R）**。
6. **观测点竞态方法学（探针工程）**：debug emit（MTE3 读 UB）与其后 V-pipe 覆盖写之间必须闭环尾即时 drain（wait MTE3_V）——set 的 wait 延后到下一 emit 的常规闭环结构在"emit 后 src 被覆盖"的观测点会竞态（非确定性 diff）；生产结构天然无此面（reduce→max 间无 MTE3 读点）。
7. **路径 2 资源可行性首判（实编读数）**：bN=256 fbm 五池全可行但 **L0B/L0C 双 64.00K 顶格零余量**（F2/F4/F5 三 L0C 形态均恰顶格；L0C 判据修正 = used≤64K 物理限 ∧ 池值 ≤ 生产已验证 128K——现状 fbm 即 InitBuffer(CO1)=131072 + used 64K）；L2 命中净效应 = 工作集 +42% vs 总流量 -29% 两反向因素**解析不可判**（GATE-6 实测 = 会话 10 立项前置硬门）。

## S9-4. 精度最终状态（收口，终态 = dda3ad08 = 会话起点 = 会话 8 终态）

- 本会话终态与会话 8 终态**逐字节同体**（r9-1 回滚 IR 级还原核验 16/16 逐字节 diff r8_src_iter3 无差异 + md5 一致）⇒ 会话 8 收口全量精度门禁（锚 9/9+14/14、压力、l0/l1/coverage/all，r8_cache_finalg）对终态有效，无失效风险；回滚态新鲜复验 l0 10/10 PASS（r91_l0_rollback.log，r9_cache_rb2 隔离缓存）。
- r9-1 迭代过程中的完整门禁记录（探针/双锚/压力/l0/l1/coverage/all/IR/UB）见 optimization_log.md [RESULT-#R9-1]——全部通过，回滚判据纯由性能规则驱动（#R3-3/#R8-1 同款纪律：**精度全过 + 性能负 → 回滚，变体留存**）。
- 精度红线四项全程保持（宽通道变体上逐项再论证 + 探针实证，回滚后天然保持）；decode 不回退硬约束：IR 级（候选态 14/16 守卫含 decode×5 逐字节；回滚态 16/16）+ msprof 级（候选态 decode 5 case 带内；回滚态 = 基线本体）双成立；bench 口径权威判定留 Orchestrator 轮末复评（本地为相对口径）。

## S9-5. 遗留问题与建议（会话 9 终态评估）

| 候选/case | 状态 | 结论 / 建议 |
|-----------|------|-------------|
| ③ V 宽通道族 | ❌ **证据性关闭（族级，双向结论归档）** | Stage 1 实测负响应（V.vec +21.8% 机制证伪）+ Stage 2 EV≤0 预判不实施；重试前置 = bN≥256 行体量下的宽化 EV 重估（与路径 2 组合，scratch 级）；变体 mha_r91_edit_v1.py 留存（精度全过态） |
| **路径 2 fbm bN=256（case9 主攻候选）** | 📁 **scoping 完成 → 会话 10 立项决策** | 资源矩阵 5/5 实编 PASS（双顶格零余量警示）；EV -12~-19% 条件化；**立项前置硬门 = GATE-6 L2 命中实测**（scratch 变体 msprof L2Cache.csv；命中 <90% ⇒ 收益反转预警 ⇒ 否决或降级）；工程量 = rev4-S1 量级（F5 形态蓝图 5 要点入库 [SCOPING-PATH2-R9]） |
| case9（0.39x，M1 线 0.4 差 0.01） | 归因更新 | 占空受限（C.MTE2 顶格 57% + wait_id2 交接暴露 25% + 爬坡 15%）；① flat 已删可删段、③ 已证伪 ⇒ 规则内剩余杠杆唯一 = 路径 2（槽数减半的结构删除类）；bench 0.39 vs M1 线 0.4 差 0.01 = 路径 2 若兑现 -12% 级则 0.44+ 越线 |
| case10 候选态 +3.04% 边缘读数 | ✅ 方差归因 | IR 逐字节域 ⇒ 内核因果排除（会话 8 case20 +1.75% 先例同款）；回滚态复测澄清（msprof_r9_rb） |
| decode 不回退 / 兜底域零触碰 | ✅ 全程成立 | 候选态 IR 守卫 decode×5 逐字节 + 回滚态 16/16；msprof 守卫带内 |
| cann-bench 复评 + 提交副本 | ⬜ Orchestrator 轮末统一执行（分工纪律） | 终态 dda3ad08（与复评基线同体，预期分数不变） |

**结论**：会话 9 = **判决会话**（零净变更、1 回滚、双路径裁决完备）。路径 1（③ 宽通道）经全额实施 + 全精度门禁 + op 三签名判定获得**判决性证伪**（事件税可削但 vec 主体不可宽化缩减，V-paced 减速侧 1:1 传导 = 因果系数双向确证）——比"零响应"更强的关闭证据入库；路径 2（bN=256）完成资源可行性首判（5/5 实编 PASS）+ EV 条件化修正 + 蓝图移交，**会话 10 立项决策输入齐备**（GATE-6 L2 实测为前置硬门）。中止条款②未触发（consecutive_no_improvement = 1 < 3）；按预算纪律收口（迭代 1/4-6 + scoping 归档，未强行耗尽——负 EV 验证不实施纪律）。

## S9-6. 会话 9 产物清单

- kernel: `examples/mha/mha.py`（**终态 1926 行，md5 dda3ad0877ba4df74bbd559f6912eeed = 会话起点 = 终态**；r9-1 改动已 IR 级完整还原）
- `history_version/`: `mha_impl_r4_perf10.py`（=开工快照 dda3ad08）、`mha_impl_r4_perf11.py`（=r9-1 改前备份，同体 dda3ad08；回滚目标）——无采纳态快照新增（零净变更会话，会话 7 先例同款）
- `perf_tuning/`: `baseline_r9.json`（会话基线 + op base 侧读数索引）、`msprof_r9_base/`（12 case 同会话基线）、`msprof_r9_iter1/`（r9-1 候选态 12 case）、`msprof_r9_rb/`（回滚态抽查 4 case）、optimization_log.md（会话 9：[ANALYSIS] Step 0 + [ANCHOR-PLAN-R9] + [P0-WIDE-MECH] + [SCOPING-PATH2-R9] + [IMPL/RESULT/PERF-#R9-1] + [STAGE2-RULING]）、perf_log.md（Iteration r9-1/r9-2）、final_report.md §S9（本节）、debug_log.md（会话 9 节：探针伪影双发现 + 缓冲方案三次迭代 + alloc_var/None/命名漂移三解析层约束发现全记录）
- 归因数据（scratch）：/tmp/opencode/r9_op_case7_iter1/（op A/B 候选态）、r9_probes/（**P0 机制探针族**：probe_r9_wide_p0v2（8 段位级对比）/p0dbg/mscan/bisect2/v6/v1011/e78/decide（reduce 异常二分定位链）+ **probe_r91_wide.py 全 kernel 探针 23 shape** + stress_r91_wide.py（33 项）+ probe_r9_bn256_budget.py（Stage-3 GATE 矩阵）+ repro_deadlock/repro_segfault（最小复现）+ parse_op_ab.py（op 双口径工具）+ 全部日志）、r9_src_iter1/（候选态 IR 16 目标）+ r9_src_rollback/（回滚还原验证 16/16）、**mha_r91_edit_v1.py（回滚变体留存，52dd7e8a，2405 行，精度门禁全过态）**、r9_cache_*（会话隔离缓存族）
- 评测报告: ⬜ Orchestrator 轮末统一执行（本轮未跑——分工纪律；提交副本 mha_submission/ 未触碰）

---

# 会话 10（perf_session=10，2026-09-30/10-01，rev4 轨路径 2 专项：fbm prefill bN 128→256 结构加宽；迭代 1/软预算 4-6 + 收口（中断恢复），采纳 0 回滚 1（结构性不可行））

> 对象：会话 9 终态 mha.py（1926 行，md5 dda3ad0877ba4df74bbd559f6912eeed，cann-bench 72.59 / 0.96x）。
> 任务（用户 2026-09-30 基于 [SCOPING-PATH2-R9] 全件决策立项）：路径 2 = fbm prefill bN=256 结构加宽（T2S 同取向共池 + F5 L0C；GATE-6 L2 命中实测 = 前置硬门）+ ③×bN256 组合宽化 EV 复测机会（scratch 级）。
> 开工快照 = history_version/mha_impl_r4_perf12.py；改前备份 = perf13（同体 dda3ad08）。
> **本节由中断恢复会话落档（如实）**：前次会话于 T1 判决矩阵 3/3 运行完成（2026-09-30 22:52）、落档未做之时被外部中断；恢复收口 = 2026-10-01 00:40-01:15+08:00（现场核验 + 预登记判据宣告 + 回滚三重验证 + 轻量复验 + 落档补全，零新探索、零 bN=256 重试）。

## S10-1. 最终成果

| 指标 | 会话 10 起点 | 会话 10 终态 | 变化 |
|------|-----------|-------------|------|
| kernel | dda3ad08（1926 行） | **dda3ad08（1926 行）= 起点**（r10-1 结构性不可行回滚；md5 一字不差核验 + IR 16/16 逐字节还原（disable_cache + r10_cache_rb 新鲜 dump，manifest 逐字节一致）+ 轻量复验全绿（l0 10/10 + 锚 23/23 BITEXACT + 压力 4×12 nondet=0）） | **零净变更** |
| 路径 2 bN=256（case9 主攻候选） | scoping 移交（GATE1-5 实编 5/5 PASS + EV -12~-19% 条件化 + F5 蓝图 + GATE-6 前置硬门） | **结构性不可行判决（[DESIGN_ERROR] 判据字面命中）**：T2S 实施 HANG → bisect/E 系毒化矩阵（15 实测变体）→ 双重毒化归纳 → E21 边际理论证伪 → T1（v4ksp 全架构唯一可表达推广，2382 行）3/3 HANG → E22 全排空补强 3/3 HANG → 预登记穷尽条款命中 + 静态不可能性证明族；**bN 上界定理 = 新硬件硬限知识入库** | 判决性关闭（禁止重试） |
| ③ V 宽通道重试前置 | "bN≥256 行体量摊薄 repeat 税"（Stage4.5 复活窗口判据） | **规则内不可达**（随路径 2 不可行）⇒ ③ 关闭升级为**规则内永久**（复活前置仅剩新硬件（L0B 扩容类）/框架级新形态证据） | 关闭升级 |
| 大 prefill 规则内杠杆版图 | ①已兑现 + ②③条件关闭 + ④前置待激活 + 路径 2 开放 | **收束**：①已兑现、②③④条件/永久关闭、路径 2 结构性不可行 ⇒ case9 M1 线 0.4（0.39x 差 0.01）与大 prefill ≥1.0x（B 层理想流水上限 0.54-0.79x）均规则内不可达，**仅剩架构级方案 C** | 版图收束 |
| cann-bench | 72.59 / 0.96x | （复评由 Orchestrator 轮末统一执行；终态与复评基线同体，预期分数不变——复评可作环境漂移校验） | 不变预期 |

## S10-2. 会话 10 优化项与逐轮实测（完整判决链见 optimization_log.md [RESULT-#R10-1] + debug_log D10-1..D10-4）

| 迭代 | 优化项 | 实测结果 | 精度 | 采纳 |
|------|--------|---------|------|------|
| r10 基线 | —（msprof_r10_base median n=25：case9 869.87/case4 28.21/case7 855.54/case10 34.61/case14 378.26/case16 562.01/case20 13.98 us；vs r9_base 全带内 = 环境一致；baseline_r10_iter1.json 落盘） | — | — | — |
| r10-1 | #R10-1 路径 2 T2S bN=256（12 条硬约束实施，c748cfb2/2259 行）→ 首跑全形状族 HANG → 判决链：printf 51 检查点（系统级环定位，printf 不可见域 = M/FIX 管内部）+ 静态令牌审计 4 轮全平衡（排除协议死锁）+ bisect 族 3（毒源 = nc1 第二 burst）+ E 系 9（双毒化路径独立成立）+ E21（边际理论证伪）+ T1 v4ksp 全架构（唯一可表达推广）+ E22（全排空补强，恢复会话补档） | **全部 bN=256 形态 aicore timeout（507014）挂死**：PROD ×4 shape / E13/E15/E16/E17/E21 ×s256（+f512）/ **T1 3/3（s256/f512/s192，EXIT=1 在案）**/ E22 3/3（s256/f512/case9）；唯二 RUNS = bisectD（双删 nc1）+ D@single192 = 毒源定界证据；数值门未达 | **fail（HANG，非数值性）**；legacy bN=128 域同文件 PASS（缺陷域隔离证明） | ❌ **NO — 结构性不可行**（预登记穷尽条款；非性能规则回滚；回滚三重验证 + 轻量复验全绿） |
| r10-2 | Stage-4 收口（中断恢复：判据宣告 + 回滚执行 + 三重验证 + 落档补全 + E22 偏差项补档） | 回滚态 l0 10/10 + 锚-A 9/9 + 锚-B 14/14 BITEXACT + 继承压力 4×12 nondet=0；IR 16/16 逐字节还原；工具教训⑤零残留关闭 | pass（回滚态） | —（收口） |

## S10-3. 会话 10 结构性发现（新增硬知识）

1. **手动 mma 流水的 bN 上界定理（硬件硬限，判决性）**：form_b_mma 类手动 K/N-split 流水的安全编排必要条件 = **B 操作数工作集 ≤ L0B 64KB（含 K-split 链全部在途 chunk）**。bN=256∧D=256 ⇒ K^T/V 各 128KB = 2× 超限 ⇒ 相位内 L0B 重用数学上不可避免 ⇒ 双重毒化下任何编排挂死。仓内互证先例族：gqa_fwd_varlen:112-114 block_N=256 的 L0B 128KB>64KB 否决 + rev4 DESIGN §4.5-e decode mma 化分池否决推导（同一数学，三处独立出现）。
2. **双重毒化机制（挂死无报错 = 硬件级停滞）**：(a) **L0B 相位内覆写 @~50ns 边际挂死**（E16/E13：M 管 SetFlag<M_MTE1> 在 mma 退休点触发，cube 对 L0B 物理读排空滞后于退休信号）；(b) **K128-mma 相位深度 ≥3 或 B 操作数二次读取挂死**（E17/E21/E15/PROD）。生产安全谱系三条全部满足 (a)(b) 规避（legacy 跨槽重载被下一槽 MTE2 到达门控 ~0.5-1us / v4ksp 4 分立槽 load-first 零相位内重用 / legacy G2 每池槽单次读取）；E21 证明 ~0.3-0.8us 边际门控不足（(b) 为主因）、E22 证明 gemm_v0 级全 M 排空亦不足、T1 证明分立槽谱系推广至 B 工作集超限时必然引入相位内重载 ⇒ **编排空间彻底封死**。推论：静态令牌审计平衡不能排除该类缺陷（4 轮审计全平衡仍挂）——判决必须靠运行时变体矩阵。
3. **E 系毒化矩阵方法学（挂死类缺陷判决性定位手段）**：单因子删除/单因子保留变体矩阵（双删定界毒源 → 单侧删证对称性 → 载荷/mma 分离证双路径独立 → 边际梯度证伪边际理论 → 全排空封死谱系上限）+ 生成码复核排除 codegen 伪影——15 变体 ~7 小时完成"协议死锁怀疑 → 硬件硬限定理"归因闭环（子进程隔离 + timeout 纪律全程）。
4. **scoping 方法学盲区（GATE 矩阵修正入库）**：实编预算门（GATE-2 L0B 64.00K 顶格 PASS）验证的是**槽容量合法性**，不验证**工作集相位内重用的编排合法性**——**"L0B 工作集（B 操作数在途总量）≤ 64KB"必须作为独立预算门先于 GATE 矩阵执行**；"双顶格零余量警示"必要不充分。
5. **③ 族关闭升级为规则内永久 + 大 prefill 规则内杠杆版图收束**：①已兑现（r8-2/r8-3 flat 采纳态 = 现生产；case9 r3 终态 1379.56 → 869.87us 累计 -37%）；②证据性关闭（复活前置 = MTE2 管收缩后 MTE1 成 pacing 管的新结构证据）；③**规则内永久关闭**（重试前置 bN≥256 规则内不可达）；④w3 瘦身条件性关闭（激活前置"①-③ 落地 + B régime FIX/L2 绑定"链条断裂）；路径 2 结构性不可行 ⇒ **case9 M1 线 0.4 与大 prefill ≥1.0x 均只剩架构级方案 C**（框架虚拟通道 L0C→UB/UB→L1 = A 层 2.3-3.4x 唯一入口（会话 6 禁令源码级实证 = 需框架改造）+ 单域 flash 重构；analysis 建议 4"架构决策 = 1.77-3.4x 空间唯一钥匙"，建议以 case9 为样板先做框架级可行性验证再投资决策）。
6. **工具级教训 5 条（D10-1 在案，防再踩）**：①探针 jit 工厂零参 def + out_idx 组合 → lowering 期 C 段错误（exit 139 无 traceback）；②bisect runner f-string `{{_OP}}` 双花括号转义 → 子进程字面路径 FileNotFoundError；③pkill -f 模式串自杀（bracket-trick）；④occurrence 计数器内层循环双增（先物化再映射）；⑤**别名重命名绑定序漂移（k_l1_h1）——恢复会话收尾关闭**：漂移仅存 r10-1 变体族（名字级，功能全同），生产态源码 + 16 IR 双 grep 零残留，随回滚消解（处置结论 = 接受回滚消解，非接受名字级漂移）。
7. **构建工具教训（D10-3）**：Form A flat 体 = fbm flat 骨架逐字镜像 ⇒ 行窗口手术必须以 prim_func 区域锚（"rev4 Form B-mma" 注释）约束搜索起点，否则 next() 命中错误体（ruff F821 运行前捕获，零损失）。

## S10-4. 精度最终状态（收口，终态 = dda3ad08 = 会话起点 = 会话 8/9 终态）

- 终态与会话 8/9 终态**逐字节同体**：md5 dda3ad0877ba4df74bbd559f6912eeed 一字不差 + 1926 行 + ast.parse OK + ruff clean + **IR 16/16 逐字节还原**（r10_src_rollback vs r8_src_iter3 `diff -rq` 零差异 + manifest 逐字节一致；参照链 provenance 使用前复核 r8_src_iter3 == r9_src_rollback diff exit 0）⇒ 会话 8 收口全量精度门禁在案态对终态有效。
- 回滚态新鲜轻量复验（r9-3 同款口径，r10_cache_rbv 隔离缓存）：l0 **10/10** + 锚-A **9/9 BITEXACT**（vs Developer final 6505 硬守卫）+ 锚-B **14/14 BITEXACT**（vs r3 final + 阻塞口径）+ 继承压力 **4 配置 ×12 nondet=0**——全绿。
- r10-1 候选族（T2S/bisect/E 系/E21/T1/E22）**全部 HANG——T1/E22 数值正确性未判（HANG 先于数值门，如实声明）**；精度红线四项零候选采纳全程保持；decode 不回退硬约束：回滚态 IR 16/16 含 decode×5 逐字节成立；legacy bN=128 域在缺陷态文件中 PASS（bisect_r10.log）= 缺陷域隔离 + 现生产态无暴露面。
- 工具教训⑤收尾：k_l1_h1/v_l1_h1 生产态零残留（源码 + IR 双验证）——D10-1 待办关闭。

## S10-5. 遗留问题与建议（会话 10 终态评估）

| 候选/case | 状态 | 结论 / 建议 |
|-----------|------|-------------|
| 路径 2 fbm bN=256 | ❌ **结构性不可行（[DESIGN_ERROR] 判据字面命中）** | 双重毒化 + bN 上界定理 + 全部可表达形态穷尽（15 实测变体 + 静态不可能性证明族）；**禁止任何 bN=256 重试（穷尽条款）**；遗留工件族留存 = 知识资产防重复探索；全局流转（是否触发 Stage-1 设计回退）由 Orchestrator 裁决——本 agent 预判 = 候选证据性关闭而非设计回退（现行 rev4 设计产出健康终态 72.59/0.96x，不可行的是可选优化项而非现设计） |
| GATE-6/7 + 条件化 EV | ⬜ **未执行 → 作废归档** | HANG 先于性能门：GATE-6 L2 命中实测/GATE-7 FIX 税净值未执行；[SCOPING-PATH2-R9] EV（-12~-19% @L2≥90% / 0~+10% @命中塌缩）随不可行作废（如实声明）；③×bN256 组合复测机会同灭 |
| ③ V 宽通道族 | ❌ 关闭升级为**规则内永久** | 重试前置（bN≥256 行体量摊薄 repeat 税）规则内不可达；复活前置仅剩新硬件（L0B 扩容类）/框架级新形态证据；变体 mha_r91_edit_v1.py（精度全过态）留存仍有效 |
| ② L0B 4 槽 / ④ w3 瘦身 | ❌/⏸ 条件性关闭（维持） | ② 复活前置 = MTE1 成 pacing 管新结构证据；④ 激活前置（①-③ 落地 + B régime FIX/L2 绑定）链条断裂 |
| case9（0.39x，M1 线 0.4 差 0.01） | ❌ **规则内不可达** | 唯一规则内杠杆（路径 2 兑现 -12% 级即 0.44+ 越线）已证结构性不可行 ⇒ M1 case 线放弃规则内路径，转架构级评估 |
| 大 prefill ≥1.0x / 架构级"方案 C" | 📁 **唯一剩余方向** | 框架虚拟通道（L0C→UB/UB→L1，A 层 2.3-3.4x 唯一入口，需框架改造——会话 6 禁令源码级实证在案）+ 单域 flash 重构；建议 = analysis 建议 4：以 case9 为样板做框架级可行性验证后再投资决策（超出算子内规则预算，需独立立项） |
| decode 域 | ⬜ 非本会话判决范围 | decode mma 化已有 rev4 §4.5-e 预算不可行推导在案（分池 128KB>64KB 硬证据 + decode wall=每槽 MTE2 带宽无杠杆；analysis 20260928 建议 1"decode ⑤"被该硬证据覆盖）——若重启需先推翻该推导 |
| cann-bench 复评 + 提交副本 | ⬜ Orchestrator 轮末统一执行（分工纪律） | 终态 dda3ad08（与复评基线同体，预期分数不变）；提交副本 mha_submission/ 全程零触碰 ✓ |

**结论**：会话 10 = **判决性关闭会话**（零净变更、1 结构性不可行回滚、路径 2 判决链完备闭环）。获得比"性能证伪"更强的关闭证据：**硬件硬限定理（bN 上界 = B 工作集 ≤ L0B 64KB）+ 双重毒化机制 + 毒化矩阵方法学 + scoping 盲区修正（工作集门先于槽容量门）**——为未来任何手动 mma 流水类 kernel 永久关闭"bN=256 结构加宽"方向（15 变体遗留工件族 = 防重复探索资产）。大 prefill 规则内杠杆版图就此收束：规则内路径全部穷尽或条件关闭，case9 M1 线 0.4 与 ≥1.0x 目标均转架构级投资决策（方案 C）。中止条款②未触发（consecutive_no_improvement = 2 < 3）；**[DESIGN_ERROR]（路径 2）提交 Orchestrator 裁决**；禁止 bN=256 重试条款永久生效。

## S10-6. 会话 10 产物清单

- kernel: `examples/mha/mha.py`（**终态 1926 行，md5 dda3ad0877ba4df74bbd559f6912eeed = 会话起点 = 终态**；r10-1 改动经回滚完整还原，IR 16/16 逐字节验证）
- `history_version/`: `mha_impl_r4_perf12.py`（=开工快照 dda3ad08）、`mha_impl_r4_perf13.py`（=改前备份/回滚目标，同体 dda3ad08）——无采纳态快照新增（零净变更会话，会话 7/9 先例同款）
- `perf_tuning/`: `baseline_r10_iter1.json`、`msprof_r10_base/`（7 case 同会话基线）、optimization_log.md（会话 10：[ANALYSIS] Step 0 + [ORDER-CHECK] + [IMPL-#R10-1] + **[RESULT-#R10-1]（恢复会话补档：全判决链 + bN 上界定理 + 工件清单）**）、perf_log.md（Iteration r10-1/r10-2）、final_report.md §S10（本节）、debug_log.md（D10-1 bisect 判决矩阵 + D10-2 E 系扩展与双重毒化 + D10-3 E21/T1 预登记判据 + **D10-4 收口（恢复会话：T1 终局读数/E22 偏差项补档/回滚三重验证/轻量复验/中断-恢复经过）**）
- 判决链数据（scratch，/tmp/opencode/r10_probes/ = **知识资产留存区，防未来重复探索**）：**mha_T1.py（04a41a97，2382 行，v4ksp 全架构终局变体）**+ E 系变体全文件（mha_bisectB/D/F.py、mha_dbg.py（51 printf 检查点）、mha_E1/E2/E4/E5/E6/E13/E15/E16/E17/E21/E22.py）+ make_*.py 变换脚本族 + probe_r10_bn256.py/stress_r10_bn256.py/smoke_r10.py/bisect_r10.py/budget_r10.py/run_shape.py + 生成码 dump（hang_single192.cc/hang_flat512.cc/hang_E16_s256.cc/hang_E17_s256.cc/refix_G_B_single_d256.cc/refix_G_case4_B_p3_kv2.cc）+ 日志族（**run_T1_{s256,f512,s192}.log = 3/3 HANG 判决证据（EXIT=1）**/run_E22_*（3/3 HANG 补强）/run_E*（毒化矩阵）/bisect_r10.log/smoke_r10.log/run_PROD_s256.log/dump_*.log/msprof_r10_base_case*.log）+ 恢复会话产物（dump_rollback_r10.py/run_dump_rollback_r10.sh/run_rbv_r10.sh/dump_rollback.log/rbv_{l0,anchors,anchor_stress}.log）；/tmp/opencode/r10_src_rollback/（IR 16/16 还原验证）+ r10_cache_*（会话隔离缓存族，含恢复会话 r10_cache_rb/rbv）
- **如实声明**：①r10-1 PROD 缺陷态源文件（c748cfb2，2259 行）无独立留存副本（回滚 cp 按任务书执行，pyc 被覆盖）——实现内容由 E 系派生全文件 + make_*.py + [IMPL-#R10-1] 规范 + hang_*.cc 实质保留，知识无损；②T1/E22 数值正确性未判（HANG 先于数值门）；③GATE-6/7 未执行，条件化 EV 作废归档
- 评测报告: ⬜ Orchestrator 轮末统一执行（本轮未跑——分工纪律；提交副本 mha_submission/ 未触碰；.orchestrator_state.json 未读写）

---

# 会话 11（perf_session=11，2026-10-04，decode D=256 duty 墙专项：S2 单侧非对称 L1 双缓冲 + S3 能力门；迭代 3/软预算 3，采纳 0 回滚 1（性能下降）+ 能力门关闭 1）

## S11-1. 最终成果

| 项 | 结果 |
|---|---|
| 生产终态 | `examples/mha/mha.py` md5 **dda3ad0877ba4df74bbd559f6912eeed**（1926 行）= 会话起点 = 会话 8/9/10 终态（cann-bench 20261003 权威：20/20 pass，72.63，0.96x） |
| 采纳 | **0 项**（#R11-1 S2 单侧 K-DB 精度门禁全过但性能 −2.00%/−2.70% 劣化 → 性能下降规则回滚；#R11-2 S3 门② FAIL → 关闭落档） |
| 判决性知识 | **gemm_v0 稳态守恒律**（机制证明 + 实测双确认）+ **预取流重排税**（−2.7% 引擎效率）+ **S3 三门判决**（链式 BITEXACT 能力 / region 切片损坏第 4 例证 / id 预算）+ **needClear=(dst_offset==0) codegen 规则** |
| 穷尽声明 | decode D=256 duty 墙（case15/18 的 1.45× 上界）**规则内路径穷尽**：S1 推演负 / S2 实测负 / S3 门② FAIL+布局税负 / mma 化永久否决（S10-3）/ vector 重构撞不回退硬约束 ⇒ **维持现态 = 规则内最优；破局仅剩架构级**（gemm_v0 模板 per-buffer 同步改造 / 框架虚拟通道，移交立项决策） |
| 中止条款 | **consecutive_no_improvement = 3（r9-1 + r10-1 + r11-1）⇒ 条款②形式触发**，Stage 3 终止判定权在 Orchestrator |

## S11-2. 会话 11 优化项与逐轮实测

| 轮 | 项 | 域 | 精度 | 性能（msprof median n=25，同会话新鲜基线） | 判决 |
|---|---|---|---|---|---|
| r11-1 | #R11-1 S2 单侧 K 侧 L1 DB（3D [2,256,256] 无 annotate + gemm_v0 运行期切片 + 4 预取点 + 零新增 flag） | form_b_gemm_v0 ∧ D=256 ∧ kv≥2（case15/18） | **全门禁过**（probe 34/34 含 22 目标 BITEXACT + 12 守卫；锚 9/9+14/14；压力继承 4+扩展 17 专项 ×12 nondet=0；l0/l1/coverage/all 全绿；IR 15/17 守卫 + 2/2 目标改变；L1 实编 400KB） | case15 505.17→**515.25 (+2.00%)**、case18 504.05→**517.68 (+2.70%)**；op：duty 68.45→68.89%（不动）、mte1/cube/vec 三管精确零变化、**mte2 busy +2.86% 同字节**（74.8→72.8 GB/s/核）、V wait_id0 +7.9us 阴影 | **NO — 回滚**（性能下降规则；变体留存 mha_r111_edit_v1.py 89644d68 精度全过态） |
| r11-2 | #R11-2 S3 三道 API 能力门（scratch，零生产变更） | — | — | — | **①PASS（init=False 链式 = BITEXACT vs 单段）/ ②FAIL（A 列段切片损坏 rel 1.01）/ ②'FAIL（B 列段切片损坏 rel 0.99）/ ③PASS（id 预算）⇒ S3 关闭** |
| r11-3 | 收口（终态确认 + 三重还原核验 + 轻量复验 + 落档） | — | pass（终态 = 在案态） | — | 闭合 |

## S11-3. 会话 11 结构性发现（新增硬知识）

1. **gemm_v0 稳态守恒律**（decode duty 墙终局归因；模板源码 common.h:1189-1372 + r11-1 实测双确认）：prologue `SetFlag/WaitFlag<MTE2_MTE1>(0)` **管级全排空**（+`FIX_M` 全排空）+ 每 tile `PipeBarrier<PIPE_M>` 标量阻塞 + 标量顺序发射 ⇒ **每槽 wall = MTE2 字节时间 + gemm 标量阻塞时间，与 L1 槽数/预取编排无关**——gemm 调用前入队的预取把该 gemm 的 MTE1 起点推迟预取全量；gemm 返回后发出的预取与单缓冲同发出点（epilogue MTE1_MTE2 释放 = 最早合法点，现生产码已在此点）。31% duty 缺口（X1+X2≈1.64us/槽）规则内不可压缩；**gemm_v0 调用边界 = 硬墙**（任务书分支 B 前提实测成立：duty 签名不动）。唯一可回收项 = 任务/轮边界先导（~0.3-0.6us/task），实测被 2. 吞没。
2. **预取流重排税**：decode 带宽墙域（MTE2 busy 速率已贴 512B 行布局上限）的载入流重排（K 预取与 V/w2 流时间交叠 + L1 写目的地区交替）⇒ 同字节下 MTE2 busy 期速率 −2.7%（74.8→72.8 GB/s/核）+ FIX −3.8% 同族——**重排类优化在此域具负外部性**；wall Δ ≈ mte2 busy Δ（关键路径 = 引擎效率本身）。未来任何 decode 域 MTE2 流序改动以此为先验。
3. **S3 三门判决**：①gemm_v0(init=False) 跨调用 L0C 链式累加**位级等价**（2×K128 vs 单段 K256 = 0.0000e+00——kL0 tile 下同为 K 升序 fp32 累加；能力入库，未来 K-split 类设计可免值级探针）；②/②'**A/B 操作数 L1 列段 region 切片 = 静默数据损坏**（_retrieve_ptr 朴素行主序 ≠ zN 物理偏移；[IMPL-#R10-1] 同族第 4 例证——**规则固化：gemm_v0/mma 操作数只可用整缓冲或首维（槽维）切片，任何尾维列段切片禁止**）；③兜底域 id 预算宽裕（用户 flag=0；gemm_v0 内部 id0/1；自由 id2-7/通道）。S3 残余形态（独立半缓冲+整宽列窗载入）可表达但 B 侧半宽载入付 256B 行 −33% 布局税（微基准 A/B 判决）落在唯一主墙上 ⇒ EV 负，关闭。
4. **needClear=(dst_offset==0) codegen 规则**（copy_gm_to_l1 末参由 codegen 按 dst 偏移派生，common.h:62-67）：3D 双槽下 side≠0 槽载入**不清尾**——尾块落 side-1 时 stale 行进 gemm ⇒ 被掩码列 w1 = q·garbage（可 NaN/Inf）；安全性 = V 侧 **select 替换制**掩码（reduce 前 -inf 整体替换）+ 尾槽必触发 mask 构建（runtime skip 守卫恒真）+ gemm 列独立无串扰 + 256×256 zN 无 fractal 洞——探针 T_tail300/k10_tail300 实证 BITEXACT。**未来任何 3D 槽化/非零偏移 L1 载入改动必查此规则**。
5. **DB 槽位机制位级安全第六度入库**：22/22 BITEXACT + kv=10 跨轮形生成码记账恒等式实证（头 1 + 批内 7+1 + 轮尾 1 = 10 = 消费数；#R4-2b 恒等式在 decode 域再证）。

## S11-4. 精度最终状态（收口，终态 = dda3ad08 = 会话起点 = 会话 8/9/10 终态）

- 权威评测：cann_performance_eval_20261003_122353（20/20 pass，综合 72.63，平均 0.96x）——终态同体，复评由 Orchestrator 轮末统一执行（预期分数不变）。
- 回滚三重还原核验：md5 一字不差 + IR **17/17 逐字节还原**（r11_src_rollback vs r11_src_base == r8_src_iter3 provenance 链，disable_cache + r11_cache_rb 新鲜 dump）+ ast/ruff clean。
- 回滚态轻量复验（r9-3/r10 同款口径）：l0 **10/10**（L0_RB_EXIT=0）+ 锚-A **9/9 BITEXACT** + 锚-B **14/14**（BATTERY_RESULT: PASS）+ 继承压力 **4×12 nondet=0**。
- r11-1 变体（未采纳）精度档案：全门禁过态（S11-2 表）——其 BITEXACT 探针/压力/锚日志为未来复用该机制的免探针依据。

## S11-5. 遗留问题与建议（会话 11 终态评估）

1. **decode D=256（case15 0.68x / case18 0.81x）规则内穷尽**——1.45× 上界空间（999→1452 GB/s）仅剩架构级入口：(a) **gemm_v0 模板级改造**（prologue 管级全排空 → per-buffer 同步，打破守恒律的最小框架改动；影响面 = 全部 gemm_v0 调用方，需框架侧立项）；(b) vector 路径重构（baseline IncreFlash 达成方式，存在性证明在案 analysis §6.2 路线 3；撞 decode 不回退硬约束 = 设计级决策）；(c) 框架虚拟通道/单域 flash（会话 6/10 在案）。**建议：不再规则内立项；如架构级投入，(a) 为最小改动面候选，先做框架可行性验证。**
2. **S1（bN=128 双侧 DB）不建议实测**——推演性关闭（守恒律 + 协议税 + 重排税 + 先导减半三链，r11-1 归因 (3)）；如用户坚持实测，本会话全部工具（探针/压力/门禁/dump 链）可低成本复用，但预期净负。
3. **禁止规则内重试清单更新**：decode D=256 duty 墙全族（S1/S2/S3/mma 化/vector）——三链判决在案（S2 实测负 + S3 门 FAIL + 守恒律机制证明）；与 S10 穷尽条款（bN=256 mma）并列。
4. **中止条款②（连续 3 次无提升）形式触发**——Stage 3 是否终止由 Orchestrator 裁决；本 Agent 判断：规则内杠杆版图已全域闭合（大 prefill S10-3-4 + decode/小 prefill analysis_20261003 §6 + duty 墙本轮），**建议置 SUCCESS 收口维持 72.63/0.96x 健康终态**。
5. 测量缺口（如实声明）：S1 未实测（推演性关闭标注）；case18 变体态 op 模式未采（case15 op 已给机制签名，case18 wall 同向 +2.70% 一致，增量信息低）；重排税的 HBM/DRAM 页局部性微观归因未做独立隔离实验（busy 速率下降为实测事实，机理为推断标注）。

## S11-6. 会话 11 产物清单

- 生产/备份：`mha.py`（dda3ad08 终态）；`history_version/mha_impl_r4_perf14.py`（改前备份）/`perf15.py`（还原态同体）；变体留存 `/tmp/opencode/r11_probes/mha_r111_edit_v1.py`（89644d68，2038 行，精度门禁全过态）
- 本报告/日志：`optimization_log.md`（[ANALYSIS]-S11 Step0 + [ORDER-CHECK]/[DB-ANALYSIS]/[IMPL/SELF-CHECK/RESULT-#R11-1] + [IMPL/RESULT-#R11-2] + 收口判定）、`perf_log.md`（Iteration r11-1..3）、`baseline_r11_iter1.json`、本报告 §S11
- 测量数据：`/tmp/opencode/r11_base/`（新鲜基线 9 case wall + json）、`/tmp/opencode/r11_iter1/`（变体 9 case wall + case15 op 模式 CSV + perf_runner.log）
- IR 链：`/tmp/opencode/r11_src_base/`（17 shape 开工基线 == r8_src_iter3 16/16 + case18 新增）、`r11_src_var/`（变体 15/17+2/2）、`r11_src_rollback/`（还原 17/17）；缓存 r11_cache_{base,var,g1,smoke,iter1,s3,rb,rbv}
- 探针/工具：`/tmp/opencode/r11_probes/`（probe_r111_kvdb.py + stress_r111_kvdb.py + smoke_r111.py + probe_s3_gates.py + dump_src_r11.py（17 shape）+ dump_{base,var,rollback}_r11.py + run_{wall,gates,perf,rbv}_r11*.sh + 日志族 r111_{probe,anchors,anchor_stress,stress_ext,l0,l1,coverage,all,l0_rollback}.log + rbv_{anchors,anchor_stress}.log + smoke_r111.log + probe_s3_gates.log + dump_{var,rollback}.log + parse_op_r11.py）
- 引用在案资产：analysis_20261003_decode_smallprefill_roofline.md（立项依据 §2/§6.2）、decode_roofline_a1/op_parsed.json（op 基线）、gate0b_verdict_iter25.md（≤15 约束）、optimization_log [IMPL-#R4-2b]/[RESULT-#R4-3]/[RESULT-#R8-1]/[IMPL-#R10-1]、common.h:1189-1372（gemm_v0 模板）

**收尾补记（2026-10-04，用户指令）**：按用户提供的 MindStudio 采集命令示例（`msprof op simulator --soc-version=Ascend910B4 ...` → 本环境实际 `--soc-version=Ascend910_9362`，Orchestrator 直接执行，非调优迭代），在**当前终态 kernel（dda3ad08，采集前/后 md5 双端记录一致，含提交副本）**上新鲜采集 case15 指令级流水数据：`mindstudio_export/case15_simulator_20261004/OPPROF_20261004184651_ZHPYKPCKGDIANDFZ/`（simulator/ 60 子核 trace + merged trace.json + visualize_data.bin + dump/ 3 文件；未加 `--dump=true` ⇒ 无 *.dump 中间产物需清理；1.8GB，SIM_EXIT=0，模型 8.03min / Total tick 840,596）；转移用 tarball `mindstudio_export/mha_case15_simulator_20261004.tar.gz`（154,637,673 B / 256 entries，pigz -t 完整性校验 OK，root=case15_simulator_20261004 与 case9 约定一致）。关键读数（core0 busy-union，`core0_tracestats_full.txt` 2615 行）：camodel per-core duration ≈445.5us（60 核 spread 仅 0.3% = decode 960 task/20 核满射）**快于**真机 wall 503.84us（0.88×）——与 case9 方向（camodel 1.83× 慢）相反，camodel 绝对值双向不可信纪律再证；C 侧 MTE2 busy 71.97% = 最大真实工作管、V 侧 FLOWCTRL 92.48%、C MTE2 195 段 ≈ 96 槽×2+Q/w2，与真机 op（duty 68.5%/wait_id2 122.7us/vec 16.3%）结构一致。**首次采集失败如实记录**：`nohup &` 从不设防交互 shell 发射，发射侧 60s 超时 kill 直达模型进程（Model Stop 与 kill 时刻精确一致 17:43:57、无 Total tick、profile toml 全 0 字节、parse 报 `Get Init data failed`）⇒ `setsid nohup ... & disown` 脱离会话后重跑成功——长时采集工具级教训入 README §5-7。采集命令/环境/坑/分析导引详见 `mindstudio_export/case15_simulator_20261004/README.md`；用途 = decode D=256 **架构级选项**（gemm_v0 模板级 per-buffer 同步改造 / vector GEMV 路径重构 / 框架虚拟通道）立项评审的流水图依据（S11-5 规则内结论不依赖本导出件）。

**收尾补记 2（2026-10-04，用户流水图判读跟进：case15 MTE2 空泡结构量化）**：用户对 case15_simulator_20261004 导出件的判读「MTE2 是主管线且存在空泡」经 trace 精确量化证实：core0 cubecore MTE2 = 195 段（192 个 128KB 大段 mean 1.668us，速率 74.8GB/s/核贴 512B 行布局上限）+ **193 个间隙全部落在 636-653ns 极窄带（med 636ns / p99 653ns / max 653ns），合计 124.05us = 27.9% wall**——固定结构气泡（每装载段后一个），非随机空闲；槽周期 4.635us = 2×(1.674 装载 + 0.650 气泡)；MTE2 指令组成 336 条 MOV_OUT_TO_L1_MULTI_ND2NZ = 48 task × 7（Q1+K2+V2+w2×2）与 tiling 模型精确对账；间隙窗口内 MTE1/CUBE 活跃（gemm_v0 k-step 链：4 step × 每-step PIPE_M + prologue/epilogue 管级排空）。**关键算术**：每槽间隙总量 1.30us < 每槽装载 3.34us ⇒ "把装载塞进气泡"数学上装不下，掩盖的稳态目标只能是 MTE2 队列永不断流（duty→100%，wall→3.34us/槽 = 1.39×）——而这要求装载跨越 gemm_v0 调用边界不被 prologue 管级 WaitFlag<MTE2_MTE1>(id0) 收编，即守恒律（§S11-3）封锁的正是这条路。真机交叉：camodel 空泡 27.9% vs 真机 MTE2 idle 31.5%（duty 68.5%），结构一致（camodel 仅定性纪律不变）。方案空间两轴梳理（流水掩盖 11 项 / 访存利用率 7 项，排除 12 项已试/在案否决，幸存 3 项均立项级：A9 bN=128 手动 mma decode 变体+k/v DB【唯一未尝试的 kernel 内形态，预算同形共池 [2,128,128] 与生产 fbm 同构，分析外推待 §4.5-e 新行推导+毒化谱系核验】、A10 gemm_v0 模板级 per-buffer 同步改造【框架层最小改动面，改造后 S2 变体 mha_r111_edit_v1.py 精度全过态零探针复活】、A11 vector GEMV 重构【baseline 已证 1448GB/s/duty 89.6%】）见 2026-10-04 primary 会话综合（本补记为证据锚点）。访存利用率轴结论：busy 期速率（聚合 1496≈微基准 B 上限 1452，w2 命中抬高）与唯一字节（K/V 单遍无复用，L2 读命中 3.0%=仅 w2）双双贴物理上限 ⇒ 无独立杠杆，提高访存利用率在本 case 与 duty 恢复完全等价。

---

# §S12 会话 12（perf_session=12，2026-10-04/05）：A10 gemm_v0 模板级 per-buffer 同步改造（用户授权框架级，采纳）

## S12-1. 最终成果

**A10 采纳（本谱系首个框架级采纳态）**：case15 **508.11→463.58us（−8.76%）** / case18 **508.57→465.36us（−8.50%）**（同会话新鲜基线，n=25 median，3% 门 2.9 倍）。decode D=256 duty 墙（§S11 守恒律 + camodel 193 间隙 27.9% wall）被模板级 opt-in per-buffer 同步打破：MTE2 duty 68.3→73.7%，busy 速率零重排税（75.0→75.8 GB/s/核），wall 收益主体 = C.scalar_t −41.5us（气泡压缩实体），cube/vec/GM 字节三守恒（BITEXACT 一致）。roofline 比值 0.680x → **~0.742x**（case15 local 口径 463.58 vs 理想 ~344）；上界 1.39-1.45× 中已兑现 1.096×，残差 = G2 相 SEM 协议窗（升级臂移交，S12-5）。

**终态**：mha.py = **0fb193b2**（2152 行，perf17 同体）+ **框架 common.h/ascend.py opt-in 参数**（git 工作区态，未 commit——仅用户明示才提交）。**框架与算子成对生效**：cann-bench 复评前置 = build_wheel_ascend.sh 重打包 + 评测环境重装（setup.py 已携带 tl_templates，打包脚本零调整；optimization_log [RESULT-#R12-2] 附录 B 全文方案）。

## S12-2. 会话 12 优化项与逐轮实测

| 轮 | 内容 | 结果 |
|---|---|---|
| r12-1 (Phase A) | 可行性+设计（零生产变更）：模板细读 / 事件 id 三源查证（核内 SetFlag 无 ≤15 条款——属跨核 CrossCoreSetFlag；静态 Tensor 场景 id 6/7 禁用）/ plumbing 三通道（选 (b) header-only：模板串 = Python f-string 粘贴 + codegen 不解析 ⇒ 零 rebuild）/ 生效链（TL_TEMPLATE_PATH 实机核验 + pip 打包携带）/ 回归面（73 文件 186 调用点 + 4 UT）/ FIFO 稳态设计（**关键修正：per-buffer 同步只消除排空收编，不改发射点——S2 after-return 位仍时序中性，收益必要条件 = pre-call 发射点前移**）| **GO**（四项 NO-GO 判据逐项否证）|
| r12-2 (Phase B+C) | 框架最小改造（common.h +34/−3：kMTE2Sync 第 9 默认模板参 + prologue if constexpr；ascend.py +33/−2：mte2_sync kwarg 条件拼接）+ kv_db 域 S0-S6 七站点重设计（pre-call K 预取全局守卫跨轮 / G1 奇偶展开 mte2_sync 0/1 / SIG_VP FIFO 前缀 / V after-return + S3 轮头 priming / Q 重定位 S0/S4）| **采纳**：−8.76%/−8.50%，duty +5.4pp，全门禁绿（精度 8 层 + 非 mha 消费者 90/90）|
| r12-3 (Phase D) | 采纳态收口：docstring rev4r12 注记 + IR 同一性核验（测量态 == 采纳态）+ l0 复验 10/10 + 落档四件套 + 生效链方案 | 完成 |

守卫 case（IR 逐字节域）：case10 −0.69 / case11 +0.52 / case14 −0.13 / case16 −0.07 / case9 +0.23 / case7 +0.11% 带内；case20 +2.81% = 漂移带归因（IR 相同 + 14us 小核抖动 + 其余 6 守卫证环境稳定，session-8 先例口径）。

## S12-3. 会话 12 结构性发现（新增硬知识）

1. **守恒律破解条件精确化**：#R11-1 守恒律（管级 prologue 排空收编预取）的完整表述 = 「排空收编」+「发射点钉死」双重封锁。A10 只解除前者；后者要求装载发射点前移至 pre-call 位（iter idx 标量前段发 K(idx+1)），使每笔装载获得整个 gemm 调用期的流式窗口——after-return 位（S2/现生产）在两种同步模式下均时序中性。稳态 = MTE2 队列永不断流（发射序 = 消费序 FIFO 不变量下，队列位置事件 = per-buffer 事件）。
2. **核内 vs 跨核旗标约束分界**：「同一 flagId ≤15 次 set」属 CrossCoreSetFlag（跨核信号量计数器）；核内 SetFlag/WaitFlag(ISASI) 文档约束 = 成对出现 + eventID 0-7（静态 Tensor 场景 6/7 禁用，开发者自管）。核内旗标 1-bit vs 计数器语义无文档明文 ⇒ 鲁棒化设计 = 每 (type,id) 严格 Set→Wait 1:1 交替 outstanding≤1（对两种语义同时安全；form_a SIG 令牌 + gemm_v0 k-step 链生产先例）。
3. **通道 (b) header-only plumbing 模式**：gemm_v0 模板实参串 = 前端 f-string 全文粘贴（codegen GemmOpCodegen 不解析、op 注册按名识别、L0 TBuf 全局声明）⇒ 追加默认模板参 = 零 C++/零 rebuild 的框架扩展通道；默认路径逐字节不变可机器证明（17/17 IR dump diff + 拼接重构逐字符核验 + 90/90 消费者 UT）。运行期事件 id 发射已有生产先例（form_a SIG_L0AB+side）；运行期模板参不存在 ⇒ 消费点奇偶展开是编译期 id 绑定的唯一形态。
4. **重排税的符号条件**：r11 S2（交叠无间隙消除）busy 速率 −2.7% vs r12 A10（交叠伴间隙消除）+1.1% ——带宽墙域流序重排的引擎税只在收益缺位时显性为净负；duty 兑现时税为二阶（mte1_t +3.2% = L1 双主控并发读写的真实二阶项，已计入净收益）。
5. **框架级采纳的生效链条款**：框架资产（header + 前端 Python）随 wheel 分发 ⇒ 采纳判决必须携带「重打包 + 评测环境重装 + 缓存隔离」方案（本轮实证：TL_TEMPLATE_PATH install 态优先包内 src/，develop 态回退源码树；opt-in shape 模板串变化 ⇒ cache key 必变，无毒缓存路径）。

## S12-4. 精度最终状态（收口，终态 = 0fb193b2 + 框架 opt-in 参数）

采纳态全门禁：smoke 6/6 BITEXACT 无挂死 → 探针 22 目标 + 12 守卫 BITEXACT vs perf16（含新同步形态全拓扑：kv{2,3,4}/kv{9,10,16} 跨轮/causal×3/尾块 640/300/bf16/scale/my_count{1,2,mixed}/最小 1×1/zeros + golden 阻塞口径）→ 锚-A 9/9 + 锚-B 14/14 → 压力继承 4 + 扩展 17（kv_db 被改路径专项）×12 nondet=0 → l0 10/10 + l1 19/19 + coverage 19/0 + all EXIT=0（哨兵 4/4）→ 非 mha 消费者 4 UT 90/90 → 红线四项 = AIV 逐字节相同域 → docstring 后复验 l0 10/10 + IR 同一性 True。位级档案链：perf16（改前）/ e8a9d2da（测量态）/ perf17=0fb193b2（采纳态）。

## S12-5. 遗留问题与建议（会话 12 终态评估）

1. **升级臂（移交下一会话，需独立立项预算）**：G2 相残差 = SEM_WS2 协议窗内 MTE2 无可发流（v_l1 单槽 + L1 400/511.75KB 预算封死 bN=256 第二 V 槽）⇒ **bN=128 缩块 + k/v 双侧 DB（268KB）+ 同模板 opt-in 模式**（= S1 形态在 A10 解锁后复活；r11 对 S1 的 EV 否决依据是管级同步下的守恒律，该前提已被本轮废除）。工程量级 = S1 级全形重构（bN 涟漪：host 谓词/工作区形状/SEM 轮计数 ×2/V 侧 block_N 参数化域/探针矩阵重基线 + 槽×2 事件税实测），需 Phase-A 级设计先行；预期空间 = duty 73.7→~90%（wall 再 −10~15%，推测性 EV 未实测，如实标注）。
2. **duty 73.7% 与 85%+ 目标差**的其余成分（轮/任务边界先导、Q 载入、S3 轮头 V 暴露）均为 <0.1us/slot 级二阶项，升级臂实施时一并覆盖。
3. **Orchestrator 轮末统一项**：build_wheel_ascend.sh 重打包 + 评测环境重装 + cann-bench 复评 + mha_submission/ 同步（**框架 + 算子成对提交**——仅算子提交将退回默认路径 = 收益归零）。git commit 仅在用户明示时执行（common.h/ascend.py/mha.py 三文件 + perf_tuning 档案）。
4. case20 类小核 IR-相同域 wall 漂移（+2.8%）建议在守卫口径中固化为「IR 逐字节域免计时归因」条款的注脚先例（本轮沿用 session-8 口径，无新风险）。

## S12-6. 会话 12 产物清单

- 生产态：examples/mha/mha.py（0fb193b2，2152 行）+ src/tl_templates/ascend/common.h + tilelang/language/ascend.py（opt-in 参数，工作区态）
- 备份链：history_version/mha_impl_r4_perf16.py（改前 dda3ad08）/ mha_impl_r4_perf17.py（采纳态同体）/ /tmp/opencode/r12_framework_backup/{common.h.pre_r12, ascend.py.pre_r12, framework_r12_phaseB.diff（git diff 全文 124 行）}
- IR 档案：/tmp/opencode/r12_src_base（17/17 == r11_src_base = 默认路径不变机器证明）+ /tmp/opencode/r12_src_var（15 同 + 2 目标变更）+ r12_probes/kv10_var.cc（多轮运行期 id 发射核验）
- 测量档案：/tmp/opencode/r12_base/{wall,op}（新鲜基线 + op duty 基线）+ /tmp/opencode/r12_iter1/{wall,op}（采纳态）+ op_parsed.json 双份 + baseline_r12_iter2.json
- 门禁日志：/tmp/opencode/r12_probes/r122_{smoke,probe,anchors,anchor_stress,stress_ext,l0,l1,coverage,all}.log + gates_runner.log + perf_r122.log + baseline_r12.log + dump_{base,var}_r12.log
- 工具：/tmp/opencode/r12_probes/{make_r122.py, dump_{base,var}_r12.py, dump_kv10_r12.py, smoke_r122.py, probe_r122_kvdb.py, stress_r122_kvdb.py, run_{baseline,wall,perf,gates}_r12*.sh, parse_op_r12.py, quickcheck_base.py}
- 变体留存：/tmp/opencode/r12_probes/mha_r122_edit_v1.py（e8a9d2da，2120 行 = 测量态）
- 缓存：/tmp/opencode/r12_cache_{base,var,g1,final,consumer}（专属隔离全程）

## S12-7. cann-bench 权威复评（Orchestrator 补充执行，cann_final_eval_20261005_111311）

会话 12 子任务按分工纪律未执行权威复评与提交副本同步。Orchestrator 侧独立验证（全部通过后才同步副本并复评）：

1. **`test_mha.py --level all` 独立复跑 exit 0 "Test Passed!"**（L0 10/10 + L1 19/19 + L2 12/12 正确拒绝 + Boundary 6/6 + 哨兵 4/4 PERF_PASS，kernel_path 6.3~7.4s ≤120s 门；专属缓存 r12_cache_orch，日志 /tmp/opencode/r12_orch_all_rerun.log）。
2. **终态/备份链核对**：mha.py 0fb193b2（2152 行）= history_version/mha_impl_r4_perf17.py；perf16 = dda3ad08（改前）；框架备份 common.h.pre_r12（d113a6c6）/ascend.py.pre_r12 + framework_r12_phaseB.diff 在案；git 工作区仅含意图内框架改动（common.h +37/ascend.py +35，未 commit——3rdparty/tvm buffer.py 脏标记为 9 月 17 日项目期既有适配补丁，非本会话产物，已核 mtime+diff 内容）。
3. **生效链核验（附录 B 假设的实测修正）**：评测发射环境（set_env.sh + cann-bench PYTHONPATH）下 `python -c "import tilelang"` 解析 = **源码树**（0.1.4+0d5bce5d），env.py L177-182 develop 回退 ⇒ TL_TEMPLATE_PATH = 仓内 src/（改后 common.h 82e80025 生效）——**附录 B 预设的 build_wheel_ascend.sh 重打包+重装对本次复评不必要**（偏差如实记录；wheel 成对分发条款对**外部分发/提交**场景仍然有效）；复评进程日志首行 EVAL_TILELANG=源码树 + 提交副本 md5 0fb193b2 双落档。
4. **提交副本同步**：mha_submission/cann_bench/mha.py dda3ad08→**0fb193b2**（旧副本备份 /tmp/opencode/mha_submission_backup_before_r12_sync.py）；run_evaluation.sh 自动完成算子包重打包+pip 重装。
5. **门禁日志抽查**：r122_probe（PASS）/r122_anchors（9/9+BATTERY PASS）/r122_all（EXIT=0）/r122_anchor_stress（PASS）与子任务报告逐项吻合；[RESULT-#R12-2] UT 90/90 算术自洽（44+32+8+6）。

**权威结果**（复评命令与会话 3-8 同口径：--device-id 0 --processes-per-card 1 --timeout-per-process 300 --warmup 3 --repeat 5 --profiler-level Level1 --perf-batch-case-memory-limit-mb 32768 --perf-batch-cases，新鲜评测缓存 r12_eval_cache，EVAL_EXIT=0，原始日志 /tmp/opencode/r12_cannbench_reeval.log）：

| 指标 | 会话 12 起点（=20261003_122353） | 会话 12 终态 | 变化 |
|------|-----------------------------------|--------------|------|
| 综合得分 | 72.63/100 | **72.86/100** | +0.23 |
| 平均加速比 | 0.9592x | **0.9656x** | +0.0064 |
| 通过率 | 20/20 | 20/20 | 保持 |
| **case15** | 507.90us / 0.680x | **463.94us / 0.744x** | **-8.7% / +0.064**（与本地 -8.76% 吻合） |
| **case18** | 504.34us / 0.811x | **464.88us / 0.879x** | **-7.8% / +0.069**（与本地 -8.50% 吻合） |

**核验**：目标双 case 收益经权威口径确认为真实（非本地口径估算）；**decode 不回退硬约束 5/5 达标**（case10 0.990≥0.90 / case13 0.891≥0.69 / case14 1.390≥0.99 / case16 0.777≥0.52 / case20 2.113≥0.82）；其余 18 case 全部 ±0.025 加速比漂移带内（最大 case20 -0.025 / case13 +0.024，均属 IR 逐字节相同域的评测方差，session-8 先例口径；case12/13 的 -1.7/-2.7% wall 漂移同归因）——**无一实质回退**。回退条款不触发（72.86>72.63 且 0.9656>0.9592）。

**终态声明**：生产 = mha.py 0fb193b2 + 框架 opt-in 参数（common.h/ascend.py，git 未 commit）——**框架与算子成对**：任何一侧单独回退/分发都会使 kv_db 域行为失配（成对提交纪律入 journal e1）。升级臂（bN=128 k/v 双侧 DB，EV -10~15% 推测性外推）移交下一会话独立 Phase-A 级设计。

---

# §S13 会话 13（perf_session=13，2026-10-05）：升级臂 bN=128 缩块 + k/v 双侧 L1 DB（用户授权，采纳）

## S13-1. 最终成果

**#R13-1 采纳（纯算子侧，框架零触碰）**：case15 **463.64→421.64us（−9.06%）** / case18 **465.12→423.56us（−8.94%）**（同会话新鲜基线，n=25 median，3% 门 3 倍，Δ>20σ 双目标同向）。r12 duty 残差主体（G2 相 SEM_WS2 协议窗内 MTE2 无可发流）被双侧 DB 消除：bN 256→128 腾出 L1 使 v_l1 获得第二槽 ⇒ V(idx+1) 发射点从 after-return（时序中性位）前移至 **pre-call S5'**（Set SIG_VP 之后 ⇒ 不入 gemm2(idx) FIFO 前缀 ⇒ 流式穿越调用期 + 填充下一 SEM_WS2 协议窗）。**duty 73.5→86.4%（+12.9pp），wait_id2 119.6→66.4us（靶心实体）**；cube_t 与 GM_to_L1 字节精确守恒（k-step 守恒 768/核 + 512B 行布局无税的机器证明）。roofline 比值 ~0.742x → **~0.82x**（case15 local 口径 421.64 vs MTE2 地板 ~342）；r11 时代「规则内穷尽」声明中的 S1 臂在 A10 解锁后实测兑现（时代对照知识，S13-3-①）。

**终态**：mha.py = **c316a004**（2240 行，perf19 同体）+ 框架 common.h/ascend.py **r12 opt-in 参数态不变**（82e80025/1c3a00ed，本会话零触碰）。**成对提交条款沿用 r12**（算子依赖框架 opt-in 通道；缺一收益归零）。

## S13-2. 会话 13 优化项与逐轮实测

| 轮 | 内容 | 结果 |
|---|---|---|
| r13-1 (Phase A) | 全形设计（零生产变更）：bN 涟漪全清单（host 派生/L1 池/kL0 档 S2-3-1 规则复核/V 侧参数化域/SEM-批深交互）+ 分派拆分决策（单块域 S_kv≤256 legacy 逐字节）+ 事件 id 预算审计（3/6 零新增 ⇒ 框架零改动）+ 精度域分化锚计划（锚成员全员域外核验）+ scalar 固定段预算（EV 诚实降档 [−4,−16]%）| **GO**（五项 NO-GO 判据逐项否证）|
| r13-2 (Phase B+C) | 五区域实施（host bN 分支 + kL0 收窄 + v_l1 三元式 + S3 静态槽 + S5' pre-call/消费点运行期切片）；IR 守卫 20 shape（17 逐字节 + 3 目标）；L1 读数 274432B=268KB；needClear 新规则全文入库；smoke 6/6 → 探针 24+12 → 锚 → 压力 4+19 → l0/l1/coverage/all 全绿；新鲜基线 + 变体同口径 + op case15 | **采纳**：−9.06%/−8.94%，duty +12.9pp，签名 (a) 兑现 |
| r13-3 (Phase D) | 采纳态收口：docstring rev4r13 + IR 同一性（3a6bf589）+ perf19 备份 + 终态新鲜复跑全绿 + 落档四件套 | 完成 |

守卫 case（IR 逐字节域）：case11 −1.89 / case14 −0.35 / case16 −0.23 / case20 +0.28 / case9 +0.18 / case7 +0.06% 带内；case10 +1.12% = 漂移带归因（IR 相同 + 34.9us 小核发射抖动，session-8/r12 先例口径）。

## S13-3. 会话 13 结构性发现（新增硬知识）

1. **缩块换 DB 窗口定律（同步体制依赖）**：同一结构改动（bN÷2 + 双侧 DB）的净符号由同步体制决定——r11 管级排空时代 S1 推演负（守恒律：缩块纯付槽数×2 税），r13 per-buffer 时代实测 −9%（缩块买到的 pre-call 窗口 → duty +12.9pp 压倒固定段税 +74.9us）。**「推演性关闭」结论必须绑定其体制前提归档**（本谱系方法论修正：r11 的 S1 否决依据在 A10 后自动失效，任务书复活条款的机制根据）。
2. **needClear codegen 规则全文（r11 档案精化）**：`need_clear = is_zero(dst_offset) ∨ 版本化整行例外`（realTailM 为编译期 IntImm ∧ tail_m==dst_m ∧ CanProve(offset % (dstM×dstN))==0，codegen_ascend.cc L2979-3045，commit 059f0a65/#1723）；运行期尾块拷贝（T.min 不折叠形）恒 false；**满 tile 载入 needClear=true 时 helper 触发条件短路 = 零运行时成本**（tailM==dstM ∧ tailN==dstN ∧ fractalExtent==dstM×dstN）。配套精度工具：**有限 stale 归纳证明**（槽首载满宽 by 域界 ⇒ stale 行恒有限 ⇒ 掩码列 P=0×有限=0 无 NaN 通路）= V 侧（无下游 select 替换）的必备论证，与 r11 K 侧 select 替换制互补。
3. **k-step 守恒设计规则**：缩块臂的 kL0 档位第一原则 = bN÷2 ∧ kL0×2 ⇒ 每核步数守恒（768）⇒ cube_t/每-step 标量链零增量（实测 cube_t 49.80→49.80 精确命中）；kL0=256 单步档被 #R4-2a 实测禁令排除（ping-pong 失效）。
4. **FIFO 前缀单 id 覆盖双槽 DB 的判据**：V 侧 pre-call 位（Set 后）⇒ 单 id SIG_VP 足够；K 侧预取跨前一 Wait ⇒ 奇偶 id 必需。判据 =「同型 buffer 的两个 Set 是否可在一个 Wait 之前 outstanding」。事件 id 预算 3/6，双侧 DB 零新增（框架零改动的根据）。
5. **V 侧工作量缩放律（定量化）**：phase2（D 宽 w3 往返）∝ 槽数（+50% 字节：C.fix_t +34.8us / V.mte2_t +47.2us 实测）；phase1 ∝ 元素数（不变）。余量判定：V wait 空闲 432→385us 仍富余 ⇒ 无 pacing 反转——**槽数×2 臂的 V 侧税基线公式**。
6. **新 pacing 地板 = C.scalar 管**：scalar_t 411.0 ≈ c.wall 412.4（99.7%）——duty 86.4% 的残差主体是每槽 scalar 链（384 调用固定段 + 768 步链 + 192 槽 SEM/copy 指令）。本域规则内杠杆穷尽（S1/S2/S3/A10/S13 链闭合）；进一步压缩 = scalar 路径结构级重构（超 Stage-3 规则范畴）。

## S13-4. 精度最终状态（收口，终态 = c316a004，框架 r12 态不变）

非 BITEXACT 域（bN 变更 ⇒ rescale 粒度 + GEMM2 K 链）主门 = golden 阻塞口径：探针 24 目标 matched=1.00000 全体、MARE_abs 余量 160-640x（T_valrange 2x = 既存 golden 栅格属性，vs_old BITEXACT 证明非本轮引入）；12 守卫 BITEXACT vs perf18（分派拆分零泄漏：D256 单块×2/D128×5/D64/form_a×2/fbm×2）；锚-A 9/9 BITEXACT 硬守卫 + 锚-B 14/14（成员全员域外 = S13-4c 核验兑现，无降级项）；smoke 6/6 无挂死；压力继承 4 + 扩展 19（双侧 DB 槽轮转/跨轮/任务边界/奇偶批/双槽尾块/掩码翻转 640 型）×12 nondet=0；l0 10/10 + l1 19/19 + coverage 19/0 + all EXIT=0（哨兵 4/4）；采纳态终态新鲜复跑全绿（r133 日志族）；红线四项定位复核（栅格链逐元素同一/预归一特化 legacy 隔离/sumexp+1e-30 逐字保留/掩码链 bN 参数化等价 + 640 迁移先例）。位级档案链：perf18（改前 0fb193b2）/ 22fffbe5（测量态）/ perf19 = c316a004（采纳态）。消费者 UT 门不适用（框架零改动，如实登记）。

## S13-5. 遗留问题与建议（会话 13 终态评估）

1. **本域收口声明（任务书 §1 条款的采纳分支）**：decode D=256 kv_db 域规则内杠杆穷尽（五臂链条闭合），残余 = scalar 固定段地板 + mte2 细流速率税（−6.3%，64KB×2 频次+四目的区交替的引擎级成本）——**收口决策交还用户**；若未来立项，唯一方向 = scalar 路径结构级重构（如 G2 相 SEM/copy 指令瘦身或调用合并），属新 kernel 形态范畴。
2. **case15/18 bench 预期（换算预登记，权威 = Orchestrator 复评）**：case15 0.744→~0.818x / case18 0.879→~0.965x（本地相对口径）；decode 不回退 5 线全为 IR 逐字节域预期带内不变（case10 本地 +1.12% 漂移带已在守卫口径条款内）。
3. **Orchestrator 轮末统一项**：cann-bench 复评 + mha_submission/ 同步（**成对提交条款沿用 r12**：本轮为纯算子改动，但运行依赖 r12 框架 opt-in 参数态——提交副本同步时框架 diff 必须随同）；git commit 仅用户明示时执行。
4. **知识资产留存**：/tmp/opencode/r13_probes/（变体全文件 + make/dump/smoke/probe/stress/gates/wall/perf/final 脚本族 + 全门禁日志）+ r13_{base,iter1}/（基线与变体全测量数据 + op_parsed 双份）+ r13_src_{base,var}（20 shape IR 档案，base==r12 采纳态机器证明）——防重复探索。

## S13-6. 会话 13 产物清单

- 生产态：examples/mha/mha.py（c316a004，2240 行）；框架 = r12 态零触碰（common.h 82e80025 / ascend.py 1c3a00ed）
- 备份链：history_version/mha_impl_r4_perf18.py（改前 0fb193b2）/ mha_impl_r4_perf19.py（采纳态同体）
- IR 档案：/tmp/opencode/r13_src_base（20 shape，17 共享 shape == r12_src_var 逐字节 = provenance 链）+ r13_src_var（17 同 + 3 目标变更）
- 测量档案：/tmp/opencode/r13_base/{wall,op,op_parsed_case15.json}（新鲜基线）+ r13_iter1/{wall,op,op_parsed_case15.json}（采纳态）
- 门禁日志：/tmp/opencode/r13_probes/r131_{smoke,probe,anchors,anchor_stress,stress_ext,l0,l1,coverage,all}.log + r133_{anchors,anchor_stress,stress_ext,coverage,all}.log + gates_runner.log + final_runner.log + perf_r131.log + baseline_r13.log + dump_{base,var}_r13.log
- 报告落档：optimization_log.md（[ANALYSIS]-S13/[DB-ANALYSIS]/[DESIGN-S13]/[IMPL+SELF-CHECK+RESULT-#R13-2]/r13-3 收口）+ perf_log.md（r13-1/2/3）+ baseline_r13_iter{1,2}.json + 本报告 §S13

## S13-7. cann-bench 权威复评（Orchestrator 补充执行位——本轮子任务按分工纪律未执行，留空待填）

## S13-7. cann-bench 权威复评（Orchestrator 补充执行，cann_final_eval_20261005_155454）

会话 13 子任务按分工纪律未执行权威复评与提交副本同步。Orchestrator 侧独立验证（全部通过后才同步副本并复评）：

1. **`test_mha.py --level all` 独立复跑 exit 0 "Test Passed!"**（L0 10/10 + L1 19/19 + L2 12/12 正确拒绝 + Boundary 6/6 + 哨兵 4/4 PERF_PASS；专属缓存 r13_cache_orch，日志 /tmp/opencode/r13_orch_all_rerun.log，起跑 md5 c316a004 落档）。
2. **终态/备份链核对**：mha.py c316a004（2240 行）= perf19；perf18 = 0fb193b2（改前）；**框架文件 r13 零改动**（common.h 82e80025 / ascend.py 1c3a00ed = r12 态，git diff --stat 与 r12 档案逐字一致 +67/-5）；3rdparty/tvm 脏标记 = 9 月 17 日既有补丁（非本谱系产物，r12 已核）。
3. **门禁日志抽查**：r131_probe（PASS，24 目标阻塞口径 matched=1.00000 / MARE_abs 6.1e-05 余量 160x + 12 守卫 BITEXACT）/ r131_anchors（9/9 + BATTERY PASS）/ r131_stress_ext（PASS）/ r133_all（EXIT=0）——与子任务报告逐项吻合。
4. **生效链**：评测发射环境 python 解析 = 源码树 tilelang（EVAL_TILELANG 落档复评日志首段）+ 提交副本 md5 c316a004 + common.h 82e80025 三点落档；框架零改动 ⇒ r12 生效链结论直接沿用（wheel 成对分发条款对外部提交仍有效）。
5. **提交副本同步**：mha_submission/cann_bench/mha.py 0fb193b2→**c316a004**（旧副本备份 /tmp/opencode/mha_submission_backup_before_r13_sync.py）。

**权威结果**（同口径：--device-id 0 --processes-per-card 1 --timeout-per-process 300 --warmup 3 --repeat 5 --profiler-level Level1 --perf-batch-case-memory-limit-mb 32768 --perf-batch-cases，新鲜评测缓存 r13_eval_cache，EVAL_EXIT=0，日志 /tmp/opencode/r13_cannbench_reeval.log）：

| 指标 | r12 终态（20261005_111311） | r13 终态（20261005_155454） | 三轮累计（起点 20261003_122353） |
|------|------------------------------|------------------------------|----------------------------------|
| 综合得分 | 72.86/100 | **73.11/100** | 72.63 → 73.11（+0.48） |
| 平均加速比 | 0.9656x | **0.9747x** | 0.9592 → 0.9747（+0.0155） |
| 通过率 | 20/20 | 20/20 | 保持 |
| **case15** | 463.94us / 0.744x | **418.00us / 0.826x** | 507.90→418.00（**-17.7%**）/ 0.680→0.826 |
| **case18** | 464.88us / 0.879x | **424.04us / 0.964x** | 504.34→424.04（**-15.9%**）/ 0.811→**0.964（逼近基线）** |

**核验**：目标双 case 权威口径 -9.9%/-8.8% 与本地 -9.06%/-8.94% 吻合（子任务预登记换算值 0.818/0.965x vs 权威 0.826/0.964 = 命中）；**decode 不回退 5/5 达标**（case10 0.974≥0.90 / case13 0.876≥0.69 / case14 1.397≥0.99 / case16 0.790≥0.52 / case20 2.169≥0.82）；其余 18 case ±0.03 内（最大 case4 -0.030 = IR 逐字节相同域评测方差，case20 +0.056 / case1 +0.026 正向漂移同归因）——无一实质回退；回退条款不触发（73.11>72.86 且 0.9747>0.9656）。

**终态声明**：生产 = mha.py **c316a004**（2240 行）+ 框架 opt-in 参数（r12 态，git 未 commit）——成对提交纪律沿用。**decode D=256 域规则内杠杆链条闭合**（S1 推演负→前提废除→S13 实测采纳 / S2 实测负 / S3 门 FAIL / A10 采纳 / mma 化永久否决 / vector 重构超范畴）；残余墙 = C.scalar 固定段地板（411us≈99.7% wall）+ mte2 细流速率税（-6.3%），进一步压缩需 scalar 路径结构级重构（超 Stage-3 规则范畴，是否立项 = 用户决策）。
