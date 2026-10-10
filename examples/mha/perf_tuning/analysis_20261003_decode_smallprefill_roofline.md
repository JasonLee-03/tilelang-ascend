# MHA decode / MTP / 小 prefill 域 roofline 与利用率专项分析

- 日期: 2026-10-03/04 | 类型: **纯分析轮（mode=analysis，零生产变更）**
- 对象: `examples/mha/mha.py` 生产态（**1926 行，md5 `dda3ad0877ba4df74bbd559f6912eeed`**——开工/收工双核验一致，全程零写入）
- 权威评测: `/mnt/workspace/cann/20260917_mha_V01/cann-bench-20261002/reports/cann_performance_eval_20261003_122353.json`（20/20 pass，综合 72.63，平均 0.96x）
- case 定义: `cann-bench-20261002/tasks/level4/mha/cases.yaml`（case10-18 note 中 Skv 字样与实际不符，**一律以 input_shape 为准**）
- 本轮全部新鲜数据: `/tmp/opencode/decode_roofline_a1/`（wall/ op/ base_*/ micro_* 各子目录 + roofline_rows.json + op_parsed.json）
- 硬件常数: **直接引用** `analysis_20260928_theoretical_perf_and_ideal_pipeline.md` §0（Cube 峰值 270.3T / 持续 202.5T fp16；HBM 读 1472 / 写 1481 / 混合 1250 GB/s；L2 ≥4TB/s、168MB；C.MTE2 持续 125.2、C.MTE1 191.3、C.FIX 64.1、V.MTE2 56.8、V.MTE3 30.6 GB/s/核；launch floor ≈9us）——本轮新数据与之无矛盾（微基准连续装载 20 核聚合 1469 GB/s = HBM 读峰 1472 的 99.8%，交叉验证硬件常数表可信）。
- 判读纪律（继承 §S6-3-3 / gate0c 纪律 #1）: **真机 msprof = 唯一定量依据；camodel 本轮完全未运行**；微基准数字标注"算子外隔离实验"性质。

---

## [ANALYSIS_VERDICT]

**decode 域（case10-15）的实测有效带宽 622-999 GB/s 不是实现缺陷，而是 BSHD 布局下 256B/512B 跨步访存的 HBM 突发效率物理墙（H1）+ gemm_v0 每-step 排空导致的 MTE2 占空缺口（H2）的叠加，且已被算子外微基准与 baseline 对照双向钉死。** 具体判决：**(1) D=128 decode（case10/11/12/13/16/17，占 decode 族 6/9）= H1 布局墙主导，无规则内空间**——256B 行/8KB 跨步的纯装载微基准 20 核聚合上限仅 966 GB/s（=连续对照 1469 的 66%），单核 56.3 GB/s（=连续 250 的 22%，4.4× 突发惩罚）；生产实测每槽 wall 5.80us 中 94% 就是这条 K/V 布局墙（5.43us），只剩 6% 是 duty；**CANN baseline 自己在 case11 上也被同一堵墙钉在 ~930 GB/s（其 cube 路径 MTE2 duty 94.6% 已近饱和，busy 速率 49 GB/s/核 = 我方微基准聚合 48.3）**——即 baseline 对 D=128 decode 并无带宽优势，eval 中 case11/12/13 的 0.87-0.93x 差距主体来自记账条件差（本轮 baseline 新鲜实测 300.52us vs eval 记账 285.67us）与我方固定开销/尾块效应，而非带宽效率（双方 wall 平均速率同级 46-47 GB/s/核）。**(2) D=256 decode（case15/18）= H2 duty 主导，理论有 ~1.45× 空间但被既有硬限锁死**——512B 行布局墙高达 1452 GB/s（≈峰值），baseline 用 vector IncreFlash 路径（MTE2 duty 89.6%）实测达 1448 GB/s 证明空间真实存在；我方走 cube gemm_v0 路径 duty 仅 68.5%（每槽 5.25us 中 31% 是排空气泡），实测 999 GB/s = 布局墙的 69%。但提升 duty 的两条路都被在案硬限否决：mma 化撞 §4.5-e D2' 行 L0B 分池 128KB>64KB + bN 上界定理（S10-3-1/2）；k/v L1 双缓冲撞 §4.5-a D2 行 272KB→528KB>511.75KB L1 硬限（五环证伪链第 3 环）。**(3) MTP（case16/17/18）**：16/17 = D=128 布局墙（同(1)），18 = D=256 duty（同(2)）；S=2 的 causal 填充效率 0.12 在带宽墙下无关紧要（归档维持）。**(4) 小 prefill（case1/2/5/6/19/20）**：wall = launch floor(~9us) + V-softmax 稳态 + 协议交接三项分解——case1/5/20 floor 占比 59-92%（floor 墙，B 层上界即 floor，无带宽/算力杠杆）；case2 V-paced（vec 69.4%，与 case7 同构，w1 瘦身已被 §S7-3-3 双 régime 关闭且本案 fixp 非 pacing 同样适用）；case6/19 混合（mte2 34-52% + fix 16-29% + vec 49-53%，无一独占 pacing），已达 1.27/1.43x；case6 B 层上界 2.61x 的差额主因 = L2 流量界 12.6us + floor，其中 w1/w2/w3 往返已证 L2 介质（本轮 decode 域 L2 写命中 84-98% 再证 gate0c 事实 #2），消除它的转化率已被 gate0c 实测为 0.12-1.54%（工作消除侧关闭）。**逐族判决：decode D=128 = 墙确认（H1，含 baseline 同墙实证）；decode D=256 = 条件空间（H2，~1.45× 上界，前置条件 = 推翻 §4.5-e L0B 预算或引入 vector 路径重构，均超出规则内杠杆）；MTP = 同 decode 分域判决；小 prefill = floor/V-softmax/L2 介质三墙，case2 有 V 相干化残值但撞 ③ 族规则内永久关闭（S10-3-5），其余无空间。** 归档结论复核：§4.5-e **维持**（新增 baseline 对照与每槽分解两项独立证据）；§S4-3-3 **修正细化**（"wall=每槽 MTE2 带宽"对 D=128 强化成立、对 D=256 需拆分——带宽上限非墙、duty 才是墙，"协议结构已非约束"对 D=256 不成立）；20260928 建议⑤ **修正**（"mma 化+宽流"的收益前提在 D=256 被证实、机制被预算否决维持，D=128 侧连收益前提都不存在——布局墙内 mma 化无 EV）；decode 不回退 bench 口径值全部维持（现值 case10 0.996/13 0.867/14 1.391/16 0.783/20 2.138，与本轮新鲜 bench 一致）。

---

## 0. 数据来源与口径总表

| 数据 | 来源 | 口径 |
|---|---|---|
| bench wall / baseline / speedup | `cann_performance_eval_20261003_122353.json`（2026-10-03 12:23） | cann-bench 权威（含每-launch 固定开销记账） |
| 本地 wall（17 case 新鲜） | `/tmp/opencode/decode_roofline_a1/wall/*.json` | 纯 `msprof --task-time=on` → op_summary Task Duration，warmup 5 / iters 20 → n=25 **median**，parse_prof.py 解析（gate0 iter24 同口径）；**同会话连续采集 15:24-15:31 UTC**；专属缓存 `cache_wall` |
| op 模式 pipe/L2/Memory（17 case 新鲜） | `/tmp/opencode/decode_roofline_a1/op/case*/OPPROF_*/{PipeUtilization,Memory,L2Cache,OpBasicInfo,...}.csv` | `msprof op --kernel-name=main_kernel --aic-metrics=PipeUtilization,ResourceConflictRatio,L2Cache,Memory,BasicInfo,ArithmeticUtilization,MemoryUB,MemoryL0`，单发（warmup 0/iters 1），20 block 跨核均值 + max/min（iter24/26 同口径） |
| baseline op 模式（case15/11） | `/tmp/opencode/decode_roofline_a1/base_case{15,11}/` | `torch_npu.npu_fused_infer_attention_score`（BSH 零拷贝视图，num_heads=N，scale=1/√D），`msprof op --kernel-name='*'`；case15 TaskDur 347.46us vs eval 345.29 ✓ 吻合 |
| 微基准（MTE2 装载形态矩阵） | `/tmp/opencode/decode_roofline_a1/micro_*/` + `micro_mte2.py` + `run_micro.sh` | **算子外隔离实验**：独立 tilelang Expert kernel（4 pass_configs 全 False，与生产同构），纯 GM→L1 `T.copy` 循环 ×200 iter，生成码核验 `copy_gm_to_l1<half,512,128>(..., stride 4096, ...)` = 8KB 跨步/256B 行逐字复现生产 decode 访存；msprof timeline median |
| r7 时代 case6 真机 op | `/tmp/opencode/r7_op_case6/`（仍在，TaskDur 27.46us） | 复用性判定：case6 = Form A 单块域，r8 flat 只触 generic 多块域 ⇒ IR 逐字节未变；本轮新鲜 op（C wall 24.2us，fix 28.5%）与 r7 归档"真机 fixp 28.8%"一致 ⇒ **新鲜数据为准，r7 作交叉印证** |
| roofline 模型 | `analysis_20260928_roofline_model.py`（复用结构/常数）+ 本轮 `roofline_recompute.py`（新 wall 重驱） | A/B 层定义不变；w1/w2/w3 GM 往返按 gate0c 事实 #2 标注 **L2 介质**（本轮 decode 域 L2 写命中 84-98% 独立再证） |
| 漂移检查 | 本地 wall vs `msprof_r10_base`（case10 34.22/34.61、case14 377.41/378.26、case16 559.59/562.01、case20 14.14/13.98） | 全部 ±1.5% 带内 ⇒ 环境稳定，无跨会话漂移；case10 历史 +4.6% 单调漂移先例（会话 9）未复现 |

**生产完整性**：`mha.py` md5 开工 = 收工 = `dda3ad0877ba4df74bbd559f6912eeed`（1926 行）；`test_mha.py`/DESIGN.md/`mha_submission/` 零触碰；`.orchestrator_state.json` 未读写。全部探针/微基准/采集产物在 `/tmp/opencode/decode_roofline_a1/`。

---

## 1. 逐 case roofline + 利用率总表（Q2 主表）

约束常数：Cube 持续 202.5 TFLOPS（峰值 270.3）；HBM 读 1472 / 混合 1250 GB/s；floor 9us。
AI 岭点 = 202.5T/1250G ≈ 162 FLOP/B。**decode/MTP 全族 AI = 1-2 FLOP/B，距岭点 2 个数量级 ⇒ 全族深度内存绑定，算力利用率 <1%（量化见表中 %cust 列，0.3-1.0%）。**

| case | 族/form | bench us | local us | baseline us | now(bench) | FLOP(G) | uniq HBM(MB) | AI | 实测 TFLOPS | %of 202.5T | 实测 GB/s(bench) | %of 1472 | roofline 约束 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 小pf/A | 10.38 | 9.81 | 10.00 | 0.96 | 0.03 | 0.8 | 43 | 3.3 | 1.6% | 76 | 5% | **floor**（92%） |
| 2 | 小pf/A | 45.54 | 45.05 | 32.80 | 0.72 | 0.54 | 6.3 | 86 | 11.8 | 5.8% | 138 | 9% | **V.vec 稳态**（69.4%）|
| 5 | 小pf/A | 16.26 | 15.16 | 10.50 | 0.65 | 0.20 | 2.4 | 85 | 12.4 | 6.1% | 145 | 10% | **floor**（59%）+imb 1.36 |
| 6 | 小pf/A | 25.84 | 24.62 | 32.80 | 1.27 | 1.07 | 10.5 | 102 | 41.6 | 20.5% | 406 | 28% | 混合（mte2 52/fix 29/vec 53）+L2 |
| 19 | 小pf/A | 22.98 | 22.07 | 32.80 | 1.43 | 0.27 | 6.3 | 43 | 11.8 | 5.8% | 274 | 19% | floor（41%）+混合 |
| 20 | 小pf/A | 15.34 | 14.14 | 32.80 | 2.14 | 0.54 | 8.4 | 64 | 35.0 | 17.3% | 547 | 37% | floor（64%） |
| 3† | 中pf/A | 133.48 | 131.83 | 131.10 | 0.98 | 4.30 | 25.2 | 171 | 32.2 | 15.9% | 189 | 13% | V.vec（78.6%） |
| 4† | D256pf/Bmma | 28.10 | 28.05 | 32.80 | 1.17 | 0.54 | 6.3 | 86 | 19.2 | 9.5% | 224 | 15% | 混合+floor |
| 10 | decode/Bgv0 | 53.68 | 34.22 | 53.44 | 1.00 | 0.03 | 33.6 | 1 | 0.6 | 0.3% | 625(b)/981(l) | 42/67% | **H1+H4+floor**（imb 1.50） |
| 11 | decode/Bgv0 | 307.34 | 301.65 | 285.67 | 0.93 | 0.27 | 268.7 | 1 | 0.9 | 0.4% | 874 | 59% | **H1 布局墙**（92% of 966） |
| 12 | decode/Bgv0 | 309.22 | 294.89 | 272.27 | 0.88 | 0.27 | 269.0 | 1 | 0.9 | 0.4% | 870 | 59% | **H1**（94%） |
| 13 | decode/Bgv0 | 569.82 | 562.00 | 494.09 | 0.87 | 0.50 | 504.3 | 1 | 0.9 | 0.4% | 885 | 60% | **H1**（93%） |
| 14 | decode/Bgv0 | 379.18 | 377.41 | 527.55 | 1.39 | 0.27 | 270.5 | 1 | 0.7 | 0.4% | 713 | 49% | H1（74%）+单块批处理协议 |
| 15 | decode/Bgv0 | 507.90 | 503.84 | 345.29 | 0.68 | 0.50 | 504.3 | 1 | 1.0 | 0.5% | 993 | 68% | **H2 duty**（69% of 1452） |
| 16 | MTP/Bgv0 | 561.00 | 559.59 | 439.39 | 0.78 | 1.01 | 504.8 | 2 | 1.8 | 0.9% | 900 | 61% | **H1**（93%） |
| 17 | MTP/Bgv0 | 491.52 | 493.69 | 564.31 | 1.15 | 0.80 | 405.0 | 2 | 1.6 | 0.8% | 824 | 56% | H1（85%）+单块域 |
| 18 | MTP/Bgv0 | 504.34 | 505.07 | 408.79 | 0.81 | 1.01 | 505.3 | 2 | 2.0 | 1.0% | 1002 | 68% | **H2 duty**（69% of 1452） |

† case3/4 = 参考项（任务书"参考 3/4"）。b/l = bench/local 口径；case10 双口径差 +57%（bench 53.68 vs local 34.22）为**在案未归因口径差**（20260928 假设 #5 同款：53.94 vs 34.98），其余 16 case 双口径差 ≤8.5%（小 kernel floor 记账差）。

**L2 侧流量核算（gate0c 事实 #2 口径）**：decode 每 case 的全部 GM 访问流量（20260928 模型 implMB 列，复用有效）= 390-739MB，其中 HBM 唯一字节 = 上表 uniq 列，差额 = w1/w2/w3 workspace 往返 + K/V L1 重载，**物理介质为 L2**（本轮新鲜证据：decode 全族 C 侧 L2 写命中 84.2-97.9%、case15/18 读命中 3.0% = 纯 K/V HBM 流 + w2 命中、D=128 族读命中 ~52.5% = 计数粒度效应见 §2 注）。L2 聚合 ≥4TB/s ⇒ 这部分流量对 decode wall 非绑定（decode wall 由 HBM 唯一字节 ÷ 布局效率决定）。

**vs baseline 利用率对比（Q2 核心问题"CANN 怎么做到的"）**：
- case15 baseline 345.29us = 503.3MB/345.29 = **1458 GB/s = HBM 读峰 99%**。本轮 op 模式实测其实现：TaskDur 347.46us，**cube 全空闲（GM→L1 合计 0.3MB），AIV vector 路径承载全部 K/V（GM→UB 合计 492MB ≈ unique 503MB 单遍无重载），V 侧 MTE2 duty 89.6%**，即 S=1 触发 IncreFlashAttention **vector-GEMV 专用路径**（decode 专用重排 + 无 workspace 往返 + 高 duty 流式装载）。
- case11 baseline 300.52us（本轮实测；eval 记账 285.67）= 268.7MB/300.52 = **894 GB/s——与我方 890（local）同级**。其实现 = cube 路径（GM→L1 271.7MB 单遍），MTE2 duty 94.6%（vs 我方 70.6%）但 busy 期速率仅 49.1 GB/s/核（vs 我方 64.2，我方含 w2 L2 命中抬高）——**双方 wall 平均速率同为 ~46-47 GB/s/核 ≈ 微基准 A 形态聚合上限 48.3**。即 **D=128 decode 上 baseline 没有带宽魔法，它同样撞 256B 突发墙，只是 duty 更高、fixed 开销更小**。
- **EV 存在性结论**：D=256 decode 的 1458 GB/s 证明"512B 行布局 + 高 duty 流式"可达近峰值（微基准 B 形态 1452 独立印证）；D=128 decode 的 ~930 GB/s 就是该布局的物理天花板（微基准 A 形态 966），**任何实现（含 CANN 自家）都只能逼近 966，不能超越**。

---

## 2. decode 管线利用率实测（Q1 证据链主干）

新鲜 op 模式（单发，20 block 跨核均值；完整 JSON = `op_parsed.json`）：

| case | C wall | C.mte2% | mte2 busy速率 GB/s/核 | C.mte1_stall us | C.cube% | C.fix% | C.wait_id2 us | V.vec% | V.wait_id0/4 us | L2 rd/wr命中% | wall imb |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 42.3 | 57.7 | 58.4 | 37.0 | 6.4 | 6.7 | 0.0 | 13.6 | 17.9/18.7 | 52.9/30.2 | **1.50** |
| 11 | 301.1 | 70.6 | 64.2 | 195.0 | 8.5 | 8.7 | 91.2 | 18.7 | 134.9/129.8 | 52.2/96.1 | 1.08 |
| 12 | 296.9 | 70.4 | 66.3 | 279.2 | 8.7 | 9.1 | 0.6 | 22.1 | 114.1/141.3 | 52.2/84.2 | 1.03 |
| 13 | 552.4 | 71.1 | 66.6 | 521.1 | 8.8 | 9.2 | 0.0 | 21.1 | 215.0/263.1 | 52.9/91.5 | 1.03 |
| 14 | 375.6 | 65.3 | 60.9 | 281.6 | 7.0 | **21.7** | 70.8 | 29.7 | 206.8/125.6 | 52.7/96.1 | 1.02 |
| 15 | 498.6 | 68.5 | **74.8** | 356.7 | 9.8 | 13.1 | **122.7** | 16.3 | 241.2/231.4 | **3.0**/97.9 | 1.01 |
| 16 | 551.4 | 70.8 | 66.7 | 525.2 | 8.8 | 9.2 | 0.0 | 35.8 | 50.4/242.1 | 52.9/91.4 | 1.04 |
| 17 | 485.1 | 68.2 | 64.6 | 466.9 | 7.2 | 14.2 | 4.4 | 42.8 | 30.8/174.2 | 52.8/94.8 | 1.03 |
| 18 | 498.3 | 68.2 | 74.9 | 356.0 | 9.8 | 13.0 | 121.0 | 16.3 | 240.7/230.4 | 3.0/97.9 | 1.02 |

读数判读：

1. **算力利用率量化确认极低**：cube 6.4-9.8% busy；实测 TFLOPS 0.6-2.0 = 持续峰值的 0.3-1.0%（§1 表）。decode 优化与算力完全无关。
2. **MTE2 是 C 侧唯一大头**（57.7-71.1% busy），但 **duty < 100%**：mte1_stall（gemm_v0 每-step `PipeBarrier<PIPE_M>` 排空链的 scalar 签名）达 wall 的 65-96%——与归档 gemm_v0 黑盒结构一致。MTE2 idle 的 29-42% = gemm_v0 排空 + SEM 交接暴露（H2 的直接观测）。
3. **busy 期速率分层**：D=256（case15/18）74.8-74.9 GB/s/核 vs D=128 58-67——与微基准"512B 行恢复效率"一致。注意 D=128 的 busy 速率（64-67）**高于**其 HBM 布局上限（48.3/核聚合）因为分子含 w2（p_l1）L2 命中读；K/V 纯 HBM 部分被 966 GB/s 聚合上限封顶。
4. **L2 命中结构（gate0c 事实 #2 在 decode 域再证）**：C 侧写命中 84-98%（w1/w3 FIX 写 = L2 驻留，从不去 DRAM）；case15/18 读命中 3.0% = w2 占比（8KB/264KB）精确吻合 ⇒ K/V 读 100% HBM、workspace 读 100% L2。D=128 族的 ~52.5% 读命中为**计数器粒度效应（推断，基于计数对账 ±0.05% 吻合）**：256B 行读在 128B 计数单元下呈 1 miss + 1 hit 对（case11 block0 miss 合计 53,277 ≈ 52 槽×1024 行 = 53,248 ✓，hit 合计 59,930 ≈ K/V 行第二单元 53,248 + w2 6,656 = 59,904 ✓）——**不代表 K/V 有真实 L2 复用**，DRAM 流量 = 唯一字节（有效带宽与 uniq/wall 一致交叉验证）。
5. **V 侧非约束**：vec 13.6-42.8% busy，wait_id0+id4（等 C 交付 w1/w3）占 V wall 主体。C.wait_id2（等 V 交付 w2）在 case11/14/15/18 为 71-123us（12-30% C wall）= 批边界 CV 耦合 ping-pong 的暴露段，但被 MTE2 主墙 overshadow（wait_id2 与 mte2 busy 部分重叠）。
6. **H4（任务映射）**：仅 case10 显著——32 task vs 20 核（12 核 2 task、8 核 1 task），C wall max/min = **1.50**；其余 decode case 1.01-1.08（近满射）。case10 的 local wall 34.22us 分解 ≈ 2 task × 4 槽 × ~3.7us + floor ≈ 与 HBM 均摊速率兼容——**case10 若做 task→core 均匀化（32 task 摊 20 核为 1.6），理论 wall ≈ 27-29us（-15~-20% local 口径）**，但 bench 口径（53.68）被 +57% 口径差主导，收益在 bench 记账内会被稀释（见 §6 判决）。

**每槽 wall 分解（Q1 判决性算术）**：
- case11（D=128，52 槽/核）：wall/槽 = 5.80us；K/V 256KB @ 微基准布局上限 48.3 GB/s/核 = **5.43us（94%）**；剩余 0.37us（6%）= gemm_v0 排空暴露 + SEM。**→ H1 即墙，duty 优化残值 ≤6%。**
- case15（D=256，96 槽/核）：wall/槽 = 5.25us；K/V 256KB @ 布局上限 72.6 GB/s/核 = **3.61us（69%）**；剩余 1.64us（31%）= duty 缺口。**→ H2 即墙，理论残值 31%（对应 999→1452 GB/s）。**

---

## 3. 微基准判决实验（H1-H4 权重，算子外隔离）

设计：独立 tilelang Expert kernel（与生产同 4 pass_configs=False），纯 `T.copy` GM→L1 ×200 iter，每 iter 128KB（全形态可比）。形态矩阵 = {A: 256B行/8KB跨步（生产 decode D=128 逐字同构，生成码 `copy_gm_to_l1<half,512,128>(stride 4096)` 核验）；B: 512B行/8KB跨步（decode D=256）；C: 全连续（对照=重排后理想布局）；D: 8KB 整行 3D tile} × {stream（HBM 唯一字节，>>L2）；l2（同 tile 重读，L2 驻留）} × {1 核；20 核}。

| 形态 | 1核 stream | 1核 L2 | 20核 stream（聚合） | 20核 L2（聚合） |
|---|---|---|---|---|
| **A** 256B行/8KB跨步 | **56.3** | 142.6 | **966 GB/s（48.3/核）** | 2379 |
| **B** 512B行/8KB跨步 | 144.8 | 255.7 | **1452 GB/s（72.6/核）** | 3492 |
| **C** 连续（对照） | 249.6 | 253.9 | **1469 GB/s（73.5/核）= HBM 读峰 99.8%** | 4446 |
| D 8KB整行 3D | — | — | **异常作废**（17.7us，退化 codegen：3D copy 生成 `copy_gm_to_l1<half,32,128>` 单行形态，未按预期搬运 128KB/iter；如实声明，不影响 A/B/C 判决） | 同 |

（单位 GB/s/核；聚合 = ×20。n=20 median，stdev ≤2.2%。）

**判决**：
- **H1 成立且为 D=128 主墙**：256B 行/8KB 跨步 vs 连续 = 单核 56.3 vs 249.6（**4.4× 惩罚**）、20 核聚合 966 vs 1469（**66%**）。惩罚在 L2 驻留态仍存在（142.6 vs 253.9 = 56%）⇒ 机制 = MTE2/L2 小突发事务开销 + DRAM 页局部性双重损失，**不是 HBM fabric 真墙（H3 排除：连续对照达 99.8% 峰值）**。512B 行即恢复 99%（B/stream 1452）⇒ 惩罚是 256B 粒度的阶跃，非线性渐变——与生产 D=256 族（case15/18 实测 991-1002）vs D=128 族（870-902）的分层精确对应。
- **H2 成立且为 D=256 主墙**：生产 MTE2 duty 57.7-71.1%（§2），busy 期速率已贴布局上限 ⇒ wall 缺口 = duty 缺口；每槽分解 case15 31% / case11 6%。H2 的物理来源 = gemm_v0 每-step `PipeBarrier<PIPE_M>` 排空 + 单缓冲 k_l1/v_l1（下一槽 MTE2 必须等当前 gemm_v0 末次 MTE1 读）+ 批边界 SEM。
- **H3 排除**：见上（C 对照 1469≈1472）。
- **H4 仅 case10**：imb 1.50（§2-6）；其余 1.01-1.08。
- **交叉验证**：微基准 A 聚合 966 = 20260928 归档"折算单核 48-51 GB/s"的独立复现（48.3/核）——**归档 wall 数字正确，本轮补上了它缺失的机制归因（布局上限而非引擎上限）**；归档参照系"C.MTE2 持续 125.2 GB/s/核"是 case9 的 **L2 命中供数** busy 速率（9× 重载域），对 decode 的 HBM 唯一字节流不适用——本轮单核连续 stream 249.6 GB/s 证明引擎本身远快于 125.2，"缺口"从来不在引擎。

---

## 4. 小 prefill floor / 稳态分解（Q3）

wall = launch/dispatch floor（≈9us）+ 稳态工作 + 协议/交接。新鲜 op 数据（§2 表同源）+ floor 占比：

| case | local wall | floor 占比 | pacing 结构（实测） | B 层上界(20260928 模型) | B/now(local) | 分解判读 |
|---|---|---|---|---|---|---|
| 1 | 9.81 | **92%** | C wall 8.6us，全 pipe <25% | 9.0（floor） | 92% | 纯 floor；B 层即 floor ⇒ 无稳态杠杆 |
| 2 | 45.05 | 20% | **V.vec 69.4% = pacing**（case7 同构）；C.mte2 42.3/fix 23.3 | 35.1（V.vec） | 78% | 稳态 = V softmax 链；差额 10us = 批边界+爬坡 |
| 5 | 15.16 | **59%** | 全 pipe <35%；imb 1.36 | 9.8（V.vec） | 65% | floor+imb 主导 |
| 6 | 24.62 | 37% | 混合：mte2 51.9/vec 52.8/fix 28.5/cube 16.6，无独占 pacing | 12.6（L2 流量界） | 51% | 稳态多管平衡；L2 流量界 = w1/w2/w3 往返（L2 介质） |
| 19 | 22.07 | 41% | 混合：vec 48.6/mte2 34.5/fix 16.2 | 9.0（floor） | 41% | floor+稳态各半 |
| 20 | 14.14 | 64% | mte2 50.3/vec 41.0/fix 23.1 | 9.0（floor） | 64% | floor 主导 |
| 3† | 131.83 | 7% | **V.vec 78.6%**；C.wait_id2 75.6 | 129.2（V.vec） | 98% | 已贴 B 层（V-paced 稳态墙） |
| 4† | 28.05 | 32% | 混合 mte2 43.9/vec 47.1/fix 25.4 | 20.3（C.mte2） | 72% | floor+混合稳态 |

**可压缩性判定**：
- **floor 项（case1/5/20 主墙，case19/4/6 半墙）**：≈9us = 60 块 MIX kernel 的 dispatch/SEM init/爬坡，为现 kernel 形态校准值（20260928 §0）。压缩它 = 减块数/简化协议初始化，属结构级改动且波及全域；bench 口径下 baseline 微小 shape 有 32.8us 固定地板（5 case 同值），我方 floor 优势已兑现为 case19/20 的 1.43/2.14x。**无规则内进一步杠杆。**
- **V.vec 稳态项（case2 主墙、case3 已贴界）**：攻击手段 = ③ V 批相干宽通道——已被会话 9 **判决性证伪**（V.vec +21.8% 反升，宽 op 执行周期 > 2× 窄 op）且经 bN 上界定理升级为**规则内永久关闭**（S10-3-5：重试前置 bN≥256 规则内不可达）。case2 的 B/now 78% 差额（~10us）由批边界/爬坡构成，① flat 已在 r8-3 兑现 -33.08%（case2 67.11→44.91 即本轮 45.05 的来源）——**剩余差额无在案杠杆**。
- **§S6-5 "w1 瘦身重试"关闭确认**：§S7-3-3 关闭依据 = case6 真机 fixp 28.8% 非 pacing（camodel 92% 为模型放大）。本轮新鲜 op：case6 fix 28.5%（与 r7 一致，r7_op_case6 复用性成立）、case2 fix 23.3%、case19 16.2%、case20 23.1%——**小 prefill 全域无一 fixp 独占 pacing（最高 case6 28.5% 且与 mte2/vec 混合）**⇒ 关闭判决对小 prefill 域同样成立，**维持关闭**。
- **L2 流量界（case6 B 层 12.6us）**：w1/w2/w3 往返为 L2 介质（本轮 L2 写命中 25-58% 小 prefill 域 + gate0c decode/prefill 域 98.26% 双证）。gate0c iter26 已实测工作消除侧转化率 0.12-1.54%、绝对上界臂为负 ⇒ **消除 L2 往返无 wall 收益，维持关闭**。

---

## 5. 归档结论复核判决表（Q4）

| 归档结论 | 判决 | 新证据（本轮） |
|---|---|---|
| **§4.5-e decode mma 化预算不可行**（D1/D2/D3 行：分池 128KB>64KB；D3 同形可行但无收益） | **维持** | ① bN 上界定理（S10-3-1/2，B 工作集 ≤ L0B 64KB）为其强化版，本轮无新数据触碰其前提；② 本轮新增两项独立证据支持"维持"而非"绕过"：(a) D=128 域即使 mma 化成功，收益上限 = 每槽 6% duty 残值（§2 每槽分解）——预算否决之外再加"无利可图"；(b) D=256 域 31% duty 残值真实存在（case15 上界 1.45×），但 baseline 达成它用的是 **vector 路径而非 cube mma**（§1），即"mma 化"甚至不是唯一技术路线——然而 vector 路径重构 = 分派域重写 + decode 不回退硬约束（IR 级守卫）正面冲突，爆炸半径 ≥ rev4-S1 量级 |
| **§S4-3-3 "decode 多块 wall = 每槽 MTE2 带宽，协议结构已非约束"** | **修正细化** | 对 **D=128：强化成立**——每槽 5.80us 的 94% = K/V @ 966 GB/s 布局上限（微基准 A + baseline case11 同墙实证），协议/duty 残值仅 6%；"MTE2 带宽"需精确为"**布局效率折减后的 HBM 带宽**"（48.3/核 vs 引擎能力 250/核连续）。对 **D=256：拆分**——带宽上限（72.6/核）非墙（生产只用到 69%），**墙 = duty（gemm_v0 排空 + 单缓冲 + 批边界 SEM = 协议/引擎结构）**，故"协议结构已非约束"对 D=256 不成立；case15/18 C.wait_id2 121-123us（24% wall）亦为协议暴露段实证 |
| **20260928 建议⑤ "decode 域 mma 化 + 宽流"（S10-5 归档行：被 §4.5-e 覆盖）** | **修正** | 收益前提分域判决：D=256 域前提**成立**（duty 缺口 31% = 1.45× 上界，baseline 1448 GB/s 存在性证明）但机制（mma 化）维持预算否决，替代机制（vector 路径）撞不回退硬约束；D=128 域前提**不成立**（duty 残值仅 6%，布局墙 966 内无 EV——20260928 "895→~1400 GB/s" 的目标值对 D=128 物理不可达，微基准 A 直接证伪）。建议⑤的"对齐 HBM 读峰"表述应修正为"对齐**布局折减后**的读峰" |
| **decode 不回退硬约束 bench 口径值**（case10 0.996/13 0.867/14 1.391/16 0.783/20 2.138） | **维持（数值更新确认）** | 本轮权威 eval（20261003）逐 case 核对：case10 0.9955/13 0.8671/14 1.3913/16 0.7832/20 2.1382——与在案线一致；历史门槛（10≥0.90/13≥0.69/14≥0.99/16≥0.52/20≥0.82）全部带余量达标 |
| **gate0c 事实 #2（CV workspace GM 往返 = L2 往返）** | **维持 + 域扩展** | 本轮 decode 全 9 case 新鲜 L2Cache：C 写命中 84.2-97.9%（w1/w3）、case15/18 读命中 3.0% 精确 = w2 占比 ⇒ workspace 读亦 L2；D=128 族 ~52.5% 读命中已归因为 128B 计数粒度效应（miss 计数与逐行模型对账 ±0.05%，§2-4），K/V 无真实 L2 复用 ⇒ decode 的 HBM 唯一字节核算（§1 uniq 列）成立 |
| **20260928 硬件常数表 §0** | **维持** | 微基准连续对照 1469 GB/s = HBM 读峰 1472 的 99.8%（独立方法交叉验证）；L2 聚合 20 核 4446 GB/s ≥ 4.0TB/s 下界（一致）；floor 9us 与 case1 local 9.81/case20 14.14 分解兼容 |
| **#R4-3 decode 扁平批处理零收益判决 + 归因知识①②** | **维持** | 本轮 duty 数据补齐其机制账：协议轮 SEM 税坍缩后（~0.2-0.3us/轮），decode 批边界暴露并入 29-42% MTE2 idle 中的小头（D=128 全域残值 6%）⇒ "轮次削减不再可测"获每槽级定量支持 |
| **#R4-1 barrier 纯化 / #R3-1 case14 批处理收益** | **维持** | case14 现值 1.391x 为 decode 族最高（单块批处理 + barrier 纯化红利在案）；本轮 op 显示其 fix 21.7% 为 decode 族最高（bN=128 小槽 ⇒ 每 task w1+w3 16KB 占 MTE2 字节 25%，FIX 64.1 GB/s 慢于 MTE2 ⇒ 时间占比放大），mte2 duty 65.3% 仍有 H2 残值但受同一单缓冲结构约束 |

---

## 6. 逐族最终判决（剩余 EV 量化 + 杠杆候选/墙确认）

### 6.1 decode D=128 族（case10/11/12/13/16/17；含 MTP 16/17）——**墙确认（H1 布局墙），无规则内空间**

- **剩余 EV 量化**：实测 870-902 GB/s（local 口径 891-981）vs 布局上限 966 ⇒ **残余 ≤8%（case11-13/16），case17 15%（含单块域协议零头），case10 特殊（下）**。每槽分解：94% 已是 K/V HBM 物理时间。即使 duty 修到 100%，wall 收益上限 = 6%（case11 301.65→~284us，speedup 0.93→0.99）。
- **墙确认证据（判决性，三链独立）**：① 微基准 A 形态：单核 56.3 / 聚合 966 GB/s，L2 驻留态惩罚仍在（142.6 vs 253.9）⇒ 引擎/事务级墙；② baseline case11 实测（本轮 op 首采）：cube 路径 duty 94.6% 近饱和仍只有 894 GB/s ⇒ 业界最优实现同墙；③ 生产每槽 5.80us 中 5.43us = 966 GB/s 上限的算术必然。
- **杠杆候选核对**：突破 966 需 K/V 行 ≥512B 连续或整行连续（微基准 B/C：1452/1469）。可行路线仅两条，均否决：(a) **host 侧重排**（BSHD→BSH-merged/NZ）——违反 §4.8 host 轻量化红线（禁止 `.contiguous()`/重排类全量搬运）且 bench 口径把重排时间计入 wall（重排 = 读 503MB+写 503MB ≥ 685us >> 全部收益）；(b) **kernel 内跨 head 合并读**（一次 MTE2 读 [bN, N_all, D] 整行 8KB 连续再分 head 消费）——L1 预算否决：bN=512×32head×128×2B = 4MB >> 511.75KB；缩到 bN=16 则每槽 256KB→8KB 槽深暴涨 64×，SEM/协议轮爆炸，且 GEMM1 需按 head 切片 L1（MTE1 列段分解 ×32），事件税先例（884→1326）表明方向性劣化。**结论：H1 墙 = BSHD 契约 + L1 预算 + host 红线的联合硬限，维持"无空间"判决**；eval 中 0.87-0.93x 的差距主体是 baseline 记账更快（285.67 vs 本轮实测其 300.52）+ 我方固定开销，而非可回收的带宽效率。
- **case10 特例（H4）**：imb 1.50（32 task/20 核）。task→core 均匀化（如 core_num=16 或 32 task 重排）理论 local wall -15~-20%（34.22→~28us），但 (i) bench 口径差 +57% 未归因，bench 收益无法可靠预估；(ii) 改动触碰 decode 兜底域分派（不回退硬约束 IR 守卫管辖）；(iii) EV 量级 = bench 0.996→至多 ~1.05-1.1（若口径差为常数加性）。**条件空间：仅当 Orchestrator 判定 case10 bench 口径差归因完成后才值得立项**，本轮如实标注为"低置信度条件候选"。

### 6.2 decode D=256 族（case15/18）——**条件空间：H2 duty 缺口 31%，上界 1.45×，两条路线均撞在案硬限**

- **剩余 EV 量化**：999-1002 GB/s → 布局上限 1452（=baseline 实测 1448）⇒ **wall 上界收益 = 503.84→~347us = 1.45×（case15 speedup 0.68→0.99；case18 0.81→1.17）**。这是 decode 全族唯一 >10% 的在案 EV。
- **物理机制**：duty 68.5%→~90% 需消除 gemm_v0 每-step 排空 + K/V 单缓冲串行（MTE2 与 mma 重叠）。
- **与硬限相容性核对（两条路线）**：
  - 路线 1 **mma 化**（原建议⑤）：§4.5-e D2' 行分池 128KB>64KB 否决 + bN 上界定理（B 工作集 ≤ L0B 64KB，K^T/V 各 128KB 相位内重用不可避免 ⇒ 双重毒化挂死，S10-3-2 实测 15 变体穷尽）。**维持否决——本轮无新数据触碰其前提。**
  - 路线 2 **k/v L1 双缓冲**（提 duty 的最小改动）：D=256 decode L1 现占 272KB（§4.5-a D2 行），双缓冲 k/v ⇒ 528KB > 511.75KB **硬限否决**（五环证伪链第 3 环"decode 带宽+预算"的预算半支本轮独立复核成立）。bN=256→128 缩块可腾 L1（k/v 各 64KB，双缓冲后 k128+v128+q8+p4 ≈ 268KB ≤ 511.75KB ✓）但槽数 ×2 ⇒ SEM 轮 ×2 + 协议税回涨先例（#R3-1 复合税模型）+ case14（bN=128 单块域 duty 65.3% 并未更高）表明净 EV 存疑，**未实测，如实标注为未验证候选**。
  - 路线 3 **vector 路径重构**（baseline 的达成方式）：S=1∧D=256 分派到 V 侧 GEMV 形态（UB 内 softmax+PV，K/V GM→UB 流式）。与"decode 兜底域源码逐字不变 = 构造性不回退"（§4.5-e 设计决策）**正面冲突**；工程量为新 kernel 形态（非调优），V.MTE2 56.8 GB/s/核 × 40 AIV = 2272 GB/s 聚合引擎能力充足，但 UB 预算/D=256 行宽需全新推导。**超出 Stage-3 规则内范畴，仅作为存在性证明记录。**
- **判决**：**条件空间成立但规则内不可达**——三条路线分别撞 L0B 预算定理、L1 预算、不回退硬约束。若 Orchestrator/用户未来立项"decode D=256 专项"，路线 2 的 bN=128 缩块变体是唯一规则内可表达候选（需探针先行，EV 未验证）。

### 6.3 MTP（case16/17/18）——**同 decode 分域判决**

case16（D=128 多块）= 6.1 墙（93% of 966，S=2 causal eff≈满，填充效率 0.12 在带宽墙下无关紧要——归档维持）；case17（D=128 单块批处理域，Skv=256）= 85% of 966，残值含单块协议零头，已享 #R3-1 红利（1.15x）；case18（D=256）= 6.2 条件空间（1.45× 上界同 case15，eval 0.81→~1.17）。

### 6.4 小 prefill（case1/2/5/6/19/20，参考 3/4）——**floor/V-softmax/L2 介质三墙，无规则内空间；case2 残值 ~22% 但杠杆族已永久关闭**

- **剩余 EV 量化（B/now local 口径）**：case1 92%（B=floor=墙）；case2 78%（B 层 35.1 = V.vec 稳态界，残值 ~10us = 批边界/爬坡）；case5 65%（floor+imb）；case6 51%（B 层 12.6 = L2 流量界——gate0c 工作消除转化率 0.12-1.54% 实测归零 ⇒ B 层此项**不可兑现**，实际残值以 floor+混合稳态为界）；case19 41%（B=floor）；case20 64%（B=floor）；case3 98%（已贴界）；case4 72%。
- **杠杆候选核对**：③ V 宽通道（攻击 case2 的 V.vec 69.4%）= **规则内永久关闭**（S9 判决性证伪 V.vec +21.8% + S10-3-5 前置不可达）；w1 瘦身 = 双 régime 关闭且本轮确认小 prefill 无 fixp 独占 pacing（§4）；L2 往返消除 = gate0c 实测归零；floor 压缩 = 结构级（60 块 MIX dispatch），bench 口径下我方 floor（9us）对 baseline 地板（32.8us）已是净优势来源（case19/20 的 1.43/2.14x 即其兑现）。
- **判决**：**无空间（墙确认）**。case2 的 0.72x 差距 = V-softmax 稳态（B 层界内）+ baseline 32.8us 地板记账优势；case5 的 0.65x = floor+imb 对 baseline 10.5us 的微小 shape 地板。B 层名义上界（case6 2.61x/case19 3.64x）与 now 的差额主体为 L2 流量界与 floor——前者已被 gate0c 实测判定不可兑现，后者为 kernel 形态常数。**小 prefill 域规则内杠杆穷尽，维持现状即最优收口。**

---

## 7. 假设与缺口声明（如实）

1. **微基准 = 算子外隔离实验**：纯 `T.copy` 循环无 gemm/FIX/SEM 并发，测的是该访存形态的 MTE2/HBM 速率天花板；生产内核中 MTE2 与 FIX 写（L2 介质）、V 侧 GM→UB 并发可能引入额外 L2 端口争用——生产 busy 期速率（58-75 GB/s/核）与微基准（48-73 聚合均摊）的一致性表明该争用 ≤ 测量分辨力，但**未单独隔离定量**。
2. **Pattern D（8KB 整行 3D tile）作废**：3D `T.copy` 生成退化 codegen（单行形态），17.7us 读数无物理意义；未修复重测（A/B/C 已覆盖判决所需全部对照，D 仅作锦上添花）。
3. **case10 bench/local 口径差 +57% 仍未归因**（20260928 假设 #5 延续）：两口径均如实并呈；不影响 D=128 墙判决（其余 5 case 双口径一致 ≤5%）。
4. **baseline case11 本轮实测 300.52us vs eval 记账 285.67us（+5.2%）**：可能为 eval 采集会话条件差异或 metrics 重放开销；两者均 <我方 307.34，"baseline 同墙"结论对两个值都成立（894-939 GB/s 均 ≈ 966 上限的 93-97%）。baseline 仅采 case15/11 两形状（任务书指定），case13/16/17 未采——按 D=128 同布局外推，**标注为推断非实测**。
5. **每槽分解的槽数核算**采用编译期 tiling 常数（bM/bN/kv_iters/task→core 映射，mha.py:195-237 host 推导逐字复核），未做运行期逐槽 trace；case11 的 52 槽/核 与 GM_to_L1 14.08MB/272KB = 51.8 槽对账 ✓。
6. **B 层上界沿用 20260928 模型**（结构未变——flat 批处理 r8 已在其"now"之后落地，模型 B 层的"完美流水"假设对单块/decode 域仍有效）；case2/3 的 B/now 用本轮新鲜 local wall 重算。模型 V.vec 线性缩放 ±30% 不确定度（20260928 假设 #3）对 case2 残值区间同样适用。
7. **w2 读命中对 MTE2 busy 速率的抬高未逐字节剥离**（D=128 族 busy 64-67 vs 纯 K/V HBM 上限 48.3/核）：按字节占比（w2 16KB/272KB = 5.9%）与 L2 速率估算，剥离后 busy 期 K/V 速率 ≈ 48-52 GB/s/核 = 布局上限带内——推断与 §2-4 miss 计数对账一致，但未做独立隔离测量。
8. **camodel 全程未运行**（纪律 #1）；本轮全部定量 = 真机 msprof + 微基准 + baseline 对照。
9. 本轮零 kernel 运行于默认缓存（全部专属 `TILELANG_CACHE_DIR=/tmp/opencode/decode_roofline_a1/cache_{wall,micro}`，rev4 缓存纪律）；device 0（采集期间 npu-smi 无他进程）。

## 8. 产物清单

- 本报告: `examples/mha/perf_tuning/analysis_20261003_decode_smallprefill_roofline.md`
- 采集数据（scratch，可复算）: `/tmp/opencode/decode_roofline_a1/`
  - `wall/case*.json`（17 case 本地 wall median，n=25）+ `log_case*.txt`
  - `op/case*/OPPROF_*/`（17 case 真机 op 模式 8 类 CSV）+ `op_parsed.json`（解析产物）+ `parse_op.py`
  - `base_case{15,11}/`（baseline FusedInferAttentionScore op 模式）+ `bench_baseline.py`
  - `micro_*/`（微基准 14 有效配置 + 2 作废 D）+ `micro_mte2.py` + `run_micro.sh` + `micro_A_src.cc`（生成码核验）
  - `roofline_recompute.py` + `roofline_rows.json`（§1 总表复算脚本）
  - `run_wall.sh` / `run_metrics.sh`（采集 driver，gate0 iter24 命令形态逐字沿用）
- 引用在案资产: `analysis_20260928_*`（常数表+模型）、`final_report.md` §S4/S6/S7/S9/S10、`optimization_log.md` [#R3-1/#R4-1/#R4-3]、`DESIGN.md` §4.5-a/c/d/e、`gate0c_verdict_iter26.md` 事实 #2/B.4、`/tmp/opencode/r7_op_case6/`（交叉印证）
