# MHA 算子理论性能上限与理想流水排布分析

- 日期: 2026-09-28 | 类型: 纯分析（mode=analysis，零代码变更）
- 对象: `examples/mha/mha.py` rev4 终态（1593 行，md5 `ee5bf9c93f979b9fc5b111a3c4121888`）
- 权威评测: `.agents/skills/cann-bench/reports/cann_performance_eval_20260922_172117.{md,json}`
  （20/20 pass，综合 71.79，平均加速比 **0.9108x**，baseline = CANN `FusedInferAttentionScore`，真机 Ascend910_9362）
- 流水数据: `perf_tuning/msprof_case9_20260923/`（终态新鲜采集：真机 op 模式 CSV + camodel core0 三核 trace/instr_exe + MindStudio 导出）
- 建模脚本: `/tmp/opencode/mha_roofline_v5.py`（本报告全部数字可复算）

---

## 0. 硬件规格取证（全部实测/仓内证据，无记忆猜测）

| 参数 | 取值 | 证据来源 |
|---|---|---|
| 芯片 | Ascend910_9362，20 AIC + 40 AIV（cv_1_2），1650 MHz | `torch_npu.npu.get_device_properties(0)`；`msprof_case9_20260923/op_device/OpBasicInfo.csv`（Current/Rated Freq=1650，Block Dim 20/Mix 40）|
| L2 / HBM 容量 | 168 MB / 62.7 GB | 同上 props（L2_cache_size=168MB）|
| 片上内存 | UB 192KB，L1 512KB，L0A/L0B 64KB，L0C 128KB | TBE `get_soc_spec`（Ascend910B4，SHORT_SOC_VERSION=Ascend910B）|
| Cube 峰值 | 16×16×16 fp16 MAC/cy/核 = 4096 MAC/cy → **270.3 TFLOPS**（20 核）| TBE `cube_m/k/n_size=16`；1650MHz |
| Cube 持续 | **3069 MAC/cy（202.5 TFLOPS）**= 峰值 75% | case9 真机反推：19.33G 填充 MAC ÷ (20 核 × cube_ratio 0.202 × 944.74us)；camodel 指令级 MMAD 149cy/[32,128]×[128,128] = 3518 MAC/cy 发射率交叉印证 |
| HBM 带宽 | **读 1472 / 写 1481 / 混合读写 1250 GB/s** | 2026-09-28 torch_npu 1GB 微基准（max 0.729ms / fill 0.725ms / copy 1.718ms）；**独立旁证**：baseline case15 = 503MB ÷ 345.29us = 1458 GB/s |
| L2 聚合带宽 | **≥ 4.0 TB/s（下界）** | 40MB L2 驻留 copy 微基准 21us（含发射开销污染 → 保守下界）；case9 生产态实测聚合 GM 访问 2230MB/944.74us = 2.36 TB/s |
| 各管道持续速率（busy 期） | C.MTE2 125.2（≈48% of 隐含峰值 264）、C.MTE1 191.3、C.FIX 64.1（≈26%）、V.MTE2 56.8、V.MTE3 30.6 GB/s/核 | case9 真机 op 模式 `Memory.csv`+`PipeUtilization.csv` block0（datas ÷ ratio×wall）|
| V 矢量吞吐 | 1.058 us/(iter·组) @ 12288 fp32 元素（含碎片化税：36×BAR + 34×MOVEMASK/iter）| case9 真机 vec_ratio 0.495×944.74us ÷ 442 iter；camodel `veccore0_instr_exe.csv` 指令构成 |
| 启动+爬坡地板 | ≈ 9 us | 校准：case20 模型工作量 ~6us vs 实测 14.5us；case1 ~2us vs 10.3us |

> 关键旁证：MMAD/FIX/MTE1 指令计数与源码结构**完全自洽**（core0：442 迭代 ×4 MMAD = 1768 ✓、×3 FIX = 1326 ✓、causal eff_kv 均值 4.5 = 144/32 ✓），模型可信度有指令级锚定。

---

## 1. 任务一：理论性能上限（换算 vs baseline 加速比）

### 1.1 方法：三层模型

| 层 | 含义 | 约束集 |
|---|---|---|
| **A（算法屋顶线）** | 理想 flash kernel（无 GM workspace 往返）+ 两种分块取优（大块减 KV 重载 vs 现块高 causal 填充效率 0.89）| max(cube 持续 202.5T*, HBM 唯一字节 1472/1250, L2 重载 ≥4TB/s, MTE1 191.3×20, VEC 4.2Tel/s*, floor 9us)；A* = cube 峰值 270.3T 变体 |
| **B（架构上限）** | **当前实现结构**（CV 分离 + w1/w2/w3 GM 往返 + 现分块 + 现指令结构），假想**完美流水**（全管道零暴露重叠）| 每迭代 max(7 管道实测持续速率) × 每核迭代数 + 任务边界爬坡；芯片级 max(HBM 唯一字节, L2 全部 GM 访问流量) |
| now | 实测（cann-bench 权威）| — |

\* 标注 = 假设项（见 §4）；两者在全部 20 case 均非绑定约束，不影响结论。

工作量模型与实现逐项对齐并经验证：case9 GM 流量模型 2332MB vs 实测 Memory.csv 加总 2230MB（-4.4%，valid_n 尾部裁剪）；GM→L1 模型 1285MB vs 实测 1256MB ✓；L0C→GM 442 vs 432 ✓；GM→UB 442 vs 432 ✓；UB→GM 106.5 vs 104 ✓。

### 1.2 全 20 case 理论加速比表

| case | 形态 | 分块 | 理论 tA us（约束）| **spA** | spA* | 架构上限 tB us（约束）| **spB** | 实测 us | **now** | B/now |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 64×128 | 9.0（floor）| 1.11 | 1.11 | 9.0（floor）| 1.11 | 10.72 | 0.93 | 84% |
| 2 | A | 64×128 | 9.0（floor）| 3.64 | 3.64 | 35.1（V.vec）| 0.93 | 67.92 | 0.48 | 52% |
| 3 | A | 64×128 | 26.8（hbm）| **4.88** | 4.88 | 129.2（V.vec）| 1.01 | 170.10 | 0.77 | 76% |
| 4 | Bmma | 32×128 | 9.0（floor）| 3.64 | 3.64 | 20.3（C.mte2）| 1.62 | 36.48 | 0.90 | 56% |
| 5 | A | 64×128 | 9.0（floor）| 1.17 | 1.17 | 9.8（V.vec）| 1.07 | 18.88 | 0.56 | 52% |
| 6 | A | 64×128 | 9.0（floor）| 3.64 | 3.64 | 12.6（L2）| 2.61 | 25.90 | 1.27 | 49% |
| 7 | A | 64×128 | 190.8（cube）| **2.27** | 2.74 | 797.2（V.vec）| **0.54** | 936.18 | 0.46 | 85% |
| 8 | A | 64×128 | 190.8（cube）| **2.25** | 2.72 | 797.2（V.vec）| **0.54** | 922.50 | 0.47 | 86% |
| 9 | Bmma | 32×128 | 190.8（cube）| **1.77** | 2.14 | 626.1（pipe/L2）| **0.54** | 957.74 | 0.35 | 65% |
| 10 | Bgv0 | 16×512 | 22.8（hbm）| **2.34** | 2.34 | 22.8（hbm）| 2.34 | 53.94 | 0.99 | 42%† |
| 11 | Bgv0 | 16×512 | 182.5（hbm）| 1.57 | 1.57 | 182.5（hbm）| 1.57 | 309.28 | 0.92 | 59% |
| 12 | Bgv0 | 16×512 | 182.5（hbm）| 1.49 | 1.49 | 182.5（hbm）| 1.49 | 306.30 | 0.89 | 60% |
| 13 | Bgv0 | 16×512 | 342.3（hbm）| 1.44 | 1.44 | 342.3（hbm）| 1.44 | 562.60 | 0.88 | 61% |
| 14 | Bgv0 | 16×128 | 183.1（hbm）| **2.88** | 2.88 | 183.1（hbm）| 2.88 | 377.38 | 1.40 | 49% |
| 15 | Bgv0 | 16×256 | 342.3（hbm）| **1.01** | 1.01 | 342.3（hbm）| 1.01 | 504.02 | 0.69 | 68% |
| 16 | Bgv0 | 16×512 | 342.6（hbm）| 1.28 | 1.28 | 342.6（hbm）| 1.28 | 564.82 | 0.78 | 61% |
| 17 | Bgv0 | 16×256 | 274.6（hbm）| 2.05 | 2.05 | 274.6（hbm）| 2.05 | 494.66 | 1.14 | 56% |
| 18 | Bgv0 | 16×256 | 342.6（hbm）| 1.19 | 1.19 | 342.6（hbm）| 1.19 | 504.48 | 0.81 | 68% |
| 19 | A | 64×128 | 9.0（floor）| 3.64 | 3.64 | 9.0（floor）| 3.64 | 23.28 | 1.41 | 39% |
| 20 | A | 64×128 | 9.0（floor）| 3.64 | 3.64 | 9.0（floor）| 3.64 | 15.44 | 2.12 | 58% |

† case10 本地 msprof 34.98us vs bench 53.94us 存在口径差（B/now 按 bench 口径 42%，按本地口径 65%）。

**汇总（平均加速比 vs baseline）**：

| 层 | mean | geomean | 非地板 14 case（3,4,7-18）mean |
|---|---|---|---|
| **A 理论上限**（理想 kernel，持续速率）| **2.35x** | 2.10x | 2.09x |
| A*（cube 峰值绝对上限）| 2.41x | 2.16x | 2.24x |
| **B 架构上限**（现结构完美流水）| **1.63x** | 1.39x | 1.42x |
| now 实测 | 0.91x | 0.83x | 0.85x |

按族：

| 族 | A | B | now | 判读 |
|---|---|---|---|---|
| 小 prefill（1,2,5,6,19,20）| 2.81x | 2.17x | 1.13x | 双侧 floor 主导；baseline 对微小 shape 有 ~32.8us 固定地板（5 case 同值），理论加速比在 bench 记账内成立 |
| 大 prefill（3,7,8）| 3.13x（A* 3.45）| **0.70x** | 0.57x | **B < 1：现架构即使完美流水也跑不赢 baseline**（workspace 往返 + V 碎片化 + FIX/L2 墙）|
| D256 prefill（4,9）| 2.71x | 1.08x | 0.63x | 同上，程度稍轻（bM=32 填充效率 0.89 高）|
| decode（10-15）| 1.79x | 1.79x | 0.96x | **A=B：纯 HBM 墙，架构无关**；差距全部在流带宽利用率（622-999 GB/s vs 1472）|
| MTP（16,17,18）| 1.51x | 1.51x | 0.91x | 同 decode（S=2 走兜底域，填充效率 0.12 但带宽墙下无关紧要）|

### 1.3 关键洞察

1. **"1.0x 平均"在本 bench 上是可超越的**：全 case 理论上限 mean 2.35x。当前 0.91x = 理论上限的 **39%**、架构上限的 **56%**。
2. **decode/MTP 族天花板被 baseline 自己钉死在 HBM 墙**：baseline case15 = 345.29us ≈ 503MB/1472GB/s = 342.3us 的 **99%**——CANN decode kernel 已打满 HBM 读带宽。case15/18 的理论最大加速比 ≈ **1.01/1.19x**，任何实现都不可能显著超越；case13/16 上限 1.44/1.28x。decode 族的可得空间 = 把我们的有效带宽从 622-999 GB/s 提到 ~1400 GB/s（case10/14 上限最高：2.34/2.88x，因 baseline 在这两个 case 自身效率只有 43%/51%）。
3. **大 prefill 是"架构墙"而非"调度墙"**：case7/8/9 每次启动 GM 访问 2181-2332MB（理论最小 134MB 的 **16-17 倍**），其中 w1/w2/w3 workspace 往返 1032MB（46%）。L2 聚合 ≥4TB/s 下仅内存系统流量就构成 ~545-583us 地板（case9 baseline 才 337.63us）→ **现 CV-over-GM 架构对 case7/8/9 的理论最好成绩 ≈ 0.54-0.79x，永远到不了 1.0x**；要进入 1.8-2.7x（A 层）必须消除 workspace 往返（需框架级 L0C→UB/L1 虚拟通道，或单核域 flash 结构）。
4. **cube 利用率在全族都极低且不构成约束**：case9 cube_ratio 20.2%；即使 A 层理想 kernel，大 prefill 的 cube 时间（190.8us）也只比 HBM/L2 项（107/151us）略高。**该算子本质是"内存系统 + 同步"问题，不是算力问题**。
5. baseline 自身效率（tA/base）：case15 99%、case13 69%、case9 57%、case7 44%、case10 43%——大 prefill 上 baseline 离屋顶线也远，这正是 2-3x 理论空间的来源。

---

## 2. 任务二：最理想流水排布

### 2.1 现状解剖（case9，终态新鲜数据 2026-09-23，真机 op 模式为权威）

kernel 结构：三分派（form_a / form_b_mma / form_b_gemm_v0），C↔V 经 3 个 GM workspace（w1 scores fp32 / w2 P fp16 / w3 PV fp32）+ 6 个跨核 SEM 交换；NUM_STAGES=8 槽批协议，每外层批 = **四相位**：C-GEMM1(w1) → V-softmax(w2) → C-GEMM2(w3) → V-acc_o rescale。case9 每核 442 迭代（causal eff_kv 均值 4.5 → 批深半空转，任务边界 103 次/核）。

真机管道占用（wall 944.74us 占比）：

| 核 | 管道 | busy% | 暴露等待 |
|---|---|---|---|
| C（AIC×20）| scalar | **88.0%** | mte1_stall **466us（49%wall）**、wait_id2（MTE2）310us、wait_id5 86us |
| | MTE2 | 53.1% | 最忙实体管道（K/V/P 装载 139KB/iter）|
| | FIX | 35.7% | w1+w3 fp32 写出 48KB/iter @64GB/s |
| | MTE1 | 35.5% | 34.9 条 LOAD_2D/iter（colseg 分解）|
| | **cube** | **20.2%** | 4×MMAD/iter，仅 0.36us/iter |
| V（AIV×40）| scalar | 63.0% | wait_id4 411us + wait_id0 225us（等 C 交付 w1/w3）|
| | vec | 49.5% | 1.058us/iter/组（碎片化：36×BAR+34×MOVEMASK/iter，46904 个矢量事件，均值 14ns）|
| | MTE2/MTE3 | 20.1%/9.3% | w1/w3 读 48KB、w2/O 写 |
| | FLOWCTRL（camodel）| **74.2%** | V 的主体时间在等 C —— 与真机 wait_id 分解一致 |

判决性归因（会话 7 双侧 A/B 排除法，final_report §S7）：加速 V 相位 → C 侧等待 1:1 增加；加速 C-GEMM1（mte1_stall -54% 靶心命中）→ C wait_id2 +221us 1:1 吸收，wall 均 +0.3%。**wall = 四相位每槽交接延迟链**，单侧/速度/字节类手段在此结构下数学上无杠杆（吸收律）。

### 2.2 理想流水排布（目标形态）

**总原则：把"每槽 4 次跨核 SEM 传播 + 管道排空"的串行延迟链，改造成"连续令牌流 + 深前瞻 + 宽指令"的吞吐型流水。**

理想稳态节拍（case9，每迭代 1.112us 由 C.MTE2 139KB @125.2GB/s 决定）：

```
C 核（每 1.11us 一个迭代节拍，全管道重叠）：
  MTE2 : [K,V(i+2) 128KB][P w2(i+1) 8KB] ──连续流──►
  MTE1 : K/V chunk(i+1)→L0B 4槽旋转（64列粒度，前瞻深度2）──►
  cube : QK^T(i) 2×chunk ∥ PV(i-1) 2×chunk（各 0.36us，占空 32%）
  FIX  : w1(i) 16KB ∥ w3(i-1) 32KB（0.75us，占空 67%）
  SEM  : 信用制令牌（credit-based），槽释放即发，无批边界排空

V 核（与 C 异相位流水，槽粒度交接而非批粒度）：
  MTE2 : w1(i) 16KB ∥ w3(i-1) 32KB 连续流
  vec  : 8 槽合并宽指令 softmax（每批一次 [128×128] 级 2D op 序列，
         消除每迭代 36×BAR/34×MOVEMASK → 指令数 ÷5-8）
  MTE3 : w2(i) 4KB ∥ O（任务尾）
```

五个结构性手段（按因果顺序，**必须联合落地**——吸收律决定单项无效）：

| # | 手段 | 攻击的实测瓶颈 | 量化依据 |
|---|---|---|---|
| ① | **flat (task,kv) 连续令牌流**：把 (任务, kv 块) 对摊平为单一工作队列，槽跨任务边界不排空 | 批深 4.5/8 半空转 + 103 次/核任务边界排空 | 蓝图已归档（#R4-3 机制 17/17 BITEXACT + §S4-3-5 状态复位时序 + 会话 7 pacing 判定）；单独 EV -5~-15% |
| ② | **C 侧 MTE1 深前瞻**：L0B 池由 2×[128,128] 改 4×[128,64] 旋转（同 64KB），装载前瞻深度 0→2 | mte1_stall 466us（49% wall）——现结构 GEMM1/GEMM2 各耗尽双槽，跨迭代零前瞻 | MTE1 busy 仅 335us 但 stall 466us → 纯延迟暴露非吞吐不足；代价：MMAD 4→8/iter 发射税 + w3 colseg FIX 事件 3→5/iter（事件税先例：884→1326 已观测）|
| ③ | **V 侧批相干化**：8 槽 w1 一次宽 softmax 通道 | vec 49.5% busy 中碎片化税（46904 事件、36 BAR/iter）| V 元素量仅需 ~90us（4.2Tel/s 理想）vs 现 468us；B- 层（相干化）使 case7/8 上限 0.54→0.70x |
| ④ | **w3 dtype 瘦身**（fp32→fp16/bf16，需过精度红线复核）| FIX 64GB/s 墙 + L2 流量 2332→1750MB | B régime 下 case7 绑定管道即 C.fix（1.022us/iter）；注意 **w1 不可瘦身**（精度红线：score 双舍入栅格要求 w1 fp32 主线）；w1 瘦身已双 régime 证据性放弃（§10.3-①）|
| ⑤ | **decode 域 mma 化 + 宽流**：gemm_v0 每-step `PipeBarrier<PIPE_M>` 换成 Form B-mma 式令牌链；K/V 装载对齐 HBM 读峰 | decode 有效带宽 622-999 vs 1472 GB/s；case13 wall 562 vs HBM 界 342us | decode 族 C FLOWCTRL≈0（归档）→ 瓶颈在 gemm_v0 步进排空 + bM=16 窄行 MTE2 模式；**受 decode 不回退硬约束管辖（IR 级守卫），改动需独立立项** |

### 2.3 量化差距分解（case9：945us → ？）

| 阶段 | wall | vs baseline 337.63us | 依据 |
|---|---|---|---|
| 现状 | 944.7us | 0.353x | 真机 msprof（权威）|
| 仅①（flat 批处理，归档 EV）| ~800-900us | 0.38-0.42x | 会话 7 scoping（EV -5~-15%）|
| ①+②+③（完美流水，B 层）| **626us**（pipe 492 + 爬坡 134；L2 流量界 583us 近平行）| **0.54x** | 模型：442 iter × max 管道 1.112us（C.MTE2）|
| ①-④（+w3 瘦身，L2 -25%）| ~560-600us | ~0.58x | 模型推算（±） |
| A 层（去 workspace 架构，理想 kernel bM=64）| **191us**（cube 持续界）；峰值 cube 158us（MTE1 界）| **1.77x / 2.14x** | 需框架级虚拟通道或单域 flash 重构 |

结论：**现架构完美流水的收敛点 ≈ 0.54-0.61x（case9）**，与 A 层 1.77-2.14x 之间的 3 倍差距全部是 **CV-over-GM workspace 架构成本**（每迭代 139KB MTE2 中 56KB、FIX 48KB、L2 流量 2332MB 中的 1032MB 都是 w1/w2/w3 往返）。camodel 已实证 L0C→UB / UB→L1 无设备实现（虚拟通道禁令）→ 该差距**规则内不可回收**，与历史归档结论一致。

### 2.4 三分类清单

| 分类 | 项 | 状态/出处 |
|---|---|---|
| **规则内可实施** | ① flat(task,kv) 批处理（蓝图齐备，爆炸半径全 20 case，需独立会话预算）；② L0B 4 槽旋转（新提案，需探针验证 MMAD 发射税与 FIX 事件税净值）；③ V 批相干化（emit_v 重构，共享三形态需全域复测）；⑤ decode mma 化（受不回退硬约束，IR 守卫管辖）| #R4-3+§S4-3-5+会话7 判定；②③为本分析新增（未验证，EV 为模型推算）|
| **需框架/codegen 支持** | L0C→UB / UB→L1 虚拟通道（A 层的唯一入口，消除 workspace 往返）；T.Pipelined address_map 修复（R9）；threads=2 VidReduction BSHD 修复（R8）| 虚拟通道禁令源码级实证（会话 6）；R8/R9 护栏 |
| **已证伪/证据性放弃（勿重试）** | 单侧加速与减字节类手段（吸收律 1:1 定量实证 r7-1/r7-2）；w1 dtype 瘦身（双 régime msprof 零响应，§10.3-①）；k/v L1 双槽（五环证伪链闭合，§10.3-③）；CROSS_INTERVAL 扩大（禁提或 EV≈0）；kv-outer 循环互换（rev4 S2 四步定量论证降级：UB 硬否决 + GM spill 持平略负）| final_report 会话 4-7、DESIGN §10.3 |

### 2.5 与理论上限的自洽性

- decode/MTP 族：理想流水（⑤）达成后直接收敛到 A=B 的 HBM 墙（1.28-2.88x），**排布即上限**，无架构差距。
- 大 prefill 族：理想流水（①-④）收敛到 B 层 0.54-0.79x；A 层 2.3-3.4x 只有架构变更可达 → **两层之间的差距是明确的投资决策点**（框架虚拟通道 or 接受 B 层收口）。
- 小 case 族：floor 主导（我方 ~9us vs baseline 微小 shape 32.8us 地板），流水改造收益有限，1.1-3.6x 主要靠压低自身 floor（60 块 MIX kernel 的 dispatch/SEM init 成本）。

---

## 3. 建议优先级（仅建议，本任务不实施）

1. **decode 域 ⑤（mma 化 + 带宽流优化）**——唯一"A=B"族：完美流水即达理论上限，decode 6 case + MTP 3 case 合计 mean 0.93x → 潜在 1.4-1.8x，对平均加速比杠杆最大（9/20 case）；受不回退约束需独立立项。
2. **①+②+③ 联合（大 prefill 完美流水）**——case7/8/9 从 0.46/0.47/0.35x → ~0.54-0.61x（+15-70%）；①蓝图现成，②③需探针先行；必须联合（吸收律）。
3. **④ w3 瘦身**——在 ①-③ 落地后激活（B régime 下 FIX/L2 才成为绑定管道），精度红线复核前置。
4. **架构决策**（虚拟通道/单域 flash 重构）——投入大，但那是 1.77-3.4x 空间的唯一钥匙；建议以 case9 为样板做框架级可行性验证后再决策。

## 4. 假设与未验证项（如实声明）

1. **VEC_PEAK = 64 fp32 通道/AIV（2048-bit）为假设值**——仓内未找到 AIV 通道数的直接文档证据；经验证在全部 20 case 的 A 层均非绑定约束（最大占比 case3 4.2us/26.8us），不影响任何结论。
2. **L2 聚合带宽 ≥4TB/s 为微基准下界**（21us 测量含发射开销污染，真值可能 6-8TB/s）——若真值更高，case7/9 的 B 层由 L2 界（583us）回到管道界（626us），spB 结论不变；case7 的 B 上界从 0.54 升至 0.70x（V 相干化后）。
3. **V.vec 跨形态线性元素缩放 ±30% 不确定**——form_a（case7/8）无 rev4 态真机 op 模式复测，B 层 per-iter 1.411us 由 case9（form_b_mma）缩放外推；若实际次线性（固定指令开销占比高），case7/8 的 B 层绑定管道转为 C.fix（1.022us/iter），spB 0.54→0.70x。
4. **FLOOR 9us 按现 kernel 形态校准**——理想 kernel（更简单结构）floor 可能更低；小 case 的 spA（3.64x）依赖 baseline 32.8us 地板的 bench 记账，属"bench 口径内理论值"。
5. **case10 bench（53.94us）与本地 msprof（34.98us）口径差未归因**——两者均如实列出；decode 结论不受影响（其余 decode case 双口径一致 <5%）。
6. camodel 数据全程仅作机制定性（判读纪律 §S6-3-3：模型 FIXP 94.98% vs 真机 35.7% 放大三度实证）；全部定量结论以真机 op 模式 CSV + 微基准为唯一依据。
7. 本分析未运行任何 kernel（零代码变更、零评测复跑）；②③⑤ 的预期收益为模型推算，实施前需探针/竞态/精度全门禁验证。

## 5. 产物与复算

- 本报告: `perf_tuning/analysis_20260928_theoretical_perf_and_ideal_pipeline.md`
- 建模脚本（含全部常数出处注释）: `perf_tuning/analysis_20260928_roofline_model.py`（输出快照 `analysis_20260928_roofline_output.txt`）
- 引用数据: `msprof_case9_20260923/{README.md, summary_r7_ab.txt, op_device/*.csv, simulator_core0/*_instr_exe.csv, core0_tracestats_full.txt}`；`final_report.md` §S4/S6/S7；`.orchestrator_state.json` perf_iteration.session7_result

---

## 附录：理论最佳流水排布示意图（2026-09-28 用户追问追加）

> 样板 = case9 / Form B-mma（bM=32×bN=128，D=256，442 iter/核）；节拍墙 T=1.112us/slot（C.MTE2 139KB @125.2GB/s）。数字出处同正文 §0/§2。

### 附.1 现状排布（真机实测，wall 944.7us，节拍 2.14us/iter → 47% 气泡）

```
时间 ───────────────────────────────────────────────────────────────────────────►
        ◄───── G1 相位 ─────►◄─ 排空 ─►◄───── G2 相位 ─────►◄─ 排空 ─►║ 任务边界 ║
        （批深名义 8 槽，causal eff_kv 均值 4.5 ⇒ 半空转；每核 103 次边界重灌）

C.MTE2  ██ K,V(0..4)+装填 ██░░░░░░░░░░░░░░░██ V(0..4)+P(w2) ██░░░░░░░░║ refill ░░
C.MTE1  ▒▒ 2槽池·零前瞻 ▒▒←stall→          ░░░░░            ▒▒stall→ ║
C.CUBE  ▒MM▒等▒MM▒等▒MM▒ (20.2%)           ░░░░             ▒MM▒等    ║
C.FIX   ░ w1(0..4) fp32 ░                  ░░░              ░ w3 colseg×2 ░
C.scalar████ 88% busy：mte1_stall 466us(49%wall) + wait_id2 310us + wait_id5 86us ████
        ─────────────────────── 跨核 SEM（每槽 4 次传播+排空）────────────────────
V.MTE2  ░░░░ 空转（等 w1）░░░░░░░░░░░░░    ██ w1 ██         ░░░ ██ w3 ██║
V.VEC   ░░░░ 空转 ░░░░░░░░░░░░░░░░░░░░░    ██ SM 碎片化 ██   ░░░ ██ AO ██║
V.MTE3  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    █ w2 █           ░░░░░       ║
V.FLOW  ████ 74.2% wall 在等 C（wait_id4 411us + wait_id0 225us）██████████
```

### 附.2 理想稳态：flat(task,kv) 连续令牌流（槽粒度交错，无批边界）

```
稳态每节拍 T=1.112us：C 侧同时推进 G1(i+1) 与 G2(i)，V 侧同时推进 SM(i) 与 AO(i-1)

           ◄─────── 节拍 t ───────►◄────── 节拍 t+1 ──────►◄────── 节拍 t+2 ─────►
C.MTE2     █ K,V(i+2) 128KB + w2(i) 8KB █ K,V(i+3)+w2(i+1) █ K,V(i+4)+w2(i+2) █  100%★节拍墙
C.MTE1     ▒ G1(i+1):K→4×[128,64] ▒ G2(i):V→4×[128,64] ▒ G1(i+2) ▒ …（前瞻2槽）  64%
C.CUBE     ░ G1(i+1):MM×4链(KC=64) ░ G2(i):MM×4链(NC=64) ░ G1(i+2) ░ …            37%
C.FIX      ▓ w1(i+1) 16KB ▓ w3(i) 4×8KB colseg ▓ w1(i+2) ▓ …                      67%
                │set WS1_C2V(i+1)         │set WS3_C2V(i)
                ▼                         ▼
V.MTE2     ▒ w1(i) 16KB ▒ w3(i-1) 32KB ▒ w1(i+1) ▒ …（双槽乒乓，2×AIV 分担）      39%
V.VEC      █ SM(i) 相干宽指令 █ AO(i-1) rescale █ SM(i+1) █ …                       18%
V.MTE3     ▒ w2(i) 8KB ▒ O(仅任务尾·直写窗口) ▒ …                                   12%
                │set WS2_V2C(i) ────► C 在节拍 t+1 读 w2(i)（信用闭环，槽 i 即刻释放）

★ 相位差：G1(i)@t-1 → SM(i)@t → G2(i)@t+1 → AO(i)@t+2 —— 链条跨 4 槽，
  NUM_STAGES=8 信用 ≥ 2×链深 ⇒ 延迟全隐藏；flat 队列使槽跨任务边界不排空。
```

### 附.3 C 核迭代内微观排布（MTE1 stall 根治：L0B 4 槽旋转）

```
【现状】l0b_pool = 2×[128,128]（64KB 顶格）—— G1/G2 各耗尽双槽，跨迭代零前瞻
MTE1   [K c0 32KB][K c1 32KB]            [V n0][V n1]            ╳等待╳ [次迭代 c0]…
CUBE         └ 等装载 ┘[MM c0 149cy][MM c1]      [MM n0][MM n1]
FIX                       [w1]                              [w3 s0][w3 s1]
                    ▲ 纯延迟暴露：scalar mte1_stall 466us（MTE1 busy 仅 335us < stall）

【理想】l0b_pool = 4×[128,64]（同 64KB）—— MMAD 读操作数即释槽，装载前瞻深度 2
MTE1   [c0][c1][c2][c3]→[n0][n1][n2][n3]→[c0'][c1']…  连续流（16KB/块@191GB/s=84ns）
CUBE       [MM c0][MM c1][MM c2][MM c3]→[MM n0]…      （45ns/块 < 84ns ⇒ MTE1 成次级墙 64%）
FIX                       [w1 16KB]………………[w3 s0][s1][s2][s3]
                    ▲ 暴露 stall ≈ 0
代价（如实）：MMAD 发射 4→8 次/iter；w3 FIX 事件 3→5 次/iter（事件税先例 884→1326 已观测）
```

### 附.4 槽生命周期 / SEM 信用环（协议不变，仅组织方式变）

```
  C.G1(i) ──FIX w1(i)──► [WS1_C2V] ──► V.MTE2 w1(i) ──► [WS1_V2C] ──► 槽i释放（C 可重灌）
                                            │
                                       V.VEC SM(i)（8 槽合并一次宽通道，消 36×BAR/34×MOVEMASK）
                                            │
  C.MTE2 w2(i) ◄──[WS2_V2C]◄── V.MTE3 w2(i)     ──[WS2_C2V]──► V 侧 w2 缓冲复用
  C.G2(i) ──FIX w3(i)──► [WS3_C2V] ──► V.MTE2 w3(i) ──► [WS3_V2C] ──► V.VEC AO(i)
```

### 附.5 decode 域理想排布（A=B=HBM 墙；目标有效带宽 895 → ~1400 GB/s）

```
C.MTE2  ████ K(j) ████ V(j) ████ K(j+1) ████ V(j+1) ████  永不停流（节拍=HBM 1472GB/s÷20核）
C.MTE1  ▒▒ K→L0B 令牌链 ▒▒ V→L0B ▒▒（⑤mma化：消除 gemm_v0 每-step PipeBarrier<PIPE_M> 排空）
C.CUBE  ░MM(j)░（bM=16 填充效率 6% —— 带宽墙下无关紧要）
C.FIX   ░w1░░w3░（④w3 瘦身可选：fp32→fp16，FIX/L2 流量 -25%）
V.*     ██ SM(j) ██ 槽粒度交错（非批粒度；decode 无 rescale 链时走单块特化直写）
```

### 附.6 逐管道占空预算（case9，T=1.112us）

| 管道 | 每槽工作量 | 理想占空 | 现状占空（真机） | 差距性质 |
|---|---|---|---|---|
| C.MTE2 | K,V 128KB + w2 8KB | **100%（节拍墙）** | 53.1% | 气泡=交接链，非带宽 |
| C.MTE1 | 136KB（4 槽旋转） | 64% | 35.5%（+stall 466us 暴露） | 前瞻深度 0→2 |
| C.CUBE | 8×MMAD = 2.1M MAC | 37% | 20.2% | 始终非瓶颈 |
| C.FIX | w1 16KB + w3 32KB | 67% | 35.7% | ④瘦身后 →45% |
| V.MTE2 | w1+w3 48KB ÷2AIV | 39% | 20.1% | — |
| V.VEC | SM+AO 相干化 | **18%** | 49.5%（碎片化税） | ③批相干化 |
| V.MTE3 | w2 8KB + O(任务尾) | 12% | 9.3% | — |

收敛点：理想 wall = 442×1.112us + 爬坡 ≈ **626us（case9 0.353x→0.54x）**；626→191us（A 层 1.77x）的差距 = w1/w2/w3 全部流量（CV-over-GM 架构成本），规则内不可回收。吸收律约束：附.2/附.3 手段（①flat 令牌流 ②L0B 4 槽 ③V 相干化）**必须联合落地**，单项实施被对侧等待 1:1 吸收（r7-1/r7-2 实证）。
