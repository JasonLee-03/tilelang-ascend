# GQA Roofline 访存瓶颈分析报告（Stage 3 · round 2 · iteration 6 · analysis_only）

- 日期：2026-10-19（遗留数据采集会话 07:57–08:09 由前一会话完成；本会话核验 + 补采 10:45–10:50 + 分析成文）
- 对象：`examples/gqa/gqa.py` shipped 版（md5 `bd470e072d0a48faeac1e786306932e8`，开工/收工核验一致，本轮**未修改**
  `gqa.py` / `test_gqa.py`（md5 `3e2d08881cb7b6f36d5c89b2fd37a146`）；`.orchestrator_state.json` 零触碰）
- 用户任务书："采集相应的流水数据，通过 roofline 分析访存瓶颈，细化到具体管线，根据具体管线的空闲率，
  分析出主要访存瓶颈导致原因，制定下一步优化方案" —— 5 项交付逐项对应 §1（流水数据采集核验+补采）、
  §4（roofline 定位）、§5（九管线 busy/idle）、§6（空闲率+wait-id 定量归因）、§7（优化方案清单）
- 方法学纪律：全部真实 NPU（camodel 全量仿真禁用，R1 既有 sim trace 仅作指令级佐证引用）；峰值全部有出处
  （§3 附录：C 微基准实测 + 平台 ini 规格交叉对照，禁止凭记忆）；伪影过滤规则沿用 iter4/5（本轮 A/B 矩阵
  0 行被过滤——数据干净）；跨会话数字仅作参照锚点并显式标注

---

## 0. 结论速览（verdict）

**主要访存瓶颈不在 HBM（唯一例外 decode），而在"expert CV 分离架构的 ws GM(L2) 往返 + aux 纯搬运税 +
标量/握手发射"三层：**

1. **prefill main kernel（flat kv1/kv2 域）**：ws_s/ws_p/ws_o 的 C↔V 交换往返占 L2 侧流量的 **62–77%**
   （R4 实测 320MB/417MB），每 u 付出 320KB ws 字节 vs 96KB 有效操作数字节（**3.3× 倍率**）；HBM 侧
   只有 197–514 GB/s（L2 命中 85–92% 吸收）——**税的主体是 L2 带宽占用 + 往返两跳延迟进入 pacing 链**
   （R4 的 C-id1/V-id3 互等各 ~104µs = wall 55%，aic_scalar 83.7% 为全表最忙管线），不是 HBM 字节。
2. **decode（dense_expert）**：唯一的真 HBM 墙——K/V 深流 134.2MB 首触必读，物理速率 **920 GB/s =
   MTE2 规格屋顶（1.056 TB/s）的 87% / 实测混合 HBM 屋顶（~980 GB/s）的 94%**，热点核 aic_mte2 95.7%。
    roofline 已贴屋顶，剩余理论空间 ≤13%（含 7/6 任务不均衡），**不可为清单**（§7.4）。
3. **aux 纯搬运税（prefill e2e 的最大单项）**：transpose_qkv + transpose_out 占 e2e 设备时间的
   **23–61%**（同会话 B 矩阵；eval 跨会话下界口径 19–46%）。其中 **transpose_out 并行度坍缩是重灾区**：
   任务网格 `B·ceil(Nq/bm)·ceil(S/bn)` 在中小 shape 只有 1–16 任务（R1 = **1 个任务 = 40 个向量核中的
   1 个在干活**，42.6 GB/s = L2 屋顶的 1.2%），单核 12µs 超过 main kernel 本体（10.6µs）。
4. **可回收空间排序（e2e 口径，模型外推）**：aux 修复（P1 重铺 tile：−5..−31%/shape；P2 main 直写
   BSND 消除 transpose_out：−10..−44%）> main kernel 结构性瘦身（P3 FIX→UB 直达消除 ws_s/ws_o 往返：
   低置信 −10..−18%，需可行性探针）> kv2 vec busy 窄化（+1–2%）；decode/R1-main/F-launch 贴屋顶或
   触框架地板，不可为。

---

## 1. Phase 0 —— 遗留数据核验（复用/补采/丢弃清单）

### 1.1 核验方法与结果总表

对前一会话落盘的 40 个 `msprof_op_iter6_*` 目录逐项核验：CSV 完整性（非零文件数）、派发正确性
（`[PROF_DONE] chosen=` 日志 + block dim）、数值锚点（±10% 容差）、C 微基准自洽性。

| 组 | 目录 | 核验结果 | 判定 |
|---|---|---|---|
| A 矩阵（main kernel，18 目录） | `msprof_op_iter6_A_r{1..6}_s{1..3}` | 18/18 rc=0、每目录 8 CSV 全非零；chosen 逐 run 核验（r1=per_head_bm64、r2/r3/r6=g2stack_flat、**r4=g2stack_flat_kv2（iter5 采纳 kernel）**、r5=dense_expert）；6 工况 TaskDur 锚点全命中（§1.2）；伪影行过滤命中 **0 行**；全部 1650/1650 MHz | **全部复用** |
| B 矩阵（aux 全链，12 目录） | `msprof_op_iter6_B_r{1..6}_allk_s{1,2}` | 12/12 rc=0；prefill 各目录含 kernel_kernel/{0,1} + main_kernel/0 三实例（transpose_qkv → main → transpose_out 完整链，blk 与任务网格公式逐一吻合 §4.4）；r5 decode 仅 main_kernel（BSND 直供无 aux，launch-count=3 下无隐藏 launch）；main 数值与 A 矩阵交叉一致（±1.5%） | **全部复用** |
| C 微基准（10 目录） | `msprof_op_iter6_C_{copy,copyl2,gemm,torch_gemm,vec}_s{1,2}` | **copy/copyl2 s1-s2（4 目录）无效**：脚本 `src[base, :]` 1-D 切片 bug——每迭代实际只搬 256B（Memory.csv GM_to_UB=24KB/task ÷ 96 iter = 256B），TaskDur 61µs 对声称的 251.7MB 折算 8 TB/s 为物理不可能值 → **丢弃**（目录保留在盘作披露）；gemm 数据自洽但 kernel 本身 latency-bound 不能作 cube 屋顶（§3.2）；torch_gemm / vec 有效 | **copy/copyl2 修复后补采 s3+s4（4 目录，本会话 10:45–10:50）；gemm/torch_gemm/vec 复用** |
| 工具 | `prof_gqa_iter6.py` / `run_matrix_iter6.sh` / `parse_msprof_iter6.py` / `microbench_peak_iter6.py` | harness 契约完整（单发 gqa() + chosen 打印 + 软派发校验）；parse 工具有一处 None 格式化 bug（torch_gemm 无 ArithmeticUtilization 行时崩溃，导致前会话 vec 组未输出）→ 本会话修复（分析工具，允许修改）；microbench copy_kernel 切片 bug 修复（perf_tuning/ 内注释标记，s1/s2 无效原因写明） | 修复后复用 |

补采命名沿用既有规则续采不覆盖：`msprof_op_iter6_C_copy_s{3,4}`、`msprof_op_iter6_C_copyl2_s{3,4}`，
rc=0，npu-smi 健康、无残留进程（本会话复核 0 个 msprof/python 探针进程）。

### 1.2 数值锚点核验（A 矩阵 TaskDur 3 样本均值 vs 任务书锚点）

| 工况 | 锚点（任务书） | iter6 实测 | 偏差 | 判定 |
|---|---|---|---|---|
| R1 per-head | ~10.8µs（iter3） | **10.34** [10.04–10.66] | −4.3% | ✓ |
| R2 flat kv1（c1 类） | ~26µs | **27.22** [26.40–27.86] | +4.7% | ✓ |
| R3 flat kv1（c3 类） | ~61µs | **60.53** [60.12–60.86] | −0.8% | ✓ |
| R4 flat_kv2（c4 类，iter5 采纳值） | ~188µs | **189.65** [188.74–190.42] | +0.9% | ✓ |
| R5 decode dense_expert | ~145µs | **145.89** [145.12–147.16] | +0.6% | ✓ |
| R6 flat kv1（c2 类） | ~27.5µs | **28.02** [27.56–28.68] | +1.9% | ✓ |

全部在 ±10% 容差内（实际最大偏差 4.7%，散布 ≤6%）→ **A/B 矩阵整体可信，零目录重采**。

### 1.3 派发正确性佐证（R4 重点）

采集日志逐 run `[PROF_DONE] chosen=['g2stack_flat_kv2'] expected=g2stack_flat_kv2`（3/3 样本），
blk=20（Fixed-Core expert）、TaskDur 189.65µs 与 iter5 Gate 2 采纳值 187.97µs 同带（+0.9%，跨日漂移内）。
日志中 21+14 条 `[WARN]` 均为无害项：r1 软校验字符串前缀不匹配（`per_head_bm64` vs expected `per_head`，
chosen 实为正确派发）+ msprof 输出目录组写权限提示。

---

## 2. 采集矩阵数据总表

### 2.1 A 矩阵（main kernel，msprof op --kernel-name=main_kernel，FULL metric 组）

metrics = PipeUtilization,Memory,L2Cache,ArithmeticUtilization,MemoryUB,MemoryL0,ResourceConflictRatio；
每工况 3 样本；含 wait-id 遥测列（**本轮 A 矩阵自带遥测，无需补采**，语义沿用 iter3 §2.3 结构核对：
C 侧 id1/id2/id5 = 等 V 释放 ws_s / 等 ws_p 就绪 / 等 ws_o 释放；V 侧 id0/id3/id4 = 等 ws_s 就绪 /
等 C 消费 ws_p / 等 ws_o 就绪）。

| 工况 | TaskDur µs（均值[散布]） | blk | 热点核窗口 aic/aiv µs | GM→L1 MB | L0C→GM MB | GM→UB MB | UB→GM MB | L2 rd/wr hit % |
|---|---|---|---|---|---|---|---|---|
| R1 | 10.34 [6.0%] | 16 | 6.48 / 7.28 | 1.50 | 1.00 | 1.50 | 0.50 | 85.3 / 1.5 |
| R2 | 27.22 [5.4%] | 20 | 22.75 / 24.39 | 10.0 | 8.0 | 12.0 | 4.0 | 84.4 / 31.4 |
| R3 | 60.53 [1.2%] | 20 | 56.00 / 57.65 | 40.0 | 32.0 | 48.0 | 16.0 | 88.6 / 70.3 |
| R4 | 189.65 [0.9%] | 20 | 185.12 / 186.72 | 128.0 | 128.0 | 128.0 | 48.0 | 91.7 / 87.4 |
| R5 | 145.89 [1.4%] | 20 | 142.13 / 143.01 | 136.75 | 32.0 | 32.0 | 8.1 | 61.3 / 84.1 |
| R6 | 28.02 [4.0%] | 20 | 23.48 / 25.07 | 15.0 | 12.0 | 12.0 | 6.0 | 85.4 / 48.7 |

（路径量为 Memory.csv 逐行求和、3 样本均值；样本间逐位稳定 ≤0.1%。）

### 2.2 B 矩阵（aux 全链分离，同会话）与 e2e 份额

| 工况 | transpose_qkv µs (blk) | main µs (blk) | transpose_out µs (blk) | e2e Σ µs | **aux 占 e2e** | eval 跨会话参照† |
|---|---|---|---|---|---|---|
| R1 | 4.67 (6) | 10.61 (16) | **12.01 (1)** | 27.29 | **61.1%** | —（无对应 case） |
| R2 (c1) | 10.02 (48) | 26.06 (20) | 13.21 (4) | 49.29 | **47.1%** | 37.5%（aux 14.34 vs main 23.86） |
| R3 (c3) | 18.38 (32) | 61.34 (20) | 24.10 (8) | 103.82 | **40.9%** | 37.7%（37.06 vs 61.29） |
| R4 (c4) | 27.77 (160) | 187.71 (20) | 29.31 (16) | 244.79 | **23.3%** | 18.6%（42.66 vs 186.12） |
| R5 (c11) | —（无 aux，BSND 直供 ✓） | 146.70 (20) | — | 146.70 | **0%** | 0%（无 kernel_kernel 记录） |
| R6 (c2) | 10.85 (56) | 27.52 (20) | **22.10 (3)** | 60.47 | **54.5%** | 46.2%（26.4 vs 30.69） |

† eval 参照 = `reports/prof_data/level4/gqa/{1,2,3,4,11}/.../op_statistic.csv`（2026-10-18 评测会话，
5 迭代连续、无 per-kernel metrics 注入）——**跨会话仅作量级参照**（纪律）。两口径差异（本会话 aux 高
20–60%）来源推断：msprof op 逐 kernel 指标注入对小 kernel 的相对开销 + L2 冷热态差异；**排序与结论
在两口径下一致**：prefill aux 税 19–61%，decode 为 0。这直接解释 cann-bench 的 prefill avg_speedup
~0.45 vs decode ~0.96。

pad kernel（`_pad_kv`）不在本矩阵 6 工况内（全部 Skv 对齐 + decode Skv=2048 对齐 → 无 pad launch，
B_r5 的 launch-count=3 采集证实无隐藏 kernel）；非对齐 shape 的 pad 税未测（§8 未验证项）。

---

## 3. 峰值来源附录（roofline 屋顶，禁止凭记忆——全部有出处）

### 3.1 实测屋顶（C 微基准，本会话补采后有效集）

| 屋顶 | 实测值 | 方法与 shape | 样本 |
|---|---|---|---|
| **cube（供应商 kernel 锚）** | **240.1 TFLOPS** fp16 | torch.matmul（MatMulV3_ND…high_performance）4096³ fp16，TaskDur 572.42µs [571.92–572.92]，aic_cube 92.7%、热点核窗口 572.0µs ≈ TaskDur（单波 blk=20）；137.4 GFLOP 解析核算 ✓ | s1,s2 |
| **HBM 混合读写（流式）** | **1.307 TB/s**（L2 侧口径 503.3MB r+w）；**HBM 侧 ~0.98 TB/s**（读侧 50% L2 命中折扣后 377MB） | 修复后 copy_kernel：40 vec 任务 ping-pong GM→UB→GM，64KB tile ×96 iter，251.7MB/方向 ≫ L2 168MB；TaskDur 384.95µs 均值 [384.24, 385.66] 散布 0.4%；aiv_mte2 busy 76.7% | **s3,s4（补采）** |
| **L2 驻留混合读写** | **3.578 TB/s**（335.5MB r+w） | copyl2：私有 4-tile 环 ×64 pass，足迹 20MB ≪ 168MB，L2 read hit 96.9% / write hit 93.8%；TaskDur 93.77µs 均值 [94.16, 93.38] 散布 0.8% | **s3,s4（补采）** |
| **vector（add 类）** | **4.39 TFLOPS** fp16（94.7% busy → 折合 100% busy ≈4.6） | vec bench：40 任务 UB 内 vadd 链 6000 iter ×4096 elem；msprof aiv_vec_fops=1.505 GFLOP / TaskDur 342.91µs；**注意 op 类**：单操作数 add + repeat 开销，仅为理论 FMA 屋顶（16.9 TFLOPS = 128 lane×2×40 核×1.65G）的 26% —— 用作 vec-bound 比较的**同类 op 量级锚**，不当 FMA 屋顶用 | s1,s2 |

s1/s2 的 copy/copyl2 **无效已丢弃**（切片 bug，§1.1）；gemm（tilelang Expert 朴素 4096³）见 §3.2。

### 3.2 规格屋顶（平台 ini 交叉对照，出处：`/home/developer/Ascend/cann-9.2.0/aarch64-linux/data/platform_config/Ascend910_9362.ini`）

| 规格项 | ini 值 | 推导屋顶 | 与实测对照 |
|---|---|---|---|
| cube | 20 核 × 16×16×16（=8192 FLOP/cyc fp16）× cube_freq 1500MHz（实测运行频率 **1650/1650**，全部 40+ 采集 run 的 OpBasicInfo Current/Rated） | **245.8 TFLOPS @1500 / 270.3 @1650** | torch_gemm 240.1 = 97.7% @1500 口径 / 88.8% @1650 口径——两口径均自洽；**报告统一用 @1650 = 270.3 作理论屋顶、240.1 作经验屋顶** |
| GM(MTE2) 读 | ddr_rate = 32 B/cyc/core | 52.8 GB/s/core × 20 = **1.056 TB/s 聚合** | decode 物理 K/V 流 920 GB/s = 87%（§6.2）；copy 微基准 HBM 侧 ~0.98 TB/s = 93%——**双独立路径闭合** |
| L2 读 | l2_rate = 110 B/cyc/core（写 86） | AIC 侧聚合 3.63 TB/s 读 | copyl2 实测混合 3.578 TB/s ≈ 同量级 ✓ |
| UB→GM 写 | ub_to_ddr_rate = 64 B/cyc/core | 40 × 105.6 = 4.2 TB/s（接口） | 受 HBM 共享上限约束，未独立达至（copy 混合实测为准） |
| vector | vec_calc_size = 128 lane | fp16 FMA 16.9 TFLOPS / 单 op 8.45 Tops | vec bench add 类 4.39 TFLOPS = 26%（op 类差异如上） |
| 容量 | L2 = 176160768B = **168MB**；L0A/B 64KB、L0C 128KB、L1 512KB、UB 192KB/vid | —— | 与 flat_pipeline_analysis §4.2 的 192MB 旧口径相比按 ini 修正为 168MB（ws ≤13MB ≪ 168MB 结论不变） |

**tilelang Expert gemm 微基准（gemm_s1/s2）判定为无效屋顶**（数据保留、如实披露）：TaskDur 1858µs =
74 TFLOPS，512 block × 每 block 窗口 ~88µs 而 cube 忙仅 ~22µs/block——每 k 迭代两次 `T.barrier_all` +
无流水的 copy→gemm 串行使 kernel latency-bound（每迭代 wall 1.375µs vs 计算 0.31µs）；其 PipeUtilization
的 cube_ratio 行与 fops 计数内部不自洽（1.2% × TaskDur 折算与 137 GFLOP 解析量差 23×，多波次 block 行
口径问题——iter4/5 伪影行同类）。cube 屋顶一律以 torch_gemm 为准。

### 3.3 计数器口径校准（解析模型对账得出，全部标注推断级别）

| 现象 | 校准结论 | 依据 |
|---|---|---|
| GM_to_L1（prefill 系）= 解析(Q+K+V+ws_p) **+ 1×Q** | 系统性 +1×Q 计数怪癖（R1 +20%、R2/R3/R6 +25%、R4 +14.3% 恰均为一个 Q 体量；decode 无此偏差 +0.2%） | 解析模型 §4.1 逐路径对账；**推断**：Q 双子盒 [64,128]×2 装入 Nz 布局 q_l1 时按目标盒足迹计数或双盒描述符重复计数——未从文档核实，roofline 以实测路径量为准，双口径差 ≤25%（仅 GM_to_L1 一列） |
| read_main_memory ≈ GM_to_L1 + GM_to_UB − Q | 聚合列与分路径和自洽（R2/R3/R4 偏差 <0.1%、R5 <0.2%） | 对账表 §4.1 |
| write_main_memory ≈ L0C_to_GM + UB_to_GM + L1_to_GM(est) | 同上（偏差 ≤0.3%） | 同上 |
| L2Cache 读计数单位 = 128B 线；**写计数单位 = 512B** | 读：R5 aic_rd 总 ×128B = 142.7MB ≈ GM_to_L1 136.75（+4%）；写：R4 (aic_wr+aiv_wr)×512B = 134MB ≈ L0C_to_GM+UB_to_GM+L1_to_GM = 177MB 的 76%… R5 ×512B = 33.6MB ≈ 41.7MB 路径和（−19%）——512B 口径两工况一致优于 128B（差 4×） | **推断**（无文档出处，按两工况路径量最小二乘校准）；HBM 写流量估算按 512B/计数并注明 ±20% 口径带 |
| PipeUtilization ratio 归一化窗口 = **TaskDur**（非 per-core 窗口） | gemm 微基准多波次下 cube_ratio×TaskDur = 19.9µs/block ≈ 解析 mma 时间；单波 kernel（blk=20）两口径重合（窗口≈TaskDur） | 对账（§3.2 gemm 行）；与 iter3 用法一致 |
| decode L2 read hit 61.3% 与首触物理矛盾 | K/V 每 (bz,kg) 恰读一次、零跨任务复用 → 物理 HBM 读 ≥134.2MB；"命中"半数疑为 256B 取数按 2×128B 计数（1 miss+1 hit）之类计数伪影 | **推断**；decode HBM 结论采用物理下限口径 920 GB/s（§6.2），计数器口径（514 GB/s）并列披露 |

---

## 4. Roofline 定位（每 kernel：FLOPs / 字节量解析模型 → AI → 实测点 → 距屋顶/脊点）

### 4.1 解析模型（公式；系数从 gqa.py 源码核实）

**FLOPs**（有效口径 = 2·M·N·K 对有效维度；padded 口径另列）：

- prefill stacked 系（flat kv1 / flat kv2 / g2stack）：每 u（= 每 kv 块迭代）G1+G2 = `2 × 2·M_stack·N·K`
  = 2×2×128×128×128 = 8.39 MFLOP；总量 = 8.39M × U，U = tasks × kv_useful；
  tasks = `B·N_kv·(G/2)·ceil(S/64)`（G=N_q/N_kv，stacked 双头 M_stack=128）。
  softmax vector 项 ≈ 4 op/elem × M_stack×N × U ≈ GEMM 的 ~1%（列入 vec busy 讨论，不入 AI 主项）。
- per-head（R1）：每 task G1+G2 = 2×2×64×128×128 = 4.19 MFLOP × 16 task。
- decode（R5）：有效 = `2 GEMM × 2·B·N_q·S·S_kv·D` = 537 MFLOP；padded（mma 实跑 M_tile=16，
  由 L0C/GM_to_UB/UB_to_GM 三路径计数联立校准，§4.2）= 2×2×16×128×128×2048u = 2.15 GFLOP（4× 填充税）。

**GM/L2 字节**（每路径解析式；K/V 是否每 task 重载——**是**，从代码核实：`emit_g1` 每 u 发
`T.copy(K[bz,kg,kv窗口],k_l1_*)` / `T.copy(V[...],v_l1_*)`（gqa.py:3570–3581），L1 槽按 kv 奇偶双槽、
**无跨任务驻留** → K/V tile 每 u 重载一次；唯一字节 = B·N_kv·S_kv·D·2B×2，重载倍率 = tasks×kv/U ÷ 唯一量）：

| 路径 | 解析式（prefill stacked 系） | R4 值 |
|---|---|---|
| GM→L1（MTE2-C） | `Q: tasks×128×128×2B` + `K,V: U×128×128×2B×2` + `ws_p 回读: U×128×128×2B(fp16)` | 16+32+32+32 = 112MB（实测 128 = +1×Q 口径 §3.3） |
| L0C→GM（FIX） | `ws_s + ws_o: U×128×128×4B×2`（fp32 累加直写） | 128MB（实测 128.0，**0.0%**） |
| GM→UB（MTE2-V） | `ws_s + ws_o 回读: U×128×128×4B×2` + `mask(causal): U×64KB fp32` | 128MB + 0（实测 128.0，**0.0%**） |
| UB→GM（MTE3-V） | `ws_p: U×128×128×2B` + `Output: tasks×128×128×2B` | 32+16 = 48MB（实测 48.0，**0.0%**） |
| **ws 往返合计** | `U×(64+64+64+64+32+32)KB = U×320KB`（ws_s 写+读、ws_o 写+读、ws_p 写+读） | **320MB** |
| decode 特化 | GM→L1 = `U×64KB(K,V) + U×4KB(ws_p,M16) + tasks×4KB(Q)`；L0C→GM = GM→UB = `U×16KB(ws_s+ws_o fp32 [16,128])`；UB→GM = `U×4KB(ws_p) + Output` | 136.6/32/32/8.1MB（实测 136.75/32.0/32.0/8.125，**≤0.2%**） |
| per-head 特化（R1） | GM→L1 = `16×(16+32+32)KB`；L0C→GM = `16×2×32KB(scores+partial fp32)`；GM→UB = L0C 回读 + mask `16×32KB`；UB→GM = Output `16×16KB` | 1.25/1.0/1.5/0.25MB（实测 1.5/1.0/1.5/0.5——GM→L1 +Q 口径、UB→GM +256KB 未解释 2×（**推断**：AUTO CV-combine 双 vid 计数），绝对量 0.25MB 可忽略） |

**解析 vs 实测总对账**：15/18 个（路径×工况）组合偏差 ≤0.2%，其余 3 处为已披露口径差
（+1×Q 系、R1 UB→GM）——解析模型可信，交叉验证要求（任务书方法学 3）满足。

### 4.2 交叉验证：解析字节 × 实测 TaskDur → 带宽利用率 vs 管线 busy%

| 工况 | GM→L1 均速 GB/s | aic_mte2 busy | **busy 期折算速率** | 对照屋顶 | 自洽性 |
|---|---|---|---|---|---|
| R5 | 983（全装载口径；物理 K/V 下限口径 920） | 86.2%（热点核 95.7%） | 983/0.862 = **~1140 GB/s** | 规格 MTE2 聚合 1056 GB/s | **108% 标称——busy 期达到/略超规格接口速率**（超出部分的解释二选一：ini ddr_rate 为 1500MHz 标称偏保守，或读流量含真实 L2 命中成分；两种解释下结论不变：管线贴接口上限跑）✓ |
| R4 | 708 | 29.2% | 2423 GB/s | L2 读聚合 3.63 TB/s（规格） | 67%（L2 hit 91.7% → 走 L2 高速路 ✓） |
| R3 | 693 | 29.3% | 2364 GB/s | 同上 | 65% ✓ |
| R2 | 385 | 18.0% | 2141 GB/s | 同上 | 59% ✓ |
| V 侧（R4） | GM→UB 708 | aiv_mte2 22.5% | 3145 GB/s | L2 读（AIV 侧） | ws_s/ws_o 回读 ~100% L2 命中（aiv_rd hit）→ 纯 L2 速率，88% of copyl2 混合屋顶 ✓ |
| FIX（R4） | L0C→GM 708 | fixpipe 27.7% | 2555 GB/s | 无 ini 规格（fix→GM 速率未列）| **相对口径**：与 L2 写聚合 86B/cyc×20×1.65G=2.84TB/s 同量级（90%）|

结论：三路 MTE2/FIX 的 busy% × 实测字节 与双屋顶结构（GM 接口 1.056 TB/s / L2 3.58–3.63 TB/s）定量自洽；
不自洽点（+1×Q、decode hit 计数、FIX 无规格）已在 §3.3 逐项披露。

### 4.3 主 kernel roofline 定位表（实测点 = (AI, 实测速率)）

脊点：HBM 混合 = 270.3/1.307 = **207 FLOP/B**；L2 混合 = 270.3/3.578 = **76 FLOP/B**。

| 工况 | kernel | FLOPs eff | AI（L2 侧口径） | 实测 TFLOPS eff | 距 cube 屋顶 | L2 侧流量 / 速率 | 距 L2 屋顶 | HBM 侧速率 | **bound 判定** |
|---|---|---|---|---|---|---|---|---|---|
| R1 | per_head_bm64 | 67 MFLOP | 14.1 | 6.5 | 2.4% | 4.5MB / 460 GB/s | 12.8% | 197 GB/s | **延迟/launch 受限**（单波 16 task、无任何管线 >45%、4 核闲置；AI<脊点但两屋顶都远未及——roofline 图上落在"斜线以下的空旷区"= 纯延迟区） |
| R2 | g2stack_flat | 537 MFLOP | 15.9 | 19.7 | 7.3% | 32.1MB / 1237 GB/s | 34.6% | 438 GB/s | **F 固定开销 + pacing**（F≈10.6µs = wall 39%，iter3 拟合；稳态段 vec 34.8%） |
| R3 | g2stack_flat | 2.147 GFLOP | 16.0 | 35.5 | 13.1% | 128.3MB / 2223 GB/s | 62.1% | 405 GB/s | **vec/scalar 共界 + pacing**（vec 62.4% = pacer、scalar-C 57.6%；L2 侧 62% 但未贴） |
| R4 | g2stack_flat_kv2 | 8.590 GFLOP | 19.6 | 45.3 | 16.8% | 417.2MB / 2307 GB/s | **64.5%** | 234 GB/s | **scalar 发射/pacing 界**（scalar-C 83.7% 全表最忙；vec 64.4% pacer；L2 侧 64.5%——三重接近但都未贴屋顶，节拍由握手链设定） |
| R5 | dense_expert | 537M eff / 2.15G padded | 2.5 / 10.3 | 3.7 / 14.7 | 1.4% / 5.4% | 208.6MB / 1499 GB/s | 41.9% | **920 GB/s（物理读）** | **HBM/MTE2 墙**（87% 规格 / 94% 经验屋顶；热点核 mte2 95.7%；AI 2.5 ≪ 脊点 207，深度访存受限 ✓） |
| R6 | g2stack_flat | 805 MFLOP | 18.2 | 28.7 | 10.6% | 42.1MB / 1577 GB/s | 44.1% | 477 GB/s | **F + vec/scalar**（F≈10.6µs = 38%；vec 47.3%） |

**判读**：六个 main kernel 的 AI（2.5–19.6）全部在脊点（76/207）左侧 = 名义访存受限域；但除 R5 外
没有一个贴着带宽屋顶跑（L2 侧 12.8–64.5%）——**"访存瓶颈"的实际形态不是 HBM 字节不够快，而是
(a) ws 往返把 L2 侧流量放大 3.3×（§6.1）、(b) 该流量串在 C↔V 握手链上以两跳延迟设定节拍（pacing）、
(c) 浅 task 工况被 F 固定开销稀释、(d) 只有 decode 是字面意义的 HBM 墙**。

### 4.4 aux kernel（纯搬运，AI≈0）：带宽利用率 vs 实测屋顶

任务网格公式从源码核实并与 blk 实测逐一吻合：transpose_qkv `total = q_total+2·kv_total`（V2 共享 bm
路径 `_shared_bm`→bm=min(512,S,Skv,768)；V1-Q 路径 q_bn=min(Nq,128)），`blk=ceil(total/2)`；
transpose_out（`_transpose_v1`）`total = B·ceil(Nq/bm)·ceil(S/bn)`，bm=16 if Nq≥48 else 8、bn=128
（S 整除约束），blk=ceil(total/2)。

| aux 实例 | 字节 r+w | TaskDur | 聚合速率 | **% of L2 混合屋顶** | blk（任务数） | 病灶 |
|---|---|---|---|---|---|---|
| tqkv R1 | 0.75MB | 4.67µs | 165 GB/s | 4.6% | 6 (12) | launch 地板主导（字节太小） |
| tqkv R2 | 6.0MB | 10.02 | 613 | 17.1% | 48 (96) | 并行度尚可、速率中低 |
| tqkv R3 | 18.0MB | 18.38 | 1003 | 28.0% | 32 (64) | 同上 |
| tqkv R4 | 40.0MB | 27.77 | **1475** | **41.2%** | 160 (320) | aux 中效率最高（3.2 波次） |
| tqkv R6 | 7.0MB | 10.85 | 660 | 18.4% | 56 (112) | — |
| **tout R1** | 0.50MB | **12.01** | **42.6** | **1.2%** | **1 (1)** | **单任务！1/40 向量核干活**（PipeUtilization 行内 min 0%/max 46% 直接证实同 block 双 vid 一忙一闲） |
| tout R2 | 4.0MB | 13.21 | 310 | 8.7% | 4 (8) | 任务饥饿 |
| tout R3 | 16.0MB | 24.10 | 680 | 19.0% | 8 (16) | 任务饥饿 |
| tout R4 | 32.0MB | 29.31 | 1118 | 31.2% | 16 (32) | 尚可 |
| **tout R6** | 6.0MB | **22.10** | 278 | 7.8% | **3 (6)** | 任务饥饿（22µs > main 的 80%） |

**aux 空闲率归因 = "无事可做"（网格太小，多数向量核零任务）+ launch 地板 ~2.5–3.8µs/kernel**（iter3 §4
六工况一致值）；搬运效率本身不差（R4-tout 1118 GB/s 证明 strided 写可达 L2 屋顶 31%）。
transpose_out 的 bm/bn 选择（8/16 × 128）是为 UB 预算与整除约束设计的，从未按"任务数 ≥ 向量核数"
铺满——**这是纯 host 侧参数问题，kernel 逻辑无需动**（§7.1 P1）。

（口径注：本表速率按 Memory.csv 的 KB 列以十进制 MB 折算，较 1024 精确字节口径低 ~2.4%——
如 R1-tout 精确值 43.7 GB/s；对占屋顶百分比与全部结论无影响。§2/§4.2/§4.3 主 kernel 表为字节精确。）

---

## 5. 九管线逐管线 busy/idle 表（A 矩阵，min–max across cores + 均值；ratio 窗口 = TaskDur）

管线口径：AIC 六管（cube/mte1/mte2/mte3/fixpipe/scalar）+ AIV 三管（vec/mte2/mte3）= 九管线，
aiv_scalar 附加列出（十行全表）。idle% = 100 − busy 均值。busy 含等待阻塞的口径注意：scalar 行的
busy 包含 WAIT 指令阻塞期（iter1 §6-5 两可性披露延续）——归因以 wait-id/stall 直测列为准。

### R1 per_head_bm64（wall 10.34µs；热点核窗口 aic 6.48 / aiv 7.28）

| 管线 | busy min–max | busy 均值 | idle 均值 | idle 归因（四分法） |
|---|---|---|---|---|
| aic_cube | 3.6–3.6% | 3.6% | 96.4% | 等对侧核（V softmax 串行链）+ 无事可做（单 task/核） |
| aic_mte1 | 4.3–4.3% | 4.3% | 95.7% | 同上（链上游） |
| aic_mte2 | 9.9–22.4% | 15.2% | 84.8% | 等数据搬运完成即无后续（单波） |
| aic_mte3 | 0% | 0% | 100% | 无事可做（该 kernel 无 MTE3 工作） |
| aic_fixpipe | 6.2–7.9% | 6.9% | 93.1% | 等 cube |
| aic_scalar | 37.6–49.6% | 41.5% | 58.5% | 含 C-id1 阻塞 3.93µs（38% wall，等 V）+ AUTO BAR 链发射 |
| aiv_vec | 15.2–15.2% | 15.2% | 84.8% | 等对侧核（3 次 C↔V 交接，sim trace 佐证 §6.4） |
| aiv_mte2 | 5.1–15.8% | 8.4% | 91.6% | 等数据（ws/scores 就绪） |
| aiv_mte3 | 3.0–3.8% | 3.4% | 96.6% | Output 一次写出后无事 |
| aiv_scalar | 19.2–40.9% | 28.6% | 71.4% | 含 V-id0 3.21µs（31%）+ AUTO BAR |

另：**核外 launch/排空 3.3–3.8µs = wall 的 32–37%**（iter3 §4 逐样本实测口径）+ 4/20 核整核闲置（波次不满）。

### R2 g2stack_flat（wall 27.22；窗口 22.75/24.39；4u/热点核；F≈10.6µs=39%）

| 管线 | busy min–max | 均值 | idle 均值 | 归因 |
|---|---|---|---|---|
| aic_cube | 7.4–10.3% | 8.0% | 92.0% | 等对侧核（V 产 ws_p：C-id2 3.60µs）+ 等 flag 节流（C-id1 9.51µs pacing） |
| aic_mte1 | 7.1–10.4% | 7.9% | 92.1% | 链上游（等 mte2 供 L1） |
| aic_mte2 | 13.3–26.2% | 18.0% | 82.0% | 等 flag 节流（pacing 把 C 前跑压到 ~1 步 → 队列浅，iter4 结论） |
| aic_mte3 | 0% | 0% | 100% | 无工作 |
| aic_fixpipe | 10.6–16.4% | 12.2% | 87.8% | 等 cube |
| aic_scalar | 31.8–51.4% | 37.3% | 62.7% | 阻塞主体 = id1 9.51 + id5 9.58 + id2 3.60（等 V/等 flag） |
| aiv_vec | 32.3–44.4% | 34.8% | 65.2% | F 段（灌注/排空 ~7.8µs）+ 等 C 消费（V-id3 8.02µs） |
| aiv_mte2 | 10.8–17.3% | 13.0% | 87.0% | 等数据（ws_s/ws_o FIX 完成） |
| aiv_mte3 | 7.4–11.5% | 8.6% | 91.4% | 等 flag 节流（ws_p 写被 C 侧单 token 放行节拍限制，mte3_wait cflt 42.5%） |
| aiv_scalar | 29.7–50.3% | 36.4% | 63.6% | 阻塞主体 = id3 8.02 + id0 5.91 + id4 3.50 |

### R3 g2stack_flat（wall 60.53；窗口 56.00/57.65；13u/热点核）

| 管线 | busy min–max | 均值 | idle 均值 | 归因 |
|---|---|---|---|---|
| aic_cube | 13.3–14.6% | 14.3% | 85.7% | 等对侧核（V softmax = pacer；C-id2 10.19µs）+ pacing（id1 35.05µs） |
| aic_mte1 | 13.8–15.7% | 15.0% | 85.0% | 链上游 |
| aic_mte2 | 26.3–32.5% | 29.3% | 70.7% | 等 flag 节流（pacing；busy 期 2.26 TB/s = L2 读规格 62% ✓） |
| aic_mte3 | 0% | 0% | 100% | 无工作 |
| aic_fixpipe | 20.3–24.9% | 22.6% | 77.4% | 等 cube |
| aic_scalar | 49.3–62.5% | 57.6% | 42.4% | 阻塞主体 = **id1 35.05µs（58% wall）** + id5 12.29 + id2 10.19 |
| aiv_vec | 58.5–63.4% | **62.4%** | 37.6% | **pacer 本体**（busy 2.91µs/u = iter3 2.93 复现）；idle = id3 20.34（等 C 消费 ws_p，节流）+ id0/id4 13.56（等 C 数据）+ vector_stall 22.57 |
| aiv_mte2 | 17.8–24.2% | 21.0% | 79.0% | 等数据（ws_s FIX）；含 mask 装载（causal 64KB/u） |
| aiv_mte3 | 13.5–16.7% | 15.4% | 84.6% | 等 flag 节流（mte3_wait cflt **74.6%** = 队列被 ws_p 放行节拍打满） |
| aiv_scalar | 45.7–55.9% | 50.2% | 49.8% | id3/id0/id4 阻塞 + 发射 |

### R4 g2stack_flat_kv2（wall 189.65；窗口 185.12/186.72；26 task=52u/热点核）

| 管线 | busy min–max | 均值 | idle 均值 | 归因 |
|---|---|---|---|---|
| aic_cube | 17.7–18.6% | 18.2% | 81.8% | 等对侧核 + 等 flag 节流（C-id1 **104.30µs = 55% wall**，pacing 承重）；cube_wait cflt 24.2% |
| aic_mte1 | 17.2–18.4% | 17.8% | 82.2% | 链上游 |
| aic_mte2 | 27.1–31.2% | 29.2% | 70.8% | 等 flag 节流（前跑 ~1 步 → 队列浅）；busy 期 2.31 TB/s（L2 64%） |
| aic_mte3 | 0% | 0% | 100% | 无工作 |
| aic_fixpipe | 26.1–29.1% | 27.7% | 72.3% | 等 cube；busy 期 ~2.44 TB/s（ws_s/ws_o fp32 128MB） |
| **aic_scalar** | 79.6–86.7% | **83.7%** | 16.3% | **全表最忙管线**——阻塞主体 = id1 104.30（等 V 释放 ws_s）+ id5 53.52 + id2 32.39；真实发射工作量远小于 busy 值（wait 计入 busy 口径 §5 头注） |
| aiv_vec | 62.6–65.8% | **64.4%** | 35.6% | **pacer 本体**（busy 2.35µs/u ≈ iter5 实测 2.38）；idle = **id3 104.28µs（pacing 镜像）** + id4 28.95 + id0 9.25 + vector_stall 39.41 |
| aiv_mte2 | 19.8–25.3% | 22.5% | 77.5% | 等数据（ws_s/ws_o FIX 就绪，链在 C） |
| aiv_mte3 | 13.0–16.5% | 14.9% | 85.1% | 等 flag 节流（ws_p 单 token 放行；mte3_wait cflt 59.6%） |
| aiv_scalar | 52.1–59.3% | 56.1% | 43.9% | id3/id4/id0 阻塞 + 发射 |

### R5 dense_expert decode（wall 145.89；窗口 142.13/143.01；7 task×16kv/热点核）

| 管线 | busy min–max | 均值 | idle 均值 | 归因 |
|---|---|---|---|---|
| aic_cube | 8.5–10.0% | 9.1% | 90.9% | **等数据搬运**（K/V 流；cube_wait cflt **86.9%**） |
| aic_mte1 | 25.9–30.9% | 28.0% | 72.0% | 等 mte2 供 L1（**scalar_mte1_stall 127.76µs = 87.5% wall**——MTE1 队列饿在 MTE2 后端） |
| **aic_mte2** | **79.3–95.7%** | **86.2%** | 13.8% | **墙本体**：busy 期达/略超规格接口标称速率（§4.2，108% nominal）；mte2_wait cflt 86.6% = 队列满阻塞 |
| aic_mte3 | 0% | 0% | 100% | 无工作 |
| aic_fixpipe | 24.9–31.1% | 27.3% | 72.7% | ws_s/ws_o [16,128] 小块（32MB），非瓶颈 |
| aic_scalar | 69.1–83.8% | 75.2% | 24.8% | 阻塞主体 = mte1_stall（等数据搬运）；wait-id 小（id1 2.49 + id2 6.02） |
| aiv_vec | 30.4–35.9% | 32.7% | 67.3% | **等对侧核**（V-id0 47.80 + V-id4 58.67 = 106.5µs = 73% wall——等 C 的 K/V 数据流，根因=带宽，iter3 结论复证） |
| aiv_mte2 | 20.2–28.5% | 23.9% | 76.1% | 等数据（ws_s 就绪链在 C） |
| aiv_mte3 | 9.2–12.4% | 10.6% | 89.4% | ws_p 小块（4KB/u） |
| aiv_scalar | 25.3–32.3% | 28.1% | 71.9% | id0/id4 阻塞 + vector_stall 33.85 |

### R6 g2stack_flat（wall 28.02；窗口 23.48/25.07；5u/热点核；F≈10.6µs=38%）

| 管线 | busy min–max | 均值 | idle 均值 | 归因 |
|---|---|---|---|---|
| aic_cube | 9.5–12.4% | 11.7% | 88.3% | 等对侧核（id2 4.70）+ pacing（id1 8.07） |
| aic_mte1 | 9.4–13.3% | 11.9% | 88.1% | 链上游 |
| aic_mte2 | 20.0–28.9% | 25.1% | 74.9% | 等 flag 节流（pacing） |
| aic_mte3 | 0% | 0% | 100% | 无工作 |
| aic_fixpipe | 14.1–19.8% | 17.9% | 82.1% | 等 cube |
| aic_scalar | 43.9–67.9% | 57.1% | 42.9% | id1 8.07 + id5 9.71 + id2 4.70 阻塞 |
| aiv_vec | 38.6–49.8% | **47.3%** | 52.7% | pacer；F 段占比高（浅 task）；idle = id3 9.45 + id0/id4 10.35 |
| aiv_mte2 | 10.4–16.5% | 13.4% | 86.6% | 等数据（无 mask——noncausal） |
| aiv_mte3 | 9.0–13.9% | 11.9% | 88.1% | 等 flag 节流（mte3_wait cflt 49.9%） |
| aiv_scalar | 34.7–51.0% | 42.3% | 57.7% | id3/id0/id4 阻塞 |

**跨工况管线要点**：(1) aic_mte3 恒 0%（该 kernel 家族 C 侧无 UB→GM 路径——Output 全走 V 侧）；
(2) 唯一贴屋顶的管线 = R5 的 aic_mte2（86.2%/热点核 95.7%）；(3) prefill 系最忙管线是 scalar（R4 83.7%）
与 vec（R3/R4 62–64%）——**计算屋顶（cube）从未接近（≤18.2%）**；(4) V 侧 mte3 的 cflt wait
（42.5–74.6%）是 pacing 单 token 节流的直接队列签名。

---

## 6. 主要访存瓶颈导致原因（定量归因，落到具体数据通路）

每条标注【实测】/【模型】/【推断】。

### 6.1 prefill flat 域（R2/R3/R4/R6；bench c1/c2/c3/c4/c7/c18/c20）——根因 (a) ws 往返 + pacing 链

**通路**：`G1 mma → L0C → FIX → GM(ws_s fp32) → L2 → MTE2-V → UB → softmax → MTE3-V → GM(ws_p fp16)
→ L2 → MTE2-C → L1(p_l1) → MTE1 → L0A → G2 mma → L0C → FIX → GM(ws_o fp32) → L2 → MTE2-V → UB →
AO 累加 → MTE3-V → GM(Output)`——每 u 三次 GM(L2) 往返、两跳方向切换（C→V→C→V）。

1. **字节倍率**【实测+模型】：每 u ws 字节 320KB（ws_s 64 写+64 读、ws_o 64 写+64 读、ws_p 32 写+32 读）
   vs 有效操作数 96KB（Q 摊薄+K+V）= **3.3×**；占 L2 侧总流量 R4 77% / R3 62% / R6 71% / R2 62%。
2. **HBM 份额 ≈ 0**【实测】：L2 读命中 85–92%、写命中 31–87%，ws ≤13MB ≪ 168MB L2 常驻 →
   HBM 侧仅 234–514 GB/s（R4 最低 234，因 ws 全 L2 吸收）。**该架构税不烧 HBM 字节，烧的是
   L2 带宽占用（64.5% 屋顶）与每跳延迟**。
3. **延迟入链 = pacing 承重**【实测，iter4/5 结论复证】：R4 的 C-id1（G1 头等待 V 释放 ws_s）104.30µs
   与 V-id3（SM 头等待 C 消费 ws_p）104.28µs 互为镜像，各 = wall 55%；单 release token 把 C 前跑压到
   ~1 步（iter4 实证：解除节流 → 生产门全面膨胀 +10.6..+29.7% 劣化）。**ws 往返的两跳 GM/L2 延迟
   （每跳 ~0.5–1µs 量级，推断）就是这条 pacing 链的物理设定项**：稳态节拍 P₂≈3.05–3.11µs/u vs
   vec busy 2.35–2.38µs/u，残余 0.67–0.76µs/u（22%）= 交接/往返伪影（iter5 exposure/kv 1.181 同口径）。
4. **scalar 发射界**【实测+推断】：R4 aic_scalar 83.7%（热点核 86.7%）为全表最忙——扣除 wait 阻塞后
   真实发射占比未分离（口径两可性 §5 头注），但即便按 iter4 "wait 计入 busy" 口径，scalar 管线也
   同时承担 flag 链发射 + 每 u ~5 个 macro 块的循环控制；**kv2 展平后循环体 5 发射块（kv1 为 3）——
   scalar 已接近发射带宽，任何新增每 u 标量工作都会直接吃掉 wall**（对 §7 方案的约束）。
5. **K/V 重载**【实测+模型】：stacked 域 K/V 每 u 重载（无跨任务 L1 驻留，§4.1），R4 重载倍率 8×
   （唯一 8MB → L1 流量 64MB）、R3 16×（唯一 1MB → 16MB）；L2 命中吸收后 HBM 代价 ≈ 唯一字节
   ——**不是本域瓶颈**（与 decode 的本质区别）。

**归因结论**：flat 域主要访存瓶颈 = **ws GM(L2) 往返的"延迟×握手"税**（不是 HBM 带宽税）：它同时
(a) 占用 62–77% 的 L2 侧流量、(b) 通过 C↔V 单 token pacing 链把两跳延迟变成稳态节拍的 22%、
(c) 让 scalar/vec 而非 cube 成为最忙管线。回收路径 = 消除往返本身（P3 FIX→UB）或压缩 V 侧 busy（P5）。

### 6.2 decode 域（R5；bench c11–c17 类）——根因 (b) K/V 深流式装载 = HBM 墙【实测，双重闭合】

1. **物理字节下限**【模型，代码核实】：G_inner=4 头共享装载、每 (bz,kg) 恰读一次 → K/V 唯一字节
   = 16×8×2048×128×2B×2 = **134.2MB，零冗余可削**（与 GM_to_L1 实测 136.75MB 差 +Q/ws_p 小包 ✓）。
2. **速率贴屋顶**【实测】：134.2MB / 145.89µs = **920 GB/s = 规格 MTE2 聚合屋顶（1.056 TB/s）87% =
   经验 HBM 混合屋顶（~980 GB/s）94%**；热点核 aic_mte2 95.7%、busy 期折算速率 ~1140 GB/s ≥ 规格标称
   （108%，§4.2 口径注）；per-core 物理 K/V 流 50.3 GB/s = 规格 52.8 的 95%。
3. **下游全线等搬运**【实测】：scalar_mte1_stall 127.76µs（87.5% wall，MTE1 饿于 MTE2 后端）、
   cube_wait cflt 86.9%、V-id0+id4 = 106.5µs（73% wall，等 C 的数据流）——iter3 "V 空闲 72% 根因是
   带宽非握手" 结论在 iter6 数据下原样成立。
4. **剩余空间定量**【模型】：理论地板 = 134.2MB ÷ 屋顶 ∈ [127.1µs（规格 1.056 TB/s）, 136.9µs（经验
   HBM 侧 0.98 TB/s）] → 剩余 **6–13%**（需 100% 接口效率 + 7/6 任务完全均衡）；对齐 split-K（kv 分半
   到两核）也只能逼近该地板且引入跨核归约 ws+flag 新链。
   **判定：不可为清单**（§7.4）——bench decode 组 speedup 0.97–1.06 已是全 bench 最好档。

### 6.3 aux 纯搬运税——根因 (c)【实测】

1. **e2e 份额**：同会话 23.3–61.1%（§2.2），eval 跨会话下界口径 18.6–46.2%——**prefill 各组 e2e 的
   最大单项可回收池**（超过 main kernel 内所有剩余杠杆之和）。
2. **病灶分解**：transpose_out = 任务网格饥饿（1–32 任务 vs 40 向量核；R1 极端 = 1 任务 1 核，
   PipeUtilization 同 block 双 vid 行 0%/46% 直接证实）+ 每 kernel launch 地板 2.5–3.8µs；
   transpose_qkv 相对健康（12–320 任务，R4 达 1475 GB/s）。
3. **可达速率证据**【实测】：同 kernel 家族在足够并行度下（tqkv R4 blk=160）达 L2 屋顶 41%；
   tout R4（blk16）31%——**重铺 tile 后 tout 各工况按 0.9–1.2 TB/s + launch 地板折算：R1 12.0→~3.5µs、
   R2 13.2→~7、R3 24.1→~16、R6 22.1→~9、R4 29.3→~24（模型外推，中-高置信）**。
4. **可被 main 吸收性**【代码事实】：decode 路径已是 BSND 直供直写（gqa.py:742-754 "the four host-side
   BSND<->BHSD transposes of the removed"、:966-968 s-major 连续盒写）；prefill 的 Output 写出盒
   `[half,D]` 改 BSND 即为行 stride N_q·D 的 strided 盒——与 transpose_out 内部完全同款的 MTE3
   strided 写（速率已被 tout 实测证明 0.28–1.12 TB/s）。**吸收路径存在，无新硬件原语需求**。

### 6.4 R1 小 shape（per-head）——launch 地板 + 单波串行链【实测+sim 佐证】

核外 launch/排空 3.3–3.8µs（32–37% wall，iter3 逐样本）+ 核内 4 相位单任务串行链（3 次 C↔V 交接
暴露 ~1.02µs sim 时基，`msprof_sim/OPPROF_20261016163102_PRXNKARSFXKBQCTP` trace，iter1 §3.3 提取，
本轮复用未重采）+ 4/20 核闲置。**main kernel 本体无低风险杠杆（iter3 不立项维持）；但 R1 类 e2e 的
61% 是 aux——P1/P2 对该域收益最大（−31%/−44% e2e 模型外推）**。

### 6.5 假设判定总表（任务书候选 (a)–(e)）

| 假设 | 判定 | 关键数据 | 标注 |
|---|---|---|---|
| (a) ws GM 往返倍率 | **成立——prefill 主根因**（形态 = L2 带宽占用 + pacing 延迟，非 HBM 字节） | 3.3×/u、62–77% L2 流量、id1/id3 各 104µs（R4）、HBM 侧仅 234 GB/s | 实测+模型 |
| (b) K/V 重复装载 | stacked 域：重载 8–16× 但 L2 吸收 → **非瓶颈**；decode：零重载但唯一 134MB → **HBM 墙成立（87–94% 屋顶）** | §6.1-5 / §6.2 | 实测+模型 |
| (c) aux 纯搬运税 | **成立——e2e 口径最大单项**（23–61%）；transpose_out 网格饥饿为主病灶；可被 main 吸收（decode 先例） | §2.2 / §6.3 | 实测 |
| (d) mask 与 Output 路径 | mask：causal fp32 64KB/u = GM→UB 的 33%（R2/R3 同为 4/12、16/48MB），L2 热、无 HBM 税——**次级项**（P6）；Output：连续盒写无病，BSND 直写=P2 的载体 | §4.1 对账（R2 +4MB/R3 +16MB = mask ✓） | 实测+模型 |
| (e) L2 命中结构 | 分层清晰：prefill HBM 侧 197–514 GB/s（ws/K-V 重载全被 L2 吸收）vs decode 920 GB/s（首触流式无复用）；计数器口径两处伪影已校准披露 | §3.3 / §4.3 | 实测+推断（口径校准） |

---

## 7. 下一步优化方案清单（按 收益×置信度/风险 排序，供用户决策 iter7——不替用户决定实施范围）

> 通用纪律（全项适用）：变体先落 `perf_tuning/` 独立文件（Gate 0）；精度 = 逐位一致优先（调度/布局类
> 改动的预期性质）+ `test_gqa.py --level all` 全绿；性能 = 同会话交替 ≥3+3、>3% 噪声阈值；**动
> flag/槽深/步序/V 侧节拍的方案必须过 wait-id 判别门（iter4 pacing 承重教训）**；本地 event 单 token
> 纪律（iter5 死锁教训）；NaN/Inf fill 纪律（iter2 教训）；gate 外 shape 逐位不变。

### P1【推荐 iter7 主项·低风险】transpose_out 重铺 tile（aux 并行度修复，host-only）

- **目标域**：全部 prefill e2e（bench c1–c4/c7/c18/c20 + R1 类默认 shape）；main kernel 零触碰。
- **机制**：`out_plan()` 的 (bm,bn) 从"Nq≥48?16:8 × 128"改为**按任务数铺满向量核**——约束保持
  （bn 整除 S、bm 尾行 guard 已有、UB 预算 [bn,R]≤192KB）：目标 `B·ceil(Nq/bm)·ceil(S/bn) ≥ 40`
  （≥1 波），候选规则如 bm→8/4（尾 guard 兜底）、bn→min(128, S/ceil(40/(B·ceil(Nq/bm))))。
  kernel 本体（`_transpose_v1`）不动。
- **预期收益**（模型外推 §6.3-3，中-高置信——锚 = 同家族 tout-R4/tqkv-R3 实测速率 + launch 地板）：
  tout 时长 R1 12.0→~3.5µs / R2 13.2→~7 / R3 24.1→~16 / R6 22.1→~9 / R4 29.3→~24；
  **e2e：R1 −31% / R6 −21% / R2 −13% / R3 −8% / R4 −4%**。bench prefill 组 speedup 折算
  ~0.45→0.52–0.58（推断，e2e 弹性 = aux 份额×回收率）。
- **风险**：低。纯 host 参数；输出**逐位一致**（同数据同盒序重排任务，无算术变化）；最坏 = 小 bn 的
  strided 写盒变小致 mte3 效率下降（Gate 2 实测兜底，参数可回退）。
- **工作量**：小（~15 行 host + 变体 runner 复用 iter5 基础设施）。
- **Gate**：G0 变体独立编译+逐位 runner（L0 全集含非对齐 S 走 fallback 路径不变）；G1 免（无算术）；
  G2 msprof B 链同会话 A/B ≥3+3（e2e Σ 与 tout 单核）；G3 接入后全套件 + gate 外逐位 + 派发臂 probe。

### P2【推荐 iter7 次主项·中风险】main kernel BSND 直写（消除 transpose_out）

- **目标域**：flat kv1 + flat kv2 两 kernel（覆盖 bench c1/c2/c3/c4/c7/c18/c20）；per-head/g2stack/
  prefill_expert 回退路径保留 transpose_out 不动。
- **机制**：V 侧 FIN 的 Output 盒从 BHSD 连续（`Output[bz,hq,行窗,:]`）改为 BSND strided 盒
  （`Out[bz, 行窗, hq, :]`，行 stride N_q·D）——与 transpose_out 内核同款 MTE3 strided 写；adapter
  （`gqa()` prefill 分支）对声明 BSND 输出的 kernel 跳过 transpose_out。数据通路变化：省掉
  Output 的 UB→GM(BHSD)→GM 读→UB→GM(BSND) 整段往返 = 每 task 2×32KB 字节 + 一次 launch。
- **预期收益**（模型外推，中置信）：e2e 上界 = 删 tout 全部时长（R1 −44%/R2 −27%/R3 −23%/R6 −36%/
  R4 −12%）；折减项 = V 流内 strided 写变慢 + FIN 节拍扰动（V 是 pacer！mte3 工作量 +32KB/task，
  busy 期按 tout 实测 strided 速率 0.3–1.1 TB/s 折算 +0.03–0.1µs/task）→ **现实区间 R1 −35..−44% /
  R2 −20..−27% / R3 −16..−23% / R6 −28..−36% / R4 −9..−12%**。与 P1 互斥收益（P2 落地则 P1 只对
  回退路径有意义）——**建议 P1 先行（确定性收益立即入袋），P2 作为 iter7/8 的结构项**。
- **风险**：中。触 V 侧 FIN（adopted kernel 的写回段）：mte3 队列加深可能移动 id3/id4 节拍
  （**wait-id 判别门必过**：G2 采集含 wait-id 矩阵复采，若 V-id3 上涨吃掉收益 → 回退）；算术不变
  → 逐位一致可判；NaN/Inf fill 纪律不受触（FIN 算术不动）。strided MTE3 盒在本仓库已被 tout 长期
  验证，无新原语。
- **工作量**：中（2 kernel × FIN 段 ~30 行 + adapter 分支 + 变体 runner/audit 改造）。
- **Gate**：G0 变体独立编译 + 逐位（10 用例含 NaN/Inf、mc=2、混合 mc、padrow/padcol——iter5 Gate 0
  集直接复用）；G2 同会话双臂交替 ≥3+3 + e2e 链 B 式采集 + wait-id 判别；G3 全套件 + gate 外 8 shape
  逐位 + 派发臂 probe。

### P3【结构探针·高收益潜力/高不确定性】FIX→UB 直达消除 ws_s/ws_o 往返（对根因 (a) 的直接攻击）

- **目标域**：flat kv1/kv2（先 R4 域验证）。
- **机制**：G1/G2 的 L0C 结果不再 FIX→GM(ws)→L2→MTE2-V→UB，改 **FIX→UB 直写对侧 vid 的 io_buf**
  （同核 AIC↔AIV，硬件路径存在性证据：per-head/prefill_g2 的 `T.copy(acc_s_l0c, acc_s_ub)` AUTO
  CV-combine 已在产线使用，gqa.py:234/279/531/583；ini `l0_c_to_ub_rate=256B/cyc` + intrinsic
  `fix_pipe_l0c2out`）。收益面 = 删 ws_s/ws_o 的 256MB L2 流量（R4 的 61%）+ 删两组 cross-sem
  （SEM_WS1/WS3）→ C↔V 交接从"GM 两跳+跨核 flag"变"同核 FIX 队列+本地 flag"，pacing 链的物理
  延迟设定项被拆除。ws_p（V→C 方向，喂 L0A 必需过 L1）保留 GM 路径（R4 64MB）。
- **预期收益**（低置信外推）：稳态节拍残余 0.67–0.76µs/u 若收回一半 → R4 main −10..−18%、R3 −8..−15%；
  若 V 侧 io_buf 双缓冲不足或 FIX→UB 带宽低于预期 → 收益归零。**必须 Gate 0 可行性探针先行**：
  expert 模式（P28_EXPERT_PASS_CONFIGS，CV_COMBINE 关闭）下手写 `T.copy(l0c, ub_frag)` 是否可编译/
  正确——**本仓库 expert 路径无先例，可行性未验证（本轮最大不确定项）**。
- **风险**：高。门控拓扑重构（6 cross-sem 删 4）= iter4 教训的高危区（pacing 承重——拆除后 C 前跑
  深度失控的镜像风险）；UB 预算 +32KB/vid（io_buf 需双缓冲接 FIX 流：现 ~154KB → ~186KB/192KB，
  **贴顶，可能挤压**）；跨 scope（C 写 V 的 UB）的 tilelang 语义需探针确认。
- **工作量**：探针小（独立 30 行小 kernel）；若可行，正式变体大（≈iter5 C2 量级的一整轮）。
- **Gate**：G0 = 探针 kernel（mma→FIX→UB→vec 读→写回比对）+ 若过：变体全 Gate 0–3 + wait-id 判别
  （预期签名：id1/id3 归零、wall 降幅 ≈ Σ残余；若等待转移到生产门且 wall 不动 → iter3 §3-3 判别
  逻辑复用，结论"V 吞吐下限"同样有价值）。
- **决策建议**：iter7 只跑 G0 探针（廉价、不动 gqa.py），可行性结论供 iter8 立项。

### P4【P2 之后的延伸·高风险】transpose_qkv 消除（main kernel BSND 直读）

- **目标域**：flat kv1/kv2；K/V/Q 装载盒从 BHSD 连续改 BSND strided（K/V：`K[b, kv窗, kg, :]` 行
  stride N_kv·D；Q：`Q[b, 行窗, hq, :]`）。decode 直读先例存在但形态不同（S=1 的 [G_inner,D] 连续小盒，
  gqa.py:754）；prefill 的 **strided GM 盒 → Nz 布局 L1**（k_l1 为 make_zn_layout）组合在本仓库无先例，
  MTE2 的 ND-strided→NZ 重排能力需探针。
- **预期收益**（模型外推，中置信）：删 tqkv 全部（e2e 再 −17%/−20%/−18%/−11%/−18% 于 P2 之后）+
  K/V strided 读若变慢会部分回吐（main 内 mte2 busy +？未测）。
- **风险**：高（Nz+strided 未验证；触 C 侧装载链 = K/V flag 协议）；**工作量**：大。
- **Gate**：G0 探针（strided→Nz L1 拷贝微基准）先于一切；后续同 P2 四门。
- **决策建议**：排在 P2 落地验证之后（同一段 dispatch/adapter 基建复用），不与 P2 同轮（归因隔离）。

### P5【廉价补充项】kv2 域 V 侧 busy 窄化（iter5 next_hint 1 的落地）

- **机制**：flat_kv2 的 SM(t,kv0) 头部 neg_sm **整缓冲** cap 重填窄化为仅 prv 槽（设计文档 B.4-3 的
  原方案，实现时保守取了整缓冲）；+0.18µs/kv 的 flat 形态 vec 开销（iter5 实测归因）预计收回 1/3–1/2。
- **预期收益**：R4/c7 main −1..−2%（vec busy 2.35→~2.28µs/u，pacer 直接传导）——低但确定性强。
- **风险**：低-中——**触 NaN 纪律**（prv 槽 cap 重填是 iter2 NaN 泄漏修复的承重件；窄化必须证明
  cur 槽被 reduce_max 全覆写后才被读，设计文档 B.4-3 已论证值等价）；boundary_sparse NaN/Inf 双 kv
  块用例必跑（iter5 Gate 0 集含）。
- **工作量**：小（~5 行 + audit 脚本同步）。**Gate**：G0 逐位（10 用例）→ G2 同会话 ≥3+3 →G3。
- 可与 P1 同轮（正交：P1 在 aux、P5 在 main V 侧），但归因需各自 vs BASE 独立臂。

### P6【低优先·低置信】causal mask 流量消除（in-kernel 再生）

- **机制**：causal 域的 mask fp32 GM 读（64KB/u = R3 GM→UB 的 33%）改为 V 侧 in-kernel 生成
  （`j > i+(S_kv−S)` 比较 + select 填 -1e30，vec ALU 工作 ~0.05–0.1µs/u）；或 mask 降位宽（fp16
  不可行——-1e30 超出 fp16 表示域，**否决**；int8/bool + kernel 内放大 = 可行但触 mask 语义）。
- **预期收益**：L2 流量 −12.5..−33%（causal 工况）；**wall 收益 −0..−4%（低置信）**——mask 装载
  与 V 流重叠度高（aiv_mte2 仅 21% busy），且 vec 是 pacer、再生工作落在 vec 上可能负收益。
- **风险**：中（mask 语义 = 精度承重件，_BIG_NEG 支配性注释 gqa.py:5464-5471；生成式必须逐位等价
  现有 mask 读路径的输出）。**工作量**：中。
- **决策建议**：仅当 P1/P2 落地后 causal 域（c1/c3/c8/c10）仍落后时再评估；本轮仅记录。

### 7.4 不可为清单（roofline 已贴屋顶/触地板项，写明距离）

| 项 | 距屋顶/地板 | 判定依据 |
|---|---|---|
| **decode（c11–c17）主体** | HBM 物理读 920 GB/s = 规格 MTE2 屋顶 **87%** / 经验混合屋顶 **94%**；理论地板 127.1–136.9µs vs 现 145.89 = 剩余 **6–13%**（需 100% 接口效率+完全任务均衡，split-K 引入跨核归约新链） | §6.2 双重闭合【实测】；bench 0.97–1.06 已最好档 |
| **F 的 launch 段** | 核外 2.5–3.8µs = 框架/驱动地板（六工况一致，代码不可触及） | iter3 §4 逐样本实测 |
| **kv1 flat 环深/步序** | iter4 双臂实测证伪（ns4 +10.6..+29.7%、G2 前移 +9.6..+24.0%）；pacing 承重 | iter4 candidate JSON |
| **R1 main kernel 专家化** | 上限 ~17%（scalar 链减半假设）且 net-negative 前科语境；R1 e2e 的真瓶颈在 aux（61%）→ 由 P1/P2 覆盖 | iter3 §5.2 |
| **ci 粗化** | iter3 R6 A/B 实测证伪（批式 = g2stack 慢 14.7%） | iter3 §3-4 |
| **cube 算力方向的一切投入** | 全工况 cube ≤18.2%、TFLOPS eff ≤16.8% 屋顶——算力富余 5.4–40×，任何"提高算力利用"的方案都不在关键路径 | §4.3【实测】 |

---

## 8. 未验证项 / 推断标注汇总（如实报告）

1. **P1–P6 的全部收益数字为模型外推**（锚 = 本轮/iter3/iter5 实测速率与时长），非实测；采纳判定
   一律以 Gate 2 同会话 A/B 为准（纪律延续）。
2. **P3 的 expert 模式 FIX→UB 可行性未验证**（AUTO CV-combine 先例 ≠ expert 手写路径可用）——
   本报告将其定位为"探针先行"项，可行性结论不在本轮承诺范围。
3. **P4 的 strided GM→Nz L1 MTE2 能力未验证**（decode 先例是连续小盒，形态不同）。
4. **计数器口径三处推断**（§3.3）：+1×Q 计数怪癖的机制（Nz 目标盒足迹/双盒重复计数）未从文档或
   生成代码核实；L2 写计数 512B 单位为两工况校准推断；decode "61% 读命中"的计数伪影解释为推断
   （物理口径 920 GB/s 不依赖该解释——首触无复用的字节下限是代码事实）。
5. **scalar_ratio 是否含 wait 阻塞**的两可性延续 iter1 §6-5（未从文档确认）；§5/§6 的归因以
   wait-id/stall 直测列为准，该两可不改变任何结论方向（两种口径下 pacing 链都是 R4 的节拍设定者）。
6. **B 矩阵 aux 时长为 metrics 注入上界口径**（eval 无注入下界低 20–60%，§2.2 脚注）；aux 份额结论
   在两口径下一致，但 P1/P2 的收益绝对值在 bench harness 下预期偏向下界口径。
7. **pad kernel（_pad_kv）税未测**（本矩阵 6 工况全对齐无 pad launch；非对齐 shape 的 e2e 含 pad，
   量级预期同 tout 类——未验证）。
8. **vec 屋顶的 op 类差异**：add 类实测 4.39 TFLOPS 不能直接换算 GQA softmax 的 exp/max/fp32 混合流
   的屋顶；本报告 vec-bound 判定全部用 busy%（同类相对比较），未用绝对 TFLOPS 断言。
9. **R1 的 wait-id（id0/id1）语义**为 AUTO 生成 flag 的方向性推断（iter3 §7-2 延续，无源码级映射）。
10. **tilelang gemm 微基准的 cube_ratio 行内部不自洽**（§3.2）——该 bench 已整体弃用于屋顶，不影响
    任何结论；但提示多波次（blk≫核数）场景的 PipeUtilization 行口径需按 TaskDur 归一理解。
11. **eval harness L2 冷热态**（CacheClean 每迭代刷 L2）与本会话单发采集的 ws 热态差异：iter3 §7-8
    假设延续（推断，未做 controlled 实验）——bench 分数折算时 aux/main 都偏冷态，方向上 P1/P2 收益
    在 bench 下可能更大（aux 读的是冷 GM）。

## 9. 本轮查阅的 skills / 参考

- `tilelang-perf-optimization/SKILL.md` + `references/performance-antipatterns.md`（roofline 方法学、
  反模式对照——iter3 §9 结论复核无新增违例；本轮 analysis_only 无新代码）
- 平台规格：`/home/developer/Ascend/cann-9.2.0/aarch64-linux/data/platform_config/Ascend910_9362.ini`
  （cube_freq/核数/L2/ddr_rate/l2_rate/容量全表 §3.2）；CANN RAG 两次检索（HBM 规格、msprof 计数器
  语义）均空结果——计数器口径改由解析模型对账校准（§3.3，如实标注推断级）
- iter1–iter5 遗产：`flat_pipeline_analysis.md`（§2–§3 管线分解/容量包络/ws 流量结构）、
  `cv_overlap_resurvey_iter3.md`（工况矩阵/wait-id 语义与遥测方法/F 拆解）、`c2_design_iter5.md`
  （flat_kv2 结构/门控承重结论/vec busy 归因）、`perf_log.md` iter1–5、`baseline_iter3.json`（结构模板）
- 源码只读核实：`gqa.py`（emit_qload/emit_g1/emit_g2 数据通路 :3538–3646、_transpose_qkv/_transpose_v1/
  _pad_kv 任务网格 :4855–5228、out_plan :5371、gqa() 适配链 :5574–5666、per-head L0C→UB 先例 :234/279/531/583、
  decode BSND 直供注释 :742–754/966–968）
- 评测参照（只读）：`reports/cann_final_eval_20261018_113143.*`、`reports/prof_data/level4/gqa/{1,2,3,4,11}`
  op_statistic.csv（跨会话标注）
- R1 sim trace（复用未重采）：`examples/gqa/msprof_sim/OPPROF_20261016163102_PRXNKARSFXKBQCTP`

## 10. 产出物与测量完整性

- 本报告：`examples/gqa/perf_tuning/roofline_analysis_iter6.md`
- 基线记录：`perf_tuning/baseline_iter6.json`；轮次日志：`perf_log.md`（iter6 条目）
- 解析模型工具（新增）：`perf_tuning/model_roofline_iter6.py`（输出快照 `/tmp/opencode/iter6_model_out.txt`）
- 原始数据：44 目录 = 遗留 40（A18+B12+C10，其中 C_copy_s{1,2}/C_copyl2_s{1,2} 判定无效保留披露）
  + 本会话补采 4（C_copy_s{3,4}、C_copyl2_s{3,4}）；解析汇总 `/tmp/opencode/iter6_parse_{A,B,C}.txt`
- 工具修复（perf_tuning/ 内，允许范围）：`parse_msprof_iter6.py`（None 格式化 bug）、
  `microbench_peak_iter6.py`（copy_kernel 切片 bug，注释标记 s1/s2 无效原因）
- 完整性：gqa.py / test_gqa.py md5 开工/收工核验一致（bd470e07… / 3e2d0888…）；
  `.orchestrator_state.json` 未读写；补采批后 npu-smi 健康（Health OK、无进程残留）、
  msprof/探针残留进程 0；全部采集 run 1650/1650 MHz 额定频率；A/B 矩阵伪影过滤 0 行命中。
