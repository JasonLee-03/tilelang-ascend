# MHA 算子优化历史（逐会话详细摘录）

> 采集对象：`examples/mha/`（MHA forward, BSHD）
> 采集来源：`perf_tuning/optimization_log.md`（1708 行，逐迭代门禁全文）、`perf_tuning/perf_log.md`（603 行，跨轮摘要）、`.orchestrator_state.json`（阶段/评分权威）、`perf_tuning/final_report.md`
> 采集范围：会话 1 ~ 会话 12（含迭代 24-26 纯测量轮），截至 2026-10-04
> 记录方式：每会话给出「背景/目标 → 基线 → 优化点逐项（状态·机制·性能）→ 会话结论」

---

## 0. 全局总览

### 0.1 cann-bench 权威成绩链

| 阶段 | 报告 | 综合分 | 平均加速比 | 通过 |
|------|------|--------|-----------|------|
| 原始 | — | 54.18 | 0.13x | 20/20 |
| 会话 1（Developer 轨收口） | `cann_final_eval_20260918_220619` | 61.76 | 0.42x | 20/20 |
| 会话 2（Developer 轨收口） | `cann_final_eval_20260919_154250` | 65.05 | 0.55x | 20/20 |
| 会话 3（Expert rev3 首轮） | `cann_final_eval_20260920_005958` | 68.50 | 0.7017x | 20/20 |
| 会话 4（Expert rev3） | `cann_final_eval_20260920_141424` | 70.21 | 0.7813x | 20/20 |
| 会话 5（Expert rev3） | — | 71.40 | 0.8721x | 20/20 |
| 会话 6（Expert rev3 收口） | `cann_final_eval_20260922_100108` | 71.70 | 0.9054x | 20/20 |
| rev4 S1（attempt1） | `cann_final_eval_20260922_172117` | 71.79 | 0.9108x | 20/20 |
| 会话 8（flat 全域采纳） | `cann_final_eval_20260930_083840` | **72.59** | **0.96x** | 20/20 |

会话 9 / 10 / 11 终态与 72.59/0.96x 权威评测量体同体，不重跑（同体先例适用）；会话 12 进行中。

### 0.2 三条调优轨

- **Developer 轨**（会话 1-2，共 15 迭代）：`alloc_shared`+自动同步+全 pass，做常规调优直到触结构边界（65.05/0.55x）。
- **Expert rev3 轨**（会话 3-6，共 13 迭代）：切成 Expert 手动流水线，达成 M1/M2 平均线（71.70/0.9054x）。
- **rev4 轨**（会话 7 起）：Form B→A mma 化 + 大 prefill 联合专项，最终 72.59/0.96x。

### 0.3 计时与判定规则（全程继承）

- 噪声带 **3%**；计时唯一依据 = msprof timeline `op_summary` Task Duration 中位数（warmup 5 + iters 20）；cann-bench 为准权威。
- 优化规则：**>3% 提升采纳 / 无变化回滚 / 性能下降回滚 / 精度失败必回滚 / 运行失败回滚**。
- 中止条款②：**连续 3 次无提升（<3%）** 触发收口。
- 精度红线四项（全程不可破）：① 双舍入栅格链（r16 qk 落点）；② 单块预归一化特化；③ sumexp+1e-30 全掩码行保护；④ 右下角掩码链语义。
- camodel **仅作机制定性**，禁作收益定量（会话 6 判例后成纪律）。

---

## 会话 1（perf_session=1，2026-09-18，Developer 轨首轮；迭代 1-9）

**背景/目标**：`mha.py`（Developer 模式 threads=1 cv_1_1，T.serial KV 循环），目标 cann-bench 20 cases（910b2，baseline = CANN FusedInferAttentionScore）。初始 0.13x。

**算子类型**：CV 融合混合型（`KERNEL_TYPE_MIX_AIC_1_1`）。瓶颈 = **sync + vector-instruction-issue**（AIC 98.5% 等待 AIV；aiv_scalar 49.6% ≈ aiv_vec 56.6%；cube 利用率仅 2%）。

**优化点（Part A）与实施**：

| # | 优化点 | 迭代 | 状态 | msprof 关键收益 |
|---|--------|------|------|----------------|
| 1 | causal KV 运行期裁剪（`T.serial(0, eff_kv_iters)`，右下角对齐公式） | 1 | ✅ 采纳 | case7 -46.6%（1.87x） |
| 2 | softmax/rescale 行循环 broadcast 2D 化（专用 scratch `m_i_2d/r_2d`） | 2 | ✅ 采纳 | case7 -32.3% / case10 -50.9% / case14 -41.4% |
| 7 | Fill+Add→Copy（`acc_s_ub_` 退役，释放 16KB UB） | 3 | ✅ 采纳（#3 使能项，单独中性） | — |
| 3 | 掩码链 2D 化 + row_thresh hoist（uint8 bit-packed mask） | 3 | ✅ 采纳 | case7 -41.3% / case1 -39.2% |
| 6 | decode 窄块 block_M=16（S<=16 编译期分支） | 4 | ✅ 采纳 | case10 -31.6% / case14 -25.5% |
| 5 | Fixed Core（`core_num=min(block_num,20)` 连续 chunk 任务循环） | 5 | ✅ 采纳 | case14 -18.7% / case1 -17.0% / workspace 114MB→1.1MB |
| 4 | 全可见迭代掩码运行期跳过 | 6 | ✅ 采纳 | case7 -12.1% / case16 -9.1% |
| 8d | decode 分支 block_N=128（迭代减半） | 7 | ✅ 采纳 | case10 -38.7% / case14 -34.0% / case16 -36.1% |
| 8 | 全局 bN=128 | — | ❌ UB 不可行（prefill 242KB>192KB） | — |
| 9 | K/V 手动双缓冲预取 | 8 | ❌ Developer 边界不可行（AUTO_SYNC 下 codegen 零重叠） | — |

**关键途中缺陷修复**：
- 迭代 2 首版把 broadcast scratch 与跨核 L0C→UB copy 目标（`acc_o_ub`）复用 → AUTO 同步 pass 按 buffer 依赖重排标志，l0 5/10 FAIL；修复 = 新增专用 scratch。**入库教训：broadcast scratch 不得与跨核 copy 目标别名**。
- 迭代 5 首版 `T.serial(my_start,my_end)` 双运行期界触发 `AUTO_CV_COMBINE` range 推导崩溃；修复 = count 形式 `for t in T.serial(my_count): task = my_start + t`。

**收口（迭代 9）**：`--level all` exit 0（L0 10/10 / L1 19/19 / L2 12/12 / Boundary 6/6 / 哨兵 4/4）；cann-bench 61.76（+7.58），平均 0.42x（3.2x），20/20 全提升 1.84x~6.24x，无一回退。
**遗留**：剩余差距主体 = Developer 结构性同步成本（AUTO_SYNC 逐 op barrier + 每迭代 3 段 CV workspace GM 往返）。

---

## 会话 2（perf_session=2，2026-09-19，Developer 轨追加预算；迭代 10-15）

**目标**：保持 Developer，深挖剩余空间（会话 1 终态 65.05 前为比较基线）。

**优化点**：

| # | 优化点 | 迭代 | 状态 | 关键收益 |
|---|--------|------|------|----------|
| 10a | decode bN=256（S≤16∧D≤128∧S_kv>128） | 10 | ✅ 采纳 | case10 -28.8% / case16 -26.3% |
| 10b | D=256 decode bN=256（GEMM1 kL0Size=64） | 11 | ✅ 采纳 | case15 -22.7% / case18 -22.8% |
| 11 | prefill bN=128（全 D，MEMORY_PLANNING 实测放行零手术） | 12 | ✅ 采纳 | case7 -30.2% / case20 -30.2% / case3 -25.1% / case4 -19.3% |
| 13 | decode bN=512（D≤128∧S_kv>256，GEMM1 kL0Size=32） | 13 | ✅ 采纳 | case10 -21.8%（累计 -44.7%）/ case16 -8.5% |
| 12 | 反模式全 11 项复扫 | 14 | ✅ 完成（零新可修复项） | — |
| — | decode bN=512 @D=256 | — | ❌ L1 536KB>511.75KB | — |
| — | prefill bN=256 / bM=128 | — | ❌ UB naive 336/448KB | — |
| — | heads-as-M / split-KV / CV workspace 削减 / 批量 softmax | — | ❌ 证据性放弃（数学不适用 / 收益为负 / AUTO pass 不可源码干预 / Expert 专属） | — |

**关键发现**：gemm_v0 `transpose_B` 路径 `nTile==N` 不分片，GEMM1 B-tile 受 L0B 64KB 预算硬限 → 导出 bN/kL0Size 档位规则（后续会话反复复用）。

**收口（迭代 15）**：`--level all` exit 0；cann-bench 65.05（+3.29），平均 0.55x；19/20 case 提升。
**遗留**：tile 已触 L0B/L1/UB 三面硬限；剩余 case7/8/9 差距 = AUTO_SYNC barrier 链 + CV workspace GM 往返，需 **Expert 化**（设计模式变更，建议独立立项）。

---

## 会话 3（perf_session=3，2026-09-19 起，Expert rev3 轨首轮；r3-1 ~ r3-5）

**背景**：切入 Expert 版 `mha.py`（rev3-attempt1 终态，cann-bench 67.88 / 0.68x）。目标：平均加速比逼近 1.0x；M1 ≥0.75x 且 case7/8/9 ≥0.4x；**case14 恢复 ≥0.79x 为首位主攻**。

**基线**（msprof median us）：case1 12.91 / case7 928.00 / case9 1398.35 / case10 40.53 / case11 349.63 / case14 746.08 / case16 763.42 / case20 31.30。

**瓶颈归因**：case14 = 纯协议延迟（pipe 全空转<40%，任务间零重叠）；case9 = `barrier_all` 串行化（AIC pipe busy 和 ≈95% 零重叠）；case7 = 双瓶颈（C 侧 k/v L1 单缓冲 + V 侧 vector 近饱和）。

**优化点**：

| # | 优化点 | 迭代 | 状态 | 关键结果 |
|---|--------|------|------|----------|
| R3-1 | Form B single_kv_block 任务批处理（8 任务/协议轮） | r3-1 | ✅ 采纳 | case14 -28.9%（0.71→0.995x）/ case16 -15.3% |
| R3-3 | w1 dtype 瘦身 fp32→dtype（r16 落点迁 fixpipe） | r3-2 | ❌ 回滚（+1.0% 无变化） | RNE 位级等价知识入库（11/11 BITEXACT） |
| R3-2 | Form A k/v L1 双缓冲 | r3-3 | ❌ 回滚（+0.5% 无变化） | 3D `[2,bN,D]` L1 annotate 机制入库（10/10 BITEXACT） |
| R3-4 | CROSS_INTERVAL 1→2（判决性实验） | r3-4 | ❌ 回滚（噪声带内且 case9/10 偏劣） | SEM 会合延迟假设证伪 |
| R3-6 | Form B barrier 细化 | — | ⏸ 移交会话 4 | case9 AIC pipe busy 和 ≈95% 归因就绪 |
| R3-7 | Form B→A D256 mma 化 | — | ⏸ 条件触发 | 工程量大（K/N-split + L0B 共池顶格） |
| R3-8 | 扁平 (task,kv) 批处理 | — | ⏸ 移交 | case14 收益模型已验证 |
| R3-10 | case14 vid1 空半区减负 | — | ❌ 证据性放弃（并入 R3-1 解决） | — |

**收口（r3-5）**：触发中止条款②（连续 3 次无提升）；`--level all` exit 0；cann-bench 68.50（+0.62），平均 0.7017x。**case14 0.71→0.995x（回退修复，超额）**，case12/13/16/17 同路径收益。
**归因知识**：case7 wall≈920-930us 的约束 = **每 KV 块迭代的固定结构成本**，单点调度/流量/频率手段已穷尽。

---

## 会话 4（perf_session=4，2026-09-20，Expert rev3 第二轮；r4-1 ~ r4-5）

**目标**：冲击里程碑 M1（平均 ≥0.75x 且 case9 ≥0.4x）。主攻两条移交线索：R3-6（=R4-1）+ R3-8（=R4-3）。§10.3-⑤ Form B→A mma 化**设计级禁止实施**（仅积累证据）。

**优化点**：

| # | 优化点 | 迭代 | 状态 | 关键结果 |
|---|--------|------|------|----------|
| R4-1 | Form B `barrier_all` 纯移除（15 处，依赖 gemm_v0 prologue/epilogue 自平衡 flag + SEM pipe 作用域） | r4-1 | ✅ **采纳** | case9 -26.0%（0.240→0.331x）/ case14 -28.7%（0.995→1.40x）/ case16 -12.6% / case11 -10.7% / case10 -11.4% |
| R4-2a | prefill D256 GEMM1 kL0Size=256 单 K-step | r4-2 | ❌ 回滚（劣化 +3.7%） | kNumSteps=1 关闭 L0 ping-pong；单 mma k=256 ≡ 链式 2×k=128 位级等价 |
| R4-2b | Form B prefill D256 k/v L1 双缓冲（预取式零新增 flag） | r4-3 | ❌ 回滚（+0.9% 无变化） | 全 pipe 读数零变化（MTE2 暴露被吸收非关键路径） |
| R4-3 | 扁平 (task,kv) 批处理 decode 推广（=R3-8） | r4-4 | ❌ 回滚（6/6 噪声带） | **协议税模型修正**：#R4-1 后每轮 SEM 开销 ~3.6us→~0.25us，轮次削减收益归零 |
| R4-4 | #R3-7 Form B→A D256 mma 化 | — | 🚫 设计级禁实施 | case9 三证据链积累完成 |
| R4-5 | case7/8 bN=256 变体 | — | ⏸ 未启动（条款②触发） | — |

**收口（r4-5）**：触发条款②；终态 = #R4-1 采纳态（md5 `e80f7f65`，947 行）；cann-bench 70.21（+1.71），平均 0.7813x。
**M1 判定：部分达成**——平均 ≥0.75x ✅；case9 ≥0.4x ❌（0.324，规则内下限）；破局唯一路径 = §10.3-⑤ 设计级 mma 化。

---

## 会话 5（perf_session=5，2026-09-21，Expert rev3 第三轮；r5-1 ~ r5-3）

**触发**：用户基于 case1 camodel 流水图观察「C 核等待明显长于 V 核；V 核 MTE 与 VECTOR 串行；VECTOR 计算过碎片化」，mode=implement。验收锚定单块 Form A 类（case1/6/19/20）。

**复核确认**：C FLOWCTRL 83.18% / V 59.74%；MTE∩VECTOR 仅 5.30%（几乎完全串行）；VECTOR 切成 11 段（2 段真实 softmax）。**根因 = `batch_iters=1` 使 NUM_STAGES=8 批处理与 io_buf[2] 双槽空转，每 task 严格串行四相位链**。

**新证据（历史未记录）**：同 (b,h) 相邻 task 的 K/V 同字节冗余重读（core0 task1/task2 K 源地址相同；case6 s_blocks=8 时 7/8 K、7/8 V 冗余）。

**优化点**：

| # | 优化点 | 迭代 | 状态 | 关键结果 |
|---|--------|------|------|----------|
| R5-1 | Form A single_kv_block 任务批处理（V 侧分派门 1 行 + C 侧补镜像 117 行；V 侧批分支复用 Form B 已验证） | r5-1 | ✅ **采纳** | **case1 -11.28% / case6 -44.40% / case19 -34.80% / case20 -40.48%**；7/7 守卫噪声带内 + 12/12 守卫 IR 逐字节 |
| R5-3 | 同 (b,h) 相邻槽 K/V L1 驻留（跳过同字节冗余重读） | r5-2 | ❌ **精度 FAIL 回滚** | 非确定性数据损坏；4 实验收窄，根因未定案 |
| R5-2 | 单块 O 直写 FIX(L0C)→Output | — | ⏸ 移交会话 6 | — |

**#R5-3 证伪链**：① 探针 6/6 DIFF；② 损坏非确定性（同输入两次 3/64 vs 24/64 坏 slice）；③ 恒真守卫 3/3 BITEXACT（机制无误）；④ 谓词/生成码/flag 位置/L1 别名全部正确；⑤ K-only 变体真实数据下同样失败（case20 全零是假阴性）。**结论：手动 pipe-pair flag 协议下「省略一次冗余 GM→L1 DMA」本身不安全**。

**机制层验证（camodel 新验收件）**：MTE||VECTOR 重叠 5.30%→20.93%（io_buf 双槽真正激活）；两段 softmax 间空隙 2.684→0.582us（-78.3%）。

**收口（r5-3）**：预算纪律收口（非强制耗尽）；终态 = #R5-1 采纳态（md5 `3d56c5dd`，1086 行）；cann-bench 71.40 / 0.8721x。

---

## 会话 6（perf_session=6，2026-09-21，Expert rev3 第四轮；r6-1 ~ r6-3）

**触发**：用户提案「L0/L1 显式多缓冲 + MTE2 预取 + 20 核深流水」→ analysis 可行性分析 → 授权执行序 Iter-1/2/3。

**优化点**：

| # | 优化点 | 迭代 | 状态 | 关键结果 |
|---|--------|------|------|----------|
| R6-1（=R5-2） | 单块 O 直写 FIX(L0C)→Output（删 V phase-2 + WS3 SEM 对 + w3 fp32 GM 往返） | r6-1 | ✅ **采纳** | case1 -10.66% / case6 -18.05% / case19 -8.23% / case20 -22.55% |
| R6-2 | Form A 单块批处理 q/k/v L1 双槽 + 批内/批间预取 | r6-2 | ❌ 回滚（4/4 噪声带且方向不一致） | camodel 与真机方向背离 → **camodel 禁作定量** |
| R6-3 | #R5-3 根因算子外最小复现调查 | r6-3 | 归档（根因未定案，不上 kernel） | 单核 + 20 核两级均不复现 |

**#R6-1 前置门**：L0C→BSHD 4D 运行期行切片独立探针 8/8（尾块 valid_m=26/30/62、bf16、sentinel 无漏写）。**机制**：V wall -13.6%、V FLOWCTRL -46.6%、V MTE2 -27.3%、VECTOR 段 10→8；C wall +3.4%（接管 Output FIX 成为新 pacing 侧）。

**#R6-2 判决性知识**：camodel C wall -19% 与真机 +0.57% **方向背离**；真机 wall = GM/L2 总流量墙。**camodel 用途限定 = 机制是否激活的定性判定，禁止收益定量**。

**收口**：cann-bench 71.70 / 0.9054x（M2 平均线达成）。终态 = `bbfcbcaf`（1113 行）。

---

## 会话 7（perf_session=7，2026-09-22，rev4 轨首轮；r7-1 ~ r7-3）

**对象**：rev4-attempt1（1593 行，md5 `ee5bf9c9`，71.79 / 0.9108x）。结构 = 三分派（form_a / form_b_mma 手动 K/N-split mma 流水 / form_b_gemm_v0 decode 兜底禁触）。

**S7-Step 0 case9 新鲜归因**：S1 mma 化后 `aic_scalar_mte1_stall` 739.4→412.2us（-44%，靶心命中），但 `aic_wait_id2` +114%（瓶颈迁移）。case9 新墙 = **C↔V 四相位耦合链延迟 + D256 流量强度**，而非单管吞吐。

**优化点**：

| # | 优化点 | 迭代 | 状态 | 关键结果 |
|---|--------|------|------|----------|
| R7-1 | w1 dtype 瘦身重试（scope=form_b_mma 专属） | r7-1 | ❌ 回滚（+0.31%/+1.16% 零响应） | 机制激活但 wall 不动，被 C mte1_stall +33us 1:1 吸收；§10.3-① 升级为**双 régime 证据性放弃** |
| R7-2 | Form B-mma k/v L1 双槽（3D ZN × 列段 transpose 新组合） | r7-2 | ❌ 回滚（+0.35%/+2.58% 零响应） | **scalar_mte1_stall 412→188us（-54%）靶心命中**，但 C wait_id2 324→545us（+68%）1:1 吸收；**§10.3-③ 五环证伪链闭合** |
| R7-3 | 剩余候选处置（评估归档） | r7-3 | 无 kernel 变更 | flat 批处理移交会话 8；CROSS_INTERVAL=2 / w3 plan-a 不启动 |

**核心机制知识**：r7-1（加速 V-P1）+ r7-2（加速 C-G1）**双侧 A/B 完备** → case9 wall = **四相位每槽交接延迟链**（批深仅 4.5 槽摊薄不足），**字节/速度类手段数学上无杠杆**（吸收量 = 节省量，1:1）。

**收口补录**：发现并处置**默认缓存污染事件**（Stage-2 时代毒条目与当前源同 cache key 冲突）；清除后锚 10/10 BITEXACT。

---

## 会话 8（perf_session=8，2026-09-29，rev4 第二轮；大 prefill 联合专项）

**触发**：用户立项「方案 A 联合专项」——case7/8/9 的 now→B 层差距（占 case9 差距 63%）= 四相位每槽交接延迟链。规则内唯一杠杆包 = ① flat(task,kv) 连续令牌流 + ② C 侧 L0B 4 槽旋转 + ③ V 侧批相干化。

**S8 优化点**：

| # | 优化点 | 迭代 | 状态 | 关键结果 |
|---|--------|------|------|----------|
| R8-1 | Stage-1 ② C 侧 L0B 4×16KB 槽旋转（同取向 v4ksp 形态） | r8-1 | ❌ 回滚（case9 +0.23% 零响应） | 判据(a) FAIL：mte1_stall 434→425us（-2%）；**新硬知识：mte1_stall = 四相位链伴生症状而非杠杆**；② 族证据性关闭 |
| R8-2 | Stage-3 ① flat(task,kv) 连续令牌流 prefill 扩展（form_b_mma generic 域，顺序游走状态机映射运行期非均匀 eff_kv） | r8-2 | ✅ **采纳** | case9 -7.96% / case4 -20.72%；靶心 wait_id5 87.6→2.2us（-97.5%） |
| R8-3 | ① flat 域扩展（form_a 多块 = case7/8/3/2/5） | r8-3 | ✅ **采纳** | **case7 -7.50% / case3 -21.60% / case2 -33.08% / case5 -18.13%**；wait_id5 -95.2% |
| R8-4 | Stage-4 联合裁决 + Step D 条件评估 + 收口 | r8-4 | 无 kernel 结构变更 | ① 全域落地；② 证据性关闭；③ 移交会话 9 |

**关键方法学（ORDER-PLAN 数据驱动修订）**：② 机制激活失败后，将 ③（单侧吞吐类，必被吸收）与 ①（交接结构删除类）执行序对调，先做 ①。**结果验证「吸收律边界」**：单侧吞吐类手段被 1:1 吸收，**交接结构删除类（flat）不被吸收**（wait_id5/id2/id4 三处同降 + 真实工作管零变化 + wall -8%）。

**flat 收益模型**：收益 = f(批深空缺率, 任务边界密度)，全序吻合：case2(-33.1) > case3(-21.6) > case4(-20.7) > case5(-18.1) > case9(-8.1) > case7(-7.4) > case8(-6.3)。

**Step C 联合态全 20 case**：7/7 ① 目标域收益 -6.26%~-33.08%；13/13 守卫带内零回退。**Step D**：② 复活被否决；③ VW=2 宽通道作为评估对象移交会话 9。
**收口**：cann-bench **72.59 / 0.96x**（历史最佳）；终态 md5 `dda3ad08`（1926 行）。

---

## 会话 9（perf_session=9，2026-09-30，rev4 轨；case9 双路径专项）

**任务**：路径 1 = ③ V 宽通道 VW=2 主攻（判决性判据 = V.vec 显著降 ∧ C.wait_id2 同步降 ∧ wall 降）；路径 2 = fbm prefill bN=128→256 仅 scoping 探针。

**优化点与归档**：

| # | 内容 | 迭代 | 状态 | 关键结果 |
|---|------|------|------|----------|
| R9-1 | ③ V 宽通道 VW=2（form_a flat D=128 域） | r9-1 | ❌ 回滚（性能下降） | case7 **+12.41%** / case8 +16.04% / case3 +14.35%；**三签名全 FAIL 反向**（V.vec +21.8%）；**③ 族证据性关闭** |
| — | 路径 2 scoping（fbm bN=256） | r9-2 | 归档（无 kernel 变更） | GATE1-5 实编 5/5 PASS（L0B/L0C 双顶格）；EV 修正 -12~-19%（条件化，GATE-6 L2 命中未测） |
| R9-3 | 收口（回滚验证 + 落档） | r9-3 | 无 kernel 变更 | IR 16/16 逐字节还原；净变更 = 0 |

**重要机制知识**：③ 事件税侧签名全兑现（v_scalar -11.7%/v_stall -35.6%/v_mte3 -27.9%/wait_id0 -44.2%），但 **V.vec 主管道 +21.8% 吞没全部节省并反转 wall**。**V-paced 减速侧对照实验 → pacing 管因果系数 ≈1 双向确证（新硬知识）**。

**探针工程五条工具级硬知识**：`[N,1]` 2D copy 的 32B 块粒度陷阱 / 观测点 emit 竞态 / None-assign segfault / alloc_var init 约束 / 别名重命名漂移。

**Stage 2 不实施**（数据驱动裁决）：三签名第一签名机制级证伪跨 form 成立；case9 域 EV≤0。

---

## 会话 10（perf_session=10，2026-09-30 起，rev4 轨；路径 2 实施）

**优化点**：

| # | 内容 | 迭代 | 状态 | 结果 |
|---|------|------|------|------|
| R10-1 | 路径 2 fbm prefill bN=128→256 结构加宽（T2S 同取向 L0B + F5 L0C + K/V L1 半区拆分 + 谓词迁移） | r10-1 | ❌ **结构性不可行**（非性能规则） | 全形状族 HANG（aicore timeout 507014） |
| R10-2 | Stage-4 收口（判决宣告 + 回滚 + 三重验证 + 落档） | r10-2 | 无 kernel 变更 | 净变更 = 0 |

**判决链（15 实测变体毒化矩阵）**：bisect 族 3 + E 系 9（E1/E2/E4/E5/E6/E13/E15/E16/E17）+ E21 + E22 + T1 = 15 变体。**T1（v4ksp 全架构推广）3/3 HANG ⇒ 预登记穷尽判据命中**。

**新硬知识**：
- **bN 上界定理**：手动 mma 流水的 bN 上界 = **B 操作数工作集（含 K-split 链全部在途 chunk）≤ L0B 64KB**。bN=256∧D=256 ⇒ K^T/V 各 128KB = 2× 超限 ⇒ 相位内 L0B 重用不可避免。
- **L0B 双重毒化机制**：(a) L0B 相位内覆写 @~50ns 边际挂死（E16/E13）；(b) K128-mma 相位深度 ≥3 或 B 二次读取挂死（E17/E21/E15/PROD）。
- **scoping 盲区**：GATE-2 验证的是槽容量合法性，不验证工作集相位内重用的编排合法性。
- **E 系毒化矩阵方法学**：单因子删除/保留变体矩阵定位挂死类缺陷。

**回滚三重验证**：md5 `dda3ad08` 一字不差 / 1926 行 / ast+ruff + IR 16/16 逐字节还原 + 轻量复验全绿。
**版图收束**：① 已兑现、② 证据性关闭、③ 规则内永久关闭、路径 2 结构性不可行 ⇒ case9 M1 线 0.4 与大 prefill ≥1.0x 均规则内不可达，仅剩架构级方案 C。

---

## 迭代 24-26（2026-10-02，mode=perf_measure，GATE 0 纯测量轮；零 kernel 变更）

**背景**：在无新候选时，用「消融上界探针 + 计数器差分法」对剩余候选做预筛，把 EV 上界在实现前钉死。

| 轮 | 内容 | 结论 |
|----|------|------|
| Iter 24 | GATE 0 预筛：case9/case7 pacing 判定 + Q prologue 消融（`q_l1` 双缓冲攻击 ramp） | **GATE 2 ❌ NO-GO**（EV 上界 [−1.48%,+0.33%] ≪ 3%；case7 显著劣化 p=0.025）；GATE 1 铺垫对象消失 |
| Iter 25 | GATE 0b 交接延迟链消融上界 | 总 EV 上界 **case9 −9.358% / case7 −13.426%**；缺口① CONDITIONAL GO / 缺口③ 延迟侧 NO-GO |
| Iter 26 | GATE 0c：GATE A 静态预算审计 + G1 工作消除分量实测 | **GATE A FAIL**（缺口① 关闭，L0B 128KB=2.00× 超限）；**G1 工作消除侧 case9 −0.598% / case7 −0.042%，绝对上界臂显著劣化** ⇒ EV 版图彻底闭合 |

**关键硬知识**：
- **吸收律定量化**：`wait_idN` 之和把可回收 wall **高估 6.8~9.4×**（延迟侧）/ **64.8~844×**（工作消除侧）。
- **CV-over-GM workspace 往返物理上是 L2 往返而非 DRAM 往返**（C 写命中 98.26%、V 读命中 99.9989%、miss 0.55）⇒「省 1032 MB / 46% L2 流量」类宣称价值 ≈0，且写侧不可单独移除。
- **pacing 身份分域**：case9 = C↔V 四相位交接延迟链；case7 = AIV VEC（77.5%，不可削减真实计算）。两域禁止混用/外推。
- **方法学资产**：消融上界探针 = 零生产风险、≈25 分钟/候选的预筛门；工作消除类探针与交接类探针判读规律不同，须分章。

**最终裁决**：规则内杠杆穷尽；架构级方案 C 合计上界至多 1.10×/1.16×（非会话 10 所称 2.3-3.4x，覆盖率仅 13-24%，被否证），不值得立项。建议置 SUCCESS 收尾。

---

## 会话 11（perf_session=11，2026-10-04；decode D=256 duty 墙专项）

**用户立项**：S2 优先决策路径（decode D=256，case15/18 的 1.45× 上界空间）。

**优化点**：

| # | 内容 | 迭代 | 状态 | 关键结果 |
|---|------|------|------|----------|
| R11-1 | decode D=256 K 侧 L1 双缓冲（3D `[2,256,256]` + gemm_v0 运行期切片 + 4 预取点 + 零新增 flag） | r11-1 | ❌ 回滚（性能下降） | case15 +2.00% / case18 +2.70%；**机制激活但 duty 签名不动**（68.45→68.89%） |
| R11-2 | S3 三道 API 能力门 scratch 验证 | r11-2 | 能力门判决（零生产变更） | 门① PASS / **门② FAIL** / 门③ PASS ⇒ S3 关闭 |
| R11-3 | Stage-4 收口 | r11-3 | 无 kernel 变更 | 净变更 = 0 |

**核心机制知识**：
- **gemm_v0 稳态守恒律**：prologue 管级 MTE2_MTE1 全排空 + 每 tile `PipeBarrier<PIPE_M>` 标量阻塞 + 顺序发射 ⇒ **每槽 wall = MTE2 字节时间 + gemm 标量阻塞，与 L1 槽数无关**；31% duty 缺口规则内不可压缩，gemm_v0 调用边界 = 硬墙。
- **预取流重排税**：带宽墙域（busy 已贴布局上限）的 MTE2 流重排具负外部性（-2.7% 引擎效率），吞没边界先导收益。
- **needClear=(dst_offset==0) codegen 规则** + V 侧 select 掩码安全性证明。
- **region 切片损坏第 4 例证**（plan-a C-region / 附.3 T.view / r10 mma B / 本轮 gemm_v0 A+B）。

**decode D=256 duty 墙规则内路径穷尽**：S1（推演负）/ S2（实测负）/ S3（门② FAIL + 布局税）/ mma 化（bN 上界定理永久否决）/ vector 路径重构（超 Stage-3 规则范畴）⇒ 维持 `dda3ad08` 即规则内最优。**consecutive_no_improvement = 3 ⇒ 中止条款②形式触发**。

---

## 会话 12（perf_session=12，2026-10-04 起；A10 框架级改造）

**用户明示授权框架级改动**：「先尝试 A10，若 A10 没有收益则尝试 A9」。

**r12-1（Phase A，analysis 轮，零生产变更）**：

- **A10 设计** = gemm_v0 模板级 per-buffer 同步改造：
  - `common.h` 追加第 9 默认模板参 `int32_t kMTE2Sync = -1`，prologue `if constexpr (kMTE2Sync >= 0)` 分支——opt-in 时仅 `WaitFlag<MTE2_MTE1>(id)`，不再自 set 全排空（打破 #R11-1 守恒律的唯一规则外杠杆）。
  - `ascend.py` 追加 `mte2_sync=None` kwarg，仅非 None 时在模板串尾追加；**零 C++ 改动、零 rebuild**。
  - `mha.py` kv_db 分支重建：**FIFO 发射序=消费序不变量 + pre-call 位预取**（发射点前移），parity ids `SIG_KDB0/1` + `SIG_VP`，严格 1:1 flag 交替；Q 载入位重排（任务尾预载 Q(t+1)）。
- **六项可行性核查**：A12-1 模板细读 / A12-2 事件 id 约束（**核内 API 无「≤15 次」条款**，该条款属跨核 `CrossCoreSetFlag`；id 0-5 可用，6/7 禁用）/ A12-3 plumbing 选型（通道 b）/ A12-4 生效链（本地 JIT 即时生效；pip 包需重打包+重装）/ A12-5 回归面（73 文件/186 调用点 + 4 专用 UT）/ A12-6 FIFO 稳态时序设计。
- **GO/NO-GO 判决：GO**（进入 Phase B+C）。默认路径逐字节不变有机器证明（16/17 IR 守卫）；未验证三项（核内旗标 1-bit vs 计数器语义 / 重排税量级 / G2 相 duty 兑现比例）由 r12-2 msprof 实测裁决。
- **硬门**：16/17 IR 守卫（默认路径字节同一性机器证明）+ smoke 先行（会话 10 HANG 纪律）+ **duty 签名** (a)：wall 收益必须伴 duty 68.5%→85%+；watch mte2 busy 速率重排税（r11 -2.7%）。
- **r12-1 = 全局 iter30（分析轮，consecutive_no_improvement 计数不变 = 3）。**

> 后续：r12-2 = 框架最小编辑 + kv_db 重建；若 duty 部分兑现，第二臂 = bN=128 k/v dual-DB（任务书条款）。截至采集时点尚未执行。

---

## 附：跨会话可复用机制知识汇总

| 类别 | 知识 | 来源 |
|------|------|------|
| 缓存/别名 | broadcast scratch 不得与跨核 L0C→UB copy 目标别名 | 会话 1 迭代 2 |
| L0 预算 | gemm_v0 `transpose_B` `nTile==N` 不分片；GEMM1 B-tile ≤ L0B 64KB | 会话 2 |
| 位级等价 | fixpipe RNE ≡ V 侧 vcast RNE（→ w1 瘦身免探针） | 会话 3 / 7 |
| L1 机制 | 3D `[2,bN,D]` L1 annotate 前导槽位维可用且位级安全 | 会话 3 |
| 同步 | Form B `barrier_all` 可纯移除（gemm_v0 自平衡边界） | 会话 4 |
| 协议税 | #R4-1 后每轮 SEM 开销 ~3.6us→~0.25us（收益不可叠加） | 会话 4 |
| 度量纪律 | camodel 仅机制定性，禁收益定量 | 会话 6 |
| 吸收律 | 单侧吞吐类被 1:1 吸收；交接结构删除类不被吸收 | 会话 7 / 8 |
| 伴生症状 | mte1_stall 是四相位链伴生症状，非杠杆 | 会话 8 |
| 收益模型 | flat 收益 = f(批深空缺率, 任务边界密度) | 会话 8 |
| pacing | case9 = 交接延迟链；case7 = AIV VEC；两域禁混用 | 迭代 24-26 |
| 硬件硬限 | 手动 mma 流水 bN 上界 = B 操作数工作集 ≤ L0B 64KB | 会话 10 |
| L2 事实 | CV-over-GM workspace 往返是 L2 往返非 DRAM；写侧不可单独移除 | 迭代 26 |
| 守恒律 | gemm_v0 每槽 wall 与 L1 槽数无关，调用边界是硬墙 | 会话 11 |
| region | region 切片偏移 = 朴素行主序 ≠ ZN fractal，会静默损坏 | 会话 9 / 10 / 11 |
