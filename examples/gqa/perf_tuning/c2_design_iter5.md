# C2 设计文档 — flat 双流展平扩展到 kv_iters==2 uniform-noncausal（Stage 3 · round 2 · iteration 5）

- 日期：2026-10-17
- 对象域：`use_g2_stack ∧ kv_iters==2 ∧ noncausal（kv_useful≡2 恒定）∧ stacked_block_num ≥ 2*P28_NUM_CORES(=40)`
  —— bench **c4**（B4 S256 Skv256 Nq64 Nkv8 bf16 noncausal，stacked=512，mc=26/25）与
  **c7**（B2 Sq128 Skv256 Nq32 Nkv8 bf16 noncausal，stacked=64，mc=4/3；shape 已从
  `.agents/skills/cann-bench/tasks/level4/gqa/cases.yaml` case_id 7 核实：Q=[2,128,32,128]、
  K/V=[2,256,8,128]、attrs is_causal:false、note "cross" —— iter3 §7-6 的未实测项本轮补上）。
  两工况 use_mask=False（Skv 对齐 + noncausal，gqa.py:3686 判据）。causal / 变长 kv==2 一律 gate 外走
  shipped g2stack 零改动。
- 基线核验（开工时）：gqa.py md5 `cbce74410fee5f615489d0391b3ea620`、test_gqa.py md5
  `3e2d08881cb7b6f36d5c89b2fd37a146` —— 与任务书一致 ✓。
- 硬约束来源：iter4 机制结论（candidate_iter4_C1/C3.json framework_findings）——**ns=2 槽深、
  release 信号量单 token、C 步序 G1-first、门控原样继承 shipped flat kv==1**；C2 唯一收益来源 =
  删除任务边界的批次 drain/refill + 批次粒度互替串行 → u 粒度稳态跨边界连续，**不叠加任何其他
  结构改动**（不加深环、不重排步序）。

---

## Phase A —— 收益重估与 go/no-go

### A.1 数据与方法

全部取自 iter3 已落盘的单会话 msprof 数据（`msprof_op_iter3_r4_s{1..3}`、`r6_flat_a{1..3}`、
`r6_g2s_b{1..3}`；提取脚本 `phaseA_iter5_extract.py`，本轮无新采集）。PipeUtilization.csv 行结构
本轮核实：每 block 一行 `cube0`（aic_* + C 侧 wait-id）+ 两行 `vector*`（aiv_* + V 侧 wait-id）——
iter3 的 "max across cores" 口径实为 max across rows，热点核归属近似（iter3 §7-1）延续。
OpBasicInfo 无频率列（freq=NA）——iter3 报告的 1650MHz 额定频率核验来自其采集会话记录，本轮
Phase A 直接复用其 TaskDur 数值（锚点 207.85µs 与任务书一致）。

**R4（g2stack ns=2，热点核 26 task = 52 kv，3 样本均值）**：

| 量 | 值 | 每 kv | 每 task |
|---|---|---|---|
| TaskDur | 207.85µs [206.94–209.24] | 3.997 | 7.994 |
| V 窗口（hot_aiv） | 204.90µs | 3.940 | 7.881 |
| V vec busy | 114.40µs（3 样本逐位稳定 114.398） | **2.200** | 4.400 |
| **V 暴露（窗口−vec，下界口径）** | **90.50µs** | **1.740** | **3.481** |
| V-id4（AO 批等 C 的 G2 批） | 105.50µs | 2.029 | 4.058 |
| V-id0（SM 批等 C 的 G1 批） | 22.35µs | 0.430 | 0.860 |
| V-id3（等 C 消费 ws_p，release 侧） | 2.02µs | 0.039 | 0.078 |
| C-id2（G2 批等 V 的 SM 批） | 109.47µs | 2.105 | 4.210 |
| C-id5（G2 批等 V 的 AO 批释放 ws_o） | 46.87µs | 0.901 | 1.803 |
| C-id1（等 V 释放 ws_s） | 0.00µs（热点核） | 0 | 0 |
| C busy：cube/mte1/mte2/fix/scalar | 34.2/31.6/58.2/55.7/100.1µs | 0.66/0.61/1.12/1.07/1.92 | — |

（wait 列为上界口径：scalar 阻塞期间已发射的 MTE/vec 指令继续执行，且各等待段互相重叠——
V 侧 wait 列合计 129.9µs > 暴露 90.5µs 即为证；iter3 §7-3 纪律延续，**不用 wait 列求和做收益模型**。）

**R6 flat kv==1（同会话对照，热点核 5 task = 5 u，3 样本均值）**：

| 量 | flat（shipped，采纳实现） | g2stack ns=1 强制回退臂 |
|---|---|---|
| TaskDur | 27.53µs | 32.27µs |
| V 窗口 | 24.54µs | 29.22µs |
| vec busy | 13.45µs = **2.691/u** | 12.01µs = 2.402/u |
| 边际周期 P（iter3 §4 两点拟合 fill≈7.8） | **P6 = 3.35µs/u** | ≈4.5–4.6µs/u（fill_g2s 6–7 推断） |
| **边际残余（P − vec busy）** | **0.66µs/u** | ≈2.1–2.2µs/u |

### A.2 iter4 门归属修正后的重估

iter3 的 −13..−21% 所依据的模型是 "wall = busy + 往返残余，消除残余 1:1 转化为 wall"。iter4
证伪了该模型在**深环/步序杠杆**上的用法（release 停顿与有效 pipe 排空重叠；解除 1-token 节流 →
生产门延迟全面膨胀）。但该证伪**不适用于 C2 的机制类**，理由：

1. C2 不动门控深度：6 个 cross-sem 的 init token、每 u set/wait 计数、FIFO 有效门
   （G1(u)←SM(u−1)、SM(u)←G2(u−1)、G2(u)←AO(u−1)，1-token 紧一拍语义）与 shipped flat kv==1
   **逐信号量同构**——iter4 实证过该门控是承重结构，C2 原样继承（§B.3 计数表）。
2. C2 攻击的等待不是 pacing 停顿，而是**批次粒度的数据生产等待**：R4 的 V-id4（4.06µs/task）与
   C-id2（4.21µs/task）是 "AO 批必须等整个 G2 批 / G2 批必须等整个 SM 批" 的硬数据依赖空转
   （release 侧 V-id3≈0.08/task、C-id1≈0 证明 pacing 门在 g2stack 里根本不紧——没有 iter4 那种
   可解除的节流）。展平后这些等待被 2-beat 前瞻错拍吸收，转化为 u 粒度交接残余。
3. 该机制类在同一家族已两次实测兑现：iter2 ns=1→flat（R3 −23.3%，采纳）；iter3 R6 同会话 A/B
   flat vs g2stack-ns1 同 shape −14.7%。C2 = 同一变换施于 ns=2 批次（其批内交错已回收 ns=1
   损失的一半，故预期幅度介于两者之间偏小）。

**每 task 边界暴露份额分离**（任务书要求；结构归属为模型推断，锚定实测列值）：
g2stack ns=2 每 task 的 V 暴露 3.48µs 中——
- (i) **批内互替主项**：V-id4 类（AO 批 ← G2 批数据就绪）窗口差份额 ≈ 3.48 × 105.5/(105.5+22.3+2.0)
  ≈ **2.8µs/task（1.4/kv，≈81%）**——展平主攻击面；
- (ii) **边界相邻项**：V-id0 类（SM 批 ← C 的 Qload+G1 批 refill 链）≈ **0.6µs/task（0.3/kv，≈17%）**
  + FIN/fill 的 V 流串行位（在 vec busy 内，≈0.5–0.7µs/task，推断）——展平后 SM(u) 与 Qload
  错拍、FIN 移到 SM(t+1,kv0) 之后（§B.2），串行位保留但不再阻塞下一任务的 kv0 softmax；
- (iii) release 侧 ≈ 0.08µs/task（≈2%）——无可收，也不动。
边界份额**可分辨且非小量**：即便只收 (ii)（0.3/kv + FIN 位减半），52kv × ~0.45µs ≈ 23µs ≈ −11%，
仍 >3% 阈值；(i) 是主要上行空间。

### A.3 修正后预期收益区间

flat-kv2 边际周期 P₂ = vec busy（R4 实测 2.20µs/u，flat 形态开销 +0.05..0.15：kv1 域实测
flat−g2s vec 差 +0.29/u 系 FIN 每拍一次，kv2 的 FIN 每 2 拍一次，减半折算，推断）+ 边际残余
（R6 flat noncausal 实测 **0.66µs/u**；上界取 causal 域 0.95–1.15 的 0.90——R4 为 noncausal，
且 kv2 的 FIN/Qload 摊薄每 u 开销低于 kv1）：

| 口径 | P₂ (µs/u) | fill'+launch (µs) | wall = 52·P₂+F' | vs 207.85 |
|---|---|---|---|---|
| 乐观 | 2.90 | 10.9 | 161.7 | **−22%** |
| **中枢** | **3.05** | **12.4** | **171.0** | **−17.7%** |
| 保守 | 3.35（=R6 实测 P6 平移） | 13.9 | 188.1 | **−9.5%** |
| 悲观（残余膨胀 1.7×，kv1 域从未观测到） | 3.65 | 14.0 | 203.8 | ≈持平（break-even 线） |

fill' 8–11µs（kv1 实测 7.8 + 更深 prologue 链 Qload→G1(0)→SM(0)→G2(0)→AO(0)，推断）；
launch 2.9（六工况实测一致）。

**置信度：中-高。** 锚全部来自同家族同门控实测（R4 busy、R6 边际周期与残余、iter2/iter3 两次
机制兑现）；唯一外推是 FIN 摊薄与 fill' 增量。break-even 需要 flat 残余相对实测值膨胀 1.7×——
iter4 的膨胀机制（解除节流→队列加深）在 C2 中不存在（门控不动）。

### A.4 go/no-go 决策

**GO。** 中枢 −17.7%（区间 −9.5%..−22%）≫ 3% 阈值；no-go 判据（中枢 <3% 或边界份额不可分辨
地小）均不成立。风险登记与缓解见 §B.6；若 Gate 0 逐位一致失败且不可定位修复、或 Gate 2 <3%，
按任务书规则如实记无采纳（连续无提升 → 2）。

---

## Phase B —— 变体设计

### B.1 结构总览（对照母体与范本）

- **算术母体**：shipped `_gqa_kernel_prefill_g2stack`（gqa.py:2472–2969，ns=2/ci=2 批次结构）——
  SM/AO/FIN 的每条算术指令、fill 值、槽语义（neg_sm idx h*2+cur、sumexp_is/r_factors idx i*2+h、
  acc_o/sumexp idx h）逐行继承，仅索引重映射（i→kvpar、h→h、任务奇偶新增维）。
- **结构范本**：shipped `_gqa_kernel_prefill_g2stack_flat`（gqa.py:2982–3423，kv==1 展平流）——
  macro 发射模式、alloc_var 尾声绑定、init/destroy token 布局、@T.macro 显式传参、ws/L1/L0/UB
  分配全部继承。
- **展平流**：每核 u = 0..U−1（U = 2·mc），task_of(u) = my_start + u//2，kv_of(u) = u%2。
  uniform 保证每 task 恰 2 个 u（noncausal → kv_useful≡2 恒定，无变长流机制）。
  C/V 两侧的发射序列由**同一张调度表**（§B.2）生成——flag 计数镜像不变式按构造成立。

工厂：`_gqa_kernel_prefill_g2stack_flat_kv2(B, S_padded, S_kv_padded, sm_scale, is_causal, ...)`
——命名对齐现有风格；host 侧 gate 断言：`kv_iters==2 ∧ not is_causal ∧ D==128 ∧ block_M==64 ∧
block_N==128 ∧ G%2==0 ∧ block_num ≥ 2*P28_NUM_CORES ∧ dtype∈{fp16,bf16}`（is_causal 直接 assert
拒绝——uniform 子类先行，causal kv==2 gate 外走 shipped）。

### B.2 调度表（C/V 同式生成的单一事实源）

**C 流**（步序 G1-first，硬约束 1）：

```
prologue:  Qload(t0);  G1(0,kv0)
loop tt = 0 .. mc-2:                     # T.serial(my_count - 1)，对展平
    G1(tt,kv1)      # u=2tt+1：q_l1 免等 full token（内容驻留），set free
    G2(tt,kv0)      # u=2tt
    Qload(tt+1)     # 任务边界：双 wait free token（消费 G1(tt,kv0)+G1(tt,kv1)）
    G1(tt+1,kv0)    # u=2tt+2：wait Q full token
    G2(tt,kv1)      # u=2tt+1
epilogue:  xt1 = my_count-1 (alloc_var)
    G1(xt1,kv1); G2(xt1,kv0); G2(xt1,kv1)
destroy:   同 kv1 flat + SIG_Q_L1 (MTE1→MTE2) ×2
```

**V 流**（拍序 AO→[FIN]→SM(+2)，kv1 flat 的 AO(jj)→FIN(jj)→SM(jj+2) 推广；FIN 仅任务尾）：

```
prologue:  SM(0,kv0); SM(0,kv1)
loop tt = 0 .. mc-2:
    AO(tt,kv0)      # u=2tt：fill acc_o/sumexp 槽[tt%2]（按任务 fill，iter2 NaN 教训）
    SM(tt+1,kv0)    # u=2tt+2：neg_sm prv 槽 cap 重填（见 B.4）
    AO(tt,kv1)      # u=2tt+1：真实 rescale 链（无 fill）
    FIN(tt)         # 任务尾：归一化 + Output 写出
    SM(tt+1,kv1)
epilogue:  xt1 = my_count-1 (alloc_var)
    AO(xt1,kv0); AO(xt1,kv1); FIN(xt1)
destroy:   同 kv1 flat
```

与单循环模型 beat x: `AO(x), [x 奇: FIN(x//2)], SM(x+2)` 逐项等价（展开验证：beat 2tt =
AO(2tt),SM(2tt+2)；beat 2tt+1 = AO(2tt+1),FIN(tt),SM(2tt+3) ✓）。对展平后循环体含 5 个发射块
（kv1 为 3 个），迭代数减半（mc−1 vs mc−2），每 u 标量发射摊销与 kv1 flat 同量级。

**mc=2 最小边界**（gate 下限 block_num=40）：C = Qload,G1(0,0) | tt=0: G1(0,1),G2(0,0),Qload(t1),
G1(1,0),G2(0,1) | G1(1,1),G2(1,0),G2(1,1)；V = SM(0,0),SM(0,1) | AO(0,0),SM(1,0),AO(0,1),FIN(0),
SM(1,1) | AO(1,0),AO(1,1),FIN(1)。所有 u∈{0..3} 恰覆盖一次 ✓。mc 混合（r_tasks≠0，如 c7 的
4/3）：每核独立流，cross-flag 为同组 AIC↔AIV 核内信号、ws 按 cid 索引，无跨核耦合 ✓。

### B.3 flag/sem 布局与计数镜像（家族最高危 bug 类的构造性防御）

cross id 0–5 沿用（**零新增**；6–10 余量不动）。init token 布局与 kv1 flat 完全一致：
C-scope init `WS2_C2V ×1`；V-scope init `WS1_V2C ×1`、`WS3_V2C ×1`（单 token 承重门控原样继承，
硬约束 1 / iter4 finding 3）。本地 event id：K 0/1、V 2/3、P 4、Q 5、L0AB 0/1、L0C 0/1、
IO_UB 0/1、MASK 2、S_HALF 0——0–5 已满、6/7 保留，零新增。

U = 2mc 个 u 的全流计数表（生成器按 §B.2 调度表机械累加，audit 脚本静态复核）：

| 信号 | set 方/处 | wait 方/处 | sets | waits | 残差 | FIFO 有效门（1-token 语义） |
|---|---|---|---|---|---|---|
| WS1_C2V (id0) | C FIX / 每 G1 | V / 每 SM 头 | U | U | 0 | SM(u) ← G1(u)（数据就绪，精确） |
| WS1_V2C (id1) | V MTE2 / 每 SM 尾 + init1 | C / 每 G1 头 | U+1 | U | 1 | G1(u) ← SM(u−1)（承重 pacing，紧一拍） |
| WS2_V2C (id2) | V MTE3 / 每 SM 尾 | C / 每 G2 中 | U | U | 0 | G2(u) ← SM(u)（数据就绪，精确） |
| WS2_C2V (id3) | C MTE2 / 每 G2 尾 + init1 | V / 每 SM 头 | U+1 | U | 1 | SM(u) ← G2(u−1)（承重，紧一拍） |
| WS3_C2V (id4) | C FIX / 每 G2 尾 | V / 每 AO 头 | U | U | 0 | AO(u) ← G2(u)（数据就绪，精确） |
| WS3_V2C (id5) | V MTE2 / 每 AO 尾 + init1 | C / 每 G2 头 | U+1 | U | 1 | G2(u) ← AO(u−1)（承重，紧一拍） |

与 kv1 flat 的每 u 计数/pipe 归属/门拓扑**逐项同构**（mc→U 替换）——该拓扑已随 shipped flat
在 iter2 采纳、iter3/iter4 BASE 臂累计 40+ 次上机验证。本地 flag 对（每 u 计数同 kv1 的每 u；
Q 协议见下）：

**Q 协议（kv2 特有的唯一新本地协议——单 token 形式）**：q_l1 被同任务两个 G1（kv0/kv1）连读、
任务边界才重装。首版采用"双 free token"（init ×2、两次读各 set、Qload 双 wait）——**mc=2 冒烟
即 aicore timeout 死锁**：shipped 代码全部本地 flag 协议均为深度 1 ping-pong，本地 event 的
pending≥2 从未被验证，双 set 疑被合并/不支持（cross-sem 的计数深度是 iter4 单独实证的，不可
外推到本地 event）。修正为单 token 形式：
- G1(·,kv0)：wait MTE2→MTE1 full（Qload 已装）；copy q_l1→l0a[0]；**不 set**。
- G1(·,kv1)：**免 wait**（内容自 kv0 的 wait 后驻留，期间无 Qload——C 流串行发射保证）；
  copy q_l1→l0a[1]；**set MTE1→MTE2 free ×1**——MTE1 为顺序队列，set_flag 在 MTE1 流中位于
  两次 q 读之后，故单 token 即隐含"本任务两次读均完成"（与 shipped 的 P_L1/K_L1 释放语义同款）。
- Qload(t+1)：wait MTE1→MTE2 free ×1；copy Q→q_l1；set MTE2→MTE1 full ×1。
- init：free ×1（同 kv1 flat）；destroy：wait free ×1（同 kv1 flat）。
- 计数：free 方向 sets = 1(init)+mc，waits = mc(Qload)+1(destroy) ✓；full 方向 sets = mc，
  waits = mc（仅 G1 kv0）✓。**深度/计数与 kv1 flat 的 Q 协议完全同构**（每任务装载 1 次、
  读 2 次但只有末读释放），本地 event pending 深度 ≤1，无未验证语义依赖。

pending 深度推导（cross）：数据就绪 sem 严格 1:1（producer lead ≤1）；release sem 的 producer
被 consumer 数据链回环约束（SM(x)←G1(x) FIX；G2(x)←SM(x) MTE3；AO(x)←G2(x) FIX），
completed-producer ≤ issued-consumer + 1(init) + 1(紧一拍 lead) → **max pending ≈ 3 ≤ 6 安全界**
（hw 15；kv1 flat 同拓扑实测 40+ 运行零死锁，iter4 ns4 臂 pending≈5 亦零死锁）。

### B.4 V 侧状态槽逐 u liveness 推演（硬约束 6——本轮最高风险点，不照抄 kv==1 结论）

槽位重映射（kv1 → kv2）：

| 状态 | kv1 flat（任务奇偶= u 奇偶） | **kv2 flat** | 写者 | 读者 |
|---|---|---|---|---|
| acc_o [4,half,D] | idx (x%2)*2+h | **idx (t%2)*2+h（任务奇偶）** | AO(t,kv0) fill+加、AO(t,kv1) 链加 | FIN(t)、AO(t,kv1) |
| sumexp [4,half,1] | idx (x%2)*2+h | **idx (t%2)*2+h** | 同上 | FIN(t) |
| sumexp_is [4,half,1] | idx (x%2)*2+h | **idx (kv%2)*2+h（任务内 kv 序）** | SM(t,kv) | AO(t,kv) |
| r_factors [4,half,1] | idx (x%2)*2+h | **idx (kv%2)*2+h** | SM(t,kv) | AO(t,kv) |
| neg_sm [4,half,1] | idx h*2+cur，cur≡0（kv1 退化） | **idx h*2+cur，cur=kv%2（g2stack 原语义）** | SM(t,kv) | SM(t,kv)（prv 读） |

**为什么不能照抄 kv1 的 u 奇偶**：kv2 中 AO(2t)→AO(2t+1) 是同任务真实数值链（acc_o/sumexp
跨两个 u 存活）；若按 u 奇偶分槽，AO(2t+1) 会读错槽（读到任务 t−? 的陈旧数据）——**精度腐蚀
而非死锁**，Gate 0 逐位对比是唯一可靠捕获手段（任务书原文）。故 acc_o/sumexp 用**任务奇偶**，
而 sumexp_is/r_factors 的生命周期是"SM(u) 写 → AO(u) 读 → SM(u+2) 覆写"，与 kv1 完全同构，
用 **kv 序**（= u%2 = 写读窗口内的区分维）即可。

逐 u 推演（V 流程序序 = 拍序；所有状态槽读写均在 V pipe 上按序执行，MTE2/MTE3 不触碰这些槽）：

1. **sumexp_is/r_factors[kv槽]**：SM(t,kv) 于 beat 2t+kv−2 写槽[kv*2+h]；AO(t,kv) 于 beat 2t+kv
   读之（间隔 2 拍，期间仅 SM/AO 使用另一 kv 槽——kv0/kv1 槽互不干扰）；下一写者 SM(t+1,kv)
   于 beat 2t+kv（**同拍、程序序在 AO(t,kv) 之后**：beat 2t = AO(t,0),SM(t+1,0)；beat 2t+1 =
   AO(t,1),FIN(t),SM(t+1,1)）→ 读完成后覆写 ✓。覆写内容死区验证：AO(t,kv) 是槽[kv] 在
   SM(t+1,kv) 前唯一读者 ✓。
2. **acc_o/sumexp[任务槽 t%2]**：写者链 AO(t,0) fill→加、AO(t,1) 链加（beat 2t、2t+1）；读者
   FIN(t)（beat 2t+1，程序序在 AO(t,1) 后）；下一写者 AO(t+2,0) 的 fill（beat 2t+4）——间隔
   3 拍（beat 2t+2/2t+3 处理任务 t+1 的槽[(t+1)%2]）✓。FIN(t) 的原地 div 之后无读者 ✓。
   **重叠窗口翻倍核查**（任务书风险点）：与 AO(t,·) 并发窗口的异任务写者只有 SM 系（写
   sumexp_is/r_factors/neg_sm，不写 acc_o/sumexp）与 AO(t+1,·)（异槽）——任务奇偶双槽在 kv2 下
   **仍然充分**，因为 acc_o/sumexp 的写者只有 AO/FIN（V 单流串行），跨任务重叠仅发生在
   "SM(t+2 系) 与 AO(t 系) 同拍" 且两者槽集不相交（SM 不触 acc_o/sumexp）✓。
3. **neg_sm[h*2+cur]**：SM(t,0)：fill [h*2+1]=cap（新任务 prv 重置，替代 g2stack 的任务头整体
   fill——[h*2+0] 由 reduce_max 全覆写，cap fill 为死存储，可省；值等价见 B.5-3）→ reduce_max
   写 [h*2+0] → min 读 [h*2+1]=cap ✓。SM(t,1)：reduce_max 写 [h*2+1]（覆 cap，同 g2stack）→
   min 读 [h*2+0]（SM(t,0) 的 max，V 序 1 拍前）→ r_factors[1*2+h] = [h*2+1]−[h*2+0]（真实
   rescale）✓。死区：[h*2+1] 在 SM(t,1) 算完 r_factors 后死亡（下一读者 SM(t+1,0) 的 fill）✓；
   [h*2+0] 在 SM(t,1) 的 min 后死亡（SM(t+1,0) 的 reduce_max 全覆写，不读 dest）✓。
   **NaN 纪律**：prv 槽每任务 cap 重填 → 前任务 NaN max 不跨任务泄漏；acc_o/sumexp 每任务
   AO(t,0) 头部 fill（iter2 免-fill NaN 教训原样继承，位置 V 序安全：同槽前读者 FIN(t−2) 在
   3 拍前）✓。init 一次性 fill（neg_sm 全 cap、acc_o/sumexp 全 0）保留（kv1 模式），首任务
   SM(0,0)/AO(0,0) 前的槽状态与 g2stack 首任务 fill 后一致 ✓。
4. **io_buf/acc_s_half/work_ub/buf_2d/mask_v**（tile 临时件）：SIG_IO_UB+side / SIG_S_HALF
   ping-pong 协议逐字继承 kv1（kv1 的 FIN 每拍一次比 kv2 更密，已验证）；kv2 的 FIN 每 2 拍
   一次，acc_s_half 争用更松 ✓。

**结论：奇偶双槽（4 槽 = 2 任务奇偶 × 2 head）在 kv2 的必需 overlap 下充分**，无需扩 UB、
无需新 id；offset 结构重映射为 AO(t,kv)→FIN(t)→SM(t+1,kv)（即 u 域的 AO(x)→[FIN]→SM(x+2)）。

### B.5 与 shipped g2stack 的算术等价（逐位一致论证 → Gate 0 判据）

1. **SM(t,kv,h)**：指令序列 = g2stack phase-1 体（i→kv、槽索引 i*2+h→kv*2+h、neg_sm cur/prv
   同式）；mask 列窗 idx*block_N→kv*block_N 同值（idx=k*ns+i=kv，num_outer=1）；ws_s 读列窗
   同 tile。fill 差异仅 neg_sm[h*2+0] 的 cap 死存储省略（reduce_max 全覆写 dest，不读旧值）。
2. **AO(t,kv,h)**：exp(r)→sumexp·=r→sumexp+=is→broadcast→acc_o·=r→ws_o 读→acc_o+=work——
   逐 op 同序同槽值；kv0 头部的 acc_o/sumexp fill 值 = g2stack 任务头 fill 值（0.0），位置从
   "phase1 前" 移到 "AO(kv0) 头"——期间无读者（V 串行 + 槽死区推演 B.4-2）→ 算术序列不变。
3. **neg_sm cap 语义**：g2stack 任务头 fill 全 4 槽 cap；变体只重填 prv 槽——cur 槽被 reduce_max
   全覆写后才被读，两者可观测值逐步相等。
4. **FIN(t,h)**：sumexp+=1e-30→div→acc_s_half→Output 盒（同 (bz,hq0+h,bx,vid) 寻址）逐字同。
5. **C 侧**：G1/G2 的 mma(init=True) tile、K/V/Q GM 盒、l0 奇偶、ws 写读列窗逐 u 同 g2stack
   的 (k=0,i=kv) 批内序列；ws_s/ws_p/ws_o 内容与批次形式逐位相同（同 GEMM 同数据）。
6. 推论：Output 应**逐位一致**（纯调度改动）。非逐位 = bug，定位修复，不放宽容差。

### B.6 资源核算（iter1 §4.2 方法）与风险登记

| 资源 | kv1 flat（shipped） | kv2 变体 | 增量 |
|---|---|---|---|
| L0A/L0B/L0C | [2,128,128] f16/f16/f32 = 64K/64K/128K B（恰满包络） | 同 | **0** |
| L1 | q 32K + k0/k1 64K + v0/v1 64K + p 32K = 192KB | 同 | **0** |
| UB/vid | io 32K+work 16K+buf2d 16K+accs 8K+acc_o 64K+mask 16K+状态 ~2K ≈ 154KB | 同（槽维数不变，索引重映射） | **0** |
| GM ws | (2.62+1.31+2.62)MB = 6.55MB ≪ L2 192MB | 同 | **0** |
| cross id / 本地 id | 0–5 / 0–5（6/7 保留） | 同 | **0 新增** |
| pending 深度 | ≤3（实测） | ≤3（B.3 推导） | 0 |

风险登记（概率降序）：
1. **状态槽索引错位**（B.4 的 t%2 vs u%2 混用）→ 精度腐蚀非死锁；缓解：逐位 Gate 0（10 用例
   含 NaN/Inf 注入 + mc=2/混合 mc/pad 列/行）+ 生成器单调度表 + audit 脚本静态比对槽表达式。
2. **Q 双 token 协议竞态**（新本地协议）→ 死锁或 q_l1 撕裂；缓解：计数表 B.3 静态平衡 +
   mc=2 最小边界用例 + 混合 mc（异构计数核并发）。
3. **框架位约束**：my_count−1 仅经 alloc_var 语句位绑定进 macro（iter2 finding 3）；宏参
   kvpar 为 Python 字面量（trace 期分支，use_mask 同模式）；@T.macro 显式传参（finding 1）。
4. **codegen 差异**（循环体 5 发射块）→ 编译失败或 scalar 膨胀；缓解：独立变体文件先编译
   （Gate 0 首用例即验证），get_kernel_source 行数对照记录。

### B.7 交付物与 harness（不改 gqa.py）

- `gqa_flat_kv2_variant.py`：变体工厂（以 kv1 flat 块为结构母版手工重构 + 溯源注释；
  `audit_flat_kv2_iter5.py` 提供 gen_variants 式行级断言：源块边界防漂移 + 算术/flag 行溯源
  匹配 + 调度表计数镜像静态复核）。
- `diff_iter5_kv2.txt`：变体 kernel 块 vs kv1 flat 块的 unified diff（审计工件）。
- `run_iter5_variant.py`：Gate 0 runner（内联同源 golden；monkeypatch 仅限变体构建路径）。
- `prof_gqa_ab_iter5.py`：双臂探针（patch `G._gqa_kernel_prefill_g2stack` 的 g2stack spec 构建：
  命中 gate 时转发变体工厂；variant_invoked + chosen==["g2stack"] 双断言，静默编译回退不可能）。
- `run_matrix_iter5.sh`：同会话双臂交替（每轮 base→c2 轮换臂序），msprof 输出
  `msprof_op_iter5_{r4,c7}_{base,c2}_s{1..5}`（不覆盖既有目录）。

---

## Phase C/D/E 结果回填（2026-10-18，mode=implement 执行后证据）

详细数据见 `candidate_iter5_C2.json` 与 `perf_log.md` iter5 条目；此处仅回填判决与设计偏差。

### C.1 Gate 0（独立精度）：PASS 10/10

`run_iter5_variant.py`（内联同源 golden）：r4/c7-like（直建 + dispatch 重定向双层）、mc=2 整除边界
（stacked=40）、混合 mc 3/2 与 7/6（r_tasks≠0）、padcol（Skv=200，use_mask kv 索引列窗）、
padrow（S=100）、稀疏 NaN/Inf 注入**双 kv 块**（真实 rescale 链携带特殊值）——全部与 shipped
g2stack **逐位一致**（NaN 位置 / Inf 位置+符号 / 有限位精确相等），golden viol=0，零 fallback。
gate 隔离 2 例（causal kv2、stacked=32 ∈ [20,40)）：变体未被调用、chosen=g2stack、
patched-vs-plain 逐位不变。**§B.5 的逐位一致论证被实证确认。**

**实施期设计修正（唯一）**：§B.3 首版 Q 协议（init free ×2、G1 双读各 set、Qload 双 wait）在
mc2 冒烟**死锁**（aicore timeout 507014，prologue 挂起签名）——本地 event id 的 token 叠加
不被支持（cross-sem 的计数深度是 iter4 单独实证，不可外推到本地 set/wait_flag）。修正为单
token 形式：G1(kv1) 为唯一 free setter（MTE1 顺序队列排空语义 ⇒ 单 token 隐含双读完成），
Qload 单 wait / 单 full set，init/destroy ×1——计数与深度与 shipped kv1 flat **完全同形**。
修正后审计重跑全绿、10/10 通过。§B.3 表中 Q 行以此为准（audit 脚本 LINES/CONTRIB 已同步）。

### C.2 Gate 2（性能，同会话双臂交替 5+5，20/20 额定频率）

| shape | BASE（shipped g2stack ns2） | C2 变体 | Δ |
|---|---|---|---|
| R4 c4-like（主判据） | 206.96µs [204.26–211.08] | **187.97µs** [187.02–190.06] | **−9.18%** |
| c7-like（次判据，首测） | 38.34µs [37.42–39.18] | **35.52µs** [35.10–35.84] | **−7.34%** |

BASE 复现 iter3 锚点（207.85 → 206.96，−0.43% ∈ ±3%）；双臂分布不重叠。**≥3% 判据双 shape
满足 → 采纳。** 对照 Phase A：实测落在保守缘（−9.5% 档）——偏差来源 = flat 形态 vec busy
+9.3µs（2.20→2.38µs/kv；per-task neg_sm cap 重填 + AO 头 fill + FIN 节拍的开销，推断），
A.3 表对此项估计偏低（+0.05..0.15 → 实测 +0.18µs/kv）。

遥测机制结论（`telemetry_iter5_compare.txt`）：批互替等待如设计塌缩（C-id2 −72%、V-id4 −72%），
等待**转移**到 release/pacing 侧（C-id1 +100.6µs、V-id3 +103.0µs）——进入 kv1-flat 同款
pacing-bound 稳态，且无 iter4 式生产门延迟膨胀（前跑深度未变）。exposure/kv 1.726→1.181。

### C.3 Gate 3（采纳接入）：PASS 全链

备份 `history_version/gqa_impl_s3_iter5.py`（md5 cbce7441…）→ 工厂 + dispatch gate 接入
（g2stack 保留为编译回退，iter2 模式）→ **连带修复 3 处预存 ruff format 偏差**（用户批准；
整文件 `ruff format` 机械重排，**AST 等价证明**：ast.dump 前后逐字节相等 725521 chars，
新增函数恰 1 个、删除 0）→ ruff 0.15.14 + 0.16.7（CI 版，wheel 下载）**双版本四门禁 exit 0**
→ 全套件 18+6 全绿 exit 0 零 fallback → 自检 Test Passed! → gate 外 **8/8 逐位不变**且派发臂
probe 核验未动 → in-gate 2/2 逐位一致且 chosen 迁移 `g2stack → g2stack_flat_kv2` → msprof
兜底 R1 10.57µs（锚 10.83）/ R3 60.92µs（锚 61.09）无回退 → instrumentation grep=0、
npu-smi 健康、残留进程 0。新 gqa.py md5 = **bd470e072d0a48faeac1e786306932e8**。
