# GQA "flat 跨任务软件流水" 优化假设适用性分析（Stage 3 · analysis_only）

- 日期：2026-10-16
- 对象：`examples/gqa/gqa.py`（未修改）；评估用户提出的 "flat 把 (task, kv) 压成连续令牌流，
  4 相位（G1/SM/G2/AO）跨任务软件流水" 假设
- 方法论：先定性（延迟/吞吐）→ 再看重叠空间 → 再查资源与框架约束 → 最后给动手方案（本轮不实施）
- 本轮硬约束遵守情况：`gqa.py` / `test_gqa.py` 未做任何修改；所有性能数字来自 msprof
  （simulator 或 op 模式）；未读写 `.orchestrator_state.json`

---

## 0. 结论速览

**verdict：部分适用（限定工况）— 置信度：高（对"适用/不适用"的工况划分）；中（对收益量级）**

| 工况 | 派发 kernel | tasks/core | flat 适用性 | 预期收益（推断） |
|---|---|---|---|---|
| 默认小 shape B=1,S=128,Nq=8,Nkv=2,D=128 causal | `_gqa_kernel_expert`（per-head AUTO，16 blocks） | **1** | **不适用**（无第二任务可交错） | 0%（需走替代措施 §5.3） |
| L0-c1 B=2,Nq=32,Nkv=8 | `_gqa_kernel_prefill_g2stack`（ns=1） | 3–4 | **适用（收益受限）** | ~1.2–1.5×（推断） |
| c3 类 B=2,Nq=128,Nkv=8 | g2stack（ns=1） | 12–13 | **适用（收益最大区）** | ~1.35–1.65×（推断） |
| c4 类 B=4,S=256,Skv=256,Nq=64,Nkv=8 | g2stack（kv=2,ns=2，任务内已交错） | 25–26 | **适用（跨任务边界扩展）** | ~1.17–1.36×（推断，低置信） |

- 用户假设的 "交接延迟全程暴露的串行态 wall = N×(4 相位 + 4 次交接)" **不完全属实**：
  - g2stack 已存在 `AO(t-1) ∥ G1(t)` 跨任务重叠（代码依赖链证明 + 拟合模型佐证）；
  - 真正暴露的串行段是 `SM(t)`（vector ~2.6µs/task）夹在 `G1(t)` 与 `G2(t)` 之间，
    以及每 task 的 Q 装载、cross-flag/GM-workspace 往返延迟、scalar 发射开销；
  - 单 task/core 的工况（默认 shape）根本没有跨任务重叠空间，flat 无从生效。
- 关键实测支撑：**同一 kernel、同一 [128,128] stacked tile 下，ns=2 任务内批次交错比 ns=1
  串行省 28.6%/task（3.76 vs 5.27 µs/kv 等效）**——证明交错机制本身有效；flat 是把该机制
  延伸到任务边界（kv=1 时任务内批次深度坍缩为 1 的工况）。
- 资源核算：flat 化（ns=2 槽位、单 q_l1、单 acc_o）**不新增 L0A/L0B/L0C 占用，L1 不变
  （192KB/512KB），UB 不变（~130KB/192KB），GM ws 3.3→6.6MB（L2 192MB 常驻）**，全部放得下。
- 框架约束：只能用 expert 模式 `T.serial` + 手动 flag 重写（`T.Pipelined` 不可嵌套、causal
  谓词会 segfault，仓库已有记录）；本地 event id 每 pipe 对可用 0–5 已用满（6/7 硬件保留），
  cross flag id 0–10 尚有 5 个余量——flat 方案必须复用现有 id 空间（§4.3 给出可行排布）。

---

## 1. 证据来源与采集命令

### 1.1 已有 simulator 数据（默认小 shape，昨天采集，直接复用）

- 目录：`examples/gqa/msprof_sim/OPPROF_20261016163102_PRXNKARSFXKBQCTP/`
- 采集命令（见 `examples/gqa/prof_gqa_pipeline.py` 文档串）：
  `msprof op simulator --soc-version=Ascend910_9362 --kernel-name=main_kernel --output=examples/gqa/msprof_sim python3 examples/gqa/prof_gqa_pipeline.py`
- shape：B=1 S=128 S_kv=128 N_q=8 N_kv=2 D=128 causal fp16 → 派发 `_gqa_kernel_expert(bm=64)`
  （per-head AUTO，`block_num = B*N_q*q_blocks = 16`，16 核各 1 task）
- 模型总 tick 33777；main_kernel 窗口 = trace 时间轴 5.190 → 10.294 µs，**wall 5.104 µs**（sim 时基）

### 1.2 本轮补采的真实 NPU 数据（Ascend910_9362，msprof op 模式）

环境契约：仓库根、`unset PYTHONPATH`、`export ACL_OP_INIT_MODE=1`；tilelang 0.1.4+ubuntu.24.4.cann920（site-packages）。

| # | shape | 命令（prof_gqa_pipeline.py 参数） | 输出目录 | Task Duration | Block Dim |
|---|---|---|---|---|---|
| R1 | 默认 B=1,S=128,Skv=128,Nq=8,Nkv=2,D=128,causal,fp16 | 默认 | `perf_tuning/msprof_op_default/OPPROF_20261016173026_AHDMSLSSWRRTUPXX` | **9.94 µs** | 16 |
| R2 | L0-c1 B=2,S=128,Skv=128,Nq=32,Nkv=8,D=128,causal,fp16 | `--batch 2 --nq 32 --nkv 8` | `perf_tuning/msprof_op_l0c1/OPPROF_20261016173054_HIGYPGUEBTPGXPNG` | **30.04 µs** | 20 |
| R3 | c3 类 B=2,S=128,Skv=128,Nq=128,Nkv=8,D=128,causal,bf16 | `--batch 2 --nq 128 --nkv 8 --dtype bf16` | `perf_tuning/msprof_op_c3like/OPPROF_20261016180317_LORUJDGRRJOJRUNC` | **77.44 µs** | 20 |
| R4 | c4 类 B=4,S=256,Skv=256,Nq=64,Nkv=8,D=128,noncausal,bf16 | `--batch 4 --seq 256 --seq-kv 256 --nq 64 --nkv 8 --dtype bf16 --no-causal` | `perf_tuning/msprof_op_c4like/OPPROF_20261016180404_HQYZBYLGCJKNBCZK` | **204.42 µs** | 20 |

统一命令模板：
`msprof op --kernel-name=main_kernel --aic-metrics=PipeUtilization --output=<dir> python3 examples/gqa/prof_gqa_pipeline.py <flags>`
（四次运行 msprof 均报告 "aicore compute usage lower than 20%"。）

### 1.3 本轮补采的 simulator 数据（L0-c1 / g2stack）

- 命令：`msprof op simulator --soc-version=Ascend910_9362 --kernel-name=main_kernel --output=examples/gqa/perf_tuning/msprof_sim_l0c1 python3 examples/gqa/prof_gqa_pipeline.py --batch 2 --nq 32 --nkv 8`
- 状态：**采集中止（如实报告）**。仿真运行 75 分钟（msopprof 762% CPU 持续活跃）仍未
  到达 dump/parse 阶段，未产出 trace.json，已终止。原因分析（推断）：对照默认 shape
  模型阶段仅 7.35s/33777 tick，g2stack 每 task 数十次 cross-flag set/wait × 4 task × 60
  核块，camodel 对跨核信号量等待的仿真吞吐过低。残留部分产物：
  `msprof_sim_l0c1/OPPROF_20261016173720_YHUCFYARSETWMIFK/device0/tmp_dump/`（44KB，
  仅 aicore_binary.o + config）。
- 对本报告结论的影响：**无实质影响**。task 边界重叠结构的判定依据两条独立证据链：
  (a) 代码依赖链推导（§3.1 事实 1-4，flag/semaphore 门条件逐一列出）；
  (b) 真机三深度（4/13/26 task-per-core）wall-busy 拟合模型（§3.2，ns=1→ns=2 的
  −28.6%/kv 为同 kernel 同 tile 的直接对照实验）。指令级逐段核实推迟到实施轮
  （届时可对变体 kernel 采集——变体 task 数更少、或改用真实 NPU 多组 shape 差分法）。

### 1.4 kernel 派发确认（事实，非推断）

- 派发逻辑：`gqa.py:3168 _resolve_kernel_chain` + `gqa.py:3192 gqa_expert`
- 默认 shape：`use_expert_prefill = ... and (B*N_q*q_blocks) >= P28_NUM_CORES`，
  16 < 20（`P28_NUM_CORES=20`，gqa.py:44）→ False → per-head ladder `_gqa_kernel_expert(bm=64)`
  （gqa.py:3588-3614；L0C 预算闸门 `_perhead_bm_ladder` gqa.py:3163）。
  tilelang 缓存佐证：`~/.tilelang/cache/7bc162f6.../kernel.cu`（main_kernel，无 ws 参数）。
- L0-c1/c3 类：`use_g2_stack = True`（L0-c1 stacked 实例 = 2*8*2*2 = 64 ≥ 20；c3 类 = 2*8*8*2 = 256）
  → `_gqa_kernel_prefill_g2stack(ns=min(kv_iters,8)=1, ci=2)`（gqa.py:3503, 3514-3540）。
  缓存佐证：`~/.tilelang/cache/c0468f48.../kernel.cu`（main_kernel 带 ws_s/ws_p/ws_o；
  l0a InitBuffer 65536 = [2,128,128] fp16 → M_stack=128 → g2stack；
  生成代码含 `if (cid < 4) condval=1 ... for (t < condval+3)` → 4 核 4 task、16 核 3 task，
  与 task_range(64 tasks/20 cores) 一致）。
- c4 类：kv_iters=2 → g2stack `ns=2`（gqa.py:3515）。
- R1 Block Dim=16 / R2-R4 Block Dim=20 与上述派发一致（msprof OpBasicInfo）。

---

## 2. Step 1 定性：延迟受限 or 吞吐受限

### 2.1 实测利用率汇总（msprof op，PipeUtilization.csv，min–max across cores）

| 流水 | R1 默认(16blk) | R2 L0-c1(20blk) | R3 c3类(20blk) | R4 c4类(20blk) |
|---|---|---|---|---|
| aic_cube | 4.0% | 6.7–9.0% | 10.5–11.4% | 16.1–16.8% |
| aic_mte2 | 10.9–23.4% | 14.6–22.9% | 22.1–25.6% | 24.6–28.0% |
| aic_fixpipe | 6.8–8.2% | 9.6–13.9% | 15.2–18.4% | 24.8–27.1% |
| aic_scalar | 41.8–53.7% | 28.8–47.7% | 40.5–49.0% | 43.4–48.3% |
| aiv_vec | 16.9% | 25.9–34.5% | 41.6–45.1% | **53.9–56.1%** |
| aiv_mte2 | 5.7–14.2% | 10.1–17.0% | 14.9–19.0% | 16.4–20.9% |
| aiv_mte3 | 3.4–3.8% | 6.5–10.0% | 9.9–12.1% | 12.7–14.7% |
| aiv_scalar | 21.9–42.9% | 29.6–45.5% | 42.9–50.2% | 39.3–42.7% |

（R2 中 9.0%/34.5% 为 4-task 核、6.7%/25.9% 为 3-task 核——task_range 静态不均衡 64=4×4+16×3，
热点核决定 wall；比例 9.0/6.7≈4/3 ✓。）

### 2.2 定性判定

- **默认小 shape（R1）：延迟受限（latency-bound），单波次。**
  - 16 task < 20 cube 核 → 每核恰 1 task，4 核闲置；kernel 内不存在第二波。
  - cube 忙 4.0%，vec 忙 16.9%——没有任何流水接近饱和；wall 由
    "launch → Q/K 装载 → G1 → 交接 → SM → 交接 → G2 → 交接 → AO/归一化 → 写回"
    单条依赖链的**延迟**决定（§3.3 sim 时间轴逐段证实）。
  - simulator 佐证：wall 5.104µs 内 MMAD 总忙 0.300µs（5.9%），V 侧 WAIT_FLAG_DEVI
    累计 3.51µs、C 侧 2.54µs（`core0.*_instr_exe.csv` 首行）。
- **L0-c1（R2）：延迟受限为主、开始向吞吐过渡。** 3–4 task/core，vec 25.9–34.5%，
  cube ≤9%。每 task wall 7.51µs vs vector 忙 2.60µs——~65% 的 wall 是气泡/串行交接/装载。
- **c3 类（R3）：过渡区。** 13 task/core，vec 41.6–45.1% 成为最忙计算流水，
  但 scalar（40.5–50.2%）与之持平；仍未饱和 → 交错空间大。
- **c4 类（R4）：接近吞吐区（vector-bound 趋势）。** 26 task/core × kv=2（ns=2 任务内批次），
  vec 53.9–56.1% 为全表最高；fixpipe 24.8–27.1% 次之。说明**任务内 2 级交错已把
  vector 利用率从 ~43%（R3 同 tile）拉到 ~55%**，但离饱和（>80%）仍有距离——
  任务边界 drain/refill 仍在漏气。
- 结论：**两种工况结论不同**——小 shape（1 task/core）是纯延迟区，flat 跨任务流水
  在其定义域外；≥3 task/core 的 g2stack 工况处于"未饱和的吞吐过渡区"，
  交错深度是主要杠杆，flat 在此定义域内。

---

## 3. Step 2 重叠空间：现状流水结构与量化

### 3.1 g2stack（L0-c1/c3/c4 的实际 kernel）当前重叠结构（代码事实）

C 侧程序序（gqa.py:2654-2797，`T.Scope("C")`）：
```
for t in serial(my_count):            # task 循环（task_range 静态切分）
    Q(t) → q_l1                        # 单槽 q_l1，SIG_Q_L1(=5) 双向 ownership
    for k in serial(num_outer):        # kv 批次循环，num_outer = ceil(kv_iters/ns)
        wait_cross(SEM_WS1_V2C)        # ws_s 已释放
        GEMM1 batch: for i in batch_iters:  K(u)→k_l1[i%2] → l0a/l0b[i%2] → mma → l0c[i%2] → FIX→ws_s[cid,i] ; 每 ci 个 set SEM_WS1_C2V
        wait_cross(SEM_WS3_V2C)        # ws_o 已释放（= 上一批 AO 完成）
        GEMM2 batch: for i in batch_iters:  V(u)→v_l1[i%2] → wait SEM_WS2_V2C(ci 粒度) → ws_p[cid,i]→p_l1 → mma → l0c[i%2] → FIX→ws_o[cid,i] ; 每 ci 个 set SEM_WS3_C2V
        set_cross(SEM_WS2_C2V)         # ws_p 已消费
```
V 侧程序序（gqa.py:2811-2962，`T.Scope("V")`）：
```
for t: fill(acc_o/sumexp/neg_sm)
    for k: phase1 SM batch: wait SEM_WS2_C2V; for i: [wait SEM_WS1_C2V(ci)] ws_s[cid,i]→io_buf→work_ub; mask; online-softmax; →acc_s_half→ws_p[cid,i]; set SEM_WS2_V2C(ci); set SEM_WS1_V2C
           phase2 AO batch: for i: [wait SEM_WS3_C2V(ci)] r_factors rescale; ws_o[cid,i]→io_buf→work_ub→acc_o+=; set SEM_WS3_V2C
    final: sumexp+1e-30 → div → Output（MTE3）
```

**结构性事实（依赖链推导，非猜测）：**
1. **kv 迭代内**已有双缓冲 + 定向 flag 跨迭代重叠：l0a/l0b/l0c `[2,...]` 按 `i%2` ping-pong，
   k_l1/v_l1 各双槽，ws_* 按 `[cid, i]` 多槽（i < ns），cross-flag 按 cross_interval=2 成组。
   ——即 skill flash_attn_optimize.md §1/§2 的 "num_stages 多级流水 + 细粒度 flag" 模式已实现。
2. **但批次窗口封在 task 内部**：`num_outer`/`batch_iters` 循环嵌套在 `for t` 之内，
   ws 槽位索引是 batch 内 `i`（gqa.py:2739 `ws_s[cid, i, ...]`），每个 task 重置为 0。
   **kv_iters=1（L0-c1/c3）时 ns=1、batch_iters=1 → 任务内交错深度坍缩为 1，
   4 相位在任务内纯串行**；这正是用户 flat 假设针对的空档。
3. **跨 task 已有部分重叠**：`G1(t) ∥ AO(t-1)`。证明：C 侧 G1(t) 的门条件是
   (a) cube 流水空闲（G2(t-1) mma+FIX 已发射完），(b) `SEM_WS1_V2C`（V 在 SM(t-1) 结束时
   set，gqa.py:2914），(c) l0a/l0b/l0c slot0 由 G2(t-1) 的 M/FIX 释放——全部在 AO(t-1)
   开始之前满足；而 AO(t-1) 只读 ws_o、只写 V 侧 UB，与 G1(t)（写 ws_s、用 C 侧 L0/L1）
   无资源冲突。V 侧 SM(t) 则必须等 G1(t) 完成（ws_s 数据依赖）→ **SM 段 cube 空闲**；
   C 侧 G2(t) 必须等 SM(t)（ws_p）且等 AO(t-1)（SEM_WS3_V2C，gqa.py:2745）→
   **G2 段 vector 空闲（AO(t-1) 已做完）**。
4. **单槽 ws 阻断更深交错**：SM(t+1) 写 ws_p[cid,0] 必须等 G2(t) 的 MTE2 读完 ws_p[cid,0]
   （SEM_WS2_C2V，gqa.py:2794/2847）；G2(t+1) 写 ws_o[cid,0] 必须等 AO(t) 读完
   （SEM_WS3_V2C）。→ ns=1 时稳态周期 = `Q装载 + G1 + SM + G2`（AO 被 G1(t+1) 掩盖），
   SM 全程暴露在 cube 空闲窗口。

### 3.2 量化：现状气泡占比（真实 NPU 拟合模型）

同 tile（g2stack stacked [128,128]、D=128）三工况拟合 `wall_hot_core = F + N_task × P_task`：

- R2（4 task/core，ns=1）：30.04 = F + 4×P1
- R3（13 task/core，ns=1）：77.44 = F + 13×P1
- 解得 **P1 = 5.27 µs/task，F ≈ 8.97 µs**（launch + 流水灌注/排空 + 首任务冷装载；两点拟合自洽）
- R4（26 task/core，ns=2 即每 task 2 kv）：204.42 = F + 26×P2 → **P2 = 7.52 µs/task = 3.76 µs/kv**
- **任务内 2 级批次交错收益 = 1 − 3.76/5.27 = 28.6%/kv**（机制有效性的直接实证）

每 task（stacked kv tile）流水忙时（由 R2/R3 PipeUtilization × wall / task 数）：

| 流水 | 忙时/task (µs) | 备注 |
|---|---|---|
| aiv_vec（每 vid） | 2.60 (R2) / 2.59 (R3) / 2.16 (R4,每kv) | **最忙计算流水** |
| aiv_scalar | 3.15 (R2) / 2.77 (R3) / 1.61 (R4,每kv) | flag 链 + 循环控制发射 |
| aic_scalar | 2.91 (R2) / 2.68 (R3) / 1.81 (R4,每kv) | 同上（C 侧） |
| aic_mte2 | 1.66 (R2) / 1.43 (R3) / 1.38 (R4,每kv) | Q/K/V/ws_p GM→L1 |
| aic_fixpipe | 0.96 (R2) / 1.01 (R3) / 1.02 (R4,每kv) | l0c→ws_s/ws_o GM 写 |
| aic_cube | 0.67 (R2) / 0.65 (R3) / 0.65 (R4,每kv) | 2×MMAD [128,128,128] |
| aiv_mte2 | 1.16 (R2) / 1.02 (R3) | ws_s/ws_o/mask GM→UB |
| aiv_mte3 | 0.65 (R2) / 0.65 (R3) | ws_p/Output UB→GM |

→ **ns=1 每 task wall 5.27µs 中，任何单一流水忙时 ≤3.2µs；≥2.1µs（≥40%）是
流水间暴露的串行等待/交接延迟**（推断：由忙时与 wall 的差值分解，非逐指令实测；
L0-c1 sim trace 落盘后可逐段核实）。scalar 发射（~2.7µs）与 vector（~2.6µs）
共同构成 flat 化的稳态下界。

### 3.3 默认小 shape（per-head AUTO kernel）sim 时间轴（core0，16 核一致）

simulator trace.json（§1.1），单位 µs，sim 时基（真实 NPU wall 9.94µs ≈ sim 5.10µs × ~1.95，
GM 延迟被 sim 低估）：

```
5.190  kernel start（scalar 取指/startup ≈0.82µs，占 wall 16%）
5.921  V-MTE2 mask→UB (0.444)          ┐ 与 Q/K 装载并行 ✓
6.014  C-MTE2 Q→L1 (0.465)             │
6.019  C-MTE2 K→L1 (0.617)             ┘
6.640  C-MTE1 L1→L0A/L0B ×12 (0.15)
6.830  CUBE MMAD #1 = G1 (0.150)
6.984  FIXP L0C→UB (G1 scores, 0.399) → 7.383 SET_CROSS_CORE
7.386  V 等待释放；7.388 V-MTE2 (0.291) → 7.680 V softmax 开始
       ← C→V 交接段暴露 0.30µs（其中 0.29µs 为一条与 G1 结果交付同拍的 V 侧 MTE2；
         该 MTE2 的源码对应未逐条核实，疑似 CV-combine 机制的 UB 数据整备——标注推断）
7.680–8.425  VECTOR softmax（~100 条指令，每条间夹 AUTO BAR）
8.426  V-MTE3 P(UB→L1) (0.169) → 8.595 SET_CROSS_CORE
8.598  C 等待释放（WAIT_FLAG_DEVI 6.06 发射、8.60 放行，全程 2.54µs，其中暴露 ~1.22µs）
8.600  C-MTE2 V→L1 (0.237)   ← V 装载未预取，串行在交接后（源码序 gqa.py:275 在 softmax 之后）
8.840  C-MTE1 ×12 → 9.030 CUBE MMAD #2 = G2 (0.150)   ← V→C 交接段 ~0.43µs（含 V 装载串行段）
9.184  FIXP L0C→UB (0.284) → 9.468 SET_CROSS_CORE
9.471  V 等待释放；9.472 V-MTE2 (0.289，性质同上标注推断) → 9.763 VADD (AO)
       ← C→V 交接段暴露 0.29µs
9.844–10.131  归一化 VADDS/VBRCB/VCOPY/VDIV(0.154)/VCONV
10.132 V-MTE3 Output (0.159) → 10.294 kernel end
```
- 3 次 C↔V 交接段暴露合计 ≈1.02µs = wall 的 20%（含交接后同拍的 V 侧 MTE2 段，见上标注）；
  launch/startup 16%；
  cube 忙 0.30µs = 5.9%；V-MTE2 忙 1.03µs、FIXP 0.68µs。
- **每核只有 1 个 task**：不存在"下一任务的 G1"来填补气泡 → flat 无对象。
- 16 task < 20 核：4 核全程闲置（波次不满，事实）。

### 3.4 对用户假设的逐条判定

| 用户假设 | 判定 | 证据 |
|---|---|---|
| "现状是 G1→SM→G2→AO 全串行、交接延迟全程暴露" | **部分属实** | ns=1 g2stack：任务内 4 相位确实串行（§3.1 事实 2/4），SM 段 cube 空闲、G2 段 vector 空闲；但 AO(t-1)∥G1(t) 已重叠（§3.1 事实 3），launch 后首 task 之外每 task 只暴露 3 段串行（Q+G1、SM、G2），非 4 段 |
| "wall = N × (4 相位 busy + 4 次交接)" | **量级正确、结构需修正** | 实测 P1=5.27µs/task ≈ Σ(主要流水忙时)+暴露交接；但 AO 已被掩盖、Q/K/V 装载部分与计算并行 |
| "flat 4 相位跨任务交错可隐藏交接" | **方向正确，收益有界** | 稳态下界 = max(vec 2.6, scalar 2.7-3.2, cube+mte 链) ≈ 2.7-3.2µs/task，非 max(cube,vector)≈2.6 单独决定；scalar 发射与 vector 同量级（§2.1），flat 不减 scalar 工作量 |
| 隐含"所有工况都受益" | **不成立** | 1 task/core（默认 shape）无跨任务对象；fixed 开销 F≈9µs 在浅 task 工况占比 30%+，flat 无法触及 |

**剩余可重叠空间（占 wall 百分比，推断）**：
- c3 类：P1 5.27 → flat 稳态 ~2.7-3.8µs/task（取 ns=2 实证 3.76 为保守值、
  vec/scalar 忙时下界 2.7 为乐观值）→ wall 77.4 → ~47-58µs，**收益 25-40%**；
- L0-c1：30.04 → ~9+4×(2.7..3.8) = 19.8-24.2µs，**收益 19-34%**（F 占比大，收益被稀释）；
- c4 类：任务边界仍有 drain/refill（vec 55% 未满），跨任务 flat 预计再收
  204→150-175µs（1.17-1.36×，推断，无 ns>2 同 tile 对照点，置信度低于上两行）；
- 默认 shape：**0%**（不适用）。

---

## 4. Step 3 资源与框架约束核算

### 4.1 flat 化目标态（评估对象的具体化）

把每核的 `(t, k)` 二元循环压平为单索引流 `u = flatten(t, k)`（每核长度
`Σ_t kv_useful(t)`），ws 槽位改按 `u % ns_flat`（ns_flat ≥ 2）索引，C 侧发射序变为
`… G1(u) → G2(u-1) → G1(u+1) → G2(u) …`，V 侧变为 `… SM(u) → AO(u-2) → SM(u+1) → AO(u-1) …`，
即用户目标的 4 相位错拍 `G1(i+1) | SM(i) | G2(i-1) | AO(i-2)`。

### 4.2 片上/片外容量核算（flat 增量 vs 现額，均按 g2stack 实际 tile：block_M=64、
M_stack=128、block_N=D=128、half=32、fp16 数据/fp32 累加）

硬件包络（事实，来源 `src/target/codegen_ascend.cc:34-38`、`src/transform/ascend_memory_planning.cc:35-39`、
performance-antipatterns.md L31-38 表，三处一致）：
L0A=L0B=64KB，L0C=128KB，L1=512KB(524032B)，UB=192KB(196352B)/vid，L2≈192MB。

| 存储 | 现额占用（编译产物实测，cache c0468f48 kernel.cu L30-59） | flat 增量 | flat 后 | 判定 |
|---|---|---|---|---|
| L0A | [2,128,128]fp16 = 64KB（**恰满包络**） | 0（G1(u)/G2(u-1) 用相反 parity 槽，无冲突） | 64KB | ✓ |
| L0B | [2,128,128]fp16 = 64KB（恰满） | 0（同上） | 64KB | ✓ |
| L0C | [2,128,128]fp32 = 128KB（**恰满 131072B envelope**） | 0（G1 写 l0c[u%2]、G2 写 l0c[(u-1)%2]） | 128KB | ✓ |
| L1 | q 32K + k0/k1 64K + v0/v1 64K + p 32K = 192KB | 0（q_l1 保持单槽，见 §4.3 约束 C1；可选 p_l1 双槽 +32KB 仍 ✓） | 192–224KB / 512KB | ✓ |
| UB/vid | acc_o 32K + io_buf 32K + work 16K + buf_2d 16K + mask_v 16K + acc_s_half 8K + 状态向量 ~1.5K + tmp 8K ≈ **130KB**（末 buffer 结束于 132352B 偏移） | ~0：acc_o/neg_sm/sumexp 单套即可（V 程序序内 AO(t)+finalize(t) 与 AO(t+1) 不交叠、SM 不触 acc_o——§4.4 论证）；sumexp_is/r_factors 已按 ns 维（ns_flat=2 → +0.5KB） | ~131KB / 192KB | ✓ |
| GM ws | ws_s 20×1×64K + ws_p 20×1×32K + ws_o 20×1×64K = 3.2MB | ns_flat=2 → 6.5MB；ns_flat=4 → 13MB | ≤13MB ≪ L2 192MB | ✓（L2 常驻，antipatterns.md L293：ws < L2 可享 L2 带宽） |

**结论：4 相位错拍所需的全部缓冲放得下，且 L0 三件套零增量**（parity 槽天然错开）。
唯一"放不下"的是 q_l1 双缓冲所需的新 flag id（§4.3 C1），但有零成本绕行。

### 4.3 框架/硬件同步资源约束（查证事实，非记忆）

- **C1 本地 event id 已用满**：SetFlag/WaitFlag(ISASI) eventID 范围 0–7，且
  **"事件 ID 不能使用 6 和 7（可能与内部使用的事件 ID 冲突）"**（CANN API 文档
  SetFlag/WaitFlag(ISASI) 约束说明，RAG 检索确认）→ 每 pipe 对实际可用 0–5 共 6 个。
  g2stack 的 MTE1↔MTE2 双向 id 空间均已占满（K0/1、V2/3、P4、Q5，gqa.py:2582-2590）。
  → **q_l1 无法双缓冲**。绕行（零新增 id）：保持单槽 q_l1 + 现有 SIG_Q_L1 协议；
  Q(t+1) 的 MTE2 装载在 MTE1 完成 q_l1→l0a 复制后即释放（MTE1_MTE2 id5 set，
  gqa.py:2797 语义前移到 flat 流中每 task 的最后一个 G1 l0a 复制后），装载窗口与
  G1(t) mma / G2(t-1) 重叠 → 基本隐藏（推断，需 sim 验证）。
- **C2 cross-core flag id 有余量**：CrossCoreSetFlag flagId 范围 **0–10**（同 RAG 文档），
  现用 0–5（SEM_WS1/2/3 × C2V/V2C）→ 剩 6–10 共 5 个可用于 flat 的槽位粒度信号
  （如 ws_s/ws_o 双槽各半计数）或保持计数语义不加新 id。
- **C3 同一 flagId 计数器最多 15 次设置**（同文档约束）：flat 设计中 set/wait 成对、
  错拍深度 ≤ ns_flat+2 ≤ 6 → pending 计数 ≤6 <15 ✓（现有 dense_expert ns=16 深度批次
  已在真机通过精度测试，佐证计数语义为 pending 深度而非终身累计——推断，标注）。
- **C4 T.Pipelined 不可用**：(a) 不支持嵌套（fa_opt 优化文档 L255"重要：T.pipelined
  不支持嵌套使用"；antipatterns.md L336 同）；(b) causal 运行时 skip 谓词在 Pipelined
  内会 **segfault 编译器**（gqa.py:117-121 仓库实测记录）；(c) expert kernel 的
  C/V 双 scope + 手动 flag 结构与 Pipelined 的自动 stage 改写不兼容（P28_EXPERT_PASS_CONFIGS
  已关全部 AUTO pass）。→ **flat 只能手写在 T.serial 循环层**（与现有 expert kernel 同一
  表达层，无新原语需求；antipatterns.md L336 也明示"核内 double buffer 推荐手写 flat pattern"）。
- **C5 内存层级合规（AGENTS.md 原则 3）**：flat 不新增数据路径——C↔V 仍经
  L0C→FIX→GM(ws)→MTE2→UB、UB→MTE3→GM(ws_p)→MTE2→L1→MTE1→L0A，全部为既有合法层级
  路径，无 GM→L0 越级。

### 4.4 算法依赖：online softmax rescale 链是否阻断 G1(i+1) ∥ SM(i)？

**不阻断（代码事实 + 推导）：**
- `G1(u) = Q(t)·K(k)^T` 只读 q_l1/k_l1、写 ws_s[u%ns]，**不读任何 softmax 状态**
  （neg_sm/sumexp/r_factors/acc_o 均只在 V 侧，gqa.py:2892-2901/2924-2928）。
- rescale 链（neg_sm 滑窗 min、r_factors=exp(neg_sm[cur]−neg_sm[prv])、sumexp 累乘）
  全部沿 V 程序序 in-order 执行；flat 流中同 task 的 SM(u) 与 AO(u') 保持相对序
  （V 侧按 u 升序发射），跨 task 状态互不依赖（不同 (bz,kg,gb,bx) 的 softmax 独立）。
- kv_iters=1 时 r_factors 恒为哨兵下溢 0、acc_o 零初始（gqa.py:2532-2535 注释：
  "ns==1 single-batch is harmless"）→ AO(t) 即"finalize 前的单次累加"，
  acc_o 单套缓冲 + 每 task 首个 AO 前 fill 即可（liveness：AO(t,k_last)+finalize(t)
  在 V 序中先于 AO(t+1,k_0)，中间穿插的 SM 不触 acc_o）✓。
- causal 变长（kv_useful(t) 随 bx 变）：flat 流长度逐 task 变化，C/V 两侧用**同一
  公式**（gqa.py:2694-2699 与 2835-2840 已镜像）生成一致的 u 序列——现有不变式
  "identical batch_iters sequence → flag pair counts match" 平移为
  "identical flattened sequence → set/wait counts match"，可保持。

### 4.5 对照实现检查（AGENTS.md：从示例入手）

- `examples/flash_attention/fa_opt/flash_attn_bhsd_expert_h16_d128.py`：与 g2stack 同构
  （task_range + `for t: for k: G1批→G2批`，L92/157/169/175/206），**无跨 task flat 先例**。
- `examples/pipeline/sparse_flash_attn_gqa_pipeline.py`：AUTO 模式 `T.Pipelined(NI, num_stages=2)`
  （L128）——核间流水靠 Pipelined，仅 kv 维，不跨 task。
- `examples/pipeline/flash_attn_bshd_pipeline.py` 等同目录：均为 Pipelined kv 维流水。
- → **flat 跨任务流水在本仓库无现成模板**，属于新模式实现（风险计入 §5.2），
  但其全部构件（ws 多槽、cross sem、parity l0、task_range）都已在 g2stack 内验证过。

---

## 5. Step 4 结论与动手方案（本轮不实施）

### 5.1 三选一判定：**部分适用（限定工况）**

1. **适用**：g2stack 派发域中 **每核 ≥3 task 且 kv_iters 小（1–2）** 的 prefill 工况
   ——L0-c1（收益 19–34%，推断）、c3 类（25–40%，推断，**首选验证 shape**）、
   c4 类（任务边界扩展，17–36% 折算 1.17–1.36×，推断低置信）。
   论据：§3.1 结构空档（ns 坍缩）+ §3.2 实证（ns=2 批次 −28.6%/kv）+ §4 资源全放得下。
2. **不适用**：每核 1 task 的小 shape（默认 B=1 类，16 实例 < 20 核）——flat 的定义域
   （跨任务交错）为空；该工况走 §5.3 替代措施。
3. **不适用（无需做）**：decode 路径（dense_expert/d256，kv_iters ≥4、ns 至 16）——
   任务内深批次已提供交错深度，跨 task flat 增量边际（未测，推断；decode 每核 task 数
   由 B*N_kv*G_batches 决定，大 batch decode 可另行评估）。
4. 用户假设中"串行态"表述**部分失实**（AO∥G1 已重叠），但核心洞察成立：
   **SM 段（~2.6µs/task）暴露的 cube 空闲 + 单槽 ws 阻断的 G2(t+1)/SM(t+1) 前跑，
   是 ns=1 工况最大的可回收气泡**。
5. 附加发现（与 flat 独立）：**固定开销 F≈9µs/kernel**（R2/R3 两点拟合），在 L0-c1 占
   wall 30%——flat 无法触及，需单列优化项（§5.3-D）。

### 5.2 实施蓝图（若 Orchestrator 批准进入实施轮）

**改动对象**：新增 `_gqa_kernel_prefill_g2stack_flat`（复制 g2stack 改造，**不动现有
kernel**，符合最小改动/回滚原则），dispatch 侧在 `gqa_expert` specs 中于 `g2stack` 之前
插入候选，gate 条件：`use_g2_stack and kv_iters <= 2 and q_tasks >= 2`
（即每核 ≥2 task 才可能有跨任务交错；q_tasks<2 保持现 kernel 零行为变化）。

**循环结构 flat 化**（C 侧伪码，V 侧镜像）：
```
# 每核展平流：u = 0..U-1，task_of(u)/kv_of(u) 由 task_range(cid) + kv_useful(t) 公式
# 在 C/V 两侧用同一表达式推导（保持 flag 计数镜像不变式）
prologue: Qload(task(0)); G1(0) → ws_s[cid, 0]
for u in 1..U-1:
    if task_of(u) != task_of(u-1): Qload(task_of(u))      # 单槽 q_l1，SIG_Q_L1 协议不变
    G1(u)  → l0[u%2] → ws_s[cid, u%2]                    # K(u) 装载与 G2(u-1) 共用 MTE2/MTE1 队列
    G2(u-1) ← ws_p[cid, (u-1)%2] → ws_o[cid, (u-1)%2]    # V(u-1) 装载提前发射
epilogue: G2(U-1)
```
V 侧：`for u: SM(u) → ws_p[cid,u%2]；if u>=1: AO(u-1)[+ 若 kv_of(u-1) 为该 task 末次:
finalize+Output]`（错拍深度 1 即可让 SM(u) ∥ G2(u-1) ∥ AO(u-1) ∥ G1(u) 四相位共存；
深度 2 的 G1(u+1) 前跑由 ws_s 双槽自动解锁，无需显式 lookahead）。

**缓冲与同步规划**：
- ws_s/ws_p/ws_o：`num_stages=2`（工厂传 ns_flat=2；workspace_idx 自动扩容，GM 6.5MB）。
- l0a/l0b/l0c：沿用 `[2,...]`，parity = u%2（G1/G2 天然错开）。
- cross sem：沿用 6 个计数信号量；set/wait 粒度从 ci 批内改为**每 u 一次**
  （ci=1 语义），错拍深度 ≤4 → 计数安全（C3）。若实测同步过密，再评估 id 6–10 拆双槽信号（C2）。
- 本地 flag：K/V 槽 parity 改 u%2；P/Q/MASK/IO/S_HALF 协议原样保留（C1 约束下零新增 id）。
- causal：kv_useful(t) 进入展平流长度；C/V 两侧同公式（沿用 gqa.py:2694-2699 模式）。
  全 masked task（kv_useful=0）在流中贡献 0 个 u，两侧一致跳过。

**预期收益量级（推断，依据 §3.2 模型）**：c3 类 77.4 → 47-58µs（1.33-1.65×）；
L0-c1 30.0 → 20-24µs（1.25-1.5×）；c4 类 204 → 150-175µs（1.17-1.36×）。
达到条件：scalar 发射不恶化（flat 循环控制每 u 摊销）+ Q 装载隐藏成立。

**风险点（按概率排序）**：
1. C/V flag 计数失配 → 死锁或数据腐蚀（本 kernel 家族历史最高危 bug 类，
   gqa.py:2516-2520 注释）。缓解：变体先在非 causal + 整除 shape 验证，再开 causal；
   编写 set/wait 配对静态断言脚本（perf_tuning/ 下）。
2. ci=1 每 u 同步过密 → scalar 开销回升吃掉收益（gqa.py:3546-3547 已记录 ci=1
   "over-synchronizes" 于任务内场景）。缓解：备选"每 2u 成组" 的 ci=2 flat 变体。
3. q_l1 单槽在任务切换处的 MTE1 释放时序 → Q 装载暴露。缓解：sim trace 验证；
   必要时把 q_l1→l0a 复制提前到 G1(u) 头部（现 `if i<2` 模式平移）。
4. task_range 不均衡（4/3）依旧 → flat 收益被热点核稀释。独立问题，见 §5.3-C。
5. 编译器对展平后更长循环体的 codegen 差异（scalar 指令膨胀）。缓解：get_kernel_source()
   对比指令数。

**分步验证门（每步精度不回退才继续）**：
- Gate 0：变体文件 `perf_tuning/gqa_flat_variant.py` 独立编译（不改 gqa.py），
  `python examples/gqa/test_gqa.py --level all` 用变体 monkey-patch 跑通**不适用**——
  契约禁止改测试；改为变体自带 runner：`perf_tuning/run_flat_variant.py` 内联
  test_gqa 同源 golden（_ref_gqa 逻辑复制）跑 L0 全集 shape。
- Gate 1（编译+精度）：变体在 c3 类 shape 精度通过（vs fp32 golden，atol/rtol=1e-2 口径同 test_gqa）。
- Gate 2（性能）：`msprof op --kernel-name=main_kernel` 采变体 vs 原 kernel 同 shape，
  Task Duration 下降 ≥10% 且 cube/vec 利用率上升（PipeUtilization 对照 §2.1 表）。
- Gate 3（回归）：变体接入 dispatch gate 后，仓库根
  `unset PYTHONPATH; export ACL_OP_INIT_MODE=1; python examples/gqa/test_gqa.py --level all`
  全绿；msprof 复采 R1-R4 四工况确认无回退（默认 shape 因 gate 条件不派发变体，须逐位不变）。
- 任一门失败：保留数据、回退变体，按失败分类规则记录（本轮规则：性能下降/无变化即弃）。

### 5.3 替代/并行措施（flat 不适用区 + 独立收益项）

- **A. 默认小 shape（1 task/core 延迟区）**：
  1) **V 装载预取**：`_gqa_kernel_expert` 中 `T.copy(V..., v_l1)`（gqa.py:275）上移到
     softmax 块之前（数据依赖上 v_l1 仅被 G2 消费；AUTO sync 会按依赖插 flag）。
     sim 证据：现 V 装载 0.237µs 完全串行在 SM→G2 交接后（§3.3 8.600 行）；
     真机预计省 0.3-0.6µs/task（GM 延迟更大）。改动一行级，低风险，**建议先做**。
  2) scalar 开销审视：R1 aic_scalar 43-54% 为全表最高——AUTO 模式 threads=2 的
     逐指令 BAR（sim trace V 侧 ~100 个 BAR 事件）是嫌疑源；专家化小 shape 变体
     曾被实测"net-negative below ~20 instances"（gqa.py:3246-3247），若做需重新实证。
  3) 4 个闲置核无法利用是 task 数物理上限（16 实例），非缺陷；如需填满需拆
     block_M=32（32 task）——但每 task 交接开销不缩小，推断净负（未测，标注推断）。
- **B. g2stack 任务内深挖（不依赖 flat）**：kv_iters≥2 工况把 ns 从 min(kv,8) 提到
  与 ci 联动调参（c4 类 fixpipe 24.8-27.1% 提示 FIX→GM ws 写开始拥塞，ns 加大需
  连带评估 ws 带宽）——属现有参数空间，改动最小。
- **C. 任务均衡**：64=4×4+16×3 的 +33% 热点核税。可选：task_range 改
  "余数任务分给轻核"无济于事（已是整数最优）；根治需更细粒度 task（per-head 化
  128 task → 6.4/core，但 per-head 单 task 效率低于 stacked，R1 vs R2 对比可证）
  或动态窃取（框架无原语，成本高）。**判定：维持现状，热点税计入 flat 收益折算**。
- **D. 固定开销 F≈9µs**：拆解 launch/灌注/排空占比需 sim trace（§1.3 数据落盘后补）；
  候选措施：首 task K/V 预取提前到 prologue、init flag 链精简。独立小项。

---

## 6. 未验证项 / 风险披露（如实报告）

1. **L0-c1 g2stack 的 sim 指令级 trace 未获得**（采集 75 分钟后中止，§1.3）——
   "AO(t-1)∥G1(t) 已重叠" 与 "每 task 气泡 ≈2.1µs" 依据代码依赖链推导 +
   真机利用率拟合，**非逐指令实测**；实施轮开工前建议以差分法补证
   （真实 NPU 上对同一 shape 采 `--launch-count` 多次或对比变体 kernel，
   避免 camodel 跨核 flag 仿真吞吐问题）。
2. flat 收益数字全部为**模型推断**（P1/P2/F 拟合 + 忙时下界），未实测；
   c4 类外推置信度最低（无 ns>2 对照点）。
3. "cross-flag 计数器 15 次上限 = pending 深度语义" 为**推断**（依据：现有 ns=16
   decode kernel 真机通过精度测试，累计 set 次数远超 15）；文档原文"同一 flagId 的
   计数器最多设置 15 次"未明确累计/在途语义。flat 实施时保持错拍深度 ≤6 以规避。
4. R3 用 bf16（c3 同 dtype）而 R1/R2 用 fp16：跨 dtype 的每 task 成本视为相同
   （同 tile 同指令数，fp16/bf16 在 cube/vector 上吞吐一致——推断，未单独 AB 测）。
5. PipeUtilization 的 scalar_ratio 是否含等待周期未从文档确认——若含，则 §3.2
   "scalar 与 vector 同为下界"偏悲观（flat 收益上限更高）；若不含则相反。两可情形
   均不改变 verdict 方向。
6. F≈9µs 的构成（launch vs 灌注 vs 冷装载）未拆解；且 `wall = F + N×P` 双参数模型
   仅由 R2/R3 两点（4 与 13 task/core）**恰好定解、无残差校验**——F 与 P1 的分离
   未被第三个独立 ns=1 数据点验证（R4 为 ns=2，P 不同）。若 F 实际随 task 数弱变化，
   §3.4/§5.2 的收益区间会平移；实施轮应以变体 vs 原 kernel 的**同 shape 直接对比**
   （Gate 2）为准，不以本模型外推值为准。
7. 真实 NPU 上 msprof 的 Task Duration 含 profiling 注入开销的绝对量未标定
   （同一 kernel 的 sim wall 5.10µs vs 真机 9.94µs 差异主要来自 GM 延迟建模，
   非全部为 profiling 开销——推断）。

---

## 7. 附录

### 7.1 g2stack L0-c1 sim trace 补充（采集中止，未产出）

本轮未获得指令级 trace（§1.3 中止说明）。task 边界 drain/refill 的逐段量化移交实施轮：
优先用真实 NPU 差分法（同 shape 变体 vs 原 kernel 的 PipeUtilization + Task Duration 对比，
Gate 2 已含），必要时再尝试 camodel（可考虑缩小 shape 至 2 核可承载的 task 数以规避仿真吞吐问题）。

### 7.2 反模式对照记录（agent 契约要求）

对照 `tilelang-perf-optimization/references/performance-antipatterns.md`：

| 条目 | 现状 | 处置 |
|---|---|---|
| 关注项 A（任务数>核数仍按任务数 launch） | expert 系全部 Fixed-Core(20)+task_range ✓；**per-head AUTO ladder 按 block_num launch**——默认 shape 16<20 不触发；c5 类 D=256 shape block_num=128>20 **触发** | **有意保留**：per-head ladder 是 D≠128/不适格 shape 的 AUTO 回退路径，无 ws/task_range 机制，Fixed-Core 化 = 重写该 kernel；收益面窄（仅 D=256 prefill），不在本轮范围。记录在案 |
| 关注项 B（任务数<核数仍按 24 launch） | 不触发（per-head 按 16 launch；expert 有 ≥20 实例 gate） | ✓ |
| CV overlap 未开启 | per-head AUTO 四 pass 全开 ✓；expert 手动 sem ✓（深度不足是本报告主题） | ✓ |
| 冗余全局同步 | expert 全定向 flag，无 barrier_all ✓ | ✓ |
| 指令拼接未融合 | g2stack softmax 已用 axpy（gqa.py:2897）✓ | ✓ |
| tile 过小 / 过大 | stacked M=128 恰满 L0A/L0C 包络（设计注释 gqa.py:2500-2503）✓ | ✓ |
| ws 超 L2 | 3.3MB（flat 后 ≤13MB）≪192MB ✓ | ✓ |
| Vector 逐元素 for loop / 多行 tile 逐块归约 / 正交轴串行化 / CV_COMBINE 误分核 | 未发现（softmax 为 tile 级原语；批量 2-phase 形式已把 rescale 归并到 phase2） | ✓ |

### 7.3 本轮 skills / 参考文档查阅清单

- `tilelang-perf-optimization/SKILL.md`（工作流、msprof 口径）
- `tilelang-perf-optimization/references/best-practices/flash_attn_optimize.md`
  （num_stages 批次流水、细粒度 flag、cross-core semaphore、静态任务分配、批量 softmax
  ——g2stack 即该模板族；BSND/msprof 采集口径）
- `tilelang-perf-optimization/references/performance-antipatterns.md`（全清单对照，§7.2；
  硬件容量表 L31-38；L2/ws 设计要点 L293；L336 "手写 flat pattern" 提法）
- `tilelang-perf-optimization/references/optimization-guide.md`（目录级筛查：§2.2 double
  buffer、§2.9 Fixed Core、§2.10 pass_configs、§2.12 host 侧——现状均合规）
- `tilelang-api-best-practices`（SKILL.md 索引 + references/api-schedule-sync.md
  set_cross_flag mode 语义 L206-227）
- CANN RAG：CrossCoreSetFlag(ISASI)（flagId 0-10、计数 15 上限、mode 语义）、
  SetFlag/WaitFlag(ISASI)（eventID 0-7、6/7 保留禁用）
- 仓库源码：`src/target/codegen_ascend.cc:34-38`、`src/transform/ascend_memory_planning.cc:35-39`
  （容量包络）；`tilelang/language/ascend.py:191-231`（set/wait_cross_flag 签名与 mode 默认值）
- 对照实现：`examples/flash_attention/fa_opt/flash_attn_bhsd_expert_h16_d128.py`、
  `examples/pipeline/sparse_flash_attn_gqa_pipeline.py`

### 7.4 产出物清单

- 本报告：`examples/gqa/perf_tuning/flat_pipeline_analysis.md`
- 真机采集：`perf_tuning/msprof_op_default/OPPROF_20261016173026_AHDMSLSSWRRTUPXX/`、
  `msprof_op_l0c1/OPPROF_20261016173054_HIGYPGUEBTPGXPNG/`、
  `msprof_op_c3like/OPPROF_20261016180317_LORUJDGRRJOJRUNC/`、
  `msprof_op_c4like/OPPROF_20261016180404_HQYZBYLGCJKNBCZK/`（各含 PipeUtilization.csv、OpBasicInfo.csv、dump/）
- sim 采集（**中止，未产出 trace**）：`perf_tuning/msprof_sim_l0c1/OPPROF_20261016173720_YHUCFYARSETWMIFK/device0/tmp_dump/`（44KB 残留）+ `msprof_sim_l0c1_run.log`（含中止前完整日志）
- 基线记录：`perf_tuning/baseline_iter1.json`；轮次日志：`perf_tuning/perf_log.md`

---

## 8. iter2 实施结果回填（2026-10-17，mode=implement）

本节为 Stage 3 iteration 2（范围 A+B，用户批准）执行后的证据回填；详细数据见
`candidate_iter2_A.json` / `candidate_iter2_B.json` / `perf_log.md`。

### 8.1 verdict 实证确认

- **B（flat 主项）已实施并采纳**：`_gqa_kernel_prefill_g2stack_flat` 接入
  gqa.py dispatch（gate=`use_g2_stack and kv_iters==1 and stacked>=2*NUM_CORES`，
  g2stack 保留为编译回退）。真机 msprof 同会话交替采样：
  - R3 c3 类（13 task/core，B1 首选）：78.94µs → 60.51µs，**−23.3%**
    （§5.2 预测 1.35-1.65×，实测 1.30×，落在预测带下沿外侧附近）；
  - R2 L0-c1（4 task/core，补充）：28.88µs → 26.55µs，**−8.1%**
    （§3.4 预测 19-34% 未全兑现——固定开销 F 稀释，方向与"收益受限"判定一致）；
  - B0 独立 runner 7/7 与 shipped kernel **逐位一致**；B2' 全套件 24/24 PASS；
    B3 gate 外 4 形状逐位不变、R1 噪声带内。
- **A（V 装载预取）回滚**：同会话交替 7+7 样本，候选 10.95µs vs BASE 10.53µs
  （+4.0%，噪声带内偏负，未达 >3% 改善门槛）。§5.3-A1 的 sim 证据（0.237µs
  暴露装载）在真机 AUTO codegen 下未转化为收益（原因未验证，标注推断：
  MTE2 提前发射与 veccore mask 装载争用 GM/L2 路径，或 AUTO sync 重新串行化）。
- flat 拟合模型更新：P_flat ≈ 3.9µs/task（vs ns=1 的 5.27、ns=2 的 3.76/kv），
  F ≈ 10.3µs——残余 ~1µs/task 高于 vector-busy 下界（≈2.6-2.9µs），归因交接/
  flag 往返延迟（**推断**，无指令级 trace 佐证，见 §6 项 1）。

### 8.2 iter1 未验证项的证据更新（硬性约束 5）

- **§6 项 3（cross-flag 15 次上限语义）：已实证解决** —— shipped g2stack 与
  flat 变体在 mc=26 工况（c4 类）每信号量 set 26+ 次于真机正常运行且精度通过
  ⇒ 上限为 **pending 深度**（在途 set−wait 差）而非终身累计。flat 变体错拍
  深度 ≤3，安全。
- §6 项 6（F≈9µs 构成）：仍未拆解（flat 后 F≈10.3µs，量级不变）；
  R2 类浅 task 工况的最大剩余优化项。
- §6 项 1（L0-c1 指令级 trace）：camodel 采集 75min 中止后未重试；
  "AO∥G1 已重叠"推导获得**间接实证**（flat 实测收益量级与"SM 暴露为主、
  AO 已隐藏"的模型自洽：若 AO 也未隐藏，flat 收益应显著更大）。
- §6 项 2（收益为模型推断）：R3/R2 已由真机实测取代；R4（kv=2）gate 未覆盖，
  维持推断状态。

### 8.3 蓝图偏差记录（代码现实优先）

1. **gate 收紧**：任务书/蓝图 gate 为 `kv_iters<=2`，实现为 `kv_iters==1`
   特化（展平流退化为任务序列、kv_useful≡1 可证、免除 causal 块跳过机制）；
   kv==2 由 shipped g2stack ns=2 任务内批次继续服务（其 −28.6%/kv 收益 iter1
   已实测），变长双流展平未构建未验证，不接入。
2. **免 fill 优化回退**：首版依赖 r_factors 下溢自复位（有限输入逐位等价），
   但 NaN/Inf 输入经同奇偶状态槽跨任务泄漏（boundary_sparse_special
   nan_pos_mismatch=512）→ 恢复 AO 头部的按任务 acc_o/sumexp fill（V 序安全，
   算术序列与 shipped 完全一致），全套件转绿。
3. **框架层新发现**（蓝图未预见，均已解决并记录）：prim_func 内嵌 def 不被
   parser 接受 → @T.macro 显式传参；含 T.if_then_else 的 my_count±k 表达式
   不得内联进 tile-op 索引/SetFlag id → T.alloc_var 语句位绑定。
