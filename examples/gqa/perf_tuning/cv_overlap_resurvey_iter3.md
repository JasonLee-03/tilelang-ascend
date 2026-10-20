# GQA CV 并行掩盖重梳理报告（Stage 3 · round 2 · iteration 3 · analysis_only）

- 日期：2026-10-17（测量会话 21:04–21:24，单会话完成全部 24 组 msprof 采集）
- 对象：`examples/gqa/gqa.py` shipped 版（md5 `cbce74410fee5f615489d0391b3ea620`，测量前后核验一致，本轮**未修改**
  `gqa.py` / `test_gqa.py`）
- 用户诉求："CV 核之间的空闲等待太多，能不能 CV 之间进行并行掩盖，提升整体性能"
- 本轮任务：重测瓶颈分布（含新增 R5 decode / R6 c2-like 工况）+ 产出按优先级排序的 CV 掩盖候选清单；
  **不实施**，用户确认优先级后进入 iter4+
- 方法学纪律：全部真实 NPU（未用 camodel，遵守 iter1 禁令）；A/B 同会话交替 ≥3+3；
  `unset PYTHONPATH` + `ACL_OP_INIT_MODE=1` + `LD_LIBRARY_PATH=/usr/local/sqlite3/lib:...`（sqlite 导出正常，
  全部 24 组均产出非零 CSV）；21 组 main_kernel 采集全部 1650/1650 MHz 额定频率；批后 npu-smi 健康、无残留进程

---

## 0. 结论速览（verdict）

**能，且空间集中在两个派发域；decode 域实测排除；小 shape per-head 域杠杆已弱。**

| 域 | 工况 | CV 空闲现状（本轮实测） | 可掩盖空间 | 首选候选 |
|---|---|---|---|---|
| flat（kv==1，iter2 已流水化） | R2/R3/R6-a（bench c1/c2/c3/c18/c20） | 残余互等：V 等 ws_p 释放 1.67–2.06µs/u + C 等 ws_s 释放 1.76–2.81µs/u（2 槽环往返耦合） | wall −4..−15%/shape（模型外推） | **C1：ws 环 2→4 槽（ns_flat=4）** |
| g2stack ns=2（kv==2） | R4（bench c4/c7） | **shipped 代码中最大 CV 串行区**：C 等 V softmax 输出 = wall 的 53%，V 等 C G2 输出 = 49%（互替占用签名） | wall −13..−21%（实测外推） | **C2：flat 双流展平（uniform-kv 子类先行）** |
| decode dense_expert（ns=16） | R5（bench c11 类） | V 空闲 72% 窗口，但根因是 **C 侧 MTE2 87–96% 饱和**（K/V GM→L1 带宽），非交接延迟 | ≈0（CV 掩盖无法创造带宽） | 不立项（iter1 推断 → **本轮实测证实**） |
| per-head AUTO（1 task/core） | R1（bench c1 类小 shape 及默认） | C 侧 flag 等待 ≈36–40% wall；launch/排空 ≈3.3–3.8µs（31–35%） | 低（A 项预取已失败；专家化记录为净负*） | 不推荐立项 |

\* 引用核对：gqa.py 中 "net-negative below ~4 KV iterations or ~20 instances"（现 gqa.py:3697–3700）为
**decode dense expert 语境**的实测记录，非 prefill per-head 直接实证——iter1 §5.3-A2 的引用需以此澄清；
prefill 侧仅有 `use_expert_prefill ≥20 实例` 的 gate 惯例（gqa.py:3938–3944），R1（16 实例）上
prefill_expert 形式**无直接 A/B 实测记录**（未验证项 §7-10）。

**R6 c2 回归复核结论：不成立（澄清）。** 同会话交替 A/B：flat 27.53µs（散布 1.0%） vs 强制 g2stack 回退
32.27µs（散布 1.3%）→ **flat 快 14.7%**。评测会话自身的 main_kernel 5 次迭代带（min 27.46 / avg 31.08 /
max 32.24µs，op_statistic.csv）同时覆盖 29.06 与 31.82 两个跨会话数字 → "+9.5% 回归"属评测会话内/跨会话
方差，非 flat gate 内代码回归。**不建议收窄 gate。**

---

## 1. T1 事实核对：dispatch 链与 gate（现行号）

派发入口：`_resolve_kernel_chain`（gqa.py:3621，按 specs 顺序编译、首个成功者缓存）+
`gqa_expert`（gqa.py:3645）。行号为当前 shipped 版实测（iter1/iter2 记录的 3168/3192 已漂移）。

### 1.1 prefill（S>4）specs 顺序（gqa.py:3919–4138）

1. `use_expert_prefill`（gqa.py:3938–3944）= D==128 ∧ fp16/bf16 ∧ (kv==1 ∨ 偶数 kv≥2) ∧
   (非 causal ∨ S_kv≥S) ∧ B·N_q·q_blocks ≥ 20
2. `use_g2_stack`（gqa.py:3956）= use_expert_prefill ∧ G 偶 ∧ B·N_kv·(G/2)·q_blocks ≥ 20
3. **flat gate**（gqa.py:3981–3982）：`kv_iters == 1 and flat_block_num >= 2*P28_NUM_CORES(=40)`
   → `g2stack_flat` spec 排第一；`g2stack(ns=min(kv,8), ci=2)` 紧随其后为编译回退（gqa.py:4006–4031）
4. `prefill_expert`（gqa.py:4033–4064）→ per-head AUTO ladder `_perhead_bm_ladder`
   （gqa.py:3616–3618, 4079–4105）→ `g2` 末位回退（gqa.py:4112–4135）

### 1.2 decode（1≤S≤4，gqa.py:3688–3813）

`use_expert`（gqa.py:3701–3708）= D==128 ∧ kv≥4 偶 ∧ G%G_inner==0 ∧ B·N_kv·G_batches ≥ 20 →
`dense_expert(ns=min(16,kv), ci=4 if kv≥16∧M_tile≤32 else 2)`；D256 → `dense_expert_d256`；否则 `dense_auto`。

### 1.3 六工况派发实证（probe_dispatch_iter3.py 实测输出，非推断）

| 工况 | shape | 派发（chosen） | tasks/core | 备注 |
|---|---|---|---|---|
| R1 | B1 S128 Skv128 Nq8 Nkv2 causal fp16 | `per_head_bm64` | 1（16 blocks，4 核闲置） | 16 < 20 → 非 expert |
| R2 | B2 S128 Nq32 Nkv8 causal fp16 | `g2stack_flat` | 4/3（64 tasks） | = bench c1 |
| R3 | B2 S128 Nq128 Nkv8 causal bf16 | `g2stack_flat` | 13/12（256 tasks） | = bench c3 |
| R4 | B4 S256 Skv256 Nq64 Nkv8 noncausal bf16 | `g2stack`（ns=2） | 26/25（512 tasks） | = bench c4，kv==2 → flat gate 外 |
| R5 | B16 S1 Skv2048 Nq32 Nkv8 noncausal fp16 | `dense_expert`（ns=16, ci=4） | 7/6（128 tasks） | = bench c11 |
| R6 | B2 S128 Nq48 Nkv4 noncausal fp16 | `g2stack_flat`；强制回退臂 → `g2stack` | 5/4（96 tasks） | = bench c2（G12） |

bench 映射补充（cases.yaml 核对）：flat 域覆盖 **c1/c2/c3/c18/c20**（kv==1 且 block_num≥40；
c18 MQA Nkv1 → 2·1·16·2=64 ✓）；g2stack ns≥2 域覆盖 **c4/c7（kv2，均 noncausal→uniform）、c8（kv8 causal
变长）、c9（kv4 noncausal uniform）、c10（kv4 causal 变长）**；c5/c6（D256）与 c19（MHA G=1）走
per-head ladder / prefill_expert。**c7 实为 non-causal**（note "cross"，attrs is_causal: false）——
kv==2 的两个 bench 工况（c4/c7）都是 uniform kv_useful≡2，见 §5 C2 的风险降级。

flat kernel 结构核对（gqa.py:2982–3423）：C 侧稳态步 `G1(t+1)→G2(t)→Qload(t+2)`（3361–3364）、
V 侧稳态拍 `AO(jj)→FIN(jj)→SM(jj+2)`（3405–3408）、SEM ids 0–5（3028–3033）、ws 槽 = 任务奇偶
（`par/side = x % 2`）、V 侧状态槽按任务奇偶双份 + 每任务 fill（3226–3240，S3-iter2-B fix）——与 iter2
交付记录一致，无漂移。

---

## 2. T2 真机重测矩阵（msprof op，PipeUtilization + Task Duration）

### 2.1 wall 与跨轮对照（main_kernel Task Duration µs）

| 工况 | iter1 (10-16) | iter2 (10-17 上午) | **iter3 本轮（3 样本均值 [min–max]）** | vs iter2 |
|---|---|---|---|---|
| R1 per-head | 9.94 | 10.53（BASE 均值） | **10.83 [10.70–10.90]** 散布 1.8% | +2.8%（会话漂移带内） |
| R2 flat | 30.04（g2stack） | 26.55 / 25.94（flat） | **26.13 [25.42–26.76]** 散布 5.1% | ≈0%（复现 iter2 采纳值） |
| R3 flat | 77.44（g2stack） | 60.51 / 60.92（flat） | **61.09 [60.64–61.84]** 散布 2.0% | +0.5%（复现） |
| R4 g2stack ns2 | 204.42 | — | **207.85 [206.94–209.24]** 散布 1.1% | +1.7%（漂移带内） |
| R5 decode（新） | 未测 | 未测 | **144.66 [144.18–144.94]** 散布 0.5% | 首测 |
| R6 c2 flat（新） | 未测 | 未测 | **27.53 [27.40–27.68]** 散布 1.0% | 首测 |
| R6 c2 g2stack（新） | 未测 | 未测 | **32.27 [32.02–32.44]** 散布 1.3% | 首测 |

flat 域今日 wall 与 iter2 采纳时一致（±0.5–2.8%），iter2 收益无隐性回退。评测 harness 口径差
（eval c1 avg 24.16 / c2 avg 31.08 / c4 avg 200.90 / c11 avg 149.81 vs 本轮 26.13/27.53/207.85/144.66）
为跨 harness 数字，按纪律**不作对比依据**（eval 每迭代间跑 CacheClean 刷 L2，疑致 cold-ws 偏慢——推断，
未验证，§7-8）。

### 2.2 利用率重测（min–max across cores；对照 iter1 §2.1 表）

| 流水 | R1 (16blk) | R2 flat | R3 flat | R4 ns2 | R5 decode | R6 flat | R6 g2stack |
|---|---|---|---|---|---|---|---|
| aic_cube | 3.6% | 7.7–10.7% | 13.1–14.6% | 15.7–16.7% | 8.6–10.1% | 9.9–12.4% | 8.4–10.5% |
| aic_mte2 | 9.2–25.1% | 13.7–26.9% | 24.6–32.8% | 23.9–30.4% | **80.9–96.3%** | 19.2–29.0% | 17.8–28.0% |
| aic_fixpipe | 6.2–8.0% | 11.4–16.3% | 20.4–24.2% | 24.1–27.4% | 25.4–30.7% | 14.3–20.5% | 12.1–16.8% |
| aic_scalar | 39.5–48.9% | 32.8–51.9% | 50.6–62.3% | 42.5–49.4% | 70.5–84.0% | 45.1–66.8% | 36.8–52.5% |
| aiv_vec | 15.2% | 33.5–46.2% | **57.6–63.4%** | 52.6–55.5% | 31.0–36.2% | 40.0–49.8% | 30.0–37.5% |
| aiv_mte2 | 5.1–17.1% | 9.6–15.7% | 18.1–27.7% | 15.9–24.3% | 21.4–28.1% | 10.2–18.1% | 7.4–15.7% |
| aiv_mte3 | 3.0–3.9% | 7.4–11.2% | 13.9–17.4% | 12.4–16.0% | 9.4–12.5% | 9.6–15.0% | 8.3–12.5% |
| aiv_scalar | 20.1–37.7% | 29.4–49.1% | 45.0–54.6% | 38.3–43.4% | 25.9–32.8% | 35.2–51.0% | 34.9–47.8% |

要点：flat 化后 R3 的 aiv_vec 从 iter1 的 41.6–45.1% 提升到 **57.6–63.4%**（最忙计算流水，pacer 侧），
aic_cube 仍仅 ~14%——cube 远未饱和，瓶颈不在算力；R2/R3 的 aic_scalar 升到 50–62%（flat 每 u 一次
握手 + 更长循环体发射）。R5 的 aic_mte2 87–96% 为**全矩阵唯一接近饱和的流水**（decode 带宽墙）。

### 2.3 wait-id 遥测（本轮新证据，T3 的核心输入）

PipeUtilization.csv 含分 cross-flag-id 的 scalar 等待时间列。语义映射经结构核对成立：expert 系 kernel
V 侧只 wait {id0=SEM_WS1_C2V(ws_s 就绪), id3=SEM_WS2_C2V(ws_p 已消费), id4=SEM_WS3_C2V(ws_o 就绪)}，
C 侧只 wait {id1=SEM_WS1_V2C(ws_s 已释放), id2=SEM_WS2_V2C(ws_p 就绪), id5=SEM_WS3_V2C(ws_o 已释放)}，
与 CSV 列一一对应（gqa.py:3028–3033 SEM 定义）。**这把 iter2 "残余 ~1µs/task 归因交接/flag 往返（推断）"
升级为分信号量实测定位。**

热点核等待时间（µs，跨 3 样本 × 全核取 max 的上界口径；样本间同字段波动 ≤15%，不影响结论量级；
括号内 = 每 u / 每 kv 折算，按热点核任务数）：

| 等待项 | R2 flat (4u) | R3 flat (13u) | R6 flat (5u) | R6 g2s (5t) | R4 ns2 (52kv) | R5 decode |
|---|---|---|---|---|---|---|
| C id1（等 V 释放 ws_s） | 9.83 (2.46/u) | **36.50 (2.81/u)** | 8.81 (1.76/u) | 2.65 (0.53/t) | 2.28 (0.04/kv) | 2.37 |
| C id2（等 V 产出 ws_p） | 4.33 (1.08/u) | 10.17 (0.78/u) | 5.63 (1.13/u) | **14.75 (2.95/t)** | **110.34 (2.12/kv)** | 7.63 |
| C id5（等 V 释放 ws_o） | 9.91 (2.48/u) | 14.29 (1.10/u) | 10.26 (2.05/u) | 10.99 (2.20/t) | 49.62 (0.95/kv) | —* |
| V id0（等 C 产出 ws_s） | 5.74 (1.44/u) | 9.49 (0.73/u) | 5.24 (1.05/u) | 7.85 (1.57/t) | 27.53 (0.53/kv) | **47.08** |
| V id3（等 C 消费 ws_p） | 8.25 (2.06/u) | **21.66 (1.67/u)** | 10.12 (2.02/u) | 2.34 (0.47/t) | 2.60 (0.05/kv) | 1.92 |
| V id4（等 C 产出 ws_o） | 3.14 (0.79/u) | 5.36 (0.41/u) | 5.64 (1.13/u) | **16.94 (3.39/t)** | **107.50 (2.07/kv)** | **55.38** |

\* R5 的 aic id5 列在 CSV 中为 NA（dense_expert ns=16 的 release-wait 折叠形式不同，未深究——非本轮对象）。

三个签名模式：
- **flat 域（R2/R3/R6-a）**：C 主等待 = id1（V 消费 ws_s 慢于 C 产出），V 主等待 = id3（C 消费 ws_p
  慢于 V 产出）——**双向都在等对方的"释放"信号** = 2 槽环的往返延迟耦合（V 是 pacer：R3 V 窗口 58.63 >
  C 56.91；V vec 62–63%）。
- **g2stack kv==2（R4）**：C 主等待 = id2（V 的 softmax 产出），V 主等待 = id4（C 的 G2 产出）——
  **双向都在等对方的"新数据"** = 相位互替串行（C 干活时 V 等、V 干活时 C 等），每 kv 各 ~2.1µs，
  合计占 wall ~50%+。这是任务内批次窗口（ns=2 封在 task 内）+ 任务边界 drain/refill 的叠加。
- **decode（R5）**：V 等 C 的数据（id0+id4 = 102µs/核 = 72% 窗口），但 C 不是"在等 V"——C 的 MTE2
  自身 87–96% 饱和。等待的根因是带宽而非握手。

---

## 3. T3(a)：flat 域残余交接的定位（不实施，给方案与预期）

**残余量化（R3 热点核，13 u，均值口径）**：V 窗口 4.48µs/u = vec 忙 2.93 + 暴露 ~1.55µs/u；
其中 V id3 1.67 + id0 0.73 + id4 0.41（等待间/与 vec 排空有重叠，不能直接求和——scalar 阻塞期间
已发射的 vec 指令仍在跑，故 1.55 为窗口差值口径、2.81 为等待列直接口径，两者为同一现象的上下界）。
C 窗口 4.35µs/u，其中 id1 阻塞 2.81µs/u。iter2 的 "残余 ~1µs/task" 即窗口差值口径的 1.55µs/u
（今日数据略高，同量级）。

**能否进一步定位？能，且本轮已完成大半：**
1. **已完成**：wait-id 分解把残余从"泛指的交接/flag 往返"定位到**具体信号量**——V-id3（等 C 的
   G2(t) MTE2 读完 ws_p[t%2]）与 C-id1（等 V 的 SM(x−2) 读完 ws_s[x%2]）互为镜像，均为 **2 槽环
   释放信号绕环一周的延迟**。
2. **纸面推导（offset-3 vs offset-2 vs 深环，基于现有数据即可判别）**：
   - 现行 V 拍序 AO(jj)→FIN(jj)→SM(jj+2)：SM(x) 的 id3 门 = C step x−2 中 G2(x−2) 的 MTE2 读，
     而 C step x−2 的头部是 G1(x−1)——其 id1 等待实测 2.81µs/u，即 G2 的 ws_p 读被排在一个
     ~2.8µs 的阻塞段之后 → 放行 SM(x) 平均比 V 需要晚 ~1.7µs。**V-id3 等待的机制解释闭合**
     （推导，基于实测等待值的时序拼装，非指令级 trace）。
   - offset-2（SM 提前一拍）**不可行**：ws_p 仍只有 2 槽，SM(x) 需要 G2(x−2) 先读走 ws_p[x%2]，
     提前发射只会把 id3 等待搬到更早的指令位置，wall 不变（纸面判定，无需实测）。
   - **深环（4 槽）**：SM(x) 的门变为 G2(x−4)（早两拍完成）→ id3 期望 →≈0；C 的 G1(x+1) 门变为
     SM(x−3) → id1 期望大幅下降；稳态周期从 "忙+往返残余" 变为 pacer（V 侧）纯吞吐
     ≈ vec 2.93µs/u + 发射残差 → **R3 wall 预期 51.8–56.9µs（−7..−15%）**（模型外推）。
3. **判别实验（iter4 Gate 2 附带）**：若 ns_flat=4 后 id3/id1 归零且 wall 下降 ≈ Σ残余 → 往返假设
   成立；若等待转移到 V-id0/C-id2（生产门）而 wall 不动 → V 侧吞吐已是下限，转向 V 侧工作量压缩
   （收益上限即 vec 忙时 38.04µs，R3 wall 的 62%）。两种结果都有明确后续动作，无浪费轮次。
4. **ci 粒度对照**：flat 的每 u 一次 set/wait 已是最细粒度；粗化（每 2u 成组）= 回到 g2stack 相位
   批式结构（R6-b 臂实测更慢 14.7%）→ **不立项**。

---

## 4. T3(b)：F≈10.3µs 固定开销拆解（本轮部分完成）

用 "TaskDur − 热点核窗口" 与 "窗口−N·P 两点拟合" 分离（R2/R3 同为 causal flat 域，P_ss 一致假设）：

| 成分 | 量值 | 依据 |
|---|---|---|
| launch/teardown（核外） | **2.5–3.8µs**（六工况一致：R1 3.30–3.78 / R2 2.70–2.85 / R3 2.54–3.21 / R4 2.81–3.11 / R5 2.50–3.07 / R6f 2.87–3.09） | 逐样本 TaskDur − 热点核窗口，直接实测 |
| 核内灌注+排空+首任务冷装载 | **≈7.8µs**（flat causal，均值口径；仅取 s2 样本对时 8.7µs，口径差 ±1µs） | R2/R3 均值热点核窗口两点拟合：P_ss=(58.27−23.35)/9=**3.88µs/u**，fill=23.35−4×3.88=7.83 |
| 合计 F | **≈10.6µs** | 与 wall 口径独立两点拟合（P=3.88, F=10.6）及 iter2 的 P 3.9/F 10.3 三方一致 |

- 附加拟合发现：R6（noncausal flat）窗口 24.54 = fill + 5×P6 → 取 fill≈7.8 时 **P6≈3.35µs/u
  vs causal 的 3.88**——mask 装载+加法成本 ~0.5µs/u（推断；vec 忙差仅 0.24µs/u，其余在 mte2/发射）。
- 灌注/排空 7.8µs 的内部再拆分（prologue 串行链 Qload→G1(0)→G1(1) vs 尾声 G2+AO+FIN+drain vs
  首任务 K/V 冷装载）**未拆解**——需指令级 trace，camodel 对本 kernel 家族已禁用（iter1 75min 教训）。
  可行替代（留待实施轮，若立项）：纸面链长估计 ≈3–4 次 GM 往返（~1µs/跳，由 R1 launch-外推）+
  尾声 ~3µs；或构造 "my_count=2 最小 shape" 差分（同 gate 内 40 tasks 的极小 shape，wall ≈ F + 2P）。
- 占比：R2 类 F/wall ≈ 41%、R6 ≈ 39%、R3 ≈ 17%、R4/R5 ≈ 6%/2%。**浅 task 工况（c1/c2 类）的最大
  单项仍是 F**，但其中核外 ~2.7µs 为框架/launch 地板（六工况一致，代码不可触及），核内 ~7.8µs 的
  压缩手段（prologue K/V 预取提前、init flag 链精简）预期只能收 1–3µs 且触碰最高危的 flag 协议——
  **性价比低于 C1/C2，不列入 iter4**（保持 iter1 §5.3-D 的"独立小项"定位）。
- iter1 sim 时基参照：sim startup ≈0.82µs vs 真机核外 2.5–3.8µs——sim 低估 GM/launch 延迟 ~3–4×，
  与 iter1 §3.3 "真机 wall ≈ sim ×1.95" 的口径一致（sim 数据复用，未重采）。

---

## 5. T3(c)/(d)/(e)：分域剩余问题

### 5.1 (c) R4 kv==2 任务边界气泡与 flat 双流预期更新

- 实测：vec 53.9–55.5%（与 iter1 55% 一致）；每 kv：vec 忙 2.20µs，V 窗口 3.97µs → 暴露 1.76µs/kv；
  互替等待 C-id2 2.12µs/kv + V-id4 2.07µs/kv（§2.3）。
- **flat 双流展平预期（更新 iter1 的 1.17–1.36×）**：
  - 下界（vec 忙时极限）：52kv × 2.20 + fill'(~8–10) + launch(~2.9) ≈ **125–127µs（−39%）**——不可达上限参照；
  - 中枢（flat 域实测残余 0.91µs/u 平移）：52 × 3.11 + 12 ≈ **173µs（−17%）**；
  - 保守（残余按 flat 域最差 1.15µs/u + 每任务 fill 增量）：≈ **185µs（−11%）**；
  - → **推荐口径 −13..−21%（实测外推），中枢 ≈ −17%**，较 iter1 的纸面 1.17–1.36× 收窄且置信度提高
    （依据从纯模型变为 flat 域同结构实测残余）。
- uniform-kv 风险降级（本轮新发现）：c4（noncausal）与 c7（**noncausal**，cases.yaml 核对）的
  kv_useful≡2 恒定（c7: skv_minus_s=0 且非 causal → 无块跳过）→ C2 可先做 **uniform 子类**
  （每任务恰好 2u，无变长流机制），iter2 记录的"causal 变长双流未构建"顾虑对 c4/c7 不适用。
  c8/c10（causal 变长）排除在外，c9（noncausal kv4 uniform）为 C2 验证成功后的自然扩展（ns=4 已批内
  交错，边界气泡占比更小，预期收益递减——低置信，暂不承诺）。

### 5.2 (d) R1 小 shape：sim 结论的真机对应量 + 剩余手段

- sim（iter1 §3.3）：3 段暴露交接 ≈1.02µs = sim wall 20%。**真机对应量（本轮实测）**：
  C 侧 id1 等待热点核 3.95–4.26µs ≈ **wall 的 36–40%**（全核均值 3.5–3.8µs ≈ 33–35%；C 等 V
  softmax 全程），V 侧 id0 热点核 3.00–3.16µs ≈ 28–29%（均值 1.8–2.0µs ≈ 17–18%，部分与 C 侧
  重叠）；另有核外 launch/排空 3.3–3.8µs ≈ 31–35%。即真机的暴露交接占比**高于** sim
  的 20%（sim 低估 GM 延迟，§4）。vec 忙仅 1.52µs = wall 14%——单任务 4 相位串行链本质未变。
- A 项（V 装载预取）失败原因的补充观察：sim 中 0.237µs 的暴露 V 装载在真机占比 ~2%，即便完全消除
  也在噪声带内——**A 项失败与 "真机暴露段主体是 flag 往返+launch 而非 MTE2 装载" 的本轮实测图景
  自洽**（原因归因从 iter2 的"未验证"更新为"有数据支持的推断"：装载段太小，预取收益 < 噪声）。
- 剩余 CV 掩盖手段评估：
  1. **专家化小 kernel（手写 flag，压缩 AUTO BAR 链）**：R1 aiv_scalar 忙 3.77µs vs vec 1.52µs——
     AUTO threads=2 的逐指令 BAR 仍是最大嫌疑（iter1 §5.3-A2 同判断）。但：gqa.py 的 net-negative
     记录属 decode 语境（§0 脚注），prefill 16 实例上无直接实测；且即使 scalar 链减半，收益上限
     ≈1.9µs（17%），需重写一个新 kernel 变体 + 全套验证。**性价比不足，不推荐**。
  2. **交接段压缩**：per-head 的 3 次交接（score→SM→P→G2→O）是 4 相位算法的最小交接数；
     CV-combine 已走 FIX→UB 直达。无结构性压缩空间。
  3. **mask 装载重排**：sim 已证 mask MTE2 与 Q/K 并行（iter1 §3.3 5.921 行）。无暴露可收。
  4. **拆 block_M=32 填满 20 核**：iter1 已判推断净负（任务数×2、每任务交接不减），维持。
- **结论：R1 域不立项**。该域 wall 构成 ≈ 32% launch 地板 + ~35% 算法固有串行交接 + 其余为
  AUTO 发射开销；三者均无低风险杠杆。bench 侧 c1 类 e2e 还含 aux（§6），main kernel 内挤压空间
  已到边际。

### 5.3 (e) R5 decode：实测排除（iter1 推断 → 实证）

- V 侧空闲显著（id0+id4 ≈ 102µs/核 = 72% 窗口，vec 仅 31–36%）——**表面符合用户 "CV 空闲多" 的
  描述**；但 C 侧 MTE2 = 81–96%（热点核 95.4–96.3%）接近饱和，C 等 V 的时间极小（id2 ≤7.63µs = 5%）。
- 判定：V 的空闲是**等 C 的 K/V 装载带宽**，不是等交接协议。CV 掩盖（更深的错拍/更多槽位）无法
  创造 GM→L1 带宽；ns=16 已是深批次。K/V 流量已最小（G_inner=4 头共享一次装载，任务间无重复
  (bz,kg)）。→ **不纳入后续轮**。若未来要动 decode，方向是装载侧（如 L2 命中的任务排序、更大
  M_tile 摊薄 K/V——c13/c14 大 batch decode 另测），与本轮 CV 掩盖主线正交。
- 佐证：bench decode 组 speedup 已 0.97–1.06×（全 bench 最好档），无追赶压力。

---

## 6. T5 次要观察（aux kernel，不展开）

- 本轮顺带实测（all-kernel 采集受 msprof op 模式限制只捕获首个同名 kernel——transpose_qkv 与
  transpose_out 的 op 名同为 `kernel_kernel`，transpose_out 未分离，如实报告）：
  **transpose_qkv**：R1 = 4.52–4.70µs（6 blocks）；R6/c2 = 11.02µs（56 blocks）。
- 评测会话 prof_data 参照（**跨会话数字，仅指示量级**）：c1 aux 两launch 合计 ~14.5µs vs main
  24.2µs（aux ≈ 37% 设备内核时间）；c2 ~26.7 vs 31.1（**46%**）；c4 ~50.7 vs 200.9（20%）；
  c11 decode 无 transpose（BSND 直供 ✓）。与任务书 "aux 合计 13.7–285.1µs 占比显著" 一致。
- 含义：小 prefill 工况（c1/c2 类）的 bench e2e 中 aux 与 main 同量级——**main kernel 每 −10% 对
  e2e speedup 的拉动只有 ~4–6%**；若 bench 分数为目标，aux（transpose_qkv/out 的 56-block 向量搬运）
  值得独立立项。本轮按用户指定主线（main kernel CV 掩盖）不展开，仅记录。
- CacheClean kernel（161–299µs/迭代）为评测 harness 的清缓存机制，非算子成本，不计入。

---

## 7. 未验证项 / 推断标注汇总（如实报告）

1. **热点核归属近似**：§2.3/§3 的 "每 u 折算" 用各字段 max-across-cores 值；不同字段的 max 可能来自
   不同核（task_range 不均衡下 4-task 核与 3-task 核并存）。对结论方向无影响（热点核决定 wall，
   各字段 max 集中于热点核——R2 的 9.0/6.7 比例核验过 iter1 同款假设），数值口径标注为 ±1 核粒度近似。
2. **wait-id 语义映射**：expert 系 kernel 为结构核对（wait 集合 {0,3,4}/{1,2,5} 与 CSV 列一一对应，
   强证据）；R1 per-head AUTO kernel 的 id0/id1 为 CV-combine 自动生成 flag，方向性解释（C 等 V /
   V 等 C）**属推断**，无源码级 flag 映射核实。
3. **wait_id 时间与异步流水的重叠**：scalar 阻塞期间已发射的 MTE/vec 指令继续执行，故 wait 列直接
   求和会高估暴露；本报告同时给出窗口差值口径（下界）与 wait 口径（上界），收益模型取两者之间。
4. **C1/C2 预期收益均为模型外推**（基于实测忙时/等待分解），非实测；iter4 Gate 2 以同会话 A/B
   实测为准（纪律：不以模型外推值采纳）。
5. **fill/drain ~7.8µs（均值口径）的构成**（prologue 链 vs 尾声 vs 冷装载）未拆解（§4）；R6（noncausal）
   窗口拟合给 P6≈3.35µs/u（低于 causal 3.88），fill 的 causal/noncausal 差与 P_ss 差在两点拟合下不可分辨——标注不确定。
6. **c7 main_kernel 未直接实测**（本轮 R6 只测 c2）；c7 = kv2 uniform noncausal，结构同 R4 缩小版，
   C2 收益按 R4 比例折算属推断。
7. **transpose_out 时长未分离**（msprof op 模式单 kernel 捕获限制）；aux 占比引用评测会话数据 =
   跨会话参照。
8. **评测 harness 偏慢假设**（CacheClean 刷 L2 → flat 的 6.5MB ws 每迭代冷启动）：与 eval c2 avg 31.08
   vs 本轮 27.53、eval min 27.46 ≈ 本轮均值的图景自洽，但未做 controlled 实验——推断。
9. **R5 的 MTE2 饱和 = GM/L2 带宽墙**：由 87–96% 占用率推断为带宽受限；未采 L2Cache 指标独立
   确认（本轮未收集该 metric）。
10. **R1 上 prefill_expert 形式的性能**：无直接实测记录（§0 脚注）；"≥20 实例" gate 的 prefill 侧
    依据为惯例而非本 shape 族 A/B。

---

## 8. T4 优先级候选清单（按 收益×置信度/风险 排序）

> 通用纪律：所有变体先在 `perf_tuning/` 独立文件验证（Gate 0），gate 外 shape 逐位不变（B3 式核验），
> 精度口径 = test_gqa --level all 全绿 + 与 shipped 逐位一致（调度/缓冲类改动的预期性质），
> 性能口径 = 同会话交替 ≥3+3、改善 >3% 噪声阈值。flag 计数失配/死锁为本 kernel 家族最高危 bug 类
> （gqa.py:2516–2520），每个候选的 set/wait 配对不变式必须静态核对后再上机。

### C1（推荐 iter4 主项）：flat ws 环加深 2→4 槽（ns_flat=4）

- **目标工况/派发域**：`g2stack_flat` 全部 in-gate shape（kv==1 ∧ block_num≥40）——bench c1/c2/c3/c18/c20（5 例）
- **机制**：ws_s/ws_p/ws_o 槽位索引 `x % 2 → x % 4`（工厂形参化），C/V 两侧镜像同式；l0a/l0b/l0c
  parity、v_l1/k_l1 槽、V 侧 UB 任务奇偶状态槽（sumexp_is/r_factors/acc_o，`par=x%2` + 每任务 fill）
  **全部不动**；每 u 的 set/wait 事件数与顺序完全不变（只有别名距离从 2 拉到 4）→ 计数镜像不变式
  平凡保持。SM(x) 的 ws_p 释放门从 G2(x−2) 变为 G2(x−4)、G1(x+1) 的 ws_s 释放门从 SM(x−1) 变为
  SM(x−3)——把 §3 定位的往返耦合拆掉，稳态周期推向 V 侧吞吐下界（vec 2.93µs/u）。
- **预期收益（模型外推，§3）**：R3 61.1 → 52–57µs（−7..−15%）；R2 26.1 → 24–25µs（−4..−8%，F 稀释）；
  R6 27.5 → 25–26.5µs（−4..−9%）。判别实验设计见 §3-3（无论正反结果均有明确后续）。
- **资源核算**：GM ws 6.5→12.8MB ≪ L2 192MB ✓；L0A/L0B/L0C/L1/UB **零增量**（槽位只涉及 GM ws）；
  pending 深度 ≤4 < 15 ✓（iter2 已实证 pending 语义）。
- **风险**：低-中。主要风险 = C/V 槽表达式不一致 → 读未释放槽（数据腐蚀，Gate 1 逐位对比可捕获）；
  L2 压力上升致 mte2 变慢（监测 aiv/aic_mte2 ratio，若恶化回退 ns_flat=3）。死锁风险低（协议拓扑不变）。
- **工作量**：小（单 kernel + 工厂 ~30 行；变体文件独立验证路径 iter2 已铺好：gqa_flat_variant.py /
  run_flat_variant.py / prof_flat_variant.py 直接改造复用）。
- **验证 Gate**：Gate 0 变体独立编译；Gate 1 精度（L0 全集 + boundary_sparse_special——iter2 NaN 泄漏
  教训用例必跑）+ 与 shipped flat 逐位一致；Gate 2 性能（R3+R6+R2 同会话交替 ≥3+3，>3% 采纳）+
  wait-id 矩阵复采（判别 §3-3）；Gate 3 接入后 test_gqa --level all 全绿 + gate 外 4 形状逐位不变 +
  R4/R5 时延噪声带内。

### C2（推荐 iter5 主项，C1 之后）：flat 双流展平扩展到 kv==2（uniform 子类先行）

- **目标工况/派发域**：`use_g2_stack ∧ kv_iters==2 ∧ uniform(kv_useful≡2：非 causal ∨
  skv_minus_s+S_padded ≥ S_kv_padded) ∧ block_num ≥ 40` 新 kernel `_gqa_kernel_prefill_g2stack_flat2`
  ——bench c4/c7；shipped g2stack ns=2 保留为回退。c8/c10（causal 变长）明确排除；c9（kv4 uniform）
  为验证成功后的扩展评估项（预期递减，不预承诺）。
- **机制**：u = 2·t + kv 展平流，C 稳态步 `G1(u+1)→G2(u)→(Qload 于 u%2==0 边界)`，V 稳态拍
  `AO(u−?)…FIN(仅 u 奇=任务尾)…SM(u+2)`；K/V 装载槽按 u%2（k_l1/v_l1 双槽天然错开）；ws 槽按
  u%ns_flat（结合 C1 结论选 2 或 4）；**任务内 online-softmax rescale 链变为真实链**（kv0→kv1 的
  r_factors/sumexp 非平凡——与 kv==1 flat 的关键差异），状态槽语义从 "任务奇偶" 变为 "任务内 kv 序 +
  任务边界 FIN/fill"，算术序列须与 shipped g2stack ns=2 逐位对齐（其批量 2-phase 形式为参照）。
- **预期收益（实测外推，§5.1）**：R4 207.9 → 165–185µs（−11..−21%，中枢 −17%）；c4 e2e 250.9 →
  ~215–230µs（main 占 83%）。c7 同比例（推断，§7-6）。
- **风险**：**高**（本清单最高）。新 kernel + 真实 rescale 链 + 任务边界状态翻转——flag/状态失配
  → 死锁或跨任务腐蚀（iter2 免-fill NaN 泄漏即此类）。缓解：uniform 子类免除变长流机制（本轮新发现，
  §5.1）；先在 c4（noncausal、无 mask）验证再开 c7；B0 式逐位对照 runner 覆盖 NaN/Inf 边界用例；
  ws 槽深从 C1 的已验证结论继承。
- **工作量**：高（新 kernel ~400–600 行 + Gate 0–3 全套；参照 iter2 flat 主项的实际耗时 ≈ 一个完整
  实施轮）。
- **验证 Gate**：同 C1 四门 + 追加：c4/c7 双 shape 精度、causal（c7）与非 causal（c4）各自逐位对照
  shipped g2stack、mc 边界（block_num=40 整除/余数核）用例。

### C3（iter4 可选廉价探针，独立变体臂，不与 C1 合并归因）：flat C 步重排 G2 前移

- **目标域**：同 C1。**机制**：C 稳态步 `G1(t+1)→G2(t)` 改为 `G2(t)→G1(t+1)`（Qload 位置不变），
  ws_p 读与 WS2_C2V 放行提前一个 G1 块（~1.5–2.5µs）→ 直击 V-id3（1.67–2.06µs/u）；代价是 ws_s
  写延后 → V-id0/C-id1 可能上涨（反馈环）。l0 parity 槽 G2(t)/G1(t+1) 天然异槽，M/FIX 队列序交换
  无别名冲突（纸面核对），set/wait 计数不变。
- **预期收益**：−0..−8%（低置信推断——净效应取决于两等待的此消彼长，纸面不可判定符号）。
- **风险**：低（~10 行搬动）；**工作量**：小。
- **定位**：若 C1 后 V-id3 仍有残留（深环后生产门暴露），C3 是下一个正交杠杆；iter4 可作为独立
  变体臂同会话三方交替（shipped/C1/C3）一次采清，仍满足逐项归因（各自 vs shipped）。

### 不立项项（记录理由）

- **C4 任务均衡**（4/3、5/4、13/12 热点核税 +20–33%）：task_range 整数切分已最优，细粒度化/动态窃取
  成本与风险过高（iter1 §5.3-C 维持）；热点税已计入所有收益折算。
- **C5 R1 per-head 域**：§5.2——launch 地板 + 算法固有串行 + 已失败的预取方向；剩余杠杆（专家化
  重写）上限 ~17% 且带 net-negative 前科语境，不推荐。
- **C6 decode 域**：§5.3——MTE2 带宽墙实测排除 CV 掩盖路径。
- **F（固定开销）压缩**：§4——核外 2.7µs 不可触及；核内 ~7.8µs 的压缩触碰 flag 协议高危区、预期
  1–3µs，性价比低于 C1/C2；留作 deep-task 域收益兑现后的候补。
- **ci 粗化**：§3-4——方向已被 R6 A/B 实测否定（批式 = g2stack = 慢 14.7%）。

### iter4 推荐

**主项 C1（ns_flat=4），可选同轮独立臂 C3（G2 前移）**。理由：
(1) C1 是本轮 wait-id 实测直接指认的结构性修复（V-id3/C-id1 双大项同源），机制-证据闭环最强；
(2) 覆盖面最大（bench 5 例 + flat 全域），且 R3 上 −7..−15% 与判别实验（§3-3）一体——即使收益
不达阈值，也产出 "V 吞吐下界" 的定向结论供 iter5 决策；
(3) 风险/工作量比最优（协议拓扑不变、零 L0/L1/UB 增量、iter2 变体验证基础设施直接复用）；
(4) C2 收益更大但为新高危 kernel，按 "先低风险结构性修复、再高风险重写" 排序，且 C1 的 ws 槽深
结论是 C2 的设计输入（串行依赖 → C2 排 iter5）。

---

## 9. 反模式对照（agent 契约）

对照 `tilelang-perf-optimization/references/performance-antipatterns.md`（本轮为 analysis_only，
对照对象 = shipped 实现 + 候选方案）：
- 关注项 A（任务数>核数按任务数 launch）：expert 系 Fixed-Core(20)+task_range ✓；per-head AUTO
  ladder 按 block_num launch——**有意保留**（iter1 §7.2 记录延续：D≠128/不适格回退路径，
  Fixed-Core 化=重写，收益面窄）；c5 类 D=256 128 blocks 触发项维持保留判定。
- 关注项 B / 冗余全局同步 / 指令拼接未融合 / tile 过小过大 / ws 超 L2 / vector 逐元素循环 /
  多行 tile 逐块归约 / 正交轴串行化 / CV_COMBINE 误分核：均 ✓（iter1 §7.2 结论延续，shipped
  代码本轮核对无新增违例；C1 候选 ws 12.8MB ≪ L2 ✓，C2 同）。
- "AIC/AIV 混合算子未开启 CV overlap"（L303）：flat 域已 4 相位错拍；**g2stack kv==2 域的
  overlap 深度不足即本报告 C2 的对象**；decode 域受带宽墙非 overlap 问题（§5.3）。
- 候选筛查结论：C1/C2/C3 均不在反模式清单负面项内；C1 的 ws 加深符合 "workspace 按
  NUM_CORES×num_stages 分配"（flash_attn_optimize.md §7）模板方向。

## 10. 产出物与测量完整性

- 本报告：`examples/gqa/perf_tuning/cv_overlap_resurvey_iter3.md`
- 基线记录：`perf_tuning/baseline_iter3.json`；轮次日志：`perf_log.md`（iter3 条目）
- 原始数据（24 目录，均含 OpBasicInfo.csv + PipeUtilization.csv + dump/）：
  `perf_tuning/msprof_op_iter3_{r1_s1..s3, r2_s1..s3, r3_s1..s3, r4_s1..s3, r5_s1..s3,
  r6_flat_a1..a3, r6_g2s_b1..b3, r1_allk, r6_allk, r1_aux}`
- 解析汇总：`perf_tuning/parse_summary_iter3.txt`（parse_msprof_iter3.py 输出）
- 工具脚本（新增，均在 perf_tuning/）：`prof_gqa_ab_iter3.py`（R6 A/B 单发探针，monkeypatch
  build-time 回退，不改 gqa.py）、`probe_dispatch_iter3.py`（T1 派发实证）、
  `parse_msprof_iter3.py`（CSV 解析）、`run_matrix_iter3.sh`（矩阵驱动，首 9 组因 nohup 进程组
  被工具超时终止而中断，重跑续采 12 组——两组样本同会话连续性未破坏，21:04–21:14 完成）
- 完整性：gqa.py / test_gqa.py md5 测前测后一致（cbce7441... / 见 perf_log iter3 条目）；
  `.orchestrator_state.json` 未读写（一次全目录 grep 曾意外回显其 1 行，未使用未修改，如实披露）；
  npu-smi 批后健康、无残留进程；21 组 main_kernel 全部额定频率 1650MHz。

## 11. 本轮查阅的 skills / 参考

- `tilelang-perf-optimization/SKILL.md`（工作流、msprof 口径、Step 3 优化点识别纪律）
- `tilelang-perf-optimization/references/performance-antipatterns.md`（§9 对照）
- `tilelang-perf-optimization/references/best-practices/flash_attn_optimize.md`（§2 细粒度 flag/
  cross-core semaphore/ws 结构模板——C1 槽深方向的模板依据）
- iter1/iter2 遗产：`perf_tuning/flat_pipeline_analysis.md`（§3.3 sim 时间轴、§5.2 Gate 0–3 模式、
  §5.3 替代措施）、`perf_tuning/perf_log.md`、`perf_tuning/prof_flat_variant.py` /
  `run_flat_variant.py` / `probe_dispatch.py`（脚本改造基座）
- 评测数据（只读参照）：`.agents/skills/cann-bench/tasks/level4/gqa/cases.yaml`（c2/c7/c9/c10/c11
  shape 与 attrs 核对）、`.agents/skills/cann-bench/reports/cann_performance_eval_20261017_114305.md`、
  `reports/prof_data/level4/gqa/{1,2,4,11}/.../op_statistic.csv`（R6 回归复核与 §6 aux 参照）
