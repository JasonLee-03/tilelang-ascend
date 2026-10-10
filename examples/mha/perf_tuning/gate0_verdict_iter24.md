# GATE 0 预筛门判决书 — iteration 24（mode = perf_measure）

> **裁决：`[GATE0_VERDICT] FAIL`** — GATE 2（`q_l1` flat 域双缓冲）**NO-GO**；GATE 1（`unit_flag=0b11` 铺垫）**NO-GO（作为 GATE 2 铺垫，其对象已消失）**。
> 本轮为纯测量轮，**未产出任何采纳代码**，生产 kernel 逐字节未变（md5 `dda3ad08` 前后双复核）。

- 算子：`examples/mha/`（mha.py 1926 行，md5 `dda3ad0877ba4df74bbd559f6912eeed`）
- 测量窗口：2026-10-02 00:36 – 01:13（local，UTC+8）
- 判决依据：6 次 interleaved paired wall A/B（3 repeat × 2 arm × 2 case）+ 4 个 msprof-op metrics arm（prod/probe × case9/case7）+ 2 个消融探针（fbm 2 行 delta / Form A 3 行 delta）
- 归档：`perf_tuning/msprof_gate0_iter24/`（564 KB / 65 文件）、`perf_tuning/probe_gate0_qablate/`、`perf_tuning/baseline_iter24.json`

---

## 1. 测量环境与口径

| 项 | 值 |
|---|---|
| 设备 | Ascend910（platform_version 5，Ascend910_9362 级），**20 AIC + 40 AIV** |
| 频率 | aic 1650 MHz / aiv 1650 MHz，**Current == Rated = 1650（无降频）** |
| CANN / 驱动 | **9.2.0** / **467992** |
| 主机 | aarch64，Linux 5.10.0-182.0.0.95，40 CPU；torch 2.7.1 + torch_npu（`device_count=2`） |
| npu-smi | 2×Ascend910 **Health OK**，轮次开始时 `No running processes found` |
| msprof | `/home/developer/Ascend/cann-9.2.0/bin/msprof` |
| **wall 口径** | 纯 `msprof --task-time=on` 时间线模式 → `op_summary_*.csv` 的 `Task Duration(us)`，**warmup 5 / iters 20 → n=25 取 median**，由仓内已标定 `parse_prof.py` 解析（该口径曾对 cann-bench case1 锚 46.925 µs == T_cand 46.9 µs 校验）。与 `msprof_r10_base` 同口径可直接比。 |
| **metrics 口径** | `msprof op --kernel-name=main_kernel --aic-metrics=PipeUtilization,ResourceConflictRatio,L2Cache,Memory,BasicInfo,ArithmeticUtilization,MemoryUB,MemoryL0`，单发（warmup 0 / iters 1），**20 block 全量跨核均值 + 关键 block（argmax wall）分别报** |
| **camodel** | **本轮完全未运行**（纪律 #1：camodel 三度实证方向背离，仅可定性）。以下全部定量结论均为真机 msprof。 |
| 缓存纪律 | 4 个会话隔离 `TILELANG_CACHE_DIR`：`cache_prod` / `cache_probe`（fbm）/ `cache_probe_fa`（Form A）/ `cache_smoke`。**生产缓存从未与任何探针共享**；未使用默认缓存。 |
| 采样次数 / 噪声带 | 每 arm 3 次 interleaved 重复（prod/probe 交替执行以抵消漂移）。**prod 侧 median 重复间极差：case9 0.78 µs（0.09%）/ case7 0.23 µs（0.03%）**；paired-diff SEM ≈ 1.2 µs（≈0.14%）⇒ 本轮实际分辨力比历史 ±3~4% run-to-run 带紧一个数量级。 |

**mha.py md5 复核结果（硬性纪律 #3）**

| 时点 | md5 | 行数 | 判定 |
|---|---|---|---|
| 轮次开始 | `dda3ad0877ba4df74bbd559f6912eeed` | 1926 | ✅ 与任务书 `dda3ad08` 一致 |
| 构建 fbm 探针后 | `dda3ad0877ba4df74bbd559f6912eeed` | 1926 | ✅ UNCHANGED（探针脚本内置断言） |
| 构建 Form A 探针后 | `dda3ad0877ba4df74bbd559f6912eeed` | 1926 | ✅ UNCHANGED（探针脚本内置断言） |
| **轮次结束** | **`dda3ad0877ba4df74bbd559f6912eeed`** | **1926** | ✅ **UNCHANGED** |

**生产态复现验证（新鲜基线，本轮实测）**

| case | 本轮 median（3 次均值） | r10_base 历史 | Δ | 判定 |
|---|---|---|---|---|
| case9 | **869.547 µs** | 869.87 | **−0.04%** | ✅ 复现 |
| case7 | **856.600 µs** | 855.54 | **+0.12%** | ✅ 复现 |

单发交叉核对：case9 `msprof op` TaskDur 870.00 µs（无 metrics 重放臂）/ 885.78 µs（metrics 重放臂）；case7 863.40 µs。独立 `tl_prod_c9_r1` median 872.12（stdev 6.453）。全部落在带内。

---

## 2. Q1 判决：C scalar 余量 = **充足**（GATE 2 可以新增标量游走）

### 2.1 关键发现：msprof 直接分离了「真实发射」与「stall 归属」两个计数器

`PipeUtilization.csv` 同时提供 `aic_scalar_time(us)`（busy 总量）与 `aic_scalar_single_time(us)` / `aic_scalar_dual_time(us)`（单发射 / 双发射**占用时间**）。二者与 stall 桶（`*_mte1_stall` / `*_wait_idN` / `*_wait_ib` / `*_cube_stall`）**加和可闭合**，因此拆分是**直接测量**，不是反推。

| 口径 | case9（20 block 均值） | case9 关键 block(#3) | case7（20 block 均值） | case7 关键 block(#6) |
|---|---|---|---|---|
| AIC wall（`aic_time`） | 839.09 µs | 881.94 µs | 836.31 µs | 859.53 µs |
| **scalar busy** | **818.47 µs / 97.54%** | 852.52 µs / 96.66% | **764.53 µs / 91.42%** | 802.31 µs / 93.34% |
| ↳ 工具 `aic_scalar_ratio` 列 | 92.48%（min 87.86 / **max 96.33**） | — | 88.59%（min 84.33 / max 92.97） | — |
| `scalar_single_time` | 20.25 µs | 21.06 µs | 11.95 µs | 12.30 µs |
| `scalar_dual_time` | 56.29 µs | 58.28 µs | 50.99 µs | 52.66 µs |
| **真实标量发射（single+dual）** | **76.55 µs = 9.12% wall** | 79.34 µs = 9.00% | **62.94 µs = 7.53% wall** | 64.96 µs = 7.56% |
| **stall 归属（busy − issue）** | **741.92 µs = 88.42% wall** | 773.17 µs = 87.67% | **701.60 µs = 83.89% wall** | 737.35 µs = 85.79% |
| **真实发射占 scalar busy 比** | **9.35%** | 9.31% | **8.23%** | 8.10% |

### 2.2 stall 成分拆分（全部为直接测量的计数器；加和闭合度见末列）

| stall 桶 | case9 均值 | %wall | case7 均值 | %wall |
|---|---|---|---|---|
| `aic_scalar_mte1_stall` | **559.04 µs** | **66.62%** | 209.30 µs | 25.03% |
| `aic_scalar_wait_id2`（SEM_WS2_V2C，等 V 交付 w2/P） | **176.32 µs** | **21.01%** | **530.89 µs** | **63.48%** |
| `aic_scalar_wait_id1`（SEM_WS1_V2C） | 6.14 µs | 0.73% | 10.14 µs | 1.21% |
| `aic_scalar_wait_id5`（SEM_WS3_V2C） | 3.43 µs | 0.41% | 6.04 µs | 0.72% |
| `aic_scalar_wait_ib` | 2.06 µs | 0.25% | 1.60 µs | 0.19% |
| `aic_scalar_cube_stall` | 0.33 µs | 0.04% | 0.00 µs | 0.00% |
| `aic_scalar_mte2_stall` / `mte3_stall` / `wait_time` | 0 / 0 / 0 | 0% | 0 / 0 / 0 | 0% |
| **桶加和** | 747.31 µs | 89.06% | 757.97 µs | 90.63% |
| **加和 / stall 残差** | **100.7%**（近可加，轻微重叠） | | **108.0%**（`wait_id2` 与 `mte1_stall` 部分重叠） | |

> 闭合度 100.7% / 108.0% ⇒ 桶间存在**轻微重叠**（同一等待被同时记入 `wait_id2` 与 `mte1_stall`）。这不影响 Q1 结论：结论只依赖 `single+dual`（真实发射）与 `scalar_time`（busy）这一对**互斥且直接测量**的计数器，二者之差 741.92 / 701.60 µs 已被上表桶加和**完全覆盖（≥100%）**。

### 2.3 判决

- **scalar 余量 = 充足。** 真实标量发射仅占 AIC wall 的 **9.12%（case9）/ 7.53%（case7）**，占 scalar busy 的 **9.35% / 8.23%**。所谓「C scalar 95.5%」是 `aic_scalar_ratio`（**busy 比，含 stall**），其中约 **91% 是 stall 归属**，不是发射饱和。
- **GATE 2 可以新增标量游走**（预取地址计算、双槽 side 索引、`SIG_Q_L1+1` token 收支）——标量发射侧余量 ≥ 90%，不构成否决理由。
- **对任务书前提的修正**：`final_report.md:680` 的「C scalar 95.5%」措辞暗示其为工作管成分，实为 busy 比。本轮实测 case9 `aic_scalar_ratio` 均值 92.48%、max-block 96.33%，历史 95.5% 落在 per-block 带内 ⇒ **数值可复现，但语义需更正为「busy 比，≈91% 是 stall」**。
- 侧证：探针（消融 Q prologue）令 `scalar_dual_time` case9 −2.24 µs / case7 `scalar_single_time` −3.43 µs，即被删除的 6 个 flag 往返确实计入发射侧；但相对 741.92 µs 的 stall 侧完全无关紧要。

---

## 3. Q2 三分量归因复采

### 3.1 case9（B=4,S=1024,S_kv=1024,N=16,D=256,causal → form_b_mma，flat 域）

**AIC（cube0，20 block 均值 / 关键 block #3）**

| 项 | 均值 µs | 均值 %wall | 关键 block µs | 关键 %wall | 工具 ratio 列均值（min~max） |
|---|---|---|---|---|---|
| wall `aic_time` | 839.09 | 100% | 881.94 | 100% | — |
| MTE2 | **496.18** | **59.13%** | 505.87 | 57.36% | **56.07%（53.03~59.34）** |
| FIXPIPE | 335.82 | 40.02% | 354.42 | 40.19% | 37.95%（36.07~40.63） |
| MTE1 | 335.42 | 39.97% | 348.06 | 39.47% | 37.90%（36.32~39.33） |
| CUBE | 190.99 | 22.76% | 198.11 | 22.46% | 21.58%（20.70~22.39） |
| MTE3 | 0.0012 | 0.00% | 0.0012 | 0.00% | ~0 |
| scalar busy | 818.47 | 97.54% | 852.52 | 96.66% | 92.48%（87.86~96.33） |

| AIC 暴露等待 | 均值 µs | 均值 %wall | 关键 block | max-ratio block |
|---|---|---|---|---|
| `mte1_stall` | 559.04 | 66.62% | 549.43 / 62.30% | — |
| `wait_id2` | 176.32 | **21.01%** | 137.90 / 15.64% | **25.67%（block #12）** |
| `wait_id1` | 6.14 | 0.73% | 3.55 | — |
| `wait_id5` | 3.43 | 0.41% | 8.47 | — |
| `wait_id0` / `wait_id3` / `wait_id4` | **列不存在** | — | — | — |

> **结构事实（非测量缺口）**：AIC 只在 V2C 方向等待。SEM 映射（mha.py:240-245）= `WS1_C2V=0 / WS1_V2C=1 / WS2_V2C=2 / WS2_C2V=3 / WS3_C2V=4 / WS3_V2C=5`；id0/id3/id4 是 **C 侧 set（C2V）**，AIC 永不 wait ⇒ msprof 不产出该三列。任务书要求的「AIC `wait_id0`~`wait_id5` 读全」在硬件语义上只有 id1/id2/id5 三项存在，已全读。

**AIV（vector0，20 block 均值）**

| 项 | 均值 µs | 均值 %wall |
|---|---|---|
| wall `aiv_time` | 839.72 | 100% |
| **VECTOR** | **468.58** | **55.80%**（工具 ratio 列 52.95%） |
| MTE2 | 213.02 | 25.37%（ratio 24.07%） |
| MTE3 | 92.41 | 11.01%（ratio 10.44%） |
| scalar busy | 605.86 | 72.15%（ratio 68.46%） |
| `wait_id0`（等 C 交付 w1） | **272.80** | **32.49%** |
| `wait_id4`（等 C 交付 w3） | **264.22** | **31.46%** |
| `wait_time` | 132.53 | 15.78% |
| `vector_stall` | 67.95 | 8.09% |
| `wait_id3` | 1.84 | 0.22% |
| **FLOWCTRL** | **不可得** | 真机 msprof 无此计数器（camodel trace 专有概念；纪律 #1 禁止作定量依据）→ **如实记为测量缺口**；最接近的真机代理 = `wait_id0+wait_id4+wait_time` = 669.55 µs（79.7% wall） |

**wall（权威口径）**：`op_summary` median = **869.547 µs**（3 次重复均值；单次 869.66 / 869.10 / 869.88）

### 3.2 case7（B=4,S=1024,S_kv=1024,N=32,D=128,causal → form_a，flat 域）

**AIC（cube0，20 block 均值 / 关键 block #6）**

| 项 | 均值 µs | 均值 %wall | 工具 ratio 列均值（min~max） |
|---|---|---|---|
| wall `aic_time` | 836.31 | 100% | — |
| MTE2 | **468.45** | **56.01%** | **54.28%（52.02~56.85）** |
| FIXPIPE | 301.60 | 36.06% | 34.95% |
| MTE1 | 205.12 | 24.53% | 23.77% |
| CUBE | 167.00 | 19.97% | 19.35% |
| scalar busy | 764.53 | 91.42% | 88.59%（84.33~92.97） |
| `mte1_stall` | 209.30 | 25.03% | — |
| **`wait_id2`** | **530.89** | **63.48%** | max-ratio block #2 = **66.79%** |
| `wait_id1` | 10.14 | 1.21% | — |
| `wait_id5` | 6.04 | 0.72% | — |

**AIV（vector0，20 block 均值 / 关键 block #6）**

| 项 | 均值 µs | 均值 %wall | 关键 block |
|---|---|---|---|
| wall `aiv_time` | 837.18 | 100% | 860.12 |
| **VECTOR** | **648.99** | **77.52%** | 666.45 / 77.48% |
| MTE2 | 214.19 | 25.58% | 221.52 |
| MTE3 | 98.48 | 11.76% | 99.93 |
| scalar busy | 686.86 | 82.05% | 732.79 / 85.20% |
| `vector_stall` | 262.14 | 31.31% | 263.58 |
| `wait_time` | 250.17 | 29.88% | 263.12 |
| `wait_id4` | 161.40 | 19.28% | 147.10 |
| `wait_id0` | 68.49 | 8.18% | 101.52 |
| `wait_id3` | 2.35 | 0.28% | 0.00 |

**wall（权威口径）**：median = **856.600 µs**（856.69 / 856.65 / 856.46）

### 3.3 三分量实测 vs 历史，逐项裁决

| 分量 | 历史值 | 本轮实测（case9） | 裁决 |
|---|---|---|---|
| **MTE2 ≈57%** | C.MTE2 57.8% | 工具 ratio 列均值 **56.07%**（per-block 带 53.03~59.34）；关键 block 57.36%；ratio-of-means 59.13% | ✅ **成立**（57.8% 落在 per-block 带内，均值口径 −1.7pp） |
| **`wait_id2` ≈25%** | ≈25% | 均值口径 **21.01%**；max-ratio block(#12) **25.67%**；max 绝对量 218.99 µs / max wall = 24.83% | ⚠️ **口径依赖**：**max-block 口径成立（25.67% ≈ 25%）**，均值口径**偏移至 21.0%**（−4pp）。历史 25% 为 max-block 读数 |
| **`ramp` ≈15%** | 模型 134 µs = 15% | **直接测量 ≤ 0**（见 §4）；模型自算的 Q 可攻击切片仅 **0.78% wall** | ❌ **不成立**（被直接测量判决性推翻） |

| 分量 | 历史值 | 本轮实测（case7） | 裁决 |
|---|---|---|---|
| V.vec 634.39 µs = 77.5% | 634.39 µs / 77.5% | block0 绝对值 **634.387268 µs（逐位复现）**；均值口径 **648.99 µs = 77.52%** | ✅ **成立（精确复现）** |
| C `wait_id2` 547.70 µs = 67% | 547.70 µs / 67% | 均值 **530.89 µs = 63.48%**；block0 **546.20 µs**；max-ratio block **66.79%** | ✅ **成立**（绝对量 −3.1%，max-block 口径 66.79% ≈ 67%） |

### 3.4 两 case 的 pacing 管身份判定（**分别判定，均经消融直接验证，不混用**）

**case9 = C↔V 四相位交接往返延迟链（latency-bound），非任何单管吞吐**
- 最大真实工作管 = **AIC MTE2 496.18 µs（59.13%）**，但**已被消融直接证伪为 pacing 管**：从中移除 32.40 µs（6.53%）后 wall 反升 0.29%，且 C 的 `wait_id2` 同步 +54.39 µs。若 MTE2 真为 pacing 管，wall 应下降约 32 µs。
- 最大**不可削减**真实工作管 = **AIV VEC 468.58 µs（55.80%）**，消融下 `aiv_vec_time` **逐位零变化（468.5841 → 468.5841，Δ=0.0000）**。
- 双向大互等签名：C 跨核等待 22.1%（id2 21.01 + id1 0.73 + id5 0.41）；**V 跨核等待 64.2%**（id0 32.49 + id4 31.46 + id3 0.22）。两侧同时大互等 = 交接往返链受限，而非单侧吞吐受限。

**case7 = AIV VEC paced（V-paced），经消融直接确认**
- 最大真实工作管 = **AIV VEC 648.99 µs（77.52%）**，显著高于 AIC MTE2 468.45 µs（56.01%）。
- **C 侧余量 = `wait_id2` 63.48%（530.89 µs）** ⇒ C 有超过六成 wall 在等 V。
- 消融从 C 侧移除 48.04 µs 真实工作（MTE2 −34.44 + MTE1 −13.60）→ wall **+0.87%（p=0.025 显著劣化）**，V 侧真实工作管零变化 ⇒ **V-paced 判决性确认**。

> **重要口径警示（两 case 不得混用）**：case9 与 case7 的 AIC MTE2 ratio 极为接近（56.07% vs 54.28%），但 pacing 身份**完全不同**——case9 是交接延迟链受限（V.vec 与 C.MTE2 仅差 5.9%，且互等双向都大），case7 是 V 吞吐受限（V.vec 77.52% vs C.MTE2 54.28%，差 23pp，C 侧 63.5% 余量）。仅看 C.MTE2 会误判两者同型。

---

## 4. Q3 `ramp` 直接测量

### 4.1 方法 A — 消融上界探针（判决性，已完成）

**探针构造（严禁覆盖 `mha.py`；均在独立目录）**

| 探针 | 路径 | md5 | delta |
|---|---|---|---|
| fbm（case9 域） | `perf_tuning/probe_gate0_qablate/mha_qablate.py` | `07a0b1dd40ad480b83da155526a18cf9` | **2 行**：L1514 `if kv_g1 == 0:` → `if fb * NUM_STAGES + i == 0:`；L1559 `if kv_g1 == eff_kv_iters - 1:` → 同 |
| Form A（case7 域） | `perf_tuning/probe_gate0_qablate/mha_qablate_fa.py` | `d0233d6f8223bae2753816c786cc74c2` | **3 行**：L1070 / L1084 / L1090 → 同式 |

- 效果：C 侧 Q prologue（GM→L1 `T.copy(Q[…], q_l1)` + fbm 的 `q_l1→l0a_s` 两次 MTE1 + Form A 的 `q_l1→l0a_s[side]` + 完整 `SIG_Q_L1`/`SIG_L0AS` token 轮）**由每任务一次（103 次/核）改为整 kernel 一次**。
- **token 收支与生产态同构**（init 1 set + tail 1 set vs head 1 wait = 1 dangling，与生产态末任务 dangling 完全一致）⇒ **无死锁**（两探针均编译并运行成功，无 HANG）。
- 生成器脚本内置断言：md5 前 8 位必须 == `dda3ad08`、行数必须不变、**差异行数必须恰为 2 / 3**、构建后回读生产态 md5 必须未变。全部通过。

**数值错误已确认的声明（硬性要求）**

> 两探针输出**数值上故意错误**，已实测确认；**未做任何精度对比作为采纳依据、未纳入任何 baseline、未提交**。

| 探针 | out_sum | golden ref_sum | max_abs_diff | mean_abs_diff | rel_max |
|---|---|---|---|---|---|
| fbm / case9 | −14403.6973 | −14595.2021 | **8.447e-01** | 1.296e-02 | **8.447e-01** |
| Form A / case7 | −7524.0005 | −7240.4751 | **5.900e-01** | 1.169e-02 | **5.900e-01** |

rel_max 达 59~84% ⇒ 消融机制**确实激活**（跨任务 Q 未更新），排除假阴性。

**机制确认（真机计数器，20 block 均值）— 消融在生成码中真实生效**

| 计数器 | case9 prod → probe | Δ | case7 prod → probe | Δ |
|---|---|---|---|---|
| `aic_mte2_instructions` | 1486.8 → 1385.4 | **−101.4（−6.82%）** | 1486.8 → 1385.4 | **−101.4（−6.82%）** |
| `GM_to_L1_datas(KB)` | 64307.2 → 62684.8 | **−1622.4（= −101.4 × 16 KB，精确）** | 38502.4 → 36880.0 | **−1622.4（精确）** |
| `aic_mte1_instructions` | 16079.8 → 15674.2 | −405.6（= −3.98/任务 × 102） | 10239.6 → 9223.0 | **−1016.6（−9.93%）** |
| `aic_mte2_time(µs)` | 496.18 → 463.78 | **−32.40（−6.53%）** | 468.45 → 434.01 | **−34.44（−7.35%）** |
| `aic_mte1_time(µs)` | 335.42 → 327.43 | −7.99（−2.38%） | 205.12 → 191.51 | −13.60（−6.63%） |
| `aic_scalar_mte1_stall(µs)` | 559.04 → 513.34 | **−45.70（−8.17%）** | 209.30 → 149.08 | **−60.22（−28.77%）** |
| `aic_scalar_time(µs)` | 818.47 → 811.77 | −6.70（−0.82%） | 764.53 → 746.99 | −17.54（−2.29%） |
| `aic_cube_time(µs)` | 190.99 → 190.98 | **−0.02（−0.01%，零变化）** | 167.00 → 166.98 | **−0.02（零变化）** |

**V 侧真实工作管零变化（吸收律判据）**

| 计数器 | case9 prod → probe | Δ | case7 prod → probe | Δ |
|---|---|---|---|---|
| `aiv_vec_time(µs)` | 468.5841 → 468.5841 | **0.0000（逐位相同）** | 648.9919 → 648.9984 | **+0.0065（+0.00%）** |
| `aiv_mte2_instructions` | 922.6 → 922.6 | **0** | — | — |
| `aiv_mte3_instructions` | 564.2 → 564.2 | **0** | — | — |
| `GM_to_UB_datas(KB)` | 11059.2 → 11059.2 | **0** | — | — |
| `UB_to_GM_datas(KB)` | 2662.4 → 2662.4 | **0** | — | — |

**C 侧跨核等待增量（吸收通道）**

| | case9 | case7 |
|---|---|---|
| `wait_id2`（等 V 交付 w2/P） | 176.32 → **230.71 = +54.39 µs（+30.85%）** | 530.89 → **588.56 = +57.68 µs（+10.86%）** |
| `wait_id5`（等 V 交付 w3） | 3.43 → **7.79 = +4.36 µs（+127.0%）** | 6.04 → **19.31 = +13.26 µs（+219.6%）** |
| **跨核等待合计增量** | **+58.75 µs** | **+70.94 µs** |
| 对照：C 侧真实工作削减 | −32.40（MTE2）−7.99（MTE1） | −34.44（MTE2）−13.60（MTE1） |

**附加证据 — L2：Q 装载是 L2 缺失主源，移除后命中率显著上升，wall 仍不动**

| | case9 | case7 |
|---|---|---|
| `aic_r0_read_cache_miss_allocate` | 20548.8 → 13405.5（**−34.76%**） | 11359.4 → 7435.3（**−34.55%**） |
| `aic_read_hit_rate(%)` | 92.01 → **94.65（+2.64pp）** | 92.63 → **94.96（+2.34pp）** |
| `aic_total_hit_rate(%)` | 92.51 → 94.94 | 93.53 → 95.51 |
| `aic_mte2_wait_ratio`（ResourceConflict） | — | 0.4937 → 0.4228（−14.35%） |

⇒ **case9/case7 的 wall 不是带宽/DRAM/L2 受限**。（与模型自算一致：Q 流量 = 2048 任务 × 16 KB = 33.55 MB，仅占 case9 实现态 GM 流量 2332 MB 的 **1.44%**。）

**wall A/B 结果（3 次 interleaved paired 重复，权威 `op_summary` median 口径）**

| case | arm | r1 | r2 | r3 | 均值 | 重复极差 |
|---|---|---|---|---|---|---|
| case9 | prod | 869.66 | 869.10 | 869.88 | **869.547** | 0.78 µs |
| case9 | probe | 874.40 | 871.34 | 870.41 | **872.050** | 3.99 µs |
| case7 | prod | 856.69 | 856.65 | 856.46 | **856.600** | 0.23 µs |
| case7 | probe | 862.55 | 863.36 | 866.28 | **864.063** | 3.73 µs |

| 统计量 | case9 | case7 |
|---|---|---|
| paired diff（probe − prod） | +4.74 / +2.24 / +0.53 | +5.86 / +6.71 / +9.82 |
| 均值差 | **+2.503 µs = +0.288%** | **+7.463 µs = +0.871%** |
| stdev / SEM | 2.117 / 1.222 | 2.085 / 1.204 |
| t（df=2） | 2.048 | 6.201 |
| **双侧 p** | **0.177 → 不显著** | **0.025 → 显著** |
| 95% CI | [−2.756, +7.763] µs = **[−0.317%, +0.893%]** | [+2.285, +12.642] µs = **[+0.267%, +1.476%]** |
| **`wall(prod) − wall(probe)`** | **−2.503 µs** | **−7.463 µs** |

### 4.2 方法 B — 时间轴识别：**真机粒度不足，如实放弃 per-event 识别；已用计数器差分法直接测量替代**

**尝试 1：`--aic-metrics=TimelineDetail --dump=true`（真机）→ 失败**
```
[ERROR] Kernel context save failed, dump kernel args failed
[WARN]  Msopt dump kernel failed
[ERROR] Parse kernel data failed, kernel name is main_kernel, the order is 0
[WARN]  Profiling kernels result is: 0 success, 1 failed.
[WARN]  TimelineDetail data collection failed
```
产物 `simulator/` 目录**为空**。msprof 帮助文本明示 `--core-id` "only effective when `--aic-metrics=<TIMELINEDETAIL>` and **only effective in simulation products**" ⇒ **per-event MTE2 时间轴是 camodel 专有能力**，而 camodel 被纪律 #1 禁止作定量依据。

**尝试 2：`--instr-profiling=on --instr-profiling-freq=1000`（真机）→ 原始数据采集成功，但无可解析导出**
- 真机 PC 级采样 slice **确实产出**：`device_0/data/instr.group_{1..19}_aic.0.slice_0`、`instr.group_N_aiv0/aiv1`。
- 但 msprof CLI 导出**无任何指令级热点表**：`device_0/sqlite/` 仅含 `ai_core_op_summary / ascend_task / biu_perf / freq / metric_summary / op_counter / step_trace / time / voltage`（**无 instr 表**）；`mindstudio_profiler_output/` 仅含 `api_statistic / op_statistic / op_summary / step_trace / task_time`。解码需 MindStudio，**本环境无 MindStudio**。

⇒ **无法在真机逐事件识别 ~103 个 16 KB Q 装载。如实记为测量缺口，未用模型值或历史值冒充。**

**替代方案（直接测量，且对判决目的更强）：计数器差分法**

由 4.1 的两个 msprof-op arm 直接相减即得方法 B 所求的「总时长」，无需逐事件时间轴：

| 量 | case9 实测 | 来源 |
|---|---|---|
| Q 装载事件数 / 核 | **101.4**（≈102 = 103 任务 − 保留 1 次） | `aic_mte2_instructions` Δ，直接计数器 |
| Q 装载字节 / 核 | **1622.4 KB**（= 101.4 × 16 KB，**精确**） | `GM_to_L1_datas` Δ，直接计数器 |
| **Q 装载 MTE2 busy 总时长 / 核** | **32.40 µs = 3.86% of AIC wall** | `aic_mte2_time` Δ，直接计数器 |
| 单次 Q 装载 MTE2 busy | **0.3195 µs / 16 KB** | 32.40 / 101.4 |
| **非重叠暴露部分** | **≤ 0 µs** | 方法 A wall Δ（−2.50 µs，p=0.177） |
| **⇒ 已重叠效率** | **≈100%（3.86% 的 MTE2 占用已被完全隐藏）** | 上两行相除 |

case7 同法：Q 装载 MTE2 busy 总时长 34.44 µs（4.12% of AIC wall）+ l0a_s 重载 MTE1 13.60 µs；非重叠暴露部分 **≤0（实测显著为负 −7.46 µs）**。

### 4.3 `ramp` 实测值 vs 模型值 134 µs

**模型复算（逐字取自 `analysis_20260928_roofline_model.py:174-178` 的 `else` 分支 + L29-45 常数 + L127-137 `model()`；自校验 `tB_cores` = 625.9 µs vs 记录 626.1（case9）/ 797.4 vs 797.2（case7）⇒ 转录正确）**

| 项 | case9 | case7 |
|---|---|---|
| `t_task = q_task/CM2 + bM·D·db/CFIX + 2·c_pipe` | 0.1309 + 0.2556 + 2.2240 = **2.6105 µs** | 0.1309 + 0.2556 + 2.8220 = **3.2085 µs** |
| `my_cnt` | 103 | 103 |
| **`ramp`（模型）** | **134.44 µs = 15.46% of wall** | **165.24 µs = 19.29% of wall** |
| ↳ Q GM→L1 切片 | **6.74 µs = 5.0% of ramp = 0.78% of wall** | **6.74 µs = 4.1% = 0.79% of wall** |
| ↳ 每任务 FIX 切片 | 13.16 µs = 9.8% = 1.51% | 13.16 µs = 8.0% = 1.54% |
| ↳ `2·c_pipe` 填充切片 | **114.54 µs = 85.2% = 13.17% of wall** | **145.33 µs = 88.0% = 16.97% of wall** |

**偏差裁决**

| 比较 | 偏差 |
|---|---|
| 实测 Q prologue 暴露 vs 模型 `ramp` 全项 134.44 µs | **−134.44 µs（实测 ≤0）⇒ 模型高估 ≥ 无穷倍；按 95% CI 上界 0.33% 计，高估 ≥45×** |
| 实测 Q prologue 暴露 vs 模型 Q 切片 6.74 µs（0.78%） | **−6.74 µs ⇒ 该切片已 100% 被重叠，可回收量为 0** |
| **对任务书前提的更正** | 任务书「`ramp` ≈15%，GATE 2 的 EV 上界完全取决于这个数」**过度归因**：按模型自身算术，GATE 2（`q_l1` 双缓冲）只能攻击 **Q GM→L1 切片 = 6.74 µs = 0.78% of wall**，**在测量之前就已低于 3% 采纳门**。`ramp` 的 85.2% 是 `2·c_pipe` 通用流水填充项，与 `q_l1` 缓冲深度无关。 |

**根因（推断，明确标注为推断而非测量）**：模型的 `else` 分支「逐任务未摊销 ramp」对 **post-#R8 flat 域已过期**。flat 把任务打包进连续 8 槽批，**任务边界不排空流水**（排空/等待在**批**边界：批头 `T.wait_cross_flag(SEM_WS1_V2C)`、批尾 `T.set_cross_flag`，~58 次/核而非 ~103 次/核）；任务边界的 Q 装载是在已运行的槽流中**内联发射**的，其 0.3195 µs/事件的 MTE2 占用被并发的 M/FIX/V 管完全遮蔽。占 `ramp` 85.2% 的 `2·c_pipe` 填充项在 flat 域同样未按任务实现——**证据**：消融连同 token 轮一并删除了整个任务边界串行化，wall 零响应。

### 4.4 **GATE 2 EV 上界带（%）**

> 消融探针在方法学上**严格强于**任何加速类干预：它测的是「无限加速（完全消除）」极限，完美预取不可能优于完全消除。

| case | EV 上界（完全消除 Q prologue 的 wall 收益） | 统计状态 |
|---|---|---|
| case9 | **≤ +0.33%**（95% CI 上界；点估计 **−0.29%**，即负） | p=0.177 不显著 ⇒ 与 0 不可区分 |
| case7 | **≤ 0%（显著为负）**：95% CI **[−1.476%, −0.267%]** 劣化 | **p=0.025 显著** |
| **合并判决带** | **[−1.48%, +0.33%]**，点估计 **[−0.87%, −0.29%]** | — |
| 任务书 0.5~1.0 因子折算 | 因子只作用于**正**的可回收量；实测可回收量 ≤0 ⇒ 折算后仍为 **[−1.48%, +0.33%]** | — |
| **对比 3% 采纳门 / 噪声阈** | **上界 0.33% ≪ 3%，低 ≥9×**；对比模型宣称的 15% 低 **≥45×** | — |

---

## 5. 预筛门裁决（吸收律三步法）

### Step 1 — pacing 管认定（**分别判定**）

| case | 最大真实工作管 | pacing 身份 | 对侧等待余量 | 认定方式 |
|---|---|---|---|---|
| **case9** | AIC MTE2 496.18 µs（59.13%） | **C↔V 四相位交接往返延迟链（latency-bound）**；AIC MTE2 **被消融证伪为 pacing 管**；最大不可削减管 = AIV VEC 468.58 µs（55.80%，消融下逐位零变化） | C 跨核等待 22.1%（id2 21.01%）；**V 跨核等待 64.2%**（id0 32.49 + id4 31.46） | **消融直接验证** |
| **case7** | **AIV VEC 648.99 µs（77.52%）** | **V-paced** | **C 侧 `wait_id2` 63.48%（530.89 µs）= C 侧巨大余量** | **消融直接验证**（移除 C 侧 48.04 µs 真实工作 → wall +0.87% 显著劣化） |

### Step 2 — 方案1（`q_l1` 双缓冲）分类

**= 单侧速度类（C 侧延迟/吞吐改善），并附带「事件类」负项。**

| 类别 | 是否命中 | 依据 |
|---|---|---|
| 字节类（减 GM/L2 流量） | ❌ | Q 字节数不变（仍每任务 16 KB，只是提前发）；本轮实测 Q 流量仅占实现态 GM 流量 1.44%，且 L2 命中率 +2.64pp 仍不移动 wall |
| **结构删除类（删交接/删轮界）** | ❌ **零个交接被删除** | flat 已删轮界（#R8，−8.09%）；`q_l1` 双缓冲不删除任何 SEM/SIG 交接，反而**新增** `SIG_Q_L1+1` token 收支与双槽 side 索引 |
| **事件类** | ⚠️ **负项** | 新增事件（第二套 SIG_Q_L1 token、side 索引算术、预取地址计算）；真实发射侧余量充足（§2）⇒ 不构成硬性阻塞，但净事件数增加 |
| **单侧速度类** | ✅ **主分类** | 只加速 C 侧 Q 装载的暴露；case9 pacing = 交接延迟链、case7 pacing = AIV VEC ⇒ **两者均非 C 侧 Q 装载** |

⇒ 按吸收律：单侧速度类在「pacing 管不在被加速侧」时**被 1:1 吸收**。本轮实测把它推得更远——**过度吸收（over-absorbed，净劣化）**：case9 C 侧真实工作 −32.40 µs → C 跨核等待 +58.75 µs；case7 C 侧 −48.04 µs → C 跨核等待 +70.94 µs。

**这是仓内吸收律的第 5、第 6 例**（前四例 r7-1 / r7-2 / r8-1 / r9-1），且是**首次以「完全消除」上界探针形式给出定量上界**——前四例是「实施某优化后 wall ±0.3%」，本轮是「把被攻击的成本项整项删掉，wall 仍不动甚至变差」，证据强度更高。

### Step 3 — 是否需要减速侧对照实验？

**不需要。理由（方法学）：**

1. **消融上界探针严格强于减速侧对照。** 减速侧对照的作用是「确认被攻击侧不在关键路径上」；而完全消除探针直接给出了**所有加速类干预的收益上界**。上界已 ≤ +0.33%（case9）/ ≤ 0（case7 显著为负），任何减速侧对照的结果都不可能把这个上界抬高。
2. **吸收因果已被直接测量证实，无需间接推断**：`wait_id2` +54.39 µs（case9）/ +57.68 µs（case7）与 C 侧真实工作削减量 −32.40 / −48.04 µs **同向且量级相当甚至更大**，同时 **V 侧真实工作管逐位零变化**（`aiv_vec_time` Δ=0.0000 / +0.0065 µs；V 侧指令数与字节数 Δ=0）——这正是吸收律的完整签名（单侧节省 = 对侧等待增量），三项条件全部满足。
3. 因此**不存在「不能直接进 GATE 2 之外的第三条路径需要对照实验来打开」**的情形；GATE 2 可直接判 NO-GO，不必再花预算。

### 裁决

| 门 | 裁决 | 依据 |
|---|---|---|
| **GATE 2（`q_l1` flat 域双缓冲攻击 ramp）** | ❌ **NO-GO** | ① EV 上界实测 **[−1.48%, +0.33%]**，比 3% 采纳门低 ≥9×（完全消除探针，严格上界）；② 模型自身算术显示可攻击切片仅 **0.78% wall**（6.74 µs），测量前已低于门槛；③ 分类 = 单侧速度类，而 case9 pacing = 交接延迟链、case7 pacing = AIV VEC，**均非 C 侧 Q 装载**；④ 吸收律第 5/6 例，且 case7 为**统计显著劣化**（p=0.025）；⑤ `ramp` 的 85.2% 是 `2·c_pipe` 通用填充项，与 `q_l1` 缓冲深度无关，**方案攻击面与目标项不匹配** |
| **GATE 1（`unit_flag=0b11` 铺垫）** | ❌ **NO-GO（就其"铺垫"定位）**；作为独立候选 = **未实测** | ① 其存在理由 = 为 GATE 2 铺垫，**GATE 2 已实测 NO-GO ⇒ 铺垫对象消失**；② 作为独立候选本轮**未实测**，按纪律 #4 **不作预测性否决**；③ 但已测得的先验需如实告知：`unit_flag=0b11`（据 `docs/api_docs/T.copy.md:102/159` + `examples/deepseek_v4/lightning_indexer.py:666-667`）= 用硬件 mma→fixpipe 流水替代软件 `M↔FIX` flag 轮，属 **C 侧单侧事件删除类**；本轮两度实测证明 C 侧单侧削减被 1:1+ 吸收（case7 显著劣化）⇒ 若要立项，必须**改判攻击面**（转向 V 侧或交接往返链本身），或先补一次 **C 侧 M↔FIX 事件删除类的独立消融上界探针**（本轮方法学可直接复用，成本 ≈ 25 分钟） |
| **缺口①（C 侧 GEMM1/GEMM2 相位串行）** | ⏸ 本轮禁止触碰（用户明示不走设计回退） | 未测量，未裁决 |
| **缺口③（CV-over-GM 三 workspace 往返）** | 📌 本轮获得**新的定量支持** | 两 case 双向大互等（case9：C 22.1% / V 64.2%；case7：C 63.5% / V 28%+）+ 消融下 C 侧真实工作削减被完全转化为跨核等待增量 ⇒ **wall 由 C↔V 交接往返链设定，而该链的物理路径就是三 workspace GM 往返**。与 r10-1 结论「仅剩架构级方案 C」一致，本轮以直接消融证据强化 |

**对既有结论的更新**：`r10-1` 的 bottleneck_type 记为「transfer/structure（case9 占空受限：C.MTE2 顶格 57.8% + wait_id2 25% + 爬坡 15%）」。本轮实测修正为：**`sync`（C↔V 四相位交接往返延迟链）**——三分量中 MTE2 57.8% 成立但**已被证伪为 pacing 管**、`wait_id2` 25% 仅在 max-block 口径成立（均值 21.0%）、爬坡 15% **被直接测量推翻（≤0）**。

---

## 6. 如实声明

### 6.1 已确认事实（直接测量，真机 msprof）

1. 生产态复现：case9 median **869.547 µs**（vs r10_base 869.87，−0.04%）、case7 **856.600 µs**（vs 855.54，+0.12%）。
2. `mha.py` md5 = `dda3ad0877ba4df74bbd559f6912eeed` / 1926 行，**轮次开始、两次探针构建后、轮次结束四点复核全部一致，生产态零变更**；轮末追加 `ast.parse` OK + `ruff check` All checks passed。
2b. **新鲜精度复验：`test_mha.py --level l0` = 10/10 PRECISION_PASS，exit 0**（fp16/bf16 × aligned/causal × D=64/128/256 × scale 变体；max_abs 0.000e+00 ~ 9.766e-04，全部在 gate 内）。
3. C 侧 scalar 真实发射 = `single+dual` = **76.55 µs（case9，9.12% wall / 9.35% scalar busy）**、**62.94 µs（case7，7.53% / 8.23%）**；scalar busy 的 **≈91% 是 stall 归属**。
4. case9 三分量：MTE2 ratio 均值 **56.07%**（带 53.03~59.34）；`wait_id2` 均值 **176.32 µs = 21.01%**，max-ratio block **25.67%**；`ramp` **≤0（实测）**。
5. case7：AIV VEC **648.99 µs = 77.52%**（block0 绝对值 634.387268 µs **逐位复现历史 634.39**）；C `wait_id2` **530.89 µs = 63.48%**（block0 546.20 µs，max-block 66.79%）。
6. 消融机制真实生效：`aic_mte2_instructions` −101.4/核、`GM_to_L1_datas` −1622.4 KB/核（= 101.4 × 16 KB **精确**）、`aic_mte1_instructions` −405.6（case9）/ −1016.6（case7）。
7. Q 装载 MTE2 busy 总时长 = **32.40 µs/核（case9，3.86% of AIC wall）**，单次 **0.3195 µs / 16 KB**；case7 = **34.44 µs（4.12%）**。
8. V 侧真实工作管在消融下**零变化**：case9 `aiv_vec_time` **Δ = 0.0000（逐位相同）**、V 侧 MTE2/MTE3 指令数与 GM↔UB 字节数 Δ = 0；case7 `aiv_vec_time` Δ = +0.0065 µs（+0.00%）。
9. C 侧跨核等待增量：case9 **+58.75 µs**（id2 +54.39 / id5 +4.36）；case7 **+70.94 µs**（id2 +57.68 / id5 +13.26）。
10. wall A/B：case9 **+0.288%**（p=0.177 不显著，95% CI [−0.317%, +0.893%]）；case7 **+0.871%**（**p=0.025 显著**，95% CI [+0.267%, +1.476%]）。
11. L2：消融使 `aic_read_cache_miss_allocate` −34.76%（case9）/ −34.55%（case7），`aic_read_hit_rate` +2.64pp / +2.34pp，**wall 仍不动 ⇒ 非带宽/L2 受限**。
12. 两探针**无 HANG**、输出**数值错误已实测确认**（rel_max 8.447e-01 / 5.900e-01）。
13. 模型 `ramp` 复算 = **134.44 µs（case9）/ 165.24 µs（case7）**，其中 **Q GM→L1 切片仅 6.74 µs = 0.78% / 0.79% of wall**（`tB_cores` 自校验 625.9 vs 626.1 / 797.4 vs 797.2）。
14. 生产态**已合规实现** skill 参考 `best-practices/flash_attn_optimize.md:169-181` 的「持久化 Q 缓冲双向 ownership 闭环」标准形（每任务 1 个完整 `SIG_Q_L1` 轮，全部 `q_l1→L0A` 读在 ownership 区间内，任务尾归还）。GATE 2 属**超出该文档范式**的扩展。
15. 反模式对照（`references/performance-antipatterns.md`）**无未处置项**：「AIC/AIV 混合未开启 CV overlap」不适用（已有 3-workspace + SEM 0-5 + NUM_STAGES=8 flat 批流水，比该条建议的 `num_stages=2` 更深；其补救手段 `T.Pipelined`+`AUTO_CV_COMBINE/SYNC` 被 DESIGN §8.6-1/§3.4 **明令禁止**——**有意保留，理由 = 设计级禁令 + 手动等价物已在位且更激进**）；「冗余全局同步」不适用（`T.barrier_all` 被 §8.6-6 禁止，全部为细粒度 `set_flag/wait_flag` + `set_cross_flag/wait_cross_flag`，正是该条推荐的补救形态）；「纯 AIV memory bound 未做双 buffer」不适用（V 侧 `io_buf`/`io3` 乒乓在案）；「launch core 数」合规（tasks 2048 ≫ cores 20，按 core_num=20 launch）；「tile size 过小」非可调项（`bN` 受 L0B 64 KB 硬限，bN=256 为 r10-1 穷尽条款永久禁令）；该条 L335 的 `cross_interval=2` 调参建议**已被 r3-4 判决证伪**。

### 6.2 推断（非直接测量，已标注）

1. **case9/case7 wall 由 C↔V 四相位交接往返延迟链设定**——由「两侧同时大互等 + 消融下 C 侧节省 1:1+ 转为 C 侧跨核等待增量 + V 侧真实工作管逐位零变化」三项实测签名**推断**；msprof 无「交接往返延迟」直接计数器。
2. **模型 `else` 分支的逐任务 ramp 对 flat 域已过期**——由「消融连同 token 轮删除整个任务边界串行化后 wall 零响应」+「flat 的排空点在批边界（~58/核）而非任务边界（~103/核）」的代码结构**推断**；未对批边界做独立消融。
3. **case7 显著劣化的机理**——候选解释：(a) 探针每槽新增 `fb*NUM_STAGES+i==0` 标量比较 ⇒ **已被实测否证**（case7 `scalar_single_time` −3.43 µs、case9 `scalar_dual_time` −2.24 µs，发射侧实际下降）；(b) 移除 Q 的 MTE2 流改变了 GM/L2 访问交织，降低剩余 K/V/P/w2 流的 DRAM 并行度 ⇒ **部分反向证据**（L2 命中率上升、`aic_mte2_wait_ratio` 下降 14.35%）；(c) Q 装载原为隐式**节拍间隔器**，移除后 C 侧到达信号量的突发更紧，加剧交接竞争 ⇒ 与 `wait_id2`/`wait_id5` 大幅上升一致，但**未独立验证**。**结论：劣化机理未定案**；不影响 NO-GO（NO-GO 只依赖「收益 ≤0」这一实测事实）。
4. **GATE 1 作为独立候选的先验**——由本轮 C 侧单侧削减被吸收的两例**外推**；GATE 1 本身**未实测**。

### 6.3 未验证项 / 测量缺口

| 缺口 | 原因 | 处置 |
|---|---|---|
| **AIV `FLOWCTRL` busy%** | 真机 msprof `PipeUtilization.csv` **无该列**；FLOWCTRL 是 camodel trace 专有概念，纪律 #1 禁止作定量依据 | 如实放弃；提供最接近的真机代理 = `wait_id0+wait_id4+wait_time`（case9 669.55 µs = 79.7% wall；case7 479.66+250.17 µs） |
| **AIC `wait_id0` / `wait_id3` / `wait_id4`** | **硬件语义上不存在**：id0/id3/id4 是 C2V 方向的 C 侧 **set**，AIC 永不 wait ⇒ msprof 不产出该三列 | 如实说明；AIC 侧 `wait_id1/id2/id5` 已全读 |
| **方法 B：真机 per-event MTE2 时间轴** | `TimelineDetail` 真机采集**失败**（`simulator/` 空、`Parse kernel data failed`；msprof 帮助明示仅 simulation products 有效）；`--instr-profiling=on` 原始 slice 采集成功但 **msprof CLI 无指令级导出表**，解码需 MindStudio（本环境无） | **如实放弃 per-event 识别，未用模型值/历史值冒充**；改用**计数器差分法**直接测得 Q 装载事件数/字节/总时长（§4.2），对判决目的等价且更强 |
| **case9 wall Δ 的统计功效** | n=3 paired 重复，df=2 ⇒ p=0.177 不显著 | 已如实报告 CI **[−0.317%, +0.893%]**；**NO-GO 不依赖显著性**，依赖「上界 ≤ +0.33% ≪ 3% 门」；若需收窄可加重复，但上界已远低于门槛，追加采样不改变裁决 |
| **精度复验** | 生产 kernel 逐字节未变（md5 四点复核）⇒ session-8/9/10 在案全量门禁逻辑上继续有效 | **本轮仍新鲜复跑了 `test_mha.py --level l0`：10/10 PRECISION_PASS，exit 0，"Test Passed!"**（隔离缓存 `cache_prod`）。`--level all` / 锚-A / 锚-B / 压力 nondet 未重跑（byte-identical + l0 全绿 ⇒ 无漂移信号）；探针数值**故意错误**，明确排除在任何精度判定之外 |
| **缺口①（GEMM1/GEMM2 相位串行）** | 用户明示本轮禁止触碰（不走设计回退） | 未测量、未裁决 |
| **GATE 1 独立 EV** | 本轮非其测量目标 | **未实测**，不作预测性否决；给出复用本轮方法学的低成本探针路径（≈25 分钟） |
| **case9 之外的 fbm 域（case4）/ Form A 其余成员（case2/3/5/8）** | 预算与判决相关性：case9/case7 已是两域旗舰且结论同向 | 未测；两域同构复现（吸收律先例口径）⇒ 外推风险低，但**未实测即不作为结论** |

### 6.4 环境错误

**无。** 全程未出现 `ImportError` / `ModuleNotFoundError` / `set_env.sh` 相关错误；`source set_env.sh` 正常，torch_npu `device_count=2`，两卡 Health OK，msprof 全部调用 exit 0（TimelineDetail 的 `exit 0` + `[WARN] data collection failed` 属**工具能力边界**，非环境故障）。**本轮消耗 iteration_index = 24，不申请环境错误豁免。**

### 6.5 硬性纪律合规自查

| # | 纪律 | 状态 |
|---|---|---|
| 1 | msprof 为唯一定量依据；camodel 仅定性 | ✅ **本轮完全未运行 camodel** |
| 2 | `disable_cache()` 或会话专属 `TILELANG_CACHE_DIR` | ✅ 4 个隔离缓存目录；生产缓存从未与探针共享；未触碰默认缓存 |
| 3 | 禁止修改 `mha.py`；探针写独立目录；轮末 md5 复核 | ✅ 探针在 `perf_tuning/probe_gate0_qablate/`（`mha_qablate.py` / `mha_qablate_fa.py`，**不以 `mha.py` 命名，不产生 import 遮蔽**）；md5 四点复核全部 `dda3ad08` |
| 4 | 禁止预测性否决 | ✅ GATE 2 由 6 次 A/B + 4 个 metrics arm 实测否决；GATE 1 独立候选**明标未实测** |
| 5 | 禁止读写 `.orchestrator_state.json` | ✅ 全程未读未写 |
| 6 | AGENTS.md 六项核心原则 | ✅ 如实报告（§6.1-6.3 分栏）；先验证再结论（全部数值有 CSV 出处）；区分事实/推断/建议（§6.1/6.2/§5）；最小必要改动（探针 2/3 行 delta，复用既有 `bench_mha.py` CASES 表与 `parse_prof.py` 标定口径） |
| 7 | 本轮禁止触碰项 | ✅ DESIGN §8.6-10 共池相位纪律 / bN=256 / k-v L1 双缓冲 / V 宽通道 / CROSS_INTERVAL / #R5-3 K-V 冗余消除 —— **全部零触碰**（探针仅改 Q prologue 的**执行次数条件**，未改任何缓冲深度、bN、相位结构或 K/V 路径） |

---

## 7. skills_consulted

- `.agents/skills/tilelang-perf-optimization/SKILL.md`（6 步工作流；Step 1 基线采集口径 `msprof op`、Step 2 算子类型判断、Step 3 强制先读参考文档再实施、Double Buffer 的 `[DB-ANALYSIS]` 前置要求）
- `.agents/skills/tilelang-perf-optimization/references/performance-antipatterns.md`（L27 硬件 buffer size 表；L60-124 launch core 数关注项 A/B；L154-196 冗余全局同步；L303-337 AIC/AIV 混合未开启 CV overlap（含 L335 `cross_interval=2` 建议，已被 r3-4 证伪）；L340-383 纯 AIV 未做双 buffer；L420-460 正交轴串行化）
- `.agents/skills/tilelang-perf-optimization/references/best-practices/flash_attn_optimize.md`（L38-103 多级流水与 num_stages；**L104-191 细粒度 Flag 同步机制，尤其 L169-181「持久化 Q 缓冲双向 ownership 闭环」标准形 = 生产态已合规实现的依据**；L195-251 L0 分块与 MMA intrinsic）
- `docs/api_docs/T.copy.md`（L19/L35/L102/L148-159 `unit_flag` 语义：仅 L0C→GM fixpipe 路径生效，`0b10` 累加 / `0b11` flush，须与前序 `T.mma(unit_flag=0b11)` 配对，不可单独使用）—— 用于判定 GATE 1 的机制归类
- `examples/deepseek_v4/lightning_indexer.py`（L538 / L666-667 / L756-764 `unit_flag=0b11` 硬件 mma→fixpipe 流水替代软件 M_FIX/FIX_M pre-set 的仓内先例）—— GATE 1 机制归类的仓内证据

> 未查阅：`optimization-guide.md`（本轮为测量轮，无优化点实施，Step 4 未进入）、`cube_optimization_path.md`、`vector-practices/`、`tilelang-api-best-practices`（本轮未新增/修改任何 API 调用，探针仅改条件表达式）。

---

## 8. 一句话结论

**GATE 0 = FAIL**：真机消融上界探针实测 Q prologue 的可回收暴露成本为 **≤0**（case9 +0.288% p=0.177 不显著；case7 +0.871% **p=0.025 显著劣化**），模型 `ramp` 134 µs（15%）中 GATE 2 可攻击的 Q 切片按其自身算术仅 **6.74 µs = 0.78% wall**，且已 100% 被重叠 ⇒ **GATE 2 NO-GO（EV 上界 ≤ +0.33%，比 3% 采纳门低 ≥9×）；GATE 1 作为其铺垫同步 NO-GO（独立候选未实测，不作预测性否决）**；附带成果 = **C scalar 95.5% 之谜解决（真实发射仅 9.1%/7.5%，≈91% 是 stall，标量余量充足）**、**case9 pacing 身份由「C.MTE2 顶格」修正为「C↔V 四相位交接往返延迟链」**、**吸收律第 5/6 例（首次以完全消除上界形式定量）**。
