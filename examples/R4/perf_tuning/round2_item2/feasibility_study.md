# Round 2 · Item #2 —— K1 块级阈值剪枝：离线可行性研究（强制前置门）

- **日期**：2026-10-10
- **执行者**：`@tilelang-op-perf-tuner`（Stage 3，round 2 第 3 项）
- **起始交付 kernel sha256**：`19e51e8d8ee126b920cf48c915d863a9be3a349e2c68cf728d35d482b48bcc57`
- **结束时交付 kernel sha256**：`19e51e8d8ee126b920cf48c915d863a9be3a349e2c68cf728d35d482b48bcc57`（**逐字节未变**）
- **判定**：**`[ITEM2_NO_GO]`** —— 收益上限 **0.000000%**（实测）/ **0.002212%**（解析、与种子无关），
  均远低于用户设定的 3% 硬门槛（相差约 **1356 倍**）。
- **未写任何交付 kernel 代码**。`r4_fused_sampling.py`、`test_r4_fused_sampling.py`、
  `cann_bench_submission/cann_bench/_r4_kernels.py` 三者的 sha256 在研究前后完全一致
  （`gate_sha256_before.txt` / `gate_sha256_after.txt`，`diff` 为空）。

---

## 0. 结论摘要（先看这里）

| 判据 | 实测值 | 门槛 | 结论 |
|------|--------|------|------|
| (a) 剪枝率 —— 18 个 bench case | **0 / 5478 块 = 0.000000** | — | 无可剪枝块 |
| (a) 剪枝率 —— 38 个门禁配置 | **0 / 2661 块 = 0.000000** | — | 门禁对剪枝结构性失明 |
| (a) 剪枝率 —— 解析期望上界（与种子无关） | 最大 **1.122e-4**（case 7/15/17，N=37） | — | 实测 0 不是种子偶然 |
| (b) topk+merge 占 K1 的比例 | **26.4%（N=1）→ 78.8%（N=37）** | — | 占比很高（机制本身有价值） |
| **折算 36-case geomean 上限（实测剪枝率）** | **0.000000%** | **> 3%** | **NO-GO** |
| 折算 36-case geomean 上限（解析期望剪枝率） | **0.002212%** | > 3% | NO-GO |
| 反事实上限（假想"完美预言剪枝"100% 命中） | 98.036%（6 个已探测 case） | — | 说明瓶颈在**数据**，不在机制 |
| 第 2 节第 5 点（块内条件分支可实现性） | **可实现**：真实 C++ `if`，Sort 确被跳过 | — | 风险已排除（但无收益可取） |
| 剪枝谓词自身开销（0% 命中时白付） | **+0.100% ~ +0.948%**（均值 +0.185%） | — | 实施将导致**净回退** |

**一句话**：机制在 Ascend 上**完全可实现**、被剪掉的 Sort **确实是 K1 的主导成本（最高 78.8%）**，
但 cann-bench 与四门禁所用的 **i.i.d. 随机 logits 上，剪枝条件几乎永不成立** —— 期望剪枝率上界仅
1.1e-4，因此收益上限为 0.002212%，达不到 3% 门槛。且实施后每块还要白付一次
`PipeBarrier<PIPE_ALL>` + 2×`GetValue` + 分支（实测 +0.185% 均值），**期望净效果为负**。

---

## 1. 范围与方法

### 1.1 严格遵守的前置门约束

- 全程 **未修改交付 kernel**。所有探针都是 `perf_tuning/round2_item2/` 下的**独立新文件**
  （`probe_k1_nosort.py` / `probe_k1_pruned.py`），只用于测量与生成码取证。
- 性能数字全部来自 **msprof**（`torch_npu.profiler`，`ProfilerLevel.Level1`，
  `ExportType.Text`，`schedule(wait=0, warmup=3, active=5, repeat=1)`，读
  `kernel_details.csv` 的 `Duration(us)` 取中位数）—— 与 cann-bench 同一原语、同一统计量
  （依据：`src/kernel_eval/eval/perf_eval.py:88-115`、`src/kernel_eval/base/perf_strategy.py:_median`）。
  **未使用任何 Python / torch 计时**。
- 输入分布**复现 harness 口径**：`DataGenerator(input_dist="uniform")`（`data_generator.py:71`，
  默认 uniform，见 `:333-345`）+ `case_seed = (eval_seed + sha256(case_id_str)[:8] % 2^31) % 2^31`
  （`evaluator.py:317-322`，`eval_seed` 默认 0，见 `cli.py:370`）+ `case_id_str = f"{op_dir}_{case_id}"`
  （`cann_loader.py:392-394`）。
- 环境配方按任务书 §4 执行，cwd 在 repo 根之外；`tilelang` 校验通过：
  `0.1.4+d75300ed0a88b0f4b2eef8a62b43f1930fdae44b` /
  `/home/developer/.local/lib/python3.12/site-packages/tilelang/__init__.py`。
- **未读写** `.orchestrator_state.json`（禁令 1）。**未改** `test_r4_fused_sampling.py` /
  `proto.yaml` / `cases.yaml|csv`（禁令 3）。**未碰 BV**（禁令 4）。**未顺带做 #5 / #1a**（禁令 5）。

### 1.2 剪枝语义的形式化（用于全部测量）

```
tau_k_run(b) := 块 0..b-1 全部元素中第 K_MAX 大的值
              == run_v[K_MAX-1] == min(run_v)     （run_v 恰为已见元素的降序 top-K_MAX）
块 b 可剪枝  <=>  block_max(b) < tau_k_run(b)     （严格 LT，绝不能是 LE）
```

块 0 自然不可剪枝：`run_v` 初值为 `-inf` ⟹ `tau_k_run = -inf` ⟹ `block_max < -inf` 恒假。
这与任务书"前 K_MAX/BV 个块不可剪枝"的要求一致，且**无需特例分支**。

---

## 2. (a) 剪枝率

### 2.1 模拟器自检（先证明工具本身正确）

`prune_rate.py::self_test()` 在 **8 项（4 类构造）答案已知先验**的输入上验证，并逐块交叉校验
`tau_k_run` 与"已见元素真实第 K_MAX 大值"完全相等（`verify=True`）：

| 输入构造 | n_blk_full | 可剪枝 | 期望 | 结果 |
|----------|-----------|--------|------|------|
| 严格降序（sort 后 reverse） | 8 | **7** | 7 = N−1 | ✅ 机制在有序数据上确实生效 |
| 严格升序 | 8 | 0 | 0 | ✅ |
| i.i.d. uniform × 3 trial（各 16 块） | 16 | 0 / 0 / 0 | 0 | ✅ |
| i.i.d. gaussian × 3 trial（各 16 块） | 16 | 0 / 0 / 0 | 0 | ✅ |

> **自检曾抓出一个真实 bug**：初版用 `np.partition(...)[-K_MAX:]` 的结果直接取 `[K_MAX-1]`
> 当阈值，但该切片**内部无序**，取到的是 top-K 集合里的任意元素而非第 K 大值，
> 导致阈值虚高、i.i.d. 输入上**虚报 1/8 块可剪枝**。修正为保持 `run_v` 降序后，
> 降序/升序/i.i.d. 三类先验全部吻合。**若未发现此 bug，本研究会得出完全相反（且错误）的 GO 结论。**

### 2.2 实测：18 个 cann-bench case（uniform 配方，harness 同种子）

`min_margin` = 该行所有块上 `block_max − tau_k_run` 的最小值。**为正即不可剪枝**；越接近 0 越"险"。

| case | B | V | BV | K_MAX | n_blk_full | 总块数 | 可剪枝 | 剪枝率 | min_margin |
|-----:|--:|--:|--:|-----:|-----:|-----:|-----:|-----:|-----:|
| 1 | 2 | 256 | 4096 | 256 | **0** | 0 | 0 | n/a | n/a（仅尾块，结构上不可能） |
| 2 | 1 | 4096 | 4096 | 256 | 1 | 1 | 0 | 0.000000 | n/a（单块） |
| 3 | 8 | 16384 | 4096 | 256 | 4 | 32 | 0 | 0.000000 | +0.3023 |
| 4 | 4 | 16384 | 4096 | 256 | 4 | 16 | 0 | 0.000000 | +35.4011 |
| 5 | 7 | 4409 | 4096 | 256 | 1 | 7 | 0 | 0.000000 | n/a（单块） |
| 6 | 8 | 10240 | 4096 | 256 | 2 | 16 | 0 | 0.000000 | +0.9317 |
| 7 | 4 | 152064 | 4096 | 256 | 37 | 148 | 0 | 0.000000 | +0.0075 |
| 8 | 4 | 16384 | **2048** | **128** | 8 | 32 | 0 | 0.000000 | +0.1248 |
| 9 | 64 | 4096 | 4096 | 256 | 1 | 64 | 0 | 0.000000 | n/a（单块） |
| 10 | 4 | 4096 | 4096 | 256 | 1 | 4 | 0 | 0.000000 | n/a（单块） |
| 11 | 8 | 8192 | 4096 | 256 | 2 | 16 | 0 | 0.000000 | +10.9972 |
| 12 | 16 | 32768 | 4096 | 256 | 8 | 128 | 0 | 0.000000 | +32.1161 |
| 13 | 6 | 2304 | **2048** | **128** | 1 | 6 | 0 | 0.000000 | n/a（单块） |
| 14 | 32 | 1024 | 4096 | 256 | **0** | 0 | 0 | n/a | n/a（仅尾块，结构上不可能） |
| 15 | 64 | 152064 | 4096 | 256 | 37 | 2368 | 0 | 0.000000 | **+0.0016** |
| 16 | 128 | 32768 | 4096 | 256 | 8 | 1024 | 0 | 0.000000 | +0.1063 |
| 17 | 16 | 152377 | 4096 | 256 | 37 | 592 | 0 | 0.000000 | +0.0109 |
| 18 | 64 | 65536 | 4096 | 256 | 16 | 1024 | 0 | 0.000000 | +1.6414 |
| **合计** | | | | | | **5478** | **0** | **0.000000** | |

- **case 1 / 14：`n_blk_full == 0`**，整行走尾块路径 ⟹ 剪枝**结构上不可能**（且尾块是任务书红线，禁止修改）。
- **case 2 / 5 / 9 / 10 / 13：`n_blk_full == 1`**，主循环只跑块 0，而块 0 永不可剪 ⟹ 同样是结构性 0。
  这 5 个 case 合计 82 块，**全部无候选**。
- 真正有候选空间的只有 11 个 case（`n_blk_full ≥ 2`），共 5396 块，**实测可剪 0 块**。

### 2.3 实测：四门禁的全部输入（`randn × 8.0` 配方）

`gate_blindness.py` 从**未修改的** `test_r4_fused_sampling.py` 里用 `ast.literal_eval` 直接解析
`L0_CASES` / `L1_CASES` / `STAB_CASES` / `ADV_CASES`，逐配置重放归并链（同样 `verify=True`）：

| 门禁组 | 配置数 | 总块数 | 可剪枝 | 最大 n_blk_full |
|--------|-------:|-------:|-------:|-----:|
| L0（`--level all`） | 8 | 2186 | **0** | 37 |
| L1（`--level all`） | 18 | 416 | **0** | 37 |
| STAB（`--level stability` / `stress`） | 8 | 52 | **0** | **2** |
| ADV（tail-adversary） | 4 | 7 | **0** | **1** |
| **合计** | **38** | **2661** | **0** | |

> **⚠️ 重大附带发现（门禁失明）**：`--level stability` 与 `--level stress` 共用的 `STAB_CASES`
> 最大 `n_blk_full` 只有 **2**，`ADV_CASES` 只有 **1**；L2 的特殊值用例（`bnd_sparse_nan_w_out`、
> `bnd_sparse_neginf_*`、`bnd_dbound_*`）全部是 `B=4, V=4096, BV=4096` ⟹ `n_blk_full = 1`。
> 也就是说：**即使实施剪枝，四门禁也无法检测到任何剪枝逻辑错误** —— 无论谓词写对、写反、
> 还是彻底写错，门禁输出都完全一致（剪枝路径永不执行）。任务书 §5.4 寄望
> "`--level stress`（UB 投毒）是最直接的检测器"，**这一假设不成立**：投毒检测的是陈旧 UB 读取，
> 而剪枝根本不触发，`l_dst`/`merged` 不会产生新的残留状态。
> ⟹ 实施 #2 将引入**不可验证的死代码**，这是独立于收益的第二个 no-go 理由。

### 2.4 解析上界：实测 0 **不是**种子偶然

对任意连续 i.i.d. 分布（uniform / gaussian 皆可，只依赖连续性）：

- 已见 `n = b·BV` 个样本时，`tau_k_run` 是第 `K_MAX` 大值，其 CDF 值
  `U := F(tau_k_run) ~ Beta(a, c)`，`a = n − K_MAX + 1`，`c = K_MAX`。
- 块 `b+1` 可剪 ⟺ 其 `BV` 个元素全部 `< tau_k_run`，条件概率 `U^BV`，故

```
P(prune) = E[U^BV] = B(a+BV, c) / B(a, c) = Π_{i=0}^{BV-1} (a+i)/(n+1+i)
```

（`analytic_prune_bound.py`，log 空间求和）

| case | N=n_blk_full | E[可剪枝块]/行 | E[剪枝率] | P(最后一块可剪) |
|-----:|----:|----:|----:|----:|
| 3 / 4 | 4 | 5.29e-33 | 1.32e-33 | 5.29e-33 |
| 6 / 11 | 2 | 1.41e-79 | 7.06e-80 | 1.41e-79 |
| 8 | 8 | 3.77e-08 | 4.71e-09 | 3.52e-08 |
| 12 / 16 | 8 | 1.24e-15 | 1.55e-16 | 1.24e-15 |
| 18 | 16 | 9.21e-08 | 5.76e-09 | 6.46e-08 |
| **7 / 15 / 17** | **37** | **4.15e-03** | **1.12e-04** | **8.94e-04** |
| 1 / 14 | 0 | 0 | 0 | n/a |
| 2 / 5 / 9 / 10 / 13 | 1 | 0 | 0 | n/a |

**全集最大期望剪枝率 = 1.122e-4（0.011%）**，出现在 N=37 的三个 case。
直观解释：`P ≈ exp(−K_MAX/b)`，只有当已见块数 `b` 与 `K_MAX=256` 同量级（即 `V/BV ≳ 256`，
`V ≳ 10^6`）时才可能可观。bench 最大 `V = 152377` ⟹ `b ≤ 37` ⟹ 期望率被压在 1e-4 量级。

**这不是实现细节，而是数据分布的固有性质**：i.i.d. 数据下 `block_max` 是**极值序统计量**
（贴近分布上界），而 `tau_k_run` 只是已见数据的 **第 256 大值**（高分位但远非极值）。
一个 4096 元素的 i.i.d. 块几乎必然含有超过运行阈值的元素。剪枝要对**空间有序/聚簇**
（如降序排列）的 logits 才有效 —— 自检里的降序用例 7/8 命中正印证了这一点。

---

## 3. (b) Sort/topk+merge vs online-softmax 成本拆分

### 3.1 msprof 实测（决定性数据）

三个变体，**同一 shape、同一输入、同一 profiler 会话配方**，各自单独 profile K1（不含 K2）：

- `real` = 交付 kernel 的 `r4_topk_stats`（未改动，直接 import）
- `nosort` = 逐行复制 K1，**仅删掉主循环里的 topk+merge**（softmax 路径一字未动）
- `pruned` = 把 topk+merge 包进 `with T.If(block_max >= tau_k_run)`

| case | B | V | BV | K_MAX | n_blk | TAIL | real (µs) | nosort (µs) | pruned (µs) | **topk+merge (µs)** | **占 K1** | pruned−real | 谓词开销 | µs/块 |
|-----:|--:|--:|--:|----:|---:|---:|------:|------:|------:|------:|------:|------:|------:|-----:|
| 2 | 1 | 4096 | 4096 | 256 | 1 | 0 | 32.52 | 23.94 | 31.88 | 8.58 | **26.38%** | −0.64 | −1.968% | 8.580 |
| 3 | 8 | 16384 | 4096 | 256 | 4 | 0 | 64.50 | 28.84 | 64.86 | 35.66 | **55.29%** | +0.36 | +0.558% | 8.915 |
| 7 | 4 | 152064 | 4096 | 256 | 37 | 512 | 419.47 | 90.10 | 419.89 | 329.37 | **78.52%** | +0.42 | +0.100% | 8.902 |
| 8 | 4 | 16384 | 2048 | 128 | 8 | 0 | 52.74 | 21.10 | 53.24 | 31.64 | **59.99%** | +0.50 | +0.948% | 3.955 |
| 15 | 64 | 152064 | 4096 | 256 | 37 | 512 | 835.72 | 177.46 | 840.72 | 658.26 | **78.77%** | +5.00 | +0.598% | 17.791 |
| 16 | 128 | 32768 | 4096 | 256 | 8 | 0 | 426.12 | 142.34 | 429.85 | 283.78 | **66.60%** | +3.73 | +0.875% | 35.472 |

**读数：**

1. **topk+merge 占 K1 的 26.4%（N=1）→ 78.8%（N=37）**，随 `n_blk_full` 单调上升
   （epilogue 是固定成本，主循环里 Sort 占绝对主导）。`n_blk_full ≥ 4` 的区间为 **55.3%–78.8%**。
   ⟹ 任务书对"Sort 是 K1 主体"的判断**成立**，(b) 不是限制因素。
2. **谓词开销实测 +0.100% ~ +0.948%（均值 +0.185%）**。case 2 的 −1.968% 只有 1 个块，
   属噪声（该 case 绝对值仅 32 µs）。⟹ 在剪枝率 = 0 的真实负载上，实施 #2 是**净回退**。
3. `µs/块` 随占用度上升（case 3/7 约 8.9 µs/块 vs case 16 约 35.5 µs/块），
   因为 `b_num = ceil(B/2)` 超过物理核数后每核串行处理多个 block（case 16：`b_num=64`）。
   这说明**降低每块 Sort 成本对高占用 case 收益更大** —— 但前提仍是要有块能被剪。

### 3.2 静态生成码指令普查（交叉印证，任务书认可的另一种口径）

`probe_insn_census.py` 解析主循环体，并按 `src/tl_templates/ascend/common.h:1685-1898`
把 `tl::ascend::TopK` **展开**为硬件指令（`Sort()` = `ArithProgression` + `Sort32` + 归并树
`MrgSort`（`numSegs → ceil(numSegs/4)` 直到 1，每组一次）+ `DataCopy`；`TopK()` = `Sort()` + `DataCopy`）。

**case 15（BV=4096, K_MAX=256）每块：**

| 类别 | 指令构成 |
|------|---------|
| **online-softmax（不可省）** | `copy_gm_to_ub<4096>` ×1；`reduce_max<4096>` ×1；`reduce_sum<4096>` ×1；`Broadcast<4096>` ×1；`Sub[4096]` ×1；`Exp[4096]` ×1；标量 ×6（`Max[1]`×2、`Sub[1]`、`Exp[1]`、`Mul[1]`、`Add[1]`）⟹ **5 趟 4096 元素全宽 pass + 6 条标量** |
| **topk+merge（可剪）** | `TopK(K=256, repeats=128, actual=4096)` → `ArithProgression(4096)` + `Sort32`(**128 reps**) + **43× `MrgSort`**（stages `[(128,32),(32,8),(8,2),(2,1)]`）+ 2× `DataCopy`；`TopK(K=256, repeats=16, actual=512)` → `ArithProgression(512)` + `Sort32`(**16 reps**) + **5× `MrgSort`** + 2× `DataCopy`；`GatherMask` ×2；`copy_ub_to_ub<256>` ×2 ⟹ **合计 48× MrgSort + 144 Sort32 repeats** |

**case 8（BV=2048, K_MAX=128）每块：** `Sort32` 合计 **72 repeats**、**24× MrgSort**
（`TopK(K=128,repeats=64,actual=2048)` → 21 MrgSort；`TopK(K=128,repeats=8,actual=256)` → 3 MrgSort）。

静态口径与 msprof 实测方向一致：softmax 侧是 **O(BV) 的 5 趟线性 pass**，
topk 侧是 **O(BV·log₄BV) 的归并排序网络**（4096 元素 → 48 次 MrgSort + 144 个 Sort32 repeat），
后者主导 ⟹ 与实测 78.8% 吻合。

> **精度红线 2 复核（顺带取证）**：两个 `TopK` 的 `padCount` 在所有 case 均为 **0**
> （`alignedCount = repeatTimes×32` 恰等于 `actualCount`：4096=128×32、512=16×32、
> 2048=64×32、256=8×32）⟹ `Sort()` 的 masked `Duplicate` 残量路径（dav_c220 上失效）
> **从未被走到**，缺陷 A 未被触发。这是 §6 中"若实施则 `actual_num` 必须保持编译期常量"的现有基线。

---

## 4. 折算到 36-case（18 × 2 任务）geomean 的上限

```
ceiling_case = 剪枝率_case × topk_share_of_K1 × K1_share_of_task
ceiling_36   = 1 / geomean(1 − ceiling_case) − 1
```

`K1_share_of_task` 取自 `perf_tuning/round2_item4/k1split_{fused,mask}_cand.json`
—— 即**交付 kernel `19e51e8d`** 的 msprof 数据按位置拆分为 K1/K2（该口径与 round 1 一致）。

| 任务 | K1 占任务比例（实测区间） |
|------|--------------------------|
| `R4FusedSampling`（K1+K2g） | **0.5179 ~ 0.7647** |
| `R4MaskSample`（K1+K2m） | **0.8011 ~ 0.9472** |

（任务书给的 "K1 占 mask 91.5–96.5%、fused 46–79%" 与此同量级；差异来自测量会话与拆分方式，
本报告一律使用自己实测的值。）

### 4.1 三种口径的上限

| 口径 | 剪枝率 | 36-case geomean 上限 | vs 3% 门槛 |
|------|--------|--------------------:|-----------|
| **实测（bench 真实输入）** | 0 / 5478 = **0.000000** | **0.000000%** | **差 ∞（不达标）** |
| **解析期望（与种子无关）** | ≤ **1.122e-4** | **0.002212%** | **差 ~1356 倍（不达标）** |
| 反事实：完美预言剪枝（100% 命中） | 1.0 | **98.036%**（6 个已探测 case） | 达标——但**物理上不可达** |

反事实逐 case（仅供 Orchestrator 判断机制的潜在价值）：

| case | topk_share | K1_share(fused) | cf_fused | K1_share(mask) | cf_mask |
|-----:|-----:|-----:|-----:|-----:|-----:|
| 2 | 26.38% | 0.7581 | 20.001% | 0.9427 | 24.873% |
| 3 | 55.29% | 0.6576 | 36.355% | 0.9309 | 51.464% |
| 7 | 78.52% | 0.5829 | 45.767% | 0.9217 | 72.370% |
| 8 | 59.99% | 0.5900 | 35.396% | 0.9004 | 54.015% |
| 15 | 78.77% | 0.5841 | 46.008% | 0.9193 | 72.413% |
| 16 | 66.60% | 0.6230 | 41.487% | 0.9286 | 61.843% |

> 这张表是本研究最有信息量的副产品：**K1 的性能确实被 Sort 主导（最高 78.8%），
> 而且 K1 占两个任务的 52%–95%**。所以"减少 Sort 工作量"这个方向本身极有价值 ——
> 但**块级阈值剪枝不是能在 i.i.d. logits 上减少 Sort 的手段**。
> 真正可行的替代方向见 §7.3（不属本轮范围，未实施、未测）。

---

## 5. 第 2 节第 5 点结论：块内条件分支在 Ascend 上的可实现形态

**结论：可实现，且是真正的运行时分支，Sort 会被真实跳过。"只能 mask 化 ⟹ 收益归零"的风险不成立。**

### 5.1 直接证据（生成码）

`emitted_cc/K1_case15_pruned.cc:126-139`（探针 `probe_k1_pruned.py` 编译产物）：

```cpp
      AscendC::PipeBarrier<PIPE_ALL>();
      if (tau_run.GetValue(0) <= m_blk.GetValue(0)) {
        tl::ascend::TopK<float>(l_dst[0], z_ub[0], tmp_ub.ReinterpretCast<float>()[0], 256, 128, 4096);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::GatherMask<float>(l_val[0], l_dst[0], 1);
        tl::ascend::copy_ub_to_ub<float, float, 256>(merged[0], run_v[0], 1, 256, 256, 1, 256, 512);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::copy_ub_to_ub<float, float, 256>(merged[256], l_val[0], 1, 256, 256, 1, 256, 512);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::TopK<float>(m_dst[0], merged[0], tmp_ub.ReinterpretCast<float>()[0], 256, 16, 512);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::GatherMask<float>(run_v[0], m_dst[0], 1);
```

对照 `emitted_cc/K1_case15_real.cc:126`：同一行 `TopK` **不在任何 `if` 内**，无条件执行。

要点：

1. 这是**真正的 C++ `if` 语句**，不是谓词化/mask 化的 `Select`。两个 `TopK`（即两次 `Sort`）
   完整位于分支体内 ⟹ 条件为假时**一条 Sort 指令都不执行**。
2. 条件 `tau_run.GetValue(0) <= m_blk.GetValue(0)` 中的 `GetValue(0)` 是**设备侧从 UB 读标量到通用寄存器**，
   即**数据依赖的运行时分支**，编译期不可知 —— 正是所需形态。
3. 静态循环边界 `T.serial(n_blk_full)` **未被改动**（生成码仍是 `for (int32_t b = 0; b < 37; ++b)`），
   符合"Ascend 只支持静态循环边界"的约束；条件执行发生在**块内**，不需要动态循环边界。

### 5.2 API 依据（未凭记忆猜测，禁令 6）

| 用法 | 依据 |
|------|------|
| `with T.If(<UB 标量比较>): with T.Then():` 包裹块内语句 | **交付 kernel 内已有两处 NPU 验证先例**：`r4_fused_sampling.py:300-303`（epilogue 的 `tau_p` 前缀和循环）、`r4_fused_sampling.py:547-550`（K2g 跨块 argmax 归并 `T.If(blk_max[0] > best_val[0])`）。二者都已通过四门禁 + cann-bench 36/36 |
| 生成码形态（真实 `if` + `GetValue`） | 同文件生成码实证：`round2_item4/emitted_cc_case15/before.cc:190`（`if ((1 <= k_i.GetValue(0)) && ...)`）、`:199`、`:237`、`:241`、`:245`、`:249` |
| `block_max` 无需额外规约 | 主循环已有 `T.reduce_max(z_ub, m_blk[0,0:1], dim=-1)`（`r4_fused_sampling.py:212`），剪枝谓词可直接复用 `m_blk` ⟹ **谓词本身不引入新的全宽 pass** |
| `tau_run[0,0] = run_v[K_MAX-1]` 标量取元素 | 同文件 `:284` `tau_k[0] = run_v[k_i[0] - 1]`、`:302-303` `prefix[0,0] = prefix[0,0] + q2[0,l]` 先例 |

### 5.3 但该形态带来一个**代价**与一个**隐患**

- **代价（已实测）**：分支前多一条 `AscendC::PipeBarrier<PIPE_ALL>`，每块多 2 次 `GetValue` + 1 次分支判断，
  每个**保留**块还多一次 `tau_run[0,0] = run_v[K_MAX-1]`（UB 标量读 + `SetValue`）。
  实测净开销 **+0.100% ~ +0.948%**（§3.1）。在剪枝率 = 0 时，这是**纯亏损**。
- **隐患（新发现，见 §7.1）**：为避开 `T.Else` 而写成 `keep = (block_max >= tau)`，
  在 **NaN 输入下与 `prune = (block_max < tau)` 不等价**。

---

## 6. 任务书要求的正确性点复核（独立复核，非盲信）

| # | 要求 | 复核结论 |
|---|------|---------|
| 1 | 单调性：`tau_k_run` 单调不减 | **成立**。`S_b ⊂ S_{b+1}` ⟹ 超集的第 `K_MAX` 大值 ≥ 子集的第 `K_MAX` 大值（标准序统计量性质）。实测侧：模拟器以 `verify=True` 对 **5478 + 2661 + 自检** 每一块交叉校验 `tau_run == 已见元素真实第 K_MAX 大值`，全部相等 ⟹ `run_v` 语义与推导一致，单调性随之成立 |
| 2 | 剪枝条件必须严格 `<` | **确认**。生成码为 `if (tau_run <= m_blk) { …topk… }`（keep 侧），其对偶即 `prune ⟺ m_blk < tau_run`，是**严格 LT**。`block_max == tau_k_run` 的块**被保留** ⟹ `n_eq` / `tau_p` / `L` 不被破坏。⚠️ 但 NaN 下两形式不等价，见 §7.1 |
| 3 | 前 K_MAX/BV 个块不可剪枝 | **成立**。`BV ≥ K_MAX` 在全部 18 case 成立（4096≥256、2048≥128），块 0 归并后 `run_v` 即被真实值填满。且块 0 时 `tau_run = -inf`（`run_v` 初值），`block_max < -inf` 恒假 ⟹ **块 0 天然不可剪，无需特例** |
| 4 | `run_v` 语义必须完全不变 | **成立**（在探针中）。被剪枝块不写 `run_v`，而剪枝的充要条件正是"该块无元素能进入全局 top-K_MAX" ⟹ `run_v` 仍恰为全局 top-K_MAX 的有序值。注：探针未做输出比对（因 NO-GO 未实施），此为**推导**而非实测 |
| 5 | online-softmax 路径逐字节不动 | **成立**（在探针中）。`probe_k1_nosort.py` / `probe_k1_pruned.py` 的 softmax 段与交付 kernel 逐行相同；`probe_k1_pruned.py` 由 `probe_k1_nosort.py` **文本变换**生成，`diff` 显示差异**仅限**：函数名、docstring、`tau_run` 缓冲声明+fill、以及被 `T.If` 包裹的 topk 区段（见 §8 的 diff 记录）。`m_run`/`s_run` 更新对**每个块**都执行，包括被剪枝块 |
| 6 | 尾块路径（`if TAIL > 0:`）禁止修改 | **成立**。探针的尾块区段与交付 kernel 完全相同；`T.If` 只包裹主循环内的 topk+merge。**交付 kernel 本身未被触碰**，故无需尾块 sha256 前后对比（`gate_sha256_before.txt` == `gate_sha256_after.txt`，全文件级一致） |
| 7 | 精度红线 1（非 32B 倍数 UB→UB 拷贝） | **未引入任何新拷贝**。探针仅新增 `tau_run` 一个 `[1,1]` fp32 标量缓冲（走 `T.tile.fill` / 标量赋值，非 `T.copy`）；`actual_num` 与所有 `copy_ub_to_ub` 计数保持 `256`（=1024 B，32B 倍数）不变 |
| 8 | 精度红线 2（`count % 32 != 0` 的 topk） | **不受威胁**。关键点：剪枝**不改变任何 `actual_num`** —— 它只决定"是否执行"，不改变"执行时的元素数"。生成码中两个 `TopK` 的 `actualCount` 仍是编译期常量 **4096 / 512**（case 8：**2048 / 256**），`padCount` 实测均为 **0**（§3.2）。⟹ 任务书担心的"`actual_num` 变成动态值"**不会发生**，因为 `T.If` 是控制流而非张量长度参数 |

---

## 7. 附带发现（对 Orchestrator 有价值，均**未实施**）

### 7.1 ⚠️ NaN 下 `>=` 与 `<` 不等价（若将来实施必须处理）

为避开 `T.Else`，自然的写法是 `keep = (block_max >= tau_k_run)`。但在 fp32 NaN 语义下：

| block_max | `prune = (block_max < tau)` | `keep = (block_max >= tau)` | 是否等价 |
|-----------|----------------------------|----------------------------|---------|
| 正常值 | — | — | ✅ 等价 |
| **NaN** | `NaN < tau` = **false** → **保留** | `NaN >= tau` = **false** → **剪掉** | ❌ **不等价** |

即：单分支写法会**静默丢弃含 NaN 的块**，使 `run_v` / `tau` / `valid` 与现行语义分叉。
`tau_k_run` 自身为 NaN 时更严重（`NaN <= x` 恒假 ⟹ 后续所有块被剪，`run_v` 冻结在 NaN）。
DESIGN §5.7 把 NaN 列为"不支持输入、语义未定义"，门禁里也只有一个 `bnd_sparse_nan_w_out`
且期望 WARN —— 但那条用例 `n_blk_full = 1`，**对本差异完全不敏感**（§2.3）。
⟹ 若将来实施，必须用 `T.If(prune) / T.Else(keep)` 双分支形式，或显式论证 NaN 不可达。

### 7.2 #4 腾出的 16384 B UB 预算，#2 **根本用不到**

剪枝只需新增一个 `tau_run` `[1,1]` fp32 标量缓冲（UB 按 32 B 粒度分配 ⟹ 约 **32 B**），
外加复用已有的 `m_blk`（block_max）与 `run_v`。
⟹ round 2 #4 "排在 #2 之前以腾预算"的排序理由，**对 #2 并不构成约束**；
那 16384 B 对 #2 是闲置的。若 Orchestrator 要继续用这笔预算，应重新评估其归属（例如 #1a 的部分和缓冲）。

### 7.3 若目标是"减少 K1 的 Sort 工作量"，更有希望的方向（**未实施、未测、仅为建议**）

本研究证明了 **Sort 占 K1 的 55%–79%**、**K1 占任务的 52%–95%**，这是真实的优化空间。
但块级阈值剪枝在 i.i.d. logits 上无效。可能更有效的方向（需各自独立立项验证）：

1. **把 `TopK(K=256)` 的全排序换成"选择"而非"排序"**：当前 `Sort()` 对 4096 元素做完整归并排序
   （48× MrgSort），但 K1 只需要**第 K_MAX 大的值**与 top-K_MAX 的**集合**（`run_v` 后续只被
   按降序遍历用于 `tau_p` 前缀和与 D-TIE 计数）。若硬件/库支持 partial-select（如 `TopK` 的
   截断归并、或先按块内阈值粗筛再排序），可省下大部分归并 stage。
   **风险**：`tau_p` 依赖 `run_v` 的**严格降序**，任何"只选不排"的方案都必须补一次 256 元素排序。
2. **两级块粒度**：把 `BV=4096` 的 topk 拆成"先对 4096 做便宜的粗筛（如按 256 分段取段内 top-N），
   再对小集合排序"。这与剪枝不同，它对 i.i.d. 数据同样有效。
   **风险**：`BV` 是 cann-bench 契约属性（#3 已判定 BLOCKED），只能改**内部**粒度而非 `BV` 本身。
3. **反模式清单「关注项 B：任务数低于物理 AI Core 数」在 K1 上真实命中**
   （round 2 §23.3 已记录：case 7 `b_num = ceil(4/2) = 2`，仅用 2/20 核）。
   这是 #1a（split-V）的动机，与 Sort 成本正交，**可能是更大的收益来源**。

---

## 8. go / no-go 判定

```
收益上限 = 剪枝率 × topk+merge 占比
        = 0.000000 × (0.2638 ~ 0.7877)
        = 0.000000
折算 36-case geomean = 0.000000%   ≤ 3%   ⟹  NO-GO
（解析期望口径：1.122e-4 × 0.7877 × 0.9193 → 36-case geomean = 0.002212% ≤ 3% ⟹ NO-GO）
```

**判定：`[ITEM2_NO_GO]`。按任务书 §1 的硬门槛，停止，不写 kernel 代码。**

### 8.1 因此**没有**做的事（如实列出）

| 事项 | 状态 |
|------|------|
| 修改 `r4_fused_sampling.py` | **未做**（sha256 未变） |
| 备份 `history_version/r4_fused_sampling_impl_s3r2_item2_attempt0.py` | **未做**（无改动可备份；`cp` 只会产生一个与 `19e51e8d` 相同的冗余副本） |
| 同步 `cann_bench_submission/cann_bench/_r4_kernels.py` | **无需**（仍与交付 kernel 逐字节相同，已 `cmp` 验证 IDENTICAL） |
| 尾块区段 sha256 前后对比 | **无需**（全文件 sha256 前后一致，是更强的证据） |
| `actual_num` 编译期常量证明 | **仅对探针给出**（§6 #8）；交付 kernel 未变，无需证明 |
| UB 实测（18 case × K1）前后对比 | **未做**（无改动）。§7.2 给出的是**分析值**（约 +32 B），非 18-case 实测 |
| 精度判据（bitwise / 容差） | **不适用**（无候选版本）。附带说明：由于剪枝率 = 0，任何正确实现的剪枝版本在 bench 与门禁输入上都会是 **bitwise 一致**的（归并链中间状态完全不变）—— 但这只是推论，**未实测** |
| cann-bench 36-case A/B 评测 | **未做**（无候选版本，A/B 无意义）。故本报告**不提供** 36-case geomean 提速 / worst-case delta / 逐 case 对比 |
| 新增 JIT 特化数与编译耗时 | **交付侧为 0**。探针侧：3 个变体 × 2 个 case = 6 次编译（研究用，不进交付） |

### 8.2 `[DESIGN_ERROR]` 判定：**不触发**

任务书要求"至少尝试 3 种优化方案后"才可标注。本项**在离线研究阶段即判定不可行**，
未进入实施，也未尝试 3 种方案。且这不是设计层错误 —— 设计文档并未主张剪枝可行，
`#2` 是 round 2 的**候选优化项**，本研究正是它应有的前置筛选。
**根因是数据分布（i.i.d. logits）与机制（阈值剪枝）不匹配，不是 tiling 策略或内存层级安排缺陷。**

---

## 9. 验证缺口（如实列出）

1. **(b) 只探测了 6 / 18 个 case**（2, 3, 7, 8, 15, 16）。其余 12 个 case 的 topk_share 标记为
   `n/p`（not probed），**未做外推**。由于剪枝率恒为 0，上限乘积与 topk_share 无关，
   故该缺口**不影响 NO-GO 判定**；但 §4.1 的反事实上限只覆盖这 6 个 case。
2. **探针未做输出正确性比对**。`probe_k1_pruned.py` 只用于生成码取证与计时，
   **没有**验证它在任何输入上与交付 kernel 输出一致（因 NO-GO 未实施，无需验证）。
3. **msprof 探针未复现 harness 的 `CannBenchCacheClean`**（harness 在每个测量 step 前清 L2）。
   绝对 µs 可能比 harness 口径略乐观；但三个变体在**完全相同**条件下测量，
   **比值**（上限公式唯一用到的量）不受影响。已在 `probe_run_perf.py` docstring 中显式声明。
4. **`kernel_details.csv` 行数偶尔多于 5**（case 7/15/16 的 real/pruned 出现 6–7 行，
   期望 5 行）。取的是**中位数**，对个别多余行稳健；但多余行的成因（疑似 profiler step 边界
   或 pre-flight 残留）**未定位**。`nosort` 变体在全部 6 个 case 都恰为 5 行。
5. **`K1_share_of_task` 来自上一个会话**（`round2_item4/k1split_*_cand.json`）。
   任务书指出存在 ~0.3–1.7% 会话级漂移。本处用的是**同会话内的 K1/K2 比值**，漂移大部分抵消；
   且它只出现在"反事实上限"里（ informational ），**不进入** NO-GO 判定链。
6. **解析模型假设 logits 连续 i.i.d.**。bench 的 uniform 配方满足；若真实业务 logits 有
   **空间有序性**（例如上游做过排序、或按频率聚簇），期望剪枝率会显著上升 ——
   自检的降序用例（7/8 命中）即为极端例子。**cann-bench 契约内不存在这种输入**，
   故对本任务的采纳判据无影响；但若 Orchestrator 关心生产场景，这是一个需要单独取样的问题。
7. **未测量 `T.Else` 双分支形态的生成码**（§7.1 的 NaN 安全写法）。因 NO-GO 未实施，
   该形态是否能在 Ascend 上正确 codegen **未验证**。
8. **`.orchestrator_state.json` 全程未读写**（禁令 1）。

---

## 10. 复现方法

```bash
cd /mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4   # cwd 必须在 repo 根之外
export PYTHONPATH=/home/developer/Ascend/cann-9.2.0/python/site-packages:/home/developer/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe:
export ACL_OP_INIT_MODE=1
export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:$LD_LIBRARY_PATH
D=perf_tuning/round2_item2

# (a) 剪枝率：18 bench case + 门禁配方（含 8 项先验自检）
python $D/prune_rate.py --also-gate --out $D/prune_rate_result.json
# (a') 门禁失明审计：38 个门禁配置
python $D/gate_blindness.py $D/gate_blindness.json
# (a'') 解析上界（与种子无关）
python $D/analytic_prune_bound.py $D/analytic_bound.json
# 生成码 dump（第 5 点取证）+ 静态指令普查（(b) 交叉印证）
python $D/probe_dump_cc.py $D/emitted_cc 15 8
python $D/probe_insn_census.py $D/emitted_cc > $D/insn_census.txt
# (b) msprof 实测成本拆分（需 NPU；6 case × 3 变体）
export TILELANG_CACHE_DIR=/tmp/opencode/r4_item2/tlcache
python $D/probe_run_perf.py $D/probe_perf.json 2 3 7 8 15 16
# 上限折算到 36-case geomean
python $D/ceiling_fold.py $D/ceiling.json
# 四门禁（在未改动的交付件上复验）
bash $D/run_gates.sh
```

---

## 11. 产物清单

| 文件 | 内容 |
|------|------|
| `feasibility_study.md` | 本报告 |
| `prune_rate.py` / `prune_rate_result.json` / `prune_rate.log` | (a) 剪枝率模拟器（含 8 项先验自检 + 逐块 tau 交叉校验）与结果 |
| `gate_blindness.py` / `gate_blindness.json` / `gate_blindness.log` | §2.3 门禁失明审计（38 配置 / 2661 块） |
| `analytic_prune_bound.py` / `analytic_bound.json` | §2.4 解析期望剪枝率（Beta 序统计量精确式） |
| `probe_k1_nosort.py` | 探针：K1 去掉 topk+merge（softmax 路径逐行同交付件） |
| `probe_k1_pruned.py` | 探针：topk+merge 包进 `T.If`（第 5 点取证 + 谓词开销计时） |
| `probe_dump_cc.py` / `emitted_cc/*.cc` / `insn_census.txt` | 生成码 dump（real/nosort/pruned × case 15/8）+ 静态指令普查 |
| `probe_insn_census.py` | 静态普查器（含 `TopK`→`Sort32`/`MrgSort` 展开模型） |
| `probe_run_perf.py` / `probe_perf.json` / `probe_perf.log` | (b) msprof 实测（3 变体 × 6 case） |
| `ceiling_fold.py` / `ceiling.json` | §4 上限折算到 36-case geomean |
| `run_gates.sh` / `gate_all.log` / `gate_stability.log` / `gate_stress.log` / `gate_coverage.log` / `gate_exits.txt` | §12 四门禁复验 |
| `gate_sha256_before.txt` / `gate_sha256_after.txt` | 交付件 sha256 前后对比（一致） |

\* 已废弃标记：`prune_rate.log` **已落盘**（本文件同目录）。该脚本被**独立复跑过一次**，
两次结果完全一致（`AGGREGATE over 18 cases: prunable 0 / 5478 blocks rate=0.0`，
`prune_rate_result.json` 的 `bench_totals` 亦为 `{total_blocks: 5478, prunable_blocks: 0}`）
⟹ 剪枝率测量**确定性可复现**，不是单次运行的偶然。

`prune_rate_result.json` 另含 `--also-gate` 的**补充测量**：在门禁配方（`manual_seed(0); randn(B,V)*8.0`）下
另取 6 个形态（`(4,16384)`、`(64,152064)`、`(8,8192)`、`(6,2304/BV2048)`、`(4,4096)`、`(128,32768)`）
共 **3434** 个主块，可剪枝 **0**。这与 §2.3 的 38 配置 / 2661 块审计是**两组独立取样**，结论一致。

---

## 12. 四门禁复验（在**未改动**的交付件上）

按任务书 §7 执行，cwd = `examples/R4`：

| 门禁 | 命令 | exit | 计数 |
|------|------|-----:|------|
| all | `python test_r4_fused_sampling.py --level all` | **0** | **74** `PRECISION_PASS` / **0** `PRECISION_FAIL`；**38** `BOUNDARY_PASS` / **0** `BOUNDARY_WARN`；`Test Passed!` |
| stability | `python test_r4_fused_sampling.py --level stability` | **0** | 8 configs × 20 reps 全 `PRECISION_PASS`（`selfdiff=0 goldenfail=0 invariant=0`）；4 tail-adversary 全 **`dropped=0`**，TAIL=**313 / 257 / 511 / 261**（+ 对照 TAIL=2048 / TAIL=0） |
| stress | `python test_r4_fused_sampling.py --level stress` | **0** | **8/8** UB-poison `PRECISION_PASS`，`reps=200 selfdiff=0 goldenfail=0 invariant=0`；`Test Passed!` |
| coverage | `coverage_check.py test_r4_fused_sampling.py --proto proto.yaml` | **0** | **25 PASS / 0 MISS / 0 N/A → PASS** |

sha256（门禁运行前后各取一次，`diff` 为空）：

```
19e51e8d8ee126b920cf48c915d863a9be3a349e2c68cf728d35d482b48bcc57  r4_fused_sampling.py
02a44b79624e477c570c3c632fcdcef01dcf60041de3372a177aa947d53efb50  test_r4_fused_sampling.py
19e51e8d8ee126b920cf48c915d863a9be3a349e2c68cf728d35d482b48bcc57  cann_bench_submission/cann_bench/_r4_kernels.py
```

> 这四项门禁是对**未改动**交付件的复验（本项产出无候选版本）。它们证明的是
> "研究过程没有污染交付件、当前状态仍全绿"，**不是**"某个优化通过了门禁"。
