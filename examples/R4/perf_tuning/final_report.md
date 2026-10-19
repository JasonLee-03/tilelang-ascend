# R4 fused sampling — Stage 3 性能调优最终报告 (final_report.md)

日期：2026-10-10 · 硬件：Ascend910_9362 (910b2, cube_core=20 / vec_core=40) · CANN 9.2.0 · tilelang 0.1.4+d75300ed · python 3.12.9
目标类型：`best_effort`（无硬性数值目标，以 roofline 达成率 + geomean speedup 提升为准）

---

## 1. 执行摘要

对精度已通过的 R4 算子执行完整性能调优流程（Step 1 基线 → Step 2 类型判定 → Step 3 优化点识别 → Step 4 逐项实施 → Step 5 验证）。
**采纳 2 项核内微优化，geomean speedup（vs 未融合 torch 基线）：fused 6.73× → 10.35× (+53.8%)，mask 11.02× → 15.67× (+42.2%)，全 36 case 无一回退 >3%（worst +0.00%），精度门禁全绿。**

关键结论（任务要求「实测复核，不盲信」已知瓶颈）：msprof K1/K2 位置拆分显示，**真实主导瓶颈并非 split-V 假设的核心利用率，而是两处实现级浪费**：
1. K1 epilogue 的 tie_m `T.Parallel(K_MAX)` 循环被 codegen 降级为 256 次标量 `GetValue/SetValue` + **每轮 `PipeBarrier<PIPE_ALL>`**（全流水 drain）。
2. K2g 每块用 `topk(K=1, N=4096)` 做**完整 Sort(4096)** 仅为取 1 个 argmax。

二者分别由 [#1]/[#2] 消除。split-V（核心利用率）经实测为**次要**瓶颈（仅 case7 类极小 B+超大 V 显著），且属**设计层架构重构**（见 §6），Stage 3 内不实施，建议交 Stage 1。

- **交付 kernel sha256：`bbe47fa4…`（基线）→ `ac8eba47f77e11342eabb9c4e30a44531b3e655b481b0dc28fd5a66a2738fc5b`（优化后）**
- submission 副本 `cann_bench_submission/cann_bench/_r4_kernels.py` 已同步（sha256 一致）。
- 迭代轮数：2 轮采纳（[#1][#2]）+ 1 轮不可行探测（[#4] cumsum）；未触及「连续 3 轮无提升」中止条件。

---

## 2. 用户确认的调优参数（遵守，未自行更改）

| 参数 | 取值 | 遵守情况 |
|---|---|---|
| 目标类型 | best_effort | ✅ 以 roofline 达成率 + geomean speedup 提升为准 |
| 测试 shape | 全 18-case × 2 任务 | ✅ 每优化点跑全 36 对 |
| 噪声阈值 | 3% | ✅ 任何 case 回退 >3% 即判不可采纳（实际 worst +0.00%） |
| 最大迭代数 | 10 | ✅ 用 2 采纳 + 1 探测；安全优化空间已穷尽 |
| 改动范围 | 允许改主文件，收敛后重跑门禁+eval，同步 submission/sha256 | ✅ 见 §7/§8 |

---

## 3. 方法与测量口径

- **性能数据来源**：cann-bench `run_evaluation.sh`（msprof `kernel_details.csv` → 每 kernel 名 median-across-steps 求和；warmup=3 repeat=5，freq_boost + L2 clean，subprocess 隔离）。**未使用任何 Python/torch 计时作为性能结论**（禁令 2）。
- **基线**：复用 20261009 msprof 报告（`cann_final_eval_20261009_11*.json`，kernel sha256 `bbe47fa4` 与调优前一致）+ `split_kernels.py` 的 K1/K2 位置拆分。
- **基线复现校验**：本次重跑 R4MaskSample → 18/18 PASS，逐 case max |drift| = 2.39% < 3% 噪声阈 → 基线可信、无环境漂移。
- **A/B 判据**：候选 kernel 跑同口径全 18-case × 2 任务 eval，逐 case 对比基线，>3% 回退即拒绝。
- **算子类型（Step 2）**：`get_kernel_source()` 实证三 kernel `gemm/mma/cube 使用数=0`，全部计算在 `if ASCEND_IS_AIV` 块内 → **纯 Vector (AIV)**。（注：codegen 为 `KERNEL_TYPE_MIX_AIC_1_2` + 空分配 L0A/L0B，是 AUTO_CV_COMBINE 对纯 Vector 的无害空转，见 [#8]。）

---

## 4. 采纳的优化点

### [#1] K1 epilogue tie_m 标量循环向量化（迭代 1，采纳）
- **根因**：`T.Parallel(K_MAX)` 的 `tie_m[l]=if_then_else(run_v[l]>/== tau_k[0],1,0)` 因标量 `tau_k[0]` 依赖被 codegen 降级为 256 次标量 `GetValue/SetValue` 循环，**每轮 1~2 个 `PipeBarrier<PIPE_ALL>`**（K1_topk_stats.cc L169-193）；n_gt/n_eq 各一次 = 512 次全流水 drain。
- **改动**：`broadcast(tau_k→[K_MAX])` + `compare(GT/EQ)` + `select(1.0/0.0)` + `reduce_sum`（3 向量指令/256 元素）。新增 tie_mask[32]u8 + tie_one[256]f32 + tie_tkbc[256]f32（+~2KB UB）。
- **bitwise 等价**：fp32 GT/EQ 精确；0/1 计数 ≤256 在 fp32 精确，reduce_sum 树序 vs 标量序对整数和无差异 → n_gt/n_eq/tieok/vflag 不变。
- **效果**：PipeBarrier_ALL 768→5。单独贡献 fused geomean 6.73→8.52 (+26.6%)、mask 11.02→15.62 (+41.8%)。

### [#2] K2g 每块 topk(K=1) 全排序 → reduce_max + argmax-index（迭代 2，采纳）
- **根因**：`T.tile.topk(topk_dst, z_ub, 1, BV)` 对 4096 元素做**完整 Sort**（Sort32×128 + MrgSort 归并树，K2_gumbel.cc L160）仅取 1 个 max+index。K2g 是 fused 任务最重 kernel（case15 base=1060us，54%）。
- **改动**：`reduce_max(z_ub)→blk_max` + `compare(z_ub==blk_max,EQ)` + `select(idx?sentinel)` + `reduce_min→最小 argmax index`。idx[BV] loop-invariant hoist；复用 dead buffer g_ub（masked_idx）、mask（EQ mask）；删除 topk_dst（避免 dead alloc_shared codegen 失败）。
- **bitwise 等价**：max 精确无舍入 → blk_max==topk_dst[0]；tie 时 reduce_min 取最小 index == torch.argmax-first == V-b4 验证的 topk(K=1) 行为。全 -inf 块 argmax=0 但 merge `-inf>best_val`=false 不更新（与现行一致）。
- **narrow-reduce 核查**：reduce 为 full-width [BV=4096]=16KB（非 ≤256B narrow 路径），与 K1 `reduce_max([1,4096],dim=-1)` 同 normal 路径。
- **效果**：K2g TopK<>/Sort<> 归零。fused geomean 8.52→10.35（大 V case 7/15/16/17/18 再降 21~23%）；mask 持平（[#2] 不影响 mask，i2 vs i1 全 ±1.8% 噪声内）。

---

## 5. 逐 case 优化前后对比

（after = 收敛 kernel sha256 ac8eba47；base = 20261009 报告 median；attain = t_hw/elapsed）

### r4_fused_sampling (K1 + K2g gumbel)

| case | B | V | b_num | base_us | after_us | Δ% | t_hw | attain before→after | speedup vs torch |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 256 | 1 | 64.05 | 35.26 | -44.9% | 1.00 | 1.56%→2.84% | 6.11× |
| 2 | 1 | 4096 | 1 | 72.28 | 42.26 | -41.5% | 1.00 | 1.38%→2.37% | 9.32× |
| 3 | 8 | 16384 | 4 | 148.48 | 100.18 | -32.5% | 1.06 | 0.71%→1.06% | 7.88× |
| 4 | 4 | 16384 | 2 | 133.06 | 85.70 | -35.6% | 1.00 | 0.75%→1.17% | 7.60× |
| 5 | 7 | 4409 | 4 | 91.13 | 54.98 | -39.7% | 1.00 | 1.10%→1.82% | 11.08× |
| 6 | 8 | 10240 | 4 | 119.76 | 77.60 | -35.2% | 1.00 | 0.84%→1.29% | 8.68× |
| 7 | 4 | 152064 | 2 | 978.77 | 731.02 | -25.3% | 4.89 | 0.50%→0.67% | 3.37× |
| 8 | 4 | 16384 | 2 | 121.76 | 91.70 | -24.7% | 1.00 | 0.82%→1.09% | 7.15× |
| 9 | 64 | 4096 | 32 | 146.97 | 86.24 | -41.3% | 2.15 | 1.46%→2.49% | 14.09× |
| 10 | 4 | 4096 | 2 | 72.70 | 42.30 | -41.8% | 1.00 | 1.38%→2.36% | 12.52× |
| 11 | 8 | 8192 | 4 | 98.50 | 62.76 | -36.3% | 1.00 | 1.02%→1.59% | 9.86× |
| 12 | 16 | 32768 | 8 | 233.61 | 161.50 | -30.9% | 4.23 | 1.81%→2.62% | 11.50× |
| 13 | 6 | 2304 | 3 | 40.14 | 23.58 | -41.3% | 1.00 | 2.49%→4.24% | 17.95× |
| 14 | 32 | 1024 | 16 | 67.94 | 37.60 | -44.7% | 1.00 | 1.47%→2.66% | 13.64× |
| 15 | 64 | 152064 | 32 | 1974.21 | 1471.52 | -25.5% | 78.31 | 3.97%→5.32% | 17.92× |
| 16 | 128 | 32768 | 64 | 986.06 | 697.76 | -29.2% | 33.81 | 3.43%→4.85% | 16.13× |
| 17 | 16 | 152377 | 8 | 986.81 | 737.40 | -25.3% | 19.62 | 1.99%→2.66% | 9.60× |
| 18 | 64 | 65536 | 32 | 896.38 | 652.60 | -27.2% | 33.77 | 3.77%→5.17% | 17.14× |

**geomean speedup vs torch: 6.727× → 10.346× (+53.8%)；geomean roofline 达成率 1.43% → 2.19% (×1.54)**

### r4_mask_sample (K1 + K2m mask)

| case | B | V | b_num | base_us | after_us | Δ% | t_hw | attain before→after | speedup vs torch |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 256 | 1 | 50.35 | 26.10 | -48.2% | 1.00 | 1.99%→3.83% | 6.51× |
| 2 | 1 | 4096 | 1 | 58.98 | 34.16 | -42.1% | 1.00 | 1.70%→2.93% | 11.31× |
| 3 | 8 | 16384 | 4 | 95.27 | 70.32 | -26.2% | 1.00 | 1.05%→1.42% | 12.28× |
| 4 | 4 | 16384 | 2 | 80.72 | 55.84 | -30.8% | 1.00 | 1.24%→1.79% | 12.96× |
| 5 | 7 | 4409 | 4 | 64.91 | 39.74 | -38.8% | 1.00 | 1.54%→2.52% | 13.37× |
| 6 | 8 | 10240 | 4 | 79.93 | 54.66 | -31.6% | 1.00 | 1.25%→1.83% | 11.17× |
| 7 | 4 | 152064 | 2 | 487.61 | 462.76 | -5.1% | 1.35 | 0.28%→0.29% | 9.83× |
| 8 | 4 | 16384 | 2 | 72.21 | 60.02 | -16.9% | 1.00 | 1.38%→1.67% | 12.66× |
| 9 | 64 | 4096 | 32 | 120.23 | 69.60 | -42.1% | 1.00 | 0.83%→1.44% | 12.65× |
| 10 | 4 | 4096 | 2 | 59.18 | 34.10 | -42.4% | 1.00 | 1.69%→2.93% | 14.42× |
| 11 | 8 | 8192 | 4 | 71.78 | 47.30 | -34.1% | 1.00 | 1.39%→2.11% | 11.69× |
| 12 | 16 | 32768 | 8 | 129.04 | 104.04 | -19.4% | 1.17 | 0.90%→1.12% | 17.08× |
| 13 | 6 | 2304 | 3 | 27.23 | 14.84 | -45.5% | 1.00 | 3.67%→6.74% | 24.72× |
| 14 | 32 | 1024 | 16 | 54.38 | 29.22 | -46.3% | 1.00 | 1.84%→3.42% | 13.16× |
| 15 | 64 | 152064 | 32 | 985.92 | 935.08 | -5.2% | 21.54 | 2.18%→2.30% | 54.26× |
| 16 | 128 | 32768 | 64 | 566.07 | 468.66 | -17.2% | 9.34 | 1.65%→1.99% | 24.02× |
| 17 | 16 | 152377 | 8 | 498.11 | 473.86 | -4.9% | 5.40 | 1.08%→1.14% | 31.28× |
| 18 | 64 | 65536 | 32 | 474.88 | 426.28 | -10.2% | 9.30 | 1.96%→2.18% | 30.91× |

**geomean speedup vs torch: 11.018× → 15.671× (+42.2%)；geomean roofline 达成率 1.37% → 1.95% (×1.42)**

> 说明：大 V case（7/15/17）提升较小（-5%）——其 K1 由每块 topk(256,4096) 全排序主导（现有原语无 partial-select），epilogue/K2g 占比小；进一步提升需 split-V（§6）。中小 V case 提升大（-25~-48%），因 epilogue（[#1]）与 K2g（[#2]）占比高。

---

## 6. 拒绝 / 推迟的优化点及原因

| # | 优化点 | 判定 | 原因 |
|---|---|---|---|
| [#3] | **split-V 并行化** | **推迟（设计层增强，交 Stage 1）** | 算法可行（online-softmax/top-256/argmax 均可跨 split 归约，按 block 边界切分可保 topk count%32==0 红线）。ROI 潜力 +30~50% geomean（9/18 小 B case 受益，case7 K1 ~4.7×）。**但**：① 需新增 K1_partial+K1_reduce+K2g_partial+K2g_reduce（4+ kernel）+GM workspace+双路条件分派 = 设计层架构重构，冲突「最小改动/禁止推翻重写」；② **correctness-sensitive**：改变 online-softmax 归约顺序 → s 变 ulp → tau_p/okp 在 p_i 边界可能跳变 → tau 变 → 威胁 **tok_out/valid 精确匹配门禁**（低概率但真实），需 Stage-2 级 bitwise 严谨度；③ 无现成纯 Vector 跨核归约示例可复用。任务指引：设计层问题交 Orchestrator 决定回退 Stage 1，不在 Stage 3 强行打补丁。**非 DESIGN_ERROR（现行设计正确且已 +54%/+42%）。** |
| [#4] | tau_p 前缀和向量化 | 不适用（探测确认） | probe 实证 `T.cumsum` 在 Ascend 纯 Vector + R4_PASS_CONFIGS 下 **InternalError（TVM pass 阶段失败）**，无可用向量 cumsum/scan；tau_p 单调守卫 prefix-scan 有真串行依赖，保留串行。 |
| [#5] | K2m div→mul(reciprocal) | 不适用 | optimization-guide §2.13 L1129 明确反建议（多一步除法引入浮点误差）；K2m 仅占 mask 8%，收益 <噪声阈。 |
| [#6] | K1 topk(256,4096) 全排序 | 不适用 | tilelang `TopK` 模板恒做完整 Sort，无 partial-select/quickselect 原语；需 256 有序值供 tau_k/tau_p/merge。分层归并已是现有原语下最优。 |
| [#7] | 消除 K1 z_ub→zt_ub UB→UB 拷贝 | 暂不适用 | UB→UB 拷贝快（片内 16KB）；改 rank-1 牵动 online-softmax reduce/broadcast 维度，中风险低收益。 |
| [#8] | 关 AUTO_CV_COMBINE（消 L0A/L0B 空分配） | 不适用 | 分析：MIX_AIC_1_2 下 20 块并发×2 AIV=40 AIV 全用；纯 AIV 亦 (cid,vid) 2 子核/块→同样 40 AIV，**并发度不变无占用收益**；L0A/L0B 是 Cube 侧不占 Vector UB。pass_configs 为 §2.10「最后手段」，预测零收益。 |
| [#9] | Double Buffer / T.Pipelined | 不适用 | §2.2.0：纯 AIV Double Buffer 必须 `AUTO_SYNC=False`+手写 set_flag/wait_flag（Expert 形态），与 R4 **Developer 模式**（alloc_shared+全自动同步，DESIGN §7.1 零手写 flag/scope）根本冲突；DESIGN §6.3 已论证 K1 循环携带依赖使 Pipelined 收益有限。 |
| [#10] | 多行 Tile 粒度扩展 | 不适用 | §2.13 针对 GroupNorm 类双循环；R4 一行一任务（topk 语义强制整行驻留单核 UB，DESIGN §6.5）。 |
| [#11] | 归约遍数融合（online softmax 3→2pass） | 不适用（已实现） | K1 已是 online-softmax 单遍融合，达 2-pass 下限。 |
| [#13] | Host 侧预处理内化 / D2H 消除 | 不适用（P0 已合规） | host wrapper 无 F.pad/contiguous/to(dtype)/permute/cat/clone/in-place 算术；无 kernel 输出 D2H 后处理（DESIGN §5.6 已审计）。 |
| [#14] | 特定 dtype 硬件指令适配 | 不适用 | R4 仅 fp32+int32，无 transpose，无非硬件路径 dtype。 |

---

## 7. 精度门禁结果（收敛 kernel sha256 ac8eba47）

| 门禁 | 命令 | 结果 | exit |
|---|---|---|---|
| 全量分层 | `test_r4_fused_sampling.py --level all` | **74 PRECISION_PASS / 0 FAIL + 38 BOUNDARY_PASS / 0 WARN** | 0 |
| 稳定性 | `--level stability`（含于 all，reps≥20） | PASS | 0 |
| UB-poison 压力 | `--level stress`（8 configs × 200 reps） | **8/8 PASS**（selfdiff=0 goldenfail=0 invariant=0） | 0 |
| 覆盖门禁 | `coverage_check.py test_... --proto proto.yaml` | **25 PASS / 0 MISS / 0 N/A** | 0 |
| cann-bench 精度 | eval 18-case × 2 任务 | **36/36 PASS**（fused 18/18 + mask 18/18） | 0 |

- L0-B gumbel tok_out **bitwise ratio=1.0000 / max=0**（[#2] argmax 替换与 topk(K=1) 逐位一致）。
- **精度红线（§4）未突破**：[#1] 不涉及 topk/TAIL；[#2] 消除 K2g topk(K=1)（BV=4096 %32==0，本就对齐），未引入非 32B 倍数 UB→UB 拷贝，未引入 count%32≠0 的 topk。stress（UB-poison）层全绿佐证 TAIL_PAD 约束完好。

---

## 8. 交付物同步 / sha256

- 主 kernel `examples/R4/r4_fused_sampling.py`：`bbe47fa4…` → **`ac8eba47f77e11342eabb9c4e30a44531b3e655b481b0dc28fd5a66a2738fc5b`**
- submission 副本 `cann_bench_submission/cann_bench/_r4_kernels.py`：**已同步**（sha256 == ac8eba47，build.sh verbatim 机制 + 手动 cp 双保险）。
- 备份（回滚点）：`history_version/r4_fused_sampling_impl_s3_attempt0_baseline.py`（bbe47fa4）、`_impl_s3_attempt1.py`（[#1] 后）、`_impl_s3_attempt2.py`（[#2] 后 = 当前 ac8eba47）。
- **t_hw / baseline_raw metadata 的 `kernel_sha256: bbe47fa4` 有意保留**——它们是「基线/roofline 模型针对哪个 kernel 推导」的**溯源记录**，t_hw 是固定 roofline 下界（未对 ac8eba47 重新推导），达成率提升反映实现向同一 roofline 逼近，非模型变更。
- active cann-bench metadata（`bench_lab/.../metadata/910b2.json`）无 kernel sha256 字段，eval 不校验 → 无 mismatch 风险。
- ⚠️ `.orchestrator_state.json` 全程未读写（禁令 1）。其中的「交付 sha256」记录应由 **Orchestrator** 依本报告从 bbe47fa4 刷新为 **ac8eba47**。

---

## 9. 未验证项与残留风险（如实报告）

1. **split-V（[#3]）未实施**：仅完成算法可行性 + ROI + 风险分析，**未编写/未测量**。其 +30~50% geomean 潜力为**分析估计，非实测**。若采纳需 Stage 1 重设计（含归约顺序固定化 / tok 边界 bitwise 策略）。
2. **t_hw roofline 未对 ac8eba47 重新推导**：[#1]/[#2] 移除了实现级浪费（标量循环、full-sort-for-argmax），t_hw 模型的 vec_ops 计数基于 bbe47fa4 的算法本质操作。达成率 before→after 用**同一** t_hw（固定下界），提升来自 elapsed 下降。若需精确 ac8eba47 的 t_hw，须重跑 `hap_thw_model/` 脚本（本次未做，判定为超出 Stage 3 范围且不影响 A/B 结论）。
3. **[#2] 的 bitwise 等价在测试种子/shape 下已验证**（L0-B ratio=1.0、stress 8/8、all 74/0）；理论上 tie 处理（reduce_min 取最小 index）比 topk(K=1) 的排序稳定性**更确定**，但未穷举所有可能输入分布。
4. **噪声阈内判定**：mask 任务 [#2] 前后 i2 vs i1 变化全在 ±1.8%（<3% 噪声），判定为「[#2] 不影响 mask」（符合预期，K2g 不在 mask 路径），非性能回退。
5. 大 V case（7/15/17）绝对达成率仍低（fused 0.67~5.3%，mask 0.29~2.3%）——受 K1 每块 topk(256) 全排序 + 核心利用率双重限制，是 split-V 的目标域，本阶段未解决。

---

## 10. 结论与建议

- **Stage 3 性能调优成功收敛**：采纳 [#1][#2] 两项 bitwise 安全的核内向量化，**fused geomean +53.8% / mask +42.2%，全 36 case 零回退，精度四门禁全绿**，roofline 达成率 fused ×1.54 / mask ×1.42。
- 交付 kernel sha256 = **ac8eba47**，submission 已同步，回滚点齐备（attempt0/1/2）。
- **建议 Orchestrator**：将 **split-V（[#3]）作为 Stage 1 重设计候选**评估——它是剩余最高杠杆（大 V/小 B case，尤其 case7 达成率 0.67%），但属设计层架构重构 + correctness-sensitive（tok_out/valid 精确匹配），应在 Stage 1/2 的 bitwise 严谨度下实施，而非 Stage 3 性能补丁。
