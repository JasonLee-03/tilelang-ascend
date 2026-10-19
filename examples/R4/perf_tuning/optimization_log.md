# R4 fused sampling — Stage 3 优化记录 (optimization_log.md)

算子类型（Step 2 判定）：**纯 Vector（AIV）**。
证据：`get_kernel_source()` 三个 kernel（K1 r4_topk_stats / K2 r4_mask_sample / K2g r4_gumbel_sample）
均满足 `gemm/mma/cube 使用数 = 0`，全部计算在 `if ASCEND_IS_AIV { ... }` 块内。
⚠️ 注意：三 kernel 均被 codegen 成 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` 并
`InitBuffer(ascend_l0a/l0b, 65536)`——这是 `TL_ASCEND_AUTO_CV_COMBINE=True` 对纯 Vector 算子的
「无害空转」副作用（DESIGN §7.3 已记录），L0A/L0B 被分配但从不使用。构成 antipatterns 末项
「纯 Vector 算子的 AUTO_CV_COMBINE」关注项，但 R4 用 `alloc_shared`（非 `alloc_var`）且精度已验证，
未发生误分核 → 见 [#8]。

## 基线（Step 1，复用 20261009 msprof 报告 + 本次复现校验）

- 数据来源：`bench/reports/cann_final_eval_20261009_11*.json`（kernel sha256 `bbe47fa4…` 未变）。
- 复现校验：本次 `run_probe.sh R4MaskSample base_repro` → 18/18 PASS，逐 case max |drift| = 2.39% < 3% 噪声阈 → 基线可信。
- eval 成本：单 operator ~114s（18 case），both ~4min。
- 任务构成：`r4_fused_sampling` = K1 + **K2g(gumbel)**；`r4_mask_sample` = K1 + **K2m(mask)**。
- geomean speedup vs torch：fused **6.73×** / mask **11.02×**。
- 瓶颈定位（case15 B=64 V=152064，msprof K1/K2 位置拆分）：
  - mask 任务：K1=911.6us（92%）+ K2m=74.8us（8%）
  - fused 任务：K1=914.5us（46%）+ K2g=1059.6us（54%）
- 核心利用率根因：grid `b_num=ceildiv(B,2)`，V 维在块内 `T.serial(n_blk)` 串行；
  case7(B=4)→b_num=2，roofline 达成率 **0.50%**（最低）。多数 case b_num<20（欠占用）。

---

## Part A — 优化点清单

> 参考文档均已阅读：`performance-antipatterns.md`（全文）、`optimization-guide.md`
> （§2.2/2.2.0/2.2.1、§2.6、§2.7、§2.9、§2.10、§2.11、§2.13、§2.15、§2.16、§3.1/3.2）、
> `vector-practices/vector_reduce_pass_fusion.md`（全文）、`best-practices/vector_add_pipeline.md`（全文）。

[#1] **K1 epilogue tie_m 标量循环向量化**（参考: antipatterns L124-150「Vector Core 内逐元素 for loop」+ optimization-guide §2.6 L381-400 指令向量化）：**适用** —
`get_kernel_source()` 实证：K1.cc L169-178 / L184-193 两个 `T.Parallel(K_MAX)` tie_m 填充循环
被降级为 256 次标量 `GetValue/SetValue` 循环，**每轮 1~2 个 `PipeBarrier<PIPE_ALL>`**（全流水 drain）。
n_gt=count(run_v>tau_k)、n_eq=count(run_v==tau_k) 可改为 `T.tile.compare(scalar src1) + T.tile.select + T.reduce_sum`
（3 条向量指令 / 256 元素）。结果 **bitwise 相同**（0/1 计数 ≤256 在 fp32 精确；GT/EQ 语义与标量一致）。
低风险，每行 epilogue 都执行 → 全 case 受益（小 V case 收益最大）。

[#2] **K2g 每块 topk(K=1) 全排序 → reduce_max + argmax-index**（参考: optimization-guide §2.15 L1321-1343 argmax 索引模式 + NaN 处理）：**适用** —
`get_kernel_source()` 实证：K2_gumbel.cc L160 `TopK<float>(...,K=1,repeatTimes=128,actual=4096)`
= 对 4096 元素做**完整 Sort**（Sort32×128 + MrgSort 归并树）仅为取 1 个 max+index。
可改为 `T.reduce_max`（取值）+ `T.tile.compare(EQ) + T.tile.select(idx) + T.reduce_min`（取最小 index，
tie-break 与现行「严格 GT + 最小 index」一致）。K2g 是 fused 任务最重 kernel（case15=1060us，54%），大 V case 收益最大。
中风险（tok_out 需 bitwise 一致，须重跑 stress+gumbel 全套）。

[#3] **split-V 并行化**（参考: DESIGN §12.2「最高杠杆方向」+ antipatterns L92-120「launch core 数」关注项 B + optimization-guide §2.9 Fixed Core）：**适用（原理）但高风险** —
`b_num=ceildiv(B,2)` 使多数 case 核心欠占用（case7 b_num=2 → 0.50% 达成率）。沿 V 切分 + 跨 split 归约
（K1: online-softmax m/s combine + top-256 list merge；K2g: partial-argmax merge；K2m: 无需归约，可直接切）。
**风险**：① 跨核归约需 GM workspace + 归约 kernel（Developer 模式无简单跨核归约原语）；② 改变 TAIL 语义 →
须重新证明 topk `padCount==0` 与 UB→UB 32B 对齐（精度红线 §4）；③ partial buffer + workspace 的 UB 预算。
属结构性重构（非算法重写，online-softmax/topk-merge/gumbel-argmax 语义不变）。留待 #1/#2 后评估。

[#4] **K1 epilogue tau_p 前缀和标量循环向量化**（参考: optimization-guide §2.15 正交轴向量化）：**不适用** —
K1.cc L146-154：tau_p 是**真串行 prefix-scan**（`prefix` 跨迭代累加 + 单调守卫 early-stop），
非正交轴。tilelang 无向量 cumsum/scan 原语，无法安全向量化。保留串行（DESIGN §6.4 已论证正确性）。

[#5] **K2m div → mul(reciprocal)**（参考: antipatterns L241-262「sqrt+div→rsqrt+mul」+ optimization-guide §2.7 指令融合）：**不适用（本轮）** —
K2m `w=exp(z-m)/s`，s 为行标量。虽可预计算 inv_s=1/s 后 mul，但 optimization-guide §2.13 L1129 明确警告
「不要先算 rstd=1/std 再 mul（多一步除法引入浮点误差）」；且 K2m 仅占 mask 任务 8%（case15=74.8us），
收益 < 噪声阈且引入精度风险 → 不做。

[#6] **减少 K1 每块 topk(256,4096) 全排序成本**（参考: optimization-guide §2.7）：**不适用** —
K1.cc L117 `TopK(256,128,4096)` + L124 `TopK(256,16,512)`。tilelang `TopK` 模板恒做完整 Sort
（无 partial-select/quickselect 原语），且需要 256 个**有序**值供 tau_k/tau_p/merge。分层归并已是现有原语下最优。

[#7] **消除 K1 z_ub→zt_ub UB→UB 拷贝（N2 rank-1）**（参考: DESIGN §5.3 N2 L1068-1080）：**暂不适用（低收益）** —
K1 每块 `T.copy(z_ub[0,:BV], zt_ub)`（16KB UB→UB）为满足 topk 的 rank-1 源要求。UB→UB 拷贝快（片内），
且改 z_ub 为 rank-1 会牵动 online-softmax 的 reduce/broadcast 维度（中风险）。收益低，暂缓。

[#8] **关闭 AUTO_CV_COMBINE 消除 L0A/L0B 空分配**（参考: antipatterns L461-484 + optimization-guide §2.10 L511-537 pass_configs 最后手段）：**暂不适用** —
纯 Vector + AUTO_CV_COMBINE 使 kernel 成 MIX_AIC_1_2 并空分配 L0A/L0B（各 64KB，Cube 侧，不占 UB）。
但 R4 用 `alloc_shared` 非 `alloc_var`，未发生误分核，精度已验证。§2.10 明示 pass_configs 调整为「最后手段」，
且关闭可能改变 codegen 分核行为 → 需先确认 L0A/L0B 空分配是否真影响性能（Cube 侧 buffer，理论不占 Vector 资源）。留待最后评估。

[#9] **Double Buffer / T.Pipelined 核内流水**（参考: optimization-guide §2.2 + best-practices/vector_add_pipeline.md）：**不适用** —
§2.2.0 决策表：纯 AIV Vector 算子做 Double Buffer **必须 `AUTO_SYNC=False` + 手写 set_flag/wait_flag**
（vector_add_pipeline 模板用 `alloc_ub`+`T.Scope("V")`+手写 flag = Expert 形态）。这与 R4 的
**Developer 模式**（`alloc_shared` + 全自动同步，DESIGN §7.1「零手写 flag/scope」）根本冲突，等同 Expert 回退。
且 DESIGN §6.3 已论证 K1 循环携带依赖（m_run/s_run/run_v）使 Pipelined 收益有限 + UB 预算紧张。→ 不做。

[#10] **多行 Tile 粒度扩展**（参考: optimization-guide §2.13 L921-1024）：**不适用** —
§2.13 针对 GroupNorm/RMSNorm 类「外层通道循环 + 内层 tile」双循环。R4 是「一行一任务」单行处理
（DESIGN §6.5：topk 语义强制整行 vocab 驻留单核 UB），无正交多行可合并。

[#11] **归约遍数融合（Online Softmax 3→2 pass）**（参考: vector_reduce_pass_fusion.md 全文）：**不适用（已实现）** —
K1 已是 online-softmax 单遍融合（同一块内 reduce_max + exp + reduce_sum，跨块用 m_run/s_run 重标定），
即该文档的 Pass-1 最优形态。K2m/K2g 是 Pass-2（用 m/s/tau 逐元素）。已达 2-pass 下限。

[#12] **正交轴向量化 / int32 转 fp32**（参考: optimization-guide §2.15 L1229-1409）：**部分适用于 [#2]** —
§2.15 的 argmax-with-index（compare+select+min）正是 [#2] K2g 替代 topk(K=1) 的方法论来源。
R4 无其他正交轴串行标量扫描（除已归入 [#1]/[#4] 的 epilogue）。

[#13] **Host 侧预处理内化 / D2H 消除**（参考: optimization-guide §2.12 L714 + §2.14 L1152）：**不适用（P0 已合规）** —
Host wrapper `r4_fused_sampling`（L514-576）静态检查：无 `F.pad`/`.contiguous()`/`.to(dtype)`/`permute`/`cat`/`clone`/in-place 算术；
无 kernel 输出 D2H 后处理（DESIGN §5.6 已审计，valid.cpu() 属 dispatch 决策，非计算）。P0 Host 侧已零违规。

[#14] **特定 dtype 硬件指令适配**（参考: optimization-guide §2.16 L1413）：**不适用** —
R4 仅 fp32 + int32，无 transpose，无非硬件路径 dtype（bf16/int8/int64）。

---

## Part B — [ORDER-PLAN] 实施顺序

依赖分析（optimization-guide 三规则：布局依赖 / 数量依赖 / 配置依赖）：

- [#1] 改 K1 epilogue（tie_m 填充），[#2] 改 K2g 主循环（argmax）——**不同 kernel、不同代码区域，无相互依赖**。
- [#3] split-V 会重构 K1/K2g 的 grid 与归约结构：#1 的 epilogue 将移入 split-V 的归约 kernel，#2 的 argmax 将变为 per-split。
  → **[#3] 必须在 [#1]/[#2] 之后**（否则 #1/#2 的改动被 split 重构覆盖/纠缠，无法独立验证与回滚）。
- UB 预算（数量依赖）：[#1] 加 ~1KB（ones/mask 小 buffer）；[#2] 加 idx[BV]+sel[BV]≈+32KB（K2g）；[#3] 加 partial+workspace。
  → 逐项实施后各自验证 UB 不超限（memory planning 编译期报错即超限）。
- pass_configs（配置依赖）：[#8] 关 AUTO_CV_COMBINE 属「最后手段」，排在所有 kernel 级优化之后。

```
[ORDER-PLAN] 实施顺序：
1. [#1] K1 tie_m 标量循环向量化 — 前置依赖: [无] — 理由: bitwise 安全、全 case 受益、最低风险，先确立 A/B 测量置信度
2. [#2] K2g topk(K=1) → reduce_max+argmax — 前置依赖: [无]（与 #1 独立）— 理由: fused 任务最重 kernel 的最大浪费点，大 V case 高收益
3. [#3] split-V 并行化 — 前置依赖: [#1, #2]（避免重构纠缠）— 理由: 最高杠杆（核心利用率），但高风险，须在 #1/#2 稳定后评估
4. [#8] 关 AUTO_CV_COMBINE — 前置依赖: [#1,#2,#3]（pass_configs 最后手段）— 理由: 仅当确认 L0A/L0B 空分配影响性能时
```

不实施（本轮判定不适用）：[#4] tau_p scan（无 cumsum 原语）、[#5] div→mul（§2.13 反建议 + 收益<噪声）、
[#6] topk(256) 全排序（无 partial-select 原语）、[#7] z_ub 拷贝（低收益中风险）、[#9] Double Buffer（与 Developer 模式冲突）、
[#10] 多行 tile（单行任务）、[#11] 归约融合（已实现）、[#13] Host 内化（P0 已合规）、[#14] dtype 适配（无该 dtype）。

---

## Step 4 — 逐项实施

### 迭代 1 — [#1] K1 epilogue tie_m 标量循环向量化

[ORDER-CHECK] 准备实施: [#1] K1 tie_m 向量化 | 前置依赖: [无 ✅]（ORDER-PLAN 第 1 项）| 结论: [✅]
- 调用域未扩大（同 B/V/dtype/tail/stage，仅替换 epilogue 内 tie_m 填充实现）→ 无需 [KERNEL-REUSE-PRECHECK]/[KERNEL-REUSE-AUDIT]。
- 非 Double Buffer 类 → 无需 [DB-ANALYSIS]。

[IMPL-#1] 已阅读 antipatterns L124-150（Vector Core 内逐元素 for loop：优先 broadcast+整 tile 消除循环）
+ optimization-guide §2.6 L381-400（指令向量化）。关键约束:
- compare scalar src1 走 `compare_scalar`（ascend_tile.py:1881-1893，K2g `t_v[0,0]` 同款 V-b5 已验证）；mode GT/EQ 对 fp32 精确。
- select `VSEL_TENSOR_SCALAR_MODE`：src0=tensor(1.0)、src1=scalar(0.0)（ascend_tile.py:707-796）。
- mask buffer = [K_MAX//8]=[32] uint8（bit-packed，K2g `[BV//8]` 同款）。
- bitwise 等价证明：n_gt=count(run_v>tau_k)、n_eq=count(run_v==tau_k)，标量 `tau_k<run_v[l]`≡向量 `run_v>tau_k`(GT)；
  0/1 计数 ≤256 在 fp32 精确，reduce_sum 树序 vs 标量序对整数和无差异 → n_gt/n_eq bitwise 相同，下游 tieok/vflag 不变。
- UB 增量：tie_mask[32]u8 + tie_one[256]f32 ≈ +1KB（余量 44988B，安全）。

[SELF-CHECK] 3 处 Edit 仅涉及 [#1]（tie_mask/tie_one/tie_tkbc 声明 + tie_one 初始化 + tie_m 两循环→broadcast+compare+select+reduce）。未动 topk/TAIL/主循环/host。

[RESULT-#1] 优化点: K1 tie_m 标量循环向量化 | 精度: **pass**（all 74/0/38/0 exit0, stress 8/8 exit0, L0 pass, bitwise 等价）
| 性能(msprof cann-bench 18case×2task, warm cache): 
  - fused geomean speedup vs torch: 6.7272 → **8.5164 (+26.60%)**；平均加速比 0.169→0.21
  - mask  geomean speedup vs torch: 11.0175 → **15.6221 (+41.79%)**；平均加速比 0.153→0.23
  - 逐 case：36/36 全部改善或持平，**无任何 case 回退 >3%**（worst +0.00%）
  - 大 V case（7/15/17）仅 -2.5~-5%（topk 主导，epilogue 占比小）；中小 V case -16~-48%（epilogue 占比大）
| 对比: [采纳] — 全 case 无回退 + geomean 显著提升。回滚点 history_version/r4_fused_sampling_impl_s3_attempt1.py
| 根因确认: K1.cc 原 2×256 标量 tie_m 循环（每轮 PipeBarrier<PIPE_ALL>）→ 2×(Compare+Select+reduce_sum)，PipeBarrier_ALL 768→5。

### 迭代 2 — [#2] K2g topk(K=1) 全排序 → reduce_max + argmax-index

[ORDER-CHECK] 准备实施: [#2] K2g argmax | 前置依赖: [无 ✅]（与 #1 独立，ORDER-PLAN 第 2 项；#1 已采纳）| 结论: [✅]
- 调用域未扩大（同 B/V/dtype/tail；仅替换 K2g 块内 argmax 机制）→ 无需 KERNEL-REUSE-AUDIT。非 Double Buffer → 无需 DB-ANALYSIS。

[IMPL-#2] 已阅读 optimization-guide §2.15 L1321-1343（argmax 索引模式 compare+select+min）+ ascend_tile.py reduce_max/reduce_min（reduce_ascend.py:362/404）、compare(1837)、select(707)、arith_progression(389)。关键约束:
- K2g.cc L160 实证 `TopK<float>(...,K=1,repeatTimes=128,actual=4096)` = 对 4096 做完整 Sort（Sort32×128+MrgSort 树）仅取 1 个 max → 巨大浪费。
- 替换: reduce_max(z_ub[BV])→blk_max；compare(z_ub==blk_max,EQ)→mask；select(idx?sentinel)→reduce_min→最小 argmax index。
- **narrow-reduce 约束核查**: K2g docstring 曾避 T.reduce_*（担心 codegen_ascend.cc:2145 ≤256B narrow 约束）。但本 reduce 是 **full-width [BV=4096]=16KB**（非 real_shape/切片窄归约），与 K1 `reduce_max(z_ub[1,4096],dim=-1)` 同 normal 路径（K1 已验证编译+精度）→ 不触发约束。
- **bitwise 等价**: max 精确无舍入 → blk_max==topk_dst[0]；tie 时 reduce_min 取最小 index == torch.argmax first == V-b4 验证的 topk(K=1) 行为。全 -inf 块: blk_argmax=0 但 merge `-inf>best_val`=false 不更新（与现行一致）。tau<=m 保证每行至少 1 个非 mask 元素。
- compare_scalar 用 blk_max[0]（循环体内单次发射，与 L508 t_v_scalar 不同名，无 #1 式重定义冲突）。
- **UB 复用**: masked_idx 复用 g_ub（L509 后 dead）、eq_mask 复用 mask（L510 后 dead）；新增 idx[BV]16KB + blk_max[1] + blk_argmax[1]。**必须删除 topk_dst 声明**（dead alloc_shared → codegen "Cannot find pre-allocated address" 失败，K1 docstring L182-184 警示）。
- IDX sentinel = float(BV)=4096.0 > 任何 local index(0..4095)，reduce_min 正确取最小 argmax。

[SELF-CHECK] 3 处 Edit 仅涉及 [#2]（K2g buffer 声明去 topk_dst+加 idx/blk_max/blk_argmax、idx hoist、topk→reduce_max/compare/select/reduce_min）。g_ub/mask 复用为 dead-buffer 复用；未动 K1/K2m/host/TAIL。

[RESULT-#2] 优化点: K2g topk(K=1) 全排序 → reduce_max+argmax | 精度: **pass**（all 74/0/38/0 exit0, stress 8/8 exit0, L0-B tok bitwise ratio=1.0/max=0）
| 性能(msprof 18case×2task): 
  - fused geomean speedup vs torch: iter1 8.5164 → **iter2 10.3463**（vs base 6.7273 = **+53.80%**）；平均加速比 0.21→0.26
  - 大 V case（7/15/16/17/18）fused 再降 21~23%（sort(4096) 消除）；中小 V 降 10~19%
  - mask geomean 15.6221→15.6706（[#2] 不影响 mask，i2 vs i1 全 ±1.8% 噪声内，符合预期）
  - **36/36 case 无回退 >3% vs base**
| 对比: [采纳] — fused 大幅提升、mask 持平、零回退、bitwise 精度。回滚点 history_version/r4_fused_sampling_impl_s3_attempt2.py
| 根因确认: K2g.cc TopK<>=0/Sort<>=0（原每块 full Sort(4096) 仅取 K=1 max），改 reduce_max+reduce_min(argmax)。

### [#4] tau_p 向量化 — 探测后确认不可行（证据）
- probe `/tmp/opencode/cumsum_probe.py`：最小 Ascend 纯 Vector kernel 调 `T.cumsum(xb,yb,dim=0)` → **InternalError（TVM transform pass 阶段失败）**，cumsum 在本 wheel/硬件的纯 Vector + R4_PASS_CONFIGS 下不 lowering。
- 结论：无可用 Ascend 向量 cumsum/scan 原语 → tau_p 的单调守卫 prefix-scan 无法向量化，保留串行（与 Part A [#4] 判定一致，现有实证支撑）。PipeBarrier_ALL 768→5 已由 [#1] 消除 tie_m 部分达成；tau_p 的 256-iter 串行是算法固有（scan 依赖）。

### 迭代 3 — [#3] split-V 并行化：深度可行性分析后判定「设计层增强，交 Stage 1」，Stage 3 内不实施

[ORDER-CHECK] 准备实施: [#3] split-V | 前置依赖: [#1 ✅ 采纳, #2 ✅ 采纳] | 结论: [❌ 不在 Stage 3 实施 — 设计层重构]

**实测复核（任务要求「不盲信」已知瓶颈）**: msprof K1/K2 位置拆分显示，真实主导瓶颈是
① K1 epilogue tie_m 标量循环（每轮 PipeBarrier_ALL）→ [#1] 修复；② K2g topk(K=1) 对 4096 全排序 → [#2] 修复。
二者合计 +53.8%(fused)/+42.2%(mask) geomean，零回退。split-V 假设的「核心利用率」瓶颈经实测为**次要**，
仅集中于极小 B + 超大 V 的 case（case7 b_num=2 达成率 0.67%）。

**split-V 可行性分析（算法层：可行）**:
- online-softmax 跨 split 合并满足结合律: m=max(m_c); s=Σ s_c·exp(m_c−m)。m 精确无舍入。
- top-256 跨 split 合并: global top-256 = top-256(∪ split top-256)，用 topk(256, NS·256) 归并（NS·256 %32==0 ✓）。
- K2g partial-argmax 跨 split 合并: (val,idx) 取 max-val/最小-idx，与现行严格-GT 归并一致。
- 按 **block 边界切分 V**（每 split 处理整数个 BV 块，尾块只落最后 split）→ topk count 仍为 BV 或 TAIL_PAD（均 %32==0），
  **精度红线（§4）可保持**。K2m 逐元素独立，split-V 无需归约（但仅占 mask 8%，低价值）。

**ROI 估计（若实施）**: 9/18 case（B<40 且 n_blk≥2：case3/4/5/6/7/8/11/12/17）受益；case7 K1 ~4.7×、case17 ~3.5×。
geomean 额外 +30~50%（叠加现有 +54%/+42%）。大 V 大 B case（15/16/18，b_num≥32）核已饱和，split-V 仅增归约开销 → 需条件分派避免回退。

**❌ 为何 Stage 3 内不实施（3 条客观理由，非能力问题）**:
1. **设计层架构重构**: 需新增 K1_partial + K1_reduce + K2g_partial + K2g_reduce（4+ kernel）+ GM workspace + 双路条件分派
   （NS=1 保留现行单 kernel 避免小 V/大 B 回退）。与 AGENTS.md「最小必要改动」「禁止推翻重写」冲突。
   任务指引明确：「设计层问题不要在 Stage 3 内强行打补丁，返回标记交 Orchestrator 决定是否回退 Stage 1」。
2. **correctness-sensitive（威胁精确匹配门禁）**: split-V 改变 online-softmax 归约顺序 → s 变 ulp 级 →
   q2=exp(run_v−m)/s 变 ulp → tau_p 前缀和在 p_i 边界处可能跳变 L → tau 变 → 经 K2g mask(z≥tau) 翻转 tok_out（**int32 精确匹配门禁**），
   经 okp(prefix vs p_i) 翻转 valid（**int32 精确匹配门禁**）。这不是性能补丁，是需 Stage-2 级 bitwise 严谨度的正确性敏感改动
   （debug_log.md 记录 Stage 2 为 tau_out garbage/bitwise 投入大量工作）。低概率但真实的 boundary-flip 风险。
3. **无现成纯 Vector 跨核归约示例可复用**（AGENTS.md「从示例入手」）: 调研 examples/ 无 pure-vector split-reduction 范式；
   R4 虽已有多 kernel+GM 传递先例（K1→K2），但 top-256 跨核归约需从零构建。

**判定**: split-V 是**设计层增强机会（非 DESIGN_ERROR — 现行设计正确且已优化 +54%/+42%）**，建议作为 **Stage 1 重设计候选**
交 Orchestrator 决策（含 bitwise 精度策略：归约顺序固定化 / 阈值放宽 / tok 边界处理）。Stage 3 保留 [#1]+[#2] 的干净、已验证、零回退成果。

### [#8] 关闭 AUTO_CV_COMBINE — 分析后判定不实施（无预期收益）
- 生成代码为 KERNEL_TYPE_MIX_AIC_1_2 + 空分配 L0A/L0B（各 64KB）。分析: b_num 块各需 1 AIC（空转）+2 AIV（实算），
  20 AIC 核 → 20 块并发 × 2 AIV = 40 AIV 全部使用。纯 AIV 亦是 (cid,vid) 2 子核/块 → 20 块并发 → 同样 40 AIV。**并发度不变，无占用收益**。
- L0A/L0B 是 Cube 侧 buffer，不占 Vector UB（151KB/196KB 不受影响）。DESIGN §7.3 记为「无害空转」。
- pass_configs 调整为 §2.10「最后手段」，且改变全 kernel codegen 分核行为需全量重验，预测零收益 → 不实施。
