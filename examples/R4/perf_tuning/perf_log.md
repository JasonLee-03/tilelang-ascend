# R4 fused sampling — Stage 3 perf_log.md（跨轮结构化索引）

> 本文件是 Stage 3 各轮迭代的**结构化摘要索引**（agent 契约要求的 `perf_log.md`）。
> 详细论证、表格与证据在每轮自己的日志里：
> - round 1 → `optimization_log.md` + `baseline.json` + `final_report.md`
> - round 2 → `optimization_log_round2.md` + `baseline_round2.json` + `round2_bv_probe/`

算子类型：**纯 Vector (AIV)**（三 kernel `gemm/mma/cube 使用数 = 0`，全部计算在 `if ASCEND_IS_AIV` 块内）
性能口径：**msprof**（cann-bench `kernel_details.csv`）。**从未使用 Python/torch 计时作为性能结论。**
噪声阈：**3%**

---

## Iteration round1-#1 — 2026-10-10T00:43（round 1，已采纳）
- bottleneck_type: compute（Vector 核指令级浪费）
- optimization: K1 epilogue `tie_m` 的 `T.Parallel(K_MAX)` 标量循环 → `broadcast + compare(GT/EQ) + select + reduce_sum`
- baseline_time: fused geomean 6.7273× / mask 11.0175×（vs raw-torch）
- candidate_time: fused 8.5164× / mask 15.6221×
- improvement: fused +26.60% / mask +41.79%；36/36 case 无回退（worst +0.00%）
- precision: pass（all 74/0 + 38/0；stress 8/8；bitwise 等价）
- adopted: **yes**
- rollback_reason: —（回滚点 `history_version/r4_fused_sampling_impl_s3_attempt1.py`）
- next_hint: K2g 每块 `topk(K=1)` 对 4096 元素做完整 Sort，是 fused 任务最重 kernel 的最大浪费点

## Iteration round1-#2 — 2026-10-10T01:03（round 1，已采纳）
- bottleneck_type: compute
- optimization: K2g 每块 `topk(K=1, N=BV)` 全排序 → `reduce_max` + `compare(EQ)` + `select(idx/sentinel)` + `reduce_min`
- baseline_time: fused 8.5164×（round1-#1 后）
- candidate_time: fused **10.3463×** / mask 15.6706×
- improvement: fused +21.9%（vs round-1 起点累计 **+53.80%**）；mask 持平（±1.8% 噪声内，符合预期：K2g 不在 mask 路径）；36/36 无回退
- precision: pass（all 74/0 + 38/0；stress 8/8；L0-B tok bitwise ratio=1.0000 / max=0）
- adopted: **yes**
- rollback_reason: —（回滚点 `history_version/r4_fused_sampling_impl_s3_attempt2.py` = 收敛版 `ac8eba47`）
- next_hint: split-V（核心利用率）是剩余最高杠杆，但属设计层重构 + correctness-sensitive

## Iteration round1-#3/#4/#8 — 2026-10-10T01:21（round 1，探测后不实施）
- bottleneck_type: other（核心利用率 / 原语缺失 / pass_configs）
- optimization: split-V（#3）、`tau_p` 前缀和向量化（#4）、关 AUTO_CV_COMBINE（#8）
- baseline_time / candidate_time: **未测量**（#3 未编写代码；#4 probe 实证 `T.cumsum` 在纯 Vector + R4_PASS_CONFIGS 下 InternalError；#8 分析判定并发度不变）
- improvement: N/A
- precision: N/A（无代码改动）
- adopted: **no**
- rollback_reason: #3 = 设计层架构重构（4+ kernel + GM workspace + 双路分派）且威胁 `tok_out`/`valid` 精确匹配门禁，交 Stage 1；#4 = 无可用向量 cumsum/scan 原语；#8 = 预测零收益且 pass_configs 为「最后手段」
- next_hint: 见 round 2

---

## Iteration round2-#3 — 2026-10-10T15:57（round 2 第 1 项，**BLOCKED**）
- bottleneck_type: other（**不是性能瓶颈问题，而是调优落点不存在**）
- optimization: BV tile size 调优（`optimization-guide.md §2.11 UB 预算优化`，Vector 核 tile size 首选方法）
- baseline_time: **round 1 收敛值** = fused geomean **10.3503×** / mask **15.6911×**（vs raw-torch，18 case × 2 任务 = 36 case）；kernel sha256 `ac8eba47…`；详见 `baseline_round2.json`
- candidate_time: **无**（不存在可测量的 BV 变更 → 未产出任何性能数字）
- improvement: **N/A（未测量，不是 0，也不是「持平」）**
- precision: **pass**（交付版本未改动；四门禁本轮**实际复跑**全绿：all 74/0 + 38/0 exit0；stability exit0 + 4 哨兵 `dropped=0`；stress 8/8 exit0；coverage 25/0/0 exit0）
- adopted: **no — `[ITEM3_BLOCKED]`**
- rollback_reason: **无需回滚：从未修改 kernel**（sha256 before == after == `ac8eba47…`；`test_…py` 亦未改，sha256 `02a44b79…`）
- 阻断根因（证据见 `optimization_log_round2.md` §2）: `BV` 不是 kernel 侧自由参数，而是 cann-bench **契约属性**，由
  `case.attrs` 经 `param_builder.py:203-206` → `evaluator.py:437` → `r4_ops.py:37 (BV=BV)` 按关键字传入；
  `r4_fused_sampling.py:545` 的 `BV=4096` 默认值在 bench 与门禁两条测量路径上都是**死代码**
  （已用真实 harness 代码端到端实证：case 8 得到 `{'K_MAX': 128, 'BV': 2048}`）。
  三条落点判定：(A) 改 kernel 默认值 = 可证明的 no-op；(B) 改 `cases.yaml/csv` = 越界改评测集定义；
  (C) 静默覆盖 attr = 契约违约 + 静默弱化 `D-PARAM-BV` 覆盖。
- 静态/实测筛选结论（§3/§4）:
  - **TAIL 精度红线 54/54 组全部合规**（18 case × BV∈{4096,4608,5120}，全部取自生成源码实证）：
    `padCount ≡ 0`（`TAIL_PAD % 32 == 0` 结构性保证，与 BV 无关）、所有 `copy_ub_to_ub` 字节宽度 %32==0、
    C3 全 ok、`K ≤ actual_num` 全 ok ⟹ **红线不是淘汰因素**
  - **UB 预算才是**：实测三 kernel 账目（上限 196352 B）——
    BV=4096 最差 188,004 B（95.75%，余 8,348 B）；BV=4608 最差 196,196 B（99.92%，**余 156 B**）；
    BV=5120 最差 196,260 B（99.95%，**余 92 B**）；**BV=6144 K2g 编译期硬失败**
    （实测 `LOG(FATAL) Memory allocation failed for: tmp_ub required: 49152, new memory available: 48704`）
  - 全可行域扫描（BV ∈ [1024,8192] 步长 32）：3872…4064 因 **C3** 不可行（V=4096 → `TAIL < K_MAX=256`，host 硬抛）
- 连带影响核实: **`t_hw` 依赖 BV — 是**（`r4_thw_common.py:219/234/310-328`；
  `compute_r4_*.py:95-98` 的闭式常数硬编码 `K_MAX/BV == 1/16`）。改 BV 会使已发布 `t_hw_us` 与全部达成率失真，需重跑 `hap_thw_model/`
- 新发现（既有文档缺陷，非本轮改动）: **DESIGN §4.5 的 K1 UB 模型低估 34%**（预测 123,976 B vs 实测最差 188,004 B；
  漏算 `m2d[1,BV]`、尾块 5 个缓冲、`alloc_shared` 的 32 B 对齐、上限应为 196352 而非 188416），
  且 **§4.5 完全没有 K2g 账目**（实测 `36·BV + BV/8 + 484`，BV=4096 时 148,452 B = 75.6%，是决定 6144 不可行的那个 kernel）
- 新增 JIT 特化: **44**（编译耗时单列，**不计入性能**；K1≈6 s / K2m≈4.7 s / K2g≈5.1 s 每个，合计 ≈4.5 min）；
  其中被交付物使用的 = **0**；2 个是 BV=4096/2048 合法配置，42 个是 BV=4608/5120 探针专用死条目
- consecutive_no_improvement: **+1**（本轮未产生提升；Stage 3 中止条件由 Orchestrator 判定）
- design_error: **no**（现行设计正确、四门禁全绿、round 1 已 +53.8%/+42.2%；BV 不可上移是 UB 物理预算 + 契约定义的联合结果）
- next_hint: ① 若认为 BV 应可由实现自选，须先在**评测集层面**改 `proto.yaml`/`cases.yaml` 契约并重推 `t_hw`（超出 Stage 3 权限）；
  ② 剩余性能杠杆不在 BV —— K2g 的 int32 `And` int16-native「halving」workaround 导致 `2*BV` 过量分配
  （`r4_fused_sampling.py:409-410/437-440/472`）是**常量因子 2× 的真实浪费**，与 BV 无关，改 BV 不能消除；
  ③ split-V 仍是设计层最高杠杆（round 1 已归档，交 Stage 1）；
  ④ 建议修订 DESIGN §4.5 的 UB 模型（补 `m2d`/尾块/32 B 对齐/196352 上限 + 新增 K2g 表）

## Iteration round2-#4 — 2026-10-10T17:35（round 2 第 2 项，**采纳为 enabling change, perf-neutral**）
- bottleneck_type: **transfer**（冗余 UB→UB 数据搬运）+ **sync**（连带 1 次 `PipeBarrier<PIPE_V>`/满块）；次要价值 = **UB 容量**
- optimization: 消除 K1 主块路径仅为满足 topk rank-1 源要求而存在的 `T.copy(z_ub[0,:BV], zt_ub)`，
  并消除 `zt_ub = T.alloc_shared([BV])`。做法：`z_ub` 由 rank-2 `[1,BV]` 改 **rank-1 `[BV]`**，
  使其同时充当 GM→UB 目的地与 topk 源（= 把同文件 K2g `r4_gumbel_sample:436/468/522` 的既有已验证形态搬到 K1）。
  5 处语义变更：`z_ub` rank-2→rank-1、删 `zt_ub` 分配、`reduce_max` 的 out 改 BufferRegion `m_blk[0,0:1]`、
  删 UB→UB 拷贝、`topk` 源 `zt_ub`→`z_ub`。`e_ub`/`m2d` **保持 rank-2**（`binary_op`/`unary_op` 只比 `prod(extent)`，
  rank-agnostic；且 `m2d` 改 rank-1 不可行——rank-1 broadcast 要求 rank-1 `[1]` 源，得到它需 1 元素 UB→UB 拷贝，4 B<32 B 硬禁）
- baseline_time: **同会话重测的 `ac8eba47` 基线**（非引用旧报告）—— fused geomean speedup 基准 elapsed；
  端到端 score fused 59.199724 / mask 58.416610；K1 隔离基线见 `round2_item4/k1_isolated_ab.txt`
- candidate_time: 端到端 score fused 59.225076 / mask 58.438490
- improvement: 端到端 geomean fused **+0.352%** / mask **+0.360%**；K1 隔离 fused **+0.319%** / mask **+0.523%**；
  最差单 case 回退 fused +0.285% / mask +0.548%；**0 个 case 回退 >3%**；36 个 K1 测量中 30 快 / 1 平 / 5 慢。
  **⟹ 判定为噪声内，不声称提速**：两个对照组同量级移动 —— fused 的 K2g（代码未动）geomean **+0.348%**（比 K1 还大，
  符号分布 14/0/4 与 K1 的 14/1/3 同形）；mask 的 `n_blk_full==0` K1（生成码逐字节相同、UB Δ=0、JIT 缓存键未变）
  geomean **+0.531%**，与受影响集 +0.521% 同量级，单点甚至 −0.98%。会话漂移标定（旧报告 vs 本轮同 kernel 新基线）
  = fused +0.136% / mask +0.378% / 单 case 最大 +1.71%
- precision: **pass** —— 四门禁全绿（all 74/0 + 38/0 exit0；stability 12/0 exit0 + 4 哨兵 TAIL=313/257/511/261 全 `dropped=0`；
  stress 8/8 UB-poison exit0；coverage 25/0/0 exit0）**且 bitwise 完全一致**：22 shape × 2 派发路径 × 5 输出 = **220/220 张量逐位相同**
  （sha256 + 原始字节 `==` + shape/dtype 三重校验，非容差判定），harness 侧 36/36 case 全 `mismatch=0, max_abs_diff=0`
- adopted: **yes —— 走判据 (b)「性能中性但为后续使能」，标注 "enabling change, perf-neutral"，由 Orchestrator 最终裁定是否保留**
  （(a) 不成立：+0.35% ≪ 3%。(b) 三条件全满足：geomean 在 ±3% 内 ✓ / bitwise 完全一致 ✓ / UB 实测余量确实增加 ✓）
- UB 实测（非模型，解析生成源 `GetWithOffset` 布局 = planner 的 "shared.ub memory usage"）:
  最差 case 6 **188004 → 171620 B**（95.75% → 87.40%），余量 **8348 → 24732 B（+16384 B，×2.96）**，与任务书 §5 预测精确吻合。
  16 个 `n_blk_full>0` case 的 Δ 是**单一常量 −BV*4**（14×−16384 / 2×−8192）；
  `n_blk_full==0` 的 case 1/14 **Δ=0**（构造性 no-op）；**K2m/K2g 四个配置全部 Δ=0**（无跨 kernel 耦合）
- 尾块红线: **L219-249 逐字节未改** —— marker 区间 sha256 `37f19e3e…1618c8`（before == after）、
  严格 31 行区间 sha256 `098e4bc1…deb608`（before == after），两区间 `diff` 均无输出。
  **追加生成码级证明**：全 18 case 抽取尾块相关指令行（排除布局行）逐 case diff → **全部 IDENTICAL**；
  尾块缓冲尺寸逐 case 未变（TAIL_PAD 320/2048/512/256/832 全保留），仅地址偏移因腾出 `zt_ub` 而下移
- narrow-reduce 核实（任务书 §4 关键点 3）: **rank-1 `[BV]` 全宽 reduce 仍走 normal 路径**。三重证据 ——
  ① 源码：narrow 分支唯一入口 `if (physical_row > 0)`（`codegen_ascend.cc:2098`），而 `physical_row` 仅在
  `row_const != n_const` 时才被前端 append（`reduce_ascend.py:352`）；rank-1 `[BV]` 两者同为 BV → 不 append → normal。
  ② 生成码：before/after 全部 36 个 `.cc` 中 `grep -c narrow` **均为 0**。
  ③ 逐行比对：`reduce_max<float, 1, 4096, -1>(m_blk[0], z_ub[0], tmp_ub[0], true)` **改前后逐字节相同**
  （模板参数与 out 表达式都没变；`reduce_sum` 同样未变）。
  **⟹ 同时关闭了 DESIGN §5.3 N2 自陈的缺口**（"满块全宽 region 是否回退 normal 路径未验证"）：
  N2 当年放弃本优化的判断在其目标场景（**列切片窄 region**，`valid<BV` → 撞 ≤256 B）下正确，但外推到满块全宽属**过度保守**
- rollback_reason: **不适用（未回滚）**。回滚点已备妥：`history_version/r4_fused_sampling_impl_s3r2_item4_attempt0.py` = `ac8eba47…`（`cp` 建立）
- kernel sha256: `ac8eba47f77e11342eabb9c4e30a44531b3e655b481b0dc28fd5a66a2738fc5b` → **`19e51e8d8ee126b920cf48c915d863a9be3a349e2c68cf728d35d482b48bcc57`**；
  submission 副本 `cann_bench_submission/cann_bench/_r4_kernels.py` **已同步**（`diff -q` 逐字节相同，wheel 内解包核对亦为 `19e51e8d…`）；
  `test_r4_fused_sampling.py` **未改**（`02a44b79…`，无断言弱化）；`proto.yaml` / `cases.yaml|csv` **未改**
- 新增 JIT 特化（编译耗时单列，**不计入性能**）: **16 个 K1**（= 18 case 中 `n_blk_full>0` 的那 16 个）+ **0 个 K2m/K2g**；
  case 1/14 **命中旧缓存（0 新增，0.1 s vs 其余 5.7–6.1 s）** ⟹ IR 与缓存键未变，是彻底 no-op 的独立证据。
  单特化编译 **5.7–6.1 s**，16 个一次性合计 ≈ **94 s**。本轮全部活动共新增 40 个缓存目录
  （16 交付物使用 + 24 个 `bitwise_capture` 探针专用死条目）。**candidate eval 期间新增 0 条 ⟹ 编译耗时未污染 msprof 数字**
- consecutive_no_improvement: **+1**（本轮未产生可判定的性能提升；(b) 类改动的价值在 UB 使能而非提速。Stage 3 中止条件由 Orchestrator 判定）
- design_error: **no**（任务书要求「至少尝试 3 种优化方案后」才可标注，本轮方案一次即成功；
  且本项证明的是**设计文档过度保守而非设计错误**）
- next_hint: ① 按批准序下一项是 **#1a（split-V）** —— round 1 已判定其属设计层架构重构（4+ kernel + GM workspace + 双路分派）
  且威胁 `tok_out`/`valid` 精确匹配门禁，建议交 Stage 1；若仍在 Stage 3 内做，**#4 腾出的 16384 B 对其部分和缓冲有用**。
  注：反模式清单「关注项 B：任务数低于物理 AI Core 数」在 K1 上真实命中（case 2 的 B=1 → `b_num=1`，仅用 1 AIC + 2 AIV，
  而 910b2 有 20 AIC / 40 AIV）⟹ **小 B 场景的剩余杠杆在 V 轴并行化，不在核内微优化**。
  ② **本项使能了一条改前物理不可行的优化**：反模式「纯 AIV memory bound 未做流水/双 buffer」的补救需给 `z_ub` 加双 buffer
  （+16384 B）—— 改前最差 case 余量仅 8348 B → 204388 B **溢出 8036 B，不可能**；改后余量 24732 B → 188004 B ≤ 196352 B
  **静态可行** ⟹ 「`z_ub` 双 buffer / `T.Pipelined(num_stages=2)`」现在是可测候选了（三个 BV 缓冲全双 buffer 仍溢出 24420 B，不可行；
  且加性估算只作参考，须实测 memory planning 结果，因 planner 会做 liveness 复用）。
  ③ **本轮有意未做的后续候选**：`T.tile.sub(e_ub, z_ub, m_run[0,0])` 的 scalar-src1 形态（`ascend_tile.py:1106-1127`，
  走 `AscendC::Adds(dst,src0,-src1)`；同文件 K2g:534 已有 scalar src1 的 NPU 验证先例）可**再消除 `m2d` 的 16384 B + 1 次 broadcast**，
  把 K1 最差余量从 24732 B 提到约 **41116 B**。IEEE-754 下 `a-b` 与 `a+(-b)` 逐位相同，但这是**推理非本硬件实测**，
  且会改变实际算术指令、与本轮「纯搬运重构 ⟹ bitwise 一致」的判据冲突，故留作独立一轮。
  ④ 既存 ruff `F841 AT1`（L442，round 1 [#2] 遗留，改前备份同样报）建议在 #5（同函数）或收尾一并清理。
  ⑤ 建议修订 **DESIGN §5.3 N2**（补记：全宽 plain Buffer 走 normal 路径，`zt_ub` 可消除），与 round 2 §5.1/§5.2 的 §4.5 UB 模型缺陷一并修。
- skills_consulted: `tilelang-perf-optimization`（`SKILL.md` + `references/performance-antipatterns.md` 全 11 节逐节对照，
  结果见 `optimization_log_round2.md` §14.3：1 项直接改善、1 项命中但有意保留并给出量化使能结论、1 项不适用/已合规、
  6 项不命中）；`tilelang-api-best-practices` 未直接调用 —— 本项所需的 API 形态全部由**一手源码**核实
  （`tilelang/language/reduce_ascend.py`、`ascend_tile.py`、`src/target/codegen_ascend.cc`、`src/op/ascend.cc`、
  `tl_templates/ascend/common.h`）+ **同文件 K2g 的 NPU 已验证先例**，符合 AGENTS.md 核心原则 1/2 的查证顺序
- 详细论证与全部证据: `perf_tuning/optimization_log_round2.md` §13–§23；原始数据 `perf_tuning/round2_item4/`
  （`ub_ab_table.md`、`k1_isolated_ab.txt`、`perf_cmp_36case.{txt,json}`、`bitwise_verdict.txt`、
  `bitwise_manifest_{before,after}.json`、`tail_sha256_{before,after}.txt`、`emitted_cc_case15/{before,after}.cc`、
  `gate_*.log`、`gate_exits.txt`，以及 8 个可复现脚本）

## Iteration round2-#2 — 2026-10-10T19:10（round 2 第 3 项，**`[ITEM2_NO_GO]` — 离线研究判定，未写 kernel 代码**）

- bottleneck_type: **compute**（K1 主循环的 `Sort` 归并网络：**48× MrgSort + 144 Sort32 repeats / 块**，
  占 K1 的 **26.4%–78.8%**）—— 瓶颈定位**成立**，但**该瓶颈无法用块级阈值剪枝削减**
- optimization: **无**。#2（`block_max < tau_k_run` ⟹ 跳过该块 topk+merge）在**强制前置门**（离线可行性研究）
  阶段即判定 NO-GO，**未写任何 kernel 代码**
- baseline_time: **不适用**（无候选 ⟹ 无 A/B；**未重跑 36-case eval**）。交付件仍为 `19e51e8d`，
  本轮在**未改动**的它上面复验四门禁全绿
- candidate_time: **不存在**
- improvement: **0.000000%（实测上限）**；解析期望口径上限 **0.002212%**（门槛 3%，差约 **1356 倍**）
- precision: **pass**（对**未改动**交付件的复验，非候选版本精度）。四门禁 exit 全 **0**：
  `all` = 74 PRECISION_PASS/0 FAIL + 38 BOUNDARY_PASS/0 WARN；`stability` = 8×20 reps 全 PASS +
  4 tail-adversary TAIL=313/257/511/261 全 `dropped=0`；`stress` = **8/8** UB-poison PASS（reps=200）；
  `coverage` = **25 PASS / 0 MISS / 0 N/A**
- adopted: **no**（**无改动可采纳**；不是"采纳后回滚"，而是"实施前即判定不可行"）
- rollback_reason: **不适用**（未改动 ⟹ 无需回滚）。交付件 sha256 **起点 == 终点 == `19e51e8d…`**
  （`round2_item2/gate_sha256_{before,after}.txt`，`diff` 为空）；submission 副本仍与交付件 `cmp` IDENTICAL；
  **未创建** `history_version/…_item2_attempt0.py`（改前即 NO-GO，`cp` 只会产生与 `19e51e8d` 逐字节相同的冗余副本）
- **NO-GO 依据（三重独立证据）**：
  ① **剪枝率实测 = 0**：18 个 bench case 共 **5478** 个主块，可剪 **0**（harness 同种子 uniform 输入，
  最"险"的 case 15 `min_margin = +0.0016` 仍为正）；38 个门禁配置共 **2661** 块，可剪 **0**。
  ② **解析上界与种子无关**：`tau_k_run` 的 CDF 值 `U ~ Beta(n−K+1, K)` ⟹ `P(整块可剪) = E[U^BV] = Π(a+i)/(n+1+i)`；
  全集最大期望剪枝率 **1.122e-4**（N=37 的 case 7/15/17），`P ≈ exp(−K_MAX/b)`，需 `V ≳ 10^6` 才可观。
  **根因是数据分布固有性质**：i.i.d. 下 `block_max` 是极值序统计量（贴近上界），`tau_k_run` 只是第 256 大值（高分位非极值）。
  ③ **实施将净回退**：0% 命中时谓词仍要白付 `PipeBarrier<PIPE_ALL>` + 2×`GetValue` + 分支，
  msprof 实测 **+0.100%~+0.948%**（均值 **+0.185%**）
- **第 2 节第 5 点（最大技术风险）已排除**：块内条件分支在 Ascend 上**可实现且是真实运行时分支** ——
  `round2_item2/emitted_cc/K1_case15_pruned.cc:127` 生成 `if (tau_run.GetValue(0) <= m_blk.GetValue(0)) { tl::ascend::TopK<…>(…); … }`，
  两个 `TopK`（=两次 `Sort`）完整位于分支体内 ⟹ **确被跳过，非 mask 化**；静态循环边界 `for (b=0; b<37; ++b)` 未变。
  API 依据全部来自**交付 kernel 内已通过四门禁 + cann-bench 36/36 的先例**（`:300-303` epilogue、`:547-550` K2g）
  + 生成码实证（`before.cc:190/199/237`），**未凭记忆猜 API**
- **反事实上限（本报告最有信息量的副产品）**：若假想"完美预言剪枝"100% 命中，6 个已探测 case 的
  36-case geomean 上限达 **98.036%**（fused 20.0–46.0%、mask 24.9–72.4%）⟹ **Sort 确实是 K1 主体、K1 确实占任务 52%–95%**，
  "减少 Sort 工作量"这个方向极有价值，**但块级阈值剪枝不是能在 i.i.d. logits 上达成它的手段**
- **⚠️ 附带发现 1（门禁失明）**：`STAB_CASES` 最大 `n_blk_full` = **2**、`ADV_CASES` = **1**、
  L2 特殊值用例（NaN/-inf/dbound）全是 `B=4,V=4096,BV=4096` ⟹ `n_blk_full=1`。
  **即使实施剪枝，四门禁也无法检测任何剪枝逻辑错误**（谓词写对/写反/写错，输出完全一致）。
  任务书 §5.4 寄望"`--level stress` UB 投毒是最直接的检测器"**不成立** —— 剪枝不触发 ⟹ 不产生新的残留 UB 状态。
  ⟹ 实施 #2 会引入**不可验证的死代码**，这是独立于收益的第二个 no-go 理由
- **⚠️ 附带发现 2（NaN 语义分叉）**：为避开 `T.Else` 的自然写法 `keep = (block_max >= tau)`，
  在 `block_max` 为 NaN 时给出"剪掉"，而规范式 `prune = (block_max < tau)` 给出"保留" —— **不等价**。
  唯一 NaN 门禁用例 `n_blk_full=1` ⟹ 对该差异完全不敏感。若将来实施必须用 `T.If/T.Else` 双分支
- **⚠️ 附带发现 3（#4 的预算 #2 用不到）**：剪枝只需 `tau_run` 一个 `[1,1]` fp32（UB 约 **32 B**），
  `block_max` 复用已有 `reduce_max` 结果 ⟹ **#4 腾出的 16384 B 对 #2 闲置**。
  不构成 #4 的回滚理由（#4 按 (b) 类采纳，价值独立），但建议 Orchestrator 重新评估该预算归属（如 #1a 部分和缓冲）
- **⚠️ 附带发现 4（方法论）**：模拟器自检**抓出一个会导致相反结论的 bug** —— 初版误把
  `np.partition(...)[-K_MAX:]` 的**无序**切片末元素当第 K_MAX 大值，在 i.i.d. 输入上**虚报 1/8 块可剪枝**（会误判 GO）。
  修正为保持 `run_v` 降序 + 8 项（4 类构造）先验自检（降序 7/8 命中 ✓、升序 0 ✓、iid-uniform ×3 = 0 ✓、iid-gauss ×3 = 0 ✓）
  + 逐块 `tau_k_run` 与真实第 K_MAX 大值交叉校验（`verify=True`，覆盖 5478+2661+自检全部块）后才得出正确结论。
  ⟹ **任何"剪枝率"类离线研究都必须自带已知答案的先验用例**
- 精度红线自查（虽无实施，仍逐条核对）：红线 1 **未引入**非 32B 倍数 UB→UB 拷贝（探针只加 `T.tile.fill`/标量赋值的 `[1,1]`）；
  红线 2 **不受威胁** —— 剪枝是**控制流**不改张量长度参数，实测生成码 `actualCount` 仍为编译期常量 **4096/512**
  （case 8：**2048/256**），`padCount` 全为 **0**；红线 3 剪枝条件取**严格 `<`**（生成码 keep 侧 `tau_run <= m_blk`，
  对偶即 `block_max < tau` ⟹ `block_max == tau_k_run` 的块**被保留**，`n_eq`/`tau_p`/`L` 语义不变）；
  尾块红线 **未触碰**（交付件全文件 sha256 前后一致，强于区段级对比）
- consecutive_no_improvement: **+1**（本轮未产生性能提升；与 #4 的 +1 累计为 **2**。Stage 3 中止条件由 Orchestrator 判定）
- design_error: **no**（任务书要求「至少尝试 3 种优化方案后」才可标注；本项在**离线研究阶段**即判定不可行，
  未进入实施、未尝试 3 种方案。根因是**数据分布与机制不匹配**，非 tiling / 内存层级缺陷；
  DESIGN.md 从未主张剪枝可行，`#2` 本就是 round 2 的**候选**项，本研究正是它应有的前置筛选）
- next_hint: ① 按批准序下一项是 **#5（K2g And halving）**。提示：`r4_fused_sampling.py:443` 的 `BV2 = 2*BV`
  过分配正是该缺陷的现行 workaround，消除它可望显著降低 K2g 的 UB 占用（`k2g_4096 = 148452` B）；可顺带清理
  既存 ruff `F841 AT1`（`:442`）。**但注意**：K2g 仅占 fused 任务 **23.5%–48.2%**（= 1 − K1_share），
  且 #5 的收益是 **UB 容量**而非时间 ⟹ **建议先做与本轮同规格的离线可行性研究再决定是否实施**。
  ② 若目标是真正削减 K1 的 Sort 时间（本研究已证其占 K1 55%–79%），报告 §7.3 列了 3 个替代方向
  （**partial-select 代替全排序** / **两级块粒度粗筛**（内部粒度，不碰契约 BV）/ **反模式「任务数低于物理核数」即 #1a**），
  **均未实施、未测**，需各自独立立项。③ 反模式清单「关注项 B」在 K1 上真实命中（case 7 `b_num=2`，仅用 2/20 核）
  ⟹ 小 B 场景的剩余杠杆在 **V 轴并行化（#1a）**，不在核内微优化
- 验证缺口（如实）：(b) 只探测 **6/18** case（其余标 `n/p`，**未外推**；因剪枝率恒 0 ⟹ 不影响判定）；
  探针**未做输出正确性比对**（未实施，无需）；msprof 探针**未复现 harness 的 `CannBenchCacheClean`**
  （绝对 µs 可能略乐观，但三变体条件相同 ⟹ **比值不受影响**，已在脚本 docstring 声明）；
  `kernel_details.csv` 偶尔多于 5 行（case 7/15/16 的 real/pruned 出现 6–7 行，取中位数故稳健，**成因未定位**）；
  `K1_share_of_task` 取自上一会话 `round2_item4/k1split_*_cand.json`（用同会话内比值，且只进入反事实上限，**不进入判定链**）；
  解析模型假设 logits **连续 i.i.d.**（bench 满足；若生产 logits 有空间有序性则剪枝率会上升，**契约内不存在此类输入**）；
  **未测 `T.Else` 双分支形态**能否在 Ascend 正确 codegen；`.orchestrator_state.json` 全程未读写
- skills_consulted: `tilelang-perf-optimization`（性能数据采集 / 瓶颈诊断 / 反模式对照的方法论口径）；
  `tilelang-api-best-practices` 未直接调用 —— 本项所需 API 形态全部由**交付 kernel 内的 NPU 已验证先例** +
  **一手源码**核实（`src/tl_templates/ascend/common.h:1685-1898`、`tilelang/language/` 既有用法、
  `round2_item4/emitted_cc_case15/before.cc`），符合 AGENTS.md 核心原则 1/2 的查证顺序
- 详细论证与全部证据: **`perf_tuning/round2_item2/feasibility_study.md`**（11 节 + 复现命令）；
  编排标记 `[ORDER-CHECK]` / `[IMPL-#2]` / `[SELF-CHECK]` / `[RESULT-#2]` 见 `optimization_log_round2.md` §24–§29；
  原始数据与可复现脚本见 `perf_tuning/round2_item2/`（11 个脚本 + `prune_rate_result.json`、`gate_blindness.json`、
  `analytic_bound.json`、`probe_perf.json`、`ceiling.json`、`insn_census.txt`、`emitted_cc/*.cc`、`gate_*.log`、`gate_exits.txt`）

## Iteration round3-VERIFY — 2026-10-11T15:55（round 3，**rev3 split-V 的首次实测验证；kernel 未改动**）
- **本轮性质**：验证轮，非优化轮。rev3 split-V 架构在 Stage 2 已实现并精度通过，但**此前一个性能数字都没测过** ——
  DESIGN §12.4 的 1.15×–1.29×（fused）/ 1.11×–1.20×（mask）/ case7 3.5×–4.2× **全部是模型估计**。
  本轮任务 = 实测验证 + 关闭 5 个悬空项（U2/V-S3、U3/V-S8、U4/V-S7、§11.4 步骤 11、GATE 1–4）。
  **未寻找新优化点，未实施任何代码改动**：交付 kernel sha256 前后均为 `a764e117…`
  （`gate_sha256_before.txt` == `gate_sha256_after.txt`，`sha256_stable_before_after_gates=YES`）
- bottleneck_type: **compute**（Sort/top-k 串行链）+ **核心饥饿**（18 case 中 14 个 `b_num < 20`，
  最差 case7 仅用 2/20 cube 组、达成率 0.67%）⟹ rev3 用 V 轴 split 把 Sort 侧并行到 `b_num*S` 个 block
- optimization: （本轮无新增）验证对象 = Stage 2 交付的 rev3 split-V：`K1A`(grid=`b_num*S`) → `K1B`(grid=`b_num`)
  → `K2gA`(grid=`b_num*S`) → `K2gB`(grid=`b_num`)；mask 任务 = `K1A → K1B → K2m`（K2m 不切，约束 4）。
  现行 3 个 kernel 逐字节冻结保留（本轮用 `verify_frozen.py` 复核：L91-364 / L367-412 / L415-555
  三段 sha256 `d3909063…` / `f787d1ad…` / `ca90de8d…` 在 base 与 cand 间**完全一致**）。
  分派由 host 侧 `_r4_split_plan()` 决定，`S_cap = CUBE_CORE_NUM // b_num`（运行时探测 = **20**）
- baseline_time: **同会话三趟 A/B**（不引用任何旧报告）—— pass 序 `base1 → cand → base2`，
  base2 与 base1 是**同一份代码**故为纯漂移对照。6/6 eval `EXIT_CODE=0`、6/6 报告 18/18 pass 0 fail：
  base1 fused `elapsed` 见 `cann_performance_eval_20261011_150336.json`（score **59.211626**）、
  base1 mask `_150535.json`（score **58.422600**）；base2 fused `_151140.json`（**59.236929**）、
  base2 mask `_151335.json`（**58.428332**）。口径 = msprof `kernel_details.csv`，warmup=3 repeat=5，
  每 step 求和后取中位数。**全程无 Python/torch 计时**；本轮复核该口径恒等式：`ksplit_fine.py` 的
  per-case `total_us` 与报告 `elapsed_us` **精确相等**（cand fused case7 `42.440+97.100+36.640+2.840=179.060`）
- candidate_time: cand fused `_150737.json`（score **60.762869**）、cand mask `_150934.json`（**59.249786**）
- improvement:
  - **36-case 合并 geomean = `1.2309×`**；fused 18 = **`1.2796×`**（score **+2.620%**）；
    mask 18 = **`1.1840×`**（score **+1.416%**）
  - **6 个 split case（全部提速，0 回退）**：fused case3 **1.744×** / case4 **2.045×** / case7 **`4.068×`** /
    case8 **2.119×** / case12 **1.488×** / case17 **1.682×**（geomean **2.0624×**）；
    mask case3 **1.386×** / case4 **1.632×** / case7 **2.668×** / case8 **1.582×** / case12 **1.402×** /
    case17 **1.471×**（geomean **1.6435×**）。elapsed 降幅 fused −32.80%…−75.42%、mask −27.88%…−62.51%
  - **12 个非 split case（代码逐字节相同 ⟹ 理论必须精确 1.000×）**：实测 fused **1.0079×** / mask **1.0050×**，
    最差单 case **+0.433%**（mask case9）、最好 **−2.606%**（fused case1）；**24/24 落在 ±3% 内**。
    **判定：无意外耦合** —— 同代码对照 base2/base1 在**同一个 case1 上就移动了 −2.436%**，
    最大"收益"被漂移完全解释；三向 pairwise geomean（fused cand/base1 −0.786%、cand/base2 −0.329%、
    对照 base2/base1 −0.459%）同量级。**结构性证明**（强于统计）：① base 臂 prewarm 仅 **2/72** 冷编译，
    且正是 cand 臂走 split 因而不编译 `K1` 的 case12/17 ⟹ 其余全部命中**同一缓存目录**、加载**同一份二进制**；
    ② `ksplit_fine.py` 显示 12 个非 split case 在**两臂都是 2 次 launch + `Block Num=[b_num,b_num]`**，
    6 个 CSV 全部 `anomaly_rows=0/144`
  - **漂移标定**：geomean(base2/base1) = fused **−0.324%** / mask **−0.036%** / 36-case **−0.180%**；
    单 case 最大 +1.099%、最小 −2.436% ⟹ **本轮实测漂移与 round 2 #4 的 0.3%–1.7% 同源同量级**
- **模型验证（本轮核心交付）**：§12.4 的预测**成立**。fused geomean 实测 1.2796× 落在预测带
  **[1.151, 1.292]** 内；mask 1.1840× 落在 **[1.107, 1.196]** 内；case7 fused **4.068×** 落在
  **[3.52, 4.15]** 内；case7 mask **2.668×** 比 `a_launch=0` 上界 2.66 **高 0.3%**（贴边，非显著超出）。
  两者都落在带的**上半区** ⟹ 与「真实 `a_launch` 接近 0 而非假设的 3.0 us」自洽。
  **模型错在哪（有证据）**：① `F(256)=21.63 us/行` **不是** `b_num`-无关常数 —— case4 的 `K1B` 实测
  **18.94 us < F 本身**，而 case3（V/K/BV/S/NBS 全同，仅 `b_num` 4 vs 2）的 `K1B` = **32.94 us**；
  同样 14 us 的 `b_num` 差在 **base 臂也存在**（`K1` 65.62 vs 51.28）⟹ 是 kernel 既有属性，非 rev3 引入。
  这解释了 case4 的 +34.6%（fused）/+24.6%（mask）最大偏差，也是 E3 拟合出 8/24 负值的同一根因。
  ② case17 是唯一模型**偏乐观**的（−2.2%/−4.5%）：`b_num=8 ⟹ S_cap=20//8=2`，每 split 仍串行走
  `NBS=19` 块（`K1A=183.84 ≈ 19×9.68`）⟹ 定理 P2 的单波约束真实生效，模型抓对了。
  ③ `t(4096,256)=8.902 us/块` **被确认**：`K1A` 每块 case7 `42.44/4=10.61`、case12 `40.70/4=10.18`、
  case17 `183.84/19=9.68`，一致略高于 8.902（K1A 还含 GM 读与部分归并），无异常
- **U2 / V-S3 关闭 —— `a_launch` 实测**：因 cann-bench 计分口径是 **Duration 之和**（间隙不计入），
  guard 该建模的是**launch 的器件侧时长地板**，不是 host 发射延迟。三估计量：
  **E1（无模型地板，主用）** rev3 的 `K2gB`（UB **仅 132 B**，全部有效工作 = 每行折叠 ≤10 个偏结果）
  实测 Duration：case4 **1.540**（min）/ case3 2.080 / case8 2.580 / case7 2.840（median）/ case12 2.980 /
  case17 **3.100**（max）⟹ **`a_launch < 1.540 us`**（K2gB 仍有非零折叠工作，真地板更低），
  保守规划值 **≤ 3.100 us = 出厂默认 8.0 us 的 38.8%**。
  **E2（最小二乘）** `K2gB = a0 + a1·b_num + a2·S` ⟹ `a0 = **−0.050 us**`、`a1=+0.326`、`a2=+0.231`，
  最大残差 0.128 us ⟹ 截距 ≈ 0，器件时长几乎全由 `b_num`/`S` 工作项解释，**无可测的常数 launch 地板**。
  **E3（§12.4 模型残差）如实弃用**：24 个残差 **+0.206…+14.702 us 且 8 个为负**（case4 `K1B` 推出 −13.79 us），
  自相矛盾，根因同上 ①；仅作为「模型的 F/p 不可分离」的证据保留，**不作为 `a_launch` 取值**。
  与 §12.4 的 `a_launch ∈ [0,8] us` 全区间敏感性交叉校验：实测 `[0, 3.1] us` **完全落在**该区间内，
  其零回退证明被实测确认 —— **36 个测量中 0 个回退 >3%**
- **guard 放宽（只分析，未改；需 Orchestrator 批准）**：`_SPLIT_A_LAUNCH_US` **只在 cost guard 内、
  在 S/NBS/S_TAIL 推导之后**被读取 ⟹ 放宽只翻 `use_split` 布尔值，**不改任何几何**。扫描
  （复现式经自检：在出厂默认下与交付 `_r4_split_plan` **18/18 精确一致**）：
  8.0→{3,4,7,8,12,17}(6)；4.0→同；**3.20→+case11(7)**；**≤2.74→+case6,+case11(8) 并饱和**。
  其余 10 个非 split case 是**结构性**封锁（`S_cap=20//b_num=0` 的 case9/15/16/18、`=1` 的 case14；
  `n_blk_total=1` 的 case1/2/10；`n_blk_full<2` 的 case5/13）⟹ **任何放宽都到不了它们**。
  **收益为实测而非模型**：`probe_guard_relax.py` 仅在**进程内**猴补丁该常数
  （交付文件 sha256 前后一致，`kernel_file_unmodified=true`），同口径 msprof 双臂对比 ——
  case6 fused **1.4442×**(−30.76%) / mask **1.3089×**(−23.60%)；case11 fused **1.2479×**(−19.86%) /
  mask **1.1537×**(−13.33%)；**4/4 全部 5 个输出 bitwise 相同**；Block Num `[12,4,12,4]`/`[12,4,4]`/
  `[8,4,8,4]`/`[8,4,4]`，split-并行 12 与 8 均 ≤ 20 ⟹ 定理 P2 仍成立。
  折算到已发布 geomean：fused 1.2796→**1.3222×**(+3.33%)、mask 1.1840→**1.2114×**(+2.32%)、
  36-case 1.2309→**1.2656×**(**+2.82%**)。低风险佐证：这两个几何（`S=3/NBS=1/S_TAIL=2`、
  `S=2/NBS=1/S_TAIL=-1`）**已被 Stage-2 静态门禁编译并断言干净**（`static_gate_all.log` 的 `--- cb6:`
  K1A UB 106496/196352、K1B 104036/196352；`--- cb11:` K1A 90112/196352、K1B 79460/196352；
  均 `padCount_ALL0=True`、`all32B=True`、`dynamic=[]`；整体 `FAILURES: 0`、`VERDICT: PASS`）。
  **未实施的前置条件**：① 属本轮预授权外的 kernel 改动；② **4 条测试表项**的硬编码 `expect_use_split`
  会失真（扫描 84 条含 shape 的表项 ⟹ `L1_CASES/l1_tailmid_V`、`L1B_CASES/l1b_tailmid_V`、
  `STAB_CASES/stab_B7_V4409_bv2048_gumbel`、`STAB_CASES/stab_B5_V10240_gumbel`），须同 commit 改测试文件；
  ③ 改完须重跑 5 门禁 + 36-case A/B
- **U3 / V-S8 关闭 —— case8（`K_MAX=128, BV=2048`）单独实测**：fused 91.22(base1)/91.26(base2) →
  **43.04 us = 2.119×（−52.82%）**；mask 59.94/59.76 → **37.88 us = 1.582×（−36.80%）**；`mismatch=0`。
  逐 kernel：base `K1=53.980 + K2g=37.280` → cand `K1A=6.460 + K1B=26.480 + K2gA=7.400 + K2gB=2.580`
  （K1 侧 1.639×、K2g 侧 3.735×）。两趟 base 相差仅 0.04% ⟹ 效应远超漂移。
  **判定：保留，未动用预授权退路**（`preauthorised_retreat_used=false`，未加 `and K_MAX == 256`，
  kernel sha256 未变）。且**与 §12.4 warning 3 的预期相反** —— 该 warning 因 R-F128（`F(128)` 不可分离、
  拟合误差 −15.0%）判定 case8 的 1.57×/1.87× **偏乐观**；实测 mask 与预测几乎重合（+0.8%）、
  fused **超出**（+13.3%）⟹ **R-F128 可降级**：fused 任务上 case8 是**第二好**的 split case
  （2.119×，仅次于 case7 的 4.068×），mask 上第三（1.582×）
- **U4 / V-S7 关闭 —— msprof 运行时 `Block Num`**：`ksplit_fine.py` 对每 step 四重门禁
  （launch 数 / `Block Num` 序列 / 序列之和 / **定理 P2**：split-**并行** kernel 的 `Block Num ≤ cube_core_num`），
  期望值**由交付的 `_r4_split_plan` 派生**（单一真相源，§15.3.1 步骤 1）故不可能与实现脱节。
  **6/6 CSV 全绿**：`steps_attributed=144`、`anomaly_rows=0`、P2 **PASS**、Block Num **PASS**；
  cand 两臂 `split=6 current=12`，base 四臂 `split=0 current=18` ⟹ **864/864 step 无异常**。
  运行时实测表（cand 臂）：case3 fused `[16,4,16,4]` / mask `[16,4,4]`；case4 `[8,2,8,2]`/`[8,2,2]`；
  **case7 `[20,2,20,2]`/`[20,2,2]`**；case8 `[16,2,16,2]`/`[16,2,2]`；case12 `[16,8,16,8]`/`[16,8,8]`；
  case17 `[16,8,16,8]`/`[16,8,8]`；12 个非 split case 两臂均 `[b_num,b_num]`。
  **定理 P2 运行时成立，零违例**；最紧点 = **case7 的 `K1A`/`K2gA` `Block Num = 20 == cube_core_num = 20`**，
  恰 1 满波。**case7 fused 实测 `[20, 2, 20, 2]`（4 行）与 Stage 2 干跑预测逐元素吻合**（取自 msprof，非取自 plan 函数）
- **§11.4 步骤 11 关闭 —— cann-bench 36/36 端到端**：A/B 的 cand 趟**就是**该端到端运行。
  `cann_correctness_eval_20261011_150635`（R4FusedSampling）**18/18** pass、worst `mismatch_count=**0**`；
  `_150836`（R4MaskSample）**18/18**、`mismatch_count=**0**`；两份 performance 报告的 accuracy 段同为 18/18 + 0。
  **⟹ 36/36 `mismatch_count = 0`**。前置条件已核：submission 副本 `cmp` 逐字节相同；wheel 由 `build.sh`
  重建 rc=0（容器 hash `d9f31ae1… → bdfa160b…` **仅因 wheel 内嵌时间戳**，已解包验证其中
  `_r4_kernels.py` 仍是 `a764e117…`）；`Block Num` gate 已更新；README 已记录 harness 偏离。
  两个已知环境坑均已处置（`R4_HTML_SHIM=1` + sitecustomize shim 上 `PYTHONPATH`；
  `LD_LIBRARY_PATH` 以 `/usr/local/sqlite3/lib` 为前缀 ⟹ 全部 `elapsed_us` 非零，即该坑未踩）
- **GATE 1–4 关闭**：`bench/hap_thw_model/gates_report.py` **exit 0** —— GATE1 **PASS**
  （softmax 20/20 + arg_max 20/20）、GATE2 **PASS**（36/36，全 vector-bound）、GATE3 **PASS**、
  GATE4 **PASS**。`t_hw` **未重推导**（依 §15.3.2：唯一变化是 `+S` 次跨 split 归并，case7 仅 **+0.15%**，
  远低于 GATE1 容差；GATE1 精确复现已发布值 ⟹ 该决定被验证）
- **roofline 达成率变化（含 2 趟 GM 读的口径注记）**：`attain% = 100·t_hw_us/elapsed_us`，两臂同一 `t_hw`。
  fused 全 18 geomean **2.1961% → 2.8101%**（1.2796×）；split-6 **1.3612% → 2.8073%**（2.0624×）。
  mask 全 18 **1.9493% → 2.3080%**（1.1840×）；split-6 **1.0803% → 1.7754%**（1.6435×）。
  标杆单 case：**fused case7 达成率 0.672% → 2.733%**（§12.2(d) 中 36/40 个 AIV 闲置的最差核心饥饿 case，
  现在 K1A/K2gA 上是 40/40 满负荷）。
  ⚠️ **§15.3.2 要求的口径注记（本轮为计算而非口述）**：rev3 split 路径**两趟**读 GM 的 `logits`
  （K1A 供 Sort 侧、K1B 供 softmax 侧），而已发布 `t_hw` 只计**一趟**（`hap_lib.py:20` unique-bytes /
  no-reload-factor 约定）。按 `t_hw_2pass = max(t_cube, 2·t_read, t_write, t_vector)`（同样 ≥1.0 截断）重算：
  **fused 0/6 受影响** —— `2·t_read` 处处低于 `t_vector`（case7 `2×1.2672=2.5344 < 4.8946`）⟹ 地板不变、
  仍全 vector-bound，**上述 fused 达成率无偏**；**mask 3/6 受影响（case7/12/17）** —— mask 任务算术强度仅
  **~2.57 ops/byte，本就低于 3.125 ridge point**（GATE4），vec/read 余量薄至 1.062× ⟹ 计入 reload 后
  归因翻转为 **READ-bound**：case7 `t_hw 1.3464→**2.5345**`(1.882×)、达成率 0.779%→**1.466%**；
  case12 `1.1676→**2.1847**`、1.579%→**2.955%**；case17 `5.3964→**10.1587**`、1.679%→**3.162%**。
  **⟹ 任何已发布的 mask split-case 达成率都必须带此注记**；GATE2 的「全 vector-bound」是**针对已发布
  unique-bytes 模型**的结论、原样 PASS，若将来把 reload 计入 `t_hw` 则 GATE1 重推导式与 GATE2 都须同步更新
- **🔴 本轮实测中发现的重大如实保留项 —— 计分口径 vs 墙钟 SPAN**（`analyze_span.py` / `span_vs_scored.txt`）：
  为反推 `a_launch` 而读 kernel 间隙时发现间隙**很大且两臂不对称**。因 `elapsed_us` 是 **Duration 之和**，
  间隙**不计分**；但不报就会让 rev3 端到端看起来比实际好，故两口径并列。
  `span_us = 每 step（末 kernel 结束 − 首 kernel 开始）的中位数`，`gap_us = span − scored`：
  **fused geomean SCORED `1.2796×` 但 SPAN `0.7443×`**（launch 数变化的 6 个 split case 上 SPAN 仅
  **`0.3831×`**，12 个不变的 1.0375×）；**mask SCORED `1.1840×` 但 SPAN `0.8005×`**（split 上 **`0.4387×`**）。
  base 臂每次 launch 的 host 间隙中位数 = fused **362.7 us** / mask **386.2 us**。
  具体到 **case7 fused**：base `K1=425.04 + gap 3.00 + K2g=303.58` ⟹ span **732.00 us**；
  cand `42.44 + 416.00 + 97.10 + 539.50 + 36.64 + 334.25 + 2.84 + 304.75` ⟹ span **1360.25 us**
  —— 计分器件时间 **728.42 → 179.06 us（4.068× 更好）**，span **1.86× 更差**。
  **机理（有据，非臆测）**：① **base 臂本就**为每个「首 kernel 短于该开销」的 case 付 ~363–386 us/launch
  （18 个 fused case 中 **13 个**间隙 310–392 us，另 5 个（7/15/16/17/18）仅 2.8–11.5 us）⟹ **既有、非 rev3 引入**；
  ② 间隙近乎**常数**、与 kernel 和 shape 无关 ⟹ 固定 host 侧（Python/ACL 发射准备）开销的特征，不是器件成本；
  ③ 当某 kernel 跑得够久、host 得以异步排队下一次发射时，间隙**塌缩到 ~2.8–3.0 us**；
  ④ rev3 把 **1–2 个长 kernel 换成 3–4 个短 kernel** ⟹ 原先被隐藏的开销在 split case 上暴露出来。
  **没有任何 kernel 变慢**：case7 fused 的 K1 侧 `425.04 → 139.54 us`（3.046×）、K2g 侧 `303.58 → 39.48 us`（7.690×）。
  **⟹ 这是 host 侧软件开销发现，不是器件回退**：cann-bench 计分值、roofline 达成率、核心利用率论证
  （4/40 → 40/40 AIV）都是器件侧且都真实改善。显而易见的缓解是 **ACL graph capture**
  （分派刻意只用 host Python 整数运算以免破坏 capture —— §5.6 audit block 2），届时 launch 开销降到 us 级、
  器件侧收益即为实际运行收益。**此项未验证 —— 本轮未做任何 graph-capture 测量**
- precision: **pass** —— 因 A/B 曾两度把 `r4_fused_sampling.py` 换成 BASE 再换回、并三次重建 wheel，
  故在**最终交付态**重跑 5 门禁（`run_gates_round3.sh` / `gate_exits.txt`）：
  static **exit 0**（10 s，**38 STATIC_PASS / 0 FAIL**）、all **exit 0**（35 s，**143 PRECISION_PASS / 0 FAIL** +
  **56 BOUNDARY_PASS / 6 WARN** + 38 STATIC_PASS）、stability **exit 0**（18 s，**21 PASS / 0 FAIL**，
  `STAB_split=4 ADV_split=4`，**8 个 ADV 全 `dropped=0`**）、stress **exit 0**（23 s，**12/12** 全
  `selfdiff=0 goldenfail=0 invariant=0`，含 4 个 split-path 配置）、coverage **exit 0**（**25 PASS / 0 MISS / 0 N/A**）。
  四份测试日志 `grep -cE "_FAIL\]"` 均为 **0**；`sha256_stable_before_after_gates=YES`。
  **计数与 Stage-2 / Orchestrator 参考值逐项相同（38 · 143+56/6WARN · 21 · 12 · 25-0-0），无一下降。**
  6 条 BOUNDARY_WARN 仍是 3 个已知 V-S4 探针名 × 2 任务（`bnd_split0_all_neginf`、
  `bnd_split0to4_all_neginf`、`bnd_splitmid_all_neginf`），每条带 `tier2_bitwise=5/5` 与
  「EXPECTED pre-existing degrade, NOT a rev3 regression」标注 ⟹ **按硬约束 6 未触碰**
- adopted: **yes** —— 采纳 rev3（`a764e117`）为新基线；**本轮未产生新的代码编辑**，
  「采纳」的含义是「rev3 现已**实测确认**有效」而非「改了什么」。`consecutive_no_improvement` **归 0**
  （round 2 累计为 2）。回滚**不需要**：不存在「无可测收益甚至回退」的情形
- rollback_reason: 无
- **JIT 编译成本（单列披露，硬约束 7；不计入上述任何性能数字）**：
  ① **4 个 rev3 新 split kernel 冷编译**（`probe_cold_split_compile.py`，对**全新空** `TILELANG_CACHE_DIR`，
  即冷机器首次运行的真实代价）：**23 个新特化**（24 次调用，`K2gB(16,2)` 被 case12/17 共用）、
  合计 **117.16 s**、均值 **5.09 s**、min 4.49 s / max 6.00 s；按 kernel：K1A **30.05 s**(6)、
  K1B **34.23 s**(6)、K2gA **30.32 s**(6)、K2gB **22.56 s**(5)；交付文件 sha256 前后一致。
  ② **暖缓存**下 24 次 split-kernel 实例化各仅 **0.02–0.13 s**；cand prewarm 的 21 次冷编译（合计 111.44 s）
  **全是冻结 kernel**（K1/K2g/K2m）中共享缓存未见过的 shape，base prewarm 的 2 次（15.89 s）是 case12/17 的 `K1`。
  ③ **eval 墙钟**（fused, mask）：base1 `123 s, 119 s`；cand `124 s, 116 s`；base2 `125 s, 116 s`
  ⟹ cand 臂墙钟**不比** base 臂慢，因两臂均已暖
- 反模式对照（`tilelang-perf-optimization/references/performance-antipatterns.md`）：本轮 kernel 未改动，
  对 split 架构的对照结论为**正面** —— 无跨级 GM→L0 搬运（全部为 `T.copy` GM↔UB 与 32 B 对齐的 UB↔UB，
  由静态门禁 assert-④ 复验 `padCount_all0=True`、`all32B=True`）、无动态循环边界（`dyn_bounds=[]`）、
  未新增串行标量链（`F(256)` 的 `tau_p` epilogue 仍**只跑一次**、在 K1B 内，正如 §12.2 所预测）。
  **一处有意保留、按要求记录**：split 路径**两趟读 GM 的 `logits`**，是拿 HBM 带宽换核心并行度。
  保留理由：18 case 中 14 个 `b_num < 20` 属**核心饥饿**而非带宽受限（达成率 0.67%–5.3%），
  实测确认该交换在器件时间上划算（case7：4/40 → 40/40 AIV、4.068×）；其代价在上方 §roofline 与
  SPAN 两处**量化披露而非隐藏**
- next_hint: ① **最高优先 = 关闭 SPAN 口径的风险**（上面 🔴 项）：做一次 **ACL graph capture** 下的
  端到端测量，判定器件侧收益能否在墙钟上兑现。**这一项同时决定 ②③ 的方向** —— 若 capture 下 launch 开销降到 us 级，
  则 §16.2 的**选项 A（4-pass，理论 ~6× vs rev3 选项 D 的 ~3.1×）值得作为增量**，因为反对 A 的主要理由
  （「4 次 launch 押在未实测的 `a_launch` 上」）已被本轮 `a_launch < 1.54 us` 消除；
  反之若 eager 模式就是部署形态，则**任何增加 launch 数的方案（含 A）都不可取**，
  而应优先压缩 host 侧发射准备成本。这两个事实方向相反，**不做 capture 测量无法裁决**。
  ② 可立即拿到的 **+2.82%**（36-case geomean 1.2309→1.2656×）= 放宽 guard 收纳 case6/case11，
  已实测（1.15×–1.44×、4/4 bitwise 相同、几何已被静态门禁覆盖），但需 Orchestrator 批准 +
  同 commit 改 4 条测试表项 + 重跑全门禁与 A/B。③ 若要修正 §12.4 模型的**绝对**预测能力，
  需一次专门的 `b_num` 扫描重新标定 `F`（本轮已证 `F(256)=21.63` 不是 `b_num`-无关常数）；
  模型的 **geomean 带**本轮已被验证可用，故此项非阻塞
- 验证缺口（如实）：**未做 ACL graph capture 测量**（🔴 项 1）；**~360 us/launch 的 host 间隙未做根因定位**
  （其行为特征有据 —— 常数、base 臂既有、被长 kernel 掩盖 —— 但未剖析 host 侧以区分 Python wrapper /
  张量分配 / ACL 发射各占多少，也未确定其中多少属 msprof 插桩开销）；`a_launch` 是**上界而非点值**
  （E1 `<1.540 us`、E2 截距 ≈0，但没有「零工作量 launch」的估计量，下游一律用保守的 ≤3.100 us）；
  §12.4 的 `F(256)` 被证伪为 `b_num`-无关常数但**未重新标定**；guard 放宽的收益出自
  `probe_guard_relax.py`，该探针**未复现 harness 的 `CannBenchCacheClean`**（绝对 us 可能略乐观，
  但两臂条件相同 ⟹ 只用其比值，已在 docstring 声明）；**case6/11 的 split 路径当前无运行时精度覆盖**
  （探针只在一个输入分布上验了 bitwise A==B，特化本身静态断言干净，但交付的运行时套件把这两个 shape
  走的是**现行**路径 ⟹ 放宽 guard 须同时让 tier1/tier2/boundary/stress 覆盖其 split 形态）；
  漂移对照**只有 1 组重复**（base1/base2 每任务一对，足以解释全部非 split 残差，但要更紧的漂移界需更多重复）；
  `t_hw` **有意未重推导**（若将来把 reload 计入模型，GATE1 重推导式与 GATE2 归因都须更新，
  且 mask case7/12/17 会翻为 READ-bound）；**未探索任何新优化点**（本轮任务书即如此限定）；
  `.orchestrator_state.json`（含 `.bak_*`）全程**未读写**
- skills_consulted: `tilelang-perf-optimization`（性能数据采集口径 / 瓶颈诊断 / 反模式对照方法论）；
  API 形态未凭记忆 —— 全部由交付 kernel 内已上板验证的先例与一手源码核实（`_r4_split_plan`、
  `detect_cube_core_num`、4 个 split kernel 签名），符合 AGENTS.md 核心原则 1/2 的查证顺序
- 详细论证与全部证据：**`perf_tuning/optimization_log_round3.md`**（12 节，含 `[RESULT-…]` 结论行）、
  **`perf_tuning/final_report_round3.md`**、**`perf_tuning/baseline_round3.json`**（逐 case 结构化基线）；
  原始数据与可复现脚本见 **`perf_tuning/round3_splitv_perf/`**：`ab_all.sh`/`ab_run.sh`（三趟 A/B 驱动）、
  `env_r4.sh`（环境配方）、`prewarm.py`+`prewarm_{cand_coldcheck,base_warm}.json`（缓存对称性与编译普查）、
  `cmp_round3.py`+`ab_3pass_comparison.{txt,json}`、`ksplit_fine.py`+`ksplit_*.{txt,json}`（6 个 CSV 的
  逐 kernel 拆分与 Block Num/P2 门禁）、`csv_manifest.txt`、`analyze_a_launch.py`+`a_launch_analysis.{txt,json}`、
  `probe_guard_relax.py`+`guard_relax_probe.json`+`guard_relax_projection.json`、
  `probe_cold_split_compile.py`+`probe_cold_split_compile.json`、`analyze_followups.py`+`followups_analysis.{txt,json}`、
  `analyze_span.py`+`span_vs_scored.{txt,json}`、`verify_frozen.py`+`verify_frozen_base_vs_cand.txt`、
  `run_gates_round3.sh`+`gate_{static,all,stability,stress,coverage}.log`+`gate_exits.txt`+`gate_sha256_{before,after}.txt`、
   `gates_report_round3.log`、`eval_{base1,cand,base2}_{R4FusedSampling,R4MaskSample}.log`

## Iteration round3-HOSTATTR — 2026-10-11T18:10（round 3 续做，**只读根因调查；kernel 未改动**）
- **本轮性质**：窄范围续做，综合已完成的 7 个探针实测（`round3_host_overhead/probe1..7` +
  `corrected_span` + `reconcile`），产出 **`round3_host_overhead/attribution_report.md`**。关闭上一轮
  `next_hint` 的 **#1 项（ACL graph capture 未测）** 与 **验证缺口 #2（~360 us/launch host 间隙未根因定位）**。
  **未重跑任何昂贵测量**；唯一计算 = 用 `corrected_span.json` 已测的 device durations + 文档化 `model_span`
  重算固定 H（499.6/433.9）下的 SPAN geomean（纯算术，无 NPU/profiler）。交付 kernel sha256 前后均
  **`a764e117…`**（`sha256_before.txt == sha256_after.txt`，本会话再次 live 复核一致；test `cea988cb…`、
  proto `41ab7970…` 未改）。`.orchestrator_state.json` 全程未读写。
- bottleneck_type: **host-side software overhead（sync/dispatch 类，非 device compute/transfer）**。
  逐 launch host 成本分解（probe3 阶梯 L0→L5，median over cases 1/3/13/7）：
  **C1 msprof 插桩 = 65.73 us/launch**（L5−L3；probe1 `_c1_isolation` 交叉验证 +58.85）·
  **C2 cann-bench harness guards = 233.40 us/launch**（L3−L0；probe7 faithful harness 复现）·
  **C4 tilelang 框架级 dispatch ≈ 135–158 us/launch**（probe2b `model_prep`；单一最大原语 `Analyzer()` =
  **122.603 us**，n=500）· **C5 ACL launch 残差 ≈ 17–38 us/launch**（probe2b `residual`，减法得到，
  `model_residual_max_us=38.461`）· **C3 我们的 wrapper <2 %**（probe5 `C_pct` 0.97–1.97 %，4.01–15.76 us/call）。
  加性校验：C4+C5 ≈ L0=200.5；L0+C2+C1 = **499.6 = L5** ✓。GC 关闭后 Analyzer 仍 120.352 us ⟹ 非 GC 抖动。
- **是否 R4 特有 = 否，框架级**（probe2 `controls`）：`torch.add`(NPU) 仅 **20.9 us/launch**；**空 tilelang
  kernel（零 R4 逻辑）= 160.6 us（0-alloc）/ 180.5 us（1-alloc）**；R4 wrapper L0=200.5 us 只比最小 tilelang
  高 ~20–40 us。⟹ ≳90 % 的裸发射成本是任意 tilelang kernel 都要付的框架 dispatch（Analyzer 为主），非 rev3 引入。
- **rev3 是否让 host 变重 = 否**（probe3 `base_vs_cand`）：base(2 launch) vs cand(4 launch) 的**单次**成本
  case3 195.118→180.531、case7 196.168→179.056（`per_launch_ratio_cand_over_base` 0.9252/0.9128，反而略低）。
  rev3 只是把 launch 数 2→4，单次成本不变 ⟹ §7 SPAN 回退纯属「launch 数 × 固定 host 成本」，非单次变贵。
- **⭐ SPAN 结论稳健性重估**（corrected_span 模型，V1 0.058 %/V2 7.583 % 均通过；固定 H 行本轮重算，
  H=200.5/graph 行精确复现存量 geomean）：
  - fused all-18：msprof-H499.6 **0.74434**（≈ round-3 发布 0.7443，差 0.005 % — 自洽校验）· guards-H433.9
    0.75091 · bare-H200.5 0.83513 · **ACL graph 1.12421**；split-6：0.41214 / 0.42311 / 0.58159 / **1.40886**
  - mask all-18：0.79803（≈ round-3 0.8005）/ 0.79866 / 0.86548 / **1.08422**；split-6：0.50819 / 0.50938 /
    0.64807 / **1.26760**
  - **裁决**：eager（任意 H，含裸调 200.5）下 rev3 的 split 路径 SPAN **是回退**（split-6 <1、all-18 <1，
    因 2 次额外 launch × 200–500 us host > 省下的 ~65–80 us device）；**ACL graph 下翻为赢**（all-18 1.08–1.12×、
    split-6 1.27–1.41×）；**契约 scored（device 时长和，不计 gap）恒为 1.28×/1.18×，与 host 口径无关**。
- **ACL graph 裁决**：可用（api_surface 全 true，torch_npu graphs.py + 仓库先例 rms_rope_aclgraph.py）·
  **逐位正确**（probe4 `bitwise_equal_to_eager=true`；probe6 realistic copy-in+replay，5 组不同输入值 × 5 输出
  × 4 case 全 `per_output_bitwise=[true]×5`）· 收益（per-launch host 200–500→**2.8–6.6 us**，30–70×；
  `speedup_L0` 3.5–6.9×；额外 launch 边际 host ≈ 0.09/−0.51 us）· 代价（copy-in D2D **80–89 us/step**、
  烘焙设备地址 ⟹ `input_pool.py` 轮换输入必须走 copy-in、capture 一次性 3.5–5.5 ms、动态 shape 需逐个 capture、
  仅测 4/18 case）。
- **§16.2 选项 A 裁决 = 暂不做 A，维持 D（rev3）交付**。graph 数据消除了「多 launch 昂贵」的反对理由
  （graph 下 +2 launch 只多 ~6–13 us host），但 A 仍被三点否决：① A 的 ~6× 器件优势是**从未实测**的模型估计，
  而 round-3 §4 已证伪该模型的绝对预测（F 非 b_num-可分、case4 +34.6 %）；② A 的 m/s bitwise 依赖 Pass2 精确
  还原 + Pass4 精确重放（两个论证），D 是构造性 bitwise（硬约束 2）；③ 使 A 变便宜的 graph 部署本身未全量验证
  且非当前契约目标（cann-bench 记 device 时长，D 已 1.28×）。重启 A 的前置测量：4-pass K1 vs K1A+K1B 的
  **纯 device msprof A/B** + Pass2/Pass4 bitwise 原型 + graph 全 18 case 验证。
- **缓解优先级**：M1 ACL graph capture（唯一消除根因，30–70×，已证 bitwise）≫ M2 框架级 lazy Analyzer+cached
  stream（−95.9 %，但属 tilelang 框架改动、不在本算子范围）> M3 `_r4_split_plan` 缓存 / M4 workspace 复用
  （均 C3，<2 %，低优先）> M6 wrapper 瘦身（≈0，不值得）；**M5 合并 K1A+K1B 减 launch = 不可行**（需 grid 同步，
  DESIGN §2.5 已核实 Developer 模式无 grid barrier/semaphore，自造 atomic 自旋有波间死锁风险）。
- precision: **未改动 kernel ⟹ 精度状态不变**（沿用 round-3 全绿：static 38/0、all 143 PASS+56 BND/6W、
  stability 21/0、stress 12/12、coverage 25-0-0）。本轮不触发精度复验（无实现改动）。
- adopted: **N/A（调查轮，无代码采纳/回滚）**。consecutive_no_improvement 不变（本轮不产生性能采纳事件）。
- rollback_reason: 无（kernel 未改）。
- design_error: **no**（根因是 host 侧软件栈开销 + harness/msprof 口径，非 R4 设计缺陷；已实证 ≥3 类缓解
  M1–M6，M1 可消除，无需回退 Stage 1）。
- next_hint: ① 若交付目标含 eager 墙钟 ⟹ **落地 M1（ACL graph capture + copy-in）** 并补全 18 case graph A/B
  （当前仅 4/18；device-bound 的 15/16/18 与 guard-marginal 的 6/11 未测）；② 若要裁决 §16.2 选项 A ⟹ 先做
  4-pass K1 vs K1A+K1B 的纯 device msprof A/B + Pass2/Pass4 bitwise 原型；③ M3/M4（<2 %）不建议单独实施。
- 验证缺口（如实）：`corrected_span` 是**模型估计非直接测量**（V1 只锚定 base 臂 span，cand 臂 split span 无直接
  锚点 ⟹ split-6 带 ±7.6 % 不确定度，应读作「≈0.4–0.5，确定 <1」）；**C5（ACL launch）是减法残差**，未能与
  FFI 参数编组分离（无探针单测 `lib.call`）；C2 的 233 us 未按 guard 子层跨全部 case 细分（属 harness 代码、非我方所有）；
  graph 仅 4/18 case 实测，G=15.588 us/launch 由 4 个样本推得并统一外推到 18 case；`probe_guard_relax.py` 未含
  harness 的 `CannBenchCacheClean`（绝对 us 或略乐观，只用比值）。
- skills_consulted: `tilelang-perf-optimization`（host 侧性能数据采集口径 / 瓶颈诊断 / 反模式对照方法论；
  本轮所有 host 计时均标注方法 `perf_counter_ns` 与重复次数 n=25..500，符合「禁止用 Python 计时声称 kernel 时长」
  的边界 —— 测的是 host 发射准备，device 时长一律取自 round-3 msprof CSV）。API 形态未凭记忆：ACL graph 依据
  `torch_npu/npu/graphs.py` 行号 + 仓库先例 `examples/aclgraph/rms_rope_aclgraph.py`；grid 同步不可得依据 DESIGN §2.5
  已核实的源码结论。
- 详细论证与全部证据：**`round3_host_overhead/attribution_report.md`**（9 节，C1–C5 每个数字标注来源文件+字段路径）。

---

## Iteration round3-itemA — 2026-10-12T20:25（round 3 最后一项，**已采纳**：split guard 常数放宽到实测值）

- **本轮性质**：单一授权的 kernel 改动 + 强制配套。round 3 §3.1 已实测出 `a_launch` 并建议放宽 guard，
  但因超出该轮预授权而**未实施**；本项实施它，并完成 Orchestrator 指定的 4 项配套（4 处陈旧
  `expect_use_split`、case6/11 运行时 split-path 覆盖、`_r4_split_plan` 复现自检、SPAN 后果量化）。
- bottleneck_type: **compute**（Sort/top-k 串行链）+ **核心饥饿**。case 6/11 的 `b_num=4`，只用 4/20
  cube 组 ⟹ V 轴 split 把 Sort 侧并行到 `b_num*S`（cb6 4→**12**、cb11 4→**8**，均 ≤20，theorem P2 成立）。
  guard 常数本身是 **host 侧纯 int/float 算术**，不是设备瓶颈；本项修的是"保守常数把两个可获益的 case
  挡在并行路径之外"这一**调度层**问题。
- optimization: `_SPLIT_A_LAUNCH_US` **8.0（Stage 2 保守占位）→ 1.540（E1 model-free 实测下界）**。
  kernel diff = **1 hunk，+41/−1 行**，其中唯一的非注释行是该赋值本身（`verify_frozen_itemA.py`:
  `hunks=1`、`code_added=['_SPLIT_A_LAUNCH_US = 1.540']`、`only_constant_changed=True`）。
  `_SPLIT_SAFETY=1.5` **未动** ⟹ 三重收益判据仍要求 `pred_saving > 1.5*pred_cost`。
  **3 个冻结 kernel 逐字节未变**（L91-364 `d3909063…`、L367-412 `f787d1ad…`、L415-555 `ca90de8d…`，
  行区间与 sha256 前后完全一致且等于 rev2 基线常数）；7 个 `@tilelang.jit` 集合与顺序未变。
- **取值依据**（写在代码注释里，非事后追认）：1.540 = E1 中 `K2gB`（rev3 链最小完整 kernel，UB=132B）
  在 6 个 split fused case 上的 msprof `Duration` **最小值**；因 K2gB 仍做非零折叠工作（每行 ≤10 partial），
  真实 launch floor **严格低于** 1.540 ⟹ 该值是 `a_launch` 的**保守上界**，使 `pred_cost` 继续被高估、
  guard 继续偏向"不切"（安全方向）。E2 最小二乘截距 −0.050us≈0 独立佐证。**否决**：3.100（E1 max，
  但 >3.013 ⟹ case 6 不翻转，达不成本项目标）、2.840（E1 median，含折叠工作 ⟹ 非纯 launch）、
  2.74（扫描格点，无独立测量含义）、0.0/−0.050（E2 截距，§3 自陈为 6 点 3 参薄拟合外推、
  "只是量级交叉验证而非答案"，且 0 会把 launch 记账从 guard 中删掉）。
- **该值下 `_r4_split_plan` 准入集合 = `{3,4,6,7,8,11,12,17}`（8 个）**，与预期完全一致
  （`verify_split_plan_itemA.py` **SELF-CHECK PASS**，exit 0：直接读文件中已交付的常数、不做 monkey-patch，
  故不可能与交付逻辑偏离；18 case 的 `(S,NBS,S_TAIL)` **全部未变**——常数只在 cost guard 内、
  步骤 1–6 导出几何**之后**才被读取；theorem P2 全成立；`_SPLIT_SAFETY` 仍为 1.5）。
  阈值解析：case 11 在 a<3.987 翻转、case 6 在 a<3.013 翻转，**a ≤ 3.01 后集合饱和于 8** ⟹ 1.540 非刀刃值。
  其余 10 case 为**结构性**不可达（`S_cap=0`: 9/15/16/18；`=1`: 14；`n_blk_total=1`: 1/2/10；
  `n_blk_full<2`: 5/13），任何 `a_launch` 取值都翻不动。
- baseline_time: **改前在本会话先重跑 5 项门禁**（`gate_pre_*`，`a764e117`），逐项精确复现 Orchestrator
  所报计数：static **38/0**、all **143 PRECISION_PASS/0 FAIL + 56 BOUNDARY_PASS/6 WARN + 38 STATIC**、
  stability **21/0**、stress **12/0**、coverage **25-0-0**，5/5 exit 0；`STAB=4/12 ADV=4/8` split 配置、
  ADV 8/8 `dropped=0`、`cube_core_num=20` 探测所得。
- candidate_time: **同会话 3-arm A/B**（`ab_all_itemA.sh`，序 base1 `a764e117`(a=8.0) → cand `c59a83bc`
  (a=1.540) → base2 `a764e117` 纯漂移对照），两算子 × 18 case，6/6 eval **exit 0**、6/6 报告 18/18 pass 0 fail。
  口径 = cann-bench → msprof `kernel_details.csv`，warmup=3 repeat=5，每 step `Duration` 求和取中位数。
  **全程无 Python/torch 计时**。每臂内先跑 `prewarm_itemA.py`（base1 90 specs / cand 96 / base2 90，
  **cand cold=0**），编译不落入任何计时窗口。
- improvement:
  - **判据 1（case 6/11 逐任务，harness 口径）全部 >3%**：case6 fused 78.000→58.500us = **1.3333×**
    （−25.00%，漂移对照 +0.00%）；case6 mask 55.160→44.560 = **1.2379×**（−19.22%，对照 +0.54%）；
    case11 fused 62.740→54.160 = **1.1584×**（−13.68%，对照 +0.35%）；case11 mask 47.160→43.360 =
    **1.0876×**（−8.06%，对照 +0.17%）。运行时 `Block Num` 证实确已走 split（cb6 fused `[4,4]→[12,4,12,4]`、
    mask `[4,4]→[12,4,4]`；cb11 fused `[4,4]→[8,4,8,4]`、mask `[4,4]→[8,4,4]`），6 份 CSV 全部
    `Block Num 序列 == plan 期望 PASS`、`theorem P2 PASS`、`anomaly_rows=0`、`steps_attributed=144`，
    base 臂 `split=6 current=12` → cand 臂 `split=8 current=10`。
  - **与 round 3 §3.1 的独立探针互证**（不同会话、不同脚本）：1.4442/1.3089/1.2479/1.1537× vs 本轮
    harness 1.3333/1.2379/1.1584/1.0876×；harness 读数偏低是因其 per-step 求和含 `CannBenchCacheClean`
    而探针不含（已记录 caveat，两臂一致 ⟹ 比值不受影响）。
  - **交错空探针二次复现**（`probe_null_case13.py`，5 轮，kernel 文件 sha256 前后均 `c59a83bc` 未改）：
    case6 fused **1.4366×**（−30.39%，per-round 离散仅 0.12pp）、case6 mask **1.3167×**（−24.05%）、
    case11 fused **1.2350×**（−19.03%）、case11 mask **1.1272×**（−11.29%）。
  - **geomean（device time，harness 口径）**：cand/base1 36-case 合并 **1.0170×**、fused-18 **1.0227×**、
    mask-18 **1.0113×**、4 个翻转测量 **1.2009×**。与 §2.3 **实测**的 pre-rev3 锚点**复合**
    （该臂本轮**未重跑**，故为复合而非新测量）：fused 1.2796×1.0227=**1.3086×**、
    mask 1.1840×1.0113=**1.1974×**、合并 1.2309×1.0170=**1.2518×**，即较交付值 **+1.70%**。
    低于 §3.1 的 +2.82% 投影，因为投影用的是探针比值（step 和不含 `CannBenchCacheClean`）而本轮是
    harness 端到端口径。cann-bench `score`：fused 60.7416→**60.8588**、mask 59.2318→**59.2894**
    （漂移对照 60.7181 / 59.2055）。
- **判据 2 的仲裁（本项最重要的方法论环节，未被"论证"绕过）**：单样本 A/B 恰好标出 1 对越界 ——
  **case 13 mask 14.620→15.240us（+4.24%，绝对 +0.62us）**，而同码漂移对照读 14.720us（+0.68%）。
  `cmp_itemA.py` 因此如实输出 criterion2=**FAIL**、`OVERALL=DO NOT ADOPT`，该原始记录**未被覆盖**，
  完整保留为 `cmp_itemA_single_sample_1run.log`。case 13 = `(B=6,V=2304,K_MAX=128,BV=2048)`：
  `n_blk_full=2304//2048=1 < 2` ⟹ 两个 guard 值下 `_r4_split_plan` 均返回 `(False,2,1,1)`，
  **cost guard 根本未被触及**；`ksplit_fine_itemA.py` 独立确认三臂的 launch 序列（`K1`+`K2m`）与
  Block Num 完全相同。随后用两个实验实测判定，而非靠推理：
  - **交错空探针**：5 个 **NULL** 探针（两臂 dispatch 可证同一 ⟹ 任何差异都是纯噪声）的 per-round
    |delta| 包络达 **5.18%**；被标记的 case13 mask 读 **−0.42%，+4.24% 不复现**；而更小的
    case1 mask（11.14us，同为 identical-dispatch）纯噪声即达 **−5.03%**，量级已超过被标记值。
  - **harness 口径重复采样**（因判据是以 harness 口径陈述的，而空探针不含 `CannBenchCacheClean`，
    不能单独仲裁）：`ab_repeat_mask_itemA.sh` 交替重跑 `R4MaskSample` 各 3 次（6/6 exit 0，
    kernel 恢复 `c59a83bc` 并举证、submission 副本 `cmp` 一致、wheel 重建），得到 **base 5 样本 /
    cand 4 样本**：case13 mask base `[14.62,14.72,14.58,14.78,14.70]` 中位 **14.700**、
    cand `[15.24,14.70,14.96,14.80]` 中位 **14.880** ⟹ **中位差 +1.22%（非 +4.24%）**，
    两分布**重叠**（cand min 14.70 ≤ base max 14.78）。原先的 +4.24% 是"cand 最差样本 vs base 最好样本"
    的**单样本假象**。预设判据（重叠且中位差落在合并臂内离散内 ⟹ 抖动；cand 整体高于 base 且分布不重叠
    ⟹ 真回退 ⟹ 回滚）返回 **MEASUREMENT SCATTER**。全 18 个 mask case 中 harness 自身噪声底最高达
    **3.98%** 臂内离散 ⟹ **3% 门槛在最小 case 上等于或低于 harness 自身逐 case 噪声**，
    这正是必须用分布而非单样本仲裁的原因。其余非标记 mask case 无一回退 >3%（最差中位差 case1 **+0.54%**）。
- precision: **pass**。① 5 项门禁在候选态全 exit 0 且计数**全部不低于**基线：static **48/0**（38→48）、
  all **151 PRECISION_PASS/0 FAIL**（143→151）**+ 56 BOUNDARY_PASS/6 WARN + 48 STATIC**、
  stability **25/0**（21→25）、stress **16/0**（12→16）、coverage **25-0-0**；
  **6 条 BOUNDARY_WARN 与改前逐条相同**（pre/cand/post 三方 WARN 集合 `diff` 为空，仍是既有的
  `bnd_split0_all_neginf`/`bnd_split0to4_all_neginf`/`bnd_splitmid_all_neginf` × mask/fused，
  **无新增**）。② **A/B 期间 kernel 文件被换过 9 次，故在最终恢复态第三次重跑全部 5 项门禁**
  （`gate_post_*`）：计数与候选态完全一致，`sha256_stable_before_after_gates=YES`。
  ③ cann-bench 精度 **36/36 `accuracy.passed`、180/180 输出张量 `mismatch_count=0`、
  worst `max_abs_diff=0.0`**；另在 9 个池化 mask 臂上逐一复核（每臂 18/18、90/90、mismatch 0）。
  ④ 新增 case6/11 覆盖全部通过：4 个 L1 双路用例 `path=SPLIT` 且几何正确（cb6 `S=3 NBS=1 S_TAIL=2`、
  cb11 `S=2 NBS=1 S_TAIL=-1`）、**`tier2_bitwise=5/5`**、Tier-1 每个输出 `r=1.0000`；
  4 个 STAB 配置 `reps=20 selfdiff=0 goldenfail=0 invariant=0`；UB 投毒下 `reps=200` 同样全零。
  ⑤ **断言零弱化（AST 级举证，`verify_no_weakening.py`）**：52 个函数前后同数，**0 删除**，
  唯一变更体 `test_r4_rev3_l1` 经剥离 docstring 后**可执行体完全相同**；9 张用例表
  **0 条目丢弃、0 非 tag 字段变更、0 tag 列表缩减**（4 处仅 tag **增长**，新增 6 条）；
  `check_precision` 函数体**逐字节相同**；`REV3_FROZEN_SHA256`/`REV3_UB_LIMIT`/`STAB_REPS`/
  `COVERAGE_NA` 未变；`COVERAGE_MANIFEST` 无数值下降、无键删除；**无任何覆盖维度丢 tag**，10 个维度增长。
- adopted: **yes**（`[RESULT-itemA]` ADOPT）。回滚基线 `history_version/r4_fused_sampling_impl_s3r3_itemA_attempt0.py`
  （== `a764e117`）与 `…test…_impl_s3r3_itemA_attempt0.py`（== `cea988cb`）保留未用。
  `consecutive_no_improvement` **归零**（本项产生实测性能采纳事件）。
- rollback_reason: 无。
- design_error: **no**。根因是 host 侧调度常数过于保守（Stage 2 未实测即取 8.0us 占位），
  非 tiling 策略或内存层级安排的设计缺陷；单一常数修正即释放 4 个测量的 1.09×–1.33×，
  且 `_r4_split_plan` 的几何导出与 theorem P2 均无需改动。
- **SPAN 后果（量化并如实披露，非拒绝理由）**：case6/11 的 launch 数 2→4（fused）/ 2→3（mask）。
  用 `round3_host_overhead/analyze_corrected_span.py` 的已验证模型（V1 0.058% / V2 7.583%）+ 本项
  自己的 msprof 逐 launch 时长：**SCORED 口径（契约指标）改善** 1.3333/1.1584/1.2379/1.0876×，
  但**SPAN 在所有部署假设下都变长** —— 裸 eager（H=200.51us）0.3772/0.3626/0.5072/0.5026×、
  harness guards（H=433.91）0.3537/0.3469/0.5034/0.5012×、guards+msprof（H=499.64）
  0.3510/0.3451/0.5029/0.5010×、**ACL graph（15.588us/launch）0.9012/0.8075/0.9450/0.8671×**。
  ⚠️ **这比 §9 的 split-6 图景更差，须谨慎读**：case6/11 是所有 split case 中**设备收益最小**的两个
  （`pred_saving` 8.902us；实测 fused 19.50/8.58us、mask 10.60/3.80us），而 launch 数惩罚同样是 +2/+1；
  graph 下 +2 launch 的 device-timeline 残差 ≈31.2us **超过** 19.50us 的设备节省 ⟹ 0.90×/0.81×，
  而非 §9 对已切 6 case 报的 >1（其收益大得多，case7 达 4.05×）。另需澄清两个"graph"数口径不同且都对：
  **2.8–6.6us** 是 replay 下的 *host CPU* per-launch（`probe4.cap_L0_per_launch`，边际 ≈0.09/−0.51us），
  **15.588us** 是 SPAN 递推所需的 *device-timeline* 残差（`attribution_report.md` L268 明确区分）。
  按用户裁定（保留 rev3、以设备时间为契约口径）**这不是拒绝理由**；但确实意味着 case6/11 是
  **对 launch 数最敏感**的两个 split case，若交付目标将来变为 eager 墙钟，M1（ACL graph capture +
  D2D copy-in）需优先针对这两个 case 重估。**本项未实施任何 M1–M6**（遵指令）。
- **新增 JIT 特化与编译耗时（单列，硬约束 7；不在任何计时窗口内）**：**17 个**新特化
  （`preflight.json` P3），冷编译 **87.15s** 合计、**均 5.13s/spec**（对全新空 `TILELANG_CACHE_DIR`），
  同 17 个 spec 的 warm 复测 **0.0s**。构成：K1A×4、K1B×5、K2gA×4、K2gB×4。
  ⚠️ **比 §3.1 的措辞更大**：该节说放宽"不引入新的 JIT 特化几何"对 18 个**契约** case 是正确的
  （本轮已确认 18 个 case 的 `(S,NBS,S_TAIL)` 全未变），但契约侧仍新增 **8 个编译作业**
  （split kernel 是不同函数、不同参数元组），而 **4 个翻转的测试表条目**再加 12 个，去重后共 **17 个**。
- 交付态：`r4_fused_sampling.py` = `c59a83bc…`（1280 行，7 个 `@tilelang.jit`）；
  `test_r4_fused_sampling.py` = `dc69ad6f…`（2523 行）；submission 副本 `_r4_kernels.py` = `c59a83bc…`、
  **`cmp` 一致（rc=0）**；wheel 重建 rc=0 且从 wheel 内**提取** `_r4_kernels.py` 验证为 `c59a83bc…`；
  `proto.yaml` = `41ab7970…` **未动**（mtime 仍为 10-07）；bench 侧 `cases.yaml|csv` **未动**（mtime 仍为 10-09）；
  `.orchestrator_state.json` **从未读写**（其 mtime 18:48 早于本会话最早产物 18:52）。
  ⚠️ 测试文件在 `gate_post` 之后又改了**一次纯注释重排**（一行 101 字符 → ≤100，项目风格；
  sha `2f837ab1…` → `dc69ad6f…`）。虽是注释，仍**重跑而非断言**：三项证明
  （`verify_no_weakening` / `verify_split_plan_itemA` / `verify_frozen_itemA`）与 5 项门禁全部在最终态
  重跑（`gate_final_*`），计数与 `gate_cand`/`gate_post` **完全一致**
  （static 48/0、all 151/0 + 56/6WARN + 48 STATIC、stability 25/0、stress 16/0、coverage 25-0-0，
  5/5 exit 0，`sha256_stable_before_after_gates=YES`，6 条 WARN 集合与改前 `diff` 为空）。
  格式合规：本项新增/变更行 **0 行超 100 字符**，测试文件全文超宽行反而 **102→101**
  （128 字符的 `l1b_tailmid_V` 被重排）；`ruff` 仅 1 条 `F841`（`r4_fused_sampling.py:442` `AT1` 未用）——
  **改前即存在**（`a764e117` 备份里同样有）且位于**冻结 kernel `r4_gumbel_sample`（L415-555）内部**，
  按禁令 2 不得修改，故刻意保留。
- next_hint: ① §7 的 `gates_report.py` GATE 1–4 与 §7.1 roofline 达成率表**本项未重跑** —— case6/11
  从"vector-bound 单 kernel"变为 split-kernel 归属，若 Orchestrator 要让 §7 表格反映新分派，需补跑
  （`t_hw` 按 §12 item 9 刻意不重导）；② 融合任务未做重复采样（被标记对在 mask 侧），case13 fused
  单样本 +0.86%、空探针 −2.02%，无标记但样本数为 1；③ `stab_B5_V10240_gumbel` 因翻转**失去
  current-path 对照角色**（覆盖计数不受影响，但套件里不再有 32B 对齐 tail 的 current-path 稳定性对照），
  若要恢复需新增一条 current-path 配置（属改测试表，需授权）；④ 两个**契约外**几何现已被运行时行使
  （`(BV=2048,K_MAX=256)`@S=3、`(5,10240,…)`@S=3；契约标定的两个特化是 `(4096,256)` 与 `(2048,128)`，
  该配对下 guard 回落到默认 `t_sort=8.902`）—— 二者均已通过 20 次逐位重复 + 200 次 UB 投毒，属**已验证**，
  但其 `pred_saving` 是估计值而非标定值；⑤ `a_launch` 仍是**上界而非隔离值**（无探针单测裸 ACL launch），
  方向保守故安全。
- 验证缺口（如实）：判据 2 依赖**仲裁**而非单样本直接通过（原始 FAIL 记录已保留）；pre-rev3 锚点
  **未重测**，1.2518× 为复合值；SPAN 全为**模型估计**（cand 臂 split span 无直接锚点，沿用 §9 的
  ±7.6% 不确定度；graph 残差 15.588us/launch 由 **4 个样本**推得并统一外推）；融合任务重复采样缺失；
  §7 GATE 1–4 与 roofline 表未重跑。
- skills_consulted: `tilelang-perf-optimization`（性能数据采集口径 / 瓶颈诊断 / 反模式对照方法论；
  本项所有设备时长一律取自 cann-bench→msprof `kernel_details.csv`，host 计时仅用于冷编译披露并明确单列，
  符合"禁止用 Python/torch 计时声称 kernel 时长"的边界）；`tilelang-api-best-practices` 与
  `examples/` 未被用于新 API（本项不新增任何 API 调用，仅改一个 host 侧 float 常数）。
  反模式对照见 `optimization_log_round3.md` §13.5：设备侧无新增反模式（编辑不含设备代码），
  §11 记录的**一处有意保留**（split 路径两次读 GM `logits`，以 HBM 带宽换核心并行度）本轮未变，
  其代价已在 §13.2(5) 量化而非隐藏。
- 详细论证与全部证据：**`optimization_log_round3.md` §13**（含 `[ORDER-CHECK]` / `[IMPL-itemA]` /
  `[SELF-CHECK]` / `[RESULT-itemA]` 四个标记）+ **`round3_itemA_guard/`**（26 个文件：预检、
  18-case 复现表、3-arm A/B 原始报告、空探针、9 臂仲裁、SPAN 模型、冷编译、冻结证明、
  零弱化 AST 证明、5 项门禁 ×2 轮日志、可复现脚本）。
