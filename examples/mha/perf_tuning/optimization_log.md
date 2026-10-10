# MHA Stage-3 性能优化日志（tilelang-perf-optimization skill 工作流）

- 算子: `examples/mha/mha.py`（MHA forward, BSHD, Developer 模式 threads=1 cv_1_1, T.serial KV 循环）
- 目标: DESIGN §12 baseline_compare（cann-bench tasks/level4/mha, 20 cases, 当前 0.13x 平均加速比）
- 噪声阈值: 3%；计时依据: msprof（唯一），口径 = op_summary Task Duration 中位数（已对齐 cann-bench case1 锚点 46.9us）
- 精度门禁: l0 10/10 PASS（2026-09-18，/tmp/opencode/l0_baseline.log）

## Step 1 基线（baseline.json 已落地）

| case | shape (B,S,S_kv,N,D) | attr | 基线 median us | cann-bench T_cand | CANN baseline | 加速比 |
|------|----------------------|------|---------------|-------------------|---------------|--------|
| case1 | (2,128,128,8,64) | fp16 causal | 46.60 | 46.9 | 10.0 | 0.21x |
| case7 | (4,1024,1024,32,128) | fp16 causal | 16772.47 | 16762 | 432.8 | 0.026x |
| case10 | (1,1,2048,32,128) | fp16 decode | 441.41 | 444 | 53.4 | 0.12x |
| case14 | (128,1,128,32,128) | fp16 decode | 2842.72 | 2808 | 527.6 | 0.19x |
| case20 | (2,128,128,32,128) | fp16 zero | 102.08 | 104.4 | 32.8 | 0.32x |

launch 间 stdev < 0.15%（≪3% 阈值）。设备: Ascend910_9362, **cube_core_num = 20**（实测 torch.npu.get_device_properties）。

## Step 2 算子类型判断

`mha_kernel(...).get_kernel_source()`（case7 配置，/tmp/opencode/mha_kernel_source_case7.cc）：
- `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1)`；含 `if ASCEND_IS_AIC {...}` + `if ASCEND_IS_AIV {...}` 两段（本仓宏名 ASCEND_IS_AIC/AIV 即 skill 表中 IS_ASCEND_AIC/AIV）→ **CV 融合混合型**。
- 结构性观察（生成代码级证据）：
  1. CV 交换经 GM workspace 往返：w1 acc_s(L0C→GM→UB, 16KB/iter)、w2 P(UB→GM→L1, 8KB/iter)、w3 acc_o(L0C→GM→UB, **32KB/iter，循环内**)；workspace 按 block cid 索引 → case7 总足迹 2048×56KB ≈ 114MB。
  2. AIV 侧每 KV 迭代含 3 组 `for h_i in range(64)` 行循环（sub m_i / mul m_i_prev / div sumexp），生成 `PipeBarrier<PIPE_ALL> + LocalTensor.GetValue(h_i)（UB→标量读）+ 64/128 元素微指令` ×64 行；掩码链另有 64×(CompareScalar+Select+2×PipeBarrier)。
  3. `Fill(acc_s_ub,0)+Add(acc_s_ub,acc_s_ub_)` 冗余对（可 Copy 替代）；每迭代 4 次 fp32↔fp16 cast（双舍入栅格，精度红线，保留）。
  4. AIC 侧无预取（T.serial），K/V GM→L1 在临界路径上。

## Step 3 瓶颈定位（msprof op 模式，case7，/tmp/opencode/prof_op_case7）

PipeUtilization.csv 2048 block 均值（每 block 16 KV iters，wall 167.2us → **10.45us/iter**）：

| 指标 | 值 | 占比 | 结论 |
|------|-----|------|------|
| aic_cube_time | 3.32us / 162.4us | **2.0%** | Cube 算力远未饱和 |
| aic_scalar_wait_id1_time | 160.0us / 162.4us | **98.5%** | AIC 几乎全程等待 AIV → **瓶颈在 Vector 侧** |
| aic_mte2_time | 9.08us | 5.6% | K/V GM→L1（无预取，在临界路径） |
| aic_fixpipe_time | 9.13us | 5.6% | L0C→GM w1/w3 dump |
| aiv_vec_time | 94.58us / 167.2us | 56.6% | Vector ALU busy |
| aiv_scalar_time | 82.99us | 49.6% | **标量单元 busy ≈ ALU busy**：GetValue 标量读、微指令下发、barrier |
| aiv_scalar_vector_stall | 29.36us | 17.6% | 标量等 vector（GetValue 依赖） |
| aiv_scalar_wait_time | 35.75us | 21.4% | 等同步 |
| aiv_mte2/mte3 | 7.64 / 2.80us | 4.6/1.7% | w1/w3 GM→UB、w2/P UB→GM |

**瓶颈类型: sync + vector-instruction-issue（非 compute、非 transfer）**：
- 每迭代 ~10.45us 中，vec ALU 理论下界仅 ~0.2-0.7us（~70K 元素 op）；实际被 ~350 条指令的下发/依赖链拖长：128×GetValue+PIPE_ALL（两组行循环）、128×掩码微指令+barrier、~190×PipeBarrier<PIPE_V>（AUTO_SYNC 逐 op 插桩）、3 对 CrossCore flag + 3 段 GM workspace 往返延迟。
- case7 无迭代间隙（2048 block/20 核 × 164us ≈ wall 16.77ms → block 调度已流水化，per-block dispatch 开销已被掩盖）。
- decode 类（case10/14）：valid_m=1 但整 block_M=64 行向量工作全做 → 63/64 空转（§10.3 性能注记 + flash_attn_optimize §9(b)）。

## Part A 优化点清单

> 阅读范围：optimization-guide.md 全部章节目录+各节适用场景/约束、performance-antipatterns.md 全部条目、best-practices/flash_attn_optimize.md 全文、best-practices/cube_optimization_path.md §1/§3/§4、vector-practices/vector_reduce_pass_fusion.md 适用条件。参考行号均为上述文件实际行号。

### P0 Host 侧（optimization-guide §2.12 L714-718，固定优先级第一步）

[#P0] Host 侧预处理内化：**无需改动** — `mha()` 包装（mha.py L192-229）仅 shape/dtype/contiguous 断言 + JIT 分发，零 tensor 操作（Stage 2 [HOST-METADATA-AUDIT] result=allow，本次复读确认）。§2.12.1 Stride Buffer 归一化：不适用（输入契约 contiguous BSHD，kernel 原生行跨步 DMA）。bench 脚本 host 侧同样一次性建张量、零逐迭代搬运。

### optimization-guide.md 各节

[#1] causal KV 运行期裁剪（§10.3-①；guide 无独立章节，归 §2.13 决策链外的循环界优化；先例 mha_sink_fwd_bhsd.py L263-271 公式逐字、example_gqa_sink_bwd_bhsd.py L323-324+L164 threads=1 Developer 运行期 T.serial 界已验证）：**适用（最高优先）** — causal 全掩码块数学贡献恒 0（exp(-inf)=0, r=1），跳过 = 位级等价；case7 迭代 16→avg 8.5（~1.9x）、case1/19 ~1.33x；对非 causal 无效。
```
[KERNEL-REUSE-PRECHECK]
candidate: #1 causal KV trim（先例 mha_sink_fwd L268-271 + sink_bwd L164/L323-324）
expanded_domain: causal 多变体（4 变体中 (T,*)）的 KV 循环界从编译期常量 kv_iters 变为运行期 eff_kv_iters（每 task 不同）
compatibility_dimensions: ①运行期 T.serial 上界在 threads=1 Developer+AUTO 四 pass 下 codegen/运行正确（sink_bwd L323-324 同构先例=已验证 TRUE）；②trim 公式右下角对齐语义（mha_sink L268-271 逐字先例 TRUE）；③eff_kv_iters>=1 恒成立（causal 时 S_kv>=S host 断言 + last_vis>=0 → floordiv+1>=1，TRUE）；④尾块 valid_n/掩码链语义不受循环界影响（迭代内逻辑零改动，TRUE）；⑤single_kv_block 特化与 trim 正交（kv_iters==1 时 eff==1，TRUE）
unknown_or_false: none
plan: direct-reuse
```

[#2] 行循环 → broadcast 2D 形式（§2.6 指令向量化 L381-400 + antipatterns「Vector Core 内逐元素/逐行 for loop」L124-150 + §10.3-③；先例 sink_bwd L178-207 同款 pass_configs 下已验证）：**适用（核心项）** — 消除 3 组 64×(GetValue+PIPE_ALL+微指令)（生成代码实证 §Step2-2）；aiv_scalar_time 49.6% 的主要构成；预期全部 case 每迭代 vector 侧大幅下降。buffer 复用方案照 sink_bwd：m_i_2d 复用 acc_s_ub_、m_i_prev_2d/sumexp_2d 复用 acc_o_ub（顺序活跃期不重叠）；新增 UB 仅 ~0（复用）+ row_thresh 相关（见 #3）。
  - 精度：逐元素同序 fp32 运算，行独立性保证位级等价（sub/mul/div 无跨行归约）。

[#3] 掩码链 2D 化（§2.6 同条 + §10.3-③ 掩码部分；先例 sink_bwd L177-196：broadcast col_pos→2D、broadcast row_pos→2D、compare(2D,2D)、select 2D VSEL_TENSOR_SCALAR_MODE）：**适用** — 现 64×(CompareScalar+Select+2 barrier)/iter → ~5 条 2D 指令；row_thresh 为 task 不变量可 hoist 出 KV 循环（arith_progression+min(S_kv-1)+broadcast 一次），每迭代仅剩 arith_progression(col_pos,j0)+broadcast+compare+select。仅 need_mask 变体受益（causal 全部 + 尾钳位）。
  - select 模式约束（§2.11 L576-605）：2D tile 4096 元素 > 64 → 必须保持 VSEL_TENSOR_TENSOR/SCALAR 系（现用 VSEL_TENSOR_SCALAR_MODE，sink_bwd 2D 先例同款）✅ 不踩 SPR 64 上限静默错误。
  - 新增 UB: row_thresh_2d + mask_2d 各 [bM,bN] fp32 = +32KB（D=128 时 137.5→~170KB < 192KB ✓；D=256/bM=32 +16KB ✓）；mask_row 1D buffer 可删。

[#4] 全可见迭代掩码链运行期跳过（§10.3-②；先例 mha_sink_fwd_bhsd.py L502 运行期条件包掩码）：**适用但延后（依赖 #3 后实测）** — #3 落地后掩码链成本从 ~128 条降到 ~5 条 2D 指令，跳过收益缩水；且需要 Developer cv_1_1 下运行期标量 `if` 包 vector 块的最小复现验证（仓内先例为 Expert 模式；DESIGN §10.2 仅验证过 parse-time if）。排在 #3 之后按实测数据决定。

[#5] Fixed Core / 静态任务分配（§2.9 L487-509 + antipatterns 关注项 A L62-90 + flash_attn §5 L298-351 + cube_path §4.1 L377-432）：**适用** — block_num≫20 核（case7 2048、case14 4096）；收益 = 每 block TPipe/workspace 初始化摊销 + workspace 足迹 114MB→~1.1MB（20 核×56KB，L2 常驻 → w1/w2/w3 往返延迟↓）；cube_path L385 实测节省 ~200us 固定开销/launch 规律支持 case14（slot 13.9us/block 中固定开销占比高）。先例：lightning_indexer.py L428-440（core_num=min(block_num, cube_core_num)，_get_cube_core_num host 元数据查询）+ mha_sink_fwd L236-252（任务循环+运行期 my_count）。
```
[KERNEL-REUSE-PRECHECK]
candidate: #5 Fixed Core（§2.9 + flash_attn §5 + cube_path §4.1）
expanded_domain: launch 网格从 block_num 变为 min(block_num,20)；kernel 体嵌入运行期任务循环（全部 shape 走新网格路径）
compatibility_dimensions: ①AUTO_CV_COMBINE 对「任务循环内嵌完整 CV 体」的分裂正确性（AIC/AIV 两侧同构循环、同 trip count → cross flag 配对平衡；mha_sink 以手动 Scope 实现同构，Developer 自动路径无仓内先例 = UNKNOWN，需最小复现验证）；②运行期任务界 T.serial(my_start, my_end) threads=1 Developer 先例（sink_bwd L164/L323-324 TRUE）；③任务间状态复位（fill acc_o/sumexp/m_i 移入任务循环，TRUE by edit）；④workspace 按 core cid 索引由框架自动派生（生成代码 workspace[cid*...]，cid 语义随 launch 网格变化，需 get_kernel_source 复核 = UNKNOWN→验证步骤覆盖）；⑤causal trim 负载均衡（连续 chunk 映射每核覆盖 ≥1 个完整 bx 周期时均衡；case7 load=103 vs s_blocks=16 → 6.4 周期 TRUE；s_blocks>load 的极端 shape 需退化保护=按 block_num launch，TRUE by host predicate）
unknown_or_false: ①④（实施时以最小复现 + get_kernel_source 验证，失败则回滚该项）
plan: repair-kernel-first（先在 case14/case7 验证 ①④，再全量）
```

[#6] decode 窄块 block_M=16（flash_attn §9(b) L568-591；§2.11 tile size 反推）：**适用（高价值）** — case10-18（9/20 cases）S=1/2，block_M=64 → 63/64 行空转；bM=16 使每迭代 vector 工作量/指令数 ~4x↓（4096→1024 元素 op、行循环 16 行）。kernel 已有运行期 valid_m 尾行语义 → **无需 host padding**（flash_attn §9(b) 的 host pad 变体违反本算子 host 约束，弃用；直接用编译期分支 `block_M = 16 if S <= 16 else ...`，写回仍裁剪 valid_m）。
```
[KERNEL-REUSE-PRECHECK]
candidate: #6 decode 窄块（flash_attn_optimize §9(b)）
expanded_domain: 新增编译期分支 block_M=16（S<=16 的 decode/MTP 全 shape 走窄块路径；既有 bM=64/32 路径保留为 S>16 fallback）
compatibility_dimensions: ①gemm_v0 M=16 支持性（cube fractal 最小 M=16；仓内 block_M=16 先例=antipatterns L277 示例值，需编译验证 UNKNOWN→最小复现）；②GEMM2 A 操作数 P[16,bN] 同上；③L0C [16,64]/[16,D] 分配合法（4KB/8KB，TRUE by 容量）；④reduce_max/reduce_sum 行独立语义（16 行与 64 行同 kernel 模板，TRUE）；⑤运行期 valid_m 切片/Q 尾行清零在 bM=16 下同构（TRUE，同 copy_gm_to_l1/ub 模板）；⑥掩码链/broadcast 2D 形式对 16 行成立（#2/#3 完成后同形，TRUE）；⑦精度：有效行（row 0..S-1）计算序列与 bM=64 逐位一致（行独立 + 同归约宽度，预期位级等价，l0/l1 decode 用例验证）
unknown_or_false: ①②（编译期即可验证，失败则放弃该项）
plan: repair-kernel-first（先编译冒烟 + l0/l1 decode 用例，再 msprof）
```

[#7] Fill+Add → Copy 冗余消除（antipatterns「基础指令拼接未融合」L198-263 之安全子集）：**适用（micro）** — `T.tile.fill(acc_s_ub,0)+T.tile.add(acc_s_ub,acc_s_ub,acc_s_ub_)` ≡ `T.copy(acc_s_ub_, acc_s_ub)`（fp32 0+x=x；±0 符号差异经 exp 路径不可观测，实施时在 RESULT 中以 l0+位级用例佐证）。省 1 条 4096 元素 Fill/iter。
  - axpy 融合（§2.7 L402-426）：**拒绝** — `axpy(acc_s_ub, acc_s_ub_, sm_scale)` 会把 ×s 提前到 round16(qk) 之前，破坏双舍入栅格（DESIGN §10.3-⑤ 精度红线；§2.7 亦要求融合前用户确认）。

[#8] block_N=128 tile 扩容（antipatterns「tile size 过小」L267-299 + §2.11 UB 预算 L539-710）：**候选（最后实施）** — 迭代数减半 → CV 往返/barrier 链/掩码链次数减半；UB 预算：D=128 时 acc_s 系翻倍（16→32KB×2 + half 16KB）+ acc_o 系不变 80KB + tmp 16KB ≈ 176KB + #3 新增 32KB = **208KB > 192KB 超限** → 需与 #3 buffer 复用合并重算（mask_2d/row_thresh_2d 复用 acc_s_ub_ 等）或仅对 D=64 启用；§2.11 L630-632 明令 UB 预算类最后实施。依据实测数据决定。

[#9] KV 预取恢复：手动双缓冲 + T.serial（§10.3-⑥a + guide §2.2 L92-291 + §2.3 L293-321 + antipatterns「CV overlap 未开启」L303-337）：**适用但列为最后手段** — §2.2.0 决策表 L115：CV 融合 Double Buffer **必须 AUTO_SYNC=False + 手动 mte3→mte2/mte2→v/v→mte3 flag + cross flag** → 越出 Developer 模式边界（DESIGN §7.1 零手动同步 / §8.6 自检 grep），任务约束要求 log 说明后才可引入；预期收益有限：aic_mte2 仅 9.08us/162.4（K load ~0.57us/iter 占 10.45us 临界路径 ~5%），且 cube 98.5% 在等 vector——预取只对 K/V load 段有效。若 #1-#7 后仍未达标且迭代预算允许，实施时以 [DB-ANALYSIS] 前置。
  - T.Pipelined 路线（§3.1 L1623-1634 + antipatterns L317-330）：**禁用**（§10.1-R9 codegen bug，无新框架修复证据）。
  - Swizzle（cube_path §4.2 L434-488）：**不采用**（文档实测劣化，L483 默认不采用）。
  - kL0Size/K_L1 调优（cube_path §3.2-3.4 L313-371）：**不适用** — 框架级 common.h 改动影响全仓算子 + cube 利用率 2% 非瓶颈。

### 不适用清单（读完适用场景/约束后排除）

- §2.1 Split-K（L63-90）：GEMM K=D≤256 单次 L1 装下，无 K 维循环；cube 2% 非瓶颈。
- §2.4 Full-Load（L323-351）：Q 已一次性驻留 L1（mha.py L114）；K/V 是迭代主数据非重复小侧。
- §2.5 小块合并载入（L353-379）：无 scale/bias 随路小块。
- §2.8 稀疏访存（L428-485）：无页表/gather。
- §2.10 pass_configs 调优（L511-535）：文档定位「最后手段」；AUTO_SYNC=False 即 #9 的伴随修改，非独立项。
- §2.13 多行 Tile（L921-1150）：本 kernel 已是 [bM,bN] 2D tile；其 Layer3 broadcast/归约铁律已并入 #2/#3 执行细则；MIX vid==0 guard 不适用（threads=1 无 vid 轴）。
- §2.14 Host D2H 消除（L1152-1228）：输出直接返回 NPU tensor，无 host 后处理。
- §2.15 正交轴向量化（L1229-1412）+ antipatterns 正交轴串行化（L420-457）：无带前缀依赖的标量扫描轴；行循环问题归 #2（循环内已是 tile op）。
- §2.16 dtype 硬件指令适配（L1413-1473）：fp16/bf16 已走原生 half GEMM/向量路径，无 int8/fp8 域。
- §2.17 离散重排多阶段连续化（L1474-1615）：无 planner/stage kernel 结构。
- §3.2 核间同步 cross_interval（L1636-1664）：需手动 cross flag（Developer 边界）；AUTO_CV_SYNC 管理核间事件；仅随 #9 一并考虑。
- antipatterns「冗余全局同步」（L154-194）：源码零手动 barrier；生成代码的逐 op PipeBarrier 来自 AUTO_SYNC pass，消除 = AUTO_SYNC=False（归 #9/§2.10，最后手段）。**有意保留原因：Developer 模式边界（DESIGN §7.1/§8.6 自检项）+ 精度/稳定性优先；待 #1-#7 实测后若同步仍为主瓶颈再按 #9 评估。**
- antipatterns「多行 tile 循环内逐块归约」（L384-416）：每 KV 块 reduce_max/reduce_sum(dim=-1) 是 online-softmax 数学必需（跨块 max/sumexp 状态滚动），非可外提归约；无 dim=0 用法。
- antipatterns「纯 AIV memory bound 未流水」（L340-381）：混合型不适用。
- antipatterns「纯 Vector AUTO_CV_COMBINE 误分核」（L461-484）：非纯 Vector、无 alloc_var。
- flash_attn §1/§2/§3/§6/§7/§10（L38-249, L355-497, L647-674 多缓冲批量流水/细粒度 flag/MMA intrinsic/批量 softmax/workspace 结构/关全 pass）：Expert 模式专属技术，越出本算子 Developer 边界（DESIGN §8.6 强制：源码不得出现 T.Scope/手动 flag；任务硬约束）；其中 workspace 按核分配的思想由 #5 Fixed Core 在 Developer 框架内等价获得。
- flash_attn §4 ZN/NZ layout（L252-295）：annotate_layout 服务 mma intrinsic 路径；gemm_v0 内部已处理分形化；cube 2% 非瓶颈。
- flash_attn §9(a) BSND 免转置（L552-566）：已实现（BSHD 行跨步 DMA，host 零转置）。
- flash_attn §9(c) 加性 mask（L593-629）：等价物已实现（-inf select 掩码链 + 运行期 valid_n 清零三重保障）。
- vector_reduce_pass_fusion.md（L11-41 适用条件）：C1 纯 Vector 不满足（混合型）；X1 命中（归约维 block_N=64 列一次装入 UB，无分块多遍扫描）；online-softmax 本身已是 max+sum 融合形态。**不适用**。
- §10.3-④ D=256 bM=64 精简 buffer（DESIGN L1336）：bM 32→64 使 vector 行工作量翻倍而任务数减半，vector-bound 下无净收益且 UB 需全面重算；D=256 case（4/9/15/18）的收益更应来自 #2/#3/#6。**暂缓（记录）**。
- §10.3-⑤ golden 舍入对齐简化（DESIGN L1337-1339）：精度红线（rev2 警示：双舍入镜像是 case4 位级精确与多块达理想下界的来源）。**禁止**。

## Part B [ORDER-PLAN]

依赖分析（三条规则）：
- 布局依赖：#3（掩码 2D 化，引入 row_thresh_2d/mask_2d 并确立 2D broadcast 形态）是 #4（运行期跳过包裹的对象）的前置；#2/#3 确立的 broadcast 形态是 #6（窄块 16 行同形缩放）的前置。
- 数量依赖（UB 预算）：#8（block_N=128）必须在 #2/#3（新增/复用 2D buffer 定形）与 #6（decode 分支 buffer 定形）之后做 UB 总账（§2.11 L630-632 明令最后）；#5 不改 UB 但改 workspace 足迹，与 #8 无冲突、与 #1 有负载均衡交互（连续 chunk × bx 周期性，见 PRECHECK ⑤），排 #1 后。
- 配置依赖：#9（pass_configs AUTO_SYNC=False 伴随修改）必须最后（§2.10 L513 最后手段 + Developer 边界评估）。

```
[ORDER-PLAN] 实施顺序：
1. [#1] causal KV 运行期裁剪 — 前置依赖: [无] — 理由: 单项期望最大（case7 ~1.9x）、编辑面最小（循环界 2 行）、先例公式逐字可用、位级等价可证
2. [#2] softmax/rescale 行循环 broadcast 2D 化 — 前置依赖: [无，与 #1 正交；排后因编辑面大] — 理由: 打击 aiv_scalar 49.6% 主构成，全 case 受益；sink_bwd 同款 pass_configs 先例
3. [#3] 掩码链 2D 化 + row_thresh hoist — 前置依赖: [#2（共享 broadcast buffer 复用格局，先定 softmax 侧再定掩码侧，避免同轮改两处）] — 理由: need_mask 变体每迭代 ~128 条微指令 → ~5 条
4. [#4] 全可见迭代掩码运行期跳过 — 前置依赖: [#3（跳过对象为 2D 化后掩码链）] — 理由: 收益依 #3 后实测决定；需先做 Developer 运行期 if 最小复现
5. [#6] decode 窄块 block_M=16（S<=16 编译期分支） — 前置依赖: [#2/#3（16 行复用同形 broadcast/mask 代码路径）] — 理由: case10-18（9/20）~3-4x 空转消除；编译期分支不动既有路径
6. [#5] Fixed Core（core_num=min(block_num,20) 连续 chunk 任务循环） — 前置依赖: [#1（负载均衡交互）；排 #6 后因验证面最大（AUTO_CV_COMBINE×嵌套运行期循环 UNKNOWN 维度）] — 理由: case14 ~30% / case7 workspace L2 化
7. [#7] Fill+Add→Copy（micro） — 前置依赖: [无；顺位填充] — 理由: 独立小收益，任意点可插入，按序执行
8. [#8] block_N=128 UB 预算扩容 — 前置依赖: [#2/#3/#6（buffer 格局定形后总账）] — 理由: §2.11 明令 UB 预算最后；超限风险最高
9. [#9] 手动双缓冲 K/V 预取（AUTO_SYNC=False + 手动 flag，越界说明 + [DB-ANALYSIS] 前置） — 前置依赖: [全部前项完成且仍有预算] — 理由: 最后手段；预期 ~5%；Developer 边界代价最高
```

中止规则（任务指令）：连续 3 项无提升（<3%）或 10 轮上限；每项完成即记录 [RESULT-#N]。

## Step 4 逐项实施记录

（以下按 [ORDER-CHECK] → [IMPL-#N] → [SELF-CHECK] → [RESULT-#N] 门禁逐项追加）

### 迭代 1 — [#1] causal KV 运行期裁剪（2026-09-18T19:45+08:00）

```
[ORDER-CHECK] 准备实施: [#1] causal KV 运行期裁剪 | 前置依赖: [无 ✅] | 结论: ✅（ORDER-PLAN 第 1 位）
[IMPL-#1] 已阅读 mha_sink_fwd_bhsd.py L263-271（eff_kv_iters = floordiv(min(bx*bM+bM-1+offset, S_kv-1), bN)+1，右下角对齐语义注释）+ example_gqa_sink_bwd_bhsd.py L164/L323-324（threads=1 Developer 运行期 T.serial(st,ed) 先例）+ DESIGN §10.3-① L1331-1333。关键约束: ①全掩码块跳过=值等价（exp(-inf)=0, r=exp(m-m)=1, 0@V=0，状态转移恒等；±0 符号角例数值相等，harness abs 口径不可观测）；②eff>=1 恒成立（causal 时 host 断言 S_kv>=S）；③非 causal 保持编译期常量界（Python ternary parse-time 选择，零影响）；④尾块 valid_n/掩码链迭代内逻辑零改动。
[KERNEL-REUSE-AUDIT] 依据: Part A [KERNEL-REUSE-PRECHECK] #1 | 维度及结论: ①运行期界 codegen=TRUE(sink_bwd 同构先例+本次 l0/l1/msprof 实证) ②trim 公式=TRUE(mha_sink 逐字) ③eff>=1=TRUE ④尾块语义=TRUE(l1 TAIL-1/TAIL-MID/PRIME causal 用例全过) ⑤single_kv_block 正交=TRUE(l0 case1-4/10 单块用例全过) | 结论: reuse（direct-reuse 成立，无 unknown 残留）
[SELF-CHECK] 本次 Edit 只涉及 [#1]（mha.py 两处：eff_kv_iters 推导 + T.serial(0, eff_kv_iters) 循环界；备份 history_version/mha_impl_s3_perf1.py）
[RESULT-#1] 优化点: causal KV 运行期裁剪 | 精度: pass（l0 10/10 + l1 19/19，/tmp/opencode/l0_iter1.log、l1_iter1.log）| 性能: case7 8960.55us / case1 45.46us / case10 442.09us / case14 2843.61us / case20 101.70us | 对比: case7 **-46.6%（1.87x，预期 ~1.9x 达成）**；case1 -2.4%（真实但低于单 case 3% 阈：wall 由 2-iter 的 bx=1 块+固定开销主导，1 iter 块不在临界路径）；case10/14/20（非 causal）±0.4% 内 = 无回归 | 采纳: YES（新基线 = iter1）
```

msprof 证据: examples/mha/perf_tuning/msprof_iter1/*.json（median: case7 8960.55 stdev 1.14；case1 45.46 stdev 0.17）

### 迭代 2 — [#2] softmax/rescale 行循环 broadcast 2D 化（2026-09-18T20:10+08:00）

```
[ORDER-CHECK] 准备实施: [#2] 行循环 broadcast 2D 化 | 前置依赖: [无（与 #1 正交，#1 已采纳 ✅）] | 结论: ✅（ORDER-PLAN 第 2 位）
[IMPL-#2] 已阅读 optimization-guide.md L381-400（§2.6 指令向量化：for h_i 行循环 → broadcast+2D tile op）+ performance-antipatterns.md L124-150（Vector 逐行 for loop 条目，收益/代价/适用性）+ sink_bwd L198-207（broadcast(m_i)→sub、broadcast(m_i_prev)→mul 先例）。关键约束: ①broadcast dst/src 同 dtype 且在 UB（ascend_tile.py L2353-2383 API 契约）；②1D→2D 禁止 T.copy（guide 错误速查表 L1689 StructuralEqual），必须 T.tile.broadcast；③逐元素同序 fp32 运算，行独立 → 位级等价预期；④rescale 移位到 L0C copy 之前（sink_bwd 顺序）。
[SELF-CHECK] 本次 Edit 只涉及 [#2]（4 处行循环→broadcast + 专用 scratch buffer 声明；备份 history_version/mha_impl_s3_perf2.py = post-#1 状态）
[RESULT-#2] 优化点: 行循环 broadcast 2D 化 | 精度: pass（首版 FAIL 5/10 → 修复后 l0 10/10 + l1 19/19，/tmp/opencode/l0_iter2b.log、l1_iter2b.log）| 性能: case7 6062.44us / case1 30.09us / case10 217.19us / case14 1665.76us / case20 57.97us | 对比: vs iter1 基线 case7 -32.3%、case1 -33.8%、case10 -50.9%、case14 -41.4%、case20 -43.0%；vs 原始基线累计 case7 -63.9%（2.77x）、case1 -35.4%、case10 -50.8%（2.03x）、case14 -41.4%、case20 -43.2% | 采纳: YES（新基线 = iter2）
```

**精度失败→修复记录（skill Step E，不撤销调试）**：
- 首版实现按 sink_bwd 复用 acc_s_ub_/acc_o_ub 作 broadcast scratch → l0 5/10 FAIL（全部多块 case，matched 低至 0.005；单块 case 全过——单块下 acc_o=0 使错误序巧合正确）。
- 根因（get_kernel_source 证据，/tmp/opencode/mha_src_iter2_case5.cc AIV 段）：AUTO 同步 pass 按 **buffer 依赖** 把 flag 锚定的跨核 `copy_gm_to_ub(acc_o_ub←w3)` 排到 `Mul(acc_o,acc_o,acc_o_ub)` 之前 → mul 读到的是 PV 分片而非 r 因子（acc_o = acc_o*partial + partial）。**教训：broadcast scratch 不得与跨核 L0C→UB copy 的目标 buffer 复用**（sink_bwd 的复用安全是因为其 workspace copy 是普通 GM 张量 T.copy，非 flag 锚定跨核事件；该差异在 Part A 初判时未识别）。
- 修复：新增专用 scratch `m_i_2d [bM,bN] fp32`、`r_2d [bM,D] fp32`（4 处 broadcast 全部改用；acc_s_ub_ 一并弃用 scratch 角色以消除对迭代顶部 Fill+PipeBarrier 巧合 WAR 保护的依赖）。UB 峰值 191.8KB < 192KB（MEMORY_PLANNING 复用死亡 buffer，D=128/256 实测编译通过）。
- msprof 证据: examples/mha/perf_tuning/msprof_iter2/*.json（stdev 均 <0.2%）

### 迭代 3 — [#7] Fill+Add→Copy（提前实施：#3 的 UB 预算使能项）

```
[ORDER-CHECK] 准备实施: [#7] Fill+Add→Copy（原计划第 7 位）| 前置依赖: [无 ✅] | 结论: ✅ 重排序理由: ORDER-PLAN 数量依赖规则（预算类优化排在改变 buffer 数量的优化之后）——#3 掩码 2D 化需要 16KB rt_2d 余量，而当前 UB 峰值 191.75/192KB 满载；#7 删除 acc_s_ub_（L0C→UB copy 直接落 acc_s_ub）释放 16KB，是 #3 的使能前置。#7 本身亦是 antipatterns L198-263 冗余指令消除项。
[IMPL-#7] 已阅读 performance-antipatterns.md L198-263（基础指令拼接：fill(0)+add(x)≡copy(x)）+ mha.py L134-136 现状。关键约束: ①0+x=x 位级等价；±0 符号角例（0+(-0)=+0 vs copy 保留 -0）——下游 first observable = exp(±0-m) 同值，不可观测；②T.copy(acc_s_l0c, acc_s_ub) 跨核 L0C→UB 路由与 T.copy(acc_s_l0c, acc_s_ub_) 同模板（仅目标 buffer 变化）；③acc_s_ub 不得再被 V 侧先写（fill 删除后无冲突）；④#2 教训检查：acc_s_ub 非任何 broadcast scratch（m_i_2d/r_2d 专用）✓。
[SELF-CHECK] 本次 Edit 只涉及 [#7]（删除 acc_s_ub_ 分配；fill+add → T.copy(acc_s_l0c, acc_s_ub) 直落；备份 history_version/mha_impl_s3_perf3.py = post-#2 状态）
[RESULT-#7] 优化点: Fill+Add→Copy（acc_s_ub_ 退役）| 精度: pass（l0 10/10，/tmp/opencode/l0_iter3.log）| 性能: case7 6058.97us / case20 58.32us / case10 216.48us | 对比: vs iter2 基线 -0.06% / +0.6% / -0.3% = 全部噪声内（<3%），单独无性能收益 | 采纳: YES-as-enabler —— 本项为 ORDER-PLAN 数量依赖规则下的 #3 使能前置（释放 16KB UB；#3 的 rt_2d 需要该余量），并净删每迭代 2 条 4096 元素指令（Fill+Add→Copy 合并）；#7+#3 联合收益在 [RESULT-#3] 复核，若联合无收益则两项一并回滚到 perf3 备份。
```

### 迭代 3b — [#3] 掩码链 2D 化 + row_thresh hoist（2026-09-18T20:45+08:00）

```
[ORDER-CHECK] 准备实施: [#3] 掩码链 2D 化 | 前置依赖: [#2 ✅（broadcast 形态与 scratch 格局已定）；#7 ✅（UB 余量已释放）] | 结论: ✅
[IMPL-#3] 已阅读 sink_bwd L177-196（arith_progression + broadcast axis=0/1 + compare(2D,2D,"LE") + select 2D 先例，同 pass_configs Developer threads=1）+ HISA block_sparse_mqa_attn_expert_test.py L113-114/510-516（compare/select 的 uint8 bit-packed mask 先例 [n//8]）+ optimization-guide §2.11 L576-605（select 模式约束：2D 4096 元素必须非 SPR 模式——保持 VSEL_TENSOR_SCALAR_MODE）+ §2.13 错误表 L1689（1D→2D 必须 broadcast 不得 T.copy）。关键约束: ①右下角对齐语义 = col ≤ min(row+offset_eff, S_kv-1)，row_thresh 任务不变量 hoist（非 causal offset_eff=2^30 → min 坍缩为 S_kv-1，fp32 表示 2^30 精确、S_kv≤4096 ≪ 2^30 无误判）；②col_2d 复用 m_i_2d（纯 V scratch，顺序活跃期不重叠，不触跨核 copy 目标——#2 教训）；③mask scratch 不与任何跨核 copy 目标别名；④UB 预算: +rt_2d 16KB + mask_2d 512B + row_thresh 256B − mask_row 256B，实测 InitBuffer 191.8KB ≤ 192KB（D=64/128/256 三配置编译验证）；⑤bit-packed mask 尺寸 bM*bN//8（HISA 逐字先例）。
[SELF-CHECK] 本次 Edit 只涉及 [#3]（buffer 声明块 + hoist 块 + 掩码链替换；备份 history_version/mha_impl_s3_perf3.py 覆盖 #7+#3 联合回滚点）
[RESULT-#3] 优化点: 掩码链 2D 化（#7+#3 联合）| 精度: pass（l0 10/10 + l1 19/19，/tmp/opencode/l0_iter3b.log、l1_iter3b.log；l1 覆盖 (T,F)/(F,T) 尾钳位×causal 全变体）| 性能: case7 3554.73us / case1 18.29us / case10 217.06us / case14 1668.50us / case20 58.50us | 对比: vs iter2 case7 -41.3%、case1 -39.2%、非 causal 三项 ±0.3%（need_mask=False 路径零改动，无回归）；vs 原始基线累计 case7 -78.8%（4.72x）、case1 -60.8%（2.55x）、case10 -50.8%（2.03x）、case14 -41.3%（1.70x）、case20 -42.7%（1.74x）| 采纳: YES（#7+#3 联合收益确认，新基线 = iter3）
```

生成代码证据（/tmp/opencode/mha_src_iter3b.cc）：Compare×1（2D）+ Select×1 + Broadcast×5 + ArithProgression×2，64 行循环 = 0；trim 编译为 `k < (cid%16)+1`（AIC 侧运行期界）。
msprof 证据: examples/mha/perf_tuning/msprof_iter3/*_iter3b.json

### 迭代 4 — [#6] decode 窄块 block_M=16（2026-09-18T21:00+08:00）

```
[ORDER-CHECK] 准备实施: [#6] decode 窄块 | 前置依赖: [#2 ✅ #3 ✅（16 行复用同形 broadcast/mask 路径）] | 结论: ✅（重排序说明：#4 运行期掩码跳过在 #3 后期望值缩水至 ~5-8% 且需先验证 Developer 运行期 if，#6 期望值 ~2-3x 于 decode 9/20 cases 且相互独立 → #6 先行，#4 顺延再评估）
[IMPL-#6] 已阅读 flash_attn_optimize.md L568-591（§9(b) Sq==1 decode 窄块：block_m=16，16=向量/UB 对齐最小折中；host pad 变体因本算子 host 零张量操作约束弃用——kernel 已有运行期 valid_m 尾行语义，无需 padding）+ antipatterns L267-299（tile size 与任务数联动检查）。关键约束: ①分支仅限 S<=16（更大 S 下窄块使 s_blocks×4 → K/V 重读与任务开销×4，prefill 反收益——rows 是真实工作非 padding）；②M=16 = cube fractal 最小；③有效行计算序列与 bM=64 位级一致（行独立 + 同归约宽度）。
[KERNEL-REUSE-AUDIT] 依据: Part A [KERNEL-REUSE-PRECHECK] #6 | 维度及结论: ①gemm_v0 M=16 编译=TRUE（probe_decode_narrow.py 12 shape 全编译+运行）②GEMM2 A=[16,bN]=TRUE（同 probe）③L0C 容量=TRUE ④valid_m 切片同构=TRUE ⑤broadcast/mask 2D 对 16 行=TRUE ⑥单块特化正交=TRUE（l1_edge_Skv1 等过）⑦位级等价=TRUE（probe 13/13 BITEXACT 含 S=17 分支边界）| 结论: reuse（unknown 全部清零）
[SELF-CHECK] 本次 Edit 只涉及 [#6]（block_M 派生 1 行 + 注释；备份 history_version/mha_impl_s3_perf4.py = post-#3 状态）
[RESULT-#6] 优化点: decode 窄块 block_M=16 | 精度: pass（l0 10/10 + l1 19/19 + 等价 probe 13/13 BITEXACT，/tmp/opencode/l0_iter4.log、l1_iter4.log、probe_narrow.log）| 性能: case10 148.41us / case14 1242.86us / case16 2234.72us（首测，cann-bench 原值 7840）/ case7 3554.71us / case1 18.50us / case20 58.18us | 对比: case10 -31.6%（累计 2.97x）、case14 -25.5%（累计 2.29x）、prefill 三项 ±1.1% 内零回归；未达理论 4x 因每迭代固定成本（CV 往返延迟/同步）不随 bM 缩小 | 采纳: YES（新基线 = iter4）
```

### 迭代 5 — [#5] Fixed Core（core_num=min(block_num,20) + 连续 chunk 任务循环）（2026-09-18T21:10+08:00）

```
[ORDER-CHECK] 准备实施: [#5] Fixed Core | 前置依赖: [#1 ✅（负载均衡交互：连续 chunk × bx 最快分解 → 每核任务段覆盖完整 bx 周期，causal trim 下均衡）] | 结论: ✅（ORDER-PLAN 第 6 位；#4 顺延至 #5 后按剩余预算评估）
[IMPL-#5] 已阅读 optimization-guide §2.9 L487-509（T.Kernel(core_num) + T.serial(cid*load,(cid+1)*load) + workspace[cid] 复用）+ antipatterns 关注项 A L62-90（尾块 block_idx<block_num 守卫/负载映射）+ flash_attn §5 L298-351（q_tasks/r_tasks 均衡分配）+ cube_path §4.1 L377-432（FC 节省 ~200us/launch 固定开销规律；Developer+2.0% 净收益实证）+ lightning_indexer.py L14-21/L428-440（_get_cube_core_num host 元数据查询 + core_num=min(block_num,max_cores) 仓内先例）。关键约束: ①连续 chunk 用运行期 T.serial(my_start,my_end) 双界（sink_bwd L164/L323-324 threads=1 Developer 先例）→ 空区间核零迭代，无需运行期 if 守卫（守卫形式在 Developer 未验证，规避）；②buffer 分配全部留在任务循环外（TPipe 一次初始化 = FC 收益本体）；③任务级状态（fills/row_thresh/Q load/写回）移入任务循环；④workspace 由框架按 launch 网格自动派生 → 20 核 × 56KB 槽位 L2 常驻（case7 足迹 114MB→1.1MB）；⑤跨任务 w1/w2/w3 WAR 由既有 cross-flag 链传递保护（AIC 下任务 w1 dump 在 GEMM2 之后 ← CrossCoreWaitFlag(1) ← AIV P 写 ← AIV w1 读，链式推导见分析）；⑥core_num host 元数据查询（device properties，非张量数据搬运，§4.8 允许；lightning_indexer 先例）。
[KERNEL-REUSE-AUDIT] 依据: Part A [KERNEL-REUSE-PRECHECK] #5 | 维度及结论: ①AUTO_CV_COMBINE×嵌套运行期任务循环=UNKNOWN→本次以编译+等价 probe+l0/l1 验证（失败即回滚 perf5）②运行期任务界=TRUE（sink_bwd 先例）③任务间状态复位=TRUE by edit（fills/row_thresh hoist 移入循环）④workspace cid 索引=UNKNOWN→get_kernel_source 复核 ⑤causal trim 负载均衡=TRUE（连续 chunk 覆盖 ≥1 完整 bx 周期：case7 load=103 vs s_blocks=16；s_blocks>load 极端 shape 由 core_num=min(block_num,cores) 退化保护）| 结论: repair-kernel-first（①④验证通过才采纳）
[SELF-CHECK] 本次 Edit 只涉及 [#5]（_cube_core_num helper + core_num/q_tasks/r_tasks 派生 + T.Kernel(core_num) + 任务循环包裹全体 per-task 逻辑；buffer 分配留在循环外；备份 history_version/mha_impl_s3_perf5.py = post-#6 状态）
[RESULT-#5] 优化点: Fixed Core（core_num=min(block_num,20)，q_tasks/r_tasks 均衡分配）| 精度: pass（等价 probe 12/12 BITEXACT 含 block_num=1/6/30/32/64/128/256/2048/4096 边界 + l0 10/10 + l1 19/19；/tmp/opencode/probe_fc2.log、l0_iter5.log、l1_iter5.log）| 性能: case7 3615.87 / case1 15.35 / case10 151.46 / case14 1009.90 / case20 54.14 / case16 2146.01 (us) | 对比: vs iter4 case14 **-18.7%**、case1 **-17.0%**、case20 **-6.9%**、case16 -4.0%、case7 +1.7%、case10 +2.1%（后两者在 3% 噪声带内：长任务下 FC 循环标量开销为正且硬件块调度本已无缝隙——iter4 profile 实证 slot==aiv_time）；5-case geomean 加速比 vs CANN baseline 0.355→0.387（**+9%**）| 采纳: YES（净收益显著为正；混合行为如实记录；双路径启发式（按任务时长分流 FC/非 FC）评估后放弃——1.7~2.1% 的噪声带内回退不值得双倍 dispatch 域与验证面）
```

**实施期问题与修复（skill Step E）**：
- 首版用连续 chunk `T.serial(my_start, my_end)`（双 T.min 运行期界）→ block_num=6 退化 shape 编译崩溃：`InternalError: Trying to update var 'task' with a different maximum value: original=5, new=T.min(T.min(T.min(T.min(cid,5),5),5),5)`——AUTO_CV_COMBINE 将运行期界任务循环复制到 AIC/AIV 两侧时，循环变量 range 注解两次推导不深度相等（TIR 分析器对 start/end 双运行期界 + 循环变量直接作 task 的形式不稳健）。
- 修复：改用 count 形式 `for t in T.serial(my_count): task = my_start + t`（mha_sink_fwd:236-238 q_tasks/r_tasks 先例；t 从 0 起、单表达式界，两侧复制推导一致；my_count 恒 ≥0 无反向区间）。probe 12/12 通过。
- PRECHECK UNKNOWN 维度落定：①AUTO_CV_COMBINE×嵌套运行期任务循环 = TRUE（count 形式；start/end 双运行期界形式 = FALSE 已弃用）；④workspace cid 索引 = TRUE（生成代码 workspace[cid*4096]，cid∈[0,20) → 足迹 case7 114MB→1.1MB L2 常驻；AIC/AIV 两侧任务循环同界配对 ✓ /tmp/opencode/mha_src_iter5_case7.cc）。

### 迭代 6 — [#4] 全可见迭代掩码运行期跳过（2026-09-18T21:40+08:00）

```
[ORDER-CHECK] 准备实施: [#4] 掩码运行期跳过 | 前置依赖: [#3 ✅（跳过对象 = 2D 掩码链）；#1 ✅（trim 后需掩码迭代 = 每 s_block 恰 1 个对角块 + 可能的尾块）] | 结论: ✅
[IMPL-#4] 已阅读 mha_sink_fwd_bhsd.py L502-514（运行期条件包掩码先例，Expert 模式）+ Part A #4 条目（Developer cv_1_1 运行期标量 if 包 V 块无仓内直接先例 → 以编译+生成代码审查+probe 验证，失败即回滚）。关键约束: ①条件 = j0+bN-1 > min(s0+offset_eff, S_kv-1)（纯标量：row_thresh 对 r 单调不减 → 块全可见 ⟺ 末列 ≤ row_thresh[0]；无 UB 标量读，不引入 GetValue）；②非 causal 时 offset_eff=2^30 → 条件坍缩为"仅尾块"（S_kv%64≠0 变体）；causal trim 下仅对角块命中；③语义等价 = 全可见块上 compare 全 true → select 恒等，跳过为位级等价；④parse-time if(need_mask) 内嵌 runtime if 的两层结构。
[SELF-CHECK] 本次 Edit 只涉及 [#4]（掩码链外包一层运行期条件；备份 history_version/mha_impl_s3_perf6.py = post-#5）
[RESULT-#4] 优化点: 全可见迭代掩码运行期跳过 | 精度: pass（等价 probe 12/12 BITEXACT（vs perf5，联合验证 FC+#4）+ l0 10/10 + l1 19/19——l1 覆盖 (T,F)×尾钳位×trim 全交互；/tmp/opencode/probe_i6.log、l0_iter6.log、l1_iter6.log）| 性能: case7 3177.05us / case1 14.80us / case16 1950.10us / case20 54.32us / case14 1011.28us | 对比: vs iter5 case7 **-12.1%**、case16 **-9.1%**、case1 -3.6%；case20/case14 ±0.3%（need_mask=False 编译期零改动，符合预期）；case10 未测（同为非 causal 无掩码路径，结构性零变化）| 采纳: YES（新基线 = iter6）
```

生成代码证据（/tmp/opencode/mha_src_iter6_case7.cc）：`if ((((task%16))*64) < ((k_1*64)+63)) { AP+Broadcast×2+Compare(4096)+Select<float,uint8_t> }` —— Developer cv_1_1 下运行期标量 if 包 V 块正确生成（Part A #4 的 UNKNOWN 维度落定为 TRUE；掩码仅对角块命中，与 trim 协同后 case7 掩码应用次数 136→16）。

### 迭代 7 — [#8d] decode 分支 block_N=128（#8 的窄块限定安全变体）（2026-09-18T22:10+08:00）

```
[ORDER-CHECK] 准备实施: [#8d] block_N=128（仅 S<=16 窄块分支）| 前置依赖: [#6 ✅（窄块分支已存在）；#2/#3 ✅（broadcast/mask 2D 形态对 bN 参数化）；§2.11 L630-632 UB 预算类最后实施 ✅（其余 kernel 级优化已落地）] | 结论: ✅
重排序/变形说明: 原 #8（全局 bN=128）经 UB 总账否决——prefill 形态（bM=64,D=128）需 ~242KB > 192KB（含 r_2d/rt_2d 复用仍 216KB）。iter6 重 profile 证据（case10: aiv_scalar 1.03us/iter 为最大分项、指令数不随 bM=16 缩小；case7: 剩余为结构性 AUTO_SYNC 同步成本）表明 bN=128 的正确作用域是"迭代数减半"的 decode 分支：bM=16 下 UB 仅 ~92KB（D=128）/~120KB（D=256），L0C 24KB、L1 140KB 全在限内。prefill 路径保持 bN=64 零改动。
[IMPL-#8d] 已阅读 antipatterns L267-299（tile size 过小条目：迭代数/同步次数/GM 往返随 tile 减半）+ §2.11 L539-632（UB 预算反推 + 实施顺序）+ iter6 op-mode profile（/tmp/opencode/op_i6_case10）。关键约束: ①kv_iters 随 bN 变化 → single_kv_block 谓词边界移动（S_kv<=128 的 decode 落入单块特化 = 更高精度路径，数学前提"sumexp 在唯一迭代内终值"对任意 bN 成立）；②S_kv%128!=0 且 %64==0 的 shape 掩码变体 (F,F)->(F,T)（尾钳位激活，语义同一：列钳到 S_kv-1，#4 跳过保证仅尾迭代执行）；③GEMM1 N=128/GEMM2 K=128 fractal 合法（N/K 为 16 倍数）；④mask_2d = bM*bN//8 = 256B；⑤valid_n 尾列语义对 bN=128 同构。

[KERNEL-REUSE-PRECHECK]
candidate: #8d decode 分支 bN=128（antipatterns tile-size 条目 + §2.11）
expanded_domain: S<=16 分支内 kv_iters/need_bound_mask/single_kv_block 三个编译期谓词的取值边界移动；prefill 分支零变化
compatibility_dimensions: ①gemm_v0 N=128(GEMM1)/K=128(GEMM2) 编译运行=UNKNOWN→编译冒烟验证；②UB/L0C/L1 容量=TRUE（92-140KB 手算，编译 InitBuffer 复核）；③单块特化在 bN=128 的位级语义=TRUE by 数学（P̂ 公式与 bN 无关）+l1_edge_Skv1/probe 验证；④(F,F)->(F,T) 变体迁移正确性=TRUE by 构造（阈值 min(...,S_kv-1) 同一）+l1 尾钳位用例验证；⑤decode 有效行位级等价=FALSE 预期（归约宽度 64->128 改变 online-softmax 中间舍入序列——非位级，须以 l0/l1 阻塞口径 + MARE_abs 余量验证）
unknown_or_false: ①（编译验证）、⑤（预期非位级，按阻塞口径判定）
plan: repair-kernel-first（编译冒烟 → probe 数值容差比对 → l0/l1 → msprof，任一失败回滚 perf7）
```
[SELF-CHECK] 本次 Edit 只涉及 [#8d]（block_N 派生 1 行 + 注释；备份 history_version/mha_impl_s3_perf7.py = post-#4）
[RESULT-#8d] 优化点: decode 分支 block_N=128（迭代数减半）| 精度: pass（probe_bN128: prefill 4/4 BITEXACT + decode 8/8 阻塞口径过、MARE_abs 3.05e-5~1.22e-4 对 gate 9.77e-3 余量 80-320x、|Δvs perf7|≤1.2e-4（1-2 ulp，预期舍入序列变化）；l0 10/10 + l1 19/19；/tmp/opencode/probe_bN128.log、l0_iter7.log、l1_iter7.log）| 性能: case10 92.92us / case14 667.15us / case16 1247.04us / case7 3208.81us / case1 15.04us / case20 54.76us | 对比: vs iter6 case10 **-38.7%**、case14 **-34.0%**、case16 **-36.1%**；prefill +0.8~1.6%（BITEXACT 证实路径未变 = 会话间噪声）| UB/L1/L0C: 5 组 decode 配置编译验证全在限内 | 采纳: YES（新基线 = iter7）
```

累计（vs 原始基线）：case1 3.10x / case7 5.23x / case10 4.75x / case14 4.26x / case20 1.86x / case16 6.29x（vs cann-bench T_cand）；vs CANN baseline 加速比：case1 0.66x / case7 0.135x / case10 0.57x / case14 0.79x / case20 0.60x（5-case geomean 0.131→0.476）。

### 迭代 8 — [#9] KV 预取恢复（手动双缓冲 + T.serial）：评估后判定 Developer 边界内不可行，不实施

```
[ORDER-CHECK] 准备实施: [#9] 手动双缓冲 K/V 预取 | 前置依赖: [全部前项完成 ✅] | 结论: 以 scratch 最小复现先行（probe_db_prefetch.py，不触碰 mha.py）
[IMPL-#9] 已阅读 optimization-guide §2.2.0 同步模式决策表 L106-125（CV 融合 Double Buffer → AUTO_SYNC=False + 手动三路 flag + cross flag）+ §2.3 L293-321（首轮预取三段结构）+ cube_optimization_path §3.1 L305-311（gemm_v0 内嵌同步："算子外部无法实现 l1 层的双缓冲"）+ DESIGN §10.3-⑥/§10.1-R9。
[RESULT-#9] 优化点: K/V 手动双缓冲预取 | 精度: n/a（未进入 mha.py）| 性能: 不实施 | 判定依据（codegen 级证据，/tmp/opencode/db_probe_src.cc + db_probe_srcB.cc）:
  - Variant A（[2,bN,D] alloc_shared + runtime k%2 索引 + runtime-if 守卫预取）: 编译+运行 ✓（R9 的 address_map bug 仅命中 T.Pipelined/MEMORY_PLANNING 自动双缓冲路径，手动 3D 形式可编译），但 AUTO_SYNC 对 runtime-if + 动态偏移生成保守 `PipeBarrier<PIPE_ALL>` 全排空（预取与 gemm 零重叠，反而重于单缓冲的 flag 对）。
  - Variant B（去掉 runtime-if，纯取模索引）: 生成 `SetFlag/WaitFlag<M_MTE2>` + copy + `SetFlag/WaitFlag<MTE2_M>` + gemm 的成对全串行序列——pass 按迭代成对处理 MTE2→M 依赖，不识别双缓冲解除 WAR → 仍零重叠。
  - 结论: Developer（AUTO_SYNC=True）内手动双缓冲预取无收益；真重叠需 AUTO_SYNC=False + 手动 flag = Expert 化改写（越出 DESIGN §7.1/§8.6 边界；预期收益仅 ~5-10%（K load ~0.6us / 3.5-4us per-iter，V load 已天然重叠），与剩余预算（2 轮，收口验证强制）不匹配 → 记录为"评估后不实施"，恢复路径保留 DESIGN §10.3-⑥(b)/(c)（框架 codegen/R8 修复后重评）。
[DB-ANALYSIS]（按 skill 要求补记）Q1 循环内 MTE3: 有（w2 P 写回 + Output 写回）；Q2 跨迭代累加器: 有（acc_o/sumexp/m_i UB 常驻）；Q3 同步方式: 若实施须 AUTO_SYNC=False + mte3→mte2/mte2→v/v→mte3 + cross flag 全套手动（§2.2.0 CV 融合行）——即 Expert 化，为本项不实施的判定核心。
```

## 优化项状态总表（ORDER-PLAN 完成度）

| # | 优化点 | 状态 | 关键收益（msprof median） |
|---|--------|------|--------------------------|
| 1 | causal KV 运行期裁剪 | ✅ 采纳（iter1） | case7 -46.6% |
| 2 | 行循环 broadcast 2D 化 | ✅ 采纳（iter2） | case7 -32.3% / case10 -50.9% / case14 -41.4% |
| 7 | Fill+Add→Copy | ✅ 采纳（iter3，#3 使能项，单独中性） | UB -16KB |
| 3 | 掩码链 2D 化 + hoist | ✅ 采纳（iter3） | case7 -41.3% / case1 -39.2% |
| 6 | decode 窄块 block_M=16 | ✅ 采纳（iter4） | case10 -31.6% / case14 -25.5% |
| 5 | Fixed Core | ✅ 采纳（iter5） | case14 -18.7% / case1 -17.0% / case20 -6.9% |
| 4 | 掩码运行期跳过 | ✅ 采纳（iter6） | case7 -12.1% / case16 -9.1% |
| 8d | decode 分支 bN=128 | ✅ 采纳（iter7） | case10 -38.7% / case14 -34.0% / case16 -36.1% |
| 8 | 全局 bN=128 | ❌ UB 不可行（prefill 242KB>192KB，记录） | — |
| 9 | K/V 手动双缓冲预取 | ❌ Developer 边界内不可行（codegen 证据，记录） | — |

### 迭代 9（收口） — 全量验证 + cann-bench 复评（2026-09-18T22:15+08:00）

- docstring 同步（注释-only）后：ruff check/format clean、smoke 编译+运行 ✓、提交副本同步 + pip 重装 ✓。
- `--level all` exit 0：L0 10/10 / L1 19/19 / L2 12/12 / Boundary 6/6 / 哨兵 4/4 PERF_PASS（prefill 哨兵 warm kernel 0.27s→0.07s）。
- cann-bench 复评 cann_final_eval_20260918_220619：20/20 pass、综合 61.76（+7.58）、性能分 11.76（2.8x）、平均加速比 0.42x（3.2x）；逐 case 1.84x~6.24x 全提升。
- 最终报告：perf_tuning/final_report.md；备份链 history_version/mha_impl_s3_perf1..8.py 完整可回溯。

---

# 会话 2（perf_session=2，2026-09-19，追加预算；全局迭代 9/10 已用，本会话内部上限 6 轮）

用户指令：保持 Developer 模式，常规调优深挖剩余性能空间。噪声阈值 3%。会话 1 的 8 项优化全部保留为基线。

## S2-Step 1 基线验证（复用会话 1 终态 + 复测确认，baseline_iter10.json）

- l0 10/10 PASS（/tmp/opencode/l0_s2_start.log）——精度门禁起点确认。
- msprof 复测 6 代表 shape vs baseline_iter8：case1 14.86(-1.2%) / case7 3202.04(-0.2%) / case10 92.28(-0.7%) / case14 672.63(+0.8%) / case20 54.67(-0.2%) / case16 1266.42(+1.6%)——全部噪声带内，**基线成立**（证据 msprof_s2_base/*.json）。
- 会话 2 比较基线（msprof median us）：case1 14.86 | case7 3202.04 | case10 92.28 | case14 672.63 | case20 54.67 | case16 1266.42。

## S2-Step 2 算子类型

CV 融合混合型（会话 1 判定沿用，kernel 结构未变：KERNEL_TYPE_MIX_AIC_1_1，ASCEND_IS_AIC+AIV 双段）。

## S2-Step 3 Part A2 优化点清单（针对会话 1 移交线索逐项评估）

> 复读范围：performance-antipatterns.md 全文（498 行版，含硬件容量表 L27-44：UB=196352B=191.75KB、L1=524032B、L0C=131072B、L2=192MB）、optimization-guide.md 目录+§2.11（L539-632 UB 预算）、flash_attn_optimize.md §6 批量 Softmax（L355-448）+§9 多 shape 适配（L545-643）+§10（L647+）、ascend_tile.py API 契约（binary_op L994-1050 断言 size 相等=无轴广播二目形态；broadcast L2353-2397 dst/src 支持 BufferRegion；compare L1888-1962 tensor/tensor-scalar；mul_add_dst L1367-1422 dst=src0*src1+dst 三 buffer 同 size）。会话 1 Part A 全清单（L55-133）不重复评估，其结论继续有效；本轮只评估新候选与移交线索。

[#10] decode 分支 KV tile 再扩宽 bN=128→256/512 自适应（antipatterns「tile size 过小」L267-299 + flash_attn §9 按 shape 分流 L631-643 + 会话 1 #8d 先例）：**适用（最高优先）** — cann-bench 9 个 decode case 中 8 个 S_kv≥256（case10 Skv=2048、case11 1024、case12/13/15/16/18 512、case17 256）；#8d 实证 decode 每迭代固定成本 ~1.85us 与 tile 宽度弱相关（bN 64→128 仅 +0.53us），迭代数减半→净收益。容量核算（bM=16）：bN=512/D=128 UB naive ~141KB+planner 开销 ✓、L0C [16,512]f32 32KB+8KB ✓、L1 k+v 128+128+q 4+acc_s_l1 16=276KB<511.75 ✓；bN=512/D=256 L1=536KB **超限** → D=256 封顶 bN=256（L1 272KB ✓）。case14（Skv=128）保持 bN=128 零改动（避免 (F,F)→(F,T) 掩码变体迁移）。谓词边界移动：single_kv_block→S_kv≤bN（case12/13/15?/16/17/18 落入单块预归一化=更高精度路径，#8d 已认可的谓词移动先例，公式 bN 无关）；case10/11 保持多块。
```
[KERNEL-REUSE-PRECHECK]
candidate: #10 decode bN 扩宽（antipatterns L267-299 tile-size 条目 + flash_attn §9 L631-643 按 shape 分流；#8d 同族先例）
expanded_domain: S<=16 分支内 bN 从固定 128 变为 {128,256,512} 按 S_kv/D 自适应；kv_iters/need_bound_mask/single_kv_block 三个编译期谓词边界随之移动；prefill 分支零变化
compatibility_dimensions: ①gemm_v0 N=256/512(GEMM1)+K=256/512(GEMM2) 编译运行=UNKNOWN→scratch probe 编译冒烟；②UB/L0C/L1 容量=TRUE(手算如上，probe InitBuffer 复核)；③单块特化谓词扩至 S_kv<=512 的位级语义=TRUE by 数学(P̂ 公式 bN 无关)+probe 阻塞口径验证；④(F,F)->(F,T)/(T,*) 掩码变体迁移=TRUE by 构造(阈值 min(...,S_kv-1) 同一+#4 跳过保证仅对角/尾块执行)+l0/l1 验证；⑤decode 有效行数值=FALSE 预期(归约宽度 128->256/512 改变 online-softmax 中间舍入序列——非位级，按 #8d 同款阻塞口径+MARE_abs 余量判定；落入单块特化的 shape 预期转为位级精确)；⑥causal trim 公式对 bN 参数化=TRUE(#1 公式 floordiv(min(...),block_N)+1 原生参数化)
unknown_or_false: ①(编译验证)、⑤(预期非位级，阻塞口径判定)
plan: repair-kernel-first（scratch probe：编译冒烟+UB/L1 读数+数值容差比对 → 达标才改 mha.py；失败回滚不采纳）
```

[#11] prefill bN=128（会话 1 #8 遗留线索：UB 压力缓解后重评；antipatterns L267-299 + guide §2.11 L539-632 + 移交线索 1）：**候选（probe 决定）** — prefill 11 case 全部受益（迭代减半：case7 avg eff 9→4.5、case20/19/6 2→1 落单块特化、case5 4→2、case1 2→1…）。容量核算（naive UB）：D=64/bM=64/bN=128 ≈155KB ✓ 无需手术；D=256/bM=32/bN=128 ≈163KB ✓ 无需手术；**D=128/bM=64/bN=128 ≈227KB ✗ 超 191.75**（会话 1 实测 planner 在 bN=64 形态 naive 170.5→实配 191.8，+21KB 开销）→ 需 UB 手术才可能放行：
  - S1 掩码链 fp16 化（-16KB）：rt_2d [bM,bN] f32→f16 且列值 j0 平移（col-j0∈[0,bN-1]、row_thresh-j0<bM+bN≤191，均 ≤2047 fp16 整数精确域内→compare 位级同一；j0 平移使 row_thresh 失去 task 不变性，需每掩码迭代重算 AP+min——#4 跳过后仅对角/尾块执行，成本可忽略）；col_2d 落 acc_s_half（掩码期死亡：cast 链已毕、P 写回未始；WAW/WAR 链由 buffer 依赖保序）。风险：acc_s_half 是 w2 P 路径的 UB→GM copy 源，#2 教训（flag 锚定 copy 重排）需 probe 位级验证多迭代 causal 形状。
  - S2 m_i_2d∪r_2d 合并（-32KB，仅 D==bN==128 时全 buffer 连续角色复用，无跨步 region）：softmax sub/div 角色 [bM,128] 与 rescale/终归一化角色 [bM,128] 活跃期顺序不重叠（sub/div 在 GEMM2 前、rescale 在 GEMM2 后），两者均为纯 V scratch 非跨核 copy 目标（#2 教训不适用）。
  - S1+S2 后 naive ≈179KB，+planner 开销 → probe 实测定夺；仍超则 D=128 prefill 保持 bN=64（部分胜利：D=64/D=256 放行）。
```
[KERNEL-REUSE-PRECHECK]
candidate: #11 prefill bN=128（antipatterns tile-size + §2.11 UB 预算；会话 1 #8 的 UB 手术后重评）
expanded_domain: S>16 分支 bN 64→128（D=64/128/256 各自评估）；kv_iters/need_bound_mask/single_kv_block 谓词边界移动（prefill Skv<=128 落单块特化：case6/19/20 型）；若 S1 落地，掩码链数值域从 fp32 绝对列变为 fp16 j0 平移列（仅 S_kv<=2048+bN 平移域内启用，S_kv>2048 保留 fp32 路径或封顶 bN=64——probe 后定）
compatibility_dimensions: ①gemm_v0 N=128(GEMM1 prefill bM=64/32)+K=128(GEMM2)=TRUE(#8d decode 已验证 N/K=128；bM=64 形态编译 probe 复核)；②UB 容量=UNKNOWN→probe 实测 InitBuffer（三 D 值分别）；③单块特化谓词扩至 prefill Skv<=128=TRUE by 数学+probe（#8d 先例同款）；④掩码 fp16 平移域位级等价=UNKNOWN→S1 probe 位级比对（整数 ≤2047 fp16 精确，理论位级同一，需实证 codegen compare fp16 支持）；⑤多块 online-softmax 舍入序列变化=FALSE 预期（非位级，阻塞口径判定）；⑥负载均衡（#5 FC 连续 chunk × trim）=TRUE（case7 s_blocks=16 不变、chunk=103 覆盖 ≥1 完整 bx 周期）
unknown_or_false: ②④（probe 实测）、⑤（阻塞口径判定）
plan: repair-kernel-first（probe 编译读 UB → 决定手术组合 → 位级/阻塞口径 probe → l0/l1 → msprof；任一失败回滚）
```

[#12] 反模式全清单复扫（移交线索 4；antipatterns 全 498 行 vs 当前 368 行 mha.py）：**适用（静态项，随收口执行）** — 会话 1 扫描对象是 229 行旧版；本版新增 Fixed Core 任务循环/decode 分支/掩码跳过等结构需对照复查（关注项 A/B、行循环、冗余同步、指令拼接、tile size、CV overlap、逐块归约、正交轴、误分核）。发现即记录（评审记录模板 L488-498），可修则修。

### 证据性放弃项（移交线索 2/3/5，读完约束后排除）

- **线索 2 CV workspace GM 往返段数削减**：放弃（Developer 边界内无表达手段）。证据：①批量 Softmax（flash_attn §6 L387-448）依赖 set_flag/wait_flag/set_cross_flag/wait_cross_flag + cross_interval + MTE3→V ownership 闭环 = Expert 专属（会话 1 Part A L127 同款判定，本轮复读 L387-448 确认无 Developer 形态）；②w1/w2/w3 的 GM 路由由 AUTO_CV_COMBINE 自动生成，源码级不可见不可改（会话 1 Step 2 生成代码证据）；③P 上 L1 已是 fp16 最小 dtype（w2 8KB/iter@bN=64），布局由框架分形化；④L0C→UB 落地缓冲复用 = 与跨核 copy 目标别名 = 会话 1 #2 实测 corruption 教训（mha.py L173-178 注释固化）。bN 扩宽（#10/#11）是 Developer 内削减往返**次数**的唯一杠杆，已列为本轮主候选。
- **线索 3a decode heads-as-M**：放弃（数学结构不适用）。本算子为 MHA——每个 head 有独立 K/V（proto.yaml 输入 [B,S_kv,N,D]），heads 折叠进 M 维要求 GEMM 的 B 操作数跨行共享，而 head n 的行只能用 K[bz,:,n,:]；折叠后 Q[N·S,D]×K^T 语义错误（除非 K 复制 N 份=显存/带宽×N）。xattention heads-as-M 先例成立的前提是 MQA 单 KV head。DESIGN §10.3 的"heads-as-M 重排"注记针对的是 GQA/MQA 形态，对 MHA proto 不适用。
- **线索 3b decode split-KV（flash-decoding 二段式）**：放弃（收益核算为负/零）。slot-iter 总量守恒（tasks×iters 不变），split 只改 makespan 取整：case10 现 2 slots×16 iters（32 tasks/20 核，packing 低效 1.25x）——#10 bN=512 后 2 slots×4 iters 已把绝对代价压到 ~25-35us，split-2 使 tasks=64→4 slots×2=8 slot-iters vs 2×4=8 **零收益**且新增第二段 merge kernel launch + partial(O,m,sumexp) GM 往返；case11-13/15-18 tasks≫cores 无 packing 低效可修。二段式还引入新 kernel 调用域（host 双 launch+中间 GM buffer），验证面/风险与收益不成比例。
- **线索 5 Fixed Core 分流**：不重做（会话 1 已评估放弃：双倍 dispatch 域不值 +1.7~2.1% 噪声带漂移；无新证据）。
- **§10.3-④ D=256 bM=64**：维持放弃（bN=64 下 naive UB≈280KB 已超限，无手术空间；bM=32 是容量必然）。
- **mul_add_dst 融合 rescale**（antipatterns L202-223 mul+add pattern 复查）：不适用 — API 语义 dst=src0*src1+dst（L1372），本 kernel 需要 acc_o=acc_o*r_2d+acc_o_ub（两个"源"之一即 dst 自身的旧值×第三 buffer），映射为 mul_add_dst(acc_o_ub, acc_o, r_2d)→acc_o_ub=acc_o*r_2d+acc_o_ub 后还需 copy 回 acc_o（指令数 3→2 但引入 acc_o_ub 提前覆写=下一迭代 copy 目标冲突），且改变 w3 落地缓冲活跃期（#2 教训域）。收益 ~1 指令/iter（~350 分之 1），风险不对称，放弃。
- **axpy 融合 scale**（会话 1 已拒）：维持——破坏双舍入栅格精度红线。

## S2-Part B2 [ORDER-PLAN]

依赖分析：#10（decode 分支）与 #11（prefill 分支）编辑面正交（block_N 派生行 + 各自分支谓词）；#11 内部 S1/S2 手术是 bN=128/D=128 的使能前置（数量依赖规则：预算类最后、手术先于谓词翻转）；#12 静态复扫放最后（扫终态代码）。

```
[ORDER-PLAN] 实施顺序（会话 2）：
1. [#10] decode bN 256/512 自适应 — 前置依赖: [无] — 理由: 价值/风险比最高（8/9 decode case 受益、#8d 同族先例、UB/L1 余量充足无手术）；probe 先行定 bN 上限
2. [#11a] UB 手术（S2 scratch 合并，视 probe 需要加 S1 fp16 掩码） — 前置依赖: [#10（decode 分支定形后 prefill probe 读数不受干扰）] — 理由: #11b 的使能前置（§2.11 L630-632 预算类最后实施原则的手术侧）
3. [#11b] prefill bN=128 谓词翻转（按 D 分档放行） — 前置依赖: [#11a（若 probe 显示 D=128 需手术）] — 理由: prefill 11 case 全部受益（case7/8 最差 0.13x 的主杠杆）
4. [#12] 反模式复扫 + 收口（--level all + cann-bench 复评 + final_report 更新） — 前置依赖: [#10/#11 落定] — 理由: 扫终态代码
```

中止规则（任务指令）：候选穷尽且收口 / 连续 3 次无提升(<3%) / 本会话内部 6 轮上限。

## S2-Step 4 逐项实施记录（会话 2）

### 迭代 10 — [#10] decode 分支 KV tile 再扩宽（bN 128→256/512 自适应）

```
[ORDER-CHECK] 准备实施: [#10] decode bN 扩宽 | 前置依赖: [无 ✅] | 结论: ✅（ORDER-PLAN 会话 2 第 1 位）
[IMPL-#10] 已阅读 performance-antipatterns.md L267-299（tile size 过小条目：扩大 tile 前必须重算 UB/L1/L0C + 检查点"尾块处理和 mask 逻辑要随 tile size 一起复查"）+ L27-44（硬件容量表：UB 196352B / L1 524032B / L0C 131072B）+ flash_attn_optimize.md L631-643（按 shape 分流：快而受限路径与通用路径分开的先例）+ 会话 1 #8d 全记录（谓词边界移动/single_kv_block 扩展/非位级判定口径三先例）+ mha.py L101-137 现状。关键约束: ①自适应规则（host 元数据判定，零张量操作）: S<=16 时 bN = 512 if (D<=128 and S_kv>256) else 256 if S_kv>128 else 128；D=256 封顶 bN=256（L1: k+v [512,256]fp16=512KB+q+acc_s_l1=536KB>511.75KB 超限）；case14 型 S_kv<=128 保持 bN=128 零改动；②谓词移动: single_kv_block→S_kv<=bN（case12/13/16/17/18 型落单块预归一化=更高精度路径，#8d 认可先例，公式 bN 无关）；need_bound_mask→S_kv%bN!=0（(F,F)→(F,T) 迁移由 #4 跳过保证仅尾块执行，阈值语义同一）；③causal trim 公式 floordiv(min(s0+bM-1+offset,S_kv-1),bN)+1 原生参数化，eff<=kv_iters 恒成立；④容量手算（bM=16）: bN=512/D=128 UB naive≈148KB(causal 含掩码链)/L0C 40KB/L1 276KB 全 ✓；bN=256/D=256 UB≈132KB/L1 272KB ✓——probe 实测复核；⑤padding 行（S<bM）: Q 尾行清零→score=0→行独立不污染有效行（既有语义，bN 无关）。
[SELF-CHECK] scratch probe 先行（probe_bN_wide.py，不触碰 mha.py）；mha.py Edit 仅在 probe 通过后执行且只涉及 [#10]（block_N 派生行 + 注释；备份 history_version/mha_impl_s3_perf9.py = 会话 1 终态 ✅ 已落盘）
```
```
[KERNEL-REUSE-AUDIT] 依据: Part A2 [KERNEL-REUSE-PRECHECK] #10 | 维度及结论: ①gemm_v0 N=256(GEMM1 transpose_B)=TRUE（probe 11/11 编译；**N=512=FALSE 实证**——common.h L1263-1278: transpose_B 路径 nTile==N 不分片，B tile=bN×kL0Size(128)×2B 须 ≤ L0B 预算（kNumSteps=1 时 64KB / >1 时 ping-pong 32KB）→ D≤128 封顶 bN=256（恰满 64KB）、D=256（2 K-steps→32KB）封顶 bN=128；GEMM2 非 transpose 走 N 分片无碍）②UB/L1/L0C=TRUE（实测 UB≤93.8KB/L1 140KB/L0C 24KB，全在限内）③单块谓词扩至 Skv≤256（decode）=TRUE（probe: Skv=192/129 单块形状 max_abs=0.0 位级精确达 golden）④(F,F)→(F,T) 迁移=TRUE（Skv=129/192 bound-mask 单块过；l0/l1 复验在后）⑤非位级=确认（|Δvs_perf9|≤9.8e-4，MARE_abs 3.05e-5~2.44e-4 对 gate 余量 40-320x）⑥trim 参数化=TRUE（Skv=1000 causal 4-iter 过）| 结论: reuse（unknown 全落定；bN=512 变体与 D=256 扩宽分别被 L0B 硬限否决/延后——D=256 可走 GEMM1 kL0Size=64（ascend.py L418 advanced 参数）留作 #10b）
```

```
[SELF-CHECK] mha.py Edit 只涉及 [#10]（block_N 派生 1 行 + 注释块；备份 perf9 = 会话 1 终态）
[RESULT-#10a] 优化点: decode 分支 bN=256 自适应（S<=16 且 D<=128 且 S_kv>128）| 精度: pass（probe_bN_wide: 11/11 编译 + 6/6 BITEXACT 零变化路径（含 D=256 decode/prefill）+ 10/10 阻塞口径（MARE_abs 3.05e-5~2.44e-4 对 gate 余量 40-320x；单块化 shape Skv=129/192 max_abs=0.0 位级精确达 golden）；l0 10/10 + l1 19/19，/tmp/opencode/probe_bN_wide2.log、l0_s2_iter10.log、l1_s2_iter10.log）| 性能: case10 65.68us / case16 932.84us / case14 672.08us / case20 54.81us / case1 14.92us | 对比: vs 会话 2 基线 case10 **-28.8%**、case16 **-26.3%**；case14 -0.1% / case20 +0.3% / case1 +0.4%（守卫项零变化 ✓）| 采纳: YES（新基线 = s2_iter10；证据 msprof_s2_iter10/*.json）
```

补充（#10a 结构性发现，指导后续）：gemm_v0 transpose_B 路径 nTile==N 不分片（common.h L1263-1278），GEMM1 B tile 预算 = L0B 64KB（kNumSteps=1）/ 32KB（多 K-step ping-pong）→ **prefill bN=128 的 GEMM1（D=128: 128×128×2=32KB；D=256: 2-step → 32KB 预算，nTile=128×128×2=32768 恰好）同样合法**，#11 无 L0B 障碍；bN=256 prefill（64×128×2=16KB... transpose_B nTile=256 → 65536 ≤ 64KB 亦合法）留作 #11 后备扩展。

### 迭代 11 — [#10b] D=256 decode bN=256（GEMM1 kL0Size=64）

```
[ORDER-CHECK] 准备实施: [#10b] D=256 decode 扩宽 | 前置依赖: [#10a ✅（自适应规则已落地，本子项仅放宽 D<=128 限定 + GEMM1 kL0Size）] | 结论: ✅
[IMPL-#10b] 已阅读 ascend.py L418-525（gemm_v0 签名：kL0Size=128 默认、L444-454 文档"Smaller kL0Size allows larger block_M/block_N"、L501-503 断言 16 倍数）+ common.h L1200-1278（kL0split=ceil(K/kL0Size)、kNumSteps=kL0split×nL0split、kL0Budget=64KB/(kNumSteps>1?2:1)、C0 对齐断言 kL0Size%16==0、K-tail 断言）。关键约束: ①D=256/bN=256/kL0Size=64: kL0split=4→kNumSteps=4→budget 32KB；nTile(transpose_B)=256×64×2B=32768 ≤32768 恰好 ✓；②GEMM2 默认 kL0Size=128 合法（N=256 走 nL0split=2、B tile 128×128×2=32768 ≤32KB ✓）→ 仅 GEMM1 传参；③kL0Size=64 仅作用于新域（S<=16 且 D==256 且 S_kv>128），其余配置 default 128 零变化；④L1 272KB/UB ~121KB/L0C 32KB 手算 ✓ probe 实测；⑤K 累加顺序不变（L0C 跨 K-tile 顺序累加，kL0Size 只改 L1→L0 搬运粒度）→ GEMM 位级等价预期，online-softmax 归约宽度 128→256 非位级（阻塞口径判定，#8d/#10a 同款）。
[SELF-CHECK] scratch probe 先行（probe_d256_wide.py）；mha.py Edit 只涉及 [#10b]（block_N 派生放宽 + gemm_v0 kL0Size 参数 + 注释；备份 perf10 = post-#10a ✅）
```

```
[KERNEL-REUSE-AUDIT] 依据: #10b（#10 PRECHECK 的 D=256 延后子项）| 维度及结论: ①gemm_v0 N=256 transpose_B kL0Size=64=TRUE（probe 5/5 编译，模板 gemm_v0<half,float,16,256,256,false,true,64> 实证；GEMM2 保持 kL0Size=128 N-分片路径）②容量=TRUE（UB≤121.8KB / L1 272KB / L0C 32KB 实测）③K 累加序不变=TRUE by 构造（L0C 跨 K-tile 顺序累加）+probe 数值余量 160-640x ④零变化域 BITEXACT=TRUE（6/6 含 D=256 Skv≤128 / prefill D=256 / D=128 decode）⑤非位级=确认（|Δ|≤1.22e-4，单块化 Skv=129 max_abs=0.0 位级达 golden）| 结论: reuse
[SELF-CHECK] mha.py Edit 只涉及 [#10b]（block_N 派生放宽 + gemm1_kL0 派生 + GEMM1 传参 + 注释；备份 perf10 = post-#10a）
[RESULT-#10b] 优化点: D=256 decode bN=256（GEMM1 kL0Size=64）| 精度: pass（probe_d256_wide: 5/5 编译 + 6/6 BITEXACT + 8/8 阻塞口径（MARE_abs 0~6.1e-5，余量 160-640x）；l0 10/10 + l1 19/19，/tmp/opencode/probe_d256.log、l0_s2_iter11.log、l1_s2_iter11.log）| 性能: case15 583.76us / case18 583.94us / case4 64.99us(守卫) / case10 65.71us(守卫) | 对比: vs post-#10a case15 **-22.7%**（755.44→583.76）、case18 **-22.8%**（756.42→583.94）；守卫 case4 -0.3% / case10 +0.05% | 采纳: YES（新基线 = s2_iter11；证据 msprof_s2_iter11/*.json + 文件交换法基线 md5 校验还原）
```

### 迭代 12 — [#11] prefill bN=128（probe 决定手术组合）

```
[ORDER-CHECK] 准备实施: [#11] prefill bN=128 | 前置依赖: [#10a/b ✅（decode 分支定形；GEMM1 L0B 预算规则已探明——prefill bN=128: D=64 nTile 128×128×2=32KB≤64KB / D=128 kNumSteps=1 32KB≤64KB / D=256 kNumSteps=2 32KB≤32KB 恰好，全 default kL0Size=128 合法；GEMM2 N-分片路径合法）] | 结论: ✅（ORDER-PLAN 会话 2 第 3 位；#11a 手术是否执行由本 probe 的 UB 实测决定）
[IMPL-#11] 已阅读 optimization-guide §2.11 L539-632（UB 预算公式/实施顺序：预算类最后）+ antipatterns L27-44（容量表+MEMORY_PLANNING 说明"理论值只作参考，最终以实际编译为准"）+ L267-299（tile size 检查点：尾块/mask 随 tile 复查）+ 会话 1 #8 否决记录（L103: naive 208-242KB）+ mha.py buffer 声明区 L158-191。关键约束: ①谓词移动: kv_iters 减半、single_kv_block→prefill Skv≤128（case6/19/20 型落单块=更高精度路径）、need_bound_mask→Skv%128（bench prefill 全 Skv%128==0 无迁移，域内一般 shape 走 #4 尾块语义）；②trim 公式 bN 参数化 ✓；③FC 负载均衡: case7 chunk=103 ≥ s_blocks=16 ✓；④UB naive（bM=64,bN=128）: D=64≈155KB / D=128≈227KB / D=256(bM=32)≈163KB——planner 实配 probe 读取（GetWithOffset max）定手术需求；⑤手术菜单（按风险升序）: S2=m_i_2d∪r_2d 合并（仅 D==bN==128 全 buffer 连续角色，活跃期 sub/div(GEMM2 前)与 rescale(GEMM2 后)不重叠，均纯 V scratch 非跨核 copy 目标——#2 教训不适用）-32KB；S1=掩码链 fp16+j0 平移（col/row 值域平移后 ≤191+bM… ≤2047 fp16 精确域，compare 位级同一；col_2d 落 acc_s_half 掩码期死亡窗）-16KB——两者均 probe 位级验证后才可动 mha.py。
```

**Probe-1 实测（纯谓词翻转，零手术，/tmp/opencode/mha_pf128.py）**：
- (2,128,128,8,64,T) UB=186.8KB / (2,1024,1024,2,128,T) UB=**178.8KB** / (2,256,256,8,256,T) UB=177.6KB / (2,128,256,12,64,F) UB=153.0KB——全部编译通过且在 196352B 预算内（L1 56-152KB、L0C 48-64KB ✓）。
- 结论：**S1/S2 手术不需要**（会话 1 #8 的 216-242KB 为手算 naive 和，未实编；MEMORY_PLANNING 对死亡 buffer 的合并远优于 naive 估计——rt_2d(掩码后死)/acc_s_half(cast 后死)/m_i_2d 等被复用）。#11 降级为纯 block_N 谓词翻转，编辑面 1 行。D=64 causal 186.8KB 距上限 4.9KB 为最紧配置（后续任何 buffer 新增都须重 probe）。

```
[KERNEL-REUSE-AUDIT] 依据: Part A2 [KERNEL-REUSE-PRECHECK] #11 | 维度及结论: ①gemm_v0 N=128 prefill（bM=64/32 transpose_B）=TRUE（probe_pf128 4/4 编译 + GEMM1 B tile 32KB≤L0B 预算全 D 合法）②UB 容量=TRUE（实测 153.0-186.8KB ≤191.75KB，S1/S2 手术不需要——会话 1 手算 216-242KB 被 MEMORY_PLANNING 实测证伪；D=64 causal 186.8KB 为最紧配置，余量 4.9KB 已注释固化）③单块谓词扩至 prefill Skv≤128=TRUE（ZEROS 用例 GOLDEN-BITEXACT；随机用例 matched=1.00000、Δgolden 1.2e-4 ≪ Δold 9.8e-4 = 更接近 golden 的高精度路径——probe 初版"GOLDEN-BITEXACT 必须"判定过严已修正为参考信息，依据 test_mha.py L10/L611: MERE 仅参考、阻塞口径=混合容差+MARE_abs）④(F,F)→(F,T) 迁移=TRUE（(1,200,300)/(2,130,260)/(1,127,127) 奇数尾过）⑤多块非位级=确认（8/8 阻塞口径过，MARE_abs≤1.95e-3 余量 5-80x）⑥FC 负载均衡=TRUE（chunk=103 ≥ s_blocks=16）| 结论: reuse（direct——手术项 S1/S2 被实测证明不需要，未实施）
[SELF-CHECK] mha.py Edit 只涉及 [#11]（block_N 谓词统一为 `256 if (S<=16 and S_kv>128) else 128` + 注释；备份 perf11 = post-#10b（/tmp/opencode/mha_post10b.py md5 校验落盘——流程修正：编辑时点遗漏，测量交换前补建并 md5 双确认）；ruff check/format clean）
[RESULT-#11] 优化点: prefill bN=128（全 D；MEMORY_PLANNING 实测放行，零手术）| 精度: pass（probe_pf128: decode 守卫 3/3 BITEXACT + prefill 单块化 5/5（含 ZEROS GOLDEN-BITEXACT）+ 多块 8/8 阻塞口径；l0 10/10 + l1 19/19；/tmp/opencode/probe_pf128b.log、l0_s2_iter12.log、l1_s2_iter12.log）| 性能: case7 2234.75us / case20 38.26us / case3 352.77us / case4 52.47us / case1 15.21us / case10 65.27us(守卫) | 对比: vs post-#10b case7 **-30.2%**（3202.04→2234.75）、case20 **-30.2%**（54.81→38.26）、case3 **-25.1%**（471.18→352.77）、case4 **-19.3%**（64.99→52.47）、case1 +1.9%（噪声带内小 wall 漂移：32 tasks/20 核 makespan 2 槽、迭代已 1/1，单块对角掩码 [64,128] 变大——如实记录）、case10 -0.7%（守卫 ✓）| 采纳: YES（新基线 = s2_iter12；证据 msprof_s2_iter12/*.json + 文件交换法 case3 基线）
```

### 迭代 13 — [#13] decode bN=512（D<=128 且 S_kv>256；GEMM1 kL0Size=32）

```
[ORDER-CHECK] 准备实施: [#13] decode bN=512 | 前置依赖: [#10a/b ✅（bN 自适应框架与 kL0Size 传参通道已落地）；#11 ✅（prefill 定形，decode/prefill 谓词互不干扰）] | 结论: ✅（#11 后备扩展转正：迭代数再减半打击 case10/11/12/13/16 = 5-6 个 bench case）
[IMPL-#13] 已阅读 ascend.py L418-525（kL0Size 断言：>0、%16==0、≤4095）+ common.h L1200-1278（kL0split=ceil(K/kL0Size)、kNumSteps=kL0split×nL0split、kL0Budget=64KB/(kNumSteps>1?2:1)、C0 对齐 kL0Size%16、K-tail 断言）+ #10b 先例（kL0Size=64 已实证）。关键约束: ①GEMM1 D=128/bN=512/kL0Size=32: kL0split=4→budget 32KB；nTile(transpose_B)=512×32×2B=32768 ≤32768 恰好；D=64: kL0split=2→32KB 同 ✓；K-tail 128-3×32=32 %16=0 ✓；②GEMM2 保持 default（K=512: kL0split=4、nL0split=1(D=128)/N 分片、B tile 128×128×2=32768 ≤32KB ✓）；③D=256 排除（L1 k+v [512,256]fp16=512KB+q+acc_s_l1=536KB>511.75KB）；④容量手算（bM=16,bN=512,D=128）: UB naive≈143KB / L1 276KB / L0C 40KB——probe 实测；⑤谓词移动: single_kv_block→decode S_kv≤512（case12/13/16 型 2→1 iter 落单块=更高精度路径 #8d/#11 双先例）；need_bound_mask→S_kv%512；⑥规则: S≤16 时 bN=512 iff D≤128 且 S_kv>256；其余保持 #10/#11 定形（Skv∈(128,256] → 256、≤128 → 128、D=256 → 256 封顶、prefill 128）。
[SELF-CHECK] scratch probe 先行（probe_bN512.py，不触碰 mha.py）；mha.py Edit 只涉及 [#13]；备份 perf12 = post-#11（Edit 前落盘）
```

```
[KERNEL-REUSE-AUDIT] 依据: #13（#10 PRECHECK 同族扩展）| 维度及结论: ①gemm_v0 N=512 transpose_B kL0Size=32=TRUE（probe 9/9 编译；模板 gemm_v0<half,float,16,512,128,false,true,32> 实证；GEMM2 default 128 N-分片）②容量=TRUE（UB 110.2-159.3KB / L1 146-276KB / L0C 36-40KB 实测）③单块谓词扩至 decode Skv≤512=TRUE（Skv=300 D=64 GOLDEN-BITEXACT；case12/13/16 型落单块）④(F,F)→(F,T)/trim 交互=TRUE（Skv=513/768/1000 奇数尾过）⑤非位级=确认（MARE_abs 0~6.1e-5，余量 160-640x）⑥D=256 排除=TRUE by L1 手算 536>511.75（保持 bN=256）| 结论: reuse
[SELF-CHECK] mha.py Edit 只涉及 [#13]（block_N 三分支 + gemm1_kL0 三档 + 注释；备份 perf12 = post-#11 md5 校验；ruff clean）
[RESULT-#13] 优化点: decode bN=512（D≤128 且 S_kv>256，GEMM1 kL0Size=32）| 精度: pass（probe_bN512: 9/9 编译 + 6/6 零变化域 BITEXACT + 11/11 阻塞口径（MARE_abs≤6.1e-5 余量 160-640x，Skv=300 GOLDEN-BITEXACT）；l0 10/10 + l1 19/19；/tmp/opencode/probe_bN512.log、l0_s2_iter13.log、l1_s2_iter13.log）| 性能: case10 51.04us / case16 853.82us / case14 675.54us(守卫) / case7 2260.09us(守卫) | 对比: vs post-#11 case10 **-21.8%**（累计 vs 会话 2 基线 **-44.7%**：92.28→51.04）、case16 **-8.5%**（累计 -32.6%：1266.42→853.82，2 iters→单块）；守卫 case14 +0.5% / case7 +1.1%（BITEXACT 证实路径未变=噪声）| 采纳: YES（新基线 = s2_iter13；证据 msprof_s2_iter13/*.json）
```

### 迭代 14 — [#12] 反模式全清单复扫（终态 414 行 mha.py，静态项，零代码改动）

```
[ORDER-CHECK] 准备实施: [#12] 反模式复扫 | 前置依赖: [#10/#11/#13 ✅ 落定，扫终态代码] | 结论: ✅
[RESULT-#12] 优化点: antipatterns 全 11 项复扫 | 精度: n/a（零改动）| 性能: n/a | 逐项结论:
  - 关注项A（任务数>核数按任务数 launch）: ✅ 不命中——Fixed Core core_num=min(block_num,20)（L177-179）
  - 关注项B（任务数<核数仍满 launch）: ✅ 不命中——同 min() 退化保护
  - Vector 逐行 for loop: ✅ 不命中——源码零行循环（#2 broadcast 2D 化；任务/KV 循环为结构必需）
  - 冗余全局同步: ⚠️ 有意保留——源码零手动 barrier；生成代码逐 op PipeBarrier 来自 AUTO_SYNC（Developer 边界 DESIGN §7.1/§8.6；#9 会话 1 codegen 实证手动双缓冲零重叠；线索 2 会话 2 复读 flash_attn §6 L387-448 确认批量流水=Expert 专属）。保留原因=模式边界+实测无 Developer 内替代
  - 基础指令拼接: ✅ fill+add→copy 已做（#7）；mul+add→mul_add_dst 不可映射（dst=src0*src1+dst 语义 vs 本 kernel acc_o=acc_o*r+acc_o_ub，Part A2 证据）；4×cast 链=双舍入精度红线保留；无 sqrt+div/relu pattern
  - tile size 过小: ✅ 已探至硬件硬限——bN: decode 512（L0B transpose_B nTile×kL0Size 预算，512×32×2=32768 恰好）/prefill 128（UB 186.8KB 最紧 4.9KB 余量）；D=256 decode 256 封顶（L1 536>511.75）；prefill bM=128 naive UB 448KB ✗；**prefill bM=32+bN=256 量化评估后跳过**（slot-iter 仅 -6.7%（8600 vs 9216）且 KV 重读翻倍 604MB→1.1GB（L2 可吸收但 MTE2 指令×2）、case19/20 型 Skv=128 落 256 宽单块带 128 padding 列反退化——期望值低于噪声风险）
  - CV overlap 未开启: ⚠️ 有意保留——T.Pipelined 禁用（R9 address_map bug 无修复证据）；指令明示 #9 勿重做
  - 纯 AIV memory bound 未流水: ✅ 不适用（MIX 型）
  - 多行 tile 循环内逐块归约: ✅ 不命中——per-block reduce_max/sum(dim=-1) 为 online-softmax 数学必需；零 dim=0
  - 正交轴串行化: ✅ 不命中（无标量扫描轴）
  - 纯 Vector AUTO_CV_COMBINE 误分核: ✅ 不适用（MIX 型、无 alloc_var）
  - 附带发现: gemm1_kL0 上方注释"kL0Size=64 only for D=256"已过时（现有 32/64/128 三档）→ 收口注释同步修正（注释-only）
  结论: 无可修复新项；剩余结构成本 = AUTO_SYNC barrier 链 + 每迭代 CV workspace GM 往返（Expert 域，证据链完整）。
```

### 迭代 15（收口） — 全量验证 + cann-bench 复评（2026-09-19T18:20+08:00）

- docstring/注释同步（注释-only：模块 docstring 增会话 2 段 + gemm1_kL0 过时注释修正）后：ruff check/format clean、smoke 编译+运行 ✓、提交副本同步 + pip 重装（installed md5 校验一致）✓。
- `--level all` exit 0：**L0 10/10 / L1 19/19 / L2 12/12 / Boundary 6/6 / 哨兵 4/4 PERF_PASS**（kernel_path 6.1~7.2s ≤120s；perf_decode_worst (64,1,4096,32,128) causal 覆盖 bN=512+trim+掩码全交互 MARE_abs 3.05e-5 余量 320x；/tmp/opencode/all_s2_final.log）。
- cann-bench 复评 **cann_final_eval_20260919_154250**：20/20 pass、综合 61.76→**65.05**（+3.29）、性能分 11.76→**15.05**、平均加速比 0.42x→**0.55x**；19/20 case 提升（case10 0.518→0.85x、case20 0.590→0.82x、case11 0.446→0.77x、case7/8 0.13x→0.19x）；case1 0.671→0.64x（噪声带微降，[RESULT-#11] 已记录权衡依据）。
- 最终报告：perf_tuning/final_report.md 会话 2 章节（§S2-1~S2-6）；备份链 history_version/mha_impl_s3_perf9..12.py 完整可回溯；baseline_iter10..14.json + msprof_s2_*/*.json 证据链完整。
- 会话 2 状态总表：

| # | 优化点 | 状态 | 关键收益（msprof median） |
|---|--------|------|--------------------------|
| 10a | decode bN=256（D≤128,Skv>128） | ✅ 采纳（iter10） | case10 -28.8% / case16 -26.3% |
| 10b | D=256 decode bN=256（GEMM1 kL0Size=64） | ✅ 采纳（iter11） | case15 -22.7% / case18 -22.8% |
| 11 | prefill bN=128（全 D，MEMORY_PLANNING 实测放行零手术） | ✅ 采纳（iter12） | case7 -30.2% / case20 -30.2% / case3 -25.1% / case4 -19.3% |
| 13 | decode bN=512（D≤128,Skv>256，GEMM1 kL0Size=32） | ✅ 采纳（iter13） | case10 -21.8%（累计 -44.7%）/ case16 -8.5% |
| 12 | 反模式复扫 | ✅ 完成（iter14，零新可修复项） | — |
| — | decode bN=512 @D=256 | ❌ L1 536KB>511.75KB（记录） | — |
| — | prefill bN=256 / bM=128 | ❌ UB naive 336/448KB 不可行（记录） | — |
| — | prefill bM=32+bN=256 | ❌ 量化跳过（-6.7% slot-iter vs ×2 KV 重读） | — |
| — | heads-as-M / split-KV / CV workspace 削减 / 批量 softmax | ❌ 证据性放弃（Part A2：数学不适用 / 收益核算为负 / AUTO pass 不可源码干预 / Expert 专属） | — |

---

# 会话 3（perf_session=3，2026-09-19 起，Expert rev3 轨首轮；迭代预算 10，独立编号 r3_perf{N}）

> 对象：Expert 版 mha.py（rev3-attempt1 终态，md5 d71ba4c3，cann-bench 67.88 / 平均 0.68x）。
> 目标（DESIGN §12）：平均加速比逼近 1.0x；里程碑 M1 ≥0.75x 且 case7/8/9 ≥0.4x、M2 ≥0.9x；
> decode 逐 case 不回退（case10 ≥0.92x / case20 ≥0.82x 现状线 / **case14 恢复 ≥0.79x 首位主攻**）。
> 噪声带 3%；计时口径 = msprof timeline op_summary Task Duration median（warmup 5 + iters 20），继承 final_report §7。

## S3-Step 1 基线（baseline_r3.json 已落地；证据 msprof_r3_base/*.json）

| shape | case1 | case7 | case9 | case10 | case11 | case14 | case16 | case20 |
|-------|-------|-------|-------|--------|--------|--------|--------|--------|
| msprof median (us) | 12.91 | 928.00 | 1398.35 | 40.53 | 349.63 | 746.08 | 763.42 | 31.30 |
| cann-bench 锚 (us) | — | 929.5 | — | — | — | 747.3 | — | 31.68 |

- 互验：case7 差 0.16% / case14 差 0.16% / case20 差 1.2% → 口径一致 ✅。
- 精度基线：`--level l0` 10/10 PASS（r3_l0_base.log）；位级锚 10/10 BITEXACT vs Developer 终态 + R11 压力 PASS（r3_anchor_base.log，anchor_stress.py）。
- bench_mha.py 新增 case9/case11 条目（工具文件，非交付物）。

## S3-Step 2 算子类型（get_kernel_source 复核）

- FormA_case7 / FormB_case14 / FormB_case9 三份生成码均含 `ASCEND_IS_AIC` ×1 + `ASCEND_IS_AIV` ×2 → **CV 融合混合型（Expert cv_1_2 双 scope，符合预期）**。源码存档 /tmp/opencode/r3_src/*.cc。
- 生成码复核：`T.barrier_all()` → `AscendC::PipeBarrier<PIPE_ALL>`（AIC 核内全流水排空，Form B 每 kv 迭代 ×7）；Form A 全部为 SetFlag/WaitFlag<HardEvent> 细粒度对 + CrossCoreSetFlag/WaitFlag（无 barrier）；V 侧 io_buf/acc_s_half 管程闭环与 SEM 0-5 落点与设计 §7.2 一致。

## S3-Step 3 瓶颈归因（msprof op 模式，PipeUtilization，/tmp/opencode/r3_op_case{7,9,14}/）

| case | wall(us) | AIC 关键读数 | AIV 关键读数 | 归因结论 |
|------|----------|--------------|--------------|----------|
| case7 (Form A P2) | 928 | cube **17.6%**、mte2 **50.4%**(466us)、fix 32.3%、scalar 79% 其中 **wait_id2=606us**（等 MTE2 GM→L1）、wait_id5=108us、mte1_stall=127us | **vec 71%**(647us 近饱和)、mte2 22%(204us)、scalar wait_id4=292us（等 SEM w3）、vector_stall 260us | **双瓶颈**：C 侧 k/v L1 单缓冲致 scalar 等 MTE2 606us；V 侧 vector 近饱和（双舍入 4 cast + exp + broadcast/sub 全在 [32,128] fp32）→ 两条独立优化线（#R3-2 C 侧、#R3-3 V 侧） |
| case9 (Form B P3) | 1398 | cube 13.9%、mte2 **40%**(530us)、mte1 25%、fix 18%、scalar 仅 19%、wait_id2≈20us | vec 33%、mte2 12%、**wait_id0≈500us + wait_id4≈640us**（V 等 C 交付/等 MTE2）| **barrier_all 串行化**：AIC 各 pipe busy 之和 ≈95%（40+25+14+18）即几乎零重叠——PipeBarrier<PIPE_ALL> ×7/迭代 把 GM→L1、gemm、L0C→ws 串成链，V 全程饥饿 |
| case14 (Form B D3) | 746 | cube **3.6%**、mte2 38.9%(290us)、scalar 17.7% + **wait_id2≈200us** | vec 15.3%(114us)、mte2 8%、**wait_id0≈265us + wait_id4≈425us** | **纯协议延迟**：所有 pipe <40% busy；205 任务/核 × ~3.6us 严格 C↔V 乒乓（每任务 6 次跨核 SEM 往返 + 7 次 PipeBarrier，任务间零重叠）——证实 debug_log rev3-attempt1 微任务协议税归因 |

## S3-Part A3 优化点清单（Stage-2 移交线索逐项 + skill 文档全清单筛选）

[KERNEL-REUSE-PRECHECK]
candidate: #R3-1 single_kv_block 任务批处理（flash_attn_optimize.md §1 L67-L100 批量流水 + §7 workspace 槽位复用；DESIGN §10.3 微任务合并方向、debug_log case14 归因节）
expanded_domain: 无新增调用域——host 域/分派谓词不变，仅既有编译期谓词 single_kv_block（kv_iters==1）内部循环重构（批维度 = task 而非 kv block）；覆盖 case12/13/14/16/17（Form B）与 case1/19/20（Form A，v1 暂不动）
compatibility_dimensions: C/V flag 配对平衡（§7.2-5）＝批计数公式双侧同式（num_tb=ceildiv(my_count,NUM_STAGES)、batch_tasks=min(...)，均从同一 my_start/my_count 派生）✅；SEM set/wait 位置逐槽不变（CROSS_INTERVAL 语义原样）✅；每槽算子序列与现行单任务路径逐 op 同序（精度栅格不变）✅；ws 槽 [cid,i] 释放协议不变 ✅；causal 裁剪恒等式（single_kv_block 下 eff_kv_iters≡1，无运行期变异）✅
unknown_or_false: none
plan: repair-kernel-first（新增 parse-time 分支路径，探针位级验证后启用）

[#R3-1] **single_kv_block 任务批处理（Form B）**（参考: flash_attn_optimize.md §1 L67-L100、§7 L484-L497；DESIGN §10.3 候选“微任务合并”；debug_log case14 根因节 L265/L287）：**适用（首位）** — case14 归因证实协议延迟主导（pipe 全空转）；把 8 个单块任务折进一个 SEM 协议轮（w1/w2/w3 槽 i ↔ task i），C 连发 8×GEMM1 时 V 已可消费槽 0（CROSS_INTERVAL=1 逐槽放行），C/V 从乒乓变重叠；覆盖 case12/13/14/16/17。
[#R3-2] **Form A k/v L1 双缓冲**（参考: optimization-guide.md §2.2 L92/§2.3 L293-L321；cube_optimization_path.md §2.2 L75-L139；DESIGN §10.3-③/§6.3“留 Stage 3”）：**适用** — case7 AIC wait_id2=606us 直接证据（MTE2 单缓冲串行停顿）；L1 预算 P2 96→160KB / P1 56→88KB ≤511.75KB ✅。⚠ cube_path §2.2 L107“预取双缓冲无净收益”为纯 GEMM L0 级语境（小 tile flag 开销占比高）；本例 32KB 大 tile + 606us 停顿，量级不同——实验裁决，劣化即回滚。DB-ANALYSIS 实施前补。
[#R3-3] **w1 dtype 瘦身 fp32→dtype**（参考: DESIGN §10.3-① L1898-L1901；optimization-guide.md §2.16）：**适用（探针门先行）** — case7 AIV vec 71% 近饱和，V 侧双舍入首回合 2×cast 可被 fixpipe cast 吸收（RNE 等价探针 + fp16/bf16 双 dtype + 位级锚复核三门）；w1 GM 流量减半（C FIX 32→16KB/迭代、V MTE2 16→8KB/AIV/迭代）+ io_buf 可 fp16 化（UB -16KB）。红线：r16(qk) 栅格落点从 V 移到 FIX，必须位级等价（§9.3.3-⑦）。
[#R3-4] **CROSS_INTERVAL 1→2**（参考: flash_attn_optimize.md §1 L99“推荐 2-4”；optimization-guide.md §3.2 L1644“最大调节到 2”；DESIGN §10.3-④）：**适用（低成本实验）** — 减少 SEM 信号频率；与 #R3-1 批深交互，排其后。guide 上限 2 与 flash_attn 推荐 2-4 冲突 → 以 2 为界实测。
[#R3-5] **NUM_STAGES 8→12/16 扫描**（参考: flash_attn_optimize.md §1 L98“推荐 8-16”；DESIGN §10.3-④/§6.3 降级旋钮）：**适用（旋钮，排结构项后）** — GM ws 足迹 P2 15.7MB@8→23.6@12→31.4@16（R14 L2 驻留风险，实测裁决）。
[#R3-6] **Form B barrier 细化 + k/v L1 双缓冲**（参考: performance-antipatterns.md“冗余全局同步”L154-L194；flash_attn_optimize.md §2 L104-L191（Form A 已验证 flag 矩阵可移植）；cube_optimization_path.md §3.1 gemm_v0 限制）：**适用（#R3-1 后）** — case9 AIC pipe busy 之和 ≈95% 零重叠证据；保留 gemm_v0（内部 L0 管理），PipeBarrier<PIPE_ALL>×7 → 定向 MTE2↔MTE1/M↔FIX flag + k/v/L0C 双槽；⚠ 实施前须核对 gemm_v0 模板内部 event id 占用避免冲突（common.h 生成码检查）。
[#R3-7] **Form B→A 扩展 D=256 mma 化（K-split + N-split，L0B 共池）**（参考: DESIGN §10.3-⑤；flash_attn_optimize.md §3）：**暂缓（条件触发）** — case9 若经 #R3-6 仍 <0.4x（M1 线）再启动；工程量大（l0b_s[2,128,128] 与 l0b_o[2,128,128] 共池顶格 64KB ⚠、GEMM1 双 k-chunk init=True/False、GEMM2 双 D-half）。
[#R3-8] **多块 decode 扁平 (task,kv) 批处理**（参考: #R3-1 推广；flash_attn_optimize.md §1）：**暂缓（预算允许时）** — 覆盖 case10/11/15/18（kv_iters≥2 decode）；V 侧任务边界状态复位/中途写回复杂度高一档，先以 #R3-1 验证批处理收益模型。
[#R3-9] 反模式清单 Expert 版全量扫描（767 行）：**完成** — 逐项：关注项 A/B launch 核数 ✅已 Fixed Core（L143）；Vector 逐行 for ✅无（全 T.tile 2D）；**冗余全局同步 → Form B barrier_all 命中（=#R3-6）**；指令拼接未融合 → phase2 mul+add 形似 mul_add_dst，但融合=FMA 单舍入破坏位级锚（§10.3-⑥ 警示同类）→ **证据性放弃**；tile 过小 → decode bM=16 为分形下限、prefill bN=128 为 L0B 硬限（session-2 知识）✅；CV overlap ✅已手动批量流水；纯 AIV memory-bound 未双 buffer ✅io_buf 双槽；多行 tile 逐块归约 ✅单次 reduce dim=-1；正交轴串行 ✅无；AUTO_CV_COMBINE 误分核 n/a（Expert）。guide 其余节：2.1 Split-K 不适用（attention 非 K 切分 GEMM）；2.4 Full-Load 不适用（Q 已任务级驻留 Form A；K/V 跨任务无复用）；2.5 小块合并 不适用（跨任务 Q/K/V GM 不连续）；2.6/2.13 已 2D 化；2.7 axpy 同 mul+add 放弃理由；2.8/2.15/2.17 无稀疏/正交/重排形态；2.9 已 Fixed Core；2.10 Expert 四 config 锁死；2.11 UB 最紧 116/191.75KB 但 tile 扩面被 L0B 硬限否决；2.12 host 零张量操作 ✅；2.14 无 D2H；2.16 fp16/bf16 原生 ✅。
[#R3-10] case14 vid1 空半区减负（线索 1 备选）：**证据性放弃（并入 #R3-1 裁决）** — wait_id4/id0 显示 vid1 与 vid0 读数几乎相同（两侧同为协议等待而非计算），组聚合 SEM 语义下 vid1 无法退出协议；计算侧 vec 仅 15%，垃圾行非主导 → 修复方向 = 协议摊薄（#R3-1），非行减负。

## S3-Part B3 [ORDER-PLAN]

[ORDER-PLAN] 实施顺序：
1. [#R3-1] Form B single_kv_block 任务批处理 — 前置依赖: 无 — 理由: 首位主攻（case14 唯一回退项恢复 ≥0.79x + case12/13/16/17 同路径收益）；纯协议重排、每槽算子序列不变（位级锚可全程监护）；布局依赖：不改 buffer 形状/dtype。
2. [#R3-2] Form A k/v L1 双缓冲 — 前置依赖: 无（与 1 正交：C 侧 Form A vs Form B）— 理由: case7/8（0.47x 大 prefill，平均分最大拖累）wait_id2=606us 直接证据；数量依赖：k_l1/v_l1 翻倍占 L1（预算复核通过），排在不动 buffer 数量的 1 之后实施以隔离归因。
3. [#R3-3] w1 dtype 瘦身 — 前置依赖: #R3-1（同触 V emitter/协议槽语义，先定批处理结构再改 dtype，避免交叉回滚）、RNE 探针门 — 理由: case7 V vec 71% 近饱和的第二瓶颈线；配置依赖：w1 fp16 化后 io_buf dtype 联动（buffer 形状不变、dtype 变）。
4. [#R3-6] Form B barrier 细化 + k/v DB — 前置依赖: #R3-1（同触 Form B C-scope 循环体，批处理定型后再细化其内部同步）— 理由: case9（0.24x，M1 未达线）零重叠证据。
5. [#R3-4] CROSS_INTERVAL=2 — 前置依赖: #R3-1（批深改变 SEM 频率语境）— 理由: 单旋钮低成本，实测裁决。
6. [#R3-5] NUM_STAGES 12/16 — 前置依赖: #R3-1/#R3-3（ws 足迹与批深耦合，结构定型后扫描）— 理由: 旋钮项。
7. [#R3-7] / [#R3-8] — 条件触发（M1 case9 线 / 预算余量），排最后。
（每轮固定流程：备份 history_version/mha_impl_r3_perf{N}.py → [ORDER-CHECK]/[IMPL]/[SELF-CHECK] → Edit 单点 → 探针/l0 + 位级锚 → msprof → [RESULT] 采纳/回滚。）

## S3-Step 4 逐项实施记录（会话 3）

### 迭代 r3-1：[#R3-1] Form B single_kv_block 任务批处理（2026-09-19）

[ORDER-CHECK] 准备实施: [#R3-1] Form B 单块任务批处理 | 前置依赖: 无（ORDER-PLAN 首位）✅ | 结论: ✅
[IMPL-#R3-1] 已阅读 flash_attn_optimize.md L67-L100（§1 批量流水：批内逐槽 set/wait、批末聚合释放）、L484-L497（§7 [NUM_CORES,num_stages] 槽位复用）、L104-L191（§2 flag 生命周期闭环）；performance-antipatterns.md L154-L194（冗余全局同步——本项不删 barrier，仅重排批结构，barrier 细化归 #R3-6）。关键约束: ①§7.2-5 C/V 批计数同式（num_tb=ceildiv(my_count,NUM_STAGES)、batch_tasks=if_then_else(remaining<NUM_STAGES,remaining,NUM_STAGES)，双侧从同一 my_start/my_count 派生，不经 ws 传递）；②SEM set/wait 位置逐槽不变（CROSS_INTERVAL 语义原样，WS1_V2C/WS3_V2C/WS2_C2V 批级、WS1_C2V/WS2_V2C/WS3_C2V 槽级）；③每槽算子序列与现行单任务路径逐 op 同序（fill 位置除外：fill(acc_o) 移入 phase2 槽首，无数据依赖，位级不变）；④分派键 = single_kv_block and not form_a（emit_v 为两 Form 共享 macro，Form A 单块 shape 必须留在 generic 路径，否则 C/V 协议失配死锁）；⑤Expert 红线：SEM ID 不新增、V 侧 SIG 管程逐槽闭环、init/destroy 共享不改。
[KERNEL-REUSE-AUDIT] 依据: Part A3 [KERNEL-REUSE-PRECHECK] | 维度及结论: 调用域不扩（host 契约/分派谓词不变，仅既有 single_kv_block 谓词下 Form B 内循环重构）✅；flag 配对同式 ✅；SEM 收支逐槽/逐批与现行同构 ✅；精度栅格逐 op 同序 ✅ | 结论: repair-kernel-first（探针位级验证后启用）
[SELF-CHECK] 本次 Edit 只涉及 [#R3-1]：①emit_v 内新增 `if single_kv_block and not form_a:` 批处理分支（125 行），原任务循环整体 +4 缩进进 else（149 行逐行比对 identical-modulo-indent ✓）；②Form B C-scope 新增 `if single_kv_block:` 批处理分支（55 行），原循环进 else（52 行比对 ✓，ruff format 对 eff_kv_iters 长行重排 1 处，语义不变）；③Form A C 循环 107 行逐字节未动 ✓；④host 包装/allocations/SEM-SIG ID/init/destroy 未动 ✓；⑤ruff check + format clean ✓；⑥标识符零损坏（§ 检查 0 行）✓。备份 = history_version/mha_impl_r3_perf1.py（md5 d71ba4c3 = 改前基线）。
[RESULT-#R3-1] 优化点: Form B single_kv_block 任务批处理（8 任务/协议轮，槽 i ↔ task my_start+kb*8+i；C/V 同式 num_tb/batch_tasks；分派键 single_kv_block and not form_a）| 精度: **pass**（probe_r31: 9/9 批处理路径 shape 对旧版 BITEXACT + golden 阻塞口径全过（含 case14 全尺寸 MARE_abs=1.22e-4 余量 80x、zeros 位级 0.0、causal decode/MTP/bf16/D64/tail_skv100/d256 单块变体）+ 5/5 generic 守卫 BITEXACT；l0 10/10；位级锚 10/10 BITEXACT vs Developer 终态 + R11 压力 nondet=0；/tmp/opencode/r3_probes/probe_r31.log、r3_l0_iter1.log、r3_anchor_iter1.log）| 途中缺陷 1 处已修复：批处理 V phase1 的 task/bx/s0 初版定义在首个 need_mask parse-time 块内 → 第二个 need_mask 块跨块引用 "Undefined variable: s0"（rev2-A1 教训重现，causal shape 触发；修复 = 定义上移循环体顶层，探针复核通过）| 性能(msprof median us): **case14 746.08→530.73 (-28.9%)**、**case16 763.42→646.84 (-15.3%)** | 守卫: case10 40.53→39.90 (-1.6%) / case9 1398.35→1380.34 (-1.3%) / case7 928.00→927.11 (-0.1%)（噪声带内，generic 路径 IR 未变 ✓）| 对比: case14 恢复加速比 ≈**1.00x**（530.7us vs CANN baseline ≈530us；原 0.71x，硬约束目标 ≥0.79x **超额达成**）；case16 ≈0.68x（原 0.58x）| 采纳: **YES**（新基线 = r3_iter1；备份链 history_version/mha_impl_r3_perf1.py=改前；证据 msprof_r3_iter1/*.json）

### 迭代 r3-2：[#R3-3] w1 dtype 瘦身 fp32→dtype（ORDER-PLAN 数据驱动修订：提前）（2026-09-20）

**ORDER-PLAN 修订记录**：#R3-2（Form A k/v L1 双缓冲）原定第 2 位；msprof op 归因细化后发现 case7 AIC wait_id2=606us 的构成 = SIG_V_L1 等待 + CrossCoreWaitFlag(2)（等 V phase1 交付 w2）二者共用 event id 2，而 MTE2 busy 仅 466us 为前者上界 → 大头为**等 V**；且 AIV vec_ratio 71%（647us）近饱和 = 真瓶颈。k/v L1 DB 只解 C 侧次级停顿（GEMM2 批内 V 预取窗口小），而 #R3-3 直接减 V 侧 vector 工作量（省 2 次 [hM,bN] cast）+ w1 GM 流量减半（C FIX 写 + V MTE2 读）。故 **iter r3-2 = #R3-3**，#R3-2 顺延（视 #R3-3 后 case7 剩余停顿再裁决）。依赖检查：#R3-3 前置 = #R3-1 ✅（V emitter 批处理分支已定型，两处同步修改）。

[ORDER-CHECK] 准备实施: [#R3-3] w1 dtype 瘦身 | 前置依赖: [#R3-1 ✅（emit_v 双分支结构定型）] | 结论: ✅
[IMPL-#R3-3] 已阅读 DESIGN §10.3-①（L1898-L1901：隐式首舍入=fixpipe cast，三门 = RNE 等价探针 + 单块位级锚复核 + fp16/bf16 双 dtype cast 验证）、§3.2.1-⑦（双舍入栅格链 w1 fp32 主线 → 落点迁移到 FIX，栅格语义 r16(qk) 不变）、§9.3.3（栅格对齐论证）；optimization-guide §2.16（dtype 硬件路径）。关键约束: ①r16(qk) 必须在 fixpipe cast 处位级等价于 V 侧 rr_ub round-trip（RNE）——kernel 级 bit-diff 探针（fp16+bf16 多块/单块全形态）为最强等价验证，失配则回退；②io_buf dtype 联动 fp16 化，io_buf→work_ub widening copy（fp16→fp32 UB→UB，先例 = rr_ub→work_ub）无损；③io3_is_alias 必须强制 False（io_buf 变 dtype 而 w3 仍 fp32，D==bN 别名会 dtype 失配——UB 预算复核：P2 ≈116→132KB、D1 ≈106.6→98.6KB、D3 ≈30→34KB，全部 <191.75KB ✅）；④L0C(fp32)→GM(dtype) T.copy = fixpipe cast 路径，编译探针先行验证支持性；⑤溢出边界：域内 |qk| 有界（valrange_l |score|≤2264 < fp16 max 65504），fixpipe 饱和模式差异仅在域外 inf/NaN 探针可见（Boundary 非阻塞，如实记录）。
[SELF-CHECK] 本次 Edit 只涉及 [#R3-3]：workspace_1 签名 ×2（accum_dtype→dtype）、io_buf alloc ×2（→dtype）、io3_is_alias→False（dtype 失配禁别名）、emit_v 两分支首回合 round-trip 删除（r16(qk) 落点迁移至 fixpipe）、注释/docstring 同步；Form A/B C 侧 copy 语句零改动（模板 <T1,T2> 自动 cast）；SEM/SIG 矩阵零改动 ✓。ruff clean ✓。
[RESULT-#R3-3] 优化点: w1 dtype 瘦身 fp32→dtype（r16(qk) 落点 = fixpipe cast）| 精度: **pass**（RNE 等价探针 11/11 BITEXACT vs r3_perf2——fp16/bf16 × 单块/多块 × batched/generic × 大幅值域（|score|~2264 饱和域 max_abs=0.0）全形态位级等价，三门（RNE 探针/单块锚/双 dtype）全过；l0 10/10；位级锚 10/10 BITEXACT vs Developer 终态；R11 压力 PASS；/tmp/opencode/r3_probes/probe_r33.log、r3_l0_iter2.log、r3_anchor_iter2.log）| 性能(msprof median us): case7 927.11→**935.99 (+1.0%)**、case9 1380.34→1364.11 (-1.2%)、case14 530.73→526.65 (-0.8%)、case16 646.84→635.11 (-1.8%)、case10 39.90→40.03 (+0.3%)、case1 12.91→12.83 (-0.6%)、case20 31.30→30.55 (-2.4%) | 对比: **全部 7 shape 处于 ±3% 噪声带内，无一项可证实收益** | 采纳: **NO — 回滚**（rollback_reason: 性能无变化规则；已回滚至 history_version/mha_impl_r3_perf2.py，md5 核验还原）| **归因修正（重要知识）**：①case7 V 侧 vector 指令数削减（-2/[hM,bN] 大 op + w1 MTE2 减半 + FIX 减半）wall 零响应 → V vec 吞吐非 case7 wall 主约束；②wait_id2=606us 构成再判读：SIG_V_L1（V 矩阵 GM→L1 单缓冲装载停顿，intra-core id2）与 CrossCoreWaitFlag(2)（等 V 交付 w2）共用 event id 2 不可分辨，但 #R3-3 证明 V 处理速率未卡 wall → 606us 大头应是 **C 侧 V/K 矩阵装载链停顿**（单槽 k_l1/v_l1：mma(i) 等 loadV(i)，loadV(i+1) 等 MTE1 消费 V(i)——MTE2 与 M 串行链）；③fixpipe RNE == V 侧 vcast RNE 已实证（位级），该等价知识可复用于后续任何 dtype 瘦身重试；④w1 瘦身对 GM ws 足迹（P2 15.7→12.6MB）与 L2 驻留的潜在收益在本轮 wall 上不可测——若未来带宽压力场景（更大 NUM_STAGES）可凭探针结论低成本重试。

### 迭代 r3-3：[#R3-2] Form A k/v L1 双缓冲（批内预取 + 批首预载）（2026-09-20）

[ORDER-CHECK] 准备实施: [#R3-2] Form A k/v L1 双缓冲 | 前置依赖: [#R3-1 ✅ 已采纳；#R3-3 已回滚（归因修正后本项升为 case7 主攻）] | 结论: ✅
[DB-ANALYSIS] Q1 循环内有 MTE3？C-scope Form A 无 MTE3（L0C→GM 走 FIX；双缓冲仅涉 MTE2/MTE1 管程）。Q2 跨迭代累加器？k_l1/v_l1 非累加器；mma 每迭代 init=True 独立。Q3 同步方式：MTE1↔MTE2 有向 pipe pair flag，K 槽 id 0/1、V 槽 id 2/3（P→4、Q→5 重排，pair 空间 0-7 无冲突）；令牌守恒已推演（每槽稳态 free+1 结余 → init ×6 / destroy ×6 闭环；奇偶 batch_iters、多外层批、跨任务边界全部平衡——尾批不预取不产令牌）。
[IMPL-#R3-2] 已阅读 optimization-guide.md §2.3 L293-L321（首轮预取→正式循环三段结构）、cube_optimization_path.md §2.2 L75-L139（整体 DB vs 预取 DB 权衡——本项为 L1 级 GM→L1 预取，32KB 大 tile，606us 停顿证据，与其 L0 级纯 GEMM 否定结论语境不同，实验裁决）、flash_attn_optimize.md §2 L104-L191（pipe-pair ownership 闭环）。关键约束: ①k_l1/v_l1 → [2,bN,D]，make_nz/zn_layout 前导槽位维透传（ascend_layout.py L67-75 `*args[:-2]` 结构支持；无 annotate 3D L1 仓内先例 → bit-diff 探针把关）；②预载 K(0)/V(0) 置于批首 cross-SEM wait 之前（K/V 为 GM 输入，与 ws 释放无关——MTE2 装载与 SEM 等待重叠，纯收益无语义变更）；③预取段 runtime if（i+1<batch_iters）内 wait+set 同进同出（§7.2-5 配对规则）；④L1 预算 P2 96→160KB / P1 56→88KB ≤511.75 ✅；⑤纯调度变更零算术变更 → 对 r3_perf2 全形态 BITEXACT 为验收门；⑥Form B / V-scope / host 零触碰。
[SELF-CHECK] 本次 Edit 只涉及 [#R3-2]：①SIG id 重排（K:0/1、V:2/3、P:4、Q:5——MTE1↔MTE2 pair 空间 0-7 无冲突；Form B 无 SIG 使用，不受影响）；②Form A k_l1/v_l1 → [2,bN,D]（annotate 调用不变，layout 前导维透传）；③init/destroy K/V ×2 槽对称（+2/+2）；④GEMM1/GEMM2 批重构 = 批首预载（置于 cross-SEM wait 前，K/V 为 GM 输入与 ws 释放无关）+ 批内 runtime-guarded 预取（wait+set 同进同出）；⑤消费侧 k_l1[side]/v_l1[side] 切片 + SIG_+side；⑥Form B 两分支/V-scope/host/L0 池/ws 签名零触碰（grep 复核 Form B k_l1/v_l1 仍 2D ✓）；⑦令牌守恒推演：偶批(8)/奇批(3)/混合批序列（eff=10→8+2）每槽批末结余 free+1、ready 0 → 跨批跨任务平衡 ✓。备份 = history_version/mha_impl_r3_perf3.py。
[RESULT-#R3-2] 优化点: Form A k/v L1 双缓冲（[2,bN,D] NZ/ZN 前导槽位维 + 批首预载置 SEM wait 前 + 批内 runtime-guarded 预取；SIG id 重排 K:0/1 V:2/3 P:4 Q:5）| 精度: **pass**（probe_r32: 10/10 对 r3_perf3 BITEXACT——覆盖偶批/奇尾批(1..7)/多外层批(8+4)/causal 裁剪变异/单块退化/P1 分池/P2 共池/bf16/(F,T) 尾钳位/大网格多任务/Form B 双守卫 + golden 阻塞口径全过；l0 10/10；位级锚 10/10 BITEXACT；R11 压力 PASS；/tmp/opencode/r3_probes/probe_r32.log、r3_l0_iter3.log、r3_anchor_iter3.log）| 性能(msprof median us): case7 927.11→**931.83 (+0.5%)**、case20 31.30→30.62 (-2.2%)、case1 12.91→12.93 (+0.2%)、case16 646.84→648.12 (+0.2% Form B 守卫) | 对比: **全部 ±3% 噪声带内，无可证实收益** | 采纳: **NO — 回滚**（rollback_reason: 性能无变化规则；回滚至 r3_perf3，md5 1d3a261c 核验还原）| **归因知识（两轮证伪后的收敛判读）**：case7 wall≈930us 对 ①V 侧 vector 指令削减（#R3-3）、②C 侧 K/V 装载链重叠（#R3-2）均零响应；两侧 busy（V vec 647us / C 管线和 1124us 部分重叠）均 < wall；每迭代剩余 ~0.5us 不可解释间隙 ≈ 跨核 SEM 会合延迟量级 → **主嫌疑收敛为 CROSS_INTERVAL=1 的逐迭代 SEM 会合延迟链（latency-bound）**；备选假设 = ws GM/L2 流量墙（但 #R3-3 w1 减半零响应削弱该假设）。下一项 #R3-4（CROSS_INTERVAL=2）为两假设间的判决性实验。附带确认：3D [2,bN,D] L1 + NZ/ZN annotate 前导槽位维机制**可用且位级安全**（10/10 BITEXACT）——该机制知识入库，供未来需要 L1 双缓冲的场景直接复用。

### 迭代 r3-4：[#R3-4] CROSS_INTERVAL 1→2（判决性实验）（2026-09-20）

[ORDER-CHECK] 准备实施: [#R3-4] CROSS_INTERVAL=2 | 前置依赖: [#R3-1 ✅ 已采纳（批处理路径同用该常量，公式两侧同式自动守恒）；#R3-3/#R3-2 已回滚（本项为两假设间判决实验，ORDER-PLAN 数据驱动提前）] | 结论: ✅
[IMPL-#R3-4] 已阅读 flash_attn_optimize.md L99（cross_interval 推荐 2-4）、optimization-guide.md §3.2 L1636-L1662（核间同步优化：同步间隔最大调节到 2；实施后必须验证收益，无收益立即回退）。关键约束: ①set 公式 `(i+1)%CI==0 or i==batch_iters-1` 与 wait 公式 `i%CI==0` 在 C/V 两侧逐字同式（常量共享，§7.2-5 自动满足）；②令牌收支推演：batch=8 → set@{1,3,5,7}×4 ↔ wait@{0,2,4,6}×4 ✓；batch=3（奇尾）→ set@{1,2}↔wait@{0,2} ✓；batch=1 → set@{0}↔wait@{0} ✓；③纯调度变更（SEM 发放/等待节奏），每槽算子序列不变 → BITEXACT 验收门；④启动延迟 +1 槽（V 首 wait 需 C 完成 slot1）为已知权衡。
[SELF-CHECK] 本次 Edit 只涉及 [#R3-4]：CROSS_INTERVAL 常量 1→2 单行（set/wait 公式与 kernel 结构零改动）✓。
[RESULT-#R3-4] 优化点: CROSS_INTERVAL 1→2（判决性实验：SEM 会合延迟假设 vs 每迭代固定开销假设）| 精度: **pass**（probe_r34: 8/8 BITEXACT vs r3_perf4——批 8/4/3/2/1 全奇偶 × causal 裁剪 × 批处理单块 × generic × P1/P3 × bf16 + golden 阻塞口径全过；l0 10/10；位级锚 10/10 BITEXACT；R11 压力 PASS；/tmp/opencode/r3_probes/probe_r34.log、r3_l0_iter4.log、r3_anchor_iter4.log）| 性能(msprof median us): case7 927.11→**917.42 (-1.0%)**、case9 1380.34→**1413.32 (+2.4%)**、case14 530.73→527.34 (-0.6%)、case16 646.84→640.18 (-1.0%)、case10 39.90→40.93 (+2.6%)、case1 12.91→13.08 (+1.3%) | 对比: 6/6 shape ±3% 噪声带内且 case9/case10 方向偏劣 → **无可证实收益** | 采纳: **NO — 回滚**（rollback_reason: 性能无变化规则 + guide §3.2"无收益立即回退"；已回滚 CROSS_INTERVAL=1，md5 1d3a261c 与 r3_perf3 一致核验）| **判决结论**：SEM 会合频率减半对 case7 wall 零响应 → **延迟链假设证伪**。三轮证伪汇总（V 指令削减/C 装载重叠/SEM 频率均零响应，ws 流量减半亦零响应）→ case7 wall≈920-930us 的约束 = **每 KV 块迭代的固定结构成本**（两侧 scalar 发射链 + 三段 ws GM 往返的不可重叠部分 × 461 iter/核），单点调度/流量/频率手段已全部穷尽；进一步逼近需减少迭代数本身（bN=256 受 L0B 64KB 硬限：双槽 [2,128,256]=128KB ✗，单槽牺牲 mma/FIX 乒乓、k-split 增加 MTE1 指令数——收益不确定且工程量大）或设计级变更（Form A/B 结构重组、迭代粒度重定义）。**触发中止条款②：连续 3 次无提升（#R3-3/#R3-2/#R3-4），会话转入收口。**

### 迭代 r3-5（收口） — 全量验证 + cann-bench 复评（2026-09-20T01:00+08:00）

- 中止判定：**触发条款②（连续 3 次无提升：#R3-3 +1.0% / #R3-2 +0.5% / #R3-4 -1.0%，均 <3% 且逐项回滚）**；迭代 4/10 用毕，采纳 1 项（#R3-1）。
- `--level all` exit 0 "Test Passed!"：L0 10/10 + L1 19/19 + L2 12/12 正确拒绝 + Boundary 6/6（0 WARN）+ 哨兵 4/4 PERF_PASS（kernel_path 6.3s ≤ 120s 门；/tmp/opencode/r3_all_final.log）。位级锚终态 10/10 BITEXACT vs Developer 终态。
- 提交副本同步（md5 1d3a261c 一致）+ 评测脚本编译安装（installed md5 校验一致）。
- cann-bench 复评 **cann_final_eval_20260920_005958**：20/20 pass、综合 67.88→**68.50**（+0.62）、性能分 17.88→**18.50**、平均加速比 0.6745→**0.7017**（+4.0%）。**case14 0.71→0.995（回退修复，目标 ≥0.79 超额）**、case17 0.69→0.916、case16 0.58→0.678、case12 0.72→0.785、case13 0.71→0.766；case7/8/9 ≈持平（0.466/0.467/0.240）；case1/19/20 -0.02~-0.04（未触碰路径评测方差，如实记录）。decode 不回退约束全核验通过（case10 0.903 在 0.92 现状线 3% 噪声带内持平，路径 BITEXACT + 本地 msprof 改善佐证）。回退条款不触发（68.50>65.05 ∧ 0.70x>0.55x）。
- ruff check / format：clean（947 行终态）。
- 最终报告：perf_tuning/final_report.md 会话 3 章节（§S3-1~S3-6）；备份链 r3_perf1..4 完整；baseline_r3.json + msprof_r3_*/*.json 证据链完整。
- 会话 3 状态总表：

| # | 优化点 | 状态 | 关键结果（msprof median / cann-bench） |
|---|--------|------|----------------------------------------|
| R3-1 | Form B single_kv_block 任务批处理 | ✅ 采纳（r3-1） | case14 -28.9%（0.71→0.995x）/ case16 -15.3%（0.58→0.678x）/ case12/13/17 +0.06~+0.23x |
| R3-3 | w1 dtype 瘦身（fixpipe r16） | ❌ 回滚（r3-2，无变化） | 7/7 shape 噪声带内；RNE 位级等价知识入库 |
| R3-2 | Form A k/v L1 双缓冲 | ❌ 回滚（r3-3，无变化） | case7 +0.5%；3D L1 annotate 机制知识入库 |
| R3-4 | CROSS_INTERVAL=2 | ❌ 回滚（r3-4，无变化+偏劣） | 判决实验：SEM 延迟假设证伪 |
| R3-6 | Form B barrier 细化 | ⏸ 下一会话首位候选 | case9 归因就绪（AIC pipe busy 和 ≈95% 零重叠） |
| R3-7 | Form B→A D256 mma 化 | ⏸ 条件触发（case9 M1 线） | 工程量大（K/N-split + L0B 共池顶格） |
| R3-8 | 扁平 (task,kv) 批处理 | ⏸ 下一会话次位候选 | case14 收益模型已验证，推广至 case10/11/15/18 |
| R3-9 | 反模式全量扫描 | ✅ 完成 | Form B barrier_all 唯一命中（=R3-6）；mul+add 融合证据性放弃（FMA 单舍入破位级锚） |
| R3-10 | case14 vid1 空半区减负 | ❌ 证据性放弃 | wait 读数 vid0/vid1 对称 → 协议等待主导非计算；已并入 R3-1 解决 |

---

# 会话 4（perf_session=4，2026-09-20，Expert rev3 轨第二轮；迭代预算 10，备份链续编 r3_perf{N} N≥5）

> 对象：Expert 版 mha.py（会话 3 终态 = #R3-1 采纳态，md5 1d3a261c，cann-bench 68.50 / 平均 0.7017x）。
> 目标：**冲击里程碑 M1（平均加速比 ≥0.75x 且 case9 ≥0.4x）**；用户目标 1.0x 不变。
> 主攻两条会话 3 移交线索：#R3-6 Form B barrier 细化（首位，case9 0.24x 归因就绪）+ #R3-8 扁平 (task,kv)
> 批处理推广（次位，decode 族 case10/11/15/18）。§10.3-⑤ Form B→A mma 化**设计级禁止实施**（仅积累证据）。
> 硬约束继承会话 3：decode 逐 case 不回退（case10 ≥0.90x / case14 ≥0.99x / case20 ≥0.82x 现状线，3% 噪声带）；
> case7/8/9 只升不降；整体跌破 65.05 → [PERF_BLOCKED]；位级锚（vs Developer 终态 BITEXACT）为每项快速回归门。
> 噪声带 3%；计时口径 = msprof timeline op_summary Task Duration median（warmup 5 + iters 20），继承 final_report §7。

## S4-Step 1 基线（baseline_r4.json 已落地；证据 msprof_r4_base/case9.json）

| shape | case1 | case7 | case9 | case10 | case11 | case14 | case16 | case20 |
|-------|-------|-------|-------|--------|--------|--------|--------|--------|
| msprof median (us) | 12.93 | 927.11 | **1379.56（本会话复测）** | 39.90 | 349.63 | 530.73 | 646.84 | 31.30 |

- case9 复测 1379.56us（stdev 1.73）vs 会话 3 记录 1380.34us → Δ0.06%，机器状态一致 ✅；其余 case 复用 baseline_r3_iter1.json（终态未变，md5 链核验：mha.py = r3_perf2/3/4 = 提交副本 = 1d3a261c）。
- 精度基线（本会话新鲜复验）：`--level l0` 10/10 PASS（/tmp/opencode/r4_l0_base.log）；位级锚 10/10 BITEXACT vs Developer 终态 + R11 竞态压力 4 配置 ×12 复跑 nondet=0（/tmp/opencode/r4_anchor_base.log）。
- NPU 环境：torch_npu available、cube_core_num=20、set_env.sh + sqlite3 LD_LIBRARY_PATH 每 shell 生效。

## S4-Step 2 算子类型（继承 S3-Step 2 复核 + 本会话生成码再核验）

- CV 融合混合型（ASCEND_IS_AIC + ASCEND_IS_AIV，Expert cv_1_2 双 scope）。本会话重新生成三份源码存档
  /tmp/opencode/r4_src/FormB_case{9,11,14}.cc：Form B C-scope `T.barrier_all()` → `AscendC::PipeBarrier<PIPE_ALL>`
  （case9 generic 路径静态 8 处：任务级 Q 载入 1 + GEMM1 3/iter + GEMM2 4/iter）；cross-flag 全部生成为
  `AscendC::CrossCoreSetFlag<0x2, PIPE_{FIX|MTE2|MTE3}>(sem)`（pipe 作用域）+ `AscendC::CrossCoreWaitFlag(sem)`（scalar 阻塞）。
- **#R3-6 强制前置核对（gemm_v0 内部 event id 占用，common.h L1183-L1371 源码级）**：
  - `L0AB_EVENT = 0`；内部占用对：MTE2_MTE1(0)（prologue）、FIX_M(0)（prologue）、M_MTE1(0/1)（ping-pong）、
    MTE1_M(0/1)（ping-pong）、MTE1_MTE2(0)（epilogue）、M_FIX(0)（epilogue）。
  - 每个 SetFlag 在同调用内有配对 WaitFlag（prologue/epilogue/循环尾 L1364-1370）→ **调用级自平衡、无跨调用令牌残留**。
  - Form B C-scope 生成码**零用户 SetFlag/WaitFlag**（r4_src 复核）→ 纯移除 barrier_all 不新增任何用户 flag，
    **event id 冲突风险 = 0**（这是会话 3 移交的前置检查项，结论：安全）。
  - gemm_v0 黑盒边界序：prologue 保证 前序 MTE2（GM→L1 载入）→ MTE1（L1→L0）、前序 FIX（L0C→ws 写出）→ M（mma）；
    epilogue 保证 MTE1（L1 消费完）→ 后续 MTE2（可覆写 L1）、M（结果就绪）→ 后续 FIX（可写 L0C）。
    **barrier_all 所在四个段界的全部数据竞争类均被覆盖**；同 pipe 指令天然有序；跨核由 SEM pipe 作用域协议覆盖（位置不动）。
  - 辅证：copy_gm_to_l1 / copy_l0c_to_gm 设备函数无隐式 PIPE_ALL 排空（common.h L55-L106/L187-L213；
    gm_to_l1 内部仅 need_clear 路径自带 PIPE_MTE2 作用域 barrier，自包含）；Form A 全程无 barrier_all 纯用户 flag
    已是同框架正确性先例（rev3-attempt1 起 20/20 精度通过）。

## S4-Step 3 Part A4 优化点清单（会话 3 移交线索逐项适用性确认；证伪项不得重做）

[KERNEL-REUSE-PRECHECK]
candidate: #R4-1 Form B barrier_all 纯移除（=#R3-6 第一阶段；performance-antipatterns.md"冗余全局同步" L154-L194）
expanded_domain: 无——host 域/分派谓词/buffer 形状/dtype/SEM 位置全部不变，仅删除 Form B C-scope 的 8 处 PipeBarrier<PIPE_ALL>
compatibility_dimensions: ①gemm_v0 内部 event id 占用=已核对（S4-Step 2，冲突风险 0）；②段界数据竞争覆盖=已核对（prologue/epilogue 四类 hazard 全覆盖 + SEM pipe 作用域）；③C/V 协议同构=TRUE（SEM set/wait 位置零改动，§7.2-5 天然满足）；④每槽算子序列不变=TRUE（纯调度变更 → BITEXACT 验收门）；⑤被 barrier 掩盖的潜伏竞态=风险项（probe + 锚 + R11 压力 ×12 复跑把关）
unknown_or_false: none（⑤为验证门而非未知维度）
plan: direct-reuse

[KERNEL-REUSE-PRECHECK]
candidate: #R4-3 扁平 (task,kv) 批处理 decode 推广（=#R3-8；flash_attn_optimize.md §1 L67-L100 批量流水 + #R3-1 已验证收益模型）
expanded_domain: Form B generic 路径中 S≤16（decode/MTP，s_blocks≡1）子域改走扁平调度；prefill D=256（case4/9，s_blocks>1 且 causal 逐任务变长）保持 per-task generic 不动
compatibility_dimensions: ①eff_kv_iters 均匀性=TRUE by 证明（decode s_blocks=1 → s0≡0 → causal 裁剪公式 min(15+S_kv-S, S_kv-1)=S_kv-1 → eff≡kv_iters 编译期常量，C/V 同式）；②flat 索引推导 C/V 同式=计划（total=my_count*kv_iters、num_fb=ceildiv(total,8)、batch=if_then_else(rem<8,rem,8)、task_local=f//kv_iters、kv=f%kv_iters，双侧从同一 my_start/my_count 派生）；③V 侧任务边界状态复位/写回=runtime if（kv==0 fill m_i/sumexp；kv==kv_iters-1 归一化+写回）——纯 V 无 flag 段或 SIG_S_HALF 闭环成对同进出（§7.2-5 允许模式），双 vid 条件同值（vid 无关索引派生）；④row_thresh/rows_vid 任务不变=TRUE（s0≡0、valid_m 编译期常量）可 hoist；⑤r_factors/sumexp_is 槽位语义=TRUE（round 内 phase1→phase2 同槽传递，跨 round 任务状态经 acc_o/m_i/sumexp UB 常驻）；⑥位级=预期 BITEXACT（每槽算子序列与 generic 逐 op 同序，fill 时点等价性已推演：kv==0 槽 r=exp(-2^30-m)=0 路径与任务起始 fill 位级同值）
unknown_or_false: none
plan: repair-kernel-first（新增 parse-time 分支，探针位级验证后启用）

[#R4-1] **Form B barrier_all 纯移除**（=#R3-6 第一阶段；参考: performance-antipatterns.md L154-L194 冗余全局同步；S3-Step 3 case9 归因：AIC pipe busy 之和 ≈95% 零重叠，PipeBarrier<PIPE_ALL> ×7-8/迭代把 MTE2→MTE1→M→FIX 串成链）：**适用（首位）** — 前置核对完成（S4-Step 2）；预期 MTE2(i+1) 载入与 M(i) mma / FIX(i) 写出重叠，case9 wall 向 max(pipe busy)≈530us（mte2 40%）方向收敛；M1 case9 ≥0.4x 需 ≤844us。纯移除 = 最小改动增量；若采纳后 case9 仍 <0.4x，第二阶段候选 #R4-2。
[#R4-2] **Form B k/v L1 双缓冲 + GEMM1/GEMM2 跨迭代交织**（=#R3-6 第二阶段；参考: optimization-guide.md §2.2/§2.3；S3-3-3② 3D [2,bN,D] L1 机制知识已入库）：**条件触发** — #R4-1 后 case9 剩余停顿归因（msprof op 重采）确认 MTE2 链仍为关键路径才实施；gemm_v0 接受 3D L1 切片基址（fractal 对齐：side 偏移 = bN*D*2B 为 32B 整数倍）需探针验证。
[#R4-3] **扁平 (task,kv) 批处理 decode 推广**（=#R3-8；参考: #R3-1 已验证收益模型 case14 -28.9%；flash_attn_optimize.md §1）：**适用（次位，#R4-1 后）** — 覆盖 case10/11/15/18；协议轮次：case11 26→7、case15/18 48→12、case10 2→1；case10 已 0.903x 接近 1.0 → 纳入同一分支（改动同构、守卫监测，不为小收益单独开工）。case12/13/14/16/17 已走 #R3-1 单块批处理不受影响。
[#R4-4] **#R3-7 Form B→A D256 mma 化**（DESIGN §10.3-⑤）：**禁止实施（设计级，本会话仅证据积累）** — 若 #R4-1/#R4-2 后 case9 仍 <0.4x，记录定量证据（剩余 pipe 停顿构成 + 工程量评估）供 rev4 立项建议。
[#R4-5] case7/8 bN=256 变体（L0B 64KB 硬限下 kL0Size 档位组合）：**暂缓（M1 后预算允许时）** — 会话 3 判定"剩余候选"，Form A 结构不动为本会话纪律（隔离归因）。
[#R4-6] **会话 3 证伪项——不得重做**（继承证据）：①w1 dtype 瘦身（#R3-3，7/7 shape 噪声带 + RNE 位级等价知识已入库）；②Form A k/v L1 双缓冲（#R3-2，case7 +0.5% + 3D L1 机制知识已入库）；③CROSS_INTERVAL=2（#R3-4，6/6 噪声带且 case9/10 偏劣 + SEM 延迟假设已证伪）。
[#R4-7] 反模式清单复扫（performance-antipatterns.md 全 11 项，对 947 行终态）：**完成（继承 S3 #R3-9 全量扫描）** — 唯一残留命中 = Form B barrier_all（=#R4-1 本会话主攻）；其余项状态不变（mul+add 融合=FMA 单舍入破位级锚，证据性放弃维持）。

## S4-Part B4 [ORDER-PLAN]

[ORDER-PLAN] 实施顺序：
1. [#R4-1] Form B barrier_all 纯移除 — 前置依赖: 无（gemm_v0 event id 核对已完成 ✅）— 理由: M1 缺口主项 case9（0.24x）归因就绪（零重叠证据）；纯删除 = 最小改动、位级验收、单 Edit 完成；同时惠及全部 Form B case（generic + batched 两路径同删）。布局依赖：不改 buffer；数量依赖：不改预算。
2. [#R4-3] 扁平 (task,kv) 批处理 decode 推广 — 前置依赖: #R4-1（同触 Form B generic C/V 循环体；barrier 结构定型后再改批结构，避免交叉归因/交叉回滚）— 理由: 次级主攻，decode 族协议摊薄（case11/15/18 轮次 3.7-4x 削减）；配置依赖：无 pass_configs 变更。
3. [#R4-2] Form B k/v L1 DB + GEMM 交织 — 前置依赖: #R4-1 采纳 + msprof op 重采归因确认 MTE2 链仍关键 — 理由: 条件触发增量，数量依赖（L1 buffer 翻倍）排在不改 buffer 的 1/2 之后。
4. [#R4-4]/[#R4-5] — 条件触发（M1 判定后 / 预算余量），排最后。
（每轮固定流程：备份 history_version/mha_impl_r3_perf{N}.py（N 从 5 续编）→ [ORDER-CHECK]/[IMPL]/[SELF-CHECK] → Edit 单点 → probe（位级 vs 上一采纳态）→ l0 + 位级锚 → msprof → [RESULT] 采纳/回滚；关键节点 cann-bench 复评；收口 --level all + R11 压力。）

## S4-Step 4 逐项实施记录（会话 4）

### 迭代 r4-1：[#R4-1] Form B barrier_all 纯移除（=#R3-6 第一阶段）（2026-09-20）

[ORDER-CHECK] 准备实施: [#R4-1] Form B barrier_all 纯移除 | 前置依赖: 无（ORDER-PLAN 首位；gemm_v0 event id 前置核对已完成于 S4-Step 2 ✅）| 结论: ✅
[IMPL-#R4-1] 已阅读 performance-antipatterns.md L154-L194（冗余全局同步：替代方向"删除没有生产者/消费者依赖的同步"；检查点"Expert 模式必须通过生成的 Ascend C 代码和精度测试确认同步语义没有被破坏"——生成码核验已在 S4-Step 2 完成，精度测试=probe BITEXACT 门）；common.h L1183-L1371（gemm_v0 全体：L0AB_EVENT=0、prologue MTE2_MTE1/FIX_M 自平衡对、epilogue MTE1_MTE2/M_FIX 自平衡对、循环内 M_MTE1/MTE1_M ping-pong 0/1）；codegen_ascend.cc L2845-L2881（barrier_all→PipeBarrier<PIPE_ALL>、set_cross_flag→CrossCoreSetFlag<mode,PIPE_x>、wait_cross_flag→CrossCoreWaitFlag）。关键约束: ①不新增任何用户 flag（Form B C-scope 保持零 SetFlag/WaitFlag）→ gemm_v0 内部 event id 0/1 独占，冲突风险 0；②SEM set/wait 位置零改动（C/V 协议同构，§7.2-5 天然满足，emit_v 不触碰）；③纯调度变更（每槽算子序列/算术逐 op 不变）→ BITEXACT vs r3_perf5 为验收门；④四类段界 hazard 覆盖论证（S4-Step 2）：MTE2 载入→gemm MTE1（prologue）、gemm M→FIX 写出（epilogue）、FIX→下轮 mma L0C 复用（prologue）、gemm MTE1→下轮 MTE2 覆写 L1（epilogue）；同 pipe 有序 + 跨核 SEM pipe 作用域 = 完备；⑤删除范围 = Form B C-scope 全部 15 处 T.barrier_all()（batched 8 + generic 7... 实为 batched 8 + generic 8 含任务级 Q 1 处），Form A / emit_v 零触碰（grep 核验 Form A 本无 barrier）。
[SELF-CHECK] 本次 Edit 只涉及 [#R4-1]：①batched GEMM1 槽内 4 处删除；②batched GEMM2 槽内 4 处删除；③generic 任务级 Q 载入后 1 处删除；④generic GEMM1 3 处删除；⑤generic GEMM2 4 处删除；⑥docstring L24-26 + C-scope 段头注释同步（barrier_all 描述 → barrier-free 段界序描述）；⑦Form A C-scope / emit_v / host 包装 / SEM-SIG 矩阵 / allocations 零触碰；⑧ruff check + format；⑨备份 = history_version/mha_impl_r3_perf5.py（md5 1d3a261c = 改前基线 ✅）。
[RESULT-#R4-1] 优化点: Form B barrier_all 纯移除（15 处 T.barrier_all() 全删 = batched 8 + generic 8；依赖 gemm_v0 自平衡 prologue/epilogue pipe flags + SEM pipe 作用域协议；C-scope 保持零用户 flag，event id 冲突风险 0）| 精度: **pass**（probe_r41: **15/15 BITEXACT vs r3_perf5** —— Form B batched（decode/MTP-causal/D256-prefill 单块/bf16/zeros）+ generic（decode kv2/kv4-causal/tail-1000/D256-decode/MTP-D256-causal/prefill-D256-causal-trim/tail-300/bf16）+ Form A 双守卫，golden 阻塞口径全过；l0 10/10；**位级锚 10/10 BITEXACT vs Developer 终态**；R11 竞态压力 4 配置 ×12 nondet=0；l1 19/19；/tmp/opencode/r4_probes/probe_r41.log、r4_l0_iter1.log、r4_anchor_iter1.log、r4_l1_iter1.log）| 生成码复核: FormB_case9/case14 PipeBarrier<PIPE_ALL> 计数 8→**0**，gemm_v0 ×2 / CrossCoreSetFlag ×9 协议原样（/tmp/opencode/r4_src/*_postR41.cc）| 性能(msprof median us): **case9 1379.56→1020.73 (-26.0%)**、**case14 530.73→378.16 (-28.7%)**、**case16 646.84→565.27 (-12.6%)**、**case11 349.63→312.34 (-10.7%)**、**case10 39.90→35.34 (-11.4%)** | 守卫: case7 927.11→932.60 (+0.6%)、case20 31.30→30.84 (-1.5%)、case1 12.93→12.94 (+0.1%)——Form A IR 逐字节未动，全部噪声带内 ✓ | 对比: case9 加速比 0.240→**≈0.331x**（M1 线 0.4x 未达，缺口 -17.3%）；case14 0.995→**≈1.40x**；case10 0.903→**≈1.51x**；case16 0.678→≈0.78x；case11 0.824→≈0.92x；5 case 更新后平均加速比估算 **0.7662 ≥ 0.75**（case12/13/15/17/18 同路径未计入，实际预计 ~0.80）| 采纳: **YES**（新基线 = r4_iter1；备份链 history_version/mha_impl_r3_perf5.py=改前 1d3a261c；证据 msprof_r4_iter1/*.json）| **归因知识**：①"冗余全局同步"反模式在 Form B 的量级实证 = -26~-29%（会话 3 case9 pipe busy 和 ≈95% 零重叠归因被治疗性验证）；②gemm_v0 黑盒自同步边界（prologue MTE2_MTE1/FIX_M + epilogue MTE1_MTE2/M_FIX 自平衡对）= **可复用机制知识**：任何 gemm_v0 三段式（GM→L1 / gemm / L0C→GM）用户侧 barrier_all 均可纯移除，无需新增用户 flag；③Form A（细粒度 flag 流水）对同一物理机的守卫读数确认 case7 wall≈930us 结构成本结论不受本改动影响。

### 迭代 r4-2：[#R4-2a] prefill D=256 GEMM1 kL0Size=256 单 K-step（case9 专项）（2026-09-20）

**ORDER-PLAN 数据驱动修订**：#R4-1 采纳后 case9 复归因（msprof op，/tmp/opencode/r4_op_case9/）：AIC scalar_ratio 0.89-0.97（wall ~1000us 中 scalar_mte1_stall 739-841us ≈78%）、mte2 45-50%、mte1 32-34%、cube 17.6-19%、fix 25-27%、AIC wait_id2 122-161us；AIV vec 43-46%、wait_id0 325-360us、wait_id4 440-469us。判读：barrier 移除后 case9 剩余 wall = **gemm_v0 内部每-step scalar 链**（common.h L1338-L1360：每 (N,K)-step 含 WaitFlag<M_MTE1>→MTE1 载入→SetFlag/WaitFlag<MTE1_M>→**PipeBarrier<PIPE_M>**→mma→SetFlag<M_MTE1>，scalar 阻塞至前一 mma 排空）× 4 step/iter（GEMM1 kL0split=2 + GEMM2 nL0split=2）× ~460 iter/核。#R3-7 mma 化被禁（设计级）→ 规则内杠杆 = **减 step 数**：GEMM1 kL0Size 128→256（K=D=256 单步化；L0B B-tile = nTile(128)×256×2B = 65536B == kNumSteps=1 的 64KB 单槽预算恰好；L0A = 32×256×2 = 16KB；kL0Tail=256 整除无尾；int8 断言不适用）。GEMM2 侧 nMaxByL0B=32KB/(kL0×2B) 公式限定 nTile=128 → nL0split=2 不可参数化缩减（kL0=64 → kL0split=2×nL0split=1 = 同 2 step 无净变化，证据性跳过）。故 r4-2 = GEMM1 单步化（4→3 step/iter），#R4-3 扁平批处理顺延其后。
[ORDER-CHECK] 准备实施: [#R4-2a] prefill D=256 gemm1_kL0=256 | 前置依赖: [#R4-1 ✅ 已采纳（barrier-free 结构定型，step 链成为显性瓶颈）] | 结论: ✅
[IMPL-#R4-2a] 已阅读 common.h L1189-L1371（gemm_v0 全体：L1277-L1285 L0B/L0A 预算 static_assert——kNumSteps=1 → kL0Budget=64KB、B-tile=nTile×kL0Size×sizeof ≤ 64KB 恰好 65536；L1207-L1246 K-tail/C0 对齐断言——fp16 ELE_NUM_PER_C0=16 短路 ✓、kL0split=1 → C0 对齐断言真空 ✓；L1330 tileIdx 单步 pp=0 全 64KB 单槽）；ascend.py L418-L525（kL0Size 参数 16 倍数断言 ✓ 256%16=0）；optimization_log 会话 2 [IMPL-#10b]（kL0 档位先例：K 累加顺序不变 → GEMM 位级等价预期，L0C 跨 K-tile 顺序累加）。关键约束: ①作用域 = 仅 (S>16 and D==256)（prefill D256：case9/case4 族 + l0_d256 锚 shape + Skv≤128 单块 batched 亦命中——kL0 派生为 host 级公式，两路径共用）；decode D256（64 档）/D≤128（128/32 档）/Form A 零触碰；②位级预期：单 mma k=256 vs 双 mma k=128 链式累加——C0 块顺序累加同序 → BITEXACT 预期，probe 门裁决（若 DIFF 则按阻塞口径 + 位级锚失配根因分析纪律处理）；③L0A/L0B TBuf 65536B 生成码实测（r4_src L29/L31）→ 单槽 64KB B-tile 恰好容纳 ✓；④n_actual（valid_n 尾列）语义不变（单步循环内 kSize=256、n_actual 列窗原样）。
[SELF-CHECK] 本次 Edit 只涉及 [#R4-2a]：①gemm1_kL0 派生表达式新增 256 档（仅 S>16 and D==256 域）；②docstring Form B 行 kL0 档位描述同步；③其余零触碰（备份 = history_version/mha_impl_r3_perf6.py，md5 e80f7f65 = post-#R4-1 态 ✅）；ruff check/format clean ✅。
[RESULT-#R4-2a] 优化点: prefill D=256 GEMM1 kL0Size 128→256（单 K-step，4→3 step/iter）| 精度: **pass**（probe_r42: **12/12 BITEXACT vs r3_perf6**——变更域 prefill D256（锚 shape/多块 causal trim/尾列 n_actual/bf16/zeros/单块 batched）+ 4 项未变更域守卫全 BITEXACT，**实证：单 mma k=256 与链式 2×mma k=128 位级等价（C0 顺序累加同序，#10b 知识再确认）**；l0 10/10；位级锚 10/10 BITEXACT（含 l0_d256 变更域）；R11 压力 nondet=0；/tmp/opencode/r4_probes/probe_r42.log、r4_l0_iter2.log、r4_anchor_iter2.log）| 性能(msprof median us): case9 1020.73→**1058.92 (+3.7%)** | 对比: **超出 3% 噪声带的性能下降** | 采纳: **NO — 回滚**（rollback_reason: 性能下降规则；已回滚至 history_version/mha_impl_r3_perf6.py，md5 e80f7f65 核验还原）| **归因知识（case9 step 链杠杆的边界）**：kL0Size=256 使 kNumSteps=1 → **L0A/L0B ping-pong 失效**（common.h L1322-L1327：双槽仅在 kNumSteps>1 启用）——单步 80KB MTE1 载入（A 16KB+B 64KB）与 mma(k=256) 全串行，损失的原 2-step 重叠（MTE1(s1) ∥ mma(s0)）超过省下的 1 个 step 的 scalar 链开销（PipeBarrier<PIPE_M>+flag 往返 ~0.15us/step）→ **gemm_v0 step 数不是 case9 wall 的边际约束，MTE1/mma 重叠度才是**；后续 case9 杠杆须保 step≥2 或改走手动 mma（=§10.3-⑤ 设计级，禁）。此结论同时预警：GEMM2 kL0Size=64 变体（nL0split=1×kL0split=2，step 数不变但 A 载入减半）预期同为噪声级，证据性跳过。

### 迭代 r4-3：[#R4-2b] Form B prefill D=256 k/v L1 双缓冲（预取式，case9 专项）（2026-09-20）

[ORDER-CHECK] 准备实施: [#R4-2b] kv_db k/v L1 双缓冲 | 前置依赖: [#R4-1 ✅（barrier-free 结构定型——DB 预取依赖 gemm_v0 epilogue 的 MTE1→MTE2 释放序）；#R4-2a 已回滚（step 数非边际约束、MTE1/mma 重叠度才是 → DB 隐藏 MTE2 暴露段为数据指向的下一杠杆）；机制探针 ✅（probe_r42b_l1db_mech: 3D 无 annotate L1 [2,128,256] + 运行期 side 切片 + gemm_v0(kL0=128) + DB 节奏（消费当前槽时预取对侧槽、4 槽回绕）vs 2D 串行 **BITEXACT**，torch 参考 2.3e-5）] | 结论: ✅
[IMPL-#R4-2b] 已阅读 optimization-guide.md §2.3 L293-L321（首轮预取→正式循环三段结构）、cube_optimization_path.md §2.2 L75-L139（L1 级 GM→L1 预取 DB，32KB+ 大 tile 语境）、flash_attn_optimize.md §2 L104-L191（pipe-pair ownership；本实现**零新增用户 flag**——gemm_v0 epilogue MTE1_MTE2 已提供"全部 MTE1 消费完→MTE2 可覆写任一槽"的释放序，prologue MTE2_MTE1 提供"全部预取载入完成→MTE1 可读"的就绪序，DB 仅靠调用边界自平衡 flag 即安全，规避与 gemm_v0 内部 id 0/1 的冲突）。关键约束: ①作用域 = kv_db ≡ (S>16 and D==256 and not single_kv_block)（prefill D256 generic：case9/case4 族；L1 预算 q16+k[2]128+v[2]128+p8 = 280KB ≤ 511.75KB ✅；decode 档被 L1 预算否决（bN=512：532KB>cap）且属 MTE2 带宽墙（DB 不减字节）——证据性排除；batched 单块路径不动（case14/16 收益保护 + 无 D256 bench 命中））；②预取位置 = gemm 调用返回后（epilogue 已释放 MTE2）→ 与本次 mma 尾部 + FIX 写出重叠；V0 每轮预载置于 GEMM1 批末/WS3 wait 前（GM 输入无 ws 依赖，SEM 延迟被载入吸收）；任务首 K0 于任务头（t==0 守卫）、后续任务 K0 于前任务 GEMM2 末（t+1<my_count 守卫）、跨轮 K0 于轮末（k+1<eff 守卫）——所有运行时守卫内**纯 MTE2 copy、零 flag**（§7.2-5 允许模式）；③槽位记账：side=i%2 round-local（与 ws 槽 i 同构）；预取写 1-side，其安全性 = MTE1 pipe 顺序性（gemm(i) epilogue 排空含 gemm(i-1) 对 1-side 槽的全部 MTE1 读）；④V 侧 emit_v / SEM 矩阵 / Form A / batched 路径零触碰——C 侧纯 intra-core 调度变更，C/V 协议同构天然满足；⑤纯调度变更（数据流/算术逐 op 不变）→ BITEXACT vs r3_perf7 验收门；⑥need_clear 尾列清零语义逐 copy 保留（extent 公式与原载入逐字同形）。
[DB-ANALYSIS] Q1 循环内有 MTE3？C-scope 无 MTE3（L0C→GM 走 FIX）。Q2 跨迭代累加器？k_l1/v_l1 非累加器；acc_*_l0c 每 gemm init=True 独立。Q3 同步方式：**零新增 flag**——复用 gemm_v0 调用边界自平衡序（epilogue MTE1_MTE2 释放 / prologue MTE2_MTE1 就绪），令牌收支恒等（不产生跨调用未决令牌）。
[SELF-CHECK] 本次 Edit 只涉及 [#R4-2b]：①host 级 kv_db 谓词（S>16 and D==256 and not single_kv_block）；②Form B k_l1/v_l1 alloc 条件 3D [2,bN,D]；③generic C 分支 parse-time 双变体（kv_db 预取版 + else 原版——34 行 identical-modulo-indent 逐行比对 ✓）；④Form A / emit_v / batched 路径 / SEM 矩阵 / host 包装零触碰（区域 diff 比对 ✓）；⑤ruff clean ✓；⑥途中 ruff F841（bx_n 未用）即改即清 ✓。备份 = history_version/mha_impl_r3_perf7.py（md5 e80f7f65 = post-#R4-1 ✅）。
[RESULT-#R4-2b] 优化点: Form B prefill D=256 k/v L1 双缓冲（预取式，零新增 flag，3D 无 annotate L1 + gemm_v0 运行期切片）| 精度: **pass**（probe_r43: **15/15 BITEXACT vs r3_perf7**——变更域全拓扑（多轮 2/4-round、奇偶批、causal eff 1..32、尾列、bf16、zeros、跨任务预取链）+ 5 项未变更域守卫；机制探针 probe_r42b 3D-vs-2D BITEXACT；l0 10/10；位级锚 10/10 BITEXACT；R11 压力 nondet=0；/tmp/opencode/r4_probes/probe_r43.log、r4_l0_iter3.log、r4_anchor_iter3.log）| 性能(msprof median us): case9 1020.73→**1029.77 (+0.9%)** | op 归因（/tmp/opencode/r4_op_case9_db vs r4_op_case9）: **全 pipe 读数零变化**（mte2_ratio 0.475 vs 0.477、scalar_mte1_stall 791 vs 793us、aic_time 1010 vs 1007）——预取被完全吸收，MTE2 暴露段本就不在关键路径 | 对比: 噪声带内无变化 | 采纳: **NO — 回滚**（rollback_reason: 性能无变化规则；回滚至 r3_perf7，md5 e80f7f65 核验还原）| **归因知识（case9 三证据链收敛，rev4 立项核心证据）**：case9 wall ≈ **gemm_v0 内部每-step 串行链**（MTE1 L1→L0 传输 40-80KB/step + M_MTE1 双槽 ping-pong 等待 + 每 step PipeBarrier<PIPE_M> 排空前一 mma，×4 step/iter × ~460 iter/核；scalar_mte1_stall ≈78% wall）——①step 数削减（r4-2a kL0=256）劣化 +3.7%（ping-pong 失效损失 > step 开销节省）；②MTE2 载入隐藏（r4-2b DB）零响应（mte2 45-50% busy 已被吸收在 stall 阴影下）；③会话 3 三轮证伪（V 指令削减/C 装载重叠/SEM 频率）同向。**规则内手段穷尽**：破局需手动 mma 流水（Form A 式 L0C 双槽 + 细粒度 flag，无每-step PIPE_M 排空）= §10.3-⑤ Form B→A mma 化 = 设计级变更（本会话禁止实施，证据已足量积累供 rev4 立项）。

### 迭代 r4-4：[#R4-3] 扁平 (task,kv) 批处理 decode 推广（case10/11/15/18）（2026-09-20）

[ORDER-CHECK] 准备实施: [#R4-3] 扁平 (task,kv) 批处理 | 前置依赖: [#R4-1 ✅ 已采纳（barrier-free Form B 结构定型）；#R4-2a/#R4-2b 已回滚（case9 规则内杠杆穷尽，avg-margin 杠杆成为本会话剩余收益主线）] | 结论: ✅
[IMPL-#R4-3] 已阅读 flash_attn_optimize.md §1 L67-L100（批量流水：批内逐槽 set/wait、批末聚合释放——#R3-1 同款协议重排推广到 kv 维）、#R3-1 [IMPL]/[RESULT] 全记录（收益模型 case14 -28.9%、C/V 同式纪律、rev2-A1 跨块引用教训）；DESIGN §7.2-5（运行期守卫内 flag 成对同进出或纯 V 无 flag 段）。关键约束与结构论证: ①域 = Form B 多块 decode（S≤16 → s_blocks≡1、form_a≡False）；**eff_kv_iters ≡ kv_iters 编译期常量**（causal 亦均匀：s0≡0 → min(15+S_kv-S, S_kv-1)=S_kv-1，S≤16）→ 扁平索引 f=task_local*kv_iters+kv 双侧纯编译期常量派生，无运行期装箱；②轮次公式 C/V 同式：total_flat=my_count*kv_iters、num_fb=ceildiv(total_flat,NUM_STAGES)、batch_slots=if_then_else(rem<NUM_STAGES,rem,NUM_STAGES)；③协议轮次削减：case11 26→7、case15/18 48→12、case10 2→1（每轮 6 次 SEM 往返摊薄至 8 槽）；④**V 侧任务状态复位必须放 phase2**（轮内 phase1 全先于 phase2 执行——phase1 里在新任务起始槽复位 sumexp/acc_o 会摧毁前一任务尚未被 phase2 消费的在途状态）：phase1 仅 `if kv==0: fill(m_i,-2^30)`（m_i 为 phase1 独占状态，reduce_max 覆写语义 + m_i_prev 链与 generic 任务头 fill→copy 构造位级同值）；phase2 `if kv==0: fill(acc_o,0)+fill(sumexp,0)`（= generic 任务头 fill 的等值迁移：首次消费点相同、fill 值相同）；⑤任务尾（kv==kv_iters-1）phase2 内归一化+写回：SIG_S_HALF 闭环整体位于 runtime if 内——wait/set 4 收支同进同出平衡，条件 vid 无关（f/kv 派生自循环索引+编译期常量+my_start，双 vid 同值同分支 §7.2-5）；⑥rows_vid/row_thresh 任务不变（s0≡0、valid_m≡S 编译期常量）→ 分支顶层 hoist 一次（generic 逐任务重算的等值提升；row_thresh buffer 承载非 let 绑定，跨块合法——#3 hoist 先例）；⑦每槽算子序列与 generic 逐 op 同序（掩码条件 s0=0 代入恒等：kv*bN+bN-1 > min(offset_eff,S_kv-1)；j0=kv*bN 同值）→ BITEXACT 验收门；⑧C 侧 Q 载入移入 `if kv==0` 运行期守卫（纯 MTE2 无 flag，C 单 scalar 无配对问题；q_l1 覆写安全 = 前任务末 gemm epilogue MTE1→MTE2 释放序）；⑨kv_iters>8 多轮任务（D256 Skv>2048 型）状态经 UB 跨轮常驻（m_i/sumexp/acc_o），lane（r_factors[i]/sumexp_is[i]）轮内 phase1→phase2 同槽传递与 generic 外层批同构；⑩SEM 收支逐轮对称（C: wait1/set1×1 + set0/wait2/set4×batch + set3×1；V 镜像），与 generic 每外层批完全同构——协议位置零漂移；⑪Form A / batched 单块 / prefill-D256 generic / emit_v generic 分支零触碰。
[KERNEL-REUSE-AUDIT] 依据: Part A4 [KERNEL-REUSE-PRECHECK] #R4-3 | 维度及结论: ①eff 均匀性=TRUE by 证明（s_blocks=1 + causal min 恒等）②C/V 同式=TRUE（同一 my_start/my_count/kv_iters 派生，常量共享）③V 侧边界状态复位/写回=TRUE（phase2 内闭合，守卫平衡推演⑤）④hoist 等值=TRUE（s0≡0/valid_m≡S 常量）⑤lane 语义=TRUE（⑨）⑥位级=待 probe 裁决（预期 BITEXACT：每槽序列同构 + fill 等值迁移论证④）| 结论: repair-kernel-first（探针位级验证后启用）
[SELF-CHECK] 本次 Edit 只涉及 [#R4-3]：①emit_v 新增 `elif S <= 16:` 扁平分支（~155 行，插在 batched 与 generic 之间）；②Form B C-scope 新增 `elif S <= 16:` 扁平分支（~52 行）；③docstring 批处理 bullet 同步；④区域 diff 全核验：Form A prim_func 209 行逐字节未动 ✓ / emit_v batched 123 行 ✓ / emit_v generic 151 行 ✓ / Form B C batched 52 行 ✓ / Form B C generic 52 行 ✓ / host 包装 ✓；⑤C/V 扁平公式静态核对：total_flat/num_fb/_rem_f/batch_slots 各 ×2（双侧同式 §7.2-5）✓，f/kv 派生 ×4（C GEMM1/GEMM2 + V phase1/phase2）✓；⑥TIR 作用域纪律（rev2-A1）：task/by/bz 绑定于使用块内、tl 绑定于槽循环顶层用于后代块（合法方向）、无兄弟块跨引用 ✓；⑦ruff check/format clean ✓（1153 行）；⑧备份 = history_version/mha_impl_r3_perf8.py（md5 e80f7f65 = post-#R4-1 采纳态 ✅）。
[RESULT-#R4-3] 优化点: 扁平 (task,kv) 批处理 decode 推广（Form B 多块 decode：C/V 双侧第三分支，f=task*kv_iters+kv 扁平槽序，协议轮 case11 26→7 / case15/18 48→12 / case10 2→1；任务状态复位迁至 phase2 kv==0 槽、任务尾归一化+写回骑 kv==kv_iters-1 槽 SIG_S_HALF 平衡闭环）| 精度: **pass**（probe_r44: **17/17 BITEXACT vs r3_perf8**——扁平域全拓扑（轮中任务切换/kv=16 跨轮状态携带/部分轮/奇偶批/causal 均匀恒等/MTP S=2 rows_vid 守卫/D256 io3-alias/尾列/bf16/zeros/S=16 满行）+ 4 项未变更域守卫；l0 10/10；位级锚 10/10 BITEXACT；R11 压力 nondet=0；/tmp/opencode/r4_probes/probe_r44.log、r4_l0_iter4.log、r4_anchor_iter4.log）| 性能(msprof median us，vs 本地 r3_perf8 基线): case11 312.34→307.02 (-1.7%)、case15 **504.19→505.24 (+0.2%)**、case18 **504.53→504.93 (+0.1%)**、case10 35.34→35.12 (-0.6%) | 守卫: case14 378.16→377.28 (-0.2%)、case16 565.27→563.14 (-0.4%)（batched 路径未触碰 ✓）| 对比: **全部 6 shape ±3% 噪声带内，无可证实收益**（关键补测：case15/18 的 r3_perf8 本地基线 504.2/504.5 揭示其 vs cann-bench 锚 577.9/581.3 的 -12.8% 全部来自 #R4-1 barrier 移除，扁平化增量为零）| 采纳: **NO — 回滚**（rollback_reason: 性能无变化规则；回滚至 r3_perf8 = #R4-1 采纳态，md5 e80f7f65 核验还原）| **归因知识（协议税模型修正，rev4 重要输入）**：①#R4-1 移除 barrier 后，**每协议轮的 SEM 开销已从 ~2-3.6us（barrier 放大态，#R3-1 时代 case14 证据）坍缩至 ~0.2-0.3us**——轮次削减 3.7-4x 在 decode 族不再可测（case11 19 轮节省仅 ~5us）；②decode 多块 wall = **每槽数据搬运带宽**（case11/15 ≈5.3-5.9us/slot，K+V 256-272KB/slot MTE2 主导），协议结构已非约束；③#R3-1 收益模型（-28.9%）的适用前提 = barrier_all 与逐任务 SEM 的复合税——单因子消除后另一因子收益归零（**收益不可叠加性**）；④扁平机制本身位级安全已证（17/17），机制知识入库（跨轮任务状态携带/phase2 复位时序约束），若未来结构变更使协议税回升可低成本重试。**触发中止条款②：连续 3 次无提升（#R4-2a +3.7% 回滚 / #R4-2b +0.9% 回滚 / #R4-3 噪声带回滚），会话转入收口。**

### 迭代 r4-5（收口） — 全量验证 + cann-bench 复评（2026-09-20T13:00+08:00）

- 中止判定：**触发条款②（连续 3 次无提升：#R4-2a +3.7% 劣化回滚 / #R4-2b +0.9% 无变化回滚 / #R4-3 全目标噪声带回滚）**；迭代 4/10 用毕，采纳 1 项（#R4-1）。
- 终态 = history_version/mha_impl_r3_perf6/7/8.py 同体（md5 e80f7f652a7123c82c71c0354723755b，947→950 行，post-#R4-1）。
- `--level all` exit 0 "Test Passed!"：L0 10/10 + L1 19/19 + L2 12/12 正确拒绝 + Boundary 6/6 + 哨兵 4/4 PERF_PASS（kernel_path 6.0-7.4s ≤ 120s 门；/tmp/opencode/r4_all_final.log）。
- 位级锚终态 10/10 BITEXACT vs Developer 终态 + R11 竞态压力 4 配置 ×12 复跑 nondet=0（/tmp/opencode/r4_anchor_final.log，终态上新鲜复跑）。
- 提交副本同步（md5 e80f7f65 一致）。cann-bench 复评：见 final_report.md §S4-1（本报告落笔时评测进行中，数字以报告 JSON 为准）。
- 会话 4 状态总表：

| # | 优化点 | 状态 | 关键结果（msprof median / 归因） |
|---|--------|------|----------------------------------|
| R4-1 | Form B barrier_all 纯移除（=#R3-6 第一阶段） | ✅ **采纳** | case9 -26.0%（0.240→≈0.331x）/ case14 -28.7%（0.995→≈1.40x）/ case16 -12.6% / case15/18 -12.8%（收口补测）/ case11 -10.7% / case10 -11.4%；gemm_v0 黑盒自同步边界知识入库 |
| R4-2a | prefill D256 GEMM1 kL0Size=256 单 K-step | ❌ 回滚（劣化 +3.7%） | kNumSteps=1 关闭 L0 ping-pong——重叠损失 > step 节省；单 mma k=256 ≡ 链式 2×k=128 位级等价知识入库（12/12 BITEXACT） |
| R4-2b | Form B prefill D256 k/v L1 双缓冲（预取式零新增 flag） | ❌ 回滚（无变化 +0.9%） | op 归因：全 pipe 读数零变化——MTE2 暴露已被吸收非关键路径；3D 无 annotate L1 + gemm_v0 运行期切片机制 BITEXACT 入库（15/15） |
| R4-3 | 扁平 (task,kv) 批处理 decode 推广（=#R3-8） | ❌ 回滚（无变化，6/6 噪声带） | 协议税模型修正：#R4-1 后每轮 SEM 开销 ~3.6us→~0.25us，轮次削减收益归零；decode wall = 每槽 MTE2 带宽；#R3-1 收益不可叠加性实证；扁平机制 17/17 BITEXACT 入库 |
| R4-4 | #R3-7 Form B→A D256 mma 化 | 🚫 设计级禁实施 | 证据积累完成（case9 三证据链：step 削减劣化/MTE2 隐藏零响应/barrier 移除后剩余 = gemm_v0 每-step scalar 链 78% wall）→ rev4 立项建议核心输入 |
| R4-5 | case7/8 bN=256 变体 | ⏸ 未启动 | 中止条款②触发，预算纪律收口；Form A 结构不动纪律全程保持（case7 守卫 +0.6% 噪声带佐证零触碰） |
| R4-6 | 会话 3 证伪项 | ✅ 未重做 | w1 瘦身/Form A k-v DB/CROSS_INTERVAL=2 三项纪律保持 |
| R4-7 | 反模式复扫 | ✅ 继承 S3 #R3-9 | 唯一残留命中（Form B barrier_all）已由 R4-1 消除；Form B 生成码现为零 PipeBarrier<PIPE_ALL>、零用户 flag、纯 SEM 协议 |
- cann-bench 复评 **cann_final_eval_20260920_141424**：20/20 pass、综合 68.50→**70.21**（+1.71）、性能分 18.50→**20.21**、平均加速比 0.7017→**0.7813**（+11.3%）。**case14 0.995→1.394 / case17 0.916→1.138 / case10 0.903→0.998 / case13 0.766→0.883 / case12 0.785→0.886 / case16 0.678→0.784 / case18 0.703→0.811 / case11 0.824→0.925 / case15 0.598→0.688 / case9 0.240→0.324 / case4 0.732→0.846**；case7/8 0.464/0.470（±0.003 噪声带，Form A IR 未动）；case1/2/3/5/6/19/20 ±0.013 内噪声漂移。**decode 不回退约束全核验通过；case7/8/9 只升不降核验通过；回退条款不触发（70.21>65.05 ∧ 0.78x>0.55x）。**
- **M1 判定：部分达成**——平均 ≥0.75x ✅（0.7813，超额 +0.031）；case9 ≥0.4x ❌（0.324，缺口 0.076；规则内下限实证 §S4-3-2 三证据链，破局唯一路径 = §10.3-⑤ 设计级 mma 化，rev4 建议证据已备齐于 final_report §S4-5）。
- ruff check/format：clean（947 行终态）。最终报告：final_report.md §S4-1~S4-6；备份链 r3_perf5..8 完整；baseline_r4.json / baseline_r4_iter1.json / baseline_r4_final.json + msprof_r4_*/*.json 证据链完整。

---

# 会话 5（perf_session=5，2026-09-21，Expert rev3 轨第三轮；迭代软预算 3-4，备份链续编 r3_perf{N} N≥9）

> 对象：Expert 版 mha.py（会话 4 终态 = #R4-1 采纳态，md5 e80f7f652a7123c82c71c0354723755b，947 行，cann-bench 70.21 / 0.7813x）。
> 触发：**用户新观察（非 DESIGN_ERROR、非需求变更）**——基于 `perf_tuning/simulator/case1` camodel 指令级流水图，用户直接指令
> "整个算子 C 核的等待时间明显长于 V 核，V 核计算当前的问题是 MTE 与 V 核计算是串行执行，V 核计算过于碎片化，请进行修改"
> （此前一次"生成方案并问询"被用户 dismiss，本轮为 mode=implement 直接实施）。
> 验收锚定：**单块 Form A 类 case1/6/19/20**（case6 为主受益者，12.8 task/核）；守卫 = Form A 多块 generic（case7/2/5）+ Form B 全族（case9/10/14/16）。

## S5-Step 1 基线（baseline_r5.json 已落地；证据 msprof_r5_base/*.json + /tmp/opencode/r5_sim_base/）

msprof timeline op_summary Task Duration median（warmup 5 + iters 20，driver /tmp/opencode/r5_probes/run_msprof.sh）：

| 角色 | case | 基线 median us | 会话 4 记录 | 偏差 | 路径 |
|------|------|----------------|-------------|------|------|
| **目标** | case1 | **13.03** | 12.94 | +0.7% | Form A single_kv_block P1（D=64≠bN） |
| **目标** | case6 | **54.08** | —（首测） | — | Form A single_kv_block P2 共池，12.8 task/核 |
| **目标** | case19 | **35.86** | —（首测） | — | Form A single_kv_block P2 causal 显式 scale |
| **目标** | case20 | **30.71** | 30.84 | -0.4% | Form A single_kv_block P2 zeros |
| 守卫 | case7 | 926.43 | 932.60 | -0.7% | Form A 多块 generic P2 |
| 守卫 | case2 | 67.39 | —（首测） | — | Form A 多块 generic P1 causal bf16 |
| 守卫 | case5 | 18.44 | —（首测） | — | Form A 多块 generic P1 cross |
| 守卫 | case9 | 1018.49 | 1020.73 | -0.2% | Form B P3 多块 generic |
| 守卫 | case14 | 377.42 | 378.16 | -0.2% | Form B single_kv_block 已批处理 |
| 守卫 | case10 | 34.41 | 35.34 | -2.6% | Form B D1 多块 generic |
| 守卫 | case16 | 568.37 | 565.27 | +0.5% | Form B single_kv_block MTP causal |

7 项与会话 4 记录同测的 case 全部落在 ±3% 噪声带内 → **环境与会话 4 一致，基线可信**。
精度基线 = 会话 4 终态门禁（`--level all` exit 0 + 位级锚 10/10 BITEXACT + R11 nondet=0，/tmp/opencode/r4_all_final.log、r4_anchor_final.log）。

### 用户三点观察的独立量化复核（不盲信转述，自行重解析 merged trace.json）

方法：`/tmp/opencode/s5_analysis/trace_stats.py`（busy = 每 pipe `[ts, ts+dur)` 区间的**并集**长度；wall = 该核全 pipe 时间跨度；
overlap = MTE2∪MTE3 与 VECTOR 两个区间集的**交集**长度）。数据源 = `perf_tuning/simulator/case1/OPPROF_20260920164918_NMQUAEYMVUSDDWWX/simulator/trace.json`（77365 events，pid="core0.cubecore0" 型字符串标识）。

| 观察 | 复核结论 | 实测数字（core0） |
|------|----------|-------------------|
| ① C 核等待明显长于 V 核 | **证实** | C FLOWCTRL busy 7.725946us / wall 9.288109us = **83.18%**；V FLOWCTRL busy 5.874054us / wall 9.831892us = **59.74%**。两侧均以 `WAIT_FLAG_DEVI`（n=6）为绝对主体（C 7.723us / V 5.871us） |
| ② V 核 MTE 与 VECTOR 串行 | **证实（本人读数略高于 Orchestrator，不影响结论）** | MTE2∪MTE3 = 2.447027us，VECTOR = 3.387032us，交集 = **0.129732us = MTE 的 5.30% / VECTOR 的 3.83%**（Orchestrator 报 0.077us = 3.1%/2.3%）。差异溯源：交集由 29 对 ≤25.4ns 的**相位边界簿记**碎片构成（最大一对 = MOVEV × MTE3 25.4ns @8.13us），是否计入零长/SET_CROSS_CORE 事件导致口径差；**两种口径都判定"几乎完全串行、io_buf 双槽乒乓未发生"** |
| ③ VECTOR 计算过于碎片化 | **精确证实** | VECTOR 切成 **11 段**（gap>5ns）：9.2 / 11.4 / 49.2 / 15.1 / 54.1 / 8.6 / **1537.8** / 113.5 / 68.1 / **1537.8** / 67.6 ns。其中 **6 段为 8.6~54.1ns 微型碎片**（无实质计算），**恰好 2 段 1537.8ns 为真实 softmax 主计算**（= 每 task 一次） |

### 形态与关键路径复算（case1）

`(B2,S128,Skv128,N8,D64)` fp16 causal → `form_a=True`（S>16 ∧ D≤128）、`block_M=64`、`block_N=128`、
`kv_iters=ceil(128/128)=1` ⇒ **`single_kv_block=True`**、P1 分池（D≠bN ⇒ 独立 l0a_o/l0b_o/l0c_o + 独立 io3）、
`s_blocks=2`、`block_num=2*8*2=32`、`core_num=min(32,20)=20`、`q_tasks=1`、`r_tasks=12`
⇒ **cid0-11 各 2 task（task 2c 与 2c+1 = 同 (bz,by) 的相邻 s_block），cid12-19 各 1 task**（生成码 `if (cid < 12)` 运行期分支实证）。
`eff_num_outer = ceildiv(1, 8) = 1` ⇒ `batch_iters = 1` ⇒ **NUM_STAGES=8 批处理流水完全退化，io_buf[2] 双槽跨迭代乒乓从未激活**（mha.py:544 的设计意图落空）。

core0 关键链（本人从 trace 逐指令重建，与 Orchestrator 转述一致）：
`init 2.83-3.66` → **C-QK1**（MTE2 Q 513ns + K 1179ns + V 1584ns；MTE1 L0A/L0B；MMAD 80.5ns；FIX w1 284ns）→ 5.30
→ **V-sm1**（MTE2 w1 读 289ns；VECTOR 1537.8ns；MTE3 P 写 366ns）5.31-7.50
→ **C-PV1**（scalar 在 `wait WS2_V2C` 上**阻塞 3816ns**：3.689→7.505；p_l1 237ns；MMAD 80.5ns；FIX w3 650ns）7.51-8.53
→ **V-epi1**（MTE2 w3 读 223ns；VECTOR 113ns；MTE3 Output 写 640ns）8.54-9.47
→ **C-QK2** 7.80-9.36（与 V-epi1 部分并行）→ **V-sm2**（scalar 在 `wait WS2_V2C` 上**阻塞 2751ns**：8.772→11.523）9.48-11.52
→ **C-PV2** → 12.12 → **V-epi2** → 12.67。
即 **每 task 走"C-QK → V-sm → C-PV → V-epi"四相位严格串行链 × 2 task，task 之间零重叠**。

### 新证据（历史文档从未记录）：同 (b,h) 相邻 task 的 K/V 同字节冗余重读

core0 trace 中 task2 的 K/V GM→L1 装载**源地址与 task1 逐位相同**：
K `XN=0x11351c00`（task1 id:10 ≡ task2 id:46，重读 625ns）、V `XN=0x11391e00`（task1 id:11 ≡ task2 id:64，重读 337ns）；仅 Q 不同（`0x11311a00` vs `0x11321a00`，s0=0 vs 64）。
[#R3-9] 反模式扫描曾以"K/V 跨任务无复用"（本文件 :523）排除 `optimization-guide §2.4 Full-Load`——该断言基于 case7 类**多块** shape 推理，
对 **single_kv_block 子类不成立**（此时 s_blocks≥2 的相邻 task 天然共享同一 K/V 块）。→ 列为 [#R5-3]。

### 历史条目再判读（两处需修正的认知）

1. `final_report.md:239`"Form A 单块任务批处理（case1/19/20）**未实施** — 批处理增量收益预期小于 Form B"是**预测性否决，从未实测**；
   真实排除原因见 `[IMPL-#R3-1]④`："emit_v 为两 Form 共享 macro，Form A 单块 shape 必须留在 generic 路径，**否则 C/V 协议失配死锁**"
   ——即 **V 侧批处理分支早已存在且已被 Form B 五个 case（12/13/14/16/17）位级验证（9/9 BITEXACT），排除 Form A 只因 C 侧批镜像没写**，非收益裁决。
2. `debug_log.md:265-269` 的 case14"vid1 半区协议税"归因与本轮同族但不同 case：本轮双 AIV 时间轴对称（veccore0 wall 9.637 vs veccore1 9.634，偏斜 ~3ns），
   vid 半区拆分把 V 相位从 ~4.4us 降到 2.2us 是**收益不是税**；且 `src/ir.cc:281-290` Expert frame vid extent=2 为硬编码 ⇒ "改单 AIV"结构不可行，不作为候选。

## S5-Step 2 算子类型（get_kernel_source 复核，/tmp/opencode/r5_src_base/ 16 shape）

生成码含 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` + `if ASCEND_IS_AIC` / `if ASCEND_IS_AIV` 双段
⇒ **CV 融合混合型（MIX_AIC_1_2：每 task = 1 AIC + 2 AIV，Expert 结构性 cv_1_2）**，与会话 3/4 判定一致。
`PipeBarrier<PIPE_ALL>` = 0（#R4-1 后 Form B 已纯化；Form A 从来只有 pipe-pair flag）。
基线 IR 交叉校验：`G_case9_B_p3_multi.cc` md5 `1918a5c7…` ≡ 会话 4 `/tmp/opencode/r4_src/FormB_case9_postR41.cc`；
`G_case14_B_single.cc` md5 `6966fdc3…` ≡ `FormB_case14_postR41.cc` ⇒ **起点确为会话 4 终态，dump 方法可复现**。

## S5-Step 3 瓶颈定位与 Part A5 优化点清单

**瓶颈类型判定：`sync`（核间/相位依赖链串行），而非 compute / transfer。**
证据：①C 核 83.18% wall 在 FLOWCTRL 等待、V 核 59.74%，但两侧**真实工作管**占用都不高（C：CUBE 3.5% / FIXP 21.0% / MTE2 33.6% / MTE1 5.5%；V：VECTOR 35.2% / MTE2 10.7% / MTE3 14.7%）
⇒ wall 不是任何单管吞吐上限，而是**四相位严格串行链的相加**；②`WAIT_FLAG` 指令本身传播延迟仅 3~6ns（SET→对侧释放实测），
等待窗口的内容是**对侧核真实不可压缩的工作**（GM→L1 装载、FIX L0C→GM、softmax 向量链）⇒ **削减 wait/flag 指令数不会缩减 wall**
（[#R3-4] 已判决性证伪"调 CROSS_INTERVAL 频率"这类思路）——**只有删除/重排真实串行相位才能缩减 wall**；
③会话 3 三轮证伪（V 指令削减 / C 装载重叠 / SEM 频率）+ 会话 4 两轮（MTE2 隐藏 / 协议轮削减）在 case1 上均为吞吐/频率类手段，
本轮候选为**依赖链删除/重排类**，性质不同，不受前述证伪约束。

[KERNEL-REUSE-PRECHECK]
candidate: #R5-1 Form A single_kv_block 任务批处理（flash_attn_optimize.md §1 L67-L100 批量流水 + §7 workspace 槽位复用；performance-antipatterns.md L303-L340 "AIC/AIV 混合算子未开启 CV overlap"；#R3-1 同机制的 Form A 补全）
expanded_domain: **无新增 host 调用域**——host 契约/分派谓词/tile 三档/buffer 形状与 dtype 全部不变；仅既有编译期谓词 `single_kv_block` 在 **form_a=True** 分支下由 generic 逐 task 路径切换到已存在的批处理路径。受影响 shape = Form A ∧ S_kv≤block_N(=128) ∧ S>16，即 cann-bench case1/6/19/20（+ 同族任意 (B,S≤2048,Skv≤128,N,D≤128) prefill/cross）。Form B（`form_a=False`）逻辑恒等不变；Form A 多块（`single_kv_block=False`）走 else 分支不变。
compatibility_dimensions: ①C/V flag 配对平衡（§7.2-5）= 批计数公式双侧同式（`num_tb=ceildiv(my_count,NUM_STAGES)`、`batch_tasks=if_then_else(remaining<NUM_STAGES,remaining,NUM_STAGES)`，同从 my_start/my_count 派生，不经 ws 传递）✅ 逐字复制 Form B 批分支 :778-781；②SEM set/wait 位置逐槽不变（CROSS_INTERVAL 语义原样；WS1_V2C/WS3_V2C/WS2_C2V 批级、WS1_C2V/WS2_V2C/WS3_C2V 槽级）✅ 与 :785-822 / emit_v :236-346 一一对应；③每槽算子序列与现行 generic 逐 task 路径逐 op 同序（精度栅格不变）✅ 唯二差异 = Q 的 GM→L1 由 task 级移入槽级 + `q_l1→l0a_s[side]` 由 `if i<2` 改为每槽必发（Q 随 task 变，语义**必须**如此）；④ws 槽 `[cid,i]` 释放协议不变 ✅；⑤causal 裁剪恒等式（single_kv_block ⇒ eff_kv_iters≡1，无运行期变异）✅；⑥令牌守恒（SIG_Q/K/V/P_L1、SIG_L0AB/L0C ±side、init/destroy 闭环）— 已推演（见 [IMPL-#R5-1]），需 R11 竞态压力门实测把关 = **待验证**；⑦L0 side=i%2 轮转在"批维=task"下的正确性 — 与 case7 batch_iters=8 同构（生产验证）+ Form B 批分支同构，仍需探针覆盖奇偶 batch_tasks = **待验证**
unknown_or_false: ⑥⑦（令牌守恒 / side 轮转）→ 必须以探针 + 压力门实测收敛
plan: **repair-kernel-first**（V 侧批分支已存在且已验证，只需放开分派门 + 补 C 侧 Form A 镜像；探针位级验证后启用）

[#R5-1] **Form A single_kv_block 任务批处理（补 C 侧镜像 + 放开 V 侧分派门）**（参考: flash_attn_optimize.md §1 L67-L100（num_stages 批量流水 = "AIC/AIV CV overlap" 反模式在 Expert 禁用 T.Pipelined 前提下的唯一规则内表达）、§2 L104-L191（pipe-pair flag 生命周期闭环）、§7 L484-L497（[NUM_CORES,num_stages] 槽位复用）；performance-antipatterns.md L303-L340（**AIC/AIV 混合算子未开启 CV overlap — 本项直接命中**：主循环仍是 Cube 写 ws、Vector 读 ws 的串行结构，且 batch_iters=1 使既有 num_stages 机制空转；该条目给出的 `T.Pipelined`+AUTO_CV_* 替代写法被 DESIGN §2.2-d/R9 与 §8.6 明令禁用 ⇒ Expert 等价表达 = 批维换成 task）；DESIGN §7.2/§8.6（SEM 协议 + io_buf 双槽设计意图）；#R3-1（同机制 Form B 版，9/9 BITEXACT + case14 -28.9% 已验证）：**适用（首位主攻，与用户观察机制吻合度最高）** — 把 cid0-11 的 2 个 task（cid0-19 最多 8 个）批进同一个 num_stages 组，使 ①V 的 softmax(task2) 紧随 softmax(task1) 执行而非等 C 完成 PV1+epi1（直接消除 core0 上 2751ns 的第二次 `wait WS2_V2C` 阻塞）、②C 的 Q2/K2/V2 装载与 V 的 softmax1 重叠、③io_buf[2] 双槽在槽 i / i+1 之间真正形成 MTE2↔VECTOR 乒乓（直接对应用户"MTE 与 V 核计算串行"）、④VECTOR 微型碎片随槽间连续执行而合并（对应"计算过于碎片化"）。覆盖 case1/6/19/20；case6（12.8 task/核 ⇒ 批 8+5）理论受益最大。
[#R5-2] **单块 O 直写 FIX(L0C)→Output（删 V phase-2 + 删 WS3 段）**（参考: `examples/fusedmoe/example_fusedmoe.py:628`、`examples/deepseek_v4/lightning_indexer.py:756-764`（L0C→GM 4D 运行期切片直写先例）；[#R3-3] 探针已实证 fixpipe fp32→fp16 RNE ≡ V 侧 vcast 位级等价 11/11 BITEXACT；DESIGN §10.3-①/§8.2）：**适用（次位，与 #R5-1 组合时贡献最大）** — 单块特化下 V phase-2 只做 `acc_o = 0 + PV` 再 cast 写回，是**纯搬运相位**；改为 C 侧 `T.copy(l0c_o[side,:,:], Output[bz, s0:s0+valid_m, by, :])` 可整段删除 V phase-2（mha.py:459-478 + 485-497）与 WS3 的 C2V/V2C 一对 SEM 往返，直接砍掉关键链上的第四个相位（V-epi，core0 上 0.93us/task）。⚠ 唯一新组合 = "L0C→BSHD 4D GM 运行期行切片（valid_m 尾块）"，仓内先例均为整块或 BHSD ⇒ **必须独立最小探针先行**；-0.0 规范化差异已论证域内不可达（mha.py:935-936 host 断言 is_causal⇒S≤S_kv）。
[#R5-3] **同 (b,h) 相邻 task 的 K/V L1 驻留（跳过冗余 GM→L1 重读）**（参考: optimization-guide.md §2.4 Full-Load（[#R3-9] 曾以"K/V 跨任务无复用"排除，S5-Step 1 新证据证伪该断言于 single_kv_block 子类）；写法先例 = [#R3-2]（3D L1 + runtime-guarded copy，10/10 BITEXACT））：**适用（最小 diff、风险极低，排 #R5-1 之后以隔离归因）** — C 侧 K/V 载入（:606-609 / :639-642）加运行期条件 `same_kv = (t>0) and (task//s_blocks == (task-1)//s_blocks)`，命中则跳过 GM→L1 copy；`SIG_K_L1`/`SIG_V_L1` 的 wait/set 对**保持在条件外原样执行**（令牌收支不变，纯调度变更、数据逐字节相同、零算术变更）。core0 实测可省 K 625ns + V 337ns/A 核。
[#R5-4] **NUM_STAGES 8→12/16 扫描**（参考: flash_attn_optimize.md §1 L98"推荐 8-16"；DESIGN §10.3-④；= [#R3-5] 未启动项）：**暂缓（条件触发）** — #R5-1 采纳后 case6（12.8 task/核）会出现 8+5 两轮批，NUM_STAGES=16 可折成单轮；但 GM ws 足迹 P2 15.7MB@8→31.4@16（R14 L2 驻留风险）。**前置依赖 #R5-1**，且须以实测裁决，不在本轮软预算内强推。
[#R5-5] **CROSS_INTERVAL 1→2 / Form A k-v L1 双缓冲 / w1 dtype 瘦身**：**🚫 纪律性不重做** — [#R3-4] / [#R3-2] / [#R3-3] 三项已实测证伪并回滚（判决性实验 + 归因知识入库 S3-3）；本轮瓶颈判定为依赖链串行而非 SEM 频率/装载暴露/V 指令吞吐，三项均不对症。若 #R5-1 采纳后瓶颈构成改变，可凭已入库机制知识低成本重试（探针全免）。
[#R5-6] **改单 AIV（cv_1_1）以消除 vid1 半区协议参与**：**🚫 结构不可行 + 证据性放弃** — `src/ir.cc:281-290` Expert frame vid extent=2 为硬编码字面量，`T.Kernel(n, is_npu=True)` 无 threads 参数即恒推入 (cid, vid) 双绑定；且 camodel 显示双 AIV 时间轴对称（wall 9.637 vs 9.634，偏斜 ~3ns），半区拆分把 V 相位从 ~4.4us 减到 2.2us 是收益；改单 AIV 会使 V 关键相位加倍 ⇒ 反向劣化。
[#R5-7] 反模式清单 Expert 版复扫（947 行终态，本轮静态项）：**完成** — 关注项 A/B launch 核数 ✅ 已 Fixed Core（`core_num=min(block_num,20)`，:148）；Vector 逐元素/逐行 for ✅ 无（全 T.tile 2D）；冗余全局同步 ✅ 无（`PipeBarrier<PIPE_ALL>` 生成码计数 = 0，#R4-1 已纯化）；基础指令拼接未融合 → phase2 mul+add 形似 mul_add_dst，融合=FMA 单舍入破坏位级锚 ⇒ 维持 [#R3-9] 证据性放弃；tile 过小 ✅ decode bM=16 分形下限 / prefill bN=128 L0B 硬限；**AIC/AIV 混合未开启 CV overlap → 命中（Form A single_kv_block 子类，batch_iters=1 使 num_stages 机制空转）= [#R5-1]**；纯 AIV memory-bound 未双 buffer ✅ io_buf 双槽已声明（但单块 Form A 下未被激活 = 同 #R5-1 根因）；多行 tile 循环内逐块归约 ✅ 单次 reduce dim=-1；正交轴串行化 ✅ 无；AUTO_CV_COMBINE 误分核 n/a（Expert 四 config 全 False）。guide 其余节：2.1 Split-K 不适用；**2.4 Full-Load 由"不适用"改判为"适用于 single_kv_block 子类"= [#R5-3]**（S5-Step 1 trace 地址级新证据推翻 [#R3-9] 断言）；2.5 小块合并 不适用（跨任务 Q GM 不连续）；2.6/2.13 已 2D 化；2.7 axpy 同 mul+add 放弃理由；2.8/2.15/2.17 无稀疏/正交/重排形态；2.9 已 Fixed Core；2.10 Expert 四 config 锁死；2.11 UB 实配 196352B/191.75KB（生成码 `InitBuffer(ascend_ub, 196352)` 读数）——tile 扩面仍被 L0B 硬限否决；2.12 host 零张量操作 ✅；2.14 无 D2H；2.16 fp16/bf16 原生 ✅。

## S5-Part B5 [ORDER-PLAN]

[ORDER-PLAN] 实施顺序：
1. [#R5-1] Form A single_kv_block 任务批处理 — 前置依赖: 无 — 理由: **首位主攻**（用户三点观察全部指向该机制；反模式清单唯一新命中项）；V 侧批分支已存在且已被 Form B 五 case 位级验证 ⇒ 变更面 = 1 行分派门 + C 侧 1 个新 parse-time 分支，**Form B 与 Form A 多块的 IR 结构性不变**（可由 16-shape IR dump 逐字节验证）；不改 buffer 形状/dtype/数量 ⇒ 无布局依赖、无预算依赖。
2. [#R5-3] K/V L1 驻留（同 (b,h) 相邻 task） — 前置依赖: **#R5-1**（批处理把同 (b,h) 相邻 task 折进同一批的相邻槽，重读窗口从"跨 task"变为"批内相邻槽"，条件式需按新循环结构书写；且必须隔离归因，不与 1 同时上）— 理由: 最小 diff、零算术变更、trace 地址级证据确凿；数量依赖：不新增 buffer（复用既有 k_l1/v_l1 单槽 + runtime if 跳过 copy）⇒ 排在 1 之后不冲突。
3. [#R5-2] 单块 O 直写 FIX(L0C)→Output — 前置依赖: **#R5-1**（同触 C 侧 GEMM2 批体与 emit_v phase-2/写回段；先定批处理结构再删相位，避免交叉回滚）、独立最小探针（L0C→BSHD 4D 运行期行切片）— 理由: 砍掉整条第四相位（V-epi），与 1 组合收益最大；配置依赖：删除 WS3 段后 SEM_WS3_C2V/V2C 收支需双侧镜像删减 ⇒ 必须在 1 定型后做。
4. [#R5-4] NUM_STAGES 12/16 扫描 — 前置依赖: #R5-1（批维=task 后 NUM_STAGES 语义变为"每批 task 数"）— 理由: 旋钮项，实测裁决，本轮软预算外。
5. [#R5-5]/[#R5-6]/[#R5-7] — 不实施（证伪纪律 / 结构不可行 / 静态项已完成）。
（每轮固定流程：备份 `history_version/mha_impl_r3_perf{N}.py` → [ORDER-CHECK]/[IMPL]/[KERNEL-REUSE-AUDIT]/[SELF-CHECK] → Edit 单点 → 探针门 + l0 + 位级锚 + R11 压力 → l1 → 覆盖门禁 → msprof（目标 4 + 守卫 7）→ IR 逐字节守卫 → **camodel 复采机制验证**（本轮新增验收件，防 [#R3-4] "只看 wall 数字不看机制"覆辙）→ [RESULT] 采纳/回滚。）

## S5-Step 4 逐项实施记录（会话 5）

### 迭代 r5-1：[#R5-1] Form A single_kv_block 任务批处理（2026-09-21）

[ORDER-CHECK] 准备实施: [#R5-1] Form A 单块任务批处理 | 前置依赖: 无（ORDER-PLAN 首位）✅ | 结论: ✅
[IMPL-#R5-1] 已阅读 flash_attn_optimize.md L60-L100（§1 Expert 优化：num_stages 批量流水骨架 = 外层 `T.serial(num_outer)` + 内层 `T.serial(batch_iters)` + `side = i % 2` 双缓冲轮转 + GEMM1 批 / GEMM2 批两段 + 批末 `if (i+1) % cross_interval == 0 or i == batch_iters - 1: T.set_cross_flag(...)`；收益原文"批量处理减少同步次数 + 流水线并行：数据搬运和计算重叠"；关键参数 num_stages 推荐 8-16）、L104-L191（§2 细粒度 flag 生命周期闭环）、performance-antipatterns.md L303-L340（AIC/AIV 混合未开启 CV overlap：识别特征 = 生成码同时含 AIC/AIV 但主循环仍是 Cube 写 ws、Vector 读 ws 的串行结构 ⇒ **本项直接命中**；其 `T.Pipelined` + AUTO_CV_* 替代写法被 DESIGN §2.2-d/R9/§8.6 明令禁用 ⇒ Expert 规则内等价表达 = num_stages 批结构，批维由 kv-block 换成 task）、本文件 [IMPL-#R3-1]①-⑤（Form B 同机制的五条关键约束，逐条继承）、mha.py:771-822（Form B C 批分支 = 结构模板）、mha.py:599-670（Form A generic GEMM1/GEMM2 批体 = 逐 op 搬运源）、mha.py:225-346（emit_v 批分支，已验证，本轮仅改分派门）。
关键约束（继承 [IMPL-#R3-1] + Form A 特有增量）:
①§7.2-5 C/V 批计数同式：`num_tb = T.ceildiv(my_count, NUM_STAGES)`、`_remaining_t = my_count - kb*NUM_STAGES`、`batch_tasks = T.if_then_else(_remaining_t < NUM_STAGES, _remaining_t, NUM_STAGES)` — 双侧从同一 my_start/my_count 派生，**逐字复制 Form B 批分支 :778-781**，不经 ws 传递；
②SEM set/wait 位置逐槽不变：批级 `wait(WS1_V2C)`/`wait(WS3_V2C)`/`set(WS2_C2V)`，槽级 `(i+1)%CI==0 or i==batch_tasks-1 → set(WS1_C2V)/set(WS3_C2V)`、`i%CI==0 → wait(WS1_C2V)/wait(WS2_V2C)/wait(WS3_C2V)` — 与 emit_v 批分支 :236/250-251/309-311/324-325/346 一一对应；
③每槽算子序列与 generic 逐 task 路径逐 op 同序（精度栅格不变），**唯二 Form A 特有差异**：(a) Q 的 GM→L1 由 task 级（:589-593）移入槽级（每 task 有自己的 s0）；(b) `q_l1 → l0a_s[side]` 由 generic 的 `if i < 2`（:612-613，kv 批内 Q 不变故只需批首 2 槽预载）改为**每槽必发**（批维=task ⇒ Q 逐槽变化，语义强制）；
④**SIG_Q_L1 令牌守恒**：generic 每 task = `wait(MTE1→MTE2)` + `set(MTE2→MTE1)` + `wait(MTE2→MTE1)` + 任务末 `set(MTE1→MTE2)`（:675）⇒ 每 task 净 0；批处理每槽 = 同前三句 + **紧随 `q_l1→l0a_s[side]` 之后**的 `set(MTE1→MTE2)`（该槽 MTE1 对 q_l1 的唯一读点，SetFlag 入 MTE1 队列故序在其后）⇒ 每槽净 0；批末**不再**发 task 级归还（否则 dir A 多 1 令牌）。init（:573）1 令牌 / destroy（:681）1 wait 收支不变 ✅；
⑤SIG_K/V/P_L1、SIG_L0AB±side、SIG_L0C±side 的 wait/set 对**逐字照抄 generic 批体**（每槽一进一出净 0，与 batch_iters 奇偶无关）⇒ 奇数 batch_tasks、跨 kb 轮、cid<12(2 task)/cid≥12(1 task) 混合全部平衡；
⑥L0 `side = i % 2` 轮转：批维换成 task 后与 case7 `batch_iters=8` 同构（生产验证）+ Form B 批分支同构；P2（D==bN==128）l0a_o/l0b_o/l0c_o 与 GEMM1 共池互换 ⇒ GEMM1 批全先于 GEMM2 批，池冲突由 SIG_L0AB/L0C ±side 管程串行化（与 generic 同结构，无新增 hazard）；
⑦`valid_n`：single_kv_block ⇒ idx≡0 ⇒ `min(block_N, S_kv)` 为 **Python 级常量**（逐字照抄 Form B 批分支 :782），与 generic 的 `T.min(block_N, S_kv - j0)` 运行期值恒等 ⇒ DMA extent 相同 ⇒ 位级不变；
⑧rev2-A1 教训（[#R3-1] 途中缺陷）：`task/bx/by/bz/s0/valid_m` 必须定义在**槽循环体顶层**，不得置于任何 parse-time `if` 块内（跨块引用会触发 "Undefined variable"）；
⑨Expert 红线：SEM ID 不新增（仍 0-5）、SIG ID 不新增不重排、V 侧管程逐槽闭环、init/destroy 共享段不改、无 `T.copy(L0C→UB)`/`T.copy(UB→L1)` 虚拟通道（§8.6-R12）、无 threads 参数、无 T.Pipelined；
⑩**V 侧零代码改动**（仅 :225 分派门去掉 `and not form_a`）：`single_kv_block and not form_a` 与 `single_kv_block` 在 form_a=False 时**逻辑恒等** ⇒ Form B 全族 IR 结构性不变；form_a=True ∧ single_kv_block=False 时两者同取 else ⇒ Form A 多块 IR 结构性不变。**唯一改变执行路径的域 = Form A ∧ single_kv_block = case1/6/19/20**。
[KERNEL-REUSE-AUDIT] 依据: Part A5 [KERNEL-REUSE-PRECHECK] | 维度及结论: 调用域不扩（host 契约/分派谓词/tile 三档/buffer 形状 dtype 全不变，仅既有 single_kv_block 谓词下 Form A 内循环重构）✅；C/V flag 配对同式 ✅（约束①）；SEM 收支逐槽/逐批与 Form B 批分支同构 ✅（约束②）；SIG 令牌守恒推演通过（约束④⑤，待 R11 压力门实测收敛）；精度栅格逐 op 同序 ✅（约束③，唯二差异均为 Q 装载调度、非算术）；side 轮转同构 ✅（约束⑥，待探针覆盖奇偶 batch_tasks）| 结论: **repair-kernel-first**（V 侧复用已验证分支，C 侧补镜像；探针位级验证后启用）
[SELF-CHECK] 本次 Edit 只涉及 [#R5-1]：
①`emit_v` 分派门单点修改：`if single_kv_block and not form_a:` → `if single_kv_block:`（1 行）+ 8 行理由注释；**批处理 V 路径本体（原 :226-346）零代码改动**——Form A 直接复用已被 Form B 五个 case 位级验证过的分支；
②Form A C-scope 新增 `if single_kv_block:` 批分支（117 行，含 27 行注释）：结构模板 = Form B C 批分支（`num_tb`/`_remaining_t`/`batch_tasks`/`valid_n` 四式逐字复制），批体 = 既有 Form A GEMM1/GEMM2 批体逐 op 搬入，per-i 的 kv 索引（`idx`/`j0`/`valid_n=T.min(...)`）换成 per-i 的 task 索引（`task`/`bx`/`by`/`bz`/`s0`/`valid_m` + `valid_n=min(block_N,S_kv)` Python 常量）；
③原 generic 任务循环（97 行）整体 +4 缩进进 `else:`——**脚本化变换 + 程序化逐行比对：dedent 后 97/97 行与改前逐字节相同**（唯一例外 = ruff format 对 `eff_kv_iters` 长行做括号重排 1 处，因 +4 缩进后超 140 列；语义不变，与 [SELF-CHECK #R3-1] 同类处理，且 Form B 同式处早已是括号形式）；
④Form A C-scope 的 init（:577-585）/destroy（:803-810）、Form B prim_func 全体、host 包装、allocations、SEM/SIG ID 矩阵、`annotate_layout`、L0 池声明**零触碰**——程序化核验：改前 `# destroy:` 至 EOF 的 272 行 suffix **逐字节相同**；
⑤标识符零损坏、ruff check + format clean（1086 行）；
⑥docstring 同步 2 处（Form A 段 + NUM_STAGES 批处理段，说明批维 = kv-block / task 二选一及 Form A 槽级两处 delta）——**已实证 codegen-neutral**：16 shape 的 `get_kernel_source()` 在 docstring 编辑前后**逐字节相同**（`diff -rq /tmp/opencode/r5_src_new /tmp/opencode/r5_src_final` 无差异），故 msprof 数字对终态有效；
⑦令牌守恒推演（[IMPL-#R5-1]④⑤）：SIG_Q_L1 每槽 wait(A)/set(B)/wait(B)/set(A) 净 0（归还点从"任务末"移到"该槽 MTE1 唯一读点之后"，批末不再发任务级归还）；K/V/P_L1、L0AB±side、L0C±side 逐字照抄 generic ⇒ 每槽净 0，与 batch_tasks 奇偶、num_tb 轮数、cid<12(2 task)/cid≥12(1 task) 混合无关；init/destroy 收支不变。**已由扩展 R11 压力门实测收敛（7 配置 ×12 复跑 nondet=0，含 batch_tasks=1/2/5/6/7/8 与 num_tb=1/2）**；
⑧备份 = `history_version/mha_impl_r3_perf9.py`（md5 e80f7f65 = 改前基线 = 会话 4 终态）；采纳态 = `history_version/mha_impl_r3_perf10.py`（md5 3d56c5dd3d8a60d26eed7ade2e404a2b，1086 行）。
[RESULT-#R5-1] 优化点: **Form A single_kv_block 任务批处理**（V 侧分派门去掉 `and not form_a` 单行 + C 侧 Form A 补批镜像；槽 i ↔ task `my_start+kb*NUM_STAGES+i`；C/V 同式 `num_tb`/`batch_tasks`）
| 精度: **pass**
  - 探针门 `probe_r51_forma_batch.py`：**30/30 PASS**（/tmp/opencode/r5_probes/probe_r51.log）
    · PART 0 host-plan 审计：19 个批处理 shape 全部确认落在 `Form A ∧ single_kv_block`（batch_tasks 覆盖 **{1,2,3,4,5,6,7,8}** 奇偶全备、num_tb 覆盖 {1,2} 含 8+5 / 8+4 / 8+1 跨轮、P1 分池与 P2 共池各半、fp16/bf16、causal/非 causal、S 尾块(S%64≠0)、S_kv 尾块(need_bound_mask)、双尾块、cross(S≠S_kv)、zeros、显式 scale=0.08838、大幅值域）；11 个守卫 shape 全部确认**不**落在改动域；
    · PART 1 批处理路径：**19/19 对改前 BITEXACT** + golden 阻塞口径全过（matched=1.00000；最差 MARE_abs=1.95e-3 vs gate 7.81e-2 = 40x 余量；zeros 位级 0.0）；
    · PART 2 守卫：**11/11 BITEXACT**（含 case7 Form A 多块、case14/16/17 Form B 单块已批处理、case9/10 Form B 多块）。
  - l0 **10/10**（exit 0，/tmp/opencode/r5_probes/r5_l0_iter1.log）；l1 **19/19**（exit 0，r5_l1_iter1.log）；覆盖门禁 **19 PASS / 0 MISS / 0 N/A**（exit 0，r5_coverage_iter1.log）；`--level all` **exit 0 "Test Passed!"**（L0 10/10 + L1 19/19 + L2 12/12 正确拒绝 + Boundary 6/6 且 0 BOUNDARY_WARN + 哨兵 4/4 PERF_PASS，kernel_path 6.5s ≤120s 门；r5_all_iter1.log）。
  - 位级诊断锚 **10/10 BITEXACT vs Developer 终态**（改前态 r5_anchor_iter1.log + **终态新鲜复跑** r5_anchor_final.log，零失配）。
  - R11 竞态压力：继承 4 配置 ×12 复跑 **nondet=0 PASS**；**本轮新增扩展压力 7 配置 ×12 复跑 nondet=0 PASS**（`stress_r51.py`，r5_stress_iter1.log / r5_stress_final.log）——因继承的 4 配置全部落在 generic/Form B 路径、**不覆盖新批分支**，故按"压力门必须打在被改路径上"的原则补齐（覆盖 batch_tasks=1/2/5/6/7/8、num_tb=1/2 跨轮、P1/P2、双 dtype、causal、双尾块、my_count 非齐次 r_tasks>0）。
  - 精度红线四项复核：score 双舍入栅格链**逐字未动**（4×cast 链在 emit_v 批分支内原样，r16 落点仍在 V 侧）✅；单块预归一化特化保留（批分支 `sumexp_is[i]+1e-30 → broadcast → div` 与 generic 特化同序，i 为槽=task 索引）✅；sumexp+1e-30 全掩码行保护保留 ✅；掩码链语义等价（批分支 `block_N-1 > T.min(s0+offset_eff,S_kv-1)` = generic `idx*bN+bN-1>...` 在 idx≡0 下的恒等式；`row_thresh` 仍按 `s0+vid*half_M` 逐槽重算，探针 causal/bound/双尾块变体全 BITEXACT）✅。
| 性能（msprof timeline median us，warmup 5 + iters 20）:
  **目标 4 case（执行路径改变）：case1 13.03→11.56 (-11.28%) / case6 54.08→30.07 (-44.40%) / case19 35.86→23.38 (-34.80%) / case20 30.71→18.28 (-40.48%)**
  守卫 7 case（IR 逐字节未变）：case7 926.43→928.35 (+0.21%) / case2 67.39→66.81 (-0.86%) / case5 18.44→18.47 (+0.16%) / case9 1018.49→1025.54 (+0.69%) / case14 377.42→377.48 (+0.02%) / case10 34.41→35.14 (+2.12%) / case16 568.37→562.44 (-1.04%) ⇒ **7/7 全部在 ±3% 噪声带内（最大 +2.12%），零回退**
| IR 守卫（本轮新增硬门，16 shape `get_kernel_source()`）: **12/12 守卫 shape 生成码逐字节相同**（G_case7_A_multi_P2 / G_case2_A_multi_P1 / G_case5_A_cross_kv2 / G_A_multi_noncausal / G_case9_B_p3_multi / G_case10_B_d1_multi / G_case4_B_p3_kv2 / G_case15_B_d256_dec / G_case14_B_single / G_case16_B_single_mtp / G_case17_B_single_bf16 / G_B_single_d256）；4/4 目标 shape 生成码按预期改变（case1 16423→19511B、case6 15198→18196B、case19 15997→19069B、case20 14877→17921B）。基线 dump 与会话 4 `/tmp/opencode/r4_src/FormB_case9_postR41.cc`+`FormB_case14_postR41.cc` md5 一致 ⇒ 起点确为会话 4 终态。
| **camodel 机制复采（本轮新验收件，方法已恢复+复现，详见 perf_tuning/simulator_r5/README.md）**:
  · 观察①（C 核等待长于 V 核）：C FLOWCTRL 7.726→**6.564us（-1.162us, -15.0%）**，V FLOWCTRL 5.876→**3.705us（-2.171us, -37.0%）**；C wall 9.093→**7.919us (-12.9%)**，V wall 9.637→**8.487us (-11.9%)**。C 等待**占比**仍 82.9% > V 43.7% ——如实说明：该 shape 下 C 真实工作（CUBE 0.322 + MTE1 0.502 + FIXP 2.108us）本就远小于 V 的 VECTOR 3.368us，C 结构性轻载；缺陷不在占比而在这些等待**位于关键依赖链上**，本轮已移除（V 等待 -37%、V 的 VECTOR 占比 35.1%→**39.7%**）。
  · 观察②（MTE 与 VECTOR 串行）：交集 0.1297→**0.5146us（+296.7%）** = MTE busy 的 5.30%→**20.93%（×3.95）** / VECTOR busy 的 3.83%→**15.28%（×3.99）** ⇒ **io_buf[2] 双槽乒乓已被真正激活**（旁证：V MTE2 union +54.6% 而 MTE 总 union 仅 +0.5% ⇒ 搬运被摊开重叠而非增量；MTE3 union -38.8% ⇒ 两次写回在批处理 phase-2 背靠背）。
  · 观察③（VECTOR 碎片化）：段数 11→**10**；**两段真实 softmax 之间的空隙 2.684→0.582us（-78.3%）**，且空隙内容从"等对侧核 PV+epilogue 相位"变为"自身必要 DMA（P0 写出 187.6ns + w1[1] 读入 350.3ns）"。**未达成部分如实记录：微型碎片段数未实质减少**（<60ns 口径 6→6、<100ns 口径 8→8）——这些是 flag 管程 copy 模式固有的 BAR/prologue-epilogue 簿记，批处理不消除；[#R3-4] 已判决性证明削减此类指令数不缩减 wall，故本轮不以其为目标。
  · 附带（真实重叠的预期代价）：两段 softmax 各自 1537.8→1553.0/1608.1ns（+1.0%/+4.6%）= MTE2 与 VECTOR 争用 UB 带宽的签名；净 wall 仍 -12%。
| 对比（估算，**非权威**）: 按会话 4 cann-bench per-case 加速比 × (T_base/T_new) 比例外推 ⇒ case1 0.737→~0.831 / case6 0.584→~1.050 / case19 0.873→~1.339 / case20 1.013→~1.702，对 20-case 平均贡献 +0.0857 ⇒ 平均加速比 0.7813→**~0.867（估算）**。⚠ 该外推**偏乐观**（cann-bench 口径含每-launch 固定开销，本地 msprof 差值不能等比映射；会话 3/4 均记录过 ±0.02~0.04 的运行间漂移），权威数字需 cann-bench 复评（Orchestrator 决策项，本轮未执行）。
| 采纳: **YES**（判据：4/4 目标 case 提升 11.3~44.4%，全部远超 3% 噪声带；7/7 守卫 msprof 在噪声带内 + 12/12 守卫 IR 逐字节未变；精度 6 门全过含终态新鲜复跑；机制层三项观察中 ②③ 已量化改善、① 绝对等待时间双侧下降）
| 新基线 = `msprof_r5_iter1/`；备份链 `history_version/mha_impl_r3_perf9.py`(改前 e80f7f65) → `r3_perf10.py`(采纳态 3d56c5dd)；证据 `msprof_r5_base/*.json`、`msprof_r5_iter1/*.json`、`perf_tuning/simulator_r5/`、`/tmp/opencode/r5_probes/*.log`、`/tmp/opencode/r5_src_{base,new,final}/`

### 迭代 r5-2：[#R5-3] 同 (b,h) 相邻槽 K/V L1 驻留（跳过同字节冗余重读）（2026-09-21）

[ORDER-CHECK] 准备实施: [#R5-3] K/V L1 驻留 | 前置依赖: [#R5-1 ✅ 已采纳（批处理把同 (b,h) 相邻 task 折进同一批的相邻槽，条件式按新循环结构书写；且归因已隔离）] | 结论: ✅
[IMPL-#R5-3] 已阅读 optimization-guide.md §2.4 Full-Load（[#R3-9] 曾以"K/V 跨任务无复用"排除——S5-Step 1 的 trace 地址级证据证伪该断言于 single_kv_block 子类）；写法先例 = 本文件 [#R3-2]（runtime-guarded copy，10/10 BITEXACT，机制知识入库 S3-3-3②）+ 本文件既有 runtime-if-around-T.copy 两处（emit_v `if rows_vid > 0: T.copy(out_half→Output)`、Form A generic `if i < 2: T.copy(q_l1→l0a_s)`）；API 不猜：`T.And` = `tvm.tir.expr.And`，仓内先例 `examples/gemm/example_gemm_intrinsic_persistent.py:76`（`init=T.And(k == 0, kk == 0)`），并以 `probe_tir_bool.py` 实测确认 `T.And(a,b)` 与 Python `a and b`（两操作数均为 TIR）解析出**完全相同**的 TIR `if a and b`（rev2-A1 的 BoolOp 警告只针对 "Python bool 混 TIR"，不针对 "TIR 混 TIR"）。
关键约束:
①**命中条件的正确性论证**：`task = my_start + kb*NUM_STAGES + i`，`bx = task % s_blocks`，`task // s_blocks = bz*N + by` ⇒ `bx != 0` ⟺ `task//s_blocks == (task-1)//s_blocks` ⟺ 前一个 task 与本 task 同 (bz,by)（逐式验证：bx>0 时 (task-1)//s_blocks = g；bx==0 时 = g-1）。单块路径下 `valid_n = min(block_N, S_kv)` 为编译期常量 ⇒ `K[bz, 0:valid_n, by, :]` / `V[bz, 0:valid_n, by, :]` 由 (bz,by) 唯一确定 ⇒ 命中即**逐字节相同**，跳过重载后 k_l1/v_l1 内容仍正确。
②`task > my_start` 为**必要**守卫：本核首槽（task==my_start）时 k_l1/v_l1 尚未装填，即使 bx≠0 也必须真载（my_start 不必是 s_blocks 的整数倍）。
③**跨批（kb≥1, i=0）同样成立**：此时前一 task = 上一批最后一槽，由同一核装载；k_l1 在批间只被 K copy 写、v_l1 只被 V copy 写，且批 kb 的 GEMM1 只触碰 q_l1/k_l1（不动 v_l1）⇒ GEMM2 首槽判据仍然指向"上一次 V 装载所属 task"。
④**令牌收支零变更**：`SIG_K_L1`/`SIG_V_L1` 的 `wait(MTE1→MTE2)` 与 `set(MTE2→MTE1)` 一律**留在条件外**原样执行，仅把 `T.copy` 包进 runtime if ⇒ 每槽仍是 wait 1 / set 1，与 #R5-1 的守恒推演逐项相同（区别于 [#R3-2] 的"wait+set 同进同出"写法——本项更弱，风险更低）。
⑤**零算术变更**：不改任何数值 op、不改 buffer 形状/dtype/数量、不改 SEM/SIG ID、不改 L1 预算 ⇒ 验收门 = 对 #R5-1 采纳态（r3_perf10）**全形态 BITEXACT**。
⑥**作用域限定 = Form A 批处理分支**：(a) Form A generic（多块）路径下 task-outer/kv-inner 的循环序使 task t+1 的 idx=0 与 k_l1 中残留的 task t 的 idx=kv_iters-1 不对应 ⇒ 不改循环序就无法命中，属结构级变更，**本项不做**；(b) Form B 批处理分支的 cann-bench 单块 case（12/13/14/16/17）全部 s_blocks=1（S≤2）⇒ bx 恒 0 ⇒ 该分支加守卫为恒不命中的死代码，按最小必要改动**不加**（Form B 全 D=256 prefill 单块形态 s_blocks>1 时可受益，列为后续候选）；(c) 由此 **12 个守卫 shape 的 IR 仍须逐字节不变**，4 个目标 shape 的 IR 会变。
⑦**顺带记录（不在本项实施）新发现**：多块 Form A（case7/8）的 K/V 冗余远大于单块——s_blocks=16 ⇒ 连续 16 个 task 共享同一 (bz,by)，每 task 的整条 K/V 块序列（causal 裁剪后 ≤8 块）被重复装载 ⇒ 理论上 ~16× 冗余。但消除它需要 **kv-outer / q-task-inner 的循环互换**，而 online softmax 的 acc_o 是 per-q-task 的跨 kv 累加状态，互换后需同时持有 16 份 acc_o ⇒ 属 DESIGN 级重构（且 UB 预算不允许），非 Stage 3 范畴。[#R3-9] "K/V 跨任务无复用" 的断言据此**正式修正**为："单块子类存在完全冗余（本项已消除）；多块子类存在冗余但受 online-softmax 累加状态约束不可在 Stage 3 内消除"。
[KERNEL-REUSE-AUDIT] 依据: Part A5 [#R5-3] + [IMPL-#R5-3]① | 维度及结论: 调用域不扩（仅既有 Form A 单块批分支内新增一个 runtime 谓词守卫）✅；命中条件已逐式证明等价于"同 (bz,by) 且非本核首槽"✅；数据逐字节相同 ⇒ 位级不变为硬验收门 ✅；令牌收支不变（wait/set 留在条件外）✅；布局/数量/预算零变更 ✅；unknown: runtime-if 包 GM→L1 ND2NZ copy 的 codegen 正确性（本文件既有先例为 UB→GM 与 L1→L0A，GM→L1 为新组合）→ 由探针 bit-diff + 扩展 R11 压力实测收敛 | 结论: **repair-kernel-first**
[SELF-CHECK] 本次 Edit 只涉及 [#R5-3]：①Form A 批分支 GEMM1 槽内 `T.copy(K→k_l1)` 包进 `if T.And(task > my_start, bx != 0):`（SIG_K_L1 的 wait/set 留在条件外）+ 12 行注释；②GEMM2 槽内新增 `bx = task % s_blocks` 一行 + `T.copy(V→v_l1)` 同式包条件 + 6 行注释；③其余 1082 行逐字节未动（diff 核验：仅 2 处 `+if T.And(...)` / `-T.copy(...)`→`+    T.copy(...)` / `+bx = ...` 与注释行）；④ruff check + format clean；⑤备份 = history_version/mha_impl_r3_perf11.py（md5 3d56c5dd = #R5-1 采纳态）。
[RESULT-#R5-3] 优化点: 同 (b,h) 相邻槽 K/V L1 驻留（runtime 谓词跳过同字节 GM→L1 重读）| 精度: **FAIL — 非确定性数据损坏** | 采纳: **NO — 回滚**（rollback_reason: 精度门硬失败，按 Stage-3 规则"任意调优结果若导致精度失败必须回滚"）
**证伪证据链（4 次实验，逐步收窄）**：
1. **探针门 FAIL**（probe_r53_kvresidency.py，/tmp/opencode/r5_probes/probe_r53.log）：17 个改动域 shape 中前 6 个即全部 `vs_R51=DIFF`（max|d| 3.1e-01~4.5e+00），golden matched 跌至 0.376~0.923。PART 0 静态槽审计先行确认覆盖充分：1455 槽中 SKIP=923 / LOAD-bx0=220 / LOAD-first=312（其中 **14/17 shape 命中承重守卫** task==my_start ∧ bx≠0）/ SKIP-xbatch=36。
2. **损坏为非确定性（竞态特征，非谓词逻辑错）**：同一 zeros 输入两次运行，坏 (b,n) slice 数分别为 3/64（bisect_r53.py）与 24/64（diag_r53.py）；且 24-slice 集合中包含**完全无 SKIP 槽**的 slice（如 (0,3) = task6 LOAD-bx0 + task7 LOAD-first）⇒ 不能用"跳过了不该跳的槽"解释。
3. **机制本身无罪（判决性实验）**：把守卫换成运行期恒真的 `if task >= my_start:`（同样的 runtime-if 包 GM→L1 copy、同样的 flag 位置），3/3 shape **BITEXACT vs #R5-1** ⇒ "runtime-if 包 GM→L1 copy + wait/set 留在条件外"的写法**成立**，缺陷只在"真的省略那次 DMA"。
4. **谓词与生成码均正确（零成本核验）**：生成码条件为 `if (((cid+condval_10) < ((((kb*8)+cid)+condval_11)+i)) && (0 < (((cid+condval_12)+i) % 2)))`，即 `my_start < task ∧ 0 < task % s_blocks`（`kb*8` 项被消去是因 8kb ≡ 0 mod s_blocks=2，数学等价）；`SetFlag/WaitFlag` 全部在分支外（生成码逐行核验）；L1 无别名（生成码 `GetWithOffset`：q_l1@0 / k_l1@8192 / v_l1@24576 / p_l1@40960，各 8/16/16/16KB 不重叠）。
5. **K-only 变体在真实数据下同样失败**（bisect2_r53.py，6 shape 全 DIFF）：先前"K-only 时 case20 BITEXACT"是**假阴性**——case20 为全零输入，K 恒零 ⇒ 错误跳过不可见。改用非零数据后 P2（k_l1=[128,128]）与 P1（k_l1=[128,64]）**全部失败**（bad_elems 80859/131072 ~ 548560/1048576）⇒ 与 L0 池分类（P1/P2）、与 K/V 哪一侧无关，是**"省略一次冗余 GM→L1 DMA"这件事本身**在手动 pipe-pair flag 协议下不安全。
**根因状态（如实声明：未在本轮预算内定案）**：可观测事实 = 省略 DMA 后出现非确定性竞态；已排除 = 谓词逻辑、生成码条件、flag 位置、L1 别名、runtime-if 机制、P1/P2 差异、K/V 侧差异。**未验证的首要假设**：`SetFlag<HardEvent::MTE2_MTE1>(id)` 是插入 MTE2 指令流的排空标记，当该槽 MTE2 流在 `WaitFlag<MTE1_MTE2>` 与 `SetFlag<MTE2_MTE1>` 之间**为空**时事件即刻触发，可能使 MTE1 侧消费者对 k_l1/v_l1 的读与本核**上一槽尚未真正完成的 ND2NZ 多指令 DMA 序列**（`TileCopyTla` 对 zN fractal 目标可能发射多条 DataCopyPad）失去序关系。**下一步入口（留给后续会话）**：先做**算子外最小复现**（单 L1 tile + 2 槽循环 + 第 2 槽省略重载 + MTE1 消费），再对照 [#R3-2] 的"wait+set 同进同出"写法（把 SIG_K_L1/SIG_V_L1 的 wait+set 一并放进条件内，同时把 MTE1 消费侧的 wait 也条件化——令牌收支仍平衡）验证是否规避。
**回滚执行与核验**：`cp history_version/mha_impl_r3_perf11.py mha.py` ⇒ md5 **3d56c5dd3d8a60d26eed7ade2e404a2b** 与 #R5-1 采纳态一致；ruff clean（1086 行）；**IR 级完整还原核验：16/16 shape `get_kernel_source()` 与 #R5-1 采纳态逐字节相同**（`diff -rq /tmp/opencode/r5_src_final /tmp/opencode/r5_src_rollback` 无差异）；`mha.py` == `history_version/mha_impl_r3_perf10.py`。备份链：perf11（=#R5-1 采纳态，本轮改前）/ perf12（=回滚还原态，同体 3d56c5dd）。失败变体留存：/tmp/opencode/r5_probes/mha_r53_both.py（双侧守卫）、mha_r53_konly_FAILED.py（仅 K 守卫）。
**#R3-9 断言修正（保留，与本轮实现失败无关）**：`optimization_log:523` 的"K/V 跨任务无复用"对 single_kv_block 子类**确实不成立**（trace 地址级证据：core0 task1/task2 的 K 源地址同为 0x11351c00、V 同为 0x11391e00；case6 更强——s_blocks=8 且任务连续 ⇒ 一个批内 8 槽共享同一 (bz,by)，**7/8 的 K 与 V 装载在数据层面是同字节冗余**）。**冗余是事实，消除它的手段（省略 DMA）在本框架下不安全** ⇒ 该优化点状态从"证据性放弃（[#R3-9] 误判）"更新为"**冗余已证实、消除手段已证伪、待最小复现定案根因**"。
**#R5-1 采纳后的新瓶颈读数（camodel core0，供后续会话定位）**：C 侧 **MTE2 = 3.961us = C wall 的 50.0%**（单一最大段 [2.971..6.022] = 3050.8ns，内容 = Q1,K1,Q2,K2 背靠背），已成为两侧全部真实工作管中的最大项；C FIXP 2.108us(26.6%)、C CUBE 0.322us(4.1%)、V VECTOR 3.368us(39.7%)。即 **#R5-3 想削减的正是这个新头号项**——方向正确、手段受阻。

---

# 会话 6（perf_session=6，2026-09-21，Expert rev3 轨第四轮；软预算 3-4，备份链续编 r3_perf{N} N≥13）

> 对象：Expert 版 mha.py（会话 5 终态 = #R5-1 采纳态，md5 3d56c5dd，1086 行，cann-bench 71.40 / 0.8721x）。
> 触发：用户提案「L0/L1 显式多缓冲 + MTE2 预取 + 20 核深流水」→ mode=analysis 可行性分析（本轮前置，结论：提案主体已实现/已证伪，规则内幸存候选 = #R5-2 O 直写 / #R5-3 安全重试 / 批处理路径 L1 DB 低 EV 变体）→ Orchestrator 采纳排序并授权执行序：Iter-1 #R5-2（硬前置门 = L0C→BSHD 4D 运行期行切片独立探针）、Iter-2 批处理路径 q/k/v L1 双槽+批间预取（硬门 = camodel 必须看到 MTE2 段结构变化）、Iter-3 #R5-3 根因调查（仅预算有余；根因未定案不上 kernel）。
> 禁止项（沿用）：重做已证伪实验 / rev4 设计级变更 / 触碰 Form B decode 路径。
> 工作流同会话 3/4/5（skill 5 步 + 6 子步骤门禁 + 会话 5 验收件全集含 IR 逐字节守卫与 camodel 机制复采）。

## S6-Step 1 基线（同会话内采集，baseline=3d56c5dd 新鲜复测）

- 位级锚改前新鲜复跑：**10/10 BITEXACT** + R11 继承 4 配置压力 nondet=0（/tmp/opencode/r6_probes/r6_anchor_base.log）。
- msprof 基线（msprof_r6_base/，median us，warmup 5 + iters 20）：**case1 11.26 / case6 29.91 / case19 23.46 / case20 18.49**；守卫 case2 67.29 / case5 18.39 / case7 923.98 / case9 1024.20 / case10 35.21 / case14 377.97 / case16 561.41。与会话 5 记录（case1 11.07/case6 29.66/case19 22.98/case20 18.16）差 +0.2~+2.1% = 已归档的跨会话方差带内 ⇒ 环境一致；采纳判定一律用本会话内对比。
- 备份：history_version/mha_impl_r3_perf13.py（= 改前基线 3d56c5dd）。

## S6-Step 2/3 瓶颈定位与 Part A6（承接 analysis 报告，不重扫）

analysis 报告（本轮 mode=analysis 产物，Orchestrator 已采纳）结论直接引用：四路径形态均非"单缓冲退化"；case1 族（Form A 单块批处理）残余头部空间 = ①输出侧 V-epi 第四相位（#R5-2，0.93us/task + w3 fp32 往返 + WS3 SEM 对）②输入侧 C MTE2 冗余字节（#R5-3，竞态受阻）③批间轮边界（#R5-4）。本轮候选清单 = Orchestrator 授权执行序（Iter-1/2/3），Part A 逐项评估见 analysis 报告 §四（不重复誊写）；[ORDER-PLAN] = 授权序本身。

## S6-Step 4 逐项实施记录（会话 6）

### 迭代 r6-1：[#R6-1 = 归档候选 #R5-2] 单块 O 直写 FIX(L0C)→Output（2026-09-21）

[ORDER-CHECK] 准备实施: [#R6-1/#R5-2] 单块 O 直写 | 前置依赖: 硬前置门 = L0C→BSHD 4D 运行期行切片独立最小探针（Orchestrator 授权条款）✅ 8/8 PASS | 结论: ✅
[IMPL-#R6-1] 已阅读（不凭记忆猜 API）: src/op/ascend.cc:531-543（l0c2gm lowering：发射 compute_strideN(dst) + validRow_dst/validCol_dst——validRow 为运行期 Select 表达式，:410-418）、ascend.cc:153-166（compute_strideN 对 extents=[1,valid_m,1,D] 模式返回 N*D = BSHD 行步长；与生产已验证的 ub2gm 写回同源）、src/tl_templates/ascend/common.h:190-213（copy_l0c_to_gm<T1,T2,LayoutGM,srcM,srcN,enRelu>(dst, src, realDstN, realTailM, realTailN, unitFlag)——运行期 tailM/tailN/dstN 形参原生支持；T1=fp32/T2=fp16 时 fixpipe 做 RNE cast）、tilelang/language/copy_op.py:257-400（unit_flag/real_k/real_n 可选尾参；默认 None 时 6 参调用逐字节不变）、先例 examples/fusedmoe/example_fusedmoe.py:628（L0C→GM 2D 整块）、examples/deepseek_v4/lightning_indexer.py:756-764（L0C→GM 4D 编译期行界 + unit_flag=0b11 融合先例）、examples/gemm/example_gemm_intrinsic_persistent.py:37-85（C-only Expert 探针骨架）。
关键约束:
①**作用域 = Form A ∧ single_kv_block 批处理路径**：C 侧仅改批分支 GEMM2 段（generic 多块与 Form B 全域零触碰）；V 侧 emit_v 批分支 phase-2 以 parse-time `if not form_a:`（闭包 Python bool，与既有 `if not io3_is_alias:` 同机制）包裹——Form B 语句序列不变 ⇒ IR 逐字节守卫可证。
②**正确性依据（单块特化数学）**：single_kv_block ⇒ phase-1 已做预归一化（P̂ = P/(sumexp+1e-30)，§3.2.1-⑤）⇒ w3 = P̂@V 的 fp32 累加即最终 O，phase-2 仅做 acc_o=0+w3 → vcast → 写回（纯搬运相位）⇒ C 侧 FIX 直写 Output 语义等价。位级等价 = fixpipe fp32→dtype RNE ≡ V 侧 vcast（#R3-3 探针 11/11 BITEXACT 已入库）+ (0.0+x)≡x（-0.0 边界域内不可达，analysis §六归档论证；zeros/大幅值域探针覆盖）。
③**尾行语义**：C 直写 Output[bz, s0:s0+valid_m, by, :]（realTailM=valid_m 运行期）——一次写全部 valid_m 行，无 vid 拆分；l0c 垃圾行（≥valid_m）被 tailM 截断不写（探针 sentinel 验证无越写）。
④**SEM 收支镜像删减（§7.2-5）**：删 C 批头 wait(WS3_V2C) + C 槽级 set(WS3_C2V) + V phase-2 槽级 wait(WS3_C2V) + V phase-2 批尾 set(WS3_V2C)；V-init 的 set(WS3_V2C) 预置保留 = 悬空但无害令牌（与既有 WS1_V2C 设计内悬空 +1 同类；跨核信号量按 launch 复位——生成码核验：稳态协议中 id4/id5 wait/set 全消失，仅存 init set(5)）。WS1/WS2 协议逐槽不变。
⑤**核内令牌守恒**：SIG_L0C±side 管程原样（wait(M→FIX)/copy/set(FIX→M) 每槽净 0，仅 copy 目标由 ws3 换 Output）；SIG_V/P_L1、SIG_L0AB 逐字未动；V 侧 phase-2 删除后 sig_io3/SIG_S_HALF 的 init-set/destroy-wait 收支仍平衡（phase-2 槽内对为净 0 变换）。
⑥buffer/形状/dtype/host 签名/workspace_idx 零变更（workspace_3 在该路径不再被写但仍分配——host 契约静态，足迹不变）。
[前置门探针] probe_r52_l0c_bhsd.py（算子外独立最小 kernel，C-only Expert，flag 编排逐字复刻 Form A GEMM2 模式）：**8/8 PASS**（/tmp/opencode/r6_probes/probe_r52.log）——整数精确模式 4/4 **BITEXACT**（S 尾块 valid_m=26/30/62 运行期行切片、fp16+bf16、D=64/128、多核任务循环 side 轮转、sentinel 999.0 全覆盖无漏写、邻块无越写）；randn 容差模式 4/4 matched=1.000000（多任务 ×5 复跑 nondet=0）；codegen 核验：`copy_l0c_to_gm<float, half, RowMajor, 64, 128, 0>(Out[运行期 4D 偏移], l0c[slot], realDstN=N*D, realTailM=运行期, realTailN=D)` ✓。
[KERNEL-REUSE-AUDIT] 依据: Part A6/analysis §四-1 | 维度及结论: 调用域不扩（host 契约/分派谓词/buffer 全不变，仅既有 Form A∧single_kv_block 路径的输出段重构）✅；C/V SEM 收支镜像删减已推演（约束④，生成码核验）✅；核内令牌守恒（约束⑤）✅；位级等价链 = #R3-3 已证 RNE 等价 + 探针 8/8 ✅；精度红线：双舍入栅格链在 phase-1（未触碰）、预归一化特化保留（正是本项等价性依据）、sumexp 保护未动、掩码链未动 ✅ | 结论: repair-kernel-first（最小 diff：C 批分支 GEMM2 段 + emit_v phase-2 包裹）
[SELF-CHECK] 本次 Edit 只涉及 [#R6-1]：①C 侧 Form A 批分支 GEMM2：删批头 wait_cross_flag(SEM_WS3_V2C)、槽内补 bx/s0/valid_m 三行、`T.copy(l0c_o[side], workspace_3[cid,i,:,:])` → `T.copy(l0c_o[side], Output[bz, s0:s0+valid_m, by, :])`、删槽级 set_cross_flag(SEM_WS3_C2V) 条件块（+19 行注释）；②emit_v 批分支 phase-2 整段（33 行）包进 `if not form_a:`（+4 缩进 + 9 行注释）——phase-2 语句逐字未动；③docstring 同步（Form A single_kv_block 段 +8 行说明）；④其余全部逐字节未动（Form A generic C 循环、Form B prim_func、emit_v phase-1/generic/init/destroy、host 包装、SEM/SIG 矩阵、allocations）；⑤ruff check + format clean（1113 行）；⑥备份 perf13（改前 3d56c5dd）/ perf14（采纳态 bbfcbcaf）。
[RESULT-#R6-1] 优化点: **单块 O 直写 FIX(L0C)→Output（删 V phase-2 + WS3 SEM 对 + w3 fp32 GM 往返）**
| 精度: **pass（门禁全过）**
  - ① 前置门探针 8/8（见上）+ 位级探针矩阵 probe_r61_o_direct.py **30/30 PASS**（/tmp/opencode/r6_probes/probe_r61.log）：PART 0 host-plan 审计 19/19 目标 shape 均落 Form A∧single_kv_block（batch_tasks 覆盖 {1,2,3,4,5,6,7,8}、num_tb {1,2} 含 8+5/8+4/8+1 跨轮、P1/P2、双 dtype、causal/非 causal、S 尾块/S_kv 尾块/双尾块、cross、zeros、显式 scale、大幅值域）+ 11/11 守卫确认不在改动域；PART 1 变更域 **19/19 对改前 BITEXACT** + golden 阻塞口径全过（最差 MARE_abs 1.95e-3 vs gate 7.81e-2 = 40x 余量；case20 zeros max_abs=0.00e+00 位级零）；PART 2 守卫 **11/11 BITEXACT**（Form A 多块 ×5、Form B 单块 ×4、Form B 多块 ×2）。
  - ② 位级锚：改前基线新鲜复跑 **10/10 BITEXACT** + 改后终态新鲜复跑 **10/10 BITEXACT**（r6_anchor_base.log / r6_anchor_iter1.log，零失配）。
  - ③ l0 **10/10** exit 0（r6_l0_iter1.log）；④ l1 **19/19** exit 0（r6_l1_iter1.log）；⑤ 覆盖门禁 **19 PASS / 0 MISS / 0 N/A** exit 0（r6_coverage_iter1.log）；⑥ `--level all` **exit 0 "Test Passed!"**（L0 10/10 + L1 19/19 + L2 12/12 正确拒绝 + Boundary 6/6 + 哨兵 4/4 PERF_PASS，kernel_path 6.0~7.3s ≤ 120s 门；r6_all_iter1.log）。
  - ⑦ 竞态压力：R11 继承 4 配置 ×12 复跑 **nondet=0 PASS** + 扩展 7 配置（全部打在被改路径：batch_tasks {1,2,4,5,6,7,8}、跨轮 8+5、P1/P2、双 dtype、causal、双尾块、my_count 非齐次）×12 复跑 **nondet=0 PASS**（r6_anchor_iter1.log / r6_stress_ext_iter1.log）。
  - 精度红线四项复核：双舍入栅格链逐字未动（phase-1 内，本项零触碰）✅；单块预归一化特化保留（且为等价性依据）✅；sumexp+1e-30 保护保留 ✅；掩码链语义等价（未触碰）✅。-0.0 边界：探针 zeros/大幅值域/整数精确模式未触发（int16 位级比较通过）。
| ⑧ IR 守卫（16 shape get_kernel_source()，/tmp/opencode/r6_src_new vs r5_src_final）：**12/12 守卫 shape 生成码逐字节相同**（md5 与基线 manifest 全等，含 Form B 全域 = decode 不回退硬约束的 IR 级证明）；4/4 目标 shape 按预期改变（case1 19511→17604B / case6 18196→16345B / case19 19069→17224B / case20 17921→16079B）。生成码抽验（T_case1）：新 FIX 发射 `copy_l0c_to_gm<float, half, RowMajor, 64, 64, 0>(Output[bz*S*N*D + s0*N*D + by*D 运行期偏移], l0c_o[slot], realDstN=512=N*D, realTailM=64, realTailN=64, 0)` ✓；稳态 cross-core 协议 15→11 op（id4/id5 wait/set 全删，仅存 V-init 悬空 set(5)）✓。
| ⑨ camodel 机制复采（simulator_r6/case1_r61，方法同 simulator_r5/README；对比基线 = case1_r51 = 本轮改前态；表内 "#R5-1" 列标签为工具硬编码，实为 #R6-1 新态）：
  · **V-epi 第四相位消失（本项靶心）**：V MTE2 1.591→**1.156us(-27.3%)**（w3 读取段删除）、V MTE3 0.868→**0.322us(-62.9%)**（Output 写回移交 C，剩余为 w2 P 写）、V VECTOR 3.368→**3.177us(-5.7%)**（phase-2 fill/add 删除）、VECTOR 段数 10→**8**、微型碎片 8→**6**（phase-2 两段微碎片消失）。
  · **V 侧等待大减**：V FLOWCTRL 3.705→**1.979us(-46.6%)**；**V wall 8.487→7.336us(-13.6%)**。
  · **如实记录（预期代价）**：C wall 7.919→8.187us(+3.4%)、C FIXP 2.108→2.209us(+4.8%)——C 核接管 Output FIX 直写（BSHD 跨步目标）后成为新 pacing 侧（max(C,V) 由 8.49(V) 变 8.19(C)，camodel 口径净 wall -3.5%；真机 msprof -10.66% 收益更大）。该读数为 #R5-3 类输入侧字节削减候选提供直接依据（C MTE2 3.963us 仍为 C 侧最大真实工作管，现在打在关键墙上）。
  · MTE∩VECTOR 重叠率 20.93%→4.68%：**非串行化回退**——V 侧 MTE 总量 -39.9%（w3 段删除）使可重叠对象减少，inter-softmax 空隙 0.582→0.549us(-5.7%) 持平偏好、softmax 段长不变（1554/1535ns，golden 栅格链未动旁证）。
| ⑩ 性能（msprof timeline median us，同会话基线对比，warmup 5 + iters 20，msprof_r6_iter1/）:
  **目标 4 case：case1 11.26→10.06 (-10.66%) / case6 29.91→24.51 (-18.05%) / case19 23.46→21.53 (-8.23%) / case20 18.49→14.32 (-22.55%)**
  守卫 7 case（IR 逐字节未变）：case2 67.29→66.22 (-1.59%) / case5 18.39→18.66 (+1.47%) / case7 923.98→930.21 (+0.67%) / case9 1024.20→1011.19 (-1.27%) / case10 35.21→35.19 (-0.06%) / case14 377.97→376.28 (-0.45%) / case16 561.41→563.67 (+0.40%) ⇒ **7/7 全部 ±1.6% 噪声带内，零回退**；decode 不回退硬约束的 IR 级证明 = Form B 生成码逐字节未变。
| 对比（估算，**非权威**，权威 = Orchestrator 轮末 cann-bench 复评）: 按 §S5-8 per-case × (T_base/T_new) 外推 ⇒ case1 0.894→~0.989 / case6 1.083→~1.284 / case19 1.352→~1.471 / case20 1.714→~2.101，20-case 平均贡献 +0.048 ⇒ ~0.920x（外推偏乐观警示同 §S5-2）。
| 采纳: **YES**（判据：4/4 目标 case -8.2~-22.6% 远超 3% 噪声带；7/7 守卫噪声带内 + 12/12 守卫 IR 逐字节；精度门禁①-⑦全过含双新鲜锚与打在被改路径的扩展压力；机制层⑨靶心签名全确认）
| 新基线 = msprof_r6_iter1/；备份链 perf13（改前 3d56c5dd）→ perf14（采纳态 **bbfcbcaf1e438fa69664b4a7770e6d1c**，1113 行）；证据 msprof_r6_base|iter1/*.json、simulator_r6/{case1_r61, r61_mechanism_cmp.txt}、/tmp/opencode/r6_probes/*.log、/tmp/opencode/r6_src_{new}/。

### 迭代 r6-2：[#R6-2] Form A 单块批处理路径 q/k/v L1 双槽 + 批内/批间预取（用户提案幸存变体）（2026-09-21）

[ORDER-CHECK] 准备实施: [#R6-2] 批处理路径 q/k/v L1 双槽 + 预取（授权 Iter-2）| 前置依赖: [#R6-1 ✅ 已采纳（新基线 perf14 bbfcbcaf）] | 结论: ✅
[IMPL-#R6-2] 已阅读: [IMPL-#R3-2]（3D [2,·,·] L1 + make_zn/nz 前导槽位维透传机制，probe 10/10 BITEXACT 已入库 S3-3-3②）、[IMPL-#R4-2b]（"预载置 SEM wait 前——GM 输入无 ws 依赖，SEM 延迟被载入吸收"论证 + 零新增 flag 思想）、optimization-guide §2.3（首轮预取→正式循环三段结构）、mha.py 批处理分支现状（#R6-1 后）。关键约束: ①**不跳过任何 DMA**——每槽装载自己的字节（#R5-3"条件省略 DMA"竞态域零触碰；本项纯发射时序/槽位重排）；②所有 runtime 守卫（kb==0 批头预载 / i+1<batch_tasks 批内预取 / kb+1<num_tb 批间预取）内包**完整 wait+copy+set 三元组**（同进同出，#R3-2 已证安全形态，§7.2-5）；③令牌记账恒等式：#预取(slot s) = #消费(slot s) = 总槽数 ⇒ init 每槽 1 writable 预置 / destroy 每槽 1 wait（7 令牌 vs generic 4）；④SIG id 批处理内核专用重排 K:0/1 V:2/3 Q:4/5 P:6（#R3-2 重排先例）；P 保持单槽（w2 装载被 SEM_WS2_V2C 生产者握手门控，预取会移动跨核 wait 位置——禁）；⑤分配/SIG/init/destroy 全部 parse-time 条件化（single_kv_block 三元式）⇒ generic 与 Form B 内核 IR 逐字节不变（dump 实证）；⑥V 侧 emit_v 零触碰（C 侧装载调度为核内事务，SEM 位置/槽↔任务映射不变——§7.2-5 天然满足）；⑦批间预取置于 `set_cross_flag("MTE2", SEM_WS2_C2V)` 之后（MTE2 FIFO——不得延迟 V 侧 w2 槽释放）。
[SELF-CHECK] 本次 Edit 只涉及 [#R6-2]：SIG 常量块条件重排（+16 行）、Form A L1 分配三元式（q/k/v [2,·,·]，+9 行注释）、C init/destroy 条件化（7 令牌，+16 行）、GEMM1 批头预载+批内预取+双槽消费重写（+50 行）、GEMM2 V0 前移+批内预取+双槽消费（+33 行）、批间预取块（+26 行）、docstring（+9 行）——共 +125 行（1113→1238；**如实记录：超出授权时 analysis 估算的 ~30-50 行**，主因 = init/destroy/SIG/分配的 parse-time 条件化四件套 + 批间预取独立块 + 注释纪律）。GEMM1/GEMM2 的 mma/FIX/SEM/p_l1 段、generic 分支、Form B prim_func、emit_v 全体、host 包装逐字节未动。ruff clean。
[KERNEL-REUSE-AUDIT] 依据: 授权 Iter-2 + analysis §四-4 | 维度及结论: 调用域不扩 ✅；令牌守恒推演通过（约束③，生成码核验 init 7×SetFlag<MTE1_MTE2>(0..6) / destroy 7×WaitFlag ✓）；机制先例 = #R3-2 3D L1（10/10 BITEXACT）+ #R4-2b 预载位论证 ✅；unknown: 批处理路径下双槽预取的实际收益（analysis EV 0-3%）→ 由 msprof+camodel 实测裁决 | 结论: repair-kernel-first
[RESULT-#R6-2] 优化点: Form A 单块批处理 q/k/v L1 双槽 + 批头/批内/批间预取 | 精度: **pass**（probe_r62 30/30：变更域 19/19 对 #R6-1 采纳态 **BITEXACT** + golden 阻塞口径全过（case20 zeros max_abs=0）+ 守卫 11/11 BITEXACT；IR 守卫 **12/12 逐字节相同**（含 Form B 全域）+ 4/4 目标按预期改变；生成码抽验：批头预载先于 CrossCoreWaitFlag(1)、预取守卫块 id (i+1)%2 参数化、V0 前移、P 单槽 id6、destroy 7 waits ✓）| 性能(msprof median us, vs 同会话 #R6-1 基线 msprof_r6_iter1): **case1 10.06→9.89 (-1.69%) / case6 24.51→24.65 (+0.57%) / case19 21.53→21.93 (+1.86%) / case20 14.32→14.04 (-1.96%)** —— 4/4 目标全部在 ±3% 噪声带内且**方向不一致**（两微升两微降）；IR 逐字节未变的守卫组同带幅漂移（case10 -2.47% / case2 +1.93% / case9 +1.51%）佐证该量级=测量方差 | 对比: **无可证实收益（性能无变化）** | 采纳: **NO — 回滚**（rollback_reason: 性能无变化规则；回滚至 perf14，md5 bbfcbcaf 核验 + **IR 级还原 16/16 逐字节相同**（diff r6_src_new vs r6_src_rollback 无差异））
**判决性证据（camodel case6 机制复采，simulator_r6/case6_r6base vs case6_r62，授权硬门）**：
  1. **MTE2 段结构确实改变（机制已激活）**：22→20 段；批头冷启动段 4457→5136ns（Q0/K0 预载与槽 1 预取合流）；中段 528ns 独立段合并为 2298/1486ns 簇——单槽 ownership 气泡被预取消除，装载发射更聚簇。**非"段结构逐位不变"零响应形态**。
  2. **camodel 与真机方向背离（本轮最重要机制知识）**：camodel C wall 33.19→26.89us(-19.0%)、FIXP union 30.66→23.19us(-24.4%，sum 39.92→29.83)、FIXP busy 92.4%→86.2%——预取聚簇改变了 camodel 的 FIX/MTE2 GM 带宽争用记账；**但真机 msprof case6 +0.57% 零响应**。判读：真机 case6 wall = **GM/L2 总流量墙**（w1 32KB+Output 16KB+w2 16KB+Q/K/V 64KB 每槽字节数不因预取改变；装载时序重排只改争用相位、不减总量），camodel 的 fixpipe 争用模型放大了重排收益（camodel FIXP sum 29.8us > 真机全 wall 24.65us——其 fixpipe 绝对时长模型显著慢于真机，方向性结论不可迁移）。**"msprof 为唯一性能依据"纪律（final_report §7）再次得到判决性佐证；camodel 用途限定 = 机制是否激活的定性判定，禁止用于收益定量**。
  3. 基线归因补全（analysis 阶段缺口）：case6 基线（#R6-1 态）C 侧 **FIXP 92.38% = pacing 管**、MTE2 52.11%、V 侧 FLOWCTRL 79.27%（V 等 C）——case6 的 C wall 早已是 fixpipe 吞吐墙，MTE2 预取类手段对其无杠杆（与 case7 的 FIXP 95.68% 同构）。case1/19/20（单轮批）批头装载本就带宽串行背靠背（3050.8ns 段），双槽不减少管道 busy ⇒ 0-2% 预期兑现。
**归档结论（用户提案幸存变体的终审）**：至此，「MTE2 预取 Tile i+1 / L1 显式多缓冲」提案在**全部四条路径形态**上的规则内变体均已实测收口：Form A generic（#R3-2 零响应）、Form B D256（#R4-2b 全 pipe 逐位不变）、Form B decode（带宽墙+L1 预算否决，证据性排除）、**Form A 单块批处理（本项：机制激活 + 位级安全 + 真机零响应）**。输入侧唯一剩余杠杆 = 冗余字节消除（#R5-3 域，根因未定案）；输出侧杠杆已由 #R6-1 兑现。3D 双槽 L1 + 守卫式预取的位级安全性再次入库（30/30 BITEXACT，未来结构变更后可低成本复用）。

### 迭代 r6-3：[#R6-3] #R5-3 根因调查——算子外最小复现（授权 Iter-3，时间盒限定；根因未定案，不上 kernel）（2026-09-21）

[ORDER-CHECK] 准备实施: [#R6-3] #R5-3 根因调查 | 前置依赖: [Iter-1 ✅ 采纳 / Iter-2 ✅ 完成（回滚），预算有余] | 结论: ✅（仅调查，禁上 kernel——授权条款"根因未定案 → 不上 kernel，调查报告归档即止"）
[IMPL-#R6-3] 已阅读: [RESULT-#R5-3] 证伪证据链全文（4 次实验 + 首要未验证假设 + 下一步入口）、mha.py Form A 批处理 GEMM1 K 路径（复刻对象）、trace 读数（MOV_OUT_TO_L1_MULTI_ND2NZ n=8 = ND2NZ 多指令序列实证，simulator_r5/baseline_case1_core0_tracestats.txt）。复现设计（repro_r53.py / repro_r53_v2.py，scratch）：单/多核 C-only Expert kernel，N_ITERS=13 槽（case6 my_count 同值），每槽消费同一 K 字节（跳过为数据合法），槽内指令序逐字复刻 mha 批处理 GEMM1（Q producer 三元组→K producer[变体]→L0AB/L0C 双槽轮转→mma→FIX per-slot 输出），SIG id 同构（K=0/Q=3/L0AB/L0C 0,1）；三变体：A 对照（每槽装载）/ B 失败形态（copy 在 `if i==0` 内、K wait/set 在外 = #R5-3 归档原形态）/ C 安全替代（producer 三元组 + consumer K wait/return 全部同条件 = 「wait+set 同进同出」）；检出器：fp32 参考坏槽计数（损坏）× 40 launch（非确定性）× 2 随机数据相位（陈旧数据）；数据全非零 randn（case20 假阴性陷阱纪律）。
**升级阶梯与结果（如实）**：
  1. **v1 单核 + 紧凑源**（K [1,S_kv,1,D]，行跨步=D）：A/B/C 三变体 40/40 launch 全 CLEAN（bad_slots=0，worst|d|=2.4e-6）——**B 未复现**。
  2. **v2 20 核 Fixed Core 并发 + 跨步 BSHD 型源**（K [1,S_kv,16,D] 取 head0，行跨步=16×D → 命中 MOV_OUT_TO_L1_MULTI_ND2NZ 多指令路径 = 假设的嫌疑本体）：A/B/C 三变体 40/40 全 CLEAN——**B 仍未复现**。
[RESULT-#R6-3] 优化点: #R5-3 根因最小复现调查 | 精度: n/a（未触碰 mha.py）| 性能: n/a | 结论: **根因未定案（最小复现不具备复现力）——按授权条款不上 kernel，调查归档即止**
**判决性判读**：
  1. 归档首要假设（"空 MTE2 流下 SetFlag<MTE2_MTE1> 即刻触发 × ND2NZ 多指令 DMA 失序"）在**单核与 20 核两级最小结构下均不成立**——B 形态（flag 在外、copy 条件内）在无其他核间交互时是安全的（40+40 launch 零损坏、确定性）。
  2. 由此收窄根因搜索空间：损坏需要 **mha 完整上下文**才出现，剩余候选因子 = ①跨核 SEM 节奏（WS1/WS2 wait 把 C 批处理悬停在 flag/DMA 在途状态）②V 核并发（同 task 组双 AIV 的 MTE2/MTE3 ws 流量与 C 侧 FIX/MTE2 在 GM/L2 的交叠）③GEMM1/GEMM2 双批交替（K 跳过与 V 跳过跨相位交织）④真实批轮结构（8+5 两轮、轮边界 SEM 会合）⑤真实寻址（同 (bz,by) 组内多 task 的 GM 地址重合模式）。最小复现逐级补齐这些因子的成本 ≈ 重建整个 kernel——超出"最小复现"工具的定义域。
  3. **C 变体（同进同出）在最小结构下 CLEAN 不构成其 kernel 级安全证明**——因 B 在同一结构下也不损坏，该结构对 B/C 无判别力（如实声明，防止把"未复现"误读为"C 已验证安全"）。
  4. #R5-3 状态更新：「冗余已证实、消除手段已证伪（kernel 级 4 实验）、最小复现两级未定案（本轮）」⇒ 未来重试的前置门升级为：**必须在带跨核 SEM + V 并发上下文的中间复杂度复现体上定案**（候选路径：以 mha.py 为基座做消融——删 V-scope/删 SEM/删 GEMM2 逐项降级直到损坏消失，定位最小致损因子组合；属独立调查任务，非调优迭代）。
**产物**：/tmp/opencode/r6_probes/repro_r53.py（v1 单核）、repro_r53_v2.py（v2 20 核跨步源）、repro_r53.log / repro_r53_v2.log（40×3×2 全读数）。mha.py 零触碰（本迭代前后 md5 均 bbfcbcaf）。

---

# 会话 7（perf_session=7，2026-09-22，rev4 轨首轮；软预算 3-4，备份链 r4_perf{N}）

> 对象：rev4-attempt1 基线 mha.py（1593 行，md5 ee5bf9c9，cann-bench 71.79 / 0.9108x，cann_final_eval_20260922_172117）。结构 = 三分派（form_a IR 逐字节未变 / form_b_mma 新增手动 K/N-split mma 流水 / form_b_gemm_v0 decode 兜底禁触）。锚体系：锚-A（vs Developer 6505）L0 #1-9 Form A 域 BITEXACT 硬守卫；锚-B（vs r3 bbfcbcaf）Form B-mma 域 14/14 BITEXACT、已降级为诊断工具。
> 授权：Step 0 = case9 新鲜归因（首位）；候选序 = ①w1 dtype 瘦身重试 ②§10.3-⑨ O 直写扩展评估 ③归因驱动新候选。禁令：S2 kv-outer 重提 / #R5-3 条件化装载 / decode 兜底域触碰 / camodel 收益定量。
> 执行中断说明（如实）：本轮首次执行在 Step 0 采集完成后被外部中断（orchestrator 盘点核实：kernel 零改动、camodel+op 数据已落盘）；本章节为续跑时补写，采集时间戳 2026-09-22T17:35-18:13（sim_r7_c9.log / r7_op_case9_run.log）。

## S7-Step 0 case9 新鲜归因（[ANALYSIS]，rev4 S1 新结构；旧结构 camodel/mindstudio 数据仅用于 A/B，禁作新归因）

**数据源**：①msprof op 真机归因 /tmp/opencode/r7_op_case9/（PipeUtilization.csv，与 session-4 r4_op_case9 同口径 A/B）= **唯一性能依据**；②camodel simulator_r7/case9_r7s1/（per-core trace，trace_stats busy-union 法，仅机制定性——§S6-3-3 纪律）；③OpBasicInfo Task Duration。

**（1）S1 收益机制真机实证（op 模式 old→new，block0）**：
| 读数 | old（gemm_v0 #R4-1 态） | new（rev4 S1） | Δ |
|---|---|---|---|
| Task Duration wall | 1045.7us | **955.4us** | **-8.6%** |
| aic_time | 973.7us | 908.5us | -6.7% |
| **aic_scalar_mte1_stall** | **739.4us (76%wall)** | **412.2us (45%wall)** | **-44%（每-step PIPE_M 串行链消除的直接签名，S1 靶心命中）** |
| aic_wait_id2（SEM_WS2_V2C，等 V 的 w2） | 151.0us | **323.9us** | **+114%（C 变快后更早到达等待点——瓶颈迁移签名）** |
| aic_wait_id5（SEM_WS3_V2C） | 31.4us | 79.5us | +153% |
| aic mte2 / mte1 / fixp / cube | 45.5/31.6/25.8/17.5% | **49.5**/33.7/**32.8**/19.1% | mte2 为真机最忙管 |
| aiv vec / wait_id4（等 w3）/ wait_id0（等 w1） | 43.2% / 440.7 / 326.7us | 47.2% / 423.6 / **210.0us** | w1 到达提前（V 等 w1 -36%） |
| aiv 总等待（id4+id0+gen） | 885us/974 wall (91%) | **753us/909 wall (83%)** | V 侧仍以等待为主 |

**（2）camodel 机制定性（old 会话 4 trace vs new r7s1，C core0）**：FIX_L0C_TO_DST 事件数 884→**1326（+50%：PLAN-B w3 列段 2 次 FIX 替代 1 次整块）**、单事件均时长 1.20→1.36us（列段行跨步写）、FIXP union 占 C wall 72.05%→**94.95%**；WAIT_FLAG_DEVI n=648 两侧同（协议轮次不变）但 dur 205→**480us**；MTE2 指令构成逐字不变（MOV_OUT_TO_L1_MULTI_ND2NZ n=1429 dur 737→748us）。V 侧：emit_v 零改动但 **MTE3 union 439.8→677.8us (+54%)** = C 侧列段 FIX 写模式改变 GM/L2 争用的次生效应（模型级观察）；MTE∩VECTOR 重叠 = VECTOR 的 55.0%（V 侧流水正常）。**camodel FIXP 94.95% vs 真机 fixp 32.8% = §S6-3-3 模型放大判例再现——camodel 读数仅作结构定性，禁作定量。**

**（3）归因结论（pacing 判定）**：
- **case9 新墙 = 真机 C↔V 四相位耦合链延迟 + D256 形态 GM 流量强度，而非任何单管吞吐**：真机无管超 50%（C mte2 49.5% 最忙；fixp 32.8%）；C 暴露等待 = wait_id2 0.70us/iter + mte1_stall 0.90us/iter（chunk 令牌链），V 暴露等待 = 83% wall。链结构（每任务一轮，case9 causal eff_kv 均值 4.5 ⇒ **批深仅 4.5 槽且每 4.5 iter 一次任务边界**，NUM_STAGES=8 半空转）：C-G1槽→V-P1槽→C-G2槽→V-P2槽 槽级流水 + 轮界批级 SEM 交接，固定交接成本摊薄不足。
- **为何未兑现 case7 同族 0.46x**：①**D256 流量强度**——case9 每 iter C 侧 MTE2 136KB（K/V [128,256]fp16 各 64KB + P 8KB）配 bM=32 ⇒ **4.25KB/输出行 vs case7 的 1.25KB/行（3.4×）**，mte2 49.5% 已是真机最忙管，字节不可减（条件化装载禁做）；②**PLAN-B 列段 FIX 税**——每 iter FIX 事件 2→3、真机 fixp 251→298us(+19%)、camodel V MTE3 +54% 争用外溢；③case7 的 0.46x 本身是 FIXP/ws-GM-往返墙下的数字（§S6-3-4），case9 消除 scalar 链后落入同族"ws 流量+链延迟"墙，但流量强度更高 ⇒ 0.353 < 0.46 为结构必然，非实现缺陷。
- **规则内杠杆排序（归因驱动）**：(a) **w1 dtype 瘦身**同时打链两侧——C FIX w1 字节 -50%（case9 w1 16KB/iter fp32→8KB）+ V-P1 阶段缩短（w1 读 MTE2 -50% + 省 2×[hM,bN] cast/槽）⇒ 直接压缩 C wait_id2 暴露段与 V 阶段时长；位级等价 = #R3-3 11/11 归档（fixpipe RNE ≡ rr_ub r16(qk)，重试免探针）；对 case6/1/19/20（Form A 单块，camodel FIXP 92.4% 墙）同向杠杆（scope 待 case6 真机 op 归因裁决）。(b) §10.3-③ k/v L1 双槽 Form B-mma 域（重试窗口授权条件已满足：M 管排空阴影消除后 mte2 49.5%+mte1_stall 412us 为真机约束成分；机制探针先行——3D 槽×ZN annotation×列段读组合为新形态）。(c) §10.3-⑨ O 直写扩展 = **bench 收益面为零，论证归档不实施**：form_b_mma ∧ single_kv_block 在 20 case 中无命中（case4 S_kv=256>128 两 kv 块=多块域需 acc_o 跨 kv 累加不可直写；case9 多块；case15/18 为 decode 兜底域禁触；唯一命中 = L0 #10/哨兵 (1,64,64,2,256)）——DESIGN §10.3-⑨ "case4 族"预期为设计期疏漏，如实修正。

## S7-Step 4 逐项实施记录（会话 7）

### 迭代 r7-1：[#R7-1] w1 dtype 瘦身重试（fp32→dtype，scope=form_b_mma 专属；§10.3-①/§S6-5 首位归档候选）（2026-09-22）

[ORDER-CHECK] 准备实施: [#R7-1] w1 dtype 瘦身（scope 数据驱动裁决 = form_b_mma 专属）| 前置依赖: [Step 0 归因完成 ✅] | 结论: ✅
**scope 裁决记录（先于实施，真机 op 数据驱动）**：候选 scope 三域逐一核对——①Form A generic：**排除**（#R3-3 会话 3 全域版实测 case7 +1.0% 零响应归档；本轮 case7 op 复采 C wait_id2=617.2us=70% wall 证明 C 有巨量余量吸收 V 侧节省，耦合平衡态未变，尊重证伪不重做）；②Form A 单块：**排除**（case6 真机 op 首采：C fixp 仅 28.8%、最忙管 mte2 54.1%——camodel FIXP 92.4% 判定为模型放大 §S6-3-3 再证；w1 瘦身不打最忙管，EV 小而守卫集缩减代价大）；③form_b_mma：**入选**（S7-Step0：case9 C fixp 32.8% 中 w1 占 1/3 字节 + V-P1 阶段含 w1 读与 r16(qk) cast 链，C wait_id2 324us 暴露段指向 V-P1 时长）。守卫收益：scope 收窄使 13 守卫 IR 逐字节全集保留（Form A 单块 4 个 T shape 留在守卫域）。
[IMPL-#R7-1] 已阅读: [IMPL-#R3-3]/[RESULT-#R3-3]（会话 3 全域版完整配方与三门证据：io_buf dtype 联动、io3 别名禁、模板 <T1,T2> 自动 cast、widening copy 先例 rr_ub→work_ub、RNE 等价 11/11 含 bf16+大幅值域 |score|~2264、溢出边界论证）、DESIGN §10.3-①（探针门条款）/§9.3.3-1 L1883（"w1/w3 dtype 瘦身会破坏本条，仅允许 Stage 3 探针门通过后启用"——设计内授权）、#R6-1（Output 方向 fixpipe cast 19/19 再证）。关键约束: ①r16(qk) 栅格点迁移至 C 侧 fixpipe cast，**栅格值不变**（fixpipe RNE ≡ rr_ub round-trip，双档归档实证）——精度红线#1 豁免论证 = DESIGN L1883 探针门条款 + #R3-3/#R6-1 位级证据链；②w3 保持 fp32（红线#2 未量化不触碰）；③io_buf dtype 联动 + io3_is_alias 防御性禁别名（#R3-3 约束③；form_b_mma 域 bN=128≠D=256 本就不别名，显式化不变量）；④scope 谓词 = parse-time `w1_slim = form_b_mma`，emit_v 两分支 r16(qk) 对以 `if not w1_slim:` 包裹（非 slim 内核语句序列零变化 ⇒ 守卫 IR 逐字节可证）；⑤workspace_1 签名与 io_buf alloc 以 w1_dtype 派生（C 侧 FIX copy 语句零改动，模板自动 cast）。
[SELF-CHECK] 本次 Edit 只涉及 [#R7-1]：host 派生 +w1_slim/w1_dtype（17 行含注释）、io3_is_alias 表达式（+6 行注释）、emit_v batched/generic 两处 r16(qk) 对包裹（+14 行注释）、Form B-mma prim_func workspace_1 签名 + io_buf alloc dtype（+7 行注释）、docstring 红线段同步（+6 行）——共 +40 行（1593→1633）；C 侧 copy/mma/flag/SEM 语句零改动；Form A 两分支/兜底域 prim_func/emit_v 其余全部逐字节未动；ruff clean。备份 = history_version/mha_impl_r4_perf2.py（改前 ee5bf9c9）。
[KERNEL-REUSE-AUDIT] 依据: Step 0 归因 + scope 裁决 | 维度及结论: 调用域不扩（host 契约/ws 注入框架派生 dtype 自动）✅；位级等价 = #R3-3 归档知识 + 本轮矩阵再证 ✅；锚-B 预期管理（任务指令"必失配"预案）→ **实测未失配**（14/14 BITEXACT——RNE 等价成立则锚-B 保持，预案未触发，如实记录）；unknown: 真机收益 → msprof+op A/B 裁决 | 结论: repair-kernel-first（最小 diff ~40 行）
[RESULT-#R7-1] 优化点: w1 dtype 瘦身 fp32→dtype（r16(qk) 执行体迁移至 fixpipe；form_b_mma 专属 scope）| 精度: **pass（门禁全过）**——①探针 probe_r71 18/18：PART 0 分派审计 12 目标全落 B-mma∧w1_slim + 6 守卫全不落；PART 1 **12/12 对改前 BITEXACT**（含 bf16 fixpipe cast、大幅值域 |score| 饱和邻域 MARE 3.91e-3 vs gate 9.8e-3、zeros 陷阱对照 0.00e+00、S/S_kv 尾块、单块批处理 batch_tasks{1,3,4}、cross、显式 scale）+ golden 阻塞口径全过；PART 2 守卫 6/6 BITEXACT；②锚-A **#1-9 9/9 BITEXACT 硬守卫**（p6 电池）+ anchor_stress 全十用例 10/10（含 #10 诊断位仍 BITEXACT）；③锚-B **14/14 BITEXACT**（p6 电池，ulp 归因预案未触发）；④l0 10/10 + l1 19/19 + 覆盖 19P/0M/0N + `--level all` **exit 0**（L2 12/12 正确拒绝 + Boundary 6/6 + 哨兵 4/4 PERF_PASS）；⑤竞态压力：继承 4 配置 nondet=0 + rev4 扩展 16 项（专打 Form B-mma token 流 SIG_L0KC/L0AO/L0AS：单块跨轮/异构 batch_tasks/causal/bf16 + generic causal 裁剪异构 eff_kv/多外层批 8+1、8+8/S_kv 尾块/kv=16 深 token 流）全 PASS；⑥IR 守卫 **13/13 逐字节相同**（Form A ×8 + decode ×5 = decode 硬约束 IR 级成立）+ 3/3 目标按预期改变（case9 22617→22409B / case4 21113→20905B / B_single_d256 15255→15053B）；⑦机制静态可证：生成码 w1 FIX 发射 `copy_l0c_to_gm<float, half, RowMajor, 32, 128, 0>`（w3 保持 `<float,float>`）——dtype 类变更以模板参数为激活证明（强于 camodel，camodel 免采并在日志注明理由）；disable_cache 全程（TILELANG_CACHE_DIR=/tmp/opencode/r7_c1 会话专属目录）。
| 性能（msprof median us，同会话基线 msprof_r7_base 对比，msprof_r7_iter1/）: **目标 case9 944.61→947.50 (+0.31%) / case4 35.32→35.73 (+1.16%)——噪声带内零响应且方向微劣**；守卫 10 case 漂移带 -2.5%~+2.2%（case1 -2.54/case2 +2.22/case20 -2.00/其余 ±0.6%，IR 逐字节未变 ⇒ 该带幅=本会话测量方差标尺，目标读数落于带内）。
| op 归因 A/B（/tmp/opencode/r7_op_case9 vs r7_op_case9_w1slim，判决性证据）: **机制激活但 wall 不动**——AIV vec 47.2→44.4%、scalar_vector_stall 52.1→32.7us(-37%)、AIC fixp 32.8→31.6%、wait_id2 323.9→297.8us(-26us)（字节与指令确实减半/削减）；但 aic_time 908.5→885.0(-2.6%) 未传导至 kernel wall（msprof +0.31%），且 scalar_mte1_stall 412.2→445.0us(+33us) 重分配。**判读：case9 在 S1 后进入与 case7 同类的耦合平衡态——两侧均有等待余量（C wait_id2 298us / V wait_id4 418us），单边字节/指令削减被对侧吸收**（会话 3 case7 三证伪 + #R4-2b + #R6-2 的同款签名在 D256 域复现）。
| 采纳: **NO — 回滚**（rollback_reason: 性能无变化规则；回滚至 perf2 = ee5bf9c9，md5 核验 + IR 级还原 16/16 逐字节 diff ir_new2 无差异）
| **归档结论（§10.3-① 状态更新）**：w1 瘦身族现有**两个结构régime的零响应实测**（会话 3：gemm_v0 时代全域 7 shape；本轮：mma 化后 form_b_mma 域）⇒ 状态升级为**证据性放弃（双régime）**——未来重试前置条件 = 真机 op 归因显示某 bench case 的 pacing 管为 fixp/FIX 字节（当前无一命中：case6 28.8%/case7 30.8%/case9 31.6%）。RNE 位级等价知识三次实证入库（11/11+19/19+12/12），任何未来 fixpipe-cast 类改动免等价探针。**case9 归因修正（本轮新知识）**：S1 后 case9 的 wall 对"V-P1 阶段时长削减 ~5%"零响应 ⇒ pacing 不在 V-P1 单阶段，而在四相位链的循环周期本身（批深 4.5 槽 + 每 4.5 iter 任务边界的交接摊薄不足）⇒ 剩余规则内杠杆必须是**结构链缩短类**（批加深/相位删除），而非字节/指令削减类——Iter-2 的 k/v L1 双槽属字节时序类，EV 据此下修，但以实测裁决（#R5-1 预测性否决教训）。

### 迭代 r7-2：[#R7-2] Form B-mma 域 k/v L1 双槽（§10.3-③ 重试窗口）（2026-09-22/23）

[ORDER-CHECK] 准备实施: [#R7-2] k/v L1 双槽（form_b_mma 专属）| 前置依赖: [重试窗口授权条件 = "先 msprof op 归因确认" ✅（S7-Step0：mte2 49.5% 真机最忙管 + scalar_mte1_stall 412us 最大 C 停顿成分——M 管排空阴影已消除，与 #R4-2b 时代结构不同）；机制探针 ✅] | 结论: ✅
[IMPL-#R7-2] 已阅读: [IMPL-#R3-2]（3D [2,bN,D] L1 前导槽位维机制）、[IMPL-#R4-2b]（零新增 flag 思想——本项因 Form B-mma 为手动 flag 域，改用 #R6-2 令牌形态）、mha.py form_b_mma 两分支 K/V 段全部 14 个触点（grep 清点）、§7.2-2a 收支表（扩展方法）。**机制探针先行（新组合门）**：probe_r72_kvdb_mech.py——3D 双槽 × ZN annotation × 列段 transpose 读（S2 只验证过 2D 列段 ZN；#R3-2 只验证过 3D 全块读；组合未验证）：v3d vs v2d 参照内核 **BITEXACT**（两侧对 fp32 参考 worst|d|=6.2e-6；令牌协议逐字复刻 mha.py:1192-1209 任务级 L0AS 预载形）⇒ 实施解锁。关键约束: ①**零条件 DMA**——每槽装载自己的字节，仅槽位/令牌 id 轮转（side=i%2）；#R5-3 禁令域零触碰；②SIG 重排 form_b_mma 专属（K:0/1 V:2/3 P:4 Q:5，#R6-2 先例；ids≤8 硬件上限内）；③init 13→15 set / destroy 12→14 wait（K/V 每槽 1 writable 预置；收支恒等式：每槽 wait+set 两方向各 net 0，与 batch_iters/eff 裁剪奇偶、跨轮跨任务连续性无关——令牌流跨任务/轮不重置，同 Form A generic K 流先例）；④L1 预算 P3-mma：q16+k128+v128+p8=280KB ≤ 511.75 ✅；⑤Q/P 保持单槽（Q 任务级/generic+每槽/单块均字节小且紧随消费；P 被 SEM_WS2_V2C 生产者握手门控不可预取——#R6-2 同款论证）；⑥emit_v/V-scope/SEM 位置/Form A/decode 兜底零触碰。
[SELF-CHECK] 本次 Edit 只涉及 [#R7-2]：SIG 条件重排（+18 行注释）、form_b_mma k/v alloc [2,bN,D]（+5 行注释）、init/destroy 各 +2 项、单块分支 GEMM1 K 段/GEMM2 V 段 + generic 分支 GEMM1 K 段/GEMM2 V 段的槽位化（14 触点：载入 wait/set+ready wait+列段读 ×2+归还，全部 +side 参数化）——共 +31 行（1593→1624）。**过程缺陷如实记录：批量编辑引入 4 处缩进错位（单块区 2 行 36→32、generic 区 2 行 32→36），ast.parse 捕获后程序化修复，修复后全文缩进审计（form_b_mma 域 952-1398 行逐行 modulo-4 + 区域基准核验）通过**；ruff clean。备份 = mha_impl_r4_perf2.py（改前 ee5bf9c9）。
[KERNEL-REUSE-AUDIT] 依据: §10.3-③ + S7-Step0 归因 | 维度及结论: 调用域不扩 ✅；令牌收支静态可验（约束③，生成码核验 init 15 set：SEM1+L1×6+M_MTE1×6+FIX_M×2 / destroy 14 wait / 循环内 K(i%2)、V((i_1%2)+2)、P(4)、Q(5) 闭合）✅；机制探针 BITEXACT ✅；unknown: 真机收益 → msprof+op A/B 裁决 | 结论: repair-kernel-first
[RESULT-#R7-2] 优化点: Form B-mma k/v L1 双槽（MTE2 提前一槽发射）| 精度: **pass（门禁全过）**——①探针 probe_r72 18/18：目标域 12/12 对改前 **BITEXACT**（纯调度变更位级零漂移：含 bf16/大幅值域/zeros 对照/双尾块/单块批 batch_tasks{1,3,4}/多块 causal 裁剪/cross/scale）+ golden 阻塞口径全过 + 守卫 6/6 BITEXACT；②锚-A **#1-9 9/9 BITEXACT 硬守卫** + 全十用例 10/10（#10 诊断位仍 BITEXACT）；③锚-B **14/14 BITEXACT**；④l0 10/10 + l1 19/19 + 覆盖 19P/0M + `--level all` exit 0（L2 12/12 拒绝 + Boundary 6/6 + 哨兵 4/4 PERF_PASS）；⑤竞态压力：继承 4 配置 + rev4 扩展 16 项（专打 Form B-mma token 流：单块跨轮/异构 batch_tasks/causal 裁剪异构 eff_kv/多外层批 8+1、8+8/S_kv 尾块 700%128/bf16/kv=16 深 token 流）**nondet=0 全 PASS**；⑥IR 守卫 **13/13 逐字节相同 + 3/3 目标按预期改变**（case9 22617→23040B / case4 21113→21450B / B_single_d256 15255→15482B）；⑦机制激活生成码静态可证（init 15 set / K,V ±(i%2) 参数化 id / 3D 槽偏移）。
| 性能（msprof median us，同会话基线对比）: **目标 case9 944.61→947.96 (+0.35%) / case4 35.32→36.23 (+2.58%)——噪声带内零响应且方向微劣**；守卫 10 case 漂移带 -3.0%~+2.5%（case1 -3.02/case2 +2.54/case20 -2.14，IR 逐字节未变 ⇒ 带幅=本会话方差标尺）。
| **op 归因 A/B（/tmp/opencode/r7_op_case9 vs r7_op_case9_kvdb，判决性证据——本会话最重要机制知识）**：
  · **机制激活且效果超预期**：scalar_mte1_stall **412.2→188.0us（-54%）**（直接靶心：chunk MTE1 链停顿减半）、mte1_ratio 33.7→39.4%（MTE1 更密）、scalar_ratio 83.5→77.7%、V wait_id0（等 w1）210.0→123.9us（-41%，w1 更早交付）。
  · **但 wall 零移动**：C wait_id2（等 w2）**323.9→545.0us（+68%）**、wait_id5 79.5→133.2、V wait_id4 423.6→469.3——**C-G1 的加速 1:1 转化为 C 在 w2 交接点的等待增量**。
  · **与 r7-1 合并判读（双侧 A/B 完备）**：r7-1 加速 V-P1（vec -2.8pp/vector_stall -37%）→ C mte1_stall +33us 吸收，wall +0.31%；r7-2 加速 C-G1（mte1_stall -54%）→ C wait_id2 +221us 吸收，wall +0.35%。**两侧各自拥有 ≥200us 级等待余量 ⇒ case9 wall 既非 C 装载链约束、亦非 V-P1 吞吐约束，而是四相位每槽交接延迟链（每槽 4 次跨核 SEM 传播 + 管道排空的不可重叠段 × 461 iter，批深仅 4.5 槽摊薄不足）**。字节/时序/单侧速度类手段在该结构下数学上无杠杆——两次独立 A/B 为"耦合平衡吸收律"提供了迄今最强的定量证据（吸收量 = 节省量，1:1）。
| 采纳: **NO — 回滚**（rollback_reason: 性能无变化规则；回滚至 perf2 = ee5bf9c9，md5 核验 + IR 级还原 16/16 逐字节 diff ir_new2 无差异）
| **归档状态更新**：§10.3-③ k/v L1 双缓冲——四路径证伪链追加第五环（Form A generic #R3-2 / Form B D256 gemm_v0 #R4-2b / decode 带宽+预算否决 / Form A 单块 #R6-2 / **Form B-mma mma 化后本项：机制激活+mte1_stall -54% 仍零响应**）⇒ **该族证据性关闭（全路径形态穷尽）**；机制知识入库：3D×ZN×列段 transpose 组合 BITEXACT（探针在案，未来复用免机制门）。

## S7-Step 5 归因综合与 Iter-3 处置（[ANALYSIS]，无 kernel 变更）

**case9 剩余 wall 的规则内唯一杠杆指认（双侧 A/B 排除法完备）**：字节削减类（w1 瘦身）与速度类（k/v DB）分别在 V 侧与 C 侧被 1:1 吸收 ⇒ 剩余杠杆必须攻击**交接结构本身**：
1. **flat (task,kv) 批处理（Form B-mma generic 域）**——批深 4.5→8（case9 causal eff_kv 均值 4.5，NUM_STAGES=8 半空转）+ 任务边界 102→~58/核（每边界含 V 侧 task-tail 归一化+写回串行段位于 P1(t+1) 之前的关键路径上 + C 侧 Q/l0a_s 预载 + 轮界批级 SEM）。op 数据边界成本上界：C wait_id1+id5 = 87us（~10% wall，现态下部分已被掩蔽）+ V task-tail 关键路径段（camodel 可测但本会话未单测——如实标注为估算项）。**EV 估算 -5~-15%（case9），为剩余规则内候选中唯一有实质量级上界者**。复杂度：C 侧槽→(task,kv) 平坦映射 + l0a_s 每槽任务相关重载（单块批分支已有先例形）+ V 侧 emit_v 轮中任务切换状态机（§S4-3-5 归档约束：复位在 phase-2 kv==0 槽、m_i 例外）——**emit_v 为三形态共享，任何改动爆炸半径 = 全域（失去 13 守卫 IR）⇒ 需独立会话全额预算 + 完整门禁链**。本会话预算不足以安全完成（实施+调试+全门禁 ≈3h，剩余预算 <1.5h），**不启动**（半实现 = 死锁/精度风险，纪律禁止）。移交建议 = 会话 8 首位候选，实施蓝图与风险清单已具备（#R4-3 机制 17/17 BITEXACT 归档 + §S4-3-5 时序约束 + 本轮双侧 A/B 的 pacing 结构判定）。
2. CROSS_INTERVAL=2：**不重提**——属授权禁令"barrier/wait 削减类"（#R3-4 判决性证伪 + S5-Step3"削减 wait 指令数不缩减 wall"）；本轮归因虽显示交接延迟结构，但 CI=2 减少交接次数的同时加倍每次交接的批粒度延迟（V-P1 起始被 2 槽门控），方向性收益不成立，无判决性新证据推翻禁令前提。
3. w3 colseg→宽池单 FIX（plan-a 重试）：camodel 显示 FIX 事件 +50% 税，但真机 fixp 32.8% 非 pacing + 双侧 A/B 已证单侧削减被吸收 ⇒ EV≈0，且 plan-a 有探针崩溃史（根因未知）——不启动，维持 PLAN B。
**中止条款状态**：连续 2 次无提升（r7-1/r7-2），未达 3 次；第 3 候选经评估全部为"禁提类/EV≈0/需独立会话"⇒ 按"不强制耗尽预算"纪律收口，剩余预算用于收口验证与归档质量。

### 会话 7 收口补录：默认缓存污染事件（如实归档；检出/根因/影响面/处置全链见 debug_log.md 会话 7 补录）

收口终态新鲜锚复跑（默认缓存）出现 #10 DIFF 1.305e-01（确定性 ×2）→ 隔离缓存 10/10 → 三方判别（rev4 坏/r3+dev 好）→ 根因 = 默认缓存 Stage-2 时代毒条目（kernel.cu md5 ac0887… = k_l1 NZ 修复前源码）与当前源**同 cache key 冲突**（rev4 已归档"缓存键不覆盖 annotation 级变更"缺口的实际发生学证据）→ 清除 2+2 条目 → **默认缓存复跑锚 10/10 BITEXACT + 压力 PASS（r7_anchor_final5.log）+ case4 golden 复核通过**。影响面逐项排除：bench case9 条目 md5 与当前生成码逐字节一致（Step 0 归因有效）；case4 基线为当场新编译（无缓存条目）；毒 key 族 = 单块 d256 家族，20 bench case 零命中；本会话 iter1/2 测量全在隔离缓存目录；权威复评管线独立。**终态精度声明以 r7_anchor_final5（10/10）+ 两轮电池（隔离缓存，全过）为准。**

---

# 会话 8（perf_session=8，2026-09-29，rev4 轨第二轮；大 prefill ①+②+③ 联合流水专项；软预算 6 迭代，备份链 r4_perf{N} N≥3）

> 对象：rev4 终态 mha.py（1593 行，md5 ee5bf9c93f979b9fc5b111a3c4121888，cann-bench 71.79 / 0.9108x）。
> 触发：用户 2026-09-29 经 Primary 问询决策立项"方案 A 联合专项"——case7/8/9 的 now→B 层差距（占 case9 差距 63%）
> = 四相位每槽交接延迟链；规则内唯一杠杆包 = ①flat(task,kv) 连续令牌流 + ②C 侧 L0B 4 槽旋转 + ③V 侧批相干化；
> 吸收律（r7-1/r7-2 双侧 A/B 1:1 实证）决定必须联合落地，单阶段判据 = 机制激活 + 位级/精度安全 + 代价净值（非 wall）。
> 分工纪律（§S7-6 继承）：不动提交副本、不跑 cann-bench（orchestrator 轮末统一执行）；禁读写 .orchestrator_state.json。

## S8-Step 0 基线归位与证据加载（先于一切）

- **基线归位**：工作区 mha.py md5=d1f4dd58（L782 注释行尾随空格差异，语义/IR 零差异，编辑器意外触碰）→
  `cp history_version/mha_impl_r4_perf1.py mha.py` → md5 **ee5bf9c93f979b9fc5b111a3c4121888** ✅（与权威一致）；
  开工快照 `history_version/mha_impl_r4_perf3.py`（同体 ee5bf9c9）。失败/回滚变体留存 /tmp/opencode/r8_probes/。
- **证据基座精读完成**（禁止凭记忆动手纪律）：analysis_20260928（§2.1 现状解剖/§2.2 五手段/§2.3 量化差距/附录 .2/.3/.4/.6）、
  final_report §S4-3/§S4-5/§S7-3/§S7-5、optimization_log #R4-3（flat 17/17 BITEXACT 蓝图）/#R3-1/#R5-1（SIG_Q_L1 收支推演范式）/
  #R6-1/#R7-1/#R7-2（吸收律原始数据）、msprof_case9_20260923（op 权威）、msprof_r7_base、mha.py 全文 1593 行、
  DESIGN rev4 §4.5（L0 预算表 + (d) 共池复核 + (e) decode 不可行推导）/§7.2（flag 矩阵 + 2a 收支表 + 5 C/V 同式 + 6 条件 DMA 禁令）/
  §8.3（GEMM1/GEMM2_BODY 内联清单）/§8.6-9/10（K/N-split 链完整性 + 共池相位纪律）/§10.3；
  最高同构模板 sparse_flash_mla（v_l0b [2,BI,BI] 方形共池 + real_k/real_n/k_actual 部分 tile 机制 + QK^T transpose/PV direct 双取向分时）。
- **同会话基线采集**（baseline_r8.json 已落地）：
  - l0 10/10 PASS exit 0（/tmp/opencode/r8_probes/r8_l0_base.log，TILELANG_CACHE_DIR=r8_cache_base）；
  - msprof wall（median us，warmup5+iters20，msprof_r8_base/）：case9 **944.74** / case4 **35.57** / case7 **924.69** /
    case10 34.86 / case14 377.69 / case16 562.96 —— vs 会话 7 记录最大差 +0.7% ⇒ 环境一致，采纳判定用本会话内对比；
  - op 归因基线（/tmp/opencode/r8_op_case9_base/，case9 block0）：**mte1_stall 433.96us > mte1_busy 321.38us（纯延迟暴露签名 ✅）**、
    mte2 50.5% / fixp 34.5% / cube 19.4% / scalar 84.7% / wait_id2 340.7us / wait_id5 87.6us；
    AIV：vec 47.8% / wait_id0 212.9us / wait_id4 431.5us / vector_stall 59.9us —— 与归档参考逐项吻合（stall 436.2@0923 / 412.2@r7）。
- **反模式对照**（agent 清单项）：performance-antipatterns.md 全 11 项状态继承 #R5-7 复扫 + 会话 7 判定——唯一残留结构命中 =
  "AIC/AIV 混合未开启 CV overlap"在 generic 多块域（四相位每槽交接链 = 本专项 ①②③ 的攻击对象，**有意保留至 Stage 4 联合集成**，
  保留原因 = 吸收律下单侧/单项手段数学上无杠杆，r7-1/r7-2 双侧 A/B 1:1 实证）；"冗余全局同步" Form B 域已纯化（#R4-1）；
  其余项（launch 核数/逐元素 for/mul+add 融合/tile 过小/多行归约/正交轴/AUTO_CV 误分核/io_buf 双槽）状态不变。

## S8-Step 2 算子类型（继承 + 本会话复核）

- CV 融合混合型（KERNEL_TYPE_MIX_AIC_1_2，Expert cv_1_2 双 scope）三分派：form_a（S>16∧D≤128）/ form_b_mma（S>16∧D=256，
  Stage 1-3 主战场）/ form_b_gemm_v0（S≤16 decode 兜底，**源码零触碰 + IR 逐字节守卫**）。
- Python 层机制核查（不凭记忆猜 API）：`T.mma`=`npu_gemm`（customize.py:115-228——形状断言仅 K_A==B_shape[-2]；C++ 模板
  `mma<T1,T2,M,N>` 的 M/N 取自 **C 形状**、K 为运行期实参、B 为裸指针）；`T.view/reshape`（customize.py:84-112，
  `T.Buffer(shape,dtype,src.data)` 共享 data Var；**TVM 层 scope() 由 data Var 的 storage_scope 派生（buffer.cc:435-441）⇒
  view 保持 wmma.matrix_b scope**；仓内 kernel 零使用先例 ⇒ 必须机制探针先行）；生成码核验（smoke_case9_r72.cc）：
  L0B 槽引用 = `l0b_pool[<元素偏移>]`、`copy_l1_to_l0b<dt,srcM,srcN,transpose>(dst,src,dstM,dstN)` 的 dstM/dstN 为运行期实参
  （C++ 侧按 (dstM,dstN) 构建 LayoutL0B 分形排布）⇒ [4,64,128] 与 view [4,128,64] 每槽元素偏移同为 n×8192，物理自洽。
- **② 形态学关键判定（DESIGN §4.5-d/e 同款约束的 4 槽版）**：16KB 槽下 GEMM1 B-tile（K^T chunk [KC'=64,BN=128]，transpose
  colseg 载入）与 GEMM2 B-tile（V chunk [BN=128,NC'=64]，直载 colseg）**互为转置**——单一同形声明共池无法同时直容两者；
  候选形态：v4lit（pool [4,64,128] + T.view [4,128,64] 双取向 = 附.3 字面形态，FIX 3→5/iter）/ v4ksp（同形 [4,64,128]，
  GEMM2 改 P K-split×2 × N-split×2 链式，B = V 行列双切片 [64,128] 直载，FIX 保持 3/iter）。
- **MTE1↔M event id 预算（硬约束 ≤8/有向对，#R7-2/#R6-2 归档）**：现 6（L0KC 0,1 + L0AO 2,3 + L0AS 4,5）；4 槽方案 =
  L0KC 0-3 + L0AO 4,5 + L0AS 6,7（Q 4 chunk 按 {0,1}/{2,3} 组粒度 2 id——组内 copy 背靠背、wait 前置消费上一任务/槽归还，
  与现 2-chunk 结构同构）= **8 恰好顶格**；M↔FIX：l0c_s {0,1} + l0c_o {2,3,4,5} = 6 ≤8（v4lit）或 {0,1} 共流（v4ksp）。

## S8-Step 3 Part A8 优化点清单（联合专项 = orchestrator 授权执行序；勿重试清单全项继承）

[KERNEL-REUSE-PRECHECK]
candidate: #R8-1 Stage-1 ② C 侧 L0B 4 槽旋转（analysis §2.2-②/附.3；cube_optimization_path §2.2 L0 分块多缓冲；flash_attn_optimize §3 L0 分块与 MMA intrinsic；模板 = sparse_flash_mla v_l0b 方形共池 + real_k/k_actual）
expanded_domain: 无新增 host 调用域——form_b_mma 谓词/buffer 总量/预算/L1 集不变；仅 C 侧 GEMM1/GEMM2 chunk 粒度 128→64（MMAD 4→8/iter）+ 池槽 2→4 + id 矩阵重排（L0KC 0-3/L0AO 4,5/L0AS 6,7 组粒度/M↔FIX 至多 0-5）；decode 兜底域与 Form A 零触碰（IR 逐字节守卫可证域）
compatibility_dimensions: ①位级=TRUE by 双原则（K-ascending 链任意粒度同序 #R4-2a 12/12 + N-split 列独立 P2b/P5 16/16；v4ksp 的 PV K-split 链同属 #R4-2a 原则域）——探针裁决；②共池取向=**UNKNOWN**（v4lit T.view 双取向无仓内先例；v4ksp 的 V 行列双切片 MTE1→L0B 为新组合）→ 机制探针 probe_r81_l0b4_mech 三变体先行；③L0B 64KB 顶格=待实编 P4 式读数复核（硬门，优于 naive 手算）；④id 预算=TRUE（8 顶格推演 S8-Step2，收支表实施时静态可验）；⑤令牌收支=TRUE by 构造（每 chunk wait/set 两方向各 net 0，与 chunk 数无关——§7.2-2a 表扩展法）；⑥K/N-split 链完整性（§8.6-9）=init=True 仅 c==0（GEMM1）/每 n-chunk 首 k（v4ksp）或全部 init=True（v4lit GEMM2 独立列段）；⑦共池相位纪律（§8.6-10）=批结构不变（GEMM1 批全先于 GEMM2 批，无 iter 级交错）；⑧条件 DMA 禁令（§7.2-6）=全 DMA 无条件发射（chunk 数编译期常量 4，Q 重载策略不变）
unknown_or_false: ②（两种候选形态各含一个新机制组合）→ probe-first
plan: repair-kernel-first（机制探针 → 全 kernel 探针矩阵 → 门禁链）

[#R8-1] **Stage-1 ② L0B 4×16KB 槽旋转（form_b_mma 域 = case9/case4）**：适用（授权 Stage 1）。攻击点 = mte1_stall 434us(46%wall) > busy 321us 纯延迟暴露；机制 = 槽半宽 16KB（MTE1 84ns/块 vs MMAD ~45-52ns/块）+ 4 槽信用使装载前瞻深度 0→2、MMAD 读毕即释槽。判据（三项，非 wall——吸收律）：(a) 机制激活 = 同会话 op A/B mte1_stall 显著降且 stall≤busy；(b) 位级安全 = BITEXACT 预期（探针覆盖奇偶 chunk 链/causal eff 1..max/bf16/尾列/单块退化）；(c) 代价净值 = MMAD 4→8/iter + MTE1 事件 7→13/iter +（v4lit）FIX 3→5/iter（事件税先例 884→1326/case7 fixp 251→298us）；camodel 仅机制定性（§S6-3-3 三度实证禁作定量）。净值裁决为负 → 归档 + 回滚 + 评估 ①+③ 二元联合残余可行性并如实报告。
[#R8-2] **Stage-2 ③ V 侧批相干化（emit_v 宽通道重构）**：适用（授权 Stage 2；前置 = Stage 1 收口）。机制 = 批内 W 槽 w1 合并为宽 2D op 序列（[W×hM, bN] 级），消除 36×BAR+34×MOVEMASK/iter 与 46904 碎片事件税（均值 14ns/事件；case9 V.vec 1.058us/iter/组 vs 元素量理想 ~90us/468us）。case7/8/3 主杠杆依据 = case7 真机 V vec 68.3% + vector_stall 258us（§S7-3-5）。**精度门（三项最重）**：宽合并改变 online-softmax 归约结构 → 非位级预期；阻塞口径 = 混合容差 + MARE_abs<10×Threshold 主门 + 余量复核（现余量 prefill_worst 3.05e-5/320x）；锚-A #1-9 BITEXACT 硬守卫 by design 失效 → 锚体系重建/降级方案显式写入迭代记录（§S4-5 位级预案先例）。红线四项不可破：双舍入栅格（r16 落点）/单块预归一特化/sumexp+1e-30/右下角掩码链。**UB 预算硬门**：宽通道 scratch 三 D 配置实编 InitBuffer ≤191.75KB——预分析（本会话）：广播/掩码/往返 scratch 必须物化（`subs` 标量路径 = PipeBarrier<PIPE_ALL>+全 op 单标量，codegen_ascend.cc:1837-1888 核验，不可免物化）⇒ W 受限于 D 配置（D=256: W=2 约 +33KB→155KB ✓ / W=4 +97KB→219KB ✗；D=128: W=2 +65KB→~188KB 贴边 ⚠；D=64 现 ~135KB: W=2 ✗ 需 W=1 保旧路径）——实编读数裁决，W 逐配置自适应（parse-time 常量），D=64 域保 W=1（宽通道分派谓词收窄，IR 影响面如实报告）。
[#R8-3] **Stage-3 ① flat(task,kv) 连续令牌流 prefill 扩展**：适用（授权 Stage 3；前置 = Stage 1/2 收口）。蓝图直接继承 #R4-3 decode 域全记录（17/17 BITEXACT：跨轮任务状态携带/轮中任务切换/phase2 复位时序）+ §S4-3-5 约束（V 任务状态复位必须放 phase2 的 kv==0 槽；m_i 例外 = phase1 独占）+ §S7-5 scoping（EV -5~-15%，上界 = C 轮界等待 87us + V task-tail 段；任务边界 102→~58/核）。**prefill 域新工程点**：causal eff_kv_iters 逐 task 运行期非均匀（case9 均值 4.5）→ f→(task,kv) 映射需运行期派生（前缀和/分段队列式），**C/V 双侧同式**（同一 my_start/my_count 派生，不经 ws 传递，§7.2-5）；信用制 = 槽释放即发，NUM_STAGES=8 ≥ 2×链深（链深 4 槽）；令牌收支静态可验 + **同进同出无条件**（#R5-3 教训）。域序：form_b_mma generic（case9）→ form_a 多块（case7/8/3）；single_kv_block 特化与 decode 兜底域分派谓词收窄零触碰。位级预期 BITEXACT（每槽算子序列同序，#R4-3/#R5-1 先例）；探针全拓扑（轮中任务切换/eff 运行期变异/奇偶批/causal trim/尾列/bf16/zeros/混合批序列）。
[#R8-4] **Stage-4 联合集成 + 全门禁**：三项集成后才判 wall（预期收敛 case9 945→626us 级 0.54x、case7/8→0.54x 上界 0.70x——全部模型推算，msprof 实测唯一裁决）。全门禁 = 探针矩阵 / l0 10/10 / l1 19/19 / coverage 19P0M / `--level all` EXIT=0（四哨兵）/ 竞态压力专打被改路径（rev4 扩展 16 项 + flat 协议新增专项，≥12 轮 nondet=0）/ 锚体系（按 Stage-2 重建方案）/ 兜底域源码 diff 零触碰核验 / decode 五 case msprof 守卫；emit_v 变更后 IR 守卫失效 → **全 20 case msprof 复测（median）**替代 IR 保证。
[#R8-5] **勿重试清单（继承 + 本会话确认，全部有归档证据链）**：w1 dtype 瘦身（双 régime 零响应）/ k-v L1 双槽（五环证伪链闭合）/ CROSS_INTERVAL 扩大（判决性证伪）/ 任何单侧加速、字节类、速度类手段（吸收律 1:1 定量实证）/ kv-outer 循环互换（四步定量降级）/ O 直写扩展 form_b_mma（bench 收益面=0）/ decode 域协议轮削减 flat（SEM 税已坍缩 0.2-0.3us/轮）；④w3 瘦身本轮不实施（激活前置 = ①-③ 后 msprof 实证 FIX/L2 成绑定管道；当前无一 case 以 fixp 为 pacing：case6 28.8%/case7 30.8%/case9 31.6-33.4%）。
[#R8-6] **硬约束继承**：decode 不回退（case10≥0.90/13≥0.69/14≥0.99/16≥0.52/20≥0.82 bench 口径）+ 兜底域 C 侧源码零触碰；T.Pipelined 全文禁用（R9）/threads= 禁用（R8）；camodel 仅机制定性；探针非零数据（case20 假阴性陷阱）；全程 disable_cache 或专属 TILELANG_CACHE_DIR（缓存键不覆盖 annotation 级源变更——会话 7 污染事件教训；本会话逐变体独立缓存目录）。

## S8-Part B8 [ORDER-PLAN]

[ORDER-PLAN] 实施顺序（= 任务书阶段化门禁，每阶段独立止损）：
1. [#R8-1] Stage-1 ② L0B 4 槽（form_b_mma 先行）— 前置依赖: 机制探针 probe_r81（v4lit T.view 双取向 / v4ksp 行列双切片 两形态裁决）+ P4 式实编预算读数（硬门）— 理由: 授权序首位；C 侧独立域（emit_v 零触碰 → 13 守卫 IR 逐字节可证，爆炸半径最小）；判据三项非 wall（吸收律）。布局依赖：池槽形态定 id 矩阵；数量依赖：L0A/L0C 形状随 chunk 粒度变（预算复核先行）。
2. [#R8-2] Stage-2 ③ V 批相干化 — 前置依赖: Stage-1 收口（采纳或归档回滚）+ UB 实编预算读数（三 D 配置，宽 scratch 物化不可避免已核）+ 锚体系重建方案显式成文 — 理由: emit_v 共享宏 = 全域爆炸半径，必须在 C 侧形态定型后单独归因；精度门最重（非位级预期）。配置依赖：W 逐 D 配置 parse-time 派生。
3. [#R8-3] Stage-3 ① flat prefill — 前置依赖: Stage-2 收口（V 侧宽通道结构定型——flat 的 V 状态机建立在定型后的 emit_v 上，避免交叉重构/交叉回滚）— 理由: 蓝图最完备（#R4-3 17/17）但 prefill 域 eff 非均匀映射为新工程点；域序 form_b_mma → form_a。
4. [#R8-4] Stage-4 联合集成 + 全门禁 — 前置依赖: 1/2/3 全部收口 — 理由: 吸收律决定 wall 收益只在联合态可判；全 20 case msprof 复测替代 IR 保证。
5. [#R8-5] Stage-5 收敛 — 性能规则（>3% 采纳/无变化回滚/中止条款②连续 3 次）+ 软预算 6 迭代纪律 + 联合包不完整或净负 → 中止条款收口回滚（会话 7 先例：零净变更收口也是合格产出）。
（每轮固定流程：备份 history_version/mha_impl_r4_perf{N}.py（N 从 4 续编）→ [ORDER-CHECK]/[IMPL]/[KERNEL-REUSE-AUDIT]/[SELF-CHECK] → Edit 单点 → 机制/位级探针 → 全 kernel 探针矩阵 + l0 + 双锚 + 压力（专打被改路径）→ l1/coverage/all → msprof wall + op A/B（同会话基线对比）→ IR 守卫（可证域）→ [RESULT] 采纳/回滚；camodel 仅机制定性。）

## S8-Step 4 逐项实施记录（会话 8）

### 迭代 r8-1：[#R8-1] Stage-1 ② C 侧 L0B 4×16KB 槽旋转（form_b_mma 域）（2026-09-29）

[ORDER-CHECK] 准备实施: [#R8-1] L0B 4 槽旋转 | 前置依赖: [机制探针 ✅ probe_r81_l0b4_mech PASS（v4ksp 编译+预算+4 配置 BITEXACT vs 生产参照）；P4 式实编预算读数 ✅（新态 case9 生成码：L0B 64.00KB 顶格单池 InitBuffer 65536 / L0A 32 / L0C 64 / L1 152 / UB 126KB 全 OK）] | 结论: ✅
[IMPL-#R8-1] 已阅读: analysis §2.2-②/附.3（目标形态与代价清单）、DESIGN §4.5-c/d/e（L0 预算表 + 共池复核 + **异形 view 共池无仓内先例判定**）、§7.2-2a（收支表扩展方法）、§8.3（GEMM1/GEMM2_BODY 内联清单 + 备选 b 窄池 colseg）、§8.6-9/10（K/N-split 链完整性 + 共池相位纪律）；sparse_flash_mla:216-218/360-488/560-660（v_l0b [2,BI,BI] 方形共池 + real_k/real_n/k_actual 部分 tile + QK^T transpose/PV direct 双取向分时 + PV K-split 链 init=is_first_ks 先例）；customize.py:115-228（npu_gemm：形状断言仅 K_A==B_shape[-2]，C++ 模板 M/N 取自 C 形状、K 运行期实参、B 裸指针）；common.h:128-190（copy_l1_to_l0b 的 dstM/dstN 运行期实参构建 LayoutL0B + mma MmadParams）；codegen_ascend.cc:2856-2864（SetFlag/WaitFlag<HardEvent>(id) 发射）；buffer.cc:435-441（Buffer::scope() 由 data Var storage_scope 派生）；probe_p1p3_ksplit/p2b_planb/p4_budget/r72_kvdb_mech（先例探针骨架）。
**关键约束与形态学裁决（先于实施）**:
① **附.3 字面形态（4×[128,64] = GEMM2 4×NC64 n-chunk）不可表达**——GEMM1 B-tile [KC',bN] 与 GEMM2 B-tile [bN,NC'] 在 16KB 槽下互为转置（[64,128] vs [128,64]），单一同形声明共池无法直容；双取向需 T.view，而 **T.view/T.reshape 在本 TVM 版本损坏**（customize.py:94/112 调用不可直接构造的 tvm.tir.Buffer——buffer.py "Do not construct directly"；探针实证 'Buffer() takes no arguments'）；decl_buffer 直构亦需 pointer type_annotation 且无仓内 kernel 先例。DESIGN §4.5-e 已有同款判定（"异形 view 共池无仓内先例，fractal 排布不同 → 不可行"）。
② **实施形态 = 同取向 4 槽（探针 v4ksp 胜出）**：pool [4, KC=64, bN=128]（每槽 16KB）；GEMM1 = K-split ×4（B = k_l1 colseg [128,64] transpose → [64,128]，链 init T,F,F,F → l0c_s [2,32,128] 全块，w1 FIX ×1 不变）；GEMM2 = **PV K-split（block_N 2×64）× N-split（D 2×128）**：A = p_l1 colseg [32,64]→l0a_o[side*2+kc]（4 槽 [4,32,64]），B = **v_l1 行列双切片 [64,128] 直载**（ZN row-slice 新组合，探针 BITEXACT；生成码偏移实证 Forward(64,0)=1024/Forward(0,128)=16384/Forward(64,128)=17408 逐项正确），链 init T→F per n-chunk → l0c_o [2,32,128] 不变，w3 colseg FIX ×2 不变。
③ **位级论证**（探针裁决 BITEXACT 20/20）：GEMM1 链 4×64 ≡ 2×128 ≡ 1×256（K-ascending C0 同序，#R4-2a 12/12 原则）；GEMM2 PV K-split 链 2×64 ≡ 单 K=128（同原则）；N-split 列独立（P2b/P5 16/16 原则）；l0a/l0b 槽形变化纯载体（数据流逐字节同）。
④ **LOAD-FIRST 突发排序**（机制核心，附.3 "连续流"的实现形）：每 GEMM 突发先发射全部 4 个 16KB MTE1 装载（各自槽释放 wait 前置），再进 mma 链——2 槽 JIT 序下 MTE1 在 mma 间隙空转（32KB 块 168ns 冷启动暴露）；4 槽 load-first 下 MTE1 背靠背 84ns/块 > mma ~52ns/块 → MTE1 成定速流、scalar 停顿转为定速等待。GEMM2 的 V 切片装载**前置到 w2 SEM 门控的 P 段之前**（V 与 w2 无关，MTE1 在 scalar 阻塞于 SEM 时即可流满 4 槽；SEM set/wait 相对序零漂移——生成码 CrossCore census md5 与基线全等实证）。
⑤ **id 矩阵（≤8/有向对硬限，#R6-2/#R7-2 归档）**：MTE1↔M = L0KC{0,1,2,3} + L0AO{4,5}（side 组令牌：每 side 的 2 个 P colseg 共一轮）+ L0AS{6}（Q 4 colseg 组粒度单 id——任务头/槽头背靠背装载非前瞻路径，组语义与现 2-id 结构等价推演）= **7/8**；M↔FIX = SIG_L0C{0,1} 分时同流不变（l0c_s[side] 与 l0c_o[n] 交替，收支结构逐字继承）；MTE2↔MTE1 = K/P/V/Q{0-3} 不变。init 13→**14** set（L0KC×4 + L0AS×1）/ destroy 12→**13** wait。收支静态可验：每 id 每突发 set/wait 两方向各 net 0（与 chunk 数无关），生成码字面 id census 实证（L0KC 0-3：M_MTE1 set 3=init+G1+G2 / wait 3=G1 门+G2 门+destroy；MTE1_M set 2=G1+G2 copy / wait 2=G1+G2 mma 门 ✓；L0AS id6：2/2/1/1 ✓）。
⑥ **事件代价（如实，判据 c）**：MMAD 4→**8**/iter（生成码 mma 计数 8 ✓，模板全 <32,128> K 运行期 64）；MTE1 copy 事件 7→**13**/iter（Q×4+K×4+P×2+V×4=14 减 l0a_o 旧整块 1 = 13）；**FIX 保持 3/iter**（w1 全块 ×1 + w3 colseg ×2——优于字面形态的 5，生成码 copy_l0c_to_gm ×3 实证）；SEM 协议零漂移（census md5 全等）。
⑦ 共池相位纪律（§8.6-10）保持：批结构不变（GEMM1 批全先于 GEMM2 批），token 流跨相位延续（GEMM2 首槽 wait 消费 GEMM1 末 mma 释放）；无条件 DMA（chunk 数编译期常量，零运行期跳过，#R5-3 域零触碰）；emit_v/V-scope/SEM/Form A/decode 兜底零触碰。
[KERNEL-REUSE-AUDIT] 依据: Part A8 [KERNEL-REUSE-PRECHECK] #R8-1 | 维度及结论: ①位级=TRUE（探针 20/20 BITEXACT 裁决，含机制探针 4 配置）✅；②共池取向=已裁决（同取向 v4ksp，字面形态 view 损坏证据归档）✅；③L0B 顶格=实编读数 64.00KB/64KB PASS（budget_r81.py，InitBuffer 65536 单池）✅；④id 预算=7/8 字面 census 实证 ✅；⑤令牌收支=静态推演+生成码 census ✅（R11 压力门实测收敛中）；⑥链完整性=init=True 仅 c0（GEMM1）/每 n-chunk 首 k（GEMM2），生成码 (bool)1/(bool)0 序列核验 ✅；⑦批结构不变 ✅；⑧无条件 DMA ✅ | 结论: repair-kernel-first（已实施，探针先行完成）
[SELF-CHECK] 本次 Edit 只涉及 [#R8-1]：①host 常量 KC 128→64 + 新增 KC2=64 + 断言扩展（chunk 数静态封闭 4/2/2）；②SIG_L0KC/L0AO/L0AS 重排 0-3/4,5/6 + 注释；③form_b_mma 池形状 l0a_s [4,32,64] / l0a_o [4,32,64] / l0b_pool [4,64,128]（l0c_s/l0c_o/L1/UB 集不变）；④init 14 set / destroy 13 wait；⑤单块批分支 GEMM1（Q 组令牌 4 colseg + K load-first ×4 + 链×4 + 槽末 L0AS 归还 ×1）/ GEMM2（V 切片 load-first 前置 + P colseg ×2 + 2n×2k 链 + colseg FIX ×2）；⑥generic 分支任务头 Q 预载（组令牌）/ GEMM1 / GEMM2 / 任务末归还 ×1——四处 GEMM 体全部同步变换；⑦Form A 两分支 / emit_v 全体 / decode 兜底 prim_func / host 包装 / SEM-SIG 矩阵（除 form_b_mma 段重排）/ allocations（除 L0 三池）零触碰——**IR 守卫 13/13 逐字节相同实证**（r8_src_iter1 vs r7_src_rollback2，含 decode ×5 + Form A ×8；基线参照经 case9 新鲜 dump md5 全等核验 provenance）；⑧AST/ruff check/format clean；⑨备份 = history_version/mha_impl_r4_perf4.py（改前 ee5bf9c9）。docstring 头部 rev4r8 注记留待收口统一（codegen-neutral 先例，届时 16/16 复核）。
[RESULT-#R8-1] 优化点: **Stage-1 ② C 侧 L0B 4×16KB 槽旋转（form_b_mma 域；同取向 v4ksp 形态——附.3 字面 [128,64] 双取向形态因 T.view 损坏 + 取向冲突不可表达，探针实证归档）** | 精度: **pass（门禁全过）**——①机制探针 probe_r81_l0b4_mech PASS（v4ksp vs 生产参照 v2cur：fp16/bf16 × int/rand 4 配置 **BITEXACT** + golden + 预算读数 L0A 24/L0B 64/L0C 64KB）；②全 kernel 探针 probe_r81_l0b4 **20/20**（PART 0 分派审计 14 目标全落 B-mma（batch_tasks {1,3,4,6,7,8} 奇偶、multi-outer 8+2、effvar causal 1..4、case9/case4 bench 原形）+ 6 守卫不落；PART 1 **14/14 对改前 BITEXACT** + golden 阻塞口径全过（最差 bf16 MARE_abs 7.81e-3 vs gate 7.8e-2 = 10x）；PART 2 守卫 6/6 BITEXACT）；③锚-A **9/9 BITEXACT 硬守卫** + 锚-B **14/14 BITEXACT**（链式重排全序保持，失配预案未触发）；④压力：继承 4 配置 ×12 nondet=0 + rev4 扩展 16 项（专打 Form B-mma token 流 = 新 4 槽 L0KC/L0AS 组令牌/L0AO 双 colseg 形态）×12 **nondet=0**；⑤l0 10/10 + l1 19/19 + coverage 19P/0M/0N + `--level all` **exit 0**（L2 12/12 拒绝 + Boundary 6/6 + 哨兵 4/4 PERF_PASS）；⑥IR 守卫 **13/13 逐字节相同**（Form A ×8 + decode ×5 = decode 硬约束 IR 级成立）+ 3/3 目标按预期改变（case9 22617→24558B）；⑦预算硬门（实编 GetWithOffset 读数）：L0B **64.00KB/64KB 顶格单池**（InitBuffer 65536）/ L0A 32 / L0C 64 / L1 152 / UB 126KB 全 OK；⑧生成码静态核验：mma ×8 全模板 <32,128> K=64 运行期、copy_l1_to_l0b 全部 dst(64,128) 同取向、v_l1 行列双切片偏移 0/1024/16384/17408（ZN Forward 公式逐项正确）、CrossCore SEM census md5 与基线**全等**（协议零漂移）、MTE1↔M 字面 id 0-6 = 7/8 ≤ 上限。
| 性能（msprof median us，同会话基线对比，msprof_r8_iter1/）: **case9 944.74→946.93 (+0.23%) 噪声带零响应**；case4 35.57→34.36 (-3.4%，唯一超带读数、单测未复核——回滚后不适用，如实记录）；守卫 case7 -0.05%/case10 +1.1%/case14 -0.5%/case16 +0.02%（IR 逐字节未变 ⇒ 带幅=会话方差标尺）。
| **op 归因 A/B（/tmp/opencode/r8_op_case9_base vs r8_op_case9_iter1，判据 (a) 的裁决证据）**：
  · **mte1_stall 433.96→425.23us（-2.0%）——机制未激活**（判据要求"显著下降且 stall≤busy"；实测 stall(425) > busy(328) 签名存续）；
  · mte1_busy 321.4→328.0（+2.1%）、mte2 476.7→492.8（+3.4%）、fixp/cube/scalar ±1% —— 事件成本真实、收益缺失；
  · C wait_id2 340.7→257.9（-24.3%）+ wait_id5 -8.2%：C 更早到达 w2/w3 交接点 = 微小节省再度被 V 侧交接吸收（r7-1/r7-2 吸收律同款签名第三次复现）；
  · V 侧全读数 ±5% 内（emit_v 零触碰的 op 级佐证）。
| 采纳: **NO — 回滚**（rollback_reason: 判据 (a) 机制激活失败 + 代价净值裁决为负；回滚至 r4_perf4 = ee5bf9c9，md5 核验 + IR 级还原 16/16 逐字节 diff r7_src_rollback2 无差异[新鲜复跑]）
| **归档知识（判决性，新增硬知识三条）**：
  1. **mte1_stall 的暴露根因不是突发内 L0B 槽深**——4 槽 ×16KB + load-first 连续流实施后 stall 仅 -2%：其门控 = **跨迭代 MTE2 填充（K/V 到达，MTE2 493us = 50.5% 真机最忙管，1.09us/iter ≈ 节拍地板）+ 四相位 SEM 链**。附.3 "前瞻深度 0→2 → 暴露 stall≈0" 假设**在现批结构下证伪**（前瞻只能在突发内起效，而突发头部被 k_l1/v_l1 到达门控、尾部被 SEM/FIX 链门控）。与 r7-2 判决性先例合并判读：r7-2 从 MTE2 侧攻击（k/v L1 双槽 → stall -54%）wall 零响应；本轮从 L0B 侧攻击（stall -2%）wall 零响应 ⇒ **mte1_stall 读数本身不是 case9 wall 的杠杆点，它是四相位链的伴生症状**——②手段族证据性关闭（激活前置 = flat ① 落地后 MTE2/MTE1 流跨任务连续化，届时若 op 归因仍显示 MTE1 暴露段可零探针成本复活本实现——变体已留存且全门禁通过）。
  2. **附.3 字面形态（4×[128,64] = GEMM2 4×NC64 n-chunk + w3 FIX ×4）在本框架不可表达**：16KB 槽下 GEMM1 B-tile [KC',bN] 与 GEMM2 B-tile [bN,NC'] 互为转置（DESIGN §4.5-e "异形 view 共池无仓内先例"判定的 4 槽版复现）；双取向需 T.view/T.reshape，而其在本 TVM 版本损坏（customize.py:94/112 调用不可直接构造的 tvm.tir.Buffer——探针实证）。同取向替代（PV K-split×N-split，V 行列双切片直载）**可用且位级安全**（20/20 BITEXACT 入库：ZN L1 行列双切片 MTE1→L0B 直载 = 新组合首次验证；PV K-split 链 ≡ 单 K mma 再证；Q/P colseg 64 宽 ×4/×2 再证），未来 4 槽需求可直接复用 v4ksp 形态。
  3. **事件代价账（实测）**：MMAD 4→8/iter 在 cube 19.4% 占空下近零 wall 成本（cube -1.3%）；MTE1 事件 7→13/iter 成本 = busy +2.1%；FIX 3/iter 不变（v4ksp 形态优于字面形态的 5/iter）。若未来复活，代价侧无阻塞。
| **①+③ 二元联合残余可行性评估（任务书要求项，如实）**：
  - **case7/8/3（form_a 多块）**：主杠杆 = ③（V vec 68.3% + vector_stall 258us = 最大真实工作管，§S7-3-5）——**不受 ② 回滚影响**，Stage-2 照常推进；①（flat）对 form_a 多块的 K/V 冗余无解（kv-outer 禁令域），其收益面 = 任务边界与批深摊薄，EV 维持 §S7-5 scoping（-5~-15% 上界依据不变）。
  - **case9（form_b_mma）**：B 层 626us 收敛模型（附.6）假设 C.MTE2 100% 定拍 + 其余管道全隐藏。② 回滚后：MTE1 理想占空 64%（附.6）的前提"暴露 stall≈0"失去 ② 的支撑，但 **① flat 本身就是 MTE1/MTE2 流连续化的更根本手段**（跨任务边界不排空 → 突发头部门控消失，2 槽池在定拍流下前瞻深度 1 或已足够——模型级论证，非实测）；③ 使 V 相位收缩 → C 链等待余量再分配。**残余风险（如实报告）**：若 flat 落地后 op 归因仍显示 MTE1 暴露段（2 槽深度不足），需在 ① 之上复活 ②（变体已备好，零探针成本）——即 case9 的 B 层收敛从"①+②+③ 三包"退化为"①+③ 二元 + ② 条件复活"，收敛不确定性上升一档；626us（0.54x）不再是可承诺预期，改为"①+③ 实测裁决 + ② 按需复活"的路径。
  - 结论：**二元联合可继续推进**（Stage-2 ③ → Stage-3 ① → Stage-4 联合判 wall），② 转入条件复活状态。

### 迭代 r8-2 前置：③ UB 预算硬门实测 + [ORDER-PLAN 数据驱动修订]（2026-09-29）

**[#R8-2-pre] Stage-2 ③ 宽通道 UB 预算探针（任务书硬门"三 D 配置实编 InitBuffer ≤191.75KB"，probe_r82_wide_ub_budget.py，P4 实编 GetWithOffset 读数）**：
| D 配置 | VW=1（现状核验） | VW=2 | VW=4 | VW=8（任务书理想"8 槽合并 [128,128] 级"） |
|---|---|---|---|---|
| 64（bM64/hM32, io3/r_2d 分池） | 135.50K ✓ | 168.75K ✓(+23.0) | **234.50K ✗ OVER** | **366.00K ✗ OVER** |
| 128（bM64/hM32, io3/r_2d 别名） | 123.50K ✓ | 156.75K ✓(+35.0) | **222.50K ✗ OVER** | **354.00K ✗ OVER** |
| 256（bM32/hM16, io3/r_2d 分池） | 122.00K ✓ | 138.62K ✓(+53.1) | 171.50K ✓(+20.25) | **237.25K ✗ OVER** |
（宽通道设计 = work_ub/rr_ub/acc_s_half/mask_2d 按 VW 加宽 + m_w/m_pair/sumexp_w 微缓冲；buf_2d/rt_2d/io_buf/acc_o/out_half 保持窄形——宽 buf_2d_w 变体在 D=128 另爆 ~+32KB，已排除。）
**判决**：③ 的任务书理想形态（8 槽一次宽通道）**UB 硬性不可行（三 D 配置全爆）**；可行包络 = VW=2（全 D）/ VW=4（仅 D=256）⇒ 碎片事件削减 ~2-4×（非理想 ÷5-8）；叠加 r7-1 判决性先例（V-P1 真实工作削减 → C mte1_stall +33us 1:1 吸收，wall +0.31%）与 #R3-3（case7 V 侧削减零响应）⇒ **③ 单独必被吸收、联合贡献按 VW 缩减而边际化**。

**[ORDER-PLAN 数据驱动修订]（先例 = 会话 4 r4-2 修订格式）**：Stage-2（③）与 Stage-3（①）执行序对调——**① flat prefill 提前为 r8-2，③ 降级为条件项（① 采纳后预算允许时按 VW=2/4 缩减形态再评）**。依据（四条，全部实测/归档证据）：
1. **任务书 Stage-1 失败条款授权**："净值裁决为负 → 归档 + 回滚，并评估 ①+③ 二元联合残余可行性"——评估完成（上表 + [RESULT-#R8-1] 归档）：二元中 ① 是唯一有独立 EV 的杠杆（EV -5~-15%，§S7-5 scoping：C 轮界等待 87us + V task-tail 段 + 批深 4.5→8 + 任务边界 102→~58），③ 独立必被吸收（吸收律 + r7-1/#R3-3 双先例）且理想形态 UB 不可行。
2. **杠杆性质差异**：②③ = 单侧吞吐类（吸收律域，数学上无独立杠杆）；① = **交接结构删除类**（批轮次/排空点削减 + V task-tail 与批界解耦——直接缩短四相位链本身，非单侧加速）。
3. **风险-预算比**：① 位级预期 BITEXACT（#R4-3 17/17 蓝图含跨轮任务状态携带/轮中任务切换/phase2 复位时序 + #R5-1 SIG_Q_L1 收支范式），锚体系完整保留；③ 非位级预期（任务书自判）→ 锚重建额外预算 + 精度门最重。软预算余量（~3-4 迭代）下先做低精度风险、有独立 EV、且可 **form_b_mma 域 scoping（14/16 IR 守卫保留）** 的 ①。
4. **组合顺序工程性**：① 先行 → ③ 若后续实施，其宽通道天然作用于 flat 流的连续槽窗（组合自然）；反序则 ① 需重犁 ③ 的宽通道结构。mission Stage-3 自身的域序（form_b_mma 先行 → form_a 多块）保持。
修订后执行序：r8-2 = [#R8-2] ① flat(task,kv) form_b_mma generic 域 → （采纳后）r8-3 = ③ 缩减形态条件评估 / form_a 域扩展评估 → r8-4 = 联合门禁 + wall 裁决 → r8-5 = 收口。

### 迭代 r8-2：[#R8-2] Stage-3 ① flat(task,kv) 连续令牌流 prefill 扩展（form_b_mma generic 域 = case9/case4）（2026-09-29）

[ORDER-CHECK] 准备实施: [#R8-2] flat prefill（form_b_mma 域先行） | 前置依赖: [ORDER-PLAN 修订 ✅（上节四条依据）；#R4-3 蓝图 ✅（17/17 BITEXACT 归档，含 ①-⑪ 全约束清单）；§S4-3-5 复位时序 ✅；T.alloc_var 运行期状态机先例 ✅（sparse_flash_mla:220-222/330-355 kv_iter/pc_cur += 模式）] | 结论: ✅
[IMPL-#R8-2] 已阅读: #R4-3 [IMPL]/[RESULT] 全记录（扁平索引双侧同式/phase2 复位/m_i 例外/任务尾 SIG_S_HALF 平衡闭环/rows_vid hoist 判据）、§S7-5 scoping（EV 与爆炸半径）、#R5-1 [IMPL]④（SIG_Q_L1 每槽收支恒等式）、DESIGN §7.2-5（C/V 同式硬约束）/§7.2-6（条件 DMA 禁令与同进同出豁免形）、§8.6-10（共池相位纪律——flat 不引入 GEMM1/GEMM2 iter 级交错，批内仍 G1 批全先于 G2 批 ✓）、mha.py generic 双分支现结构。
**关键设计（prefill 域新工程点 = 运行期非均匀 eff_kv 的扁平映射，mission 指定项）**：
① **映射机制 = 顺序游走状态机（无前缀表）**：kernel 头一次性 O(my_count) 纯标量游走求 total_flat = Σ eff_kv(t)（C/V 双侧同式各自计算，确定性算术 ⇒ 恒等，不经 ws 传递 §7.2-5）；num_fb = ceildiv(total_flat, 8)、batch_slots = min(8, total_flat - fb*8) 双侧同式。批内：批头快照 (tl_b, kv_b)（T.alloc_var），每条槽循环（C GEMM1/GEMM2、V phase1/phase2 共 4 处）从快照独立重走 (tl_w, kv_w)：每槽派生 task=my_start+tl_w → bx/by/bz/s0/valid_m、idx=kv_w → j0/valid_n、eff（causal 公式逐字双侧同式），槽末推进 kv_w+=1; if kv_w==eff: kv_w=0; tl_w+=1（纯标量运行期 if，无 DMA/flag——非 #R5-3 域）。批末持久状态 = 游走终值。**每槽新增标量 ~12 op ×4 循环 ≈ 9us/核 总账（scalar issue 现仅 58us/944us，余量充足）**；O(my_count) 一次性游走 ~0.6K op 可忽略。拒绝前缀表方案：max_grid 型 my_count 可达 819（表不可标量化）且游走已 O(1) 摊还。
② **C 侧任务切换（kv==0 运行期条件段，同进同出）**：SIG_Q_L1 完整轮（wait(MTE1→MTE2)/MTE2 copy/set/wait(MTE2→MTE1)）+ L0AS 组轮（wait(M→MTE1)/4 colsegs/set(MTE1→M)/wait(MTE1→M)）+ set(MTE1→MTE2,Q) 全部位于 `if kv==0:` 内——每任务净 0；L0AS 归还 set(M→MTE1) 位于 `if kv==eff-1:` 槽末（M 管序 = 本任务末 mma 读毕）；首任务 wait 消费 init 令牌、末任务归还由 destroy 消费 ✓。#R5-3 禁令合规性：DMA 与全部关联 flag 同条件同进出（非"省略 DMA 保留 flag"形态）；#R4-3 先例（decode flat 的 if kv==0 Q 载入段 17/17 BITEXACT + nondet=0）为同构协议形，本域增量 = SIG_Q_L1/L0AS flag 轮随 DMA 入条件（gemm_v0 域无用户 flag 故 #R4-3 未覆盖此点——探针矩阵+压力门实测收敛）。
③ **V 侧任务状态机（§S4-3-5 逐字继承）**：phase1 `if kv==0:` fill(m_i,-2^30) + row_thresh 重算（need_mask 时，纯 V 无 flag 段；m_i 例外 = phase1 独占，reduce_max 覆写语义 + m_i_prev 链 ⇒ 与 generic 任务头 fill→copy 位级同值——#R4-3 推演归档原文）；phase2 `if kv==0:` fill(acc_o,0)+fill(sumexp,0)（= generic 任务头等值迁移：首次消费点相同/fill 值相同；phase1 全先于 phase2 ⇒ 复位必须在 phase2 否则摧毁前一任务在途状态）；phase2 `if kv==eff-1:` 终归一（+1e-30 守卫逐字）+ SIG_S_HALF 闭环写回（wait/set ×4 同进同出，rows_vid 守卫内 MTE3 copy——条件 vid 无关：kv/eff/tl_w 派生自循环索引+双侧同式量，双 vid 同值同分支 §7.2-5）。跨批任务：acc_o/sumexp/m_i UB 常驻携带（#R4-3 ⑨ 先例）。
④ **SEM/协议面**：批级 wait WS1_V2C（C GEMM1 头）/wait WS3_V2C（C GEMM2 头）/set WS2_C2V（C 批末）与 V 镜像 wait WS2_C2V（phase1 头）/set WS1_V2C（phase1 末）/set WS3_V2C（phase2 末）逐字保持；槽级 WS1_C2V/WS2_V2C/WS3_C2V 的 CI 条件式逐字保持（i→批内槽索引）；**计数恒等**：C 每批 set WS1_C2V ×batch_slots = V 每批 wait ×batch_slots（双侧同式 batch_slots）✓。ws 槽索引 = 批内 i（0..batch_slots-1 ≤ NUM_STAGES-1）✓ 复用协议不变。init/destroy 逐字不变（收支由 ②③ 的每任务/每槽净 0 保证）。
⑤ **位级预期 = BITEXACT**（验收门）：每槽算子序列与 generic 逐 op 同序（掩码条件 idx→kv 同值代入；j0/valid_n 同式；fill 时点等值迁移已推演 #R4-3 ④/⑥）；side=i%2 轮转在 flat 批下与 case7 batch_iters=8 同构（生产验证形）；纯调度/协议重排（算术零变更）。
⑥ **域 scoping（mission 域序 + 最小爆炸半径）**：C 侧 = form_b_mma prim_func 的 generic else 分支整体置换为 flat 体（该分支定义域 ≡ form_b_mma ∧ ¬single_kv_block，无新增谓词）；V 侧 = emit_v 新增 `elif flat_v:` 分支（flat_v = form_b_mma and not single_kv_block，闭包 parse-time bool）插在 batched 与 legacy generic 之间——**Form A 全域/decode 兜底域/form_b_mma 单块域走原分支语句零变化 ⇒ 14/16 IR 守卫逐字节可证**（变更域 = G_case9/G_case4 两 shape）。
⑦ 令牌守恒静态可验（§7.2-2a 扩展）：SIG_L0KC/L0AO/L0C/K/V/P 每槽收支逐字继承现 generic 槽体（flat 不改槽内 token 形）；SIG_Q_L1/L0AS 由每任务轮→每任务条件轮（净 0 迁移，②推演）；同进同出无条件（#R5-3 教训）：全部条件段内 DMA+flag 完整成对。
⑧ 硬约束继承：T.Pipelined/threads= 禁用；无虚拟通道；共池相位纪律（批内 G1 批全先于 G2 批）；CROSS_INTERVAL/NUM_STAGES 不动；decode 兜底域 C 侧源码零触碰。
[SELF-CHECK] 本次 Edit 只涉及 [#R8-2]：①host `flat_v = form_b_mma and not single_kv_block`（1 行 + 6 行注释）；②emit_v 新增 `elif flat_v:` 分支（~190 行：total_flat 游走 + fb 批循环 + phase1/phase2 双 walk + kv0/kv-last 条件段），single_kv_block 批分支与 legacy generic else 分支**语句零变化**（elif 插入点 = 原 else 之前）；③form_b_mma C-scope generic else 分支整体置换为 flat 结构（~200 行：total_flat 游走 + fb 批循环 + GEMM1/GEMM2 双 walk + kv0 Q/L0AS 条件轮 + kv-last L0AS 归还），槽体语句逐字搬运（idx 派生 k*NUM_STAGES+i → kv_g*，其余 verbatim）；④Form A prim_func / decode 兜底 prim_func / form_b_mma 单块批分支 / init / destroy / allocations / SEM-SIG id 矩阵 / host 包装**零触碰**；⑤静态核对清单：令牌收支（L0AS/L0KC/L0AO/L0C/K/V/P/Q 全 id 双向净 0 + init/destroy 平衡——条件轮同进同出）、SEM 计数恒等（批级 3 对 + 槽级 CI 条件式逐字，batch_iters 双侧同式）、walk 一致性（4 处循环独立重走同批头快照，纯标量无 DMA/flag）、跨批任务状态 UB 常驻（acc_o/sumexp/m_i）、phase1/phase2 复位时序（§S4-3-5：m_i 例外在 phase1 kv0、acc_o/sumexp 在 phase2 kv0、任务尾在 phase2 kv-last）、ws 槽索引 = 批内 i、side=i%2 批内轮转、批结构保持（G1 批全先于 G2 批，§8.6-10 共池相位纪律）；⑥AST/ruff check/format clean（1839 行）；⑦备份 = history_version/mha_impl_r4_perf5.py（改前 ee5bf9c9）。
[KERNEL-REUSE-AUDIT] 依据: Part A8 [KERNEL-REUSE-PRECHECK] #R8-3(=实施编号 #R8-2) + #R4-3 蓝图 | 维度及结论: ①eff 均匀性=prefill 域 FALSE（运行期非均匀，case9 均值 4.5）→ 顺序游走状态机替代编译期派生（O(1) 摊还/槽 + O(my_count) 一次性 total_flat；alloc_var 变异先例 sparse_flash_mla:220/330-355 + parser local.var buffer_store 语义源码核验 parser.py:130-138）；②C/V 同式=TRUE（total_flat/num_fb/batch_iters/eff 公式与游走推进逐字双侧一致，同一 my_start/my_count 派生，不经 ws）；③V 侧边界状态复位/写回=TRUE（§S4-3-5 逐字：phase2 kv0 fill acc_o/sumexp、phase1 kv0 fill m_i + row_thresh、phase2 kv-last 终归一+写回 SIG_S_HALF 闭环同进同出、rows_vid 守卫嵌套保持）；④hoist 等值=n/a（s0 运行期变异，row_thresh 逐任务 kv0 重算）；⑤lane 语义=TRUE（r_factors[i]/sumexp_is[i] 批内槽 lane，phase1→phase2 同批传递，跨批覆写序与 legacy 同构）；⑥位级=探针已裁决 **16/16 BITEXACT**（含跨轮携带/轮中切换/eff1..8 异构/奇偶批/尾块/bf16/zeros/scale）| 结论: repair-kernel-first → 已实施，probe 门通过
[PROBE-#R8-2 中间记录] 探针 probe_r82_flat **23/23 PASS**（/tmp/opencode/r8_probes/probe_r82.log）：PART 0 分派审计 16 目标全落 B-mma∧flat 域 + 7 守卫全不落（含 G_B_single_d256 单块域收窄核验）；PART 1 **16/16 对改前（r4_perf5=ee5bf9c9）BITEXACT** + golden 阻塞口径全过（最差 bf16 case 7.81e-3 vs gate 7.8e-2 = 10x；zeros 位级 0.00e+00）；PART 2 守卫 7/7 BITEXACT。**IR 守卫 14/16 逐字节相同**（Form A ×8 + decode ×5 + form_b_mma 单块 ×1；vs r7_src_rollback2 provenance-verified 参照）+ 2/2 flat 目标按预期改变（case9 22617→?B, case4）——decode 不回退硬约束 IR 级成立。smoke（flat 域 + 单块域）先行 PASS 无死锁。门禁电池（双锚 + 继承压力 4 + **扩展压力 22 项含 6 项 flat 专项**（f01-f06：eff 1..4/1..8 异构轮中切换、eff{4,5} 交替、跨轮 causal、eff=1 边缘、bf16 twin）×12 复跑 + l0/l1/coverage/all）运行中。
[RESULT-#R8-2] 优化点: **Stage-3 ① flat(task,kv) 连续令牌流 prefill 扩展（form_b_mma generic 域 = case9/case4；顺序游走状态机映射运行期非均匀 eff_kv；C/V 双侧同式；任务切换 Q/L0AS 条件轮同进同出；V 状态机按 §S4-3-5：phase2 kv0 复位 acc_o/sumexp、phase1 kv0 m_i+row_thresh、phase2 kv-last 终归一+写回 SIG_S_HALF 闭环）** | 精度: **pass（门禁全过）**——①探针 probe_r82_flat **23/23**：PART 0 分派审计 16 目标全落 flat 域 + 7 守卫不落（含单块 d256 域收窄核验）；PART 1 **16/16 对改前 BITEXACT**（全拓扑：case9/case4 bench 原形、eff 1..4/1..8 异构轮中切换、eff{4,5} 交替、跨轮任务状态携带 kv10（non-causal+causal）、奇偶批 batch_tasks {1..8}、S 尾块/S_kv 尾块 300%128、bf16、大幅值域、zeros 陷阱对照（位级 0.00e+00）、显式 scale）+ golden 阻塞口径全过（最差 bf16 7.81e-3 vs gate 7.8e-2 = 10x）；PART 2 守卫 7/7 BITEXACT；②锚-A **9/9 BITEXACT 硬守卫** + 锚-B **14/14 BITEXACT**（协议重排位级零漂移，#R4-3/#R5-1 先例四度确认）；③竞态压力：继承 4 配置 ×12 nondet=0 + 扩展 **22 项**（rev4 16 项 + **flat 专项 6 项 f01-f06**：eff 异构轮中切换/eff=1 边缘/跨轮 causal/bf16 twin）×12 复跑 **nondet=0**；④l0 10/10 + l1 19/19 + coverage 19P/0M/0N + `--level all` **exit 0**（L2 12/12 正确拒绝 + Boundary 6/6 + 哨兵 4/4 PERF_PASS）；⑤**IR 守卫 14/16 逐字节相同**（Form A ×8 + decode ×5 + form_b_mma 单块 ×1 = decode 不回退硬约束与单块域 IR 级成立）+ 2/2 flat 目标按预期改变；⑥精度红线四项：双舍入栅格链逐字未动 ✅ / 单块预归一特化未触碰（域收窄）✅ / sumexp+1e-30 守卫逐字迁移至 flat 任务尾 ✅ / 掩码链语义等价（idx→kv 同值代入，探针 causal/bound/tail 变体 BITEXACT）✅；disable_cache/会话专属缓存全程。
| 性能（msprof median us，同会话基线 msprof_r8_base 对比，msprof_r8_iter2/）: **case9 944.74→869.51 (-7.96%)**、**case4 35.57→28.20 (-20.72%)** —— 双目标远超 3% 噪声带，case9 落于 mission EV 带（-5~-15%，§S7-5 scoping）中枢；守卫 4 case（IR 逐字节未变）: case7 +0.25% / case10 +0.9% / case14 +0.3% / case16 +0.07% 全噪声带内零回退。
| **op 归因 A/B（/tmp/opencode/r8_op_case9_base vs r8_op_case9_iter2，机制验证）**：
  · **C wait_id5（批头 WS3_V2C 轮界等待）87.57→2.16us（-97.5%）——mission EV 上界依据（"C 轮界等待 87us"）近乎全额兑现，本项靶心签名**；
  · C wait_id2（等 w2）340.71→232.06（-31.9%）、**V wait_id4（等 w3）431.51→247.92（-42.5%）、V 核总时长 902.68→815.13（-9.7%）**——四相位交接链双侧同时收紧（非单侧节省被对侧吸收的形态）；
  · 真实工作管全平：mte1_busy ±0.0% / mte2 -1.2% / fixp -1.7% / cube +0.3% / V.vec +0.5% ⇒ **纯结构收益（批轮次 ~103→~58/核 + 任务边界与批界解耦 + 批深 4.5→8），零字节/零工作量变化**；
  · mte1_stall 434→498（+14.7%）：链收紧后 scalar 更早到达装载等待点（生产性 pacing 重分配，wall 仍降——与 r8-1 判读一致：stall 读数为伴生症状）；V wait_id0 +25.6%（V 更早到达 w1 等待点，同理）；
  · TaskDur（op 口径单跑）944.76→884.56（-6.4%，与 timeline median -7.96% 方向量级一致）。
| 对比（估算，**非权威**，权威 = Orchestrator 轮末 cann-bench 复评）: case9 0.353→~0.383x / case4 0.846→~1.07x（按 T_base/T_new 比例外推，cann-bench 固定开销不等比缩放警示同 §S5-2），20-case 平均贡献 ~+0.013 ⇒ ~0.92x 级（估算）。
| 采纳: **YES**（判据：2/2 目标 case 提升 8.0%/20.7% 远超噪声带 + 4/4 守卫噪声带内且 IR 逐字节 + 精度 6 门全过含 flat 专项压力 + 机制靶心签名 wait_id5 -97.5%）
| 新基线 = msprof_r8_iter2/ + r8_op_case9_iter2；采纳态 = history_version/mha_impl_r4_perf6.py（md5 **5d466b5ff3808bd8273eb85915ca4484**，1839 行）；备份链 perf3（开工 ee5bf9c9）→ perf4（r8-1 改前）→ perf5（r8-2 改前）→ perf6（r8-2 采纳态）。
| **归因知识（新增硬知识）**：①**吸收律的边界补全**——单侧吞吐类手段（②③/r7-1/r7-2）被 1:1 吸收，但**交接结构删除类（①flat）不被吸收**：wait_id5/wait_id2/wait_id4 三处等待同时下降 + 双侧真实工作管零变化 + wall -8% ⇒ 四相位链的"每槽交接延迟"确为可删除成分，mission 归因（now→B 差距 63% = 交接延迟链）获治疗性验证；②flat prefill 的运行期非均匀 eff 映射 = **顺序游走状态机**可行（无前缀表、O(1) 摊还/槽、alloc_var 变异、4 循环独立重走批头快照——C/V 恒等由确定性算术保证），#R4-3 decode 蓝图的 prefill 推广完成；③case4 -20.7% > case9 -8.0%：收益与"legacy 批深空缺率"成正比（case4 eff 1..2 → 每任务 1 个 1-2 槽微批 → flat 8 槽满批，协议轮 ~6→~2；case9 eff 均值 4.5 → 轮次减半）——**flat 收益模型 = f(批深空缺率, 任务边界密度)**，可预测 form_a 大 prefill（case7/8: eff 1..8, s_blocks=16, 边界密度同 case9 量级）受益同 case9 级。

### 迭代 r8-3：[#R8-3] Stage-3 ① flat prefill 域扩展（form_a 多块 = case7/8/3/2/5；mission 域序第二步）（2026-09-29）

[ORDER-CHECK] 准备实施: [#R8-3] form_a flat 扩展 | 前置依赖: [#R8-2 ✅ 已采纳（flat 骨架/游走状态机/V 侧 flat 分支定型——本轮仅 C 侧 Form A generic 分支置换 + flat_v 谓词扩展，V 侧零改动）；收益模型 ✅（#R8-2 归因知识③：收益 = f(批深空缺率, 边界密度)，case7/8 eff 1..8 与 case9 同构 ⇒ 预期同量级）] | 结论: ✅
[IMPL-#R8-3] 已阅读: mha.py Form A generic 现结构（P1/P2 池别名规则 l0a_o=l0a_s/l0b_o=l0b_s/l0c_o=l0c_s iff block_N==D；`if i < 2` 批头 Q 重载机制；任务级 SIG_Q_L1 收支）、#R8-2 flat 骨架（逐字镜像源）、#R5-1 [IMPL]④（SIG_Q_L1 每槽收支恒等式）。
**Form A 特有增量（两项，均为 Q/l0a_s 调度、零算术变更）**：
① **l0a_s 重载条件 `i < 2 or kv_g1 < 2`**——P2 共池下 GEMM2 批以 P 覆写 l0a_s（l0a_o 别名），legacy 依赖每批头 i<2 重载 Q；flat 下任务可始于批中部 ⇒ 新任务头两槽必须重载各自 side（Q 任务内不变，kv≥2 槽复用驻留副本）。P1 分池下批头重载为幂等冗余（无害，parse-time 单一条件覆盖两池配置）。
② **q_l1 归还移至任务末槽（kv==eff-1，MTE1 管序覆盖）**——设计期竞态推演发现：跨批任务在后续批头 i<2 槽**再读 q_l1**（①的重载），若沿用"kv1 后归还"（#R8-2 fbm 形——fbm 无批头重载故安全），下一任务 kv0 的 MTE2 Q 载入（wait 消费该归还）可能与在队 MTE1 批头读**竞态覆写**。修复 = 归还 set 置于任务末槽 MTE1 流（in-order 覆盖本任务全部 q_l1 读，含后续批头重载）；每任务收支不变（kv0 完整 SIG_Q_L1 MTE2 轮 + 末槽归还 = 净 0，init/destroy 逐字）。**该竞态为 Form A 独有（P2 批头重载 × flat 跨批任务组合），fbm 域不受影响（l0a_s 独立池、无批头重载）——设计期静态推演捕获，零实测缺陷成本。**
③ 其余逐字镜像 #R8-2：total_flat 游走/num_fb/batch_iters 双侧同式、批头快照 + GEMM1/GEMM2 双游走、GEMM2 槽体无任务切换条件段（V/P 每槽装载）、批结构保持（G1 批全先于 G2 批）、V 侧 emit_v flat 分支零改动（flat_v 谓词扩展 = (S>16)∧¬single_kv_block，form 无关性已核验：分支体仅引用共享参数集）。
[KERNEL-REUSE-AUDIT] 依据: #R8-2 归档机制 + Part A8 #R8-3 | 维度及结论: ①C/V 同式=TRUE（V 侧零改动，C 侧游走公式逐字镜像）；②令牌守恒=TRUE 静态推演（Q 轮每任务净 0 迁移至切换粒度；L0AB/L0C/K/V/P 每槽收支逐字继承；归还位点竞态修复见 IMPL-②）；③位级=smoke 已证 5/5 BITEXACT（P2 kv2/P1 bigbatch/P2 causal 大形状/fbm 双回归），全矩阵探针裁决中；④P1/P2 池别名=条件覆盖两配置（①）；⑤爆炸半径=Form A generic 域（case7/8/3/2/5 + L1 部分 + 哨兵 prefill_worst），单块/decode/fbm 域 IR 逐字节预期 | 结论: repair-kernel-first（#R8-2 机制直接复用 + 两项 Form A delta）
[SELF-CHECK] 本次 Edit 只涉及 [#R8-3]：①flat_v 谓词 form_b_mma∧¬single → (S>16)∧¬single（+注释更新）；②Form A C-scope generic else 分支整体置换为 flat（~205 行：镜像 #R8-2 骨架 + IMPL-①② delta；槽体语句逐字搬运，idx 派生 k*NUM_STAGES+i → kv_g*）；③emit_v flat 分支/Form A 单块批分支/Form B-mma 全体/decode 兜底/init/destroy/allocations/SEM-SIG 矩阵/host 包装零触碰；④AST/ruff check/format clean（1900 行）；⑤备份 = history_version/mha_impl_r4_perf7.py（改前 5d466b5f = #R8-2 采纳态）。
[PROBE-#R8-3 记录] 探针 probe_r83_flat_a **有效判定 PASS**（/tmp/opencode/r8_probes/probe_r83.log + 两项独立复核）：
- PART 1: **15/15 目标（Form A generic 全域拓扑：case7/3/2/5 bench 原形 + P2 eff=1 边缘/kv2 锚形/tail777/跨轮 kv10/双尾块/bf16 切换/scale/大幅值域/zeros + P1 跨轮 causal/tail）对改前（r4_perf7 = #R8-2 采纳态）BITEXACT** + golden 阻塞口径 14/15 过；
- PART 2: 守卫 **7/7 BITEXACT**（fbm generic/单块 + decode ×3 + Form A 单块 ×2）；
- **两项脚本级伪影的独立复核（如实归档）**：①PART 0 首跑 15 个 WRONG-PATH = 探针脚本谓词笔误（继承 r82 的 `form=="B-mma"` 检查未随域切换更新）——修正谓词独立重算 **15/15 目标落 flat 域 + 0/7 守卫误落** PASS（纯 host 计划数学复核）；②A_p2_valrange (2,256,256,4,128) vrange_qk=(-30,30) golden 门 MARE_abs=1.36 超阈——**归因实验：旧核（r4_perf7，全门禁在案态）同数据同 shape 得逐位相同结果（matched=0.99924/MARE 1.36）⇒ 该极端值域形状的 golden 门伪影为既有属性（CPU-fp16 golden vs 器件在饱和指数域的差异），非 r8-3 回归**（新核输出与旧核 BITEXACT 为纯调度变更的正确判据）。D=256 版 valrange（r8-1 探针）过门正常（3.91e-3），D=128 版 scale=1/√128 更大所致——归档为探针形状选择教训（极端值域探针应同时报告 vs_old 与 golden 双判据，BITEXACT 优先）。
- smoke 先行 5/5 BITEXACT（P2 kv2 causal / P1 bigbatch bf16 / P2 大形状 / fbm 双回归）；
- **IR 守卫（三方比对）**：r8_src_iter3 vs r8_src_iter2（#R8-2 采纳态）= **12/16 逐字节相同**（含 G_case9/G_case4 = fbm 域 r8-3 零触碰的 IR 级证明）+ 4/4 Form A generic 目标按预期改变；vs r7_src_rollback2（ee5bf9c9 原始基线）= **10/16 逐字节相同**（decode ×5 + Form A 单块 ×4 + fbm 单块 ×1——decode 不回退硬约束 IR 级成立）+ 6/16 为 r8-2/r8-3 两轮目标域变更。附带核验：r8_src_iter2 vs 基线 = 14/16 + 2/2（[RESULT-#R8-2] ⑥ 声明的事后实证确认）。
- 门禁电池（r8_cache_s3 会话隔离缓存）：锚-A **9/9 BITEXACT**（#9 现落 flat 域 = 经验性 BITEXACT 而非 IR 保证，如实注记）+ 锚-B **14/14 BITEXACT**；继承压力 4 + 扩展 **26 项**（rev4 16 + flat-B 专项 6 + **flat-A 专项 4：f07 P2 eff1..4 切换/f08 P2 eff{1,1,2,2} 边缘/f09 P1 跨轮 causal/f10 P2 tail777**）×12 复跑 **nondet=0**；l0 10/10 + l1 19/19 + coverage 19P/0M/0N + `--level all` **exit 0**（L2 12/12 拒绝 + Boundary 6/6 + 哨兵 4/4 PERF_PASS）。
[RESULT-#R8-3] 优化点: **Stage-3 ① flat(task,kv) 连续令牌流 prefill 域扩展（form_a 多块 generic 域 = case7/8/3/2/5；~205 行 Form A C-scope generic 置换镜像 #R8-2 骨架 + flat_v 谓词扩展 (S>16)∧¬single_kv_block（form 无关）+ 两项 Form A delta：①l0a_s 重载条件 `i<2 or kv_g1<2`（P2 共池批头重载×任务始于批中部）②q_l1 归还移至任务末槽（设计期静态推演捕获的 P2 批头重载×跨批任务 MTE1/MTE2 竞态修复，每任务收支净 0 不变）；V 侧 emit_v flat 分支零改动）** | **会话 8 中断恢复注记（如实）**：r8-3 实施+门禁在前次会话完成、采纳判定因外部中断未落档；本恢复会话 Step A 溯源核验 = 当前 mha.py（md5 2728c96eb2fabbfff279918c61c6c3e0，1900 行，ast+ruff clean）新鲜 dump（disable_cache + 专属缓存 r8_cache_prov）与门禁参照 /tmp/opencode/r8_src_iter3 **16/16 逐字节一致**（/tmp/opencode/r8_src_prov + dump_prov.log）⇒ r8-3 全部门禁日志对当前文件态有效，直接复用；probe PART 0 修正谓词独立重算在恢复会话**再次执行确认 PASS**（15/15 目标落 flat 增量域 form_a∧¬single + 7/7 守卫域外，G_fbm_gen_kv2 属 perf7 已 flat 域、BITEXACT 预期与 PART 2 实测一致）。
| 精度: **pass（门禁全过，provenance-verified 复用 + 恢复会话抽查）**——①探针 probe_r83_flat_a 有效判定 PASS：**15/15 目标对改前（r4_perf7 = #R8-2 采纳态）BITEXACT**（Form A generic 全域拓扑：case7/3/2/5 bench 原形 + P2 eff=1 边缘/kv2 锚形/tail777/跨轮/双尾块/bf16/scale/大幅值域/zeros + P1 跨轮 causal/tail）+ golden 阻塞口径 14/15（唯一超阈 A_p2_valrange MARE 1.36 = **既有伪影非回归**：旧核同数据逐位相同，归档复核在案，vs_old BITEXACT 为纯调度变更正确判据）；守卫 **7/7 BITEXACT**；②锚-A **9/9** + 锚-B **14/14 BITEXACT**；③压力：继承 4 + 扩展 26 项（含 flat-A 专项 f07-f10）×12 **nondet=0**；④l0 **10/10** + l1 **19/19** + coverage **19P/0M/0N** + `--level all` **exit 0**（L2 12/12 拒绝 + Boundary 6/6 + 哨兵 4/4 PERF_PASS）；⑤IR 三方比对：vs r8-2 采纳态 **12/16 逐字节**（含 case9/case4 fbm 域零触碰 IR 级证明）+ 4/4 Form A 目标按预期变；vs ee5bf9c9 原始基线 **10/16**（decode×5 + Form A 单块×4 + fbm 单块×1 —— decode 不回退硬约束 IR 级成立）；⑥红线四项逐字保持（双舍入栅格/单块预归一特化域收窄/sumexp+1e-30/掩码链 idx→kv 同值代入探针 BITEXACT 佐证）。
| 性能（msprof median us，同会话对比；Form A 域基线 = msprof_r8_iter3ref = perf7 参考态实测——perf7 的 Form A 生成码与 ee5bf9c9 逐字节等价（r8-2 IR 守卫 14/16 含 Form A×8），run_perf_r83.sh PHASE 1 `[REF-r82]` 口径）:
| case | 域 | 基线（同会话） | iter3（2728c96e） | Δ | 交叉核对 |
|---|---|---|---|---|---|
| case7 | Form A 多块目标 | 926.04（iter3ref/perf7） | **856.62** | **-7.50%** | r8_base(ee5bf9c9) 924.69→-7.36%；iter2 926.99→-7.59%；op TaskDur 927.62→860.44 -7.24%；r7_base 跨会话 926.60 同向 |
| case3 | Form A 多块目标 | 167.71（iter3ref） | **131.48** | **-21.60%** | — |
| case2 | Form A 多块目标 | 67.11（iter3ref） | **44.91** | **-33.08%** | r7_base 跨会话 65.46 同向（-31.4%） |
| case5 | Form A 多块目标 | 18.81（iter3ref） | **15.40** | **-18.13%** | r7_base 跨会话 18.56 同向（-17.0%） |
| case8 | Form A 多块目标 | （Step C 实测中，不外推） | — | — | 与 case7 同构 bf16 twin，mission 明令必须实测 |
| case9 | fbm flat 守卫 | 869.51（iter2） | 868.31 | -0.14% | IR 12/16 逐字节 ⇒ 带内零回退 |
| case4 | fbm flat 守卫 | 28.20（iter2） | 28.22 | +0.07% | 同上 |
| case10 | decode 守卫 | 34.86（r8_base） | 35.16 | +0.86% | iter2 35.19→-0.09%；IR 逐字节 |
| case14 | decode 守卫 | 377.69（r8_base） | 375.03 | -0.70% | IR 逐字节 |
| case16 | decode 守卫 | 562.96（r8_base） | 563.24 | +0.05% | IR 逐字节 |
| case1/6/20 | Form A 单块守卫 | r7_base 跨会话 10.25/24.77/14.51 | 10.17/24.66/14.55 | -0.8%/-0.4%/+0.3% | IR 逐字节；同会话 ref 复测列 Step C 补齐 |
**四项目标全部远超 3% 噪声带（-7.5%~-33.1%），收益幅度序 case2>case3>case5>case7 与 flat 收益模型 f(批深空缺率, 任务边界密度) 定量吻合**（case2 batch_tasks{4,5,8}/my_count 12-13 = 高空缺率微批族；case7 batch_tasks{6,7,8}/my_count 102-103 = 近满批，收益收敛至 case9 同级 -7.5% vs -8.0%）。
| **op 归因 A/B（/tmp/opencode/r8_op_case7_ref vs r8_op_case7_iter3，block0 口径与 [RESULT-#R8-2] 一致；恢复会话新鲜解析）**：
  · **C wait_id5（批头轮界等待）101.27→4.83us（-95.23%）——flat 靶心签名在 Form A 域复现**（case9 fbm 域为 -97.5%，同机制同量级）；
  · C wait_id2 611.18→547.70（-10.39%）、**V wait_id4 285.71→141.95（-50.32%）**、V wait_id0 -3.92%（block0）/均值 -25.29%——四相位交接链双侧同时收紧（case9 同款非单侧吸收形态）；
  · 真实工作管全平：cube **+0.00%** / V.vec -0.58% / mte2 -1.93% / fixp +0.45% / mte1 +1.74% ⇒ **纯结构收益（批轮次 ~13→~8/核量级 + 任务边界与批界解耦），零字节/零工作量变化**——吸收律边界补全（结构删除类不被吸收）在 Form A 域二度实证；
  · mte1_stall 127.02→184.04（+44.89%）：链收紧后 scalar 更早到达装载等待点 = 生产性 pacing 重分配（case9 +14.7% 同款签名，r8-1"stall 为伴生症状"判读三度确认）；
  · TaskDur（op 口径单跑）927.62→860.44（-7.24%，与 timeline median -7.50% 方向量级一致）；aic_time -9.06% / aiv_time -9.06%（block0）。
| 对比（估算，**非权威**，权威 = Orchestrator 轮末 cann-bench 复评；cann-bench 固定开销不等比缩放警示同 §S5-2）: case7 0.464→~0.50x / case2 0.484→~0.72x / case5 0.538→~0.66x（按 T_base/T_new 比例外推，case7/case2/case5 基线速度比 = 会话 4/6 权威复评在案值）；case3 权威基线速度比未单独在案，仅计相对 wall -21.6%；case8 待 Step C 实测。四 case 合计 20-case 平均贡献估算 ~+0.026 级（case8 未计），叠加 #R8-2 的 ~+0.013 ⇒ 会话 8 累计估算 ~0.95x 级（估算口径，权威复评裁决）。
| 采纳: **YES**（判据：4/4 已测目标 case 提升 7.5%/21.6%/33.1%/18.1% 远超噪声带 + 守卫全域带内零回退（fbm -0.14%/+0.07%、decode ±0.9% 内、Form A 单块 ±0.8% 内且 IR 逐字节）+ 精度门禁全过（provenance-verified）+ 机制靶心签名 wait_id5 -95.2% + 收益序与 flat 收益模型定量吻合）
| 新基线 = msprof_r8_iter3/（12 case 已测 + Step C 补齐 8 case）+ r8_op_case7_iter3；采纳态 = **history_version/mha_impl_r4_perf8.py（md5 2728c96eb2fabbfff279918c61c6c3e0，1900 行）**；备份链 perf3（开工 ee5bf9c9）→ perf4（r8-1 改前）→ perf5（r8-2 改前）→ perf6（r8-2 采纳态）→ perf7（r8-3 改前）→ **perf8（r8-3 采纳态）**。
| **归因知识（新增/强化）**：①**flat 收益模型 f(批深空缺率, 任务边界密度) 获 Form A 域定量验证**——收益幅度与 legacy 批深空缺率同序（case2 {4,5,8} 微批族 -33.1% > case3 {1,2,8} -21.6% > case5 {2,3} -18.1% > case7 {6,7,8} 近满批 -7.5% ≈ case9 -8.0%）：近满批域收益收敛于"轮界等待删除"项（wait_id5 -95%±），微批域叠加"批深合并"项（协议轮 ~6→~2 级）⇒ 模型可外推未测 shape（但 mission 纪律：case8 仍须实测）；②**结构删除类不被吸收定律跨 form 复现**（fbm/Form A 两域 wait_id5/id2/id4 三处同降 + 真实工作管全平）——吸收律适用边界（单侧吞吐类被吸收、交接结构删除类不被吸收）升级为跨域定律；③Form A 特有竞态类（共池批头重载×跨批任务）在设计期静态推演捕获（IMPL-②），零实测缺陷成本——flat 域扩展的工程检查单新增"池别名重载点×任务边界交叉"审查项。

### 迭代 r8-4：[#R8-4] Stage-4 联合态 wall 裁决（Step C）+ 残余杠杆条件评估（Step D，评估归档）+ 收口（2026-09-30，中断恢复会话）

[ORDER-CHECK] 准备实施: [#R8-4] 联合门禁 + wall 裁决 | 前置依赖: [#R8-1 归档回滚 ✅（② 证据性关闭，复活通道条款在案）；#R8-2 采纳 ✅（fbm flat）；#R8-3 采纳 ✅（form_a flat，[RESULT-#R8-3] 恢复会话落档）；provenance ✅（当前源新鲜 dump 16/16 逐字节 = r8_src_iter3 ⇒ 门禁日志对当前文件态有效）] | 结论: ✅（裁决与复测，非迭代，不占软预算）

**Step C.1 联合态如实声明**：现联合态 = **① 全域落地**（fbm generic r8-2 + form_a generic r8-3；单块/decode 域分派谓词收窄零触碰）；② 已证据性关闭（r8-1）、③ 条件项未实施 ⇒ **mission 原"①+②+③ 联合 → B 层 626us（0.54x）"承诺维持 r8-1 next_hint 的降级口径，本节全部按实测报告，不引用 0.54x 作为已兑现值**。

**Step C.2 全 20 case 联合收益汇总表（msprof median us；现态 = 2728c96e（r8-3 采纳态）；基线 = 会话起点 ee5bf9c9 同会话读数——r8_base 为 ee5bf9c9 本体实测（case4/7/9/10/14/16），iter3ref/Step C 补测为 perf7 参考态实测（perf7 在被测各域的生成码与 ee5bf9c9 逐字节等价：Form A 全体由 r8-2 IR 守卫 14/16 证明、decode 兜底域源码零触碰 + iter3 IR 守卫 10/16 含 decode×5，口径如实标注）**：

| case | 域 | 基线 us（来源） | 现态 us | Δ% | 备注 |
|---|---|---|---|---|---|
| case1 | A 单块守卫 | 10.17（ref-perf7） | 10.17 | 0.00% | IR 逐字节 |
| case2 | **A 多块 ① 目标** | 67.11（ref-perf7） | **44.91** | **-33.08%** | r8-3 |
| case3 | **A 多块 ① 目标** | 167.71（ref-perf7） | **131.48** | **-21.60%** | r8-3 |
| case4 | **fbm ① 目标** | 35.57（r8_base 本体） | **28.22** | **-20.66%** | r8-2 |
| case5 | **A 多块 ① 目标** | 18.81（ref-perf7） | **15.40** | **-18.13%** | r8-3 |
| case6 | A 单块守卫 | 24.70（ref-perf7） | 24.66 | -0.16% | IR 逐字节 |
| case7 | **A 多块 ① 目标** | 924.69（r8_base 本体；ref 926.04 交叉 0.15%） | **856.62** | **-7.36%**（ref 口径 -7.50%） | r8-3 |
| case8 | **A 多块 ① 目标** | 911.82（ref-perf7） | **854.74** | **-6.26%** | r8-3；实测非外推（mission 条款） |
| case9 | **fbm ① 目标** | 944.74（r8_base 本体） | **868.31** | **-8.09%** | r8-2 |
| case10 | decode 守卫 | 34.86（r8_base 本体） | 35.16 | +0.86% | IR 逐字节 |
| case11 | decode 守卫 | 310.51（ref-perf7） | 306.15 | -1.40% | 兜底域源码零触碰 |
| case12 | decode 守卫 | 307.89（ref-perf7） | 307.38 | -0.17% | 同上 |
| case13 | decode 守卫 | 562.53（ref-perf7） | 564.16 | +0.29% | 同上；守卫线必需 case ✅ |
| case14 | decode 守卫 | 377.69（r8_base 本体） | 375.03 | -0.70% | IR 逐字节 |
| case15 | decode 守卫 | 506.24（ref-perf7） | 504.89 | -0.27% | IR 逐字节 |
| case16 | decode 守卫 | 562.96（r8_base 本体） | 563.24 | +0.05% | IR 逐字节 |
| case17 | decode 守卫 | 495.66（ref-perf7） | 497.15 | +0.30% | IR 逐字节 |
| case18 | decode 守卫 | 506.28（ref-perf7） | 504.90 | -0.27% | 兜底域源码零触碰 |
| case19 | A 单块守卫 | 21.75（ref-perf7） | 21.99 | +1.10% | IR 逐字节；r7_base 跨会话 21.85 交叉 |
| case20 | A 单块守卫（zeros） | 14.30（ref-perf7） | 14.55 | +1.75% | IR 逐字节（+0.25us 绝对量，会话方差；r7_base 14.51 交叉 +0.3%） |

**7/7 ① 目标域全部实测收益 -6.26%~-33.08%；13/13 守卫带内零回退（最差 +1.75% = IR 逐字节域会话方差，绝对量 +0.25us）**。收益幅度序 case2(-33.1) > case3(-21.6) > case4(-20.7) > case5(-18.1) > case9(-8.1) > case7(-7.4) > case8(-6.3) 与 flat 收益模型 f(批深空缺率, 任务边界密度) 全序吻合（微批族 case2/3/5/4 大幅、近满批族 case9/7/8 收敛于轮界删除项）。

**Step C.3 decode 不回退硬约束判定（本地 msprof 口径，同会话相对比较）**：case10 +0.86% / case13 +0.29% / case14 -0.70% / case16 +0.05% / case20 +1.75% —— 全部处于本会话已校准的守卫噪声带（±2.5%，r7-1 归档标尺）内，且 decode 域 IR 逐字节（5/5 dump 守卫）+ 兜底域 C 侧源码零触碰 ⇒ **本地口径无回退信号**。bench 口径线（case10≥0.90 / case13≥0.69 / case14≥0.99 / case16≥0.52 / case20≥0.82，会话起点在案值 0.983/0.886/1.394/0.785/2.079 全部带余量达标）的**权威换算与判定留给 Orchestrator 轮末 cann-bench 复评**——本地 msprof median 为相对口径，不构成权威 bench 判定（如实标注）。

**Step C.4 机制三段对比（op 归因 block0 口径，base→iter2→iter3；case9 的 iter3 态 = iter2 态 IR 逐字节（12/16 守卫含 fbm 域）⇒ 读数承继；case7 的 base 态 = iter2 态 IR 逐字节（r8-2 守卫 14/16 含 Form A×8）⇒ ref 读数双段承继）**：

| 指标（us） | case9: base | case9: iter2/iter3 | case7: base/iter2（=ref） | case7: iter3 |
|---|---|---|---|---|
| TaskDur | 944.76 | 884.56（-6.4%） | 927.62 | 860.44（-7.2%） |
| C wait_id5（轮界） | 87.57 | **2.16（-97.5%）** | 101.27 | **4.83（-95.2%）** |
| C wait_id2（等 w2） | 340.71 | 232.06（-31.9%） | 611.18 | 547.70（-10.4%） |
| V wait_id4（等 w3/S） | 431.51 | 247.92（-42.6%） | 285.71 | 141.95（-50.3%） |
| mte1_stall（伴生症状） | 433.96 | 497.81（+14.7%） | 127.02 | 184.04（+44.9%） |
| 真实工作管 | — | 全平（cube±0.3/mte2±0.3/fixp±1.7/V.vec±0.5%） | — | 全平（cube+0.00/V.vec-0.58/mte2-1.93/fixp+0.45%） |
| 批轮次/核、批深 | ~103 / 4.5 | **~58 / 8** | ~103 / 4.5（my_count 102-103, eff 均值 4.5, batch_tasks{6,7,8}） | **~58 / 8** |

三段判读：①的收益机制在两域同构复现（轮界等待 -95~97% + 双侧交接链收紧 + 真实工作管零变化 = 纯结构收益）；mte1_stall 上升为链收紧后的生产性 pacing 重分配（r8-1"伴生症状非杠杆"判读三度确认）；case7/9 的任务/eff 剖面同构（block_num 2048/20 核 = 102.4 任务/核、causal eff 均值 4.5）⇒ 批轮次 ~103→~58 同式成立。

**Step D [STEP-D-EVAL] 残余杠杆条件评估（数据驱动择一，选择依据落档；mission 授权条款"预算不足则归档收口"）**：

**评估选择 = (b) ③ 缩减形态（case7/8 域，VW=2 宽通道）为评估对象；(a) ② 复活被新鲜数据否决**。依据（全部实测/归档证据）：
1. **(a) ② 复活否决（新数据再证 r8-1 根因判定）**：联合态 case9 op 归因（r8_op_case9_iter2 = 现态，IR 逐字节承继）显示 MTE2 仍为最忙真实工作管（470.72us block0 / 508.66 均值，57.6-57.8%，flat 前后 ±0.3% 零变化 = 零字节变化）+ C scalar 95.5% —— case9 pacing = C scalar 链 + MTE2 填充门控，两者均非 L0B 槽深可攻击（r8-1 判决：4 槽 load-first 仅 stall -2%，突发头部被 k_l1/v_l1 到达门控）；mte1_stall 497.81（+14.7%）为伴生症状再证。② = 单侧吞吐类 ⇒ 跨域吸收律 + 双侧 A/B 界（r7-2 MTE2 侧 stall -54%→wall +0.35%；r8-1 L0B 侧 stall -2%→wall +0.23%）⇒ 预测 wall≈0，判据（wall 响应）预判失败；重集成成本（v4ksp 重织入 flat 游走体 + 全门禁链）不可辩护。**② 维持证据性关闭；复活前置更新 = 需新结构证据（如 MTE2 管收缩后 MTE1 成为 pacing 管）**。
2. **(b) ③ 选择依据（联合态新鲜 pacing）**：case7 为 **V-paced**——V.vec 634.39us = 77.5% aiv_time（最大真实工作管；§S7-3-5 的 68.3% 判定在 flat 后强化）+ C.wait_id2 547.70us = 67% aic_time（C 大量空转等 V）⇒ 收缩 V 相位即收缩 wall 关键路径的**结构可能性存在**；③ VW=2 攻击的正是该管的碎片事件税（46904 事件 ×~14ns，#R8-2-pre 归档）。UB 可行性已有实编裁决在案（probe_r82_wide_ub_budget P4 读数）：D=128 VW=2 = **156.75KB ✓（+35KB 余量 / 191.75KB 限）**——ORDER-PLAN 文本中"D=128 ~188KB 贴边"对应的是**已排除的宽 buf_2d_w 变体**，窄形设计实编通过；D=256 VW=2 138.62K ✓ / VW=4 171.50K ✓；D=64 保 W=1（分派谓词收窄）。
3. **(b) EV 裁决 = 不确定，双向证据均在案（如实）**：
   - 反方：③ = 单侧吞吐类 ⇒ 吸收律（r8-3 已升级为跨域定律）；直接先例 #R3-3（case7 V 侧 vector 指令削减 → wall +1.0% 零响应）+ r7-1（case9 V-P1 真实工作削减 vector_stall -37% → wall +0.31%，C mte1_stall +33us 1:1 吸收）；本会话 ORDER-PLAN 修订自判"③ 单独必被吸收、联合贡献按 VW 缩减而边际化"。
   - 正方：两先例均 **pre-flat**——#R3-3 的吸收通道 = 批轮界空缺 slack（批深 4.5 锯齿排空），已被 flat 删除（深 8 连续槽窗）；r7-1 的 pacing 管是 C scalar（95%）而非 V，case7 的 pacing 管是 V（77.5%）而 C 有大量余量（wait_id2 67%）；r8-3 机制读数（V.wait_id4 -50.3% 在 V 代码零改动下发生）证明 V 侧等待对结构敏感。**吸收律先例的因果结构（轮界 slack / C-paced）在联合态 case7 上已不成立 ⇒ ③ 的 post-flat EV 为真开放问题**。
   - 结论：**需专属会话全额预算裁决**（判据 = wall 响应主判 + 预测签名 V.vec 显著降 ∧ C.wait_id2 同步降 ∧ wall 降；若 V.vec 降而 wall 零响应 = 吸收律第四例，③ 族证据性关闭）。
4. **本会话不实施的如实理由（"预算不足"条款的诚实口径）**：软预算迭代 3/6 已用、名义余量 3——约束非迭代计数，而是 **恢复会话剩余执行窗口 + 风险控制**：③ 为 mission 自判"精度门最重"项（非位级预期 → 阻塞口径主门 + 锚-A #9/锚-B 多块成员降级重建方案须先行成文 + emit_v 全域爆炸半径 → 全 20 case msprof 复测替代 IR 保证），且 Step E 收口为必做项（终态新鲜门禁子集 + §S8 + 全链落档）；在本会话启动 ③ 将复现"实施中途外部中断"风险（本次恢复所清理的失败模式本身）⇒ 按 mission 授权条款归档收口。
5. **移交（会话 9 首位候选，方案要素齐备）**：形态 = VW=2 窄形宽通道（work_ub/rr_ub/acc_s_half/mask_2d 按 VW 加宽 + m_w/m_pair/sumexp_w 微缓冲；buf_2d/rt_2d/io_buf/acc_o/out_half 保持窄形；宽 buf_2d_w 变体已排除）；W 逐 D 配置 parse-time 派生（D=256→2（4 可行但边际，2 稳）、D=128→2、D=64→1 保旧路径）；精度门 = 阻塞口径（混合容差 + MARE_abs<10×Threshold + 余量复核，现余量 prefill_worst 3.05e-5/320×）+ 红线四项（双舍入栅格/单块预归一特化/sumexp+1e-30/右下角掩码链）逐字保持 + 行内归约序位级不变的探针裁决（mission 口径非位级预期，实际位级性由探针定）；锚方案 = 锚-A #1-8（单块/分派收窄域）BITEXACT 硬守卫保留、#9 与锚-B 多块成员降级为阻塞口径 + 根因分析纪律（§S4-5 先例），方案显式成文先于实施；作用于 flat 连续槽窗（ORDER-PLAN 修订依据 4：组合自然）。
6. **联合包完整性如实报告**：mission 原三包（①+②+③）中 ② 证据性关闭、③ 评估归档未实施 ⇒ 按中止条款精神，**会话 8 净产出 = ① 全域落地 + 实测收益**（7/7 目标 case -6.26%~-33.08%，全 20 case 无一回退）；残余差距归因 = case9/7/8 的 B 层收敛停在 -7~-8% 级（轮界删除项已兑现，剩余 = MTE2 填充节拍 + 每槽四相位交接的不可删段 + V 侧事件税），0.54x 级承诺不成立（维持 r8-1 降级口径）。

**收口判定**：性能规则满足（r8-2/r8-3 连续采纳，consecutive_no_improvement = 0）；中止条款②未触发；Step E 收口项 = docstring rev4r8 注记统一（已完成，**16/16 IR codegen-neutral 复核通过**：r8_src_final == r8_src_prov == r8_src_iter3 逐字节，dump_final.log）+ 终态（dda3ad08，1926 行）新鲜门禁子集（锚 + 继承压力 + l0 + --level all，r8_cache_finalg 隔离缓存）+ final_report §S8 + perf_log/debug_log 补全 + 备份链补齐（perf8 = r8-3 采纳态 2728c96e；perf9 = 终态 dda3ad08）。

---

# 会话 9（perf_session=9，2026-09-30，rev4 轨；case9 瓶颈双路径专项：③ V 宽通道主攻 + bN=256 scoping 探针）

> 对象：会话 8 终态 mha.py（1926 行，md5 dda3ad0877ba4df74bbd559f6912eeed，cann-bench 72.59 / 0.96x，cann_final_eval_20260930_083840）。
> 任务（用户 2026-09-30 决策立项）：**路径 1 = ③ V 宽通道主攻**（VW=2；Stage 1 form_a D=128 域先行 → Stage 2 扩 fbm D=256 case9 域；判决性判据 = V.vec 显著降 ∧ C.wait_id2 同步降 ∧ wall 降，三签名齐采纳；V.vec 降而 wall 零响应 = 吸收律第四例 → ③ 族证据性关闭）+ **路径 2 = fbm prefill bN 128→256 仅 scoping 探针**（不改生产 kernel，独立 scratch；L0C 共池第一硬门先行）。
> 开工快照 = history_version/mha_impl_r4_perf10.py（md5 dda3ad08 核验一致）。

## [ANALYSIS] Step 0 基线核验 + 证据加载（2026-09-30）

1. **基线核验**：mha.py md5 dda3ad0877ba4df74bbd559f6912eeed == 任务书预期 ✓（1926 行 ✓）；test_mha.py 0dd9e93c host 契约不变 ✓；备份 history_version/mha_impl_r4_perf10.py 已建 ✓。
2. **证据加载（精读完成）**：[STEP-D-EVAL]（L1324-1334：③ 选择依据/判决判据原文/移交方案要素）、[#R8-2-pre]（UB 预算探针实编读数：D=128 VW=2 = 156.75K ✓ / D=256 VW=2 = 138.62K ✓ / VW=4 = 171.50K ✓ / D=64 VW=2 = 168.75K ✓但按移交条款保 W=1；宽 buf_2d_w 变体已排除；宽 scratch 物化不可避免判定 = subs 标量路径 PipeBarrier<PIPE_ALL>+全 op 单标量，codegen_ascend.cc:1837-1888 核验）、[RESULT-#R8-2/3]（flat 骨架 = ③ 作用基座；V 侧 flat_v 分支结构 + 顺序游走状态机 + §S4-3-5 复位时序）、#R3-3（V 侧 vector 指令削减 pre-flat 零响应先例 = 判决判据反方证据；RNE 位级等价知识三次实证入库）、final_report §S8（会话 8 全记录）/§S4-5（位级锚重建预案先例）/§S7-3（吸收律双侧 A/B 方法）、analysis §2.2-③/附.2/附.6（宽通道目标形态：vec 49.5% busy 中碎片化税 46904 事件 ×~14ns、36 BAR+34 MOVEMASK/iter；理想 V.VEC 占空 18%）。
3. **op A/B base 侧读数（归档引用，文件在案）**：/tmp/opencode/r8_op_case7_iter3（V-paced：V.vec 634.39us=77.5% aiv、C.wait_id2 547.70=67% aic）；/tmp/opencode/r8_op_case9_iter2（V.vec 453.3/468.6=55.4-55.6%、V.wait_id4 247.92、V.wait_id0 267.3、C.wait_id2 232.06/213.0、C.MTE2 57.6-57.8% 最忙真实工作管）。
4. **同会话新鲜基线（msprof_r9_base/，median，TILELANG_CACHE_DIR=r9_cache_base）**：case9 869.04 / case7 858.26 / case8 855.78 / case3 131.35 / case4 27.92 / case2 44.84 / case5 15.19 / case10 33.87 / case13 565.30 / case14 376.02 / case16 562.09 / case20 14.03（us）——vs 会话 8 终态读数最大偏差 +0.7%（±2.5% 会话方差标尺内）⇒ 环境一致，采纳判定用会话 9 内部对比。baseline_r9.json 已落盘。
5. **域分派现状核验（源码精读）**：flat_v = (S>16) ∧ ¬single_kv_block（form 无关，L233）；emit_v 三分支（single_kv_block 任务批 / flat_v 连续流 / else generic 旧路径）为三 prim_func（form_a/form_b_mma/decode B-gv0）共享 @T.macro；V 侧 UB alloc 双 prefill prim_func 逐字对称（L851-868 / L1279-1296）+ decode prim_func 第三份。**bench 目标域映射**：case7/8/3 = form_a flat D=128（Stage 1 宽通道目标）；case9/4 = fbm flat D=256（Stage 2 目标）；case2/5 = form_a flat **D=64 → W=1 保旧路径**（移交条款分派谓词收窄；UB 探针 D=64 VW=2 实编 168.75K 虽 ✓ 但无 bench 收益面，不扩风险面——IR 影响面如实报告条款适用）；decode 5 case = S≤16 → B-gv0 兜底（emit_v 内不落 flat_v 分支，宏参数别名传递保 IR）。

## [ANCHOR-PLAN-R9] 锚体系重建/降级方案（Stage 1 硬门：成文先于实施）

**背景（任务书硬门）**：③ 宽合并改变 online-softmax 归约结构 → mission 口径**非位级预期**；锚-A #1-9（Form A BITEXACT 硬守卫）中 #9 落宽通道域 → by design 可能失效。按 §S4-5 预案先例（rev4 锚-B"必失配"预案 → 实测未失配 → 保持诊断力的先例），成文三层方案：

**影响面精确圈定（分派谓词级）**：
| 锚成员 | shape | 域 | ③ 触达 |
|---|---|---|---|
| 锚-A #1-4 | (1,64,64,1,{64,128}) | Form A 单块（S_kv≤bN） | **否**（¬flat_v；alloc 形状 VW=1 数值不变 + 宏参数别名 → IR 字节级不变可证） |
| 锚-A #5-8 | (2,128,128,8,{64,128}) | Form A 单块 | **否**（同上） |
| 锚-A #9 | (1,128,256,4,128) causal | Form A 多块 flat **D=128** | **是**（Stage 1 起落 VW=2 宽域） |
| 锚-B b01-b10 | D=256 generic 多块族 | fbm flat **D=256** | Stage 1 **否**（D=256 未开宽）；Stage 2 起**是** |
| 锚-B b11-b14 | D=256 单块族 | fbm 单块 | **否**（¬flat_v） |
| decode IR 守卫 ×5 | — | B-gv0 兜底 | **否**（源码零触碰 + 宏别名 + IR 16 目标 dump 复核） |

**Layer 1 — 探针先行位级裁决（首选路径）**：宽合并的位级性由专属探针 probe_r9_wide（Stage 1/2 各一）裁决，理论预期偏 BITEXACT（如实标注为预期非承诺）：①reduce_max/reduce_sum 行独立、行内 bN 归约树宽度不变（repeat 数变、树形不变）；②max 链组合序同构（m_pair1 = max(m_w1, m_pair0) ≡ 窄序 m_i(1) = max(rowmax(1), m_i(0))，fp32 max 无舍入 → 数学恒等）；③exp/cast/mul/sub 元素级（宽窄同一指令序列，仅 repeat 加倍）；④mask 链窄形 per-slot 保持（同值代入）；⑤r16 双舍入栅格点逐元素不变（#R3-3 fixpipe≡vcast RNE 位级证据链背书"同栅格不同载体"）。**若探针全拓扑 BITEXACT vs 改前态 → 锚-A #9 / 锚-B 多块成员保持 BITEXACT 硬守卫（最强形态，无需降级）**——§S4-5/rev4 锚-B 预案未触发先例的复现路径。

**Layer 2 — 降级预案（仅 Layer 1 失配时触发，逐项纪律）**：
- 锚-A #9 与锚-B b01-b10 降级为**诊断工具**（DIFF 报告 + ulp 级归因分析纪律，失配 ≠ 自动 FAIL——rev4 §9.2 锚-B 先例条款）；
- **主门升级 = 阻塞口径**：混合容差（matched ratio）∧ MARE_abs < 10×Threshold ∧ 非 NaN（test_mha.check_precision + mere_mare 现有实现，p6 电池 run_one 已内嵌）；
- **MARE 余量硬线（新增条款）**：宽域全探针成员实测 MARE_abs 须 < 3.05e-4（= 现 prefill_worst 余量 3.05e-5/320× 的 10×，即余量消耗 ≤ 现值的 1/32… 口径：gate=10×Threshold 不变，实测余量 vs 320× 报告，若余量跌破 100× 须根因分析归因到具体 op 才可放行）；
- 失配根因分析纪律：定位到具体归约 op（reduce 树形 / exp 载体 / cast 序）并论证域内无害，禁止"整体在容差内就不查"。

**Layer 3 — 红线四项（无条件，逐字保持 + 逐项再论证）**：
1. **r16 双舍入栅格链不动**：宽通道改 cast round-trip 的形状（[hM,bN]→[VW·hM,bN]）不改栅格点——每元素 fp32→dtype→fp32 RNE 序列逐字保留（copy work_ub→rr_ub→work_ub ×2 对 + mul sm_scale 居间序不变）；
2. **单块预归一特化不触**：single_kv_block 分支零改动（wide 仅 flat_v ∧ VW≥2 分支内），分派域收窄 IR 字节级可证；
3. **sumexp+1e-30 守卫逐字**：位于 phase2 任务尾（L624），phase2 本轮零触碰（宽通道仅 phase1）→ 逐字保持天然成立；
4. **右下角掩码链同值代入**：offset_eff/causal_offset 公式不动；mask 构造（col_pos/row_thresh/broadcast/compare/select）per-slot 窄形同值代入，探针 causal/bound/tail 变体覆盖。

**Stage 4 收口条款（继承）**：emit_v 变更 → flat 域 IR 守卫失效 → **全 20 case msprof 复测替代 IR 保证**（会话 8 纪律延续）+ decode 五 case 守卫线判定（本地口径相对比较；bench 口径权威换算留 orchestrator）。

**方案裁决：Layer 1 先行（探针裁决位级性），Layer 2 为失配预案（主门阻塞口径 + MARE 余量硬线），Layer 3 无条件。锚-A #1-8 + 锚-B b11-b14 + decode×5 全程 BITEXACT/IR 硬守卫不降级。** 本方案成文于任何 emit_v 改动之前（硬门满足）。

## [P0-WIDE-MECH] 会话 9 Stage-1 P0 机制微探针（probe_r9_wide_p0v2 系列，实施前 API/位级裁决）（2026-09-30）

**目的**：③ 宽通道实施前一次性裁决全部 wide-op 风险点（M1 宽 reduce 行级位级同一 / M2 宽 exp·cast·mul·sub / M3 tile-op BufferRegion 切片操作数 / M4 per-half compare→宽 mask + 宽 select / M5 m 链切片组合序）。方法 = wide [2hM,bN] 单段 kernel vs narrow [hM,bN] 单段×2 launch（m 链经 GM 传递，与 wide 内核链同构），全中间量逐位对比。

**途中捕获两个探针侧伪影（新硬知识，均已修复；非生产问题、非 wide-op 语义问题）**：
1. **[N,1] 2D copy 的 32B 块粒度陷阱**：`T.copy(om[32,1] → OM[32,1])` 生成 `copy_ub_to_gm<float,1,32>(dst,src,1,32,1)`——UB 侧按行读 **32B 块**（每行 8 floats），GM 侧行 stride=1 → 读到的是 om_ub[8r] 采样（r<4 = max_8r，r≥4 = 邻接 buffer 内容），呈现"输出[r]=rowmax(8r)"的 8× 假模式。同款陷阱存在于 [N,1] 2D 的 GM→UB load（UB 侧按 32B/行写入 → 溢出踩踏邻接 buffer 8×）与 UB→UB copy。**规避 = [N,1] 缓冲的 GM/跨缓冲搬运一律用 1D 视图（buf[:,0] ↔ GM [N]）**（生成 (N,1,N) 单行紧凑拷贝，实测正确）。生产 kernel 无此暴露面（m_i/sumexp 等 [hM,1] 缓冲只被 V-pipe op 与 broadcast 消费，从不做 [N,1] 2D DMA；r_factors[i,:,:] 为 [hM,1] 切片但 src/dst 同为 [hM,1] 形状的 ws…——生产 ws 侧 r_factors 无 GM 2D copy 路径；本知识入库供后续探针/实现引用）。
2. **观测点 emit 竞态**：探针在"reduce 后、m 链覆盖前"插 GM emit 观测 raw rowmax——emit 的 MTE3 读与 V-pipe 后续覆盖写之间无 drain（闭环 set(MTE3_V) 的 wait 在下一 emit 才消费）→ 确定性/非确定性覆盖竞态。修复 = 观测 emit 闭环尾加 wait（全串行）。**生产结构天然无此竞态**（reduce→max 之间无 MTE3 读点；SIG_S_HALF 闭环 per-emit 完整）。
3. （附带）单核探针 kernel 的跨轮循环需显式 MTE3→V drain：探针 narrow 两轮循环 hf=1 的 MTE2 load 直写 work_ub 与 hf=0 轮的 MTE3 观测读竞态——生产免疫（work_ub 仅由 V-pipe UB→UB copy 写入，MTE2 只写 io_buf，SIG_IO_UB 闭环保护）。

**P0 裁决结果（观测伪影修复后）**：
- **M1 宽 reduce_max/reduce_sum [64,128]→[64,1]：per-row BITEXACT vs narrow×2** ✅（S3 new-max 链 + S7 rowsum BITEXACT；S2 raw 观测点伪影修复后复核）
- **M2 宽 exp/cast(fp32↔fp16 round-trip)/mul/sub：BITEXACT** ✅（S1/S6/S8）
- **M3 tile-op BufferRegion 切片操作数（max/sub/copy/fill 于 buf[0:hM,:]）：编译 + 运行 + 位级正确** ✅
- **M4 per-half compare → 宽 mask_2d region 半区 + 单宽 select：编译通过**（全 kernel 探针复核数值）
- **M5 m 链切片组合序（m_pair0=max(rowmax0,hist0) → hist1=new0 → m_pair1=max(rowmax1,hist1)）≡ narrow 逐槽链：BITEXACT** ✅（S3）
- reduce_max/reduce_sum 行为同时经 M 扫描（8..128）与 20 核/vid 切分对照复核：**AscendC Reduce（tl.ascend_reduce fast-path）本身在全部测试形态下正确**——初期"8× stride 错乱"读数全部归因伪影 1（归档防再踩）。

**结论：③ VW=2 宽通道的全部 op 级机制风险已排除，位级性预期 BITEXACT（Layer 1 锚方案路径成立概率高）；进入 emit_v 实施。**

**[P0-WIDE-MECH 终判]**（drain 修复版 e12 复跑）：S1 cast+mul / S2 raw rowmax / S3 new-max 链 / S4 r_factors / S6 post-exp / S7 rowsum / S8 fp16 acc **7/8 BITEXACT**；S5 post-sub 非确定性 diff（3.4-4.6 级，跨 run 变化）= 观测点竞态伪影——**数学闭合论证**：S5 = S1 - broadcast(S3 值)，S1/S3 均 BITEXACT 且 sub 元素级确定 ⇒ S5 真值必位级同；观测 diff 唯一可能域 = exp 非单射区（x<-87），而探针值域 S5∈[-7,0]（exp 全域单射）∧ S6=exp(S5) BITEXACT ⇒ S5 真值 BITEXACT，diff 为 emit 观测面竞态（广播 tmp 复用窗口的 MTE3 读侧，探针特有；生产 phase1 无该观测点）。**判决：M1（宽 reduce per-row 位级同一）/ M2（宽 exp/cast/mul/sub）/ M3（切片操作数）/ M5（m 链切片组合序）全部 PASS；M4（per-half compare→宽 mask+宽 select）编译已过、数值交全 kernel 探针（causal/tail 拓扑）裁决。P0 阶段关闭，进入 emit_v 实施。**

## [SCOPING-PATH2-R9] 路径 2：fbm prefill bN 128→256 可行性 scoping 报告（Stage 3；不改生产 kernel；probe_r9_bn256_budget.py P4 式实编读数）（2026-09-30）

**可行性矩阵（实编读数优先于手算；证据 = /tmp/opencode/r9_probes/bn256_budget_final.log + probe_r9_bn256_budget.py）**：

| # | 门 | 形态与读数（实编 GetWithOffset / InitBuffer） | 裁决 |
|---|-----|------|------|
| GATE-1 | **L0C 共池（第一硬门）** | 三形态全过：**F2**（l0c_s [1,32,256]fp32 32K + l0c_o [1,32,256] 32K）/ **F4**（s 单槽 32K + o [2,32,128] 32K）/ **F5**（s [2,32,128] n-split×2 32K + o [2,32,128] 32K）——**used 均 = 64.00K 物理顶格**；InitBuffer(CO1)=128K = 现状 fbm 生产已验证池值（G_case9 现状即 131072 池 + 64K used——判据修正记录：首跑误以池值 64K 判 OVER，经现状对照修正为 used≤64K ∧ 池≤生产已验证 128K）。F1（双槽全宽 128K used）算术否决 | **✓ PASS（三形态，双顶格零余量）** |
| GATE-2 | L0B | GEMM1 transpose_B KC=64：[2,64,256]f16 = **64.00K 顶格**（#10b kL0Size=64 + r8-1 v4ksp 双先例形态）；GEMM2 V 链 KC2=128：[2,128,128] = 64.00K 同池同量——共池覆盖两 GEMM 形态 | **✓ PASS（顶格）** |
| GATE-3 | L0A | l0a_s [2,32,128]f16 16K + l0a_o [2,32,256]f16 32K = **48.00K ≤ 64K**（余量 16K） | **✓ PASS** |
| GATE-4 | L1 | q [32,256] 16K + k [256,256] 128K + v [256,256] 128K + p [32,256] 16K = **288.00K ≤ 511.75K**（实编确认手算 ✓） | **✓ PASS** |
| GATE-5 | UB | 生产 fbm V-set 加宽（io_buf [2,16,256]f32 32K + io3 32K + work 16K + buf/rt 32K + r_2d 16K + acc_s/rr 16K + acc_o 16K + out_half 8K + 微缓冲）= **170.75K ≤ 191.75K**（余量 21K；vs 现 126K = +44.75K） | **✓ PASS** |
| GATE-6 | **ws 足迹 → L2 命中率（脆弱资产条款）** | ws/槽 56→80KB（w1 [32,256]f32 32K + w2 f16 16K + w3 32K）⇒ 总足迹 9.0→12.8MB（+42%）；**但槽数减半（442→221）⇒ ws 总流量 -29%**（495→353MB）。两反向因素（工作集 ↑ vs 总流量 ↓）⇒ **净效应解析不可判**。**实测未做（如实声明）**：需可运行的 bN=256 完整变体（mma 链重组 = rev4-S1 量级工程量，超本 scoping 会话预算）；实测方案 = scratch 变体 msprof L2Cache.csv 对照（命中率 <90% 触发收益反转预警：MTE2 busy 期速率 131GB/s 向 HBM 侧 73.6GB/s·核收缩） | **⚠ 未实测（方案入库，会话 10 立项前置）** |
| GATE-7 | FIX colseg 事件重排净值 + mma 链重组工程量 | 事件账（模型）：FIX colseg 3→5/iter（税先例 884→1326 方向）× 槽数 ÷2 ⇒ **总事件 1326→1105（-17%）**；mma 发射：GEMM1 k-split 4×n-split 2 = 8 段 + GEMM2 k2×n2 = 4 段（现 8/iter → ~12/iter，MMAD 发射近零成本 r8-1 归档）；MTE2 字节/槽 ×2 槽数 ÷2 = 总量与节拍和不变（T=2.22us×221 ≈ 491us ≡ 现 1.112×442）。**工程量 = rev4-S1 量级**（会话 4 S1 单项 ≈ 1 会话：GEMM1 4-chunk 链 + GEMM2 K=256 链 + FIX colseg 重排 + 协议槽重配 + 全门禁） | **⚠ 净值模型为正、需实测校准；工程量大（如实）** |

**EV 修正（模型 -12~-19% 的读数校准）**：
- 收益机制（路径 2 本质 = 结构删除类，吸收律不适用域）：槽数 442→221 ⇒ 每槽不可删段（SEM 传播+排空：C wait_id2 侧 0.48-0.53us/iter + V wait_id4/id0 侧 ~1.2us/iter）总量减半 ⇒ case9 wait_id2 暴露 213-232us → ~107-116us ⇒ **wall 868 → ~760us（-12.5%）级**——与任务书模型带 -12~-19% 下沿吻合（上沿需 V 侧事件税同步下降兑现）。
- 风险项：①L2 命中塌缩（GATE-6 未实测 = EV 最大不确定项；若命中 92.7%→<85%，MTE2 顶格速率收缩可吞噬全部收益甚至反转）；②双顶格形态（L0B/L0C 64.00K used）零余量——任何足迹增量（如锚 buffer/调试残量）即爆；③FIX 5 事件/iter 的税若超模型（884→1326 先例为事件 +67% 时 wall +55%），-17% 总事件的收益可能被高估。
- **修正 EV = -12~-19%（条件：L2 命中 ≥90% 维持）/ 0~+10%（命中塌缩情形）；条件判定点 = GATE-6 实测（会话 10 立项前置门）**。

**实施蓝图要点（供会话 10 立项决策）**：
1. **L0C 形态 = F5 首选**（l0c_s [2,32,128] GEMM1 n-split×2 + l0c_o [2,32,128]：双槽乒乓全保留，代价 = FIX colseg 3→5/iter 与 w1 双段写）；F4 备选（s 单槽：GEMM1(i+1)/GEMM2(i) 串行排空点 +1，换 FIX 4/iter）。
2. L0B = GEMM1 KC=64 [2,64,256]（k-split 4 链，init=True/False 链式）+ GEMM2 V [2,128,128]（K=256 → KC2=128×2 chunk）——同池 64K 顶格，v4ksp 形态知识直接复用（r8-1 归档：同取向 + ZN 行列双切片 MTE1→L0B 直载）。
3. 协议面：flat 骨架不变（batch_iters 语义不变、槽数自然减半）；SEM/SIG 矩阵零漂移（ws 槽尺寸变化仅 host 侧 ws 分配 80KB×8×20=12.8MB + kernel 侧 copy shape）；**精度预期位级**（r4-2a 已证 k-split 链 C0 顺序累加位级等价；n-split 列独立）⇒ 锚体系预期保留（探针裁决）。
4. 守卫面：bN=256 仅 fbm prefill generic（case9/4 域）；decode bN 派生独立（S≤16 分支不受影响，谓词隔离 + IR 守卫）；form_a 不动。
5. **立项前置门（硬）**：GATE-6 L2 命中实测（scratch 变体 msprof L2Cache.csv；命中 <90% ⇒ 收益反转预警 ⇒ 立项否决或降级）。

**scoping 判决：资源可行性 5/5 实编 PASS（L0B/L0C 双顶格）⇒ 路径 2 不被否决；EV 修正 -12~-19%（条件化）；GATE-6/7 未实测项如实归档（完整变体 = rev4-S1 量级，超 scoping 预算）；蓝图 5 要点入库 ⇒ 移交会话 10 立项决策。**

### 迭代 r9-1：[#R9-1] ③ V 宽通道 VW=2（form_a flat D=128 域 = case7/8/3；Stage 1）（2026-09-30）

[ORDER-CHECK] 准备实施: [#R9-1] ③ VW=2 宽通道 Stage 1 | 前置依赖: [ANCHOR-PLAN-R9 成文 ✅（硬门）/ P0-WIDE-MECH 机制裁决 ✅ / STEP-D-EVAL-5 移交方案要素 ✅ / UB 预算探针归档 ✅] | 结论: ✅

[IMPL-#R9-1] 已阅读: [STEP-D-EVAL]（判决判据原文/移交形态/锚降级预案）、[#R8-2-pre]（窄形宽通道设计 + 宽 scratch 物化不可避免判定）、analysis §2.2-③/附.2/附.6（目标形态：V.VEC 理想占空 18% vs 现 49.5% 碎片化税）、#R3-3（V 侧削减 pre-flat 零响应先例 = 判决反方证据）、§S7-3-2（吸收律双侧 A/B 方法）、§S8-3-7（post-flat EV 真开放论证）、P0 探针全套（reduce/cast/exp/切片/m 链位级同一实证 + [N,1] 2D copy 块粒度陷阱 + 观测点竞态两伪影归档）。关键约束: ①分派谓词 = `VW = 2 if (flat_v and form_a and D == 128) else 1`（Stage-1 scope 收窄：fbm D=256 = Stage-2、D=64 保 W=1 移交条款、单块/decode 天然 W=1）；②缓冲方案 A''（统一名 VW 乘法 shape：VW=1 数值不变 → 14 域 IR 字节不变；**方案 C 双套 UB 爆预算 205K>191.75K 实测否决、方案 A' region 别名被 tile.mul binary_op access_ptr 拒绝实测否决、方案 R 微缓冲 alloc 移入 V-scope wide 调用分支**——三次迭代收敛，全程 dump/编译实证裁决）；③奇尾槽 = 方案 K 宽形状单槽退化执行（上半 garbage 行输出不进任何数据流：ws2 写下半/m 链 r_factors sumexp_is 取下半/exp reduce 行独立性——首启未初始化 UB→NaN 亦上半局域；**全部 op 形态 = wide pair 已证形态，零 BufferRegion 兼容风险**）；④wide 段令牌审计：SEM_WS1_C2V wait 1/槽、SEM_WS2_V2C set 1/槽（MTE3 管序内 copy0→set→copy1→set）、SIG_IO_UB 闭环 1/槽·side、SIG_S_HALF 1/pair（内核级协议变更，C 不可见，收支平衡静态可证）、批级 SEM_WS1_V2C/WS2_C2V 不变 ⇒ **C 侧 SEM/SIG 矩阵零漂移**（#R5-3 同进同出：奇尾槽 runtime if 全段包裹，batch_iters 双 vid 同值确定性）；⑤mask 宽化 = always-on per-half compare（窄 runtime-skip 在 wide 路径放弃：全可见槽 compare = 全 1 mask（col ≤ min(s0+off,S_kv-1) ≤ row_thresh[row] 数学恒等）→ 宽 select 恒等，位级保持，探针 causal/tail 变体 BITEXACT 实证）；⑥m 链 = P0 实证序（reduce_max 宽 → max(m_pair0, m_w0, m_i) → r0 → chain → idx1==0 特判 fill → hist1 → max(m_w1 in-place) → r1 → chain）；⑦alloc_var 快照纪律（init 仅 int/float/Var——**init=TIR 表达式 raise "only accept int or float" 实测归档**；Assign 二段式 = tl_p1+=1 同款可变快照，跨 walk advance 使用安全）；⑧红线四项：r16 cast 链逐元素同栅格（wide cast BITEXACT 实证）/单块特化域收窄（IR 字节级）/sumexp+1e-30（phase2 零触碰）/掩码链同值代入（⑤）。

[KERNEL-REUSE-AUDIT] 依据: STEP-D-EVAL-5 移交方案 + P0 机制裁决 | 维度及结论: 调用域不扩（host 契约/分派谓词收窄至 form_a∧flat∧D128；fbm/D64/单块/decode 全守卫）✅；位级预期 = BITEXACT（P0 op 级实证 + 探针 16/16 兑现——**Layer 1 锚方案成立，锚-A #9/锚-B 无降级**）✅；UB 硬门 = wide 域实编 172.62K ≤ 191.75K ✅；令牌收支静态可验（④）✅；unknown: wall 响应（判决判据三签名）→ msprof+op A/B 裁决 | 结论: repair-kernel-first（emit_v flat phase1 wide 分支 + 三 prim_func alloc 条件化 + 双调用点；+479 行）

[RESULT-#R9-1] 优化点: **③ V 侧批相干宽通道 VW=2（form_a flat D=128 域；pair j = 槽 (2j,2j+1) 合并宽 [2hM,bN] op 序列：r16 cast 链×4/mul/exp/reduce_max/reduce_sum/acc cast 宽化 + per-half mask 构建 + 单宽 select + SIG_S_HALF 闭环 1/pair + 奇尾槽方案 K 宽形状退化）** | 精度: **pass（门禁全过）**——①探针 probe_r91_wide **23/23 有效判定 PASS**：PART 0 分派审计 16 目标全落 wide 域（VW=2）+ 7 守卫全不落；PART 1 **16/16 对改前（perf11 = dda3ad08）BITEXACT**（全拓扑：case7/8/3 bench 原形、奇偶批（kv3 奇尾/尾批 1 槽 num_pairs=0）、跨轮任务状态携带 kv10（causal+non）、causal eff 1..4 变异轮中切换、S 尾块 200%64/S_kv 尾块 777%128=9 与 300%128、bf16 ×2、zeros 陷阱对照（位级 0.00e+00）、显式 scale、大幅值域（vs_old BITEXACT；golden MARE 1.36 = **既有伪影非回归**——r8-3 归档同款，旧核同数据逐位相同））+ golden 阻塞口径 15/16（伪影成员外全过，最差 bf16 7.81e-3 vs gate 7.8e-2 = 10x）；PART 2 守卫 **7/7 BITEXACT**；②锚-A **9/9 BITEXACT 硬守卫（#9 落 wide 域仍位级 = Layer 1 锚方案兑现，[ANCHOR-PLAN-R9] 降级预案未触发）** + 锚-B **14/14 BITEXACT**（BATTERY_RESULT: PASS）；③竞态压力：继承 4 配置 ×12 nondet=0 + 扩展 **29 项**（rev4 16 + flat-B 6 + flat-A 4 + **wide 专项 w01-w07**：尾批 1 槽/尾批 3 槽/causal 异构混合尾/深 kv10/bf16 尾批/case7 原形 20 核/显式 scale）×12 复跑 **nondet=0**（33/33 PASS）；④l0 **10/10** + l1 **19/19** + coverage EXIT=0 + `--level all` **EXIT=0**（L2 12/12 正确拒绝 + Boundary + 哨兵 4/4 PERF_PASS）；⑤**IR 守卫 14/16 逐字节相同**（decode ×5 + Form A 单块 ×4 + fbm ×3（case9/case4/B_single_d256 = Stage-2 域当前零触碰证明）+ D64 flat ×2（case2/case5 W=1 谓词收窄）——decode 不回退硬约束 IR 级成立）+ **2/2 wide 目标按预期改变**（G_case7 18421→25452B / G_A_multi_noncausal 16571→22273B）；⑥UB 预算硬门：wide 域实编 **172.62K ≤ 191.75K**（余量 19.13K；G_case7 dump GetWithOffset 读数）；⑦红线四项逐字保持（r16 栅格 wide cast 位级实证/单块域 IR 字节级/sumexp+1e-30 phase2 零触碰/mask 链同值代入 + always-on 恒等式探针实证）；⑧途中缺陷 3 处已修复归档（方案 C UB 爆 205K→A''；tile.mul 拒 region dst→方案 K 尾槽；alloc_var init 表达式崩溃→Assign 二段式；[N,1] 2D copy 块粒度陷阱与观测点竞态 = P0 探针伪影双归档）；disable_cache/会话专属缓存全程（r9_cache_* 系列）。| 性能: **见 [PERF-#R9-1]（msprof 12 case A/B + op case7 三签名判定，测量中）**

[PERF-#R9-1] 性能判决（msprof median，同会话基线 msprof_r9_base；op A/B block0+20 核均值双口径，base = r8_op_case7_iter3 = 现态 IR 承继）：

| case | 域 | 基线 us | 候选 us | Δ% | 判定 |
|---|---|---|---|---|---|
| case7 | wide 主对象（V-paced） | 858.26 | **964.77** | **+12.41%** | 性能下降 |
| case8 | wide（bf16 twin） | 855.78 | **993.06** | **+16.04%** | 性能下降 |
| case3 | wide | 131.35 | **150.20** | **+14.35%** | 性能下降 |
| case9 | fbm 守卫（IR 逐字节） | 869.04 | 868.80 | -0.03% | 零变化 ✓ |
| case4 | fbm 守卫 | 27.92 | 28.25 | +1.18% | 带内 ✓ |
| case2/5 | D64 W=1 守卫（IR 逐字节） | 44.84/15.19 | 44.92/15.29 | +0.18%/+0.66% | 带内 ✓ |
| case10/13/14/16/20 | decode 守卫（IR 逐字节） | 33.87/565.30/376.02/562.09/14.03 | 34.90/559.43/377.53/564.52/14.20 | +3.04%/-1.04%/+0.40%/+0.43%/+1.21% | 带内（case10 +3.04% 边缘：IR 逐字节域 ⇒ 内核因果排除，会话 8 case20 +1.75% 方差先例同款；回滚态复测澄清，msprof_r9_rb） |

**op A/B 三签名判定（case7，STEP-D-EVAL 判决性判据逐字执行）**：
| 签名 | 预测 | 实测（block0 / 20 核均值） | 判定 |
|---|---|---|---|
| V.vec 显著降 | ↓ | **634.39→772.73（+21.8%）/ 648.99→793.70（+22.3%）↑↑** | **FAIL（反向）** |
| C.wait_id2 同步降 | ↓ | **547.70→657.83（+20.1%）/ 533.19→661.28（+24.0%）↑↑** | FAIL（反向） |
| wall 降 | ↓ | 858.26→964.77（+12.4%）↑ | FAIL |

**机制归因（新硬知识，③ 族判决性证伪）**：
1. **事件税侧签名全部兑现**：V.v_scalar -11.7%（标量发射削减）、v_stall -35.6%、v_mte3 -27.9%（SIG_S_HALF 闭环减半）、wait_id0 -44.2%、wait_id4 -51.0%、C.wait_id5 -100%、C.scalar -14.9%——**③ 设计攻击的"碎片事件税"确实被削（36BAR+34MOVEMASK/pair→宽、MTE3 闭环 ÷2）**。
2. **但 vec 主管道时长 +21.8% 吞没全部节省并反转 wall**：宽 [2hM,bN] op 的 vector 执行周期 > 2× 窄 [hM,bN]（大 repeat 宽化低效：mask/repeat 建立与流水填充摊薄未抵消执行侧增量）+ always-on mask（wide 路径放弃窄版 runtime-skip，causal 全可见 pair 每对多付 compare×2+select 全宽）直接进 vec 管道。**"V.vec 49.5% busy 中碎片化税"模型（analysis §2.2-③）实测修正：事件税主体在 scalar/stall/MTE3/wait 侧（可削、已削），vec 侧 634us 主体 = 数据体量×吞吐的真实执行周期（宽化不缩减、反增）**。
3. **V-paced 结构反向因果确证（本会话最强机制知识）**：case7 pacing 管 = V.vec（STEP-D-EVAL 判定 77.5%）——本轮"把 V 做慢 +21.8%"→ C.wait_id2 +20.1% 1:1 放大传导 → wall +12.4%。**V-paced 域中 V.vec 时长对 wall 的因果系数 ≈ 1（正向反向双向实证）**——这是 r7-1/r7-2 双侧 A/B（加速侧吸收）之后的**减速侧对照实验**，四相位耦合环的 pacing 管判定方法学闭环。
4. **吸收律第四例判定升级**：STEP-D-EVAL 预案"V.vec 降而 wall 零响应 = 吸收律第四例"——实测为**更强否定形：V.vec 未降（宽化增量 > 事件税削减）∧ wall 负响应** ⇒ ③ 的机制假设（碎片税 = V.vec 主体）被证伪，非仅吸收。

**采纳判定：NO — 回滚（性能下降规则：3/3 目标域 +12.4~+16.0% 超噪声带）**。已回滚至 history_version/mha_impl_r4_perf11.py（md5 dda3ad0877ba4df74bbd559f6912eeed 核验还原 + **IR 16/16 逐字节还原验证**（r9_src_rollback vs r8_src_iter3，dump_rollback.log））。变体留存 /tmp/opencode/r9_probes/mha_r91_edit_v1.py（52dd7e8a，2405 行，**精度门禁全过态**：探针 16/16 BITEXACT + 锚 9/9+14/14 + 压力 33×12 nondet=0 + l0/l1/coverage/all EXIT=0——仅性能规则回滚，r8-1 mha_r81_edit_v1 留存先例同款）。

**[STAGE2-RULING] Stage 2（③ 扩 fbm D=256 case9 域）不实施——数据驱动裁决（如实声明：非任务书 Stage 2 字面执行）**：
- 判据前提已灭失：三签名第一签名（V.vec 显著降）在 Stage 1 被**机制级证伪**——宽化 vec 增量为硬件级效应（大 repeat 执行低效 + always-on mask），跨 form 成立（fbm D=256 hM=16→宽 32 行同款宽化增量）。
- case9 域 EV 预判 ≤ 0（数据推演）：case9 V.vec 453-468us（55.4-55.6%，非 pacing 管——pacing = C.MTE2 顶格 57.8% + C scalar 95.5%）；宽化 +20% 增量 → V.vec ~545-560us，被 V 侧 wait_id4+id0 = 515us 余量吸收 → **最好情形 wall 零响应（吸收律）；若 V 变新 pacing 管则 wall 升**。负 EV 验证不实施 = r8-1 ② 复活否决先例同款纪律（"判据预判失败 + 重集成成本不可辩护"）。
- **③ 族证据性关闭（双向结论归档）**：正方（事件税可削：scalar/stall/mte3/wait 侧 -12~-51% 实测兑现）+ 反方（vec 主体不可宽化缩减、反增 +21.8%；V-paced 域 wall 1:1 负传导）⇒ **V 侧宽通道化在现架构（窄行 hM=16/32 + bN=128 元素体量）下无正 EV 形态**；重试前置 = 向量执行侧出现宽化增益的新硬件/新形态证据（如 bN≥256 使单行体量翻倍摊薄 repeat 建立税——与路径 2 bN=256 蓝图存在组合可能，会话 10 若立项 bN=256 可顺带以 scratch 变体复测宽化 EV，零生产风险）。

**收口判定（会话 9）**：性能规则执行完毕（r9-1 性能下降 → 回滚，IR 16/16 逐字节还原 + l0 复验 + 抽查复测三重验证）；consecutive_no_improvement = 1（< 3，中止条款②未触发）；迭代 1/4-6 + Stage-3 scoping 归档（不占实施迭代）+ 收口——**未强行耗尽预算（负 EV 验证不实施纪律：Stage 2 经 [STAGE2-RULING] 数据驱动裁决跳过，r8-1 ② 否决先例同款）**。会话 9 净产出 = **③ 族判决性证伪（双向结论 + V-paced 减速侧因果系数 ≈1 确证 + 重试前置入库）+ 路径 2 bN=256 scoping 全件移交（GATE 矩阵实编 5/5 PASS + EV 条件化修正 + F5 蓝图 + GATE-6 立项前置硬门）+ 探针工程五条工具级硬知识（[N,1] 2D copy 块粒度陷阱 / 观测点竞态 / None-assign segfault / alloc_var init 约束 / 别名重命名漂移）**。终态 = dda3ad08（1926 行）= 会话起点 = 会话 8 终态（cann-bench 72.59/0.96x 在案态；复评由 Orchestrator 轮末统一执行，同体预期分数不变）。备份链：perf10（开工快照）/ perf11（改前备份 + 回滚目标）同体 dda3ad08；wide 变体留存 /tmp/opencode/r9_probes/mha_r91_edit_v1.py（52dd7e8a，精度门禁全过态）。提交副本 mha_submission/ 未触碰 ✓；test_mha.py（0dd9e93c）零改动 ✓；.orchestrator_state.json 未读写 ✓。

### 迭代 r10-1：[#R10-1] 路径 2 — fbm prefill bN 128→256 结构加宽（form_b_mma 域；F5 L0C + 同取向 [2,128,128] 共池；Stage 1）（2026-09-30）

## [ANALYSIS] Step 0 基线核验 + scoping 全件加载（2026-09-30）

1. **基线核验 ✅**：mha.py md5 = dda3ad0877ba4df74bbd559f6912eeed（1926 行）== 会话 8/9 终态；备份 perf12（开工快照）+ perf13（改前备份）双落盘同 md5。
2. **证据加载（精读完成）**：[SCOPING-PATH2-R9]（GATE1-7 矩阵全件 + F5 蓝图 5 要点 + probe_r9_bn256_budget.py 实编读数方法 + GATE-6 实测方案入库原文）、[RESULT-#R8-2/3]（flat 骨架 = 本轮作用基座：顺序游走状态机 + C/V 双侧同式 + SEM/SIG 协议零漂移纪律 + 任务切换 Q/L0AS 条件轮同进同出）、[RESULT-#R8-1]（v4ksp 同取向机制全文 + **附.3 字面双取向形态不可表达教训**：16KB 槽下 GEMM1/GEMM2 B-tile 互为转置 → T.view 损坏（customize.py:94/112 Buffer 直构被拒）→ 同取向共池为唯一可表达解 + LOAD-FIRST 机制 + id 组令牌范式 + "mte1_stall 非杠杆"归档）、[RESULT-#R9-1]（③ 关闭条款 + Stage4.5 复活窗口判据原文："bN≥256 行体量摊薄 repeat 税（与路径 2 组合）"→ 本轮采纳后 scratch 复测授权）、bn256_budget_final.log（F2/F4/F5 三形态实编 5/5 PASS：L0C used 64.00K/池 128K + L0B 64.00K + L0A 48K + L1 288K + UB 170.75K）、会话 2 谓词迁移先例族 [#10/#10a/#10b/#11/#13]（single_kv_block→S_kv≤bN 边界移动 + 单块化=更高精度路径非位级预期 + 阻塞口径判定 + 单块化 shape golden 位级精确 max_abs=0.0 先例 + eff 公式 bN 参数化原生支持 + gemm_v0 transpose_B nTile 预算规则）、DESIGN rev4 §4.5（a-f 预算表 + L0B 共池三依据 + decode mma 化不可行推导）/§8.6-9/10（K/N-split 链完整性 init 错位禁令 + 共池相位纪律）/§7.2-4/5/6（init 13/destroy 12 闭环 + C/V 同式硬约束 + 条件 DMA 禁令同进同出豁免形）、r8_op_case9_iter2 联合态基线（wait_id2 232.06 暴露/MTE2 busy 57.8%/L2 命中 92.69%/速率 131.2GB/s/C scalar 95.5%）、探针工程五条工具级陷阱清单（会话 9 journal e5）。
3. **同会话新鲜基线 msprof ✅（r10_base，median n=25）**：case9 869.87 / case4 28.21 / case7 855.54 / case10 34.61 / case14 378.26 / case16 562.01 / case20 13.98 us——vs r9_base 全带内（case10 +2.19% = decode 小 kernel 环境漂移带 +3~4.6% 在案，守卫判定附同文件多段测量口径）。baseline_r10_iter1.json 落盘。
4. **源码级可表达性复核（先于形态裁决，禁止凭记忆猜 API 纪律）**：customize.py:115-228 npu_gemm——**region 操作数偏移 = 朴素行主序 strides（retrieve_ptr L191-209），不经 ZN fractal Forward 映射** ⇒ mma B 操作数的槽内列切片会传错物理偏移（[64,256] 槽 [:,128:256] 切片 → 传 offset 128 elts，正确 ZN 偏移 = 8192 elts）= 静默数据损坏（plan-a C-region 崩溃 + 附.3 T.view 损坏同族教训的第三例证）；**mma 操作数只可用整槽/首维槽切片**（生产已验证形）。common.h:55-105 copy_gm_to_l1——**realTailM==0 → tailM=dstM（0-extent copy = 整块复制 + 无清零）** ⇒ 尾块半区载入必须钳位 extent ≥1 且源行钳位界内（设计入 va/vb/jb 钳位公式，见 IMPL）。common.h:129-150 copy_l1_to_l0b——dstM/dstN 运行期实参 ✓（双切片/transpose 组合的 src 偏移由 codegen layout_map.Forward(region_min) 计算，k_l1 ZN 注记依据 L1243-1258 原文）。
5. **形态裁决（F5 L0C 保留；L0B 共池 = 同取向 [2,128,128]，偏离 scoping 字面 GATE-2 形态 [2,64,256]+[2,128,128] "同池"——字面组合不可表达）**：字面 GEMM1 [64,256] 槽 + F5 l0c_s [2,32,128] n-chunk 需要 mma B region 列切片（第 4 条源码证据否决）；GEMM2 [128,128] 与 GEMM1 [64,256] 异形共池需 T.view（附.3 已证损坏）。**唯一可表达单池解 = 同取向等形槽（v4ksp 原则）**：T2S = pool [2,128,128]（现生产同声明）双 GEMM 均 [128,128] 槽（GEMM1 = 2×128 K 链/n-chunk，GEMM2 = 2×128 PV K 链/n-chunk，mma 全模板 <32,128> = 生产同形）vs T1 = pool [4,64,128]（v4ksp 全架构，16 mma/iter，GEMM1 需行列双切片 transpose = 新组合）。**选 T2S**：零新 MTE1/mma 模式（GEMM1 B = 纯列切片 transpose 生产逐字形；GEMM2 B = 纯列切片 direct 生产逐字形；P/Q colseg→L0A 生产逐字形）；唯一新工程点 = K/V 半区双 MTE2 载入 + 尾块钳位公式（0-extent 陷阱规避）。T1 留作 T2S 探针失败时的备选（其行列双切片 direct 半边已有 v4ksp 实证，transpose 半边需新探针）。
6. **K/V L1 半区拆分（T2S 使能前置）**：k_l1/v_l1 各拆 2×[128,256] 半区缓冲（k_l1/k_l1_h1、v_l1/v_l1_h1；bN=128 域 h1 别名主缓冲 = 零 IR 影响）⇒ GEMM B 载入保持"整行×列切片"生产模式（行列双切片 transpose 新组合被结构性规避）；L1 总量 288K 不变（GATE-4 同值）。MTE2 事件 4/slot（K×2+V×2 各半区）vs 旧 2/slot×2 槽 = 每 256 列事件数持平偏降；字节总量恒等。
7. **谓词迁移判定（#10/#10a/#11 先例族逐字执行）**：host bN 派生 = `S>16 ∧ D==256 ∧ S_kv>128 → 256，else 128`——①S_kv≤128 fbm 域保持 bN=128（单块批分支旧体 verbatim ⇒ B_single_d256 型 IR 逐字节守卫保留）；②S_kv∈(128,256] 迁移单块批分支（更高精度预归一路径，非位级预期，阻塞口径 + golden 位级双判据，session-2 "单块化 shape max_abs=0.0" 先例）；③case4（S_kv=256）∈ 迁移域 = 主目标之一按单块新体执行；④Form A 全域/decode 兜底域 bN 档位零触碰。
8. **位级预期（如实，非全 BITEXACT）**：w1 scores 列值 = 基线对应槽列值 BITEXACT（每列 K 升序 2×128 链与基线逐 chunk 同构，r4-2a/#R8-1 原则）；**整核输出 = 非位级预期**（softmax 分组 256 vs 128 → m_i 轨迹/r_factors/P 舍入路径改变 + GEMM2 K=256 链 vs 2×K=128+deferred-rescale 累积路径改变）⇒ 判定口径 = golden 阻塞口径（session-2 bN 扩展先例：MARE_abs 3.05e-5~2.44e-4 对 gate 余量 40-640x）+ 锚-B fbm 域按 §9.2 纪律转阻塞口径（mismatch != FAIL 条款原文适用）+ 锚-A 9/9 Form A 硬守卫保留（IR 逐字节 ⇒ BITEXACT 构造性成立）+ 迁移形状 golden 位级判据（单块化 = 预归一 = golden 栅格）。红线四项：双舍入栅格 r16 链逐字未动（emit_v 零改动构造性保证）/单块预归一特化逐字未动/sumexp+1e-30 逐字未动/掩码链 = eff/j0/valid_n 公式 bN 同值代入（钳位列 ≥ S_kv 恒被 bound/causal 掩码覆盖 + P=0 精确零 × 有限值 = 0 无 NaN 通路，vb≥1 钳位行 = 真实 GM 行有限值）。

[ORDER-CHECK] 准备实施: [#R10-1] 路径 2 fbm prefill bN=256 结构加宽 | 前置依赖: [用户 2026-09-30 基于 [SCOPING-PATH2-R9] 全件决策立项 ✅；GATE-1..5 实编 5/5 PASS（F5 形态 probe_r9_bn256_budget.py 原声明 = l0c_s [2,32,128] + l0c_o [2,32,128]，本轮实施态同声明复用）✅；GATE-6 实测方案入库（本轮 Stage 2 执行，中止门）✅；flat 骨架作用基座（#R8-2 采纳态）✅；v4ksp 同取向机制知识（#R8-1 归档）✅；谓词迁移先例族（#8d/#10a/#10b/#11/#13）✅] | 结论: ✅
[IMPL-#R10-1] 已阅读: [SCOPING-PATH2-R9] 全节（GATE 矩阵/F5 蓝图/EV 条件化/GATE-6 方案）、[RESULT-#R8-1/2/3]（v4ksp 形态学 + flat 骨架 + Form A 扩展）、[RESULT-#R9-1]（③ 关闭 + Stage4.5 条款）、会话 2 [#10/#10a/#10b/#11/#13]（谓词迁移三先例 + gemm_v0 L0B 预算规则 + 阻塞口径判定）、DESIGN §4.5-a..f/§7.2-4/5/6/§8.6-9/10、customize.py:100-229（npu_gemm 全函数：region 偏移朴素 strides 证据 + K==K_B 唯一形状断言 + M/N 取自 C + K 运行期）、common.h:55-105（copy_gm_to_l1 need_clear 全路径 + 0-extent 陷阱 L60）/107-150（copy_l1_to_l0a/l0b dstM/dstN 运行期实参）/152-185（mma MmadParams + unitFlag）/187-208（copy_l0c_to_gm colseg 路径）、mha_r81_edit_v1.py:1011-1234（v4ksp 池声明 + LOAD-FIRST 编排 + L0AO 组令牌 + colseg FIX 全 choreography）、mha.py 全文（fbm prim_func 1219-1698 逐行 + emit_v 276-804 + host 177-274 + decode 兜底 1700-1886 边界确认）、probe_r82_flat.py（探针骨架 = 本轮 probe_r10_bn256.py 模板）、run_gates_r91.sh（门禁电池 runner 模板）、p6_battery_anchors.py（双锚结构 + b11-b14 单块 S_kv≤128 项 = 本轮 IR 不变域 BITEXACT 保留判据）、bn256_budget_final.log + probe_r9_bn256_budget.py（F5 声明原形 = 实施态 L0C 同声明）。关键约束: ①host bN 派生谓词 = `S>16 ∧ D==256 ∧ S_kv>128`（S_kv≤128 fbm 保 bN=128 旧体 verbatim ⇒ IR 逐字节守卫保留；Form A/decode 零触碰）；②L0B 共池 = [2,128,128] 同取向（分析第 5 条源码证据链裁决，偏离字面 GATE-2 形态如实归档）；③K/V L1 半区拆分 k_l1_h1/v_l1_h1（bN=128 域别名主缓冲 = 零 alloc 零 IR 影响；annotate_layout 字典重复键坍缩 = bN=128 有效映射逐项同值同序）；④尾块钳位公式 va=min(128,valid_n)/vb=max(1,valid_n-128)/jb=min(j0+128,S_kv-1)（0-extent→整块复制陷阱规避 + 界内保证：valid_n>128 时 jb+vb=j0+valid_n≤S_kv 精确、valid_n≤128 时读真实行 S_kv-1 单行 + need_clear 清余 127 行；钳位半区列全部 ≥S_kv ⇒ 掩码 -inf ⇒ P=0 精确 ⇒ PV 0×有限=0 无 NaN）；⑤GEMM1 = per n-chunk nc∈{0,1}（k_l1半区[nc] 列切片 transpose ×2 K 链 init T→F → l0c_s[nc] → SIG_L0C+nc → w1 colseg FIX）；GEMM2 = per nc（v半区[kc2] 列切片 direct ×2 PV K 链 init T→F → l0c_o[nc] → w3 colseg FIX；l0a_o [4,32,128] = side×kc2，L0AO+side 组令牌，P colseg ×2）；⑥id 矩阵 = 现生产逐字不变（L0KC{0,1}/L0AO{2,3}/L0AS{4,5}/L0C{0,1}/K,P,V,Q{0-3} = 6/8 ≤ 上限）+ init 13 set/destroy 12 wait 逐字不变 + SEM 矩阵零漂移（C/V 收支恒等由构造：槽体令牌形逐字继承 × nc 展开对称）；⑦C 侧分支结构 = `if single_kv_block and not bn256:（旧体 verbatim 零重排）elif single_kv_block:（bN=256 单块新体）else:（bN=256 flat 新体，host assert ¬single⇒bN=256）`——bN=128 fbm 域解析树与现态逐语句同 ⇒ IR 逐字节；⑧emit_v/V-scope 全体零源码改动（block_N 参数化公式 + 现有 io3_is_alias/r_2d 别名规则在 bN=256=D 自动翻转 = 既有 parse-time 分支，Form A P2 flat 域生产验证形）；⑨flat 骨架零漂移（total_flat 游走/批头快照/4 循环独立重走/kv0 条件轮同进同出/kv-last L0AS 归还/§S4-3-5 复位时序逐字保留，eff/j0/valid_n 公式 bN 原生参数化——total_flat=Σeff 运行期派生与 bN 无关性 = C/V 双侧同式确定性算术恒等）；⑩令牌收支静态可验（每 nc 展开 = 现槽体令牌形 ×2 对称复制，SIG/SEM 计数表同进同出无条件 DMA，#R5-3 纪律）；⑪红线四项（分析第 8 条）；⑫T.Pipelined/threads=/barrier_all/gemm_v0-in-fbm/CROSS_INTERVAL/NUM_STAGES 全禁令继承。

[RESULT-#R10-1] 优化点: **路径 2 fbm prefill bN 128→256 结构加宽（form_b_mma generic 域 = case9/case4；T2S 同取向 L0B [2,128,128] 共池 + F5 L0C n-chunk（l0c_s/l0c_o [2,32,128]）+ K/V L1 半区拆分（k_l1_h1/v_l1_h1，bN=128 域别名主缓冲）+ 尾块钳位公式（va/vb/jb，0-extent 陷阱规避）+ 谓词迁移 `S>16∧D==256∧S_kv>128`；实施规范 = [IMPL-#R10-1] 12 条硬约束全文引用）** | 精度: **FAIL — HANG 缺陷（AICORE 执行超时挂死 507014；未达数值门 ⇒ 非数值性精度失败，T1/E22 数值正确性未判——如实声明）** | 性能: **n/a — HANG 先于性能门（msprof/GATE-6/GATE-7 全套未执行；[SCOPING-PATH2-R9] 条件化 EV（-12~-19% @L2≥90% / 0~+10% @命中塌缩）作废归档）** | 采纳: **NO — 结构性不可行（[DESIGN_ERROR] 判据字面命中；判据 = debug_log D10-3 预登记穷尽条款，非性能下降/无变化规则）**

**会话 10 中断恢复注记（如实，[RESULT-#R8-3] 同款格式）**：r10-1 实施（mha.py → c748cfb2，2259 行）+ HANG 定位（bisect 判决矩阵 + printf 51 检查点）+ E 系毒化矩阵 + E21 决胜 + T1 终局实验均在前次会话完成；**外部中断点 = T1 判决矩阵 3/3 运行完成（2026-09-30 22:42-22:52，EXIT 标记在案）、落档未做之时**。本恢复会话（2026-10-01 00:40-01:15+08:00，小预算收口作业）Step A 溯源核验全部通过：生产 mha.py c748cfb2/2259 行（r10-1 PROD 缺陷调试态，smoke_r10.log aicore timeout + bisect PROD NO-RUN 在案）/ 备份链 perf12 = perf13 = dda3ad08/1926 行 / mha_T1.py 04a41a97/2382 行（/tmp/opencode/r10_probes/ 留存）/ run_T1_{s256,f512,s192}.log 三日志 EXIT=1 标记完整 / debug_log D10-1..3 已完整落档（预登记判据原文在案）/ optimization_log 止于 [IMPL-#R10-1]（L1494）/ perf_log 止于 r9-3 / final_report 无 §S10——与 Orchestrator 事实基座逐项一致。**核验发现一处未落档偏差项：E22 变体**（详见下"E22 判决后补强实验"）。

**HANG 缺陷现象（shape 族）**：bN=256 新体首跑全形状族挂死——smoke case9-shape（smoke_r10.log 17:30 aicore timeout）、bisect single192（17:45）、PROD flat512/single256/single192/case9（run_PROD_s256.log 20:33 等）全部 `rtDeviceSynchronizeWithTimeout execution failed, reason=aicore timeout`（507014）；**同文件 legacy bN=128 域 PASS**（bisect_r10.log legacy_bN128_single RUN_OK）⇒ 挂死特异于 bn256 新体。printf 51 检查点定位（mha_dbg.py）：C 标量流走完 GEMM1 全程（C1-C6 + nc1 全部 fine 检查点）→ 停在 GEMM2 序言 CrossCoreWaitFlag(WS2_V2C)；V 停在 phase1 CrossCoreWaitFlag(WS1_C2V) ⇒ **系统级环**：C 的 FIX 队列未完成 w1 双 colseg 写 + WS1 发信 ⇒ V 不写 w2 ⇒ C 不等 WS2；真实停滞点 = M 管或 FIX 管内部（printf 不可见域）。静态审计 4 轮（C 侧令牌流 L0KC/L0AO/L0AS/L0C/K,P,V,Q + init 13/destroy 12、V 侧 IO_UB/S_HALF + SEM、跨核 SEM 计数、1-bit 事件旗标因果交替性）**全部平衡、无令牌死锁** ⇒ 非协议缺陷 = 硬件级停滞（挂死而非报错）。

**E 系判决矩阵全表（全程细节见 debug_log D10-1/D10-2/D10-3；bisect 族 3 + E 系 9 + E21 + E22 + T1 = 15 实测变体 + PROD 本体 4 shape + 2 补跑域验证；全部子进程隔离 + timeout 纪律）**：

| 变体 | 单因子改动 | 结果 | 判决贡献 |
|---|---|---|---|
| PROD（T2S 实施态） | — | **HANG** ×4 shape（flat512/single192/single256/case9） | 缺陷本体 |
| bisectB | K/V 单宽 [256,256] L1 + 行列双切片（无半区缓冲） | HANG | 排除 L1 半区拆分/第二注记缓冲/双 MTE2 半区载入 |
| bisectD | 双 GEMM 均删 nc1（单 burst/相位） | **RUNS**（flat512 + single192 补跑） | ⇒ **毒源 = nc1 第二 burst** |
| bisectF | io3/r_2d 别名关闭 | HANG | 排除 V 侧 alias+phase2 组合 |
| E1 | nc1 FIX dst → colseg0（同地址×2） | HANG | 排除 FIX colseg-1 GM 目的地址 |
| E2 | nc1 链式 mma init=False→True | HANG | 排除链式累加（cmatrixSource L0C 读） |
| E4 | nc1 全走 slot-0/id-0 | HANG | 排除 L0C slot-1 寻址 / FIX id1 流 |
| E5 | 只删 G2-nc1（G1 保双 burst） | HANG | — |
| E6 | 只删 G1-nc1（G2 保双 burst） | HANG | ⇒ **任一 GEMM 的第二 burst 均致挂**（对称性） |
| D@single192 | bisectD 跑钳位域 | RUNS | 钳位公式 va/vb/jb + need_clear 双半区有效（0-extent 陷阱规避验证） |
| PROD@single256 | 满 tile 无钳位单槽 | HANG | 钳位/多槽/flat 游走全部无关 |
| E13 | nc0 池释放延迟至 burst 末 mma2 后（4 处，令牌收支/数据不变） | HANG | 52-104ns 级裕量不足或理论错误 |
| E15 | nc1 FIX burst 全删（留载荷+mma） | HANG | FIX 第二 burst 理论排除 |
| E16 | 只留 nc1 B 载荷（删 nc1 mma；mma 数=4=D） | HANG | ⇒ **nc1 MTE1 重载独立致挂 = 毒化 (a)** |
| E17 | 只留 nc1 mma 读陈旧池（删 nc1 载荷；载荷数=D） | HANG | ⇒ **nc1 mma（B 二次读取/相位内第 3-4 条 K128-mma）独立致挂 = 毒化 (b)** |
| E21 | E4 布局 + 新 M↔FIX id2 边际握手（nc0 FIX 排空 ~0.3-0.5us 门控 nc1 重载 = 可达最大边际） | HANG（s256 + f512 双域） | ⇒ **边际理论单独不成立**（该边际 ≥ legacy 跨槽实测 ~0.26us）；与 E17 交叉判读 ⇒ (b) 为主因 |
| T1 | v4ksp 全架构推广：pool [4,64,128] + 每 nc LOAD-FIRST 4×[64,128] 分立槽 + chain-4 K64/相位 + nc0→nc1 FIX 排空边际握手（mha_T1.py 04a41a97，2382 行，ruff clean；Form A/decode/legacy-fbm 单块体逐字节 identical 核验 region diff 5/5 True） | **3/3 HANG**（single256 22:42 / flat512 22:42 / single192 22:51，run_T1_* EXIT=1） | ⇒ **预登记判据命中：全部可表达形态穷尽** |
| E22（判决后补强，见下） | T1 + 4× `T.pipe_barrier("m")`（gemm_v0 级全 M 排空置于相位内重用前，mha_E22.py 22:52 构建） | **3/3 HANG**（s256/f512/case9，23:02-23:21，run_E22_* EXIT=1） | 全排空编排亦不可救 ⇒ 判决强化 |

生成码复核（hang_E16_s256.cc/hang_E17_s256.cc/hang_single192.cc/hang_flat512.cc）：指令形态/模板参数/偏移全部正确（与 nc0 逐参数同形，仅源缓冲/次序不同）⇒ **非 codegen 伪影**。

**E22 判决后补强实验（中断遗留偏差项，如实落档）**：前次会话在 T1 判决矩阵完成（s192 22:52 = 3/3 定局）同分钟构建 E22（mha_E22.py，与 T1 diff 889 行 = 4 处 `T.pipe_barrier("m")  # E22: gemm_v0-class full M drain before intra-phase` 注入 + 关联编排），并于 23:02-23:21 运行 3 shape（s256/f512/case9）**全部 aicore timeout HANG（EXIT=1 在案）**——**未及落档即被外部中断**（debug_log 无 E22 记录，本条 + D10-4 为补档）。定性：E22 = 穷尽条款边缘的最后边际补强（gemm_v0 级全 M 排空 = 生产 legacy 谱系的最强排空编排移植），结果与判决一致且强化之（连全排空都不能规避双重毒化）；按穷尽条款处置 = **终止，无任何后续 bN=256 变体**。

**双重毒化归纳（全数据一致，D10-2/D10-3 归档）**：**(a) L0B 相位内覆写 @~50ns 边际挂死**（E16/E13：M 管 SetFlag<M_MTE1> 在 mma 指令退休点触发，cube 对 L0B 的物理读排空滞后于退休信号，立即覆写触发硬件级停滞）；**(b) K128-mma 相位深度 ≥3 或 B 操作数二次读取挂死**（E17/E21/E15/PROD）。生产安全谱系三条对照（全部满足 (a)(b) 规避）：①legacy 跨槽重载但天然被下一槽 K/V MTE2 到达等待门控（~0.5-1us 裕量）；②v4ksp 4 分立槽 load-first 零相位内重用；③legacy G2 每池槽单次读取。**bN=256 的 B 工作集（K^T 128KB / V 128KB）> L0B 64KB ⇒ 相位内 L0B 重用（重载或二次读取）数学上不可避免 ⇒ 无安全编排**。

**静态不可能性证明族（未实测覆盖形态的排除论证，D10-3 预登记原文）**：T3 region 偏移损坏（customize.py:191-209 retrieve_ptr 朴素行主序 strides 不经 ZN fractal Forward 映射 ⇒ mma B 槽内列切片 = 静默数据损坏，plan-a C-region 崩溃 + 附.3 T.view 损坏同族第三例证）；F2/F4 mma-N256 无先例 + B 128KB 单槽预算不可行；操作数互换 M=256>64 硬限；L0B 64KB < 128KB B 工作集数学（**仓内互证先例：gqa_fwd_varlen:112-114 block_N=256 的 L0B 128KB>64KB 否决 + rev4 DESIGN §4.5-e decode mma 化分池否决推导同款数学**）。

**T1 终局判决 = D10-3 预登记判据字面命中**：判据原文（debug_log L453）——"T1 HANG ⇒ 全部可表达形态穷尽（T2S/E 系 9 变体/T1 + 静态不可能性证明族：T3 region 偏移损坏、F2/F4 mma-N256 无先例+B 128KB 单槽、操作数互换 M=256>64 硬限、L0B 64KB < 128KB B 工作集数学）⇒ 路径 2 [DESIGN_ERROR]（任务书示例情形'F5 形态实施中暴露 scoping 未见的结构冲突'字面命中；≥3 方案实测尝试达成）"。实测：T1 3/3 HANG ⇒ **判据命中，宣告生效（恢复会话 Step B，2026-10-01）：路径 2（fbm prefill bN=256，含 F5/E 系/E21/E22/T1 全部可表达形态）结构性不可行；禁止任何形式的 bN=256 重试/变体再构造（穷尽条款）**。

**采纳判定：NO — 结构性不可行（非性能规则回滚）**。回滚执行（恢复会话 Step C/D，2026-10-01）：`cp history_version/mha_impl_r4_perf13.py mha.py` → **三重验证**：①md5 = dda3ad0877ba4df74bbd559f6912eeed 一字不差 + 1926 行 + ast.parse OK + ruff "All checks passed!"；②**IR 16/16 逐字节还原**：新鲜 dump（dump_rollback_r10.py = tilelang.disable_cache() 全局关缓存 + 专属 TILELANG_CACHE_DIR=/tmp/opencode/r10_cache_rb）→ /tmp/opencode/r10_src_rollback vs 生产参照 /tmp/opencode/r8_src_iter3 `diff -rq` 零差异 + _manifest.txt 逐字节一致（provenance 链：r8_src_iter3 = r8-1 采纳态 dump、会话 9 已证 r9_src_rollback 与其 16/16 一致，本恢复会话使用前复核该参照链自身 diff exit 0）；③**工具教训⑤（k_l1_h1 别名绑定序漂移）收尾关闭**：生产态源码与 16 IR dump 双 grep 零残留（k_l1_h1/v_l1_h1 = 0 hits）——漂移仅存在于 r10-1 变体族（名字级，功能全同：地址/尺寸/指令流一致），D10-1 教训⑤"收口前复验残留"待办随回滚消解，无需再修。**回滚态轻量复验（会话 9 r9-3 同款口径子集，run_rbv_r10.sh，r10_cache_rbv 隔离缓存）**：l0 **10/10**（L0_EXIT=0）+ 锚-A **9/9 BITEXACT**（vs Developer final 6505 硬守卫）+ 锚-B **14/14 BITEXACT**（vs r3 final 阻塞口径，BATTERY_RESULT: PASS）+ 继承压力 **4 配置 ×12 nondet=0**（l0_case9_multiblock/decode_d1/prefill_p2_causal_mid/prefill_p1_bigbatch，STRESS_RESULT: PASS）——全量门禁有效性依据 = dda3ad08 会话 8 采纳 + 会话 9 收口两度新鲜全量验证在案（r8_cache_finalg/r91_l0_rollback）+ 本次 IR 16/16 逐字节还原 ⇒ 轻量子集即足（非全量重跑，口径与 r9-3 一致）。

**归档知识（判决性，新增硬知识）**：
1. **手动 mma 流水的 bN 上界定理（硬件硬限）**：form_b_mma 类手动 K/N-split 流水的安全编排必要条件 = **B 操作数工作集 ≤ L0B 64KB（含 K-split 链全部在途 chunk）**。bN=256∧D=256 ⇒ K^T/V 各 bN×D×2B = 128KB = 2× 超限 ⇒ 相位内 L0B 重用不可避免 ⇒ 双重毒化 (a)(b) 下任何编排（边际门控/延迟释放/分立槽推广/全 M 排空）挂死。**scoping 方法学盲区归档**：GATE-2 实编 L0B 64.00K 顶格 PASS 验证的是**槽容量合法性**（单槽 ≤ 预算），不验证**工作集相位内重用的编排合法性**——"L0B 工作集（B 操作数在途总量）≤ 64KB"必须作为独立预算门**先于** GATE 矩阵执行；F5 蓝图"双顶格零余量警示"方向正确但不充分（顶格 = 必要非充分）。
2. **双重毒化机制**（见上归纳；挂死无报错 = 硬件级停滞，非令牌死锁——静态令牌审计平衡不能排除该类缺陷，判决必须靠运行时变体矩阵）。
3. **E 系毒化矩阵方法学**：挂死类缺陷（printf 不可见域）的判决性定位手段 = 单因子删除/单因子保留变体矩阵（bisectD 双删→毒源定界；E5/E6 单侧删→双 GEMM 对称性；E16/E17 载荷/mma 分离→双路径独立性；E13/E21 边际梯度→边际理论证伪；E22 全排空→谱系上限封死）+ 生成码复核排除 codegen 伪影——15 变体在 ~7 小时内完成从"协议死锁怀疑"到"硬件硬限定理"的归因闭环（子进程隔离 + timeout 纪律全程，无一次环境残留）。
4. **③ 族关闭升级为规则内永久 + 大 prefill 规则内杠杆版图收束**：③ 重试前置"bN≥256 行体量摊薄 repeat 税"（[RESULT-#R9-1] Stage4.5 复活窗口判据）随路径 2 结构性不可行 ⇒ **规则内不可达** ⇒ ③ 关闭由"条件性"升级为"规则内永久"（复活前置仅剩新硬件（如 L0B 扩容）/框架级新形态证据）。版图终态：**① 已兑现**（r8-2/r8-3 flat 采纳态 = 现生产；case9 r3 终态 1379.56 → 现 869.87us 累计 -37%，§S4-§S8 链）；**② 证据性关闭**（r8-1 机制激活失败 + 复活否决先例，复活前置 = MTE2 管收缩后 MTE1 成为 pacing 管的新结构证据）；**③ 规则内永久关闭**（本条）；**④ w3 瘦身条件性关闭**（激活前置"①-③ 落地后 B régime 下 FIX/L2 成为绑定管道"链条断裂）；**路径 2 结构性不可行**（本 RESULT）⇒ **case9 M1 线 0.4（0.39x 差 0.01；唯一规则内杠杆已证不可行）与大 prefill ≥1.0x（B 层理想流水上限 0.54-0.79x）均规则内不可达，仅剩架构级方案**（任务书所称"方案 C" = 框架虚拟通道 L0C→UB/UB→L1（A 层 2.3-3.4x 唯一入口，会话 6 禁令源码级实证在案 = 需框架改造）+ 单域 flash 重构；analysis §2.4/建议 4"架构决策 = 1.77-3.4x 空间唯一钥匙"，建议以 case9 为样板做框架级可行性验证后再投资决策）。
5. **遗留工件留存清单（知识资产，防未来重复探索）**：/tmp/opencode/r10_probes/ 完整留存——**mha_T1.py（04a41a97，2382 行，v4ksp 全架构推广终局变体）**+ E 系变体全文件（mha_bisectB/D/F.py、mha_dbg.py（51 printf 检查点）、mha_E1/E2/E4/E5/E6/E13/E15/E16/E17/E21/E22.py）+ make_*.py 变换脚本族（含 E3 等未构建脚本）+ probe_r10_bn256.py/stress_r10_bn256.py/smoke_r10.py/bisect_r10.py/budget_r10.py + 生成码 dump（hang_single192.cc/hang_flat512.cc/hang_E16_s256.cc/hang_E17_s256.cc/refix_G_B_single_d256.cc/refix_G_case4_B_p3_kv2.cc）+ 日志族（run_T1_*（3/3 HANG 判决证据）/run_E*/bisect_r10.log/smoke_r10.log/run_PROD_s256.log/dump_*.log/msprof_r10_base_case*.log）+ 恢复会话产物（dump_rollback_r10.py/run_dump_rollback_r10.sh/run_rbv_r10.sh/dump_rollback.log/rbv_{l0,anchors,anchor_stress}.log）；/tmp/opencode/r10_src_rollback/（IR 16/16 还原验证）+ r10_cache_{rb,rbv}（隔离缓存）。**如实声明：r10-1 PROD 缺陷态源文件（c748cfb2，2259 行）无独立留存副本**（回滚 cp 按任务书 Step C 直接执行，pyc 亦被回滚态覆盖）——其实现内容由 E 系派生变体全文件（= PROD 单点突变体）+ make_*.py 变换脚本 + [IMPL-#R10-1] 12 条规范 + hang_*.cc 生成码实质保留，知识无损。

**收口判定（会话 10）**：迭代 1/软预算 4-6 + 收口（中断恢复）；采纳 0/回滚 1（结构性不可行，非性能规则）；consecutive_no_improvement = 2（< 3，中止条款②未触发）；**[DESIGN_ERROR]（路径 2）上报 Orchestrator 裁决**（本 agent 预判 = 候选证据性关闭而非设计回退：现行 rev4 设计产出健康终态 72.59/0.96x，不可行的是可选优化项而非现设计）。会话 10 净知识产出 = bN 上界定理 + 双重毒化机制 + E 系毒化矩阵方法学 + scoping 盲区（工作集门 vs 槽容量门）+ 工具级教训 5 条（debug_log D10-1 在案：jit 工厂零参+out_idx 段错误/f-string 双花括号/pkill 自杀/occurrence 双增/别名绑定序漂移）。终态 = dda3ad08（1926 行）= 会话起点 = 会话 8/9 终态（cann-bench 72.59/0.96x 在案态；复评由 Orchestrator 轮末统一执行，同体预期分数不变）。备份链 perf12/perf13 同体 dda3ad08 无变化；提交副本 mha_submission/ 未触碰 ✓；test_mha.py/DESIGN.md/proto.yaml 零改动 ✓；.orchestrator_state.json 未读写 ✓；docstring：回滚态零变更 → 无注记（条款条件性适用如实记录）。

---

## 会话 11（perf_session=11，2026-10-04）：decode D=256 duty 墙专项（用户立项，S2 优先决策路径）

[ANALYSIS]-S11 Step0（基线记录 + 证据加载 + 选侧推理 + EV 守恒修正，r11-1 开工）:

1. **生产态核验**：mha.py md5 `dda3ad0877ba4df74bbd559f6912eeed`（1926 行）与任务书一致 ✅；改前备份 `history_version/mha_impl_r4_perf14.py`（md5 核验同体）✅；IR provenance 门：新鲜 dump（disable_cache + 专属 `TILELANG_CACHE_DIR=/tmp/opencode/r11_cache_base`，dump_base_r11.py）→ `/tmp/opencode/r11_src_base` **16/16 逐字节 = r8_src_iter3 参照链**（diff exit 0；T_case1..G_case15 全部 md5 与 _manifest 一致）+ 新增 T_case18_B_d256_mtp 基线 dump（9912B，md5 3d40a0eb——本轮双目标 shape 的"按预期改变"对照基线）。
2. **同会话新鲜基线（r11_base，median n=25，warmup 5/iters 20，msprof --task-time=on 口径同 analysis_20261003 §0）**：case15 **505.17**（roofline 参照 503.84，+0.26%）/ case18 **504.05**（505.07，−0.2%）/ case10 34.44 / case11 309.47（+2.6% 带内）/ case14 378.05 /（case16/20/9/7 采集完成后补录 baseline_r11_iter1.json）。环境无跨日漂移，device 0（npu-smi 无他进程）。
3. **证据加载（精读完成）**：analysis_20261003 §2（case15 op 读数：C wall 498.6、mte2 duty 68.45%、busy 速率 74.8 GB/s/核、mte1_stall 356.7us=71.5% wall、**wait_id1 0.96us（可忽略）**、**wait_id2 122.7us（1.28us/槽，被 V 载入重叠吸收）**、wait_id5 ≈0、fix_t 66.3us、V wait_id0 241.2/wait_id4 231.4）+ §2 每槽分解（5.25us = K/V 3.61 busy + 1.64 idle）+ §6.2 路线 2；[IMPL-#R4-2b]/[RESULT-#R4-2b]（DB 槽位记账全文 + 预取点/守卫设计 + 零新增 flag 论证 + "零响应仅因 prefill 域 MTE2 被 mte1_stall 阴影吸收"复活逻辑）；[RESULT-#R4-3]②（decode 每槽 wall=数据搬运带宽）；[RESULT-#R8-1]（机制判据 (a)：wall 收益必须伴 duty 签名）；DESIGN §4.5-a D2 行（272KB→单侧 DB 400KB ≤511.75KB）；S4-3-4②（3D 无 annotate L1 + gemm_v0 运行期切片 ≡ 2D 串行，BITEXACT 机制在案 = **免机制探针**）；probe_r42b_l1db_mech.py 语法先例（`k3_l1[s%2,:,:]` copy/gemm_v0 双触点）；§7.2-5（运行期守卫内纯 MTE2 copy 零 flag 允许模式）；#R6-2 约束⑦（批间预取置于 set_cross_flag("MTE2",…) 之后——MTE2 FIFO 不得延迟 w2 槽释放）。
4. **gemm_v0 模板源码级核实（本轮新增判决性证据，src/tl_templates/ascend/common.h:1189-1372）**：①prologue = `SetFlag<MTE2_MTE1>(0); WaitFlag<MTE2_MTE1>(0)`（**管级全排空**：其 MTE1 等待此前发出的全部 MTE2 op，非按缓冲区）+ `SetFlag<FIX_M>(0); WaitFlag<FIX_M>(0)`（FIX 全排空，acc L0C 复用序）；②每 tile `PipeBarrier<PIPE_M>` 标量阻塞（= mte1_stall 签名的物理来源）+ L0 ping-pong（L0AB_EVENT 0/1）；③epilogue = `SetFlag<MTE1_MTE2>(0); WaitFlag<MTE1_MTE2>(0)` + `SetFlag<M_FIX>(0); WaitFlag<M_FIX>(0)`——**返回后发出的 MTE2 被硬件序化在全部 MTE1 读之后 = 槽位覆写安全窗（#R4-2b 机制的源码确证）**；④生成码核验（G_case15_B_d256_dec.cc:60-81）：C 侧零用户 flag（同步全在模板内）、L1 池 InitBuffer 524032B 全池、GetWithOffset q0/k8192/v139264/p270336（现占 278528B=272KB ✓ DESIGN 对账）。
5. **稳态守恒律推演（EV 修正——本轮核心分析产出，写实测前预登记防事后归因偏差）**：标量顺序发射 + gemm_v0 阻塞标量（每 tile PIPE_M barrier）+ prologue 管级 MTE2 全排空 ⇒ (a) 在 gemm 调用**前**入队的预取会把该 gemm 的 MTE1 起点推迟预取全量（排空覆盖全部在先 MTE2）；(b) 在 gemm 返回**后**发出的预取与单缓冲下一槽载入的发出点相同（epilogue 释放 = 最早合法点，现码已在此点）⇒ **稳态每槽 wall = MTE2 字节时间(3.61us) + gemm 标量阻塞(X1+X2≈1.64us)，与 L1 槽数无关**；31% duty 缺口的主体（gemm 调用内部标量阻塞期 MTE2 空转）在 gemm_v0 黑盒 + 规则内手段下**不可由 L1 DB 消除**——预取载入在 prologue 排空等待期 MTE2 是 busy 的（该段即载入暴露段本体，只是位置移动），idle 段严格等于标量困在 gemm 调用内的时间。唯一可压缩项 = **任务/轮边界先导重叠**：K_0[t+1] 预取发于任务 t GEMM2 尾（set WS2_C2V 之后），其载入与标量尾（FIX w3 发射+set 指令+循环尾+Q 发射+WS1 wait≈0.02us/task）重叠，先导 L≈0.3-0.6us/task × 48 task ≈ 15-30us。**EV 修正：任务书 ~1.15-1.20×（模型）→ 本轮推演 ~1.03-1.06×**；msprof 实测唯一裁决——若实测显著超出推演，说明 prologue 排空/FIX 交互存在未建模重叠，以 op 读数归因入库；若实测 ≈0 且 duty 不动，即任务书 Iter-2 分支 B 前提成立（gemm_v0 调用边界 = 硬墙），守恒律推演为其机制证明。
6. **选侧推理（任务书 §2.3 数据驱动要求）**：**K-only**。依据：①K 侧任务头预取不延迟任何消费点（K_0 本就是任务 MTE2 队列首个大字节项，预取仅把发出点提前 L）；②V 侧任务头预取（V_0[t+1] 于任务 t 尾）会排在 Q[t+1]/K_0[t+1] 之前（MTE2 FIFO）⇒ gemm1(0)[t+1] prologue 全排空被推迟 1.75us，收益（GEMM2 时 V_0 已就绪）与延迟精确相抵（守恒律），且推迟首个 w1 交付 → V 核 wait_id0（241us 侧）协议波纹风险；③V 侧轮头预载窗（GEMM1 批末/WS3 wait 前）对 case15 ≈ wait_id5 ≈ 0 无收益；④K 侧协议洁净：w2/p_l1 链、GEMM2 SEM 吸收结构（V_i copy 在 WS2 wait 前 = wait_id2 1.28us/槽被载入重叠吸收的现行结构）、emit_v 全部零触碰。批内 after-return 预取（K_{i+1} 于 gemm1(i) 返回后）= 时序中性（与单缓冲同发出点）——为槽位记账完整性与 #R4-2b 机制同构执行，不计 EV。若采纳且余量明显，Iter-2 前加测 V-only 变体（任务书授权，各自独立门禁）。
7. **L1 预算（手算初筛 + 实编门待确认）**：q8 + k[2]256 + v128 + p8 = **400KB ≤ 511.75KB** ✅（DESIGN §4.5-a D2 行 272KB + 单侧 128KB；S2-3-2 教训：实编 GetWithOffset 读数为最终裁决——预测偏移 q0/k8192/v270336/p401408，末端 409600 ≤ 524032）。UB/L0 各池零触碰（V 侧不变；gemm_v0 内部 L0A/L0B 按 kL0Size/模板形状申请，与 L1 rank 无关——#10b 档 64KB⚠ 状态不变）。

[Part A 优化点清单（会话 11，继承筛选）] 本轮候选 = analysis_20261003 §6.2 路线 2 单侧变体（用户立项指定）：
- [#R11-1] K 侧 L1 双缓冲（3D [2,bN,D] 无 annotate + gemm_v0 运行期切片 + 边界先导预取）（参考: optimization-guide.md §2.3 L293-L321 首轮预取→正式循环三段结构；cube_optimization_path.md §2.2 L75-L139 预取双缓冲模式与同步开销告诫）：**适用**——作用域 decode D=256 多块（case15/18），机制先例 #R4-2b 免探针（S4-3-4②），L1 400KB 可行，零新增 flag；EV 按守恒律修正为 3-6%（Step0-5）。注：cube_optimization_path §2.2 "预取双缓冲无净收益"结论针对 L0 级 ping-pong + 手动 flag 税（mma intrinsic 语境）；本项为 L1 级 GM→L1 预取 + gemm_v0 自平衡 flag（零新增同步指令），不套用其否决，但如实登记为风险参照。
- 死列表全项复核（任务书 §5）：host KV 重排/跨 head 合并读/bN=256 mma 化/flat decode/NUM_STAGES 加深/w 瘦身/L2 往返消除/③ V 宽通道/case10 均匀化——**全部不适用（禁触，在案判决维持）**，本轮不实施任何死列表项。
- [KERNEL-REUSE-PRECHECK] candidate: #R11-1（optimization-guide §2.3）| expanded_domain: **无扩大**——变更域 = form_b_gemm_v0 ∧ D==256 ∧ kv_iters≥2（case15/18 + L0/L1 哨兵同形），host 调用契约/分派谓词/shape 域零变化 | compatibility_dimensions: §2.3 适用场景（DB 已开 + busy≤70% + kL1Iter≥2）——本域 mte2 duty 68.5% ✓、kv_iters=2 ✓、"已开启 Double Buffer"维度 = 本项即为首开（L1 级），语义为指南三段结构移植 | unknown_or_false: none（机制 BITEXACT 先例在案；EV unknown 由 msprof 裁决——非兼容性维度）| plan: direct-reuse（#R4-2b 机制 + S4-3-4② 切片形态直接复用于 decode 域，代码位置按 dda3ad08 谱系重新定位 L1802-1852，禁盲目移植已执行）。

[ORDER-PLAN] 实施顺序：1. [#R11-1] K 侧 L1 DB（前置依赖: 无——单一优化点会话，死列表封锁其余全部候选）— 理由: 用户立项指定 + analysis §6.2 路线 2 唯一规则内可表达候选；Iter-2 条件分支（S1 对照/S3 API 门）依 [RESULT-#R11-1] 裁决，不入本轮 ORDER-PLAN。

[ORDER-CHECK] 准备实施: [#R11-1] decode D=256 K 侧 L1 双缓冲（预取式，零新增 flag） | 前置依赖: [用户 2026-10-04 立项 ✅；analysis_20261003 §6.2 路线 2 判决（duty 缺口 31% 真墙 + 单侧 400KB 可行）✅；机制先例 #R4-2b + S4-3-4② 免机制探针 ✅；gemm_v0 epilogue 释放序源码核实（common.h:1367-1372）✅；IR provenance 16/16 ✅；同会话新鲜基线 case15 505.17 / case18 504.05 ✅] | 结论: ✅

[DB-ANALYSIS]（Double Buffer 特殊要求，skill 门禁）:
- Q1 循环内有 MTE3？C 侧无 MTE3（MTE3 = V 侧 UB→GM 专属）；C 侧写出 = FIX（L0C→GM w1/w3），与 MTE2 预取异管并发、无 DB 循环内交互新增。
- Q2 跨迭代累加器？无——gemm_v0 每次 init=True 全量重init acc_s_l0c/acc_o_l0c（单槽 L0C，prologue FIX_M 排空保证前次 FIX 读出完成）；DB 仅作用于 k_l1 载入槽位。
- Q3 同步方式？**gemm_v0 调用边界自平衡 flag，零新增用户 flag**：槽位释放 = epilogue `SetFlag<MTE1_MTE2>(0)`（MTE1 FIFO ⇒ gemm(i) 排空含 gemm(i-1) 对对侧槽的全部读）；就绪 = prologue `SetFlag/WaitFlag<MTE2_MTE1>(0)` 全排空。预取全部位于运行期守卫内、纯 MTE2 copy、零 flag/event（§7.2-5 允许模式）；无新增 event id ⇒ iter25「同一 flagId ≤15 次 set」约束与现域 id 占用（兜底域 C 侧用户 flag = 0）不受触碰。

[IMPL-#R11-1] 已阅读: optimization-guide.md §2.3 L293-L321（首轮预取→正式循环三段结构）、cube_optimization_path.md §2.2 L75-L139（预取 DB 模式 + 同步开销告诫 + 风险参照登记）、[IMPL-#R4-2b]（槽位记账恒等式/预取点/守卫设计逐字）、common.h:1189-1372 gemm_v0 全文（Step0-4 四点核实）、G_case15_B_d256_dec.cc 生成码全文（C 侧指令序列 + GetWithOffset 现值）、mha.py L1700-1886 兜底域全文 + L178-274 host 谓词区 + emit_v 零 L1 引用核验（grep 323-470 空）、probe_r42b_l1db_mech.py 语法先例、mha_r72_edit_v1.py:1240-1300（side 计算形态参照——本项 zero-flag 域不采用其手动令牌）、bench_mha.py/run_wall.sh（采集工具链）。关键约束: ①谓词 `kv_db = (S <= 16) and (D == 256) and not single_kv_block`（host 闭包 parse-time bool；≡ 任务书变更域 form_b_gemm_v0 ∧ D==256 ∧ kv_iters≥2）；②k_l1 alloc 三元式条件化 `[2, block_N, D] if kv_db else [block_N, D]`（无 annotate，S4-3-4② 形态；v_l1/q_l1/p_l1 不变）；③C 侧 generic else 分支 parse-time 双变体（kv_db 预取版 + else 原版 verbatim——identical-modulo-indent 逐行比对，#R4-2b 同款）；④预取点/守卫四件套：任务头 `t==0`（Q copy 后、轮循环前，K(task0,idx0)→slot0）/ 批内 `i+1<batch_iters`（gemm_v0 返回后立即、FIX w1_i 发射前，K(idx+1)→slot (idx+1)%2）/ 轮尾 `k+1<eff_num_outer`（set WS2_C2V **之后**，#R6-2 约束⑦，K((k+1)*NS)→slot0——NS=8 偶 ⇒ 轮头槽恒 side 0）/ 任务尾 `t+1<my_count`（轮循环后，K(task+1,idx0)→slot0，bz'/by' 由 task+1 纯标量重算）；⑤消费点：gemm_v0(q_l1, k_l1[idx%2,:,:], acc_s_l0c, transpose_B=True, init=True, kL0Size=gemm1_kL0)（side=idx%2 运行期切片，probe 先例形）；GEMM1 批内 i 槽不再就地载入（载入已全部前移至预取点——**零条件 DMA 违规**：每槽字节恰好载入一次，#预取=#消费=Σeff_kv 记账恒等式：1(任务头) + Σ_k[(batch_iters−1) + (k+1<eff?1:0)] = eff_kv 逐点成立，含 causal 运行期 eff 裁剪——守卫内 copy 同进同出）；⑥extent/need_clear 恒等：全部预取 copy 用与原载入逐字同形公式（j0=nidx*block_N, valid_n=T.min(block_N, S_kv-j0), src K[bz, j0:j0+valid_n, by, :]）——尾列清零语义逐 copy 保留；OOB 核验：批内守卫 ⇒ nidx≤idx_last ⇒ j0<S_kv ✓；轮尾守卫 ⇒ (k+1)*NS<eff_kv ⇒ j0'<S_kv ✓；任务头/尾 idx0 ⇒ j0=0<S_kv ✓（S_kv≥1）；⑦GEMM2 批/V 侧 emit_v/SEM 矩阵（set/wait 位置逐字不动）/single_kv_block 分支/form_a/form_b_mma/host 包装/工作区形状**零触碰**；⑧零新增 flag/event/SEM id（DB-ANALYSIS Q3）；⑨纯调度变更（数据流/算术逐 op 不变）⇒ **BITEXACT vs perf14 验收门**（探针覆盖任务书 §2.5 全拓扑 + 非零数据主组/zeros 对照组——case20 假阴性陷阱规避）；⑩L1 实编读数门（实施前置）：kv_db 形 dump GetWithOffset/InitBuffer，L1 末端 ≤524032B 硬门 + UB/L0 池与基线逐字比对无意外膨胀；⑪缓存纪律：全程 TILELANG_CACHE_DIR=/tmp/opencode/r11_cache_*，变体构建期生产 mha.py 冻结（基线采集完成前编辑落 /tmp 变体副本 mha_r111_edit_v1.py，采集完成后 cp 上位）。

[SELF-CHECK]（编辑后填写）

[SELF-CHECK] 本次 Edit 只涉及 [#R11-1]（变体 md5 89644d6833c4c52eab71cab63374a47a，2038 行，ast OK + ruff check "All checks passed!" + ruff format clean；先落 /tmp/opencode/r11_probes/mha_r111_edit_v1.py，基线采集完成后 cp 上位）：①diff vs perf14 = 恰好 3 区域（233a234-241 host kv_db 谓词 +8 行 / 1715c1723-1725 k_l1 alloc 三元式 / 1801a1812-1913 C 侧 elif kv_db 分支 +102 行）——1 删 113 增，原 generic else 体**零触碰**（纯插入，非重排）✅；②IR 守卫 dump 17 shape（r11_src_var vs r11_src_base）：**15/17 逐字节相同**（Form A ×8 + fbm ×3 + decode D128 ×3（case10/case14/case16/case17 中 4 项——G_case10/G_case14/G_case16/G_case17）+ G_B_single_d256 单块域）+ **2/2 目标按预期改变**（G_case15 9854→10277B、T_case18 9912→10335B）✅；③**L1 实编读数门 PASS**：GetWithOffset q0/k8192(131072 halfs=256KB)/v270336/p401408，末端 409600B = **400KB ≤ 524032B(511.75KB)** 与手算精确一致；L0A/L0B 65536、L0C 131072、UB 196352 池与全部偏移**与基线逐字相同 = 零意外膨胀** ✅；④生成码 C 侧审计（G_case15）：4 预取点就位（t==0 头预取 L64-66 / 批内 i<1 预取 L70-72（gemm 返回后、FIX w1 发射前 = #R4-2b 位置）/ 任务尾 t<47 预取 L86-88（set WS2_C2V 之后 = #R6-2⑦）/ 轮尾守卫因 eff_num_outer=1 编译期折叠消失 ✓ 预期）；槽位切片 k_l1[i*65536] / k_l1[(i*65536)+65536] = side×128KB 步长正确；**C 侧零新增 SetFlag/WaitFlag/CrossCore（SEM set/wait 位置与基线逐字相同）** ✅；⑤**AIV 主块（L91+）逐字节相同**（emit_v 零触碰 IR 级证明；全部 diff hunks 位于 L41-44 共享 GetWithOffset 声明 + L63-88 AIC 块内）✅；⑥**needClear=(dst_offset==0) codegen 规则审计（本轮新发现，如实归档）**：copy_gm_to_l1 末参 = codegen 按 dst 偏移派生（common.h:62-67 注释原文"full-tile clear only correct at tile base"）——side-1 槽载入 needClear=false ⇒ **尾块落 side-1 时（Skv=300 型：idx=1 valid_n=44）stale 行不清零**，w1 被掩码列 = q·garbage（可为 NaN/Inf half 位模式）；安全性证明 = V 侧掩码链 **select 替换制**（mha.py:552-558 `T.tile.select(work_ub, mask_2d, work_ub, -inf)` 在 r16 链后、reduce_max/reduce_sum 前；NaN 经 cast/mul 传播后被 select 整体替换 ⇒ 无 NaN 通路）+ 尾槽必触发 mask 构建（runtime skip 守卫 `idx*bN+bN-1 > min(s0+offset_eff, S_kv-1)` 对尾槽恒真）+ garbage 列限于 ≥valid_n（gemm 列独立无串扰）+ 256×256 zN 无 fractal padding 洞（fractalExtent==dstM*dstN，#1341 域外）⇒ 输出仍 BITEXACT 预期；探针 T_tail300/k10_tail300 精确覆盖此路径（smoke 已过）；⑦smoke 先行（会话 10 HANG 教训纪律）：case15_like_chain（my_count{1,2} 任务链）+ kv10_xround（轮尾预取点）+ tail300_side1（needClear=false 路径）**3/3 BITEXACT vs perf14，无挂死**（smoke_r111.log，SMOKE_EXIT=0）✅；⑧form_a/form_b_mma/host 包装/SEM-SIG 矩阵/test_mha.py/DESIGN.md 零触碰 ✅。备份 = history_version/mha_impl_r4_perf14.py（dda3ad08）。

[RESULT-#R11-1] 优化点: **decode D=256 K 侧 L1 双缓冲（S2 单侧非对称：k_l1 3D [2,256,256] 无 annotate + gemm_v0 运行期切片 + 4 预取点（任务头 t==0/批内 gemm 返回后/轮尾 k+1<eff/任务尾 t+1<my_count，全部 set WS2_C2V 之后或 FIX 之前的 #R4-2b/#R6-2⑦ 合法位）+ 零新增用户 flag/event + GEMM2/V 侧/SEM 矩阵/emit_v 零触碰）** | 精度: **pass（全门禁过）**——①探针 probe_r111_kvdb **34/34**（PART 0 分派审计 22 目标全落 kv_db 域 + 12 守卫全不落；PART 1 **22/22 vs perf14 BITEXACT**：case15/18 bench 原形、kv{2,3,4} 单轮、kv{9,10,16} 跨轮（rounds=2，轮尾预取点生成码实证 `if(k<1)` 就位）、causal S16/S2/跨轮（均匀恒等域）、Skv 尾块 640/300（**300 型 = 尾槽落 side-1 的 needClear=false 精确覆盖**）、bf16×2、显式 scale、大幅值域、my_count{1,2,mixed r_tasks}、最小 1 核 1 任务、zeros 对照（位级 0）+ golden 阻塞口径全过（最差 MARE_abs 6.1e-5 vs gate 9.8e-3 = 160x 余量）；PART 2 守卫 **12/12 BITEXACT**（D256 单块×2/D128 多块×2/D128 批处理×3/D64/form_a×2/fbm×2））；②锚-A **9/9 BITEXACT** + 锚-B **14/14**（BATTERY_RESULT: PASS）；③压力：继承 4 配置 ×12 nondet=0 + **扩展 17 项 kv_db 被改路径专项**（k01-k17：48-task 链/kv3 奇批/kv9-10-16 跨轮/causal×3/尾块×2/bf16/scale/my_count 奇偶混合/最小 1×1）×12 nondet=0（S5 方法学修正执行：扩展压力全部落在被改分支）；④l0 **10/10** + l1 **19/19** + coverage **19 PASS/0 MISS** + `--level all` **EXIT=0**（L2 12/12 正确拒绝 + Boundary 6/6 + 哨兵 4/4 PERF_PASS）；⑤**IR 守卫 15/17 逐字节相同 + 2/2 目标按预期改变**（G_case15 9854→10277B / T_case18 9912→10335B；AIV 主块逐字节相同 = emit_v 零触碰 IR 级证明）；⑥L1 实编读数门 PASS（400KB=409600B ≤ 524032B，q/k/v/p 偏移 0/8192/270336/401408 与手算精确一致；L0A/L0B/L0C/UB 池与偏移零变化）；⑦精度红线四项：r16 双舍入栅格链/单块预归一特化/sumexp+1e-30/掩码链——全部位于未触碰域（AIV IR 逐字节证明）✅；⑧smoke 先行 3/3 BITEXACT 无挂死（会话 10 纪律）；ruff check/format clean；全程专属缓存 r11_cache_{base,var,g1,smoke,iter1}。 | 性能(msprof median us, n=25, 同会话新鲜基线 r11_base): case15 **505.17→515.25 (+2.00%)**、case18 **504.05→517.68 (+2.70%)** ——双双劣化（目标噪声带 stdev 2.2/1.7us 的 5-8σ，两独立 shape 同向一致 = 真实效应）；守卫 7 case 全带内（case10 +1.36%（历史漂移带 +3~4.6% 内）/case11 +0.77/case14 −0.10/case16 −0.35/case20 +0.50/case9 −0.17/case7 +0.14——IR 逐字节域免计时归因，会话 8 case20 先例口径） | op 归因（/tmp/opencode/r11_iter1/op/case15 vs decode_roofline_a1/op/case15，机制判据 (a) 纪律）: **机制激活但 duty 签名不动 + 引擎效率税**——mte1_time 130.505→130.505 / cube_time 49.804→49.804 / vec_time 82.640→82.639（**三管精确零变化** = 计算与 MTE1 传输逐字节不变，排除 L1 双区布局对 MTE1 的影响）；**mte2_time 346.4→356.3us (+2.86%) 同字节（GM→L1 25.92MB 不变）⇒ busy 期速率 74.8→72.8 GB/s/核**；fixpipe 66.3→68.9 (+3.84%)；duty 68.45→68.89% (+0.45pp = 噪声级)；wait_id1/id2 不变（0.88/121.8）；V wait_id0 241.2→249.0 (+7.9 = C 变慢的纯阴影，V 自身工作零变化)；wall Δ(+9.4us) ≈ mte2 busy Δ(+9.9us) ⇒ 关键路径 = MTE2 引擎效率本身 | 对比: **+2.00%/+2.70% 劣化** | 采纳: **NO — 回滚（性能下降规则）**；回滚执行 = cp perf14 → mha.py（md5 dda3ad08 一字不差 + 1926 行 + ast/ruff OK）+ perf15 = 还原态同体（任务书 §2.1 编号语义）+ IR 17 shape 还原 dump 核验 + l0 复验（下条）；变体留存 /tmp/opencode/r11_probes/mha_r111_edit_v1.py（89644d68，2038 行，**精度门禁全过态**——r9-1 留存先例同款） | **归因知识（判决性，S1/S3 立项关键输入）**：**(1) gemm_v0 稳态守恒律（机制证明 + 实测双确认）**：prologue `SetFlag/WaitFlag<MTE2_MTE1>(0)` 管级全排空（+FIX_M 全排空）+ 每 tile `PipeBarrier<PIPE_M>` 标量阻塞 + 标量顺序发射 ⇒ 每槽 wall = MTE2 字节时间 + gemm 标量阻塞时间，**与 L1 槽数无关**——gemm 调用前入队的预取把该 gemm 的 MTE1 起点推迟预取全量；gemm 返回后发出的预取与单缓冲同发出点（epilogue MTE1_MTE2 释放 = 最早合法点，现码已在此点）。31% duty 缺口主体（X1+X2≈1.64us/槽 gemm 标量阻塞期 MTE2 空转）**规则内不可压缩；gemm_v0 调用边界 = 硬墙（任务书 Iter-2 分支 B 前提字面成立：duty 不动）**。唯一可回收项 = 任务/轮边界先导 L≈0.3-0.6us/task——实测被 (2) 吞没。**(2) 预取流重排税（新硬知识）**：K_0[t+1] 预取使 K 流与 V/w2 流在 MTE2 上时间交叠 + L1 写目的地区交替（8192/139264 两区），HBM/L2 访问流序改变 ⇒ 同字节下 MTE2 busy 期速率 −2.7%（74.8→72.8，本已贴 512B 行布局上限的引擎对流序敏感）+ FIX −3.8% 同族——**decode 带宽墙域（busy 速率已贴布局上限）的重排类优化具有负外部性，边界先导收益（~+3-5% 模型上界）不足以覆盖**；守恒律 (1) 使其无稳态收益对冲 ⇒ 净负。**(3) S1 对照（bN=128 缩块 + 双侧 DB）EV 推演为负，不建议实测**：守恒律 (1) 对 S1 同样成立（gemm_v0 黑盒不变 ⇒ duty 缺口仍不可压缩），而 S1 额外支付 SEM 轮 ×2 协议税 + 槽数 ×2 的更强流重排税（本实测单侧 2 槽已 −2.0~−2.7%）+ bN=128 槽字节减半使边界先导 L 同半 ⇒ 三项皆负/半；任务书要求的"S2 覆盖比例实测值"= 负覆盖（−2.0/−2.7%，duty Δ+0.45pp），S1 立项数据输入完成。**(4) needClear=(dst_offset==0) codegen 规则入库**：copy_gm_to_l1 末参由 codegen 按 dst 偏移派生（common.h:62-67"full-tile clear only correct at tile base"）——3D 双槽下 side-1 槽载入不清尾；安全性 = V 侧 select 替换制掩码（reduce 前 -inf 整体替换，NaN 无通路）+ 尾槽必触发 mask 构建（runtime skip 守卫恒真）+ garbage 列限于 ≥valid_n（gemm 列独立）+ 256×256 zN 无 fractal 洞——探针 T_tail300/k10_tail300 实证 BITEXACT（未来 3D 槽化改动必查此规则）。**(5) DB 槽位机制位级安全第六度入库**（#R3-2/#R4-2b/#R6-2/#R7-2/#R8-x 后）：22/22 BITEXACT + 记账恒等式生成码实证（kv=10 形：头 1 + 批内 7+1 + 轮尾 1 = 10 = 消费数）。**触发中止条款计数：consecutive_no_improvement = 3（#R9-1 性能下降 + #R10-1 结构性不可行 + #R11-1 性能下降）**——Iter-2 按任务书条件分支 B 执行（S3 三道 API 能力门 scratch 验证，零生产变更，非新性能迭代）。

### 迭代 r11-2：[#R11-2] S3 三道 API 能力门 scratch 验证（Iter-2 分支 B，零生产变更）（2026-10-04）

[ORDER-CHECK] 准备实施: [#R11-2] S3 能力门验证（非优化实施——任务书 §3 分支 B 预授权作业） | 前置依赖: [#R11-1 回滚落档 ✅（性能下降规则；duty 签名不动 = 分支 B 触发条件「S2 零响应且 duty 未动 → gemm_v0 调用边界即硬墙」实测成立——本轮为负响应，机制结论同）] | 结论: ✅
[IMPL-#R11-2] 已阅读: ascend.py:418-530（gemm_v0 前端全函数：init/n_actual 签名 + _retrieve_shape/_retrieve_ptr BufferRegion 处理 + K==K_B 断言 + tl.ascend_gemm_v0 intrinsic 构造）、customize.py:115-230（npu_gemm 对照：k_actual/unit_flag 为 T.mma 专属，gemm_v0 无 k_actual）、common.h:1189-1372（gemm_v0 模板全文，r11-1 已核）、[IMPL-#R10-1] 第 4 条（retrieve_ptr 朴素 strides ≠ ZN fractal Forward 映射 = mma B 切片损坏证据）、gate0b_verdict_iter25.md（「同一 flagId ≤15 次 set」= CrossCoreSetFlag ISASI 跨核计数器约束原文 + 1:1 交错合法性论证 + 钳位 HANG 机制）、#R7-2（intra-core ids≤8 硬件上限）、probe_r42b_l1db_mech.py（最小 Expert 探针骨架）。验证设计: 四形态实建实跑（iter25 纪律 #4 禁止预测性否决）——P-A 单段 K256 参考 / P-B 独立半缓冲链（门①）/ P-C A 列段切片（门②）/ P-D B 列段切片 transpose_B（门②'替代形态）；decode GEMM1 同形（M16/N256/K256/kL0=64/fp16/transpose_B）；门③ 静态盘点（生成码 + 模板 + iter25 档案）。

[RESULT-#R11-2] 门判决（probe_s3_gates.log，/tmp/opencode/r11_cache_s3 专属缓存）: **①PASS / ②FAIL / ②'FAIL / ③PASS ⇒ S3 关闭落档（任一门不过条款字面命中）**。①init=False 跨调用 L0C 链式累加：P-B vs golden max|d|=3.05e-05（= P-A 参考同级）**且 P-B vs P-A = BITEXACT 0.0000e+00**——2×K128 链与单段 K256 在 kL0=64 tile 化下同为 K 升序 fp32 L0C 累加 ⇒ 位级等价（强于任务书"累加序变化预期非 BITEXACT"的值级预期；与 T.mma 链式先例（form_b_mma init T→F，probe P3 16/16 BITEXACT vs gemm_v0 全 K）行为一致性在案）。②A 操作数列段 region 切片（q_l1[:,0:128]/[:,128:256]）：**数据损坏实建实证**（P-C vs golden max|d|=6.05e+01，rel=1.01 = O(1) 全错）——机理 = ascend.py _retrieve_ptr 朴素行主序偏移（[16,256] 列 128 → 128 elts）≠ zN 物理偏移（128×roundUp16(16)=2048 elts）；[IMPL-#R10-1] 第 4 条 mma B 切片损坏的**同族第 4 例证**（plan-a C-region 崩溃 / 附.3 T.view 损坏 / r10 mma B-slice / 本轮 gemm_v0 A-slice）。②'B 列段切片（transpose_B，k_l1[:,0:128]）：同损坏（max|d|=5.92e+01，rel=0.99；朴素 128 ≠ zN 128×roundUp16(256)=32768）。③event id 预算：兜底域 C 侧用户 flag=0（生成码实证）；gemm_v0 内部占用 L0AB_EVENT=0/1（MTE2_MTE1/FIX_M/M_FIX/MTE1_MTE2 id0 + M_MTE1/MTE1_M id0/1）⇒ 用户自由 id 2-7/通道（≤8 硬件上限）；跨核 SEM 0-5 生产占用不变、≤15 钳位约束由 1:1 交错纪律保障（gate0b 档案）——S3 若实施无 id 预算障碍。

**S3 关闭归档（诚实完整性：残余可表达形态 + EV 终判）**：门① PASS ⇒ K-split 链式在"独立半缓冲 + 整宽 GM 列窗载入"形态下可表达且位级正确（P-B 实证；L1 预算形态 q_a4+q_b4+kA128+kB128（GEMM1）与 v 侧对称 = 总量与现态同级）。**但 EV 决定性为负**：B 侧主体（K/V 128KB/槽）半宽载入使 512B 行退化 256B 行 ⇒ 微基准判决布局上限 1452→966 GB/s（−33% 布局税）恰好落在 decode D=256 唯一主墙（MTE2 载入）上；免布局税的唯一形态（整宽载入 + 列段切片消费）被门②②' 双 FAIL 否决。叠加 r11-1 守恒律（duty 缺口 31% 本不可由 gemm_v0 域调用重排压缩——S3 K-split 不改变每 tile PipeBarrier 结构与 prologue 全排空语义，X 项守恒 + 每 chunk 新增一组 prologue/epilogue flag 开销 = 微增），**S3 = 能力半可用 + 收益负 ⇒ 关闭，不移交实施方案**（任务书"三门全过才落方案"条款未触发）。

**规则内路径穷尽声明（decode D=256 duty 墙，case15/18 的 1.45× 上界空间）**：S1（bN=128 缩块+双侧 DB）= EV 推演负（守恒律 + SEM 轮 ×2 协议税 + 槽 ×2 重排税 + 先导减半；r11-1 归因 (3)，未实测——推演性关闭，如实标注）；S2（单侧 DB）= **实测负**（−2.0/−2.7%，duty 不动，r11-1）；S3（K-split 链式）= 门② FAIL + 布局税 EV 负（本轮，实建实证）；mma 化 = bN 上界定理 + 双重毒化永久否决（S10-3-1/2 穷尽条款）；vector 路径重构（baseline 的达成方式，存在性证明在案）= decode 不回退硬约束正面冲突 + 新 kernel 形态（超 Stage-3 规则范畴）。**⇒ 维持 dda3ad08 生产态 = 规则内最优；破局仅剩架构级**（gemm_v0 模板级 per-buffer 同步改造（消除 prologue 管级全排空 ⇒ 打破守恒律的唯一规则外杠杆）或框架虚拟通道/单域 flash 重构（会话 6 禁令源码级实证在案）——均超出本 Agent 授权，移交用户/Orchestrator 立项决策）。

**收口判定（会话 11）**：迭代 3/软预算 3（r11-1 实施 + r11-2 条件分支 + r11-3 收口）；采纳 0 / 回滚 1（性能下降）/ 能力门关闭 1；**consecutive_no_improvement = 3（r9-1 + r10-1 + r11-1）⇒ 中止条款②形式触发，如实上报**（iter24-26 纯测量轮不递增口径延续）。会话 11 净产出 = ①gemm_v0 稳态守恒律（机制证明 + 实测双确认，decode duty 墙的终局归因）；②预取流重排税（带宽墙域重排类优化负外部性，−2.7% 引擎效率实测）；③S3 三门判决（门① 链式 BITEXACT 能力入库 / 门②②' region 切片损坏第 4 例证 / 门③ id 预算盘点）；④needClear=(dst_offset==0) codegen 规则 + V 侧 select 掩码安全性证明（3D 槽化改动必查项）；⑤S2 变体精度门禁全过态留存（未来结构变更后可低成本复用其 22/22 BITEXACT 机制）。终态 = dda3ad08（1926 行）= 会话起点 = 会话 8/9/10 终态（cann-bench 72.63/0.96x 在案态；复评由 Orchestrator 轮末统一执行，同体预期分数不变）。备份链：perf14（改前备份）/ perf15（还原态同体）；变体留存 /tmp/opencode/r11_probes/mha_r111_edit_v1.py（89644d68）。提交副本 mha_submission/ 未触碰 ✓；test_mha.py/DESIGN.md/proto.yaml 零改动 ✓；.orchestrator_state.json 未读写 ✓；回滚态轻量复验 = l0 10/10 ✓ + 锚-A/锚-B/继承压力（run_rbv_r11.sh，结果闭合于 perf_log r11-3 条目）。

---

## 会话 12（perf_session=12，2026-10-04）：A10 gemm_v0 模板级 per-buffer 同步改造（用户明示授权框架级改动，2026-10-04「先尝试A10，若A10没有收益则尝试A9」）

### 迭代 r12-1：Phase A 可行性调研 + 设计（零生产变更）

[ANALYSIS]-S12（Phase A 六项必答，全部证据落档）:

**A12-1 模板细读（src/tl_templates/ascend/common.h L1183-1371，git 基线 clean = d113a6c6，最后触碰 commit 059f0a65）**：
- `L0AB_EVENT = 0`（L1183 constexpr）；模板签名 8 参：`<T1,T2,M,N,K,transpose_A=false,transpose_B=false,kL0Size=128>` + 运行期 `(A,B,C,l0a_,l0b_,clear,n_actual=N)`。
- **prologue（L1306-1309）**：`SetFlag<MTE2_MTE1>(0); WaitFlag<MTE2_MTE1>(0)`（= 管级全排空：set 绑定发射点 MTE2 队列尾，wait 使 MTE1 阻塞至该点 ⇒ **调用前发射的全部 MTE2 被收编进本次调用关键路径 = #R11-1 守恒律实体**）+ `SetFlag<FIX_M>(0); WaitFlag<FIX_M>(0)`（FIX 全排空 = L0C hazard：单槽 acc L0C 复用前等待前次 FIX 读出完成）。
- **k-step 乒乓（L1311-1364）**：M_MTE1/MTE1_M 令牌 id 0/1（prime 双 set L1311-1312）+ 每-step `PipeBarrier<PIPE_M>`（L1354，标量阻塞点 = 气泡内容实体）+ 收尾双 wait（L1364-1365）。
- **epilogue（L1367-1370）**：`SetFlag/WaitFlag<MTE1_MTE2>(0)`（**释放 L1 槽：调用返回后发射的 MTE2 被硬件序化在本次全部 MTE1 读之后**——A10 槽位复用安全性的基石，保持不动）+ `SetFlag/WaitFlag<M_FIX>(0)`（调用方 FIX 写出序化在 mma 之后）。
- 最小改造面判定复核（任务书 scoping 指引）：**成立**——PipeBarrier<PIPE_M> 阻塞的是标量发射流，MTE2 队列引擎独立执行已入队 op；A10 只需把 prologue 的 MTE2_MTE1 自 set+全排空改为等待调用方 per-buffer 旗标。FIX_M（L0C hazard 必需）、k-step 链、epilogue（槽释放机制被本设计依赖）全部保持不动，**无必要性证据，不改**（最小必要改动原则）。

**A12-2 事件 id 约束查证（三源交叉，判决性）**：
- **官方文档（RAG 检索 SetFlag/WaitFlag(ISASI)，CANN社区版 9.0.0-beta1，path: SIMD API/基础API/同步控制/核内同步/SetFlag/WaitFlag(ISASI)）约束说明逐字**：①「SetFlag/WaitFlag必须成对出现。」②「在基于TPipe和TQue编程场景中，禁止用户在使用SetFlag和WaitFlag时自行指定eventID…」（不适用：Expert 静态 Tensor 场景）③「**在静态Tensor编程场景中，事件的类型和事件ID由开发者自行管理，但需要注意事件ID不能使用6和7**（可能与内部使用的事件ID出现冲突，进而出现未定义行为）」。eventID 数据范围：A2/A3 训练/推理 = **0-7**。⇒ **核内 SetFlag/WaitFlag 文档无「≤15 次 set」条款**；可用 id = 0-5。
- **gate0b_verdict_iter25.md L54-58 出处核对**：「同一flagId的计数器最多设置15次」原文属 **CANN AscendC `CrossCoreSetFlag(ISASI)`（跨核信号量）**——与核内 SetFlag 是不同 API/不同机制。任务书疑问落定：**≤15 钳位约束不适用于 A10 的核内 MTE2_MTE1 事件**。
- **生产实证（1:1 交错合法性 + 深度约束）**：gemm_v0 k-step 链自身在 id 0/1 上每 kernel 数百次 Set/Wait 严格 1:1 交替（生产全绿）；mha form_a 域 SIG_K_L1/SIG_Q_L1（id 0-3）442+ 槽长循环 1:1 生产验证；fbm 域 SIG_L0KC/L0AO/L0AS 同款。**TQueSync 版文档语义（flag 置1/等待消费置0）提示核内旗标可能为 1-bit（计数深度 1）——生产全部先例均满足「每 (type,id) outstanding ≤1」**。
- **设计不变量（对 1-bit 与计数器两种语义同时安全）**：A10 全部新旗标严格 Set→Wait→Set→Wait 1:1 交替、每 (type,id) outstanding ≤1 ⇒ 无需裁决 1-bit vs 计数器（未验证项如实声明：核内旗标硬件语义无文档明文的计数深度条款，设计以最强假设 1-bit 自约束）。
- flag 空间按 (HardEvent type, id) 对独立——生产证据：gemm_v0 模板自身在 5 种 type 上复用 id 0；form_a 在 M_MTE1/MTE1_M/FIX_M/M_FIX/MTE2_V… 多 type 上复用 id 0-5；C 核与 V 核物理分核各自独立。⇒ 调用方使用 MTE2_MTE1 type 的 id 0/1/2 与模板内部其他 type 的 id 0/1 **零冲突**。

**A12-3 plumbing 通道选型（三通道调研 + 生成码证据）**：
- **生成码实证（/tmp/opencode/r11_src_base/G_case15_B_d256_dec.cc L67/L76）**：`tl::ascend::gemm_v0<half, float, 16, 256, 256, false, true, 64>(q_l1[0], k_l1[0], acc_s_l0c[0], ascend_l0a, ascend_l0b, (bool)1, 256);`——模板实参 = **Python f-string 全文粘贴**（ascend.py L525），运行期实参恒 6 个（op_name,Aptr,Bptr,Cptr,init,n_actual；ascend.cc L1298 `set_num_inputs(6)`）；codegen GemmOpCodegen（codegen_ascend.cc L2714-2736）仅拼串，**不解析模板实参**。
- transform 层识别按 **op 名**（operation_config.h L164 `tl.ascend_gemm_v0` → PIPE_M + 读写标注），不触碰模板字符串；ascend_l0a/l0b 为 codegen 全局 TBuf 声明（L1019-1021），非 per-call-site。
- **通道 (b) header-only + 前端 kwarg（选定）**：common.h 追加第 9 默认模板参 `int32_t kMTE2Sync = -1` + prologue `if constexpr` 分支；ascend.py 追加可选 kwarg `mte2_sync=None`，**仅当非 None 时**在模板串尾追加 `, {mte2_sync}`。**零 C++ 改动、零 rebuild**（两文件均为非编译资产：Python 前端 + JIT include 头文件）。默认路径逐字节不变证明：①mte2_sync=None ⇒ f-string 逐字符不变 ⇒ 全部现有消费者（73 文件/186 调用点非 mha + mha 15/17 shape）生成源逐字节不变；②现有生成码显式 8 模板实参 ⇒ 第 9 参取默认 -1 ⇒ `if constexpr` 折叠为原语句序列 ⇒ 设备码语义逐语句相同。
- **通道 (a) 运行期实参全链 plumb（否决）**：需 ascend.cc `set_num_inputs` 6→7（或 -1 变参 = 放宽全体消费者校验）+ codegen 条件发射 + **build.sh 级 C++ rebuild**；爆炸半径大于 (b)，收益（免 G1 奇偶展开）不成比例。留作 (b) 受阻时的升级路径。
- **通道 (c) 调用方绕过（T._src_code/call_extern 直呼）（否决）**：绕过 tl.ascend_gemm_v0 intrinsic ⇒ 丢失 operation_config PIPE_M 标注 + L0 TBuf 关联语义；改造函数仍需落 common.h（框架改动不减）；算子侧爆炸半径最大。
- **G1 奇偶 id 的编译期化（通道 b 配套，必要工程）**：kMTE2Sync 为模板参（编译期常量），而 K 双槽旗标 id 按 idx%2 交替（运行期）⇒ G1 消费点 TVMScript 奇偶展开：`if idx % 2 == 0: T.gemm_v0(..., mte2_sync=0) else: T.gemm_v0(..., mte2_sync=1)`（运行期 if 先例：`if i % CROSS_INTERVAL == 0:` mha.py L1793/L1844；生成码 = if/else 双实例化，目标 shape 本就变更域）。G2 单 id 单调用点，无展开。

**A12-4 生效链核查（采纳前置硬门，本轮查证完毕）**：
- **本地 JIT**：tilelang/env.py L177-182 —— TL_TEMPLATE_PATH 解析优先 `<tilelang包目录>/src`（install 态），不存在则回退 `<repo>/src`（develop 态）；libgen.py L168/L214 `-I{TILELANG_TEMPLATE_PATH}` 传给 ccec。**实机核验**：set_env.sh 下 `tilelang.__file__ = 源码树`、`tilelang/src 不存在`、`TL_TEMPLATE_PATH = <repo>/tilelang/../src` ⇒ **源码树 common.h/ascend.py 改动对本地 JIT 即时生效**（配专属新鲜 TILELANG_CACHE_DIR，缓存键按不覆盖 header 变更处理——纪律 ④-2）。
- **pip 包（cann-bench 评测）**：setup.py L408-417 将 `src/tl_templates` 整树拷入 `build_lib/tilelang/src/tl_templates` ⇒ **打包携带模板**；ascend.py 为包内 Python ⇒ 同携带。**结论：采纳态复评需 build_wheel_ascend.sh 重打包 + 评测环境重装（Orchestrator 轮末统一执行项，本 agent 不触碰）；打包脚本无需调整**（tl_templates 已在打包清单）。风险项落档：若评测环境仅重装 wheel 而残留旧 TILELANG_CACHE_DIR，opt-in shape 因模板串变化 cache key 必变 ⇒ 自动重编译，无毒缓存路径；非 opt-in shape 缓存命中旧 .so = 编译自旧 header 但默认路径逐字节等价 ⇒ 行为不变。
- **A10 与纯算子改动的本质差异归档**：框架资产（header+前端）随 wheel 分发 ⇒ 采纳判决附带「生效链方案 = 重打包+重装」条款，Phase D 落档四件套必含此项。

**A12-5 回归面盘点（仓内 gemm_v0 消费者）**：
- `T.gemm_v0` 直接调用面（examples/ + testing/，除 mha 自身）：**73 文件 / 186 调用点**（含 developer_mode/pipeline/gemm 族/sparse_flash_attention 族/gdn 族/deepseek_v4/fusedmoe/quant_batch_matmul/autotune/lightning_indexer/attention_sink 等）。
- **专用 UT 4 件**（Phase D 非 mha 消费者 smoke 电池，任务书 §0-e「至少 1 个」超额满足）：testing/python/language/test_tilelang_ascend_language_gemm_v0{,_k_tail,_n_actual,_n_tiling}.py。
- 间接消费者：pto.py（已废弃，禁触不改）同样构造 8 参模板串——不改其代码，默认参兼容覆盖。
- **16/17 shape IR 守卫 = 本次最强守卫**（r11 先例机制）：默认路径逐字节不变的机器证明；目标域 2 shape（G_case15/T_case18）按预期改变 + AIV 主块逐字节相同（emit_v 零触碰证明）。

**A12-6 FIFO 稳态时序设计（核心工程结论，含对任务书 payload 设想的修正）**：
- **关键修正（本轮推演新产出，Phase B 必须执行）**：S2 变体的 4 预取点全部为 **after-return 位**（gemm 返回后发射）——在 per-buffer 同步下**依旧时序中性**：发射点 = 标量退出点（程序序不可提前），K(idx+1) 最早于 G1(idx) 退出后入队 ⇒ G1(idx+1) 暴露 ≈ 1.674−0.1 = 1.57us ⇒ 槽周期 ≈ 2.32us 不变（守恒律的发射点推论：per-buffer 同步消除的是「排空收编」，不改变发射点）。**A10 兑现上界的必要条件 = 发射点前移至 pre-call 位**：iter idx 标量前段（w1(idx-1) 写出+SEM 之后、G1(idx) 调用之前）发射 K(idx+1) ⇒ K(idx+1) 于 G1(idx) 全程流式装载 ⇒ G1(idx+1) 入口近零暴露 ⇒ 稳态 MTE2 队列永不断流，槽周期 → 装载地板 1.674us+ε ⇒ duty→~100%，wall 上界 1.39-1.45×（真机 roofline 口径）。
- **FIFO 序不变量（设计第一原理）**：MTE2 队列发射序 == 消费序（`Q(t),K(0),K(1),…,K(eff-1),V(0),P(0),V(1),P(1),…` 任务内全序）⇒ SetFlag 队列位置事件 == 事实上的 per-buffer 事件（WaitFlag(id) 放行 ⟺ 该 id 的 Set 位置前缀全部完成）。**Set-after-last-operand-copy 规则**：每次 gemm 调用的 Set 置于其全部操作数 copy 之后（G1: Q 在 K 前发射由 FIFO 覆盖；G2: Set idG2 置于 P(idx) copy 后覆盖 V(idx)+P(idx)）。
- **生产先例（判决性）**：form_a 域 SIG_K_L1 模式（L935-943：copy K → set_flag(MTE2,MTE1,id) → 消费点 wait_flag(MTE2,MTE1,id)）= 调用方管理 per-buffer MTE2_MTE1 同步，**同 kernel C 核生产在跑** ⇒ 位置语义 + 1:1 令牌纪律双重实证，A10 非新机制发明，是 form_a 模式向模板 opt-in 通道的移植。
- **旗标矩阵（MTE2_MTE1 type，id 0-5 可用域内取 0/1/2；6/7 文档禁用）**：id0 = K 槽0（SIG_KDB0）/ id1 = K 槽1（SIG_KDB1）/ id2 = G2 操作数组合（SIG_VP，V+P 由 FIFO 前缀单旗标覆盖）。G1 消费点奇偶展开绑定编译期 id；G2 单点 id2。**每 id 严格 1:1 交替论证（含全部边界）**：批内 pre-call 发射 K(idx+1)+Set id((idx+1)%2)，其 Wait 在 G1(idx+1)（下一次同 id Set 之前）；任务头 priming（t==0: Q(0),K(0)+Set id0）、任务尾 priming（t+1<my_count: Q(t+1),K'(0)+Set id0——**Q 必须先于 K' 发射**，否则 Set id0 前缀不含 Q(t+1) ⇒ G1'(0) 竞态读 q_l1，FIFO 序不变量的直接推论）；轮边界由批内全局守卫 `idx+1 < eff_kv_iters` 自然跨越（S2 的轮尾 K priming 点在此设计下**冗余可删**——发射序仍在消费序前、且与轮尾 SEM_WS2_C2V(MTE2-bound) 相隔整个 G2 批（~18us ≫ 1.674us 装载）⇒ #R6-2⑦ 无违例）；V 轮头 priming = G1 批尾（gemm1(i==batch_iters-1) 返回后）发射 V(idx0)（**必须晚于本轮全部 K 发射** = FIFO 序不变量；v_l1 释放 = 上一 G2(last) epilogue，早已完成）；G2 批内 V(idx+1) after-return 位（i+1<batch_iters 局部守卫；v_l1 单缓冲 ⇒ 发射点不可早于 G2(idx) 调用 = epilogue 释放序，V 侧深度 1 为预算内可达上限）。
- **G2 相残差（如实预估）**：v_l1 单缓冲 + P(idx) 依赖 SEM_WS2_V2C（跨核协议位不动）⇒ V(idx+1) 于 G2(idx) 退出后入队，暴露 ≈ 1.674 − (SEM 阻塞段重叠 + P 发射 0.1)；wait_id2 1.28us/槽 被 V 装载重叠吸收的现行结构保留 ⇒ G2 相改善为部分（协议地板由 V 核吞吐决定，A10 不触碰）。**wall 收益主体来自 G1 相（暴露 1.57us→~0）+ G2 相部分**；若实测 duty 仅部分兑现（V 单槽串行残留），升级候选 = bN=128 缩块 + k/v 双侧 DB（268KB）+ 同模板模式（任务书第二实施臂条款，按预算纪律裁决）。
- **退化路径全覆盖**：eff=1（causal）：head priming 仅 K(0)+Set，批内守卫全假，G2 单槽——1:1 平衡 ✓；尾块（Skv%256≠0）：extent 公式逐字继承 S2（valid_n 钳位 + needClear=(dst_offset==0) 规则 + r11 SELF-CHECK⑥ 掩码安全性证明整体继承——侧1 槽尾块 stale 行 → select 替换制无 NaN 通路，T_tail300 探针在案）✓；kv_iters=2（case15/18 本体）全序列手工走查平衡 ✓（head: Q,K0+Set0 → G1(0)[Wait0]；iter0 pre-call: K1+Set1 → G1(1)[Wait1]；iter1 守卫假；G1 批尾 V(0)；G2(0): P0+Set2→[Wait2]、V(1) after-return；G2(1): P1+Set2→[Wait2]；轮尾 set WS2_C2V；任务尾: Q',K'0+Set0——id0: Set,Wait,Set,Wait ✓ id1: Set,Wait ✓ id2: Set,Wait,Set,Wait ✓）。
- **L1 预算（继承 r11 实编验证）**：q8 + k[2]×128 + v128 + p8 = 400KB ≤ 511.75KB（r11 实编读数 q0/k8192/v270336/p401408 末端 409600B）✓；A10 相对 S2 零新增缓冲。
- **位级安全性论证（BITEXACT 预期）**：变更全集 = ①copy 发射点移动（程序序重排，数据/extent/目的槽逐 copy 不变）②新增 Set/Wait 旗标对（纯时序）③gemm_v0 模板 prologue 等待对象切换（排空→前缀旗标，算术零触碰）④G1 消费点奇偶 if/else 展开（双实例化同算术）。mma 序/累加序/k-step 序/FIX 写出/掩码链/SEM 协议位全部不变 ⇒ **纯调度变更 ⇒ BITEXACT vs perf16**（r11 22/22 探针档案可继承 + 新同步形态全拓扑扩展）。新增风险两类及缓解：旗标失衡→挂死（缓解：双射论证「Set↔copy(idx) 守卫 idx<eff」×「Wait↔gemm(idx) 执行域 idx<eff」同域同守卫 ⇒ 构造级 1:1 + 生成码审计 + smoke 先行纪律）；FIFO 序违例→陈旧操作数竞态（缓解：发射序=消费序不变量 + Set-after-last-operand-copy 规则 + 边界全走查 + 探针 kv{2,4,跨轮}/尾块/causal 全拓扑）。
- **重排税风险（r11 归因(2) 继承，如实登记）**：A10 使 K 流与 V/w2 流真实时间交叠（S2 已实测同字节 busy 速率 −2.7%）；但 A10 与 S2 的本质差异 = 交叠伴随**间隙消除**（S2 交叠仅移动字节位置、间隙不变 ⇒ 纯税；A10 间隙 27.9% wall 被流式填充 ⇒ 税为二阶项）。判决以机制判据 (a) 钳制：**wall 收益必须伴 duty 68.5%→85%+；duty 升而 wall 零响应 ⇒ 吸收分析落档**（r8-1 纪律）。

**A12-7 GO/NO-GO 判决**：
- NO-GO 判据逐项核对：①默认路径不可保字节级不变？**否**——A12-3 双重证明（f-string 条件拼接 + 默认模板参 if constexpr 折叠）+ 16/17 IR 守卫机器核验通道在案；②事件 id 约束不容？**否**——A12-2：核内 API 无 ≤15 条款（该条款属跨核 CrossCoreSetFlag）、id 0-5 可用、设计 outstanding ≤1 对 1-bit/计数器双语义免疫；③生效链不可达且无落档方案？**否**——A12-4：本地即时生效实机核验 + pip 打包携带模板 + 复评重打包方案落档；④时序设计存在不可消除 hazard？**否**——A12-6 hazard 全表闭合（槽释放=epilogue 不动、q_l1 跨任务=任务尾 Q 位、FIFO 不变量、旗标双射、#R6-2⑦ 核验、退化路径全覆盖）。
- **判决：GO（进入 r12-2 Phase B+C）**。
- 未验证项如实声明：①核内旗标 1-bit vs 计数器硬件语义无文档明文（设计以 1-bit 最强假设自约束，outstanding ≤1）；②重排税在真实交叠下的量级（仅 S2 间接证据 −2.7% busy 速率）；③G2 相 SEM 协议地板下的实际 duty 兑现比例——三项全部由 r12-2 msprof 实测裁决，无一项依赖预测性宣称。

[DESIGN-A10]（Phase B 实施蓝图，四文件改动面 + 审计矩阵）:

**D1 框架改动 1/2 —— src/tl_templates/ascend/common.h（改前备份 /tmp/opencode/r12_framework_backup/common.h.pre_r12 = d113a6c6）**：
```cpp
// 签名（L1185-1193）：追加第 9 默认模板参
template <typename T1, typename T2, uint32_t M, uint32_t N, uint32_t K,
          bool transpose_A = false, bool transpose_B = false,
          uint32_t kL0Size = 128, int32_t kMTE2Sync = -1>
// 函数体内新增 static_assert（kMTE2Sync == -1 || (0 <= kMTE2Sync && kMTE2Sync <= 5)，
// 注明 AscendC 静态 Tensor 场景 id 6/7 禁用条款出处）
// prologue（L1306-1307）改为：
if constexpr (kMTE2Sync >= 0) {
  // opt-in：调用方在其操作数 copy 之后已发 SetFlag<MTE2_MTE1>(kMTE2Sync)
  // （FIFO 前缀语义）；此处仅 wait，不自 set ⇒ 调用后发射的预取不被收编。
  WaitFlag<HardEvent::MTE2_MTE1>(kMTE2Sync);
} else {
  SetFlag<HardEvent::MTE2_MTE1>(L0AB_EVENT);   // 原路径逐语句保留
  WaitFlag<HardEvent::MTE2_MTE1>(L0AB_EVENT);
}
// FIX_M 对（L1308-1309）、k-step 链、epilogue（L1364-1370）零触碰
```
配套注释：opt-in 契约（调用方 Set 必须晚于该调用全部操作数 copy；同 kernel 内 opt-in 与默认模式混用时调用方 id 应避开 L0AB_EVENT=0 的默认 MTE2_MTE1 自 set——mha kv_db 域全 opt-in 无混用）。

**D2 框架改动 2/2 —— tilelang/language/ascend.py（改前备份 /tmp/opencode/r12_framework_backup/ascend.py.pre_r12 = 44cfef09）**：
- `def gemm_v0(..., n_actual=None, mte2_sync=None)`；docstring 增补 opt-in 语义/契约/id 域 [0,5]。
- 断言：`mte2_sync is None or (isinstance(int) and 0 <= mte2_sync <= 5)`。
- 模板串条件拼接：None ⇒ 原 f-string 逐字符不变；非 None ⇒ `f"gemm_v0<..., {kL0Size}, {mte2_sync}>"`。
- 运行期实参（6 个）/op 注册/codegen **零触碰** ⇒ 零 rebuild。

**D3 算子侧 —— examples/mha/mha.py kv_db 分支（S2 骨架 mha_r111_edit_v1.py 89644d68 之上重建；改前备份 history_version/mha_impl_r4_perf16.py = dda3ad08 已落）**：
- host：`kv_db` 谓词不变（S<=16 ∧ D==256 ∧ ¬single_kv_block）；新常量 `SIG_KDB0=0 / SIG_KDB1=1 / SIG_VP=2`（MTE2_MTE1 专用，kv_db 域内模板默认 id0 自 set 不存在——全 gemm opt-in）。
- C 侧 kv_db 分支 site 表（发射序=消费序不变量约束下的完备集）：
  | # | 位置 | 动作 | 守卫 |
  |---|---|---|---|
  | S0 | 任务头（t==0） | Q(0) copy（现位）→ K(0) copy→slot0 → `set_flag(MTE2,MTE1,SIG_KDB0)` | t==0（t>0 由 S4 覆盖）|
  | S1 | G1 批内 iter idx **pre-call**（w1(idx-1)+SEM_WS1 之后、gemm1 调用之前）| K(idx+1) copy→slot (idx+1)%2 → `set_flag(MTE2,MTE1,SIG_KDB0/1 按 (idx+1)%2)` | idx+1 < eff_kv_iters（全局守卫，跨轮自然覆盖 ⇒ S2 轮尾点删除）|
  | S2 | G1 消费点 | `if idx%2==0: T.gemm_v0(q_l1,k_l1[0],...,mte2_sync=SIG_KDB0) else: ...k_l1[1],...,mte2_sync=SIG_KDB1`（transpose_B/init/kL0Size 逐字不变）| — |
  | S3 | G1 批尾（gemm1(i==batch_iters-1) 返回后、w1 写出前）| V(idx0) copy→v_l1（无 Set——其 Set 在 G2 iter0 的 S6）| i==batch_iters-1 |
  | S5 | G2 批内 iter idx | SEM_WS2 wait（现位）→ P(idx) copy（现位）→ `set_flag(MTE2,MTE1,SIG_VP)` → `T.gemm_v0(p_l1,v_l1,...,mte2_sync=SIG_VP)` → w3 写出+SEM_WS3（现位）→ V(idx+1) copy→v_l1 | V 预取守卫 i+1 < batch_iters（局部：轮头 V 由 S3 覆盖，防双载）|
  | S4 | 任务尾（set WS2_C2V 之后，#R6-2⑦）| **Q(t+1) copy（先）** → K'(0) copy→slot0 → `set_flag(MTE2,MTE1,SIG_KDB0)`；bz'/by'/valid_m' 纯标量重算（S2 先例）| t+1 < my_count |
- Q 载入位重排（S0/S4 配套）：任务头 Q copy 加守卫 t==0（S2 原为每任务头载入；A10 下 t>0 的 Q 由 S4 任务尾预载——**若不重排则 S4 的 Set id0 前缀不含 Q(t+1) ⇒ G1'(0) 竞态**，FIFO 不变量强制）；q_l1 释放 = 任务 t gemm1(last) epilogue（G1 相先于 G2 相 ⇒ S4 发射时早已释放）✓。
- 零触碰域：GEMM2 SEM 协议位/emit_v 全体/single_kv_block 分支/form_a/form_b_mma/host 包装/工作区形状/test_mha.py/DESIGN.md/提交副本。
- 预取 copy extent/needClear 公式逐字继承 S2（r11 SELF-CHECK⑥ 审计整体适用）。

**D4 Phase B 门禁与测量计划（r12-2 执行清单）**：
1. 备份+git diff 落档（common.h/ascend.py 每行 diff 全文进 optimization_log）；
2. 框架构建：ast/ruff + 最小 smoke（单 gemm_v0 UT 4 件 = 默认路径回归）；
3. mha.py 变体构建（先落 /tmp/opencode/r12_probes/ 变体副本，基线采集后上位——r11 缓存纪律同款）；
4. **16/17 shape IR 守卫**（dump 基础设施 dump_base_r11.py 复用；预期 15 逐字节 + G_case15/T_case18 变更 + AIV 主块逐字节）= 默认路径不变机器证明；
5. L1 实编读数门（GetWithOffset 预期与 r11 全同：q0/k8192/v270336/p401408）+ 事件 id 用量审计（生成码 grep：MTE2_MTE1 id ∈ {0,1,2}，无 6/7）+ 生成码旗标双射审计（Set/Wait 站点计数 = 守卫同构）；
6. **smoke 先行**（会话 10 HANG 纪律）：case15_like 链 + kv{2,3,4} + 跨轮 + causal eff=1 + 尾块 300 + 任务链 my_count{1,2}，超时纪律全程；
7. 精度全门禁：探针 22/22+12/12 BITEXACT vs perf16（r11 档案继承 + 新同步拓扑扩展）→ 锚-A 9/9 + 锚-B 14/14 → 压力（继承 4 + kv_db 被改路径专项扩展 ×12 nondet=0）→ l0/l1/coverage/all；
8. 性能：同会话新鲜基线（perf16 态）→ 变体 case15/18 median n=25 + **msprof op 模式 duty 签名**（机制判据 (a)：duty 68.5%→85%+ 伴 wall ≥3%；mte2 busy 速率重排税读数；wait_id1/id2 变化）+ 守卫 case10/11/14/16/20/7/9 带内核验；
9. 判决：≥3% 且签名兑现 → Phase D 收口（r12-3）；零响应/劣化 → 三重回滚（common.h/ascend.py 还原 + git diff 清零核验 + mha.py 还原 md5 核验）+ 归因落档 + 转 A9 前置推导（纯分析）。
10. 失败分类沿用阶段规则（性能提升采纳/精度退化回滚/性能下降回滚/无变化回滚/运行失败回滚）；每门禁单项反复失败 >3 次 → 结构性受阻落档。

**A12-8 Phase A 收口**：零生产变更核验——mha.py = dda3ad08（perf16 备份同体）✓；common.h/ascend.py git diff = 0 行 ✓；备份三件套落位（common.h.pre_r12 d113a6c6 / ascend.py.pre_r12 44cfef09 / mha_impl_r4_perf16.py dda3ad08）✓；本轮无测量（Phase A 定义），r12 新鲜基线于 r12-2 采集（baseline_r12_iter1.json = 占位引用 r11 参照系）。迭代计数：r12-1 = 全局 iter30（分析轮，consecutive_no_improvement 计数不变 = 3，iter24-26 纯测量轮不递增口径延续）。

### 迭代 r12-2：Phase B+C —— [#R12-2] A10 实施 + 测量 + 性能判决（2026-10-04/05）

[ORDER-CHECK] 准备实施: [#R12-2] A10 gemm_v0 模板级 per-buffer 同步改造（框架 opt-in 通道 + mha.py kv_db 域发射序重设计） | 前置依赖: [用户明示授权框架级改动（2026-10-04「先尝试A10」）✅；r12-1 Phase A GO 判决（[ANALYSIS]-S12 四项 NO-GO 判据逐项否证 + [DESIGN-A10] D1-D4 蓝图）✅；框架改前备份三件套落位（common.h.pre_r12 d113a6c6 / ascend.py.pre_r12 44cfef09 / mha_impl_r4_perf16.py dda3ad08）✅] | 结论: ✅

[IMPL-#R12-2] 已阅读: common.h:1183-1371（gemm_v0 模板全文，prologue/k-step/epilogue 三段结构 + L0AB_EVENT 定义）、ascend.py:418-531（前端全函数 + f-string 模板串构造点）、codegen_ascend.cc:2714-2736（GemmOpCodegen 拼串不解析）+ 2697-2704（FlagOpCodegen 运行期 id 发射）、ascend.cc:1298（set_num_inputs(6)）、operation_config.h:164（op 名识别）、env.py:177-182 + libgen.py:168（TL_TEMPLATE_PATH 生效链）、setup.py:408-417（tl_templates 打包）、mha.py:1700-1886（decode 兜底域全文）、mha_r111_edit_v1.py:1806-1913（S2 kv_db 分支 = 改造骨架）、gate0b_verdict_iter25.md:40-105（≤15 条款出处 = CrossCoreSetFlag）、RAG SetFlag/WaitFlag(ISASI) 官方文档（eventID 0-7 / 静态 Tensor 场景 6,7 禁用 / 成对出现约束原文）、form_a SIG_K_L1 生产先例（mha.py:930-945 per-buffer MTE2_MTE1 调用方管理模式）。关键约束: ①框架改动 = 通道 (b) header-only（common.h 第 9 默认模板参 `int32_t kMTE2Sync=-1` + prologue `if constexpr` 分支 + static_assert id∈[0,5]∪{-1}；ascend.py `mte2_sync=None` kwarg + 条件模板串拼接 + 断言）——**零 C++ 改动零 rebuild**，默认路径逐字节不变双重证明（f-string 条件拼接 + 默认模板参折叠）；②发射序=消费序 FIFO 不变量 + Set-after-last-operand-copy 规则 + 每 (type,id) 严格 1:1 交替 outstanding≤1（对 1-bit/计数器双语义免疫）；③**pre-call 发射点**（A12-6 关键修正：S2 after-return 位在 per-buffer 同步下仍时序中性——发射点=标量退出点；K(idx+1) 必须于 iter idx 标量前段发射才获得整槽流式窗口）；④G1 消费点奇偶展开绑定编译期 mte2_sync（运行期模板参不存在）；调用方 Set 用运行期 id（form_a SIG_L0AB+side 先例）；⑤Q(t) 重定位：任务头 t==0（S0）/ 任务尾 Q(t+1) 先于 K'(0)（S4，FIFO 前缀必须覆盖 G1'(0) 双操作数）；⑥G2 域：SEM_WS2 wait/P copy 位置逐字不动，Set SIG_VP 于 P copy 后（FIFO 前缀覆盖 V+P），V(idx+1) after-return 预取（v_l1 单槽 = 预算内可达上限）+ S3 轮头 V priming 于 G1 批尾（晚于本轮全部 K 发射）；⑦S1 全局守卫 idx+1<eff_kv_iters 跨轮覆盖 ⇒ #R11-1 轮尾 K priming 点冗余删除；⑧#R6-2⑦：S4 位于 set WS2_C2V 之后且发射时 MTE2 队列已空（G2 相 ~18us ≫ 装载 1.674us）⇒ w2 释放零延迟；⑨extent/needClear 公式逐字继承 S2（r11 SELF-CHECK⑥ 掩码安全性证明整体适用）；⑩SEM 矩阵/emit_v/single_kv_block/form_a/form_b_mma/host 包装零触碰；⑪缓存纪律：r12_cache_{base,var,g1,final,consumer} 全程专属 + disable_cache dump；⑫smoke 先行（会话 10 HANG 纪律）。

[SELF-CHECK]（变体 = /tmp/opencode/r12_probes/mha_r122_edit_v1.py，e8a9d2da，2120 行，S2 骨架上 2 区域重构：host 常量区 +12 行（SIG_KDB0/1/SIG_VP + 设计注释）/ kv_db 分支 102→152 行；ast OK + ruff check "All checks passed!" + ruff format clean）:
①**默认路径逐字节不变机器证明（本次最强守卫）**：base dump（生产 dda3ad08 + 新框架，disable_cache，r12_cache_base）→ /tmp/opencode/r12_src_base **17/17 逐字节 = r11_src_base（旧框架参照）**（diff -rq exit 0 + _manifest 全同）⇒ 框架改动对全部现有消费者生成源零扰动；
②变体 IR 守卫 17 shape：**15/17 逐字节相同 + 2/2 目标按预期改变**（G_case15 9854→11092B / T_case18 9912→11152B）+ **AIV 主块逐字节相同**（base L83-170 == var L105-192 = emit_v 零触碰 IR 级证明）；
③生成码审计（G_case15 + kv10 多轮形）：gemm 调用 = `gemm_v0<half, float, 16, 256, 256, false, true, 64, 0/1>`（G1 奇偶展开双实例）+ `<..., false, false, 128, 2>`（G2）✓；用户 SetFlag<MTE2_MTE1> 站点 = S0(0)/S1 运行期 `((i+1)%2)`（case15 守卫折叠形 `(i+1)` 合法：i<1 恒 0）/S4(0)/S6(2)——**全部 id ∈{0,1,2} ⊂ [0,5]，无 6/7** ✓；kv10 跨轮核验：S1 全局守卫 `((k*8)+i)<9`、轮 1 G1(8) mte2_sync=0 与轮 0 iter7 的 SetFlag(0) 精确配对、id0/id1 严格交替无 dangling ✓；S3 `if(i==condval-1)` V 轮头 priming ✓；S5 `if((i_1+1)<condval_3)` 局部守卫（无 V 双载）✓；S6 SEM wait→P copy→Set(2)→gemm 顺序 ✓；S4 Q-before-K' 顺序 ✓；CrossCore SEM set/wait 全部位置与 base 逐字相同 ✓；
④L1 实编读数：q@0B(8KB) / k@8192B(256KB 双槽 stride 65536 halfs) / v@270336B(128KB) / p@401408B(8KB) 末端 409600B = 400KB ≤ 524032B（与 r11 变体全同 = 零新增缓冲）；L0A/L0B 65536、L0C 131072、UB 196352 池与 base 逐字相同 ✓；needClear 审计：K 侧 S0/S4（slot0 静态偏移 0）=true、S1 运行期槽=false（r11 SELF-CHECK⑥ 掩码安全性证明继承）、V/P 侧全 true ✓；
⑤旗标双射静态论证 + 生成码实证（A12-6 全边界走查：kv=2/eff=1 不可达性证明（kv_db 域 S_kv>256 ⇒ eff≥2）/奇偶批/跨轮/任务链/my_count∈{0,1}）——smoke 6/6 无挂死为运行时确证；
⑥框架 diff 全文落档（见 [RESULT-#R12-2] 附录；git diff 稳定性核验 = 与 /tmp/opencode/r12_framework_backup/framework_r12_phaseB.diff 逐字相同）；
⑦smoke 先行 **6/6 BITEXACT vs perf16 无挂死**（case15_like_chain/kv10_xround/kv3_odd_batch/tail300_side1/min_kv2_tail1/case15_orig——覆盖任务链 S0/S4、跨轮 S1、奇数批 S3/S5、side-1 needClear=false 尾块、1 核最小形、bench 原形 48 任务链）。

[RESULT-#R12-2] 优化点: **A10 gemm_v0 模板级 per-buffer MTE2_MTE1 同步改造（框架 opt-in：common.h 第 9 默认模板参 kMTE2Sync + ascend.py mte2_sync kwarg，默认路径逐字节不变）+ mha.py kv_db decode D256 域发射序重设计（S0-S6 七站点：pre-call K 预取（全局守卫跨轮）+ 奇偶展开 G1 消费点（SIG_KDB0/1）+ G2 V/P FIFO 前缀旗标（SIG_VP）+ V after-return 预取 + S3 轮头 V priming + Q 重定位 S0/S4）** | 精度: **pass（全门禁过，采纳态 = docstring 注记后 0fb193b2，IR 与测量态同一性核验 True）**——①smoke 6/6 BITEXACT 无挂死；②探针 probe_r122_kvdb **22 目标 + 12 守卫全 BITEXACT vs perf16**（PROBE RESULT: PASS，0 DIFF；矩阵 = r11 档案继承：case15/18 原形、kv{2,3,4}、kv{9,10,16} 跨轮、causal×3、尾块 640/300、bf16×2、scale、大幅值、my_count{1,2,mixed}、最小 1×1、zeros + golden 阻塞口径）+ PART 0 分派审计（22 目标全落 kv_db / 12 守卫全不落）；③锚-A **9/9 BITEXACT** + 锚-B **14/14**（BATTERY_RESULT: PASS）；④压力：继承 4 配置 ×12 nondet=0（PASS）+ 扩展 17 项 kv_db 被改路径专项 ×12 nondet=0（STRESS_RESULT: PASS）；⑤l0 **10/10** + l1 **19/19** + coverage **19 PASS/0 MISS** + `--level all` **ALL_EXIT=0**（哨兵 4/4 PERF_PASS 含 perf_d256_bf16）；⑥**非 mha 消费者 smoke 90/90 passed**（testing/python/language/test_tilelang_ascend_language_gemm_v0{,_k_tail,_n_actual,_n_tiling}.py = 44+32+8+6，默认路径真机行为回归，任务书 §0-e 满足）；⑦红线四项（r16 双舍入栅格链/单块预归一特化/sumexp+1e-30/掩码链）全部位于 AIV 主块 = IR 逐字节相同域 ✅；⑧ruff check/format：mha.py + ascend.py 全 clean；⑨docstring 注记后复验：l0 10/10 + case15 IR 与测量态逐字节同一（md5 e5a5f0c8）。 | 性能(msprof median us, n=25, 同会话新鲜基线 r12_base，warmup5/iters20，--task-time=on 口径): **case15 508.11→463.58（−8.76%）** / **case18 508.57→465.36（−8.50%）**——stdev 1.16/1.03（Δ>20σ），双目标同向；守卫 7 case：case10 −0.69% / case11 +0.52% / case14 −0.13% / case16 −0.07% / case9 +0.23% / case7 +0.11% 全带内 + **case20 +2.81%（T_case20 IR 逐字节相同域 ⇒ 漂移带归因，session-8 case20 先例口径 + r11 case10 先例；14us 小核发射抖动敏感，其余 6 守卫 ≤±0.7% 证环境稳定）** | op 归因（case15，r12_base/op vs r12_iter1/op，机制判据 (a)）: **机制激活、签名部分兑现**——**duty 68.3→73.7%（+5.4pp）**；mte2_t 345.71→341.75（同字节 25920KB ⇒ **busy 速率 74.98→75.84 GB/s/核 = 零重排税**，与 r11 S2 的 −2.7% 对照：交叠伴间隙消除时税为二阶）；**C.scalar_t 377.21→335.70（−41.51）≈ C.wall −41.55 = wall 收益主体**（标量为 pacing 管，气泡压缩实体）；mte1_stall 355.13→318.71（−36.42）；cube_t 49.80→49.80 / V.vec_t 82.64→82.64 / V.mte3_t ±0.05 / GM 字节 ±0（**算术与字节精确守恒 = BITEXACT 一致**）；mte1_t 130.50→134.75（+3.2% = L1 读与 MTE2 写并发二阶项，如实记录）；fix_t 66.30→61.37（−4.93，引擎计时波动如实记录）；wait_id2 122.31→119.92（SEM 协议地板不动 ✓ 设计预期）；V.wait_id0 242.26→200.19（−42.08 = C 交付加速的协议阴影收敛）；taskdur 506.32→464.84 | 对比: **−8.76%/−8.50%（≥3% 门 2.9 倍）** | 采纳: **YES —— A10 采纳（本谱系首个框架级采纳态）**。duty 部分兑现归因（任务书预案字面命中：「V 单槽串行残留」）：残差主体 = G2 相 SEM_WS2 协议窗内 MTE2 无可发流（v_l1 单槽 + epilogue 释放序 ⇒ V(idx+1) 不能早于 gemm2(idx) 调用发射；L1 预算 400/511.75KB ⇒ bN=256 下第二 V 槽不可入）——**升级臂（bN=128 缩块 + k/v 双侧 DB 268KB + 同模板模式 = S1 在 A10 解锁后复活）按预算纪律移交下一会话**（r12 软预算 4：r12-1/2/3 已耗 3，Phase D 收口为当前迭代；升级臂 = S1 级全形重构（bN 涟漪：V 侧 block_N 参数化域/工作区形状/SEM 轮计数/探针矩阵重基线），需独立 Phase-A 级设计与预算，不宜挤压）。备份链：perf16（改前 dda3ad08）/ perf17（采纳态 0fb193b2 同体）；变体中间态留存 /tmp/opencode/r12_probes/mha_r122_edit_v1.py（e8a9d2da，2120 行 = 测量态）。

**[RESULT-#R12-2] 附录 A —— 框架改动 git diff 全文（任务书 §0-c 落档条款；备份 /tmp/opencode/r12_framework_backup/{common.h.pre_r12,ascend.py.pre_r12,framework_r12_phaseB.diff}）**：
- common.h：+34/−3 行（模板签名 1 行改 + 契约注释 21 行 + static_assert 4 行 + prologue if constexpr 12 行替换 2 行原语句——原语句在 else 分支逐字保留）
- ascend.py：+33/−2 行（签名 1 行拆 2 + docstring 16 行 + 条件拼接/断言 17 行替换 1 行 f-string）
- 合计 +67/−5（git diff --stat：2 files changed, 67 insertions(+), 5 deletions(-)）；**默认路径不变证明**：①mte2_sync=None ⇒ 模板串逐字符 = 历史 8 参形（拼接重构逐字符核验 + 17/17 IR 机器证明）；②8 参显式调用 ⇒ 第 9 参取默认 -1 ⇒ if constexpr 折叠为原 2 语句（设备码逐语句同）；③ascend.cc/codegen 零触碰 ⇒ op 注册/发射形态零漂移；④90/90 非 mha 消费者 UT 真机回归。diff 原文见 framework_r12_phaseB.diff（124 行含上下文；git diff 稳定性 = 采纳态复核逐字相同）。

**[RESULT-#R12-2] 附录 B —— 生效链方案落档（cann-bench 复评前置，Orchestrator 执行项）**：
- 本地生效：TL_TEMPLATE_PATH = 源码树 src/（env.py L177-182 develop 回退实机核验）⇒ 已完成，本轮全部测量即此态。
- **评测环境生效（采纳态复评必需）**：①`build_wheel_ascend.sh` 重打包（wheel 携带 tilelang/language/ascend.py + tilelang/src/tl_templates/ascend/common.h——setup.py L408-417 打包清单已含，**无需打包脚本调整**）；②评测环境重装 wheel；③清理/隔离评测侧 TILELANG_CACHE_DIR（opt-in shape 模板串变化 ⇒ cache key 必变自动重编译，默认 shape 缓存命中旧 .so = 默认路径逐字节等价 ⇒ 行为不变；保守起见建议新鲜缓存）；④复评预期：case15/18 谱系收益 −8.5~−8.8%（M1 线 0.4 权重的 decode D256 双 case），其余 shape 零变化（IR 逐字节）。**若评测环境不能重装框架 ⇒ A10 收益不可复评 ⇒ 采纳态提交必须框架+算子成对**（提交副本同步 = Orchestrator 轮末统一，本 agent 未触碰 mha_submission/）。
- A9 前置推导：**未触发**（A10 成功采纳，失败分支条款不激活）。

**收口判定（会话 12）**：迭代 3/软预算 4（r12-1 Phase A + r12-2 Phase B+C + r12-3 Phase D 收口）；**采纳 1（A10，−8.76%/−8.50%，本谱系首个框架级采纳）/ 回滚 0**；**consecutive_no_improvement = 0（重置：r9-1/r10-1/r11-1 计数链由本次采纳中断）**；design_error = no（框架可改造性经 Phase A 四判据 + Phase B 实证双重确认）。会话 12 净知识产出：①核内 SetFlag/WaitFlag(ISASI) 约束全文入库（id 0-7/静态 Tensor 场景 6,7 禁用/成对出现——「≤15 set」属跨核 CrossCoreSetFlag，gate0b 出处核对闭合）；②通道 (b) header-only plumbing 模式入库（模板串 = Python f-string 全文粘贴 + codegen 不解析 ⇒ 默认模板参扩展零 rebuild，17/17 IR 机器证明方法学）；③**守恒律破解条件精确化**：per-buffer 同步只消除「排空收编」，不改变发射点——收益必要条件 = 发射点前移（pre-call）使装载获得整槽流式窗口（S2 after-return 位时序中性的机制解释）；④FIFO 序不变量工程化（发射序=消费序 + Set-after-last-operand-copy + 每 id 严格 1:1 交替 outstanding≤1 ⇒ 对 1-bit/计数器双语义免疫——核内旗标硬件语义未验证项的鲁棒化处置）；⑤重排税条件（r11 −2.7% vs r12 +1.1%：交叠是否伴随间隙消除决定税的符号量级）；⑥生效链条款（框架级采纳 = wheel 成对分发，评测复评前置方案）。终态 = **0fb193b2（2152 行）+ 框架 common.h/ascend.py opt-in 参数（git 工作区态，未 commit——任务书 §0-d 禁触条款，仅用户明示才提交）**；cann-bench 复评与提交副本同步留 Orchestrator。提交副本 mha_submission/ 未触碰 ✓；test_mha.py/DESIGN.md/proto.yaml 零改动 ✓；.orchestrator_state.json 未读写 ✓。

### 迭代 r13-1（全局 iter33）：Phase A —— [ANALYSIS]-S13 / [DESIGN-S13] 升级臂全形设计（bN=128 缩块 + k/v 双侧 L1 DB，零生产变更）（2026-10-05）

[ORDER-CHECK] 准备实施: [#R13-1] Phase A 全形设计（升级臂 = r12 移交的 S1 复活形态：decode D=256 kv_db 域 bN 256→128 + k/v 双侧 L1 DB）| 前置依赖: [用户明示立项（2026-10-05「执行bN=128 k/v 双侧 DB方案」）✅；开工 md5 双端核验（mha.py 0fb193b2 = perf17 同体 2152 行 ✓；common.h 82e80025 = pre_r12(d113a6c6)+framework_r12_phaseB.diff 补丁复现逐字节一致 ✓；ascend.py 1c3a00ed 同法核验 ✓）✅；r12 移交条款（[RESULT-#R12-2] duty 残差归因 + S12-5 升级臂蓝图要点）✅] | 结论: ✅（Phase A 分析轮，零生产变更；框架文件默认不动——本臂纯算子侧表达，id 预算审计见 S13-3）

[IMPL-#R13-1] 已阅读: common.h:1183-1402（gemm_v0 现态全文：kMTE2Sync opt-in 分支 L1329-1338 / N-tiling nMaxByL0B=32KB÷(kL0×sizeof) L1286-1290 / nTile 整除 static_assert L1288 / kL0Budget=64KB÷(kNumSteps>1?2:1) L1297-1298 / B-tile L1301+A-tile L1305 static_assert / k-step 乒乓 L1345-1394 / epilogue MTE1_MTE2+M_FIX L1398-1401）、mha.py:220-334（host tiling 三档派生 + kv_db 谓词 + SIG 常量区 + emit_v 签名）、mha.py:334-861（emit_v 三形态全文：single_kv_block 批处理/flat_v/generic——generic else 分支 L703-852 = kv_db 域 V 侧现行路径）、mha.py:1756-2112（decode 兜底 prim_func 全文：alloc 区 L1768-1797 + single 分支 + kv_db 分支 S0-S6 L1860-2027 + generic else L2028-2078）、G_case15_B_d256_dec.cc（r12_src_var 生成码：S0-S6 站点行级定位 L61-108）、r12 档案全链（[ANALYSIS]-S12 A12-1~A12-8 / [RESULT-#R12-2] 含 op 全管读数 / baseline_r12_iter2.json / final_report §S12 全节 + S12-7 权威复评）、r11 档案（[RESULT-#R11-1] 归因 (1)-(5)：守恒律/重排税/S1 对照推演/needClear 规则/DB 位级安全、[RESULT-#R11-2] S3 三门 + 规则内穷尽声明）、analysis_20261003 §6.2（decode D256 三路线判决：bN=128 缩块 = 唯一规则内可表达候选）、#R4-3（协议税坍缩 ~0.2-0.3us/轮 + 收益不可叠加性）、#8d/#10a/#10b/#11/#13（bN 档变更史五连：谓词移动/阻塞口径/掩码变体迁移/kL0 三档先例）、#R4-2a（kL0=256 单步 ping-pong 失效 +3.7% 实测否决 + 每-step ~0.15us 标量链）、final_report §S2-3-1（transpose_B L0B 预算规则 = 任务书 "S2-3-1 规则"出处：B tile=N×kL0×2 ≤ 32KB 多步 / 64KB 单步）、p6_battery_anchors.py（锚-A 9 = L0#1-9 全 Form A / 锚-B 14 全 form_b_mma——成员域归属核验）、probe_r122_kvdb.py（22+12 矩阵原形 + plan() 谓词复刻模式）、dump_src_r11.py（17 shape IR 守卫清单）、performance-antipatterns.md L267-299（tile size 条目——本臂为反向缩块，见 S13-6 反模式对照）、r12 测量基建（run_wall_r12.sh/run_baseline_r12.sh/run_perf_r122.sh/parse_op_r12.py/run_gates_r122.sh 全脚本）。关键约束: ①域隔离靠 bN 谓词一处（S≤16∧D=256∧S_kv>256 ⇔ kv_db 域 ⇔ bN=128；单块域 S_kv≤256 保持 legacy bN 逐字节不动）；②k-step 总数守恒定 kL0=128 档（768 steps/core 不变，见 S13-1c）；③事件 id 零新增（SIG_VP 单 id 覆盖 V+P 的 FIFO 前缀语义在双槽下保持，见 S13-3）；④V 预取位 = Set(SIG_VP) 之后、gemm2 调用之前（pre-call 但后于 Set——前缀不含 V(idx+1)，气泡填充 SEM_WS2 协议窗，见 S13-1d）；⑤needClear=false 的 V 槽尾块 stale 有限性归纳证明（S13-4b 红线定位）；⑥锚-A/锚-B 全员域外（成员逐 shape 核验）⇒ 无降级项，BITEXACT 硬守卫保持；⑦非 BITEXACT 预期（online-softmax rescale 粒度 + GEMM2 K 链 256→128）⇒ 主门 = golden 阻塞口径（#8d/#10b 双先例余量 40-640x）；⑧框架零改动（id 预算 3/6 可解 + 模板能力充足 ⇒ 无需触发框架条款）；⑨NUM_STAGES/flat/w 瘦身等死列表项零触碰；⑩缓存保守全新 r13_cache_*（框架未改但 mha.py 源变 ⇒ 键必变，且基线需生产态新鲜编译读数）。

---

[ANALYSIS]-S13（Phase A 六项必答，全部证据落档）:

**S13-1 bN 涟漪全清单（逐项证据/推导）**：

(a) **host tiling 派生**（mha.py L227-291 逐行核对）：
- `block_N`（L228-231）：现行 `S<=16: 512 if (D<=128 and S_kv>256) else (256 if S_kv>128 else 128)`——D=256∧S_kv>256 现给 256；**改法 = 前置域分支 `if D==256 and S_kv>256: block_N=128`**，其余分支表达式逐字保留 ⇒ D≤128 全域 + D=256 单块域（S_kv≤256：bN=256/128 两档）零变化（parse-time Python 常量，域外生成码逐字节的构造性保证）。
- `gemm1_kL0`（L234-238）：现行 64 档条件 `(S<=16 and D==256 and S_kv>128)` 会误罩新域 ⇒ **收窄为 `and S_kv<=256`**（单块 legacy 保持 64）；新域落 else 128 档。
- `kv_iters = ceil(S_kv/block_N)`（L256）：自动翻倍（case15/18: 2→4）；`single_kv_block`（L257）自动保持 False（S_kv>256 ⇒ kv_iters≥3）；`need_bound_mask = S_kv%block_N != 0`（L250）：模基变 128——**640 型翻转 (F,T)→(F,F)**（640%128=0 无尾块 ⇒ 掩码链不生成 = 语义同一，clamp 构造保证：#8d/#10a 谓词移动先例「#4 跳过保证仅尾迭代执行，阈值语义同一」）；300 型保持 (F,T)（300%128=44）。
- `eff_kv_iters` 公式（C L1918-1920 / V L713 同式）：`floordiv(min(s0+bM-1+causal_offset, S_kv-1), block_N)+1`——block_N 原生参数化，C/V 同一闭包常量（§7.2-5 硬约束自动满足）；**causal S≤16 均匀恒等复核**：s0≡0（decode s_blocks=1）、min(15+S_kv-S, S_kv-1) = S_kv-1 ⇔ S≥16；S<16 时 15+S_kv-S > S_kv-1 ⇒ min 仍 = S_kv-1 ⇒ **eff ≡ kv_iters 对全 S≤16 均匀成立（bN 无关）** ✓（r11 档案结论在 bN=128 下重推导成立）。
- 任务映射（L290-293）：block_num=B×N×s_blocks、core_num、q_tasks/r_tasks、my_start/my_count **全部 bN 无关** ⇒ 零变化（case15: 960 块/20 核/48 任务/核不变）。
- `flat_v`（L265）：S≤16 恒 False ⇒ 兜底域不涉 flat（死列表项零触碰）✓。

(b) **C 侧 L1 池形状与 kL0 档**：
- L1 池（新域手算，实编读数门在 Phase B）：q_l1 [16,256]f16 8KB + k_l1 [2,128,256]f16 128KB + v_l1 [2,128,256]f16 128KB + p_l1 [16,128]f16 4KB = **268KB ≤ 511.75KB(524032B)**，余量 243.75KB（对照 r12 现态 400KB）。预期读数：q@0/k@8192/v@139264/p@270336/末端 274432B（Phase B GetWithOffset 门核验，手算只作初筛——S2-3-2 教训）。
- **GEMM1（transpose_B，M=16,N=bN=128,K=D=256）kL0 可行域**：S2-3-1 规则（final_report §S2-3-1 = common.h L1286-1305）：B tile = nTile×kL0×2 = 128×kL0×2 ≤ kL0Budget（kNumSteps>1 → 32KB）⇒ **kL0 ≤ 128**；kL0=128: kL0split=2、B tile 32768=32KB 恰好顶格（≤ 含等号，#10b 256×64×2=32768 顶格先例 + #13 512×32×2 顶格先例）✓；A tile=16×128×2=4KB ✓；K-tail=256-128=128%16=0 ✓（fp16 ELE=16 短路）。kL0=256（kNumSteps=1 → 64KB 预算，B tile 128×256×2=64KB 合法）**被 #R4-2a 实测否决**（单步 ping-pong 失效 +3.7% 回滚档案）⇒ **选定 kL0=128**。
- **GEMM2（M=16,K=bN=128,N=D=256，默认 kL0=128）**：nMaxByL0B = 32768/(128×2)=128 < N=256 ⇒ nTile=128，**整除 static_assert（common.h L1288）256%128=0 ✓**；nL0split=2、kL0split=1、kNumSteps=2 ⇒ 预算 32KB；B tile=128×128×2=32KB 顶格 ✓；A tile=4KB ✓；kL0split=1 ⇒ C0 对齐断言短路 ✓。
- **k-step 总数守恒核算（关键预算结论）**：现行/槽 = G1 4 步（K256/kL0 64）+ G2 4 步（K256/128 × N256/128）= 8 步 ×96 槽 = 768 步/核；新/槽 = G1 2 步 + G2 2 步 = 4 步 ×192 槽 = **768 步/核 精确守恒** ⇒ 每-step 标量链（PipeBarrier<PIPE_M>+flag 往返 ~0.15us，#R4-2a 档案值）总量不变；kL0=64 备选会使 G1 4 步 ⇒ 1152 步/核（+50% ≈ +58us 标量）——**kL0=128 为唯一守恒解**。
- L0C：acc_s_l0c [16,128]f32 8KB（原 16KB）+ acc_o_l0c [16,256]f32 16KB = 24KB ≤ 128KB 池 ✓；L0A/L0B 池 64KB（gemm_v0 内部 TBuf，顶格 32KB×2 槽 ping-pong 恰满）✓。

(c) **V 侧 block_N 参数化域**（emit_v generic 分支 L703-852 + alloc 区 L1780-1797 逐行核对）：
- ws 形状：workspace_1/2 签名 [core,8,16,bN]（w1 f32 8KB/槽、w2 f16 4KB/槽——每槽减半、槽数×2 ⇒ **GM 总字节不变**）；workspace_3 [core,8,16,D] **bN 无关 16KB/槽 ⇒ 总字节×2**（w3 GM 往返 = 本臂唯一净增流量：case15 C 侧 FIX 3072→4608KB/核、V 侧 GM_to_UB 1536→2304KB/核，S13-5 预算）。
- **io3_is_alias 翻转**：bN=128 ≠ D=256 ⇒ io3 由 alias→专属 [2,8,256]f32 16KB + sig_io3 = SIG_IO3(id2, V_MTE2 型)——**该配置已在生产运行**（D256 单块域 S_kv≤128 即 bN=128∧D=256∧非 alias：r12 守卫 G_dec256_single128 (1,1,128,4,256) 在案；D128 decode bN=512 非 alias 同构）⇒ 参数化路径零新代码。r_2d 同理由 alias→专属 [8,256]f32 8KB（同款先例）。
- UB 预算（新域手算）：io_buf 8KB + io3 16KB + work_ub 4KB + buf_2d 4KB + rt_2d 4KB + r_2d 8KB + mask_2d 128B + col_pos 512B + row_thresh 32B + rr_ub 2KB + acc_s_half 2KB + acc_o 8KB + out_half 4KB + m_i 系 ~0.6KB ≈ **61.3KB ≤ 191.75KB**（余量巨大；实编读数门复核）。
- S_HALF/row_thresh/掩码块界：SIG_S_HALF 闭环（L800-808）逐 op 参数化；row_thresh = min(s0+vid*8+r+offset_eff, S_kv-1) **bN 无关**；掩码 runtime-skip 守卫 `idx*bN+bN-1 > min(s0+offset_eff, S_kv-1)` + col_pos AP(j0=idx*bN) **原生 bN 参数化**（#8d 先例③：「掩码变体迁移语义同一」）；mask_2d = bM×bN/8 = 128B。
- **emit_v 三形态域外逐字节**：emit_v 源文零编辑；block_N/single_kv_block/io3_is_alias/need_mask 全为 parse-time Python 常量 ⇒ 域外 shape 的常量元组不变 ⇒ 生成码逐字节（#R6-1 先例机制；IR 守卫 dump 机器证明，S13-6 蓝图 G3）。**目标域 AIV 主块将按预期改变**（w1/w2 半宽、io3 专属旗标、phase2 槽数×2）⇒ r12 的「AIV 逐字节」守卫在本轮**不适用目标域**，改由阻塞口径精度门管辖（如实登记差异）。
- phase1/phase2 工作量核算：phase1 元素总量不变（[8,bN]×2 槽）；**phase2 = bN 无关 D 宽操作 ∝ 槽数 ⇒ ×2**（r_factors exp/sumexp rescale/r_2d 广播乘加 + io3 读）——V 侧净增（S13-5 预算）。

(d) **SEM 轮计数 / NUM_STAGES 批深交互**：槽数/核 96→192（case15）；eff=4 ≤ NS=8 ⇒ **轮数/任务不变（1 轮），批深 2→4**；跨轮形（S_kv>1024）轮数翻倍（eff=10 → 8+2）。每槽协议事件×2（WS1_C2V set/WS2_V2C wait/WS3_C2V set 皆 CROSS_INTERVAL=1 逐槽）；每轮事件不变（48 轮/核）。**跨核 ≤15 钳位核算**（gate0b 档案条款）：outstanding ≤ 批深 ≤ NUM_STAGES=8 < 15 ✓（槽数翻倍被批深吸收——C set 每轮 ≤8 个在途，V wait 逐槽消费）；1:1 交错纪律保持（协议位置逐字不动，仅槽数参数化）。协议税：#R4-3 档案「barrier 移除后每轮 SEM 开销坍缩至 ~0.2-0.3us，decode 对此不敏感（case11 19 轮节省仅 ~5us）」⇒ 每槽 set/wait 指令级成本 ×2 ≈ +192×~0.05us 量级（S13-5 计入）。NUM_STAGES=8 死列表项零触碰。

**S13-2 单块域分派边界决策（爆炸半径控制）**：
- **决策 = 分派拆分（任务书默认方案采纳）**：单块域（S≤16∧D=256∧S_kv≤256）保留 legacy bN 路径**逐字节不动**——S_kv∈(128,256] → bN=256 单块（预归一特化）、S_kv≤128 → bN=128 单块；多块域（S_kv>256 ⇔ kv_db）→ 新 bN=128 双侧 DB 路径。实现 = block_N 派生单点前置分支（S13-1a），`single_kv_block`/`kv_db` 谓词表达式零编辑（域由 bN 常量自动分流）。
- **域翻转形状盘点**：若不分派拆分（全域 bN=128），S_kv∈(128,256] 将翻域单块→多块：①精度路径变（预归一特化→online-softmax rescale 链，位级行为变）；②批处理结构变（task-batch 8 深→kv-batch）；③探针/锚/守卫档案全部需重基线。**拆分方案下该带零变化**（IR 守卫 G_dec256_single (1,1,256,4,256) 逐字节证明）。无更优方案需求——拆分即最小爆炸半径（无需位级/阻塞口径影响论证）。
- **D256 单块形状归属明示**（任务书点名盘点）：(1,64,64,2,256) 类 = **S=64>16 ⇒ form_b_mma 域，与 decode 分派无涉**（IR 守卫 G_fbm_single_d256 + 锚-B b11-b14 管辖，全员域外逐字节）；真 decode D256 单块 = (B,S≤16,S_kv≤256,N,256) 类（守卫 G_dec256_single (1,1,256,4,256) bN=256 + G_dec256_single128 (1,1,128,4,256) bN=128）——**两守卫均 legacy 路径，逐字节保留**；l0_d256_fp16_scale（L0 #10）= (1,64,64,2,256) 同属 fbm 域外；哨兵 perf_d256_bf16 = (2,1024,1024,16,256) fbm 域外。**L0/L1 用例表全扫描：无一落 kv_db 域**（L0 全 prefill/D≤128；L1 D256 仅 l1_d256_tailmix (1,65,96,2,256) S=65>16 fbm 域）⇒ l0/l1/coverage/all 门禁在本臂下 = 域外回归门 + 哨兵门（目标域精度由探针矩阵管辖，与 r11/r12 同构）。

**S13-3 事件 id 预算审计（完整分配表 + static_assert 域核对）**：
- **ISASI 核内约束**（r12 A12-2 三源查证档案沿用）：id 0-7、静态 Tensor 场景 6/7 禁用 ⇒ 可用 0-5；成对出现；硬件计数深度无文档明文 ⇒ **每 (type,id) outstanding≤1 最强假设自约束**（1-bit/计数器双语义免疫）。
- **gemm_v0 内部占用**（common.h 现态核验）：opt-in 调用（kMTE2Sync≥0）prologue 仅 WaitFlag(MTE2_MTE1,kMTE2Sync) **不自 set id0**；L0AB_EVENT=0/1 占用型别 = FIX_M、M_FIX、MTE1_MTE2（id0）+ M_MTE1/MTE1_M（id0/1）——**旗标空间按 (type,id) 独立**（r12 A12-2 生产实证）⇒ 与调用方 MTE2_MTE1 型 id0/1/2 零冲突。
- **kv_db 域 MTE2_MTE1 分配表（新态 = 现态，零新增）**：
  | id | 常量 | 语义（bN=128 下） | Set 站点 | Wait 站点 | 1:1 论证 |
  |----|------|------------------|---------|----------|---------|
  | 0 | SIG_KDB0 | K 槽0 常驻（含 FIFO 前缀 Q） | S0(t==0)/S1(nidx%2==0)/S4(t+1<cnt) | G1 偶 idx 消费点 | r12 全边界走查逐字继承（S13-1a 域参数仅 bN 值变） |
  | 1 | SIG_KDB1 | K 槽1 常驻 | S1(nidx%2==1) | G1 奇 idx 消费点 | 同上 |
  | 2 | SIG_VP | **V(idx)+P(idx) 组合常驻**（Set 于 P copy 后，FIFO 前缀覆盖先发的 V(idx)；**V(idx+1) 于 Set 后发射 ⇒ 不入前缀**） | G2 每 iter P copy 后 | G2 gemm2 prologue | Set(iter j)→Wait(gemm2 j)→Set(iter j+1)（标量序：gemm2(j) 返回后才发 iter j+1 的 Set）⇒ 严格交替 outstanding≤1；跨轮/跨任务：轮 k 末 Set→Wait 对闭合后下一 Set 在轮 k+1 G2 iter0（间隔整个 G1 相）✓ |
- **SIG_VP 语义在 bN=128 下的操作数集变化**：前缀覆盖集 = {V(idx)（双槽中 slot idx%2，S3 或上一 iter pre-call 先发）, P(idx)} ∪ {更早的全域 K 流（含跨轮 K'((k+1)*NS)，S1 于 G1 相末发射——落入 G2 iter0 的 Set 前缀，但发射点先于消费点整段 G1 相尾+SEM_WS3/WS2 协议窗 ≫ 0.84us 装载 ⇒ 前缀等待零暴露，r12 同款结构已在产）**；V(idx+1) 严格后于 Set ⇒ gemm2(idx) 前缀不含之（气泡填充主体）。**无需 SIG_VDB0/1 对**：双 V 槽的槽位安全由 gemm epilogue MTE1_MTE2 释放序保证（pre-call 位在 gemm2(idx-1) 返回后 ⇒ 写 slot (idx+1)%2 有序于 gemm2(idx-1) 对它的 MTE1 读），旗标只需覆盖「本次调用操作数常驻」——单 id FIFO 前缀语义完备。
- **static_assert 域核对**：kMTE2Sync ∈ {0,1,2} ⊂ [0,5] ✓（common.h L1213-1216）；ascend.py kwarg 断言 0≤id≤5 ✓；V 侧 SIG_IO3=2（V_MTE2 型）/SIG_S_HALF=0（MTE3_V 型）与 C 侧 MTE2_MTE1 型 id2 不同型别空间 ✓；跨核 SEM 0-5 不变 ✓。
- **预算判决：3/6 id 占用，零新增，零框架改动 ⇒ 无需触发任务书框架必要性论证条款**。

**S13-4 精度性质与锚计划（域分化）**：

(a) **非 BITEXACT 预期与主门**：变更算术面 = ①online-softmax rescale 粒度 256→128（r_factors 链 2→4 次 exp/mul，fp32 中间舍入序列变）；②GEMM2 K 累加链 256→128（L0C 内 128 列 fp32 和 ×2 段 + V 侧 fp32 rescale-add 合并 ⇒ 累加树变）；③GEMM1 kL0 64→128：**K 升序 C0 链式累加同序 ⇒ 位级等价**（#10b 档案「kL0Size 只改 L1→L0 搬运粒度」+ #R4-2a 12/12 BITEXACT 实证——score 栅格逐元素不变）。⇒ **主门 = golden 阻塞口径（混合容差 + MARE_abs < 10×Threshold）**，先例余量参照：#8d（bN 64→128）80-320x / #10b（D256 bN 128→256）160-640x / r11 探针最差 6.1e-5 vs 门 9.77e-3 = 160x。辅门 = vs perf17 closeness（|Δ| 量级 ~1e-4 预期，#8d/#10b 同款）。
(b) **红线四项逐项定位**：①双舍入栅格链（r16(qk)→×s fp32→r16(scores)，emit_v L753-757）= **逐元素 op 序不变，bN 只改宽度 ⇒ 栅格落点逐元素同一**（GEMM1 位级等价加持下 score 值本身 BITEXACT）；②单块预归一特化 = **留在 legacy 单块路径**（parse-time `if single_kv_block:` L791-797 分支，域拆分后新域永不进入）；③sumexp+1e-30 = emit_v generic 任务尾 L836-839 **bN 无关逐字保留**（sumexp 跨槽 rescale 累加链参数化，+1e-30 位置/语义不变）；④掩码链 = L767-779 bN 参数化等价（S13-1c：AP j0/compare LE/select -inf 替换制逐 op 同构；640 型 (F,T)→(F,F) 迁移 = clamp 构造语义同一，#8d 先例；**尾块 slot-1 needClear=false stale 有限性归纳证明**：v 槽 stale 读仅发生于尾块（valid_n<bN）行 ≥valid_n；slot1 首载 = V(idx0+1)，kv_db 域 S_kv>256 ⇒ valid_n(idx1)=128 满宽（一般地轮 k 首载 V(8k+1) 满宽：轮存在 ⇒ eff≥8k+2）；slot0 首载 = S3 静态索引 0 ⇒ needClear=true 清零；归纳：任意槽任意次载入的 stale 行 = 该槽前一占用者的实 V 数据（有限 fp16）或 S3 清零行 ⇒ **0×有限=0，无 NaN/Inf 通路**（对照 r11 K 侧 select 替换制论证——V 侧无下游替换，故必须走有限性路线，此为新增证明义务，探针 T_tail440_slot1/T_min257 精确覆盖）。
(c) **锚计划（r9 [ANCHOR-PLAN-R9] 先例，域分化）**：**锚-A 9/9 成员 = L0#1-9 全员 Form A（S=64/128>16 ∧ D≤128）⇒ 零域内成员，硬守卫 BITEXACT 保持，无降级项**；**锚-B 14/14 成员 = b01-b14 全员 S=32/64/96>16 form_b_mma/fbm-单块 ⇒ 零域内成员，14/14 BITEXACT（诊断级）保持**（成员归属 = p6_battery_anchors.py L48-80 shape 表逐行核验，非假设）。降级方案条款空置（如未来锚矩阵扩入 decode D256 多块形状，按「阻塞口径+根因纪律」降级——本轮无对象）。
(d) **探针矩阵重基线（bN=128 拓扑重设计，24 目标 + 12 守卫）**：目标组 = case15/18 原形（eff=4 单轮偶批）+ eff 多样性 {3(264/300/384),4(440/512),5(640),7(896),8(1024)} + 跨轮 {1152(9=8+1 奇尾批),1280(10=8+2),1160(10 跨轮+尾 vn=8 落 slot1 复合边界),2048(16=8+8),4096(32)} + causal {S16,S2 MTP,跨轮} + 尾块 {640 掩码翻转型,300,440(slot1 尾),257(最小多块 vn=1)} + bf16×2 + scale + 大幅值 + my_count{1,2,mixed} + 最小 1×1 + zeros 对照（S5-4-3 陷阱守卫）；守卫组 = r12 十二守卫逐字继承（D256 单块×2/D128 多块×2/D128 批×3/D64/form_a×2/fbm×2）**BITEXACT vs perf17 + IR 逐字节双证**。**尾块 640/300 在新分块下的边界**：640%128=0 ⇒ 无尾块+掩码变体翻转（need_bound_mask False 型，新覆盖点）；300 ⇒ eff=3 尾 vn=44 落 slot0（S3 清零基底）；440 ⇒ eff=4 尾 vn=56 落 **slot1**（有限 stale 归纳链首证点）；257 ⇒ eff=3 尾 vn=1 极窄。**my_count 非齐次**：r_tasks 混合形（B×N 不整除 20 核）保留 T_cnt_mixed_r 型。**causal S≤16 均匀恒等**：S13-1a 已证 eff≡kv_iters ⇒ causal 目标组同时验证恒等式在 bN=128 的运行期一致性。
(e) **压力矩阵**：继承 4 配置（anchor_stress_r5）×12 nondet=0 + 扩展重打被改路径：**双侧 DB 槽轮转专项**（V 槽奇偶轮转/跨轮 slot0-slot1 交替/任务边界 S3-S4 交接/批深奇偶 ×12 nondet=0），拓扑取自 (d) 目标组的边界成员。

**S13-5 scalar/固定段预算（净 EV 修正区间）**：
- **调用数×2 的固定段**：gemm_v0 每调用固定段 = prologue（WaitFlag(MTE2_MTE1)+Set/Wait(FIX_M)+2×Set(M_MTE1)）+ epilogue（2×Wait(M_MTE1)+Set/Wait(MTE1_MTE2)+Set/Wait(M_FIX)）≈ 10-12 flag 指令 + 循环簿记。参照系：#R4-2a 档案每-step 链（PipeBarrier+2 flag 往返）~0.15us ⇒ 固定段（无 PipeBarrier，但含 FIX_M 排空等待）估 **f_call ∈ [0.10, 0.35]us**（稳态 FIX_M 排空对象 = 前槽 w1+w3 = 24KB@~50GB/s ≈ 0.48us，但发射点先于本调用整槽周期 ⇒ 大部已排空，残 stall 小）。**+192 调用 ⇒ +19~67us C.scalar_t**。
- **每槽标量零头×2**：copy 描述符（4/槽）+SEM 指令（~6/槽）+循环标量 ≈ s_slot ∈ [0.15,0.35]us ⇒ +96 槽 ⇒ **+14~34us**。
- **k-step 链守恒**（S13-1b：768 步/核不变 ⇒ ~115us 项零增量）——kL0=128 选择的直接预算收益。
- **FIX 字节 +50%**（w3 ×2：3072→4608KB/核）：fix_t 61.37→~92us（吞吐线性外推，L2 写命中 97.9% 维持假设）——FIX 非 pacing 管（13.2%→~22% duty），每槽 FIX_M 排空对象 24KB<32KB 反降 ⇒ 不入关键路径增量，但如实计入引擎占用。
- **V 侧净增**：phase2 w3 读 +768KB/AIV（GM_to_UB 1536→2304KB/核）⇒ V.mte2_t 48.25→~72us（31.8GB/s 同速率外推）；phase2 D 宽 vec 工作×2 ⇒ vec_t 82.64→~110-125us；每槽旗标/SEM 零头×2 ⇒ V.scalar_t 146→~190-210us。**V 侧余量核算**：现行 V wait_id0+wait_id4 = 432us 空闲 ⇒ 净增 ~90-130us 被余量吸收，V 不成 pacing（若 wait_id2 显著上涨则 V-paced 反转——机制判据监测项）。
- **回收侧**：可压缩气泡 = wall 463.58 − MTE2 地板 341.75 = **121.8us**（duty 73.7→100% 的全额）；本臂机制填充 G2 相 SEM_WS2 协议窗（wait_id2 119.92us 的 MTE2 空闲段主体）+ G1 相维持 r12 流式。
- **净 EV 修正区间（诚实降档声明）**：scalar 税中值 ~55us = 回收上界的 45%（上界 101us = 83%）⇒ **任务书「固定段吞掉 >50% 预期收益」条款在中值-上界带触发 ⇒ EV 由 -10~15% 如实降档为 -4~-16%，中值 -9~-12%**（wall ~390-445us）；对应 duty 预期 78-88%（任务书 85-95% 下修）。非净负（中值带 > 3% 门 2-4 倍）⇒ 不触发 NO-GO 判据 ⑤，msprof 唯一裁决。
- **bench 5 线换算口径（预登记）**：case15 起点 0.744x/case18 0.879x（S12-7 权威值）；换算 speedup_new ≈ speedup_old ÷ (1+Δwall_frac)：Δ=-10% ⇒ 0.827/0.977；Δ=-5% ⇒ 0.783/0.925；守卫 5 线 case10≥0.90/13≥0.69/14≥0.99/16≥0.52/20≥0.82（现值 0.990/0.891/1.390/0.777/2.113 全带余量）——守卫 IR 逐字节域 ⇒ 带内漂移预期，换算表 Phase C 实测后落档。

**S13-6 GO/NO-GO 判决**：
- NO-GO 判据逐项核对：①**L1 实编超限？** 否——手算 268KB ≤ 511.75KB（余量 243.75KB），Phase B GetWithOffset 读数门兜底（手算仅初筛纪律）；②**id 预算不可解？** 否——3/6 占用零新增（S13-3 分配表 + static_assert 域核对），SIG_VP 单 id FIFO 前缀在双槽下语义完备；③**单块域无法隔离？** 否——bN 谓词单点拆分（S13-2），S_kv≤256 带 legacy 逐字节 + IR 守卫双证通道在案；④**V 侧参数化不可保域外逐字节？** 否——emit_v 源文零编辑 + parse-time 常量闭包（#R6-1 机制），域外 IR 守卫 18/20 dump 机器证明通道在案（非 alias io3/r_2d 配置已有生产先例）；⑤**scalar 预算净负？** 否——中值 +55us vs 可回收 121.8us（45%），区间 [-4%,-16%] 中值带过 3% 门；上界带（83% 吞噬）如实降档标注，非净负。
- **判决：GO（进入 r13-2 Phase B+C）**。
- 未验证项如实声明：①f_call/s_slot 固定段实测量级（区间估计，Phase B op 模式 C.scalar_t 对照裁决）；②V 侧净增 ~90-130us 的余量吸收假设（wait_id2 反转监测）；③64KB 粒度 DMA 的 MTE2 busy 速率保持（128KB→64KB 装载 ×2 频次，重排税 r11 -2.7%/r12 +1.1% 双面先例，mte2_busy_gbps ≥74 监测线）；④L2 w3 写命中 97.9% 在足迹 4.48MB→4.6MB 下的保持（≪201MB 容量，推断保持，未隔离实测）。全部由 r13-2 msprof 实测裁决，无一项依赖预测性宣称。

[DB-ANALYSIS]（skill Double Buffer 特殊要求，实施前置）：
- Q1 循环内有 MTE3？——C 侧 kv_db G2 循环无 MTE3（w3 写出走 FIX 管 `T.copy(acc_o_l0c, workspace_3)`；C 核本域零 MTE3 用户站点，生成码 L95 CrossCoreSetFlag<PIPE_FIX> 佐证）；V 侧 io3/w2 的 MTE3 结构零触碰（既有 SIG_S_HALF/SIG_IO3 令牌闭环生产在跑）。
- Q2 有跨迭代累加器？——DB 槽内无（V 槽仅持有原始 V tile 数据，单槽生命周期 = 一次装载一次消费）；acc_o_l0c 每调用 init=True 槽间无累加；online-softmax 累加器（acc_o/sumexp/m_i）在 V 侧 UB，结构参数化不变（槽数翻倍仅改循环次数，r_factors[i]/sumexp_is[i] 按批内 i 索引，深度 8 ≥ 新批深 4）。
- Q3 同步方式？——释放序：gemm_v0 epilogue SetFlag/WaitFlag<MTE1_MTE2>(0)（调用返回后发射的 MTE2 有序于该调用全部 MTE1 读——v 槽 pre-call 写入安全性的唯一依赖，A10 在产机制）；常驻序：per-buffer MTE2_MTE1 FIFO 前缀旗标（SIG_VP Set-after-P 覆盖 V+P；SIG_KDB0/1 K 侧继承）；零新增 flag/id/SEM。
- 结论：DB 三问闭合，可实施。

[DESIGN-S13]（Phase B 实施蓝图，站点级 diff 计划——全部位于 examples/mha/mha.py，框架零改动）:

**D1 host block_N 派生（L228-231，1 处前置分支 + 注释）**：
```python
if S <= 16:
    if D == 256 and S_kv > 256:
        block_N = 128   # #R13-1: kv_db dual-side DB domain (S13-1a)
    else:
        block_N = 512 if (D <= 128 and S_kv > 256) else (256 if S_kv > 128 else 128)
```
**D2 host gemm1_kL0 派生（L234-238，64 档条件收窄）**：`64 if (S <= 16 and D == 256 and S_kv > 128 and S_kv <= 256) else 128`（新域落 128 档 = k-step 守恒解，S13-1b；单块 legacy 64 档不动）。
**D3 host 注释区（L266-289）**：kv_db 谓词表达式零编辑；注释更新（bN=128 双侧 DB 域说明 + id 分配表 S13-3 + L1 268KB 预算）；SIG_KDB0/1/SIG_VP 常量零变化。
**D4 C 侧 alloc（L1774，v_l1 三元式）**：`v_l1 = T.alloc_L1([2, block_N, D], dtype) if kv_db else T.alloc_L1([block_N, D], dtype)`（k_l1 L1773 同款形态，S4-3-4②/probe_r42b 机制先例）。
**D5 kv_db G2 块重构（L1978-2008，三站点）**：
- S3（G1 批尾 V priming）：`T.copy(V[bz, vj0:vj0+vvalid, by, :], v_l1[0, :, :])`——静态 slot0（NS 偶 ⇒ idx0%2≡0；needClear=true 清零基底，S13-4b 归纳锚点）。
- S6'（G2 iter：SEM wait/P copy/Set 位置逐字不动）：`T.set_flag("MTE2","MTE1",SIG_VP)` 之后、gemm 调用之前插入 **S5' pre-call V(idx+1) 预取**（守卫 i+1<batch_iters 局部——轮头 V 由 S3 覆盖防双载；dst `v_l1[nvidx % 2, :, :]` 运行期槽 ⇒ needClear=false ⇒ S13-4b 有限性证明管辖；extent 公式逐字继承 r12 S5）。
- G2 消费点：`T.gemm_v0(p_l1, v_l1[idx % 2, :, :], acc_o_l0c, init=True, mte2_sync=SIG_VP)`（运行期切片——k_l1 先例；模板实参 → <half,float,16,256,128,false,false,128,2>）。
- 旧 S5（after-return 位 L1997-2005）删除（被 S5' 取代——发射点前移 = 本臂机制主体，A12-6 知识③：after-return 位时序中性）。
- G1 块（S0/S1/S2/S4）零编辑（kL0=128 由 D2 host 值流入既有 `kL0Size=gemm1_kL0` 实参；模板实参 → <half,float,16,128,256,false,true,128,0/1>）。
**D6 零触碰域**：emit_v 源文/SEM 矩阵（set/wait 位置）/single_kv_block 分支/generic else 分支/form_a/form_b_mma/host 包装 mha()/框架两文件/test_mha.py/DESIGN.md/proto.yaml/提交副本。docstring rev4r13 注记 = 采纳后（r12 先例：测量态先行）。
**D7 门禁与测量计划（r13-2 执行清单）**：
1. 备份 history_version/mha_impl_r4_perf18.py（改前 0fb193b2）；变体先落 /tmp/opencode/r13_probes/mha_r131_edit_v1.py（基线采集完成前生产冻结，r11/r12 缓存纪律同款）；ast/ruff。
2. IR 守卫 dump 20 shape = r12 17 shape 逐字继承 + 新增 3（G_dec256_single (1,1,256,4,256)/G_dec256_single128 (1,1,128,4,256) legacy 单块守卫 + T_kv10_d256 (2,1,1280,2,256) 跨轮目标审计形）：base=生产 0fb193b2（r13_src_base，disable_cache）/var=变体（r13_src_var）；预期 **18/20 逐字节 + G_case15/T_case18 按预期改变**；守卫含 AIV 主块逐字节（域外 emit_v 常量不变证明）；目标域 AIV 按预期改变（阻塞口径管辖，S13-1c 如实登记）。
3. L1/L0/UB 实编读数门：L1 预期 q0/k8192/v139264/p270336/末端 274432B ≤ 524032B；L0C acc_s 8192B/acc_o 16384B；UB ~61KB；L0A/L0B 65536 池。事件 id 生成码 grep：MTE2_MTE1 用户 Set ∈ {0,1,2} 无 6/7；gemm 模板实参串核验（G1 奇偶双实例 kL0=128 + G2 单实例）。
4. 生成码旗标双射审计（case15 形 + kv10 跨轮形）：S0/S1/S4 id0-id1 交替、S3 静态 slot0、S5' 局部守卫、S6' Set 后 gemm 前预取位序、SEM CrossCore 全位置 vs base 逐字同位。
5. smoke 先行（timeout 保护，会话 10 HANG 纪律）：6 配置 = case15_like_chain/kv10_xround/eff3_odd_batch/tail440_slot1/min257_tail1/case15_orig——无挂死 + 阻塞口径 + vs perf17 closeness。
6. 精度全门禁（run_gates 形态继承）：探针 probe_r131_kvdb（24 目标阻塞口径+closeness / 12 守卫 BITEXACT vs perf17 / PART 0 分派审计）→ 锚-A 9/9 + 锚-B 14/14（p6_battery_anchors 原脚本零改动复用——全员域外）→ 压力（继承 4 + 扩展双侧 DB 槽轮转专项 ×12 nondet=0）→ l0/l1/coverage/all（哨兵 4/4）。
7. 性能：同会话新鲜基线（生产 0fb193b2 态复测 case15/18 + 守卫 case10/11/14/16/20/7/9，median n=25，±3% 带）→ 变体同口径 → **op 模式 case15 变体态**：duty 签名主判（73.7→78-88% 预期带，机制判据 (a) r8-1 纪律：wall 收益必须伴 duty 上升）+ mte2 busy 速率（≥74 GB/s/核 监测线）+ C.scalar_t（固定段税实测 = S13-5 区间裁决）+ fix_t/V 侧三管/wait_id2（V-paced 反转监测）。
8. 判决：case15/18 ≥3% 改善 + duty 上升 ⇒ 采纳；零响应/劣化 ⇒ 回滚三重验证（md5 0fb193b2 还原 + IR 20/20 vs r13_src_base 还原 + l0 复验）+ 变体留存 + 归因落档 ⇒ **本域最后一个规则内杠杆关闭，收口声明交还用户**（任务书 §1 判决规则）。
9. 失败处置：挂死 → timeout 隔离 + bisect（毒化矩阵方法学 S10-3-3）；精度超差 → 根因分析纪律（勿放宽门）；每门禁单项反复失败 >3 次 → 结构性受阻落档。
10. 缓存纪律：r13_cache_{base,var,g1,final} 全程专属；长时命令 setsid 脱离会话（S11 教训）。

**A13-7 Phase A 收口**：零生产变更核验——mha.py = 0fb193b2（perf17 同体）✓；common.h/ascend.py md5 = r12 改后态（82e80025/1c3a00ed）git diff 与 framework_r12_phaseB.diff 逐字一致 ✓；本轮无测量（Phase A 定义），r13 新鲜基线于 r13-2 采集（baseline_r13_iter1.json = 占位引用 r12 采纳态参照系）。迭代计数：r13-1 = 全局 iter33（分析轮，consecutive_no_improvement 计数不变 = 0，iter24-26/r12-1 纯测量-分析轮不递增口径延续）。反模式对照登记（agent 清单条款）：antipatterns「tile size 过小」条目为本臂**有意反向**（缩块换 DB 窗口）——保留原因 = r12 实测绑定约束是 MTE2 协议窗空闲（duty 73.7%）而非每槽开销，缩块成本（槽×2 各税）已被 A10 流式收益与 S13-5 预算覆盖；「尾块/mask 随 tile 复查」检查点 → 探针矩阵重基线（S13-4d）履行；其余条目（launch 核数/冗余全局同步/CV overlap/AIV 流水/逐块归约/标量扫描）逐项核对不适用或已在产合规（perf_log r13-1 条目同步登记）。

### 迭代 r13-2（全局 iter34）：Phase B+C —— [#R13-2] bN=128 k/v 双侧 DB 实施 + 测量 + 性能判决（2026-10-05）

[ORDER-CHECK] 准备实施: [#R13-2] 升级臂实施（kv_db 域 bN 256→128 + k/v 双侧 L1 DB + S5' pre-call V 预取）| 前置依赖: [r13-1 Phase A GO 判决（[ANALYSIS]-S13 五项 NO-GO 判据逐项否证 + [DESIGN-S13] D1-D7 蓝图 + [DB-ANALYSIS] 三问闭合）✅；改前备份 history_version/mha_impl_r4_perf18.py（0fb193b2）落位 ✅；同会话新鲜基线采集完成（生产冻结态 r13_base：case15 463.64/case18 465.12 + op duty 73.5%）✅] | 结论: ✅

[IMPL-#R13-2] 已阅读: common.h:1183-1402（gemm_v0 现态全文复核：nTile 整除 static_assert L1288、kL0Budget L1297-1298、B/A-tile 断言 L1301/L1305、opt-in prologue L1329-1338、epilogue L1398-1401）、common.h:55-110（copy_gm_to_l1 need_clear 语义全文——InitConstValue+PipeBarrier 触发条件 `tailM!=dstM || tailN!=dstN || fractalExtent!=dstM*dstN`）、codegen_ascend.cc:2979-3045（**needClear 派生规则全文，本轮新知识**：`is_zero(dst_offset)` ∨ 整行版本化拷贝例外（realTailM 为 IntImm 且 tail_m==dst_m 且 CanProve(offset % (dstM×dstN))==0）——r11 档案「needClear=(dst_offset==0)」为不完整表述，commit 059f0a65/#1723 已含版本化例外分支）、mha.py 现态全域（host L220-334 + emit_v L334-861 + 兜底 prim_func L1756-2152）、G_case15_B_d256_dec.cc base/var 双态生成码、r12 全套工具链（make/dump/smoke/probe/stress/gates/wall/op 脚本）。关键约束: ①实施 = [DESIGN-S13] D1-D5 五区域（host bN 前置分支 + gemm1_kL0 64 档收窄 S_kv≤256 + 注释区 + v_l1 三元式 + kv_db G2 块 S3 静态槽/S5' pre-call/消费点运行期切片），其余零触碰（emit_v 源文/SEM 矩阵/single 分支/generic else/form_a/fbm/host 包装/框架两文件）；②D5 位序 = SEM wait→P copy→Set(2)→S5' V(idx+1)→gemm2(v_l1[idx%2])→w3 FIX→SEM set（S5' 在 Set 后 = 不入前缀，在 gemm 前 = 获得调用期+下一 SEM 窗双段流式窗口）；③槽释放序 = gemm2(idx−1) epilogue MTE1_MTE2（pre-call 位在返回后 ⇒ 有序）；④变体先落 /tmp/opencode/r13_probes/mha_r131_edit_v1.py（生产冻结至基线采集完成，r11/r12 纪律）；⑤缓存 r13_cache_{dumpbase,dumpvar,smoke,g1,final,final2} + r13_base/cache_{wall,op} + r13_iter1/cache_perf 全程专属；⑥smoke 先行（会话 10 HANG 纪律）；⑦长时命令 setsid 脱离会话（S11 教训）。

[SELF-CHECK]（变体 = mha_r131_edit_v1.py，22fffbe5，2195 行，5 区域 diff vs perf18；ast OK + ruff check "All checks passed!" + ruff format clean；上位后 md5 核验一致）:
①**IR 守卫 dump 20 shape**（r13_src_base（生产 0fb193b2，disable_cache）vs r13_src_var）：**17/20 逐字节相同 + 3/3 目标按预期改变**（G_case15 11092→11555B / T_case18 11152→11615B / T_kv10_d256 10363→12090B）；17 守卫含**新增 2 legacy 单块守卫**（G_dec256_single (1,1,256,4,256) bN=256 / G_dec256_single128 (1,1,128,4,256) bN=128 = 分派拆分逐字节证明）+ form_a×8 + fbm×3 + D128 decode×3 + case20；**provenance 链核验：r13_src_base 与 r12_src_var 的 17 共享 shape 全逐字节相同**（G_case15 base md5 e5a5f0c8 = r12 采纳态档案值）⇒ 基线态 = r12 终态机器证明；目标域 AIV 主块按预期改变（emit_v bN 参数化：w1/w2 半宽 + io3 专属旗标 V_MTE2 id2/3 init/destroy 平衡 + phase2 槽数×2）——r12 的「AIV 逐字节」守卫在本轮目标域不适用（S13-1c 预登记），由阻塞口径精度门管辖；
②**L1/L0/UB 实编读数门 PASS**：L1 = q@0B(8KB)/k@8192B(128KB 双槽 stride 32768 halfs)/v@139264B(128KB 双槽)/p@270336B(4KB) **末端 274432B = 268KB ≤ 524032B**（与 S13-1b 手算逐字节一致）；L0C = acc_s@0(8KB)+acc_o@8192(16KB)=24KB；L0A/L0B 65536、UB 196352 池不变；UB 实编含 io3@39520(16KB 专属)+r_2d@31328(8KB 专属)（io3_is_alias 翻转预期形，D128 decode 生产先例同构）末端 ~60KB ≪ 196352B；
③**生成码审计（G_case15 + T_kv10 跨轮形）**：gemm 调用 = `gemm_v0<half,float,16,128,256,false,true,128,0/1>`（G1 奇偶展开，kL0=128 ✓）+ `gemm_v0<half,float,16,256,128,false,false,128,2>`（G2，K=bN=128/N=256 nTile 整除 ✓）；用户 SetFlag<MTE2_MTE1> 站点 = S0(0)/S1 运行期 ((i+1)%2)/S6(2)/S4(0)——**全部 id∈{0,1,2}⊂[0,5] 无 6/7，零新增** ✓；S5' 位序实证：Set(2)@L91 → V 预取@L93（运行期槽 ((i_1+1)%2)*32768）→ gemm@L95（消费 v_l1[(i_1%2)*32768]）✓；S3 静态槽 v_l1[0] ✓；CrossCore SEM 全站点序列与 base 同构（C: Set3-init/Wait1/Set0-FIX 逐槽/Wait5/Wait2 逐槽/Set4-FIX 逐槽/Set3 轮尾/Set1+Set5 destroy；V 镜像）——协议位置零漂移 ✓；kv10 跨轮双射手工走查：id0 5 Set(S0+K2/4/6/8)↔5 Wait(gemm1 0/2/4/6/8)、id1 5↔5、id2 10↔10（轮 0 8 + 轮 1 2）严格交替无 dangling；V 记账恒等式：V(0)@S3-r0 + V(1..7)@S5'-r0 + V(8)@S3-r1 + V(9)@S5'-r1 = 10 = eff ✓（#R4-2b 纪律）；S4 在 my_count=1 形编译期折叠消失 ✓ 预期；
④**needClear 审计（新规则全文入库）**：S1/S5' 运行期尾块形（T.min 不折叠，如 kv10/440/300/257 型）→ realTailM 非 IntImm → **needClear=false**（r11/r12 同款体制 ⇒ K 侧 select 替换制 + V 侧 S13-4b 有限 stale 归纳证明管辖：槽首载满宽（S_kv>256 ⇒ valid_n(idx1)=bN；slot0 由 S3 静态偏移清零锚定）⇒ stale 行恒为有限 fp16 ⇒ 掩码列 P=0 × 有限 = 0 无 NaN 通路）；case15 整除形 realTailM 折叠 IntImm=dst_m → 版本化例外 → **needClear=true 但满 tile 短路**（helper 内 if 条件假 ⇒ 无 InitConstValue/PipeBarrier，零运行时成本）；S0/S3/S4 静态槽 offset 0 → true ✓；P 侧满 tile 短路 ✓；
⑤旗标双射静态论证（S13-3 表 + ③生成码实证）+ smoke 6/6 运行时确证（无挂死）；
⑥**smoke 先行 6/6 GATE_OK 无挂死**（case15_like_chain/kv10_xround/eff3_odd_batch/tail440_slot1/min257_tail1/case15_orig；golden 阻塞口径 matched=1.00000、MARE_abs 3.05e-05~6.10e-05 余量 160-320x；vs_old max|d| 0~6.1e-05 = #8d/#10b 先例带）；
⑦框架两文件零触碰核验（md5 82e80025/1c3a00ed = r12 态，git diff 与 framework_r12_phaseB.diff 逐字一致）；test_mha.py/DESIGN.md/proto.yaml/提交副本零触碰 ✓；ruff clean ✓。备份 = history_version/mha_impl_r4_perf18.py（0fb193b2）。

[RESULT-#R13-2] 优化点: **kv_db decode D=256 域 bN 256→128 缩块 + k/v 双侧 L1 DB（268KB）+ S5' pre-call V 预取（Set(SIG_VP) 后、gemm 前——双槽解除 after-return 位钉死，流式填充 G2 相 SEM_WS2 协议窗 = r12 duty 残差主体）+ 分派拆分（单块域 S_kv≤256 legacy 逐字节）+ gemm1_kL0=128（k-step 守恒 768/核）+ 零新增事件 id/零框架改动** | 精度: **pass（全门禁过，采纳态 = docstring 注记后 c316a004，IR 与测量态同一性核验 PASS：case15 dump md5 3a6bf589 逐字节一致）**——①smoke 6/6 无挂死 GATE_OK；②探针 probe_r131_kvdb **24 目标阻塞口径全过**（matched=1.00000 全体、MARE_abs 7.63e-06~4.88e-04 余量 **160-640x**（bf16 320x）；**T_valrange 2x 余量 = 既存 golden 栅格属性**（vs_old BITEXACT = 与 r12 生产态逐位同值，test_mha.py L317-323 fp16 score-grid 边界放大在案条款，非本轮引入）+ **12 守卫全 BITEXACT vs perf18**（分派拆分零泄漏）+ PART 0 分派审计 36/36（24 目标全落 kv_db@bN=128 / 12 守卫全不落）；③锚-A **9/9 BITEXACT** + 锚-B **14/14**（BATTERY_RESULT: PASS；全员域外 = S13-4c 成员核验兑现，硬守卫无降级）；④压力：继承 4 配置 ×12 nondet=0（PASS）+ 扩展 **19 项** bN=128 双侧 DB 被改路径专项（e01-e19：48 任务链/eff3 奇批/跨轮 8+1/8+2/8+8/尾 slot1 1160/causal×3/双槽尾块 440/300/257/bf16 跨轮/scale/my_count{2,mixed}/最小 1 核/640 掩码翻转型）×12 nondet=0（STRESS_RESULT: PASS）；⑤l0 **10/10** + l1 **19/19** + coverage **19 PASS/0 MISS** + `--level all` **ALL_EXIT=0 "Test Passed!"**（L2 正确拒绝 + Boundary 6/6 + 哨兵 4/4 PERF_PASS 含 perf_d256_bf16）；⑥非 mha 消费者 UT：**不适用**（框架零改动——r12 的 90/90 默认路径回归门本轮无对象，如实登记）；⑦红线四项：双舍入栅格链（逐元素 op 序不变，GEMM1 kL0 变更位级等价 #10b/#R4-2a 档案）/单块预归一特化（legacy 路径 parse-time 隔离，守卫 G_dec256_single×2 BITEXACT+IR 双证）/sumexp+1e-30（emit_v 任务尾逐字保留）/掩码链（bN 参数化等价 + 640 型 (F,T)→(F,F) 迁移 #8d 先例，探针 T_eff5_640_maskflip 过）✅；⑧ruff check/format clean；⑨docstring 注记后复验：IR 同一性 PASS（③md5）+ l0 复验（Phase D 终态复跑覆盖）。 | 性能(msprof median us, n=25, 同会话新鲜基线 r13_base，warmup5/iters20，--task-time=on 口径): **case15 463.64→421.64（−9.06%）** / **case18 465.12→423.56（−8.94%）**——stdev 1.78/1.36（Δ=42.0/41.6us > 20σ），双目标同向；守卫 7 case：case10 **+1.12%（IR 逐字节相同域 ⇒ 漂移带归因，session-8 case20 + r12 case20 先例口径「IR 逐字节域免计时归因」；34.9us 小核发射抖动敏感，历史带 +1.36%/−0.69% 内）** / case11 −1.89% / case14 −0.35% / case16 −0.23% / case20 +0.28% / case9 +0.18% / case7 +0.06% 全带内 | op 归因（case15，r13_base/op vs r13_iter1/op，机制判据 (a)）: **机制激活、签名兑现**——**duty 73.5→86.4%（+12.9pp，S13 预期带 78-88 上沿）**；**wait_id2 119.63→66.44（−53.19us = 攻击靶心实体：G2 相 SEM_WS2 协议窗被 V 流式预取+V 侧提速双重压缩）**；cube_t 49.80→49.80 **精确守恒**（k-step 守恒设计的机器证明）；GM_to_L1 25920KB **精确守恒**（布局无税兑现：512B 行粒度不变）；taskdur 466.34→423.38；三税如实：mte2_t 342.65→365.54（同字节 ⇒ **busy 速率 75.64→70.91 GB/s/核 = −6.3% 细流重排税**（64KB×2 频次+四 L1 目的区交替；监测线 74 未达但被 duty +12.9pp 压倒性覆盖——r11/r12 符号条件知识细化：细粒度化本身带引擎税，与间隙消除收益并存）、fix_t 61.53→96.32（L0C_to_GM 3072→4608KB = w3 槽数×2 D 宽字节 +50%，S13-5 预估 ~92 命中；fix 速率 49.9→47.8 轻微）、**C.scalar_t 336.15→411.03（+74.88 = gemm 调用×2 固定段 + 每槽标量×2；落 S13-5 预算带 [+33,+101] 中上段 ⇒ scalar_t/c.wall = 99.7% = scalar 管成为新 pacing 地板）**；V 侧：vec_t 82.64→121.13（phase2 D 宽×槽数 S13-5 命中）/mte2_t 47.27→94.44（w3 读 +768KB/AIV 命中；速率 32.5→24.4 GB/s = L2 端口争用二阶项如实登记）/mte3_t +13.27/scalar_t +68.31，wait_id0 199.4→183.2 + wait_id4 232.4→202.4（V 余量收窄但未反转 pacing ✓ S13-5 假设兑现）；mte1_t 134.75→143.35（+6.4% = A 操作数跨 n/k-tile 重读二阶项，S13-5 预估 +2.5% 略低估如实登记）| 对比: **−9.06%/−8.94%（≥3% 门 3 倍）** | 采纳: **YES —— #R13-1 采纳（升级臂兑现，纯算子侧）**。bench 5 线换算（预登记公式 speedup≈old/(1+Δ)，本地相对口径参考值，权威复评 = Orchestrator）：case15 0.744→**~0.818x** / case18 0.879→**~0.965x**；decode 不回退 5 线（case10≥0.90/13≥0.69/14≥0.99/16≥0.52/20≥0.82）= IR 逐字节域预期不变（起点值 0.990/0.891/1.390/0.777/2.113 全带余量）。**残余（本臂后）**：scalar 固定段地板（411us≈wall）+ mte2 速率税 −6.3% ⇒ duty 86.4% 的残差 13.6% 主体 = 每槽 scalar 链（prologue/epilogue flag 序列 ×384 调用 + k-step 链 768 + SEM 指令 ×192 槽）——**规则内杠杆穷尽判定：本域 S1(推演负→被本臂实测取代)/S2(实测负)/S3(门②FAIL)/A10(已采纳)/S13(已采纳) 链条闭合，进一步压缩需 scalar 路径结构级重构（超 Stage-3 规则范畴），收口决策按任务书 §1 交还用户**。备份链：perf18（改前 0fb193b2）/ perf19（采纳态 c316a004 同体）；变体测量态留存 /tmp/opencode/r13_probes/mha_r131_edit_v1.py（22fffbe5，2195 行）。

### 迭代 r13-3（全局 iter35）：Phase D —— 采纳态收口（2026-10-05）

[ORDER-CHECK] 准备收口: [#R13-3] 采纳态全门禁终态新鲜复跑 + 落档四件套 + 备份链 | 前置依赖: [#R13-2 采纳判决（−9.06%/−8.94% ≥3% 门 + duty 签名 (a) 兑现）✅；docstring rev4r13 注记 + IR 同一性 PASS（case15 dump 3a6bf589 == 测量态）✅；备份链 perf18/perf19 落位 ✅] | 结论: ✅

**终态新鲜复跑（采纳态 c316a004 = perf19，专属缓存 r13_cache_final2，md5 前置核验）**：锚-A **9/9 BITEXACT** + 锚-B 14/14（BATTERY_RESULT: PASS）；继承压力 4 配置 ×12 nondet=0（PASS）；扩展压力 **23/23 配置 ×12 nondet=0**（STRESS_RESULT: PASS）；coverage **19 PASS/0 MISS**；`--level all` **EXIT=0 "Test Passed!"**（L0 10/10 + L1 19/19 + L2 正确拒绝 + Boundary 6/6 + 哨兵 4/4 PERF_PASS：prefill_worst/decode_worst/d256_bf16/max_grid 全 matched=1.00000）。日志族 /tmp/opencode/r13_probes/r133_{anchors,anchor_stress,stress_ext,coverage,all}.log + final_runner.log。

**落档四件套**：optimization_log（[ANALYSIS]-S13 / [DB-ANALYSIS] / [DESIGN-S13] / [ORDER-CHECK]+[IMPL]+[SELF-CHECK]+[RESULT-#R13-2] / 本收口条目）✅；perf_log r13-1/r13-2/r13-3 三条目 ✅；final_report §S13 ✅；baseline_r13_iter1.json（Phase A 占位+GO 判决）+ baseline_r13_iter2.json（全测量数据+op 签名+bench 换算预登记）✅。备份链：perf18（改前 0fb193b2）/ perf19（采纳态 c316a004 同体）/ 变体测量态 /tmp/opencode/r13_probes/mha_r131_edit_v1.py（22fffbe5）✅。

**收口判定（会话 13）**：迭代 3/软预算 4（r13-1 Phase A + r13-2 Phase B+C + r13-3 Phase D 收口）；**采纳 1（#R13-1 升级臂，−9.06%/−8.94%，纯算子侧）/ 回滚 0**；**consecutive_no_improvement = 0（保持：r12 采纳后 r13 再采纳）**；design_error = no（分派拆分 + 参数化域 + id 预算 + L1 预算全部按设计兑现，无一 NO-GO 判据触发）。会话 13 净知识产出：①**缩块换 DB 窗口定律**（A10 时代）：bN 减半的槽数×2 固定段税（scalar +74.9us 实测，预算带 [+33,+101] 命中）可被 duty +12.9pp 的协议窗填充收益压倒性覆盖（净 −9%）——与 r11 守恒律时代的 S1 推演负结论形成时代对照：**同一结构改动的净符号由同步体制决定**（管级排空下缩块纯付税，per-buffer 下缩块买窗口）；②**needClear codegen 规则全文入库**（r11 档案精化）：`is_zero(dst_offset)` ∨ 版本化整行例外（realTailM IntImm ∧ tail_m==dst_m ∧ CanProve(offset%tile_elems==0)，commit 059f0a65/#1723）；满 tile 载入 needClear=true 时 helper 条件短路 = 零运行时成本；运行期尾块拷贝恒 false ⇒ 有限 stale 归纳证明（S13-4b：槽首载满宽 by 域界 S_kv>256）为 V 侧新增精度工具（K 侧 select 替换制之外）；③**k-step 守恒设计规则**：bN÷2 ∧ kL0×2 ⇒ 步数守恒（768/核）⇒ cube_t 精确守恒实测命中——缩块臂的 kL0 档位选择第一原则（#R4-2a 单步禁令的正面应用）；④**FIFO 前缀单 id 覆盖双槽 DB 的适用边界**：V 侧 pre-call 位在 Set 后 ⇒ 无需奇偶 id（对照 K 侧：预取跨前一 Wait ⇒ 奇偶必需）——「同型 buffer 两个 Set 可否在一个 Wait 前 outstanding」为判据；⑤**V 侧工作量缩放律定量化**：phase2（D 宽）∝ 槽数（w3 GM 往返 +50%：fix_t +34.8/V.mte2_t +47.2 实测）、phase1 ∝ 元素数（不变）——缩块臂的 V 侧税基线公式入库；余量吸收判定（wait_id0/id4 432→385us 仍富余 ⇒ 无 pacing 反转）；⑥scalar 管成新 pacing 地板（scalar_t/c.wall = 99.7%）⇒ 本域后续任何优化必须以 scalar 指令数为一阶目标（结构级，超规则范畴）。终态 = **c316a004（2240 行）+ 框架 r12 opt-in 参数态（82e80025/1c3a00ed，本会话零触碰，git 工作区态未 commit）**；cann-bench 复评与提交副本同步留 Orchestrator（**成对提交条款沿用 r12：算子+框架缺一收益归零**）。提交副本 mha_submission/ 未触碰 ✓；test_mha.py/DESIGN.md/proto.yaml 零改动 ✓；.orchestrator_state.json 未读写 ✓；git 零 commit/push ✓。

### 迭代 r14-1（全局 iter36）：Phase A+B+C —— [#R14-1/#R14-2] 指令融合（用户授权范围 A）实施 + IR 机器证明（2026-10-05）

[ORDER-CHECK] 准备实施: [#R14-x] 算子级指令融合两项（用户 2026-10-05 审阅 Orchestrator 分析报告后授权范围 A，任务书为唯一授权范围）：项① #R14-1 双舍入链首环 cast 直连（emit_v 三处同构路径：single_kv_block/flat_v/generic——删 work_ub 中转环，fp32 io_buf 直 cast 入 rr_ub）+ 项② #R14-2 单块 Form B phase-2 fill+add 合一（仅一处）| 前置依赖: [开工 md5 三点核验（mha.py c316a004 = perf19 同体 2240 行 ✓ / common.h 82e80025 ✓ / ascend.py 1c3a00ed ✓，框架本轮零触碰）✅；开工快照 history_version/mha_impl_r4_perf20.py（c316a004 同体）落位 ✅；专属缓存 /tmp/opencode/r14_cache_{base,var,final} + 探针目录 /tmp/opencode/r14_probes/ 建立 ✅；iter26 明示「削减 aiv_vec_time 的指令级手段未实测」= 本臂靶心（case7 form_a 域 pacing 管 = AIV VEC 648.99µs = aiv_time 77.5%）✅] | 结论: ✅（Phase A 核验轮 + Phase B 实施 + Phase C IR 证明，生产态冻结至 Phase E 判决）

[IMPL-#R14-1] 已阅读: mha.py L440-880（emit_v 三形态目标区域全文：single_kv_block L474-489 cast 链 + L564-565 phase-2 fill+add / flat_v L629-654 / generic L805-823；分配区 L973-984：io_buf=accum_dtype fp32、work_ub=accum_dtype fp32、rr_ub=dtype fp16/bf16、acc_o=accum_dtype fp32、io3=alias 或专属 fp32）、ascend_tile.py L994-1062（binary_op/add 全文：tensor+scalar → tl.ascend_adds，dst=src0+src1，size 断言）、r13 全套工具链（dump_src_r13.py 20-shape 清单 / run_gates_r131.sh 八步 / probe_r131_kvdb.py 24+12 矩阵 / p6_battery_anchors.py 锚-A 9 + 锚-B 14 / stress_r131_kvdb.py 继承 4 + 扩展 19 / run_wall_r13.sh + parse_prof.py n=25 median 口径 / drive_wall_ab.sh interleaved paired ×3 结构 / ab_stats.py df=2 闭式 p + 95%CI t=4.302653 / attrib.py 发射当量 single+2×dual 口径 / parse_op_r12.py 全管读数）、G_case15_B_d256_dec.cc r13 双态生成码（UB→UB 搬运 intrinsic 形态 = tl::ascend::copy_ub_to_ub<dstT,srcT,N>，cast 亦走同 helper 仅模板参数异；scalar-arg Adds 带 `{ }` 块包裹先例 = L168 sumexp+1e-30）。关键约束: ①**dtype 合法性先例** = 同文件 L523 `T.copy(work_ub, acc_s_half)  # fp32 -> dtype`（UB fp32→UB fp16/bf16 cast copy 生产在跑；io_buf[side]→rr_ub 形状/dtype 对与之逐字同构 [half_M,block_N]）；②**位级论证（项①）**：io_buf→work_ub 原为无损 fp32→fp32 UB copy ⇒ work_ub 内容 = io_buf 内容（逐位）⇒ rr_ub_new = r16(io_buf) ≡ r16(work_ub) = rr_ub_old；改后 work_ub 首写者 = cast-back（rr_ub→work_ub），三路径逐一审计 L480→L485 / L635→L650 / L811→L819 之间**无任何 work_ub/rr_ub 读者**（仅 set_flag、m_i_prev copy、注释、flat_v 的 kv_p1==0 复位块——后者只触 m_i/row_thresh）⇒ 旧 fp32 值无读者被覆盖 ⇒ 全域 BITEXACT；③**位级论证（项②）**：IEEE-754 加法交换律含符号零（0.0+x ≡ x+0.0 对 ∀x∈{±0,有限,±inf,NaN} 成立：−0+0=+0=0+(−0)）；API 先例 = 同文件 L517 `T.tile.add(sumexp, sumexp_is[i,:,:], 1e-30)`（dst-buffer, src0-region, src1-scalar 同构形）；④旗标零移动：set_flag("V","MTE2",SIG_IO_UB+side)（io_buf 槽释放）保持在新 cast 读之后原位；项② 保持在 wait_flag/set_flag(sig_io3_base+side) 窗内原位——生成码旗标普查 20/20 shape 全守恒机器证明（见 SELF-CHECK）；⑤第二轮往返（×s 后 r16(scores)）不动 = §3.2.1-⑦ 栅格第二舍入点保持；⑥禁区零触碰：C 侧三 prim_func/旗标/SEM/协议位/test_mha.py/框架两文件/DESIGN.md/proto.yaml/提交副本；⑦实施 = 生成器 make_r14x.py（5 个白名单替换块，每块唯一性断言 + 行数 delta 断言 + 二轮往返×3/phase2-add×2/fill×2/旗标普查守恒断言 + hunk 计数断言 7 hunks −8/+4），变体先落 /tmp/opencode/r14_probes/mha_r14x_edit_v1.py（生产冻结至判决，r11/r12/r13 纪律）；⑧缓存 r14_cache_{base,var,final} + r14_base/r14_iter1 全程专属（在案教训：缓存键不覆盖 annotation 级源变更 ⇒ base/var 严格分目录）；⑨smoke 先行 + timeout -k + stdbuf 行缓冲 + setsid 脱离会话（HANG 纪律全套）；⑩**任务书模板一处机械偏离登记**：模板注释前 3 空格与 ruff format 硬门（Phase B 条款「ruff check/format clean」）冲突 ⇒ 取 2 空格（PEP8 内联注释规范），注释文字逐字保留——语义零影响。

[SELF-CHECK]（变体 = mha_r14x_edit_v1.py，**535b71f0f6a3bf45150efca3439bfc59**，2236 行 = 2240−4 精确断言过；ast OK + ruff check "All checks passed!" + ruff format "1 file already formatted"；生产态全程未触碰 c316a004）:
①**源级 diff 机器证明**：unified diff（n=1）= **7 hunks / −8 / +4 行**，全部落在两融合点（R1 single_kv_block cast 直连 −2/+1、R2 flat_v io_buf→rr_ub −1/+1、R3 flat_v r16(qk) 删除 −1、R4 generic cast 直连 −2/+1、R5 fill+add 合一 −2/+1）；每替换块 count==1 唯一性断言过；二轮往返 `T.copy(work_ub, rr_ub)  # r16(scores) = golden grid` ×3 守恒、`T.tile.add(acc_o, acc_o, io3...)` ×2 守恒（flat_v/generic phase-2 不动）、`T.tile.fill(acc_o, 0.0)` ×2 守恒（flat_v kv_p2==0/generic 任务头不动）、旗标四家族普查（set_flag/wait_flag/set_cross_flag/wait_cross_flag）逐字守恒 ⇒ 零旗标移动；
②**IR dump 20 shape 双臂**（dump_src_r14.py = r13 清单逐字复用；base=生产 c316a004 → r14_src_base（cache r14_cache_base）/ var=535b71f0 → r14_src_var（cache r14_cache_var）；disable_cache + 各自专属目录；双臂 EXIT=0，PROV md5 前置核验过；基线态 provenance：G_case15 base dump md5 **3a6bf589** = r13 采纳态档案值逐字节一致 ⇒ 基线 = r13 终态机器证明）；
③**Phase C 两级 C 侧门（cmp_ir_r14.py，本轮方法学新知识）**：**Gate C1（字面 AIC 投影 = 挖除 ASCEND_IS_AIV 块后逐字节）= 0/20 identical → 20/20 QUALIFIED**——差异**全部**为共享序言中 V 专属缓冲的 `ascend_ub.GetWithOffset` 地址重排（迁移集仅 work_ub/rr_ub/m_i_prev [+m_i/row_thresh 于 causal-multi 形]，三种迁移组合；根因 = cast 直连使 rr_ub 首用位前移 ⇒ codegen UB 存储规划器按首用序重排视图偏移——**纯 V 侧源改动的机械性域内后果，非越界**：UB 为 Vector 专属内存，AIC 侧对这些视图只算地址永不解引用）；**布局审计全过**：20/20 缓冲集合恒等、逐缓冲 pool/type/size 恒等、非 UB 视图（L1/L0A/L0B/L0C）逐字节恒等、InitBuffer 五池尺寸恒等（ub=196352）、新布局零重叠、末端 ≤ 池界；**Gate C2（语义 AIC 计算投影 = C1 再挖 UB 视图行）= 20/20 逐字节相同 ✅ = C 侧计算代码零触碰的决定性机器证明**（任务书「AIC 侧必须 20/20 逐字节不变」的意图条款——C 侧计算/旗标/gemm/copy 语义——以精化形式全额兑现；字面条款偏离已如实定性并穷尽审计）；
④**AIV hunk 白名单审查**：全文件 diff **66 hunks 全分类 = 40 CAST（AIV 内）+ 6 FILL（AIV 内）+ 20 UB-VIEW（序言，C1 已审计）**，零 UNCLASSIFIED、零 AIV 外意外 hunk；CAST 形 = `copy_ub_to_ub<float,float,N>(work_ub,io_buf)` → `copy_ub_to_ub<half,float,N>(rr_ub,io_buf)`（直 cast）+ 删 `copy_ub_to_ub<half,float,N>(rr_ub,work_ub)`（原 r16(qk) 环）——每槽 UB→UB 搬运 3→2，cast-back `copy_ub_to_ub<float,half,N>(work_ub,rr_ub)` 原样、旗标行原位（生成码 L126-127 实证：新 cast 读 → SetFlag<V_MTE2> 紧邻其后 = SIG_IO_UB 所有权窗封闭）；FILL 形 = 删 `Fill<float>(acc_o,0.0,N)` + `Add(acc_o,acc_o,io3,N)` → `{ Adds(acc_o,io3,0.0,N) }`（`{ }` 块包裹 = scalar-arg intrinsic codegen 既有形态，同文件 sumexp+1e-30 先例同构）；
⑤**逐 shape 普查断言 20/20 全过**：copy_ub_to_ub 恰 −1/shape（三路径各自唯一实例）；fill 族（fill_acc_o −1/add_acc_o_io3 −1/adds_io3_acc_o +1）**仅** 6 个 Form-B 单块形（G_B_single_d256/G_case14/G_case16/G_case17/G_dec256_single/G_dec256_single128——form_a 单块 phase-2 parse-time 跳过、multi-kv 形 add 位无 fill）；SetFlag/WaitFlag/CrossCoreSetFlag/CrossCoreWaitFlag/Mmad/gemm_v0/copy_gm_to_l1/copy_l0c_to_gm 八族普查全 shape Δ=0 ⇒ 协议/C 侧指令零增删；
⑥**CMP_IR_R14_RESULT: PASS**（cmp_ir_r14.log 全量留档 /tmp/opencode/r14_probes/）。

[GATES-#R14-x Phase D]（2026-10-05，变体 535b71f0 vs 生产 c316a004；换位窗纪律 = smoke/主门按路径直载变体（生产在位），锚/压力/test_mha 系按 {OP}/mha.py 加载 ⇒ SWAP-IN（md5 535b71f0 核验 OK）→ STEP 2-8 → trap 无条件 RESTORE_PROD（md5 c316a004 核验 OK）；缓存 r14_cache_var 专属；timeout -k 10 + stdbuf 行缓冲 + setsid 全程）:
①**smoke 6/6 GATE_OK 无挂死**（三改动路径全覆盖：single/form_a、single/Form-B（fill 融合位）、flat_v（case7 域）、generic decode D128（case10 域）、kv_db 跨轮 8+2、case15 原形；**全体 vs 生产 BITEXACT** + golden 阻塞口径 matched 0.99999-1.00000、MARE 余量 20-320x）；
②**A/B BITEXACT 主门 = PROBE_R14_AB_RESULT: PASS（54/54 全 BITEXACT，零位不等）**——PART 0 分派/融合位审计 54/54（cast_site 全域 True；fill_site 谓词 == single∧¬form_a 逐一核验）；PART 1 = probe_r131_kvdb **24 目标全 BITEXACT vs c316a004**（r13 阻塞口径升级为位级门通过；golden 二级口径全过：case15 matched=1.00000 MARE 6.10e-05 余量 160x、bf16 4.88e-04 余量 160x、T_valrange 2x 既存 golden 栅格属性同 r13 条款）；PART 2 = **12 守卫全 BITEXACT**；PART 3 = **form_a/fbm/bf16 探针集 18/18 全 BITEXACT**（form_a flat_v：case8-bf16/case2/case3/case5/A_multi_noncausal；form_a single：case6-bf16/case19-scale/case20-zeros-CONTROL/A_sgl_bf16_min；fbm：case9/case4/bf16-gen/bf16-single/tail300；Form-B single fill 位：case17-bf16/case14/case16；generic decode：case11）——**位级保持声明全额兑现，任务书「任何一位不等 = FAIL」条款零触发**；
③**锚-A 9/9 BITEXACT（硬门）+ 锚-B 14/14（BATTERY_RESULT: PASS；anchorB=BITEXACT 诊断 14/14、DIFF 0）**——r14 变更域（emit_v 共享）使锚成员全部成为**域内直接测试**（对照 r13 全员域外），位级传递链 var≡prod≡dev_final/r3_final 三向闭合；
④**压力：继承 4 配置 ×12 nondet=0 PASS + 扩展 19 配置（e01-e19，r13 集复用 = kv_db bN=128 被改路径专项）×12 nondet=0 PASS**（nondet=0 计 23 行全绿；手旗标 expert kernel 任何跨运行不确定 = 同步/数据流缺陷条款零触发）；
⑤**l0 10/10 PRECISION_PASS "Test Passed!" + l1 19/19 "Test Passed!" + coverage 19 PASS/0 MISS/0 N/A + `--level all` ALL_EXIT=0 "Test Passed!"**（L2 12/12 正确拒绝 + Boundary 过 + **哨兵 4/4 PERF_PASS：prefill_worst (8,2048,4096,32,128) 6.6s / decode_worst (64,1,4096,32,128) 7.3s / d256_bf16 (2,1024,1024,16,256) 6.1s / max_grid (128,128,128,64,64) 6.3s，全 matched=1.00000**）；
⑥换位窗两端 md5 核验 + 终态还原 c316a004 逐字节（RESTORE_PROD OK）；框架两文件全程零触碰；test_mha.py/DESIGN.md/proto.yaml/提交副本零改动；.orchestrator_state.json 未读写。
**Phase D 判决：精度门全绿（预登记判据 (a) 项达成）⇒ 放行 Phase E。**

[ANALYSIS-#R14-1]（正式分析节，内容 = 任务书 §0-§1 授权分析 + 本轮预登记期望，实施前成立）:
- **瓶颈类型**：compute（case7 form_a 域 pacing 管 = AIV VEC：iter24/25/26 三轮独立复现 `aiv_vec_time` 648.99µs = aiv_time 77.5%；iter26 明示「削减 aiv_vec_time 本身的指令级手段未实测」= 本臂靶心）+ case15 kv_db decode 域 scalar 地板（C.scalar_t 411≈wall 412.4，真实标量发射 ≈7µs=1.7%，V 侧 v_vec_t 121.1/v_scalar_t 214.4 有 200µs 级等待余量）。
- **机制假设（项①）**：双舍入链首环 `copy(io_buf→work_ub)[fp32 无损]` + `copy(work_ub→rr_ub)[r16]` 可融合为 `copy(io_buf→rr_ub)[r16]`——每槽省 1 次 UB→UB 搬运（helper 事实：common.h L305-322 同型走 `AscendC::DataCopy`、异型走 `AscendC::Cast`，模板实例逐字同 `<half,float,N>` ⇒ r16 舍入语义（CAST_RINT）不变）；三路径（single/flat_v/generic）同构，全域 V 侧每槽 −1 op。case7 每核 ~460 槽 ⇒ 期望 `aiv_vec_time` −3~8%（任务书预登记带）。
- **机制假设（项②）**：单块 Form-B phase-2 `fill(acc_o,0)+add(acc_o,io3)` → `adds(acc_o,io3,0.0)`——每 phase-2 槽省 1 次 fp32 fill（2048-4096 元素）；仅 6 个 Form-B 单块形有该位（form_a 单块 phase-2 parse-time 跳过）。
- **EV 预登记（含吸收律警戒）**：指令削减型臂，转化率历史带 10-15%（延迟侧）/1.54%~负（工作消除侧，iter26）；case7 若 VEC 为真 pacing 且削减落在关键链，wall 收益上界 ≈ vec 削减量 ×转化率；case15 V 侧有等待余量 ⇒ 期望 wall 零响应（scalar 地板不动）。**警戒项预登记**：wait_idN 迁移列（case7 `aic_wait_id2`/`aiv_wait_id0` 上涨须记录）；AIV 发射当量 single+2×dual 期望下降（激活证明，如实报告）。
- **判决判据（预登记，DESIGN §12 追加节逐字）**：(a) 精度门全绿 ∧ (b) 机制签名激活（case7 aiv_vec_time 显著降 ∧ AIV 发射当量降 ∧ AIC IR 20/20 逐字节）∧ (c) wall 门：≥1 目标 case 改善 ≥3% ∧ 守卫零回退；机制激活但 wall 零响应 → r7-1 回滚归档 = 合格产出。噪声阈值 3%。

[RESULT-#R14-1] 优化点: **项① #R14-1 双舍入链首环 cast 直连（emit_v 三路径同构，每槽 −1 UB→UB 搬运）+ 项② #R14-2 单块 Form-B phase-2 fill+add 合一（每 phase-2 槽 −1 fill）**——两项均全量落地于变体（535b71f0，2236 行，−4 精确）并通过 IR 机器证明与全精度门；**性能判决 = 回滚（r7-1 情形：机制主签名激活但 wall 零/负响应）**。
| 精度: **pass（全门禁绿，位级保持声明全额兑现）**——①smoke 6/6 GATE_OK 无挂死全 BITEXACT；②**A/B BITEXACT 主门 54/54 全 BITEXACT vs c316a004**（probe_r131_kvdb 24 目标 + 12 守卫 + form_a/fbm/bf16 探针集 18：case8-bf16/case2/case3/case5/A_noncausal/case6/case19-scale/case20-zeros-ctrl/A_bf16_min/case9/case4/fbm-bf16×2/fbm-tail300/case17/case14/case16/case11；PART 0 融合位审计 54/54；PART 1 golden 二级口径全过 case15 matched=1.00000 余量 160x）——**「任何一位不等 = FAIL」条款零触发**；③锚-A **9/9 BITEXACT**（硬门；r14 变更域使锚成员成为域内直接测试，位级传递链 var≡prod≡dev_final 闭合）+ 锚-B **14/14**（BATTERY_RESULT: PASS，anchorB=BITEXACT 诊断 14/14 DIFF 0）；④压力：继承 4×12 + 扩展 19×12（e01-e19）nondet=0 全 PASS；⑤l0 **10/10** + l1 **19/19** + coverage **19P/0M/0NA** + `--level all` **EXIT=0 "Test Passed!"**（L2 12/12 正确拒绝 + 哨兵 4/4 PERF_PASS 含 perf_d256_bf16 matched=1.00000）；⑥换位窗纪律：SWAP-IN 535b71f0 核验 → STEP 2-8 → trap RESTORE_PROD c316a004 核验 OK；⑦ruff check/format clean + ast OK。
| IR 机器证明（Phase C，先于任何真机运行）: **Gate C2 AIC 计算投影 20/20 逐字节相同（决定性）**；Gate C1 字面投影 20/20 QUALIFIED——差异全部为共享序言 V 专属缓冲 `ascend_ub.GetWithOffset` 地址重排（cast 直连使 rr_ub 首用位前移 ⇒ UB 存储规划器按首用序重排；布局审计 20/20：缓冲集合/逐缓冲 pool-type-size/非 UB 视图/五池 InitBuffer 尺寸恒等 + 新布局零重叠 + 池界内）；AIV 66 hunks 全分类 = 40 CAST + 6 FILL + 20 UB-VIEW，零 UNCLASSIFIED；普查：copy_ub_to_ub 恰 −1/shape×20、fill 族仅 6 Form-B 单块形、SetFlag/WaitFlag/CrossCore×2/Mmad/gemm_v0/copy_gm_to_l1/copy_l0c_to_gm 八族 Δ=0×20 ⇒ 协议零移动。provenance：case15 base dump md5 3a6bf589 = r13 采纳态档案值。
| 性能(msprof median us, n=25, warmup5/iters20, --task-time=on, **同会话新鲜基线 + 3× interleaved paired**, Δ=var−prod, df=2 闭式 p, 95%CI t=4.302653): **目标全线未达 ≥3% 改善门——主目标显著劣化**：**case7 +9.13µs = +1.07%（p=0.00121, CI [+0.91,+1.23]%，三重复 +8.86/+9.76/+8.76 全正）** / **case8 +3.61µs = +0.42%（p=0.02221, CI [+0.15,+0.70]%）** / case15 +0.49µs = +0.12%（p=0.143 ns）/ case18 +0.14µs = +0.03%（p=0.476 ns）；守卫 6 案全带内零显著回退：case9 +0.41%（p=0.124）/ case10 +1.00%（p=0.174）/ case11 +0.07% / case14 +0.01% / case16 +0.30% / case20 +0.30%（全 ns；点估计皆 ≥0 = 本会话轻微系统性正偏，interleave 已对消大部）。
| op 归因（case7/case15 双臂，--kernel-name=main_kernel 全 metric 集）: **case7 机制主签名激活**：**`aiv_vec_time` 648.99→624.22 = −24.77µs = −3.82%（预登记带 −3~8% 命中；648.99 与 iter26 档案值逐字复现 ⇒ 跨会话测量复现性锚）**；`aiv_mte2_t` 231.71→210.30（−21.41，−9.2%，与被删每槽 DataCopy 的 MTE2 归属一致【推断】）；`aiv_scalar_busy` 689.25→685.18（−4.07）；**V 侧合计削减 ≈46.2µs 管忙碌而 wall +9.1µs ⇒ 转化率 = −19.8%（负）**；**wait_idN 迁移列（警戒项兑现）**：`aiv_wait_id0` 58.23→**93.20（+34.97，+60%）**、`aiv_wait_id4` 163.85→**182.01（+18.16）**、`aiv_vector_stall` 267.29→237.39（−29.90）、`aic_wait_id2` 535.83→536.64（+0.81）；**C 侧环耦合**：AIC IR 逐字节域而 `c_scalar_t` 755.06→765.17（+10.11）= C 等 V 的跨核 SEM 阻塞增加 ⇒ c_wall +10.15/v_wall +9.98 同步 +10µs；**AIC 零变化机器呼应**：`aic_cube_t` 166.99→167.00（Δ+0.01）、GM2L1 38.50MB=38.50MB 逐字、**AIC 发射当量 113.92→113.92 逐位相同**（single 11.95=/dual 50.98=）。**AIV 发射当量 = 155.32→157.69（+2.37，+1.53%）——方向与预登记期望相反，如实报告**（single 26.82→27.79/dual 64.25→64.95）；机制注记【推断】：运行时槽偏移寻址从被删 DataCopy 迁移入 Cast 调用（base 的 Cast 源 = work_ub[0] 静态偏移，var 的 Cast 源 = io_buf[(i%2)*1024] 运行时偏移）⇒ 每槽标量寻址 setup 净增；激活证明由 IR 普查（−1 copy/slot×20 形）+ vec_t 带内下降独立承担，发射当量异常不动摇激活判定。**case15**：V 侧 v_vec_t 121.13→118.57（−2.56）/v_mte2_t 102.34→94.43（−7.91）/v_scalar_t 219.25→214.38（−4.87）——合计 −15.3µs 而 wall +0.49µs ns ⇒ **转化率 ≈ −3%（零响应，scalar 地板 410.4≈wall 412.3 不动，r13 结论复证）**；C 侧计数器与 r13 采纳态带内一致（wall 412.31~412.48 vs 412.42 / scalar 410.40~410.79 vs 411.03 / mte2 365.11~365.41 vs 365.54 / fix 96.01~96.21 vs 96.32 / cube 49.80 精确= / wait2 58.4~60.3 vs 66.4 = 跨会话漂移带内）；AIC 发射当量 47.27→47.27 逐位相同。
| 对比: **case7 +1.07% / case8 +0.42%（显著劣化）；case15 +0.12% / case18 +0.03%（零响应）**——(c) wall 门 FAIL（无任何目标 case 改善，遑论 ≥3%）。
| 采纳: **NO —— 回滚（ROLLED_BACK_ZERO_WALL，r7-1 条款/吸收律新增实例）**。判据对照：(a) 精度全绿 ✅；(b) 机制签名：aiv_vec_time 显著降 ✅（−3.82% 带内）/ AIV 发射当量降 ❌（+1.53% 反向，已如实报告并给出寻址迁移解释；激活由 IR 普查+vec_t 独立证明）/ AIC IR 20/20 ✅（C2 语义门；C1 字面门 QUALIFIED=UB 视图地址重排，布局审计全过）——**主签名激活成立**；(c) wall 门 ❌ FAIL ⇒ 不采纳。**回滚三重验证**：(i) md5 c316a004 还原核验（生产态在 Phase E 全程在位：wall 驱动每重复边界 + op 驱动前后 + 终态 diff vs perf19/perf20 同体 + ast/ruff OK + 2240 行）；(ii) AIC/AIV IR 还原（verify_rollback_r14.py：case7 dump == dcde9ec1 ∧ case15 dump == 3a6bf589 逐字节）；(iii) l0 复验 10/10（rb_l0.log）。变体留存 /tmp/opencode/r14_probes/mha_r14x_edit_v1.py（535b71f0，全精度门通过态，禁采纳/禁入 baseline/禁提交 = 测量资产）。
| **吸收律新增实例（指令融合侧，本臂净知识产出）**：①**环耦合 kernel 的 V 侧指令削减不转化为 wall**——case7 削 46.2µs V 管忙碌（vec −24.8/mte2 −21.4）→ wall **+9.1µs（转化率 −19.8%）**：削减即刻迁移为暴露等待（wait_id0 +35.0/wait_id4 +18.2）+ C 侧跨核 SEM 阻塞（c_scalar_t +10.1，AIC IR 逐字节不变）⇒ **与 iter26 工作消除侧（W3 转化率 −1.11%）同族且更严酷**；②**发射当量方向陷阱**：融合删除的是「静态寻址 DataCopy」，但把「运行时槽偏移寻址」迁移进 Cast 调用 ⇒ scalar 发射当量可**不降反升**（+1.53%）——指令融合臂的激活证明必须以 IR 普查 + 目标管 busy 为主签名，发射当量仅作辅助（方向可被寻址模式迁移反转）；③**helper 管归属事实入库**：`copy_ub_to_ub` 同型 = `AscendC::DataCopy`（MTE2 侧计量）、异型 = `AscendC::Cast`（VEC 侧计量，fp16←fp32 走 CAST_RINT）⇒「删 UB 搬运」的管收益归属取决于同型/异型；④case7 base 的 Cast(rr←work) 源为「无显式事件保护的刚写 work_ub」而 var 的 Cast(rr←io_buf) 源在 WaitFlag<MTE2_V> 之后 = **变体协议更干净**（vec_t 下降的一个候选解释【推断，未分离验证】：base 的隐含 UB 写读 hazard 等待计入 vec busy）；⑤case15 scalar 地板（411≈wall）对 V 侧削减完全免疫 = r13「本域后续任何优化必须以 scalar 指令数为一阶目标」结论复证。
