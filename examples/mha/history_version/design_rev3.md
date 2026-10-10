# mha 算子设计文档（rev3）

> Stage 1 产出（tilelang-op-design + tilelang-op-test-design 场景 A），revision_index=2。
> 编程模式由用户（需求方）在 Primary 上下文明确决策变更为 **Expert**（2026-09-19，
> 独立立项重做设计），本文档不得改回 Developer/混合模式。
> 上一版设计：`history_version/design_rev2.md`（Developer threads=1，已验证语义基准；
> 位级语义载体 `history_version/mha_impl_developer_final_6505.py`，cann-bench 65.05/100）。

## 0. 修订记录（rev2 → rev3）

### 0.1 变更动因（用户需求变更驱动，非 DESIGN_ERROR）

Developer 版（rev2 设计 + 两轮 Stage 3 调优，15 次迭代、12 项优化全采纳）终态：
cann-bench 综合 **65.05/100**、平均加速比 **0.55x**（最佳 case10 0.85x，最差大
prefill case7/8/9 ≈ **0.17-0.19x**）。调优收口结论（perf_tuning/final_report.md
§S2-5，证据 optimization_log.md 15 轮门禁）：**Developer 模式内已触结构边界**——
tile 尺寸三面顶硬限（L0B/L1/UB）、指令数已 2D 化、迭代数已裁剪+单块化、核映射已
Fixed Core，剩余差距主体为**模式结构性成本**，全部由 AUTO pass 生成、源码不可表达：

1. **AUTO_SYNC 逐 op barrier 链**：每个 vector op 间插入 `PipeBarrier<PIPE_ALL>`
   全排空或成对串行 flag（final_report §5-#9 codegen 实证），无法批量流水；
2. **每 KV 迭代 3 段 CV workspace GM 往返（w1/w2/w3）+ 每迭代 cross-flag 对**：
   AUTO_CV_COMBINE 在 threads=1 cv_1_1 下自动生成，C↔V 逐迭代串行交接，
   GEMM1(i)/softmax(i)/GEMM2(i) 无重叠；threads=2 消 workspace 路径被 R8 框架
   bug 阻塞（rev2 §10.1-R8）；
3. **单 AIV 执行全部 vector 工作**（threads=1 cv_1_1 无行分担）。

用户在 Primary 上下文决策：**编程模式 Developer → Expert，独立立项重做设计，目标
cann-bench 加速比逼近 1.0x**。Expert 化收益路径（flash_attn_optimize.md §1-§8，
仓内已验证载体：gqa_fwd_varlen 17.78ms/51.72 TFlops、mha_sink_fwd CI 通过）：
显式 L1/UB/L0A/L0B/L0C 管理 + T.mma L0 双缓冲 + num_stages 批量流水 +
cross_interval 批量跨核同步 + 细粒度 pipe-pair flag + 批量 Softmax（deferred
rescale）+ vid 二分（双 AIV 分担 vector 工作）+ Fixed Core 静态任务分配。

### 0.2 rev2 → rev3 变更清单

| # | 章节 | rev2 内容（Developer 终态） | rev3 修订（Expert） | 依据/证据出处 |
|---|------|---------------------------|--------------------|--------------|
| M1 | §2/§7/§8 | Developer：四 pass_configs 全开、threads=1 cv_1_1、AUTO 同步、消除写法（无 workspace/vid/T.Scope） | **Expert**：四 pass_configs 全 False、`T.Kernel(core_num, is_npu=True) as (cid, vid)`（vid=blockIdx.y extent 2 结构性绑定）、手动 `T.Scope("C"/"V")`、显式 GM workspace（`workspace_idx=[4,5,6]`）、手动 set_flag/wait_flag + set_cross_flag/wait_cross_flag、手动 vid 二分（half_M=bM//2 行/vid） | 用户决策（Primary 2026-09-19）；kernel.py:249-263（is_npu 无 threads → Expert frame）；ir.cc:281-290（cid=blockIdx.x, vid=blockIdx.y extent 2）；programming-model-guide §1/§2.2；flash_attn_bhsd.py:36 / gqa_fwd_varlen.py:220 / mha_sink_fwd_bhsd.py:187 |
| M2 | §3/§6 | 单一形态：gemm_v0 + T.serial(kv_iters) 逐迭代 C↔V 串行 | **C-scope 双形态**（host 元数据分派，V-scope 共享）：Form A（prefill S>16 ∧ D≤128）= T.mma + L0A/L0B/L0C 手动双缓冲 + 细粒度 flag 全流水；Form B（decode S≤16 全 D + prefill D=256）= gemm_v0（kL0Size 三档继承）+ barrier 级同步。两形态共用 num_stages 批量 workspace 协议 + cross_interval 批量跨核同步 | mha_sink_fwd_bhsd.py:206-360（Form A 载体）；flash_attn_bhsd.py:86-114（Form B 载体）；gemm_v0 签名 ascend.py:418（kL0Size/n_actual）；T.mma = npu_gemm 别名（language/__init__.py:75、customize.py:115）；Form B 保留理由 = decode bN=512/256 的 L0B 预算由 gemm_v0 内部 kL0Size 分片解决（session-2 #13/#10b 实测，final_report §S2-3-1）；D=256 prefill 避免 mma 手动 K/N-split 的 L0B 64KB 顶格风险（§5.2 分派谓词论证） |
| M3 | §4.3/§4.5/§5.3 | alloc_shared/fragment + MEMORY_PLANNING 自动放置；实配可优于 naive（session-2 教训） | **显式 alloc_L1/alloc_ub/alloc_L0A/L0B/L0C + 六配置类预算表**（P1/P2/P3/D1/D2/D3）；Expert 下 MEMORY_PLANNING=False → planner 仅顺序编址、**无 liveness 合并，naive 声明和≈实配**；全部顶格（≥90% 占用）条目强制**实编 GetWithOffset 读数验证** | ascend_memory_planning.cc:59-92（auto=False 仅编址，address_map attr 恒生成）；codegen_ascend.cc:805-840（AllocateNode 一律走 address_map_ ICHECK）；mha_sink:190-231（分配先例）；final_report §S2-3-2（教训反向固化） |
| M4 | §4.4/§8.2 | "片上直连"表述（AUTO_CV 内部实际经 GM workspace 中转，调优期实证 w1/w2/w3 三段往返） | **显式 3-workspace 数据流**（w1 scores fp32 / w2 P dtype / w3 PV fp32，`[core_num, NUM_STAGES, ...]` 索引）+ **虚拟通道禁令**：Expert kernel 不得写 `T.copy(L0C→UB)`/`T.copy(UB→L1)` 裸片上拷贝——两者为虚拟通道，框架一律重写为 GM workspace 路径；"每 KV 迭代 0 段 GM 往返"物理不可达，rev3 目标 = 3 段保留但批量化/流水化/私有索引化（§8.2 诚实对比表） | ascend.cc:289/298（copy_ub_to_l1/copy_l0c_to_ub 标记 virtual_channel）；ascend_workspace_reduction.cc:1001-1004（替换表 {copy_ub_to_l1→copy_ub_to_gm, copy_l0c_to_ub→copy_l0c_to_gm}）；tl_templates grep：copy_l0c_to_ub 无设备实现（仅 copy_l0c_to_gm，common.h:190）；三个 Expert 参考实现全部显式 workspace（flash_attn_bhsd.py:32-34/96/103/110；gqa:216-218；mha_sink:183-185/313/333/357） |
| M5 | §5.2/§5.4 | bM 16/32/64 + bN 三档 + trim + 尾块三重保障（单 V 核全行） | **全部继承 + vid 二分尾块新规则**：每 vid 行数 `rows_vid = clamp(valid_m - vid*half_M, 0, half_M)`，rows_vid=0 时运行期 if 守卫跳过写回（decode valid_m=1 → vid0 写 1 行、vid1 跳过）；掩码/softmax 对全部 half_M 行计算（垃圾行有限、行独立，rev2 论证继承） | mha_sink:614-622（vid 半区写回先例，整除场景）；运行期 if 语句先例 = Developer 终态 #4（mha.py:329 运行期 TIR 条件已验证）；copy_ub_to_gm 运行期 maskShapeM（common.h:269-283） |
| M6 | §9.2 | L0 十用例 + Developer 路径映射 | **L0 十用例表逐字继承**（接口契约不变）；路径映射更新为 Expert 形态（Form A/B、vid 二分、num_stages 批量、deferred rescale）；新增**非阻塞诊断锚点**：fp32 w1/w3 主线下 rev3 输出目标与 Developer 终态位级一致（失配≠FAIL 但触发同步缺陷根因分析） | test-design 场景 A 重跑（§9.2）；位级一致性论证 §9.3.3-6；flash_attn_optimize §9(a) 正确性要求（任何数值错误=同步/数据流缺陷） |
| M7 | §10 | R8/R9 为 Developer 语境"禁用依据" | **R8/R9 Expert 复检结论 = 均不触发**（§2.2-c/d 源码证据链 + §10.1 条目更新）；新增 R10-R16（Expert 特有：BSHD×Expert 首例组合、flag 配对、虚拟通道误用、顶格预算、workspace 足迹、掩码链 V-scope 首用、整体劣于 Developer 终态回退） | ascend_vid_reduction.cc:1146-1152/307/666/1109；ir.cc:281-318；kernel.py:249-263；codegen_ascend.cc:805-840 |
| M8 | §12 | Stage 3 调优目标（追加节，锚点 54.18） | **重写为 rev3 性能目标**：锚点 65.05/0.55x + per-case 终态表；目标平均加速比逼近 1.0x（里程碑 M1 ≥0.75x / M2 ≥0.9x）；case7/8/9 ≥0.4x 中期；decode 类不回退；**中止/回退条款**（精度不收敛或整体 <65.05 → 回滚 Developer 终态） | final_report §S2-1 per-case 表；用户目标（Primary 决策）；备份就位 history_version/{mha_impl_developer_final_6505.py, design_rev2.md, test_mha_developer_final.py} |

### 0.3 继承清单（不变项，逐字/等值保留）

| 继承项 | 出处 | rev3 落点 |
|--------|------|-----------|
| 算子语义/公式/支持域（BSHD、fp16/bf16、右下角对齐 causal、全掩码行→0、B≤128/S≤2048/S_kv≤4096/N≤64/D∈{64,128,256}） | rev2 §1/§4.1/§4.2 | §1/§4 逐字 |
| host 契约 `mha(query, key, value, scaleValue=-1.0, is_causal=False)` + 全部断言 + out_idx=[3] 语义（零张量操作） | rev2 §3.3/§4.8 | §3.3/§4.8（workspace_idx 由框架注入，host 调用形式不变——先例 flash_attn_bhsd.py:254 三参调用七参 main） |
| golden 函数（需求方提供，原样） | rev2 §9.1 | §9.1 逐字 |
| 精度口径（**用户裁决 2026-09-18，不可更改**：阻塞 = 混合容差 + MARE_abs < 10×Threshold；MERE/MARE_rel 参考值） | rev2 §9.3.1/§9.3.2 | §9.3 逐字 |
| L0 十用例表 + 哨兵四条 + kernel 路径 120s 门 + golden 降本规定 | rev2 §9.2/§9.4 | §9.2/§9.4（路径映射更新） |
| **算法层优化全部 12 项**（模式无关）：① causal KV 运行期裁剪 ② broadcast 2D 行循环替代 ③ 掩码链 2D 化+row_thresh hoist ④ 全可见迭代运行期跳过 ⑤ 单块预归一化特化（谓词 S_kv≤bN）⑥ sumexp+1e-30 保护 ⑦ score 双舍入栅格对齐 ⑧ decode 窄块 bM=16 ⑨ bN 三档自适应（decode 512/256/128、prefill 128）+ gemm1_kL0 三档（32/64/128）⑩ Fixed Core ⑪ valid_m/valid_n 运行期尾块 ⑫ BSHD 行跨步 DMA | Developer 终态 mha.py（65.05 实现）+ final_report §3/§S2-2 | §3.2.1 逐项落点映射表 |
| proto.yaml（接口契约不变） | rev2 §11.5 | §11.5 复核无变化 |
| test_mha.py 测试资产（golden + L0 10/L1 19/L2 12/Boundary 6/哨兵 4 全套件 + 双口径 harness + 阻塞判定逻辑，844 行） | Stage 2/3 产出 | §9.5 零改动复用（host 签名不变即成立） |

### 0.4 为何不会再犯同一错误（rev1/rev2 教训对照）

- **rev1 教训（threads=2 默认陷阱、MERE 口径）**：rev3 的 vid=2 是 Expert 启动帧的
  **结构性绑定**（kernel.py:249-263：is_npu 且无 threads → vid=blockIdx.y extent 2；
  带 threads → dev_mode frame），与 Developer threads=2（threadIdx.x 绑定、激活
  AscendVidReduction 自动折半）是**不同机制**——rev3 全文不使用 threads 参数，
  vid 二分为源码显式（[bM//2] buffer + vid*half_M 偏移），不存在"自动折半" pass
  介入点（§2.2-c 五点证据链）。MERE 口径已由用户裁决关闭（§9.3.1 逐字继承）。
- **rev2 教训（R8/R9 框架 bug、哨兵门限定义）**：R8/R9 在 Expert 形态下逐项复检
  为"不触发"（§2.2-c/d，源码行号证据），且 rev3 保留其 Developer 语境禁令作为
  回退护栏（任何"改回 Developer threads=2 / T.Pipelined"的尝试仍命中 rev2 禁令
  与本文 §10.2）；哨兵 kernel 路径门与 golden 降本规定逐字继承（§9.4）。
- **session-2 教训（MEMORY_PLANNING 实配 ≪ naive 手算）**：Expert 下该教训**反向
  固化**——MEMORY_PLANNING=False 时 planner 无 liveness 合并
  （ascend_memory_planning.cc:59），naive 声明和≈实配，预算表（§4.5/§5.3）按
  naive 和判定；全部"=上限"顶格条目（P2/P1 l0b 64KB、Form B 内部 L0B 64KB、
  D1 L1 276KB 等 ⚠ 标记）一律以**实编 GetWithOffset/编译探针读数**为 Stage 2
  强制验证步骤，手算不作最终依据。
- **session-1 #2 教训（scratch 与跨核 copy 目的地别名 → 静默损坏）**：Expert 下
  升级为硬性设计规则（§8.6-2）：V-scope 所有 broadcast/mask scratch（buf_2d/rt_2d）
  与跨核 workspace copy 目的地（io_buf/work_ub/acc_s_half 经 MTE2/MTE3 flag 管程）
  严格分离；复用必须有完整双向 flag ownership 闭环（flash_attn_optimize §8
  "复用物理缓冲的前提是完整的 ownership 生命周期"原文约束）。

---

## 1. 概述

### 1.1 算子名称

mha（Multi-Head Attention，多头注意力前向）

### 1.2 功能描述

对 BSHD 布局（`[B, S, N, D]`）的 Q/K/V 三个张量执行缩放点积多头注意力：每个
(batch, head) 独立计算 `softmax(Q·Kᵀ·scaleValue)·V`，支持可选的右下角对齐因果掩码
（causal mask），输出与 query 同 shape、同 dtype、contiguous。

### 1.3 数学公式

$$
y = \mathrm{softmax}\!\left(Q K^{\mathsf T} \cdot s\right) V,\qquad
s = \begin{cases} \texttt{scaleValue} & \texttt{scaleValue} > 0 \\ 1/\sqrt{D} & \texttt{scaleValue} \le 0 \end{cases}
$$

对每个 batch `b`、head `n`、query 行 `i`、key 列 `j`：

```
scores[b,i,n,j]   = (Q[b,i,n,:] · K[b,j,n,:]) * s
若 is_causal:      scores[b,i,n,j] = -inf   当 j > i + (S_kv - S)     # 右下角对齐 45° 对角线
attn[b,i,n,:]     = softmax(scores[b,i,n,:], dim=-1)                   # 全 -inf 行 → 权重全 0（golden all_masked 保护）
y[b,i,n,:]        = Σ_j attn[b,i,n,j] * V[b,j,n,:]
```

### 1.4 算法描述

多步融合算子（GEMM → mask → softmax → GEMM），采用 **Flash-Attention 风格的
online softmax 分块算法**（数学上与一次性 softmax 严格等价）。rev3 在 rev2 语义
基础上引入 Expert 批量流水执行结构（**deferred rescale 批量 softmax**，与逐迭代
rescale 代数同值且运算次序逐位一致，见 §8.4/§9.3.3-6）：

1. **任务划分**：每个 kernel 任务负责一个 `(b, n, s_block)` 的输出条带
   `[block_M, D]`；Q 条带常驻 L1，沿 S_kv 维分块迭代 K/V。Fixed Core：
   `core_num = min(block_num, 设备 cube 核数)` 个核各自循环处理连续任务块
   （继承 Developer 终态 #5）。每任务行由 **2 个 AIV 按 vid 二分**（各
   half_M=bM//2 行，Expert 结构性 cv_1_2）。
2. **GEMM1（Cube）**：`acc_s = Q_tile · K_tileᵀ`（fp32 累加到 L0C），结果经
   FIX 管写入 `workspace_1[cid, i]`（GM，批量 NUM_STAGES 槽位）。
3. **双舍入 golden 栅格对齐（Vector，继承 rev2-A2）**：golden 的 scores 实际经历
   两次半精度舍入——`round16(q·k)`（fp16/bf16 matmul 输出）再 `round16(· × s)`
   （fp32 opmath 标量乘后落回 dtype）。kernel 精确镜像：V 侧从 w1（**fp32 主线**）
   读入后先做 fp32→dtype→fp32 round-trip（= round16(qk)），fp32 域乘 `s`，再做
   一次 round-trip（= round16(scores)）。2 的幂 scale（D=64/256）位级等价；
   D=128（s=1/√128 非 2 的幂）≤1ulp 对齐。
4. **因果/边界掩码（Vector，pre-exp -inf 掩码，继承 rev2）**：可见性
   `visible(i,j) = (j ≤ min(i + offset, S_kv-1))`（causal）或 `j < S_kv`（非 causal），
   不可见位置置 `-inf`，与 golden 的 `masked_fill(-inf)` 语义逐位一致；
   `offset = S_kv - S`（右下角对齐；S==S_kv 时退化标准下三角）。掩码链 2D 化 +
   row_thresh 任务级 hoist + 全可见块运行期跳过（继承 Developer 终态 #3/#4），
   作用于本 vid 的 half_M 行。
5. **批量 online softmax（Vector，phase 1）**：对批内每个 KV 块 i：
   `m_new_i = max(m_prev_i, rowmax(acc_s_i))`；存 `r_factors[i] = m_prev_i - m_new_i`
   （phase 2 再 exp——exp 确定性保证与逐迭代 exp 同值）；
   `P_i = exp(acc_s_i - m_new_i)` 舍入到 dtype 写 `workspace_2[cid, i]`；
   存 `sumexp_is[i] = rowsum(P_i)`。
6. **GEMM2（Cube）**：`PV_i = P_i · V_tile_i`（P 从 w2 经 p_l1 进 L0A），fp32
   结果写 `workspace_3[cid, i]`。
7. **O 累加批（Vector，phase 2，deferred rescale）**：对批内每个 i：
   `r_i = exp(r_factors[i])`；`sumexp = sumexp*r_i + sumexp_is[i]`；
   `acc_o = acc_o*r_i + PV_i`（w3 读入 fp32）——与 rev2 逐迭代 rescale 的算式与
   运算次序逐位一致（§9.3.3-6 论证）。
8. **终归一化 + NaN 保护（Vector，继承）**：`acc_o /= (sumexp + 1e-30)`。
   `+1e-30` 为全掩码行保护：正常行 sumexp ≥ 1（行内最大元素贡献 exp(0)=1），
   加 1e-30 低于 fp32 ulp、数值不变；全掩码行 sumexp=0、acc_o=0 → 输出 0，
   与 golden all_masked→0 严格一致（不产生 NaN）。声明约束（causal ⇒ S≤S_kv）下
   全掩码行不可达，属防御性设计。
   **单 KV 块预归一化特化（继承 rev2-A3/Developer #8d，编译期谓词
   `kv_iters == 1` 即 `S_kv ≤ block_N`）**：sumexp 在唯一迭代内即终值，phase 1
   内先 `P̂ = round16(P / (sumexp_is[0] + 1e-30))` 再写 w2，phase 2 跳过 rescale
   与终除法——与 golden 一次性 softmax（归一化后舍入 P̂ 再 matmul）结构同构
   （近位级精确锚点：Developer 版 case4 bf16 位级精确、单块用例 MERE≤3e-6）。
9. **写回**：fp32→dtype 舍入后，每 vid 按 `rows_vid = clamp(valid_m - vid*half_M,
   0, half_M)` 行写回 `Output[b, s0+vid*half_M : s0+vid*half_M+rows_vid, n, :]`；
   rows_vid=0 的 vid 运行期跳过写回（§5.4）。

**Host 侧预处理声明**：host 路径**只做元数据校验与编译期配置推导**（纯 Python
整数/浮点运算：scaleValue 解析、block 尺寸/形态选择、core_num 设备属性查询、
block_num 计算、shape/dtype 断言），**不触碰任何张量数据**——无 transpose/
contiguous/padding/cat 等物理化操作（§4.8 HOST-METADATA-AUDIT）。workspace 张量
由 tilelang 框架按 `workspace_idx` 自动分配注入，host 调用形式不变。
BSHD→(b,n) 条带的"逻辑重排"完全由 kernel 内行跨步（row-stride）DMA 切片完成，
无 GM 物理重排 pass。

### 1.5 数据流图（Expert；Form A 展示，Form B 将 mma 段替换为 gemm_v0，workspace 协议相同）

```
host: 元数据校验 + 配置/形态分派推导（不触碰数据；workspace 由框架注入）
                                  │
════════ Cube 核（AIC，T.Scope("C")，每 cid 1 个） ════════
GM Q[B,S,N,D] ──T.copy(行跨步, valid_m 行, MTE2)──► L1 q_l1[bM,D]      （任务级常驻）
GM K[B,S_kv,N,D] ─T.copy(行跨步, valid_n, MTE2, 批内逐块)──► L1 k_l1[bN,D]
   q_l1 ──T.copy(MTE1, 批首2槽)──► L0A l0a_s[side]  ┐
   k_l1 ──T.copy(transpose=True, MTE1)──► L0B l0b_s[side] ├─ T.mma ─► L0C l0c_s[side] fp32
                                                    ┘
L0C l0c_s[side] ──T.copy(FIX)──► GM workspace_1[cid, i, bM, bN] fp32   （每 CROSS_INTERVAL 块 set_cross_flag）
GM workspace_2[cid, i] ──T.copy(MTE2)──► L1 p_l1[bM,bN] dtype
GM V ──T.copy(MTE2)──► L1 v_l1[bN,D] ──T.copy(MTE1)──► L0B l0b_o[side]
   p_l1 ──T.copy(MTE1)──► L0A l0a_o[side] ─ T.mma ─► L0C l0c_o[side]
L0C ──T.copy(FIX)──► GM workspace_3[cid, i, bM, D] fp32
════════ Vector 核（2×AIV，T.Scope("V")，每 vid 半区 half_M 行） ════════
GM w1[cid, i, vid 半区] ──T.copy(MTE2)──► UB io_buf[io_side] fp32 ──T.copy(V)──► work_ub
   work_ub: r16(qk) ─► ×s(fp32) ─► r16(scores) ─► 掩码链(-inf, 2D compare/select)
   ─► reduce_max/合并 m ─► r_factors[i]=m_prev−m_new ─► broadcast sub ─► exp
   ─► reduce_sum → sumexp_is[i] ─►（单块特化: ÷(sumexp_is[0]+1e-30)）─► cast acc_s_half
UB acc_s_half ──T.copy(MTE3)──► GM workspace_2[cid, i, vid 半区] dtype
GM w3[cid, i, vid 半区] ──T.copy(MTE2)──► io_buf ──► work_ub
   phase2: exp(r_factors[i]); sumexp=sumexp*r+sumexp_is[i]; acc_o=acc_o*r+work_ub
循环结束: acc_o ÷ (sumexp+1e-30)（通用路径）─► cast out_half ─►
GM Output[B,S,N,D] ◄──T.copy(MTE3, 行跨步, rows_vid 行, 运行期 if 守卫)──── UB
```

内存层级路径全程合法：GM ↔ L1 ↔ L0A/L0B → L0C →(FIX) GM workspace →(MTE2) UB
→(MTE3) GM。Cube↔Vector 经**显式 GM workspace**（[cid, stage] 私有索引）——这是
本框架下 CV 交互的唯一物理路径（L0C→UB / UB→L1 为虚拟通道，框架重写为 GM 路径，
§8.2 源码证据）；无 GM→L0 直搬、无跨级违规。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Expert（用户在 Primary 上下文明确决策，2026-09-19，不可更改；
rev2 的 Developer 选型随本次模式变更废止）

### 2.2 选型理由与 Expert 形态选型（最高风险决策，源码论证）

**（a）为什么 Expert**：用户决策 + 两轮调优实证 Developer 结构边界（§0.1 三项
结构性成本；final_report §S2-5 收口结论原文："剩余差距主体为模式结构性同步成本，
属 Expert 化设计变更范畴……独立立项由需求方决策"）。Expert 收益路径全部有仓内
已验证载体（gqa_fwd_varlen 17.78ms/51.72 TFlops；mha_sink_fwd CI；
flash_attn_optimize.md §1-§8 逐项对应）。

**（b）Expert 形态选型结论：`(cid, vid)` 双绑定 + vid 手动二分 + Cross-core
Semaphore + T.Scope("C"/"V")——这是本项目唯一存在的 Expert 形态**：

- 候选"threads=1 手动 C/V 分时（T.Scope 切换）"**在框架中不存在**：
  `T.Kernel(n, is_npu=True)` 不带 threads 参数 → `tilelang.is_npu_kernel_frame`
  （kernel.py:249-250）→ 启动帧**恒**推入 `cid=blockIdx.x` 与 `vid=blockIdx.y,
  extent 2` 两个绑定（ir.cc:281-290，extent 2 为硬编码字面量）。带 threads 参数
  则进入 dev_mode frame（kernel.py:262，Developer 语境）。Expert 下每任务 =
  1 AIC + 2 AIV（cv_1_2）；C/V 是**物理不同核**（T.Scope("C") 在 AIC 执行、
  T.Scope("V") 在 AIV 执行），"单核分时"无对应启动帧，不可选。
- 候选"threads=2 手动 vid 分工（Cross-core Semaphore）"即上述唯一形态（vid
  extent 2 为结构性绑定，非 threads 参数）。vid 二分收益：vector 工作量减半/核
  （softmax/exp/掩码链在 2 个 AIV 并行执行各自 half_M 行），对 vector-bound 的
  decode 与 prefill 均直接生效（对 Developer cv_1_1 单 AIV 是结构性升级）。
- 三个仓内 Expert 参考实现同形态：flash_attn_bhsd.py:36、gqa_fwd_varlen.py:220、
  mha_sink_fwd_bhsd.py:187（均 `as (cid, vid)` + [bM//2] UB buffer + vid*half_M
  偏移 + T.Scope("C"/"V") + workspace + cross_flag）。

**（c）R8 复检（强制检测，结论：Expert 形态不触发）**——rev2 §10.1-R8 的
`AscendVidReduction` pass × BSHD 4D 结构性 bug 在 Expert vid=2 形态下**不可能
触发**，证据链（全部源码实证）：

1. **触发键不匹配**：该 pass 仅在遇到 `thread_tag == "threadIdx.x"` 的
   thread_extent AttrStmt 时设置 `vid_`/`threads_cnt_ = 2`
   （ascend_vid_reduction.cc:1146-1152）。Expert 启动帧把 vid 绑定到
   **blockIdx.y**（ir.cc:288-290），全函数不存在 threadIdx.x 绑定 →
   `threads_cnt_` 恒为默认值 1。
2. **改写路径全部短路**：pass 的三处改写入口（BlockNode :307、:666、:1109）均有
   `if (threads_cnt_ != 2)` 前置短路 → Expert 下 pass 为**全局 no-op**（pass 本体
   仍在 pipeline 无条件注册：phase.py:54，但空转，无 buffer/extent 改写）。
3. **R8 三症状结构性不可达**：症状①（UB buffer 自动折半而 region extent 不改写 →
   2× 行 OOB 静默互踩）、②（运行期 valid extent → :430 ICHECK 编译崩溃）、
   ③（1D 列维 buffer 自动减半 → 64 元素写 32 槽位）全部源于 pass 的**自动**
   buffer 折半/extent 改写；Expert 的 vid 二分是**源码显式**的（用户直接声明
   `[block_M//2, ...]` UB buffer 并手写 `vid*half_M` 偏移，三参考实现同形）——
   不存在自动改写环节，三症状无发生机制。
4. **关联 pass 越靶**：AscendWorkspaceReduction 的 vid 折半逻辑（maskShapeM /=
   vector_cnt，ascend_workspace_reduction.cc:1385-1390）只作用于**虚拟通道拷贝**
   （target set = {copy_ub_to_l1, copy_l0c_to_ub}，:1001/:115-118）；rev3 强制
   显式 workspace 拷贝（§8.6-R12 禁令），target set 永不命中。
5. **先例佐证**：三个 Expert 参考实现（含运行期 eff_kv_iters 边界的 mha_sink、
   varlen 运行期长度语义的 gqa_fwd_varlen）在该 pass 存在下 CI 通过。

   **残留风险（编号 R10，非 R8）**：BSHD 4D 行跨步切片 × Expert vid 偏移 ×
   运行期 valid extent 的**组合**为仓内首例（参考实现均 BHSD 或整除场景）→
   Stage 2 强制前置探针（rev2 verify_bshd_slice/verify_t1a 探针族移植到 Expert
   脚手架，§10.1-R10）。**Developer threads=2 禁令继续保留**：若任何回退尝试
   改用 Developer frame + threads=2，R8 全证据仍然成立（rev2 §10.1-R8 逐字有效，
   本文 §10.1-R8 状态栏保留）。

**（d）R9 复检（强制检测，结论：Expert 形态不适用）**——rev2 §10.1-R9 的
`T.Pipelined(num_stages=2)` codegen bug（`Cannot find pre-allocated address for
buffer: k_l1`，codegen_ascend.cc:830 ICHECK）依赖两个前提，Expert 下均不成立：

1. 前提一"MEMORY_PLANNING 自动注入双缓冲 L1 操作数且未注册 address_map"：
   Expert 配置 `TL_ASCEND_MEMORY_PLANNING: False` → planner 仅对显式 alloc 顺序
   编址并全量注册 address_map（ascend_memory_planning.cc:59-92：auto=False 时
   external_address_map/顺序放置，address_map attr 恒生成；codegen_ascend.cc:1245
   读取该 attr），不存在"注入了但未注册"的 buffer。
2. 前提二"使用 T.Pipelined"：rev3 循环形态 = `T.serial`（外层 eff_num_outer +
   内层 batch_iters，运行期 TIR 边界）+ **手动 ping-pong**（side=i%2 的 L0 双槽、
   io_buf 双槽）——三参考实现同形态（mha_sink:278/284/433/439 运行期边界
   T.serial 在 Expert 下 CI 验证）。**T.Pipelined 在 rev3 全文禁用**（亦无收益：
   流水由 num_stages 批量结构 + 手动双缓冲表达）。

**（e）循环形态选择结论**（R9 检测的正面输出）：外层 `T.serial(eff_num_outer)`
（运行期因果裁剪边界；**C/V 双 scope 同式复制计算**保证 flag 配对平衡，
mha_sink:394-400 注释先例逐字适用）+ 内层 `T.serial(batch_iters)`
（`batch_iters = min(NUM_STAGES, eff_kv_iters - k*NUM_STAGES)`，运行期
T.if_then_else）+ 手动 side=i%2 双缓冲（L0 池 + io_buf）。

### 2.3 模式影响

| 维度 | 本算子的选择（Expert） |
|------|----------------------|
| 内存分配 | `T.alloc_L1` / `T.alloc_ub` / `T.alloc_L0A` / `T.alloc_L0B` / `T.alloc_L0C`（allocate.py:140-158）；**不使用** alloc_shared/alloc_fragment；不写 `T.annotate_address`（planner 顺序编址即可，mha_sink 先例——flash_attn_bhsd 的 annotate_address 为旧式写法、非必需） |
| 计算方式 | Cube：Form A `T.mma`（=npu_gemm，显式 L0A/L0B/L0C）；Form B `T.gemm_v0`（内部 L0 管理 + kL0Size 分片）。Vector：`T.tile.*` + `T.reduce_max/sum`（与 rev2 同族 API，落 half_M 行 buffer） |
| 作用域 | 手动 `with T.Scope("C"):` / `with T.Scope("V"):`（warpgroup.py:74） |
| 同步方式 | 全手动：C 侧 `T.set_flag/T.wait_flag`（有向 pipe pair + eventId；Form B 用 `T.barrier_all` 粗粒度替代）；跨核 `T.set_cross_flag(pipe, sem)` / `T.wait_cross_flag(sem)`（mode=2 默认 = AIC↔同组 AIV；wait 的 pipe 参数仅 A5 支持、910B 必须留空）；V 侧 MTE2↔V、MTE3↔V flag 管程；**init/destroy 闭环**（§7.2） |
| Kernel 启动 | `T.Kernel(core_num, is_npu=True) as (cid, vid)`——**禁止带 threads 参数**（带 threads → dev_mode frame → 与全 False pass_configs 构成非法混合态且落入 R8 语境；§10.2） |
| CV 数据中转 | 显式 GM workspace ×3（`workspace_idx=[4,5,6]`，`[core_num, NUM_STAGES, ...]` 索引）；**禁止虚拟通道裸拷贝** T.copy(L0C→UB)/T.copy(UB→L1)（§8.2/§8.6-R12） |
| pass_configs | 四项全 False（programming-model-guide §2.2 "Expert 极致性能"行；mha_sink:38-43 逐字同形） |

---

## 3. API 映射设计

### 3.1 公式拆解（继承 rev2 §3.1 语义，执行位置按 C/V scope 标注）

| 步骤 | 数学表达 | 说明（执行核） |
|------|----------|------|
| 1 | `acc_s = Q_tile · K_tileᵀ` | GEMM1，fp32 累加（L0C），逐 KV 块，批量写 w1（Cube） |
| 2 | `acc_s = r16(r16(acc_s) · s)` | 双舍入 golden 栅格对齐（Vector，w1 fp32 主线）；r16 = fp32→dtype→fp32 round-trip |
| 3 | `acc_s[i,j] = -inf if j > min(i+offset, S_kv-1)` | causal（offset=S_kv-S 右下角对齐）；非 causal 退化为 `j > S_kv-1` 尾列钳位（Vector） |
| 4 | `m_new_i = max(m_prev_i, rowmax(acc_s_i))` | online softmax 行最大值，phase 1（Vector） |
| 5 | `r_factors[i] = m_prev_i - m_new_i`；phase 2 `r_i = exp(r_factors[i])` | deferred rescale：exp 延迟到 phase 2，与逐迭代 exp 同值（exp 确定性；§9.3.3-6）（Vector） |
| 6 | `P_i = exp(acc_s_i - m_new_i)` | 未归一化权重 ∈[0,1]，舍入到 dtype 写 w2（Vector） |
| 7 | `sumexp_is[i] = rowsum(P_i)` | 行指数和，phase 2 折叠 `sumexp = sumexp*r_i + sumexp_is[i]`（Vector） |
| 8 | `PV_i = P_i(dtype) · V_tile_i` | GEMM2（w2→p_l1→L0A），fp32 输出写 w3（Cube） |
| 9 | `acc_o = acc_o*r_i + PV_i` | phase 2 O 累加（与 rev2 步骤 9 算式/次序逐位一致）（Vector） |
| 10 | `y = acc_o / (sumexp + 1e-30)` | 终归一化 + 全掩码行→0 保护；单块特化改在 phase 1 预归一（§1.4-8）（Vector） |

**掩码语义与 golden 的等价性**（继承 rev2，索引映射按 vid 半区细化）：行全局索引
`i_g = bx*block_M + vid*half_M + i_local`、列全局索引 `j_g = idx*block_N + j`，
掩码条件 `j_g > i_g + offset` 逐位一致；row_thresh 按 vid 半区生成
（`arith_progression(row_thresh, s0 + vid*half_M + offset_eff, 1, half_M)` +
`min(·, S_kv-1)` 钳位）。尾列（j_g ≥ S_kv，S_kv 非 block_N 整除时最后一块）由
同一阈值式钳位置 -inf，保证不进 softmax 分母——与 K/V 尾块零填充（§5.4）配合，
语义封闭。

### 3.2 TileLang API 映射（Expert）

| 步骤 | 数学表达 | TileLang API | 参数/说明 | 形态 |
|------|----------|-------------|----------|------|
| 任务网格 | Fixed Core | `T.Kernel(core_num, is_npu=True) as (cid, vid)` | core_num = min(block_num, `_cube_core_num()`)（host 设备属性查询，继承 #5；best-practices §NUM_CORES + lightning_indexer.py:14-21 先例）；**无 threads 参数** | Expert |
| 任务循环 | 均衡静态分配 | `for t in T.serial(my_count)` + `my_start/my_count`（T.if_then_else 余数分配） | mha_sink:236-238 同式；bx/by/bz 由 task 整除分解（bx 最快维，继承） | Expert |
| KV 裁剪 | eff_kv_iters | `T.floordiv(T.min(s0+bM-1+causal_offset, S_kv-1), bN)+1 if is_causal else kv_iters` | 运行期 TIR；**C/V 双 scope 同式复制**（flag 配对平衡，mha_sink:394-400）；`eff_num_outer = T.ceildiv(eff_kv_iters, NUM_STAGES)`；`batch_iters = T.if_then_else(rem < NUM_STAGES, rem, NUM_STAGES)` | Expert |
| Q/K/V 搬入 | BSHD 行跨步 | `T.copy(Q[bz, s0:s0+valid_m, by, :], q_l1)` 等（C-scope，MTE2） | 运行期 valid extent + need_clear 尾区清零（common.h:84-92，继承 rev2 验证）；SIG_Q/K/V_L1 flag 管程 | Expert |
| L1 布局 | ZN/NZ | `T.annotate_layout({q_l1: make_zn_layout(q_l1), k_l1: make_nz_layout(k_l1), p_l1: make_zn_layout(p_l1), v_l1: make_zn_layout(v_l1)})` | 仅 Form A（mma 需要 NZ 适配 transpose 载入）；Form B 不标注（gemm_v0 内部处理，flash_attn_bhsd 先例）；`from tilelang.intrinsics import make_zn_layout, make_nz_layout`（mha_sink:5/196-203） | Expert |
| GEMM1（Form A） | Q·Kᵀ | `T.copy(q_l1, l0a_s[side,:,:])`（批首 2 槽预载）+ `T.copy(k_l1, l0b_s[side,:,:], transpose=True)` + `T.mma(l0a_s[side,:,:], l0b_s[side,:,:], l0c_s[side,:,:], init=True)` | npu_copy_v2 transpose 参数（copy_op.py:262）；npu_gemm init（customize.py:115）；SIG_L0AB/L0C flag 对（mha_sink:294-309 同构） | Expert |
| GEMM1（Form B） | Q·Kᵀ | `T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True, kL0Size=gemm1_kL0)` | kL0Size 三档继承（32/64/128，§5.2）；前后 barrier_all（flash_attn_bhsd:91-94 同构） | Expert |
| L0C→w1 | scores 下放 | Form A：`T.copy(l0c_s[side,:,:], workspace_1[cid, i, :, :])`（FIX 管 + SIG_L0C 归还）；Form B：`T.copy(acc_s_l0c, workspace_1[cid, i, :, :])` | copy_l0c_to_gm fp32→fp32（common.h:187-213，T1=T2=float 无 cast = Developer 路径同语义）；每 CROSS_INTERVAL 块 `T.set_cross_flag("FIX", SEM_WS1_C2V)` | Expert |
| w1→UB | V 侧读入 | `T.copy(workspace_1[cid, i, vid*half_M : vid*half_M+half_M, :], io_buf[io_side,:,:])` + `T.copy(io_buf[io_side,:,:], work_ub)` | ws 为 4D 紧凑张量，vid 半区切片 = 行连续段（无跨步）；SIG_IO_UB 双槽管程 + `if i % CROSS_INTERVAL == 0: T.wait_cross_flag(SEM_WS1_C2V)`（mha_sink:446-457 同构） | Expert |
| 双舍入 | r16(qk)、r16(scores) | `T.copy(work_ub, acc_s_half)` + `T.copy(acc_s_half, work_ub)` → `T.tile.mul(work_ub, work_ub, sm_scale)` → 同 round-trip ×1 | copy_ub_to_ub 跨 dtype Cast（common.h，继承 rev2 验证）；链序与 Developer 终态逐字同构（mha.py:304-308） | Expert |
| 掩码链 | -inf 2D 掩码 | `T.tile.arith_progression(col_pos, j0, 1, bN)` + `T.tile.broadcast(buf_2d, col_pos, axis=0)` + `T.tile.broadcast(rt_2d, row_thresh, axis=1)` + `T.tile.compare(mask_2d, buf_2d, rt_2d, "LE")` + `T.tile.select(work_ub, mask_2d, work_ub, -T.infinity("float"), "VSEL_TENSOR_SCALAR_MODE")` | row_thresh 任务级 hoist（起点含 vid*half_M）；全可见块运行期 if 跳过：`if idx*bN+bN-1 > T.min(s0+offset_eff, S_kv-1):`（阈值按 s0 整块判定，行阈值随 r 单调不减 → 两 vid 同值同分支）；纯 V op 无 flag 介入；签名 ascend_tile.py:389/734/1888/2353 | Expert |
| 行最大值 | rowmax | `T.reduce_max(work_ub, m_i, dim=-1)` + `T.tile.max(m_i, m_i, m_i_prev)` | m_i/m_i_prev [half_M]；`T.copy(m_i, m_i_prev)` 必须在 reduce 前（rev2 §10.2 时序教训继承） | Expert |
| rescale 因子 | m_prev−m_new | `T.tile.sub(r_factors[i,:,:], m_i_prev, m_i)` | deferred：phase 1 存差值，phase 2 `T.tile.exp(r_factors[i,:,:], r_factors[i,:,:])` | Expert |
| 减最大值+exp | P | `T.tile.broadcast(buf_2d, m_i, axis=1)` + `T.tile.sub(work_ub, work_ub, buf_2d)` + `T.tile.exp(work_ub, work_ub)` | broadcast 2D 形式继承（#2）；buf_2d 专用 scratch（§8.6-2 别名禁令） | Expert |
| 行和 | rowsum | `T.reduce_sum(work_ub, sumexp_is[i,:,:], dim=-1)` | 批槽存储（phase 2 折叠） | Expert |
| 单块特化 | P̂ 预归一 | `T.tile.add(sumexp, sumexp_is[0,:,:], 1e-30)` → `T.tile.broadcast(buf_2d, sumexp, axis=1)` → `T.tile.div(work_ub, work_ub, buf_2d)` | 编译期谓词 single_kv_block；phase 2 跳过 rescale/sumexp 折叠/终除（§8.4） | Expert |
| P 写出 | →dtype→w2 | `T.wait_flag("MTE3","V",SIG_S_HALF)` → `T.copy(work_ub, acc_s_half)` → `T.set_flag("V","MTE3",SIG_S_HALF)` → `T.wait_flag("V","MTE3",SIG_S_HALF)` → `T.copy(acc_s_half, workspace_2[cid, i, vid*half_M:vid*half_M+half_M, :])` → `T.set_flag("MTE3","V",SIG_S_HALF)` → `if (i+1)%CROSS_INTERVAL==0 or i==batch_iters-1: T.set_cross_flag("MTE3", SEM_WS2_V2C)` | MTE3 ownership 闭环（mha_sink:517-528 逐字同构） | Expert |
| w2→p_l1 | GEMM2 A 上载 | `T.copy(workspace_2[cid, i, :, :], p_l1)`（C 侧全 bM 行）+ SIG_P_L1 管程 + `if i % CROSS_INTERVAL == 0: T.wait_cross_flag(SEM_WS2_V2C)` | mha_sink:330-334 同构 | Expert |
| GEMM2（Form A） | P·V | `T.copy(v_l1, l0b_o[side,:,:])`（无 transpose）+ `T.copy(p_l1, l0a_o[side,:,:])` + `T.mma(l0a_o[side], l0b_o[side], l0c_o[side], init=True)` + `T.copy(l0c_o[side,:,:], workspace_3[cid, i, :, :])` | mha_sink:320-360 同构；P2（D=bN=128）与 GEMM1 共池互换（同形） | Expert |
| GEMM2（Form B） | P·V | `T.gemm_v0(p_l1, v_l1, acc_o_l0c, init=True)` + `T.copy(acc_o_l0c, workspace_3[cid, i, :, :])` | flash_attn_bhsd:103-110 同构 | Expert |
| phase 2 累加 | acc_o*r+PV | `T.tile.exp(r_factors[i])` → `T.tile.mul(sumexp, sumexp, r_factors[i])` → `T.tile.add(sumexp, sumexp, sumexp_is[i])` → `T.tile.broadcast(buf_2d, r_factors[i], axis=1)` → `T.tile.mul(acc_o, acc_o, buf_2d)` → w3 读入（io_buf→work_ub，SIG_IO_UB + SEM_WS3_C2V）→ `T.tile.add(acc_o, acc_o, work_ub)` | mha_sink:537-560 同构 + rev2 算式次序 | Expert |
| 终归一 | ÷(sumexp+1e-30) | `T.tile.add(sumexp, sumexp, 1e-30)` → `T.tile.broadcast(buf_2d, sumexp, axis=1)` → `T.tile.div(acc_o, acc_o, buf_2d)` | 通用路径（非单块特化）；保护语义继承 | Expert |
| 写回 | rows_vid 行 | `T.copy(acc_o, out_half)` → `if rows_vid > 0:` `T.copy(out_half[0:rows_vid, :], Output[bz, s0+vid*half_M : s0+vid*half_M+rows_vid, by, :])` | SIG_S_HALF 闭环内（best-practices §6 尾注：最终 store 必须进 ownership 闭环）；`rows_vid = T.min(half_M, T.max(0, valid_m - vid*half_M))` 运行期；运行期 if 守卫（§5.4） | Expert |

#### 3.2.1 Developer 版算法优化继承映射（12 项逐项落点）

| # | 优化项（Developer 终态） | rev3 Expert 落点 | 改造性质 |
|---|------------------------|------------------|---------|
| ① | causal KV 运行期裁剪（eff_kv_iters，右下角对齐语义） | C/V 双 scope 同式复制 + `eff_num_outer = ceildiv(eff_kv_iters, NUM_STAGES)` 批量化；"全掩码块贡献恒 0 可跳过"论证不变 | 直接移植 + 外层批量化（mha_sink:268-271/394-400 先例；flag 配对平衡注释逐字适用） |
| ② | 行循环 → broadcast 2D（专用 scratch m_i_2d/r_2d） | V-scope buf_2d（m_i/r/sumexp 广播 + col_pos 广播分时）+ rt_2d（掩码行阈值广播）；scratch 与跨核 copy 目的地严格分离 | 直接移植（mha_sink 原生 broadcast 形式；别名禁令升级为 §8.6-2 硬规则） |
| ③ | 掩码链 2D 化 + row_thresh hoist + bit-packed uint8 mask | V-scope：row_thresh [half_M] 任务级 hoist（起点 + vid*half_M）+ col_pos/2×broadcast/1×[half_M,bN] compare/1×2D select | 直接移植（half_M 行版）；**Expert V-scope 首用 → R15 探针前置** |
| ④ | 全可见迭代掩码运行期跳过 | V-scope 运行期标量 if（条件按 s0 整块判定 → 两 vid 同值同分支，无 flag 配对问题；纯 V 段） | 直接移植 |
| ⑤ | 单块预归一化特化（谓词 kv_iters==1 即 S_kv≤bN） | phase 1 内预归一（§3.2"单块特化"行）；phase 2 跳过 rescale/终除；batch_iters=1 天然兼容批量结构 | 按批量结构改造（算式与 Developer 特化同序：+1e-30 → broadcast → div → cast） |
| ⑥ | sumexp+1e-30 全掩码行保护 | 终归一/单块特化两处逐字保留 | 直接移植 |
| ⑦ | score 双舍入栅格对齐（4×cast 链） | V-scope w1(fp32)→work_ub 后显式 round-trip ×2（链序同 mha.py:304-308）；w1=fp32 主线保证输入栅格与 Developer 逐位同 | 直接移植（w1 dtype 瘦身 = Stage 3 探针门候选 §10.3-①） |
| ⑧ | decode 窄块 bM=16（S≤16） | 继承；vid 二分 → half_M=8/AIV（vector 工作再减半）；**8 行 tile 为新形态 → R10 探针覆盖** | 直接移植 + vid 半区化 |
| ⑨ | bN 三档自适应（decode 512/256/128；prefill 128）+ gemm1_kL0 三档（32/64/128） | Form B（decode 全 D + prefill D=256）**逐字继承**（gemm_v0 kL0Size 参数原样）；Form A（prefill D≤128 bN=128）K=D≤128 单 mma 调用无需 K-split（§5.3 L0 表） | Form B 直接移植；Form A 由 L0 预算表覆盖 |
| ⑩ | Fixed Core（core_num 查询 + 均衡任务循环 + core 索引 workspace） | `T.Kernel(core_num)` 网格 + my_start/my_count + workspace 首维 cid（升级为 [cid, stage] 双维索引） | 直接移植（mha_sink:236-238 同式） |
| ⑪ | valid_m/valid_n 运行期尾块（三重保障） | C 侧 Q/K/V 载入运行期 extent + need_clear 清零（模板层不变）；V 侧写回 **vid 分行新规则** rows_vid + 运行期 if 守卫（§5.4）；尾列钳位掩码不变 | C 侧直接移植；V 侧按 vid 二分改造（新设计项） |
| ⑫ | BSHD 行跨步 DMA（原生进出，零 host 重排） | C 侧 Q/K/V 载入 + V 侧 Output 写回同式 4D 切片（行跨步 + head 维）；**× vid 偏移组合为仓内首例 → R10 探针** | 直接移植 + 探针门 |

### 3.3 计算伪代码

```python
pass_configs = {                                   # Expert 全手动（mha_sink:38-43 逐字同形）
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}
NUM_STAGES = 8            # 流水线深度（偶数约束；参考实现验证默认；Stage 3 可调 4-16）
CROSS_INTERVAL = 1        # 跨核同步间隔（gqa iter3 验证值；Stage 3 可调 2-4）

def _cube_core_num() -> int:                        # host 设备属性查询（继承 Developer 终态）
    try:
        import torch_npu
        return int(torch_npu.npu.get_device_properties(0).cube_core_num)
    except Exception:
        return 20                                   # 910B 级保守回退

@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)
def mha_kernel(B, S, S_kv, N, D, dtype="float16", scaleValue=-1.0, is_causal=False):
    accum_dtype = "float"
    sm_scale = scaleValue if scaleValue > 0 else (1.0 / D) ** 0.5       # host 纯 Python
    # —— tile 配置（逐字继承 Developer 终态 mha.py:120-179 三档体系）——
    block_M = 16 if S <= 16 else (32 if D == 256 else 64)
    if S <= 16:
        block_N = 512 if (D <= 128 and S_kv > 256) else (256 if S_kv > 128 else 128)
    else:
        block_N = 128
    gemm1_kL0 = (32 if (S <= 16 and D <= 128 and S_kv > 256)
                 else (64 if (S <= 16 and D == 256 and S_kv > 128) else 128))  # 仅 Form B
    # —— Expert 新增维度 ——
    form_a = (S > 16) and (D <= 128)                # Form A: mma 全流水；Form B: gemm_v0 批量
    half_M = block_M // 2                           # vid 半区行数（结构性 cv_1_2：8/16/32）
    causal_offset   = S_kv - S
    need_bound_mask = S_kv % block_N != 0
    need_mask       = is_causal or need_bound_mask
    offset_eff      = causal_offset if is_causal else 2**30
    s_blocks  = (S + block_M - 1) // block_M
    kv_iters  = (S_kv + block_N - 1) // block_N     # 编译期常量
    single_kv_block = kv_iters == 1                 # 即 S_kv <= block_N（继承谓词移动）
    block_num = B * N * s_blocks
    core_num  = min(block_num, _cube_core_num())    # Fixed Core（继承 #5）
    q_tasks, r_tasks = block_num // core_num, block_num % core_num

    @T.prim_func
    def main(
        Q: T.Tensor([B, S, N, D], dtype),
        K: T.Tensor([B, S_kv, N, D], dtype),
        V: T.Tensor([B, S_kv, N, D], dtype),
        Output: T.Tensor([B, S, N, D], dtype),
        workspace_1: T.Tensor([core_num, NUM_STAGES, block_M, block_N], accum_dtype),  # scores fp32
        workspace_2: T.Tensor([core_num, NUM_STAGES, block_M, block_N], dtype),        # P dtype
        workspace_3: T.Tensor([core_num, NUM_STAGES, block_M, D], accum_dtype),        # PV fp32
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):     # 无 threads 参数（§2.2-b）
            # ---- C 侧分配 ----
            q_l1 = T.alloc_L1([block_M, D], dtype)
            k_l1 = T.alloc_L1([block_N, D], dtype)
            v_l1 = T.alloc_L1([block_N, D], dtype)
            p_l1 = T.alloc_L1([block_M, block_N], dtype)
            if form_a:                    # Python 级双 prim_func 构建分支（§6.1 注）
                T.annotate_layout({q_l1: make_zn_layout(q_l1), k_l1: make_nz_layout(k_l1),
                                   p_l1: make_zn_layout(p_l1), v_l1: make_zn_layout(v_l1)})
                # L0 池（per-config 尺寸见 §5.3；P2 = D=bN=128 共池互换，P1 = 分池）
                l0a_s = T.alloc_L0A([2, block_M, D], dtype)          # GEMM1 A
                l0b_s = T.alloc_L0B([2, D, block_N], dtype)          # GEMM1 B=Kᵀ（P2 顶格 64KB⚠）
                l0c_s = T.alloc_L0C([2, block_M, block_N], accum_dtype)
                if D != block_N:                                     # P1（D=64）：GEMM2 独立池
                    l0a_o = T.alloc_L0A([2, block_M, block_N], dtype)
                    l0b_o = T.alloc_L0B([2, block_N, D], dtype)
                    l0c_o = T.alloc_L0C([2, block_M, D], accum_dtype)
                # P2（D==bN==128）：GEMM2 复用 l0a_s/l0b_s/l0c_s（同形互换，mha_sink 同构）
            else:                         # Form B：gemm_v0 内部 L0A/L0B + 显式 L0C
                acc_s_l0c = T.alloc_L0C([block_M, block_N], accum_dtype)
                acc_o_l0c = T.alloc_L0C([block_M, D], accum_dtype)
            # ---- V 侧分配（两形态共享，vid 半区尺寸）----
            io_buf      = T.alloc_ub([2, half_M, block_N], accum_dtype)  # w1/w3 双槽 staging
            work_ub     = T.alloc_ub([half_M, block_N], accum_dtype)     # 主计算
            buf_2d      = T.alloc_ub([half_M, block_N], accum_dtype)     # broadcast scratch
            rt_2d       = T.alloc_ub([half_M, block_N], accum_dtype)     # 掩码行阈值广播
            mask_2d     = T.alloc_ub([half_M * block_N // 8], "uint8")   # bit-packed compare 输出
            col_pos     = T.alloc_ub([block_N], accum_dtype)
            row_thresh  = T.alloc_ub([half_M], accum_dtype)
            acc_s_half  = T.alloc_ub([half_M, block_N], dtype)           # P→w2 staging
            acc_o       = T.alloc_ub([half_M, D], accum_dtype)           # O 累加器
            out_half    = T.alloc_ub([half_M, D], dtype)                 # 写回 staging（v1 独立分配）
            m_i         = T.alloc_ub([half_M], accum_dtype)
            m_i_prev    = T.alloc_ub([half_M], accum_dtype)
            sumexp      = T.alloc_ub([half_M, 1], accum_dtype)
            r_factors   = T.alloc_ub([NUM_STAGES, half_M, 1], accum_dtype)
            sumexp_is   = T.alloc_ub([NUM_STAGES, half_M, 1], accum_dtype)

            my_start = cid * q_tasks + T.if_then_else(cid < r_tasks, cid, r_tasks)
            my_count = q_tasks + T.if_then_else(cid < r_tasks, 1, 0)

            # ==================== Cube 核 ====================
            with T.Scope("C"):
                # init（模拟消费者已释放，mha_sink:241-250 同构）:
                #   set_cross_flag("MTE2", SEM_WS2_C2V)
                #   set_flag("MTE1","MTE2", SIG_K_L1/SIG_P_L1/SIG_V_L1/SIG_Q_L1)
                #   [Form A] set_flag("M","MTE1", SIG_L0AB+0/+1); set_flag("FIX","M", SIG_L0C+0/+1)
                for t in T.serial(my_count):
                    task = my_start + t
                    bx = task % s_blocks; by = task // s_blocks % N; bz = task // s_blocks // N % B
                    s0 = bx * block_M;  valid_m = T.min(block_M, S - s0)
                    eff_kv_iters = <§3.2 裁剪式>;  eff_num_outer = T.ceildiv(eff_kv_iters, NUM_STAGES)
                    # Q 载入（任务级，SIG_Q_L1 双向 ownership）
                    T.wait_flag("MTE1","MTE2", SIG_Q_L1)
                    T.copy(Q[bz, s0:s0+valid_m, by, :], q_l1)
                    T.set_flag("MTE2","MTE1", SIG_Q_L1);  T.wait_flag("MTE2","MTE1", SIG_Q_L1)
                    for k in T.serial(eff_num_outer):
                        batch_iters = T.if_then_else(eff_kv_iters-k*NUM_STAGES < NUM_STAGES,
                                                     eff_kv_iters-k*NUM_STAGES, NUM_STAGES)
                        # --- GEMM1 批：产出 scores → w1 ---
                        T.wait_cross_flag(SEM_WS1_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2;  idx = k * NUM_STAGES + i
                            j0 = idx * block_N;  valid_n = T.min(block_N, S_kv - j0)
                            # K: GM→L1（SIG_K_L1 管程）
                            T.wait_flag("MTE1","MTE2", SIG_K_L1)
                            T.copy(K[bz, j0:j0+valid_n, by, :], k_l1)
                            T.set_flag("MTE2","MTE1", SIG_K_L1)
                            if form_a:
                                # Q: L1→L0A（批首 2 槽）；K: L1→L0B 转置；mma；L0C→w1
                                T.wait_flag("M","MTE1", SIG_L0AB+side)
                                if i < 2: T.copy(q_l1, l0a_s[side,:,:])
                                T.wait_flag("MTE2","MTE1", SIG_K_L1)
                                T.copy(k_l1, l0b_s[side,:,:], transpose=True)
                                T.set_flag("MTE1","MTE2", SIG_K_L1); T.set_flag("MTE1","M", SIG_L0AB+side)
                                T.wait_flag("MTE1","M", SIG_L0AB+side); T.wait_flag("FIX","M", SIG_L0C+side)
                                T.mma(l0a_s[side,:,:], l0b_s[side,:,:], l0c_s[side,:,:], init=True)
                                T.set_flag("M","MTE1", SIG_L0AB+side); T.set_flag("M","FIX", SIG_L0C+side)
                                T.wait_flag("M","FIX", SIG_L0C+side)
                                T.copy(l0c_s[side,:,:], workspace_1[cid, i, :, :])
                                T.set_flag("FIX","M", SIG_L0C+side)
                            else:  # Form B
                                T.barrier_all()
                                T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True,
                                          init=True, kL0Size=gemm1_kL0)
                                T.barrier_all()
                                T.copy(acc_s_l0c, workspace_1[cid, i, :, :])
                                T.barrier_all()
                            if (i+1) % CROSS_INTERVAL == 0 or i == batch_iters-1:
                                T.set_cross_flag("FIX", SEM_WS1_C2V)
                        # --- GEMM2 批：消费 w2(P)，产出 PV → w3 ---
                        T.wait_cross_flag(SEM_WS3_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2;  idx = k * NUM_STAGES + i
                            # V: GM→L1（SIG_V_L1）；P: w2→p_l1（SIG_P_L1 + SEM_WS2_V2C 批量）
                            <SIG_V_L1 管程> T.copy(V[bz, j0:j0+valid_n, by, :], v_l1)
                            T.wait_flag("MTE1","MTE2", SIG_P_L1)
                            if i % CROSS_INTERVAL == 0: T.wait_cross_flag(SEM_WS2_V2C)
                            T.copy(workspace_2[cid, i, :, :], p_l1)
                            T.set_flag("MTE2","MTE1", SIG_P_L1)
                            if form_a:
                                <l0a_o/l0b_o（P2 共池）载入 + SIG 管程>
                                T.mma(l0a_o[side,:,:], l0b_o[side,:,:], l0c_o[side,:,:], init=True)
                                <SIG_L0C 管程> T.copy(l0c_o[side,:,:], workspace_3[cid, i, :, :])
                            else:
                                T.barrier_all()
                                T.gemm_v0(p_l1, v_l1, acc_o_l0c, init=True)
                                T.barrier_all()
                                T.copy(acc_o_l0c, workspace_3[cid, i, :, :])
                                T.barrier_all()
                            if (i+1) % CROSS_INTERVAL == 0 or i == batch_iters-1:
                                T.set_cross_flag("FIX", SEM_WS3_C2V)
                        T.set_cross_flag("MTE2", SEM_WS2_C2V)
                    T.set_flag("MTE1","MTE2", SIG_Q_L1)      # 任务末归还 q_l1
                # destroy: 与 init 对称的 wait_flag 闭环（mha_sink:367-375 同构）

            # ==================== Vector 核（两形态共享；vid 半区）====================
            with T.Scope("V"):
                # init（mha_sink:379-386 同构）:
                #   set_cross_flag("MTE2", SEM_WS1_V2C); set_cross_flag("MTE2", SEM_WS3_V2C)
                #   set_flag("V","MTE2", SIG_IO_UB+0/+1); set_flag("MTE3","V", SIG_S_HALF)
                for t in T.serial(my_count):
                    task/bx/by/bz/s0/valid_m 与 C 侧同式
                    eff_kv_iters/eff_num_outer 与 C 侧同式        # flag 配对平衡硬约束
                    rows_vid = T.min(half_M, T.max(0, valid_m - vid*half_M))
                    T.tile.fill(acc_o, 0.0); T.tile.fill(sumexp, 0.0); T.tile.fill(m_i, -(2**30))
                    if need_mask:                                  # 编译期变体开关（语句级 if）
                        T.tile.arith_progression(row_thresh, s0 + vid*half_M + offset_eff, 1, half_M)
                        T.tile.min(row_thresh, row_thresh, T.cast(S_kv - 1, accum_dtype))
                    for k in T.serial(eff_num_outer):
                        batch_iters = <同 C 侧同式>
                        # --- phase 1：softmax 批 ---
                        T.wait_cross_flag(SEM_WS2_C2V)
                        for i in T.serial(batch_iters):
                            io_side = i % 2;  idx = k*NUM_STAGES + i
                            # w1 → io_buf → work_ub（SIG_IO_UB 双槽 + SEM_WS1_C2V 批量）
                            T.wait_flag("V","MTE2", SIG_IO_UB+io_side)
                            if i % CROSS_INTERVAL == 0: T.wait_cross_flag(SEM_WS1_C2V)
                            T.copy(workspace_1[cid, i, vid*half_M:vid*half_M+half_M, :],
                                   io_buf[io_side,:,:])
                            T.set_flag("MTE2","V", SIG_IO_UB+io_side)
                            T.wait_flag("MTE2","V", SIG_IO_UB+io_side)
                            T.copy(io_buf[io_side,:,:], work_ub)
                            T.set_flag("V","MTE2", SIG_IO_UB+io_side)
                            # 双舍入 + 掩码链 + online softmax（§3.2 表逐行）
                            T.copy(m_i, m_i_prev)
                            <r16(qk) round-trip>; T.tile.mul(work_ub, work_ub, sm_scale); <r16(scores)>
                            if need_mask:
                                if idx*block_N+block_N-1 > T.min(s0+offset_eff, S_kv-1):  # 运行期跳过
                                    <col_pos → buf_2d/rt_2d broadcast → compare(mask_2d,"LE")
                                     → select(work_ub, mask_2d, work_ub, -inf, TENSOR_SCALAR)>
                            T.reduce_max(work_ub, m_i, dim=-1); T.tile.max(m_i, m_i, m_i_prev)
                            T.tile.sub(r_factors[i,:,:], m_i_prev, m_i)
                            T.tile.broadcast(buf_2d, m_i, axis=1); T.tile.sub(work_ub, work_ub, buf_2d)
                            T.tile.exp(work_ub, work_ub)
                            T.reduce_sum(work_ub, sumexp_is[i,:,:], dim=-1)
                            if single_kv_block:                    # 预归一化特化（§1.4-8）
                                T.tile.add(sumexp, sumexp_is[0,:,:], 1e-30)
                                T.tile.broadcast(buf_2d, sumexp, axis=1)
                                T.tile.div(work_ub, work_ub, buf_2d)
                            # P → w2（SIG_S_HALF 闭环 + SEM_WS2_V2C 批量）
                            T.wait_flag("MTE3","V", SIG_S_HALF)
                            T.copy(work_ub, acc_s_half)
                            T.set_flag("V","MTE3", SIG_S_HALF)
                            T.wait_flag("V","MTE3", SIG_S_HALF)
                            T.copy(acc_s_half, workspace_2[cid, i, vid*half_M:vid*half_M+half_M, :])
                            T.set_flag("MTE3","V", SIG_S_HALF)
                            if (i+1)%CROSS_INTERVAL==0 or i==batch_iters-1:
                                T.set_cross_flag("MTE3", SEM_WS2_V2C)
                        T.set_cross_flag("MTE2", SEM_WS1_V2C)
                        # --- phase 2：O 累加批 ---
                        for i in T.serial(batch_iters):
                            if not single_kv_block:
                                T.tile.exp(r_factors[i,:,:], r_factors[i,:,:])
                                T.tile.mul(sumexp, sumexp, r_factors[i,:,:])
                                T.tile.add(sumexp, sumexp, sumexp_is[i,:,:])
                                T.tile.broadcast(buf_2d, r_factors[i,:,:], axis=1)
                                T.tile.mul(acc_o, acc_o, buf_2d)
                            <w3 → io_buf[i%2] → work_ub：SIG_IO_UB + SEM_WS3_C2V 批量>
                            T.tile.add(acc_o, acc_o, work_ub)
                        T.set_cross_flag("MTE2", SEM_WS3_V2C)
                    if not single_kv_block:                        # 终归一化（全掩码行保护）
                        T.tile.add(sumexp, sumexp, 1e-30)
                        T.tile.broadcast(buf_2d, sumexp, axis=1)
                        T.tile.div(acc_o, acc_o, buf_2d)
                    # 写回（SIG_S_HALF 闭环内 + rows_vid 运行期守卫）
                    T.wait_flag("MTE3","V", SIG_S_HALF)
                    T.copy(acc_o, out_half)
                    T.set_flag("V","MTE3", SIG_S_HALF)
                    T.wait_flag("V","MTE3", SIG_S_HALF)
                    if rows_vid > 0:
                        T.copy(out_half[0:rows_vid, :],
                               Output[bz, s0+vid*half_M : s0+vid*half_M+rows_vid, by, :])
                    T.set_flag("MTE3","V", SIG_S_HALF)
                # destroy: 与 init 对称的 wait 闭环（mha_sink:625-629 同构）

    return main
    # form_a 实现注记：为避免 TVMScript parse-time if 的跨块变量作用域限制
    # （rev2-A1 教训），Form A/B 在工厂函数内以 Python 级分支构建两个自包含
    # prim_func（V-scope 由共享 emitter 函数生成，C-scope 各自完整），host 按
    # form_a 谓词分派——两形态独立编译、独立验证（§5.6 DISPATCH-COVERAGE）。
```

Host 包装函数（**逐字继承 rev2 §3.3**，`mha.py` 对外接口，零张量操作）：

```python
def mha(query, key, value, scaleValue=-1.0, is_causal=False):
    # —— 仅元数据校验（不触碰张量数据，§4.8）——
    assert query.dtype == key.dtype == value.dtype and query.dtype in (torch.float16, torch.bfloat16)
    assert query.is_contiguous() and key.is_contiguous() and value.is_contiguous()  # BSHD contiguous 契约
    B, S, N, D = query.shape
    S_kv = key.shape[1]
    assert key.shape == (B, S_kv, N, D) and value.shape == key.shape   # B/N/D 严格相等
    assert D in (64, 128, 256)
    assert 1 <= B <= 128 and 1 <= S <= 2048 and 1 <= S_kv <= 4096 and 1 <= N <= 64
    if is_causal:
        assert S <= S_kv, "is_causal=True requires S <= S_kv (否则出现全掩码行)"
    kernel = mha_kernel(B=B, S=S, S_kv=S_kv, N=N, D=D,
                        dtype=str(query.dtype).removeprefix("torch."),
                        scaleValue=float(scaleValue), is_causal=bool(is_causal))
    return kernel(query, key, value)   # out_idx=[3] 分配输出；workspace_idx=[4,5,6] 框架注入
```

### 3.4 API 可行性确认（不凭记忆猜 API——逐项源码/示例实证）

| API | 来源确认 | 验证状态 |
|-----|---------|---------|
| `T.Kernel(n, is_npu=True) as (cid, vid)`（Expert 帧，无 threads） | kernel.py:215-263（is_npu ∧ threads=None → is_npu_kernel_frame）；ir.cc:281-290（cid=blockIdx.x，vid=blockIdx.y extent 2）；flash_attn_bhsd.py:36 / gqa:220 / mha_sink:187 | ✅ 源码 + 三先例 |
| `T.alloc_L1/L0A/L0B/L0C/ub(shape, dtype)` | allocate.py:140-158（scope 映射 shared.l1 / wmma.matrix_a/b/accumulator / shared.ub）；mha_sink:190-231 | ✅ 源码 + 先例 |
| `T.Scope("C"/"V")` | warpgroup.py:74（_ffi_api.Scope）；三参考实现 | ✅ 源码 + 先例 |
| `T.set_flag(src, dst, eventId)` / `T.wait_flag(src, dst, eventId)` | ascend.py:233-270（有向 pipe pair + eventId；`_pipe = Literal["fix","mte1","mte2","mte3","m","v","s"]`，ascend.py:9）；mha_sink 全套用法 | ✅ 源码 + 先例 |
| `T.set_cross_flag(pipe, flag, mode=2)` / `T.wait_cross_flag(flag, pipe="")` | ascend.py:191-231（mode=2 = AIC↔同组 AIV；**wait 的 pipe 参数仅 A5 平台支持，910B 必须留空**——参考实现均单参调用）；mha_sink SEM 0-5 | ✅ 源码 + 先例 |
| `T.barrier_all()` / `T.pipe_barrier(pipe)` | ascend.py:275-300（Form B 用；Form A 循环体内禁用 barrier_all——性能反模式，§10.2） | ✅ 源码 + 先例 |
| `T.mma(A_l0a, B_l0b, C_l0c, init=…)` = npu_gemm | language/__init__.py:75（`npu_gemm as mma` 别名）；customize.py:115-190（init/n_actual/unit_flag/k_actual 签名；K 由 A 末维推导；高阶 buffer 前导维切片须定值）；mha_sink:307/351 | ✅ 源码 + 先例 |
| `T.gemm_v0(A_l1, B_l1, C_l0c, transpose_B, init, kL0Size, n_actual)` | ascend.py:418-530（A/B 必须 L1、C 必须 L0C；kL0Size 16 倍数 ≤4095）；flash_attn_bhsd.py:93/107（Expert Scope("C") 内使用先例）；本算子 Developer 终态 kL0Size 三档实测 | ✅ 源码 + Expert 先例 + 本算子实测 |
| `T.copy(src, dst, transpose=True)`（L1→L0B 转置载入） | T.copy = npu_copy_v2（language/__init__.py:53 后导入覆盖）；copy_op.py:257-266（transpose 参数，copy_l1_to_l0 专用）；mha_sink:300 | ✅ 源码 + 先例 |
| `T.copy` L0C→GM workspace（fp32→fp32；模板含 fp32→half cast 能力） | ascend.cc:225-226（copy_l0c_to_gm 分派）；common.h:187-213（模板 `<T1,T2,…>` 泛型 + realDstN/realTailM/realTailN/unitFlag）；mha_sink:313 | ✅ 源码 + 先例 |
| `T.copy` GM(4D ws)→UB vid 半区切片 | copy_gm_to_ub（common.h:233-268，运行期 maskShapeM/N + padValue + Duplicate 清零）；mha_sink:449-452（4D ws + vid*half_M 切片逐字同构） | ✅ 源码 + 先例 |
| `T.copy` GM→L1 BSHD 4D 行跨步 + 运行期 valid extent | copy_gm_to_l1 need_clear 清零（common.h:84-92，rev2 已验证）；行跨步模型同 rev2 §5.4（列维=最内维 D）；**× Expert/vid 组合 = R10 探针项** | ✅ 模板同 rev2 已验证；组合探针前置 |
| `T.copy` UB→GM 4D 行跨步 + 运行期 rows_vid | copy_ub_to_gm（common.h:269-283，运行期 maskShapeM/N）；rev2 verify_t1a 同模板 threads=1 位级验证；**× vid 偏移组合 = R10 探针项** | ✅ 模板已验证；组合探针前置 |
| `T.annotate_layout({buf: make_zn_layout(buf)/make_nz_layout(buf)})` | `from tilelang.intrinsics import make_zn_layout, make_nz_layout`（mha_sink:5）；mha_sink:196-203 / gqa:229-237；best-practices §4（ZN 适配 matmul 输入、NZ 适配 transpose） | ✅ 先例 + 文档；**× BSHD 行跨步 GM→L1 源 = R10 探针项**（失败回退：去 annotation，性能降级不换语义） |
| `T.tile.fill/add/mul/sub/div/exp/max/min/broadcast/axpy` | ascend_tile.py（fill:343、broadcast:2353（dst, src, axis=None 自动推断）、axpy:1353）；mha_sink V-scope 全套；本算子 Developer 终态位级验证（同族 op） | ✅ 源码 + 先例 + 本算子实测 |
| `T.tile.arith_progression/compare/select`（Expert V-scope） | ascend_tile.py:389/1888/734（签名同 rev2）；operation_config.h:184-237 PIPE_V 注册（模式无关 lowering）；Developer 语境位级验证（本算子 #3）；**Expert V-scope 首用 = R15 探针项**（纯 V op、无 flag 介入，风险受控） | ✅ 源码 + 跨模式同 lowering；探针前置 |
| `T.reduce_max/reduce_sum(buf, out, dim=-1)` | mha_sink:460/531（Expert V-scope 先例，[half_M, bN] 尺寸）；本算子 Developer 位级验证 | ✅ 先例 + 本算子实测 |
| `T.serial`（运行期 TIR 边界，Expert） | mha_sink:278/284/433/439（eff_num_outer/batch_iters 运行期边界 CI 验证）；gqa 同构 | ✅ Expert 先例 |
| `T.if_then_else/T.floordiv/T.ceildiv/T.min/T.max/T.cast/-T.infinity(dt)` | mha_sink:236-237/268-271；本算子 Developer 终态全套使用 | ✅ 先例 + 本算子实测 |
| `workspace_idx=[4,5,6]` + 框架注入（host 调用形式不变） | flash_attn_bhsd.py:8（jit 参数）+ :254（三参调用七参 main 实证）；gqa:86；mha_sink:48 | ✅ 先例 |
| Expert 下无 annotate_address 的地址分配 | ascend_memory_planning.cc:59-92（pass 恒运行，auto=False 顺序编址 + address_map attr 恒生成）；codegen_ascend.cc:1245/805-840；mha_sink 全文无 annotate_address（CI 通过） | ✅ 源码 + 先例 |
| 禁用项：`T.Pipelined`（任何形态） | R9 语境 + Expert 参考实现零使用（手动 ping-pong 替代） | 🚫 不使用 |
| 禁用项：`threads` 参数（Expert kernel） | kernel.py:249-263（带 threads → dev_mode frame → 与全 False pass_configs 构成非法混合态 + R8 语境） | 🚫 不使用（§10.2） |
| 禁用项：虚拟通道裸拷贝 `T.copy(L0C buf→UB buf)` / `T.copy(UB buf→L1 buf)` | ascend.cc:289/298 virtual_channel 标记 + ascend_workspace_reduction.cc:1001-1004 重写表；Expert 无 AUTO_CV_SYNC → 重写后无跨核同步插入 → 竞态 | 🚫 不使用（§8.6-R12，自检 grep 项） |
| 禁用项：`tilelang/language/pto.py` | AGENTS.md 明令废弃 | 🚫 不使用 |
| 禁用项：三维 Kernel / GPU 专用 API / `T.gemm` 通用版 | ascend-constraints §1（继承 rev2 核对结论） | 🚫 不使用 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查（Expert 语境全项重跑）

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | 涉及（逻辑网格 B×N×S_blocks） | `T.Kernel(core_num, is_npu=True)` 一维 + Fixed Core 任务循环内 task 整除分解（继承 #5，mha_sink:236-238 同式） |
| threads 参数限制（仅 1/2） | **不涉及（Expert 无 threads 参数）** | Expert 帧 vid extent 2 为结构性绑定（ir.cc:288-290 硬编码），不经 threads 参数；**全文禁止出现 threads=**（§10.2）；R8 复检结论 = Expert 不触发（§2.2-c 五点证据） |
| 动态循环边界不支持 | 涉及（S/S_kv 运行期可变 + causal 裁剪） | B/S/S_kv/N/D JIT 编译期参数（继承 rev2 §4.6）；循环边界 = 运行期 TIR 标量表达式（eff_num_outer/batch_iters），**不依赖 tensor 值**（由编译期 shape + cid/task 标量算术派生）——Expert 先例 mha_sink:270-271/278 CI 验证（区别于约束原文 `batch_sizes[bz]` 型 tensor 值边界）；kernel 内无 tensor 值边界 |
| 流水线不支持动态边界 | 涉及 | **不使用 T.Pipelined**（R9 + Expert 形态）；流水 = NUM_STAGES 批量结构（外层 T.serial(eff_num_outer) + 内层 T.serial(batch_iters) + 手动 side=i%2 ping-pong），运行期批深由 T.if_then_else 表达（mha_sink:280 同式） |
| 部分 GPU API 不可用 | 不涉及 | 全部 API 出自本项目源码/示例（§3.4 逐项） |
| L0C 容量上限（128KB） | 涉及 | 六配置类逐一核算（§4.5/§5.3 表）：最大 P1 96KB / P2 64KB / Form B 16-48KB ✅ |
| L0A/L0B 容量（各 64KB） | 涉及 | Form A：P2 l0b [2,128,128]fp16 = 64KB **顶格⚠**（mha_sink 同尺寸 CI 先例）；P1 l0b_s+l0b_o = 32+32KB **顶格⚠**；Form B：gemm_v0 内部 kL0Size 分片（三档继承，session-2 实测先例⚠）；顶格条目实编读数验证（§4.5 制度） |
| T.copy 不支持列方向 strided 切片 | 涉及（BSHD 4D 切片） | 切片列维 = D = buffer 最内维（行内连续、行间 gap=(N-1)*D），行跨步模型合法（继承 rev2 §5.4 论证 + verify_bshd_slice 位级验证）；ws↔UB 切片为 4D 紧凑张量行连续段（无跨步）；无 `k:k+1` 型列跨步 |
| Host 侧 reshape 隐式拷贝 / aclnn 调用 / 输出侧切片+reshape | 不涉及 | host 零张量变换（§4.8 审计：metadata 读取 + 纯 Python + 设备属性查询 + kernel 调用直返）；输出 = kernel BSHD 原生写回，out_idx 分配 contiguous |
| GEMM 非整除 | 涉及 | ceildiv 分块 + 运行期 valid extent + 尾列钳位掩码三重保障（继承 rev2 §5.4 + vid 分行规则 §5.4-新）；禁止 host padding |

#### 3.5.2 参考实现差异说明

无外部 GPU 参考实现。golden（§9.1）仅定义数学语义。rev3 与 **Expert 仓内参考**
的结构差异及理由：

| 差异项 | 参考实现（mha_sink_fwd / gqa_fwd_varlen / flash_attn_bhsd） | 本设计（rev3） | 理由 |
|--------|----------------------------------------------------------|---------------|------|
| 布局 | BHSD `[B,N,S,D]` | **BSHD `[B,S,N,D]` 原生行跨步**（host 零重排契约） | 需求契约；flash_attn_optimize §9(a) BSND 直消费先例（deepseek_v4/sparse_attention.py q_shape=[b,m,h,d]）；rev2 行跨步模板位级验证继承 |
| 掩码 | host 预构建 GM mask 张量 + **post-exp 乘 0/1**（mask-after-exp） | **kernel 内 pre-exp -inf 掩码链**（compare+select，无 host mask 张量） | golden 语义逐位对齐（masked_fill(-inf) 后 softmax，rev2 §3.1 索引映射）+ host 零张量操作契约（host 造 mask = 物理化违规）+ 双舍入栅格链要求掩码作用于 r16(scores) 后、exp 前；reduce_max 与 golden 同域（含掩码位）——单块位级精确锚点的前提 |
| w1 dtype | fp16（隐式首舍入） | **fp32 主线**（显式双舍入链在 V 侧完整执行） | 精度红线继承（§9.3.3）：fp32 w1 → rev3 与 Developer 终态位级一致锚点（§9.2 诊断锚）；w1→dtype 瘦身 = Stage 3 探针门候选（§10.3-①，需 fixpipe cast RNE 验证 + 单块位级探针） |
| w3 dtype | fp16（PV 块先舍入再累加） | **fp32**（继承 Developer acc_o 全 fp32 链） | rev2 §9.3.3-3 "fp32 累加/归一化，精度不低于 golden"红线；w3 fp16 会侵蚀 MARE_abs 余量（未量化），列 Stage 3 探针门候选（§10.3-①） |
| tile 配置 | bM=bN=D=128 单配置（L0 池互换前提） | **六配置类**（bM 16/32/64 × bN 128/256/512 × kL0 三档，继承 Developer 终态） | 支持域 D∈{64,128,256} + decode/prefill 全谱；P2（D=128=bN）恰落入参考实现互换池形态 |
| 尾块 | 整除断言（mha_sink:110-111）/ host padding（gqa varlen 4D padded） | **运行期 valid extent + vid 分行 rows_vid + 运行期 if 守卫**（host 零 padding） | 需求支持域任意 S/S_kv；host 契约禁止 padding；模板层运行期 extent 已验证（rev2），组合探针 R10 |
| softmax 算式 | neg_sm（负最大值）+ axpy 融合 | **继承 Developer 终态算式**（m_i/m_i_prev/max/sub/broadcast-sub），仅 rescale exp 延迟到 phase 2 | 位级一致锚点（§9.3.3-6：deferred exp 与逐迭代 exp 同值）；axpy 融合列 Stage 3 候选 |
| NUM_CORES | 硬编码 24（mha_sink/gqa） | `_cube_core_num()` 设备属性查询（异常回退 20） | 继承 Developer 终态 #5（910B2 实际核数以设备读数为准；best-practices §NUM_CORES 同口径） |
| unit_flag（mma→fixpipe 硬件融合） | 未用（常规 set/wait FIX flag） | v1 未用；Stage 3 候选 | copy_op.py:283-291 / customize.py:115 docstring（0b11 flush 融合、跨 L0C ping-pong 重叠）为进阶优化——最小必要改动原则，v1 走参考实现验证过的常规 flag 路径 |

#### 3.5.3 本项目同类实现参考（Phase 2 强制步骤 0 检索结果）

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/mha_sink_fwd_bhsd/mha_sink_fwd_bhsd.py` | **最高（Form A 结构模板）** | Expert CV 批量流水完整骨架：PASS_CONFIGS 全 False（:38-43）、(cid,vid)（:187）、L0 双槽+T.mma（:206-208/307/351）、SEM 0-5 + SIG 命名体系（:143-173）、init/destroy flag 闭环（:241-250/367-375/379-386/625-629）、NUM_STAGES 批量 + batch_iters 运行期余数（:278-280）、cross_interval（:315/331）、causal 运行期裁剪 C/V 同式（:268-271/394-400）、deferred rescale 批量 softmax（:530-560）、ZN/NZ annotate_layout（:196-203）、io_buf 双槽 + SIG_S_HALF 闭环（:443-457/517-528）、mask 运行期跳过条件结构（:502）、vid 半区写回（:614-622）、Fixed Core 静态分配（:236-238） |
| `examples/gqa_fwd_varlen/gqa_fwd_varlen.py` | **高（Form A 源头，17.78ms/51.72 TFlops 验证）** | mha_sink 结构所自出（mha_sink:65 注）；参数调优实证（iter3：num_stages 8+8 优于 14+2、cross_interval=1；iter4：io_buf 双缓冲 +16KB UB 换 MTE2/V 全重叠，docstring :96-118）；block_N=256 的 L0B 128KB>64KB 否决先例（:112-114——与本设计 Form A bN=128 上限互证） |
| `examples/flash_attention/flash_attn_bhsd.py` | **高（Form B 结构模板）** | Expert + gemm_v0 + workspace + vid 二分基础形态（:36-114）；gemm_v0 在 T.Scope("C") 内可用性实证（:93/107）；set_cross_flag/wait_cross_flag 基础协议（:98-114/184-197）；annotate_address 旧式写法（rev3 不采用——mha_sink 先例证明非必需） |
| `examples/flash_attention/fa_opt/flash_attn_bhsd_expert_h16_d128.py` | 高（Form A 参数基准） | num_stages/cross_interval 参数化入口（:32-33）、workspace [NUM_CORES, num_stages, ...] 结构（:104-106）、L0 双槽（:123-125）；best-practices §1-§8 载体实现 |
| `examples/deepseek_v4/sparse_attention.py` | 中（BSHD 4D + vid 偏移 workspace 先例） | q_shape=[b,m,h,d] BSHD 4D 切片 + vid*v_block 半区读写 workspace（:44/59-62/94/126/145/154）；注意其 pass_configs 为全开（Developer+手动 workspace 兜底写法）——vid 偏移切片形式可参考，模式语境不同 |
| `examples/attention_sink/example_gqa_sink_bwd_bhsd/…py` | 中（BSHD + 掩码链先例） | rev2 时代先例（掩码链 arith_progression/compare/select :162-194、运行期边界 T.serial :323-324）；rev3 掩码链语义承接、模式语境已变 |
| `examples/mha/history_version/mha_impl_developer_final_6505.py` | **语义基准（非结构参考）** | 65.05 分 Developer 终态：12 项算法层优化权威载体（§3.2.1 逐项映射）；位级一致锚点对照实现（§9.2 诊断锚）；docstring :7-73 = R8/R9/AUTO_SYNC 边界/硬限知识完整证据索引 |
| `examples/developer_mode/flash_attn_bshd_developer.py` | 低（Developer 对照） | 模式对照与回退路径参考（§12 回退条款） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量（继承 rev2，不变）

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| query | (B, S, N, D) | float16（代表性；全集见 §9.3） | BSHD 布局，contiguous，有限实数 |
| key | (B, S_kv, N, D) | 同 query | B/N/D 与 query 严格相等 |
| value | (B, S_kv, N, D) | 同 query | 同上 |
| scaleValue | 标量 float（host 参数，非 tensor） | — | 默认 -1.0；≤0 → 自动 1/√D；JIT 编译期常量 |
| is_causal | 标量 bool（host 参数，非 tensor） | — | 默认 False；True 时要求 S ≤ S_kv；JIT 编译期常量 |

支持范围（host 断言强制）：B∈[1,128]，S∈[1,2048]，S_kv∈[1,4096]，N∈[1,64]，
D∈{64,128,256}（64 对齐、静态）；dtype ∈ {float16, bfloat16}（三输入一致）。

### 4.2 输出张量（继承 rev2，不变）

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| Output | (B, S, N, D) | 同 query | out_idx=[3] 由 tilelang 分配；contiguous（BSHD 原生写回，vid 分行、rows_vid 裁剪，无 host 后处理） |

### 4.3 中间缓冲区（Expert 显式声明制；记号：bM=block_M，bN=block_N，hM=bM//2，dt=输入 dtype，fp32=accum_dtype）

**L1（C 侧，T.alloc_L1）**

| Buffer | Shape | dtype | 用途 |
|--------|-------|-------|------|
| q_l1 | (bM, D) | dt | Q 条带（任务级常驻，SIG_Q_L1 双向 ownership） |
| k_l1 | (bN, D) | dt | K 块（批内逐块，SIG_K_L1；Form A NZ layout） |
| v_l1 | (bN, D) | dt | V 块（SIG_V_L1；Form A ZN layout） |
| p_l1 | (bM, bN) | dt | P（GEMM2 A，从 w2 上载，SIG_P_L1；Form A ZN layout） |

**L0（Form A，T.alloc_L0A/L0B/L0C；双槽 side=i%2）**

| Buffer | Shape | dtype | 用途 |
|--------|-------|-------|------|
| l0a_s | (2, bM, D) | dt | GEMM1 A（Q；批首 2 槽预载，mha_sink:295-296 模式） |
| l0b_s | (2, D, bN) | dt | GEMM1 B（Kᵀ，transpose=True 载入） |
| l0c_s | (2, bM, bN) | fp32 | GEMM1 输出（scores） |
| l0a_o | (2, bM, bN) | dt | GEMM2 A（P；**P2（D==bN）与 l0a_s 共池互换**，仅 P1 独立分配） |
| l0b_o | (2, bN, D) | dt | GEMM2 B（V 直载无 transpose；P2 与 l0b_s 共池） |
| l0c_o | (2, bM, D) | fp32 | GEMM2 输出（PV；P2 与 l0c_s 共池） |

**L0C（Form B，T.alloc_L0C；gemm_v0 内部管理 L0A/L0B）**

| Buffer | Shape | dtype | 用途 |
|--------|-------|-------|------|
| acc_s_l0c | (bM, bN) | fp32 | GEMM1 输出（单槽） |
| acc_o_l0c | (bM, D) | fp32 | GEMM2 输出（单槽） |

**UB（V 侧，T.alloc_ub；每 AIV 各一份，承载 vid 半区 hM 行）**

| Buffer | Shape | dtype | 用途 |
|--------|-------|-------|------|
| io_buf | (2, hM, bN) | fp32 | w1/w3→UB 双槽 staging（SIG_IO_UB 0/1 管程；phase1/phase2 分时复用，mha_sink io_buf 同构；fp32 = w1/w3 fp32 主线） |
| work_ub | (hM, bN) | fp32 | 主计算缓冲（双舍入/掩码/softmax） |
| buf_2d | (hM, bN) | fp32 | broadcast scratch（m_i/r/sumexp 广播与 col_pos 广播分时复用——同为纯 V 时序不重叠段） |
| rt_2d | (hM, bN) | fp32 | 掩码行阈值广播（row_thresh → 2D） |
| mask_2d | (hM*bN//8,) | uint8 | compare bit-packed 输出（继承 #3；[n//8] uint8 先例） |
| col_pos | (bN,) | fp32 | 列全局索引 arith_progression（fp32：≤4096 ≪ 2²⁴ 精确表示，rev2 int32-compare 教训继承） |
| row_thresh | (hM,) | fp32 | 行阈值（任务级 hoist，起点含 vid*half_M） |
| acc_s_half | (hM, bN) | dt | P 半精度化（双舍入 round-trip 中转 + w2 写 staging，SIG_S_HALF 闭环） |
| acc_o | (hM, D) | fp32 | O 累加器（跨批状态） |
| out_half | (hM, D) | dt | 输出半精度化（写回 staging；v1 独立分配保 ownership 简单，D==bN 时与 acc_s_half 合并列 Stage 3） |
| m_i / m_i_prev | (hM,) | fp32 | 行最大值状态/上一值 |
| sumexp | (hM, 1) | fp32 | 行指数和状态 |
| r_factors | (NUM_STAGES, hM, 1) | fp32 | 批内 deferred rescale 差值（phase 2 exp） |
| sumexp_is | (NUM_STAGES, hM, 1) | fp32 | 批内块行和（phase 2 折叠） |

（TVMScript 作用域教训继承 rev2-A1：全部 buffer 在 Kernel 顶层无条件分配，不在
parse-time if 块内定义跨块使用变量；need_mask=False 时掩码链 buffer 仅耗
≈bN*4+hM*4+hM*bN*4+hM*bN/8 B UB，可接受。）

**GM workspace（框架注入，workspace_idx=[4,5,6]）**

| Buffer | Shape | dtype | 用途 |
|--------|-------|-------|------|
| workspace_1 | (core_num, NUM_STAGES, bM, bN) | fp32 | GEMM1 scores（C 写 FIX / V 读 MTE2；主线 fp32 = Developer 位级锚 §9.3.3-6） |
| workspace_2 | (core_num, NUM_STAGES, bM, bN) | dt | softmax P（V 写 MTE3 / C 读 MTE2→p_l1；dtype = P 原生栅格，同 Developer acc_s_half） |
| workspace_3 | (core_num, NUM_STAGES, bM, D) | fp32 | PV 输出（C 写 FIX / V 读 MTE2；fp32 = 精度红线 §9.3.3-3） |

### 4.4 内存搬运路径（层级审计）

```
GM[Q] ──MTE2(行跨步, valid_m)──► L1[q_l1] ──MTE1──► L0A          （Form A 显式 copy / Form B gemm_v0 内部）
GM[K] ──MTE2(行跨步, valid_n, 批内逐块)──► L1[k_l1] ──MTE1(transpose=True)──► L0B
L0A+L0B ──M(Form A: T.mma 双槽 / Form B: gemm_v0)──► L0C[scores]
L0C ──FIX──► GM[workspace_1[cid,i]] ──MTE2(vid 半区)──► UB[io_buf] ──V(cast)──► UB[work_ub]
UB[work_ub]（双舍入/掩码/softmax）──V(cast)──► UB[acc_s_half] ──MTE3──► GM[workspace_2[cid,i]]
GM[workspace_2[cid,i]] ──MTE2──► L1[p_l1] ──MTE1──► L0A
GM[V] ──MTE2(行跨步, valid_n)──► L1[v_l1] ──MTE1──► L0B
L0A+L0B ──M──► L0C[PV] ──FIX──► GM[workspace_3[cid,i]] ──MTE2(vid 半区)──► UB[io_buf] ─► UB[acc_o]（rescale+累加）
UB[acc_o]（终归一）──V(cast)──► UB[out_half] ──MTE3(行跨步, rows_vid, 运行期 if 守卫)──► GM[Output]
```

- 全部路径符合硬件内存层级：GM↔L1↔L0A/L0B→L0C→(FIX)GM→(MTE2)UB→(MTE3)GM；
  CV 交互经显式 GM workspace（本框架唯一物理路径，§8.2 源码证据）；无 GM→L0
  直搬、无 UB→L1 直连声明（虚拟通道禁令 §8.6-R12）。
- BSHD 行跨步切片合法性继承 rev2 §4.4：切片 extents [1, valid, 1, D]，列维 =
  最内维 D（行内连续、行间 gap=(N-1)*D 元素）→ DataCopyPad 硬件支持模型；
  ws↔UB 切片为紧凑 4D 张量的行连续段（无跨步）。

### 4.5 内存预算（Expert 显式声明制；顶格条目强制实编读数验证）

> **判定制度（session-2 教训反向固化）**：Expert 下 MEMORY_PLANNING=False，
> planner 无 liveness 合并（ascend_memory_planning.cc:59），naive 声明和≈实配；
> 但对齐 padding（AlignUp）与 gemm_v0 内部 L0 申请可能有小额附加 → **所有
> ≥90% 占用率条目（⚠ 标记）必须在 Stage 2 以实编 GetWithOffset/编译探针读数
> 复核**，手算仅作设计初筛。硬限（实测口径，继承 perf_tuning 会话 2）：
> L1 511.75KB、UB 191.75KB（每 AIV）、L0C 128KB、L0A 64KB、L0B 64KB。

**L1（C 侧共享，单份）**

| 配置类 | q_l1 | k_l1 | v_l1 | p_l1 | 合计 / 511.75KB |
|--------|------|------|------|------|------------------|
| P1 prefill D=64（bM64,bN128,Form A） | 8KB | 16KB | 16KB | 16KB | 56KB ✅ |
| P2 prefill D=128（bM64,bN128,Form A） | 16KB | 32KB | 32KB | 16KB | 96KB ✅ |
| P3 prefill D=256（bM32,bN128,Form B） | 16KB | 64KB | 64KB | 8KB | 152KB ✅ |
| D1 decode bN=512 D≤128（bM16,Form B） | 4KB | 128KB | 128KB | 16KB | 276KB ✅（Developer 终态同值实测先例） |
| D2 decode bN=256（D≤128 / D=256） | 4/8KB | 64/128KB | 64/128KB | 8KB | 140KB / 272KB ✅ |
| D3 decode bN=128 D≤128 | 4KB | 32KB | 32KB | 4KB | 72KB ✅ |
| ~~decode bN=512 D=256~~ | — | — | — | — | 536KB ✗ **排除**（继承 session-2 硬限：>511.75KB） |

**UB（每 AIV，vid 半区 hM 行；fp32 io_buf 主线）**

| 配置类 | io_buf×2 | work_ub | buf_2d | rt_2d | acc_s_half | acc_o | out_half | mask/col/thresh | r/sumexp_is/状态 | 合计 / 191.75KB |
|--------|---------|---------|--------|-------|-----------|-------|----------|-----------------|------------------|------------------|
| P1（hM32,bN128,D64） | 32KB | 16KB | 16KB | 16KB | 8KB | 8KB | 4KB | ~1.2KB | ~2.5KB | **≈104KB ✅** |
| P2（hM32,bN128,D128） | 32KB | 16KB | 16KB | 16KB | 8KB | 16KB | 8KB | ~1.2KB | ~2.5KB | **≈116KB ✅** |
| P3（hM16,bN128,D256） | 16KB | 8KB | 8KB | 8KB | 4KB | 16KB | 8KB | ~0.8KB | ~1.5KB | **≈79KB ✅** |
| D1（hM8,bN512,D128） | 32KB | 16KB | 16KB | 16KB | 8KB | 4KB | 2KB | ~2.6KB | ~1KB | **≈98KB ✅** |
| D2（hM8,bN256,D≤128/256） | 16KB | 8KB | 8KB | 8KB | 4KB | 4/8KB | 2/4KB | ~1.3KB | ~1KB | **≈53/61KB ✅** |
| D3（hM8,bN128,D≤128） | 8KB | 4KB | 4KB | 4KB | 2KB | 4KB | 2KB | ~0.7KB | ~1KB | **≈30KB ✅** |

余量充足（最紧 P2 ≈116KB，占用 60%）——与 Developer 终态 UB 186.8KB 顶格形成
对照：**vid 二分把 per-AIV 行负载减半是 Expert 的结构性 UB 红利**（为 Stage 3
的 k/v L1 双缓冲、io_buf 加深等留出空间）。

**L0（Form A；单位 KB；⚠=顶格须实编验证）**

| 配置类 | L0A / 64KB | L0B / 64KB | L0C / 128KB | 判定 |
|--------|-----------|-----------|-------------|------|
| P2（bM64,bN128,D128；共池互换——mha_sink 同构先例） | l0a[2,64,128]fp16 = 32 | l0b[2,128,128]fp16 = 64 ⚠ | l0c[2,64,128]fp32 = 64 | ✅（l0b 顶格 = 上限；mha_sink/gqa 同尺寸 CI 运行先例） |
| P1（bM64,bN128,D64；分池） | l0a_s[2,64,64]=16 + l0a_o[2,64,128]=32 → 48 | l0b_s[2,64,128]=32 + l0b_o[2,128,64]=32 → 64 ⚠ | l0c_s[2,64,128]=64 + l0c_o[2,64,64]=32 → 96 | ✅（L0B 顶格；备选共池 [2,128,128]+切片等价顶格，分池更清晰） |

**L0（Form B；用户仅声明 L0C，L0A/L0B 由 gemm_v0 内部按 kL0Size 申请）**

| 配置类 | 用户 L0C / 128KB | gemm_v0 内部 L0B 峰值 / 64KB | 判定 |
|--------|------------------|------------------------------|------|
| P3（bM32,bN128,D256,kL0=128） | acc_s 16 + acc_o 32 = 48 | GEMM1 B-tile 128×128×2B=32KB ×2(ping-pong)=64 ⚠（session-2 #11 实测先例） | ✅ |
| D1（bM16,bN512,D≤128,kL0=32） | acc_s 32 + acc_o 8 = 40 | 512×32×2B=32KB ×2=64 ⚠（session-2 #13 实测先例） | ✅ |
| D2（bN256；kL0=128 / 64） | acc_s 16 + acc_o 8/16 = 24/32 | D≤128: 256×128×2=64（kNumSteps=1）⚠ / D=256: 256×64×2×2=64 ⚠（#10a/#10b 实测先例） | ✅ |
| D3（bN128,kL0=128） | acc_s 8 + acc_o ≤8 = 16 | 128×128×2×2=64 ⚠ | ✅ |

（Form B 内部 L0B 峰值全部有 Developer 终态 session-2 实测编译探针先例——同一
gemm_v0 模板路径 common.h:1263-1278，Expert 下行为一致；⚠ 条目 Stage 2 实编复核。）

**GM workspace 足迹（core_num 按 24 上界、NUM_STAGES=8；实际以设备查询值为准）**

| 配置类 | w1 fp32 | w2 dt | w3 fp32 | 合计 |
|--------|---------|-------|---------|------|
| P2 | 6.3MB | 3.1MB | 6.3MB | ≈15.7MB |
| P1 | 6.3MB | 3.1MB | 3.1MB | ≈12.6MB |
| P3 | 3.1MB | 1.6MB | 6.3MB | ≈11.0MB |
| D1 | 6.3MB | 3.1MB | 1.6MB | ≈11.0MB |

框架按 workspace_idx 分配（torch 缓存分配器，跨调用复用）；[cid, stage] 私有
索引 → 核间零竞争、stage 间流水。性能风险（非正确性）见 R14（L2 驻留不确定；
msprof 归因 + NUM_STAGES 降档退路）。参考实现同量级足迹已验证（mha_sink
≈19MB @17.78ms/51.72 TFlops）。

### 4.6 动态轴定义（继承 rev2，不变）

B/S/S_kv/N 为运行期可变维度，采用 **JIT 编译期参数**方案（本项目 attention 示例
统一做法，非 T.symbolic）：

| 动态轴 | 声明方式 | 运行时范围 | 重编译粒度 |
|--------|----------|-----------|-----------|
| B | `mha_kernel(B=…)` Python int → JIT 常量 | 1 ~ 128 | 每个 (B,S,S_kv,N,D,dtype,scale,causal) 组合编译一次；tilelang 缓存复用 |
| S | 同上 | 1 ~ 2048 | 同上（并决定 decode/prefill 形态分派） |
| S_kv | 同上 | 1 ~ 4096 | 同上（并决定 bN 档位/单块谓词/裁剪常量） |
| N | 同上 | 1 ~ 64 | 同上 |
| D | 同上（静态语义，取值封闭 {64,128,256}） | 64/128/256 | 同上（决定 bM/L0 计划/Form 分派） |

kernel 内由动态轴派生的量：core_num/q_tasks/r_tasks（host 计算）、task/bx/by/bz/
s0/valid_m/rows_vid（运行期标量算术）、eff_kv_iters/eff_num_outer/batch_iters
（运行期 TIR，**非 tensor 值**）、valid_n（T.min 运行期切片）——均不作为 tensor
依赖边界；NUM_STAGES/CROSS_INTERVAL 为编译期常量。

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[3],                    # Output = main 第 4 参
    workspace_idx=[4, 5, 6],        # w1/w2/w3 框架分配注入（host 调用形式不变）
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,   # 手动 T.Scope
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,      # 手动 cross_flag
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,         # 手动 set/wait_flag
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,   # 显式 alloc_* 声明制
    },
)
```

（四项全 False = Expert 标准配置：programming-model-guide §2.2 "Expert 极致性能"
行 + mha_sink:38-43 逐字一致。）

### 4.8 Host 侧 Buffer 操作审计（强制，rev3 复核）

```text
[HOST-METADATA-AUDIT]
operation: query.shape / query.dtype / key.shape 读取（校验与配置推导）
input_stride -> output_stride: 不产生新张量
shares_storage / same_data_ptr: N/A（只读元数据）
aclnn_or_physical_copy: false
result: allow

operation: Python 标量运算（sm_scale 解析、bM/bN/gemm1_kL0/form_a/core_num/
  q_tasks/r_tasks/block_num/NUM_STAGES 配置推导、assert 校验）
input_stride -> output_stride: 无张量参与
shares_storage / same_data_ptr: N/A
aclnn_or_physical_copy: false
result: allow

operation: _cube_core_num()（torch_npu.npu.get_device_properties(0).cube_core_num
  设备属性查询；异常回退常量 20）
input_stride -> output_stride: 无张量参与（设备元数据查询，非数据操作）
shares_storage / same_data_ptr: N/A
aclnn_or_physical_copy: false（先例：Developer 终态 mha.py:88-96 +
  deepseek_v4/lightning_indexer.py:14-21 + best-practices §NUM_CORES 同口径）
result: allow

operation: str(query.dtype).removeprefix("torch.")（dtype 字符串化，JIT 参数）
input_stride -> output_stride: 无张量变换（仅 Python 字符串）
shares_storage / same_data_ptr: N/A
aclnn_or_physical_copy: false（host 元数据读取，非 tensor.to(dtype) 转换）
result: allow

operation: kernel(query, key, value) 调用 + 直接 return 输出
input_stride -> output_stride: 输入原样传入；输出为 out_idx 新分配 contiguous
  [B,S,N,D]；workspace_1/2/3 由框架按 workspace_idx 内部分配注入（不经 host
  代码路径，非 host 张量操作；先例 flash_attn_bhsd.py:254 三参调用七参 main）
shares_storage / same_data_ptr: 输出为框架新分配（合法：kernel 计算产物）
aclnn_or_physical_copy: false（无 host 侧 reshape/contiguous/切片/cat/pad/to/clone）
result: allow
```

结论：host 完整路径（kernel 前 + kernel 后）只含元数据读取、纯 Python 运算、
设备属性查询、kernel 调用与结果直返；**零物理化操作、零 aclnn 依赖**。与 rev2
唯一差异 = 配置推导新增 form_a/core_num 相关标量运算 + workspace 框架注入说明，
均为 metadata-only。**特别注意**：rev3 不采用参考实现的 host 预构建 mask 张量
（mha_sink build_causal_mask / gqa build_attention_mask = host 物理化，违反本
算子契约）——掩码链保持 kernel 内计算（§3.5.2 差异表）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 混合（Cube + Vector，CV 融合，Expert 显式 workspace + vid 二分）

**判定依据**: 两次 matmul（QKᵀ、PV，Cube/L0 路径）夹一个多步 softmax
（max/exp/sum/div，Vector/UB 路径），判定为融合算子（决策树"matmul +
element-wise → CV 融合"分支；test-design 综合类别 = Fusion，workspace 配置 = True）。

### 5.2 Block 划分（继承 Developer 终态三档体系 + vid 二分 + Fixed Core + 形态分派）

```python
# —— tile 三档体系（逐字继承 Developer 终态 mha.py:120-179，探针验证过的硬限解）——
block_M = 16 if S <= 16 else (32 if D == 256 else 64)
if S <= 16:
    block_N = 512 if (D <= 128 and S_kv > 256) else (256 if S_kv > 128 else 128)
else:
    block_N = 128
gemm1_kL0 = (32 if (S <= 16 and D <= 128 and S_kv > 256)
             else (64 if (S <= 16 and D == 256 and S_kv > 128) else 128))  # 仅 Form B 使用

# —— Expert 新增维度 ——
half_M = block_M // 2                  # vid 半区行数（结构性 cv_1_2：8/16/32）
form_a = (S > 16) and (D <= 128)       # Form A: mma 全流水；Form B: gemm_v0 批量
NUM_STAGES = 8                         # 批量流水深度（偶数约束；参考实现验证值；Stage 3 可调 4-16）
CROSS_INTERVAL = 1                     # 跨核同步间隔（gqa iter3 验证值；Stage 3 可调 2-4）

# —— 任务网格（继承）——
s_blocks  = ceildiv(S, block_M);  kv_iters = ceildiv(S_kv, block_N)    # 编译期
block_num = B * N * s_blocks
core_num  = min(block_num, _cube_core_num())     # Fixed Core（设备属性查询）
q_tasks, r_tasks = block_num // core_num, block_num % core_num
# task 分解（bx 最快维，继承）：bx = task % s_blocks; by = task//s_blocks % N;
#                               bz = task//s_blocks//N % B
```

选择理由（继承 + Expert 增量）：
- **bM/bN/kL0 三档**：Developer 两轮调优的实测硬限解（L0B transpose_B 预算规则
  common.h:1263-1278、L1 511.75KB、UB 191.75KB），语义与性能均已验证——最小必要
  改动原则下不重开 tile 搜索空间；Expert 的 UB 红利（vid 半区）不用于扩 tile，
  而用于留流水余量（§4.5）。
- **bM ≥ 16 且 half_M = bM/2 ≥ 8**：cube 分形下限 M≥16 作用于整块 bM（C 侧 GEMM
  不分 vid）；V 侧 half_M=8 为 tile op 行数新下限形态（R10 探针覆盖）。bM 均为
  偶数（16/32/64）→ half_M 整除无余，vid 二分无残行。
- **Form 分派谓词 `form_a = S>16 ∧ D≤128`**：Form A（mma 双槽流水）要求
  l0b [2, D, bN] ≤ 64KB → D≤128 ∧ bN=128 时恰 64KB 顶格（mha_sink 同尺寸先例）；
  D=256 需手动 K-split（l0b [2,128,128] 顶格 + 2 步累加 mma，L0A/L0B 全顶格无
  余量）——为控制 v1 风险，D=256 prefill 走 Form B（gemm_v0 kL0Size=128 内部
  2 步分片，session-2 #11 实测先例）；decode（S≤16）走 Form B 以直接继承
  kL0Size=32/64 的 bN=512/256 L0B 解（session-2 #13/#10b 实测先例，避免 mma
  形态手动 K/N-split 复杂度）。Form B→A 转换候选（decode/D=256 的 mma 化）列
  Stage 3（§10.3-⑤）。
- **NUM_STAGES=8 / CROSS_INTERVAL=1**：gqa_fwd_varlen iter3 参数扫描实证
  （8+8 两满批优于 14+2；interval=1 让 Vector 更早启动，合计 9.9% 收益）；
  NUM_STAGES 偶数约束（io_buf/L0 双槽轮转，mha_sink:112 assert 同款）。
- **cid 分解 bx 最快维**（继承）：同 head 任务相邻，K/V 重读局部性最好；Fixed
  Core 连续任务块在 #1 causal 裁剪下负载均衡（Developer 终态 #5 注释论证继承）。

### 5.3 约束分析

- **分形限制**（fp16/bf16：M≥16, K≥16, N≥16）：Form A GEMM1 M=bM(64)✅
  K=D(64/128)✅ N=bN(128)✅；GEMM2 M=bM✅ K=bN(128)✅ N=D✅。Form B 由
  gemm_v0 内部分片保障（继承 Developer 验证）。
- **对齐要求**：GM 行片段 D×2B ∈ {128,256,512}B ≥32B ✅ 且 32B 整数倍；行 pitch
  N×D×2B 为 32B 整数倍 ✅；UB/L1 buffer 尺寸 256B 整数倍 ✅；ws 切片行连续段
  bN×sizeof(fp32) ∈ {512B, 1KB, 2KB} ✅；L0 fractal 16×16 粒度：全部 tile 维度
  为 16 倍数（bM 16/32/64、bN 128/256/512、D 64/128/256）✅。
- **UB 容量**：六配置 30-116KB ≤ 191.75KB ✅（§4.5 表，最紧 P2 占 60%）。
- **L0C 容量**（128KB，强制检测 `Σ slot×M×N×4B ≤ 131072`）：Form A P2 64KB
  （共池）/ P1 96KB（分池）；Form B 16-48KB（用户 L0C 单槽）✅。保守元素规则
  `M×N ≤ 16384（fp32）`：Form A 单槽最大 64×128=8192 ✅；Form B acc_s 最大
  16×512=8192 ✅。
- **L0A/L0B**（各 64KB）：Form A P1/P2 见 §4.5（顶格 ⚠ 实编验证）；Form B
  gemm_v0 内部 kL0Size 分片（session-2 实测先例 ⚠）✅。
- **L1**（511.75KB）：最大 D1 276KB ✅；decode D=256 bN=512 = 536KB ✗ 排除
  （继承硬限）。
- **threads/vid**：Expert 帧 vid extent 2 结构性绑定（无 threads 参数；R8 复检
  不触发，§2.2-c）✅。
- **Kernel 维度**：一维 core_num ✅。

### 5.4 注意事项（非整除与尾块，显式设计；vid 二分新规则）

S/S_kv 不保证被 block 整除（S∈[1,2048] 任意、S_kv∈[1,4096] 任意）。**禁止 host
padding + crop**；输入、输出 GM 两侧一律显式 valid extent BufferRegion：

| 位置 | valid extent | 表达式 | 越界区语义 |
|------|-------------|--------|-----------|
| Q 搬入（C 侧） | valid_m | `T.min(block_M, S - bx*block_M)` | q_l1 尾行由 copy_gm_to_l1 need_clear→InitConstValue 清零（common.h:84-92 源码确认，继承 rev2）；零行产生有限垃圾 scores，仅影响自身行，写回时丢弃 |
| K/V 搬入（C 侧） | valid_n | `T.min(block_N, S_kv - j0)` | k_l1/v_l1 尾行清零 → 尾列 scores=0（有限）、v 尾行=0 |
| softmax 域 | 尾列掩码 | `thresh = min(i_g+offset_eff, S_kv-1)` 钳位（§3.1） | 尾列（j_g ≥ S_kv）置 -inf → P=exp(-inf)=0，不进 sumexp；GEMM2 中 0×0=0（v 尾行已清零，无 0×NaN 风险） |
| w1/w2/w3 交换 | 整 bM 行 | ws 槽位恒为整块 [bM, bN]/[bM, D] | 垃圾行（Q 尾行）的有限值随块交换，V 侧照常计算（行独立），写回时按 vid 裁剪丢弃 |
| Output 写回（V 侧，**vid 分行新规则**） | rows_vid | `rows_vid = T.min(half_M, T.max(0, valid_m - vid*half_M))`；`if rows_vid > 0:` 守卫内 `T.copy(out_half[0:rows_vid,:], Output[bz, s0+vid*half_M : s0+vid*half_M+rows_vid, by, :])` | vid0 承担前 min(half_M, valid_m) 行；vid1 承担其余行，valid_m ≤ half_M 时 rows_vid=0 → **运行期 if 跳过写回**（避免 0 行 copy 的未定义行为）；只写有效行，不越界、不覆盖相邻任务区域（任务间输出区域天然不相交，无竞态） |

**尾块正确性论证（三重保障继承 + vid 扩展）**：① GM→L1 拷贝尾区**确定性清零**
（源码 need_clear；杜绝陈旧 NaN 位模式导致的 0×NaN=NaN 污染）；② 尾列经阈值钳位
进 -inf 掩码，softmax 分母不含尾列；③ 输出只写 rows_vid 行（vid 分行裁剪）。
垃圾行（Q 尾行）的有限值不参与任何跨行归约（reduce dim=-1 行内闭合），行独立性
保证有效行结果与整除情形逐位一致——**vid 二分不破坏行独立性**（每 vid 的 half_M
行自成闭合域，跨 vid 无归约/无共享状态；w1/w2/w3 的 vid 半区切片互不重叠）。

**极小 shape**：S=1（decode）→ s_blocks=1、valid_m=1、bM=16、half_M=8：vid0
rows_vid=1（写 1 行）、vid1 rows_vid=0（守卫跳过）；GEMM 以 bM=16 行缓冲执行
（15 行零填充垃圾，计算合法、结果丢弃）。S_kv=1 → kv_iters=1（single_kv_block
特化）、valid_n=1，掩码钳位 thresh=0，P[:,0]=1 → 输出 = V[:,0,:] ✅。N=1 →
行 pitch = D（退化连续搬运）✅。

**BSHD 行跨步切片 × Expert vid 组合验证（Stage 2 强制第一步，R10）**：4D 切片
`Q[bz, s0:s0+valid_m, by, :]`（C 侧，extents [1,valid_m,1,D]）与
`Output[bz, s0+vid*half_M : s0+vid*half_M+rows_vid, by, :]`（V 侧，vid 偏移 +
运行期行数）在 rev2 已有 threads=1 位级验证基础（verify_bshd_slice/verify_t1a/
verify_kv_tail 探针族），但 **× Expert 帧（vid=blockIdx.y）× ZN/NZ layout
annotation 的组合为仓内首例**——Stage 2 必须先行最小复现探针（整除 + S=1/30/64/90
运行期 extent + vid 半区写回位级比对，含 annotate_layout 开/关两态），通过后再
进入 L0；应急预案（未触发保留备查）：去 layout annotation（性能降级不换语义）/
按行 1D 循环搬运（sparse_flash_attn_developer.py:148-152 gather 同构）。

### 5.5 数据搬运性能可行性（REORDER-COST，强制）

本算子**无物理布局重排**（BSHD 原生进/原生出，无 host/kernel 重排 pass）；
Expert 新增的 workspace 流量为 CV 交互固有成本（§8.2：物理路径必然，rev2 的
AUTO_CV 同样产生，且 rev3 对其做 [cid,stage] 私有索引 + dtype 主线控制）。
逐路径审计：

```text
[REORDER-COST]（rev3：Expert Fixed Core 任务路径 + Developer 终态实测锚定）
path: form_a（prefill D≤128，mma 全流水）与 form_b（decode 全 D + prefill D=256，
  gemm_v0 批量）——两路径搬运结构同构（差异仅 C 侧 L0 装载方式），合并审计；
  掩码 4 编译期变体与 single_kv_block 特化共用同一搬运结构
gm_passes: Q×1 + Output×1 + K/V × s_blocks（flash 标准 KV 重读因子 = S/block_M）
  + workspace w1/w2/w3 × 每 KV 迭代各 1 写 1 读（CV 交互固有段，[cid,stage]
  私有索引、核间零竞争；w1/w3 fp32 主线、w2 dtype——rev2 AUTO_CV 同为 3 段
  GM 往返（final_report §S2-5），rev3 段数持平、同步结构批量化）
dma_transactions: 每任务每迭代 ≈ 2·bN 行（K/V）+ 3 写 3 读 ws tile 级 + 2·hM 行
  （V 侧 ws 读，双 vid 合计 = bM 行级）；S1 哨兵（§9.4 perf_prefill_worst）：
  8192 任务 × eff≈24 迭代（causal 裁剪后均值）× (2×128 行 + ~8 tile) ≈ 5.5×10⁷
  ——与 rev2 审计值 6.8×10⁷ 同量级（裁剪均值口径差异）
average_dma_bytes: Q/K/V/Output 行片段 = D×sizeof(dt) ∈ {128,256,512}B ≥32B 对齐、
  无逐元素搬运；ws tile 段 = bN×4B(fp32)/bN×2B(dt) 连续行段 ≥512B；行跨步仅体现
  在行间 gap（DMA 引擎原生 srcGap，无额外指令）
gm_scalar_accesses: 0（全程块 DMA；无单元素 T.copy、无索引 gather）
address_div_mod_per_element: 0（地址增量由切片静态推导；每任务 3 div + 2 mod 解码
  task → bx/by/bz，C/V 双侧同式）
active_cores / serial_tasks_per_core: core_num（设备查询，910B 级 20-24）/
  ceil(block_num/core_num)（S1: 342-410）；Expert cv_1_2：每 cid = 1 AIC + 2 AIV，
  vector 工作双 AIV 分担（对 rev2 单 AIV 是 2× vector 并行度）；C∥V 跨核重叠由
  NUM_STAGES 批量流水实现（GEMM1 批(i+1) ∥ softmax 批(i) ∥ GEMM2 批(i-1) 深度重叠）
measured_kernel_path（锚点继承）: Developer 终态哨兵 kernel 路径 6.1-7.2s
  （final_report §S2-4）；Expert 首测值待 Stage 2 产出（同门限复检）
largest_case_timeout_gate: S1-S4 kernel 路径门各 ≤120s（§9.4 继承 rev2-C4 结构：
  H2D→kernel(含首次编译)→D2H；golden 成本不计入）+ case 级 1800s 防挂死参考上限
  （非阻塞）
result: pass（结构审计；实测锚定以 Stage 2 哨兵复检为准——若 kernel 路径逼近
  门禁，触发 §10.3 降级旋钮：NUM_STAGES 8→4 / CROSS_INTERVAL 1→2 / w1 dtype 瘦身）
```

S1 定量核算（继承 rev2 + workspace 增量）：GM 流量 ≈ K/V 重读 17.2GB + Q/Out
0.27GB + **ws 流量 ≈ 8192 任务×24 迭代×(32+16+32)KB×2(写+读)/2(vid 半区读侧
合计=整块) ≈ 15.1GB**——总 ≈32.6GB @≈1TB/s ≈ 33ms；算力 4·B·N·S·S_kv·D ≈
1.1 TFlop（causal 裁剪后 ~0.7 TFlop）@50-150 TFLOPS ≈ 5-15ms（gqa 先例实证
51.72 TFlops 可达）；ws 流量为新增主体，但其为**核私有顺序段**（L2 友好 +
流水隐藏），非延迟串联段——rev2 的同源流量（AUTO_CV w1/w2/w3）已被实测吸收
（哨兵 ≥9× 余量），rev3 以更优同步结构承载同量级流量，可行性判定 pass。

### 5.6 多路径分派审计（DISPATCH-COVERAGE，强制）

```text
[DISPATCH-COVERAGE]（rev3：Expert 双形态 + vid 二分 + 批量流水）
supported_domain: Q/K/V dtype ∈ {float16, bfloat16}（三者一致）；
  shape：B∈[1,128], S∈[1,2048], S_kv∈[1,4096], N∈[1,64], D∈{64,128,256}；
  attr：scaleValue∈ℝ（≤0 自动 1/√D），is_causal∈{True,False}（True ⇒ S≤S_kv）；
  输入值域：任意有限实数
generic_fallback: 单一通用 host 入口 mha()（上述全域）；kernel 侧 Form B
  （gemm_v0 批量）承担 decode 全 D + prefill D=256，Form A（mma 流水）承担
  prefill D≤128——两形态并集 = 全域，且各自内部 kv_iters≥2 通用 online softmax
  路径兜底；分派全部位于 host 纯 Python 元数据层（编译期谓词），无输入落空
specializations:
  - predicate: form_a = (S > 16) ∧ (D ≤ 128) → Form A（mma 全流水）；否则 Form B
    fallback_on_miss: Form B（谓词二分穷尽，Form B 自身覆盖剩余全域）
    equivalence_evidence: 两形态共享 V-scope 源码与 workspace/信号量协议
      （§3.3 emitter 注记）；C 侧差异仅 GEMM 执行方式（mma 双槽 vs gemm_v0
      内部 L0），数学同一（fp32 累加、K 次序一致：Form A K=D≤128 单 mma 内部
      K-loop = Form B gemm_v0 kL0Size=128 单步；§9.3.3-6 位级论证）
  - predicate: S ≤ 16 → bM=16（decode 窄块）+ bN 三档（512/256/128 按
    D≤128∧S_kv>256 / S_kv>128 / 其余）+ gemm1_kL0 三档（32/64/128）；
    S > 16 → bM = 32(D=256)/64，bN=128
    fallback_on_miss: 不适用（谓词在 S/D/S_kv 域上穷尽，逐字继承 Developer
    终态已验证分派）
    equivalence_evidence: §4.5/§5.3 预算表——仅资源占用不同，算法同一；
    Developer 终态 12 项 probe（BITEXACT/阻塞口径）实证记录（optimization_log）
  - predicate: (is_causal, S_kv % block_N == 0) → 掩码 4 变体：
      (F,T)=无掩码链；(F,F)=仅尾列钳位；(T,T)=仅 causal；(T,F)=causal+钳位
      （统一 offset_eff 阈值式，非 causal 时 min(s0+i+2³⁰, S_kv-1) 退化为钳位；
      编译期 need_mask 开关 + 运行期全可见跳过 #4 叠加）
    fallback_on_miss: 不适用（2×2 编译期穷尽）
    equivalence_evidence: §3.1 索引映射（vid 半区行偏移细化）——thresh=min(i_g+
      offset, S_kv-1) 与 golden masked_fill(j > i+offset) + 尾列裁剪逐位等价；
      运行期跳过条件按 s0 整块判定 → 两 vid 同值同分支（无 flag 配对风险）
  - predicate: kv_iters == 1（即 S_kv ≤ block_N）→ single_kv_block 预归一化特化
    fallback_on_miss: kv_iters ≥ 2 → 通用批量 online softmax（phase1/phase2
      deferred rescale），谓词二分穷尽
    equivalence_evidence: 单迭代下 sumexp 在迭代内即终值，预归一化与终归一化
      代数同值、仅舍入位置不同——且恰与 golden 一次性 softmax 结构同构；
      Developer 终态实测（probe GOLDEN-BITEXACT on Skv=129/192/300，
      optimization_log #8d/#11/#13）；rev3 算式次序逐字保持（§8.4）
  - predicate: scaleValue > 0 → s=scaleValue；否则 s=1/√D
    fallback_on_miss: 不适用（ℝ 全函数）
    equivalence_evidence: golden `if scaleValue <= 0: scaleValue = 1.0/(D**0.5)` 逐字一致
unsupported_inputs: dtype 三者不一致 / dtype∉{fp16,bf16} / D∉{64,128,256} /
  is_causal∧S>S_kv / 维度超出支持范围 / 非 contiguous 输入 / 非有限值输入
  （NaN/Inf 不在声明值域，L2/Boundary 非阻塞探测）——全部由 host 元数据断言
  拒绝（不读数据），与 supported_domain 无冲突
result: pass（覆盖命中映射：Form A→L0 #5,6,7,8,9 + 哨兵 perf_prefill_worst/
  perf_max_grid；Form B prefill D=256→L0 #10 + 哨兵 perf_d256_bf16；Form B
  decode→L1 case10/13/14/16 类（现有套件）+ 哨兵 perf_decode_worst；bN 三档→
  L1/哨兵 decode 系；掩码 4 变体→(F,T):#1,3,5,7,10 / (T,T):#2,4,6,8,9 /
  (F,F),(T,F)→L1（S_kv%bN≠0 现有用例）；single_kv_block→#1-4,10 + L1 Skv≤bN 系；
  通用多块→#9 + L1/哨兵；vid 分行/rows_vid 守卫→L1 奇数 S 系（S=1/30/90 类，
  现有套件）+ R10 探针；scale 两分支→#1-9 自动 / #10 显式）
```

分派只读取 shape/dtype/attr 元数据（纯 Python），不读取、不改动张量数据 ✅；
每条 specialization 在 §9.2 L0 计划或现有 L1/哨兵套件中至少 1 个命中用例 ✅
（decode 形态无 L0 命中为继承 rev2 的既有格局——L0 表以 prefill 规则 shape 为主，
decode 路径由 L1 + perf_decode_worst 哨兵强制覆盖，Stage 2 现有套件已含）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 任务网格 (B×N×S_blocks) | Fixed Core + 核内任务循环 | `T.Kernel(core_num, is_npu=True) as (cid, vid)` + `for t in T.serial(my_count)` | 一维网格（平台不支持三维 Kernel）；core_num=min(block_num, 设备核数) 静态均衡分配（继承 #5；mha_sink:236-238 同式）；任务间零依赖；Expert 帧 vid extent 2 结构性绑定（§2.2-b） |
| KV 外层批 (eff_num_outer) | 串行循环（**运行期 TIR 边界**） | `T.serial(eff_num_outer)`，`eff_num_outer = T.ceildiv(eff_kv_iters, NUM_STAGES)` | causal 运行期裁剪（#1 继承）+ 批量化（mha_sink:278/433 Expert 运行期边界 CI 先例）；C/V 双 scope 同式复制 → flag 配对平衡（§7.2-5 硬约束） |
| KV 内层批 (batch_iters) | 串行循环（运行期 TIR 边界） | `T.serial(batch_iters)`，`batch_iters = T.if_then_else(rem < NUM_STAGES, rem, NUM_STAGES)` | 尾批余数处理（mha_sink:280 同式）；批内 side=i%2 手动 ping-pong（L0 双槽 + io_buf 双槽） |
| 双缓冲轮转 | 编译期/运行期索引 | `side = i % 2`、`io_side = i % 2` | 手动多缓冲（R9 语境下 T.Pipelined 禁用，手动 ping-pong 为 Expert 标准形态） |
| Form A GEMM1 Q 预载 | 批内条件 | `if i < 2: T.copy(q_l1, l0a_s[side,:,:])` | 每外层批首 2 迭代预载 Q 至两 L0A 槽，其后复用（mha_sink:295-296 同构） |
| 掩码/softmax 行处理 | SIMD 2D tile op | `T.tile.broadcast/compare/select/sub/exp/div` + `T.reduce_max/sum(dim=-1)` | broadcast 2D 形式（#2/#3 继承），无标量行循环；作用于 [hM, bN] vid 半区 |
| 元素级 | SIMD 向量化 | `T.tile.*` | UB buffer 级指令 |

**Python 级形态分支注记**：Form A/B 在 `mha_kernel` 工厂内以 **Python 级 if**
构建两个自包含 prim_func（V-scope 由共享 emitter 函数生成、C-scope 各自完整），
host 按 `form_a` 谓词分派——规避 TVMScript parse-time if 的跨块变量作用域限制
（rev2-A1 教训："块内定义的变量不能跨块使用"），两形态独立编译、独立验证。
**明确不采用**：`T.Pipelined`（任何 num_stages；R9 语境 + Expert 参考实现零使用
——流水由批量结构 + 手动 ping-pong 表达）；`T.Parallel`（Expert 形态统一
T.tile.* 原语）；`T.Persistent`（Fixed Core 任务循环已实现持久化语义）。

### 6.2 循环伪代码

```python
with T.Kernel(core_num, is_npu=True) as (cid, vid):
    <C 侧/V 侧 buffer 显式分配（§4.3；Form A/B 的 L0 差异在 Python 级分支内）>
    my_start/my_count = Fixed Core 均衡分配（T.if_then_else 余数式）

    with T.Scope("C"):                                   # ══ Cube 核 ══
        <init flag 闭环（§7.2-3）>
        for t in T.serial(my_count):                     # 任务循环
            task = my_start + t; bx/by/bz 分解; s0; valid_m
            eff_kv_iters = <causal 裁剪式>; eff_num_outer = ceildiv(eff_kv_iters, NUM_STAGES)
            <SIG_Q_L1 管程> T.copy(Q[bz, s0:s0+valid_m, by, :], q_l1)
            for k in T.serial(eff_num_outer):            # 外层批（运行期边界）
                batch_iters = min(NUM_STAGES, eff_kv_iters - k*NUM_STAGES)
                T.wait_cross_flag(SEM_WS1_V2C)           # w1 槽位已释放
                for i in T.serial(batch_iters):          # GEMM1 批
                    <K 载入(SIG_K_L1) → [A] L0 双槽+mma / [B] gemm_v0 → w1[cid,i]>
                    <每 CROSS_INTERVAL 块 set_cross_flag("FIX", SEM_WS1_C2V)>
                T.wait_cross_flag(SEM_WS3_V2C)           # w3 槽位已释放
                for i in T.serial(batch_iters):          # GEMM2 批
                    <V 载入(SIG_V_L1) + w2[i]→p_l1(SIG_P_L1+SEM_WS2_V2C)
                     → [A] mma / [B] gemm_v0 → w3[cid,i]>
                    <每 CROSS_INTERVAL 块 set_cross_flag("FIX", SEM_WS3_C2V)>
                T.set_cross_flag("MTE2", SEM_WS2_C2V)    # w2 槽位释放给 V
            <set_flag("MTE1","MTE2", SIG_Q_L1)>          # 任务末归还 q_l1
        <destroy flag 闭环>

    with T.Scope("V"):                                   # ══ Vector 核（每 vid 半区）══
        <init flag 闭环（§7.2-3）>
        for t in T.serial(my_count):
            task/bx/by/bz/s0/valid_m 同式; eff_* 同式（配对平衡硬约束）
            rows_vid = T.min(half_M, T.max(0, valid_m - vid*half_M))
            <状态初始化 fill(acc_o/sumexp/m_i)>
            if need_mask: <row_thresh hoist（起点 + vid*half_M）>
            for k in T.serial(eff_num_outer):
                batch_iters 同式
                T.wait_cross_flag(SEM_WS2_C2V)           # phase 1 前置
                for i in T.serial(batch_iters):          # softmax 批
                    <w1[cid,i,vid半区] → io_buf[io_side] → work_ub（SIG_IO_UB+SEM_WS1_C2V）>
                    <m_i_prev 备份 → 双舍入链 → [need_mask ∧ 运行期非全可见] 掩码链>
                    <reduce_max → max 合并 → r_factors[i]=sub → broadcast-sub → exp
                     → reduce_sum → sumexp_is[i] → [single_kv_block] 预归一>
                    <work_ub → acc_s_half → w2[cid,i,vid半区]（SIG_S_HALF+SEM_WS2_V2C）>
                T.set_cross_flag("MTE2", SEM_WS1_V2C)    # w1 槽位释放给 C
                for i in T.serial(batch_iters):          # O 累加批（phase 2）
                    <[非单块] exp(r_factors[i]) → sumexp 折叠 → acc_o rescale>
                    <w3[cid,i,vid半区] → io_buf → work_ub（SIG_IO_UB+SEM_WS3_C2V）>
                    <acc_o += work_ub>
                T.set_cross_flag("MTE2", SEM_WS3_V2C)    # w3 槽位释放给 C
            <[非单块] 终归一 sumexp+1e-30 → div>
            <SIG_S_HALF 闭环内：acc_o → out_half → if rows_vid>0: 行跨步写回>
        <destroy flag 闭环>
```

### 6.3 流水线与预取（rev3 核心性能结构）

- **批内多级重叠（Form A）**：GM→L1（MTE2）∥ L1→L0（MTE1）∥ mma（M）∥
  L0C→w1（FIX）四段流水由 pipe-pair flag 精确衔接（side 双槽轮转）——GEMM1 批的
  K 载入(i+1) 与 mma(i) 与 w1 写出(i-1) 重叠；对照 rev2 Developer：AUTO_SYNC
  逐 op `PipeBarrier<PIPE_ALL>` 全排空（final_report §5-#9 codegen 实证）。
- **跨核批量重叠**：GEMM1 批 ∥ softmax 批 ∥ GEMM2 批经 [cid, stage] 多槽
  workspace 解耦——C 侧生产 w1(i) 时 V 侧可消费 w1(i-1)（SEM 信号量按
  CROSS_INTERVAL 批量发放）；对照 rev2：C↔V 逐迭代串行交接（每迭代 3 对
  cross-flag + 3 段 ws 往返同步等待）。
- **V 侧 MTE2/V 重叠**：io_buf 双槽——MTE2 载入 ws(i+1) 与 V 计算 ws(i) 重叠
  （gqa iter4 实证 +16KB UB 换全重叠）；phase1/phase2 分时复用 io_buf/work_ub。
- **预取现状与候选**：k_l1/v_l1 为单缓冲（SIG_K/V_L1 管程串行 GM→L1→L0）；
  L1 预算余量充足（P2 96KB/511.75KB）→ **k/v L1 双缓冲列 Stage 3 候选**
  （§10.3-③，需 flag 管程翻倍，先以 msprof 归因确认 MTE2 为瓶颈再落地）。
- **T.Pipelined 禁用依据（继承 + Expert 语境）**：rev2-R9 codegen bug 的前提
  （MEMORY_PLANNING 双缓冲注入）在 Expert 不存在，但 T.Pipelined 与手动
  ping-pong 语义重叠且无 Expert 先例——全文禁用，流水一律手动表达。
- **恢复/降级旋钮**（性能不达标或超门禁时按序尝试，每项落地后复跑 §9.4 门 +
  L0 阻塞口径）：NUM_STAGES 8→4/16；CROSS_INTERVAL 1→2/4；w1 dtype 瘦身
  （探针门）；k/v L1 双缓冲；Form B→A 扩展（D=256/decode）。

### 6.4 尾块处理

循环边界**不因尾块改变**（eff_num_outer/batch_iters 用 ceildiv/min 向上取整，
尾批作为最后 batch_iters<NUM_STAGES 的批、尾块作为批内最后一次迭代），尾块由
迭代内 `valid_m/valid_n/rows_vid` 运行期切片表达：输入侧 `Q/K/V[…, :valid]`
（GM 两侧显式 BufferRegion + need_clear 清零）、输出侧 `out_half[0:rows_vid,:] →
Output[…vid 半区…]`（运行期 if 守卫）。不使用"标量 GM 起点 + 完整 UB tile"写法；
不做 host padding + crop；掩码阈值钳位保证尾列不进 softmax 分母（§5.4 三重保障
+ vid 分行规则）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 全手动同步（Expert 标准）——四 pass_configs 全 False；kernel 源码显式
书写全部同步点：C 侧 `T.set_flag/T.wait_flag`（Form A 细粒度 pipe-pair；Form B
`T.barrier_all` 粗粒度）+ 跨核 `T.set_cross_flag/T.wait_cross_flag`（C↔V 信号量）
+ V 侧 MTE2/MTE3↔V flag 管程 + init/destroy 闭环。

### 7.2 同步点说明（flag 矩阵）

**（1）Cross-core Semaphore（AIC ↔ 2×AIV 组，mode=2 默认；ID 继承 mha_sink:143-149）**

| SEM ID | 名称 | 方向 | set 位置 | wait 位置 | 语义 |
|--------|------|------|----------|-----------|------|
| 0 | SEM_WS1_C2V | C→V | C：GEMM1 批内每 CROSS_INTERVAL 块 `set_cross_flag("FIX",·)` | V：phase1 批内 `i%CROSS_INTERVAL==0` 时 | w1[cid, 0..i] scores 就绪 |
| 1 | SEM_WS1_V2C | V→C | V：phase1 批末 `set_cross_flag("MTE2",·)` | C：每外层批 GEMM1 前 | w1 槽位已消费，可复写 |
| 2 | SEM_WS2_V2C | V→C | V：phase1 批内每 CROSS_INTERVAL 块 `set_cross_flag("MTE3",·)` | C：GEMM2 批内 `i%CROSS_INTERVAL==0` 时 | w2[cid, 0..i] P 就绪 |
| 3 | SEM_WS2_C2V | C→V | C：每外层批 GEMM2 批末 `set_cross_flag("MTE2",·)`；init 预置 | V：每外层批 phase1 前 | w2 槽位已消费，可复写 |
| 4 | SEM_WS3_C2V | C→V | C：GEMM2 批内每 CROSS_INTERVAL 块 `set_cross_flag("FIX",·)` | V：phase2 批内 `i%CROSS_INTERVAL==0` 时 | w3[cid, 0..i] PV 就绪 |
| 5 | SEM_WS3_V2C | V→C | V：phase2 批末 `set_cross_flag("MTE2",·)`；init 预置 | C：每外层批 GEMM2 前 | w3 槽位已消费，可复写 |

（wait_cross_flag 单参调用——pipe 参数仅 A5 平台支持，910B 必须留空，
ascend.py:213-231 docstring 明示；两 AIV 各自 set、AIC 一次 wait 的组聚合语义
为硬件 cross-flag 行为，参考实现 CI 验证，逐字继承不重新发明。）

**（2）C 侧 Intra-core Flag（Form A 细粒度；有向 pipe pair 独立编号，ID 继承 mha_sink:151-163）**

| Ownership | 有向 pair | ID | 语义 |
|-----------|-----------|-----|------|
| K L1 | MTE1→MTE2 / MTE2→MTE1 | SIG_K_L1=0 | k_l1 可写 / 已就绪 / 已消费归还 |
| P L1 | MTE1→MTE2 / MTE2→MTE1 | SIG_P_L1=1 | p_l1 双向 ownership |
| V L1 | MTE1→MTE2 / MTE2→MTE1 | SIG_V_L1=2 | v_l1 双向 ownership |
| Q L1 | MTE1→MTE2 / MTE2→MTE1 | SIG_Q_L1=3 | q_l1 任务级双向 ownership（任务末归还） |
| L0A/L0B 槽 0/1 | M→MTE1 / MTE1→M | SIG_L0AB=0,1 | 双槽：M 消费完毕可重载 / 载入就绪可 mma |
| L0C 槽 0/1 | FIX→M / M→FIX | SIG_L0C=0,1 | 双槽：FIX 已取出可重算 / mma 就绪可 FIX 写出 |

（Form B 无 MTE1/M/FIX 用户 flag——gemm_v0 内部自管 L1→L0 与 mma 时序，用户侧
以 `T.barrier_all()` 分隔 GM→L1 载入、gemm_v0、L0C→ws 写出三段
（flash_attn_bhsd:88-114 同构）；跨核 SEM 协议与 Form A 完全一致。）

**（3）V 侧 Intra-core Flag（每 AIV 独立，ID 继承 mha_sink:165-173）**

| Ownership | 有向 pair | ID | 语义 |
|-----------|-----------|-----|------|
| io_buf 槽 0/1 | V→MTE2 / MTE2→V | SIG_IO_UB=0,1 | 双槽：V 消费完毕可重载 / 载入就绪可读 |
| acc_s_half（含 out_half 写出复用段） | MTE3→V / V→MTE3 | SIG_S_HALF=0 | V 写就绪→MTE3 读→归还；最终 Output store 同闭环（best-practices §6 尾注：store 必须进 ownership 闭环，仅 store 前 barrier_all 不能保护下一 task 的 V 写覆盖） |

**（4）init/destroy 闭环（作用域级，逐字继承 mha_sink:241-250/367-375/379-386/625-629）**

```python
# C scope init（模拟消费者已释放）：
T.set_cross_flag("MTE2", SEM_WS2_C2V)
T.set_flag("MTE1","MTE2", SIG_K_L1); T.set_flag("MTE1","MTE2", SIG_P_L1)
T.set_flag("MTE1","MTE2", SIG_V_L1); T.set_flag("MTE1","MTE2", SIG_Q_L1)
[Form A] T.set_flag("M","MTE1", SIG_L0AB); T.set_flag("M","MTE1", SIG_L0AB+1)
         T.set_flag("FIX","M", SIG_L0C);   T.set_flag("FIX","M", SIG_L0C+1)
# C scope destroy（消费未决 init 方向 token）：与上对称的 wait_flag ×8（Form A）/×4（Form B）
# V scope init：
T.set_cross_flag("MTE2", SEM_WS1_V2C); T.set_cross_flag("MTE2", SEM_WS3_V2C)
T.set_flag("V","MTE2", SIG_IO_UB); T.set_flag("V","MTE2", SIG_IO_UB+1)
T.set_flag("MTE3","V", SIG_S_HALF)
# V scope destroy：wait_flag("V","MTE2",SIG_IO_UB/+1); wait_flag("MTE3","V",SIG_S_HALF)
```

**（5）flag 配对平衡硬约束（运行期边界的正确性前提）**：`eff_kv_iters/
eff_num_outer/batch_iters` 必须在 C/V 两 scope 以**同一 bx/常量/公式**各自计算
（不得经 workspace 传递），保证任意 causal 裁剪长度下双侧 set/wait 次数逐批
相等（mha_sink:122-127/394-400 注释先例："identical accounting to the
fixed-num_outer case；leftovers carry to next task / init / destroy"）。运行期
掩码跳过（#4）与 rows_vid 守卫（§5.4）内的 flag 均为**同 vid 局部、成对出现**
（set+wait 同进同出）或纯 V 无 flag 段——不破坏配对。

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,   # 手动 T.Scope("C"/"V")
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,      # 手动 SEM 0-5
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,         # 手动 SIG flag 矩阵
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,   # 显式 alloc_*（planner 仅编址）
}
```

（Expert 标准全 False；不写 T.annotate_address——planner 顺序编址即可满足
codegen address_map 注册（ascend_memory_planning.cc:59-92），mha_sink 先例。）

---

## 8. 融合算子设计

### 8.1 融合算子判定

**判定结果**: 是

**判定依据**: GEMM(QKᵀ) + element-wise/reduction 后处理（scale/mask/softmax）+
GEMM(PV)，Cube 与 Vector 交替协作，属 CV 融合算子（决策树"matmul +
element-wise → CV 融合"分支）。

### 8.2 CV 交互设计（Expert：显式 workspace + vid 二分）

**Workspace 规格表**：

| workspace | Shape | dtype | 用途 | 写方 | 读方 |
|-----------|-------|-------|------|------|------|
| workspace_1 | (core_num, NUM_STAGES, bM, bN) | fp32 | GEMM1 scores（含垃圾行；双舍入链输入） | C（FIX，copy_l0c_to_gm T1=T2=float 无 cast） | V（MTE2，vid 半区行切片） |
| workspace_2 | (core_num, NUM_STAGES, bM, bN) | dt（fp16/bf16） | softmax P（或单块特化 P̂） | V（MTE3，vid 半区） | C（MTE2 整块 → p_l1） |
| workspace_3 | (core_num, NUM_STAGES, bM, D) | fp32 | PV 块输出（fp32 = 精度红线） | C（FIX） | V（MTE2，vid 半区） |

**workspace_idx**: `[4, 5, 6]`（main 签名第 5/6/7 参，与装饰器顺序一致——
flash_attn_bhsd/gqa/mha_sink 同款位置约定）；框架分配注入，host 调用形式不变。

**为什么 CV 交互必须经 GM workspace（"0 段 GM 往返"不可达的源码证据）**：

1. `T.copy(L0C_buf, UB_buf)` 在 codegen 分派为 `copy_l0c_to_ub` 并标记
   `config.virtual_channel = true`（ascend.cc:296-305，注释原文 "In
   'ub -> gm -> l1' path, gm is always row major"）；`T.copy(UB_buf, L1_buf)`
   同理分派 `copy_ub_to_l1` + virtual_channel（ascend.cc:287-295）。
2. **copy_l0c_to_ub 无设备端实现**：src/tl_templates/ascend/ 全目录仅存在
   `copy_l0c_to_gm`（common.h:190）——L0C 的唯一物理出口是 FIX→GM。
3. AscendWorkspaceReduction pass 将两类虚拟拷贝**一律重写**为 GM workspace 路径
   （替换表 ascend_workspace_reduction.cc:1004：`{copy_ub_to_l1 → copy_ub_to_gm,
   copy_l0c_to_ub → copy_l0c_to_gm}`；target set :1001/:115-118），并自动注入
   workspace——这正是 Developer AUTO_CV "片上直连"API 之下实际产生 w1/w2/w3
   三段 GM 往返的机制（final_report §S2-5 实测归因互证）。
4. 三个 Expert 参考实现（flash_attn_bhsd:32-34、gqa:216-218、mha_sink:183-185）
   全部**显式**声明 GM workspace 并手写真实拷贝——Expert 下 AUTO_CV_SYNC=False，
   虚拟通道被重写后无自动跨核同步插入 → 竞态，故 rev3 禁止虚拟拷贝（§8.6-R12）。

**rev2（Developer AUTO_CV）vs rev3（Expert 显式）每 KV 迭代数据流对比（诚实口径）**：

| 维度 | rev2 Developer 终态（实测 65.05） | rev3 Expert（本设计） |
|------|----------------------------------|----------------------|
| GM workspace 段数/迭代 | 3 段（w1/w2/w3，AUTO 注入，core 索引 ~1.1MB L2 友好） | **3 段（物理必然，同上证据）**——[cid, stage] 双维索引，NUM_STAGES 槽位批内轮转 |
| ws dtype | w1 fp32 / w2 fp16 / w3 fp32（AUTO 决定） | w1 fp32 / w2 dtype / w3 fp32（显式主线，位级锚）；w1 dtype 瘦身 = Stage 3 探针门候选 |
| 跨核同步/迭代 | 3 对 cross-flag（每迭代逐段交接） | 3 对 cross-flag / **每 CROSS_INTERVAL 块**（批量摊薄；interval=1 时同为逐块但 set/wait 位于批内流水深处、非串行交接点） |
| 核内同步 | AUTO_SYNC 逐 op `PipeBarrier<PIPE_ALL>` 全排空/成对串行（codegen 实证） | pipe-pair 细粒度 flag（MTE2/MTE1/M/FIX/V/MTE3 五管并行推进；Form B 为 barrier_all 粗粒度 + 批量 SEM） |
| C↔V 重叠 | 无（逐迭代串行交接：GEMM1→softmax→GEMM2 首尾相接） | **NUM_STAGES 深度重叠**（GEMM1 批(i+1) ∥ softmax 批(i) ∥ GEMM2 批(i-1)） |
| L0 流水 | gemm_v0 内部单槽 | Form A：L0A/L0B/L0C 双槽 ping-pong（mma(i) ∥ FIX 写出(i-1) ∥ MTE1 载入(i+1)）；Form B：gemm_v0 内部（继承） |
| vector 并行度 | 1 AIV 全行 | **2 AIV 各半区**（vector 吞吐 ×2 结构性提升） |
| 结论 | — | 段数持平、**同步与重叠结构性质变**——这是 Expert 化的真实收益来源（"消 GM 段数"不是） |

### 8.3 Cube 核计算流程

```python
# Form A（prefill D≤128）：mma 双槽流水（mha_sink:283-360 同构）
T.wait_flag("MTE1","MTE2", SIG_K_L1)
T.copy(K[bz, j0:j0+valid_n, by, :], k_l1)               # GM→L1 行跨步（运行期 extent）
T.set_flag("MTE2","MTE1", SIG_K_L1)
T.wait_flag("M","MTE1", SIG_L0AB+side)
if i < 2: T.copy(q_l1, l0a_s[side,:,:])                 # Q 批首预载
T.wait_flag("MTE2","MTE1", SIG_K_L1)
T.copy(k_l1, l0b_s[side,:,:], transpose=True)           # L1→L0B NZ 转置载入
T.set_flag("MTE1","MTE2", SIG_K_L1); T.set_flag("MTE1","M", SIG_L0AB+side)
T.wait_flag("MTE1","M", SIG_L0AB+side); T.wait_flag("FIX","M", SIG_L0C+side)
T.mma(l0a_s[side,:,:], l0b_s[side,:,:], l0c_s[side,:,:], init=True)     # GEMM1
T.set_flag("M","MTE1", SIG_L0AB+side); T.set_flag("M","FIX", SIG_L0C+side)
T.wait_flag("M","FIX", SIG_L0C+side)
T.copy(l0c_s[side,:,:], workspace_1[cid, i, :, :])      # L0C→GM（FIX）
T.set_flag("FIX","M", SIG_L0C+side)
# …GEMM2 批对称：v_l1/p_l1 载入 → mma(l0a_o,l0b_o,l0c_o) → w3…

# Form B（decode 全 D + prefill D=256）：gemm_v0 批量（flash_attn_bhsd:90-110 同构）
T.copy(K[bz, j0:j0+valid_n, by, :], k_l1); T.barrier_all()
T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True, kL0Size=gemm1_kL0)
T.barrier_all()
T.copy(acc_s_l0c, workspace_1[cid, i, :, :]); T.barrier_all()
# …GEMM2 批对称：gemm_v0(p_l1, v_l1, acc_o_l0c, init=True) → w3…
```

### 8.4 Vector 核计算流程（两形态共享 emitter；vid 半区 hM 行）

```python
# phase 1（softmax 批，每 i）：
T.copy(m_i, m_i_prev)                                   # 历史 max 备份（reduce 前，时序教训）
T.copy(work_ub, acc_s_half); T.copy(acc_s_half, work_ub)  # r16(qk)（w1 fp32 主线）
T.tile.mul(work_ub, work_ub, sm_scale)                  # ×s（fp32 域）
T.copy(work_ub, acc_s_half); T.copy(acc_s_half, work_ub)  # r16(scores) = golden 栅格
if need_mask:                                           # 编译期变体（语句级 if）
    if idx*bN+bN-1 > T.min(s0+offset_eff, S_kv-1):      # 运行期全可见跳过（#4）
        T.tile.arith_progression(col_pos, j0, 1, bN)
        T.tile.broadcast(buf_2d, col_pos, axis=0)       # [bN]→[hM,bN] 列索引
        T.tile.broadcast(rt_2d, row_thresh, axis=1)     # [hM]→[hM,bN] 行阈值
        T.tile.compare(mask_2d, buf_2d, rt_2d, "LE")    # bit-packed 可见性
        T.tile.select(work_ub, mask_2d, work_ub, -T.infinity("float"),
                      "VSEL_TENSOR_SCALAR_MODE")        # 不可见 → -inf
T.reduce_max(work_ub, m_i, dim=-1); T.tile.max(m_i, m_i, m_i_prev)
T.tile.sub(r_factors[i,:,:], m_i_prev, m_i)             # deferred rescale 差值存储
T.tile.broadcast(buf_2d, m_i, axis=1); T.tile.sub(work_ub, work_ub, buf_2d)
T.tile.exp(work_ub, work_ub)                            # P = exp(s - m_new) ∈ [0,1]
T.reduce_sum(work_ub, sumexp_is[i,:,:], dim=-1)
if single_kv_block:                                     # 预归一化特化（§1.4-8）
    T.tile.add(sumexp, sumexp_is[0,:,:], 1e-30)         # 全掩码行保护同语义
    T.tile.broadcast(buf_2d, sumexp, axis=1)
    T.tile.div(work_ub, work_ub, buf_2d)                # P̂ = P/(sumexp+1e-30)
<SIG_S_HALF 闭环> work_ub → acc_s_half → w2[cid,i,vid半区]

# phase 2（O 累加批，每 i；single_kv_block 时跳过 rescale/折叠）：
T.tile.exp(r_factors[i,:,:], r_factors[i,:,:])          # r_i（与逐迭代 exp 同值）
T.tile.mul(sumexp, sumexp, r_factors[i,:,:])
T.tile.add(sumexp, sumexp, sumexp_is[i,:,:])
T.tile.broadcast(buf_2d, r_factors[i,:,:], axis=1)
T.tile.mul(acc_o, acc_o, buf_2d)                        # 历史 rescale
<w3[cid,i,vid半区] → io_buf → work_ub>
T.tile.add(acc_o, acc_o, work_ub)                       # + PV_i

# 任务尾（通用路径）：
T.tile.add(sumexp, sumexp, 1e-30)                       # 全掩码行保护
T.tile.broadcast(buf_2d, sumexp, axis=1); T.tile.div(acc_o, acc_o, buf_2d)
<SIG_S_HALF 闭环> acc_o → out_half → if rows_vid>0: Output[bz, vid半区行切片, by, :]
```

### 8.5 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,  # 手动 CV 分离（T.Scope）
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,     # 手动核间同步（SEM 0-5）
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,        # 手动核内同步（SIG 矩阵）
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,  # 手动内存规划（alloc_*）
}
```

### 8.6 注意事项

1. **Expert 帧禁带 threads 参数**：`T.Kernel(core_num, is_npu=True)`——带
   threads 会切换到 dev_mode frame（kernel.py:262），与全 False pass_configs
   构成非法混合态且落入 R8 语境（threadIdx.x 绑定 + threads=2 → VidReduction
   激活 → BSHD 4D 三重 bug）。自检 grep：kernel 源码无 `threads=`。
2. **别名禁令（session-1 #2 教训升级）**：V-scope 纯计算 scratch（buf_2d/rt_2d）
   **绝不得**与跨核 copy 目的地（io_buf/work_ub/acc_s_half/out_half）别名复用；
   任何物理复用（如 io_buf phase1/phase2 分时）必须有完整双向 flag ownership
   闭环（flash_attn_optimize §8："仅靠源码顺序或 store 前的 barrier 不足以保护
   下次复用"）。
3. **虚拟通道禁令（R12）**：源码不得出现 `T.copy(L0C buffer → UB buffer)` 或
   `T.copy(UB buffer → L1 buffer)` 裸拷贝——一律经显式 workspace（§8.2 证据）。
   自检 grep：L0C buffer 的 copy 目的地仅为 workspace_*；L1 buffer 的 copy 来源
   仅为 GM 张量/workspace_*。
4. **C/V 同式约束（§7.2-5）**：eff_kv_iters/eff_num_outer/batch_iters 公式、
   task 分解、rows_vid 之外的标量算术必须双侧逐字一致（复制粘贴级），任何单侧
   改动 = flag 配对破坏 = 死锁或静默竞态。
5. **wait_cross_flag 单参**：pipe 参数仅 A5 平台（ascend.py:222-227）；910B
   全部单参调用。
6. **Form B 的 barrier_all 位置**：仅在 gemm_v0 前后与 ws 写出后（段间分隔）；
   不得进入 V-scope 批内循环（V 侧一律 flag 管程）。Form A 全文禁用 barrier_all
   （细粒度 flag 已覆盖；barrier_all = PIPE_ALL 全排空，即 rev2 结构性成本来源）。
7. **NUM_STAGES 偶数约束**：io_buf/L0 双槽轮转要求（mha_sink:112 assert 同款，
   host 侧 assert NUM_STAGES % 2 == 0）。
8. **workspace 槽位索引 = 批内 i（非全局 idx）**：w*[cid, i, …]，i ∈
   [0, batch_iters)——SEM 释放/重取按批周转（mha_sink:313/450/524 同式）；
   跨批复用槽位由 SEM_WS*_V2C/C2V 释放语义保护。

---

## 9. 验证方案

### 9.1 Golden 函数（需求方提供，rev2 §9.1 逐字保留；Stage 2 `test_mha.py` 以此为唯一精度标杆，禁止改写语义；测试时在 **CPU** 上以与输入相同的 dtype 运行——不做 upcast，避免测试 harness 引入 NPU aclnn 依赖）

```python
import torch

def mha(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaleValue: float = -1.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """多头注意力: query [B,S,N,D], key/value [B,S_kv,N,D] → 输出 [B,S,N,D]"""
    B, S, N, D = query.shape
    S_kv = key.shape[1]

    if scaleValue <= 0:
        scaleValue = 1.0 / (D ** 0.5)

    # 转置为 [B, N, S, D]
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    # 缩放点积注意力
    scores = torch.matmul(q, k.transpose(-2, -1)) * scaleValue
    if is_causal:
        i = torch.arange(S, device=scores.device).unsqueeze(-1)
        j = torch.arange(S_kv, device=scores.device).unsqueeze(0)
        causal_mask = j > (i + (S_kv - S))  # 右下角对齐：上三角置 -inf
        scores = scores.masked_fill(causal_mask, float('-inf'))
    # F217: 全 mask 行 (整行 = -inf) 在 softmax 时得 0/0 = NaN，对齐
    # sparse_flash_attention 加显式保护 → 全 mask 行权重置 0。
    scores_max = scores.max(dim=-1, keepdim=True).values
    all_masked = torch.isinf(scores_max) & (scores_max < 0)
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    attn_weights = torch.where(all_masked, torch.zeros_like(attn_weights), attn_weights)
    attn_output = torch.matmul(attn_weights, v)

    # 转回 [B, S, N, D]；transpose 只改 stride, 而输出契约要求 contiguous
    return attn_output.transpose(1, 2).contiguous()
```

算子调用示例（用户侧契约，不变）：

```python
B, S, S_kv, N, D = 2, 128, 128, 8, 64
query = torch.randn(B, S, N, D, dtype=torch.float16, device="npu")
key = torch.randn(B, S_kv, N, D, dtype=torch.float16, device="npu")
value = torch.randn(B, S_kv, N, D, dtype=torch.float16, device="npu")
y = mha(query, key, value, scaleValue=-1.0, is_causal=False)
```

### 9.2 L0 门槛测试计划（tilelang-op-test-design 场景 A 按 rev3 设计重跑）

> **Stage 1 只给 L0**（规则 shape 为主），供 Stage 2 快速精度收敛；L1（功能，含
> 不规则/尾块/质数 shape）/L2（异常）/Boundary（特殊值）完整分层套件**已存在于
> `test_mha.py`**（Developer 阶段场景 B 产出：L1 19/L2 12/Boundary 6，接口契约
> 不变 → 零改动复用，§9.5），Stage 2 直接运行既有套件，无需重新枚举。

**算子类别判定**（test-design §2/§3）：计算类型 = 混合（CV 融合，两 matmul 夹
softmax）；复杂度 = Fusion；数学特征 = matmul+softmax+matmul；综合类别 =
**Fusion（融合算子）**；**workspace 配置 = True**（Expert 显式 workspace ×3 +
vid 二分 + 跨核信号量——与 rev2 的 False 相反，测试聚焦 CV 交互结果与同步正确性，
test-design §2.2 混合类说明）。测试策略：dtype_count=2、attr 组合 =
is_causal{F,T} × scaleValue{auto,显式}、D 配置 3 档全覆盖、快速冒烟 ≤10 用例。

**L0 用例表（十用例逐字继承 rev2 §9.2——接口契约与 host 分派输入不变）**
（tile 配置按 rev3 §5.2：prefill bM=64（D=256→32）、bN=128；输入 = CPU 生成
`randn×U[-1,1]` 确定性种子 → H2D；golden = §9.1 代码 CPU 同 dtype 运行）：

| # | 用例名 | (B,S,S_kv,N,D) | dtype | scaleValue | is_causal | 验证点 |
|---|--------|----------------|-------|-----------|-----------|--------|
| 1 | l0_min_fp16 | (1,64,64,1,64) | float16 | -1.0（auto） | False | 最小冒烟：无掩码变体、单头、N=1 退化 pitch |
| 2 | l0_min_fp16_causal | (1,64,64,1,64) | float16 | -1.0 | True | causal 最小（offset=0 标准下三角）；**掩码链 Expert V-scope 首用哨兵（R15）** |
| 3 | l0_min_bf16_d128 | (1,64,64,1,128) | bfloat16 | -1.0 | False | bf16 路径 + D=128 |
| 4 | l0_min_bf16_d128_causal | (1,64,64,1,128) | bfloat16 | -1.0 | True | bf16 + causal；**单块特化位级精确锚（Developer 版 MERE=0.0）** |
| 5 | l0_typical_fp16 | (2,128,128,8,64) | float16 | -1.0 | False | 用户示例 shape（多 batch 多头，task 分解 + Fixed Core） |
| 6 | l0_typical_fp16_causal | (2,128,128,8,64) | float16 | -1.0 | True | 典型 + causal |
| 7 | l0_typical_bf16 | (2,128,128,8,64) | bfloat16 | -1.0 | False | 典型 bf16 |
| 8 | l0_typical_bf16_causal | (2,128,128,8,64) | bfloat16 | -1.0 | True | 典型 bf16 + causal |
| 9 | l0_kvlong_fp16_causal | (1,128,256,4,128) | float16 | -1.0 | True | S<S_kv（offset=128 右下角对齐）；**L0 唯一多块用例（kv_iters=2 → 批量流水 + deferred rescale 路径）** |
| 10 | l0_d256_fp16_scale | (1,64,64,2,256) | float16 | 0.0625（显式>0） | False | D=256（bM=32 + Form B）+ scaleValue 显式分支 |

**rev3 kernel 路径映射**（供 Stage 2 归因与 coverage_check 衔接）：
- **形态**：Form A（mma 流水）→ #5,6,7,8,9（S>16 ∧ D≤128）；Form B prefill
  （D=256）→ #10；Form B decode（S≤16）→ **L0 无命中**（继承 rev2 格局：L0 表以
  prefill 规则 shape 为主），由 L1 case10/13/14/16 类 + perf_decode_worst 哨兵
  强制覆盖（§5.6 result 行）。
- **批量流水**：多外层批（eff_kv_iters > NUM_STAGES=8）→ L0 无命中（#9 为
  kv_iters=2 单外层批），由哨兵 perf_prefill_worst（eff≈16-32 迭代 → 2-4 外层批）
  + L1 大 S_kv 用例覆盖；批内 batch_iters 余数分支（尾批 <8）→ #9（batch=2）+
  哨兵尾批。
- **单块特化**（S_kv ≤ bN=128）：#1-4,10（kv_iters=1）+ #5-8（S_kv=128=bN →
  kv_iters=1，谓词移动继承 Developer #11）；通用多块路径：#9。
- **掩码 4 变体**：(F,T)→#1,3,5,7,10 / (T,T)→#2,4,6,8,9 / (F,F)/(T,F) 尾钳位
  → L1（需 S_kv%128≠0，现有套件含）。
- **vid 分行/rows_vid 守卫**：全 L0 用例结构性命中（每任务双 vid 半区；整除
  shape 下 rows_vid=half_M 恒正，0 行守卫分支由 L1 奇数 S 系 + R10 探针覆盖）。
- **覆盖矩阵映射（proto 派生）**：D-DTYPE-fp16（#1,2,5,6,9,10）、D-DTYPE-bf16
  （#3,4,7,8）、D-PARAM-is_causal 两值（F:#1,3,5,7,10 / T:#2,4,6,8,9）、
  D-PARAM-scaleValue 两分支（auto:#1-9 / 显式:#10）、D 三配置（64:#1,2,5-8 /
  128:#3,4,9 / 256:#10）。

**L0 通过标准（阻塞口径逐字继承 rev2-C1 用户裁决）**：10/10 用例按 §9.3 阻塞
口径判定通过——混合容差（§9.3.2）全过 **且** `MARE_abs < 10×Threshold`；
MERE/MARE_rel 为参考值（同步计算并打印在 `[PRECISION_*]` 行，不阻塞、不触发
升级仲裁）。`python examples/mha/test_mha.py --level l0` 退出码 0。

**rev3 新增诊断锚点（非阻塞，Stage 2 精度调试用）**：fp32 w1/w3 主线 + 继承算式
下，rev3 输出目标与 **Developer 终态 kernel**（`history_version/
mha_impl_developer_final_6505.py`）在 L0 十用例上**位级一致**（§9.3.3-6 论证）。
Stage 2 可在 L0 运行时并行执行两实现做 bit-diff（调试工具，不进交付判定）：
位级一致 → 语义移植完备，任何 vs-golden 偏差与 Developer 版同源同界；位级不一致
→ **不判 FAIL 但必须根因分析**（同步缺陷/数据流错位嫌疑优先，flash_attn_optimize
§9(a)："手写 flag 同步的 expert kernel 在任意一次运行中出现数值错误，都应视为
同步或数据流缺陷……精度回归应校验每次运行，并通过重复或并发压力测试放大低概率
竞态"）。

**特殊值用例原则（L0 仅声明，不实施；继承 rev2）**：本算子声明输入值域为**有限
实数**，NaN/Inf 输入不在支持域内（proto.yaml 同）。L2/Boundary 若探测特殊值，
必须遵循**位置敏感比较**：用"有限值 + 稀疏特殊值"混合的确定性输入（至少含一个
特殊值与一个有限值），先严格比对 `isnan`/正 inf/负 inf mask 完全一致，再仅对
双方均为有限值的位置套用 §9.3 混合容差；禁止以全 NaN/全 Inf 输入作为唯一特殊值
用例；`allclose(equal_nan=True)` 不能替代显式 mask 检查。

### 9.3 精度标准

#### 9.3.1 生态算子精度标准（需求方指定的验收标准，MERE/MARE；**用户裁决 2026-09-18，rev2-C1，不可更改，逐字继承**）

误差指标（对全部输出元素，fp32 域计算）：

```
MERE（平均相对误差）= mean( |actual - golden| / (|golden| + 1e-7) )
MARE_rel（最大相对误差）= max( |actual - golden| / (|golden| + 1e-7) )
MARE_abs（最大绝对误差）= max( |actual - golden| )
```

**阻塞判定口径（需求方用户已在 Primary 上下文裁决，2026-09-18，不可更改）**：

```
阻塞判定 = §9.3.2 混合容差全过  且  MARE_abs < 10 × Threshold
MERE 与 MARE_rel 为参考值：同步计算、打印在 [PRECISION_*] 行，不阻塞、
不触发升级仲裁（rev1 的 [MARE_REL_ESCALATION] 升级流程随本裁决废止）。
```

| dtype | Threshold | MARE_abs 阻塞线（10×Threshold） | MERE 参考线（Threshold） | MARE_rel 参考线（10×Threshold） |
|-------|-----------|--------------------------------|--------------------------|--------------------------------|
| float16 | 2⁻¹⁰ ≈ 9.77e-4 | **< 9.77e-3（阻塞）** | ~9.77e-4（参考） | ~9.77e-3（参考） |
| bfloat16 | 2⁻⁷ ≈ 7.81e-3 | **< 7.81e-2（阻塞）** | ~7.81e-3（参考） | ~7.81e-2（参考） |

（阈值结构与本仓生态先例 `examples/cann-bench/group_norm/example_group_norm.py
:487-491` 逐字一致；rev2 §9.3.1 的数学不可达性论证——MERE 被近零 golden 元素的
1e-7 clamp 分母病态主导、理想下界实测超门限——随裁决一并归档，Stage 2/3 不得再
以任何形式将 MERE/MARE_rel 计入阻塞。）

#### 9.3.2 混合容差标准（用例级阻塞判定，harness `check_precision` 实现；**rev2 原样保留，用户裁决口径的组成部分**）

> 逐元素 `|actual-golden| ≤ atol + rtol·|golden|`，整体判定
> `matched_ratio ≥ required_matched_ratio` **且** `max_abs_error ≤ max_abs_error_limit`。
> 阈值仅按 dtype（与算子类别无关）；inf/nan 位置做结构比对不计入数值容差
> （完整定义见 tilelang-op-test-design/references/precision-standard.md）。
> §4 声明的每个 dtype 均有对应行；本算子无整型 dtype。

| dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|-------|------|------|---------------------|------------------------|
| float16 | 2⁻¹⁴ (6.10e-5) | 2⁻⁹ (1.95e-3) | 1e-1 | 0.99 |
| bfloat16 | 2⁻¹⁰ (9.77e-4) | 2⁻⁶ (1.56e-2) | 1e0 | 0.99 |

#### 9.3.3 精度对齐设计（golden 栅格对齐；rev3 继承 + Expert 落点论证）

1. **scores 双舍入镜像（rev2-A2 继承）**（§1.4 步骤 3）：golden 的 scores 经历
   `round16(round16(qk)·s)` 两次半精度舍入，kernel 逐一镜像（r16(qk) → ×s(fp32)
   → r16(scores)）。Expert 落点：w1=fp32（FIX 写出无 cast，copy_l0c_to_gm
   T1=T2=float），V 侧 work_ub 上显式 round-trip ×2（acc_s_half 中转）——链序与
   Developer 终态 mha.py:304-308 逐字同构。2 的幂 scale 位级等价、D=128 ≤1ulp。
2. **P 以输入 dtype 进 GEMM2（继承）**：w2 dtype = fp16/bf16（P 原生栅格，与
   golden 半精度 attn_weights 同栅格；与 Developer acc_s_half→acc_s_l1 等价）。
3. **fp32 累加/归一化（继承，红线）**：GEMM fp32 累加（L0C）+ w3 fp32 +
   fp32 online 状态（r_factors/sumexp_is/sumexp/acc_o）+ fp32 终归一——精度
   不低于 golden 的对应 fp32 内部累加路径。**w1/w3 dtype 瘦身（fp16 化）会破坏
   本条**，仅允许在 Stage 3 探针门（单块位级 + 多块阻塞口径余量复核）通过后启用。
4. **单 KV 块预归一化特化（rev2-A3 继承）**：kv_iters==1 时 P̂ 归一化后再舍入，
   与 golden 一次性 softmax 结构同构——位级可追踪锚点（Developer 版 case4 bf16
   位级精确、单块用例 MERE≤3e-6）；rev3 落点 = phase 1 内 div（§8.4），算式
   次序逐字保持。
5. **阻塞口径下的余量（继承 rev2-C1 实测）**：多块路径残差来源为 P 归一化位置差
   与最终输出舍入，量级 ~2⁻¹¹ 相对 / ~1e-4 绝对——Developer 终态实测 MARE_abs
   余量 ≥80×（L0）至 320×（哨兵 3.05e-5 vs 门限 9.77e-3）；rev3 在位级一致
   锚点成立时继承同一余量。
6. **rev3 位级一致锚点论证（新增）**：fp32 w1/w3 主线下，rev3 与 Developer 终态
   的输出**目标位级一致**，逐因子：
   - **GEMM 累加序**：Form A mma K=D≤128 单次调用（内部 K-loop 16 步序）与
     gemm_v0 kL0Size=128 单步为同一 cube 数据通路；Form B 逐字同调用（kL0Size
     三档继承）→ fp32 累加位级同。GEMM2 同理（K=bN=128 单段）。
   - **V 侧算式链**：双舍入/掩码/reduce_max/max/sub/exp/reduce_sum 的 op 序列
     与 Developer 终态逐字同构（§8.4），tile op 为确定性 SIMD → 位级同。
   - **deferred rescale**：`r = exp(m_prev - m_new)` 的 fp32 sub 与 exp 均为
     确定性单值函数，phase 2 计算与逐迭代计算输入相同 → 输出位级同；sumexp
     折叠 `sumexp*r + sumexp_is[i]` 与 acc_o 的 `acc_o*r + PV_i` 运算次序与
     Developer 逐迭代版一致（按块序 i 升序）。
   - **vid 二分**：行独立（无跨行归约/状态），half_M 行分拆不改每行算式 →
     位级同。
   - **ws 往返**：fp32→GM→fp32 无舍入（w2 dtype 往返 = Developer acc_s_half
     同栅格）。
   该锚点为**诊断性目标**（非阻塞判定）；已知允许偏差源 = 编译器 op 融合/排序
   差异（Expert 无 AUTO pass 重排，预期极小）。失配处理见 §9.2 诊断锚点节。

### 9.4 性能可行性哨兵（强制执行，不可因 large/L1 跳过；门限结构逐字继承 rev2-C4）

**门限结构**（区分"算子 kernel 性能"与"测试 harness golden 成本"）：

1. **kernel 路径计时门（阻塞，可行性判定唯一计时依据）**：计时范围 =
   H2D → kernel（含该 shape 首次 JIT 编译）→ D2H + synchronize；每条哨兵
   ≤ **120s**（Developer 终态实测 6.1-7.2s，≥16× 余量；warm kernel 另记录作
   Stage 3 基线）。
2. **精度判定（阻塞）**：§9.3 阻塞口径（混合容差 + MARE_abs）；MERE/MARE_rel
   打印为参考值。
3. **golden 侧降本规定（保障哨兵可执行，强制执行；继承 rev2）**：分块 golden
   （按 (b,n) 独立分块，harness `golden_mha_chunked` 已有实现）；缓存复用（同
   (shape, seed, dtype, attr) 只算一次，持久化跨 attempt/轮次复用）；确定性
   抽样比对（仅 prefill 哨兵允许：固定种子 ≥16 个 (b,n) 块全元素 + 其余按确定
   性步长每块 ≥1024 元素，覆盖 causal 对角区与全可见区，方案写入测试代码）；
   可选 fp32-upcast golden 仅用于参考值口径标注。
4. **case 级墙钟参考上限 1800s（非阻塞）**：防 harness 挂死；超时只判 harness
   缺陷，不判算子可行性失败。

| 用例名 | (B,S,S_kv,N,D) | dtype/属性 | rev3 覆盖路径 | kernel 路径门（阻塞） | Developer 终态实测 | 选择理由 |
|--------|----------------|-----------|--------------|----------------------|--------------------|----------|
| perf_prefill_worst | (8,2048,4096,32,128) | fp16, causal | Form A P2：多外层批（eff≈16-32 迭代/任务 → 2-4 批）+ 最大 KV 重读 + 15.7MB ws 足迹 | ≤120s | 6.1-7.2s（final_report §S2-4） | GM 流量 + ws 流量最大规模（§5.5 核算） |
| perf_decode_worst | (64,1,4096,32,128) | fp16, causal | Form B D1（bN=512+kL0=32）：valid_m=1 → rows_vid vid0=1/vid1=0 守卫路径 + trim + 掩码全交互 | ≤120s | 同上门内 | S=1 极端尾块 × 2048 任务 × vid 分行守卫最坏情形 |
| perf_d256_bf16 | (2,1024,1024,16,256) | bf16, 非 causal | Form B P3（bM=32+kL0=128）：D=256 最大 L0/L1 占用 + bf16 GEMM | ≤120s | 同上门内 | 预算最紧配置 + bf16 路径 |
| perf_max_grid | (128,128,128,64,64) | fp16, causal | Form A P1：任务网格上限（16384 任务 → core_num 循环 ≈683 任务/核）+ single_kv_block（S_kv=128=bN） | ≤120s | 同上门内 | 任务调度/Fixed Core 最坏；支持域边界 B=128、N=64 |

哨兵判定：kernel 路径门超时 = **可行性失败** → 触发 Stage 1 回退（重选
block/NUM_STAGES/形态分派）；精度阻塞口径失败 → Stage 2 精度流程处理。causal
裁剪后算力冗余已计入（§5.5）；若 Expert 首测逼近门禁，按 §6.3 降级旋钮顺序
调整后复检。

### 9.5 测试资产复用声明（rev3 契约）

`test_mha.py`（844 行，Developer 阶段产出）依赖的**全部对外契约在 rev3 保持
不变**：`from mha import mha`（host 签名/断言/out_idx 语义逐字继承 §3.3）、
golden §9.1 原样、阻塞口径判定逻辑（rev2-C1 已同步）、L0/L1/L2/Boundary/perf
五层套件与 `--level` 接口。→ **测试资产零改动复用**；Stage 2 验收路径：
`--level l0`（精度收敛 + 位级诊断锚）→ `--level l1` → `coverage_check.py
--proto`（覆盖门禁）→ `--level all`（含哨兵）→ cann-bench 复评（§12）。
R10/R15 前置探针为 Stage 2 内部新增调试工件（探针脚本置于临时目录，不进交付物、
不改 test_mha.py）。

---

## 10. 风险点与注意事项

### 10.1 已知约束与风险

| # | 风险 | 等级 | 说明与缓解 |
|---|------|------|-----------|
| R1-R3 | 掩码链注册 / 4D 行跨步 lowering / 循环内运行期 valid_n（rev2 已关闭项） | **已关闭（继承）** | rev2 关闭证据在 threads=1 Developer 语境取得；模板层（copy_gm_to_l1/gm_to_ub/ub_to_gm、operation_config.h PIPE_V 注册）与模式无关，rev3 沿用同一物理模板——但 **× Expert 帧组合**由 R10 探针重新覆盖后方可视为关闭 |
| R4 | D=256 任务数翻倍 → KV 重读 ×2 | 低（性能） | 继承 rev2：实测吸收（perf_d256_bf16 哨兵门内）；Expert 下 vid 二分与批量流水进一步摊薄 |
| R5 | MERE/MARE_rel 口径病态 | **已解决（用户裁决，继承）** | 阻塞口径 = 混合容差 + MARE_abs（§9.3.1）；Stage 2/3 不得以任何形式将 MERE/MARE_rel 计入阻塞 |
| R6 | causal 无裁剪冗余算力 | **已消除（继承 #1）** | 运行期裁剪为 v1 标配（非候选）；Expert 下 C/V 同式复制（§7.2-5） |
| R7 | 每 shape 组合一次 JIT 编译 | 低 | 继承 rev2；tilelang 缓存 + 首次编译计入 §9.4 kernel 路径门。**Expert 增量**：form_a/form_b 双形态各编译一次（同 shape 只命中其一，无额外成本） |
| R8 | AscendVidReduction × threads=2 × BSHD 4D 结构性 bug | **Expert 形态不触发（rev3 复检关闭；Developer 语境禁令保留）** | 复检证据链 §2.2-c（五点：触发键 threadIdx.x 不匹配 / threads_cnt_≠2 全短路 no-op / 手动 vid 二分无自动改写环节 / WorkspaceReduction vid 折半越靶 / 三参考 CI 先例）。**护栏**：rev3 禁带 threads 参数（§8.6-1）——若回退 Developer 模式，threads=2 禁令与复现路径（rev2 §10.1-R8 原文、debug_log.md deviation#1 探针族）继续有效 |
| R9 | threads=1 × T.Pipelined codegen bug | **Expert 形态不适用（rev3 复检关闭；T.Pipelined 全文禁用不变）** | 复检证据链 §2.2-d（两前提均不成立：MEMORY_PLANNING=False 无自动双缓冲注入且 address_map 恒全量注册；循环形态 = T.serial + 手动 ping-pong，参考实现同形态 CI）。护栏：§6.1 明确不采用 T.Pipelined |
| R10 | **BSHD 4D 行跨步 × Expert vid 偏移 × 运行期 valid extent × ZN/NZ annotation 组合 = 仓内首例**（rev3 新增） | **高（Stage 2 第一道探针门）** | 参考实现均 BHSD（flash_attn_bhsd/gqa/mha_sink）或整除场景；BSHD Expert 仅有 sparse_attention.py 的 vid-workspace 切片旁证（且为 Developer pass_configs）。**缓解**：Stage 2 强制第一步 = 最小复现探针族（rev2 verify_bshd_slice/verify_t1a/verify_kv_tail 移植 Expert 脚手架：整除位级 + S=1/30/64/90 运行期 extent + vid 半区写回位级 + annotate_layout 开/关两态 + rows_vid=0 守卫路径），全过再进 L0；应急预案 = 去 annotation（性能降级不换语义）/ 按行 1D 循环搬运（§5.4）。判定纪律继承 rev2：写回形式判定一律以位级探针为准，禁止"对称溢出巧合"式误判 |
| R11 | **flag 配对失衡 → 死锁（hang）或静默竞态（非确定性损坏）**（rev3 新增，Expert 固有风险） | 高 | 触发场景：C/V 两侧 eff_*/batch_iters 公式不一致（§7.2-5）、init/destroy 闭环缺失（§7.2-4）、运行期 if 内单侧 set/wait、跨任务 token 泄漏。**缓解**：① 公式复制粘贴级同式 + Stage 2 静态 diff 自检；② init/destroy 逐字继承 mha_sink 模板；③ 运行期守卫（掩码跳过/rows_vid）内 flag 成对同进出或纯 V 无 flag（§7.2-5 已论证）；④ 哨兵/L0 首跑加超时看门狗（hang → 立即归因 flag 矩阵）；⑤ flash_attn_optimize §9(a) 纪律：任何一次运行数值错误 = 同步/数据流缺陷，重复+并发压力测试放大低概率竞态，warmup 不作为忽略首错的借口 |
| R12 | **虚拟通道误用**：写出 `T.copy(L0C→UB)`/`T.copy(UB→L1)` 裸拷贝（rev3 新增） | 中 | 后果：AscendWorkspaceReduction 自动注入 workspace 但 AUTO_CV_SYNC=False → 无跨核同步 → 竞态（静默损坏）。**缓解**：§8.6-3 禁令 + Stage 2 自检 grep（L0C buffer copy 目的地仅 workspace_*；L1 buffer copy 来源仅 GM/workspace）+ 源码评审项 |
| R13 | **顶格预算条目**（⚠ 标记）：P2/P1 l0b 64KB、Form B 内部 L0B 64KB（kL0 三档）、D1 L1 276KB（rev3 新增） | 中 | Expert 无 liveness 合并，naive 和≈实配 + 对齐 padding 可能小额附加。**缓解**：§4.5 判定制度——全部 ⚠ 条目 Stage 2 实编 GetWithOffset/编译探针读数复核（session-2 制度继承）；Form B 条目另有 Developer 终态同 gemm_v0 调用实测先例（session-2 #10/#11/#13）；超限时降级旋钮 = P1 分池→共池切片 / Form A→B（D=128 prefill）/ NUM_STAGES 降档 |
| R14 | **workspace GM 足迹 ≈11-16MB/编译实例**（rev3 新增；L2 驻留不确定 → 带宽压力） | 中（性能，非正确性） | [cid,stage] 私有索引核间零竞争；参考实现同量级足迹已验证性能（mha_sink ≈19MB @51.72 TFlops）。**缓解**：Stage 2/3 msprof 归因（w1/w3 流量段占比）；降级旋钮 = NUM_STAGES 8→4（足迹减半）→ w1 dtype 瘦身（探针门 §10.3-①）；torch 缓存分配器跨调用复用（无逐次 malloc） |
| R15 | **掩码链 ops（arith_progression/compare/select）Expert V-scope 首用**（rev3 新增） | 低-中 | ops 注册 PIPE_V（operation_config.h:184-237）与模式无关，lowering 同一；Developer 语境位级验证在先（本算子 #3）。风险点 = bit-packed compare 输出在 [8,512]（D1）/新半区尺寸的形态。**缓解**：L0 case2 为首用哨兵（§9.2）+ Stage 2 隔离探针（掩码链单测：4 变体 × 3 尺寸）；应急回退 = select 行循环形式（rev1 时代写法，性能降级不换语义） |
| R16 | **Expert 版整体劣于 Developer 终态（65.05）**（rev3 新增，立项级风险） | 中 | Expert 化收益为结构性预期（参考实现实证于 BHSD D=128 场景），本算子 BSHD 全谱 + 六配置类的移植效果存在不确定性；Form B（decode/D=256）的 barrier_all 粗粒度同步收益有限。**缓解/退路**：§12 回退条款（备份就位）+ 分阶段验收（Stage 2 精度收敛 → Stage 3 逐 case msprof 归因，先攻 case7/8/9 主差距）+ 降级旋钮链（§6.3） |

### 10.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| Expert kernel 带 threads 参数 | 照搬 Developer 写法 `T.Kernel(n, threads=1/2, ...)` | 切入 dev_mode frame（kernel.py:262）→ 与全 False pass_configs 构成非法混合态；threads=2 时落入 R8 语境 | `T.Kernel(core_num, is_npu=True) as (cid, vid)`，自检 grep 无 `threads=`（§8.6-1） |
| 虚拟通道裸拷贝 | 照搬 Developer 的 `T.copy(acc_s_l0c, acc_s_ub)` / `T.copy(acc_s_half, acc_s_l1)` 片上直连写法 | 框架重写注入 workspace 但无自动跨核同步 → 静默竞态 | 一律显式 workspace 三段（§8.6-3/R12） |
| C/V 两侧 eff_*/batch_iters 公式漂移 | 单侧改动裁剪公式/常量 | flag 配对失衡 → 死锁或静默竞态 | 复制粘贴级同式 + 静态 diff 自检（§7.2-5/R11） |
| init/destroy 闭环缺失 | 省略作用域首尾的预置 set / 对称 wait | 首批死锁或跨任务 token 泄漏 | 逐字继承 mha_sink 模板（§7.2-4） |
| scratch 与跨核 copy 目的地别名 | buf_2d/rt_2d 复用到 io_buf/work_ub/acc_s_half | AUTO 时代已实测 corruption（session-1 #2）；Expert 下为 flag ownership 破坏 | §8.6-2 别名禁令；复用必须双向 flag 闭环 |
| Form A 循环体内 barrier_all | 从 Form B/旧示例复制 barrier 习惯 | PIPE_ALL 全排空 = rev2 结构性成本复活，流水收益归零 | §8.6-6；Form A 仅 pipe-pair flag |
| wait_cross_flag 带 pipe 参数 | 照 A5 平台示例书写 | 910B 不支持 → 编译/运行错误 | 单参调用（ascend.py:222-227，§8.6-5） |
| NUM_STAGES 奇数 / ws 槽位用全局 idx | 参数改动或索引笔误 | 双槽轮转错位 → 数据覆盖 | host assert 偶数（mha_sink:112 同款）；槽位 = 批内 i（§8.6-7/8） |
| m_i_prev 备份时序错误 | `T.copy(m_i, m_i_prev)` 放在 reduce_max 之后 | 重标定因子恒 1，累加错误 | §8.4 固定顺序（reduce 前备份，rev2 §10.2 教训继承） |
| UB/L0 溢出 | 擅自扩 tile/加 buffer 不重算预算 | 编译失败或 address_map ICHECK | 改动前重算 §4.5 表；⚠ 条目实编读数（R13） |
| 尾列污染 softmax 分母 | S_kv 非整除且漏掉钳位掩码 | 精度系统性偏差 | §5.4 三重保障；need_bound_mask 编译期恒生效 |
| 全掩码行 0/0=NaN | S>S_kv+causal（域外）或未来域扩展 | 输出 NaN ≠ golden 0 | sumexp+1e-30 安全除法（两处落点，§8.4） |
| vid1 零行写回 | rows_vid=0 时仍执行 copy | 0-extent copy 未定义行为 | 运行期 if 守卫（§5.4 表 Output 行） |
| select SPR 模式 >64 元素静默错误 / int32 索引 compare 退化 EQ | 误用 VSEL_CMPMASK_SPR / int32 col_pos | 索引错位 / 静默错误 | 固定 VSEL_TENSOR_SCALAR_MODE + fp32 col_pos（rev2 教训继承） |
| host 侧"顺手" contiguous/transpose/mask 预构建 | 照搬 mha_sink/gqa 的 host build_mask 或 BHSD 转换 | 违反 §4.8（aclnn/物理拷贝），cann-bench 环境失败 | host 零张量变换（§4.8 审计锁定）；掩码链 kernel 内计算（§3.5.2） |
| TVMScript parse-time if 块内分配、块外使用 | Form 分支内 alloc、共享段引用 | "Undefined variable" 编译错误 | Python 级双 prim_func 构建 + 共享 emitter（§6.1 注记，rev2-A1 教训） |

### 10.3 特殊场景处理

- **S=1/2 decode（vid 尾行矩阵）**：valid_m=1/2 → vid0 rows_vid=1/2、vid1
  rows_vid=0（守卫跳过）；整 bM=16 行缓冲计算（垃圾行有限、行独立、写回裁剪）；
  causal 首行可见列 = [0, offset]=[0,S_kv-1] 全可见（offset=S_kv-1≥0），无全
  掩码行。single_kv_block（S_kv≤bN=512/256/128）命中时走预归一化特化。
- **全零输入退化**（需求实测场景 [0,0]）：scores=0 → 掩码后可见列均匀 softmax
  → 输出 = 可见 V 行均值（V 亦全零 → 输出全零）；sumexp ≥ 1 无除零。golden
  同路径同结果 ✅（L1 用例，现有套件 case20 类）。
- **causal 且 S=S_kv**：offset=0 退化标准下三角（L0 case2/6/8）。
- **causal 且 S<S_kv**：offset>0 右下角对齐（L0 case9）。
- **N=1**：行 pitch = D，行跨步退化连续搬运 ✅（L0 case1-4）。
- **block_num ≤ core_num**（小 shape）：core_num = block_num，每核 1 任务
  （Fixed Core 退化安全，继承 Developer #5 degenerate-safe 注释）。
- **Stage 3 性能优化候选清单**（正确性无关，勿在 v1 引入；每项落地后须复跑
  §9.4 kernel 路径门 + L0 阻塞口径 + 位级诊断锚复核）：
  ① **workspace dtype 瘦身**：w1 fp32→dtype（隐式首舍入 = fixpipe cast，需
    RNE 等价探针 + 单块位级锚复核 + fp16/bf16 双 dtype cast 支持验证）、
    w3 fp32→fp16（需多块 MARE_abs 余量量化 ≥10× 方可启用）——w1/w3 流量各
    -50%（R14 缓解主路径）；
  ② **unit_flag mma→fixpipe 硬件融合**：`T.mma(unit_flag=0b11)` +
    `T.copy(L0C→GM, unit_flag=0b11)` 跨 L0C ping-pong 重叠（customize.py:115
    docstring / copy_op.py:283-291）；
  ③ **k/v L1 双缓冲**：L1 余量充足（P2 96/511.75KB），MTE2 预取与 MTE1 消费
    重叠（先 msprof 归因确认 MTE2 瓶颈）；
  ④ **NUM_STAGES/CROSS_INTERVAL 扫描**：8→{4,12,16} × 1→{2,4}（gqa iter3
    方法论：bench_mark 实证选优）；
  ⑤ **Form B→A 扩展**：D=256 prefill / decode 的 mma 化（手动 K/N-split，
    §5.2 分派谓词放宽）——需 L0 分池预算复核（R13）；
  ⑥ **axpy 融合**：`(work_ub - m)*s` 双 op 融合（mha_sink:466 先例）——破坏
    双舍入链序，**必须先过 §9.3.3-1 栅格等价论证 + 位级锚复核**（rev2-⑤ 警示
    继承）；
  ⑦ **io_buf 加深/out_half 与 acc_s_half 合并**（D==bN 配置）：UB 余量换取
    MTE3 管程松弛（ownership 闭环重证）；
  ⑧ **decode 专用重排**（heads-as-M / split-KV）：session-2 证据性放弃结论
    继承（final_report §S2-5），不在 rev3 重启。

---

## 11. 交付清单

### 11.1 目录结构

```
examples/mha/
├── mha.py            # 纯 kernel（@tilelang.jit mha_kernel 双形态工厂 + host 包装 def mha(...)），可 import，无 golden/测试/__main__
├── test_mha.py       # from mha import mha + §9.1 golden + L0/L1/L2/Boundary/哨兵全套件 + main(--level)（零改动复用）
├── proto.yaml        # 算子接口规格（dtype/attr），供覆盖门禁派生应覆盖维度（rev3 复核无变化）
├── DESIGN.md         # 本设计文档（rev3）
├── debug_log.md      # Stage 2 证据记录（rev3 attempt 续写）
├── perf_tuning/      # Stage 3 调优证据（继承 + rev3 续写）
└── history_version/  # 回退备份（design_rev1/2、proto_rev1/2、Developer 终态实现与测试、各轮快照）
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成（本文件，rev3；rev2 已备份 history_version/design_rev2.md） | Expert 设计（含 §0 修订记录、L0 计划、用户裁决口径、性能目标与回退条款） |
| `proto.yaml` | ✅ rev3 复核**无变化**（接口契约不变） | dtype 全集 = §9.3.2 表（float16, bfloat16）；attrs = scaleValue(Float) + is_causal(Bool)（§11.5） |
| `mha.py` | ⬜ Stage 2 rev3 重写（**禁止修改现文件——先备份已就位**：history_version/mha_impl_developer_final_6505.py = 65.05 终态快照） | 双形态 Expert kernel + host 包装（签名逐字不变） |
| `test_mha.py` | ✅ **零改动复用**（§9.5 契约论证） | golden + 全套件 + 双口径 harness + 阻塞判定；Stage 2 只运行不修改 |
| `history_version/` | ✅ 回退备份就位 | design_rev1/rev2.md、proto_rev1/rev2.yaml、mha_impl_developer_final_6505.py、test_mha_developer_final.py、mha_impl_s2/s3 各轮快照 |
| `perf_tuning/` | ✅ 继承证据链 | final_report.md（65.05 锚点 + 硬限知识）、optimization_log.md（15 轮门禁）、perf_log.md、msprof_* 原始计时 |

### 11.3 命名规范

- 目录名: `mha`（snake_case）✅ 已存在
- kernel 文件: `mha.py`；测试文件: `test_mha.py`（顶部 `from mha import mha`）
- kernel JIT 工厂: `mha_kernel`（内部 Python 级 form_a/form_b 双 prim_func 构建）；
  host 包装: `mha(query, key, value, scaleValue=-1.0, is_causal=False)`

### 11.4 实现顺序（Stage 2 验收路径）

1. ✅ 设计文档（DESIGN.md rev3）+ proto.yaml（复核无变化）+ L0 门槛测试计划
   （§9.2，十用例继承、路径映射更新、诊断锚点新增）
2. ⬜ **前置探针（R10/R15，强制第一步，先于任何 L0 运行）**：
   a. Expert BSHD 最小复现族：整除位级 / S=1/30/64/90 运行期 extent / vid 半区
      写回位级 / rows_vid=0 守卫 / annotate_layout 开关两态（临时目录，不进交付）；
   b. 掩码链 V-scope 隔离探针（4 变体 × half_M∈{8,16,32} × bN∈{128,256,512}）；
   c. Form B 先行（gemm_v0 低风险，命中 L0 #10 + decode 语义）→ Form A
      （P2 顶格 L0 池，命中 #5-9）。
3. ⬜ kernel 实现（`mha.py` rev3 重写；现文件不动，备份已就位）+ 位级诊断锚
   bit-diff 工具（vs Developer 终态，非阻塞）
4. ⬜ L0 门槛（`--level l0` 10/10 阻塞口径 + 诊断锚分析）
5. ⬜ L1（`--level l1` 19/19）+ `coverage_check.py test_mha.py --proto proto.yaml`
   覆盖门禁（全 PASS/N-A）
6. ⬜ 全量（`--level all`：L0+L1 阻塞、L2/Boundary 记录、哨兵 4/4 kernel 路径
   门 + R11 压力复跑：重复/并发放大低概率竞态）
7. ⬜ cann-bench 复评（§12 验收方式）→ Stage 3 按 §10.3 候选清单调优

### 11.5 算子 proto.yaml（覆盖门禁用；rev3 复核 = 无变化）

接口契约不变（§9.5）→ proto.yaml 内容与 rev2 完全一致：dtype 全集取自 §9.3.2
精度表（float16、bfloat16——§4.1 仅给代表性 dtype，不作全集来源）；attrs 取自
§1.3/§4.1（scaleValue: Float, default -1.0；is_causal: Bool, default false）——
两者均影响计算路径（scale 解析分支 / 掩码 4 变体），供 `coverage_check.py
--proto` 派生 D-DTYPE-*/D-PARAM-* 覆盖维度。与 §9.3.2/§4 一致性已自检 ✅
（`examples/mha/proto.yaml` 现存文件与 history_version/proto_rev2.yaml md5 相同，
无需重写）。

```yaml
operator:
  name: mha
  category: Fusion
  formula: |
    y = softmax(Q @ K^T * s) @ V ;  s = scaleValue if scaleValue > 0 else 1/sqrt(D)
    is_causal: scores[..., i, j] = -inf for j > i + (S_kv - S)   # 右下角对齐
    all-masked row (仅域外 S>S_kv 可达): softmax → 0（golden all_masked 保护对齐）
  attrs:
    - name: scaleValue
      type: Float
      default: -1.0
      required: false
    - name: is_causal
      type: Bool
      default: false
      required: false
  inputs:
    - name: query    # [B, S, N, D]
      dtype: [float16, bfloat16]
    - name: key      # [B, S_kv, N, D]
      dtype: [float16, bfloat16]
    - name: value    # [B, S_kv, N, D]
      dtype: [float16, bfloat16]
  outputs:
    - name: output   # [B, S, N, D]，与 query 同 dtype，contiguous
      dtype: [float16, bfloat16]
  schema: mha(Tensor query, Tensor key, Tensor value, float scaleValue=-1.0, bool is_causal=False) -> Tensor output
```

---

## 12. 性能目标（rev3 重写：Expert 化立项目标 + 回退条款）

- **目标类型**: `baseline_compare` —— 对齐 cann-bench 评测（`tasks/level4/mha`，
  hardware=910b2，baseline = CANN `FusedInferAttentionScore`，
  `tasks/metadata/910b2.json` baseline_perf_us）
- **总目标（用户指定）**: cann-bench **平均加速比逼近 1.0x**（当前 Developer
  终态 0.55x）
- **里程碑（Stage 3 分阶段验收）**:
  - M1：平均加速比 ≥ **0.75x**，且 case7/8/9（大 prefill causal）各自 ≥ **0.4x**
    （当前 0.17-0.19x——Expert 结构性收益的主攻方向）；
  - M2：平均加速比 ≥ **0.9x**，case7/8/9 ≥ 0.6x；
  - 拉伸：平均 ≈1.0x。**decode 类不回退硬约束**：case10 ≥0.85x、case14 ≥0.79x、
    case20 ≥0.82x、case13 ≥0.69x、case16 ≥0.52x（逐 case 不低于 Developer 终态
    值，噪声带 3% 内视为持平）。
- **锚点数据（Developer 终态，cann_final_eval_20260919_154250，Stage 3 对照基线）**:
  - 综合得分 **65.05/100**（编译 20/20、精度 30/30、性能 15.05/50）；平均加速比
    **0.55x**；通过率 20/20。
  - per-case 加速比终态（case: speedup）：1:0.64 / 2:0.30 / 3:0.37 / 4:0.60 /
    5:0.35 / 6:0.45 / **7:0.19 / 8:0.19 / 9:0.17** / 10:0.85 / 11:0.77 /
    12:0.69 / 13:0.69 / 14:0.79 / 15:0.59 / 16:0.52 / 17:0.66 / 18:0.70 /
    19:0.68 / 20:0.82（final_report §S2-1 权威表；提升幅度最大 case11 1.72x、
    case7/8 1.43x 为会话 2 口径）。
  - 本地 msprof 口径与 cann-bench 互验一致（差 <3%，final_report §7 计时口径
    声明继承：msprof Task Duration 中位数为唯一性能依据，禁止 Python/Torch 计时）。
- **验收方式**: 每个优化项 `msprof op` kernel 级计时验证 + 精度复验
  （`test_mha.py --level l0/l1` + 位级诊断锚复核）不退化；阶段性 cann-bench
  复评（`run_evaluation.sh --task-dir tasks/level4/mha --operator MHA`）确认
  加速比/综合分提升；噪声阈值 3%。
- **测试 shape**: §9.4 四条哨兵 + cann-bench 20 cases 代表子集（case1 小 prefill /
  case7 大 prefill causal / case10 decode / case14 decode 大 B / case20 全零，
  继承 rev2 §12）。
- **中止条件**: ① Stage 3 迭代达上限（10 轮/会话）；② 连续 3 次优化无提升
  （<3%）；③ 达到用户目标（由 orchestrator 与用户确认）。
- **回退条款（立项级，用户决策附带）**:
  - **触发 A（Stage 2 精度无法收敛）**：L0 阻塞口径在前置探针通过的前提下仍
    无法 10/10（attempt 上限内），且位级诊断锚 + 分段验证定位到设计层缺陷 →
    返回 `[DESIGN_ERROR]` 由 orchestrator 决策（修订 rev4 或触发 B）；
  - **触发 B（Stage 3 实测劣于 Developer 终态）**：Expert 版 cann-bench 综合分
    < **65.05** 或平均加速比 < **0.55x**（调优预算耗尽后）→ **回滚 Developer
    终态**：`history_version/mha_impl_developer_final_6505.py` → `mha.py`、
    `history_version/design_rev2.md` → `DESIGN.md`（备份已就位，md5 已核验），
    并向 orchestrator 报告回滚与证据；
  - 回滚后 test_mha.py 仍零改动可用（契约同一）。
- **硬约束提醒（沿用并更新）**: R8/R9 在 Expert 形态不触发但 Developer 语境
  禁令保留（回滚/回退场景立即生效）；R10-R16 探针门与自检 grep 项（无
  `threads=`、无虚拟通道拷贝、C/V 同式 diff）为 Stage 2/3 例行检查项；
  §10.3 候选清单为 Stage 3 唯一优化入口（每项过精度门 + 哨兵门）。

---

## 附：rev3 质量自检结论（quality-checklist.md 逐项）

- **第 0 项 Host 合规**：✅ §4.8 审计五项全 allow（metadata 读取/纯 Python/设备
  属性查询/dtype 字符串化/kernel 调用直返；零 contiguous/padding/cat/to(dtype)/
  切片+reshape；不采用参考实现的 host mask 预构建）。
- 必须项：1 ✅（§2.1 Expert 明确 + §2.2 源码论证）；2 ✅（§3.2 逐 API 函数名+
  参数+源码行号）；3 ✅（§4.4 全路径 + §1.5 数据流图）；7 ✅（全文无占位符与
  未决模糊项；"实编读数验证"为明确的 Stage 2 强制制度）；8 ✅（§3.5.1 全项 +
  §2.2-c/d R8/R9 复检）；9 ✅（bM≥16/bN≥128/D≥64 分形满足，§5.3）；13 ✅
  （§3.5.3 八项 examples/ 路径）；19 ✅（§9.3.1/9.3.2 具体数值 + fp16/bf16
  全 dtype 行）；20 ✅（§11.5 proto.yaml 复核存在且一致）；21 ✅（§5.5
  REORDER-COST 量化）；22 ✅（§9.4 四哨兵：最大 shape/最坏 dtype/最大任务数/
  decode 极端尾块，kernel 路径 120s 阻塞门）。
- 推荐项：4 ✅（§5.2 逐档理由）；5 ✅（§7 flag 矩阵与 Expert 模式匹配）；
  6 ✅（§9.1 golden + §9.2 L0 计划，分层套件交 test-design/现有资产）；
  10 ✅（§5.3 L0C 逐配置）；11/12 ✅（§8.2 workspace 规格表 + workspace_idx
  与签名位置一致）；14/15/16/17 ✅（§3.5.2 差异表含路径/golden 来源/输出形状
  ——BSHD 原生输出无 transpose 需求）；18 ✅（无全局变量依赖，维度全参数化）。
- 三强制审计块：HOST-METADATA-AUDIT ✅（§4.8）；REORDER-COST ✅（§5.5，
  result: pass）；DISPATCH-COVERAGE ✅（§5.6，result: pass，全特化有命中用例）。
