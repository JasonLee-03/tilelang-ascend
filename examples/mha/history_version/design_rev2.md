# mha 算子设计文档（rev2）

> Stage 1 产出（tilelang-op-design + tilelang-op-test-design 场景 A）。
> 编程模式由用户在 Primary 上下文明确指定为 **Developer**，本文档不得改用其他模式
> （rev2 的 threads=1 为 Developer 模式内参数调整，非模式变更）。

## 0. 修订记录（rev1 → rev2）

rev1（`history_version/design_rev1.md`）在 Stage 2 attempt1 实现验证中触发
`[DESIGN_ERROR]` 回退。rev2 遵循**最小必要改动**原则：保留 rev1 全部已验证正确的
设计（BSHD 4D 行跨步搬运、右下角对齐因果掩码链、sumexp+1e-30 全掩码行保护、
D-adaptive block、valid extent 尾块三重保障、L0 十用例表、proto.yaml 接口规格、
Developer 模式四 pass_configs、无 workspace/vid/手动同步），仅修订以下 4 项设计层
缺陷及其牵连章节，并将设计对齐到 attempt1 已位级验证的实现形式。证据统一出处：
`examples/mha/debug_log.md`（attempt1 完整实验记录）与其引用的源码/探针脚本。

| # | 章节 | rev1 内容 | rev2 修订 | 证据出处 |
|---|------|-----------|-----------|----------|
| C1 | §9.3.1/§9.3.3/§10.1-R5 | 阻塞判定 = 混合容差 + `MERE < Threshold` + `MARE_abs < 10×Threshold`（MARE_rel 参考） | **用户裁决（Primary 上下文，2026-09-18，不可更改）**：阻塞判定 = **混合容差（§9.3.2 表不变）+ `MARE_abs < 10×Threshold`**；**MERE 与 MARE_rel 降为参考值**（同步计算并打印，不阻塞、不再触发升级仲裁）。数学依据：多 KV 块 shape 下 MERE 门限被证明不可达——理想下界（完美 fp32 kernel→舍入到 dtype→对照同 dtype golden）fp16 case5=1.64e-3/case9=1.80e-3（门限 9.77e-4）、bf16 case7=1.23e-2/case8=9.88e-3（门限 7.81e-3）；实现已达下界（case9 实测 1.78e-3）；单块特化用例 335× 余量（case4 位级精确 MERE=0.0）独立佐证实现正确。rev1 §9.3.3"MERE 余量≥1 数量级"论断有误，§10.1-R5 只对 MARE_rel 免疫——同一 1e-7 clamp 分母病态同样击中 MERE 均值 | debug_log.md attempt1 error_summary（下界测定表 + decompose_case5.py 误差分解）；用户裁决记录 |
| C2 | §2.2/§2.3/§3.2/§3.3/§3.4/§3.5/§5.3/§6.1/§7.2/§8/§10.1-R8/§10.2 | `threads=2`（消 vid 前提链） | **threads=1（cv_1_1）**。threads=2 下 `AscendVidReduction` pass 对 BSHD 4D region 的 GM↔UB 拷贝结构性误处理（按 `gm_dims-ub_dims=2` 的 unit 头维改写 extent，而行 extent 位于 dim1）：① `maskShapeM` 不减半而 UB buffer 减半 → 2× 行数 OOB 静默互踩（bisect 证实损坏行=vid1 分区）；② 运行期 valid extent → ascend_vid_reduction.cc:430 ICHECK 编译崩溃；③ 1D 列维掩码 buffer（col_pos/mask_row）被减半而 arith_progression/select 不在 size 改写列表 → 64 元素写 32 槽位。threads=1 整体跳过该 pass（src/ir.cc:314-318：vid_value==1 → npu_cv_ratio=cv_1_1），设计原样形式（运行期 valid_m/valid_n、1D col_pos、无 vid 阈值、掩码链）逐一验证位级精确（S=1/30/64/90 探针）。Developer 核心全部保留：四 pass_configs、无 workspace_idx、无 vid、无手动同步、alloc_shared/fragment。仓内先例：example_gqa_sink_bwd_bhsd.py:317（threads=1 + Developer 消除写法）。threads=2 方案移入 §10.1-R8（含复现路径） | debug_log.md attempt1 deviation#1 + pre-verification experiments（verify_bisect_chain/verify_t1a 等探针）；src/ir.cc:314-318；src/transform/ascend_vid_reduction.cc:424-435 |
| C3 | §3.3/§3.4/§4.3/§4.5/§5.3/§6/§10.1-R9/§10.3-⑥ | `T.Pipelined(kv_iters, num_stages=2)` KV 循环 | **T.serial(kv_iters)**（kv_iters 仍编译期常量，尾块语义不变）。threads=1 + T.Pipelined codegen 失败：`Cannot find pre-allocated address for buffer: k_l1`（codegen_ascend.cc:830——双缓冲 L1 gemm 操作数在 cv_1_1 下未注册 address_map；隔离复现 pipe2✗/serial✓）。代价：失去 KV 预取重叠——哨兵分相实测 kernel 路径余量 ≥9×（prefill 13.3s vs 120s 门），可接受；Pipelined/手动双缓冲列为 Stage 3 性能候选（§10.3-⑥） | debug_log.md attempt1 deviation#2（verify_t1_loop_forms.py 隔离复现）；src/target/codegen_ascend.cc:830-834；哨兵分相计时（decompose_prefill.py） |
| C4 | §9.4 | 哨兵 case 级超时门（prefill 600s 含 CPU golden） | 门限重定义为 **kernel 路径计时门（H2D→kernel(含首次编译)→D2H，每哨兵 ≤120s，阻塞）** + **golden 侧降本规定**（分块 golden、缓存复用、prefill 确定性抽样比对）+ case 级 1800s 防挂死参考上限（非阻塞）。依据：rev1 的 600s case 级门被测试侧 golden 成本击穿（605s 超时，其中 kernel 路径仅 13.3s、warm kernel 0.3s；CPU fp16 分块 golden 256 块外推 ≈1409s——torch CPU 无 fp16 GEMM 加速），超时与算子性能无关；其余三哨兵实测通过（decode 47.6s/kernel 6.90s、d256_bf16 6.9s/kernel 5.75s、max_grid 28.4s/kernel 5.63s、16384 任务网格可行） | debug_log.md perf sentinels 节（perf_run2.log + decompose_prefill.py 分相计时） |

**实现对齐项**（attempt1 经验证的非设计缺陷偏离，按 debug_log "keep in revised
design" 要求固化，使设计与已验证实现一致，不要求 kernel 改动）：

| # | 章节 | 对齐内容 | 证据出处 |
|---|------|----------|----------|
| A1 | §3.3/§4.3/§10.2 | col_pos/mask_row 改为**无条件分配**：TVMScript parse-time `if` 块内定义的变量不能跨块使用（"Undefined variable: col_pos" 复现）；语句级 `if need_mask:` 块本身可用（parse-time 求值已验证）。未用时成本 2×64×4B UB | debug_log.md deviation#3（verify_flag_if.py） |
| A2 | §1.4-步骤3/§3.1/§3.2/§3.5.2 | score **双舍入栅格对齐**：golden 实际计算 `round16(round16(qk) × s)`（fp16 matmul 输出 + fp32 opmath 标量乘）；kernel 镜像两次舍入（round-trip qk → fp32 域 ×s → round-trip scores）。2 的幂 scale 位级等价；D=128（s=1/√128）≤1ulp 对齐。替代 rev1 的单次 round-trip（缩放后） | debug_log.md deviation#4 |
| A3 | §1.4/§3.3/§5.6/§9.2/§9.3.3 | **单 KV 块预归一化特化**（编译期谓词 `kv_iters==1`）：sumexp 在唯一迭代内即终值 → P̂=round16(P/(sumexp+1e-30)) 在 GEMM2 前归一化、跳过终除法——与 golden 一次性 softmax 路径结构同构（近位级精确：MERE≤3e-6，case4 bf16 位级精确）。kv_iters≥2 回落 rev1 通用 flash 路径（fallback 不变） | debug_log.md deviation#5 + L0 results（cases 1-4,10 PASS） |

**为何不会再犯同一错误**：C1 的口径缺陷源于把"典型幅值相对误差 ~2⁻¹¹"错误外推到
clamp 加权均值 MERE——rev2 以实测理想下界（而非估计）作为口径论证基础，且阻塞判定
不再依赖任何对近零 golden 元素病态敏感的相对误差均值/极值指标；C2/C3 源于 rev1 直接
沿用 BHSD 参考实现的 threads=2+Pipelined 形式而未对 BSHD 4D region 做 pass 行为核验
——rev2 已将两项固化为源码级证据（ir.cc/codegen_ascend.cc/ascend_vid_reduction.cc
行号）+ 隔离复现路径 + §10.1-R8/R9 风险条目 + §10.2 常见错误行，Stage 2/3 任何
"改回 threads=2 或 Pipelined"的尝试都会先命中风险条目的复现证据；C4 源于门限定义
未区分"算子 kernel 性能"与"测试 harness golden 成本"——rev2 的哨兵门只度量 kernel
路径并强制 golden 降本手段，两类成本不再互相污染。

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
attn[b,i,n,:]     = softmax(scores[b,i,n,:], dim=-1)                   # 全 -inf 行 → 权重全 0（golden 的 all_masked 保护）
y[b,i,n,:]        = Σ_j attn[b,i,n,j] * V[b,j,n,:]
```

### 1.4 算法描述

多步融合算子（GEMM → mask → softmax → GEMM），采用 **Flash-Attention 风格的
online softmax 分块算法**（数学上与一次性 softmax 严格等价）：

1. **任务划分**：每个 kernel 任务负责一个 `(b, n, s_block)` 的输出条带
   `[block_M, D]`；Q 条带常驻 L1，沿 S_kv 维分块迭代 K/V。
2. **GEMM1（Cube）**：`acc_s = Q_tile · K_tileᵀ`（fp32 累加到 L0C）。
3. **双舍入 golden 栅格对齐（Vector，rev2-A2）**：golden 的 scores 实际经历两次
   半精度舍入——`round16(q·k)`（fp16/bf16 matmul 输出）再 `round16(· × s)`
   （fp32 opmath 标量乘后落回 dtype）。kernel 精确镜像该序列：GEMM1 fp32 结果先做
   fp32→dtype→fp32 round-trip（= round16(qk)），在 fp32 域乘 `s`，再做一次
   round-trip（= round16(scores)）。2 的幂 scale（D=64/256）下与 golden 位级等价；
   D=128（s=1/√128 非 2 的幂）≤1ulp 对齐（见 §3.5.2、§9.3.3；证据
   debug_log.md deviation#4）。
4. **因果/边界掩码（Vector，pre-exp -inf 掩码）**：可见性
   `visible(i,j) = (j ≤ min(i + offset, S_kv-1))`（causal）或 `j < S_kv`（非 causal），
   不可见位置置 `-inf`，与 golden 的 `masked_fill(-inf)` 语义逐位一致。
   `offset = S_kv - S`（右下角对齐；`S == S_kv` 时退化为标准下三角）。
5. **online softmax（Vector）**：行最大值 `m_i`、行指数和 `sumexp`、输出累加器
   `acc_o` 三个状态跨 KV 块迭代更新（更新规则见 §3.3 伪代码）。
6. **GEMM2（Cube）**：`acc_o_blk = P_tile · V_tile`，`P = exp(scores - m_i)` 舍入到
   输入 dtype 后作为 GEMM 左操作数。
7. **终归一化 + NaN 保护（Vector）**：`acc_o /= (sumexp + 1e-30)`。
   `+1e-30` 为全掩码行保护：正常行 `sumexp ≥ 1`（行内最大元素贡献 exp(0)=1），
   加 1e-30 低于 fp32 ulp、数值不变；全掩码行 `sumexp = 0, acc_o = 0`，
   得 `0/1e-30 = 0`，与 golden 的 all_masked→0 保护严格一致（不产生 NaN）。
   在声明约束（causal ⇒ S ≤ S_kv）下全掩码行不可达，此保护为防御性设计，
   覆盖 S>S_kv 越界输入时与 golden 行为对齐。
   **单 KV 块预归一化特化（rev2-A3，编译期谓词 `kv_iters == 1`）**：此时 sumexp
   在唯一迭代内即为终值，故在 GEMM2 前先计算 `P̂ = round16(P / (sumexp + 1e-30))`
   并跳过步骤 7 的终除法——数据流与 golden 的一次性 softmax（归一化后舍入 P̂ 再
   matmul）**结构同构**，实现近位级精确（L0 单块用例 MERE ≤ 3e-6，case4 bf16
   位级精确；debug_log.md deviation#5）。`kv_iters ≥ 2` 回落步骤 5-7 的通用
   online softmax 路径（fallback，语义不变）。
8. **写回**：fp32→dtype 舍入后按 `valid_m` 行写回 `Output[b, s0:s0+valid_m, n, :]`。

**Host 侧预处理声明**：host 路径**只做元数据校验与编译期配置推导**（纯 Python
整数/浮点运算：scaleValue 解析、block 尺寸选择、block_num 计算、shape/dtype 断言），
**不触碰任何张量数据**——无 transpose/contiguous/padding/cat 等物理化操作
（详见 §4.8 HOST-METADATA-AUDIT）。BSHD→(b,n) 条带的"逻辑重排"完全由 kernel 内
行跨步（row-stride）DMA 切片完成，无 GM 物理重排 pass。

### 1.5 数据流图

```
host: 元数据校验 + 配置推导（scaleValue→s、D→block 尺寸、block_num；不触碰数据）
                                  │
GM Q[B,S,N,D] ──T.copy(行跨步切片, valid_m 行)──► L1 q_l1[block_M,D]        （任务级常驻）
GM K[B,S_kv,N,D] ─T.copy(行跨步, valid_n 行, T.serial 逐块)──► L1 k_l1[block_N,D]
GM V[B,S_kv,N,D] ─T.copy(行跨步, valid_n 行, T.serial 逐块)──► L1 v_l1[block_N,D]
                                  │
   q_l1,k_l1 ──T.gemm_v0(transpose_B)──► L0C acc_s_l0c[block_M,block_N] fp32   (GEMM1)
                                  │ T.copy（L0C→UB 片上直连）
    UB acc_s_ub_ ──► acc_s_ub: round16(qk) ──► ×s(fp32) ──► round16(scores) ──► causal/边界掩码(-inf)
                                  │
   online softmax: m_i/sumexp/acc_o 状态更新（reduce_max → exp → reduce_sum）
                                  │
   P: acc_s_ub ──T.copy(cast)──► acc_s_half(dtype) ──T.copy──► L1 acc_s_l1     （UB→L1 直连）
                                  │
   acc_s_l1,v_l1 ──T.gemm_v0──► L0C acc_o_l0c[block_M,D] fp32                  (GEMM2)
                                  │ T.copy（L0C→UB 片上直连）
   UB acc_o_ub ──► acc_o = acc_o×exp(m_old−m_new) + acc_o_ub     （KV 循环累加）
                                  │ 循环结束
   acc_o /= (sumexp + 1e-30) ──T.copy(cast)──► acc_o_half(dtype)
                                  │
   GM Output[B,S,N,D] ◄──T.copy(行跨步切片, 仅 valid_m 行)──── acc_o_half
```

内存层级路径全程合法：GM ↔ L1 ↔（gemm_v0 内部管理 L0A/L0B）→ L0C → UB → GM；
Cube↔Vector 通过片上 buffer 直连（Developer 模式 AUTO_CV 四个 pass 负责中转与同步），
无 GM workspace 往返、无跨级搬运。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer（用户在 Primary 上下文明确指定，不可更改）

### 2.2 选型理由

- 算子为 **CV 融合**（GEMM + softmax + GEMM），Developer 模式 + `AUTO_CV_COMBINE`
  是本项目推荐路径：单 cid 轴 + 片上直连，消除 workspace/vid（design-template §8.2、
  programming-model-guide mode-examples §6）。
- **threads 参数取 1（cv_1_1，rev2-C2）**：mode-examples §6.1 的前提链以 threads=2
  为消 vid 手段，但 threads=2 会激活 `AscendVidReduction` pass，该 pass 对本算子
  BSHD 4D region 的 GM↔UB 拷贝存在**结构性误处理**（行 extent 位于 dim1 而 pass 按
  `gm_dims-ub_dims=2` 的 unit 头维改写 → maskShapeM 不减半/UB 减半 → 2× 行 OOB
  静默互踩；另有运行期 valid extent ICHECK 崩溃、1D 掩码 buffer 减半两症状，
  §10.1-R8 全证据）。threads=1 时 `npu_cv_ratio=cv_1_1`（src/ir.cc:314-318），
  **不存在 vid 分裂、pass 整体跳过**——消除写法的全部不变量（单 cid 轴、无 vid
  偏移、无 workspace_idx、片上直连、四 pass_configs）原样成立，且vid-free 更彻底。
- 仓内先例：`examples/attention_sink/example_gqa_sink_bwd_bhsd/…:317`
  （`T.Kernel(bwd_block_num, threads=1, is_npu=True) as (cid)` + Developer 四
  pass_configs 全开 + 零 T.Scope/零手动 flag，:249 注释明示"Developer +
  combineCV (4 True): threads=1"）——threads=1 消除写法为已验证一等形态。
- 结构模板 `examples/developer_mode/flash_attn_bshd_developer.py`（Developer CV
  融合完整骨架：单 cid、无 workspace、L0C→UB→L1 片上直连、online softmax 全链）
  仍然沿用；但其 `threads=2 + T.Pipelined` 两项**不移植**——该文件内部张量实为
  BHSD `[batch, heads, seq_len, dim]`（:36），行维在索引 2 恰等于 VidReduction 的
  target_dim 才免疫 pass bug；本算子 BSHD 行维在 dim1 必触发（debug_log.md
  deviation#1）。
- 无需 Expert 模式的细粒度信号量编排/多版本流水缓冲（§8.6 回退条件均不命中）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（L1/UB，由 MEMORY_PLANNING 按用途放置）+ `T.alloc_fragment`（L0C）；**不使用** `alloc_L1/alloc_ub/alloc_L0A/L0B/L0C` |
| 计算方式 | Cube：`T.gemm_v0`；Vector：`T.tile.*` + `T.reduce_max/sum` + 编译期展开行循环 |
| 作用域 | 编译器自动分离 C/V（`AUTO_CV_COMBINE`）；**不写** `T.Scope("C"/"V")` |
| 同步方式 | 全自动：`AUTO_SYNC` + `AUTO_CV_SYNC`；**不写** `T.barrier_all/set_flag/wait_flag/set_cross_flag` |
| Kernel 启动 | `T.Kernel(block_num, threads=1, is_npu=True) as (cid)`——单轴 + threads=1（cv_1_1，无 vid 分裂；**禁用 threads=2**：AscendVidReduction × BSHD 4D 结构性 bug，§2.2/§10.1-R8，rev2-C2） |
| CV 数据中转 | 片上直连：`T.copy(acc_s_l0c, acc_s_ub_)`、`T.copy(acc_s_half, acc_s_l1)`、`T.copy(acc_o_l0c, acc_o_ub)`；**无 workspace_idx、无 workspace 参数、无 vid 偏移** |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `acc_s = Q_tile · K_tileᵀ` | GEMM1，fp32 累加，逐 KV 块 |
| 2 | `acc_s *= s` | 缩放；`s` 为 host 侧解析的编译期常量 |
| 3 | `acc_s = r16(r16(acc_s) · s)`（fp32 域乘 s；r16 = fp32→dtype→fp32 round-trip） | 双舍入 golden 栅格对齐（rev2-A2：golden 的 scores = round16(round16(qk)×s)，两次半精度舍入逐一镜像） |
| 4 | `acc_s[i,j] = -inf if j > min(i+offset, S_kv-1)` | causal（offset=S_kv-S，右下角对齐）；非 causal 时条件退化为 `j > S_kv-1`（仅尾块生效） |
| 5 | `m_new = max(m_old, rowmax(acc_s))` | online softmax 行最大值状态 |
| 6 | `r = exp(m_old - m_new)` | 历史状态重标定因子 |
| 7 | `P = exp(acc_s - m_new)` | 未归一化注意力权重（≥0，≤1） |
| 8 | `sumexp = sumexp*r + rowsum(P)` | 行指数和状态 |
| 9 | `acc_o = acc_o*r + P_dtype · V_tile` | GEMM2 + 输出累加（P 舍入到输入 dtype） |
| 10 | `y = acc_o / (sumexp + 1e-30)` | 终归一化；`+1e-30` 即全掩码行→0 保护 |

**掩码语义与 golden 的等价性**（索引映射）：golden 对
`scores[..., i, j]` 在 `j > i + (S_kv - S)` 处 `masked_fill(-inf)`；kernel 内
任务级行全局索引 `i_g = bx*block_M + i`、列全局索引 `j_g = k*block_N + j`，
掩码条件 `j_g > i_g + offset` 逐位一致。尾列（`j_g ≥ S_kv`，S_kv 非 block_N 整除时
最后一块）在 golden 中不存在，在 kernel 中由同一阈值式的 `min(·, S_kv-1)` 钳位
置 `-inf`，保证不进入 softmax 分母——与 K/V 尾块零填充（§5.4）配合，语义封闭。

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 任务并行 | (b,n,s_block) 网格 | `T.Kernel(block_num, threads=1, is_npu=True) as (cid)` | `block_num = B*N*ceildiv(S, block_M)`；bx/by/bz 由 cid 整除分解；threads=1=cv_1_1（rev2-C2） | Developer |
| Q/K/V 搬入 | BSHD 行跨步切片 | `T.copy(Q[bz, s0:s0+valid_m, by, :], q_l1)` 等 | 列维 = D（buffer 最内维，行内连续 ✓）；行 pitch = N*D；`valid_m/valid_n = T.min(block, 边界-s0)` 运行期切片 | Developer |
| GEMM1 | `Q·Kᵀ` | `T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)` | A=[block_M,D] L1，B=[block_N,D] L1，C=[block_M,block_N] fragment(fp32) | Developer |
| L0C→UB | scores 下放 | `T.copy(acc_s_l0c, acc_s_ub_)` | fp32→fp32，片上直连（AUTO_CV 中转） | Developer |
| golden qk 舍入 | `r16(qk)`：fp32→dtype→fp32 | `T.copy(acc_s_ub, acc_s_half)` + `T.copy(acc_s_half, acc_s_ub)` | T.copy 跨 dtype 自动 Cast（common.h copy_ub_to_ub）；对齐 golden 的半精度 matmul 输出栅格（rev2-A2 第一次舍入） | Developer |
| 缩放 | `×s`（fp32 域） | `T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)` | src1 为编译期 float 标量 | Developer |
| golden scores 舍入 | `r16(scores)`：fp32→dtype→fp32 | 同上 `T.copy` ×2 | rev2-A2 第二次舍入；scores 与 golden 位级等价（2 的幂 s）/ ≤1ulp（D=128） | Developer |
| 列索引 | `col_pos[j]=k*block_N+j` | `T.tile.arith_progression(col_pos, k*block_N, 1, block_N)` | fp32 [block_N]；首值为运行期 TIR 表达式 | Developer |
| 掩码比较 | `j_g ≤ thresh_i` | `T.tile.compare(mask_row, col_pos, T.cast(thresh_i, "float32"), "LE")` | tensor-vs-scalar；统一单式 `thresh_i = T.min(s0+i+offset_eff, S_kv-1)`，`offset_eff = causal ? S_kv-S : 2**30`（host 编译期常量；非 causal 时 min 退化为 S_kv-1 钳位，与 rev1 双形式数学等价、与 attempt1 实现同形） | Developer |
| 掩码施加 | 不可见→ -inf | `T.tile.select(acc_s_ub[i,:], mask_row, acc_s_ub[i,:], -T.infinity("float"), "VSEL_TENSOR_SCALAR_MODE")` | bit=1（可见）取 src0 tensor，bit=0 取 -inf 标量 | Developer |
| 行最大值 | `rowmax` | `T.reduce_max(acc_s_ub, m_i, dim=-1)` | clear=True（默认），后接 `T.tile.max(m_i, m_i, m_i_prev)` 合并历史 | Developer |
| 重标定 | `exp(m_old-m_new)` | `T.copy(m_i, m_i_prev)`（reduce 前）→ `T.tile.sub(m_i_prev, m_i_prev, m_i)` → `T.tile.exp(m_i_prev, m_i_prev)` | 与验证参考逐行一致 | Developer |
| 减最大值 | `acc_s - m_i` | `for h_i in range(block_M): T.tile.sub(acc_s_ub[h_i,:], acc_s_ub[h_i,:], m_i[h_i])` | 行循环 + 行切片-标量（参考实现同款写法） | Developer |
| 指数 | `exp` | `T.tile.exp(acc_s_ub, acc_s_ub)` | 自然指数（Ascend 仅支持 exp，golden 亦为自然 softmax） | Developer |
| 行和 | `rowsum` | `T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)` | 后接 `T.tile.mul(sumexp, sumexp, m_i_prev)` + `T.tile.add(sumexp, sumexp, sumexp_i_ub)` | Developer |
| P 舍入 | fp32→dtype | `T.copy(acc_s_ub, acc_s_half)` | GEMM2 左操作数精度 = 输入 dtype（与 golden 的 fp16/bf16 权重同栅格） | Developer |
| UB→L1 | P 上 L1 | `T.copy(acc_s_half, acc_s_l1)` | Developer CV 片上直连（mode-examples §6.2 "softmax → L1 直连"） | Developer |
| GEMM2 | `P·V` | `T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)` | A=[block_M,block_N]，B=[block_N,D]，C=[block_M,D] fragment(fp32) | Developer |
| 输出累加 | `acc_o*r + blk` | `T.copy(acc_o_l0c, acc_o_ub)` → `for h_i: T.tile.mul(acc_o[h_i,:], acc_o[h_i,:], m_i_prev[h_i])` → `T.tile.add(acc_o, acc_o, acc_o_ub)` | 行循环标量乘（参考实现同款） | Developer |
| 安全归一 | `/(sumexp+1e-30)` | `T.tile.add(sumexp, sumexp, 1e-30)` → `for h_i: T.tile.div(acc_o[h_i,:], acc_o[h_i,:], sumexp[h_i])` | 全掩码行→0 保护 | Developer |
| 输出写回 | valid_m 行 | `T.copy(acc_o, acc_o_half)` → `T.copy(acc_o_half[0:valid_m,:], Output[bz, s0:s0+valid_m, by, :])` | 输出侧显式 valid extent，禁止写满块 | Developer |

> 掩码/softmax 的行循环均为**编译期 Python `range` 展开**（block_M 为 JIT 常量），
> 循环体内是行切片级 `T.tile.*` SIMD 指令——与已验证参考实现
> `flash_attn_bshd_developer.py:96-97,113-114,118-119` 完全同构。

### 3.3 计算伪代码

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(out_idx=[3], pass_configs=pass_configs)     # 无 workspace_idx
def mha_kernel(B, S, S_kv, N, D, dtype="float16", scaleValue=-1.0, is_causal=False):
    accum_dtype = "float"
    sm_scale = scaleValue if scaleValue > 0 else (1.0 / D) ** 0.5   # host 纯 Python
    block_M = 32 if D == 256 else 64          # UB/L0C 预算决定（§5.2）
    block_N = 64
    causal_offset = S_kv - S                  # 右下角对齐偏移（causal 时恒 ≥ 0，host 已断言）
    need_bound_mask = (S_kv % block_N != 0)   # 尾列掩码（编译期）
    need_mask = is_causal or need_bound_mask  # False 时完全跳过掩码链（编译期）
    offset_eff = causal_offset if is_causal else 2**30   # 统一阈值式（非 causal 退化为钳位）
    s_blocks = (S + block_M - 1) // block_M
    kv_iters = (S_kv + block_N - 1) // block_N           # 编译期常量循环边界
    block_num = B * N * s_blocks
    single_kv_block = (kv_iters == 1)         # 单 KV 块预归一化特化（rev2-A3）

    @T.prim_func
    def main(
        Q: T.Tensor([B, S, N, D], dtype),
        K: T.Tensor([B, S_kv, N, D], dtype),
        V: T.Tensor([B, S_kv, N, D], dtype),
        Output: T.Tensor([B, S, N, D], dtype),
    ):
        with T.Kernel(block_num, threads=1, is_npu=True) as (cid):   # cv_1_1（rev2-C2）
            bx = cid % s_blocks                          # s_block（最快维）
            by = cid // s_blocks % N                     # head
            bz = cid // s_blocks // N % B                # batch
            s0 = bx * block_M
            valid_m = T.min(block_M, S - s0)             # 运行期尾块行数

            q_l1 = T.alloc_shared([block_M, D], dtype)           # → L1
            k_l1 = T.alloc_shared([block_N, D], dtype)           # → L1（T.serial 单缓冲）
            v_l1 = T.alloc_shared([block_N, D], dtype)           # → L1（T.serial 单缓冲）
            acc_s_l1 = T.alloc_shared([block_M, block_N], dtype) # → L1（GEMM2 A）
            acc_s_l0c = T.alloc_fragment([block_M, block_N], accum_dtype)  # → L0C
            acc_o_l0c = T.alloc_fragment([block_M, D], accum_dtype)        # → L0C
            acc_s_ub = T.alloc_shared([block_M, block_N], accum_dtype)     # → UB
            acc_s_ub_ = T.alloc_shared([block_M, block_N], accum_dtype)
            acc_s_half = T.alloc_shared([block_M, block_N], dtype)
            acc_o = T.alloc_shared([block_M, D], accum_dtype)
            acc_o_ub = T.alloc_shared([block_M, D], accum_dtype)
            acc_o_half = T.alloc_shared([block_M, D], dtype)
            m_i = T.alloc_shared([block_M], accum_dtype)
            m_i_prev = T.alloc_shared([block_M], accum_dtype)
            sumexp = T.alloc_shared([block_M], accum_dtype)
            sumexp_i_ub = T.alloc_shared([block_M], accum_dtype)
            col_pos = T.alloc_shared([block_N], accum_dtype)   # 列全局索引（rev2-A1：
            mask_row = T.alloc_shared([block_N], accum_dtype)  # 无条件分配，TVMScript
            # 跨块变量作用域限制；need_mask=False 时仅耗 2×64×4B UB，规划期回收）

            T.tile.fill(acc_o, 0.0)
            T.tile.fill(sumexp, 0.0)
            T.tile.fill(m_i, -(2**30))
            T.copy(Q[bz, s0 : s0 + valid_m, by, :], q_l1)         # 行跨步 + valid_m 尾块

            for k in T.serial(kv_iters):                          # rev2-C3：编译期常量边界
                j0 = k * block_N
                valid_n = T.min(block_N, S_kv - j0)
                T.copy(K[bz, j0 : j0 + valid_n, by, :], k_l1)     # 尾块行由 GM→L1 拷贝自动清零
                T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                T.copy(acc_s_l0c, acc_s_ub_)                      # L0C→UB 片上直连

                T.tile.fill(acc_s_ub, 0.0)
                T.copy(m_i, m_i_prev)
                T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                # 双舍入 golden 栅格对齐（rev2-A2）：round16(qk) → ×s(fp32) → round16(scores)
                T.copy(acc_s_ub, acc_s_half)
                T.copy(acc_s_half, acc_s_ub)
                T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                T.copy(acc_s_ub, acc_s_half)
                T.copy(acc_s_half, acc_s_ub)

                if need_mask:                                     # 编译期分支（4 变体，§5.6；
                    T.tile.arith_progression(col_pos, j0, 1, block_N)  # 语句级 parse-time if
                    for i in range(block_M):                           # 已验证可用，rev2-A1）
                        # causal: thresh = min(s0+i+offset, S_kv-1)；非 causal: 退化为 S_kv-1
                        thresh_i = T.min(s0 + i + offset_eff, S_kv - 1)
                        T.tile.compare(mask_row, col_pos, T.cast(thresh_i, accum_dtype), "LE")
                        T.tile.select(acc_s_ub[i, :], mask_row, acc_s_ub[i, :],
                                      -T.infinity(accum_dtype), "VSEL_TENSOR_SCALAR_MODE")

                T.reduce_max(acc_s_ub, m_i, dim=-1)
                T.tile.max(m_i, m_i, m_i_prev)
                T.tile.sub(m_i_prev, m_i_prev, m_i)
                T.tile.exp(m_i_prev, m_i_prev)                    # r = exp(m_old - m_new)

                for h_i in range(block_M):
                    T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])
                T.tile.exp(acc_s_ub, acc_s_ub)                    # P = exp(s - m_new) ∈ [0,1]
                T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                T.tile.mul(sumexp, sumexp, m_i_prev)
                T.tile.add(sumexp, sumexp, sumexp_i_ub)

                if single_kv_block:                               # rev2-A3：预归一化特化
                    T.tile.add(sumexp, sumexp, 1e-30)             # （全掩码行保护同语义）
                    for h_i in range(block_M):
                        T.tile.div(acc_s_ub[h_i, :], acc_s_ub[h_i, :], sumexp[h_i])

                T.copy(acc_s_ub, acc_s_half)                      # P(或 P̂) → dtype
                T.copy(acc_s_half, acc_s_l1)                      # UB→L1 直连

                T.copy(V[bz, j0 : j0 + valid_n, by, :], v_l1)
                T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                T.copy(acc_o_l0c, acc_o_ub)                       # L0C→UB 片上直连

                for h_i in range(block_M):
                    T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], m_i_prev[h_i])
                T.tile.add(acc_o, acc_o, acc_o_ub)

            if not single_kv_block:                               # 通用路径终归一化
                T.tile.add(sumexp, sumexp, 1e-30)                 # （全掩码行保护：正常行
                for h_i in range(block_M):                        #   sumexp ≥ 1，+1e-30
                    T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])  # 无位级影响）

            T.copy(acc_o, acc_o_half)
            T.copy(acc_o_half[0:valid_m, :], Output[bz, s0 : s0 + valid_m, by, :])

    return main
```

Host 包装函数（同文件，`mha.py` 对外接口）：

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
    return kernel(query, key, value)   # 输出由 out_idx=[3] 分配，[B,S,N,D] contiguous
```

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|---------|---------|
| `T.Kernel(n, threads=1, is_npu=True) as (cid)` | `examples/attention_sink/example_gqa_sink_bwd_bhsd/…:317`（Developer 消除写法先例，:249 注释明示 threads=1）；`src/ir.cc:314-318`（vid_value==1 → npu_cv_ratio=cv_1_1）；本算子 attempt1 全用例运行验证 | ✅ 仓内先例 + 源码 + 本算子实测 |
| `T.alloc_shared` / `T.alloc_fragment` | flash_attn_bshd_developer.py:52-71；本算子 attempt1（threads=1 下 MEMORY_PLANNING 放置正常） | ✅ 已验证 |
| `T.gemm_v0(A, B, C, transpose_B=True, init=True)` | flash_attn_bshd_developer.py:81,109；api-compute.md §1；本算子 attempt1 | ✅ 已验证 |
| `T.serial(编译期常量边界)` KV 循环 | 本算子 attempt1 隔离复现（serial ✓ 编译+运行位级正确）；sink_bwd:323-324（运行期边界 T.serial 在 threads=1 Developer 下亦可用的更强先例） | ✅ 已验证 |
| 🚫 `T.Pipelined(num_stages=2)` @ threads=1 | codegen 失败：`Cannot find pre-allocated address for buffer: k_l1`（codegen_ascend.cc:830，双缓冲 L1 操作数未注册 address_map；attempt1 隔离复现 pipe2✗/serial✓） | 🚫 禁用（§10.1-R9；Stage 3 候选 §10.3-⑥） |
| `T.copy` L0C→UB / UB→L1 / GM↔L1 片上直连 | 同上 :82,106,110；mode-examples.md §6.2 映射表 | ✅ 已验证 |
| `T.copy` 跨 dtype Cast（fp32↔fp16/bf16，UB→UB） | 同上 :105,121；`src/tl_templates/ascend/common.h` copy_ub_to_ub（float↔half/bfloat16_t 双向 Cast 分支） | ✅ 源码确认 |
| `T.copy` 行跨步 head 切片 `[rows, head_idx, :D]`→L1 | `examples/xattention/xattention.py:381-389`（K_shared 3D 同款，CI 通过）；api-kernel-memory.md §T.copy 多维切片（列维=最内维 D ✅）；本算子 attempt1：verify_bshd_slice.py 4D 整除位级精确 + L0 全用例 | ✅ 4D 已验证（threads=1 下位级精确） |
| `T.copy` 运行期 valid extent 切片（输入/输出两侧） | api-kernel-memory.md §T.copy 动态 shape 切片（:266-315）；`examples/grouped_gemm` valid_m 模式；本算子 attempt1：verify_t1a.py（S=1/30/64/90 运行期 valid_m/valid_n 位级精确）、verify_kv_tail.py（尾列清零 ✓） | ✅ 已验证（threads=1；threads=2 下会触发 ICHECK 崩溃，§10.1-R8） |
| GM→L1 尾块自动清零 | `src/tl_templates/ascend/common.h` copy_gm_to_l1（need_clear → InitConstValue 全 tile 清零，:84-92） | ✅ 源码确认 |
| GM→UB 尾块 padValue=0 填充 | 同上 copy_gm_to_ub（:238-268，Duplicate(padValue) + DataCopyPad） | ✅ 源码确认 |
| `T.reduce_max/reduce_sum(buf, out, dim=-1)` | flash_attn_bshd_developer.py:90,100；api-compute.md §2；本算子 attempt1 | ✅ 已验证 |
| `T.tile.fill/add/mul/sub/exp/max/div` | flash_attn_bshd_developer.py:73-119；本算子 attempt1（AUTO_CV + threads=1 + T.serial 下位级验证） | ✅ 已验证 |
| `T.tile.arith_progression(buf, first, 1, count)`（first 为运行期表达式） | `examples/attention_sink/example_gqa_sink_bwd_bhsd/…:162,178`；ascend_tile.py:389（签名/dtype/≥0 约束） | ✅ 已验证示例+源码 |
| `T.tile.compare(dst, src, 运行期标量, "LE")` | `examples/sparse_flash_attention/example_sparse_flash_attn_mask.py:196`（`T.float32(s_i)` 运行期 cid 派生标量） | ✅ 已验证示例 |
| `T.tile.select(dst, mask, src, -T.infinity(dt), "VSEL_TENSOR_SCALAR_MODE")` | 同上 :214；`example_gqa_sink_bwd_bhsd.py:188-194`；ascend_tile.py:734 | ✅ 已验证示例 |
| 掩码链 ops 在 AUTO_CV 机制中的注册 | `src/transform/common/operation_config.h:184-237`（compare/compare_scalar/select/arith_progression/broadcast/bitwise_and 均注册为 PIPE_V）；本算子 attempt1：verify_mask_chain.py / verify_chain_isolate.py + L0 causal 用例（2/4/6/8/9）位级验证。**rev2 注**：threads=1 下 AscendVidReduction 整体跳过，rev1 关注的"行切片 size 与 M 分裂"问题不复存在；该 pass 在 threads=2 下对 1D 列维 buffer（col_pos/mask_row）反而有减半 bug（§10.1-R8 症状③） | ✅ 已验证（R1 关闭） |
| TVMScript parse-time `if PY_BOOL:` 语句块 / 跨块变量作用域 | 本算子 attempt1 verify_flag_if.py：语句级 `if need_mask:` 块可用（parse-time 求值，Muls 存在/缺失探针）；**块内定义的变量不能跨块使用**（"Undefined variable" 复现）→ 跨块 buffer 必须无条件分配（§3.3 col_pos/mask_row，rev2-A1） | ✅ 已验证（限制已固化） |
| `T.min` / `T.cast` / `-T.infinity(dt)`（TIR 表达式） | api-kernel-memory.md:297；sink_bwd:147-148；sparse_mask:214 | ✅ 文档+示例 |
| 禁用项：`tilelang/language/pto.py` | AGENTS.md 明令废弃 | 🚫 不使用 |
| 禁用项：`T.Kernel` 三维 / `threads>2` / **`threads=2`（本算子 BSHD 4D，§10.1-R8）** / **`T.Pipelined`（threads=1 下，§10.1-R9）** / 动态边界循环 / `T.gemm` 通用版 | ascend-constraints.md §1 + attempt1 实证 | 🚫 不使用（§3.5.1 逐项核对） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | 涉及（逻辑网格 B×N×S_blocks 为三维） | `T.Kernel(block_num)` 一维网格 + cid 整除分解 `bx/by/bz`（参考 flash_attn_bshd_developer.py:47-50）。任务间完全独立、无共享输出 buffer，无需 block_metadata 表 |
| threads 参数限制（仅 1 或 2） | 涉及 | **`threads=1`（cv_1_1，rev2-C2）**：threads=2 激活 AscendVidReduction，对 BSHD 4D GM↔UB 拷贝结构性误处理（OOB 静默互踩/ICHECK 崩溃/1D buffer 减半，§10.1-R8 三重证据）；threads=1 跳过该 pass（src/ir.cc:314-318），设计原样形式逐一位级验证（verify_t1a.py），Developer 消除写法不变量全部保留 |
| 动态循环边界不支持 | 涉及（S/S_kv 运行期可变） | B/S/S_kv/N/D 全部作为 **JIT 编译期参数**传入（每个 shape 组合编译一次，tilelang 缓存管理）；kernel 内所有循环边界（`kv_iters`、`range(block_M)`）均为编译期常量，**kernel 内无依赖 tensor 值的循环边界**。尾块不进循环边界，用 `valid_m/valid_n` 运行期切片表达 |
| 流水线不支持动态边界 | 涉及 | **主方案 `T.serial(kv_iters)`（rev2-C3）**：`kv_iters = ceildiv(S_kv, block_N)` 为编译期常量（动态边界约束天然满足）；threads=1 下 `T.Pipelined(num_stages=2)` codegen 失败（address_map 未注册 k_l1，§10.1-R9）→ 不采用。运行期因果裁剪边界（`eff_kv_iters`，mha_sink 模式）仍**不在 v1 采用**——sink_bwd:323-324 已证明运行期边界 T.serial 在 threads=1 Developer 下可用（技术障碍解除），列为 Stage 3 性能优化候选（§10.3-①） |
| 部分 GPU API 不可用 | 不涉及 | 无外部 GPU 参考实现；全部 API 出自本项目 examples/源码（§3.4） |
| L0C 容量上限（128KB） | 涉及 | `acc_s_l0c + acc_o_l0c`：最大配置 D=128/block_M=64 → 16KB+32KB=48KB；D=256/block_M=32 → 8KB+32KB=40KB ✅（§5.3） |
| T.copy 不支持列方向 strided 切片 | 涉及（BSHD 4D 切片） | 切片 `Q[bz, s0:s0+valid_m, by, :]` 的列维 = D = buffer 最内维（行内连续，行间 gap = (N-1)*D 元素）→ 属硬件支持的"行内连续、行间 gap"模型 ✅；**无** `k:k+1` 型列跨步切片（§5.4 附最小复现验证要求） |
| Host 侧 reshape 隐式拷贝 | 不涉及 | host 路径零张量变换（§4.8 审计）；输出直接返回 kernel 产物（tilelang out_idx 分配，天然 contiguous） |
| Host 侧 torch.nn.functional / cat / to(dtype) 等 aclnn 调用 | 不涉及 | host 仅 Python 标量运算 + 断言 + kernel 调用；golden 仅在测试侧 CPU 运行（§9.4 注） |
| 输出侧切片 + reshape 隐式 contiguous | 不涉及 | kernel 直接输出 `[B,S,N,D]` 完整 buffer（尾块经 valid_m 切片写回），host 无后处理 |

#### 3.5.2 参考实现差异说明

无外部 GPU 参考实现。用户提供的 PyTorch golden（§9.1）仅用于定义数学语义，
不映射其 API。golden 与 kernel 的数值语义差异及对齐手段：

| 差异项 | golden（PyTorch/NPU 语义） | 本 kernel（Ascend） | 对齐方案 |
|--------|---------------------------|--------------------|----------|
| scores 精度栅格 | `matmul(q,k)` 输出 fp16/bf16（半精度舍入）×scale 后半精度——即 `round16(round16(qk)·s)` 两次舍入 | GEMM fp32 累加（L0C） | **双舍入镜像（rev2-A2）**：round-trip(qk) → fp32 域 ×s → round-trip(scores)（§1.4 步骤 3），scores 落在 golden 精确栅格上，消除最大系统性误差源 |
| softmax 数值路径 | `F.softmax(dim=-1)`：半精度输入、fp32 内部累加、半精度输出（**归一化后**舍入 P̂） | kv_iters≥2：online softmax fp32 域，**未归一化** P 舍入到 dtype 后进 GEMM2，归一化推迟到终态；**kv_iters==1（rev2-A3 特化）**：预归一化 P̂=round16(P/(sumexp+1e-30)) 后进 GEMM2、跳过终除法——与 golden 结构同构 | 单块路径位级可追踪（case4 bf16 位级精确、单块用例 MERE≤3e-6）；多块路径残差 ~2⁻¹¹ 相对量级，由 §9.3 阻塞口径（混合容差 + MARE_abs）覆盖——MERE 参考值的不可达下界论证见 §9.3.1（flash 类算子固有：P 预归一化舍入无法复刻 torch 归一化后舍入的位级决策） |
| 布局 | 内部 transpose 为 [B,N,S,D] 计算，输出 `.contiguous()` 回 BSHD | 原生 BSHD 行跨步 DMA，无物理 transpose | 输出天然 [B,S,N,D] contiguous，与 golden 输出契约逐位同形 |
| 全掩码行 | softmax NaN → `torch.where(all_masked, 0, w)` 显式置 0 | `sumexp+1e-30` 安全除法 → 输出 0 | 语义一致（均输出全 0 行、无 NaN）；声明域内（causal⇒S≤S_kv）该行不可达，属防御性对齐 |
| scale 舍入位置 | `round16(qk) × s →（fp32 opmath）→ round16` | 双舍入镜像同序（rev2-A2）：`r16(qk) → ×s(fp32) → r16` | 2 的幂 s（D=64/256 自动 scale=2⁻³/2⁻⁶，及 case10 显式 0.0625）位级等价；D=128（s=1/√128 非 2 的幂）实测 ≤1ulp 对齐（debug_log.md deviation#4），仅舍入边界元素触发，阻塞口径余量内 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/developer_mode/flash_attn_bshd_developer.py` | **高度相似（结构模板，部分形式不移植）** | Developer CV 融合完整骨架：单 cid、4 项 pass_configs 全开、无 workspace、alloc_shared/fragment、online softmax 全链（fill/add/mul/reduce_max/max/sub/exp/reduce_sum/行循环）、L0C→UB→L1 片上直连、out_idx=[3]。**rev2 注**：其 `threads=2 + T.Pipelined(num_stages=2)` 不移植——该文件内部张量实为 BHSD（:36 `shape=[batch, heads, seq_len, dim]`），行维在索引 2 恰为 VidReduction target_dim 才免疫 pass bug；本算子 BSHD 行维在 dim1，threads=2 必触发 §10.1-R8 |
| `examples/attention_sink/example_gqa_sink_bwd_bhsd/example_gqa_sink_bwd_bhsd.py` | **高（threads=1 消除写法 + 掩码技术双先例）** | ① `T.Kernel(bwd_block_num, threads=1, is_npu=True) as (cid)` + Developer 四 pass_configs 全开 + 零 T.Scope/零手动 flag（:249,:317，rev2-C2 直接先例）；② kernel 内 causal 掩码构造：`arith_progression(row/col 索引) → compare("LE") → select(-T.infinity, VSEL_TENSOR_SCALAR_MODE)`（:162-194）；右对齐 causal 语义；③ `T.serial(loop_st, loop_ed)` 运行期边界裁剪在 threads=1 Developer 下可用的先例（:323-324，Stage 3-① 参考） |
| `examples/sparse_flash_attention/example_sparse_flash_attn_mask.py` | 高（掩码+BSHD） | 逐行 `compare(mask, col_idx, 运行期标量, "LE")` + 行循环 `select(row, mask, row, -inf, TENSOR_SCALAR)`（:196-214）；**BSHD 4D 布局**（q_shape=[batch,seq,heads,dim]）先例 |
| `examples/xattention/xattention.py` | 中（搬运先例） | `T.copy(K_shared[row0:row1, head_idx, :D], k_l1)` 行跨步 head 切片 GM→L1（:381-389，运行期 head 索引）——本设计 BSHD 搬运的同构先例 |
| `examples/mha_sink_fwd_bhsd/mha_sink_fwd_bhsd.py` | 中（MHA 语义） | MHA（无 GQA）任务分解 by=head；全掩码行精度教训（:564-607 注释：归一化基准不当→输出坍缩）；causal KV 裁剪优化思路（Stage 3） |
| `examples/flash_attention/flash_attn_bhsd.py` | 中（对照组） | 同算法 Expert/workspace 版：L0C→workspace→UB 两跳数据流对照，佐证 Developer 直连等价性 |
| `examples/gqa_fwd_varlen/gqa_fwd_varlen.py` + `test_gqa_fwd_varlen.py` | 中（语义对照） | 右对齐 causal（`k_idx <= q_idx + offset`）与 SDPA 左对齐差异说明（test:100-152）；"避免 S>S_kv+causal（不可见行→NaN）"约束先例（test:422） |
| `examples/developer_mode/sparse_flash_attn_developer_vid_reduce.py` | 中（改造对照） | workspace+vid → threads=2 消除写法的已验证对照（mode-examples §6.1 前提链；**rev2 注**：该对照为 BHSD 场景，本算子因 §10.1-R8 采用 threads=1 消除写法——消除不变量同、threads 取值异） |
| `examples/linear_attention_and_rnn/linear_attention_causal.py` | 低-中 | `functools.lru_cache` 包装编译复用（多 shape 顺序测试）；B/L 符号化备选方案 |
| `examples/softmax/example_online_softmax.py` | 低 | online softmax 数学与 `CAST_NONE/CAST_RINT` cast 模式常量命名 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| query | (B, S, N, D) | float16（代表性；全集见 §9.3） | BSHD 布局，contiguous，有限实数 |
| key | (B, S_kv, N, D) | 同 query | B/N/D 与 query 严格相等 |
| value | (B, S_kv, N, D) | 同 query | 同上 |
| scaleValue | 标量 float（host 参数，非 tensor） | — | 默认 -1.0；≤0 → 自动 1/√D；JIT 编译期常量 |
| is_causal | 标量 bool（host 参数，非 tensor） | — | 默认 False；True 时要求 S ≤ S_kv；JIT 编译期常量 |

支持范围（host 断言强制）：B∈[1,128]，S∈[1,2048]，S_kv∈[1,4096]，N∈[1,64]，
D∈{64,128,256}（64 对齐、静态）；dtype ∈ {float16, bfloat16}（三输入一致）。

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| Output | (B, S, N, D) | 同 query | out_idx=[3] 由 tilelang 分配；contiguous（BSHD 原生写回，无 host 后处理） |

### 4.3 中间缓冲区

配置记号：`bM = block_M`（D=256→32，否则 64），`bN = block_N = 64`，
`dt = 输入 dtype`，`fp32 = accum_dtype`。

| Buffer 名 | Shape | dtype | 存储层级（Developer 自动放置） | 用途 |
|-----------|-------|-------|------|------|
| q_l1 | (bM, D) | dt | alloc_shared → L1 | Q 条带（任务级常驻，GEMM1 A） |
| k_l1 | (bN, D) | dt | alloc_shared → L1 | K 块（T.serial 单缓冲，GEMM1 B；双缓冲为 Stage 3 候选 §10.3-⑥） |
| v_l1 | (bN, D) | dt | alloc_shared → L1 | V 块（T.serial 单缓冲，GEMM2 B；同上） |
| acc_s_l1 | (bM, bN) | dt | alloc_shared → L1 | P（GEMM2 A，UB 直连上载） |
| acc_s_l0c | (bM, bN) | fp32 | alloc_fragment → L0C | GEMM1 输出 |
| acc_o_l0c | (bM, D) | fp32 | alloc_fragment → L0C | GEMM2 输出 |
| acc_s_ub_ | (bM, bN) | fp32 | alloc_shared → UB | L0C→UB 落地缓冲 |
| acc_s_ub | (bM, bN) | fp32 | alloc_shared → UB | softmax 主计算缓冲（缩放/舍入/掩码/exp） |
| acc_s_half | (bM, bN) | dt | alloc_shared → UB | P 半精度化（兼作 golden 舍入 round-trip 中转） |
| acc_o | (bM, D) | fp32 | alloc_shared → UB | 输出累加器（跨 KV 块状态） |
| acc_o_ub | (bM, D) | fp32 | alloc_shared → UB | GEMM2 结果落地/累加暂存 |
| acc_o_half | (bM, D) | dt | alloc_shared → UB | 输出半精度化（写回前） |
| m_i / m_i_prev | (bM,) | fp32 | alloc_shared → UB | 行最大值状态 / 重标定因子 |
| sumexp / sumexp_i_ub | (bM,) | fp32 | alloc_shared → UB | 行指数和状态 / 当前块行和 |
| col_pos | (bN,) | fp32 | alloc_shared → UB | 列全局索引 k*bN+j（rev2-A1：无条件分配——TVMScript 跨块变量作用域限制；need_mask=False 时仅耗 256B，掩码链仍由语句级 `if need_mask:` 编译期消除） |
| mask_row | (bN,) | fp32 | alloc_shared → UB | compare 位掩码行缓冲（同上，无条件分配） |

### 4.4 内存搬运路径

```
GM[Q] ──T.copy(行跨步, valid_m)──► L1[q_l1] ──(gemm_v0 内部 L1→L0A)──► L0A
GM[K] ──T.copy(行跨步, valid_n, T.serial 逐块)──► L1[k_l1] ──(内部 L1→L0B, transpose_B)──► L0B
L0A+L0B ──gemm_v0──► L0C[acc_s_l0c] ──T.copy(片上直连)──► UB[acc_s_ub_] → UB[acc_s_ub]
UB[acc_s_ub]（缩放/舍入/掩码/softmax）──T.copy(cast dt)──► UB[acc_s_half] ──T.copy──► L1[acc_s_l1] ──► L0A
GM[V] ──T.copy(行跨步, valid_n, T.serial 逐块)──► L1[v_l1] ──► L0B
L0A+L0B ──gemm_v0──► L0C[acc_o_l0c] ──T.copy(片上直连)──► UB[acc_o_ub] → UB[acc_o]（rescale+累加）
UB[acc_o]（终归一）──T.copy(cast dt)──► UB[acc_o_half] ──T.copy(行跨步, valid_m)──► GM[Output]
```

- 全部路径符合硬件内存层级：GM↔L1↔L0A/L0B→L0C→UB→GM；无 GM→L0、GM→UB→L1 之外的
  跨级违规（UB→L1 直连为 Developer CV 融合法定直连，mode-examples §6.2）。
- BSHD 行跨步切片合法性：切片 extents `[1, valid, 1, D]`，列维（最后一个 extent≠1
  的维）= D = buffer 最内维 → "行内连续、行间 gap=(N−1)·D 元素"，属 DataCopyPad
  硬件支持模型（api-kernel-memory.md §T.copy 多维切片表 ✅ 行）；**不涉及**列方向
  strided（`k:k+1` 型）禁用形态。

### 4.5 UB 内存预算

按声明总量计（Developer MEMORY_PLANNING 负责实际放置；预算基准 = 已验证参考实现
flash_attn_bshd_developer 以 ~180.5KB 声明量在 D=512 下编译运行通过）：

**D=64（bM=64, bN=64）**

| Buffer | 大小 (Bytes) |
|--------|-------------|
| acc_s_ub / acc_s_ub_ (64×64×fp32) | 16384 × 2 |
| acc_s_half (64×64×fp16) | 8192 |
| acc_o / acc_o_ub (64×64×fp32) | 16384 × 2 |
| acc_o_half (64×64×fp16) | 8192 |
| m_i/m_i_prev/sumexp/sumexp_i (64×fp32 ×4) | 1024 |
| col_pos/mask_row (64×fp32 ×2) | 512 |
| **总计** | **≈ 82,432 / 196,608 (192KB, A2/A3)** ✅ |

**D=128（bM=64, bN=64）**

| Buffer | 大小 (Bytes) |
|--------|-------------|
| acc_s_ub / acc_s_ub_ (64×64×fp32) | 16384 × 2 |
| acc_s_half (64×64×fp16) | 8192 |
| acc_o / acc_o_ub (64×128×fp32) | 32768 × 2 |
| acc_o_half (64×128×fp16) | 16384 |
| 状态向量 ×4 + col_pos/mask_row | 1536 |
| **总计** | **≈ 124,544 / 196,608** ✅ |

**D=256（bM=32, bN=64）**

| Buffer | 大小 (Bytes) |
|--------|-------------|
| acc_s_ub / acc_s_ub_ (32×64×fp32) | 8192 × 2 |
| acc_s_half (32×64×fp16) | 4096 |
| acc_o / acc_o_ub (32×256×fp32) | 32768 × 2 |
| acc_o_half (32×256×fp16) | 16384 |
| 状态向量 (32×fp32 ×4) + col_pos/mask_row (64×fp32 ×2) | 1024 |
| **总计** | **≈ 104,960 / 196,608** ✅ |

L1 预算（512KB）：最大配置 D=256：q_l1 16KB + k_l1 32KB + v_l1 32KB + acc_s_l1 4KB
= 84KB（T.serial 单缓冲，rev2-C3）✅；Stage 3 若启用 k/v 手动双缓冲（§10.3-⑥）
→ ≈148KB，仍在预算内 ✅。
L0C 预算见 §5.3；L0A/L0B（gemm_v0 内部管理）最大单操作数：D=256 时
A=(32×256×2B)=16KB、B=(256×64×2B)=32KB，均 ≤64KB ✅。

### 4.6 动态轴定义

B/S/S_kv/N 为运行期可变维度，采用 **JIT 编译期参数**方案（本项目 attention 示例
统一做法，非 T.symbolic）：

| 动态轴 | 声明方式 | 运行时范围 | 重编译粒度 |
|--------|----------|-----------|-----------|
| B | `mha_kernel(B=…)` Python int → JIT 常量 | 1 ~ 128 | 每个 (B,S,S_kv,N,D,dtype,scale,causal) 组合编译一次；tilelang 缓存复用，可选 `functools.lru_cache` 包装（linear_attention_causal.py:119-133 先例） |
| S | 同上 | 1 ~ 2048 | 同上 |
| S_kv | 同上 | 1 ~ 4096 | 同上 |
| N | 同上 | 1 ~ 64 | 同上 |
| D | 同上（静态语义，取值封闭 {64,128,256}） | 64/128/256 | 同上（并决定 block_M 配置） |

kernel 内由动态轴派生的量：`block_num`（host 计算）、`s0/bx/by/bz`（cid 运行期
分解）、`valid_m/valid_n`（T.min 运行期切片）——**均不作为循环边界**；kernel 内
循环边界 `kv_iters`、`range(block_M)` 对该编译实例全部静态 ✅。

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[3],                 # Output 为 main 的第 4 个参数，由框架分配返回
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,    # 自动 CV 分离（threads=1 → cv_1_1，无 vid 分裂）
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,       # 自动核间同步
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,          # 自动核内同步
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,    # 自动内存规划（L1/UB 放置+双缓冲）
    },
)   # 无 workspace_idx（Developer CV 融合消除写法，mode-examples §6）
```

### 4.8 Host 侧 Buffer 操作审计（强制）

```text
[HOST-METADATA-AUDIT]
operation: query.shape / query.dtype / key.shape 读取（校验与配置推导）
input_stride -> output_stride: 不产生新张量
shares_storage / same_data_ptr: N/A（只读元数据）
aclnn_or_physical_copy: false
result: allow

operation: Python 标量运算（sm_scale 解析、block_M/block_N/block_num/valid 配置、assert 校验）
input_stride -> output_stride: 无张量参与
shares_storage / same_data_ptr: N/A
aclnn_or_physical_copy: false
result: allow

operation: str(query.dtype).removeprefix("torch.")（dtype 字符串化，用于 JIT 参数）
input_stride -> output_stride: 无张量变换（仅 Python 字符串）
shares_storage / same_data_ptr: N/A
aclnn_or_physical_copy: false（注意：这是 host 元数据读取，不是 tensor.to(dtype) 转换）
result: allow

operation: kernel(query, key, value) 调用 + 直接 return 输出
input_stride -> output_stride: 输入原样传入；输出为 out_idx 新分配 contiguous [B,S,N,D]
shares_storage / same_data_ptr: 输出为框架新分配（合法：kernel 计算产物，非 host 变换）
aclnn_or_physical_copy: false（无 host 侧 reshape/contiguous/切片/cat/pad/to/clone）
result: allow
```

结论：host 完整路径（kernel 前 + kernel 后）只含元数据读取、纯 Python 运算、
kernel 调用与结果直返；**零物理化操作、零 aclnn 依赖**，输出 contiguous 契约由
kernel 的 BSHD 原生写回直接满足（无需 golden 那样的 transpose+contiguous）。

**rev2 复核**：host 路径与 rev1 完全一致（四项修订均在 kernel/测试侧，不触碰
host 元数据路径）；attempt1 实现复检结论 `result: allow`（debug_log.md
[HOST-METADATA-AUDIT] 节：shape/dtype/is_contiguous 读取 + Python 断言 +
kernel 分派 + 直返，无任何 reshape/contiguous/切片/cat/pad/to/clone）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 混合（Cube + Vector，CV 融合）

**判定依据**: 两次 matmul（QKᵀ、PV，Cube/L0 路径）夹一个多步 softmax
（max/exp/sum/div，Vector/UB 路径），判定为融合算子（决策树"matmul +
element-wise 前/后处理 → CV 融合"分支；test-design 分类 = Fusion）。

### 5.2 Block 划分

```python
block_N = 64                       # KV 块：所有 D 配置统一
block_M = 32 if D == 256 else 64   # Q 块：由 UB 预算决定（D=256 时 64 会超 192KB，§4.5）
s_blocks  = ceildiv(S, block_M)    # host Python
kv_iters  = ceildiv(S_kv, block_N) # JIT 编译期常量
block_num = B * N * s_blocks       # 一维任务网格
# cid 分解（seq 最快 → 同 head 任务相邻，K/V 重读局部性最好）：
bx = cid % s_blocks;  by = cid // s_blocks % N;  bz = cid // s_blocks // N % B
```

选择理由：
- `block_M=64`：GEMM M 维 64 ≥ 分形下限 16，任务数与 Q 重用的平衡点；与 UB 预算
  联立后可覆盖 D≤128。
- `block_N=64`：与已验证参考（flash_attn_bshd_developer block_N=64）一致；
  fp32 行向量 64 元素 = 256B，恰好满足 compare/mask 的 256B 对齐下限；
  K/V T.serial 单缓冲下 L1 占用适中（双缓冲后仍 ✅，§4.5）。
- `D=256 → block_M=32`：acc_o 系三缓冲（fp32×2+fp16）随 D 线性增长，bM=64 时
  声明量 ≈203KB 超 UB；bM=32 降至 ≈105KB（§4.5）。备选（bM=64 + 删 acc_s_ub_
  ≈187KB）余量过窄，留作 Stage 3 调优项。

### 5.3 约束分析

- **分形限制**（fp16/bf16：M≥16, K≥16, N≥16）：
  GEMM1 M=bM(64/32)✅ K=D(64/128/256)✅ N=bN(64)✅；GEMM2 M=bM✅ K=bN(64)✅ N=D✅。
- **对齐要求**：GM 行片段 = D×2B ∈ {128, 256, 512}B ≥ 32B ✅ 且为 32B 整数倍；
  行 pitch N×D×2B 为 32B 整数倍 ✅；UB/L1 buffer 尺寸均为 256B 整数倍 ✅。
- **UB 容量**：三配置声明量 82.4KB / 124.5KB / 105.0KB ≤ 192KB ✅（§4.5，
  基准 = 参考实现 180.5KB 已验证可编译运行）。
- **L0C 容量**（128KB，强制检测 `bM×bN×sizeof(fp32) ≤ 131072`）：

| 配置 | acc_s_l0c | acc_o_l0c | 合计 | 判定 |
|------|-----------|-----------|------|------|
| D=64, bM=64 | 64×64×4 = 16KB | 64×64×4 = 16KB | 32KB | ✅ |
| D=128, bM=64 | 16KB | 64×128×4 = 32KB | 48KB | ✅ |
| D=256, bM=32 | 32×64×4 = 8KB | 32×256×4 = 32KB | 40KB | ✅ |

  保守元素规则 `M×N ≤ 16384（fp32）`：最大单 fragment = 64×128 = 8192 ✅
  （若未来启用 D=256+bM=64，acc_o_l0c = 64×256 = 16384 恰在上限，需重新评估）。
- **L0A/L0B**（64KB）：最大操作数 D=256：A=(32×256×2B)=16KB、B=(256×64×2B)=32KB ✅。
- **L1**（512KB）：最大 ≈84KB（T.serial 单缓冲，§4.5；Stage 3 双缓冲候选 ≈148KB 仍 ✅）。
- **threads**：**1** ✅（仅允许 1/2；**threads=2 禁用**——AscendVidReduction × BSHD 4D
  结构性 bug，§10.1-R8；threads=1 → cv_1_1 跳过该 pass，src/ir.cc:314-318）。
- **Kernel 维度**：一维 block_num ✅。

### 5.4 注意事项（非整除与尾块，显式设计）

S/S_kv 不保证被 block 整除（S∈[1,2048] 任意、S_kv∈[1,4096] 任意）。**禁止 host
padding + crop**；输入、输出 GM 两侧一律显式 valid extent BufferRegion：

| 位置 | valid extent | 表达式 | 越界区语义 |
|------|-------------|--------|-----------|
| Q 搬入 | valid_m | `T.min(block_M, S - bx*block_M)` | q_l1 尾行由 copy_gm_to_l1 `need_clear→InitConstValue` 清零（源码确认）；零行产生有限垃圾 scores，仅影响自身行（softmax/累加/GEMM 行独立），写回时丢弃 |
| K/V 搬入 | valid_n | `T.min(block_N, S_kv - k*block_N)` | k_l1/v_l1 尾行清零 → 尾列 scores=0（有限）、v 尾行=0 |
| softmax 域 | 尾列掩码 | `thresh = min(…, S_kv-1)` 钳位（§3.1） | 尾列（j_g ≥ S_kv）置 -inf → P=exp(-inf)=0，不进 sumexp；GEMM2 中 0×0=0（v 尾行已清零，无 0×NaN 风险） |
| Output 写回 | valid_m | 同 Q | `T.copy(acc_o_half[0:valid_m,:], Output[bz, s0:s0+valid_m, by, :])`——只写有效行，不越界、不覆盖相邻任务区域（任务间输出区域天然不相交，无竞态） |

**尾块正确性论证**（三重保障）：① GM→L1/UB 拷贝尾区**确定性清零**（源码
copy_gm_to_l1 need_clear / copy_gm_to_ub Duplicate(padValue=0)，杜绝陈旧 NaN
位模式导致的 0×NaN=NaN 污染）；② 尾列经阈值钳位进 -inf 掩码，softmax 分母
不含尾列；③ 输出只写 valid_m 行。垃圾行（Q 尾行）的有限值不参与任何跨行
归约（reduce dim=-1 行内闭合），行独立性保证有效行结果与整除情形逐位一致。

**极小 shape**：S=1（decode）→ s_blocks=1、valid_m=1，GEMM 以 bM 行缓冲执行
（63/64 行为零填充垃圾，计算合法、结果丢弃）；S_kv=1 → kv_iters=1、valid_n=1，
掩码钳位 thresh=0，P[:,0]=1 → 输出 = V[:,0,:] ✅。N=1 → 行 pitch = D（退化为
连续搬运）✅。

**BSHD 行跨步切片验证（rev1 规定的 Stage 2 强制第一步——已完成 ✅，rev2 关闭）**：
4D 切片 `Q[bz, s0:s0+valid_m, by, :]`（extents [1,valid_m,1,D]，列维=最内维 D）在
attempt1 以最小复现对照验证：verify_bshd_slice.py（整除 shape GM→UB 位级精确）、
verify_t1a.py（threads=1 下运行期 valid_m/valid_n，S=1/30/64/90 位级精确）、
verify_kv_tail.py（循环内运行期 valid_n + 尾列自动清零 ✓），并经 L0 十用例全量
复核——**无静默错位**。教训记录：threads=2 时代的 W1/W2 写回探针"通过"实为对称
溢出巧合（bisect 后证伪，debug_log.md deviation#1），写回形式判定一律以 threads=1
位级探针为准。应急预案（未触发，保留备查）：按行循环 1D 连续搬运
`for r in range(bM): T.copy(Q[bz, s0+r, by, :], q_l1[r, :])`
（sparse_flash_attn_developer.py:148-152 gather 循环同构）。

### 5.5 数据搬运性能可行性（REORDER-COST，强制）

本算子**无物理布局重排**（BSHD 原生进/原生出，无 host/kernel 重排 pass、无
GM 中转），成本模型即 flash 标准 KV 重读模型。逐路径审计：

```text
[REORDER-COST]（rev2 复核：threads=1 单核任务路径 + attempt1 实测锚定）
path: generic（唯一搬运路径；BSHD 原生行跨步 DMA，掩码 4 编译期变体与
  single_kv_block 特化共用同一搬运结构——特化只改归一化位置，不改数据流）
gm_passes: Q×1 + Output×1 + K/V × s_blocks（flash 标准 KV 重读因子 = S/block_M）
dma_transactions: 每任务 ≈ bM(Q) + 2·bN·kv_iters(K,V) + valid_m(Out)；
  最大哨兵 S1（§9.4）：≈ 64 + 2×64×64 + 64 ≈ 8320/任务 × 8192 任务 ≈ 6.8×10⁷
average_dma_bytes: D×sizeof(dt) = 128B(D=64,fp16) / 256B(D=128) / 512B(D=256)——全部 ≥32B 对齐、
  无逐元素搬运；行跨步仅体现在行间 gap（DMA 引擎原生 srcGap，无额外指令）
gm_scalar_accesses: 0（全程块 DMA；无 T.copy 单元素、无索引 gather）
address_div_mod_per_element: 0（地址增量由切片静态推导；仅每任务 3 div + 2 mod 解码 cid）
active_cores / serial_tasks_per_core: 24 / ceil(block_num/24)（S1: 342；S4: 683）——
  cv_1_1 下每任务占 1 个 AI Core 对（1 AIC + 1 AIV），任务并发仍受 24 AIC 上限；
  T.serial 无预取重叠的代价已被实测吸收（下行）
measured_kernel_path（attempt1，threads=1 + T.serial，debug_log.md perf 节）:
  S1 prefill: kernel 路径 13.3s（编译 5.8s + warm 0.3s + H2D 1.1s + D2H 0.1s + 输入侧 6.3s 中
    H2D 计入项）；S2 decode: kernel 6.90s；S3 d256: 5.75s；S4 max_grid: 5.63s（16384 任务）
largest_case_timeout_gate: S1-S4 kernel 路径门各 ≤120s（§9.4 rev2；实测最坏 13.3s → ≥9× 余量）；
  case 级 1800s 防挂死参考上限（非阻塞，golden 成本不计入可行性门）
result: pass
```

S1 定量核算：GM 流量 ≈ K/V 重读 32×(2×8×4096×32×128×2B) ≈ 17.2GB + Q/Out 0.27GB
→ @≈1TB/s ≈ 18ms；算力 4·B·N·S·S_kv·D ≈ 1.1 TFlop（causal 无裁剪含 ~50% 掩码区
冗余）→ @50-150 TFLOPS ≈ 7-22ms；DMA 发射 6.8×10⁷ 事务/24 核 ≈ 2.8×10⁶/核 ≈ 1.6ms。
**实测锚定（attempt1）**：S1 warm kernel 0.3s——高于纯理论值（任务调度/启动与 T.serial
无预取的开销），但仍对 120s kernel 路径门有 ≥400× 余量；理论-实测差属 Stage 3 优化
空间（§10.3-①/⑥），不构成可行性风险。大张量主路径**无逐元素 strided GM 访问** ✅。

### 5.6 多路径分派审计（DISPATCH-COVERAGE，强制）

```text
[DISPATCH-COVERAGE]（rev2 复核：threads=1 单核路径 + attempt1 特化清单）
supported_domain: Q/K/V dtype ∈ {float16, bfloat16}（三者一致）；
  shape：B∈[1,128], S∈[1,2048], S_kv∈[1,4096], N∈[1,64], D∈{64,128,256}；
  attr：scaleValue∈ℝ（≤0 自动 1/√D），is_causal∈{True,False}（True ⇒ S≤S_kv）；
  输入值域：任意有限实数
generic_fallback: 单一通用 kernel（上述全域；kv_iters≥2 的 online softmax 通用路径），
  编译期配置选择构成全函数分派，无输入落空。
  threads=1 复核：分派全部位于 host 纯 Python 元数据层，与核数无关；kernel 内
  cv_1_1 单核执行（C/V 由 AUTO_CV_COMBINE 自动分离、AUTO_CV_SYNC 自动同步），
  无任何 vid 条件分派/行分担分支——threads 变更不产生新分派路径，覆盖矩阵不变
specializations:
  - predicate: D == 256 → (block_M,block_N)=(32,64)；D ∈ {64,128} → (64,64)
    fallback_on_miss: 不适用（谓词在 D 域上穷尽；D∉{64,128,256} 被 host 断言拒绝）
    equivalence_evidence: §4.5/§5.3 UB/L0C/L1 预算表——仅资源占用不同，算法同一
  - predicate: (is_causal, S_kv % block_N == 0) → 掩码 4 变体：
      (F,T)=无掩码链；(F,F)=仅尾列钳位；(T,T)=仅 causal；(T,F)=causal+钳位
      （统一 offset_eff 阈值式，§3.2：非 causal 时 min(s0+i+2³⁰, S_kv-1) 退化为钳位）
    fallback_on_miss: 不适用（2×2 编译期穷尽）
    equivalence_evidence: §3.1 索引映射——thresh=min(i_g+offset, S_kv-1) 与 golden
      masked_fill(j > i+offset) + 尾列裁剪逐位等价；off 时掩码链整体编译期消除
      （语句级 if need_mask，rev2-A1 已验证 Muls 存在/缺失）
  - predicate: kv_iters == 1 → single_kv_block 预归一化特化（rev2-A3）
    fallback_on_miss: kv_iters ≥ 2 → 通用 online softmax 路径（本设计 §1.4 步骤 5-7，
      谓词二分穷尽，无第三分支）
    equivalence_evidence: 单迭代下 online softmax 的 m/sumexp 在迭代内即终值，
      预归一化与终归一化代数同值、仅舍入位置不同——且预归一化恰好与 golden 的
      一次性 softmax（归一化后舍入 P̂）结构同构；实测近位级精确（L0 cases 1-4,10：
      MERE ≤ 3e-6，case4 bf16 位级精确，debug_log.md L0 results）
  - predicate: scaleValue > 0 → s=scaleValue；否则 s=1/√D
    fallback_on_miss: 不适用（ℝ 全函数）
    equivalence_evidence: golden `if scaleValue <= 0: scaleValue = 1.0/(D**0.5)` 逐字一致
unsupported_inputs: dtype 三者不一致 / dtype∉{fp16,bf16} / D∉{64,128,256} /
  is_causal∧S>S_kv / 维度超出支持范围 / 非 contiguous 输入 / 非有限值输入
  （NaN/Inf 不在声明值域，Stage 2 L2/Boundary 非阻塞探测）——全部由 host 元数据
  断言拒绝（不读数据），与 supported_domain 无冲突
result: pass（attempt1 实测：D=256→case10 PASS；single_kv_block→cases 1-4,10 PASS；
  multi-block 通用路径→cases 5-9 混合容差+MARE_abs 全过；掩码 (F,T)/(T,T) 变体
  L0 已覆盖；(F,F)/(T,F) 尾钳位变体需 S_kv%64≠0 → L1 扩展覆盖，Stage 2 attempt2）
```

分派只读取 shape/dtype/attr 元数据（纯 Python），不读取、不改动张量数据 ✅；
每条 specialization 在 §9.2 L0 计划中至少 1 个命中用例（D=256→case10；掩码 4
变体→(F,T):cases 1/3/5/7/10、(T,T):cases 2/4/6/8/9、(F,F)/(T,F) 尾钳位→L1 扩展
（需 S_kv%64≠0，Stage 2 attempt2）；single_kv_block→cases 1-4/10、通用多块路径→
cases 5-9；scale 两分支→cases 1-9 自动 / case10 显式）✅。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 任务网格 (B×N×S_blocks) | block 级并行 | `T.Kernel(block_num, threads=1, is_npu=True) as (cid)` | 一维网格（平台不支持三维 Kernel）；任务间零依赖；threads=1=cv_1_1 单 V 核（无 vid 行分担，VidReduction 不触发，rev2-C2） |
| KV 迭代 (kv_iters) | 串行循环（编译期常量边界） | `T.serial(kv_iters)` | rev2-C3：threads=1 + T.Pipelined codegen 失败（address_map 未注册双缓冲 k_l1，§10.1-R9）；T.serial 已验证编译+运行位级正确，kv_iters 为编译期常量、尾块语义不变；预取重叠损失由哨兵实测吸收（kernel 路径 ≥9× 余量），双缓冲/流水恢复列 Stage 3（§10.3-⑥） |
| 掩码行循环 (block_M) | 编译期展开 | `for i in range(block_M)`（Python range） | 行切片 compare/select，阈值 = 运行期 bx 表达式 + 编译期 i；单 V 核执行全部行（无 vid 分裂，行切片 size 安全性问题不复存在） |
| softmax/累加行循环 (block_M) | 编译期展开 | `for h_i in range(block_M)` | `acc_s_ub[h_i,:] -= m_i[h_i]`、`acc_o[h_i,:] *= m_i_prev[h_i]`、`acc_o[h_i,:] /= sumexp[h_i]`——参考实现同款写法，attempt1 位级验证 |
| 元素级 | SIMD 向量化 | `T.tile.*` / `T.reduce_*` | UB buffer 级指令，无标量循环 |

**明确不采用（v1）**：`T.serial(运行期边界)` 因果 KV 裁剪（mha_sink `eff_kv_iters` /
sink_bwd `loop_st:loop_ed` 模式）——主循环已是 T.serial，运行期边界的技术障碍已解除
（sink_bwd:323-324 在 threads=1 Developer 下验证可用），但 v1 保持编译期常量边界以
最小化变量；作为 Stage 3 优化候选（§10.3-①，收益 ~2× causal 算力）。
**明确不采用**：`T.Persistent`（任务数≫核数时调度收益有限，且参考实现未用）。
**明确禁用**：`T.Pipelined(num_stages≥2)` @ threads=1（codegen bug，§10.1-R9；
恢复条件见 §10.3-⑥）。

### 6.2 循环伪代码

```python
with T.Kernel(block_num, threads=1, is_npu=True) as (cid):      # cv_1_1（rev2-C2）
    bx/by/bz = cid 分解; s0 = bx*block_M; valid_m = T.min(block_M, S-s0)
    <buffer 分配（§4.3；col_pos/mask_row 无条件分配，rev2-A1）>
    T.tile.fill(acc_o, 0.0); T.tile.fill(sumexp, 0.0); T.tile.fill(m_i, -(2**30))
    T.copy(Q[bz, s0:s0+valid_m, by, :], q_l1)                 # 任务级一次
    for k in T.serial(kv_iters):                              # 编译期常量边界（rev2-C3）
        <GEMM1 → 双舍入对齐 → 掩码 → online softmax（含 single_kv_block
         预归一化分支，rev2-A3）→ GEMM2 → 累加>（§3.3）
    if not single_kv_block:                                   # 通用路径终归一化
        T.tile.add(sumexp, sumexp, 1e-30)                      # 全掩码行保护
        for h_i in range(block_M): T.tile.div(acc_o[h_i,:], acc_o[h_i,:], sumexp[h_i])
    T.copy(acc_o, acc_o_half)
    T.copy(acc_o_half[0:valid_m,:], Output[bz, s0:s0+valid_m, by, :])
```

### 6.3 流水线与预取（rev2-C3 重写）

- **主方案：`T.serial(kv_iters)` 无循环级流水**。K/V 的 GM→L1 搬运与当块
  GEMM/softmax 串行执行，无预取重叠；C/V 片上直连的生产/消费事件仍由
  `AUTO_CV_SYNC` 自动同步（与循环形态无关）。
- **T.Pipelined 禁用依据（threads=1）**：`T.Pipelined(kv_iters, num_stages=2)` 在
  cv_1_1 下 codegen 失败——`Cannot find pre-allocated address for buffer: k_l1`
  （codegen_ascend.cc:830：MEMORY_PLANNING 产生的双缓冲 L1 gemm 操作数未注册进
  address_map）。attempt1 隔离复现：同 kernel pipe2 ✗ / serial ✓
  （verify_t1_loop_forms.py，debug_log.md deviation#2）。
- **性能影响已量化、可接受**：哨兵分相实测（§9.4）kernel 路径最坏 13.3s
  （prefill，含首次编译 5.8s；warm 0.3s），对 120s 门 ≥9× 余量——无预取的损失
  被 24 核任务级并行吸收（8192 任务 ≫ 24 核，核间天然重叠）。
- **恢复路径（Stage 3 候选，§10.3-⑥）**：(a) T.serial + 手动 k/v 双缓冲
  （ping-pong 索引，L1 84→148KB 仍 ✅ §4.5）；(b) 推动 codegen address_map 修复后
  恢复 T.Pipelined；(c) VidReduction BSHD 修复后评估 threads=2 + Pipelined
  （须先复跑 §10.1-R8 探针）。任何恢复都以 §9.4 kernel 路径门复检为前提。
- 不使用 num_stages>2 的多版本缓冲（属 §8.6 回退场景，本设计不命中）。

### 6.4 尾块处理

循环边界**不因尾块改变**（kv_iters/s_blocks 用 ceildiv 向上取整，尾块作为最后一次
迭代），尾块由迭代内 `valid_m/valid_n` 运行期切片表达：输入侧 `Q/K/V[…, :valid]`
（GM 两侧显式 BufferRegion）、输出侧 `acc_o_half[0:valid_m,:] → Output[…, :valid_m]`。
不使用"标量 GM 起点 + 完整 UB tile"写法；不做 host padding + crop；掩码阈值钳位
保证尾列不进 softmax 分母（§5.4 三重保障）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 全自动同步（Developer 标准）——kernel 源码中**零手动同步原语**
（无 `T.barrier_all` / `T.set_flag` / `T.wait_flag` / `T.set_cross_flag` /
`T.wait_cross_flag` / `T.Scope`）。

### 7.2 同步点说明

手动同步点：无。自动同步职责划分：

| 同步需求 | 负责 pass | 说明 |
|---------|-----------|------|
| 核内管线事件（MTE2→M→FIX→V→MTE3 等） | `TL_ASCEND_AUTO_SYNC` | GM→L1 完成后才 gemm、L0C→UB 落地后才 vector 等，由 pass 按数据依赖插入 |
| Cube↔Vector 片上直连中转与核间事件 | `TL_ASCEND_AUTO_CV_SYNC` | acc_s_l0c→acc_s_ub_、acc_s_half→acc_s_l1、acc_o_l0c→acc_o_ub 三处直连的生产/消费同步 |
| C/V 代码分离（cv_1_1 单 V 核） | `TL_ASCEND_AUTO_CV_COMBINE` | threads=1 下自动 CV 分离为 1 AIC + 1 AIV；**无 vid 行分担**（行循环/行切片 op 全部在单 V 核执行，VidReduction pass 不触发，rev2-C2） |
| buffer 层级放置 | `TL_ASCEND_MEMORY_PLANNING` | alloc_shared → L1/UB 判定（gemm 操作数→L1，vector 操作数→UB）；T.serial 单缓冲（无双版本 k_l1/v_l1，rev2-C3） |

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

（四项全开 = Developer 模式标准配置，与 flash_attn_bshd_developer.py:13-18 逐字一致。）

---

## 8. 融合算子设计

### 8.1 融合算子判定

**判定结果**: 是

**判定依据**: GEMM(QKᵀ) + element-wise/reduction 后处理（scale/mask/softmax）+
GEMM(PV)，Cube 与 Vector 交替协作，属 CV 融合算子（决策树"matmul + element-wise
→ CV 融合"分支）。

### 8.2 CV 交互设计（按编程模式）

**Developer 模式（本算子采用，消除 workspace/vid）**——不产出 workspace 规格：

- `T.Kernel(block_num, threads=1, is_npu=True) as (cid)`（单轴 + threads=1=cv_1_1）✅
- 装饰器无 `workspace_idx`，`main` 签名无 `workspace_*` 参数（仅 Q/K/V/Output）✅
- Cube↔Vector 全部片上 `alloc_shared/alloc_fragment` 直连：

| 语义角色 | 片上直连（本设计） |
|----------|-------------------|
| Cube 输出 QKᵀ | `T.copy(acc_s_l0c, acc_s_ub_)`（L0C→UB） |
| Vector 回写概率 P | `T.copy(acc_s_half, acc_s_l1)`（UB→L1） |
| Cube 输出 PV | `T.copy(acc_o_l0c, acc_o_ub)`（L0C→UB） |

- 中转/双缓冲/同步交给四个 pass（§7.2）；无 GM workspace 往返、无 vid 偏移。
- 前提链说明（rev2-C2 修订）：mode-examples.md §6.1 的"threads=2 → 消 vid → 消
  workspace"针对 vid 分裂场景；本设计 threads=1（cv_1_1）**根本不存在 vid 轴**
  （src/ir.cc:314-318），单 cid 轴绑定与消除写法同形，消 workspace 结论不变——
  消除写法的全部不变量（无 workspace_idx、无 vid 偏移、片上直连、四 pass_configs、
  零手动同步）逐项成立。threads=1 消除写法仓内先例：sink_bwd:249,317。
  已验证对照：flash_attn_bshd_developer.py（消除写法，但其 threads=2/Pipelined
  不移植，§3.5.3 注）vs flash_attn_bhsd.py（旧 workspace+vid 写法，仅对照不采用）。
- mode-examples §6.6 回退条件核查：无多版本核间流水需求（T.serial 主方案；双缓冲预取为 Stage 3
  性能项 §10.3-⑥，不引入 workspace）、无细粒度信号量编排需求、无编译器报错
  （attempt1 全用例编译通过）→ **不回退**，不设计 workspace 规格。

### 8.3 Cube 核计算流程

```python
# Developer（消除写法）：Cube 输出直连片上 buffer，无 workspace
T.copy(K[bz, j0:j0+valid_n, by, :], k_l1)
T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)   # GEMM1
T.copy(acc_s_l0c, acc_s_ub_)                                    # L0C → alloc_shared 直连
# …Vector 侧 softmax/mask（§8.4）…
T.copy(V[bz, j0:j0+valid_n, by, :], v_l1)
T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)                 # GEMM2
T.copy(acc_o_l0c, acc_o_ub)                                     # L0C → alloc_shared 直连
```

（C/V 归属由 AUTO_CV_COMBINE 自动切分，此处按语义角色展示，源码不写 T.Scope。）

### 8.4 Vector 核计算流程

```python
# Developer（消除写法，threads=1/cv_1_1）：片上 buffer 直读，无 workspace、无 vid 偏移
T.tile.fill(acc_s_ub, 0.0); T.copy(m_i, m_i_prev)
T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
T.copy(acc_s_ub, acc_s_half); T.copy(acc_s_half, acc_s_ub)      # r16(qk)（rev2-A2）
T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)                        # ×s（fp32 域）
T.copy(acc_s_ub, acc_s_half); T.copy(acc_s_half, acc_s_ub)      # r16(scores)＝golden 栅格
<掩码链：arith_progression + 行循环 compare/select(-inf)>        # §3.3（if need_mask）
T.reduce_max(acc_s_ub, m_i, dim=-1); T.tile.max(m_i, m_i, m_i_prev)
T.tile.sub(m_i_prev, m_i_prev, m_i); T.tile.exp(m_i_prev, m_i_prev)
<行循环 sub m_i>; T.tile.exp(acc_s_ub, acc_s_ub)
T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
T.tile.mul(sumexp, sumexp, m_i_prev); T.tile.add(sumexp, sumexp, sumexp_i_ub)
if single_kv_block:                                             # rev2-A3 预归一化特化
    T.tile.add(sumexp, sumexp, 1e-30); <行循环 div sumexp>       # P̂ = r16(P/(sumexp+1e-30))
T.copy(acc_s_ub, acc_s_half); T.copy(acc_s_half, acc_s_l1)      # P(或 P̂) → dtype → L1 直连
<行循环 mul m_i_prev>; T.tile.add(acc_o, acc_o, acc_o_ub)       # rescale + 累加
# 循环外（仅通用路径）：
if not single_kv_block:
    T.tile.add(sumexp, sumexp, 1e-30)                           # 全掩码行保护
    <行循环 div sumexp>                                          # 终归一化
T.copy(acc_o, acc_o_half)
T.copy(acc_o_half[0:valid_m, :], Output[bz, s0:s0+valid_m, by, :])   # 输出
```

### 8.5 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 CV 分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,      # 自动核间同步
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 自动同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # 内存规划
}
```

### 8.6 注意事项

- **threads=1（cv_1_1）为本算子强制选择（rev2-C2）**：无 vid 轴、无 vid 偏移，
  消 workspace 结论与消除写法一致；源码全文不得出现 `vid`、`workspace`、
  `T.Scope`、手动同步原语（自检 grep 项不变）。**不得改回 threads=2**——
  AscendVidReduction × BSHD 4D 结构性 bug（§10.1-R8，OOB 静默互踩为**非确定性**
  损坏，常规精度测试可能漏检）；不得以 mode-examples §6.1 前提链或参考实现
  flash_attn_bshd_developer（内部 BHSD，免疫）为由恢复。
- 本设计 T.serial 无循环级流水；C/V 直连的生产/消费事件由 AUTO_CV_SYNC 负责，
  不额外叠加手工多级缓冲（双缓冲预取为 Stage 3 候选，§10.3-⑥）。
- 无 `workspace_idx`，无需校验参数位置一致性（该注意事项仅适用回退写法）。
- vid 行分担安全性约束在 threads=1 下不再适用（VidReduction 不触发）；记录备查：
  若未来修复 pass 后评估 threads=2，须重新核验行切片 size 与 1D buffer 处理
  （§10.1-R8 症状③表明该 pass 对 1D 列维 buffer 现状即有 bug）。

---

## 9. 验证方案

### 9.1 Golden 函数

以下代码由需求方提供，**原样纳入**（Stage 2 `test_mha.py` 以此为唯一精度标杆，
禁止改写语义；测试时在 **CPU** 上以与输入相同的 dtype 运行——不做 upcast，
避免测试 harness 引入 NPU aclnn 依赖）：

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

算子调用示例（用户侧契约）：

```python
B, S, S_kv, N, D = 2, 128, 128, 8, 64
query = torch.randn(B, S, N, D, dtype=torch.float16, device="npu")
key = torch.randn(B, S_kv, N, D, dtype=torch.float16, device="npu")
value = torch.randn(B, S_kv, N, D, dtype=torch.float16, device="npu")
y = mha(query, key, value, scaleValue=-1.0, is_causal=False)
```

### 9.2 L0 门槛测试计划

> 由 `tilelang-op-test-design`（场景 A，从本 design 提取）生成。**Stage 1 只给 L0**
> （规则 shape，block 整除），供 Stage 2 快速精度收敛；L1（功能，含不规则/尾块/
> 质数 shape，按 test-design §6 确定性生成：`S = bM·k+1 / bM·k+bM//2 / nearest_prime`
> 等，GEMM 类三轴非对齐必出）/ L2（异常：dtype 不一致、D∉{64,128,256}、
> causal∧S>S_kv 等应被 host 断言拒绝）/ Boundary（特殊值）的完整分层套件由
> Stage 2 调 tilelang-op-test-design 场景 B 在 L0 通过后扩展，**不在此手工枚举**。

**算子类别判定**（test-design §3）：计算类型 = 混合（CV 融合，两 matmul 夹
softmax）；复杂度 = Fusion；数学特征 = matmul+softmax+matmul；综合类别 =
**Fusion（融合算子）**；workspace 配置 = False（Developer 消除写法）。
测试策略：dtype_count=2、attr 组合 = is_causal{F,T} × scaleValue{auto,显式}、
D 配置 3 档全覆盖、规则 shape 快速冒烟（≤10 用例）。

**L0 用例表**（block 配置：D∈{64,128}→(bM,bN)=(64,64)；D=256→(32,64)；
全部 shape 满足 S % bM == 0 且 S_kv % bN == 0；输入 = CPU 生成
`randn×U[-1,1]` 确定性种子 → H2D；golden = §9.1 代码 CPU 同 dtype 运行）：

| # | 用例名 | (B,S,S_kv,N,D) | dtype | scaleValue | is_causal | 验证点 |
|---|--------|----------------|-------|-----------|-----------|--------|
| 1 | l0_min_fp16 | (1,64,64,1,64) | float16 | -1.0（auto） | False | 最小冒烟：无掩码变体、单头、N=1 退化 pitch |
| 2 | l0_min_fp16_causal | (1,64,64,1,64) | float16 | -1.0 | True | causal 最小（offset=0 标准下三角） |
| 3 | l0_min_bf16_d128 | (1,64,64,1,128) | bfloat16 | -1.0 | False | bf16 路径 + D=128 |
| 4 | l0_min_bf16_d128_causal | (1,64,64,1,128) | bfloat16 | -1.0 | True | bf16 + causal |
| 5 | l0_typical_fp16 | (2,128,128,8,64) | float16 | -1.0 | False | 用户示例 shape（多 batch 多头，cid 分解） |
| 6 | l0_typical_fp16_causal | (2,128,128,8,64) | float16 | -1.0 | True | 典型 + causal |
| 7 | l0_typical_bf16 | (2,128,128,8,64) | bfloat16 | -1.0 | False | 典型 bf16 |
| 8 | l0_typical_bf16_causal | (2,128,128,8,64) | bfloat16 | -1.0 | True | 典型 bf16 + causal |
| 9 | l0_kvlong_fp16_causal | (1,128,256,4,128) | float16 | -1.0 | True | S<S_kv（offset=128 右下角对齐，非标准下三角）|
| 10 | l0_d256_fp16_scale | (1,64,64,2,256) | float16 | 0.0625（显式>0） | False | D=256（bM=32 配置）+ scaleValue 显式分支 |

覆盖矩阵映射（供 Stage 2 coverage_check 衔接）：D-DTYPE-fp16（#1,2,5,6,9,10）、
D-DTYPE-bf16（#3,4,7,8）、D-PARAM-is_causal 两值（F:#1,3,5,7,10 / T:#2,4,6,8,9）、
D-PARAM-scaleValue 两分支（auto:#1-9 / 显式:#10）、D 三配置（64:#1,2,5-8 /
128:#3,4,9 / 256:#10）。L0 不含尾块/质数 shape（L1 强制项，Stage 2 扩展）。
kernel 路径映射（rev2）：single_kv_block 特化→#1,2,3,4,10（kv_iters=1）；
通用 online softmax 路径→#5,6,7,8,9（kv_iters≥2）；掩码 (F,T)→#1,3,5,7,10 /
(T,T)→#2,4,6,8,9；(F,F)/(T,F) 尾钳位变体留待 L1（需 S_kv%64≠0）。

**L0 通过标准（rev2-C1）**：10/10 用例按**修订后 §9.3 阻塞口径**判定通过——
混合容差（§9.3.2）全过 **且** `MARE_abs < 10×Threshold`；MERE/MARE_rel 为参考值
（同步计算并打印在 `[PRECISION_*]` 行，不阻塞、不触发升级仲裁）。
`python examples/mha/test_mha.py --level l0` 退出码 0。
（harness 同步说明：现行 `test_mha.py` 的 `eco_ok` 仍按 rev1 口径把 MERE 计入
阻塞——Stage 2 attempt2 须按本节口径同步判定逻辑，属实现层同步，不改用例表。）

**特殊值用例原则（L0 仅声明，不实施）**：本算子声明输入值域为**有限实数**，
NaN/Inf 输入不在支持域内（proto.yaml 同）。Stage 2 的 L2/Boundary 若探测特殊值，
必须遵循**位置敏感比较**：用"有限值 + 稀疏特殊值"混合的确定性输入（至少含一个
特殊值与一个有限值），先严格比对 `isnan`/正 inf/负 inf mask 完全一致，再仅对双方
均为有限值的位置套用 §9.3 混合容差；禁止以全 NaN/全 Inf 输入作为唯一特殊值用例
（数据错位可能被掩盖）；`allclose(equal_nan=True)` 不能替代显式 mask 检查。

### 9.3 精度标准

#### 9.3.1 生态算子精度标准（需求方指定的验收标准，MERE/MARE；rev2-C1 修订阻塞口径）

误差指标（对全部输出元素，fp32 域计算）：

```
MERE（平均相对误差）= mean( |actual - golden| / (|golden| + 1e-7) )
MARE_rel（最大相对误差）= max( |actual - golden| / (|golden| + 1e-7) )
MARE_abs（最大绝对误差）= max( |actual - golden| )
```

**阻塞判定口径（rev2，需求方用户已在 Primary 上下文裁决，2026-09-18，不可更改）**：

```
阻塞判定 = §9.3.2 混合容差全过  且  MARE_abs < 10 × Threshold
MERE 与 MARE_rel 为参考值：同步计算、打印在 [PRECISION_*] 行，不阻塞、
不触发升级仲裁（rev1 的 [MARE_REL_ESCALATION] 升级流程随本裁决废止）。
```

| dtype | Threshold | MARE_abs 阻塞线（10×Threshold） | MERE 参考线（Threshold） | MARE_rel 参考线（10×Threshold） |
|-------|-----------|--------------------------------|--------------------------|--------------------------------|
| float16 | 2⁻¹⁰ ≈ 9.77e-4 | **< 9.77e-3（阻塞）** | ~9.77e-4（参考） | ~9.77e-3（参考） |
| bfloat16 | 2⁻⁷ ≈ 7.81e-3 | **< 7.81e-2（阻塞）** | ~7.81e-3（参考） | ~7.81e-2（参考） |

（阈值结构与本仓生态先例 `examples/cann-bench/group_norm/example_group_norm.py:487-491`
逐字一致：`"float16": (2**-10, 10*2**-10), "bfloat16": (2**-7, 10*2**-7)`；该生态
实现将 MARE 落为**最大绝对误差**（:571-580）——rev2 阻塞口径（混合容差 + MARE_abs）
与生态实现口径一致。）

**修订依据（数学证明 + 实测下界，debug_log.md attempt1）**：rev1 的阻塞门限
`MERE < Threshold` 对多 KV 块（S_kv > block_N=64）shape **数学不可达**——
对"理想下界"（完美 fp32 kernel → 舍入到 dtype → 对照同 dtype §9.1 golden）的
实测表明，golden 自身的半精度舍入噪声在近零输出元素上已超 MERE 门限：

| 用例 | dtype | 理想下界 MERE | rev1 门限 | attempt1 实测 kernel MERE |
|------|-------|---------------|-----------|---------------------------|
| case5 (2,128,128,8,64) | fp16 | 1.64e-3 | 9.77e-4 | 2.23e-3 |
| case9 (1,128,256,4,128) causal | fp16 | 1.80e-3 | 9.77e-4 | **1.78e-3（已达下界）** |
| case7 (2,128,128,8,64) | bf16 | 1.23e-2 | 7.81e-3 | 1.72e-2 |
| case8 同 causal | bf16 | 9.88e-3 | 7.81e-3 | 1.16e-2 |

机制：MERE 均值被近零 golden 元素主导（1e-7 clamp 分母病态；注意力输出含符号
穿越，随机数据下必然存在——case5 中 |g|<1e-4 的元素占 0.14% 却贡献 56% 的
MERE，该桶平均相对误差 0.96）。golden 在这些元素上的自身舍入噪声已超门限，
故任何 online softmax kernel（P 预归一化舍入，无法复刻 torch 归一化后舍入的
位级决策）都不可能通过——除非逐位复制 torch 舍入路径，算法上不可能。这正是
rev1 §10.1-R5 已为 MARE_rel 识别的缺陷类（同一 clamp 分母病态），rev1 仅对
MARE_rel 免疫而未覆盖 MERE 均值，属设计层口径缺陷。

**实现正确性的独立证据（口径修订不掩盖实现问题）**：单 KV 块用例在 kv_iters==1
预归一化特化（§1.4 步骤 7，rev2-A3）下位级可追踪——L0 cases 1/2/3/4/10 以
MERE 2.2e-7..2.9e-6 通过（≥335× 余量），case4（bf16 causal）**位级精确**
（max_abs=0.0）；混合容差（§9.3.2）与 MARE_abs 在全部 10 个 L0 用例（含 5 个
多块用例）通过。多块用例的 MERE 实测值处于/贴近理想下界（case9: 1.78e-3 ≤
下界 1.797e-3），即实现已达该算法路径的精度最优。

#### 9.3.2 混合容差标准（用例级阻塞判定，harness `check_precision` 实现；rev2 维持不变）

> 逐元素 `|actual-golden| ≤ atol + rtol·|golden|`，整体判定
> `matched_ratio ≥ required_matched_ratio` **且** `max_abs_error ≤ max_abs_error_limit`。
> 阈值仅按 dtype（与算子类别无关）；inf/nan 位置做结构比对不计入数值容差
> （完整定义见 tilelang-op-test-design/references/precision-standard.md）。
> §4 声明的每个 dtype 均有对应行；本算子无整型 dtype。
> rev2-C1：本表与阈值**原样保留**，与 `MARE_abs < 10×Threshold`（§9.3.1）共同
> 构成修订后的完整阻塞口径；attempt1 实测 10/10 全过。

| dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|-------|------|------|---------------------|------------------------|
| float16 | 2⁻¹⁴ (6.10e-5) | 2⁻⁹ (1.95e-3) | 1e-1 | 0.99 |
| bfloat16 | 2⁻¹⁰ (9.77e-4) | 2⁻⁶ (1.56e-2) | 1e0 | 0.99 |

#### 9.3.3 精度对齐设计（golden 栅格对齐，降低系统性偏差；rev2 修订论证）

1. **scores 双舍入镜像（rev2-A2）**（§1.4 步骤 3）：golden 的 scores 经历
   `round16(round16(qk)·s)` 两次半精度舍入，kernel 逐一镜像（r16(qk) → ×s(fp32)
   → r16(scores)），消除最大系统性误差源；2 的幂 scale 位级等价、D=128 ≤1ulp。
2. **P 以输入 dtype 进 GEMM2**：与 golden 的半精度 attn_weights 同栅格。
3. **fp32 累加/归一化**：GEMM fp32 累加（L0C）+ fp32 online 状态 + fp32 终归一，
   精度不低于 golden 的对应 fp32 内部累加路径。
4. **单 KV 块预归一化特化（rev2-A3）**：kv_iters==1 时 P̂ 归一化后再舍入，与
   golden 一次性 softmax 结构同构——位级可追踪（case4 bf16 位级精确、单块用例
   MERE ≤ 3e-6），为整个精度证据链提供与多块路径独立的锚点。
5. **阻塞口径下的余量（rev2-C1 修订）**：多块路径残差来源为 P 归一化位置差
   （golden 归一化后舍入 vs kernel 归一化前舍入）与最终输出舍入，量级 ~2⁻¹¹
   相对 / ~1e-4 绝对——在阻塞口径门限内余量 ≥1 个数量级（fp16 MARE_abs 门限
   9.77e-3 vs attempt1 实测 ≤1.2e-4，≥80×；混合容差 matched_ratio 实测
   ≥0.9985 ≥ 0.99；10/10 全过）。
   **rev1 勘误**：原文"在两套标准门限内余量 ≥1 个数量级"把该结论错误外推到
   MERE——MERE 是 1e-7 clamp 加权的相对误差**均值**，被近零 golden 元素主导，
   典型幅值相对误差（~2⁻¹¹）不代表其取值；实测连 golden 自身舍入噪声都超 MERE
   门限（§9.3.1 下界表）。故 rev2 阻塞口径不含 MERE（降为参考值），余量论证
   只对混合容差与 MARE_abs 成立——并以 attempt1 实测数据（而非估计）为据。

### 9.4 性能可行性哨兵（强制执行，不可因 large/L1 跳过；rev2-C4 重定义门限）

**rev1 缺陷与实测证据**：rev1 的 case 级超时门（prefill 600s，含 CPU golden）被
**测试侧 golden 成本**击穿而非算子性能——attempt1 实测 perf_prefill_worst 605s
超时，分相计时（decompose_prefill.py）：输入生成 6.3s / H2D 1.1s / 编译+首次
kernel 5.8s / **warm kernel 0.3s** / D2H 0.1s → **kernel 路径合计 13.3s ≪ 600s**；
CPU fp16 分块 golden 16 块 88.1s → 256 (b,n) 块外推 ≈1409s（torch CPU 无 fp16
GEMM SIMD 加速），单项即 2.3× 整个 case 预算。其余三条哨兵实测均在原门内
（decode case 级 47.6s/kernel 6.90s、d256_bf16 6.9s/kernel 5.75s、max_grid
28.4s/kernel 5.63s，16384 任务网格调度可行）。

**rev2 门限结构**（区分"算子 kernel 性能"与"测试 harness golden 成本"）：

1. **kernel 路径计时门（阻塞，可行性判定唯一计时依据）**：计时范围 =
   H2D → kernel（含该 shape 首次 JIT 编译）→ D2H + synchronize；每条哨兵
   ≤ **120s**（实测最坏 13.3s → ≥9× 余量；warm kernel 另记录作 Stage 3 基线）。
2. **精度判定（阻塞）**：按修订后 §9.3 阻塞口径（混合容差 + MARE_abs）；
   MERE/MARE_rel 打印为参考值。
3. **golden 侧降本规定（保障哨兵可执行，强制执行）**：
   - **分块 golden**：按 (b,n) 独立分块计算（batch/head 独立性保证与 §9.1
     整体式数学等价；harness `golden_mha_chunked` 已有实现），控制 CPU 峰值内存；
   - **缓存复用**：同 (shape, seed, dtype, attr) 的 golden 结果只计算一次，
     持久化缓存、跨 attempt/跨轮次复用（测试侧缓存文件，不进算子交付物、
     不影响判定语义）；
   - **确定性抽样比对（仅 prefill 哨兵允许）**：固定种子选取 ≥16 个 (b,n) 块对
     做全元素比对 + 其余块对每块按确定性步长抽样 ≥1024 元素；抽样集必须同时
     覆盖 causal 对角区（部分掩码行）与全可见区；阻塞口径在抽样集上判定，
     抽样方案（种子/步长）写入测试代码保证可复现；
   - **可选**：MERE/MARE_rel 参考值可另以 fp32-upcast golden 计算（仅影响参考
     值口径标注，阻塞判定一律以 §9.1 同 dtype golden 语义为准）。
4. **case 级墙钟参考上限 1800s（非阻塞）**：防 harness 挂死的保护性上限，
   超时只判 harness 缺陷（记录修复），**不判算子可行性失败**。

| 用例名 | (B,S,S_kv,N,D) | dtype/属性 | 覆盖路径 | kernel 路径门（阻塞） | attempt1 实测 kernel 路径 | 选择理由 |
|--------|----------------|-----------|----------|----------------------|---------------------------|----------|
| perf_prefill_worst | (8,2048,4096,32,128) | fp16, causal | 通用多块路径：最大 KV 重读（s_blocks=32）+ 最大实测规模 prefill | ≤120s | 13.3s（编译 5.8s + warm 0.3s） | GM 流量 ≈17.5GB、任务 8192×64 iters、DMA ≈6.8×10⁷ 事务（§5.5 核算） |
| perf_decode_worst | (64,1,4096,32,128) | fp16, causal | valid_m=1 尾块路径 + 最大 S_kv decode | ≤120s | kernel 6.90s（case 级 47.6s） | S=1 极端尾块 × 2048 任务 × 64 KV iters；63/64 行填充浪费的最坏情形 |
| perf_d256_bf16 | (2,1024,1024,16,256) | bf16, 非 causal | D=256（bM=32 配置）+ bf16 最大占用 | ≤120s | kernel 5.75s（case 级 6.9s） | L0C/UB/L1 最大占用配置 + bf16 GEMM 路径 |
| perf_max_grid | (128,128,128,64,64) | fp16, causal | 任务网格上限（16384 任务，B/N 支持域最大值） | ≤120s | kernel 5.63s（case 级 28.4s） | 任务调度/启动开销最坏；支持域边界 B=128、N=64 实测 |

哨兵判定：kernel 路径门超时 = **可行性失败** → 触发 Stage 1 回退（重选
block/裁剪策略）；精度阻塞口径失败 → Stage 2 精度流程处理。causal 无裁剪的
~2× 算力冗余已计入预算并实测吸收（threads=1 + T.serial 无预取下仍 ≥9× 余量）；
若未来 kernel 路径逼近门禁，启用 §10.3-①（KV 裁剪）/⑥（预取恢复）优化。

---

## 10. 风险点与注意事项

### 10.1 已知约束与风险

| # | 风险 | 等级 | 说明与缓解 |
|---|------|------|-----------|
| R1 | 掩码链 ops（arith_progression/compare/select）在 AUTO_CV 组合下的注册与正确性 | **已关闭（rev2）** | attempt1 实证：threads=1 + AUTO_CV + T.serial 下掩码链位级验证（verify_mask_chain/verify_chain_isolate 探针 + L0 causal 用例 2/4/6/8/9）；ops 均注册 PIPE_V（operation_config.h:184-237）。rev1 担忧的"Pipelined 组合"形态已随 C3 废止；vid 安全性问题随 threads=1 不复存在 |
| R2 | 4D 行跨步切片 `Q[bz, s0:s0+valid_m, by, :]` lowering 的 srcGap 正确性 | **已关闭（rev2）** | attempt1 最小复现验证位级精确：verify_bshd_slice.py（整除）+ verify_t1a.py（运行期 valid extent，S=1/30/64/90）+ L0 十用例复核；应急预案（按行 1D 循环搬运）未触发、保留备查（§5.4）。教训：threads=2 时代 W1/W2 探针"通过"为对称溢出巧合——判定一律以 threads=1 bisect 为准 |
| R3 | KV 循环内运行期 valid_n 切片（仅尾迭代 <block_N） | **已关闭（rev2）** | Pipelined 废止后该风险语境消失；T.serial + 运行期 valid_n 已验证（verify_kv_tail.py：尾列精确清零 ✓，尾块三重保障 §5.4 成立）。注意：**threads=2 下运行期 valid extent 会触发 ICHECK 编译崩溃**（§10.1-R8 症状②），本项关闭以 threads=1 为前提 |
| R4 | D=256 + bM=32 任务数翻倍 → KV 重读 ×2 | 低（性能） | §5.5 已计入预算且实测吸收（perf_d256_bf16 kernel 5.75s ≪ 120s 门）；Stage 3 可评估 bM=64+buffer 精简（≈187KB 声明量，参考实现 180.5KB 先例） |
| R5 | 生态相对误差口径（MERE/MARE_rel）对近零 golden 元素数学不可达 | **已解决（rev2-C1，用户裁决）** | rev1 仅对 MARE_rel 免疫、遗漏 MERE 均值同病态（1e-7 clamp 分母）——attempt1 以理想下界实测证明（§9.3.1 表：4/4 多块配置下界超门限，kernel 已达下界）。需求方用户已裁决：阻塞口径 = 混合容差（§9.3.2）+ MARE_abs < 10×Threshold；MERE/MARE_rel 降为参考值（打印不阻塞），升级仲裁流程废止。Stage 2 不得再以任何形式将 MERE/MARE_rel 计入阻塞 |
| R6 | causal 无 KV 裁剪 → ~2× 冗余算力 | 低（性能） | 正确性不受影响（全掩码块贡献 0）；实测吸收（prefill kernel 路径 ≥9× 门限余量）；Stage 3 优化项（§10.3-①，sink_bwd 运行期边界 T.serial 先例已解除技术障碍） |
| R7 | 每个 shape 组合一次 JIT 编译（B/S/S_kv/N 编译期参数化） | 低 | 本项目 attention 示例统一做法；tilelang 缓存 + 可选 lru_cache 包装；测试按配置分组复用编译产物；首次编译已计入 §9.4 kernel 路径门实测（5.8s） |
| R8 | **AscendVidReduction pass × threads=2 × BSHD 4D GM↔UB 拷贝 = 结构性 bug（rev2 新增，threads=2 禁用依据）** | 高（若误用 threads=2） | 三重症状（attempt1 bisect 实证，debug_log.md deviation#1）：① region [1,rows,1,D] 的行 extent 在 dim1，pass 按 target_dim=gm_dims−ub_dims=2（unit 头维）改写 → `ExtentIsEqualOne` 短路 → extent 不减半而 UB buffer 减半 → codegen 发射 `copy_ub_to_gm<...,32>(maskShapeM=64,…)` = **2× 行数 OOB 静默互踩**（非确定性；bisect 证实损坏行=恰为 vid1 分区）；② 运行期 valid extent → ascend_vid_reduction.cc:430 `ICHECK(ExtentIsEqualOne)` 编译崩溃；③ 1D 列维 buffer（col_pos/mask_row [block_N]）shape[0] 被减半而 arith_progression/select 不在 IsTileOp size 改写列表 → 64 元素写 32 槽位。参考实现 flash_attn_bshd_developer 免疫仅因其内部为 BHSD（行维=索引 2=target_dim）。**缓解/固化**：threads=1（cv_1_1）整体跳过该 pass（src/ir.cc:314-318），设计原样形式逐一位级验证（verify_t1a.py）。**恢复 threads=2 的前置条件**：该 pass 修复 BSHD 4D 支持后，复跑 /tmp/opencode 系 bisect/W1/W2/guard 探针 + 全量 L0 位级复核，并回报 orchestrator——在此之前任何 threads=2 改动视为设计违规。复现路径：debug_log.md pre-verification experiments 条目（verify_bisect_chain.py 等） |
| R9 | **threads=1 × T.Pipelined(num_stages=2) codegen 失败（rev2 新增，Pipelined 禁用依据）** | 中（若误用 Pipelined） | `Cannot find pre-allocated address for buffer: k_l1`（codegen_ascend.cc:830 ICHECK：MEMORY_PLANNING 双缓冲产生的 L1 gemm 操作数在 cv_1_1 下未注册 address_map）。attempt1 隔离复现：同 kernel pipe2 ✗ / serial ✓（verify_t1_loop_forms.py）。**缓解/固化**：T.serial 主方案（§6.3），性能损失已由哨兵量化吸收（kernel 路径 ≥9× 余量）；恢复预取走 §10.3-⑥ 三候选（手动双缓冲 / codegen 修复 / R8 修复后 threads=2+Pipelined），均以 §9.4 门复检为前提 |

### 10.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB 溢出 | D=256 误用 bM=64（≈203KB>192KB） | 编译/运行失败 | 按 §5.2 D-adaptive 配置；改动 block 前重算 §4.5 预算 |
| L0C 溢出 | bM×D×4B 超限（如 bM=64,D=512 时 acc_o_l0c=128KB 贴限） | segfault | 本设计 D≤256 已核算 ✅；扩 D 域必须重算 §5.3 |
| 尾列污染 softmax 分母 | S_kv 非整除且漏掉钳位掩码 | 精度系统性偏差（分母偏大） | §5.4 三重保障；need_bound_mask 编译期恒生效（变体 2/4） |
| 0×陈旧NaN=NaN 污染 GEMM2 | 假设尾块 buffer 为陈旧数据 | 有效行输出 NaN | 已由源码排除：GM→L1/UB 尾区确定性清零（§5.4 ①）；仍保留 P=0 掩码双保险 |
| 全掩码行 0/0=NaN | S>S_kv+causal（域外）或未来域扩展 | 输出 NaN ≠ golden 0 | sumexp+1e-30 安全除法（§1.4 步骤 7）；正常行 sumexp≥1 不受影响 |
| T.Parallel 内写条件分支 | 用 `if`/`T.if_then_else` 做元素级掩码 | 编译错误（SIMD 限制） | 已规避：掩码用 compare+select（api-compute §3.1 替代方案） |
| select SPR 模式 >64 元素静默错误 | 误用 VSEL_CMPMASK_SPR 于宽行 | 索引错位 | 固定使用 VSEL_TENSOR_SCALAR_MODE（无 64 元素上限） |
| int32 索引直接 compare | 用 int32 索引缓冲做 LT/GT | 退化为 EQ 静默错误 | col_pos 用 fp32（≤4096 « 2²⁴ 精确表示） |
| m_i_prev 覆盖时序错误 | `T.copy(m_i, m_i_prev)` 放在 reduce_max 之后 | 重标定因子恒 1，累加错误 | 伪代码已固定顺序：copy(m_i→m_i_prev) 在 reduce_max 之前（参考实现同序） |
| host 侧"顺手" contiguous/transpose | BSHD→BHSD 在 host 转置 | 违反 §5 约束（aclnn/物理拷贝），cann-bench 环境失败 | §4.8 审计锁定：host 零张量变换；布局适配全在 kernel 行跨步 DMA |
| 改回 threads=2 或残留 vid | 照搬 flash_attn_bshd_developer（内部 BHSD，免疫）threads=2 或旧 workspace+vid 示例代码 | AscendVidReduction × BSHD 4D 三重 bug：OOB 静默互踩（非确定性，精度测试可能漏检）/ICHECK 崩溃/1D buffer 减半（§10.1-R8） | threads=1 固化（cv_1_1）；§8.6 自检 grep：无 vid/workspace/Scope/barrier 残留；恢复 threads=2 须满足 R8 前置条件 |
| threads=1 + T.Pipelined | 照搬参考实现 num_stages=2 双缓冲 | codegen `Cannot find pre-allocated address for buffer: k_l1`（address_map 未注册，§10.1-R9） | KV 循环用 T.serial（编译期常量边界）；预取恢复走 §10.3-⑥ |
| TVMScript parse-time `if` 块内分配、块外使用 | `if need_mask:` 内 alloc col_pos，掩码链外引用 | "Undefined variable: col_pos" 编译错误（跨块变量作用域限制，attempt1 复现） | 跨块 buffer 无条件分配（§3.3/§4.3，rev2-A1）；语句级 `if PY_BOOL:` 块本身可用（parse-time 求值，已验证） |

### 10.3 特殊场景处理

- **S=1/2 decode**：valid_m=1/2，整 bM 行缓冲计算（垃圾行有限、行独立、写回裁剪）；
  causal 首行可见列 = [0, offset]=[0,S_kv-1] 全可见（offset=S_kv-1≥0），无全掩码行。
  性能注记：decode 下 (bM−1)/bM 行浪费为 flash 分块固有代价，专用 decode 路径
  （如 xattention 的 heads-as-M 重排）超出本算子范围，不在 v1 引入。
- **全零输入退化**（需求实测场景 [0,0]）：scores=0 → causal/非 causal 掩码后
  可见列均匀 softmax → 输出 = 可见 V 行均值（V 亦全零 → 输出全零）；sumexp ≥ 1，
  无除零。golden 同路径同结果 ✅（L1 用例，Stage 2 落地）。
- **causal 且 S=S_kv**：offset=0，退化为标准下三角（L0 case2/6/8 覆盖）。
- **causal 且 S<S_kv**：offset>0 右下角对齐（L0 case9 覆盖；gqa_fwd_varlen test
  的右对齐语义对照已核）。
- **N=1**：行 pitch = D，行跨步退化为连续搬运 ✅（L0 case1-4）。
- **Stage 3 性能优化候选清单**（正确性无关，勿在 v1 引入；每项落地后须复跑
  §9.4 kernel 路径门 + L0 阻塞口径）：
  ① causal KV 运行期裁剪（`eff_kv_iters = floordiv(min(bx*bM+bM-1+offset, S_kv-1), bN)+1`，
  mha_sink:268-271 先例；主循环已是 T.serial，运行期边界可用性已有 sink_bwd:323-324
  threads=1 Developer 先例——技术障碍已解除，~2× causal 收益）；
  ② 全可见块的掩码链运行期跳过（mha_sink_fwd_bhsd.py:502 条件先例）；
  ③ 行循环 → broadcast 2D 形式（sink_bwd:178-207 先例，指令数 ~bM×2→6）；
  ④ D=256 的 bM=64 精简 buffer 方案；
  ⑤ golden 舍入对齐简化——**rev2 警示**：双舍入镜像（§1.4 步骤 3）是单块路径
  位级精确（case4 MERE=0.0）与多块路径达理想下界的对齐来源，移除/简化前必须
  重证 MARE_abs 与混合容差余量，且不得影响 §9.3.1 参考值可比性；
  ⑥ **KV 预取恢复（rev2-C3 新增）**：(a) T.serial + 手动 k/v 双缓冲（ping-pong
  索引，L1 84→148KB 仍 ✅ §4.5）；(b) 推动 codegen address_map 修复（§10.1-R9）
  后恢复 T.Pipelined；(c) VidReduction BSHD 修复（§10.1-R8）后评估 threads=2 +
  Pipelined 全流水——(c) 须先复跑 R8 探针套件。

---

## 11. 交付清单

### 11.1 目录结构

```
examples/mha/
├── mha.py            # 纯 kernel（@tilelang.jit mha_kernel + host 包装 def mha(...)），可 import，无 golden/测试/__main__
├── test_mha.py       # from mha import mha + §9.1 golden + L0 用例（§9.2）+ 哨兵（§9.4）+ main(--level)
├── proto.yaml        # 算子接口规格（dtype/attr），供覆盖门禁派生应覆盖维度
├── DESIGN.md         # 本设计文档
└── README.md         # 使用说明（可选）
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成（本文件，rev2） | 设计文档（含修订记录 §0、L0 门槛测试计划、修订后阻塞口径） |
| `proto.yaml` | ✅ 已完成（rev2 复核无变化） | dtype 全集 = §9.3.2 表（float16, bfloat16）；attrs = scaleValue(Float) + is_causal(Bool)（§11.5）——接口规格与 rev1 一致 |
| `mha.py` | ✅ Stage 2 attempt1 已产出（rev2 设计已对齐） | 纯 kernel + host 包装；与本设计 §3.3 伪代码同形（threads=1/T.serial/双舍入/无条件分配/单块特化），**无需改动** |
| `test_mha.py` | ✅ Stage 2 attempt1 已产出（判定逻辑待 attempt2 同步） | golden（§9.1 原样）+ L0 十用例 + 哨兵 + `--level` main 已就绪；attempt2 按 §9.2/§9.3.1 修订口径同步 `eco_ok` 判定（MERE 移出阻塞）并按 §9.4 新门限结构改造哨兵（kernel 路径计时 + golden 降本），再扩展 L1/L2/Boundary |
| `debug_log.md` | ✅ Stage 2 attempt1 证据记录 | rev2 全部修订的实测证据出处（下界测定/bisect/分相计时） |
| `history_version/` | ✅ 回退备份 | design_rev1.md（被回退设计）、proto_rev1.yaml、mha_impl_s2_attempt1.py（attempt1 实现快照） |

### 11.3 命名规范

- 目录名: `mha`（snake_case）✅ 已存在
- kernel 文件: `mha.py`；测试文件: `test_mha.py`（顶部 `from mha import mha`）
- kernel JIT 函数: `mha_kernel`；host 包装: `mha(query, key, value, scaleValue=-1.0, is_causal=False)`

### 11.4 实现顺序

1. ✅ 设计文档（DESIGN.md rev2）+ proto.yaml（复核无变化）+ L0 门槛测试计划（§9.2，用例表不变、判定口径修订）
2. ✅ Stage 2 前置验证：4D 行跨步切片最小复现（§5.4，已关闭）+ 最小 causal 用例先行（attempt1 完成）
3. ✅ kernel 实现（`mha.py`，attempt1 产出，rev2 设计已对齐、无需改动）
4. ✅ 测试文件骨架（`test_mha.py`：golden 原样 + L0 十用例 + 哨兵 + main，attempt1 产出）
5. ⬜ Stage 2 attempt2：测试判定逻辑同步修订口径（§9.2/§9.3.1：MERE/MARE_rel 移出阻塞）+ 哨兵改造（§9.4：kernel 路径计时门 + golden 降本规定）→ L0 门槛重判（10/10 预期通过，attempt1 已实证混合容差+MARE_abs 全过）+ 哨兵门复检
6. ⬜ 扩展分层套件（L1 功能/L2 异常/Boundary 特殊值，tilelang-op-test-design 场景 B；L1 须覆盖 (F,F)/(T,F) 尾钳位变体与三轴非对齐）+ `coverage_check.py --proto` 覆盖门禁
7. ⬜ 全量套件运行（L0/L1 须通过；L2/Boundary 失败仅记录不阻塞）

### 11.5 算子 proto.yaml（覆盖门禁用，Stage 1 产出）

已机械派生至 `examples/mha/proto.yaml`：dtype 全集取自 §9.3.2 精度表
（float16、bfloat16——§4.1 仅给代表性 dtype，不作全集来源）；attrs 取自 §1.3/§4.1
（scaleValue: Float, default -1.0；is_causal: Bool, default false）——两者均影响
计算路径（scale 解析分支 / 掩码 4 变体），供 `coverage_check.py --proto` 派生
D-DTYPE-*/D-PARAM-* 覆盖维度。内容与 §9.3.2/§4 一致性已自检 ✅。

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

## 12. 性能目标（Stage 3 调优，2026-09-18 追加）

> 本节为 orchestrator 按用户确认追加的 Stage 3 调优目标，不修改既有任何章节。

- **目标类型**: `baseline_compare` —— 对齐 cann-bench 评测（`tasks/level4/mha`，hardware=910b2）
- **对比基线**: CANN `FusedInferAttentionScore`（`tasks/metadata/910b2.json` 的 `baseline_perf_us`；`t_hw_us` 为硬件理论上界锚点）
- **当前锚点数据**（cann_final_eval_20260918_173752，未调优实现）:
  - 综合得分 54.18/100（编译 20/20，精度 30/30，性能 4.18/50）
  - 平均加速比 0.13x（区间 0.026x~0.31x；最差 = case7/8 大 prefill causal 16.8ms vs baseline 433us）
- **验收方式**: 每个优化项用 `msprof op` kernel 级计时验证（禁止 Python/Torch 计时）+ 精度复验（`test_mha.py --level l0/l1`）不退化；阶段性以 cann-bench 复评（`run_evaluation.sh --task-dir tasks/level4/mha --operator MHA`）确认加速比/HAP 提升
- **噪声阈值**: 3%（提升 < 3% 视为无提升）
- **中止条件**: ① 迭代达到 10 轮上限；② 连续 3 次优化无提升（<3%）；③ 达到用户目标（加速比显著逼近基线，由 orchestrator 与用户确认）
- **测试 shape**: §9.4 四条哨兵 + cann-bench 20 cases 的代表子集（case1 小 prefill / case7 大 prefill causal / case10 decode / case14 decode 大 B / case20 全零）
- **硬约束提醒**（沿用 §10.1-R8/R9）: threads=2（AscendVidReduction × BSHD 4D bug）与 T.Pipelined（cv_1_1 codegen address_map bug）在框架修复前禁用；预取恢复走"手动双缓冲 + T.serial"路径（§10.3-⑥）；causal KV 裁剪（§10.3-①，预期 ~2×）可用运行期边界 T.serial（先例 example_gqa_sink_bwd_bhsd.py:323-324）
