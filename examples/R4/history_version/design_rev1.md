# r4_fused_sampling 算子设计文档 —— Part V01-b（Gumbel-max 融合采样增量）

> **文档定位**：本文是 R4 融合采样链 **V01-b 增量变体**的设计文档，聚焦 V01-b 相对 V01-a 的
> **delta**，并自足到能让 Stage 2 developer 直接实现。共享基础设施（K1 `r4_topk_stats`、
> online-softmax、分层 top-k 归并、UB 预算模型、运行/复现约束、§10.4 草稿修正清单、V01-a 已验证的
> 实现范式）**引用 V01-a 的 `examples/R4/DESIGN.md`（V01.1，2062 行，已在真实 NPU 验证通过）与
> `examples/R4/r4_fused_sampling.py`（378 行）**，本文不重复其已验证内容。
>
> **为什么单独成文（而非在 DESIGN.md 追加 Part II）**：① V01-a 的 `DESIGN.md` 是**已验证冻结的交付物**，
> 任务硬性要求「不得删改其已验证内容」——单独成文彻底规避误改风险；② V01-b 是 delta，独立文档更聚焦、
> 对 Stage 2 更自足；③ 2062 行文档再插 Part II 会变得难以维护。本文与 `DESIGN.md` 并列，交叉引用。
>
> **权威输入（本文全部结论基于以下已读工件）**：`DESIGN_script.md`（原始草稿，§3.5 Step7b / §3.6 T3 /
> §3.7 Pass2 V01-b / §5.4 K2 / §5.3.4 LCG / D3 / D6 / §8.3 / §9.2 / §10 R1/R13 / §13.3）、
> `DESIGN.md`（V01-a V01.1）、`r4_fused_sampling.py`、`r4_ref.py`、`debug_log.md`（10 项实现层坑）、
> `examples_experiment/random_1d/{design.md,random_1d.py}`（LCG 参考）、以及**已安装 wheel**
> `tilelang 0.1.4+d75300ed`（`~/.local/lib/python3.12/site-packages/tilelang/language/ascend_tile.py`）
> 的源码核实。

---

## 0. V01-b 范围与 V01-a 资产复用

### 0.1 V01-b 目标（已与用户确认）

给 K2 增加 **`FUSE_GUMBEL=True`** 编译期开关路径，在 kernel 内用 **Gumbel-max trick（定理 T3）**
直接输出采样 token id `tok_out[B] int32`，**消除 host 侧 `torch.multinomial`**（草稿 D6：6 op → 2 op；
访存 §8.3：V01-a 594 MB → V01-b 297 MB）。

### 0.2 保留 / 不变 / 新增

| 组件 | V01-b 处置 | 说明 |
|------|-----------|------|
| **K1 `r4_topk_stats`** | 🟢 **完全不变** | V01-b 复用其 `tau`/`valid`（以及 `m`/`s`，但 Gumbel 路径**不需要** `m`/`s`，见 T3）。keep 集 `S_i = {j : z_ij >= tau_i}` 直接来自 K1 的 `tau_out` |
| **K2 `r4_mask_sample`（`FUSE_GUMBEL=False`）** | 🟢 **完全不变** | V01-a 的 `w_out[B,V]` 路径逐字保留（已验证），不动其代码 |
| **K2 Gumbel 路径（`FUSE_GUMBEL=True`）** | 🔴 **新增** | 新 kernel 函数 `r4_gumbel_sample`，输出 `tok_out[B] int32`（见 §3.3） |
| **host wrapper `r4_fused_sampling`** | 🟡 **扩展** | 增 `FUSE_GUMBEL` 分派：`True` → K1 + `r4_gumbel_sample` → `tok_out`（无 multinomial）；`False` → V01-a 路径不变 |
| **`r4_ref.py`（golden）** | 🟡 **扩展** | 增「忠实 Gumbel-max 参考」（bitwise，§9.1）；V01-a 的 `golden_r4` 不变 |
| **`proto.yaml`** | 🟡 **扩展为超集** | inputs 增 `rand`，outputs 增 `tok_out`，attrs 增 `FUSE_GUMBEL`（§11.6） |

### 0.3 out of scope（用户明确排除）

确定性 seed 路径（float64 Gumbel，草稿 D3/R1 —— TileLang-Ascend 对 float64 零支持）、SGLang 集成、
logprob 融合、ACLGraph 入图、完整多特化表、bf16。**V01-b 仅 fp32 + 单特化 S1'（`K_MAX=256, BV=4096`）**，
代表性 `V=152064`，与 V01-a 一致。

---

## 1. 概述

### 1.1 算子名称

`r4_fused_sampling`（V01-b 增量）—— 新增 Gumbel-max 融合采样 kernel **`r4_gumbel_sample`**。

### 1.2 功能描述

在 V01-a「softmax → top-k → top-p → min-p 融合 + 输出屏蔽后概率 `w_out`」的基础上，V01-b 用
**Gumbel-max trick** 把「采样」也融进 K2：对保留集 `S_i = {j : z_ij >= tau_i}` 直接做
`y_i = argmax_{j∈S_i}[z_ij + G_ij]`，**在 z 域一步得到 token id**，免除 `exp/div/log/归一化`（定理 T3），
并免除 host `torch.multinomial`。

**计算语义**（记第 `i` 行、vocab 位置 `j`，`z = logits`；K1 已产出 `tau_i`）：

```
# ---- Gumbel 噪声（LCG 生成 u，再反变换）----
state_ij = seed_i + j                                   # 每元素独立（random_1d 范式）
state_ij = A*state_ij + C   (×3 轮, int32 环绕 mod 2^32) # LCG，A=1103515245, C=12345
u_ij     = clamp( state_ij(as fp32) * 2^-32 + 0.5 ,  2^-24, 1 - 2^-24 )   # → (0,1)，f32 安全
G_ij     = -ln( -ln( u_ij ) )                            # Gumbel(0,1)（定理 T3 / 草稿 Step7b）
# ---- Gumbel-max（z 域直接 argmax，免归一化）----
cand_ij  = (z_ij >= tau_i) ? (z_ij + G_ij) : -inf        # 被屏蔽元素以 -inf 自然排除
y_i      = argmax_j cand_ij                              # = tok_out[i]
```

**适用场景**：与 V01-a 同（LLM 推理 decode 每 step 的输出级采样，纯访存瓶颈）。V01-b 进一步消除
`multinomial` 的一遍全 vocab 读（148.5 MB）与其 host 调度，把采样收敛为 kernel 内一遍 argmax。

### 1.3 数学公式

**定理 T3（Gumbel-max 免除归一化与对数；草稿 §3.6，本文复核并沿用）**

对保留元素 `j∈S_i`，`ln w_ij = z_ij - m_i - ln s_i + const_i`，其中 `-m_i - ln s_i` 为**行内常量**，
不改变 argmax：

```
y_i = argmax_j [ ln w_ij + G_ij ] = argmax_{j ∈ S_i} [ z_ij + G_ij ]
```

**收益**：Pass 2 无需 `exp/div/log`，直接在 z 域 argmax ⟹ K2 Gumbel 路径**不读 `m_in`/`s_in`**
（只读 `tau_in` 做 keep 掩码 + `rand` 做噪声）。被屏蔽元素（`z < tau`）以 `-inf` 参与 argmax 自然排除。

**Gumbel 分布正确性**：若 `G_ij ~ Gumbel(0,1)` i.i.d.，则 `argmax_j[p_ij 的对数 + G_ij]` 服从
`Categorical(p_i)`（Gumbel-max trick 标准结论）。T3 把「对数概率」换成等价的 `z_ij`（差行内常量），
故 `y_i ~ Categorical(softmax(z_i) 限制在 S_i 上并重归一化)` = 目标采样分布。⟹ **分布一致性可验（§9.5）**。

**LCG（线性同余生成器，random_1d 范式）**：`state ← (A·state + C) mod 2^32`，`A=1103515245, C=12345`，
初值 `state = seed_i + j`（每元素独立、可完全并行），迭代 3 轮。int32 溢出自动截断（mod 2^32），与
`ctypes.c_int32` 语义一致（`random_1d.py:76-78` 的 `reference_random_1d` 给出精确 golden，`atol=0/rtol=0`）。

### 1.4 算法描述

**Pass 2（K2 Gumbel，`FUSE_GUMBEL=True`；单次 GM 读 `logits` + 标量读 `tau`/`rand` + 标量写 `tok_out`）**：

```
init: best_val ← -inf,  best_idx ← 0            # 跨块 running argmax（[1] UB buffer）
for b in 0 .. n_blk-1:                           # n_blk = ceildiv(V, BV) = 38（静态）
    z ← GM[logits[row, b*BV : (b+1)*BV]]         # T.copy（gm2ub，尾块自动 clamp + pad_value=-inf）
    tau ← GM[tau_in[row]]                         # 行标量（1 elem）
    # (a) LCG 生成本块噪声（int32，元素独立）
    state ← arith_progression(seed_row + b*BV, 1, BV)   # = seed_i + 全局 idx
    for _ in range(3): state ← A*state + C               # int32 环绕（T.tile.mul/add）
    # (b) Gumbel 反变换（fp32）
    g ← cast(state → fp32); g ← g*2^-32 + 0.5; g ← clamp(g, 2^-24, 1-2^-24)
    g ← -ln(g); g ← -ln(g)                        # 两次 T.tile.ln + 两次取负（T.tile.mul by -1）
    # (c) 屏蔽 + 候选（z 域，定理 T3）
    keep ← compare(z, tau, "GE")                  # T.tile.compare（scalar src1，无需 broadcast）
    cand ← select(keep, z + g, -inf)              # T.tile.add 后 T.tile.select（scalar -inf）
    # (d) 块内 argmax-with-index（topk K=1）
    topk(topk_dst, cand_rank1, K=1, BV)           # topk_dst = [blk_val, blk_local_idx, ...]（fp32 交错）
    blk_val ← topk_dst[0];  blk_idx ← topk_dst[1] + b*BV   # gather_mask/cast 或直接标量读（§3.4 D-ARGMAX）
    # (e) 跨块归并（严格 GT → 平票取更小 index，与 torch.argmax 一致）
    if blk_val > best_val:  best_val ← blk_val;  best_idx ← blk_idx
tok_out[row] ← cast(best_idx → int32)            # 标量写 GM（1 elem）
```

**为什么必须分块 argmax + 跨块归并（不可全 vocab 一次 argmax）**：与 V01-a 同理——`V=152064` fp32
= 594 KB ≫ 184 KB 可用 UB，`T.tile.topk` 要求 src 全量驻留 UB ⟹ 必须分块。argmax 满足**结合律**
（`max(A∪B) = max(max A, max B)`），故分块 argmax + 跨块「取更大者」归并**精确等价**于全行 argmax
（归纳证明同 V01-a §1.4 分层归并，K 从 K_MAX 退化为 1）。∎

### 1.5 数据流图

```
GM logits[B,V] --T.copy(分块 BV, pad -inf)--> UB z_ub[BV]
GM tau_in[B]   --标量读(1 elem/行)---------> UB t_v[1]        (来自 K1，keep 阈值)
GM rand[B]     --标量读(1 elem/行)---------> UB seed_v[1]     (每行 LCG 种子)
                                                |
   state_ub[BV](int32) <-- arith_progression(seed_v[0]+b*BV,1,BV) + 3x(mul A, add C)  [LCG]
                                                | cast(int32->fp32)
   g_ub[BV](fp32) <-- *2^-32 +0.5, clamp[2^-24,1-2^-24], ln, neg, ln, neg            [Gumbel]
                                                |
   keep_mask[BV/8](uint8) <-- compare(z_ub, t_v[0], "GE")
   cand: z_ub <-- add(z_ub, g_ub); select(z_ub, keep_mask, z_ub, -inf)               [z 域候选]
                                                | (z_ub 已是 rank-1 [BV]，直接喂 topk)
   topk_dst[8](fp32) <-- T.tile.topk(z_ub, K=1, BV)   [块内 argmax, 用 tmp arena]
                                                | blk_val=topk_dst[0], blk_idx=topk_dst[1]+b*BV
   best_val[1],best_idx[1] <-- if blk_val>best_val: 更新    [跨块归并, 循环 n_blk 次]
                                                | cast(best_idx -> int32), 标量写
                                          GM tok_out[B](int32)
   ======================= K1（不变）/ K2 Gumbel 边界 =======================
   K1 r4_topk_stats: GM logits,top_ks,top_ps,min_ps -> GM m_out,s_out,tau_out,valid   (V01-a 原样)
```

> **Host 侧 Buffer 操作约束合规**（ascend-constraints.md §5）：V01-b host 路径只做「张量分配（含
> `rand[B]` 种子准备）+ 前置校验 C3/C6/C7 + K1 调用 + K2 Gumbel 调用 + `valid` 分派」，**无**
> `.contiguous()`/`reshape` 物理化/`torch.cat`/`.to(dtype)`/aclnn 计算。相比 V01-a，V01-b **消除了
> host `torch.multinomial`**（V01-a DEV-1 偏差），host 侧更干净。审计见 §5.6。

---

## 2. 编程模式选型

### 2.1 模式结论

**沿用 V01-a 的 Developer 主体 + Ascend 专属 `T.tile.*` 混合形态**（用户在 V01-a 已明确确认接受，
V01-b 不重新问、不改模式——见 `.orchestrator_requirements.json` 的 carry_over_rationale）。

具体落地（与 V01-a §2.1 一致）：
- 内存分配：**`T.alloc_shared`**（不用 `T.alloc_ub`/`T.alloc_L0*`）；标量用 `[1]`/`[1,1]` `alloc_shared` buffer
- 作用域：**不写 `T.Scope("C"/"V")`**（`TL_ASCEND_AUTO_CV_COMBINE` 自动处理）
- 同步：**不写 `T.barrier_all`/`T.set_flag`/`T.wait_flag`/`T.pipe_barrier`**（`TL_ASCEND_AUTO_SYNC` 自动插入）
- pass_configs：四个自动类开关**全开**（§7.3，与 V01-a 同）
- 计算表达：高层 `T.copy` + Ascend 专属 `T.tile.*`（无高层等价物处）

### 2.2 选型理由（V01-b 的每个新步骤都落在已认可的混合家族内）

V01-a §2.3/§10.5 已论证「Developer 主体 + `T.tile.topk`/`gather_mask`」是仓库既有范式
（`examples/developer_mode/gelu_mul_developer.py` 就在 `alloc_shared` + Developer pass_configs 下用
`T.tile.*`），且 Expert 判别标志（`alloc_ub`/`T.Scope`/手动 flag/pass_configs 全关）**一条都不用**。
V01-b 新增的三类操作**全部属于同一已认可家族**：

| V01-b 新步骤 | 用到的 `T.tile.*` | 是否属已认可混合家族 | 证据 |
|-------------|------------------|---------------------|------|
| **LCG 随机数** | `arith_progression` / `add` / `mul`（int32） | ✅ 是 | `random_1d.py`（Developer pass_configs，仅分配用 `alloc_ub`——V01-b 改 `alloc_shared`，见 §3.5.2）；`attention_sink/...:162`、`per_block_cast_lossless_kernel.py:771`（arith_progression 生产用例） |
| **Gumbel 反变换** | `cast` / `mul`(scalar) / `add`(scalar) / `clamp` / `ln` | ✅ 是 | `act_quant.py:78-84`（`clamp`→`round`→`cast` 生产链，in-place fp32）；`ln` 见 `example_cross_entro.py:99`、V01-a K1 `T.tile.ln(lnmu, muc)` |
| **argmax-with-index** | `topk`(K=1) / `gather_mask` / `cast` / `compare` / `select` | ✅ 是 | `topk_selector.py:48-50`、`moe_topk_gate.py:66-69`（topk+gather_mask+cast 生产链，V01-a K1 已用 topk+gather_mask）；`compare`/`select` 见 V01-a K2 |

**结论**：V01-b 全链路（LCG + Gumbel + argmax）可在 Developer 混合形态实现，**无需任何 Expert 判别标志**。

### 2.3 模式影响

| 维度 | V01-b 的选择（与 V01-a 一致） |
|------|------------------------------|
| 内存分配 | `T.alloc_shared`（纯 Vector ⟹ `InferAllocScope` 解析为 `shared.ub`，V01-a §2.3 论据 2）；标量 `[1]`/`[1,1]` buffer |
| 计算方式 | `T.copy` + `T.tile.*`（arith_progression/add/mul/cast/clamp/ln/topk/gather_mask/compare/select/fill）+ 跨块归并的 `with T.If/T.Then` 标量更新 |
| 作用域 | 编译器自动分离（`AUTO_CV_COMBINE`），不写 `T.Scope` |
| 同步方式 | 全自动（`AUTO_SYNC`），零手写 barrier/flag |
| Kernel 轴 | `T.Kernel(b_num, is_npu=True) as (cid, vid)` + `row = cid*VEC_NUM + vid`（**不用 `threads=2`**，V01-a §6.5 的 4 条硬约束理由对 V01-b 同样成立：topk 需整块驻留单核 UB、topk 无 per-row 语义、纯 Vector 无 workspace 可消、仓库纯 Vector 生产实现一致形态） |
| CV 交互 | 不涉及（无 Cube）⟹ 无 workspace、无 `workspace_idx`、无跨核 flag |

### 2.4 Developer 混合模式可行性最终判定

**判定：V01-b 全链路（LCG + Gumbel + argmax-with-index）可在 Developer 混合形态实现，无任何步骤
只能用 Expert 判别标志完成。✅ 无矛盾点需上报。**

- LCG（int32 element-wise mul/add + arith_progression）⟹ `T.tile.*`，Developer `alloc_shared`，`random_1d`
  已证（其唯一 Expert 痕迹是 `alloc_ub`，V01-b 改 `alloc_shared`，见 §3.5.2）✅
- Gumbel（cast/mul/add/clamp/ln）⟹ `T.tile.*`，`act_quant.py` 生产链已证 ✅
- argmax-with-index ⟹ `T.tile.topk(K=1)` + `gather_mask`/`cast`（V01-a K1 已用 topk+gather_mask 于
  `alloc_shared`，V-1 上板 PASS）✅
- 跨块 argmax 归并 ⟹ `with T.If(blk_val>best_val): with T.Then():` 标量更新（V01-a `tau_k`/`prefix`
  同款，V-3 上板 PASS）✅

**唯一残留的术语级说明**（同 V01-a R-A，非阻塞）：`topk`/`gather_mask`/`arith_progression`/`clamp`/`ln`
无纯 Developer 高层等价物，属「Developer 主体 + Ascend 专属 `T.tile.xxx`」混合模式（guide §1 明言这是
大多数实际算子的形态）。**未使用任何 Expert 判别标志，未擅自改模式。**

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 | 所在 kernel |
|------|----------|------|------------|
| G1 | `state_j ← seed_i + (b*BV + local_j)` | LCG 初值（每元素独立） | K2 Gumbel |
| G2 | `state ← A*state + C`（×3，int32 环绕） | LCG 3 轮 | K2 Gumbel |
| G3 | `u ← clamp(state_fp32 * 2^-32 + 0.5, 2^-24, 1-2^-24)` | int32→fp32→(0,1) 映射 + f32 安全 clamp | K2 Gumbel |
| G4 | `G ← -ln(-ln(u))` | Gumbel(0,1) 反变换（两次 ln + 两次取负） | K2 Gumbel |
| G5 | `keep_j ← (z_j >= tau_i)` | 单阈值掩码（复用 K1 的 tau） | K2 Gumbel |
| G6 | `cand_j ← keep_j ? (z_j + G_j) : -inf` | z 域候选（定理 T3，免归一化） | K2 Gumbel |
| G7 | `(blk_val, blk_local_idx) ← argmax_j cand_j` | 块内 argmax-with-index | K2 Gumbel |
| G8 | `blk_idx ← blk_local_idx + b*BV` | 局部→全局 index | K2 Gumbel |
| G9 | `if blk_val > best_val: (best_val,best_idx) ← (blk_val,blk_idx)` | 跨块归并（严格 GT） | K2 Gumbel |
| G10 | `tok_out[i] ← cast(best_idx → int32)` | 标量写 GM | K2 Gumbel |

### 3.2 TileLang API 映射（Developer 混合形态）

> 每行均给出**已核实的证据**（wheel 源码行号 / 示例 / 测试）。遵循 AGENTS.md 核心原则 1：
> ① `tilelang-api-best-practices` → ② `examples/` 同类实现 → ③ wheel 源码 + 官方测试。**无凭记忆猜测项。**
> wheel = `~/.local/lib/python3.12/site-packages/tilelang/language/ascend_tile.py`（提交 `d75300ed`，与仓库源码仅 docstring 差异）。

| 步骤 | TileLang API（Developer 混合） | 参数要点 | 证据（已核实） | 替代方案 |
|------|-------------------------------|---------|---------------|---------|
| Grid / 任务映射 | `T.Kernel(b_num, is_npu=True) as (cid, vid)`；`row = cid*VEC_NUM+vid`；`if row < B` | `b_num=T.ceildiv(B,VEC_NUM)`，`VEC_NUM=2` | **V01-a `r4_fused_sampling.py:306-308`（K2 已验证）**；`topk_selector.py:32-33` | 见 §2.3（不用 `threads=2`） |
| UB 分配 | `T.alloc_shared(shape, dtype)` | rank-1 `[BV]` 用于 topk 源 / LCG / Gumbel | V01-a §2.3 论据 2（`ascend_infer_buffer_scope.cc:398-402` 纯 Vector→`shared.ub`） | `T.alloc_ub`（Expert，不用；见 §3.5.2 对 random_1d 的改写） |
| 分块搬入（含尾块） | `T.copy(logits[row, b*BV], z_ub, pad_value=-T.infinity("float32"))` | GM 2D 标量起点 → 1D UB；尾块自动 clamp + pad | **V01-a `r4_fused_sampling.py:326`（K2 已验证）**；`topk_selector.py:43`；尾块机制 `test_..._tail_block.py:18-31` | 显式 BufferRegion |
| 行标量读（tau/seed） | `T.copy(tau_in[row:row+1], t_v[0,0:1])` / `T.copy(rand[row:row+1], seed_v)` | GM→UB 1 元素（**GM↔UB 的 1-elem copy 正常**，UB→UB 才禁） | **V01-a `r4_fused_sampling.py:320-322`（已验证）**；`debug_log.md` 发现 5 | 标量 `t_v[0]=tau_in[row]` |
| **LCG 索引生成** | `T.tile.arith_progression(state_ub, seed_v[0] + b*BV, 1, BV)` | 生成 `[seed+b*BV, ..., seed+b*BV+BV-1]`（int32）；`first_value` 可为运行时 PrimExpr | wheel `:389`；`random_1d.py:55`；`attention_sink/...:162`（`q_row+v_row` 复合运行时）；`per_block_cast_lossless_kernel.py:771`（int32 buffer，运行时 first_value） | 两步：`arith_progression(state_ub, b*BV,1,BV)` + `add(state_ub, state_ub, seed_v[0])` |
| **LCG 3 轮（int32 环绕）** | `T.tile.mul(state_ub, state_ub, A)` + `T.tile.add(state_ub, state_ub, C)`（in-place，×3） | `A=1103515245, C=12345`（int32 标量）；int32 mul/add **自动 mod 2^32 环绕** | wheel `:1048`(mul,src1 可 PrimExpr)/`:1026`(add)；`random_1d.py:59-61`（golden `ctypes.c_int32` `atol=0/rtol=0` 精确匹配 ⟹ **环绕语义已证**）；in-place `T.tile.*` 见 V01-a `r4_fused_sampling.py:159-160` | random_1d 原样：分离 `temp_ub` + 填充 `a_ub`/`c_ub` buffer（多占 32 KB，仅作 fallback） |
| **int32→fp32** | `T.tile.cast(g_ub, state_ub, "CAST_RINT", BV)` | int32→fp32；mode 对 int→float 无实质影响（无小数部分），大整数按 RNE 舍入 | wheel `:1927`；`topk_selector.py:50`/`moe_topk_gate.py:69`（fp32→int32 用 `CAST_ROUND`）；`act_quant.py:82-84`（cast 生产链） | `"CAST_NONE"`（待 Stage 2 实测择一，见 V-b3） |
| **u 仿射映射** | `T.tile.mul(g_ub, g_ub, 2.3283064365386963e-10)` + `T.tile.add(g_ub, g_ub, 0.5)` | `2^-32`（fp32 精确，2 的幂）；in-place | wheel `:1048`/`:1026`（src1 可 PrimExpr 标量）；标量运算见 V01-a `r4_fused_sampling.py:219`（`T.tile.max(muc, mu_i, 1e-30)`） | `mul_add_dst`（wheel `:1340`，FMA 一步；待实测） |
| **f32 安全 clamp** | `T.tile.clamp(g_ub, g_ub, 5.9604644775390625e-08, 0.9999999403953552, BV)` | `[2^-24, 1-2^-24]`（**两端均 fp32 精确可表示**，见 §3.4 D-CLAMP）；in-place | wheel `:2228`；**`act_quant.py:78`（`T.tile.clamp(x_ub_fp, x_ub_fp, -127.0, 127.0, count)` 生产 in-place fp32）** | `clamp_min`(`:2181`)+`clamp_max`(`:2134`) 两步 |
| **两次 ln** | `T.tile.ln(g_ub, g_ub)`（in-place，×2，中间夹取负） | 自然对数；in-place 安全（V01-a `T.tile.exp(e_ub,e_ub)` 已证 in-place） | wheel `:1211`；api-compute.md `:306`；`example_cross_entro.py:99`、V01-a `r4_fused_sampling.py:220` | — |
| **取负（×2）** | `T.tile.mul(g_ub, g_ub, -1.0)` | in-place 乘 -1 标量 | wheel `:1048`（src1 可 PrimExpr）；标量运算见 V01-a `:219` | `T.tile.sub(g_ub, zero_buf, g_ub)` |
| **keep 掩码** | `T.tile.compare(keep_mask, z_ub, t_v[0,0], "GE")` | `src1` 可为标量 BufferLoad ⟹ `tl.ascend_compare_scalar`，**无需 broadcast t2d** | wheel `:1837`（`:1894` PrimExpr/float → compare_scalar）；V01-a `r4_fused_sampling.py:333`（用 broadcast t2d，本文优化为 scalar） | V01-a 形态：`broadcast(t2d,t_v)`+`compare(mask,z_ub,t2d,"GE")` |
| **z+G** | `T.tile.add(z_ub, z_ub, g_ub)`（in-place，z_ub 变 cand） | z 域候选（定理 T3） | wheel `:1026`；in-place 见 V01-a `:159` | 分离 cand buffer |
| **屏蔽 select** | `T.tile.select(z_ub, keep_mask, z_ub, -T.infinity("float32"), "VSEL_TENSOR_SCALAR_MODE")` | `src1` 标量 -inf；in-place | wheel `:707`（`:816` PrimExpr→TENSOR_SCALAR_MODE）；**V01-a `r4_fused_sampling.py:334`（`select(w_ub,mask,e_ub,0.0,"VSEL_TENSOR_SCALAR_MODE")` 已验证）** | — |
| **块内 argmax-with-index** | `T.tile.topk(topk_dst, z_ub, 1, BV)` | K=1；`topk_dst ≥ aligned_topk(K=1)=ceil(2/8)*8=8` fp32；输出 `[blk_val, blk_idx, ...]` 交错；src 必须 rank-1 静态 shape；`BV%32==0`✓ | wheel `:550`（`:558` 输出交错 (value,index)）；**`test_..._topk.py:324-327`（`test_topk_k_boundary` K∈{1,64} fp32 已测）**；`:140-141`（`[0::2]`=val、`[1::2]`=idx）；`sort` docstring wheel `:424-426`（idx=0-based 原位）；`topk_selector.py:48`、`moe_topk_gate.py:66` | ~~`wholereducemax(ORDER_VALUE_INDEX)`~~ **已 `@deprecated()`（wheel `:1528`），禁用**（§3.4 D-ARGMAX） |
| **取 (val,idx) 分量** | 主选：`blk_val←topk_dst[0]`、`blk_idx←topk_dst[1]`（标量读）；备选：`gather_mask("P0101"/"P1010")`+`cast` | K=1 时直接标量读最简；idx 为 fp32（`<BV=4096<2^24` ⟹ 精确整数） | 标量读：V01-a `r4_fused_sampling.py:214`（`run_v[k_i[0]-1]` 动态标量读，V-3 PASS）；gather 备选：wheel `:608`（`"P0101"`=偶=值、`"P1010"`=奇=idx），`topk_selector.py:49`、`moe_topk_gate.py:68`、`test_..._topk.py:169-170` | — |
| **跨块 argmax 归并** | `with T.If(topk_dst[0] > best_val[0]): with T.Then(): best_val[0]←topk_dst[0]; best_idx[0]←topk_dst[1]+b*BV` | 严格 GT ⟹ 平票取更小 index（=torch.argmax 首现，§3.4 D-TIE-ARGMAX） | **V01-a `r4_fused_sampling.py:229-233`（`with T.If(prefix<=p_i): with T.Then():` 已验证）**；`debug_log.md` 发现 10（`T.If` 必配 `T.Then`）；标量赋值 `:214` | `T.tile.compare("GT")`+`T.tile.select`（`[1]` buffer，V-b4 待实测） |
| **best_idx→int32 写 GM** | `T.tile.cast(tok_i32, best_idx, "CAST_ROUND", 1)` + `T.copy(tok_i32, tok_out[row:row+1])` | `best_idx<V=152064<2^24` ⟹ fp32 精确 ⟹ cast 无损；1-elem UB→GM copy 正常 | wheel `:1927`；`topk_selector.py:50`；1-elem UB→GM 见 V01-a `r4_fused_sampling.py:284-287` + `debug_log.md` 发现 5 | 直接以 int32 维护 best_idx（省末尾 cast） |
| 初始化 | `T.tile.fill(best_val, -T.infinity("float32"))`；`T.tile.fill(best_idx, 0)` | running argmax 初值 | wheel `:343`；V01-a `r4_fused_sampling.py:137-139` | — |
| 循环 | `T.serial(n_blk)` 分块；`range(3)` LCG（编译期常量） | trip count 全静态 | V01-a `r4_fused_sampling.py:324`；`random_1d.py:59` | `T.Pipelined`（§6.3 不启用） |

**明确禁用 / 排除的 API（均有 wheel 源码证据）**

| API | 状态 | 证据 |
|-----|------|------|
| `T.tile.wholereducemax` / `wholereducemin` / `wholereducesum`（含 `ReduceOrder="ORDER_VALUE_INDEX"`） | 🔴 **wheel `@deprecated()`** | wheel `ascend_tile.py:1528/1545/1562`（`ReduceOrder` 默认在 `:1537`）。**草稿 §3.6/§3.7 的 argmax-with-index 核心原语** ⟹ V01-b 改用 `T.tile.topk(K=1)`+`gather_mask`/标量读（§3.4 D-ARGMAX） |
| `T.tile.block_reduce_max` / `block_reduce_min` | 🔴 **`@deprecated()`** | wheel `:1711/1753` |
| `T.reduce_max`（返回值无 index） | ⚠️ 不适用 | 只给 max **值**不给 **index** ⟹ 无法做 argmax-with-index；故必须用 `topk(K=1)` |
| `T.cumsum` | 🔴 Ascend 不可用 | V01-a §3.5.1（`src/op/reduce.cc:320-341`）；V01-b Gumbel 路径**不需要** cumsum（argmax 非前缀和） |
| `T.tile.atomic_add` | 🔴 不用 | destination must be GM（wheel `:333`）；V01-b「一行一任务」无跨核归约 |
| `T.alloc_ub` / `T.Scope("V")` / `T.barrier_all` / `T.set_flag` / `T.wait_flag` / `T.pipe_barrier` | 🔴 Expert 标志，不用 | `random_1d.py`/`moe_topk_gate.py` 用了这些 ⟹ V01-b **改写为 `alloc_shared` + 自动同步**（§3.5.2） |
| `tilelang.cache.clear_cache()` / `disable_cache()` | 🔴 生产禁用 | `random_1d.py:7` 有 `clear_cache()`（**演示用，V01-b 禁止照抄**，见 §11.5 R-JIT） |
| `float64` / `double` | 🔴 零支持 | V01-a §3.5.1；草稿 D3/R1 ⟹ 确定性 seed 路径排除（V01-b 亦不涉及） |
| `tilelang/language/pto.py` | 🔴 已废弃 | AGENTS.md 核心原则 1 |

---

### 3.3 计算伪代码（Developer 混合形态）

```python
import tilelang
from tilelang import language as T

# 沿用 V01-a 的常量与 pass_configs（r4_fused_sampling.py:27-35）
R4_PASS_CONFIGS = {"tl.ascend_auto_sync": True, "tl.ascend_memory_planning": True,
                   "tl.ascend_auto_cv_combine": True, "tl.ascend_auto_cross_core_sync": True}
VEC_NUM = 2
LCG_A, LCG_C = 1103515245, 12345           # random_1d.py:20-21
INV_2_32 = 2.3283064365386963e-10          # 2^-32（fp32 精确）
U_MIN, U_MAX = 5.9604644775390625e-08, 0.9999999403953552   # 2^-24, 1-2^-24（fp32 精确）

def _aligned_topk(K, elems_per_block=8):    # V01-a r4_fused_sampling.py:56-58
    return ((2 * K + elems_per_block - 1) // elems_per_block) * elems_per_block


@tilelang.jit(out_idx=[3], pass_configs=R4_PASS_CONFIGS)
def r4_gumbel_sample(B: int, V: int, BV: int):
    """K2 Gumbel 路径（FUSE_GUMBEL=True）：z 域 Gumbel-max argmax -> tok_out[B] int32。
    不读 m_in/s_in（定理 T3）；只读 logits + tau_in + rand（每行 LCG 种子）。"""
    n_blk = T.ceildiv(V, BV)                # 静态 38（V=152064, BV=4096）
    b_num = T.ceildiv(B, VEC_NUM)
    AT1 = _aligned_topk(1)                  # = 8（K=1, fp32）

    @T.prim_func
    def main(
        logits : T.Tensor([B, V], "float32"),   # in
        tau_in : T.Tensor([B],    "float32"),   # in（来自 K1）
        rand   : T.Tensor([B],    "int32"),     # in（每行 LCG 种子；§3.4 D-RAND）
        tok_out: T.Tensor([B],    "int32"),     # out
    ):
        with T.Kernel(b_num, is_npu=True) as (cid, vid):
            row = cid * VEC_NUM + vid
            if row < B:
                # ---- 全部 alloc_shared（纯 Vector → shared.ub）；rank-1 [BV] 作 topk 源（N2）----
                z_ub    = T.alloc_shared([BV],        "float32")   # logits 块 → 就地变 cand
                state_ub= T.alloc_shared([BV],        "int32")     # LCG 状态（in-place 3 轮）
                g_ub    = T.alloc_shared([BV],        "float32")   # Gumbel 噪声（cast dst + 就地变换）
                mask    = T.alloc_shared([BV // 8],   "uint8")     # keep 掩码
                topk_dst= T.alloc_shared([AT1],       "float32")   # [blk_val, blk_idx, ...]（K=1）
                t_v     = T.alloc_shared([1, 1],      "float32")   # tau_i
                seed_v  = T.alloc_shared([1],         "int32")     # seed_i
                best_val= T.alloc_shared([1],         "float32")   # running argmax 值
                best_idx= T.alloc_shared([1],         "float32")   # running argmax 全局 index（<2^24 精确）
                tok_i32 = T.alloc_shared([1],         "int32")     # 输出标量

                # 触及每个 buffer（scope 推断，debug_log.md 发现 3）
                T.tile.fill(best_val, -T.infinity("float32"))
                T.tile.fill(best_idx, 0.0)
                T.tile.fill(tok_i32, 0)
                T.copy(tau_in[row : row + 1], t_v[0, 0:1])         # GM→UB 1-elem（正常）
                T.copy(rand[row : row + 1], seed_v)

                for b in T.serial(n_blk):                          # 静态 38；无单独尾块路径（§6.4）
                    # (0) 搬入 logits 块（尾块自动 clamp + pad -inf）
                    T.copy(logits[row, b * BV], z_ub, pad_value=-T.infinity("float32"))
                    # (a) LCG：state = seed_i + 全局 idx，3 轮 int32 环绕
                    T.tile.arith_progression(state_ub, seed_v[0] + b * BV, 1, BV)
                    for _ in range(3):
                        T.tile.mul(state_ub, state_ub, LCG_A)      # int32 环绕 mod 2^32
                        T.tile.add(state_ub, state_ub, LCG_C)
                    # (b) Gumbel：u = clamp(state*2^-32+0.5)，G = -ln(-ln(u))
                    T.tile.cast(g_ub, state_ub, "CAST_RINT", BV)   # int32 → fp32
                    T.tile.mul(g_ub, g_ub, INV_2_32)
                    T.tile.add(g_ub, g_ub, 0.5)
                    T.tile.clamp(g_ub, g_ub, U_MIN, U_MAX, BV)
                    T.tile.ln(g_ub, g_ub)                          # ln(u)  （u∈(0,1) → 负）
                    T.tile.mul(g_ub, g_ub, -1.0)                   # -ln(u) （正）
                    T.tile.ln(g_ub, g_ub)                          # ln(-ln(u))
                    T.tile.mul(g_ub, g_ub, -1.0)                   # G = -ln(-ln(u))
                    # (c) keep 掩码 + z 域候选（定理 T3）
                    T.tile.compare(mask, z_ub, t_v[0, 0], "GE")    # scalar src1 → compare_scalar
                    T.tile.add(z_ub, z_ub, g_ub)                   # z + G（就地）
                    T.tile.select(z_ub, mask, z_ub, -T.infinity("float32"), "VSEL_TENSOR_SCALAR_MODE")
                    # (d) 块内 argmax-with-index（topk K=1；z_ub 已是 rank-1 [BV]，直接喂）
                    T.tile.topk(topk_dst, z_ub, 1, BV)             # topk_dst = [blk_val, blk_local_idx, ...]
                    # (e) 跨块归并（严格 GT → 平票取更小 index）
                    with T.If(topk_dst[0] > best_val[0]):
                        with T.Then():
                            best_val[0] = topk_dst[0]
                            best_idx[0] = topk_dst[1] + b * BV     # 局部 → 全局（fp32 精确）
                # 写 GM（best_idx fp32 → int32）
                T.tile.cast(tok_i32, best_idx, "CAST_ROUND", 1)
                T.copy(tok_i32, tok_out[row : row + 1])            # UB→GM 1-elem（正常）

    return main


def r4_fused_sampling(logits, top_ks, top_ps, min_ps, need_min_p_sampling, rand=None,
                      K_MAX=256, BV=4096, FUSE_GUMBEL=False):
    """host wrapper：C3/C6/C7 前置校验 + K1 + K2（按 FUSE_GUMBEL 分派）。
    FUSE_GUMBEL=False → V01-a 路径（w_out，逐字不变）；True → V01-b（tok_out，无 multinomial）。"""
    # ... V01-a 的 C6/C7/C3/dtype 前置校验（r4_fused_sampling.py:356-372）原样保留 ...
    k1 = r4_topk_stats(B, V, K_MAX, BV, bool(need_min_p_sampling))
    m_out, s_out, tau_out, valid = k1(logits, top_ks, top_ps, min_ps)   # K1 不变
    if not FUSE_GUMBEL:
        w_out = r4_mask_sample(B, V, BV)(logits, m_out, s_out, tau_out)  # V01-a K2 不变
        return m_out, s_out, tau_out, valid, w_out                       # host 保留 multinomial
    # V01-b：Gumbel 融合采样（rand 必需；每行 LCG 种子）
    tok_out = r4_gumbel_sample(B, V, BV)(logits, tau_out, rand)
    return m_out, s_out, tau_out, valid, tok_out                         # valid==0 的行 host 回退 torch
```

> **伪代码说明**
> 1. `z_ub`/`state_ub`/`g_ub` 均为 **rank-1 `[BV]`**：topk 源必须 rank-1（`max_actual_num=Σshape` 陷阱，
>    V01-a N2/R-C）；gm2ub `T.copy(logits[row, b*BV], z_ub)` 对 rank-1 UB 有效（`topk_selector.py:43` 同款）。
> 2. **LCG/Gumbel 全部 in-place**（`state_ub`、`g_ub`、`z_ub` 各自就地变换）⟹ 省 buffer（V01-a `T.tile.exp(e_ub,e_ub)`
>    已证 in-place 安全）。in-place LCG 省掉 random_1d 的 `temp_ub`/`a_ub`/`c_ub`（§3.4 D-RAND）。
> 3. `T.copy` 于 **GM↔UB 的 1 元素**（`tau_in[row:row+1]→t_v`、`tok_i32→tok_out[row:row+1]`）正常；
>    **禁止 1 元素 UB→UB `T.copy`**（4B<32B DMA 对齐 → aicore 异常 507015，`debug_log.md` 发现 5）——
>    本设计的标量更新全用 `T.tile.*` / `T.If+T.Then` 标量赋值，无 UB→UB 1-elem copy。
> 4. `best_val`/`best_idx`/`tok_i32`/`t_v`/`seed_v` 等标量 buffer 至少被一个 `T.tile.fill`/`T.copy`/`T.If` 赋值
>    触及（避免「仅 T.Parallel 触及的 buffer 作用域无法解析」，`debug_log.md` 发现 3）。
> 5. `T.If` 必配 `with T.Then():`（`debug_log.md` 发现 10；V01-a `r4_fused_sampling.py:230-231` 同款）。
> 6. **无 `T.reduce_*`**：V01-b argmax 走 `topk(K=1)`，不用 `T.reduce_max/sum` ⟹ **完全规避 V01-a V-2 的
>    narrow-reduce ≤256B 硬约束**（`codegen_ascend.cc:2145`）与尾块精确尺寸缓冲的复杂度（§6.4）。
> 7. **FUSE_GUMBEL 的落地形态**：推荐**独立 kernel 函数 `r4_gumbel_sample`** + host 分派（保 V01-a
>    `r4_mask_sample` 逐字不变，最安全）；等价备选是单函数 `r4_mask_sample(B,V,BV,FUSE_GUMBEL)` 内
>    `if FUSE_GUMBEL:` 编译期分支（会改动已验证函数签名，次选）。二者都是「编译期开关」，Stage 2 择一。

### 3.4 关键设计决策（V01-b 相对草稿的实质改进）

#### D-ARGMAX —— argmax-with-index 用 `topk(K=1)`，不用 deprecated 的 `wholereducemax(ORDER_VALUE_INDEX)`

草稿 §3.6/§3.7 用 `T.tile.wholereducemax(cand, ReduceOrder="ORDER_VALUE_INDEX")` 做块内 argmax。
**该 API 在已安装 wheel 中已 `@deprecated()`**（`ascend_tile.py:1528`，默认 `ReduceOrder="ORDER_VALUE_INDEX"`
在 `:1537`）——与 V01-a 弃用 `wholereduce*` 同一结论（V01-a §3.5.2 / R-DEPR）。

**替代（已核实可行）**：`T.tile.topk(dst, src, K=1, actual_num)` 输出 `[val0, idx0, ...]` 交错对
（wheel `:558` docstring；`sort` docstring `:424-426` 明示 idx = 0-based 原位）。K=1 即块内 argmax-with-index。
- **K=1 已被官方测试覆盖**：`test_..._topk.py:324-327 test_topk_k_boundary`，`@parametrize("K",[1,64])`，
  dtype `"float"`（fp32），`run_test_topk_basic(128, 64, K=K)` ⟹ K=1 fp32 编译+运行+精度已测。
- **index 精确性**：topk 输出的 idx 以 fp32 存储；本算子 idx < `V=152064 < 2^24` ⟹ **fp32 精确表示整数**
  （`test_..._topk.py:17` 的「float16 index precision loss for actual_num>2048」是 **float16** 限制，fp32 无此问题）。
- **tie-break = 最小 index**：`moe_topk_gate.py:347-349` 的 golden `stable_topk = torch.sort(descending,stable=True)[:num_topk]`
  且 `:408 np.testing.assert_equal`（精确相等）⟹ topk 平票返回**最小 index**。⟹ 跨块用**严格 GT** 归并
  （`blk_val > best_val` 才更新）时，平票保留更早块（更小全局 index）⟹ **整体 = 全行最小 index argmax
  = `torch.argmax` 首现语义**（D-TIE-ARGMAX）。
- **提取 (val,idx)**：K=1 时最简是**直接标量读** `topk_dst[0]`（val）、`topk_dst[1]`（idx）（V01-a `run_v[k_i-1]`
  动态标量读已证，V-3 PASS）；备选 `gather_mask("P0101"/"P1010")`+`cast`（`topk_selector.py:49-50` 生产链）。
- **topk tmp arena**：即使 K=1，fp32 topk 仍自动注入 `4×AlignUp(BV,32)×4B = 64 KB` tmp arena（arena 取决于
  src 尺寸 BV，非 K）——已计入 §4.5 UB 预算。

#### D-RAND —— 随机数方案：**每行种子 `rand[B] int32` + kernel 内联 LCG**（决策 + 分歧点）

草稿对此**自相矛盾**：§3.2 说 `rand[B,BV] int32` 是 host 侧 `random_1d` 生成的种子流输入；§5.3.4/§3.7 说
kernel 内联 LCG（`lcg(rand[i,b])`）；而 §8.3 的访存核算把 V01-b 记为 **「≈297 R = 2 读」**（K1 读 logits +
K2 读 logits），**未计入任何 `[B,V]` 或 `[B,BV]` rand 的 GM 读**。三者不自洽。任务 risk#4 明确要求「定清楚」。

**本文决策：Design C —— host 传每行标量种子 `rand[B] int32`，kernel 内联 LCG 3 轮生成完整 `[B,V]` 噪声流。**
理由（四条，均指向 Design C 最优）：

| 维度 | Design C（每行种子 `[B]` + 内联 LCG）✅ **采纳** | Design D（host 传 `[B,V]` 随机流） |
|------|-----------------------------------------------|-----------------------------------|
| **访存（§8.3 一致）** | K2 只读 logits（148.5 MB）+ 标量种子 ⟹ **总 2 读 = 297 MB**，与草稿 §8.3 吻合 | K2 读 logits + rand[B,V] ⟹ **3 读 = 445 MB**，**丢失 V01-b 的访存收益** |
| **统计正确性** | `state=seed_i+j`（j 遍历全 vocab）⟹ 每位置独立噪声（random_1d 范式，元素独立） | 同样每位置独立（host 生成） |
| **bitwise 可复现** | golden 用 `reference_random_1d(V, seed_i)`（`ctypes.c_int32` 精确 int32）逐行复现 ⟹ LCG 部分**逐位精确** | golden 需复现 host 的 [B,V] 生成（若 host 用 random_1d kernel 则又要复现其并行序） |
| **生产/测试** | 生产：每请求一个 seed（scheduler 给）；测试：固定小 seed ⟹ 完全确定 | 生产需 host 生成大随机流（额外开销）；测试需固定 [B,V] 流 |
| **UB / 实现** | 内联 LCG 用 `state_ub[BV] int32`（16 KB），复用 random_1d 已证范式 | 省 LCG buffer，但增 GM 读 + host 生成 |

⟹ **Design C 胜出**：唯一同时满足「§8.3 的 2 读访存收益」「bitwise 可复现」「random_1d 范式复用」的方案。
`rand` 输入 shape 定为 **`[B] int32`（每行一个 LCG 种子）**，kernel 用 `arith_progression(seed_i + b*BV, 1, BV)`
展开为块内全局索引，再 3 轮 LCG。

> ⚠️ **分歧点（上报 Orchestrator/用户，见 §10.4）**：任务的 proto 提示写的是 `rand[B,BV] int32`，本文按
> Design C 定为 **`rand[B] int32`**（每行标量种子）。这是**有意偏离**，依据是草稿 §8.3 的访存核算本身
> 就要求内联 LCG（否则丢失 V01-b 的核心收益），且 `[B,BV]`/`[B,V]` 的 host 随机流会引入第三遍 GM 读。
> 若用户坚持 `[B,BV]`（或 `[B,V]`）host 生成流（Design D），需接受 V01-b 访存退化到 3 读（≈445 MB，
> 仍优于 V01-a 的 594 MB，因省了 w_out 写 + multinomial 读）。**默认按 Design C 实现。**

> **LCG 参数 A/C 的定位**：`A=1103515245, C=12345` 是 **固定常量**（random_1d 的经典 ANSI C 参数），
> **非用户可调 attr**；作为 kernel 内 Python 常量硬编码（不进 JIT 特化键、不进 proto attrs 的可变维度）。
> 每行 `seed_i` 是运行时张量元素（`rand[B]`），亦不进特化键。

#### D-UMAP —— u 映射：`u = state_fp32 · 2^-32 + 0.5`（修正草稿 `·2^-31` 的符号 bug）

草稿 §3.7 写 `u = lcg(rand) · 2^-31`。但 LCG 输出是 **signed int32** ∈ `[-2^31, 2^31)`，`·2^-31` 映射到
`[-1, 1)`——**负半轴（约 50% 的样本）会被 clamp 到下界 `2^-24`**，得到**同一个** Gumbel 噪声 ⟹ **严重破坏
采样分布**（半数候选共享最小噪声）。这是草稿的实质 bug。

**修正（本文决策）**：`u = state_fp32 · 2^-32 + 0.5`，把 signed int32 `[-2^31, 2^31)` **仿射映射到 `[0, 1)`**：
- `state=-2^31 → u=0`（clamp 到 `2^-24`）；`state=0 → u=0.5`；`state=2^31-1 → u≈1`（clamp 到 `1-2^-24`）。
- LCG 输出近似均匀 ⟹ u 近似均匀 `[0,1)` ⟹ Gumbel 正确。
- **等价写法**：`u = (state + 2^31) · 2^-32`（把 signed 视作 biased unsigned）。选 `·2^-32 + 0.5` 因其只需
  fp32 `mul`(2 的幂，精确) + `add`(0.5)，无需 int32 加偏置（避免 `state+2^31` 的 int32 溢出）。
- **精度**：`state_fp32` 对 `|state|>2^24` 有 RNE 舍入 ⟹ u 有约 2^24 个不同值（对 V=152064 候选**远超**足够）；
  `2^-32`（2 的幂）与 `0.5` 在 fp32 精确 ⟹ 仿射映射无额外误差（除 cast 舍入）。
- **为何不用 `bitwise_and` 取高位**：任务 risk#3 问「是否需要 `T.tile.bitwise_and`」——**不需要**。仿射
  `·2^-32+0.5` 已把全 32 位 signed 值域均匀映射到 [0,1)，无需位掩码提取；int32 环绕由 mul/add 硬件自动完成
  （random_1d 已证），亦无需 `bitwise_and` 做 mod 2^32。

#### D-CLAMP —— f32 Gumbel 数值稳定性：clamp 到 `[2^-24, 1-2^-24]`（消解草稿 D3/R1）

草稿 D3/R1 指出：f32 下若 clamp 上界用 `1-2^-32`，则 `1-2^-32` 在 f32 舍入为 `1.0`，`ln(1.0)=0`，
`-ln(0)=+inf` ⟹ G=+inf 数值错误（这是排除**确定性 float64 路径**的理由）。

**V01-b 的非确定性 f32 路径用 `[2^-24, 1-2^-24]` 消解此问题**（任务指定，本文核实）：
- `2^-24 = 5.9604644775390625e-08` 与 `1-2^-24 = 0.9999999403953552` **均为 fp32 精确可表示**
  （`1-2^-24` 恰是 fp32 中 `<1.0` 的最大值：`[0.5,1)` binade 的 ULP = `2^-24`）。
- `u∈[2^-24, 1-2^-24]` ⟹ `ln(u)∈[-16.635, -5.96e-8]`（有限负）⟹ `-ln(u)∈[5.96e-8, 16.635]`（有限正）
  ⟹ `ln(-ln(u))∈[-16.635, 2.811]`（有限）⟹ **`G=-ln(-ln(u))∈[-2.811, 16.635]`（全程有限，无 ±inf/nan）**。
- **统计影响可忽略**：Gumbel CDF `F(x)=exp(-exp(-x))`；下界截断 `F(-2.811)≈6e-8`、上界截断
  `1-F(16.635)≈6e-8` ⟹ 合计约 `1.2e-7` 概率失真，远低于分布检验（§9.5）的 `p>0.01` / `KL<1e-3` 门槛。
- **`T.tile.ln` 对 clamp 后输入的行为**：输入恒 `[5.96e-8, 16.635]`（正、远离 0 与 inf）⟹ ln 良态；
  in-place `T.tile.ln(g_ub, g_ub)` 安全（V01-a `T.tile.exp(e_ub,e_ub)` in-place 已证）。

#### D-TIE-ARGMAX —— argmax 平票：严格 GT 归并 ⟹ 最小 index（=torch.argmax 首现）

`z+G` 中 G 为连续随机噪声 ⟹ 精确平票测度为零（L0 连续 `randn` logits 下实际不发生）。为 bitwise 稳健：
- **块内**：`topk(K=1)` 平票返回最小 index（D-ARGMAX 已证 = `torch.sort(stable)`）。
- **跨块**：`with T.If(blk_val > best_val)`（**严格 GT**）⟹ 平票时保留更早块（更小全局 index）。
- 合成 ⟹ **全行最小 index argmax**，与 golden 的 `torch.argmax(cand, dim=-1)`（首现）**逐位一致**。

---

## 3.5 技术约束确认

### 3.5.1 本项目已知限制检查（复用 V01-a §3.5.1 + V01-b 专项）

| 约束 | V01-b 是否涉及 | 处理方案 |
|------|---------------|----------|
| **不支持三维 Kernel** | No | `T.Kernel(b_num, is_npu=True)` 一维 grid；`as (cid,vid)` 是同 block 内 Cube/Vector 核解构，非第三维（V01-a 同） |
| **threads 参数限制（仅 1/2）** | No | 不指定 `threads`（§2.3；V01-a §6.5 的 4 条硬约束对 V01-b 同样成立） |
| **动态循环边界不支持** | Yes → 已消解 | `T.serial(n_blk)`、`range(3)`（LCG）trip count 均由编译期常量 `V`/`BV` 导出，全静态；`B` 只影响 grid + 行守卫 |
| **流水线不支持动态边界** | No（不启用流水线） | §6.3 |
| **部分 GPU API 不可用** | Yes → 已消解 | argmax 不用 `T.cumsum`；`wholereducemax` deprecated ⟹ 用 `topk(K=1)`（D-ARGMAX） |
| **L0C 容量上限（128KB）** | N/A | 纯 Vector，零 matmul，不用 L0A/L0B/L0C |
| **GEMM 非整除** | N/A | 无 GEMM |
| **`T.copy` 列方向 strided 切片** | No | 全部行内连续切片 `logits[row, b*BV]`（最内维连续） |
| **Host 侧 reshape/aclnn 隐式调用** | No（且优于 V01-a） | host 只做张量分配 + 校验 + kernel 调用 + valid 分派；**V01-b 消除了 V01-a 的 host `torch.multinomial`（DEV-1）** ⟹ host 侧零 aclnn 计算（§5.6 审计） |
| **`T.tile.topk` 对 fp32 in-place 写 -inf padding** | Yes → 已消解 | 作用于 UB scratch `z_ub`（非 GM）；`BV%32==0` ⟹ `AlignUp(actual_num,32)<=BV` 无越界（V01-a N3；`test_..._topk.py:181-183`） |
| **`T.tile.topk` 的 `max_actual_num=Σ src.shape`** | Yes → 已消解 | topk 源用 **rank-1 `z_ub[BV]`**（`max_actual_num=BV` 精确，V01-a N2/R-C） |
| **归约的尾块 narrow-reduce ≤256B 约束** | **N/A（V01-b 不用 reduce）** | ⭐ V01-b argmax 走 `topk(K=1)`，**无 `T.reduce_max/sum`** ⟹ **完全规避** V01-a V-2 的 `codegen_ascend.cc:2145` 硬约束与尾块精确尺寸缓冲复杂度（§6.4） |
| **LCG int32 环绕语义（mod 2^32）** | Yes → 已证 | `random_1d.py` 的 `reference_random_1d`（`ctypes.c_int32`）与 kernel `T.tile.mul/add` int32 **`atol=0/rtol=0` 精确匹配** ⟹ 环绕语义已证；V01-b 内联同款（§3.2）。**无需 `bitwise_and`**（int32 mul/add 硬件自动截断，非手动掩码） |
| **`arith_progression` 运行时 first_value** | Yes → 已证 | `seed_v[0]+b*BV`（BufferLoad+loop_var×const）；`attention_sink:162`（`q_row+v_row` 复合运行时）、`per_block_cast:771`（int32 运行时 first_value）已证 |
| **f32 Gumbel 数值稳定性** | Yes → 已消解 | clamp `[2^-24,1-2^-24]`（fp32 精确）⟹ G∈[-2.81,16.64] 全有限（D-CLAMP，消解草稿 D3/R1） |

### 3.5.2 参考实现差异说明（草稿 / random_1d → 本 Developer 方案）

| 差异项 | 草稿 `DESIGN_script.md` / `random_1d.py` | V01-b 本设计 | 转换依据 |
|--------|------------------------------------------|-------------|---------|
| **argmax-with-index** | `wholereducemax(ReduceOrder="ORDER_VALUE_INDEX")`（7 参数低层） | **`T.tile.topk(K=1)` + 标量读 `dst[0]/[1]`**（备选 gather_mask+cast） | 草稿 API **已 `@deprecated()`**（wheel `:1528`）；topk K=1 已测（`test_..._topk.py:324`）；顺带消解草稿 R11（wholereducemax 签名复杂） |
| **rand 输入** | `[B,BV]` host 流 / kernel 内联（草稿自相矛盾） | **`[B]` 每行种子 + kernel 内联 LCG** | D-RAND（§8.3 访存一致性 + bitwise 可复现）；**分歧点上报 §10.4** |
| **u 映射** | `lcg·2^-31`（signed → [-1,1)，负半轴坍缩） | **`lcg·2^-32 + 0.5`（→[0,1) 均匀）** | D-UMAP（修正草稿符号 bug） |
| **Gumbel clamp** | 草稿 D3 担心 `1-2^-32` f32 失效 | **`[2^-24, 1-2^-24]`（fp32 精确，G 全有限）** | D-CLAMP（消解 D3/R1） |
| **LCG 内存分配** | `random_1d.py` 用 **`T.alloc_ub`（Expert）** | **`T.alloc_shared`（Developer）** | V01-a §2.3 论据 2（纯 Vector→`shared.ub`）；random_1d 的 `alloc_ub` 是其 Expert 痕迹，V01-b 改写 |
| **LCG buffer 形态** | random_1d：分离 `idx_ub/state_ub/temp_ub/a_ub/c_ub/seed_ub`（6 个 buffer） | **`state_ub`（in-place arith_progression+3 轮）+ `seed_v[1]` 标量**（省 temp/a/c/idx/seed buffer） | in-place `T.tile.*` 已证（V01-a `:159-160`）；scalar mul/add 已证（V01-a `:219`）；省 UB（§4.5） |
| **cache 清理** | `random_1d.py:7 tilelang.cache.clear_cache()` | **禁止**（生产用持久 `TILELANG_CACHE_DIR`） | V01-a R-JIT / `debug_log.md` next_hint 4；clear_cache 仅演示用 |
| **pass_configs** | random_1d 用 `PassConfigKey` 枚举 3 项 | **原始字符串形式 4 项全开**（V01-a `R4_PASS_CONFIGS`） | V01-a §4.7；`xllm_kernels/utils.py:3-10`（规避 PassConfigKey 导出时机依赖） |
| **尾块** | — | **统一块路径**（gm2ub pad -inf + topk actual_num=BV），无单独尾块缓冲 | V01-b 无 reduce ⟹ 无 V-2 narrow 约束（§3.5.1）；topk 对 -inf pad 安全（`topk_selector.py:56` pad_value=-inf 已证） |

### 3.5.3 本项目同类实现参考（必须列出）

| 文件路径 | 相似度 | 关键参考点（已逐一读取核实） |
|----------|--------|------------------------------|
| **`examples/R4/r4_fused_sampling.py`（V01-a，已 NPU 验证）** | ⭐⭐⭐ **最高**（直接扩展对象） | `:292-337` K2 `r4_mask_sample` 骨架（V01-b 在此基础上加 Gumbel 路径）；`:306-308` Kernel/行守卫；`:320-322` GM→UB 1-elem 标量读；`:326` gm2ub `pad_value=-inf`；`:333-334` `compare`+`select("VSEL_TENSOR_SCALAR_MODE")`；`:167` rank-2→rank-1 copy（topk 源）；`:214` UB 动态标量读；`:229-233` `with T.If/T.Then` 标量循环；`:159-160` in-place `T.tile.*`；`:219` scalar `T.tile.max`；`:27-35` pass_configs/VEC_NUM/_aligned_topk；`:340-378` host wrapper + C3/C6/C7 |
| **`examples/tile_kernels/moe/moe_topk_gate.py`** | ⭐⭐⭐ **最高**（argmax-with-index 同构） | `:66-69/91-94` **`topk`→`gather_mask("P1010")`→`cast("CAST_ROUND")` 生产链**（取 idx）；`:45-48` `topk_dst[2*aligned]` 尺寸约定；**`:347-349 stable_topk=torch.sort(descending,stable=True)` + `:408 assert_equal` ⟹ topk tie-break=最小 index**（D-ARGMAX/D-TIE-ARGMAX 的证据） |
| **`examples/topk_selector/example_topk_selector.py`** | ⭐⭐⭐ 高 | `:48-50` `topk`→`gather_mask("P1010")`→`cast`；`:43` GM 2D 标量起点→1D UB copy；`:56` `pad_value=-inf`；`:57-121` 对拍框架；`:127` `N max ~8192 for 192KB UB` |
| **`examples_experiment/random_1d/random_1d.py`** | ⭐⭐⭐ **最高**（LCG 同构） | `:20-21` `A/C`；`:50-52` fill 常量；`:55` `arith_progression(idx,global_base,1,block_size)`；`:57` `add(state,idx,seed)`；`:59-61` **3 轮 `mul`/`add` int32 环绕**；`:69-80` **`reference_random_1d`（`ctypes.c_int32` 精确 golden）**；`:93` `assert_close(rtol=0,atol=0)` ⟹ int32 环绕 bitwise 已证。⚠️ `:7 clear_cache()`、`:43-48 alloc_ub` **不照抄**（§3.5.2） |
| **`examples/deepseek_v4/act_quant.py`** | ⭐⭐⭐ 高（Gumbel 数值链同构） | **`:78 T.tile.clamp(x_ub_fp, x_ub_fp, -127.0, 127.0, count)`（in-place fp32 clamp，scalar 上下界）**；`:80 round`；`:82-84 cast(mode=...,count=...)`；`:70 T.reduce_max` ⟹ clamp/cast 生产链直接印证 Gumbel 的 clamp+cast |
| **`examples/attention_sink/example_gqa_sink_bwd_bhsd/*.py`** | ⭐⭐ | `:162 arith_progression(row_pos, q_row+v_row, 1, hm)`（**复合运行时 first_value**）；`:178/416/451 arith_progression(col_pos, kv, 1, block_N)`（运行时 first_value）⟹ 印证 D-RAND 的 `seed_v[0]+b*BV` |
| **`examples/tile_kernels/quant/per_block_cast_lossless_kernel.py`** | ⭐⭐ | `:771 arith_progression(..., block_idx*const*4, 4, ...)`（**int32 buffer + 运行时 first_value + diff≠1**）⟹ 印证 int32 arith_progression |
| **`examples/cross_entropy_loss/example_cross_entro.py`** | ⭐⭐ | `:99 T.tile.ln`；`:113-115` UB 动态标量读（V01-b best_idx 归并同款） |
| **`testing/python/language/test_tilelang_ascend_language_topk.py`** | ⭐⭐ **API 语义权威** | **`:324-327 test_topk_k_boundary`（K∈{1,64} fp32 已测）**；`:140-141/169-170`（`[0::2]`=val、`[1::2]`=idx）；`:116 _cpu_topk_ref`（idx=首 actual_num 内 0-based 位）；`:17`（float16 idx 精度限制，**fp32 无此问题**）；`:181-183/201-206`（fp32 src 尾部 in-place -inf padding） |
| ❌ `examples_experiment/random_1d`（`alloc_ub`/`clear_cache` 部分） | — | Expert 分配 + 演示用清缓存，**V01-b 改写为 `alloc_shared` + 持久缓存**（§3.5.2） |

### 3.5.4 源码级证据索引（wheel 提交 `d75300ed` 上复核）

| 结论 | 源码位置（wheel `~/.local/lib/python3.12/site-packages/tilelang/language/ascend_tile.py` 除注明外） |
|------|---------|
| `wholereducemax/sum`（草稿 argmax 原语）已 `@deprecated()` | `:1528/1545/1562`（`ReduceOrder="ORDER_VALUE_INDEX"` 默认在 `:1537`） |
| `topk` 输出交错 `(value,index)`；src 需静态 shape；`max_actual_num=Σ src.shape` | `:550-605`（docstring `:558-561`；`:578-590` Σshape） |
| `topk` K=1 fp32 已测；idx=0-based 原位；`[0::2]`=val/`[1::2]`=idx | `testing/.../test_..._topk.py:324-327/116/140-141/169-170` |
| topk tie-break=最小 index（stable） | `moe_topk_gate.py:347-349/408`（golden `torch.sort(stable=True)`+`assert_equal`） |
| `gather_mask` `"P0101"`=偶=值、`"P1010"`=奇=idx | `:608-666`（docstring `:625-631`） |
| `clamp(out,buf,min,max,count)` in-place fp32 生产可用 | `:2228`；`act_quant.py:78` |
| `ln(dst,src)` 元素级自然对数 | `:1211`；api-compute.md `:306` |
| `cast(dst,src,mode,count)`；mode 列表 | `:1927-1954` |
| `arith_progression(buf,first,diff,count)`；first 可运行时 PrimExpr | `:389-409`；`attention_sink:162`、`per_block_cast:771` |
| `mul/add/max` 的 src1 可为 PrimExpr 标量 | `:1026/1048/1070`；V01-a `r4_fused_sampling.py:219`（scalar max） |
| `compare` src1 可标量（PrimExpr/float）→ `compare_scalar` | `:1837-1903`（`:1894` 分支） |
| `select(dst,mask,src0,scalar,"VSEL_TENSOR_SCALAR_MODE")` | `:707-826`（`:816` 分支）；V01-a `r4_fused_sampling.py:334` |
| `bitwise_and` 存在（V01-b **不用**：int32 mul/add 硬件自动环绕） | `:1092` |
| int32 mul/add 环绕 mod 2^32（与 `ctypes.c_int32` 一致） | `random_1d.py:59-61/76-78/93`（`atol=0/rtol=0` 精确匹配） |
| fp32 topk 自动 tmp arena = `4×AlignUp(BV,32)×4B = 64 KB`（BV=4096） | `src/transform/allocate_tmp_buffer.cc:578-596`+`:1195-1206`（V01-a §4.5 已推导，取 max 单一 arena） |
| `TOP_K_ALL=1<<30`；A3=dav-2201；UB 184KB 可用 | V01-a §3.5.4（`sampling_params.py:40`；`target.py:145-147`；`libgen.py:152-155`） |

### 3.5.5 API 检索顺序合规记录（AGENTS.md 核心原则 1）

| 层级 | 查阅对象 | 命中 / 结论 |
|------|---------|-----------|
| **① 速查表** | `tilelang-api-best-practices/references/api-compute.md` | ✅ 命中：`T.tile.topk`（`:695-716`，输出 `(val0,index0,...)`）、`gather_mask`（`:422-437`，`"P0101"`）、`arith_progression`（`:458`）、`T.tile.ln`（`:306`）、`T.reduce_max`（`:95`，返回值无 index ⟹ 不适用 argmax） |
| **①→② 升级** | `clamp` / `cast`（速查表未详列 in-place fp32 形态） | 降级到 ② `examples/`：`act_quant.py:78-84`（clamp→round→cast 生产链）确认 in-place fp32 clamp + scalar 上下界 |
| **①→② 升级** | `topk(K=1)` 是否可行 + tie-break | ② `test_..._topk.py:324-327`（K=1 fp32 已测）+ `moe_topk_gate.py:347-408`（tie-break=最小 index） |
| **①→② 升级** | LCG int32 环绕 + arith_progression 运行时 first_value | ② `random_1d.py`（环绕 bitwise 已证）+ `attention_sink:162`/`per_block_cast:771`（运行时 first_value） |
| **②→③ 升级** | `wholereducemax(ORDER_VALUE_INDEX)` 是否可用 | ③ wheel 源码 `:1528` 显示 `@deprecated()` ⟹ **以 wheel 为准否决草稿**，改 `topk(K=1)` |
| **③ 源码** | `compare` scalar src1 / `mul/add` scalar src1 / `select` scalar 分支 | ③ wheel `:1894/1026/1048/816` 逐一确认 |
| **禁用** | `pto.py` / `clear_cache` / `alloc_ub` / `T.Scope` / 手动 flag | 🔴 全程未引用（AGENTS.md 原则 1 + V01-a §3.5.2） |

**结论**：V01-b **无凭记忆猜测的 API**；速查表覆盖的（topk/gather_mask/arith_progression/ln/reduce_max）
均核对一致，速查表未覆盖的（clamp/cast in-place、topk K=1 tie-break、LCG int32 环绕、arith_progression
运行时 first_value、wholereducemax deprecated）均下探到 `examples/` + wheel 源码 + 官方测试三重确认。

---

## 4. 数据规格与内存规划

### 4.1 输入张量

**K1 `r4_topk_stats`（不变）**：`logits[B,V] fp32`、`top_ks[B] int32`、`top_ps[B] fp32`、`min_ps[B] fp32`
（V01-a §4.1 原样）。

**K2 Gumbel `r4_gumbel_sample`（V01-b 新增）**：

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `logits` | `(B, V)` | `float32` | temperature 已缩放 logits（与 K1 同一张量，二次读；行内允许 `-inf`，不含 NaN） |
| `tau_in` | `(B,)` | `float32` | K1 产出的合成阈值 `tau_i`（keep 掩码 `z>=tau`）；**不读 `m_in`/`s_in`**（定理 T3） |
| `rand` | `(B,)` | `int32` | **每行 LCG 种子 `seed_i`**（D-RAND / Design C）。kernel 内联展开为 `[B,V]` Gumbel 噪声流。测试用固定小种子（`seed_i+V<2^31` 保 bitwise）；生产用每请求种子 |

> **`rand` shape 决策**：`[B] int32`（每行标量种子），**非**草稿/任务提示的 `[B,BV]`——见 §3.4 D-RAND
> 与 §10.4 分歧点。dtype 仍为 int32（覆盖门禁的 D-DTYPE-int32 维度不受 shape 影响）。

**编译期常量参数**：`B`（运行时，仅影响 grid）、`V=152064`（静态）、`K_MAX=256`（静态，K1 用）、
`BV=4096`（静态，S1' 特化）、`FUSE_GUMBEL`（host 分派开关，选 K2 路径）。LCG 的 `A/C` 为 kernel 内常量（非特化键）。

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 | 产出 kernel |
|--------|-------|-------|------|------------|
| `m_out`/`s_out`/`tau_out`/`valid` | `(B,)`×4 | fp32×3 + int32 | V01-a K1 原样（`tau_out` 供 K2 Gumbel 消费；`valid` 供 host 分派回退） | K1（不变） |
| **`tok_out`** | `(B,)` | **`int32`** | **V01-b 新增**：采样 token id `y_i = argmax_{j∈S_i}[z_ij+G_ij]`。`valid_i==0` 的行 host 丢弃并回退 torch | K2 Gumbel |
| `w_out`（`FUSE_GUMBEL=False`） | `(B, V)` | `float32` | V01-a 路径原样（与 tok_out 互斥，由 FUSE_GUMBEL 选择） | K2 `r4_mask_sample`（不变） |

### 4.3 中间缓冲区（K2 Gumbel，S1'：`BV=4096`, fp32, 单行/任务）

| Buffer | Shape | dtype | 存储层级 | 用途 |
|--------|-------|-------|----------|------|
| `z_ub` | `(BV,)`=`(4096,)` | float32 | UB (`alloc_shared`) | gm2ub 搬入 logits 块；就地变 `cand`（z+G，masked）；**rank-1**（topk 源，N2） |
| `state_ub` | `(4096,)` | int32 | UB | LCG 状态（arith_progression + 3 轮 in-place mul/add，int32 环绕） |
| `g_ub` | `(4096,)` | float32 | UB | Gumbel 噪声（cast dst + in-place `*2^-32 +0.5`/clamp/ln/neg/ln/neg） |
| `mask` | `(BV//8,)`=`(512,)` | uint8 | UB | keep 掩码（`compare(z,tau,"GE")`），512 B |
| `topk_dst` | `(8,)` | float32 | UB | `topk(K=1)` 输出 `[blk_val, blk_idx, pad...]`，`aligned_topk(1)=8` |
| `t_v` | `(1,1)` | float32 | UB | 行标量 `tau_i`（GM→UB 1-elem copy） |
| `seed_v` | `(1,)` | int32 | UB | 行标量 `seed_i`（GM→UB 1-elem copy） |
| `best_val` | `(1,)` | float32 | UB | 跨块 running argmax 值（init -inf） |
| `best_idx` | `(1,)` | float32 | UB | 跨块 running argmax 全局 index（`<2^24` 精确；init 0） |
| `tok_i32` | `(1,)` | int32 | UB | 输出标量（cast best_idx→int32，UB→GM 1-elem copy） |
| `tmp_ub`（**编译器自动注入**） | `(65536,)` | uint8 | UB (`shared.ub`) | topk 的 tmp arena，`= 4×AlignUp(BV,32)×4B = 64 KB`（§4.5） |

> **相比 V01-a K2 的新增 buffer**：`state_ub`（int32 LCG）、`g_ub`（Gumbel）、`topk_dst`（argmax）、
> `best_val/best_idx/tok_i32/seed_v`（归并标量）。**消除**了 V01-a K2 的 `e_ub/w_ub/m2d/s2d/t2d`
> （Gumbel 路径无 exp/div/广播 tau——compare 用 scalar src1）。`mask` 从 `[1,BV/8]` 简化为 `[BV/8]`。

### 4.4 内存搬运路径

**纯 Vector —— 不涉及 L1/L0A/L0B/L0C，无跨级搬运**（AGENTS.md 核心原则 3：GM↔UB 单级，全经 `T.copy`）：

```
K2 Gumbel（循环 n_blk=38 次）:
  GM[logits]  --T.copy(gm2ub, 尾块 auto-clamp + pad -inf)-->  UB[z_ub (BV)]
  GM[tau_in]  --标量读(1 elem/行, 循环外一次)---------------->  UB[t_v (1,1)]
  GM[rand]    --标量读(1 elem/行, 循环外一次)---------------->  UB[seed_v (1)]
  UB[seed_v]+b*BV --arith_progression--> UB[state_ub (BV) int32]
  UB[state_ub] --3x(mul A, add C) int32 环绕--> UB[state_ub]     (LCG, in-place)
  UB[state_ub] --cast(int32->fp32)--> UB[g_ub (BV)]
  UB[g_ub] --*2^-32 +0.5, clamp, ln, neg, ln, neg--> UB[g_ub]    (Gumbel, in-place)
  UB[z_ub] --compare(.,t_v,"GE")--> UB[mask (BV/8) uint8]        (keep)
  UB[z_ub] --add(z_ub,g_ub); select(mask,.,-inf)--> UB[z_ub]     (cand, in-place)
  UB[z_ub] --T.tile.topk(K=1, BV)--> UB[topk_dst (8)]            (块内 argmax, 用 tmp_ub arena)
  UB[topk_dst] --标量读 [0]/[1] + T.If/T.Then 归并--> UB[best_val/best_idx (1)]
  UB[best_idx] --cast(->int32)--> UB[tok_i32] --T.copy(ub2gm 1-elem)--> GM[tok_out]
```

**GM 遍历次数与流量核算**（B=256, V=152064, fp32；单张量 = 148.5 MB）：

| 阶段 | 读 | 写 |
|------|----|----|
| K1（不变） | 1 × logits = 148.5 MB | 4 × `[B]` ≈ 4 KB（可忽略） |
| **K2 Gumbel** | 1 × logits = 148.5 MB（+ `tau_in`/`rand` 各 `[B]` ≈ 2 KB 可忽略） | 1 × `tok_out[B]` = 1 KB（可忽略） |
| **合计（V01-b）** | **≈ 297 MB** | **≈ 0（ε）** |

对比草稿 §8.3：V01-b 总流量 ≈ **297 MB**（**2 读，0 写**），相对现状 Path 2（1931 MB）**6.5x**、
相对 V01-a（594 MB）**2x**。**这正是 D-RAND 选 Design C（内联 LCG）而非 Design D（host [B,V] 流）的
量化理由**——Design D 会增第三遍读（rand[B,V] 148.5 MB）退化到 445 MB。

### 4.5 UB 内存预算（复核草稿 §5.4.2 的 104 KB）

**目标平台**：Ascend910_9362（实测）→ A3 → arch22 / DAV_2201 → UB 192 KB，向量可用 **184 KB = 188,416 B**
（V01-a §4.5 已核，佐证 `rope_half_interleaved.py:218`、`topk_selector.py:127`）。

**K2 Gumbel `r4_gumbel_sample`（S1'：`BV=4096`, fp32, 单行/任务）**

| Buffer | Shape | dtype | Bytes |
|--------|-------|-------|-------|
| `z_ub` | (4096,) | fp32 | 16,384 |
| `state_ub` | (4096,) | int32 | 16,384 |
| `g_ub` | (4096,) | fp32 | 16,384 |
| `mask` | (512,) | uint8 | 512 |
| `topk_dst` | (8,) | fp32 | 32 |
| `t_v/seed_v/best_val/best_idx/tok_i32` | (1,)/(1,1) ×5 | fp32/int32 | ~20 |
| **`tmp_ub`（topk 自动注入）** | — | uint8 | **65,536** |
| **总计** | | | **115,252 B ≈ 112.6 KB** |
| **判定** | | | **✅ 112.6 KB ≤ 184 KB，余量 71.4 KB（39%）** |

**`tmp_ub` arena 推导**（V01-a §4.5 同模型）：topk fp32、src `z_ub[BV=4096]` ⟹ `max_actual_num=4096`、
`factor=4`、`count=AlignUp(4096,32)=4096` ⟹ arena `= 4×4096×4 = 65,536 B = 64 KB`（取所有 call 的 max，
单一共享 arena）。K=1 不减小 arena（arena 取决于 src 尺寸 BV，非 K）。

> 🔑 **修正草稿 §5.4.2**：草稿估 V01-b K2 ≈ **104 KB**，但那是 **`BV=8192`** 且**漏算了 topk 的 64 KB
> 自动 tmp arena**（与 V01-a 修正草稿 §5.3.6/§5.3.7 的同一错误）。本设计用 **`BV=4096`（S1'，与 V01-a
> 一致的单特化）+ 正确 arena 模型**，实得 **≈112.6 KB ≤ 184 KB**（余量 39%）。若误用 `BV=8192`：
> arena 翻倍到 128 KB + 数据 buffer 翻倍 ⟹ ≈**217 KB ≫ 184 KB 溢出**（草稿 104 KB 严重低估）。
> **V01-b 固定 `BV=4096`。**

**UB 占用通式（K2 Gumbel，单位 Byte，`BV%32==0`）**
```
UB_K2gumbel(BV) = 4·BV(z_ub) + 4·BV(state_ub) + 4·BV(g_ub) + BV/8(mask)
                + 16·AlignUp(BV,32)(topk arena) + ~52(topk_dst+标量)
              ≈ 28.125·BV + 52
   BV=4096 → 115,252 B ≈ 112.6 KB ✅；BV=8192 → ~230 KB ❌（溢出）；BV=2048 → ~57 KB ✅（Stage 3 双缓冲候选）
```

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 | 处理方式 |
|--------|----------|-----------|---------|
| `B` | `@tilelang.jit` Python int 形参 → `T.Tensor([B,…])` + `b_num=T.ceildiv(B,VEC_NUM)` | decode 1~256；prefill 尾 1~1024 | 仅影响 grid + 行守卫 `if row<B`，不作循环边界（V01-a 同）。**R-JIT：B 进入 JIT 键**（V01-a §4.6，缓解：持久缓存 + 禁 clear_cache） |
| `V` | 编译期常量 | `{32000,128256,129280,151936,152064}`（V01-b 只做 152064） | 必须静态（topk src 静态 shape；n_blk 静态） |
| `seed_i`（`rand` 元素） | 运行时张量元素（int32） | 测试：固定小种子；生产：每请求种子 | kernel 内标量读入 `seed_v`，**不进特化键** |

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[3], pass_configs=R4_PASS_CONFIGS)   # tok_out 自动分配
def r4_gumbel_sample(B: int, V: int, BV: int): ...
# R4_PASS_CONFIGS 沿用 V01-a（r4_fused_sampling.py:27-32，原始字符串形式，四项全开）
```
- K1 `r4_topk_stats`（`out_idx=[4,5,6,7]`）与 K2 `r4_mask_sample`（`out_idx=[4]`）**不变**。
- 不指定 `workspace_idx`/`threads`（无 CV 交互；§2.3）。
- 编译目标 `target="ascendc"`（自动）→ bisheng `--npu-arch=dav-2201`。

---

## 5. Tiling 策略

### 5.1 计算类型

**纯 Vector**（零 matmul）：LCG（int32 mul/add）、Gumbel（cast/mul/add/clamp/ln）、argmax（topk K=1，
标 `PIPE_V`）、跨块归并（标量）。不使用 Cube/L1/L0*。

### 5.2 Block 划分

```python
VEC_NUM = 2                          # 每 Cube 核配 2 Vector 核（V01-a 同）
b_num   = T.ceildiv(B, VEC_NUM)      # 一行一任务（B=256 → 128 block，超订可跑）
n_blk   = T.ceildiv(V, BV)           # 每行 vocab 分块数 = 38（V=152064, BV=4096，静态）
BV      = 4096                       # S1' 特化（与 V01-a 一致）
```
- **任务粒度**：一行 = 一任务（`row = cid*VEC_NUM+vid`），B 行映射到 `min(B, vec_core_num)` 个 Vector 核
  （运行时 `detect_vec_core_num()` 探测，Ascend910_9362 实测 40，**禁硬编码 48**，V01-a §3.5.2）。
- **无跨核归约**（一行一任务）⟹ 不需 `atomic_add`、无跨核 barrier（草稿 D7，V01-a §6.5）。

### 5.3 约束分析

- **对齐约束**：`topk` src 大小需 32 或 32 的倍数（api-compute.md `:716`）。`BV=4096`（`%32==0`）✓；
  尾块 `TAIL=V%BV=512`（`%32==0`）✓ —— 但 V01-b **统一按 BV 处理尾块**（gm2ub pad -inf），src 恒 `[BV]` ✓。
- **UB 容量**：112.6 KB ≤ 184 KB ✓（§4.5，余量 39%）。
- **L0 容量**：无 Cube，不适用。
- **topk `max_actual_num=Σ src.shape`**：src 用 rank-1 `z_ub[BV]` ⟹ `max_actual_num=BV=4096` 精确（N2/R-C）。
- **fp32 index 精确性**：topk 输出 idx（fp32）< `BV=4096 < 2^24` ✓；全局 idx < `V=152064 < 2^24` ✓（D-ARGMAX）。

### 5.4 注意事项（非整除与尾块，显式设计）

- **非整除 `V%BV`**：`V=152064, BV=4096` → `n_blk=38`（37 满块 + 1 尾块 `TAIL=512`）。用 `T.ceildiv(V,BV)`
  + gm2ub `pad_value=-inf` 自动处理尾块（`T.copy` lowering 期 clamp，`test_..._tail_block.py:18-31`）。
- **⭐ V01-b 尾块统一路径（比 V01-a 更简）**：V01-b K2 Gumbel **无 `T.reduce_*`**（argmax 走 topk）⟹
  **不触发** V01-a V-2 的 narrow-reduce ≤256B 硬约束（`codegen_ascend.cc:2145`）⟹ **无需**为尾块单独分配
  精确尺寸 `[1,TAIL]` 缓冲。尾块与满块走**同一代码路径**：`T.copy(..., pad_value=-inf)` 填 `[TAIL:BV)` 为
  -inf ⟹ keep 掩码 `(-inf >= tau)=False` ⟹ cand=-inf ⟹ topk(K=1) 不选 pad 区（-inf 最小）⟹ argmax 正确。
  topk `actual_num` 可传 `BV`（含 pad，pad=-inf 无害）或 `TAIL`（精确）；**推荐传 `BV`**（统一循环，无编译期分支）。
- **topk 对 fp32 src in-place 写 -inf padding**：作用于 UB scratch `z_ub`（非 GM），`AlignUp(actual_num,32)<=BV`
  无越界（N3；`test_..._topk.py:181-183`）。`topk_selector.py:56`/`moe_topk_gate.py:56` 均用 `pad_value=-inf` + topk，已证安全。
- **host 侧不做 padding + crop**（ascend-constraints.md §5）：尾块全在 kernel 内 `T.copy` 自动 clamp 处理。

### 5.5 数据搬运性能可行性

| 结构/dtype 路径 | 代表性最大 case | GM pass | DMA 数/平均字节 | GM 标量访问 | 地址 div/mod | AIV 并行度 | 结论 |
|-----------------|-----------------|---------|------------------|-------------|--------------|------------|------|
| K2 Gumbel（fp32） | B=256, V=152064, BV=4096 | 1 读 logits（148.5 MB）+ ε 标量 | 每行 38 块 gm2ub × 256 行 ≈ 9728 次 DMA，平均 16 KB/次 | 每行 2 次 1-elem（tau/seed 读）+ 1 次（tok 写）× 256 ≈ 768 | 每块 `b*BV` 一次（编译期常量 × loop var） | 40 vec 核，每核 `ceil(256/40)=7` 行串行 | **可行**：大张量主路径为连续块 DMA（非逐元素 strided GM 访问）；标量访问 O(B) 而非 O(B·V) |

> **`[REORDER-COST]`**：无物理布局重排（logits 行内连续读、tok_out 标量写）；`gm_passes=1`(logits)+ε；
> `dma_transactions≈9728`（B=256）；`average_dma_bytes≈16384`；`gm_scalar_accesses≈768`（O(B)，非 O(numel)）；
> `address_div_mod_per_element≈0`（块级 `b*BV`）；`active_cores/serial_tasks_per_core=40/7`；
> `largest_case_timeout_gate`：见 §9.4（perf_max_batch B=256 → 1200 s）；**result: pass**。

### 5.6 Host 侧 Buffer 操作审计（ascend-constraints.md §5）

```text
[HOST-METADATA-AUDIT]（V01-b host 路径，逐项）
operation: rand[B] int32 种子张量准备（torch.arange/常量，或 scheduler 提供）
  -> 数据准备（新建小张量），非对 kernel 输入/输出的物理化改动；result: allow
operation: C6(B==0)/C7(V<K_MAX)/C3(tail<K_MAX)/dtype 前置校验（V01-a 原样）
  -> 纯 Python 整数/元数据判断 + host 拦截，不触碰数据；result: allow
operation: k1(logits,...) / r4_gumbel_sample(logits,tau_out,rand) kernel 调用
  -> kernel 调用；result: allow
operation: valid 的 D2H 校验（torch.all(valid)）+ 按 valid 分派回退
  -> 结果验证/分派（读 [B] int32），非对 logits/tok_out 的物理化；result: allow
operation: （对比 V01-a）torch.multinomial(w_out,1)
  -> V01-b 已消除（Gumbel 融合）⟹ host 侧零 aclnn 计算算子；result: N/A（优于 V01-a DEV-1）
```
**结论**：V01-b host 路径**无** `.contiguous()`/`reshape` 物理化/`torch.cat`/`.to(dtype)`/aclnn 计算；
相比 V01-a **消除了 `torch.multinomial`（DEV-1 偏差）**，host 侧更干净、完全合规。

### 5.7 分派覆盖审计

```text
[DISPATCH-COVERAGE]（V01-b）
supported_domain: logits[B,V] fp32（V∈特化表，代表 152064）；rand[B] int32；单特化 S1'(K_MAX=256,BV=4096)；
                  FUSE_GUMBEL∈{True,False}
generic_fallback: valid_i==0 的行 -> host 回退 torch 采样路径（correct-by-fallback，V01-a D-GUARD 同）；
                  FUSE_GUMBEL=False -> V01-a w_out + torch.multinomial 路径（不变）
specializations:
  - predicate: FUSE_GUMBEL==True（编译期 bool）
    fallback_on_miss: FUSE_GUMBEL==False -> r4_mask_sample(w_out) + multinomial
    equivalence_evidence: 定理 T3（z 域 Gumbel-max argmax ⟺ 对 w 采样）；分布一致性 §9.5
  - predicate: valid_i==1（K1 守卫，运行时）
    fallback_on_miss: valid_i==0 -> host 丢弃 tok_out[i]，回退 torch
    equivalence_evidence: V01-a D-GUARD（k_i∉[1,K_MAX] / nucleus>K_MAX / 超额平票 -> 回退）
unsupported_inputs: bf16 / float64 / 确定性 seed（sampling_seed≠None）/ V<K_MAX / tail<K_MAX -> host 抛
                    NotImplementedError 或旁路（V01-a C3/C7 + 草稿 B4 同）；与 supported_domain 不冲突
result: pass（每条 specialization 与 fallback 均在 §9 验证计划有命中 case）
```

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| batch（B 行） | block 级并行 | `T.Kernel(b_num, is_npu=True) as (cid,vid)`，`row=cid*VEC_NUM+vid` | 一行一任务 ⟹ 无跨核归约、UB 私有 |
| 行有效性 | 条件守卫 | `if row < B:` | 处理 B 不被 VEC_NUM 整除 |
| vocab 分块 | 静态迭代 | `for b in T.serial(n_blk)` | trip count=38（编译期常量）；**尾块统一在内（§5.4），无单独尾块发射** |
| LCG 3 轮 | 编译期常量循环 | `for _ in range(3)` | 固定 3 轮（random_1d 同） |
| 块内元素级 | 向量化（T.tile.*） | `arith_progression`/`mul`/`add`/`cast`/`clamp`/`ln`/`compare`/`select`/`topk` | 全 `T.tile.*` 向量原语（BV 元素），无 T.Parallel 逐元素（避免标量 T.exp/log 无法 codegen，V01-a 发现 6） |
| 跨块 argmax 归并 | 条件标量更新 | `with T.If(blk_val>best_val): with T.Then():` | 顺序依赖（running argmax）⟹ 串行标量；严格 GT（D-TIE-ARGMAX） |
| **不启用** | — | ~~`T.Pipelined`~~/~~`T.Persistent`~~ | §6.3 |

### 6.2 循环伪代码

```python
with T.Kernel(b_num, is_npu=True) as (cid, vid):
    row = cid * VEC_NUM + vid
    if row < B:
        # alloc_shared 分配；fill(best_val,-inf)/fill(best_idx,0)；copy tau_in->t_v, rand->seed_v
        for b in T.serial(n_blk):                    # 静态 38（含尾块，统一路径）
            T.copy(logits[row, b*BV], z_ub, pad_value=-inf)   # gm2ub（尾块 auto-clamp）
            T.tile.arith_progression(state_ub, seed_v[0]+b*BV, 1, BV)   # LCG 初值
            for _ in range(3):                        # LCG 3 轮（int32 环绕）
                T.tile.mul(state_ub, state_ub, LCG_A); T.tile.add(state_ub, state_ub, LCG_C)
            T.tile.cast(g_ub, state_ub, "CAST_RINT", BV)       # int32->fp32
            T.tile.mul(g_ub,g_ub,INV_2_32); T.tile.add(g_ub,g_ub,0.5)
            T.tile.clamp(g_ub,g_ub,U_MIN,U_MAX,BV)             # Gumbel u
            T.tile.ln(g_ub,g_ub); T.tile.mul(g_ub,g_ub,-1.0)
            T.tile.ln(g_ub,g_ub); T.tile.mul(g_ub,g_ub,-1.0)   # G=-ln(-ln(u))
            T.tile.compare(mask, z_ub, t_v[0,0], "GE")         # keep
            T.tile.add(z_ub,z_ub,g_ub)                         # z+G
            T.tile.select(z_ub, mask, z_ub, -inf, "VSEL_TENSOR_SCALAR_MODE")   # cand
            T.tile.topk(topk_dst, z_ub, 1, BV)                 # 块内 argmax
            with T.If(topk_dst[0] > best_val[0]):
                with T.Then():
                    best_val[0] = topk_dst[0]; best_idx[0] = topk_dst[1] + b*BV
        T.tile.cast(tok_i32, best_idx, "CAST_ROUND", 1)
        T.copy(tok_i32, tok_out[row:row+1])            # 标量写 GM
```

`T.serial` 与向量原语分工：**跨块 running argmax 有顺序依赖 ⟹ `T.serial` + `T.If/T.Then` 标量归并；
块内 LCG/Gumbel/cand/argmax 无顺序依赖 ⟹ 全 `T.tile.*` 向量原语（BV 元素并行）**。

### 6.3 流水线优化

**V01-b 不启用 `T.Pipelined`**（理由同 V01-a §6.3）：① 块间存在循环携带依赖（`best_val`/`best_idx`
跨块串行归并）；② 双缓冲需 `z_ub`/`state_ub`/`g_ub` 各 ×2 ⟹ K2 从 112.6 KB 升至 ≈177 KB，逼近 184 KB；
③ V01-b 首要目标是精度收敛（bitwise tok_out）与功能正确，性能调优属 Stage 3。**Stage 3 可选**：
`T.Pipelined(stage=2)` 重叠 gm2ub（需 BV 降到 2048 腾双缓冲，arena 随之减半到 32 KB）；`g_ub` alias `state_ub`
（cast 后 state 不再需要）；K1+K2 Gumbel 合并单 kernel（B=1 时第二遍命中 L2）。

### 6.4 尾块处理

**V01-b 尾块统一路径（无单独尾块代码）**——见 §5.4：因 argmax 走 `topk(K=1)` 而非 `T.reduce_*`，
**不触发 narrow-reduce ≤256B 约束**（V01-a V-2 的根因），故尾块与满块共用同一 `T.serial(n_blk)` 循环体，
仅靠 gm2ub `pad_value=-inf` 填充 `[TAIL:BV)`。这**显著简化**了 V01-b K2（对比 V01-a K1 需为尾块单独分配
精确尺寸 `[1,TAIL]` 缓冲 + 内联独立 reduce）。

---

## 7. 同步策略

### 7.1 同步模式

**全自动同步（Developer 模式）**——零手写同步原语（不出现 `T.barrier_all`/`T.set_flag`/`T.wait_flag`/
`T.pipe_barrier`/`T.Scope`）。⚠️ 注意：`random_1d.py`/`moe_topk_gate.py` 用了手动 `T.set_flag`/`T.pipe_barrier`/
`T.barrier_all`（Expert 形态）——**V01-b 不照抄**，改全自动（§3.5.2）。

### 7.2 同步点说明（无手动同步点；下列依赖由编译器自动插入）

| 依赖 | 类型 | 自动同步来源 |
|------|------|-------------|
| `T.copy(GM→z_ub)` 完成后再 `compare/add(z_ub)` | MTE2→V | `TL_ASCEND_AUTO_SYNC` |
| `arith_progression/mul/add(state_ub)` 完成后再 `cast(g_ub,state_ub)` | V→V RAW | `TL_ASCEND_AUTO_SYNC` |
| Gumbel 链（cast→mul→add→clamp→ln→mul→ln→mul，均 in-place `g_ub`） | V→V RAW/WAW | `TL_ASCEND_AUTO_SYNC` |
| `add(z_ub,g_ub)`/`select(z_ub,...)` 完成后再 `topk(topk_dst,z_ub)` | V→V RAW | `TL_ASCEND_AUTO_SYNC` |
| `topk` 写 `topk_dst` 完成后再标量读 `topk_dst[0]/[1]` | V→V RAW | `TL_ASCEND_AUTO_SYNC` |
| `topk` 与其 `tmp_ub` arena 复用（arena 无内容契约，每 call 可覆写） | V→V WAR/WAW | `AUTO_SYNC` + `allocate_tmp_buffer.cc:500-502`（`kWorkspaceWriteAccess`） |
| `z_ub`/`state_ub`/`g_ub` 生命周期不重叠的地址复用 | — | `TL_ASCEND_MEMORY_PLANNING` |
| **跨核同步** | — | **不需要**（一行一任务，无跨核归约/共享 buffer/atomic_add） |
| **Cube↔Vector 核间同步** | — | **不适用**（零 Cube）；`AUTO_CV_SYNC` 空转（与 V01-a 一致） |
| **K1→K2 Gumbel 的 kernel 间顺序** | — | host 顺序调用两次 `@tilelang.jit`（launch 天然串行，经 GM 传 `tau_out`） |

### 7.3 pass_configs 配置

```python
R4_PASS_CONFIGS = {                    # 沿用 V01-a（r4_fused_sampling.py:27-32，原始字符串形式）
    "tl.ascend_auto_sync": True,             # ① 自动核内同步（替代全部手写 barrier/flag）
    "tl.ascend_memory_planning": True,       # ② 自动内存规划（复用中间 buffer）
    "tl.ascend_auto_cv_combine": True,       # ③ 自动 CV 分离（替代手写 T.Scope）
    "tl.ascend_auto_cross_core_sync": True,  # ④ 自动核间同步（本算子无 Cube，空转）
}
```
依据 V01-a §4.7/§7.3 + `programming-model-guide` §2.1/§2.2：纯 Vector 必需 ①②；③④ 无害空转（与仓库
纯 Vector 示例一致）。**K1/K2(w_out)/K2(Gumbel) 三个 kernel 共用同一 `R4_PASS_CONFIGS`。**
若 Stage 2 发现 ③ 导致纯 Vector 的 `alloc_shared` 核归属误判，可单独关 ③④ 保留 ①②（V01-a §7.3 同，唯一允许的调整）。

---

## 8. 融合算子设计

### 8.1 融合算子判定

**判定结果：否**（就 skill 模板定义的「融合算子 = GEMM + element-wise 后处理」而言）。V01-b 是**纯 Vector
多算子融合**（LCG + Gumbel + argmax 融合进 K2），但**无 GEMM/Cube** ⟹ 不涉及 CV 融合、workspace、`workspace_idx`。

### 8.6 注意事项

- 无 Cube 核 ⟹ 无 L1/L0A/L0B/L0C、无 CV 片上直连、无 workspace、无跨核 flag（§2.3/§7.2）。
- 装饰器无 `workspace_idx`，签名无 `workspace_*` 参数。
- 「融合」体现在：K2 Gumbel 路径把「LCG 随机数生成 + Gumbel 反变换 + keep 掩码 + z 域 argmax + 跨块归并」
  融进**单个 kernel 的单遍 vocab 扫描**，替代 V01-a 的「K2 输出 w_out + host multinomial」两步。

---

## 9. 验证方案（两层：① bitwise 忠实参考 + ② 分布一致性）

> 本章 §9.2 的 **L0 门槛测试计划**由 `tilelang-op-test-design`（**场景 A：从 design.md**）产出。
> 按 Stage 1 契约，**本阶段只生成 L0**（以 ① bitwise 对拍忠实 Gumbel-max 参考为主 + LCG 单元对拍）；
> ② 分布一致性（χ²/KL/覆盖率/熵）作为 **L1/L2 判据留待 Stage 2**（`tilelang-op-test-design` 场景 B，
> 从真实 `examples/R4/*.py` 实现扩展）。**本章不手工枚举 L1/L2/Boundary 用例**（仅在 §9.5 给出方法论）。

### 9.0 算子类别判定（`tilelang-op-test-design` §3.2）

| 判定项 | 结论 | 依据 |
|--------|------|------|
| 计算类型 | **纯 Vector** | §5.1：零 matmul，topk/LCG/Gumbel 全标 `PIPE_V` |
| 复杂度级别 | **Fusion** | LCG + Gumbel + 屏蔽 + argmax + 跨块归并 多步融合 |
| 数学特征 | **Selection（argmax）+ RNG（LCG）+ 变换（Gumbel）** | §1.3 |
| **综合类别** | **Fusion（纯 Vector 融合算子）** | `COVERAGE_CATEGORY="Fusion"`（与 V01-a 同） |
| 测试重点 | ① **bitwise tok_out**（忠实参考）；② LCG int32 环绕 bitwise；③ 分块 argmax 归并正确性（n_blk 变化）；④ 尾块统一路径（pad -inf 不污染 argmax）；⑤ keep 掩码复用 K1 tau；⑥ 分布一致性（L1/L2） | — |
| 特殊说明 | tok_out 是 **int32 离散** ⟹ 主判据为**精确匹配**（非容差）；相对 **torch.multinomial 不可逐位对拍**（RNG 流不同），必须对拍**忠实 Gumbel-max 参考**（同 LCG + 同 Gumbel + 同 argmax） | §9.1 |

### 9.1 Golden 函数（扩展 `examples/R4/r4_ref.py`）

V01-b 在 `r4_ref.py` 增两个 golden（V01-a 的 `golden_r4` **不变**）：

**(1) `golden_lcg(M, seed)` —— LCG 精确参考（直接复用 `random_1d.py:69-80` 的 `reference_random_1d`）**

```python
import ctypes, numpy as np, torch
LCG_A, LCG_C = 1103515245, 12345
def golden_lcg(M, seed):                      # = reference_random_1d(M, seed)
    out = np.zeros(M, dtype=np.int32)
    for i in range(M):
        s = seed + i                          # Python int（测试用 seed+M<2^31 → 无溢出，与 kernel arith_progression 一致）
        s = ctypes.c_int32(s * LCG_A + LCG_C).value   # 3 轮 int32 环绕
        s = ctypes.c_int32(s * LCG_A + LCG_C).value
        s = ctypes.c_int32(s * LCG_A + LCG_C).value
        out[i] = s
    return torch.from_numpy(out)              # int32 [M]，与 kernel state_ub 逐位相同
```

**(2) `golden_gumbel_sample(logits, tau, rand, ...)` —— 忠实 Gumbel-max 参考（bitwise 主判据）**

```python
def golden_gumbel_sample(logits, tau, rand):
    """忠实复现 kernel 的 LCG + Gumbel + argmax（fp32 镜像 kernel 算术），返回 tok_ref[B] int32。
    ⚠️ Gumbel 部分用 fp32（镜像 kernel），不用 float64 —— 目标是 bitwise 匹配 kernel 的 fp32 运算。
    tau 用 K1 的 tau_out（隔离测 K2-Gumbel），或用 golden_r4 的 tau_ref（端到端）——二者 V01-a 已证逐位一致。"""
    B, V = logits.shape
    z = logits.detach().to(torch.float32).cpu()          # fp32（镜像 kernel）
    tau = tau.detach().to(torch.float32).cpu()
    tok_ref = torch.zeros(B, dtype=torch.int32)
    INV2_32, UMIN, UMAX = 2.0**-32, 2.0**-24, 1.0 - 2.0**-24
    for i in range(B):
        lcg = golden_lcg(V, int(rand[i])).to(torch.float32)   # int32 精确 → fp32（镜像 kernel cast）
        u = (lcg * INV2_32 + 0.5).clamp(UMIN, UMAX)           # fp32 仿射 + clamp（镜像 kernel 运算顺序）
        G = -torch.log(-torch.log(u))                         # fp32 两次 log（镜像 kernel T.tile.ln）
        cand = torch.where(z[i] >= tau[i], z[i] + G,
                           torch.full_like(z[i], float("-inf")))  # keep 掩码 + z 域候选
        tok_ref[i] = int(torch.argmax(cand).item())           # 最小 index argmax（= kernel 严格 GT 归并）
    return tok_ref
```

**Golden 一致性要求**（`tilelang-op-develop` checklist #9/#10）：
- **逐运算镜像 kernel**：LCG（int32 环绕，`golden_lcg`）→ cast fp32 → `*2^-32` → `+0.5` → clamp → `log` →
  `neg` → `log` → `neg` → `z+G`（fp32）→ `where(z>=tau)` → `argmax`。**运算顺序与 dtype 必须与 §3.3 伪代码一致**
  （否则 bitwise 失配）。
- **argmax tie-break**：`torch.argmax` 返回**首现（最小 index）** = kernel 严格 GT 归并（D-TIE-ARGMAX）✓。
- **输出形状**：`tok_ref[B] int32`，与 §4.2 一致。
- **`rand` 种子约束**：测试用固定小种子（`seed_i + V < 2^31`）⟹ `golden_lcg` 的 `seed+i`（Python int 无溢出）
  与 kernel `arith_progression`（int32）逐位一致。生产种子可任意（LCG 环绕，仅分布验证）。

> **⚠️ bitwise 的唯一残留风险（R-GUMBEL-LN，§10.1）**：`torch.log`（host 软件 fp32）与 Ascend `T.tile.ln`
> （硬件 fp32）可能差 ~1 ULP ⟹ G 差 ~1e-7 相对 ⟹ `z+G` 差 ~1e-7。**只有当某行 top-2 候选的 `z+G` 差距
> < ~1e-6 时才会翻转 argmax**。对连续 `randn×8` logits + O(1) Gumbel 噪声，top-2 gap 典型 O(0.1~1) ≫ 1e-6
> ⟹ **bitwise tok_out 在 L0 固定种子下几乎必然成立**。若个别 shape 出现翻转（近简并），属真实数值边界
> （两个 index 都是 ULP 扰动下的合法 argmax），非 kernel bug —— Stage 2 应换种子或记录该边界，**不误报 `[DESIGN_ERROR]`**。
> 缓解见 §10.2 V-b2（先对拍 `u`/`G` 中间量确认 ln ULP 差异量级）。

### 9.2 L0 门槛测试计划（`tilelang-op-test-design` 场景 A 产出）

> **L0 定位**：规则 shape（block 整除 / 静态尾块）、快速冒烟、供 Stage 2 精度收敛。**≤10 用例**（本计划 8 个
> 端到端 + 1 个 LCG 单元探针）。全部 fp32（V01-b 唯一支持的计算 dtype）。所有用例的 `valid` 均须**全 1**
> （K1 守卫不触发，靠 §9.2.1 数据配方保证）。`(K_MAX, BV)` 均为 S1' 特化 `(256, 4096)`。
> **主判据 = tok_out int32 逐位相同**（① bitwise 对拍忠实 Gumbel-max 参考）。

#### 9.2.0 L0 两层结构（本计划的验证骨架）

| 层 | 对象 | 金标准 | 判据 | 定位 |
|----|------|--------|------|------|
| **L0-A：LCG 单元探针** | kernel 内联 LCG 的 int32 state 流（用最小 probe kernel 或直接对拍 `golden_lcg`） | `reference_random_1d(M, seed)`（= `golden_lcg`，`ctypes.c_int32`） | **int32 逐位相同**（`atol=0/rtol=0`） | 先隔离验证 LCG（arith_progression + 3 轮 int32 环绕）正确，再验全链 |
| **L0-B：bitwise 端到端** | `tok_out[B] int32`（K1 + K2-Gumbel 全链） | `golden_gumbel_sample(logits, tau_out, rand)` | **int32 逐位相同**（`tok_out == tok_ref` 全等） | 主判据：验证 T3 + LCG + Gumbel + argmax + 跨块归并的实现正确性 |

> **为什么 L0 以 ① bitwise 为主**：tok_out 相对 `torch.multinomial` **不可逐位对拍**（LCG RNG ≠ multinomial
> RNG），但对**忠实 Gumbel-max 参考**（同 LCG/同 Gumbel/同 argmax/固定 seed）**可逐位对拍**——这是比分布
> 验证**强得多**的收敛判据（能精确定位 LCG/Gumbel/argmax/归并任一环节的实现 bug）。② 分布一致性（§9.5）
> 验证「Gumbel-max + 正确 RNG ⟹ 服从目标 categorical」，属统计判据，留 L1/L2（Stage 2）。

#### 9.2.1 🔴 测试数据配方（**强制**，沿用 V01-a §9.2.1 + V01-b 种子约束）

```python
torch.manual_seed(0)                 # 确定性 logits
LOGIT_SCALE = 8.0                    # ⚠️ 强制值（V01-a §9.2.1 实测：保证 K1 的 valid 全 1，nucleus ≤ K_MAX）
logits = torch.randn(B, V, dtype=torch.float32) * LOGIT_SCALE
top_ks = torch.full((B,), k,   dtype=torch.int32)
top_ps = torch.full((B,), p,   dtype=torch.float32)
min_ps = torch.full((B,), mu,  dtype=torch.float32)
# ---- V01-b 专属：rand 种子（每行 LCG 种子，int32）----
rand = torch.arange(B, dtype=torch.int32)        # 固定小种子（seed_i = i）；满足 seed_i + V < 2^31（bitwise 前提）
# 或多 seed 用例：rand = torch.full((B,), SEED, dtype=torch.int32)，SEED ∈ {0, 42, 1234, ...}
```

**两条强制约束**：
1. **`LOGIT_SCALE = 8.0`**（V01-a §9.2.1 已实测论证）：保证 K1 的 `valid` 全 1（nucleus ≤ K_MAX=256），
   否则 keep 集为空/守卫触发 ⟹ tok_out 无意义、L0「假失败」（易误判为 kernel bug 甚至误报 `[DESIGN_ERROR]`）。
   **Stage 2 若改 scale/分布，必须先用 `golden_r4` 确认 `valid_ref` 全 1。**
2. **`rand` 种子满足 `seed_i + V < 2^31`**（如 `arange(B)` 或小常量）：保证 kernel `arith_progression`
   的 `seed_i + global_idx`（int32）不溢出 ⟹ 与 `golden_lcg` 的 `seed+i`（Python int 无溢出）逐位一致。
   生产种子可任意（溢出则 LCG 环绕，仅影响分布验证，不影响正确性）。

#### 9.2.2 L0-B 端到端用例矩阵（8 个，bitwise tok_out）

**通用配置**：`dtype=float32`；`K_MAX=256`；`BV=4096`；`VEC_NUM=2`；`LOGIT_SCALE=8.0`；`torch.manual_seed(0)`；
`rand=arange(B)`（除 `l0b_seed_var`）。**每例先跑 K1 得 `tau_out`/`valid`，断言 `valid` 全 1，再跑 K2-Gumbel。**

| 用例名 | B | V | 满块/`TAIL` | k | p | μ | `NEED_MIN_P` | 覆盖目标 |
|--------|---|---|-------------|---|---|---|--------------|---------|
| `l0b_single_block` | 2 | 4096 | 1 / 0 | 40 | 1.0 | 0.0 | `False` | **最小功能**：单块（n_blk=1，无跨块归并）、无尾块、top-k keep；`b_num=1` |
| `l0b_multiblock` | 8 | 16384 | 4 / 0 | 40 | 1.0 | 0.0 | `False` | **跨块 argmax 归并**（4 块，严格 GT + best_idx 更新）、block 整除无尾块 |
| `l0b_topp_nucleus` | 8 | 16384 | 4 / 0 | 40 | 0.95 | 0.0 | `False` | **top-p keep 集** + Gumbel argmax（keep 集大小随 nucleus 变） |
| `l0b_minp_T2` | 8 | 16384 | 4 / 0 | 40 | 0.95 | 0.05 | `True` | **min-p（定理 T2）keep 集** + Gumbel argmax |
| `l0b_prod_v_tail` | 16 | **152064** | 37 / **512** | 40 | 0.95 | 0.05 | `True` | **代表性生产 V + 统一尾块路径**（gm2ub pad -inf，topk actual_num=BV）⟹ C1/C2/N2/N3 全覆盖 |
| `l0b_full_occupancy` | 40 | **152064** | 37 / 512 | 20 | 0.9 | 0.1 | `True` | **满核占用**（40 vec 核 = 20 cube × VEC_NUM 2，`b_num=20`）；最大 DMA 量 + 最大 tmp arena（64 KB） |
| `l0b_k_eq_1` | 2 | 16384 | 4 / 0 | **1** | 1.0 | 0.0 | `False` | **keep 集 = 1 元素**（`tau=m`）：argmax 恒为该元素（Gumbel 不改变结果），验证 keep 掩码退化正确 |
| `l0b_seed_var` | 8 | 16384 | 4 / 0 | 40 | 0.95 | 0.0 | `False` | **种子依赖**：对 `SEED∈{0,42,1234}` 各跑一次，tok_out 随 seed 变化且各自 bitwise 匹配对应 golden（验证 LCG 种子注入正确） |

**矩阵覆盖自检**：单块 / 多块（跨块归并）/ 含尾块 三种分块形态齐备 ✅；三个阈值源（k / p / μ）各自绑定
keep 集 ≥1 例 ✅；`NEED_MIN_P` 两取值 ✅；k 下边界（1）✅；`B` 覆盖 2/8/16/40（含 grid 收缩与满核）✅；
`V` 覆盖 4096/16384/152064 ✅；种子变化（LCG 注入）✅。

#### 9.2.3 L0-A LCG 单元探针（1 个，先于 L0-B 跑）

| 探针名 | 方法 | 判据 |
|--------|------|------|
| `l0a_lcg_unit` | 最小 probe kernel（或直接从 `r4_gumbel_sample` 导出 `state_ub`）对 `M∈{256, 4096}`、`seed∈{0, 42, 1234}` 生成 LCG int32 流，对拍 `golden_lcg(M, seed)`（= `reference_random_1d`） | **int32 逐位相同**（`atol=0/rtol=0`，同 random_1d 的 `assert_close`） |

> L0-A 是 L0-B 的**前置隔离**：若 L0-B 的 tok_out 失配，先查 L0-A（LCG 是否正确），再查 Gumbel（u/G 中间量，
> V-b2），再查 argmax/归并。分级定位，避免全链一起调。对应 V01-a 的 V-1/V-2/V-3 上板探针范式。

#### 9.2.4 每个 L0-B 用例的验证项

| 输出 | dtype | 比对方式 | 期望 |
|------|-------|---------|------|
| `valid`（K1） | int32 | `check_precision(·, valid_ref, "int32")` | **全 1**（§9.2.1 保证；否则先查数据配方，非 kernel bug） |
| `tau_out`（K1，可选回归） | float32 | 对拍 `golden_r4` 的 `tau_ref` | V01-a 已验证逐位一致（回归哨兵，确认 K1 未被 V01-b 改动破坏） |
| **`tok_out`（K2-Gumbel）** | **int32** | **`check_precision(·, tok_ref, "int32")`（0/0/0/1.0 精确）** | **主判据：逐位相同**（`matched_ratio=1.0, max_abs=0`） |

**L0 通过判据（Stage 2 精度收敛门禁）**

```
L0-A：LCG int32 流逐位相同（3 seed × 2 M = 6 项，全 int32 精确）
L0-B：8 用例 × tok_out int32 逐位相同（+ valid 全 1）
    全部满足 matched_ratio==1.0 AND max_abs_error==0（int32 精确）
⟹ 打印 [PRECISION_PASS]，测试进程 exit(0)
任一失败 ⟹ [PRECISION_FAIL]，exit(1)，Stage 2 进入精度调试
          （分级定位：L0-A LCG → u/G 中间量 → argmax/归并；仅当定位为设计层缺陷时才返回 [DESIGN_ERROR]）
```

**L0 明确不含（留给 Stage 2 的 `tilelang-op-test-design` 场景 B）**：② 分布一致性（χ²/KL/熵，§9.5）；
不规则/质数 V；`B=1`/`B=1024`；`k∉[1,K_MAX]`/`TOP_K_ALL`；`p` 极小（触发 `ok_p=0`）；全等行（触发 `tie_ok=0`）；
含 `-inf`/NaN 输入；bf16（不支持，L2 负向）；空 tensor（`B=0`，L2 负向）；`LOGIT_SCALE≠8.0` 平坦分布（守卫负向）；
大 seed（`seed+V≥2^31`，LCG 环绕，仅分布验证）；FUSE_GUMBEL=False 回归（属 V01-a 已验证域）。

### 9.3 精度标准

> 采用**混合容差**：逐元素 `|actual-golden| <= atol + rtol·|golden|`，整体判定 `matched_ratio >= required_matched_ratio`
> **且** `max_abs_error <= max_abs_error_limit`。阈值**仅按 dtype**。完整定义见 `tilelang-op-test-design/references/precision-standard.md`。

| dtype | atol | rtol | max_abs_error_limit | required_matched_ratio | V01-b 适用的对象 |
|-------|------|------|---------------------|------------------------|-----------------|
| **int32** | 0 | 0 | 0 | 1.0（精确匹配） | **`tok_out`（主判据，bitwise）**、LCG state 流（L0-A）、`valid`、`top_ks`/`rand`（输入） |
| **float32** | 2⁻¹⁶ = 1.53e-5 | 2⁻¹⁰ = 9.77e-4 | 1e-2 | 0.99 | `tau_out`（K1 回归）、**Gumbel 中间量 `u`/`G`（若导出对拍，V-b2）**、`logits`/`top_ps`/`min_ps`（输入） |

**§4 声明的每个 dtype 均已在上表有对应行**（int32 与 float32，无其他 dtype —— bf16/float64 属 V01-b 范围外）。

> **主判据是 int32 精确（tok_out bitwise）**，非浮点容差。fp32 混合容差仅用于**中间量诊断**（V-b2 导出 u/G
> 定位 ln ULP 差异）与 K1 tau 回归。这是 V01-b 与 V01-a 精度策略的关键区别：V01-a 主判据是 w_out 的 fp32
> 混合容差；V01-b 主判据是 tok_out 的 int32 bitwise（离散 token id，无「接近正确」——要么对要么错）。

> **特殊浮点值的位置契约**（沿用 V01-a §9.3）：输入允许含 `-inf`（上游屏蔽），不允许 NaN（上游已清洗）。
> V01-b 的 `-inf` 在 keep 掩码处被 `select` 为 `-inf`（cand），argmax 自然排除 ⟹ tok_out 不受影响。
> Gumbel 的 `G∈[-2.81,16.64]` 恒有限（D-CLAMP）⟹ `z+G` 不产生新的 inf/nan（除非 z 本身 -inf，已被掩码）。
> Stage 2 扩展 Boundary 时：`tau_out` 的 `-inf` mask 须精确一致；tok_out 对退化输入（全等行 → valid=0）由 host 回退。

**逐输出的精度分析**

| 输出 | 误差来源 | 量级 | 是否被阈值覆盖 |
|------|---------|------|---------------|
| `tok_out`（int32） | ① LCG int32（**逐位精确**，random_1d 已证）；② u/G 的 fp32 ln ULP（~1e-7 相对）；③ argmax 对 ②的鲁棒性（top-2 gap O(1) ≫ 1e-6） | **0（逐位）**，除非 top-2 近简并（测度 ~0） | ✅ int32 精确匹配；近简并风险见 R-GUMBEL-LN（§10.1），L0 固定种子 + 连续 logits 下不触发 |
| LCG state（int32，L0-A） | int32 mul/add 环绕 | **0（逐位）** | ✅ `random_1d` 的 `assert_close(rtol=0,atol=0)` 已证同款 |
| `u`/`G`（fp32，V-b2 诊断） | cast 舍入 + ln ULP | ~1e-7 相对 | ✅ fp32 混合容差（rtol=9.77e-4 有 ~1000x 余量） |
| `tau_out`（fp32，K1 回归） | V01-a 已验证 | 逐位/≤1.91e-6 | ✅（V01-a §9.3） |

### 9.4 性能可行性哨兵（强制执行，不可因 large/L1 跳过）

| 用例名 | Shape | dtype/属性 | 覆盖路径 | 单 case timeout | 选择理由 |
|--------|-------|-----------|---------|-----------------|----------|
| `perf_worst_gumbel` | `B=64, V=152064` | fp32；`K_MAX=256,BV=4096`；`k=40,p=0.95,μ=0.1,NEED_MIN_P=True`；`LOGIT_SCALE=8.0`；`rand=arange` | K1（37 满块+1 尾块）+ K2-Gumbel（38 块 × [LCG+Gumbel+topk(K=1)+归并]）；**最大 DMA 任务数 + 64 KB tmp arena 同时命中** | **900 s**（含首次 JIT：bisheng dav-2201 编译，秒~分钟级） | 用户关键 case：接近满核（64 行/40 vec 核→2 波）+ 生产 V + 三阈值全绑定；对标 V01-a `perf_worst_k1k2`（1.2 ms），V01-b 应更快（K2 省 exp/div + 省 multinomial） |
| `perf_max_batch` | `B=256, V=152064` | fp32；同上属性 | 同上，**最大 GM 流量**（K1+K2 ≈ 297 MB，§4.4）+ `b_num=128` 超订 | **1200 s** | 最大任务数 case（每核 `ceil(256/40)=7` 行）；对标 V01-a（3.6 ms + multinomial 5.9 ms），V01-b 消除 multinomial 5.9 ms |

**执行约束**（沿用 V01-a §9.4 + `design-template.md §9.4`）：
- 测试数据、随机数（`rand` 种子）、golden（`golden_lcg`/`golden_gumbel_sample`/`golden_r4`）的 LCG/sort/argmax
  **均在 CPU 完成**；运行阶段只做 **H2D → TileLang kernel（K1, K2-Gumbel）→ D2H**，避免测试 harness 引入 aclnn。
- **V01-b 消除 host `torch.multinomial`** ⟹ 哨兵**不含** multinomial（对比 V01-a 哨兵含 multinomial 5.9 ms）。
  哨兵应体现 V01-b 的端到端收益（K1 + K2-Gumbel，无第三步）。
- 哨兵用例**必须实际执行并计时**，不得以「large/L1」为由跳过；超时即失败。首次 JIT 耗时单独记录纳入 warmup 预算。

### 9.5 分布一致性（②，L1/L2 判据，**Stage 2 场景 B 落地**，本文仅给方法论）

> 定理 T3 改变 RNG 流（LCG 替代 multinomial 内部 RNG），故 tok_out **相对 torch.multinomial 不可逐位对拍**。
> ① bitwise 忠实参考（§9.1/§9.2）验证「实现正确性」；② 分布一致性验证「Gumbel-max + 正确 RNG ⟹ 服从目标
> categorical 分布」（草稿 §9.2）。**② 属统计判据，需多 seed 大样本，留 L1/L2 由 Stage 2 扩展**（本 Stage 1 不生成）。

| 检验 | 方法（草稿 §9.2） | 判据 |
|------|-------------------|------|
| 边缘分布 | 固定 logits（一行），用 **N≈100k 个不同 seed** 各采样一次（`rand` 变 seed），统计 token 频率 vs 目标 `softmax(z)·1[z≥tau]` 重归一化 | **χ² 检验 `p > 0.01`** |
| 与参考的 KL | `D_KL(p_ref ‖ p_ours)`（p_ref = 目标 categorical，p_ours = 采样经验分布） | **`< 1e-3`** |
| top-k 覆盖率 | 采样落在真实 nucleus（keep 集 `S_i`）内的比例 | **`> 1 - 1e-4`** |
| 熵 | 采样经验熵 vs 理论熵 `-sum(p̃ ln p̃)` | **相对误差 < 1%** |

**Stage 2 落地要点（场景 B）**：② 用 `_run_precision` 之外的**统计判据函数**（χ²/KL/熵不是逐元素容差）；
多 seed 通过循环 `rand=full((B,), seed)` 实现（每 seed 一次 kernel 调用，或 B=N 行同 logits 不同 seed 一次调用）；
大样本 N≈100k 注意 JIT 缓存（B 进 JIT 键 ⟹ 用固定 B、变 seed 值，避免重编译风暴，R-JIT）。**② 失败不阻塞 L0**
（L0 只认 ① bitwise）；② 是 L1/L2 的分布门禁。

---

## 10. 风险点与注意事项

### 10.1 已知约束与风险（V01-b 专项；V01-a 的 R-A~R-JIT 见其 DESIGN.md §10.1，仍适用）

| # | 风险 | 等级 | 影响 | 应对 / 证据 |
|---|------|------|------|-----------|
| **R-ARGMAX** | 草稿的 argmax-with-index 原语 `wholereducemax(ORDER_VALUE_INDEX)` 已 `@deprecated()` | 🟢 已消解 | 若照抄草稿会 DeprecationWarning + 未来移除 | 改用 `T.tile.topk(K=1)`（wheel `:1528` deprecated 证据；`test_..._topk.py:324` K=1 已测；`moe_topk_gate` 生产链）。顺带消解草稿 R10/R11 |
| **R-GUMBEL-LN** | **bitwise 唯一残留风险**：`torch.log`（host fp32）vs Ascend `T.tile.ln`（硬件 fp32）差 ~1 ULP ⟹ G 差 ~1e-7 ⟹ 若 top-2 候选 `z+G` 差距 < ~1e-6 则 argmax 翻转 | 🟡 中（需知悉，L0 固定种子下几乎不触发） | 个别 shape 的 tok_out 可能与 golden 差 1 个 index | ① 连续 `randn×8` logits + O(1) Gumbel ⟹ top-2 gap 典型 O(0.1~1) ≫ 1e-6，**L0 固定种子下 bitwise 几乎必然成立**；② 若翻转：属真实数值边界（两 index 均为 ULP 扰动下合法 argmax），**非 kernel bug，不误报 `[DESIGN_ERROR]`**——换种子或记录；③ V-b2 先对拍 u/G 中间量量化 ln ULP 差异；④ 分布验证（②）不受影响（ULP 不改变分布） |
| **R-LCG-MATURITY** | `random_1d` 在 `examples_experiment/`（草稿 R13），成熟度未定；V01-b 依赖其 LCG 范式 | 🟢 低 | 仅影响 V01-b 的 RNG | ① LCG 公式极简（`state=seed+idx`，3 轮 `A*state+C`，int32 环绕），`random_1d` 的 `reference_random_1d`（`ctypes.c_int32`）给出 **bitwise golden**（`atol=0/rtol=0`）；② **L0-A LCG 单元探针**（§9.2.3）先隔离验证 int32 环绕 bitwise 正确，再验全链；③ V-b1 上板对拍 |
| **R-INT32-WRAP** | `T.tile.mul/add` 对 int32 是否 mod 2^32 环绕（与 `ctypes.c_int32` 一致） | 🟢 低（已证） | LCG 正确性 | `random_1d.py:59-61/93` 的 kernel `T.tile.mul/add` int32 与 `reference_random_1d`（`ctypes.c_int32`）`assert_close(rtol=0,atol=0)` **精确匹配** ⟹ 环绕语义已证。**无需 `bitwise_and`**（硬件自动截断）。V01-b 内联同款，V-b1 复验 |
| **R-UMAP** | u 映射的 int32→fp32 cast 舍入是否与 torch `.to(float32)` bitwise 一致 | 🟢 低 | 影响 u/G 的 bitwise（但 tok_out 对 ULP 鲁棒，见 R-GUMBEL-LN） | `state` 经 `T.tile.cast(...,"CAST_RINT"/"CAST_NONE")` → fp32；大整数按 RNE 舍入（与 torch 同）。V-b3 上板对拍 u 中间量确认 cast mode；即便差 ULP，tok_out 仍鲁棒 |
| **R-RAND-SCHEME** | rand 方案（每行种子 `[B]` + 内联 LCG）偏离任务/草稿的 `[B,BV]` 提示 | 🟡 中（**分歧点，上报 §10.4**） | proto 接口 shape | D-RAND 决策为 `[B] int32`（Design C）：唯一同时满足 §8.3 访存（2 读）+ bitwise 可复现 + random_1d 复用的方案。**上报 Orchestrator/用户确认**；若坚持 `[B,BV]`/`[B,V]`（Design D）则访存退化到 3 读（445 MB，仍优于 V01-a 594 MB） |
| **R-UB-V01B** | V01-b K2 的 UB 预算（草稿 §5.4.2 估 104 KB 有误） | 🟢 已消解 | 编译期 UB 溢出 | §4.5 复核：`BV=4096` + 正确 arena 模型（topk 64 KB）⟹ **≈112.6 KB ≤ 184 KB（余 39%）**。草稿 104 KB 是 `BV=8192` 且漏算 arena（真实 ≈217 KB 溢出）⟹ **V01-b 固定 `BV=4096`** |
| **R-K1-REGRESSION** | V01-b 改动 host wrapper / r4_ref.py 可能破坏 V01-a 已验证的 K1/K2(w_out) | 🟢 低 | V01-a 回归 | ① K1 `r4_topk_stats` + K2 `r4_mask_sample` **逐字不动**（§0.2）；② V01-b 新增独立 `r4_gumbel_sample`；③ host wrapper 仅在 `FUSE_GUMBEL=True` 分支走新路径，`False` 分支原样；④ L0-B 含 `tau_out` 回归哨兵（§9.2.4）确认 K1 未破坏 |
| **R-JIT** | `B` 进入 JIT 特化键（V01-a 同）；分布验证 N≈100k 若变 B 会重编译风暴 | 🟡 中 | 首 token 延迟 / 测试耗时 | V01-a §4.6 R-JIT 缓解（持久 `TILELANG_CACHE_DIR` + 禁 `clear_cache`）；分布验证用**固定 B、变 seed 值**（§9.5），避免 B 变化重编译 |
| **R-F64 / R-CUMSUM / R-DEPR** | float64 零支持 / cumsum 不可用 / wholereduce deprecated | 🟢 低（V01-b 不涉及或已消解） | — | float64：确定性 seed 路径排除（out of scope）；cumsum：Gumbel 路径不用（argmax 非前缀和）；wholereduce：改 topk(K=1)（R-ARGMAX） |

### 10.2 Stage 2 上板验证项（按优先级，实现前必跑；对应 V01-a §10.2 的 V-1~V-4 范式）

| # | 验证项 | 方法 | 若不通过的兜底 |
|---|--------|------|--------------|
| **V-b1** | **LCG int32 环绕 bitwise**（R-LCG-MATURITY/R-INT32-WRAP）—— L0-A 探针 | 最小 probe kernel：`arith_progression(state, seed+base,1,M)` + 3 轮 `mul(A)/add(C)`，导出 int32 流，对拍 `reference_random_1d(M,seed)`（M∈{256,4096}, seed∈{0,42,1234}），`assert_close(rtol=0,atol=0)` | 若 in-place mul/add 或 scalar A/C 失败 → 回退 random_1d 原样（分离 `temp_ub` + fill `a_ub`/`c_ub` buffer，多占 32 KB，§4.5 余量足够） |
| **V-b2** | **Gumbel 中间量 u/G 对拍**（R-GUMBEL-LN/R-UMAP）—— 量化 ln ULP | probe 导出 `u`（clamp 后）与 `G`，对拍 `golden` 的 fp32 `u`/`G`，用 fp32 混合容差（§9.3）测 max_abs；确认 ln ULP 差异量级 ≪ top-2 gap | 若 ln ULP 过大导致 tok 翻转 → 记录该边界（非 bug）；或 golden 改用 torch_npu 在 NPU 上算 G（同硬件 ln，bitwise 对齐） |
| **V-b3** | **int32→fp32 cast mode**（R-UMAP） | probe 对 `state`（含大整数 >2^24）用 `CAST_RINT` 与 `CAST_NONE` 各导出 fp32，对拍 torch `.to(float32)`，选 bitwise 一致的 mode | 若两 mode 均差 ULP → 任一即可（tok_out 对 ULP 鲁棒，R-GUMBEL-LN）；记录差异 |
| **V-b4** | **topk(K=1) + 跨块标量归并**（argmax-with-index 端到端） | 最小 probe：对 `[BV]` cand（含 -inf 掩码）跑 `topk(K=1)`，标量读 `dst[0]/[1]`，`with T.If/T.Then` 跨块归并，对拍 `torch.argmax`；验证 tie-break=最小 index、-inf pad 不被选、best_idx 全局偏移正确 | 若标量读 `dst[0]/[1]` 失败 → 改 `gather_mask("P0101"/"P1010")`+`cast`（moe_topk_gate 生产链）；若 `[1]` buffer 的 `T.If` 归并失败 → 改 `T.tile.compare("GT")`+`T.tile.select`（`[1]` buffer，V01-a `T.tile.max` on `[1]` 已证同类可行） |
| **V-b5** | **`compare` scalar src1 + `select` scalar -inf**（keep 掩码） | probe 验证 `compare(mask, z_ub, t_v[0,0], "GE")`（scalar src1 → compare_scalar）与 `select(z_ub,mask,z_ub,-inf,"VSEL_TENSOR_SCALAR_MODE")` 正确 | 若 scalar compare 失败 → 回退 V01-a 的 `broadcast(t2d,t_v)`+`compare(mask,z_ub,t2d,"GE")`（多占 16 KB t2d，§4.5 余量足够） |
| **V-b6** | **端到端最小闭环**（= `l0b_single_block`） | `B=2,V=4096,k=40,p=1.0,μ=0,rand=arange`，K1+K2-Gumbel，tok_out 对拍 `golden_gumbel_sample` bitwise | 按 `tilelang-error-fixer`/`tilelang-debug-helper` 分段定位（V-b1→V-b2→V-b4 逐级） |

### 10.3 常见错误与已知偏差（V01-b 必须规避 V01-a 的 10 项实现层坑，来自 `debug_log.md`）

| V01-a 已知坑（debug_log.md） | V01-b 如何规避 |
|------------------------------|---------------|
| ① narrow-reduce `real_shape`/BufferRegion ≤256B(64 fp32) 硬约束（`codegen_ascend.cc:2145`） | **V01-b K2 无 `T.reduce_*`**（argmax 走 topk）⟹ **完全不触发**（§3.5.1/§6.4）。尾块统一路径，无需精确尺寸缓冲 |
| ② 标量 `T.alloc_var` 重绑定结果错误(返回0)、标量 `T.exp/T.log` 无法 codegen、`alloc_var(init=T.infinity())` 被拒 | 标量全用 `[1]`/`[1,1]` `alloc_shared` buffer + `T.tile.*`（ln/max/mul/add/fill）；**Gumbel 的 `-ln(-ln(·))` 用 `T.tile.ln` 作用于 `g_ub[BV]` 向量 buffer，绝不用标量 `T.log`**（§3.3） |
| ③ 1 元素 UB→UB `T.copy` 触发 aicore 异常 507015（<32B DMA 对齐） | 标量 buffer 间**禁用 `T.copy`**；running argmax 用 `with T.If/T.Then` 标量赋值（`best_val[0]=topk_dst[0]`）或 `T.tile.*` 就地更新；**GM↔UB 的 1-elem copy 正常**（tau/seed 读、tok 写） |
| ④ `T.serial(0)`（V<BV→n_blk_full=0）消除循环体致死 buffer "Cannot find pre-allocated address"（`codegen_ascend.cc:828`） | V01-b 用 `T.serial(n_blk)`（`n_blk=ceildiv(V,BV)≥1` 恒正，含尾块）⟹ **不会出现 `T.serial(0)`**；buffer 分配在 `if row<B` 内、循环外，恒被触及 |
| ⑤ `T.tile.topk` 对 fp32 in-place 写 -inf padding，只作用 UB scratch，禁喂 GM 视图 | topk 源是 UB scratch `z_ub[BV]`（非 GM）；`BV%32==0` ⟹ 无越界（N3，§3.5.1） |
| ⑥ `TOP_K_ALL=1<<30`（非 -1）；valid 守卫 = k_ok AND ok_p AND tie_ok | K1 逻辑不变（V01-b 复用 K1 的 tau/valid）；K2-Gumbel 不重算守卫，host 按 valid 分派回退 |
| ⑦ 仅被 T.Parallel 触及的 buffer 作用域无法解析 → 每个 alloc_shared buffer 至少被一个 T.copy/T.tile.*/T.reduce_* 触及 | 每个 buffer 都被 `T.tile.fill`/`T.copy`/`T.tile.*` 触及（§3.3 伪代码：`fill(best_val/best_idx/tok_i32)`、`copy(t_v/seed_v)`、其余被 arith_progression/cast/topk 等触及） |
| ⑧ `@T.prim_func` 内嵌套 def 被拒 | V01-b K2-Gumbel 循环体**内联**（无嵌套 def；§3.3） |
| ⑨ `T.If` 必须配 `with T.Then():` | 跨块归并用 `with T.If(topk_dst[0]>best_val[0]): with T.Then():`（§3.3/§6.2） |
| ⑩ topk 源必须 rank-1（`max_actual_num=Σshape` 陷阱） | `z_ub` 直接用 rank-1 `[BV]`（§3.3 伪代码说明 1；N2/R-C） |

**V01-b 新增注意事项**：
- **`random_1d.py:7 clear_cache()` 禁止照抄**（R-JIT，生产用持久缓存；§3.5.2/§11.5）。
- **`random_1d.py` 的 `alloc_ub` 改 `alloc_shared`**（Developer 形态；§3.5.2）。
- **`moe_topk_gate.py` 的手动 `T.set_flag/wait_flag/barrier_all/pipe_barrier` 不照抄**（V01-b 全自动同步；§7.1）。
- **u 映射用 `·2^-32+0.5` 而非草稿 `·2^-31`**（D-UMAP，修正符号 bug）。

### 10.4 相对 V01-a / 草稿的关键调整（本增量设计的差异化说明）

> 本 V01-b 是 V01-a 之上的**增量**（非回退重做，`design_revision_count=0`）。下列为 V01-b 相对
> **草稿 `DESIGN_script.md`** 与 **V01-a `DESIGN.md`** 的关键调整，及「为何不会重蹈草稿覆辙」。

| # | 位置 | 草稿/V01-a 原状 | V01-b 调整 | 依据 |
|---|------|----------------|-----------|------|
| 1 | argmax-with-index | 草稿 §3.6/§3.7：`wholereducemax(ORDER_VALUE_INDEX)` | **`topk(K=1)` + 标量读/gather_mask+cast** | wheel `:1528` deprecated；`test_..._topk.py:324` K=1 已测；`moe_topk_gate` 生产链（D-ARGMAX） |
| 2 | u 映射 | 草稿 §3.7：`lcg·2^-31`（signed→[-1,1)，负半轴坍缩） | **`lcg·2^-32+0.5`（→[0,1) 均匀）** | D-UMAP（修正草稿符号 bug，否则半数样本共享最小噪声、分布崩坏） |
| 3 | Gumbel clamp | 草稿 D3 担心 `1-2^-32` f32 失效（→G=+inf） | **`[2^-24,1-2^-24]`（fp32 精确，G∈[-2.81,16.64] 全有限）** | D-CLAMP（消解草稿 D3/R1；任务指定，本文核实 fp32 可表示性） |
| 4 | rand 方案 | 草稿自相矛盾（§3.2 `[B,BV]` host 流 vs §5.3.4 内联 LCG vs §8.3 2 读） | **每行种子 `[B] int32` + kernel 内联 LCG（Design C）** | D-RAND（§8.3 访存一致 + bitwise 可复现 + random_1d 复用）；**分歧点上报** |
| 5 | K2 UB 预算 | 草稿 §5.4.2：V01-b ≈104 KB（`BV=8192`，漏算 arena） | **≈112.6 KB（`BV=4096` + topk 64 KB arena 正确计入）** | §4.5（V01-a 修正模型同源；草稿漏算 arena 同 V01-a §10.4 修正 5） |
| 6 | 尾块 | V01-a K1 需精确尺寸 `[1,TAIL]` 缓冲（narrow-reduce 约束） | **统一块路径**（无 reduce ⟹ 无 narrow 约束，gm2ub pad -inf） | §5.4/§6.4（argmax 走 topk，规避 V-2 坑） |
| 7 | LCG 实现 | `random_1d.py`：`alloc_ub` + `clear_cache` + 6 buffer + 手动同步 | **`alloc_shared` + 持久缓存 + in-place（state_ub 单 buffer）+ 自动同步** | §3.5.2（Developer 形态改写） |
| 8 | 验证策略 | 草稿 §9.2：仅分布一致性（"不可逐位对拍"） | **两层：① bitwise 忠实参考（L0 主判据）+ ② 分布一致性（L1/L2）** | §9（任务验证洞察：对忠实 Gumbel-max 参考可 bitwise，比分布验证强得多） |
| 9 | host multinomial | V01-a：保留 `torch.multinomial`（DEV-1 偏差） | **消除**（Gumbel 融合直接出 tok_out） | §0.1/§5.6（草稿 D6：6 op→2 op） |

**为何不会重蹈草稿覆辙**：草稿的三处实质缺陷（wholereducemax deprecated、u 映射符号 bug、K2 UB 漏算 arena）
均已在本文逐一识别并给出已核实的替代/修正（D-ARGMAX/D-UMAP/§4.5），且每处替代都有 wheel 源码 + examples +
官方测试的三重证据（§3.5.4/§3.5.5）。V01-a 的 10 项实现层坑（debug_log.md）已在 §10.3 逐条给出 V01-b 规避方案。

### 10.5 Developer 混合模式可行性最终判定（供 Orchestrator 决策）

**判定：V01-b 全链路（LCG + Gumbel + argmax-with-index + 跨块归并）可在 Developer 混合形态实现，
无任何步骤只能用 Expert 判别标志完成。✅ 无矛盾点需上报（与 V01-a §10.5 结论一致）。**

- LCG ⟹ `T.tile.arith_progression/mul/add`（int32，Developer `alloc_shared`，random_1d 已证，仅改分配形态）✅
- Gumbel ⟹ `T.tile.cast/mul/add/clamp/ln`（act_quant 生产链已证）✅
- argmax-with-index ⟹ `T.tile.topk(K=1)` + 标量读/gather_mask+cast（topk_selector/moe_topk_gate 已证，V01-a K1 已用 topk）✅
- 跨块归并 ⟹ `with T.If/T.Then` 标量更新（V01-a tau_k/prefix 同款，V-3 PASS）✅
- keep 掩码 ⟹ `T.tile.compare`(scalar)+`select`(scalar)（V01-a K2 同款）✅

**唯一残留术语级说明（同 V01-a R-A，非阻塞）**：`topk`/`gather_mask`/`arith_progression`/`clamp`/`ln` 无纯
Developer 高层等价物 ⟹ 属「Developer 主体 + Ascend 专属 `T.tile.xxx`」混合模式（guide §1：大多数实际算子的形态）。
**未使用任何 Expert 判别标志（`alloc_ub`/`T.Scope`/手动 flag/pass_configs 全关），未擅自改模式。**
**不标记 `[DESIGN_ERROR]`** —— 设计完整、可实现、无根本性阻塞。

**需 Orchestrator/用户知悉的分歧点（非阻塞，见 §10.4）**：`rand` 输入 shape 定为 `[B] int32`（每行种子，
Design C），偏离任务/草稿的 `[B,BV]` 提示。依据充分（§8.3 访存 + bitwise 可复现），**默认按 Design C 实现**；
若用户坚持 `[B,BV]`/`[B,V]` host 生成流（Design D），需接受访存退化到 3 读（仍优于 V01-a）。

---

## 11. 交付清单

### 11.1 目录结构（V01-b 在 V01-a 基础上扩展，不新建目录）

```
examples/R4/
├── r4_fused_sampling.py     # 【扩展】K1(不变) + K2 r4_mask_sample(不变) + 新增 r4_gumbel_sample + host wrapper 增 FUSE_GUMBEL 分派
├── r4_ref.py                # 【扩展】golden_r4(不变) + 新增 golden_lcg + golden_gumbel_sample
├── test_r4_fused_sampling.py# 【扩展】V01-a 用例(不变) + 新增 L0-A/L0-B（bitwise）；L1/L2 分布由 Stage 2 场景 B 扩展
├── proto.yaml               # 【扩展为超集】inputs 增 rand，outputs 增 tok_out，attrs 增 FUSE_GUMBEL（§11.6）
├── DESIGN.md                # 【不变】V01-a 已验证设计（冻结，禁改）
├── DESIGN_v01b.md           # 【本文】V01-b 增量设计
├── DESIGN_script.md         # 【不变】原始草稿
├── debug_log.md             # 【Stage 2 追加】V01-b 实现日志
└── history_version/         # 备份
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN_v01b.md` | ✅ 本文 | V01-b 增量设计文档（含 L0 门槛测试计划 §9.2） |
| `proto.yaml` | ✅ 本次更新 | V01-a+V01-b 超集（§11.6），覆盖门禁 `coverage_check.py --proto` 用 |
| `r4_fused_sampling.py`（+`r4_gumbel_sample`） | ⬜ 待实现（Stage 2） | 新 kernel，纯 `@tilelang.jit`；K1/K2(w_out) 不动 |
| `r4_ref.py`（+`golden_lcg`/`golden_gumbel_sample`） | ⬜ 待实现（Stage 2） | 忠实 Gumbel-max golden（bitwise）；`golden_r4` 不动 |
| `test_r4_fused_sampling.py`（+L0-A/L0-B） | ⬜ 待实现（Stage 2） | bitwise tok_out + LCG 单元；分布 L1/L2 由场景 B 扩展 |

### 11.3 命名规范

- 新 kernel：`r4_gumbel_sample`（snake_case，与 `r4_topk_stats`/`r4_mask_sample` 同族）。
- golden：`golden_lcg` / `golden_gumbel_sample`（与 `golden_r4` 同族）。
- 测试用例：`l0a_*`（LCG 单元）/ `l0b_*`（bitwise 端到端），区别于 V01-a 的 `l0_*`。

### 11.4 实现顺序（Stage 2）

1. ✅ 设计文档（本文）+ proto.yaml 更新（§11.6）+ L0 门槛计划（§9.2）
2. ⬜ `r4_ref.py` 扩展：`golden_lcg`（复用 reference_random_1d）+ `golden_gumbel_sample`（镜像 §3.3 算术）
3. ⬜ `r4_fused_sampling.py` 扩展：`r4_gumbel_sample` kernel + host wrapper `FUSE_GUMBEL` 分派
4. ⬜ **V-b1~V-b6 上板探针**（§10.2，实现前/中必跑，分级隔离 LCG→Gumbel→argmax）
5. ⬜ L0-A（LCG 单元 bitwise）→ L0-B（tok_out bitwise）收敛通过（`--level l0`）
6. ⬜ Stage 2 场景 B 扩展 L1/L2（含 ② 分布一致性 §9.5）+ Boundary + 覆盖门禁 `coverage_check.py --proto`
7. ⬜ 性能哨兵（§9.4）+ V01-a 回归（确认 K1/K2(w_out) 未破坏）

### 11.5 运行 / 复现说明（Stage 2 必须遵守，沿用 V01-a §11.5）

- **用已安装 wheel**（`tilelang 0.1.4+d75300ed` @ `~/.local/lib/python3.12/site-packages/tilelang`）。
- **不要 `source set_env.sh`**（仓库源码未编译会 shadow → `No module named tvm`）。
- 从**中立 cwd / script dir** 运行（如 `/tmp/opencode/...`），`import tilelang` 须解析到 wheel 路径。
- 设 **`ACL_OP_INIT_MODE=1`**；持久 **`TILELANG_CACHE_DIR`**。
- **禁止 `tilelang.disable_cache()` / `tilelang.cache.clear_cache()`**（R-JIT：B 进 JIT 键，靠持久缓存避免重编译风暴）。
  ⚠️ **`random_1d.py:7` 的 `clear_cache()` 是演示用，V01-b 禁止照抄**（§3.5.2/§10.3）。
- 环境事实：Ascend910_9362 → A3/arch22（dav-2201）；20 Cube/40 Vector（运行时 `detect_vec_core_num`，
  `VEC_NUM=2`，**禁硬编码 48**）；UB 192KB（向量可用 184KB）；CANN 9.2.0；torch 2.7.1/torch_npu 2.7.1.post4。

### 11.6 算子 proto.yaml（覆盖门禁用，V01-a+V01-b 超集，本次更新）

> 见同目录 `examples/R4/proto.yaml`（本次更新为 V01-a+V01-b 超集）。相对 V01-a 的变化：
> **inputs 增 `rand`（int32）**；**outputs 增 `tok_out`（int32）**；**attrs 增 `FUSE_GUMBEL`（Bool）**；
> formula 增定理 T3 + Gumbel + LCG。dtype 全集仍为 `{float32, int32}`（与 §9.3 精度表一致，无新增 dtype）。
> 覆盖门禁 `coverage_check.py --proto` 据此派生 V01-b 的 D-DTYPE-int32（rand/tok_out）与 D-PARAM-FUSE_GUMBEL 维度。

---

## 修订记录

| 版本 | 日期 | 变更 | 作者 |
|------|------|------|------|
| V01-b.0 | 2026-10-07 | 初版：V01-b Gumbel-max 融合采样增量设计（Stage 1，mode=first_design，attempt_index=1，design_revision_count=0）。基于 V01-a（已 NPU 验证）扩展 K2 Gumbel 路径；核心决策 D-ARGMAX（topk K=1 替代 deprecated wholereducemax）/ D-RAND（每行种子 [B]+内联 LCG，Design C，分歧点上报）/ D-UMAP（·2^-32+0.5 修正草稿符号 bug）/ D-CLAMP（[2^-24,1-2^-24] 消解 D3）/ D-TIE-ARGMAX（严格 GT=最小 index）；两层验证（① bitwise 忠实参考 L0 主判据 + ② 分布一致性 L1/L2）；UB 复核 ≈112.6 KB≤184 KB；Developer 混合模式可行（无 Expert 步骤）。 | tilelang-op-analyst (Stage 1) |
