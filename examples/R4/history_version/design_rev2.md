# r4_fused_sampling 算子设计文档

> **范围声明（V01-a，已由用户确认收敛）**
> 本设计只覆盖 **V01-a**：K1 `r4_topk_stats` + K2 `r4_mask_sample` 两个 kernel，输出屏蔽后概率
> `w_out[B,V]`，**host 侧保留 `torch.multinomial` 做最终采样**；**仅 fp32**；**单特化 S1'**
> （`K_MAX=256, BV=4096`），代表性 `V=152064`。
>
> **明确排除（out of scope，本设计不涉及）**：V01-b 的 Gumbel-max 融合 / `tok_out` /
> `r4_lcg.py` / LCG 随机数；SGLang backend 注册（`r4_dispatch.py`、`register_sampler_backend`）；
> logprob 融合；确定性 seed 路径（依赖 float64，TileLang 零支持）；ACLGraph 入图；完整多特化表；bf16。
>
> **算法与需求的权威参考**：同目录 `DESIGN_script.md`（1161 行草稿，Expert 模式）。
> 本文吸收其数学（§3.5–3.7、定理 T1/T2）与边界分析（§7），但 **API 映射已按用户指定的
> Developer 模式全量重做**，且草稿中若干结论经源码复核后**已修正**（见 §10.4 修正清单）。

---

## 1. 概述

### 1.1 算子名称

`r4_fused_sampling` —— 融合采样链（Fused Top-k / Top-p / Min-p Sampling）

由两个 TileLang kernel 组成：

| Kernel | 名称 | 职责 |
|--------|------|------|
| **K1** | `r4_topk_stats` | 单遍分块扫描 vocab，产出 online-softmax 统计量 `m`/`s`、合成阈值 `tau`、降级守卫 `valid` |
| **K2** | `r4_mask_sample` | 二次扫描 vocab，按单阈值 `tau` 屏蔽并重算概率，写出 `w_out[B,V]` |

### 1.2 功能描述

输入 temperature 已缩放的 logits `[B, V]`，把 SGLang 采样链中
「softmax → top-k 截断 → top-p(nucleus) 截断 → min-p 截断」四步融合为 2 个 kernel，
输出屏蔽后（未重归一化）概率 `w_out[B,V]` 交 host 侧 `torch.multinomial` 采样。

**计算语义**（记第 `i` 行、vocab 位置 `j`，`z = logits`）：

```
m_i   = max_j z_ij
s_i   = sum_j exp(z_ij - m_i)
p_ij  = exp(z_ij - m_i) / s_i                                  # softmax
S_i   = { j : rank_desc_i(j) < k_i }
        ∩ { j : excl_prefix_i(j) <= p_i }
        ∩ { j : p_ij >= (1/s_i) * mu_i }
w_ij  = p_ij * 1[j in S_i]                                     # 不重归一化（multinomial 视其为权重）
```

**适用场景**：LLM 推理 decode 阶段每 step 必调的输出级采样；纯访存瓶颈型算子
（计算量近乎为零，代价全在 `[B,V]` 全 vocab 反复遍历）。NPU 上 `torch.compile` 被禁用
（`sampler.py:882 @torch.compile(dynamic=True, disable=is_npu())`），无法靠图编译消解，只能算子融合。

### 1.3 数学公式

**核心化简（定理 T1，源自 `DESIGN_script.md` §3.6，本设计沿用并复核）**

top-k / top-p / min-p 三者均为「按值单调的阈值筛选」，且 `masked_fill_(…, 0.0)` 不改变其他
元素的值，故顺序施加等价于取阈值最大者：

```
S^(k) ∩ S^(p) ∩ S^(mu)  =  { j : z_ij >= tau_i },
tau_i = max( tau^(k)_i , tau^(p)_i , tau^(mu)_i )
```

**定理 T2（min-p 阈值在 z 域的闭式，`s` 完全消去）**

由 `max_j p_ij = 1/s_i`（top-k 必保留最大值元素）：

```
p_ij >= mu_i / s_i
  <=>  exp(z_ij - m_i) / s_i >= mu_i / s_i
  <=>  exp(z_ij - m_i) >= mu_i
  <=>  tau^(mu)_i = m_i + ln(mu_i)          （mu_i = 0  =>  tau^(mu)_i = -inf）
```

**定理 T3（Gumbel-max 免除归一化与对数）—— 本 V01-a 范围不使用**，仅记录：T3 服务于 V01-b
的 kernel 内采样。V01-a 保留 `torch.multinomial`，故 K2 仍需显式计算 `exp` 与 `div`。

**三个阈值的求法**

| 阈值 | 表达式 | 求解位置 |
|------|--------|---------|
| `tau^(k)` | `run_v[k_i - 1]`（跨块归并得到的降序 top-`K_MAX` 列表第 `k_i` 项） | K1 收尾 |
| `tau^(p)` | 在降序 top-`K_MAX` 列表上做**严格前缀和（不含自身）**扫描，取最后一个满足 `prefix <= p_i` 的 `run_v[l]` | K1 收尾，UB 内标量循环 |
| `tau^(mu)` | `m_i + ln(mu_i)`（定理 T2）；`mu_i = 0` 或 `NEED_MIN_P=False` 时取 `-inf` | K1 收尾 |

**top-p 判据与被替换实现严格对齐**（已读 `sglang/python/sglang/srt/layers/sampler.py:858-860` 核对）：

```python
probs_sum  = torch.cumsum(probs_sort, dim=-1)            # 含自身的前缀和
top_p_mask = probs_sum - probs_sort > top_ps.view(-1,1)  # 减自身 → 严格前缀和（不含自身）
probs_sort.masked_fill_(top_p_mask, 0.0)                 # 严格 > 才屏蔽 → 边界元素保留
```

即保留秩 `l` 当且仅当 `sum_{t<l} q_it <= p_i`。本设计的标量循环用同一判据（§3.3 / §6.4）。

**min-p 判据**（`sampler.py:862-865`，严格 `<` 屏蔽 ⟹ 相等保留）与本设计的 `z >= tau` 精确等价（定理 T2）。

### 1.4 算法描述

**Pass 1（K1，单次 GM 读）** —— 分块 `b = 0 … n_blk-1`，`n_blk = ceildiv(V, BV)`：

```
init: m <- -inf,  s <- 0,  run_v[0:K_MAX] <- -inf
for b in blocks:
    z <- GM[logits[row, b*BV : (b+1)*BV]]                      # T.copy（尾块自动 clamp）
    # (a) online softmax（单遍、分块、数值稳定）
    b_max <- reduce_max(z, valid_cols)                         # T.reduce_max 全量归约（尾块用精确尺寸 [1,TAIL] 缓冲，N1/V01.1）
    m_new  = max(m, b_max)
    s      = s * exp(m - m_new)                                # 重标定
    m      = m_new
    e      = exp(z - m)                                        # T.Parallel
    s      = s + reduce_sum(e, valid_cols)                     # T.reduce_sum 全量归约（同上，§10.2 V-2）
    # (b) 块内 top-K_MAX → 跨块分层归并
    copy(z -> zt)                                              # rank-2 → rank-1（§5.3 N2）
    T.tile.topk(l_dst, zt, K_MAX, valid)                       # [val,idx] 交错，降序
    l_val <- T.tile.gather_mask(l_dst, "P0101")                # 取偶数位 = 值，弃 idx
    merged <- concat(run_v, l_val)                             # 2*K_MAX
    T.tile.topk(m_dst, merged, K_MAX, 2*K_MAX)
    run_v  <- T.tile.gather_mask(m_dst, "P0101")               # 仍为降序 top-K_MAX

# 收尾（全部在 K_MAX=256 的 UB 小数组上，代价可忽略）
tau_k <- (1 <= k_i <= K_MAX) ? run_v[k_i - 1] : -inf
tau_u <- NEED_MIN_P ? (mu_i > 0 ? m + ln(mu_i) : -inf) : -inf        # 定理 T2
tau_p <- UB 内标量循环求严格前缀和（替代不可用的 T.cumsum）
n_gt, n_eq <- 在 run_v 上向量化统计                                   # D-TIE
tau   <- max(tau_k, tau_p, tau_u)                                    # 定理 T1
valid <- (1 <= k_i <= K_MAX) AND ok_p AND tie_ok                     # D-GUARD
写 GM: m_out, s_out, tau_out, valid
```

**Pass 2（K2，单次 GM 读 + 单次 GM 写）**：

```
for b in blocks:
    z <- GM[logits[row, b*BV : ...]]
    w <- select(z >= tau_i, exp(z - m_i) / s_i, 0.0)             # T.Parallel + T.if_then_else
    GM[w_out[row, b*BV : ...]] <- w                              # T.copy（尾块自动 re-clamp）
```

**为什么必须分层归并（不可全 vocab 一次 top-k）**：`V=152064` fp32 = **594 KB ≫ 184 KB** 可用 UB
（超约 3.2 倍）。`T.tile.topk` 要求 src 全量驻留 UB，故必须分块 + 归并；归并后 UB 占用
**与 V 无关**，只与 `K_MAX` 相关。

**分层归并正确性（归纳证明）**：任一元素若属于全行 top-`K_MAX`，则它必属于其所在块的
top-`K_MAX`（块内 top-K 是全行 top-K 的超集限制）。设处理完前 `t` 块后 `run_v` = 前缀
`blk_0..blk_{t-1}` 的 top-`K_MAX`；第 `t+1` 步取 `run_v ∪ local_val` 的 top-`K_MAX`，
而全行前 `t+1` 块的 top-`K_MAX` 必落在该并集内 ⟹ 归纳成立。循环结束后 `run_v` = 全行
top-`K_MAX`（降序）。∎

### 1.5 数据流图

```
GM logits[B,V] --T.copy(分块 BV)--> UB z_ub[1,BV]
                                      |
              +-----------------------+----------------------------+
              v                       v                            v
   T.reduce_max(dim=-1)      T.Parallel: e=exp(z-m)      T.copy --> zt_ub[BV] (1D)
   全量归约(V01.1/N1)        T.reduce_sum(dim=-1)        T.tile.topk(K_MAX, valid)
   -> m_blk[1]               全量归约(V01.1/N1)          -> l_dst[2*K_MAX]
   online 更新 m,s            -> s_blk[1] -> 累加 s        T.tile.gather_mask("P0101")
                                                          -> l_val[K_MAX]
                                                                    |
                                     run_v[K_MAX] --concat--> merged[2*K_MAX]
                                                                    | T.tile.topk(K_MAX, 2*K_MAX)
                                                                    v
                                                          run_v[K_MAX]（跨块归并）
                                                                    |  (循环 n_blk 次)
                                                                    v
                             UB 标量/小向量收尾: tau_k, tau_p, tau_u -> tau ; tie -> valid
                                                                    | 标量写 GM
                                                                    v
                                     GM m_out[B], s_out[B], tau_out[B], valid[B]
   ======================= K1 / K2 边界 =======================
GM logits[B,V] --T.copy--> UB z_ub[1,BV] --T.Parallel(select+exp+div)--> UB w_ub[1,BV]
                                                                    | T.copy
                                                                    v
                                                              GM w_out[B,V]
                                                                    | (host)
                                                                    v
                                              torch.multinomial(w_out, 1) -> token_id[B]
```

> **Host 侧 Buffer 操作约束合规**：R4 的 host 路径只做「张量分配 + kernel 调用 + 结果验证/分派」，
> 不含 `.contiguous()` / `reshape` 物理化 / `torch.cat` / `F.pad` / `.to(dtype)` 等。
> `torch.multinomial(w_out, 1)` 是 **V01-a 用户明确保留的下游独立算子**（非 R4 自身计算的
> host 替代），完整审计见 §5.6。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Developer 模式**（用户明确指定，非默认推导 —— 符合 AGENTS.md 核心原则 6）

具体落地形态：

- 内存分配：**`T.alloc_shared`**（不用 `T.alloc_ub` / `T.alloc_L0*`）；标量用 `T.alloc_var`
- 作用域：**不写 `T.Scope("C"/"V")`**（由 `TL_ASCEND_AUTO_CV_COMBINE` 自动处理）
- 同步：**不写 `T.barrier_all` / `T.set_flag` / `T.wait_flag`**（由 `TL_ASCEND_AUTO_SYNC` 自动插入）
- 计算表达：**高层 op 优先** —— `T.copy` / `T.reduce_max` / `T.reduce_sum` /
  `T.Parallel` + `T.exp` / `T.max` / `T.log` / `T.if_then_else`
- pass_configs：四个自动类开关**全开**（§7.3）

### 2.2 选型理由

算子特征分析（依据 `tilelang-op-design` §5 决策树）：

| 特征 | 判定 | 依据 |
|------|------|------|
| 计算类型 | **纯 Vector** | 只有 reduction / elementwise / selection，**零 matmul** ⟹ 不使用 Cube、不触碰 L1/L0A/L0B/L0C |
| 复杂度级别 | **Fusion（多算子融合）** | softmax + top-k + top-p + min-p 四步融合，跨块归并 + 收尾标量算法 |
| 数学特征 | Reduction + Selection | `max(dim)` / `sum(dim)` / `exp` / `ln` / 排序选择 |
| 动态 shape | 部分 | `B` 运行时变（仅影响 grid）；`V` 编译期静态 |

⟹ 纯 Vector + 零 Cube ⟹ **不需要 Expert 模式的 L0A/L0B/L0C 显式分配与手动 CV 同步**；
Developer 模式的自动内存规划（`TL_ASCEND_MEMORY_PLANNING`）恰好用于消化本算子大量的中间 UB buffer。
用户已明确指定 Developer，本节结论与用户选择一致，**未使用默认值绕过**。

### 2.3 ⚠️ 关键澄清：`T.tile.topk` 在 Developer 模式下的定位（必读）

**用户提出的核心张力**：草稿 §5.3.3 全程用 Expert 低层原语（`T.alloc_ub` + `T.Scope("V")` +
`T.tile.wholereducemax/wholereducesum/topk/gather_mask` + 手动 barrier），而用户选择 Developer 模式。

**经源码与示例复核后的结论（三条论据）**：

1. **`T.tile.*` 不是 Expert 模式专属命名空间。** 仓库自己的 Developer 模式标杆目录
   `examples/developer_mode/gelu_mul_developer.py` 就是 `T.alloc_shared`（`:26-29`）+
   `T.tile.mul/add/exp/div`（`:33-46`）+ Developer pass_configs（`:8-11`），
   **无 `T.Scope`、无手动同步**。按 `tilelang-programming-model-guide` §1 的分类，
   「Developer 主体 + 少量 Ascend 专属 `T.tile.xxx`」正是**混合模式**，而该 guide 明言
   「大多数实际算子使用混合模式」。Expert 模式的真正判别标志是：`T.alloc_L0*` /
   `T.Scope("C"|"V")` / 手动 `set_flag`+`wait_flag` / pass_configs 全关 —— **本设计一条都不用**。

2. **`T.tile.topk` 可安全作用于 `T.alloc_shared` buffer**（完整证据链，已在 wheel 提交 `d75300ed` 复核）：
   - `src/transform/common/operation_config.h:248-250` 将 `tl.ascend_topk` 标为 **`PIPE_V`**；
   - `src/transform/ascend_infer_buffer_scope.cc:86-95` 的 `IsVectorFunction()` 对非 GEMM、
     不含 `copy/memcpy/dma` 关键字的 op 返回 true ⟹ `tl.ascend_topk` 记为 `used_in_vector`；
   - 同文件 `:398-402`：`original_scope == "shared"` 且 `used_in_vector && !used_in_cube`
     ⟹ **`corrected_scope = "shared.ub"`**；
   - 本算子纯 Vector、零 Cube ⟹ 所有 `alloc_shared` 均落到 UB。✅

3. **草稿依赖的 `T.tile.wholereducemax` / `wholereducesum` 在已安装 wheel 中已被 `@deprecated()`**
   （`~/.local/lib/python3.12/site-packages/tilelang/language/ascend_tile.py:1528 / 1562`；
   `block_reduce_max/min` 亦 deprecated 于 `:1711 / :1753`）⟹ 即便走 Expert 也不该用它们。
   Developer 模式改用**非 deprecated 的 `T.reduce_max` / `T.reduce_sum`**
   （`tilelang/language/reduce_ascend.py:362 / 452`），该路径由
   `examples/developer_mode/sparse_flash_attn_developer_vid_reduce.py:155,166` 与
   `examples/softmax/example_online_softmax.py:75,83` 双重生产级验证。

**⟹ Developer 模式可行性判定：V01-a 全链路（含 top-k 分层归并）可在 Developer 模式实现，
无需改回 Expert，也无需编造 API。** 唯一残留的术语级说明与 2 项上板验证项见 §10.1 R-A / R-B
（**V01.1：Stage 2 已上板验证完毕** —— V-1 ✅ PASS；V-2 发现 narrow-reduce ≤256 B 硬约束后改用精确尺寸尾块缓冲 ✅，见 §10.2）。

### 2.4 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（纯 Vector ⟹ 由 `InferAllocScope` 解析为 `shared.ub`）；标量原设计用 `T.alloc_var`，**V01.1：实测不可靠（重绑定结果错误），已改用 `[1]`/`[1,1]` `alloc_shared` buffer + `T.tile.*` 运算**（`debug_log.md` 发现 1/6） |
| 计算方式 | `T.reduce_max` / `T.reduce_sum`（高层 Ascend fast-path）+ `T.Parallel` elementwise + `T.tile.topk` / `T.tile.gather_mask`（无高层等价物的两步） |
| 作用域 | 编译器自动分离（`TL_ASCEND_AUTO_CV_COMBINE`）；**不写 `T.Scope`** |
| 同步方式 | **全自动**（`TL_ASCEND_AUTO_SYNC`）；不写任何 barrier / flag |
| Kernel 轴 | `T.Kernel(b_num, is_npu=True) as (cid, vid)` + `row = cid*VEC_NUM + vid`（**不用 `threads=2`**，硬约束理由见 §6.5） |
| CV 交互 | **不涉及**（无 Cube）⟹ 无 workspace、无 `workspace_idx`、无跨核 flag |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 | 所在 kernel |
|------|----------|------|------------|
| 1 | `z <- GM[logits[row, b*BV : (b+1)*BV]]` | 分块搬入，尾块自动 clamp | K1 / K2 |
| 2 | `b_max <- max_j z_j`（仅 valid 列） | 块内行最大值 | K1 |
| 3 | `m_new <- max(m, b_max)`；`s <- s * exp(m - m_new)` | online softmax 重标定 | K1 |
| 4 | `e_j <- exp(z_j - m_new)` | 稳定化指数 | K1 |
| 5 | `s <- s + sum_j e_j`（仅 valid 列） | 累加 exp 和 | K1 |
| 6 | `local <- topK_MAX(z)`（降序 val/idx 交错） | 块内 top-K | K1 |
| 7 | `local_val <- local[0::2]` | 取偶数位 = 值，弃 idx | K1 |
| 8 | `run_v <- topK_MAX(concat(run_v, local_val))` | 跨块分层归并 | K1 |
| 9 | `tau_k <- run_v[k_i - 1]` | top-k 阈值 | K1 收尾 |
| 10 | `tau_u <- m + ln(mu_i)`（`mu_i=0` → `-inf`） | 定理 T2 | K1 收尾 |
| 11 | `tau_p <-` 严格前缀和扫描 `run_v` | top-p 阈值（标量循环替代 `T.cumsum`） | K1 收尾 |
| 12 | `n_gt / n_eq <-` 在 `run_v` 上统计 | tie 检测（§3.4 D-TIE） | K1 收尾 |
| 13 | `tau <- max(tau_k, tau_p, tau_u)`；`valid <-` 守卫合取 | 定理 T1 合成 + 降级守卫 | K1 收尾 |
| 14 | `keep_j <- (z_j >= tau)` | 单阈值比较 | K2 |
| 15 | `w_j <- keep_j ? exp(z_j - m)/s : 0.0` | 屏蔽后概率（不重归一化，D-NORENORM） | K2 |

### 3.2 TileLang API 映射（Developer 模式）

> 每一行均给出**已核实的证据**（示例 / 测试 / 源码行号）。遵循 AGENTS.md 核心原则 1：
> 先查 `tilelang-api-best-practices`，再查 `examples/` 同类实现，文档未覆盖再查源码。本表全部经检索确认，无凭记忆猜测项。

| 步骤 | TileLang API（Developer） | 参数要点 | 证据（已核实） | 替代方案 |
|------|--------------------------|---------|---------------|---------|
| Grid / 任务映射 | `T.Kernel(b_num, is_npu=True) as (cid, vid)`；`row = cid*VEC_NUM + vid`；`if row < B` | `b_num = T.ceildiv(B, VEC_NUM)`，`VEC_NUM = 2` | `examples/topk_selector/example_topk_selector.py:32-33`（B=1024 → b_num=512 超订，实测可跑）；`examples/cross_entropy_loss/example_cross_entro.py:60-62`（`if bn < n_2_num` 行守卫）；`examples/xllm_kernels/fused_gdn_gating.py:152-153` | `T.Kernel(…, threads=2) as (cid)` —— **不可用**，硬约束理由见 §6.5 |
| UB 分配 | `T.alloc_shared(shape, "float32")` | rank-2 `[1, BV]` 用于 copy+reduce；rank-1 `[BV]` 用于 topk | `fused_gdn_gating.py:169-178`（纯 Vector + `alloc_shared`）；`examples/developer_mode/sparse_flash_attn_developer_vid_reduce.py:90-112`；scope 解析 `src/transform/ascend_infer_buffer_scope.cc:398-402` | `T.alloc_ub`（Expert 分配，本设计不用；仅作 §10.1 R-B 兜底） |
| 标量变量 | `T.alloc_var("float32", init=…)` / `T.alloc_var("int32", init=…)` | 返回 1 元素 Buffer，用 `v[0]` 访问 | `tilelang/language/allocate.py:71-125`；`examples/xattention/xattention.py:205-210`；`testing/python/language/test_tilelang_ascend_language_alloc_var.py` | `T.alloc_shared([1], …)` |
| 分块搬入（含尾块） | `T.copy(logits[row, b*BV], z_ub, pad_value=-T.infinity("float32"))` | GM 2D 标量起点 → UB | `example_topk_selector.py:43`（同形态 2D 标量起点 → 1D UB）；`example_cross_entro.py:82`、`example_online_softmax.py:72`（`pad_value=-T.infinity`）；尾块 clamp 机制 `testing/python/language/test_tilelang_ascend_language_tail_block.py:18-31`（`src/op/ascend.cc::compute_valid_extent`） | 显式 BufferRegion 切片 |
| UB→UB 拷贝 | `T.copy(z_ub[0, :BV], zt_ub)` | rank-2 → rank-1 | `fused_gdn_gating.py:188-196`（`T.serial` 内 UB→UB `T.copy`） | `T.Parallel` 逐元素赋值 |
| 块内行最大值 | **`T.reduce_max(z_ub, m_blk, dim=-1)`**（满块全量归约） | ⚠️ **V01.1（V-2 实测）**：`real_shape`/BufferRegion narrow-reduce 逻辑宽度 ≤ **256 B = 64 个 fp32**（`codegen_ascend.cc:2145`），尾块 `valid=512` 不可用 ⟹ 尾块改**精确尺寸 `[1,TAIL]` 缓冲 + 全量 reduce**（§10.2 V-2、§5.3 N1） | `tilelang/language/reduce_ascend.py:362-404`；`example_online_softmax.py:75`；`sparse_flash_attn_developer_vid_reduce.py:155`；narrow 约束 `src/target/codegen_ascend.cc:2145`（`debug_log.md` V-2） | ~~`T.tile.wholereducemax`~~ **已 `@deprecated()`（wheel `ascend_tile.py:1528`），禁用** |
| 块内求和 | **`T.reduce_sum(e_ub, s_blk, dim=-1)`**（满块全量归约） | fp32 行归约；尾块约束与解法同上（精确尺寸 `[1,TAIL]` 缓冲 + 全量 reduce，§10.2 V-2） | `reduce_ascend.py:452-500`；`example_online_softmax.py:83`；`sparse_flash_attn_developer_vid_reduce.py:166`；narrow 约束 `codegen_ascend.cc:2145` | ~~`T.tile.wholereducesum`~~ **已 `@deprecated()`（wheel `:1562`），禁用** |
| running max | 标量 `T.max(m_run[0], m_blk[0])` | — | `sparse_flash_attn_developer_vid_reduce.py:158`（`T.max` 于 `T.Parallel` 内） | `T.tile.max(dst, s0, s1)`（`ascend_tile.py`，非 deprecated）；`example_online_softmax.py:76` |
| 指数 | `T.exp(x)`（`T.Parallel` 内与标量处均可） | — | `sparse_flash_attn_developer_vid_reduce.py:160,164`；`example_online_softmax.py:78`（`T.tile.exp`） | `T.tile.exp(dst, src)`（`ascend_tile.py`，非 deprecated） |
| 自然对数 | 标量 `T.log(x)`；向量 `T.tile.ln(dst, src)` | 定理 T2 的 `ln(mu)` | wheel 已确认 `T.log` 可导出；`T.tile.ln` 用例 `example_cross_entro.py:99`、`fused_gdn_gating.py:233` | — |
| 逐元素 exp/sub（步骤 4） | `for j in T.Parallel(BV): e_ub[0,j] = T.exp(z_ub[0,j] - m_new)` | 1×BV | `sparse_flash_attn_developer_vid_reduce.py:162-164`（`-` 与 `T.exp` 于 `T.Parallel`） | `T.tile.sub` + `T.tile.exp`（`example_online_softmax.py:81-82`） |
| **块内 top-K** | **`T.tile.topk(l_dst, zt_ub, K_MAX, valid)`** | `dst` ≥ `aligned_topk = ceil(2K/8)*8 = 512`（fp32，`elems_per_block = 32/4 = 8`）；`src` **必须编译期静态 shape**；`actual_num` 可符号 | wheel `ascend_tile.py:550-604`；用例 `example_topk_selector.py:48`、`examples/tile_kernels/moe/moe_topk_gate.py:66,91,101,118,128,133`；测试 `testing/python/language/test_tilelang_ascend_language_topk.py`（`test_topk_basic` / `test_topk_2d` / `test_topk_large_actual_num` / `test_topk_src_inplace`） | **无高层等价物** —— 见 §2.3 |
| **取 (val,idx) 的值分量** | **`T.tile.gather_mask(l_val, l_dst, "P0101")`** | `"P0101"` = **取偶数位**（topk 输出 `[v0,i0,v1,i1,…]` ⟹ 偶数位即值） | wheel `ascend_tile.py:608-637` docstring 明列 `"P0101": Extract elements at even indices`；交叉验证 `test_..._topk.py:169-170`（`out_values = b[0::2]`、`out_indices = b[1::2]`）；反证 `example_topk_selector.py:49` 用 `"P1010"` 取 idx | 标量循环 `dst[i] = src[2*i]` |
| 归并 concat | `T.copy(run_v, merged[:K_MAX])` + `T.copy(l_val, merged[K_MAX:])` | UB→UB | `fused_gdn_gating.py:188-196` | `T.Parallel` 赋值 |
| 初始化 run_v | `T.tile.fill(run_v, -T.infinity("float32"))` | 首轮归并前必须置 `-inf`（N4） | `ascend_tile.py` `fill`；用例 `example_online_softmax.py:65-66`、`example_cross_entro.py:78-79`、`sparse_flash_attn_developer_vid_reduce.py:124-126` | `T.Parallel` 赋值 |
| `tau_k = run_v[k_i-1]` | **UB 动态下标标量读**：`tau_k[0] = run_v[k_i[0] - 1]` | `k_i` 来自 GM，运行时值 | **`example_cross_entro.py:113-115`**：`for n_idx in T.serial(...): if y_ub[n_idx] >= 0 and ...: l_n_32[n_idx] = -x_32[n_idx, y_ub[n_idx]]` —— `T.serial` 内用 UB 运行时值做下标，生产级用例 | 向量化 select-sum：`sum_l run_v[l] * 1[l == k_i-1]`（`T.Parallel` + `T.reduce_sum`，零动态下标） |
| top-p 严格前缀和 | `for l in T.serial(K_MAX)` + `with T.If(prefix[0] <= p_i[0])`（**单调守卫替代 break**，§6.4） | 256 次串行 | 动态下标同上；`T.If/T.Then` 用例 `fused_gdn_gating.py:200-224` | `T.cumsum` —— **Ascend 不可用**，见 §3.5.1 |
| tie 统计 | `for l in T.Parallel(K_MAX): tie_m[l] = T.if_then_else(run_v[l] > tau_k[0], 1.0, 0.0)` → `T.reduce_sum(tie_m, cnt, dim=-1)` | 256 元素，代价可忽略 | `T.if_then_else` 用例 `examples/pipeline/sparse_flash_attn_gqa_pipeline.py:121`、`fused_gdn_gating.py:157-166`；reduce 同上 | `T.tile.compare` + `T.tile.select`（`fused_gdn_gating.py:234-253`） |
| 合成阈值 / 守卫 | 标量 `T.max(a, b)`、`T.if_then_else(cond, a, b)`、比较运算 | — | 同上 | `T.tile.max` / `T.tile.compare` + `T.tile.select` |
| K2 屏蔽 + 概率 | `for j in T.Parallel(BV): w_ub[0,j] = T.if_then_else(z_ub[0,j] >= t_v[0], T.exp(z_ub[0,j] - m_v[0]) / s_v[0], 0.0)` | 单次遍历完成比较+选择+exp+div | `sparse_flash_attn_developer_vid_reduce.py:149-183`（`T.Parallel` elementwise on `alloc_shared`，含 `+ * - / T.exp T.max`） | `T.tile.compare` + `T.tile.select` + `T.tile.div`（多一次掩码物化，草稿方案） |
| 搬出（含尾块） | `T.copy(w_ub, w_out[row, b*BV])` | ub2gm **自动 re-clamp**，只写有效 extent | `test_..._tail_block.py:30,41-42`（"ub2gm re-clamps the store"）；`test_..._copy_runtime_extent.py:90` | 显式 `T.copy(w_ub[0,:valid], w_out[row, off:off+valid])` |
| 循环 | `T.serial(n)` 分块/标量循环；`T.Parallel(n)` 元素级；`range(n)` 仅编译期常量 | — | `example_online_softmax.py:68,91`；`fused_gdn_gating.py:188`；`example_topk_selector.py:42,44` | `T.Pipelined(stage=2)`（§6.3，V01-a 不启用） |

> **V01.1 实现层选型注记（Stage 2 NPU 验证后，详见 `debug_log.md`）**：上表「主选/替代」二选一处，
> 实际实现一律选择了 **`T.tile.*` 向量形态**（`max/exp/ln/sub/div/broadcast/compare/select/fill`）与
> **`[1]`/`[1,1]` `alloc_shared` 标量 buffer**，未使用 `T.alloc_var` / 标量 `T.exp`/`T.log` /
> `T.Parallel` 内 `T.exp` —— 原因：`alloc_var` 重绑定能编译但**结果错误**、标量 `T.exp/T.log` 在纯
> Vector kernel **无法 codegen**、**1 元素 UB→UB `T.copy` 触发 aicore 异常 507015**（实现层发现 1/5/6）。
> `T.tile.*` 形态引入少量 broadcast 临时量与掩码 buffer（如 K2 的 `m2d/s2d/t2d`、`mask[1,BV/8]`），
> 均在 UB 预算内（编译 + 运行验证通过）。另：reduce 行的 `real_shape` narrow 路径对尾块 `valid=512`
> 不可用（`codegen_ascend.cc:2145` ≤256 B = 64 fp32），尾块改精确尺寸 `[1,TAIL]` 缓冲 + 全量 reduce
> （§10.2 V-2）。以上均为表内已列替代项的实现层选择，**API 选型、内存层级、算法与 tiling 不变**。

**明确禁用 / 排除的 API（均有源码证据）**

| API | 状态 | 证据 |
|-----|------|------|
| `T.tile.wholereducemax` / `wholereducemin` / `wholereducesum` | 🔴 **wheel 中 `@deprecated()`** | `~/.local/lib/python3.12/site-packages/tilelang/language/ascend_tile.py:1528 / 1545 / 1562`。草稿 §5.3.3 的核心归约原语 ⟹ **本设计全部改用 `T.reduce_max` / `T.reduce_sum`** |
| `T.tile.block_reduce_max` / `block_reduce_min` | 🔴 **`@deprecated()`** | 同上 `:1711 / :1753` |
| `T.cumsum` | 🔴 **Ascend 不可用** | `src/op/reduce.cc:320-341`：`local.fragment` → `LOG(FATAL) << "CumSum for fragment not implemented"`；`shared`/`shared.dyn` → 生成 `tl::CumSum2D<threads,…>::run` 的 `call_extern`，其中 `threads = T.thread_bounds->extent` 是 **GPU thread block 语义**；而 `tl::CumSum2D` **只定义于 `src/tl_templates/cuda/reduce.h:70`，无 Ascend 模板** ⟹ 必然编译失败；其他 scope → `ICHECK(false)` |
| `T.tile.sort` / `T.tile.merge_sort` | 🔴 本设计不用 | `sort` 需 `dst >= 2*aligned_count`，V 规模 UB 不足；跨块归并已由 `topk` 承担 |
| `T.tile.atomic_add` | 🔴 本设计不用 | docstring 明示 destination **must be GM**，对非 global scope 抛 `ValueError`；本设计「一行一任务」无跨核归约，不需要 |
| `T.alloc_ub` / `T.Scope("V")` / `T.barrier_all` / `T.set_flag` / `T.wait_flag` / `T.annotate_address` | 🔴 Expert 标志，Developer 模式不用 | `tilelang-programming-model-guide` §1 / §3.1 / §3.2 |
| `float64` / `double` | 🔴 零支持 | 草稿 §5.3.4 已核（`tilelang/language/*.py` 零命中）⟹ 确定性 seed 路径排除（本 V01-a 亦不涉及） |
| `tilelang/language/pto.py` | 🔴 已废弃 | AGENTS.md 核心原则 1 |

### 3.3 计算伪代码（Developer 模式）

```python
import tilelang
from tilelang import language as T

VEC_NUM   = 2                 # 每 Cube 核配 2 个 Vector 核（fused_gdn_gating.py:26）
TOP_K_ALL = 1 << 30           # ⚠️ sglang/srt/sampling/sampling_params.py:40（不是 -1，草稿有误）

R4_PASS_CONFIGS = {           # Developer / 纯 Vector（programming-model-guide §2.2「纯 Vector 算子」行）
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


@tilelang.jit(out_idx=[4, 5, 6, 7], pass_configs=R4_PASS_CONFIGS)
def r4_topk_stats(B: int, V: int, K_MAX: int, BV: int, NEED_MIN_P: bool):
    n_blk_full   = V // BV                          # 静态：V=152064, BV=4096 → 37
    TAIL         = V - n_blk_full * BV              # 静态：512
    b_num        = T.ceildiv(B, VEC_NUM)            # 一行一任务（B=1024 → 512 block，超订可跑）
    ALIGNED_TOPK = ((2 * K_MAX + 7) // 8) * 8       # fp32 elems_per_block=8 → K_MAX=256 → 512
    # host 侧前置断言（C3/C7）：V % BV == 0 or V % BV >= K_MAX ；V >= K_MAX

    @T.prim_func
    def main(
        logits : T.Tensor([B, V], "float32"),   # in
        top_ks : T.Tensor([B],    "int32"),     # in
        top_ps : T.Tensor([B],    "float32"),   # in
        min_ps : T.Tensor([B],    "float32"),   # in
        m_out  : T.Tensor([B],    "float32"),   # out
        s_out  : T.Tensor([B],    "float32"),   # out
        tau_out: T.Tensor([B],    "float32"),   # out
        valid  : T.Tensor([B],    "int32"),     # out
    ):
        with T.Kernel(b_num, is_npu=True) as (cid, vid):
            row = cid * VEC_NUM + vid
            if row < B:                                     # 行守卫（cross_entropy:62 同款）
                # ---- Developer 模式：全部 alloc_shared（纯 Vector ⟹ 解析为 shared.ub）----
                z_ub   = T.alloc_shared([1, BV],        "float32")   # copy + reduce 源（rank-2，N1）
                zt_ub  = T.alloc_shared([BV],           "float32")   # topk 源（rank-1，N2）
                e_ub   = T.alloc_shared([1, BV],        "float32")   # exp(z-m)
                l_dst  = T.alloc_shared([ALIGNED_TOPK], "float32")   # 块内 topk 输出
                l_val  = T.alloc_shared([K_MAX],        "float32")
                merged = T.alloc_shared([2 * K_MAX],    "float32")
                m_dst  = T.alloc_shared([ALIGNED_TOPK], "float32")   # 归并 topk 输出
                run_v  = T.alloc_shared([K_MAX],        "float32")   # running top-K（降序）
                tie_m  = T.alloc_shared([K_MAX],        "float32")   # tie 0/1 掩码
                m_blk  = T.alloc_shared([1], "float32")
                s_blk  = T.alloc_shared([1], "float32")
                cnt    = T.alloc_shared([1], "float32")

                m_run  = T.alloc_var("float32", init=-T.infinity("float32"))
                s_run  = T.alloc_var("float32", init=0.0)
                k_i    = T.alloc_var("int32")
                p_i    = T.alloc_var("float32")
                mu_i   = T.alloc_var("float32")
                tau_k  = T.alloc_var("float32", init=-T.infinity("float32"))
                tau_p  = T.alloc_var("float32", init=-T.infinity("float32"))
                tau_u  = T.alloc_var("float32", init=-T.infinity("float32"))
                tau    = T.alloc_var("float32", init=-T.infinity("float32"))
                prefix = T.alloc_var("float32", init=0.0)
                vflag  = T.alloc_var("int32",   init=1)
                n_gt   = T.alloc_var("float32", init=0.0)
                n_eq   = T.alloc_var("float32", init=0.0)

                # ---- 逐行标量参数（GM → UB var，各 1 元素）----
                k_i[0]  = top_ks[row]
                p_i[0]  = top_ps[row]
                mu_i[0] = min_ps[row]

                T.tile.fill(run_v, -T.infinity("float32"))    # N4：首轮归并前必须 -inf

                # ======== Pass 1 主循环：online softmax + 分层 top-k 归并 ========
                def block_body(b_off, valid):
                    """valid 为编译期静态 Python int（BV 或 TAIL）
                       ⟹ real_shape / actual_num / T.Parallel 边界全静态（C8）。"""
                    T.copy(logits[row, b_off], z_ub, pad_value=-T.infinity("float32"))
                    # (a) online softmax
                    T.reduce_max(z_ub, m_blk, dim=-1, real_shape=[1, valid])
                    m_new = T.max(m_run[0], m_blk[0])
                    s_run[0] = s_run[0] * T.exp(m_run[0] - m_new)   # 重标定
                    m_run[0] = m_new
                    for j in T.Parallel(BV):
                        e_ub[0, j] = T.exp(z_ub[0, j] - m_new)
                    T.reduce_sum(e_ub, s_blk, dim=-1, real_shape=[1, valid])
                    s_run[0] = s_run[0] + s_blk[0]
                    # (b) 块内 top-K_MAX + 跨块归并
                    T.copy(z_ub[0, :BV], zt_ub)                    # rank-2 → rank-1（N2）
                    T.tile.topk(l_dst, zt_ub, K_MAX, valid)
                    T.tile.gather_mask(l_val, l_dst, "P0101")      # 偶数位 = 值
                    T.copy(run_v, merged[:K_MAX])
                    T.copy(l_val, merged[K_MAX:])
                    T.tile.topk(m_dst, merged, K_MAX, 2 * K_MAX)
                    T.tile.gather_mask(run_v, m_dst, "P0101")

                for b in T.serial(n_blk_full):                     # 满块，静态 37 次
                    block_body(b * BV, BV)
                if TAIL > 0:                                       # 编译期条件，静态尾块
                    block_body(n_blk_full * BV, TAIL)

                # ======== 收尾：三阈值合成（定理 T1/T2）+ tie 检测 + 守卫 ========
                # tau_k：k_i 越界（TOP_K_ALL 或 > K_MAX）→ 取 -inf，由守卫拦截
                k_ok    = T.if_then_else(k_i[0] >= 1 and k_i[0] <= K_MAX, 1, 0)
                tau_k[0] = T.if_then_else(k_ok == 1,
                                          run_v[k_i[0] - 1],        # UB 动态下标（cross_entropy:115 同款）
                                          -T.infinity("float32"))
                # tau_u（定理 T2）：mu<=0 或 NEED_MIN_P=False → -inf；clamp 避免 log(0)
                if NEED_MIN_P:
                    tau_u[0] = T.if_then_else(mu_i[0] > 0.0,
                                              m_run[0] + T.log(T.max(mu_i[0], 1e-30)),
                                              -T.infinity("float32"))
                # tau_p：严格前缀和（不含自身）扫描降序 run_v。
                #        prefix 单调不减 ⟹ 一旦 prefix > p_i 后续守卫恒假，无需 break（§6.4）
                for l in T.serial(K_MAX):
                    with T.If(prefix[0] <= p_i[0]):
                        tau_p[0]  = run_v[l]
                        prefix[0] = prefix[0] + T.exp(run_v[l] - m_run[0]) / s_run[0]
                # D-GUARD：循环自然终止（prefix 越过 p_i）才算 tau_p 可信；p>=1.0 时不启用 nucleus
                ok_p = T.if_then_else(p_i[0] >= 1.0, 1,
                                      T.if_then_else(prefix[0] > p_i[0], 1, 0))
                # D-TIE：仅 top-k 绑定时需要；在 256 元素列表上向量化统计
                for l in T.Parallel(K_MAX):
                    tie_m[l] = T.if_then_else(run_v[l] > tau_k[0], 1.0, 0.0)
                T.reduce_sum(tie_m, cnt, dim=-1)
                n_gt[0] = cnt[0]
                for l in T.Parallel(K_MAX):
                    tie_m[l] = T.if_then_else(run_v[l] == tau_k[0], 1.0, 0.0)
                T.reduce_sum(tie_m, cnt, dim=-1)
                n_eq[0] = cnt[0]

                tau[0] = T.max(T.max(tau_k[0], tau_p[0]), tau_u[0])        # 定理 T1
                # tie_ok 分级判据（V01.1，与 §3.4 D-TIE 一致；NPU 验证）：
                tie_ok = T.if_then_else(tau[0] > tau_k[0], 1,      # ① top-p/min-p 严格绑定 → 值语义精确等价
                                        T.if_then_else(n_gt[0] + n_eq[0] > k_i[0], 0,   # ② 超额平票 → 降级
                                        T.if_then_else(n_gt[0] + n_eq[0] < K_MAX, 1,    # ③ 未占满 ⟹ sum==k_i 精确
                                        T.if_then_else(n_eq[0] <= 1.0, 1, 0))))         # ④ 占满：无平票观察 → 接受
                vflag[0] = T.if_then_else(k_ok == 1,
                                          T.if_then_else(ok_p == 1 and tie_ok == 1, 1, 0), 0)

                m_out[row]   = m_run[0]
                s_out[row]   = s_run[0]
                tau_out[row] = tau[0]
                valid[row]   = vflag[0]
    return main


@tilelang.jit(out_idx=[5], pass_configs=R4_PASS_CONFIGS)
def r4_mask_sample(B: int, V: int, BV: int):
    n_blk = T.ceildiv(V, BV)                       # 静态 38
    b_num = T.ceildiv(B, VEC_NUM)

    @T.prim_func
    def main(
        logits: T.Tensor([B, V], "float32"),   # in
        m_in  : T.Tensor([B],    "float32"),   # in
        s_in  : T.Tensor([B],    "float32"),   # in
        tau_in: T.Tensor([B],    "float32"),   # in
        w_out : T.Tensor([B, V], "float32"),   # out
    ):
        with T.Kernel(b_num, is_npu=True) as (cid, vid):
            row = cid * VEC_NUM + vid
            if row < B:
                z_ub = T.alloc_shared([1, BV], "float32")
                w_ub = T.alloc_shared([1, BV], "float32")
                m_v  = T.alloc_var("float32")
                s_v  = T.alloc_var("float32")
                t_v  = T.alloc_var("float32")
                m_v[0]  = m_in[row]
                s_v[0]  = s_in[row]
                t_v[0]  = tau_in[row]
                for b in T.serial(n_blk):
                    T.copy(logits[row, b * BV], z_ub, pad_value=-T.infinity("float32"))
                    for j in T.Parallel(BV):
                        w_ub[0, j] = T.if_then_else(
                            z_ub[0, j] >= t_v[0],
                            T.exp(z_ub[0, j] - m_v[0]) / s_v[0],
                            0.0)
                    T.copy(w_ub, w_out[row, b * BV])   # ub2gm 自动 re-clamp，尾块只写有效列（C2）
    return main
```

> **伪代码说明**
> 1. `block_body` 是 JIT 工厂函数内的普通 Python 闭包（TileLang 前端在 trace 期展开）；
>    `valid` 为**编译期静态** Python int（`BV` 或 `TAIL`）⟹ 不触发「动态循环边界」限制。
>    若 Stage 2 更倾向宏形式，可改写为 `@T.macro`（`examples/xllm_kernels/utils.py:43-100` 有用例）。
> 2. 标量 GM 读写作 `var[0] = tensor[row]` / `tensor[row] = var[0]`（单元素 GM 标量访问，
>    每行仅 7 次，§5.5 已核算占比 6.6e-5）。等价写法 `T.copy(tensor[row:row+1], var)`，Stage 2 择一。
> 3. `m_new` / `k_ok` 等中间量在伪代码中写作 Python 局部变量，实现时应落到 `T.alloc_var`
>    或直接内联到表达式中（避免 TIR trace 期的 Python 值/符号值混淆）。
>    **V01.1 修正**：Stage 2 实测 `T.alloc_var` 重绑定编译通过但**结果错误**（返回 0），标量
>    `T.exp/T.log` 于纯 Vector kernel 无法 codegen ⟹ 全部标量实际落 **`[1]`/`[1,1]` `alloc_shared`
>    buffer + `T.tile.*` 运算 + `T.copy(region)` GM 标量读写**；且**标量 buffer 间禁用 `T.copy`**
>    （4 B < 32 B DMA 对齐 → aicore 异常 507015），running-max 用 `T.tile.max` 就地更新
>    （`debug_log.md` 发现 1/5/6）。
> 4. **V01.1（Stage 2 实现修正，NPU 验证）**：① `block_body` 中 `real_shape=[1, valid]` narrow-reduce
>    受 `codegen_ascend.cc:2145` **逻辑宽度 ≤ 256 B（= 64 个 fp32）**硬约束，尾块 `valid=512`
>    **编译失败**（`InternalError: Check failed: (nn * elem_bytes <= 256)`；BufferRegion 列切片兜底
>    同为 narrow 路径亦失败）⟹ 实际实现：满块全量 reduce（不传 `real_shape`），尾块内联独立的
>    **精确尺寸 `[1,TAIL]` 缓冲 + 全量 reduce**（§10.2 V-2、§5.3 N1）。② `@T.prim_func` 内嵌套
>    `def block_body()` 闭包被前端拒绝（"Function must be decorated"）⟹ 满块/尾块循环体**内联**
>    （note 1 的 `@T.macro` 备选亦未使用）。③ 满块 `[1,BV]` buffer 的分配必须置于**编译期
>    `if n_blk_full > 0:` 守卫**内：`V < BV` 时 `T.serial(0)` 消除循环体，无条件分配的 buffer
>    成为死 buffer，报 "Cannot find pre-allocated address for buffer"（`codegen_ascend.cc:828`，
>    `l1_edge_V_eq_KMAX` V=256 实测；`debug_log.md` L0 首轮失败信号 3）。伪代码的算法语义在实际
>    实现中均未改变。

### 3.4 关键设计决策（本设计相对草稿的实质改进）

#### D-TIE —— tie（平票）处理：由「索引序丢弃」改为「向量化检测 + 守卫降级」

**草稿 §5.4.3 的方案不可实现**：它要求对 `z_j == tau` 的元素按 index 升序保留前 `n_eq_need` 个，
需要**块内前缀计数**（`rank_eq(j)`），即对等值掩码做 cumsum —— 而 `T.cumsum` 在 Ascend 不可用
（§3.5.1）。改用逐元素标量循环则每块 4096 次 × 38 块 ≈ **15.6 万次标量操作/行**，性能不可接受。

**本设计的替代方案（全向量化，O(K_MAX) 而非 O(V)）**：

先厘清语义边界（已读 `sampler.py:846-879` 核对）：

| 绑定约束 | 参考实现的保留判据 | 本设计 `z >= tau` 是否等价 |
|---------|-------------------|--------------------------|
| **top-p** | `probs_sum - probs_sort > top_ps` → 屏蔽（**严格 `>`**，边界元素保留） | ✅ **精确等价**（值阈值语义，平票自然全保留） |
| **min-p** | `probs_sort < threshold` → 屏蔽（**严格 `<`**，相等保留） | ✅ **精确等价** |
| **top-k** | `arange(V) >= top_ks` → 屏蔽（**按秩**，恰好保留 `k` 个） | ⚠️ 有平票时保留集偏大 |

⟹ **只有 top-k 绑定（`tau == tau_k`）时才存在 tie 风险**，且只需在长度 `K_MAX=256` 的归并列表上判定：

```
n_gt = #{l < K_MAX : run_v[l] >  tau_k}        # T.Parallel + T.reduce_sum，256 元素
n_eq = #{l < K_MAX : run_v[l] == tau_k}
因 tau_k = run_v[k_i - 1]，恒有 n_gt <= k_i - 1 且 n_gt + n_eq >= k_i

# V01.1 分级判据（与 r4_fused_sampling.py 实现一致，NPU 验证；短路求值 ①→②→③→④）
tie_ok = (tau > tau_k)                          # ① top-p/min-p 严格绑定 → 与参考精确等价
         OR ( n_gt + n_eq <= k_i                # ② 超额平票（阈值保留数 > k_i）→ 降级 0
              AND ( n_gt + n_eq < K_MAX         # ③ 列表未被平票占满 ⟹ 已捕获全部平票，
                    OR n_eq <= 1 ) )            #    结合恒等式 sum >= k_i ⟹ sum == k_i 精确 → 接受
                                                # ④ 列表占满（sum == K_MAX，必有 k_i == K_MAX）：
                                                #    列表内未在 tau_k 观察到平票（n_eq == 1，仅自身）
                                                #    → 接受；被平票占满（n_eq > 1）→ 真实平票数未知 → 降级
```

`tie_ok == 0` 时置 `valid = 0` → host 整批回退 torch 路径（**correct-by-fallback**）。

> **⚠️ V01.1 修正（Stage 2 NPU 验证后）**：V01-a 原公式 `(n_gt+n_eq < K_MAX AND n_gt+n_eq <= k_i)`
> 在 `k_i == K_MAX` 时**恒假**（此时恒有 `n_gt + n_eq ≡ K_MAX`），与下方论证 2 及 §9.2
> 「`valid_ref` 全 1」**自相矛盾** —— `l0_k_eq_kmax`（k=256=K_MAX、连续 `randn` 数据）实测
> `valid ratio = 0.0000`（`debug_log.md` L0 首轮失败信号 2）。已改为上述分级判据 ①~④：
> **保留原安全意图**（平票占满的行仍降级；全等行/全零行仍正确 `valid=0`，Boundary
> `bnd_all_equal_valid` / `bnd_all_zero_valid` 实测通过），同时满足连续分布（无精确有限值平票，
> `n_eq == 1`）下含 `k_i == K_MAX` 的快路径 `valid=1`（`l0_k_eq_kmax` 通过）。
>
> **残留已知缺口（已知限制，非阻塞）**：`n_gt/n_eq` 只在归并 top-`K_MAX` 列表上统计，若**恰好
> 2 个相等值横跨 rank `K_MAX-1`/`K_MAX` 边界**（`z_sort[K_MAX-1] == z_sort[K_MAX] == tau_k` 且
> `k_i == K_MAX`），列表中第 `K_MAX` 个等值对 kernel 不可见（检测需 `z_sort[K_MAX]`），判据 ④
> 可能误接受。该缺口仅在「退化数据（精确平票）+ `k_i == K_MAX`」同时成立时才可能触发；
> 连续分布的 L0/L1 与 Boundary 退化套件（全等/全零/`-inf`）均不触发。已记入
> `r4_fused_sampling.py` 代码注释与 `debug_log.md`。

**两条重要的无害化论证**：

1. **`tau == -inf` 时的平票无害**：被多保留的元素满足 `z_j = -inf` ⟹ `w_j = exp(-inf - m)/s = 0`
   ⟹ 对 `torch.multinomial` 的权重贡献为 **0**，分布不变。故 `-inf` 平票（草稿 B10 指出的主要平票源，
   来自上游屏蔽）**无需处理**。这也解释了为何 `tie_ok` 只在 `tau == tau_k` 且 `tau_k` 有限时才可能为 0。
2. **L0 用例不触发守卫**：连续 `randn` fp32 logits 出现精确有限值平票的概率可忽略，
   `n_gt + n_eq == k_i` 恒成立：`k_i < K_MAX` 时判据 ③ 给出 `tie_ok = 1`；`k_i == K_MAX` 时
   `n_eq == 1`（仅 `tau_k` 自身）、判据 ④ 给出 `tie_ok = 1` ⟹ 全部 L0 用例 `valid = 1`
   （含 `l0_k_eq_kmax`，NPU 实测通过）。退化输入（全等行/全零行）由 Stage 2 的 Boundary 套件
   覆盖（实测 13/13 PASS，正确 `valid = 0`）。

#### D-GUARD —— 降级守卫判据修正（草稿 §3.7 的 `valid` 逻辑有缺陷）

草稿写 `valid[i] <- (L >= 0) ? 1 : 0`，其中 `L` 是前缀和循环最后一次赋值的下标，初值 `-1`。
但 `prefix` 初值为 `0`、`p_i ∈ (0,1]` ⟹ `prefix <= p_i` 在 `l = 0` **必然成立** ⟹ `L >= 0` 恒真
⟹ **该守卫永不触发**，草稿 D8/B9 描述的「nucleus > K_MAX 时 tau_p 被低估」实际未被拦截。

**修正后的判据**（利用 `prefix` 单调不减）：

```
循环自然终止（找到跨越点） <=> 退出时 prefix > p_i
若退出时 prefix <= p_i ⟹ K_MAX 个候选耗尽仍未达 p ⟹ 真实 nucleus 可能 > K_MAX
                       ⟹ tau_p 被低估 ⟹ valid = 0
特例：p_i >= 1.0 ⟹ nucleus 为全集，tau_p = -inf 本就正确 ⟹ ok_p = 1（不触发守卫）
```

最终：**`valid = (1 <= k_i <= K_MAX) AND ok_p AND tie_ok`**

对应草稿 §7 边界条件的覆盖：B5（`TOP_K_ALL = 1<<30`）、B8（`k_i > K_MAX`）由 `1 <= k_i <= K_MAX`
拦截；B9（nucleus > K_MAX）由 `ok_p` 拦截；平票由 `tie_ok` 拦截。
守卫触发时性能不劣于现状（此类输入本就 `npu_top_k_top_p_eligible = False` → 走 Path 3，
`sglang/.../sampling_batch_info.py:263-265`）。

#### D-NORENORM —— `w_out` 不做重归一化

`w_out[i,j] = exp(z_ij - m_i)/s_i * 1[z_ij >= tau_i]`，其中 `s_i` 是**全 vocab** 的 exp 和，
故 `sum_j w_ij =`（保留集概率质量）`<= 1`，**不等于 1**。这与被替换实现一致：
`sampler.py:836 / 868` 直接把屏蔽后的 `probs` 交 `torch.multinomial`，而
`multinomial_with_seed` docstring（`sampler.py:893-894`）明示
*"The values are treated as weights and **do not need to sum to 1**"* ⟹ multinomial 内部隐式归一化。
**因此 K2 无需第二遍求和归一化**，省一次全 vocab 遍历（草稿 §3.5 Step 6 的观察正确，本设计沿用）。

---

## 3.5 技术约束确认

### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| **不支持三维 Kernel** | **No** | `T.Kernel(b_num, is_npu=True)` 一维 grid；`as (cid, vid)` 是同一 block 内的 Cube/Vector 核编号解构，**非第三维** |
| **threads 参数限制（仅 1 或 2）** | **No** | **不指定 `threads`**（用默认）。硬约束理由见 §6.5 |
| **动态循环边界不支持** | **Yes → 已消解** | 分块循环 `T.serial(V // BV)`、尾块 `if TAIL > 0`、收尾 `T.serial(K_MAX)` 的 trip count 均由**编译期常量** `V`/`BV`/`K_MAX` 导出，全静态。`B` 只出现在 grid 大小 `b_num = T.ceildiv(B, VEC_NUM)` 与行守卫 `if row < B` 中，**不作为循环边界**。`real_shape` / `actual_num` / `T.Parallel(valid)` 亦全静态（C8） |
| **流水线不支持动态边界** | **No（V01-a 不启用流水线）** | 见 §6.3（含 4 条不启用理由与 Stage 3 可选清单） |
| **部分 GPU API 不可用** | **Yes → 已消解** | `T.cumsum` 不可用（`src/op/reduce.cc:320-341`；`tl::CumSum2D` 仅存于 `src/tl_templates/cuda/reduce.h:70`）⟹ top-p 前缀和改为 `K_MAX=256` 上的 UB 标量循环 + 单调守卫 |
| **L0C 容量上限（128 KB）** | **N/A** | 纯 Vector，**零 matmul**，不使用 L0A/L0B/L0C ⟹ 无 L0C 溢出风险 |
| **GEMM 非整除** | **N/A** | 无 GEMM |
| **`T.copy` 列方向 strided 切片** | **No** | 全部为 2D `[B,V]` 的**行内连续**切片 `logits[row, b*BV]`，最内维连续；无 4D+ 切片 |
| **Host 侧 `reshape` 隐式拷贝** | **No** | host 路径无 `reshape`/`permute`/`transpose`；审计见 §5.6 |
| **Host 侧 aclnn 隐式调用** | **部分（用户明确保留）** | `torch.multinomial(w_out, 1)` 为 V01-a 用户明确保留的**下游独立算子**，非 R4 计算的 host 替代；`valid` 的 D2H 校验属分派逻辑。完整审计 + 偏差记录 DEV-1 见 §5.6 / §10.3 |
| **输出侧切片 + reshape 隐式 contiguous** | **No** | `w_out` 由 kernel 直接写成 `[B,V]`，host 不切片不 reshape |
| **`T.tile.topk` 对 float32 in-place 写 `-inf` padding** | **Yes → 已消解** | §5.3 注 N3：ascendc 上把 `[actual_num, buffer_size)` 全填 `-inf`（`test_..._topk.py:181-183` docstring 明示），作用对象是 UB scratch `zt_ub`，**不触碰 GM**；且 `BV % 32 == 0` ⟹ `AlignUp(actual_num,32) <= BV`，无越界写 |
| **`T.tile.topk` 的 `max_actual_num = Σ src.shape`（求和非求积）** | **Yes → 已消解** | §5.3 注 N2：专用 rank-1 `zt_ub[BV]` 作 topk 源，使 `max_actual_num = BV` 精确，规避 `1+BV` 导致的 `repeatTimes` 超界读风险 |
| **归约的尾块有效宽度** | **Yes → 已消解（V01.1 修正）** | 原方案 `real_shape=[1, valid]` narrow-reduce；Stage 2 上板实测该路径（含 BufferRegion 列切片兜底）受 `codegen_ascend.cc:2145` **逻辑宽度 ≤ 256 B = 64 个 fp32** 约束，尾块 `valid=512` 不可用 ⟹ **改用精确尺寸 `[1,TAIL]` 缓冲 + 全量 reduce**（§5.3 注 N1、§10.2 V-2，NPU 验证通过）。仍**不依赖 `pad_value` 保证正确性**（`test_..._tail_block.py:45-51` 实测：依赖 `-inf` pad 会在 NPU 上产生 inf/nan） |

### 3.5.2 参考实现差异说明（草稿 Expert 方案 → 本 Developer 方案）

| 差异项 | 草稿 `DESIGN_script.md`（Expert） | 本设计（Developer） | 转换依据 |
|--------|----------------------------------|--------------------|---------|
| 内存分配 | `T.alloc_ub(…)` | **`T.alloc_shared(…)`** | guide §3.2 转换表；`InferAllocScope` 对纯 Vector 解析为 `shared.ub`（`ascend_infer_buffer_scope.cc:398-402`） |
| 作用域 | `T.Scope("V")`（隐含） | **不写 Scope** | guide §3.1 步骤 3；`AUTO_CV_COMBINE` 自动分离 |
| 同步 | 手动 `barrier` / `set_flag` / `wait_flag` | **全自动，零手写同步** | guide §3.1 步骤 4；`AUTO_SYNC=True` |
| Kernel 轴 | `T.Kernel(b_num, is_npu=True) as (cid, vid)` | **同**（保留；`threads=2` 不可用，§6.5） | 纯 Vector 生产级实现一致形态（`fused_gdn_gating.py:152`） |
| 行归约 max | `T.tile.wholereducemax(dst, src, mask, repeattimes, dstrepstride, srcblkstride, srcrepstride, ReduceOrder=…)`（7 参数低层签名，草稿 R11 标为「调参成本」风险） | **`T.reduce_max(z_ub, m_blk, dim=-1)`**（高层 API；满块全量归约，尾块精确尺寸 `[1,TAIL]` 缓冲 —— V01.1，`real_shape` narrow 路径对 valid>64 fp32 不可用，§10.2 V-2） | 草稿 API **已 `@deprecated()`**（wheel `ascend_tile.py:1528`）；高层 API 由 `example_online_softmax.py:75` + `sparse_flash_attn_developer_vid_reduce.py:155` 验证。**顺带消解草稿 R10/R11 两项风险** |
| 行归约 sum | `T.tile.wholereducesum(…)`（7 参数） | **`T.reduce_sum(e_ub, s_blk, dim=-1)`**（同上，V01.1） | 同上（wheel `:1562` deprecated） |
| elementwise exp/sub/div | `T.tile.exp` / `T.tile.sub` / `T.tile.div` 链 | **`T.Parallel` + `T.exp` / 运算符**（保留 `T.tile.*` 作替代） | `sparse_flash_attn_developer_vid_reduce.py:149-183` |
| compare / select | `T.tile.compare(dst, s0, s1, "GE")` + `T.tile.select(dst, mask, s0, s1, mode)`（需物化 `keep_ub[8192] uint8`） | **`T.if_then_else` 于 `T.Parallel` 内**（单次遍历完成比较+选择+exp+div，**省掉独立掩码 buffer**） | 同上 |
| top-k | `T.tile.topk` + `T.tile.gather_mask`（Expert UB） | **`T.tile.topk` + `T.tile.gather_mask`（Developer `alloc_shared`）** —— 无高层等价物 | §2.3；两者均**非 deprecated**，且 `PIPE_V` ⟹ `alloc_shared` 落 UB |
| top-p 前缀和 | UB 标量循环（草稿 D2） | **同**（草稿此处结论正确，本设计沿用） | `T.cumsum` 不可用 |
| tie 处理 | `n_gt` / `n_eq_need` / `rank_eq(j)` 索引序丢弃 | **向量化 `n_gt`/`n_eq` 检测 + `valid` 守卫降级**（D-TIE） | 草稿方案需块内 cumsum ⟹ 不可实现 |
| `valid` 判据 | `(L >= 0) ? 1 : 0`（**恒真，守卫失效**） | **`(1<=k_i<=K_MAX) AND ok_p AND tie_ok`**（D-GUARD） | 修正草稿缺陷 |
| 目标芯片 | Ascend910_**9382** | **Ascend910_9362（实测）** | 二者同含 `910_93` ⟹ `determine_platform()` 均判 **A3**（`tilelang/utils/target.py:145-147`），`libgen.py:152-155` 固定 `--npu-arch=dav-2201` ⟹ **arch 一致，结论不变** |
| Vector 核数 | 假设 **48**（Cube 24） | **运行时 `detect_vec_core_num()` 探测 → 实测 40**（Cube 20），`VEC_NUM=2` | `examples/xllm_kernels/utils.py:20-40`；**禁止硬编码 48** |
| `TOP_K_ALL` | `-1` | **`1 << 30` = 1073741824** | `sglang/python/sglang/srt/sampling/sampling_params.py:40`（已读源码） |
| UB 预算（topk 部分） | 「topk 相关 7 KB」（S1' 合计 41 KB） | **自动注入 tmp arena = `4 × AlignUp(BV,32) × 4 B` = 65,536 B（64 KB）**；K1 合计 ≈ **121 KB** | `src/transform/allocate_tmp_buffer.cc:578-596`（wheel 提交 `d75300ed` 同）：fp32 topk `factor=4`，`count = ceil(max_actual_num/32)*32`；`GetTmpBufferSize_`（`:1195-1206`）取 **max** 而非 sum ⟹ 单一共享 arena |
| K2 `keep_ub` | `[8192] uint8` = 8 KB | **消除**（`T.if_then_else` 内联）。⚠️ V01.1：实际实现采用替代项 `T.tile.compare`+`T.tile.select`（标量 `T.exp`/运算符于纯 Vector kernel 无法 codegen，`debug_log.md` 发现 6），掩码 buffer 仅 `[1, BV/8] uint8` = **512 B**，仍远小于草稿 8 KB | `sparse_flash_attn_developer_vid_reduce.py:149-183` |
| 落点 | `sgl-kernel-npu/.../sample/r4/` + `r4_dispatch.py` backend 注册 | **`examples/R4/` 独立算子**（无 dispatch/tiling/lcg 模块） | 用户明确指定 V01-a 范围 |

### 3.5.3 本项目同类实现参考（必须列出）

| 文件路径 | 相似度 | 关键参考点（已逐一读取核实） |
|----------|--------|------------------------------|
| **`examples/cross_entropy_loss/example_cross_entro.py`** | ⭐⭐⭐ **最高**（结构同构） | `:60-62` `T.Kernel(n_num, is_npu=True) as (cid,vid)` + `bn = cid*VEC_NUM+vid` + **`if bn < n_2_num` 行守卫**；`:78-95` 两遍 online softmax（`T.reduce_max`/`T.tile.max`/`T.reduce_sum` 全套）；`:82` `T.copy(…, pad_value=-T.infinity(CAL_DTYPE))`；`:99` `T.tile.ln`；**`:113-115` `T.serial` 内用 UB 运行时值做动态下标 `x_32[n_idx, y_ub[n_idx]]`** —— 正是本设计 `tau_k` / `tau_p` 循环所需 |
| **`examples/topk_selector/example_topk_selector.py`** | ⭐⭐⭐ **最高**（top-k 同构） | `:10-18` pass_configs；`:24` `b_num = T.ceildiv(B, VEC_NUM)`；`:32-33` `row_id = (cid*VEC_NUM+vid) % B`；**`:43` `T.copy(x[row_id, bn*block_N], x_ub)`（GM 2D 标量起点 → 1D UB）**；**`:48` `T.tile.topk(topk_dst, score_accum, top_k, N)`**；**`:49` `T.tile.gather_mask(topk_index, topk_dst, "P1010")`（取奇数位=idx，反证 `"P0101"`=值）**；`:50` `T.tile.cast`；`:57-121` 对拍框架（`ref_program` + `count_per_row_mismatches` + `count_total_mismatches` + `assert accuracy > 0.99`）；**`:127` `N (max ~8192 for 192KB UB)`** |
| **`examples/softmax/example_online_softmax.py`** | ⭐⭐⭐ 高 | `:7-11` 纯 Vector pass_configs；`:35-38` `T.ceildiv` + `VEC_NUM=2` + `sub_block_M`；`:52` Kernel 形态；`:65-66` `T.tile.fill(prev_max, -T.infinity(…))` / `fill(prev_sum, 0.0)` 初始化；**`:68-85` 完整 online softmax 单遍分块循环（本设计 K1(a) 的直接模板）**；`:72` `pad_value=-T.infinity`；`:75/83` `T.reduce_max`/`T.reduce_sum(dim=-1)`；`:108-114` 含 `(34,130,32,32)` 非整除配置且实测通过 ⟹ 尾块 clamp 有效 |
| **`examples/developer_mode/sparse_flash_attn_developer_vid_reduce.py`** | ⭐⭐ **Developer 模式标杆** | `:11-15` 四个 pass_configs 全 True；`:85` `T.Kernel(block_num, threads=2, is_npu=True) as (cid)`；`:90-112` **全 `T.alloc_shared` / `T.alloc_fragment`**；**`:149-183` `T.Parallel` elementwise（`+ * - / T.exp T.max`）直接作用于 `alloc_shared` buffer**；**`:155` `T.reduce_max(acc_s_ub, m_i, dim=-1)`、`:166` `T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)`** ⟹ 证明高层 reduce 在 Developer 模式 + `alloc_shared` 下可用 |
| **`examples/developer_mode/gelu_mul_developer.py`** | ⭐⭐ **Developer 模式标杆** | `:8-11` Developer pass_configs；`:22` `T.Kernel(m_num*n_num, threads=2, is_npu=True) as (cid)`；**`:26-29` `T.alloc_shared` + `:33-46` `T.tile.mul/add/exp/div` 混用** ⟹ 证明 `T.tile.*` 与 Developer 模式共存是仓库既有范式（§2.3 论据 1） |
| **`examples/xllm_kernels/fused_gdn_gating.py`** | ⭐⭐ 生产级形态 | `:26` `VEC_NUM = 2`；`:29` `MAX_VEC_CORE_NUM = detect_vec_core_num()`（**运行时探测，不硬编码**）；`:30` `BATCH_SIZE_SPECIALIZATIONS`；`:37-43` `select_launch_block_num(*, num_batches, vec_core_num) = min(num_batches, vec_core_num)`；`:143-154` `compile_max_batch` + `num_batches: T.int32` 运行时参数范式（R-JIT 的可选缓解）；`:152-153` `T.Kernel(…) as (cid,vid)` + `task_id = cid*VEC_NUM+vid`；**`:169-178` 纯 Vector 全 `T.alloc_shared`**；`:188-196` `T.serial` 内 UB→UB `T.copy`；`:200-224` `T.If/T.Then/T.Else`；`:226-257` `T.tile.{cast,axpy,mul,abs,exp,add,ln,compare,select,sigmoid}` 生产级链 |
| **`examples/xllm_kernels/utils.py`** | ⭐⭐ | **`:3-10` `DEFAULT_ASCEND_PASS_CONFIGS`（原始字符串形式，规避 `PassConfigKey` 导出时机依赖）**；`:12` `DEFAULT_ASCEND_BISHENG_ARCH = "dav-2201"`；**`:20-40` `detect_vec_core_num(default=48)`**（探测 `vector_core_num`/`aiv_core_num`/`vec_core_num`）；`:43-100` `@T.macro` 用法 |
| **`examples/tile_kernels/moe/moe_topk_gate.py`** | ⭐⭐ | `:15-20` pass_configs；`:38` Kernel 形态；**`:66/91/101/118/128/133` 多处 `T.tile.topk` 生产级用法**（ping-pong 双缓冲）；`:45-48` `topk_dst[2*aligned_num_topk]` + `gather_mask` 尺寸约定 |
| `testing/python/language/test_tilelang_ascend_language_topk.py` | ⭐⭐ **API 语义权威** | `:30-33` 四个 pass_configs 全 True；`:37-40` `_aligned_topk(K,dtype) = ceil(2K/elems_per_block)*elems_per_block`（fp32 `elems_per_block=8`）；**`:86-89` + `:149-154` 2D src 语义 = 「flattened (row-major) 前 actual_num 个元素」**；**`:169-170` `out_values = b[0::2]`、`out_indices = b[1::2]`**（⟹ `"P0101"`=值）；**`:181-183` ascendc 上 fp32 src 的 `[actual_num:buffer_size)` 全部被 in-place 填 `-inf`**；`:349` `test_topk_2d(M=2, per_row=16, K=6, actual_num=18)`；`:411` `actual_num ∈ {2049, 4096}` 大 count 覆盖 |
| `testing/python/language/test_tilelang_ascend_language_narrow_reduce.py` | ⭐⭐ | **`:51` `reduce_fn(ub, acc, dim=-1, real_shape=[M, valid])`**；**`:68` `reduce_fn(ub[:, offset:offset+valid], acc, dim=-1)`** —— narrow-reduce 两种写法。⚠️ **V01.1（V-2 实测）**：两种写法同受 `codegen_ascend.cc:2145` 逻辑宽度 ≤256 B（=64 fp32）约束（该测试通过恰因宽度低于限值），本算子尾块 `valid=512` **均不可用** ⟹ 改精确尺寸 `[1,TAIL]` 缓冲 + 全量 reduce（§10.2 V-2） |
| `testing/python/language/test_tilelang_ascend_language_tail_block.py` | ⭐⭐ | `:10-56` **尾块机制权威说明**：`:18` `src/op/ascend.cc::compute_valid_extent`；`:20-31` 所有 GM↔片上 `T.copy` 在 lowering 期 clamp（VECTOR: gm2ub + ub2gm）；`:41-42` **「ub2gm re-clamps the store → pad_value is irrelevant」**；**`:45-51` 「reduce 必须用 `real_shape=[rows, cols]` 告知逻辑有效宽度；依赖 `-inf` pad 会在 NPU 上产生 inf/nan（verified）」** ⟹ 本设计**不依赖 `pad_value` 保证归约正确性**（N1 结论不变）。⚠️ V01.1：`real_shape` 机制本身对 `valid > 64` fp32 不可用（`codegen_ascend.cc:2145` ≤256 B），尾块改**精确尺寸 `[1,TAIL]` 缓冲 + 全量 reduce**（§10.2 V-2，NPU 验证） |
| `testing/python/language/test_tilelang_ascend_language_copy_runtime_extent.py` | ⭐ | `:88-90` gm2ub / ub2gm 双向 runtime inner-extent 切片 `A[..., 0:n] ↔ a_ub[:, 0:n]`（C2 的显式写法备选） |
| `testing/python/language/test_tilelang_ascend_language_scalar_index.py` | ⭐ | `:38-46` UB buffer 的 2D 标量 load/store（`buf[b,i] = table[b,i]*2`） |
| `examples/reduce/example_reduce_min.py` | ⭐ | A2/A3 可用的 reduce 参照形态 |
| ❌ `examples/reduce/example_{col,row}_reduce_max_slice_buffer.py` | — | `target="pto"` → A5 专属，**排除** |
| ❌ `examples/sort/example_merge_sort.py` | — | `N=64` + `T.Kernel(1)` 单核全 UB 驻留，仅原语演示，**V 规模不可用** |

### 3.5.4 源码级证据索引（本设计新增核实，全部在已安装 wheel 对应提交 `d75300ed` 上复核）

| 结论 | 源码位置 |
|------|---------|
| `alloc_shared`(scope `shared`) 对纯 Vector 解析为 `shared.ub` | `src/transform/ascend_infer_buffer_scope.cc:398-402`；`IsVectorFunction` 同文件 `:86-95` |
| `tl.ascend_topk` / `tl.ascend_reduce` / `tl.ascend_gather_mask` 属 `PIPE_V` | `src/transform/common/operation_config.h:213-214, 248-251` |
| fp32 topk 自动 tmp arena = `4 × (ceil(max_actual_num/32)×32) × 4 B` | `src/transform/allocate_tmp_buffer.cc:578-596`（`GetAscendCWorkspaceSpec`）；同处 `ICHECK(is_half‖is_float) << "dav-2201 Sort/TopK only supports float16/float32"` ⟹ **fp32 在 dav-2201 受支持** |
| tmp arena 取 **max** 而非 sum，单一共享 `shared.ub` buffer | `allocate_tmp_buffer.cc:1195-1206`（`GetTmpBufferSize_`：`shape_size = std::max(shape_size, spec.primary_bytes)`）、`:1156-1158`（`PointerType(…, "shared.ub")`） |
| fp32 `reduce_sum`（dim=-1, rows=1, cols=4096）tmp = 16,384 B；`reduce_max` = 256 B | `allocate_tmp_buffer.cc:364-372`（`AscendCReduceUsesTmp`）+ `:400-410`（sum 分支 `power=FloorPowerOfTwo(cols)`）/ `:411-420`（max 分支 `kVectorRepeatBytes=256`） |
| `T.cumsum` Ascend 不可用 | `src/op/reduce.cc:320-341`；`tl::CumSum2D` 仅 `src/tl_templates/cuda/reduce.h:70` |
| `T.tile.topk` 的 `max_actual_num = Σ src.shape`（**求和，非求积**） | wheel `tilelang/language/ascend_tile.py:578-590` |
| `gather_mask` 的 `"P0101"` = 偶数位 | wheel `ascend_tile.py:625-631` docstring |
| `wholereduce*` / `block_reduce_*` 已 deprecated | wheel `ascend_tile.py:1528 / 1545 / 1562 / 1711 / 1753` |
| 显式 `tmp=` 必须 scope `shared.ub`（本设计**不传** `tmp`，交由编译器注入） | `tilelang/language/ascend.py:141-142` |
| `real_shape` 仅对 rank-2 buffer 生效 | `tilelang/language/reduce_ascend.py:99-108`（`_resolve_reduce_logical_shape`）、`:318-328` |
| A3 判定：`"910_93" in soc_name.upper()` | `tilelang/utils/target.py:145-147`（`determine_platform`） |
| ascendc target 固定 `--npu-arch=dav-2201` | `tilelang/jit/adapter/libgen.py:152-155` |
| `TOP_K_ALL = 1 << 30` | `sglang/python/sglang/srt/sampling/sampling_params.py:40` |
| Path 3 参考语义（top-k/top-p/min-p 精确判据） | `sglang/python/sglang/srt/layers/sampler.py:846-879` |
| Path 2 参考语义（`npu_top_k_top_p` + min-p on unsorted probs） | `sampler.py:826-845` |
| multinomial 视输入为权重、无需归一化 | `sampler.py:893-894` docstring |
| `npu_top_k_top_p_eligible = all(1 <= top_k <= 1024)` | `sglang/python/sglang/srt/sampling/sampling_batch_info.py:263-265`；`:574` merge 时 `&=` 收紧 |
| `need_min_p_sampling = any(min_p > 0)` | `sampling_batch_info.py:262` |
| 已安装 wheel 版本与路径 | `tilelang 0.1.4+d75300ed0a88b0f4b2eef8a62b43f1930fdae44b` @ `~/.local/lib/python3.12/site-packages/tilelang`（`reduce_ascend.py` 与仓库源码**仅 docstring 差异**，代码一致） |

### 3.5.5 API 检索顺序合规记录（AGENTS.md 核心原则 1）

按「① `tilelang-api-best-practices` → ② `examples/` 同类实现 → ③ 源码」的强制顺序执行，逐级记录：

| 层级 | 查阅对象 | 命中 / 结论 |
|------|---------|-----------|
| **① 速查表** | `tilelang-api-best-practices/references/api-compute.md` | ✅ **命中** `:93-97` `T.reduce_{sum,max,min}(buffer, out, dim=-1, clear=True, real_shape=None, *, tmp=None)` —— 与本设计签名**完全一致**；`:107` *"`real_shape`：2D slice buffer 的逻辑有效范围"* ⟹ **印证 N1（`z_ub` 必须 rank-2）**；`:121` `clear=False` ⟹ `new_out = max(old_out, reduced_result)`（online softmax 可用的累加语义）；`:147` 用例 `T.reduce_max(in_shared, out_shared, dim=-1, real_shape=[4,4])` ⟹ **印证 `alloc_shared` + `real_shape` 组合是官方推荐写法**；`:134` 非法 `dim`/`real_shape`/输出 shape 在**前端直接报错**而非静默 |
| **① 速查表** | `tilelang-api-best-practices/references/api-kernel-memory.md` | ✅ **命中** `:150` *"Developer 模式下 `T.alloc_shared` 默认使用动态 scope `shared`，由编译器推断为 `shared.l1` 或 `shared.ub`"* ⟹ **正面印证 §2.3 论据 2**（纯 Vector ⟹ 推断为 `shared.ub`）；`:480-482` 纯 Vector 的 `alloc_shared` 用例 |
| **① 速查表** | `tilelang-api-best-practices/references/api-schedule-sync.md` | 同步原语章节 —— 本设计**不使用任何手动同步**（§7.1），仅确认 `T.serial` / `T.Parallel` 语义 |
| **①→③ 升级** | `T.tile.topk` / `T.tile.gather_mask` | ⚠️ **速查表未覆盖** ⟹ 按原则 1 降级到 ② `examples/`（`topk_selector`、`moe_topk_gate`）与 ③ 源码（wheel `ascend_tile.py:550-637`）+ 测试（`test_..._topk.py`），完成 `aligned_topk` 尺寸、`"P0101"`=偶数位、2D src 扁平语义、in-place `-inf` padding、tmp arena 公式五项确认 |
| **①→③ 升级** | `T.cumsum` 可用性 | ⚠️ **速查表未标注 Ascend 不可用** ⟹ 降级到 ③ 源码 `src/op/reduce.cc:320-341` + `src/tl_templates/cuda/reduce.h:70` 才得到「必然编译失败」的确定结论 |
| **①→③ 升级** | `wholereducemax/sum` 是否可用 | ⚠️ 草稿标为「✅ 已验证」，但 ③ 源码（**wheel**）显示 `@deprecated()` ⟹ **以 wheel 源码为准否决草稿结论** |
| **禁用** | `tilelang/language/pto.py` | 🔴 已废弃，全程未引用（AGENTS.md 核心原则 1） |

**结论**：本设计**无凭记忆猜测的 API**；速查表覆盖的 3 项（`reduce_*`、`alloc_shared`、调度原语）
均与速查表一致，速查表未覆盖的 3 项（`topk`、`gather_mask`、`cumsum`）均已下探到 `examples/` +
wheel 源码 + 官方测试三重确认，并在 §3.2「明确禁用/排除的 API」表中留下证据行号。

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `logits` | `(B, V)` | `float32` | temperature **已缩放**的 logits `z_ij`。连续内存；**行内允许含 `-inf`**（上游屏蔽产生）；不含 NaN（上游 `sanitize_nan_logits` 已清洗，`sampler.py:131-139`） |
| `top_ks` | `(B,)` | `int32` | 每行 `k_i`。合法域 `1 <= k <= K_MAX`；`k > K_MAX`（含 `TOP_K_ALL = 1<<30`）⟹ `valid=0` 降级 |
| `top_ps` | `(B,)` | `float32` | 每行 `p_i ∈ (0, 1]`；`1.0` = 不启用 nucleus |
| `min_ps` | `(B,)` | `float32` | 每行 `mu_i ∈ [0, 1]`；`0.0` = 不启用 min-p |

**编译期常量参数**（`@tilelang.jit` 函数的 Python 形参，非张量）：

| 参数 | 类型 | V01-a 取值 | 说明 |
|------|------|-----------|------|
| `B` | int | 运行时给定（1~1024） | 见 §4.6 与 R-JIT |
| `V` | int | **152064**（代表性） | vocab size，**必须编译期静态**（`T.tile.topk` 要求 src 静态 shape；`n_blk`/`TAIL` 亦须静态） |
| `K_MAX` | int | **256** | top-k 归并缓冲上限（S1' 特化） |
| `BV` | int | **4096** | vocab 分块大小（S1' 特化） |
| `NEED_MIN_P` | bool | `True` / `False` | `False` 时 K1 完全跳过 `tau^(mu)` 计算 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 | 产出 kernel |
|--------|-------|-------|------|------------|
| `m_out` | `(B,)` | `float32` | `m_i = max_j z_ij` | K1 |
| `s_out` | `(B,)` | `float32` | `s_i = sum_j exp(z_ij - m_i)` | K1 |
| `tau_out` | `(B,)` | `float32` | 合成阈值 `tau_i = max(tau^(k), tau^(p), tau^(mu))`（定理 T1） | K1 |
| `valid` | `(B,)` | `int32` | 降级守卫：`1` = 结果可信；`0` = 触发守卫（`k_i ∉ [1,K_MAX]` / nucleus > `K_MAX` / 有限值超额平票），host 须回退 torch 路径 | K1 |
| `w_out` | `(B, V)` | `float32` | 屏蔽后概率 `w_ij = p_ij * 1[z_ij >= tau_i]`，**未重归一化**（D-NORENORM），直接交 `torch.multinomial` | K2 |

**中间产物（不落 GM）**：`run_v[K_MAX]`（降序 top-`K_MAX` 值列表）、`l_val[K_MAX]`、`merged[2*K_MAX]`。
调试时可临时经 `T.dump_tensor` / `T.printf` 导出（`tilelang/language/print.py`）。

**K2 不接收 `valid`**（与草稿 §3.1 的差异）：`valid==0` 时 host 直接丢弃 `w_out` 并回退 torch，
K2 无条件计算即可，省一个 GM 读参数与分支。

### 4.3 中间缓冲区

**K1 `r4_topk_stats`**（S1'：`K_MAX=256, BV=4096`，单任务 = 单行）：

| Buffer | Shape | dtype | 存储层级 | 用途 |
|--------|-------|-------|----------|------|
| `z_ub` | `(1, 4096)` | float32 | UB (`alloc_shared`) | GM 分块搬入；`T.reduce_max/sum` 源（满块全量归约；尾块另设精确尺寸 `[1,TAIL]` 缓冲 `zt/et/mt/ztt`，§5.3 N1/V01.1） |
| `zt_ub` | `(4096,)` | float32 | UB | `T.tile.topk` 源（**rank-1，使 `max_actual_num` 精确 = BV**，§5.3 N2） |
| `e_ub` | `(1, 4096)` | float32 | UB | `exp(z - m)`，`T.reduce_sum` 源 |
| `l_dst` | `(512,)` | float32 | UB | 块内 topk 输出（val/idx 交错），`aligned_topk = ceil(2*256/8)*8 = 512` |
| `l_val` | `(256,)` | float32 | UB | `gather_mask("P0101")` 提取的块内 top-256 值 |
| `merged` | `(512,)` | float32 | UB | `concat(run_v, l_val)`，归并 topk 的源 |
| `m_dst` | `(512,)` | float32 | UB | 归并 topk 输出 |
| `run_v` | `(256,)` | float32 | UB | **跨块 running top-`K_MAX`（降序）** —— 核心状态 |
| `tie_m` | `(256,)` | float32 | UB | tie 检测的 0/1 掩码（`n_gt` / `n_eq` 复用同一 buffer） |
| `m_blk` / `s_blk` / `cnt` | `(1,)` ×3 | float32 | UB | reduce 输出 |
| `m_run, s_run, k_i, p_i, mu_i, tau_k, tau_p, tau_u, tau, prefix, vflag, n_gt, n_eq` 等 | `(1,)`/`(1,1)` ×15+ | float32 / int32 | ~~`alloc_var`~~ **`alloc_shared`（V01.1：`alloc_var` 重绑定结果错误，全部标量改 `[1]`/`[1,1]` UB buffer + `T.tile.*`，`debug_log.md` 发现 1/6）** | 标量状态 |
| `tmp_ub`（**编译器自动注入**） | `(65536,)` | uint8 | UB (`shared.ub`) | topk / reduce 的公共 tmp arena，尺寸 = 所有 call 的 **max**（§4.5） |

**K2 `r4_mask_sample`**：

| Buffer | Shape | dtype | 存储层级 | 用途 |
|--------|-------|-------|----------|------|
| `z_ub` | `(1, 4096)` | float32 | UB | 二次分块搬入 |
| `w_ub` | `(1, 4096)` | float32 | UB | 屏蔽后概率 |
| `m_v, s_v, t_v` | `(1,1)` ×3 | float32 | ~~`alloc_var`~~ **`alloc_shared`（V01.1，同上）** | 行标量 `m_i, s_i, tau_i` |

> 草稿 K2 的 `keep_ub[8192] uint8`（8 KB）**已消除**：Developer 模式用 `T.if_then_else` 在
> `T.Parallel` 内一次完成「比较 + 选择 + exp + div」，无需物化独立掩码 buffer。
> **V01.1 修正**：实际实现采用 `T.tile.compare`+`T.tile.select`（标量 `T.exp`/运算符于纯 Vector
> kernel 无法 codegen，`debug_log.md` 发现 6），掩码 buffer 仅 `[1, BV/8] uint8` = **512 B**，
> 仍远小于草稿的 8 KB；另引入 `m2d/s2d/t2d [1,BV]` broadcast 临时量与 `e_ub`（均在 UB 预算内，
> 编译 + 运行验证通过）。

### 4.4 内存搬运路径

**纯 Vector 算子 —— 不涉及 L1 / L0A / L0B / L0C，无跨级搬运**（符合 AGENTS.md 核心原则 3：
GM ↔ UB 单级，全部经 `T.copy`）：

```
K1:  GM[logits]  --T.copy(gm2ub, 尾块自动 clamp；V01.1: 尾块直接拷入精确尺寸 [1,TAIL] 缓冲)-->  UB[z_ub (1,BV)]
     UB[z_ub]    --T.copy(ub2ub)---------------------------------->  UB[zt_ub (BV)]
     UB[z_ub]    --T.reduce_max(dim=-1, 全量归约/V01.1)----------->  UB[m_blk (1)]  --scalar--> var m_run
     UB[z_ub]    --T.Parallel(exp)-------------------------------->  UB[e_ub (1,BV)]
     UB[e_ub]    --T.reduce_sum(dim=-1, 全量归约/V01.1)----------->  UB[s_blk (1)]  --scalar--> var s_run
     UB[zt_ub]   --T.tile.topk(K_MAX, valid)---------------------->  UB[l_dst (512)]   [用 tmp_ub arena]
     UB[l_dst]   --T.tile.gather_mask("P0101")-------------------->  UB[l_val (256)]
     UB[run_v] + UB[l_val] --T.copy(ub2ub ×2)--------------------->  UB[merged (512)]
     UB[merged]  --T.tile.topk(K_MAX, 2*K_MAX)-------------------->  UB[m_dst (512)]   [用 tmp_ub arena]
     UB[m_dst]   --T.tile.gather_mask("P0101")-------------------->  UB[run_v (256)]      (循环 n_blk 次)
     UB[run_v]   --标量/小向量收尾(tau_k, tau_p, tau_u, tie)------->  var tau, vflag
     GM[top_ks/top_ps/min_ps] --标量读(1 elem/行)------------------>  var k_i, p_i, mu_i
     var m_run/s_run/tau/vflag --标量写(1 elem/行)----------------->  GM[m_out/s_out/tau_out/valid]

K2:  GM[logits]  --T.copy(gm2ub, pad_value=-inf)------------------>  UB[z_ub (1,BV)]
     GM[m_in/s_in/tau_in] --标量读(1 elem/行)---------------------->  var m_v, s_v, t_v
     UB[z_ub]    --T.Parallel(if_then_else + exp + div)----------->  UB[w_ub (1,BV)]
     UB[w_ub]    --T.copy(ub2gm, 尾块自动 re-clamp)--------------->  GM[w_out]        (循环 n_blk 次)
```

**GM 遍历次数与流量核算**（B=256, V=152064, fp32；单张量 = 256×152064×4 B = 148.5 MB）：

| 阶段 | 读 | 写 |
|------|----|----|
| K1 | 1 × `logits` = 148.5 MB | 4 × `[B]` ≈ 4 KB（可忽略） |
| K2 | 1 × `logits` = 148.5 MB（+ 3 × `[B]` 可忽略） | 1 × `w_out` = 148.5 MB |
| host `torch.multinomial` | 1 × `w_out` = 148.5 MB | `[B]` 可忽略 |
| **合计** | **≈ 445.5 MB** | **≈ 148.5 MB** |

对比草稿 §8.3：V01-a 总流量 ≈ **594 MB**，相对现状 Path 2（≈1931 MB）**3.25x**、
Path 3（≈2969 MB）**5.0x**。最大确定性收益是**消除 Path 3 的 O(V log V) 全 vocab 排序与 `torch.cumsum`**。

### 4.5 UB 内存预算

**目标平台**：Ascend910_9362（实测）→ `determine_platform()` = **A3** → arch22 / **DAV_2201**
→ UB 总量 **192 KB**，系统保留 8 KB，**向量可用 184 KB = 188,416 B**。
（192 KB 佐证：`examples/pos_embedding/RotaryPositionEmbedding/rope_half_interleaved.py:218`
`UB_LIMIT = 196608  # 192KB`；`examples/topk_selector/example_topk_selector.py:127`；
`examples/gqa_fwd_varlen/gqa_fwd_varlen.py:128`。）

**K1 `r4_topk_stats`（S1'：`K_MAX=256, BV=4096`, fp32, 单行/任务）**

| Buffer | Shape | dtype | Bytes |
|--------|-------|-------|-------|
| `z_ub` | (1, 4096) | fp32 | 16,384 |
| `zt_ub` | (4096,) | fp32 | 16,384 |
| `e_ub` | (1, 4096) | fp32 | 16,384 |
| `l_dst` | (512,) | fp32 | 2,048 |
| `l_val` | (256,) | fp32 | 1,024 |
| `merged` | (512,) | fp32 | 2,048 |
| `m_dst` | (512,) | fp32 | 2,048 |
| `run_v` | (256,) | fp32 | 1,024 |
| `tie_m` | (256,) | fp32 | 1,024 |
| `m_blk`,`s_blk`,`cnt` | (1,)×3 | fp32 | 12 |
| 15 × `alloc_var` | (1,)×15 | fp32/int32 | 60 |
| **`tmp_ub`（自动注入）** | — | uint8 | **65,536** |
| **总计** | | | **123,972 B ≈ 121.1 KB** |
| **判定** | | | **✅ 121.1 KB ≤ 184 KB，余量 62.9 KB（34%）** |

**`tmp_ub` arena 尺寸推导**（`allocate_tmp_buffer.cc:578-596` + `:1195-1206`，单一共享 arena 取 **max**）：

| 消费 call | 公式（fp32） | Bytes |
|-----------|-------------|-------|
| 块内 `topk`（src `zt_ub[4096]`） | `factor 4 × (ceil(4096/32)×32 = 4096) × 4 B` | **65,536** ← max |
| `reduce_sum`（rows=1, cols=4096, fp32） | `1 × FloorPowerOfTwo(4096)=4096 × 4` | 16,384 |
| 归并 `topk`（src `merged[512]`） | `4 × 512 × 4` | 8,192 |
| tie 的 `reduce_sum`（cols=256） | `1 × 256 × 4` | 1,024 |
| `reduce_max`（rows=1, cols=4096, fp32） | `1 × 256`（cols ≥ 64 ⟹ `kVectorRepeatBytes`） | 256 |

> 🔑 **关键设计洞察（草稿完全遗漏）**：fp32 topk 的自动 tmp arena = **`16 × AlignUp(BV, 32)` 字节**，
> 与 `K_MAX` **无关**、与 `BV` **线性相关**。草稿 §5.3.6/§5.3.7 把「topk 相关」估为 7 KB、
> S1' 合计 41 KB，**低估约 3 倍**（真实 ≈121 KB）。
> 这给出明确的调优杠杆：`BV=2048 → arena 32 KB（K1 合计 ≈88 KB）`；`BV=1024 → arena 16 KB（≈72 KB）`，
> 代价是 `n_blk` 翻倍/四倍（38 → 75 → 149）。V01-a 按用户指定固定 `BV=4096`，余量 62.9 KB 充足。

**K1 UB 占用通式（供未来开其他特化时快速校验，单位 Byte）**

```
UB_K1(BV, K_MAX) = 12·BV                      # z_ub + zt_ub + e_ub，各 4·BV（fp32）
                 + 16·AlignUp(BV, 32)         # topk 自动 tmp arena（= max over calls，见上表）
                 + 36·K_MAX                   # l_dst 8K + l_val 4K + merged 8K + m_dst 8K + run_v 4K + tie_m 4K
                 + 72                         # 3 × reduce 输出 + 15 × alloc_var
   约束：UB_K1 ≤ 188416 (184 KB)
   ⟹ BV % 32 == 0 时简化为：  28·BV + 36·K_MAX + 72 ≤ 188416
```

| 特化 | `28·BV + 36·K_MAX + 72` | ≤ 188416 | 结论 |
|------|------------------------|----------|------|
| **S1'（本设计）`BV=4096, K_MAX=256`** | **123,976 B ≈ 121.1 KB** | ✅ 余 62.9 KB | **V01-a 采用** |
| `BV=4096, K_MAX=1024` | 151,624 B ≈ 148.1 KB | ✅ 余 36.0 KB | 可行（Stage 3 若需覆盖平坦分布 / 大 top_k，见 R-KMAX） |
| `BV=2048, K_MAX=256` | 66,744 B ≈ 65.2 KB | ✅ 余 118.8 KB | 可行，`n_blk=75`（Stage 3 双缓冲流水线的候选配置） |
| `BV=6144, K_MAX=256` | 181,176 B ≈ 176.9 KB | ⚠️ 余 7.1 KB | 临界，不推荐（`V=152064` 时 `TAIL=1088 ≥ K_MAX` 仍满足 C3） |
| **`BV=8192, K_MAX=256`（草稿 S1）** | **238,408 B ≈ 232.8 KB** | ❌ **超 44.4 KB** | **不可行**（草稿标为「✅ 余 111 KB」，系漏算 tmp arena） |
| **`BV=8192, K_MAX=1024`（草稿 S2'）** | **266,240 B ≈ 260.0 KB** | ❌ **超 76.0 KB** | **不可行**（草稿标为「✅ 余 90 KB」） |

> **`BV` 上界（`K_MAX=256` 固定时）**：`BV ≤ (188416 − 36·256 − 72) / 28 = 6400`，
> 叠加 `BV % 32 == 0` ⟹ 实用上界 **`BV = 6144`**。V01-a 取 4096（用户指定）。

**K2 `r4_mask_sample`**

| Buffer | Shape | dtype | Bytes |
|--------|-------|-------|-------|
| `z_ub` | (1, 4096) | fp32 | 16,384 |
| `w_ub` | (1, 4096) | fp32 | 16,384 |
| 3 × `alloc_var` | (1,)×3 | fp32 | 12 |
| `tmp_ub` | — | — | 0（无 topk / reduce / broadcast；若 codegen 为 `T.exp` 路由到需 tmp 的 intrinsic，估算 ≤ 2×16 KB，余量足以覆盖） |
| **总计** | | | **32,780 B ≈ 32.0 KB** |
| **判定** | | | **✅ 32.0 KB ≤ 184 KB，余量 152 KB** |

**反面验证（为何不能全 vocab 一次 top-k）**：`T.tile.topk` 要求 src 全量驻留 UB。
`V=152064` fp32 → 594 KB ≫ 184 KB（**超 3.2 倍**）；且 tmp arena 会再要 `16 × 152064 = 2.34 MB`。
`example_topk_selector.py:127` 亦明示 *"N (max ~8192 for 192KB UB)"*。
⟹ **必须分层归并**（草稿 D1 结论正确，本设计沿用）。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 | 处理方式 |
|--------|----------|-----------|---------|
| `B`（batch / token 数） | `@tilelang.jit` 函数的 Python int 形参 → 决定 `T.Tensor([B, …])` 的 GM shape 与 `b_num = T.ceildiv(B, VEC_NUM)` | decode 1~256；prefill 尾 token 1~1024 | **仅影响 grid 大小与行守卫 `if row < B`**，不作循环边界；算法结构与 UB 占用与 `B` 完全无关（一行一任务） |
| `V`（vocab） | 编译期常量 | `{32000, 128256, 129280, 151936, 152064}`（V01-a 只做 152064） | **必须静态**：`T.tile.topk` 要求 src 静态 shape；`n_blk`/`TAIL` 须静态 |
| `k_i, p_i, mu_i` | 运行时张量元素（`int32` / `float32`） | `k ∈ [1,1024] ∪ {TOP_K_ALL}`、`p ∈ (0,1]`、`mu ∈ [0,1]` | kernel 内标量读入 `alloc_var`，**不进入编译期特化键** |

> **⚠️ R-JIT（对草稿 §3.4「B 不进入特化键」的修正）**：草稿声称 `B` 不进入特化键。
> 经复核，**在当前约束下无法完全做到**：`logits` 的 GM shape 为 `[B, V]`，`B` 必须出现在
> `T.Tensor` 声明中 ⟹ 成为 `@tilelang.jit` 的编译期参数 ⟹ **每个不同 `B` 触发一次重编译**。
> 唯一能消除该依赖的写法是「固定 `B_MAX` 声明 shape + 运行时 `num_batches: T.int32`」
> （`fused_gdn_gating.py:143-154` 的 `compile_max_batch` + `num_batches` 范式），
> 但那要求 host 把 `[B,V]` 张量补成 `[B_MAX,V]` —— **属 `ascend-constraints.md §5` 明令禁止的
> host 侧 padding**。⟹ 本设计接受 `B` 进入 JIT 键，缓解措施：
> ① `TILELANG_CACHE_DIR` 持久化编译产物；② 生产环境**禁止** `tilelang.disable_cache()` /
> `clear_cache()`（示例代码中的调用仅为演示）；③ 实际 decode 的 `B` 由 scheduler 量化到有限集合，
> 稳态命中缓存。记为风险 **R-JIT**（§10.1）。

### 4.7 JIT 配置

```python
import tilelang
from tilelang import language as T

# Developer 模式 / 纯 Vector（programming-model-guide §2.2「纯 Vector 算子」行 + §2.1 四个开关说明）
R4_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,           # 自动核内同步 → 零手写 barrier
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,     # 自动 buffer 生命周期复用
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,     # 自动 CV 分离 → 零手写 T.Scope
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,        # 自动核间同步（本算子无 Cube，空转但保持一致）
}

@tilelang.jit(out_idx=[4, 5, 6, 7], pass_configs=R4_PASS_CONFIGS)
def r4_topk_stats(B: int, V: int, K_MAX: int, BV: int, NEED_MIN_P: bool): ...

@tilelang.jit(out_idx=[5], pass_configs=R4_PASS_CONFIGS)
def r4_mask_sample(B: int, V: int, BV: int): ...
```

- **不指定 `workspace_idx`**、签名无 `workspace_*` 参数（无 CV 交互，§8）。
- **不指定 `threads`**（§6.5）。
- 编译目标 `target="ascendc"`（自动）→ bisheng `--npu-arch=dav-2201`（`libgen.py:152-155`）。
- **备选写法**：若担心 `PassConfigKey` 导出时机，可改用原始字符串
  `{"tl.ascend_auto_sync": True, "tl.ascend_memory_planning": True, "tl.ascend_auto_cross_core_sync": True, "tl.ascend_auto_cv_combine": True}`
  —— `examples/xllm_kernels/utils.py:3-10` 的既有做法，可直接复用其 `DEFAULT_ASCEND_PASS_CONFIGS`。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：**纯 Vector**

**判定依据**：算子只含 reduction（`max` / `sum`）、elementwise（`exp` / `ln` / `sub` / `div` /
`compare` / `select`）与 selection（`topk` / `gather_mask`），**零 matmul** ⟹ 不使用 Cube 核、
不使用 L1 / L0A / L0B / L0C。`src/transform/common/operation_config.h` 中本设计用到的全部 intrinsic
（`tl.ascend_reduce`、`tl.ascend_topk`、`tl.ascend_gather_mask`）均标注 **`PIPE_V`**。

### 5.2 Block 划分

```python
VEC_NUM = 2                                   # 每 Cube 核配 2 个 Vector 核
                                              # examples/xllm_kernels/fused_gdn_gating.py:26
vec_core_num  = detect_vec_core_num()         # 运行时探测 → Ascend910_9362 实测 40
                                              # examples/xllm_kernels/utils.py:20-40
                                              # 🔴 禁止硬编码 48（草稿假设值，与实测不符）
cube_core_num = vec_core_num // VEC_NUM       # 实测 20

BV     = 4096                                 # vocab 分块（S1' 特化，用户指定）
K_MAX  = 256                                  # top-k 归并缓冲上限（S1' 特化，用户指定）
n_blk  = T.ceildiv(V, BV)                     # V=152064 → 38（37 满块 + 1 尾块 TAIL=512）
b_num  = T.ceildiv(B, VEC_NUM)                # grid：一行一任务；B=64 → 32 block

# 并行映射：task_id ↔ row_id（一行一任务，无跨核归约）
row = cid * VEC_NUM + vid                     # ∈ [0, b_num*VEC_NUM)
if row < B:  process_row(row)                 # 行守卫
```

**Tile Shape 汇总**

| 维度 | Tile | 值（S1'） | 选择理由 |
|------|------|----------|---------|
| vocab（V） | `BV` | **4096** | ① `BV % 32 == 0` ⟹ topk 的 `AlignUp(actual_num,32)` 不越界；② UB 预算通式 `28·BV + 36·K_MAX + 72 ≤ 188416`（§4.5）⟹ `K_MAX=256` 时 **`BV ≤ 6400`**，4096 有余量（121.1 KB / 184 KB）；③ `n_blk = 37 满块 + 1 尾块 = 38 次迭代`，使 online-softmax 的标量重标定次数可控；④ 尾块 `TAIL = 512 ≥ K_MAX = 256` ⟹ 满足 topk 的 `K ≤ actual_num`（C3） |
| top-k（K） | `K_MAX` | **256** | 覆盖 decode 主流 `top_k ∈ {20, 40, 50, 100}`；归并 buffer 仅 `O(K_MAX)` 与 V 无关；收尾标量循环 256 次代价可忽略 |
| batch（B） | 1 行 / 任务 | — | **一行一任务** ⟹ 无跨核归约、无 `atomic_add`（GM-only）、UB 私有（草稿 D7 结论正确，且被 §6.5 的 topk 语义强制） |
| grid | `b_num` | `ceildiv(B,2)` | 允许超订（`topk_selector` B=1024 → b_num=512 实测可跑），运行时按核数分波调度 |

### 5.3 约束分析

- **对齐约束**：`BV = 4096`，fp32 尾轴 4096 元素 = 16,384 B，是 32 B 的整数倍 ✅；
  `TAIL = 512` 元素 = 2,048 B，同为 32 B 整数倍 ✅。`T.copy` gm2ub 要求 32 B 对齐 —— 满足。
- **UB 容量**：K1 = 121.1 KB，K2 = 32.0 KB，均 ≤ 184 KB ✅（§4.5）
- **L0 容量**：**不适用**（纯 Vector，无 Cube 计算）
- **L1 容量**：**不适用**
- **L2 复用**：A3 L2 = 168 MB。K2 二次读 `logits`：`B=1` 时单行 594 KB ≪ 168 MB ⟹ **高命中率**；
  `B=256` 时 148.5 MB ≈ L2 容量 ⟹ 部分命中。decode 小 batch（R4 主战场）恰好受益。
- **核数利用**：`B ≥ 40` 满核；`B < 40` 时 `b_num = ceildiv(B,2)` 收缩 grid（不空转）。
  `B = 1` ⟹ `b_num = 1`，仅 1 个 Vector 核工作 —— decode 单流场景 R4 本身耗时极短，
  核利用率非首要指标（草稿 D7 权衡成立）。

**设计注记（Stage 2 实现时必读）**

- **N1 —— 尾块归约用「精确尺寸缓冲 + 全量 reduce」（V01.1 修正，NPU 验证）**：原设计为
  rank-2 `[1,BV]` `z_ub` + `real_shape=[1, valid]` narrow-reduce（`real_shape` 只对 rank-2 生效，
  `reduce_ascend.py:99-108/:318-328`）。Stage 2 上板实测（`v2_narrow_reduce.py`，`debug_log.md` V-2）：
  **narrow-reduce 路径（`real_shape` 窄于 buffer，或列切片 BufferRegion）受
  `src/target/codegen_ascend.cc:2145` 硬约束 —— 逻辑宽度 `nn*elem_bytes <= 256` B = 64 个 fp32
  元素**，R4 尾块 `valid=512`（2048 B）远超此限
  （`InternalError: Check failed: (nn * elem_bytes <= 256) ... got 512 elements of 4 bytes`）。
  ⟹ **实际实现**：满块 `[1,BV]` **全量 reduce**（无需 `real_shape`）；尾块单独分配**精确尺寸
  `[1,TAIL]` 缓冲**（`zt/et/mt/ztt`，`T.copy` gm2ub 自动 clamp 到 TAIL）+ **全量 reduce（不传
  `real_shape`）**，走 normal reduce 路径（`example_online_softmax` 已证 block_N=128 fp32 = 512 B
  > 256 可用）。V=152064、BV=4096（37 满块 + 512 尾块）的 online-softmax `m/s` 与 CPU 对拍正确
  （m 逐位相同、s rtol < 1e-3）。原结论不变：**不依赖 `pad_value` 保证归约正确性**
  （`test_..._tail_block.py:45-51` 实测：依赖 `-inf` pad 会在 NPU 上产生 inf/nan）。
  narrow 宽度 ≤ 64 fp32 时 `real_shape`/BufferRegion 仍可用（`test_..._narrow_reduce.py`
  的测试通过恰因宽度低于限值）。
- **N2 —— `zt_ub` 必须是 rank-1 `[BV]`（topk 专用）**：`T.tile.topk` 的
  `max_actual_num = Σ src.shape`（**求和而非求积**，wheel `ascend_tile.py:578-590`；
  `test_..._topk.py:86-89` docstring 与 `:349` 用例 `M=2, per_row=16, actual_num=18` 双重印证）。
  若直接把 `[1, BV]` 传给 topk，`max_actual_num = 1 + 4096 = 4097` ⟹ `repeatTimes = 129`
  ⟹ 排序会覆盖 `129×32 = 4128 > 4096` 个槽位，**存在读越界 32 元素（128 B）并污染 top-K 结果的风险**。
  ⟹ 用 rank-1 `zt_ub[BV]`（`max_actual_num = 4096` 精确）作 topk 源，代价 16 KB UB + 每块一次
  UB→UB `T.copy`（预算充足，见 §4.5）。
  **优化选项 —— V01.1：已放弃**：原拟省去 `zt_ub`，改用 rank-1 `z_ub[BV]` +
  BufferRegion 归约 `T.reduce_max(z_ub[0:valid], m_blk, dim=-1)`。V-2 实测 BufferRegion 列切片
  与 `real_shape` 走**同一 narrow-reduce codegen 路径**，受 `codegen_ascend.cc:2145` ≤256 B
  （=64 fp32）约束 —— 该优化的目标场景（尾块 `valid=512` 的窄 region）**不可用**；满块全宽
  region 是否回退 normal 路径未验证。⟹ 实际实现放弃该优化（尾块已改精确尺寸 `[1,TAIL]`
  缓冲 + 全量 reduce，N1；`zt_ub` 保留）。
- **N3 —— topk 的 in-place `-inf` padding 安全**：`test_..._topk.py:181-183` docstring 明示
  *"float32: src IS modified in-place … On ascendc the whole tail `[actual_num:buffer_size)` is padded"*。
  作用对象是 UB scratch `zt_ub`（**非 GM 视图**）⟹ 不污染 `logits`（草稿 R5 的正确规避方式）；
  且 `buffer_size = BV`，padding 上界不越 buffer ✅。
- **N4 —— `run_v` 首轮必须 `T.tile.fill(-inf)`**：否则未初始化 UB 内容会参与首次归并，结果不确定。
- **N5 —— `T.tile.topk` 与 `alloc_shared` 的组合未被仓库现有用例覆盖**（现有 6 处用例全部用
  `alloc_ub`）。scope 推导链已在 §2.3 论据 2 从源码证明可行，并列为**上板验证项 V-1**（§10.1 R-B）。
  **V01.1：Stage 2 已上板验证 ✅ PASS**（B=64,N=4096,top_k=2048 与 B=1024,N=1024,top_k=128
  两组 accuracy=1.00000，兜底未触发，风险关闭）。

### 5.4 注意事项（非整除与边界，显式设计）

**非整除事实**：`V = 152064`，`BV = 4096` ⟹ `152064 = 37 × 4096 + 512`，**不整除**，尾块 `TAIL = 512`。

| # | 约束 | 处理方案 |
|---|------|---------|
| **C1** | **输入侧尾块**（gm2ub） | `T.copy(logits[row, b*BV], z_ub, pad_value=-T.infinity("float32"))`。lowering 期由 `src/op/ascend.cc::compute_valid_extent` 自动 clamp：`valid = Select(shape-off >= extent, extent, Select(shape-off > 0, shape-off, 0))`（`test_..._tail_block.py:20-31`）。归约侧（**V01.1 修正，V-2 实测**）：`real_shape`/BufferRegion narrow-reduce 受 `codegen_ascend.cc:2145` ≤256 B（=64 fp32）约束，尾块 `valid=512` **编译失败** ⟹ 尾块用**精确尺寸 `[1,TAIL]` 缓冲 + 全量 reduce**（N1）；仍**不依赖 `pad_value` 保证正确性**（其被明确标注为 *"backend-dependent and NOT reliable as a correctness mechanism"*）。双保险。 |
| **C2** | **输出侧尾块**（ub2gm，K2） | `T.copy(w_ub, w_out[row, b*BV])` —— **ub2gm 自动 re-clamp，只写有效 extent**（`test_..._tail_block.py:30,41-42`）。**无需** host padding + crop，**无需**手写 valid-extent 切片（若 Stage 2 倾向显式写法，`T.copy(w_ub[0,:valid], w_out[row, off:off+valid])` 亦经 `test_..._copy_runtime_extent.py:90` 验证）。 |
| **C3** | **`T.tile.topk` 要求 `K ≤ actual_num`** | 尾块 `actual_num = TAIL = 512 ≥ K_MAX = 256` ✅。⚠️ **wheel 版本无该校验**（repo 新版 `ascend_tile.py:653-654` 才加），故必须 **host 侧前置断言**：`V % BV == 0 or V % BV >= K_MAX`，否则抛 `NotImplementedError`。S1' 满足。 |
| **C4** | **`actual_num ≥ 1`** | `actual_num = 0` 会触发硬件 aicore 异常（repo 版 `ascend_tile.py:651-652` 明示）。`TAIL > 0` 由编译期 `if TAIL > 0` 保证 ✅ |
| **C5** | **`B` 不被 `VEC_NUM` 整除** | 行守卫 `if row < B`（`example_cross_entro.py:62` 同款）。奇数 `B`（如 `B=1`）安全。 |
| **C6** | **`B == 0`** | **host 侧拦截**，直接返回空张量，**不下发 kernel**（zero-sized grid launch 已知会 abort，草稿 B1 引 `multi_layer_eagle.py:80-84`）。 |
| **C7** | **`V` 不在特化表 / `V < K_MAX`** | host 侧抛 `NotImplementedError`，回退 torch 路径。 |
| **C8** | **满块与尾块的代码复用** | 用 JIT 工厂函数内的 Python 闭包 `block_body(b_off, valid)`（或 `@T.macro`），`valid` 为**编译期静态** int（`BV` 或 `TAIL`）⟹ `real_shape` / `actual_num` / `T.Parallel(valid)` 全静态，不触发动态边界限制。 |

> **⚠️ 非整除显式设计声明**（`ascend-constraints.md §5`）：输入侧（C1）与输出侧（C2）均由
> lowering 期的 `compute_valid_extent` 动态切片裁剪搬运，**归约侧以精确尺寸尾块缓冲（`[1,TAIL]`）
> + 全量 reduce 保证逻辑有效宽度**（N1/V01.1：`real_shape` narrow-reduce 对 `valid > 64` fp32
> 不可用，`codegen_ascend.cc:2145` ≤256 B 约束，§10.2 V-2）。**host 侧不做任何 padding + crop**。

### 5.5 数据搬运性能可行性

**本算子非数据重排算子**（无 permute / transpose / 布局变换；`logits` 与 `w_out` 同为 `[B,V]`
行主序连续），故 `ascend-constraints.md §4.2` 的 `[REORDER-COST]` 记录为 **N/A（无物理布局变化）**。
但仍给出 GM/DMA/并行度成本核算：

```text
[REORDER-COST]
path: r4_fused_sampling (K1 + K2, pure Vector, no layout change)
gm_passes: 3 读 + 1 写（K1 读 logits 1 次；K2 读 logits 1 次 + 写 w_out 1 次；
           host multinomial 另读 w_out 1 次）
dma_transactions: K1 = B × n_blk_full(+1 尾块) = B × 38（gm2ub, 16 KB/次）
                       + B × 38（ub2ub, 16 KB/次）
                  K2 = B × 38（gm2ub）+ B × 38（ub2gm）
                  B=64 → 每 kernel ≈ 2432 次 DMA，K1+K2 ≈ 4864 次
average_dma_bytes: 16,384 B（= BV × 4，满块）；尾块 2,048 B。
                   均为 32 B 整数倍、长 DMA，无短搬运（不存在数十万次短 DMA）
gm_scalar_accesses: 每行 K1 7 次（top_ks/top_ps/min_ps 各 1 读；m/s/tau/valid 各 1 写）
                    + K2 每行 3 次 → B × 10；B=64 → 640 次
                    相对 numel = B×V = 9.73M 占比 6.6e-5 ⟹ 远小于 numel ✅
address_div_mod_per_element: 0（行内连续；行偏移 row*V 每行算 1 次，非每元素）
active_cores / serial_tasks_per_core: min(B, 40) 个 Vector 核；每核 1 行（B ≤ 40）
                                      或 ceil(B/40) 行（超订分波）；
                                      每行串行执行 n_blk = 38 次分块迭代
largest_case_timeout_gate: B=64, V=152064, fp32 → 单 case 900 s（含首次 JIT 编译，§9.4）
result: pass
```

**结论**：主路径全为 16 KB 长 DMA、**零逐元素 strided GM 访问**、**零 per-element 地址 div/mod**
⟹ 纯访存瓶颈算子的理想形态，`pass`。

### 5.6 Host 侧 Buffer 操作审计

```text
[HOST-METADATA-AUDIT]
operation: logits / top_ks / top_ps / min_ps 由调用方（测试 harness）在 CPU 构造后 .npu() 一次性 H2D
input_stride -> output_stride: contiguous [B,V] / [B] → 不变
shares_storage / same_data_ptr: true（无 host 侧变换）
aclnn_or_physical_copy: false（H2D 属测试数据准备，ascend-constraints §5 明确允许）
result: allow

[HOST-METADATA-AUDIT]
operation: m_out / s_out / tau_out / valid / w_out 的分配
input_stride -> output_stride: N/A（由 tilelang.jit 的 out_idx 机制在 device 侧分配）
shares_storage / same_data_ptr: N/A
aclnn_or_physical_copy: false
result: allow

[HOST-METADATA-AUDIT]
operation: kernel 调用 r4_topk_stats(...) → r4_mask_sample(...)
input_stride -> output_stride: 不变
shares_storage / same_data_ptr: true
aclnn_or_physical_copy: false（TileLang 生成的 .so 直接 launch，不经 aclnn 算子库）
result: allow

[HOST-METADATA-AUDIT]
operation: valid.cpu() + Python all() —— 降级守卫判定（D-GUARD）
input_stride -> output_stride: [B] int32 device → host
shares_storage / same_data_ptr: false（D2H 拷贝，属分派决策所需的标量级读回）
aclnn_or_physical_copy: false（D2H memcpy，非 aclnn 计算算子）
result: allow —— 依据：ascend-constraints §5 允许「数据准备、kernel 调用和结果验证」；
                    本操作是 dispatch 决策，不产出算子的任何计算结果。
                    ⚠️ 代价：一次 D2H 同步（草稿 D8 已论证「相对正确性必要」）。
                    ⚠️ 该同步会破坏 ACLGraph 图捕获 —— ACLGraph 入图属本 V01-a 范围外。

[HOST-METADATA-AUDIT]
operation: torch.multinomial(w_out, num_samples=1)
input_stride -> output_stride: w_out [B,V] fp32 contiguous → token_id [B,1] int64
shares_storage / same_data_ptr: false（新建输出张量）
aclnn_or_physical_copy: true（aclnn 算子）
result: allow（有条件）—— 依据：这是 V01-a 用户明确要求保留的【下游独立算子】，
                    不是 R4 自身计算（softmax/top-k/top-p/min-p 屏蔽）的 host 替代；
                    R4 的全部融合计算均在 kernel 内完成，w_out 是其真实输出。
                    ⚠️ 与 cann-bench「host 侧零 aclnn」提交规范不兼容 —— 但本算子落点是
                    examples/R4/ 独立算子（非 SGLang backend 集成、非 cann-bench 提交），
                    且用户已明确将 multinomial 融合划入 out-of-scope（V01-b）。
                    ⟹ 记录为已知偏差 DEV-1（§10.3），不阻塞本阶段交付。
```

**无违规项**：host 路径不含 `.contiguous()` / `reshape` 物理化 / `permute` / `torch.cat` /
`torch.stack` / `F.pad` / `.to(dtype)` / `.clone()` / `x[:] = …` / in-place 算术。
`logits` 的 temperature 缩放（**`sampler.py:249 logits.div_(sampling_info.temperatures)`**，
已读源码核对；草稿引用的 `:641` 为过期行号）+ 下游 `torch.nn.functional.log_softmax`
（`sampler.py:262`）**保留在调用方**，不属 R4 host 路径 —— 草稿 D5 的结论正确：
`div_` 是 in-place 且 `log_softmax` 依赖缩放后的同一张量，若并入 kernel 会导致重复计算或额外张量。

### 5.7 分派覆盖审计

```text
[DISPATCH-COVERAGE]
supported_domain:
  dtype   : logits/top_ps/min_ps/m_out/s_out/tau_out/w_out = float32；top_ks/valid = int32
  shape   : logits [B, V]，B ∈ [1, 1024]，V = 152064（V01-a 单特化 S1'；
            结构上支持任意满足 C3/C7 的静态 V）
  attr    : K_MAX = 256，BV = 4096，NEED_MIN_P ∈ {True, False}
            逐行 k_i ∈ [1, K_MAX]，p_i ∈ (0,1]，mu_i ∈ [0,1]
generic_fallback: host 侧回退 —— valid 中存在 0、或前置校验不通过时，整批交
                  torch 参考路径（r4_ref.py 的 Path 3 语义，等价 sglang sampler.py:846-879）
specializations:
  - predicate: V == 152064 and K_MAX == 256 and BV == 4096 and dtype == float32   (S1')
    fallback_on_miss: host 前置校验抛 NotImplementedError → torch 参考路径
    equivalence_evidence: 定理 T1/T2（§1.3）+ 分层归并正确性归纳证明（§1.4）
                          + golden 逐行复刻 sampler.py:846-879（§9.1）
  - predicate: NEED_MIN_P == False
    fallback_on_miss: N/A（编译期开关，两分支均实现；False 分支跳过 tau_u，等价 mu=0）
    equivalence_evidence: 定理 T2 的 mu=0 ⟹ tau_u = -inf ⟹ 在 max() 中不生效
  - predicate: 逐行 k_i ∉ [1, K_MAX]（含 TOP_K_ALL = 1<<30）
    fallback_on_miss: valid[row]=0 → host 整批回退
    equivalence_evidence: D-GUARD；与现状一致（此类输入本就 npu_top_k_top_p_eligible=False → Path 3）
  - predicate: 逐行 nucleus 规模 > K_MAX（ok_p == 0）
    fallback_on_miss: valid[row]=0 → host 整批回退
    equivalence_evidence: D-GUARD（修正草稿 §3.7 的失效判据）
  - predicate: top-k 绑定且存在有限值超额平票（tie_ok == 0）
    fallback_on_miss: valid[row]=0 → host 整批回退
    equivalence_evidence: D-TIE；-inf 平票已论证无害（w=0）
unsupported_inputs:
  - dtype ≠ float32（bf16 推迟）
  - sampling_seed ≠ None（确定性路径，依赖 float64，TileLang 零支持）
  - logits 含 NaN（上游 sanitize_nan_logits 已清洗，kernel 不做二次防护）
  - B == 0（host 拦截，不下发 kernel）
  - V 不满足 C3/C7（host 前置校验拦截）
  ⟹ 与 supported_domain 不冲突（均为域外，且全部有 host 拦截或回退）
result: pass
```

**验证命中**：每条 specialization 与 fallback 均在 §9.2 L0 计划或 §9.4 哨兵中有命中 case
（S1' → 全部 L0；`NEED_MIN_P=False` → l0_01/02/03/07/08；`=True` → l0_04/05/06；
`k_i ∉ [1,K_MAX]` / `ok_p=0` / `tie_ok=0` 三条 fallback 属**负向路径**，由 Stage 2 的
L2/Boundary 覆盖 —— 本阶段仅声明其存在与判据，**不在此枚举 L1/L2/Boundary 用例**）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| batch（B 行） | **block 级并行** | `T.Kernel(b_num, is_npu=True) as (cid, vid)`，`row = cid*VEC_NUM + vid` | 一行一任务 ⟹ 无跨核归约、UB 私有、不需 `atomic_add`（草稿 D7 + §6.5 的 API 语义强制） |
| 行有效性 | **条件守卫** | `if row < B:` | 处理 `B` 不被 `VEC_NUM` 整除（C5） |
| vocab 满块 | **静态迭代** | `for b in T.serial(V // BV)` | trip count = 37，编译期常量 ✅ |
| vocab 尾块 | **编译期条件** | `if TAIL > 0: block_body(…)` | `TAIL = 512` 为编译期常量；单独发射使 `valid` 全静态（C8） |
| 块内元素级 | **向量化并行** | `for j in T.Parallel(BV)` | exp / sub / div / select 逐元素，无顺序依赖 |
| top-p 前缀和 | **静态串行标量循环** | `for l in T.serial(K_MAX)` + `with T.If(prefix <= p_i)` | 严格顺序依赖 ⟹ 必须串行；**单调守卫替代 `break`**（§6.4） |
| tie 统计 | **向量化并行** | `for l in T.Parallel(K_MAX)` + `T.reduce_sum` | 256 元素，全向量化 |
| **不启用** | — | ~~`T.Pipelined`~~ / ~~`T.Persistent`~~ | 见 §6.3 |

### 6.2 循环伪代码

```python
with T.Kernel(b_num, is_npu=True) as (cid, vid):        # b_num = ceildiv(B, 2)，静态
    row = cid * VEC_NUM + vid
    if row < B:                                          # 行守卫
        <分配 alloc_shared / alloc_var>
        <读 k_i, p_i, mu_i>；T.tile.fill(run_v, -inf)     # N4
        for b in T.serial(V // BV):                      # 静态 37 次
            block_body(b * BV, BV)                       # valid = BV（静态）
        if TAIL > 0:                                     # 编译期常量条件
            block_body((V // BV) * BV, TAIL)             # valid = TAIL = 512（静态）
        <收尾：tau_k / tau_u / tau_p 标量循环 / tie 向量化 / 守卫 / 写 GM>
```

`T.Parallel` 与 `T.serial` 的分工原则：**有顺序依赖的（online softmax 重标定、top-p 前缀和、
跨块归并）用 `T.serial`；无顺序依赖的逐元素/逐候选操作用 `T.Parallel`**。

### 6.3 流水线优化

**V01-a 不启用 `T.Pipelined`**，理由：

1. `ascend-constraints.md §1` 明示「流水线不支持动态边界」。本设计的分块循环边界**是静态的**
   （`V // BV`），技术上可用；但——
2. 块内存在**严格的循环携带依赖**：`m_run` / `s_run` / `run_v` 三个状态跨块串行更新
   （online softmax 的重标定 `s <- s*exp(m_old - m_new)` 与分层归并都依赖上一块结果）。
   `T.Pipelined(stage=2)` 只能重叠 **MTE2 搬运**与计算，无法重叠归约/归并本身，收益有限。
3. **UB 预算**：双缓冲需要 `z_ub` / `zt_ub` / `e_ub` 各 ×2 ⟹ K1 从 121 KB 升至 ≈170 KB，
   逼近 184 KB 上限，风险高、余量小。
4. V01-a 的首要目标是**精度收敛与功能正确**（Stage 2 门禁），性能调优属 Stage 3（可选阶段）。

**Stage 3 可选优化清单**（不在本阶段实施，仅记录）：
`T.Pipelined(stage=2)` 重叠 gm2ub（需同步把 `BV` 降到 2048 以腾出双缓冲 UB，arena 亦随之减半至 32 KB）；
`e_ub` alias `z_ub`（`TL_ASCEND_MEMORY_PLANNING` 可能已自动完成）；
N2 的 rank-1 单 buffer 优化（省 16 KB + 每块一次拷贝）；
K1+K2 合并为单 kernel 双遍历（草稿 §5.5 的 V02 项，B=1 时第二遍大概率命中 L2）。

### 6.4 尾块处理

见 §5.4 C1 / C2 / C8。要点复述：

- **满块**：`for b in T.serial(V // BV)`，`valid = BV = 4096`（静态）
- **尾块**：`if TAIL > 0:` 单独发射一次 `block_body((V//BV)*BV, TAIL)`，`valid = TAIL = 512`（静态）
- 归约侧（**V01.1 修正**，N1 / §10.2 V-2）：满块 `[1,BV]` 全量 reduce；尾块**精确尺寸 `[1,TAIL]`
  缓冲 + 全量 reduce**（`real_shape`/BufferRegion narrow-reduce 对 `valid > 64` fp32 **不可用**，
  `codegen_ascend.cc:2145` ≤256 B 约束）；仍**不依赖 `pad_value` 保证正确性**
- topk 侧传 `actual_num = valid`（静态，且 `TAIL=512 ≥ K_MAX=256` 满足 C3）
- 输出侧 `T.copy` 自动 re-clamp（C2）
- **host 侧不做 padding + crop**

**top-p 标量循环为何不需要 `break`**：`prefix` 从 0 起、每轮加上非负量 `exp(run_v[l]-m)/s`
⟹ **单调不减**。一旦 `prefix > p_i`，其后所有 `l` 的守卫 `prefix <= p_i` 恒假 ⟹ 不再更新
`tau_p` / `prefix`。这与「首次跨越即 break」**语义完全等价**，且规避了 Ascend 对动态循环边界 /
`T.loop_break()` 的支持不确定性。循环退出后的 `prefix` 值恰好用于 D-GUARD 的 `ok_p` 判定。

### 6.5 ⚠️ 为什么不用 Developer 模式推荐的 `threads=2` + `as (cid)`

`tilelang-programming-model-guide` §3.1.1 推荐 Developer 模式用
`T.Kernel(block_num, threads=2, is_npu=True) as (cid)` 来消除 `vid` / `workspace`。
**本设计不采用**，理由（均为硬约束，非偏好）：

1. **该推荐的适用域是 CV 融合**：§3.1.1 标题即「Developer 模式 **CV 交互**：优先消除 workspace / vid」，
   前提链 `threads=2 → vid 消除 → workspace 消除` 服务于 Cube↔Vector 片上直连。
   本算子**纯 Vector、零 Cube、零 workspace**，该链条的终点（消 workspace）本就是空集。
   guide §2.2 的「纯 Vector 算子」行只要求 `AUTO_SYNC` + `MEMORY_PLANNING`，**未要求 `threads=2`**。
2. **`threads=2` 会把一个 block 的工作自动切给 2 个 Vector 核**，而 `T.tile.topk`
   要求**整行 vocab 块驻留在单个核的 UB** 内做排序（`test_..._topk.py:149-154`：top-k 在
   flattened buffer 的前 `actual_num` 个元素上计算）。若一行被切到 2 核，每核只持有半个块
   ⟹ top-k 与 whole-row reduce 结果均错误。
3. **`T.tile.topk` 无法做 per-row 批量 top-k**：2D src 的语义是「flattened 后取前 `actual_num` 个」
   （`ascend_tile.py:578-590` 的 `max_actual_num = Σ shape`；`test_..._topk.py:86-89` docstring），
   **不是**每行独立 top-k ⟹ 无法用 `[BR, BV]` 的多行 tile 来喂 topk。
   ⟹ **「一行一任务」是被 API 语义强制的，不是设计偏好。**
4. **仓库全部纯 Vector 生产级实现均用 `as (cid, vid)`**：
   `example_online_softmax.py:52`、`example_cross_entro.py:60`、`example_topk_selector.py:32`、
   `fused_gdn_gating.py:152`（后者正是 `alloc_shared` + 全 auto pass_configs 的 Developer 形态）。

**结论**：保留 `as (cid, vid)` + 显式 `row = cid*VEC_NUM + vid`。这**不构成 Expert 模式回退** ——
Expert 的判别标志（`alloc_ub` / `T.Scope` / 手动 flag / pass_configs 全关）本设计一条都不用（§2.1）。
记为设计注记，供 Orchestrator / 用户知悉。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**全自动同步（Developer 模式）**

- **零手写同步原语**：不出现 `T.barrier_all()`、`T.set_flag()`、`T.wait_flag()`、
  `T.set_cross_flag()`、`T.wait_cross_flag()`、`T.pipe_barrier()`、`T.annotate_address()`。
- **零手写作用域**：不出现 `with T.Scope("C")` / `with T.Scope("V")`。
  （`TL_ASCEND_AUTO_CV_COMBINE` 开启时手写 `T.Scope` 反而可能干扰编译器，见 guide §2.1 ③ 注意事项。）

### 7.2 同步点说明

**无手动同步点。** 由编译器自动插入的同步覆盖以下依赖（列出以供审阅，**不需手写**）：

| 依赖 | 类型 | 自动同步来源 |
|------|------|-------------|
| `T.copy(GM → z_ub)` 完成后再 `T.reduce_max(z_ub, …)` | MTE2 → V | `TL_ASCEND_AUTO_SYNC` |
| `T.copy(z_ub → zt_ub)` 完成后再 `T.tile.topk(zt_ub, …)` | V/MTE3 → V | `TL_ASCEND_AUTO_SYNC` |
| `T.tile.topk` 写 `l_dst` 完成后再 `T.tile.gather_mask(l_dst, …)` | V → V（同 buffer RAW） | `TL_ASCEND_AUTO_SYNC` |
| `T.copy(run_v → merged)` / `T.copy(l_val → merged)` 完成后再 `T.tile.topk(merged, …)` | V → V RAW | `TL_ASCEND_AUTO_SYNC` |
| `T.reduce_*` 与其 `tmp_ub` arena 的复用（**arena 无内容契约，每个 call 可覆写**） | V → V WAR/WAW | `TL_ASCEND_AUTO_SYNC` + `allocate_tmp_buffer.cc:500-502` 的 `kWorkspaceWriteAccess` 依赖建模 |
| `z_ub` / `e_ub` / `zt_ub` 等生命周期不重叠 buffer 的地址复用 | — | `TL_ASCEND_MEMORY_PLANNING` |
| **跨核同步** | — | **不需要**：一行一任务 ⟹ 无跨核归约、无共享 buffer、无 `atomic_add`（草稿 D7） |
| **Cube↔Vector 核间同步** | — | **不适用**：零 Cube 操作。`TL_ASCEND_AUTO_CV_SYNC` 仍开启以与仓库惯例一致（`example_online_softmax.py:7-11` 未开、`example_topk_selector.py:12-17` 开），本算子无 CV 边界故为空转 |
| **K1 → K2 的 kernel 间顺序** | — | 由 host 侧顺序调用两次 `@tilelang.jit` 函数保证（kernel launch 天然串行化，中间经 GM 传递 `m/s/tau/valid`） |

> ⚠️ 跨代际陷阱（草稿 D9 引 Programming Guide `:2330`）：*"A2 会在算子之间清理多余计数，
> 而 Ascend 950 要求严格配对"* → 手工 MTE 事件漏配对在 A2/A3 能跑过、A5 会挂。
> **本设计不使用任何显式 MTE 事件配对**（全依赖 `TL_ASCEND_AUTO_SYNC`），故不受影响。

### 7.3 pass_configs 配置

```python
R4_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,           # ① 自动核内同步（替代全部手写 barrier/flag）
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,     # ② 自动内存规划（复用中间 buffer）
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,     # ③ 自动 CV 分离（替代手写 T.Scope）
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,        # ④ 自动核间同步（本算子无 Cube，空转）
}
```

依据 `tilelang-programming-model-guide` §2.1（四个开关语义）与 §2.2（场景表）：
纯 Vector 算子**必需** ①②；③④ 按 guide 表格对纯 Vector 标为「❌ 不需要」，但开启为**无害空转**
且与仓库既有纯 Vector 示例一致（`example_topk_selector.py:12-17` 四项全开、
`example_cross_entropy_loss` 开 ①②③、`test_..._topk.py:30-33` 四项全开）。
**保守起见本设计四项全开**，与 `examples/xllm_kernels/utils.py:3-10` 的
`DEFAULT_ASCEND_PASS_CONFIGS` 完全一致。

> 若 Stage 2 发现开启 ③（AUTO_CV_COMBINE）导致纯 Vector kernel 的 `alloc_var` 定义/使用核归属
> 误判（`ascend-constraints.md §6` 方案 B 提示的已知现象），可**单独关闭 ③④** 而保留 ①②
> （即 guide §2.2「纯 Vector 算子」行的最小配置）。这是唯一允许的 pass_configs 调整，
> 不改变 Developer 模式定性。

---

## 8. 融合算子设计

### 8.1 融合算子判定

**判定结果**：**否**（就本 skill 模板定义的「融合算子 = GEMM + element-wise 后处理」而言）

**判定依据**：R4 融合了**四个采样步骤**（softmax / top-k / top-p / min-p），属**算法级融合**，
但**不含任何 GEMM / Cube 计算** ⟹ 不存在 Cube↔Vector 核间交互，不适用本章节的 CV 融合设计。

### 8.2 CV 交互设计

**不适用**。具体地：

- **无 workspace 规格**（不产出 workspace 表）
- **无 `workspace_idx`**（`@tilelang.jit` 装饰器不含该参数）
- **kernel 签名无 `workspace_*` 参数**
- **无 `vid` 二分偏移用于 CV 交互**（`vid` 在本设计中仅用于「一行一任务」的行号分解
  `row = cid*VEC_NUM + vid`，与 CV 交互无关）

这与 guide §3.1.1 的「Developer 模式默认消除 workspace/vid」目标一致 —— 本算子起点即无 workspace。

### 8.3 / 8.4 Cube 核 / Vector 核计算流程

- **Cube 核**：**不使用**（零 `T.gemm_v0`、零 `alloc_L1` / `alloc_fragment` 用于 L0C）
- **Vector 核**：承担全部计算，流程见 §3.3 伪代码与 §4.4 搬运路径

### 8.5 pass_configs 配置

见 §7.3（与同步策略共用同一份配置）。

### 8.6 注意事项

- 本算子**不触碰** `TL_ASCEND_AUTO_CV_SYNC` / `AUTO_CV_COMBINE` 的实质逻辑（无 Cube），
  因此规避了 `examples/mhc_post/benchmark_zh.md` V0→V1 记录的 *"CV 同步 bug 阻止单 kernel 融合"* 风险
  （草稿 §8.1 的观察正确）。
- K1 与 K2 是**两个独立 kernel**，不是单 kernel 内的 CV 融合；顺序由 host 侧调用次序保证。
- 若 Stage 3 考虑把 K1+K2 合并为单 kernel 双遍历（草稿 §5.5 的 V02 项），仍是**纯 Vector 单核内**
  的两遍循环，**不引入 CV 交互**，本章结论不变。

---

## 9. 验证方案

> 本章 §9.2 的 **L0 门槛测试计划**由 `tilelang-op-test-design`（**场景 A：从 design.md**）产出。
> 按 Stage 1 契约，**本阶段只生成 L0**；L1（功能，含不规则/尾块 shape）/ L2（异常）/
> Boundary（特殊值）的完整分层套件由 **Stage 2 在 L0 通过后**调 `tilelang-op-test-design`
> （场景 B：从 `examples/R4/*.py` 真实实现）扩展。**本章不手工枚举 L1/L2/Boundary 用例。**
>
> **验证状态（V01.1）**：本章计划已由 Stage 2 全部落地并在**真实 NPU（Ascend910_9362/A3）验证
> 通过** —— **L0 8/8、L1 18/18、L2 9/9、Boundary 13/13、覆盖门禁 24/24 PASS**；性能哨兵达标
> （K1+K2：B=64→**1.2 ms** / B=256→**3.6 ms**，在 §9.4 预算内）；精度实测余量 ≥ 4 个数量级
> （§9.3）。实现层的 10 项发现（`alloc_var` 重绑定结果错误、标量 `T.exp/T.log` 无法 codegen、
> 1 元素 UB→UB `T.copy` 触发 aicore 异常 507015、`T.If` 需配 `T.Then` 等）与 L0 首轮 3 处失败的
> 根因及修复详见 `examples/R4/debug_log.md`。

### 9.0 算子类别判定（`tilelang-op-test-design` §3.2 流程）

| 判定项 | 结论 | 依据 |
|--------|------|------|
| 计算类型 | **纯 Vector** | §5.1：零 matmul，全部 intrinsic 标 `PIPE_V` |
| 复杂度级别 | **Fusion** | 4 个采样步骤融合 + 跨块分层归并 + 收尾标量算法，多算子组合 |
| 数学特征 | **Reduction + Selection（含 Softmax 子式 `exp+sum+div`）** | §1.3 公式 |
| **综合类别** | **Fusion（纯 Vector 融合算子）** | `COVERAGE_CATEGORY = "Fusion"` |
| 测试重点 | ① 多输出一致性（`m/s/tau/valid/w` 五个输出各自对拍）；② 分块归并正确性（`n_blk` 变化）；③ 尾块归约正确性（V01.1：精确尺寸 `[1,TAIL]` 缓冲 + 全量 reduce，§10.2 V-2）；④ 守卫 `valid` 语义；⑤ 阈值三源（top-k / top-p / min-p）分别绑定与共同绑定 | — |
| 特殊说明 | `w_out` **不做重归一化**（D-NORENORM）⟹ golden 必须同样不归一化，否则整体差一个常数因子 | §3.4 D-NORENORM |

### 9.1 Golden 函数（草案，落在 `examples/R4/r4_ref.py`）

> **为什么独立成 `r4_ref.py`**：参考实现较复杂（需逐行复刻 `sampler.py:846-879` 的 Path 3 语义 +
> 反推 R4 特有的 `tau` / `valid` 两个新输出），独立文件便于审阅与复用；`test_r4_fused_sampling.py`
> 从中 `import`。这是草稿 §9.1 的明确要求（*"必须逐行复刻 …而非重写数学公式——避免参考实现与
> 被测实现共享同一处误解"*）。
>
> **金标准来源已核实**：`/mnt/workspace/cann/20261006_R4_V01/sglang/python/sglang/srt/layers/sampler.py:846-879`
> （Path 3，纯 torch、语义完全明确）；`TOP_K_ALL = 1 << 30` 来自
> `sglang/python/sglang/srt/sampling/sampling_params.py:40`。

```python
# examples/R4/r4_ref.py
# ⚠️ V01.1：本草案的 Stage 1 原版在 tau_p/L 处存在 bug（用 top-k 屏蔽后的 q 求前缀和，
# nucleus>k 时 L 虚增 → ok_p 误判 0 → valid 误判）。Stage 2 已修正为基于 raw probs_sort
# （未屏蔽）的严格前缀和求 tau_p/L，并经 NPU 验证通过（L0 8/8、valid 全 1）；w_ref 的
# sglang 屏蔽链保持不变。下方代码为修正后版本，与 examples/R4/r4_ref.py 一致；
# 详见 debug_log.md（L0 首轮失败信号 1）。
import torch

TOP_K_ALL = 1 << 30          # sglang/srt/sampling/sampling_params.py:40


def golden_r4(logits, top_ks, top_ps, min_ps, need_min_p_sampling,
              K_MAX=256, top_k_all=TOP_K_ALL):
    """R4 融合采样链的 PyTorch 参考实现。

    屏蔽语义逐行复刻 sglang sampler.py:846-879（Path 3），并额外反推 R4 特有的
    tau（定理 T1 合成阈值）与 valid（D-GUARD / D-TIE 降级守卫）。

    全程在 float64 上计算，避免 golden 自身的 fp32 舍入噪声掩盖被测误差。

    Returns: (m_ref, s_ref, tau_ref, valid_ref, w_ref)
             float32 [B], float32 [B], float32 [B], int32 [B], float32 [B,V]
    """
    B, V = logits.shape
    NEG_INF = float("-inf")
    z = logits.detach().to(torch.float64).cpu()
    ks = top_ks.detach().to(torch.int64).cpu().clone()
    ps = top_ps.detach().to(torch.float64).cpu()
    mus = min_ps.detach().to(torch.float64).cpu()

    # ---- Step 2: softmax 统计量（sampler.py:847 torch.softmax 的显式等价式）----
    m_ref = z.max(dim=-1).values                                   # [B]
    e = torch.exp(z - m_ref.unsqueeze(-1))
    s_ref = e.sum(dim=-1)                                          # [B]  = sum exp(z-m)
    probs = e / s_ref.unsqueeze(-1)                                # [B,V] == torch.softmax(z,-1)

    # ---- Step 3: top-k 截断（sampler.py:848,851-856，按秩）----
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)     # :848
    ks_eff = torch.where(ks == top_k_all, torch.full_like(ks, V), ks)   # :851-852
    rank = torch.arange(V, dtype=torch.int64).unsqueeze(0)              # :853-855

    # ---- RAW（top-k 前）严格前缀和（V01.1 修正）：必须在 probs_sort 被屏蔽重绑定前捕获 ----
    # 参考实现是 top-k 之后才做 top-p，但对任意秩 l < k，屏蔽后的严格前缀和 == raw 前缀和
    # ⟹ 定理 T1 的 tau_p 是 RAW nucleus 阈值 z_sort[L_raw]。
    # 若用屏蔽后的 q 求 L：当 nucleus>k（sum(top-k probs) ≤ p）时，被 top-k 置零的尾部
    # 满足 excl_q ≤ p 被误计为 kept → L 虚增到 V-1（实测 152063）→ ok_p 误判 0 →
    # valid 误判（Stage 2 L0 首轮实测 bug，见 debug_log.md）。
    probs_sort_raw = probs_sort
    excl_raw = probs_sort_raw.cumsum(dim=-1) - probs_sort_raw
    kept_raw = excl_raw <= ps.unsqueeze(-1)

    q = probs_sort.masked_fill(rank >= ks_eff.unsqueeze(-1), 0.0)       # :856  (top-k 后、top-p 前)

    # ---- Step 4: top-p(nucleus) 截断（sampler.py:858-860，严格前缀和 > p 才屏蔽）----
    # ⚠️ V01.1：本屏蔽链（q → excl_q → kept_p）**只服务 w_ref**（sglang 语义逐行复刻）；
    # tau_p/L 的反推**必须用** Step 3 的 kept_raw（RAW 前缀和），不能用 kept_p ——
    # kept_p 会把 q=0 的秩 ≥k 尾部也计为 kept，nucleus>k 时 L 虚增（定理 T1 要求 raw 阈值）。
    excl_q = q.cumsum(dim=-1) - q                                  # :858-859 严格前缀和（不含自身）
    kept_p = excl_q <= ps.unsqueeze(-1)                            # :859 判据（严格 > 才屏蔽）
    probs_sort = q.masked_fill(~kept_p, 0.0)                       # :860

    # ---- Step 5: min-p 截断（sampler.py:862-865，严格 < 才屏蔽）----
    if need_min_p_sampling:
        thr = probs_sort[:, 0] * mus                                # :863（rank0 恒保留 ⟹ = max prob）
        probs_sort = probs_sort.masked_fill(
            probs_sort < thr.unsqueeze(-1), 0.0)                    # :864-865

    # ---- Step 6: 不重归一化（D-NORENORM；sampler.py:893-894 multinomial 视其为权重）----
    # 散回原始 vocab 索引域，得到 w_ref（与 kernel 的 w_out 同域，可直接逐元素比）
    w_ref = torch.zeros_like(probs)
    w_ref.scatter_(1, probs_idx, probs_sort)

    # ---- 反推 tau_ref（定理 T1：tau = max(tau_k, tau_p, tau_mu)）----
    # prob 是 z 的严格单调增函数 ⟹ 按 prob 降序 == 按 z 降序（无平票时次序完全一致）
    z_sort = z.sort(dim=-1, descending=True).values                 # [B,V]

    # tau_k：top-k 阈值 = 降序第 k_i 个 logit 值
    k_ok = (ks_eff >= 1) & (ks_eff <= K_MAX)
    idx_k = (ks_eff.clamp(min=1, max=K_MAX) - 1).unsqueeze(-1)
    tau_k = torch.where(k_ok.unsqueeze(-1), z_sort.gather(1, idx_k),
                        torch.full((B, 1), NEG_INF, dtype=torch.float64)).squeeze(-1)

    # tau_mu（定理 T2）：mu<=0 或 !need_min_p → -inf
    mu_active = (mus > 0.0) if need_min_p_sampling else torch.zeros_like(mus, dtype=torch.bool)
    tau_u = torch.where(mu_active, m_ref + torch.log(mus.clamp(min=1e-30)),
                        torch.full((B,), NEG_INF, dtype=torch.float64))

    # tau_p（V01.1）：最后一个满足 **RAW** excl <= p 的秩 L 处的 z 值 —— 与 kernel 在
    # top-K_MAX 列表上的单调守卫标量循环同判据（kernel 用的是 raw 前缀和，非 sglang 屏蔽链）
    L = kept_raw.to(torch.int64).sum(dim=-1) - 1                    # 前缀单调 ⟹ count-1 即末位
    L_safe = L.clamp(min=0, max=V - 1).unsqueeze(-1)
    tau_p = torch.where((L >= 0).unsqueeze(-1), z_sort.gather(1, L_safe),
                        torch.full((B, 1), NEG_INF, dtype=torch.float64)).squeeze(-1)

    tau_ref = torch.maximum(torch.maximum(tau_k, tau_p), tau_u)     # 定理 T1

    # ---- valid_ref（D-GUARD + D-TIE；与 kernel 判据一一对应）----
    # ok_p：nucleus 在 K_MAX 内被确定（L < K_MAX，L 基于 kept_raw），或 p>=1.0（不启用 nucleus）
    ok_p = (ps >= 1.0) | (L < K_MAX)
    # tie_ok：top-p/min-p 严格绑定时值语义精确等价；top-k 绑定时要求无超额有限值平票
    n_gt = (z > tau_k.unsqueeze(-1)).sum(dim=-1)
    n_eq = (z == tau_k.unsqueeze(-1)).sum(dim=-1)
    tie_ok = (tau_ref > tau_k) | ((n_gt + n_eq) <= ks_eff)
    valid_ref = (k_ok & ok_p & tie_ok).to(torch.int32)

    return (m_ref.to(torch.float32), s_ref.to(torch.float32),
            tau_ref.to(torch.float32), valid_ref, w_ref.to(torch.float32))
```

**Golden 自验证记录（Stage 1 CPU 实跑 + Stage 2 修正后 NPU 复验，全部通过）**

| 断言 | 结果 |
|------|------|
| 输出 shape / dtype 与 §4.2 一致（`[B]` fp32 ×3、`[B]` int32、`[B,V]` fp32） | ✅ |
| `m_ref` == `logits.max(-1).values` | ✅ |
| `s_ref` == `sum(exp(z - m))`（≠ `logsumexp(z)`，后者还差 `exp(m)` 因子） | ✅ |
| **定理 T1 验证**：`{j : w_ref[j] > 0}` 与 `{j : z[j] >= tau_ref}` **逐位相同** | ✅ 全部 8 个 L0 配置通过 |
| `w_ref` == `where(z >= tau, softmax(z), 0)`（`rtol=1e-4, atol=1e-6`） | ✅ |
| `sum_j w_ref[j] <= 1`（D-NORENORM，未重归一化） | ✅（实测 0.032~1.0） |
| `valid_ref` 全 1（8 个 L0 配置 × 全部行） | ✅（**仅在 `LOGIT_SCALE=8.0` 下成立**，见 §9.2 数据配方；⚠️ 以 V01.1 修正后的 golden 为准，见下表后注） |
| 每个 L0 配置的实际保留元素数 `<= k_i` | ✅ |

> ⚠️ **V01.1 修正记录（Stage 2 L0 首轮）**：草案原版 `tau_p`/`L` 用 top-k 屏蔽后的 `q` 求前缀和，
> 在 nucleus>k 的配置（`l0_prod_v_tail` valid ratio=0.9375、`l0_full_occupancy` 0.9250）下
> `valid_ref` 非全 1（L 虚增到 152063 → `ok_p=0`）；m/s/tau/w 四项首轮即全过。修正为 **raw
> `probs_sort` 严格前缀和**后 `valid_ref` 全 1，NPU 对拍通过（L0 8/8，valid int32 精确匹配）。
> 详见 `r4_ref.py` 与 `debug_log.md`（L0 首轮失败信号 1）。

**Golden 一致性要求**（`tilelang-op-develop` checklist #9/#10）：

- **逐行对应源码**：Step 3/4/5 的三处 `masked_fill` 判据（`>=` / `>` / `<`）与
  `sampler.py:856 / 859-860 / 864-865` **逐字符一致**，不做「数学等价改写」。
- **`w_ref` 链与 `tau_ref` 链分离（V01.1）**：`w_ref` 走 sglang 屏蔽链（`q → kept_p → min-p`）；
  `tau_p`/`L`/`ok_p` 走 **RAW** 链（`probs_sort_raw → kept_raw`，定理 T1 要求 raw nucleus 阈值）。
  两条链混用（草案原 bug）会在 nucleus>k 时虚增 L 并误判 `valid`。
- **输出形状匹配**：`m_ref/s_ref/tau_ref` = `[B]` fp32、`valid_ref` = `[B]` int32、
  `w_ref` = `[B, V]` fp32，与 §4.2 完全一致。
- **golden 用 float64 计算**，只在返回前降到 fp32 ⟹ golden 自身舍入噪声 ≪ 被测容差。
- **L0 用例输入为 `randn × LOGIT_SCALE` 的连续值** ⟹ 无精确平票、无 `-inf`，`z_sort` 与
  `probs_sort` 次序一致，`tau_ref` 的秩域反推无歧义。退化输入（平票 / `-inf` / 全等行）由
  Stage 2 的 Boundary 套件覆盖，届时 `tau_ref` 需改用「保留集最小 z 值」重构法（Stage 2 事项）。

### 9.2 L0 门槛测试计划（`tilelang-op-test-design` 场景 A 产出）

> **L0 定位**：规则 shape、快速冒烟、供 Stage 2 精度收敛。**≤10 用例**（本计划 8 个）。
> 全部 fp32（V01-a 唯一支持的计算 dtype）。所有用例的 `valid_ref` 均为**全 1**（守卫不触发）。
> 所有用例的 `(K_MAX, BV)` 均为 S1' 特化 `(256, 4096)`；`l0_01~l0_04/07/08` 用**小 V 加速冒烟**
> （`V` 是编译期常量，小 V ⟹ `n_blk` 小 ⟹ 编译与运行都快）。

#### 9.2.1 🔴 测试数据配方（**强制**，否则 L0 会因守卫误触发而全批失败）

```python
torch.manual_seed(0)                 # 确定性
LOGIT_SCALE = 8.0                    # ⚠️ 强制值，见下方实测论证
logits = torch.randn(B, V, dtype=torch.float32) * LOGIT_SCALE
top_ks = torch.full((B,), k,   dtype=torch.int32)
top_ps = torch.full((B,), p,   dtype=torch.float32)
min_ps = torch.full((B,), mu,  dtype=torch.float32)
```

**为什么必须 `LOGIT_SCALE = 8.0`（本阶段 CPU 实测，非估算）**

`K_MAX = 256` 意味着 **top-p 的 nucleus 必须能装进 256 个候选**，否则 D-GUARD 的 `ok_p = 0`
⟹ `valid = 0` ⟹ 整批回退，L0 无法验证快路径。而 nucleus 规模强依赖 logits 的**尖锐度**：
`randn` 原样（scale=1）的 softmax 近乎均匀分布，nucleus 会覆盖大半个 vocab。

`p = 0.95` 下、64 行样本的实测 **nucleus 最大规模**（必须 `< K_MAX = 256`）：

| `LOGIT_SCALE` | V=4096 | V=16384 | V=152064 | 判定 |
|--------------|--------|---------|----------|------|
| 1.0（裸 `randn`） | 3,067 | 12,201 | 112,793 | ❌ 守卫全触发 |
| 3.0 | 642 | 1,989 | 15,735 | ❌ |
| 5.0 | 82 | 150 | **844** | ❌（V=152064 超限） |
| 6.0 | 56 | 70 | **265** | ❌（V=152064 刚好超限） |
| 7.0 | — | — | — | ❌（在 §9.2.2 的 8 用例矩阵上仍有行 `valid=0`） |
| **8.0** | **22** | **27** | **69** | ✅ **全部 `< 256`，余量 ≥ 3.7x** |
| 10.0 | 17 | 19 | 28 | ✅（但分布过尖，`kept` 恒为 1~2，判别力弱） |

**在 §9.2.2 的完整 8 用例矩阵上逐 scale 实跑 `golden_r4` 的结果**：

| `LOGIT_SCALE` | 全部行 `valid == 1` | `p<1.0` 用例的 `kept` 范围 |
|--------------|--------------------|---------------------------|
| 5.0 | ❌ False | topp 2..40, minp 1..12, prod 3..23, full 1..15 |
| 6.0 | ❌ False | topp 1..40, minp 1..8, prod 2..17, full 1..12 |
| 7.0 | ❌ False | topp 1..26, minp 1..8, prod 1..12, full 1..10 |
| **8.0** | ✅ **True** | **topp 1..18, minp 1..7, prod 1..12, full 1..9** |

⟹ **`LOGIT_SCALE = 8.0` 是同时满足「守卫不触发」与「nucleus 有足够多样性（1~18 个元素，
使 top-p 前缀和循环真正迭代多次）」的唯一取值。**

> ⚠️ **Stage 2 注意**：若改用其他 scale 或其他随机分布，**必须先跑 `golden_r4` 确认
> `valid_ref` 全 1**，否则 L0 会因守卫触发而「假失败」，容易误判为 kernel bug 甚至误报
> `[DESIGN_ERROR]`。这一条是 L0 计划的一部分，不可省略。
>
> ⚠️ **同时记为设计适用性约束（§10.1 R-KMAX）**：`K_MAX=256` 只在「真实 nucleus ≤ 256」时
> 才能走快路径。真实 LLM 的 post-LM-head logits 远比 `randn` 尖锐（温度缩放后更是如此），
> 该条件在生产上通常成立；但对**高温 + 平坦分布 + `top_k = TOP_K_ALL`** 的组合，守卫会频繁触发
> ⟹ 回退 torch（性能不劣于现状 Path 3，草稿 D8 结论正确）。

#### 9.2.2 L0 用例矩阵（8 个，已用 `golden_r4` 在 CPU 上逐例实跑验证）

**通用配置**：`dtype = float32`；`K_MAX = 256`；`BV = 4096`；`VEC_NUM = 2`；
`LOGIT_SCALE = 8.0`（§9.2.1）；`torch.manual_seed(0)`。

| 用例名 | 级别 | B | V | 满块数 / `TAIL` | k | p | μ | `NEED_MIN_P` | 实测绑定阈值 | 覆盖目标 |
|--------|------|---|---|-----------------|---|---|---|--------------|-------------|---------|
| `l0_min_single_block` | L0 | 2 | 4096 | 1 / 0 | 40 | 1.0 | 0.0 | `False` | **k** | **最小功能验证**：单块（无跨块归并）、无尾块、仅 top-k；`b_num=1`；`kept=40` |
| `l0_multiblock_topk` | L0 | 8 | 16384 | 4 / 0 | 40 | 1.0 | 0.0 | `False` | **k** | **跨块分层归并**（4 块）、block 整除无尾块、仅 top-k；`kept=40` |
| `l0_topp_nucleus` | L0 | 8 | 16384 | 4 / 0 | 40 | 0.95 | 0.0 | `False` | **p** | **top-p 前缀和路径**（标量循环 + 单调守卫 + `ok_p`）；`kept=1..18` |
| `l0_minp_T2` | L0 | 8 | 16384 | 4 / 0 | 40 | 0.95 | 0.05 | `True` | **p + μ** | **定理 T2 的 `ln μ` 路径** + 三阈值 `max()` 合成；`kept=1..7` |
| `l0_prod_v_tail` | L0 | 16 | **152064** | 37 / **512** | 40 | 0.95 | 0.05 | `True` | **p + μ** | **代表性生产 V + 静态尾块**（`real_shape=[1,512]`、topk `actual_num=512`）⟹ C1/C2/N1/N2/N3 全覆盖；`kept=1..12` |
| `l0_full_occupancy` | L0 | 40 | **152064** | 37 / 512 | 20 | 0.9 | 0.1 | `True` | **p + μ** | **满核占用**（40 vec 核 = 20 cube × VEC_NUM 2，`b_num=20`）；`μ=0.1` 使 `tau_u` 更易绑定；最大 DMA 量；`kept=1..9` |
| `l0_k_eq_1` | L0 | 2 | 16384 | 4 / 0 | **1** | 1.0 | 0.0 | `False` | **k** | **k 下边界**：`tau_k = run_v[0] = m`（等价 argmax），`n_gt=0, n_eq=1`；`kept=1` |
| `l0_k_eq_kmax` | L0 | 2 | 16384 | 4 / 0 | **256** | 1.0 | 0.0 | `False` | **k** | **k 上边界 = K_MAX**：归并列表全用满，`tau_k = run_v[255]`，`tie_ok` 的 `< K_MAX` 边界；`kept=256` |

**矩阵覆盖自检**：三个阈值源（k / p / μ）各自绑定至少 1 例 ✅；`NEED_MIN_P` 两取值各 ≥3 例 ✅；
单块 / 多块 / 含尾块三种分块形态齐备 ✅；`k` 的下边界（1）与上边界（256）齐备 ✅；
`B` 覆盖 2 / 8 / 16 / 40（含 `B < 40` 的 grid 收缩与 `B = 40` 的满核）✅；
`V` 覆盖 4096 / 16384 / 152064 ✅。

**每个 L0 用例的验证项（5 个输出全部对拍，缺一不可）**

| 输出 | dtype | 比对方式 | 期望 |
|------|-------|---------|------|
| `m_out` | float32 | `check_precision(·, m_ref, "float32")` | max 无算术 ⟹ 实际应**逐位相同**；仍按 fp32 混合容差判 |
| `s_out` | float32 | 同上 | fp32 分块累加 + 重标定 vs golden float64 求和，相对误差 ~1e-6 |
| `tau_out` | float32 | 同上 | `tau_k`/`tau_p` 是输入 logit 的原值传递（排序/归并保值）⟹ 期望逐位相同；`tau_u = m + ln μ` 分支有 ~1e-7 相对舍入 |
| `valid` | int32 | `check_precision(·, valid_ref, "int32")`（0/0/0/1.0 精确匹配） | **全 1**（§9.2.1 保证） |
| `w_out` | float32 | 同上 | 主体判据；`required_matched_ratio = 0.99` |

**L0 通过判据（Stage 2 精度收敛门禁）**

```
所有 8 个用例 × 5 个输出 = 40 项比对，全部满足：
    matched_ratio >= required_matched_ratio(dtype)  AND  max_abs_error <= max_abs_error_limit(dtype)
且 valid 逐元素精确相等（int32，0 误差），且 valid_ref 全 1（否则先查 §9.2.1 数据配方，不是 kernel bug）
⟹ 打印 [PRECISION_PASS]，测试进程 exit(0)
任一失败 ⟹ [PRECISION_FAIL]，exit(1)，Stage 2 进入精度调试
          （仅当定位为设计层缺陷时才返回 [DESIGN_ERROR]）
```

**L0 明确不含（留给 Stage 2 的 `tilelang-op-test-design` 场景 B）**：不规则/质数 V、`B=1`、
`B=1024`、`k ∉ [1,K_MAX]`、`TOP_K_ALL = 1<<30`、`p` 极小（触发 `ok_p=0`）、全等行（触发
`tie_ok=0`）、含 `-inf` 输入、含 NaN 输入、bf16（不支持，属 L2 负向）、空 tensor（`B=0`，属 L2 负向）、
`LOGIT_SCALE ≠ 8.0` 的平坦分布（会触发守卫，属 L2/Boundary 的**负向守卫验证**）。

### 9.3 精度标准

> 采用**混合容差**：逐元素 `|actual-golden| <= atol + rtol·|golden|`，整体判定
> `matched_ratio >= required_matched_ratio` **且** `max_abs_error <= max_abs_error_limit`。
> 阈值**仅按 dtype**（与算子类别无关）。完整定义见
> `tilelang-op-test-design/references/precision-standard.md`。

| dtype | atol | rtol | max_abs_error_limit | required_matched_ratio | 本算子适用的输出 |
|-------|------|------|---------------------|------------------------|-----------------|
| **float32** | 2⁻¹⁶ = 1.53e-5 | 2⁻¹⁰ = 9.77e-4 | 1e-2 | 0.99 | `m_out`, `s_out`, `tau_out`, `w_out`, 以及输入 `logits`/`top_ps`/`min_ps` |
| **int32** | 0 | 0 | 0 | 1.0（精确匹配） | `valid`（输出）, `top_ks`（输入） |

**§4 声明的每个 dtype 均已在上表有对应行**（float32 与 int32，无其他 dtype —— bf16 属 V01-a 范围外）。

> **特殊浮点值的位置契约**：本算子的输入**允许含 `-inf`**（上游屏蔽产生，草稿 B10），
> **不允许含 NaN**（上游 `sanitize_nan_logits` 已清洗，kernel 不做二次防护 ⟹ NaN 属 L2 负向域）。
> 因此 Stage 2 扩展 Boundary 套件时必须遵守：
> - 测试输入使用**「有限值 + 稀疏 `-inf`」混合的确定性数据**，保证至少存在一个 `-inf` 和一个有限值；
>   **禁止**用全 `-inf` 行作为唯一特殊值用例（会掩盖归并/索引错误）；
> - 比较时先分别严格检查 `isnan`、正 `Inf`、负 `Inf` 的 mask 完全一致，
>   再**仅对双方均为有限值的位置**应用上表的混合容差；
> - `allclose(equal_nan=True)` **不能**替代显式 mask 检查；
> - `w_out` 中被屏蔽位置的 golden 值恒为 `0.0`（有限），而 `-inf` 只出现在 `tau_out`
>   （当 `k_i ∉ [1,K_MAX]` 且无 top-p/min-p 绑定时）⟹ `tau_out` 的 `-inf` mask 必须精确一致。

**逐输出的精度分析（为什么上表阈值足够）**

| 输出 | 误差来源 | 量级估计 | 是否被 fp32 阈值覆盖 |
|------|---------|---------|---------------------|
| `m_out` | 纯比较，无算术 | **0**（逐位相同） | ✅ 远超要求 |
| `s_out` | 38 块 × 4096 项 fp32 分块累加 + 每块重标定 `s*exp(m_old-m_new)` | 相对误差 ~1e-6（`s` 量级 10~1e3） | ✅ `rtol=9.77e-4` 有 ~1000x 余量 |
| `tau_out` | 排序/归并**保值**（无算术）；`tau_p` 亦为 `run_v[l]` 原值 | **0**（逐位相同），除非 `tau = tau_u = m + ln(μ)` 分支 | ✅ `tau_u` 分支：`ln` 相对误差 ~1e-7，`atol+rtol·|tau|` 覆盖（`|tau|` 量级 ~10） |
| `valid` | 整数判定 | 0 或 1（离散） | ✅ int32 精确匹配；判据边界（`prefix > p` 在 `p→1` 时）见 §10.1 R-P |
| `w_out` | ① `exp(z-m)` 舍入 ~1e-7 相对；② `/s` 舍入；③ `s` 的 ~1e-6 相对误差传递；④ **`tau_u` 域变换差异**：参考在 prob 域比 `probs < (1/s)·μ`，本设计在 log 域比 `z < m + ln μ`，两者数学等价但 fp32 舍入差 ~1e-7 相对 ⟹ **阈值边界处的单个元素可能进/出保留集** | 主体 ~1e-6 相对；边界元素为「0 vs 一个有限概率」的**离散跳变** | ✅ 主体被 `rtol=9.77e-4` 覆盖；边界跳变元素数量极少（`randn` 连续分布下落在 1e-7 相对窗口的期望个数 ≈ `V × 1e-7 ≈ 0.015` 个/行）⟹ 远低于 `1 - 0.99 = 1%` 的不匹配预算，且 `max_abs_error` 受单个概率值上界 `1/s ≪ 1e-2` 约束 |

### 9.4 性能可行性哨兵（强制执行，不可因 large/L1 跳过）

| 用例名 | Shape | dtype/属性 | 覆盖路径 | 单 case timeout | 选择理由 |
|--------|-------|-----------|---------|-----------------|----------|
| `perf_worst_k1k2` | `B=64, V=152064` | float32；`K_MAX=256, BV=4096`；`k=40, p=0.95, μ=0.1, NEED_MIN_P=True`；**`LOGIT_SCALE=8.0`**（§9.2.1，否则守卫触发使测的是回退路径而非 kernel） | K1（37 满块 + 1 尾块，每块 2 topk + 3 reduce）+ K2（38 块 elementwise）+ host `multinomial`；**最大 DMA 任务数（≈4864 次）与最大 tmp arena（64 KB）同时命中** | **900 s**（含首次 JIT：bisheng dav-2201 编译两个 kernel，秒~分钟级） | 用户关键 case：接近满核（64 行 / 40 vec 核 → 2 波）+ 生产 V + 三阈值全绑定；`B=64` 亦为草稿 §12 迭代一的 ST 标准用例 |
| `perf_max_batch` | `B=256, V=152064` | float32；同上属性（`LOGIT_SCALE=8.0`） | 同上，**最大 GM 流量**（K1+K2 ≈ 594 MB，§4.4）与 `b_num=128` 超订调度 | **1200 s** | 最大任务数 case（§5.5 `[REORDER-COST]` 的 `active_cores/serial_tasks_per_core` 上界：每核 `ceil(256/40)=7` 行） |

**执行约束**（`design-template.md §9.4`）：

- 测试数据、随机数、特殊值和 golden 的排序/掩码/scatter 重排**均在 CPU 完成**（`golden_r4` 内已
  `.cpu()` + float64）；运行阶段只做 **H2D → TileLang kernel（K1, K2）→ D2H**，
  避免测试 harness 引入额外 aclnn 依赖。
- **例外**：`torch.multinomial(w_out, 1)` 在 NPU 上执行（V01-a 用户明确保留的下游算子，§5.6 DEV-1）。
  L0/哨兵的**精度判定不依赖它**（只对拍 K1/K2 的 5 个输出），故其存在不影响精度门禁；
  仅作为端到端连通性冒烟（确认 `w_out` 可被 multinomial 接受，即非负、不全零）。
- 哨兵用例**必须实际执行并计时**，不得以「large/L1」为由跳过；超时即视为失败。
- 首次运行的 JIT 编译耗时须单独记录并纳入 warmup 预算（草稿 §9.4 / B15）。

---

## 10. 风险点与注意事项

### 10.1 已知约束与风险

| # | 风险 | 等级 | 影响 | 应对 / 证据 |
|---|------|------|------|-----------|
| **R-A** | **术语张力：`T.tile.topk` / `T.tile.gather_mask` 无纯 Developer 高层等价物**。TileLang-Ascend 全仓（`tilelang/language/`、`examples/`、`testing/`）**不存在**任何 `alloc_shared`+`T.Parallel` 形态的 top-k / sort / argsort 原语 | 🟡 **中（需 Orchestrator/用户知悉，非阻塞）** | 按 `tilelang-programming-model-guide` §1 的严格分类，本设计属「**混合模式**」（Developer 主体 + Ascend 专属 `T.tile.xxx`），而非「纯 Developer」 | **不视为设计失败**，依据三条：① guide 明言「大多数实际算子使用混合模式」，且混合模式定义即「使用 Developer 的 pass_configs，不写 `T.Scope` 和手动同步」——本设计完全符合；② 仓库**自己的 Developer 标杆目录** `examples/developer_mode/gelu_mul_developer.py` 就在 `alloc_shared` + Developer pass_configs 下使用 `T.tile.*`；③ 本设计**未使用任何 Expert 判别标志**（`alloc_ub` / `T.Scope` / 手动 flag / pass_configs 全关）。**已按用户要求保留 Developer 定性，未擅自改回 Expert。** 若用户坚持「零 `T.tile.*`」，则 top-k 分层归并**在纯 Developer 模式下无法实现**，需用户重新决策（详见 §10.5 结论） |
| **R-B** | **`T.tile.topk` + `T.alloc_shared` 的组合无现成用例**：仓库 6 处 topk 用例（`topk_selector`、`moe_topk_gate` ×2、`lightning_indexer` ×2、`test_..._topk.py`）**全部用 `T.alloc_ub`** | 🟡 中 | 若 scope 推导在某种边界下把 buffer 判成 `shared.l1`，topk 会失败 | 已从源码证明可行（§2.3 论据 2 完整链条：`operation_config.h:248` `PIPE_V` → `IsVectorFunction` → `ascend_infer_buffer_scope.cc:398-402` `shared`→`shared.ub`）。**上板验证项 V-1（§10.2）—— V01.1：Stage 2 实测 ✅ PASS（两组配置 accuracy=1.00000），兜底未触发，风险关闭**。**兜底方案**：仅把 `zt_ub` / `l_dst` / `l_val` / `merged` / `m_dst` / `run_v` 这 6 个 topk 相关 buffer 改回 `T.alloc_ub`（`alloc_ub` 与 `alloc_shared` 落到同一物理 UB，`allocate.py:156-157` vs `:31-46`），**其余保持 `alloc_shared`**，仍不引入 `T.Scope`/手动同步 ⟹ 不改变 Developer 定性，属最小局部例外 |
| **R-C** | **`T.tile.topk` 的 `max_actual_num = Σ src.shape`（求和非求积）**，2D src 会多算 | 🟡 中 | 若把 `[1,BV]` 直接喂 topk ⟹ `max_actual_num=4097` ⟹ `repeatTimes=129` ⟹ 排序覆盖 4128 槽位 > 4096，**可能读越界并污染 top-K** | **已在设计层消解**：N2 规定用专用 rank-1 `zt_ub[BV]`（`max_actual_num = 4096` 精确）。证据 `ascend_tile.py:578-590` + `test_..._topk.py:86-89,349` |
| **R-D** | **归约的尾块有效宽度不能靠 `pad_value=-inf`** | 🟢 已关闭（V01.1） | 依赖 `-inf` pad 会在 NPU 上产生 inf/nan（**仓库实测结论**） | **已消解（V01.1 修正）**：原 `real_shape=[1, valid]` 方案经 V-2 实测对尾块 `valid=512` **不可用**（`codegen_ascend.cc:2145`：narrow-reduce 逻辑宽度 ≤256 B = 64 fp32，BufferRegion 兜底同为 narrow 路径亦失败）⟹ 改用**精确尺寸 `[1,TAIL]` 缓冲 + 全量 reduce**，NPU 验证通过（§10.2 V-2、§5.3 N1）。证据 `test_..._tail_block.py:45-51`、`debug_log.md` V-2 |
| **R-E** | **fp32 topk 的自动 tmp arena = `16 × AlignUp(BV,32)` B，与 BV 线性相关**（草稿完全遗漏，低估约 3 倍） | 🟡 中 | UB 预算若按草稿的 41 KB 规划，实际会到 121 KB；若把 BV 提到 8192，K1 将达 ≈**229 KB ≫ 184 KB** ⟹ **编译期 UB 溢出** | **已核算并给出闭式约束**：§4.5 完整预算表 + 通式 **`28·BV + 36·K_MAX + 72 ≤ 188416`**（推导见 §4.5）⟹ `K_MAX=256` 时 **`BV ≤ 6400`**。草稿的 S1（`BV=8192`）与 S2'（`K_MAX=1024, BV=8192`）**在本设计的预算模型下不可行**（分别 ≈229 KB / ≈266 KB），若未来要开这些特化必须把 `BV` 降到 ≤4096。V01-a 固定 `BV=4096, K_MAX=256` → 121 KB ✅ |
| **R-P** | **`tau_p` 的浮点边界判定**：`ok_p` 依赖 `prefix > p_i`，`prefix` 是 256 项 fp32 累加 | 🟢 低 | `p_i` 极接近 1.0（如 0.9999999）时，`prefix` 的 ~1e-6 相对误差可能让 `ok_p` 误判 ⟹ 保守地 `valid=0` 回退（**不影响正确性，只影响是否走快路径**） | 误判方向是**保守**的（宁可回退）。L0 用 `p ∈ {0.9, 0.95, 1.0}` 远离边界。`p_i = 1.0` 已由 `ok_p` 的显式特判处理 ⟹ 不受影响 |
| **R-TIE** | **有限值超额平票时 `valid=0` 整批回退**（D-TIE 的 correct-by-fallback） | 🟢 低 | 退化输入（全等行、量化 logits）会走 torch 慢路径 | 与现状一致（此类输入本就多走 Path 3）；`-inf` 平票已论证**无害**（`w=0`）不触发回退；L0 用连续 `randn` 不触发。Boundary 套件（Stage 2）验证守卫确实触发 |
| **R-KMAX** | ⭐ **`K_MAX=256` 的适用性依赖「真实 nucleus ≤ 256」**：top-p 阈值只在 top-`K_MAX` 有序列表上求前缀和，若真实 nucleus 更大则 `ok_p=0` ⟹ 整批回退。**本阶段 CPU 实测量化了该边界**（§9.2.1）：`p=0.95` 下，裸 `randn`（scale=1）在 `V=152064` 的 nucleus 达 **112,793**，scale=5 仍有 **844**，scale=6 为 **265**，**只有 scale ≥ 8 才降到 69** | 🟡 **中** | ① **L0 测试数据必须用 `LOGIT_SCALE=8.0`**，否则守卫全触发、L0「假失败」（极易被误判为 kernel bug 甚至误报 `[DESIGN_ERROR]`）；② 生产上对「高温 + 平坦分布 + `top_k=TOP_K_ALL`」组合，快路径命中率低 | ① §9.2.1 已把 `LOGIT_SCALE=8.0` 定为**强制测试数据配方**并附实测论证表；② 守卫回退**不劣于现状**（此类输入本就 `npu_top_k_top_p_eligible=False` → 走 Path 3，草稿 D8 结论正确）；③ 真实 LLM 的 post-LM-head logits 远比 `randn` 尖锐，且生产常配合温度缩放，nucleus 通常在数十~数百量级 ⟹ S1' 对主流量适用；④ 若 Stage 3 需要覆盖平坦分布，可开 `K_MAX=1024` 特化（§4.5 通式给出 `BV=4096, K_MAX=1024` → **148.1 KB ≤ 184 KB** 可行；但 `BV=8192` 组合不可行，见 R-E） |
| **R-JIT** | **`B` 无法移出 JIT 特化键**（对草稿 §3.4 的修正） | 🟡 中 | 连续批处理下 `B` 频繁变化 ⟹ 重编译风暴（bisheng + TVM lower，远重于 Triton） | 唯一消除手段需 host padding，**被 `ascend-constraints.md §5` 明令禁止**（§4.6 R-JIT 完整论证）。缓解：`TILELANG_CACHE_DIR` 持久化 + **生产禁用 `disable_cache()`/`clear_cache()`** + decode 的 `B` 由 scheduler 量化到有限集合。V01-a 单特化 + 少量 L0 用例 ⟹ 编译次数可控 |
| **R-NPU** | ~~本阶段无 NPU 实机验证~~ **已关闭（V01.1）**：Stage 2 已在真实 NPU（Ascend910_9362/A3）完成全链路验证 | 🟢 已关闭 | 全部数值结论由「待上板验证」升级为 **NPU 实测通过** | §10.2 的 V-1~V-4 已全部执行；**L0 8/8、L1 18/18、L2 9/9、Boundary 13/13、覆盖门禁 24/24 PASS**；性能哨兵 K1+K2 B=64→**1.2 ms** / B=256→**3.6 ms**（§9.4 预算内）；精度实测余量 ≥ 4 个数量级。实现层 10 项发现（`alloc_var` 重绑定结果错误、标量 `T.exp/T.log` 无法 codegen、1 元素 UB→UB `T.copy` 触发 aicore 异常 507015、`T.If` 需配 `T.Then` 等）详见 `examples/R4/debug_log.md` |
| **R-DEPR** | **草稿依赖的 `wholereducemax`/`wholereducesum` 已 deprecated** | 🟢 低（已消解） | 若照抄草稿会产生 DeprecationWarning 且未来版本可能移除 | 本设计已全部改用 `T.reduce_max`/`T.reduce_sum`（§3.2）。**顺带消解草稿 R10（`gather_mask` pattern 未知）与 R11（`wholereducemax` 签名复杂）两项风险**：`"P0101"`=偶数位=值 已由 wheel docstring + `test_..._topk.py:169-170` 双重确认 |
| **R-F64** | **`float64` 零支持** ⟹ 确定性 seed 路径不可实现 | 🟢 低（本范围外） | — | V01-a 明确排除（用户 scope）。草稿 D3/R1 结论正确，本设计沿用 |
| **R-CUMSUM** | **`T.cumsum` Ascend 不可用** | 🟢 低（已消解） | top-p 前缀和无原语 | 分层归并把规模从 `V=152064` 降到 `K_MAX=256` ⟹ UB 标量循环 + 单调守卫（§6.4）。草稿 D2 结论正确，本设计沿用并补上「无需 break」的单调性论证 |

### 10.2 Stage 2 上板验证项（按优先级，实现前必跑；**V01.1：Stage 2 已全部执行完毕**，结果随各行标注，详见 `debug_log.md`）

| # | 验证项 | 方法 | 若不通过的兜底 |
|---|--------|------|--------------|
| **V-1** | `T.tile.topk` 作用于 `T.alloc_shared` buffer（R-B）—— **结果 ✅ PASS**（两组配置 accuracy=1.00000，兜底未触发） | 最小复现：把 `examples/topk_selector/example_topk_selector.py` 的 5 个 `T.alloc_ub` 全改 `T.alloc_shared`，跑 `python example_topk_selector.py`（B=64,N=4096,top_k=2048 + B=1024,N=1024,top_k=128），要求 `accuracy > 0.99` | 仅 topk 相关 6 个 buffer 改回 `alloc_ub`（R-B 兜底），不引入 `T.Scope`/手动同步 —— **未触发** |
| **V-2** | 尾块归约 `real_shape=[1, valid]` 在 rank-2 `[1,BV]` + `alloc_shared` 下正确（N1/R-D）—— **结果 ❌ 主方案与文档兜底均失败**（发现新硬约束），**实际解法已验证 ✅** | 最小复现：仿 `test_..._narrow_reduce.py`，`M=1, buf_n=4096, valid=512` + `alloc_shared`，与 CPU `window.max()/sum()` 对拍。**实测**：`InternalError: Check failed: (nn * elem_bytes <= 256) ... got 512 elements of 4 bytes` —— `src/target/codegen_ascend.cc:2145` 硬约束：**narrow-reduce 路径（`real_shape` 窄于 buffer，或列切片 BufferRegion）要求逻辑宽度 ≤ 256 B = 64 个 fp32 元素**；R4 尾块 `valid=512`（2048 B）远超此限 | ~~BufferRegion 形式 `T.reduce_max(z_ub[:, 0:valid], m_blk, dim=-1)`~~ —— **同为 narrow 路径，同样触发 256 B 约束，不可用**（`narrow_reduce.py:68` 测试通过仅因其宽度低于限值）。**实际解法（✅ NPU 验证）**：尾块分配**精确尺寸 `[1,TAIL]` 缓冲**（`zt/et/mt/ztt`，`T.copy` gm2ub 自动 clamp）+ **全量 reduce（不传 `real_shape`）**，走 normal reduce 路径；满块 `[1,BV]` 亦全量 reduce。V=152064、BV=4096（37 满块 + 512 尾块）的 online-softmax `m/s` 与 CPU 对拍正确（m 逐位相同、s rtol<1e-3） |
| **V-3** | UB 动态下标标量读 `run_v[k_i[0] - 1]`（`k_i` 来自 GM）—— **结果 ✅ PASS**（`ks=[1,2,40,128,256,257,0,100]` 与期望逐位相同，含越界 257/0 → `-inf` 守卫正确） | 最小复现：仿 `example_cross_entro.py:113-115` 写一个「用 GM int32 值索引 UB fp32 数组」的 kernel | 改用向量化 select-sum：`sum_l run_v[l]·1[l == k_i-1]`（`T.Parallel` + `T.reduce_sum`，§3.2 已列替代方案）—— **未触发**（该替代亦验证可用，作备份） |
| **V-4** | 端到端最小闭环：`B=2, V=4096, k=40, p=1.0, μ=0` 的 K1+K2，`m/s/tau/valid/w` 全部对拍 golden（即 `l0_min_single_block`）—— **结果 ✅ 最终 L0 8/8 通过**（首轮 3 处 valid 失败：根因为 §9.1 golden 草案 `tau_p/L` bug、§3.4 D-TIE 公式 `k_i==K_MAX` 自相矛盾、`V<BV` 编译期死 buffer —— 均实现层修复并回写本文档 V01.1，见 `debug_log.md`） | 直接跑 L0 第一个用例 | 按 `tilelang-error-fixer` / `tilelang-debug-helper` 流程分段定位（`T.dump_tensor` 导出 `run_v` 中间态） |

### 10.3 常见错误与已知偏差

| 错误 / 偏差 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| **UB 溢出** | 把 `BV` 提到 8192（草稿 S1/S2'） | 编译期失败或 segfault（K1 ≈250 KB > 184 KB） | 严守 `BV ≤ 6729`（R-E 公式）；V01-a 固定 4096 |
| **topk 结果被污染** | 把 rank-2 `[1,BV]` 直接喂 `T.tile.topk` | top-K 值错误（越界读 32 槽位） | N2：专用 rank-1 `zt_ub` |
| **尾块归约出 inf/nan** | 依赖 `pad_value=-inf` 让 reduce 读到 pad 区 | `m`/`s` 错误 ⟹ 全行 `w` 错误 | N1（V01.1）：尾块**精确尺寸 `[1,TAIL]` 缓冲 + 全量 reduce**（`T.copy` gm2ub 自动 clamp），不依赖 `pad_value`；满块全量 reduce |
| **narrow-reduce 编译失败：`InternalError: Check failed: (nn * elem_bytes <= 256)`** | `T.reduce_max/sum` 传窄于 buffer 的 `real_shape`、或列切片 BufferRegion，且逻辑宽度 > **256 B（= 64 个 fp32）**（如尾块 `valid=512` → "got 512 elements of 4 bytes"） | 编译期 InternalError，**主方案与 BufferRegion 兜底均不可用** | `codegen_ascend.cc:2145` 硬约束（V-2 实测）；改用**精确尺寸缓冲 + 全量 reduce（不传 `real_shape`）**（§10.2 V-2，NPU 验证）。narrow 宽度 ≤ 64 fp32 时仍可用 |
| **编译期 "Cannot find pre-allocated address for buffer"** | `V < BV`（`n_blk_full=0`）时 `T.serial(0)` 消除循环体，无条件分配的 `[1,BV]` buffer 成为死 buffer（`codegen_ascend.cc:828`，`l1_edge_V_eq_KMAX` V=256 实测） | 编译失败 | 满块 buffer 分配移入**编译期 `if n_blk_full > 0:` 守卫**（与循环同域）；纯尾块路径走独立精确尺寸缓冲；V=256/512 NPU 验证通过（`debug_log.md` L0 首轮失败信号 3） |
| **首轮归并结果不确定** | 忘记 `T.tile.fill(run_v, -inf)` | 未初始化 UB 参与排序 | N4 |
| **`logits` 被污染** | 把 GM 视图直接喂 `T.tile.topk`（fp32 in-place 写 `-inf`） | 上游 `log_softmax` 等后续消费者读到 `-inf` | N3：只对 UB scratch `zt_ub` 调 topk（草稿 R5 的正确规避） |
| **守卫永不触发** | 照抄草稿 `valid = (L >= 0)` | nucleus > K_MAX 时 `tau_p` 被低估 ⟹ 分布错误且静默 | D-GUARD：`valid = k_ok AND ok_p AND tie_ok` |
| **`TOP_K_ALL` 判定错误** | 照抄草稿的 `-1` | `k_i = 1<<30` 被当成合法值 ⟹ `run_v[1<<30 - 1]` 越界读 | 用 `1 << 30`（`sampling_params.py:40`），并由 `k_ok` 拦截 |
| **核数硬编码 48** | 照抄草稿 | Ascend910_9362 实测 40 ⟹ grid 超订比例错算 | 运行时 `detect_vec_core_num()`（`utils.py:20-40`） |
| **DEV-1：host 侧 aclnn** | `torch.multinomial(w_out, 1)` | 与 cann-bench「host 零 aclnn」提交规范不兼容 | **用户明确保留的 V01-a 下游算子**（非 R4 计算的 host 替代），且落点是 `examples/R4/` 独立算子而非 cann-bench 提交 ⟹ 记为已知偏差，不阻塞。消除它属 V01-b（Gumbel-max 融合，out of scope） |
| **DEV-2：`T.Kernel` 用 `as (cid, vid)` 而非 `threads=2 as (cid)`** | Developer 模式 CV 推荐写法 | 与 guide §3.1.1 表述不完全一致 | §6.5 给出 4 条硬约束理由（topk 需整行驻留单核 UB、topk 无 per-row 语义、纯 Vector 无 workspace 可消、仓库纯 Vector 生产实现一致形态）。**不构成 Expert 回退** |

### 10.4 对草稿 `DESIGN_script.md` 的修正清单（本设计已改，Stage 2 以本文档为准）

| # | 草稿位置 | 草稿结论 | 修正后 | 依据 |
|---|---------|---------|-------|------|
| 1 | 表头 / §1.1 / §13.4 | 目标芯片 Ascend910_**9382**，Vector 核 **48**（Cube 24） | Ascend910_**9362**（实测），Vector 核 **40**（Cube 20），**运行时探测** | 二者同属 `910_93` ⟹ `determine_platform()` 均判 A3（`target.py:145-147`），`--npu-arch=dav-2201` 不变（`libgen.py:152-155`）⟹ **arch 结论不变**，仅核数需实测 |
| 2 | §3.2 / §7 B5 | `TOP_K_ALL` = **-1** | **`1 << 30`** | `sglang/.../sampling_params.py:40` |
| 3 | §5.3.3 / §5.3.4 | `T.tile.wholereducemax` / `wholereducesum` 可用（标 ✅ 已验证） | 二者在已安装 wheel 中 **`@deprecated()`** ⟹ 改用 `T.reduce_max` / `T.reduce_sum` | wheel `ascend_tile.py:1528 / 1562` |
| 4 | §5.3.3 / §10 R10 | `gather_mask` 取 val 的 pattern「**待 Phase-0 验证**（推测 `"P0101"`）」 | **已确认为 `"P0101"`**（偶数位=值），R10 风险消除 | wheel `ascend_tile.py:625-631` docstring + `test_..._topk.py:169-170` |
| 5 | §5.3.6 / §5.3.7 | topk 相关 UB = **7 KB**；S1' 合计 **41 KB** | 自动 tmp arena = **64 KB**；K1 合计 **≈121 KB** | `allocate_tmp_buffer.cc:578-596` + `:1195-1206` |
| 6 | §5.3.7 / §5.1 | S1（`BV=8192`）合计 73 KB「✅ 余 111 KB」；S2'（`K_MAX=1024,BV=8192`）94 KB | 二者在正确预算模型下分别 ≈**250 KB / ≈282 KB**，**均超 184 KB 不可行**；`BV ≤ 6729` | 同 5，公式 `28×BV + O(K_MAX) ≤ 188416` |
| 7 | §3.7 / §7 D8 | `valid[i] <- (L >= 0) ? 1 : 0` | 该判据**恒真、守卫失效**；改为 `k_ok AND ok_p AND tie_ok` | D-GUARD（`prefix` 初值 0 ≤ `p_i` ⟹ `l=0` 必然进入） |
| 8 | §5.4.3 | tie 用 `rank_eq(j)` 索引序丢弃 | **不可实现**（需块内 cumsum）；改为向量化 `n_gt/n_eq` 检测 + 守卫降级 | D-TIE + `T.cumsum` 不可用 |
| 9 | §3.4 / §5.1 | 「`B` 不进入特化键」 | **无法完全做到**（GM shape 含 `B`；消除手段需 host padding，被禁止） | §4.6 R-JIT |
| 10 | §5.3.3 | `T.tile.topk` 的 src 用 `[BV]` 或 `[1,BV]` 未区分 | 必须用 **rank-1 `[BV]`** 专用 buffer（`max_actual_num = Σ shape` 陷阱） | N2 / R-C |
| 11 | §5.3.4 | `gather_mask` 等标为「arch22 专属，docstring 明示不支持 pto」 | 本设计**不需要**依赖该论证（已弃用 `wholereduce*`）；`topk`/`gather_mask` 在 ascendc(dav-2201) 上受支持，有 `ICHECK(is_half‖is_float) << "dav-2201 Sort/TopK only supports float16/float32"` 正面佐证 | `allocate_tmp_buffer.cc:583-584` |
| 12 | §4.2 / §11 | 落点 `sgl-kernel-npu/.../sample/r4/`，含 `r4_dispatch.py` / `r4_tiling.py` / `r4_lcg.py` | 落点 **`examples/R4/` 独立算子**，仅 3 个文件（§11.1） | 用户明确指定 V01-a 范围 |
| 13 | §9.1 | L0 = 「`m_out`/`s_out` 对拍」；L1 = `tau`；L2 = 掩码集合 | 本设计的 **L0 覆盖全部 5 个输出**（`m/s/tau/valid/w`），分层语义按 `tilelang-op-test-design` §5 重定义（L0 门槛 / L1 功能 / L2 异常负向 / Boundary 特殊值） | `tilelang-op-test-design` SKILL.md §5、§9.1 |
| 14 | §1.3 / §3.5 Step 1 / §3.5 Step 8 / D4 / D5 | `sampler.py` 行号：`:641`（`div_`）、`:646`（`log_softmax`）、`:246-252`（精度门槛注释）、`:779-848`（R4 本体）、`:794-812`（Path 2）、`:814-847`（Path 3） | **全部为过期行号**（sglang 仓库已演进）。实测：`div_` = **`:249`**；精度门槛注释 = **`:251-255`**；`F.log_softmax` = **`:262`**；R4 本体 `top_k_top_p_min_p_sampling_from_logits_ascend` = **`:811-879`**；Path 2 = **`:826-845`**；Path 3 = **`:846-879`** | 已逐行读取 `/mnt/workspace/cann/20261006_R4_V01/sglang/python/sglang/srt/layers/sampler.py` 核对；**本文档所有 sglang 引用均已改用实测行号**，Stage 2 写 `r4_ref.py` 时以 **`:846-879`** 为准 |

### 10.5 Developer 模式可行性最终判定（供 Orchestrator 决策）

**判定：V01-a 全链路（含 top-k 分层归并）可在用户指定的 Developer 模式下实现。✅**

- **online softmax 的 max/sum 归约** ⟹ `T.reduce_max` / `T.reduce_sum`（高层、非 deprecated、
  Developer 标杆示例双重验证）✅
- **exp / ln** ⟹ `T.exp`（`T.Parallel` 内）/ `T.log`（标量）/ `T.tile.ln`（向量备选）✅
- **compare / select / elementwise** ⟹ `T.Parallel` + `T.if_then_else` ✅
- **top-p 前缀和** ⟹ UB 标量循环 + 单调守卫（`T.cumsum` 不可用，已消解）✅
- **top-k / 排序** ⟹ ⚠️ **无纯 Developer 高层原语**，使用 Ascend 专属 `T.tile.topk` +
  `T.tile.gather_mask`。二者**非 deprecated**、标 `PIPE_V`、可作用于 `alloc_shared`
  （源码链条已证），且仓库自己的 `examples/developer_mode/` 标杆就混用 `T.tile.*` ⟹
  按 guide §1 定义属「混合模式（Developer 主体 + Ascend 专属 `T.tile.xxx`）」，
  guide 明言这是**大多数实际算子的形态**。

**需要用户/Orchestrator 知悉的唯一矛盾点（R-A）**：若用户的「Developer 模式」意图是
**严格零 `T.tile.*`**（只用 `alloc_shared`/`alloc_fragment` + `T.Parallel` + `T.copy` + `T.reduce_*`），
则 **top-k 分层归并无法实现** —— 全仓不存在任何纯高层的 top-k/sort/argsort 原语，
且用 `T.Parallel` 手写比较排序在 `V=152064` 规模下既无原语支撑也无性能可行性。
此时的可选路径：
1. **接受混合模式**（本设计，推荐）—— 保留 Developer 的分配/同步/pass_configs 语义，
   top-k 一步用 Ascend 专属原语；
2. 改用 **Expert 模式**（草稿原方案）—— 但需接受 `wholereduce*` 已 deprecated 的事实，
   仍要改用 `T.reduce_*`，收益为零；
3. **缩小范围**：去掉 top-k，只做 softmax + top-p + min-p（`tau_k = -inf`）——
   可在纯 Developer 模式实现，但**不满足用户的 V01-a 需求**（top-k 是核心）。

**本设计选 1，且未擅自更改用户指定的模式定性**（§2.1 全部 Expert 判别标志均不使用）。
**不标记 `[DESIGN_ERROR]`** —— 设计完整、可实现、无根本性阻塞。

---

## 11. 交付清单

### 11.1 目录结构

```
examples/R4/
├── r4_fused_sampling.py         # 纯 kernel：r4_topk_stats(K1) + r4_mask_sample(K2) + host 包装
│                                #   （@tilelang.jit，可 import，无 golden / 无测试 / 无 __main__）
├── r4_ref.py                    # golden：逐行复刻 sglang sampler.py:846-879 + 反推 tau/valid（§9.1）
├── test_r4_fused_sampling.py    # from r4_fused_sampling import …; from r4_ref import golden_r4
│                                #   + 精度框架 + L0 用例 + main（--level 分发）
├── proto.yaml                   # 算子接口规格（dtype/attr），供覆盖门禁派生应覆盖维度
├── DESIGN.md                    # 本设计文档
└── DESIGN_script.md             # 原始草稿（**参考输入，保留原样，不删不改**）
```

> **命名说明**：本算子有 2 个 kernel + 1 个独立 golden，故 kernel 文件用 `r4_fused_sampling.py`
> （与草稿 §4.1 一致）、golden 用 `r4_ref.py`（草稿 §9.1 要求逐行复刻参考实现，较复杂 ⟹ 允许独立文件，
> 测试文件 import 它）。这与 `design-template.md §11.1` 的「单 kernel 单文件」默认布局的偏差已在此说明。
> Stage 2 生成 `test_r4_fused_sampling.py` 时，`tilelang-op-test-design` 场景 B 只填充其
> `test_*_l1/l2/boundary` 桩体，**不改 kernel 文件、不改 main 分发器**。

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ **已完成（本文件）** | 标准 11 章设计文档，Developer 模式 + V01-a 范围 |
| `proto.yaml` | ✅ **已完成** | 算子接口规格（dtype 全集取自 §9.3 精度表、attr 取自 §4.1/§1），覆盖门禁 `coverage_check.py --proto` 用 |
| `DESIGN_script.md` | ✅ 已存在（**保留原样**） | 1161 行 Expert 模式草稿，算法/需求权威参考 |
| `r4_fused_sampling.py` | ⬜ 待实现（Stage 2） | 纯 kernel（`@tilelang.jit`）+ host 包装 + 前置校验（C3/C6/C7） |
| `r4_ref.py` | ⬜ 待实现（Stage 2） | golden（§9.1 草案已给出完整代码） |
| `test_r4_fused_sampling.py` | ⬜ 待实现（Stage 2） | 精度框架（`get_precision`/`check_precision`）+ 8 个 L0 用例 + `main(--level)` |

### 11.3 命名规范

- 目录名：`R4`（**已存在，沿用**；Orchestrator 指定的算子目录）
- kernel 文件：`r4_fused_sampling.py`（snake_case）
- 测试文件：`test_r4_fused_sampling.py`（顶部 `from r4_fused_sampling import r4_topk_stats, r4_mask_sample, r4_fused_sampling`）
- golden 文件：`r4_ref.py`（顶部 `from r4_ref import golden_r4`）
- kernel 函数名：`r4_topk_stats`（K1）、`r4_mask_sample`（K2）、`r4_fused_sampling`（host 包装）

### 11.4 实现顺序

1. ✅ 设计文档（`DESIGN.md`）+ **`proto.yaml`** + **L0 门槛测试计划**（§9.2）—— Stage 1 交付（V01.1：Stage 2 验证后已对齐）
2. ✅ **上板验证项 V-1 ~ V-3**（§10.2）—— Stage 2 完成：V-1/V-3 ✅ 无需兜底；V-2 发现 narrow-reduce ≤256 B（64 fp32）约束，改精确尺寸尾块缓冲 + 全量 reduce ✅
3. ✅ kernel 实现（`r4_fused_sampling.py`，纯 `@tilelang.jit`）—— K1 `r4_topk_stats` + K2 `r4_mask_sample` + host 包装，NPU 验证通过
4. ✅ golden（`r4_ref.py`，§9.1 草案，**已含 tau_p/L raw 前缀和修正**）+ 测试文件（`test_r4_fused_sampling.py`）
5. ✅ **L0 门槛测试通过（精度收敛）** —— 8 用例 × 5 输出全过（首轮 3 处 valid 失败已修复，`debug_log.md`）
6. ✅ 扩展分层套件（L1 18/18 / L2 9/9 / Boundary 13/13，由 `tilelang-op-test-design` **场景 B** 生成）
7. ✅ 覆盖门禁：`coverage_check.py --proto` → category=Fusion，**24 PASS / 0 MISS / 0 N/A（exit 0）**
8. ✅ 全量套件运行通过 —— 26 PRECISION_PASS / 0 FAIL；22 BOUNDARY_PASS / 0 WARN；性能哨兵 K1+K2 B=64→**1.2 ms**、B=256→**3.6 ms**（§9.4 达标）
9. ⬜（可选，Stage 3）性能调优：§6.3 优化清单（§9.4 哨兵已达标；`T.Pipelined(stage=2)` 需先把 BV 降到 2048 腾双缓冲）

### 11.5 运行 / 复现说明（Stage 2 必须遵守）

**目标平台事实（已验证）**

| 项 | 值 | 来源 |
|---|---|---|
| 芯片（实测） | **Ascend910_9362** | Orchestrator 实测 |
| 平台判定 | **A3** | `tilelang/utils/target.py:145-147`（`"910_93" in name.upper()` → `"A3"`） |
| 架构 | **arch22 / DAV_2201** | `tilelang/jit/adapter/libgen.py:152-155`（ascendc target 固定 `--npu-arch=dav-2201`） |
| Cube 核 / Vector 核 | **20 / 40**（`VEC_NUM = 2`） | 运行时 `detect_vec_core_num()`（`examples/xllm_kernels/utils.py:20-40`）；**禁止硬编码 48** |
| L2 / UB | 168 MB / 192 KB（向量可用 **184 KB**） | §4.5 |
| CANN | 9.2.0 | Orchestrator 实测 |
| torch / torch_npu | 2.7.1 / 2.7.1.post4 | Orchestrator 实测 |
| tilelang（**已安装 wheel**） | `0.1.4+d75300ed0a88b0f4b2eef8a62b43f1930fdae44b` @ `~/.local/lib/python3.12/site-packages/tilelang`（自带 tvm，含全部 `T.tile.*` API） | 本阶段已 `import tilelang` 实测确认 |
| 端到端环境状态 | ✅ 已通过 `quick_verify.py` 验证（编译 + 运行 ascendc kernel，exit 0） | Orchestrator 实测 |

**🔴 运行约束（Stage 2 必须遵守，否则会误判为设计/实现错误）**

1. **使用已安装的 tilelang wheel**（`~/.local/lib/python3.12/site-packages/tilelang`）。
2. **不要 `source set_env.sh`** —— 它把仓库根加进 `PYTHONPATH`，导致**未编译的仓库
   `tilelang/` 源码 shadow 已安装 wheel**，报 `No module named tvm`。
3. **从中立 cwd 运行**，或确认仓库根不在 `PYTHONPATH` 中
   （`python -c "import tilelang,os;print(os.path.dirname(tilelang.__file__))"` 应打印
   `~/.local/lib/python3.12/site-packages/tilelang`，**不是**仓库路径）。
4. 设 **`ACL_OP_INIT_MODE=1`**。
5. 仓库的 `examples/`、`tilelang/language/`、`src/` **仅作 API 参考**（本设计的所有源码引用
   均已在 wheel 对应提交 `d75300ed` 上复核），不作为运行时导入路径。
6. **生产/反复调试时不要调用 `tilelang.disable_cache()` / `tl.cache.clear_cache()`**
   （示例代码中的调用仅为演示）；配 `TILELANG_CACHE_DIR` 持久化以缓解 R-JIT。

---

## 11.6 算子 proto.yaml（覆盖门禁用，Stage 1 产出）

> **dtype 全集取自本文档 §9.3 精度表**（float32 + int32，共 2 行；§4.1 输入张量的 dtype 与之一致）
> + **§4.1/§1.3** 的 attr 机械派生。是覆盖门禁 `coverage_check.py --proto` 的**权威 dtype/attr 来源**。
> checker 只读 `operator.inputs[].dtype` 与 `operator.attrs[].name`。
> 文件已同步落盘至 `examples/R4/proto.yaml`。

```yaml
operator:
  name: r4_fused_sampling
  category: Fusion
  formula: |
    m_i   = max_j z_ij
    s_i   = sum_j exp(z_ij - m_i)
    p_ij  = exp(z_ij - m_i) / s_i
    tau_i = max( tau_k_i, tau_p_i, tau_mu_i )
      tau_k_i  = sort_desc(z_i)[k_i - 1]                       # top-k 阈值（分层归并求得）
      tau_p_i  = sort_desc(z_i)[L_i],  L_i = max{ l : sum_{t<l} q_it <= p_i }   # top-p 严格前缀和
      tau_mu_i = m_i + ln(mu_i)   (mu_i = 0 -> -inf)           # 定理 T2
    w_ij  = p_ij * 1[z_ij >= tau_i]                            # 定理 T1；不重归一化
    valid_i = 1[ 1 <= k_i <= K_MAX ] * ok_p_i * tie_ok_i       # D-GUARD / D-TIE 降级守卫
  attrs:
    - name: K_MAX
      type: Int
      default: 256
      required: true
    - name: BV
      type: Int
      default: 4096
      required: true
    - name: NEED_MIN_P
      type: Bool
      default: false
      required: true
    - name: top_k
      type: Int
      default: 40
      required: true
    - name: top_p
      type: Float
      default: 0.95
      required: true
    - name: min_p
      type: Float
      default: 0.0
      required: true
    - name: need_min_p_sampling
      type: Bool
      default: false
      required: true
  inputs:
    - name: logits
      dtype: [float32]
    - name: top_ks
      dtype: [int32]
    - name: top_ps
      dtype: [float32]
    - name: min_ps
      dtype: [float32]
  outputs:
    - name: m_out
      dtype: [float32]
    - name: s_out
      dtype: [float32]
    - name: tau_out
      dtype: [float32]
    - name: valid
      dtype: [int32]
    - name: w_out
      dtype: [float32]
  schema: |
    r4_fused_sampling(Tensor logits[B,V]:f32, Tensor top_ks[B]:i32,
                      Tensor top_ps[B]:f32, Tensor min_ps[B]:f32,
                      bool need_min_p_sampling,
                      int K_MAX=256, int BV=4096)
      -> (Tensor m_out[B]:f32, Tensor s_out[B]:f32, Tensor tau_out[B]:f32,
          Tensor valid[B]:i32, Tensor w_out[B,V]:f32)
    # kernels: r4_topk_stats(B,V,K_MAX,BV,NEED_MIN_P) -> (m_out,s_out,tau_out,valid)
    #          r4_mask_sample(B,V,BV)                 -> (w_out)
```

**attr 语义说明**（供 Stage 2 派生 `D-PARAM-*` 覆盖维度）

| attr | 类型 | 取值域 | 影响的计算路径 |
|------|------|-------|--------------|
| `K_MAX` | Int（编译期） | V01-a 固定 256 | 归并 buffer 尺寸、收尾标量循环长度、`k_ok` 守卫上界 |
| `BV` | Int（编译期） | V01-a 固定 4096（约束 `BV ≤ 6729`，R-E） | `n_blk`、`TAIL`、UB 预算、topk tmp arena 尺寸 |
| `NEED_MIN_P` | Bool（编译期） | True / False | False 时 K1 **完全跳过** `tau_mu` 计算分支 |
| `top_k` | Int（逐行运行时） | `[1, K_MAX]` 合法；`> K_MAX` 或 `TOP_K_ALL=1<<30` → `valid=0` | `tau_k` 取值、`k_ok` 守卫、`tie_ok` 判据 |
| `top_p` | Float（逐行运行时） | `(0, 1]`；`1.0` = 不启用 | `tau_p` 前缀和循环的守卫、`ok_p` 判据 |
| `min_p` | Float（逐行运行时） | `[0, 1]`；`0.0` = 不启用 | `tau_mu = m + ln(mu)`（定理 T2），`mu=0` 特判 `-inf` |
| `need_min_p_sampling` | Bool（host 标量） | True / False | 与 `NEED_MIN_P` 对应；`sampling_batch_info.py:262` = `any(min_p > 0)` |

**一致性约束核对**：`inputs[].dtype` = `{float32, int32}`，与 §9.3 精度表的 2 行 dtype **完全一致** ✅；
`outputs[].dtype` 同上 ✅；`attrs[].name` 覆盖全部 7 个影响计算路径的属性 ✅。

---

## 12. 性能目标（Stage 3 用户确认，2026-10-09）

> 本章由 `@tilelang-op-orchestrator` 在 Stage 3 启动前追加，不修改上文任何既有内容。
> 前次 Stage 3 曾于 2026-10-08 被用户主动跳过（`perf_tuning_requested="no"`）；本次用户明确要求启动性能优化。

### 12.1 用户确认的调优参数

| 字段 | 取值 | 说明 |
|------|------|------|
| 性能目标类型 | `best_effort` | 无硬性数值目标；以 roofline 达成率与 geomean speedup 提升为准 |
| 目标数值 | — | `best_effort` 类型不适用 |
| Baseline 路径 | `examples/R4/bench/reports/cann_final_eval_20261009_11{0101,0214,0320,0425}.json` + `examples/R4/bench/metadata_staging/baseline_raw_910b2_18cases.json` | 18-case × 2 任务已实测基线（fused/mask 各 2 次运行） |
| 测试 shape | **全 18-case 矩阵**（`r4_fused_sampling` 与 `r4_mask_sample` 各 18 case） | 任何 case 不得回退超过噪声阈值 |
| 噪声阈值 | 3% | 性能变化 <3% 视为噪声，不采纳该优化 |
| 最大迭代数 | 10 | 中止条件：达到 10 轮 / 连续 3 次无提升 |
| 交付件改动范围 | **允许直接改主文件**，优化后重跑精度门禁 + cann-bench 三阶段评测，并同步更新 metadata/submission 的 sha256 | 已交付 sha256 `bbe47fa4…` 将失效，必须刷新 |

### 12.2 已知瓶颈（Stage 2 / bench 阶段的量化结论，非推测）

- **Grid 并行度只绑 batch**：`b_num = T.ceildiv(B, VEC_NUM=2)`，vocab 维在 block 内以 `T.serial(n_blk)` 串行扫描（`r4_fused_sampling.py:108 / 340 / 410`）。
- **核心利用率实测**：case 7 (B=4) → 2/20 组核心；case 15 (B=64) → 32；case 16 (B=128) → 64。
- **Roofline 达成率仅 1%–4%**：case 15 fused `elapsed=1974.74us` vs `t_hw=78.31us` → 3.97%；mask `985.44us` vs `21.54us` → 2.19%。
- **最高杠杆方向**：split-V 并行化（grid 沿 V 扩展 + `m/s/tau` 跨 split 归约 + K2 partial-argmax 合并），对小 B LLM decode 收益最大。

### 12.3 精度红线（Stage 3 不得突破）

Stage 2 的 `TAIL_PAD = ceil(TAIL/32)*32` + `ztp=[1,TAIL_PAD]`（从 GM 以 `pad_value=-inf` 载入）修复，是针对两个 TileLang wheel 缺陷的**结构性规避**：① `tl::ascend::Sort()` 修补余数的 masked `AscendC::Duplicate` 在 dav_c220 上不生效；② UB→UB 的 `T.copy`（`copy_ub_to_ub` / `AscendC::DataCopy`）在 count 非 32B 倍数时静默截断到 8 的倍数。

因此任何性能优化都**不得**重新引入非 32B 倍数的 UB→UB 拷贝，或 `count % 32 != 0` 的 `T.tile.topk` 调用。完整根因见 `.orchestrator_state.json → resolved_failure.root_cause_ESTABLISHED`。

---

## 修订记录

| 版本 | 日期 | 修改人 | 修改内容 |
|---|---|---|---|
| V01-a-Stage1 | 2026-10-06 | `@tilelang-op-analyst`（Stage 1） | 初版标准 DESIGN.md。基于 `DESIGN_script.md` 草稿的算法/数学（定理 T1/T2、online softmax、分层 top-k 归并），按**用户指定的 Developer 模式**全量重做 API 映射；范围收敛到 V01-a（K1+K2、fp32、单特化 S1'、保留 `torch.multinomial`）。**13 项草稿结论经源码复核后修正**（§10.4），其中 3 项为实质缺陷：`valid` 守卫判据恒真失效（D-GUARD）、tie 索引序方案不可实现（D-TIE）、UB 预算低估约 3 倍（R-E）。新增 §2.3/§6.5/§10.5 三处 Developer 模式可行性论证与 `threads=2` 例外说明。产出 `proto.yaml` 与 8 用例 L0 门槛测试计划（§9.2）。 |
| V01.1 | 2026-10-06 | `@tilelang-op-analyst`（Stage 1 定向文档对齐） | Stage 2 实现验证后文档对齐（**非 `[DESIGN_ERROR]` 回退**，设计主体——算法/tiling/内存层级/API 选型/Developer+`T.tile.topk` 混合模式定性/V01-a 范围/L0 计划——不变）：① §9.1 golden 的 `tau_p`/`L` 修正为 **raw `probs_sort` 严格前缀和**（sglang 屏蔽链仅用于 `w_ref`）；② §3.4 D-TIE 改**分级判据**（消除 `k_i==K_MAX` 时与「论证 2」/§9.2 的自相矛盾）+ 显式记录残留已知缺口（横跨 rank `K_MAX-1`/`K_MAX` 的 2 值平票，非阻塞）；③ §10.2 V-2 尾块归约：`real_shape`/BufferRegion narrow-reduce 对 `valid>64` fp32 不可用（`codegen_ascend.cc:2145` ≤256 B 约束），改**精确尺寸 `[1,TAIL]` 缓冲 + 全量 reduce**；④ 标注全链路已 NPU（Ascend910_9362/A3）验证通过（L0 8/8、L1 18/18、L2 9/9、Boundary 13/13、覆盖门禁 24/24、哨兵 B=64→1.2ms / B=256→3.6ms）。同步更新：§2.3/§2.4/§3.2/§3.3 注 3-4/§3.5.1/§3.5.2/§3.5.3/§5.3 N1・N2・N5/§5.4 C1・非整除声明/§6.4/§9 前言/§10.1 R-B・R-D・R-NPU/§10.3/§11.4。 |

