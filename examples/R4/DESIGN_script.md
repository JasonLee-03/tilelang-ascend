# R4 算子设计文档：融合采样链（Fused Sampling Chain）

| 项 | 内容 |
|---|---|
| 算子编号 | R4 |
| 算子名称 | 融合采样链（Fused Top-k / Top-p / Min-p Sampling） |
| 被替换实现 | `top_k_top_p_min_p_sampling_from_logits_ascend()` |
| 宿主文件 | `sglang/python/sglang/srt/layers/sampler.py:779-848` |
| 触发入口 | `Sampler._forward_ascend_backend()`（`sampler.py:623-647`） |
| 所属子系统 | Model Runner → 输出级（Sampling Stage） |
| 功能归类 | Reduction + Elementwise + Selection（**纯 Vector，不使用 Cube**） |
| **目标芯片** | **Ascend910_9382（A3）** |
| **目标架构** | **arch22 / DAV_2201**（`CMakeLists.txt:7-15` 映射：A2=Ascend910B\*、A3=Ascend910_9382 → `arch22`） |
| **开发框架** | **TileLang-Ascend v0.1.4**，`target="ascendc"`，bisheng arch `dav-2201` |
| 硬件资源 | UB 192 KB（向量可用 184 KB，系统保留 8 KB）；Vector 核 48（`VEC_NUM=2` × Cube 核 24） |
| 依赖 CANN 算子 | `torch_npu.npu_top_k_top_p`（V01 保留）、`torch.multinomial`（V01-a 保留） |
| 版本 | V01 |
| 日期 | 2026-10-04 |

---

## 1. 概述

### 1.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | `r4_fused_sampling`（K1 `r4_topk_stats` + K2 `r4_mask_sample`） |
| 算子类别 | Reduction / Elementwise / Selection（纯 Vector） |
| 支持数据类型 | **fp32**（logits）→ fp32（probs）/ int32（token_id）；bf16 输入待 Phase-0 验证 |
| **目标芯片** | **Ascend910_9382（A3）** |
| **目标架构** | **arch22（DAV_2201）** |
| 编译目标 | `target="ascendc"` → 生成 Ascend C 源码 → bisheng 编译 `.so` |
| 明确排除 | `target="pto"`（A5/Ascend950 专属）；`float64`（框架零支持） |

### 1.2 算子功能

将 SGLang 在昇腾 NPU 上的**采样链**从 6~19 个 eager 算子融合为 1~2 个 TileLang kernel。输入 temperature 缩放后的 logits `[B, V]`，输出采样得到的 token id `[B]`（及可选的屏蔽后概率 `[B, V]`）。

融合覆盖的计算步骤：

```
temperature 缩放 → softmax → top-k 截断 → top-p(nucleus) 截断 → min-p 截断 → 采样
```

**不覆盖**（V01 明确排除，见 §6 D7/D8）：
- `logprob` 计算（保留 `torch.log_softmax`，受 `sampler.py:246-252` 逐位对齐门槛约束）
- 确定性采样路径（`sampling_seed ≠ None`，依赖 float64 Gumbel 噪声）
- 惩罚链（`penaltylib/`，属上游独立算子）

### 1.3 在 SRT 中的位置

```
SRT = Engine + Scheduler + Memory Management + Model Runner + Attention Backend
                                                    ↑
                                              R4 落在这里（输出级）
```

| SRT 组件 | R4 关系 |
|---|---|
| Engine | ❌ 不属 |
| Scheduler | ⚠️ **参数来源 + 隐式耦合**：`sampling_batch_info.py:245-247` 的 `all(1 <= top_k <= 1024)` 决定走快路径还是降级路径 |
| Memory Management | ❌ 不属（不碰 KV 池 / Radix Tree） |
| **Model Runner** | ✅ **本体**：`model_runner.py:1988 next_token_ids = self.sampler(...)` |
| Attention Backend | ❌ 不属（在 attention 与 LM head 之后） |

调用位置：

```
ModelRunner.forward
  ├─ LogitsProcessor.forward            logits_processor.py:495
  │    └─ LogitsProcessorOutput(next_token_logits)     :574/:592/:1142
  ├─ Sampler._preprocess_logits         sampler.py:131（custom processor + sanitize_nan_logits）
  └─ ★ Sampler.forward                  sampler.py:140
       └─ :220 if self.use_ascend_backend:
            └─ :623 _forward_ascend_backend
                 ├─ :641 logits.div_(temperatures)          ← in-place
                 ├─ :643 _sample_from_logits  (:586)
                 │    └─ :611 top_k_top_p_min_p_sampling_from_logits_ascend (:779)  ← R4 本体
                 └─ :646 logprobs = torch.log_softmax(logits, dim=-1)   ← V01 不融合
```

后端开关：`sampler.py:126` `self.use_ascend_backend = get_exec().kernel.sampling_backend == "ascend"`；注册机制 `:683 register_sampler_backend` / `:697 create_sampler`，`_BUILT_IN_SAMPLING_BACKENDS = {"flashinfer", "pytorch", "ascend"}`（`:75`）。

→ **集成可走 `register_sampler_backend("ascend_tilelang", ...)` 注册第四个 backend，无需改动 `forward` 主逻辑，也无需引入 `KernelBackend.TILELANG`。**

### 1.4 为什么需要融合

采样本身计算量近乎为零（掩码比较 + 归约），但当前实现对 `[B, V]` 全 vocab 张量反复遍历，是**纯访存瓶颈**；且 NPU 上 `torch.compile` 被禁用（`sampler.py:850 @torch.compile(dynamic=True, disable=is_npu())`），无法靠图编译消解，只能靠算子融合。

---

## 2. 现状基线（被替换的 eager 链）

### 2.1 四条互斥执行路径

| 路径 | 触发条件 | 代码位置 | 全 vocab 遍历次数 |
|---|---|---|---|
| **Path 0** 贪心 | `sampling_info.is_all_greedy`（所有 `top_k <= 1`） | `sampler.py:181-197` | 1~3 |
| **Path 1** 简单采样 | `simple_sampling_case` = `not need_top_p and not need_top_k and not need_min_p` | `sampler.py:597-606` | 2~8 |
| **Path 2** NPU 快路径 | `hasattr(torch_npu,"npu_top_k_top_p") and npu_top_k_top_p_eligible` | `sampler.py:794-812` | **6（seed=None）/ ≈14（seed≠None）** |
| **Path 3** 降级路径 | 上述不成立（含 `top_k > 1024`） | `sampler.py:814-847` | **≈13（seed=None）/ ≈19（seed≠None）+ 1 次 O(V log V) 排序** |

### 2.2 Path 2 快路径逐算子清单（V01 主要目标）

```python
# sampler.py:794-812
:795  logits_top_k_top_p = torch_npu.npu_top_k_top_p(logits, top_ps, top_ks)   # ① CANN 黑盒（内含 top-k+top-p）
:796  probs_top_k_top_p  = logits_top_k_top_p.softmax(dim=-1)                  # ② 全 vocab
:799  min_p_thresholds   = probs_top_k_top_p.max(dim=-1) * min_ps              # ③④ 全 vocab（仅 need_min_p）
:800  min_p_mask         = probs_top_k_top_p < min_p_thresholds.view(-1, 1)    # ⑤ 全 vocab
:801  probs_top_k_top_p.masked_fill_(min_p_mask, 0.0)                          # ⑥ 全 vocab
:804  batch_next_token_ids = torch.multinomial(probs_top_k_top_p, num_samples=1) # ⑦ 全 vocab（仅 seed=None）
# seed≠None 分支额外：
:806  logprobs_top_k_top_p = probs_top_k_top_p.to(torch.float64)               # ⑧ 全 vocab，显存 ×2
:809  logprobs_top_k_top_p.log_()                                              # ⑨ 全 vocab
:810  multinomial_with_seed(...)                                               # ⑩~⑭ 见 §3.5 Step 7b
```

### 2.3 Path 3 降级路径的性能悬崖

```python
# sampler.py:814-847
:815  probs = torch.softmax(logits, dim=-1)                        # 全 vocab
:816  probs_sort, probs_idx = probs.sort(dim=-1, descending=True)  # 🔴 O(V log V)，产出 2 个 [B,V] 张量
:820  top_ks.masked_fill_(topk_all_mask, probs.shape[1])           # TOP_K_ALL → V
:821  top_k_mask = torch.arange(0, V, device=...).view(1,-1) >= top_ks.view(-1,1)  # 全 vocab + 每次新建 arange
:823  probs_sort.masked_fill_(top_k_mask, 0.0)
:826  probs_sum = torch.cumsum(probs_sort, dim=-1)                 # 🔴 全 vocab 前缀和
:827  top_p_mask = probs_sum - probs_sort > top_ps.view(-1, 1)
:828  probs_sort.masked_fill_(top_p_mask, 0.0)
:831  min_p_thresholds = probs_sort[:, 0] * min_ps                 # 已排序，O(1) 取 max
:833  probs_sort.masked_fill_(min_p_mask, 0.0)
:844  probs_idx = probs_idx.to(torch.int32)                        # 全 vocab
:845  batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index)
```

**触发条件**（`srt/sampling/sampling_batch_info.py:245-247`）：

```python
npu_top_k_top_p_eligible = all(1 <= r.sampling_params.top_k <= 1024 for r in reqs)
# :499  batch merge 时进一步收紧
self.npu_top_k_top_p_eligible &= other.npu_top_k_top_p_eligible
```

⚠️ `all(...)` + `&=` 语义 → 连续批处理混批时，**batch 内任一请求 `top_k > 1024` 或 `top_k < 1`，整批降级到 O(V log V) 全 vocab 排序**。这是 R4 最大的性能杠杆点，也是本设计要消除的悬崖。

### 2.4 访存量核算（B=256, V=152064, fp32）

单个 `[B,V]` fp32 张量 = 256 × 152064 × 4 B = **148.5 MB**

| 路径 | 全 vocab 访问次数 | HBM 流量 |
|---|---|---|
| Path 2 / seed=None | 6 | ≈ 891 MB |
| Path 2 / seed≠None | ≈14（含 fp64 张量 ×2 体积） | ≈ 2.4 GB |
| Path 3 / seed=None | ≈13 + sort | ≈ 1.9 GB + 排序临时 |
| Path 3 / seed≠None | ≈19 + sort | ≈ 3.2 GB + 排序临时 |
| **本设计（K1+K2）** | **3 读 + 1 写** | **≈ 594 MB** |
| **本设计（K1+K2+Gumbel 融合）** | **3 读 + ε 写** | **≈ 445 MB** |

每 decode step 必调 → 收益直接体现为 **TPOT**。

---

## 3. 算子规格

### 3.1 函数签名

**Host 侧 Python 封装**：

```python
def r4_fused_sampling(
    logits: torch.Tensor,          # [B, V] fp32，已 temperature 缩放
    top_ks: torch.Tensor,          # [B]    int32
    top_ps: torch.Tensor,          # [B]    fp32
    min_ps: torch.Tensor,          # [B]    fp32
    need_min_p_sampling: bool,     # 标量，编译期常量
    *,
    return_masked_probs: bool = True,   # V01-a: 输出 w 供 torch.multinomial
    fuse_gumbel_sample: bool = False,   # V01-b: 内部完成采样
    rand_seed: Optional[torch.Tensor] = None,   # [B] int64，仅 fuse_gumbel_sample
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    ...
```

**TileLang kernel 签名**：

```python
@tl.jit(out_idx=[...], pass_configs=DEFAULT_ASCEND_PASS_CONFIGS)
def r4_topk_stats(B: int, V: int, K_MAX: int, BV: int, NEED_MIN_P: bool):
    @T.prim_func
    def main(
        logits : T.Tensor([B, V],     "float32"),   # in
        top_ks : T.Tensor([B],        "int32"),     # in
        top_ps : T.Tensor([B],        "float32"),   # in
        min_ps : T.Tensor([B],        "float32"),   # in
        m_out  : T.Tensor([B],        "float32"),   # out  行最大值
        s_out  : T.Tensor([B],        "float32"),   # out  exp 和
        tau_out: T.Tensor([B],        "float32"),   # out  合成阈值
        valid  : T.Tensor([B],        "int32"),     # out  降级守卫标志
    ): ...

@tl.jit(out_idx=[...], pass_configs=DEFAULT_ASCEND_PASS_CONFIGS)
def r4_mask_sample(B: int, V: int, BV: int, FUSE_GUMBEL: bool):
    @T.prim_func
    def main(
        logits : T.Tensor([B, V],     "float32"),   # in
        m_in   : T.Tensor([B],        "float32"),   # in
        s_in   : T.Tensor([B],        "float32"),   # in
        tau_in : T.Tensor([B],        "float32"),   # in
        valid  : T.Tensor([B],        "int32"),     # in
        rand   : T.Tensor([B, BV],    "int32"),     # in（仅 FUSE_GUMBEL）
        w_out  : T.Tensor([B, V],     "float32"),   # out（V01-a）
        tok_out: T.Tensor([B],        "int32"),     # out（V01-b）
    ): ...
```

### 3.2 输入

| 参数 | Shape | dtype | 语义 | 约束 |
|---|---|---|---|---|
| `logits` | **`[B, V]`** | `float32` | temperature 缩放后的 logits $z'_{i,j}$ | 连续内存；`logits_processor.py:1131 logits = logits.float()` 保证 fp32；**行内允许含 `-inf`**（上游屏蔽） |
| `top_ks` | **`[B]`** | `int32` | 每行 top-k 的 $k_i$ | 取值 `1 ≤ k ≤ K_MAX`，或 `TOP_K_ALL`（= -1，语义为 $V$）；`k > K_MAX` 触发降级守卫 |
| `top_ps` | **`[B]`** | `float32` | 每行 nucleus 的 $p_i$ | `(0, 1]`；`1.0` 表示不启用 |
| `min_ps` | **`[B]`** | `float32` | 每行 min-p 的 $\mu_i$ | `[0, 1]`；`0.0` 表示不启用 |
| `need_min_p_sampling` | 标量 | `bool` | 是否启用 min-p | **编译期常量**，`False` 时 K1 跳过 $\tau_{\min p}$ 计算 |
| `rand`（V01-b） | **`[B, BV]`** | `int32` | LCG 随机种子流 | 仅 `fuse_gumbel_sample=True`；由 host 侧 `random_1d` LCG 生成 |

**维度符号**：

| 符号 | 含义 | 典型值 |
|---|---|---|
| $B$ | batch 内序列数（token 数） | decode：1~256；prefill 尾 token：1~1024 |
| $V$ | vocab size，**编译期静态** | 151936 / 152064（Qwen 系）、128256（Llama3）、129280（DSv3）、32000（Mixtral） |
| $k$ | top-k 的 K | 1~1024（`npu_top_k_top_p` 硬限制），本设计 `K_MAX ∈ {64,128,256,512,1024}` |
| $p$ | top-p 的 nucleus 概率 | 0.9 / 0.95 / 1.0 |
| $\mu$ | min-p 阈值比例 | 0.0 / 0.05 / 0.1 |
| `BV` | vocab 分块大小 | 2048 / 4096 / 8192（受 UB 约束，见 §5.5） |
| `n_blk` | 分块数 = $\lceil V / BV \rceil$ | V=152064, BV=4096 → **38** |

### 3.3 输出

| 参数 | Shape | dtype | 语义 |
|---|---|---|---|
| `m_out` | **`[B]`** | `float32` | $m_i = \max_j z'_{i,j}$ |
| `s_out` | **`[B]`** | `float32` | $s_i = \sum_j e^{z'_{i,j} - m_i}$ |
| `tau_out` | **`[B]`** | `float32` | 合成阈值 $\tau_i = \max(\tau^{(k)}_i, \tau^{(p)}_i, \tau^{(\mu)}_i)$ |
| `valid` | **`[B]`** | `int32` | 降级守卫：`1` = 结果有效；`0` = nucleus 超出 `K_MAX`，host 侧须回退 torch 路径 |
| `w_out`（V01-a） | **`[B, V]`** | `float32` | 屏蔽后**已归一化**概率 $w_{i,j}$，交 `torch.multinomial` |
| `tok_out`（V01-b） | **`[B]`** | `int32` | 采样得到的 token id $y_i$ |

**中间产物（K1 内部，不落 GM）**：`topk_val[B, K_MAX]` fp32、`topk_idx[B, K_MAX]` int32 —— 仅在需要调试/对拍时导出。

### 3.4 形状特化表（应对 JIT 重编译）

TileLang JIT 为每个新形状重编译（bisheng + TVM lower，远重于 Triton）。必须做形状收敛：

| 特化维度 | 取值集合 | 数量 | 依据 |
|---|---|---|---|
| $V$ | `{32000, 128256, 129280, 151936, 152064}` | 5 | 模型固定，进程生命周期内不变 |
| `K_MAX` | `{64, 128, 256, 512, 1024}` | 5 | 覆盖 `npu_top_k_top_p` 全域 |
| `BV` | `{4096}`（默认）/ `{2048, 8192}` | 1~3 | UB 预算权衡 |
| `NEED_MIN_P` | `{True, False}` | 2 | `sampling_batch_info.py:244` |
| `FUSE_GUMBEL` | `{True, False}` | 2 | V01-a / V01-b |
| $B$ | **运行时参数，不特化** | 1 | 通过 `T.Kernel(b_num)` 分核，$B$ 仅影响 grid |

→ 最坏 5×5×3×2×2 = **300 个特化**；实际部署按模型收敛到 **≤ 10 个**。$B$ 不进入特化键是关键设计（避免连续批处理导致重编译风暴）。

参照实现：`examples/xllm_kernels/fused_gdn_gating.py` 的 `BATCH_SIZE_SPECIALIZATIONS = tuple(range(2, 49, 2))` + `select_launch_block_num(num_batches, vec_core_num)`（`examples/xllm_kernels/utils.py:37`）。

缓存策略：`TILELANG_CACHE_DIR` 持久化；⚠️ **生产环境禁止 `tilelang.disable_cache()` / `tl.cache.clear_cache()`**（示例代码中的调用仅为演示）。

### 3.5 数学公式

记 batch 内第 $i$ 个请求、vocab 位置 $j \in [0, V)$。

#### Step 0 — 上游预处理（不在本算子内）

```python
# sampler.py:135-137  _preprocess_logits
z ← sanitize_nan_logits(custom_logit_processor(z))
```

#### Step 1 — Temperature 缩放

```python
# sampler.py:641（in-place，Path 1/2/3 共用）
logits.div_(sampling_info.temperatures)
```

$$z'_{i,j} = \frac{z_{i,j}}{T_i}$$

> V01 保留在 host 侧（`_forward_ascend_backend:641`），**不并入 kernel**——因 `:646` 的 `log_softmax` 也依赖缩放后的 `logits`，并入会导致重复计算。见 §6 D6。

#### Step 2 — Softmax

$$m_i = \max_{j} z'_{i,j} \qquad s_i = \sum_{j=0}^{V-1} e^{\,z'_{i,j} - m_i} \qquad p_{i,j} = \frac{e^{\,z'_{i,j} - m_i}}{s_i}$$

**在线（online）形式**（单遍历，分块累加，参照 `examples/softmax/example_online_softmax.py`）：

对第 $b$ 块，设块内统计量为 $(m^{(b)}, s^{(b)})$，则

$$m^{(b)}_{\text{new}} = \max\!\big(m^{(b-1)},\ \max_{j \in \text{blk}_b} z'_{i,j}\big)$$

$$s^{(b)}_{\text{new}} = s^{(b-1)} \cdot e^{\,m^{(b-1)} - m^{(b)}_{\text{new}}} \;+\; \sum_{j \in \text{blk}_b} e^{\,z'_{i,j} - m^{(b)}_{\text{new}}}$$

#### Step 3 — Top-k 截断

设 $\pi_i$ 为 $p_{i,\cdot}$ 的降序排列索引，$q_{i,l} = p_{i,\pi_i(l)}$。`TOP_K_ALL`（-1）时置 $k_i = V$（`sampler.py:820`）：

$$\mathcal{S}^{(k)}_i = \{\, j : \operatorname{rank}^{\downarrow}_i(j) < k_i \,\}, \qquad p_{i,j} \leftarrow 0 \quad \forall j \notin \mathcal{S}^{(k)}_i$$

#### Step 4 — Top-p（nucleus）截断

前缀和 $C_{i,l} = \sum_{t \le l} q_{i,t}$。代码判据（`sampler.py:826-828`）为**严格前缀和（不含自身）**：

$$C_{i,l} - q_{i,l} > p_i \;\Longrightarrow\; q_{i,l} \leftarrow 0$$

等价于保留最小集合 $\mathcal{S}^{(p)}_i$ 使 $\sum_{l \in \mathcal{S}^{(p)}_i} q_{i,l} \ge p_i$（**含**使累积首次达到阈值的那个元素）：

$$L_i = \max\Big\{\, l : \sum_{t < l} q_{i,t} \le p_i \,\Big\}, \qquad \mathcal{S}^{(p)}_i = \{\pi_i(0), \dots, \pi_i(L_i)\}$$

#### Step 5 — Min-p 截断

```python
# 快路径 sampler.py:799-801（全 vocab reduce）
min_p_thresholds = probs.max(dim=-1) * min_ps
# 降级路径 sampler.py:831（已排序，O(1)）
min_p_thresholds = probs_sort[:, 0] * min_ps
```

$$\theta_i = \Big(\max_j p_{i,j}\Big) \cdot \mu_i, \qquad p_{i,j} < \theta_i \;\Longrightarrow\; p_{i,j} \leftarrow 0$$

两处实现语义等价（top-k 必保留最大值元素，故 $\max_j p_{i,j} = q_{i,0}$）。

#### Step 6 — 归一化

🔴 **原实现无显式 renorm 算子**，由 `torch.multinomial` 隐式完成（`multinomial_with_seed` docstring `sampler.py:857-859`：*"The values are treated as weights and **do not need to sum to 1**"*）。独立 renorm 算子位于非 NPU 路径：`kernels/ops/sampling/renorm_triton.py:92 top_p_renorm_probs_triton`、`:135 top_k_renorm_probs_triton`、`sampler.py:924 top_p_normalize_probs_torch`。

$$\tilde p_{i,j} = \frac{w_{i,j}}{\sum_{j'} w_{i,j'}}, \qquad w_{i,j} = p_{i,j} \cdot \mathbb{1}[\,j \in \mathcal{S}^{(k)}_i \cap \mathcal{S}^{(p)}_i \cap \mathcal{S}^{(\mu)}_i\,]$$

#### Step 7a — 采样（非确定性，`sampling_seed is None`）

$$y_i \sim \operatorname{Categorical}(\tilde p_{i,\cdot}) \qquad \text{（`torch.multinomial`，`sampler.py:804`）}$$

#### Step 7b — 采样（确定性，`sampling_seed ≠ None`）🔴 V01 不支持

Gumbel-max trick（`sampler.py:851-892 multinomial_with_seed`）：

$$U_{i,j} = \frac{\operatorname{murmur32}(\text{seed}_i,\ \text{pos}_i,\ j)}{2^{32}-1} \in [0,1]$$

$$\tilde U_{i,j} = \operatorname{clamp}\!\Big(\ln U_{i,j},\ \min=\text{FLT}_{\min},\ \max=-2^{-32}\Big)$$

$$G_{i,j} = -\ln\big(-\tilde U_{i,j}\big) \sim \operatorname{Gumbel}(0,1)$$

$$\boxed{\,y_i = \arg\max_j \Big[\ \ln w_{i,j} + G_{i,j}\ \Big]\,}$$

```python
# sampler.py:874-891 逐行对应
:874  seed = seed.to(torch.uint64)
:875  col_indices = torch.arange(m, device=logprobs.device)
:876  hashed = murmur_hash32(seed, positions, col_indices)     # kernels/ops/sampling/murmur_hash.py:104
:880  x = hashed.to(torch.float64) / torch.iinfo(torch.uint32).max
:885  x.log_().clamp_(min=torch.finfo(x.dtype).min, max=-(2.0**-32)).neg_()   # → −ln U
:886  x.log_().neg_()                                                        # → −ln(−ln U) = G
:889  x.add_(logprobs.to(torch.float64))
:891  return torch.argmax(x, dim=1, keepdim=True)
```

**排除理由**：`sampler.py:877-878` 注释 *"it is **critical** to keep gumbel noise calculation in float64 to avoid numerical instability"*，而 **TileLang-Ascend 对 `float64` 零支持**（`tilelang/language/*.py` 中 `float64`/`double` 零命中；`T.tile.atomic_add` 支持 dtype 仅 `float16/float32/int16/int32/bfloat16`）。f32 下 $U \to 1$ 时 $1 - 2^{-32}$ 舍入为 `1.0`，$\ln 1.0 = 0$，clamp 上界失效 → $G = +\infty$。见 §10 R1。

#### Step 8 — Logprob 🔴 V01 不融合

```python
# sampler.py:646（因 :641 是 in-place div_，此处 logits 已是 z'）
logprobs = torch.log_softmax(logits, dim=-1)
```

$$\operatorname{lp}_{i,j} = \log\operatorname{softmax}(z')_{i,j} = \big(z'_{i,j} - m_i\big) - \ln s_i$$

若 `SGLANG_RETURN_ORIGINAL_LOGPROB`（`sampler.py:205-206`），在缩放**前**另存：

$$\operatorname{lp}^{\text{orig}}_{i,j} = z_{i,j} - \max_k z_{i,k} - \ln \sum_k e^{\,z_{i,k} - \max_k z_{i,k}}$$

**排除理由（精度硬门槛）**：`sampler.py:246-252` 注释原文——

> *"Deterministic inference must derive the returned logprobs from `F.log_softmax` — the same kernel prefill rescoring uses — not `log(softmax(x))` below: **the two disagree at ~1e-6 despite being mathematically equivalent, which breaks bitwise prefill/decode logprob alignment.**"*

即要求舍入顺序严格为 $(z'_j - m) - \ln s$，**禁止** $\ln\!\big(e^{z'_j-m}/s\big)$。TileLang 侧 `T.tile.ln` + `T.tile.exp` 的舍入行为与 CANN `log_softmax` 是否逐位一致**未验证**，V01 不冒险。佐证 `sampler.py:759-761`：*"we cannot use log_softmax directly... we'd better come up with a solution to use log_softmax"*。

### 3.6 三个化简定理（本设计的数学核心）

#### 定理 T1 — 三重掩码折叠为单阈值

top-k、top-p、min-p 三者均为「按值单调的阈值筛选」，且 `masked_fill_(..., 0.0)` 不改变其他元素的值，故**顺序施加等价于取阈值最大者**：

$$\mathcal{S}^{(k)} \cap \mathcal{S}^{(p)} \cap \mathcal{S}^{(\mu)} \;=\; \Big\{\, j : z'_{i,j} \ge \tau_i \,\Big\}, \qquad \tau_i = \max\!\big(\tau^{(k)}_i,\ \tau^{(p)}_i,\ \tau^{(\mu)}_i\big)$$

**top-p 阈值不受 top-k 屏蔽影响的证明**：`sampler.py:823` 先 `masked_fill_(top_k_mask, 0.0)`，`:826` 再 `cumsum`。被屏蔽位置 $t \ge k$ 的 $q_t = 0$，对 $l < k$ 的前缀和 $\sum_{t<l} q_t$ 无贡献 → $\tau^{(p)}$ 在 top-k 前缀上计算与在全序列上计算**结果相同**。Path 2 中 `npu_top_k_top_p` 内部同理。$\blacksquare$

**收益**：3 次全 vocab 掩码遍历（`:800/:801` + `:821/:823` + `:827/:828`）→ **1 次 `T.tile.compare` + 1 次 `T.tile.select`**。

#### 定理 T2 — min-p 阈值在 $z$ 域的闭式（无需 $s$）

由 $\max_j p_{i,j} = e^{m_i - m_i}/s_i = 1/s_i$（top-k 必保留最大值元素）：

$$p_{i,j} \ge \theta_i \iff \frac{e^{z'_{i,j}-m_i}}{s_i} \ge \frac{\mu_i}{s_i} \iff e^{z'_{i,j}-m_i} \ge \mu_i \iff \boxed{\ \tau^{(\mu)}_i = m_i + \ln \mu_i\ }$$

$s_i$ **完全消去**。$\mu_i = 0$ 时 $\tau^{(\mu)} = -\infty$（不屏蔽），需特判避免 $\ln 0$。

#### 定理 T3 — Gumbel-max 免除归一化与对数

对保留元素 $j$，$\ln w_{i,j} = z'_{i,j} - m_i - \ln s_i + \text{const}_i$，其中 $-m_i - \ln s_i$ 为**行内常量**：

$$y_i = \arg\max_j \big[\ln w_{i,j} + G_{i,j}\big] = \arg\max_{j \in \mathcal{S}_i} \big[\,z'_{i,j} + G_{i,j}\,\big]$$

**收益**：Pass 2 无需计算 `exp`、`div`、`log`，直接在 $z$ 域做 argmax → 只需 `LCG 随机` + `T.tile.add` + `T.tile.compare` + `T.tile.select` + `T.tile.wholereducemax(ReduceOrder="ORDER_VALUE_INDEX")`。被屏蔽元素以 $-\infty$ 参与 argmax 即自然排除。

> **注意**：T3 仅适用于 **V01-b 的非确定性采样**。它改变 RNG 流（LCG 替代 `torch.multinomial` 内部 RNG），故输出与参考实现**不逐位相同**，只能做**分布一致性**验证（§9.2）。

### 3.7 融合后目标公式（两遍）

**Pass 1（K1 `r4_topk_stats`，单次 GM 读）** —— 分块 $b = 0 \dots n_{\text{blk}}-1$：

```
初始化: m ← −inf,  s ← 0,  running_topk ← ∅

for b in 0 .. n_blk−1:
    z_ub ← GM[logits[i, b·BV : (b+1)·BV]]                        # T.copy
    # (a) online softmax 统计
    b_max ← wholereducemax(z_ub)                                   # T.tile.wholereducemax
    if b_max > m:  s ← s · exp(m − b_max);  m ← b_max
    e_ub ← exp(z_ub − m)                                           # T.tile.exp / add
    s ← s + wholereducesum(e_ub)                                   # T.tile.wholereducesum
    # (b) 块内 top-K_MAX（升序归并）
    T.tile.topk(local_dst, z_ub, K_MAX, actual_num_b)              # [val,idx] 降序对
    local_val ← gather_mask(local_dst, "P0101")                    # 取值（弃 idx）
    merged ← concat(running_val, local_val)                        # 2·K_MAX
    T.tile.topk(running_new, merged, K_MAX, 2·K_MAX)
    running_val ← gather_mask(running_new, "P0101")

# 收尾（K_MAX ≤ 1024，UB 内标量/向量循环）
τ_k ← (k_i ≤ K_MAX) ? running_val[k_i − 1] : −inf
τ_μ ← need_min_p ? m + ln(μ_i) : −inf                              # 定理 T2
# top-p：在有序 top-K_MAX 列表上求前缀和
prefix ← 0;  τ_p ← −inf;  L ← −1
for l in 0 .. K_MAX−1:
    if prefix > p_i: break                                         # 判据同 sampler.py:827
    τ_p ← running_val[l];  L ← l
    prefix ← prefix + exp(running_val[l] − m) / s
valid[i] ← (L ≥ 0) ? 1 : 0                                         # 降级守卫
τ[i] ← max(τ_k, τ_p, τ_μ);   m[i] ← m;   s[i] ← s
```

**Pass 2（K2 `r4_mask_sample`，单次 GM 读 + 单次 GM 写）**：

```
# V01-a：输出屏蔽后概率
for b in 0 .. n_blk−1:
    z_ub ← GM[logits[i, b·BV : (b+1)·BV]]
    keep ← compare(z_ub, τ_i, "GE")                                # T.tile.compare
    e_ub ← exp(z_ub − m_i)                                         # 仅 keep 处有效
    w_ub ← select(keep, e_ub / s_i, 0.0)                           # T.tile.select
    GM[w_out[i, b·BV : ...]] ← w_ub                                # T.copy

# V01-b：额外融合 Gumbel-max（定理 T3），免除 w_out 写回
    g_ub ← −ln(−ln(clamp(lcg(rand[i,b]) · 2^-31, 2^-24, 1−2^-24)))
    cand ← select(keep, z_ub + g_ub, −inf)
    (blk_max, blk_arg) ← wholereducemax(cand, ReduceOrder="ORDER_VALUE_INDEX")
    if blk_max > best_val: best_val ← blk_max; best_idx ← b·BV + blk_arg
tok_out[i] ← best_idx
```

**遍历次数对比**：

| 路径 | 现状全 vocab 遍历 | 本设计 | 削减 |
|---|---|---|---|
| Path 2 / seed=None | 6 | **3 读 + 1 写**（V01-a）/ **3 读**（V01-b） | 1.5x / 2x |
| Path 3 / seed=None | ≈13 + O(V log V) sort | **3 读 + 1 写** | **≈3.3x + 消除排序** |
| Path 2 / seed≠None | ≈14（含 fp64 ×2 体积） | 不支持，回退 torch | — |

> ⚠️ 修正早期评估：曾估计「~9x 访存削减」，实际**上限约 2x~3.3x**（早期把 fp64 升位误当无条件开销，且未计入 `npu_top_k_top_p` 内部遍历）。**最大确定性收益是消除 Path 3 的 O(V log V) 全 vocab 排序与 `torch.cumsum`。**

---

## 4. 架构设计

### 4.1 逻辑视图

| 模块 | 职责 | 核心文件 |
|------|------|---------|
| **kernel** | TileLang kernel 定义（K1/K2），`T.prim_func` + `T.Kernel(is_npu=True)` | `r4_fused_sampling.py` |
| **tiling** | 形状特化表、`BV`/`K_MAX` 选择、核数探测、JIT 缓存管理 | `r4_tiling.py` |
| **dispatch** | 后端注册、降级守卫（`valid==0` 回退 torch）、dtype/shape 校验 | `r4_dispatch.py` |
| **ref** | PyTorch 参考实现（逐位对拍金标准） | `r4_ref.py` |
| **test** | 数值对拍 / 分布一致性 / 边界 / 性能 | `tests/` |

**模块依赖**：

```
sglang sampler.py:126  use_ascend_backend
      │
      ▼
r4_dispatch.py  ──register_sampler_backend("ascend_tilelang")──▶  sampler.py:683
      │
      ├──▶ r4_tiling.py     选择 (V, K_MAX, BV, NEED_MIN_P) 特化 → JIT 缓存命中/编译
      │         └──▶ utils.detect_vec_core_num() / select_launch_block_num()
      │
      ├──▶ r4_fused_sampling.py   K1 → K2
      │
      └──▶ valid==0 ? ──▶ 回退 sampler.py:779 原生路径（降级守卫）
```

### 4.2 开发视图

```
sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/sample/r4/
├── __init__.py                     # 导出 r4_fused_sampling
├── r4_fused_sampling.py            # K1 r4_topk_stats + K2 r4_mask_sample
├── r4_tiling.py                    # 特化表 + BV/K_MAX 选择 + 核数探测
├── r4_dispatch.py                  # backend 注册 + 降级守卫 + 校验
├── r4_ref.py                       # PyTorch 参考实现（复刻 sampler.py:779-848）
├── r4_lcg.py                       # LCG 随机数（V01-b，参照 examples_experiment/random_1d/）
├── design.md                       # 本文档副本
└── tests/
    ├── test_r4_vs_torch.py         # 数值对拍（Path 2 快路径）
    ├── test_r4_vs_torch_path3.py   # 数值对拍（Path 3 降级路径，top_k>1024）
    ├── test_r4_distribution.py     # 分布一致性（χ² / KL 散度）
    ├── test_r4_boundary.py         # 边界：B=1、V=32000、k=1、k=K_MAX、p=1.0、μ=0
    ├── test_r4_guard.py            # 降级守卫触发验证
    └── bench_r4.py                 # msprof op + do_bench 双工具采性能
```

> ⚠️ 路径中 `sgl_kernel_npu` 出现两次（实际布局：`python/sgl_kernel_npu/{README.md, setup.py, sgl_kernel_npu/}`）。

### 4.3 运行视图

**数据流（单行 = 单任务，`row_id = cid * VEC_NUM + vid`）**：

```
GM  logits[B, V] ──┐
                   │ ① T.copy 分块搬入（BV 元素/次，n_blk 次）
                   ▼
UB  z_ub[BV] ──────┬─▶ ② wholereducemax → b_max（标量）
                   ├─▶ ③ exp(z_ub − m) → e_ub[BV]
                   ├─▶ ④ wholereducesum(e_ub) → 累加 s（标量）
                   └─▶ ⑤ T.tile.topk(z_ub, K_MAX) → local_dst[2·K_MAX]
                              │
                              ▼ ⑥ gather_mask 取值 → local_val[K_MAX]
                        merged[2·K_MAX] ← concat(running_val, local_val)
                              │
                              ▼ ⑦ T.tile.topk(merged, K_MAX) → running_val（跨块归并）
                   ⋮ （循环 n_blk 次）
                   ▼
UB  running_val[K_MAX] ──▶ ⑧ 标量循环求 τ_k / τ_p / τ_μ → τ（定理 T1/T2）
                   │
                   ▼ ⑨ T.copy 落 GM
GM  m[B], s[B], tau[B], valid[B]
                   │
    ═══════════════╪═══════════════ K1 / K2 边界（或单 kernel 内二次遍历）
                   ▼
GM  logits[B, V] ──┐ ⑩ 二次分块搬入（L2 命中率高：B=1 时 594 KB ≪ A3 L2）
                   ▼
UB  z_ub[BV] ──────┬─▶ ⑪ compare(z_ub, τ, "GE") → keep
                   ├─▶ ⑫ select(keep, exp(z_ub−m)/s, 0) → w_ub      [V01-a]
                   ├─▶ ⑫' lcg → g_ub; select(keep, z_ub+g_ub, −inf) [V01-b，定理 T3]
                   └─▶ ⑬' wholereducemax(ORDER_VALUE_INDEX) → 跨块 argmax
                   ▼
GM  w_out[B, V]（V01-a）  /  tok_out[B]（V01-b）
```

**执行流程**：

```
r4_fused_sampling(logits, top_ks, top_ps, min_ps, need_min_p_sampling)
     │
     ├──▶ r4_tiling: 查特化表 (V, K_MAX, BV, NEED_MIN_P)
     │         ├── 命中 TILELANG_CACHE_DIR → 直接加载 .so
     │         └── 未命中 → tilelang.jit 编译（bisheng dav-2201）→ 写缓存
     │
     ├──▶ select_launch_block_num(B, vec_core_num=48) → b_num
     │
     ├──▶ K1 r4_topk_stats  →  m[B], s[B], tau[B], valid[B]
     │
     ├──▶ torch.all(valid) ?
     │         ├── False ──▶ 🔴 回退 sampler.py:779 原生路径（整批降级）
     │         └── True  ──▶ 继续
     │
     ├──▶ K2 r4_mask_sample →  w_out[B,V]（V01-a）或 tok_out[B]（V01-b）
     │
     └──▶ V01-a: torch.multinomial(w_out, 1) → token_id[B]
```

### 4.4 用户视图

| 调用方式 | 说明 |
|---------|------|
| **SGLang backend 注册** | `register_sampler_backend("ascend_tilelang", factory)`（`sampler.py:683`），经 `get_exec().kernel.sampling_backend` 选中 |
| **直接 Python 调用** | `from sgl_kernel_npu.sample.r4 import r4_fused_sampling` |
| **torch.ops 注册（可选）** | 走 `examples/torch_tl_ascend/` 路径：`kernel.adapter.lib_generator.libpath` 取 `.so` → `torch.ops.sgl_npu.r4_fused_sampling` |
| **ACLGraph 入图** | 参照 `examples/aclgraph/rms_rope_aclgraph.py`，`torch.npu.NPUGraph` capture/replay；⚠️ JIT 编译必须在 warmup 期完成（`npu_cudagraph_backend.py:94-101` 两次 warmup） |

---

## 5. 实现方案

### 5.1 特化维度总览（TilingKey 等价机制）

TileLang 无 Ascend C 的 `TilingKey` / `ASCEND_TPL_SEL_PARAM`，等价机制是 **`@tl.jit` 的编译期常量参数 + Python 侧分派表**：

| 参数 | 类型 | 取值范围 | 说明 |
|-----|------|---------|------|
| `V` | int（编译期） | `{32000, 128256, 129280, 151936, 152064}` | vocab size，必须静态（`T.tile.topk` 要求 src 为 compile-time static shape） |
| `K_MAX` | int（编译期） | `{64, 128, 256, 512, 1024}` | top-k 归并缓冲上限 |
| `BV` | int（编译期） | `{2048, 4096, 8192}` | vocab 分块大小 |
| `NEED_MIN_P` | bool（编译期） | `{True, False}` | 是否计算 $\tau^{(\mu)}$ |
| `FUSE_GUMBEL` | bool（编译期） | `{True, False}` | V01-a / V01-b |
| `B` | int（**运行时**） | 1~1024 | **不进入特化键**，仅影响 `T.Kernel(b_num)` grid |
| `k_i, p_i, μ_i` | tensor（运行时） | — | 逐行参数，kernel 内读取 |

**特化选择表**：

| 特化 | 触发条件 | 参数 | 适用场景 |
|-----|---------|---------|---------|
| **S1 标准** | `k_i ≤ 256` 且 `V ≤ 152064` | `K_MAX=256, BV=8192` | 绝大多数 decode（top_k=20/40/50） |
| **S2 大 K** | `256 < k_i ≤ 1024` | `K_MAX=1024, BV=4096` | 保守采样配置 |
| **S3 小 vocab** | `V ≤ 32000` | `K_MAX=256, BV=8192` | Mixtral 等 |
| **S4 无 top-k** | `top_ks == TOP_K_ALL` 且 `top_ps < 1.0` | `K_MAX=1024, BV=4096` + **强制守卫检查** | 🔴 nucleus 可能 > K_MAX，高概率触发降级 |
| **S5 降级** | `k_i > K_MAX` 或 `valid == 0` | 回退 `sampler.py:779` | 兜底 |

### 5.2 参数结构（TilingData 等价）

Ascend C 的 `TilingData` 结构体在 TileLang 中等价于 **编译期常量 + kernel 形参**：

```python
# r4_tiling.py
@dataclass(frozen=True)
class R4Spec:
    V: int                  # 编译期
    K_MAX: int              # 编译期
    BV: int                 # 编译期
    NEED_MIN_P: bool        # 编译期
    FUSE_GUMBEL: bool       # 编译期
    n_blk: int              # 派生 = ceildiv(V, BV)
    b_num: int              # 派生 = select_launch_block_num(B, vec_core_num)

    @property
    def cache_key(self) -> tuple:
        return (self.V, self.K_MAX, self.BV, self.NEED_MIN_P, self.FUSE_GUMBEL)
```

```python
# 编译期常量表（r4_tiling.py）
VOCAB_SPECIALIZATIONS   = (32000, 128256, 129280, 151936, 152064)
KMAX_SPECIALIZATIONS    = (64, 128, 256, 512, 1024)
BV_DEFAULT              = 4096
DEFAULT_ASCEND_PASS_CONFIGS = {          # examples/xllm_kernels/utils.py:3-11
    "tl.ascend_auto_sync": True,
    "tl.ascend_memory_planning": True,       # ★ 自动 buffer 复用，消除中间物化
    "tl.ascend_auto_cross_core_sync": True,
    "tl.ascend_auto_cv_combine": True,
}
DEFAULT_ASCEND_BISHENG_ARCH = "dav-2201"     # examples/xllm_kernels/utils.py:12  ← A3
```

> 采用**原始字符串** pass config（而非 `tl.PassConfigKey.*`）以避免对 tilelang 导出时机的版本依赖，此为 `examples/xllm_kernels/utils.py:4-5` 的既有做法。

### 5.3 Kernel K1：`r4_topk_stats`

#### 5.3.1 触发条件
所有非贪心、非 simple 的 NPU 采样请求（Path 2 + Path 3 统一入口）。

#### 5.3.2 并行划分

```python
VEC_NUM = 2                                        # examples/xllm_kernels/fused_gdn_gating.py:26
MAX_VEC_CORE_NUM = detect_vec_core_num()           # utils.py:20，A3 默认 48
b_num = select_launch_block_num(num_batches=B, vec_core_num=vec_core_num)   # utils.py:37

with T.Kernel(b_num, is_npu=True) as (cid, vid):
    task_id = cid * VEC_NUM + vid                  # fused_gdn_gating.py:153 同款索引分解
    # 一行一任务：task_id ↔ row_id，无跨核归约
    for row_id in T.serial(rows_per_iter):         # fused_gdn_gating.py:131 同款分批
        ...
```

**关键设计：一行一任务** → 每任务独占一行的直方图/归并缓冲，**无跨核竞争**，因此不需要 `T.tile.atomic_add`（该 API 仅限 GM 目标，见 §10 R4）。参照 `examples/topk_selector/example_topk_selector.py:33` `row_id = (cid * VEC_NUM + vid) % B`。

#### 5.3.3 TileLang API 映射

| 计算步骤 | TileLang API | 参数签名 | A3 验证 | 约束说明 | 替代方案 |
|---------|-------------|---------|---------|---------|---------|
| 分块搬入 | `T.copy` | `(src_region, dst_ub)` | ✅ arch22 | GM→UB，32 B 对齐 | — |
| 行内最大值 | **`T.tile.wholereducemax`** | `(dst, src, mask, repeattimes, dstrepstride, srcblkstride, srcrepstride, ReduceOrder="ORDER_VALUE_INDEX")` | ✅ **arch22 专属** | docstring 明示 *"does not support pto target"* → 仅 A2/A3；`ORDER_VALUE_INDEX` 同时返回值与索引 | `T.reduce_max`（`language/reduce.py:42`）；⚠️ `T.tile.block_reduce_max` 已 `@deprecated()`（`ascend_tile.py:1853`），**禁用** |
| 逐元素 max | `T.tile.max` | `(dst, src0, src1)` | ✅ arch22 | ⚠️ **逐元素**非归约（`ascend_tile.py:1173-1174`）；`src1` 可为标量；dst 可 alias src0（in-place） | — |
| 指数 | `T.tile.exp` | `(dst, src0)` | ✅ arch22 | `unary_op(dst, src0, "exp")`（`ascend_tile.py:1255-1262`） | `T.tile.exp_experiment`（`:2981`） |
| 自然对数 | `T.tile.ln` | `(dst, src0)` | ✅ arch22 | `unary_op(dst, src0, "ln")`（`ascend_tile.py:1326-1333`）；用于定理 T2 的 $\ln\mu$ | — |
| 行求和 | **`T.tile.wholereducesum`** | `(dst, src, mask, repeattimes, dstrepstride, srcblkstride, srcrepstride)` | ✅ **arch22 专属** | `ascend_tile.py:1678` | `T.tile.sum_experiment(dst, src, sumParams)`（`:3230`，要求 `inner * sizeof(T)` 为 32 B 整数倍）；`T.reduce_sum`（`reduce.py:79`） |
| 减法 | `T.tile.sub` / `T.tile.add` | `(dst, src0, src1)` | ✅ arch22 | `add` 的 `src1` 可为负标量，实现减法 | — |
| 除法 | `T.tile.div` | `(dst, src0, src1)` | ✅ arch22 | — | `T.tile.reciprocal` + `mul` |
| **块内 top-K** | **`T.tile.topk`** | `(dst, src, K, actual_num, *, tmp=None)` | ✅ arch22 | `ascend_tile.py:599-630`。🔴 **`src` 必须 compile-time static shape**；🔴 **float32 输入会被 in-place 写入 `-inf` padding**（`actual_num`→`aligned_count`）→ 必须先 `T.copy` 到 UB _scratch_，**禁止直接传 GM 视图**；`dst` 需 ≥ `aligned_topk = ceil(2K / elems_per_block) * elems_per_block`，`elems_per_block = 32/sizeof(dtype)`；`actual_num` **可为 symbolic**（支持动态尾块）；`tmp` 为 UB scratch，缺省由编译器注入 | 分层归并（本设计采用）；🔴 `T.tile.sort`（`:434`）需 `dst ≥ 2*aligned_count`，V 规模下 UB 不足 |
| 取 (val,idx) 分量 | **`T.tile.gather_mask`** | `(dst, src, src1Pattern, *, tmp=None)` | ✅ arch22 | `ascend_tile.py:671`。已知 `"P1010"` 取 **idx**（`example_topk_selector.py:49`）；取 **val** 的 pattern 待 Phase-0 验证（推测 `"P0101"`） | 标量循环 `dst[i] = src[2*i]` |
| 类型转换 | `T.tile.cast` | `(dst, src, mode, count)` | ✅ arch22 | `ascend_tile.py:2076`；mode ∈ `{"CAST_NONE","CAST_RINT","CAST_ROUND","CAST_MODE_LOW2HIGH","CAST_MODE_HIGH2HIGH"}` | — |
| 比较 | `T.tile.compare` | `(dst, src0, src1, mode)` | ✅ arch22 | `ascend_tile.py:1979`；`src1` 可为标量；mode 含 `"GE"`/`"LT"` 等 | — |
| 条件选择 | `T.tile.select` | `(dst, selMask, src0, src1, selMode, *, tmp=None)` | ✅ arch22 | `ascend_tile.py:770`；`src1` 可为标量（如 `0.0` / `-inf`） | — |
| 循环 | `T.serial` / `range` | `(n)` | ✅ arch22 | `T.serial` 支持符号 trip count；`range` 需编译期常量 | `T.Pipelined(stage=N)`（用于 §8.2 双缓冲） |

#### 5.3.4 API 验证记录

| API 名称 | 源码路径 | 验证状态 | 备注 |
|---------|-------------|---------|------|
| `T.tile.wholereducemax` | `tilelang/language/ascend_tile.py:1644` | ✅ 已验证 | docstring 明示不支持 pto → **A3(arch22) 专属可用**；`ReduceOrder="ORDER_VALUE_INDEX"` |
| `T.tile.wholereducesum` | `ascend_tile.py:1678` | ✅ 已验证 | 同上，arch22 专属 |
| `T.tile.block_reduce_max` | `ascend_tile.py:1854` | ⚠️ **`@deprecated()`** | **禁用**，改用 `wholereducemax` |
| `T.tile.topk` | `ascend_tile.py:599` | ✅ 已验证 | 有完整 example（`examples/topk_selector/`）+ 精度对拍框架 |
| `T.tile.sort` | `ascend_tile.py:434` | ⚠️ 有条件 | 内部 `sort32` + `merge_sort`；`dst ≥ 2*aligned_count` → **V 规模不可用** |
| `T.tile.gather_mask` | `ascend_tile.py:671` | ⚠️ 待验证 | `"P1010"`=idx 已确认；取 val 的 pattern 需 Phase-0 实测 |
| `T.tile.exp` / `ln` / `div` / `mul` / `add` / `sub` / `abs` | `ascend_tile.py:1255/1326/…` | ✅ 已验证 | 均走 `unary_op` / 二元 op，arch22 可用 |
| `T.tile.compare` / `select` / `cast` | `ascend_tile.py:1979/770/2076` | ✅ 已验证 | `fused_gdn_gating.py:226-257` 有生产级用例 |
| `T.tile.atomic_add` | `ascend_tile.py:313` | 🔴 **不可用于本设计** | docstring：*"the destination **must be GM**"*，`:333-334` 对非 global scope 抛 `ValueError`；支持 dtype 仅 `float16/float32/int16/int32/bfloat16`（**无 float64**）→ **UB 内直方图方案被否决**，改用「一行一任务 + 私有 UB 归并缓冲」 |
| `T.cumsum` | `language/reduce.py:149` | 🔴 **不可用** | `src/op/reduce.cc:322` 对 `local.fragment` 直接 `LOG(FATAL) << "CumSum for fragment not implemented"`；`:326-336` 仅 `shared`/`shared.dyn` 走 `tl::CumSum2D<threads,…>` 的 **CUDA `call_extern`**（`threads = T.thread_bounds->extent` 为 GPU 语义）；`:339` 其他 scope `ICHECK(false)` → **top-p 前缀和改为 UB 内标量循环**（$K_{MAX} \le 1024$，代价可接受） |
| `float64` / `double` | `tilelang/language/*.py` | 🔴 **零支持** | 全量 grep 零命中；examples 中仅 `generative_recommendation/golden.py`（PyTorch 参考实现，非 kernel）→ **确定性路径（Step 7b）排除** |
| LCG 随机数 | `examples_experiment/random_1d/{design.md,random_1d.py}` | ⚠️ **experiment 成熟度** | `A=1103515245, C=12345`，int32 溢出截断（模 $2^{32}$），3 轮迭代，*"每个元素独立计算，无顺序依赖，支持完全并行"* → V01-b 需自行内联，用 `T.tile.{mul,add,bitwise_and,cast,div}` 实现 |

**验证检查清单**：
- [x] 已用 grep 搜索 API 全部变体（`^def .*(reduce|max|min|sum)`）
- [x] 已确认 API 在 arch22（A3）上可用（`wholereduce*` docstring 明示 pto 不支持 → 反向确认 arch22 支持）
- [x] 已确认 dtype 支持（fp32 全覆盖；**float64 零支持**）
- [x] 已确认参数签名与源码一致
- [x] 已确认 UB / tmp buffer / 对齐约束（`T.tile.topk` 的 `aligned_topk`、`sum_experiment` 的 32 B 倍数）
- [x] API 不可用时已确定替代方案（`atomic_add`→私有 UB；`cumsum`→标量循环；`float64`→排除确定性路径）
- [ ] **Phase-0 待实测**：`gather_mask` 取 val 的 pattern 字符串；`wholereducemax` 的 `mask/repeattimes/*stride` 具体取值；bf16 logits 输入支持

#### 5.3.5 数据流设计

见 §4.3 步骤 ①~⑨。

#### 5.3.6 内存管理

以 **S1 特化（`K_MAX=256, BV=8192`）** 为例，单任务（一行）UB 分配：

| Buffer | Shape | dtype | 大小 | 用途 |
|---|---|---|---|---|
| `z_ub` | `[BV]` = `[8192]` | fp32 | 32 KB | GM 分块搬入的 scratch（**必需**：`T.tile.topk` 会 in-place 写 `-inf` padding） |
| `e_ub` | `[BV]` | fp32 | 32 KB | `exp(z_ub − m)` 中间结果 |
| `local_dst` | `[2·K_MAX]` = `[512]` | fp32 | 2 KB | 块内 topk 输出（val,idx 交错） |
| `local_val` | `[K_MAX]` = `[256]` | fp32 | 1 KB | 提取出的值 |
| `running_val` | `[K_MAX]` | fp32 | 1 KB | 跨块归并的 running top-K |
| `merged` | `[2·K_MAX]` = `[512]` | fp32 | 2 KB | 归并输入 |
| `topk_tmp` | `[2·K_MAX]` | fp32 | 2 KB | `T.tile.topk` 内部 scratch（显式提供以控预算） |
| 标量 | `m, s, b_max, prefix, τ_k, τ_p, τ_μ, τ` | fp32 | < 1 KB | `T.alloc_local` |
| **合计** | | | **≈ 73 KB** | |

#### 5.3.7 UB 容量验证

> ⚠️ 所有 Buffer 之和必须 ≤ 实际可用 UB

| 平台 | 总 UB | 向量可用 | 系统保留 | 验证公式 |
|------|-------|---------|---------|---------|
| **DAV_2201（Ascend910B / Ascend910_9382 = A3）** | **192 KB** | **184 KB** | 8 KB | `sum(buffers) ≤ 184 * 1024 = 188416 B` |
| DAV_3510（Ascend950DT / Ascend910PR = A5） | 248 KB | 248 KB | 0 KB | *本设计不适用* |

192 KB 佐证：`examples/pos_embedding/RotaryPositionEmbedding/rope_half_interleaved.py:218` `UB_LIMIT = 196608  # 192KB`；`:206` `block_M * rope_dim * factor <= 192KB (UB capacity)`；`examples/topk_selector/example_topk_selector.py:127` `help="Matrix dimension N (max ~8192 for 192KB UB)"`；`examples/gqa_fwd_varlen/gqa_fwd_varlen.py:128` `UB: +16KB (132.8→148.8 KB < 192 KB limit)`。

**各特化的 Buffer 总和验证**：

| 特化 | `K_MAX` | `BV` | z_ub | e_ub | topk 相关 | **合计** | ≤184 KB |
|---|---|---|---|---|---|---|---|
| **S1 标准** | 256 | 8192 | 32 KB | 32 KB | 7 KB | **73 KB** | ✅ 余 111 KB |
| S1' 紧凑 | 256 | 4096 | 16 KB | 16 KB | 7 KB | **41 KB** | ✅ 余 143 KB |
| **S2 大 K** | 1024 | 4096 | 16 KB | 16 KB | 28 KB | **62 KB** | ✅ 余 122 KB |
| S2' 大 K 大块 | 1024 | 8192 | 32 KB | 32 KB | 28 KB | **94 KB** | ✅ 余 90 KB |
| S4 无 top-k | 1024 | 4096 | 16 KB | 16 KB | 28 KB | **62 KB** | ✅ |

> `e_ub` 可与 `z_ub` **alias**（`exp` 计算后 `z_ub` 不再需要）→ 再省 16~32 KB，由 `tl.ascend_memory_planning` 自动完成或手工复用。

🔴 **反面验证（为何不能用 `T.tile.topk` 直接处理全 vocab）**：`T.tile.topk` 要求 src 全量驻留 UB。V=152064 fp32 → `score_accum[V]` = **594 KB ≫ 184 KB**。`examples/topk_selector/example_topk_selector.py:127` 明确 *"max ~8192 for 192KB UB"*（其 UB 分配 `:35-39`：`x_ub[block_N]` + `score_accum[N]` + `topk_dst[2*top_k]` + `topk_index[top_k]` + `output_index[top_k]`，N=8192/top_k=2048 时约 80 KB + sort 内部 tmp 64 KB ≈ 144 KB）。**V 超出 UB 上限约 18 倍 → 必须分层归并。**

### 5.4 Kernel K2：`r4_mask_sample`

#### 5.4.1 触发条件
K1 返回且 `torch.all(valid) == True`。

#### 5.4.2 内存管理（S1 特化，`BV=8192`）

| Buffer | Shape | dtype | 大小 | 用途 |
|---|---|---|---|---|
| `z_ub` | `[8192]` | fp32 | 32 KB | 二次搬入 |
| `keep_ub` | `[8192]` | uint8 | 8 KB | `compare` 输出的掩码 |
| `w_ub` | `[8192]` | fp32 | 32 KB | V01-a：屏蔽后概率 |
| `g_ub` | `[8192]` | fp32 | 32 KB | V01-b：Gumbel 噪声（与 `w_ub` 互斥，可 alias） |
| `cand_ub` | `[8192]` | fp32 | 32 KB | V01-b：`z + G`（可 alias `z_ub`） |
| 标量 | `τ, m, s, best_val, best_idx, tie_cnt` | — | < 1 KB | |
| **合计（V01-a）** | | | **≈ 72 KB** | ✅ ≤ 184 KB |
| **合计（V01-b）** | | | **≈ 104 KB** | ✅ ≤ 184 KB |

#### 5.4.3 平票（tie）处理

`T.tile.compare(z_ub, τ, "GE")` 会保留**所有** $z_j = \tau$ 的元素，而 `torch.topk` 只保留恰好 $k$ 个。当存在重复值时保留集偏大，分布略有偏差。

**处理策略**（K2 内实现，成本 O(1) 额外遍历）：
```
n_gt ← count(z_j > τ)                     # 严格大于的元素数
n_eq_need ← k_i − n_gt                     # 还需从 z_j == τ 中取几个
# 对 z_j == τ 的元素按 index 升序保留前 n_eq_need 个（与 torch.topk 的 index 序 tie-break 一致）
keep_j ← (z_j > τ) OR (z_j == τ AND rank_eq(j) < n_eq_need)
```
其中 `rank_eq(j)` 为等值元素中的 index 序，通过 K2 分块循环内的**运行计数器**维护（块内用 `T.tile.compare` + `wholereducesum` 求块内等值个数，跨块累加标量）。

⚠️ fp32 连续 logits 下精确平票概率极低，但**上游屏蔽产生的 `-inf`/相同 padding 值会大量平票** → 必须实现。

### 5.5 K1 + K2 是否合并为单 kernel

| 方案 | 优点 | 缺点 | 决策 |
|---|---|---|---|
| **分离（K1 + K2）** | 各自 UB 预算宽松；K1 失败可提前守卫回退；便于分级对拍 | 2 次 launch；`logits` 读 2 遍 | ✅ **V01 采用** |
| 单 kernel 双遍历 | 1 次 launch；第二遍 GM 读大概率命中 L2（B=1 时 594 KB ≪ A3 L2） | UB 需同时容纳两阶段 buffer（≈ 130 KB，仍可行）；守卫逻辑复杂；调试困难 | V02 优化项 |

> **`logits` 二次读的 L2 收益**：B=1 时单行 594 KB，B=256 时 148.5 MB（超 L2）。故 L2 收益仅在小 batch 显著 → decode 小 batch 场景（R4 的主战场）恰好受益。

---

## 6. 设计决策与权衡

### D1 — 为什么用「分层 top-k 归并」而非「直方图阈值搜索」？

| 方案 | 否决/采纳理由 |
|---|---|
| 全 vocab `T.tile.topk` | 🔴 否决：需 `score_accum[V]` 全量驻留 UB，V=152064 → 594 KB ≫ 184 KB |
| UB 内直方图 + `atomic_add` | 🔴 否决：`T.tile.atomic_add` docstring *"destination **must be GM**"*，`ascend_tile.py:333-334` 对非 global scope 抛 `ValueError` |
| GM 直方图 + `atomic_add` | 🔴 否决：需额外 GM workspace + 原子冲突；且 bin 边界依赖 `z_min`/`m`，需额外遍历 |
| 阈值二分迭代 | 🔴 否决：每次迭代一遍 V，~24 次迭代 = 24 遍，远超收益 |
| **分层 top-k 归并** | ✅ **采纳**：仅用已验证原语（`T.tile.topk` + `gather_mask`）；UB 占用与 V **无关**（仅与 `K_MAX` 相关）；结果**精确**（无分箱近似）；顺带产出有序 top-K 列表，使 top-p 前缀和可直接在 $K_{MAX} \le 1024$ 的 UB 小数组上标量循环完成，**绕开 `T.cumsum` 的 Ascend lowering 缺失** |

### D2 — 为什么 top-p 前缀和用 UB 标量循环而非 `T.cumsum`？

`T.cumsum`（`language/reduce.py:149`）在 Ascend 后端**不可用**：`src/op/reduce.cc:320-341` 的 `CumSumOp::Lower` 中，`local.fragment` → `LOG(FATAL) << "CumSum for fragment not implemented"`；`shared`/`shared.dyn` → 生成 `tl::CumSum2D<threads, dim, reverse>::run` 的 `call_extern`，其中 `threads = T.thread_bounds->extent` 是 **GPU thread block 语义**（昇腾 Expert 模式为 `T.Kernel(block_num, is_npu=True) as (cid, vid)`，无 `threads` 概念）；其他 scope → `ICHECK(false)`。

而 D1 的分层归并已把前缀和的规模从 $V = 152064$ 降到 $K_{MAX} \le 1024$ → **UB 内标量循环 1024 次，在 Vector 核上代价可忽略**（对比全 vocab 遍历）。

### D3 — 为什么排除确定性路径（`sampling_seed ≠ None`）？

`sampler.py:877-878` 注释明示 Gumbel 噪声**必须** float64；而 TileLang-Ascend 对 float64 零支持（§5.3.4）。f32 下 clamp 上界 $-2^{-32}$ 失效（$1 - 2^{-32}$ 在 f32 舍入为 `1.0`，$\ln 1.0 = 0$，$G = +\infty$）→ 数值错误。

**权衡**：确定性采样主要用于 RL on-policy 训练与可复现实验，**在线推理主流量走非确定性路径**（`sampling_seed is None`，6 op）。V01 覆盖主流量，确定性路径保留 torch 实现（`sampler.py:810`），通过 `sampling_seed is not None` 前置判断直接旁路本算子。

### D4 — 为什么 logprob 不融合？

`sampler.py:246-252` 要求 logprob 逐位复现 `F.log_softmax`（~1e-6 差异即破坏 prefill/decode 对齐）。TileLang 侧 `T.tile.exp` / `T.tile.ln` / `T.tile.div` 的舍入行为与 CANN `log_softmax` kernel **未验证是否逐位一致**。

**权衡**：融合 logprob 的边际收益（省 1~2 遍全 vocab）远小于破坏确定性推理的风险。V01 保留 `sampler.py:646 torch.log_softmax`，并在 §12 迭代三作为独立验证项（先做逐位对拍实验，通过后再融合）。

### D5 — 为什么 temperature 缩放（`:641`）不并入 kernel？

`_forward_ascend_backend:641` 的 `logits.div_(temperatures)` 是 **in-place**，且 `:646` 的 `torch.log_softmax(logits)` **依赖缩放后的同一张量**。若并入 kernel，则 `:646` 需要额外的缩放张量或重复计算 → 反而增加访存。

**例外**：若 D4 的 logprob 融合在迭代三落地，则 temperature 可一并吸收（此时无外部依赖）。

### D6 — 为什么 V01-a 保留 `torch.multinomial`？

| 方案 | op 数 | 风险 | 决策 |
|---|---|---|---|
| **V01-a**：K1+K2 输出 `w[B,V]`，保留 `torch.multinomial` | 6 → **3**（K1 + K2 + multinomial） | 🟢 极低：`multinomial` 是 CANN 成熟算子，无需随机数 | ✅ **V01 必交付** |
| **V01-b**：K2 内融合 Gumbel-max（定理 T3），直接输出 `tok[B]` | 6 → **2** | 🟡 中：需 LCG（`examples_experiment/` 成熟度待定）；改变 RNG 流，只能做分布一致性验证 | ✅ V01 增量交付 |

分阶段可在**同一 kernel 骨架**上完成（`FUSE_GUMBEL` 编译期开关），V01-a 的对拍结果直接复用为 V01-b 的中间校验点。

### D7 — 为什么「一行一任务」而非「多核协作一行」？

- **无跨核归约** → 不需要 `atomic_add`（GM-only，§5.3.4）、不需要跨核 barrier
- **UB 私有** → `running_val` / `merged` 无竞争
- **与 `topk_selector` 一致**：`example_topk_selector.py:33` `row_id = (cid * VEC_NUM + vid) % B`，已验证形态
- 代价：$B < 48$ 时核利用率不足。缓解：`select_launch_block_num(num_batches=B, vec_core_num=48)`（`utils.py:37`）按 B 收缩 grid；decode 小 batch 场景下 R4 本身耗时极短，核利用率非首要指标

### D8 — 降级守卫（`valid`）为什么必需？

定理 T1 的 top-p 阈值 $\tau^{(p)}$ 是在 **top-$K_{MAX}$ 有序列表**上求前缀和得到的。当真实 nucleus 规模 $> K_{MAX}$（高温 + 平坦分布 + `top_k = TOP_K_ALL`）时，循环走完 $K_{MAX}$ 仍 `prefix ≤ p` → $\tau^{(p)}$ 被低估 → **保留集偏大，分布错误**。

**守卫**：`valid[i] = (L ≥ 0) ? 1 : 0`；host 侧 `torch.all(valid)` 为 False 时**整批回退** `sampler.py:779` 原生路径。代价是一次 `[B]` int32 的 D2H 同步（或 device 侧 branch）—— 相对正确性必要。

> 该场景在当前实现中同样会走 Path 3（`top_k = TOP_K_ALL` → `npu_top_k_top_p_eligible = False`），故**守卫触发时性能不劣于现状**。

### D9 — 为什么目标是 A3(arch22) 而非 A5(arch35)？

- TileLang-Ascend `README.md:52`：*"specifically tested and validated on the following NPUs: **A2 and A3**"*；A5 仅 `target="pto"` + camodel 仿真
- 本设计核心依赖的 `T.tile.wholereducemax` / `wholereducesum` docstring 明示 *"does not support pto target"* → **A5 上不可用**
- `CMakeLists.txt:7-15`：A3 = Ascend910_9382 → `arch22` → `DAV_2201`，UB 192 KB
- bisheng arch：`dav-2201`（`examples/xllm_kernels/utils.py:12`）

⚠️ 跨代际陷阱（Guide `:2330`）：*"A2 会在算子之间清理多余计数，而 Ascend 950 要求严格配对"* → MTE 事件漏配对在 A2/A3 能跑过、A5 会挂。本设计**不使用显式 MTE 事件配对**（依赖 `tl.ascend_auto_sync`），故不受影响；但若 §8.2 引入手工流水线，需重新评估。

---

## 7. 边界条件与异常路径

| # | 条件 | 行为 | 依据 |
|---|---|---|---|
| B1 | `B == 0` | 直接返回 `torch.empty((0,), dtype=int32)` | 参照 `sampler.py:170-171` 的 `_is_hip and logits.shape[0]==0` 处理；⚠️ `multi_layer_eagle.py:80-84` 记录 *"triton-ascend 尚不支持 zero-sized grid kernel launch abort"* → **必须 host 侧拦截，不可下发 kernel** |
| B2 | `is_all_greedy`（所有 `top_k ≤ 1`） | **不进入本算子**，走 `sampler.py:181-197` 的 `torch.argmax` | Path 0 已足够高效（1 遍） |
| B3 | `simple_sampling_case` | **不进入本算子**，走 `sampler.py:597-606` | 无 top-k/top-p/min-p，K1 退化为纯 softmax，收益不抵 launch 成本 |
| B4 | `sampling_seed is not None` | **不进入本算子**，走原生 Path 2/3 | D3：float64 不支持 |
| B5 | `top_ks[i] == TOP_K_ALL`（-1） | $\tau^{(k)} = -\infty$（不屏蔽）；**强制启用 D8 守卫** | `sampler.py:819-820` 语义：`top_ks.masked_fill_(topk_all_mask, V)` |
| B6 | `top_ps[i] == 1.0` | $\tau^{(p)} = -\infty$（不屏蔽） | nucleus 为全集 |
| B7 | `min_ps[i] == 0.0` 或 `need_min_p_sampling == False` | $\tau^{(\mu)} = -\infty$；**必须特判避免 $\ln 0 = -\infty$ 污染 `max`** | 定理 T2 |
| B8 | `k_i > K_MAX` | `valid[i] = 0` → 整批回退 | D8 |
| B9 | nucleus > `K_MAX` | `valid[i] = 0` → 整批回退 | D8 |
| B10 | `logits` 含 `-inf`（上游屏蔽） | 正常参与 `max`/`exp`（$e^{-\infty} = 0$）；但会**大量平票** → §5.4.3 tie 处理必须生效 | `sampler.py:137 sanitize_nan_logits` 已清 NaN，`-inf` 保留 |
| B11 | `logits` 含 NaN | 上游 `sanitize_nan_logits`（`:137`）已清洗；kernel 内**不做**二次防护 | 保持与参考实现一致 |
| B12 | 全行 $z_j$ 相同 | $m = z$，$s = V$，$p_j = 1/V$；top-k 取前 $k$ 个（tie-break by index） | §5.4.3 |
| B13 | $V$ 不在特化表 | host 侧抛 `NotImplementedError`，回退原生路径 | §3.4 |
| B14 | 尾块 `actual_num < BV` | `T.tile.topk(…, actual_num)` 支持 symbolic；🔴 float32 会 **in-place 写 `-inf` padding** → 因操作对象是 UB scratch `z_ub`（非 GM），**安全** | `ascend_tile.py:614-618` |
| B15 | JIT 缓存未命中 | 首次调用触发 bisheng 编译（秒级）→ **必须在 ACLGraph warmup 期完成** | `npu_cudagraph_backend.py:94-101` 两次 warmup |

---

## 8. 性能优化

### 8.1 并行策略

```python
VEC_NUM = 2                                   # 每 Cube 核配 2 个 Vector 核
MAX_VEC_CORE_NUM = detect_vec_core_num()      # A3 = 48（utils.py:20 默认值，运行时探测
                                              #  torch.npu.get_device_properties() 的
                                              #  vector_core_num / aiv_core_num / vec_core_num）
b_num = select_launch_block_num(num_batches=B, vec_core_num=48)   # utils.py:37

with T.Kernel(b_num, is_npu=True) as (cid, vid):
    task_id = cid * VEC_NUM + vid             # 2D 任务索引分解
    rows_per_iter = max(1, B // b_num)        # fused_gdn_gating.py:131
```

- **任务粒度**：一行 = 一任务，$B$ 行映射到 $\min(B, 48)$ 个 Vector 核
- $B \ge 48$：满核占用；$B < 48$：核数收缩（避免空转）
- **不使用 Cube 核** → 不触碰 `TL_ASCEND_AUTO_CV_SYNC` / `AUTO_CV_COMBINE`，规避 `examples/mhc_post/benchmark_zh.md` V0→V1 记录的 *"CV 同步 bug 阻止单 kernel 融合"*

### 8.2 流水线设计

| 手段 | 实现 | 预期收益 |
|---|---|---|
| **双缓冲 GM→UB** | `T.Pipelined(stage=2)` 包裹分块循环 | 隐藏 MTE2 搬运延迟；参照 `examples/mhc_post/benchmark_zh.md` V5（`T.Pipelined(stage=2)` 是 V0→V10 5.98x 的关键一步） |
| **UB buffer 复用** | `tl.ascend_memory_planning: True`（自动）；`e_ub` alias `z_ub`（手工） | 省 16~32 KB UB，允许更大 `BV` |
| **cast 融合** | `T.tile.cast` 与相邻 elementwise 合并 | 参照 mhc_post V6 |
| **L2 复用** | K2 二次读 `logits` 命中 L2（B=1 时 594 KB） | decode 小 batch 显著 |
| **手工 MTE 事件**（V02） | `utils.py:44-70` 的 `mte2_notify_v` / `v_wait_mte2` / `v_notify_mte3` / `mte3_wait_v` 宏，配合 `tl.ascend_auto_sync=False` | 更细粒度重叠；⚠️ 需严格配对（D9 跨代际陷阱） |

### 8.3 访存收益核算（B=256, V=152064, fp32，单位 MB）

| 阶段 | 现状 Path 2 | 现状 Path 3 | **本设计 V01-a** | **本设计 V01-b** |
|---|---|---|---|---|
| `npu_top_k_top_p` / `softmax` | 148.5 R + 148.5 W | 148.5 R + 148.5 W | — | — |
| `sort` | — | 148.5 R + 297 W（`probs_sort`+`probs_idx`） | — | — |
| `max` / `mul` / `lt` / `masked_fill_` | 4 × 148.5 R + 3 × 148.5 W | 5 × 148.5 R + 4 × 148.5 W | — | — |
| `cumsum` | — | 148.5 R + 148.5 W | — | — |
| `to(int32)` / `gather` | — | 297 R + 148.5 W | — | — |
| `multinomial` | 148.5 R | 148.5 R | 148.5 R | — |
| **K1**（1 读） | — | — | 148.5 R | 148.5 R |
| **K2**（1 读 + 1 写） | — | — | 148.5 R + 148.5 W | 148.5 R |
| **合计** | **≈ 1188 R + 743 W** | **≈ 1633 R + 1336 W** | **≈ 445 R + 149 W** | **≈ 297 R** |
| **总流量** | **1931 MB** | **2969 MB** | **594 MB** | **297 MB** |
| **削减比** | 1.0x | 1.0x | **3.25x** | **6.5x** |

> 注：现状栏为**保守下界**（未计 `npu_top_k_top_p` 内部遍历、`torch.arange` 临时张量、fp64 升位的 ×2 体积）。计入后实际削减比更高。

### 8.4 ACLGraph 入图

- 参照 `examples/aclgraph/rms_rope_aclgraph.py`（已验证 `torch.npu.NPUGraph` capture/replay）
- ⚠️ Guide `:2530` 明确 aclGraph **"不承载图编译和图优化功能"** → 真正的融合必须在单 kernel 内完成，ACLGraph 只消除剩余 launch 开销
- ⚠️ JIT 编译必须在 warmup 期完成（`npu_cudagraph_backend.py:94-101`：*"Two warmups so kernels are loaded and one-time setup is paid before capture"*）
- 验证方法：`ASCEND_GLOBAL_LOG_LEVEL=1` + `ASCEND_PROCESS_LOG_PATH=logs`，搜 host 日志 `capture model` 关键字（0=禁用 / 1=启用）
- 🔴 **D8 守卫的 `torch.all(valid)` 是 device→host 同步，会破坏图捕获** → 入图时需改为 device 侧 branch 或放弃守卫（改为 host 侧静态判定 `top_ks.max() ≤ K_MAX`）

---

## 9. 测试验证方案

### 9.1 数值对拍（分级）

| 级别 | 对象 | 金标准 | 判据 |
|---|---|---|---|
| **L0 中间量** | `m_out`, `s_out` | `logits.max(-1)`, `torch.logsumexp` | `allclose(rtol=1e-5, atol=1e-6)` |
| **L1 阈值** | `tau_out` | 从 `r4_ref.py` 的有序列表反推 | **逐位相同**（fp32 精确比较） |
| **L2 掩码** | `w_out` 的非零位置集合 | `sampler.py:795-801` 的 `min_p_mask` 结果 | **集合完全相同**（tie 处理生效验证） |
| **L3 概率值** | `w_out` 的非零值 | 参考实现 | `allclose(rtol=1e-5)` |
| **L4 token id** | `tok_out`（V01-a） | `torch.multinomial(w_ref, 1)` 同种子 | 分布一致性（§9.2），**非逐位** |

参照形态：`examples/topk_selector/example_topk_selector.py:57-121` 的 `ref_program` + `count_per_row_mismatches` + `count_total_mismatches` + `assert accuracy > 0.99`。

**参考实现** `r4_ref.py` 必须**逐行复刻** `sampler.py:779-848`（含 Path 2 与 Path 3 两条），而非重写数学公式——避免「参考实现与被测实现共享同一处误解」。

### 9.2 分布一致性（V01-b 专属）

因定理 T3 改变 RNG 流，token id 不可逐位对拍，改验分布：

| 检验 | 方法 | 判据 |
|---|---|---|
| 边缘分布 | 固定 logits，采样 N=100k 次，统计 token 频率 | $\chi^2$ 检验 $p > 0.01$ |
| 与参考的 KL | $D_{KL}(\hat p_{\text{ref}} \,\|\, \hat p_{\text{ours}})$ | $< 10^{-3}$ |
| top-k 覆盖率 | 采样落在真实 nucleus 内的比例 | $> 1 - 10^{-4}$ |
| 熵 | 采样经验熵 vs 理论熵 $-\sum \tilde p \ln \tilde p$ | 相对误差 $< 1\%$ |

### 9.3 边界用例

覆盖 §7 全部 B1~B15，重点：

```
B=1 / B=48（满核）/ B=1024
V=32000 / 128256 / 152064
k=1 / 2 / 64 / 1024 / TOP_K_ALL
p=1.0 / 0.95 / 0.5 / 极小（触发守卫）
μ=0.0 / 0.05 / 1.0
need_min_p_sampling = True / False
logits 全相同 / 全 -inf（除一个）/ 含大量平票 / 单峰极尖（k=1 即覆盖 p=0.999）
尾块 actual_num < BV（V 不整除 BV）
```

参照 harness 形态：`examples/cann-bench/sparse_flash_attention/test_sparse_flash_attention.py`（L0/L1/L2/Boundary 分层 + fp64 golden + `COVERAGE_CATEGORY = "Fusion"`）。

### 9.4 性能基准

| 指标 | 工具 | 目标 |
|---|---|---|
| kernel-only 时延 | `triton.testing.do_bench` + `msprof op` **双工具交叉验证** | K1+K2 < 现状 Path 2 的 1/2 |
| HBM 流量 | `msprof` memory 指标 | 与 §8.3 核算吻合（±15%） |
| 端到端 TPOT | SGLang `bench_serving` | decode TPOT 下降可测 |
| 核占用 / 流水线 | `msprof` + `examples/mhc_post/` 的分析方法 | Vector 核占用 > 80%（B ≥ 48） |
| JIT 编译耗时 | 首次调用计时 | 记录并纳入 warmup 预算 |

> 参照 `examples/mhc_post/benchmark_zh.md` 的方法论：Ascend 910B / CANN 9.0.0，`do_bench` 与 `msprof op` 双工具交叉验证，V0→V10 逐步记录每步优化的实测收益。

---

## 10. 风险评估与应对

| # | 风险 | 等级 | 影响 | 应对 |
|---|---|---|---|---|
| **R1** | **float64 零支持** → 确定性路径（`sampling_seed≠None`）无法实现 | 🔴 高 | 覆盖范围缩小到非确定性路径 | D3：V01 明确排除，`sampling_seed is not None` 前置旁路；V02 评估 f32 + 补偿算法（Kahan 求和 / 双 f32 拆分）能否满足 `:877-878` 的稳定性要求 |
| **R2** | **`T.cumsum` Ascend lowering 缺失**（`reduce.cc:322 LOG(FATAL)`；`:326-336` 走 CUDA `tl::CumSum2D`） | 🔴 高 | top-p 前缀和无原语 | D2：分层归并把规模降到 $K_{MAX}\le1024$，UB 标量循环替代 |
| **R3** | **UB 192 KB vs V=152064**（超 18 倍） | 🔴 高 | 无法全 vocab 驻留 | D1：分层 top-k 归并，UB 占用与 V 无关 |
| **R4** | **`T.tile.atomic_add` 仅限 GM** | 🟡 中 | 直方图方案不可行 | D1/D7：一行一任务，私有 UB，无跨核归约 |
| **R5** | **`T.tile.topk` 对 float32 in-place 写 `-inf` padding** | 🟡 中 | 污染输入 → logprob 错误 | §5.3.3：必须先 `T.copy` 到 UB scratch，**禁止传 GM 视图**；B14 已确认安全 |
| **R6** | **logprob 逐位对齐门槛**（`:246-252`，~1e-6 即破坏） | 🔴 高 | 破坏确定性推理 | D4：V01 不融合 logprob，保留 `torch.log_softmax` |
| **R7** | **nucleus > K_MAX** 时 $\tau^{(p)}$ 低估 | 🟡 中 | 分布错误 | D8：`valid` 守卫 + 整批回退（性能不劣于现状 Path 3） |
| **R8** | **tie-break 与 `torch.topk` 不一致** | 🟡 中 | 保留集偏大 | §5.4.3：`n_gt` / `n_eq_need` 运行计数器，按 index 升序保留 |
| **R9** | **JIT 重编译风暴**（连续批处理 → B 动态） | 🟡 中 | 首 token 延迟抖动 | §3.4：**B 不进入特化键**；特化数 ≤ 10；`TILELANG_CACHE_DIR` 持久化；生产禁用 `disable_cache()` |
| **R10** | **`gather_mask` 取 val 的 pattern 未知**（仅确认 `"P1010"`=idx） | 🟢 低 | 阻塞实现 | Phase-0 实测；兜底用标量循环 `dst[i] = src[2*i]` |
| **R11** | **`wholereducemax` 低层签名复杂**（`mask/repeattimes/dstrepstride/srcblkstride/srcrepstride`） | 🟡 中 | 调参成本 | Phase-0 参照 `examples/reduce/example_reduce_min.py`（A2/A3 可用）反推；兜底用 `T.reduce_max`（`reduce.py:42`）或 `T.tile.max` 树形归约 |
| **R12** | **包名冲突**：`tilelang-ascend/setup.py:31 PACKAGE_NAME="tilelang"` vs `sglang/python/pyproject.toml:84 tilelang==0.1.12` | 🔴 高 | 无法共存 | `pyproject_npu.toml` 与 `docker/npu.Dockerfile` 均无 `tilelang` → **NPU 独立镜像可行**；并在 `kernels/ops/attention/dsa/tilelang_kernel.py:46` 加 `is_npu()` 拦截（该文件用了 `T.alloc_barrier`/`pdl_trigger`/`pdl_sync`/`warp_reduce_sum`，tilelang-ascend 中不存在） |
| **R13** | **`random_1d` 在 `examples_experiment/`**，成熟度未定 | 🟢 低 | 仅影响 V01-b | V01-a 不依赖随机数；V01-b 自行内联 LCG 公式 |
| **R14** | **`block_reduce_max` 已 `@deprecated()`** | 🟢 低 | 误用风险 | §5.3.3 明确禁用，统一走 `wholereducemax` |
| **R15** | **无本地 NPU 硬件** | 🔴 高 | 无法实测验证 | 全部数值结论标注为「待上板验证」；Phase-0 优先在 A3 实机跑通 `examples/topk_selector/` + `examples/reduce/example_reduce_min.py` 建立基线 |

---

## 11. 交付件清单

**必需**：

```
sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/sample/r4/
├── __init__.py
├── r4_fused_sampling.py        # K1 + K2
├── r4_tiling.py                # 特化表 + BV/K_MAX 选择
├── r4_dispatch.py              # backend 注册 + 降级守卫
├── r4_ref.py                   # 逐行复刻 sampler.py:779-848
└── tests/                      # §9 全部用例
```

**V01-b 追加**：`r4_lcg.py`

**可选**：
- `CMakeLists.txt` + AOT 打包（走 `examples/torch_tl_ascend/` → `.so` → `torch.ops.sgl_npu.*`）
- `bench_r4.py` 性能报告
- ACLGraph 入图验证脚本

**文档**：本 `DESIGN.md` + `REQUIREMENTS.md`（可选）+ `PLAN.md`（迭代计划）

---

## 12. 迭代规划

| 迭代 | 目标 | 代码开发 | UT 开发 | ST 用例 |
|------|------|---------|--------|-------|
| **Phase 0**（3 天）<br>环境与原语验证 | 打通 A3 实机 + 消解 R10/R11 | 跑通 `examples/topk_selector/`、`examples/reduce/example_reduce_min.py`、`examples/softmax/example_online_softmax.py`；实测 `gather_mask` 取 val 的 pattern、`wholereducemax` 的 stride 参数 | — | — |
| **迭代一**（1 周）<br>K1 骨架 | 单特化（S1'：`K_MAX=256, BV=4096`）+ 单 dtype(fp32) | K1 `r4_topk_stats`：online softmax + 分层 topk 归并 + 定理 T1/T2 阈值合成 + `valid` 守卫 | L0（`m`/`s`）+ L1（`tau`）对拍 | L0 标准用例：B=64, V=152064, k=40, p=0.95, μ=0 |
| **迭代二**（1 周）<br>K2 + V01-a 打通 | 端到端可用（保留 `torch.multinomial`） | K2 `r4_mask_sample`（`FUSE_GUMBEL=False`）+ tie 处理 + `r4_dispatch` backend 注册 | L2（掩码集合）+ L3（概率值）对拍；§7 B1~B15 边界 | 多 shape（B∈{1,48,256}，V∈{32000,128256,152064}）+ Path 3（k>1024）对拍 |
| **迭代三**（1 周）<br>特化完备 + 性能 | 全特化表 + 双缓冲 + 性能达标 | `K_MAX∈{64..1024}` × `BV∈{2048,4096,8192}` 全特化；`T.Pipelined(stage=2)`；UB alias 优化 | 全覆盖 + 守卫触发验证 | §9.4 性能基准：K1+K2 < 现状 Path 2 的 1/2；HBM 流量与 §8.3 吻合 |
| **迭代四**（1 周）<br>V01-b Gumbel 融合 | 消除 `torch.multinomial`，6 op → 2 op | `r4_lcg.py` + `FUSE_GUMBEL=True` 分支（定理 T3） | §9.2 分布一致性（χ² / KL / 熵） | 端到端 TPOT 对比 |
| **迭代五**（可选）<br>ACLGraph + logprob | 入图 + 评估 logprob 融合 | ACLGraph capture 验证；logprob 逐位对拍**实验**（先验证再决定是否融合） | logprob 逐位一致性（若通过） | ACLGraph 开启下的 TPOT |
| **V02**（后续） | 确定性路径 + 单 kernel 双遍历 | f32 补偿算法评估（R1）；K1+K2 合并；device 侧守卫（免 D2H 同步） | — | — |

---

## 13. 附录：关键代码索引

### 13.1 SGLang 侧（被替换 / 参照）

| 路径 | 关键行 |
|---|---|
| `sglang/python/sglang/srt/layers/sampler.py` | **`:779-848`** R4 本体；`:623-647` NPU 入口；`:586-621` `_sample_from_logits`；`:126` backend 开关；`:140-235` `forward`；**`:246-252`** 精度门槛注释；`:850-892` `multinomial_with_seed`（Gumbel-max）；`:683/:697` backend 注册；`:75` 内置 backend 集合；`:131-139` `_preprocess_logits`；`:719-777` 非 NPU 参照实现；`:924` `top_p_normalize_probs_torch` |
| `sglang/python/sglang/srt/sampling/sampling_batch_info.py` | **`:245-247`** eligibility 判定；`:108` 字段定义；`:499` merge 收紧；`:117` temperatures |
| `sglang/python/sglang/srt/model_executor/model_runner.py` | `:79` import；`:652` `create_sampler()`；**`:1988`** 调用点；`:2036` `compute_logprobs_only` |
| `sglang/python/sglang/srt/layers/logits_processor.py` | `:430` `LogitsProcessor`；`:495` `forward`；`:202/:206` `LogitsProcessorOutput.next_token_logits`；**`:1131`** `logits = logits.float()`（fp32 佐证） |
| `sglang/python/sglang/kernels/ops/sampling/renorm_triton.py` | **`:62 _renorm_from_pivots`**（pivot/阈值法参照）；`:15 _mask_and_partial_sum_kernel`；`:39 _normalize_kernel`；`:92 top_p_renorm_probs_triton`；`:135 top_k_renorm_probs_triton` |
| `sglang/python/sglang/kernels/ops/sampling/top_p_renorm_triton.py` | `:15`、`:38` |
| `sglang/python/sglang/kernels/ops/sampling/murmur_hash.py` | `:7 rotl32`、`:17 fmix32`、`:30 murmur3_mix`、`:51 murmur_hash32_kernel`、`:104 murmur_hash32` |
| `sglang/python/sglang/srt/sampling/penaltylib/repetition_penalty.py` | **`:10`** `@torch.compile(dynamic=True, disable=_is_npu)`（上游退化点，V02 可扩展融合边界） |
| `sglang/python/sglang/srt/managers/scheduler_components/logprob_result_processor.py` | 下游 logprob 消费（受 R6 约束） |

### 13.2 sgl_kernel_npu 侧（对照物 / 落点）

| 路径 | 说明 |
|---|---|
| `sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/sample/argmax_softmax_prob.py` | ⭐ 最直接 Triton 对照物 |
| `.../sample/probability.py` | 概率计算对照物 |
| `.../sample/chain_speculative_sampling.py`、`tree_speculative_sampling_target_only.py`、`verify_tree_greedy.py` | 投机解码采样（SP1/SP2，共享原语） |
| `sgl-kernel-npu/csrc/` | 🔴 **33 个算子目录中零个 vocab 级采样算子**（全量 grep 验证）；唯一碰 logits 的是 `pytorch_extensions.cpp:99 apply_token_bitmask`（grammar 约束解码，语义不同） |

### 13.3 TileLang-Ascend 侧

| 路径 | 关键行 / 说明 | A3 |
|---|---|---|
| `tilelang-ascend/examples/topk_selector/example_topk_selector.py` | ⭐ **`:19-54` kernel 主体**（`T.tile.topk` + `gather_mask` + `cast` 完整用法）；`:32 with T.Kernel(b_num, is_npu=True) as (cid, vid)`；`:33 row_id = (cid*VEC_NUM+vid) % B`；`:35-39` UB 分配；`:48 T.tile.topk`；`:49 gather_mask("P1010")`；`:50 T.tile.cast`；`:57-121` 对拍框架（`:121 assert accuracy > 0.99`）；**`:127`** `N (max ~8192 for 192KB UB)` | ✅ |
| `tilelang-ascend/examples/softmax/example_online_softmax.py` | ⭐ 单遍历 online softmax（Step 2） | ✅ |
| `tilelang-ascend/examples/cross_entropy_loss/example_cross_entro.py` + `benchmark_summary.md` | logprob / log_softmax 数值路径（迭代五） | ✅ |
| `tilelang-ascend/examples/reduce/example_reduce_min.py`、`example_reduce_min_pipeline.py` | ⭐ **A2/A3 可用的 reduce 参照**（反推 `wholereduce*` 的 stride 参数） | ✅ |
| `tilelang-ascend/examples/xllm_kernels/fused_gdn_gating.py` | ⭐ **生产级形态范本**：`:26 VEC_NUM=2`；`:29 MAX_VEC_CORE_NUM`；`:115-117` block/task 计算；`:131 rows_per_iter`；`:152-153 T.Kernel(...) as (cid,vid)` + `task_id`；`:183-257` 纯 elementwise 链（`exp/mul/cast/axpy/abs/add/ln/compare/select/sigmoid`）；`BATCH_SIZE_SPECIALIZATIONS = tuple(range(2,49,2))`；`REF_CHECK_NUM_BATCHES_LIST = [1,16,275,4096,65536]` | ✅ |
| `tilelang-ascend/examples/xllm_kernels/utils.py` | ⭐ **`:3-11 DEFAULT_ASCEND_PASS_CONFIGS`**（原始字符串形式）；**`:12 DEFAULT_ASCEND_BISHENG_ARCH = "dav-2201"`**（A3）；**`:20 detect_vec_core_num(default=48)`**；`:37 select_launch_block_num`；`:44-70` MTE 事件宏 | ✅ |
| `tilelang-ascend/examples/aclgraph/rms_rope_aclgraph.py` | ACLGraph 入图范本（`torch.npu.NPUGraph`）；⚠️ `:257 tl.cache.clear_cache()` 仅演示用 | ✅ |
| `tilelang-ascend/examples/mhc_post/benchmark_zh.md` | ⭐ 方法论范本：V0→V10 演进、5.98x vs CANN、`do_bench` + `msprof op` 双工具交叉验证；V0→V1 记录 *"CV 同步 bug 阻止单 kernel 融合"*（本设计不用 Cube 故规避） | ✅ |
| `tilelang-ascend/examples/cann-bench/sparse_flash_attention/test_sparse_flash_attention.py` | 测试 harness 形态：fp64 golden + L0/L1/L2/Boundary 分层 + `COVERAGE_CATEGORY = "Fusion"` | ✅ |
| `tilelang-ascend/examples/torch_tl_ascend/` | AOT 化路径：`kernel.adapter.lib_generator.libpath` → `.so` | ✅ |
| `tilelang-ascend/examples_experiment/random_1d/{design.md,random_1d.py}` | LCG：`A=1103515245`、`C=12345`、int32 模 $2^{32}$、3 轮、元素独立可并行（V01-b） | ⚠️ experiment |
| `tilelang-ascend/tilelang/language/ascend_tile.py` | `topk:599`、`sort:434`、`sort32:1687`、`gather_mask:671`、`select:770`、`exp:1255`、`max:1173`、`min:1190`、**`wholereducemax:1644`**、`wholereducemin:1661`、**`wholereducesum:1678`**、`block_reduce_max:1854`（`@deprecated`）、`compare:1979`、`cast:2076`、`ln:1326`、`atomic_add:313`、`sum_experiment:3230`、`reduce_sum_experiment:3102` | — |
| `tilelang-ascend/tilelang/language/reduce.py` | `reduce_max:42`、`reduce_min:63`、`reduce_sum:79`、`reduce_absmax:119`、`cumsum_fragment:135`、**`cumsum:149`（🔴 Ascend 不可用）** | — |
| `tilelang-ascend/tilelang/language/allocate.py` | `alloc_ub:156` | — |
| `tilelang-ascend/src/op/reduce.cc` | **`:320-341 CumSumOp::Lower`**（R2 的证据：`:322 LOG(FATAL)`、`:326-336 tl::CumSum2D` CUDA `call_extern`、`:339 ICHECK(false)`） | — |
| `tilelang-ascend/tilelang/language/ascend.py` | `:26-27` `"e4m3_float8"` / `"e5m2_float8"` dtype 映射（本设计不用，M1 相关） | — |
| `tilelang-ascend/setup.py` | **`:31 PACKAGE_NAME = "tilelang"`**（R12 包名冲突证据） | — |
| `tilelang-ascend/README.md` | **`:52`** *"tested and validated on ... A2 and A3"*；`:179` Cube/Vector 经 GM/L2 交换 + Developer/Expert 模式说明 | — |
| `tilelang-ascend/docs/TileLang-Ascend Programming Guide.md` | `:2330` 跨代际 MTE 配对陷阱；`:2530` aclGraph 不承担图编译/优化 | — |
| ❌ `tilelang-ascend/examples/reduce/example_{col,row}_reduce_max_slice_buffer.py` | `:16 target="pto"` → **A5 专属，排除** | ❌ |
| ❌ `tilelang-ascend/examples/sort/example_merge_sort.py` | 非 PTO 但 `N=64` + `T.Kernel(1)` 单核全 UB 驻留 → **仅原语演示，V 规模不可用** | ⚠️ |

### 13.4 A3 硬件规格来源

| 项 | 值 | 来源 |
|---|---|---|
| SOC → arch 映射 | A2=Ascend910B\*、A3=Ascend910_9382 → `arch22`；A5=Ascend950\* → `arch35` | `sgl-kernel-npu/csrc/CMakeLists.txt:7-15` |
| bisheng arch | `dav-2201` | `tilelang-ascend/examples/xllm_kernels/utils.py:12` |
| UB 总量 / 向量可用 / 保留 | 192 KB / 184 KB / 8 KB | 设计模板 DAV_2201 行；佐证 `rope_half_interleaved.py:218 UB_LIMIT = 196608`、`example_topk_selector.py:127`、`gqa_fwd_varlen.py:128`、`cann-bench/transpose/transpose.py:345` |
| Vector 核数 | 48（默认值，运行时经 `torch.npu.get_device_properties()` 探测 `vector_core_num`/`aiv_core_num`/`vec_core_num`） | `utils.py:20` |
| `VEC_NUM` | 2（每 Cube 核配 2 Vector 核）→ Cube 核 24 | `fused_gdn_gating.py:26` |
| A3 支持验证状态 | ✅ 已实测验证（A2 + A3） | `README.md:52` |

---

## 修订记录

| 版本 | 日期 | 修改人 | 修改内容 |
|---|---|---|---|
| V01 | 2026-10-04 | — | 初版：A3(arch22) + TileLang-Ascend(`target="ascendc"`) 的 R4 融合采样链设计；提出定理 T1/T2/T3；确定分层 top-k 归并方案；明确排除确定性路径（float64 零支持）与 logprob 融合（逐位对齐门槛） |
