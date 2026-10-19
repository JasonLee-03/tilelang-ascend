# r4_fused_sampling 算子设计文档（rev3 — split-V 并行架构）

> **版本**：V01-a + V01-b + rev1(RNG) + rev2(tail 精度红线) + **rev3(split-V)**
> **rev3 触发原因**：`USER_DIRECTED_ENHANCEMENT`（**不是** `[DESIGN_ERROR]` 回退）
> **rev3 交付基准**：现行 kernel `examples/R4/r4_fused_sampling.py`，
> sha256 = `19e51e8d8ee126b920cf48c915d863a9be3a349e2c68cf728d35d482b48bcc57`
> （本 Stage 1 已用 `sha256sum` 复核，与调度指令一致）
> **上一版备份**：`examples/R4/history_version/design_rev2.md`
> （本 Stage 1 已用 `diff -q DESIGN.md history_version/design_rev2.md` 核实二者相同 ⟹ **干净重启确认**）

---

## 0. rev3 摘要（必读，先看这一节）

### 0.1 本次回退的性质

**这不是 `[DESIGN_ERROR]` 回退。** 现行 rev2 设计是**正确的**：四门禁全绿
（`all` = 74 PASS / 0 FAIL + 38 BOUND / 0 WARN、`stability` = 12、`stress` = 8/8、
`coverage` = 25/0/0），cann-bench 36/36 `mismatch_count = 0`；两轮 Stage 3 Subagent 均回答
`design_error: no`。本次是**用户指令驱动的设计增强**（trigger = `USER_DIRECTED_ENHANCEMENT`）：
rev2 设计存在**并行度天花板**（`grid = ceildiv(B, VEC_NUM)`，vocab 维在 block 内串行扫描），
需架构级增强才能突破。

### 0.2 用户已确认的 4 项硬约束（本设计全部满足，未用默认值绕过任何一项）

| # | 约束 | 本设计的落实 | 落点 |
|---|------|-------------|------|
| 1 | **继续 Developer 模式** | 全部 4 个新 kernel 仍为 `alloc_shared` + 零手写同步 + `R4_PASS_CONFIGS` 四项全开。**未使用任何 Expert 原语**（`alloc_L1/ub/L0A/L0B/L0C`、`T.Scope`、`T.barrier_all`、`T.set_flag/wait_flag` 均不出现）。可行性用**编译实证**而非推理确认 | §2、§2.5、§13.6 |
| 2 | **必须 bitwise 等价** | 5 个输出与 `19e51e8d…` **逐位相同**。达成方式：**根本不切分 online-softmax 扫描**——softmax 侧代码逐条保留在 K1B 中，归约顺序不变；只切分 Sort/top-k 侧，而 top-256 归并链由「多重集 → 唯一有序值数组」性质保证 bitwise 不变 | §14（逐 pass 完整论证） |
| 3 | **双路分派，现行 kernel 逐字节保留** | 现行 `r4_topk_stats` / `r4_gumbel_sample` **一字不改**保留，作为大 `b_num` 路径 + bitwise 对照基准。新增 4 个 kernel（K1A/K1B/K2gA/K2gB）只在 host wrapper 内按判据分派 | §13.2、§13.4、§15.1 |
| 4 | **范围 = K1 + K2g** | K1 → K1A + K1B；K2g → K2gA + K2gB。**K2m 本轮不切**（实测仅占 mask 任务 3.1–10.2%） | §13.1、§13.5 |

> ⚠️ **一处对约束 3 字面表述的收紧（已给出证据，需 Orchestrator/用户知悉并确认）**：
> 约束 3 的字面判据是「`b_num >= 20` 走现行路径，`b_num < 20` 走 split-V」。
> 本设计采用**更强、且被约束 3 自身理由所蕴含**的判据：`S_cap = CUBE_CORE_NUM // b_num`，
> `S_cap < 2` 即走现行路径 ⟹ 实际边界是 **`b_num > 10`（即 `B > 20`）走现行路径**。
> 理由（§5.8.3 定理 P2，含证明）：`b_num ∈ [11,19]` 时 `S=2` 会把 grid 推到 `>20` ⟹ 2 波，
> 而「2 波 × 半量块」与「1 波 × 全量块」的 Sort 工作量**完全相同**，只多一次 launch ⟹ 严格变慢。
> 这正是约束 3 引用 case15 论证的同一机制，只是边界点从 20 收紧到 10。
> **该收紧只减少 split 路径的适用范围，不改变「`b_num>=20` 逐字节走现行路径」的承诺。**

### 0.3 rev3 的核心结论一览

| 项 | 结论 | 证据强度 |
|---|---|---|
| Developer 模式下 split-V 可行？ | **可行** | ✅ **编译实证**：4 个原型 kernel 全部 `get_kernel_source()` 成功（§13.6） |
| 需要跨核归约原语 / grid barrier？ | **不需要** | ✅ 源码核实：`T.barrier_all` 是 **PIPE 级**（`tilelang/language/ascend.py:275-286`，`tl.ascend_pipe_barrier "ALL"`），非 grid 级；`tl.ascend_auto_cross_core_sync` 是 **Cube↔Vector** 同步（`src/transform/ascend_combinecv.cc:36-38`、`:43-58`），非 block 间。本设计**完全不用**任何跨核原语 —— 跨 split 归约靠**多 kernel 顺序 + GM workspace** 完成 |
| Orchestrator §3.2 的 Pass2/Pass4 落点？ | **答 D（其他形态）：Pass2/Pass4 都不需要存在** | 见 §16.2：不切分 softmax 扫描 ⟹ 无前缀最大值还原、无标量重放 |
| 新增 kernel 数 | **4**（K1A、K1B、K2gA、K2gB）；现行 K1/K2g/K2m **3 个保留不变** ⟹ 模块内共 7 个 `@tilelang.jit` | §13.1 |
| GM workspace | K1: `pv[B*S, K_MAX]` fp32；K2g: `pv_val[B*S]` + `pv_idx[B*S]` fp32。最大（case7, S=10）= 40960 + 160 + 160 = **41280 B ≈ 40.3 KB** | §4.8（精确公式 + 18-case 表） |
| UB 账目（**实测**，非 §4.5 模型） | K1A ≤ **106496 B (54.24%)**、K1B ≤ **89444 B (45.55%)**、K2gA ≤ **148420 B (75.59%)**、K2gB = **132 B (0.07%)**；上限 **196352 B**。现行 K1 = 171620 B (87.40%) ⟹ **K1 侧 UB 余量从 24732 B 增至 89856 B（3.6×）** | §4.5（解析生成源 `ascend_ub.GetWithOffset<T>(n,off)` 实测） |
| `padCount == 0` 在新分解下仍成立？ | **成立** | ✅ 编译实证：10 个 split-path case 的全部 TopK 调用点 `(K, repeatTimes, actual_num)` 均满足 `repeatTimes*32 == actual_num` ⟹ `padCount = 0`（§4.9 表） |
| UB→UB 拷贝 32B 对齐仍成立？ | **成立** | ✅ 编译实证：全部 `copy_ub_to_ub` 元素计数 ∈ {128,256,320,512,832,2048}，×4B ∈ {512,1024,1280,2048,3328,8192} B，全部 `%32==0`（§4.9 表） |
| `proto.yaml` 是否变更 | **不变**（文件保持原样，sha256 `41ab797002568e89fcc9a9eaa58be4f1cdfce20b6406abd0fdbf2ad11f7f8949`） | §15.1 |
| 分派到 split 路径的 case 数 | **6/18**：{3, 4, 7, 8, 12, 17} | §13.4（含 18-case 全表 + guard 推导） |
| 预期收益（**模型估计，非实测**） | fused 任务 geomean **1.15×–1.29×**、mask 任务 geomean **1.11×–1.20×**；单 case 最高 **fused 3.5×–4.2×（case7）**；**12 个非 split case 精确 1.000×（代码逐字节不变 ⟹ 零回退，构造性保证）** | §12.4（含对未测量参数 `a_launch` 的敏感性分析） |
| 主要风险 | ① `a_launch`（每次 kernel launch 固定开销）**未实测**，是小 case 收益的唯一未知量；② `F(K_MAX=128)` 固定成本**无法由现有数据分离**（§12.2 的 2 个数据点相互矛盾）；③ GM 读 1 趟 → 2 趟；④ JIT 特化键增加 `S` 维度 | §10.1 |

### 0.4 相对 rev2 的关键调整（一句话版）

> **只把 Sort/top-k 从串行主循环里搬出去做成 split 并行，其余一切（online-softmax 扫描、
> 尾块精度红线、epilogue、三阈值、守卫判据、契约）逐条不变。**

### 0.5 为什么不会再犯同一错误

rev3 **不存在「上一版错误」**（本次非 `[DESIGN_ERROR]` 回退，§0.1）。但 rev3 明确规避了
历史上已踩过的 4 类坑，逐条给出结构性（非人工审查性）保证：

| 历史坑 | 出处 | rev3 的结构性规避 |
|--------|------|------------------|
| 缺陷 A：`topk` 的 `count` 非 32 倍数 → masked `AscendC::Duplicate` 在 dav_c220 不生效 → 陈旧 UB 槽位被当真实候选（曾致 `tau_out` 返回 2.7e+19~4.5e+37） | rev2 §10.3 / `debug_log.md` | **D-SPLIT-BVALIGN**（split 边界按块对齐）⟹ `actual_num` 取值集合与现行**完全相同** `{BV, 2*K_MAX, TAIL_PAD}`，全部 `%32==0`。已对 10/10 split-path case **编译实证** `padCount==0`（§4.9） |
| 缺陷 B：UB→UB `T.copy` 对非 32B 倍数 count 静默截断到 8 的倍数（实测 313→{312}、261→{256..260}、511→{504..510}、257→{256}） | rev2 §10.3 | 同上 ⟹ 全部 `copy_ub_to_ub` 元素计数已编译实证为 8 的倍数（§4.9）。rev3 **新增的 1 元素拷贝全部是 GM↔UB**，不属该缺陷域（§3.2、N5） |
| 归约顺序改变 → `s` 变 ulp 级 → `tau_p` 前缀和在 `p_i` 边界跳变 → `tau` 变 → `tok_out`/`valid`（int32 精确匹配门禁）翻转 | Stage 3 round1 [#3] 风险 ②（`optimization_log.md`） | **D-SPLIT-SCOPE**：online-softmax 扫描**根本不切分**，K1B 中的 softmax 代码与现行**逐条相同**、块序相同 ⟹ `m/s` 的 bitwise 等价是**构造性的**，不存在舍入路径分歧（§14.1、§14.2） |
| Stage 3 round1 [#3] 曾写下「online-softmax 跨 split 合并满足结合律: `m=max(m_c); s=Σ s_c·exp(m_c−m)`」 | `perf_tuning/optimization_log.md` 迭代 3 | ⚠️ **该论断在数学上成立但在 fp32 下 bitwise 不成立**（Orchestrator §3.1 的裁定正确，§16.1 给出独立复核）。rev3 **不采用**该合并式，因此不受影响 |

---

## 1. 概述

### 1.1 算子名称

`r4_fused_sampling`（R4 融合采样链，V01-a + V01-b 超集）

### 1.2 功能描述

把 sglang `Sampler` 的「top-k / top-p / min-p 三重屏蔽 + 采样」链路融合为 TileLang kernel：

- **K1 `r4_topk_stats`**：单遍分块扫描 → online-softmax 统计量 `(m, s)` + 合成阈值 `tau`（定理 T1/T2）
  + 降级守卫 `valid`。
- **K2**（按 `FUSE_GUMBEL` 二选一）：
  - `r4_mask_sample`（K2m，V01-a）：第二遍 → 屏蔽后未重归一化概率 `w_out[B,V]`，host 保留 `torch.multinomial`。
  - `r4_gumbel_sample`（K2g，V01-b）：kernel 内 Gumbel-max（定理 T3）→ 直接产出 `tok_out[B]` int32。

**适用场景**：LLM decode 阶段的采样前处理，尤其**小 batch（`B ≤ 20`）+ 大 vocab（`V ≈ 152064`）**
—— 这正是现行设计核心利用率最低（case7 `B=4` 只用 2/20 组核心，roofline 达成率 0.67%）
而 rev3 收益最大的区间。

**rev3 变化**：K1 与 K2g 各增加一条 **split-V 并行实现路径**（K1A+K1B、K2gA+K2gB），
由 host wrapper 按 §13.4 判据分派；**对外接口、语义、输出、契约完全不变**。

### 1.3 数学公式

**定理 T1（三重屏蔽折叠为单阈值）** —— 沿用 rev2，未变：

```
S^(k) ∩ S^(p) ∩ S^(mu) = { j : z_ij >= tau_i },   tau_i = max( tau^(k)_i, tau^(p)_i, tau^(mu)_i )
```

**定理 T2（min-p 阈值在 z 域的闭式，`s` 完全消去）** —— 沿用 rev2，未变：

```
p_ij >= mu_i / s_i  <=>  exp(z_ij - m_i) >= mu_i  <=>  tau^(mu)_i = m_i + ln(mu_i)   (mu_i = 0 => -inf)
```

**定理 T3（Gumbel-max 免归一化）** —— 沿用 rev2/V01-b，未变：

```
y_i = argmax_{j ∈ S_i} [ z_ij + G_ij ],   S_i = { j : z_ij >= tau_i },   G_ij = -ln(-ln(u_ij))
x_ij     = (seed_i * PRIME_W + j) mod 2^32                     # PRIME_W = 0x9E3779B1
state_ij = lowbias32(x_ij)                                     # M1=0x7FEB352D, M2=0x846CA68B
u_ij     = clamp( fp32(state_ij) * 2^-32 + 0.5, 2^-24, 1-2^-24 )
tok_out_i = argmax_j (z_ij >= tau_i ? z_ij + G_ij : -inf)      # 最小 index tie-break
```

**三个阈值的求法**（全部在 K1 收尾、`K_MAX` 宽 UB 小数组上完成）：

| 阈值 | 表达式 | rev3 落点 |
|------|--------|-----------|
| `tau^(k)` | `run_v[k_i - 1]`（降序 top-`K_MAX` 列表第 `k_i` 项） | **K1B**（epilogue，代码不变） |
| `tau^(p)` | 降序列表上**严格前缀和（不含自身）**扫描，取最后一个 `prefix <= p_i` 的 `run_v[l]` | **K1B**（`T.serial(K_MAX)` 串行 scan，代码不变） |
| `tau^(mu)` | `m_i + ln(mu_i)` | **K1B**（代码不变） |

**⭐ rev3 新增的三条数学事实（split-V bitwise 正确性的基础，完整证明见 §14）**：

```
(M1) 前缀最大值的顺序无关性：max(m_0,...,m_b) 在 fp32 下与结合/交换顺序完全无关（max 无舍入）
(M2) top-K 值多重集确定性：top-K(∪_s M_s) 的【降序值数组】只由值多重集决定，与归并树形状无关
     依据：top-K(A ∪ B) = top-K( top-K(A) ∪ top-K(B) )，且降序排序一个值多重集的结果唯一
(M3) argmax-with-min-index 的可结合性：
     fold_{严格GT}( (v_j, j)_{j=0..n-1} ) = ( max_j v_j,  min{ j : v_j = max_j v_j } )
     对【连续区间】的分组折叠（先组内、再按组序）结果与不分组完全相同
```

⚠️ **(M2)/(M3) 的前提是输入不含 NaN**（NaN 的比较全为 False，破坏全序）。
本算子契约**不含 NaN**（rev2 §9.3：上游 `sanitize_nan_logits` 已清洗，NaN 属 L2 负向域）。
§14.6 明确记录该前提及其在 NaN 下的失效方式。

### 1.4 算法描述

**现行（rev2，保留为 fallback 路径与 bitwise 基准）**

```
K1 每行：m_run=-inf, s_run=0, run_v=[-inf]*K_MAX
  for b in 0..n_blk_full-1:                     # 串行扫描 V
    (a) online softmax:  m_blk=max(z_b); m_new=max(m_run,m_blk); s_run*=exp(m_run-m_new);
                         m_run=m_new; s_blk=Σexp(z_b-m_new); s_run+=s_blk
    (b) 分层 top-k 归并: l_val=top_K(z_b); run_v=top_K(run_v ‖ l_val)
  尾块（TAIL>0）：同 (a)(b)，但 softmax 用精确尺寸 [1,TAIL] 缓冲做全量 reduce，
                  top-k 源经 [1,TAIL_PAD] 的 -inf 填充缓冲（精度红线，逐字节保留）
  epilogue: q2=exp(run_v-m)/s; tau_k; tau_u; tau_p(串行前缀和); n_gt/n_eq; tau=max(3者);
            okp/tieok/vflag → m_out/s_out/tau_out/valid
K2g 每行：for b: hash→Gumbel→cand=(z>=tau)?z+G:-inf→(blk_max,blk_argmax)→严格GT跨块折叠 → tok_out
```

**rev3 split-V 路径（新增，仅在 §13.4 判据命中时启用）**

```
K1  = K1A（split 并行，只做 (b) 分层 top-k 归并，产出 per-split run_v_s → GM workspace pv）
    → K1B（每行串行：完整 (a) online softmax【代码逐条不变】+ 跨 split 归并 run_v_s
            + epilogue【代码逐条不变】→ m_out/s_out/tau_out/valid）
K2g = K2gA（split 并行：per-split (blk_max, blk_argmax) 严格GT折叠 → GM workspace pv_val/pv_idx）
    → K2gB（每行串行：按 split-id 升序严格GT折叠 → tok_out）
K2m = 不变（用户约束 4：本轮不切）
```

**为什么这样切（而不是切 softmax）**：实测 Sort/top-k 占 K1 的 **26.4%–78.8%**
（`perf_tuning/round2_item2/probe_perf.json`，§12.2），而 online-softmax 侧只占 21.5%（case7）。
切 Sort 捕获了绝大部分收益，且**不触碰任何浮点归约顺序** ⟹ bitwise 风险最低（§0.5 第 3 行）。

### 1.5 数据流图

```
                    ┌───────────── host wrapper  r4_fused_sampling ─────────────┐
                    │ C6/C7/C3/dtype 前置校验（不变） → §13.4 分派判据           │
                    │ （全 host 侧 Python int 算术，不读任何 tensor 元素值）     │
                    └──────┬────────────────────────────────────┬───────────────┘
                 use_split == True                       use_split == False
                 （6/18 case）                          （12/18 case，含 b_num>=20）
                          │                                       │
   ┌──────────────────────▼──────────────────────┐   ┌────────────▼────────────────┐
   │ SPLIT-V 路径（rev3 新增 4 kernel）          │   │ 现行路径（代码逐字节不变）  │
   │                                             │   │                             │
   │ K1A  grid = b_num*S  ──pv[B*S,K_MAX]──┐     │   │ K1  grid = b_num            │
   │      top-k only，S 路并行             │     │   │   softmax+topk+epilogue     │
   │                                       ▼     │   │      │ m_out s_out          │
   │ K1B  grid = b_num                           │   │      │ tau_out valid        │
   │      softmax(逐条不变) + 跨split归并        │   │      ▼                      │
   │      + epilogue(逐条不变)                   │   │ K2m grid=b_num  → w_out     │
   │      │ m_out s_out tau_out valid            │   │  或                          │
   │      ▼                                      │   │ K2g grid=b_num  → tok_out   │
   │ K2gA grid = b_num*S ─pv_val/pv_idx[B*S]─┐   │   └────────────┬────────────────┘
   │      gumbel argmax，S 路并行            ▼   │                │
   │ K2gB grid = b_num ── tok_out                │                │
   └──────────────────────┬──────────────────────┘                │
                          └──────────────────┬─────────────────────┘
                                             ▼
                    (m_out, s_out, tau_out, valid, w_out | tok_out)
                    —— 两条路径 bitwise 相同（§14 论证 + §9.2 L0-D1/L0-E1 实测断言）
```

---

## 2. 编程模式选型

### 2.1 模式结论

**Developer 模式（纯 Vector）+ Ascend 专属 `T.tile.topk` / `T.tile.*` 混合。**

**rev3 维持 Developer 模式不变**（用户硬约束 1）。**未使用任何 Expert 原语。**
rev3 全部 7 个 kernel 的形态都是：`T.alloc_shared` + `T.copy` + `T.tile.*` + `T.reduce_*`
+ `T.serial`/`T.If` + `R4_PASS_CONFIGS` 四项全开 + 零手写同步。

### 2.2 选型理由

沿用 rev2（用户指定 + 纯 Vector 算子 + 仓库既有范式）。rev3 新增两条：

1. **split-V 不需要 Expert 模式**。跨 split 归约的协调**完全由 host 侧多 kernel 顺序调用 +
   GM workspace 承担**，kernel 内部仍是「一任务一核、零跨核通信」的 Developer 形态。
   已核实：本设计不使用 grid barrier / semaphore / spin-wait / atomic —— 这些在 Developer
   模式下要么**不存在**（grid barrier，§2.5 源码核实），要么**语义不满足 bitwise**
   （atomic 累加顺序不定，§3.2 禁用表）。
2. **rev3 的可行性是编译实证的，不是推理的**。4 个原型 kernel 已在安装 wheel
   `0.1.4+d75300ed0a88b0f4b2eef8a62b43f1930fdae44b` 上全部 `get_kernel_source()` 成功，
   生成码的循环边界 / 分支 / 地址算式 / 对齐属性已逐条核对（§13.6）。

### 2.3 ⚠️ 关键澄清：`T.tile.topk` 在 Developer 模式下的定位（沿用 rev2，未变）

`T.tile.topk` / `T.tile.gather_mask` 是 Ascend 专属 `T.tile.*` 原语（`tilelang/language/ascend_tile.py`），
**不是** Expert 模式标志。Developer 模式的判定依据是 **buffer 分配方式 + 同步方式 + pass_configs**
（`tilelang-programming-model-guide` §1）：本设计全部 `T.alloc_shared`、零手写同步、四个
pass_config 全开 ⟹ Developer。仓库既有 Developer 示例大量混用 `T.tile.*`
（`examples/developer_mode/gelu_mul_developer.py:33-46`、
`examples/xllm_kernels/fused_gdn_gating.py:226-257`）。

### 2.4 模式影响

| 方面 | Developer 模式下的表现 |
|------|----------------------|
| Buffer 分配 | 全 `T.alloc_shared`（纯 Vector ⟹ 编译器推断为 `shared.ub`，`src/transform/ascend_infer_buffer_scope.cc:398-402`） |
| 同步 | 全自动（`tl.ascend_auto_sync`），零手写 barrier/flag |
| 内存规划 | 自动（`tl.ascend_memory_planning`）⟹ **UB 溢出走 `LOG(FATAL)` 硬失败而非静默溢出**（§7.3 安全网，rev3 必须保持开启） |
| Cube 核 | 不使用（生成码为 `KERNEL_TYPE_MIX_AIC_1_2` + 空 L0A/L0B 分配；Stage 3 round1 [#8] 已判定「无害空转」，不关闭） |
| **rev3 新增：kernel 数量** | 模块内 `@tilelang.jit` 从 3 个增至 **7 个**；单次算子调用最多 launch **4** 个（split 路径 + fused 任务：K1A→K1B→K2gA→K2gB） |
| **rev3 新增：JIT 特化键** | 增加 `S`/`NBS`/`S_TAIL`，但三者由 `(B,V,BV,CUBE_CORE_NUM)` **函数依赖**确定 ⟹ 同一设备上不增加独立特化维度（§4.6、§10.1 R-JIT2） |

### 2.5 ⭐ rev3 新增：Developer 模式下跨 split 协调的可行性论证（含被否决方案）

**问题**：split-V 需要把 S 个 split 的局部结果合并成每行一个结果。合并**必须按固定顺序**
（约束 2），因此需要某种「所有 split 都写完 → 再合并」的顺序保证。

**已核实的可用原语清单**（逐一读源码，非凭记忆；AGENTS.md 核心原则 1）：

| 原语 | 语义 | 能否用于跨 split 顺序保证 | 证据（已读源码） |
|------|------|--------------------------|------------------|
| `T.barrier_all()` | `tl.ascend_pipe_barrier "ALL"` —— **单核内**全流水 drain | ❌ 只同步本核的 Scalar/Vector/Cube/MTE 流水，**不跨 AI Core** | `tilelang/language/ascend.py:275-286`；docstring 原文「all instructions in all pipelines (Scalar, Vector, Cube, MTE, etc.) issued before this barrier are completed before any subsequent instructions are executed」—— 全文无跨核语义 |
| `T.pipe_barrier(pipe)` | 同上，限定单 pipe | ❌ 同上 | `ascend.py:289-300` |
| `tl.ascend_auto_cross_core_sync` | **Cube ↔ Vector** 核间同步（同一 AI Core 内的 AIC/AIV） | ❌ 名字里的 "cross core" 指 **AIC↔AIV**，不是 block 间 | `src/transform/ascend_combinecv.cc:36-38`（`ascendAutoCrossCoreSync = "tl.ascend_auto_cross_core_sync"`）+ `:43-58` `struct CrossCoreSyncPoint { int scope; // 0: cube, 1: vec … }` |
| `T.set_flag` / `T.wait_flag` | MTE 事件配对，**核内** | ❌ 且属 Expert 标志（rev3 不用） | `ascend.py`（`tl.ascend_set_flag` / `tl.ascend_wait_flag`） |
| `T.tile.atomic_add` | GM 原子加（Ascend DMA atomic add） | ⚠️ 存在且 Developer 可用（`examples/unsorted_segment_sum/unsorted_segment_sum.py:50`），但**累加顺序由硬件调度决定 ⟹ 非确定 ⟹ 直接违反约束 2**；且只有 `add`，无法表达 `max` / `argmax-min-index` | `tilelang/language/ascend_tile.py:215`（`_ATOMIC_ADD_V1_ERR`）、`:313-346`（docstring「Atomically add a local tensor tile into a GM destination tile」，无顺序保证条款） |
| grid-wide barrier / semaphore / spin-wait | — | ❌ **TileLang 前端未暴露** | `tilelang/language/*.py` 全文检索：`barrier_all` 是唯一 barrier 且为 PIPE 级；无 `grid_sync` / `semaphore` / `arrive` / `wait_grid` 任何命中 |

**⟹ 结论：kernel 内跨 block（跨 AI Core）顺序保证在 Developer 模式下不可得。**

**因此**：
- Orchestrator §3.2 的**选项 B**（Pass2/Pass4 并入 Pass3 的 kernel，由每行某一个 block 承担，
  需 grid 级同步 / semaphore）⟹ **在本项目当前 wheel 上不可实现**，否决。
- **选项 C**（并入 Pass1，仍需同步）⟹ 同理否决。
- **选项 A**（Pass2/Pass4 各作独立小 kernel）⟹ **可实现**，但需 +2 次 launch，
  且 Pass2/Pass4 的串行标量链本身有成本（§16.2 量化）。
- **本设计选择选项 D**：**让 Pass2/Pass4 根本不需要存在**（§16.2 详述）——
  把「必须串行的部分」（online-softmax 扫描 + epilogue + 跨 split 归并）整体留在一个
  `grid = b_num` 的 kernel（K1B / K2gB）里；该 kernel 内部**没有任何跨 block 依赖**，
  因此**不需要**任何跨核同步；跨 kernel 的顺序由 **host 侧顺序 launch 天然保证**
  （与 rev2 的 K1→K2 完全同一机制，rev2 §7.2 已记录：「kernel launch 天然串行化，
  中间经 GM 传递 `m/s/tau/valid`」）。

**这是本设计能在 Developer 模式内成立的根本原因**，也是它优于 4-pass 分解的关键：
4-pass 分解的 Pass2/Pass4 是**行内串行标量 pass**，若要与 Pass1/Pass3 同 kernel 就必须 grid 同步
（不可得），只能拆成独立 kernel（选项 A，+2 次 launch）；而本设计把串行部分与「跨 split 归并」
**合并进同一个 `grid = b_num` 的 kernel**，且因为 softmax 扫描未被切分，**串行标量重放根本不需要**。

---

## 3. API 映射设计

### 3.1 公式拆解

沿用 rev2 §3.1，未变（online softmax 5 步 + 分层 top-k 归并 4 步 + epilogue 6 步 + K2m 屏蔽 3 步
+ K2g hash/Gumbel/argmax 5 步）。**rev3 只改变这些步骤落在哪个 kernel、由几个核并行执行，
不改变任何一个步骤的 API 选型与参数。**

### 3.2 TileLang API 映射（Developer 模式）

**rev2 §3.2 的整张映射表（20 行）在 rev3 中逐行有效，无任何 API 替换**，完整表见
`history_version/design_rev2.md` §3.2（其中 `reduce_max/reduce_sum`、`T.tile.topk`、
`gather_mask("P0101")`、`T.copy(pad_value=)`、`alloc_shared`、动态下标、`T.If/T.Then`、
`arith_progression`、`bitwise_rshift/lshift/and`、`clamp`、`ln`、`cast` 的证据链已在 rev2 完成
「速查表 → examples → 源码 → 官方测试」四重核实）。下表只列 rev3 因架构调整而**新出现**
或**落点改变**的行：

| 步骤 | TileLang API（Developer） | 参数要点 | 证据（rev3 新核实） | 落点 |
|------|--------------------------|---------|---------------------|------|
| **扁平化 2D grid** | `T.Kernel(b_num * S, is_npu=True) as (cid, vid)`；`task = cid*VEC_NUM + vid`；`row = task // S`；`sid = task % S` | `S` 是**编译期常量**（JIT 形参）⟹ `//`、`%` 都是常量除/模 | ⭐ **仓库既有范式**：`examples/unsorted_segment_sum/unsorted_segment_sum.py:66-68` `T.Kernel(num_segments * d_blocks, is_npu=True) as (cid, vid)` + `seg = cid // d_blocks; d_blk = cid % d_blocks`；另 `examples/pipeline/gemm_v0_pipeline.py:31`、`matmul_add_pipeline.py:33`（`T.Kernel(m_num*n_num)`）、`sparse_flash_attn_gqa_pipeline.py:79`（四维乘积）⟹ **一维 grid 承载多维任务映射是 `ascend-constraints.md §1`「不支持三维 Kernel」的官方替代方案**。**rev3 编译实证**：生成码 `K1A_case7.cc` 的地址算式为 `row = cid/5`、`sid = (cid%5)*2 + vid`、`b = (cid%5)*8 + vid*4 + lb`，与 `task=cid*2+vid; row=task//10; sid=task%10; b=sid*4+lb` **代数等价**（已逐项代入 `cid∈[0,20)×vid∈[0,2)` 核对，40 个任务与 `B*S=4*10=40` 双射） | K1A / K2gA |
| **静态 split 内循环 + 运行时守卫** | `for lb in T.serial(NBS)` + `b = sid * NBS + lb` + `if b < n_blk_full:` | `NBS = ceil(n_blk_total / S)` 是**编译期常量** ⟹ 满足「Ascend 只支持静态循环边界」 | `ascend-constraints.md §1`「动态循环边界不支持 → 预计算最大循环次数，用 `T.serial(max_iters)` + 条件判断」（官方替代方案原文）+ **rev3 编译实证**：`K1A_case7.cc:42` `for (int32_t lb = 0; lb < 4; ++lb)`（静态）、`:44` `if (((((cid % 5) * 8) + (vid * 4)) + lb) < 37)`（运行时守卫降级为**真实 C++ `if`**）。与 Stage 3 round2 已记录的证据同源：`perf_tuning/round2_item2/emitted_cc/K1_case15_pruned.cc:127-137` | K1A / K2gA |
| **尾块归属 hoist** | `if TAIL > 0:`（编译期）内 `if sid == S_TAIL:`（运行时标量比较），`S_TAIL = n_blk_full // NBS` 是编译期常量 | 避免为尾块再跑一整轮 `NBS` 次循环；`alloc` 与 `use` 同词法作用域（N6） | **rev3 编译实证**：该变体（`/tmp/opencode/refine3.py::k1a_v3`）对 case5/6/7/13/17（`TAIL ∈ {313,2048,512,256,825}`）全部编译通过；`if <TIR expr> == <Python int>` 与现行 `if row < B`（`r4_fused_sampling.py:124`）同形态 | K1A |
| **GM workspace 写（per-split 部分结果）** | `T.copy(run_v, pv[row * S + sid, :])` | UB→GM，`K_MAX` 个 fp32 = 1024 B（K=256）/ 512 B（K=128），均 32B 对齐 | **rev3 编译实证**：`K1A_case7.cc:84` `tl::ascend::copy_ub_to_gm<float, 256>(pv[((cid * 512) + (vid * 256))], run_v[0], 10240, 1, 256)` | K1A |
| **GM workspace 读（跨 split 归并）** | `for s in T.serial(S): T.copy(pv[row*S + s, :], pvb)` | GM→UB，同上尺寸；`s` 是**静态循环变量** ⟹ 归并顺序确定为 split-id 升序 | **rev3 编译实证**：`K1B_case7.cc` `copy_gm_to_ub<float,256>(pvb[0], pv[((cid*5120)+(vid*2560))+(s*256)], 10240, 1, 256, 0.0)`；循环 `for (int32_t s = 0; s < 10; ++s)`（静态） | K1B |
| **跨 split top-K 归并** | 与现行**块间归并完全同一组 API**：`T.copy(run_v, merged[:K_MAX])` + `T.copy(pvb, merged[K_MAX:])` + `T.tile.topk(m_dst, merged, K_MAX, 2*K_MAX)` + `T.tile.gather_mask(run_v, m_dst, "P0101")` | `actual_num = 2*K_MAX`（512 或 256），`repeatTimes = 2*K_MAX/32` ⟹ `padCount = 0` | **rev3 编译实证**：`TopK K=256 repeats=16 actual=512 aligned=512 padCount=0`（§4.9 全表 10/10 case） | K1B |
| **per-split argmax 部分结果写/读** | `T.copy(best_val, pv_val[row*S+sid : row*S+sid+1])` / `T.copy(pv_val[row*S+s : …+1], bv)` | 1 元素 fp32 **GM↔UB**。⚠️ **这不是 UB→UB** ⟹ 不触发 32B 截断红线（红线只约束 UB→UB 的 `AscendC::DataCopy`） | **rev3 编译实证**：`K2gA_case7.cc` `copy_ub_to_gm<float,1>(pv_val[((cid*2)+vid)], best_val[0], 40, 1, 1)`；`K2gB_case7.cc` `copy_gm_to_ub<float,1>(bv[0], pv_val[(((cid*20)+(vid*10))+s)], 40, 1, 1, 0.0)`。⚠️ 与 rev2 记录的实现层发现 5「1 元素 UB→UB `T.copy` 触发 aicore 异常 507015」**不冲突**：那条禁令针对 UB→UB，此处两端都在 GM | K2gA / K2gB |
| **K2gB 跨 split 严格 GT 折叠** | `with T.If(bv[0] > best_val[0]): with T.Then(): best_val[0]=bv[0]; best_idx[0]=bi[0]` | 与现行 K2g 的**块间归并逐字相同**（`r4_fused_sampling.py:547-550`），只是折叠对象从「块」变成「split」 | `T.If/T.Then` 用例 `examples/xllm_kernels/fused_gdn_gating.py:200-224`；**rev3 编译实证**：`K2gB_case7.cc` 仅 3 个 `PipeBarrier<PIPE_ALL>`、1 个运行时 guard、**UB = 132 B** | K2gB |

**明确禁用 / 排除的 API**：rev2 §3.2 的禁用表在 rev3 中**全部继续有效**
（`T.tile.wholereduce*` / `block_reduce_*` 已 `@deprecated()`；`T.cumsum` Ascend 不可 lowering；
`float64` 零支持；`tilelang/language/pto.py` 已废弃）。rev3 **新增两条禁用**：

| API | 状态 | rev3 禁用理由与证据 |
|-----|------|--------------------|
| `T.tile.atomic_add` | 🔴 **rev3 明确禁用**（rev2 只是「不用」） | 语义是 DMA 原子累加，**累加顺序由硬件调度决定 ⟹ 结果非确定 ⟹ 直接违反约束 2（bitwise）**；且只有 `add`，无 `max`/`argmax`。证据：`ascend_tile.py:313-346`；`examples/unsorted_segment_sum/unsorted_segment_sum.py:8` 自述「adaptive strategy: atomic_add (default)」用于**容许非确定顺序**的 seg-sum，与 R4 的 bitwise 要求本质不同 |
| grid barrier / semaphore / spin-wait | 🔴 **不可用（前端未暴露）** | §2.5 已逐条核实源码。⟹ 任何依赖「kernel 内等待其他 block」的方案在 Developer 模式下**不可实现**，Stage 2 **不得尝试**（含「用 `atomic_add` 造计数器 + 循环读 GM 自旋」——除顺序问题外，还有**波间死锁**：若生产者 block 落在第 2 波而消费者在第 1 波自旋，则永久挂死） |
| `T.cumsum`（用于前缀最大值 / 前缀和向量化） | 🔴 **rev3 再次确认禁用** | 本 wheel + 硬件 + 纯 Vector + `R4_PASS_CONFIGS` 下**不 lowering**：Stage 3 round1 实测 `InternalError`（TVM transform pass 阶段失败，`perf_tuning/optimization_log.md` [#4] 证据段）；源码侧 `src/op/reduce.cc:320-341` + `tl::CumSum2D` **仅存在于** `src/tl_templates/cuda/reduce.h:70`（无 Ascend 模板）。⟹ **rev3 的 `tau_p` 前缀和、以及任何前缀 scan 必须保持串行** |

### 3.3 计算伪代码（rev3 split-V 路径）

```python
# ============ host（编译期常量计算 + 分派；无任何 tensor 数据运算）============
def r4_fused_sampling(logits, top_ks, top_ps, min_ps, need_min_p_sampling,
                      K_MAX=256, BV=4096, rand=None, FUSE_GUMBEL=False):
    B, V = logits.shape
    <C6 / C7 / C3 / dtype 前置校验 —— 与现行 L586-604 逐字相同，不变>

    # ---- §13.4 分派判据（全为 Python int/float 运算，零 tensor 元素访问）----
    b_num       = ceildiv(B, VEC_NUM)
    n_blk_full  = V // BV
    TAIL        = V - n_blk_full * BV
    n_blk_total = n_blk_full + (1 if TAIL > 0 else 0)
    CUBE        = detect_vec_core_num() // VEC_NUM          # 运行时探测，绝不硬编码
    S_cap       = CUBE // b_num                             # 1-波上限（定理 P2）
    S           = max(1, min(S_cap, n_blk_total))
    NBS         = ceildiv(n_blk_total, S)
    S           = ceildiv(n_blk_total, NBS)                 # 回代收缩 ⟹ 无空 split（定理 P3）
    NBS         = ceildiv(n_blk_total, S)
    S_TAIL      = n_blk_full // NBS if TAIL > 0 else -1     # 尾块所属 split（编译期常量）
    use_split   = (S >= 2) and (n_blk_full >= 2) and (pred_saving > SAFETY * pred_cost)

    if not use_split:
        # ===== 现行路径：以下与 rev2 交付版逐字节相同（约束 3）=====
        m_out, s_out, tau_out, valid = r4_topk_stats(B, V, K_MAX, BV, need_min_p)(
                                           logits, top_ks, top_ps, min_ps)
        if not FUSE_GUMBEL:
            return m_out, s_out, tau_out, valid, r4_mask_sample(B,V,BV)(logits,m_out,s_out,tau_out)
        return m_out, s_out, tau_out, valid, r4_gumbel_sample(B,V,BV)(logits,tau_out,rand)

    # ===== split-V 路径 =====
    pv = torch.empty((B*S, K_MAX), dtype=torch.float32, device=logits.device)   # 仅分配
    r4_topk_stats_splitA(B, V, K_MAX, BV, S, NBS, S_TAIL)(logits, pv)
    m_out, s_out, tau_out, valid = r4_topk_stats_splitB(B, V, K_MAX, BV, S, need_min_p)(
                                       logits, top_ks, top_ps, min_ps, pv)
    if not FUSE_GUMBEL:
        w_out = r4_mask_sample(B, V, BV)(logits, m_out, s_out, tau_out)   # K2m 不切（约束 4）
        return m_out, s_out, tau_out, valid, w_out
    pv_v = torch.empty((B*S,), dtype=torch.float32, device=logits.device)
    pv_i = torch.empty((B*S,), dtype=torch.float32, device=logits.device)
    r4_gumbel_sample_splitA(B, V, BV, S, NBS)(logits, tau_out, rand, pv_v, pv_i)
    tok_out = r4_gumbel_sample_splitB(B, S)(pv_v, pv_i)
    return m_out, s_out, tau_out, valid, tok_out


# ============ K1A：split 并行，只做分层 top-k 归并 ============
@tilelang.jit(pass_configs=R4_PASS_CONFIGS)          # 无 out_idx：pv 是 host 分配的 workspace
def r4_topk_stats_splitA(B, V, K_MAX, BV, S, NBS, S_TAIL):
    n_blk_full = V // BV; TAIL = V - n_blk_full*BV; TAIL_PAD = ceil32(TAIL)
    b_num = T.ceildiv(B, VEC_NUM); AT = _aligned_topk(K_MAX)
    @T.prim_func
    def main(logits: T.Tensor([B,V],"float32"), pv: T.Tensor([B*S,K_MAX],"float32")):
        with T.Kernel(b_num * S, is_npu=True) as (cid, vid):
            task = cid*VEC_NUM + vid;  row = task // S;  sid = task % S
            if row < B:
                l_dst=alloc_shared([AT]); l_val=alloc_shared([K_MAX])
                merged=alloc_shared([2*K_MAX]); m_dst=alloc_shared([AT])
                run_v=alloc_shared([K_MAX])
                T.tile.fill(run_v, -inf)                       # N4：首轮归并前必须 -inf
                if n_blk_full > 0:                             # 编译期；alloc 与 use 同作用域（N6）
                    z_ub = alloc_shared([BV])                  # rank-1（N2：topk 源必须 rank-1）
                    for lb in T.serial(NBS):                   # 静态边界
                        b = sid*NBS + lb
                        if b < n_blk_full:                     # 运行时守卫 → 真实 C++ if
                            T.copy(logits[row, b*BV], z_ub)
                            T.tile.topk(l_dst, z_ub, K_MAX, BV)             # actual=BV, %32==0
                            T.tile.gather_mask(l_val, l_dst, "P0101")
                            T.copy(run_v, merged[:K_MAX]); T.copy(l_val, merged[K_MAX:])
                            T.tile.topk(m_dst, merged, K_MAX, 2*K_MAX)      # actual=2K, %32==0
                            T.tile.gather_mask(run_v, m_dst, "P0101")
                if TAIL > 0:                                   # 精度红线区段，逐字节沿用 rev2
                    ztp = alloc_shared([1,TAIL_PAD]); ztt = alloc_shared([TAIL_PAD])
                    if sid == S_TAIL:                          # 运行时标量比较（1 次，非 NBS 次）
                        T.copy(logits[row, n_blk_full*BV], ztp, pad_value=-inf)
                        T.copy(ztp[0,:TAIL_PAD], ztt)                       # TAIL_PAD*4 B, %32==0
                        T.tile.topk(l_dst, ztt, K_MAX, TAIL_PAD)            # padCount==0（结构性）
                        T.tile.gather_mask(l_val, l_dst, "P0101")
                        T.copy(run_v, merged[:K_MAX]); T.copy(l_val, merged[K_MAX:])
                        T.tile.topk(m_dst, merged, K_MAX, 2*K_MAX)
                        T.tile.gather_mask(run_v, m_dst, "P0101")
                T.copy(run_v, pv[row*S + sid, :])              # → GM workspace（无条件写满）


# ============ K1B：每行串行；softmax 扫描 + 跨 split 归并 + epilogue ============
@tilelang.jit(out_idx=[5,6,7,8], pass_configs=R4_PASS_CONFIGS)
def r4_topk_stats_splitB(B, V, K_MAX, BV, S, NEED_MIN_P):
    @T.prim_func
    def main(logits, top_ks, top_ps, min_ps, pv, m_out, s_out, tau_out, valid):
        with T.Kernel(T.ceildiv(B,VEC_NUM), is_npu=True) as (cid, vid):
            row = cid*VEC_NUM + vid
            if row < B:
                <全部标量/epilogue buffer 声明 + fill 初始化 + 3 个 per-row 参数 GM→UB
                 —— 与现行 r4_topk_stats L126-183 逐条相同>
                # ---- (a) online softmax：与现行 L191-261 逐条相同（含尾块红线区段）----
                if n_blk_full > 0:
                    z_ub=alloc_shared([BV]); e_ub=alloc_shared([1,BV]); m2d=alloc_shared([1,BV])
                    for b in T.serial(n_blk_full):
                        T.copy(logits[row, b*BV], z_ub)
                        T.reduce_max(z_ub, m_blk[0,0:1], dim=-1)  # rank-1 源 ⟹ 目的端传 BufferRegion
                        T.tile.max(m_old, m_run, m_blk)
                        T.tile.sub(tmp, m_run, m_old); T.tile.exp(tmp, tmp)
                        T.tile.mul(s_run, s_run, tmp)
                        T.tile.max(m_run, m_run, m_blk)
                        T.tile.broadcast(m2d, m_run)
                        T.tile.sub(e_ub, z_ub, m2d); T.tile.exp(e_ub, e_ub)
                        T.reduce_sum(e_ub, s_blk, dim=-1)
                        T.tile.add(s_run, s_run, s_blk)
                    # ⚠️ 此处【不含】现行的 (b) top-k 归并 —— 它已整体移到 K1A
                if TAIL > 0:
                    zt=alloc_shared([1,TAIL]); et=alloc_shared([1,TAIL]); mt=alloc_shared([1,TAIL])
                    <softmax 尾块 11 步，与现行 L250-261 逐条相同>
                    # ⚠️ 同样【不含】现行的尾块 top-k（已移到 K1A）
                # ---- (b') 跨 split top-K 归并，严格按 split-id 升序 ----
                pvb = alloc_shared([K_MAX])
                for s in T.serial(S):                            # 静态边界 S
                    T.copy(pv[row*S + s, :], pvb)
                    T.copy(run_v, merged[:K_MAX]); T.copy(pvb, merged[K_MAX:])
                    T.tile.topk(m_dst, merged, K_MAX, 2*K_MAX)
                    T.tile.gather_mask(run_v, m_dst, "P0101")
                # ---- (c) epilogue：与现行 L271-362 逐条相同，一字不改 ----
                <q2 / tau_k / tau_u / tau_p 串行前缀和 / okp / n_gt / n_eq / tau / tieok / vflag
                 / 4 个 GM 写>


# ============ K2gA：split 并行 Gumbel argmax ============
@tilelang.jit(pass_configs=R4_PASS_CONFIGS)          # 无 out_idx：pv_val/pv_idx 是 workspace
def r4_gumbel_sample_splitA(B, V, BV, S, NBS):
    n_blk = T.ceildiv(V, BV); BV2 = 2*BV
    @T.prim_func
    def main(logits, tau_in, rand,
             pv_val: T.Tensor([B*S],"float32"), pv_idx: T.Tensor([B*S],"float32")):
        with T.Kernel(T.ceildiv(B,VEC_NUM)*S, is_npu=True) as (cid, vid):
            task=cid*VEC_NUM+vid; row=task//S; sid=task%S
            if row < B:
                <全部 buffer 声明 + fill + seedP/idx 预处理
                 —— 与现行 r4_gumbel_sample L456-484 逐条相同>
                for lb in T.serial(NBS):                        # 静态边界
                    b = sid*NBS + lb
                    if b < n_blk:                               # 尾块自动含在内（K2g 无独立尾块路径）
                        <hash 3 轮 + Gumbel 5 步 + keep-mask + cand
                         + reduce_max/EQ/select/reduce_min
                         —— 与现行 L488-545 逐条相同，唯一差别见下方 ⚠️>
                        with T.If(blk_max[0] > best_val[0]):
                            with T.Then():
                                best_val[0] = blk_max[0]
                                best_idx[0] = blk_argmax[0] + b*BV
                T.copy(best_val, pv_val[row*S+sid : row*S+sid+1])
                T.copy(best_idx, pv_idx[row*S+sid : row*S+sid+1])
    # ⚠️ 唯一的语义要点：以下两处必须用【全局块号 b】而非局部 lb
    #      T.tile.arith_progression(state_ub, seedP_v[0] + b*BV, 1, BV2)   # hash counter 起点
    #      best_idx[0] = blk_argmax[0] + b*BV                               # 局部 index → 全局 index
    #    若误用 lb，则 G_ij 与 tok_out 全错（§14.5 的 bitwise 论证依赖这一点）


# ============ K2gB：每行串行，跨 split 严格 GT 折叠 ============
@tilelang.jit(out_idx=[2], pass_configs=R4_PASS_CONFIGS)
def r4_gumbel_sample_splitB(B, S):
    @T.prim_func
    def main(pv_val, pv_idx, tok_out: T.Tensor([B],"int32")):
        with T.Kernel(T.ceildiv(B,VEC_NUM), is_npu=True) as (cid, vid):
            row = cid*VEC_NUM + vid
            if row < B:
                bv=alloc_shared([1]); bi=alloc_shared([1])
                best_val=alloc_shared([1]); best_idx=alloc_shared([1])
                tok_i32=alloc_shared([1],"int32")
                T.tile.fill(best_val,-inf); T.tile.fill(best_idx,0.0); T.tile.fill(tok_i32,0)
                for s in T.serial(S):                           # 严格 split-id 升序
                    T.copy(pv_val[row*S+s : row*S+s+1], bv)
                    T.copy(pv_idx[row*S+s : row*S+s+1], bi)
                    with T.If(bv[0] > best_val[0]):             # 严格 GT ⟹ 平票保留更早（更小 index）
                        with T.Then(): best_val[0]=bv[0]; best_idx[0]=bi[0]
                T.tile.cast(tok_i32, best_idx, "CAST_ROUND", 1)
                T.copy(tok_i32, tok_out[row:row+1])
```

### 3.4 关键设计决策

沿用 rev2 §3.4 的 **D-GUARD / D-TIE / D-NORENORM / D-RAND / D-UMAP / D-CLAMP / D-TIE-ARGMAX**
（**全部不变**，分级判据与现行 `r4_fused_sampling.py:340-357` 逐字一致）。rev3 新增 5 项：

| 编号 | 决策 | 理由 |
|------|------|------|
| **D-SPLIT-SCOPE** | **只切分 Sort/top-k 侧（K1）与 Gumbel-argmax 侧（K2g），不切分 online-softmax 扫描** | ① 实测 Sort 占 K1 的 **26.4%–78.8%**（`probe_perf.json`，§12.2），是唯一值得切分的大头；② 不切 softmax ⟹ `m/s` 的 bitwise 等价是**构造性的**（代码逐条不变），不依赖任何舍入路径论证 ⟹ 约束 2 的风险从「需证明」降为「自动成立」；③ 不需要 Orchestrator §3.2 的 Pass2/Pass4，因此不需要 grid 同步（Developer 模式下不可得，§2.5）⟹ 约束 1 自动满足；④ `T.cumsum` 不可用（§3.2）⟹ 前缀 scan 本来也无法向量化，切分 softmax 只能落成串行标量 pass，收益有限 |
| **D-SPLIT-1WAVE** | `S_cap = CUBE_CORE_NUM // b_num`（**整数除，强制 split 后仍 ≤ 1 波**） | 定理 P2（§5.8.3，含证明）：超过 1 波时「多波 × 少量块」与「少波 × 多量块」的 Sort 工作量相同，只增加 launch ⟹ 严格无益。这正是约束 3 引用 case15 论证的机制，本设计把它一般化为公式 |
| **D-SPLIT-NOEMPTY** | `NBS = ceil(n_blk_total/S)` 后**回代收缩** `S = ceil(n_blk_total/NBS)` | 定理 P3（§5.8.3，含证明 + 反例）：朴素 `S = min(S_cap, n_blk_total)` 可能产生**空 split**（如 `n_blk_total=5, S=4 → NBS=2 →` 各 split 得 `[2,2,1,0]`）。空 split 会写出全 `-inf` 的 `run_v_s`（虽由 N4 fill 保证确定值，但白白浪费一个任务槽并可能触发未预期路径）。收缩后每 split ≥1 块，**结构性排除空 split** |
| **D-SPLIT-GUARD** | 除 `S>=2` 外，再加 `n_blk_full >= 2` **且** `pred_saving > SAFETY × pred_cost`（`SAFETY=1.5`，`a_launch_est=8.0us` 保守默认） | 模型显示（§12.4 敏感性表）：当 `a_launch ≥ 6us` 时，`n_blk_total=2` 的 case（5/11/13）会**回退到 0.60×–0.90×**。guard 把这 3 个 case 挡在现行路径上 ⟹ **18 case 全部 ≥1.000×，零回退（构造性保证）**。`a_launch` 一旦被 Stage 2 实测，只需更新该常量，guard 会自动放宽（无需改结构） |
| **D-SPLIT-BVALIGN** | **split 边界强制按块（= `BV` 元素）对齐**，不按元素任意切 | 保证每 split 内每个 top-k 源的 `actual_num` 仍是 `BV` / `2*K_MAX` / `TAIL_PAD`（三者 `%32==0`）⟹ `padCount==0` 与 UB→UB 32B 对齐**结构性保持**（§4.9 编译实证）。若按元素任意切，`actual_num` 会变成任意值 ⟹ 直接踩精度红线缺陷 A/B。这正是 Orchestrator §4 建议的「强制 split 边界按 BV 对齐」 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查（rev3 逐项重做）

| 约束（`ascend-constraints.md §1`） | rev3 是否触发 | 处理 / 证据 |
|------|--------------|------------|
| **不支持三维 Kernel**（`T.Kernel` 只接受一维 block 数） | ⚠️ **概念上需要 2D 任务空间 (row, split)** | ✅ **已解决**：用**一维 grid `b_num*S` + 整除/取模映射**，这是 `ascend-constraints.md §1` 给出的官方替代方案（`block_metadata` 预计算机制的简化形态；本例映射规则足够简单，无需 metadata 表）。仓库既有范式：`examples/unsorted_segment_sum/unsorted_segment_sum.py:66-68`。**rev3 编译实证通过**（§13.6） |
| **threads 参数限制（只支持 1 或 2）** | ❌ 不触发 | rev3 全部 kernel 沿用 `as (cid, vid)` 形态（`VEC_NUM=2`），**不指定 threads**。rev2 §6.5 已论证为何不用 `threads=2 + as (cid)`，该论证在 rev3 继续有效 |
| **动态循环边界不支持** | ⚠️ **split-V 强依赖循环边界** | ✅ **已解决**：`S`、`NBS`、`n_blk_full`、`n_blk_total`、`S_TAIL` **全部是 host 侧 Python int**，经 `@tilelang.jit` 形参传入 ⟹ 编译期常量。**rev3 编译实证**：生成码全部为 `for (int32_t lb = 0; lb < 4; ++lb)`、`for (int32_t s = 0; s < 10; ++s)`、`for (int32_t b = 0; b < 37; ++b)`、`for (int32_t l = 0; l < 256; ++l)` —— **零动态边界**。运行时可变的只有 `b = sid*NBS + lb`（用于 GM 地址与守卫比较），与现行 kernel 的 `b*BV` 同类 |
| **流水线不支持动态边界** | ❌ 不触发 | rev3 不使用 `T.Pipelined`（沿用 rev2 §6.3 结论：不启用） |
| **部分 GPU API 不可用** | ❌ 不触发 | rev3 未引入任何新 API；§3.2 新增行全部是既有 `T.Kernel`/`T.copy`/`T.serial`/`T.If` 的组合 |
| **L0C 容量上限（A2/A3 = 128KB）** | ❌ **不适用** | 纯 Vector 算子，**零 Cube 计算**（rev2 §8.1 已判定；Stage 3 round1 [#8] 实证生成码为 `KERNEL_TYPE_MIX_AIC_1_2` + 空 L0A/L0B 各 64KB）。无 `block_M × block_N` 概念 ⟹ 无 `block_M × block_N ≤ 16384` 约束、无 L0C 溢出风险 |
| **GEMM 非整除风险** | ❌ **不适用**（无 GEMM） | 但 rev3 **确有**两层非整除：`V % BV != 0`（尾块）与 `n_blk_total % S != 0`（split 不均）。两者处理见 §5.4 / §5.8 |
| **`T.copy` 不支持列方向 strided 切片** | ❌ 不触发 | rev3 全部切片形态：`logits[row, b*BV]`（2D 标量起点，与现行同）、`pv[row*S+sid, :]`（2D，最内维完整）、`pv_val[i : i+1]`（1D 区间）、`merged[:K_MAX]`/`merged[K_MAX:]`（1D 区间）。**无 4D+ 切片、无列方向 strided** |
| **Host 侧 `reshape`/`cat`/aclnn 隐式调用** | ❌ 不触发 | rev3 host 新增操作只有 `torch.empty(...)` **分配**（无数据搬运）+ Python int/float 算术。详见 §5.6 审计（3 个新增 `[HOST-METADATA-AUDIT]` 块，全部 `result: allow`） |
| **输出侧切片 + reshape 隐式 contiguous** | ❌ 不触发 | rev3 输出张量 shape/dtype 与 rev2 完全相同，无切片后 reshape |

#### 3.5.2 参考实现差异说明

rev2 §3.5.2（草稿 Expert 方案 → 本 Developer 方案）继续有效。rev3 无新的外部参考实现输入
（split-V 架构参考的是**本仓库内**的 `unsorted_segment_sum` 与 R4 自己的 round2 探针，非外部实现）。

#### 3.5.3 本项目同类实现参考（必须列出）

**rev2 §3.5.3 的整张表（14 行）在 rev3 中继续全部有效**（online softmax / topk / gather_mask /
动态下标 / 尾块 clamp / pass_configs / 核数探测 的证据链未变），完整表见
`history_version/design_rev2.md` §3.5.3。rev3 **新增 2 行**（split-V 架构的直接参考）：

| 文件路径 | 相似度 | 关键参考点（rev3 已逐一读取核实） |
|----------|--------|------------------------------|
| ⭐ **`examples/unsorted_segment_sum/unsorted_segment_sum.py`** | ⭐⭐⭐ **最高**（rev3 架构同构） | `:21-52` Strategy 1 `atomic_add_kernel`：`num_blocks = ceil(N/rows_per_core)`、`rows_per_vec = rows_per_core // VEC_NUM`、`for j in T.serial(rows_per_vec): row = cid*rows_per_core + j*VEC_NUM + vid` —— **一维 grid 承载多行任务的既有范式**；`:50` `T.tile.atomic_add(Output[seg_ub[0], d_blk*block_D], row_ub)`（rev3 **不用**，见 §3.2 禁用理由）；**`:60-68` Strategy 2 `seg_reduce_kernel`：`T.Kernel(num_segments * d_blocks, is_npu=True) as (cid, vid)` + `seg = cid // d_blocks; d_blk = cid % d_blocks`** ⟹ **这正是 rev3 K1A/K2gA 的 `(row, sid) = (task//S, task%S)` 扁平化 2D grid 的仓库既有先例**；`:163-179` 三策略 host 分派 + 「int64 not supported by atomic_add, cast to int32」的 dtype 约束记录 ⟹ **多路径按 host 判据分派也是既有范式**（对应 rev3 的双路分派） |
| **`examples/R4/perf_tuning/round2_item2/probe_k1_nosort.py` / `probe_k1_pruned.py`** | ⭐⭐⭐ **最高**（rev3 K1A/K1B 的直接原型） | `:39-136` `k1_nosort`：**把 per-block topk+merge 整段移除、online-softmax 路径完全不动**的 K1 变体 ⟹ **这正是 rev3 K1B 的形状**（softmax 保留、topk 移出）；`:104` 「keep the now loop-unused topk buffers alive (dead alloc_shared has no pre-allocated address → codegen fails)」⟹ rev3 必须遵守的死缓冲禁忌（N7）；`:12` `k1_pruned`：`with T.If(block_max >= tau_k_run)` 包裹 topk ⟹ **块内运行时条件分支确实 lower 成真实 C++ `if`** 的证据来源（生成码 `emitted_cc/K1_case15_pruned.cc:127-137`），rev3 的 `if b < n_blk_full` / `if sid == S_TAIL` 依赖同一机制 |

（另：`examples/R4/perf_tuning/round2_bv_probe/ub_probe.py` + `bv_ub_sweep.py` 是 rev3 §4.5
**UB 实测法**的来源，属工具而非算子参考，列于 §11.2。）

#### 3.5.4 源码级证据索引（rev3 新增核实项）

rev2 §3.5.4 的整张表（21 行）继续有效。rev3 新增：

| 结论 | 源码位置 | 核实方式 |
|------|---------|---------|
| `topk` 的 `max_actual_num = Σ src.shape`（**求和非求积**）⟹ rank-2 `[1,BV]` 源会给 `1+BV=4097` → `repeatTimes=129` → 覆盖 `129*32=4128 > 4096` 槽位，**静默读越界 32 元素** | `tilelang/language/ascend_tile.py:635-647`（`max_actual_num = 0` → `for dim in src.shape: … max_actual_num += dim_simplified.value` → `repeatTimes = (max_actual_num + 31) // 32`） | 本 Stage 1 **已读源码逐行确认**。⟹ rev3 K1A 的 topk 源仍必须是 **rank-1 `[BV]` / `[TAIL_PAD]`**（与现行一致，N2） |
| `topk` 要求 src **静态 shape**、`K >= 1`、`actual_num >= 1`、`K <= actual_num` | `ascend_tile.py:636-655`（四处 `raise ValueError`） | 已读源码。⟹ rev3 的 `S`/`NBS` 必须是编译期常量（否则 `src.shape` 含符号维 → 直接 `ValueError`）；且 `S >= 2` 时每 split ≥1 块（定理 P3）⟹ `actual_num >= BV >= K_MAX` 恒成立 |
| `padCount = repeatTimes*32 − actualCount`；`padCount > 0` 且 `actualCount % (32/sizeof(T)) != 0` 时走 **masked `AscendC::Duplicate`**（dav_c220 上不生效 → 未初始化 UB 槽位被当真实候选） | 仓库 `src/tl_templates/ascend/common.h:1715-1758`（`alignedCount = repeatTimes*blockSize`、`padCount = alignedCount - actualCount`、`if (padCount > 0)`、`inBlockOffset == 0` 的简单 Duplicate 分支 vs `masks[2]` 的 masked 分支）；**安装 wheel 同一逻辑在 `…/site-packages/tilelang/src/tl_templates/ascend/common.h:1663-1707`** | 已读**两处**源码并 `diff` 比对（两文件整体不同，但 `Sort` 该段逻辑逐行一致，仅行号偏移）。⟹ **`padCount == 0` 由 `actualCount % 32 == 0` 结构性保证，与 BV 无关**（`repeatTimes = ceil(actual/32)` ⟹ `actual%32==0 ⇒ aligned==actual ⇒ padCount=0`，整个 `if (padCount>0)` 块被跳过，masked 分支永不执行）——**Orchestrator §4 的该论断成立** |
| `ascend_memory_planning` 的**兄弟路径 `check_overflow = false`（静默溢出）**；R4 只因开了 `tl.ascend_memory_planning=True` 才走 `LOG(FATAL)` 硬失败 | `src/transform/ascend_memory_planning.cc:598`（`bool check_overflow = false; // reserve memory overflow check`）+ `:606`（`if (offset + buf_size > memory_limits_[scope] && check_overflow)`） | 已读源码。⟹ **rev3 全部 4 个新 kernel 必须继续传 `pass_configs=R4_PASS_CONFIGS`（含 `tl.ascend_memory_planning=True`）**，否则 UB 溢出会静默（§7.3、§10.1 R-MEM） |
| `T.reduce_*` 对 rank-1 源要求目的端传 **BufferRegion** | `tilelang/language/reduce_ascend.py:133-176`（`_validate_reduce_out_shape`：`expected_shapes = [reduced_shape]` (+`keepdim_shape`)；`if out_is_region and any(_shape_can_embed_expected_shape(out_extent, expected_shape) …)`） | 已读源码。⟹ rev3 K1B 的 `T.reduce_max(z_ub, m_blk[0, 0:1], dim=-1)`（rank-1 源）**必须保留 `[0,0:1]` 区域写法**；K2gA 的 `T.reduce_max(z_ub, blk_max, dim=-1)`（rank-1 源 + `[1]` 目的）沿用现行已验证写法 |
| `T.barrier_all` 是 **PIPE 级**（单核），非 grid 级 | `tilelang/language/ascend.py:275-286` | 已读源码 + docstring 全文 |
| `tl.ascend_auto_cross_core_sync` 是 **Cube↔Vector** 同步（`scope: 0=cube, 1=vec`） | `src/transform/ascend_combinecv.cc:36-38` + `:43-58` | 已读源码 |
| `T.tile.atomic_add` 只支持 local→GM，无顺序保证 | `tilelang/language/ascend_tile.py:215`（`_ATOMIC_ADD_V1_ERR`）、`:235-346` | 已读源码 + docstring |
| `T.cumsum` 在本 wheel/硬件的纯 Vector + `R4_PASS_CONFIGS` 下**不 lowering** | Stage 3 round1 实测：`perf_tuning/optimization_log.md`「[#4] tau_p 向量化 — 探测后确认不可行（证据）」：probe `/tmp/opencode/cumsum_probe.py` → `InternalError`（TVM transform pass 阶段失败）；源码侧 `src/op/reduce.cc:320-341` + `tl::CumSum2D` **仅** `src/tl_templates/cuda/reduce.h:70` | ⟹ **rev3 不得试图向量化 `tau_p` 前缀和，也不得用 cumsum 实现前缀最大值**。这是 D-SPLIT-SCOPE（不切 softmax）的独立佐证 |
| 一维 grid 可承载乘积任务数 | `examples/pipeline/gemm_v0_pipeline.py:31`（`T.Kernel(m_num * n_num)`）、`matmul_add_pipeline.py:33`、`sparse_flash_attn_gqa_pipeline.py:79`（四维乘积）、`unsorted_segment_sum.py:66` | 已读示例 |
| `tau_p` 串行循环在生成码中是 **3 × `PipeBarrier<PIPE_ALL>` / 迭代** | `perf_tuning/round2_item2/emitted_cc/K1_case15_real.cc:194-202`（`for (int32_t l = 0; l < 256; ++l) { PipeBarrier<PIPE_ALL>(); if (prefix.GetValue(0) <= p_i.GetValue(0)) { tau_p.SetValue(…); prefix.SetValue(…); } PipeBarrier<PIPE_ALL>(); PipeBarrier<PIPE_ALL>(); }`） | 已读生成码。⟹ **运行时 `PIPE_ALL` = 2 + 3×K_MAX = 770（K=256）/ 386（K=128）次全流水 drain，每行**。这是 §12.2 中 `F(K_MAX)` 固定成本的物理来源，也是 split-V **无法消除**的串行地板 |

#### 3.5.5 API 检索顺序合规记录（AGENTS.md 核心原则 1）

rev3 按「① `tilelang-api-best-practices` → ② `examples/` 同类实现 → ③ 源码」强制执行：

| 层级 | 查阅对象 | 命中 / 结论 |
|------|---------|-----------|
| ① 速查表 | `tilelang-api-best-practices`（`api-compute.md` / `api-kernel-memory.md` / `api-schedule-sync.md`） | rev2 已完成三项命中（`reduce_*` 签名、`alloc_shared` scope 推断、调度原语语义）。rev3 **未引入新计算原语** ⟹ 无需重新命中；`api-schedule-sync.md` 已确认无 grid 级同步原语可用 |
| ② examples | `examples/unsorted_segment_sum/unsorted_segment_sum.py` | ✅ **命中**：扁平化 2D grid（`cid // d_blocks`、`cid % d_blocks`）+ 多策略 host 分派 —— rev3 K1A/K2gA 的 grid 映射与双路分派均以此为范式 |
| ② examples | `examples/pipeline/gemm_v0_pipeline.py`、`matmul_add_pipeline.py`、`sparse_flash_attn_gqa_pipeline.py`、`examples/topk_selector/`、`examples/xllm_kernels/fused_gdn_gating.py` + `utils.py` | ✅ 命中：`T.Kernel(乘积)` 与 `detect_vec_core_num()` 运行时探测（rev3 用它派生 `CUBE_CORE_NUM = vec//VEC_NUM`，**不硬编码 20**） |
| ②→③ 升级 | **跨 block 同步 / grid barrier 是否存在** | ⚠️ 速查表与 examples **均未提供** ⟹ 下探 ③ 源码：`ascend.py:275-300`（`barrier_all` = PIPE 级）、`ascend_combinecv.cc:36-58`（cross-core = AIC↔AIV）、`ascend_tile.py:215/313-346`（`atomic_add` 仅 local→GM）⟹ **确定结论：Developer 模式无 grid 级顺序原语**，据此否决 Orchestrator §3.2 选项 B/C（§2.5） |
| ③ 源码 | `ascend_tile.py:635-655`、`common.h`（仓库 + wheel 两处）、`reduce_ascend.py:133-176`、`ascend_memory_planning.cc:598/606`、`ascend_combinecv.cc:36-58` | ✅ 全部逐行读取并记录行号（§3.5.4） |
| **④ 编译实证（rev3 新增层级）** | 4 个原型 kernel 的 `get_kernel_source()` + 生成码逐条核对（循环边界 / 分支 / 地址算式 / TopK 参数 / UB→UB 计数 / UB 总量） | ✅ 见 §13.6、§4.5、§4.9 —— **rev3 的所有可行性结论都有编译产物支撑，不是纯推理** |
| 🔴 禁用 | `tilelang/language/pto.py` | 已废弃，全程未引用 |

**结论**：rev3 **无凭记忆猜测的 API**。速查表未覆盖的 1 项（跨 block 同步）已下探到源码得到
确定性否决结论；所有新增 API 用法（扁平 grid、静态 split 循环、GM workspace 读写）
**既有 examples 先例、又经编译实证**。

---

## 4. 数据规格与内存规划

### 4.1 输入张量（不变）

| 名称 | shape | dtype | 说明 |
|------|-------|-------|------|
| `logits` | `[B, V]` | float32 | 唯一大张量。`B ∈ [1,1024]`，`V` 编译期常量 |
| `top_ks` | `[B]` | int32 | 每行 `k_i`；`TOP_K_ALL = 1<<30` 表示不限 |
| `top_ps` | `[B]` | float32 | 每行 `p_i ∈ (0,1]` |
| `min_ps` | `[B]` | float32 | 每行 `mu_i ∈ [0,1]` |
| `rand` | `[B]` | int32 | 仅 `FUSE_GUMBEL=True`：每行 hash 种子 `seed_i`（D-RAND Design C） |

### 4.2 输出张量（不变）

| 名称 | shape | dtype | 路径 |
|------|-------|-------|------|
| `m_out` | `[B]` | float32 | 两路 |
| `s_out` | `[B]` | float32 | 两路 |
| `tau_out` | `[B]` | float32 | 两路 |
| `valid` | `[B]` | int32 | 两路 |
| `w_out` | `[B, V]` | float32 | `FUSE_GUMBEL=False` |
| `tok_out` | `[B]` | int32 | `FUSE_GUMBEL=True` |

### 4.3 中间缓冲区（UB，rev3 按 kernel 重列）

**K1A（split 并行 top-k）** —— `BV=4096, K_MAX=256, TAIL_PAD=512` 时的名义账目：

| buffer | shape | dtype | 字节 | 说明 |
|--------|-------|-------|------|------|
| `z_ub` | `[BV]` | fp32 | 16384 | **rank-1**（N2：topk 源必须 rank-1）；GM→UB 目的地 + topk 源双用（rev2 round2 #4 的优化保留） |
| `l_dst` | `[AT=512]` | fp32 | 2048 | topk 输出（交织 `[v,i,v,i,…]`），`AT = ceil(2K/8)*8` |
| `l_val` | `[K_MAX=256]` | fp32 | 1024 | `gather_mask("P0101")` 取偶数位 = 值 |
| `merged` | `[2*K_MAX=512]` | fp32 | 2048 | `[run_v ‖ l_val]` |
| `m_dst` | `[AT=512]` | fp32 | 2048 | 归并 topk 输出 |
| `run_v` | `[K_MAX=256]` | fp32 | 1024 | 本 split 的运行 top-`K_MAX` 值列表 |
| `ztp` | `[1, TAIL_PAD]` | fp32 | `4*TAIL_PAD` | **仅 `TAIL>0`**：GM→UB 自填 `-inf` 的对齐缓冲（精度红线） |
| `ztt` | `[TAIL_PAD]` | fp32 | `4*TAIL_PAD` | **仅 `TAIL>0`**：rank-1 topk 源 |
| *(编译器注入)* `tmp_ub` | — | uint8 | **65536** | topk 的 Sort32/MrgSort ping-pong arena。依据 `src/transform/allocate_tmp_buffer.cc:578-596`（`GetAscendCWorkspaceSpec`：fp32 topk arena = `4 × (ceil(max_actual_num/32)×32) × 4 B`）、`:1195-1206`（`GetTmpBufferSize_`：`shape_size = std::max(shape_size, spec.primary_bytes)` ⟹ 取 **max** 而非 sum，多调用点共享单一 `shared.ub` buffer）、`:1156-1158`（`PointerType(…, "shared.ub")`） |
| **名义合计** | | | **≈ 85 KB + 8×TAIL_PAD** | 但**实测值以 §4.5 为准**（内存规划器的 liveness 复用不可靠地建模，round2 已证「模型只能是上界」） |

**K1B（串行 softmax + 跨 split 归并 + epilogue）**：

| buffer | shape | dtype | 字节 | 说明 |
|--------|-------|-------|------|------|
| `z_ub` | `[BV]` | fp32 | 16384 | GM→UB + `reduce_max` 源（rank-1 ⟹ 目的端传 `m_blk[0,0:1]` BufferRegion） |
| `e_ub` | `[1, BV]` | fp32 | 16384 | `exp(z − m)` |
| `m2d` | `[1, BV]` | fp32 | 16384 | `broadcast(m_run)`（⚠️ rev2 §4.5 曾**漏算**此项） |
| `zt`/`et`/`mt` | `[1, TAIL]` ×3 | fp32 | `12*TAIL` | **仅 `TAIL>0`**：尾块精确尺寸 + 全量 reduce（rev2 V-2 结论，逐字节保留；⚠️ rev2 §4.5 曾**漏算**这 3 项） |
| `pvb` | `[K_MAX]` | fp32 | 1024 | ⭐ **rev3 新增**：从 GM workspace 读入的 per-split `run_v_s` |
| `merged` / `m_dst` / `run_v` | `[512]`/`[512]`/`[256]` | fp32 | 2048+2048+1024 | 跨 split 归并用。⚠️ **K1B 不含 `l_dst`/`l_val`**（它不做 per-block topk）⟹ 按 N7 **必须删除**这两个声明 |
| epilogue 组 | `rv2`,`q2`,`mk2`,`sk2`,`tie_m`,`tie_one`,`tie_tkbc` 各 `[K_MAX]` + `tie_mask [K_MAX//8]` u8 | fp32/u8 | 6×1024 + 32 | 与现行完全相同（含 Stage 3 round1 [#1] 的向量化 tie 计数） |
| 标量组 | `m_blk`,`m_run`,`m_old`,`s_blk`,`s_run`,`tmp`,`k_i`,`p_i`,`mu_i`,`muc`,`lnmu`,`tau_k`,`tau_p`,`tau_u`,`tau`,`prefix`,`n_gt`,`n_eq`,`okp`,`tieok`,`vflag` | fp32/i32 | 21 × **32** = 672 | ⚠️ `alloc_shared` **32B 对齐**（rev2 §4.5 误按 `alloc_var` 的 4B 常量计算 ⟹ 低估） |
| *(编译器注入)* `tmp_ub` | — | uint8 | **16384** | 只有 `topk(actual=512)` 一个调用点 ⟹ arena 远小于 K1A 的 65536 |

**K2gA**：与现行 `r4_gumbel_sample` **完全同一组 buffer**（`z_ub[BV]` fp32、
`state_ub/h_ub/tmp_ub [2*BV]` int32（over-alloc：int16-native `And` 只处理 `count/2` 个 int32 的
halving workaround）、`g_ub[BV]` fp32、`mask[BV//8]` u8、`idx[BV]` fp32、
`blk_max/blk_argmax/t_v/seed_v/seedP_v/best_val/best_idx/tok_i32` 各 `[1]`）。
K2gA 只是把跨块折叠结果写到 GM workspace 而非 `tok_out`。
**K2gB**：只需 `bv/bi/best_val/best_idx/tok_i32` 五个 `[1]` buffer ⟹ **实测 132 B**。

### 4.4 内存搬运路径

**内存层级**：GM ↔ UB（Vector）。**无 L1 / L0A / L0B / L0C 参与**（零 Cube 计算）
⟹ **无跨级访问违规**（AGENTS.md 核心原则 3）。

| 路径 | rev2（现行） | rev3（split 路径） | 变化 |
|------|-------------|-------------------|------|
| GM → UB（`logits`） | K1: 1 趟 `V`；K2: 1 趟 `V` | **K1A: 1 趟 `V`（S 路并行）+ K1B: 1 趟 `V`（串行）**；K2gA: 1 趟 `V`（S 路并行） | ⚠️ **K1 的 `logits` GM 读从 1 趟增至 2 趟**（§10.1 R-BW）。K2g 不变（本就只读 1 趟，且未拆分其扫描） |
| UB → GM（per-split 部分结果） | 无 | ⭐ **新增**：`pv[B*S, K_MAX]` fp32（K1）；`pv_val/pv_idx[B*S]` fp32（K2g） | 极小：case7 最大 40960 + 320 B（§4.8） |
| GM → UB（per-split 部分结果） | 无 | ⭐ **新增**：K1B 读 `pv` 共 `S*K_MAX` fp32/行；K2gB 读 `2*S` fp32/行 | 极小（case7：10 KB + 80 B 每行，占 `logits` 每行 608 KB 的 **1.7%**） |
| UB → GM（最终输出） | `m/s/tau/valid`（`20*B` B）+ `w_out`（`4*B*V`）或 `tok_out`（`4*B`） | **完全相同** | 不变 |
| UB → UB | `run_v→merged[:K]`、`l_val→merged[K:]`、`ztp→ztt` | 完全相同 + K1B 新增 `pvb→merged[K:]` | 全部 32B 对齐（§4.9 编译实证） |

**Host 侧**：**零数据搬运**。rev3 新增的 host 操作只有 `torch.empty(...)` **分配**
（无 `reshape`/`cat`/`pad`/`copy_`/`.to()`/aclnn）。详见 §5.6 审计。

### 4.5 ⭐ UB 内存预算（rev3 重做：**实测法**，废弃 rev2 §4.5 模型）

**上限：196352 B**（从生成源 `InitBuffer(ascend_ub, 196352)` 解析得到；⚠️ rev2 §4.5 误写 188416）。

**⚠️ rev2 §4.5 的分析模型已被 Stage 3 round2 证实有 4 处错误 + 1 处缺失，rev3 一律不使用**：
① 漏算 `m2d[1,BV]`；② 漏算尾块 5 个缓冲；③ 用 `alloc_var` 常量而实为 `alloc_shared` 32B 对齐；
④ 上限写 188416 而实为 196352；⑤ **完全无 K2g 账目**（round2 实测公式 `36·BV + BV/8 + 484`）。
round2 结论原文（`perf_tuning/round2_bv_probe/bv_ub_sweep.py` docstring）：
*「The analytic model … is EXACT when the memory planner does no liveness reuse … but it is only
an UPPER BOUND once reuse kicks in … **Feasibility must therefore be decided by MEASUREMENT, not model.**」*

**rev3 采用的实测法**（复用 `perf_tuning/round2_bv_probe/ub_probe.py`，其 docstring 明确
「`total_used = max(off + n*sizeof(T))` —— exactly what `src/transform/ascend_memory_planning.cc:582-585`
computes for the DLOG line `Scope shared.ub memory usage: X/196352 bytes`」）：

```
编译特化 → get_kernel_source() → 正则解析全部 ascend_ub.GetWithOffset<T>(n, off)
→ total = max(off + n * sizeof(T))；region = 解析 InitBuffer(ascend_ub, N)
```

**rev3 实测结果**（本 Stage 1 在安装 wheel `0.1.4+d75300ed…` 上**实际执行**；
探针脚本 `/tmp/opencode/r4_splitv_ub_probe.py` + `/tmp/opencode/refine3.py`，
覆盖 12 个 case × 2 个特化 × 4 个 kernel = 48 次编译，全部成功）：

| kernel | 最差 case | **实测 UB (B)** | 占 196352 | **余量 (B)** | buffer 数 | vs 现行 |
|--------|-----------|-----------------|-----------|--------------|-----------|---------|
| **K1A**（split top-k） | case6（B=8,V=10240,TAIL=2048,S=3） | **106496** | **54.24%** | **89856** | 9 | 现行 K1 = **171620 B (87.40%)**，余量 24732 ⟹ **余量增至 3.6×** |
| **K1B**（softmax+归并+epilogue） | case17（TAIL_PAD=832, S=2） | **89444** | **45.55%** | **106908** | 38 | — |
| **K2gA**（split gumbel argmax） | 所有 `BV=4096` case | **148420** | **75.59%** | **47932** | 15 | 现行 K2g = **148452 B (75.6%)**，余量 47900 ⟹ **基本不变**（差 32 B，来自 workspace 写缓冲的标量对齐） |
| **K2gB**（argmax 归并） | — | **132** | **0.07%** | **196220** | 5 | 新增，可忽略 |
| **K2m**（不切） | — | **98912** | **50.4%** | **97440** | — | round2 实测，**rev3 不变** |

**逐 case K1A 实测明细**（10 个 split-path case，全部 `OK`）：

| case | B | V | K_MAX | BV | TAIL | TAIL_PAD | S | NBS | **K1A UB** | % | 余量 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 8 | 16384 | 256 | 4096 | 0 | 0 | 4 | 1 | 90112 | 45.89% | 106240 |
| 4 | 4 | 16384 | 256 | 4096 | 0 | 0 | 4 | 1 | 90112 | 45.89% | 106240 |
| 5 | 7 | 4409 | 256 | 4096 | 313 | 320 | 2 | 1 | 92672 | 47.20% | 103680 |
| 6 | 8 | 10240 | 256 | 4096 | 2048 | 2048 | 3 | 1 | **106496** | **54.24%** | 89856 |
| 7 | 4 | 152064 | 256 | 4096 | 512 | 512 | 10 | 4 | 94208 | 47.98% | 102144 |
| 8 | 4 | 16384 | 128 | 2048 | 0 | 0 | 8 | 1 | 45056 | 22.95% | 151296 |
| 11 | 8 | 8192 | 256 | 4096 | 0 | 0 | 2 | 1 | 90112 | 45.89% | 106240 |
| 12 | 16 | 32768 | 256 | 4096 | 0 | 0 | 2 | 4 | 90112 | 45.89% | 106240 |
| 13 | 6 | 2304 | 128 | 2048 | 256 | 256 | 2 | 1 | 47104 | 23.99% | 149248 |
| 17 | 16 | 152377 | 256 | 4096 | 825 | 832 | 2 | 19 | 96768 | 49.28% | 99584 |

（case5/6/11/13 的 `S` 是「几何上可 split」的值；其中 5/6/11/13 会被 §13.4 的 **guard 否决**而走现行路径，
但 K1A 仍必须能编译这些 `S`（L0 白盒用例 L0-E1 会显式构造 `S=1`），故一并实测。）

**为什么 K1A/K1B 的 UB 远低于现行 K1**：现行 K1 在**同一个 kernel** 里同时活着
`{z_ub, e_ub, m2d}`（3×16 KB softmax 侧）+ `{l_dst, l_val, merged, m_dst, run_v}` +
**`topk(actual=4096)` 的 65536 B tmp arena**（实测生成源中的 `(off=19840, n=65536, uint8_t)`）
+ 尾块 5 个缓冲。rev3 把它们拆到两个 kernel：K1A 只保留 topk 侧 + arena(65536)，
K1B 只保留 softmax 侧 + 一个**小得多**的 arena(16384，因为只有 `topk(actual=512)`)。
⟹ **拆分本身即降低了 UB 峰值**，这是 rev3 的**附带收益**（未来若要增大 `BV`，
K1 侧余量从 24732 B 增至 89856 B）。

**⚠️ 必须遵守的死缓冲禁忌（N7）**：`alloc_shared` 的 buffer 若在某条编译期分支下**完全未被使用**，
则无预分配地址 → codegen 失败（`Cannot find pre-allocated address for buffer`，`codegen_ascend.cc:828`；
现行 kernel docstring L186-190 与 `probe_k1_nosort.py:104` 均有记录）。
⟹ rev3 规定：K1B **必须删除** `l_dst`/`l_val` 声明；K1A **必须不声明** `e_ub`/`m2d`/`zt`/`et`/`mt`
及全部 epilogue buffer。

**⚠️ `alloc` 与 `use` 必须同词法作用域（N6，本 Stage 1 实测踩到）**：把 `z_ub = T.alloc_shared([BV])`
放在 `if n_blk_full > 0:` 内、而把使用它的 `for lb in T.serial(NBS):` 放在该 `if` **之外**，
会得到 `error: Undefined variable: z_ub`（TileLang 的 AST 变换把 buffer 定义作用域绑定到 enclosing `if`）。
`ztp`/`ztt` 同理。**正确写法**：`alloc` 与其后的整个循环/尾块代码都在同一个 `if` 块内
——**这正是现行 kernel 的结构**（`r4_fused_sampling.py:191-237` / `:240-269`），rev3 沿用。

**⚠️ `BV` 上界（round2 已实测，rev3 沿用）**：`BV=6144` **编译期硬失败**
（`LOG(FATAL) Memory allocation failed for: tmp_ub required 49152, available 48704`）；
`BV ∈ [3872, 4064]` 因 C3（`V=4096 → TAIL < K_MAX=256`，host 硬抛）不可行。
⟹ `BV=4096` 仍在可行上界附近。**注意 `BV` 是 cann-bench 契约属性（§15.2），rev3 不修改 `BV`，
`S` 的选取完全不依赖修改 `BV`（约束 7）。**

### 4.6 动态轴定义

| 轴 | 动态性 | 说明 |
|----|--------|------|
| `B` | **编译期常量**（JIT 特化键） | rev2 R-JIT 结论不变：`B` 进特化键 ⟹ 每个新 `B` 触发一次 JIT |
| `V` | **编译期常量** | 决定 `n_blk_full`/`TAIL`/`TAIL_PAD` |
| `K_MAX`, `BV`, `NEED_MIN_P`, `FUSE_GUMBEL` | 编译期常量 | **契约属性**（§15.2），rev3 不修改 |
| ⭐ `S`, `NBS`, `S_TAIL` | **编译期常量（rev3 新增特化键）** | 三者由 `(B, V, BV, CUBE_CORE_NUM)` **函数依赖**确定 ⟹ 同一设备上**不增加独立特化维度**（同一 `(B,V,K_MAX,BV,NEED_MIN_P)` 只对应一个 `S`）。⚠️ 但 `CUBE_CORE_NUM` 变化（换设备）会改变 `S` ⟹ 跨设备 JIT 缓存不可复用（§10.1 R-JIT2） |
| `row`, `sid`, `b`, `lb`, `s` | **运行时** | 只用于 GM 地址算式与守卫比较，**绝不作为循环边界**（§3.5.1） |
| `k_i`, `p_i`, `mu_i`, `seed_i` | 运行时张量元素 | 不进特化键 |

### 4.7 JIT 配置

```python
R4_PASS_CONFIGS = {                      # rev3 全部 7 个 kernel 一律使用，一字不改
    "tl.ascend_auto_sync": True,             # ① 核内自动同步（替代全部手写 barrier/flag）
    "tl.ascend_memory_planning": True,       # ② 自动内存规划 + ⭐ UB 溢出 LOG(FATAL) 安全网（§7.3）
    "tl.ascend_auto_cross_core_sync": True,  # ③ AIC↔AIV 自动同步（本算子无 Cube ⟹ 空转）
    "tl.ascend_auto_cv_combine": True,       # ④ 自动 CV 分离（同上，空转）
}
```
原始字符串形式（非 `PassConfigKey` 枚举），依据 `examples/xllm_kernels/utils.py:3-10`
（规避 `PassConfigKey` 导出时机依赖）。

**rev3 新增 kernel 的 `out_idx`**：

| kernel | `out_idx` | 理由 |
|--------|-----------|------|
| `r4_topk_stats_splitA` | **无** | `pv` 是 host 分配的 workspace，作为普通入参传入（不能用 `out_idx`，否则每次调用都新分配、无法跨 kernel 传递） |
| `r4_topk_stats_splitB` | `[5,6,7,8]` | `m_out, s_out, tau_out, valid` |
| `r4_gumbel_sample_splitA` | **无** | `pv_val`/`pv_idx` 同上 |
| `r4_gumbel_sample_splitB` | `[2]` | `tok_out` |

### 4.8 ⭐ GM workspace 布局与尺寸（rev3 新增）

| workspace | shape | dtype | 元素数 | 字节公式 | 写入者 | 读取者 |
|-----------|-------|-------|--------|----------|--------|--------|
| `pv` | `[B*S, K_MAX]` | fp32 | `B*S*K_MAX` | `4*B*S*K_MAX` | K1A（`pv[row*S+sid, :]`） | K1B（`pv[row*S+s, :]`，`s` 升序） |
| `pv_val` | `[B*S]` | fp32 | `B*S` | `4*B*S` | K2gA | K2gB |
| `pv_idx` | `[B*S]` | fp32 | `B*S` | `4*B*S` | K2gA | K2gB |

**布局说明**：
- 采用 **2D `[B*S, K_MAX]`** 而非 3D `[B, S, K_MAX]`：3D 本可用（`ascend-constraints.md §1` 的
  strided 切片限制针对 4D+），但 2D 布局使 `pv[row*S+s, :]` 的行偏移是**单一标量表达式**，
  与 `unsorted_segment_sum.py:50` 的 `Output[seg_ub[0], d_blk*block_D]` 同形态（既有先例），
  且已编译实证（生成码 `copy_gm_to_ub<float,256>(pvb[0], pv[…+(s*256)], 10240, 1, 256, 0.0)`）。
- **行主序连续**：`pv[i, :]` 是 `K_MAX` 个连续 fp32 ⟹ 每次读/写都是 1024 B（K=256）或 512 B（K=128），
  均 32B 对齐，无 strided 访问。
- **`pv_idx` 用 fp32 而非 int32** 存 argmax 索引：与现行 K2g 一致（`best_idx` 本就是 fp32 `[1]` buffer，
  `best_idx < 2^24` 时 fp32 精确 —— `V ≤ 1024×… < 2^24` 恒成立，最后 `T.tile.cast(…, "CAST_ROUND", 1)`
  转 int32）。**保持 fp32 是 bitwise 等价的必要条件**（避免 int32↔fp32 往返引入任何差异）。
- **每个 `(row, sid)` 无条件写满自己的 `pv` 行**（K1A 末尾的 `T.copy(run_v, pv[…])` 在任何分支之外）
  ⟹ 配合定理 P3（无空 split），**`pv` 不存在未写行** ⟹ host 可用 `torch.empty` 而非 `torch.zeros`（§5.6）。

**18-case workspace 尺寸表**（`S` 由 §13.4 判据得出；「—」表示走现行路径、不分配 workspace）：

| case | B | V | K_MAX | S | `pv` (B) | `pv_val`+`pv_idx` (B) | **合计 (B)** | KB | 占同 case `logits` 比例 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 256 | 256 | — | — | — | 0 | 0 | — |
| 2 | 1 | 4096 | 256 | — | — | — | 0 | 0 | — |
| 3 | 8 | 16384 | 256 | 4 | 32768 | 256 | 33024 | 32.3 | 6.3% |
| 4 | 4 | 16384 | 256 | 4 | 16384 | 128 | 16512 | 16.1 | 6.3% |
| 5 | 7 | 4409 | 256 | (guard 否决) | — | — | 0 | 0 | — |
| 6 | 8 | 10240 | 256 | (guard 否决) | — | — | 0 | 0 | — |
| **7** | **4** | **152064** | **256** | **10** | **40960** | **320** | **41280** | **40.3** | **1.7%** |
| 8 | 4 | 16384 | 128 | 8 | 16384 | 128 | 16512 | 16.1 | 6.3% |
| 9 | 64 | 4096 | 256 | — | — | — | 0 | 0 | — |
| 10 | 4 | 4096 | 256 | — | — | — | 0 | 0 | — |
| 11 | 8 | 8192 | 256 | (guard 否决) | — | — | 0 | 0 | — |
| 12 | 16 | 32768 | 256 | 2 | 16384 | 128 | 16512 | 16.1 | 0.8% |
| 13 | 6 | 2304 | 128 | (guard 否决) | — | — | 0 | 0 | — |
| 14 | 32 | 1024 | 256 | — | — | — | 0 | 0 | — |
| 15 | 64 | 152064 | 256 | — | — | — | 0 | 0 | — |
| 16 | 128 | 32768 | 256 | — | — | — | 0 | 0 | — |
| **17** | **16** | **152377** | **256** | **2** | **32768** | **256** | **33024** | **32.3** | **0.34%** |
| 18 | 64 | 65536 | 256 | — | — | — | 0 | 0 | — |

**最大 workspace = 41280 B ≈ 40.3 KB（case7）**，为同 case `logits`（2433088 B）的 **1.7%**。
⟹ 显存开销**可忽略**。

> 📝 **对 Orchestrator §3.2 代价 ③ 的修正**：Orchestrator 估算「workspace 显存 `B×n_blk×4B×2`
> （case15 ≈ 18.9 KB）」——该公式对应的是 4-pass 分解要存 `m_b`/`s_b` **每块两个标量**。
> **rev3 不需要 `m_b`/`s_b` workspace**（softmax 未切分），只需要 `run_v_s`（每 split `K_MAX` 个值），
> 公式是 `4·B·S·(K_MAX+2)`。case7 实际 41280 B，比 Orchestrator 对 case15 的估算大约 2.2×，
> 但 case15 走现行路径**不分配** workspace ⟹ 实际最大是 case7 的 40.3 KB。量级判断（可忽略）一致。

**上界公式（供任意 shape 核算）**：`workspace_bytes = 4*B*S*(K_MAX + 2)`，
其中 `S ≤ min(CUBE//b_num, n_blk_total) ≤ CUBE = 20` ⟹ `≤ 80*B*(K_MAX+2)`。
对 `B=1024, K_MAX=256`：`≤ 21.0 MB`（仍远小于 `logits` 的 `4*1024*V`，`V=152064` 时为 623 MB）。
实际约束更紧：`B=1024 ⟹ b_num=512 ⟹ S_cap = 0 ⟹ S=1 ⟹ 走现行路径、workspace = 0`。

### 4.9 ⭐ `padCount == 0` 与 32B 对齐在 split-V 新分解下的重新证明（编译实证）

**这是精度红线（§12.3 / rev2 §10.3）在 rev3 下的重新验证，Orchestrator §4 明确要求
「必须重新证明每 split 的尾块也满足 `padCount==0` 与 32B 对齐」。**

**先验论证**（D-SPLIT-BVALIGN + 定理 P1）：split 边界按**块**对齐 ⟹ 每 split 内每个 top-k 源的
`actual_num` 只可能是 `BV`（满块）、`2*K_MAX`（归并）、或 `TAIL_PAD`（尾块）。三者都 `%32 == 0`：

- `BV ∈ {4096, 2048}`：`4096%32=0`、`2048%32=0` ✅
- `2*K_MAX ∈ {512, 256}`：`512%32=0`、`256%32=0` ✅
- `TAIL_PAD = ceil(TAIL/32)*32`：`%32 == 0` **由定义结构性保证，与 `TAIL`、与 `BV`、与 `S` 全部无关** ✅

依据 `ascend_tile.py:647`（`repeatTimes = (max_actual_num + 31)//32`）+
`common.h:1715-1717`（`alignedCount = repeatTimes*blockSize; padCount = alignedCount − actualCount`）
⟹ `actual % 32 == 0 ⇒ aligned == actual ⇒ padCount == 0`，整个 `if (padCount > 0)` 块被跳过，
**masked `AscendC::Duplicate` 分支永不执行**（缺陷 A 结构性不可触发）。

**关键洞察**：rev3 **未引入任何新的 `actual_num` 取值** —— 集合仍是 `{BV, 2*K_MAX, TAIL_PAD}`，
与现行完全相同。**split-V 改变的是「哪个核处理哪个块」，不改变「每个块有多宽」。**
这就是 D-SPLIT-BVALIGN 的全部意义，也是 Orchestrator §4「建议强制 split 边界按 BV 对齐」的落实。

**编译实证**（本 Stage 1 对全部 10 个几何可 split 的 case 逐个解析生成源的**全部**
`TopK<T>(dst, src, tmp, K, repeatTimes, actual_num)` 与**全部** `copy_ub_to_ub<…, count>`）：

| case | 生成源中全部 TopK 调用点 `(K, repeatTimes, actual_num)` | `padCount` | `copy_ub_to_ub` 元素计数集合 | ×4B | 全部 `%32==0`？ |
|---|---|---|---|---|---|
| 3 / 4 / 11 / 12 | `(256,128,4096)`, `(256,16,512)` | **全 0** | {256} | {1024} | ✅ |
| 5 | `(256,128,4096)`, `(256,16,512)`, **`(256,10,320)`**, `(256,16,512)` | **全 0** | {256, 320} | {1024, 1280} | ✅ |
| 6 | `(256,128,4096)`, `(256,16,512)`, **`(256,64,2048)`**, `(256,16,512)` | **全 0** | {256, 2048} | {1024, 8192} | ✅ |
| 7 | `(256,128,4096)`, `(256,16,512)` ×3 | **全 0** | {256, 512} | {1024, 2048} | ✅ |
| 8 | `(128,64,2048)`, `(128,8,256)` | **全 0** | {128} | {512} | ✅ |
| 13 | `(128,64,2048)`, `(128,8,256)` ×3 | **全 0** | {128, 256} | {512, 1024} | ✅ |
| 17 | `(256,128,4096)`, `(256,16,512)`, **`(256,26,832)`**, `(256,16,512)` | **全 0** | {256, 832} | {1024, 3328} | ✅ |

（case5 的 `(256,10,320)`：`TAIL=313 → TAIL_PAD=320`，`ceil(320/32)=10` ✅；
case6 的 `(256,64,2048)`：`TAIL=2048 → TAIL_PAD=2048`，`ceil(2048/32)=64` ✅；
case17 的 `(256,26,832)`：`TAIL=825 → TAIL_PAD=832`，`ceil(832/32)=26` ✅ ——
**这正是 rev2 修复的那个非 32 对齐尾块 shape，在 split-V 下 `padCount` 仍为 0**。）

**结论**：
1. **`padCount == 0` 在全部 split-path case 的全部 top-k 调用点上成立** ⟹ 缺陷 A 结构性不可触发。
2. **全部 UB→UB 拷贝的字节数都是 32 的倍数** ⟹ 缺陷 B 结构性不可触发。
3. rev3 **未引入任何新的 `actual_num` 取值** ⟹ 精度红线的规避机制**完整保持，且保持方式与 `BV`/`S` 无关（结构性）**。

⚠️ **Stage 2 必须做的运行时确认**（编译期只证明了「计数对齐」，不能证明「无陈旧 UB 被消费」）：
把现有 `--level stress` 的 **UB 投毒层**（`test_r4_fused_sampling.py:1327-1348` `_make_ub_poisoner`，
投毒 `1e30` 后立刻 launch，`_POISON_ELEMS` 覆盖全 UB）**同样施加到 K1A/K1B/K2gA/K2gB**，
断言 `tau_out` 不出现 `1e30` 量级垃圾值、且 20 次重复 launch **逐位相同**。
这是缺陷 A 的唯一运行时检测手段（rev2 的间歇性 ~13%/launch 缺陷正是由它 + cann-bench case5 暴露）。
已写入 §9.2 L0 计划的强制项。

---

## 5. Tiling 策略

### 5.1 计算类型

**纯 Vector**（element-wise + reduction + Ascend 专属 `topk`/Sort32/MrgSort 向量指令）。
**零 Cube 计算** ⟹ 无 L1/L0A/L0B/L0C、无 GEMM、无 CV 融合需求。rev3 未改变此判定。

复杂度级别：**Fusion**（多 kernel 链：K1 → K2；rev3 进一步拆为 K1A → K1B → K2gA → K2gB）。

### 5.2 Block 划分

**rev2（现行路径，逐字节保留）**

```
grid = b_num = ceildiv(B, VEC_NUM=2) 个 block；每 block = 1 AIC + 2 AIV
task = cid*VEC_NUM + vid  ∈ [0, 2*b_num)；row = task；守卫 if row < B
并行映射：task_id ↔ row_id（一行一任务，无跨核归约）
V 维在 block 内以 T.serial(n_blk_full) + 尾块 串行扫描   ← 这就是 rev2 的并行度天花板
```

**rev3 split-V 路径（新增）**

```
K1A / K2gA：grid = b_num * S 个 block
            task = cid*VEC_NUM + vid ∈ [0, 2*b_num*S) = [0, B*S)
            row  = task // S          ← 行号
            sid  = task %  S          ← split 号
            守卫 if row < B
            并行映射：task_id ↔ (row_id, split_id) 双射（B*S 个任务，一任务一 (行, split)）
            每 split 负责全局块号区间 [sid*NBS, min((sid+1)*NBS, n_blk_total))
            ⚠️ 任务数 = B*S ≤ 2*b_num*S_cap ≤ 2*CUBE_CORE_NUM = 40 = 全部 AIV ⟹ 恒 ≤ 1 波（定理 P2）

K1B / K2gB：grid = b_num（与现行完全相同）
            row = cid*VEC_NUM + vid；守卫 if row < B
            每行串行：K1B 扫全部 n_blk_full 块做 softmax + 读 S 份 run_v_s 归并 + epilogue
                      K2gB 读 S 份 (val, idx) 折叠
```

**Tile Shape**：
- **块级 tile**：`BV × 1`（每块 `BV` 个 vocab 元素，一行）。rev3 **不改 `BV`**（§15.2：`BV` 是契约属性）。
- ⭐ **rev3 新增 split 级 tile**：每 split `NBS * BV` 个元素（`NBS = ceil(n_blk_total/S)`）。
  split 边界**必然落在块边界上**（定理 P1）⟹ 不存在「半个块」的 split。

### 5.3 约束分析

rev2 §5.3 的 N1–N5 全部继续有效：

| 编号 | 约束 | rev3 状态 |
|------|------|----------|
| **N1** | 尾块 `reduce` 不能用 `real_shape`/BufferRegion narrow 路径（逻辑宽度 ≤256 B = 64 fp32，`codegen_ascend.cc:2145`）⟹ 尾块用**精确尺寸 `[1,TAIL]` 缓冲 + 全量 reduce** | ✅ K1B 逐字节沿用现行 L241-261；**K1A 不做任何 reduce**（只做 topk）⟹ 不涉及 |
| **N2** | topk 源必须 **rank-1**（`max_actual_num = Σ shape` 陷阱，§3.5.4） | ✅ K1A 的 `z_ub[BV]` / `ztt[TAIL_PAD]` 均 rank-1 |
| **N3** | topk 对 src 做 **in-place `-inf` 填充**（`[actual_num : buffer_size)`） | ✅ K1A 中 `z_ub` 的最后一次读是 `topk` 本身，下一轮迭代开头被 GM→UB 覆写 ⟹ 与现行同一论证（`actual_num == BV == buffer size` ⟹ 实际无填充） |
| **N4** | `run_v` 首轮归并前必须 `fill(-inf)` | ✅ K1A **每个 split 各自** `fill(-inf)`；§14.3 论证其正确性（每 split 至少 1 个满块 ⟹ ≥`BV ≥ K_MAX` 个真实候选 ⟹ `-inf` 初值必被完全挤出） |
| **N5** | 1 元素 UB→UB `T.copy` 触发 aicore 异常 507015 ⟹ 标量用 `T.tile.*` 就地运算 | ✅ rev3 新增的 1 元素拷贝全部是 **GM↔UB**（`pv_val`/`pv_idx`），**不是** UB→UB ⟹ 不触发（§3.2 已注，且编译实证生成码为 `copy_gm_to_ub<float,1>` / `copy_ub_to_gm<float,1>`） |

**rev3 新增约束**：

| 编号 | 约束 | 处理 |
|------|------|------|
| **N6** | ⭐ **`alloc_shared` 的声明与使用必须在同一词法 `if` 作用域** | 本 Stage 1 实测踩到（`error: Undefined variable: z_ub` / `ztp`）。规定：`z_ub` 声明在 `if n_blk_full > 0:` 内且其 `for` 循环也在同一 `if` 内；`ztp/ztt` 声明在 `if TAIL > 0:` 内且尾块代码也在同一 `if` 内。**这正是现行 kernel 的结构**（`r4_fused_sampling.py:191-237` / `:240-269`），rev3 不改动它 |
| **N7** | ⭐ **死 `alloc_shared` → codegen 失败** | `Cannot find pre-allocated address for buffer`（`codegen_ascend.cc:828`）。K1B **不再需要** `l_dst`/`l_val` ⟹ **必须删除**；K1A **不需要** `e_ub`/`m2d`/`zt`/`et`/`mt` 及全部 epilogue buffer ⟹ **必须不声明** |
| **N8** | ⭐ **split 数 `S` 必须使 grid 落在 1 波内** | `S_cap = CUBE_CORE_NUM // b_num`（定理 P2，§5.8.3）。⚠️ **不得**为「追求更大 S」而放宽到 2 波 —— 定理 P2 证明那严格无益 |
| **N9** | ⭐ **不得出现空 split** | `S = ceil(n_blk_total / NBS)` 回代收缩（定理 P3，§5.8.3）。空 split 会白占任务槽并可能让 K1B 读到「只含 `-inf` 初值」的 `run_v_s`（虽然 N4 保证其确定，但 §14.3 的论证以「每 split ≥1 满块」为前提） |
| **N10** | ⭐ **`S >= 2 ⟹ n_blk_full >= 1` 恒成立** | 证明：`S >= 2 ⟹ n_blk_total >= 2`（因 `S ≤ n_blk_total`）；若 `n_blk_full == 0` 则 `n_blk_total == 1`（只有尾块）⟹ `S == 1`，矛盾。**∎** ⟹ K1A 在 split 路径下 `z_ub` **必然被使用**（不会死分配），`if n_blk_full > 0:` 分支必然进入 |
| **N11** | ⭐ **K2gA 的 hash counter 起点与 index 偏移必须用全局块号 `b`，不能用局部 `lb`** | `arith_progression(state_ub, seedP_v[0] + b*BV, 1, BV2)` 与 `best_idx[0] = blk_argmax[0] + b*BV`。误用 `lb` 会导致 `G_ij` 全错（每 split 重复同一段噪声）+ `tok_out` 索引错误。§14.5 的 bitwise 论证**依赖这一点**，L0-A2/A4 会捕获 |

### 5.4 注意事项（非整除与边界，显式设计）

rev3 有**两层**非整除，都必须显式设计（`ascend-constraints.md §5`：用输入/输出两侧显式
valid extent / BufferRegion，host 侧不允许 padding + crop）：

**第 1 层：`V % BV != 0`（尾块）** —— 与 rev2 完全相同，**处理方式逐字节保留**：
- 输入侧（softmax）：`T.copy(logits[row, n_blk_full*BV], zt)` 载入**精确尺寸** `[1, TAIL]`
  （GM→UB 自动 clamp 到 `V`，`src/op/ascend.cc::compute_valid_extent`）。
- 输入侧（top-k）：`T.copy(logits[row, n_blk_full*BV], ztp, pad_value=-inf)` 载入 `[1, TAIL_PAD]`
  （不足部分由 GM→UB 自填 `-inf`），再 `T.copy(ztp[0, :TAIL_PAD], ztt)` hop 到 rank-1
  （`TAIL_PAD*4` B 是 32 的倍数 ⟹ `AscendC::DataCopy` 不截断）。
- 输出侧：K1 无 `V` 宽输出（只写 `[B]` 标量）；K2m 的 `T.copy(w_ub, w_out[row, b*BV])` 由 ub2gm
  自动 re-clamp（rev2 C2，不变）。
- **host 侧不做 padding + crop** ✅

**第 2 层（rev3 新增）：`n_blk_total % S != 0`（split 不均）**
- **显式 valid extent**：每 split 的有效块数 `= max(0, min(n_blk_total, (sid+1)*NBS) − sid*NBS)`，
  是 `sid` 的运行时函数。用**静态循环 `T.serial(NBS)` + 运行时守卫 `if b < n_blk_full`** 表达
  （`ascend-constraints.md §1`「预计算最大循环次数 + 条件判断」的官方替代方案）。
- **输入侧裁剪**：由守卫裁剪 ⟹ 不越界读 `logits`（`b < n_blk_full ⟹ b*BV + BV ≤ n_blk_full*BV ≤ V`）。
- **输出侧裁剪**：`pv[row*S+sid, :]` 每个 split **都写满 `K_MAX`**（真实候选不足 `K_MAX` 时由
  `run_v` 的 `-inf` 初值填充，N4）⟹ **无写竞态**（每个 `(row,sid)` 有独占的 `pv` 行）、**无空洞**。
- **无空 split**（定理 P3）⟹ 不存在「某 split 完全不写 `pv` 导致 K1B 读到未初始化 GM」的情形。
- **负载均衡**：最大 split 块数 `= NBS = ceil(n_blk_total/S)`，最小 `= n_blk_total − (S−1)*NBS`。
  case7：`[4,4,4,4,4,4,4,4,4,2]`（不均 4:2，临界路径由 4 决定）；case17：`[19,19]`（完全均衡）。
  ⚠️ **不均不会导致正确性问题**（每 split 独立），只影响临界路径 ⟹ 已计入 §12 的 `max(per)` 建模。
- **host 侧不做 padding**：`pv` 直接按 `[B*S, K_MAX]` 分配，`S` 已收缩使无空行 ⟹ 无需 crop。

### 5.5 数据搬运性能可行性

**rev2 §5.5 的结论（`[REORDER-COST]`、每核 DMA 任务数）在现行路径上不变。** rev3 split 路径新增核算：

| 项 | 现行 | rev3 split 路径 | 评估 |
|---|---|---|---|
| `logits` GM→UB 事务数（每行） | `n_blk_total` 次 | **`2 × n_blk_total`** 次（K1A + K1B 各一遍） | ⚠️ **翻倍**。但每次事务的**尺寸不变**（`BV` 或 `TAIL`）⟹ 无「小事务化」劣化。且 K1 是 **compute-bound（Sort 主导）而非 bandwidth-bound**：case7 每核读 608 KB 用 419 us ⟹ **1.45 GB/s/核**，远低于 910b2 单核可达带宽 ⟹ **带宽不是瓶颈**（§12.2 实测证据 + `compute_r4_fused_sampling.py:105-110` 的 t_hw 模型结论「VECTOR-BOUND by 3.84x」） |
| GM 地址解码 / 标量访问 | `n_blk_total` 个 2D 标量起点 | `2 × n_blk_total` 个 | 同上，非瓶颈 |
| workspace GM 写/读 | 0 | K1A 写 `S*K_MAX` fp32/行；K1B 读同量。K2gA 写 `2*S`；K2gB 读同量 | **可忽略**：case7 每行 10 KB 写 + 10 KB 读，对比 `logits` 每行 608 KB ⟹ **3.3%** |
| **并行度（活跃 AIV 数）** | `min(2*b_num, 40)` | K1A/K2gA：`min(2*b_num*S, 40)`；K1B/K2gB：`min(2*b_num, 40)` | ⭐ **case7：4 → 40（10×）**；case3/8：8 → 32/16；case12/17：16 → 32 |
| 每核串行任务数 | `ceil(b_num/20)` 波 × 2 行 | K1A/K2gA：`ceil(b_num*S/20)` 波 = **1**（定理 P2）× 每任务 `NBS` 块 | ⭐ **每核块数从 `n_blk_total` 降到 `NBS = ceil(n_blk_total/S)`**，波数不增 |
| 总 DMA 字节数（case7，每行） | 608 KB | 1216 KB + 20 KB | **2.03×**。因 compute-bound，墙钟时间仍降 ~3× (§12.4) |

### 5.6 Host 侧 Buffer 操作审计（rev3 重做，含 3 个新增块）

```text
[HOST-METADATA-AUDIT]  # rev3 新增项 1：workspace 分配
operation: pv     = torch.empty((B*S, K_MAX), dtype=torch.float32, device=logits.device)
           pv_val = torch.empty((B*S,),       dtype=torch.float32, device=logits.device)
           pv_idx = torch.empty((B*S,),       dtype=torch.float32, device=logits.device)
input_stride -> output_stride: N/A（新建未初始化张量，不是任何输入 tensor 的 view）
shares_storage / same_data_ptr: false（全新分配，不与 logits/输出共享存储）
aclnn_or_physical_copy: false —— torch.empty 只做 device 内存分配（aclrtMalloc），
                                不搬运数据、不调用任何 aclnn 计算算子、不做 dtype 转换
result: allow
        依据：ascend-constraints §5 允许「数据准备、kernel 调用和结果验证」；
              workspace 分配属 kernel 调用的必要前置，与 tilelang.jit 的 out_idx 内部所为同质
              （out_idx 亦由框架 torch.empty 分配）。
        ⚠️ 明确禁止的替代写法（rev3 一律不用）：
           torch.zeros(...)（会触发 aclnn 填充算子 + 一次全量 device 写）
           pv.view / reshape / permute / contiguous（物理化）
           torch.cat / stack / F.pad（aclnn）
           pv.copy_(...) / pv[...] = ...（in-place 数据搬运）
        ⟹ rev3 选择 torch.empty 而非 torch.zeros 的依据：K1A/K2gA 对每个 (row, sid)
           都【无条件写满】自己的 pv 行（§4.8），且定理 P3 保证无空 split
           ⟹ pv 无未写行 ⟹ 初值无关 ⟹ 可安全用 empty，省一次 device 填充。

[HOST-METADATA-AUDIT]  # rev3 新增项 2：分派判据计算
operation: §13.4 分派判据（b_num / n_blk_full / TAIL / n_blk_total / S_cap / S / NBS /
           S_TAIL / pred_saving / pred_cost / use_split）
input_stride -> output_stride: N/A
shares_storage / same_data_ptr: N/A
aclnn_or_physical_copy: false —— 全部是 Python int / float 标量算术，
                                输入只有 B、V、K_MAX、BV 与 detect_vec_core_num() 的返回值，
                                【不读取任何 tensor 的元素值】
result: allow
        ⚠️ 关键性质：分派【不依赖 tensor 内容】⟹ 不引入 D2H 同步、不破坏 ACLGraph 图捕获。
           （对比 rev2 §5.6 记录的 valid.cpu() D2H 同步：那是 D-GUARD 的必要代价，rev3 未新增此类同步。）
        ⚠️ detect_vec_core_num() 会调用 torch.npu.get_device_properties()：这是【设备属性查询】，
           非 tensor 操作、非 aclnn 计算算子；且现行 kernel 已在用（r4_fused_sampling.py:70-83），
           rev3 未新增此类调用点（只多调用一次以派生 CUBE_CORE_NUM）。

[HOST-METADATA-AUDIT]  # rev3 新增项 3：kernel 调用序列
operation: r4_topk_stats_splitA(...) → r4_topk_stats_splitB(...) →
           r4_mask_sample(...)  或  (r4_gumbel_sample_splitA(...) → r4_gumbel_sample_splitB(...))
input_stride -> output_stride: 不变
shares_storage / same_data_ptr: true（pv / pv_val / pv_idx 在 A→B 之间按引用传递，无拷贝）
aclnn_or_physical_copy: false（TileLang 生成的 .so 直接 launch，不经 aclnn 算子库）
result: allow —— 与 rev2 的 K1→K2 顺序调用同质，只是从 2 次 launch 增至 3~4 次
```

**rev2 §5.6 的其余审计项（`logits` H2D、`out_idx` 分配、`valid.cpu()` D2H 分派、
`torch.multinomial` DEV-1 偏差）在 rev3 中全部原样有效，未新增违规项。**

**rev3 host 路径违规自检**：不含 `.contiguous()` / `reshape` 物理化 / `permute` / `torch.cat` /
`torch.stack` / `F.pad` / `.to(dtype)` / `.clone()` / `x[:] = …` / in-place 算术 / `torch.zeros`。
✅ **无违规**

### 5.7 分派覆盖审计

rev2 §5.7 的 `[DISPATCH-COVERAGE]` 块在 rev3 中**整体有效**（`supported_domain` /
`generic_fallback` / 5 条 `specializations` / `unsupported_inputs` 均不变，完整块见
`history_version/design_rev2.md` §5.7）。rev3 **新增 1 条 specialization**：

```text
[DISPATCH-COVERAGE]  # rev3 追加
  - predicate: §13.4 判据 use_split == True
               （S >= 2 且 n_blk_full >= 2 且 pred_saving > 1.5 * pred_cost）
    fallback_on_miss: use_split == False → 走现行 kernel（r4_topk_stats / r4_gumbel_sample），
                      代码逐字节不变 ⟹ 行为与 rev2 交付版（sha256 19e51e8d…）完全一致
    equivalence_evidence:
      §14（bitwise 等价性逐 pass 论证：M1 前缀最大值顺序无关 + M2 top-K 值多重集确定性
           + M3 argmax-with-min-index 可结合 + softmax 扫描代码逐条未改）
      §4.9（padCount==0 与 32B 对齐的编译实证，10/10 split-path case 全绿）
      §9.2 L0-D1（双路 bitwise 对照用例，torch.equal 逐位断言，5 个输出全覆盖）
      §9.2 L0-E1（S=1 退化用例：split 代码路径在 S=1 时必须与现行 kernel 逐位相同）
  ⟹ 与 supported_domain 不冲突：use_split 只是【同一 supported_domain 内的实现路径选择】，
     不扩大也不缩小任何输入域。域外输入（dtype≠fp32 / NaN / B==0 / C3·C7 不满足）
     仍由 rev2 的 host 前置校验拦截，且【分派判据在这些校验之后执行】。
result: pass
```

**验证命中**（rev3）：新增 specialization 的两条分支在 §9.2 L0 计划中均有命中 case
（`use_split=True` → L0-A1..A4 / B1..B5 / C1 / F1；`use_split=False` → L0-C2 / C3 / E1 的对照侧 + D1 的对照侧）。

### 5.8 ⭐ split-V 分块与 `S` 选取规则（rev3 核心新增）

#### 5.8.1 定义

```python
VEC_NUM        = 2                                    # 每 AIC 配 2 个 AIV（现行常量，不变）
CUBE_CORE_NUM  = detect_vec_core_num() // VEC_NUM     # 运行时探测，绝不硬编码
                                                      # Ascend910_9362 (910b2): 40//2 = 20 ✅
b_num          = ceildiv(B, VEC_NUM)                  # 现行 grid
n_blk_full     = V // BV
TAIL           = V - n_blk_full * BV
TAIL_PAD       = ceil(TAIL / 32) * 32                 # 精度红线，不变
n_blk_total    = n_blk_full + (1 if TAIL > 0 else 0)  # = ceildiv(V, BV)
```

#### 5.8.2 `S` 选取算法（全 host 侧 Python int，编译期常量）

```python
S_cap  = CUBE_CORE_NUM // b_num                 # 步骤 1：1-波上限（定理 P2）
S      = max(1, min(S_cap, n_blk_total))        # 步骤 2：不超过总块数
NBS    = ceildiv(n_blk_total, S)                # 步骤 3：每 split 的静态循环上界
S      = ceildiv(n_blk_total, NBS)              # 步骤 4：⭐ 回代收缩 ⟹ 无空 split（定理 P3）
NBS    = ceildiv(n_blk_total, S)                # 步骤 5：用收缩后的 S 重算 NBS
S_TAIL = n_blk_full // NBS if TAIL > 0 else -1  # 步骤 6：尾块所属 split（编译期常量）
```

#### 5.8.3 三条定理（含证明）

**定理 P1（split 边界按 `BV` 对齐 ⟹ 精度红线保持）**
split `s` 负责全局块号 `[s*NBS, min((s+1)*NBS, n_blk_total))`；块号 `b < n_blk_full` 是宽度恰为
`BV` 的满块，块号 `b == n_blk_full`（若存在）是尾块。⟹ 每个 top-k 源的
`actual_num ∈ {BV, 2*K_MAX, TAIL_PAD}`，三者 `%32 == 0`（§4.9）⟹ `padCount == 0`
且全部 UB→UB 字节数 `%32 == 0`。**∎**
（编译实证：§4.9 表，10/10 case 全绿。）

**定理 P2（`S_cap = CUBE_CORE_NUM // b_num` 是 1-波上限，超过则严格无益）**
设每块 Sort 成本为 `t`（实测常数，§12.2），每行 `n` 块，`C = CUBE_CORE_NUM`。
- 不 split：`waves = ceil(b_num / C)`，每波每核处理 `n` 块 ⟹ Sort 时间 `= waves × n × t`。
- split 为 `S`：`grid = b_num × S`，`waves' = ceil(b_num·S / C)`，每任务 `ceil(n/S)` 块
  ⟹ Sort 时间 `= waves' × ceil(n/S) × t`。
- 情形 1：`b_num·S ≤ C` ⟹ `waves' = 1`，时间 `= ceil(n/S)·t ≤ (n/S + 1)·t` ⟹ **接近 S 倍加速**。
- 情形 2：`b_num ≤ C` 但 `b_num·S > C`。取最小越界值 `S = 2`、`b_num ∈ (C/2, C]`
  （`C=20` 时即 `b_num ∈ [11,20]`）：`waves' = ceil(2·b_num/20) = 2`，
  时间 `= 2 × ceil(n/2) × t ≥ n·t` ⟹ **≥ 不 split 的时间**，且多一次 kernel launch ⟹ **严格变慢**。
  对更大的 `S`，`waves' × ceil(n/S) ≥ ceil(b_num·S/C) × n/S ≥ (b_num/C)·n ≥ n`（当 `b_num ≥ C`），
  同样无益。
- 这正是 Orchestrator §2 用 case15 数值验证的结论（`b_num=32, S=2 → blocks=64 → waves=ceil(64/20)=4
  × scan(V/2)=228 → 912us ≈ 现状 911.6us`），本定理把它一般化到 `b_num ∈ [11,19]`。**∎**

**定理 P3（步骤 4 的回代收缩排除空 split）**
设 `n = n_blk_total`，`NBS = ceil(n/S₀)`（`S₀` 为步骤 2 的 `S`）。若 `(S₀−1)·NBS ≥ n`，
则 split `S₀−1` 的块数 `= n − (S₀−1)·NBS ≤ 0` ⟹ **空 split**。
取 `S = ceil(n/NBS)`：由 `ceil` 定义 `(S−1)·NBS < n` ⟹ split `S−1` 的块数
`= n − (S−1)·NBS ≥ 1` ✅；对其余 `s < S−1`，块数 `= NBS ≥ 1` ✅。⟹ **无空 split**，
且 `S = ceil(n/NBS)` 是满足该性质的**最大**整数。**∎**
反例（说明收缩必要）：`n=5, S₀=4 → NBS=ceil(5/4)=2 → (4−1)·2 = 6 ≥ 5` ⟹ 朴素 `S=4` 给出
`[2,2,1,0]`，split 3 为空；收缩后 `S = ceil(5/2) = 3` ⟹ `[2,2,1]` ✅。
**实测命中**：§9.2 L0-F1 用例（`B=10, V=20480`）⟹ `b_num=5, S_cap=4, n_blk_total=5`
⟹ 朴素 `S=4` 会产生空 split，收缩为 `S=3`（per-split `[2,2,1]`）。

#### 5.8.4 18-case `S` 全表（本 Stage 1 实际计算，`CUBE_CORE_NUM=20`）

| case | B | V | K_MAX | BV | b_num | n_blk_full | TAIL | n_blk_total | S_cap | **S** | NBS | grid | waves | S_TAIL | per-split 块数 | 分派 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 256 | 256 | 4096 | 1 | 0 | 256 | 1 | 20 | **1** | 1 | 1 | 1 | 0 | [1] | 现行（`n_blk_total=1`） |
| 2 | 1 | 4096 | 256 | 4096 | 1 | 1 | 0 | 1 | 20 | **1** | 1 | 1 | 1 | −1 | [1] | 现行（`n_blk_total=1`） |
| **3** | 8 | 16384 | 256 | 4096 | 4 | 4 | 0 | 4 | 5 | **4** | 1 | 16 | 1 | −1 | [1,1,1,1] | **split** |
| **4** | 4 | 16384 | 256 | 4096 | 2 | 4 | 0 | 4 | 10 | **4** | 1 | 8 | 1 | −1 | [1,1,1,1] | **split** |
| 5 | 7 | 4409 | 256 | 4096 | 4 | 1 | 313 | 2 | 5 | (2) | (1) | (8) | (1) | (1) | [1,1] | 现行（**guard 否决**：`n_blk_full=1 < 2`） |
| 6 | 8 | 10240 | 256 | 4096 | 4 | 2 | 2048 | 3 | 5 | (3) | (1) | (12) | (1) | (2) | [1,1,1] | 现行（**guard 否决**：save 8.9us < 1.5×16.4us） |
| **7** | **4** | **152064** | **256** | **4096** | **2** | **37** | **512** | **38** | **10** | **10** | **4** | **20** | **1** | **9** | **[4,4,4,4,4,4,4,4,4,2]** | **split** |
| **8** | 4 | 16384 | 128 | 2048 | 2 | 8 | 0 | 8 | 10 | **8** | 1 | 16 | 1 | −1 | [1]×8 | **split** |
| 9 | 64 | 4096 | 256 | 4096 | 32 | 1 | 0 | 1 | 0 | **1** | 1 | 32 | 2 | −1 | [1] | 现行（`b_num=32 ≥ 20`，约束 3） |
| 10 | 4 | 4096 | 256 | 4096 | 2 | 1 | 0 | 1 | 10 | **1** | 1 | 2 | 1 | −1 | [1] | 现行（`n_blk_total=1`） |
| 11 | 8 | 8192 | 256 | 4096 | 4 | 2 | 0 | 2 | 5 | (2) | (1) | (8) | (1) | −1 | [1,1] | 现行（**guard 否决**：save 8.9us < 1.5×14.9us） |
| **12** | 16 | 32768 | 256 | 4096 | 8 | 8 | 0 | 8 | 2 | **2** | 4 | 16 | 1 | −1 | [4,4] | **split** |
| 13 | 6 | 2304 | 128 | 2048 | 3 | 1 | 256 | 2 | 6 | (2) | (1) | (6) | (1) | (1) | [1,1] | 现行（**guard 否决**：`n_blk_full=1 < 2`） |
| 14 | 32 | 1024 | 256 | 4096 | 16 | 0 | 1024 | 1 | 1 | **1** | 1 | 16 | 1 | 0 | [1] | 现行（`n_blk_total=1`） |
| 15 | 64 | 152064 | 256 | 4096 | 32 | 37 | 512 | 38 | 0 | **1** | 38 | 32 | 2 | 0 | [38] | 现行（`b_num=32 ≥ 20`，约束 3） |
| 16 | 128 | 32768 | 256 | 4096 | 64 | 8 | 0 | 8 | 0 | **1** | 8 | 64 | 4 | −1 | [8] | 现行（`b_num=64 ≥ 20`，约束 3） |
| **17** | **16** | **152377** | **256** | **4096** | **8** | **37** | **825** | **38** | **2** | **2** | **19** | **16** | **1** | **1** | **[19,19]** | **split** |
| 18 | 64 | 65536 | 256 | 4096 | 32 | 16 | 0 | 16 | 0 | **1** | 16 | 32 | 2 | −1 | [16] | 现行（`b_num=32 ≥ 20`，约束 3） |

（括号 `( )` 内的值是「几何上可 split 但被 guard 否决」的 `S`/`NBS`/`grid`，K1A 仍需能编译它们
——L0-E1 白盒用例会显式构造。）

**统计**：几何可 split（`S ≥ 2`）= **10/18**；通过 guard 实际 split = **6/18** = {3, 4, 7, 8, 12, 17}；
走现行路径 = **12/18**（其中 4 个因 `b_num ≥ 20`、5 个因 `n_blk_total = 1`、3 个因 guard）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| kernel | 循环 | 类型 | 边界 | 并行性 |
|--------|------|------|------|--------|
| **现行 K1**（不变） | `for b in T.serial(n_blk_full)` | 串行 | 编译期常量 | 行间并行（grid `b_num`），块间串行 |
| **现行 K1**（不变） | `for l in T.serial(K_MAX)`（`tau_p` 前缀和） | 串行 | 编译期常量 | 无（真串行 scan，`T.cumsum` 不可用） |
| **现行 K2m / K2g**（不变） | `for b in T.serial(n_blk)` | 串行 | 编译期常量 | 同上 |
| ⭐ **K1A** | `for lb in T.serial(NBS)` | 串行 | **编译期常量 `NBS`** | ⭐ **`(row, split)` 双维并行**（grid `b_num*S`），split 内块间串行 |
| ⭐ **K1B** | `for b in T.serial(n_blk_full)` | 串行 | 编译期常量 | 行间并行（grid `b_num`），块间串行（**必须串行：online-softmax 递推**） |
| ⭐ **K1B** | `for s in T.serial(S)` | 串行 | **编译期常量 `S`** | 无（**必须按 split-id 升序**，约束 2） |
| ⭐ **K1B** | `for l in T.serial(K_MAX)`（`tau_p`） | 串行 | 编译期常量 | 无（不变） |
| ⭐ **K2gA** | `for lb in T.serial(NBS)` | 串行 | **编译期常量 `NBS`** | ⭐ **`(row, split)` 双维并行** |
| ⭐ **K2gB** | `for s in T.serial(S)` | 串行 | **编译期常量 `S`** | 无（**必须按 split-id 升序**） |

**调度原语选择**：全部 `T.serial`。**不使用** `T.Parallel`（除 epilogue 内已有的向量 `T.tile.*`）、
`T.Pipelined`、`T.Persistent`。理由：
- `T.Pipelined` 要求静态边界（满足）但会引入 double-buffer ⟹ **UB 翻倍**。K2gA 已占 75.59%，
  double-buffer 会溢出。且 rev2 §6.3 已论证不启用。
- `T.Parallel` 用于元素级并行，本算子的元素级操作已全部由 `T.tile.*` 向量指令承担。
- **`T.serial(S)` 的顺序性是 bitwise 的必要条件**（约束 2：归约必须按固定 split-id 序）。
  `T.Parallel(S)` 会破坏顺序 ⟹ **禁止**。

### 6.2 循环伪代码

见 §3.3（完整伪代码，含全部循环与守卫）。

### 6.3 流水线优化

**rev3 不启用 `T.Pipelined`**（沿用 rev2 §6.3 结论），rev3 新增两条理由：
1. **UB 预算**：K2gA 实测 148420 B (75.59%)，double-buffer 会 `×2` 主缓冲 ⟹ 必然溢出 196352 B。
2. **跨 kernel 依赖无法流水**：K1A → K1B 是**全局**依赖（K1B 需要**所有** split 的 `pv` 写完），
   不是块内相邻迭代的依赖 ⟹ 无软件流水可言，只能靠 kernel 边界（§2.5）。

### 6.4 尾块处理

**rev2 的尾块处理逐字节保留**，rev3 只做「归属划分」：

| 尾块工作 | rev2 | rev3 split 路径 |
|---------|------|----------------|
| softmax（`zt/et/mt` 精确尺寸 `[1,TAIL]` + 全量 reduce） | K1 主 kernel 内，`if TAIL > 0:` 区段 | **K1B**，`if TAIL > 0:` 区段，**代码逐条相同** |
| top-k（`ztp[1,TAIL_PAD]` 的 `pad_value=-inf` 载入 → `ztt[TAIL_PAD]` hop → `topk(..., TAIL_PAD)`） | K1 主 kernel 内，同一 `if TAIL > 0:` 区段 | **K1A**，`if TAIL > 0:` 区段 + `if sid == S_TAIL:` 守卫，**代码逐条相同** |
| K2g 尾块 | 无独立路径（`pad_value=-inf` + keep-mask 自然排除，「tail shares full path」） | **K2gA**，同一机制，`if b < n_blk` 守卫（`n_blk = ceildiv(V,BV)` 已含尾块） |

⚠️ **尾块的 split 归属**：`S_TAIL = n_blk_full // NBS`（编译期常量）。
- case7：`n_blk_full=37, NBS=4 ⟹ S_TAIL = 9`（最后一个 split），该 split 块数 = 2（1 满块 + 1 尾块）。
- case17：`n_blk_full=37, NBS=19 ⟹ S_TAIL = 1`，该 split 块数 = 19（18 满块 + 1 尾块）。
- case5：`n_blk_full=1, NBS=1 ⟹ S_TAIL = 1`。
- case6：`n_blk_full=2, NBS=1 ⟹ S_TAIL = 2`。
- `TAIL == 0` 时 `S_TAIL = -1`，`if TAIL > 0:` 整个区段被编译期消除 ✅

⚠️ **尾块不参与 split 边界切分**：尾块永远是「块号 `n_blk_full`」这一个整体块，
被完整分配给 `S_TAIL` 这一个 split。**不存在「尾块被切成两半」** ⟹ `TAIL_PAD` 语义与 `S` 完全解耦
⟹ 这是定理 P1（`padCount==0` 保持）的关键。

### 6.5 ⚠️ 为什么不用 Developer 模式推荐的 `threads=2` + `as (cid)`（沿用 rev2，rev3 强化）

rev2 §6.5 的论证在 rev3 继续有效（`threads=2 + as (cid)` 会消除 `vid`，使 grid 语义变成
「1 block = 2 AIV 协同处理同一任务」，与本算子「一行一任务」的映射不符）。

**rev3 强化理由**：split-V 的任务空间是 **`B × S` 二维**，需要 `task = cid*VEC_NUM + vid` 这个
**已有的 2 路展开**来把二维任务映射到一维 grid。若改用 `threads=2 + as (cid)`，
则 `grid` 必须直接等于 `B*S`，而 `B*S` 最大为 `1024*20 = 20480` —— 虽然 `examples/topk_selector`
证明 `b_num=512` 可跑，但 20480 个 block 的调度开销与 `threads=2` 的语义变化都属未验证领域。
**保持 `as (cid, vid)` 形态 ⟹ 与现行 kernel 完全同构，最小改动**（AGENTS.md 核心原则 4）。

---

## 7. 同步策略

### 7.1 同步模式

**全自动同步（Developer 模式）** —— rev3 不变。

- **零手写同步原语**：不出现 `T.barrier_all()`、`T.set_flag()`、`T.wait_flag()`、
  `T.set_cross_flag()`、`T.wait_cross_flag()`、`T.pipe_barrier()`、`T.annotate_address()`。
- **零手写作用域**：不出现 `with T.Scope("C")` / `with T.Scope("V")`。
- ⭐ **rev3 新增：零跨核（跨 AI Core）同步需求**。这不是「省略了」，而是**架构上不需要**（§2.5 选项 D）。

### 7.2 同步点说明

**无手动同步点。** rev2 §7.2 的整张自动同步依赖表在 rev3 中继续有效，rev3 新增 4 行：

| 依赖 | 类型 | 同步来源 | rev3 状态 |
|------|------|---------|-----------|
| `T.copy(GM → z_ub)` 完成后再 `T.reduce_max`/`T.tile.topk` | MTE2 → V | `tl.ascend_auto_sync` | K1A / K1B / K2gA 均沿用 |
| `T.tile.topk` 写 `l_dst` 完成后再 `gather_mask(l_dst, …)` | V → V（同 buffer RAW） | `tl.ascend_auto_sync` | K1A / K1B 均沿用 |
| `T.copy(run_v → merged)` / `T.copy(pvb → merged)` 完成后再 `topk(merged, …)` | V → V RAW | `tl.ascend_auto_sync` | K1A / K1B 均沿用 |
| `T.reduce_*` 与其 `tmp_ub` arena 复用 | V → V WAR/WAW | `tl.ascend_auto_sync` + `allocate_tmp_buffer.cc:500-502` 的 `kWorkspaceWriteAccess` 依赖建模 | 沿用 |
| buffer 生命周期不重叠时的地址复用 | — | `tl.ascend_memory_planning` | 沿用 |
| ⭐ **K1A 写 `pv[row*S+sid, :]` → K1B 读 `pv[row*S+s, :]`** | **跨 kernel，GM RAW** | **host 侧顺序 launch**（同一 stream 上的 kernel 天然串行化）。**不需要任何核内/核间同步原语** | ⭐ rev3 新增。与 rev2 的「K1 → K2 经 GM 传递 `m/s/tau/valid`」**完全同一机制** |
| ⭐ **K1A 内部各 split 之间** | **无依赖** | 每 `(row, sid)` 写 `pv` 的**独占行** ⟹ 无 WAW/WAR/RAW | ⭐ rev3 新增：**零同步需求** |
| ⭐ **K2gA 写 `pv_val/pv_idx` → K2gB 读** | 跨 kernel，GM RAW | 同上（host 顺序 launch） | ⭐ rev3 新增 |
| ⭐ **K2gA 内部各 split 之间** | 无依赖 | 独占 `pv_val[row*S+sid]` / `pv_idx[row*S+sid]` | ⭐ rev3 新增：**零同步需求** |
| **跨核（跨 AI Core）归约** | — | **不需要**：rev3 把「必须串行的归约」全部放进 `grid = b_num` 的单个 kernel（K1B/K2gB），该 kernel 内部一任务一核、无共享 buffer、无 `atomic_add` | ⭐ **rev3 的关键性质**（§2.5 选项 D） |
| **Cube↔Vector 核间同步** | — | 不适用（零 Cube 操作）；`tl.ascend_auto_cross_core_sync` 仍开启以与仓库惯例一致（无害空转） | 不变 |

> ⚠️ 跨代际陷阱（rev2 §7.2 引 Programming Guide `:2330`）：*「A2 会在算子之间清理多余计数，
> 而 Ascend 950 要求严格配对」* → 手工 MTE 事件漏配对在 A2/A3 能跑过、A5 会挂。
> **rev3 与 rev2 一样不使用任何显式 MTE 事件配对**（全依赖 `tl.ascend_auto_sync`），故不受影响。
> **rev3 进一步强化**：连「跨核」这个概念都不引入 ⟹ A5 迁移时 rev3 的同步面与 rev2 完全相同。

### 7.3 pass_configs 配置

见 §4.7（`R4_PASS_CONFIGS` 四项全开，rev3 全部 7 个 kernel 一律使用，一字不改）。

**⭐ rev3 强制要求（安全网依赖，Orchestrator §6）**：
`src/transform/ascend_memory_planning.cc:598` 的兄弟路径是 `check_overflow = false`（**静默溢出**）；
R4 只因开了 `tl.ascend_memory_planning=True` 才走 `LOG(FATAL)` 硬失败（`:606`）。
⟹ **新增的 K1A/K1B/K2gA/K2gB 必须全部保持该 pass_config 开启**，否则 UB 溢出会静默产生错误结果
而非编译失败。已在 §3.3 伪代码的 `@tilelang.jit(pass_configs=R4_PASS_CONFIGS)` 中体现，
并在 §9.2 L0 计划中列为**强制静态检查项**（Stage 2 必须断言 4 个新 kernel 的 `pass_configs`
与 `R4_PASS_CONFIGS` 相同）。

**rev2 §7.3 的「唯一允许调整」条款继续有效**：若 Stage 2 发现开启 ③（AUTO_CV_COMBINE）导致
纯 Vector kernel 的 `alloc_var` 定义/使用核归属误判，可单独关闭 ③④ 而保留 ①②。
⚠️ **rev3 追加限制**：**② `tl.ascend_memory_planning` 绝对不可关闭**（安全网）。

---

## 8. 融合算子设计

### 8.1 融合算子判定

**是融合算子**（Fusion）：K1 + K2 两个（rev3 split 路径下最多 4 个）kernel 构成一条链，
经 GM 传递中间量。rev3 未改变此判定，只增加了链长。

**CV 融合判定：不适用**（零 Cube 计算）。

### 8.2 CV 交互设计

**不适用**（纯 Vector，无 Cube 核参与）。rev3 亦不引入任何 Cube 操作。
生成码为 `KERNEL_TYPE_MIX_AIC_1_2` + 空 L0A/L0B 分配（各 64 KB），Stage 3 round1 [#8] 已判定
「无害空转，并发度不变，无占用收益」，rev3 沿用该结论，**不调整 `tl.ascend_auto_cv_combine`**。

**⭐ rev3 特别说明：split-V 没有引入 workspace/vid 的 CV 语义变化。**
rev3 的 `pv`/`pv_val`/`pv_idx` 是**跨 kernel 的 GM 数据通路**（与 rev2 的 `m_out/s_out/tau_out`
同质），**不是** CV 融合语境下的「Cube→Vector workspace」。design-template §8.2 所说的
「Developer 默认消除 workspace/vid」针对的是 CV 融合，与本设计无关；
本设计的 `vid` 是 `VEC_NUM=2` 的 AIV 子核编号（rev2 起即如此，§6.5 已论证不可消除）。

### 8.3 / 8.4 Cube 核 / Vector 核计算流程

**Cube 核**：无计算（空转）。
**Vector 核**：见 §3.3 伪代码 + §6.1 循环表。

### 8.5 pass_configs 配置

见 §4.7 / §7.3。

### 8.6 注意事项

1. **kernel 链长从 2 增至最多 4**（split 路径 + fused 任务）。链上任一 kernel 失败都会导致
   后续 kernel 读到未初始化的 workspace ⟹ Stage 2 的调试必须**逐 kernel 验证**
   （`T.dump_tensor` / 单独调用 K1A 后 D2H 检查 `pv`），不能只看最终输出。已写入 §10.3。
2. **workspace 不做 zero-init**（§5.6 论证）⟹ 若 K1A 因 bug 漏写某行，K1B 会读到垃圾。
   **调试期**建议临时改用 `torch.zeros` 以区分「漏写」与「算错」；**交付版必须用 `torch.empty`**
   （避免 aclnn 填充算子进入 host 路径）。
3. **launch 顺序不可交换**：K1A 必须在 K1B 前，K2gA 必须在 K2gB 前，K1B 必须在 K2gA 前
   （K2gA 读 `tau_out`）。同一 stream 上的顺序 launch 天然满足；**禁止**用多 stream 并发。

---

## 9. 验证方案

### 9.0 算子类别判定（`tilelang-op-test-design` §3.2 流程）

| 维度 | 判定 | 依据 |
|------|------|------|
| 计算类型 | **纯 Vector** | §1.3 公式只含 `max / exp / sum / div / ln / sort / argmax`，**无 `matmul`/`@`**；生成码零 Cube 计算（Stage 3 round1 [#8] 实证） |
| 复杂度级别 | **Fusion** | 多 kernel 链：rev2 为 K1→K2（2 个）；**rev3 split 路径为 K1A→K1B→K2gA→K2gB（4 个）** |
| 数学特征 | **Softmax 类 + Reduction 类 + Selection（top-k / argmax）混合** | `exp+sum+div` 组合（Softmax 类）；`max(dim)/sum(dim)`（Reduction 类）；`sort_desc[k-1]` 与 `argmax`（Selection） |
| 综合类别 | **Fusion（Softmax-hybrid sampling chain）** | 测试策略取「Fusion」+「Softmax（精度按 dtype）」的并集 |

**⟹ 测试策略要点**（`tilelang-op-test-design` §2.2/§2.5）：
1. Fusion 类 ⟹ 必须测**完整数据流**（不能只测单 kernel）；rev3 追加**逐 kernel 中间量验证**（§10.3）。
2. Softmax 类 ⟹ **精度按 dtype**（§9.3 混合容差表）。
3. Reduction 类 ⟹ **归约维度/归约顺序**是重点；rev3 的核心风险正是「归约顺序」⟹ **§9.2 的 Tier-2 bitwise 对照是本算子 rev3 的首要门禁**，优先级高于容差判定。
4. ⚠️ rev3 属 `tilelang-op-test-design` §4.1 Phase 3.5 所说的「融合算子 + 多步复杂算子」⟹ **闸门确认适用**：本 L0 计划即为闸门内容，Stage 2 落地前不得自行删减用例。

### 9.1 Golden 函数（`examples/R4/r4_ref.py`，rev3 **不变**）

rev3 **不修改 golden**。`r4_ref.py` 的 `golden_r4(logits, top_ks, top_ps, min_ps,
need_min_p_sampling, K_MAX=256, top_k_all=TOP_K_ALL)`（`:17`）与
`golden_gumbel_sample(logits, tau, rand)`（`:223`）+ `golden_hash` / `golden_hash_batch`（`:172/:202`）
继续作为唯一权威参考。

**理由**：split-V 是**实现路径变更**，不是**语义变更**（§5.7：`use_split` 只是同一
`supported_domain` 内的路径选择）。golden 描述的是语义 ⟹ 不变。

**rev2 §9.1 记录的关键修正继续有效**（Stage 1 原版在 `tau_p`/`L` 处用 top-k 屏蔽后的 `q` 求前缀和
→ nucleus>k 时 `L` 虚增 → `ok_p` 误判 0 → `valid` 误判；Stage 2 已修正为基于 **raw `probs_sort`
（未屏蔽）的严格前缀和**）。完整 golden 代码见 `history_version/design_rev2.md` §9.1
与 `examples/R4/r4_ref.py`（二者一致，rev2 已核对）。

**Golden 的实现约束**（rev2 §9.4，rev3 沿用）：测试数据、随机数、特殊值和 golden 的
排序/掩码/scatter 重排**均在 CPU 完成**（`golden_r4` 内已 `.cpu()` + float64）；
运行阶段只做 **H2D → TileLang kernel → D2H**，避免测试 harness 引入额外 aclnn 依赖。

### 9.2 ⭐ L0 门槛测试计划（`tilelang-op-test-design` **场景 A** 产出，rev3 重做）

> **L0 定位**：规则 shape 为主、快速冒烟、供 Stage 2 精度收敛。
> **rev3 用例数 = 15**（rev2 为 8）。**超出 skill 推荐的 ≤10**，理由：
> ① rev3 是**架构级改动**，必须覆盖双路分派的**两侧**与**边界**；
> ② 用户 §9 明确列出 5 类强制覆盖点（`n_blk_full >> 2`、分派边界、双路 bitwise 一致性、
> `S` 边界、`TAIL%32!=0` 全谱），任何一类缺失都会让 L0 失去门禁意义；
> ③ 除 A2/B5/C2/C3/D1（`V=152064` 或 `B≥22`）外，其余 10 例都是小 `V`（≤ 32768），
> 编译与运行都快。**Stage 2 不得删减本表任何用例**（§9.0 要点 4）。
>
> **L0 明确不含**（留给 Stage 2 在 L0 通过后调 `tilelang-op-test-design` **场景 B** 扩展 L1/L2/Boundary）：
> 不规则/质数 `V`（除 §9.2.3 B 组的 5 个尾块谱）、`B=1`、`B=1024`、`k ∉ [1,K_MAX]`、
> `TOP_K_ALL = 1<<30`、`p` 极小（触发 `ok_p=0`）、全等行（触发 `tie_ok=0`）、含 `-inf` 输入、
> 含 NaN 输入、bf16（不支持，L2 负向）、空 tensor（`B=0`，L2 负向）、
> `LOGIT_SCALE ≠ 8.0` 的平坦分布（L2/Boundary 负向守卫验证）、`--level stress` 的 200 reps 深压。

#### 9.2.1 🔴 测试数据配方（**强制**，沿用 rev2；否则 L0 会因守卫误触发而全批假失败）

```python
torch.manual_seed(0)                 # 确定性
LOGIT_SCALE = 8.0                    # ⚠️ 强制值，实测论证见 rev2 §9.2.1（本文件 history_version 备份）
logits = torch.randn(B, V, dtype=torch.float32) * LOGIT_SCALE
top_ks = torch.full((B,), k,   dtype=torch.int32)
top_ps = torch.full((B,), p,   dtype=torch.float32)
min_ps = torch.full((B,), mu,  dtype=torch.float32)
rand   = <确定性 int32 种子，沿用现有 test 套件的 _make_inputs 配方>
```

**为什么必须 `LOGIT_SCALE = 8.0`**（rev2 已用 CPU 实测论证，rev3 沿用）：`K_MAX = 256` 意味着
top-p 的 nucleus 必须能装进 256 个候选，否则 D-GUARD 的 `ok_p = 0` ⟹ `valid = 0` ⟹ 整批回退，
L0 无法验证快路径。rev2 的逐 scale 实测表：`scale ∈ {1,3,5,6,7}` 在 `V=152064` 下 nucleus 分别为
`{112793, 15735, 844, 265, —}`，**全部 ≥ 256 ⟹ 守卫全触发**；`scale=8.0` 时 nucleus 最大 69
（余量 ≥ 3.7×），且在完整用例矩阵上 `valid` 全 1；`scale=10.0` 虽也全 1 但 `kept` 恒为 1~2，判别力弱。
⟹ **`8.0` 是唯一同时满足「守卫不触发」与「nucleus 有足够多样性」的取值。**

**⭐ rev3 新增：本 Stage 1 已在 CPU 上对全部 15 个 L0 shape 实跑 `golden_r4` 复核**
（`torch.manual_seed(0)`、`LOGIT_SCALE=8.0`、逐例调用 `r4_ref.golden_r4`）：

| 用例 | B | V | K_MAX | k | p | μ | `NEED_MIN_P` | `valid_ref` 全 1 |
|---|---|---|---|---|---|---|---|---|
| A1 | 16 | 32768 | 256 | 20 | 0.9 | 0.1 | True | ✅ **True** |
| A2 | 4 | 152064 | 256 | 40 | 0.95 | 0.05 | True | ✅ **True** |
| A3 | 8 | 16384 | 256 | 40 | 0.95 | 0.05 | True | ✅ **True** |
| A4 | 4 | 16384 | 128 | 40 | 0.95 | 0.0 | False | ✅ **True** |
| B1 | 4 | 20737 | 256 | 40 | 0.95 | 0.05 | True | ✅ **True** |
| B2 | 4 | 20793 | 256 | 40 | 0.95 | 0.05 | True | ✅ **True** |
| B3 | 4 | 20991 | 256 | 40 | 0.95 | 0.05 | True | ✅ **True** |
| B4 | 4 | 20741 | 256 | 40 | 0.95 | 0.05 | True | ✅ **True** |
| B5 | 16 | 152377 | 256 | 40 | 0.95 | 0.05 | True | ✅ **True** |
| C1 | 20 | 32768 | 256 | 20 | 0.9 | 0.1 | True | ✅ **True** |
| C2 | 22 | 32768 | 256 | 20 | 0.9 | 0.1 | True | ✅ **True** |
| C3 | 40 | 32768 | 256 | 20 | 0.9 | 0.1 | True | ✅ **True** |
| D1 | 4 | 152064 | 256 | 40 | 0.95 | 0.05 | True | ✅ **True** |
| E1 | 2 | 16384 | 256 | 40 | 0.95 | 0.0 | False | ✅ **True** |
| F1 | 10 | 20480 | 256 | 40 | 0.95 | 0.05 | True | ✅ **True** |

**15/15 全部 `valid_ref` 全 1** ⟹ 本 L0 计划**不会**因守卫误触发而假失败。

> ⚠️ **Stage 2 注意**（rev2 原文，rev3 沿用并加强）：若改用其他 scale 或其他随机分布，
> **必须先跑 `golden_r4` 确认 `valid_ref` 全 1**，否则 L0 会「假失败」，容易误判为 kernel bug
> 甚至误报 `[DESIGN_ERROR]`。这一条是 L0 计划的一部分，不可省略。
> ⚠️ rev3 追加：B1–B4 的 `V ∈ {20737, 20741, 20793, 20991}` 是**为本 L0 新造的 shape**
> （不在 cann-bench 18-case 内），其 `TAIL ∈ {257, 261, 313, 511}` 覆盖了 `%32` 余数
> `{1, 5, 25, 31}` 四个值；加上 B5 的 `TAIL=825`（余 25，`TAIL_PAD=832`）与 A2 的 `TAIL=512`（余 0，对齐），
> **尾块对齐谱共 6 个点**，与现有 STAB/ADV 套件的 `TAIL ∈ {313, 257, 511, 261, 825}` **完全对应**（§9.2.4）。

#### 9.2.2 ⭐ rev3 双层验收判据（L0 的核心变化）

rev3 的 L0 每个用例都必须**同时**通过两层，缺一不可：

| 层 | 对照对象 | 判据 | 说明 |
|---|---|---|---|
| **Tier 1**（沿用 rev2） | `golden_r4` / `golden_gumbel_sample`（CPU float64 参考） | §9.3 的 **dtype 混合容差**：`matched_ratio ≥ required_matched_ratio` **且** `max_abs_error ≤ max_abs_error_limit`；int32 逐元素精确相等 | 与 rev2 L0 完全相同 |
| ⭐ **Tier 2**（rev3 新增，**更严**） | **现行 kernel**（`r4_topk_stats` / `r4_gumbel_sample` / `r4_mask_sample`，sha256 `19e51e8d…`）在同一输入上的输出 | **逐位相同**：`torch.equal(actual.view(torch.int32), baseline.view(torch.int32))`（对 fp32 用 `view(int32)` 比原始位模式，确保 `±0`、`-inf`、`NaN` 位模式都一致；对 int32 直接 `torch.equal`） | ⭐ **这是约束 2「必须 bitwise 等价」的直接可执行断言。** Tier 1 通过但 Tier 2 失败 ⟹ **判 L0 FAIL**（说明出现了 ulp 级偏差，虽然容差内但违反约束 2，会在 `tau_p` 前缀和边界处偶发翻转 `valid`/`tok_out`） |

**Tier 2 的实现方式（不改契约）**：测试文件直接 import 并调用**模块级 `@tilelang.jit` 函数**——
split 侧调 `r4_topk_stats_splitA` + `r4_topk_stats_splitB`（自己 `torch.empty` 出 `pv`），
基准侧调 `r4_topk_stats`。两侧都是**同一模块的公开函数**，
⟹ **无需给 host wrapper 增加任何 `_force_path` 之类的私有 kwarg，`r4_fused_sampling` 的签名与
`proto.yaml` 的 schema 完全不动**（§15.1）。

```python
# Tier-2 断言的最小实现骨架（Stage 2 落地到 test_r4_fused_sampling.py 的 L0 层）
def _bits(t):                      # fp32/int32 -> 可比位模式的 int32 视图
    return t.detach().cpu().view(torch.int32) if t.dtype == torch.float32 else t.detach().cpu()

def _assert_bitwise(tag, name, actual, baseline):
    ok = (actual.shape == baseline.shape and actual.dtype == baseline.dtype
          and torch.equal(_bits(actual), _bits(baseline)))
    print(f"[BITWISE_{'PASS' if ok else 'FAIL'}] {tag} {name} "
          f"shape={tuple(actual.shape)} dtype={actual.dtype}")
    return ok
```

**Tier 2 的适用范围**：
- **A / B / C1 / F 组（split 路径）**：Tier 1 + Tier 2 都要（与现行 kernel 对照）。
- **C2 / C3 组（现行路径）**：只需 Tier 1 + **一个结构性断言**：`use_split == False`
  （直接断言 host 分派函数返回的 `S == 1`），因为它们走的就是现行 kernel，Tier 2 恒真、无信息量。
- **D1 / E1 组**：**Tier 2 就是用例本身**（专门的 bitwise 对照用例，见 §9.2.3）。

#### 9.2.3 L0 用例矩阵（15 个；`golden_r4` 已在 CPU 上逐例实跑验证 `valid_ref` 全 1）

**通用配置**：`dtype = float32`（V01-a/b 唯一支持的计算 dtype）；`VEC_NUM = 2`；
`CUBE_CORE_NUM = 20`（910b2 实测，运行时探测）；`LOGIT_SCALE = 8.0`；`torch.manual_seed(0)`；
两个任务（`FUSE_GUMBEL=False` 的 mask 任务 与 `=True` 的 fused 任务）**各跑一遍**
⟹ 15 shape × 2 任务 = **30 次运行**。

**A 组 —— split 路径基本功能（4 例）**

| 用例名 | B | V | K_MAX | BV | b_num | n_blk_full | TAIL | **S** | NBS | per-split | 覆盖目标 |
|--------|---|---|---|---|---|---|---|---|---|---|---------|
| `l0_A1_split_s2_aligned` | 16 | 32768 | 256 | 4096 | 8 | **8** | 0 | **2** | 4 | [4,4] | **最小非平凡 S**（S=2）+ **`n_blk_full=8 > 2`** + block 整除无尾块 + 每 split 多块（4）。对应 cann-bench **case12** |
| `l0_A2_split_smax_prodV` | 4 | 152064 | 256 | 4096 | 2 | **37** | 512 | **10** | 4 | [4×9, 2] | ⭐ **S = 最大值（10，grid 恰好 20 = 满 1 波）** + **`n_blk_full=37 >> 2`** + 对齐尾块（`TAIL%32==0`）+ **split 负载不均**（4 vs 2）+ 生产 V。对应用户 §9「`n_blk_full` 显著 >2」与「S=最大值」两项强制点；对应 cann-bench **case7** |
| `l0_A3_split_s4_oneblk` | 8 | 16384 | 256 | 4096 | 4 | 4 | 0 | **4** | 1 | [1,1,1,1] | ⭐ **每 split 恰好 1 块**（`NBS=1` ⟹ split 内循环只跑 1 次，`run_v` 的 `-inf` 初值只被挤出一次）—— 用户 §9「S 使某 split 只有 1 块」的强制点。对应 cann-bench **case3** |
| `l0_A4_split_k128_bv2048` | 4 | 16384 | **128** | **2048** | 2 | **8** | 0 | **8** | 1 | [1]×8 | ⭐ **第二个契约特化 `(K_MAX=128, BV=2048)`**（约束 7：必须同时正确处理两种特化）+ S=8。对应 cann-bench **case8** |

**B 组 —— `TAIL % 32 != 0` 在 split 路径下的强制覆盖（5 例；精度红线缺陷 A/B 的回归面）**

| 用例名 | B | V | TAIL | `TAIL%32` | TAIL_PAD | K_MAX | BV | S | NBS | S_TAIL | 覆盖目标 |
|--------|---|---|---|---|---|---|---|---|---|---|---------|
| `l0_B1_split_tail257` | 4 | 20737 | **257** | **1** | 288 | 256 | 4096 | 6 | 1 | 5 | 最小非对齐余数（1）。`TAIL_PAD−TAIL = 31` ⟹ **31 个 `-inf` 填充槽**，最大化「陈旧 UB / 截断」的暴露面。对应现有 `stab_B3_V4353`（TAIL=257） |
| `l0_B2_split_tail313` | 4 | 20793 | **313** | **25** | 320 | 256 | 4096 | 6 | 1 | 5 | **rev2 原始缺陷 shape**（cann-bench case5 的 TAIL）。对应现有 `stab_case5_B7_V4409` + `ADV_CASES (2,4409)` |
| `l0_B3_split_tail511` | 4 | 20991 | **511** | **31** | 512 | 256 | 4096 | 6 | 1 | 5 | **最大可能非对齐余数（31）**。对应现有 `stab_B9_V4607` + `ADV_CASES (2,4607)` |
| `l0_B4_split_tail261` | 4 | 20741 | **261** | **5** | 288 | 256 | 4096 | 6 | 1 | 5 | 覆盖 `TAIL=261`（用户 §9 明列）。`261 ≥ K_MAX=256` 恰好满足 C3 ⟹ **C3 判据的下边界**。对应现有 `ADV_CASES (5,261)` |
| `l0_B5_split_tail825` | 16 | 152377 | **825** | **25** | **832** | 256 | 4096 | **2** | **19** | **1** | ⭐ **大 `n_blk_full`（37）+ 非对齐大尾块（832）+ 小 S（2）** 的组合；`S_TAIL=1` ⟹ 尾块落在**非零号 split**（B1–B4 的 `S_TAIL=5` 也是非零，但 B5 的尾块 split 还同时背 18 个满块 ⟹ 负载最重）。对应 cann-bench **case17**（rev2 修复的回归面） |

> **B 组的 `TAIL` 谱 = {257, 261, 313, 511, 825}**，与用户 §9 要求「保留既有 `TAIL%32!=0` 强制覆盖
> （TAIL=313/257/511/261/825）」**逐项对应，一个不缺**。`%32` 余数谱 = {1, 5, 25, 31, 25}，
> 覆盖最小(1)、最大(31)、以及 rev2 实际踩坑的 25。

**C 组 —— 双路分派边界（3 例；用户 §9「`b_num<20` 与 `b_num>=20` 两侧的分派边界用例」）**

| 用例名 | B | V | b_num | `S_cap = 20//b_num` | **S** | 期望路径 | 覆盖目标 |
|--------|---|---|---|---|---|---|---------|
| `l0_C1_dispatch_bn10_split` | **20** | 32768 | **10** | **2** | **2** | ⭐ **split** | **分派边界的 split 侧最后一个值**（`b_num=10` ⟹ `S_cap=2` ⟹ 可 split）。断言：`S == 2` 且走 split 路径 且 Tier-1/Tier-2 通过 |
| `l0_C2_dispatch_bn11_current` | **22** | 32768 | **11** | **1** | **1** | ⭐ **现行** | **分派边界的现行侧第一个值**（`b_num=11` ⟹ `S_cap=20//11=1` ⟹ 不可 split）。断言：`S == 1` 且 `use_split == False` 且输出与 rev2 交付版一致。⭐ 与 C1 构成 **`b_num=10 → 11` 的锐利边界对**（§0.2 收紧判据的直接验证） |
| `l0_C3_dispatch_bn20_current` | **40** | 32768 | **20** | **1** | **1** | ⭐ **现行** | **用户约束 3 的字面边界**（`b_num >= 20` 走现行单波路径，代码逐字节不变）。断言：`use_split == False`；且此 case 的 `grid = b_num = 20` 恰好 1 波 ⟹ 验证「现行路径 = 单波」的原始设计意图 |

> ⚠️ **C 组同时验证了 §0.2 的判据收紧**：用户字面规则会在 `b_num ∈ [11,19]` 走 split，
> 本设计走现行。C2（`b_num=11`）是这一差异的**唯一可观测点**，Stage 2 必须断言其为**现行路径**。
> 若 Orchestrator/用户否决该收紧，则 C2 的期望值需改为 split（且 §13.4 的 `S_cap` 公式需改为
> `ceil`-based 多波版本 —— 但定理 P2 证明那样严格无益，本设计强烈不建议）。

**D 组 —— 双路 bitwise 一致性（1 例；用户 §9「同一输入分别走 split 路径与现行路径，断言逐位相同」）**

| 用例名 | B | V | K_MAX | BV | S | 覆盖目标 |
|--------|---|---|---|---|---|---------|
| `l0_D1_dualpath_bitwise` | 4 | 152064 | 256 | 4096 | **10** | ⭐⭐ **约束 2 与约束 3 的直接验证**。同一份 `logits/top_ks/top_ps/min_ps/rand` 输入，**分别**走 ①split 路径（K1A→K1B→K2gA→K2gB，以及 K1A→K1B→K2m）与 ②现行路径（K1→K2g，以及 K1→K2m），对 **5 个输出逐个** 断言 `torch.equal` 于**位模式**：`m_out`、`s_out`、`tau_out`（fp32，`view(int32)` 比较）、`valid`（int32）、`tok_out`（int32）/ `w_out`（fp32 `[B,V]`，**全 608256 个元素逐位**）。**任一元素不等即 FAIL** |

**E 组 —— `S` 退化边界（1 例；用户 §9「S=1（退化）」）**

| 用例名 | B | V | K_MAX | BV | 强制 S | NBS | 覆盖目标 |
|--------|---|---|---|---|---|---|---------|
| `l0_E1_split_S1_degenerate` | 2 | 16384 | 256 | 4096 | **1**（白盒直接调 K1A/K1B 并传 `S=1, NBS=4`） | 4 | ⭐⭐ **把 split 代码路径退化为「单 split」并与现行 kernel 逐位对照**。这是**信息量最大的单个用例**：它把「分解本身是否正确」（K1A 只做 topk、K1B 只做 softmax+归并+epilogue、`pv` 往返无损）与「S>1 的并行归并是否正确」**完全解耦**。若 E1 通过而 A2 失败 ⟹ 问题在跨 split 归并；若 E1 失败 ⟹ 问题在分解本身。⚠️ 注意 `S=1` 时 host 判据会走现行路径，故本用例必须**白盒直接调用** K1A/K1B（不经 host wrapper） |

**F 组 —— 空 split 收缩（1 例；用户 §9「S 使某 split 为空（若可能）」）**

| 用例名 | B | V | K_MAX | BV | b_num | n_blk_total | `S_cap` | 朴素 `S` | NBS | **收缩后 S** | per-split | 覆盖目标 |
|--------|---|---|---|---|---|---|---|---|---|---|---|---------|
| `l0_F1_split_shrink_noempty` | **10** | **20480** | 256 | 4096 | **5** | **5** | **4** | **4** | **2** | ⭐ **3** | **[2,2,1]** | ⭐ **定理 P3 的直接验证**：朴素 `S = min(S_cap, n_blk_total) = min(4,5) = 4` 会给出 `NBS = ceil(5/4) = 2` ⟹ 各 split 块数 `[2,2,1,0]`，**split 3 为空**。回代收缩 `S = ceil(5/2) = 3` ⟹ `[2,2,1]`，**无空 split**。断言：① host 分派函数返回 `S == 3`（**不是 4**）；② 每 split ≥1 块；③ Tier-1 + Tier-2 通过（若真出现空 split，K1B 会读到未写入的 `pv` 行 ⟹ `run_v` 被垃圾污染 ⟹ `tau_out` 错） |

> 📝 **关于「S 使某 split 为空（若可能）」的回答**：**在 rev3 的 `S` 算法下不可能**（定理 P3 是构造性保证）。
> F1 是**验证该保证真的生效**的用例（即验证步骤 4 的回代收缩没有被漏写），而不是「构造一个空 split」。
> 若 Stage 2 误把步骤 4 删掉，F1 会失败 ⟹ F1 是步骤 4 的**回归哨兵**。

#### 9.2.4 L0 与既有测试套件的关系（强制保留项）

| 既有层 | 内容 | rev3 处置 |
|--------|------|----------|
| `--level all`（L0a hash 单元 + L1 + L2 + Boundary，74 PASS / 38 BOUND） | rev2 交付版全绿 | **必须全部继续通过**。L1/L2/Boundary 的用例由 Stage 2 用 `tilelang-op-test-design` **场景 B** 扩展（增加 split 路径的对应覆盖），**不得删减现有用例** |
| `--level stability`（`STAB_CASES` 8 配置 × `STAB_REPS=20`，12 项） | 重复 launch 逐位相同 + `_tau_invariant` | **必须对 split 路径同样施加**。⚠️ 现有 `STAB_CASES` 的 8 个配置中，`B=7/1/3/9/5` 的 `b_num ∈ {4,1,2,5,3}` ⟹ `S_cap ≥ 4`，但 `n_blk_full` 分别为 `{1,1,1,1,2}` ⟹ **只有 `stab_B5_V10240`（`n_blk_full=2`）可能被 guard 接受**，而它恰好会被 guard 否决（save 8.9 < 16.4）⟹ **现有 STAB 全部走现行路径，对 split 路径零覆盖**。Stage 2 **必须新增 split 路径的 STAB 配置**（至少 A2/B5/F1 三个 shape × 20 reps） |
| `--level stress`（`_make_ub_poisoner` UB 投毒 + 200 reps，8/8） | 投毒 `1e30` 后立刻 launch，断言无垃圾值 | ⭐ **必须施加到 K1A/K1B/K2gA/K2gB**（§4.9 末尾要求）。这是精度红线缺陷 A 的**唯一运行时检测手段** |
| `_run_tail_adversary`（`ADV_CASES` 4 个 shape，哨兵最大值扫遍每个尾块 index） | 检测尾块元素被静默丢弃 | **必须在 split 路径下同样跑一遍**（尾块现在由 `S_TAIL` 号 split 处理，adversary 哨兵能验证「尾块被正确的 split 处理且未丢元素」） |
| 覆盖门禁 `coverage_check.py --proto`（25/0/0） | `proto.yaml` 派生 D-DTYPE-*/D-PARAM-* | **`proto.yaml` 不变 ⟹ 应覆盖维度不变 ⟹ 25/0/0 必须保持**。Stage 2 新增的 L1 用例需带 `tags=`（含 rev3 新维度，见下） |

**⭐ rev3 建议新增的覆盖维度 tag**（供 Stage 2 场景 B 扩展时标注；本 Stage 1 **不修改** `coverage_check.py`
与 `proto.yaml`，因为二者是契约与门禁工具，不属 Stage 1 交付范围）：

| tag | 谓词 | L0 命中 |
|-----|------|---------|
| `D-PATH-SPLIT` | `use_split == True` | A1–A4, B1–B5, C1, D1, F1 |
| `D-PATH-CURRENT` | `use_split == False` | C2, C3, E1(对照侧), D1(对照侧) |
| `D-SPLIT-S-MAX` | `S == S_cap` 且 `grid == CUBE_CORE_NUM` | A2（S=10, grid=20） |
| `D-SPLIT-S-MIN` | `S == 2` | A1, B5, C1, D1… |
| `D-SPLIT-NBS1` | `NBS == 1`（每 split 恰 1 块） | A3, A4, B1–B4 |
| `D-SPLIT-SHRINK` | 步骤 4 回代收缩生效（`S < min(S_cap, n_blk_total)`） | F1 |
| `D-SPLIT-UNEVEN` | `max(per) != min(per)`（负载不均） | A2（4 vs 2）, F1（2,2,1） |
| `D-SPLIT-TAILOWNER` | `S_TAIL > 0`（尾块不在 0 号 split） | A2(9), B1–B4(5), B5(1), F1(1) |
| `D-BITWISE-DUALPATH` | Tier-2 双路逐位对照 | 全部 A/B/C1/F + D1 + E1 |

#### 9.2.5 L0 通过判据（Stage 2 精度收敛门禁）

```
15 个用例 × 2 个任务（mask / fused）= 30 次运行。每次运行须满足：

  Tier 1（vs golden_r4 / golden_gumbel_sample，dtype 混合容差）：
      5 个输出全部满足
        matched_ratio >= required_matched_ratio(dtype)  AND  max_abs_error <= max_abs_error_limit(dtype)
      且 valid 逐元素精确相等（int32，0 误差），且 valid_ref 全 1（§9.2.1 已 CPU 预验证 15/15）

  Tier 2（vs 现行 kernel 19e51e8d…，仅 split 路径用例）：
      5 个输出全部 torch.equal 于位模式（fp32 经 view(int32)）
      含 w_out 的 [B,V] 全部 B*V 个元素（B5: 16*152377 = 2438032 个；A2/D1: 4*152064 = 608256 个）

  结构断言（C/E/F 组）：
      C1: S == 2 且 use_split == True
      C2: S == 1 且 use_split == False        ← §0.2 判据收紧的验证点
      C3: S == 1 且 use_split == False        ← 约束 3 字面边界的验证点
      E1: 强制 S=1 时 split 路径输出 == 现行路径输出（逐位）
      F1: S == 3（非 4）且 min(per_split_blocks) >= 1

  静态断言（全部用例前置，一次性）：
      ① 4 个新 kernel 的 pass_configs == R4_PASS_CONFIGS（含 tl.ascend_memory_planning=True，§7.3 安全网）
      ② 现行 3 个 kernel 的源码 sha256 段落未变（约束 3「逐字节保留」）
      ③ 4 个新 kernel 的实测 UB <= 196352（§4.5；用 ub_probe.py 的解析法）
      ④ 4 个新 kernel 生成源中全部 TopK 调用点 padCount == 0，
         全部 copy_ub_to_ub 的 count*4 % 32 == 0（§4.9；用 refine3.py 的正则法）

⟹ 全部满足：打印 [PRECISION_PASS] + [BITWISE_PASS]，测试进程 exit(0)
  任一失败：打印 [PRECISION_FAIL] 或 [BITWISE_FAIL]，exit(1)，Stage 2 进入精度调试
            （仅当定位为设计层缺陷时才返回 [DESIGN_ERROR]）
```

**单 case timeout 预算**：
- 小 V 用例（A1/A3/A4/B1–B4/C1/C2/E1/F1，`V ≤ 32768`）：**300 s**（含首次 JIT；每个新
  `(B,V,K_MAX,BV,NEED_MIN_P,S)` 组合触发一次 bisheng dav-2201 编译，秒~分钟级）
- 大 V / 大 B 用例（A2/B5/C3/D1，`V = 152064` 或 `B = 40`）：**900 s**
- **L0 全量预算：≤ 4500 s**（15 shape × 2 任务，其中 4 个大用例占主要时间；JIT 缓存命中后重跑 ≪ 此值）

### 9.3 精度标准

> 采用**混合容差**：逐元素 `|actual-golden| <= atol + rtol·|golden|`，整体判定
> `matched_ratio >= required_matched_ratio` **且** `max_abs_error <= max_abs_error_limit`。
> 阈值**仅按 dtype**（与算子类别无关）。完整定义见
> `tilelang-op-test-design/references/precision-standard.md`。

| dtype | atol | rtol | max_abs_error_limit | required_matched_ratio | 本算子适用的输出 |
|-------|------|------|---------------------|------------------------|-----------------|
| **float32** | 2⁻¹⁶ = 1.53e-5 | 2⁻¹⁰ = 9.77e-4 | 1e-2 | 0.99 | `m_out`, `s_out`, `tau_out`, `w_out`, 以及输入 `logits`/`top_ps`/`min_ps` |
| **int32** | 0 | 0 | 0 | 1.0（精确匹配） | `valid`, `tok_out`（输出）, `top_ks`, `rand`（输入） |

**§4 声明的每个 dtype 均已在上表有对应行**（float32 与 int32，共 2 行，无其他 dtype —— bf16 属范围外）。
**rev3 未改变本表任何数值**（`proto.yaml` 不变 ⟹ 覆盖门禁派生的 `D-DTYPE-*` 不变）。

> ⭐ **rev3 追加：本表是 Tier-1 判据。rev3 的**实际**验收标准比本表更严 —— 见 §9.2.2 Tier 2：
> split 路径的 5 个输出必须与现行 kernel **逐位相同**（0 ulp 偏差）。
> 上表的容差只对 **Tier 1（vs golden）** 与 **现行路径** 生效。
> 理由：约束 2 是用户硬约束；且 Stage 3 round1 [#3] 已论证 ulp 级偏差会经
> `s → q2 → tau_p 前缀和边界 → tau → {valid, tok_out}`（两个 **int32 精确匹配**门禁）放大为离散翻转。

**特殊浮点值的位置契约**（rev2 §9.3 原文，rev3 沿用）：本算子输入**允许含 `-inf`**（上游屏蔽产生），
**不允许含 NaN**（上游 `sanitize_nan_logits` 已清洗，kernel 不做二次防护 ⟹ NaN 属 L2 负向域）。
因此 Stage 2 扩展 Boundary 套件时必须遵守：
- 测试输入使用**「有限值 + 稀疏 `-inf`」混合的确定性数据**，保证至少存在一个 `-inf` 和一个有限值；
  **禁止**用全 `-inf` 行作为唯一特殊值用例（会掩盖归并/索引错误）；
- 比较时先分别严格检查 `isnan`、正 `Inf`、负 `Inf` 的 mask 完全一致，
  再**仅对双方均为有限值的位置**应用上表的混合容差；
- `allclose(equal_nan=True)` **不能**替代显式 mask 检查；
- `w_out` 中被屏蔽位置的 golden 值恒为 `0.0`（有限），而 `-inf` 只出现在 `tau_out`
  （当 `k_i ∉ [1,K_MAX]` 且无 top-p/min-p 绑定时）⟹ `tau_out` 的 `-inf` mask 必须精确一致。

> ⭐ **rev3 追加（split-V 特有的特殊值注意）**：
> - **`-inf` 在 split 路径下会出现在 `pv` 里**（`run_v` 的 `-inf` 初值，以及全被屏蔽块的 `-inf`）。
>   §14.3 / §14.5 论证了这些 `-inf` 的合并语义与现行一致，但 **Boundary 套件必须包含
>   「某 split 的全部元素都是 `-inf`」的用例**（例如把 `logits` 的前 `NBS*BV` 个元素全置 `-inf`），
>   以验证 K1B 的跨 split 归并不会因 `-inf` 而产生 `-inf − (-inf) = NaN` 之类的意外。
>   ⚠️ 本设计**先验认为该情形安全**（`topk` 只做比较与搬运，`-inf` 之间可比且有序；
>   `q2 = exp(run_v − m)/s` 中若 `run_v = -inf` 且 `m` 有限 ⟹ `exp(-inf) = 0`，安全），
>   但**未实测** ⟹ 标记为 §10.2 的 Stage 2 验证项 V-S4。
> - **`+inf` 输入**：`-inf` 与 `+inf` 在 `max`/`sort` 中都是良序的（§14.3），**但 `exp(+inf − m) = +inf`
>   会使 `s = +inf`，进而 `q2 = +inf/+inf = NaN`**。这在**现行 kernel 上同样发生** ⟹ 不是 rev3 引入的
>   回归，属既有 L2 负向域。rev3 不改变该行为。

**逐输出的精度分析（为什么上表阈值足够；rev3 更新「误差来源」列）**

| 输出 | 误差来源 | rev3 变化 | 量级估计 | 是否被 fp32 阈值覆盖 |
|------|---------|-----------|---------|---------------------|
| `m_out` | 纯比较，无算术 | **无变化**（softmax 扫描未切分，`m_run` 由同一段代码在同一块序下产生） | **0**（逐位相同） | ✅ 远超要求；且 **Tier-2 逐位相同** |
| `s_out` | `n_blk` 块 × `BV` 项 fp32 分块累加 + 每块重标定 `s*exp(m_old-m_new)` | **无变化**（同上；K1B 的 softmax 代码与现行逐条相同、块序相同） | 相对误差 ~1e-6 | ✅ `rtol=9.77e-4` 有 ~1000× 余量；且 **Tier-2 逐位相同** |
| `tau_out` | 排序/归并**保值**（无算术）；`tau_p` 亦为 `run_v[l]` 原值；`tau_u = m + ln μ` 分支有 ~1e-7 相对舍入 | ⭐ **`run_v` 的产生方式改变**（跨 split 归并 vs 逐块归并），但由 **M2（§14.3）** 保证 `run_v` 逐位相同 ⟹ `tau_k`/`tau_p` 逐位相同；`tau_u` 分支**完全不变** | **0**（除 `tau_u` 分支的既有 ~1e-7） | ✅；且 **Tier-2 逐位相同** |
| `valid` | 整数判定（`okp`/`tieok`/`vflag`） | **无变化**（全部派生自 `run_v`/`m_run`/`s_run`，三者已论证逐位相同） | 0 或 1（离散） | ✅ int32 精确匹配；且 **Tier-2 逐位相同** |
| `w_out` | ① `exp(z-m)` 舍入；② `/s` 舍入；③ `s` 的误差传递；④ `tau_u` 域变换差异（log 域 vs prob 域） | **无变化**（K2m 不切分；`m/s/tau` 逐位相同 ⟹ K2m 输入逐位相同 ⟹ 输出逐位相同） | 主体 ~1e-6 相对；边界元素为离散跳变 | ✅；且 **Tier-2 逐位相同** |
| `tok_out` | Gumbel-max argmax + 严格 GT 跨块折叠 | ⭐ **折叠层级从「块」变为「split→块」两层**，但由 **M3（§14.5）** 保证结果逐位相同 | 0（离散 index） | ✅ int32 精确匹配；且 **Tier-2 逐位相同** |

### 9.4 性能可行性哨兵（强制执行，不可因 large/L1 跳过）

rev2 §9.4 的两个哨兵**保留不变**，rev3 **新增 2 个**（覆盖 split 路径的最坏情形）：

| 用例名 | Shape | dtype/属性 | 覆盖路径 | 单 case timeout | 选择理由 |
|--------|-------|-----------|---------|-----------------|----------|
| `perf_worst_k1k2`（rev2 保留） | `B=64, V=152064` | fp32；`K_MAX=256, BV=4096`；`k=40,p=0.95,μ=0.1,NEED_MIN_P=True`；`LOGIT_SCALE=8.0` | **现行路径**（`b_num=32 ≥ 20 ⟹ S=1`）：K1（37 满块+1 尾块）+ K2 + host `multinomial`；最大 DMA 任务数 + 最大 tmp arena | **900 s** | 用户关键 case：接近满核 + 生产 V + 三阈值全绑定。⭐ **rev3 追加意义**：这是「现行路径必须零回退」的最重哨兵（约束 3） |
| `perf_max_batch`（rev2 保留） | `B=256, V=152064` | 同上 | **现行路径**（`b_num=128`）：最大 GM 流量（K1+K2 ≈ 594 MB）与超订调度 | **1200 s** | 最大任务数 case |
| ⭐ `perf_worst_split`（rev3 新增） | `B=4, V=152064` | fp32；`K_MAX=256, BV=4096`；`k=40,p=0.95,μ=0.05,NEED_MIN_P=True`；`LOGIT_SCALE=8.0` | **split 路径最坏情形**：`S=10`（最大 S）、`grid=20`（恰好满 1 波）、`n_blk_full=37`（最多块）、含对齐尾块、**4 次 kernel launch**（fused 任务：K1A→K1B→K2gA→K2gB）、workspace 40.3 KB（最大） | **900 s** | ⭐ 这是 rev3 收益最大也风险最集中的 case（cann-bench case7）。**必须实测并记录 4 个 kernel 各自的 `elapsed_us`**（用 `perf_tuning/round2_item4/k1_split.py` 的 msprof 位置拆分法），以校准 §12.2 的 `a_launch` 与验证 §12.4 的 3.5×–4.2× 预测 |
| ⭐ `perf_split_multilaunch_small`（rev3 新增） | `B=16, V=32768` | fp32；同上属性 | **split 路径 + 中等规模**：`S=2`、`grid=16`、`n_blk_full=8`、4 次 launch，但总时长仅 ~80–110 us ⟹ **launch 开销占比最高的 split case** | **600 s** | ⭐ 这是 `a_launch` 敏感性的**决定性测量点**（对应 cann-bench case12）。若此 case 实测回退 >3%，说明 `a_launch` 远大于模型假设 ⟹ 必须调高 `a_launch_est` 并收紧 `D-SPLIT-GUARD`（§13.4 步骤 4） |

**执行约束**（`design-template.md §9.4`，rev2 原文沿用）：
- 测试数据、随机数、特殊值和 golden 的排序/掩码/scatter 重排**均在 CPU 完成**；
  运行阶段只做 **H2D → TileLang kernel → D2H**。
- **例外**：`torch.multinomial(w_out, 1)` 在 NPU 上执行（V01-a 用户明确保留的下游算子，rev2 §5.6 DEV-1）。
  L0/哨兵的**精度判定不依赖它**。
- 哨兵用例**必须实际执行并计时**，不得以「large/L1」为由跳过；超时即视为失败。
- 首次运行的 JIT 编译耗时须单独记录并纳入 warmup 预算。
- ⭐ **rev3 追加**：4 个哨兵都必须**分别记录 split 路径与现行路径的墙钟时间**（对
  `perf_worst_split` / `perf_split_multilaunch_small` 还需白盒强制走现行路径做对照），
  以直接测量 §12.4 表格中的预测值。**Stage 3 不得只报 geomean 而不报逐 kernel 拆分。**

### 9.5 ⭐ 环境调用配方（rev3 强制写入；已踩坑两次）

**不要 `source set_env.sh` 后从 repo 根跑** —— 它把 repo 根塞进 `PYTHONPATH`，
未编译源码 `tilelang/`（需不存在的 `tvm`）遮蔽已安装 wheel → `ModuleNotFoundError: No module named 'tvm'`。

```bash
cd /mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4   # cwd 必须在 repo 根之外
export PYTHONPATH=/home/developer/Ascend/cann-9.2.0/python/site-packages:/home/developer/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe:
export ACL_OP_INIT_MODE=1
export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:$LD_LIBRARY_PATH   # 否则 msprof_analysis 静默把 elapsed_us 变 0
```

| 项 | 值（本 Stage 1 已实测确认） |
|---|---|
| wheel | `tilelang 0.1.4+d75300ed0a88b0f4b2eef8a62b43f1930fdae44b` @ `/home/developer/.local/lib/python3.12/site-packages/tilelang` |
| 硬件 | 910b2（`Ascend910_9362`，2 chips，**`cube_core_num = 20`，`vector_core_num = 40`**） |
| 软件栈 | CANN 9.2.0 / torch_npu 2.7.1.post4 / Python 3.12.9 |
| A3 判定 | `"910_93" in soc_name.upper()`（`tilelang/utils/target.py:145-147`）；bisheng arch 固定 `dav-2201`（`tilelang/jit/adapter/libgen.py:152-155`） |

⚠️ **rev3 追加：`CUBE_CORE_NUM` 必须运行时探测**。`detect_vec_core_num()`（`r4_fused_sampling.py:70-83`）
在 910b2 返回 **40** ⟹ `CUBE_CORE_NUM = 40 // VEC_NUM = 20`。**禁止硬编码 20**
（AGENTS.md 核心原则；且换设备会静默改变最优 `S`）。
⚠️ **仅编译（不上板）的探针配方**：`get_kernel_source()` **不需要 NPU 设备**（本 Stage 1 的
§4.5/§4.9 全部实测都是在无设备环境下用 `get_kernel_source()` + 正则解析完成的）。
Stage 2 可先用它做 UB / `padCount` / 静态边界的**编译期门禁**，再上板跑精度。

---

## 10. 风险点与注意事项

### 10.1 已知约束与风险

rev2 §10.1 的全部风险项（R-B / R-D / R-NPU / R-KMAX / R-P / R-JIT / R-E 等）在 rev3 中继续有效
（完整表见 `history_version/design_rev2.md` §10.1）。rev3 **新增 9 项**：

| 编号 | 风险 | 严重度 | 触发条件 | 缓解 / 检测 |
|------|------|--------|---------|------------|
| **R-BW** | ⭐ **`logits` 的 GM 读从 1 趟增至 2 趟**（K1A + K1B 各读一遍全 `V`） | 中 | 所有 split-path case | **已量化**：case7 每行 608 KB → 1216 KB + 20 KB workspace = **2.03×**。但 K1 是 **compute-bound**：case7 每核实测带宽仅 **1.45 GB/s**（608 KB / 419 us），t_hw 模型结论「VECTOR-BOUND by 3.84x」（`compute_r4_fused_sampling.py:105-110`）⟹ 带宽翻倍**不会**成为新瓶颈。**检测**：§9.4 `perf_worst_split` 哨兵实测；若 K1A+K1B 之和 > 现行 K1 的 1.2×，说明带宽假设错 ⟹ 回退到「只切 K2g，不切 K1」 |
| **R-LAUNCH** | ⭐ **`a_launch`（每次 kernel launch 的固定开销）未实测**，是小 case 收益的**唯一未知量** | **高** | split 路径的所有 case，尤其 `n_blk_total` 小者 | **三重缓解**：① `D-SPLIT-GUARD` 用**保守默认 `a_launch_est = 8.0 us`** + `SAFETY = 1.5`，已把 case 5/6/11/13 挡在现行路径（§13.4）；② §12.4 给出 `a_launch ∈ {0, 1.5, 3, 4.5, 6, 8}` 的**完整敏感性表**，证明**在全部取值下 18 case 的最小 speedup 恒为 1.000×**（12 个非 split case 代码逐字节不变 ⟹ 构造性零回退）；③ §9.4 新增 `perf_split_multilaunch_small` 哨兵作为**决定性测量点**，Stage 2/3 实测后更新 `a_launch_est` 常量即可（无需改结构）。⚠️ **不得**在未实测 `a_launch` 前放宽 guard |
| **R-F128** | ⭐ **`F(K_MAX=128)` 固定成本无法由现有数据分离** | 中 | case8 / case13（`K_MAX=128, BV=2048`） | **如实记录矛盾**：case8 的 `nosort=21.10 = F128 + 8·p2048` 与 case13 的 `k1=11.84 = F128 + p2048 + t2048 + tail` 联立 ⟹ `p2048 ≈ 2.09 us > p4096 = 1.80 us`，**与「`p` 应随 `BV` 单调」矛盾**。可能原因：case13 的 msprof K1/K2 位置拆分误差、或 `p(BV)` 有较大 `BV` 无关的固定分量（每块 6 个标量 op + barrier）。**处置**：§12 的模型对 `K_MAX=128` 取 **`F128 = 6.0 us`（乐观值）**，因此对 case8 的预测**偏乐观**；case8 已被 guard 接受（save 27.7 > 17.4）⟹ **Stage 2 必须优先实测 case8**，若回退 >3% 则把 `K_MAX=128` 特化整体排除出 split 路径（只需在 guard 里加 `and K_MAX == 256`，一行改动） |
| **R-WS** | ⭐ GM workspace 显存 | **低** | split 路径 | 最大 **41280 B ≈ 40.3 KB**（case7，§4.8），占同 case `logits` 的 **1.7%**。上界公式 `4·B·S·(K_MAX+2) ≤ 80·B·(K_MAX+2)`。**且 `B ≥ 21 ⟹ S=1 ⟹ workspace = 0`** ⟹ 大 B 场景零额外显存 |
| **R-SGUARD** | ⭐ **若 `S` 选错导致 `b_num ≥ 20`（或 `[11,19]`）反而变慢** | 中 | 分派判据被误改 | **结构性防护（三重）**：① `S_cap = CUBE_CORE_NUM // b_num` 是**整数除**，`b_num ≥ 11 ⟹ S_cap ≤ 1 ⟹ S = 1 ⟹ 走现行路径**（定理 P2）；② §9.2.3 **C 组 3 个用例**把边界（`b_num=10 / 11 / 20`）钉死为断言，误改判据会立刻 L0 FAIL；③ §15.3 的 `Block Num` sanity gate **有意识地更新**为「按路径分别断言」，静默失效会被门禁捕获 |
| **R-JIT2** | ⭐ **JIT 特化键新增 `S`/`NBS`/`S_TAIL`** | 低 | 编译时间 / 缓存 | 三者由 `(B, V, BV, CUBE_CORE_NUM)` **函数依赖**确定 ⟹ **同一设备上不增加独立特化维度**（一个 `(B,V,K_MAX,BV,NEED_MIN_P)` 只对应一个 `S`）。⚠️ 但**跨设备**（`CUBE_CORE_NUM` 不同）会导致缓存不可复用 ⟹ 首次运行多一次编译。模块内 `@tilelang.jit` 从 3 个增至 7 个 ⟹ 冷启动最坏编译次数增加（但每个 shape 只会用到其中 3~4 个） |
| **R-MEM** | ⭐ **`tl.ascend_memory_planning` 若被关闭 ⟹ UB 溢出静默** | **高**（若发生） | 任何新 kernel 漏传 `pass_configs` | `ascend_memory_planning.cc:598` 的 `check_overflow = false`（**已读源码核实**）⟹ 兄弟路径静默溢出。**缓解**：① §3.3 伪代码中 4 个新 kernel 全部显式写 `pass_configs=R4_PASS_CONFIGS`；② §9.2.5 **静态断言 ①**（Stage 2 必须断言 4 个新 kernel 的 `pass_configs == R4_PASS_CONFIGS`）；③ §7.3 明文规定「② 绝对不可关闭」 |
| **R-DEBUG** | ⭐ **kernel 链从 2 增至 4 ⟹ 故障定位难度上升** | 中 | Stage 2 调试期 | **缓解**：① §9.2.3 **E1 用例把「分解正确性」与「S>1 并行正确性」解耦**（先跑 E1，再跑 A 组）；② §10.3 给出**逐 kernel 中间量验证顺序**（先 D2H 检查 `pv`，再查 K1B 输出）；③ §8.6 建议调试期临时用 `torch.zeros` 分配 workspace 以区分「漏写」与「算错」 |
| **R-NAN** | ⭐ **NaN 输入会破坏 M2/M3 的前提**（比较全为 False ⟹ 全序失效 ⟹ 归并顺序影响结果） | 低（域外） | 输入含 NaN | NaN 属**契约外**（rev2 §9.3：上游 `sanitize_nan_logits` 已清洗，NaN 属 **L2 负向域**）。**rev3 不新增 NaN 防护，也不声称支持 NaN**。§14.6 明确记录该前提及其失效方式。⚠️ **但注意**：现行 kernel 在 NaN 下的行为也是未定义的（`topk` 的 Sort32/MrgSort 对 NaN 无全序）⟹ **rev3 不引入新的 NaN 回归**，只是同样不支持 |

### 10.2 Stage 2 上板验证项（rev3 新增；按优先级，实现前必跑）

rev2 §10.2 的 V-1..V-6 **已全部执行完毕**（结果随各行标注于 `history_version/design_rev2.md`）。
rev3 新增 8 项，**全部标记为「待 Stage 2 验证」**——本 Stage 1 **不得**把它们写成已验证结论：

| 编号 | 验证项 | 方法 | 判据 | 先验置信度 |
|------|--------|------|------|-----------|
| **V-S1** | ⭐⭐ **双路 bitwise 等价（约束 2 的核心）** | §9.2.3 **D1** 用例：同一输入分别走 split 与现行路径，5 个输出 `torch.equal` 于位模式 | **零元素不等**（含 `w_out` 的 `B*V` 全部元素） | **高**（§14 有逐 pass 先验论证），但**必须实测**：`topk` 对相同多重集是否保证完全相同的有序序列，只有硬件能回答（§16.3） |
| **V-S2** | ⭐⭐ **`S=1` 退化等价（分解正确性）** | §9.2.3 **E1** 用例：白盒强制 `S=1` 调 K1A/K1B，与现行 kernel 逐位对照 | 零元素不等 | **很高**（`S=1` 时 K1A 做全部 topk、K1B 做全部 softmax+1 次归并，运算序列与现行几乎逐条对应） |
| **V-S3** | ⭐ **`a_launch` 实测** | §9.4 `perf_split_multilaunch_small`（`B=16,V=32768`）+ `perf_worst_split`（`B=4,V=152064`）；用 `perf_tuning/round2_item4/k1_split.py` 的 msprof **位置拆分法**分别取 K1A/K1B/K2gA/K2gB 的 `Duration(us)` 与 `Block Num` | 反解出 `a_launch`，回填 §12.2 的模型常量表；若 `a_launch > 8 us` ⟹ **收紧** guard；若 `< 2 us` ⟹ 可考虑放宽 guard 以纳入 case 6/11 | **未知**（本 Stage 1 无 NPU 运行时测量能力，只有编译能力） |
| **V-S4** | ⭐ **「某 split 全为 `-inf`」的 Boundary 行为** | 构造 `logits`：把某个 split 覆盖的整个 `V` 区间置 `-inf`，其余为有限值；跑 split 路径与现行路径对照 | Tier-1 通过 + Tier-2 逐位相同 | **中高**（§9.3 已先验论证 `topk` 只比较搬运、`exp(-inf)=0`，但未实测） |
| **V-S5** | ⭐ **UB 投毒层施加到 4 个新 kernel** | 现有 `_make_ub_poisoner`（`test_r4_fused_sampling.py:1327-1348`，投毒 `1e30`）+ split 路径的 `STAB_CASES`（至少 A2/B5/F1）× 20 reps | `tau_out` 无 `1e30` 量级垃圾值；20 次重复 launch **逐位相同**；`_tau_invariant` 无违反 | **高**（§4.9 已编译实证 `padCount==0`），但这是缺陷 A 的**唯一运行时检测手段**，**不可跳过** |
| **V-S6** | ⭐ **split 路径的 tail-adversary 哨兵** | 现有 `_run_tail_adversary`（`ADV_SENTINEL=1.0e4` 扫遍每个尾块 index），但用 **split 路径的 shape**（B1–B5 的 `V`） | `tau`（`k=1`）必须等于哨兵值 ⟹ 证明尾块元素未被静默丢弃、且被**正确的 split** 处理 | **高**（`S_TAIL` 是编译期常量，§6.4） |
| **V-S7** | **`grid = b_num*S` 的实际调度行为** | msprof 读 `Block Num` 字段，断言 K1A/K2gA 的 `Block Num == b_num*S`、K1B/K2gB 的 `Block Num == b_num` | 逐 case 精确相等 | **很高**（生成码地址算式已核对，§3.2）。⚠️ 这也是 §15.3 `Block Num` sanity gate 更新的基础 |
| **V-S8** | **`K_MAX=128 / BV=2048` 特化的 split 收益** | 单独实测 case8（`B=4,V=16384,K=128,BV=2048,S=8`）的 K1A/K1B 拆分时间 | 若 K1A+K1B > 现行 K1 × 1.03 ⟹ **把 `K_MAX=128` 排除出 split 路径**（guard 加一行 `and K_MAX == 256`） | **中**（R-F128：`F128` 无法先验确定，§12 对 case8 的预测偏乐观） |

### 10.3 常见错误与已知偏差（rev3 新增 6 条）

rev2 §10.3 的全部条目继续有效。rev3 新增：

| # | 错误 / 偏差 | 症状 | 正确做法 |
|---|------------|------|---------|
| **E-S1** | `alloc_shared` 的声明与使用不在同一词法 `if` 作用域 | 编译期 `error: Undefined variable: z_ub`（或 `ztp`）。**本 Stage 1 实测踩到两次** | `z_ub` 声明在 `if n_blk_full > 0:` 内**且**其 `for` 循环也在同一 `if` 内；`ztp/ztt` 声明在 `if TAIL > 0:` 内**且**尾块代码也在同一 `if` 内（N6）。这正是现行 kernel 的结构 |
| **E-S2** | K1B 保留了 `l_dst`/`l_val` 声明（从现行 K1 复制粘贴时未删） | 编译期 `Cannot find pre-allocated address for buffer`（`codegen_ascend.cc:828`），因为 K1B 不做 per-block topk ⟹ 二者是死缓冲 | **必须删除**（N7）。同理 K1A **必须不声明** `e_ub`/`m2d`/`zt`/`et`/`mt` 及全部 epilogue buffer |
| **E-S3** | ⭐ K2gA 的 hash counter / index 偏移误用局部 `lb` 而非全局 `b` | `tok_out` 全错（每 split 重复同一段 Gumbel 噪声，且 index 落在 `[0, NBS*BV)` 而非 `[0, V)`）。**精度会明显失败，不会静默** | `arith_progression(state_ub, seedP_v[0] + b*BV, 1, BV2)` 与 `best_idx[0] = blk_argmax[0] + b*BV`，其中 `b = sid*NBS + lb`（N11） |
| **E-S4** | ⭐ K1B 的跨 split 归并用了 `T.Parallel(S)` 或树形归并 | **Tier-2 bitwise FAIL**（归并顺序不定 ⟹ `run_v` 可能不同）；Tier-1 可能仍 PASS ⟹ **容易被误判为「精度够了」** | **必须** `for s in T.serial(S)` 严格升序（约束 2）。⚠️ 这正是 §9.2.2 设 Tier-2 的原因：**Tier-1 容差判定检测不出归约顺序错误** |
| **E-S5** | ⭐ K1A 把 `pv` 声明为 `out_idx` 输出 | 每次调用都新分配 ⟹ K1B 拿到的是另一个张量（或 host 无法把它传给 K1B） | `pv` 必须是**普通入参**，由 host `torch.empty` 分配后传入（§4.7） |
| **E-S6** | ⭐ 漏掉 `S` 算法的步骤 4（回代收缩） | 特定 shape 下出现**空 split** ⟹ K1B 读到未写入的 `pv` 行（`torch.empty` 的垃圾）⟹ `tau_out` 出现垃圾大值（**症状与 rev2 的缺陷 A 完全一样，极易误诊**） | 步骤 4 `S = ceildiv(n_blk_total, NBS)` 必须存在；**F1 用例是它的回归哨兵**（§9.2.3）。⚠️ 调试时若见到「`tau_out` 间歇性垃圾大值」，**先查空 split，再查 `padCount`** |

**调试顺序建议（R-DDEBUG 缓解）**：
```
1. 先跑 E1（S=1 白盒退化）      -> 若失败：分解本身错（K1A/K1B 职责划分、pv 往返）
2. 再跑 A3（NBS=1，每 split 1 块）-> 若失败：跨 split 归并错（顺序 / run_v 初值 / pv 索引）
3. 再跑 A2（S=10，最大并行）      -> 若失败：大 S 下的负载不均 / grid 映射 / S_TAIL
4. 再跑 B1-B5（尾块谱）           -> 若失败：S_TAIL 归属 / TAIL_PAD / padCount
5. 再跑 F1（收缩）                -> 若失败：S 算法步骤 4
6. 最后跑 D1（双路 bitwise 全谱） -> 终极判据
每一步都用 T.dump_tensor / D2H 检查中间量：先 pv（K1A 输出），再 m_out/s_out（K1B softmax 侧），
再 run_v（K1B 归并后），再 tau_out（epilogue）。不要只看最终输出。
```

### 10.4 对草稿 / 前版结论的修正清单（rev3）

| # | 被修正的结论 | 出处 | rev3 的修正 | 证据 |
|---|-------------|------|------------|------|
| 1 | 「online-softmax 跨 split 合并满足结合律: `m=max(m_c); s=Σ s_c·exp(m_c−m)`」⟹（隐含）可直接用于 split-V | Stage 3 round1 [#3]，`perf_tuning/optimization_log.md` 迭代 3 | ⚠️ **该式数学成立但 fp32 下 bitwise 不成立**，**rev3 不采用它**（改为不切分 softmax 扫描）。Orchestrator §3.1 的裁定正确 | §16.1（独立复核，含 ulp 偏差传播链） |
| 2 | 「split-V 假设的『核心利用率』瓶颈经实测为**次要**，仅集中于极小 B + 超大 V 的 case（case7）」 | Stage 3 round1 [#3]「实测复核」段 | ⚠️ **部分修正**：round1 的实测是在 [#1]+[#2] 已消除 epilogue tie_m 与 K2g topk(K=1) **之后**做的，那时 Sort 已成为 K1 的 78.5%（case7）/66.6%（case16）/55.3%（case3）⟹ **「核心利用率」瓶颈在 [#1][#2] 之后重新成为主要矛盾**。rev3 的量化动机见 §12.2 | `probe_perf.json`（round2 实测，晚于 round1）；`insn_census.txt`（Sort32 repeats 144 / MrgSort 48 每块） |
| 3 | 「ROI 估计：9/18 case（B<40 且 n_blk≥2：case3/4/5/6/7/8/11/12/17）受益；case7 K1 ~4.7×、case17 ~3.5×」 | Stage 3 round1 [#3]「ROI 估计」段 | ⚠️ **修正为 6/18**（{3,4,7,8,12,17}）：case5/6/11 被 `D-SPLIT-GUARD` 否决（`n_blk_full < 2` 或 save < 1.5×cost）。**case7 K1 修正为 ~2.8×–3.1×**（非 4.7×，因 softmax 侧 21.5% 不可并行 + 尾块 + 新增归并），**case17 修正为 ~1.6×**（非 3.5×，因 `b_num=8 ⟹ S_cap=2`，只能 2 路） | §12.4（模型 + 敏感性表）；§5.8.3 定理 P2 |
| 4 | rev2 §4.5 的 UB 分析模型 | `history_version/design_rev2.md` §4.5 | ⚠️ **废弃**（4 处错误 + 无 K2g 账目，round2 已证）。rev3 §4.5 全部改为**实测值** | §4.5（48 次编译实测） |
| 5 | Orchestrator §3.2「workspace 显存 `B×n_blk×4B×2`（case15 ≈ 18.9 KB）」 | 调度指令 §3.2 代价 ③ | ⚠️ **公式不适用**：那是 4-pass 分解存 `m_b`/`s_b` 的公式。rev3 不切 softmax ⟹ 不需要 `m_b`/`s_b` workspace，公式是 `4·B·S·(K_MAX+2)`；且 case15 走现行路径**不分配** workspace，实际最大是 case7 的 **40.3 KB** | §4.8（含 18-case 全表） |
| 6 | Orchestrator §3.2「kernel launch 2 次→3+ 次」 | 调度指令 §3.2 代价 ② | ✅ **量级正确，rev3 精确化**：split 路径下 mask 任务 **2→3 次**（K1A/K1B/K2m），fused 任务 **2→4 次**（K1A/K1B/K2gA/K2gB）。**代价大小取决于未实测的 `a_launch`** ⟹ 列为 R-LAUNCH（高严重度）+ V-S3（决定性测量点）+ `D-SPLIT-GUARD`（保守默认 8 us） | §12.4 敏感性表；§9.4 哨兵 |
| 7 | Orchestrator §3.2「该方案完全绕开『Developer 模式无跨核归约原语』的障碍」 | 调度指令 §3.2「额外收益」 | ✅ **结论正确，但理由需修正**：4-pass 分解之所以「绕开」是因为 Pass2/Pass4 只能作**独立 kernel**（选项 A，+2 launch）——它并没有真的绕开，而是**付出了 launch 代价**。rev3 的选项 D 是**真正绕开**：串行部分与跨 split 归并合并进同一个 `grid=b_num` 的 kernel，**串行标量重放根本不需要存在** | §2.5、§16.2 |

### 10.5 Developer 模式可行性最终判定（供 Orchestrator 决策）

**判定：✅ 在 Developer 模式内完全可行，无需转 Expert / 混合模式。**

| 判据 | 结论 | 证据 |
|------|------|------|
| 是否需要 Expert 专属 buffer（`alloc_L1/ub/L0A/L0B/L0C`）？ | **否** | rev3 全部用 `T.alloc_shared`（§3.3 伪代码），4 个原型 kernel 已编译通过 |
| 是否需要手写作用域（`T.Scope("C"/"V")`）？ | **否** | 纯 Vector，无 CV 边界 |
| 是否需要手写同步（`barrier_all`/`set_flag`/`wait_flag`）？ | **否** | §2.5 论证 + §7.2：跨 split 顺序由 **host 顺序 launch** 保证，kernel 内零跨核依赖 |
| 是否需要 grid barrier / semaphore / atomic？ | **否**（且前两者不可用、atomic 违反 bitwise） | §2.5 源码核实表 |
| 是否需要动态循环边界？ | **否** | `S`/`NBS`/`n_blk_full`/`n_blk_total`/`S_TAIL` 全是 host 侧 Python int（§3.5.1），生成码已实证全静态 |
| 是否需要三维 Kernel？ | **否** | 一维 `grid = b_num*S` + `//`、`%` 映射（`unsorted_segment_sum.py:66-68` 先例） |
| pass_configs 是否需要调整？ | **否** | 4 个新 kernel 全部沿用 `R4_PASS_CONFIGS`（§4.7），且 `tl.ascend_memory_planning=True` 是**强制**的（R-MEM） |
| **编译可行性** | ✅ **已实证** | 4 个原型 kernel × 12 case × 2 特化 = **48 次 `get_kernel_source()` 全部成功**，无 `LOG(FATAL)`、无 `ValueError`（§4.5、§13.6） |

**⟹ 不返回 `[DESIGN_ERROR]`。** 约束 1（继续 Developer 模式）满足。

---

## 11. 交付清单

### 11.1 目录结构

```
examples/R4/
├── DESIGN.md                      # ⭐ 本文件（rev3，覆盖写入；rev2 已备份）
├── proto.yaml                     # 契约，rev3 不变（sha256 41ab7970…）
├── r4_fused_sampling.py           # ⭐ Stage 2 修改：新增 4 个 kernel + host 分派；
│                                  #    现行 3 个 kernel 逐字节保留（约束 3）
├── r4_ref.py                      # golden，rev3 不变
├── test_r4_fused_sampling.py      # ⭐ Stage 2 修改：L0 扩至 15 例（含 Tier-2 bitwise 层）
│                                  #    + STAB/ADV 增加 split 路径配置；现有用例不得删减
├── debug_log.md                   # Stage 2 追加 rev3 调试记录
├── history_version/
│   ├── design_rev1.md             # 只读
│   ├── design_rev2.md             # 只读（rev3 的对照基准）
│   └── r4_fused_sampling_impl_*.py  # 各阶段回滚点
├── bench/
│   └── hap_thw_model/             # ⭐ §15.3：Block Num sanity gate + t_hw 模型需有意识更新
├── perf_tuning/                   # Stage 3 round1/round2 记录（只读参考）
└── cann_bench_submission/         # ⭐ §15.4：sha256 需刷新
```

### 11.2 文件清单

| 文件 | rev3 处置 | 责任阶段 |
|------|----------|---------|
| `DESIGN.md` | **覆盖写入**（本文件） | Stage 1（本次） |
| `proto.yaml` | **不动**（§15.1 论证契约不变） | — |
| `r4_fused_sampling.py` | 新增 `r4_topk_stats_splitA/splitB`、`r4_gumbel_sample_splitA/splitB`、`detect_cube_core_num()`（或由 `detect_vec_core_num()//VEC_NUM` 派生）、host 分派判据函数 `_r4_split_plan(B,V,K_MAX,BV)`；**现行 3 个 kernel 一字不改** | Stage 2 |
| `test_r4_fused_sampling.py` | L0 层扩至 §9.2.3 的 15 例 + Tier-2 bitwise 层 + §9.2.5 的 4 项静态断言；STAB/ADV 增加 split 配置 | Stage 2 |
| `perf_tuning/round2_bv_probe/ub_probe.py` | **复用**（§4.5 实测法）；Stage 2 需扩展其 `probe()` 的 kernel 字典以包含 4 个新 kernel | Stage 2 |
| `perf_tuning/round2_item4/k1_split.py` | **复用**（msprof 位置拆分法）；⚠️ 其 `Block Num == ceil(B/2)` 交叉校验需按 §15.3 更新为「按路径分别断言」 | Stage 2/3 |
| `bench/hap_thw_model/*.py` | §15.3 的有意识更新（`Block Num` gate、`t_hw` 口径、`K_MAX/BV==1/16` 硬编码常数） | Stage 3 |

### 11.3 命名规范

| 对象 | 命名 | 依据 |
|------|------|------|
| 新 kernel | `r4_topk_stats_splitA` / `r4_topk_stats_splitB` / `r4_gumbel_sample_splitA` / `r4_gumbel_sample_splitB` | `snake_case`（AGENTS.md 开发规范）；`splitA/splitB` 后缀明确「split 路径的第 1/2 段」，与现行无前缀名并列不冲突 |
| host 分派函数 | `_r4_split_plan(B, V, K_MAX, BV) -> (use_split, S, NBS, S_TAIL)` | 前导下划线 = 模块私有；返回具名元组便于测试断言（§9.2.5 结构断言） |
| workspace | `pv` / `pv_val` / `pv_idx`（`pv` = **p**artial **v**alues） | 简短、与生成码中的变量名一致，便于 msprof/生成码对照 |
| 测试用例 | `l0_A1_split_s2_aligned` … `l0_F1_split_shrink_noempty` | 组号 + 序号 + 语义，与 §9.2.3 表格一一对应 |

### 11.4 实现顺序（Stage 2）

```
步骤 1  静态门禁（不上板，只用 get_kernel_source）：
        ① 4 个新 kernel 编译通过；② UB ≤ 196352（对齐 §4.5 实测值）；
        ③ 全部 TopK 的 padCount == 0；④ 全部 copy_ub_to_ub 的 count*4 % 32 == 0；
        ⑤ 全部 for 循环边界为字面常量；⑥ pass_configs == R4_PASS_CONFIGS
        —— 任一失败立即停，不进上板（省 NPU 时间）
步骤 2  E1（S=1 白盒退化，bitwise vs 现行）      ← 分解正确性
步骤 3  A3（NBS=1）→ A1（S=2）→ A4（K=128 特化）  ← 跨 split 归并
步骤 4  A2（S=10 最大并行）                       ← grid 映射 / 负载不均
步骤 5  B1–B5（尾块谱）                           ← S_TAIL / TAIL_PAD
步骤 6  F1（收缩）+ C1/C2/C3（分派边界）           ← S 算法 / 分派判据
步骤 7  D1（双路 bitwise 全谱，5 输出 × 2 任务）   ← 约束 2/3 终极判据
步骤 8  现有 `--level all` 全套回归（74 PASS / 38 BOUND 必须保持）
步骤 9  V-S5 UB 投毒 + V-S6 tail-adversary（split 路径）
步骤 10 V-S3 a_launch 实测 → 回填 §12.2 → 复核 D-SPLIT-GUARD
步骤 11 cann-bench 36/36 mismatch_count == 0 复跑 + sha256 刷新（§15.4）
```

### 11.5 运行 / 复现说明（Stage 2 必须遵守）

见 §9.5（环境配方，含 `PYTHONPATH` / `ACL_OP_INIT_MODE` / `LD_LIBRARY_PATH` 与「不要从 repo 根跑」的坑）。

```bash
# L0 精度收敛（rev3 双层判据）
cd /mnt/workspace/cann/20261006_R4_V01/tilelang-ascend/examples/R4
export PYTHONPATH=/home/developer/Ascend/cann-9.2.0/python/site-packages:/home/developer/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe:
export ACL_OP_INIT_MODE=1
export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:$LD_LIBRARY_PATH
python3 test_r4_fused_sampling.py --level l0          # 期望 [PRECISION_PASS] + [BITWISE_PASS]，exit 0
python3 test_r4_fused_sampling.py --level all         # 期望 74+ PASS / 0 FAIL，exit 0
python3 test_r4_fused_sampling.py --level stability   # 期望 12+ 项（含 rev3 新增 split 配置）
python3 test_r4_fused_sampling.py --level stress      # 期望 8/8+（含 UB 投毒施加到新 kernel）

# 编译期静态门禁（无需 NPU；本 Stage 1 已用同法产出 §4.5 / §4.9）
python3 perf_tuning/round2_bv_probe/ub_probe.py "<[(kernel,args,label), ...]>"

# 覆盖门禁（proto.yaml 不变 ⟹ 应覆盖维度不变 ⟹ 25/0/0 必须保持）
python3 ../../.agents/skills/tilelang-op-test-design/scripts/coverage_check.py \
        test_r4_fused_sampling.py --proto proto.yaml
```

### 11.6 算子 proto.yaml

**rev3 不产出新的 `proto.yaml`：现有文件保持原样不动**（sha256
`41ab797002568e89fcc9a9eaa58be4f1cdfce20b6406abd0fdbf2ad11f7f8949`，本 Stage 1 已用 `sha256sum` 核实）。
完整论证见 **§15.1**。

现有 `proto.yaml` 的关键字段（rev3 全部不变）：

| 字段 | 值 | rev3 影响 |
|------|-----|----------|
| `operator.name` | `r4_fused_sampling` | 不变 |
| `operator.category` | `Fusion` | 不变 |
| `operator.formula` | K1（定理 T1/T2）+ K2m + K2g（定理 T3 + lowbias32） | **不变**（split-V 是实现路径，非语义） |
| `operator.attrs` | `K_MAX`(256), `BV`(4096), `NEED_MIN_P`(false), `FUSE_GUMBEL`(false), `top_k`(40), `top_p`(0.95), `min_p`(0.0), `need_min_p_sampling`(false) | **不变**（⭐ `S`/`NBS`/`S_TAIL` **不列为 attr**，理由见 §15.1） |
| `operator.inputs` | `logits`[f32], `top_ks`[i32], `top_ps`[f32], `min_ps`[f32], `rand`[i32] | **不变** |
| `operator.outputs` | `m_out`[f32], `s_out`[f32], `tau_out`[f32], `valid`[i32], `w_out`[f32], `tok_out`[i32] | **不变** |
| `operator.schema` | 见文件 `:118-132` | **不变**（`pv`/`pv_val`/`pv_idx` 是**内部 workspace**，不属对外 schema） |

**dtype 全集与 §9.3 精度表一致**（float32 + int32，2 行）⟹ 覆盖门禁 `coverage_check.py --proto`
派生的 `D-DTYPE-float32` / `D-DTYPE-int32` 与 8 个 `D-PARAM-*` 维度**全部不变** ⟹ 现有 **25/0/0** 必须保持。

---

## 12. 性能目标

> 本章 rev2 由 `@tilelang-op-orchestrator` 在 Stage 3 启动前追加。rev3 **保留 §12.1/§12.3 原文**，
> **更新 §12.2（补入 round2 的实测成本模型）**，并**新增 §12.4（rev3 的量化预测）**。

### 12.1 用户确认的调优参数（rev2 原文，rev3 不变）

| 字段 | 取值 |
|------|------|
| 性能目标类型 | `best_effort`（无硬性数值目标；以 roofline 达成率与 geomean speedup 提升为准） |
| 测试 shape | **全 18-case 矩阵**（`r4_fused_sampling` 与 `r4_mask_sample` 各 18 case）；**任何 case 不得回退超过噪声阈值** |
| 噪声阈值 | **3%** |
| 最大迭代数 | 10（中止条件：达到 10 轮 / 连续 3 次无提升） |
| 交付件改动范围 | 允许直接改主文件，优化后重跑精度门禁 + cann-bench 三阶段评测，并同步更新 metadata/submission 的 sha256 |

### 12.2 已知瓶颈（rev3 更新：补入 round2 的**实测**成本模型）

**rev2 原文的 4 条定性结论仍然成立**（grid 只绑 batch、核心利用率实测、roofline 达成率 1–4%、
split-V 是最高杠杆方向）。rev3 补入 round2 的**定量**实测：

**(a) 波数定律**（Orchestrator §2 给出，本 Stage 1 用 `probe_perf.json` + `ceiling.json` 独立复核 ✅）

```
K1_us ≈ ceil(b_num / 20) × [ F(K_MAX) + n_blk_row × ( p(BV) + t(BV,K_MAX) ) ]
```

**本 Stage 1 对 6 个 probe case 的拟合结果**（`n_blk_row = ceil(V/BV)`）：

| case | B | V | K_MAX | BV | b_num | waves | n_blk | 模型 K1 (us) | **实测 K1 (us)** | 误差 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 4096 | 256 | 4096 | 1 | 1 | 1 | 32.3 | **32.52** | **−0.6%** |
| 3 | 8 | 16384 | 256 | 4096 | 4 | 1 | 4 | 64.4 | **64.50** | **−0.1%** |
| 7 | 4 | 152064 | 256 | 4096 | 2 | 1 | 38 | 419.5 | **419.47** | **+0.0%** |
| 8 | 4 | 16384 | 128 | 2048 | 2 | 1 | 8 | 44.8 | **52.74** | **−15.0%** ⚠️ |
| 15 | 64 | 152064 | 256 | 4096 | 32 | 2 | 38 | 839.0 | **835.72** | **+0.4%** |
| 16 | 128 | 32768 | 256 | 4096 | 64 | 4 | 8 | 429.0 | **426.12** | **+0.7%** |

**标定常数**（`K_MAX=256, BV=4096`，由 case3/case7 的 `nosort` 联立解出，并被 case2/15/16 交叉验证）：

| 常数 | 值 | 含义 | 标定来源 | 置信度 |
|------|-----|------|---------|--------|
| `t(4096,256)` | **8.902 us/块** | per-block topk+merge（Sort32 128 reps + MrgSort 43 + 归并 topk 16 reps + MrgSort 5 = 192 单位） | case7: `329.37/37`；case3: `35.66/4 = 8.915` | **高**（2 个独立数据点，差 0.15%） |
| `p(4096)` | **1.802 us/块** | per-block online-softmax（1 GM 读 + reduce_max + broadcast + Sub + Exp + reduce_sum + 6 标量） | case3/case7 的 `nosort` 联立：`34p = 61.26` | **高**（并被 case15/16 交叉验证到 1.3%） |
| `F(256)` | **21.63 us/行** | per-row 固定成本 = launch + prologue + **epilogue**（含 `tau_p` 的 `T.serial(256)` 串行 scan） | case2/3/7/15/16 **5 个独立拟合**，范围 21.4–22.8 | **高** |
| `t_mrg(512,256)` | **0.974 us** | 跨 split 归并的 `topk(2*K_MAX=512, K=256)`（Sort32 16 reps + MrgSort 5 = 21 单位 × `t4096/192`） | 由 `insn_census.txt` 的单位计数 + `t(4096,256)` 推得 | **中**（推算，未直接实测） |
| `t(2048,128)` | **3.955 us/块** | `K_MAX=128, BV=2048` 特化的 per-block topk+merge（96 单位） | case8: `31.64/8` | **高** |
| `t_mrg(256,128)` | **0.453 us** | 同上的归并（11 单位 × `t2048/96`） | 推算 | **中** |
| `F(128)` | ⚠️ **无法确定（4–14 us）** | `K_MAX=128` 的 per-row 固定成本 | case8 `nosort=21.10 = F128 + 8·p2048` 与 case13 `k1=11.84 = F128 + p2048 + t2048 + tail` **联立矛盾**（解出 `p2048 ≈ 2.09 > p4096 = 1.80`，违反单调性） | 🔴 **低 —— R-F128，§12 对 case8 取乐观值 6.0 us** |
| `a_launch` | ⚠️ **未实测** | 每次 kernel launch 的固定开销 | **无数据** | 🔴 **未知 —— R-LAUNCH，§12.4 做 0–8 us 全区间敏感性分析** |

**`F(256) = 21.63 us` 的物理来源（本 Stage 1 新发现，有生成码证据）**：
`perf_tuning/round2_item2/emitted_cc/K1_case15_real.cc:194-202` 显示 `tau_p` 前缀和循环为
```c
for (int32_t l = 0; l < 256; ++l) {
  AscendC::PipeBarrier<PIPE_ALL>();
  if (prefix.GetValue(0) <= p_i.GetValue(0)) { tau_p.SetValue(0, run_v.GetValue(l));
                                               prefix.SetValue(0, prefix.GetValue(0) + q2.GetValue(l)); }
  AscendC::PipeBarrier<PIPE_ALL>();
  AscendC::PipeBarrier<PIPE_ALL>();
}
```
⟹ **3 × `PipeBarrier<PIPE_ALL>` / 迭代 × 256 迭代 = 768 次全流水 drain（每行）**，
外加 5 × `GetValue` / 迭代 = 1280 次 UB→标量寄存器读。`21.63 us / 770 ≈ 28 ns` 每次 drain，量级合理。
**⟹ `F` 主要是 epilogue 而非 launch**，这对 rev3 是**关键利好**：
增加 kernel 数只增加 `a_launch`（小），**不会**每次都再付一遍 21.63 us 的 epilogue
（epilogue 只在 K1B 里跑一次）。

**⚠️ 但 `F` 也是 split-V 无法消除的串行地板**：case2（`n_blk=1`）的 K1 = 32.52 us 中
**66% 是 `F`** ⟹ 对 `n_blk_total` 小的 case，split-V 收益趋近 0（这正是 `D-SPLIT-GUARD` 的依据）。

**(b) K1 占比**（rev2 §12.2 定性 → rev3 用 `ceiling.json` 定量）：mask 任务 **80.1%–94.7%**（18 case 中位 ~92%）、
fused 任务 **51.8%–76.5%**。
**(c) Sort/topk 占 K1**：**26.4% / 55.3% / 60.0% / 66.6% / 78.5% / 78.8%**（case 2/3/8/16/7/15，`probe_perf.json` 实测）。
**(d) 核心饥饿**：18 case 中 **14 个 `b_num < 20`**；最差 case7（B=4）只用 **2/20** 组核心，roofline 达成率 **0.67%**。
**(e) K2g 占比**：fused 任务 **23.4%–48.7%**（中位 37.6%）；mask 任务 0%（K2g 只在 fused 路径）。
**(f) K2m 占比**：mask 任务 **3.1%–10.2%** ⟹ **本轮不切**（约束 4，上限收益 +3.18%）。

### 12.3 精度红线（rev2 原文，rev3 **强化**）

Stage 2 的 `TAIL_PAD = ceil(TAIL/32)*32` + `ztp=[1,TAIL_PAD]`（从 GM 以 `pad_value=-inf` 载入）修复，
是针对两个 TileLang wheel 缺陷的**结构性规避**：
① `tl::ascend::Sort()` 修补余数的 masked `AscendC::Duplicate` 在 dav_c220 上不生效；
② UB→UB 的 `T.copy`（`copy_ub_to_ub` / `AscendC::DataCopy`）在 count 非 32B 倍数时静默截断到 8 的倍数。

因此任何性能优化都**不得**重新引入非 32B 倍数的 UB→UB 拷贝，或 `count % 32 != 0` 的 `T.tile.topk` 调用。

**⭐ rev3 强化**：本设计通过 **D-SPLIT-BVALIGN**（split 边界按块对齐）使
`actual_num` 的取值集合与现行**完全相同** `{BV, 2*K_MAX, TAIL_PAD}`，
并已对 **10/10** split-path case **编译实证** `padCount == 0` 与全部 UB→UB `count*4 % 32 == 0`（§4.9）。
⟹ **红线在 rev3 下是结构性保持的，不依赖人工审查。** Stage 2 仍须执行 §9.2.5 的静态断言 ③④
（把该实证固化为门禁）与 V-S5 的 UB 投毒运行时检测。

### 12.4 ⭐ rev3 的量化预测（**全部为模型估计，非实测**）

**方法**：把 §12.2 的标定常数代入 split-V 的分段模型：

```
T_K1_old = waves(b_num) × [ F + n_blk_full·(p+t) + tailcost ]
T_K1A    = waves(b_num·S) × [ a_launch + maxblk_full·t + (tailcost_t if S_TAIL 拥有尾块) ]
T_K1B    = waves(b_num)   × [ a_launch + F + n_blk_full·p + tailcost_s + S·t_mrg ]
T_K1_new = T_K1A + T_K1B
（K2g 同构，但无 F(K_MAX) epilogue，per-block 成本 8.02 us(BV=4096) / 4.01 us(BV=2048)，
 由 ceiling.json 的 case7 k2_us_fused=304.58/38 与 case13 反推）
任务级：mask = K1 + K2m（K2m 不切）；fused = K1 + K2g
最终取值：用【模型比值】new/old 乘到【实测的 k1_us/k2_us】上（避免模型绝对误差污染结论）
```

**结果（`a_launch = 3.0 us`，18 case 全表；`use_split` 由 §13.4 guard 决定）**

| case | B | V | b_num | S | 分派 | mask old→new (us) | **mask ×** | fused old→new (us) | **fused ×** |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 256 | 1 | 1 | 现行 | 26.1 → 26.1 | **1.000** | 34.9 → 34.9 | **1.000** |
| 2 | 1 | 4096 | 1 | 1 | 现行 | 33.9 → 33.9 | **1.000** | 42.1 → 42.1 | **1.000** |
| **3** | 8 | 16384 | 4 | 4 | **split** | 70.6 → 53.4 | **1.32** | 100.3 → 67.2 | **1.49** |
| **4** | 4 | 16384 | 2 | 4 | **split** | 55.7 → 42.4 | **1.31** | 85.4 → 56.1 | **1.52** |
| 5 | 7 | 4409 | 4 | (2) | 现行(guard) | 40.0 → 40.0 | **1.000** | 55.0 → 55.0 | **1.000** |
| 6 | 8 | 10240 | 4 | (3) | 现行(guard) | 55.4 → 55.4 | **1.000** | 77.6 → 77.6 | **1.000** |
| **7** | **4** | **152064** | **2** | **10** | **split** | **461.7 → 179.7** | ⭐ **2.57** | **730.2 → 187.7** | ⭐ **3.89** |
| **8** | 4 | 16384 | 2 | 8 | **split** | 59.6 → 38.0 | **1.57** | 91.3 → 48.7 | **1.87** |
| 9 | 64 | 4096 | 32 | 1 | 现行 | 69.5 → 69.5 | **1.000** | 86.4 → 86.4 | **1.000** |
| 10 | 4 | 4096 | 2 | 1 | 现行 | 34.1 → 34.1 | **1.000** | 42.1 → 42.1 | **1.000** |
| 11 | 8 | 8192 | 4 | (2) | 现行(guard) | 47.4 → 47.4 | **1.000** | 62.7 → 62.7 | **1.000** |
| **12** | 16 | 32768 | 8 | 2 | **split** | 104.0 → 79.5 | **1.31** | 161.3 → 112.3 | **1.44** |
| 13 | 6 | 2304 | 3 | (2) | 现行(guard) | 14.8 → 14.8 | **1.000** | 23.5 → 23.5 | **1.000** |
| 14 | 32 | 1024 | 16 | 1 | 现行 | 29.3 → 29.3 | **1.000** | 37.4 → 37.4 | **1.000** |
| 15 | 64 | 152064 | 32 | 1 | 现行 | 931.4 → 931.4 | **1.000** | 1469.5 → 1469.5 | **1.000** |
| 16 | 128 | 32768 | 64 | 1 | 现行 | 466.3 → 466.3 | **1.000** | 695.1 → 695.1 | **1.000** |
| **17** | **16** | **152377** | **8** | **2** | **split** | **472.2 → 307.2** | ⭐ **1.54** | **736.2 → 426.9** | ⭐ **1.72** |
| 18 | 64 | 65536 | 32 | 1 | 现行 | 424.1 → 424.1 | **1.000** | 650.3 → 650.3 | **1.000** |
| | | | | | **geomean** | | **1.158×** | | **1.229×** |

**⭐ 对未实测参数 `a_launch` 的敏感性分析（R-LAUNCH 的定量处置）**

| `a_launch` | mask geomean | **mask 最小** | mask 最大 | fused geomean | **fused 最小** | fused 最大 |
|---|---|---|---|---|---|---|
| 0.0 us | 1.196× | **1.000×** | 2.66× | 1.292× | **1.000×** | 4.15× |
| 1.5 us | 1.176× | **1.000×** | 2.61× | 1.260× | **1.000×** | 4.02× |
| **3.0 us** | **1.158×** | **1.000×** | **2.57×** | **1.229×** | **1.000×** | **3.89×** |
| 4.5 us | 1.141× | **1.000×** | 2.53× | 1.202× | **1.000×** | 3.77× |
| 6.0 us | 1.126× | **1.000×** | 2.49× | 1.179× | **1.000×** | 3.66× |
| 8.0 us | 1.107× | **1.000×** | 2.43× | 1.151× | **1.000×** | 3.52× |

**⭐ 关键性质：`min = 1.000×` 在 `a_launch` 的整个取值区间上恒成立。**
这不是巧合，而是**构造性保证**：12 个非 split case 走的是**逐字节不变的现行代码**（约束 3）
⟹ 它们的时间**精确不变**；6 个 split case 由 `D-SPLIT-GUARD` 用**保守的 `a_launch_est = 8.0 us`**
筛选 ⟹ 即使真实 `a_launch` 达到 8 us，它们仍全部 > 1.0×。
**⟹ rev3 满足 §12.1「任何 case 不得回退超过噪声阈值 3%」的要求，且是零回退（不是「回退 <3%」）。**

**split 路径 6 个 case 的单独 geomean**（`a_launch = 3 us`）：mask **1.554×**、fused **1.855×**。

**⚠️ 预测的不确定性（如实标注）**：
1. **全部为模型估计，非实测**。本 Stage 1 只有编译能力（`get_kernel_source()`），**无 NPU 运行时测量能力**。
2. 模型对 `K_MAX=256, BV=4096` 的拟合误差 **≤ 0.7%**（5/6 probe case），可信度高。
3. ⚠️ 模型对 `K_MAX=128, BV=2048`（case8）的拟合误差 **−15.0%**（`F128` 不可分离，R-F128）
   ⟹ **case8 的 1.57×/1.87× 预测偏乐观**，必须 V-S8 优先实测。
4. `t_mrg` 是由 `insn_census.txt` 的 Sort32/MrgSort **单位计数推算**的，非直接实测 ⟹ ±30% 不确定度。
   它对结论影响小（`S·t_mrg` 最大为 `10×0.974 = 9.7 us`，占 case7 K1_new 的 7%）。
5. K2g 的固定成本（模型取 6.0 us）**未标定** ⟹ 对 `n_blk_total` 小的 case 误差可达 +34%~+42%
   （见 §12.2 表外的 K2g 模型 vs `ceiling.json` 对比）。但这些 case **全部走现行路径** ⟹ 不影响结论。

---

## 13. ⭐ split-V 并行架构（rev3 核心新增章节）

### 13.1 架构总览：kernel 数量与职责划分

| # | kernel | grid | 职责 | 并行性 | rev3 状态 |
|---|--------|------|------|--------|----------|
| 1 | `r4_topk_stats`（K1） | `b_num` | softmax + topk + epilogue（全） | 行间 | **保留，逐字节不变**（现行路径 + bitwise 基准） |
| 2 | `r4_mask_sample`（K2m） | `b_num` | 屏蔽后概率 `w_out` | 行间 | **保留，不变**（约束 4：本轮不切） |
| 3 | `r4_gumbel_sample`（K2g） | `b_num` | Gumbel-max argmax → `tok_out` | 行间 | **保留，逐字节不变**（现行路径 + bitwise 基准） |
| 4 | ⭐ `r4_topk_stats_splitA`（K1A） | **`b_num*S`** | **只做**分层 top-k 归并 → `pv[row*S+sid, :]` | **(行, split) 双维** | **新增** |
| 5 | ⭐ `r4_topk_stats_splitB`（K1B） | `b_num` | online-softmax 扫描（**代码逐条不变**）+ 跨 split top-k 归并 + epilogue（**代码逐条不变**）→ 4 个输出 | 行间 | **新增** |
| 6 | ⭐ `r4_gumbel_sample_splitA`（K2gA） | **`b_num*S`** | hash + Gumbel + keep-mask + 块内 argmax + split 内严格 GT 折叠 → `pv_val/pv_idx[row*S+sid]` | **(行, split) 双维** | **新增** |
| 7 | ⭐ `r4_gumbel_sample_splitB`（K2gB） | `b_num` | 跨 split 严格 GT 折叠 → `tok_out` | 行间 | **新增** |

**launch 次数**：现行路径 mask 2 / fused 2；**split 路径 mask 3（K1A,K1B,K2m）/ fused 4（K1A,K1B,K2gA,K2gB）**。

**职责划分的关键**：K1A 只做 **Sort 侧**（可并行、占 K1 的 26.4–78.8%），
K1B 做 **softmax 侧 + 归并 + epilogue**（必须串行、但只占 21.5%）。
这个切法直接来自 §12.2 的实测成本拆分，不是先验选择。

### 13.2 双路分派与「现行 kernel 逐字节保留」的落实

```python
# host wrapper 内（伪代码，完整见 §3.3）
plan = _r4_split_plan(B, V, K_MAX, BV)      # -> (use_split, S, NBS, S_TAIL)；纯 Python int 算术
if not plan.use_split:
    # ===== 现行路径：下面调用的 3 个 kernel 的源码【一字不改】=====
    ...r4_topk_stats(B, V, K_MAX, BV, NEED_MIN_P)...
    ...r4_mask_sample(B, V, BV)... / ...r4_gumbel_sample(B, V, BV)...
else:
    ...r4_topk_stats_splitA / splitB / (r4_mask_sample | r4_gumbel_sample_splitA / splitB)...
```

**「逐字节保留」的可执行含义**（Stage 2 必须满足，§9.2.5 静态断言 ②）：
- `r4_topk_stats`、`r4_mask_sample`、`r4_gumbel_sample` 三个函数的**函数体源码文本不变**
  （含注释、含 round1 [#1]/[#2] 的优化、含 round2 #4 的 `z_ub` rank-1 改造与已删除的 UB→UB hop）。
- 允许的变化**仅限**：① 文件内新增 4 个函数；② host wrapper `r4_fused_sampling` 内部新增分派逻辑
  （其**签名与位置参数顺序不变**，§15.1）；③ 模块级新增 `detect_cube_core_num` / `_r4_split_plan` 辅助函数。
- ⚠️ **rev2 round2 #4 已删除的「L211 的 UB→UB 拷贝」不得复活**（Orchestrator §3.1 特别指出）。
  rev3 的 K1A 沿用 round2 #4 之后的形态：`z_ub` 是 rank-1 `[BV]`，**同时**作为 GM→UB 目的地与 topk 源，
  无中间 hop。**K1B 也不需要该 hop**（它不做 topk）。

**现行 kernel 作为 bitwise 对照基准的用法**：§9.2.2 的 Tier-2 直接 import 并调用它们。
⟹ **它们必须保持可独立调用**（不能变成私有或被合并），这也是「逐字节保留」的一个附带好处。

### 13.3 `S` 选取规则与双路判据（含定理，完整证明见 §5.8.3）

```
S_cap  = CUBE_CORE_NUM // b_num                 # 定理 P2：1-波上限（整数除！）
S      = max(1, min(S_cap, n_blk_total))
NBS    = ceildiv(n_blk_total, S)
S      = ceildiv(n_blk_total, NBS)              # 定理 P3：回代收缩 ⟹ 无空 split
NBS    = ceildiv(n_blk_total, S)
S_TAIL = n_blk_full // NBS if TAIL > 0 else -1  # 尾块所属 split（编译期常量）

use_split = (S >= 2)                            # 几何上可分
        and (n_blk_full >= 2)                   # 至少 2 个满块（否则 Sort 工作量太小）
        and (pred_saving > SAFETY * pred_cost)  # 收益守卫（D-SPLIT-GUARD）
其中：
  maxblk_full  = max over s of min(n_blk_full, max(0, (s+1)*NBS - s*NBS))
  pred_saving  = (n_blk_full - maxblk_full) * t_est(BV, K_MAX)
  pred_cost    = a_launch_est + S * t_mrg_est(K_MAX)
  SAFETY       = 1.5
  a_launch_est = 8.0 us   # ⚠️ 保守默认，V-S3 实测后更新此单一常量即可
  t_est(4096,256) = 8.902 us ; t_est(2048,128) = 3.955 us      # §12.2 实测标定
  t_mrg_est(256)  = 0.974 us ; t_mrg_est(128)  = 0.453 us      # §12.2 推算
```

**双路判据与用户约束 3 的关系**（§0.2 已声明，此处给出机制）：
- `b_num ≥ 20` ⟹ `S_cap = 20 // b_num = 0` ⟹ `S = max(1, min(0, ·)) = 1` ⟹ `use_split = False`
  ⟹ **走现行路径，代码逐字节不变** ✅ **约束 3 满足**
- `b_num ∈ [11, 19]` ⟹ `S_cap = 1` ⟹ `S = 1` ⟹ **走现行路径**（⚠️ 比约束 3 字面更严，理由见 §0.2 + 定理 P2）
- `b_num ≤ 10` ⟹ `S_cap ≥ 2` ⟹ 再由 `n_blk_total ≥ 2`、`n_blk_full ≥ 2`、收益守卫三重过滤

### 13.4 18-case 分派结果（本 Stage 1 实际计算）

见 **§5.8.4** 的完整表格。摘要：

| 分派 | case 数 | case 列表 | 原因 |
|------|--------|-----------|------|
| **split-V** | **6** | **3, 4, 7, 8, 12, 17** | 三重判据全通过 |
| 现行（`b_num ≥ 20`） | 4 | 9, 15, 16, 18 | `S_cap = 0`（约束 3） |
| 现行（`n_blk_total = 1`） | 5 | 1, 2, 10, 14, **（+ 5 见下）** | 只有 1 个块，无法切分 |
| 现行（guard 否决） | 3 | 5, 11, 13 | `n_blk_full = 1 < 2`（case5/13）；save 8.9 us < 1.5×cost（case11） |
| 现行（guard 否决） | 1 | 6 | save 8.9 us < 1.5×16.4 us |

（case5 的 `n_blk_total = 2` 但 `n_blk_full = 1` ⟹ 归入 guard 否决；上表第 3 行的「5」应为 case1/2/10/14 共 4 个，
case5 归 guard 否决。合计 `6 + 4 + 4 + 4 = 18` ✅）

**guard 的逐 case 推导**（`a_launch_est = 8.0`, `SAFETY = 1.5`）：

| case | n_blk_full | S | maxblk_full | `pred_saving` (us) | `1.5 × pred_cost` (us) | 判定 |
|---|---|---|---|---|---|---|
| 3 | 4 | 4 | 1 | (4−1)×8.902 = **26.7** | 1.5×(8+4×0.974) = **17.8** | ✅ split |
| 4 | 4 | 4 | 1 | **26.7** | **17.8** | ✅ split |
| 5 | **1** | (2) | — | — | — | ❌ `n_blk_full < 2` |
| 6 | 2 | (3) | 1 | (2−1)×8.902 = **8.9** | 1.5×(8+3×0.974) = **16.4** | ❌ 8.9 < 16.4 |
| 7 | 37 | 10 | 4 | (37−4)×8.902 = **293.8** | 1.5×(8+10×0.974) = **26.6** | ✅ split |
| 8 | 8 | 8 | 1 | (8−1)×3.955 = **27.7** | 1.5×(8+8×0.453) = **17.4** | ✅ split |
| 11 | 2 | (2) | 1 | **8.9** | 1.5×(8+2×0.974) = **14.9** | ❌ 8.9 < 14.9 |
| 12 | 8 | 2 | 4 | (8−4)×8.902 = **35.6** | **14.9** | ✅ split |
| 13 | **1** | (2) | — | — | — | ❌ `n_blk_full < 2` |
| 17 | 37 | 2 | 19 | (37−19)×8.902 = **160.2** | **14.9** | ✅ split |

### 13.5 K2m 不切的理由（约束 4 的量化确认）

`ceiling.json` 实测：K2m 占 mask 任务的 **3.1%（case7: 36.16/461.7）– 10.2%**，中位 ~5%。
即使 K2m 能被完全消除（不可能，它产出 `w_out[B,V]` 必须写 GM），mask 任务上限收益也只有 **+3.18%**
（Orchestrator §2 的数字，本 Stage 1 用 `ceiling.json` 复核一致）。
而 K2m **本身没有跨块归约**（逐元素独立 ⟹ 天然可 split，无需任何 workspace）——
所以「不切」不是因为难，而是因为**收益 < 噪声阈值 3% 的量级**，且会增加 1 次 launch。
**⟹ 本轮不切 K2m 是正确的成本收益决策。** 若未来 `a_launch` 实测很小（<2 us），
K2m 的 split 是**最低风险的后续增强**（无归约 ⟹ 无 bitwise 风险），可作为独立提案。

### 13.6 ⭐ 编译实证记录（本 Stage 1 实际执行，rev3 可行性的硬证据）

**环境**：wheel `tilelang 0.1.4+d75300ed0a88b0f4b2eef8a62b43f1930fdae44b`；
`PYTHONPATH` / `ACL_OP_INIT_MODE` / `LD_LIBRARY_PATH` 按 §9.5 配方；cwd = `/tmp/opencode`（repo 根之外）。
**方法**：`get_kernel_source()`（**不需要 NPU 设备**）+ 正则解析生成源。
**探针脚本**：`/tmp/opencode/r4_splitv_ub_probe.py`（K1A v1 / K1B / K2gA / K2gB 四原型）、
`/tmp/opencode/refine.py`（K1A 单循环变体，**编译失败**，暴露 N6）、
`/tmp/opencode/refine3.py`（K1A v3 尾块 hoist 变体，**成功**）、
`/tmp/opencode/perf_model.py`（§12.4 的成本模型）。

| 实证项 | 结果 |
|--------|------|
| **编译成功率** | K1A(v3) × 10 split-case + K1B × 3 + K2gA × 1 + K2gB × 1 + K1A(v1) × 12 + K1B × 12 + K2gA × 12 + K2gB × 12 = **≥ 48 次 `get_kernel_source()` 全部成功**，无 `LOG(FATAL)`、无 `ValueError`、无 `InternalError` |
| **一维 grid 承载 2D 任务映射** | ✅ `T.Kernel(b_num*S, is_npu=True) as (cid, vid)` + `task=cid*VEC_NUM+vid; row=task//S; sid=task%S` 编译通过。生成码 `K1A_case7.cc` 折叠为 `row = cid/5`、`sid = (cid%5)*2 + vid`、`b = (cid%5)*8 + vid*4 + lb`；已逐项代入 `cid∈[0,20)×vid∈[0,2)` 核对为**与源语义代数等价的双射**（40 任务 ↔ `B*S = 4*10 = 40`） |
| **静态循环边界** | ✅ 生成码全部为字面常量：`for (int32_t lb = 0; lb < 4; ++lb)`、`for (int32_t s = 0; s < 10; ++s)`、`for (int32_t b = 0; b < 37; ++b)`、`for (int32_t l = 0; l < 256; ++l)`、`for (int32_t b = 0; b < 8; ++b)`、`for (int32_t l = 0; l < 128; ++l)`。**零动态边界** |
| **运行时守卫 → 真实 C++ `if`** | ✅ `K1A_case7.cc:44` `if (((((cid % 5) * 8) + (vid * 4)) + lb) < 37) {`；`K1B_case*.cc` 9 个运行时 `if`；`K2gA_case7.cc` 2 个；`K2gB_case7.cc` 1 个 |
| **`padCount == 0`** | ✅ 全部 TopK 调用点，10/10 case（§4.9 详表） |
| **UB→UB 32B 对齐** | ✅ 全部 `copy_ub_to_ub` 计数 ∈ {128,256,320,512,832,2048} 元素 = {512,1024,1280,2048,3328,8192} B（§4.9 详表） |
| **GM↔UB 的 1 元素拷贝可用** | ✅ `copy_gm_to_ub<float,1>(bv[0], pv_val[…], 40, 1, 1, 0.0)` 与 `copy_ub_to_gm<float,1>(pv_val[…], best_val[0], 40, 1, 1)` 均正常生成（不触发 N5 的 UB→UB 禁令） |
| **尾块 `pad_value` 路径不变** | ✅ `copy_gm_to_ub<float, 512, 1>(ztp[0], logits[…151552], 608256, 1, 512, -CUDART_INF_F)` —— 与现行 K1 的尾块载入**同一形态** |
| **UB 实测** | ✅ K1A ≤ 106496、K1B ≤ 89444、K2gA ≤ 148420、K2gB = 132（§4.5 详表），全部 < 196352 |
| **epilogue 未被 split 破坏** | ✅ `K1B_case7.cc` 的 `PipeBarrier` 计数 `{PIPE_V: 32, PIPE_ALL: 5}` 与现行 `K1_case15_real.cc` 的 `{PIPE_V: 37, PIPE_ALL: 5}` 同量级，且 `tau_p` 循环仍是 `for (l=0; l<256; ++l)` 内含 3 个 `PIPE_ALL` ⟹ epilogue 代码逐条未变 |
| ❌ **暴露的陷阱（已固化为 N6/E-S1）** | `refine.py` 的单循环变体失败：`error: Undefined variable: z_ub`（`alloc` 在 `if n_blk_full>0:` 内、`use` 在其后的 `for` 里）；修补后又 `error: Undefined variable: ztp`（同一原因）。**⟹ `alloc` 与 `use` 必须同词法 `if` 作用域**，`refine3.py` 的 v3 形态（尾块 hoist 到 `if TAIL>0:` + `if sid == S_TAIL:`）成功 |

---

## 14. ⭐ bitwise 等价性论证（rev3 最关键章节）

> **本章的目标**：对约束 2「`m_out/s_out/tau_out/valid/tok_out` 与现行 kernel（sha256 `19e51e8d…`）
> 逐位相同」给出**逐 pass 的论证**，并**明确区分**「先验可证明」与「必须 Stage 2 实测」。
> 遵循 AGENTS.md 通用原则 3（区分事实/推断/建议）与调度指令 §12 禁令 6
> （禁止把未验证推导写成已验证结论）。

### 14.0 论证结构与结论摘要

| 输出 | 论证路径 | 先验可证明？ | 残留不确定性 |
|------|---------|-------------|-------------|
| `m_out` | softmax 扫描代码**逐条未改** + 块序未改 | ✅ **构造性成立**（不需要任何浮点论证） | 无 |
| `s_out` | 同上 | ✅ **构造性成立** | 无 |
| `tau_out` | `run_v` 逐位相同（M2）+ epilogue 代码**逐条未改** | ⚠️ **依赖 M2** | 🔴 **`topk` 对相同多重集是否产出完全相同的有序序列 —— 必须实测（V-S1）** |
| `valid` | 派生自 `run_v`/`m_run`/`s_run`，三者已论证 | ⚠️ 依赖 M2 | 同上 |
| `w_out` | K2m 代码**完全未改** + 输入 `m/s/tau` 逐位相同 | ✅ **构造性成立**（给定 `tau` 相同） | 依赖 M2（经 `tau`） |
| `tok_out` | `G_ij` 逐位相同（hash 是 `j` 的纯函数）+ M3（argmax 折叠可结合） | ⚠️ **依赖 M3 + `reduce_max/reduce_min` 的顺序无关性** | 🔴 **必须实测（V-S1）** |

**⟹ 5 个输出中，`m_out`/`s_out`/`w_out`（在 `tau` 相同的前提下）是构造性 bitwise；
`tau_out`/`valid`/`tok_out` 依赖 M2/M3 两条数学性质 + 硬件原语的顺序无关性。
后者是本设计唯一的实质性 bitwise 风险，已列为 V-S1（最高优先级 Stage 2 验证项）。**

### 14.1 ⭐ 为什么 rev3 不需要「朴素 split-merge」，以及为什么朴素 merge 确实不 bitwise（对 Orchestrator §3.1 的独立复核）

**Orchestrator §3.1 的论断：成立。** 本 Stage 1 独立复核如下。

现行 K1 主循环（`r4_fused_sampling.py:206-222`，其中 round2 #4 已删除 L211 的 UB→UB hop）：

```
m_blk   = reduce_max(z_b)                       # 块 b 的局部最大值
m_new   = max(m_run, m_blk)                     # 含本块的运行最大值
s_run  *= exp(m_run_prev - m_new)               # 重标定已累加部分
m_run   = m_new
s_blk   = reduce_sum( exp(z_b - m_new) )        # ← 关键：用【含本块的运行最大值】
s_run  += s_blk
```

⟹ 块 `b` 对 `s` 的贡献是 `Σ_{i∈b} exp(z_i − M_b)`，其中 `M_b = max(z_0 … z_{end of b})` 是**前缀最大值**。

**朴素 split-merge**（Stage 3 round1 [#3] 曾写下、Orchestrator §3.1 质疑的形式）：
split `j` 用**自己的局部最大值** `m_j = max_{i∈j} z_i` 算 `s_j = Σ_{i∈j} exp(z_i − m_j)`，
事后合并 `m = max_j m_j`、`s = Σ_j s_j · exp(m_j − m)`。

**数学上**：`s_j · exp(m_j − m) = Σ_{i∈j} exp(z_i − m_j) · exp(m_j − m) ≟ Σ_{i∈j} exp(z_i − m)`。
**fp32 下不成立**，有两处独立的舍入分歧：
1. **单项分歧**：`exp(a−b) · exp(b−c) ≠ exp(a−c)`。设 `a−b = x`、`b−c = y`、`a−c = x+y`。
   `fl(fl(exp(x)) · fl(exp(y)))` 与 `fl(exp(fl(x+y)))` 是两个不同的舍入路径，
   一般相差 1–2 ulp（`exp` 本身在 dav-2201 上是 SFU 近似，`hap_lib` 记为 1 个 SFU op）。
   即使 `exp` 是精确的，两次乘法/一次乘法的舍入也不同。
2. **求和顺序分歧**：现行是 `s_run` 沿块序 `0,1,2,…` **逐个累加并在每步重标定**
   （`s_run = s_run·exp(…) + s_blk`，一个**串行 scan**）；朴素 merge 是先算 S 个独立的 `s_j`
   （每个内部按块序累加，但**重标定只在 split 内**），再对 S 个结果做一次合并。
   两者的**加法结合树完全不同** ⟹ 舍入不同。

**偏差传播链**（说明为什么 1 ulp 也不可接受）：
```
s 差 1 ulp
  → q2 = exp(run_v − m)/s 差 ~1 ulp（K1B epilogue）
  → tau_p 的严格前缀和 prefix = Σ q2 差 ~1 ulp
  → 若 prefix 恰好在某 l 处跨越 p_i（prefix <= p_i 的判据翻转）
      ⟹ L 变 ±1 ⟹ tau_p = run_v[L] 变成【另一个 logit 原值】（不是 ulp 级，是离散跳变！）
  → tau = max(tau_k, tau_p, tau_u) 可能变
  → ① K2g 的 keep-mask (z >= tau) 翻转 ⟹ tok_out 变（int32 精确匹配门禁 FAIL）
    ② okp = 1[prefix > p_i] 翻转 ⟹ valid 变（int32 精确匹配门禁 FAIL）
    ③ K2m 的 w_out 在阈值边界处「0 vs 一个有限概率」离散跳变
```
**⟹ Orchestrator §3.1 的裁定完全正确：朴素 split-merge 违反约束 2，rev3 不采用。**
（这也修正了 Stage 3 round1 [#3] 写下的「满足结合律 ⟹ 可用」的隐含结论，见 §10.4 第 1 条。）

**rev3 的规避方式**：**根本不计算 `m_j`/`s_j`，也不做任何跨 split 的 softmax 合并。**
softmax 扫描完整地留在 K1B 里，块序 `0,1,…,n_blk_full−1` 与现行完全相同 ⟹
`s_run` 的每一步运算、每一个操作数的位模式都与现行**逐一相同** ⟹ **构造性 bitwise**。

### 14.2 `m_out` / `s_out` 的 bitwise 论证（构造性，无需浮点分析）

**命题**：K1B 产出的 `m_run`（→`m_out`）与 `s_run`（→`s_out`）与现行 K1 **逐位相同**。

**论证**：逐项对照 K1B 与现行 K1 的 softmax 代码：

| 步骤 | 现行 K1（`r4_fused_sampling.py`） | rev3 K1B | 是否相同 |
|------|-----------------------------------|----------|---------|
| buffer 声明 | `m_blk/m_run/m_old/s_blk/s_run/tmp [1,1]`、`z_ub[BV]`、`e_ub[1,BV]`、`m2d[1,BV]`；尾块 `zt/et/mt [1,TAIL]` | **同一组，同一 shape，同一 dtype** | ✅ |
| 初始化 | `fill(m_run, -inf)`、`fill(s_run, 0.0)`（L181-182） | 同 | ✅ |
| per-row 参数载入 | `T.copy(top_ks[row:row+1], k_i)` 等 3 个（L177-179） | 同 | ✅ |
| 满块循环边界 | `for b in T.serial(n_blk_full)`（L206） | 同（**块序 0..n_blk_full−1**） | ✅ |
| 满块 GM 载入 | `T.copy(logits[row, b*BV], z_ub)`（L207，无 `pad_value`） | 同 | ✅ |
| 满块 softmax 11 步 | L212-222：`reduce_max(z_ub, m_blk[0,0:1], -1)` → `max(m_old,m_run,m_blk)` → `sub(tmp,m_run,m_old)` → `exp(tmp,tmp)` → `mul(s_run,s_run,tmp)` → `max(m_run,m_run,m_blk)` → `broadcast(m2d,m_run)` → `sub(e_ub,z_ub,m2d)` → `exp(e_ub,e_ub)` → `reduce_sum(e_ub,s_blk,-1)` → `add(s_run,s_run,s_blk)` | **逐条相同，顺序相同** | ✅ |
| 尾块 softmax | L240-261（`if TAIL > 0:` 区段的前 11 步，精确尺寸 `[1,TAIL]` + 全量 reduce） | **逐条相同** | ✅ |
| 输出写回 | `T.copy(m_run[0,0:1], m_out[row:row+1])`、`T.copy(s_run[0,0:1], s_out[row:row+1])`（L359-360） | 同 | ✅ |
| **唯一差异** | 满块循环内**还有** (b) top-k 归并 6 步（L232-237）；尾块区段**还有** top-k 6 步（L262-269） | **这 12 步被移到 K1A** | ⚠️ 见下 |

**关键：被移走的 12 步是否影响 `m_run`/`s_run`？**
- (b) 段只写 `l_dst`/`l_val`/`merged`/`m_dst`/`run_v`，**不写** `m_run`/`s_run`/`m_blk`/`s_blk`/`m_old`/`tmp`/`z_ub`/`e_ub`/`m2d`。
  （逐行核对 L232-237：`topk(l_dst, z_ub, …)` **读** `z_ub`、写 `l_dst`；`gather_mask(l_val, l_dst, …)`；
  `copy(run_v, merged[:K_MAX])`；`copy(l_val, merged[K_MAX:])`；`topk(m_dst, merged, …)`；`gather_mask(run_v, m_dst, …)`。）
- ⚠️ **唯一的潜在耦合：N3（`topk` 对 src 做 in-place `-inf` 填充）**。现行代码的注释（L224-231）已论证：
  *「`z_ub`'s last read is the `T.tile.sub` above and it is overwritten by the GM→UB copy at the top of
  the next iteration, so topk's in-place `-inf` padding (N3) cannot poison anything.
  **`actual_num == BV == buffer size` → `padCount == 0`**, so Sort()'s masked-Duplicate remainder path
  is never taken」* ⟹ **满块的 `topk` 根本不修改 `z_ub`**（`padCount == 0` ⟹ 无填充）。
  尾块的 `topk` 源是**独立的** `ztt`（不是 `z_ub`/`zt`）⟹ 也不影响 softmax 侧。
- ⟹ **移除 (b) 段对 `m_run`/`s_run` 无任何影响**（既无数据依赖，也无 in-place 副作用）。

**∎ `m_out`/`s_out` bitwise 等价成立（构造性，不依赖任何浮点性质）。**

**编译层佐证**：`K1B_case7.cc` 的 `copy_gm_to_ub<float, 4096>(z_ub[0], logits[…+(b*4096)], 608256, 1, 4096, 0.0)`
与现行 K1 的对应指令**同形态**；`PipeBarrier` 计数 `{PIPE_V: 32, PIPE_ALL: 5}`（K1B）vs
`{PIPE_V: 37, PIPE_ALL: 5}`（现行）—— 差的 5 个 `PIPE_V` 正是被移走的 topk 段的 barrier，
softmax 段的 barrier 结构未变。

### 14.3 ⭐ `run_v`（→ `tau_out` / `valid`）的 bitwise 论证：M2 与 `topk` 的确定性

**这是 rev3 bitwise 论证中唯一不能纯先验完成的部分。**

**命题 M2**：设行 `i` 的全部 `V` 个值构成多重集 `M`。现行 K1 沿块序做
`run_v ← top_K(run_v ‖ top_K(M_b))`（`b = 0…n_blk_total−1`，`run_v` 初值 `[-inf]*K`）；
rev3 先对每 split `s` 做 `run_v_s ← top_K(∪_{b∈s} M_b)`（同样的逐块归并链），
再沿 split-id 升序做 `run_v ← top_K(run_v ‖ run_v_s)`。
**若 `top_K(X)` 返回的是「`X` 的值多重集中最大的 `K` 个值，按降序排列」，则两种方案的 `run_v` 逐位相同。**

**证明**：
1. **引理 L1（top-K 的嵌套性）**：`top_K(A ∪ B) = top_K( top_K(A) ∪ top_K(B) )`。
   *证*：设 `x ∈ top_K(A∪B)`，即 `A∪B` 中至多 `K−1` 个元素严格大于 `x`。
   若 `x ∉ top_K(A) ∪ top_K(B)`，则 `A` 中至少有 `K` 个元素 > `x` **或** `B` 中至少有 `K` 个 > `x`
   （否则 `x` 会在某一侧入选）。任一情形都给出 `A∪B` 中至少 `K` 个元素 > `x`，矛盾。
   反之 `top_K(A) ∪ top_K(B)` 中的元素显然属于 `A∪B`，且 `|top_K(A) ∪ top_K(B)| ≤ 2K`，
   其 top-K 必等于 `top_K(A∪B)`（因为任何不在 `top_K(A)∪top_K(B)` 里的 `A∪B` 元素都不够大）。**∎**
2. 由 L1 归纳：对任意**归并树形状**（左偏链、平衡树、任意分组），只要叶子是 `M` 的一个划分，
   最终结果都是 `top_K(M)`。⟹ 现行的「逐块左偏链」与 rev3 的「先 split 内左偏链、再 split 间左偏链」
   **产出同一个 `top_K(M)`**。**∎（引理层面）**
3. **`run_v` 只存值不存索引**：`gather_mask(l_val, l_dst, "P0101")` 取偶数位 = 值
   （rev2 §3.2 已用 `testing/python/language/test_tilelang_ascend_language_topk.py:169-170`
   的 `out_values = b[0::2]` 交叉验证）。⟹ `run_v` 是 `top_K(M)` 的**降序值数组**。
4. **降序值数组由值多重集唯一确定**：把多重集 `M` 的 top-K 元素按降序排列，结果唯一
   （重复值不可区分；`-inf`/`+inf` 各自全相等）。**∎**

**⟹ M2 在「`top_K` 是精确的 top-K 选择 + 精确降序排序」的前提下成立。**

**⚠️ 该前提的残留不确定性（必须 Stage 2 实测，V-S1）**：

`T.tile.topk` 的底层是 `tl::ascend::TopK<float>` → `Sort32`（每 32 元素块内排序）
→ `MrgSort` 归并树（`common.h`，本 Stage 1 已读：`stages=[(128,32),(32,8),(8,2),(2,1)]` 对 `actual=4096`）。
需要确认的是：

| 待确认点 | 为什么重要 | 先验判断 | 实测方法 |
|---------|-----------|---------|---------|
| **(i) `Sort32`/`MrgSort` 是否是「稳定且全序」的排序** | 若硬件排序对**相等值**的处理依赖它们在源数组中的位置（即排序 (value, index) 对），则**不同的输入顺序会给出不同的 index 排列**。但 `run_v` **只取值**，相等值的 index 差异**不可见** ⟹ 安全 | ✅ **先验认为安全**：`common.h` 的 `Sort` 用 `ArithProgression` 生成升序 index 数组，`Sort32(tmp, src, indices, repeatTimes)` 排的是 **(value, index) 对**，index 互异 ⟹ **全序** ⟹ 输出唯一。而 `gather_mask("P0101")` 丢弃 index ⟹ `run_v` 只依赖值多重集 | V-S1 的 D1 用例（`torch.equal` 于位模式，`w_out` 全 `B*V` 元素） |
| **(ii) 大量重复值（all-equal 行）** | `run_v` 会是 `K_MAX` 个相同值 ⟹ 排序结果平凡相同 | ✅ **先验认为安全**（相同值不可区分）。⚠️ 但 all-equal 行会触发 `tie_ok = 0` ⟹ `valid = 0` ⟹ 属 D-TIE 的既有降级路径，rev3 未改变 | Stage 2 的 L2/Boundary（现有套件已有全等行用例） |
| **(iii) `±inf` 值** | `-inf` 是**合法输入**（上游屏蔽）；`run_v` 初值也是 `-inf` | ✅ **先验认为安全**：`-inf` 在 fp32 中是全序的良定义元素，`Sort32`/`MrgSort` 用 fp32 比较 ⟹ `-inf` 排在最后。若某 split 的真实候选不足 `K_MAX`，其 `run_v_s` 尾部是 `-inf`（N4 初值），跨 split 归并后这些 `-inf` 仍排在最后 ⟹ 与现行「`run_v` 初值 `-inf` 被逐步挤出」的最终状态**是同一个值多重集**（见下方 ⚠️） | V-S1 + V-S4（某 split 全 `-inf`） |
| **(iv) NaN** | NaN 的比较全为 False ⟹ 全序失效 ⟹ 排序结果**依赖输入顺序** | 🔴 **M2 在 NaN 下不成立**。但 NaN 是**契约外**（rev2 §9.3；且现行 kernel 同样未定义）⟹ 不是 rev3 引入的回归 | 不测（L2 负向域） |

**⚠️ (iii) 的严格论证（`-inf` 初值的挤出等价性）**：
- 现行：`run_v` 初值 `[-inf]*K`，沿 `n_blk_total` 个块逐个归并。每块贡献 `K_MAX` 个真实值
  （满块 `BV ≥ K_MAX` 个；尾块 `TAIL_PAD ≥ TAIL ≥ K_MAX`，由 C3 host 校验保证）。
  归并 `n_blk_total` 次后，`run_v` = `top_K( {−inf}×K ∪ M )`。因 `|M| = V ≥ K_MAX = K`（C7 保证）
  且 `M` 中至少有一个 `> −inf` 的元素（除非整行都是 `-inf`），一般情形下 `run_v` 中**不含初值 `-inf`**。
- rev3：每 split `run_v_s` 初值 `[-inf]*K`，归并其 `per[s]` 个块（`per[s] ≥ 1`，定理 P3）
  ⟹ `run_v_s = top_K({−inf}×K ∪ M_s)`。K1B 再归并 S 次
  ⟹ `run_v = top_K( {−inf}×K ∪ ∪_s M_s ) = top_K({−inf}×K ∪ M)`（因为 `{−inf}×K` 在 L1 引理下
  与 `top_K({−inf}×K) = {−inf}×K` 等价，且 `-inf` 是吸收性的最小元素）。
  **⟹ 两侧的 `run_v` 是同一个多重集的 top-K ⟹ 逐位相同。** ✅
- **退化情形：整行全是 `-inf`**。现行 ⟹ `run_v = [−inf]*K`。rev3 ⟹ 每 `run_v_s = [−inf]*K`，
  归并后仍 `[−inf]*K`。✅ 相同。（此时 `m_run = −inf`、`s_run = 0`、`q2 = exp(−inf −(−inf))/0 = exp(NaN)/0 = NaN`
  ⟹ **现行与 rev3 同样产生 NaN** ⟹ 不是 rev3 的回归；且该输入属契约外。）

**结论**：M2 **在数学上成立**，且 (i)(ii)(iii) 三种边界都有先验论证支持；
**但「`tl::ascend::TopK` 的实现是否严格等价于精确 top-K + 精确降序排序」只有硬件能回答**
⟹ **V-S1 是必须实测项，本设计不声称已验证。**

**epilogue 部分**：`q2 / tau_k / tau_u / tau_p / okp / n_gt / n_eq / tau / tieok / vflag`
的代码在 K1B 中**逐条未改**（§3.3 伪代码已标注「与现行 L271-362 逐条相同」），
输入 `run_v`/`m_run`/`s_run`/`k_i`/`p_i`/`mu_i` 已论证逐位相同
⟹ **epilogue 输出逐位相同（构造性，给定 `run_v` 相同）**。
特别地，`n_gt`/`n_eq` 只在 `run_v` 的 `K_MAX` 个元素上统计（round1 [#1] 已向量化为
`compare + select + reduce_sum`，且其 bitwise 等价性已由 round1 论证并被 36/36 cann-bench 验证），
**不扫全 `V`** ⟹ `run_v` 相同即安全。**这印证了 Orchestrator §3.3 的判断。**

### 14.4 `tau_k` / `tau_p` / `L` 的派生安全性（对 Orchestrator §3.3 的复核）

Orchestrator §3.3 论断：*「`n_gt`/`n_eq` 只在 `run_v` 的 256 个元素上统计，不扫全 `V` ⟹ `run_v` 相同即安全。
`tau_k`/`tau_p`/`L` 全由 `run_v` 派生 ⟹ 安全。」*

**裁定：✅ 成立。** 逐项核对现行代码：
- `tau_k[0] = run_v[k_i[0] - 1]`（L284）⟹ 纯 `run_v` 索引 ✅
- `tau_p` 的 `T.serial(K_MAX)` 循环读 `run_v[l]` 与 `q2[0,l]`（L299-303），
  而 `q2 = exp(run_v − m)/s`（L273-278）⟹ 纯 `run_v`/`m_run`/`s_run` 派生 ✅
- `L`（隐含在 `tau_p` 的循环终止位置）⟹ 由 `prefix` 与 `p_i` 的比较决定，`prefix = Σ q2` ⟹ 同上 ✅
- `n_gt`/`n_eq`（L316-322）⟹ `compare(tie_mask, run_v, tie_tkbc, "GT"/"EQ")` + `select` + `reduce_sum`，
  只在 `[K_MAX]` 上 ✅
- `tau_u = m_run + ln(mu)`（L289-294）⟹ 纯 `m_run`/`mu_i` 派生，**与 `run_v` 无关** ✅
- `okp`（L307）⟹ `p_i` 与 `prefix` ✅；`tieok`/`vflag`（L340-357）⟹ `tau`/`tau_k`/`n_gt`/`n_eq`/`k_i`/`okp` ✅

**⟹ 全部 6 个 epilogue 输出量都是 `{run_v, m_run, s_run, k_i, p_i, mu_i}` 的纯函数，
不依赖 `V` 上的任何逐元素信息。三者已论证逐位相同 ⟹ epilogue 逐位相同。∎**

**⚠️ rev2 已记录的既有缺口在 rev3 中不变**（`r4_fused_sampling.py:338-339` 注释）：
*「Known gap: a tie pair straddling ranks `K_MAX−1`/`K_MAX` is invisible here (would need `z_sort[K_MAX]`)」*。
⟹ 这是 rev2 的**已接受偏差**，rev3 **既不修复也不恶化**（`run_v` 逐位相同 ⟹ 缺口位置逐位相同）。

### 14.5 ⭐ K2g（`tok_out`）的 bitwise 论证：M3 与跨 split tie-break

**Orchestrator §3.4 的三点「有利」判断，本 Stage 1 逐条复核：**

| Orchestrator 判断 | 裁定 | 论证 |
|------------------|------|------|
| **hash 是 `j` 的纯函数 ⟹ 每 split 可独立生成自己 V 区间的 `G_j`，无需跨 split 通信** | ✅ **成立** | `x_ij = (seed_i·PRIME_W + j) mod 2^32`（`arith_progression(state_ub, seedP_v[0] + b*BV, 1, BV2)`，`b*BV + 局部偏移 = 全局 j`）；`state = lowbias32(x)` 是**逐元素**的 3 轮位运算；`u = clamp(state·2^-32 + 0.5, …)`、`G = -ln(-ln(u))` 也都是**逐元素**的。⟹ **`G_ij` 只依赖 `(seed_i, j)`，与 split 划分完全无关**。⚠️ **前提是 `b` 必须是全局块号（N11）** —— 若误用局部 `lb`，则 `j` 全错 |
| **round1 [#2] 已把每块 argmax 从 `topk(K=1)` 改为 `reduce_max + compare(EQ) + select + reduce_min`，天然可跨 split 合并** | ✅ **成立** | 该改造产出的 `(blk_max, blk_argmax)` 是一个**可结合的 (max, min-index) 对**（见下方 M3 证明）。而 `topk(K=1)` 形态**也**可结合（它同样产出 (max, min-index)），但 round1 [#2] 的形态**避免了 full Sort(4096)** ⟹ 与 rev3 的 split 正交且叠加受益 |
| **K2g 需要 `tau`（来自 K1）⟹ 天然满足（本就是两个 kernel）** | ✅ **成立** | rev3 的 launch 顺序是 K1A → K1B → K2gA → K2gB，K2gA 读 `tau_out`（K1B 的输出）⟹ 依赖天然满足，**且 rev3 未改变该依赖的方向或强度** |

**命题 M3（跨 split 的 argmax-with-min-index 折叠 bitwise 等价）**

现行 K2g（`r4_fused_sampling.py:546-550`）的跨块折叠：
```
best_val ← -inf ; best_idx ← 0.0
for b = 0 .. n_blk-1:                                  # 全局块号升序
    if blk_max[b] > best_val:                          # 严格 GT
        best_val ← blk_max[b] ; best_idx ← blk_argmax[b] + b*BV
```
**引理 L2**：上述折叠的终值 = `( max_b blk_max[b] ,  min{ b*BV + blk_argmax[b] : blk_max[b] = max } )`，
**当且仅当**初值 `best_val = -inf` 严格小于所有 `blk_max[b]`，或所有 `blk_max[b] = -inf` 时终值为初值 `(-inf, 0.0)`。
*证*：严格 GT ⟹ 只有在**严格更大**时才更新 ⟹ 遇到相等值不更新 ⟹ 保留**最早**（块号最小）的那个；
块内 `blk_argmax = reduce_min{ j : z_j = blk_max }` ⟹ 块内最小 index。
两者合成 ⟹ 全局最小 index。终值是 `max` 与「达到 max 的最小全局 index」。
`-inf` 初值情形：`-inf > -inf` 为 False ⟹ 永不更新 ⟹ 终值 `(-inf, 0.0)`。**∎**

**rev3 的两层折叠**：
```
# K2gA（每 split 独立）：与现行【逐条相同】的循环，只是 b 的范围是 [sid*NBS, min((sid+1)*NBS, n_blk))
best_val_s ← -inf ; best_idx_s ← 0.0
for lb = 0 .. NBS-1:  b = sid*NBS + lb
    if b < n_blk:
        <hash/Gumbel/mask/argmax —— 与现行 L488-545 逐条相同>
        if blk_max[0] > best_val_s[0]:
            best_val_s ← blk_max ; best_idx_s ← blk_argmax + b*BV      # ⭐ 全局块号 b
→ pv_val[row*S+sid] = best_val_s ; pv_idx[row*S+sid] = best_idx_s

# K2gB（每行串行）：与现行【逐条相同】的折叠形态，只是折叠对象从「块」变成「split」
best_val ← -inf ; best_idx ← 0.0
for s = 0 .. S-1:                                                      # ⭐ split-id 严格升序
    if pv_val[row*S+s] > best_val:                                     # 严格 GT
        best_val ← pv_val[...] ; best_idx ← pv_idx[...]
tok_i32 ← cast(best_idx, "CAST_ROUND") ; → tok_out[row]
```

**证明 rev3 终值 = 现行终值**：
1. 由引理 L2 应用于 K2gA 的 split `s`：`(best_val_s, best_idx_s) = ( max_{b∈s} blk_max[b] , min{ b*BV+blk_argmax[b] : b∈s, blk_max[b] = max_{b∈s} } )`，
   或（当 split `s` 的所有块都是 `-inf`）`(-inf, 0.0)`。
2. 由引理 L2 应用于 K2gB：终值 `= ( max_s best_val_s , best_idx_{s*} )`，其中
   `s* = min{ s : best_val_s = max_s best_val_s }`（严格 GT ⟹ 取**最小 split-id**）。
3. `max_s best_val_s = max_s max_{b∈s} blk_max[b] = max_b blk_max[b]` ✅（`max` 结合/交换，且 fp32 `max` **精确无舍入** ⟹ 顺序无关，这是 M1 的实例）。
4. **index 部分**：需证 `best_idx_{s*} = min{ b*BV + blk_argmax[b] : blk_max[b] = 全局 max }`。
   设全局 max 为 `μ`，达到 `μ` 的块集合为 `B_μ`。
   - `s* = min{ s : max_{b∈s} blk_max[b] = μ } = min{ s : s 含有 B_μ 中的块 } = min{ s : B_μ ∩ s ≠ ∅ }`。
   - 在 split `s*` 内，`best_idx_{s*} = min{ b*BV + blk_argmax[b] : b ∈ s*, blk_max[b] = μ }`
     （由步骤 1，注意 `max_{b∈s*} = μ`）。
   - 对任何 `s > s*`，其贡献的 index 都 `≥ s*NBS*BV > s** 的 index 上界` ⟹ 严格更大（因 split 按**块号连续区间**划分，
     `b ∈ s ⟹ b ≥ s*NBS`，故 `b*BV + blk_argmax[b] ≥ s*NBS*BV`，对 `s > s*` 严格大于 `s*` 内任何 index 的上界
     `(s*+1)*NBS*BV − 1`）。
   - 对任何 `s < s*`，由 `s*` 的最小性，`B_μ ∩ s = ∅` ⟹ `best_val_s < μ` ⟹ 严格 GT 判定不触发 ⟹ 不影响终值。
   ⟹ `best_idx_{s*} = min{ b*BV + blk_argmax[b] : b ∈ B_μ } = 全局最小 index` ✅
5. **`-inf` 退化情形**：若所有块的 `blk_max = -inf`（整行都被 `tau` 屏蔽），
   则每 `best_val_s = -inf`、`best_idx_s = 0.0`；K2gB 中 `-inf > -inf` 为 False ⟹ 终值 `(-inf, 0.0)`
   ⟹ `tok_out = cast(0.0) = 0`。**现行**：同样 `-inf > -inf` False ⟹ 终值 `(-inf, 0.0)` ⟹ `tok_out = 0`。✅ **相同**
6. `blk_argmax` 本身：`reduce_max(z_ub, blk_max, -1)` + `compare(mask, z_ub, blk_max[0], "EQ")`
   + `select(g_ub, mask, idx, float(BV), …)` + `reduce_min(g_ub, blk_argmax, -1)` —— 这四步在 K2gA 中
   **逐条未改**，且 `z_ub`（该块的 `cand` 值）在两条路径下**逐位相同**
   （因 `G_ij` 逐位相同 + `logits` 逐位相同 + `tau` 逐位相同 + `compare/select` 逐条相同）。
   ⚠️ `reduce_max`/`reduce_min` 的**顺序无关性**：`max`/`min` 在 fp32 下**精确无舍入** ⟹ 归约树形状无关 ⟹ 逐位相同。**✅ 先验成立**
7. `cast(best_idx, "CAST_ROUND", 1)` 与 `T.copy(tok_i32, tok_out[...])` ⟹ 逐条未改，输入逐位相同 ⟹ 输出逐位相同。

**∎ M3 成立（在 `tau` 逐位相同、`G_ij` 逐位相同、`max/min` 精确无舍入的前提下，全部先验可证）。**

**⚠️ M3 的残留不确定性（必须 Stage 2 实测）**：
- **(a) `tau` 必须逐位相同** —— 这依赖 §14.3 的 M2（`run_v` 逐位相同）⟹ **M3 的 bitwise 结论继承 M2 的不确定性**。
  若 `tau` 差 1 ulp，`compare(mask, z_ub, t_v[0,0], "GE")` 在阈值边界的**单个元素**上可能翻转
  ⟹ `cand` 集合变 ⟹ `tok_out` 可能跳到另一个 index。**这是 rev3 最尖锐的风险点。**
- **(b) `reduce_max`/`reduce_min` 对全宽 `[BV]` rank-1 源的归约树是否与 `BV` 无关地确定** ——
  先验认为是（`BV` 是编译期常量，归约树由 codegen 静态生成），且 round1 [#2] 的改造已被
  36/36 cann-bench + stress 8/8 验证 ⟹ 置信度高。但 rev3 未改变这一步，故不引入新风险。
- **(c) `+inf` 的 `G_ij`**：`G = -ln(-ln(u))`，`u ∈ [2^-24, 1-2^-24]` ⟹ `-ln(u) ∈ [2^-24 附近, 24·ln2]`
  ⟹ `G` 有限。`cand = z + G`：若 `z = +inf` 则 `cand = +inf`，多块都 `+inf` 时 tie-break 取最小 index
  ⟹ 与现行一致（`+inf > +inf` 为 False，严格 GT 保留最早）。✅ 先验安全，但未实测。

### 14.6 ⭐ 明确列出「无法先验证明、必须 Stage 2 实测」的点

**这是本章最重要的部分**（调度指令 §12 禁令 6：禁止把未验证推导写成已验证结论）。

| # | 无法先验证明的点 | 为什么先验不够 | 实测方法（§10.2 编号） | 若失败的退路 |
|---|-----------------|---------------|----------------------|-------------|
| **U1** | 🔴 **`tl::ascend::TopK<float>` 对相同的值多重集，是否**总是**产出完全相同的降序值序列** —— 含 (a) 大量重复值、(b) `±inf`、(c) `K == actual_num`（全取）三种边界 | 本 Stage 1 读了 `common.h` 的 `Sort32`/`MrgSort` 归并树结构，确认它排的是 (value, index) 对且 index 互异 ⟹ **理论上全序唯一**。但：① `MrgSort` 的 `ifExhaustedSuspension = false` 与 `validBit` 语义未读 AscendC 文档；② `Sort32` 是**硬件指令**，其对相等值/`-inf` 的行为无源码可查；③ rev3 的归并树**形状**与现行不同（现行 38 次 `topk(512)`；rev3 是 每 split `per[s]` 次 + K1B `S` 次），若硬件排序对**输入顺序**敏感（不稳定排序 + 相等值），则形状不同 ⟹ 结果可能不同 | **V-S1**（D1 用例，`torch.equal` 于位模式，`w_out` 全 `B*V` 元素）+ **V-S4**（某 split 全 `-inf`）+ Stage 2 的 L2/Boundary（全等行、含 `-inf` 行） | 若 U1 失败：① 首先确认是否是 `-inf` 初值挤出问题（§14.3 (iii)）；② 若是相等值的排序不稳定 ⟹ **改为「所有 split 的 `run_v_s` 先在 K1B 里拼成 `[S*K_MAX]` 再做一次 `topk(K_MAX, S*K_MAX)`」**（单次排序 ⟹ 形状唯一，但 `S*K_MAX` 必须 `%32==0` ✅ 且 `≤ UB` 需重算）；③ 若仍失败 ⟹ **返回 `[DESIGN_ERROR]`**，因为约束 2 无法满足 |
| **U2** | 🔴 **`a_launch` 的实际值** | 本 Stage 1 无 NPU 运行时测量能力（只有 `get_kernel_source()` 编译能力） | **V-S3**（§9.4 两个新哨兵 + `k1_split.py` 位置拆分法） | 更新 `a_launch_est` 单一常量；若 `> 8 us` 则收紧 guard（可能使 split case 从 6 降到 3~4） |
| **U3** | ⚠️ **`F(K_MAX=128)` 与 `p(2048)` 的真值**（现有 2 个数据点相互矛盾，R-F128） | 同上，且 msprof 的 K1/K2 位置拆分本身有误差 | **V-S8**（case8 单独实测） | 若 case8 回退 >3% ⟹ guard 加一行 `and K_MAX == 256`，把第二特化排除出 split 路径（split case 从 6 降到 5） |
| **U4** | ⚠️ **`grid = b_num*S` 在真实硬件上的调度是否真是「1 波」** | 生成码地址算式已核对（§3.2），但**波数**是 runtime 调度行为，取决于 driver 的 block 派发策略。若 driver 不按 `cid` 顺序派发、或 `b_num*S = 20` 时仍有 1 个 block 落入第 2 波，则定理 P2 的前提失效 | **V-S7**（msprof 读 `Block Num` + `Duration`；并对比 `grid=20` 与 `grid=19` 的时长是否连续） | 若发现 `grid=20` 落在 2 波 ⟹ `S_cap` 改为 `(CUBE_CORE_NUM - 1) // b_num`（留 1 个 block 余量） |
| **U5** | ⚠️ **K1A 的 4 个 TopK 调用点共享单一 `tmp_ub` arena（65536 B）时，是否存在 arena 复用竞态** | `allocate_tmp_buffer.cc:1195-1206` 的 `GetTmpBufferSize_` 取 `max` ⟹ 单一共享 arena；rev2 §7.2 记录了「arena 无内容契约，每个 call 可覆写」并依赖 `kWorkspaceWriteAccess` 建模。rev3 的 K1A 有 **4 个** TopK 调用点（满块 2 + 尾块 2），比现行 K1 的调用点**分布**不同（现行的 4 个调用点也是 2+2，但分属 `if n_blk_full>0` 与 `if TAIL>0` 两个区段 —— rev3 结构相同）⟹ **先验认为无新风险** | **V-S5**（UB 投毒 + 20 reps 逐位相同） | 若发现竞态 ⟹ 给尾块 topk 显式传独立 `tmp=`（`tilelang/language/ascend.py:141-142` 要求 scope `shared.ub`），UB 需重算 |
| **U6** | ⚠️ **`torch.empty` 的 workspace 在某些 driver 下是否可能返回带「毒值」的复用显存，从而在 K1A 漏写时掩盖 bug** | 这是**测试有效性**问题而非正确性问题：`empty` 可能恰好返回全 0 的复用页 ⟹ 漏写不被发现 | **V-S5** 的 UB 投毒 + §8.6 建议的「调试期临时用 `torch.zeros`」+ **F1 用例**（专测空 split） | 无需退路（属测试方法学） |

**先验可证明（不需实测）的部分**：
- ✅ `m_out`/`s_out` bitwise（§14.2，**构造性**：代码逐条相同 + 被移走的 12 步无数据/副作用耦合）
- ✅ `w_out` bitwise（给定 `tau` 相同；K2m 代码完全未改）
- ✅ epilogue 是 `{run_v, m_run, s_run, k_i, p_i, mu_i}` 的纯函数（§14.4，逐行核对）
- ✅ M1（前缀最大值顺序无关，`max` 精确无舍入）
- ✅ M3 的**折叠代数**（§14.5 步骤 1-6，含 `-inf` 退化情形与 index 严格性论证）
- ✅ `padCount == 0` 与 UB→UB 32B 对齐（§4.9，**且已编译实证**，不只是先验）
- ✅ UB 不溢出（§4.5，**已实测**）
- ✅ 静态循环边界、一维 grid 承载 2D 映射、运行时守卫降级为真实 `if`（§13.6，**已编译实证**）

---

## 15. ⭐ 契约与评测基础设施连带影响（rev3 核心新增章节）

### 15.1 `proto.yaml`：**不变**（明确声明）

**结论：rev3 不修改 `examples/R4/proto.yaml`，文件保持原样。**
本 Stage 1 已用 `sha256sum` 记录其当前值：
`41ab797002568e89fcc9a9eaa58be4f1cdfce20b6406abd0fdbf2ad11f7f8949`。
Stage 2 交付前必须复验该 sha256 **未变**（§9.2.5 静态断言的隐含项）。

**逐字段论证（为什么每个字段都不需要变）**：

| `proto.yaml` 字段 | 是否需要变 | 论证 |
|------------------|-----------|------|
| `operator.name` = `r4_fused_sampling` | ❌ 不变 | 算子名不变 |
| `operator.category` = `Fusion` | ❌ 不变 | 仍是多 kernel 融合链（链长从 2 增至最多 4，但 category 粒度不区分链长） |
| `operator.formula` | ❌ 不变 | ⭐ **split-V 是实现路径变更，不是语义变更**。`m_i = max_j z_ij`、`s_i = Σ_j exp(z_ij − m_i)`、`tau_i = max(tau_k, tau_p, tau_mu)`、`valid_i`、`w_ij`、`tok_out_i = argmax_j cand_ij` 的**数学定义完全不变**。§14 论证的正是「新实现产出与旧实现逐位相同的值」⟹ 公式无需重写 |
| `operator.attrs`（8 项：`K_MAX`, `BV`, `NEED_MIN_P`, `FUSE_GUMBEL`, `top_k`, `top_p`, `min_p`, `need_min_p_sampling`） | ❌ 不变 | ⭐ **`S` / `NBS` / `S_TAIL` 不列为 attr**，理由三条：① 它们是 **`(B, V, K_MAX, BV, CUBE_CORE_NUM)` 的函数**（§5.8.2 的算法是确定性的），**不是用户可调参数**；② 用户/夹具**从不显式指定它们**（对比 `BV`：`param_builder.py:203-206` 说明 `cases.yaml` 永远显式指定 `BV`，故 `BV` 是契约属性）；③ 把它们列为 attr 会让覆盖门禁派生出 `D-PARAM-S-*` 维度，而 `S` 的取值域依赖设备（`CUBE_CORE_NUM`）⟹ 门禁不可移植。**这与 `proto.yaml:86-88` 对 hash 常量 `PRIME_W`/`M1`/`M2` 的既有处理完全同构**（「是固定常量，非用户可调 attr … 不进 JIT 特化键、不列为可变 attr」）—— `S` 虽进特化键，但同样是**派生量**而非**契约量** |
| `operator.inputs`（5 项） | ❌ 不变 | ⭐ **`pv` / `pv_val` / `pv_idx` 不列为 input**：它们是**内部 workspace**，由 host wrapper 用 `torch.empty` 分配（§5.6 审计），**不由调用方提供**，也不出现在 `r4_fused_sampling(...)` 的签名里。与 `proto.yaml:126-129` 注释块把 kernel 级签名与算子级 schema 分开记录的做法一致 |
| `operator.outputs`（6 项） | ❌ 不变 | 输出的 name/shape/dtype 全部不变（§4.2） |
| `operator.schema` | ❌ 不变 | ⭐ `r4_fused_sampling(logits, top_ks, top_ps, min_ps, need_min_p_sampling, rand, K_MAX, BV, FUSE_GUMBEL)` 的**签名与位置参数顺序完全不变**。分派在函数体内部完成，对调用方透明。**双路分派在 host wrapper 内部完成 ⟹ cann-bench 18 case 仍全部有效** |

**⟹ 覆盖门禁 `coverage_check.py --proto` 的应覆盖维度完全不变**：
`D-DTYPE-float32` / `D-DTYPE-int32`（来自 `inputs[].dtype`）+ 8 个 `D-PARAM-*`（来自 `attrs[].name`）。
现有 **25 PASS / 0 MISS / 0 N/A** 的结果**必须保持**（§9.2.4）。

**⚠️ 一处需在 `proto.yaml` 之外记录的变化**：`proto.yaml:126-129` 的注释块列出了 3 个 kernel 的签名
（`r4_topk_stats` / `r4_mask_sample` / `r4_gumbel_sample`）。rev3 新增 4 个 kernel。
**建议**（Stage 2 可选，**非必须**）：在该注释块追加 4 行 kernel 签名以保持文档同步。
⚠️ 但这会改变 `proto.yaml` 的 sha256 ⟹ **若追加，必须同步更新覆盖门禁的基线记录**；
**若不追加，`proto.yaml` 逐字节不变，门禁零风险**。
**本设计的推荐是「不追加」**（最小必要改动原则，AGENTS.md 通用原则 4）：kernel 清单属于
`DESIGN.md §13.1` 的职责，不属对外契约文件。

### 15.2 `BV` / `K_MAX` 是契约属性，`S` 的选取**不依赖修改它们**（约束 7 的落实）

**背景**（round2 #3 教训，Orchestrator §7）：`r4_ops.py` 文档明写「params are passed by keyword,
mapped from `proto.inputs[].name` then `case.attrs`」；`param_builder.py:203-206` 里签名默认值只是
`elif` 兜底，`cases.yaml` **永远显式指定 `BV`** ⟹ `r4_fused_sampling.py:565` 的 `BV=4096` 默认值
在**所有测量路径上是死代码**。夹具 `BV` 本就异构：**16 case = 4096，case 8/13 = 2048 + K_MAX=128**。

**rev3 的落实**：
1. **`S` 的算法（§5.8.2）把 `BV` 与 `K_MAX` 当作只读输入**，从不修改它们。
   `S` 只依赖 `(B, V, BV, CUBE_CORE_NUM)`；`NBS`/`S_TAIL` 只依赖 `(V, BV, S)`。
2. **两个特化都被显式处理并实测**：
   - `(K_MAX=256, BV=4096)`：case 3/4/7/12/17（split 路径）+ case 1/2/5/6/9/10/11/14/15/16/18
   - `(K_MAX=128, BV=2048)`：case 8（split 路径）+ case 13
   ⟹ §4.5 的 UB 实测、§4.9 的 `padCount`/对齐实测、§5.8.4 的 `S` 表**都覆盖了两个特化**。
3. **`t_est` / `t_mrg_est` 按特化取值**（§13.3）：`t(4096,256)=8.902`、`t(2048,128)=3.955`、
   `t_mrg(256)=0.974`、`t_mrg(128)=0.453` ⟹ guard 对两个特化都用**各自的实测常数**判定，不混用。
4. ⚠️ **`AT = _aligned_topk(K_MAX)` 已按 `K_MAX` 参数化**（`r4_fused_sampling.py:86-88`：
   `ceil(2K/8)*8` ⟹ `K=256→512`、`K=128→256`），rev3 的 K1A/K1B 沿用同一函数 ⟹ 自动适配两个特化。
5. ⚠️ **R-F128 / V-S8**：`K_MAX=128` 特化的收益预测偏乐观（`F128` 不可分离）⟹ 若 case8 实测回退 >3%，
   退路是 guard 加一行 `and K_MAX == 256`（**只影响分派，不影响契约**）。

### 15.3 ⭐ `Block Num == ceildiv(B,2)` sanity gate 与 `t_hw` 模型的有意识更新

#### 15.3.1 `Block Num` sanity gate

**现状**（本 Stage 1 已定位）：
- `Block Num` 是 msprof `kernel_details.csv` 的**一列**（`bench/reports/prof_data/**/kernel_details.csv:1`
  的表头含 `…,Wait Time(us),Block Num,Mix Block Num,…`）。
- 该 gate 的**R4 侧实现**在 `perf_tuning/round2_item4/k1_split.py`：
  `:9` 文档「Case attribution is CROSS-CHECKED against `Block Num == ceil(B/2)` for every step」；
  `:32` 读 `int(r["Block Num"])`；`:42-46` `exp = -(-B_BY_CASE.get(ci,0)//2)`；
  `if blk1 != exp or blk2 != exp: bad += 1`。
- 它同时被 `.orchestrator_state.json:161` 记录为 cann-bench 评测的 9 项 sanity gate 之一
  （「`Block Num == ceildiv(B,2)` 16/16」），并在 `:499` 被明确标为
  「**that gate MUST be consciously updated, not silently broken**」。

**为什么会失效**：rev3 split 路径下，一个 step 内的 `main_kernel` 行数与 `Block Num` 都变了：

| 路径 | 任务 | 每 step 的 `main_kernel` 行数 | 各行 `Block Num`（按 launch 顺序） |
|------|------|------------------------------|-----------------------------------|
| 现行 | mask | **2** | `[b_num, b_num]`（K1, K2m） |
| 现行 | fused | **2** | `[b_num, b_num]`（K1, K2g） |
| ⭐ split | mask | **3** | `[b_num*S, b_num, b_num]`（K1A, K1B, K2m） |
| ⭐ split | fused | **4** | `[b_num*S, b_num, b_num*S, b_num]`（K1A, K1B, K2gA, K2gB） |

⟹ `k1_split.py:44` 的 `if len(seq) != 2: bad += 1; continue` 会**静默丢弃**所有 split-path step
（6 个 case × 8 step = **48 个 step 被丢弃**），且 `blk != exp` 会对 K1A/K2gA 报 bad。
**这正是「静默失效」——不报错，只是悄悄少统计 1/3 的数据。**

**有意识的更新方案**（Stage 2/3 必须执行，**不得跳过**）：

1. **把期望值改为「由 `_r4_split_plan` 派生」**，而不是硬编码 `ceil(B/2)`：
   ```python
   # k1_split.py 的更新（示意）
   from r4_fused_sampling import _r4_split_plan          # 复用 host 的同一判据函数 ⟹ 单一真相源
   use_split, S, NBS, S_TAIL = _r4_split_plan(B, V, K_MAX, BV)
   bn = -(-B // VEC_NUM)
   expect_rows   = (3 if task == "mask" else 4) if use_split else 2
   expect_blocks = ([bn*S, bn, bn*S, bn][:expect_rows] if use_split and task == "fused"
                    else [bn*S, bn, bn]            if use_split and task == "mask"
                    else [bn, bn])
   assert len(seq) == expect_rows, (ci, task, len(seq), expect_rows)
   assert [b for _, b, _ in seq] == expect_blocks, (ci, task, ...)
   ```
   ⭐ **关键设计**：期望值**从 host 的同一判据函数派生** ⟹ 判据一改，gate 自动跟着改，
   **不可能出现「gate 与实现脱节」**（这是「有意识更新」而非「静默失效」的技术保证）。
2. **在 gate 报告里显式打印每 case 的路径与 `S`**，使「哪些 case 走了 split」成为**可见的一等输出**
   （而不是隐含在时长里）。建议直接引用 §5.8.4 的表作为期望基线。
3. **若该 gate 也在 cann-bench harness 侧（read-only，另一 agent 所有）存在**：
   R4 侧**不得修改 harness**；改为在 `bench/hap_thw_model/README.md` 与 `cann_bench_submission/README.md`
   中**书面记录该偏离**，并附上 §5.8.4 的完整 `Block Num` 期望表，交 harness owner 更新。
   ⚠️ 本 Stage 1 **未能在 R4 目录内定位到 harness 侧的 gate 实现**
   （`grep -rn "Block Num" cann_bench_submission/ bench/hap_thw_model/gates_report.py` 无命中；
   `gates_report.py` 的 GATE 1–4 全部是 **t_hw 模型自洽性**检查，与 Block Num 无关）
   ⟹ **信息缺口，如实记录**：Stage 2/3 必须先定位该 gate 的真实实现位置，再决定是改还是记录偏离。
4. **新增 gate**（建议）：`sum(Block Num over the step's kernels) 必须 == 计划值之和`，
   以及 `K1A/K2gA 的 Block Num <= CUBE_CORE_NUM`（**定理 P2 的运行时验证**：split 后仍 ≤ 1 波）。
   这条 gate 能在运行时捕获「`S_cap` 公式被误改」⟹ 是 R-SGUARD 的第三重防护。

#### 15.3.2 `t_hw` 模型：**基本不需重推导，但有 3 处需说明**

**结论：`t_hw` 不需要重推导。** 依据（本 Stage 1 已读 `compute_r4_fused_sampling.py` 与 `r4_thw_common.py`）：

| 关注点（Orchestrator §8.3） | 裁定 | 论证 |
|---------------------------|------|------|
| **`r4_thw_common.py:219` 的 `_CASE_SEMANTIC_FIELDS` 含 `BV`** | ✅ **不受影响** | 该元组是「`cases.yaml` 与内联扩展表之间的**交叉校验字段**」（`:222-226` 文档），用于确保两个 case 来源的语义一致。rev3 **不改任何 case 的 `B`/`V`/`K_MAX`/`BV`/`need_min_p`/dtype** ⟹ 交叉校验照常通过 |
| **`r4_thw_common.py:234`** | ✅ **不受影响** | 该行是 `_row_from_yaml` 返回字典的 `"BV": int(attrs["BV"])`（本 Stage 1 已读 `:228-240`）—— 只是**读取** case 的 `BV`，不涉及 kernel 的块分解 |
| **`r4_thw_common.py:310-328` 的 `K1HW = n_blk_full*BV + TAIL_PAD`** | ⚠️ **语义仍正确，但需补一条注记** | `tail_pad_split()` 的 docstring 说 `K1HW` 是「the width the topk#1 sweep ACTUALLY touches in the fixed kernel」。rev3 下：**每行**被 topk#1 扫过的总宽度**仍然是** `n_blk_full*BV + TAIL_PAD`（K1A 覆盖全部块，只是分给了 S 个 split）⟹ **`K1HW` 的值不变**。⚠️ 但「单个 kernel 单次 launch 扫过的宽度」变成了 `maxblk_full*BV + (TAIL_PAD if S_TAIL 拥有尾块)`。**建议**：Stage 3 在 `tail_pad_split` 的 docstring 追加一句说明「rev3 split 路径下 `K1HW` 是**全行**宽度，分散在 `S` 个 split 上；per-launch 宽度见 `DESIGN.md §5.8.4` 的 per-split 块数」 |
| **`compute_r4_*.py:95-98` 硬编码 `K_MAX/BV == 1/16`** | ✅ **仍然有效** | 原文：*「they add a constant `4*K_MAX/BV = 4*(1/16) = 0.25` ops/elem (`K_MAX/BV` is 1/16 for BOTH specializations: 256/4096 and 128/2048)」*。rev3 **不改 `K_MAX` 也不改 `BV`**（§15.2）⟹ `1/16` 关系**继续成立** |

**⭐ 但有 3 处 ops/elem 记账在 rev3 下发生了（微小）变化，Stage 3 应评估是否更新**：

| 项 | 现行 | rev3 split 路径 | 对 `t_hw` 的影响 |
|---|---|---|---|
| **`logits` 的 GM 读** | K1 读 1 趟 | K1A + K1B 共读 **2 趟** | ⚠️ **`t_hw` 不变**！依据：`compute_r4_fused_sampling.py:26` 明写模型约定是 **`hap_lib.py:20`「Unique bytes only (no reload factors)」** + `README.md:13`「只计 unique bytes」⟹ **reload 被模型显式排除**。rev3 的 2 趟读属 reload ⟹ 按模型自身约定**不计入** `t_hw`。⚠️ 但这意味着 **`t_hw` 会略微高估 rev3 的可达效率**（真实 HBM 流量翻倍）⟹ **达成率口径需在报告中加注**（见下） |
| **跨 split 归并的额外 topk** | 每行 `n_blk_total` 次 `topk(2*K_MAX)` | 每行 `n_blk_total + S` 次（K1A 内 `n_blk_total` 次 + K1B 内 `S` 次） | `4*K_MAX/BV` 项变为 `4*K_MAX*(n_blk_total + S)/V = (4*K_MAX/BV)·(1 + S/n_blk_total)`。case7：`S/n_blk = 10/38 = 26.3%` ⟹ 该项从 `0.25` 增至 `0.316` ops/elem，而总量是 `45.25` ops/elem ⟹ **`t_hw` 增加 +0.15%**。case12：`2/8 = 25%` ⟹ 同量级。**可忽略，但应在 GATE 1 的自洽性重推导中体现**（`gates_report.py:24-41` 的 `run_gate1` 会独立重推导已发布的 `t_hw_us` 并报 `worst_rel_pct`；+0.15% 远低于其容差） |
| **K2g 的跨块折叠** | 每行 `n_blk` 次 2-op 折叠 | K2gA 每行 `n_blk` 次 + K2gB 每行 `S` 次 | `2/BV` 项变为 `2*(n_blk+S)/V`。case7：`+26%` on a `2/4096 = 0.00049` ops/elem term ⟹ **完全可忽略** |

**⟹ `t_hw` 的数值变化 < 0.2%，不需要重推导，GATE 1 的自洽性检查会继续 PASS。**

**⭐ 达成率口径如何保持可比（Orchestrator §8.3 的第三问）**：

`t_hw` 是**全核 roofline**（所有 40 个 AIV 满负荷时的理论最短时间）。达成率 = `t_hw / elapsed`。

| 口径 | rev2（现行） | rev3（split 路径） | 可比性 |
|------|-------------|-------------------|--------|
| `t_hw` | 全核 roofline，unique bytes | **同一公式、同一值**（变化 <0.2%，可视为不变） | ✅ **直接可比** |
| `elapsed` | 只用了 `2*b_num` 个 AIV（case7：4/40） | K1A/K2gA 用 `2*b_num*S` 个（case7：40/40），K1B/K2gB 仍用 `2*b_num` | ✅ **直接可比**（`elapsed` 一直是墙钟/msprof 实测，与用多少核无关） |
| **达成率** | case7 fused **0.67%**（极低，因为 36/40 个核闲置） | 预测 case7 fused **~2.6%**（提升 ~3.9×，与 §12.4 的 fused 3.89× 一致） | ✅ **可比且是 rev3 的主要 KPI** |
| ⚠️ **HBM 流量口径** | 1 趟 `logits` 读 | **2 趟** | ⚠️ **不可直接比**：若报告里出现「实测 HBM 带宽利用率」这类指标，rev3 会**看起来更差**（同样的墙钟时间搬了 2 倍字节）。**处置**：Stage 3 报告必须**显式注明**「rev3 split 路径的 K1 有 2 趟 `logits` 读；`t_hw` 按模型的 unique-bytes 约定只计 1 趟，故达成率对 split 路径**略偏乐观**（case7 的额外 608 KB/行 vs 总 2433 KB/行 ⟹ 若按真实字节重算 `t_hw`，`hbm_read` 分量从 `2.08` 增至 `~3.2` ps/elem，仍远低于 `vector 8.0` ps/elem ⟹ **瓶颈判定 VECTOR-BOUND 不变**，达成率修正幅度 <15%）」 |

#### 15.3.3 `gates_report.py` 的 GATE 1–4 影响逐项

| GATE | 内容（`gates_report.py:24-235`，本 Stage 1 已读） | rev3 影响 |
|------|--------------------------------------------------|----------|
| **GATE 1** 模型自洽性 | `C.run_gate1()` 独立重推导已发布的 910b2 `t_hw_us`（softmax 20/20 + arg_max 20/20），报 `worst_rel_pct` | ✅ **PASS 不变**（`t_hw` 变化 <0.2%，§15.3.2）。⚠️ 若 Stage 3 决定把 `+S` 次归并计入 ops/elem，需同步更新 `run_gate1` 的重推导式，否则 GATE 1 会 FAIL —— **这就是「有意识更新」而非「静默失败」** |
| **GATE 2** 瓶颈归因 | 36 个 (task, case) 对的 `t_vector` vs `t_hbm_read` vs `t_hbm_write` vs `t_scalar`，断言全 vector-bound | ⚠️ **需复核**：rev3 的 2 趟读使 `t_hbm_read` 分量翻倍。但 §15.3.2 已算出 case7 的 `hbm_read` 从 `2.08` → `~3.2` ps/elem，仍 `< vector 8.0` ⟹ **归因结论不变**。若 Stage 3 把 reload 计入模型，需重跑 GATE 2 确认 36/36 仍全 vector-bound |
| **GATE 3** 排序合理性 | `t_hw_us >= 1.0`；`t_hw == round(max(raw,1.0),4)`；`raw == max(4 comps)`；跨任务同 case 的 `fused >= mask` | ✅ **PASS 不变**（`t_hw` 的 4 个分量结构未变；`fused >= mask` 的关系未变——K2g 仍比 K2m 重） |
| **GATE 4** 算术强度 + 达成率解释 | `40.0 element-ops per 4 bytes read = 10.0 elem-op/byte`，vs 910b2 ridge point `3.125` | ⚠️ **需加注**：rev3 split 路径的**真实**算术强度是 `45.25 ops / 8 bytes = 5.66 elem-op/byte`（2 趟读），仍 `> 3.125` ridge point ⟹ **仍 vector-bound，结论不变**，但数值需在报告中区分「模型口径 10.0」与「rev3 实测口径 5.66」 |

### 15.4 ⭐ 已发布分数的可比性与评测可追溯性（Orchestrator §8.4）

**背景**：convention-A/B 分数（**56.47 / 56.04** 与 **93.24 / 95.32**）对应的是 **Stage 3 之前**的 kernel
（sha256 `bbe47fa4…`）；当前交付已是 **`19e51e8d…`**（Stage 3 round1 [#1]+[#2] 之后，
fused geomean speedup vs torch 从 6.7273 → **10.3463**、mask 从 11.0175 → **15.6706**）。
rev3 会再产生第三个 kernel 版本。

**⟹ 已发布分数与当前代码已经不可比一次了；rev3 会造成第二次不可比。必须有可追溯方案。**

**rev3 的处置（4 条，Stage 2/3 必须执行）**：

1. **建立 kernel 版本 ↔ 分数的显式对照表**，写入 `bench/hap_thw_model/README.md` 与
   `cann_bench_submission/README.md`：

   | 版本标签 | `r4_fused_sampling.py` sha256（前 8 位） | 内容 | fused geomean vs torch | mask geomean vs torch | convention-A/B 分数 |
   |---------|------------------------------------------|------|------------------------|----------------------|--------------------|
   | `v2-pre-s3` | `bbe47fa4…` | Stage 2 交付版（rev2 设计） | 6.7273 | 11.0175 | **56.47 / 56.04**（A）、**93.24 / 95.32**（B）—— ⚠️ 待 Stage 3 确认这两组分数究竟对应哪个 sha256 |
   | `v3-s3r1` | `19e51e8d…` | + round1 [#1] tie_m 向量化 + [#2] K2g argmax | **10.3463** | **15.6706** | 未发布 |
   | `v4-s3r2` | （round2 未改主文件，仅探针）| = `v3-s3r1` | 同上 | 同上 | 同上 |
   | ⭐ `v5-rev3` | **待 Stage 2 产出** | + split-V 双路分派 | **预测 1.229× of `v3-s3r1`**（§12.4，模型估计） | **预测 1.158× of `v3-s3r1`** | **必须重跑并重发布** |

   ⚠️ **信息缺口（如实记录）**：本 Stage 1 **无法确认** convention-A/B 的 `56.47/56.04` 与
   `93.24/95.32` 究竟对应 `bbe47fa4…` 还是 `19e51e8d…`（`.orchestrator_state.json` 为只读，
   本 Stage 1 未在其中检索到该映射；`bench/reports/` 下的 `cann_final_eval_*.json` 未逐个打开核对）。
   **Stage 3 必须先补齐这张表的「分数 ↔ sha256」映射，再重发布任何新分数。**

2. **重发布规则**：`v5-rev3` 的分数**不得替换**已发布分数，而应**并列发布**并标注 kernel sha256。
   理由：cann-bench 的分数是「(算子实现, 评测配置)」二元组的属性，实现变了就是另一个数据点。

3. **sha256 刷新清单**（Stage 2 交付前必须全部更新，rev2 §12.1 已要求，rev3 扩大范围）：
   - `examples/R4/r4_fused_sampling.py`（新增 4 kernel + host 分派）
   - `examples/R4/test_r4_fused_sampling.py`（L0 扩至 15 例 + Tier-2 bitwise 层）
   - ⭐ `examples/R4/cann_bench_submission/cann_bench/_r4_kernels.py`（**这是主 kernel 文件的提交副本**，
     本 Stage 1 已核实其 `:108/:371/:441` 三处 `b_num = T.ceildiv(B, VEC_NUM)` 与主文件同构
     ⟹ rev3 必须同步 4 个新 kernel + 分派逻辑，否则**提交版与开发版行为不一致**）
   - ⭐ `examples/R4/cann_bench_submission/cann_bench/r4_ops.py`（若其 host 侧调用逻辑需跟随分派）
   - ⭐ `examples/R4/cann_bench_submission/dist/cann_bench-1.0.0-py3-none-any.whl`（`build.sh` 重打包）
   - `examples/R4/bench/metadata_staging/*`（baseline / sha256 记录）
   - ❌ **`examples/R4/proto.yaml` 不变**（§15.1）
   - ❌ **`cases.yaml` / `cases.csv` 不变**（约束：禁止修改）

4. **cann-bench 36/36 `mismatch_count=0` 必须复跑**：rev3 的 bitwise 等价若成立，
   36/36 应**继续为 0**（且这是比 §9.2.2 Tier-2 更强的端到端验证，因为它跑的是**提交副本**
   `_r4_kernels.py` 而非开发文件）。**这是 rev3 的最终验收门禁之一。**

---

## 16. ⭐ 对 Orchestrator §3 预分析的逐条裁定（rev3 核心新增章节）

> 调度指令 §12 禁令 6：*「禁止把未验证推导写成已验证结论。第 3 节全部是 Orchestrator 的**未验证推理**，
> 你必须逐条标注『已核实（附文件:行号）』或『待 Stage 2 验证』」*，
> 且 §14 要求 *「凡与我结论不同处必须明确指出并给证据」*。本章逐条执行。

### 16.1 §3.1「朴素 split-merge 不可能 bitwise 等价」

**裁定：✅ 成立（Orchestrator 结论正确）。**
**证据强度：已核实（本 Stage 1 独立复核，含代码行号 + 舍入路径分析 + 偏差传播链）。**
详见 **§14.1**：
- 现行代码行号已核对：`r4_fused_sampling.py:206-222`（满块 11 步）、`:240-261`（尾块），
  其中 round2 #4 已删除的 UB→UB hop 位置（原 L211）已确认不存在 ✅
- 两处独立的 fp32 舍入分歧：① 单项 `exp(x)·exp(y) ≠ exp(x+y)`；② 求和结合树不同（串行 scan vs split 内 scan + split 间合并）
- 偏差传播链：`s`(1 ulp) → `q2` → `prefix` → **`L` 离散跳变 ±1** → `tau_p` 变成**另一个 logit 原值**
  （非 ulp 级！）→ `tau` → `{tok_out, valid}`（两个 **int32 精确匹配**门禁）+ `w_out` 边界元素离散跳变

**⭐ 与 Orchestrator 结论不同处：无。但 rev3 补充一条 Orchestrator 未指出的连带修正**：
Stage 3 round1 [#3]（`perf_tuning/optimization_log.md` 迭代 3「split-V 可行性分析（算法层：可行）」）
曾写下 *「online-softmax 跨 split 合并满足结合律: `m=max(m_c); s=Σ s_c·exp(m_c−m)`。`m` 精确无舍入」*。
该式**数学成立**，但**fp32 下 bitwise 不成立**（正是 §3.1 指出的问题）。
⟹ **round1 的这条「可行」判定在 bitwise 约束下应被推翻**，已记入 §10.4 第 1 条。
rev3 不采用该合并式，故不受影响。

### 16.2 §3.2「4-pass 分解可能满足 bitwise」+「Pass2/Pass4 落在哪里？」

#### 16.2.1 对 4-pass 分解本身的裁定

**裁定：⚠️ 部分成立 —— bitwise 论证正确，但「额外收益」的理由错误，且 rev3 不采用它。**

| Orchestrator 的论断 | rev3 裁定 | 证据 |
|--------------------|----------|------|
| Pass1（每块 `m_b = reduce_max(z_b)` → GM）跨 split 并行 | ✅ 数学/实现均可行 | 同 §14.2 的 `reduce_max` 论证 |
| Pass2（由 `m_b` 还原运行最大值序列 `m_new_b`）串行/行，标量 | ✅ **bitwise 正确**：`max` 精确无舍入 ⟹ 前缀最大值与顺序无关（M1，§14.5 步骤 3 同源） | 已核实（数学性质 + `T.tile.max` 是 fp32 精确比较） |
| Pass3（用**正确的** `m_new_b` 算 `s_b = reduce_sum(exp(z_b − m_new_b))`）跨 split 并行 | ✅ **bitwise 正确**：`m_new_b` 位模式与现行串行执行到该块时的 `m_run` 完全相同 ⟹ `s_b` 逐位相同 | 已核实（`reduce_sum` 对同一 `[1,BV]` 输入是静态归约树，`BV` 编译期常量） |
| Pass4（按原块序执行 `m_new=max(m_run,m_b); s_run*=exp(m_run−m_new); m_run=m_new; s_run+=s_b`）串行/行，标量 | ✅ **bitwise 正确**：运算序列与现行 L213-222 **逐条对应** | 已核实（逐行对照） |
| 「**为何可能 bitwise 成立**」的整段论证 | ✅ **成立** | 同上 |
| ⭐「**额外收益：完全绕开『Developer 模式无跨核归约原语』的障碍 —— 不需要任何跨核归约原语，只需 GM workspace + 两个廉价串行标量 pass**」 | 🔴 **理由需修正** | 4-pass 并没有「绕开」障碍，而是**把障碍转化成了 launch 代价**：Pass2/Pass4 是**行内串行**的，若要与 Pass1/Pass3 同 kernel 就必须 grid 同步（**§2.5 已核实：Developer 模式不可得**）⟹ 只能作**独立 kernel**（选项 A）⟹ **+2 次 launch**。所以「绕开」的实际含义是「用 2 次额外 launch 换取不需要 grid 同步」。rev3 的选项 D 是**真正**绕开：串行部分与跨 split 归并合并进同一个 `grid=b_num` 的 kernel，且**串行标量重放根本不需要存在** |
| 代价 ①「GM 读 1 趟 → 2 趟」 | ✅ 成立，rev3 同样承担 | §4.4、§5.5、R-BW |
| 代价 ②「kernel launch 2 次 → 3+ 次（小 case 是 launch-bound：case13 fused 全程仅 23.5us）」 | ✅ **量级成立，rev3 精确化** | 4-pass 会是 **2 → 5 次**（Pass1/2/3/4 + K2）；rev3 是 **2 → 3 次（mask）/ 4 次（fused）**。case13 fused 23.5us 已用 `ceiling.json` 复核：`k1_us_fused=12.16 + k2_us_fused=11.32 = 23.48us` ✅。⚠️ **但 rev3 的 guard 会把 case13 挡在现行路径**（`n_blk_full=1 < 2`）⟹ 该风险对 rev3 不生效 |
| 代价 ③「workspace 显存 `B×n_blk×4B×2`（case15 ≈ 18.9KB）」 | ⚠️ **公式不适用于 rev3** | 那是存 `m_b`/`s_b` 的公式。rev3 不切 softmax ⟹ 不需要 `m_b`/`s_b` ⟹ 公式是 `4·B·S·(K_MAX+2)`，最大 **40.3 KB（case7）**；case15 走现行路径**不分配** workspace。详见 §4.8 + §10.4 第 5 条 |

#### 16.2.2 ⭐ 对「Pass2/Pass4 落在哪里？」的回答：**选 D（其他可行形态）—— 二者都不需要存在**

**A / B / C 的逐项裁定**：

| 选项 | 裁定 | 理由（含证据） |
|------|------|---------------|
| **A**：Pass2/Pass4 各作独立小 kernel（+2 次 launch） | ⚠️ **可行但非最优** | 可行（不需要任何 grid 同步）。代价：K1 从 1 kernel 变 **5 kernel**（Pass1/2/3/4 + K2）。对小 case 是致命的（`a_launch` 未实测，R-LAUNCH）。且 Pass2/Pass4 的串行标量链本身有成本：Pass4 需 `n_blk × 6` 个标量 op，而**每个标量 op 在生成码里带 `PipeBarrier`**（`K1_case15_real.cc` 的 `tau_p` 循环实测 3 × `PIPE_ALL`/迭代 ⟹ 28 ns/drain）⟹ case7 的 Pass4 ≈ `38 × 6 × ~28ns ≈ 6.4 us`（**估计，未实测**），并非 Orchestrator 所说的「廉价」到可忽略 |
| **B**：并入 Pass3 的 kernel，由每行某一个 block 承担（需 grid 级同步/semaphore） | 🔴 **不可行** | **§2.5 已逐条核实源码**：`T.barrier_all` 是 PIPE 级（`ascend.py:275-286`）；`tl.ascend_auto_cross_core_sync` 是 AIC↔AIV（`ascend_combinecv.cc:36-58`）；`T.tile.atomic_add` 顺序不定且只有 add（`ascend_tile.py:215/313-346`）；前端**无任何** grid barrier/semaphore。⚠️ 即使自造「atomic 计数器 + 循环读 GM 自旋」，还有**波间死锁**：若生产者 block 落在第 2 波而消费者在第 1 波自旋 ⟹ 永久挂死 |
| **C**：并入 Pass1（仍需同步） | 🔴 **不可行** | 同 B（Pass1 结束时其他 block 的 `m_b` 尚未写出） |
| ⭐ **D（rev3 采用）**：**让 Pass2/Pass4 不需要存在** | ✅ **可行，且已编译实证** | **不切分 online-softmax 扫描** ⟹ 没有「局部 `m_j`/`s_j` 需要事后补偿」⟹ **不需要 Pass2 还原前缀最大值，也不需要 Pass4 重放标量链**。softmax 扫描完整地留在 K1B（`grid = b_num`，每行串行，块序与现行完全相同）⟹ `m/s` 的 bitwise 是**构造性的**（§14.2），不是「靠精确重放论证出来的」。**唯一被切分的是 Sort/top-k 侧**，其跨 split 归并用 M2（多重集确定性）论证。**K1 只需 2 个 kernel（K1A + K1B），不是 5 个** |

**为什么 D 优于 A（定量对比，case7，`a_launch` 记为 `L`）**：

| 方案 | K1 kernel 数 | 串行部分 | 并行部分 | K1 预测时长 |
|------|-------------|---------|---------|------------|
| 现行 | 1 | `F + n_blk·(p+t) = 21.6 + 38×10.7 = 428 us` | — | **419.5 us**（实测 419.47） |
| **A（4-pass）** | **4** | Pass2（`38` 个标量 max，~1 us 估计）+ Pass4（`38×6` 标量 + epilogue `F=21.6`，~28 us 估计）≈ **29 us** | Pass1（`38/10 × ~0.6 us` ≈ 2.3 us）+ Pass3（`4 × (1.2 + 8.9) ≈ 40.4 us`） | ≈ **69 us + 4L** ⟹ 理论上限 ~6× |
| ⭐ **D（rev3）** | **2** | K1B：`F + n_blk·p + S·t_mrg = 21.6 + 68.5 + 9.7 = 99.8 us` | K1A：`maxblk·t = 4 × 8.9 = 35.6 us` | ≈ **135 us + 2L** ⟹ **~3.1×** |

**⟹ A 在理论上更快（~6× vs ~3.1×），但 rev3 仍选 D**，理由（如实权衡）：
1. **bitwise 风险等级不同**：D 的 `m/s` bitwise 是**构造性的**（代码逐条未改，零浮点论证）；
   A 的 `m/s` bitwise 依赖 **Pass2 精确还原 + Pass4 精确重放**两个论证，且 Pass4 必须**逐条复刻**
   现行 L213-222 的 6 步标量运算顺序 —— 任何一处偏差（例如把 `T.tile.max(m_run,m_run,m_blk)` 与
   `T.tile.max(m_old,m_run,m_blk)` 的先后写反）都会静默产生 ulp 偏差。**约束 2 是硬约束，
   降低证明负担的价值高于 ~2× 的额外性能。**
2. **launch 数 4 vs 2**：`a_launch` 未实测（R-LAUNCH，高严重度）。A 方案把 4 个 `L` 押在一个未知量上。
3. **代码复杂度与调试难度**：A 需要 5 个 kernel（含 K2）+ 2 个额外 workspace（`m_b`/`s_b` 数组）
   + Pass4 的标量重放逻辑；D 只需 4 个 kernel + 1 个 workspace。R-DEBUG。
4. **A 的额外收益部分被 F 抵消**：`F = 21.63 us`（epilogue）在 A 与 D 中**都必须串行付一次**，
   它是 case7 K1 的 5.2%，但对 case3（K1=64.4us）是 **33.6%** ⟹ A 相对 D 的优势在小 case 上被大幅压缩。
5. **D 是 A 的严格子集，可增量升级**：若 D 落地后 Stage 3 实测发现 `a_launch` 很小（<2 us）
   且 K1B 的串行 softmax（`n_blk·p`）成为新瓶颈，**可以在 D 的基础上再拆分 K1B**（把 softmax 侧
   改成 Pass1/2/3/4），复用 D 已有的 workspace 与分派框架 ⟹ **D 不是死路，A 可以作为 D 的后续增量**。
   反之从 A 退回 D 则是重写。

**📌 结论：rev3 选 D。A 记录为「已评估、理论更快、但因 bitwise 证明负担与 `a_launch` 未知而未采用，
可作为 D 落地后的增量升级路径」。**

### 16.3 §3.3「top-256 归并链 bitwise 安全」

**裁定：✅ 成立（Orchestrator 结论正确），但有一处必须实测的前提。**
**证据强度：数学部分已核实；硬件 `topk` 确定性部分「待 Stage 2 验证」。**

| Orchestrator 的论断 | rev3 裁定 | 详见 |
|--------------------|----------|------|
| `run_v` **只存值不存索引** | ✅ 已核实 | `gather_mask(l_val, l_dst, "P0101")` 取偶数位 = 值（`ascend_tile.py:608-637` docstring +`test_..._topk.py:169-170` 的 `b[0::2]` 交叉验证，rev2 §3.2 已三重核实） | §14.3 步骤 3 |
| `topk` 输出有序 | ✅ 已核实 | `common.h` 的 `Sort32` + `MrgSort` 归并树（本 Stage 1 已读 `:1715-1790`）；且现行代码依赖其降序性（`tau_k = run_v[k_i-1]`）并已被 36/36 cann-bench 验证 | §14.3 |
| 按 split-id 有序归约得**相同多重集** → 相同有序数组 ⟹ bitwise 相同 | ✅ **数学上成立** | §14.3 的引理 L1（`top_K(A∪B) = top_K(top_K(A) ∪ top_K(B))`，含完整证明）+ 步骤 4（降序值数组由多重集唯一确定） |
| `n_gt`/`n_eq` 只在 `run_v` 的 256 个元素上统计（round1 [#1] 已向量化为 `compare+select+reduce_sum`），**不扫全 V** ⟹ `run_v` 相同即安全 | ✅ **已核实** | 逐行核对 `r4_fused_sampling.py:316-322`（`compare(tie_mask, run_v, tie_tkbc, "GT"/"EQ")` + `select` + `reduce_sum`，全部在 `[K_MAX]` 上） | §14.4 |
| `tau_k`/`tau_p`/`L` 全由 `run_v` 派生 ⟹ 安全 | ✅ **已核实** | 逐行核对 L273-307（`q2` ← `run_v`,`m_run`,`s_run`；`tau_k` ← `run_v[k_i-1]`；`tau_p`/`prefix` ← `run_v`,`q2`,`p_i`；`okp` ← `prefix`,`p_i`；`tau_u` ← `m_run`,`mu_i`，**与 `run_v` 无关**） | §14.4 |
| ⭐「**请复核**：`topk` 对相同多重集是否保证产出完全相同的有序序列（含 NaN / ±inf / 大量重复值）。若不能保证，设计中必须处理」 | ⚠️ **部分先验成立，部分是 rev3 的首要实测项** | 见下表 |

**⭐ 对「`topk` 确定性」的逐边界裁定**（这是 Orchestrator 明确要求复核的点）：

| 边界 | 先验裁定 | 论证 | 是否需实测 |
|------|---------|------|-----------|
| **一般有限值（无重复）** | ✅ 安全 | (value, index) 对全序唯一（index 由 `ArithProgression` 生成、互异，`common.h:1723` `AscendC::ArithProgression<T>(dst, T(0), T(1), alignedCount)`）⟹ 排序结果唯一 ⟹ 值序列唯一 | 由 V-S1 一并覆盖 |
| **大量重复值（含 all-equal 行）** | ✅ **安全**（值序列层面） | 重复值**不可区分** ⟹ 降序值数组唯一，与排序稳定性无关。⚠️ index 序列会依赖稳定性，但 `run_v` **丢弃 index** ⟹ 不可见。⚠️ all-equal 行会触发 `tie_ok=0`（`n_eq = K_MAX > 1`）⟹ `valid=0` ⟹ 属既有 D-TIE 降级路径，rev3 未改变 | 由 Stage 2 的 L2/Boundary（现有套件已有全等行用例）覆盖 |
| **`-inf`（合法输入 + `run_v` 初值）** | ✅ **安全**，且已严格论证 | §14.3 (iii)：`-inf` 在 fp32 全序中良定义、排最后；「`-inf` 初值被挤出」的等价性已用多重集论证（两侧都是 `top_K({−inf}×K ∪ M)`）；**整行全 `-inf` 的退化情形两侧都得 `run_v = [−inf]*K`** | ⭐ **V-S4**（构造「某 split 全 `-inf`」）—— 先验安全但未实测 |
| **`+inf`** | ✅ 安全（同 `-inf`，全序良定义、排最前） | 同上。⚠️ 但 `+inf` 会使 `s = +inf` ⟹ `q2 = NaN`，**这在现行 kernel 上同样发生** ⟹ 非 rev3 回归，属既有 L2 负向域 | 不需（域外） |
| **NaN** | 🔴 **M2 不成立** | NaN 的所有比较为 False ⟹ 全序失效 ⟹ 排序结果**依赖输入顺序** ⟹ 归并树形状不同 ⟹ 结果可能不同。**但 NaN 是契约外输入**（rev2 §9.3：上游 `sanitize_nan_logits` 已清洗，NaN 属 **L2 负向域**），且**现行 kernel 在 NaN 下同样未定义** ⟹ **rev3 不引入新的 NaN 回归，也不声称支持 NaN** | 不需（域外）；但 §10.1 R-NAN 已明确记录该前提 |
| **`K == actual_num`（全取，例如 `TAIL=256, K_MAX=256` 的 case1）** | ✅ 安全 | `topk` 要求 `K <= actual_num`（`ascend_tile.py:653-655` 的 `ValueError` 校验）；`K == actual_num` 时输出就是全排序 ⟹ 唯一 | case1 走现行路径；L0-B1（`TAIL=257 > K_MAX=256`）覆盖近边界 |
| ⭐ **`tl::ascend::TopK` 的实现是否严格等价于「精确 top-K + 精确降序排序」** | 🔴 **只有硬件能回答** | 本 Stage 1 读了 `common.h` 的归并树结构（`Sort32` → `MrgSort` stages `[(128,32),(32,8),(8,2),(2,1)]`），确认它排 (value,index) 对。**但**：① `MrgSort` 的 `ifExhaustedSuspension = false` 与 `validBit = (1<<groupCount)-1` 的精确语义未查 AscendC 官方文档；② `Sort32`/`MrgSort` 是**硬件指令**，其对相等值与 `-inf` 的行为无源码可查；③ **rev3 的归并树形状与现行不同**（现行 38 次 `topk(512)`；rev3 是每 split `per[s]` 次 + K1B `S` 次）——若硬件排序对输入顺序敏感，形状不同 ⟹ 结果可能不同 | ⭐⭐ **V-S1（最高优先级）**：D1 用例 `torch.equal` 于位模式 |
| **若 V-S1 失败的退路** | — | ① 先排查是否是 `-inf` 初值挤出问题；② 若是相等值的排序不稳定 ⟹ **改为「K1B 把 S 份 `run_v_s` 拼成 `[S*K_MAX]` 后做单次 `topk(K_MAX, S*K_MAX)`」**（单次排序 ⟹ 树形状唯一；`S*K_MAX % 32 == 0` ✅；UB 需重算：`S=10, K_MAX=256 ⟹ [2560]` fp32 = 10240 B + arena `4×2560×4 = 40960 B` ⟹ K1B UB 从 89444 增至约 120 KB，仍 < 196352 ✅ **本 Stage 1 已预算**）；③ 若仍失败 ⟹ **返回 `[DESIGN_ERROR]`**（约束 2 无法满足） | §14.6 U1 |

### 16.4 §3.4「K2g（#1c）split 分析」

**裁定：✅ 全部成立（Orchestrator 的 3 条「有利」+ 1 条「依赖」+ 1 条「风险」逐条正确）。**
**证据强度：已核实（含 M3 的完整证明）。** 详见 **§14.5**。

| Orchestrator 的论断 | rev3 裁定 | 补充 / 证据 |
|--------------------|----------|------------|
| hash 是 `j` 的纯函数 ⟹ 每 split 可独立生成自己 V 区间的 `G_j`，无需跨 split 通信 | ✅ **成立** | `x_ij = (seed_i·PRIME_W + j) mod 2^32` + `lowbias32` + `u` + `G` **全部逐元素**。⚠️ **前提是 `b` 必须是全局块号（N11/E-S3）**：`arith_progression(state_ub, seedP_v[0] + b*BV, 1, BV2)` 与 `best_idx = blk_argmax + b*BV`，其中 `b = sid*NBS + lb`。误用 `lb` 会导致每 split 重复同一段噪声 |
| round1 [#2] 的 `reduce_max + compare(EQ) + select + reduce_min` **天然可跨 split 合并**（(max, min-index) 对的合并可结合） | ✅ **成立**，且 rev3 给出了完整证明 | §14.5 引理 L2 + 步骤 1-4（含「`s > s*` 的 index 严格更大」的区间不重叠论证，这是「可结合」的**关键前提**：split 必须是**连续块区间**，不能是交错划分） |
| K2g 需要 `tau`（来自 K1）⟹ 天然满足（本就是两个 kernel） | ✅ **成立** | rev3 launch 顺序 K1A → K1B → K2gA → K2gB ⟹ K2gA 读 `tau_out`（K1B 输出），依赖方向与强度均未变 |
| **bitwise 风险**：tie-break 是「严格 GT + 最小 index」；跨 split 合并时若两 split 最大值相等，必须取**全局最小 index** ⟹ 合并顺序与 index 比较须精确设计 | ✅ **风险识别正确，rev3 的设计已精确处理** | **设计要点三条**：① K2gB 用 **`T.serial(S)` 严格 split-id 升序**（禁止 `T.Parallel`，E-S4）；② 用**严格 GT**（`bv[0] > best_val[0]`，不是 `>=`）⟹ 相等时**不更新** ⟹ 保留更早（更小 index）的 split；③ split 按**连续块区间**划分 ⟹ `s` 越大 index 越大 ⟹ 「更早的 split」= 「更小的 index」。**∎ 已证明得到全局最小 index（§14.5 步骤 4）** |
| ⭐「请给出 bitwise 论证」 | ✅ **已给出**（§14.5，含 6 个步骤 + `-inf` 退化情形 + `cast` 步骤） | **残留不确定性 3 条**（§14.5 末尾）：**(a) `tau` 必须逐位相同 —— 这继承 §14.3 M2 的不确定性，是 rev3 最尖锐的风险点**（`tau` 差 1 ulp ⟹ `compare(mask, z_ub, t_v[0,0], "GE")` 在阈值边界的单个元素翻转 ⟹ `tok_out` 跳到另一个 index）；(b) `reduce_max/reduce_min` 的归约树顺序无关性（先验成立：`max/min` 精确无舍入；且 round1 [#2] 已被 36/36 + stress 8/8 验证）；(c) `+inf` 的 `G_ij`（先验安全，未实测） |

**⭐ 与 Orchestrator 结论不同处：无。** rev3 补充的是**证明**（Orchestrator 只给了判断）
与**一条 Orchestrator 未指出的依赖链**：K2g 的 bitwise **传递性地依赖** K1 的 `tau` bitwise
⟹ **U1（`topk` 确定性）失败会同时击穿 `tau_out`、`valid`、`w_out`、`tok_out` 四个输出**，
而不只是 `tau_out`。这使 V-S1 成为 rev3 的**单点最高风险**。

### 16.5 §4「精度红线」的逐条复核

| Orchestrator 论断 | rev3 裁定 | 证据 |
|------------------|----------|------|
| 缺陷 A：`topk` 的 `count` 非 32 倍数 → `Sort32` 消费 `ceil(count/32)*32` → masked `AscendC::Duplicate` 在 dav_c220 不生效 → 未初始化 UB 槽位被当真实候选（曾致 `tau_out` 返回 2.7e+19~4.5e+37） | ✅ **已核实** | 本 Stage 1 读了**仓库与 wheel 两处** `common.h`：仓库 `src/tl_templates/ascend/common.h:1715-1758`、wheel `…/site-packages/tilelang/src/tl_templates/ascend/common.h:1663-1707`。`padCount = alignedCount − actualCount`；`if (padCount > 0)` → `inBlockOffset == 0` 的简单 `Duplicate` 分支 **vs** `masks[2]` 的 **masked** `AscendC::Duplicate(src[alignedActual], negInf, masks, (uint8_t)1, (uint16_t)1, (uint8_t)0)` 分支 —— 后者正是 rev2 记录的不生效路径 |
| 缺陷 B：UB→UB `T.copy` 在 count 非 32B 倍数时静默截断到 8 的倍数（313→{312}、261→{256..260}、511→{504..510}、257→{256}） | ✅ **采信**（rev2 Stage 2 的实测结论，本 Stage 1 未重跑） | `history_version/design_rev2.md` §10.3 + `debug_log.md`；rev3 的处置见 §4.9 |
| 现行规避（尾块 `if TAIL > 0:` 区段，**逐字节保留禁改**）：`TAIL_PAD=ceil(TAIL/32)*32`；经 `ztp=[1,TAIL_PAD]` 从 GM 以 `pad_value=-inf` 载入；hop 恰好 `TAIL_PAD`；`topk(...,TAIL_PAD)` 使 `padCount==0` | ✅ **rev3 逐字节保留** | §3.3 K1A 伪代码的 `if TAIL > 0:` 区段与现行 `r4_fused_sampling.py:240-269` 的 top-k 部分**逐条相同**；§6.4 表格明确「代码逐条相同」；编译实证生成码 `copy_gm_to_ub<float, 512, 1>(ztp[0], logits[…151552], 608256, 1, 512, -CUDART_INF_F)` 与现行同形态 |
| `padCount==0` 由 `TAIL_PAD%32==0` **结构性**保证，与 `BV` 无关（已核实 `ascend_tile.py:646` + `common.h:1712-1755`） | ✅ **已核实，且 rev3 独立复核成立** | `ascend_tile.py:647`（`repeatTimes = (max_actual_num + 31) // 32`；⚠️ Orchestrator 写的 `:646` 差 1 行，实际在 `:647`）+ `common.h:1715-1717`（仓库）/`:1663-1665`（wheel）。**rev3 补强**：定理 P1 证明 split-V 下 `actual_num` 的取值集合**与现行完全相同** ⟹ 保证不仅「与 BV 无关」，还「**与 S 无关**」 |
| 新风险：split-V 改变每 split 处理的 V 区间长度 ⟹ 必须重新证明每 split 的尾块也满足 `padCount==0` 与 32B 对齐 | ✅ **风险识别正确，rev3 已完成重新证明** | §4.9：先验论证（定理 P1）+ **10/10 split-path case 的编译实证**（全部 TopK 调用点 `padCount=0`、全部 `copy_ub_to_ub` 计数 `%32B==0`）。⚠️ **关键洞察**：rev3 **不改变任何块的宽度**，只改变「哪个核处理哪个块」⟹ `actual_num` 集合不变 ⟹ 风险实际比 Orchestrator 预想的小 |
| 建议**强制 split 边界按 BV 对齐** | ✅ **已采纳**，升格为设计决策 **D-SPLIT-BVALIGN** + 定理 P1 | §3.4、§5.8.3 |
| 已核实的 API 陷阱：`topk` 的 `max_actual_num = Σ src.shape`（求和非求积，`ascend_tile.py:635-645`）⟹ rank-2 `[1,BV]` 源给 `1+BV=4097` → `repeatTimes=129` → 覆盖 4128>4096，静默读越界 32 元素。topk 源必须 rank-1 `[BV]` | ✅ **已核实** | 本 Stage 1 逐行读了 `ascend_tile.py:635-647`：`max_actual_num = 0` → `for dim in src.shape: … max_actual_num += dim_simplified.value` → `repeatTimes = (max_actual_num+31)//32`。**确为求和**。⟹ rev3 K1A 的 `z_ub` 是 rank-1 `[BV]`、`ztt` 是 rank-1 `[TAIL_PAD]`（N2） |
| `T.reduce_*` 对 rank-1 源要求目的端传 **BufferRegion**（`m_blk[0,0:1]`）而非 Buffer，否则 `reduce_ascend.py:130-172` 的 `expected_shapes=[[],[1]]` 校验 `raise ValueError` | ✅ **已核实** | 本 Stage 1 读了 `reduce_ascend.py:133-176` 的 `_validate_reduce_out_shape`：`expected_shapes = [reduced_shape]`（+`keepdim_shape` 若不同）；`if out_is_region and any(_shape_can_embed_expected_shape(out_extent, expected_shape) …)` ⟹ **region 形式有额外宽容路径**。⚠️ 对 rank-1 源 + `dim=-1`，`reduced_shape = []`（空）、`keepdim_shape = [1]` ⟹ 与 Orchestrator 写的 `[[],[1]]` 一致 ✅。rev3 K1B 保留 `m_blk[0, 0:1]` 写法 |

### 16.6 §5「UB 预算」的复核

| Orchestrator 论断 | rev3 裁定 |
|------------------|----------|
| 上限 **196352 B** | ✅ 已核实（生成源 `InitBuffer(ascend_ub, 196352)`；rev3 §4.5 的实测法直接解析该值） |
| 实测（BV=4096，18 case 最差）：**K1 = 171620 B（87.40%，余 24732）**、K2g = 148452 B（75.6%，余 47900）、K2m = 98912 B（50.4%，余 97440） | ✅ **采信**（round2 实测）。⚠️ rev3 补充：本 Stage 1 对 **case7** 单独实测现行 K1 = **140900 B**（`ub_usage` 解析），小于 171620 ⟹ **171620 是 18 case 的最差值**（应为 `TAIL_PAD` 更大的 case，如 case17 的 832 或 case6 的 2048），Orchestrator 的「18 case 最差」表述正确 |
| `§4.5` 已证实 4 处错误 + 完全无 K2g 账目；K2g 实测公式 `36·BV + BV/8 + 484` | ✅ **采信并落实**：rev3 §4.5 **废弃**该模型，全部改为实测值。⚠️ rev3 交叉验证 K2g 公式：`36×4096 + 4096/8 + 484 = 147456 + 512 + 484 = 148452` ✅ 与 Orchestrator 数字完全一致；rev3 实测 K2gA = **148420**（差 32 B，来自 workspace 写缓冲的标量对齐差异）⟹ **公式与实测互相印证** |
| 新 DESIGN 须给出正确 UB 账目（复用 `bv_ub_sweep.py` 的实测法：解析 `ascend_ub.GetWithOffset<T>(n,off)` 取 `max(off+n·sizeof(T))`） | ✅ **已执行**：§4.5 用**完全相同的方法**（本 Stage 1 自己实现了 `ub_usage()`，正则与 `ub_probe.py:_PAT` 一致），对 4 个新 kernel × 12 case 做了 48 次编译实测 |
| `BV=6144` 编译期硬失败（`LOG(FATAL) … tmp_ub required 49152, available 48704`）；`BV∈[3872,4064]` 因 C3 不可行 | ✅ **采信并沿用**（§4.5 末尾）。⚠️ rev3 补充一个**利好**：K1A/K1B 的 UB 余量分别增至 89856 / 106908 B ⟹ 若未来要重探 `BV=6144`，**在 rev3 的拆分结构下 K1 侧可能变得可行**（`tmp_ub` 的 49152 B 需求 vs K1A 的 89856 B 余量）。但这属未来工作，且 **`BV` 是契约属性（§15.2），rev3 不动它** |

### 16.7 §6「安全网依赖」的复核

**裁定：✅ 已核实。** `src/transform/ascend_memory_planning.cc:598` 为
`bool check_overflow = false; // reserve memory overflow check`，`:606` 为
`if (offset + buf_size > memory_limits_[scope] && check_overflow)` ⟹ **该路径静默溢出**。
rev3 的落实：§4.7 / §7.3 强制 4 个新 kernel 全部传 `pass_configs=R4_PASS_CONFIGS`
（含 `tl.ascend_memory_planning=True`），并把它列为 §9.2.5 的**静态断言 ①**（Stage 2 必须断言）
+ §10.1 **R-MEM**（高严重度）。

### 16.8 §9「已知门禁失明」的复核与落实

**裁定：✅ 问题识别正确，rev3 的 L0 计划已逐项修掉。**

Orchestrator 指出：round2 #2 发现**现有测试套件无任何配置 `n_blk_full > 2`**
（STAB 最大 2、ADV 最大 1、L2 特殊值全是 `B=4,V=4096,BV=4096`→1）
⟹ 任何依赖多块迭代次数的逻辑都无法被现有套件验证。
**本 Stage 1 已用 `STAB_CASES`（`test_r4_fused_sampling.py:1136-1160`）与 `ADV_CASES`（`:1149-1155`）复核 ✅**：
`STAB_CASES` 的 8 个配置 `(B,V,BV)` = `(7,4409,4096)→n_blk_full=1`、`(1,4096,4096)→1`、
`(3,4353,4096)→1`、`(9,4607,4096)→1`、`(7,4409,2048)→2`、`(5,10240,4096)→2` ⟹ **最大 2** ✅；
`ADV_CASES` = `(2,4409)→1`、`(3,4353)→1`、`(2,4607)→1`、`(5,261)→0` ⟹ **最大 1** ✅。
**Orchestrator 的数字准确。**

| Orchestrator 要求 | rev3 L0 计划的落实 | 用例 |
|------------------|-------------------|------|
| `n_blk_full` 显著 >2 的配置（如 V=152064/BV=4096 → 37） | ✅ 3 个用例 `n_blk_full = 37`，2 个 = 8，3 个 = 4~5 | **A2**(37,S=10)、**B5**(37,S=2)、A1(8)、A4(8)、F1(5)、A3(4)、B1-B4(5) |
| **`b_num<20` 与 `b_num>=20` 两侧的分派边界用例**（B=38/39/40/41 附近） | ✅ 3 个用例，且**额外提供了比用户建议更锐利的边界**：用户建议的 `B=38/39/40/41`（`b_num=19/20/20/21`）在**本设计的收紧判据下全部走现行路径**（`S_cap = 20//19 = 1`）⟹ 那 4 个点**无判别力**。真正的锐利边界是 **`b_num=10 → 11`（`B=20 → 22`）**，即 C1/C2。同时保留 C3（`B=40`, `b_num=20`）作为**用户字面边界**的验证点 | **C1**(`B=20`,`b_num=10`,split)、**C2**(`B=22`,`b_num=11`,现行)、**C3**(`B=40`,`b_num=20`,现行) |
| **双路一致性用例**：同一输入分别走 split 与现行路径，断言**逐位相同** | ✅ 专设 D 组，且**升级为贯穿全 L0 的 Tier-2 判据**（不只是 1 个用例，而是**所有 split-path 用例都要做**） | **D1**（专设，5 输出 × 2 任务）+ **全部 A/B/C1/F 组的 Tier-2** |
| split 数 S 的边界：S=1（退化）、S=最大值、S 使某 split 只有 1 块、S 使某 split 为空（若可能） | ✅ 4 项全覆盖。「S 使某 split 为空」的答案是**结构上不可能**（定理 P3），故 F1 是**验证该保证生效**的回归哨兵 | **E1**(S=1 退化,白盒)、**A2**(S=10 最大,grid 恰 20)、**A3**(NBS=1,每 split 1 块)、**A4**(S=8,NBS=1)、**F1**(朴素 S=4 会产生空 split → 收缩为 3) |
| 保留既有 `TAIL%32!=0` 强制覆盖（TAIL=313/257/511/261/825） | ✅ **5/5 逐项对应，一个不缺**，且全部在 **split 路径**下（现有 STAB/ADV 全走现行路径 ⟹ 这是 rev3 新增的覆盖面） | **B1**(257)、**B4**(261)、**B2**(313)、**B3**(511)、**B5**(825) |
| 保留 UB 投毒层 | ✅ 列为 **V-S5**（§10.2）+ §4.9 末尾的强制要求 + §9.2.4 表格的 `--level stress` 行 | V-S5 |
| ⭐ rev3 **额外发现**的失明点 | 🔴 **现有 `STAB_CASES` 对 split 路径零覆盖**：其 8 个配置的 `n_blk_full ∈ {0,1,1,1,2,2}` ⟹ 全部被 `D-SPLIT-GUARD` 否决 ⟹ **现有 stability gate 跑 20 reps 也不会碰 split 路径一行代码**。同样 `_run_tail_adversary` 的 4 个 `ADV_CASES` 也全部走现行路径 | §9.2.4 已要求 Stage 2 **必须新增 split 路径的 STAB 配置**（至少 A2/B5/F1 × 20 reps）与 **split 路径的 tail-adversary**（用 B1-B5 的 `V`）；V-S6 |

### 16.9 §2「量化动机」的复核

| Orchestrator 论断 | rev3 裁定 | 独立复核结果 |
|------------------|----------|-------------|
| 波数定律 `K1_us ≈ ceil(ceil(B/2)/20) × scan(V)`；20 = cube_core_num，VEC_NUM=2 | ✅ **成立，且 rev3 给出了更精确的形式与标定常数** | §12.2：`K1_us = ceil(b_num/20) × [F(K_MAX) + n_blk_row×(p(BV)+t(BV,K_MAX))]`，`F(256)=21.63us`、`p(4096)=1.802us/块`、`t(4096,256)=8.902us/块`。**对 6 个 probe case 的拟合误差：5 个 ≤0.7%，1 个（case8, K_MAX=128）−15.0%** |
| 同 V=152064：case7(1wave) 451.3us vs case15(2waves) 911.6us → 比值 2.02 对 wave 比 2.00 | ✅ **成立** | ⚠️ 本 Stage 1 从 `probe_perf.json` 读到的是 case7 `real_us=419.47`、case15 `835.72`（比值 **2.00**），从 `ceiling.json` 读到 case7 `k1_us_mask=425.50`、case15 `856.28`（比值 **2.01**）。Orchestrator 的 `451.3/911.6`（比值 2.02）应来自另一次 msprof 运行。**三组数据比值都在 2.00–2.02 ⟹ 结论稳健**，绝对值有 ~5% 的运行间波动（属 msprof 正常噪声） |
| 同 V=32768：case12(1wave) 120.3us vs case16(4waves) 532.9us → 4.43 对 4.00 | ✅ **方向成立，但 4.43 ≠ 4.00 的偏差 rev3 已解释** | §12.2 模型：case12 = `1×(21.63 + 8×10.70) = 107.3us`；case16 = `4×(21.63 + 8×10.70) = 429.0us`。**比值恰为 4.00**（因为 `b_num=8` 与 `64` 的每核行数比是 4）。实测 `ceiling.json`：case12 `k1_us_mask=95.12`、case16 `433.02` ⟹ 比值 **4.55**。Orchestrator 的 4.43 与此一致。⟹ **偏差来源是 `F` 的摊薄**：case12 只 1 波 ⟹ `F` 占 20%；case16 4 波 ⟹ `F` 也 ×4，但**实测中 `F` 未严格按波数线性放大**（可能与 L2 cache / 调度重叠有关）。**这是模型的一个已知局限，已在 §12.4 的不确定性列表中体现** |
| 每 wave 单位成本近似恒定：V≈152064 时 case 7/15/17 = 451.3/455.8/455.9 us | ✅ **成立** | `ceiling.json`：case7 `k1_us_mask=425.50`、case15 `856.28`（= 2 波 × 428.1）、case17 `429.26`（1 波，`b_num=8`）⟹ **每波 425.5 / 428.1 / 429.3，极差 0.9%** ✅ 强证据 |
| K1 占比：mask 任务 **91.5–96.5%**、fused 任务 **46–79%** | ⚠️ **mask 区间需修正** | `ceiling.json` 的 `k1_share_mask` 实测范围是 **80.10%（case13）– 94.71%（case10）**，中位 ~92%。⟹ **Orchestrator 的下界 91.5% 偏高**（case13 只有 80.1%，因其 `K_MAX=128` 且 `n_blk_total=2` ⟹ K1 极小而 K2m 相对大）。`k1_share_fused` 实测 **51.78%（case13）– 76.47%（case10）** ⟹ Orchestrator 的「46–79%」**下界偏低**（实测最小 51.78%）。**rev3 §12.2(b) 采用实测区间：mask 80.1–94.7%、fused 51.8–76.5%** |
| 核心饥饿：18 case 中 **14 个 `b_num < 20`**；最差 case7 只用 **2/20** 组核心，roofline 达成率 **0.67%** | ✅ **已核实** | 本 Stage 1 逐 case 计算 `b_num = ceil(B/2)`：`<20` 的是 case 1,2,3,4,5,6,7,8,10,11,12,13,14,17 = **14 个** ✅；`≥20` 的是 9(32),15(32),16(64),18(32) = 4 个 ✅ |
| ⚠️ **split-V 对 `b_num>=20` 零收益甚至有害**（case15 若 S=2 → blocks=64 → waves=4 × scan(V/2)=228 → 912us ≈ 现状 911.6us） | ✅ **成立，且 rev3 已把它一般化为定理 P2** | §5.8.3 定理 P2 的完整证明；⚠️ **rev3 进一步指出该机制同样适用于 `b_num ∈ [11,19]`**（`S=2` ⟹ 2 波 ⟹ 严格无益）⟹ 判据边界从 20 收紧到 10（§0.2，**这是 rev3 与 Orchestrator 结论的唯一实质性差异，已在 §0.2 显著标注并用 C2 用例钉死**） |
| K1 内部成本拆分：`topk+merge` 占 K1 **26.4/55.3/60.0/66.6/78.5/78.8%**（case 2/3/8/16/7/15） | ✅ **已核实** | `probe_perf.json` 的 `topk_merge_share_of_K1` 字段：case2 `0.263838`、case3 `0.552868`、case7 `0.785205`、case8 `0.599924`、case15 `0.788…`、case16 `0.666…` ⟹ **6/6 完全一致** |
| 静态生成码普查（case15 每块）：softmax 侧 = 5 趟 4096 全宽 pass + 6 标量；topk 侧 = 48× MrgSort + 144 Sort32 repeats | ✅ **已核实** | `insn_census.txt` 的 `K1_case15_real.cc` 段：`reduce_max_elems 4096`、`reduce_sum_elems 4096`、`Exp[4096]`、`Sub[4096]`（4 趟全宽）+ `Broadcast 1`（第 5 趟）+ `scalar_op 6`；`TopK::MrgSort 48`、`TopK_total_Sort32_repeats 144` ⟹ **逐项一致** |
| ⟹ **Sort 是可并行化的主要收益来源** | ✅ **成立，且是 rev3 的 D-SPLIT-SCOPE 的直接依据** | §12.2(c) + §13.1 |
| K2g 占比：fused 任务 **23.4–48.7%（中位 37.6%）**，mask 任务 0% | ✅ **已核实** | `ceiling.json`：`k2_us_fused / (k1_us_fused + k2_us_fused)` 逐 case 计算，范围 **23.4%（case16: 262.08/695.1）– 48.7%**；mask 任务的 `k2_us_mask` 是 **K2m**（不是 K2g）⟹ K2g 在 mask 路径确实 0% ✅ |

### 16.10 §11「必读既有工件」的阅读确认

| 工件 | 是否已读 | 用途 |
|------|---------|------|
| `history_version/design_rev2.md`（= 覆盖前的 `DESIGN.md`，已 `diff -q` 核实相同） | ✅ 已读（章节结构全表 + §1.3 数学 + §3.2 API 映射表 + §3.5.3/3.5.4/3.5.5 证据链 + §4.5 UB + §5.6/5.7 审计 + §7 同步 + §9.2 L0 + §9.3 精度 + §9.4 哨兵 + §12 性能 + 修订记录） | rev3 的对照基准；**rev2 的全部有效内容在 rev3 中或原文沿用、或显式指向备份**，无静默丢失 |
| `history_version/design_rev1.md` | ✅ 已确认存在（117998 B） | 历史已否决路径参考（rev1 的 RNG 由仿射 LCG 改为 counter-based hash） |
| `r4_fused_sampling.py`（`19e51e8d…`，**已 sha256 复核**） | ✅ **全文已读**（620 行；K1 L92-364、K2m L367-412、K2g L415-555、host wrapper L558-620） | rev3 的 bitwise 基准；§14.2/§14.3/§14.4/§14.5 的逐行对照全部基于此 |
| `test_r4_fused_sampling.py` | ✅ 已读关键段（分层结构 `:11-13`、`_make_inputs :124`、`_run_case :133`、`STAB_CASES :1136-1160`、`ADV_CASES :1149-1155`、`_run_tail_adversary :1256`、`_make_ub_poisoner :1327-1348`、`_tau_invariant :1160`） | §9.2.4 的「既有套件关系」表 + §16.8 的失明点复核 |
| `proto.yaml`（`41ab7970…`，**已 sha256 复核**） | ✅ **全文已读**（132 行） | §15.1 的逐字段论证 |
| `debug_log.md` | ⚠️ **未逐行读**（60267 B）；已通过 rev2 §10.3 / `optimization_log.md` / 主 kernel docstring 的转述掌握其全部关键结论（缺陷 A/B 的实测数值、发现 1/3/5/6、V-b1..V-b6） | 如实记录：rev3 未直接引用 `debug_log.md` 的行号 |
| `perf_tuning/optimization_log.md`（round1） | ✅ 已读关键段（Part A [#1]-[#8] 判定、迭代 1/2 的 RESULT、[#3] split-V 可行性分析、[#4] cumsum 探测证据、[#8] AUTO_CV_COMBINE 分析） | §10.4 第 1/2/3 条修正、§3.2 `T.cumsum` 禁用、§12.2 成本拆分 |
| `perf_tuning/optimization_log_round2.md`（round2） | ⚠️ **部分读**（grep 到 `:435`「从未在 NPU 上运行过」与 `:571`「launch core 数 … case2 B=1 → b_num=1 → 只用 1 个 AIC + 2 个 AIV … 命中关注项 B」）；未逐行读全文（827+ 行） | 如实记录 |
| `perf_tuning/round2_bv_probe/`（`bv_ub_sweep.py`、`ub_probe.py`、`bv_ub_measured.json`、`bv_static_analysis.py`） | ✅ 已读（`ub_probe.py` 全文、`bv_ub_sweep.py:1-60`、`bv_ub_measured.json` 前 40 行） | §4.5 的 UB 实测法**直接复用其正则与公式** |
| `perf_tuning/round2_item2/`（`probe_perf.json`、`insn_census.txt`、`ceiling.json`、`analytic_bound.json`、`prune_rate_result.json`、`emitted_cc/*.cc`、`probe_k1_nosort.py`、`probe_k1_pruned.py`） | ✅ **已读并大量使用**（6 个 probe case 的 real/nosort/topk 拆分、case15/case08 的指令普查、18 case 的 k1/k2 时长与占比、生成码的循环与 barrier 结构） | §12.2 的**全部**标定常数、§12.4 的模型、§3.5.3 的参考实现、§14.3 的 `tau_p` barrier 证据 |
| `perf_tuning/round2_item4/`（`k1_split.py`） | ✅ 已读（`:1-46`） | §15.3.1 的 `Block Num` gate 定位与更新方案 |
| `bench/hap_thw_model/`（`r4_thw_common.py`、`compute_r4_fused_sampling.py`、`gates_report.py`、`t_hw_*_detail.json`） | ✅ 已读关键段（`:150-250` case 加载与 18-case 内联扩展、`:300-335` `tail_pad_split`/`block_split`、`compute_r4_fused_sampling.py:85-115` ops/elem 闭式与 VECTOR-BOUND 结论、`gates_report.py:24-45/98-130` GATE 1/3） | §15.3.2 / §15.3.3 的 t_hw 与 GATE 影响逐项裁定 |
| `.orchestrator_state.json` | ⚠️ **仅 grep，未整体读取，未写入**（禁令 1 遵守） | 只用于确认 `Block Num` gate 的记录位置（`:161`、`:499`、`:506`、`:508`、`:172`） |

---

## 修订记录

| 版本 | 日期 | 修改人 | 修改内容 |
|---|---|---|---|
| V01-a-Stage1 | 2026-10-06 | `@tilelang-op-analyst`（Stage 1） | 初版标准 DESIGN.md。基于 `DESIGN_script.md` 草稿的算法/数学（定理 T1/T2、online softmax、分层 top-k 归并），按**用户指定的 Developer 模式**全量重做 API 映射；范围收敛到 V01-a。**13 项草稿结论经源码复核后修正**（§10.4），其中 3 项为实质缺陷。产出 `proto.yaml` 与 8 用例 L0 门槛测试计划。 |
| V01.1 | 2026-10-06 | `@tilelang-op-analyst`（Stage 1 定向文档对齐） | Stage 2 实现验证后文档对齐（**非 `[DESIGN_ERROR]` 回退**）：① §9.1 golden 的 `tau_p`/`L` 修正为 raw `probs_sort` 严格前缀和；② §3.4 D-TIE 改分级判据；③ §10.2 V-2 尾块归约改精确尺寸 `[1,TAIL]` 缓冲 + 全量 reduce；④ 标注全链路 NPU 验证通过。 |
| V01-b + rev1 + rev2 | 2026-10-07 ~ 10-09 | Stage 1/2/3 | V01-b（K2g Gumbel-max 融合，定理 T3）+ rev1（RNG 由仿射 LCG 改为 counter-based avalanche hash lowbias32）+ rev2（尾块精度红线：`TAIL_PAD` + `ztp=[1,TAIL_PAD]` 的 `pad_value=-inf` 载入，规避 wheel 缺陷 A/B）。详见 `history_version/design_rev1.md` / `design_rev2.md`。 |
| **rev3** | **2026-10-10** | **`@tilelang-op-analyst`（Stage 1，第 2 次设计修订，`design_revision_count=2`）** | ⭐ **`USER_DIRECTED_ENHANCEMENT` —— 不是 `[DESIGN_ERROR]` 回退。** 现行 rev2 设计**正确**（四门禁全绿：`all` 74 PASS/0 FAIL+38 BOUND/0 WARN、`stability` 12、`stress` 8/8、`coverage` 25/0/0；cann-bench 36/36 `mismatch_count=0`；两轮 Stage 3 均答 `design_error: no`），但存在**并行度天花板**（`grid = ceildiv(B,2)`，vocab 维块内串行；18 case 中 14 个 `b_num<20`，最差 case7 只用 2/20 组核心、roofline 达成率 0.67%）。<br>**量化动机**（`probe_perf.json`/`ceiling.json`/`insn_census.txt` 实测，本 Stage 1 独立复核）：波数定律 `K1_us = ceil(b_num/20) × [F + n_blk×(p+t)]`，标定 `F(256)=21.63us`、`p(4096)=1.802us/块`、`t(4096,256)=8.902us/块`，对 6 个 probe case 拟合误差 5 个 ≤0.7%；**Sort/top-k 占 K1 的 26.4%–78.8%** 且是唯一可并行化的大头（每块 Sort32 144 repeats + MrgSort 48）。<br>**核心设计**：新增 4 个 kernel（K1A/K1B/K2gA/K2gB），**只把 Sort/top-k 与 Gumbel-argmax 切成 S 路并行，online-softmax 扫描完全不切** ⟹ `m/s` 的 bitwise 等价是**构造性的**（K1B 的 softmax 代码与现行逐条相同、块序相同），不需要 Orchestrator §3.2 的 Pass2/Pass4（**答「选项 D」**），因此**不需要任何 grid 级同步**（§2.5 已核实 Developer 模式不提供）。`S_cap = CUBE_CORE_NUM // b_num`（1-波上限，定理 P2）+ 回代收缩（无空 split，定理 P3）+ 三重收益 guard。<br>**满足 4 项硬约束**：① Developer 模式（4 个原型 kernel **编译实证**通过，48 次 `get_kernel_source()` 全成功）；② bitwise（§14 逐 pass 论证 + §9.2.2 Tier-2 双层判据 + D1/E1 专设用例）；③ 双路分派、现行 3 个 kernel **逐字节保留**；④ 范围 = K1 + K2g，**K2m 不切**。<br>**实测产出**：UB 账目 K1A ≤106496 B(54.24%)/K1B ≤89444(45.55%)/K2gA ≤148420(75.59%)/K2gB =132(0.07%)，**K1 侧余量从 24732 B 增至 89856 B**；`padCount==0` 与 UB→UB 32B 对齐在 **10/10** split-path case 编译实证通过；GM workspace 最大 40.3 KB；L0 计划扩至 **15 例**（全部 15 个 shape 已在 CPU 上实跑 `golden_r4` 确认 `valid_ref` 全 1）。<br>**对 Orchestrator §3 预分析的逐条裁定**（§16）：§3.1 ✅成立；§3.2 ⚠️bitwise 论证成立但「绕开跨核归约障碍」的理由需修正、**Pass2/4 落点选 D**；§3.3 ✅成立（`topk` 硬件确定性列为 U1 必测）；§3.4 ✅全部成立（补充 M3 完整证明 + 指出 K2g 的 bitwise **传递性依赖** K1 的 `tau`）；§4/§5/§6 ✅全部已核实（含仓库与 wheel 两处 `common.h` 比对）；§2 ⚠️K1 占比区间按实测修正为 mask 80.1–94.7%、fused 51.8–76.5%。<br>**唯一实质性差异**：分派边界从字面的 `b_num>=20` **收紧到 `b_num>10`**（定理 P2 证明 `b_num∈[11,19]` 时 split 严格无益），已在 §0.2 显著标注并由 L0-C2 用例钉死，**需 Orchestrator/用户知悉确认**。<br>**契约影响**：`proto.yaml` **不变**（§15.1 逐字段论证，sha256 `41ab7970…` 保持）；`Block Num` sanity gate 需**有意识更新**为「由 `_r4_split_plan` 派生期望值」（§15.3.1，含 4 步方案）；`t_hw` **不需重推导**（变化 <0.2%，且模型约定是 unique-bytes/no-reload-factor），但达成率报告需加注 2 趟读的口径差异（§15.3.2）；已发布分数需建立 **kernel sha256 ↔ 分数** 对照表并**并列发布**（§15.4）。<br>**如实标注的未验证项**（§14.6）：U1 `topk` 硬件确定性（🔴 单点最高风险，V-S1）、U2 `a_launch` 未实测（🔴，V-S3，已做 0–8us 全区间敏感性分析证明**零回退**）、U3 `F(128)` 不可分离（⚠️，V-S8）、U4 `grid=b_num*S` 的真实波数（⚠️，V-S7）、U5 TopK arena 复用竞态（⚠️，V-S5）、U6 `torch.empty` 掩盖漏写（⚠️，F1+V-S5）。<br>**预期收益（模型估计，非实测）**：fused geomean **1.15×–1.29×**、mask geomean **1.11×–1.20×**；单 case 最高 fused **3.5×–4.2×**（case7）；**12 个非 split case 精确 1.000×（零回退，构造性保证）**。<br>同步更新章节：§0（新增）、§1.2/1.3/1.4/1.5、§2.1/2.2/2.4、**§2.5（新增）**、§3.2/3.3/3.4/3.5.1/3.5.3/3.5.4/3.5.5、§4.3/4.4/**4.5（重做）**/4.6/4.7/**4.8（新增）**/**4.9（新增）**、§5.2/5.3/5.4/5.5/5.6/5.7/**5.8（新增）**、§6.1/6.3/6.4/6.5、§7.1/7.2/7.3、§8.1/8.2/8.6、§9.0/**9.1**/**9.2（重做，8→15 例 + Tier-2）**/9.3/9.4/**9.5（新增）**、§10.1（+9 项）/10.2（+8 项 V-S*）/10.3（+6 条 E-S*）/**10.4（新增）**/10.5、§11.1–11.6、§12.2（补实测模型）/12.3（强化）/**12.4（新增）**、**§13–§16（全部新增）**。 |
