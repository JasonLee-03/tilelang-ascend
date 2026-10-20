# P4 wave-2 设计文档 — transpose_qkv 消除推广到 g2stack + prefill_expert
Stage 3 · iteration 11 · mode=implement · round 3 收官实施轮（用户批准的预算延期轮）

- 基线：gqa.py md5 `f59b561212e60a50b78a51671999ef7c`（iter10 P4 wave-1 采纳后；开工前已核验）
- test_gqa.py md5 `3e2d08881cb7b6f36d5c89b2fd37a146`（未动，禁改）
- 靶点 kernel：`_gqa_kernel_prefill_g2stack`（bench c8/c9/c10 域）+
  `_gqa_kernel_prefill_expert`（bench c19 域）
- 机制：wave-1（iter10 已采纳、全链验证）的 in_bsnd 能力位逐字推广——main kernel
  直接从调用方 BSND 张量 strided 装载 Q/K/V，transpose_qkv launch 在对齐域被消除
- 部分采纳允许：一域过线一域否 → 只集成过线域（任务书授权）

---

## 1. 前置核验：wave-1 探针五结论对两靶点 kernel 的适用性（逐条核对）

wave-1 探针（probe_iter10_P4.md，GO 判定）的五项结论逐一对照两靶点 kernel：

| # | wave-1 结论 | g2stack 适用性 | prefill_expert 适用性 |
|---|---|---|---|
| 1 | lowering `compute_strideN`（src/op/ascend.cc:153-165）把 BSND 盒尾 singleton head 维折叠进 realSrcN=D·N，每盒仍是单条 `copy_gm_to_l1` | **适用**——每盒独立走同一 lowering；逐盒代数见 §2（含 stacked Q 双盒与 N_kv=1 连续退化边界） | **适用**——单 Q 盒 + parity K/V 盒，extents 与 wave-1 逐字同形 |
| 2 | ND→NZ 为 MTE2 随路转换（Catlass TileCopyTla），无 UB 中转 | **适用**——dst 侧（q_l1/k_l1 nz/v_l1 zn/p_l1 zn）逐字不动，layout 注解锁定 dst fractal，与 src 布局无关 | **适用**——同左 |
| 3 | strided MTE2 读速率无可测劣化（wave-1 Δmain +0.3..1.5µs = 0.5-2.6%，被双缓冲 prefetch 吸收；decode 产线 87-94% MTE2 屋顶） | **适用（先验强）**——K/V 盒与 decode dense_expert K/V 盒**同形同 runtime-idx 结构**（§3.2）；ns 批次环内双缓冲 prefetch 结构与 wave-1 flat 相同；Gate 2 wait-id 判别门强制复核 | **适用**——同左；c19 ns=1 单批次，K/V 每任务各 1 盒 |
| 4 | RAG asc_copy_gm2l1 约束：n_burst≤4095、burst_len≤65535×32B、gap 单位 32B | **适用**——四 bench shape 逐盒核算全部在界内（§2.3） | **适用**——c19 N_q=N_kv=32 为 gap 最大者，仍 ≪ 界（§2.3） |
| 5 | 生成码审计方法：copy_gm_to_l1 计数两臂相等、realSrcN 恰 D→N·D 翻转、flag/DataCopy 全源计数相等、realTailM/need_clear 多重集相等 | **适用**——audit 函数直接复用（probe_iter10_P4.py audit_codegen 模式）；g2stack 期望 q_l1 2 盒/k_l1 2/v_l1 2/p_l1 1，p_l1（ws_p 工作区读）必须保持 128 | **适用**——期望 q_l1 1/k_l1 2/v_l1 2/p_l1 1 |

**结论：无适用性缺口**。两处需要专门核对的差异点（任务书点名）：
(a) g2stack stacked-M Q 双盒 → §2 逐盒代数核对；
(b) g2stack/prefill_expert 的 K/V 装载在 **runtime `if i % 2 == 0:` parity 分支**内
（wave-1 flat 为 host-static side 分支）→ §3.2 产线同构先例核对。

## 2. g2stack stacked 盒逐盒代数核对（任务书要求项）

### 2.1 stacked-M Q 双盒

g2stack 一个任务服务 2 个 Q head（hq0, hq0+1），Q 装载为**两个独立 T.copy**，
目的地是 q_l1 [M_stack=128, D] 的静态切片（dense-expert 已验证的 sliced-L1 模式）：

- 盒 1（BASE BHSD）：`Q[bz, hq0, bx*BM:(bx+1)*BM, :]`，region extents = [1, 1, BM, D]
  → compute_strideN：strideN=D；dim2 extent=BM≠1 → 停 → realSrcN = **D=128（连续）**
- 盒 1（W2 BSND）：`Q[bz, bx*BM:(bx+1)*BM, hq0, :]`，extents = [1, BM, 1, D]
  → strideN=D；dim2（head）extent==1 → strideN=D·N_q；dim1 extent=BM≠1 → 停
  → realSrcN = **D·N_q（strided）**
- 盒 2：hq0+1，extents 逐字相同 → 同 realSrcN=D·N_q；GM 基址差 D·2B=256B（32B 对齐 ✓）

**stacked 拼接维（M_stack）不影响 strideN 折叠假设**，机理三条：
1. `compute_strideN` 只读 **source region 的 extents 与 source tensor shape**——
   M_stack 只存在于 **dst 侧**（q_l1 切片几何），两臂 dst 逐字不变；
2. 每个 T.copy 独立 lowering：两盒各自是 extents=[1,BM,1,D] 的单盒 → 各自单条
   `copy_gm_to_l1<T,128,128>`（dstM/dstN = L1 tile 几何，两臂相同）；
   need_clear 由 dst offset 是否为 tile 基址决定（盒 1 offset 0 → true，盒 2 → false，
   两臂相同，dst 不动）；
3. **wave-1 直接先例**：已采纳并全链验证的 flat kernel `emit_qload`（gqa.py:3152-3167）
   与 g2stack Q 装载**代数逐字同构**——同样的双盒、同样的 hq0/hq0+1、同样的
   q_l1[0:BM]/[BM:M_stack] 静态切片 dst、同样的 BSND strided src 形态
   （`Q[bz, rows, hq0, :]`），其生成码审计（codegen_iter10_p4_*.cu：q_l1 盒 realSrcN
   恰翻转为 N_q·D、计数不变）与 25/25 逐位验证已经覆盖了这一形态。
   g2stack 与 flat 的差异仅在装载点所处的循环结构（批次环 vs 展平稳态），
   盒代数零差异。

### 2.2 K/V parity 双盒（两 kernel 同构）

- BASE：`K[bz, kg, idx*BN:(idx+1)*BN, :]`，extents=[1,1,BN,D] → realSrcN=D（连续）
- W2：`K[bz, idx*BN:(idx+1)*BN, kg, :]`，extents=[1,BN,1,D] → realSrcN=**D·N_kv**
- V 同 K（v_l1 zn dst 不变）；k_l1_0/k_l1_1 parity 分支 dst 不变
- **N_kv=1 退化边界**（MQA 类，如 ns1_stacked20 用例）：realSrcN = D·1 = D = 连续——
  strided 机制自动退化为 BASE 同款描述符，代数闭合（无 gap）
- **runtime idx 窗口**：`idx = k*num_stages + i` 为 runtime 值，源基址 runtime 计算——
  与 BASE 的 runtime 基址（dim2 上的 idx 窗）同一机制，仅所在维不同

### 2.3 n_burst / burst_len / gap 界核对（RAG asc_copy_gm2l1，wave-1 探针 §1.3 口径）

盒参数：burst_len = D·2B = 256B = 8 DataBlock（≤65535 ✓）；32B 对齐：盒基址
= (bz·S·N + row0·N + head)·D·2B，D·2B=256B 为 32B 整数倍 ✓。

| shape | 盒 | n_burst（≤4095） | src_gap（32B 单位，≤65535） |
|---|---|---|---|
| c8（Nq32 Nkv8） | Q 双盒 | 64 ✓ | (32−1)·8 = 248 ✓ |
| c8 | K/V | 128 ✓ | (8−1)·8 = 56 ✓ |
| c9（Nq64 Nkv8） | Q | 64 ✓ | (64−1)·8 = 504 ✓ |
| c10（Nq128 Nkv8） | Q | 64 ✓ | (128−1)·8 = 1016 ✓（wave-1 c3 同值已验证） |
| c9/c10 | K/V | 128 ✓ | 56 ✓ |
| c19（Nq32 Nkv32） | Q | 64 ✓ | (32−1)·8 = 248 ✓ |
| c19 | K/V | 128 ✓ | 248 ✓ |
| ns1_stacked20（Nq20 Nkv1） | Q | 64 ✓ | (20−1)·8 = 152 ✓ |
| ns1_stacked20 | K/V | 128 ✓ | 0（N_kv=1 连续退化）✓ |

全部在界内；最大 gap 1016 与 wave-1 已验证的 c3 类（N_q=128）同值。

## 3. prefill_expert 装载结构差异（对 wave-1 flat）

1. **Q 单盒**（非 stacked）：`Q[bz, hq, rows, :]` → 全 q_l1 [BM, D]——比 wave-1
   的双盒更简单；extents 代数与 wave-1 Q 盒逐字相同（hq 替代 hq0）。
2. **K/V runtime parity 分支**：`if i % 2 == 0:`（i 为 T.serial 批内变量）——
   **产线同构先例**：decode `_gqa_kernel_dense_expert` 的 K/V 装载
   （gqa.py:1228-1231 / :1265-1268）是**完全相同的结构**——同 runtime
   `if i % 2 == 0:` parity 分支、同 `idx = k*num_stages + i` runtime 窗、同
   k_l1_0/1、v_l1_0/1 双槽——但读的是 **BSND strided 盒**
   `K[bz, idx*BN:(idx+1)*BN, kg, :]`。即：wave-2 的 K/V 编辑目标形态
   **在本文件产线代码中已在产**（自 Stage 2，iter6 实测 87-94% MTE2 屋顶）。
   wave-1 flat 用 host-static side 分支 + 静态窗口（0:block_N），wave-2 的
   runtime-idx + runtime-parity 形态由 decode 产线覆盖——两条先例合起来
   覆盖全部编辑形态。
3. **ns 批次环**：K/V 装载位于 GEMM1/GEMM2 批次环内（ns=min(kv_iters,8)：
   c8 ns=8、c9/c10 ns=4、c19 ns=1），SIG_K_L1/SIG_V_L1 本地事件双缓冲——
   与 wave-1 flat 的 paced 环结构位置相同（iter10 wait-id 门 5/5 PASS，
   Σwait 膨胀 ≤7.8% 收益）；本轮 Gate 2 wait-id 门强制复跑（§6）。
4. **out_bsnd 共存**：两 kernel 的 FIN 段已带 iter9 wave-2 的 `if out_bsnd:`
   分支（输出侧能力位）；本轮加的是**输入侧** `if in_bsnd:`（装载段）。
   两能力位在同一 kernel 内独立共存，互不触碰（FIN 在 kv 环外、装载在环内；
   flag 协议各自独立）。对齐域上 in_bsnd ∧ out_bsnd 同时为真 → **单 kernel 链**
   （BSND 进 → BSND 出，零 aux），与 wave-1 flat 域的最终形态一致。

## 4. 编辑位点清单（每 kernel 恰 5 处功能位点，wave-1 逐字模式）

行号锚定于 gqa.py md5 `f59b5612…`（生成器/集成器行级断言，漂移即中止）：

### 4.1 `_gqa_kernel_prefill_expert`（块 1997-2500）

| 位点 | 行 | 内容 |
|---|---|---|
| sig | 2014 | `out_bsnd=False,` → 追加 `in_bsnd=False,` |
| decl | 2129-2135 | out_shape 行后插入 `qkv_q_shape`/`qkv_kv_shape` host 变量；Q/K/V 声明改用变量 |
| qload | 2207-2210 | 单盒 `if in_bsnd:` BSND strided `else:` shipped 逐字保留 |
| k | 2234-2243 | parity 双分支各自 `if in_bsnd:` BSND `else:` shipped 逐字 |
| v | 2277-2286 | 同 k |

### 4.2 `_gqa_kernel_prefill_g2stack`（块 2503-3028）

| 位点 | 行 | 内容 |
|---|---|---|
| sig | 2520 | 同上 |
| decl | 2637-2643 | 同上 |
| qload | 2722-2729 | **双盒 stacked**：`if in_bsnd:` 两盒 BSND strided（hq0 / hq0+1 → q_l1 两切片）`else:` shipped 逐字 |
| k | 2750-2759 | 同 prexpert（文本逐字相同） |
| v | 2793-2802 | 同 k |

**位点顺序注意**：这两个 kernel 内为 sig < decl < qload < k < v（wave-1 flat 为
sig < qload < k < v < decl——flat 的声明在宏之后）。生成器的 diff 投影断言按
本顺序适配。

**不动的东西（生成器 audit 强制，wave-1 同款）**：全部 set/wait flag 行逐字节相等；
SEM/SIG id 与 token 数；槽深（q_l1 单槽、k/v_l1 双槽 parity、ws ns 深）；C/V 步序；
UB/L1/L0/ws 几何；算术；`if out_bsnd:` FIN 写（iter9 wave-2 机制原样）；
causal runtime kv_useful 环界；use_mask 分支。`in_bsnd=False` 生成码字节不变
（else 分支 shipped 行逐字保留）。

## 5. host 侧接线（最小改动，wave-1 机制零改动复用）

iter10 已建成的输入侧基建**全部通用**，本轮 host 改动仅 3 处功能编辑：

1. `_P4_BSND_KERNELS`：`("g2stack_flat", "g2stack_flat_kv2")` →
   追加 `"g2stack", "prefill_expert"`（= `_P2_BSND_KERNELS` 全集；
   子集关系 _P4 ⊆ _P2 保持 → adapter 的 in_bsnd⇒out_bsnd NameError 闭合论证不变）
2. `gqa_expert` g2stack spec builder：追加 `in_bsnd=in_bsnd_eff,`
3. `gqa_expert` prefill_expert spec builder：追加 `in_bsnd=in_bsnd_eff,`

**零改动复用**（结构性核对）：
- `in_bsnd_eff = bool(in_bsnd) and S_padded == S and S_kv_padded == S_kv`——
  纯 shape 谓词，域无关，双对齐 gate 对两新域语义相同（S_kv 非对齐的零填充列 +
  use_mask 语义留在 tqkv tail-mode；S 非对齐的零填充行同理）
- `chain_key` 已携带 in_bsnd_eff（factory BHSD 调用者与 adapter BSND 请求永不共链）
- `_run.in_bsnd = bool(in_bsnd_eff) and chosen in _P4_BSND_KERNELS`——名单扩容自动生效；
  编译回退（per-head ladder / g2）自动退回旧路径
- adapter resolve-then-supply 重排（iter10 已建）：`getattr(kernel_fn, "in_bsnd", False)`
  能力位驱动 BSND 直通 / shipped tqkv 路径——**对 kernel 名字零耦合**，无需改动
- `_prefill_run_padded`：对齐域 pad_S=pad_Skv=0 → 直通分支，Q/K/V（BSND 原张量）
  直喂 kernel_fn——零改动
- **单一 gate 变量纪律**（iter10 Gate 0 harness 缺陷教训）：in_bsnd_eff 单点驱动
  spec kwarg 与 adapter skip，两新域自动继承该结构（集成结构免疫双驱动分叉）

**注释/文档同步**（纯注释编辑，随集成）：_P4_BSND_KERNELS 注释块（删去
"as are g2stack / prefill_expert this wave"）、gqa_expert docstring in_bsnd 段、
adapter 两处 wave-1 注释、_run.in_bsnd 注释、两 kernel docstring 增补 in_bsnd 段。

### 5.1 chain 组合逻辑核对（in_bsnd × out_bsnd 共存矩阵）

| 域 | in_bsnd_eff | out_bsnd(bsnd_eff) | 链形态 |
|---|---|---|---|
| flat kv1/kv2 对齐（wave-1，7 case） | T | T | 单 kernel（不变，不得扰动） |
| g2stack/prexpert 对齐（**本轮**，c8/c9/c10/c19） | **T（新）** | T（iter9 已建） | **单 kernel（新）** |
| g2stack/prexpert S 对齐但 S_kv 非对齐（padcol） | F | T | tqkv(tail) + main（现状保持） |
| g2stack/prexpert S 非对齐 | F | F | tqkv(tail) + main + tout(s_real)（现状保持，P1 重铺生效） |
| per-head / g2 / dense_auto 回退 | F（名单外） | F | shipped 全链（现状保持） |
| decode | —（无 tqkv） | —（BSND 直供） | 不变 |

## 6. Gate 判据（预注册于 candidate_iter11_P4w2.json，不得移动球门）

### Gate 0（run_iter11_p4w2_gate0.py，公共 gqa() API 双臂 monkeypatch，同进程串行）
- **L1 覆盖域**（14 例）：g2stack 6（c8/c9/c10 bench 原 shape + kv2-causal 变长 +
  ns1-stacked20 边界（含 N_kv=1 连续退化）+ 小混合 mc causal）+ prexpert 4
  （c19 bench 原 shape + MHA noncausal use_mask=False + G3 奇数群 kv2-causal +
  G3 kv4-causal 变长）+ 特殊值 4（boundary_sparse NaN/Inf 注入 ×2 域 +
  fullrange ±65504 ×2 域）。判据：ARM 最终输出与 BASE 完整链 **raw-bit
  逐位一致**（view(int16)）+ golden viol_arm==viol_base==0（f6 口径；fullrange
  用 golden-invariance 判据，iter7/8 先例披露）+ variant invoked + tqkv
  SKIP(ARM)/RUN(BASE) + chosen==expected。
- **L3 隔离**（10 例）：per-head r1/c5（tqkv RUN 双臂）、**wave-1 flat r2/r4
  （tqkv ABSENT 双臂——wave-1 机制不得扰动**，这是"flat wave-1 不动"的直接证据）、
  decode r5/dense_auto（tqkv absent 双臂）、**域判定边界 4 例**：
  s130 g2stack 非对齐 S（gate F → tqkv RUN + tout RUN）、s100e prexpert 非对齐 S、
  **padcol_skv400 g2stack S 对齐/S_kv 非对齐**（gate F → tqkv RUN tail-mode、
  tout SKIP）、flat/g2stack stacked 数边界由覆盖例 ns1-stacked20（=20 → g2stack）
  与隔离例 r2（=64 ≥40 → flat）双侧夹逼。判据：variant NOT invoked + tqkv 行为
  双臂一致 + 输出逐位一致 + chosen 一致。
- 生成码审计：probe E1 对 g2stack（c8-like causal + c9-like noncausal use_mask=F +
  ns1-Nkv1 退化）与 prexpert（c19-like causal + MHA noncausal）共 5 配置，
  两臂源码对拷（计数/realSrcN/tail/need_clear/flag 全查）。

### Gate 2（同会话交替 msprof ≥3+3，B 矩阵 e2e = Σ TaskDur）
- **主判据（预注册，任务书原文）**：c19-like（最大单项）≥3% 且分布不重叠
  **AND**（c8-like **或** c10-like）≥3% 且分布不重叠——两者都必须过。
- 次判据：c9-like——如实报告；覆盖域任何工况 e2e 劣化 >3% 否决。
- 守卫：r2（wave-1 域，|Δ|≤3% 重叠带）、r1（per-head 排除域）、r5（decode）——
  任何 e2e 劣化 >3% 否决。
- main TaskDur 增量 tripwire：Δmain > 50%·被消除 tqkv → 经济学复核并披露
  （wave-1 口径）。
- **wait-id 判别门（强制）**：Σwait 膨胀 > 被消除 tqkv → REJECT（阈值 100%，
  iter10 口径）；pacing 镜像 **C-id2/V-id4**（ns 批次域的互替 pacer，iter3/5
  实证）±20% 带 + C-id1/V-id3（release 镜像）参考报告；镜像越带且收益 <50%
  预测 → 降级披露（iter9 c19 微分量披露分支先例）。
- BASE 复现锚点：c19 49.86-50.14 / c8 797.67-800.61 / c9 605.21 / c10 1071.94 /
  r2 26.35-26.91 / r5 146.07-146.51——≥20µs 分量 ±3%、微 kernel 链（r1 e2e
  ~20.4µs，tout launch 地板主导）±10%（iter8 finding 7 分档）；偏差如实报告。
- 频率/伪影纪律：全样本 1650MHz 核验、按 TaskDur 过滤不可能行（iter4 finding 5）、
  采集分段 + setsid、parse 伪影过滤。

### Gate 3（仅采纳后）
备份 `history_version/gqa_impl_s3_iter11.py`（记 md5 f59b5612…）→ 最小化集成
（行级断言可复现，dry-run 先行）→ 全套件 `unset PYTHONPATH; export
ACL_OP_INIT_MODE=1; python3 examples/gqa/test_gqa.py --level all` **24/24 exit 0
零 fallback** → `python3 examples/gqa/gqa.py` Test Passed! → 域外逐位 ≥5 例
（含 wave-1 覆盖域 r2 类、per-head r1、decode r5、非对齐 skv200、dense_auto）→
域内逐位抽验 → ruff 0.15.14 + 0.16.7 双门禁 → msprof 兜底（**集成文件零
monkeypatch**：c19/c9 链 tqkv ABSENT、skv200/skv400 tqkv PRESENT、r1 P1 tout
blk16 存活、r2 wave-1 单 kernel 链不变）→ instrumentation grep=0 → npu-smi
健康零残留。任一失败恢复备份（md5 == f59b5612…）。

## 7. 经济学（预注册，全部用既有实测数据）

收益 = 被消除的 tqkv TaskDur 全额（e2e=ΣTaskDur 口径，launch 地板已含在内——
iter8 披露口径；wave-1 兑现率 92.4-103.3% 为校准参照）；成本 = Δmain。

| case | 域 | tqkv µs（出处） | e2e shipped µs（出处） | 份额=中枢 | 悲观（Δmain=20%·tqkv） |
|---|---|---|---|---|---|
| c19 | prexpert | 16.50（iter9 phaseA） | 49.86（iter9 W2）/50.14（iter10 守卫） | **33.0%** | 26.4% |
| c8 | g2stack | 51.14（iter9 phaseA） | 797.67（iter9 W2）/800.61（iter10 守卫） | **6.4%** | 5.1% |
| c9 | g2stack | 44.32（iter9 phaseA） | 605.21（iter9 W2） | **7.3%** | 5.9% |
| c10 | g2stack | 89.99（iter9 phaseA） | 1071.94（iter9 W2） | **8.4%** | 6.7% |

四工况悲观带全部 >3% 阈值。B 矩阵口径对 bench 视角保守（iter9 finding 5：
bench-wall 与 kernel-sum 差在小 shape 冷/热 L2，c19 42.94 vs 57.40 先例）。
wave-1 兑现率参照 → 预期 c19 −30..−34%、c8 −5.9..−6.6%、c9 −6.7..−7.5%、
c10 −7.7..−8.7%（推断级，Gate 2 实测定案）。

L2/HBM 流量消除（每调用）：c8 ~96MB（Q 32MB+K/V 16MB 的 BSND 读+BHSD 写+回读）、
c9 ~96MB、c10 ~160MB、c19 ~12MB。

## 8. 反模式对照（performance-antipatterns.md，本轮改动预检 + 上机后复查）

1. **launch core 数关注项**：wave-2 再删 4 个 bench case 的每调用一次 tqkv
   launch（c8/c9/c10/c19 → 单 kernel 链；累计 11/20 单 kernel）。有意保留项
   延续：per-head AUTO ladder 按 block_num launch + 其 tqkv/tout（iter1 §7.2 +
   iter9 记录，理由不变）。
2. **tile size 过小**：装载盒几何逐字节不变（Q [64,128]、K/V [128,128]），
   256B strided burst = decode 产线同款。
3. **纯 AIV memory-bound 未流水**：最强形式——整 aux kernel 消除再 +4 case。
4. **冗余全局同步**：零 flag 行改动（生成器断言强制）。
5. **AIC/AIV CV overlap**：expert C/V scope 分离结构不动。
6. **ws/L2 指引**：覆盖域每调用少一趟 BSND 读+BHSD 写+回读（§7 流量表）。
**零新增违例预期**；上机后对集成代码复查一遍（Gate 3 前）。

## 9. 风险与回滚

| 风险 | 缓解 |
|---|---|
| runtime-parity 分支内 host-static `if in_bsnd:` 的 trace 语义 | out_bsnd 已在同两 kernel 的 runtime FIN 段用同款嵌套（产线验证）；生成码审计两臂计数相等兜底 |
| ns 批次环内 strided 装载拖慢 paced 稳态 | wave-1 同款结构位置实测 Δmain ≤2.6%；wait-id 判别门强制（膨胀>收益即 REJECT） |
| c8/c10 大 shape 采集时长 | 分段 + setsid；launch-count=3 单发链（iter9 同 shape 已跑通） |
| 一域失败 | 部分采纳授权：集成器按域参数化，否决域数据留档 |
| 任何 Gate 失败 | gqa.py 零触碰（Gate 3 前禁改）；已集成则恢复备份 md5 f59b5612… |

## 10. 产物清单（iter11 标识）

gen_variant_iter11_p4w2.py + gqa_p4w2_inbsnd_iter11_variant.py + diff_iter11_p4w2.txt /
probe_iter11_p4w2.py + probe_iter11_p4w2.log + probe_iter11_p4w2_results.json +
codegen_iter11_p4w2_*_{base,w2}.cu / candidate_iter11_P4w2.json /
run_iter11_p4w2_gate0.py + gate0_iter11_*.log / prof_gqa_ab_iter11.py +
run_matrix_iter11.sh + msprof_op_iter11_* + parse_iter11_ab.{py,txt} /
（采纳时）integrate_iter11_p4w2.py + diff_iter11_p4w2_integration.txt +
save/compare_gate3_outputs_iter11.py + gate3_iter11_* + run_g3_iter11.sh +
msprof_op_iter11_G3_* + parse_iter11_g3.{py,txt} + gqa_impl_iter11_before.py +
../history_version/gqa_impl_s3_iter11.py

---

## 11. Gate 结果摘要（判据预注册于 candidate_iter11_P4w2.json，未移动）

- **前置核验**：探针 6/6 PASS（E1 审计 6 配置全过：计数相等、realSrcN 恰翻转、
  p_l1 保持 128、tail/need_clear 多重集相等、flag/DataCopy 全源计数相等；E2 逐位
  2/2；E3 方向性 +1.60%/+0.67% 无劣化）。新发现一处（对称、无害、已披露）：
  ns=1 时 parity-else K/V 盒被编译器**双臂对称**死消除（c19 4=4、ns1 5=5 盒）——
  census 期望改为 ns-aware（探针期望缺陷修正，判据未动）。
- **Gate 0**：24/24 PASS（14 覆盖域逐位一致含 NaN/Inf + fullrange；10 隔离含
  wave-1 flat tqkv ABSENT 双臂、域判定边界 3 例、per-head/decode 全逐位）。
- **Gate 2**：主判据 PASS——**c19 +33.41%**（预测 33.0）非重叠 **AND c8 +5.40% +
  c10 +7.16%** 非重叠；次判据 c9 +5.70%；守卫 r2 +2.71%/r1 −0.56%/r5 +0.37%
  全部重叠带内；锚点 7/7 带内；tqkv 消除 4/4（w2 臂链 = ['main'] only）；
  Δmain tripwire 未触发（最大 21.8%）；wait-id 门 4/4 PASS（Σwait 膨胀最大
  26.1% < 100%，批次 pacer C-id2/V-id4 平直；V-id0↑/C-id5↓ 互补吸收指纹披露）。
- **Gate 3（两域均采纳）**：备份 f59b5612… → 集成（20 断言编辑，+224/−76，
  19 hunk，AST OK，全文件 flag 行序列相等断言）→ 新 md5
  **9aa645dcd318ff7d5eabba910b56125f** → ruff 0.15.14+0.16.7 四门禁 exit 0 →
  全套件 **24/24 零 fallback** → 自检 **Test Passed!** → 集成前后逐位 **17/17**
  （6 域内 tqkv 1→0；11 域外 census 一致）→ msprof 兜底（零 monkeypatch）：
  c19/c9 链 ['main'] only、skv200/skv400g tqkv PRESENT（tail 167/290µs）、
  r1 tqkv+main+tout blk16 P1 存活、r2 wave-1 不变、r5 锚内 → instrumentation
  grep=0 → npu-smi Health OK 零残留。
- **覆盖（bench 视角）**：c8/c9/c10/c19 四 case 成为单 kernel 链；累计 **11/20
  case 单 kernel 链**（BSND 进 → BSND 出，零 aux launch）。
