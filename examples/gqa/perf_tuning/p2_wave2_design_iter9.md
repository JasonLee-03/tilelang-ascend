# P2 wave-2 设计文档 — 把 iter8 已验证的 BSND 直写机制推广到剩余 transpose_out 域（Stage 3 · iteration 9）

- 日期：2026-10-19（本会话 21:25 起）
- 基线：`gqa.py` md5 `3aa9cfa2625cdce96a62ce967785fe85`（iter8 P2 wave-1 采纳后；当日 cann-bench
  20/20 PASS，score 69.25，avg_speedup 0.6961）、`test_gqa.py` md5 `3e2d08881cb7b6f36d5c89b2fd37a146`（不动）。
  开工前已重新核验一致。
- 范围（用户批准）：P2 wave-2 —— per-head `_gqa_kernel_expert` / `prefill_expert` /
  `g2stack` 三个剩余 tout 域；非对齐 S 本轮继续排除（`transpose_out(s_real=)` 保留、P1 重铺继续
  生效；FIN 行级 masked 写 crop 方案不做，留档）。
- 依据：iter8 next_hint 1（wave-2 = 同机制推广，宿主基建已就绪）、p2_bsnd_design_iter8.md
  §2.2（wave-1 域外清单）、candidate_iter8_P2.json 7 条 framework_findings。

---

## 0. 结论速览（Phase A/B 判定）

| 域 | Phase A 经济判定 | Phase B 正确性判定 | 最终 |
|---|---|---|---|
| per-head（`_gqa_kernel_expert`，AUTO） | go（r1 −25.4% / c5 −9.6% / c6 −4.9% 中枢） | **结构性不可行**：AUTO threads=2 的 vid-reduction 对 strided BSND 写错误降级（实证 BITFAIL） | **排除**（如实退出，不硬改，非 DESIGN_ERROR） |
| prefill_expert（手动 flag expert） | go（c19 中枢 −13.1%，悲观 −12.8%） | 逐位一致（bit probe + Gate 0 6/6） | **推进 Gate 0/2/3** |
| g2stack（手动 flag expert） | go（c8 −6.1% / c9 −8.3% / c10 −12.0% 中枢，悲观带 −5.7..−11.5%） | 逐位一致（bit probe + Gate 0 8/8） | **推进 Gate 0/2/3** |

覆盖 bench 用例：c19 + c8/c9/c10 = **4/20**（叠加 wave-1 的 7/20 → BSND 直写累计 11/20）。
per-head 排除使 c5/c6 + r1 留在 shipped BHSD + tout(s_real) + P1 重铺路径（现状不回退）。

---

## 1. Phase A —— 分域 go/no-go（先算后建）

### 1.1 实测口径

BASE 臂 B 矩阵普查（`msprof op --launch-count=3 --aic-metrics=PipeUtilization,Memory,L2Cache`，
2 样本/shape，14/14 rc=0，全实例 1650/1650MHz，伪影行 0）：
`msprof_op_iter9_PA_*` + `parse_iter9_phaseA.py` + `phaseA_iter9_summary.json`。
e2e = 全链 TaskDur 之和（tqkv + main + tout）。派发矩阵 = `census_iter9_dispatch.py`
（纯 host 谓词复现，14 处行锚定 + md5 断言；c5/c6→per_head_bm64、c19→prefill_expert(ns=1)、
c8→g2stack(ns=8 causal)、c9→g2stack(ns=4)、c10→g2stack(ns=4 causal)；全部对齐 S）。

### 1.2 实测数据（BASE，µs）

| shape | 域 | e2e（2 样本） | tout（blk） | main | tqkv | tout 份额 |
|---|---|---|---|---|---|---|
| r1 | per-head | 20.12 / 19.88 | 5.48 / 5.04（16） | 10.09 | 4.65 | **26.3%** |
| c5 | per-head | 76.84 / 77.36 | 7.68 / 7.64（16） | 58.83 | 10.61 | 9.9% |
| c6 | per-head | 598.58 / 598.66 | 31.04 / 30.32（16） | 531.92 | 36.02 | 5.1% |
| c19 | prefill_expert | 56.98 / 57.82 | 7.58 / 7.88（16） | 33.17 | 16.50 | **13.5%** |
| c8 | g2stack | 860.04 / 856.26 | 55.28 / 55.88（64） | 751.43 | 51.14 | 6.5% |
| c9 | g2stack | 658.02 / 660.32 | 59.58 / 56.64（32） | 556.74 | 44.32 | **8.8%** |
| c10 | g2stack | 1225.68 / 1223.66 | 152.28 / 154.76（64） | 981.16 | 89.99 | **12.5%** |

任务书警示的"g2stack 大 shape tout 份额可能仅 3-7%"经实测修正为 **6.5–12.5%**（P1 重铺后
c8/c10 的 blk64、c9 的 blk32 网格已健康但字节量大：c10 tout = 128MB r+w ≈ 153µs）。

### 1.3 边际收益模型（wave-1 标定复用）

- **收益 = tout TaskDur 全额**（launch 地板已含于 TaskDur，e2e=Σ TaskDur 口径下不重复计入 ——
  iter8 披露原文沿用）。
- **成本 = main strided FIN 增量**：wave-1 probe 标定 +13..+30ns/盒（[32,128] 盒，1.07–1.17×），
  按盒数折算全暴露上界：c19 +0.2µs（128 任务×1 盒/20 核）、c8/c9 +3.1µs（1024×2/20）、
  c10 +6.1µs（2048×2/20）；且 wave-1 实证 4/5 工况 main **反而更快**（输出张量少一趟
  "写 BHSD→被 tout 读回"的 L2 写+读），故该列是悲观上界。
- 净中枢：c19 **−13.1%**、c8 **−6.1%**、c9 **−8.3%**、c10 **−12.0%**；悲观带（成本全暴露）
  −12.8% / −5.7% / −7.9% / −11.5% —— 全部 >3% → 两域 **go**。
- per-head（经济学上 go：r1 −25.4% / c5 −9.6% / c6 −4.9%）→ 被 §2 的正确性判定否决。

bench 视角备注（推断级，显式标注）：cann-bench 的 e2e wall（10-19 评测 c5 65.74 / c19 42.94µs）
低于本口径 kernel-sum（77.10 / 57.40µs），差值集中在小 shape 的 main 分量（推测评测为热 L2 稳态
多迭代均值，msprof --launch-count=3 偏冷）。tout 为带宽主导分量、冷热差异小 → **B 矩阵口径的
改善百分比对 bench 视角是保守估计**。大 shape（c8 858 vs 860.68）两口径几乎重合。

---

## 2. Phase B —— per-head 域结构性不可行判定（证据链）

### 2.1 机制

per-head `_gqa_kernel_expert` 是 **AUTO 模式** kernel（AUTO_CV_COMBINE/AUTO_SYNC/
MEMORY_PLANNING/AUTO_CV_SYNC 全开，无手动 flag 行），`T.Kernel(block_num, threads=kernel_threads)`，
其中 `use_pipeline = (not is_causal) and D<=128 and kv_iters>=2 and kv_iters%2==0`；
causal / 奇数 kv / D≠128 → **threads=2 串行形**。输出段为单盒尾写
`T.copy(acc_o_half, Output[bz, hq, bx*BM:(bx+1)*BM, :])`。

2 行编辑（同 wave-1）后，AUTO codegen 对两种 threads 形态的降级**不同**
（`codegen_audit_iter9.log` + `codegen_iter9_perhead_*_{base,w2}.cu`）：

| 形态 | BASE（连续 BHSD 写） | W2（strided BSND 写） | 判定 |
|---|---|---|---|
| threads=1（流水形） | `copy_ub_to_gm<half,128,64>(Output[cid*8192], …, 128, 64, 128)`：blockCount=64、dstStride=0 | `(Output[bx*65536+hq*128], …, 1024, 64, 128)`：blockCount=64、dstStride=(1024−128)×2=1792B、**无 vid 项** | **正确**（单 vid 全盒） |
| threads=2（串行形，r1/c5/c6 全在此） | `(Output[cid*8192+vid*4096], …, 128, **32**, 128)`：vid-reduction 把 [64,128] 盒二分为每 vid blockCount=32、偏移 vid×32×128 | `(Output[bx*65536+**vid*4096**+hq*128], …, 1024, **64**, 128)`：blockCount **未二分**（=64）却保留**连续布局**的 vid 偏移 vid×32×128=vid*4096 | **错误降级** |

threads=2 W2 的数值解码（r1：N_q=8、D=128、N_q·D=1024）：vid0 写 s∈[bx·64, bx·64+64)、
vid1 写 s∈[bx·64+4, bx·64+68)（4096=4×1024 → 行偏移 4）——**两 vid 重叠写 60 行 + 越界溢出
4 行到下一 bx 块**；且每 vid 的 acc_o_half 只被 AUTO 二分的 cast（`copy_ub_to_ub …, 32,128,…`）
填充了 32 行，blockCount=64 的读取越出本 vid 数据。

### 2.2 实证（`bit_probe_iter9.log`，公共 gqa() API 双臂 raw-bit 比较）

| 用例 | 形态 | 结果 |
|---|---|---|
| r1_perhead_causal_t2 | threads=2 | **BITFAIL** maxabsdiff=5.93e+04 |
| c5_perhead_d256_t2 | threads=2（D256） | **BITFAIL** maxabsdiff=NaN |
| perhead_pipe_t1 | threads=1 | BITOK（maxabsdiff=0） |
| c19_prexpert（对照） | expert 手动 flag | BITOK |
| c9_g2stack_kv4（对照） | expert 手动 flag | BITOK |

### 2.3 排除范围与替代方案评估

- **bench per-head 用例 r1/c5/c6 全部 causal → threads=2 → 全部损坏**；threads=1 正确的子域
  （非 causal ∧ D≤128 ∧ 偶 kv≥2）没有任何 bench 用例落入（此类 shape 实例数≥20 时走 expert 系，
  <20 的微 shape 非 bench 目标）→ 收窄 gate 到 threads=1 **零收益**，否决。
- 在 per-head 签名中暴露 vid、手写 expert 式 vid-二分 strided 盒：输出段变手动而**计算段仍是
  AUTO vid-二分**（cast 32 行/vid），两者对齐未证；且这是 kernel 结构改造，远超批准的
  "每 kernel ~2 行 Output 段编辑"，本轮否决。
- 修 tilelang AscendVidReduction pass：超出算子调优范围 → 记入 framework_findings 供
  工具链侧跟进。
- **处置**：per-head 保持 shipped BHSD + tout(s_real 能力) + P1 重铺（r1 tout 已 12.08→5.26µs，
  iter7 收益不回退）。集成**不得触碰** `_gqa_kernel_expert`（纪律 #9：结构性不可行如实退出，
  不硬改；此为框架 pass 限制而非算子设计错误 → 不标 DESIGN_ERROR）。

---

## 3. 覆盖域改动点（每 kernel 恰 2 行编辑，wave-1 同构）

### 3.1 布局映射与地址代数（继承 wave-1 §3，双向核对）

元素 `(b, hq, s, d)`：BHSD 偏移 `((b·N_q+hq)·S_padded+s)·D+d`；BSND 偏移
`((b·S_padded+s)·N_q+hq)·D+d`；`_transpose_v1` 语义（x=(B·M, n_pad, R)→y=(B·N, M·R)，
M=N_q/N=S/R=D）与 BSND 式代数恒等 → 同一元素同一地址（wave-1 已 15/15 bit-exact 实证）。

### 3.2 两个 kernel 的编辑点（`diff_iter9_p2w2.txt`，生成器行级断言）

| kernel | 声明（gqa.py 行） | 写盒（gqa.py 行） |
|---|---|---|
| `_gqa_kernel_prefill_expert`（1997–2468） | :2120 `Output: T.Tensor([B, N_q, S_padded, D])  # BHSD` → `[B, S_padded, N_q, D]  # BSND` | :2459 `Output[bz, hq, bx*BM+vid*half : +half, :]` → `Output[bz, bx*BM+vid*half : +half, hq, :]` |
| `_gqa_kernel_prefill_g2stack`（2471–2969） | :2603 同上 | :2960 `Output[bz, hq0+h, bx*BM+vid*half : +half, :]` → `Output[bz, bx*BM+vid*half : +half, hq0+h, :]` |

生成器 audit（强制）：flag 行集合逐字节相等（SIG_S_HALF 对、6 个 SEM、全部 set/wait 行）；
SEM_*/SIG_* 常量计数不变；`alloc_L1/L0A/L0B/L0C/alloc_shared`、ws_s/ws_p/ws_o 声明、
Q/K/V/Mask 声明、annotate_layout 计数全等；code-only 投影恰 2 行差异/kernel；diff hunk = 2/kernel；
`ast.parse` 通过。FIN 写留在 shipped 的 `wait V→MTE3(SIG_S_HALF) … set MTE3→V(SIG_S_HALF)` 对内
——**协议零改动**（wave-1 §4 的 pacing 不变性论证逐字适用：MTE3 指令条数不变、只有 dstStride 变、
不新增/删除任何 flag 事件、vid 行归属不变 → 无竞写面）。

### 3.3 生成码审计（expert 域，`codegen_audit_iter9.log`）

指令形态与 wave-1 相同：单条 `tl::ascend::copy_ub_to_gm<T,128,32>` → 一条 `DataCopyPad` +
`DataCopyExtParams`；`copy_ub_to_gm`/`copy_gm_to_ub`/`copy_ub_to_ub` 计数两臂逐一相等
（6 配置全部 COUNTS_EQUAL=True）。描述符差异（c19 例）：

- BASE：`copy_ub_to_gm<bf16,128,32>(Output[task*8192 + vid*4096], acc_s_half, 128, 32, 128)`
  → blockCount=32、blockLen=256B、dstStride=0
- W2：`copy_ub_to_gm<bf16,128,32>(Output[bz*524288 + bx*262144 + vid*131072 + hq*128], acc_s_half, 4096, 32, 128)`
  → blockCount=**32（保持 vid 二分）**、blockLen=256B、dstStride=(4096−128)×2=7872B

W2 偏移逐项代数核对（c19：N_q=32、D=128、S=128、N_q·D=4096）：bz 步长 S·N_q·D=524288 ✓；
bx 步长 BM·N_q·D=64×4096=262144 ✓；vid 步长 half·N_q·D=32×4096=131072 ✓（**正确重算**，
对照 per-head 的 4096 陈旧偏移）；hq 步长 D=128 ✓。g2stack c9（N_q=64：bz 4194304 / bx 524288 /
vid 262144 / hq0 256·(kg·4+gb) / h 128）与 c8（N_q=32、S=1024：bz 4194304 / bx 262144 /
vid 131072 / hq0 256·(kg·2+gb) / h 128）同法逐项核对通过（记录于 candidate JSON）。

### 3.4 crop / 尾行语义（继承 wave-1 §3.4）

覆盖 gate = `S % 64 == 0`（host 侧 `bsnd_eff`，kernel 不含 gate）→ `S_padded == S`：
prefill_expert 行窗 `[bx·64+vid·32, +32) ⊆ [0, S)`（bx < q_blocks=S/64）；g2stack 同式；
head 维 `hq0+h < N_q`（G 偶数由 use_g2_stack 门保证，prefill_expert hq < N_q）→ 无越界行，
无需 masked 写。非对齐 S 由 host gate 排除（Gate 0 s100e/s100g/s100p/l1_s_tail 隔离实证）。
causal 变长（kv_useful 逐任务不同）**不触输出段**：FIN 写在 kv 循环后按任务恒定发射一次，
与 causal 跳过无关（Gate 0 c8(kv8 causal)/c10(kv4 causal)/kv2 causal 逐位一致实证）。

### 3.5 UB liveness（继承 wave-1 §5）

FIN 段 UB 缓冲（acc_o/sumexp/buf_2d/acc_s_half/r_factors/neg_sm…）与其 flag 序化
（acc_s_half 双用由 SIG_S_HALF 单对完全序化）零改动；P2 不新增 UB 分配、不改任何 buffer
形状/生命周期；目的地是 GM。生成器 audit 断言全部 alloc 行计数不变。L2 附带效应：覆盖域
输出张量不再经历"写 BHSD→读→写 BSND"三遍（c10 每调用省 128MB L2/HBM 流量），方向为正、
不纳入判据。

---

## 4. 宿主侧改造（最小化集成方案，Gate 3 用）

wave-1 已建机制**原样复用**，增量仅 3 处：

1. 两个 kernel 工厂增 kwarg `out_bsnd=False`：`out_shape` host 变量选
   `[B, S_padded, N_q, D]`（True）或 shipped `[B, N_q, S_padded, D]`（False，**逐字节不变**）；
   FIN 写改 `if out_bsnd:` strided 盒 `else:` shipped 盒逐字保留 → out_bsnd=False 生成码字节不变。
2. `gqa_expert` 的 g2stack spec 与 prefill_expert spec builder 传 `out_bsnd=bsnd_eff`
   （`bsnd_eff = bool(out_bsnd) and S_padded == S` 对齐 gate，wave-1 已存在）；per-head ladder
   与 g2 fallback **不传**（保持 BHSD）。
3. `_P2_BSND_KERNELS` 能力集扩展：`("g2stack_flat", "g2stack_flat_kv2", "g2stack",
   "prefill_expert")` —— `_run.out_bsnd = bsnd_eff and chosen in _P2_BSND_KERNELS` 语义不变：
   编译回退到 per-head/g2 时能力位自动 False → adapter 继续调 transpose_out（安全回退）。

零改动项：`chain_key`（wave-1 已携带 bsnd_eff）、`_prefill_run_padded`（wave-1 已支持
out_bsnd 的 dim-1 crop 分支 + assert）、`gqa()` adapter（能力位判断已存在）、
`transpose_out`/`out_plan`/`_retile_out_plan`（P1，对域外继续生效）、decode 路径、
`_gqa_kernel_expert`（per-head，排除域）、`_gqa_kernel_prefill_g2`（末位回退，保持 BHSD）、
test_gqa.py。

派发矩阵核验（`census_iter9_dispatch.py`，md5+14 行锚定）：wave-2 覆盖 = chosen ∈
{prefill_expert, g2stack} ∧ S%64==0 → bench c8/c9/c10/c19；域外 = per-head（r1/c5/c6 +
odd-kv/causal-Skv<S/微实例回退形）、wave-1 flat（7 case）、decode（7 case）、非对齐 S
（padrow/l1_s_tail 类，tout(s_real) 保留）、g2（末位回退）。

---

## 5. Gate 0 / Gate 2 / Gate 3 方案（判据预注册于 candidate_iter9_P2w2.json）

- **Gate 0**（`run_iter9_p2w2_gate0.py`，公共 gqa() API 双臂同进程、双缓存清理）：
  L1 覆盖 14 例（prefill_expert 6：c19/ns2 非causal/ns4 causal 变长/padcol/boundary_sparse
  NaN+Inf/fullrange ±65504；g2stack 8：c9/c8/c10 同形 + kv2 causal 变长 + ns1 stacked=20 边界 +
  padcol causal + boundary_sparse + fullrange）逐位一致（raw-bit view(int16)）+ golden
  viol_arm==viol_base==0（f6 口径；fullrange 用 golden 不变性判据，iter7/8 先例披露）+
  L2 kernel 级 `transpose_out(main_pre_BHSD) == main_post_BSND` 逐位 + 派发证据
  （invoked/chosen/tout SKIP-RUN）；L3 隔离 11 例（per-head r1/c5/odd_kv3 —— **排除域必须未动**、
  wave-1 r2/r4、decode r5/dense_auto、非对齐 s100e/s100g/s100p/l1_s_tail —— 两覆盖域的
  gate 隔离证据）。
- **Gate 2**（`run_matrix_iter9.sh` + `prof_gqa_ab_iter9.py` + `parse_iter9_ab.py`）：同会话
  交替 ≥3+3（s100 守卫 2+2），轮转臂序；主判据 c19（prefill_expert）、c8+c9（g2stack，两者
  都必须过）；次判据 c10；守卫 r1/r2/r5 ±3%、s100e/s100g ±3%；main TaskDur ≤+5%；
  **wait-id 判别门**（两域都是手动 flag expert kernel，iter4 签名检查逐字适用：Σwait 膨胀 >
  被消除 tout → REJECT；V-id3 +20% ∧ 收益 <50% 预测 → 降置信）；tout ABSENT 实证；
  session health：BASE 复现 Phase A 均值 ±3%（r1 ±10% 微 kernel 带，预注册自 iter8 finding 7）。
- **Gate 3**（仅对过线域；`integrate_iter9_p2w2.py` 行锚定断言编辑）：备份 → 集成 →
  全套件 24/24 → 自检 → 域外逐位（含 per-head/wave-1/decode/非对齐 ≥5 例）→ ruff 双版本
  四门禁 → msprof 兜底（集成文件零 monkeypatch：新覆盖域 tout ABSENT、per-head r1 tout 仍在
  且 P1 重铺生效）→ instrumentation grep=0 → npu-smi 健康。部分采纳允许（能力位天然分域）。

---

## 6. 风险清单与缓解

| 风险 | 缓解 | 状态 |
|---|---|---|
| per-head AUTO strided 写损坏 | 域排除（§2 证据链），集成不触碰 `_gqa_kernel_expert`，Gate 0 L3 以 r1/c5 隔离实证未动 | 已闭合 |
| causal 变长 kv（c8 kv8 / c10 kv4 / kv2 causal）改动面 | FIN 在 kv 循环外恒定发射，与 kv_useful 无关；Gate 0 三个 causal 变长用例逐位一致 | 已闭合 |
| ns=1 批形（c19）与 ns=8 深批（c8）的 FIN 上下文差异 | 同一 SIG_S_HALF 对结构；Gate 0 覆盖 ns1/ns2/ns4/ns8 | 已闭合 |
| 大 shape main 增量（c10 2048 任务×2 盒） | wave-1 标定 +6.1µs 上界 = main 0.6%；护栏 ≤+5% + wait-id 门 | Gate 2 判定 |
| 编译回退（g2stack→prefill_expert→per-head→g2）布局歧义 | 能力位由**已解析的 chosen** 推导（回退到 per-head/g2 自动 False → tout 保留）；chain_key 携带 bsnd_eff | 机制已存在 |
| 双 expert kernel 并发死锁（评测 attempt-1 case15 前科） | 全部采集/冒烟/Gate 0 单进程串行；无并发对照实验 | 纪律执行中 |

---

## 7. 产物清单

- 本文档：`perf_tuning/p2_wave2_design_iter9.md`
- 派发普查：`census_iter9_dispatch.py` + `census_iter9_dispatch.txt`（14 锚 + md5 断言）
- Phase A：`run_phaseA_iter9.sh` + `msprof_op_iter9_PA_*`（14 目录）+ `parse_iter9_phaseA.py`
  + `phaseA_iter9_summary.json`
- 变体：`gen_variant_iter9_p2w2.py` + `gqa_w2_bsnd_iter9_variant.py` + `diff_iter9_p2w2.txt`
  （2 hunk/kernel，flag 行逐字节相等）+ `diff_iter9_p2w2_with_perhead_pre_verdict.txt`（per-head
  曾试证据）
- 生成码审计：`codegen_audit_iter9.py` + `codegen_audit_iter9.log` +
  `codegen_iter9_{perhead_r1,perhead_c5,perhead_pipe,prexpert_c19,g2stack_c9,g2stack_c8}_{base,w2}.cu`
- 逐位探针：`bit_probe_iter9.py` + `bit_probe_iter9.log`（per-head t2 BITFAIL / pipe+expert BITOK）
- Gate 0：`run_iter9_p2w2_gate0.py` + `candidate_iter9_p2w2_gate0.log`（25/25）
- Gate 2：`prof_gqa_ab_iter9.py` + `run_matrix_iter9.sh` + `parse_iter9_ab.py` +
  `msprof_op_iter9_W2_*`
- 判据预注册 + 全数据：`candidate_iter9_P2w2.json`

---

## 8. 实测结果附录（Gate 0/2/3 完成后回填；判据未移动）

### 8.1 Gate 0（`run_iter9_p2w2_gate0.py`，25/25 PASS）

- L1 覆盖 14 例（prefill_expert 6 + g2stack 8）：BASE 完整链 vs ARM 直写 **raw-bit 逐位一致**；
  L2 kernel 级 `transpose_out(main_pre_BHSD) == main_post_BSND` 逐位一致；golden
  viol_arm==viol_base==0（f6 口径）；NaN/Inf 注入双域（nan_arm==nan_base==nan_ref=275/128、
  Inf 位置+符号+有限位全等）；fullrange ±65504 双域（g2stack golden 有限 viol=0；prexpert
  golden 饱和 → golden 不变性判据，iter7/8 先例披露）。
- L3 隔离 11 例：per-head 排除域（r1/c5/odd_kv3）未 patch 且逐位不变、wave-1（r2/r4）tout
  ABSENT 两臂、decode ×2、非对齐 gate 隔离（s100e/s100g/s100p/l1_s_tail）tout(s_real) 保留且逐位。

### 8.2 Gate 2（同会话交替 3+3 / 守卫 2+2，50/50 rc=0，全 1650/1650MHz，伪影 0 行）

| 工况 | 角色 | BASE e2e | W2 e2e | **实测改善** | Phase A 预测 | 误差 | 分布 |
|---|---|---|---|---|---|---|---|
| c19 | **PRIMARY**（prefill_expert） | 57.41 | **49.86** | **+13.15%** | −13.1% | +0.05pp | 不重叠 |
| c8 | **PRIMARY**（g2stack） | 855.69 | **797.67** | **+6.78%** | −6.1% | +0.7pp | 不重叠 |
| c9 | **PRIMARY**（g2stack） | 657.87 | **605.21** | **+8.00%** | −8.3% | −0.3pp | 不重叠 |
| c10 | 覆盖-次判据 | 1231.62 | 1071.94 | **+12.97%** | −12.0% | +1.0pp | 不重叠 |
| r1 | 域外守卫（per-head 排除域） | 20.67 | 21.01 | −1.61% | ~0 | — | 重叠 |
| r2 | 域外守卫（wave-1） | 37.55 | 37.13 | +1.12% | 0 | — | 重叠 |
| r5 | 域外守卫（decode） | 146.07 | 145.80 | +0.19% | 0 | — | 重叠 |
| s100e/s100g | 非对齐守卫 | 235.00/777.55 | 235.88/776.78 | −0.37%/+0.10% | 0 | — | 重叠 |

- tout 消除实证：4 覆盖工况 w2 臂链内实例仅 tqkv+main，**tout ABSENT**。
- main 成本护栏（限 +5%）：c19 +1.65% / c8 **−0.35%** / c9 +0.69% / c10 +0.09%；
  收益兑现率 99.3/105.2/91.1/100.3%。
- wait-id 判别门 4/4 PASS：Σwait 膨胀 +0.36/+1.06/+4.02/+8.35µs ≪ 被消除 tout
  7.60/55.14/57.82/159.15µs（最大占比 5%，阈值 100%）；c8/c9/c10 pacing 镜像平直；
  c19 V-id3 +20.7% 系 0.36→0.43µs 微分量（+0.07µs）噪声级触发**披露分支**（收益=预测
  100% → 按预注册不降级）。
- session health：BASE 复现 Phase A 均值 c19 +0.01% / c8 −0.29% / c9 −0.20% / c10 +0.57%
  （±3% 内）、r1 +3.37%（预注册 ±10% 带内）。

### 8.3 Gate 3（两域均过线 → 一次集成）

| 项 | 结果 |
|---|---|
| 备份 | history_version/gqa_impl_s3_iter9.py + perf_tuning/gqa_impl_iter9_before.py（均 3aa9cfa2…） |
| 集成 | `integrate_iter9_p2w2.py` 18 处行锚定断言编辑，+102/−27 行 16 hunk；md5 3aa9cfa2… → **606b07ff474ec42b8244f0831327d75b** |
| ruff | 0.15.14 + 0.16.7 四门禁 exit 0（最长行 140=上限） |
| 全套件 | 18 PRECISION_PASS + 6 BOUNDARY_PASS = **24/24，exit 0，零 fallback**（Nq32Nkv32 用例直验 prefill_expert 覆盖域） |
| 自检 | `python3 examples/gqa/gqa.py` → **Test Passed!**（prefill_causal=bench c1 viol=0） |
| 逐位 pre/post | **15/15**（final 全逐位；wave-2 5 例 kernel 级等价；域外 main BHSD 逐位 5/5 含 per-head；newly-skipped 恰=wave-2 集合；P1 计划域外逐一保持） |
| msprof 兜底 | 集成文件零 patch：c19/c9 **tout ABSENT**（e2e 49.06/606.59，与 Gate 2 W2 差 −1.6%/+0.2%）；**r1 tout 仍在 5.51µs blk16**；r2/r5 未动 |
| instrumentation | grep=0 |
| npu-smi | Health OK、AICore 0%、残留进程 0 |
| harness 披露 | 比较器 `main_is_bhsd` 断言误用真实 S（非对齐用例 main 为 padded 形状）→ 改 S_padded；逐位判据逐字未动（iter7/8 先例），修正后 15/15 |
