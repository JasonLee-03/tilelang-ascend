# P4 探针报告 — transpose_qkv 消除（main kernel BSND strided 直读）
Stage 3 · iteration 10 · Phase P（探针门控，先于一切构建）

- 日期：2026-10-20（探针会话）
- 基线：gqa.py md5 `606b07ff474ec42b8244f0831327d75b`（iter9 采纳后；探针前后核验一致）
- 探针产物：`probe_iter10_P4.py` / `probe_iter10_P4.log` / `probe_iter10_P4_results.json` /
  `gen_variant_iter10_p4.py` / `gqa_p4_inbsnd_iter10_variant.py` / `diff_iter10_p4.txt` /
  `codegen_iter10_p4_{r2_like_nc,c1_like_causal,r4_like_kv2}_{base,p4}.cu`
- 纪律：设备串行（单进程顺序执行）、禁 camodel、零 fp16/bf16 torch d2d 物化、gqa.py 零触碰

---

## 0. VERDICT：**GO**

预注册判据（任务书 §P6，逐项核对）：

| 判据 | 结果 | 证据 |
|---|---|---|
| (a) 表达性不可行 | **未触发** | 两臂编译通过；每盒仍是单条 `copy_gm_to_l1`，指令数两臂相等（kv1 19=19、kv2 16=16）；无 UB 中转、无多指令退化 |
| (b) Nz 关无法绕过且中转成本吃掉收益 | **未触发** | ND→NZ 由 MTE2 随路转换完成（Catlass TileCopyTla），无中转；decode 产线先例同款路径 |
| (c) 逐位失败 | **未触发** | 5/5 用例 raw-bit 逐位一致（含 NaN/Inf 注入 + fullrange） |
| (d) 净收益中枢 <3% | **未触发** | 中枢 12.8–27.8%（flat 域，Δmain 方向性实测无劣化；悲观假设 Δmain=20%·tqkv 仍 10–22%） |

**first-wave 范围（锁定）**：`_gqa_kernel_prefill_g2stack_flat`（kv1）+
`_gqa_kernel_prefill_g2stack_flat_kv2`（kv2）两个 expert 工厂（同一 emit-macro 结构，
一个 kernel 家族的两支）——bench 覆盖 **c1/c2/c3/c4/c7/c18/c20 共 7 case**。
对齐 gate：**S%64==0 ∧ S_kv%128==0**（非对齐 S 与非对齐 S_kv/padcol 的零填充语义
留在 transpose_qkv 路径，同 P2 wave-1 排除模式）。

**排除域（探针中核实并声明）**：
1. **per-head AUTO 域（r1/c5/c6）**——iter9 framework finding 1 的 AscendVidReduction
   strided 盒错误降级在 **UB→GM 写侧**实证；P4 的盒是 **GM→L1 读侧**，AUTO 模式对读侧
   盒的 vid 拆分行为**未经测试**（本探针未触碰 AUTO kernel）。按保守原则沿用 iter9
   同款域排除：first-wave 限 expert 手动 flag 系（读侧盒在 expert kernel 中由源码显式
   发射，无 AUTO vid 拆分介入——生成码审计确认每盒单条指令、realSrcN 正确）。
2. **g2stack（c8/c9/c10）+ prefill_expert（c19）**——机制同构可迁移（同为 expert 手动
   flag kernel、同 K/V 装载形态），但按范围纪律（本轮须走完全部 Gate，宁窄勿半）留待
   next_hint。c19 是当前 tqkv 份额最大域（33%），next wave 首选。
3. **decode（c11–c17）**——已是 BSND 直供，无 tqkv，天然域外。

---

## 1. 探针项 1：表达性查证（不凭记忆，三重证据）

### 1.1 源码链路（tilelang lowering → codegen → C++ 模板）

- **`src/op/ascend.cc:153-165 compute_strideN`**：从 copy region 的 extents 计算 GM 源
  行步长——尾维起 extent==1 的 singleton 维逐层折叠进 strideN
  （`strideN = shape[-1]`；`extents[i]==1 → strideN *= shape[i]`）。
  - BHSD 盒 `K[bz, kg, rows, :]`（extents=[1,1,block_N,D]）→ realSrcN = D = 128（连续）
  - BSND 盒 `K[bz, rows, kg, :]`（extents=[1,block_N,1,D]）→ singleton head 维折叠 →
    realSrcN = D·N_kv（strided）✓ 自动、无需任何新 API
- **`src/op/ascend.cc:211-213 + 545-551`**：`src.scope()=="global" && dst.scope()=="shared.l1"`
  → `copy_gm_to_l1`，runtime 参数 = (compute_strideN(src), validRow_src, validCol_src, dstM, dstN)。
- **`src/target/codegen_ascend.cc:2936/2979-3046`**：发射
  `copy_gm_to_l1<T,dstM,dstN>(dst[off], src[off], realSrcN, realTailM, realTailN, need_clear)`；
  need_clear 由 dst offset 是否为 tile 基址决定（与布局无关，两臂同构）。
- **`src/tl_templates/ascend/common.h:56-107 copy_gm_to_l1`**：
  `MakeLayoutFromTag(LayoutGM{tailM, realSrcN})` 构造 strided GM 源 layout →
  `TileCopyTla<AtlasA2, src(ND), dst(zN fractal)>` —— **ND→NZ 重排由 MTE2 随路转换完成**
  （dst layout = `tla::MakeLayout<T, LayoutL1_>(dstM, dstN)`，LayoutL1=zN）。
- GM→UB（`copy_gm_to_ub`，ascend.cc:224-247）同样携带 compute_strideN 的 strideN——
  但 P4 不涉及（Q/K/V 全部走 L1 路径；Mask 路径不变）。

### 1.2 产线先例（本仓库在产代码，非实验）

`_gqa_kernel_dense_expert`（decode，gqa.py:1000）自 Stage 2 起就从 **BSND GM strided 盒**
装载 K/V 进 **Nz/ZN L1**：
- K：`T.copy(K[bz, idx*block_N:(idx+1)*block_N, kg, :], k_l1_{0,1})`（gqa.py:1229/:1231，
  k_l1 = `make_nz_layout`，行 stride N_kv·D）
- V：`T.copy(V[bz, idx*block_N:(idx+1)*block_N, kg, :], v_l1_{0,1})`（gqa.py:1266/:1268，
  v_l1 = `make_zn_layout`）
- 速率已被 iter6 §6.2 双重闭合标定：decode K/V 物理流 **920 GB/s = 规格 MTE2 屋顶 87% /
  经验混合屋顶 94%**——strided 读在 HBM 大流形态下几乎无速率惩罚。
- P4 的 prefill K/V 盒与 decode 盒**同形**（[block_N=128, D=128]，行 stride N_kv·D）；
  Q 盒 [block_M=64, D=128] 行 stride N_q·D 是同一 MTE2 模式的参数变体。
- 注：decode 的 Q 装载（gqa.py:1193-1213）是每 s 一个 [G_inner,D] **连续**小盒
  （hq 维相邻），与 prefill Q 盒（跨 s strided）形态不同——prefill Q 盒的 strided 形态
  由本探针 E1/E2 首次直接验证（通过）。

### 1.3 CANN RAG（MTE2 描述符约束）

`asc_copy_gm2l1`（C API/数据搬运（矩阵），CANN 社区版 9.0.0-beta1）：A2/A3 均支持；
高维切分搬运形态 `(dst, src, n_burst, burst_len, src_gap, dst_gap, padMode)`：
- n_burst ∈ [1, 4095]；burst_len ∈ [1, 65535]（单位 DataBlock=32B）；gap 单位 32B
- P4 盒参数全部在界内：K/V 盒 n_burst=128、Q 盒 64；burst_len=256B/32=8；
  src_gap 最大 = (N_q−1)·D·2B/32 = (128−1)·8 = 1016（c3 类 N_q=128）≪ 65535
- 约束：src 起始地址按 dtype 对齐（BSND 盒偏移 = 整数×D×2B，fp16/bf16 2B 对齐 ✓，
  实际 DataCopy 路径要求 32B 对齐——盒偏移 (bz·S·N+hq0)·D·2B 中 D·2B=256B 为 32B 整数倍 ✓）
- SIMD API 侧 `DataCopyExtParams(blockCount, blockLen, srcStride, dstStride, rsv)` 同族语义
  （"切片数据搬运"）；tilelang 实际经 Catlass TileCopyTla 使用 MTE2 LoadData 类指令，
  参数在 decode 产线已长期实证。

### 1.4 examples/ 先例搜索

仓内 BSND 直读 attention kernel：**decode 路径（dense/dense_expert/dense_expert_d256）
输入侧即 BSND 直供**（gqa.py:742-754 "the four host-side BSND<->BHSD transposes of the
removed"——iter6 §6.3 引用的产线注释）。prefill 侧输入 BSND 直读本探针为首例（E1/E2 通过）。

---

## 2. 探针项 2：Nz 布局关

**结论：无需绕过——MTE2 硬件随路转换直接落 Nz，无 UB 中转。**

- 机制：`copy_gm_to_l1` 的 dst 是 `tla::MakeTensor<..., LayoutL1_ /* zN */>`，Catlass
  TileCopy 对 (GM ND strided 源 → L1 zN fractal 目的) 走 MTE2 的 ND2NZ 随路转换
  （RAG："随路转换ND2NZ搬运"为 DataCopy 家族正式能力；C API asc_copy_gm2l1 即 cube
  数据搬运指令）。
- 实证（本探针 E1 生成码审计，`codegen_iter10_p4_*.cu`）：P4 臂每个 Q/K/V 盒仍是
  **单条** `tl::ascend::copy_gm_to_l1<half,128,128>(...)`，仅 realSrcN 变化；
  模板维（dstM=128,dstN=128 = L1 tile 几何）、realTailM（Q 盒 64 / K/V 盒 128）、
  need_clear（首盒 true / 拼接盒 false）与 BASE 臂**逐 site 多重集相等**。
- 产线双重实证：decode K→nz_layout、V→zn_layout 均由 strided BSND 源直落（§1.2）。
- 中转成本核算：**不适用**（无中转）。

---

## 3. 探针项 3：strided MTE2 速率

**方法说明**：iter6 C 组微基准标定了连续 GM→L1/UB 屋顶（copy 1.307 TB/s L2 侧）；
wave-1 标定了 MTE3 strided **写** 1.07–1.17×/盒。本轮 MTE2 strided **读**采用
**真实 kernel 差分法**（变体 vs shipped 同会话交替，E3）+ 产线锚（decode 920 GB/s =
87–94% 屋顶）双重标定，未另建独立微基准（预算纪律；差分法直接给出对经济学有用的
Δmain，比孤立速率更贴判据）。

**E3 实测（probe-grade wall，同会话交替 3×30 reps 中位数，`probe_iter10_P4.log`）**：

| 用例 | main-only base µs | main-only p4 µs | Δ | 判读 |
|---|---|---|---|---|
| r2_like_nc（Nq48Nkv4） | 206.37 | 207.80 | **+0.70%** | wall 口径（含 ~180µs host 开销），Δ 为弱信号 |
| c1_like_causal（Nq32Nkv8） | 206.54 | 206.37 | **−0.08%** | 无劣化 |
| r4_like_kv2（Nq64Nkv8 bf16） | 214.95 | 213.94 | **−0.47%** | 无劣化 |

- **方向性结论：strided 装载在 main kernel 中无可测劣化**（三用例 −0.47%..+0.70%，
  均在 host-loop wall 噪声带内）。与 iter8 finding 5（消除 aux 后 main 常因 L2 少一趟
  写+读而更快）一致：tqkv 的 BHSD 写出 + main 读回被消除后，L2 压力下降。
- 定量 Δmain（msprof TaskDur 口径）**预留给 Gate 2**（判据级数据不在探针口径采）。
- 产线锚（推断级标注）：prefill K/V 盒与 decode 盒同形同 stride 量级；decode 实测
  87–94% 屋顶 ⇒ strided 读速率惩罚 ≈0 的先验强。

---

## 4. 探针项 4：最小正确性冒烟（E2，5/5 PASS）

方法：直建双臂（不经 adapter）——BASE = shipped 工厂 + `transpose_qkv`（真实产线链）；
P4 = 变体工厂 `in_bsnd=True` + 原始 BSND 张量直喂。判据 = raw-bit（`view(int16)`，
iter8 finding 3）+ golden f6 口径（NPU 输入 dtype golden、fp16 atol 1e-3 / bf16 2e-2、
rtol 1e-2、viol 双臂不变性）。

| 用例 | 域 | 结果 |
|---|---|---|
| r2_like_nc（B2S128Nq48Nkv4 fp16 noncausal，= bench c2 shape） | flat kv1 | **bit_identical ✓** viol 0/0 max_diff 1.2e-4 |
| c1_like_causal（B2S128Nq32Nkv8 fp16 causal，= bench c1 shape） | flat kv1 | **bit_identical ✓** viol 0/0 max_diff 4.9e-4 |
| r4_like_kv2（B4S256Nq64Nkv8 bf16 noncausal，= bench c4 shape） | flat kv2 | **bit_identical ✓** viol 0/0 max_diff 9.8e-4 |
| boundary_sparse_c1（q NaN + v ±Inf 注入，iter5 用例集） | flat kv1 | **bit_identical ✓** special-struct equal（nan 608=608=608ref、inf 位置+符号、有限位全等） |
| fullrange_r2（±65504 fp16） | flat kv1 | **bit_identical ✓** golden-invariance（golden 饱和 nan_ref=1572864，iter7/8 先例判据，显式披露） |

逐位一致的机理论证（设计文档级）：对齐域（S%64==0 ∧ S_kv%128==0）上 transpose_qkv 是
**纯重排**（无零填充、无值变换），其输出元素 (b,head,s,d) == 输入元素 (b,s,head,d)；
变体从同一源地址读同一元素集进同一 L1 zn/nz tile ⇒ 下游 mma/softmax/FIN 输入逐位相同。
E2 实证与该论证闭合。

---

## 5. 探针项 5：经济学核算（E4，全部用既有实测数据）

**收益** = 被消除的 tqkv TaskDur（e2e = Σ TaskDur 口径下 launch 地板已含在 TaskDur 内，
不重复计入——iter8 披露口径）。**成本** = Δmain（装载段 strided 增量；E3 方向性 ≈0）。

| case | 域 | tqkv µs（实测出处） | e2e shipped µs（出处） | 份额=净收益中枢 | 悲观（Δmain=20%·tqkv） |
|---|---|---|---|---|---|
| r2（=c1 shape） | flat kv1 | 10.02（iter6 B 矩阵 R2） | 37.16（iter8 Gate2） | **27.0%** | 21.6% |
| r6（=c2 shape） | flat kv1 | 10.85（iter6 R6） | 39.04（iter8） | **27.8%** | 22.2% |
| r3（=c3 shape） | flat kv1 | 18.38（iter6 R3） | 79.88（iter8） | **23.0%** | 18.4% |
| r4（=c4 shape） | flat kv2 | 27.77（iter6 R4） | 217.16（iter8） | **12.8%** | 10.2% |
| c7 | flat kv2 | 未实测（Gate 2 BASE 普查补采） | 47.69（iter8） | ~15–23%（推断级：Q 段同 r2、K/V 段 Skv 加倍） | — |
| c19 | prefill_expert | 16.50（iter9 phase_a） | 49.86（iter9） | 33.1%（**next wave 首选**） | 26.5% |
| c8/c9/c10 | g2stack | 51.14/44.32/89.99（iter9） | 797.67/605.21/1071.94（iter9） | 6.4/7.3/8.4%（next wave） | 5.1/5.9/6.7% |

- 装载字节量核算（strided 速率差 × 字节）：flat kv1 每任务 Q 32KB + K 32KB + V 32KB；
  r2 类 96 任务/20 核 ≈ 5 任务/核 ⇒ ~470KB/核装载。若 strided 读速率打 9 折
  （wave-1 MTE3 写侧实测带的镜像假设），增量 ≈ 470KB×0.11/80GB/s ≈ 0.6µs/核 —— 
  为 tqkv 10µs 的 6%，且 K/V 装载在双缓冲 paced 环内、MTE2 有富余（iter6：prefill 域
  aic_mte2 busy ≤27%），大概率被重叠吸收（E3 实测方向 Δ≈0 佐证）。
- **净收益区间（first-wave 域）**：中枢 **12.8–27.8%**，悲观带 **10.2–22.2%** —— 
  全部 ≫3% 采纳阈值 ⇒ 判据 (d) 通过。

---

## 6. Gate 判据预注册（go 后实施，不得移动球门）

### Gate 0（逐位 + 隔离，run_iter10_p4_gate0.py，公共 gqa() API 双臂 monkeypatch）
- 覆盖域（first-wave：flat kv1/kv2 ∧ S%64==0 ∧ S_kv%128==0）：ARM 最终输出与 BASE
  完整链 **raw-bit 逐位一致**；golden viol_arm==viol_base==0（f6 口径）；NaN/Inf 注入
  （iter5 用例集 ×2 域）+ fullrange（golden-invariance 判据，iter7/8 先例）；
  ARM 中变体 invoked ∧ transpose_qkv SKIPPED；BASE 中 tqkv RAN。
- 域外隔离（逐位不变 + 派发证据）：per-head（r1/c5 类）、g2stack（c8/c9 类）、
  prefill_expert（c19 类）、decode（r5）、**非对齐 S flat 命中**（s100 类 → tqkv RAN +
  tout(s_real) RAN）、**padcol Skv 非对齐 flat 命中**（Skv=200 → tqkv RAN，K/V 零填充列
  语义走 shipped 链）、wave-1 out_bsnd 语义不回退。
- 生成码审计（E1 已完成 kv1/kv2 各 1 配置；Gate 0 阶段对集成后的工厂再审计 1 轮，
  iter9 方法：装载描述符代数核对）。

### Gate 2（同会话交替 msprof，≥3+3）
- 口径：B 矩阵 e2e = Σ TaskDur（全链）；main TaskDur 单列。
- **主判据（first-wave 两族各一）**：r2（flat kv1，=c1 shape）+ r4（flat kv2，=c4 shape）
  e2e 改善 **≥3%** 且分布不重叠 ⇒ 采纳必要条件（两者都必须过）。
- 次判据：r6（=c2）、r3（=c3）、c7（kv2 cross）——护栏性质，劣化 >3% 否决。
- 域外守卫：r1（per-head）、r5（decode）、c19（prefill_expert）、c8（g2stack）——
  e2e 劣化 >3% 否决（P4 不得触碰任何域外路径）。
- **main TaskDur 增量如实记录**（P4 与 P2 判据不同：main 可能吸收装载成本，采纳判据
  看 e2e 净值；但若 Δmain > 50%·tqkv 则经济学复核并披露）。
- **wait-id 判别门（强制**——K/V 装载在 paced 稳态环内，iter4 教训）：Σwait 膨胀
  对照被消除的 tqkv TaskDur，膨胀 > tqkv ⇒ 否决（iter8/9 同款判据）；pacing 镜像
  V-id3 / C-id1 须平直（±20% 带内，iter4 式节拍迁移即否决并回滚）。
- BASE 复现锚点：≥20µs 分量 ±3%、微 kernel 分量（≤20µs）±10%（iter8 finding 7 分档）。
- 频率/伪影纪律：全样本 1650MHz 核验、PipeUtilization 不可能行按 TaskDur 过滤
  （iter4 finding 5）、采集分段 + setsid、伪影过滤（parse 脚本）。

### Gate 3（仅采纳后）
备份 `history_version/gqa_impl_s3_iter10.py`（记 md5）→ 最小化集成（行锚定断言，
iter8/9 integrate 模式）→ 全套件 `unset PYTHONPATH; export ACL_OP_INIT_MODE=1;
python3 examples/gqa/test_gqa.py --level all` **24/24 exit 0 零 fallback** →
`python3 examples/gqa/gqa.py` Test Passed! → 域外逐位 ≥5 例（集成前后 final 输出
raw-bit）→ ruff 0.15.14（+0.16.7 若可获取）双门禁 → msprof 兜底（**集成文件零
monkeypatch**：覆盖域链内 tqkv ABSENT、非对齐域 tqkv 仍在、域外 tout 语义不变）→
instrumentation grep=0 → npu-smi 健康零残留。任一失败恢复备份（md5 == 606b07ff…）。

---

## 7. no-go 分支预案（未触发，留档）

若探针失败，本轮以本报告收官，替代路径评估：
1. **tqkv 自身并行度/速率优化**（iter6 §6.3：tqkv 相对健康，12–320 任务、R4 达
   1475 GB/s = L2 屋顶 41%）——剩余空间小（网格已足），期望 <3%，不立项。
2. **tqkv 与 main 融合**（main kernel 内联 transpose）——侵入 paced 环，风险高于 P4，
   且 P4 已证明"消除"优于"融合"。
3. **框架需求**：无新增（P4 所需能力全部在产线/源码/RAG 三重证据内）。

## 8. 未验证项 / 推断标注（如实报告）

1. E3 的 Δmain 为 host-loop wall 口径（含 ~180µs host 开销），**方向性证据**而非定量；
   定量以 Gate 2 msprof TaskDur 为准。
2. c7 的 tqkv TaskDur 未直接实测（份额 ~15–23% 为 shape 模型推断级）——Gate 2 BASE
   臂普查自然补采。
3. per-head AUTO 域的 GM→L1 读侧 strided 盒行为**未测试**（保守排除，非已证不可行）；
   若 tilelang 修复 AscendVidReduction（iter9 framework finding 1），读侧行为需一并核验。
4. "strided 读速率惩罚 ≈0" 的机理解释（MTE2 随路转换 + L2 吸收）为推断级；实证 =
   E3 方向 + decode 产线锚 + Gate 2 定量。
5. 探针 E1 编译使用 tilelang JIT 缓存（base 臂 ~0.7s = 缓存命中），P4 臂为全新编译
   （~7.1s）——两臂生成码均本会话 dump 审计，不受缓存影响。
