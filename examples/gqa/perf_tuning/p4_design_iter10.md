# P4 设计文档 — transpose_qkv 消除（main kernel BSND strided 直读）
Stage 3 · iteration 10 · first-wave = flat kv1 + flat kv2 · **采纳**

- 基线：gqa.py md5 `606b07ff474ec42b8244f0831327d75b`（iter9 采纳后）
- 集成后：**`f59b561212e60a50b78a51671999ef7c`**（+217/−60 行，18 hunks，22 处行锚定断言编辑）
- 备份：`history_version/gqa_impl_s3_iter10.py` + `perf_tuning/gqa_impl_iter10_before.py`（均 606b07ff…）
- test_gqa.py 未动（`3e2d08881cb7b6f36d5c89b2fd37a146`）
- 探针先行：`probe_iter10_P4.md`（GO 判定 + 五项探针证据 + Gate 判据预注册）

---

## 1. 机制（输入侧镜像 iter8/9 out_bsnd 能力位基建）

### 1.1 kernel 侧（每工厂 5 处编辑，变体/集成同一文本）

`_gqa_kernel_prefill_g2stack_flat` 与 `_gqa_kernel_prefill_g2stack_flat_kv2`：

1. **签名**：`in_bsnd=False` kwarg（默认关 = shipped 生成码字节不变）
2. **声明**：`qkv_q_shape = [B, S_padded, N_q, D] if in_bsnd else [B, N_q, S_padded, D]`；
   `qkv_kv_shape = [B, S_kv_padded, N_kv, D] if in_bsnd else [B, N_kv, S_kv_padded, D]`
3. **emit_qload**：`if in_bsnd:` BSND strided 双盒 `Q[bz, rows, hq0(+1), :]`，
   `else:` shipped BHSD 盒逐字保留
4. **emit_g1 K 段**：`K[bz, kv窗, kg, :]`（行 stride N_kv·D）/ else shipped
5. **emit_g1 V 段**：同 K（kv2 的 kvpar 双窗各自转置）

**不动的东西（生成器 audit 强制）**：全部 set/wait flag 行逐字节相等；SEM/SIG id 与
token 数；槽深（q_l1 单槽、k/v_l1 双槽 parity）；C/V 步序（G1-first）；UB/L1/L0/ws
几何；算术；emit_fin 的 out_bsnd 写（wave-1 机制原样）。

### 1.2 lowering 代数（源码核实，探针 E1 生成码实证）

`src/op/ascend.cc compute_strideN`：BSND 盒 region extents=[1, rows, 1, D] 的尾
singleton（head 维）折叠 → realSrcN = D·N（K/V: N_kv·D；Q: N_q·D）；BHSD 盒
extents=[1, 1, rows, D] → realSrcN = D（连续）。发射形态
`copy_gm_to_l1<T,128,128>(dst, src, realSrcN, realTailM, realTailN, need_clear)`
——**每盒仍是单条指令**，模板维/realTailM/need_clear 与 BASE 逐 site 多重集相等
（`codegen_iter10_p4_*.cu`；kv1 19=19、kv2 16=16 条）。ND→NZ 由 MTE2 随路转换
（Catlass TileCopyTla，dst=zN fractal layout）——decode dense_expert 产线同款
（gqa.py:1229/:1266）。

### 1.3 逐位同一性论证

对齐 gate（S%64==0 ∧ S_kv%128==0）上 transpose_qkv 是纯重排（无零填充、无值变换）：
其 BHSD 输出元素 (b,head,s,d) == BSND 输入元素 (b,s,head,d)。变体从同一源地址读同一
元素集进同一 L1 zn/nz tile → mma/softmax/FIN 输入逐位相同 → final 输出逐位一致。
Gate 0 25/25 + 集成前后 13/13 raw-bit 实证闭合（含 NaN/Inf 注入、fullrange）。

### 1.4 host 侧（gqa_expert + adapter）

- `in_bsnd_eff = bool(in_bsnd) and S_padded == S and S_kv_padded == S_kv`
  （**双对齐 gate**——比 out_bsnd 多 S_kv 一项：K/V 零填充列语义必须留在 tqkv 路径）
- 两个 flat spec builder 传 `in_bsnd=in_bsnd_eff`；`chain_key` 携带 in_bsnd_eff
  （factory 调用者 BHSD 契约与 adapter BSND 请求永不共享缓存链）
- `_P4_BSND_KERNELS = ("g2stack_flat", "g2stack_flat_kv2")`；能力位
  `_run.in_bsnd = in_bsnd_eff and chosen in _P4_BSND_KERNELS`（编译回退自动退回旧路径）
- **adapter 重排**：kernel 解析（cache_key/gqa_expert）移到 transpose_qkv 之前
  （纯 host 重排——gqa_expert 只依赖 shape；NPU 流顺序不变：transpose(如需)→mask→main）；
  `if getattr(kernel_fn, "in_bsnd", False):` → BSND 原张量直通（**零 d2d 物化**，
  561103 纪律），else → shipped transpose_qkv 路径逐字保留
- NameError 闭合论证：`S_out` 只在 else 分支定义，后续 `if S_out > S_bs:` 仅在
  out_bsnd=False 时可达，而 in_bsnd ⇒ out_bsnd（gate 严格子集 + 名单子集）→ 安全
- `_prefill_run_padded` **零改动**（对齐域不进 pad 分支；Q/K/V 直通 kernel_fn）

### 1.5 Gate 0 harness 缺陷修正（披露，判据未动——iter7/8/9 先例模式）

首版 harness 把变体工厂 patch 与 tqkv pass-through 分成两个独立判断（前者无条件、
后者带 gate），非对齐 flat 命中用例（s100/padcol_skv200）上 ARM 编译了 in_bsnd
kernel 却喂 BHSD 张量（tilelang 不校验输入 shape）→ 静默错位被隔离用例当场抓住
（BITFAIL）。修正 = **同一 p4_gate 谓词单点驱动两处**（镜像集成后 in_bsnd_eff 的
结构一致性）。修正后 25/25 PASS。集成结构本身无此风险（单一变量驱动）。

---

## 2. Gate 结果（判据预注册于 candidate_iter10_P4.json，未移动）

### 2.1 Gate 0（run_iter10_p4_gate0.py，25/25 PASS）

- L1 覆盖域 14/14：c1/c2/c20/c18(MQA N_kv=1)/c3(N_q=128)/mc2 边界(stacked=40)/
  混合 mc(48%20=8)/c4/c7/kv2 混合 mc(46%20=6) + NaN/Inf 注入×2（nan_arm==nan_base==
  nan_ref：kv1 716、kv2 128；inf 位置+符号+有限位全等）+ fullrange×2（golden-
  invariance 判据，kv2 bf16 golden 自身 viol=196288 双臂相等，iter7/8 先例披露）
  ——全部 raw-bit 逐位一致 + variant invoked + tqkv SKIP(ARM)/RUN(BASE) + chosen 正确
- L3 隔离 11/11：per-head（r1/c5/odd_kv3）、prefill_expert（c19）、g2stack
  （c9-like/kv2-causal/s130）、decode（r5/auto）、**gate 隔离**（s100 flat 命中非
  对齐 S → tqkv RUN + tout(s_real) RUN；padcol_skv200 flat-kv2 命中 S 对齐但 S_kv
  非对齐 → tqkv RUN）——全部逐位不变 + variant 未装 + tqkv 行为双臂一致

### 2.2 Gate 2（同会话交替 msprof，48/48 rc=0，全 1650MHz，伪影 0 行）

| shape | 角色 | base e2e | p4 e2e | 改善 | 预测 | 分布 |
|---|---|---|---|---|---|---|
| **r2** | 主判据 kv1 | 36.11 | **26.35** | **+27.04%** | 27.0 | 不重叠 |
| **r4** | 主判据 kv2 | 216.83 | **188.08** | **+13.26%** | 12.8 | 不重叠 |
| r6 | 次判据 | 38.94 | 28.75 | +26.16% | 27.8 | 不重叠 |
| r3 | 次判据 | 80.77 | 63.17 | +21.80% | 23.0 | 不重叠 |
| c7 | 次判据 | 47.39 | 36.19 | +23.62% | ~19(est) | 不重叠 |
| r1 | 守卫 per-head | 20.17 | 20.39 | −1.09% | ~0 | 重叠 ✓ |
| r5 | 守卫 decode | 146.51 | 146.50 | +0.01% | ~0 | 重叠 ✓ |
| c19 | 守卫 prexpert | 50.14 | 50.01 | +0.26% | ~0 | 重叠 ✓ |
| c8 | 守卫 g2stack | 800.61 | 797.88 | +0.34% | ~0 | 重叠 ✓ |

- **tqkv 消除证据**：5/5 覆盖工况 p4 臂链内实例 = ['main'] only；base 臂 tqkv
  10.09/27.83/10.84/19.06/12.09µs（c7 首次实测 12.09，补齐探针估计）
- **main 成本**（判据：如实记录；tripwire Δmain>50%·tqkv）：r2 +0.32µs(+1.23%)/
  r4 −0.93(−0.49%)/r6 +0.65(+2.33%)/r3 +1.45(+2.35%)/c7 +0.90(+2.55%) —— 最大
  Δmain = tqkv 的 7.6%，tripwire 未触发；收益兑现率 92.4–103.3%
- **wait-id 判别门（强制）**：5/5 PASS——Σwait 膨胀 +0.08..+0.79µs vs 被消除 tqkv
  10.1–27.8µs（最大占比 7.8%，阈值 100%）；pacing 镜像 V-id3 −1.8%..+3.8%、
  C-id1 +0.4%..+6.5% 平直 → **无 iter4 式节拍迁移**（K/V 装载在 paced 环内但
  MTE2 富余吸收 strided 增量，双缓冲隐藏）
- 锚点：8/9 ±3% 带内；**r2 base −3.83% 越界如实披露**（小 shape 跨会话漂移，
  iter8 r1 +3.41% 先例；其 base 散布 6.3% 含一个 37.48 离群样本，同会话 A/B
  判据不受影响；决策工况主判据双臂分布不重叠）

### 2.3 Gate 3（采纳证据链）

备份（606b07ff…）→ 集成（22 处断言编辑，dry-run 先行）→ AST OK → **ruff
0.15.14 + 0.16.7 四门禁 exit 0**（最长行 140=上限）→ **全套件 24/24
（18 PRECISION_PASS + 6 BOUNDARY_PASS）exit 0 零 fallback**（覆盖域直验用例：
l0_prefill_causal_fp16=c1、l0_prefill_noncausal_fp16=c2、l0_prefill_g16_causal_bf16
=c3、l0_prefill_noncausal_bf16=c4、l0_cross_noncausal_bf16=c7、l1_mqa、非对齐
tail×3 全过）→ **自检 Test Passed!** → **集成前后逐位 13/13**（p4 域 5 例
tqkv 1→0 + final 逐位；域外 8 例 ≥5 要求：per-head×2/prexpert/g2stack/
padrow_s100/padcol_skv200/decode×2 逐位+行为一致）→ **msprof 兜底（集成文件零
monkeypatch）**：r2/c7 链内 main-only（tqkv ABSENT，e2e 26.91/35.83 落 Gate2 p4
带 +2.1%/+1.0%）、skv200 非对齐 gate tqkv PRESENT、r1 排除域 tqkv+main+tout
blk=16（P1 存活，e2e 20.58 带内）、r5 decode 146.71 锚内 → **instrumentation
grep=0**（T.printf/T.dump_tensor；:4454 print 为 shipped 原有 fallback 提示）→
**npu-smi Health OK、残留进程 0**。

---

## 3. 覆盖与排除（bench 视角）

- **新覆盖（P4）**：c1/c2/c3/c18/c20（flat kv1）+ c4/c7（flat kv2）= **7/20 case**
  的 tqkv 消除（e2e −13..−27%）；叠加已 shipped 的 out_bsnd（11/20 直写），
  **7 case 现在是"单 kernel 链"**（BSND 进 → BSND 出，零 aux）
- 排除（留档）：per-head r1/c5/c6（AUTO vid-reduction 未测读侧，保守排除）；
  c19/c8/c9/c10（expert 系机制同构可迁移，**next wave 首选 c19 33%**）；decode
  （已 BSND）；非对齐 S/S_kv（gate 排除，tqkv 保留零填充语义）
- cann-bench 影响预估（推断级）：7 case 各 −13..−27% e2e → 20-case 均速比提升
  依 shape 权重而定，建议 Orchestrator 跑固化评测实测

## 4. 反模式对照（performance-antipatterns.md）

新增代码复查：(1) launch 核数关注项——P4 再**删掉** 7 个 bench case 的每调用一次
tqkv launch（累计 11/20 直写 + 7/20 单 kernel 链）；有意保留项延续 iter1 §7.2 +
iter9 第二条（per-head ladder 及其 tout/tqkv）；(2) tile size——装载盒几何逐字节
不变（[64,128]/[128,128]），strided 256B burst 为 decode 产线同款；(3) 纯 AIV
memory-bound——最强形式：整 aux kernel 消除再 +7 case；(4) 冗余全局同步——零改动
（生成器断言）；(5) ws/L2——覆盖域每调用少一趟 tqkv 的 BSND 读 + BHSD 写 + main
回读（r4 −~50MB L2 流量）。**零新增违例**。

## 5. 产物清单

probe_iter10_P4.{py,md,log} + probe_iter10_P4_results.json + codegen_iter10_p4_*
（6 个 .cu）/ gen_variant_iter10_p4.py + gqa_p4_inbsnd_iter10_variant.py +
diff_iter10_p4.txt / candidate_iter10_P4.json / run_iter10_p4_gate0.py +
gate0_iter10_{covered,full}.log / prof_gqa_ab_iter10.py + run_matrix_iter10.sh +
msprof_op_iter10_S{1,2,3,4}_*（48 目录）+ parse_iter10_ab.{py,txt} /
integrate_iter10_p4.py + diff_iter10_p4_integration.txt /
save_gate3_outputs_iter10.py + compare_gate3_iter10.py + gate3_iter10_* /
run_g3_iter10.sh + msprof_op_iter10_G3_*（10 目录）+ parse_iter10_g3.{py,txt} /
gqa_impl_iter10_before.py + history_version/gqa_impl_s3_iter10.py
