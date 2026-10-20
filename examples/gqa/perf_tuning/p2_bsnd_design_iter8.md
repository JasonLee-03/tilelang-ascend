# P2 设计文档 — prefill main kernel BSND 直写输出，消除 transpose_out（Stage 3 · iteration 8）

- 日期：2026-10-19（本会话 15:40 起）
- 基线：`gqa.py` md5 `4b4e453e4b8f2922baefbecb85912494`（iter7 P1 采纳后），`test_gqa.py` md5
  `3e2d08881cb7b6f36d5c89b2fd37a146`（不动）
- 范围（用户批准）：**仅 P2**；wave-1 覆盖域 = flat kv1 + flat kv2 两个 kernel 的**对齐 S** 工况
- 依据：iter6 §7 P2 卡 + §6.3-4（吸收路径存在性）、iter7 next_hint 1（P1 后边际收益需重估）、
  probe_iter7_P3.md §4-2（P2 为 iter8 主收益线）

---

## 0. Phase A 结论速览

**verdict = go**（修正后边际收益中枢 e2e **−12% … −19.5%**，最悲观带 −10.4%，主判据工况 r2/r6
分别 −17.0%/−19.5% 中枢，远超 3% 阈值）。

三项先决事实全部实测闭合（非模型推断）：

1. **可表达性**：expert 模式下 V scope 的 `T.copy(acc_s_half, Output[bz, rows, hq, :])` 对
   BSND 4-D 声明张量**可编译、可运行、逐位正确**（probe `--exp num`：5 shape × 3 臂 = 15/15
   bit_exact=True，含 bhsd 参照臂与 2-D 展平臂）。
2. **指令形态**：生成码为**单条** `tl::ascend::copy_ub_to_gm<T,128,32>(dst, src, realdstN, 32, 128)`
   = 一条 `DataCopyPad` + `DataCopyExtParams(blockCount=32, blockLen=256B, dstStride=(N_q·D−D)·2B)`。
   与现行连续写（dstStride=0）**指令条数完全相同**（probe 实测 mte3_instructions 65/81/105/209
   两臂逐一相等）——差别只在描述符的 dstStride 字段，无新增标量工作。
   （2-D 展平臂 `bsnd2d` 生成码带运行时行数 clamp 条件表达式 → **弃用**，选 4-D 形式。）
3. **成本**：strided 每盒 MTE3 时间 = 连续盒的 **1.07–1.17×**（probe `--exp rate`，2 样本/臂，
   全 1650MHz），折算到真实 kernel 的盒流后 main kernel 增量上界 **+0.13 … +0.98µs**
   （= main TaskDur 的 0.5%–1.3%），远小于被消除的 tout（7.67–29.58µs）。

---

## 1. 边际收益重估（P1 后新口径；iter6 的 −9..−44% 是 P1 前口径，已作废）

### 1.1 收益项 = 被消除的 tout TaskDur（iter7 实测，P1 后新基线）

| 工况 | e2e 锚点（iter7 V2 采纳臂，µs） | tout TaskDur（P1 后，µs） | tout blk | tout 占 e2e |
|---|---|---|---|---|
| r2 (=bench c1/c20 类) | 43.70 | **7.67** | 16 | 17.6% |
| r6 (=bench c2) | 49.06 | **9.69** | 12 | 19.8% |
| r3 (=bench c3) | 98.31 | **17.63** | 16 | 17.9% |
| r4 (=bench c4) | 246.45 | **29.58** | 16 | 12.0% |
| c7 (=bench c7) | ~53（本轮 BASE 臂实测填补） | ~7.67（网格/字节与 r2 同：(2,128) blk16、4MB r+w） | 16 | ~14% |
| r1 (per-head，**域外**) | 20.05 | 4.86 | 16 | 24.2% |
| r5 (decode，**域外**) | 146.81 | 0（BSND 直供） | — | 0% |

**launch 地板不重复计入**：本项目 e2e 口径 = 全 kernel 链 TaskDur 之和，而 tout 的 TaskDur
（如 r2 的 7.67µs 对 4MB 流量）**本身已含**其 2.5–3.8µs launch/排空地板（iter3 §4、iter7 V2
模型 `wall = L + waves×max(W_OH, bpb/RATE)` 的 L 项）。故收益 = tout TaskDur 全额，**不再另加**
一次 launch 地板（任务书该项在此口径下属重复计数，显式披露）。eval harness 若含核间空隙，
则删除一次 launch 另有未量化的正向上行（不计入判据）。

### 1.2 成本项 = main kernel V 侧 FIN 改 strided 的 MTE3 增量（probe 实测标定）

probe 结构 = 逐字镜像 shipped FIN 拍（每盒：`T.copy(acc_o→acc_s_half)` UB→UB + V/MTE3 flag
往返 + MTE3 写），GM 读提出循环外，任务流重放 rep 次；两臂 V 侧工作完全相同 → 墙钟/MTE3 时间差
= strided 寻址代价。盒粒度与真实 kernel 完全一致（[half=32, D=128]，r2 类 8KB fp16 = 32×256B burst，
行 stride N_q·D）。

| shape | rep | boxes/vid(block0) | ns/box 连续(bhsd) | ns/box strided(bsnd4d) | Δns/box | 比率 | 真实 kernel boxes/vid | **预测 main MTE3 增量** |
|---|---|---|---|---|---|---|---|---|
| r2 | 8 | 64 | 185.4 | 215.3 | +30.0 | 1.16 | 8 | **+0.240µs** |
| r6 | 8 | 80 | 191.8 | 204.9 | +13.0 | 1.07 | 10 | **+0.130µs** |
| r3 | 4 | 104 | 181.4 | 211.6 | +30.2 | 1.17 | 26 | **+0.785µs** |
| r4 | 4 | 208 | 170.7 | 189.5 | +18.8 | 1.11 | 52 | **+0.978µs** |

（2 样本/臂；probe 墙钟差 −0.34/+0.40/+0.92/+1.41µs，即 strided 代价在孤立 probe 中也只有
部分暴露到墙钟。`aiv_scalar_wait_time` 两臂差 = **+0.000µs**（四 shape 全部）→ 无等待膨胀签名。
mte3_instructions 两臂逐一相等 → 纯描述符效应。）

### 1.3 净收益区间（逐工况；置信度标注）

main 增量取三档：**probe 上界**（Δns/box × boxes，全暴露）、**2×**（flag 往返使 MTE3 排空延迟
暴露两次）、**4×**（悲观：strided 排空把下一拍 FIN 的 V 拷贝阻塞，pacing 全链传导）。

| 工况 | 收益 (−tout) | 成本 +1× | **净 % (中枢)** | 成本 +2× | 净 % | 成本 +4× | **净 % (悲观)** | 置信度 |
|---|---|---|---|---|---|---|---|---|
| r2 | −7.67 | +0.24 | **−17.0%** | +0.48 | −16.5% | +0.96 | −15.4% | 高（tout 份额 17.6%、成本 ≤0.4% e2e） |
| r6 | −9.69 | +0.13 | **−19.5%** | +0.26 | −19.2% | +0.52 | −18.7% | 高 |
| r3 | −17.63 | +0.79 | **−17.1%** | +1.57 | −16.3% | +3.14 | −14.7% | 高 |
| r4 | −29.58 | +0.98 | **−11.6%** | +1.96 | −11.2% | +3.91 | −10.4% | 中-高（main 最大、tout 份额最小） |
| c7 | ~−7.67 | +0.24 | **~−14.0%** | +0.48 | ~−13.5% | +0.96 | ~−12.6% | 中（e2e 锚点本轮 BASE 臂实测填补） |

**边际收益修正区间：e2e −10.4% … −19.5%（中枢 −12% … −19.5%）**。iter6 P1 前口径的
−9..−44% 中，高段（r1 −44%）属 per-head 域（wave-1 不含），中段被 P1 提前收割
（tout 已从 12–24µs 降到 4.9–29.6µs）。

**go/no-go 判定：go**（中枢 ≫ 3%，且悲观带全部 >10%；主判据工况 r2/r6 的成本项占 e2e ≤0.4%）。

---

## 2. 范围界定（prefill 派发矩阵 → wave-1 覆盖域）

`census_iter8_dispatch.py`（纯 host 复现 gqa_expert 派发谓词，8 处行锚定 drift guard，
gqa.py md5 断言）：

### 2.1 覆盖域（chosen ∈ {g2stack_flat, g2stack_flat_kv2} ∧ S % 64 == 0）

| 域 | kernel | bench 用例 | perf 工况 |
|---|---|---|---|
| flat kv1 | `_gqa_kernel_prefill_g2stack_flat` | c1, c2, c3, c18, c20（5） | r2, r3, r6 |
| flat kv2 | `_gqa_kernel_prefill_g2stack_flat_kv2` | c4, c7（2） | r4, c7 |
| **合计** | 2 kernel | **7/20 bench 用例** | 5 工况 |

### 2.2 域外（保留 transpose_out，P1 重铺规则继续生效）

| 类别 | 用例 | 保留原因 |
|---|---|---|
| per-head expert（`_gqa_kernel_expert`） | r1、c5、c6（D=256 / 实例数不足） | 第 3 个 kernel 的 FIN 改造 = 额外变体+Gate 0 面积；r1 的 tout 已被 P1 降到 4.86µs（e2e 24.2%），收益真实但本轮预算优先给 flat 双域（7 个 bench 用例）。**留作 wave-2**（同机制、同宿主改造，可直接复用本轮基建） |
| prefill_expert | c19（G=1 MHA 退化） | 同上（第 4 个 kernel） |
| g2stack（ns≥2 批次） | c8, c9, c10（kv=4/8）、l1_s_tail | 同上（第 5 个 kernel）；c8/c9/c10 的 tout 网格已健康（≥40 任务或 P1 重铺后 wave-1） |
| decode | c11–c17, r5 | 已是 BSND 直供直写（aux=0%），**无 tout 可消** |
| **非对齐 S（padded/crop 域）** | padrow_s100（S=100→S_padded=128）、S%64≠0 的一切 prefill | **crop 复杂度取舍（如实说明）**：BSND 直写的输出张量为 `[B, S_padded, N_q, D]`，取真实 S 行需 `out[:, :S, :, :]` —— 该视图**非连续**，`.contiguous()` 落到 fp16/bf16 ViewCopy d2d，**其二进制在部分 CANN 9.x SoC 包缺失（错误码 561103）**，正是 gqa.py:5486-5489 / 4187-4190 现有注释所规避的坑；替代方案（kernel 内逐行 guard 写 + 输出张量按真实 S 声明）需要把 FIN 的整盒写拆成行级 masked 写（每盒 32 次条件发射 + 尾行部分盒），既触 V 侧节拍又引入新代码路径，**风险/收益比在 wave-1 不划算**。故非对齐 S 继续走 `transpose_out(s_real=S)`（其 n_pad 模式已能在 kernel 内完成 crop，零 d2d） |
| padcol（S 对齐、S_kv 非对齐） | padcol_skv200（S=128, S_kv=200） | **在覆盖域内**（S 对齐；输出行数只依赖 S）→ Gate 0 必测（use_mask=True 的 kv2 域用例） |

派发核验纪律：monkeypatch 只替换两个 flat 工厂 + 条件跳过 transpose_out；域外工况在 ARM 下
必须（a）变体工厂**未被调用**、（b）chosen 与 BASE 相同、（c）最终输出与 BASE 逐位一致。

---

## 3. 布局映射（设计要点 a）

### 3.1 现行（BHSD + transpose_out）

main kernel FIN（gqa.py:3268-3271 / 3792-3795）：

```
Output_bhsd : [B, N_q, S_padded, D]          声明（out_idx=[4]，tilelang 分配）
写盒        : Output[bz, hq0+h, row0 : row0+half, :]     row0 = bx*block_M + vid*half
盒形态      : [half=32, D=128] 连续（行 stride = D）→ 8KB 单块
host 尾     : transpose_out(out_padded[, s_real=S]) → [B, S, N_q, D]（_transpose_v1 kernel）
```

### 3.2 P2（BSND 直写）

```
Output_bsnd : [B, S_padded, N_q, D]          声明（同一 out_idx=[4] 位置）
写盒        : Output[bz, row0 : row0+half, hq0+h, :]
盒形态      : [32, 128] 行 stride = N_q*D → DataCopyExtParams(blockCount=32,
              blockLen=128*sizeof(T), dstStride=(N_q*D − D)*sizeof(T))
host 尾     : 无（覆盖域 S_padded == S，张量本身即 [B, S, N_q, D] 连续输出）
```

### 3.3 地址公式与双向核对

元素 `(b, hq, s, d)`：

| | 偏移（元素） |
|---|---|
| BHSD（现行 main 写） | `((b*N_q + hq)*S_padded + s)*D + d` |
| BSND（P2 main 写） | `((b*S_padded + s)*N_q + hq)*D + d` |
| transpose_out 语义（`_transpose_v1`：x=(B*M, n_pad, R) → y=(B*N, M*R)） | 读 `x[b*Nq+hq, s, d]` 写 `y[b*S + s, hq*D + d]` = `((b*S+s)*N_q + hq)*D + d` ✓ 与 BSND 式逐项相同 |

**双向核对**：
- 与 transpose_out 现行语义：上表第 2、3 行代数恒等（M=N_q, N=S, R=D）→ 同一元素同一地址。
- 与 decode 先例：`_gqa_kernel_dense_expert` 声明 `Output: T.Tensor([B, S, N_q, D])  # BSND, real S`
  （gqa.py:1118）并写 `Output[bz, s, hq0+g : hq0+ge, :]`（:1450/:1479/:1487…）—— 同一 BSND
  张量声明、同一 (b, s, head-range, :) 索引序；decode 的盒在 s 上是单行（S≤4 的 s-major 连续盒），
  prefill 的盒在 s 上是 32 行 range、head 单值 → **stride 落在行维**（decode 落在 head 维）。
  两者都由同一 `copy_ub_to_gm` 模板承载（realdstN 参数化），probe 已实证 prefill 形态逐位正确。
- 实测核对：probe `--exp num --allshapes` 15/15 bit_exact（host 侧独立枚举 `_ids` 任务分解
  作参照，每盒带唯一 tag 值 seed+k → 任何地址错位/重叠/漏写都会被抓）。

### 3.4 尾行 / 越界（设计要点 b：crop 语义）

- 覆盖域 gate = `S % 64 == 0` → `S_padded == S`，行窗 `[row0, row0+32)` 恒在 `[0, S)` 内
  （row0 = bx*64 + vid*32，bx < q_blocks = S/64，vid ∈ {0,1}）→ **无 s_real 边界外的行可写**，
  无需 mask 写、无需范围裁剪。任务书"边界外的行不得写出"在覆盖域内**由 gate 结构性保证**。
- 逐位一致判据会捕获任何越界：ARM 与 BASE 的最终输出张量同形同址比较，若 ARM 写脏了任何
  行（含 padded 行）→ 不等。Gate 0 另设 padded 域用例（padrow_s100）验证**域外路径逐位不变**
  （即 gate 生效、非对齐 shape 未被误纳入）。
- head 维：`hq0+h` 中 hq0 = kg*G + gb*2 < N_q，h ∈ {0,1} → 恒在界内（stacked 双头，G 偶数由
  工厂 assert 保证）。

---

## 4. pacing / flag 协议不变性论证（设计要点 c）

**改动面**：仅 `emit_fin` 的一条 `T.copy` 目的地下标 + `main` 的 Output 声明形状。**零改动**：
任何 `T.set_flag` / `T.wait_flag` / `T.set_cross_flag` / `T.wait_cross_flag` 行、SEM_* / SIG_*
常量、槽深（ws_s/ws_p/ws_o 的 2 槽）、release token 数（单 token）、C/V 步序与拍序
（`G1(t+1)→G2(t)→Qload(t+2)` / `AO(jj)→FIN(jj)→SM(jj+2)`）、l0 parity、UB 任务奇偶状态槽、
`task_range`、`_ids`、L1/L0/UB/ws 几何。

生成器对每条 flag 行做**逐字节比对断言**（iter7 P5 生成器同款 audit：`flag_ship == flag_var`），
diff hunk 数 ≤ 2/kernel。

**为何 FIN 的 MTE3 发射时长变化不构成协议风险（论证）**：

1. **协议层**：flag 的 set/wait 序列与 id 归属不变 → FIFO 计数语义（iter4 finding 1）不变，
   有效门（SM(x)←G2(x−1) 等）不变；pending 深度不变（每盒 1 set + 1 wait，与 shipped 同）。
   **无死锁面**：改动不新增/删除任何 flag 事件。
2. **队列层**：MTE3 指令条数不变（probe 实测两臂 instructions 逐一相等），只有 dstStride 字段
   变化 → MTE3 队列**深度**不变，每指令的**排空时长**增加 7–17%（probe：+13…+30ns/盒）。
   FIN 的下一盒 V 拷贝需等 `MTE3→V` flag（即上一盒排空），故 V 拍内 FIN 段可能延长
   ≤2×30ns = 60ns/拍（2 盒）。
3. **节拍层**：V 拍长（pacer）实测 2.35µs/u（r4）–2.93µs/u（r3），FIN 段增量 60ns = **2.0–2.6%**
   的拍长上界；但 FIN 只在**任务尾**出现（kv2：每任务 1 次；kv1：每任务 1 次，即每 u 1 次），
   且 MTE3 与 vec 是不同的硬件队列 —— vec busy（真正的 pacer，62–64%）不受影响。
   probe 的 `aiv_scalar_wait_time` 两臂差 **+0.000µs** → 孤立流中该增量未转化为等待。
4. **判别手段（Gate 2 强制）**：wait-id 遥测对比。iter4 签名 = 释放/pacing 等待（C-id1、V-id3）
   显著迁移或膨胀。预注册判据见 candidate JSON：若 Σwait 膨胀 > 被消除的 tout TaskDur → 否决；
   若 V-id3 膨胀 >20% 且 e2e 收益 < 预测的 50% → 降置信并披露。
5. **与 iter7 P3 探针的教训关系（E2/E2b）**：那是 *pass 自动插入* 的搬运与 expert 手工 vid/flag
   协议冲突（vid-reduction 半行拆分）。本改动**不引入任何 pass 插入的搬运**——`T.copy` 是用户
   手写的、位于既有 flag 对之间、目的地下标改变而已；vid 约定（`row0 = bx*block_M + vid*half`）
   逐字保留，双 vid 各写各自的 32 行，BSND 下两 vid 的目标行区间依然不相交（stride 只改变
   行间距离，不改变行的归属）→ **无竞写面**。

---

## 5. UB 缓冲复用 / liveness（设计要点 d）

FIN 段使用的 UB buffer 与其 liveness（flat kv1/kv2 相同）：

| buffer | 形状/dtype | 字节 | FIN 中角色 | 稳态拍内其他段的读者/写者 | 冲突？ |
|---|---|---|---|---|---|
| `acc_o` | [4, half, D] fp32 | 64KB | 读 `acc_o[par*2+h]`（被 div 原地归一化） | emit_ao 写同槽（上一拍）；下一任务的 emit_ao 头部 fill | 否：AO(par)→FIN(par) 同拍串序，slot 由任务奇偶双缓冲隔离（iter5 §B.4 推导） |
| `buf_2d` | [half, block_N] fp32 | 16KB | 写（broadcast sumexp） | emit_sm 的 broadcast/axpy、emit_ao 的 broadcast | 否：V 侧串行流内 produce→consume 紧邻，无跨拍存活 |
| `acc_s_half` | [half, block_N] dtype | 8KB(fp16) | 写（V 拷贝 acc_o→此）→ MTE3 读 | emit_sm 尾部同一 buffer 承载 ws_p 写（`T.copy(work_ub, acc_s_half)` → `T.copy(acc_s_half, ws_p[...])`） | **需 flag 保护，已存在**：两处都用 `SIG_S_HALF` 的 V↔MTE3 对（先 wait MTE3→V 再写，写完 set V→MTE3；MTE3 前 wait V→MTE3，写完 set MTE3→V）→ 复用被同一 flag 对完全序化 |
| `sumexp` | [4, half, 1] fp32 | 512B | 读+原地 add 1e-30 | emit_ao 写、emit_sm 的 sumexp_is | 否（slot 同 acc_o） |

**P2 不新增任何 UB 分配、不改变任何 buffer 的形状/生命周期**：目的地是 GM（Output 张量），
UB 侧 footprint 逐字节不变（生成器 audit 断言 `alloc_ub` 行全等）。`acc_s_half` 的双用
（ws_p 段 / Output 段）在 shipped 代码里已由 `SIG_S_HALF` 单 flag 对序化，P2 只改第二次使用的
**GM 目的地寻址**，不触碰该 flag 对 → liveness 结论不变。

（L2 侧附带效应：消除 tout 后，Output 张量不再经历"写 BHSD → 读 → 写 BSND"的三遍 L2/HBM
访问，r4 类省 32MB r+w 的 L2 占用；方向为正、不纳入判据。）

---

## 6. 宿主侧改造（最小化集成方案，Gate 3 用）

1. `_gqa_kernel_prefill_g2stack_flat` / `_kv2` 增 kwarg `out_bsnd=False`：
   - `False` → 现行 BHSD 声明与写盒（**逐字节不变**，域外/回退路径零影响）；
   - `True` → BSND 声明 + strided 写盒。
   （kernel 体内以 `if out_bsnd:` 分支只包住 Output 声明与 emit_fin 的目的地表达式。）
2. `gqa_expert` prefill 分支：`bsnd_ok = (S_padded == S)`（对齐 gate）；两个 flat spec 的
   builder 传 `out_bsnd=bsnd_ok`；chain_key 追加该标志（避免 factory 调用者与 adapter
   共享缓存条目导致布局歧义）；`_run.out_bsnd = bsnd_ok and chosen in ("g2stack_flat",
   "g2stack_flat_kv2")`（编译回退到 g2stack 时自动为 False → 语义安全）。
3. `_prefill_run_padded` 增 `out_bsnd` 参数：BSND 时 `crop` 分支改为 dim-1 语义
   （覆盖域恒 `S_padded == S`，assert 保护，永不物化 crop）。
4. `gqa()` adapter：`if getattr(kernel_fn, "out_bsnd", False): return out`（跳过 transpose_out）；
   否则原样（含 `s_real` 分支）。
5. `transpose_out` / `out_plan` / `_retile_out_plan` **零改动**（P1 重铺对域外用例继续生效）。

---

## 7. diff 摘要（变体，Gate 0/2 用；monkeypatch，不动 gqa.py）

`gen_variant_iter8_p2.py`（md5 锚定 + 行级断言 + 静态 audit）从 shipped gqa.py 抽取两个 kernel
块（flat kv1 = 2981–3439，flat kv2 = 3456–4027），各施加 **2 处最小编辑**：

| # | 位置（flat kv1 / kv2） | shipped | P2 变体 |
|---|---|---|---|
| 1 | 3280 / 3804（`main` 签名 Output 声明） | `Output: T.Tensor([B, N_q, S_padded, D], dtype),  # type: ignore  BHSD` | `Output: T.Tensor([B, S_padded, N_q, D], dtype),  # type: ignore  BSND` |
| 2 | 3270 / 3794（`emit_fin` 写盒目的地） | `Output[bz, hq0 + h, bx*block_M + vid*half : …+half, :],` | `Output[bz, bx*block_M + vid*half : …+half, hq0 + h, :],` |

audit 断言（生成时强制）：flag 行集合逐字节相等；`alloc_ub/alloc_L1/alloc_L0A/L0B/L0C`、ws_*
声明、`task_range`、`_ids`、C scope 全体、emit_qload/g1/g2/sm/ao 全体不变；diff hunk 数 = 2/kernel；
`ast.parse` 通过。产出 `gqa_flat_bsnd_iter8_variant.py` + `diff_iter8_p2.txt`。

宿主侧 ARM 语义（harness）：变体工厂替换 `G._gqa_kernel_prefill_g2stack_flat{,_kv2}` +
`G.transpose_out` 替换为"BSND 已就位则原样返回、否则调用原函数"的条件跳过（= 集成后
`if getattr(kernel_fn,'out_bsnd',False)` 的等价行为），双缓存清理（iter5 finding 2），
每工况断言 `chosen` 与"变体工厂是否被调用"（域外必须未被调用）。

---

## 8. 产物

- 本设计文档：`perf_tuning/p2_bsnd_design_iter8.md`
- 探针：`perf_tuning/probe_iter8_P2_bsnd.py`（num/rate 两实验）+ 生成源
  `probe_iter8_P2_num_{bhsd,bsnd4d,bsnd2d}_source.cu`
- 探针数据：`perf_tuning/msprof_op_iter8_PR_{r2,r6,r3,r4}_{bhsd,bsnd4d}_s{1,2}`（16 目录，rc=0，
  全 1650/1650MHz）+ `parse_iter8_probe_rate.py` + `probe_iter8_rate_summary.json`
  + `/tmp/opencode/iter8_probe_rate.log`
- 派发普查：`perf_tuning/census_iter8_dispatch.py`（8 处 gqa.py 行锚定 + md5 断言）
- 判据预注册：`perf_tuning/candidate_iter8_P2.json`（`criteria_PREREGISTERED`，测量前锁定）

---

## 9. 实测结果附录（Gate 0/2/3 完成后回填；判据未移动）

### 9.1 Phase A 模型 vs 实测（同会话交替 3+3，全 1650/1650MHz，42/42 rc=0）

| 工况 | 角色 | BASE e2e | P2 e2e | **实测改善** | Phase A 预测 | 误差 | 分布 |
|---|---|---|---|---|---|---|---|
| r2 | **PRIMARY** | 45.00 | **37.16** | **+17.42%** | −17.0% | +0.4pp | 不重叠（44.36 > 38.12） |
| r6 | **PRIMARY** | 48.45 | **39.04** | **+19.42%** | −19.5% | −0.1pp | 不重叠（47.52 > 39.40） |
| r3 | 覆盖-次判据 | 97.69 | 79.88 | **+18.23%** | −17.1% | +1.1pp | 不重叠 |
| r4 | 覆盖-次判据 | 246.89 | 217.16 | **+12.04%** | −11.6% | +0.4pp | 不重叠 |
| c7 | 覆盖-次判据 | 54.75（首测） | 47.69 | **+12.91%** | −14.0% | −1.1pp | 不重叠 |
| r1 | 域外护栏 | 20.73 | 20.57 | +0.80% | ~0 | — | 重叠（未动） |
| r5 | 域外护栏 | 146.15 | 146.17 | −0.01% | 0 | — | 重叠（未动） |

**模型精度**：五个覆盖工况预测误差 −1.1..+1.1 个百分点（probe 标定的 strided 成本 + tout 消除额
两项相加即命中实测），说明"probe 实测 ns/box × 盒数"的成本标定法在本平台可用。

### 9.2 成本项实测 vs probe 预测（main kernel TaskDur）

| 工况 | main BASE | main P2 | Δ% | probe 预测增量 | 实测增量 |
|---|---|---|---|---|---|
| r2 | 27.05 | 27.03 | **−0.07%** | +0.240µs | −0.02µs |
| r3 | 61.25 | 61.06 | **−0.30%** | +0.785µs | −0.19µs |
| r4 | 189.24 | 188.73 | **−0.27%** | +0.978µs | −0.51µs |
| r6 | 28.11 | 27.83 | **−1.00%** | +0.130µs | −0.28µs |
| c7 | 35.13 | 35.93 | **+2.28%** | +0.240µs | +0.80µs |

**发现（模型未预测到的方向）**：4/5 覆盖工况 main **反而更快** —— strided 写的代价被"输出张量不再
写 BHSD 再被 tout 读回"省下的 L2 一趟写+读抵消有余。probe 的孤立流测量给出的是**成本上界**，
真实 kernel 中该成本被 L2 压力下降部分/全部回收。

### 9.3 wait-id 判别门（强制；iter4 pacing 签名检查）

| 工况 | Σwait Δ | 被消除 tout | V-id3 Δ% | C-id1 Δ% | 判定 |
|---|---|---|---|---|---|
| r2 | −0.15µs | 7.89µs | −0.9% | −2.0% | PASS |
| r3 | +0.18µs | 17.83µs | −0.7% | +0.3% | PASS |
| r4 | −0.23µs | 29.55µs | −0.4% | −0.9% | PASS |
| r6 | +0.10µs | 9.65µs | +0.8% | −1.2% | PASS |
| c7 | +0.99µs | 7.72µs | −0.4% | +4.6% | PASS（最大膨胀 = 收益的 13%，阈值 100%） |

**pacing 未受扰**：镜像对（C-id1 ↔ V-id3）平直，无 iter4 式"等待迁移到生产门"签名。§4 的不变性
论证被实测支持（MTE3 指令条数不变 + FIN 不触 flag/槽/步序）。

### 9.4 Gate 3 集成后核验（零 monkeypatch，集成文件本体）

| 项 | 结果 |
|---|---|
| r2 链内实例 | `kernel_kernel:0`(tqkv) + `main_kernel:0` **两实例，tout 消失**；e2e 37.46µs（Gate 2 ARM 37.16，+0.8%） |
| r4 链内实例 | 同上，tout 消失；e2e 216.62µs（ARM 217.16，−0.2%） |
| r1（域外） | tout **仍在** 5.26µs blk=16（P1 重铺存活）；e2e 20.45µs |
| r5（decode） | main-only 145.36µs（锚 145.89/146.81/147.86，带内） |
| 全套件 | 18 PRECISION_PASS + 6 BOUNDARY_PASS = **24/24，exit 0，零 fallback** |
| 自检 | `python3 examples/gqa/gqa.py` → **Test Passed!**（其 prefill_causal 用例即 bench c1 = 覆盖域，viol=0） |
| 逐位 pre/post | **16/16**（final 全部逐位不变；域外 main BHSD 输出逐位不变 7/7；覆盖域 `transpose_out(main_pre) == main_post` 7/7；out_bsnd 能力位恰在 7 个覆盖 shape 为 True；plan 普查域外逐一不变） |
| ruff | 0.15.14 + 0.16.7 四门禁 exit 0（最长行 140 = 上限） |
| md5 | 4b4e453e4b8f2922baefbecb85912494 → **3aa9cfa2625cdce96a62ce967785fe85**（备份 history_version/gqa_impl_s3_iter8.py） |

### 9.5 session-health 披露（唯一未过的子判据，不移动球门）

预注册"BASE 复现 iter7 锚点 ±3%"在 **r1 = +3.41%**（20.73 vs 20.05）越界 → 按判据报 **FAIL** 并调查：
r1 是**域外** per-head 链（P2 零触碰：变体未调用、tout 仍 launch 同 blk=16、main +0.32%、同会话
A/B +0.80% 分布重叠）；偏差全部落在其 tout 分量（5.49µs vs iter7 A/B 4.86µs），而 **iter7 自己
Gate 3 对同一文件重采得 5.20µs**、本会话内 6 个 BASE 样本 tout 跨 **4.80–5.62µs** → 0.5MB、
launch-地板主导的微 kernel 具 ~±10% 跨会话带（iter3 §4：地板本身 2.5–3.8µs 波动）。诊断样本
（`msprof_op_iter8_P2diag_r1_{base,p2}_s{4,5,6}`，单独报告不并入决策数）：BASE 6 样本均值
20.64µs = **+2.9%（回到带内）**，P2 臂 20.68µs（−0.68%，仍"未动"）。全部 42 决策 run 额定频率、
rc=0、零探针断言失败；5 个决策相关工况锚点全在 ±3% 内 → **判定：最小工况的跨会话方差伪影，
非会话缺陷，且不影响 P2 结论**（r1 判据用同会话交替样本）。
