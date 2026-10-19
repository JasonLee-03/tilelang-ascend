# R4 fused sampling — Stage 3 **round 2** 优化记录 (optimization_log_round2.md)

> 本文件是 **round 2** 的日志，**追加**于 round 1 产物（`optimization_log.md` / `baseline.json` /
> `final_report.md`）之外，不覆盖任何 round 1 文件。
>
> round 2 基线 = **round 1 收敛值**，见 `baseline_round2.json`
> （kernel sha256 `ac8eba47f77e11342eabb9c4e30a44531b3e655b481b0dc28fd5a66a2738fc5b`）。

- 日期：2026-10-10 · 硬件：Ascend910_9362 (910b2, cube_core=20 / vec_core=40) · CANN 9.2.0
- tilelang：`0.1.4+d75300ed0a88b0f4b2eef8a62b43f1930fdae44b`（wheel，`/home/developer/.local/lib/python3.12/site-packages/tilelang/__init__.py`）
- 本轮范围（严格限定）：**只做优化点 #3（BV tile size 调优）**，未实施 split-V / K1 剪枝 / UB→UB 拷贝消除
- 算子类型：**纯 Vector (AIV)**（round 1 Step 2 判定，本轮未变；三 kernel `gemm/mma/cube 使用数=0`）
- 噪声阈：**3%**（DESIGN 未覆盖，沿用 round 1）
- 性能口径：**msprof**（cann-bench `kernel_details.csv`），**未使用任何 Python/torch 计时**（禁令 2）

---

## 0. 本轮结论（先给结论）

**`[ITEM3_BLOCKED]`** — 优化点 #3「BV tile size 调优」**在本算子的交付边界内不可实施**，
根因不是性能、也不是精度，而是 **BV 不是 kernel 侧的自由调优参数**：它由 cann-bench 任务夹具
（`cases.yaml` / `cases.csv`）的 `case.attrs` 按关键字传入，kernel 内的 `BV=4096` 默认值
（`r4_fused_sampling.py:545`）在**所有被测量路径上都是死代码**。

- **kernel 未做任何修改**：sha256 before == after == `ac8eba47…`（未产生需要回滚的改动，故本轮无回滚动作）。
- 静态 + 实测筛选结论（详见 §3/§4）：候选 `{4608, 5120}` **UB 可编译但余量仅 156 B / 92 B**；
  `6144` **K2g 编译期硬失败**（实测复现 `LOG(FATAL) Memory allocation failed for: tmp_ub required: 49152, new memory available: 48704`）。
- **TAIL 精度红线（§4）54/54 组全部合规** —— 红线**不是**淘汰因素；淘汰因素是 UB 预算 + attrs 阻断。
- 四项门禁对**交付版本（未改动的 ac8eba47）**全部复跑通过（§8）。
- 如实声明：**本轮没有产出任何性能数字对比**，因为不存在可被测量的 BV 变更（§2）。
  未测项、验证缺口见 §9。

---

## 1. Part A 补充判定 —— 为什么 round 1 漏了 BV

round 1 的 Part A（`optimization_log.md` L27-99）列了 14 个候选项，L30 明确记载读过
`optimization-guide.md §2.11`（UB 预算优化），但**没有产出任何 BV 调优条目**。补充判定如下：

| 遗漏成因 | 证据 | 是否属「清单遗漏」 |
|---|---|---|
| **§2.11 被当作「约束」而非「调优维度」** | round 1 ORDER-PLAN L110-111 只写「逐项实施后各自验证 UB **不超限**（memory planning 编译期报错即超限）」——把 UB 预算用作**被动校验**，从未反向问「UB 有余量时是否应放大 tile」 | ✅ 是（方法论遗漏） |
| **DESIGN 把 BV 冻结为「用户指定常量」** | DESIGN §5.1 L1033「V01-a 固定 `BV=4096`（用户指定）」；§4.5 L930 给出（**错误的**）上界 `BV ≤ 6400`，L1786 R-E 记 `BV ≤ 6729` → 看起来像已定案的设计常量，不像可调项 | ✅ 是（但下面第 3 条说明它其实**不可调**） |
| **Part A 的候选项被隐式限定为「kernel 文件内的改动」** | round 1 全部 14 项都是 kernel 内实现级改动；而改 BV 的**唯一有效落点是任务夹具 `cases.yaml`**（不属 agent 可改范围，见 §2） | ❌ **不是遗漏，反而是正确的隐式排除** |
| **§4.5 的 UB 模型本身算错，掩盖了「BV 已到上限」这一事实** | §4.5 通式 `28·BV + 36·K_MAX + 72 ≤ 188416` 预测 BV=4096/K=256 → 123,976 B；**实测最差 case 为 188,004 B**（低报 64 KB / 34%）。因为模型低估，round 1 无从看出「BV=4096 已是可行域上界」 | ✅ 是（且是本轮新发现的设计文档缺陷，见 §5） |

**结论**：BV 未进 round 1 清单，一半是方法论遗漏（把 §2.11 只当约束），一半是**正确的隐式排除**
（BV 落点在夹具而非 kernel）。本轮把两半都用证据钉死：§2 证明「落点阻断」，§3/§4 证明
「即便落点可改，UB 预算也几乎没有上移空间」。

---

## 2. 阻断证据 —— BV 由 `case.attrs` 提供，kernel 默认值是死代码

### 2.1 调用链（逐跳给出文件行号）

```
cases.yaml / cases.csv  ── attrs: {need_min_p_sampling, K_MAX, BV}     (18/18 case 均含 BV 键)
        │
        ▼
kernel_eval/utils/param_builder.py:203-206
        attrs = getattr(case, 'attrs', None) or {}
        for name in attr_params:                      # 非 tensor 形参
            if name in attrs: params[name] = self._convert_value(attrs[name])
        │
        ▼
kernel_eval/eval/evaluator.py:437
        ai_result = self._run_ai_op_isolated(actual_ai_func, params, ...)   # 同一个 params dict
        │
        ▼
cann_bench_submission/cann_bench/r4_ops.py:23-38
        def r4_fused_sampling(..., need_min_p_sampling=False, K_MAX=256, BV=4096):
            return _r4_kernels.r4_fused_sampling(..., K_MAX=K_MAX, BV=BV, ...)   # BV 无条件显式转发
        │
        ▼
r4_fused_sampling.py:586/590/598   k1 = r4_topk_stats(B, V, K_MAX, BV, ...)     # 用传入值
                                   k2 = r4_mask_sample(B, V, BV)
                                   k2g = r4_gumbel_sample(B, V, BV)
```

`r4_fused_sampling.py:545` 的 `BV=4096` 默认值**永远不会被查询**：上游 `r4_ops.py:37` 无条件传 `BV=BV`。

### 2.2 端到端实证（用**真实 harness 代码**跑，非推断）

```
AI operator signature: (logits, top_ks, top_ps, min_ps, rand, need_min_p_sampling=False, K_MAX=256, BV=4096)
params built by harness for case 8 -> {'need_min_p_sampling': False, 'K_MAX': 128, 'BV': 2048}
=> BV passed to the operator = 2048   K_MAX = 128
=> kernel default BV=4096 (r4_fused_sampling.py:545) is OVERRIDDEN -> never used
```

（脚本：`ParamBuilder().build_call_params(ai_op, case, [None]*5)`，`ai_op` = 真实 submission 适配器。）

### 2.3 夹具里 BV **本来就是异构的**

| case | B | V | K_MAX | **BV（attrs）** |
|---|---|---|---|---|
| 8 | 4 | 16384 | **128** | **2048** |
| 13 | 6 | 2304 | **128** | **2048** |
| 其余 16 个 | — | — | 256 | 4096 |

⟹ 不存在「一个全局 BV」可调：16 个 case 是 4096，2 个 case 是 2048。
这两条正是覆盖门禁的 `D-PARAM-BV` 维度载体（`test_r4_fused_sampling.py:455/727/1143`、
`--level stress` 的 `stab_B7_V4409_bv2048_gumbel`、`gate_coverage.log` 的 `[PASS] D-PARAM-BV`）。

### 2.4 三条可能落点的逐条判定

| 落点 | 做法 | 判定 | 理由 |
|---|---|---|---|
| **(A)** 改 kernel 默认值 `r4_fused_sampling.py:545 BV=4096 → X` | 单行改动 | ❌ **可证明的 no-op** | §2.1/§2.2：attrs 覆盖 + `r4_ops.py` 无条件转发 + 测试文件也按位置传 BV（`test_…py:58 BV=4096`，L138/306/839 位置传参）→ **bench 与门禁两条测量路径都不受影响**，实测数字必然逐位不变。按本轮规则「未跑赢 3% 即回滚」，no-op 不可保留 |
| **(B)** 改 `cases.yaml` / `cases.csv` 的 `attrs.BV` | 改任务夹具 | ❌ **越界，禁止** | cann-bench README L143「`tasks/levelN/<op>/` 下**已包含** proto.yaml、golden.py、cases.yaml、cases.csv、desc.md，**无需另外生成**」+ L187 表格把 cases.yaml/csv 列为「**开发算子的输入文件**」。改夹具 = 改评测集定义：① 与 round 1 基线不可比（基线是在 attrs BV=4096/2048 下测的）；② 使已发布的 `t_hw_us` / 达成率失真（§6）；③ 属「改考卷而非改答卷」 |
| **(C)** 让 wrapper/adapter 静默忽略传入 BV，强制内部值 | 改 `r4_ops.py` 或 kernel 内部派生 | ❌ **契约违约 + 静默弱化断言** | `proto.yaml` 把 BV 声明为算子 attr（`type: Int, default: 4096`），且 `shape_support` 明文以 BV 表述：「要求 … 且 `V % BV == 0` 或 `V % BV >= K_MAX`」。静默改用自己的 BV 会让该声明失真；更严重的是 case 8/13 将**不再真正覆盖 `D-PARAM-BV`**，而覆盖门禁仍报 `25 PASS` → 违反本轮「不得静默弱化任何既有断言」 |
| **(D)** per-kernel BV（K1 用 attr 的 BV，K2m/K2g 用内部派生宽度） | 改 wrapper 签名（任务 §3 允许） | ⚠️ **技术上可行，本轮不实施** | 见 §7 的量化分析：K2m 仅占 mask 任务 ~8%，K2g 的成本几乎全是 O(V)（与 BV 无关），**预期收益 < 3% 噪声阈**；同时 (D) 仍会让 case 8/13 的 `D-PARAM-BV` 在 K2m/K2g 上失去覆盖（同 (C) 的弱化问题）。收益低于噪声 + 覆盖弱化 ⟹ 不实施，交 Orchestrator 决策 |

---

## 3. 静态 UB 账目（三 kernel × 各候选 BV）

#### 表 A — 三 kernel UB 账目（实测 = 从生成源 `ascend_ub.GetWithOffset<T>(n,off)` 取 max(off+n*sizeof(T))，即 planner 的 "Scope shared.ub memory usage"）

| BV | K2m (BV-only) | K2g (BV-only) | K1 worst-case (over 18 cases) | K1 worst case id | 三 kernel 最大占用 | 余量 | 18/18 可编译 |
|---|---|---|---|---|---|---|---|
| **4096** | 98912 (50.37%) | 148452 (75.61%) | 188004 (95.75%) | case6 V=10240 TAIL=2048 | **188004** (95.75%) | **8348 B** | 18/18 (round-1 已验证 + 本轮复测) |
| **4608** | 111264 (56.67%) | 166948 (85.02%) | 196196 (99.92%) | case8 V=16384 TAIL=2560 | **196196** (99.92%) | **156 B** | 18/18 |
| **5120** | 123616 (62.96%) | 185444 (94.44%) | 196260 (99.95%) | case18 V=65536 TAIL=4096 | **196260** (99.95%) | **92 B** | 18/18 |
| **6144** | 148320 (75.53%) | **编译失败** | — | — | — | — | **0/18（K2g 硬失败）** |

> UB 上限 196352 B = `ASCEND_SHARED_MEM_SIZE`（src/transform/ascend_memory_planning.cc:35）。
> 超限行为：`tl.ascend_memory_planning=True` → `PlanMemoryForScope` → `LinearScanAllocator::allocate` 在找不到可复用块时 `LOG(FATAL)`（ascend_memory_planning.cc:767-771）= **编译期硬失败**（本轮已实测复现）。
> ⚠️ 兄弟路径 `PlanMemoryForScopeLinear` 的 `check_overflow=false`（:598）是**静默溢出**；R4 的安全网依赖该 pass_config 保持开启。
### 3.1 模型及其**实证校验**（不是估算）

先给出解析模型，再用**生成源码实测**逐点校验；模型只在「planner 未发生 liveness 复用」时精确，
一旦发生复用即只是**上界**——因此可行性判定一律以实测为准（§3.2）。

```
UB_K1(BV,K_MAX,V) = 580 + 60·K_MAX + AlignUp(K_MAX/8,32)
                    + (16·BV if V//BV>0 else 0)                       # z_ub e_ub m2d zt_ub
                    + max(16·AlignUp(SRC,32), 32·K_MAX)               # 单一共享 topk tmp arena
                    + (3·AlignUp(4·TAIL,32) + 8·TAIL_PAD if TAIL>0 else 0)
                    + (64 if NEED_MIN_P else 0)                       # muc + lnmu
      其中 SRC = BV (V//BV>0) 或 TAIL_PAD;  TAIL = V % BV;  TAIL_PAD = AlignUp(TAIL,32)

UB_K2m(BV) = 24·BV + AlignUp(BV/8,32) + 96
UB_K2g(BV) = 36·BV + AlignUp(BV/8,32) + 484        # 3×f32[BV] + 3×i32[2BV] + mask + 8 标量 + 256B reduce arena
```

**校验结果（pred vs 从生成源解析的 max(off+size)）：**

| kernel | 配置 | 模型 | 实测 | |
|---|---|---|---|---|
| K1 | BV=4096 K=256 V=10240 (TAIL=2048) | 188004 | 188004 | ✅ |
| K1 | BV=4096 K=256 V=152377 (TAIL=825) | 163684 | 163684 | ✅ |
| K1 | BV=2048 K=128 V=16384 (TAIL=0) | 73828 | 73828 | ✅ |
| K1 | BV=4096 K=256 V=256 (n_blk_full=0) | 29284 | 29284 | ✅ |
| K1 | **BV=4608** K=256 V=10240 | 183908 | 183908 | ✅ |
| K1 | **BV=4608** K=256 V=152064 (TAIL=0) | 163428 | 163428 | ✅ |
| K1 | **BV=5120** K=256 V=152064 (TAIL=0) | 194148 | 194148 | ✅ |
| K1 | BV=4608 K=256 V=8192 (TAIL=3584) | 235108 | **192100** | ⚠️ 模型为上界（planner 复用了 zt/et/mt） |
| K1 | BV=4608 K=256 V=16384 (TAIL=2560) | 214628 | **194148** | ⚠️ 同上（复用 2/3 尾块缓冲） |
| K1 | BV=5120 K=256 V=32768 (TAIL=2048) | 220772 | **196196** | ⚠️ 同上 |
| K1 | BV=5120 K=256 V=152064 (TAIL=3584) | 251492 | **194148** | ⚠️ 同上 |
| K2m | BV=2048 / 4096 / 4608 / 5120 | 49504 / 98912 / 111264 / 123616 | 同 | ✅×4 |
| K2g | BV=2048 / 4096 / 4608 / 5120 | 74468 / 148452 / 166948 / 185444 | 同 | ✅×4 |

> ⚠️ **方法论纠正（对本轮自身）**：我最初用「无复用」模型判 BV=4608/5120 有 4/9 个 case UB 超限，
> 结论**错误**。实测显示 planner 在 BV>4096 时会复用已释放的尾块缓冲，18/18 均可编译。
> 因此本报告一切可行性结论均以**实测**为准，模型仅作解释与上界。（教训已记入 §5.3。）

### 3.2 实测 UB 账目（18 case 全量，BV=4608 / 5120）

完整逐 case 数据：`round2_bv_probe/bv_ub_measured.json`、日志 `round2_bv_probe/bv_ub_sweep.log`。
摘要：

| BV | 18 case K1 UB 区间 | K1 最差 | K2m | K2g | 三 kernel 最差 | 余量 |
|---|---|---|---|---|---|---|
| 4096（现行） | 29,284 … 188,004 | 188,004 (95.75%) | 98,912 | 148,452 | **188,004** | **8,348 B** |
| 4608 | 29,284 … 196,196 | 196,196 (99.92%) | 111,264 | 166,948 | **196,196** | **156 B** |
| 5120 | 29,284 … 196,260 | 196,260 (99.95%) | 123,616 | 185,444 | **196,260** | **92 B** |
| 6144 | — | — | 148,320 *(模型值；K2g 先硬失败故未实测)* | **编译失败** | — | — |

**BV=6144 淘汰的硬证据（实测复现编译期 `LOG(FATAL)`）：**
```
K2g BV6144 (pred 222436 > 196352): COMPILE FAILED -> TVMError
    Memory allocation failed for: tmp_ub required: 49152, new memory available: 48704
```

### 3.3 可行域全扫描（BV ∈ [1024, 8192]，步长 32）

用**无复用上界模型**扫描得到的「全 18 case 三 kernel 均不超限」集合（保守，真实可行域略大）：

```
1024, 1184, 1376, 1536, 1568, 1664, 1728, 1760, 1824, 1888, 2048, 2304, 2368, 2400, 2432,
2464, 2560, 2592, 2752, 2784, 2816, 2880, 2912, 2944, 3008, 3072, 3104, 3136, 3168, 3200,
3296, 3328, 3424, 3456, 3488, 3520, 3552, 3584, 3744, 3776, 3840, 4096
```

两个关键观察：
1. **3840 → 4096 之间的 3872…4064 全部不可行**，原因是 **C3**（不是 UB）：V=4096 的 case 会得到
   `TAIL = 4096 − BV ∈ [32, 224] < K_MAX = 256` → host wrapper `r4_fused_sampling.py:581-582`
   直接抛 `NotImplementedError`。BV=4096 恰好让 V=4096 的 `TAIL == 0`，是这段「C3 空洞」的上沿。
2. 实测把可行域上沿从模型的 4096 推到 **5120**，但余量只有 **92 B**（0.05%）——
   任何后续 kernel 改动（哪怕 +64 B 的 `NEED_MIN_P` 缓冲）都可能翻车。**不是可交付的工程余量。**

---

## 4. TAIL 精度红线逐 case 核算（强制交付物）

#### 表 B — TAIL 红线逐 case 核算（18 case × 3 候选 BV = 54 组，全部取自**生成源码实证**）

每组 5 项：`TAIL / TAIL_PAD` · `padCount`（所有 TopK 调用）· UB→UB 拷贝字节数是否全为 32 倍数 · C3（`TAIL==0 or TAIL>=K_MAX`）· `K<=actual_num`。

| case | V | K_MAX | BV=4096: TAIL/TAIL_PAD · padCount · ub2ub字节 · C3 | BV=4608: 同 | BV=5120: 同 | 红线 |
|---|---|---|---|---|---|---|
| 1 | 256 | 256 | 256/256 · [0] · [1024] · ok | 256/256 · [0] · [1024] · ok | 256/256 · [0] · [1024] · ok | ✅ |
| 2 | 4096 | 256 | 0/0 · [0] · [1024, 16384] · ok | 4096/4096 · [0] · [1024, 16384] · ok | 4096/4096 · [0] · [1024, 16384] · ok | ✅ |
| 3 | 16384 | 256 | 0/0 · [0] · [1024, 16384] · ok | 2560/2560 · [0] · [1024, 10240, 18432] · ok | 1024/1024 · [0] · [1024, 4096, 20480] · ok | ✅ |
| 4 | 16384 | 256 | 0/0 · [0] · [1024, 16384] · ok | 2560/2560 · [0] · [1024, 10240, 18432] · ok | 1024/1024 · [0] · [1024, 4096, 20480] · ok | ✅ |
| 5 | 4409 | 256 | 313/320 · [0] · [1024, 1280, 16384] · ok | 4409/4416 · [0] · [1024, 17664] · ok | 4409/4416 · [0] · [1024, 17664] · ok | ✅ |
| 6 | 10240 | 256 | 2048/2048 · [0] · [1024, 8192, 16384] · ok | 1024/1024 · [0] · [1024, 4096, 18432] · ok | 0/0 · [0] · [1024, 20480] · ok | ✅ |
| 7 | 152064 | 256 | 512/512 · [0] · [1024, 2048, 16384] · ok | 0/0 · [0] · [1024, 18432] · ok | 3584/3584 · [0] · [1024, 14336, 20480] · ok | ✅ |
| 8 | 16384 | 128 | 0/0 · [0] · [512, 16384] · ok | 2560/2560 · [0] · [512, 10240, 18432] · ok | 1024/1024 · [0] · [512, 4096, 20480] · ok | ✅ |
| 9 | 4096 | 256 | 0/0 · [0] · [1024, 16384] · ok | 4096/4096 · [0] · [1024, 16384] · ok | 4096/4096 · [0] · [1024, 16384] · ok | ✅ |
| 10 | 4096 | 256 | 0/0 · [0] · [1024, 16384] · ok | 4096/4096 · [0] · [1024, 16384] · ok | 4096/4096 · [0] · [1024, 16384] · ok | ✅ |
| 11 | 8192 | 256 | 0/0 · [0] · [1024, 16384] · ok | 3584/3584 · [0] · [1024, 14336, 18432] · ok | 3072/3072 · [0] · [1024, 12288, 20480] · ok | ✅ |
| 12 | 32768 | 256 | 0/0 · [0] · [1024, 16384] · ok | 512/512 · [0] · [1024, 2048, 18432] · ok | 2048/2048 · [0] · [1024, 8192, 20480] · ok | ✅ |
| 13 | 2304 | 128 | 2304/2304 · [0] · [512, 9216] · ok | 2304/2304 · [0] · [512, 9216] · ok | 2304/2304 · [0] · [512, 9216] · ok | ✅ |
| 14 | 1024 | 256 | 1024/1024 · [0] · [1024, 4096] · ok | 1024/1024 · [0] · [1024, 4096] · ok | 1024/1024 · [0] · [1024, 4096] · ok | ✅ |
| 15 | 152064 | 256 | 512/512 · [0] · [1024, 2048, 16384] · ok | 0/0 · [0] · [1024, 18432] · ok | 3584/3584 · [0] · [1024, 14336, 20480] · ok | ✅ |
| 16 | 32768 | 256 | 0/0 · [0] · [1024, 16384] · ok | 512/512 · [0] · [1024, 2048, 18432] · ok | 2048/2048 · [0] · [1024, 8192, 20480] · ok | ✅ |
| 17 | 152377 | 256 | 825/832 · [0] · [1024, 3328, 16384] · ok | 313/320 · [0] · [1024, 1280, 18432] · ok | 3897/3904 · [0] · [1024, 15616, 20480] · ok | ✅ |
| 18 | 65536 | 256 | 0/0 · [0] · [1024, 16384] · ok | 1024/1024 · [0] · [1024, 4096, 18432] · ok | 4096/4096 · [0] · [1024, 16384, 20480] · ok | ✅ |

**54/54 组全部合规：True**

- `padCounts=[0]` 表示该特化中**所有** TopK 调用（块内 topk / 归并 topk / 尾块 topk）的 `padCount == 0`，
  即 `alignedCount == actualCount`，`tl::ascend::Sort()` 的 masked `AscendC::Duplicate` 修补分支（common.h:1731-1755，dav_c220 上不生效）**从不进入** → 缺陷 A 被结构性排除。
- `ub2ub_bytes` 为该特化中所有 `copy_ub_to_ub<T,T,N>` 的字节宽度集合，全部 %32==0（且 %8==0）→ 缺陷 B 的静默截断被排除。
- 结论：**TAIL 红线不是本轮的淘汰因素**；淘汰因素是 UB 预算（表 A）与下述 attrs 阻断。
### 4.1 红线的机制依据（本轮已核对源码，非引用 round 1 结论）

`T.tile.topk` 的真实生成签名（实测抓取）：
```
TopK<float>(l_dst[0], zt_ub[0], tmp_ub.ReinterpretCast<float>()[0], 256, 128, 4096);   # 块内
TopK<float>(m_dst[0], merged[0], tmp_ub.ReinterpretCast<float>()[0], 256,  16,  512);   # 归并
TopK<float>(l_dst[0], ztt[0],    tmp_ub.ReinterpretCast<float>()[0], 256,  10,  320);   # 尾块
                 (dst,          src,        tmp,                     K, repTimes, actualCount)
```
`tilelang/language/ascend_tile.py:646` → `repeatTimes = (max_actual_num + 31) // 32`，
`max_actual_num = sum(src.shape)`（**buffer** 尺寸，非 actual_num）；
`src/tl_templates/ascend/common.h:1712-1714` → `alignedCount = repeatTimes*32`，`padCount = alignedCount − actualCount`；
`common.h:1731-1755` → `padCount>0 且 actualCount % 8 != 0`（fp32 `elemPerBlock=8`）时走 **masked `AscendC::Duplicate`**，
该路径在 dav_c220 上不生效 = **缺陷 A**。

R4 的规避是**结构性**的：尾块 topk 的 src 是 `ztt[TAIL_PAD]` 且 `actual_num = TAIL_PAD`，
于是 `max_actual_num == actualCount == TAIL_PAD`，而 `TAIL_PAD % 32 == 0` ⟹ `repeatTimes*32 == TAIL_PAD`
⟹ **`padCount ≡ 0`，与 BV 取值无关**。块内 topk（src `zt_ub[BV]`, actual `BV`）同理，只要 `BV % 32 == 0`。

`ascend_tile.py:653-654` 另有 `K <= actual_num` 校验（wheel 版本已含），表 B 的 `K<=act` 列即其证据。

### 4.2 判定

- **红线 1（TAIL 变化）**：改 BV 确实改变全部 18 case 的 TAIL（表 B 可见，如 case 17：825 → 313 → 3897），
  但 `TAIL_PAD` 恒为 32 倍数 ⟹ `padCount ≡ 0` ⟹ 缺陷 A 分支**从不进入**。54/54 组合规。
- **红线 2（BV % 32 == 0）**：4096 / 4608 / 5120 / 6144 全部满足（`mask[1,BV//8]` 需 %8，topk 对齐需 %32）。
- **红线 3（UB ≤ 196352）**：见 §3.2 实测账目 —— 4096 ✅ / 4608 ✅(余 156 B) / 5120 ✅(余 92 B) / **6144 ❌ 编译失败**。
- **缺陷 B（UB→UB 静默截断）**：表 B `ub2ub字节` 列显示所有 `copy_ub_to_ub<T,T,N>` 的字节宽度
  （含最刁钻的 case 17 `TAIL=825 → TAIL_PAD=832 → 3328 B`、case 5 `313 → 320 → 1280 B`）**全部 %32 == 0**。54/54 合规。

⟹ **精度红线不是本轮的淘汰因素。** 淘汰因素是 §3（UB 余量趋零 / 6144 硬失败）与 §2（attrs 阻断）。

---

## 5. 本轮发现的既有文档/流程缺陷（如实上报，非本轮改动）

### 5.1 DESIGN §4.5 的 K1 UB 模型低估 34%

| 项 | DESIGN §4.5 (L880-930) | 实测 | 差异根因 |
|---|---|---|---|
| BV 系数 | `12·BV`（注「z_ub + zt_ub + e_ub」） | **`16·BV`** | **漏算 `m2d[1,BV]`**（`r4_fused_sampling.py:194`） |
| K_MAX 系数 | `36·K_MAX` | **`60·K_MAX`** | 漏算 tie_m / tie_one / run_v（各 `4·K_MAX`，其中 tie_one/tie_tkbc 是 round 1 [#1] 新增） |
| 常量 | `72` | **`580 (+64 if NEED_MIN_P)`** | 按 `alloc_var` 计；实际全为 `alloc_shared`，每个 1 元素缓冲占 `AlignUp(4,32)=32 B` |
| 尾块缓冲 | **完全遗漏** | `3·AlignUp(4·TAIL,32) + 8·TAIL_PAD` | V01-b rev2 的 `zt/et/mt/ztp/ztt` 是 §4.5 之后加的 |
| UB 上限 | `188416`（184 KB） | **`196352`** | `ascend_memory_planning.cc:35 ASCEND_SHARED_MEM_SIZE = 196352` |
| **BV=4096,K=256 预测** | **123,976 B** | **最差 case 188,004 B** | **低报 64,028 B (34%)** |
| 推出的 BV 上界 | `BV ≤ 6400`，「实用上界 6144」 | **6144 编译期硬失败（K2g）** | 上界结论错误 |

### 5.2 DESIGN §4.5 **完全没有 K2g 的 UB 账目**

§4.5 只给了 K1 与 K2m 两张表。实测 K2g = `36·BV + BV/8 + 484`：

| BV | K2g | 占 UB |
|---|---|---|
| 2048 | 74,468 | 37.9% |
| 4096 | **148,452** | **75.6%** |
| 4608 | 166,948 | 85.0% |
| 5120 | 185,444 | 94.4% |
| 6144 | **222,436 → 编译失败** | 113.3% |

⟹ **Orchestrator 在任务书 §3 的预判（「K2g 很可能比 K1 更紧」）部分成立**：
K2g 在 BV=4096 时占 75.6%，是**决定 6144 不可行的那个 kernel**（K1 靠复用还能挤进去，K2g 不能）；
但在 4608/5120 处，**最紧的是 K1**（99.92% / 99.95%），不是 K2g（85.0% / 94.4%）。
另外 round 1 [#2] 给 K2g 加的 `idx[BV]` 确实存在（实测布局中 `idx` 占 16,384 B @off=16576），
但它**复用了 dead 的 `g_ub`**作为 masked-index 缓冲，故净增是 `idx` 的 16 KB 而非 32 KB
（round 1 log L110 写的「idx[BV]+sel[BV] ≈ +32KB」高估了一倍）。

### 5.3 round 1 log L111 的「编译期报错即超限」结论**有条件成立**

`ascend_memory_planning.cc` 有两条规划路径：
- `PlanMemoryForScope`（`:544`）→ `LinearScanAllocator::allocate` 找不到可复用块时 `LOG(FATAL)`（`:767-771`）= **硬失败** ✅
- `PlanMemoryForScopeLinear`（`:596`）→ `bool check_overflow = false; // reserve memory overflow check`（`:598`）= **静默溢出** ⚠️

选择开关是 `memory_auto_plan = ctx->GetConfig<Bool>(kAscendMemoryPlanning, false)`（`:59-60, :320-323`），
即 pass_config `tl.ascend_memory_planning`。R4 的 `R4_PASS_CONFIGS` 把它设为 `True`（`r4_fused_sampling.py:40`）
→ 走硬失败路径，round 1 的结论成立。**但这是一条依赖 pass_config 的安全网**：若将来（如 round 1 [#8] 讨论的
pass_configs 调整）关掉它，UB 超限会变成**静默内存重叠**而非编译错误。已记录，供 Orchestrator 评估 [#8] 时参考。

---

## 6. `t_hw` roofline 模型是否依赖 BV —— **是（已核实）**

任务书 §9.1 要求核实。结论：**依赖，且依赖被硬编码进闭式常数**。

| 证据 | 位置 | 内容 |
|---|---|---|
| BV 是 t_hw 的语义字段 | `bench/hap_thw_model/r4_thw_common.py:219` | `_CASE_SEMANTIC_FIELDS = ("B","V","K_MAX","BV","need_min_p", …)` |
| BV 从 attrs 读入模型 | 同文件 `:234` | `"BV": int(attrs["BV"])` |
| 分块几何直接由 BV 算出 | 同文件 `:310-328` | `tail_pad_split(V,BV)` / `block_split(V,BV)` → `n_blk_full, TAIL, TAIL_PAD, n_blk, K1HW = n_blk_full*BV + TAIL_PAD` |
| **每元素 op 常数假设 `K_MAX/BV == 1/16`** | `compute_r4_fused_sampling.py:95-98`（`compute_r4_mask_sample.py:128-130` 同） | 「they add a constant `4*K_MAX/BV = 4*(1/16) = 0.25 ops/elem`（**K_MAX/BV is 1/16 for BOTH specializations: 256/4096 and 128/2048**），plus `6/BV` … and `2/BV` … Closed-form asymptote … `45.2519531 ops/elem = 8.0419922 ps/elem`」 |

用真实模型代码实测的 BV 敏感性（`tail_pad_split`）：

| case | V | BV=4096 `n_blk` | BV=4608 `n_blk` | BV=5120 `n_blk` |
|---|---|---|---|---|
| 15 | 152064 | 38 | 33 | 30 |
| 17 | 152377 | 38 | 34 | 30 |
| 18 | 65536 | 16 | 15 | 13 |
| 6 | 10240 | 3 | 3 | 2 |
| 11 | 8192 | 2 | 2 | 2 |

⟹ 若 BV 改为 4608（`K_MAX/BV = 1/18`）或 5120（`1/20`），已发布的 `t_hw_us` 与**所有 roofline 达成率数字都会失真**
（模型会高估 per-block 项），必须重跑 `hap_thw_model/` 重新推导并同步
`bench/metadata_staging/t_hw_910b2*.json` + active metadata。**本轮未做该重推导**（因为未采纳任何 BV 变更，无此需要）。

---

## 7. per-kernel BV（落点 D）的量化评估 —— 不实施的理由

任务书 §3 允许改 wrapper 签名做 per-kernel BV。量化后判定**收益低于噪声阈**：

| kernel | 在任务中的占比（round 1 实测，case15 B=64 V=152064） | 成本对 BV 的依赖 | BV 4096→5120 的预期收益 |
|---|---|---|---|
| K1 | mask 92% / fused 46% | **必须**遵守 attrs 的 BV（C3 / `shape_support` 就是 K1 topk 的约束） | 0（不可动） |
| K2m | mask **8%**（74.76 / 985.92 us） | 纯逐元素 O(V)；块数 `ceildiv(V,BV)` 只影响循环开销 | ≤ 20% × (每块固定开销占比) ≪ 3% × 8% ≈ **0.2% of mask 任务** |
| K2g | fused **54%**（1059.58 / 1974.21 us，round 1 前数据；[#2] 后更低） | hash 链在 `count = 2*BV` 上跑（`arith_progression(state_ub, …, BV2)`）、gumbel/compare/select/reduce 全 O(BV) ⟹ **总量 O(V)，与 BV 无关**；只有 per-block 固定开销（标量 merge、循环头）随块数下降 | 块数 −20% × 很小的固定开销 ⟹ **预期 < 3%** |

补充：K2g 真正的浪费是 **int32 `And` 的 int16-native「halving」workaround 导致的 `2*BV` 过量分配**
（`r4_fused_sampling.py:409-410, 437-440, 472`）——每块对 `2*BV` 个 int32 做 ~20 条向量指令，
其中一半是无效数据。这是**常量因子 2× 的浪费**，改 BV **不能**消除它（它与 BV 成正比）。
它属任务书明令本轮不得触碰的范围（「不动 UB→UB 拷贝」/「不做 K1 剪枝」之外的独立优化点），
本轮**未实施**，仅记录为后续候选（记作 round 2 的 #5 候选，非本轮范围）。

⟹ (D) 的期望收益 < 3% 噪声阈，且会带来 §2.4(C) 同类的 `D-PARAM-BV` 覆盖弱化。**不实施。**

---

## 8. Step 4 — 逐项实施（本轮 #3）

[ORDER-CHECK] 准备实施: [#3] BV tile size 调优 | 前置依赖: [round 1 [#1] ✅ 已采纳, [#2] ✅ 已采纳；本轮不涉及 split-V] | 结论: [❌ 不实施 — 落点阻断]
- 依据 `optimization-guide.md §2.11`（UB 预算优化，Vector 核 tile size 首选方法）：该章方法论要求「在 UB 预算内放大 tile 以减少块数/循环开销」。
- 本轮按该方法论完成了完整的可行性论证（§3 UB 账目 + §4 红线 + §6 t_hw 连带影响），**结论是落点不存在**（§2）：
  BV 由 `case.attrs` 提供，kernel 默认值是死代码；改夹具越界；静默覆盖 attr 违约并弱化 `D-PARAM-BV` 覆盖。
- 调用域判定：本轮**未修改任何 kernel 代码**，调用域未变 ⟹ 无需 [KERNEL-REUSE-PRECHECK] / [KERNEL-REUSE-AUDIT]。
- 非 Double Buffer 类 ⟹ 无需 [DB-ANALYSIS]。
- 反模式对照（`performance-antipatterns.md`）：round 1 已完成全量对照并记录 14 项判定；本轮**未引入新代码**，
  反模式状态与 round 1 收敛时一致（唯一保留项为末项「纯 Vector 算子的 AUTO_CV_COMBINE 空转 L0A/L0B」，
  round 1 [#8] 已写明保留原因：并发度不变、L0A/L0B 属 Cube 侧不占 Vector UB、pass_configs 为最后手段）。
  **本轮新增保留理由**：§5.3 表明关掉该 pass_config 族中的 `tl.ascend_memory_planning` 会把 UB 超限从
  「编译期硬失败」降级为「静默内存重叠」，而现行 BV=4096 的 UB 余量仅 8,348 B（95.75% 占用）
  ⟹ 在当前 UB 压力下调 pass_configs 的风险高于 round 1 的评估，**继续保留**。

[IMPL-#3] 已阅读（本轮实际查阅的文档/源码，均给出行号）：
- `optimization-guide.md §2.11 UB 预算优化`（Vector 核 tile size 首选方法）——本轮方法论来源
- `performance-antipatterns.md`（全文，round 1 已对照；本轮复核「有意保留项」）
- `tilelang/language/ascend_tile.py:599-670`（`topk` 签名、`repeatTimes`/`max_actual_num` 推导、`K<=actual_num` 校验）
- `src/tl_templates/ascend/common.h:1682-1760`（`Sort`：`alignedCount`/`padCount`/masked `Duplicate` 分支）
- `src/transform/ascend_memory_planning.cc:35, 59-60, 114-118, 305-324, 544-648, 667-780`（UB 上限、两条规划路径、`LOG(FATAL)` 条件）
- `src/target/codegen_ascend.cc:38`（`ASCEND_A2A3_UB_SIZE = 196352`）
- `.agents/skills/cann-bench/src/kernel_eval/utils/param_builder.py:125-210`、`eval/evaluator.py:370-386, 425-440`（attrs → kwargs → AI 算子）
- `.agents/skills/cann-bench/bench_lab/tilelang_ascend_bench/r4_{fused_sampling,mask_sample}/{cases.yaml,cases.csv,proto.yaml,golden.py}`
- `examples/R4/bench/hap_thw_model/{r4_thw_common.py,compute_r4_fused_sampling.py,compute_r4_mask_sample.py}`（t_hw 的 BV 依赖）
- `examples/R4/DESIGN.md §4.5 L880-935, §5.1 L1033, §12 R-E L1786`（既有 UB 模型与其错误）
  关键约束（本轮据以判定）：
  1. UB 上限 **196352 B**，超限在 `tl.ascend_memory_planning=True` 下是**编译期 `LOG(FATAL)`**（已实测复现）；
  2. `padCount ≡ 0` 由 `TAIL_PAD % 32 == 0` **结构性保证**，与 BV 无关（`BV % 32 == 0` 时块内 topk 同理）；
  3. UB→UB `copy_ub_to_ub<T,T,N>` 字节宽度 `4N` 必须 %32==0（缺陷 B）；
  4. C3（`V % BV == 0 or V % BV >= K_MAX`）是 **host wrapper 硬抛**，且写进了 `proto.yaml` 的 `shape_support` 契约；
  5. `BV` 是 `case.attrs` 提供的**契约属性**，非 kernel 私有常量。

[SELF-CHECK] 本轮 Edit 数量 = **0**。`r4_fused_sampling.py` / `test_r4_fused_sampling.py` /
`cann_bench_submission/**` / `cases.yaml` / `cases.csv` / `proto.yaml` / `.orchestrator_state.json` **均未修改**。
- 唯一新增文件：`perf_tuning/baseline_round2.json`、`perf_tuning/optimization_log_round2.md`、
  `perf_tuning/perf_log.md`、`perf_tuning/round2_bv_probe/**`（数据与可复现脚本），
  以及回滚点备份 `history_version/r4_fused_sampling_impl_s3r2_attempt0_baseline.py`（内容 == 现行 ac8eba47，逐位相同）。
- 未实施 split-V、未做 K1 剪枝、未动 UB→UB 拷贝（禁令 3 ✅）。
- 未读写 `.orchestrator_state.json` / `.bak_*`（禁令 1 ✅）。
- 未使用 Python/torch 计时作为性能结论（禁令 2 ✅）——本轮**没有产出任何性能数字**，§3/§4 全是编译期静态量与生成源码解析。
- 未凭记忆猜 API（禁令 4 ✅）：`topk` / `Sort` / memory-planner 行为均读源码 + 抓生成码实证。
- 未推翻重写（禁令 5 ✅）：零代码改动。

[RESULT-#3] 优化点: BV tile size 调优 | 精度: **pass（交付版本未改动，四门禁全绿复跑，见下）** |
性能: **无候选可测 → 无性能数字** | 对比: **N/A（不存在可测量的 BV 变更）** | 采纳: **否（BLOCKED）**
| rollback_reason: **无需回滚 —— 从未修改 kernel**（sha256 before == after == `ac8eba47…`）
| consecutive_no_improvement: **+1**（本轮未产生提升；由 Orchestrator 累计判断 Stage 3 中止条件）

**四项门禁（对交付版本 = 未改动的 `ac8eba47`，本轮实际复跑，非引用 round 1）：**

| 门禁 | 命令 | 结果 | exit |
|---|---|---|---|
| 全量分层 | `python test_r4_fused_sampling.py --level all` | **74 PRECISION_PASS / 0 PRECISION_FAIL + 38 BOUNDARY_PASS / 0 BOUNDARY_WARN** | **0** |
| 稳定性 | `python test_r4_fused_sampling.py --level stability` | 12 PRECISION_PASS / 0 FAIL；4 个 tail-adversary 哨兵（TAIL=313/257/511/261）**全部 `dropped=0`**，`dropped>0` 计数 = **0** | **0** |
| UB-poison 压力 | `python test_r4_fused_sampling.py --level stress` | **8/8 poisoned configs PASS**，全部 `selfdiff=0 goldenfail=0 invariant=0`（reps=200） | **0** |
| 覆盖门禁 | `coverage_check.py test_r4_fused_sampling.py --proto proto.yaml` | **25 PASS / 0 MISS / 0 N/A** | **0** |

原始日志：`round2_bv_probe/gate_{all,stability,stress,coverage}.log`、`gate_exits.txt`。

---

## 9. 如实报告：未测项、噪声内不可判定项、验证缺口

1. **本轮没有任何 BV 候选的性能实测数据。** 原因是 §2 的落点阻断：不存在「改了 BV 且仍属合法交付」的
   可测配置。因此 §10 摘要中的 36-case geomean / worst-case delta **为空，不是 0，也不是「持平」**——
   是**未测量**。禁止把它读成「测过了没收益」。
2. **BV=4608 / 5120 的编译可行性已实测（18/18 各一次，共 36 个 K1 特化 + 2×K2m + 2×K2g）**，
   但**从未在 NPU 上运行过**（只做了 `get_kernel_source()` 编译，未 launch）。
   所以「4608/5120 能否跑通、跑多快」**未知**。
3. **BV=6144 仅实测了 K2g 的编译失败**；其 K1 / K2m 未单独实测（K2m 的 148,320 B 是模型值，
   模型在 2048/4096/4608/5120 四点均精确，但未在 6144 实测）。K2g 硬失败已足以淘汰 6144，故未继续。
4. **未测的其它 BV 值**：可行域扫描（§3.3）列出的 ≤3840 的值（含 2048/3072 等）**全部未测性能**。
   其中 2048 是夹具 case 8/13 的现役值，round 1 数据里已隐含其表现，但**没有做过 2048 vs 4096 的同 case A/B**
   （case 8/13 的 V 与其它 case 不同，无法直接对比）。
5. **UB 余量 156 B / 92 B 的鲁棒性未验证**：4608/5120 能编译成功依赖 planner 的 liveness 复用，
   而复用模式对代码形态敏感（§3.1 的 ⚠️ 行显示同一 BV 下不同 V 的复用程度不同：0/3、2/3、3/3 尾块缓冲被复用）。
   **未做**「复用模式在其它 V 上是否稳定」的穷举验证——若将来真要上移 BV，这是必须补的验证。
6. **t_hw 未按新 BV 重推导**（§6）。因为未采纳任何 BV 变更，现行 `t_hw_us` / 达成率仍然有效、未失真。
7. **per-kernel BV（落点 D）只做了成本模型层面的量化（§7），未做实测 A/B。**
   「预期 < 3%」是**分析估计**，不是实测结论。
8. **`.orchestrator_state.json` 全程未读写**（禁令 1）。其中「交付 sha256」应保持 `ac8eba47…` 不变
   （本轮无交付变更），由 Orchestrator 自行确认。

---

## 10. 本轮新增 JIT 特化（编译耗时单列，**不计入任何性能数字**）

| 项 | 数值 |
|---|---|
| 本轮新增 JIT 特化（`~/.tilelang/cache` 新增目录，实测计数） | **44** |
| 其中被**交付物**使用的新特化 | **0**（交付 kernel 未改动，运行时仍只走 BV=4096 / 2048 的既有特化） |
| 44 个的构成 | 2 个是 BV=4096/2048 的**合法**配置（此前未从本模块路径缓存过：`K1(16,152377,256,4096,False)`、`K1(4,16384,128,2048,False)`）；42 个是**探针专用**的 BV=4608/5120 特化（18+18 个 K1 + 2×K2m + 2×K2g，部分命中已有缓存） |
| 编译失败（未产生缓存条目） | 1 个：`K2g(BV=6144)`（UB 硬失败，即 §3.2 的证据） |
| 单特化编译耗时（实测） | K1 ≈ 5.8–6.2 s，K2m ≈ 4.7 s，K2g ≈ 5.1 s；缓存命中 ≈ 0.0–0.2 s |
| 探针编译总耗时（估） | ≈ 44 × 6 s ≈ **4.5 min**，全部为**一次性静态可行性探针**，与性能无关 |
| 副作用 | 44 个缓存目录中 42 个是死条目（交付物不会引用 BV≠4096/2048），仅占磁盘，可按需清理 `~/.tilelang/cache` |

> 对比 round 1：round 1 全程 cache 全 warm、**0 次新编译**；本轮 44 次新编译**全部来自静态探针**，
> 不来自任何被采纳的改动。

---

## 11. 交付物同步 / sha256

| 文件 | sha256 | 状态 |
|---|---|---|
| `examples/R4/r4_fused_sampling.py`（before） | `ac8eba47f77e11342eabb9c4e30a44531b3e655b481b0dc28fd5a66a2738fc5b` | round 2 基线 |
| `examples/R4/r4_fused_sampling.py`（after） | `ac8eba47f77e11342eabb9c4e30a44531b3e655b481b0dc28fd5a66a2738fc5b` | **未改动（逐位相同）** |
| `cann_bench_submission/cann_bench/_r4_kernels.py` | `ac8eba47f77e11342eabb9c4e30a44531b3e655b481b0dc28fd5a66a2738fc5b` | **已同步（本轮无需同步动作，本就一致）** |
| `examples/R4/test_r4_fused_sampling.py` | `02a44b79624e477c570c3c632fcdcef01dcf60041de3372a177aa947d53efb50` | **未改动**（与 round 1 一致；无任何断言被弱化） |
| `history_version/r4_fused_sampling_impl_s3r2_attempt0_baseline.py` | `ac8eba47…` | 本轮回滚点（内容 == 现行，因无改动） |

**测试文件是否需要改**：**不需要，也未改**（任务书 §9.3）。BV 未变更 ⟹ `stab_*_bv2048` 类用例语义不变。
`--level stability` 的 tail-adversary 哨兵输出确认其仍以 `BV=4096 TAIL=313/257/511/261` 运行，语义与 round 1 一致。

---

## 12. 给 Orchestrator 的建议（不属本轮范围，仅供决策）

1. **#3 应标记为「落点阻断」而非「无收益」**：二者含义不同。「无收益」意味着测过了、不划算；
   本轮是**无法在合法交付边界内构造出可测配置**。若 Orchestrator 认为 BV 应可由实现自选，
   那需要先修改 **proto.yaml / cases.yaml 的契约定义**（把 BV 从 attr 降级为「实现自选的内部 tiling 常量」，
   并重推 `t_hw`）——这是**评测集层面的决定**，超出 Stage 3 权限。
2. **这不是 `[DESIGN_ERROR]`**：现行设计正确、精度四门禁全绿、round 1 已取得 +53.8%/+42.2%。
   BV 不可上移是 **UB 物理预算 + 契约定义** 的联合结果，不是设计缺陷。
   但 §5.1/§5.2 指出的 **DESIGN §4.5 UB 模型低估 34% 且完全缺失 K2g 账目**，属**文档级缺陷**，
   建议 Orchestrator 安排修订（它会导致未来任何「开新特化」的决策基于错误预算）。
3. 若仍要追性能，**剩余杠杆不在 BV**：§7 指出 K2g 的 `2*BV` int32 过量分配是常量因子 2× 的真实浪费
   （与 BV 无关），以及 round 1 已归档的 split-V（设计层，交 Stage 1）。这两项都比 BV 更有价值。

---
---

# 第 2 项 —— [#4] 消除 K1 主块路径的 UB→UB 拷贝（`zt_ub`）

> 追加于 round 2 第 2 项。前文 §0–§12 是第 1 项（#3 BV 调优，**BLOCKED**）的记录，未改动。
> 本节所有数字均为**本轮实测**（msprof / 生成源码解析 / 逐字节比对），无引用、无估算冒充实测。

## 13. `[ORDER-CHECK]` 执行序合规

| 检查项 | 结论 |
|---|---|
| 用户批准的执行序 | `#4 → #1a → #5 → #2`（#3 已 BLOCKED 出局，5 项计划余 4 项） |
| 本轮做的项 | **仅 #4** |
| 前置依赖 | **无**。#3 的结论是「落点阻断」而非「产出被 #4 依赖的中间件」；#3 全程 0 行代码改动（`sha256 before == after == ac8eba47`），所以 #4 的起点仍是 `ac8eba47` 干净基线 |
| 是否触碰 #1a / #5 / #2 | **否**。未做 split-V、未做 K2g And halving、未做 K1 剪枝 |
| 是否触碰 BV | **否**。全部 18 case 的 BV 仍由 `case.attrs` 提供（16×4096 + 2×2048），kernel 侧 BV 默认值仍是死代码 |
| 回滚点 | `history_version/r4_fused_sampling_impl_s3r2_item4_attempt0.py`（`sha256 = ac8eba47f77e11342eabb9c4e30a44531b3e655b481b0dc28fd5a66a2738fc5b`，`cp` 建立，非手工重敲） |

## 14. `[IMPL-#4]` 已读文档、行号与关键约束

### 14.1 已读（禁止凭记忆猜 API —— AGENTS.md 核心原则 1）

| 来源 | 文件:行号 | 读到的关键事实 |
|---|---|---|
| 设计文档 | `DESIGN.md:1068-1080`（§5.3 **N2**） | N2 记载 `zt_ub` 必须是 rank-1 的来龙去脉，并**明确留下一个未验证缺口**：「原拟省去 `zt_ub`，改用 rank-1 `z_ub[BV]` + BufferRegion 归约 …… V-2 实测 BufferRegion **列切片**与 `real_shape` 走同一 narrow-reduce 路径，受 ≤256 B 约束 —— **该优化的目标场景（尾块 valid=512 的窄 region）不可用**；**满块全宽 region 是否回退 normal 路径未验证**。⟹ 实际实现放弃该优化，`zt_ub` 保留」。**本项正是关闭这个缺口** |
| 设计文档 | `DESIGN.md:1082-1090`（§5.3 **N3**） | topk 对 fp32 src **原地 -inf padding**，作用域是 UB scratch 而非 GM 视图，`buffer_size = BV` 时 padding 上界不越 buffer |
| 同文件先例 | `r4_fused_sampling.py:436`（K2g `z_ub = T.alloc_shared([BV], "float32")`，注释 "rank-1 [BV] topk source (N2)"）<br>`:468`（`T.copy(logits[row, b*BV], z_ub, pad_value=…)` GM→UB **直接写 rank-1**）<br>`:522`（`T.reduce_max(z_ub, blk_max, dim=-1)`，rank-1 源，`blk_max` 为 rank-1 `[1]`）<br>`:525`（`T.reduce_min(g_ub, blk_argmax, dim=-1)`，同上）<br>`:534`（`T.tile.compare(mask, z_ub, t_v[0, 0], "GE")`，**scalar/BufferLoad src1** 先例） | **rank-1 作 topk 源、GM→UB 直接落 rank-1、rank-1 全宽 reduce 用 `dim=-1`、scalar src1 —— 四者在本文件 K2g 里都已 NPU 验证通过**。⟹ 本项不是探索新路径，是把 K2g 的既有形态搬到 K1（AGENTS.md 核心原则 2：从示例入手） |
| reduce 前端 | `tilelang/language/reduce_ascend.py:228-237` | `_legalize_reduce_dim`：rank-1 buffer 时 `normalized_dim = 1 + (-1) = 0` → **返回 -1**。⟹ **`dim=-1` 是 rank-1 的正确形态**（不是 `dim=0`；`dim=0` 也会被 normalize 到同一条路径，但 K2g 用的是 `dim=-1`，保持一致） |
| reduce 前端 | `tilelang/language/reduce_ascend.py:318-320` | `if len(buffer_extent) == 1: M = 1; N = buffer_extent[0]` ⟹ rank-1 `[BV]` 与 rank-2 `[1,BV]` 产出**同一组模板参数** `M=1, N=BV` |
| reduce 前端 | `tilelang/language/reduce_ascend.py:130-172`（`_validate_reduce_out_shape` / `_shape_can_embed_expected_shape`） | rank-1 源 + `dim=-1` → `normalized_dim=0` → `reduced_shape=[]`、`keepdim_shape=[1]` ⟹ `expected_shapes = [[], [1]]`。**rank-2 `[1,1]` 的 `out` 以 Buffer 形式传入会 `raise ValueError`**；以 **BufferRegion** 形式（`out_is_region=True`）传入则 `_shape_can_embed_expected_shape([1,1], [])` 逐元素全为 1 → **返回 True，校验通过** |
| reduce 前端 | `tilelang/language/reduce_ascend.py:337-353` | `physical_row = src_buffer.shape[-1]`，且**仅当 `row_const != n_const` 时才 append 到 args** |
| reduce codegen | `src/target/codegen_ascend.cc:2070-2078, 2096-2098` | narrow 分支的**唯一入口是 `if (physical_row > 0)`**；`:2141` `ICHECK(ndim == -1)`、`:2145` `ICHECK(nn * elem_bytes <= 256)` 都在该分支内部 |
| **narrow-reduce 结论** | 由上三条推得，并已用生成源码实证（§16.2） | rank-1 `[BV]`：`physical_row = BV = 4096`，`n_const = BV = 4096` → **相等 → 不 append → codegen `physical_row == 0` → 不进 narrow 分支 → 走 normal 路径**。**改前 rank-2 `[1,BV]` 的推导完全相同**（`shape[-1]` 同为 BV）⟹ **rank 变更不改变路径选择**。DESIGN N2 放弃本优化的理由是「**列切片** region（`z_ub[0:valid]`，valid < BV）→ `row_const=BV != n_const=valid` → 进 narrow → 撞 256 B」；本项用的是**全宽 plain Buffer**，两者是不同分支 |
| tile 二元/一元 op | `tilelang/language/ascend_tile.py:1030-1086`（`binary_op`）、`:1229-1252`（`unary_op`） | 只做 `size_0 = math.prod(dst_extent)` / `size_1 = math.prod(src0_extent)` + `assert size_0 == size_1` ⟹ **完全 rank-agnostic**。所以 `z_ub` 变 rank-1 后，`e_ub`/`m2d` **无需**跟着改 rank（`prod([BV]) == prod([1,BV]) == BV`） |
| tile broadcast | `tilelang/language/ascend_tile.py:2488-2506` | `dst_extent/src_extent` 取 `[-2:]`，随后 `if src_dim != dst_dim: raise ValueError`。**若把 `m2d` 改成 rank-1，则 `broadcast(m2d[BV], m_run[1,1])` 会 dst_dim=1 / src_dim=2 → 直接抛错**；rank-1 broadcast 要求 src 也是 rank-1 `[1]`，而那需要一次 1 元素 UB→UB 拷贝（4 B < 32 B，`r4_fused_sampling.py:16-17` 明令禁止）⟹ **`m2d` 保持 rank-2 是唯一可行选择**，本轮不动它 |
| topk | `tilelang/language/ascend_tile.py:599-668`，尤其 `:635-645` | `max_actual_num = Σ src.shape`（**求和不是求积**）+ `repeatTimes = (max_actual_num+31)//32`。rank-2 `[1,BV]` → `1+4096 = 4097` → `repeatTimes = 129` → 排序覆盖 `129×32 = 4128 > 4096` 槽位（**读越界 32 元素**）；rank-1 `[BV]` → `4096` → `repeatTimes = 128` **精确**。⟹ **topk 源必须 rank-1，这条约束成立且被本项保留**（只是把「专用 rank-1 缓冲」换成「z_ub 本身就是 rank-1」） |
| GM→UB codegen | `src/op/ascend.cc:251-278` | 第三个模板参数**仅在 `dst->shape.size() > 1` 时才发射**，值为 `compute_blocklen(dst, dst_extents)` |
| GM→UB helper | `~/.local/lib/python3.12/site-packages/tilelang/src/tl_templates/ascend/common.h:215-219` | `template <typename T, uint32_t dstN, uint32_t dstM = 1>` ⟹ **`dstM` 默认值就是 1**。所以 `copy_gm_to_ub<float,4096,1>`（rank-2）与 `copy_gm_to_ub<float,4096>`（rank-1）**实例化的是同一个模板特化**，运行期实参也完全相同 → 生成机器码逐位相同 |
| region-out reduce 先例 | `examples/fused_sigmoid_gating_delta_rule/fused_sigmoid_gating_delta_rule_varlen.py:238, :256` | `T.reduce_sum(compute_buffer, pred_1d[:, 0], dim=-1)` —— 仓库内已有 **BufferRegion 作 reduce out** 的用法 |

### 14.2 实际改动（5 处语义变更，全部落在 `if n_blk_full > 0:` 块内）

| # | 位置（改前 → 改后行号） | 改前 | 改后 | rank 变化 |
|---|---|---|---|---|
| 1 | L192 → L204 | `z_ub = T.alloc_shared([1, BV], "float32")` | `z_ub = T.alloc_shared([BV], "float32")` | **rank-2 → rank-1** |
| 2 | L195 → — | `zt_ub = T.alloc_shared([BV], "float32")` | **删除** | 消除 16384 B（BV=4096）/ 8192 B（BV=2048）分配 |
| 3 | L199 → L215 | `T.reduce_max(z_ub, m_blk, dim=-1)` | `T.reduce_max(z_ub, m_blk[0, 0:1], dim=-1)` | out 由 Buffer 改 **BufferRegion**（rank-1 源的 `expected_shapes` 是 `[[], [1]]`，`[1,1]` Buffer 会被前端拒绝） |
| 4 | L211 → — | `T.copy(z_ub[0, :BV], zt_ub)` | **删除** | 消除每满块一次的 UB→UB 拷贝（**本项主目标**） |
| 5 | L212 → L232 | `T.tile.topk(l_dst, zt_ub, K_MAX, BV)` | `T.tile.topk(l_dst, z_ub, K_MAX, BV)` | 源换成 z_ub，`K/actual_num` 参数不变 |

**未改**：`e_ub`/`m2d` 仍为 rank-2 `[1,BV]`（§14.1 已证 `binary_op`/`unary_op` rank-agnostic，且 `m2d` 改 rank-1 不可行）；
`T.reduce_sum(e_ub, s_blk, dim=-1)`（L208 → L224）**逐字未改**（e_ub 仍 rank-2，`s_blk` 仍是 `[1,1]` Buffer，校验走 `expected_shapes=[[1],[1,1]]` 命中 `[1,1]`）；
`T.tile.broadcast(m2d, m_run)`、`T.tile.sub(e_ub, z_ub, m2d)`、`T.tile.exp(e_ub, e_ub)` **逐字未改**。

**未引入任何新的 UB→UB 拷贝**（净减 1 次）；**未引入任何新的 topk 调用**（净 0，参数不变，`BV % 32 == 0` 保持）。
⟹ 红线 §6.2（新 UB→UB 拷贝字节数须 %32==0）与 §6.3（不得引入 `count % 32 != 0` 的 topk）**均满足（vacuously：无新增）**。

### 14.3 反模式对照（已逐节通读 `tilelang-perf-optimization/references/performance-antipatterns.md`，498 行 / 11 节）

按该文件「使用方式」的要求（*性能优化前用本清单做第一轮静态检查，再结合 msprof 数据定位瓶颈*），
对改后的 K1 逐节对照。**本项本身消除的就是一条反模式实例**：
「仅为满足下游 API 形状要求而做的冗余 UB→UB 数据搬运」——改后主块循环体内**不再有任何纯搬运指令**。

| 反模式章节 | 对 K1 改后实现的判定 | 依据 / 保留原因 |
|---|---|---|
| **冗余全局同步**（`:154-196`） | **本项直接改善** | 该节替代方向第 1 条「删除没有生产者/消费者依赖的同步」。删掉 `T.copy(z_ub→zt_ub)` 后，`copy`→`topk` 这条依赖消失，codegen 随之**删除了每满块一次的 `AscendC::PipeBarrier<PIPE_V>()`**（§18.4 生成码 diff 实证）。该节检查点要求「必须通过生成的 Ascend C 代码和精度测试确认同步语义没有被破坏」——已做：生成码逐行 diff（§19.1）+ 四门禁 + **UB-poison stress 8/8**（§20.1）。注意 K1 未手写任何 `barrier_all`/`sync_all`，同步全由 `tl.ascend_auto_sync` 生成 |
| **纯 AIV memory bound 算子未做流水/双 buffer**（`:340-381`） | **命中识别特征，但有意保留 —— 且本项把它从"不可能"变成"静态可行"** | K1 是纯 AIV（round 1 §3 已用 `get_kernel_source()` 实证 `gemm/mma/cube 使用数=0`），主循环 `T.serial(n_blk_full)` 内确实是 `GM→UB→Vector` 串行，符合该节识别特征。**保留原因是 UB 预算**：双 buffer 会让 UB 占用乘以 2（该节检查点第 1 条自己就点了这一点）。量化：改前最差 case 6 = 188004 B，余量 8348 B，给 `z_ub` 加双 buffer 需 +16384 B → **204388 > 196352，溢出 8036 B，物理不可行**；**改后 171620 B，余量 24732 B → +16384 B = 188004 ≤ 196352，恰好可行**（`z_ub+e_ub+m2d` 三者全双 buffer 仍需 +49152 B → 220772，溢出 24420 B，仍不可行）。⟹ **本项是这条反模式的前置使能**，但**实施双 buffer 不在本轮范围**（是独立一项，且须遵守该节检查点「理论估算只作参考，最终以内存规划结果和实际运行是否报内存不足为准」→ 必须实测，不能按上表的加性估算下结论） |
| **多行 tile 循环内逐块归约**（`:384-418`） | **不适用 / 已合规** | 该节识别特征限定「多行 tile（`[rows, block_S]`，**rows > 1**）」。K1 主块的 `z_ub`/`e_ub` 是 `[1,BV]`（rows=1，改后 `z_ub` 更是 rank-1），不匹配。且该节两条检查点均已满足：① 循环内确有 `reduce_max`/`reduce_sum`，但这是 **online softmax 的算法本质**（running max/sum 必须逐块更新；要「循环内只累积、循环外一次归约」就得把整个 V 留在 UB，V=152064 时 = 608 KB ≫ 196352 B，物理不可能）；② **K1 全程只用 `dim=-1`，从未用 `dim=0`**，规避了该节指出的「`reduce_sum(dim=0)` 在 `[rows,1]` 上行为不确定、可能产生 NaN」的坑 |
| **tile size 过小导致片上内存浪费**（`:267-301`） | **属 #3 范围，本轮不做** | 即 BV 调优 —— #3 已判 **BLOCKED**（BV 是 `case.attrs` 契约属性）。本项走的是**反方向且同样有效**的路子：不动 tile size，而是**删掉一个 BV 大小的冗余缓冲**来腾出余量。该节容量表列 UB = 196352 B，与本轮实测口径一致 |
| **基础指令拼接未融合**（`:198-265`，`mul+add` / `max(x,0)` / `sqrt+div`） | **不命中** | K1 主块无这三类 pattern；online softmax 的 `sub→exp→reduce_sum` 不是可融合的复合指令形态 |
| **Vector Core 内逐元素/逐行 for loop**（`:124-152`） | **不命中（round 1 [#1] 已消除）** | round 1 已把 epilogue 的 `T.Parallel(K_MAX)` 标量循环向量化（`tie_m` → `broadcast+compare+select+reduce_sum`）。现存的两处 `T.serial` 循环（`for l in T.serial(K_MAX)` 的 `tau_p` 前缀和、`for _ in T.serial(1)` 的 guarded 动态索引）中，前者 round 1 [#4] 已探测过 `T.cumsum` 在纯 Vector + `R4_PASS_CONFIGS` 下 `InternalError`、无可用向量 scan 原语；后者是 1 次迭代的守卫壳，不是逐元素计算 |
| **正交轴串行化**（`:420-459`） | **不命中** | K1 的并行轴是 row（`b_num × VEC_NUM` 已铺满），串行轴是 V 方向的块扫描 —— 这是算法本质（online softmax 单趟扫描），不是「可并行轴被串行化」。真正的 V 轴并行化就是 split-V（#1a，设计层） |
| **launch core 数**（关注项 A/B，`:60-122`） | **不在本项范围，如实记录** | K1 的 `b_num = ceildiv(B, VEC_NUM)`，`VEC_NUM = 2`。case 2（B=1）→ `b_num=1` → 只用 1 个 AIC + 2 个 AIV，而 910b2 有 20 AIC / 40 AIV ⟹ **命中「关注项 B：任务数低于物理 AI Core 数」，小 B 场景核心利用率极低**。这是 split-V（#1a）要解决的问题，**本项未触碰**，仅记录 |
| **AIC/AIV 混合未开 CV overlap**（`:303-338`） | **不适用** | K1 无 Cube 计算（`gemm/mma/cube 使用数=0`），不存在 AIC/AIV 流水可重叠 |

**另有一处「性能关注项」有意保留（本可再省，但超出本轮判据）**：`m2d` 的 `T.tile.broadcast`
（把标量 `m_run` 铺成 `[1,BV]` 再做 tensor-tensor `Sub`）占 **16384 B UB + 每满块 1 条向量指令**。
该文件 `:41` 明确把 `T.tile.broadcast` 列为「会增加片上内存占用」的一项。技术上可用
`T.tile.sub(e_ub, z_ub, m_run[0, 0])` 的 **scalar-src1** 形态消除它（`ascend_tile.py:1106-1127`，走 `AscendC::Adds(dst, src0, -src1)`；
同文件 K2g `:534` 已有 scalar src1 的 NPU 验证先例）。
**保留原因**：① 它会**改变实际算术指令**（`Sub` → `Adds`），与本轮「纯数据搬运重构 ⟹ 期望 bitwise 一致」的判据冲突
——虽然 IEEE-754 下 `a-b` 与 `a+(-b)` 逐位相同，但那是**推理，不是本硬件实测**；
② `m2d` 改 rank-1 这条路是**死的**（§14.1 broadcast 行：rank-1 broadcast 要求 rank-1 `[1]` 源，
而得到它必须做一次 1 元素 UB→UB 拷贝，4 B < 32 B，`r4_fused_sampling.py:16-17` 硬禁）。
⟹ 按该文件 `:3` 的要求「如果必须临时保留，需要记录 shape、dtype、原因和后续优化计划」：
**shape `[1, BV]` = `[1,4096]`（case 8/13 为 `[1,2048]`）、dtype `float32`、原因如上、后续计划见 §22.3**
（可把 K1 最差余量从 24732 B 再提到约 **41116 B**，需独立一轮验证 bitwise）。

## 15. `[SELF-CHECK]` 本次 Edit 只涉及 #4

| 检查项 | 结论 |
|---|---|
| Edit 次数 | **2**（alloc 块 1 次、topk 块 1 次），均落在 `r4_fused_sampling.py` 的 `if n_blk_full > 0:` 主块路径内 |
| 语义变更点数 | **5**（§14.2 表），其中 2 个是删除 |
| 是否涉及 #1a（split-V） | **否**，kernel 数量、GM workspace、分派逻辑均未动 |
| 是否涉及 #5（K2g And halving） | **否**，`r4_gumbel_sample` 全函数逐字未改（`state_ub/h_ub/tmp_ub` 仍是 `[BV2]`） |
| 是否涉及 #2（K1 剪枝） | **否**，`n_blk_full` 循环仍是无条件全扫，无块级 early-skip |
| 是否涉及 BV | **否** |
| 尾块路径 L219-249 | **逐字节未改**（sha256 证明见 §16.1） |
| `test_r4_fused_sampling.py` | **未改**，sha256 仍为 `02a44b79624e477c570c3c632fcdcef01dcf60041de3372a177aa947d53efb50`；**未弱化任何断言**（无需改测试，理由见 §19） |
| `proto.yaml` / `cases.yaml` / `cases.csv` | **未改** |
| `.orchestrator_state.json` / `.bak_*` | **全程未读写**（禁令 1） |
| 性能结论来源 | **仅 msprof**（cann-bench `kernel_details.csv`）。**未使用任何 Python/torch 计时**（禁令 2） |
| 是否推翻重写 | **否**，5 处最小改动（禁令 5；AGENTS.md 核心原则 4） |

## 16. 尾块红线证明 + narrow-reduce 核实（强制交付物）

### 16.1 尾块路径 L219-249 逐字节未变

抽取方式：**按内容标记定位，不按固定行号**（改动会在上方增删行，行号必然漂移）。
起点 = `# ======== tail block:` 那一行；终点 = `# ======== epilogue:` 的**前一行**。
该 marker 区间是 L219-249 的**超集**（多含 1 行空行分隔符），所以证明它逐字节一致即证明 L219-249 逐字节一致。
脚本：`perf_tuning/round2_item4/tail_region_hash.py`。

| 区段 | 改前 | 改后 | 结论 |
|---|---|---|---|
| **marker 区间**（改前 L219..L250，32 行 → 改后 L239..L270） | `37f19e3eac4548103ae5d2232075bd057c6ebeb013eb042cc8f95d65ff1618c8` | `37f19e3eac4548103ae5d2232075bd057c6ebeb013eb042cc8f95d65ff1618c8` | **逐字节一致** |
| **严格 31 行区间**（任务书指定的 L219-249 → 改后 L239-269） | `098e4bc131cd81eb183ffbdb45ca8ebee437089f99c22d7ecaebd2d941deb608` | `098e4bc131cd81eb183ffbdb45ca8ebee437089f99c22d7ecaebd2d941deb608` | **逐字节一致** |
| 区间末行（自检锚点） | `T.tile.gather_mask(run_v, m_dst, "P0101")` | 同 | 一致 |
| `diff` 抽取文本 | — | — | **两个区段均无 diff 输出** |

TAIL_PAD 红线的 4 个关键行（改前 L228/L229/L242/L243/L244）**全部在上述 hash 覆盖范围内**，故
`ztp[1,TAIL_PAD]` / `ztt[TAIL_PAD]` / `pad_value=-T.infinity` / `T.copy(ztp[0,:TAIL_PAD], ztt)` / `topk(..., TAIL_PAD)`
**构造上未被触碰**。

**追加的生成码级证明**（比源码 hash 更强）：对全 18 case 抽取尾块相关行
（`zt`/`et`/`mt`/`ztp`/`ztt` 参与的指令），排除 `GetWithOffset` 布局行后逐 case `diff`：

```
case01/05/06/07/13/14/15/17: tail CODE IDENTICAL (9 lines)
case02/03/04/08/09/10/11/12/16/18: tail CODE IDENTICAL (0 lines)   # TAIL==0，尾块被编译器消除
```

且尾块缓冲**尺寸**逐 case 未变（只有地址偏移因 `zt_ub` 腾出而下移）：

| case | TAIL | zt/et/mt | ztp/ztt |
|---|---|---|---|
| 5 | 313 | 313 | **320**（=TAIL_PAD，32 对齐） |
| 6 | 2048 | 2048 | 2048 |
| 7 / 15 | 512 | 512 | 512 |
| 13 | 256 | 256 | 256 |
| 17 | 825 | 825 | **832**（=TAIL_PAD） |

⟹ **尾块在源码级和生成码级都逐字节未变**，round 1 修复的两个 wheel 缺陷
（`Sort()` masked `Duplicate` 失效 / `copy_ub_to_ub` 非 32 B 静默截断）的规避实现完好无损。

### 16.2 narrow-reduce 约束核实（任务书 §4 关键点 3）

**结论：rank-1 `[BV]` 全宽 reduce 仍走 normal 路径，且发射的 C++ 与改前逐字节相同。**

三重证据：

1. **源码推导**（§14.1）：narrow 分支唯一入口 `if (physical_row > 0)`（`codegen_ascend.cc:2098`），
   而 `physical_row` 仅在 `row_const != n_const` 时才被前端 append（`reduce_ascend.py:352`）。
   rank-1 `[BV]`：`row_const = shape[-1] = BV`，`n_const = N = BV` → 相等 → 不 append → `physical_row == 0` → normal。
2. **生成码实证**（全 18 case）：`grep -c narrow` 在 before/after 的全部 36 个 `.cc` 中**均为 0**
   —— 没有任何 `reduce_max_narrow` / `reduce_sum_narrow` / `reduce_min_narrow` 被发射。
3. **改前后逐行比对**（case15，`perf_tuning/round2_item4/emitted_cc_case15/`）：

```
BEFORE L107:  tl::ascend::reduce_max<float, 1, 4096, -1>(m_blk[0], z_ub[0], tmp_ub[0], true);
AFTER  L106:  tl::ascend::reduce_max<float, 1, 4096, -1>(m_blk[0], z_ub[0], tmp_ub[0], true);
```

**逐字节相同**（仅行号因上方删了 2 行代码而上移 1 行）。模板参数 `<float, 1, 4096, -1>` 未变
⟹ 印证 `reduce_ascend.py:318-320` 的 `M=1, N=BV` 推导；`m_blk[0, 0:1]` 这个 BufferRegion
发射出的仍是 `m_blk[0]`（offset 0）⟹ 与改前的 plain Buffer 形式**生成同一行代码**。
`reduce_sum<float, 1, 4096, -1>(s_blk[0], e_ub[0], …)` 同样逐字节未变（e_ub 未改 rank）。

**这一步同时关闭了 DESIGN §5.3 N2 留下的缺口**：N2 说「满块全宽 region 是否回退 normal 路径**未验证**」，
现已验证 —— 全宽 plain Buffer（乃至等价的 region-out）走 normal，只有**列切片窄 region** 才落 narrow。
N2 当年放弃本优化的判断在其目标场景（尾块窄 region）下是**正确的**，但把结论外推到满块全宽是**过度保守**。

## 17. UB 实测对比（18 case × K1，非模型）

测量口径：解析生成源里的 `ascend_ub.GetWithOffset<T>(n, off)`，取 `max(off + n*sizeof(T))`
—— 即 `src/transform/ascend_memory_planning.cc:582-585` 打印 "Scope shared.ub memory usage: X/196352 bytes" 的那个量。
脚本：`perf_tuning/round2_item4/dump_k1.py`（复用 round 2 的 `ub_probe.py` + `bv_static_analysis.CASES`）。
上限 **196352 B**。**不使用 DESIGN §4.5 的模型**（round 2 §5.1 已证其低估 34%）。

完整表格见 `perf_tuning/round2_item4/ub_ab_table.md`，摘要：

| case | B | V | BV | n_blk_full | UB before | UB after | Δ | 余量 before | 余量 after |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 256 | 4096 | **0** | 29284 | 29284 | **+0** | 167068 | 167068 |
| 2 | 1 | 4096 | 4096 | 1 | 147044 | 130660 | −16384 | 49308 | 65692 |
| 3 | 8 | 16384 | 4096 | 4 | 147044 | 130660 | −16384 | 49308 | 65692 |
| 4 | 4 | 16384 | 4096 | 4 | 147044 | 130660 | −16384 | 49308 | 65692 |
| 5 | 7 | 4409 | 4096 | 1 | 153444 | 137060 | −16384 | 42908 | 59292 |
| **6** | 8 | 10240 | 4096 | 2 | **188004** | **171620** | −16384 | **8348** | **24732** |
| 7 | 4 | 152064 | 4096 | 37 | 157284 | 140900 | −16384 | 39068 | 55452 |
| 8 | 4 | 16384 | **2048** | 8 | 73828 | 65636 | **−8192** | 122524 | 130716 |
| 9 | 64 | 4096 | 4096 | 1 | 147044 | 130660 | −16384 | 49308 | 65692 |
| 10 | 4 | 4096 | 4096 | 1 | 147108 | 130724 | −16384 | 49244 | 65628 |
| 11 | 8 | 8192 | 4096 | 2 | 147108 | 130724 | −16384 | 49244 | 65628 |
| 12 | 16 | 32768 | 4096 | 8 | 147044 | 130660 | −16384 | 49308 | 65692 |
| 13 | 6 | 2304 | **2048** | 1 | 78948 | 70756 | **−8192** | 117404 | 125596 |
| 14 | 32 | 1024 | 4096 | **0** | 52836 | 52836 | **+0** | 143516 | 143516 |
| 15 | 64 | 152064 | 4096 | 37 | 157284 | 140900 | −16384 | 39068 | 55452 |
| 16 | 128 | 32768 | 4096 | 8 | 147044 | 130660 | −16384 | 49308 | 65692 |
| 17 | 16 | 152377 | 4096 | 37 | 163684 | 147300 | −16384 | 32668 | 49052 |
| 18 | 64 | 65536 | 4096 | 16 | 147108 | 130724 | −16384 | 49244 | 65628 |

**关键读数：**

- **最差 case（case 6）：188004 B（95.75%）→ 171620 B（87.40%）；余量 8348 B → 24732 B，+16384 B，×2.96。**
  与任务书 §5 的预测（"应释放 16384 B，把 K1 余量从 8348 B 提到约 24732 B"）**精确吻合**。
- **Δ 是单一常量**：BV=4096 且 `n_blk_full>0` 的 14 个 case 全部 `−16384`（= `BV*4`，即 `zt_ub` 一个缓冲）；
  BV=2048 的 2 个 case 全部 `−8192`（= `2048*4`）。**无任何 case 出现意外增减** ⟹ planner 没有
  因形态变化而产生额外的 liveness 复用差异，减少量恰好等于被删缓冲的大小。
- **`n_blk_full == 0` 的 case 1 / 14：UB 完全未变（29284 / 52836）**。这两个 case 走 `if n_blk_full > 0:` 的假分支，
  `z_ub/e_ub/m2d/zt_ub` 一个都不分配 ⟹ **本项对它们是构造性 no-op**（三重印证：源码分支、生成码 body diff = 0 行、UB Δ = 0）。
- **K2m / K2g 完全未变**：`k2m_4096 98912→98912`、`k2g_4096 148452→148452`、
  `k2m_2048 49504→49504`、`k2g_2048 74468→74468` ⟹ 无跨 kernel 耦合。
- 18/18 case 编译成功，无 UB 溢出，无一例接近上限（最差 87.40%）。

**对 #2（K1 块级剪枝）的使能价值**：K1 可用余量从 8348 B 提升到 24732 B（+16384 B）。
以 BV=4096 计，16384 B 相当于 1 个额外的 `[BV]` fp32 缓冲，或 4 个 `[K_MAX=256]` fp32 缓冲
（剪枝若要存块级 max 摘要，量级正在这里）。⟹ **#4 排在 #2 之前的意图已兑现**。

## 18. bitwise 证据（任务书 §6.4 要求 bitwise，容差型 pass 不算）

### 18.1 口径

自建捕获脚本 `perf_tuning/round2_item4/bitwise_capture.py`：对 **22 个 shape × 2 条派发路径**
（`FUSE_GUMBEL=False` → `w_out`；`FUSE_GUMBEL=True` → `tok_out`）**× 5 个输出**
（`m_out` / `s_out` / `tau_out` / `valid` / `w_out|tok_out`）= **220 个张量**，
用与 `test_r4_fused_sampling.py:_make_inputs` **完全相同的确定性配方**
（`torch.manual_seed(0)` + `randn(B,V)*LOGIT_SCALE(8.0)`）生成输入，改前改后各跑一次，
把每个输出 `.cpu().contiguous()` 的**原始字节**落盘并取 sha256。
比对脚本 `cmp_bitwise.py` 同时校验 **sha256 相等 + 原始字节 `==` 相等 + shape/dtype 相等**，任一不符则 exit 1。

22 个 shape 覆盖：18 个 bench case 的全部互异 K1 特化签名 `(V, K_MAX, BV, NEED_MIN_P)`、
`n_blk_full == 0`（2 个）、`TAIL == 0`（多个）、`TAIL ∈ {256, 313, 512, 825, 2048}`、
BV=2048（2 个）、`need_min_p=True`（3 个）、`--level stability` 的 4 个 tail-adversary 哨兵
（TAIL=313/257/511/261）、以及 `k=1` / `k=K_MAX` 两个边界。

### 18.2 结果

```
==== BITWISE SUMMARY ====
tensors compared : 220   identical: 220   differing: 0
VERDICT: ALL OUTPUTS BITWISE IDENTICAL (sha256 + raw-byte compare + shape/dtype)
```

- **220/220 张量逐位相同**（`cmp_bitwise.py` exit 0）。含 `w_out` 这种大张量
  （case15 单个 9,732,096 B）与 `tok_out`（采样结果，任何 topk 污染都会立刻翻位）。
- 两份 `manifest.json`（各 45006 B）**文件级逐字节相同** —— 即 220 个 sha256 一次性全部对上。
- **不是容差判定**：`ratio` / `max_abs` 一个都没用；判据是 `bytes == bytes`。

### 18.3 交叉印证：cann-bench harness 自己的精度判定

36/36 case（18×2 任务）在 baseline 与 candidate 两次 eval 中，
每个输出（`m_out`/`s_out`/`tau_out`/`valid`/`w_out`|`tok_out`）的
`accuracy.output_results[*]` 全部为 `mismatch_count = 0`、`max_abs_diff = 0`、`max_diff = 0.0`、`passed = true`。
即 harness 侧也是**零失配**（它对 golden 的判定阈值远松于 bitwise，而实测误差为 0）。

### 18.4 为什么 bitwise 一致是可以先验证明的（不只是实测巧合）

改前后主块循环体的**发射代码差异只有 3 处**（全 16 个 `n_blk_full>0` case 完全一致，见 §19.1）：

| 差异 | 语义影响 |
|---|---|
| `copy_gm_to_ub<float,N,1>` → `copy_gm_to_ub<float,N>` | **同一模板特化**：`common.h:215` 的 `dstM` 默认值即 1；运行期实参 `(z_ub[0], logits[…], stride, 1, N, 0.0)` 逐字未变。`maskShapeN == dstN` ⟹ `isPad=false`、不触发 `Duplicate`。⟹ **同一份机器码** |
| `copy_ub_to_ub<float,float,N>(zt_ub[0], z_ub[0], 1, N, N, 1, N, N)` + `PipeBarrier<PIPE_V>()` → **删除** | 被删的是一次 `N` 元素（`N*4` B，N=4096→16384 B = 512×32 B；N=2048→8192 B = 256×32 B）**全宽、32 B 对齐**的纯搬运。它不改变任何数值，只把 `z_ub` 的比特复制到 `zt_ub`。删掉它 ⟹ 下游直接读 `z_ub`，**读到的比特完全相同** |
| `TopK<float>(l_dst[0], zt_ub[0], tmp, K, repeatTimes, N)` → `TopK<float>(l_dst[0], z_ub[0], tmp, K, repeatTimes, N)` | **K / repeatTimes / actual_num 三个参数一字未变**（case15：`256, 128, 4096`；case8：`128, 64, 2048`），只换了源指针，而源内容按上一条逐位相同 ⟹ 同一运算、同一输入、同一输出 |

再加上 §16.2 已证的 `reduce_max` / `reduce_sum` 发射行**逐字节相同**，
⟹ **主块路径上每一个运算的输入比特、运算本身、输出比特都与改前一致，顺序也未变**。
`z_ub` 的生命周期检查：写入 = 循环首行 GM→UB；读取 = `reduce_max`(L215) 与 `T.tile.sub`(L222)；
此后仅被 `topk` 原地触碰，而**下一次迭代首行的 GM→UB 会整块覆盖它** ⟹ topk 的原地 -inf padding（N3）
不会污染任何后续读取（改前 `zt_ub` 同理，也是每次迭代被整块覆盖）。
**这是 bitwise 一致的构造性证明，§18.2 的 220/220 是它的实证确认。**

### 18.5 反证控制：`n_blk_full == 0` 的 case 1 / 14

这两个 case 的生成码 **body diff = 0 行**（§19.1），UB **Δ = 0**（§17），
且 JIT 缓存**命中旧条目**（改后重编译耗时 0.1 s，其余 15 个 case 均为 5.7–6.1 s）
⟹ 它们的 TIR/IR 与改前**完全相同**，连缓存键都没变。这是「本项对 `n_blk_full==0` 是彻底 no-op」的最强证明。

## 19. 36-case 性能对比（msprof，cann-bench 口径）

### 19.0 测量口径与同会话基线

- 工具：`.agents/skills/cann-bench/scripts/run_evaluation.sh`，`--bench-name cann --task-dir bench_lab/tilelang_ascend_bench
  --source-dir examples/R4/cann_bench_submission --device-id 0 --processes-per-card 1`；
  msprof `kernel_details.csv`，warmup=3 repeat=5，取每 step 求和后的 median。
- 环境配方与 round 1 一致（`R4_HTML_SHIM=1` + `/tmp/opencode/r4_eval/shim` 的 sitecustomize、
  `LD_LIBRARY_PATH=/usr/local/sqlite3/lib:…`、`ACL_OP_INIT_MODE=1`、`TILELANG_CACHE_DIR=/tmp/opencode/r4_tl_cache`）。
  脚本存档：`perf_tuning/round2_item4/run_eval.sh`。
- **基线不是引用 round 1/round 2 的旧报告，而是本轮在同一会话、同一环境、同一缓存目录下对 `ac8eba47` 重新实测的**，
  以消除环境漂移。四次 eval 全部 `EXIT_CODE=0`：

| 角色 | 任务 | report | 18 case | kernel sha256 |
|---|---|---|---|---|
| baseline | R4FusedSampling | `bench/reports/cann_performance_eval_20261010_170920.json` | 18/18 pass, 0 fail | `ac8eba47…` |
| baseline | R4MaskSample | `bench/reports/cann_performance_eval_20261010_171114.json` | 18/18 pass, 0 fail | `ac8eba47…` |
| candidate | R4FusedSampling | `bench/reports/cann_performance_eval_20261010_172650.json` | 18/18 pass, 0 fail | `19e51e8d…` |
| candidate | R4MaskSample | `bench/reports/cann_performance_eval_20261010_172844.json` | 18/18 pass, 0 fail | `19e51e8d…` |

- **漂移标定**（旧 round-2 报告 `012731`/`012928` vs 本轮同 kernel 新基线 `170920`/`171114`）：
  fused geomean **+0.136%**、mask geomean **+0.378%**、单 case 最大 **+1.71%**、score 差 −0.017% / −0.051%。
  ⟹ **环境稳定，且单 case 噪声可达 ~1.7%**。这条标定是下文判读的依据。

### 19.1 生成码 body diff（全 18 case，剥离 `GetWithOffset` 布局行后）

```
case01: body_diff_lines=0        # n_blk_full==0 -> 完全 no-op
case14: body_diff_lines=0        # n_blk_full==0 -> 完全 no-op
case02..13,15..18: body_diff_lines=6   # 全部是同一组 3 处变化（见 §18.4 表）
```

16 个 `n_blk_full>0` 的 case 的 diff **完全同构**，且**不含任何 reduce / 尾块 / epilogue 行**。

### 19.2 端到端 36-case 结果

完整逐 case 表：`perf_tuning/round2_item4/perf_cmp_36case.txt`（JSON：`…json`）。

| 任务 | score base → cand | geomean(cand/base elapsed) | 等效提速 | 最差单 case 回退 | 最好单 case | 回退>3% 的 case 数 | acc fail | mismatch 合计 |
|---|---|---|---|---|---|---|---|---|
| `r4_fused_sampling` (K1+K2g) | 59.199724 → 59.225076 (+0.043%) | 0.996494 | **+0.352%** | **+0.285%** (case 2) | −0.942% (case 10) | **0** | **0** | **0** |
| `r4_mask_sample` (K1+K2m) | 58.416610 → 58.438490 (+0.037%) | 0.996409 | **+0.360%** | **+0.548%** (case 14) | −1.584% (case 5) | **0** | **0** | **0** |

### 19.3 K1 隔离拆分（本轮新增，比端到端更灵敏）

两个 TileLang kernel 在 msprof 里都叫 `main_kernel`，故按 **step 内发射顺序**拆分
（K1 `r4_topk_stats` 先，K2 后）—— 与 round 1 的口径一致。脚本：`perf_tuning/round2_item4/k1_split.py`。
**方法学校验**：每个 step 恰好 3 行（1× `CannBenchCacheClean` + 2× `main_kernel`），144 step 全部如此，
`anomaly_rows = 0`；且每 case 的 `Block Num` 都等于 `ceil(B/2)`（`blk_ok = True` × 18/18），
拆分出的 `K1_us + K2_us` 之和**精确复现**报告里的 `elapsed_us`（case1: 24.460+10.480 = 35.040 = 报告 35.04；
case15: 867.780+611.200 = 1478.460 = 报告 1478.46）⟹ **拆分可信**。
K1 占端到端 **51.3%–94.5%**（全 36 个测量；fused 任务内为 51.3%–76.6%）—— 是主导 kernel。完整表：`perf_tuning/round2_item4/k1_isolated_ab.txt`。

| 任务 | K1 geomean | **K2 geomean（对照：K2 代码未动）** | K1 geomean，仅 `n_blk_full==0`（**对照：生成码逐字节相同**） | K1 geomean，仅 `n_blk_full>0`（受影响集） | K1 最差回退 |
|---|---|---|---|---|---|
| fused (K2g) | **+0.319%** | +0.348% | +0.038% | +0.355% | +0.829% |
| mask (K2m) | **+0.523%** | −0.088% | +0.531% | +0.521% | +0.199% |

符号检验：36 个 K1 测量中 **30 个变快、1 个持平、5 个变慢**（fused 14/1/3，mask 16/0/2 —— 方向一致性强）。
**对照组符号分布**：fused 的 K2g（代码未动）为 **14 负 / 0 平 / 4 正**，与 K1 的 14/1/3 **同形**；
mask 的 K2m（代码未动）为 7 负 / 1 平 / 10 正（K2m 仅 1.8–75 µs，逐 case 噪声达 ±7%，故其 geomean 对照可靠性较低；
mask 任务里更干净的对照是「`n_blk_full==0` 的 K1」，见 §19.4 第 2 条）。

### 19.4 诚实判读 —— **性能中性，不能宣称提速**

**必须如实说明：+0.352% / +0.360% 无法与噪声区分。** 三条对照证据：

1. **fused 任务里，代码完全未动的 K2g 对照 geomean 是 +0.348%**，比 K1 的 +0.319% **还大**。
   K2g 一行代码都没改，它不可能真的变快 ⟹ 这 0.35% 是**会话级整体漂移**（候选那次跑整体略快），
   K1 的同步移动被它完全解释掉。K2g 逐 case 也是 14 负 4 正，与 K1 的分布同形。
2. **mask 任务里，`n_blk_full==0` 的 K1 对照（生成码逐字节相同、UB Δ=0、连 JIT 缓存键都没变）geomean 是 +0.531%**，
   与受影响集的 +0.521% **完全同量级**。case 1 单点甚至 −0.98%。**同一份机器码不可能真的快 1%**
   ⟹ K1 的测量噪声就是 ~0.5–1.0%，而观测到的"收益"正落在这个量级里。
3. 端到端单 case 漂移标定为 **最大 1.71%**（§19.0），而本轮所有单 case 变化都在 ±1.6% 内。

**机理估计（明确标注为估计，不是实测）**：每个满块删除 1 次 `N*4` B 的 UB→UB DataCopy + 1 次
`PipeBarrier<PIPE_V>`（全 V 流水 drain）。以 case 15 计：`n_blk_full=37`，B=64 → 64 行跑在 40 个 vector core 上
（约 1.6 波），观测 K1 −9.40 µs（−1.08%）⟹ 折算每块每行约 0.16 µs，对应 16384 B ≈ 100 GB/s 的有效 UB 带宽，
量级合理。⟹ **删除的工作是真实存在的，方向（30/36 变快）也与机理一致，但幅度（端到端 ~0.2–0.4%，
K1 ~0.3–1.1%）处于本测量精度之下，不能声称统计显著的提速。**

## 20. `[RESULT-#4]`

| 字段 | 值 |
|---|---|
| 优化点 | #4 消除 K1 主块路径为满足 topk rank-1 源要求而存在的 UB→UB 拷贝（`T.copy(z_ub[0,:BV], zt_ub)`）+ 消除 `zt_ub = T.alloc_shared([BV])` |
| bottleneck_type | **transfer**（冗余 UB→UB 数据搬运）+ **sync**（连带 1 次 `PipeBarrier<PIPE_V>`/满块）；次要价值是 **UB 容量** |
| 精度 | **pass**（四门禁全绿，见 §20.1）+ **bitwise 完全一致**（220/220 张量，§18.2） |
| 性能（端到端 geomean） | fused **+0.352%**、mask **+0.360%**；最差单 case 回退 fused +0.285% / mask +0.548%；**0 个 case 回退 >3%** |
| 性能（K1 隔离） | fused K1 **+0.319%**、mask K1 **+0.523%**；**但对照组同量级移动 ⟹ 判定为噪声内** |
| 对比 | 同会话重测的 `ac8eba47` 基线（非引用旧报告），36 case 逐 case 对比见 §19.2/§19.3 |
| UB | 最差 case 188004 → **171620** B；余量 8348 → **24732** B（**+16384 B，×2.96**）；18/18 case Δ = `−BV*4` 常量；`n_blk_full==0` 与 K2m/K2g 完全未变 |
| **采纳判定** | **走 (b)：性能中性但为后续使能 —— "enabling change, perf-neutral"**。(a) 不成立（+0.35% ≪ 3% 阈值）。(b) 三条件全满足：① geomean 在 ±3% 内 ✓；② bitwise 完全一致 ✓（220/220 + harness 36/36 `mismatch=0, max_abs_diff=0`）；③ UB 实测余量确实增加 ✓（8348→24732 B，实测非模型）。⟹ **不回滚，保留改动，由 Orchestrator 最终裁定是否保留** |
| rollback_reason | **不适用**（未回滚）。回滚点仍备妥：`history_version/r4_fused_sampling_impl_s3r2_item4_attempt0.py` = `ac8eba47…` |
| consecutive_no_improvement | **+1**（本轮未产生可判定的性能提升；(b) 类改动的价值在 UB 使能而非提速。Stage 3 中止条件由 Orchestrator 判定） |
| design_error | **no** |

### 20.1 四项门禁（本轮实际复跑，非引用）

| 门禁 | 命令 | 结果 | exit |
|---|---|---|---|
| 全量分层 | `python test_r4_fused_sampling.py --level all` | **74 PRECISION_PASS / 0 PRECISION_FAIL + 38 BOUNDARY_PASS / 0 BOUNDARY_WARN**，`Test Passed!` | **0** |
| 稳定性 | `python test_r4_fused_sampling.py --level stability` | **12 PRECISION_PASS / 0 FAIL**；4 个 tail-adversary 哨兵 TAIL=**313 / 257 / 511 / 261** 全部 `dropped=0`（`dropped>0` 计数 = **0**）；8 config × 20 reps 全 `selfdiff=0 goldenfail=0 invariant=0` | **0** |
| UB-poison 压力 | `python test_r4_fused_sampling.py --level stress` | **8/8 poisoned configs PASS**，全部 `selfdiff=0 goldenfail=0 invariant=0`（reps=200）。← 对本项尤为关键：删除 `zt_ub` 后 topk 直接读 `z_ub`，投毒测试实证无残留 UB 泄漏 | **0** |
| 覆盖门禁 | `coverage_check.py test_r4_fused_sampling.py --proto proto.yaml` | **25 PASS / 0 MISS / 0 N/A → PASS** | **0** |

原始日志：`perf_tuning/round2_item4/gate_{all,stability,stress,coverage}.log`、`gate_exits.txt`。

### 20.2 cann-bench 36-case eval

四次 eval（2 baseline + 2 candidate）全部 **`EXIT_CODE=0`**，每次 **18/18 PASS、0 fail**、
`result_stage=performance`、`contains_performance=True`。单 operator wall 114–123 s（与 round 1 的 ~114 s 一致）。
candidate eval 期间 JIT 缓存**新增 0 条**（§20.3）⟹ **编译耗时未污染性能测量**。

## 21. 交付物同步 / sha256 / JIT 特化

### 21.1 sha256

| 文件 | before | after | 状态 |
|---|---|---|---|
| `examples/R4/r4_fused_sampling.py` | `ac8eba47f77e11342eabb9c4e30a44531b3e655b481b0dc28fd5a66a2738fc5b` | **`19e51e8d8ee126b920cf48c915d863a9be3a349e2c68cf728d35d482b48bcc57`** | **已改（本项）** |
| `cann_bench_submission/cann_bench/_r4_kernels.py` | `ac8eba47…` | **`19e51e8d…`** | **已同步**（由 `build.sh` 在 eval 期间 verbatim 复制；`diff -q` 与主文件**逐字节相同**） |
| `cann_bench_submission/dist/cann_bench-1.0.0-py3-none-any.whl` 内 `_r4_kernels.py` | — | **`19e51e8d…`** | **wheel 内亦为新版**（解包核对） |
| `examples/R4/test_r4_fused_sampling.py` | `02a44b79624e477c570c3c632fcdcef01dcf60041de3372a177aa947d53efb50` | `02a44b79…` | **未改**（与任务书给定值一致；无断言被弱化） |
| `examples/R4/proto.yaml` | `41ab797002568e89fcc9a9eaa58be4f1cdfce20b6406abd0fdbf2ad11f7f8949` | 同 | **未改** |
| `history_version/r4_fused_sampling_impl_s3r2_item4_attempt0.py` | — | `ac8eba47…` | **本项回滚点**（`cp` 建立） |

**测试文件是否需要改**：**不需要，也未改**。本项是纯内部数据搬运重构，不改任何契约
（proto.yaml 的输入/输出/shape_support 全不变）、不改 BV、不改 K_MAX、不改 TAIL 语义 ⟹
`stab_*_bv2048`、`adv_*` 等用例语义完全不变。**无需 Orchestrator 裁决测试改动。**

### 21.2 新增 JIT 特化与编译耗时（**单列，不计入任何性能数字**）

| 项 | 数值 |
|---|---|
| 新增 K1 特化（被交付物使用） | **16**（18 case 中 `n_blk_full>0` 的那 16 个） |
| 新增 K2m / K2g 特化 | **0**（两函数逐字未改） |
| case 1 / case 14（`n_blk_full==0`） | **命中旧缓存，0 新增**（重编译耗时 0.1 s vs 其余 15 个的 5.7–6.1 s）⟹ IR 与缓存键未变，是彻底 no-op 的独立证据 |
| 单个 K1 特化编译耗时（实测） | **5.7–6.1 s** |
| 16 个特化一次性编译总耗时 | ≈ **94 s**（实测：`dump_k1.py` after 全量跑 6.1 s×15 + case15 由前置 smoke test 编译） |
| 新增缓存目录（本轮全部活动合计） | **40**：16 个是交付物使用的 K1 特化；24 个是 `bitwise_capture.py` 的**探针专用** shape（V=4353/4607/4357/152377-B4/152064-B16/16384-B2/2304-B6 等 × K1/K2g/K2m），交付物不会引用 |
| candidate eval 期间新增缓存 | **0** ⟹ eval 全程 warm，**编译耗时未进入 msprof 的 elapsed_us** |
| 副作用 | 24 个探针死条目仅占磁盘，可按需清理 `~/.tilelang/cache`（本轮用的是 `TILELANG_CACHE_DIR=/tmp/opencode/r4_tl_cache`，共 475 条） |

### 21.3 代码风格

- 本项新增/修改的行**全部 ≤100 字符**（编辑区 L185-235 实测 max = 95）⟹ 符合 AGENTS.md「Python 行宽 100」。
  （文件内既有的 >100 行如 L60/L148/L307 等均为**改前既存**，不在本轮范围，未动。）
- `ruff check` 结果：**1 error，`F841 Local variable 'AT1' is assigned to but never used`（L442）**。
  **该告警是既存的**：对回滚点备份 `history_version/r4_fused_sampling_impl_s3r2_item4_attempt0.py`
  跑同样的 `ruff check` 得到**同一条 F841**（round 1 [#2] 删掉消费 `AT1` 的 `topk(K=1)` 后遗留）。
  **本轮未引入任何新的 ruff 告警，也未修这条既存告警**（它位于 `r4_gumbel_sample`，属 #5 的范围，修它是越界）。
  ⟹ 如实上报给 Orchestrator，建议在 #5 或收尾阶段一并清理。

## 22. 如实报告：验证缺口

1. **性能收益无法与噪声区分**（§19.4）。本轮**不声称任何提速数字**。「+0.352%/+0.360%」是实测值，
   但两个对照组（未改代码的 K2g +0.348%、生成码逐字节相同的 K1 `n_blk_full==0` +0.531%）同量级移动，
   故正确读法是**性能中性**。**禁止把它读成"测到了 0.35% 的提速"。**
2. **只跑了单次 A/B（每任务 baseline 1 次 + candidate 1 次）**。未做多次重复取分布
   （round 1 曾对 fused 做过 ×3 重复）。若 Orchestrator 需要把 ~0.3% 量级的差异做成可判定结论，
   需要 ≥3 次重复 + 配对检验；本轮**未做**，因此结论只能是「噪声内」。
3. **`T.tile.sub` 的 scalar-src1 形态（可再省 `m2d` 的 16384 B + 1 次 broadcast）未实施、未测**。
   源码依据充分（`ascend_tile.py:1106-1127`，走 `AscendC::Adds(dst, src0, -src1)`；
   同文件 K2g L534 已有 scalar src1 的 NPU 验证先例），且 IEEE-754 下 `a-b` 与 `a+(-b)` 逐位相同，
   但**这是"应该相同"的推理，不是本硬件上的实测**。本轮为守住「纯搬运重构 ⟹ bitwise 一致」的判据而**有意不做**。
   若做，K1 最差余量可再从 24732 B 提到约 41116 B。**记为后续候选，需独立一轮验证。**
4. **`m_blk[0, 0:1]` 这个 region-out 形态只在 K1 主块路径上验证过**。仓库内先例是
   `fused_sigmoid_gating_delta_rule_varlen.py:238`（`pred_1d[:, 0]`，rank-2 源的列 region），
   与本轮的「rank-1 源 + `[1,1]` 缓冲的 `[0,0:1]` region」不完全同形。
   本轮的可行性依据是**前端校验源码推导 + 18/18 case 编译通过 + 生成码逐字节等于改前**（§16.2），
   证据充分，但**未找到与本形态完全相同的既有用例**。
5. **UB 实测是编译期静态量**（解析生成源的 `GetWithOffset` 布局），**不是运行期实测峰值**。
   该口径与 round 2 完全一致，且等于 memory planner 自己打印的 "Scope shared.ub memory usage"，
   但严格说它反映的是 planner 的分配结果，不含运行期任何额外栈式占用。
6. **`--level stress` 的投毒 pattern 由测试文件固定**（本轮未改测试）。它覆盖了 8 个 config，
   但投毒的具体字节模式是否穷尽了「topk 直接读 `z_ub`」可能遇到的所有残留状态，**未独立论证**；
   依据是 8/8 PASS + `selfdiff=0` 的实测结果，以及 §18.4 的 `z_ub` 生命周期论证（每迭代首行整块覆盖）。
7. **`.orchestrator_state.json` 全程未读写**（禁令 1）。其中「交付 sha256」需由 Orchestrator 从
   `ac8eba47…` 更新为 `19e51e8d…`（若裁定保留本项）。
8. **#4 对 #2 的使能是"预算层面"的，未验证 #2 真能用掉这 16384 B**。剪枝需要多少缓冲取决于其具体算法，
   本轮只证明了 K1 余量从 8348 B 提升到 24732 B（×2.96），**没有替 #2 做可行性判断**。

## 23. 给 Orchestrator 的建议（不属本轮范围）

1. **建议保留本项**。理由：(b) 三条件全满足；bitwise 一致使其**风险面接近零**（不是"实测没退化"，
   而是"生成码可证等价 + 220 张量逐位相同"）；UB 余量 ×2.96 是 #2 的前置条件；
   同时净减 1 次/满块的 16 KB 搬运 + 1 次全流水 drain（真实工作，只是幅度在噪声下）。
   **若 Orchestrator 裁定回滚，回滚点是 `history_version/r4_fused_sampling_impl_s3r2_item4_attempt0.py`（`cp` 恢复，勿手工重敲）。**
2. **`[DESIGN_ERROR]` 判定：不触发**。任务书要求「至少尝试 3 种优化方案后」才可标注，本轮**方案一次即成功**；
   且本项恰恰**证明设计文档过度保守而非错误** —— DESIGN §5.3 N2 把「窄 region 撞 256 B」的正确结论
   外推到了「满块全宽也不可行」，并自陈"未验证"。建议 Orchestrator 安排**修订 DESIGN §5.3 N2**
   （补记：全宽 plain Buffer 走 normal 路径，`zt_ub` 可消除），与 round 2 §5.1/§5.2 指出的 §4.5 UB 模型缺陷一并修。
3. **下一项按批准序是 #1a（split-V）**。提示：round 1 已判定 split-V 属**设计层架构重构**
   （4+ kernel + GM workspace + 双路分派）且威胁 `tok_out`/`valid` 精确匹配门禁，建议交 Stage 1；
   若 Orchestrator 仍要在 Stage 3 内做，**#4 腾出的 16384 B 对它也有用**（split-V 的部分和缓冲）。
   另注：反模式清单 `:60-122`「关注项 B：任务数低于物理 AI Core 数」在 K1 上是**真实命中**的
   （case 2 的 B=1 → `b_num=1`，只用 1 AIC + 2 AIV，而 910b2 有 20 AIC / 40 AIV）——
   这正是 split-V 的动机所在，也说明**小 B 场景的剩余杠杆确实在 V 轴并行化，不在核内微优化**。
4. **本项还使能了一条此前物理不可行的优化**（§14.3 表第 2 行，量化见该节）：
   反模式「纯 AIV memory bound 算子未做流水/双 buffer」的补救需要给 `z_ub` 加双 buffer（+16384 B）。
   **改前**最差 case 6 余量仅 8348 B → 加双 buffer 得 204388 B，**溢出 8036 B，不可能**；
   **改后**余量 24732 B → 188004 B ≤ 196352 B，**静态可行**。
   ⟹ 若 Orchestrator 想继续追 K1 性能，**「z_ub 双 buffer / `T.Pipelined(num_stages=2)`」现在是可测候选了，改前不是**。
   （三个 BV 缓冲全双 buffer 仍需 +49152 B，溢出 24420 B，仍不可行。且按该清单检查点要求，
   加性估算只作参考，**必须实测 memory planning 结果**，因 planner 会做 liveness 复用。）
5. **既存 ruff F841（`AT1`，L442）**建议在 #5（K2g And halving，同一函数）或收尾阶段一并清理。

---

# Round 2 · 第 3 项：`#2` K1 块级阈值剪枝 —— **`[ITEM2_NO_GO]`**（离线研究判定，未写 kernel 代码）

> 详细研究报告：[`round2_item2/feasibility_study.md`](round2_item2/feasibility_study.md)。
> 本节只记编排层要求的四个标记与结论，数据与取证细节一律指向该报告，不重复。

## 24. `[ORDER-CHECK]` 执行序合规

| 检查项 | 结论 |
|---|---|
| 用户批准的执行序（round 2） | **`#2 → #5 → #1a`**（#3 已 BLOCKED 出局，#4 已采纳） |
| 本轮做的项 | **仅 #2**（且仅其**强制前置门**：离线可行性研究） |
| 前置依赖 | **#4 已采纳并交付**（起点 sha256 = `19e51e8d…`）。#4 的意图是"给 #2 腾 UB 预算"（K1 余量 8348 → 24732 B）。**本项研究发现该预算 #2 根本用不到**（只需约 32 B，见 §26.3 / 报告 §7.2）⟹ #4→#2 的排序理由对 #2 不构成约束，但**不构成 #4 的回滚理由**（#4 自身按 (b) 类"性能中性 + bitwise 一致 + UB 使能"采纳，其价值独立于 #2） |
| 是否触碰 #5 / #1a | **否**。未做 K2g And halving、未做 split-V（禁令 5） |
| 是否触碰 BV | **否**。18 case 的 BV 仍全部由 `case.attrs` 提供（16×4096 + 2×2048），kernel 侧 BV 默认值仍是死代码（禁令 4） |
| 是否改测试/契约文件 | **否**。`test_r4_fused_sampling.py` sha256 = `02a44b79…` 全程未变；`proto.yaml` / `cases.yaml|csv` 未触碰（禁令 3）。**无任何需要 Orchestrator 裁定的测试改动诉求** |
| 是否读写编排状态文件 | **否**。`.orchestrator_state.json`（含 `.bak_*`）全程未读写（禁令 1） |
| 回滚点 | **不适用**（无改动可回滚）。**未创建** `history_version/r4_fused_sampling_impl_s3r2_item2_attempt0.py` —— 任务书 §8 要求"改前备份"，但本项**在改前即判定 NO-GO**，`cp` 只会产生一个与 `19e51e8d` 逐字节相同的冗余副本。现有回滚点仍是 `history_version/r4_fused_sampling_impl_s3r2_item4_attempt0.py`（= `ac8eba47…`，#4 的改前基线） |
| 起点 == 终点 | **是**。`r4_fused_sampling.py` sha256 前后均为 `19e51e8d8ee126b920cf48c915d863a9be3a349e2c68cf728d35d482b48bcc57`（`round2_item2/gate_sha256_before.txt` vs `gate_sha256_after.txt`，`diff` 为空） |

## 25. `[IMPL-#2]` 已读文档、行号与关键约束（禁止凭记忆猜 API —— AGENTS.md 核心原则 1）

**本项无实施代码改动**，以下为**离线研究**阶段实际查阅的依据（文件:行号）：

| 查证目标 | 依据（文件:行号） | 结论 |
|---|---|---|
| 块内数据依赖条件分支的**可用形态** | `r4_fused_sampling.py:300-303`（epilogue `with T.If(prefix[0,0] <= p_i[0,0]): with T.Then():`）、`:547-550`（K2g `with T.If(blk_max[0] > best_val[0]):`）—— **均为交付 kernel 内已通过四门禁 + cann-bench 36/36 的 NPU 验证先例** | `T.If/T.Then` 可用于块内 UB 标量条件 |
| 该形态的**生成码是否为真实分支** | `round2_item4/emitted_cc_case15/before.cc:190, :199, :237, :241, :245, :249`（既有 `if (…GetValue(0)…)`）；**本轮新取证** `round2_item2/emitted_cc/K1_case15_pruned.cc:127-137` | **真实 C++ `if`**，两个 `TopK`（=两次 `Sort`）完整位于分支体内 ⟹ **确实被跳过，非 mask 化** |
| `block_max` 是否需额外规约 | `r4_fused_sampling.py:212`（`T.reduce_max(z_ub, m_blk[0,0:1], dim=-1)`，主循环内已存在） | **不需要**，谓词可直接复用 `m_blk` |
| 标量取 `run_v[K_MAX-1]` 的写法先例 | `r4_fused_sampling.py:284`（`tau_k[0] = run_v[k_i[0]-1]`）、`:302-303`（`prefix[0,0] = prefix[0,0] + q2[0,l]`） | 可用 `tau_run[0,0] = run_v[K_MAX-1]` |
| `TopK` 的真实硬件成本（用于 (b) 静态口径） | `src/tl_templates/ascend/common.h:1875-1898`（`TopK` = `Sort` + `DataCopy`）、`:1685-1865`（`Sort` = `ArithProgression` + `Sort32` + 归并树 `numSegs→ceil(numSegs/4)`，每组一次 `MrgSort`） | 4096 元素块 → **48× MrgSort + 144 Sort32 repeats**；`padCount = 0`（红线 2 未被触发） |
| bench 输入分布（决定剪枝率） | `src/kernel_eval/data/data_generator.py:71`（`input_dist="uniform"` 默认）、`:333-345`（`torch.rand`×range+min）；`eval/evaluator.py:317-322`（`case_seed` 公式）、`:254-256`；`cli.py:370`（`eval_seed` 默认 0）；`benches/cann_loader.py:392-394`（`case_id_str = f"{op_dir}_{case_id}"`） | 18 case 的 logits 全部是 **i.i.d. uniform**（区间见各 case `value_range`） |
| msprof 口径（禁令 2：不得用 Python/torch 计时） | `eval/perf_eval.py:88-115`（`_ExperimentalConfig(export_type=[Text], profiler_level=Level1, aic_metrics=AiCoreNone)` + `schedule(wait=0, warmup, active=repeat, repeat=1)`）、`base/perf_strategy.py:_median`（按 kernel `Name` 分组取 `Duration(us)` 中位数） | 探针复用**同一原语、同一统计量**（warmup=3, repeat=5，与 `round2_item4/k1_split.py` 的 `STEPS_PER_CASE=8 / WARMUP=3` 一致） |
| K1/K2 任务内占比 | `round2_item4/k1split_{fused,mask}_cand.json`（= 交付件 `19e51e8d` 的 msprof 位置拆分） | K1 占 fused **0.518–0.765**、占 mask **0.801–0.947** |
| 门禁输入形态（用于门禁失明审计） | `test_r4_fused_sampling.py:112-121`（`L0_CASES`）、`:409+`（`L1_CASES`）、`:1124-1147`（`STAB_CASES`）、`:1150-1155`（`ADV_CASES`）、`:124-126`（`_make_inputs` = `manual_seed(0); randn(B,V)*8.0`）、`:632-664`（L2 特殊值用例均 `B=4,V=4096`） | STAB 最大 `n_blk_full`=**2**、ADV=**1**、L2 特殊值=**1** ⟹ 门禁对剪枝**结构性失明** |

**未凭记忆猜测任何 API**：`T.If/T.Then`、`T.reduce_max`、`T.tile.topk`、`T.tile.gather_mask`、
`T.alloc_shared` 全部沿用交付 kernel 内的既有用法；探针文件由 `probe_k1_nosort.py` **文本变换**生成
`probe_k1_pruned.py`（`diff` 仅 5 处，见报告 §5.2/§6），**未推翻重写**（禁令 7）。

## 26. `[SELF-CHECK]` 本次 Edit 只涉及 #2（且交付件 Edit 数 = 0）

### 26.1 交付件改动 = 0

| 文件 | sha256（研究前） | sha256（研究后） | Edit 数 |
|---|---|---|---:|
| `examples/R4/r4_fused_sampling.py` | `19e51e8d…` | `19e51e8d…` | **0** |
| `examples/R4/test_r4_fused_sampling.py` | `02a44b79…` | `02a44b79…` | **0** |
| `examples/R4/cann_bench_submission/cann_bench/_r4_kernels.py` | `19e51e8d…` | `19e51e8d…` | **0** |
| `examples/R4/proto.yaml` / `cases.yaml` / `cases.csv` | — | — | **0** |
| `examples/R4/.orchestrator_state.json`（及 `.bak_*`） | — | — | **未读写** |

submission 副本与交付 kernel 仍**逐字节相同**（`cmp` → `IDENTICAL`），无需同步。

### 26.2 本轮新增文件（全部在 `perf_tuning/round2_item2/`，不覆盖既有产物）

`feasibility_study.md`、`prune_rate.py`、`gate_blindness.py`、`analytic_prune_bound.py`、
`probe_k1_nosort.py`、`probe_k1_pruned.py`、`probe_dump_cc.py`、`probe_insn_census.py`、
`probe_run_perf.py`、`ceiling_fold.py`、`run_gates.sh`，及产物
`prune_rate_result.json`、`gate_blindness.json`/`.log`、`analytic_bound.json`、
`probe_perf.json`/`.log`、`ceiling.json`、`insn_census.txt`、`emitted_cc/*.cc`、
`gate_{all,stability,stress,coverage}.log`、`gate_exits.txt`、`gate_sha256_{before,after}.txt`。

**探针文件是研究工具，不是交付物，也不被交付件 import。**

### 26.3 红线自查

| 红线 | 结论 |
|---|---|
| 精度红线 1（非 32B 倍数 UB→UB 拷贝） | **未引入**。探针只新增 `tau_run` 一个 `[1,1]` fp32（走 `T.tile.fill`/标量赋值，**非 `T.copy`**）；`copy_ub_to_ub` 计数仍为 256（=1024 B） |
| 精度红线 2（`count % 32 != 0` 的 `topk`） | **未受威胁**。关键论证：剪枝是**控制流**（决定是否执行），**不改变张量长度参数** ⟹ `actual_num` 仍是编译期常量。实测生成码 `TopK(…, 256, 128, 4096)` 与 `TopK(…, 256, 16, 512)` 的 `actualCount` 未变、`padCount = 0`（case 8 为 `2048`/`256`，同样 `padCount=0`） |
| 精度红线 3（D-TIE 分级判据 + `tau_p`/`L` 严格前缀和） | **未触碰**。剪枝条件取严格 `<`（生成码 keep 侧为 `tau_run <= m_blk`，对偶即 `block_max < tau` ⟹ `block_max == tau_k_run` 的块**被保留**），`n_eq` 与前缀和语义不变 |
| 尾块红线（`if TAIL > 0:` 区段禁止修改） | **未触碰**。探针尾块区段与交付件逐行相同；`T.If` 只包裹主循环内的 topk+merge。**交付件全文件 sha256 前后一致**（强于区段级对比） |
| 禁令 2（不得用 Python/torch 计时） | **遵守**。(b) 全部数字来自 `torch_npu.profiler`（msprof）的 `kernel_details.csv: Duration(us)` 中位数；(a) 是 host 侧**统计量**（剪枝块计数），不涉及任何时间测量 |

## 27. 研究结论摘要（数据详见报告）

| 项 | 实测/解析值 |
|---|---|
| **(a) 剪枝率（18 bench case，harness 同种子 uniform 输入）** | **0 / 5478 块 = 0.000000**；最"险"的 case 15 `min_margin = +0.0016`（仍为正 ⟹ 不可剪） |
| **(a') 剪枝率（38 个门禁配置，`randn×8.0`）** | **0 / 2661 块 = 0.000000** |
| **(a'') 剪枝率解析期望上界（与种子无关，Beta 序统计量精确式）** | 全集最大 **1.122e-4**（case 7/15/17，N=37）；`P ≈ exp(−K_MAX/b)`，需 `V ≳ 10^6` 才可观 |
| **(b) topk+merge 占 K1 比例（msprof 实测 6 case）** | **26.38%（N=1）→ 78.77%（N=37）**；`n_blk_full ≥ 4` 区间 **55.29%–78.77%** |
| **(b') 静态指令普查（交叉印证）** | softmax 侧 = 5 趟 4096 全宽 pass + 6 标量；topk 侧 = **48× MrgSort + 144 Sort32 repeats**（case 15/块） |
| **折算 36-case geomean 上限（实测剪枝率）** | **0.000000%** |
| **折算 36-case geomean 上限（解析期望剪枝率）** | **0.002212%** |
| **反事实上限（假想 100% 命中）** | **98.036%**（6 个已探测 case）⟹ 瓶颈在**数据分布**，不在机制 |
| **第 5 点：块内条件分支形态** | **可实现**：真实 C++ `if`（`K1_case15_pruned.cc:127`），Sort 确被跳过 ⟹ **"只能 mask 化致收益归零"的风险不成立** |
| **剪枝谓词自身开销（0% 命中时白付）** | **+0.100% ~ +0.948%**（均值 **+0.185%**）⟹ 实施将**净回退** |
| **门禁可验证性** | **失明**：STAB 最大 `n_blk_full`=2、ADV=1、L2 特殊值=1 ⟹ 即使实施，四门禁也**无法检测**任何剪枝逻辑错误（含 `--level stress` UB 投毒，任务书 §5.4 的假设不成立） |

**go/no-go 判据**：上限 `0.000000%`（实测）/ `0.002212%`（解析）**≤ 3%** ⟹ **停止，不写 kernel 代码** ⟹ **`[ITEM2_NO_GO]`**。

## 28. 附带发现（对 Orchestrator 决策有用，均**未实施**）

1. **⚠️ NaN 下 `>=` 与 `<` 不等价**：为避开 `T.Else` 的自然写法 `keep = (block_max >= tau)`，
   在 `block_max` 为 NaN 时给出"剪掉"，而规范式 `prune = (block_max < tau)` 给出"保留" —— **语义分叉**。
   `tau_k_run` 自身为 NaN 时更严重（后续所有块被剪，`run_v` 冻结在 NaN）。
   门禁里唯一的 NaN 用例 `bnd_sparse_nan_w_out` 是 `B=4,V=4096,BV=4096`（`n_blk_full=1`）⟹ **对该差异完全不敏感**。
   若将来实施，必须用 `T.If/T.Else` 双分支，或显式论证 NaN 不可达。
2. **#4 腾出的 16384 B UB 预算，#2 用不到**（只需约 32 B 的 `tau_run`）⟹ 那笔预算对 #2 闲置，
   建议 Orchestrator 重新评估其归属（如 #1a 的部分和缓冲）。
3. **"减少 K1 的 Sort 工作量"方向本身极有价值**（Sort 占 K1 55–79%、K1 占任务 52–95%），
   但**块级阈值剪枝不是能在 i.i.d. logits 上达成它的手段**。报告 §7.3 列了 3 个替代方向
   （partial-select 代替全排序 / 两级块粒度 / 反模式「任务数低于物理核数」即 #1a），
   **均未实施、未测**，需各自独立立项。
4. **模拟器自检抓出一个会导致相反结论的 bug**：初版误把 `np.partition(...)[-K_MAX:]` 的**无序**切片
   末元素当第 K_MAX 大值，导致 i.i.d. 输入上**虚报 1/8 块可剪枝**（会误判为 GO）。
   修正为保持 `run_v` 降序 + 增加 8 项（4 类构造）先验自检 + 逐块 `tau_k_run` 交叉校验后才得出正确结论。
   ⟹ 提示：任何"剪枝率"类离线研究都必须自带**已知答案的先验用例**，否则极易得出乐观的错误结论。

## 29. `[RESULT-#2]`

| 字段 | 值 |
|---|---|
| 优化点 | #2 K1 块级阈值剪枝（`block_max < tau_k_run` ⟹ 跳过该块的 `topk + merge`，只保留 online-softmax 的 m/s 更新） |
| bottleneck_type | **compute**（K1 主循环的 `Sort` 网络：48× MrgSort + 144 Sort32 repeats / 块，占 K1 的 26–79%）—— 瓶颈定位**成立**，但**该瓶颈无法用本机制削减** |
| optimization_applied | **无**（离线研究判定 NO-GO，未写任何 kernel 代码） |
| baseline_perf | **不适用**（无候选版本 ⟹ 无 A/B）。交付件仍为 `19e51e8d`，其四门禁与 cann-bench 36/36 状态由 Orchestrator 先前复核；本轮**复验四门禁全绿**（§29.1），但**未重跑 36-case eval**（无候选，A/B 无意义） |
| candidate_perf | **不存在** |
| improvement_pct | **0.000000%（上限）**；解析期望口径上限 **0.002212%** |
| test_shape | 剪枝率测量覆盖全部 **18 个 bench case**（B∈[1,128]、V∈[256,152377]、BV∈{4096,2048}、K_MAX∈{256,128}）+ **38 个门禁配置**；成本拆分 msprof 实测覆盖 **6 个 case**（2, 3, 7, 8, 15, 16） |
| precision_validation | **pass**（在**未改动**的交付件上复验，§29.1）。**不适用**于"候选版本精度"——无候选版本 |
| adopted | **no**（**无改动可采纳**；非"采纳后回滚"，而是"实施前即判定不可行"） |
| rollback_reason | **不适用**（未改动 ⟹ 无需回滚）。交付件 sha256 起点==终点==`19e51e8d…` |
| consecutive_no_improvement | **+1**（本轮未产生性能提升。与 #4 的 +1 累计为 **2**。Stage 3 中止条件由 Orchestrator 判定） |
| design_error | **no** |
| design_error_reason | **不触发**。任务书要求"至少尝试 3 种优化方案后"才可标注；本项**在离线研究阶段即判定不可行**，未进入实施、未尝试 3 种方案。且根因是**数据分布（i.i.d. logits）与机制（阈值剪枝）不匹配**，不是 tiling 策略或内存层级安排缺陷；DESIGN.md 从未主张剪枝可行，`#2` 本就是 round 2 的**候选**优化项，本研究正是它应有的前置筛选 |

### 29.1 四项门禁（本轮在**未改动**交付件上实际复跑，非引用）

| 门禁 | 命令 | 结果 | exit |
|---|---|---|---:|
| 全量分层 | `python test_r4_fused_sampling.py --level all` | **74 PRECISION_PASS / 0 PRECISION_FAIL** + **38 BOUNDARY_PASS / 0 BOUNDARY_WARN**，`Test Passed!` | **0** |
| 稳定性 | `python test_r4_fused_sampling.py --level stability` | 8 configs × 20 reps 全 `PRECISION_PASS`（`selfdiff=0 goldenfail=0 invariant=0`）；4 个 tail-adversary 哨兵 TAIL=**313 / 257 / 511 / 261** 全部 **`dropped=0`**（另含对照 TAIL=2048 / TAIL=0） | **0** |
| UB-poison 压力 | `python test_r4_fused_sampling.py --level stress` | **8/8** poisoned configs `PRECISION_PASS`，`reps=200 selfdiff=0 goldenfail=0 invariant=0`，`Test Passed!` | **0** |
| 覆盖门禁 | `coverage_check.py test_r4_fused_sampling.py --proto proto.yaml` | **25 PASS / 0 MISS / 0 N/A → PASS** | **0** |

原始日志：`round2_item2/gate_{all,stability,stress,coverage}.log`、`gate_exits.txt`（含 `GATES_DONE`）。

> **口径声明**：这四项门禁证明的是"**研究过程没有污染交付件、当前状态仍全绿**"，
> **不是**"某个优化通过了门禁"。本项**没有**候选版本，因此也**没有**
> 36-case geomean 提速 / worst-case delta / 逐 case 对比 / UB 前后对比 / 新增 JIT 特化数 —— 一律**不适用**。

### 29.2 验证缺口（如实列出，详见报告 §9）

1. (b) 只探测 **6 / 18** 个 case；其余 12 个标记 `n/p`，**未外推**（因剪枝率恒 0，不影响判定）。
2. 探针 `probe_k1_pruned.py` **未做输出正确性比对**（NO-GO 未实施，无需验证）。
3. msprof 探针**未复现 harness 的 `CannBenchCacheClean`**（每 step 前清 L2）⟹ 绝对 µs 可能略乐观；
   但三变体条件完全相同，**比值不受影响**（已在 `probe_run_perf.py` docstring 显式声明）。
4. `kernel_details.csv` 行数偶尔多于 5（case 7/15/16 的 real/pruned 出现 6–7 行）；取**中位数**故稳健，
   但多余行成因**未定位**（`nosort` 变体 6/6 case 均恰为 5 行）。
5. `K1_share_of_task` 取自**上一会话**的 `round2_item4/k1split_*_cand.json`（存在 ~0.3–1.7% 会话漂移）；
   用的是同会话内 K1/K2 **比值**，漂移大部分抵消，且它只进入"反事实上限"（informational），**不进入判定链**。
6. 解析模型假设 logits **连续 i.i.d.**（bench 的 uniform 配方满足）。若生产 logits 有**空间有序性**，
   期望剪枝率会显著上升（自检的降序用例 7/8 命中即极端例子）——**cann-bench 契约内不存在此类输入**，
   故对采纳判据无影响；若 Orchestrator 关心生产场景，需单独取样。
7. **未测量 `T.Else` 双分支形态**（§28.1 的 NaN 安全写法）在 Ascend 上能否正确 codegen。
8. `.orchestrator_state.json` 全程未读写；其中"交付 sha256"仍应为 `19e51e8d…`（**本项未改变它**）。

### 29.3 建议下一项

按批准序，下一项是 **`#5` K2g And halving**。提示（非本轮范围，未验证）：
`r4_fused_sampling.py:443` 的 `BV2 = 2*BV` 过分配正是该缺陷的现行 workaround，
若 #5 能消除它，K2g 的 UB 占用（`k2g_4096 = 148452` B）可望显著下降；
另可顺带清理既存 ruff F841（`AT1`，`:442`）。
**但请先注意**：K2g 占 fused 任务仅 23.5%–48.2%（= 1 − K1_share），且 #5 的收益是 **UB 容量**而非时间，
按本轮同样标准，建议**先做同规格的离线可行性研究**再决定是否实施。
