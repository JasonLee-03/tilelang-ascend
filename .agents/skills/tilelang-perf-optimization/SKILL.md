---
name: tilelang-perf-optimization
description: TileLang 算子性能调优与潜在性能劣化模式检查。提供性能数据采集、瓶颈诊断、优化实施、效果验证能力；也用于生成或评审算子时对照常见性能劣化模式示例检查当前 kernel 代码。内置 attention 家族（MHA/GQA/MLA/FlashAttention）等 CV 融合（Cube+Vector 跨核耦合）算子的深度调优实战手册（14 会话实证：机制定律、立项判据、消融上界探针法、msprof/camodel 读数纪律）。触发：算子精度通过后需要优化性能、性能不及预期时；用户提到 attention/MHA/FA/GQA 算子调优、cann-bench 提分、msprof/camodel 归因与读数解读（ratio / wait_id / stall 占比 / pipe 利用率含义）、跨核流水重叠度、CV 耦合瓶颈分析、调优候选立项/回滚判定时，务必使用本 skill。
---

# TileLang 性能优化

## 工作流程

```
Step 1: 基线采集（性能 + 精度）
  → Step 2: 算子类型判断
  → Step 3: 阅读参考文档并识别优化点（输出到 optimization_log.md）
  → Step 4: 逐项实施优化点
  → Step 5: 效果验证（性能 + 精度）
```

## 核心约束

- **逐项实施**：每次 Edit 只改一个优化点，改完立即验证
- **精度优先**：精度未通过禁止性能优化
- **性能验证**：必须使用 `msprof op`，禁止用 Python/Torch 计时
- **Host 轻量化**：禁止 host 侧全量数据搬运（`F.pad`、`.contiguous()`、`.to(dtype)` 等），必须移入 kernel

## 参考文档

- **优化指南**：[optimization-guide.md](references/optimization-guide.md)
- **反模式清单**：[performance-antipatterns.md](references/performance-antipatterns.md)
- **API 用法**：[tilelang-api-best-practices](../tilelang-custom-skill/tilelang-api-best-practices/SKILL.md)
- **编程模式**：[tilelang-programming-model-guide](../tilelang-custom-skill/tilelang-programming-model-guide/SKILL.md)
- **cube最佳实践**：[cube_optimization_path.md](references/best-practices/cube_optimization_path.md)
- **vector最佳实践**：[vector-practices/](references/vector-practices/)
- **attention/CV 耦合深度调优手册**：[attention_tuning_playbook.md](references/best-practices/attention_tuning_playbook.md) —— Step 2 判定为**混合型（CV 融合）**时必读；含测量纪律（msprof/camodel 读数陷阱）、归因方法学（双向 A/B、带宽墙四步法、毒化矩阵）、立项前判据（杠杆性质分类、GATE 0 消融上界门）、机制定律库（吸收律、守恒律、bN 上界定理）、守卫工程与候选治理
---

## 执行步骤

### Step 1: 基线采集

在 `examples/{op_name}/` 下查找含 `@tilelang.jit` 的脚本，运行：

```bash
msprof op --kernel-name="main_kernel" --output=./msprof_output python ./examples/{op_name}/<script_name>.py
```

精度未通过 → 禁止后续步骤。

### Step 2: 算子类型判断

**生成翻译后的 Ascend C 代码**：

在算子脚本中，JIT 编译返回的函数对象调用 `get_kernel_source()` 可获取翻译后的 Ascend C 代码：

```python
func = jit_func(batch=B, seq_len=S, ...)
print(func.get_kernel_source())
```

运行脚本后，从输出中搜索关键字判断算子类型：

| 判断依据 | 类型 | 典型算子 |
|---------|------|---------|
| `IS_ASCEND_AIC` 出现 | Cube 型 | GEMM、MatMul、Linear |
| `IS_ASCEND_AIV` 出现 | Vector 型 | RoPE、Softmax、Add |
| 两者均出现 | 混合型 | FlashAttention、SparseFlashAttention、MHA/GQA/MLA |

**混合型（CV 融合）判定后**：在进入 Step 3 之前通读 [attention_tuning_playbook.md](references/best-practices/attention_tuning_playbook.md) 的 §0-§4。CV 耦合闭环算子（Cube 与 Vector 经 workspace/L0C 通道互为生产者-消费者）的读数语义与立项判据和单管算子有系统性差异——「让某一侧更快」的直觉候选会被对侧等待 1:1 吸收（吸收律），跳过该手册直接列优化点清单大概率浪费迭代预算在结构性无效候选上。

### Step 3: 识别优化点（强制，禁止与 Step 4 合并）

先读取 `optimization-guide.md` 的目录和各章节标题 + `performance-antipatterns.md` 的各条目标题，根据算子类型（Step 2 判定）和算子实现特征初步筛选出可能适用的优化点清单。再针对每个候选优化点，读取其"适用场景"、"约束"、"使用条件"等描述，确认是否真正适用。如果是 cube 核额外参考 `best-practices/cube_optimization_path.md`，如果是 vector 核额外参考 `vector-practices/` 目录下的文档，如果是混合型（CV 融合）额外参考 `best-practices/attention_tuning_playbook.md`。

**混合型（CV 融合）候选附加门禁**（playbook §3，Part A 清单必须体现）：

1. 每个候选先做**杠杆性质前置分类**（§3.1）：交接结构删除类 / 单侧吞吐类 / 结构框架类——分类决定判决路径；
2. **单侧吞吐类**（加速/减字节/削指令/宽化）在 wall 双侧各有大等待余量或已有 1:1 吸收实证时不立项；其余情况必须先过 **GATE 0 消融上界门**（§3.2，≈25 分钟 timing-only 探针钉死 EV 天花板，上界 < 采纳门即证据性关闭，不进入实现）；
3. **交接结构删除类**优先立项，EV 用 flat 收益模型预估（§4.9：批深空缺率 × 任务边界密度）；
4. 缩块/调用数×N 类先做固定段税预算带（§3.5，EV 报名义/净双口径）；
5. 立项引用模型聚合项时逐项下钻到可攻击加数（§3.6 G-8），静态预算审计「过门」不构成可行性证据（G-9）。

在 `optimization_log.md` 中输出：

**Part A 优化点清单**：逐条标注适用/不适用 + 原因 + 参考文件行号。`pass_configs` 不是独立优化点，是伴随修改。

```
[#1] [名称]（参考: optimization-guide.md L445-L650 §2.13）：[适用/不适用] — [原因]
```

若候选会扩大现有 kernel 的调用域（新增 shape、batch、dtype、尾块或多 stage），
Part A 必须先写：

```text
[KERNEL-REUSE-PRECHECK]
candidate: <优化点及 reference 章节>
expanded_domain: <新增调用域>
compatibility_dimensions: <从该章节读取，不自行猜测>
unknown_or_false: <none/列表>
plan: direct-reuse / repair-kernel-first / new-kernel-first
```

存在 `unknown/false` 时，ORDER-PLAN 不得写 direct reuse。

> **重点**：每节都需读取其"适用场景"和"约束"描述确认是否适用。仅当章节标题明确标注了特定算子类型专属（如"Cube 核"）且当前算子不属于该类型时，才可初步排除；其余所有章节必须读约束确认，不得仅凭标题或算子类型跳过。

**Part B `[ORDER-PLAN]`**：分析依赖关系，排出实施顺序链。依赖分析三条规则：
1. **布局依赖**：改变 layout 的优化排在依赖此 layout 的优化之前
2. **数量依赖**：涉及预算的优化排在改变 buffer 数量的优化之后
3. **配置依赖**：涉及 pass_configs 的优化在相关功能实施后才改动

```
[ORDER-PLAN] 实施顺序：
1. [#N] [名称] — 前置依赖: [无] — 理由: [...]
2. [#M] [名称] — 前置依赖: [#N] — 理由: [...]
```

### Step 4: 逐项实施

**固定优先级**：先静态分析（对照 `performance-antipatterns.md`），再 P0 Host 侧优化（`optimization-guide.md` §2.12）。P0 完成后 Host 侧只允许零拷贝形状变换。

**后续优化点**按 `[ORDER-PLAN]` 逐个实施，每个走 6 子步骤：

```
0: ORDER-CHECK → A: Read 文档 → B: Edit 代码 → C: msprof op 验证 → D: 记录结果 → (失败) E: 重读文档修复
```

**门禁**：`[ORDER-CHECK]` 未写禁止 Read；`[IMPL-#N]` 未写禁止 Edit；`[RESULT-#N]` 未写禁止下一个。

Edit 调用方、dispatch 或 planner 前，再写 `[KERNEL-REUSE-AUDIT]` 复核候选章节规定的
兼容性维度。任一项仍为 unknown/false，先修复或新建 kernel，并单独验证后再扩大调用域。
不同优化点使用各自 reference 章节的维度，不套用其他优化的专属字段。

**日志格式**：
```
[ORDER-CHECK] 准备实施: [#N] [名称] | 前置依赖: [#1 ✅ / #2 ❌] | 结论: [✅/❌]
[IMPL-#N] 已阅读 <文件> L行号（§X.X），关键约束: ...
[KERNEL-REUSE-AUDIT] 依据: <章节> | 维度及结论: <...> | 结论: <reuse/repair/new>
[SELF-CHECK] 本次 Edit 只涉及 [#N]
[RESULT-#N] 优化点: [名称] | 精度: [pass/fail] | 性能: [X us] | 对比: [+/-X%]
```

**Double Buffer 特殊要求**：实施前必须完成 `[DB-ANALYSIS]`（Q1: 循环内有 MTE3？Q2: 有跨迭代累加器？Q3: 选同步方式），未完成禁止写代码。

**最佳实践参考**：

| 算子类型 | 文档 | 核心优化技术 |
|---------|------|-------------|
| Vector 型 | [RoPE 优化](references/best-practices/rope-developer-mode.md)、[归约遍数融合](references/vector-practices/vector_reduce_pass_fusion.md) | NPU 内动态生成 Mask、Tile API 向量化、参数简化 |
| Cube 型 | [GEMM Intrinsic](references/best-practices/gemm_intrinsic_optimize.md) | 多缓冲流水线、细粒度 Flag 同步、MMA intrinsic、L0 分块、负载均衡 |
| CV 融合型 | [Flash Attention](references/best-practices/flash_attn_optimize.md)（技术点实现）、[Attention CV 耦合调优手册](references/best-practices/attention_tuning_playbook.md)（判据/定律/纪律） | num_stages 流水线、批量 Softmax、Cross-core Semaphore、数据布局优化、flat (task,kv) 槽流；**多 shape 适配**（BSND 免转置、Sq==1 decode 窄块、加性 mask 屏蔽变长 Skv） |

### Step 5: 效果验证

每个优化点后执行：精度验证 → `msprof op` → 记录 → 对比基线。精度失败时保持优化调试，不撤销。

**CV 融合型附加验证纪律**（playbook §5/§7）：精度门按变更性质分层（调度类 = vs 改前态 BITEXACT 主判据；数值类 = 阻塞口径）；压力/探针门必须核对**路径覆盖**（继承配置可能全部打在旧路径上，§5.3）；A/B 验证一律 `disable_cache=True` 或会话专属缓存目录（§5.4 JIT 缓存键缺口）；回滚执行三重验证（md5 + 行数 + IR 守卫集还原，§5.5）；采纳后重判 pacing 管身份（§2.6）；关闭/回滚候选按 §7 归档（前提清单 + 失效触发器 + 推演负/实测负分级）。

调试手段：`T.printf`、`T.dump_tensor`、`get_kernel_source()`，详见 [Programming Guide](../../../docs/TileLang-Ascend%20Programming%20Guide.md)。

迭代终止：达到目标或连续 3 次无提升则中断上报。

---

## 优化记录

保存在 `examples/{op_name}/perf_tuning/`：
- `baseline.json` - 基线性能
- `optimization_log.md` - 优化记录
- `final_report.md` - 最终报告
