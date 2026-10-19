新增 `examples/mha` 算子，实现 Multi-Head Attention（MHA）前向（BSHD，Expert 模式三分派 mma
流水），代码按 wrapper + 共享层 + 三形态模块布局（5 个 .py）。cann-bench 20 case 精度全通过，
平均加速比对比 torch-npu 基线为 0.97x（本地 910B2 实测口径，综合得分 73.13）。

## Summary

新增 Multi-Head Attention（MHA）forward 算子（`examples/mha/`，wrapper + 共享层 + 三形态模块，
共 5 个 .py），BSHD 布局，Expert 模式（显式 `T.Scope("C"/"V")`、显式 L1/UB/L0A/L0B/L0C 分配、
显式 event flag 与跨核 semaphore）：

```
y = softmax(Q @ K^T * s) @ V,    s = scaleValue if scaleValue > 0 else 1/sqrt(D)
```

`is_causal` 采用**右下角对齐**掩码（`scores[..., i, j] = -inf` for `j > i + (S_kv - S)`）；
全掩码行仅域外可达，经 `sumexp + 1e-30` 保护输出 0。Q/K/V 为 fp16/bf16，score 主链与 O
累加器为 fp32。

### 模块划分

| 模块 | 层次 | 职责 |
|------|------|------|
| `mha.py` | wrapper | 算子契约 docstring、形态路由表、`mha_kernel(...)` 分发器、`mha(...)` host 包装、CLI 自检 |
| `mha_common.py` | 共享层 | `pass_configs`、流水常量、配置派生与分派谓词的唯一源、共享 V scope 宏工厂 |
| `mha_form_a.py` | 形态本体 | `mha_form_a_kernel` —— Form A（S > 16 ∧ D ≤ 128）C-scope 本体 |
| `mha_form_b_mma.py` | 形态本体 | `mha_form_b_mma_kernel` —— Form B-mma（S > 16 ∧ D == 256）手动 K/N-split 本体 |
| `mha_form_b_gemm_v0.py` | 形态本体 | `mha_form_b_gemm_v0_kernel` —— Form B gemm_v0（S ≤ 16，全 D）decode 兜底本体 |

- `mha.py` **不含任何 `T.*` 代码**：`mha(query, key, value, scaleValue=-1.0, is_causal=False)`
  只做元数据校验 + JIT 分发，零张量操作
- `mha_common.py` 具体承载：`pass_configs`（4 项全 False）、`NUM_STAGES = 8`、
  `CROSS_INTERVAL = 1`、`_cube_core_num()`、三个形态标识、`MhaConfig`（frozen dataclass）、
  `derive_config()`（host 侧配置派生**唯一源**：`block_M`/`block_N`/`gemm1_kL0` 三档体系、
  分派谓词、`half_M`、`causal_offset`、掩码判定、`s_blocks`/`kv_iters`/`single_kv_block`/
  `flat_v`/`kv_db`、任务网格 `core_num`/`q_tasks`/`r_tasks`、SEM/SIG id 空间）、
  `select_form()`（三分派谓词**唯一源**）、`make_emit_v()`（内含 `emit_v`，全仓库唯一一份
  `@T.macro`）

### wrapper 层分发

`mha_kernel(...)` 是**纯 Python 分发器**（不带 `@tilelang.jit`），函数体 = `select_form(...)`
取形态 → 形态路由表转发到对应工厂。其签名与返回值语义与拆分前的单文件布局完全一致：返回对象
仍暴露 `.get_kernel_source()`、仍可 `kernel(q, k, v)` 调用；直接调用三个形态工厂产出**同一份**
kernel。三个形态工厂的装饰器逐字为
`@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)`，形参表完全一致
`(B, S, S_kv, N, D, dtype="float16", scaleValue=-1.0, is_causal=False)`，prim_func 一律命名
`main`。

**单一源纪律**：形态谓词只在 `derive_config`/`select_form` 出现一次；`emit_v` 只在
`mha_common.py` 定义一次；三形态文件不得复制二者，只可 import。`emit_v` 必须唯一有硬性技术
理由：TVMScript 在解析期解析 `T.*` 帧，普通 Python 发射器在 prim_func 内调用会**静默丢弃**
`T.*` 发射——kernel 仍编译通过，但生成体缺失、aicore 挂在孤立 wait 上。故 `emit_v` 是
`@T.macro` 且全仓唯一。

模块拆分为**纯结构性重构，零语义变更**，有机器证明：拆分前后 22 个 shape 的生成设备码 `.cc`
**逐字节同一**（三方比对：拆分前单体 / 经 wrapper 分发 / 绕过 wrapper 直调三工厂，聚合 sha256
`abd65ef6a35ac3263121039778379a07ac67dda5fdcc76440586590c1a6c3828`）；`emit_v` 源文本逐字节
同一（sha256 `b593e7d86ac597fbaf96729c33f5ae2a`）；各 prim_func 体 `ast.dump` 同一。

### 关键设计

host 仅凭 `(S, D, S_kv)` 在三种**互斥且穷尽**的 C-scope 形态中选一，单个生成 kernel
只承载一种：

- **Form A（S > 16，D ≤ 128）：T.mma 全流水** —— L0A/L0B/L0C 双槽 + 手动 `side = i % 2`
  乒乓，L1 操作数 ZN/NZ 标注，逐 pipe-pair 细粒度 event flag。`block_N == D` 时 GEMM2 复用
  GEMM1 的 L0 池；单 KV 块子域由 FIX pipe 直接从 L0C 以运行期行数写回 `Output`
  （此处 `O = P_hat @ V` 已为终态）
- **Form B-mma（S > 16，D == 256）：手动 K/N-split mma 流水** —— GEMM1 将 K 拆两个
  `KC = 128` chunk，由 `mma(init=True → False)` 链接；GEMM2 将 N 拆两个 `NC = 128` chunk
  累积进窄 L0C 池并各自写 workspace_3 列段（cube 拒绝向 L0C 列子区 mma 的绕过方案）
- **Form B gemm_v0（S ≤ 16，全 D）：decode 兜底** —— `T.gemm_v0` 内部按 `kL0Size` 三档
  32/64/128 管理 L0A/L0B，其自平衡 prologue/epilogue pipe flag 覆盖所有段边界，无需粗 barrier
- **共享 V scope（`emit_v`，`@T.macro`）** —— C/V 经三个显式 GM workspace 以 8 深批
  （`NUM_STAGES = 8`）交换数据：workspace_1 = fp32 scores、workspace_2 = P、
  workspace_3 = fp32 PV；每个 AIV 拥有 `half_M = block_M // 2` 行
- **flat (task, kv) 连续令牌流** —— 多 KV 块 prefill 以一维 `(task, kv)` 索引批量迭代 slot，
  使一个 batch 可跨 task 边界而不排空流水；causal 下 `eff_kv_iters` 逐 task 运行期非均匀，
  故 `(task, kv)` 映射实现为 C/V 双侧逐字相同的标量游走状态机（从不经 workspace 传递）
- **单 KV 块任务批处理** —— `kv_iters == 1` 时把 `NUM_STAGES` 个 task 批为一协议轮，以手动
  slot 轮转表达 C/V 重叠（该模式 `T.Pipelined` 不可用）
- **三档 tile 系统** —— `block_M / block_N / gemm1_kL0` 每档取满足片上预算（32 KB L0B 乒乓 +
  511.75 KB L1）的最大值；行/列尾块以运行期 extent 处理，`S`/`S_kv` 无需与 tile 对齐

精度不变量（4 项，须与 golden 逐位一致，不得重排/重折叠）：① 双舍入栅格链
`r16(qk) → *s(fp32) → r16(scores)`；② 单 KV 块预归一化特化（除法在 dtype 舍入前完成）；
③ 每次最终归一化的 `sumexp + 1e-30`；④ 右下角掩码链语义。

同步模型：每个 event id 静态预算，init 预置首个消费者会等待的令牌、destroy 精确消费、
每迭代净零；跨核 semaphore 严格 1:1 Set/Wait；条件 DMA 必须把其全部 flag 放在同一条件内。

边界覆盖：`D ∈ {64, 128, 256}`、fp16/bf16、decode（S ≤ 16）/prefill（S > 16）、causal 与非
causal、单 KV 块与多 KV 块、非整除行/列尾块、`scaleValue` 显式与自适应；GQA 不涉及
（Q/K/V 同 N）。

## Changes

**算子代码（5 个文件）**

- Add `examples/mha/mha.py`（**wrapper 层**：算子契约 docstring + 形态路由表 +
  `mha_kernel(...)` 纯 Python 分发器 + `mha(query, key, value, scaleValue=-1.0,
  is_causal=False)` host 包装（元数据校验 + JIT 分发，零张量操作）+ `__main__` CLI 自检。
  **本文件不含任何 `T.*` 代码**）
- Add `examples/mha/mha_common.py`（**三形态共享层**：`pass_configs`（4 项全 False）/
  `NUM_STAGES = 8` / `CROSS_INTERVAL = 1` / `_cube_core_num()` / 三个形态标识 /
  `MhaConfig`（frozen dataclass）/ `derive_config()`（host 侧配置派生**唯一源**：
  `block_M`/`block_N`/`gemm1_kL0` 三档体系、分派谓词、`half_M`、`causal_offset`、掩码判定、
  `s_blocks`/`kv_iters`/`single_kv_block`/`flat_v`/`kv_db`、任务网格 `core_num`/`q_tasks`/
  `r_tasks`、SEM/SIG id 空间）/ `select_form()`（三分派谓词**唯一源**）/ `make_emit_v()`
  （内含 `emit_v`，**全仓库唯一一份** `@T.macro`））
- Add `examples/mha/mha_form_a.py`（`mha_form_a_kernel` —— **Form A**（S > 16 ∧ D ≤ 128）
  C-scope 本体）
- Add `examples/mha/mha_form_b_mma.py`（`mha_form_b_mma_kernel` —— **Form B-mma**
  （S > 16 ∧ D == 256）手动 K/N-split 本体）
- Add `examples/mha/mha_form_b_gemm_v0.py`（`mha_form_b_gemm_v0_kernel` —— **Form B gemm_v0**
  （S ≤ 16，全 D）decode 兜底本体）

**测试（1 个文件）**

- Add `examples/mha/test_mha.py`（分层测试套件：golden + L0 精度门 + L1（含**分派守卫 16 例**）
  + L2 + Boundary + perf 哨兵 + `--level` 分发 + pytest 入口）

**设计文档（1 个文件）**

- Add `examples/mha/DESIGN.md`（设计文档：三档 tiling 体系、event id 预算、精度门与三分派形态
  推导，含模块拆分说明与行号引用基线声明）

**本 PR 不包含**：逐迭代调优日志、实现历史快照、在线平台提交夹与框架补丁等开发产物刻意不入
本 PR。本 PR 不含任何框架改动，算子在 stock tilelang-ascend 上即可编译运行。

## Testing

本地分层套件：

```bash
python examples/mha/test_mha.py --level l0     # L0 精度收敛门
python examples/mha/test_mha.py --level all    # 全套 + perf 哨兵
```

- `--level l0` → **10/10** `[PRECISION_PASS]` + `Test Passed!`
- `--level l1` → **19/19** + `[L1-DISPATCH] 16/16`（structural=True）+ `Test Passed!`
- `--level all` → 输出 `Test Passed!`（EXIT=0）：**L0 10/10 · L1 19/19 · L1-DISPATCH 16/16 ·
  L2 12/12（correctly rejected）· Boundary 6/6 · perf 哨兵 4/4**（`PERF_PASS`，kernel_path
  6.2–8.2 s ≪ 120 s 门，`all_sentinels_ok=True`）
- 覆盖门禁：`coverage_check.py` 对 `test_mha.py` 的应覆盖维度核验 → **19 PASS / 0 MISS /
  0 N/A**
- 精度口径：blocking = mixed tolerance（dual gate：matched_ratio + max_abs_error）AND
  `MARE_abs < 10×Threshold`（fp16 9.77e-3 / bf16 7.81e-2）
- ruff check / ruff format --check：6 个 .py 全 clean（仓库根 `pyproject.toml`，
  line-length 140）
- 防作弊合规：QK / softmax / PV 全部计算在 `@tilelang.jit` kernel 内；host wrapper 仅做
  shape/dtype/contiguity 校验、`torch_npu.npu.get_device_properties(0).cube_core_num`
  metadata 查询与 JIT dispatch，无 `matmul`/`softmax`/`sdpa` 或同名 CANN 内建算子派发
- 运行环境：stock tilelang-ascend，**无需任何框架补丁**；算子代码对框架侧可选同步扩展的引用为
  **0 处**（kv_db decode 域使用 `T.gemm_v0` 默认 prologue 的全 MTE2 管排空语义）

**分派守卫**（模块拆分后新增，挂 L1 阻塞层）：16 个 shape 断言 `select_form()` 返回预期形态
（含 `S > 16` 边界两侧）、断言其闭合于三个形态标识、断言 wrapper 路由表为 3 个互异可调用对象、
断言直接 import 的三工厂互异。断言对象为纯 Python 标量，设备无关。

三形态真机 CLI 自检（`python examples/mha/mha.py ...`）全部 `Test Passed!`：

| 形态 | 命令要点 | matched |
|------|---------|---------|
| Form A | 默认（fp16，B=2 S=S_kv=128 N=8 D=64） | 0.99997 |
| Form A | `--causal --D 128` | 0.99978 |
| Form B-mma | `--S 32 --S-kv 512 --D 256 --N 4` | 0.99994 |
| Form B gemm_v0 | `--S 1 --S-kv 512 --D 256` | 1.00000 |
| Form B gemm_v0 | `--S 1 --S-kv 2048 --D 128` | 1.00000 |

**各模块独立可执行自检入口**：`examples/mha/` 下 5 个算子 `.py` 每个都可独立执行——
`cd examples && python ./mha/<module>.py`——各自以退出码 0 结束并在末行打印
`Test Passed!`。原因：`examples/bench_test.sh` 的遗留运行器会把 `examples/<op>/`
下几乎每个 `.py` 当脚本执行（`find -maxdepth 2 -name "*.py"` 收集，仅排除
`__init__.py` / `*_golden.py` / `sfa_golden.py` / `utils.py`），并要求输出含 pass
token（`Kernel Output Match` / `Test Passed!`），无可执行入口的纯库模块会被判
FAILED。自检内容：三个形态模块的 `__main__` 调用共享的
`mha_common.run_form_self_check()`——先断言 shape 确属本模块形态（防止悄悄测到
别的形态），再直接用本模块工厂建 kernel、真机运行，与 torch 参考按上文同一混合
容差门对比（Form A：B=2 S=S_kv=128 N=8 D=64；Form B-mma：B=1 S=32 S_kv=512 N=4
D=256 多 KV 块域；Form B gemm_v0：B=2 S=1 S_kv=512 N=8 D=256 kv_db 域）；
`mha_common.py` 的 `__main__` 为设备无关快检（630 点网格断言三分派互斥且穷尽、
`derive_config` 派生不变量、`make_emit_v` 的 hygienic TIRMacro 契约与闭包自由名
零缺失，纯 host 标量断言，不编译不上机）；`mha.py` 为既有 CLI 真机自检。全部
入口均由 `if __name__ == "__main__":` 守卫，import 依旧零副作用；自检为纯 host
侧代码，新增后 22-shape 生成设备码与拆分基线逐字节一致（聚合 sha256
`abd65ef6…c3828` 不变）。实机验证：5 脚本串行与 5 进程并行（模拟运行器并行 +
JIT 缓存锁竞争）各一轮，10/10 exit 0 + token 命中。

## cann-bench 算子评测报告

评测代号：cann_performance_eval_20261006_003831

| 指标 | 数值 |
|------|------|
| 评测算子数 | 1 |
| 总用例数 | 20 |
| 通过用例数 | 20 |
| 失败用例数 | 0 |
| 通过率 | 100.00% |
| 综合得分 | 73.13 |

### MHA（level4/mha）

| 指标 | 数值 |
|------|------|
| 用例数 | 20 |
| 通过数 | 20 |
| 失败数 | 0 |
| 通过率 | 100.00% |
| 平均加速比 | 0.97x |
| 编译/运行得分 | 20.00 |
| 精度得分 | 30.00 |
| 性能得分 | 23.13 |
| 得分 | 73.13 |

| 用例ID | 状态 | 耗时(μs) | 加速比 | 精度误差 |
|--------|------|----------|--------|----------|
| level4/mha_1 | ✅ | 10.66 | 0.94x | 0.000488 |
| level4/mha_2 | ✅ | 45.40 | 0.72x | 0.003906 |
| level4/mha_3 | ✅ | 135.20 | 0.97x | 0.003906 |
| level4/mha_4 | ✅ | 28.04 | 1.17x | 0.000488 |
| level4/mha_5 | ✅ | 16.00 | 0.66x | 0.000122 |
| level4/mha_6 | ✅ | 25.56 | 1.28x | 0.001953 |
| level4/mha_7 | ✅ | 863.10 | 0.50x | 0.000488 |
| level4/mha_8 | ✅ | 862.90 | 0.50x | 0.003906 |
| level4/mha_9 | ✅ | 874.06 | 0.39x | 0.000488 |
| level4/mha_10 | ✅ | 52.92 | 1.01x | 0.000031 |
| level4/mha_11 | ✅ | 308.86 | 0.92x | 0.000061 |
| level4/mha_12 | ✅ | 308.28 | 0.88x | 0.000488 |
| level4/mha_13 | ✅ | 558.42 | 0.88x | 0.000061 |
| level4/mha_14 | ✅ | 378.26 | 1.39x | 0.000122 |
| level4/mha_15 | ✅ | 421.88 | 0.82x | 0.000061 |
| level4/mha_16 | ✅ | 560.98 | 0.78x | 0.000061 |
| level4/mha_17 | ✅ | 496.92 | 1.14x | 0.000977 |
| level4/mha_18 | ✅ | 422.00 | 0.97x | 0.000061 |
| level4/mha_19 | ✅ | 22.90 | 1.43x | 0.000488 |
| level4/mha_20 | ✅ | 15.42 | 2.13x | 0.000000 |

（cases 4/6/14/17/19/20 已超基线；decode 短序列手工调优域表现最佳，mha_20 达 2.13x。）

## Known Limitations & Next Steps

优化轨迹 **0.13x → 0.97x**（Developer 轨 15 迭代 0.13x→0.55x；Expert rev3 13 迭代
→0.91x；rev4 Form B→A mma 化 + flat 全域 →0.97x）。剩余瓶颈集中于大 prefill 域
（case7/8/9，约 863/863/874 μs，0.50/0.50/0.39x）：

1. **大 prefill 主耗时 = C↔V 四相位每槽交接延迟链**（case9 批深仅 ~4.5 槽，摊薄不足）。
   单侧吞吐类手段（C 侧 L0B 4 槽旋转 / K/V L1 双缓冲 / w1 dtype 瘦身 / V 宽通道 VW=2）
   经双侧 A/B 实测**被 1:1 吸收或反向**，判定 `mte1_stall` 等为四相位链伴生症状而非杠杆，
   已证据性关闭
2. **交接结构删除类不被吸收**：`flat(task, kv)` 连续令牌流已全域落地（目标 case 收益
   -6.3% ~ -33.1%，`wait_id5` 降幅 -95%+）
3. **唯一剩余架构级杠杆 = 框架虚拟通道**（`L0C→UB` / `UB→L1`），用于消除 w1/w2/w3 的
   GM 往返（物理上为 L2 往返而非 DRAM）。当前 `T.copy(L0C→UB)`、`T.copy(UB→L1)` **无设备
   实现**，该 GM 往返是 C/V 之间唯一物理通道 ⇒ 已作为框架工程建议移交
4. **decode D=256（case15/18）duty 墙**：`gemm_v0` prologue 全排空 + 每 tile
   `PipeBarrier<PIPE_M>` 的执行模型使每槽 wall = MTE2 字节时间 + gemm 标量阻塞，**与 L1
   槽数无关**，规则内不可压缩
5. **手写 mma 流水 bN 上界定理**：`bN` 上界 = B 操作数在途工作集 ≤ L0B 64KB
   （`bN=256 ∧ D=256` 结构性不可行，实测全形状族 HANG），已固化设计约束
6. **case9 M1 线**：平均 ≥0.75x ✅；case9 ≥0.4x 实测 0.39，规则内差 0.01
7. **方法学纪律**：camodel 仅作机制定性判定，禁作收益定量（多次与真机方向背离）

已证伪方向（勿重蹈）：单侧吞吐类四族（L0B 槽旋转 / K/V L1 双缓存 / w1 dtype 瘦身 /
V 宽通道）、路径 2 `fbm` prefill `bN=128→256` 结构加宽（全形状族 HANG）、架构级方案 C
（覆盖率 13~24%，上界至多 1.10x/1.16x，证据不足，不建议立项）。

**平台适配说明**：kv_db decode 域（`S ≤ 16 ∧ D == 256 ∧ S_kv > 256` = case15/case18）
在在线平台原生 tilelang 上曾因引用 opt-in `mte2_sync` kwarg 而解析期报错；本算子已
切换为 stock `gemm_v0` 默认全排空同步（3 处 kwarg + 4 处配套用户 flag 成对去除，防止
悬挂 `MTE2_MTE1` id-0 令牌竞态），其余 18 个 case 生成码逐字节不变，kv_db 域实机
`matched = 1.00000`。stock 同步的代价为 case15/18 约 421→561 μs（r12/r13 补丁收益回吐），
属已知取舍。
