# [Example] gqa：Grouped Query Attention 前向算子（形状分类 kernel 工厂 + 零 torch d2d 搬运），CANN-Bench 20/20 全过（67.84 分），decode 持平 aclnn baseline（0.97–1.07x）

## 概述

新增 `examples/gqa/gqa.py`：GQA 前向的 TileLang-Ascend 单文件实现（kernel + dispatch + 内联 golden 自检，自包含无跨模块 import）。

```python
g = h_q // (N_q // N_kv)                                 # G 个 query head 共享 1 个 KV head
score[b,h,s,j] = scaleValue * Q[b,s,h,:] · K[b,j,g,:]    # + 右下对齐 causal mask
y[b,s,h,:]     = softmax_j(score) @ V[b,:,g,:]           # 全 mask 行输出精确 0
```

- 接口 `gqa(query, key, value, scaleValue=-1.0, is_causal=False)`，BSND in/out，`scaleValue<=0` 取 `1/sqrt(D)`；另有 shape-factory 形式 `gqa(B=…, S=…, …)`。
- 4653 行内含 **7 个注意力 kernel**（per-head 阶梯 / dense / dense_expert / dense_expert_d256 / prefill_expert / g2stack / g2 兜底）+ **3 个自写搬运 kernel**（BSND↔BHSD transpose ×2、2-in-1 pad），由形状分类 dispatch 按有序编译回退链选择。其中 4 个 expert kernel 走 Expert 模式（AUTO pass 全关，MIX_AIC_1_2，20 Cube / 40 Vector 核），其余 3 个为 AUTO 模式回退候选。
- 支持 fp16/bf16、D 64–512、任意 B/N_q/N_kv/S/S_kv（非对齐形状 in-kernel padding）。

Ascend910_9362 + CANN 9.2.0 上 CANN-Bench（level4/gqa，20 case）终验：**20/20 全过**，精度 30/30、编译运行 20/20，综合 **67.84**（平均加速比 0.63x）；**decode 持平至反超 aclnn FusedInferAttentionScore**（S=1 非因果 5 例 0.97–1.07x，3 例反超；MTP 因果 0.81–0.89x），prefill 端到端 0.30–0.68x（主 kernel 0.35–1.11x）。

## 实现要点

1. **形状分类 dispatch + 编译回退链**：decode `[dense_expert(D128) → dense_expert_d256(D256) → dense_auto]`，prefill `[g2stack → prefill_expert → per-head BM 阶梯 {64,32,16} → g2]`；factory 构造期即时编译，失败自动降级并按形状缓存（编译失败只在构造期抛出，不可能掩盖运行时/精度缺陷）。per-head block_M 按 L0C 192KB 预算门控（D512@BM64 需 320KB → BM32）。
2. **decode dense M-packing + BSND 直供**：G_inner 个共享 KV head 的 Q 头稠密打包进 GEMM M 维（KV 搬运与实例数 ÷G_inner）；s-major 行序使每 (bz,s) 的 Q 装载与输出散射都是连续 2D box；K/V strided 直读、输出直写调用方 BSND 张量，decode 路径 4 次 host transpose 全部消除（曾主导端到端耗时）。threads=1 规避 AscendVidReduction 的 sub-tile scatter 减半缺陷并去掉跨核 flag 链，较 per-head threads=2 形式**提速 2.2–7.9x**。
3. **Expert 手动 flag 深流水（C/V 跨核解耦）**：C 侧 GEMM 与 V 侧 softmax / O 累积经 6 组跨核信号量（ws_s / ws_p / ws_o × ready / consumed）+ GM workspace ns 深缓冲，按 cross_interval 粒度重叠；任务静态均分 20 Cube 核。
4. **prefill g2stack（G_inner=2 M-stacking）**：单任务同时算同 (bz,kg,bx) 的 2 个 Q 头，K/V/mask GM 流量、任务数、每迭代 C↔V 握手包全部 ÷2（标量 pipe 握手为实测最大 stall 源）；l0a 64KB / l0c 128KB 零余量贴合包络。
5. **causal 块级跳过（branch-free）**：整块被掩码的 KV 块零成本跳过（不搬 K/V/mask、不做 GEMM），表达为运行时循环上界 kv_useful；C/V 两侧同公式推导 → flag 配对数一致，避免跨核死锁。
6. **零 torch d2d 数据搬运**：自写 `transpose_qkv`（whole-block / tail / q_v1 三模式，pad 行 in-kernel 零填充）、`_pad_kv` 2-in-1（K/V 一次 launch）、`transpose_out(s_real=)` in-kernel 裁剪，fp16/bf16 路径 0 个 torch device-side 拷贝/填充算子——既规避部分 CANN 9.x SoC 包缺失的 aclnn binary（Transpose / ZerosLike / FillScalar-fp16 / ViewCopy，错误码 561103），又结构性移除 d2d→launch 竞态面；mask 一律 CPU 构造 + 单次 H2D（每调用恰 1 次 host→device 传输）。
7. **特殊值数值语义**：max-after-mask（加性掩码先于 running max）+ `-1e30` 掩码常量（fp16 全量程 dot 积 ~5.5e11 下掩码列仍赢不到 max）+ 哨兵初始化/封顶 → 全 mask 行输出精确 0，NaN/Inf 传播与 fp64 oracle 逐位一致，±65504 全量程输入 softmax 仍正确。

## 性能测试

CANN-Bench 评测框架（`cann_final_eval_20260916_213834`；npu:0，Ascend910_9362，CANN 9.2.0，torch 2.7.1 + torch_npu 2.7.1.post4，warmup 3 / repeat 5）；端到端算子耗时；baseline = aclnn FusedInferAttentionScore，`vs baseline` = baseline / 本实现。decode 单 main kernel（端到端 = 主 kernel），prefill 端到端 = 主 kernel + transpose/pad 辅助 kernel。

| # | 路径 | 配置 | 端到端 (us) | 主 kernel (us) | baseline (us) | vs baseline |
|---|---|---|---|---|---|---|
| 1 | prefill | B2/S128/G4/D128 causal fp16 | 41.78 | 26.48 | 16.40 | 0.39x |
| 2 | prefill | B2/S128/G12/D128 fp16 | 52.00 | 29.06 | 22.70 | 0.44x |
| 3 | prefill | B2/S128/G16/D128 causal fp16 | 111.18 | 74.96 | 49.20 | 0.44x |
| 4 | prefill | B4/S256/G8/D128 bf16 | 248.44 | 202.10 | 103.72 | 0.42x |
| 5 | prefill | B2/S128/G4/D256 causal fp16 | 74.68 | 57.02 | 32.80 | 0.44x |
| 6 | prefill | B2/S256/G16/D256 causal bf16 | 592.32 | 504.64 | 175.65 | 0.30x |
| 7 | prefill | B2/Sq128/Skv256/G4/D128 bf16（交叉）| 51.80 | 35.62 | 30.30 | 0.58x |
| 8 | prefill | B4/S1024/G4/D128 causal fp16 | 855.58 | 741.54 | 389.69 | 0.46x |
| 9 | prefill | B4/S512/G8/D128 bf16 | 648.84 | 548.38 | 257.79 | 0.40x |
| 10 | prefill | B4/S512/G16/D128 causal bf16 | 1265.74 | 980.66 | 685.34 | 0.54x |
| 11 | decode | B16/S1/Skv2048/G4/D128 fp16 | 145.98 | 145.98 | 146.06 | **1.00x** |
| 12 | decode | B16/S1/Skv2048/G12/D128 bf16 | 150.60 | 150.60 | 145.54 | 0.97x |
| 13 | decode | B60/S1/Skv2048/G4/D128 fp16 | 475.26 | 475.26 | 488.29 | **1.03x** |
| 14 | decode | B128/S1/Skv512/G16/D128 fp16 | 295.46 | 295.46 | 290.79 | 0.98x |
| 15 | decode | B60/S1/Skv1024/G4/D256 fp16 | 406.48 | 406.48 | 433.87 | **1.07x** |
| 16 | decode | B60/S2/Skv2048/G4/D128 causal fp16（MTP）| 546.48 | 546.48 | 443.29 | 0.81x |
| 17 | decode | B128/S4/Skv512/G16/D128 causal fp16（MTP）| 464.14 | 464.14 | 413.25 | 0.89x |
| 18 | prefill | B2/S128/MQA(G32)/D128 fp16 | 37.84 | 24.10 | 15.20 | 0.40x |
| 19 | prefill | B2/S128/MHA(G1)/D128 causal bf16 | 48.50 | 29.66 | 32.80 | 0.68x |
| 20 | prefill | B2/S128/G4/D128 causal bf16（显式 scale）| 41.38 | 26.78 | 16.40 | 0.40x |

- **decode**：dense M-packing 消除 transpose 与实例数冗余，单 main kernel 占满端到端耗时 → 全部落在 0.81–1.07x，差距集中在 MTP 因果。
- **prefill**：辅助 transpose/pad kernel 占端到端 13–44%（小形状最重）；D256 不在 expert 系列覆盖内（走 per-head BM64），主 kernel 0.35x 为最弱项；MHA(G=1) 主 kernel 1.11x 反超，端到端被辅助搬运拉回 0.68x。

## 精度验证

本 PR 只含 `gqa.py`，文件自带 `__main__` 自检（内联 PyTorch golden `_ref_gqa`，CPU fp32 比对），覆盖两条 dispatch 路径，失败 `sys.exit(1)`（满足 `examples/bench_test.sh` 的 exit code + marker 门禁）：

```bash
python examples/gqa/gqa.py   # prefill_causal max_diff=3.663e-04 / decode_noncausal 2.281e-05，viol=0 -> Test Passed!
```

- 门限 fp16 atol=1e-3 / rtol=1e-2、零容差违例；另校验 shape / dtype / contiguous / 无 NaN-Inf。CANN-Bench 20 case 精度全过（30/30，max_diff ≤ 3.9e-3）。
- 开发期另有一套本地分层测试（未随本 PR 提交）：9 L0 代表形状 + 9 L1 非对齐尾块 / MQA / MHA + 6 Boundary 特殊值 = 24/24 PASS；门限 fp16 atol=1e-3 / bf16 atol=2e-2、rtol=1e-2，特殊值用 fp64 CPU oracle + NaN/Inf 位置与符号精确比对。
- 稳健性：本地 68 用例变体评测 68/68、0 crash；特殊值矩阵 1620 cell 全过（{NaN, Inf, 全量程} × {fp16, bf16} × {G1, G4, G16} × {causal, 非 causal} × {对齐, 非对齐} × {prefill, decode} × 10 seeds）；回退退化形态 × 非对齐 × 特殊值 × 100 次重复 0 crash、逐次 bitwise-stable。

## 已知差距

- prefill 端到端落后 baseline（0.30–0.68x），主因辅助搬运占 13–44%；后续方向：BSND 直供的 prefill kernel（消除辅助搬运）+ D256 expert 变体。
- MTP 因果 decode（S=2/4）0.81–0.89x，弱于 S=1 非因果。
- 个别 tilelang 构建上回退链会打印 `[gqa][fallback]` 并降级到次优 kernel（功能/精度不受影响，性能次优）。

## 文件说明

| 文件名 | 说明 |
|--------|------|
| examples/gqa/gqa.py | 7 注意力 kernel + 3 搬运 kernel + dispatch 工厂 + BSND adapter + 内联 golden 自检（自包含单文件，4653 行）|
