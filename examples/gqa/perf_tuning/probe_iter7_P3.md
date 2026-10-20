# P3 Gate-0 可行性探针报告 — expert 模式 FIX→UB 直达（Stage 3 · iteration 7 · 只探不采）

- 日期：2026-10-19（本会话 14:38–14:52 实测）
- 对象问题（iter6 §6.1 根因 (a) / §7-P3 卡）：expert 模式下 L0C 结果能否经 FIX 直接落 UB
  （或等效片上路径），替代 `L0C →FIX→ GM(ws_s/ws_o) →L2→ MTE2→ UB` 往返。
- 硬边界遵守：探针产物全部在 `perf_tuning/`（本报告 + `probe_iter7_P3_fix2ub.py` +
  两份生成源 `probe_iter7_P3_e{1,2}_*_source.cu`）；**未接入 dispatch、未修改 gqa.py
  （P1 集成与本探针无关且在其后）、无性能采纳判定**。

## Verdict：**infeasible**（本平台 + tilelang-ascendc 后端 + expert 配置下不可达；三层证据闭合）

## 1. 证据链（全部实测/源码核实，无凭记忆项）

### 1.1 硬件/ISA 暴露层 —— 本平台（A3, arch 2201）无 l0c2ub 指令暴露

- 平台判定：`tilelang/utils/target.py:determine_platform`——soc 名含 `910_93`（本机
  ini = `Ascend910_9362`，npu-smi "Ascend910"）→ **A3**；CANN 9.2 的 asc C-API 按
  `__NPU_ARCH__` 分流：`2201`（A2/A3 实现集）与 `3510`（A5）。
- **`asc_copy_l0c2ub` 的实现只存在于 `npu_arch_3510`**
  （`.../asc/impl/c_api/instr_impl/npu_arch_3510/cube_datamove_impl/asc_copy_l0c2ub_impl.h`）；
  `npu_arch_2201/cube_datamove_impl/` 目录清单（ls 实测）只有 `l0c2gm / l0c2l1 / l12* /
  gm2l0* / gm2l1`——**fixpipe 的目的地在 2201 上只暴露 GM 与 L1**。
- tensor API 层 `asc::te::copy_l0c_to_ub`（`arch/cube/l0c_to_ub/{copy,routing,instruction}.h`）
  最终调用同一 `asc_copy_l0c2ub`——routing 对 nd/dn 目标布局有特化（非 ignore），但指令实现
  依赖 3510；在 2201 上不可编译。
- ini 的 `l0_c_to_ub_rate=256`（B/cyc）与 `Intrinsic_fix_pipe_l0c2out` 说明**微架构能力在
  规格表中存在**，但 CANN 9.2 未对本 arch 暴露对应指令——"硬件有能力"≠"可编程"。
  iter6 P3 卡把它当作路径存在性证据是不充分的（该卡自身已标注"可行性未验证，探针先行"）。

### 1.2 tilelang 后端层 —— ascendc 后端无直达路径；`T.copy(l0c, ub)` 是 ws 往返的语法糖

- 代码生成层：`src/op/ascend.cc:298` 存在 `copy_l0c_to_ub<...>` 发射分支（virtual_channel），
  但 **`tl::ascend::copy_l0c_to_ub` 模板在全部头文件中无定义**（grep 实测：repo
  `src/tl_templates/ascend/common.h`、installed tilelang、catlass、CANN——字符串只存在于
  编译产物 .so 的发射器中）。该分支输出的是**中间标记**，被
  `AscendWorkspaceReduction`（phase.py:83，无条件运行）重写。
- 重写语义（`src/transform/ascend_workspace_reduction.cc`）：非 PTO 时
  `pto_use_pipe = is_pto && cfg` 恒 false → L0cToUb 虚拟拷贝一律**物化为 GM workspace**
  （`decl_buffer(..., "global")`，按 cid×per_block_ele 分槽 + vid reduction 半行拆分）。
- **pipe 直达（唯一真实片上通道表达）只在 PTO 后端**：`copy_l0c_to_pipe`/`copy_pipe_to_ub`
  的 codegen 仅在 `codegen_ascend_pto.cc:1851-1858`，运行时模板仅在
  `src/tl_templates/pto/common.h:1525-1580`（`pto::TPUSH<Pipe, TileAcc...>` /
  `pto::TPOP<Pipe, TileUbDataND...>` + 每 pipe 2 个 flag id、C2V 方向、FIFO 双缓冲平铺
  1D buffer）。ascendc codegen（codegen_ascend.cc）**无任何 pipe op 处理**（grep = 0 命中）。
- 仓库先例核查：PTO 版 GQA（`examples/pipeline/sparse_flash_attn_gqa_pipeline_pto.py`）
  仍用**手工 GM workspace**（`T.copy(acc_s_l0c, workspace_2[cid,...])`），未用 pipe；
  developer_mode 示例的 `T.copy(acc_s_l0c, acc_s_ub_)`（flash_attn_bshd_developer.py:82 等）
  均为 AUTO 配置（见 1.3-E1）。**全仓库无 expert 模式 l0c→ub 直达先例**（iter6 判断
  "expert 路径无先例"复证成立）。

### 1.3 实测三实验（probe_iter7_P3_fix2ub.py，本会话真机）

| 实验 | 内容 | 结果 |
|---|---|---|
| **E1** | 产线 per-head AUTO kernel（gqa.py T.copy(acc_s_l0c, acc_s_ub) 先例）源 dump | 生成码 = `copy_l0c_to_gm(workspace_1[cid*8192])` + `CrossCoreSetFlag<2,PIPE_FIX>` + `copy_gm_to_ub(acc_s_ub, workspace_1[cid*8192+vid*4096], <128,32>)`——**AUTO 先例同样是 GM ws 往返**（pass 管理 ws/flag/vid 拆分），不是 FIX→UB 直达。iter6 P3 卡引用的"硬件路径存在性证据"就此修正。 |
| **E2b**（控制臂） | 最小 expert CV 分离 kernel（P28_EXPERT_PASS_CONFIGS + 手工 flag 编排，flat_kv2 choreography 微缩）：l0c→GM ws→UB→MTE3 出 | **编译 ✓ 运行 ✓ 数值 PASS**（max_diff 1.43e-06 vs torch fp32 参照）——探针脚手架健全，失败可归因 |
| **E2**（直达臂） | 同 kernel，C scope 直接 `T.copy(l0c, ub_dst)`（ub_dst 为 V 读的 UB） | **编译通过但语义非直达**：生成码 = 隐藏 `workspace_1` GM 参数 + `copy_l0c_to_gm` + 调用点插入 `copy_gm_to_ub(ub_dst, workspace_1[vid*2048], <64,32>)`——与 shipped expert 手工 ws 路径**同构**（零收益）；且**数值 FAIL（max_diff 9.82）** |

**E2 失败机理（源级定位）**：插入本身尊重了手工 cross-flag 次序（`copy_gm_to_ub` 落在
V 的 `CrossCoreWaitFlag(SEM_DATA)` 之后，源行 77→78），但 pass 自动施加 **vid-reduction
半行拆分**（`workspace_1[vid*2048]`、`<64,32>` = 每 vid 只搬 32 行）——与 expert kernel
的手工 vid 约定（gqa 家族 `row0 = h*block_M + vid*half` 的自管理拆分）冲突：ub_dst 半缺
行 + 双 vid 竞写输出。AUTO 模式下该约定由 CV-combine 全链一致持有所以产线正确；expert
配置（AUTO_CV_SYNC/CV_COMBINE 关闭）下**没有任何机制让插入的搬运与用户手工 vid/flag 协议
对齐**——这是框架语义冲突，不是可绕过的实现细节。

## 2. 阻塞点归类（任务书要求项）

1. **硬件路径约束（本平台）**：A3/arch2201 的 CANN 9.2 只暴露 fixpipe→GM/L1；l0c2ub 指令
   实现仅 A5/arch3510。——**根本阻塞，不可在算子层绕过**。
2. **API/后端缺口（ascendc）**：无 `tl::ascend::copy_l0c_to_ub` 模板、无 pipe op codegen；
   `T.copy(l0c,ub)` 只能落到 ws 往返。
3. **flag/vid 协议冲突（expert）**：即便接受 ws 往返形态，pass 的 vid-reduction 与 expert
   手工协议冲突（E2 数值 FAIL 实证）——expert 模式下该语法糖**不可安全使用**。

## 3. 替代路径评估（任务书要求项）

- **A5 目标**：`asc_copy_l0c2ub`（带 `unit_flag_mode`/`dual_dst_ctl`/量化前后处理参数）+
  PTO pipe（TPUSH/TPOP）都指向 A5 上 FIX→UB 类直达是**一等公民**。若算子未来跑 A5，
  P3 值得重启（PTO 后端 + `tl.ascend_pto_use_pipe_in_cv_copy`，注意 A5 上
  `copy_pipe_to_ub_V` 走 PIPE_V）。本平台（A3）不适用。
- **框架扩展**（tilelang-ascend-tile-api skill 范畴，非 Stage 3 算子调优范畴）：在 2201 上
  即使补 `tl::ascend::copy_l0c_to_ub` 模板也无指令可用（1.1）——**扩展无处落**；可落的方向
  是 PTO A2 pipe 的物理介质查证（workspace_reduction 注释称 A2 PTO pipe 为"2× 平铺 1D
  FIFO 双缓冲 buffer"，`copy_pipe_to_ub` 归属 PIPE_MTE2——疑似仍是存储器 FIFO 而非寄存器级
  通道；pto-isa 子模块本机未检出，无法本地核实——**needs-deeper-study 的唯一残项**，但它
  属 PTO 后端迁移议题，不在 expert-ascendc 主线）。
- **若直达存在时的时序设计（供 iter8+ 参考，纯设计推演）**：完成通知 = FIX 队列排空后
  `set_cross_flag("FIX", SEM)`（与现协议同形）；V 侧消费后 `set_cross_flag("MTE2", SEM_V2C)`
  释放——**cross-sem 数量不变（每交换 2 个），无新暴露段**；收益纯为删两跳 GM/L2 延迟与
  256MB(R4) L2 流量。真实约束在 UB 预算：io_buf 需双缓冲接 FIX 流（+32KB/vid →
  ~186/192KB 贴顶，iter6 风险注记维持）。

## 4. iter8 设计要点草案（基于本 verdict）

1. **根因 (a) ws 往返在 A3+ascendc+expert 组合下不可消除**——iter8 不要再立项"消除往返"
   类方案（P3 及其变体）；对根因 (a) 的剩余可动作项只有**降低往返的暴露**（而 iter4 已
   证明 pacing 门控是承重结构、环深/步序不可动）。
2. 主收益线维持 iter6 排序的 **P2（main 直写 BSND 消 transpose_out）**：它攻击的是根因 (c)
   的 Output 段往返（与 P1 收益部分重叠——P1 已收 tout 的网格税，P2 再删 tout 整段 +
   Output 的 UB→GM(BHSD)→读→UB→GM(BSND) 双跳），且载体（strided MTE3 盒）已被 P1 后的
   tout 实测速率背书。注意 P1 落地后 P2 的边际收益按新基线重估（tout 已从 12-24µs 降到
   5-18µs，P2 的可删项相应变小）。
3. aux 侧剩余项：transpose_qkv（P4，strided GM→Nz L1 探针先行）与 pad kernel 税（未测）。
4. P5 类 vec busy 微优化本轮实测 +0.75%/+1.15%（<3% 阈值，未采纳）——kv2 域 vec 侧
   微操的期望值已被本轮标定，iter8 不建议再投同类微操。

## 5. 产物清单

- `perf_tuning/probe_iter7_P3_fix2ub.py`（E1/E2/E2b 三实验，可复跑）
- `perf_tuning/probe_iter7_P3_e1_perhead_auto_source.cu`（AUTO 先例生成源）
- `perf_tuning/probe_iter7_P3_e2_expert_direct_source.cu`（expert 直达臂生成源——ws 物化证据）
- E2b 控制臂源与数值日志（会话内打印，max_diff 1.43e-06 PASS）
- 引用源码坐标：`src/op/ascend.cc:293-305`、`src/transform/ascend_workspace_reduction.cc`
  （1455-1478 门控、403-443 ws 物化、1041-1056 A5 分支）、`src/target/codegen_ascend_pto.cc:1851-1858`、
  `src/tl_templates/pto/common.h:1521-1580`、`tilelang/engine/phase.py:83`、
  CANN `asc/impl/c_api/instr_impl/npu_arch_{2201,3510}/cube_datamove_impl/`、
  `Ascend910_9362.ini`（l0_c_to_ub_rate=256、Intrinsic_fix_pipe_l0c2out、support_fixpipe=1）
- 查阅：tilelang-api-best-practices（references/api-kernel-memory.md:105 明确记载层级路径
  "L0C → workspace(GM) → UB"；api-schedule-sync.md CV 交换范式）、CANN RAG 一次
  （fixpipe 文档摘要：CO1→Global Memory 搬迁；asc_copy_l12fb A2/A3 支持表——RAG 未含
  l0c2ub 的 A2 暴露，与本地头文件核查一致）
