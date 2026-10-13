# mha 算子设计文档（rev4）

> Stage 1 产出（tilelang-op-design + tilelang-op-test-design 场景 A），revision_index=3。
> 编程模式由用户（需求方）在 rev3 立项时明确决策为 **Expert**（2026-09-19），rev4
> 延续该决策，**禁止回退 Developer/混合模式**。
> 上一版设计：`history_version/design_rev3.md`（Expert 双形态，md5 e97dacb6，已验证
> 并收口于 cann-bench **71.70/100、平均 0.9054x**，20/20 pass，M2 达成）。
> 实现回退锚：`history_version/mha_impl_r3_expert_final_7170.py`（md5 bbfcbcaf，
> 1113 行，与当前 `mha.py` 逐字节一致——rev4 开工基线，md5 已核验）。
> 位级语义基准（历史）：`history_version/mha_impl_developer_final_6505.py`
> （Developer 终态，cann-bench 65.05）。

## 0. 修订记录（rev3 → rev4）

### 0.1 变更动因（用户立项决策 2026-09-22，**非 DESIGN_ERROR**）

rev3 Expert 轨经四个调优会话（final_report.md 会话 3-6）收口于 cann-bench
**71.70 / 平均 0.9054x（20/20 pass，M2 平均线 0.9x 首次达成）**。唯一未达项 =
**case9 0.324x < M1 case 线 0.4x**（(4,1024,1024,16,256) fp16 causal，Form B
prefill D=256 路径）。规则内调优手段已**三证据链穷尽**（final_report §S4-3-2）：

1. gemm_v0 内部每-step 串行链 = case9 剩余 wall 的主体：MTE1 L1→L0 40-80KB/step +
   M_MTE1 双槽 ping-pong + **每-step `PipeBarrier<PIPE_M>` 排空前一 mma**；
   scalar_mte1_stall ≈78% wall × 4 step/iter × ~460 iter/核（会话 4 msprof op 归因；
   C 核 FIXP 72.05%、V 核 FLOWCTRL 79.37%，会话 4 同 IR trace 读数）。
2. 三条规则内路线全部实测证伪：①step 数削减（kL0=256 单步）→ **+3.7% 劣化**
   （kNumSteps=1 关闭 ping-pong，重叠损失 > step 开销节省）；②MTE2 载入隐藏
   （k/v L1 DB 预取）→ **零响应**（op 归因全 pipe 读数逐位不变——MTE2 45-50%
   busy 本已被吸收在 stall 阴影下）；③会话 3 三证伪（V 指令/C 装载/SEM 频率）同向。
3. 用户审阅 case9 MindStudio 指令级流水（`perf_tuning/mindstudio_export/
   case9_simulator/`，README 含分析导引；终态 bbfcbcaf 上 2026-09-22 新鲜采集，
   该路径 IR 在 #R5-1/#R6-1 两轮逐字节未变 ⇒ 与会话 4 采集机制等价）后，在
   Primary 上下文**决策立项 rev4**，范围两项：
   - **S1 = rev3 §10.3-⑤ Form B→A mma 化**：Form B 路径由 gemm_v0 改为手动
     T.mma 流水（Form A 已验证模式的扩展：手动 K-split/N-split + L0A/L0B/L0C
     双槽 + 细粒度 flag），消除 gemm_v0 每-step PipeBarrier<PIPE_M> 排空。工程
     要点与位级预案已归档在 final_report §S4-5（l0b 共池 64KB 顶格 ⚠ 分池复核、
     D=256 GEMM1 双 k-chunk init=True/False 链、GEMM2 双 D-half、位级锚失配预案）。
     目标：case9 → case7 同族结构（~0.46x 级），decode 族不回退。
   - **S2 = kv-outer 循环互换**（final_report §S5-4-6）：case7/8（0.46x）同
     (bz,by) 的 s_blocks=16 个 task 共享整条 K/V 序列，消除 ~16× 装载冗余。
     约束：UB 放不下 s_blocks 份 acc_o → 需 online softmax 状态重构设计；若论证后
     工程不可行或收益为负，允许在 rev4 内降级为"本轮不实施"，但必须显式覆盖论证。
     **rev4 结论 = 降级不实施（§0.5 完整定量论证，非静默略过）**。

破局唯一路径判定（§S4-3-2 原文）："手动 mma 流水（Form A 式 L0C 双槽 + 细粒度
flag，无每-step PIPE_M 排空）= §10.3-⑤ Form B→A mma 化 = 设计级变更"。

### 0.2 rev3 → rev4 变更清单

| # | 章节 | rev3 内容（终态 bbfcbcaf，71.70） | rev4 修订 | 依据/证据出处 |
|---|------|----------------------------------|-----------|--------------|
| N1 | §5.2/§5.6 | C-scope **双形态** host 二分派：Form A（S>16∧D≤128，T.mma 全流水）/ Form B（其余，gemm_v0） | **三分派**：Form A（不变）/ **Form B-mma（S>16 ∧ D=256，新增：手动 K-split/N-split T.mma 流水）** / Form B-gemm_v0（S≤16 decode 全 D，兜底，源码逐字不变） | 用户立项 S1；三证据链 §S4-3-2；§S4-5 rev4 建议；decode 兜底 = 任务硬点 6 授权 + §4.5-e decode mma 化 L0B 预算不可行推导 + §S4-3-3 decode wall=每槽 MTE2 带宽（mma 化无杠杆） |
| N2 | §3/§8.3 | Form B GEMM = `T.gemm_v0(q_l1,k_l1,acc_s_l0c,transpose_B=True,init=True,kL0Size=128)`（D=256 内部 2 k-step，每-step PipeBarrier<PIPE_M>）+ `T.gemm_v0(p_l1,v_l1,acc_o_l0c,init=True)` | **Form B-mma GEMM1** = 手动 2×k-chunk 链：`T.mma(l0a_s[c], l0b_pool[c], l0c_s[side], init=(c==0))`（c∈{0,1}，KC=128，K=D=256）；**GEMM2** = 2×n-chunk（NC=D-half=128）：主案 a `T.mma(l0a_o[side], l0b_pool[c], l0c_o[side,:,c*NC:(c+1)*NC], init=True)`（C region 切片）；备选案 b = l0c_o 窄池 + w3 列段 FIX×2 | T.mma=npu_gemm 签名 customize.py:115（init 链/k_actual/n_actual/unit_flag）；**仓内最强先例 sparse_flash_mla_kernel.py:371-423**（D2=256 K-split 2 步 init=True→False 链 + L1 列段 transpose 载入 :376-382 + v_l0b [2,128,128] 64KB 顶格 :218 + L0AB_EV0/EV1 双 chunk token :174-175/383-423）；N-split 列段 FIX 写出先例 :673-681；C region 切片前端支持 customize.py:151-209（retrieve_shape/retrieve_ptr 处理 BufferRegion） |
| N3 | §4.3/§4.5 | P3（prefill D=256）L0 = 用户仅声明 acc_s_l0c/acc_o_l0c 单槽，gemm_v0 内部按 kL0Size 申请 L0A/L0B（峰值 64KB⚠，session-2 #11 实测先例） | **P3-mma 显式 L0 池**：L0A 32KB（l0a_s [2kc,32,128] 任务级常驻 + l0a_o [2side,32,128]）；**L0B 64KB⚠ 共池唯一可行解**（l0b_pool [2,128,128]：GEMM1 K^T kc-chunk 与 GEMM2 V n-chunk **同形**分时——分池 128KB > 64KB 硬限）；L0C 主案 96KB（l0c_s [2,32,128] 32 + l0c_o [2,32,256] 64）/ 备选 64KB。**逐形状 L0A/L0B/L0C 预算表 + decode mma 化不可行推导表**（§4.5，硬点 1） | 共池顶格先例 = 本算子 Form A P2 l0b [2,128,128] 64KB（位级验证在案）+ sparse_flash_mla v_l0b 同尺寸；共池分时先例 = Form A P2 GEMM1/GEMM2 L0 池互换（mha.py:576-578）+ mha_sink；实编读数制度继承 §4.5 |
| N4 | §7.2 | Form B C-scope 零用户 flag（gemm_v0 自平衡段界，#R4-1）；Form A SIG_L0AB 0,1 + SIG_L0C 0,1 | Form B-mma 新增细粒度 token：**MTE1↔M pair：SIG_L0KC=0,1（l0b_pool chunk0/1）+ SIG_L0AO=2,3（l0a_o side0/1）+ SIG_L0AS=4,5（l0a_s chunk0/1）**；M↔FIX：SIG_L0C=0,1（l0c_s/l0c_o 分时同流，Form A P1 先例）；MTE2↔MTE1 与 SEM 0-5 全继承。**完整 init/set/wait/destroy 收支表**（§7.2-2a，硬点 4：禁止悬空令牌） | sparse_flash_mla L0AB_EV0/EV1 双 chunk token 结构 :383-423（init 预置 :243-244）；event id 每 pair 硬件上限 8（EVENT_ID 0-7；sparse_flash_mla P_EV=7 佐证 :172；Form B-mma MTE1↔M pair 最大用 6）；gemm_v0 内部 L0AB_EVENT=0 仅在兜底域 prim_func（common.h:1183/1306-1311），Form B-mma 域无 gemm_v0 调用 → 零占用冲突 |
| N5 | §9.2/§9.3 | 位级诊断锚：rev3 输出 vs Developer 终态（65.05）L0 十用例位级一致（10/10 目标） | **锚体系重构（硬点 2）**：阻塞口径**逐字不变**（混合容差 + MARE_abs<10×Threshold，rev2-C1 用户裁决沿用）；**锚-A**（vs Developer 终态）守卫域收窄为 **L0 #1-9（Form A 域，IR 不变 → 9/9 BITEXACT 硬守卫）**，#10 移出（Form B-mma 域，失配预期允许）；**锚-B（新增）** = Form B-mma 域输出 vs rev3 终态（bbfcbcaf）同 shape 位级比对——诊断主锚（预期大概率 BITEXACT：K 升序同序累加 + fp32 L0C + w1 fp32 主线 + emit_v 逐字；**不承诺**：tl.ascend_mma 与 tl.ascend_gemm_v0 为不同 C++ 模板），失配 ≠ FAIL → 限定累加序/指令微序根因分析，阻塞口径终判 | 任务硬点 2（"mma 累加序 ≠ gemm_v0 内部累加序 → fp32-ws 位级锚必然失配"预案）；§S4-3-4-①（单 mma k=256 ≡ 链式 2×mma k=128，C0 顺序累加同序，12/12 BITEXACT——同序链式累加位级稳定性知识）；§S4-5 位级锚失配预案原文 |
| N6 | §12 | 性能目标 = Expert 化逼近 1.0x（M1 avg≥0.75 + case7/8/9≥0.4；M2 avg≥0.9）；回退条款 = 劣于 Developer 65.05 回滚 | **rev4 重写**：锚点 71.70/0.9054x（cann_final_eval_20260922_100108）；主攻 **case9 ≥0.4x**（M1 case 线，预期 0.46x 级 = case7 同族结构）；case4 ≥0.828 不降；decode 不回退硬约束（case10 ≥0.90/case13 ≥0.69/case14 ≥0.99/case16 ≥0.52/case20 ≥0.82，当前 0.983/0.886/1.394/0.785/2.079）；Form A 域 ±3% 噪声带（IR 逐字节不变守卫）；**回退条款：rev4 终态 cann-bench <71.70 或平均 <0.9054x → 回滚 mha_impl_r3_expert_final_7170.py**（硬点 7） | §S6-7 权威复评表；任务硬点 6/7 |
| N7 | §0.5（新增） | —（rev3 §10.3 候选清单无 kv-outer 项；§S5-4-6 首次判为 DESIGN 级） | **S2 kv-outer 循环互换 = 本轮降级不实施**：完整定量论证（UB 全量状态 256KB 不可行 / GM spill 重构净流量 +78MB 持平略负 / G=2 分组对 case7 C FIXP 95.68% 吞吐墙无杠杆且爆炸半径 = Form A generic 全路径）+ 复活条件入库 | §S5-4-6/§S6-3-4/§S3-3（coupled equilibrium 三证伪）；§0.5 本文 |
| N8 | §10.1 | R10-R16（Expert 首建风险） | R10-R16 状态更新（多数已实证关闭）+ **新增 R17（Form B-mma 探针门族）/R18（位级锚预期管理）/R19（三分派覆盖）/R20（K/N-split token 收支）** | §10.1 本文 |

### 0.3 继承清单（不变项，逐字/等值保留）

| 继承项 | 出处 | rev4 落点 |
|--------|------|-----------|
| 算子语义/公式/支持域（BSHD、fp16/bf16、右下角对齐 causal、全掩码行→0、B≤128/S≤2048/S_kv≤4096/N≤64/D∈{64,128,256}） | rev3 §1/§4.1/§4.2（承 rev2） | §1/§4 逐字 |
| host 契约 `mha(query, key, value, scaleValue=-1.0, is_causal=False)` + 全部断言 + out_idx=[3]/workspace_idx=[4,5,6] 语义 | rev3 §3.3/§4.8 | §3.3/§4.8（host 唯一变更 = 分派谓词新增一个纯 Python bool `form_b_mma`，§4.8 审计不变） |
| golden 函数（需求方提供，原样） | rev3 §9.1 | §9.1 逐字 |
| **精度阻塞口径（用户裁决 2026-09-18，rev2-C1，不可更改）**：混合容差 + MARE_abs < 10×Threshold；MERE/MARE_rel 参考值不阻塞 | rev3 §9.3.1/§9.3.2 | §9.3 逐字 |
| **精度红线四项**：①score 双舍入栅格（w1 fp32 主线 + V 侧 r16 链）②单块预归一化特化 ③sumexp+1e-30 保护 ④掩码链语义 | rev3 §9.3.3 | §9.3.3（**四项载体全部在 V-scope/host——rev4 S1 变更域仅 C-scope → 红线结构性不变**，逐项论证） |
| L0 十用例表 + 哨兵四条 + kernel 路径 120s 门 + golden 降本规定 | rev3 §9.2/§9.4 | §9.2/§9.4（十用例逐字继承；路径映射按三分派更新） |
| Expert 形态全部结论：(cid,vid) 双绑定 + vid 手动二分 + pass_configs 四项全 False + T.Scope("C"/"V") + 显式 3-workspace + 虚拟通道禁令 + R8/R9 复检不触发 + T.Pipelined 全文禁用 + 无 threads 参数 | rev3 §2/§7/§8 | §2/§7/§8 等值保留 |
| NUM_STAGES=8 批量流水 + CROSS_INTERVAL=1 + single_kv_block 任务批处理（#R3-1/#R5-1）+ deferred rescale 批量 softmax + Fixed Core + causal KV 裁剪 C/V 同式 | rev3 §5.2/§6/§7 + 终态实现 | 全部继承（Form B-mma 批处理结构 = #R3-1/#R5-1 协议镜像 + mma 本体；#R5-4 NUM_STAGES 12/16 暂缓判定维持——rev4 控制变量，S1 归因纯度优先） |
| emit_v 共享 V-scope emitter（@T.macro，两分支：single_kv_block 任务批 / generic per-task） | 终态 mha.py:203-541 | **逐字零改动**（rev4 S1 不触碰 V-scope；其 `if not form_a:` phase-2 gate 为 parse-time Python bool ⇒ Form B-mma prim_func 内 form_a=False → phase-2 自动保留 = w3+phase-2 协议，与兜底域同） |
| bM/bN/gemm1_kL0 三档体系（tile 尺寸硬限解） | rev3 §5.2（承 Developer 终态） | 继承；gemm1_kL0 参数域收窄至兜底域（decode）；Form B-mma 以手动 KC=NC=128 chunk 替换（§0.4-⑨） |
| proto.yaml（接口契约不变） | rev3 §11.5 | §11.5 复核**无变化**（现文件 md5 3079c184 = history_version/proto_rev3.yaml，无需重写） |
| test_mha.py 测试资产（844 行，md5 0dd9e93c：golden + L0 10/L1 19/L2 12/Boundary 6/哨兵 4 + 双口径 harness + 阻塞判定 + 哨兵路径门） | rev3 §9.5 | §9.5 **零改动复用**（host 签名不变即成立） |

### 0.4 成果继承清单（12 项算法优化 + 4 项调优成果，逐项标注 保留/替换/扩展——硬点 3）

**（a）12 项算法优化（rev3 §3.2.1 体系）**

| # | 优化项 | rev4 状态 | mma 化后存续形态 |
|---|--------|----------|------------------|
| ① | causal KV 运行期裁剪（eff_kv_iters，C/V 同式复制） | **保留** | Form B-mma C-scope 裁剪式逐字继承（§7.2-5 配对平衡硬约束不变）；V 侧零改动 |
| ② | broadcast 2D 行循环替代（专用 scratch buf_2d/rt_2d/r_2d） | **保留** | V-scope 载体，零改动 |
| ③ | 掩码链 2D 化 + row_thresh hoist + bit-packed uint8 mask | **保留** | V-scope 载体，零改动（R15 已实证关闭） |
| ④ | 全可见迭代掩码运行期跳过 | **保留** | V-scope 载体，零改动 |
| ⑤ | 单块预归一化特化（S_kv≤bN 编译期谓词） | **保留** | V-scope phase-1 特化零改动；Form B-mma 单块分支保留 w3+phase-2 协议（#R6-1 O 直写 v1 不扩展至 mma 域——#R6-1 的等价性依据（单块预归一化 O=P̂@V 一步终值）在 Form B-mma 单块域代数上同样成立，但扩展需 emit_v gate 谓词泛化 + D256 gate 探针，列 Stage 3 候选 §10.3-⑨，v1 保 emit_v 逐字零改动） |
| ⑥ | sumexp+1e-30 全掩码行保护 | **保留** | V-scope 两处落点零改动 |
| ⑦ | score 双舍入栅格对齐（r16(qk)→×s fp32→r16(scores) 4×cast 链） | **保留** | w1 fp32 主线不变（l0c_s fp32 → FIX → w1 fp32，无 cast）；V 侧 rr_ub round-trip 链零改动——**GEMM1 执行体变更（gemm_v0→手动 mma）不触碰栅格载体**：双舍入的第一舍入点 r16(qk) 在 V 侧、输入为 w1 fp32 主线，与 C 侧 GEMM 指令形态解耦（§9.3.3-1 论证） |
| ⑧ | decode 窄块 bM=16（S≤16） | **保留** | Form B-gemm_v0 兜底域逐字不变 |
| ⑨ | bN 三档自适应（decode 512/256/128；prefill 128）+ gemm1_kL0 三档（32/64/128） | **bN 三档保留；gemm1_kL0 参数域收窄；P3-mma 域 kL0 机制被替换** | decode（兜底域）：kL0 32/64/128 逐字保留（gemm_v0 内部 L0B 分片依据）；prefill D=256（Form B-mma 域）：gemm_v0 kL0Size=128 内部 2 k-step **替换为**手动 KC=128 双 chunk mma 链（数学同一：K 升序分片 fp32 累加，§9.3.3-6；执行结构不同：chunk 级 token 流水替代每-step PIPE_M 排空——这正是 S1 收益机制） |
| ⑩ | Fixed Core（core_num 设备查询 + 均衡任务循环 + cid 索引 ws） | **保留** | 任务网格/分解/均衡分配逐字不变 |
| ⑪ | valid_m/valid_n 运行期尾块三重保障（need_clear 清零 + 掩码钳位 + rows_vid 裁剪） | **保留** | Form B-mma C 侧载入运行期 extent + need_clear 清零 + mma 全 N 执行 + 掩码钳位——论证逐字继承 Form A（§5.4-2）；K/N-split 维（D=256 静态）恒整除无尾块形态（§5.4-1） |
| ⑫ | BSHD 行跨步 DMA（原生进出，零 host 重排） | **保留 + 扩展探针** | GM 侧 4D 行跨步切片不变；**新增 L1 列段切片**（q_l1/k_l1/v_l1 的 D 维 128 列段 → L0 载入；列段 = buffer 最内维连续段，非禁令形态 §3.5.1）× NZ annotation 组合 = R17-P1 探针门（先例 sparse_flash_mla:372-382 L1 列段 + transpose 实测，且该先例全文无 annotate_layout → 备选去 annotation 路径有先例背书） |

**（b）4 项调优成果（会话 3-6 采纳项，final_report §S3-2/§S4-2/§S5-2/§S6-2）**

| # | 成果 | rev4 状态 | 说明 |
|---|------|----------|------|
| #R3-1 | Form B single_kv_block 任务批处理（case14 -28.9%，会话 3） | **保留 + 扩展** | 兜底域逐字保留（decode 单块 case14 1.394x/case17 1.143x 载体）；**扩展**：Form B-mma 单块分支 = 同协议镜像（num_tb/batch_tasks C/V 同式 + GEMM1/GEMM2 批 + SEM 位置不变）+ mma 本体（支持域 prefill D=256 ∧ S_kv≤128 可达，L0 #10 命中；§S4-3-5 任务状态复位时序约束继承——复位在 phase-2 kv==0 槽/m_i 例外规则由 emit_v 零改动自动保持） |
| #R4-1 | Form B barrier_all 纯移除（15 处，case9 -26.0%/case14 -28.7%，会话 4） | **保留（兜底域）+ 思想延续（mma 域）** | gemm_v0 自平衡段界知识（§S4-3-1：prologue SetFlag/WaitFlag<MTE2_MTE1>(0)+<FIX_M>(0)、epilogue <MTE1_MTE2>(0)+<M_FIX>(0)，common.h:1305-1371）仍是兜底域正确性依据（event id 0 由 gemm_v0 独占自平衡）；Form B-mma = 细粒度 flag 结构**全文无 barrier_all**（barrier 纯化思想的完全体——rev3 Form A 模式扩展）；"细粒度 flag 收支平衡"纪律由 §7.2-2a 收支表静态可验（硬点 4） |
| #R5-1 | Form A single_kv_block 任务批处理（case6 -44.4%/case20 -40.5%，会话 5） | **保留** | Form A 域零改动（IR 逐字节守卫集成员）；其 C 侧批镜像结构（Q per-slot 载入 + SIG_Q_L1 每槽收支 net 0 + l0a 每槽重载）= Form B-mma 单块分支的逐字镜像源 |
| #R6-1 | 单块 O 直写 FIX(L0C)→Output（case6 再 -18.1%/case20 -22.6%，会话 6） | **保留（Form A 专属）；mma 域 v1 不扩展** | Form A 单块分支零改动；Form B-mma 单块分支 v1 保留 w3+phase-2（emit_v `if not form_a:` gate 逐字不动 = V 侧零改动最小变更面；gate 机制已实证 parse-time 条件化保 IR，§S6-3-6）。**扩展列 Stage 3 候选 §10.3-⑨**（等价性依据同源：单块预归一化特化 O=P̂@V 一步终值；前置门 = #R5-2 gate 探针（8/8 在案）扩展 D=256/bM=32/valid_m 尾行 + emit_v gate 谓词泛化 form_a→(form_a or form_b_mma) + WS3 SEM 对在 mma 域的 token 重平衡论证） |

### 0.5 S2 立项论证：kv-outer 循环互换 → **本轮降级不实施**（显式覆盖论证，非静默略过）

**提案**（§S5-4-6）：case7/8（(4,1024,1024,32,128) fp16/bf16 causal，现 0.463/
0.468x）的 s_blocks=16 且任务连续分配（bx 最快维）⇒ 同 (bz,by) 的 16 个 task 各自
装载整条 K/V 块序列（causal 裁剪后平均 4.5 块/task），冗余 ~9×（裁剪口径；未裁剪
口径 16×）。kv-outer / q-task-inner 循环互换可让每个 KV 块装载一次、被 16 个
q-task 共享。**rev4 对其完成设计级论证，结论 = 降级，四步论证如下。**

**论证一：全量 kv-outer 的 UB 状态预算硬性不可行**

online softmax 的 acc_o 是 per-q-task 跨-kv 累加状态（§S5-4-6 原文约束）。互换后
每 vid 需同时持有 s_blocks=16 份 `{acc_o [half_M=32, D=128] fp32 = 16KB +
m_i/sumexp [32]/[32,1] fp32}`：16 × 16KB = **256KB > UB 191.75KB 硬限** ✗
（尚未计 m_i/sumexp ×16 ≈2KB 与 phase-2 工作缓冲）⇒ 全量互换必须状态重构
（GM spill 或任务分组），无"直接互换"路径。

**论证二：GM spill 重构 = 净流量持平略负（定量，case7 口径）**

- 现状 K/V 装载总量（causal 裁剪后）：每 (bz,by) Σ_{bx=0..15} eff_kv(bx) =
  2×(1+2+…+8) = 72 块 × 64KB（K+V 每块 [128,128]fp16×2）= 4.5MB；×128 (bz,by)
  = **576MB**。理想共享（全量 kv-outer）：每 (bz,by) K/V 各 8 块 = 512KB ×128 =
  **64MB**。冗余消除节省上限 = **512MB**。
- spill 重构代价：16 份 acc_o 置 GM（新增 w4 类 workspace [16, bM, D] fp32/核），
  每 kv 步对每 task read-modify-write：acc_o [64,128]fp32 = 32KB 读 + 32KB 写 =
  64KB/iter；case7 总 iter = 2048 task × 平均 4.5 = 9216 ⇒ **590MB 新增 GM 流量
  ≥ 512MB 节省**（净 **+78MB**，另加 m_i/sumexp 状态小流量）。
- 且新增 ws 段全部落在 V 侧 MTE2/MTE3 与 phase-2 read-modify-write 串行链上
  （V 侧 FLOWCTRL 已 59.7-79.3% busy，§S5-1/§S6-3-4 实测）——恶化 V 侧 pacing。
- ⇒ 流量账**等价或略负**，工程复杂度（新增第 4 workspace + SEM 协议扩维 +
  状态生命周期管理）无对应收益。

**论证三：分组折中（G=2）对 case7 现墙无杠杆（预期收益 ≈0）且爆炸半径大**

- G=2 分组 kv-outer（每轮 2 个 q-task 共享 K/V 装载，组间 KV 序列重扫）：UB +1 份
  acc_o = +16KB（P2 116→132KB ≤191.75KB 可行）；K/V 冗余 9×→4.5×，节省 ~256MB
  ≈ case7 总 GM 流量（≈2.1GB：K/V 576MB + ws 9216×160KB≈1.47GB + Q/Out 64MB）
  的 12%。
- **但 case7 的 wall = C 侧 FIXP 95.68% 吞吐墙**（§S6-3-4 实测，w1 fp32 写为主；
  case6 92.38% 同构）。kv-outer 不改变 w1/w3 产出总量（scores/PV 块数不变）⇒
  **FIXP 载荷不变 ⇒ wall 无杠杆**；其节省的 K/V 字节落在 C 侧 MTE2 管（51% busy，
  §S3-3-2——coupled equilibrium 下被 stall 阴影吸收；会话 3 三证伪判决：V 指令削减
  /L1 DB/CROSS_INTERVAL 减半三项单点手段全部零响应，"单边优化被另一边吸收"）。
- 成本侧：GEMM1/GEMM2/softmax 三批的槽位语义从 [cid, stage(kv)] 重构为
  [cid, kv, task-in-group]，SEM 协议 + flag 矩阵 + V 侧状态管理（m_i/sumexp/
  r_factors/sumexp_is ×G）全重构；爆炸半径 = **Form A generic 全路径**（case7/8/
  2/3/5，含已达标 case3 0.763x），失去 IR 逐字节守卫能力，回归风险与 decode
  不回退硬约束同级。预期收益 ≈0 vs 全路径重构风险 ⇒ 严重不匹配。

**论证四：替代路线指认（诚实归因，非遗漏）**

case7/8 的真实剩余杠杆 = **每迭代固定结构成本中的 FIXP/ws 字节**（§S3-3-2 +
§S6-3-4）：w1 dtype 瘦身（fp32→fp16；#R3-3 曾零响应当 case7 处于 coupled
equilibrium，§S6-5 判为下一杠杆重试候选——fixpipe RNE 等价探针 11/11+19/19 已在案，
重试免探针成本，改动 ~2 行）属 **Stage 3 规则内候选**（§10.3-①），不在 rev4 两项
立项范围，不移植进本设计（控制变量：rev4 变更面收敛于 S1）。

**结论与复活条件（入库）**：S2 于 rev4 **降级为"本轮不实施"**——依据 = 论证一
（全量 UB 硬否决）+ 论证二（spill 净流量 +78MB 持平略负）+ 论证三（G=2 对 FIXP
墙无杠杆 + 爆炸半径）。复活条件 = 若 w1/w3 瘦身等 FIXP 流量削减落地后，msprof op
归因显示 case7 C 侧 pacing 管从 FIXP 迁移至 MTE2（K/V 装载成为新墙），则重评 G=2
分组 kv-outer（UB 预算已证可行 +16KB；协议重构按论证三框架展开）。**该降级为设计
决策，orchestrator/Stage 2 不得将 kv-outer 列为 rev4 实施项。**

### 0.6 为何不会再犯同一错误（rev1/rev2/rev3 与调优会话教训对照——revision 门禁项）

- **rev1 教训（threads=2 默认陷阱、MERE 口径病态）**：rev4 全文无 threads 参数
  （Expert 帧 vid=blockIdx.y 结构性绑定；R8 五点证据链继承 §2.2-c，threads=2
  AscendVidReduction×BSHD 4D region bug 护栏与 Developer 语境禁令保留）；MERE
  口径由 rev2-C1 用户裁决关闭，§9.3 逐字继承。
- **rev2 教训（R8/R9 框架 bug、哨兵门限）**：T.Pipelined 全文禁用不变（R9 护栏）；
  Form B-mma 流水 = NUM_STAGES 批量结构 + 手动 side/chunk ping-pong（Form A 同形
  已验证形态）。**K/N-split 为编译期展开常量 2 步（D=256 静态封闭），不引入任何
  新的运行期循环边界形态**（§6.1）；哨兵 kernel 路径门与 golden 降本规定逐字继承。
- **rev3-attempt1 教训（[IMPL-#R3-1]④ C/V 协议失配死锁）**：Form B-mma 批处理
  分支严格镜像已验证协议（num_tb/batch_tasks/eff_* 公式与 SEM set/wait 位置和
  #R3-1/#R5-1 逐字同式），V 侧 emit_v 零改动 ⇒ **C/V 协议面（SEM 层）完全不变**；
  C 侧变更收敛在"每槽 GEMM 执行体内部"（SIG 层新增 token 由 §7.2-2a 收支表静态
  验证 + 探针 P3 位级先行）——协议失配的死锁类别被结构性排除。
- **#R5-3 教训（条件 DMA 非确定性损坏，4 次实验证伪链）**："跳过冗余装载"类优化
  在手动 pipe-pair flag 协议下不安全 ⇒ **rev4 Form B-mma 无任何条件 DMA**：
  K/N-split 2 步为编译期展开恒执行；单块批分支的 Q per-slot 载入 = Form A #R5-1
  已验证结构逐字镜像（其 SIG_Q_L1 每槽 net 0 收支已在案）；`if i < 2` 型批首预载
  仅存在于 generic 分支（Form A 现行已验证结构，Form B-mma generic 的 l0a_s 改为
  任务级预载后**无批内条件载入**）。任何 Stage 2 引入运行期条件装载的尝试必须遵循
  "wait+set 同进同出"（消费侧 wait 同条件化，令牌平衡——§7.2-6 硬规则，#R5-3
  备选写法口径）。
- **#R5-1 教训（预测性否决）**：以"预期收益小"为由不实施必须标注未实测——rev4 对
  decode mma 化的否决**不是预测性否决**：预算不可行 = 硬证据（§4.5-e 推导表：
  bN=512/256 双槽分池 128KB > L0B 64KB 硬限，异形无共池解）+ wall 归因 = 实测
  证据（§S4-3-3 decode 多块 wall = 每槽 MTE2 带宽，case11/15 点名读数 5.3-5.9us/
  slot）；S2 降级同样为定量论证（§0.5）而非定性预测。
- **#R6-2/camodel 纪律（§S6-3-3 判决性知识）**：camodel 仅用于"机制是否激活"定性
  判定——Form B-mma 激活判据 = MTE1/M/FIX 三管重叠段结构 + gemm_v0 每-step
  PipeBarrier<PIPE_M> 从生成码消失（IR diff 可静态验证）；**禁止用于收益定量**
  （camodel FIXP sum 29.8us > 真机全 wall 24.65us 的方向背离判例在案）；msprof
  为唯一性能依据（§9.6 纪律条款）。
- **MEMORY_PLANNING 教训（session-2 §S2-3-2）**：Expert 下 naive 声明和≈实配、
  实编读数优于手算——P3-mma 全部 ⚠ 顶格条目（L0B 64KB 共池、L0C 96KB 主案）
  强制 Stage 2 实编 GetWithOffset/编译探针读数复核（探针 P4，§4.5 制度继承）。
- **压力门方法学（§S5-5 修正）**：竞态压力配置必须**打在被改路径上**——rev4 扩展
  压力集专打 Form B-mma（单块/多块 × causal × S_kv 尾块 × 双 dtype × batch_tasks
  非齐次 × 跨轮 num_tb≥2），继承 4 配置照跑（§9.6-6）。
- **IR 守卫方法学（§S5-5 新增硬门继承）**：16 shape 生成码 dump——Form A 12 守卫 +
  兜底域 decode 守卫**逐字节不变**；Form B-mma 目标 shape 按预期改变（§9.6-2）。

---

## 1. 概述

### 1.1 算子名称

mha（Multi-Head Attention，多头注意力前向；融合推理 attention，host 契约
`mha(query, key, value, scaleValue, is_causal)` 不变）

### 1.2 功能描述

对 BSHD 布局（`[B, S, N, D]`）的 Q/K/V 三个张量执行缩放点积多头注意力：每个
(batch, head) 独立计算 `softmax(Q·Kᵀ·scaleValue)·V`，支持可选的右下角对齐因果
掩码（causal mask），输出与 query 同 shape、同 dtype、contiguous。适用场景：
推理 prefill（S>16 全 D）与 decode/MTP（S≤16）的注意力计算。

### 1.3 数学公式

$$
y = \mathrm{softmax}\!\left(Q K^{\mathsf T} \cdot s\right) V,\qquad
s = \begin{cases} \texttt{scaleValue} & \texttt{scaleValue} > 0 \\ 1/\sqrt{D} & \texttt{scaleValue} \le 0 \end{cases}
$$

对每个 batch `b`、head `n`、query 行 `i`、key 列 `j`：

```
scores[b,i,n,j]   = (Q[b,i,n,:] · K[b,j,n,:]) * s
若 is_causal:      scores[b,i,n,j] = -inf   当 j > i + (S_kv - S)     # 右下角对齐 45° 对角线
attn[b,i,n,:]     = softmax(scores[b,i,n,:], dim=-1)                   # 全 -inf 行 → 权重全 0（golden all_masked 保护）
y[b,i,n,:]        = Σ_j attn[b,i,n,j] * V[b,j,n,:]
```

### 1.4 算法描述

多步融合算子（GEMM → mask → softmax → GEMM），**Flash-Attention 风格 online
softmax 分块算法**（数学上与一次性 softmax 严格等价）+ Expert 批量流水执行结构
（deferred rescale 批量 softmax，与逐迭代 rescale 代数同值且运算次序逐位一致）。
rev4 算法语义与 rev3 **逐位一致**（§9.3.3-6 论证扩展至 Form B-mma 域），九步执行
结构继承 rev3 §1.4（①任务划分 Fixed Core + vid 二分 ②GEMM1→w1 ③双舍入 golden
栅格对齐 ④pre-exp -inf 掩码链 ⑤批量 online softmax phase1→w2 ⑥GEMM2→w3 ⑦O 累加
phase2 deferred rescale ⑧终归一 + sumexp+1e-30 NaN 保护/单块预归一化特化
⑨vid 分行 rows_vid 写回）。

**rev4 唯一算法执行体变更 = prefill D=256（Form B-mma 域）的两个 GEMM 由 gemm_v0
黑盒改为手动 K-split/N-split T.mma 流水**（C-scope 执行体内部；workspace 协议、
V-scope 算式链、任务网格、批处理结构全部不变）：

- **GEMM1**：`acc_s = Q[32,256] · Kᵀ[256,128]` 拆 2 个 k-chunk（KC=128）：
  `mma(l0a_s[0],l0b_pool[0],l0c_s[side],init=True)` → `mma(l0a_s[1],l0b_pool[1],
  l0c_s[side],init=False)` L0C fp32 链式累加——K 升序分片同序（K=0..127 →
  128..255），与 gemm_v0 kL0Size=128 内部 2 k-step 数学同一（§9.3.3-6）。
- **GEMM2**：`PV = P[32,128] · V[128,256]` 拆 2 个 n-chunk（NC=D-half=128）：
  两次 `mma(init=True)` 各写 L0C 独立列段——N 维列独立无累加依赖，分片输出与
  全 N 单次执行逐位同（列段拼接语义）。

**Host 侧预处理声明**（继承）：host 路径只做元数据校验与编译期配置推导（rev4 增量
= 一个纯 Python bool 谓词 `form_b_mma`），**不触碰任何张量数据**——无 transpose/
contiguous/padding/cat 等物理化操作（§4.8 HOST-METADATA-AUDIT）；workspace 由
框架按 workspace_idx 注入；BSHD→(b,n) 条带逻辑重排完全由 kernel 内行跨步 DMA 完成。

### 1.5 数据流图（三形态；rev4 增量 = Form B-mma 段）

```
host: 元数据校验 + 配置/三分派推导（不触碰数据；workspace 框架注入）
                                  │
════════ Cube 核（AIC，T.Scope("C")，每 cid 1 个） ════════
[Form A（prefill D≤128）与 Form B-gemm_v0（decode）数据流 = rev3 §1.5 逐字继承；
 以下仅示 Form B-mma（prefill D=256，bM=32/bN=128/KC=NC=128）增量]

GM Q[B,S,N,D] ──MTE2(行跨步,valid_m)──► L1 q_l1[32,256] (ZN)    （任务级常驻）
   q_l1[:,c*128:(c+1)*128] ──MTE1(直载)──► L0A l0a_s[c]  (c=0,1) （任务级预载，批内全 iter 复用）
GM K ──MTE2(行跨步,valid_n,批内逐块)──► L1 k_l1[128,256] (NZ)
   k_l1[:,c*128:(c+1)*128] ──MTE1(transpose=True)──► L0B l0b_pool[c]  （每 iter ×2 chunk）
   l0a_s[c] + l0b_pool[c] ──M──► L0C l0c_s[side] fp32 （c=0 init=True → c=1 init=False 链）
L0C l0c_s[side] ──FIX──► GM workspace_1[cid,i,32,128] fp32 （SEM_WS1_C2V 批量发放）
GM workspace_2[cid,i] ──MTE2──► L1 p_l1[32,128] (ZN) ──MTE1──► L0A l0a_o[side]
GM V ──MTE2(行跨步,valid_n)──► L1 v_l1[128,256] (ZN)
   v_l1[:,c*128:(c+1)*128] ──MTE1(直载,无 transpose)──► L0B l0b_pool[c] （与 GEMM1 共池分时）
   l0a_o[side] + l0b_pool[c] ──M──► L0C l0c_o[side,:,c*128:(c+1)*128] （c 各 init=True）
L0C l0c_o[side] ──FIX(全块[32,256])──► GM workspace_3[cid,i] fp32 （SEM_WS3_C2V 批量）
════════ Vector 核（2×AIV，T.Scope("V")，vid 半区 hM=16 行）════════
（emit_v 逐字零改动：w1→io_buf→work_ub→双舍入→掩码链→softmax phase1→w2 /
  w3→io3→rescale+acc_o phase2→终归一→out_half→rows_vid 写回，= rev3 §1.5 V 段）
```

内存层级路径全程合法：GM ↔ L1 ↔ L0A/L0B → L0C →(FIX) GM workspace →(MTE2) UB
→(MTE3) GM；CV 交互经显式 GM workspace（本框架唯一物理路径，虚拟通道禁令 §8.6-3
继承）；无 GM→L0 直搬、无跨级违规。**rev4 新增的 L1 列段切片（q/k/v 的 D 维 128
列段 → L0 载入）为 L1→L0 搬运的源 region 切片：列段 = buffer 最内维连续段（行内
连续 256B、行间 gap 恒定）——非"T.copy 列方向 strided 切片"禁令形态**（禁令针对
行内跨步/列维非最内维，§3.5.1 检测行；先例 sparse_flash_mla:372-382/399-409 L1
3D 列段 + transpose 实测在案）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Expert（用户 Primary 决策 2026-09-19，rev4 延续，**禁止回退
Developer/混合**；rev4 立项 = Expert 轨内的设计级性能修订，非模式变更——用户
决策 2026-09-22 原文范围 S1/S2 均为 Expert 形态内部的结构优化）

### 2.2 选型理由与 Expert 形态（rev3 五点论证全部继承，摘要 + rev4 增量复核）

- **(a) 为什么 Expert**：用户决策 + Developer 轨两轮调优实证结构边界（65.05 收口，
  final_report §S2-5"剩余差距主体为模式结构性同步成本"）；Expert 轨四轮调优已兑现
  65.05→71.70（+6.65 分，0.55x→0.9054x，M2 达成）——rev4 在 Expert 轨内继续。
- **(b) Expert 唯一形态**：`T.Kernel(core_num, is_npu=True) as (cid, vid)`（无
  threads 参数——带 threads 切入 dev_mode frame 构成非法混合态，kernel.py:249-263；
  vid=blockIdx.y extent 2 结构性绑定，ir.cc:281-290）+ 手动 T.Scope("C"/"V") +
  Cross-core Semaphore（mode=2）+ vid 手动二分（half_M=bM//2 行/vid，源码显式
  [bM//2] buffer + vid*half_M 偏移）。
- **(c) R8 复检（Expert 不触发，硬点 5 复检项）**：AscendVidReduction pass 触发键
  `threadIdx.x` 在 Expert 帧不存在（vid=blockIdx.y）→ threads_cnt_ 恒 1 → 三处
  改写入口 `if (threads_cnt_ != 2)` 全短路（ascend_vid_reduction.cc:1146-1152/
  :307/:666/:1109）= 全局 no-op；vid 二分为源码显式无自动改写环节（三症状结构性
  不可达）；WorkspaceReduction vid 折半只作用虚拟通道拷贝（target set 永不命中，
  R12 禁令保障）。**threads=2 × AscendVidReduction × BSHD 4D region bug 护栏
  继承**：rev4 全文无 threads 参数（§8.6-1 grep 自检）；任何回退 Developer 尝试
  命中 rev2 §10.1-R8 禁令（复现路径 debug_log.md deviation#1 探针族在案）。
- **(d) R9 复检（不适用）**：MEMORY_PLANNING=False → planner 仅显式 alloc 顺序
  编址 + address_map attr 恒全量注册（ascend_memory_planning.cc:59-92；
  codegen_ascend.cc:1245/805-840），无"注入未注册"buffer；循环形态 = T.serial
  （运行期 TIR 边界）+ 手动 ping-pong。**T.Pipelined 全文禁用**（硬点 5/R9 继承）；
  rev4 K/N-split 为编译期展开常量 2 步，不新增循环边界形态。
- **(e) 循环形态**：外层 T.serial(eff_num_outer) + 内层 T.serial(batch_iters) +
  手动 side=i%2 / chunk c 双缓冲——Form B-mma 与 Form A 完全同式（mha_sink:278/
  284/433/439 Expert 运行期边界 CI 先例继承；C/V 双 scope 同式复制保 flag 配对）。

### 2.3 模式影响（rev3 表继承 + rev4 增量行）

| 维度 | rev4 选择（Expert；增量加粗） |
|------|------------------------------|
| 内存分配 | T.alloc_L1/ub/L0A/L0B/L0C 显式声明制（allocate.py:140-158）；**Form B-mma 域新增显式 L0A/L0B/L0C 池（l0a_s/l0a_o/l0b_pool/l0c_s/l0c_o，§4.3）——gemm_v0 内部 L0 管理仅存于 Form B-gemm_v0 兜底域**；不使用 alloc_shared/alloc_fragment；不写 T.annotate_address（planner 顺序编址，mha_sink 先例继承） |
| 计算方式 | Cube：Form A / **Form B-mma** `T.mma`（显式 L0，**init=True/False 链式 K-split + C region 切片 N-split**）；Form B-gemm_v0 `T.gemm_v0`（kL0Size 三档，decode 域）。Vector：T.tile.* + T.reduce_max/sum（不变） |
| 作用域 | 手动 `with T.Scope("C"):` / `with T.Scope("V"):`（warpgroup.py:74，继承） |
| 同步方式 | 全手动（继承）：C 侧 set_flag/wait_flag 有向 pipe pair（**Form B-mma 扩展 SIG_L0KC/L0AO/L0AS token，§7.2-2a**）；兜底域 gemm_v0 自平衡零用户 flag（#R4-1）；跨核 set/wait_cross_flag SEM 0-5（不变）；V 侧 flag 管程（不变）；init/destroy 闭环（§7.2-4 扩展） |
| Kernel 启动 | `T.Kernel(core_num, is_npu=True) as (cid, vid)`——禁带 threads（继承 §8.6-1） |
| CV 数据中转 | 显式 GM workspace ×3（workspace_idx=[4,5,6]，[cid,stage] 索引，继承）；虚拟通道裸拷贝禁令（§8.6-3/R12）不变 |
| pass_configs | 四项全 False（programming-model-guide §2.2 Expert 行 + mha_sink:38-43 逐字，继承） |

---

## 3. API 映射设计

### 3.1 公式拆解（继承 rev3 §3.1 十步语义；rev4 增量 = Form B-mma 执行体拆分）

rev3 §3.1 十步表逐字继承（1 GEMM1 Cube / 2 双舍入 Vector / 3 掩码 Vector /
4 rowmax / 5 deferred rescale 因子 / 6 exp P / 7 rowsum / 8 GEMM2 Cube /
9 O 累加 phase2 / 10 终归一），执行核标注与掩码等价性论证（i_g/j_g 索引映射、
row_thresh vid 半区、尾列钳位）不变。**rev4 唯一语义增量 = 步骤 1/8 在 Form B-mma
域的执行体拆分**：

| 步骤 | rev3 执行体（Form B gemm_v0 域） | rev4 Form B-mma 执行体 | 数学等价性 |
|------|--------------------------------|------------------------|-----------|
| 1（GEMM1 Q·Kᵀ） | gemm_v0(transpose_B=True, kL0Size=128) 内部 2 k-step（每-step PipeBarrier<PIPE_M> 排空前一 mma——§S4-3-2 wall 根因） | 手动 2×k-chunk：`mma(l0a_s[0],l0b_pool[0],l0c_s[side],init=True)` → `mma(l0a_s[1],l0b_pool[1],l0c_s[side],init=False)`（KC=128） | K 升序分片累加同序（K=0..127 → 128..255，C0 顺序累加）；fp32 L0C 累加域同一；§S4-3-4-① 同序链式位级稳定知识（单 mma k=256 ≡ 链式 2×mma k=128，12/12 BITEXACT）+ 锚-B 诊断追踪（§9.2） |
| 8（GEMM2 P·V） | gemm_v0(init=True) 内部 N 分片（D=256 → 2 内部 step） | 手动 2×n-chunk：`mma(l0a_o[side],l0b_pool[c],l0c_o[side,:,c*NC:(c+1)*NC],init=True)`，c=0,1（NC=128；主案 a） | N 维列独立（无跨列累加）——分片输出与全 N 单次执行逐位同（列段拼接）；探针 P2 位级验证 |

### 3.2 TileLang API 映射（Expert；rev3 全表继承 + Form B-mma 增量行）

rev3 §3.2 全表继承（任务网格 T.Kernel/任务循环 T.serial+my_start/-my_count/KV 裁剪
eff_* 同式/Q-K-V 搬入 BSHD 行跨步/L1 布局 annotate_layout/GEMM1 Form A mma/
L0C→w1 FIX/w1→UB vid 半区/双舍入 rr_ub 链/掩码链五 op/rowmax/rescale/exp/rowsum/
单块特化/P 写出 SIG_S_HALF 闭环/w2→p_l1/GEMM2 Form A/phase2 累加/终归一/rows_vid
写回）。**rev3 表中"GEMM1（Form B）/GEMM2（Form B）"两行（gemm_v0）适用域收窄为
Form B-gemm_v0（decode 兜底）**；新增 Form B-mma 四行：

| 步骤 | 数学表达 | TileLang API | 参数/说明 | 形态 |
|------|----------|-------------|----------|------|
| GEMM1 K-split（Form B-mma） | Q·Kᵀ，K=D=256 拆 2×KC=128 | 任务级：`T.copy(q_l1[:, c*KC:(c+1)*KC], l0a_s[c,:,:])`（c=0,1，SIG_L0AS 闭环）；每 iter：`T.copy(k_l1[:, c*KC:(c+1)*KC], l0b_pool[c,:,:], transpose=True)` + `T.mma(l0a_s[c,:,:], l0b_pool[c,:,:], l0c_s[side,:,:], init=(c==0))` | L1 列段 region 切片（最内维连续段）+ transpose 载入（copy_op.py:261 npu_copy_v2）；**k_actual/n_actual/real_k/real_n 均不需要**——chunk 宽 128 由 region extent 静态推导（K==K_B 校验 customize.py:186-189 自动满足），bN 尾块由 L1 need_clear 清零 + 掩码钳位处理（Form A 论证逐字继承，mma 全 N 执行）；init=False 链式累加（customize.py:215：C ptr "rw"）；先例 sparse_flash_mla:371-423（D2=256 同形 K-split init=True→False 链 + 列段 transpose 载入） | Expert |
| GEMM2 N-split（Form B-mma，主案 a） | P·V，N=D=256 拆 2×NC=128 | `T.copy(v_l1[:, c*NC:(c+1)*NC], l0b_pool[c,:,:])`（直载无 transpose）+ `T.copy(p_l1, l0a_o[side,:,:])` + `T.mma(l0a_o[side,:,:], l0b_pool[c,:,:], l0c_o[side,:,c*NC:(c+1)*NC], init=True)` | C BufferRegion 列段切片：M,N 由 region extent 推导（customize.py:151-168 retrieve_shape），C ptr offset 由 retrieve_ptr region 分支计算（:191-209）；列段边界 128 = 8×16 fractal 对齐；**C region 切片 mma 为仓内首例 → 探针 P2 前置**；FIX 写出 = `T.copy(l0c_o[side,:,:], workspace_3[cid,i,:,:])` 全块（两 n-chunk mma 完成后一次，M 管序保证两段列均就绪） | Expert |
| GEMM2 备选案 b（P2 探针失败即切换，零设计变更） | 同上 | l0c_o 改 [2,bM,NC=128] 窄池 + 每 n-chunk：`T.mma(l0a_o[side],l0b_pool[c],l0c_o[side2],init=True)` → `T.copy(l0c_o[side2], workspace_3[cid,i,:,c*NC:(c+1)*NC])` 列段 FIX ×2 | sparse_flash_mla:673-681 逐字先例（workspace_o 列段 + L0C→GM）；copy_l0c_to_gm realDstN 运行期 N 支持已由 #R6-1 gate 探针 8/8 整数精确 BITEXACT 实证（src/op/ascend.cc:531-543 + common.h:190-213，§S6-3-1）；SEM_WS3_C2V set 位于两段 FIX 之后（FIX 同管序保证两段均落地） | Expert |
| l0b_pool 共池分时 | GEMM1 B=Kᵀ chunk / GEMM2 B=V chunk | 单一 `l0b_pool = T.alloc_L0B([2, KC, block_N], dtype)`；GEMM1 批 transpose 载入 / GEMM2 批直载 | 同形共池 + 相位分时（每外层批内 GEMM1 批先行、GEMM2 批后行，批内无交叉）；SIG_L0KC+0/+1 token 流跨相位延续（流式信号量无相位概念，收支恒等 §7.2-2a）；**已验证先例 = 本算子 Form A P2 共池**（mha.py:576-578 `l0b_o = l0b_s if block_N == D`：同池既接 k transpose 又接 v 直载，位级验证 + 四轮调优在案）；L0 buffer 无 annotate_layout（layout 由 copy 方式与 mma intrinsic 决定，L1 侧 ZN/NZ annotation 不变） | Expert |
| unit_flag / mma-fixpipe 融合 | （v1 不启用） | `T.mma(..., unit_flag=None)` = legacy 6 参形式（customize.py:121-122："every existing caller is byte-for-byte unchanged"，C++ 默认 unitFlag=0 每 mma 独立完结） | v1 主案 = legacy（与 Form A 现行参数形式一致，差分面最小；init=False 链正确性由 M 管序 + unitFlag=0 语义保障，探针 P3 位级验证）；unit_flag=0b10/0b11 流水模式（sparse_flash_mla 用法，含窄 N 场景 pipe_barrier("m") 特判 :395-396/:421-422）= Stage 3 候选 §10.3-② | Expert |

#### 3.2.1 继承映射

rev3 §3.2.1（Developer 12 项算法优化逐项落点表）全表继承有效；rev4 增量 =
§0.4-a/b 两张"保留/替换/扩展"标注表（12 算法项 + 4 调优成果逐项存续形态，硬点 3）。

### 3.3 计算伪代码（三形态；rev4 增量 = Form B-mma C-scope + 三分派）

```python
pass_configs = {四项全 False}                    # 逐字继承 rev3 §3.3
NUM_STAGES = 8            # 继承（#R5-4 12/16 暂缓判定维持：rev4 控制变量，S1 归因纯度优先）
CROSS_INTERVAL = 1        # 继承（#R3-4 判决实验：减半零响应，不动）

def _cube_core_num() -> int:  # host 设备属性查询（继承 Developer 终态 #5）
    try:
        import torch_npu
        return int(torch_npu.npu.get_device_properties(0).cube_core_num)
    except Exception:
        return 20           # 910B 级保守回退

@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)
def mha_kernel(B, S, S_kv, N, D, dtype="float16", scaleValue=-1.0, is_causal=False):
    accum_dtype = "float"
    assert NUM_STAGES % 2 == 0
    sm_scale = scaleValue if scaleValue > 0 else (1.0 / D) ** 0.5
    # —— tile 三档体系（逐字继承 Developer 终态硬限解，rev3 §5.2）——
    block_M = 16 if S <= 16 else (32 if D == 256 else 64)
    if S <= 16:
        block_N = 512 if (D <= 128 and S_kv > 256) else (256 if S_kv > 128 else 128)
    else:
        block_N = 128
    gemm1_kL0 = (32 if (S <= 16 and D <= 128 and S_kv > 256)
                 else (64 if (S <= 16 and D == 256 and S_kv > 128) else 128))  # 仅兜底域
    # —— rev4 三分派谓词（互斥且穷尽，§5.6 审计）——
    form_a     = (S > 16) and (D <= 128)   # Form A: mma 全流水（不变）
    form_b_mma = (S > 16) and (D == 256)   # Form B-mma: 手动 K/N-split mma 流水（新增）
    # else: Form B-gemm_v0（S <= 16 decode 全 D，兜底，源码逐字不变）
    half_M = block_M // 2                  # P3-mma: 16
    KC = 128; NC = 128                     # Form B-mma chunk 宽（编译期常量）
    if form_b_mma:
        assert D % KC == 0 and D % NC == 0  # 静态整除恒真（D==256 封闭）
    causal_offset = S_kv - S
    need_bound_mask = S_kv % block_N != 0
    need_mask = is_causal or need_bound_mask
    offset_eff = causal_offset if is_causal else 2**30
    s_blocks = (S + block_M - 1) // block_M
    kv_iters = (S_kv + block_N - 1) // block_N
    single_kv_block = kv_iters == 1
    block_num = B * N * s_blocks
    core_num = min(block_num, _cube_core_num())
    q_tasks, r_tasks = block_num // core_num, block_num % core_num

    # SEM 0-5 / SIG_K_L1=0 SIG_P_L1=1 SIG_V_L1=2 SIG_Q_L1=3 / SIG_IO_UB=0 SIG_IO3=2
    # SIG_S_HALF=0：全部逐字继承 rev3（mha.py:170-200）。
    # Form B-mma 新增 C 侧 event id（§7.2-2a）：
    #   MTE1<->M pair: SIG_L0KC = 0,1（l0b_pool chunk0/1）
    #                  SIG_L0AO = 2,3（l0a_o side0/1）
    #                  SIG_L0AS = 4,5（l0a_s chunk0/1，任务级）
    #   M<->FIX pair:  SIG_L0C  = 0,1（l0c_s / l0c_o 分时同流，Form A P1 先例）

    @T.macro
    def emit_v(...):    # 共享 V-scope emitter —— rev4 逐字零改动（mha.py:203-541）
        ...

    # ============ Python 级三分支构建三个自包含 prim_func（rev3 §6.1 注记扩展）============
    if form_a:
        @T.prim_func
        def main(Q, K, V, Output, workspace_1, workspace_2, workspace_3):
            ...   # Form A prim_func —— rev4 逐字零改动（mha.py:546-885）
        return main
    elif form_b_mma:
        @T.prim_func
        def main(
            Q: T.Tensor([B, S, N, D], dtype),
            K: T.Tensor([B, S_kv, N, D], dtype),
            V: T.Tensor([B, S_kv, N, D], dtype),
            Output: T.Tensor([B, S, N, D], dtype),
            workspace_1: T.Tensor([core_num, NUM_STAGES, block_M, block_N], accum_dtype),
            workspace_2: T.Tensor([core_num, NUM_STAGES, block_M, block_N], dtype),
            workspace_3: T.Tensor([core_num, NUM_STAGES, block_M, D], accum_dtype),
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                # ---- C 侧 L1（ZN/NZ 标注 = Form A 同构）----
                q_l1 = T.alloc_L1([block_M, D], dtype)          # [32,256] 16KB
                k_l1 = T.alloc_L1([block_N, D], dtype)          # [128,256] 64KB
                v_l1 = T.alloc_L1([block_N, D], dtype)          # [128,256] 64KB
                p_l1 = T.alloc_L1([block_M, block_N], dtype)    # [32,128] 8KB
                T.annotate_layout({q_l1: make_zn_layout(q_l1), k_l1: make_nz_layout(k_l1),
                                   p_l1: make_zn_layout(p_l1), v_l1: make_zn_layout(v_l1)})
                # ---- L0 显式池（P3-mma，§4.3/§4.5）----
                l0a_s    = T.alloc_L0A([2, block_M, KC], dtype)          # 16KB 任务级常驻
                l0a_o    = T.alloc_L0A([2, block_M, block_N], dtype)     # 16KB side ping-pong
                l0b_pool = T.alloc_L0B([2, KC, block_N], dtype)          # 64KB ⚠ 共池顶格
                l0c_s    = T.alloc_L0C([2, block_M, block_N], accum_dtype)   # 32KB
                l0c_o    = T.alloc_L0C([2, block_M, D], accum_dtype)         # 64KB（主案 a）
                # 备选案 b（P2 探针失败切换）：l0c_o = T.alloc_L0C([2, block_M, NC], accum_dtype)
                # ---- V 侧 UB：与 rev3 逐字一致（emit_v 零改动前提；P3 行 ≈79KB）----
                io_buf = T.alloc_ub([2, half_M, block_N], accum_dtype)
                io3 = io_buf if io3_is_alias else T.alloc_ub([2, half_M, D], accum_dtype)
                work_ub/buf_2d/rt_2d/r_2d/mask_2d/col_pos/row_thresh/rr_ub/acc_s_half/
                acc_o/out_half/m_i/m_i_prev/sumexp/r_factors/sumexp_is = <rev3 声明逐字>

                my_start = cid * q_tasks + T.if_then_else(cid < r_tasks, cid, r_tasks)
                my_count = q_tasks + T.if_then_else(cid < r_tasks, 1, 0)

                # ==================== Cube 核（Form B-mma）====================
                with T.Scope("C"):
                    # init（§7.2-4 扩展；13 项预置，收支表 §7.2-2a）:
                    T.set_cross_flag("MTE2", SEM_WS2_C2V)
                    T.set_flag("MTE1","MTE2", SIG_K_L1); T.set_flag("MTE1","MTE2", SIG_P_L1)
                    T.set_flag("MTE1","MTE2", SIG_V_L1); T.set_flag("MTE1","MTE2", SIG_Q_L1)
                    T.set_flag("M","MTE1", SIG_L0KC);   T.set_flag("M","MTE1", SIG_L0KC+1)
                    T.set_flag("M","MTE1", SIG_L0AO);   T.set_flag("M","MTE1", SIG_L0AO+1)
                    T.set_flag("M","MTE1", SIG_L0AS);   T.set_flag("M","MTE1", SIG_L0AS+1)
                    T.set_flag("FIX","M", SIG_L0C);     T.set_flag("FIX","M", SIG_L0C+1)

                    if single_kv_block:
                        # ---- 任务批处理分支（#R3-1/#R5-1 协议镜像 + mma 本体）----
                        num_tb = T.ceildiv(my_count, NUM_STAGES)
                        for kb in T.serial(num_tb):
                            _remaining_t = my_count - kb * NUM_STAGES
                            batch_tasks = T.if_then_else(_remaining_t < NUM_STAGES,
                                                         _remaining_t, NUM_STAGES)
                            valid_n = min(block_N, S_kv)          # 单块 idx==0
                            # --- GEMM1 批 → w1 槽 i ---
                            T.wait_cross_flag(SEM_WS1_V2C)
                            for i in T.serial(batch_tasks):
                                side = i % 2
                                task = my_start + kb*NUM_STAGES + i
                                bx/by/bz 分解; s0 = bx*block_M; valid_m = T.min(block_M, S-s0)
                                # Q: GM→L1 per-slot（#R5-1 delta-a 镜像）
                                T.wait_flag("MTE1","MTE2", SIG_Q_L1)
                                T.copy(Q[bz, s0:s0+valid_m, by, :], q_l1)
                                T.set_flag("MTE2","MTE1", SIG_Q_L1)
                                T.wait_flag("MTE2","MTE1", SIG_Q_L1)
                                # K: GM→L1（单块 j0=0）
                                T.wait_flag("MTE1","MTE2", SIG_K_L1)
                                T.copy(K[bz, 0:valid_n, by, :], k_l1)
                                T.set_flag("MTE2","MTE1", SIG_K_L1)
                                # Q: L1 列段→L0A 两 k-chunk 每槽重载（#R5-1 delta-b 镜像：
                                # Q 任务相关；SIG_Q_L1 于载入后归还，每槽 net 0。
                                # SIG_L0AS 每槽一轮：wait(M→MTE1)→载入→set(MTE1→M)→
                                # wait(MTE1→M)（M 管确认，槽头阻塞零损失）；槽末归还见下）
                                T.wait_flag("M","MTE1", SIG_L0AS)
                                T.copy(q_l1[:, 0:KC], l0a_s[0,:,:])
                                T.wait_flag("M","MTE1", SIG_L0AS+1)
                                T.copy(q_l1[:, KC:2*KC], l0a_s[1,:,:])
                                T.set_flag("MTE1","MTE2", SIG_Q_L1)     # q_l1 MTE1 读毕归还
                                T.set_flag("MTE1","M", SIG_L0AS); T.set_flag("MTE1","M", SIG_L0AS+1)
                                T.wait_flag("MTE1","M", SIG_L0AS); T.wait_flag("MTE1","M", SIG_L0AS+1)
                                # K-split mma 链（GEMM1_BODY，§8.3 宏展开；宏内零 L0AS）
                                T.wait_flag("MTE2","MTE1", SIG_K_L1)
                                <GEMM1_BODY(c=0, init=True, side)>
                                <GEMM1_BODY(c=1, init=False, side)>
                                T.set_flag("MTE1","MTE2", SIG_K_L1)     # k_l1 两 chunk 读毕归还
                                                                        #（必须位于 c1 载入后）
                                <FIX: wait(M,FIX,SIG_L0C+side) → copy(l0c_s[side] → w1[cid,i])
                                      → set(FIX,M,SIG_L0C+side)>
                                T.set_flag("M","MTE1", SIG_L0AS); T.set_flag("M","MTE1", SIG_L0AS+1)
                                # ↑ 槽末归还（M 管序：该槽 GEMM1 末 mma 后 l0a_s 释放；
                                #   GEMM2 不读 l0a_s——下一槽头 wait 消费，每槽 net 0）
                                if (i+1) % CROSS_INTERVAL == 0 or i == batch_tasks-1:
                                    T.set_cross_flag("FIX", SEM_WS1_C2V)
                            # --- GEMM2 批 → w3 槽 i（v1 保留 w3+phase-2，emit_v 零改动）---
                            T.wait_cross_flag(SEM_WS3_V2C)
                            for i in T.serial(batch_tasks):
                                side = i % 2; task 分解同式
                                T.wait_flag("MTE1","MTE2", SIG_V_L1)
                                T.copy(V[bz, 0:valid_n, by, :], v_l1)
                                T.set_flag("MTE2","MTE1", SIG_V_L1)
                                T.wait_flag("MTE1","MTE2", SIG_P_L1)
                                if i % CROSS_INTERVAL == 0: T.wait_cross_flag(SEM_WS2_V2C)
                                T.copy(workspace_2[cid, i, :, :], p_l1)
                                T.set_flag("MTE2","MTE1", SIG_P_L1)
                                T.wait_flag("M","MTE1", SIG_L0AO+side)
                                T.wait_flag("MTE2","MTE1", SIG_P_L1)
                                T.copy(p_l1, l0a_o[side,:,:])
                                T.set_flag("MTE1","M", SIG_L0AO+side)
                                T.wait_flag("MTE2","MTE1", SIG_V_L1)
                                <GEMM2_BODY(c=0, side); GEMM2_BODY(c=1, side)>   # §8.3
                                T.set_flag("MTE1","MTE2", SIG_V_L1)     # v_l1 两 chunk 读毕归还
                                <FIX: l0c_o[side] 全块 → w3[cid,i]>
                                if (i+1) % CROSS_INTERVAL == 0 or i == batch_tasks-1:
                                    T.set_cross_flag("FIX", SEM_WS3_C2V)
                            T.set_cross_flag("MTE2", SEM_WS2_C2V)
                    else:
                        # ---- generic 多块分支（case9/4 主战场；Form A generic 同构）----
                        for t in T.serial(my_count):
                            task = my_start + t; bx/by/bz; s0; valid_m
                            eff_kv_iters = <causal 裁剪式，C/V 同式逐字继承>
                            eff_num_outer = T.ceildiv(eff_kv_iters, NUM_STAGES)
                            <SIG_Q_L1 管程> T.copy(Q[bz, s0:s0+valid_m, by, :], q_l1)
                            # l0a_s 任务级预载（q kv-invariant，任务内一次；SIG_L0AS
                            # 每任务一轮：wait(M→MTE1)→载入→set(MTE1→M)→wait(MTE1→M)
                            # ——M 管阻塞前移到任务头（此时 M 管无在途工作，零损失），
                            # 任务内全部 iter 的 mma 读 l0a_s 由 M 管序天然保序）
                            T.wait_flag("M","MTE1", SIG_L0AS)
                            T.copy(q_l1[:, 0:KC], l0a_s[0,:,:])
                            T.wait_flag("M","MTE1", SIG_L0AS+1)
                            T.copy(q_l1[:, KC:2*KC], l0a_s[1,:,:])
                            T.set_flag("MTE1","M", SIG_L0AS); T.set_flag("MTE1","M", SIG_L0AS+1)
                            T.wait_flag("MTE1","M", SIG_L0AS); T.wait_flag("MTE1","M", SIG_L0AS+1)
                            T.set_flag("MTE1","MTE2", SIG_Q_L1)         # q_l1 MTE1 读毕即归还
                            for k in T.serial(eff_num_outer):
                                batch_iters = T.if_then_else(eff_kv_iters-k*NUM_STAGES <
                                                             NUM_STAGES, ..., NUM_STAGES)
                                T.wait_cross_flag(SEM_WS1_V2C)
                                for i in T.serial(batch_iters):         # GEMM1 批
                                    side = i % 2; idx = k*NUM_STAGES+i; j0 = idx*block_N
                                    valid_n = T.min(block_N, S_kv - j0)
                                    T.wait_flag("MTE1","MTE2", SIG_K_L1)
                                    T.copy(K[bz, j0:j0+valid_n, by, :], k_l1)
                                    T.set_flag("MTE2","MTE1", SIG_K_L1)
                                    T.wait_flag("MTE2","MTE1", SIG_K_L1)
                                    <GEMM1_BODY(c=0, init=True, side)>
                                    <GEMM1_BODY(c=1, init=False, side)>
                                    T.set_flag("MTE1","MTE2", SIG_K_L1)
                                    <FIX → w1[cid,i]; SEM_WS1_C2V 批量>
                                T.wait_cross_flag(SEM_WS3_V2C)
                                for i in T.serial(batch_iters):         # GEMM2 批
                                    side = i % 2; idx/j0/valid_n 同式
                                    <V 载入 SIG_V_L1 + P 载入 SIG_P_L1/SEM_WS2_V2C（同单块分支）>
                                    <l0a_o 载入（SIG_L0AO+side）>
                                    <GEMM2_BODY(c=0, side); GEMM2_BODY(c=1, side)>
                                    T.set_flag("MTE1","MTE2", SIG_V_L1)
                                    <FIX: l0c_o[side] 全块 → w3[cid,i]; SEM_WS3_C2V 批量>
                                T.set_cross_flag("MTE2", SEM_WS2_C2V)
                            # 任务末：l0a_s 归还（M 管序：本任务全部 GEMM1 mma 读毕；
                            # 下一任务头 wait(M→MTE1) 消费——每任务 set/wait 各 1 轮）
                            T.set_flag("M","MTE1", SIG_L0AS); T.set_flag("M","MTE1", SIG_L0AS+1)
                    # destroy（§7.2-4）：对称 wait ×12
                    T.wait_flag("MTE1","MTE2", SIG_K_L1); T.wait_flag("MTE1","MTE2", SIG_P_L1)
                    T.wait_flag("MTE1","MTE2", SIG_V_L1); T.wait_flag("MTE1","MTE2", SIG_Q_L1)
                    T.wait_flag("M","MTE1", SIG_L0KC);    T.wait_flag("M","MTE1", SIG_L0KC+1)
                    T.wait_flag("M","MTE1", SIG_L0AO);    T.wait_flag("M","MTE1", SIG_L0AO+1)
                    T.wait_flag("M","MTE1", SIG_L0AS);    T.wait_flag("M","MTE1", SIG_L0AS+1)
                    T.wait_flag("FIX","M", SIG_L0C);      T.wait_flag("FIX","M", SIG_L0C+1)

                # ==================== Vector 核（共享 emitter，零改动）====================
                with T.Scope("V"):
                    emit_v(...)      # rev3 mha.py:203-541 逐字（single_kv_block 分支的
                                     # `if not form_a:` phase-2 gate：form_a=False → 保留）
        return main
    else:
        @T.prim_func
        def main(...):
            ...   # Form B-gemm_v0 prim_func —— rev4 逐字零改动（mha.py:887-1073）
        return main

    # GEMM1_BODY(c, init, side) 宏展开（每 k-chunk 一轮 SIG_L0KC+c token；实现层
    # Python 宏/@T.macro，c 为编译期 Python int，非运行期循环。宏内零 SIG_L0AS——
    # l0a_s 就绪由任务头/槽头预载段的 wait(MTE1→M) 一次性确认，任务内 mma 读
    # l0a_s 由 M 管序保序，无重载者）：
    #   T.wait_flag("M","MTE1", SIG_L0KC+c)                     # 池 c 已释放
    #   T.copy(k_l1[:, c*KC:(c+1)*KC], l0b_pool[c,:,:], transpose=True)
    #   T.set_flag("MTE1","M", SIG_L0KC+c)
    #   T.wait_flag("MTE1","M", SIG_L0KC+c)
    #   若 c==0: T.wait_flag("FIX","M", SIG_L0C+side)           # l0c_s 槽可写
    #   T.mma(l0a_s[c,:,:], l0b_pool[c,:,:], l0c_s[side,:,:], init=init)
    #   T.set_flag("M","MTE1", SIG_L0KC+c)                      # 池 c 释放（下一装载）
    #   若 c==1: T.set_flag("M","FIX", SIG_L0C+side)            # scores 全块就绪
    # GEMM2_BODY(c, side) 宏展开（主案 a）：
    #   T.wait_flag("M","MTE1", SIG_L0KC+c)
    #   T.copy(v_l1[:, c*NC:(c+1)*NC], l0b_pool[c,:,:])         # 直载无 transpose
    #   T.set_flag("MTE1","M", SIG_L0KC+c)
    #   T.wait_flag("MTE1","M", SIG_L0KC+c)
    #   若 c==0: T.wait_flag("MTE1","M", SIG_L0AO+side)         # p 就绪（l0a_o 载入段 set）
    #            T.wait_flag("FIX","M", SIG_L0C+side)           # l0c_o 槽可写
    #   T.mma(l0a_o[side,:,:], l0b_pool[c,:,:],
    #         l0c_o[side,:,c*NC:(c+1)*NC], init=True)
    #   T.set_flag("M","MTE1", SIG_L0KC+c)
    #   若 c==1: T.set_flag("M","MTE1", SIG_L0AO+side)          # l0a_o 两 chunk mma 读毕归还
    #            T.set_flag("M","FIX", SIG_L0C+side)            # PV 全块就绪
```

（SIG_L0AS 令牌流定案说明：l0a_s 为"一次装载、任务内 N iter 只读"形态，1:1
set/wait 无法表达 1:N 读——采用**任务级单轮**方案：任务头（generic）/槽头（单块
批）`wait(M→MTE1)→载入→set(MTE1→M)→wait(MTE1→M)`（末 wait 将 M 管阻塞前移到
无在途工作处，零流水损失），任务末/槽末 `set(M→MTE1)` 归还（M 管序保证 set 生效
时本任务全部 GEMM1 mma 已读毕）；GEMM1_BODY 宏内零 L0AS。收支封闭：init 预置
set(M→MTE1)×2 → 首任务头 wait 消费；每任务 set/wait 各 1 轮；destroy wait×2
消费末任务归还。该结构与 #R5-1 SIG_Q_L1 每槽 net-0 收支同构（任务级资源 +
读毕归还），无条件 DMA、无条件 wait/set——#R5-3 教训合规（§7.2-6）。）

Host 包装函数（**逐字继承 rev3 §3.3**，签名/断言/out_idx 语义零变化）：

```python
def mha(query, key, value, scaleValue=-1.0, is_causal=False):
    # —— 仅元数据校验（不触碰张量数据，§4.8）：rev3 断言集逐字保留 ——
    assert query.dtype == key.dtype == value.dtype
    assert query.dtype in (torch.float16, torch.bfloat16)
    assert query.is_contiguous() and key.is_contiguous() and value.is_contiguous()
    B, S, N, D = query.shape; S_kv = key.shape[1]
    assert tuple(key.shape) == (B, S_kv, N, D) and tuple(value.shape) == (B, S_kv, N, D)
    assert D in (64, 128, 256)
    assert 1 <= B <= 128 and 1 <= S <= 2048 and 1 <= S_kv <= 4096 and 1 <= N <= 64
    if is_causal: assert S_kv >= S
    kernel = mha_kernel(B=B, S=S, S_kv=S_kv, N=N, D=D,
                        dtype=str(query.dtype).removeprefix("torch."),
                        scaleValue=float(scaleValue), is_causal=bool(is_causal))
    return kernel(query, key, value)
```

### 3.4 API 可行性确认（不凭记忆猜 API——rev3 全表继承 + rev4 新增项逐项源码/示例实证）

rev3 §3.4 全表继承（状态不变）：T.Kernel Expert 帧（kernel.py:215-263/ir.cc:281-290
+ 三先例）✅；T.alloc_L1/L0A/L0B/L0C/ub（allocate.py:140-158 + mha_sink:190-231）✅；
T.Scope("C"/"V")（warpgroup.py:74）✅；T.set_flag/wait_flag（ascend.py:233-270，
`_pipe` 字面量集 ascend.py:9）✅；T.set_cross_flag/wait_cross_flag（ascend.py:191-231，
mode=2，**wait 单参——pipe 参数仅 A5 支持，910B 必须留空**）✅；T.barrier_all/
pipe_barrier（ascend.py:275-300；Form A/Form B-mma 循环体内禁用 barrier_all）✅；
T.mma = npu_gemm 别名（language/__init__.py:75）✅；T.gemm_v0（ascend.py:418-530，
兜底域）✅；T.copy transpose（copy_op.py:257-266）✅；copy_l0c_to_gm（common.h:187-213）✅；
GM(4D ws)→UB vid 半区切片（common.h:233-268 + mha_sink:449-452）✅；GM→L1 BSHD
行跨步 + need_clear（common.h:84-92）✅；UB→GM rows_vid（common.h:269-283）✅；
annotate_layout ZN/NZ（mha_sink:5/196-203）✅；T.tile.* 全族（ascend_tile.py 行号
见 rev3）✅；T.reduce_max/sum ✅；T.serial 运行期边界（mha_sink:278/284/433/439）✅；
workspace_idx 框架注入（flash_attn_bhsd.py:8/254）✅；禁用项五条（T.Pipelined/
threads/虚拟通道裸拷贝/pto.py/三维 Kernel 与 T.gemm 通用版）🚫 不变。

**rev4 新增项（逐项源码 + 先例实证）**：

| API 用法 | 来源确认 | 验证状态 |
|----------|---------|---------|
| `T.mma(A, B, C, init=True→False)` K-split 链式累加（legacy 无 unit_flag 形式） | 签名 customize.py:115-132（init 参数；C ptr `"w" if init is True else "rw"` :215）；链式先例①sparse_flash_mla:385-420（init=True→False 链，但带 unit_flag=0b10/0b11）；链式先例②example_gqa_sink_bwd_bhsd.py:436/471/475（gemm_v0 init=False L0C 累加 CompGEMM）；**legacy 形式（unit_flag=None）的 init=False 链为仓内首用** | ⚠ 前端源码确认 + 链式语义双先例（不同参数形式）→ **探针 P3 前置**（2×mma legacy 链 vs gemm_v0 单调用 vs golden 位级/阻塞口径三重比对）；备选 = unit_flag=0b10/0b11 sparse_flash_mla 逐字模式（切换仅参数面） |
| `T.mma(..., C=l0c_o[side,:,c0:c1])` C region 列段切片 N-split（主案 a） | customize.py:151-168（retrieve_shape 处理 BufferRegion：M,N = region extent）+ :191-209（retrieve_ptr region 分支：offset 按 buffer stride 计算 + access_ptr extent）；K==K_B 校验由 region extent 推导自动满足（:186-189）；列段边界 128 = 8×16 fractal 对齐 | ⚠ 前端源码确认；**C++ mma 模板 C-offset L0C fractal 寻址仓内无直接先例 → 探针 P2 前置**（N-split 2 段 vs 全 N 单段 vs golden 位级）；备选案 b = w3 列段 FIX 写出（先例完备，切换零设计变更，§3.2） |
| `T.copy(L1[:, c0:c1], L0B, transpose=True)` L1 列段 transpose 载入 | copy_op.py:261（transpose 参数，copy_l1_to_l0 专用）；**先例 sparse_flash_mla:376-382/403-409/441-447/467-473（kv_ring[s,:,0:KL0]/[KL0:2*KL0] 列段 + transpose=True，配 real_k/real_n 运行期参数）**——real_k/real_n 为其 varlen 语义所需，本算子 chunk 宽静态 128 不需要（省参 = 差分面更小） | ✅ 源码 + 先例；**× NZ annotation 组合 = 探针 P1**（annotation 开/关两态位级；关态先例 = sparse_flash_mla 全文无 annotate_layout 亦 CI 在案 → 应急预案 = 去 k_l1 NZ annotation，性能降级不换语义） |
| `T.copy(L1[:, c0:c1], L0A)` L1 列段直载（q chunk / v chunk） | 先例 sparse_flash_mla:372-375（q_l1[0,:,0:KL0]→p_l0a[0]）/:399-402/:632-641（p_l1 列段 + real_k） | ✅ 源码 + 先例（ZN annotation × 列段组合并入探针 P1 矩阵） |
| `T.copy(L0C, ws4d[cid,i,:,c0:c1])` L0C→GM 列段写出（备选案 b） | 先例 sparse_flash_mla:673-681（workspace_o 列段 + unit_flag=0b11；不带 unit_flag 的常规列段 = 同一 realDstN 机制）；copy_l0c_to_gm 模板 common.h:187-213 + realDstN/realTailM/realTailN src/op/ascend.cc:531-543（**#R6-1 gate 探针 8/8 整数精确 BITEXACT 实证运行期形参原生支持**，§S6-3-1——该实证为行切片维度，列段维度 = realDstN 同机制） | ✅ 源码 + 先例 + 本算子实测（相邻机制）；备选案 b 启用时探针 P5 兜底（列段 FIX 位级） |
| event id 容量（MTE1↔M pair 用 0-5 共 6 个） | Ascend 硬件 EVENT_ID 每 pipe pair 0-7（8 槽）；仓内佐证 sparse_flash_mla P_EV=7（:172，mte1-mte2 pair 用到上界）；gemm_v0 内部占用 L0AB_EVENT=0（common.h:1183，MTE2_MTE1/FIX_M/M_MTE1 三 pair :1306-1311）**仅存在于兜底域 prim_func**——Form B-mma 域无 gemm_v0 调用，全 id 空间自由无冲突 | ✅ 硬件约束 + 先例佐证 + 域隔离；静态自检项（§7.2-2a 表头） |
| `T.mma` 高阶 buffer 前导维切片 `l0a_s[c,:,:]`/`l0c_o[side,:,...]`（c/side 定值） | customize.py:116 docstring（"higher-order (leading dims must be 1)"指整体 buffer 前导维；切片后 region 前导维已消去）；本算子 Form A `l0a_s[side,:,:]`（side=i%2 运行期 TIR 定值）已实现且位级验证在案；c∈{0,1} 为编译期 Python int（更弱要求） | ✅ 本算子实测 + 源码 |
| 禁用项（rev4 新增） | **Form B-mma prim_func 内禁用 T.gemm_v0**（防御 event id 0 占用冲突 + 形态纯度）；自检 grep：form_b_mma 分支源码无 `gemm_v0`、无 `barrier_all`、无 `threads=`、无虚拟通道拷贝 | 🚫 不使用 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查（rev4 全项重跑——硬点 5 全套）

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | 涉及（逻辑网格 B×N×S_blocks） | 一维 `T.Kernel(core_num, is_npu=True)` + Fixed Core 任务循环内 task 整除分解（继承 #5，零变化） |
| threads 参数限制（仅 1/2；threads=2 AscendVidReduction×BSHD 4D region bug） | **不涉及（Expert 无 threads 参数）** | R8 护栏继承（§2.2-c 五点证据链）：全文禁 `threads=`（§8.6-1 grep 自检）；vid 二分源码显式（[bM//2] buffer + vid*half_M 偏移）无自动折半 pass 介入点；rev4 无模式变更 → R8 复检结论直接继承（Expert 不触发） |
| 动态循环边界不支持（tensor 值边界） | 涉及（S/S_kv 运行期 + causal 裁剪） | B/S/S_kv/N/D JIT 编译期参数；循环边界 = 运行期 TIR 标量表达式（eff_num_outer/batch_iters/num_tb/batch_tasks，由编译期 shape + cid/task 标量算术派生，**非 tensor 值**）——Expert 先例 mha_sink:270-278 CI 验证继承；**rev4 K/N-split = 编译期展开常量 2 步（D=256 静态封闭），零新增动态边界形态** |
| 流水线不支持动态边界 | 涉及 | **T.Pipelined 全文禁用**（R9 继承，硬点 5）；流水 = NUM_STAGES 批量结构 + 手动 side=i%2 / chunk c ping-pong（Form A 同形态） |
| 部分 GPU API 不可用 | 不涉及 | 全部 API 源码/先例实证（§3.4 两表） |
| L0C 容量上限（128KB；保守元素规则 M×N≤16384 fp32） | 涉及 | P3-mma 主案 96KB（l0c_s [2,32,128]=32 + l0c_o [2,32,256]=64）/ 备选案 64KB ✅；元素规则：l0c_s 单槽 32×128=4096 ✅、l0c_o 单槽 32×256=8192 ✅；全形态逐形状表 §4.5-c（Form A P1 96/P2 64、兜底域 16-48 继承） |
| L0A/L0B 容量（各 64KB） | 涉及 | **P3-mma L0B 64KB⚠ 共池顶格 = 唯一可行解**（分池 128KB 超限——§4.5-d 复核：同形 [2,128,128] 共池分时 + token 流延续 + Form A P2 已验证先例）；L0A 32KB ✅；顶格条目实编读数硬门（探针 P4，§4.5 判定制度）；Form A P1/P2 顶格继承（实测在案）；兜底域 gemm_v0 内部峰值 64KB⚠（session-2 #10/#11/#13 实测先例继承） |
| T.copy 不支持列方向 strided 切片 | **涉及（rev4 新增 L1 列段切片检测）** | 新增切片全部为 **L1 buffer 最内维（D）连续段**（`buf[:, c*128:(c+1)*128]`：行内连续 256B、行间恒定 gap=(N-1)*D 或 L1 pitch）——非"列维非最内维 + 行内 strided"禁令形态（禁令原文针对 `x[b, m:m+M, n:n+N, k:k+1]` K>1 型 4D 错位）；GM 侧 4D 行跨步切片继承 rev3（列维 = 最内维 D，rev2 verify_bshd_slice 位级验证）；ws 列段（备选案 b）= 4D 紧凑张量最内维连续段 ✅；先例 = sparse_flash_mla L1 列段族实测（§3.4） |
| Host 侧 reshape 隐式拷贝 / torch.nn.functional / aclnn / 输出侧切片+reshape | 不涉及 | host 零张量变换（§4.8 审计六项全 allow；rev4 增量 = 纯 Python bool 谓词）；不采用参考实现 host mask/block_table 预构建（§3.5.2/§4.8） |
| GEMM 非整除 | 涉及 | S/S_kv 尾块三重保障继承（valid extent + need_clear 清零 + 掩码钳位 + rows_vid 裁剪，§5.4）；**K/N-split 维（D=256）静态整除恒真**（host assert D∈{64,128,256} ∧ form_b_mma 谓词 D==256 ∧ KC=NC=128，§3.3 assert 固化）；bN 尾块 valid_n → k_l1/v_l1 尾行清零 → 尾列 scores=0 → 掩码 -inf → P=0（Form A 论证逐字适用，mma 全 N 执行无 n_actual 需求）；禁止 host padding |

#### 3.5.2 参考实现差异说明（rev3 八行表继承 + rev4 增量行）

rev3 §3.5.2 差异表逐字继承（布局 BHSD 参考→BSHD 原生行跨步/掩码 host 预构建→
kernel 内 pre-exp -inf 掩码链/w1 fp16 参考→fp32 主线/w3 fp16→fp32/tile 单配置→
六配置类/尾块整除断言→运行期 valid extent + rows_vid/softmax neg_sm+axpy→继承
Developer 算式/NUM_CORES 硬编码→设备查询/unit_flag 未用→v1 未用）。rev4 增量
（新最高结构模板 sparse_flash_mla 对照）：

| 差异项 | sparse_flash_mla（rev4 K/N-split 结构模板） | 本设计（Form B-mma） | 理由 |
|--------|-------------------------------------------|---------------------|------|
| K-split chunk 池组织 | p_l0a/v_l0b 前导维 = k-chunk（[2,G,BI]/[2,BI,BI]），跨 KV-slot（s0/s1 双 slot）外层轮转复用 | l0a_s 前导维 = k-chunk（[2,32,128]，任务级常驻——q kv-invariant）；l0b_pool 前导维 = chunk（[2,128,128]，GEMM1 kc / GEMM2 n-chunk **共池分时**）；跨 iter 复用 = side=i%2 ping-pong | 本算子 q 任务级常驻（sparse_flash_mla q 亦任务级 :372-375）；共池分时 = 本算子 Form A P2 已验证机制（L0B 64KB 硬限下唯一可行，§4.5-d）；side ping-pong = Form A 已验证结构 |
| unit_flag | 全程 0b10（链中）/0b11（链尾）+ copy 0b11 融合；窄 N（<40）特判 pipe_barrier("m") | v1 legacy（unit_flag=None → C++ unitFlag=0 每 mma 独立完结）；0b10/0b11 = Stage 3 候选 §10.3-② | 最小必要改动 + 与 Form A 现行参数形式一致（差分面最小、位级归因最清晰）；legacy init=False 链正确性 = 探针 P3 位级兜底；窄 N 特判坑（pipe_barrier("m")）在 v1 结构下不存在（N=bN=128 恒定） |
| N-split 输出组织 | cL0 窄池 [2,G,BI] + workspace_o 列段 FIX 写出（PV_NT=4 段 nl 循环，运行期段数） | 主案 a：l0c_o 全宽 [2,32,256] + mma C region 列段 ×2 + FIX 全块 ×1；备选案 b = 参考实现同构（窄池 + w3 列段 FIX ×2，编译期 2 段） | 主案 token 最简（每 iter SIG_L0C 一轮，Form A 同构）+ FIX 段数减半；C region 切片无仓内先例 → P2 探针门 + 备选 b 完整预案（先例完备，切换零设计变更）；本算子 N-split 段数 = 2 编译期常量（sparse_flash_mla 为运行期 nl 循环，更复杂形态已 CI 在案） |
| real_k/real_n | 运行期 varlen 边界必需（:380-381/:635/:640） | 不使用（chunk 宽 128 编译期静态；bN 尾块 = L1 清零 + 掩码钳位） | 本算子尾块语义已由三重保障封闭（继承）；少一组运行期参数 = 差分面更小 |
| workspace 数量 | 6 个（s/p/o/acc/qk + Output/LSE out_idx） | 3 个（w1/w2/w3，继承） | 本算子 CV 协议不变（emit_v 零改动前提）；S2 spill 方案（需第 4 ws）已论证降级（§0.5 论证二） |

#### 3.5.3 本项目同类实现参考（Phase 2 强制步骤 0 检索结果，rev4 更新）

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/mha/mha.py`（终态 bbfcbcaf，**rev4 开工基线**） | **最高（自身继承）** | Form A = T.mma 手动全流水已验证模板（L0 双槽 side ping-pong / SIG_L0AB/L0C token 收支 / init-destroy 闭环 / `if i < 2` 批首预载）；Form B-gemm_v0 = 兜底域逐字保留载体；emit_v 共享 emitter（零改动前提）；#R3-1/#R5-1 单块任务批处理协议（Form B-mma 单块分支镜像源，含 Q per-slot 载入 + SIG_Q_L1 每槽 net 0 收支）；#R6-1 parse-time gate 机制（`if not form_a:` 保 IR 先例） |
| `examples/deepseek_v4/sparse_flash_mla/sparse_flash_mla_kernel.py` | **最高（rev4 新增，K/N-split 结构模板）** | D2=256 K-split 2 步 init=True→False 链（:385-420）；L1 列段 + transpose 载入（:376-382）；v_l0b [2,128,128] L0B 64KB 顶格（:218）；L0AB_EV0/EV1 双 chunk token + init 预置（:174-175/:243-244/:383-423）；N-split 列段 FIX 写出（:673-681）；k_actual/n_actual/unit_flag 全参数用法（:385-483/:646-653）；cL0 cs 槽轮转 GEMM1/GEMM2 分时共池（:216）；窄 N pipe_barrier("m") 特判警示（:395-396） |
| `examples/mha_sink_fwd_bhsd/mha_sink_fwd_bhsd.py` | 最高（Form A 结构模板，继承） | Expert CV 批量流水完整骨架（rev3 §3.5.3 参考点全部有效：PASS_CONFIGS 全 False :38-43 / (cid,vid) :187 / L0 双槽+T.mma :206-208/307/351 / SEM-SIG 体系 :143-173 / init-destroy 闭环 :241-250/367-375/379-386/625-629 / NUM_STAGES 批量+余数 :278-280 / causal 裁剪 C/V 同式 :268-271/394-400 / deferred rescale :530-560 / ZN-NZ :196-203 / io_buf 双槽+SIG_S_HALF :443-457/517-528 / vid 半区写回 :614-622） |
| `examples/gqa_fwd_varlen/gqa_fwd_varlen.py` | 高（Form A 源头，继承） | mha_sink 结构所自出；num_stages 8+8 / cross_interval=1 参数实证（iter3）；io_buf 双缓冲 +16KB UB 换 MTE2/V 全重叠（iter4）；**block_N=256 的 L0B 128KB>64KB 否决先例（:112-114）——与 rev4 §4.5-e decode mma 化分池否决推导互证** |
| `examples/flash_attention/flash_attn_bhsd.py` | 高（Form B-gemm_v0 模板，继承） | gemm_v0 在 T.Scope("C") 内可用性实证（:93/107）；set/wait_cross_flag 基础协议（兜底域依据不变） |
| `examples/flash_attention/fa_opt/flash_attn_bhsd_expert_h16_d128.py` | 高（Form A 参数基准，继承） | num_stages/cross_interval 参数化入口 + workspace [NUM_CORES,num_stages,...] 结构 |
| `examples/attention_sink/example_gqa_sink_bwd_bhsd/example_gqa_sink_bwd_bhsd.py` | 中（rev4 升级引用） | **gemm_v0 init=False L0C 链式累加先例**（:436/:471/:475 CompGEMM via L0C accumulation，:37 docstring 明示）——链式累加语义的 gemm_v0 侧佐证 |
| `examples/xattention/xattention.py` | 中（rev4 新增旁证） | T.mma 多段调用 + L0 池组织旁证（:411-496/:1018-1064） |
| `examples/mha/history_version/mha_impl_developer_final_6505.py` | 语义基准（继承） | 12 项算法优化权威载体；**锚-A 对照实现**（rev4 守卫域收窄为 Form A 域 L0 #1-9，§9.2） |
| `examples/mha/history_version/mha_impl_r3_expert_final_7170.py` | **锚-B 对照实现 + 回退锚（rev4 新增）** | rev3 终态（bbfcbcaf）：Form B-mma 域位级诊断锚比对基线（§9.2 锚-B）；§12 回退条款回滚目标（md5 已核验 = 当前 mha.py） |
| `examples/developer_mode/flash_attn_bshd_developer.py` | 低（Developer 对照，继承） | 模式对照与历史回退路径参考 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量（继承 rev3/rev2，不变）

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| query | (B, S, N, D) | float16（代表性；全集见 §9.3） | BSHD 布局，contiguous，有限实数 |
| key | (B, S_kv, N, D) | 同 query | B/N/D 与 query 严格相等 |
| value | (B, S_kv, N, D) | 同 query | 同上 |
| scaleValue | 标量 float（host 参数，非 tensor） | — | 默认 -1.0；≤0 → 自动 1/√D；JIT 编译期常量 |
| is_causal | 标量 bool（host 参数，非 tensor） | — | 默认 False；True 时要求 S ≤ S_kv；JIT 编译期常量 |

支持范围（host 断言强制，不变）：B∈[1,128]，S∈[1,2048]，S_kv∈[1,4096]，N∈[1,64]，
D∈{64,128,256}（64 对齐、静态封闭）；dtype ∈ {float16, bfloat16}（三输入一致）。

### 4.2 输出张量（继承，不变）

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| Output | (B, S, N, D) | 同 query | out_idx=[3] 框架分配；contiguous（BSHD 原生写回，vid 分行 rows_vid 裁剪，无 host 后处理） |

### 4.3 中间缓冲区（Expert 显式声明制；rev4 = rev3 表继承 + Form B-mma L0 池新增）

**L1（C 侧，T.alloc_L1）**：四 buffer 声明与 rev3 §4.3 逐字一致（q_l1 (bM,D) /
k_l1 (bN,D) / v_l1 (bN,D) / p_l1 (bM,bN)；Form A 与 **Form B-mma** 均 annotate
ZN/NZ（q ZN / k NZ / p ZN / v ZN）；Form B-gemm_v0 不标注）。P3-mma 尺寸 = rev3
P3 原值：q [32,256] / k [128,256] / v [128,256] / p [32,128]。

**L0（Form A）**：rev3 §4.3 表逐字继承（l0a_s/l0b_s/l0c_s + P1 分池 l0a_o/l0b_o/
l0c_o；P2（D==bN==128）共池互换）。

**L0（Form B-mma 新增显式池；KC=NC=128，bM=32，bN=128，D=256）**：

| Buffer | Shape | dtype | 大小 | 用途 |
|--------|-------|-------|------|------|
| l0a_s | (2, 32, KC=128) | dt | 16KB | GEMM1 A（Q 两 k-chunk，**任务级常驻**——q kv-invariant；generic 分支任务头预载、单块批分支每槽重载；SIG_L0AS+0/+1） |
| l0a_o | (2, 32, bN=128) | dt | 16KB | GEMM2 A（P，side=i%2 ping-pong；SIG_L0AO+0/+1） |
| l0b_pool | (2, 128, 128) | dt | **64KB ⚠ 顶格** | **共池**：GEMM1 B=Kᵀ kc-chunk（transpose 载入）/ GEMM2 B=V n-chunk（直载）——同形分时（§4.5-d 复核）；SIG_L0KC+0/+1 |
| l0c_s | (2, 32, bN=128) | fp32 | 32KB | GEMM1 输出 scores（side ping-pong；K-split 链式累加目标 c=0 init=True → c=1 init=False） |
| l0c_o | (2, 32, D=256) | fp32 | 64KB | GEMM2 输出 PV 全宽（主案 a：n-chunk 各写列段 [side,:,c*NC:(c+1)*NC]；**备选案 b 改 (2,32,NC=128) = 32KB**） |

**L0C（Form B-gemm_v0 兜底域）**：rev3 表逐字继承（acc_s_l0c (bM,bN) / acc_o_l0c
(bM,D) 单槽；gemm_v0 内部按 kL0Size 申请 L0A/L0B）。

**UB（V 侧，T.alloc_ub）**：rev3 §4.3 表逐字继承（io_buf/io3/work_ub/buf_2d/rt_2d/
r_2d/mask_2d/col_pos/row_thresh/rr_ub/acc_s_half/acc_o/out_half/m_i/m_i_prev/
sumexp/r_factors/sumexp_is + 别名规则 io3_is_alias=(block_N==D)、r_2d 别名——
**emit_v 零改动 ⇒ buffer 集/形状/别名全部不变**；P3-mma 行 = rev3 P3 行，hM=16）。

**GM workspace（框架注入，workspace_idx=[4,5,6]）**：三 ws 形状/dtype/用途逐字
继承 rev3 §4.3（w1 (core_num,NUM_STAGES,bM,bN) fp32 scores / w2 同形 dt P /
w3 (core_num,NUM_STAGES,bM,D) fp32 PV）——**Form B-mma 不新增 workspace**（备选
案 b 的 w3 列段写出 = w3 本体 region 切片，非新 buffer；S2 spill 方案已降级
§0.5，不引入 w4）。

### 4.4 内存搬运路径（层级审计；rev4 增量 = Form B-mma 段）

```
[Form A / Form B-gemm_v0 路径 = rev3 §4.4 逐字继承]
[Form B-mma（P3-mma）路径]
GM[Q] ──MTE2(行跨步,valid_m)──► L1[q_l1 32,256 (ZN)]
L1[q_l1 列段 c] ──MTE1(直载)──► L0A[l0a_s[c]]               （任务级 ×2；单块批分支每槽 ×2）
GM[K] ──MTE2(行跨步,valid_n,批内逐块)──► L1[k_l1 128,256 (NZ)]
L1[k_l1 列段 c] ──MTE1(transpose=True)──► L0B[l0b_pool[c]]   （每 iter ×2 chunk）
L0A+L0B ──M(mma init=True→False 链)──► L0C[l0c_s[side]] fp32
L0C ──FIX──► GM[workspace_1[cid,i]] ──MTE2(vid 半区)──► UB[io_buf] ─► work_ub（V 段继承）
GM[workspace_2[cid,i]] ──MTE2──► L1[p_l1] ──MTE1──► L0A[l0a_o[side]]
GM[V] ──MTE2(行跨步,valid_n)──► L1[v_l1 (ZN)]
L1[v_l1 列段 c] ──MTE1(直载)──► L0B[l0b_pool[c]]              （与 GEMM1 共池分时）
L0A+L0B ──M(mma ×2，n-chunk 各 init=True → l0c_o 列段)──► L0C[l0c_o[side]]
L0C ──FIX(全块 [32,256])──► GM[workspace_3[cid,i]] ──MTE2(vid 半区)──► UB[io3]（V 段继承）
UB[acc_o]（终归一）──V(cast)──► out_half ──MTE3(行跨步,rows_vid,运行期 if 守卫)──► GM[Output]
```

全部路径符合硬件内存层级：GM↔L1↔L0A/L0B→L0C→(FIX)GM→(MTE2)UB→(MTE3)GM；
L1→L0 列段切片为最内维连续段（§3.5.1 检测行 + sparse_flash_mla 先例）；无 GM→L0
直搬；L0C 目的地仅 workspace_*（虚拟通道禁令 §8.6-3 grep 自检继承）；BSHD 行跨步
合法性继承 rev2 §4.4（extents [1,valid,1,D]，列维最内维，DataCopyPad 硬件模型）。

### 4.5 内存预算（rev4 重写：逐形状 L0A/L0B/L0C 预算表 + decode mma 化不可行推导——硬点 1）

> **判定制度继承（session-2 教训反向固化）**：Expert 下 MEMORY_PLANNING=False，
> planner 无 liveness 合并（ascend_memory_planning.cc:59-92），naive 声明和≈实配；
> 对齐 padding（AlignUp）可能小额附加 → **全部 ≥90% 占用（⚠）条目 Stage 2 强制
> 实编 GetWithOffset/编译探针读数复核（探针 P4），MEMORY_PLANNING 实编读数口径
> 优于手算**。硬限（实测口径继承 perf_tuning 会话 2）：L1 511.75KB、UB 191.75KB
> （每 AIV）、L0C 128KB、L0A 64KB、L0B 64KB。

**（a）L1（C 侧共享，单份）**

| 配置类 | q_l1 | k_l1 | v_l1 | p_l1 | 合计 / 511.75KB |
|--------|------|------|------|------|------------------|
| P1 prefill D=64（bM64,bN128,Form A） | 8KB | 16KB | 16KB | 16KB | 56KB ✅（继承） |
| P2 prefill D=128（bM64,bN128,Form A） | 16KB | 32KB | 32KB | 16KB | 96KB ✅（继承） |
| **P3-mma prefill D=256（bM32,bN128,Form B-mma）** | 16KB | 64KB | 64KB | 8KB | **152KB ✅**（= rev3 P3 同值——形态变更不影响 L1 声明） |
| D1 decode bN=512 D≤128（bM16，兜底域） | 4KB | 128KB | 128KB | 16KB | 276KB ✅（继承，Developer 终态同值实测先例） |
| D2 decode bN=256（D≤128 / D=256，兜底域） | 4/8KB | 64/128KB | 64/128KB | 8KB | 140KB / **272KB ✅（D=256 decode L1 最大占用行）** |
| D3 decode bN=128 D≤128（兜底域） | 4KB | 32KB | 32KB | 4KB | 72KB ✅（继承） |
| ~~decode bN=512 D=256~~ | — | — | — | — | 536KB ✗ **排除继承**（>511.75KB 硬限，session-2 #13 判决） |

**（b）UB（每 AIV，vid 半区 hM 行）**：rev3 §4.5 UB 表六行逐字继承（P1 ≈104KB /
P2 ≈116KB / P3 ≈79KB / D1 ≈98KB / D2 ≈53-61KB / D3 ≈30KB，全部 ✅ ≤191.75KB）
——emit_v 零改动 ⇒ UB 声明集不变；**P3 行即 P3-mma 行**（hM=16，余量 112KB）。

**（c）L0 逐形状预算总表（rev4 核心新增；单位 KB；⚠=顶格须实编复核 P4）**

| 形态/配置 | L0A / 64KB | L0B / 64KB | L0C / 128KB | 判定 |
|-----------|-----------|-----------|-------------|------|
| Form A P2（bM64,bN128,D128；共池互换） | l0a[2,64,128]=32 | l0b[2,128,128]=64 ⚠ | l0c[2,64,128]=64 | ✅ 继承（本算子位级验证 + mha_sink/gqa 同尺寸 CI 先例） |
| Form A P1（bM64,bN128,D64；分池） | l0a_s[2,64,64]=16 + l0a_o[2,64,128]=32 → 48 | l0b_s[2,64,128]=32 + l0b_o[2,128,64]=32 → 64 ⚠ | l0c_s[2,64,128]=64 + l0c_o[2,64,64]=32 → 96 | ✅ 继承 |
| **Form B-mma P3-mma 主案 a（bM32,bN128,D256）** | l0a_s[2,32,128]=16 + l0a_o[2,32,128]=16 → **32** | **l0b_pool[2,128,128]=64 ⚠ 共池顶格（唯一可行解，(d)）** | l0c_s[2,32,128]=32 + l0c_o[2,32,256]=64 → **96** | ✅（L0B/L0C ⚠ 实编复核 P4；先例 = P1 L0C 96KB 实配在案 + P2 L0B 64KB 共池在案 + sparse_flash_mla v_l0b 同尺寸） |
| Form B-mma P3-mma 备选 b | 32 | 64 ⚠ | 32+32=**64** | ✅（P2 探针失败切换，L0C 余量更大） |
| Form B-gemm_v0 D1（bM16,bN512,kL0=32） | gemm_v0 内部管理 | 内部峰值 512×32×2B×2槽=64 ⚠（session-2 #13 实测） | acc_s[16,512]=32 + acc_o[16,128]=8 → 40 | ✅ 继承 |
| Form B-gemm_v0 D2（bN256；kL0=128/64） | 内部 | D≤128: 256×128×2=64 ⚠（#10a，kNumSteps=1）/ D=256: 256×64×2×2=64 ⚠（#10b） | acc_s 16 + acc_o 8/16 → 24/32 | ✅ 继承 |
| Form B-gemm_v0 D3（bN128,kL0=128） | 内部 | 128×128×2×2=64 ⚠ | ≤16 | ✅ 继承 |

L0C 保守元素规则复核（M×N ≤ 16384 fp32）：l0c_s 32×128=4096 ✅；l0c_o 32×256=
8192 ✅（备选 b：32×128=4096 ✅）；Form A/兜底域继承 ✅（acc_s 16×512=8192）。

**（d）L0B 共池复核（§S4-5 明示硬门："l0b_s/l0b_o 共池预算（64KB 顶格 ⚠ 需分池复核）"）**

- **分池不可行（硬限否决）**：GEMM1 B 双槽需求 = Kᵀ chunk [2, KC=128, bN=128]
  fp16 = 64KB；GEMM2 B 双槽需求 = V n-chunk [2, bN=128, NC=128] fp16 = 64KB；
  分池合计 **128KB > L0B 64KB 硬限** ✗。单槽变体（各 [1,128,128]=32KB 分池合计
  64KB）牺牲 chunk ping-pong（MTE1(c+1) ∥ M(c) 重叠消失）= 回到 gemm_v0 等效
  串行结构，S1 收益归零 → 不采纳。
- **共池为唯一可行解**：两需求**恰好同形** [2,128,128]（KC=NC=bN=128——D=256 ∧
  bN=128 组合专属的巧合）。合法性三依据：
  1. **相位分时**：每外层批内 GEMM1 批（全部 batch_iters）先行、GEMM2 批后行
     （rev3 批结构继承），批内无交叉使用；跨批边界由 SEM_WS1_V2C/SEM_WS3_V2C
     协议自然分隔（GEMM2 批末的 l0b_pool 消费完成于下一批 GEMM1 首装载之前，
     SIG_L0KC token 链保证）。
  2. **token 流延续**：SIG_L0KC+0/+1 为流式信号量（无相位概念），GEMM2 首个
     wait 自然消费 GEMM1 末次 set——跨相位收支恒等（§7.2-2a 全表核算）。
  3. **已验证先例**：本算子 Form A P2 即 l0b_s/l0b_o 共池（mha.py:576-578
     `l0b_o = l0b_s if block_N == D`：同一物理池既接 k_l1 transpose 载入又接
     v_l1 直载），位级验证 + 四轮调优在案；mha_sink 同构；sparse_flash_mla
     v_l0b [2,BI=128,BI=128] 同尺寸同组织。
- **顶格复核**：64KB = L0B 上限整值（零余量）→ **探针 P4 实编 GetWithOffset
  读数强制项**（若存在对齐 padding 附加即编译失败/读数超限——失败即暴露，无静默
  风险；P2 同尺寸实配先例 = 强证据，预期通过）。
- **L0A/L0C 不共池理由**：L0A 32KB（余量 32KB）/L0C 主案 96KB（余量 32KB）预算
  充足；l0a_s（任务级生命周期）与 l0a_o（per-iter 生命周期）共池会耦合 SIG_L0AS/
  SIG_L0AO 两套 token 流——无预算压力不做该耦合（最小必要改动）；l0c_s（GEMM1
  链式目标）与 l0c_o（GEMM2 列段目标）异形（[2,32,128] vs [2,32,256]）无共池形态。

**（e）decode 形状 mma 化 K-split 分步设计与不可行推导（任务硬点 1 指定项；支撑 §5.2 三分派兜底域决策）**

decode（S≤16，bM=16，hM=8）若 mma 化，逐形状 L0B 需求推导。**前提**：双槽
ping-pong 为 Form A/S1 收益结构的必要条件（单槽退化 = MTE1 与 M 串行 = gemm_v0
等效排空结构，收益归零，不列为可行解）；共池要求 GEMM1 B-chunk 与 GEMM2
B-chunk **同形**（异构 view 共池无仓内先例，fractal 排布不同 → 不可行）：

| decode 配置 | GEMM1 B=Kᵀ [D(k), bN(n)] 双槽需求 | GEMM2 B=V [bN(k), D(n)] 双槽需求 | 共池可能 | 结论 |
|-------------|-----------------------------------|----------------------------------|----------|------|
| D1 bN=512, D=128（case10/11/12/13/16） | K-split kc=32：[2, 32, 512]=64KB ⚠（4-step 链，l0a_s 全 K [2,16,128] 列段切片或 chunk 池 [4,16,32]=8KB） | K-split kc=128：[2, 128, 128]=64KB ⚠（4-step，bN=512/128） | **异形**（[2,32,512] vs [2,128,128]）→ 共池 ✗；分池 128KB > 64KB ✗ | **预算不可行** |
| D2 bN=256, D≤128（case17） | kc=64：[2, 64, 256]=64KB ⚠（2-step，D=128/64） | kc=128：[2, 128, 128]=64KB ⚠（2-step，bN=256/128） | **异形**（[2,64,256] vs [2,128,128]）✗ | **预算不可行** |
| D2' bN=256, D=256（case15/18） | kc=64：[2, 64, 256]=64KB ⚠（4-step，D=256/64） | K-split×N-split：[2, 128, 128]=64KB ⚠（2kc×2nc=4-step） | **异形** ✗ | **预算不可行** |
| D3 bN=128, D≤128（case14；Skv≤128 单块域） | 全 K 单步：[2, 128, 128]=64KB ⚠（D=128）/ [2,128,64]=32KB（D=64） | 全 K 单步：[2, 128, 128]=64KB ⚠（D=128）/ [2,128,64]=32KB（D=64） | D=128：**同形 ✓ 可共池**（= P3-mma 机制）；D=64 同形 ✓ | **预算可行但无收益空间**：D3 域 case14 1.394x / case17 1.143x 已高位达标（单块批处理 + barrier 纯化红利已兑现，§S4-1/§S6-7）；多块 D3 仅 S_kv∈(128,256] 窄带；decode 不回退硬约束下动 D3 = 纯风险零预期收益 → **不实施** |

- **汇总**：decode 主力形状（bN=512/256，覆盖 case10/11/12/13/15/16/17/18 共
  8 case）mma 化在"双槽 + L0B 64KB"约束下**预算硬性不可行**（GEMM1 K-chunk 与
  GEMM2 B-chunk 异形，共池无从成立，分池超限；与 gqa_fwd_varlen block_N=256
  L0B 128KB 否决先例同一硬限逻辑）；唯一同形的 D3 窄域无收益空间。
- **叠加增益论证（实测归因，非预测性否决）**：decode 多块 wall = **每槽 MTE2
  带宽**（§S4-3-3 实测：case11/15 ≈5.3-5.9us/slot，K+V 256-272KB；#R3-1 后协议
  税已坍缩至 ~0.2-0.3us/轮）——mma 化消除的是 M 管每-step 排空（prefill D256 的
  scalar_mte1_stall 78% 墙），对 MTE2 带宽墙无杠杆。decode 单块域（case14/17/20 类）
  已经 #R3-1/#R5-1/#R6-1 三轮收割（1.394/1.143/2.079x）。
- **设计决策（任务硬点 6 授权路径）**：decode 全 D 保留 gemm_v0 形态（Form
  B-gemm_v0 兜底域，**源码逐字不变 → decode 全域 IR 逐字节守卫可证 → decode
  不回退硬约束由构造性保证**（优于事后验证））；分派谓词三分（§5.2）。取舍
  论证 = 预算不可行（硬证据）+ wall 归因无杠杆（实测证据）+ 构造性不回退。

**（f）GM workspace 足迹**：rev3 §4.5 表继承（P2 ≈15.7MB / P1 ≈12.6MB / P3
≈11.0MB / D1 ≈11.0MB；core_num 24 上界、NUM_STAGES=8 口径）——Form B-mma 复用
P3 行（ws 形状不变，≈11.0MB）；R14 缓解措施继承（[cid,stage] 私有索引核间零竞争 +
torch 缓存分配器复用 + msprof 归因 + NUM_STAGES 降档退路）。

### 4.6 动态轴定义（继承 rev3/rev2，不变 + rev4 增量声明）

B/S/S_kv/N 运行期可变维度采用 JIT 编译期参数方案（rev3 §4.6 表逐字继承：每
(B,S,S_kv,N,D,dtype,scale,causal) 组合编译一次，tilelang 缓存复用；D 静态封闭
{64,128,256}）。kernel 内派生量（core_num/q_tasks/r_tasks host 计算；task/bx/by/
bz/s0/valid_m/rows_vid 运行期标量算术；eff_kv_iters/eff_num_outer/batch_iters/
num_tb/batch_tasks 运行期 TIR 非 tensor 值边界；valid_n T.min 运行期切片）继承。
**rev4 增量声明**：KC/NC（chunk 宽）为编译期 Python 常量 128；K/N-split 步数 =
D/KC、D/NC = 编译期常量 2（Python for 展开）——**不新增任何运行期边界形态**。

### 4.7 JIT 配置（继承，不变）

```python
@tilelang.jit(
    out_idx=[3],                    # Output = main 第 4 参
    workspace_idx=[4, 5, 6],        # w1/w2/w3 框架分配注入（host 调用形式不变）
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,   # 手动 T.Scope("C"/"V")
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,      # 手动 SEM 0-5
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,         # 手动 SIG flag 矩阵
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,   # 显式 alloc_*（planner 仅编址）
    },
)
```

### 4.8 Host 侧 Buffer 操作审计（强制，rev4 复核）

```text
[HOST-METADATA-AUDIT]（rev4：五项继承 + 增量一项）
operation: query.shape/dtype、key.shape/value.shape 读取（校验与配置推导）
input_stride -> output_stride: 不产生新张量
shares_storage / same_data_ptr: N/A（只读元数据）
aclnn_or_physical_copy: false
result: allow（继承）

operation: Python 标量运算（sm_scale 解析、bM/bN/gemm1_kL0/core_num/q_tasks/
  r_tasks/block_num/kv_iters/single_kv_block/NUM_STAGES 推导、assert 校验）
input_stride -> output_stride: 无张量参与
shares_storage / same_data_ptr: N/A
aclnn_or_physical_copy: false
result: allow（继承）

operation: **rev4 增量——分派谓词 form_a/form_b_mma 推导（(S>16)∧(D≤128) /
  (S>16)∧(D==256)，纯 Python bool 运算；else 兜底）**
input_stride -> output_stride: 无张量参与
shares_storage / same_data_ptr: N/A
aclnn_or_physical_copy: false
result: allow（三分派谓词互斥穷尽，§5.6 DISPATCH-COVERAGE 审计）

operation: _cube_core_num()（torch_npu.npu.get_device_properties(0).cube_core_num
  设备属性查询；异常回退常量 20）
shares_storage / same_data_ptr: N/A（设备元数据查询，非数据操作）
aclnn_or_physical_copy: false
result: allow（继承；先例 lightning_indexer.py:14-21 + best-practices §NUM_CORES）

operation: str(query.dtype).removeprefix("torch.")（dtype 字符串化，JIT 参数）
aclnn_or_physical_copy: false（Python 字符串操作，非 tensor.to(dtype)）
result: allow（继承）

operation: kernel(query, key, value) 调用 + 直接 return 输出
input_stride -> output_stride: 输入原样传入；输出为 out_idx 新分配 contiguous
  [B,S,N,D]；workspace_1/2/3 框架按 workspace_idx 内部分配注入（非 host 张量操作）
shares_storage / same_data_ptr: 输出为框架新分配（合法：kernel 计算产物）
aclnn_or_physical_copy: false（无 host 侧 reshape/contiguous/切片/cat/pad/to/clone）
result: allow（继承）
```

结论：host 完整路径（kernel 前 + kernel 后）只含元数据读取、纯 Python 运算、设备
属性查询、kernel 调用与结果直返；**零物理化操作、零 aclnn 依赖**（继承）。不采用
参考实现的 host 预构建（mha_sink/gqa 的 build_mask、sparse_flash_mla 的 host
block_table 类操作一律不引入——§3.5.2）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 混合（Cube + Vector，CV 融合，Expert 显式 workspace + vid 二分）

**判定依据**（继承 rev3 §5.1）: 两次 matmul（QKᵀ、PV，Cube/L0 路径）夹一个多步
softmax（max/exp/sum/div，Vector/UB 路径）→ 融合算子（决策树"matmul +
element-wise → CV 融合"分支；test-design 综合类别 = Fusion，workspace 配置 = True）。

### 5.2 Block 划分（rev4：三档体系继承 + 三分派谓词——硬点 6 落点）

```python
# —— tile 三档体系（逐字继承 Developer 终态 mha.py:120-179 探针验证硬限解）——
block_M = 16 if S <= 16 else (32 if D == 256 else 64)
if S <= 16:
    block_N = 512 if (D <= 128 and S_kv > 256) else (256 if S_kv > 128 else 128)
else:
    block_N = 128
gemm1_kL0 = (32 if (S <= 16 and D <= 128 and S_kv > 256)
             else (64 if (S <= 16 and D == 256 and S_kv > 128) else 128))  # 仅兜底域使用

# —— rev4 三分派谓词（替换 rev3 二分 form_a；互斥且穷尽，§5.6 审计）——
form_a     = (S > 16) and (D <= 128)   # Form A: mma 全流水（prefill D≤128，不变）
form_b_mma = (S > 16) and (D == 256)   # Form B-mma: 手动 K/N-split mma 流水（prefill D=256，新增）
# else（S <= 16）: Form B-gemm_v0 兜底（decode 全 D，源码逐字不变）

# —— Expert 维度（继承）——
half_M = block_M // 2                  # vid 半区行数（P3-mma: 16）
NUM_STAGES = 8                         # 继承（#R5-4 12/16 暂缓维持：rev4 控制变量）
CROSS_INTERVAL = 1                     # 继承（#R3-4 判决实验零响应，不动）
KC = 128; NC = 128                     # Form B-mma chunk 宽（编译期常量；D=256 → 各 2 步）

# —— 任务网格（逐字继承）——
s_blocks  = ceildiv(S, block_M);  kv_iters = ceildiv(S_kv, block_N)   # 编译期
block_num = B * N * s_blocks
core_num  = min(block_num, _cube_core_num())     # Fixed Core（设备属性查询）
q_tasks, r_tasks = block_num // core_num, block_num % core_num
# task 分解（bx 最快维，继承）：bx = task % s_blocks; by = task//s_blocks % N;
#                               bz = task//s_blocks//N % B
```

**三分派谓词论证（rev4 核心决策）**：

1. **Form B-mma 收益域 = prefill D=256 专属**：case9 三证据链（§S4-3-2）证明该域
   wall = gemm_v0 每-step M 管排空（scalar_mte1_stall ≈78%）——手动 mma 流水
   （MTE1∥M∥FIX 三管重叠 + chunk 级精确 token，无 PipeBarrier<PIPE_M>）直接消除
   该串行链。预期结构 → case7 同族（Form A，0.46x 级）：case9 与 case7 per-iter
   C 侧 FLOPs 相同（GEMM1+GEMM2 = 2×32×128×256×2 = 2×64×128×128×2 = 4.2 MFlop）、
   总 iter 数相同（task 数同为 2048 × causal 平均 4.5 = 9216，~460 iter/核）、且
   case9 w1 [32,128] FIX 字节为 case7 [64,128] 之半 ⇒ 预期 case9 ≥ case7 水位
   （§S4-5 立项预期原文："case9 → case7 同族结构（0.46x 级），M1 case9 线可达"）。
2. **decode 域不 mma 化 = 预算硬否决 + wall 无杠杆 + 构造性不回退**（§4.5-e 推导表：
   bN=512/256 双槽分池 128KB > L0B 64KB 硬限、异形无共池解；decode wall = MTE2
   带宽实测归因 §S4-3-3）——decode 全 D 落兜底域（IR 逐字节不变 ⇒ case10 ≥0.90/
   case13 ≥0.69/case14 ≥0.99/case16 ≥0.52/case20 ≥0.82 硬约束**由构造满足**）。
3. **诚实声明（与立项表述的预期差异）**：立项范围表述"D256 族 case4/15/18 同受益"
   中，case15 (60,1,512,16,256)/case18 (60,2,512,16,256) 为 **decode** D256（S≤16）
   → 落兜底域，**rev4 不使其受益**。证据：其 wall 已被 §S4-3-3 点名归因为每槽
   MTE2 带宽（case15 ≈5.3-5.9us/slot），mma 化（M 管排空消除）对其无杠杆，且
   mma 化预算不可行（§4.5-e D2' 行）。实际受益域 = **case9/case4 + 哨兵
   perf_d256_bf16 + 支持域内全部 S>16∧D=256 形状**。
4. **bM=32（D=256 prefill）继承不动**：bM 加倍（64）使 l0c_s [2,64,128]=64KB、
   l0c_o [2,64,256]=128KB → L0C 合计 192KB > 128KB ✗；tile 体系为 Developer 两轮
   探针验证的硬限解，最小必要改动原则不重开搜索空间。
5. **NUM_STAGES/CROSS_INTERVAL 不动**：#R5-4 暂缓判定维持（爆炸半径 = 全 shape
   IR 变化失去逐字节守卫 + ws 足迹翻倍）；rev4 控制变量——S1 归因纯度优先。

### 5.3 约束分析（rev4 更新）

- **分形限制**（fp16/bf16：M≥16, K≥16, N≥16）：Form B-mma GEMM1 chunk mma
  M=bM=32 ✅ K=KC=128 ✅ N=bN=128 ✅；GEMM2 chunk mma M=32 ✅ K=bN=128 ✅
  N=NC=128 ✅（主案 a C region extent (32,128)；备选 b 同）。Form A（M=64 K=64/128
  N=128）/兜底域（gemm_v0 内部分片保障）继承 ✅。
- **对齐要求**：L1 列段切片边界 c*128（128 列 × 2B = 256B 段，fractal 16 列粒度
  ×8 对齐）✅；L0C 列段 offset 128（8×16 fractal）✅；GM 行片段 D×2B ∈ {128,256,
  512}B ≥32B 且 32B 整数倍 ✅；行 pitch N×D×2B 32B 整数倍 ✅；UB/L1 buffer 尺寸
  256B 整数倍 ✅；ws 切片行连续段继承 ✅；L0 fractal 16×16 粒度：全部 tile/chunk
  维度为 16 倍数（bM 16/32/64、bN 128/256/512、D 64/128/256、KC/NC 128）✅。
- **UB 容量**：六配置 30-116KB ≤ 191.75KB ✅（§4.5-b 继承；P3-mma = P3 行 ≈79KB，
  余量 112KB——为 Stage 3 候选（io_buf 加深/O 直写扩展）留空间）。
- **L0C 容量**（128KB 强制检测 `Σ slot×M×N×4B ≤ 131072`）：P3-mma 主案 96KB /
  备选 64KB ✅；Form A P1 96/P2 64 ✅；兜底域 16-48KB ✅（§4.5-c 全表）。保守
  元素规则 M×N≤16384：全形态单槽最大 8192 ✅。
- **L0A/L0B**（各 64KB）：P3-mma L0A 32KB ✅；**L0B 64KB ⚠ 共池顶格（唯一可行解，
  §4.5-d 复核 + P4 实编硬门）**；Form A P1/P2 顶格继承（实测先例在案）；兜底域
  gemm_v0 内部峰值 64KB⚠（session-2 实测先例继承）。
- **L1**（511.75KB）：最大 D1 276KB ✅；P3-mma 152KB ✅；decode D256 bN=512
  536KB ✗ 排除继承。
- **event id 容量**：MTE1↔M pair 用 SIG_L0KC 0,1 + SIG_L0AO 2,3 + SIG_L0AS 4,5
  = 6 ≤ 8 硬件上限 ✅；MTE2↔MTE1 pair 用 0-3 ✅；M↔FIX pair 用 0-1 ✅；V 侧
  pair 继承 ✅（§7.2 全矩阵）。
- **threads/vid**：Expert 帧 vid extent 2 结构性绑定（无 threads 参数；R8 复检不
  触发继承 §2.2-c）✅。**Kernel 维度**：一维 core_num ✅。

### 5.4 注意事项（非整除与尾块；rev3 §5.4 全节继承 + Form B-mma 增量声明）

rev3 §5.4 全节逐字继承：valid extent 五行表（Q 搬入 valid_m / K-V 搬入 valid_n /
softmax 域尾列钳位 / ws 整 bM 行交换 / Output 写回 rows_vid + 运行期 if 守卫）、
尾块三重保障论证（need_clear 确定性清零 + 尾列 -inf 掩码 + rows_vid 裁剪；行独立
性 ⇒ 有效行结果与整除情形逐位一致）、vid 二分不破坏行独立性论证、极小 shape
（S=1/S_kv=1/N=1）、R10 组合探针与应急预案。

**Form B-mma 增量声明（四条）**：

1. **K/N-split 维恒整除**：chunk 维 = D（256 = 2×KC = 2×NC）静态封闭——host
   assert D∈{64,128,256} ∧ form_b_mma 谓词 D==256 ⇒ 整除恒真（§3.3 assert 固化）；
   split 维**无尾块形态**（无 k_actual/n_actual 运行期需求）。
2. **bN 尾块（S_kv % 128 ≠ 0）语义继承 Form A**：k_l1/v_l1 尾行 need_clear 清零 →
   GEMM1 尾列 scores = 0（有限垃圾）→ V 侧尾列钳位掩码 -inf → P=0 → GEMM2 中
   0×0=0（v 尾行已清零，无 0×NaN 风险）——mma 全 N（128）执行，与 gemm_v0 兜底域/
   Form A 完全同构（行独立 + 掩码封闭论证逐字适用）。
3. **bM 尾块（S % 32 ≠ 0）**：valid_m 运行期 extent + q_l1 尾行清零 + V 侧写回
   rows_vid 裁剪（继承）；v1 无 FIX 行切片形态（#R6-1 O 直写不扩展至 mma 域，
   §0.4-b——Stage 3 扩展时才涉及 realDstN/realTailM，gate 探针 8/8 机制在案）。
4. **L1 列段切片与尾块正交**：列段 = D 维（恒满 256 无尾）；行维尾块由 need_clear
   处理——两维解耦，无交互形态（探针 P1 矩阵含尾块行 × 列段组合位级验证）。

### 5.5 数据搬运性能可行性（REORDER-COST，强制；rev4 更新）

本算子无物理布局重排（BSHD 原生进出，继承）。rev4 增量 = Form B-mma 的 L1→L0
列段载入与 chunk 步数变化，逐路径审计：

```text
[REORDER-COST]（rev4：三形态；Form A / Form B-gemm_v0 行 = rev3 审计逐字继承）
path: form_b_mma（prefill D=256，手动 K/N-split mma 流水；掩码 4 变体与
  single_kv_block 特化共用同一搬运结构）
gm_passes: Q×1 + Output×1 + K/V × s_blocks（flash 标准 KV 重读因子）+ ws w1/w2/w3
  每 KV 迭代各 1 写 1 读 —— 与 rev3 Form B（gemm_v0 域）**完全同量**（S1 只改
  L1→L0→L0C 片上执行体；GM 侧字节总量不变：w1 [32,128]fp32 / w2 [32,128]dt /
  w3 [32,256]fp32 形状不动）
dma_transactions: GM 侧每任务每迭代 ≈ 2·bN 行（K/V）+ 3 写 3 读 ws tile 级（继承）；
  片上 L1→L0 由 gemm_v0 内部 4 step/iter（GEMM1 2 k-step + GEMM2 2 内部 step，
  §S4-3-2 "4 step/iter" 读数）变为手动 4 chunk 载入/iter（GEMM1 2 kc + GEMM2 2 nc）
  —— transaction 数持平，**时序从每-step PipeBarrier<PIPE_M> 排空串行变为 chunk 级
  token 流水（MTE1(c+1) ∥ M(c) ∥ FIX(c-1) 重叠）**：这是 S1 的收益机制（时序重排，
  非字节削减——如实声明）
average_dma_bytes: L1→L0 chunk = 128×128×2B = 32KB/次（gemm_v0 内部 step 40-80KB
  同量级域内，§S4-3-2 读数）；GM 行片段 = D×2B ∈ {128,256,512}B ≥32B 对齐、无
  逐元素搬运；ws tile 段 = bN×4B/bN×2B 连续行段 ≥512B（继承）
gm_scalar_accesses: 0（全程块 DMA，继承）
address_div_mod_per_element: 0（地址增量切片静态推导；每任务 3 div + 2 mod 解码，
  C/V 双侧同式，继承）
active_cores / serial_tasks_per_core: core_num（20-24）/ ceil(block_num/core_num)
  ——case9: 2048 task/20 核 ≈ 102 task/核 × eff≈4.5 iter = ~460 iter/核（§S4-3-2
  口径一致）；Expert cv_1_2 每 cid = 1 AIC + 2 AIV（vector 双 AIV 分担继承）；
  C∥V 跨核重叠 = NUM_STAGES 批量流水（GEMM1 批(i+1) ∥ softmax 批(i) ∥ GEMM2
  批(i-1)，继承）
measured_kernel_path（锚点）: rev3 终态哨兵 6.0-7.4s ≤120s 门（§S4-4 实测）；
  rev4 Form B-mma 首测值待 Stage 2 产出（同门限复检）
largest_case_timeout_gate: S1-S4 kernel 路径门各 ≤120s（§9.4 继承 rev2-C4 结构：
  H2D→kernel(含首次编译)→D2H；golden 成本不计入）+ case 级 1800s 防挂死参考上限
result: pass（GM 流量与 rev3 持平、片上 transaction 持平、收益机制 = M 管串行链
  消除；实测锚定以 Stage 2 哨兵复检 + msprof op 归因为准——若 kernel 路径逼近门禁，
  触发 §6.3 降级旋钮）
```

case9 定量核算（rev3 §5.5 方法继承）：GM 流量 ≈ K/V 重读（2048 task × 平均 4.5
iter × 64KB ≈ 576MB）+ Q/Out ≈0.1GB + ws（9216 iter × (16+8+32)KB × 2(写+读)
≈ 1.0GB；P3-mma w1=16KB/w2=8KB/w3=32KB）≈ 1.7GB @≈1TB/s ≈ 1.7ms；算力
4·B·N·S·S_kv·D = 4×4×16×1024×1024×256 ≈ 0.27 TFlop（causal 裁剪后 ~0.15）
@50-150 TFLOPS ≈ 1-3ms——与现状 wall（case9 ≈1011-1042us，20 核并行）量级口径
自洽；mma 化不改变字节总量，收益 = §S4-3-2 归因的 78% scalar_mte1_stall 域压缩
（结构对照 = Form A case7 已运行在同族流水结构上）。

### 5.6 多路径分派审计（DISPATCH-COVERAGE，强制；rev4 三分派重审——R19）

```text
[DISPATCH-COVERAGE]（rev4：Expert 三形态 + vid 二分 + 批量流水）
supported_domain: Q/K/V dtype ∈ {float16, bfloat16}（三者一致）；
  shape：B∈[1,128], S∈[1,2048], S_kv∈[1,4096], N∈[1,64], D∈{64,128,256}；
  attr：scaleValue∈ℝ（≤0 自动 1/√D），is_causal∈{True,False}（True ⇒ S≤S_kv）；
  输入值域：任意有限实数
generic_fallback: 单一通用 host 入口 mha()（上述全域）；kernel 侧三形态并集 =
  全域：Form A（S>16∧D≤128）∪ Form B-mma（S>16∧D=256）∪ Form B-gemm_v0
  （S≤16 全 D）——谓词按 (S>16, D) 二维互斥穷尽（D 静态封闭 {64,128,256}，
  S>16 二分）；各形态内部 kv_iters≥2 通用 online softmax 路径 + single_kv_block
  特化路径兜底；分派全部位于 host 纯 Python 元数据层（编译期谓词），无输入落空
specializations:
  - predicate: form_a = (S>16) ∧ (D≤128) → Form A（mma 全流水，rev4 不变）
    fallback_on_miss: form_b_mma 谓词次判；再 miss → Form B-gemm_v0
    equivalence_evidence: 三形态共享 emit_v V-scope 源码与 workspace/SEM 协议
      （§3.3 emitter 注记）；C 侧差异仅 GEMM 执行体（mma 单步 / mma K-N-split 链 /
      gemm_v0 内部 L0），数学同一：fp32 L0C 累加、K 升序一致（Form B-mma 2×KC-128
      链 = gemm_v0 kL0Size=128 内部 2 k-step 同序 = Form A K=D≤128 单 mma 内部
      K-loop 同一 cube 数据通路；§9.3.3-6 论证 + 锚-B 诊断追踪）
  - predicate: form_b_mma = (S>16) ∧ (D==256) → Form B-mma（rev4 新增）
    fallback_on_miss: 不适用（谓词在 D 静态封闭集上穷尽；D=64/128 落 Form A，
      S≤16 落兜底域）
    equivalence_evidence: 同上（K-split 链式 = K 升序分片 fp32 累加，§S4-3-4-①
      同序位级稳定知识；N-split 列独立无累加依赖）；预算依据 §4.5-c/d；探针门
      P1-P4 前置（§9.6）；L0 命中 #10（单块分支）+ 哨兵 perf_d256_bf16（多块）
      + Stage 2 Form B-mma generic 探针 battery（§9.2 补充覆盖声明）
  - predicate: S ≤ 16 → Form B-gemm_v0 兜底 + bM=16 + bN 三档（512/256/128）+
    gemm1_kL0 三档（32/64/128）；S > 16 → bM = 32(D=256)/64(D≤128)，bN=128
    fallback_on_miss: 不适用（谓词在 S/D/S_kv 域上穷尽，逐字继承 Developer 终态
      已验证分派；rev4 仅把 S>16∧D=256 子域从 gemm_v0 执行体切换到 mma 执行体，
      tile 档位不变）
    equivalence_evidence: §4.5 预算表——仅资源占用与执行体不同，算法同一；
      兜底域 Developer 终态 12 项 probe + 会话 3-6 门禁实证记录（optimization_log）
  - predicate: (is_causal, S_kv % block_N == 0) → 掩码 4 变体：(F,T)无掩码链 /
    (F,F)仅尾列钳位 / (T,T)仅 causal / (T,F)causal+钳位（统一 offset_eff 阈值式；
    编译期 need_mask 开关 + 运行期全可见跳过 #4 叠加）
    fallback_on_miss: 不适用（2×2 编译期穷尽）
    equivalence_evidence: rev3 §5.6 逐字继承（V-scope 载体零改动 ⇒ 该审计不变）
  - predicate: kv_iters == 1（S_kv ≤ block_N）→ single_kv_block 预归一化特化 +
    任务批处理分支（#R3-1/#R5-1 协议；Form B-mma 域 = 协议镜像 + mma 本体）
    fallback_on_miss: kv_iters ≥ 2 → 通用批量 online softmax（phase1/phase2
      deferred rescale），谓词二分穷尽
    equivalence_evidence: rev3 §5.6 继承（单迭代 sumexp 即终值，预归一化与终归一化
      代数同值、恰与 golden 一次性 softmax 同构；Developer 实测 GOLDEN-BITEXACT
      on Skv=129/192/300 在案）；Form B-mma 单块分支算式次序逐字保持（emit_v 零改动）
  - predicate: scaleValue > 0 → s=scaleValue；否则 s=1/√D
    fallback_on_miss: 不适用（ℝ 全函数）
    equivalence_evidence: golden `if scaleValue <= 0: scaleValue = 1.0/(D**0.5)` 逐字一致
unsupported_inputs: dtype 三者不一致 / dtype∉{fp16,bf16} / D∉{64,128,256} /
  is_causal∧S>S_kv / 维度超范围 / 非 contiguous / 非有限值输入（NaN/Inf 不在声明
  值域，L2/Boundary 非阻塞探测）——全部由 host 元数据断言拒绝（不读数据），与
  supported_domain 无冲突
result: pass（覆盖命中映射：Form A→L0 #1-9 + 哨兵 perf_prefill_worst/perf_max_grid；
  **Form B-mma 单块→L0 #10**；**Form B-mma 多块→哨兵 perf_d256_bf16 + Stage 2
  探针 battery + L1 现有 D256 用例**（§9.2 补充覆盖声明）；Form B-gemm_v0 decode→
  L1 case10/13/14/16 类 + 哨兵 perf_decode_worst；bN 三档→L1/哨兵 decode 系；
  掩码 4 变体→(F,T):#1,3,5,7,10 / (T,T):#2,4,6,8,9 / (F,F),(T,F)→L1 S_kv%bN≠0
  现有用例；single_kv_block→#1-4,10 + L1 Skv≤bN 系；通用多块→#9 + L1/哨兵；
  vid 分行/rows_vid→L1 奇数 S 系 + R10 探针；scale 两分支→#1-9 auto / #10 显式）
```

分派只读取 shape/dtype/attr 元数据（纯 Python），不读取、不改动张量数据 ✅；每条
specialization 在 §9.2 L0 计划或现有 L1/哨兵套件中至少 1 个命中用例 ✅（Form B-mma
多块路径的 L0 无命中为**明示声明的既有格局延续**——L0 表以规则 shape 十用例为契约
固定集，多块 D256 由哨兵 + Stage 2 探针 battery 强制覆盖，§9.2/§9.6）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结（rev3 表继承 + Form B-mma 增量行）

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 任务网格 (B×N×S_blocks) | Fixed Core + 核内任务循环 | `T.Kernel(core_num, is_npu=True) as (cid, vid)` + `for t in T.serial(my_count)` | 一维网格（平台不支持三维 Kernel）；静态均衡分配（继承 #5）；Expert 帧 vid extent 2 结构性绑定（继承） |
| KV 外层批 (eff_num_outer) | 串行循环（运行期 TIR 边界） | `T.serial(eff_num_outer)` | causal 运行期裁剪 + 批量化（继承；C/V 双 scope 同式复制 → flag 配对平衡 §7.2-5 硬约束） |
| KV 内层批 (batch_iters) / 单块任务批 (num_tb/batch_tasks) | 串行循环（运行期 TIR 边界） | `T.serial(batch_iters)` / `T.serial(num_tb)` + `T.serial(batch_tasks)` | 尾批余数处理（继承）；批内 side=i%2 手动 ping-pong |
| 双缓冲轮转 | 编译期/运行期索引 | `side = i % 2`、`io_side = i % 2` | 手动多缓冲（T.Pipelined 禁用语境下的 Expert 标准形态，继承） |
| **Form B-mma K/N-split（rev4 新增）** | **编译期展开常量 2 步（非循环）** | Python `for c in (0, 1)` 构建期展开 / 宏体 c=0,1 顺序书写 | KC/NC=128、D=256 静态封闭 ⇒ 步数编译期常量；**零新增运行期边界形态**（§3.5.1）；chunk 间依赖由 SIG_L0KC token + init=True/False 链表达 |
| Form A GEMM1 Q 预载 | 批内条件 | `if i < 2: T.copy(q_l1, l0a_s[side,:,:])` | 继承（Form A 域不变）；Form B-mma 对应结构 = l0a_s 任务级预载（generic）/ 每槽重载（单块批，#R5-1 delta-b 镜像） |
| 掩码/softmax 行处理 | SIMD 2D tile op | T.tile.broadcast/compare/select/sub/exp/div + T.reduce_max/sum(dim=-1) | 继承（V-scope 零改动） |

**Python 级形态分支注记（rev3 继承 + 扩展为三分支）**：三形态在 `mha_kernel`
工厂内以 Python 级 if/elif/else 构建三个自包含 prim_func（V-scope 共享 emit_v
@T.macro emitter、C-scope 各自完整），host 按 (form_a, form_b_mma) 谓词分派——
规避 TVMScript parse-time if 跨块变量作用域限制（rev2-A1 教训；@T.macro 承载
T.serial 帧的机制已由终态实现实证，mha.py:17-18 docstring），三形态独立编译、
独立验证（§5.6）。**明确不采用**（继承）：T.Pipelined（任何 num_stages；R9 +
Expert 参考实现零使用）；T.Parallel（Expert 统一 T.tile.* 原语）；T.Persistent
（Fixed Core 任务循环已实现持久化语义）。

### 6.2 循环伪代码

rev3 §6.2 骨架逐字继承（C 侧 init → 任务循环 → eff_* 同式 → Q 载入 → 外层批
[GEMM1 批 → GEMM2 批] → destroy；V 侧 init → 任务循环 → 状态初始化 → row_thresh
hoist → 外层批 [phase1 softmax 批 → phase2 O 累加批] → 终归一 → rows_vid 写回 →
destroy）。**rev4 增量 = C 侧 GEMM 执行体按形态三分**：

```python
# C 侧每槽 GEMM 执行体（批循环内 [i] 槽位；side = i % 2）：
if form_a:            # 不变（rev3 §8.3 Form A 段逐字）
    <mma 单步双槽流水：K 载入 transpose → SIG_L0AB → mma(init=True) → SIG_L0C → FIX w1>
elif form_b_mma:      # rev4 新增（§3.3 GEMM1_BODY/GEMM2_BODY 宏、§8.3 详述）
    # GEMM1：K-split 2 chunk 链
    <k_l1 载入(SIG_K_L1)> 
    for c in (0, 1):                                    # 编译期展开
        <wait SIG_L0KC+c → copy k_l1 列段 c → l0b_pool[c] transpose
         → set/wait MTE1→M → [c==0: wait SIG_L0C+side]
         → mma(l0a_s[c], l0b_pool[c], l0c_s[side], init=(c==0))
         → set M→MTE1 SIG_L0KC+c → [c==1: set M→FIX SIG_L0C+side]>
    # （l0a_s 就绪由任务头/槽头预载段 wait(MTE1→M) 一次性确认，宏内零 L0AS）
    <wait M→FIX SIG_L0C+side → copy l0c_s[side] → w1[cid,i] → set FIX→M>
    # GEMM2：N-split 2 chunk（主案 a）
    <v_l1 载入(SIG_V_L1) + p_l1 载入(SIG_P_L1/SEM_WS2_V2C) + l0a_o 载入(SIG_L0AO+side)>
    for c in (0, 1):                                    # 编译期展开
        <wait SIG_L0KC+c → copy v_l1 列段 c → l0b_pool[c]（直载）
         → set/wait MTE1→M → wait SIG_L0AO+side → [c==0: wait SIG_L0C+side]
         → mma(l0a_o[side], l0b_pool[c], l0c_o[side,:,c*NC:(c+1)*NC], init=True)
         → set M→MTE1 SIG_L0KC+c → [c==1: set M→FIX SIG_L0C+side]>
    <wait M→FIX SIG_L0C+side → copy l0c_o[side] 全块 → w3[cid,i] → set FIX→M>
else:                 # Form B-gemm_v0 兜底（不变，rev3 §8.3 Form B 段逐字，#R4-1 无 barrier）
    <copy K → gemm_v0(q_l1,k_l1,acc_s_l0c,transpose_B=True,init=True,kL0Size) → copy w1>
    <copy V/p_l1 → gemm_v0(p_l1,v_l1,acc_o_l0c,init=True) → copy w3>
```

V 侧伪代码 = rev3 §6.2 V 段逐字（emit_v 零改动）。

### 6.3 流水线与预取（rev4 核心性能结构）

- **Form B-mma chunk 级三管流水（rev4 新增，S1 收益机制）**：GEMM1 批内稳态 =
  MTE1 载入 chunk(i+1) ∥ M 执行 mma chunk(i) ∥ FIX 写出 scores(i-1)——chunk 级
  SIG_L0KC token 精确衔接（l0b_pool 双 chunk 物理分离，sparse_flash_mla
  L0AB_EV0/EV1 同构）；**对照 gemm_v0：每 k-step `PipeBarrier<PIPE_M>` 全排空
  前一 mma（§S4-3-2 wall 根因，scalar_mte1_stall ≈78%）**——手动流水把"排空等待"
  替换为"仅对应 chunk 池的精确 token 等待"。跨 iter 由 side ping-pong 延续
  （FIX(i-1) ∥ mma(i) ∥ load(i+1) 深度 = Form A 已验证结构）。
- **批内多级重叠（Form A，继承）**：GM→L1（MTE2）∥ L1→L0（MTE1）∥ mma（M）∥
  L0C→w1（FIX）四段流水 pipe-pair flag 精确衔接。
- **跨核批量重叠（三形态共享，继承）**：GEMM1 批 ∥ softmax 批 ∥ GEMM2 批经
  [cid,stage] 多槽 workspace 解耦（SEM 按 CROSS_INTERVAL 批量发放）。
- **V 侧 MTE2/V 重叠（继承）**：io_buf/io3 双槽乒乓（#R5-1 后真实激活，机制层
  验证 ×3.95 在案）。
- **预取现状与候选（继承 + 更新）**：k_l1/v_l1 单缓冲（SIG 管程串行 GM→L1→L0）；
  L1 DB 预取族四路径已全部实测证伪/排除（§S6-3-5 跨会话证据链闭合：Form A
  generic +0.5% / Form B D256 零响应 / decode 带宽墙+预算否决 / Form A 单块批
  机制激活+真机零响应）——**rev4 不重启**；Form B-mma 域若 Stage 3 msprof 归因
  显示 MTE2 成为新墙（mma 化后 M 管压缩的必然候选），k/v L1 DB 可零探针重试
  （#R4-2b 机制知识在案），列 §10.3-③。
- **T.Pipelined 禁用依据（继承）**：rev2-R9 + Expert 零先例；流水一律手动表达。
- **恢复/降级旋钮（rev4 更新，性能不达标或超门禁时按序尝试，每项落地后复跑 §9.4
  门 + L0 阻塞口径）**：①Form B-mma 主案 a → 备选案 b（P2 探针失败即切换，非性能
  旋钮）；②unit_flag=0b10/0b11 流水模式（sparse_flash_mla 逐字，Stage 3 候选
  §10.3-②）；③NUM_STAGES 8→4（ws 足迹减半，R14 缓解）；④CROSS_INTERVAL 1→2/4；
  ⑤**分派谓词收窄回退：form_b_mma 谓词加守卫条件退回 gemm_v0 域（如特定 S_kv
  窄带劣化时）——三分派结构的模块化退路（兜底域源码在位，回退 = 谓词一行改动 +
  IR 守卫复证）**；⑥终极回退 = §12 回退条款（整文件回滚 bbfcbcaf）。

### 6.4 尾块处理（继承 rev3 §6.4，不变）

循环边界不因尾块改变（eff_num_outer/batch_iters/num_tb/batch_tasks 用 ceildiv/min
向上取整，尾批 = 最后 batch<NUM_STAGES 的批、尾块 = 批内最后迭代）；尾块由迭代内
valid_m/valid_n/rows_vid 运行期切片表达（输入侧 GM 两侧显式 BufferRegion +
need_clear 清零、输出侧 out_half[0:rows_vid,:] → Output vid 半区 + 运行期 if
守卫）；不使用"标量 GM 起点 + 完整 UB tile"写法；不做 host padding + crop；掩码
阈值钳位保证尾列不进 softmax 分母。**Form B-mma 增量**：K/N-split 维（D）静态
整除无尾块（§5.4-1）；chunk 载入恒满宽 128（尾行清零语义在 L1 行维，列段维恒满）。

---

## 7. 同步策略

### 7.1 同步模式（继承 rev3 §7.1）

**模式**: 全手动同步（Expert 标准）——四 pass_configs 全 False；kernel 源码显式
书写全部同步点：C 侧 `T.set_flag/T.wait_flag`（Form A / **Form B-mma** 细粒度
pipe-pair；Form B-gemm_v0 零用户 flag——gemm_v0 自平衡段界 #R4-1）+ 跨核
`T.set_cross_flag/T.wait_cross_flag`（C↔V SEM 0-5）+ V 侧 MTE2/MTE3↔V flag
管程 + init/destroy 闭环。

### 7.2 同步点说明（flag 矩阵；rev4 = SEM/V 侧继承 + Form B-mma C 侧扩展）

**（1）Cross-core Semaphore（AIC ↔ 2×AIV 组，mode=2；rev3 §7.2-1 表逐字继承——
三形态共享同一 SEM 协议，emit_v 零改动的协议面保证）**

| SEM ID | 名称 | 方向 | set 位置 | wait 位置 | 语义 |
|--------|------|------|----------|-----------|------|
| 0 | SEM_WS1_C2V | C→V | GEMM1 批内每 CROSS_INTERVAL 块 `set_cross_flag("FIX",·)` | V phase1 批内 `i%CROSS_INTERVAL==0` | w1[cid,0..i] scores 就绪 |
| 1 | SEM_WS1_V2C | V→C | V phase1 批末 `set_cross_flag("MTE2",·)` | C 每外层批 GEMM1 前 | w1 槽位已消费可复写 |
| 2 | SEM_WS2_V2C | V→C | V phase1 批内每 CROSS_INTERVAL 块 `set_cross_flag("MTE3",·)` | C GEMM2 批内 `i%CROSS_INTERVAL==0` | w2[cid,0..i] P 就绪 |
| 3 | SEM_WS2_C2V | C→V | C 每外层批 GEMM2 批末 `set_cross_flag("MTE2",·)`；init 预置 | V 每外层批 phase1 前 | w2 槽位已消费可复写 |
| 4 | SEM_WS3_C2V | C→V | GEMM2 批内每 CROSS_INTERVAL 块 `set_cross_flag("FIX",·)` | V phase2 批内 `i%CROSS_INTERVAL==0` | w3[cid,0..i] PV 就绪 |
| 5 | SEM_WS3_V2C | V→C | V phase2 批末 `set_cross_flag("MTE2",·)`；init 预置 | C 每外层批 GEMM2 前 | w3 槽位已消费可复写 |

（wait_cross_flag 单参调用——pipe 参数仅 A5 支持，910B 必须留空，ascend.py:213-231；
两 AIV 各自 set、AIC 一次 wait 的组聚合语义为硬件行为，继承不重新发明。**Form
B-mma 单块批分支的 SEM set/wait 位置 = #R3-1 协议逐字镜像**——GEMM1 批前
wait SEM_WS1_V2C、批内 (i+1)%CI==0 or i==batch_tasks-1 时 set SEM_WS1_C2V 等，
与兜底域单块分支一一对应。）

**（2）C 侧 Intra-core Flag——Form A（继承 rev3 §7.2-2，不变）**

| Ownership | 有向 pair | ID | 语义 |
|-----------|-----------|-----|------|
| K/P/V/Q L1 | MTE1→MTE2 / MTE2→MTE1 | SIG_K_L1=0 / SIG_P_L1=1 / SIG_V_L1=2 / SIG_Q_L1=3 | L1 buffer 双向 ownership |
| L0A/L0B 槽 0/1 | M→MTE1 / MTE1→M | SIG_L0AB=0,1 | 双槽：M 消费毕可重载 / 载入就绪可 mma |
| L0C 槽 0/1 | FIX→M / M→FIX | SIG_L0C=0,1 | 双槽：FIX 取出可重算 / mma 就绪可写出 |

**（2a）C 侧 Intra-core Flag——Form B-mma（rev4 新增；完整收支表，硬点 4：禁止悬空令牌）**

| Ownership | 有向 pair | ID | 语义 |
|-----------|-----------|-----|------|
| K/P/V/Q L1 | MTE1→MTE2 / MTE2→MTE1 | SIG_K_L1=0 / SIG_P_L1=1 / SIG_V_L1=2 / SIG_Q_L1=3 | 继承（与 Form A 同 ID 同语义） |
| l0b_pool chunk0/1 | M→MTE1 / MTE1→M | **SIG_L0KC=0,1** | 共池双 chunk：M 消费毕可重载 / 载入就绪可 mma（GEMM1 kc-chunk 与 GEMM2 n-chunk 分时共用同一 token 流，§4.5-d-2） |
| l0a_o side0/1 | M→MTE1 / MTE1→M | **SIG_L0AO=2,3** | GEMM2 A（P）双槽 ping-pong |
| l0a_s chunk0/1 | M→MTE1 / MTE1→M | **SIG_L0AS=4,5** | GEMM1 A（Q k-chunk）任务级常驻：任务头（generic）/每槽（单块批）载入保护 |
| l0c_s / l0c_o 槽 0/1 | FIX→M / M→FIX | SIG_L0C=0,1 | 双槽分时同流（GEMM1 l0c_s 与 GEMM2 l0c_o 交替使用——Form A P1 先例：分池 L0C 共享 SIG_L0C token 流） |

**Form B-mma 令牌收支表（静态可验；单位 = 每次发生的事件数）**——generic 多块分支：

| Token（有向） | init 预置 | 每任务（task 级） | 每 iter（GEMM1 批） | 每 iter（GEMM2 批） | destroy 消费 | 收支判定 |
|---------------|-----------|-------------------|---------------------|---------------------|--------------|----------|
| SIG_K_L1：set(MTE1→MTE2) | 1 | — | 1（K 载入前 wait 的对偶） | — | 1（wait） | init 1 + 每 iter set 1 = 每 iter wait 1 + destroy 1 ✅ |
| SIG_K_L1：set(MTE2→MTE1) | — | — | 1（K 载入毕） | — | — | 每 iter set 1 = 每 iter wait 1（chunk0 载入前）✅ |
| SIG_V_L1 / SIG_P_L1 / SIG_Q_L1 | 各 1 / 各 1 / 1 | Q：wait 1+set(MTE2→MTE1) 1+wait 1+set(MTE1→MTE2) 1（任务级 net 0）| — | V/P：各 set(MTE2→MTE1) 1 + wait 1 | 各 wait 1 | 同 Form A 结构（q_l1 归还于 l0a_s 预载后 = MTE1 读毕即还）✅ |
| SIG_L0AS+c（c=0,1）：set(M→MTE1) | 各 1 | 任务末 set 各 1（M 管序：本任务 GEMM1 末 mma 读毕归还；任务头 wait 各 1 消费 init/上一任务 set） | — | — | 各 wait 1 | init 1 + 每任务 set 1 = 每任务 wait 1 + destroy 1 ✅ |
| SIG_L0AS+c：set(MTE1→M) | — | 任务头载入毕 set 各 1 + 紧随 wait 各 1（M 管阻塞前移到任务头，零流水损失；GEMM1_BODY 宏内零 L0AS——任务内 mma 读 l0a_s 由 M 管序保序） | — | — | — | 每任务 set 1 = wait 1 ✅ |
| SIG_L0KC+c（c=0,1）：set(M→MTE1) | 各 1 | — | 每 chunk mma 后 set 1（×2） | 每 chunk mma 后 set 1（×2） | 各 wait 1 | init 1 + GEMM1 set 1 + GEMM2 set 1 = GEMM1 wait 1 + GEMM2 wait 1 + destroy 1 ✅（跨相位 token 流延续，§4.5-d-2） |
| SIG_L0KC+c：set(MTE1→M) | — | — | 每 chunk 载入后 set 1 + mma 前 wait 1（×2） | 同（×2） | — | 每 chunk set 1 = wait 1 ✅ |
| SIG_L0AO+side：set(M→MTE1) | 各 1 | — | — | 每 iter mma 后 set 1 | 各 wait 1 | init 1 + 每 iter set 1 = 每 iter wait 1（载入前）+ destroy 1 ✅ |
| SIG_L0AO+side：set(MTE1→M) | — | — | — | 每 iter 载入后 set 1 + mma 前 wait 1 | — | set 1 = wait 1 ✅ |
| SIG_L0C+side：set(FIX→M) | 各 1 | — | 每 iter FIX 写出毕 set 1 | 每 iter FIX 写出毕 set 1 | 各 wait 1 | init 1 + GEMM1 set 1 + GEMM2 set 1 = GEMM1 wait 1（c==0 前）+ GEMM2 wait 1（c==0 前）+ destroy 1 ✅ |
| SIG_L0C+side：set(M→FIX) | — | — | 每 iter c==1 mma 后 set 1 | 每 iter c==1 mma 后 set 1 | — | set 1 = FIX 前 wait 1 ✅ |

单块任务批分支：与上表同构，差异 = ①SIG_Q_L1 收支移入每槽（per-slot wait+set
(MTE2→MTE1)+wait+set(MTE1→MTE2) net 0——#R5-1 delta 逐字镜像，其"每槽令牌平衡
与 generic 任务级记账恒等"已在 Form A 实证）；②SIG_L0AS 每槽一轮（Q 任务相关
每槽重载：槽头 wait(M→MTE1)→载入→set(MTE1→M)→wait(MTE1→M)，槽末 GEMM1 批
FIX 写出后 set(M→MTE1) 归还——"槽≈任务"同构记账，每槽 net 0）；③任务末
SIG_Q_L1 归还语句不出现（每槽 net 0 自平衡，#R5-1 同构）。
**init 预置合计 13 set**（1 SEM + 4 L1 + 2 L0KC + 2 L0AO + 2 L0AS + 2 L0C）；
**destroy 消费合计 12 wait**（4 L1 + 2+2+2 L0A/L0B + 2 L0C；SEM_WS2_C2V init
token 为设计性悬挂——C 从不 wait 自己的 init 预置方向，与 rev3 Form A/B 一致，
cross-flag 状态 per-launch 归零，#R6-1 注释同款论证）。

**（3）V 侧 Intra-core Flag（每 AIV 独立；rev3 §7.2-3 逐字继承——emit_v 零改动）**

| Ownership | 有向 pair | ID | 语义 |
|-----------|-----------|-----|------|
| io_buf 槽 0/1（w1） | V→MTE2 / MTE2→V | SIG_IO_UB=0,1 | 双槽 staging 管程 |
| io3 槽 0/1（w3；D==bN 时别名 io_buf 共享 token） | V→MTE2 / MTE2→V | SIG_IO3=2,3（别名时 =0,1） | w3 staging 管程（io3_is_alias 规则继承） |
| acc_s_half / out_half | MTE3→V / V→MTE3 | SIG_S_HALF=0 | P 写出 + Output 写回闭环（best-practices §6：最终 store 必须进 ownership 闭环） |

**（4）init/destroy 闭环（作用域级）**

```python
# C scope init —— Form B-mma（13 项；Form A 9 项/兜底域 1 项继承 rev3 §7.2-4）：
T.set_cross_flag("MTE2", SEM_WS2_C2V)
T.set_flag("MTE1","MTE2", SIG_K_L1); T.set_flag("MTE1","MTE2", SIG_P_L1)
T.set_flag("MTE1","MTE2", SIG_V_L1); T.set_flag("MTE1","MTE2", SIG_Q_L1)
T.set_flag("M","MTE1", SIG_L0KC);   T.set_flag("M","MTE1", SIG_L0KC+1)
T.set_flag("M","MTE1", SIG_L0AO);   T.set_flag("M","MTE1", SIG_L0AO+1)
T.set_flag("M","MTE1", SIG_L0AS);   T.set_flag("M","MTE1", SIG_L0AS+1)
T.set_flag("FIX","M", SIG_L0C);     T.set_flag("FIX","M", SIG_L0C+1)
# C scope destroy —— Form B-mma（12 wait，与 init 的 flag 项对称；mha_sink:367-375 同构）：
T.wait_flag("MTE1","MTE2", SIG_K_L1); T.wait_flag("MTE1","MTE2", SIG_P_L1)
T.wait_flag("MTE1","MTE2", SIG_V_L1); T.wait_flag("MTE1","MTE2", SIG_Q_L1)
T.wait_flag("M","MTE1", SIG_L0KC);    T.wait_flag("M","MTE1", SIG_L0KC+1)
T.wait_flag("M","MTE1", SIG_L0AO);    T.wait_flag("M","MTE1", SIG_L0AO+1)
T.wait_flag("M","MTE1", SIG_L0AS);    T.wait_flag("M","MTE1", SIG_L0AS+1)
T.wait_flag("FIX","M", SIG_L0C);      T.wait_flag("FIX","M", SIG_L0C+1)
# V scope init/destroy：rev3 §7.2-4 逐字继承（emit_v 零改动：
#   init = SEM_WS1_V2C + SEM_WS3_V2C 预置 + SIG_IO_UB×2 [+ SIG_IO3×2 若非别名]
#          + SIG_S_HALF；destroy = 对称 wait）
```

**（5）flag 配对平衡硬约束（继承 rev3 §7.2-5）**：eff_kv_iters/eff_num_outer/
batch_iters/num_tb/batch_tasks 必须在 C/V 两 scope 以同一 bx/常量/公式各自计算
（不得经 workspace 传递）——Form B-mma prim_func 的 C 侧公式 = emit_v 内公式
逐字复制（§3.3 伪代码已对齐）；运行期掩码跳过（#4）与 rows_vid 守卫内 flag 均为
同 vid 局部成对（set+wait 同进同出）或纯 V 无 flag 段——不破坏配对。

**（6）条件 DMA 禁令（#R5-3 教训固化，硬点 4）**：Form B-mma 全部 DMA 为无条件
发射（K/N-split chunk 恒 2 步编译期展开；单块批分支 Q per-slot 载入 = #R5-1 已
验证镜像）。**任何 Stage 2/3 引入运行期条件 DMA 的尝试（如尾块 chunk 跳过、K/V
驻留省略装载）必须"wait+set 同进同出"**（消费侧 wait 同条件化，令牌收支恒等——
#R5-3 证伪链的首要教训：省略发射侧而保留消费侧令牌 = 非确定性损坏；4 次实验
证据链 optimization_log [RESULT-#R5-3]），且以非零数据探针验证（case20 全零输入
对该类改动是假阴性陷阱，§S5-4-3）。

### 7.3 pass_configs 配置（继承，不变）

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,   # 手动 T.Scope("C"/"V")
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,      # 手动 SEM 0-5
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,         # 手动 SIG flag 矩阵
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,   # 显式 alloc_*（planner 仅编址）
}
```

（Expert 标准全 False；不写 T.annotate_address——planner 顺序编址即满足 codegen
address_map 注册，ascend_memory_planning.cc:59-92，mha_sink 先例继承。）

---

## 8. 融合算子设计

### 8.1 融合算子判定（继承）

**判定结果**: 是——GEMM(QKᵀ) + element-wise/reduction 后处理（scale/mask/softmax）
+ GEMM(PV)，Cube 与 Vector 交替协作（决策树"matmul + element-wise → CV 融合"）。

### 8.2 CV 交互设计（Expert 显式 workspace + vid 二分；rev3 §8.2 逐字继承）

Workspace 规格表（w1 fp32 scores / w2 dt P / w3 fp32 PV，[core_num, NUM_STAGES,
...] 索引）、workspace_idx=[4,5,6] 位置约定、"CV 交互必须经 GM workspace"四点
源码证据（虚拟通道标记 ascend.cc:287-305 / copy_l0c_to_ub 无设备实现 /
AscendWorkspaceReduction 重写表 :1001-1004 / 三 Expert 参考实现全部显式
workspace）、rev2 vs rev3 每迭代数据流诚实对比表——**全部逐字继承 rev3 §8.2**
（rev4 三形态共享同一 workspace 协议；Form B-mma 不新增 ws、不改 dtype 主线：
w1/w3 fp32 位级锚主线、w2 dtype 原生栅格）。

**rev4 增量声明**：CV 数据流 3-ws GM 路由为**物理下限**（虚拟通道禁令 + 多块
w3 因 rescale 累加语义必须保留——单块域 Form A 已以 #R6-1 O 直写削减至 2 段，
该红利限 Form A；Form B-mma 域的同类扩展 = Stage 3 候选 §10.3-⑨，v1 保持 3 段
= emit_v 零改动前提）。

### 8.3 Cube 核计算流程（rev4：三形态）

```python
# ═══ Form A（prefill D≤128）：mma 双槽流水 —— rev3 §8.3 逐字继承（不变）═══
# <K 载入 SIG_K_L1 → Q 批首预载 if i<2 → K transpose→l0b_s[side] → SIG_L0AB →
#  mma(l0a_s[side], l0b_s[side], l0c_s[side], init=True) → SIG_L0C → FIX w1>
# …GEMM2 批对称：v_l1/p_l1 载入 → mma(l0a_o,l0b_o,l0c_o) → FIX w3…

# ═══ Form B-mma（prefill D=256）：手动 K/N-split mma 流水（rev4 新增）═══
# ---- GEMM1（每 iter i；side=i%2；l0a_s 已任务级/槽级预载）----
T.wait_flag("MTE1","MTE2", SIG_K_L1)
T.copy(K[bz, j0:j0+valid_n, by, :], k_l1)                # GM→L1 行跨步（运行期 extent）
T.set_flag("MTE2","MTE1", SIG_K_L1)
T.wait_flag("MTE2","MTE1", SIG_K_L1)
# chunk c=0（K 前半列段 0:128，transpose 载入 → mma init=True；l0a_s 就绪已由
# 任务头/槽头预载段 wait(MTE1→M,SIG_L0AS)×2 一次性确认，宏内零 L0AS——M 管序保序）：
T.wait_flag("M","MTE1", SIG_L0KC)                        # 池 0 已释放
T.copy(k_l1[:, 0:KC], l0b_pool[0,:,:], transpose=True)   # L1 列段 → L0B（NZ）
T.set_flag("MTE1","M", SIG_L0KC)
T.wait_flag("MTE1","M", SIG_L0KC)
T.wait_flag("FIX","M", SIG_L0C+side)                     # l0c_s 槽可写
T.mma(l0a_s[0,:,:], l0b_pool[0,:,:], l0c_s[side,:,:], init=True)
T.set_flag("M","MTE1", SIG_L0KC)                         # 池 0 释放（GEMM2 n0 可装载）
# chunk c=1（K 后半列段 128:256，mma init=False 链式累加）：
T.wait_flag("M","MTE1", SIG_L0KC+1)
T.copy(k_l1[:, KC:2*KC], l0b_pool[1,:,:], transpose=True)
T.set_flag("MTE1","M", SIG_L0KC+1)
T.wait_flag("MTE1","M", SIG_L0KC+1)
T.mma(l0a_s[1,:,:], l0b_pool[1,:,:], l0c_s[side,:,:], init=False)
T.set_flag("M","MTE1", SIG_L0KC+1)
T.set_flag("M","FIX", SIG_L0C+side)                      # scores 全块就绪
T.set_flag("MTE1","MTE2", SIG_K_L1)                      # k_l1 两 chunk 读毕归还
# FIX 写出：
T.wait_flag("M","FIX", SIG_L0C+side)
T.copy(l0c_s[side,:,:], workspace_1[cid, i, :, :])
T.set_flag("FIX","M", SIG_L0C+side)
if (i+1) % CROSS_INTERVAL == 0 or i == batch_iters-1:
    T.set_cross_flag("FIX", SEM_WS1_C2V)
# ---- GEMM2（每 iter i；主案 a：l0c_o 全宽列段 ×2 + FIX 全块 ×1）----
T.wait_flag("MTE1","MTE2", SIG_V_L1)
T.copy(V[bz, j0:j0+valid_n, by, :], v_l1)                # GM→L1
T.set_flag("MTE2","MTE1", SIG_V_L1)
T.wait_flag("MTE1","MTE2", SIG_P_L1)
if i % CROSS_INTERVAL == 0: T.wait_cross_flag(SEM_WS2_V2C)
T.copy(workspace_2[cid, i, :, :], p_l1)                  # w2 → p_l1（整 bM 行）
T.set_flag("MTE2","MTE1", SIG_P_L1)
T.wait_flag("M","MTE1", SIG_L0AO+side)
T.wait_flag("MTE2","MTE1", SIG_P_L1)
T.copy(p_l1, l0a_o[side,:,:])                            # P → L0A（全 K=bN=128）
T.set_flag("MTE1","M", SIG_L0AO+side)
T.wait_flag("MTE2","MTE1", SIG_V_L1)
# n-chunk c=0（V 前半列段 0:128 直载 → mma 写 l0c_o 左半）：
T.wait_flag("M","MTE1", SIG_L0KC)
T.copy(v_l1[:, 0:NC], l0b_pool[0,:,:])                   # 直载无 transpose（共池分时）
T.set_flag("MTE1","M", SIG_L0KC)
T.wait_flag("MTE1","M", SIG_L0KC)
T.wait_flag("MTE1","M", SIG_L0AO+side)                   # p 就绪
T.wait_flag("FIX","M", SIG_L0C+side)                     # l0c_o 槽可写
T.mma(l0a_o[side,:,:], l0b_pool[0,:,:], l0c_o[side,:,0:NC], init=True)
T.set_flag("M","MTE1", SIG_L0KC)
# n-chunk c=1（V 后半列段 → mma 写 l0c_o 右半）：
T.wait_flag("M","MTE1", SIG_L0KC+1)
T.copy(v_l1[:, NC:2*NC], l0b_pool[1,:,:])
T.set_flag("MTE1","M", SIG_L0KC+1)
T.wait_flag("MTE1","M", SIG_L0KC+1)
T.mma(l0a_o[side,:,:], l0b_pool[1,:,:], l0c_o[side,:,NC:2*NC], init=True)
T.set_flag("M","MTE1", SIG_L0KC+1)
T.set_flag("M","MTE1", SIG_L0AO+side)                    # l0a_o 两 chunk mma 读毕归还
T.set_flag("M","FIX", SIG_L0C+side)                      # PV 全块就绪（M 管序保证两段列均完成）
T.set_flag("MTE1","MTE2", SIG_V_L1)                      # v_l1 两 chunk 读毕归还
T.wait_flag("M","FIX", SIG_L0C+side)
T.copy(l0c_o[side,:,:], workspace_3[cid, i, :, :])       # FIX 全块 [32,256]
T.set_flag("FIX","M", SIG_L0C+side)
if (i+1) % CROSS_INTERVAL == 0 or i == batch_iters-1:
    T.set_cross_flag("FIX", SEM_WS3_C2V)
# （备选案 b 差异：l0c_o [2,32,NC] 窄池——每 n-chunk mma 后立即 FIX 列段写出
#   T.copy(l0c_o[side2], workspace_3[cid,i,:,c*NC:(c+1)*NC]) ×2，SIG_L0C token 每
#   chunk 一轮；SEM_WS3_C2V set 位于两段 FIX 之后。sparse_flash_mla:673-681 逐字先例。）

# ═══ Form B-gemm_v0（decode 兜底）：rev3 §8.3 Form B 段逐字继承（#R4-1 无 barrier）═══
# T.copy(K → k_l1) → T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True,
#   kL0Size=gemm1_kL0) → T.copy(acc_s_l0c → w1)   （gemm_v0 自平衡段界覆盖四类
#   hazard，§S4-3-1；单块批/generic 双分支结构不变）
```

**Form B-mma 稳态流水示意（S1 收益机制；chunk 级三管重叠）**：

```
时间 →      ┌─ iter i ─────────────┬─ iter i+1 ────────────┐
MTE1:  kc0(i) kc1(i) │ n0(i) n1(i) │ kc0(i+1) kc1(i+1) …
M   :        mma0(i) mma1(i) │ mma_n0(i) mma_n1(i) │ mma0(i+1) …
FIX :              w1(i-1) …  │  w3(i-1) …   │  w1(i) …
       （gemm_v0 对照：每 step 之间 PipeBarrier<PIPE_M> 全排空——三管强制串行化，
         scalar_mte1_stall ≈78% wall，§S4-3-2；手动版仅精确 token 等待）
```

### 8.4 Vector 核计算流程（两→三形态共享 emitter；rev3 §8.4 逐字继承，零改动）

phase 1（softmax 批：m_i_prev 备份 → 双舍入 r16(qk)→×s→r16(scores) → need_mask
运行期跳过 + 掩码链 → reduce_max/max 合并 → r_factors deferred → broadcast-sub →
exp → reduce_sum → single_kv_block 预归一 → SIG_S_HALF 闭环 P→w2）、phase 2（O
累加批：exp(r_factors) → sumexp 折叠 → acc_o rescale → w3→io3→work_ub → acc_o
累加）、任务尾（终归一 +1e-30 → cast out_half → rows_vid 守卫写回）——**emit_v
源码逐字不动**（rev4 S1 变更域仅 C-scope；V 侧对形态无感知：其协议面 = SEM 0-5
+ ws 形状/dtype，三形态一致）。

### 8.5 pass_configs 配置（继承，= §7.3）

### 8.6 注意事项（rev3 八条继承 + rev4 增量两条）

1. **Expert 帧禁带 threads 参数**（继承）：`T.Kernel(core_num, is_npu=True)`；
   自检 grep 无 `threads=`。
2. **别名禁令**（继承）：V-scope 纯计算 scratch（buf_2d/rt_2d/r_2d/rr_ub）绝不
   与跨核 copy 目的地（io_buf/io3/work_ub/acc_s_half/out_half）别名复用；物理
   复用（io_buf phase1/2 分时、io3_is_alias、r_2d 别名）必须有完整双向 flag
   ownership 闭环（终态实现已验证的别名规则逐字继承）。**rev4 增量：C 侧 l0b_pool
   共池为"同形分时复用"，其 ownership 闭环 = SIG_L0KC token 流跨相位延续
   （§4.5-d-2/§7.2-2a 收支表）——与别名禁令不冲突（禁令针对无闭环复用）。**
3. **虚拟通道禁令（R12）**（继承）：不得出现 T.copy(L0C→UB)/T.copy(UB→L1) 裸拷贝；
   自检 grep：L0C buffer copy 目的地仅 workspace_*（Form A 单块分支另有 Output
   直写先例 #R6-1——L0C→GM 合法方向）；L1 buffer copy 来源仅 GM 张量/workspace_*。
4. **C/V 同式约束（§7.2-5）**（继承）：eff_*/batch_*/num_tb/batch_tasks 公式、
   task 分解必须双侧逐字一致（复制粘贴级）——**Form B-mma prim_func 的 C 侧公式
   从 emit_v 逐字复制，任何单侧改动 = flag 配对破坏 = 死锁或静默竞态**。
5. **wait_cross_flag 单参**（继承）：pipe 参数仅 A5 平台；910B 全部单参调用。
6. **barrier 纪律（rev4 更新）**：兜底域 gemm_v0 三段式已纯移除 barrier_all
   （#R4-1，自平衡段界依据）；Form A / **Form B-mma 循环体内禁用 barrier_all**
   （细粒度 flag 已覆盖；barrier_all = PIPE_ALL 全排空 = S1 要消除的结构）；
   **Form B-mma prim_func 内禁用 gemm_v0**（event id 0 占用冲突防御 + 形态纯度，
   §3.4 禁用项）。V 侧一律 flag 管程（继承）。
7. **NUM_STAGES 偶数约束**（继承）：io_buf/L0 双槽轮转要求（host assert）。
8. **workspace 槽位索引 = 批内 i（非全局 idx）**（继承）：w*[cid,i,…]，SEM 释放/
   重取按批周转；跨批复用由 SEM_WS*_V2C/C2V 释放语义保护。
9. **（rev4 新增）K/N-split 链完整性**：GEMM1 的 init=True 必须且只能出现在 c==0
   chunk（c==1 恒 init=False）；GEMM2 主案 a 两 chunk 均 init=True（独立列段）——
   init 错位 = 累加语义破坏（scores 清零/翻倍类错误）；Stage 2 静态自检项 +
   探针 P3 位级覆盖。
10. **（rev4 新增）共池相位纪律**：l0b_pool 的 GEMM1 批/GEMM2 批分时使用不得
    交叉（批结构保证：GEMM1 批全部 iter 完成后才进入 GEMM2 批——rev3 批结构继承，
    无 iter 级交叉形态）；任何 Stage 2/3 引入 GEMM1/GEMM2 iter 级交错的尝试
    （如 GEMM2(i-1) 与 GEMM1(i) 同 iter 重叠）必须重新推导 SIG_L0KC 收支并过
    R11 压力门——v1 不引入（最小必要改动）。

---

## 9. 验证方案

### 9.1 Golden 函数（需求方提供，rev2 §9.1 逐字保留；Stage 2 `test_mha.py` 以此为唯一精度标杆，禁止改写语义；测试时在 **CPU** 上以与输入相同的 dtype 运行——不做 upcast，避免测试 harness 引入 NPU aclnn 依赖）

```python
import torch

def mha(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaleValue: float = -1.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """多头注意力: query [B,S,N,D], key/value [B,S_kv,N,D] → 输出 [B,S,N,D]"""
    B, S, N, D = query.shape
    S_kv = key.shape[1]

    if scaleValue <= 0:
        scaleValue = 1.0 / (D ** 0.5)

    # 转置为 [B, N, S, D]
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    # 缩放点积注意力
    scores = torch.matmul(q, k.transpose(-2, -1)) * scaleValue
    if is_causal:
        i = torch.arange(S, device=scores.device).unsqueeze(-1)
        j = torch.arange(S_kv, device=scores.device).unsqueeze(0)
        causal_mask = j > (i + (S_kv - S))  # 右下角对齐：上三角置 -inf
        scores = scores.masked_fill(causal_mask, float('-inf'))
    # F217: 全 mask 行 (整行 = -inf) 在 softmax 时得 0/0 = NaN，对齐
    # sparse_flash_attention 加显式保护 → 全 mask 行权重置 0。
    scores_max = scores.max(dim=-1, keepdim=True).values
    all_masked = torch.isinf(scores_max) & (scores_max < 0)
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    attn_weights = torch.where(all_masked, torch.zeros_like(attn_weights), attn_weights)
    attn_output = torch.matmul(attn_weights, v)

    # 转回 [B, S, N, D]；transpose 只改 stride, 而输出契约要求 contiguous
    return attn_output.transpose(1, 2).contiguous()
```

算子调用示例（用户侧契约，不变）：

```python
B, S, S_kv, N, D = 2, 128, 128, 8, 64
query = torch.randn(B, S, N, D, dtype=torch.float16, device="npu")
key = torch.randn(B, S_kv, N, D, dtype=torch.float16, device="npu")
value = torch.randn(B, S_kv, N, D, dtype=torch.float16, device="npu")
y = mha(query, key, value, scaleValue=-1.0, is_causal=False)
```

### 9.2 L0 门槛测试计划（tilelang-op-test-design 场景 A 按 rev4 设计重跑）

> **Stage 1 只给 L0**（规则 shape），供 Stage 2 精度收敛；L1（功能，含不规则/
> 尾块/质数 shape）/L2（异常）/Boundary（特殊值）完整分层套件**已存在于
> `test_mha.py`**（场景 B 历史产出：L1 19/L2 12/Boundary 6，接口契约不变 →
> 零改动复用 §9.5），Stage 2 直接运行既有套件，无需重新枚举。

**算子类别判定（场景 A Phase 2，按 rev4 设计重跑，结论与 rev3 一致）**：计算类型
= 混合（CV 融合，两 matmul 夹 softmax，§1.3/§1.4）；复杂度 = Fusion；数学特征 =
matmul+softmax+matmul；综合类别 = **Fusion（融合算子）**；**workspace 配置 =
True**（Expert 显式 workspace ×3 + vid 二分 + 跨核信号量）。测试策略：dtype_count=2
（fp16/bf16）、attr 组合 = is_causal{F,T} × scaleValue{auto,显式}、D 三档全覆盖、
快速冒烟 ≤10 用例。**rev4 增量判定**：三形态分派（Form A/Form B-mma/Form
B-gemm_v0）为 host 元数据谓词——L0 计划的路径映射按三分派更新（下表），用例集
本身不变（接口契约与 host 分派输入不变，十用例逐字继承 rev2/rev3）。

**L0 用例表（十用例逐字继承——接口契约不变；tile 配置按 rev4 §5.2）**
（输入 = CPU 生成 `randn×U[-1,1]` 确定性种子 → H2D；golden = §9.1 代码 CPU 同
dtype 运行）：

| # | 用例名 | (B,S,S_kv,N,D) | dtype | scaleValue | is_causal | 验证点（rev4 路径映射） |
|---|--------|----------------|-------|-----------|-----------|------------------------|
| 1 | l0_min_fp16 | (1,64,64,1,64) | float16 | -1.0（auto） | False | Form A 单块批（#R5-1+#R6-1 域，IR 不变守卫）；最小冒烟/N=1 退化 pitch |
| 2 | l0_min_fp16_causal | (1,64,64,1,64) | float16 | -1.0 | True | Form A 单块批 + causal（offset=0 下三角）；掩码链守卫 |
| 3 | l0_min_bf16_d128 | (1,64,64,1,128) | bfloat16 | -1.0 | False | Form A 单块批（P2 共池域守卫）+ bf16 |
| 4 | l0_min_bf16_d128_causal | (1,64,64,1,128) | bfloat16 | -1.0 | True | Form A 单块批 + causal；**单块特化位级精确锚（MERE=0.0 历史值）守卫** |
| 5 | l0_typical_fp16 | (2,128,128,8,64) | float16 | -1.0 | False | Form A 单块批（Skv=128=bN 谓词移动）；多 batch 多头 task 分解 |
| 6 | l0_typical_fp16_causal | (2,128,128,8,64) | float16 | -1.0 | True | Form A 单块批 + causal |
| 7 | l0_typical_bf16 | (2,128,128,8,64) | bfloat16 | -1.0 | False | Form A 单块批 bf16 |
| 8 | l0_typical_bf16_causal | (2,128,128,8,64) | bfloat16 | -1.0 | True | Form A 单块批 bf16 + causal |
| 9 | l0_kvlong_fp16_causal | (1,128,256,4,128) | float16 | -1.0 | True | Form A generic 多块（kv_iters=2，S<S_kv offset=128 右下角对齐）——批量流水 + deferred rescale 路径守卫 |
| 10 | l0_d256_fp16_scale | (1,64,64,2,256) | float16 | 0.0625（显式>0） | False | **Form B-mma 单块任务批分支（rev4 新路径唯一 L0 命中）**：D=256（bM=32）+ scaleValue 显式分支 + K/N-split mma 链 + l0b 共池 |

**rev4 kernel 路径映射（供 Stage 2 归因与 coverage_check 衔接）**：
- **形态**：Form A → #1-9（S>16 ∧ D≤128）；**Form B-mma → #10**（S>16 ∧ D=256 ∧
  S_kv=64≤bN=128 → single_kv_block 单块任务批分支）；Form B-gemm_v0 decode（S≤16）
  → **L0 无命中**（继承既有格局：L0 表以 prefill 规则 shape 为主），由 L1
  case10/13/14/16 类 + 哨兵 perf_decode_worst 强制覆盖（§5.6 result 行）。
- **Form B-mma generic 多块分支（case9/4 主战场）L0 无命中——补充覆盖声明
  （强制，非遗漏）**：L0 十用例为契约固定集（规则 shape），generic D256 多块由
  三层强制覆盖：①Stage 2 探针 battery（§9.6-1 P3/P6：Form B-mma generic 最小
  shape 族 (1,64,256,2,256)/(1,64,384,1,256) causal/非 causal/尾块 S_kv%128≠0/
  bf16，位级 vs rev3 终态 + golden 阻塞口径）；②哨兵 perf_d256_bf16
  (2,1024,1024,16,256)（多块 + 最大规模 + bf16，kernel 路径 120s 阻塞门）；
  ③L1 现有 D256 多块用例（test_mha.py L1 19 用例含 D=256 系）。
- **批量流水**：多外层批（eff_kv_iters > NUM_STAGES=8）→ L0 无命中（#9 为
  kv_iters=2 单外层批），由哨兵 perf_prefill_worst（Form A）/perf_d256_bf16
  （**rev4 起命中 Form B-mma generic 多外层批**）+ L1 大 S_kv 用例覆盖；批内
  batch_iters 余数分支（尾批 <8）→ #9（batch=2）+ #10（单块批 batch_tasks 余数）
  + 哨兵尾批。
- **单块特化**（S_kv ≤ bN=128）：#1-4,10（kv_iters=1）+ #5-8（S_kv=128=bN 谓词
  移动）；通用多块：#9（Form A）+ 哨兵/探针 battery（Form B-mma）。
- **掩码 4 变体**：(F,T)→#1,3,5,7,10 / (T,T)→#2,4,6,8,9 / (F,F),(T,F) 尾钳位→
  L1（S_kv%128≠0 现有用例）+ **Form B-mma 尾钳位由探针 battery ①强制前置**
  （D256 域 L0/L1 现无用例命中 (F,F)/(T,F)，探针补位）。
- **vid 分行/rows_vid 守卫**：全 L0 用例结构性命中（整除 shape 下 rows_vid=half_M
  恒正；0 行守卫分支由 L1 奇数 S 系 + R10 探针覆盖，继承）。
- **覆盖矩阵映射（proto 派生，coverage_check.py --proto）**：D-DTYPE-fp16
  （#1,2,5,6,9,10）、D-DTYPE-bf16（#3,4,7,8）、D-PARAM-is_causal 两值（F:#1,3,5,
  7,10 / T:#2,4,6,8,9）、D-PARAM-scaleValue 两分支（auto:#1-9 / 显式:#10）、D 三
  配置（64:#1,2,5-8 / 128:#3,4,9 / 256:#10）——与 rev3 映射一致（用例集不变），
  覆盖门禁预期 19 PASS / 0 MISS / 0 N/A 维持。

**L0 通过标准（阻塞口径逐字继承 rev2-C1 用户裁决，硬点 2-①）**：10/10 用例按
§9.3 阻塞口径判定通过——混合容差（§9.3.2）全过 **且** `MARE_abs < 10×Threshold`；
MERE/MARE_rel 为参考值（同步计算打印在 `[PRECISION_*]` 行，不阻塞、不触发升级
仲裁）。`python examples/mha/test_mha.py --level l0` 退出码 0。**该口径不因 mma
化做任何调整**（用户裁决不可更改；mma 累加序差异的影响域 = 位级诊断锚预期，
不进入阻塞判定——见下）。

**rev4 位级诊断锚体系重构（硬点 2-②；非阻塞，Stage 2 精度调试用）**：

- **锚-A（继承，守卫域收窄）**：Form A 域 + Form B-gemm_v0 域输出 vs **Developer
  终态 kernel**（`history_version/mha_impl_developer_final_6505.py`）位级一致。
  rev4 该两域 IR 逐字节不变（§9.6-2 守卫）⇒ **L0 #1-9（全 Form A 域）锚-A 必须
  保持 9/9 BITEXACT**（硬守卫：任何失配 = rev4 改动越界警报，阻塞性根因分析）。
  **#10 移出锚-A 域**（Form B-mma：mma 累加序 ≠ gemm_v0 内部累加序 → vs Developer
  终态**允许失配**——任务硬点 2 判定"必然失配"的设计承接：失配 ≠ FAIL，不进入
  任何阻塞判定）。
- **锚-B（rev4 新增，Form B-mma 域诊断主锚）**：Form B-mma 域输出 vs **rev3 终态
  kernel**（`history_version/mha_impl_r3_expert_final_7170.py`，bbfcbcaf）同 shape
  位级比对（#10 + 探针 battery 全 shape + case9/4 + 哨兵 d256）。**预期管理
  （诚实声明）**：大概率 BITEXACT——依据 = K 升序同序分片累加（§S4-3-4-①：单
  mma k=256 ≡ 链式 2×mma k=128 C0 顺序累加同序 12/12 BITEXACT）+ fp32 L0C 累加域
  同一 + w1/w3 fp32 主线不变 + emit_v 逐字（V 侧算式链位级同）+ N-split 列独立；
  **但不承诺**——T.mma（tl.ascend_mma → C++ mma 模板）与 gemm_v0
  （tl.ascend_gemm_v0 → C++ gemm_v0 模板，内部 Mmad 链）为不同 C++ 模板，指令级
  微序（如 L0C 分形内累加次序/flush 时机）差异可能导致 ulp 级位差。**失配处理
  纪律**：失配 ≠ FAIL → 根因分析限定两分支：①位差量级 ≤ ulp 级且阻塞口径全过
  （MARE_abs 余量 ≥10×）→ 判定为累加序微差，记录归档（锚-B 降级为根因分析工具，
  按新栅格（rev4 终态自身）重建基线）；②位差超 ulp 级或阻塞口径失败 → **同步/
  数据流缺陷嫌疑优先**（flash_attn_optimize §9(a) 纪律继承："手写 flag 同步的
  expert kernel 在任意一次运行中出现数值错误，都应视为同步或数据流缺陷"），进入
  Stage 2 精度调试流程（分段验证：w1 位级 → P 位级 → w3 位级 → 输出，GEMM1/GEMM2
  分域定位），重复 + 并发压力放大低概率竞态。
- **锚工具形态**：Stage 2 以 bit-diff 脚本实现（两实现同输入并行运行 + 逐位比较，
  调试工件置临时目录不进交付物；rev3 同款工具扩展 rev3 终态为第二基线）。

**特殊值用例原则（L0 仅声明不实施，继承 rev2/rev3）**：本算子声明输入值域为有限
实数，NaN/Inf 不在支持域（proto.yaml 同）。L2/Boundary 探测特殊值必须位置敏感
比较（有限值 + 稀疏特殊值混合确定性输入，先严格比对 isnan/±inf mask 再对双方
有限值位置套 §9.3 混合容差；禁止全 NaN/全 Inf 唯一用例；allclose(equal_nan=True)
不能替代显式 mask 检查）。

### 9.3 精度标准

#### 9.3.1 生态算子精度标准（需求方指定验收标准，MERE/MARE；**用户裁决 2026-09-18，rev2-C1，不可更改，逐字继承**）

误差指标（对全部输出元素，fp32 域计算）：

```
MERE（平均相对误差）= mean( |actual - golden| / (|golden| + 1e-7) )
MARE_rel（最大相对误差）= max( |actual - golden| / (|golden| + 1e-7) )
MARE_abs（最大绝对误差）= max( |actual - golden| )
```

**阻塞判定口径（需求方用户已在 Primary 上下文裁决，2026-09-18，不可更改；rev4
mma 化不触碰该口径——硬点 2-① 确认）**：

```
阻塞判定 = §9.3.2 混合容差全过  且  MARE_abs < 10 × Threshold
MERE 与 MARE_rel 为参考值：同步计算、打印在 [PRECISION_*] 行，不阻塞、
不触发升级仲裁（rev1 的 [MARE_REL_ESCALATION] 升级流程随本裁决废止）。
```

| dtype | Threshold | MARE_abs 阻塞线（10×Threshold） | MERE 参考线（Threshold） | MARE_rel 参考线（10×Threshold） |
|-------|-----------|--------------------------------|--------------------------|--------------------------------|
| float16 | 2⁻¹⁰ ≈ 9.77e-4 | **< 9.77e-3（阻塞）** | ~9.77e-4（参考） | ~9.77e-3（参考） |
| bfloat16 | 2⁻⁷ ≈ 7.81e-3 | **< 7.81e-2（阻塞）** | ~7.81e-3（参考） | ~7.81e-2（参考） |

（阈值结构与生态先例 `examples/cann-bench/group_norm/example_group_norm.py:487-491`
逐字一致；MERE 数学不可达性论证随裁决归档，Stage 2/3 不得以任何形式将 MERE/
MARE_rel 计入阻塞。）

#### 9.3.2 混合容差标准（用例级阻塞判定，harness `check_precision` 实现；**rev2 原样保留，用户裁决口径的组成部分**）

> 逐元素 `|actual-golden| ≤ atol + rtol·|golden|`，整体判定
> `matched_ratio ≥ required_matched_ratio` **且** `max_abs_error ≤ max_abs_error_limit`。
> 阈值仅按 dtype（与算子类别无关）；inf/nan 位置做结构比对不计入数值容差
> （完整定义见 tilelang-op-test-design/references/precision-standard.md）。
> §4 声明的每个 dtype 均有对应行；本算子无整型 dtype。

| dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|-------|------|------|---------------------|------------------------|
| float16 | 2⁻¹⁴ (6.10e-5) | 2⁻⁹ (1.95e-3) | 1e-1 | 0.99 |
| bfloat16 | 2⁻¹⁰ (9.77e-4) | 2⁻⁶ (1.56e-2) | 1e0 | 0.99 |

#### 9.3.3 精度对齐设计（golden 栅格对齐；rev4 = rev3 继承 + mma 化红线保持论证——硬点 2-③）

1. **scores 双舍入镜像（rev2-A2 继承）**：golden scores 经历 `round16(round16(qk)·s)`
   两次半精度舍入，kernel 逐一镜像（r16(qk) → ×s(fp32) → r16(scores)）。**rev4
   载体不变**：w1=fp32 主线（Form B-mma 的 l0c_s fp32 → FIX → w1 fp32，
   copy_l0c_to_gm T1=T2=float 无 cast——与 gemm_v0 域同模板同语义）；V 侧 rr_ub
   round-trip ×2 链序逐字（emit_v 零改动）。**GEMM1 执行体变更不触碰该栅格**：
   双舍入的第一舍入点 r16(qk) 在 V 侧、作用于 w1 fp32 读入值——C 侧 cube 内部
   （fp32 累加 + K-split fp32 链式）无论 gemm_v0 还是手动 mma，输出的都是"fp32
   域 qk 累加值"，进入同一 V 侧栅格链。2 的幂 scale（D=64/256）位级等价、
   D=128（1/√128）≤1ulp 对齐（继承）。
2. **P 以输入 dtype 进 GEMM2（继承）**：w2 dtype = fp16/bf16（P 原生栅格）；
   Form B-mma 的 p_l1→l0a_o 载入不改 dtype（GEMM2 A 操作数栅格 = w2 栅格 =
   golden 半精度 attn_weights 同栅格）。
3. **fp32 累加/归一化（继承，红线）**：GEMM fp32 累加（L0C——Form B-mma 的
   l0c_s/l0c_o 均 fp32）+ w3 fp32 + fp32 online 状态 + fp32 终归一。**K-split
   链式累加在 fp32 L0C 域进行**（init=False 读-加-写 L0C fp32），中间无半精度
   落地——精度不低于 gemm_v0 内部 k-step 链（同一 fp32 累加域、同一 K 升序）。
   w1/w3 dtype 瘦身会破坏本条，仅允许 Stage 3 探针门通过后启用（§10.3-① 继承）。
4. **单 KV 块预归一化特化（rev2-A3 继承）**：kv_iters==1 时 P̂ 归一化后再舍入，
   与 golden 一次性 softmax 同构——**rev4 载体不变**（V 侧 phase-1 特化 emit_v
   逐字；Form B-mma 单块分支保留 w3+phase-2 协议 = rev3 Form B 现状语义，
   #R6-1 O 直写等价性依据（O=P̂@V 一步终值）在 mma 域代数同样成立但 v1 不启用
   ——§0.4-b-#R6-1 行）。
5. **阻塞口径下的余量（继承 rev2-C1 实测）**：多块路径残差来源 = P 归一化位置差
   与最终输出舍入，量级 ~2⁻¹¹ 相对 / ~1e-4 绝对；Developer 终态实测 MARE_abs
   余量 ≥80×（L0）至 320×（哨兵 3.05e-5 vs 门限 9.77e-3）；rev3 Expert 终态
   实测同量级余量（probe 最差 MARE_abs 1.95e-3 vs gate 7.81e-2 = 40x，§S5-5/
   §S6-4）。**rev4 Form B-mma 域余量预期**：新增残差源 = mma vs gemm_v0 累加
   微序差（若锚-B 非 BITEXACT，量级 ≤ ulp 级 fp32 → 经 V 侧栅格链传播后 ≪ 阻塞
   线；40-320× 既有裕量吸收）——探针 P3/battery 实测复核，不设新门限。
6. **位级一致锚点论证（rev3 原文继承 + rev4 Form B-mma 域扩展）**：
   - **rev3 论证（Form A/兜底域，继续有效）**：GEMM 累加序（Form A mma K=D≤128
     单次调用内部 K-loop 与 gemm_v0 kL0Size=128 单步同一 cube 数据通路；兜底域
     逐字同调用）+ V 侧算式链逐字同构 + deferred rescale 确定性 + vid 二分行独立
     + ws 往返无舍入 → 与 Developer 终态位级一致（锚-A）。
   - **rev4 扩展（Form B-mma 域 → 锚-B vs rev3 终态）**：逐因子——
     ①GEMM1：手动 2×KC-128 链（init=True→False，K 升序）vs gemm_v0 kL0Size=128
     内部 2 k-step（K 升序）：同 fp32 L0C 累加域、同分片边界（128）、同次序；
     §S4-3-4-① 实证"单 mma k=256 ≡ 链式 2×mma k=128（C0 顺序累加同序）12/12
     BITEXACT"为同序链式累加的位级稳定性提供直接知识（该实证在 gemm_v0 模板族
     内取得；跨模板（mma vs gemm_v0）外推为**高置信预期而非承诺**——差异面 =
     C++ 模板指令微序，探针 P3 实测裁决）；
     ②GEMM2：N-split 列独立（每列段 init=True 全覆盖写，无跨段累加）→ 输出
     拼接语义与全 N 单次执行逐位同（探针 P2 实测裁决）；
     ③V 侧：emit_v 逐字 → 算式链位级同（确定性 SIMD）；
     ④ws 往返：w1/w3 fp32 无舍入、w2 dtype 同栅格（继承）；
     ⑤批处理/协议：SEM/SIG 位置不影响数值（数据依赖闭合即可）——#R4-1/#R3-1/
     #R5-1/#R6-1 四项"纯调度变更位级零漂移"实证在案（§S4-4/§S5-5）。
   - 锚-B 定位为**诊断性目标**（非阻塞判定）；失配处理见 §9.2 锚-B 纪律。

### 9.4 性能可行性哨兵（强制执行，不可因 large/L1 跳过；门限结构逐字继承 rev2-C4）

**门限结构**（继承）：①kernel 路径计时门（阻塞，可行性判定唯一计时依据）：
H2D → kernel（含该 shape 首次 JIT 编译）→ D2H + synchronize，每条 ≤ **120s**
（rev3 终态实测 6.0-7.4s，≥16× 余量；warm kernel 另记录作 Stage 3 基线）；
②精度判定（阻塞）：§9.3 口径；③golden 侧降本规定（分块 golden/缓存复用/确定性
抽样比对/可选 fp32-upcast 参考口径——harness `golden_mha_chunked` 已有实现，
逐字继承）；④case 级墙钟参考上限 1800s（非阻塞，防 harness 挂死）。

| 用例名 | (B,S,S_kv,N,D) | dtype/属性 | rev4 覆盖路径 | kernel 路径门（阻塞） | 历史实测 | 选择理由 |
|--------|----------------|-----------|--------------|----------------------|----------|----------|
| perf_prefill_worst | (8,2048,4096,32,128) | fp16, causal | Form A P2 多外层批（IR 不变守卫 + 最大 KV 重读 + 15.7MB ws 足迹） | ≤120s | rev3 终态门内（§S4-4：6.0-7.4s 族） | GM/ws 流量最大规模（§5.5 核算继承） |
| perf_decode_worst | (64,1,4096,32,128) | fp16, causal | Form B-gemm_v0 D1（bN=512+kL0=32；**rev4 兜底域守卫**：valid_m=1 → rows_vid vid0=1/vid1=0 守卫 + trim + 掩码全交互） | ≤120s | 同上门内 | S=1 极端尾块 × 2048 任务 × vid 分行守卫最坏 |
| perf_d256_bf16 | (2,1024,1024,16,256) | bf16, 非 causal | **Form B-mma P3-mma generic 多外层批（rev4 起命中新路径）**：D=256 最大 L0/L1 占用 + bf16 GEMM + K/N-split 全交互 | ≤120s | rev3 终态 4.88e-4 MARE_abs（gemm_v0 域）；rev4 首测待产出 | 预算最紧配置 + bf16 + **Form B-mma 多块强制覆盖（§9.2 补充覆盖声明②）** |
| perf_max_grid | (128,128,128,64,64) | fp16, causal | Form A P1（任务网格上限 16384 任务 ≈683 任务/核 + single_kv_block，IR 不变守卫） | ≤120s | 同上门内（9.77e-4） | 任务调度/Fixed Core 最坏；支持域边界 B=128、N=64 |

哨兵判定（继承）：kernel 路径门超时 = **可行性失败** → 触发 Stage 1 回退（重选
block/NUM_STAGES/形态分派——rev4 语境下首选降级旋钮 §6.3-⑤ 分派谓词收窄回退）；
精度阻塞口径失败 → Stage 2 精度流程。若 Form B-mma 首测逼近门禁，按 §6.3 降级
旋钮顺序调整后复检。

### 9.5 测试资产复用声明（rev4 契约）

`test_mha.py`（844 行，md5 0dd9e93c）依赖的**全部对外契约在 rev4 保持不变**：
`from mha import mha`（host 签名/断言/out_idx 语义逐字继承 §3.3）、golden §9.1
原样、阻塞口径判定逻辑（rev2-C1）、L0/L1/L2/Boundary/perf 五层套件与 `--level`
接口、哨兵 kernel 路径门（含 kernel 源码 hash 记录路径门）。→ **测试资产零改动
复用**；Stage 2 只运行不修改。锚-A/锚-B bit-diff 工具与 Form B-mma 探针 battery
为 Stage 2 内部调试工件（临时目录，不进交付物、不改 test_mha.py——rev3 §9.5
同款约定）。覆盖门禁 `coverage_check.py test_mha.py --proto proto.yaml` 预期
19 PASS / 0 MISS / 0 N/A 维持（proto 不变、用例集不变）。

### 9.6 Stage 2 验收路径（rev4 分级门禁；§11.4 同款结构 + 调优会话方法学固化）

1. **探针门（强制第一步，先于任何 L0 运行；算子外最小 kernel / 隔离脚手架，
   临时目录）**：
   - **P1**：L1 列段切片 × transpose 载入 × NZ/ZN annotation 位级探针——
     `k_l1[128,256](NZ)[:, 0:128]/[:,128:256] → l0b [128,128] transpose=True` +
     `q_l1/v_l1 列段直载`，annotation 开/关两态 × 尾块行（valid_n<128）×
     fp16/bf16；对照 = 整块载入的等价拼接（整数精确 + 随机数据 BITEXACT；
     **非零数据探针纪律 §S5-4-3**）。失败应急 = 去 k_l1 NZ annotation
     （sparse_flash_mla 无 annotation 先例，性能降级不换语义）。
   - **P2**：mma C region 切片 N-split 位级探针（主案 a 前置门）——算子外最小
     kernel：`[32,128]@[128,256]` 全 N 单 mma vs 2×`[32,128]@[128,128]` 列段
     region mma（l0c[side,:,0:128]/[side,:,128:256]）BITEXACT + golden 阻塞口径。
     **失败 → 切换备选案 b**（l0c_o 窄池 + w3 列段 FIX，sparse_flash_mla 逐字
     先例；设计已完备，零设计变更），并加跑 P5。
   - **P3**：K-split init=False legacy 链位级探针——`[32,256]@[256,128]ᵀ` 单
     gemm_v0(kL0Size=128) vs 手动 2×mma(KC=128, init=True→False, legacy) vs
     golden：三重比对（BITEXACT 预期 + 阻塞口径兜底）；**该探针同时为锚-B 预期
     提供直接证据**（§9.3.3-6-①）。失败应急 = unit_flag=0b10/0b11 模式
     （sparse_flash_mla 逐字参数面切换）。
   - **P4**：P3-mma 全池实编预算读数——GetWithOffset/编译探针读取 L0A/L0B/L0C/
     L1/UB 实配（⚠ 顶格条目强制项：l0b_pool 64KB、l0c 96KB、L1 152KB；MEMORY_
     PLANNING 实编读数口径优于手算，§4.5 制度）。超限 → 备选案 b（L0C 96→64KB）
     → 仍超限触发 §6.3 降级旋钮。
   - **P5**（仅备选案 b 启用时）：w3 列段 FIX 写出位级探针（copy_l0c_to_gm
     realDstN 列段，#R6-1 gate 探针 8/8 方法移植）。
   - **P6**：Form B-mma 域 shape battery（§9.2 补充覆盖声明①）：generic 最小族
     （(1,64,256,2,256) kv_iters=2 / (1,64,384,1,256) 尾块+causal / bf16 变体 /
     (1,32,512,2,256) 多外层批）+ 单块族（#10 同构 + causal + 尾块 S_kv%128≠0）
     ——全部对 rev3 终态（bbfcbcaf）锚-B 位级 + golden 阻塞口径双判定。
   - R10/R15 历史探针族结论继承（Expert BSHD 16/16、掩码链隔离——rev3 已关闭，
     环境未变免重跑；若 Stage 2 环境校验发现异常则重跑）。
2. **形态隔离 IR 守卫（先于 kernel 改动前的基线快照 + 改后 diff）**：16 shape
   `get_kernel_source()` dump（§S5-5 方法学）——**Form A 12 守卫 shape + Form
   B-gemm_v0 decode 守卫 shape 生成码逐字节不变**（case1/2/5/6/7/8/19/20 族 +
   case10/11/14/16/17 族代表）；Form B-mma 目标 shape（case9/4/哨兵 d256/L0 #10
   族）按预期改变。基线 = 开工快照（当前 mha.py = bbfcbcaf 已核验）。
3. **kernel 实现**（mha.py rev4 修订：三分派 + Form B-mma prim_func；现文件先
   备份 history_version/）+ 锚-A/锚-B bit-diff 工具。
4. **L0 门槛**（`--level l0` 10/10 阻塞口径 + 锚-A 9/9 硬守卫 + 锚-B #10 诊断）。
5. **L1**（`--level l1` 19/19）+ `coverage_check.py --proto`（19 PASS 预期）。
6. **全量**（`--level all`：L0+L1 阻塞、L2 12/12 正确拒绝、Boundary 6/6、哨兵
   4/4 kernel 路径门）+ **R11 竞态压力**：继承 4 配置 ×12 复跑 nondet=0 +
   **rev4 扩展配置专打 Form B-mma 被改路径**（单块/多块 × causal × S_kv 尾块 ×
   双 dtype × batch_tasks {1,2,4,5,6,7,8} 非齐次 × 跨轮 num_tb≥2 × P3-mma 共池
   × K/N-split token 深度）×12 nondet=0（§S5-5 方法学修正继承：压力门必须打在
   被改路径上）。
7. **性能验证（msprof 唯一依据 + camodel 仅机制定性——§S6-3-3 纪律）**：
   case9/4 主受益 + decode/Form A 守卫全集 msprof median；camodel 机制判定 =
   Form B-mma 三管重叠段结构激活 + 生成码 PipeBarrier<PIPE_M> 每-step 排空消失
   （IR diff 静态可证）；**禁止以 camodel 数字声称收益**。
8. **cann-bench 复评**（§12 验收方式 + 回退条款判定）→ Stage 3 按 §10.3 候选清单。

---

## 10. 风险点与注意事项

### 10.1 已知约束与风险（rev4 = R1-R16 状态更新 + R17-R20 新增）

| # | 风险 | 等级 | 说明与缓解 |
|---|------|------|-----------|
| R1-R3 | 掩码链注册 / 4D 行跨步 lowering / 循环内运行期 valid_n | **已关闭（继承，rev3 实证吸收）** | R10/R15 探针族 16/16 位级通过（rev3-attempt1，mha.py:88-93 docstring）；模板层与模式无关结论继承 |
| R4 | D=256 任务数翻倍 → KV 重读 ×2 | 低（性能，继承） | 实测吸收（perf_d256_bf16 哨兵门内，rev3 终态在案）；rev4 Form B-mma 不改 GM 流量（§5.5） |
| R5 | MERE/MARE_rel 口径病态 | **已解决（用户裁决，继承）** | 阻塞口径 = 混合容差 + MARE_abs（§9.3.1）；Stage 2/3 不得将 MERE/MARE_rel 计入阻塞 |
| R6 | causal 无裁剪冗余算力 | **已消除（继承 #1）** | 运行期裁剪三形态标配；C/V 同式复制（§7.2-5） |
| R7 | 每 shape 组合一次 JIT 编译 | 低（继承 + 增量） | **rev4 增量**：三形态各编译一次（同 shape 只命中其一，无额外成本）；首次编译计入 §9.4 kernel 路径门 |
| R8 | AscendVidReduction × threads=2 × BSHD 4D 结构性 bug | **Expert 形态不触发（继承复检关闭；Developer 语境禁令保留）** | §2.2-c 五点证据链继承；rev4 全文无 threads 参数（grep 自检）；回退场景 rev2 §10.1-R8 逐字有效 |
| R9 | threads=1 × T.Pipelined codegen bug | **Expert 形态不适用（继承复检关闭；T.Pipelined 全文禁用不变）** | §2.2-d 继承；rev4 K/N-split 编译期展开不新增循环形态 |
| R10 | BSHD 4D 行跨步 × Expert vid × 运行期 extent × ZN/NZ 组合 | **已关闭（rev3 探针 16/16 位级实证）**；rev4 增量组合转 R17 | rev3-attempt1 探针族结论继承（整除/S=1/30/64/90/128 + need_clear + vid 半区写回 + rows_vid=0 守卫 + 跨任务 ws 复用 + transpose 敏感 K）；**rev4 新增 L1 列段切片 × annotation 组合 → R17-P1** |
| R11 | flag 配对失衡 → 死锁或静默竞态 | 高（继承，Expert 固有风险） | 触发场景与缓解五条继承（C/V 同式 + 静态 diff 自检 / init-destroy 闭环 / 运行期守卫内 flag 成对 / 哨兵-L0 首跑超时看门狗 / flash_attn_optimize §9(a) 纪律）；**rev4 增量**：Form B-mma 新增 SIG_L0KC/L0AO/L0AS token → §7.2-2a 收支表静态验证（硬点 4）+ P6 battery + 扩展压力配置专打（§9.6-6） |
| R12 | 虚拟通道误用（L0C→UB / UB→L1 裸拷贝） | 中（继承） | §8.6-3 禁令 + grep 自检（L0C 目的地仅 workspace_*——Form A 单块分支 Output 直写为 L0C→GM 合法方向先例）；rev4 Form B-mma v1 无 O 直写，L0C 目的地 = w1/w3 仅两种 |
| R13 | 顶格预算条目（⚠） | 中（继承 + rev4 扩容） | 继承条目（P2/P1 l0b 64KB、兜底域内部 L0B 64KB、D1 L1 276KB）+ **rev4 新增（P3-mma l0b_pool 64KB 共池、L0C 主案 96KB）**——全部 ⚠ 条目探针 P4 实编 GetWithOffset 读数复核；P3-mma l0b_pool 同尺寸先例（本算子 P2 实配 + sparse_flash_mla v_l0b）为强证据；超限降级链 = 主案 a→备选 b（L0C 96→64KB）→ §6.3 旋钮 → 分派谓词收窄回退 |
| R14 | workspace GM 足迹 ≈11-16MB/编译实例（L2 驻留不确定） | 中（性能，继承） | [cid,stage] 私有索引 + 参考实现同量级实证 + msprof 归因 + NUM_STAGES 降档退路（继承）；rev4 不新增 ws（足迹不变） |
| R15 | 掩码链 ops Expert V-scope 首用 | **已关闭（rev3 实证）** | rev3-attempt1 R15 隔离探针 + L0 case2 哨兵 + 四轮调优零异常（debug_log.md）；rev4 V 侧零改动 → 不重开 |
| R16 | Expert 版整体劣于 Developer 终态（65.05） | **已解除（rev3 实证：71.70 > 65.05，M2 达成）**；rev4 演化为 R16' | **R16'（rev4 回退条款载体）**：rev4 终态劣于 rev3 终态（cann-bench <71.70 或平均 <0.9054x）→ 回滚 bbfcbcaf（§12）；分阶段验收（探针门 → IR 守卫 → L0 → 全量 → msprof 归因 → cann-bench）+ 降级旋钮链（§6.3） |
| **R17** | **Form B-mma 新组合探针门族（rev4 新增，Stage 2 第一道硬门）** | **高（新路径首建）** | 四个仓内首例/新组合：①L1 列段切片 × transpose × NZ annotation（P1；先例 sparse_flash_mla 无 annotation 形态 → 应急 = 去 annotation）；②mma C region 列段切片 N-split（P2，主案 a——前端源码确认、C++ 模板 C-offset 无直接先例 → 应急 = 备选案 b 先例完备切换）；③init=False legacy 链式累加（P3——链语义双先例但 legacy 参数形式首用 → 应急 = unit_flag 模式切换）；④l0b_pool 64KB 共池顶格实配（P4——同尺寸先例强证据 → 应急 = 备选 b/降级链）。**全部探针前置于 L0（§9.6-1），失败均有预案，无"探针失败即设计失败"单点** |
| **R18** | **位级锚预期管理（rev4 新增，硬点 2 载体）** | 中 | 锚-A #10 移出守卫域（Form B-mma vs Developer 终态允许失配——mma 累加序 ≠ gemm_v0 内部累加序）；锚-B（vs rev3 终态）为诊断主锚：**预期大概率 BITEXACT 但不承诺**（跨 C++ 模板指令微序）；失配处理纪律 = §9.2 锚-B 两分支（ulp 级 + 阻塞口径过 → 归档降级为根因分析工具并按 rev4 终态重建基线；超 ulp 或口径失败 → 同步/数据流缺陷嫌疑优先，分段定位 w1→P→w3→输出）。**风险实质 = Stage 2 误把锚-B 失配当 FAIL 反复重构（过度调试）或把真缺陷当累加序差异放过（调试不足）——判别标准已量化（ulp 级 + 40× 余量线）** |
| **R19** | **三分派谓词覆盖（rev4 新增）** | 低 | (S>16∧D≤128)/(S>16∧D=256)/(S≤16) 三谓词互斥穷尽（D 静态封闭集 + S 二分——§5.6 DISPATCH-COVERAGE 审计 result: pass）；每特化 ≥1 命中用例（Form B-mma 多块 L0 无命中 = 明示声明 + 三层强制覆盖 §9.2）；host 分派纯 Python bool（元数据层）；**应急 = 分派谓词收窄回退（§6.3-⑤，一行改动 + IR 守卫复证）** |
| **R20** | **K/N-split token 收支失衡 → 死锁/竞态（rev4 新增，R11 的具体化）** | 中-高 | 新 token（SIG_L0KC/L0AO/L0AS）× 共池跨相位流 × 单块/generic 双分支 = 收支面扩大。缓解：①§7.2-2a 静态收支表为实现准绳（Stage 2 逐行对照自检 + 静态 diff）；②init 13/destroy 12 闭环逐字模板（§7.2-4）；③首跑超时看门狗（hang → 立即归因 flag 矩阵，R11 缓解④继承）；④P6 battery + 扩展压力 ×12 nondet=0；⑤条件 DMA 禁令（§7.2-6，#R5-3 教训）；⑥sparse_flash_mla L0AB_EV0/EV1 token 结构逐字对照（先例同构复核） |

### 10.2 常见错误（rev3 十六条继承 + rev4 新增四条）

rev3 §10.2 全表继承（Expert kernel 带 threads / 虚拟通道裸拷贝 / C/V eff_* 公式
漂移 / init-destroy 闭环缺失 / scratch 别名 / Form A 循环体 barrier_all /
wait_cross_flag 带 pipe / NUM_STAGES 奇数与槽位索引笔误 / m_i_prev 备份时序 /
UB-L0 溢出 / 尾列污染分母 / 全掩码行 NaN / vid1 零行写回 / select SPR 模式与
int32 compare 退化 / host 顺手物理化 / TVMScript parse-time if 作用域）。rev4 新增：

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| K-split init 链错位 | GEMM1 c==1 误写 init=True（或 c==0 误 init=False） | scores 减半/翻倍类系统错误（首 chunk 丢弃或未清零累加） | §8.6-9 链完整性纪律：init=(c==0) 单一表达式派生；P3 探针位级覆盖；静态自检项 |
| 共池相位交叉 | 把 GEMM2 批的 l0b 装载提前进 GEMM1 批（iter 级交错"优化"） | l0b_pool chunk 覆写未消费数据 → 非确定性损坏 | §8.6-10 共池相位纪律：v1 批结构保证分时；交错尝试必须重推 SIG_L0KC 收支 + R11 压力门 |
| Form B-mma 域混入 gemm_v0/barrier_all | 从兜底域分支复制粘贴代码 | event id 0 占用冲突（gemm_v0 内部 L0AB_EVENT=0 与 SIG_L0KC=0 同 pair 撞号）/ PIPE_ALL 排空复活 | §3.4 禁用项 + §8.6-6：form_b_mma 分支 grep 自检（无 `gemm_v0`、无 `barrier_all`） |
| 列段切片越界/错位 | c*KC 偏移笔误（如 KC 误用 D）或列段宽与 mma K 不一致 | 静默数据错位（K==K_B 校验可能仍过） | chunk 宽单一常量源（KC/NC host 派生 + assert D%KC==0）；P1/P2 探针整数精确 + transpose 敏感数据（非对称 K 矩阵探针，rev3 R10 族方法继承） |

### 10.3 特殊场景处理（rev3 继承 + rev4 更新）

- **S=1/2 decode（vid 尾行矩阵）**：兜底域路径逐字继承（valid_m=1/2 → vid0
  rows_vid=1/2、vid1 守卫跳过；single_kv_block 命中走预归一特化）。
- **全零输入退化**（case20 类）：继承（scores=0 → 均匀 softmax → 输出 = 可见 V
  行均值）；**rev4 提示：全零输入对 Form B-mma 的 K/N-split 类改动是假阴性陷阱
  （§S5-4-3 教训）——P1-P3/P6 探针一律非零随机数据 + transpose 敏感非对称矩阵**。
- **causal 且 S=S_kv**（offset=0）/ **causal 且 S<S_kv**（offset>0）/ **N=1** /
  **block_num ≤ core_num**：继承 rev3 §10.3 四条。
- **prefill D=256 极小 S_kv（S_kv≤128 单块域）**：Form B-mma 单块任务批分支
  （#R3-1 协议镜像；L0 #10 命中）——batch_tasks=1 退化安全（my_count<NUM_STAGES
  时单槽批，io_buf/L0 双槽空转但协议闭合，#R3-1 论证继承）。
- **Stage 3 性能优化候选清单（rev4 更新；正确性无关，勿在 rev4 Stage 2 引入；
  每项落地须复跑 §9.4 门 + L0 阻塞口径 + 锚体系复核）**：
  ① **w1 dtype 瘦身重试（§S6-5 首位候选）**：fp32→fp16（fixpipe r16 落点）——
    RNE 等价探针 11/11+19/19 在案免探针成本，改动 ~2 行 + ws 形状/dtype 派生；
    case6/1/19/20 现处 FIX 吞吐墙下瓶颈构成已变（§S6-5）；**rev4 增量提示：
    Form B-mma 域 case9 若 mma 化后 FIXP 成为新 pacing 管，w1 瘦身对 case9 亦
    有杠杆**；风险 = 锚-A/锚-B 位级失配预案按 #R3-3 归档口径复核；
  ② **unit_flag mma→fixpipe 硬件融合（rev4 升级：Form B-mma 域直接相关）**：
    `T.mma(unit_flag=0b10/0b11)` + `T.copy(L0C→GM, unit_flag=0b11)` 跨 L0C
    ping-pong 重叠（customize.py:118-125 docstring + copy_op.py:284-291；
    sparse_flash_mla 全程用法先例，含窄 N pipe_barrier("m") 特判警示——本算子
    N=128 恒定无该形态）；GEMM1 K-split 链为天然适配点（kc0 0b10 → kc1 0b11 →
    w1 copy 0b11）；
  ③ **k/v L1 双缓冲（Form B-mma 域重试窗口）**：四路径证伪链在案（§S6-3-5），
    但 mma 化改变 Form B-mma 域瓶颈构成（M 管排空消除后 MTE2 可能成为新墙）——
    先 msprof op 归因确认再落地（#R4-2b 机制知识在案，零探针成本重试）；
  ④ **NUM_STAGES/CROSS_INTERVAL 扫描**：继承（#R5-4 暂缓判定：爆炸半径大，
    需全 20 case 复测）；
  ⑤ **Form A bN=256 变体**：继承（L0B 双槽 128KB>64KB 硬限否决在案——除非
    单槽牺牲乒乓或 k-split 增指令，收益不确定，§S3-3-4(a)）；
  ⑥ **axpy 融合**：继承（破坏双舍入链序，必须先过 §9.3.3-1 栅格等价论证 +
    锚复核，rev2-⑤ 警示）；
  ⑦ **io_buf 加深 / out_half-acc_s_half 合并**：继承（UB 余量换 MTE3 管程松弛，
    ownership 闭环重证）；
  ⑧ **decode 专用重排（heads-as-M / split-KV）**：继承 session-2 证据性放弃结论
    （§S2-5），不重启；
  ⑨ **#R6-1 O 直写扩展至 Form B-mma 单块域（rev4 新增）**：等价性依据同源
    （单块预归一化 O=P̂@V 一步终值）；前置门 = gate 探针扩展（D=256/bM=32/
    valid_m 尾行，#R5-2 8/8 方法移植）+ emit_v gate 谓词泛化（form_a →
    form_a or form_b_mma）+ WS3 SEM 对删除后的 token 重平衡 + V 侧 init 悬挂
    token 论证扩展（#R6-1 注释同款）；收益预期 = case4 族/L0 #10 域单块 D256
    （砍 V-epi 相位 + w3 往返）；
  ⑩ **kv-outer G=2 分组（S2 复活条件挂钩）**：仅当 w1/w3 瘦身落地后 msprof
    归因显示 case7 C 侧 pacing 管从 FIXP 迁移至 MTE2 时重评（§0.5 论证三/四）。

---

## 11. 交付清单

### 11.1 目录结构

```
examples/mha/
├── mha.py            # 纯 kernel（@tilelang.jit mha_kernel 三形态工厂 + host 包装 def mha(...)），可 import，无 golden/测试/__main__
├── test_mha.py       # from mha import mha + §9.1 golden + L0/L1/L2/Boundary/哨兵全套件 + main(--level)（零改动复用）
├── proto.yaml        # 算子接口规格（dtype/attr），供覆盖门禁派生应覆盖维度（rev4 复核无变化）
├── DESIGN.md         # 本设计文档（rev4）
├── debug_log.md      # Stage 2 证据记录（rev4 attempt 续写）
├── perf_tuning/      # Stage 3 调优证据（final_report.md 六会话证据链 + mindstudio_export/case9_simulator 流水数据 + 续写）
└── history_version/  # 回退备份（design_rev1/2/3、proto_rev1/2/3、Developer 终态、r3 Expert 终态 7170 锚、各轮快照）
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成（本文件，rev4；rev3 已备份 history_version/design_rev3.md，md5 e97dacb6 核验一致） | rev4 设计（含 §0 修订记录/S2 降级论证/成果继承清单、L0 计划、锚体系重构、回退条款） |
| `proto.yaml` | ✅ rev4 复核**无变化**（host 契约不变；现文件 md5 3079c184 = history_version/proto_rev3.yaml） | dtype 全集 = §9.3.2 表（float16, bfloat16）；attrs = scaleValue(Float) + is_causal(Bool)（§11.5） |
| `mha.py` | ⬜ Stage 2 rev4 修订（**现文件 = 回退锚 bbfcbcaf 不动直至备份就位**：history_version/mha_impl_r3_expert_final_7170.py 已核验同体） | 三分派 + Form B-mma prim_func 新增；Form A/Form B-gemm_v0/emit_v 三域逐字保留 |
| `test_mha.py` | ✅ **零改动复用**（§9.5 契约论证；md5 0dd9e93c） | golden + 全套件 + 双口径 harness + 阻塞判定 + 哨兵路径门；Stage 2 只运行不修改 |
| `history_version/` | ✅ 回退备份就位 | design_rev1/2/3.md、proto_rev1/2/3.yaml、mha_impl_developer_final_6505.py（锚-A 基线）、**mha_impl_r3_expert_final_7170.py（锚-B 基线 + 回退锚）**、test_mha_developer_final.py、r3_perf1-14/s2/s3 各轮快照 |
| `perf_tuning/` | ✅ 继承证据链 | final_report.md（六会话 65.05→71.70 全证据 + §S4-5 rev4 立项建议）、optimization_log.md、mindstudio_export/case9_simulator/（流水依据 + README 导引）、msprof_*/simulator_* 原始数据 |

### 11.3 命名规范

- 目录名: `mha`（snake_case）✅ 已存在
- kernel 文件: `mha.py`；测试文件: `test_mha.py`（顶部 `from mha import mha`）
- kernel JIT 工厂: `mha_kernel`（内部 Python 级 form_a/form_b_mma/form_b_gemm_v0
  三分支 prim_func 构建）；host 包装: `mha(query, key, value, scaleValue=-1.0,
  is_causal=False)`

### 11.4 实现顺序（Stage 2 验收路径）

1. ✅ 设计文档（DESIGN.md rev4）+ proto.yaml（复核无变化）+ L0 门槛测试计划
   （§9.2：十用例继承、三分派路径映射、锚体系重构、Form B-mma 多块补充覆盖声明）
2. ⬜ **前置探针门（R17 P1-P6，强制第一步，先于任何 L0 运行）**：§9.6-1 全表
   （P1 列段×transpose×annotation / P2 C region N-split / P3 K-split legacy 链 /
   P4 实编预算读数 / P5 备选案列段 FIX / P6 Form B-mma shape battery 锚-B+阻塞
   口径）——临时目录，不进交付物
3. ⬜ IR 基线快照（16 shape，改前）→ kernel 实现（三分派 + Form B-mma prim_func；
   现文件先备份）→ IR 守卫 diff（Form A/decode 域逐字节不变 + 目标域按预期改变）
   + 锚-A/锚-B bit-diff 工具
4. ⬜ L0 门槛（`--level l0` 10/10 阻塞口径 + 锚-A 9/9 硬守卫 + 锚-B #10 诊断分析）
5. ⬜ L1（`--level l1` 19/19）+ `coverage_check.py test_mha.py --proto proto.yaml`
   （19 PASS / 0 MISS / 0 N/A 预期）
6. ⬜ 全量（`--level all`）+ R11 压力（继承 4 配置 + rev4 扩展配置专打 Form B-mma
   被改路径，×12 nondet=0）
7. ⬜ msprof 性能验证（case9/4 主攻 + decode/Form A 守卫；camodel 仅机制定性）→
   cann-bench 复评（§12 验收 + 回退条款判定）→ Stage 3 按 §10.3 候选清单调优

### 11.5 算子 proto.yaml（覆盖门禁用；rev4 复核 = 无变化）

接口契约不变（§9.5：host 签名/断言/语义逐字继承）→ proto.yaml 内容与 rev3 完全
一致：dtype 全集取自 §9.3.2 精度表（float16、bfloat16——§4.1 仅给代表性 dtype，
不作全集来源）；attrs 取自 §1.3/§4.1（scaleValue: Float, default -1.0；is_causal:
Bool, default false）——两者均影响计算路径（scale 解析分支 / 掩码 4 变体），供
`coverage_check.py --proto` 派生 D-DTYPE-*/D-PARAM-* 覆盖维度。与 §9.3.2/§4 一致性
已自检 ✅（现文件 md5 3079c184 与 history_version/proto_rev3.yaml 相同，无需重写；
**rev4 形态分派为 kernel 内部实现细节，不构成接口维度，proto 不新增字段**）。

```yaml
operator:
  name: mha
  category: Fusion
  formula: |
    y = softmax(Q @ K^T * s) @ V ;  s = scaleValue if scaleValue > 0 else 1/sqrt(D)
    is_causal: scores[..., i, j] = -inf for j > i + (S_kv - S)   # 右下角对齐
    all-masked row (仅域外 S>S_kv 可达): softmax → 0（golden all_masked 保护对齐）
  attrs:
    - name: scaleValue
      type: Float
      default: -1.0
      required: false
    - name: is_causal
      type: Bool
      default: false
      required: false
  inputs:
    - name: query    # [B, S, N, D]
      dtype: [float16, bfloat16]
    - name: key      # [B, S_kv, N, D]
      dtype: [float16, bfloat16]
    - name: value    # [B, S_kv, N, D]
      dtype: [float16, bfloat16]
  outputs:
    - name: output   # [B, S, N, D]，与 query 同 dtype，contiguous
      dtype: [float16, bfloat16]
  schema: mha(Tensor query, Tensor key, Tensor value, float scaleValue=-1.0, bool is_causal=False) -> Tensor output
```

---

## 12. 性能目标（rev4 重写：S1 立项目标 + 回退条款——硬点 6/7 载体）

- **目标类型**: `baseline_compare` —— 对齐 cann-bench 评测（`tasks/level4/mha`，
  hardware=910b2，baseline = CANN `FusedInferAttentionScore`，
  `tasks/metadata/910b2.json` baseline_perf_us）
- **锚点数据（rev3 Expert 终态，cann_final_eval_20260922_100108，rev4 对照基线）**:
  - 综合得分 **71.70/100**（编译 20/20、精度 30/30、性能 21.70/50）；平均加速比
    **0.9054x（M2 达成）**；通过率 20/20。
  - per-case 终态（case: speedup）：1:0.943 / 2:0.495 / 3:0.763 / **4:0.828** /
    5:0.538 / 6:1.283 / **7:0.463 / 8:0.468 / 9:0.324** / 10:0.983 / 11:0.927 /
    12:0.880 / 13:0.886 / 14:1.394 / 15:0.686 / 16:0.785 / 17:1.143 / 18:0.807 /
    19:1.426 / 20:2.079（§S6-7 权威表）。
  - 计时口径声明继承（final_report §7）：msprof Task Duration 中位数为唯一性能
    依据（本地口径与 cann-bench 互验差 <3%）；禁止 Python/Torch 计时；camodel
    仅机制定性（§S6-3-3）。
- **rev4 总目标（用户立项 S1）**: **case9 ≥ 0.4x**（M1 case 线达成；预期 0.46x 级
  = case7 同族结构，§5.2-1 论证）；平均加速比在 0.9054x 基础上保持或上行
  （case9 +0.14 对平均贡献 ≈ +0.007，case4 族顺带受益）。
- **守卫约束（逐项硬约束）**:
  - **decode 不回退（硬点 6，构造性保证 + 复评核验双保险）**：case10 ≥0.90 /
    case13 ≥0.69 / case14 ≥0.99 / case16 ≥0.52 / case20 ≥0.82（当前 0.983/0.886/
    1.394/0.785/2.079）——decode 全域落兜底域 IR 逐字节不变（§9.6-2 守卫），
    评测复评核验为最终裁决；
  - **Form A 域不回退**：case1/2/3/5/6/7/8/19（±3% 噪声带内视为持平；IR 逐字节
    不变守卫 + 本地 msprof 佐证双证据，§S4-1 方法学继承）；
  - **case4 ≥ 0.828**（Form B-mma 受益域，只升不降预期；§S6-7 记录 case4 历史
    ±0.018 测量方差带，判定时计入）；
  - 精度：全量门禁（L0/L1/L2/Boundary/哨兵）+ 覆盖门禁 19 PASS 维持。
- **验收方式**: 探针门 → IR 守卫 → L0/L1/coverage/all → msprof per-case 归因
  （case9/4 主攻 + 守卫集）→ cann-bench 权威复评（`run_evaluation.sh --task-dir
  tasks/level4/mha --operator MHA`，与会话 3-6 同口径参数）；噪声阈值 3%。
- **测试 shape**: §9.4 四条哨兵 + cann-bench 20 cases（case9 主验证 / case4 同族 /
  case7/8 Form A 守卫 / case10/13/14/16/20 decode 守卫 / case1/6/19/20 单块 Form A
  守卫）。
- **中止条件**: ①Stage 2 attempt 上限（5 次，orchestrator 规程）；②Stage 3 迭代
  达上限或连续 3 次无提升（调优会话规程继承）；③探针门 P1-P4 全部预案穷尽仍
  不可行 → 返回 `[DESIGN_ERROR]` 由 orchestrator 决策（rev5 或触发回退条款）。
- **回退条款（立项级，硬点 7 逐字承接）**:
  - **触发 A（Stage 2 精度无法收敛）**：L0 阻塞口径在探针门通过前提下仍无法
    10/10（attempt 上限内），且锚-B + 分段验证定位到设计层缺陷 → 返回
    `[DESIGN_ERROR]`；
  - **触发 B（rev4 终态劣于 rev3 终态）**：cann-bench 综合分 < **71.70** 或平均
    加速比 < **0.9054x**（调优预算耗尽后）→ **回滚 rev3 Expert 终态**：
    `history_version/mha_impl_r3_expert_final_7170.py`（bbfcbcaf）→ `mha.py`，
    并向 orchestrator 报告回滚与证据（DESIGN.md 是否回滚 design_rev3.md 由
    orchestrator 决策）；
  - 回滚后 test_mha.py 仍零改动可用（契约同一）；
  - **域级降级中间档（§6.3-⑤）**：触发 B 前若 msprof 归因明确仅 Form B-mma 域
    劣化（如特定 S_kv 窄带），可先执行分派谓词收窄回退（form_b_mma 加守卫条件
    部分/全部退回 gemm_v0 域，一行改动 + IR 守卫复证）——保留已验证收益域。
- **硬约束提醒（沿用并更新）**: R8/R9 Developer 语境禁令保留（回退场景立即生效）；
  R10-R20 探针门与自检 grep 项（无 `threads=`、无虚拟通道拷贝、form_b_mma 域无
  `gemm_v0`/`barrier_all`、C/V 同式 diff、K-split init 链完整性）为 Stage 2/3
  例行检查项；§10.3 候选清单为 Stage 3 唯一优化入口（每项过精度门 + 哨兵门 +
  锚体系复核）。

---

## 附：rev4 质量自检结论（quality-checklist.md 逐项）

- **第 0 项 Host 合规**：✅ §4.8 审计六项全 allow（metadata 读取/纯 Python 标量
  运算/**rev4 增量分派谓词 = 纯 Python bool**/设备属性查询/dtype 字符串化/kernel
  调用直返；零 contiguous/padding/cat/to(dtype)/切片+reshape；不采用参考实现
  host mask/block_table 预构建）。
- 必须项：1 ✅（§2.1 Expert 明确 + §2.2 五点论证继承 + rev4 增量复核）；2 ✅
  （§3.2 逐 API 函数名+参数+源码行号，含 rev4 新增四行与 §3.4 新增项实证表）；
  3 ✅（§4.4 全路径含 Form B-mma 增量 + §1.5 数据流图）；7 ✅（全文无占位符/
  TODO/待补充；"实编读数验证/探针门"为明确的 Stage 2 强制制度，主案/备选案双
  预案完备无未决模糊项）；8 ✅（§3.5.1 全项重跑含新增 L1 列段切片检测 +
  §2.2-c/d R8/R9 复检继承）；9 ✅（分形 M≥16/K≥16/N≥16 全形态复核 §5.3）；
  13 ✅（§3.5.3 十一项 examples/ 路径，含 rev4 新增 sparse_flash_mla/xattention/
  锚-B 基线）；19 ✅（§9.3.1/9.3.2 具体数值 + fp16/bf16 全 dtype 行 + 用户裁决
  口径逐字）；20 ✅（§11.5 proto.yaml 复核存在且一致，md5 核验）；21 ✅（§5.5
  REORDER-COST 三形态量化 + case9 定量核算）；22 ✅（§9.4 四哨兵：最大 shape/
  最坏 dtype（**rev4 起 perf_d256_bf16 命中 Form B-mma 新路径**）/最大任务数/
  decode 极端尾块，kernel 路径 120s 阻塞门）。
- 推荐项：4 ✅（§5.2 三分派逐条论证 + bM/NUM_STAGES 不动理由）；5 ✅（§7 flag
  矩阵与 Expert 模式匹配 + §7.2-2a 收支表）；6 ✅（§9.1 golden + §9.2 L0 计划
  含 Form B-mma 多块补充覆盖声明，分层套件交现有资产）；10 ✅（§5.3 L0C 逐配置
  + 元素规则）；11/12 ✅（§8.2 workspace 规格继承 + workspace_idx 与签名位置
  一致）；14/15/16/17 ✅（§3.5.2 差异表两代（rev3 八行继承 + sparse_flash_mla
  五行新增）含路径/结构/参数面差异——BSHD 原生输出无 transpose 需求）；18 ✅
  （无全局变量依赖，维度全参数化）。
- 三强制审计块：HOST-METADATA-AUDIT ✅（§4.8，六项 allow）；REORDER-COST ✅
  （§5.5，result: pass）；DISPATCH-COVERAGE ✅（§5.6，result: pass，三分派谓词
  互斥穷尽 + 全特化有命中用例或明示的强制补充覆盖）。
- **revision 模式专属自检**：✅ §0.1-0.2 相对上一版关键调整（N1-N8 变更清单，
  每项含依据/证据出处）；✅ §0.6 为何不会再犯同一错误（rev1/rev2/rev3 + 调优
  会话八项教训对照）；✅ 与历史备份差异化核验——rev1/rev2 = Developer 模式
  （rev4 为 Expert，模式不同）；rev3 = Expert 双形态（rev4 = 三分派 + Form B-mma
  新执行体 + S2 降级论证 + 锚体系重构，关键选择无一与历史版本完全一致）。

---

## §12 追加：会话 14（r14）指令融合臂性能目标（Orchestrator 追加，2026-10-05；不覆盖既有内容）

- **授权链**：用户 2026-10-05 基于「指令融合/标量向量化空间分析」报告选择范围 A（算子级 2 项）；范围 B（框架级隐式广播/sub-exp 原语）与 §3.1 结构级（decode D=256 手工 mma 化）不在本轮授权内，留待本臂实测结果后决策。
- **目标类型**：`best_effort` + 预登记机制判据（本臂为指令削减型，非延迟消融型）。
- **变更范围**（爆炸半径限 emit_v，V 侧；C 侧/协议/旗标/SEM 位置零触碰，框架文件零触碰）：
  1. 双舍入链首环 cast 直连：`copy(io_buf→work_ub)` + `copy(work_ub→rr_ub)` → `copy(io_buf→rr_ub)`（三处路径：single_kv_block / flat_v / generic）。位级论证：copy 无损、work_ub 原始 fp32 值无读者、§3.2.1-⑦ 栅格舍入点序逐一保持。
  2. 单块 Form B phase-2：`fill(acc_o,0)+add(acc_o,io3)` → `add(acc_o, io3, 0.0)`（x+0.0 ≡ 0.0+x 含 ±0 全域成立）。
- **采纳判据**：(a) 精度门全绿（A/B BITEXACT vs c316a004 全探针矩阵 + 锚 A 9/9 + B 14/14 + stress + l0/l1/coverage/all + smoke）；(b) 机制签名 = case7 `aiv_vec_time` 显著下降 ∧ AIV 发射当量下降 ∧ AIC 侧 IR 20/20 逐字节不变；(c) wall 门 = ≥1 目标 case（case7/8 主、case15/18 次）改善 ≥3% ∧ 守卫零回退（漂移带口径）。机制激活但 wall 零响应 → 回滚归档（r7-1 纪律），吸收律新增实例，零净变更收口 = 合格产出。
- **测试 shape**：目标 case7 (4,1024,1024,32,128) / case8 twin + case15 (60,1,512,16,256) / case18；守卫 case9/10/11/14/16/20；n=25 median、warmup 5 / iters 20、同会话新鲜基线、±3% 带。
- **噪声阈值**：3%（项目标准）。**最大迭代数**：3（预算延展 35→38，用户授权记录于 .orchestrator_state.json）。
