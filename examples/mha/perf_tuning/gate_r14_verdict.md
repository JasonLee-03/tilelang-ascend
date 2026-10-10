# Gate r14 Verdict — 指令融合臂（用户授权范围 A：算子级 2 项）

- **会话**: perf_session=14 (r14)，全局迭代 iter36（Phase A+B+C）/ iter37（Phase D+E+F），软预算 3（上限 38）
- **日期**: 2026-10-05
- **授权链**: 用户 2026-10-05 审阅 Orchestrator「指令融合/标量向量化空间分析」报告后授权范围 A（算子级 2 项：#R14-1 cast 直连 + #R14-2 fill+add 合一）；任务书为唯一授权范围。DESIGN.md §12 追加节预登记判据。
- **verdict**: **ROLLED_BACK_ZERO_WALL**（r7-1 条款：机制主签名激活但 wall 零/负响应 → 回滚三重验证 + 变体留存 + 吸收律新增实例归档 = 合格产出）

## 1. 两项融合落地状态

| 项 | 内容 | 落地状态 | 证据 |
|----|------|---------|------|
| ① #R14-1 | 双舍入链首环 cast 直连（emit_v 三路径：single_kv_block L480/485、flat_v L635/650、generic L811/819；`copy(io_buf→work_ub)`+`copy(work_ub→rr_ub)` → `copy(io_buf→rr_ub)`，r16(scores) 第二轮往返不动） | **全量落地于变体**（3/3 路径） | 生成器 5 替换块唯一性断言；IR 普查 copy_ub_to_ub 恰 −1/shape × 20 形；40 CAST hunks 白名单全过 |
| ② #R14-2 | 单块 Form-B phase-2 fill+add 合一（L564-565 → `T.tile.add(acc_o, io3[io_side,:,:], 0.0)`） | **全量落地于变体**（1/1 位点） | IR 普查 fill_acc_o −1 / add_acc_o_io3 −1 / adds_io3_acc_o +1 仅 6 个 Form-B 单块形；6 FILL hunks 白名单全过 |

- **变体**: `/tmp/opencode/r14_probes/mha_r14x_edit_v1.py` md5 **535b71f0f6a3bf45150efca3439bfc59**，2236 行（delta 精确 −4），ruff check/format clean，ast OK；生成器 `make_r14x.py`（源=冻结生产态，7 hunks −8/+4 断言过）
- **终态（回滚后生产态）**: `examples/mha/mha.py` md5 **c316a0040ff4f4445d81bb31ec94a68e**（2240 行，= perf19 = perf20 快照同体，diff 逐字节核验）；框架两文件 82e80025/1c3a00ed 全程零触碰；test_mha.py/DESIGN.md/proto.yaml/提交副本零改动；.orchestrator_state.json 未读写
- **模板机械偏离登记**: 任务书模板注释前 3 空格 → 2 空格（ruff format 硬门优先，注释文字逐字保留）

## 2. 精度门逐项结果（Phase D，全绿）

| 门 | 结果 |
|----|------|
| smoke 6/6（先行，HANG 纪律） | **PASS** — 三改动路径+fill 位+bench 目标全覆盖，全 BITEXACT vs 生产 + golden 口径过，无挂死 |
| **A/B BITEXACT 主门**（变体 vs c316a004 同输入逐位） | **PASS 54/54 全 BITEXACT** — probe_r131_kvdb 24 目标 + 12 守卫 + form_a/fbm/bf16 探针集 18（case8-bf16/case2/case3/case5/A_noncausal/case6/case19/case20-zeros-ctrl/A_bf16_min/case9/case4/fbm_bf16×2/fbm_tail300/case17/case14/case16/case11）；PART 0 融合位审计 54/54；「任何一位不等 = FAIL」条款零触发 |
| 锚-A 9/9 BITEXACT | **PASS**（硬门；r14 变更域使锚成员为域内直接测试） |
| 锚-B 14/14 | **PASS**（BATTERY_RESULT: PASS；anchorB=BITEXACT 诊断 14/14、DIFF 0） |
| 压力：继承 4 + 扩展 19 配置 ×12 | **PASS** — nondet=0 全绿（23 配置行） |
| l0 10/10 | **PASS**（PRECISION_PASS×10 "Test Passed!"） |
| l1 19/19 | **PASS**（"Test Passed!"） |
| coverage | **PASS**（19 PASS / 0 MISS / 0 N/A） |
| `--level all` | **EXIT=0 "Test Passed!"**（L2 12/12 正确拒绝；哨兵 4/4 PERF_PASS：prefill_worst 6.6s / decode_worst 7.3s / **d256_bf16 6.1s matched=1.00000** / max_grid 6.3s） |

换位窗纪律：SWAP-IN md5 535b71f0 核验 → STEP 2-8 → trap RESTORE_PROD md5 c316a004 核验 OK。缓存 r14_cache_var 专属；probe/smoke/锚/压力 disable_cache 或专属目录（假阳性缓存污染结构性排除）。

## 3. IR 机器证明结果（Phase C，先于任何真机运行）

- **AIC 20/20 逐字节？** —— **计算语义域：YES（Gate C2 = 20/20 逐字节）**；**字面域：QUALIFIED（Gate C1 = 0/20 字面相同，20/20 合格限定）**。C1 差异**全部**为共享序言中 V 专属缓冲的 `ascend_ub.GetWithOffset` 地址重排（cast 直连使 rr_ub 首用位前移 ⇒ codegen UB 存储规划器按首用序重排视图；迁移集仅 work_ub/rr_ub/m_i_prev [+m_i/row_thresh causal-multi 形]）。布局审计 20/20 全过：缓冲集合恒等、逐缓冲 pool/type/size 恒等、非 UB 视图（L1/L0A/L0B/L0C）逐字节恒等、InitBuffer 五池尺寸恒等（ub=196352）、新布局零重叠、池界内。**C 侧计算/旗标/gemm/copy 零触碰由 C2 + 八族指令普查（Δ=0×20）+ AIC 发射当量逐位相同（运行时呼应）三重证明**；字面偏离已如实定性为授权改动的机械性后果并穷尽审计（详见 optimization_log [SELF-CHECK]③）。
- **AIV diff 局限融合点？** —— **YES**：66 hunks 全分类 = 40 CAST（AIV 内，两行签名 = DataCopy→Cast 直连 + 删原 r16 环）+ 6 FILL（AIV 内，Fill+Add→Adds）+ 20 UB-VIEW（序言，C1 已审计）；零 UNCLASSIFIED、零意外指令增删；旗标四家族 + Mmad/gemm_v0/copy_gm_to_l1/copy_l0c_to_gm 普查 20/20 Δ=0。
- provenance：case15 base dump md5 3a6bf589 = r13 采纳态档案值（基线 = r13 终态机器证明）；20 形双臂 dump EXIT=0。

## 4. Wall A/B 表（Phase E：同会话新鲜基线，3× interleaved paired，n=25 median，Δ=var−prod，df=2 闭式 p，95%CI t=4.302653）

| case | 类别 | prod r1/r2/r3 (µs) | var r1/r2/r3 (µs) | Δ mean (µs) | Δ% | p | 95%CI(%) | 判读 |
|------|------|--------------------|--------------------|------------|-----|---|----------|------|
| **case7** | 主目标 | 855.80/855.63/854.36 | 864.66/865.39/863.12 | **+9.13** | **+1.07%** | **0.00121** | [+0.91,+1.23] | **显著劣化** |
| **case8** | 主目标 | 853.49/854.08/854.79 | 857.80/858.06/857.32 | **+3.61** | **+0.42%** | **0.02221** | [+0.15,+0.70] | **显著劣化** |
| case15 | 次目标 | 419.70/419.63/420.14 | 419.94/419.96/421.05 | +0.49 | +0.12% | 0.143 | [−0.10,+0.33] | ns 零响应 |
| case18 | 次目标 | 422.61/422.08/422.32 | 422.80/421.92/422.71 | +0.14 | +0.03% | 0.476 | [−0.13,+0.20] | ns 零响应 |
| case9 | 守卫 | 868.73/867.98/868.51 | 872.13/874.04/869.77 | +3.57 | +0.41% | 0.124 | [−0.28,+1.10] | ns 带内 |
| case10 | 守卫 | 34.72/35.22/34.90 | 35.38/35.30/35.21 | +0.35 | +1.00% | 0.174 | [−1.07,+3.08] | ns 带内 |
| case11 | 守卫 | 309.99/311.36/309.89 | 310.35/310.98/310.55 | +0.21 | +0.07% | 0.561 | [−0.36,+0.50] | ns 带内 |
| case14 | 守卫 | 376.42/377.90/375.67 | 375.42/377.84/376.87 | +0.05 | +0.01% | 0.948 | [−0.72,+0.74] | ns 带内 |
| case16 | 守卫 | 563.76/560.00/561.08 | 563.37/564.90/561.68 | +1.70 | +0.30% | 0.404 | [−0.94,+1.55] | ns 带内 |
| case20 | 守卫 | 14.37/14.47/14.27 | 14.35/14.36/14.53 | +0.04 | +0.30% | 0.735 | [−3.03,+3.64] | ns 带内 |

60/60 runs exit 0，生产态 md5 每重复边界+出口核验全 c316a004。**wall 门 (c)：无任何目标 case 改善（遑论 ≥3%）；主目标 case7/case8 显著劣化；守卫零显著回退（点估计皆 ≥0 = 本会话轻微系统正偏，如实登记）。**

## 5. Op 签名（case7/case15 双臂，`--kernel-name=main_kernel --aic-metrics=PipeUtilization,ResourceConflictRatio,L2Cache,Memory,BasicInfo,ArithmeticUtilization,MemoryUB,MemoryL0`）

### case7（主目标，form_a flat_v 域）

| 指标 | prod | var | Δ | 判读 |
|------|------|-----|-----|------|
| **aiv_vec_time** | **648.99** | **624.22** | **−24.77 (−3.82%)** | **机制主签名激活 ✅（预登记带 −3~8% 命中；648.99 与 iter26 档案逐字复现）** |
| aiv_mte2_t | 231.71 | 210.30 | −21.41 (−9.2%) | 被删每槽 DataCopy 的 MTE2 归属一致【推断】 |
| aiv_scalar_busy | 689.25 | 685.18 | −4.07 | 每槽 −1 helper 调用 |
| **AIV 发射当量 (single+2×dual)** | **155.32** | **157.69** | **+2.37 (+1.53%)** | **方向与预登记期望相反——如实报告**（single 26.82→27.79 / dual 64.25→64.95）；解释【推断】：运行时槽偏移寻址迁移入 Cast 调用（base Cast 源=work_ub[0] 静态，var Cast 源=io_buf[(i%2)*1024] 运行时）；激活证明由 IR 普查+vec_t 独立承担 |
| aiv_wait_id0 | 58.23 | **93.20** | **+34.97 (+60%)** | **wait_idN 迁移列（警戒项兑现）**——吸收主通道 |
| aiv_wait_id4 | 163.85 | **182.01** | **+18.16** | 同上 |
| aiv_vector_stall | 267.29 | 237.39 | −29.90 | VEC 停顿下降（busy 下降伴生） |
| c_scalar_t | 755.06 | 765.17 | +10.11 | **C 侧环耦合**：AIC IR 逐字节而 C 跨核 SEM 阻塞 +10µs ⇒ c_wall +10.15 / v_wall +9.98 |
| aic_cube_t | 166.99 | 167.00 | +0.01 | **AIC 各管零变化（IR 证明的运行时呼应）✅** |
| aic_mte1/fix_t | 205.17/303.37 | 205.00/303.30 | −0.17/−0.07 | 零变化 |
| aic_wait_id2 | 535.83 | 536.64 | +0.81 | 带内 |
| GM_to_L1 | 38.50MB | 38.50MB | 0 | 字节守恒 |
| **AIC 发射当量** | **113.92** | **113.92** | **0.00（逐位）** | C 侧零触碰最强运行时呼应 |

**吸收律核算（case7）**：削减 V 管忙碌 ≈46.2µs（vec −24.8 + mte2 −21.4）→ wall **+9.1µs** ⇒ **转化率 = −19.8%（负）**；削减量全数迁移为暴露等待（wait_id0 +35.0 / wait_id4 +18.2）+ C 侧跨核阻塞（+10.1）。

### case15（次目标，kv_db decode 域）

| 指标 | prod | var | Δ | 判读 |
|------|------|-----|-----|------|
| v_vec_t | 121.13 | 118.57 | −2.56 (−2.1%) | V 侧削减兑现 |
| v_mte2_t | 102.34 | 94.43 | −7.91 (−7.7%) | 同上 |
| v_scalar_t | 219.25 | 214.38 | −4.87 (−2.2%) | 同上 |
| aiv_wait_id0/id4 | 181.12/205.55 | 179.60/204.40 | −1.5/−1.2 | V 余量吸收（200µs 级等待不动） |
| AIV 发射当量 | 37.70 | 37.88 | +0.18 (+0.5%) | 噪声级 |
| C 侧全计数器 | wall 412.31 / scalar 410.40 / mte2 365.11 / mte1 143.33 / fix 96.21 / cube 49.80 / wait2 58.38 | 412.48/410.79/365.41/143.29/96.01/49.80/60.25 | 全带内 | **与 r13 采纳态带内一致 ✅**（412.42/411.03/365.54/143.35/96.32/49.80/66.44；wait2 差 = 跨会话漂移带） |
| AIC 发射当量 | 47.27 | 47.27 | 0.00（逐位） | 零触碰呼应 |

**转化率（case15）**：V 侧削减 ≈15.3µs → wall +0.49µs ns ⇒ **≈零响应（scalar 地板 410.4≈wall 412.3 不动，r13 结论复证）**。

## 6. 采纳/回滚依据对照预登记判据（DESIGN §12 追加节逐字）

| 判据 | 要求 | 实测 | 勾选 |
|------|------|------|------|
| (a) 精度门全绿 | A/B BITEXACT 全矩阵 + 锚 A 9/9 + B 14/14 + stress + l0/l1/coverage/all + smoke | 54/54 BITEXACT + 9/9 + 14/14 + 23 配置 nondet=0 + 10/10+19/19+19P/0M+EXIT=0 + 6/6 | ✅ |
| (b1) case7 aiv_vec_time 显著降 | 预期 −3~8% | −3.82%（648.99→624.22，带内） | ✅ |
| (b2) AIV 发射当量降 | 期望下降（激活证明） | **+1.53%（反向）**——如实报告；激活由 IR 普查（−1 copy/slot×20）+ b1 独立证明；寻址迁移解释在案 | ❌（辅助签名反向，不动摇激活判定） |
| (b3) AIC 侧 IR 20/20 逐字节 | C 侧零触碰机器证明 | C2 语义门 20/20 ✅；C1 字面门 QUALIFIED（UB 视图地址重排，布局审计全过）；AIC 发射当量运行时逐位相同 | ✅（精化口径，偏离已定性审计） |
| (c) wall 门 | ≥1 目标 case 改善 ≥3% ∧ 守卫零回退 | **无任何目标改善**：case7 +1.07%(p=0.001)/case8 +0.42%(p=0.022) 显著劣化；case15/18 ns；守卫零显著回退 | ❌ **FAIL（决定性）** |
| r7-1 条款 | 机制激活但 wall 零响应 → 回滚归档，吸收律新增实例 = 合格产出 | 机制激活（b1 带内 + IR 普查）∧ wall 零/负响应 ⇒ 条款字面命中 | **触发** |

**判决 = ROLLED_BACK_ZERO_WALL**。(c) 为决定性失败项（无改善且主目标显著劣化）；(b2) 辅助签名反向已如实登记（非激活失败——激活证据链独立且更强）。回滚三重验证全过：(i) md5 c316a004 还原（多点核验+diff perf19/perf20 同体）；(ii) IR 还原（case7 dcde9ec1 ∧ case15 3a6bf589 逐字节，ROLLBACK_IR_RESULT: PASS）；(iii) l0 复验 10/10（L0_RB_EXIT=0）。

## 7. 产物路径清单

| 产物 | 路径 |
|------|------|
| 变体（留存，禁采纳/禁入 baseline/禁提交） | `/tmp/opencode/r14_probes/mha_r14x_edit_v1.py`（535b71f0） |
| 生成器 + 日志 | `/tmp/opencode/r14_probes/make_r14x.py` / `make_r14x.log` |
| IR 双臂 dump（20 形×2） | `/tmp/opencode/r14_src_base/` / `/tmp/opencode/r14_src_var/`（含 _manifest.txt） |
| IR 比较器 + 全量审查日志 | `/tmp/opencode/r14_probes/cmp_ir_r14.py` / `cmp_ir_r14.log`（CMP_IR_R14_RESULT: PASS） |
| 门禁日志族 | `/tmp/opencode/r14_probes/r14_{smoke,probe,anchors,anchor_stress,stress_ext,l0,l1,coverage,all}.log` + `gates_runner.log` |
| wall A/B 60 runs + 统计 | `/tmp/opencode/r14_wallab/`（case*_r{1,2,3}_{prod,var}.json + `ab_stats_r14.json`）+ `wall_ab_driver.log` + `ab_stats_r14.{py,log}` |
| op 双臂数据 + 解析 | `/tmp/opencode/r14_base/op/` + `/tmp/opencode/r14_iter1/op/` + `op_{prod,var}_r14.txt` + `op_parsed.json`×2 + `attrib_r14.py` / `parse_op_r14.py` / `op_driver.log` |
| 回滚验证 | `/tmp/opencode/r14_probes/verify_rollback_r14.py` / `rb_verify_ir.log`（PASS）/ `rb_l0.log` |
| 运行驱动 | `/tmp/opencode/r14_probes/run_gates_r14.sh` / `drive_wall_r14.sh` / `drive_op_r14.sh` / `dump_src_r14.py` / `bench_r14.py` / `smoke_r14.py` / `probe_r14_ab.py` |
| 开工快照 | `examples/mha/history_version/mha_impl_r4_perf20.py`（c316a004 同体） |
| 落档 | `perf_tuning/optimization_log.md`（[ORDER-CHECK]/[IMPL-#R14-1]/[SELF-CHECK]/[GATES-#R14-x]/[ANALYSIS-#R14-1]/[RESULT-#R14-1]）+ `perf_tuning/perf_log.md`（r14-1/r14-2 条目）+ `perf_tuning/baseline_r14_iter1.json` + 本判决书 |

## 8. 如实分栏

**已确认事实**：§2/§3/§4/§5 全部测量与机器证明（60/60 wall runs exit 0、md5 多点核验、54/54 BITEXACT、IR 两级门、op 双臂读数、回滚三重验证）。
**推断（已标注）**：aiv_mte2_t −21.4µs 与被删 DataCopy 的 MTE2 归属对应（helper 源码同型=DataCopy 事实支撑，未做管级隔离实验）；AIV 发射当量 +1.53% 的寻址迁移解释；case7 base 隐含 UB 写读 hazard 计入 vec busy 的候选解释（var 的 Cast 源在 WaitFlag<MTE2_V> 后 = 协议更干净）。
**未验证项**：①三候选解释未做分离消融（超本轮授权范围与预算）；②op 模式单发（warmup 0/iters 1，r13 同口径）无重复分布——发射当量 +2.37µs 的单发测量误差带未知（AIC 侧逐位复现提示量化噪声极低）；③case8/case9/case3 等 flat_v bf16 形的 op 签名未单独采集（wall 已覆盖）；④范围 B（框架级）与 §3.1 结构级方向未实测（授权外，留待用户决策）。

## 9. 移交 Orchestrator

- 终态 = c316a004（2240 行）+ 框架 r12 opt-in 态（82e80025/1c3a00ed，零触碰）——**与 r13 采纳态完全同体，cann-bench 复评预期分数不变（72.63/0.96x），提交副本无需同步**（本轮零净变更）。
- `consecutive_no_improvement`: 0 → **1**（r13 采纳后 r14 回滚）。
- design_error = **no**（现设计健康；本臂证明的是 emit_v V 侧指令地板非绑定约束——环耦合 pacing/scalar 地板才是，与 r13/iter26 结论一致）。
- 范围 B（框架级隐式广播/sub-exp 原语）与 §3.1 结构级（decode D=256 手工 mma 化）决策留待用户（任务书授权链条款）；本臂实测数据（V 侧削减 46µs → wall −19.8% 转化）对范围 B 的 EV 预判含义：**任何不改变环耦合结构的 V 侧指令削减，转化率预期 ≤0**——建议范围 B 立项前先过该判据。
