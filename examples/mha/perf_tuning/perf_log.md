# MHA Stage-3 perf_log（跨轮摘要；详细门禁记录见 optimization_log.md）

- perf_session: 1, max_iterations: 10, noise_threshold: 3%
- 计时口径: msprof op_summary Task Duration 中位数（warmup 5 + iters 20），与 cann-bench kernel_details 口径对齐（case1 锚点验证 46.9us）
- 基线（iter1 开始前）: baseline.json / baseline_iter1.json

## Iteration 1 — 2026-09-18T19:45+08:00
- bottleneck_type: sync + vector-instruction-issue（AIC 98.5% 等待 AIV；aiv_scalar_time 49.6% ≈ aiv_vec_time 56.6%；cube 利用率 2%）
- optimization: [#1] causal KV 运行期裁剪（T.serial(0, eff_kv_iters)，右下角对齐公式 mha_sink:268-271，全掩码块跳过=值等价）
- baseline_time: case7 16772.47us / case1 46.60us / case10 441.41us / case14 2842.72us / case20 102.08us
- candidate_time: case7 8960.55us / case1 45.46us / case10 442.09us / case14 2843.61us / case20 101.70us
- improvement: case7 **-46.6%（1.87x）**；case1 -2.4%；非 causal 三项 ±0.4%（无回归，符合预期：非 causal 保持编译期界）
- precision: pass（l0 10/10 + l1 19/19）
- adopted: yes（新基线 = iter1；备份 history_version/mha_impl_s3_perf1.py = 原始版）
- rollback_reason: n/a
- next_hint: 实施 [#2] softmax/rescale 行循环 broadcast 2D 化（打击 aiv_scalar 主构成：3×64 GetValue+PIPE_ALL/iter）

## Iteration 2 — 2026-09-18T20:10+08:00
- bottleneck_type: vector-instruction-issue（aiv_scalar 49.6%：3×64 GetValue+PIPE_ALL 行循环/iter）
- optimization: [#2] softmax sub-m_i / acc_o rescale / 预归一化 div / 终归一化 div 四处行循环 → T.tile.broadcast + 2D tile op（专用 scratch m_i_2d/r_2d）
- baseline_time: case7 8960.55 / case1 45.46 / case10 442.09 / case14 2843.61 / case20 101.70 (us, iter1)
- candidate_time: case7 6062.44 / case1 30.09 / case10 217.19 / case14 1665.76 / case20 57.97 (us)
- improvement: case7 -32.3% / case1 -33.8% / case10 -50.9% / case14 -41.4% / case20 -43.0%
- precision: 首版 fail（scratch 与跨核 copy 目标别名 → AUTO pass 重排）→ 修复（专用 buffer）→ pass（l0 10/10 + l1 19/19）
- adopted: yes（备份 history_version/mha_impl_s3_perf2.py = pre-#2）
- rollback_reason: n/a（修复后采纳；根因与教训已记录 optimization_log.md 迭代 2）
- next_hint: [#3] 掩码链 2D 化 + row_thresh hoist（causal/尾钳位变体每迭代 ~128 条微指令 → ~5 条 2D 指令；注意 mask scratch 同样不得别名跨核 copy 目标）

## Iteration 3 — 2026-09-18T20:45+08:00
- bottleneck_type: vector-instruction-issue（掩码链 64×(CompareScalar+Select+2barrier)/iter）+ 冗余指令（Fill+Add）
- optimization: [#7] Fill+Add→Copy（acc_s_ub_ 退役，释放 16KB UB，单独性能中性）+ [#3] 掩码链 2D 化（AP+2×broadcast+compare+select ≈5 条 2D 指令替代 128 条微指令；row_thresh hoist；uint8 bit-packed mask）
- baseline_time: case7 6062.44 / case1 30.09 / case10 217.19 / case14 1665.76 / case20 57.97 (us, iter2)
- candidate_time: case7 3554.73 / case1 18.29 / case10 217.06 / case14 1668.50 / case20 58.50 (us)
- improvement: case7 -41.3% / case1 -39.2% / 非 causal ±0.3%（零改动路径无回归）
- precision: pass（l0 10/10 + l1 19/19，含 (T,F)/(F,T) 掩码变体全覆盖）
- adopted: yes（备份 perf3 = post-#2）
- rollback_reason: n/a
- next_hint: [#6] decode 窄块 block_M=16（S≤16 编译期分支；case10/14 预期 ~2-3x；重排序理由：#4 运行期跳过在 #3 后价值缩水至 ~5-8% 且需 Developer 运行期 if 验证，#6 期望值更高且独立）

## Iteration 4 — 2026-09-18T21:00+08:00
- bottleneck_type: vector-work（decode valid_m=1 时 63/64 行为 padding 空转）
- optimization: [#6] decode 窄块：S<=16 编译期分支 block_M=16（向量工作量/每块 CV workspace 往返 4x↓；host 无 padding——运行期 valid_m 语义复用）
- baseline_time: case10 217.06 / case14 1668.50 / case7 3554.73 / case1 18.29 / case20 58.50 (us, iter3)
- candidate_time: case10 148.41 / case14 1242.86 / case16 2234.72(首测) / case7 3554.71 / case1 18.50 / case20 58.18 (us)
- improvement: case10 -31.6% / case14 -25.5% / prefill ±1.1%（零回归）
- precision: pass（l0 10/10 + l1 19/19 + 13 shape 位级等价 probe BITEXACT，含 S=17 分支边界）
- adopted: yes（备份 perf4 = pre-#6）
- rollback_reason: n/a
- next_hint: [#5] Fixed Core（core_num=min(block_num,20) 连续 chunk 任务循环；预期 case14 ~17% 槽间隙 + case7 ~13% per-block 开销 + workspace 114MB→1.1MB L2 常驻；UNKNOWN 维度=AUTO_CV_COMBINE×嵌套运行期任务循环，需编译+probe 验证）

## Iteration 5 — 2026-09-18T21:30+08:00
- bottleneck_type: per-block 固定开销（TPipe/workspace init × block_num）+ workspace GM 足迹
- optimization: [#5] Fixed Core：T.Kernel(min(block_num,20)) + q_tasks/r_tasks 均衡任务循环（count 形式）；buffer/TPipe 一次初始化；workspace 变核索引（case7 114MB→1.1MB L2 常驻）
- baseline_time: case7 3554.71 / case1 18.50 / case10 148.41 / case14 1242.86 / case20 58.18 / case16 2234.72 (us, iter4)
- candidate_time: case7 3615.87 / case1 15.35 / case10 151.46 / case14 1009.90 / case20 54.14 / case16 2146.01 (us)
- improvement: case14 -18.7% / case1 -17.0% / case20 -6.9% / case16 -4.0% / case7 +1.7%、case10 +2.1%（噪声带内回退，长任务形态 FC 无调度空隙可填）；geomean 加速比 +9%
- precision: pass（FC 等价 probe 12/12 BITEXACT + l0 10/10 + l1 19/19）
- adopted: yes（备份 perf5 = pre-#5）
- rollback_reason: n/a（首版 T.serial(my_start,my_end) 形式触发 TIR 分析器 CV 复制 range 冲突 → 改 count 形式修复，见 optimization_log 迭代 5）
- next_hint: [#4] 全可见迭代掩码运行期跳过（case7 每 s_block 仅对角迭代需掩码，其余 bx/(bx+1) 可跳；需先最小复现验证 Developer cv_1_1 运行期标量 if 包 V 块）

## Iteration 6 — 2026-09-18T21:55+08:00
- bottleneck_type: vector-instruction-issue（全可见迭代仍执行 ~5 条 2D 掩码指令 + barrier 链）
- optimization: [#4] 运行期标量条件跳过全可见块掩码链（条件 j0+bN-1 > min(s0+offset_eff,S_kv-1)，纯标量无 UB 读；trim 协同后仅对角/尾块命中）
- baseline_time: case7 3615.87 / case1 15.35 / case16 2146.01 / case20 54.14 / case14 1009.90 (us, iter5)
- candidate_time: case7 3177.05 / case1 14.80 / case16 1950.10 / case20 54.32 / case14 1011.28 (us)
- improvement: case7 -12.1% / case16 -9.1% / case1 -3.6% / 非掩码 ±0.3%
- precision: pass（probe 12/12 BITEXACT + l0 10/10 + l1 19/19）
- adopted: yes（备份 perf6 = pre-#4）
- rollback_reason: n/a
- next_hint: 重 profile case7/case10 定剩余瓶颈；候选 [#9-lite] K/V 手动双缓冲（AUTO_SYNC=True 依赖驱动形态优先验证；预期 5-10%）或收口验证

## Iteration 7 — 2026-09-18T22:25+08:00
- bottleneck_type: sync/per-iteration 固定成本（decode：aiv_scalar 指令下发 ~1.0us/iter 与 tile 宽度无关 → 减迭代数）
- optimization: [#8d] decode 分支（S<=16）block_N=128：KV 迭代数减半（CV 往返/cross-flag/barrier 链减半）；case14 类 S_kv<=128 落入单块特化；prefill 分支 bN=64 零改动（全局 bN=128 因 UB 242KB>192KB 否决，见 optimization_log 迭代 7）
- baseline_time: case10 151.46 / case14 1011.28 / case16 1950.10 / case7 3177.05 / case1 14.80 / case20 54.32 (us, iter6)
- candidate_time: case10 92.92 / case14 667.15 / case16 1247.04 / case7 3208.81 / case1 15.04 / case20 54.76 (us)
- improvement: case10 -38.7% / case14 -34.0% / case16 -36.1% / prefill ±1.6%（BITEXACT 证实为噪声）
- precision: pass（prefill 4/4 BITEXACT；decode 8/8 阻塞口径过、余量 80-320x；l0 10/10 + l1 19/19）
- adopted: yes（备份 perf7 = pre-#8d）
- rollback_reason: n/a
- next_hint: [#9-lite] K/V 双缓冲预取先做 scratch 最小复现（R9 同族 codegen 风险：cv_1_1 双缓冲 L1 gemm 操作数）；不可行则收口（--level all + cann-bench 复评 + final_report）

## Iteration 8 — 2026-09-18T22:50+08:00
- bottleneck_type: n/a（候选评估轮）
- optimization: [#9] K/V 手动双缓冲预取——scratch 最小复现两变体证明 AUTO_SYNC=True 下零重叠（Variant A: PIPE_ALL 全排空；Variant B: M_MTE2/MTE2_M 成对全串行），Developer 边界内不可行；Expert 化（AUTO_SYNC=False+手动 flag）越出 DESIGN §7.1/§8.6 且预期收益 ~5-10% 与剩余预算不匹配 → 不实施，证据与 DB-ANALYSIS 已记录 optimization_log.md
- baseline_time / candidate_time: n/a（mha.py 未改动，基线保持 iter7）
- improvement: n/a
- precision: n/a（无改动；iter7 状态即当前状态）
- adopted: no（评估性结论，非回滚——无修改发生）
- rollback_reason: n/a
- next_hint: 收口——ruff 格式门禁 + `--level all` 全量验证（L0/L1/L2/Boundary/哨兵）+ final_report.md

## Iteration 9（收口） — 2026-09-18T22:15+08:00
- bottleneck_type: n/a（验证与收口轮，无 kernel 功能改动）
- optimization: 收口三件事——①mha.py 模块 docstring 同步 Stage-3 实况（注释-only，ruff clean，smoke 编译+运行 ✓，提交副本再同步+pip 重装 ✓）；②`test_mha.py --level all` 全量验证 exit 0 "Test Passed!"（L0 10/10 + L1 19/19 + L2 12/12 + Boundary 6/6 + 哨兵 4/4 PERF_PASS，kernel_path 6.1~7.1s ≤120s）；③cann-bench 官方复评（run_evaluation.sh --task-dir tasks/level4/mha --operator MHA）
- cann-bench 复评（cann_final_eval_20260918_220619）: 20/20 pass，综合 54.18→**61.76**，性能分 4.18→**11.76**，平均加速比 0.13x→**0.42x**（geomean 0.104x→0.369x）；20/20 case 全部提升 1.84x~6.24x，无一回退；最佳 case16 6.24x / case17 6.05x / case7 5.18x；case14 达 0.781x 逼近 baseline
- baseline_time / candidate_time: 见 final_report.md §1/§3 全表
- improvement: 会话累计（本地 msprof 代表 shape）case7 5.2x / case1 3.1x / case10 4.7x / case14 4.3x / case20 1.9x / case16 6.3x
- precision: pass（全量套件 + cann-bench 独立口径双确认；精度红线四项全保留）
- adopted: yes（最终交付 = perf8 备份 + docstring 更新）
- rollback_reason: n/a
- next_hint: 剩余差距主体为 Developer 模式结构性同步成本（AUTO_SYNC 逐 op barrier + 每迭代 3 段 CV workspace GM 往返）——逼近 baseline 需 Expert 化批量流水（flash_attn_optimize §1-§8）或 decode 专用 kernel（heads-as-M/split-KV），均属设计模式变更，建议独立立项由需求方决策（final_report §5）

---
# 会话 2（perf_session=2，2026-09-19；比较基线 = baseline_iter10.json = 会话 1 终态复测）

## Iteration 10 — 2026-09-19T14:55+08:00
- bottleneck_type: sync/per-iteration 固定成本（decode 每迭代 ~1.85us 固定 + CV 往返/cross-flag/barrier 链 × 迭代数）
- optimization: [#10a] decode 分支 bN=256 自适应（S<=16 且 D<=128 且 S_kv>128；bN=512 与 D=256 扩宽被 gemm_v0 transpose_B L0B 预算硬限否决——common.h:1278 static_assert 实证）
- baseline_time: case10 92.28 / case16 1266.42 / case14 672.63 / case20 54.67 / case1 14.86 (us, s2 基线)
- candidate_time: case10 65.68 / case16 932.84 / case14 672.08 / case20 54.81 / case1 14.92 (us)
- improvement: case10 -28.8% / case16 -26.3% / 守卫三项 ±0.4%（零变化路径）
- precision: pass（probe 6/6 BITEXACT + 10/10 阻塞口径余量 40-320x + l0 10/10 + l1 19/19）
- adopted: yes（备份 perf9 = 会话 1 终态）
- rollback_reason: n/a
- next_hint: [#10b] D=256 decode bN=256 via GEMM1 kL0Size=64（ascend.py L418 advanced 参数，L0B ping-pong 32KB=256×64×2 恰好）；随后 [#11] prefill bN=128（#10a 已证 GEMM1 L0B 无障碍，仅剩 UB 预算问题）

## Iteration 11 — 2026-09-19T15:35+08:00
- bottleneck_type: sync/per-iteration 固定成本（D=256 decode 4 iters→2 iters）
- optimization: [#10b] D=256 decode bN=256：GEMM1 传 kL0Size=64（ascend.py advanced 参数；4 K-steps ping-pong 32KB=256×64×2 恰好；K 累加序不变）；其余配置 default 128 零变化
- baseline_time: case15 755.44 / case18 756.42 / case4 65.18 / case10 65.68 (us, post-#10a)
- candidate_time: case15 583.76 / case18 583.94 / case4 64.99 / case10 65.71 (us)
- improvement: case15 -22.7% / case18 -22.8% / 守卫两项 ±0.3%
- precision: pass（probe 6/6 BITEXACT + 8/8 阻塞口径余量 160-640x + l0 10/10 + l1 19/19）
- adopted: yes（备份 perf10 = post-#10a）
- rollback_reason: n/a
- next_hint: [#11] prefill bN=128——#10a 已证 GEMM1 L0B 对 prefill bN=128 无障碍（nTile=128×128×2=32KB≤64KB），唯一约束是 UB ~227KB(D=128)；probe 实测 planner 行为后决定手术组合（S2 m_i_2d∪r_2d 合并 / S1 fp16 掩码）

## Iteration 12 — 2026-09-19T16:20+08:00
- bottleneck_type: sync/per-iteration 固定成本（prefill 迭代数减半 = CV 往返/cross-flag/barrier 链减半）
- optimization: [#11] prefill bN=128（全 D 放行）——会话 1 #8 的 UB 否决被实测推翻：MEMORY_PLANNING 实配 153.0-186.8KB ≤191.75KB（手算 naive 227KB 高估 ~48KB），零手术纯谓词翻转
- baseline_time: case7 3202.04 / case20 54.81 / case3 471.18 / case4 64.99 / case1 14.92 / case10 65.71 (us, post-#10b)
- candidate_time: case7 2234.75 / case20 38.26 / case3 352.77 / case4 52.47 / case1 15.21 / case10 65.27 (us)
- improvement: case7 -30.2% / case20 -30.2% / case3 -25.1% / case4 -19.3% / case1 +1.9%（噪声带内，如实记录）/ case10 -0.7%（守卫）
- precision: pass（probe 3/3 decode BITEXACT + 5/5 单块化 + 8/8 多块阻塞口径；l0 10/10 + l1 19/19）
- adopted: yes（备份 perf11 = post-#10b，md5 校验）
- rollback_reason: n/a
- next_hint: [#13] decode bN=512 via GEMM1 kL0Size=32（512×32×2=32768 ≤ 4-step ping-pong 32KB 恰好；L1 276KB/UB~141KB 手算 ✓）——case10 4 iters 再减半；[#12] 反模式复扫 + 收口

## Iteration 13 — 2026-09-19T17:05+08:00
- bottleneck_type: sync/per-iteration 固定成本（decode 迭代再减半 + Skv≤512 单块化）
- optimization: [#13] decode bN=512（D≤128 且 S_kv>256；GEMM1 kL0Size=32 → 512×32×2=32768 恰好填满 4-step ping-pong L0B 32KB；D=256 被 L1 536KB>511.75KB 排除保持 bN=256）
- baseline_time: case10 65.27 / case16 932.84 / case14 672.08 / case7 2234.75 (us, post-#11)
- candidate_time: case10 51.04 / case16 853.82 / case14 675.54 / case7 2260.09 (us)
- improvement: case10 -21.8%（累计 -44.7%）/ case16 -8.5%（累计 -32.6%）/ 守卫两项 +0.5~1.1%（BITEXACT 噪声）
- precision: pass（probe 6/6 BITEXACT + 11/11 阻塞口径余量 160-640x + Skv=300 GOLDEN-BITEXACT + l0 10/10 + l1 19/19）
- adopted: yes（备份 perf12 = post-#11）
- rollback_reason: n/a
- next_hint: [#12] 反模式全清单复扫（终态代码）→ 收口（--level all + cann-bench 复评 + final_report/docstring 同步）

## Iteration 14 — 2026-09-19T17:40+08:00
- bottleneck_type: n/a（静态复扫轮，零代码改动）
- optimization: [#12] antipatterns 全 11 项复扫（终态 414 行）：零新可修复项；tile size 已探至 L0B/L1/UB 三面硬限；AUTO_SYNC barrier 链与 CV workspace GM 往返为有意保留（Developer 边界，证据链见 optimization_log 迭代 14）；prefill bM=32+bN=256 量化后跳过（slot-iter -6.7% vs KV 重读 ×2）
- baseline_time / candidate_time: n/a（无改动，基线保持 s2_iter13）
- improvement: n/a
- precision: n/a（零改动）
- adopted: n/a（评估性结论）
- rollback_reason: n/a
- next_hint: 收口——docstring/注释同步 + --level all + 提交副本同步 + cann-bench 复评 + final_report

## Iteration 15（收口） — 2026-09-19T18:20+08:00
- bottleneck_type: n/a（验证与收口轮，kernel 功能零改动）
- optimization: 收口五件事——①mha.py docstring 同步会话 2 实况 + gemm1_kL0 过时注释修正（注释-only，ruff clean，smoke 编译+运行 ✓）；②`--level all` exit 0 "Test Passed!"（L0 10/10 + L1 19/19 + L2 12/12 + Boundary 6/6 + 哨兵 4/4 PERF_PASS，kernel_path 6.1~7.2s）；③提交副本同步 + pip 重装（installed md5 校验一致）；④cann-bench 官方复评；⑤final_report.md 会话 2 章节
- cann-bench 复评（cann_final_eval_20260919_154250）: 20/20 pass，综合 61.76→**65.05**，性能分 11.76→**15.05**，平均加速比 0.42x→**0.55x**；19/20 case 提升（最佳 case10 1.65x/0.85、case11 1.72x/0.77、case13 1.65x；case1 0.94x 噪声带微降已如实记录）；case7/8 1.43x、case9 1.44x
- baseline_time / candidate_time: 见 final_report.md 会话 2 §S2-1/§S2-2 全表
- improvement: 会话 2 累计（本地 msprof）case10 -44.7% / case16 -32.6% / case7 -29.4% / case20 -30.0% / case3 -25.1% / case15 -22.7% / case18 -22.8% / case4 -19.5% / case14 +0.4%(守卫) / case1 +2.4%(噪声带)
- precision: pass（全量套件 + 每优化项 probe + cann-bench 独立口径三确认；精度红线四项全保留且单块特化覆盖面扩展）
- adopted: yes（最终交付 = 当前 mha.py；备份链 perf9..12 完整可回溯）
- rollback_reason: n/a
- next_hint: Developer 模式常规调优已收口（tile 尺寸触 L0B/L1/UB 三面硬限、迭代数已裁剪+单块化、指令已 2D 化、核映射已 Fixed Core）；剩余 case7/8/9 差距主体 = AUTO_SYNC barrier 链 + CV workspace GM 往返结构成本，逼近 baseline 需 Expert 化批量流水（flash_attn §1-§8）= 设计模式变更，建议独立立项由需求方决策（final_report §S2-5）

## Session-3 Iteration r3-1 — 2026-09-20T00:20+08:00
- bottleneck_type: sync（case14 协议延迟：pipe 全空转 <40%，C↔V 乒乓 6 SEM 往返/任务 × 205 任务/核）
- optimization: #R3-1 Form B single_kv_block 任务批处理（NUM_STAGES=8 任务/协议轮，槽↔任务映射，C/V 同式批计数）
- baseline_time: case14 746.08us / case16 763.42us
- candidate_time: case14 530.73us / case16 646.84us
- improvement: case14 -28.9% / case16 -15.3%（守卫 case7/9/10 噪声带内）
- precision: pass（probe 9/9 BITEXACT+golden / 守卫 5/5 BITEXACT / l0 10/10 / 位级锚 10/10 / 压力 nondet=0）
- adopted: yes
- rollback_reason: n/a
- next_hint: #R3-2 Form A k/v L1 双缓冲（case7 AIC wait_id2=606us 证据）；case14 加速比恢复 ≈1.0x 已超 0.79x 目标

## Session-3 Iteration r3-2 — 2026-09-20T00:40+08:00
- bottleneck_type: transfer（假设：V 侧 vector/DMA 工作量）——**假设被证伪**
- optimization: #R3-3 w1 dtype 瘦身 fp32→dtype（r16(qk) 落点迁移 fixpipe cast）
- baseline_time: case7 927.11us（post-#R3-1）
- candidate_time: case7 935.99us
- improvement: +1.0%（7 shape 全部 ±3% 噪声带内，无可证实收益）
- precision: pass（RNE 等价 11/11 BITEXACT + l0 10/10 + 锚 10/10 BITEXACT + 压力 PASS）
- adopted: no
- rollback_reason: 性能无变化规则（<3%）→ 回滚至 r3_perf2（md5 1d3a261c 核验）
- next_hint: 归因修正——case7 wall 非 V 指令数约束；wait_id2=606us 大头判读为 C 侧 K/V 矩阵 GM→L1 单缓冲装载链停顿 → #R3-2 k/v L1 双缓冲升为主攻；fixpipe RNE==vcast RNE 知识已固化可复用

## Session-3 Iteration r3-3 — 2026-09-20T01:20+08:00
- bottleneck_type: transfer（假设：C 侧 K/V GM→L1 单缓冲装载链）——**假设被证伪**
- optimization: #R3-2 Form A k/v L1 双缓冲（批内预取 + 批首预载）
- baseline_time: case7 927.11us
- candidate_time: case7 931.83us
- improvement: +0.5%（case20 -2.2%/case1 +0.2% 均噪声带内）
- precision: pass（probe 10/10 BITEXACT + l0 10/10 + 锚 10/10 + 压力 PASS）
- adopted: no
- rollback_reason: 性能无变化规则 → 回滚至 r3_perf3（md5 核验）
- next_hint: 两轮证伪收敛 → 主嫌疑 = 逐迭代 SEM 会合延迟链；#R3-4 CROSS_INTERVAL=2 为判决性实验（连续无提升计数 2/3，本项若再无提升触发中止条款②）

## Session-3 Iteration r3-4 — 2026-09-20T01:50+08:00
- bottleneck_type: sync（假设：逐迭代 SEM 会合延迟链）——**假设被证伪（判决性实验）**
- optimization: #R3-4 CROSS_INTERVAL 1→2
- baseline_time: case7 927.11us / case9 1380.34us / case14 530.73us
- candidate_time: case7 917.42us / case9 1413.32us / case14 527.34us
- improvement: case7 -1.0%、case9 +2.4%、case14 -0.6%、case16 -1.0%、case10 +2.6%、case1 +1.3%（全部噪声带内，case9/10 方向偏劣）
- precision: pass（probe 8/8 BITEXACT + l0 10/10 + 锚 10/10 + 压力 PASS）
- adopted: no
- rollback_reason: 性能无变化 + guide §3.2 无收益立即回退 → CROSS_INTERVAL 还原为 1（md5 核验 == r3_perf3）
- next_hint: **连续 3 次无提升（r3-2/r3-3/r3-4）触发中止条款②** → 收口：--level all 全量 + 提交副本同步 + cann-bench 权威复评（对照 67.88/0.68x，验证 #R3-1 收益）+ final_report 会话 3 章节。case7 剩余差距归因 = 每迭代固定结构成本（scalar 发射链 + ws 往返不可重叠部分），单点手段穷尽，需迭代数削减或设计级变更（[遗留] 提交 orchestrator/用户决策）

## Iteration r4-1 — 2026-09-20T10:45+08:00 (perf_session=4)
- bottleneck_type: sync (Form B PipeBarrier<PIPE_ALL> ×7-8/kv-iter serialization; AIC pipe busy-sum ≈95% zero overlap)
- optimization: #R4-1 Form B barrier_all pure removal (15 sites; gemm_v0 self-balanced prologue/epilogue pipe flags + pipe-scoped SEM = complete hazard coverage, S4-Step 2 precheck; zero new user flags)
- baseline_time: case9 1379.56us / case14 530.73 / case16 646.84 / case11 349.63 / case10 39.90 (msprof median)
- candidate_time: case9 1020.73 / case14 378.16 / case16 565.27 / case11 312.34 / case10 35.34
- improvement: case9 -26.0% / case14 -28.7% / case16 -12.6% / case11 -10.7% / case10 -11.4%; Form A guards ±1.5% noise band
- precision: pass (probe 15/15 BITEXACT + l0 10/10 + anchor 10/10 + stress nondet=0 + l1 19/19)
- adopted: yes
- rollback_reason: n/a
- next_hint: case9 0.331x < 0.4x M1 line — re-profile case9 (msprof op) for remaining bottleneck → #R4-2 (k/v L1 DB / GEMM interleaving) decision; then #R4-3 flat decode batching

## Iteration r4-2 — 2026-09-20T11:10+08:00 (perf_session=4)
- bottleneck_type: compute/sync (case9 post-#R4-1: gemm_v0 internal step-scalar chain, AIC scalar ~93% busy, mte1_stall ~78% wall)
- optimization: #R4-2a prefill D=256 GEMM1 kL0Size 128->256 (single K-step, 4->3 steps/iter)
- baseline_time: case9 1020.73us
- candidate_time: case9 1058.92us
- improvement: +3.7% (regression beyond noise band)
- precision: pass (probe_r42 12/12 BITEXACT + l0 10/10 + anchor 10/10 + stress PASS — single mma k=256 == chained 2x k=128 bitwise)
- adopted: no
- rollback_reason: performance regression; kNumSteps==1 disables L0A/L0B ping-pong (common.h L1322-1327) — serial 80KB MTE1 + mma(k=256) loses more overlap than the saved step overhead. Restored r3_perf6 (md5 e80f7f65).
- next_hint: gemm_v0 step count is NOT the marginal constraint; MTE1/mma overlap is. case9 lever must hide MTE2 loads (k/v L1 DB) or go manual-mma (design-level, banned) → #R4-2b

## Iteration r4-3 — 2026-09-20T11:55+08:00 (perf_session=4)
- bottleneck_type: compute/sync (case9: gemm_v0 internal per-step scalar chain — MTE1+M+PipeBarrier<PIPE_M> x4 steps/iter)
- optimization: #R4-2b Form B prefill-D256 k/v L1 double buffer (prefetch-style, zero new flags, 3D unannotated L1 + gemm_v0 runtime slices)
- baseline_time: case9 1020.73us
- candidate_time: case9 1029.77us
- improvement: +0.9% (noise band, no change; op-profile: all pipe readings identical — MTE2 exposure absorbed, not on critical path)
- precision: pass (mech probe BITEXACT + probe_r43 15/15 BITEXACT + l0 10/10 + anchor 10/10 + stress PASS)
- adopted: no
- rollback_reason: no-improvement rule; restored r3_perf7 (e80f7f65). case9 lever evidence tripled -> only manual-mma (§10.3-5, design-level, banned) can break the gemm_v0 step chain; rev4 evidence banked.
- next_hint: #R4-3 flat (task,kv) decode batching (avg-margin lever, case14-style protocol amortization for case10/11/15/18)

## Iteration r4-4 — 2026-09-20T12:40+08:00 (perf_session=4)
- bottleneck_type: sync/protocol (hypothesis) -> DISPROVEN: post-#R4-1 per-round SEM overhead collapsed to ~0.2-0.3us; decode multi-kv wall = per-slot MTE2 bandwidth (5.3-5.9us/slot, K+V 256-272KB)
- optimization: #R4-3 flat (task,kv) decode batching (rounds: case11 26->7, case15/18 48->12, case10 2->1)
- baseline_time: case11 312.34 / case15 504.19 / case18 504.53 / case10 35.34 (local r3_perf8 msprof; case15/18 fresh補测)
- candidate_time: case11 307.02 / case15 505.24 / case18 504.93 / case10 35.12
- improvement: -1.7% / +0.2% / +0.1% / -0.6% — ALL noise band
- precision: pass (probe_r44 17/17 BITEXACT + l0 10/10 + anchor 10/10 + stress PASS)
- adopted: no
- rollback_reason: no-improvement rule (3rd consecutive -> abort clause 2 triggered); restored r3_perf8 (e80f7f65). #R3-1 gain model non-additivity: barrier removal already absorbed the protocol tax.
- next_hint: session closing — terminal state = #R4-1 adopted (e80f7f65); case9 0.331x floor evidence-banked for rev4 (§10.3-5 manual-mma design-level)

## Iteration r4-5 (closing) — 2026-09-20T14:30+08:00 (perf_session=4)
- abort: clause 2 (3 consecutive no-improvement: r4-2a regression / r4-2b no-change / r4-3 no-change); iterations 4/10 used; adopted 1 (#R4-1)
- terminal state: e80f7f65 (947 lines) = #R4-1 only; --level all exit 0 (L0 10/10 L1 19/19 L2 12/12 Boundary 6/6 sentinels 4/4) + anchor 10/10 BITEXACT + stress nondet=0 (fresh on terminal)
- cann-bench cann_final_eval_20260920_141424: composite 68.50->70.21, perf 18.50->20.21, avg speedup 0.7017->0.7813 (+11.3%), 20/20 pass
- M1 verdict: PARTIAL (avg 0.7813>=0.75 ACHIEVED; case9 0.324<0.4 NOT — in-rules floor, design-level §10.3-5 required, rev4 evidence banked)
- decode no-regression: ALL PASS (case10 0.998 / case14 1.394 / case20 1.013 / case13 0.883 / case16 0.784); case7/8/9 monotonic
- fallback clause: NOT triggered (70.21 > 65.05)

---

# perf_session=5 (2026-09-21, Expert rev3 track round 3; soft budget 3-4 iterations; backup chain continues r3_perf{N} N>=9)

> Trigger: user observation on the case1 camodel pipeline plot ("C-core wait clearly longer than V-core; V-core MTE and VECTOR execute serially; VECTOR compute over-fragmented"), mode=implement.
> Acceptance anchored on the single-block Form A class (case1/6/19/20); guards = Form A multi-kv generic (case7/2/5) + all Form B (case9/10/14/16).

## Iteration r5-1 — 2026-09-21T13:30+08:00 (perf_session=5)
- bottleneck_type: sync (cross-core / inter-phase dependency-chain serialization: kv_iters==1 collapsed eff_num_outer AND batch_iters to 1, so the NUM_STAGES=8 pipeline + io_buf[2] ping-pong never had a 2nd slot in flight; every task ran a strictly serial C-QK -> V-softmax -> C-PV -> V-epi chain, x my_count tasks with zero inter-task overlap)
- optimization: #R5-1 Form A single_kv_block TASK batching — emit_v dispatch gate `single_kv_block and not form_a` -> `single_kv_block` (1 line; the batched V body already existed and was already bit-validated by Form B cases 12/13/14/16/17) + new Form A C-scope `if single_kv_block:` batch mirror (117 lines; template = Form B batch branch, body = existing Form A GEMM1/GEMM2 batches carried over op-for-op, kv index -> task index). Two Form-A-specific deltas, both pure Q scheduling: Q GM->L1 moves inside the slot loop, and q_l1->l0a_s[side] is issued every slot (generic guards it with `if i<2` because Q is kv-invariant there); SIG_Q_L1 return moves to right after that single MTE1 read (per-slot token balance identical to the generic per-task one).
- baseline_time: case1 13.03 / case6 54.08 / case19 35.86 / case20 30.71 us (msprof median; case6/case19 first ever measured) | guards case7 926.43 / case2 67.39 / case5 18.44 / case9 1018.49 / case14 377.42 / case10 34.41 / case16 568.37
- candidate_time: case1 11.56 / case6 30.07 / case19 23.38 / case20 18.28 | guards case7 928.35 / case2 66.81 / case5 18.47 / case9 1025.54 / case14 377.48 / case10 35.14 / case16 562.44
- improvement: **case1 -11.28% / case6 -44.40% / case19 -34.80% / case20 -40.48%** (4/4 targets far beyond the 3% noise band; case6 with 12.8 task/core -> batches 8+5 / 8+4 is the biggest winner, as predicted). Guards 7/7 inside +/-3% (max +2.12%) **and 12/12 guard shapes byte-identical generated source** -> zero regression proven at IR level.
- precision: pass (probe_r51 **30/30** = 19 batched-path BITEXACT vs pre-change + golden blocking gate all pass (worst MARE_abs 1.95e-3 vs gate 7.81e-2) + 11 guards BITEXACT; batch_tasks coverage {1,2,3,4,5,6,7,8} odd+even, num_tb {1,2} incl. 8+5/8+4/8+1 cross-round, P1 split + P2 shared pool, fp16/bf16, causal/non-causal, S tail, S_kv tail, both tails, cross S!=S_kv, zeros, explicit scale, large value range | l0 10/10 | l1 19/19 | coverage 19 PASS/0 MISS/0 N-A | `--level all` exit 0 [L2 12/12 rejected, Boundary 6/6 0-WARN, sentinels 4/4 PERF_PASS] | anchor **10/10 BITEXACT** vs Developer final, fresh re-run on terminal state | R11 stress inherited 4 configs nondet=0 **plus a NEW extended 7-config stress aimed at the changed path** nondet=0 — the inherited 4 configs all land on generic/Form-B paths and therefore did NOT cover the new branch)
- camodel mechanism re-collection (new acceptance item this session; collection method RECOVERED + reproduced, see simulator_r5/README.md): (1) C wait 7.726->6.564us (-15.0%), V wait 5.876->3.705us (-37.0%), C wall -12.9%, V wall -11.9% — honest caveat: C's wait RATIO stays ~83% because C is structurally the lighter core here (CUBE 0.322 + MTE1 0.502 + FIXP 2.108us vs V VECTOR 3.368us); what was removed is the wait being ON the critical chain. (2) **MTE||VECTOR overlap 0.1297->0.5146us (+296.7%) = 5.30%->20.93% of MTE (x3.95) / 3.83%->15.28% of VECTOR (x3.99) -> io_buf ping-pong activated**. (3) VECTOR segments 11->10 and the **inter-softmax gap collapsed 2.684->0.582us (-78.3%)**, its content changing from "wait for the peer core's PV+epi phase" to "own necessary DMA". NOT achieved, recorded honestly: micro-fragment count essentially unchanged (6->6 @<60ns, 8->8 @<100ns) — that is BAR/prologue-epilogue bookkeeping inherent to the flag-managed copy pattern, and [#R3-4] already proved cutting such instruction counts does not shrink wall. Expected cost of genuine overlap: each softmax segment +1.0%/+4.6% (UB bandwidth contention); net wall still -12%.
- adopted: **yes** (new baseline msprof_r5_iter1; md5 3d56c5dd3d8a60d26eed7ade2e404a2b, 1086 lines)
- rollback_reason: n/a
- next_hint: #R5-3 (K/V L1 residency for same-(b,h) adjacent tasks) is now the best-supported next candidate and its case got STRONGER: camodel shows C MTE2 union rose 3.055->3.961us (33.6%->50.0% of C wall) because the batched GEMM1 now issues Q1,K1,Q2,K2 back-to-back, so the redundant same-byte K re-read (625ns on core0, GM src 0x11351c00 identical for both tasks) now sits INSIDE the pre-softmax1 burst that gates w1[1]. For case6 the win is structurally larger: s_blocks=8 and tasks are contiguous, so all 8 slots of a batch share ONE (bz,by) => 7 of 8 K loads and 7 of 8 V loads are redundant. Precedent [#R3-2] (runtime-guarded copy, 10/10 BITEXACT); keep the SIG_K_L1/SIG_V_L1 wait+set pair OUTSIDE the condition so token accounting is untouched.

## Iteration r5-2 — 2026-09-21T14:20+08:00 (perf_session=5)
- bottleneck_type: transfer (post-#R5-1 the largest real-work pipe on either core is C-side MTE2 = 3.961us = 50.0% of C wall; its single biggest segment 3050.8ns = Q1,K1,Q2,K2 back-to-back, of which K2/V2 are provably byte-identical re-reads of the same GM addresses)
- optimization: #R5-3 same-(b,h) K/V L1 residency — wrap the batched Form A GEMM1 `T.copy(K->k_l1)` and GEMM2 `T.copy(V->v_l1)` in the runtime guard `T.And(task > my_start, bx != 0)` (= "task-1 was processed by this core AND is in the same (bz,by) group", since `task//s_blocks == bz*N+by`), keeping the SIG_K_L1/SIG_V_L1 wait+set pair OUTSIDE the guard so token accounting is untouched.
- baseline_time: (not reached — precision gate failed before msprof) #R5-1 terminal: case1 11.56 / case6 30.07 / case19 23.38 / case20 18.28 us
- candidate_time: n/a
- improvement: n/a
- precision: **FAIL** — non-deterministic data corruption. Probe 6/6 first changed-path shapes DIFF vs #R5-1 (max|d| 3.1e-01~4.5e+00, golden matched down to 0.376). Corruption is NON-DETERMINISTIC (3/64 vs 24/64 bad (b,n) slices on the identical zeros input across two runs) and includes slices with NO skip slot at all. Four-experiment narrowing: (a) an always-true runtime guard (`task >= my_start`, identical code shape) is 3/3 BITEXACT -> the runtime-if-around-GM->L1 mechanism is sound; (b) the predicate is hand-verified AND matches the generated code (`my_start < task && 0 < task % s_blocks`), all SetFlag/WaitFlag stay outside the branch, L1 tiles do not alias (q@0/k@8192/v@24576/p@40960); (c) K-only variant also fails 6/6 on non-zero data (the earlier "case20 BITEXACT with K-only" was a FALSE NEGATIVE - all-zero K makes a wrongly-skipped load invisible); (d) failure is independent of P1/P2 pool class and of K vs V side. => eliding a redundant GM->L1 DMA inside a manual pipe-pair flag protocol is itself unsafe here. Root cause NOT settled within budget; leading unverified hypothesis recorded in optimization_log [RESULT-#R5-3] (empty MTE2 stream between WaitFlag and SetFlag vs a multi-instruction ND2NZ `TileCopyTla` sequence from the previous slot). Next entry point: standalone minimal repro OUTSIDE this operator, then try the [#R3-2] "wait+set both inside the guard" form (consumer-side wait conditionalised too; token balance preserved).
- adopted: **no**
- rollback_reason: precision-gate hard failure (mandatory rollback rule). Restored from history_version/mha_impl_r3_perf11.py; md5 verified 3d56c5dd3d8a60d26eed7ade2e404a2b; ruff clean; **restore verified at IR level: 16/16 shapes' get_kernel_source() byte-identical to the #R5-1 adopted state**; mha.py == history_version/mha_impl_r3_perf10.py. Failed variants preserved: /tmp/opencode/r5_probes/mha_r53_both.py, mha_r53_konly_FAILED.py.
- next_hint: the redundancy #R5-3 targeted is REAL and is now the top bottleneck (C MTE2 = 50.0% of C wall; for case6, s_blocks=8 with contiguous task assignment means 7 of every 8 K loads and 7 of 8 V loads inside a batch are byte-identical). The idea should be retried only after the elision race is root-caused via a minimal repro. Alternative structural candidate still untried: #R5-2 (single-block O direct-write FIX(L0C)->Output, deleting V phase-2 + the WS3 SEM pair) - needs its own minimal probe for "L0C -> BSHD 4D runtime row slice" first. #R5-4 (NUM_STAGES 12/16) is a 1-line knob but has a wide blast radius (every shape's IR changes, GM ws footprint doubles) so it would need a full 20-case re-measure.

## Iteration r5-3 (closing) — 2026-09-21T14:35+08:00 (perf_session=5)
- abort: session closed on budget discipline (soft budget 3-4 iterations, 2 used: r5-1 ADOPTED, r5-2 falsified) after the user's directive was fully addressed and verified at mechanism level; NOT forced to exhaustion. Abort clause 2 (3 consecutive no-improvement) did NOT trigger - only 1 no-improvement.
- terminal state: **md5 3d56c5dd3d8a60d26eed7ade2e404a2b, 1086 lines** = session-4 terminal + #R5-1 only (= history_version/mha_impl_r3_perf10.py == r3_perf12.py)
- terminal-state fresh verification: `--level all` exit 0 "Test Passed!" (L0 10/10, L1 19/19, L2 12/12 correctly rejected, Boundary 6/6 with 0 BOUNDARY_WARN, sentinels 4/4 PERF_PASS, kernel_path 6.5s <= 120s gate) + anchor 10/10 BITEXACT vs Developer final + R11 stress inherited 4 configs nondet=0 + extended 7 configs on the changed path nondet=0 + coverage 19 PASS/0 MISS/0 N/A + ruff clean
- terminal-state msprof re-measure (msprof_r5_final): case1 11.07 (-15.04% vs 13.03 baseline) / case6 29.66 (-45.16% vs 54.08) / case19 22.98 (-35.92% vs 35.86) / case20 18.16 (-40.87% vs 30.71). NOTE: iter1 vs final on the identical kernel differ by up to -4.2% (case1 11.56 -> 11.07) i.e. cross-session msprof variance exceeds the within-session stdev; the adoption decision is therefore stated on the CONSERVATIVE (iter1) figures, and both are far beyond the 3% band vs baseline.
- guards: 7/7 within +/-3% (max case10 +2.12%) AND 12/12 guard shapes byte-identical generated source
- session-5 net: **1 adopted (#R5-1) / 1 rolled back (#R5-3, falsified with a 4-experiment evidence chain)**
- NOT executed (Orchestrator decision, flagged): cann-bench authoritative re-evaluation and the submission-copy sync (/mnt/workspace/cann/20260917_mha_V01/mha_submission/cann_bench/mha.py is still the session-4 terminal e80f7f65). Deliberately left stale rather than claim an unverified score.

## Iteration r6-1 — 2026-09-21T18:55+08:00 (perf_session=6)
- bottleneck_type: sync（四相位依赖链的第四相位 V-epi + w3 fp32 GM 往返；analysis 报告承接）
- optimization: #R6-1（=归档候选 #R5-2）单块 O 直写 FIX(L0C)→Output——Form A single_kv_block 批处理路径：C 侧 GEMM2 尾 `T.copy(l0c_o[side], Output[bz, s0:s0+valid_m, by, :])`（运行期行切片，fp32→dtype fixpipe RNE cast ≡ 被删 V 侧 vcast，#R3-3 已证）替代 w3 写；删 V phase-2 整段 + WS3 SEM 对（稳态 cross-core op 15→11）。硬前置门 = 算子外独立探针 8/8（整数精确 BITEXACT 含尾块 26/30/62 行、bf16、D=128、sentinel 无漏写）。
- baseline_time: case1 11.26us / case6 29.91us / case19 23.46us / case20 18.49us（同会话 msprof_r6_base）
- candidate_time: case1 10.06us / case6 24.51us / case19 21.53us / case20 14.32us（msprof_r6_iter1）
- improvement: case1 -10.66% / case6 -18.05% / case19 -8.23% / case20 -22.55%；守卫 7/7 ±1.6% 噪声带内（12/12 IR 逐字节未变，Form B 全域 = decode 硬约束 IR 级证明）
- precision: pass（探针 30/30 BITEXACT+golden / 锚 10/10 双新鲜复跑 / l0 10/10 / l1 19/19 / coverage 19P0M / all exit0 含哨兵 4/4 / 压力继承 4+扩展 7 配置 nondet=0）
- mechanism: camodel case1 —— V wall -13.6%、V FLOWCTRL -46.6%、V MTE2 -27.3%（w3 读消失）、V MTE3 -62.9%（写回移交 C）、V VECTOR -5.7%、VECTOR 段 10→8/微碎片 8→6；如实记录：C wall +3.4%（接管 Output FIX 直写，成为新 pacing 侧）
- adopted: yes（备份链 perf13 改前 → perf14 采纳态 bbfcbcaf，1113 行）
- rollback_reason: n/a
- next_hint: C 侧成为 pacing 侧且 C MTE2 3.96us(48.4% C wall) 为最大真实工作管 ⇒ Iter-2（批处理路径 q/k/v L1 双槽+批间预取）与 #R5-3 类字节削减的收益通道都已对准关键墙；Iter-2 硬门 = camodel MTE2 段结构变化

## Iteration r6-2 — 2026-09-21T20:15+08:00 (perf_session=6)
- bottleneck_type: transfer（假定 MTE2 装载暴露；实测证伪——真机 wall = GM/L2 总流量墙 + C 侧 FIXP 吞吐墙）
- optimization: #R6-2 Form A 单块批处理 q/k/v L1 双槽（3D [2,·,·] #R3-2 机制）+ 批头预载(kb==0)+批内预取(i+1<batch_tasks)+批间预取(kb+1<num_tb)，全守卫 wait+copy+set 同进同出、零 DMA 跳过、SIG 重排 K:0/1 V:2/3 Q:4/5 P:6、init/destroy 7 令牌
- baseline_time: case1 10.06 / case6 24.51 / case19 21.53 / case20 14.32 us（msprof_r6_iter1 = #R6-1 采纳态）
- candidate_time: case1 9.89 / case6 24.65 / case19 21.93 / case20 14.04 us（msprof_r6_iter2）
- improvement: -1.69% / +0.57% / +1.86% / -1.96% —— 全部噪声带内、方向不一致（IR 未变守卫同带幅漂移佐证）
- precision: pass（probe 30/30 BITEXACT+golden；IR 守卫 12/12 逐字节）
- mechanism: camodel case6 —— MTE2 段结构已变（22→20 段、冷启动合流 5136ns、气泡消除）= 机制激活；camodel C wall -19%/FIXP -24% 与真机 +0.57% **方向背离** ⇒ 判决性知识：camodel 争用模型不可用于收益定量（其 FIXP sum 29.8us > 真机全 wall），真机 wall = 字节总量墙
- adopted: no
- rollback_reason: 性能无变化（<3% 且方向不一致）；回滚至 perf14（bbfcbcaf），IR 级还原 16/16 逐字节核验
- next_hint: MTE2 预取/L1 多缓冲族在四路径形态全部实测收口（本条目为最后一块）；输入侧仅剩 #R5-3 冗余字节消除域（根因未定案）；Iter-3 = 算子外最小复现调查

## Iteration r6-3 — 2026-09-21T21:00+08:00 (perf_session=6)
- bottleneck_type: other（根因调查，非性能迭代；mha.py 零触碰）
- optimization: #R6-3 = #R5-3 根因算子外最小复现（授权 Iter-3）：三变体（A 对照 / B 归档失败形态 flag外+copy条件内 / C 同进同出安全替代）× 两级升级（v1 单核紧凑源 / v2 20 核 Fixed Core + 跨步 BSHD 型源命中 MOV_OUT_TO_L1_MULTI_ND2NZ）× 40 launch × 2 随机数据相位，槽内指令序逐字复刻 Form A 批处理 GEMM1
- baseline_time: n/a
- candidate_time: n/a
- improvement: n/a
- precision: n/a（未触碰 kernel；复现体 A/B/C 全 CLEAN 40/40，worst|d| 2.4e-6）
- adopted: n/a（调查归档）
- rollback_reason: n/a
- 结论: **B 形态两级最小结构均不复现非确定性损坏** ⇒ 归档首要假设（空 MTE2 流 SetFlag 即刻触发×ND2NZ 失序）在最小结构不成立；根因需 mha 完整上下文（跨核 SEM 节奏 / V 并发 / GEMM1+GEMM2 交织 / 真实批轮与寻址）；C 变体 CLEAN 不构成 kernel 级安全证明（结构对 B/C 无判别力，如实声明）。#R5-3 重试前置门升级 = 以 mha.py 为基座的消融法调查（独立任务）。按授权条款：根因未定案 → 不上 kernel，归档即止
- next_hint: 若立项消融调查，优先因子 = 跨核 SEM 悬停窗口（C 批在 WS1/WS2 wait 期间 DMA/flag 在途状态）——v1/v2 均无跨核交互而全 CLEAN

## Iteration r7-1 — 2026-09-22T23:50+08:00 (perf_session=7, rev4 轨)
- bottleneck_type: transfer（假设：w1 fp32 字节占 C FIX 1/3 + V-P1 读/cast 链）→ 实测证伪为 chain-latency
- optimization: #R7-1 w1 dtype 瘦身 fp32→dtype（scope=form_b_mma 专属；r16(qk) 执行体迁移至 fixpipe RNE，#R3-3 归档等价；emit_v 两分支 `if not w1_slim:` 包裹首回合 round-trip；io_buf dtype 联动；~40 行）
- baseline_time: case9 944.61us / case4 35.32us（msprof_r7_base，同会话）
- candidate_time: case9 947.50us / case4 35.73us（msprof_r7_iter1）
- improvement: case9 +0.31% / case4 +1.16% —— 噪声带内零响应（守卫漂移带 ±2.5% 佐证）
- precision: pass（探针 18/18 BITEXACT 含 bf16+大幅值域+zeros 对照；锚-A 9/9 硬守卫；锚-B 14/14；l0/l1/coverage/all 全过；压力继承 4+扩展 16 nondet=0；IR 13/13 守卫逐字节+3/3 目标变）
- mechanism: op A/B——AIV vec 47.2→44.4%、vector_stall -37%、AIC fixp -1.2pp、wait_id2 -26us（字节/指令确实削减）但 aic_time -2.6% 未传导至 wall；mte1_stall +33us 重分配
- adopted: no
- rollback_reason: 性能无变化；回滚至 perf2（ee5bf9c9），IR 16/16 还原核验
- next_hint: §10.3-① 升级证据性放弃（双 régime 零响应）；重试前置 = 某 bench case 真机 pacing 管为 fixp/FIX 字节

## Iteration r7-2 — 2026-09-23T01:30+08:00 (perf_session=7, rev4 轨)
- bottleneck_type: sync/chain-latency（判定完成：四相位每槽交接延迟链）
- optimization: #R7-2 Form B-mma k/v L1 双槽（3D [2,bN,D] ZN × 列段 transpose 新组合机制探针先行 BITEXACT；SIG 重排 K:0/1 V:2/3 P:4 Q:5；init 15/destroy 14；零条件 DMA；~31 行）
- baseline_time: case9 944.61us / case4 35.32us
- candidate_time: case9 947.96us / case4 36.23us
- improvement: case9 +0.35% / case4 +2.58% —— 噪声带内零响应（守卫带 -3.0~+2.5%）
- precision: pass（探针 18/18 BITEXACT；锚-A 9/9+10/10；锚-B 14/14；l0/l1/coverage/all 全过；压力 4+16 nondet=0；IR 13/13+3/3）
- mechanism: op A/B 判决性——**scalar_mte1_stall 412→188us(-54%) 靶心命中**、mte1 +5.7pp、V wait_id0 -41%；但 C wait_id2 324→545us(+68%) 1:1 吸收 ⇒ wall 零
- adopted: no
- rollback_reason: 性能无变化；回滚至 perf2（ee5bf9c9），IR 16/16 还原核验
- next_hint: 与 r7-1 合并 = 双侧 A/B 完备：case9 wall = 每槽交接延迟链（批深 4.5 摊薄不足），字节/速度类手段数学上无杠杆；唯一剩余规则内杠杆 = flat (task,kv) 批处理（需独立会话：emit_v 共享域改动爆炸半径 = 全 20 case）

## Iteration r7-3 — 2026-09-23T01:45+08:00 (perf_session=7, 评估归档，无 kernel 变更)
- bottleneck_type: n/a（候选评估）
- optimization: 剩余候选处置——①flat (task,kv) 批处理（Form B-mma）：EV -5~-15%（唯一有实质量级上界的规则内杠杆），但 emit_v 为三形态共享 ⇒ 爆炸半径全域、需完整门禁 ≈3h ⇒ **移交会话 8 首位候选**（蓝图 = #R4-3 机制 17/17 归档 + §S4-3-5 时序约束 + 本轮双侧 A/B pacing 判定）；②CROSS_INTERVAL=2：授权禁令类（wait 削减）+ 方向性收益不成立，不重提；③w3 colseg→plan-a：真机 fixp 非 pacing + 吸收律 ⇒ EV≈0 + 探针崩溃史，不启动
- adopted: n/a
- 中止条款: 连续 2 次无提升（未达 3）；第 3 候选全为禁提/EV≈0/需独立会话 ⇒ 按预算纪律收口

## Iteration r8-1 — 2026-09-29T20:40+08:00 (perf_session=8, rev4 轨, 联合专项 Stage 1)
- bottleneck_type: sync（四相位每槽交接延迟链；反模式对照：唯一残留结构命中 = "AIC/AIV 混合未开启 CV overlap" generic 多块域——**有意保留至 Stage 4 联合集成**，保留原因 = 吸收律下单侧/单项手段数学上无杠杆（r7-1/r7-2 双侧 A/B 1:1 实证），单点修复只会被对侧等待吸收）
- optimization: #R8-1 Stage-1 ② C 侧 L0B 4×16KB 槽旋转（form_b_mma；同取向 v4ksp 形态：pool [4,64,128] + GEMM1 K-split×4 链 + GEMM2 PV K-split×N-split（V 行列双切片直载）+ load-first 突发排序 + V 切片前置于 SEM 门控 P 段；id 矩阵 L0KC 0-3/L0AO 4,5/L0AS 6 = 7/8；init 14/destroy 13；~104 行净增。附.3 字面 [128,64] 双取向形态不可表达——T.view 损坏 + 取向冲突，探针实证归档）
- baseline_time: case9 944.74us / case4 35.57us（同会话新鲜基线 msprof_r8_base）
- candidate_time: case9 946.93us / case4 34.36us
- improvement: case9 **+0.23%（噪声带零响应）** / case4 -3.4%（唯一超带读数，单测未复核，回滚后不适用）；守卫 4 case ±1.1%（IR 13/13 逐字节未变）
- precision: **pass（门禁全过）**——机制探针 4 配置 BITEXACT；全 kernel 探针 20/20（14 目标 BITEXACT 含 case9 原形/multi-outer/effvar/bf16/尾块/zeros 对照 + 6 守卫）；锚-A 9/9 + 锚-B 14/14；压力 4+16 项 ×12 nondet=0；l0/l1/coverage/all 全过（哨兵 4/4）；预算实编读数 L0B 64.00KB 顶格单池 PASS；SEM census md5 与基线全等
- mechanism: **op A/B 判决性——判据 (a) FAIL**：mte1_stall 434→425us（-2.0%，非显著）且 stall(425)>busy(328) 签名存续；mte1_busy +2.1%/mte2 +3.4% = 事件成本真实；C wait_id2 -24.3% = 吸收律第三次复现
- adopted: **no**
- rollback_reason: 判据 (a) 机制激活失败 + 代价净值裁决为负（任务书规则）；回滚至 r4_perf4（ee5bf9c9），IR 16/16 还原核验（新鲜 dump vs r7_src_rollback2 provenance-verified 参照）
- next_hint: **新硬知识：mte1_stall = 四相位链伴生症状而非杠杆**（MTE2 侧 r7-2 -54% / L0B 侧 r8-1 -2%，wall 双双零响应；门控 = 跨迭代 MTE2 填充 + SEM 链）⇒ ② 族证据性关闭，激活前置 = flat ① 落地后流连续化（变体留存 /tmp/opencode/r8_probes/mha_r81_edit_v1.py 全门禁通过，零探针成本复活）；①+③ 二元联合评估完成（case7/8/3 主杠杆 ③ 不受影响；case9 B 层收敛降级为 "①+③ 实测裁决 + ② 条件复活"，626us 不再可承诺）→ 推进 Stage-2 ③

## Iteration r8-2 — 2026-09-29T22:05+08:00 (perf_session=8, rev4 轨, 联合专项 Stage 3 提前 = ORDER-PLAN 数据驱动修订)
- bottleneck_type: sync（四相位每槽交接延迟链——本项直接删除链结构：批轮次/排空点/任务边界-批界耦合）
- optimization: #R8-2 ① flat(task,kv) 连续令牌流 prefill 扩展（form_b_mma generic 域先行；顺序游走状态机映射运行期非均匀 causal eff_kv，C/V 双侧同式；任务切换 Q/L0AS 条件轮同进同出；V 状态机 §S4-3-5：phase2 kv0 复位/phase1 kv0 m_i 例外/phase2 kv-last 终归一+写回闭环；~390 行净增，emit_v 新增 elif flat_v 分支 + C generic 分支置换；Form A/decode/单块域语句零变化）
- baseline_time: case9 944.74us / case4 35.57us（msprof_r8_base 同会话）
- candidate_time: case9 **869.51us** / case4 **28.20us**
- improvement: case9 **-7.96%**（mission EV 带 -5~-15% 兑现）/ case4 **-20.72%**（批深空缺率收益模型）；守卫 case7 +0.25%/case10 +0.9%/case14 +0.3%/case16 +0.07% 全带内（IR 14/16 逐字节）
- precision: pass（探针 23/23 含 16/16 flat 域 BITEXACT 全拓扑；锚-A 9/9 + 锚-B 14/14；压力 4+22 项（含 flat 专项 f01-f06）×12 nondet=0；l0/l1/coverage/all 全过哨兵 4/4；红线四项复核 ✅）
- mechanism: op A/B 靶心全中——**C wait_id5（轮界）87.6→2.2us（-97.5%）**、C wait_id2 -31.9%、V wait_id4 -42.5%、V 核总时长 -9.7%；真实工作管全平（±0.5% 内）= 纯结构收益；mte1_stall +14.7% 为链收紧后的生产性 pacing 重分配（r8-1 判读一致性再证）
- adopted: **YES**（新基线 msprof_r8_iter2；采纳态 r4_perf6 md5 5d466b5ff3808bd8273eb85915ca4484，1839 行）
- rollback_reason: n/a
- next_hint: 吸收律边界补全入库（结构删除类不被吸收）；flat 收益模型 = f(批深空缺率, 边界密度) ⇒ form_a 多块域扩展（case7/8/3，mission 域序第二步）预期同 case9 量级；③ 宽通道降级为条件项（UB 实测 VW≤2/4 + r7-1 吸收先例）——form_a 扩展后视预算评估

## Iteration r8-3 — 2026-09-30T01:45+08:00 (perf_session=8, rev4 轨, 联合专项 Stage 3 第二步 = mission 域序 form_a 扩展；采纳判定于中断恢复会话落档)
- bottleneck_type: sync（四相位每槽交接延迟链——flat 结构删除类手段的 Form A 域推广；反模式对照状态继承 r8-1 条目：唯一残留结构命中"generic 多块域 CV overlap"为本项攻击对象，其余项不变）
- optimization: #R8-3 ① flat(task,kv) 连续令牌流 prefill 域扩展（form_a 多块 generic = case7/8/3/2/5；~205 行 C-scope 置换镜像 #R8-2 骨架 + flat_v 谓词扩展 (S>16)∧¬single（form 无关，V 侧零改动）+ 两项 Form A delta：l0a_s 重载条件 `i<2 or kv_g1<2`（P2 共池批头重载×任务始于批中部）、q_l1 归还移至任务末槽（设计期静态推演捕获的 P2 批头重载×跨批任务 MTE1/MTE2 竞态修复））
- baseline_time: case7 926.04us / case3 167.71 / case2 67.11 / case5 18.81（msprof_r8_iter3ref = perf7 参考态同会话实测；perf7 的 Form A 生成码 ≡ ee5bf9c9，r8-2 IR 守卫 14/16 在案）
- candidate_time: case7 **856.62us** / case3 **131.48** / case2 **44.91** / case5 **15.40**（msprof_r8_iter3）
- improvement: case7 **-7.50%** / case3 **-21.60%** / case2 **-33.08%** / case5 **-18.13%** —— 4/4 已测目标远超 3% 噪声带；收益幅度序与 flat 收益模型 f(批深空缺率, 边界密度) 定量吻合；守卫：case9 -0.14% / case4 +0.07%（fbm flat，IR 12/16 逐字节）/ case10 +0.86% / case14 -0.70% / case16 +0.05%（decode，IR 逐字节）/ case1 -0.8% / case6 -0.4% / case20 +0.3%（Form A 单块，r7_base 跨会话对照）——全域带内零回退；case8 移交 Step C 实测（mission 禁外推）
- precision: pass（探针 15/15 目标 BITEXACT + 守卫 7/7 + PART 0 修正谓词重算 PASS + valrange 既有伪影归档复核；锚-A 9/9 + 锚-B 14/14；压力 4+26 项（含 flat-A 专项 f07-f10）×12 nondet=0；l0 10/10 + l1 19/19 + coverage 19P/0M + --level all exit 0 哨兵 4/4；IR 三方 12/16+4/4 与 vs 基线 10/16 decode×5 IR 级成立；**恢复会话 provenance 核验：当前源新鲜 dump 16/16 逐字节 = r8_src_iter3 ⇒ 门禁日志对当前文件态有效**）
- mechanism: op A/B（case7，block0）靶心复现——**C wait_id5 轮界 101.27→4.83us（-95.23%）**（case9 fbm 域 -97.5% 同款）、C wait_id2 -10.39%、V wait_id4 -50.32%；真实工作管全平（cube +0.00% / V.vec -0.58% / mte2 -1.93% / fixp +0.45%）= 纯结构收益；mte1_stall +44.89% = 生产性 pacing 重分配（伴生症状判读三度确认）
- adopted: **YES**（新基线 msprof_r8_iter3 + r8_op_case7_iter3；采纳态 r4_perf8 md5 2728c96eb2fabbfff279918c61c6c3e0，1900 行）
- rollback_reason: n/a
- next_hint: ① 全域落地（fbm generic + form_a generic；单块/decode 域分派谓词收窄零触碰）；Step C = 联合态全 20 case wall 裁决 + decode 不回退判定（case8/11/12/13/15/17/18/19 补测，bench 口径换算留 orchestrator）；Step D 残余杠杆条件评估（② 复活 vs ③ 缩减形态，数据驱动择一）；flat 收益模型跨 form 验证入库、吸收律边界升级为跨域定律

## Iteration r8-4 — 2026-09-30T02:10+08:00 (perf_session=8 恢复会话, Stage-4 联合裁决 + Step D 条件评估归档 + 收口；无 kernel 结构变更——仅 codegen-neutral docstring 注记)
- bottleneck_type: sync（联合态残余 = 每槽四相位交接不可删段 + MTE2 填充节拍（case9）/ V 侧碎片事件税（case7，V.vec 77.5% 最大真实工作管 + C.wait_id2 67%））
- optimization: 无新优化实施。Step C = 联合态全 20 case wall 裁决（① 全域：fbm r8-2 + form_a r8-3）；Step D = 残余杠杆条件评估归档——数据驱动选择 (b) ③ VW=2 宽通道为评估对象（case7 V-paced 结构依据 + UB 实编读数 156.75K ✓ 在案）、(a) ② 复活被联合态新数据否决（MTE2 门控未变 + 吸收律 + 双侧 A/B 界 ⇒ 预测 wall≈0）；③ EV = 真开放问题（吸收律先例的因果结构 pre-flat 特有，post-flat 已不成立）但为"精度门最重"项 ⇒ 按 mission 授权条款移交会话 9 首位候选（方案要素齐备落档 optimization_log [STEP-D-EVAL]-5）
- baseline_time: 会话起点 ee5bf9c9（r8_base 本体 6 case + perf7 参考态 14 case——perf7 在被测各域生成码与 ee5bf9c9 逐字节等价，IR 守卫证明在案）
- candidate_time: 联合态全 20 case（msprof_r8_iter3 + Step C 补测 8 case）
- improvement: **① 目标域 7/7 实测收益：case2 -33.08% / case3 -21.60% / case4 -20.66% / case5 -18.13% / case9 -8.09% / case7 -7.36% / case8 -6.26%（实测非外推）；守卫 13/13 带内零回退（最差 case20 +1.75% = IR 逐字节域 +0.25us 会话方差）**；收益全序与 flat 收益模型 f(批深空缺率, 边界密度) 吻合；decode 硬约束本地口径无回退信号（case10 +0.86%/13 +0.29%/14 -0.70%/16 +0.05%/20 +1.75%，bench 口径权威判定留 orchestrator）
- precision: pass（终态 dda3ad08 = r8-3 采纳态 + codegen-neutral docstring 注记：16/16 IR 逐字节复核 = r8_src_iter3；终态新鲜门禁子集 = 锚-A 9/9 + 锚-B 14/14 + 继承压力 4 配置 ×12 nondet=0 + l0 10/10 + --level all exit 0 哨兵 4/4，r8_cache_finalg 隔离缓存）
- mechanism: 三段对比（base→iter2→iter3，block0）：case9 wait_id5 87.57→2.16→承继 / case7 101.27→承继→4.83（-95.2%）；wait_id2 case9 -31.9% / case7 -10.4%；V.wait_id4 case9 -42.6% / case7 -50.3%；两域真实工作管全平 = 纯结构收益；mte1_stall 两域上升 = 伴生症状（生产性 pacing 重分配）；批轮次 ~103→~58/核、批深 4.5→8 两域同式
- adopted: n/a（评估归档 + 收口，无候选实施；会话 8 净产出 = ① 全域落地，采纳态链 r8-2/r8-3 已各自落档）
- rollback_reason: n/a
- next_hint: 会话 9 首位候选 = ③ VW=2 宽通道（case7/8 域，判据 wall 响应主判 + 预测签名 V.vec↓∧wait_id2↓∧wall↓；若零响应 = 吸收律第四例、③ 族关闭）；次位 = case9 域在 ③ 落地后的 D=256 VW=2/4 顺带评估；② 复活前置更新 = 需 MTE2 管收缩的新结构证据；B 层 626us（0.54x）承诺维持降级——剩余差距 = MTE2 节拍 + 四相位不可删段 + V 事件税，需 ③ 或设计级手段

## Iteration r9-1 — 2026-09-30T15:40+08:00 (perf_session=9, Stage-1 ③ VW=2 宽通道 form_a D=128 域)
- bottleneck_type: compute（V.vec 碎片事件税假设）→ **实测修正为 transfer/compute 复合：事件税在 scalar/wait 侧（可削），vec 侧主体 = 数据体量吞吐（宽化反增）**
- optimization: ③ V 宽通道 VW=2（flat phase1 pair 合并宽 [2hM,bN] op 序列 + per-half mask + 单宽 select + SIG_S_HALF 1/pair + 奇尾槽方案 K 宽形状退化；方案 A'' 统一缓冲名 + 方案 R 微缓冲 V-scope 内 alloc；分派谓词 flat∧form_a∧D128）
- baseline_time: case7 858.26us / case8 855.78 / case3 131.35（msprof_r9_base 同会话新鲜基线，dda3ad08 本体）
- candidate_time: case7 **964.77us** / case8 **993.06** / case3 **150.20**（msprof_r9_iter1）
- improvement: case7 **+12.41%** / case8 **+16.04%** / case3 **+14.35%** —— 3/3 目标域显著性能下降（超 3% 噪声带）；守卫 9/9 带内（case9 -0.03%/case4 +1.18%/case2 +0.18%/case5 +0.66%/case10 +3.04% 边缘（IR 逐字节域方差归因，回滚态复测澄清）/case13 -1.04%/case14 +0.40%/case16 +0.43%/case20 +1.21%）
- precision: pass（探针 16/16 目标对改前 BITEXACT + valrange 既有伪影复核 + 守卫 7/7 BITEXACT；锚-A 9/9（**#9 宽域仍 BITEXACT = Layer 1 锚方案兑现无降级**）+ 锚-B 14/14；压力 33 项（含 wide 专项 w01-w07）×12 nondet=0；l0 10/10 + l1 19/19 + coverage + all EXIT=0；IR 14/16 守卫逐字节 + 2/2 目标预期变；UB wide 域实编 172.62K ≤ 191.75K；红线四项逐字保持）
- mechanism: op A/B（case7 block0/均值双口径）**三签名全 FAIL 反向**——V.vec 634.39→772.73（**+21.8%**，签名 1 预测"显著降"证伪）∧ C.wait_id2 547.70→657.83（+20.1%）∧ wall +12.4%；事件税侧签名全兑现（v_scalar -11.7%/v_stall -35.6%/v_mte3 -27.9%/wait_id0 -44.2%/wait_id4 -51.0%/wait_id5 -100%）但被 vec 主管道增量吞没；**V-paced 减速侧对照实验 = pacing 管因果系数 ≈1 双向确证（新硬知识）**
- adopted: **NO — 回滚**（rollback_reason: 性能下降规则；已回滚 history_version/mha_impl_r4_perf11.py，md5 dda3ad08 核验 + **IR 16/16 逐字节还原验证**（r9_src_rollback == r8_src_iter3）；变体留存 /tmp/opencode/r9_probes/mha_r91_edit_v1.py = 精度门禁全过态）
- next_hint: **③ 族证据性关闭（Stage-2 fbm 扩展不实施，[STAGE2-RULING] 数据驱动裁决：机制证伪为硬件级宽化增量跨 form 成立 + case9 域 EV≤0 预判）**；V 侧宽通道重试前置 = bN≥256 行体量翻倍摊薄 repeat 税的新形态证据（与路径 2 蓝图组合可能，scratch 复测零生产风险）；consecutive_no_improvement = 1

## Iteration r9-2 — 2026-09-30T15:45+08:00 (perf_session=9, Stage-3 路径 2 fbm bN=256 scoping 归档；无 kernel 变更)
- bottleneck_type: sync/structure（case9 占空受限 58%：每槽不可删段 ×442 槽；路径 2 机制 = 槽数减半的结构删除类，吸收律不适用域）
- optimization: 无实施（scoping only，任务书授权形态）。产出 = 可行性矩阵实编读数（GATE1-5 全 PASS：L0C 三形态 used 64.00K 顶格（判据修正：现状生产池值 128K 对照）/L0B 64.00K 顶格/L0A 48K/L1 288K/UB 170.75K）+ GATE6（ws 56→80KB/槽、总足迹 +42% vs 总流量 -29% 两反向因素 ⇒ L2 命中净效应解析不可判，**实测未做如实声明**：需 rev4-S1 量级完整变体，实测方案 = scratch msprof L2Cache.csv 入库）+ GATE7（FIX 事件账 3→5/iter × 槽 ÷2 = 总事件 -17% 模型正、mma 链重组工程量 rev4-S1 量级）+ EV 修正（**-12~-19% 条件化：L2 命中 ≥90% 维持；命中塌缩情形 0~+10% 反转预警**）+ 实施蓝图 5 要点（F5 首选/v4ksp L0B 复用/flat 骨架不变/位级预期（r4-2a k-split 先例）/GATE-6 立项前置硬门）——[SCOPING-PATH2-R9] 落档 optimization_log，供会话 10 立项决策
- baseline_time: n/a（scoping，无 wall 测量对象）
- candidate_time: n/a
- improvement: n/a（EV 修正带归档）
- precision: n/a（零生产 kernel 变更；探针 scratch 独立文件）
- adopted: n/a（scoping 归档 = 合格产出形态，任务书 Stage 3 条款 6）
- rollback_reason: n/a
- next_hint: 会话 10 立项决策输入 = GATE-6 L2 命中实测（前置硬门）+ F5 形态蓝图 + ③×bN256 组合复测机会（V 侧宽化在 bN=256 行体量下的 EV 重估，scratch 级）

## Iteration r9-3 — 2026-09-30T16:05+08:00 (perf_session=9, Stage-4 联合收口；无 kernel 变更——回滚验证 + 落档)
- bottleneck_type: n/a（收口）
- optimization: 无新优化。收口项 = ①r9-1 回滚执行与三重还原验证（md5 dda3ad08 一致 + **IR 16/16 逐字节还原**（r9_src_rollback == r8_src_iter3）+ 回滚态 l0 10/10 新鲜复验）；②回滚态抽查复测 4 case（msprof_r9_rb）：case7 861.10（vs r9_base 858.26 = +0.33% 还原带内）/ case9 869.28（+0.03%）/ case10 35.43（**+4.6% = 环境漂移**：同一字节级文件三段测量 33.87→34.90→35.43 单调漂移，IR 逐字节 ⇒ 非内核因果；大 kernel case7/9 无漂移 ⇒ 漂移为 decode 小 kernel 敏感项，候选态 case10 +3.04% 边缘读数由此澄清）/ case20 14.50（+3.3%，绝对量 +0.47us，会话 8 case20 +0.25us 方差先例同款）；③落档全套（[PERF-#R9-1]/[STAGE2-RULING]/[SCOPING-PATH2-R9]/§S9/debug_log 会话 9 节/baseline_r9_iter1.json）；④全 20 case 复测条款处置：候选态上已实测 12 case（msprof_r9_iter1）+ 回滚态 IR 16/16 逐字节还原 ⇒ 全量复测保证由 IR 还原 + 会话 8 终态在案读数 + 本会话 r9_base 12 case（同文件态）承继满足（会话 7/8 回滚态先例同款口径），抽查 4 case 复核无异常；⑤docstring：回滚态零变更 → 无注记工作（条款按"若有变更"条件性适用，如实记录）
- baseline_time: r9_base（dda3ad08 本体，12 case）
- candidate_time: n/a（终态 = 基线本体）
- improvement: **会话 9 净变更 = 0**（1 回滚 + 1 scoping 归档；consecutive_no_improvement = 1 < 3，中止条款②未触发）
- precision: pass（终态 = 会话 8 收口门禁在案态 + 本会话回滚态新鲜 l0 10/10 复验 + IR 16/16 还原证明）
- adopted: n/a（收口）
- rollback_reason: n/a
- next_hint: 会话 10 首位候选 = **路径 2 fbm bN=256 立项决策**（[SCOPING-PATH2-R9] 输入齐备：GATE1-5 实编 PASS/双顶格警示/EV -12~-19% 条件化/F5 蓝图；**前置硬门 = GATE-6 L2 命中实测**（scratch 变体 msprof L2Cache.csv，命中 <90% ⇒ 否决或降级）；顺带机会 = ③×bN256 组合宽化 EV 重估（scratch 级零生产风险，S9-3-3 重试前置）；工程量预警 = rev4-S1 量级需独立会话全额预算）；case9 0.39x vs M1 线 0.4 差 0.01——路径 2 兑现 -12% 级即 0.44+ 越线

## Iteration r10-1 — 2026-10-01T01:15+08:00 (perf_session=10, #R10-1 路径 2 fbm prefill bN=256 结构加宽；判决落档 = 中断恢复会话补档，实测全程 2026-09-30 16:51-23:21)
- bottleneck_type: transfer/structure（case9 占空受限：C.MTE2 顶格 57.8% + wait_id2 交接暴露 25% + 爬坡 15%——路径 2 = 槽数 442→221 减半的结构删除类，吸收律不适用域）
- optimization: T2S 同取向 L0B [2,128,128] 共池 + F5 L0C n-chunk + K/V L1 半区拆分 + 尾块钳位 + 谓词迁移 `S>16∧D==256∧S_kv>128`（[IMPL-#R10-1] 12 条约束）；HANG 后判决链 = bisect 族 3 + E 系 9 + E21 边际决胜 + T1 v4ksp 全架构推广（2382 行）+ E22 全排空补强 = **15 实测变体毒化矩阵**（详见 [RESULT-#R10-1] + debug_log D10-1..4）
- baseline_time: msprof_r10_base（dda3ad08，median n=25）：case9 869.87 / case4 28.21 / case7 855.54 / case10 34.61 / case14 378.26 / case16 562.01 / case20 13.98 us（vs r9_base 全带内）
- candidate_time: **n/a — HANG 先于性能门**（bN=256 全部形态 aicore timeout 507014 挂死；数值/精度门未达；GATE-6 L2 命中/GATE-7 FIX 税净值未执行，[SCOPING-PATH2-R9] 条件化 EV **作废归档**）
- improvement: **会话 10 净变更 = 0**（1 结构性不可行回滚；consecutive_no_improvement = 2 < 3，中止条款②未触发）
- precision: fail（HANG 缺陷 = 非数值性失败；回滚态 dda3ad08 精度 = 会话 8/9 在案全量门禁通过态 + 本会话轻量复验全绿：l0 10/10 + 锚-A 9/9 + 锚-B 14/14 BITEXACT + 继承压力 4×12 nondet=0）
- adopted: **NO — 结构性不可行**（非性能规则回滚；判据 = debug_log D10-3 **预登记穷尽条款**："T1 HANG ⇒ 全部可表达形态穷尽 ⇒ 路径 2 [DESIGN_ERROR]"——T1 实测 3/3 HANG（run_T1_* EXIT=1）字面命中；≥3 方案实测达成（15 变体））
- rollback_reason: 路径 2（fbm prefill bN=256）结构性不可行——根因硬件硬限：**B 工作集 K^T/V 各 128KB > L0B 64KB ⇒ 相位内 L0B 重用数学上不可避免 ⇒ 双重毒化无安全编排**（(a) L0B 相位内覆写 ~50ns 边际挂死（E16/E13）；(b) K128-mma 相位深度 ≥3 或 B 二次读取挂死（E17/E21/E15/PROD）；E21 边际理论证伪 + E22 gemm_v0 级全排空亦挂 + T1 分立槽推广 3/3 挂 = 编排空间封死）。**bN 上界定理入库：手动 mma 流水的 bN 上界 = B 操作数工作集 ≤ L0B 64KB（含 K-split 链全部在途 chunk）**。回滚三重验证：md5 dda3ad08 一字不差/1926 行/ast+ruff clean + IR 16/16 逐字节还原（r10_src_rollback vs r8_src_iter3，disable_cache+r10_cache_rb 新鲜 dump，manifest 逐字节一致）+ 轻量复验全绿；工具教训⑤（k_l1_h1 别名漂移）零残留关闭
- next_hint: 大 prefill 规则内杠杆版图收束 = **①已兑现、②证据性关闭、③规则内永久关闭（重试前置 bN≥256 规则内不可达）、④激活前置链条断裂、路径 2 结构性不可行** ⇒ case9 M1 线 0.4（差 0.01）与大 prefill ≥1.0x（B 层上限 0.54-0.79x）均规则内不可达，**仅剩架构级方案 C**（框架虚拟通道 L0C→UB/UB→L1 + 单域 flash 重构，需框架级投资决策，建议以 case9 为样板先做可行性验证）；[DESIGN_ERROR]（路径 2）上报 Orchestrator 裁决（预判 = 候选证据性关闭而非设计回退）；禁止任何 bN=256 重试（穷尽条款）

## Iteration r10-2 — 2026-10-01T01:15+08:00 (perf_session=10, Stage-4 收口；中断恢复会话本体；无 kernel 变更——判决宣告 + 回滚执行 + 三重验证 + 落档补全)
- bottleneck_type: n/a（收口）
- optimization: 无新优化。收口项 = ①Step A 现场核验（生产 c748cfb2/2259 缺陷态 + perf12/13 = dda3ad08 + mha_T1.py 04a41a97/2382 + run_T1_* 三日志 EXIT=1 完整性 + debug_log D10-3 预登记判据在案——与 Orchestrator 事实基座逐项一致；**发现 E22 未落档偏差项**（T1 判决后 22:52 构建、23:02-23:21 运行 3/3 HANG）→ 补档定性 = 判决后补强证据，穷尽条款终止）；②Step B 预登记判据宣告（穷尽条款生效 + bN=256 重试禁止 + ③ 关闭升级规则内永久）；③Step C 回滚执行与三重验证（md5/行数/ast/ruff + IR 16/16 逐字节 + 参照链 provenance 复核 r8_src_iter3==r9_src_rollback diff exit 0）；④工具教训⑤收尾（k_l1_h1/v_l1_h1 生产态源码+IR 双 grep 零残留 ⇒ 漂移仅存 r10-1 变体族，随回滚消解，关闭）；⑤Step D 回滚态轻量复验（会话 9 r9-3 同款口径：l0 10/10 + 锚-A 9/9 + 锚-B 14/14 BITEXACT + 继承压力 4 配置 ×12 nondet=0，r10_cache_rbv 隔离缓存）；⑥Step E 落档补全（[RESULT-#R10-1] 全证据链 + 本 perf_log r10-1/r10-2 + debug_log D10-4 + final_report §S10）；⑦如实声明项：r10-1 PROD 缺陷态源文件无独立留存副本（实现内容由 E 系派生全文件 + [IMPL] 规范 + hang_*.cc 实质保留）
- baseline_time: n/a（终态 = 基线本体 dda3ad08；msprof 不涉及——本收口无性能测量）
- candidate_time: n/a
- improvement: 会话 10 净变更 = 0（r10-1 结构性不可行回滚 + r10-2 收口）
- precision: pass（终态 = dda3ad08 = 会话 8/9 在案态 + 回滚三重验证 + 轻量复验全绿；T1/E22 数值正确性未判 = HANG 先于数值门，如实声明）
- adopted: n/a（收口）
- rollback_reason: n/a
- next_hint: [DESIGN_ERROR]（路径 2）+ [PERF_DONE] 收口报告已提交，全局流转（是否触发 Stage-1 设计回退）由 Orchestrator 裁决；cann-bench 复评 = Orchestrator 轮末统一执行（终态与复评基线同体，预期分数不变）；遗留工件族（mha_T1.py/E 系 15 变体/毒化矩阵日志/生成码 dump）= 知识资产留存 /tmp/opencode/r10_probes/ 防未来重复探索

## Iteration 24 — 2026-10-02T01:20+08:00 (mode=perf_measure，**GATE 0 预筛门 · 纯测量轮 · 零 kernel 变更 · 零采纳代码**)

- bottleneck_type: **sync（C↔V 四相位交接往返延迟链）** — 由本轮直接消融判定，非推断。**修正 r10-1 记录的 "transfer/structure（case9 占空受限：C.MTE2 顶格 57.8% + wait_id2 25% + 爬坡 15%）"**：三分量中 MTE2 57.8% 数值成立（本轮 ratio 列均值 56.07%，per-block 带 53.03~59.34）**但已被消融证伪为 pacing 管**；wait_id2 25% 仅在 max-block 口径成立（均值口径 21.01%）；**爬坡 15% 被直接测量推翻（实测 ≤0）**。case9 pacing = 交接延迟链（两侧同时大互等：C 22.1% / **V 64.2%**；最大不可削减真实工作管 = AIV VEC 468.58 µs，消融下 **Δ=0.0000 逐位相同**）；case7 pacing = **AIV VEC 648.99 µs = 77.52%**（block0 绝对值 634.387268 µs **逐位复现历史 634.39**），C 侧 `wait_id2` 63.48% = 巨大余量。**两 case pacing 身份不同，禁止混用**（C.MTE2 ratio 相近 56.07% vs 54.28%，但 case9 是延迟链受限、case7 是 V 吞吐受限，V.vec 与 C.MTE2 差距 5.9% vs 23pp）。
- optimization: **无（本轮为预筛门测量轮，不产出采纳代码）**。执行内容 = ①生产态新鲜基线复采与复现验证；②Q1 C scalar 成分拆分（发现 msprof `scalar_single/dual_time` 与 stall 桶为**互斥可直接测量**的计数器，加和闭合 100.7%/108.0%）；③Q2 case9+case7 全 pipe busy% + 全 wait_idN 双口径（20 block 均值 + argmax wall 关键 block）复采；④**Q3 方法 A 消融上界探针 ×2 域**（fbm `mha_qablate.py` md5 07a0b1dd **2 行 delta** L1514/L1559；Form A `mha_qablate_fa.py` md5 d0233d6f **3 行 delta** L1070/L1084/L1090；均改条件为 `fb*NUM_STAGES+i==0` ⇒ Q prologue 由 103 次/核降为 1 次/核，**token 收支与生产态同构（1 dangling）⇒ 无 HANG**）；⑤Q3 方法 B 真机尝试 **双双不可达**（`TimelineDetail` 采集失败 `simulator/` 空；`--instr-profiling=on` 原始 slice 采集成功但 msprof CLI 无指令级导出表，需 MindStudio）⇒ **如实放弃 per-event 识别，改用计数器差分法直接测量替代**；⑥模型 `ramp` 项逐字复算与拆解（`tB_cores` 自校验 625.9/797.4 vs 记录 626.1/797.2）；⑦skill 反模式与最佳实践对照。**camodel 本轮完全未运行**（纪律 #1）。
- baseline_time: **case9 869.547 µs / case7 856.600 µs**（plain `msprof --task-time=on` → `op_summary` Task Duration，warmup 5 / iters 20 / n=25 取 median，3 次 interleaved 重复均值；`parse_prof.py` 仓内已标定口径）。**复现验证：case9 vs r10_base 869.87 = −0.04%；case7 vs 855.54 = +0.12% ⇒ 均在噪声内**。prod 侧 median 重复间极差 **case9 0.78 µs（0.09%）/ case7 0.23 µs（0.03%）**；paired-diff SEM ≈1.2 µs（≈0.14%）⇒ **本轮分辨力比历史 ±3~4% run-to-run 带紧一个数量级**。
- candidate_time: **n/a（无候选实现）**。探针 wall（仅用于判决，禁止采纳/入 baseline/提交）：case9 fbm 探针 872.050 µs（+0.288%）；case7 Form A 探针 864.063 µs（+0.871%）。
- improvement: **0（测量轮，无采纳）**。`consecutive_no_improvement` **维持 2**（继承 r10-1/r10-2），本轮**既不递增也不清零**（无候选 ⇒ 无"无提升"事件）；中止判定归 Orchestrator。
- precision: **pass（继承 + 本轮新鲜 l0 复验）** — 生产 `mha.py` **逐字节未变**（md5 `dda3ad0877ba4df74bbd559f6912eeed` / 1926 行，**轮次开始 + 两次探针构建后 + 轮次结束四点复核全部一致**，探针生成器内置 md5/行数/diff 行数三重断言；轮末追加 `ast.parse` OK + `ruff check` All checks passed），故 session-8/9/10 在案全量门禁（锚-A 9/9 BITEXACT + 锚-B 14/14 + 压力 4×12 nondet=0 + `--level all` exit 0 PRECISION_PASS=29/FAIL=0）继续有效；**本轮另新鲜复跑 `test_mha.py --level l0` = 10/10 PRECISION_PASS，exit 0，"Test Passed!"**（隔离缓存 `cache_prod`；max_abs 0.000e+00 ~ 9.766e-04 全在 gate 内），`--level all`/锚/压力未重跑（byte-identical + l0 全绿 ⇒ 无漂移信号）。**两探针输出数值故意错误并已实测确认**（fbm rel_max **8.447e-01** / max_abs 8.447e-01 / out_sum −14403.6973 vs ref_sum −14595.2021；Form A rel_max **5.900e-01** / out_sum −7524.0005 vs ref_sum −7240.4751）⇒ rel_max 59~84% 证明**消融机制确实激活，排除假阴性**；探针明确排除在任何精度判定之外。
- adopted: **NO — 测量轮，无可采纳对象**。GATE 0 裁决 = **`[GATE0_VERDICT] FAIL`**。
- rollback_reason: n/a（生产态全程零变更，无需回滚）。
- **GATE 裁决**：
  - **GATE 2（`q_l1` flat 域双缓冲攻击 ramp）= ❌ NO-GO**。依据五条：①**EV 上界实测 [−1.48%, +0.33%]**，点估计 [−0.87%, −0.29%]——消融探针测的是"完全消除"极限，**方法学上严格强于任何加速类干预**，故该带即上界；比 3% 采纳门低 **≥9×**，比模型宣称的 15% 低 **≥45×**。②**模型自身算术显示可攻击切片仅 6.74 µs = 0.78% wall**（`ramp` 134.44 µs 中 Q GM→L1 切片占 5.0%，`2·c_pipe` 通用填充切片占 **85.2% = 114.54 µs**，与 `q_l1` 缓冲深度**无关**）⇒ **测量前就已低于门槛，且方案攻击面与目标项不匹配**；任务书"EV 上界完全取决于 134 µs"属**过度归因**。③吸收律分类 = **单侧速度类**（零个交接被删除；字节数不变；且附带**事件类负项** = 新增 `SIG_Q_L1+1` token 与双槽 side 索引），而 case9 pacing = 交接延迟链、case7 pacing = AIV VEC，**均非 C 侧 Q 装载**。④**吸收律第 5/6 例，且首次以"完全消除"上界形式定量**（前四例 r7-1/r7-2/r8-1/r9-1 为"实施后 wall ±0.3%"，证据强度更低）：case9 移除 C 真实工作 MTE2 −32.40 µs / MTE1 −7.99 µs / mte1_stall −45.70 µs → C 跨核等待 **+58.75 µs**（id2 +54.39 = +30.85% / id5 +4.36 = +127.0%）；case7 移除 −34.44 / −13.60 / −60.22 µs → **+70.94 µs**（id2 +57.68 / id5 +13.26 = +219.6%）；**V 侧真实工作管逐位零变化**（case9 `aiv_vec_time` **Δ=0.0000**、V 侧 MTE2/MTE3 指令数与 GM↔UB 字节数 Δ=0；case7 Δ=+0.0065 µs）⇒ 吸收律三项签名全部满足。⑤**case7 为统计显著劣化**（paired n=3，t=6.201 df=2，**p=0.025**，95%CI [+0.267%, +1.476%]）；case9 不显著（t=2.048，p=0.177，95%CI [−0.317%, +0.893%]）——**NO-GO 不依赖显著性，依赖上界 ≤+0.33% ≪ 3%**。
  - **GATE 1（`unit_flag=0b11` 铺垫）= ❌ NO-GO（就其"铺垫"定位）**；作为独立候选 = **未实测**（纪律 #4，**不作预测性否决**）。依据：①其存在理由 = 为 GATE 2 铺垫，**GATE 2 已实测 NO-GO ⇒ 铺垫对象消失**；②机制归类（`docs/api_docs/T.copy.md:102/159` + `examples/deepseek_v4/lightning_indexer.py:666-667`）= 用硬件 mma→fixpipe 流水替代软件 `M↔FIX` flag 轮，属 **C 侧单侧事件删除类**，而本轮两度实测证明 C 侧单侧削减被 1:1+ 吸收（case7 显著劣化）；③若要立项必须**改判攻击面**（转向 V 侧或交接往返链本身），或先补一次 **C 侧 M↔FIX 事件删除类独立消融上界探针**（本轮方法学可直接复用，成本 ≈25 分钟）。
  - **吸收律三步法**：Step1 pacing 管认定（case9 = C↔V 四相位交接往返延迟链，消融证伪 AIC MTE2 为 pacing 管；case7 = AIV VEC，消融直接确认）；Step2 方案1 分类 = **单侧速度类 + 事件类负项**（非字节类/非结构删除类）；Step3 **减速侧对照不需要** —— 完全消除探针严格强于减速侧对照（上界已封死），且吸收因果已由「`wait_id2` 增量与 C 侧削减量同向等量+ ∧ V 侧真实工作管逐位零变化 ∧ wall 零响应/劣化」三项**直接测量**证实，对照实验无法抬高上界。
- **Q1 判决（C scalar 余量）= 充足，GATE 2 可新增标量游走**：真实发射 = `scalar_single_time + scalar_dual_time` = **case9 76.55 µs（20.25+56.29）= 9.12% of wall / 9.35% of scalar busy**；**case7 62.94 µs（11.95+50.99）= 7.53% / 8.23%**。scalar busy 的 **≈91% 是 stall 归属**（case9 741.92 µs：mte1_stall 559.04 + wait_id2 176.32 + id1 6.14 + id5 3.43 + wait_ib 2.06 + cube_stall 0.33，桶加和 747.31 = 残差 100.7%；case7 701.60 µs：wait_id2 530.89 + mte1_stall 209.30 + id1 10.14 + id5 6.04 + wait_ib 1.60，桶加和 = 残差 108.0% ⇒ **桶间轻微重叠，已如实标注**）。**该拆分为直接测量而非反推**（single/dual 与 stall 为 msprof 分离计数器且加和闭合）。**修正 `final_report.md:680`「C scalar 95.5%」语义**：该值 = `aic_scalar_ratio`（busy 比，含 stall），本轮实测 case9 均值 92.48% / max-block **96.33%**（历史 95.5% 落在 per-block 带内 ⇒ 数值可复现，语义需更正为"≈91% 是 stall"）。**Q1 曾是 GATE 2 生死门，判定为「不构成阻塞」——GATE 2 的否决理由全部来自 Q3 的 EV 上界，与标量余量无关。**
- **Q3 方法 B 如实处置**：真机 per-event MTE2 时间轴**不可达**（`TimelineDetail` 采集失败 + `--core-id` 仅 simulation products 有效；`instr-profiling` 有原始 slice 无 CLI 导出表）⇒ **未用模型值/历史值冒充测量值**；改用**计数器差分法**直接测得方法 B 所求的「总时长」：Q 装载事件数 **101.4/核**（`aic_mte2_instructions` Δ）、字节 **1622.4 KB/核 = 101.4 × 16 KB 精确**（`GM_to_L1_datas` Δ）、**MTE2 busy 总时长 32.40 µs/核 = 3.86% of AIC wall**、单次 **0.3195 µs / 16 KB**；case7 = 34.44 µs（4.12%）+ l0a_s MTE1 13.60 µs。**非重叠暴露部分 ≤0（方法 A）⇒ 已重叠效率 ≈100%**。MTE1 侧 −405.6 指令（case9，= −3.98/任务 × 102，即每次 `q_l1→l0a_s` 展开为 ~2 条 MTE1）/ −1016.6（case7，含 `i<2 or kv_g1<2` 的批头重载）。
- **附加发现（超出任务书，均为直接测量）**：①**Q 装载是 AIC L2 缺失主源** —— 消融使 `aic_r0_read_cache_miss_allocate` **−34.76%（case9）/ −34.55%（case7）**、`aic_read_hit_rate` **92.01→94.65%（+2.64pp）/ 92.63→94.96%（+2.34pp）**，**wall 仍不动 ⇒ case9/case7 非带宽/DRAM/L2 受限**（与模型自算一致：Q 流量 2048×16KB = 33.55 MB 仅占 case9 实现态 GM 流量 2332 MB 的 **1.44%**）。②`aic_cube_time` 消融下 **零变化**（case9 −0.02 µs / case7 −0.02 µs）⇒ 消融未扰动 CUBE 管，A/B 干净。③探针侧标量发射**实际下降**（case9 `scalar_dual` −2.24 µs；case7 `scalar_single` −3.43 µs）⇒ **case7 显著劣化的"新增标量比较"解释被实测否证**；劣化机理**未定案**（候选：Q 的 MTE2 流原为隐式节拍间隔器，移除后 C 到达信号量的突发更紧 ⇒ 与 id2/id5 大幅上升一致，但未独立验证）——**不影响 NO-GO**（裁决只依赖"收益 ≤0"这一实测事实）。④生产态**已合规实现** skill `best-practices/flash_attn_optimize.md:169-181`「持久化 Q 缓冲双向 ownership 闭环」标准形（每任务 1 个完整 `SIG_Q_L1` 轮、全部 `q_l1→L0A` 读在 ownership 区间内、任务尾归还）⇒ **GATE 2 属超出文档范式的扩展**。⑤反模式对照**无未处置项**（「AIC/AIV 未开启 CV overlap」不适用：已有 3-workspace + SEM 0-5 + NUM_STAGES=8 flat 批流水，比该条建议的 `num_stages=2` 更深，其补救 `T.Pipelined`+`AUTO_CV_COMBINE/SYNC` 被 DESIGN §8.6-1/§3.4 明令禁止 = **有意保留，理由 = 设计级禁令 + 手动等价物已在位且更激进**；「冗余全局同步」不适用：`barrier_all` 被 §8.6-6 禁止，全为细粒度 flag/cross_flag = 正是该条推荐形态；该条 L335 `cross_interval=2` 建议**已被 r3-4 判决证伪**；「launch core 数」合规：tasks 2048 ≫ cores 20；「tile size 过小」非可调项：bN 受 L0B 64KB 硬限 + r10-1 穷尽条款永久禁令）。
- **纪律合规**：#1 camodel 未运行 ✅；#2 四个隔离 `TILELANG_CACHE_DIR`（cache_prod/cache_probe/cache_probe_fa/cache_smoke），生产缓存从未与探针共享，未用默认缓存 ✅；#3 `mha.py` 四点 md5 复核全部 `dda3ad08`，探针在 `probe_gate0_qablate/` 且**不以 `mha.py` 命名（无 import 遮蔽）** ✅；#4 GATE 2 由 6 次 A/B + 4 个 metrics arm 实测否决，GATE 1 独立候选明标未实测 ✅；#5 `.orchestrator_state.json` 全程未读未写 ✅；#7 禁触项（§8.6-10 共池相位纪律 / bN=256 / k-v L1 双缓冲 / V 宽通道 / CROSS_INTERVAL / #R5-3）**全部零触碰**（探针仅改 Q prologue 的**执行次数条件**，未改任何缓冲深度、bN、相位结构或 K/V 路径）✅。**环境错误：无**（全程无 ImportError/ModuleNotFoundError/set_env.sh 故障；`TimelineDetail` 的 exit 0 + `[WARN] data collection failed` 属**工具能力边界**非环境故障）⇒ **本轮消耗 iteration_index=24，不申请豁免**。
- next_hint: ①**GATE 2 证据性关闭**（EV 上界 ≤+0.33% ≪ 3% 门，两域同向，case7 显著劣化）——重试前置 = 出现「C 侧 Q 装载暴露」的新结构证据（当前实测为 0），或 pacing 身份从交接延迟链/V.vec 迁移到 C 侧 Q 装载管。②**GATE 1 需改判攻击面才可立项**：作为 C 侧单侧事件删除类，其先验已被本轮两例吸收实证压制；建议要么转向 **V 侧**（case7 pacing 管 = AIV VEC 77.52%，是唯一未被任何已关闭候选攻击过的 pacing 管；但 ③V 宽通道 r9-1 已实测 wall +12.4~16.0% 劣化 ⇒ 需**非宽通道**的 V 侧向量工作量削减路径），要么转向**交接往返链本身**（= 缺口③ CV-over-GM 三 workspace，A3 物理下限，算子规则内不可回收 ⇒ 指向架构级方案 C）。③**本轮方法学资产入库**：「消融上界探针 + 计数器差分法」是**零生产风险、≈25 分钟/候选**的预筛门标准流程，严格强于"实施后测 wall"（能在实现前给出上界），建议作为后续任何单侧速度类候选的**强制前置门**；工具已归档 `msprof_gate0_iter24/tooling/`（`attrib.py` / `ab_cmp.py` / `dump_pipes.py` / `run_ab.sh` / `run_metrics.sh` / `make_probe_qablate*.py`）。④**模型待修**：`analysis_20260928_roofline_model.py:174-178` 的 `else` 分支（逐任务未摊销 ramp）对 **post-#R8 flat 域已过期**——flat 的排空点在**批**边界（~58/核）而非任务边界（~103/核），建议改为 `num_fb` 粒度或按本轮实测将 Q 切片置 0（**推断，非直接测量**：批边界未做独立消融）。⑤`consecutive_no_improvement` 维持 2，距中止条款②的 3 尚余 1；本轮无候选故不递增——**是否继续延展 GATE 专项预算归 Orchestrator/用户裁决**。⑥如实声明的测量缺口：AIV `FLOWCTRL`（真机无此计数器，camodel 专有）、AIC `wait_id0/3/4`（硬件语义上不存在，id0/3/4 为 C 侧 set）、方法 B per-event 时间轴（真机不可达）——**三项均未用模型值/历史值冒充**。

## Iteration 25 — 2026-10-02T13:12+08:00 (mode=perf_measure，**GATE 0b 交接延迟链消融上界探针 · 纯测量轮 · 零 kernel 变更 · 零采纳代码**)

- bottleneck_type: **sync（C↔V 四相位交接往返延迟链）— 本轮以「完全消除」上界探针定量确认为唯一仍在起作用的杠杆**。iter24 已判定 case9 = 交接延迟链（latency-bound）、case7 = AIV VEC paced；本轮把「交接延迟本身」的总 EV 上界钉死：**case9 −9.358%（L3 绝对上界，p=9.75e-05，95%CI [−9.756,−8.961]%）/ case7 −13.426%（p=7.61e-05，95%CI [−13.929,−12.922]%）**。**消融后 pacing 管身份翻转（max-block，直接测量）**：case9 prod AIC 878.22 / AIV 879.11（**平衡至 0.9 µs = 0.1%**）→ y_L3 AIC **803.80** / AIV 557.87 ⇒ **AIC 成新 pacing 管，其 89.4% 时间在 `scalar_mte1_stall`（687.08 µs）而 `aic_mte1_time` busy 仅 335.57（43.7%）⇒ 新限制是 MTE1 队列深度而非吞吐**；case7 prod AIC 857.84 / AIV 858.45（平衡至 0.6 µs）→ y_L3 AIC 600.82 / AIV **742.59** ⇒ **AIV 成新 pacing 管，不可削减的 `aiv_vec_time` max-block 666.39 占其 89.5%**。**两域上界与后续杠杆完全不同，禁止混用或外推。**
- optimization: **无（纯测量轮，不产出采纳代码）**。执行内容 = ①从 `mha.py` 逐行提取 6 个跨核 SEM 的 set/wait 站点、pipe 参数、频率与 1 轮错位环结构（§1.1 表）；②**查得决定性硬件约束**：CANN AscendC `CrossCoreSetFlag(ISASI)` 约束说明逐字「**同一flagId的计数器最多设置15次**」（`ascendc-api-retrieval`），并由 `codegen_ascend.cc:662-664/2871-2878` 确认 lowering = `CrossCoreSetFlag<0x2,PIPE_x>(flag)`（**带 pipe 排空语义**）/ `CrossCoreWaitFlag(flag)`（不带）；③据此推出计数器深度判据并**构造性证明**每级 outstanding 上限（L1 ≤8 = `NUM_STAGES`、L2s ≤2、L3 =1）；④**发现任务书的字面 L2（一侧 wait 全删）为硬件非法**（对侧仍被 wait 的 flagId 无界累积）⇒ 以 **L2s（删全部每槽 SEM0/2/4 对、保留全部每轮 SEM1/3/5 对）** 替代，且 L2s 消除的暴露等待**严格多于**字面 L2（case9 753.11 µs vs 541.58 µs）；⑤生成 **10 个探针**（原位 `if False: <原语句>` 改写 ⇒ 行数恒 1926、diff 行数恰等站点数、无需 `pass`；生成器四重断言）：variant-Y 合法级 4（`l1c9` 4 行 / `l1c7` 2 行 / `l2s` 9 行 / `l3` 18 行）+ variant-X 对照级 4（`xc9l1` 2 / `xc7l1` 1 / `x2s` 4 / `x3` 9）+ 非法级 HANG 演示 2（`l2c9` 6 / `l2c7` 6）；⑥**42 次 interleaved paired wall 运行**（3 repeat × 7 arm × 2 case）+ 1 次独立校准；⑦**9 个 msprof-op metrics arm**（prod/yL1/yL2s/yL3/xL3 × 两域）；⑧双向 HANG 实测（timeout 守卫）。**camodel 本轮完全未运行**（纪律 #1）。
- baseline_time: **case9 868.877 µs / case7 857.173 µs**（plain `msprof --task-time=on` → `op_summary` Task Duration，warmup 5 / iters 20 / n=25 median，3 次 interleaved 重复均值；与 iter24/r10_base 同口径）。**复现验证：case9 vs iter24 869.547 = −0.077%；case7 vs 856.600 = +0.067% ⇒ 均在噪声内**；独立校准运行 case9 median 867.590。prod 侧 median 重复间极差 case9 3.010 µs（0.35%）/ case7 3.200 µs（0.37%）；**paired-diff SEM 0.054–2.252 µs ⇒ 最佳分辨力 0.006% of wall，优于 iter24 的 ≈1.2 µs（0.14%）**。
- candidate_time: **n/a（无候选实现）**。探针 wall（仅用于判决，**禁止采纳/入 baseline/提交**）：**variant Y** case9 y_L1 **786.943**（−9.430%）/ y_L2s **786.467**（−9.485%）/ y_L3 **787.563**（−9.358%）；case7 y_L1 **746.330**（−12.931%）/ y_L2s **744.163**（−13.184%）/ y_L3 **742.093**（−13.426%）。**variant X（非法/污染）** case9 889.773（**+2.405%**）/ 883.960（+1.736%）/ 1006.363（**+15.823%**）；case7 876.863（+2.297%）/ 884.600（+3.200%）/ 904.693（+5.544%）——**全部劣于生产态**。
- improvement: **0（测量轮，无采纳）**。`consecutive_no_improvement` **维持 2**（继承 r10-1/r10-2/iter24），本轮既不递增也不清零（无候选 ⇒ 无「无提升」事件）；中止判定归 Orchestrator。
- precision: **pass（继承）** — 生产 `mha.py` **逐字节未变**（md5 `dda3ad0877ba4df74bbd559f6912eeed` / 1926 行，**轮次开始 + 10 次探针构建后各一次 + 全部硬件运行后 + 轮次结束 = 4 类检查点全部一致**，生成器内置 md5/行数/diff 行数/构建后回读四重断言，10/10 通过），故 session-8/9/10 在案全量门禁（锚-A 9/9 BITEXACT + 锚-B 14/14 + 压力 4×12 nondet=0 + `--level all` exit 0 PRECISION_PASS=29/FAIL=0）与 iter24 的新鲜 `--level l0` 10/10 PASS 继续有效；**本轮未重跑精度测试**（byte-identical ⇒ 无漂移信号，且本轮为纯测量轮无候选需验）。**10 个探针输出数值故意错误并已实测确认**：rel_max **12.478 ~ 15.578（1248%~1558%）**，比 iter24 的 Q 消融（8.447e-01 / 5.900e-01）大 **15~26 倍** ⇒ 12/12 (probe,case) 组合**消融机制毫无歧义激活，排除假阴性**；`l2c9`/`l2c7` HANG 无输出（记为 HANG 证据，不记为数值结果）。`bench_ho.py` 每次运行打印 `NUMERICALLY-WRONG-EXPECTED` + `DO NOT adopt/baseline/commit`；探针明确排除在任何精度判定之外。
- adopted: **NO — 测量轮，无可采纳对象**。GATE 0b 裁决 = **`[GATE0B_VERDICT]` 上界已测定；缺口① CONDITIONAL GO / 缺口③ 延迟侧 NO-GO + 工作消除侧 UNMEASURED**。
- rollback_reason: n/a（生产态全程零变更，无需回滚）。
- **GATE 0b 裁决（总闸门数值）**：
  - **交接延迟总 EV 上界 = case9 −9.358%（L3）/ 最佳实测 −9.485%（L2s）；case7 −13.426%（L3）**。为 3% 采纳门的 **3.1× / 4.5×**。**case9 满额兑现 → M1 从 0.39x 抬到 0.4303x 越线；越线仅需 wall −2.500% = L1 上界的 26.5% 捕获率**（3% 门 = 上界的 31.8% / case7 23.2%）。
  - **阶梯形态两域不同（paired 检验）**：case9 **L1 即饱和**（L3−L1 = +0.620 µs，**p=0.7024 不显著**；L2s−L1 p=0.7112；L3−L2s p=0.1363）；case7 **单调**（L3−L1 = −4.237 µs **p=0.0002**；L2s−L1 −2.167 p=0.0023；L3−L2s −2.070 p=0.0058）。⇒ **case9 只需消除 V 侧两个每槽握手（SEM0 w1 + SEM4 w3）即达全域上界；case7 每深一级再得 ~2 µs。**
  - **缺口①（C 侧 GEMM1/GEMM2 交错，解禁 §8.6-10）= ⚠️ CONDITIONAL GO**。**GO 的定量依据**：上界 −9.43%/−12.93% 达门 3.1×/4.3×，p≤2.1e-04，SEM≤1.19 µs；M1 越线仅需 26.5% 捕获（宽松）；且 iter24 已关闭全部替代路径（GATE 2 上界 ≤+0.33%、路径 2 永久禁令）⇒ **规则内唯一剩余的非零上界杠杆**。**只能是「有条件」的两条实测依据**：①**可行性前置门未过（阻塞项）**——消融后 case9 AIC **89.4% 时间在 `scalar_mte1_stall`** 而 MTE1 busy 仅 43.7% ⇒ 新限制是 **MTE1 队列深度**；交错 GEMM1(fb+1)/GEMM2(fb) 会使**共用 `l0b_pool` 与共用 `SIG_L0KC` 事件流**（§4.5-d-2）的并发 MTE1↔M 令牌流翻倍，正撞 **L0B 64KB 零余量 + MTE1↔M event id 8/8 零余量**（#R8-1 已用 7/8）⇒ **交错形态在本 kernel 内可能根本不可表达**；②**因果假设不完整**——case9 L1 **只**消融 V 侧等待，C 侧暴露等待却**免费坍塌**（`wait_id2` 211.86→**0.59** = −99.72%、`wait_id1` 4.81→0.03、`wait_id5` 2.34→**列消失**），双侧 `aic_time`/`aiv_time` 同步各降 ~76–78 µs ⇒ **闭环里一侧的等待就是另一侧的延迟，互相生成、不可独立归因或独立消融**；「C 等 P 时没活干」是环的**症状**而非独立病因。**建议：不做 §8.6-10 全面解禁，改为一次范围受控、探针门控的试验**——前置门 A（必须先过）= event-id/L0B 预算审计，若 >8 id 或 >64KB ⇒ **直接 NO-GO，无需写 kernel**；前置门 B = 候选实现的 `aic_scalar_mte1_stall` 若高于 y_L1 的 685.05 µs 即弃（说明把延迟换成了更坏的队列阻塞）；爆炸半径按任务书全额计入（全 20 case 复测 + 失去 IR 逐字节守卫 + timeout HANG 前例，本轮再添 `l2c9`/`l2c7` 两例）。
  - **缺口③（架构级 CV-over-GM 消除）= ❌ 延迟侧 NO-GO / ⚠️ 工作消除侧 UNMEASURED**。**NO-GO 依据**：①延迟上界 ≤9.36%/≤13.43% = **1.10×/1.15×**；②**字节侧价值 ≈0**（见下「钳形闭合」）⇒ 宣称的「省 1032 MB / 46% L2 流量」**不是收益**；③**投入产出不成比例**——为换取**至多**与规则内缺口① 同一个上界，需付出框架级/A5 级跨项目投资。**不可越界宣称的部分（如实标注）**：本上界界定的是「**保持双域结构与全部真实工作不变**，消除交接延迟」，**不界定**完整单域 flash 重构（方案 C）——后者还把 workspace 传输从**延迟**变成**被消除的工作**：case9 `aic_fixpipe_time` **337.57 µs = aic_time 的 40.1%**（L0C→GM 的 w1/w3 写）+ `aiv_mte2_time` **213.32 µs = aiv_time 的 25.3%**（GM→UB 读），**本轮按任务书范围未消融 ⇒ 该分量上界 = 未测**；会话 10 记录的「A 层 2.3-3.4x」**不能由本轮数据支持或否证**，本轮只把其中**延迟分量**钉死在 1.10×/1.15×。**后续最小探针（本轮范围外，未做）**：把 `workspace_1`/`workspace_3` 的 GM 写读重定向到单槽哑缓冲（消除传输**工作**而非延迟），须以 §2.2(b) 同款计数器表如实报告「工作项也被改了」。
  - **缺口① 与缺口③ 不可相加**：二者均不改变真实工作与字节，仅消除同一批暴露等待 ⇒ **合计上界 = 同一个数，非其和**（推断，依据 = 环对称性实测；未构造叠加探针）。
- **吸收律（第二轮实证，定量化）**：prod 暴露跨核等待合计（20-block 均值，双侧）case9 **760.59 µs** / case7 **783.32 µs**；实测回收 wall（L3）仅 **81.31 / 115.08 µs** ⇒ **转化率 10.69% / 14.69%，吸收率 89.31% / 85.31%**。**迁移目标实测指认（是迁移而非消失）**：case9 y_L1 `aiv_wait_id3` **0.33 → 308.63 µs（+92104%）**（迁到唯一残留的每轮握手）+ `aiv_scalar_vector_stall` 71.41→**224.09（+213.8%）** + `aic_scalar_mte1_stall` 533.86→**685.05（+28.3%）**；case7 y_L1 `aic_wait_id1` **10.59 → 213.81 µs（+1918.8%）** + `wait_id5` 6.70→15.29（+128.4%）+ `aic_scalar_mte1_stall` 205.14→**421.82（+105.6%）** + `aiv_scalar_vector_stall` 264.52→323.62（+22.3%）。**方法学结论：测得的 `wait_idN` 之和把可回收 wall 高估 9.35×（case9 = 760.59/81.31）~ 6.81×（case7 = 783.32/115.08）；任何以「累加 stall/wait 计数器」得出的 EV 必须除以 7~10 才是 wall 量级。** 与 iter24 的吸收律（移除 32.40 µs 真实 MTE2 工作 → wall +0.29%）为同一规律的两个方向。
- **字节侧非绑定 —— 双向钳形闭合（对 iter24 推断的复核结论 = 成立并加强）**：iter24 = 大幅改字节（`GM_to_L1` −1622.4 KB、`r0_read_cache_miss_allocate` −34.76%/−34.55%、`read_hit_rate` +2.64pp/+2.34pp）→ **wall 不动**（+0.288% p=0.177 ns / +0.871% 劣化 p=0.025）；**本轮 = 几乎不改字节**（全部字节计数器 **Δ=0.0000**：`GM_to_L1`/`read_main_memory`/`L0C_to_GM`/`GM_to_UB`/`UB_to_GM` 与 aic/aiv `mte1/mte2/mte3_instructions`；`aic_read_hit_rate` case9 **+0.01pp** / case7 +0.219pp；`r0_read_cache_miss_allocate` −0.077%/−2.99%）→ **wall 大动 −9.358%/−13.426%**。**补充直接证据：满额消融后带宽占用率仍极低** —— `GM_to_L1_bw_usage_rate` case9 26.32%→**28.80%（71.20% 空闲）**、case7 16.14%→**18.65%（81.35% 空闲）**；`aic_main_mem_read_bw` +9.409%（case9）/+15.557%（case7）**是纯算术而非流量变化——独立自洽性证明（6 位有效数字逐位吻合）**：同 arm `Task Duration` 之比 882.580017/806.679993 = **1.094089**（case9）、861.659973/745.659973 = **1.155567**（case7），与 `aic_main_mem_read_bw` 之比 **1.094090 / 1.155567**、`GM_to_L1_bw_usage_rate` 之比 **1.094088 / 1.155566** 三列逐位相同 ⇒ msprof 带宽列 = `字节数 ÷ Task Duration`，而字节数 Δ=0.0000 已直接证明 ⇒ **构成「字节未变」的第三条独立证据链**（前两条 = KB 计数器 Δ=0.0000、指令计数器 Δ=0.0000）。**注意分母是 kernel Task Duration（wall）而非 per-core `aic_time`**——case7 `aic_time` 降 31.42% 而 wall 只降 13.36%（AIC 已非 pacing 管），误用 `aic_time` 会得出错误的 15.42%。⇒ **wall 是延迟绑定，不是带宽绑定；缺口③ 的 1032 MB 字节收益值 ≈0。**
- **消融未污染工作项（上界可信的判决性判据，全部直接测量）**：case9 `aiv_vec_time` **468.5841 → 468.5841（Δ=0.0000，逐位相同，vector0 与 vector1 皆是，L1/L2s/L3 三级皆然）**、max-block **483.49 → 483.49（Δ=0.00）**；`aic_cube_time` Δ≤+0.01%（190.9943→191.0072）；`aic_mte1_time` Δ≤+0.11%；`aic_mte2_time` Δ≤+0.07%；`aic_fixpipe_time` Δ≤−0.58%。case7 `aiv_vec_time` Δ=**−0.0452 µs（−0.01%）**、max-block 666.44→666.39；`aic_cube_time` Δ≤+0.00%；`aic_mte1_time` Δ≤+1.18%；`aic_mte2_time` Δ≤−2.52%；`aic_fixpipe_time` Δ≤+1.39%。**并排除「收益来自指令变少」混淆**：标量发射当量 `single+2×dual` 恒定 —— case9 AIC 132.84/132.07/131.49/131.32/132.50（最大 Δ **−1.52 µs = −1.14% = wall 的 −0.17%**）、case9 AIV 150.42→149.11、case7 AIC 113.92→112.39、case7 AIV 155.32→153.96；而 `aic_scalar_time` 814.24→**130.24（−684 µs）** ⇒ **纯阻塞移除，非指令削减**。**建议将「发射当量恒定」列为消融类探针的强制第 4 项检查**（与「真实工作管零变化」「字节/指令计数器零变化」「rel_max 发散」并列）。
- **变体 X vs 变体 Y —— 任务书假设被实测证伪（本轮第二大发现）**：任务书设想「X（只删 wait）= 测纯延迟暴露」；**实测 X 不是纯延迟探针，而是「延迟移除 + 新增排空阻塞」，且新增项压倒性主导**：X−Y paired 差 **case9 +102.830 µs（L1，t=198.55，p<1e-4）/ +218.800 µs（L3，t=126.31，p=1e-4）**、**case7 +130.533（L1，p=1e-4）/ +162.600（L3，p=1e-4）**；X 的 wall 全部**劣于生产态**（case9 +2.405%/+1.736%/**+15.823%**，case7 +2.297%/+3.200%/+5.544%，p≤6.7e-03）。**机制（计数器直接指认）**：`CrossCoreSetFlag<mode,PIPE>` 是**阻塞至 PIPE 排空**的标量指令（`codegen_ascend.cc:2871`）；生产态由配对 wait 保持管道浅占用使其近乎免费，**删 wait 保 set ⇒ 每次 set 变成生产态从未有过的全管排空阻塞点**。证据：case9 x3 的**发射当量 132.84→132.50（指令数没变）**而 `aic_scalar_time` 814.24→**953.43**（= `aic_time` 953.48 的 **100.0%**）、`scalar_mte1_stall` +334.03、`scalar_cube_stall` 0.20→**78.00（+38055%）**、`aiv_wait_ib` 1.31→**20.75**，同时 `aiv_vec_time` **Δ=0.0000**。⇒ **X 的全部读数不可用作上界，仅作「≤15 约束有真实计时后果」的负对照。**
- **HANG 定位与硬件计数器判据（双向实测）**：CANN 文档硬约束「**同一flagId的计数器最多设置15次**」+ 生产态 `CROSS_INTERVAL=1` ⇒ 每核每 flagId **~461 次 set**，其合法性仅因 set/wait 每槽严格 1:1 交错。**推出的判据（实测确认）：计数器越过 15 后钳位，当且仅当该 flagId 不再有任何 `CrossCoreWaitFlag` 时无害；若落后侧仍 wait 它，钳位后最多放行 15 次 wait ⇒ 永久阻塞。** 实测：**「无残留 waiter」的全部 6 个 X 构造 + 4 个 Y 构造均 exit 0 无 HANG**（含 x3 = 删全部 9 个 wait、保留全部 9 个 set）；**「一侧 wait 全删」的 `l2c9`/`l2c7` 双双 HANG（240 s SIGKILL，exit 137）**。⇒ **消融级次不是单调安全的：L1（删一对）与 L3（删全部）合法，字面 L2（删一侧全部 wait）非法并 HANG**；设计逐级阶梯时必须**逐级做 outstanding 深度证明**（每槽 SEM ≤ `NUM_STAGES`=8、每轮 SEM ≤2、L3 =1 仅 init），而非假设「删得越多越接近上界故越安全」。L1 的 8/15 深度**与生产态同深度**（生产态 SEM2 每轮亦积压至 8，因 C.GEMM2(fb) 头 wait SEM5 = V.phase2(fb−1) 结束而 V.phase2 在 V.phase1 之后）⇒ L1 未引入任何生产态不存在的计数器深度。**HANG 归因局限（如实）**：`l2c9`/`l2c7` 的 stdout 因块缓冲被 SIGKILL 丢弃（日志仅 329 字节 import 警告），**无法从日志定位阻塞在哪一条 SEM**；定位依据是**构造性证明**（对侧仍被 wait 的 flagId 无界累积）而非日志，且无任何 profile 产物。
- **`wait_idN` 语义判决性确认（超出任务书）**：msprof `PipeUtilization.csv` 在生产态**只**输出 `aic` 的 id1/id2/id5 与 `aiv` 的 id0/id3/id4 —— **恰为 6 个跨核 SEM 的方向分割，无一多余无一缺失**（核内 `SetFlag/WaitFlag` 的 SIG id 0-4 一个都不出现，它们归入 `*_mte1_stall`/`*_cube_stall`/`*_vector_stall`/`wait_ib` 桶）⇒ **`wait_idN` ↔ `CrossCoreWaitFlag(N)` 1:1**。**且 msprof 对 20-block 全零的 `wait_idN` 列直接删除** ⇒ **列的存亡本身即激活信号，比读数值更抗假阴性**：prod case9 六列齐全 → y_L1 仅剩 aic id1(0.03)/id2(0.59) + aiv id3(308.63)（id0/id4 已消融、id5 间接归零）→ y_L2s 仅剩 aic id1(0.06) + aiv id3(309.44) → **y_L3 无任何 wait_id 列**。双 AIV 对称（`wait_id0` 295.34 vs 294.14，差 <1.3%）⇒ 消融在两 AIV 上对称生效无单侧偏斜。
- **附加发现（超出任务书，均为直接测量）**：①**交接是 run-to-run 方差主源** —— 生产态 per-launch stdev 是探针的 **4~12 倍**（case9 5.08–5.72 vs 0.94–1.64；case7 4.99–6.66 vs **0.48–0.68**），case7 y_L3 三次重复极差仅 **0.20 µs = 0.027%**；prod 侧 median 重复极差 case9 3.010 µs / case7 3.200 µs 亦远大于探针（0.20–2.74）。②**L2 写分配行为随时序改变而迁移（非流量变化）**：case9 y_L3 `aic_write_cache_miss_allocate` 768.5→**0.5（−99.93%）**、`aic_write_cache_hit` +768.0、`write_hit_rate` 98.26→**99.9989%**，而 `L0C_to_GM_datas` **Δ=0.0000** ⇒ 纯分配策略漂移，**不得误读为流量削减**。③`aiv_mte2_time` 在各 arm 间非单调波动（case9 213.32→200.64→217.55→206.85，±6%）⇒ GM→UB 读取时长受并发争用影响，**不是稳定判据**，故本轮以指令数与 KB 计数器（Δ=0.0000）作为字节不变的判据而非时间。④生产态 `wait_idN` 与 iter24 存在偏移（`aiv_wait_id0` 295.34 vs 272.80 = +8.3%；`aic_wait_id2` 211.86 vs 176.32 = +20%；case7 侧一致性好：534.10 vs 530.89 = +0.6%、67.28 vs 68.49、164.01 vs 161.40）—— 因 metrics arm 为**单发**（warmup 0 / iters 1）；**wall 口径（3 重复 × n=25 median）不受影响，§吸收律转化率因分子分母同取本轮数据故内部自洽**。
- **纪律合规**：#1 camodel 未运行 ✅；#2 缓存隔离 = `cache_prod`（仅生产）+ `cache_ho_<probe>`×8 + `cache_smoke`，生产缓存从未与探针共享，未用默认缓存 ✅；#3 `mha.py` 4 类检查点 md5 全部 `dda3ad08`/1926 行，10 探针均在 `probe_gate0b_handoff/` 且名 `mha_ho_*.py`（**不叫 `mha.py`，无 import 遮蔽**；bench 以 `sys.path.insert(0,_HERE)` 保证探针目录优先）✅；#4 **禁止预测性否决** —— variant X 与字面 L2 **均实建实跑**（timeout 守卫），HANG 为实测（exit 137）而非预测；≤15 约束用于**设计**安全阶梯后**双向实测确认**；唯一未实测项（方案 C 工作消除分量 G1、缺口① 捕获率下界 G2）已明确标注为缺口且给出后续探针设计，**未以推断冒充结论** ✅；#5 `.orchestrator_state.json` 全程未读未写 ✅；#6 A/B 口径与 iter24 逐字一致（同 `run_ab.sh` 结构 / 同 `parse_prof.py` / warmup 5 / iters 20 / n=25 median / 3 次 interleaved paired / paired Δ+均值差+stdev+SEM+t+**精确双侧 p**+95%CI；p 用 df=2 闭式 CDF，对 scipy 校验至 1e-6 **且复现 iter24 的 t=2.048→p=0.177 / t=6.201→p=0.025**）✅；#7 探针 rel_max 12.5–15.6 **仅作激活证明**，未作采纳依据、未入任何 baseline、未提交，`adopted=false` ✅；#9 禁触项全部零触碰 —— `DESIGN §8.6-10` 共池相位纪律未触、bN=256 未触、k/v L1 双缓冲未触、V 宽通道未触、**`CROSS_INTERVAL` 仍为 1 未改**、#R5-3 未触、**GATE 2 `q_l1` 双缓冲未复活**（探针仅改写跨核 flag 语句的发射与否，未改任何缓冲深度、bN、相位结构、K/V 路径或 Q prologue）✅。**环境错误：无导致任何测量失效者** —— `l2c7` HANG 演示被 SIGKILL 后（04:25:16 UTC）npu-smi 列出一个残留设备上下文（pid 3856768，python3，275 MB），其后 4 次 X smoke（04:28:08–04:29:15）、42 次 A/B（04:29:31–04:45:11）、9 个 metrics arm 全部正常完成，npu-smi 于 04:45:11 与轮末（05:11:58）均报 `No running processes found in NPU 5`，Health OK ⇒ **未丢弃任何 arm，本轮消耗 iteration_index=25，不申请豁免**。
- **方法学资产更新（建议写入 `tilelang-perf-optimization`，新增「交接类探针」专章 6 条）**：①**跨核 SEM 消融必须成对删 wait+set（variant Y）；只删 wait（X）在 Ascend 上是无效探针**（实测 X 比生产态慢 +1.7~+15.8%，X−Y 达 +102.8~+218.8 µs；根因 = `CrossCoreSetFlag<mode,PIPE>` 的排空阻塞；识别特征 = 发射当量不变而 `scalar_time`/`*_stall`/`wait_ib` 齐涨）。②**硬件计数器上限：同一 flagId 最多 15 次未消费 set**（CANN 约束说明）⇒ 钳位「无 waiter 则无害、有 waiter 则致命」⇒ **消融级次非单调安全，字面「删一侧全部 wait」非法并 HANG；须逐级做 outstanding 深度证明**。③**`wait_idN` 列存亡本身即激活信号**（msprof 删全零列）；且生产态只出现跨核 SEM 的方向分割 ⇒ `wait_idN` ↔ `CrossCoreWaitFlag(N)` 1:1，核内等待在 stall 桶。④**必须用发射当量（`single+2×dual`）排除「指令变少」混淆**（本轮 Δ≤0.17% of wall 而 `scalar_time` −684 µs）。⑤**吸收律定量化：`wait_idN` 之和把可回收 wall 高估 6.8~9.4×；必须同时报告迁移目标计数器以证明是迁移而非消失。** ⑥**原位 `if False: <原语句>` 改写模式**（替代 iter24 的条件改写，适用于需**删除**语句的消融）：行数恒定 ⇒ diff 行数可精确断言；不需 `pass`；`if <Python 静态 bool>:` 在 `mha.py` 中已被 `if is_causal`/`if not io3_is_alias`/`if need_mask` 大量依赖 ⇒ 解析安全性有既有先例；本轮 10 探针 1–18 行 delta **全部一次编译通过，零解析错误**。
- **如实声明的测量缺口**：**G1** 方案 C 的「工作消除」分量上界未测（case9 `aic_fixpipe` 337.57 µs/40.1% + `aiv_mte2` 213.32 µs/25.3%）——任务书范围限定为「交接**延迟**本身」，该分量需另一类探针（重定向 workspace 到哑缓冲，会同时改变字节与指令数，属不同口径），**已给出设计未执行**；**G2** 缺口① 的捕获率下界未测——**无法用消融法给出**：C 侧单独消融必然 HANG（C 的 set 是 V 的 wait 所必需），而保留 C 的 set 又必然引入 X 型排空污染 ⇒ **「C 侧单独交错」在消融法下不可干净隔离，这是方法学的硬限制而非本轮疏漏**；**G3** 逐事件时间轴仍不可达（iter24 结论沿用，本轮未重试）；**G4** `l2c9`/`l2c7` 无计数器数据（HANG 于首次 launch 前）；**G5** metrics arm 为单发（与 iter24 同口径，case9 prod `wait_idN` 有 8~20% 偏移，wall 口径不受影响）；**G6** 未测 case9/case7 以外的 18 个 case（任务书仅要求两域；**上界不得外推**至 decode 域 / `single_kv_block` 域 / Form B `gemm_v0` 域，其交接结构不同）。**六项均未用模型值或历史值冒充本轮实测。**
- next_hint: ①**规则内 EV 版图已完整闭合**：GATE 2 `q_l1` 双缓冲 ❌（iter24 上界 ≤+0.33%）/ GATE 1 `unit_flag` ❌ / 路径 2 bN=256 ❌ 永久禁令 / k-v L1 双缓冲·V 宽通道·`CROSS_INTERVAL` 扩大·#R5-3 ❌ / **字节带宽侧 ❌（本轮+iter24 双向钳形闭合）** / **缺口① ⚠️ 唯一开放（CONDITIONAL GO，前置门 = event-id/L0B 预算审计）** / **方案 C 工作消除分量 ⚠️ 开放但 UNMEASURED**。②**收尾判据建议**：若前置门 A（event-id/L0B 审计）判为不可表达 ⇒ 缺口① 关闭；此时若用户亦不为方案 C 的工作消除分量立项 ⇒ **规则内 + 架构级延迟侧杠杆确已全部穷尽，建议收尾**，并把 case9 M1 线 0.4 记为「需工作消除级重构，非延迟级手段可达」。反之缺口① 值得**一次**范围受控试验（EV/风险比 3.1× 于门，M1 越线仅需 26.5% 捕获）。③**若推进缺口①，实施蓝图锚点**：目标形态 = 让 C 在 `wait(SEM_WS2_V2C)` 期间发射 GEMM1(fb+1) 的 K 装载与 mma；**硬前置** = 为 fb+1 的 K/V/Q 工作集找到不侵占 fb 的 `l0b_pool`/`k_l1`/`v_l1`/`q_l1` 的容量（L0B 已 64.00KB 顶格、L1 已含 k_l1+v_l1+p_l1+q_l1+l0a_s+l0a_o），且 MTE1↔M 令牌流必须 ≤8 id（现 7/8）⇒ **极可能需要先做缓冲重排（设计级），而非直接交错**。④**若推进方案 C 工作消除分量**：先跑 §缺口③ 所述哑缓冲探针（≈30 分钟，零生产风险）取得工作消除上界，再决定是否值得框架级立项——**本轮已证明延迟分量只有 1.10×/1.15×，故方案 C 的全部想象空间都在工作消除分量上，该数未测之前不应做投资决策**。⑤**工具已归档**：`msprof_gate0b_iter25/tooling/`（`make_probe_handoff.py` = SITE 表 + 计数器深度证明 + 四重断言；`ab_stats.py` = df=2 闭式 p；`run_ab.sh`/`run_metrics.sh` = iter24 同口径；`ab_cmp.py`/`attrib.py`/`dump_pipes.py` = iter24 复用）；探针与 bench 在 `probe_gate0b_handoff/`；wall 43 JSON + metrics 9 arm×8 CSV + HANG/rel_max 证据 10 份在 `msprof_gate0b_iter25/`。⑥`consecutive_no_improvement` 维持 2，距中止条款②的 3 尚余 1；本轮无候选故不递增。

## Iteration 26 — 2026-10-02T15:10+08:00 (mode=perf_measure，**GATE 0c 收尾双探针：GATE A 静态预算审计 + G1 工作消除分量实测上界 · 纯测量/审计轮 · 零 kernel 变更 · 零采纳代码**)

- bottleneck_type: **本轮为「杠杆穷尽性判决」而非新瓶颈定位——两个开口同时封死**。①**缺口①（C 侧 GEMM1(fb+1)/GEMM2(fb) 交错，解禁 `DESIGN §8.6-10`）= 资源/容量瓶颈（L0B 64KB 硬限），GATE A 判 FAIL**：以生产 IR 实测证明四种实现（A-P1 / A-P2 / B-mma / #R8-1）的 L0B **全部 64.00KB / 64.00KB = 100.0% 顶格零余量**（`pipe.InitBuffer(ascend_l0b, 65536)` 逐字相同），交错使 B 操作数在途总量翻倍 ⇒ Form B-mma 需 **128KB = 硬限 2.00×**，触发 iter25 预登记的「>64KB ⇒ 直接 NO-GO，无需写 kernel」规则。②**缺口③ 工作消除侧 = sync 吸收（与延迟侧同源），G1 实测归零**：消除六段 CV-over-GM workspace 往返的真实硬件工作后，收益 **case9 −0.598%（1.0060×）/ case7 −0.042%（1.0004×，p=0.669 ns）**，绝对上界臂 W3 反而**显著劣化 case9 +0.340%（p=0.033）/ case7 +0.940%（p=0.00092）**。③**新瓶颈知识（根因级，实测）**：CV-over-GM 的 workspace 往返**物理上是 L2 往返而非 DRAM 往返**——生产态 C 侧写 L2 命中 **98.2629%**（hit 43469.8 / miss 768.0）、V 侧读 L2 命中 **99.9989%**（`aiv_r0_read_cache_miss_allocate` = **0.55**）；只删写侧会使读侧冷启（miss **0.55 → 973.70 = ×1770**、hit rate −2.20pp、case7 `aiv_mte2_time` **+12.42%**、达成带宽 **−11.01%**）⇒ **写侧不可单独移除**。④case7 的 pacing 身份在本轮**独立复现**为 AIV VEC 主导（AIV wall 均值 837.33 > AIC 836.45；`aiv_vec_time` 648.99 µs = aiv_time 的 **77.51%**，critical block **77.56%**）⇒ 纯 C 侧编排改造对 case7 的 EV 结构性 ≈0。
- optimization: **无（纯测量/审计轮，不产出采纳代码；生产态零变更）**。执行内容 = **子任务 A（GATE A，零编译零运行 + 一个受命的最小编译探针）**：①以 `budget_r81.py`（P4 法，纯正则解析已生成 C++）对生产 IR（`/tmp/opencode/r10_src_rollback`，provenance 三重锁定：`diff -rq` vs `r8_src_iter3` exit 0 + `G_case9` md5 `35a21f67…`/`G_case7` md5 `dcde9ec1…` 与 `dump_final.log` 逐字一致 + 本轮复采同 md5）与 #R8-1 IR（`r8_src_iter1`，`dbe5760b…` 与 `dump_iter1.log` 一致）做 **A-P1/A-P2/B-mma × {slot 级, chunk 级} 逐形态 L0B 工作集枚举**；②以 `GetWithOffset(size, offset)` **站点数**判定别名结构（A-P2 的 L0A/L0C 各仅 1 站点 ⇒ 别名成立；A-P1 各 2 站点 ⇒ 物理分立；B-mma L0C 2 站点 @0/@32768 ⇒ **生产态两池本已同时活**，任务书 L0C 假设核实为真）；③**逐 pipe 对 event id 枚举**并**逐字核对 #R8-1 的 7/8 分配**（`mha_r81_edit_v1.py:243-245`：`SIG_L0KC`=0 承载槽 0..3 → id 0-3、`SIG_L0AO`=4 → 4,5、`SIG_L0AS`=6 任务级单 id），并指认 **M↔FIX 的 `SIG_L0C` 双语义共流**（GEMM1 用 `+side`=fb 奇偶 `mha.py:1553/1556/1558`，GEMM2 用 `+chunk`=n-chunk `:1620/1621/1623`+`:1632/1634/1636`，仅因相位串行而合法 ⇒ 交错须拆为 4/8）；④**A.4 最小编译探针**（`probe_a4_l0b_mixed_shape.py`，compile-only `get_kernel_source()`，**从不 launch**）：`ctrl [2,128,128]^T` / `alpha [2,64,128]^T` / `beta [2,128,64]` / **`gamma` 两池共存** ⇒ **4/4 COMPILE_OK**，`copy_l1_to_l0b<half,128,64,true>` 与 `<half,128,64,false>` 双双实例化 ⇒ **64 列 L0B fractal NZ 在 A3 合法**、**混合 shape 双声明确实绕开 `T.view/T.reshape` 的 `tvm.tir.Buffer` 不可构造问题**（但 `gamma` 只覆盖 2+2=4 槽 = 64KB 顶格，B-mma 需 4+4=8 槽 = 128KB ⇒ **合法性不能拯救缺口①**）。**子任务 B（G1）**：⑤从 `mha.py` 定位六段 ws 往返（S1 C-FIX 写 w1 `1557`/`1104`；S2 C-FIX 写 w3 `1622`+`1635`/`1159`；S3 V-MTE2 读 w1 `507`；S4 V-MTE2 读 w3 `610`；S5 V-MTE3 写 w2 `567`；S6 C-MTE2 读 w2 `1599`/`1139`），**明确不消 Output 写**（`634`/`453`/`794`，非 ws 往返，单域重构仍须写结果 ⇒ 上界偏保守 ⇒ 对证伪更强）；⑥生成 **3 探针**（原位 `if False:` 改写；**本轮新增多行处理**：三个 4 行 `T.copy(` 站点只改首行为 `if False: T.copy(`，续行与收尾 `)` 自然成死体（括号续行对缩进不敏感）⇒ 行数仍守恒 1926、diff 仍每站点 1 行）：`mha_we_w1` 5 站点 `783b32b6` / `w2` 7 站点 `08e9a359` / `w3` 10 站点 `9d780015`；⑦**IR 级激活证明**（`verify_ir_workelim.py`，compile-only，2 case × 4 arm = 8 次编译 + 指令清单差分 + **逐调用溯源**）；⑧**6 次 smoke（`--check` 取 rel_max）→ 24 次 interleaved paired wall → 8 个 metrics arm**。**camodel 本轮完全未运行**（纪律 #1）。
- baseline_time: **case9 869.543 µs / case7 856.683 µs**（plain `msprof --task-time=on` → `op_summary` Task Duration，warmup 5 / iters 20 / **n=25 median**，3 次 interleaved 重复均值；与 iter24/25/r10_base 同口径）。**复现验证：case9 vs iter25 868.877 = +0.077%、vs iter24 869.547 = −0.0005%；case7 vs iter25 857.173 = −0.057%、vs iter24 856.600 = +0.010% ⇒ 全部在噪声内**。prod 侧 median 重复极差 case9 3.680 µs / case7 0.830 µs；**paired-diff SEM 0.245–1.335 µs = wall 的 0.028%–0.154%，最佳臂（case7 w3）SEM 0.245 µs = 0.028% of wall，为本专项历轮最佳**。
- candidate_time: **n/a（无候选实现）**。探针 wall（仅用于判决，**禁止采纳/入 baseline/提交**）：**case9** w1 **864.343**（−0.598%，p=7.65e-03，95%CI [−0.824,−0.372]%）/ w2 **874.613**（**+0.583%**，p=0.063）/ w3 **872.500**（**+0.340%**，p=3.30e-02，CI [+0.067,+0.613]% 全正）；**case7** w1 **856.327**（−0.042%，**p=0.669 ns**，CI [−0.403,+0.319]%）/ w2 **864.847**（**+0.953%**，p=6.22e-03）/ w3 **864.740**（**+0.940%**，p=9.22e-04，CI [+0.818,+1.063]% 全正）。**三级非单调（W1 < prod < W3 ≈ W2）⇒ 不存在「消得越多越快」的趋势。**
- improvement: **0（测量/审计轮，无采纳）**。`consecutive_no_improvement` **维持 2**（继承 r10-1/r10-2/iter24/iter25），本轮无候选 ⇒ 既不递增也不清零；中止判定归 Orchestrator。
- precision: **pass（继承）** — 生产 `mha.py` **逐字节未变**（md5 `dda3ad0877ba4df74bbd559f6912eeed` / 1926 行 / `ast.parse` OK；检查点 = 轮次开始 + A.4 编译前后 + 3 次探针构建各一次 + 8 次 IR 编译 + 6 smoke + 24 wall（每 repeat + 收尾）+ 8 metrics + 全部分析后 + **轮次结束**，**全部 UNCHANGED**；生成器内置 md5/行数/diff 行数/构建后回读/flag 家族存活数六重断言，3/3 通过），故 session-8/9/10 在案全量门禁（锚-A 9/9 BITEXACT + 锚-B 14/14 + 压力 4×12 nondet=0 + `--level all` exit 0 PRECISION_PASS=29/FAIL=0）继续有效；**本轮未重跑精度测试**（byte-identical ⇒ 无漂移信号，且纯测量轮无候选需验）。**3 个探针输出数值故意错误并已实测确认**：rel_max（6/6 组合，全部 exit 0 零 HANG）**w1 = 1.000e+00（两 case）/ w2 = 1.959e+00（case9）、2.036e+03（case7）/ w3 = 1.989e+00（case9）、1.292e+03（case7）**——量级与 iter24（8.4e-01/5.9e-01）、iter25（12.5~15.6）同属「明显错误」区间；**逐次不同是探针性质**（workspace 载荷为未初始化/陈旧 GM），激活确认改由**更强的 IR 级差分**承担，rel_max 仅作辅助。`bench_we.py` 每次运行打印 `NUMERICALLY-WRONG-EXPECTED` + `DO NOT adopt/baseline/commit`。
- adopted: **NO — 测量/审计轮，无可采纳对象**。GATE 0c 裁决 = **`[GATE0C_VERDICT]` GATE_A FAIL（缺口① 关闭，无需写 kernel）+ G1 工作消除侧上界 case9 −0.598%（1.0060×）/ case7 −0.042%（1.0004× ns），绝对上界臂 W3 双双显著劣化 ⇒ EV 版图彻底闭合 ⇒ 建议置 SUCCESS 收尾**。
- rollback_reason: n/a（生产态全程零变更，从未写入 ⇒ 无需回滚动作）。
- **GATE A 裁决（子任务 A，逐形态）**：
  - **A.1 L0B（第一顺位独立预算门，遵 scoping 盲区教训 S10-3-4）**：现状 IR 实测 **A-P1 32KB@0 + 32KB@32768（两物理池）= 64.00KB/100%**、**A-P2 单池 65536B@0 = 64.00KB/100%（`l0b_o` 别名 `l0b_s`，`mha.py:848`）**、**B-mma 单池 `[2,KC=128,bN=128]` = 64.00KB/100%**、**#R8-1 单池 `[4,KC=64,bN=128]` = 64.00KB/100%**。交错需求：**B-mma slot 级 2×[128,128]+2×[128,128] = 128KB ❌ 2.00×**；**B-mma chunk 级 load-first 4×[64,128]+4×[128,64] = 128KB ❌ 2.00×**；**B-mma chunk 级最小并发 2+2×16KB = 64KB 顶格但 4 chunk 挤 2 槽 ⇒ 相位内覆写 + B 操作数二次读取 + 相位深度 4≥3 ⇒ 同时命中 session-10 毒化机制 (a)(b) ❌ 结构性毒化**；**A-P2 slot 级解别名 128KB ❌ 2.00×**、**chunk 级 2×[64,128]+2×[128,64] = 64KB ⚠️ 顶格零余量但容量过门（load-first、深度 2<3 不触 (b)、槽数=chunk 数不触 (a)）**；**A-P1 slot 级 32+32 = 64KB ⚠️ 顶格过门（池本已分立、深度 1）**、chunk 级（KC=NC=32）32KB 余 32KB（**32 列合法性未实测**）。**#R8-1 是存在性证明**：它用 4 槽×16KB 承载 GEMM1 的 4 个 kc chunk 与 GEMM2 的 4 个 (kc2,nc) slice，**靠的正是二者相位串行时分复用同一 4 槽**（K^T 装于 `:1114-1123`、V 重装于**同一批槽** `:1175-1184`）⇒ 交错取消时分复用 ⇒ 槽数 4+4=8 ⇒ **128KB**。**这不是估算，是把一个全门禁通过、零探针成本的实现按其自身结构直接翻倍。**
  - **A.2 event id（`EVENT_ID_MAX 8` 已核源码 `event.hpp:14`，`:223` 模回绕；`SYNC_FLAG_ID_MAX 16` `type.hpp:284`，SEM 0-5 余 10，任务书已排除）**：**#R8-1 的 7/8 逐字核实无误**（余量 1），其可达性依赖两点节流——`SIG_L0KC` 一条流承载两相（**仅因相位串行合法**）+ `SIG_L0AS` 任务级单 id。**MTE1↔M**：B-mma 现状 6/8（`SIG_L0KC` 0,1 + `L0AO` 2,3 + `L0AS` 4,5）→ 交错须按相位分流 `SIG_L0KC` ⇒ 生产几何 **8/8 零余量**（L0AS 归并则 7/8）、**#R8-1 几何 4+4+2+1 = 11/8 ❌ 超 3**；A-P2 交错 **8/8 零余量**（任务级归并可降至 4/8）。**M↔FIX**：现状 2/8 且**存在双语义共流**（`l0c_s[side]` vs `l0c_o[chunk]`）⇒ 交错须拆 `l0c_s` 0,1 + `l0c_o` 2,3 = **4/8 ✅ 余 4**。**MTE2↔MTE1** 4/8 不变 ✅；**V↔MTE2 / MTE3↔V** 不变 ✅（AIV 侧不受 C 侧交错影响）。⇒ 瓶颈在 MTE1↔M；因 A.1 已对 B-mma 判 ❌，event id 在 B-mma 上非独立否决点，但为**第二道零余量/超限警示**。
  - **A.3 附带预算**：**L0C** B-mma 现状 32KB@0 + 32KB@32768 = **64KB/128KB 且两池本已同时活 ⇒ 交错后仍 64KB ✅（任务书假设核实为真）**；A-P2 现状单池 64KB（`l0c_o` 别名 `l0c_s`）⇒ 解别名 **128KB/128KB ⚠️ 顶格**；A-P1 96KB/128KB 余 32KB ✅。**L0A** B-mma 32KB/64KB ✅ 不翻倍（Q 任务常驻、P ping-pong 不变）；**A-P2 解别名 64KB/64KB ⚠️ 顶格，且任务书指出的 Q 重载成立**（GEMM2 clobber `l0a_s` ⇒ 每 GEMM1 槽须重载 Q ⇒ MTE1 流量 ×2 ⇒ 需 4 L0A side ⇒ **128KB ❌**）；A-P1 48KB/64KB 余 16KB ✅。**L1** 152.00/96.00/56.00 KB vs 511.75KB（29.7%/18.8%/10.9%）**全形态不变 ✅ 余 ≥359.75KB**；**UB** 126.00/131.50/143.50 KB vs 191.75KB（65.7%/68.6%/74.8%）**不变 ✅ 余 ≥48.25KB** ⇒ L1/UB 不构成障碍；真正警示是 **A-P2 解别名后的三重顶格**（L0B 64/64 + L0A 64/64 或 128KB❌ + L0C 128/128）。
  - **A.4（本轮唯一编译动作，compile-only 零运行）**：**4/4 COMPILE_OK**——`ctrl [2,128,128]^T` 64.00KB（生产形状正向对照，证 harness 有效）；`alpha [2,64,128]^T` 32.00KB；`beta [2,128,64]` 32.00KB；**`gamma` 两池共存 64.00KB，站点 `32768@0` + `32768@32768`（两个物理独立池）**，`copy_l1_to_l0b<half,128,64,true>` + `<half,128,64,false>` + `tl::ascend::mma` ×2 全部实例化。⇒ **①64 列 L0B fractal NZ 在 A3 合法（任务书「不可假设」项已实测关闭）；②两个不同 shape 的独立 `T.alloc_L0B` 声明可共存 ⇒ 确实绕开 `T.view/T.reshape` 的 `tvm.tir.Buffer` 不可构造问题；③但合法性 ≠ 可行性**——`gamma` 的 64KB 正是顶格零余量且只覆盖 4 槽，B-mma 需 8 槽 = 128KB，**A.4 的正面结果不能拯救缺口①**（且 #R8-1 早已给出更优解：沿 block_N 对 PV 归约做 K-split（`KC2=64`）使 GEMM2 的 B-tile 变成 V 的**行切片**，4 槽同向均一，混合 shape 无必要）。
  - **A.5 判决 = `[GATE_A_VERDICT] FAIL`（逐形态 PARTIAL）**：**B-mma（case9/case4，唯一有实测非零 EV 的形态：iter25 上界 −9.358%、M1 线 0.39x→0.4303x 越线在此）= ❌ FAIL 决定性**；**A-P2（case7）= ⚠️ PARTIAL 但 EV≈0 ⇒ 不立项**（本轮独立新证据：case7 生产态 AIV wall 837.33 > AIC 836.45 且 `aiv_vec_time` = aiv_time 的 77.51%/critical 77.56% 为不可削减真实计算，而缺口① 是纯 AIC 编排改造 ⇒ 动不了 pacing 管；与 iter25 §4.1 结论独立复现）；**A-P1（case2/5）= ⚠️ PARTIAL 但 EV 未测且不属 M1/cann-bench 代表性 L 级用例 ⇒ 不立项**。**HANG 风险等级（对两个「容量过门」形态）= HIGH，不可由静态审计排除**：前例 #R10-1 bN=256 全形态 aicore timeout 507014 HANG 且**4 轮静态令牌审计全平衡仍挂**（S10-3-2 推论）；A-P2 chunk 级具备该前例的三个高危特征——三重顶格零余量（S10-3-4「双顶格零余量警示必要不充分」）+ 并发 MTE1↔M 令牌流翻倍至 8/8 + 解禁相位纪律引入生产安全谱系从未验证过的相位内并发 L0B 占用 ⇒ 若要推进（本判决不建议）**判决必须靠运行时变体矩阵**（按 S10-3-3 E 系毒化矩阵方法学，~7 小时量级）。**⇒ 缺口① 关闭，无需编写任何 kernel，无需 Orchestrator 调度 analyst 解禁 §8.6-10。**
- **G1 裁决（子任务 B）**：
  - **探针设计与 token 收支（对任务书指令的一处有意偏离，附完整理由）**：任务书要求 variant-Y 成对删除（删 DMA 亦删其 pipe-pair flag 轮）；**本轮改为只删 DMA 语句、完整保留每一对 flag 轮（intra-core `SIG_*` 与 cross-core `SEM_*` 全部原位不动）**。理由：①variant-Y 规则的真实目的是禁止 X 型非法构造（**删 wait 保 set**，iter25 实测慢 +1.7~+15.8%，根因 = `CrossCoreSetFlag<mode,PIPE>` 阻塞至 PIPE 排空），本轮**既不删 wait 也不删 set** ⇒ X 病理结构性不可能；②**token 收支比 iter24 的「1 dangling」更严格**——生成器逐字断言四个 flag 家族存活站点数与生产态**完全相等**（`wait_cross_flag` **27/27**、`set_cross_flag` **32/32**、`set_flag(` **137/137**、`wait_flag(` **137/137**，3/3 通过），IR 级复核 `CrossCoreSetFlag` **9→9**、`SetFlag` **62→62**（case9）/**44→44**（case7）、`WaitFlag` **53→53**/**35→35**，Δ 全 0 ⇒ **零 dangling，跨核 HANG 结构性不可能**；③**≤15 硬约束自动满足**（未改任何 set/wait 配对 ⇒ 每 flagId outstanding 深度与生产态逐一相同）；④**测量隔离更干净**（连 flag 轮一起删会同时删标量指令并重排 M/FIX/V 握手次序，把「ws 工作被消除」与「协议被简化」混淆）；⑤空管道的 `set_flag` 立即排空**正是**「该工作已不存在」的物理含义，且**没有 waiter 被移除**故不会留下深占用管道。循环界（`num_fb`/`batch_iters`/`my_count`/`eff_kv_iters`/`total_flat`）纯由标量索引导出、从不读 ws **数据** ⇒ 载荷消失不可能造出无界循环（沿用 iter25 L3 论证）。
  - **消融激活（三重证据，全部直接测量）**：①**rel_max** 6/6 组合明显发散（见 precision 项），全部 exit 0 零 HANG；②**IR 级指令清单差分（比 rel_max 更强）**：case9 `copy_l0c_to_gm` **3→0**（w1×1 + w3 colseg×2）/ `copy_gm_to_ub` **2→0**（W2 起）/ `copy_ub_to_gm` **2→1**（W3 只剩 Output 写，设计如此）/ `copy_gm_to_l1` **4→3**，且**逐调用溯源**证明 prod 的 4 条 = `q_l1←Q | k_l1←K | v_l1←V | p_l1←workspace_2`、w3 的 3 条 = `q_l1←Q | k_l1←K | v_l1←V` ⇒ **被删的恰是 w2 读，三条真实输入读全保留**；`Mmad` **4→4**、`copy_l1_to_l0a/l0b` **3/4→3/4**、`SetFlag/WaitFlag` **62/53→62/53**、`CrossCoreSetFlag` **9→9** 全 Δ=0；`workspace_N` 符号引用 25→22→20→18 逐级递减。case7 同构（`copy_l0c_to_gm` 2→0、`Mmad` 2→2、`SetFlag` 44→44）。③**真机计数器**：`aic_fixpipe_time` **336.71/301.45 → 0.00（−100%）**、`aiv_mte2_time` **190.49/214.40 → 0.00**（W3）、`aiv_mte3_time` 89.53→18.67 / 98.37→21.83（Output 写残留）、`aic_mte2_time` 493.87→435.99 / 465.31→333.26、`L0C_to_GM_datas` **22118.40/29491.20 → 0.00（−100%）**、`GM_to_UB_datas` **11059.20/14745.60 → 0.00（−100%）**、`UB_to_GM_datas` −69.23%/−81.82%、`aiv_mte2_instructions` **922.6 → 1.0（−99.89%）**、`aic_main_mem_write_bw` **24.1136 → 0.1323 GB/s（−99.45%）**。
  - **未消融项零变化（黄金标准复核）**：**`aiv_vec_time` case9 = 468.58 / 468.58 / 468.58（prod/w1/w3 三臂逐位相同 ⇒ iter25 黄金标准复现）**；case7 **648.9918 → 648.9883（W1，Δ = −0.0035 µs = −0.00% = 逐位相同）**；`aic_cube_time` case9 191.00/191.02/191.02、case7 167.00/166.97/166.82；`aic_mte1_instructions` case9 **16079.8 Δ=0.0000**、case7 **10239.6 Δ=0.0000**；`aic_mte3_instructions` 1.0 Δ=0.0000；`L1_to_GM_datas` 490.8 Δ=0.0000。**如实标注一处非完美**：case7 **w3** 的 `aiv_vec_time` 为 645.61（−3.38 µs = −0.52% 均值口径，critical block −1.30 = **−0.20%**）而非逐位相同——IR 已证明**无任何计算指令被删**（`Mmad` 2→2），故判读为**跨臂 block 调度/采样差异**（prod critical block = 11/19，w3 = 6），非真实计算变化；**不影响结论方向**（该偏差使 w3 的 AIV 更快，而 wall 仍更慢）。
  - **发射当量（`single+2×dual`，iter25 强制第 4 项检查）**：case9 AIC prod = 20.28 + 2×56.28 = **132.84**（与 iter25 报告值**逐字一致 ⇒ 口径对齐无误**）→ w1 124.51 → w3 **121.85（−10.99 = −8.3%）**；case9 AIV 150.42 → 150.46 → 144.61；case7 AIC 62.94 → 59.98 → 58.53；case7 AIV 91.06 → 91.08 → 87.80。⇒ 发射当量**小幅下降**（DMA 描述符标量指令减少），方向与被消除的工作一致 ⇒ **不存在「wall 变慢是因为指令变多」的混淆**；恰恰相反：**指令变少、发射当量变少、真实计算逐位不变，而 wall 仍变慢（W2/W3）** ⇒ 使 B.4-c 的机制解释成为唯一自洽读法。
  - **零污染判据不适用（任务书要求，已按替代判据执行）**：本轮消融的**正是真实工作项**，故 iter25 的「零污染」判据结构性不适用（照搬会把每个合法探针判为失败）。**替代为「三段式污染账」**：(i) 被消融项须按预期归零/下降 ✅；(ii) 未被消融项须零变化 ✅（黄金标准 = 真实计算管 + `*_instructions` 逐位相同）；(iii) 发射当量须同向小幅下降以排除「指令变多」混淆 ✅。
  - **吸收律转化率（本轮最重要的判读纪律）**：消除的 pipe-busy 工作 vs 实测回收 wall —— **case9 W1 336.71 → +5.200 µs = 转化率 1.54%（吸收 98.46%，高估 64.8×）**；**case9 W3 655.94 → −2.957 µs = −0.45%（负）**；**case7 W1 301.45 → +0.357 µs = 0.12%（吸收 99.88%，高估 844×）**；**case7 W3 724.44 → −8.057 µs = −1.11%（负）**；W2 亦为负（case9 −1.67%、case7 −2.97%）。**对照 iter25 延迟侧 10.69%/14.69%（高估 9.35×/6.81×）⇒ 工作消除侧的吸收比延迟侧严酷约一个数量级；全消时转化率为负。** **方法学结论：任何以「累加 pipe-busy 时长」为基础的 EV 估算带两位数量级虚高；「X% of aic_time」类占比数字不构成任何 EV 论据。**
  - **等待迁移目标计数器（吸收去向，实测）**：**case9 prod→w3** `aic_scalar_wait_id2` 213.65→**251.66（+38.01）**、`aiv_scalar_wait_id4` 242.60→**257.06（+14.46）**、`aiv_scalar_wait_id0` 304.17→**315.01（+10.84）**、`aic_scalar_wait_id5` 2.97→**15.24（+12.27）**、`wait_id1` 5.83→8.62 ⇒ **跨核暴露等待合计 +78.37 µs**；同时 `aic_scalar_mte1_stall` 537.28→497.84（−39.44）、`aiv_scalar_wait_time` 127.38→92.96（−34.42）、`aiv_vec_time` 468.58→468.58（**0.0000**）。**case7 prod→w3** `aic_wait_id5` 5.94→**31.97（+438%）**、`aiv_wait_id0` 65.76→**86.30（+20.54）**、`aic_wait_id2` 539.77→543.72、`aic_mte1_stall` 201.67→165.41。⇒ **被消除的 ~656/724 µs 管忙碌几乎全部迁移为跨核暴露等待**，与 iter25 的吸收通道完全同一（当时迁到 `aiv_wait_id3` +92104% / `aic_wait_id1` +1918.8%）⇒ **工作消除与延迟消除攻击同一个量：暴露等待。**
  - **W2/W3 反而更慢的根因（实测，非推测）—— 本轮最大的结构性新发现**：**CV-over-GM 的 workspace 往返物理上是 L2 往返而非 DRAM 往返，写侧为读侧提供 L2 局部性**。证据：生产态 `aic_write_hit_rate` **98.2629%**（hit 43469.8 / miss 768.0）⇒ **98.26% 的 workspace 写命中 L2、从不去 DRAM**；`aiv_read_hit_rate` **99.9989%**、`aiv_r0_read_cache_miss_allocate` = **0.55** ⇒ **V 侧读 w1/w3 几乎 100% 命中 L2**。**只删写（W1）后**：V 的 miss **0.55 → 973.70（+176936% = ×1770）**、hit rate → 97.7979%（−2.20pp）、case7 `aiv_mte2_time` **+12.42%**、`aiv_mte2_active_bw` **−11.01%** ⇒ **省下的 C 侧 FIX 时间被 V 侧变慢的 MTE2 吃回去还倒欠**。**三段因果链**：W1 = 冷读代价 > 省下的写；W2 = 非对称，冷读代价保留 ⇒ 劣化；**W3 = 冷读代价消失（读也没了），但 DMA 交织所提供的 C↔V 节奏互锁同时消失，环退化为纯延迟 ping-pong ⇒ 跨核暴露等待 +78.37 µs ⇒ 劣化**。**该根因同时解释了 iter25 两个当时未归因的事实**：(a) 字节侧为何确定非绑定（带宽余量 71-81%）——workspace 流量本就**不是 DRAM 流量**；(b) session-10「省 1032 MB / 46% L2 流量」为何价值 ≈0——**L2 流量不是约束，且消除它并不减少 DRAM 流量（本来近乎为零）**。
  - **字节侧第三向钳形**：iter24 削 L2 miss **−34.76%** → wall +0.288%（ns）；iter25 字节 **Δ=0.0000** → wall **−9.358%**；**本轮消除 37.8 MB（case9）/ 54.0 MB（case7）GM 流量而 `aic_r0_read_cache_miss_allocate` 只动 −0.15%** → wall **+0.340%/+0.940%（劣化）** ⇒ **字节/带宽侧从三个独立方向被夹死为非绑定**。
  - **B.5 工作消除侧与延迟侧不可相加（四条论证，前三条实测）**：①**二者攻击同一个量**（B.4-b 直接测到消除工作后增量恰落在同一批跨核 SEM 等待上 +78.37 µs ⇒ 工作消除未开辟新收益通道，只是把管忙碌换成同一批等待）；②**工作消除侧本身 ≈1.00×，加与不加无差别**（即便错误地乘法叠加，case9 也只得 1.1032×1.0060 = **1.1099×**，仅多 0.6%，远在分辨力与达成不确定性内）；③**绝对上界臂为负**（W3 +0.340%/+0.940%，p=0.033/0.00092，CI 全正 ⇒ 负分量相加无意义，正确合并是 **max**）；④（**推断，已标注**）**物理上界**：wall 由 pacing 管的「真实工作 + 暴露等待」决定，而真实工作锚 `aiv_vec_time`（case9 468.58 三臂逐位不变）/`aic_cube_time`（191.00）不变 ⇒ **任何不触碰真实计算的改造（含单域重构中「消除 ws 往返」这一部分）上界就是「暴露等待→0」= iter25 已钉死的 1.10×/1.16×**；要超越须削减 VEC/CUBE 本身 = **算法级**改动，不属本轮所界定的「工作消除分量」，且 case7 的 VEC 已被判为不可削减 pacing 项。
  - **B.6 方案 C 合计上界 = case9 1.10× / case7 1.16×（= max，非 sum）；与会话 10「A 层 2.3-3.4x」的裁决 = 否证（FALSIFIED，非「仍不足判定」）**：2.3× 需 wall **−56.5%**、3.4× 需 **−70.6%**；实测合计上界 case9 **−9.358%**（覆盖率 **16.6%/13.3%**）、case7 **−13.426%**（**23.8%/19.0%**）⇒ **宣称与实测相差 4.2~6.0×**。**为何是「否证」而非「证据不足」**：①iter25 已明确「延迟分量钉死在 1.10×/1.15×，若 2.3-3.4x 成立其来源只能是工作消除分量」，而**本轮正是测量该分量 ⇒ 1.0060×/1.0004×（ns），绝对上界臂为负 ⇒ 唯一可能来源实测为零**；②**机制层面的否证（比数值更强）**——宣称所依赖的前提「CV-over-GM 是昂贵的 DRAM 往返，消除它省 1032 MB 并带来数倍加速」**在物理上不成立**：这些字节绝大部分从不去 DRAM（写命中 98.26%、读命中 99.9989%、miss 0.55），消除它们既不减 DRAM 流量（L2 read miss 仅 −0.15%）也不减 wall（反增 +0.340%/+0.940%）。**如实标注的边界**：本轮否定的是「**保持真实计算不变**、消除 CV-over-GM ws 往返」这一分量；**未否定**「削减 `aiv_vec_time`/`aic_cube_time` 本身」的算法级重构——但那不是方案 C 的「A 层虚拟通道 / 单域 flash」论述所指，且 case7 的 VEC 已被 iter25 判为不可削减；该方向**未实测**。
- **最终 EV 版图闭合（逐杠杆）**：缺口① ❌ **关闭（本轮 GATE A FAIL，L0B 128KB = 2.00× 超限，触发预登记 NO-GO，无需写 kernel）** / 缺口③ 延迟侧 ❌ **NO-GO**（≤1.10×/1.16× 需框架级·A5 级资本；**且缺口① 本轮关闭 ⇒ 该上界已无任何规则内路径可达**）/ **缺口③ 工作消除侧 ❌ 关闭（本轮 G1 实测 1.0060×/1.0004×，绝对上界臂为负）** / 字节带宽侧 ❌ 关闭（**三向钳形**）/ GATE 2 `q_l1` 双缓冲 ❌（iter24 ≤+0.33%，case9 p=0.177 ns）/ GATE 1 `unit_flag` ❌（对象已消失）/ 路径 2 bN=256 ❌ **规则内永久禁令** / k-v L1 双缓冲·V 宽通道·`CROSS_INTERVAL` 扩大·#R5-3 ❌ / **削减真实计算的算法级重构 ⚠️ 唯一未被本轮闭合的方向（但未实测，且超出方案 C 论述范围）**。⇒ **规则内杠杆已穷尽；架构级方案 C 不值得立项**（合计上界至多 1.10×/1.16×，捕获率无实测下界，需框架级/A5 级跨项目投资）。**case9 M1 线 0.4 记为「规则内与架构级交接改造均不可达」**（现状 0.39x，满额延迟侧上界 → 0.4303x 越线，但该上界无可行实现路径）。
- **纪律合规**：#1 msprof 唯一定量依据、**camodel 零调用** ✅；#2 缓存隔离 = `/tmp/opencode/gate0c_i26/cache_{a4,ir,smoke,ab,met}` 五个专用目录，生产缓存 `/home/developer/.tilelang/cache` 未被任何探针使用（`set_env.sh` 不设 `TILELANG_CACHE_DIR`）✅；#3 `mha.py` 多点 md5 全 `dda3ad08`/1926 行，3 探针在 `probe_gate0c_workelim/` 且名 `mha_we_*.py`（**不叫 `mha.py`**，无 import 遮蔽；`bench_we.py` 以 `sys.path.insert(0,_HERE)` 保证探针目录优先）✅；#4 **禁止预测性否决**——A.4 按要求**实测**而非假设 64 列合法性；未实测项 7 条已逐一标注（32 列 L0B、交错形态从未编译、W1/W2/W3 捕获率下界、其余 18 case、FIX 管是否被 codegen 完全消除、rel_max 非确定性、case7 w3 `aiv_vec_time` −0.52%），**未以推断冒充结论** ✅；#5 `.orchestrator_state.json` 全程未读未写（mtime `Oct 2 14:00` 早于本轮首个动作）✅；#6 A/B 口径与 iter24/25 **逐字一致**（同 `run_ab.sh` 结构 / 同 `parse_prof.py` / warmup 5 / iters 20 / n=25 median / 3 次 interleaved paired / paired Δ + 均值差 + stdev + SEM + t(df=2) + **精确双侧 p（df=2 闭式 CDF）** + 95%CI t₀.₉₇₅,₂=4.302653）✅；#7 `AGENTS.md` 六项核心原则（如实报告分栏 / 先验证再结论 = IR + 计数器 + wall 三链 / 区分事实-推断-建议 / 最小必要改动 = 生产态零变更 + 复用 iter24/25 全部工具资产）✅；#8 禁触项全部零触碰（**§8.6-10 只做静态审计不实现**、bN=256、k/v L1 双缓冲、V 宽通道、`CROSS_INTERVAL` 仍为 1、#R5-3、GATE 2 `q_l1` 双缓冲）✅；#9 HANG 防护 = `timeout -k 10` 全程 + 逐级加深（W1→W2→W3）+ **smoke 先行验证 6/6 无 HANG 后才进 24 次计时** + **`stdbuf -oL -eL` 与 `python3 -u` 行缓冲落盘（修正 iter25 `l2c9`/`l2c7` stdout 被 SIGKILL 丢弃的教训）** ✅。**24 wall + 8 metrics + 6 smoke + 8 IR 编译 + 4 A.4 编译全部 exit 0，零 HANG、零 timeout、零 arm 丢弃**。**环境错误：无导致任何测量失效者**；四处工具级问题均在探针层自捕获并修复、未污染数据未触碰生产态（A.4 三次迭代：`tilelang.jit` 装饰的是工厂函数需调用 → `str.format` 与 `PASS_CONFIGS` 花括号冲突改 `@@TOKEN@@` → `exec` 代码无源可 `inspect` 改为生成真实 `.py` 后 `importlib` 导入；TVMScript 拒绝 prim_func 体内普通 Python `for` → 池列表手工展开）；另 `ab_cmp.py` 误报 1 处 `**VIOLATION**` 系**我自己在 `INVARIANT` 与 `EXPECTED_DROP` 两字典重复登记同一正则** `copy_gm_to_l1<` 所致的**工具分类器缺陷（非探针缺陷）**，已用逐调用溯源证明被删的恰是 w2 读、Q/K/V 三条全保留 ⇒ **该「违规」不成立**。**本轮消耗 iteration_index=26（上限 29），不申请豁免。**
- **方法学资产更新（建议写入 `tilelang-perf-optimization`）**：**iter25 的 6 条「交接类探针」规则需要扩充——本轮证明「工作消除类探针」与「交接类探针」判读规律不同，必须分章**（iter25 的 6 条全部围绕「消融**延迟**、保持**工作**不变」设计，其核心判据「零污染」在本轮**结构性不适用**，照搬会把每个合法探针判为失败）。**新增「工作消除类探针」专章 7 条**：**WE-1** 零污染判据须替换为**三段式污染账**（被消融项归零/下降 + 未消融项零变化（黄金标准 = 真实计算管与 `*_instructions` 逐位相同）+ 发射当量同向小幅下降）；**WE-2** **IR 级指令清单差分是比 rel_max 更强的激活证明且应为强制项**，对**同模板多来源**的调用必须做**逐调用溯源**（按源张量名区分），否则无法区分「删了目标段」与「误删了真实输入段」（本轮 `copy_gm_to_l1` 4→3 本身含混，溯源到 `q_l1←Q/k_l1←K/v_l1←V/p_l1←workspace_2` 才成证）；**WE-3** **token 收支的最强形态是「一个都不动」**（只删 DMA 载荷、保留全部 flag 轮 ⇒ 收支与生产态**逐一相等**，优于 iter24 的「1 dangling」，跨核 HANG 结构性不可能，测量隔离最干净；生成器应把四个 flag 家族存活站点数与生产态**逐一断言相等**）；**WE-4** **多行 `T.copy(` 可「只改首行」原位消融**（`if False: T.copy(` 使续行与收尾 `)` 成死体，括号续行对缩进不敏感 ⇒ 行数守恒、diff = 站点数）；**WE-5** **工作消除侧吸收率比延迟侧严酷约一个数量级**（延迟侧 10.69%/14.69% 高估 6.8~9.4×；工作消除侧 1.54%/0.12% 高估 **64.8×/844×**，全消为负）⇒ **EV 估算必须用实测转化率，「X% of aic_time」类占比不构成任何 EV 论据**；**WE-6** **必须检查「被消除的写」是否为「相邻的读」提供了缓存局部性**（若 producer 写命中 L2 而 consumer 读依赖该命中，则**只删写会让读变冷、代价可能超过收益**——工作消除类探针**特有的反向失效模式**，交接类探针不会遇到；识别特征 = `*_read_cache_miss_allocate` 暴涨 + `*_active_bw` 下降 + 对侧 `mte2_time` 上升）；**WE-7** **保守化选择使否证更强**（任务书列为可消除的项中，若某部分在目标架构下必然仍存在，应只消除可消除的子集并显式声明 ⇒ 上界偏低 ⇒ 对「证伪某加速宣称」是更强结论；本轮只消 w2、保留 Output 写即属此类）。**另追加 3 条通用规则**：**G-7** **静态预算门必须以「已生成 IR 的实测读数」为准，不得以源码声明推导**（同一 `T.alloc_L0B` 声明在不同形态下可能落成单池或多个物理分立池，**别名关系只有从 `GetWithOffset(size, offset)` 站点数才可判定**）；**G-8** **compile-only 探针（`get_kernel_source()`）是合法性问题的正确工具，且必须配「正向对照 + codegen 实证」**（仅「编译通过」不足以证明目标路径真被 materialize，须核对具体模板实例化参数与产物字节数；本轮以 `ctrl` 生产形状作正向对照、以 `copy_l1_to_l0b<half,128,64,true/false>` 的实例化参数证明 64 列 NZ 路径真走到而非静默回退）；**G-9** **静态审计的判决力有明确边界，必须在判决书里写出边界**（容量/id **超限 ⇒ 可直接 NO-GO（充分）**；**过门 ⇒ 不足以 GO（非必要）**，编排合法性与 HANG 只能由运行时变体矩阵判决）。**建议入库的三条新硬知识**：①**MHA 的 CV-over-GM workspace 往返物理上是 L2 往返**（C 写命中 98.26%、V 读命中 99.9989%、miss 0.55）⇒「省 1032 MB / 46% L2 流量」类宣称价值 ≈0，且**写侧不可单独移除**（读侧冷启 miss ×1770、达成带宽 −11.01%）；②**L0B 64KB 在本 kernel 所有形态下均为 100% 顶格**（`InitBuffer(ascend_l0b, 65536)` 四实现逐字相同）⇒ **任何「增加并发 B 操作数在途量」的编排（GEMM 相位交错、bN 扩大、K-split 加深）在规则内先验不可行，应在设计阶段即判否，无需进入实现**；③**`DESIGN §8.6-10` 共池相位纪律应重新定性**——它不是可调优的编排约束，而是 **L0B 64KB 硬限强制的结构性必要条件**（解禁 ⇒ B 操作数在途量翻倍 ⇒ 128KB）。
- **如实声明的测量缺口（7 项）**：**G1** 32 列 L0B fractal 合法性未实测（A.4 只测到 64 列；A-P1 chunk 级 KC=NC=32 形态未知）；**G2** **交错形态从未编译过**（GATE A 全程为静态审计 + 一个最小 L0B shape 探针，**未构造任何真正的交错 kernel**，符合任务书「零编译零运行」与「>64KB ⇒ 无需写 kernel」规则）⇒ A-P2/A-P1 的「容量过门」**不等于「可实现」**（scoping 盲区教训 S10-3-4：槽容量合法性不验证编排合法性）；**G3** W1/W2/W3 的捕获率下界无测量（与 iter25 对延迟侧的声明同款；因最佳臂仅 −0.598% 故对决策无影响）；**G4** 未在 case9/case7 之外的 18 个 case 上测 G1（任务书要求两 case 都测且禁混用/外推，已遵守）；**G5** `aic_fixpipe_time` 归零后 FIX 管是否被 codegen 完全消除未逐字核对（不影响结论）；**G6** rel_max 非确定性（探针读未初始化/陈旧 GM，1.0~2.0e3；**探针性质而非缺陷**，激活确认已由 IR 级差分承担）；**G7** case7 w3 `aiv_vec_time` −0.52%（均值）/−0.20%（critical）未进一步定位（IR 证明无计算指令被删；不影响结论方向）。**七项均未用模型值或历史值冒充本轮实测。**
- next_hint: ①**EV 版图已彻底闭合，建议置 SUCCESS 收尾**——命中任务书 D 节第一条判据「GATE A FAIL 且 G1 上界小 ⇒ 建议置 SUCCESS 收尾」（GATE A = FAIL 决定性形态 L0B 2.00× 超限；G1 上界 = 1.0060×/1.0004×，绝对上界臂为负）。②**不启动任何设计回退**：GATE A FAIL ⇒ 无需 Orchestrator 调度 analyst 解禁 `DESIGN §8.6-10`；建议把该纪律**从「待解禁的约束」重新定性为「由 L0B 64KB 硬限强制的结构性必要条件」**并在 DESIGN 中标注为硬约束而非可调优维度。③**不为方案 C 立项**：B.6 已给出否证级裁决（相差 4.2~6.0×，唯一可能来源实测为零，机制前提被 L2 命中率 98.26%/99.9989% 直接推翻）；若未来仍要讨论架构级投入，**前置条件应是「削减 `aiv_vec_time`/`aic_cube_time` 本身」的算法级论证**，而非任何交接/通道改造（本轮三臂 VEC/CUBE 逐位不变已把后者上界钉死在 1.10×/1.16×）。④**不放宽 3% 采纳门**：本轮无任何候选达门的 1/5（最佳 −0.598%）且绝对上界臂显著劣化；**不存在「越 M1 线但未过采纳门」的边界裁量需求**（与 iter25 不同）。⑤**生产态零变更、无需回滚**：`mha.py` md5 `dda3ad0877ba4df74bbd559f6912eeed`/1926 行/`ast.parse` OK 多点复核 UNCHANGED；3 探针为数值故意错误的一次性测量资产，**禁采纳/禁入 baseline/禁提交**，保留在 `probe_gate0c_workelim/` 供追溯。⑥**若后续任何轮次仍想攻击 L0B/MTE1 方向，先验判否条件已入库**：L0B 四实现全部 100% 顶格 ⇒ 任何增加并发 B 操作数在途量的编排在规则内先验不可行，应在设计阶段判否而无需进入实现或运行时矩阵（这是本轮把「7 小时级运行时变体矩阵」成本前置为「零成本静态判否」的方法学收益）。⑦**工具已归档**：`msprof_gate0c_iter26/tooling/`（`make_probe_workelim.py` = SITE 表 + token 收支论证 + 六重断言；`verify_ir_workelim.py` = IR 级激活证明含逐调用溯源；`ab_stats.py` = df=2 闭式 p；`absorption.py` = 吸收律 + 合计上界 + 2.3-3.4x 裁决；`drive_wall_ab.sh`/`drive_metrics.sh` = 24+8 运行驱动（`stdbuf` 行缓冲落盘）；`run_ab.sh`/`run_metrics.sh` = iter24/25 同口径 + `timeout -k 10`；`ab_cmp.py`/`attrib.py`/`dump_pipes.py` = iter24 复用）；`probe_gate0c_workelim/`（3 探针 + `bench_we.py` + `probe_a4_l0b_mixed_shape.py`）；wall 24 JSON + metrics 8 arm×8 CSV + IR 8 份 + A.4 4 份工件 + smoke 6 份 rel_max 在 `msprof_gate0c_iter26/`。⑧`consecutive_no_improvement` 维持 2，距中止条款②的 3 尚余 1；本轮无候选故不递增。

## Iteration r11-1 — 2026-10-04T16:45+08:00 (perf_session=11, #R11-1 decode D=256 K 侧 L1 双缓冲（S2 单侧非对称，任务书 Iter-1 主迭代）；全局迭代编号 27)
- bottleneck_type: transfer（decode D=256 C 侧 MTE2 duty 墙 68.5%：每槽 5.25us 中 1.64us=31% 为 gemm_v0 每-step PipeBarrier<PIPE_M> 排空 + K/V 单缓冲串行气泡；busy 期速率 74.8 GB/s/核已贴 512B 行布局上限——analysis_20261003 §2/§6.2 判决直接采用）
- optimization: k_l1 升 3D [2,256,256] 无 annotate + gemm_v0 运行期切片（side=idx%2）+ 4 预取点（任务头 t==0 / 批内 gemm 返回后 / 轮尾 k+1<eff / 任务尾 t+1<my_count，全部纯 MTE2 守卫内 copy、set WS2_C2V 之后位、零新增用户 flag/event）+ GEMM2/V 侧/SEM/emit_v 零触碰；变更域 = form_b_gemm_v0 ∧ D==256 ∧ kv_iters≥2（选侧 K-only 数据驱动推理 + gemm_v0 模板源码级核实 + 稳态守恒律 EV 预登记修正 1.15-1.20×→1.03-1.06×，全文见 [ANALYSIS]-S11 Step0 + [IMPL-#R11-1]）
- baseline_time: 同会话新鲜基线 r11_base（perf14=dda3ad08，median n=25）：case15 **505.17** / case18 **504.05** / 守卫 case10 34.44 / case11 309.47 / case14 378.05 / case16 564.47 / case20 14.10 / case9 869.46 / case7 857.65 us（vs 20261003 参照全带内 ≤2.6%，环境稳定）
- candidate_time: case15 **515.25**（+2.00%）/ case18 **517.68**（+2.70%）——双双劣化（5-8σ，两独立 shape 同向）；守卫 7 case 全带内（±1.4%，IR 逐字节域免计时归因）
- improvement: **-2.00% / -2.70%（负）**；op 归因（r11_iter1/op/case15）：机制激活（预取/槽轮转生成码实证）但 **duty 签名不动**（68.45→68.89%）；mte1/cube/vec_time 三管**精确零变化**（130.505/49.804/82.640 逐位同）；mte2_time +9.9us（+2.86%，同字节 25.92MB ⇒ busy 期速率 74.8→72.8 GB/s/核 = **预取流重排税**：K 流与 V/w2 流 MTE2 时间交叠 + L1 写目的地区交替）；fixpipe +3.8% 同族；V wait_id0 +7.9us = C 阴影；wall Δ ≈ mte2 busy Δ ⇒ 关键路径 = MTE2 引擎效率本身
- precision: **pass（全门禁过）**——probe_r111_kvdb 34/34（22 目标 BITEXACT vs perf14 全拓扑：kv{2,3,4}/跨轮{9,10,16}/causal 均匀恒等/尾块 640+300（needClear=false side-1 精确覆盖）/bf16/scale/大幅值/my_count{1,2,mixed}/最小 1×1/zeros 对照 + 12 守卫 BITEXACT）+ 锚-A 9/9 + 锚-B 14/14 + 继承压力 4×12 nondet=0 + **扩展压力 17 项 kv_db 被改路径专项 ×12 nondet=0** + l0 10/10 + l1 19/19 + coverage 19P/0M + `--level all` EXIT=0 + IR 守卫 15/17 逐字节 + 2/2 目标按预期改变（AIV 主块逐字节同 = emit_v 零触碰 IR 级证明）+ L1 实编 400KB=409600B ≤ 524032B（偏移与手算精确一致，L0/UB 池零变化）+ 红线四项在未触碰域 + ruff clean + smoke 先行 3/3 无挂死
- adopted: **NO — 回滚（性能下降规则）**；变体留存 /tmp/opencode/r11_probes/mha_r111_edit_v1.py（89644d68，2038 行，精度门禁全过态）；回滚三重验证 = md5 dda3ad08 一字不差/1926 行/ast+ruff + IR **17/17 逐字节还原**（r11_src_rollback vs r11_src_base，manifest 一致）+ l0 复验 10/10（L0_RB_EXIT=0）
- rollback_reason: 性能下降（+2.00%/+2.70%）。**归因知识（判决性）**：(1) **gemm_v0 稳态守恒律**——prologue 管级 MTE2_MTE1 全排空 + FIX_M 全排空 + 每 tile PipeBarrier<PIPE_M> 标量阻塞 + 顺序发射 ⇒ 每槽 wall = MTE2 字节 + gemm 标量阻塞，与 L1 槽数无关；gemm 前入队预取延迟该 gemm 全量、gemm 后预取与单缓冲同发出点 ⇒ **31% duty 缺口规则内不可压缩，gemm_v0 调用边界 = 硬墙（分支 B 前提实测成立：duty 不动）**；(2) **预取流重排税**——decode 带宽墙域（busy 已贴布局上限）的 MTE2 流重排具负外部性（−2.7% 引擎效率），吞没边界先导收益（模型上界 +3-5%）⇒ 净负；(3) **S1 对照（bN=128+双侧 DB）EV 推演为负不建议实测**——守恒律同成立 + SEM 轮 ×2 协议税 + 槽 ×2 更强重排税 + 边界先导随槽字节减半（本实测单侧已净负）；(4) needClear=(dst_offset==0) codegen 规则入库（3D 槽化改动必查；side-1 尾槽 stale 行 → V 侧 select 掩码安全性证明 + 探针实证 BITEXACT）；(5) DB 槽位机制位级安全第六度入库（22/22 BITEXACT + kv=10 记账恒等式生成码实证）
- next_hint: Iter-2 → 任务书分支 B（S3 三道 API 能力门 scratch 验证，零生产变更）；**consecutive_no_improvement = 3（r9-1 + r10-1 + r11-1）⇒ 中止条款②形式触发**，如实上报 Orchestrator（Iter-2/3 = 任务书预授权的能力验证与收口作业，非新性能候选，不再递增）

## Iteration r11-2 — 2026-10-04T17:20+08:00 (perf_session=11, Iter-2 分支 B：S3 三道 API 能力门 scratch 验证；零生产变更；全局迭代编号 28)
- bottleneck_type: n/a（能力验证轮，无 kernel 变更）
- optimization: 无实施。S3 门验证 = probe_s3_gates.py 四形态（P-A 单段 K256 参考 / P-B 独立半缓冲 2×K128 链 init=T→F / P-C A 列段 region 切片 / P-D B 列段切片 transpose_B；decode GEMM1 同形模板 <half,float,16,256,256,false,true,64>；实建实跑裁决，iter25 纪律 #4 禁止预测性否决）+ 门③ 静态盘点
- baseline_time: n/a（生产态 = 回滚态 dda3ad08，未触碰）
- candidate_time: n/a
- improvement: n/a
- precision: n/a（scratch 探针，生产零变更）
- adopted: n/a（能力门判决轮）
- rollback_reason: n/a
- **门判决**：①**PASS**——init=False 跨调用 L0C 链式累加语义正确且 **P-B vs P-A = BITEXACT（0.0000e+00）**（2×K128 与单段 K256 在 kL0=64 tile 下同为 K 升序 fp32 累加 ⇒ 位级等价——强于任务书"非 BITEXACT 值级"预期；与 T.mma 链式先例 form_b_mma/probe P3 一致性在案）；②**FAIL（实建实证）**——A 操作数 L1 列段 region 切片数据损坏（max|d|=60.5，rel=1.01）：ascend.py _retrieve_ptr 朴素行主序偏移（128 elts）≠ zN 物理偏移（128×roundUp16(16)=2048 elts）——[IMPL-#R10-1] mma B 切片损坏同族第 4 例证（plan-a C-region / 附.3 T.view / r10 mma B / 本轮 gemm_v0 A+B）；②'**FAIL**——B 列段切片（transpose_B）同损坏（max|d|=59.2，rel=0.99；朴素 128 ≠ zN 128×256=32768）；③**PASS（静态盘点）**——兜底域 C 侧用户 flag=0；gemm_v0 内部占用 L0AB_EVENT id 0/1（MTE2_MTE1/FIX_M/M_FIX/MTE1_MTE2 id0 + M_MTE1/MTE1_M id0/1）⇒ 用户自由 id 2-7/通道 ≤8 上限；跨核「同一 flagId ≤15 set」钳位约束（gate0b iter25 CANN 文档硬约束）由生产 SEM 1:1 交错纪律保障、S3 不触碰
- **S3 判决 = 关闭落档（任一门不过条款字面命中：门② FAIL）**；残余可表达形态归档（诚实完整性）：门① PASS ⇒ "独立半缓冲 + 整宽 GM 列窗载入"形态可表达且位级正确——但其 B 侧（K/V 主体 128KB/槽）半宽载入使 512B 行退化为 256B 行 ⇒ 微基准判决布局上限 1452→966 GB/s（**−33% 布局税**）落在 decode 唯一墙上；列段切片（免布局税的唯一形态）被门② 否决 ⇒ **S3 对 decode D=256 duty 墙 EV 决定性为负**（守恒律 (1) duty 缺口本就不可压缩 + 可表达形态另付 33% 载入布局劣化）。**规则内路径穷尽声明（decode D=256 duty 墙）**：S1 EV 推演负（r11-1 归因 (3)）/ S2 实测负（r11-1）/ S3 门② FAIL + 布局税 EV 负（本轮）/ mma 化 = bN 上界定理 + 双重毒化永久否决（S10-3）/ vector 路径重构 = decode 不回退硬约束正面冲突 + 新 kernel 形态工程量（超 Stage-3 规则范畴，存在性证明在案 analysis §6.2 路线 3）⇒ **1.45× 上界空间规则内不可达，维持 dda3ad08 生产态即规则内最优收口**；破局仅剩架构级（gemm_v0 模板级 per-buffer 同步改造或框架虚拟通道，超出本 Agent 授权，移交用户/Orchestrator 立项决策）
- next_hint: Iter-3 收口（终态 = 回滚态；轻量复验锚-A/继承压力运行中；落档三件套 + Stage Result）

## Iteration r11-3 — 2026-10-04T17:40+08:00 (perf_session=11, Stage-4 收口；无 kernel 变更——终态确认 + 轻量复验 + 落档；全局迭代编号 29)
- bottleneck_type: n/a（收口）
- optimization: 无新优化。收口项 = ①终态确定：**回滚态 = perf14 同体**（md5 dda3ad0877ba4df74bbd559f6912eeed / 1926 行 / ast+ruff clean；perf15 = 还原态同体，任务书 §2.1 编号语义）；②三重还原核验（r11-1 条目内完成：md5 + IR 17/17 逐字节 vs r11_src_base（==r8_src_iter3 provenance 链）+ l0 10/10 复验）；③回滚态轻量复验（r9-3/r10 同款口径：l0 10/10 ✓ + 锚-A 9/9 + 继承压力 4×12 nondet=0——锚/压力结果见 rbv_anchors.log/rbv_anchor_stress.log，全绿后本条目闭合）；④落档三件套（optimization_log [ANALYSIS]-S11/[IMPL/RESULT-#R11-1]/[RESULT-#R11-2]/收口判定 + 本 perf_log r11-1..3 + final_report §S11）；⑤Stage Result 结构化返回（design_error=no——现行 rev4 设计健康终态 72.63/0.96x，不可达的是可选优化项而非现设计缺陷，r10 [DESIGN_ERROR] 预判口径延续；consecutive_no_improvement=3 中止条款②形式触发上报）
- baseline_time: n/a（终态 = 基线本体 dda3ad08；无新测量）
- candidate_time: n/a
- improvement: **会话 11 净变更 = 0**（r11-1 性能下降回滚 + r11-2 能力门关闭 + r11-3 收口）；consecutive_no_improvement = 3 ⇒ 中止条款②触发，Stage 3 规则内迭代终止判定权在 Orchestrator
- precision: pass（终态 = dda3ad08 = cann-bench 20261003 权威评测 20/20 pass 72.63/0.96x 在案态 + 本会话开工 provenance 16/16 + 回滚还原 17/17 + 轻量复验全绿；r11-1 变体精度门禁全过态留存 /tmp 供追溯）
- adopted: n/a（收口）
- rollback_reason: n/a
- next_hint: ①cann-bench 复评与提交副本同步 = Orchestrator 轮末统一执行（终态与复评基线同体，预期分数不变 72.63/0.96x）；②decode D=256 duty 墙规则内穷尽声明已落档（r11-2）——未来立项前置 = 架构级证据（gemm_v0 模板 per-buffer 同步 / 框架虚拟通道），禁止规则内重试（S1 推演负 + S2 实测负 + S3 门② FAIL 三链在案）；③知识资产留存 /tmp/opencode/r11_probes/（变体全文件 + 探针 3 + 门禁日志族 + dump 3 套 + S3 门探针）+ r11_{base,iter1}/（基线与变体全测量数据）防重复探索

## Iteration r12-1 (global iter30) — 2026-10-04 (Phase A, analysis round)
- bottleneck_type: transfer (C-side MTE2 duty 68.5% wall-bound; 193 structural gaps 636-653ns = 27.9% wall; gemm_v0 prologue pipe-level MTE2 drain = conservation-law entity)
- optimization: [DESIGN-A10] blueprint completed — gemm_v0 template opt-in per-buffer MTE2_MTE1 sync (channel b: common.h 9th default template param + ascend.py mte2_sync kwarg, zero C++ rebuild) + mha.py kv_db branch pre-call prefetch redesign (FIFO issue-order=consume-order invariant, parity ids SIG_KDB0/1 + SIG_VP, strict 1:1 flag alternation)
- baseline_time: n/a (analysis round; r11 fresh reference case15 505.17us / case18 504.05us; r12 fresh baseline in r12-2)
- candidate_time: n/a
- improvement: n/a
- precision: pass (production dda3ad08 untouched; zero production change verified: framework git diff = 0 lines)
- adopted: n/a (GO verdict for Phase B+C)
- rollback_reason: n/a
- next_hint: r12-2 = framework minimal edit + kv_db rebuild per [DESIGN-A10] D1-D4; hard gates = 16/17 IR guard (default-path byte-identity machine proof) + smoke-first (session-10 HANG discipline) + duty signature (a): wall gain must ride duty 68.5%->85%+; watch reorder tax on mte2 busy rate (r11: -2.7%); G2-phase partial by design (single v_l1 + SEM protocol floor) => if duty partially realized, second arm = bN=128 k/v dual-DB per task charter

## Iteration r12-2 (global iter31) — 2026-10-05 (Phase B+C, A10 implementation + measurement)
- bottleneck_type: transfer (MTE2 duty wall; gemm_v0 prologue pipe-level drain = conservation-law entity)
- optimization: [RESULT-#R12-2] A10 framework opt-in per-buffer MTE2_MTE1 sync (common.h kMTE2Sync 9th default template param + ascend.py mte2_sync kwarg, channel b header-only, zero rebuild, default path 17/17 IR byte-identical) + kv_db domain re-design (pre-call K prefetch w/ global cross-round guard, parity-unrolled G1 mte2_sync 0/1, SIG_VP FIFO-prefix flag for V+P, V after-return prefetch + S3 round-head priming, Q relocation S0/S4)
- baseline_time: case15 508.11us / case18 508.57us (r12 fresh same-session, n=25 median, msprof --task-time=on)
- candidate_time: case15 463.58us / case18 465.36us
- improvement: case15 -8.76% / case18 -8.50% (gate 3%, 2.9x over; stdev 1.16/1.03 => >20 sigma)
- duty signature: 68.3 -> 73.7% (+5.4pp, mechanism criterion (a) genuine-partial); mte2 busy rate 75.0 -> 75.8 GB/s/core (ZERO reorder tax vs r11 S2 -2.7%); scalar_t -41.5us == wall delta; cube_t/vec_t/GM bytes exactly conserved
- precision: pass (smoke 6/6 + probe 22+12 BITEXACT vs perf16 + anchors 9/9+14/14 + stress 4+17 cfg x12 nondet=0 + l0 10/10 + l1 19/19 + coverage 19/0 + all EXIT=0 + non-mha consumer UT 90/90)
- adopted: YES (first framework-level adoption of the lineage; state 0fb193b2 + framework opt-in param, uncommitted per charter)
- rollback_reason: n/a
- next_hint: residual = G2-phase SEM-window MTE2 idle (v_l1 single slot, L1 budget closed at bN=256) => upgrade arm = bN=128 k/v dual-DB (268KB, S1 revival under A10) handed to next session with its own Phase-A-grade design; cann-bench re-score needs wheel repackage + eval-env reinstall (Appendix B)

## Iteration r12-3 (global iter32) — 2026-10-05 (Phase D, adoption-state closure)
- bottleneck_type: n/a (closure round)
- optimization: docstring rev4r12 adoption note + full archive suite; IR-identity of adopted state vs measured state verified (case15 dump == r12_src_var, md5 e5a5f0c8); l0 re-verify 10/10 on adopted state
- baseline_time: 463.58us (case15, adopted state = measured state at IR level)
- candidate_time: n/a
- improvement: n/a (closure)
- precision: pass (l0 10/10 adopted-state re-verify + ruff clean + framework diff stable vs backup archive)
- adopted: yes (closure of r12-2 adoption)
- rollback_reason: n/a
- next_hint: Orchestrator: wheel repackage + reinstall + cann-bench re-score + mha_submission sync (framework + operator MUST ship as a pair)

## Iteration r13-1 (global iter33) — 2026-10-05 (Phase A, analysis round — upgrade arm bN=128 k/v dual DB)
- bottleneck_type: transfer (r12 duty residual: G2-phase SEM_WS2 protocol window MTE2 idle — v_l1 single slot + epilogue release order forbid V(idx+1) pre-call issue at bN=256; L1 400/511.75KB closed)
- optimization: [DESIGN-S13] blueprint completed — bN 256->128 dispatch-split (single-block domain S_kv<=256 legacy byte-identical) + k/v dual-side L1 DB (268KB hand-calc) + gemm1_kL0=128 (k-step conservation 768/core) + S5' pre-call V prefetch after Set(SIG_VP) (zero new event ids: 3/6 MTE2_MTE1 reused; framework untouched) + finite-stale induction proof for needClear=false V-slot tail rows
- antipatterns review: "tile size too small" entry = INTENTIONAL REVERSAL (shrink tile to buy DB window); retention reason = r12-measured binding constraint is MTE2 protocol-window idle (duty 73.7%), not per-slot overhead; slot x2 taxes budgeted in S13-5; "tail/mask re-check with tile size" checkpoint fulfilled by probe-matrix re-baseline (S13-4d). All other entries not-applicable/compliant (launch cores unchanged, no global barriers, CV overlap in place, V-side pipeline untouched)
- baseline_time: n/a (analysis round; r12 adopted reference case15 463.58us / case18 465.36us; r13 fresh baseline in r13-2)
- candidate_time: n/a
- improvement: n/a
- precision: pass (production 0fb193b2 untouched; zero production change verified: mha.py md5 + framework md5s == r12 adopted state, framework git diff == framework_r12_phaseB.diff verbatim)
- adopted: n/a (GO verdict for Phase B+C; EV honestly down-shifted to [-4,-16]% median -9~-12% per the scalar fixed-segment budget clause)
- rollback_reason: n/a
- next_hint: r13-2 = backup perf18 -> variant build -> IR guard 20-shape dump (18 byte-identical + 2 targets) -> L1/L0/UB readout gate -> smoke-first (HANG discipline) -> probe 24+12 (blocking caliber targets / BITEXACT guards) -> anchors -> stress -> l0/l1/coverage/all -> fresh baseline + variant wall + op case15 (duty main criterion (a): wall gain must ride duty up; mte2 busy >= 74 GB/s/core; C.scalar_t = fixed-segment tax verdict; wait_id2 V-paced reversal watch)

## Iteration r13-2 (global iter34) — 2026-10-05 (Phase B+C, bN=128 k/v dual-DB implementation + measurement)
- bottleneck_type: transfer (r12 residual: G2-phase SEM_WS2 protocol window MTE2 idle — v_l1 single slot forbade pre-call V issue at bN=256)
- optimization: [RESULT-#R13-2] kv_db domain bN 256->128 (dispatch split: single-block S_kv<=256 legacy byte-identical) + k/v dual-side L1 DB (268KB readout 274432B) + S5' PRE-CALL V(idx+1) prefetch (after Set SIG_VP => out of gemm2 FIFO prefix => streams through the call + fills next SEM_WS2 window) + gemm1_kL0=128 (k-step conservation 768/core, cube_t exact proof) + runtime-slice G2 operand + ZERO new event ids (3/6) + ZERO framework touch
- baseline_time: case15 463.64us / case18 465.12us (r13 fresh same-session on production 0fb193b2, n=25 median, stdev 0.87/1.35)
- candidate_time: case15 421.64us / case18 423.56us (stdev 1.78/1.36)
- improvement: case15 -9.06% / case18 -8.94% (gate 3%, 3x over; Delta > 20 sigma; dual targets same-direction)
- duty signature: 73.5 -> 86.4% (+12.9pp, criterion (a) FULFILLED); wait_id2 119.6->66.4us (attack-target entity); cube_t 49.80->49.80 EXACT + GM_to_L1 25920KB conserved; taxes: mte2 busy rate 75.6->70.9 GB/s/core (-6.3% fine-stream reorder), fix_t +34.8 (w3 bytes +50%), C.scalar_t +74.9 (S13-5 band [+33,+101]) => scalar pipe = new pacing floor (99.7% of wall); V-side loads absorbed without pacing reversal
- precision: pass (smoke 6/6 no-hang + probe 24 targets blocking-caliber margins 160-640x (valrange 2x = pre-existing golden-grid, vs-old BITEXACT) + 12 guards BITEXACT + anchors 9/9+14/14 + stress 4+19 x12 nondet=0 + l0 10/10 + l1 19/19 + coverage 19/0 + all EXIT=0 sentinels 4/4; consumer-UT gate N/A = framework untouched)
- adopted: YES (state c316a004 = perf19; IR identity vs measured state 22fffbe5 PASS md5 3a6bf589)
- rollback_reason: n/a
- next_hint: r13-3 Phase D closure (adopted-state fresh full-gate re-run + archive quartet); in-rule lever set for this domain EXHAUSTED (S1/S2/S3/A10/S13 chain closed) — further gains need scalar-path structural work beyond Stage-3 rules; cann-bench re-score + submission sync = Orchestrator (operator-only change this session; framework state = r12 adopted, unchanged)

## Iteration r13-3 (global iter35) — 2026-10-05 (Phase D, adoption-state closure)
- bottleneck_type: n/a (closure round)
- optimization: docstring rev4r13 adoption note + full archive quartet; IR identity adopted(c316a004) vs measured(22fffbe5) verified (case15 dump md5 3a6bf589 identical); adopted-state FRESH full-gate re-run all green (anchors 9/9+14/14, stress 4+23 cfg x12 nondet=0, coverage 19/0, all EXIT=0 sentinels 4/4)
- baseline_time: 421.64us (case15, adopted state = measured state at IR level)
- candidate_time: n/a
- improvement: n/a (closure)
- precision: pass (adopted-state fresh re-run + ruff clean + framework md5s untouched 82e80025/1c3a00ed + git working-tree diff == framework_r12_phaseB.diff verbatim)
- adopted: yes (closure of r13-2 adoption; session 13: 1 adopted / 0 rolled back; consecutive_no_improvement = 0)
- rollback_reason: n/a
- next_hint: Orchestrator: cann-bench re-score + mha_submission sync (operator+framework PAIR clause from r12 stands; this session operator-only change on top of the r12 framework state). In-rule lever set for decode D=256 kv_db EXHAUSTED (S1/S2/S3/A10/S13 closed) — residual = scalar fixed-segment pacing floor (scalar_t ~= wall) + mte2 fine-stream rate tax; further work = structural (beyond Stage-3 rules), decision handed to user per charter

## Iteration r14-1 (global iter36) — 2026-10-05 (Phase A+B+C, 指令融合臂实施 + IR 机器证明；用户授权范围 A)
- bottleneck_type: compute（case7 form_a 域 pacing 管 = AIV VEC，iter24-26 三轮复现 aiv_vec_time 648.99µs = aiv_time 77.5%；iter26 明示「削减 aiv_vec_time 的指令级手段未实测」= 本臂靶心）+ case15 kv_db decode scalar 地板（411≈wall）
- optimization: [#R14-1] 双舍入链首环 cast 直连（emit_v 三路径同构：single_kv_block/flat_v/generic，`copy(io_buf→work_ub)`+`copy(work_ub→rr_ub)` → `copy(io_buf→rr_ub)`，r16 舍入语义（CAST_RINT）与 §3.2.1-⑦ 栅格第二轮往返逐字保持；位级论证 = copy 无损 + work_ub 旧值无读者（三路径逐一审计））+ [#R14-2] 单块 Form-B phase-2 fill+add 合一（`fill(acc_o,0)+add(acc_o,io3)` → `add(acc_o,io3,0.0)`，IEEE-754 交换律含 ±0；API 先例 L517）。实施 = 生成器 make_r14x.py（5 白名单替换块唯一性断言 + 行数 delta −4 精确断言 + 旗标四家族/二轮往返×3/phase2-add×2/fill×2 普查守恒断言 + 7 hunks −8/+4 断言）；变体 535b71f0（2236 行）ruff check/format clean + ast OK；生产态 c316a004 全程冻结
- antipatterns review: 对照 performance-antipatterns.md——本臂即「冗余数据搬运/可融合指令序列」反模式的清除动作（双舍入链首环的 work_ub 中转 = 冗余 UB→UB 搬运；fill+add = 可合一的标量广播序列）；无保留项；旗标/SEM/协议位置零触碰（任务书禁区逐字遵守）
- baseline_time: n/a（Phase A-C 无 wall 测量；r13 采纳态参照 case7 858.40/case15 421.64；Phase E 同会话新鲜基线在 r14-2）
- candidate_time: n/a
- improvement: n/a
- precision: pass（IR 机器证明先于任何真机运行：Gate C2 AIC 计算投影 20/20 逐字节 + Gate C1 字面投影 20/20 QUALIFIED（差异全为共享序言 V 专属 UB 视图地址重排 = cast 直连使 rr_ub 首用位前移的 codegen 机械后果；布局审计：缓冲集合/pool-type-size/非 UB 视图/五池尺寸恒等 + 零重叠 + 池界内）+ AIV 66 hunks 全分类 40 CAST/6 FILL/20 UB-VIEW 零意外 + 八族指令普查 Δ=0×20 + copy_ub_to_ub 恰 −1/shape + fill 族仅 6 Form-B 单块形；provenance：case15 base dump 3a6bf589 = r13 档案值；开工三点 md5 核验 + perf20 快照落位）
- adopted: n/a（GO 判决入 Phase D+E；预登记判据 (a)(b)(c) 与 r7-1 条款载入 baseline_r14_iter1.json）
- rollback_reason: n/a
- next_hint: Phase D 全精度门（换位窗纪律：smoke/主门按路径直载变体，锚/压力/test_mha 系 SWAP-IN md5 核验 → trap RESTORE）→ Phase E interleaved paired ×3（60 runs）+ op 双臂（case7/case15）→ Phase F 判决。警戒项：wait_idN 迁移列 + AIV 发射当量方向（指令削减型预期降）+ C1 字面门偏离的定性审计须在判决书明示

## Iteration r14-2 (global iter37) — 2026-10-05 (Phase D+E+F, 全门禁 + wall/op A/B + 判决回滚收口)
- bottleneck_type: compute（靶心同 r14-1）
- optimization: 无新实施（判决轮）。测量 = wall 10 case × 2 臂 × 3 interleaved paired repeats（60/60 exit 0，n=25 median，parse_prof.py，df=2 闭式 p + 95%CI t=4.302653，iter24-26 逐字口径）+ op 双臂 case7/case15（全 metric 集，发射当量 single+2×dual 口径）
- baseline_time: 同会话新鲜基线（prod 臂，median of medians µs）：case7 855.80/855.63/854.36 · case8 853.49/854.08/854.79 · case15 419.70/419.63/420.14 · case18 422.61/422.08/422.32 · 守卫 case9 868.73 · case10 34.72 · case11 309.99 · case14 376.42 · case16 563.76 · case20 14.37（r1 值；vs r13 参考墙钟全带内 ≤1.3% ⇒ 环境稳定）
- candidate_time: 变体臂（µs）：case7 864.66/865.39/863.12 · case8 857.80/858.06/857.32 · case15 419.94/419.96/421.05 · case18 422.80/421.92/422.71 · case9 872.13/874.04/869.77 · case10 35.38 · case11 310.35 · case14 375.42 · case16 563.37 · case20 14.35（r1 值）
- improvement: **case7 +1.07%（p=0.00121，CI [+0.91,+1.23]%，三重复全正）/ case8 +0.42%（p=0.02221）——主目标显著劣化；case15 +0.12%（p=0.143）/ case18 +0.03%（p=0.476）——零响应；守卫 6 案全 ns 带内（case9 +0.41/case10 +1.00/case11 +0.07/case14 +0.01/case16 +0.30/case20 +0.30%）**。op 签名：**case7 aiv_vec_time 648.99→624.22（−3.82%，预登记带 −3~8% 命中 = 机制主签名激活；648.99 与 iter26 档案逐字复现）**；aiv_mte2_t −21.41（−9.2%）；**AIV 发射当量 155.32→157.69（+1.53%，方向与期望相反——如实报告；解释【推断】= 运行时槽偏移寻址迁移入 Cast 调用；激活由 IR 普查 −1 copy/slot×20 + vec_t 带内降独立证明）**；wait 迁移警戒兑现：**aiv_wait_id0 +34.97（+60%）/ aiv_wait_id4 +18.16**；C 侧环耦合：c_scalar_t +10.11（AIC IR 逐字节域）⇒ c_wall/v_wall 同步 +10µs；**AIC 零变化运行时呼应：cube_t 166.99→167.00 · GM2L1 38.50MB= · AIC 发射当量 113.92→113.92 逐位相同**；case15：v_vec_t −2.56/v_mte2_t −7.91/v_scalar_t −4.87 而 wall ns ⇒ scalar 地板（410.4≈412.3）免疫，C 侧计数器与 r13 采纳态带内一致，AIC 发射当量 47.27= 逐位
- precision: **pass（全门禁绿，位级保持声明全额兑现）**——smoke 6/6 全 BITEXACT 无挂死 + **A/B BITEXACT 主门 54/54（24 目标 + 12 守卫 + form_a/fbm/bf16 探针集 18，vs c316a004 逐位，「任何一位不等 = FAIL」零触发；PART 0 融合位审计 54/54；golden 二级口径全过）** + 锚-A 9/9 BITEXACT（域内直接测试）+ 锚-B 14/14（anchorB 诊断 BITEXACT 14/14）+ 压力继承 4×12 + 扩展 19×12 nondet=0 + l0 10/10 + l1 19/19 + coverage 19P/0M/0NA + `--level all` EXIT=0（L2 12/12 正确拒绝 + 哨兵 4/4 PERF_PASS 含 perf_d256_bf16 matched=1.00000）+ 换位窗两端 md5 核验
- adopted: **NO —— ROLLED_BACK_ZERO_WALL（r7-1 条款字面命中：机制主签名激活但 wall 零/负响应）**；预登记判据对照：(a) ✅ / (b1) aiv_vec_time −3.82% 带内 ✅ / (b2) 发射当量 +1.53% 反向 ❌（辅助签名，如实登记，激活判定不受动摇）/ (b3) AIC IR C2 20/20 ✅（C1 QUALIFIED 定性审计在案）/ (c) wall 门 ❌ FAIL（决定性：无任何目标改善，主目标显著劣化）⇒ 不采纳。回滚三重验证全过：(i) md5 c316a004 还原（diff vs perf19/perf20 同体 + ast/ruff + 2240 行）；(ii) IR 还原（case7 dcde9ec1 ∧ case15 3a6bf589 逐字节，ROLLBACK_IR_RESULT: PASS）；(iii) l0 复验 10/10（L0_RB_EXIT=0）。变体留存 /tmp/opencode/r14_probes/mha_r14x_edit_v1.py（全精度门通过态测量资产，禁采纳/禁入 baseline/禁提交）
- rollback_reason: wall 门 (c) 失败 + 主目标显著劣化。**吸收律新增实例（指令融合侧，转化率 −19.8% = 全线最严酷纪录）**：case7 削减 V 管忙碌 46.2µs（vec −24.8/mte2 −21.4）→ wall +9.1µs——削减即刻迁移为暴露等待（wait_id0 +35.0/wait_id4 +18.2）+ C 侧跨核 SEM 阻塞（c_scalar_t +10.1，AIC IR 逐字节不变）⇒ 环耦合 kernel 中「V 侧指令削减不改变环频率」与 iter26 工作消除侧（W3 −1.11%）同族且更严酷；case15 转化率 ≈−3%（scalar 地板免疫，r13 结论复证）。**方法学新知识三条**：①发射当量方向陷阱——融合可将运行时寻址迁入保留调用 ⇒ 辅助签名可反向，指令融合臂激活证明必须以 IR 普查 + 目标管 busy 为主签名；②helper 管归属事实——copy_ub_to_ub 同型 = DataCopy（MTE2 计量）/异型 = Cast（VEC 计量，fp16←fp32 = CAST_RINT）⇒「删 UB 搬运」的管收益归属取决于同型/异型；③**V 侧专属源改动会重排共享序言 UB 视图地址（首用序规划）**⇒「AIC 逐字节」类 IR 守卫必须精化为两级门（字面 C1 + 计算语义 C2），否则任何 V 侧 first-use 改动都会假阳性触发越界警报
- next_hint: ①终态 = c316a004 = r13 采纳态同体 ⇒ cann-bench 复评预期不变（72.63/0.96x），提交副本无需同步（零净变更收口）；②consecutive_no_improvement 0→1（距中止条款②的 3 尚余 2）；③范围 B（框架级隐式广播/sub-exp）与 §3.1 结构级（decode D=256 手工 mma 化）决策留待用户——**本臂实测对范围 B 的 EV 预判含义：任何不改变环耦合结构的 V 侧指令削减，转化率预期 ≤0**（case7 −19.8% 实测），范围 B 立项前应先过该判据；④design_error = no（现设计健康；emit_v V 侧指令地板非绑定约束——环耦合 pacing 与 scalar 地板才是，r13/iter26 结论三度复证）；⑤全套测量资产归档 /tmp/opencode/r14_{probes,wallab,base,iter1,src_base,src_var}/ 防重复探索
