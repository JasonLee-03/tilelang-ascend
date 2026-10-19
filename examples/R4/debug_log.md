# R4 融合采样链 —— Stage 2 调试日志

## 上板验证 V-1 ~ V-3（§10.2，实现前必跑）—— attempt 1

环境：tilelang wheel `0.1.4+d75300ed` @ `~/.local/.../tilelang`（非仓库源码）；Ascend910（npu-smi 可见）；
`ACL_OP_INIT_MODE=1`；`TILELANG_CACHE_DIR=/tmp/opencode/r4_tl_cache`；中立 cwd（`/tmp/opencode/r4_val`）运行，
`import tilelang` 实测解析到 wheel 路径。验证脚本：`/tmp/opencode/r4_val/v1_topk_shared.py`、
`v2_narrow_reduce.py`、`probe_tauk.py`、`probe_buf.py`、`probe_scalar.py`。

### V-1 —— `T.tile.topk` 作用于 `T.alloc_shared`（R-B / N5）
- 方法：把 `example_topk_selector.py` 的 5 个 `T.alloc_ub` 全改 `T.alloc_shared`，R4 Developer pass_configs（原始字符串形式）。
- 结果：**✅ PASS**。`B=64,N=4096,top_k=2048` → accuracy=1.00000（0/131072 mismatch）；`B=1024,N=1024,top_k=128` → accuracy=1.00000。
- 兜底：**未触发**。R-B 风险清除，topk 相关 buffer 全部用 `alloc_shared`。

### V-2 —— 尾块归约 `real_shape=[1,valid]`（N1 / R-D）
- 方法：仿 `narrow_reduce.py`，`M=1, buf_n=4096, valid=512`，`alloc_shared`，对拍 CPU window max/sum。
- 结果：**❌ 主方案与文档兜底均失败**。codegen 硬约束（`src/target/codegen_ascend.cc:2145`）：
  narrow-reduce 路径（`real_shape` 窄于 buffer，或列切片 BufferRegion）要求逻辑宽度 `nn*elem_bytes <= 256`
  = **64 个 fp32 元素**。R4 尾块 `valid=512`（2048 B）超限：
  `InternalError: Check failed: (nn * elem_bytes <= 256) ... got 512 elements of 4 bytes`。
  设计 §10.2 的 V-2 兜底（`reduce(z_ub[:,0:512],…)` BufferRegion）**同为 narrow 路径，同样触发该约束**。
- **兜底（已触发并验证）**：尾块改用**精确尺寸 `[1,TAIL]` 缓冲 + 全量 reduce（不传 real_shape）**，
  走 normal reduce 路径（`example_online_softmax` 已证 block_N=128 fp32=512 B>256 可用）。
  实测：V=152064、BV=4096（37 满块 + 512 尾块）的 online-softmax `m/s` 与 CPU 对拍 **✅ 正确**
  （`v2_narrow_reduce.py`，m 逐位相同、s rtol<1e-3）。满块 `[1,BV]` 全量 reduce 亦正确。
- 性质：**实现层调整**（算法/tiling/内存层级/模式/API 选型均不变），非设计错误。

### V-3 —— UB 动态下标 `run_v[k_i-1]`（k_i 来自 GM）
- 方法：`[1]` int32 buffer 存 k_i，`T.serial(1)` 内 if 守卫 + `run_v[kb[0]-1]` 动态下标读（cross_entropy:113-115 同款）。
- 结果：**✅ PASS**（`probe_tauk.py` T1）。`ks=[1,2,40,128,256,257,0,100]` → `[1000,999,961,873,745,-inf,-inf,901]` 与期望逐位相同（含越界 257/0 → -inf 守卫）。
- 兜底：**未触发**（select-sum 替代方案 T2 亦验证可用，作为备份）。

### 附加实现层发现（对设计 §3.3 伪代码的修正，均为实现层，非设计错误）
1. **`T.alloc_var` 标量语义不可靠**：`v = A[cid]` 重绑定能编译但**结果错误**（返回 0.0）；
   `T.alloc_var(init=T.infinity())` 直接被拒（"only accept int or float"）；标量 `T.exp/T.log` 于 alloc_var 编译失败。
   ⟹ **全部标量改用 `[1]` `alloc_shared` buffer + `T.tile.*` + `T.copy(region)` GM 标量读写**
   （`example_online_softmax`/`cross_entropy` 既有生产范式；probe B1/B2/B3 实测正确）。
   设计 §3.3 note 3 已预留"落到 alloc_var 或内联"的余地，此处选 `[1]` buffer。
2. **`@T.prim_func` 内嵌套 `def body()` 闭包被拒**（"Function must be decorated"）⟹ 满块/尾块 body **内联**
   （设计 §3.3 note 1 提供 `@T.macro` 备选；内联最简）。
3. **仅用于 `T.Parallel` 的 buffer 作用域无法解析**（"unresolved dynamic storage scope shared … not used in any Cube/Vector operation"）
   ⟹ 每个 `alloc_shared` buffer 必须至少被一个 `T.copy`/`T.tile.*`/`T.reduce_*` 触及（如 `T.tile.fill` 初始化）。
4. `T.max(a,b)` 二元标量 max 在 kernel body 直接对 buffer 元素调用会失败；running-max 用 `T.tile.max(dst,s0,s1)`（`[1]` buffer，B3 实测正确）。

### 结论
V-1 ✅、V-3 ✅ 无需兜底；V-2 触发兜底（精确尺寸尾块缓冲 + 全量 reduce），已验证正确。
所有兜底/修正均为**实现层**，设计的算法、tiling、内存层级、Developer 模式定性、API 选型全部保持不变，
**不构成 `[DESIGN_ERROR]`**。可进入正式实现。

### 附加实现层发现（续，probe_softmax_tile / probe_loop / probe_epi3 / probe_merge 实测）
5. **1 元素 UB→UB `T.copy` 触发 aicore 异常（507015）**：`T.copy(m_blk[1,1], m_run[1,1])`（4 B < 32 B DMA 对齐）
   运行期崩溃。online_softmax 用 `sub_block_M≥16` 行故未暴露此问题（其 `[16,1]` 拷贝=64 B 对齐）。
   ⟹ **标量 buffer 间禁用 `T.copy`**，running-max 用 `T.tile.max(m_run,m_run,m_blk)` 就地更新 +
   `T.tile.max(m_old,m_run,m_blk)` 快照旧值（probe_loop C1 实测 m/s 正确）。1 元素 UB→**GM** 拷贝（输出）正常（M1 验证）。
6. **纯 Vector kernel 内标量 `T.exp`/`T.log` 无法 codegen**（"Unresolved call Op(tir.exp/log)"）；
   `T.alloc_var` 标量重绑定编译通过但**结果错误**（返回 0）。⟹ 全部标量用 `[1,1]`/`[1]` `alloc_shared` buffer，
   全部算术用 `T.tile.*`（exp/ln/sub/mul/div/max/add/broadcast），逐元素 exp 用 `T.tile.exp` 而非 `T.Parallel`+`T.exp`。
   （`example_online_softmax` 纯 Vector 全程 `T.tile.*`，印证。）
7. **`q[l]=exp(run_v[l]-m)/s` 向量化**：`T.copy(run_v→rv2[1,K])` + `T.tile.broadcast(mk2,m_run,axis=1)` +
   `T.tile.sub/exp` + `T.tile.broadcast(sk2,s_run)` + `T.tile.div`（probe_epi3 实测 tau/valid 与 CPU 逐位一致）。
8. **`tau_u`**：`T.tile.max(muc,mu_i,1e-30)`（clamp 避免 ln0）+ `T.tile.ln` + `T.tile.add`，再 `T.serial(1)` 内
   `if mu<=0: tau_u=-inf` 标量覆盖。tie 计数 `T.reduce_sum` 直接写入 `n_gt/n_eq`（避免 UB→UB copy）。
9. **分层 top-K 归并**（block topk→gather_mask("P0101")→concat→merge topk→gather_mask）：V=4096/16384/152064
   含尾块，与 `torch.topk` 集合逐位一致（max_abs_diff=0.0）。`aligned_topk=ceil(2*256/8)*8=512`。
10. **`T.If` 必须配 `with T.Then():`**（fused_gdn_gating:200-224 同款）。

### 验证脚本清单（/tmp/opencode/r4_val/）
v1_topk_shared.py（V-1✅）、v2_narrow_reduce.py（V-2 兜底✅）、probe_tauk.py（V-3✅ T1/T2）、
probe_buf.py（B1/B2/B3 GM标量copy+tile.exp/ln/max✅）、probe_loop.py（C1 online无UB2UB copy✅）、
probe_softmax_tile.py、probe_bisect*.py（定位 1-elem copy 崩溃）、probe_epi3.py（完整 epilogue✅）、
probe_merge.py（分层归并✅）。

---

## Attempt 1 — 2026-10-06T10:55:50Z
- mode: first_impl
- classification: precision_pass
- fail_category: none
- test_level: all（L0 → 收敛通过后扩展 L1/L2/Boundary → 全量）
- coverage: L0=8 用例×5 输出=40 项 | L1=18 用例×5 输出 | L2=9 用例（4 异常 + 4 守卫 + 1 空 tensor）| Boundary=13 用例（5 输出 × -inf / 5 输出 × DBOUND / 2 × ZERO / 1 × NaN）
- boundary_warnings: none（L2 9/9 [BOUNDARY_PASS]，Boundary 13/13 [BOUNDARY_PASS]，0 WARN）
- 覆盖门禁: `coverage_check.py --proto proto.yaml` → category=Fusion，**24 PASS / 0 MISS / 0 N/A → PASS (exit 0)**（Fusion 无可豁免维度，COVERAGE_NA 为空）
- 性能哨兵（§9.4 强制）: perf_worst_k1k2 B=64/V=152064 → K1+K2 **1.2 ms**，valid 全 1，5 输出全部对拍通过（w max_abs 3.58e-07），总耗时 2.0 s / 预算 900 s；perf_max_batch B=256/V=152064 → JIT 10.7 s，K1+K2 **3.6 ms**，multinomial 5.9 ms，valid 全 1，w 非负且 256/256 行非零，总耗时 11.3 s / 预算 1200 s。均 **OK**。
- changes:
  - 新建 `r4_fused_sampling.py`（378 行，纯 kernel：K1 r4_topk_stats + K2 r4_mask_sample + host 包装 r4_fused_sampling + C3/C6/C7 前置校验 + detect_vec_core_num；无 golden/测试/__main__）
  - 新建 `r4_ref.py`（94 行，golden 逐行复刻 sglang sampler.py:846-879 Path 3 + 反推 tau/valid）
  - 新建 `test_r4_fused_sampling.py`（527 行，精度框架 + 8 L0 + 18 L1 + 9 L2 + 13 Boundary + main(--level)）
  - 备份 `history_version/r4_fused_sampling_impl_s2_attempt1.py`（改 kernel 前）
- 精度结果（全量 --level all，exit 0，"Test Passed!"）: **26 [PRECISION_PASS] / 0 [PRECISION_FAIL]**（L0 8 + L1 18），**22 [BOUNDARY_PASS] / 0 [BOUNDARY_WARN]**
  - m_out: matched_ratio=1.0000, max_abs=**0.00e+00**（逐位相同，全部用例）
  - s_out: matched_ratio=1.0000, max_abs ≤ 4.77e-07
  - tau_out: matched_ratio=1.0000, max_abs ≤ 1.91e-06
  - valid(int32): matched_ratio=1.0000, max_abs=**0**（精确；L0/L1 valid_ref 全 1）
  - w_out: matched_ratio=1.0000, max_abs ≤ 3.58e-07
  - 阈值（fp32）atol=1.53e-5 / rtol=9.77e-4 / cap=1e-2 / ratio≥0.99 → 实测余量 ≥ 4 个数量级
- L0 首轮失败信号与修复（3/8 失败，均只在 valid 输出；m/s/tau/w 首轮即全过）:
  1. `l0_prod_v_tail`(B=16,V=152064) valid ratio=0.9375、`l0_full_occupancy`(B=40,V=152064) valid ratio=0.9250，且 **valid_ref 非全 1**。
     根因：**golden §9.1 草案 bug** —— tau_p/L 用 top-k 屏蔽后的 `q` 求前缀和；当 nucleus>k（sum(top-k probs) ≤ p）时，
     被 top-k 置零的尾部满足 `excl_q ≤ p` 而被计为 kept → L 虚增到 V-1（实测 152063）→ ok_p 误判 0。
     定理 T1 要求 tau_p 为 **raw**（未屏蔽）nucleus 阈值（rank l<k 时屏蔽前缀和 == raw 前缀和），kernel 用的正是 raw top-K_MAX 前缀和 → kernel 正确。
     修复：golden 改用 `probs_sort_raw` 的严格前缀和求 L/tau_p（w_ref 的 sglang 屏蔽链保持不变，故 w_out 仍对拍通过）。
  2. `l0_k_eq_kmax`(k=256=K_MAX) valid ratio=0.0000。根因：**设计 §3.4 自相矛盾** —— D-TIE 公式的 `n_gt+n_eq < K_MAX`
     在 k_i=K_MAX 时恒为假（此时 n_gt+n_eq≡K_MAX），与同节「论证 2：n_gt+n_eq==k_i ⟹ tie_ok=1」及 §9.2「valid_ref 全 1」冲突；
     §9.1 golden 用的 `(n_gt+n_eq) <= ks_eff`（全 vocab 计数）反而给出 valid=1。
     修复：kernel tie_ok 改为 `tau>tau_k OR (sum>k_i → 0) OR (sum<K_MAX → 1) OR (n_eq<=1 → 1 else 0)`。
     即保留 §3.4 的安全意图（列表被平票占满 → 降级，全等行仍正确 valid=0），又满足连续分布无平票时 valid=1。
     已验证：全等行/全零行 → valid 全 0（Boundary PASS）；k=257/k=0/TOP_K_ALL/平坦分布 → valid 全 0（L2 PASS）。
     **残留已知缺口（已记入代码注释）**：恰有 2 个相等值横跨 rank K_MAX-1/K_MAX 边界时 kernel 无法察觉（需 z_sort[K_MAX]），
     仅退化数据 + k_i==K_MAX 同时成立才可能，Boundary/L1 连续数据不触发。
  3. `l1_edge_V_eq_KMAX`(B=2,V=256) 编译 InternalError: "Cannot find pre-allocated address for buffer: z_ub"（codegen_ascend.cc:828）。
     根因：V<BV → n_blk_full=0 → `T.serial(0)` 消除满块循环体 → z_ub/e_ub/m2d/zt_ub 成为死 buffer，内存规划未分配地址。
     V=256 属支持域（C3: 256%4096=256≥K_MAX ✓；C7: V≥K_MAX ✓）故必须支持。
     修复：把 4 个 BV 尺寸 buffer 的分配移入编译期 `if n_blk_full > 0:` 守卫（与满块循环同域）。
     已验证 V=256/512（n_blk_full=0，纯尾块路径）5 输出全部正确。
- error_summary: none（最终全量 exit 0）
- design_error_reason: none —— 三处失败根因均为「golden 草案 bug / 设计文档内部自相矛盾 / 编译期死 buffer」，
  算法、tiling、内存层级路径、API 选型、Developer 模式定性全部成立且可实现；均已在实现层修复，**不构成 [DESIGN_ERROR]**。
  （特别地：V-2 的 narrow-reduce 256 B 上限属实现层机制替换，设计的尾块正确归约意图已用「精确尺寸尾块缓冲 + 全量 reduce」达成。）
- rollback: no（保留全部修改；备份仅作为 attempt1 基线留存）
- backup_path: examples/R4/history_version/r4_fused_sampling_impl_s2_attempt1.py
- instrumentation_cleaned: n/a（本次未使用 T.printf/T.dump_tensor 插桩；定位靠最小复现脚本 + golden 分量对拍，脚本均在 /tmp/opencode/r4_val/，未进入交付目录）
- next_hint:
  1. 交付可用：L0/L1 全过 + 覆盖门禁 24/24 PASS + 哨兵达标。Stage 3（可选）可做 §6.3 优化清单
     （T.Pipelined(stage=2) 需先把 BV 降到 2048 腾双缓冲；e_ub alias z_ub；N2 的 rank-1 单 buffer 省 16 KB；K1+K2 合并单 kernel 双遍历）。
  2. 若 Stage 3 需覆盖平坦分布/大 top_k，按 §4.5 通式开 K_MAX=1024 特化（BV=4096 → 148.1 KB ≤ 184 KB 可行；BV=8192 组合不可行）。
  3. 建议 Stage 1 回写 DESIGN.md：§9.1 golden 的 tau_p/L 应基于 raw probs_sort；§3.4 D-TIE 公式与「论证 2」/§9.2「valid_ref 全 1」需统一（k_i==K_MAX 边界）。
  4. 运行约束提醒：必须用已安装 wheel、**不要 source set_env.sh**、设 ACL_OP_INIT_MODE=1 与 TILELANG_CACHE_DIR，
     且**不要调 disable_cache()/clear_cache()**（R-JIT：B 进入 JIT 键，靠持久缓存避免重编译风暴）——故 checklist #11 的
     disable_cache() 在本算子按 §11.5.6 有意省略，测试文件 main() 内已注释说明。

---

# R4 融合采样链 —— Stage 2 调试日志（V01-b Gumbel-max 增量）

## 上板验证 V-b1 ~ V-b6（DESIGN_v01b.md §10.2，实现前必跑）—— V01-b attempt 1

环境：tilelang wheel `0.1.4+d75300ed` @ `~/.local/.../tilelang`（非仓库源码，实测 `python -c "import tilelang"` 解析到 wheel）；
Ascend910（npu-smi 可见）；`ACL_OP_INIT_MODE=1`；`TILELANG_CACHE_DIR=/tmp/opencode/r4_tl_cache`；中立 cwd
`/tmp/opencode/r4_val_v01b` 运行；未 source set_env.sh；未调 disable_cache/clear_cache。
探针脚本：`/tmp/opencode/r4_val_v01b/probe_vb.py`（Probe A = LCG；Probe B = Gumbel 链 + argmax，导出 u/G/tok）。

### V-b1 —— LCG int32 环绕 bitwise（R-LCG-MATURITY / R-INT32-WRAP）
- 方法：Probe A `arith_progression(state, seed_v[0]+b*BV, 1, BV)` + 3 轮 in-place `T.tile.mul(state,state,A)`/`add(state,state,C)`（scalar A/C，alloc_shared，Developer 形态），导出 int32 流对拍 `golden_lcg`（== `reference_random_1d` 的 ctypes 逐位移植）。M∈{256,4096}×BV∈{256,4096}×seed∈{0,42,1234}。
- 结果：**✅ PASS（9/9 逐位相同，mism=0）**。同时验证 golden_lcg（向量化 int64+显式 wrap）== ctypes 参考逐位一致。
- 结论：`arith_progression` 运行时 first_value（BufferLoad+loop_var×const）、in-place int32 scalar mul/add、int32 硬件环绕 mod 2^32（**无需 bitwise_and**）全部 bitwise 正确；random_1d 的 alloc_ub 改 alloc_shared 后可用。兜底未触发。

### V-b2 —— Gumbel 中间量 u/G 对拍（R-GUMBEL-LN，量化 ln ULP）
- 方法：Probe B 导出 u（clamp 后）与 G，对拍 golden 的 host fp32 u/G。
- 结果：**u max_abs=1.19e-07（~1–2 ULP）、G max_abs=3.64e-04 / max_rel=1.04e-03**（非逐位，符合预期：host `torch.log` 软件 fp32 vs Ascend `T.tile.ln` 硬件 fp32 差 ~1 ULP，经 -ln(-ln()) 放大）。
- **关键量化**：同一 Probe B 实测 top-2 候选 `z+G` 最小 gap = **4.58e-02**，≫ G 的 ULP 误差 3.64e-04（**约 125x 余量**）⟹ argmax 对 ln ULP 鲁棒，不翻转。R-GUMBEL-LN 在 L0 固定种子 + 连续 logits 下**不触发**（与 §10.1 预测一致）。

### V-b3 —— int32→fp32 cast mode（R-UMAP）
- 方法：Probe B 对 `CAST_RINT` 与 `CAST_NONE` 各导出 u 对拍 torch `.to(float32)`。
- 结果：**两 mode 给出完全相同的 u（max_abs=1.19e-07）与相同 tok（bitwise）**。⟹ int→float 的 cast mode 无实质影响（无小数部分），~1 ULP 差异来自 fp32 仿射 `*2^-32+0.5` 的末次舍入（host vs 硬件），**非 cast mode 问题**。采用 `CAST_RINT`（与 §3.3 伪代码一致）。tok_out 对 ULP 鲁棒，无需 bitwise u。

### V-b4 —— topk(K=1) + 跨块标量归并（argmax-with-index 端到端）
- 方法：Probe B 对 `[BV]` cand（含 -inf 掩码 + 尾块 pad）跑 `T.tile.topk(topk_dst,z_ub,1,BV)`，标量读 `topk_dst[0]`（val）/`topk_dst[1]`（idx），`with T.If(topk_dst[0]>best_val[0]): with T.Then(): best_idx[0]=topk_dst[1]+b*BV`（严格 GT）跨 2 块归并，对拍 `golden_gumbel_sample`（torch.argmax 首现）。
- 结果：**✅ tok_out bitwise=True，mism=0/4**。验证：topk K=1 取 (val,idx)、标量读 dst[0]/[1]、严格 GT 归并 = 最小 index tie-break、-inf pad 不被选、best_idx 全局偏移 `+b*BV` 正确。兜底（gather_mask/compare-select）未触发。

### V-b5 —— compare scalar src1 + select scalar -inf（keep 掩码）
- 方法：Probe B `T.tile.compare(mask,z_ub,t_v[0,0],"GE")`（scalar src1 → compare_scalar）+ `T.tile.select(z_ub,mask,z_ub,-inf,"VSEL_TENSOR_SCALAR_MODE")`（scalar -inf）。
- 结果：**✅ 正确**（tok bitwise 蕴含 keep 掩码正确；V-b4 同批验证）。scalar src1 免 broadcast t2d 成立。兜底（V01-a broadcast 形态）未触发。

### V-b6 —— 端到端最小闭环（= l0b_single_block）
- 方法：`B=2,V=4096,k=40,p=1.0,μ=0,rand=arange`，K1+K2-Gumbel，tok 对拍 golden bitwise。
- 结果：**✅ tok=[1472,739] == golden，bitwise=True**（Probe B + 独立 smoke 双验证）。

### V-b 小结
V-b1 ✅ / V-b2 ✅（ULP 量化，gap≫ULP 鲁棒）/ V-b3 ✅（RINT==NONE，tok 鲁棒）/ V-b4 ✅ / V-b5 ✅ / V-b6 ✅。
**设计 §3.2/§3.3 的全部新 API（arith_progression/in-place int32 mul-add/cast/clamp/ln/compare-scalar/select-scalar/topk(K=1)/标量读/T.If-T.Then 归并）在真实 NPU 上逐一验证可用**；V01-a 的 10 项坑全部规避（无 reduce、无标量 T.log、无 1-elem UB→UB copy、无 T.serial(0)、buffer 全被触及、rank-1 topk 源）。
⟹ **实现层无阻塞**，进入正式实现。

---

## Attempt 1 (V01-b) — 2026-10-07T11:55:00Z
- mode: first_impl
- classification: **design_error**
- fail_category: design_rng_independence（设计层 RNG 方案统计缺陷）
- test_level: all（L0 收敛通过 → 场景 B 扩展 L1/L2/Boundary/分布 → 全量）
- coverage: L0=V01-a 8 + V01-b L0-A 9 + L0-B 10 | L1=V01-a 18 + V01-b 16 | 分布=1（N=100k）| L2=V01-a 9 + V01-b 10 | Boundary=V01-a 13 + V01-b 6
- boundary_warnings: none（L2 全 [BOUNDARY_PASS]，Boundary 全 [BOUNDARY_PASS]，0 WARN；含 V01-b 稀疏 -inf / 大 seed 环绕 / dbound / 退化守卫 / 稀疏 NaN 均 PASS）
- 覆盖门禁: `coverage_check.py --proto proto.yaml` → category=Fusion，**25 PASS / 0 MISS / 0 N/A → PASS (exit 0)**；**D-PARAM-FUSE_GUMBEL 由 1 MISS 转 PASS（got 31）**
- changes:
  - 扩展 `r4_ref.py`（+95 行 → 189 行）：新增 `golden_lcg`（向量化 int64+显式 wrap，== ctypes `reference_random_1d` 逐位）/ `golden_lcg_reference_ctypes` / `golden_lcg_batch` / `golden_gumbel_u_G`（V-b2 诊断）/ `golden_gumbel_sample`（§9.1 逐运算镜像 kernel）。**golden_r4 逐字不变**。
  - 扩展 `r4_fused_sampling.py`（+115 行 → 493 行）：新增常量 LCG_A/LCG_C/INV_2_32/U_MIN/U_MAX + kernel `r4_gumbel_sample`（§3.3 伪代码，alloc_shared + T.tile.* Developer 混合，零 Expert 标志）+ host wrapper 增 `rand`/`FUSE_GUMBEL` 分派（**追加在 K_MAX/BV 之后**以保 V01-a 位置参数调用不破）。**K1 r4_topk_stats + K2 r4_mask_sample 逐字不变**。
  - 扩展 `test_r4_fused_sampling.py`（+596 行 → 1123 行）：V01-b L0-A（_lcg_probe 单元 bitwise）+ L0-B（8 例 tok bitwise）+ L1（16 例，含 §6 确定性 shape + FUSE_GUMBEL tag）+ 分布一致性（χ²/KL/覆盖率/熵，N=100k 分块 B=10000 单次 JIT）+ L2（10 负向）+ Boundary（6）；COVERAGE_MANIFEST 增 D-PARAM-FUSE_GUMBEL。**V01-a 全部用例保留（回归）**。
  - 备份 `history_version/r4_fused_sampling_impl_s2_v01b_attempt1.py` / `r4_ref_impl_s2_v01b_attempt1.py` / `test_r4_fused_sampling_impl_s2_v01b_attempt1.py`（改前基线）。
- 精度结果（全量 --level all，**exit 1**）: **61 [PRECISION_PASS] / 1 [PRECISION_FAIL]**，**38 [BOUNDARY_PASS] / 0 [BOUNDARY_WARN]**
  - V01-a 回归：L0 8/8 + L1 18/18 + L2 9 + Boundary 13 **全 PASS**（m/s/tau/valid/w 与 V01-a 交付一致；**K1/K2(w_out) 未被 V01-b 破坏**）。
  - V01-b **bitwise 全过**：L0-A 9/9（LCG int32 mism=0）、L0-B 10/10（tok_out int32 **matched_ratio=1.0 / max_abs=0** 逐位相同，valid 全 1，tau 回归 max≤1.9e-6）、L1 16/16（tok 逐位，含单块/多块/尾块/质数 V=4409/B=1/V=K_MAX=256/BV=2048/K_MAX=128/多种子/生产 V=152064）。
  - **唯一失败 = 分布一致性**：`chi2=5058.68/dof=39 p=0.0000`（门槛 p>0.01）、`KL=3.42e-02`（门槛<1e-3）、`H_relerr=1.71e-01`（门槛<1%）、`coverage=1.000000`（达标，keep 掩码正确）、`valid=100000/100000`。
- error_summary: Gumbel-max 采样分布严重偏离目标 categorical（χ² 超阈 ~130x，p≈0）。**非编译/运行崩溃、非 bitwise 失配、非 LOGIT_SCALE 守卫、非 R-GUMBEL-LN ULP 翻转**。
- design_error_reason: **根因在设计层 RNG 方案（§1.2/§1.3 `state_ij = seed_i + j` + §3.4 D-RAND 的仿射 LCG），非实现层**。证据链（分级定位 + CPU 对照实验，`/tmp/opencode/r4_val_v01b/`）：
  1. **kernel 逐位忠实于设计**：L0-B tok_out 与「逐运算镜像设计公式」的 golden_gumbel_sample 逐位相同（ratio=1.0/max=0）⟹ 实现无偏差。
  2. **CPU 复现设计公式 == kernel 的坏分布**：纯 CPU 按设计公式（state=seed+j，3 轮仿射 LCG，u，G，argmax）模拟，χ²=5057.6（kernel 实测 5058.68），KL=3.42e-02 —— 与 kernel 完全一致 ⟹ 坏分布来自**设计的公式本身**，非 Ascend 硬件/实现。
  3. **对照组证明方法论正确**：真·i.i.d. Gumbel 噪声（torch.rand→-ln(-ln)）在同一 target（keep 集 softmax 重归一）下 χ²=31.3 p=0.80 ⟹ Gumbel-max 定理、target 分布、χ²/KL 方法论均正确；坏在设计的噪声不是 i.i.d. Gumbel。
  4. **数学根因**：仿射 LCG 满足 `LCG³(seed+j) = A³·(seed+j) + K (mod 2^32)`，即 `state_ij` 仅依赖 `(seed_i + j)` ⟹ `G_ij = f(seed_i + j)` 是**秩-1「行波」噪声场**（Gumbel 边缘分布正确，但跨 j **强相关、非独立**；变 seed 仅整体平移相位）。设计 §1.2「每元素独立（random_1d 范式）」的断言**不成立**——LCG 序列是 equidistributed（均匀）但**非 independent**，而定理 T3 的统计正确性要求 G_ij **i.i.d.** Gumbel。故 argmax 不服从 categorical。
  5. **实现层无法修复**：增加仿射 LCG 轮数无效（实测 8 轮 χ²=46293.7，**更差**——LCGⁿ(seed+j) 对任意 n 仍仿射于行波结构）；须改为 (seed_i, j) 的**非线性混合 / counter-based hash**。CPU 验证：splitmix 风格 avalanche hash(seed,j) → χ²=51.8 p=0.083 KL=8.2e-4（**双双达标**）；简单 xor(seed, j·PRIME) → χ²=102.4（仍不达标，混合强度不足）。
  ⟹ 属设计错误识别清单「设计选用的方案存在根本缺陷（算法/统计层）」+「根因定位指向 design 的核心选择（RNG 构造）」。**不构成实现 bug，不可在实现层修补**（改 RNG 构造 = 改 §1.3 公式 + §3.3 伪代码 + §3.4 D-RAND + §9.1 golden，且需 Stage 1 重新核实 hash 所需 T.tile 位运算 API 在 Ascend 的可用性）。
- rollback: **no**（保留忠实实现 + 分布门禁 exit 1 如实暴露缺陷；未硬扛、未伪造通过。V01-a 逐字未动且回归全过；V01-b kernel 结构 + V-b1~V-b6 验证成果可复用，Stage 1 修订 RNG 后仅需改噪声构造 + golden，其余骨架不动。回滚与否由 Orchestrator 决策；备份已在 history_version/）。
- backup_path: examples/R4/history_version/r4_fused_sampling_impl_s2_v01b_attempt1.py（+ r4_ref / test 同批备份）
- instrumentation_cleaned: n/a（未用 T.printf/T.dump_tensor；定位靠 Probe A/B 最小复现 + CPU 对照实验，脚本均在 /tmp/opencode/r4_val_v01b/，未进入交付目录；Probe B 的 u/G 导出为探针专用，交付 kernel r4_gumbel_sample 无调试导出）
- next_hint:
  1. **回 Stage 1 修订 DESIGN_v01b.md 的 RNG 方案（D-RAND / §1.2 / §1.3 / §3.3 / §9.1 golden）**：把 `state_ij = seed_i + j` + 仿射 LCG 替换为 **(seed_i, j) 的非线性 counter-based hash**（CPU 已验证 splitmix avalanche hash 达标：χ²=51.8 p=0.083 KL=8.2e-4）。保留「每行种子 rand[B] + kernel 内联展开」的 Design C 访存收益（仍 2 读），仅改逐元素噪声构造。
  2. Stage 1 须核实 hash 所需原语在 Ascend 的可用性：`T.tile.bitwise_xor`/位移/乘法混合（wheel 已知 `bitwise_and` @ :1092；xor/shift 需查证），或改用可用原语组合的 avalanche；并同步更新 golden（逐运算镜像新 hash）与 L0-A 单元探针。
  3. **V01-b 的 argmax/Gumbel 反变换/跨块归并/keep 掩码/尾块统一路径均已 NPU 验证正确（V-b1~V-b6 + L0-B/L1 bitwise 全过），修订仅需替换 LCG 噪声生成一段**，其余 kernel 骨架、UB 预算（§4.5）、host 分派、测试框架可直接复用。
  4. **V01-a（FUSE_GUMBEL=False）不受影响**：本次回归 L0 8/8 + L1 18/18 + L2/Boundary 全过，K1/K2(w_out) 逐字未动，仍可独立交付。
  5. 运行约束沿用：wheel、不 source set_env.sh、ACL_OP_INIT_MODE=1、持久 TILELANG_CACHE_DIR、不 disable/clear_cache、detect_vec_core_num 运行时探测。

---

## V01-b rev1（design_revision 后重实现）—— Stage 2 上板探针 V-b0~V-b6（attempt 1，mode=first_impl）

环境：tilelang wheel @ `~/.local/lib/python3.12/site-packages/tilelang`（实测 `import tilelang` 解析到此，非仓库源码）；
NPU 可见（`/dev/davinci10/11`，Ascend910_9362/A3/dav-2201）；`ACL_OP_INIT_MODE=1`；`TILELANG_CACHE_DIR=/tmp/opencode/r4_tl_cache`；
`cd examples/R4` 运行；未 source set_env.sh；未 disable/clear_cache。探针脚本均在 `/tmp/opencode/`（非交付目录）。

### 🔴 两个根因（分级隔离定位，20+ 最小探针）——均为实现层，非设计层

**根因 A：int32 `bitwise_and`（buffer⊕buffer）只处理 M/2 个 int32 元素（"halving"）。**
- 证据链：`r4_bitop_probe`（fill/rshift 全 256 对；`rshift & 0xFFFF` 逻辑右移 mism=128/256，first_bad=128）；
  `r4_and_probe`（buffer-and M=128→处理64、256→128、512→256、4096→2048，**精确半数**）；
  `r4_binop_scope`（buffer `add`/`sub`/`mul` 全 256/256 ✅，**唯 `and` 128/256**）。
- 机理（C++ 编译错误实证，`r4_or_probe`）：`AscendC::And`/`Or`/`Not` 在 dav_c220 **仅 int16/uint16**
  （`kernel_operator_vec_binary_impl.h:390/457` static_assert）；TileLang 对 int32 调 `And` 时把 count 当 **int16** 消耗
  ⟹ count=M 只处理 M 个 int16 = M/2 个 int32。`bitwise_or` 同样 halve；`bitwise_xor`/`not` int32 **编译失败**；
  scalar-and（`tl.ascend_bitwise_ands`）**未注册**（实证 `Operator tl.ascend_bitwise_ands is not registered`，与 design §0.0.3 一致）。
- **design §0.0.3 的缺口**：其"buffer⊕buffer int32 and 可用"基于 **compile-only 探针（只验编译不验运行）+ 误读 `per_block_cast:262`（实为 scalar `bitwise_and(dst,dst,255)`，走别的 lowering）**。运行时 int32 and 实际 halve ⟹ lowbias32 的 `x&t` 无法用单个 int32 and 实现。
- **修复（over-allocation，实现层）**：and 的 3 个操作数 buffer 声明为 int32[**2·BV**]、真实数据放 [0:BV)，则 count=2·BV → 处理 2·BV int16 = **BV int32 = [0:BV) 全覆盖**（[BV:2BV) 为无关垃圾，element-wise 不污染 [0:BV)）。
  实证 `r4_overalloc_iso`：rounds=1 plain[M] lo[0:128]ok=128（半数）→ **OVERALLOC[2M] mism=0/256 全对**。
- **逻辑右移免 and（shift-trick）**：`x>>>s = a − ((a>>(32−s))<<(32−s))`，`a=x>>s`（算术）——仅用 rshift/lshift/sub（全 int32 可用、全处理）。实证 `r4_shift15_probe`：S=15/16 的 t=x>>>s mism=0/256。

**根因 B：`HASH_M2` 十进制常量算错（design §3.3 line 438）。**
- design §3.3 写 `HASH_M2 = -2073151861  # 0x846CA68B 的 int32`——**算错**：`-2073151861 = 0x846E368B`，而 lowbias32 的 M2 = **0x846CA68B = 2221713035 → int32 = -2073254261**。
- 定位：`r4_iso_r2_auth`（权威 golden）—— t=x>>>15 ✅、xo=x^t ✅、**r2=xo*M2 ❌ mism=256 first_bad=0**；Python 反推有效乘数 K=-2073151861=0x846E368B（=我照抄 design 的错误值），而 `golden xo0*M2 = kernel r2[0]` 当我用**正确** M2 时逐位相同。
- **`r4_ref.py` 的 golden 用 hex `HASH_M2 = 0x846CA68B`（正确）+ `_i32()` 转换 ⟹ golden 正确，仅 design §3.3 十进制注释错**。修复：kernel 常量从 hex 推导 `_i32(0x846CA68B) = -2073254261`（PRIME_W=_i32(0x9E3779B1)=-1640531535 ✓、HASH_M1=_i32(0x7FEB352D)=2146121005 ✓ 与 design 一致，唯 M2 需修正）。
- 性质：**实现层常量修正**（design 算法/公式 M2=0x846CA68B 明确且 CPU 已验证；仅 §3.3 十进制转写笔误），**非 [DESIGN_ERROR]**。

### V-b 逐项结论（修复后）

- **V-b0（UB 预算）**：hash-only over-alloc 链（state/h/tmp int32[2·BV]）在 BV=4096 编译 **COMPILE_OK**（`r4_vb_probe`）。完整 kernel 的 UB 实测见下文实现后核实；over-alloc 使 3 个 hash buffer 翻倍 ⟹ BV=4096 保守 ~192KB 逼近上限，若溢出按 design §4.5 降 **BV=2048**（hash 3×16KB + z/g 2×8KB + arena 32KB ≈ 96KB）。
- **V-b1（hash int32 仿真链 bitwise）**：`r4_correct_const`（正确 M2 + over-alloc + shift-trick）—— **M∈{256,4096}×seed∈{0,42,1234} 全 mism=0/256·4096，kernel==golden_hash 且 golden_hash==golden_hash_u32ref(canonical lowbias32)** ✅。fixtures（§9.1）spot-check 全对。
- **V-b2/V-b3（u/G 中间量、cast mode）**：沿用 rev0 已验证结论（u ~1–2 ULP、G max_rel~1e-3、top-2 gap≫ULP、CAST_RINT==CAST_NONE，tok 对 ULP 鲁棒）——Gumbel 反变换段未改（只换 state 来源），本次不重复探针，由 L0-B tok bitwise 蕴含验证。
- **V-b4/V-b5/V-b6（topk K=1 归并 / compare-select / 端到端）**：沿用 rev0 已 NPU 验证（argmax/跨块归并/keep 掩码/尾块统一路径未改），由 L0-B 端到端 tok bitwise 验证。

### 探针方法学备注
- 关键教训：**compile-only 探针不足以验运行语义**（and 编译 OK 但运行 halve）；**ad-hoc golden 易错**（多个探针的临时 golden 算错，最终一律以 `r4_ref.golden_hash`（==canonical u32ref）为权威）。
- 分级隔离：原语单独（P1/P2/P3 arith_progression/mul ✅）→ 位运算（P4/P5/P6 fill/rshift/and，定位 and halve）→ 逐轮（round1 ✅ / round2 定位 M2）→ 权威 golden 逐步（t/xo ✅、r2 ❌ → M2）。
- `T.view`（→`T.Buffer` deprecated wrapper）与 `tir.decl_buffer(data=...)` int16 别名在 prim_func 内**均不解析到 UB 实际地址**（读垃圾）；`reinterpretcast` 需**同 shape**（int32[M]→int16[2M] 变形状不支持）⟹ int16-view 路线放弃，采用 over-allocation。

---

## Attempt 1 (V01-b rev1 重实现) — 2026-10-07T10:45:50Z
- mode: first_impl（design_revision_count=1，回退后计数清零，attempt_index=1/5）
- classification: **precision_pass**
- fail_category: none
- test_level: all（L0 收敛通过 → 复核 L1/L2/Boundary/分布 → 全量 --level all）
- coverage: L0 = V01-a 8 + V01-b L0-A 9（3 seed×{256/256,4096/4096,4096/256}）+ L0-B 11（8 例，seed_var 展开 3）| L1 = V01-a 18 + V01-b 16 | 分布 1（N=100k, k=40）| L2 = V01-a 9 + V01-b 10 | Boundary = V01-a 13 + V01-b 6
- boundary_warnings: **none**（38 [BOUNDARY_PASS] / 0 [BOUNDARY_WARN]；L2 非法输入全部正确拒绝，Boundary 稀疏 -inf/大 seed 环绕/dbound/退化守卫/稀疏 NaN 全 PASS）
- 覆盖门禁: `coverage_check.py examples/R4/test_r4_fused_sampling.py --proto examples/R4/proto.yaml` → category=Fusion，**25 PASS / 0 MISS / 0 N/A → PASS (exit 0)**；**D-PARAM-FUSE_GUMBEL got 31 PASS**
- changes:
  - `r4_fused_sampling.py`（553 行）：① 常量块 LCG_A/LCG_C → **hash 常量（从 hex 经 `ctypes.c_int32` 推导）** `PRIME_W=_i32(0x9E3779B1)=-1640531535`、`HASH_M1=_i32(0x7FEB352D)=2146121005`、`HASH_M2=_i32(0x846CA68B)=-2073254261`（**修正 design §3.3 的十进制笔误 -2073151861=0x846E368B**）+ `import ctypes`；② `r4_gumbel_sample` kernel RNG 段重写：LCG → lowbias32 avalanche（**over-alloc int32[2·BV] 的 state_ub/h_ub/tmp_ub** + shift-trick 逻辑右移 + over-alloc int32 And 仿真 XOR + seedP_v 标量 mul + 每块 fill(state,0)+arith_progression(count 2·BV)）；Gumbel 反变换/compare/select/topk(K=1)/跨块归并/尾块统一路径/host 分派**结构不变**（仅 state 来源变）。**K1 r4_topk_stats + K2 r4_mask_sample BYTE-IDENTICAL**（diff 核验 12203/2224 chars 全等）。
  - `test_r4_fused_sampling.py`（1160 行）：import `LCG_A/LCG_C`→`PRIME_W/HASH_M1/HASH_M2`、`golden_lcg/golden_lcg_reference_ctypes`→`golden_hash/golden_hash_u32ref`；L0-A 探针 `_lcg_probe`→`_hash_probe`（精确镜像 kernel 的 over-alloc lowbias32 链）+ `l0a_lcg_unit`→`l0a_hash_unit`（对拍 golden_hash，交叉校验 u32ref）；LCG 注释→hash。**V01-a 全部用例 + V01-b L0-B/L1/L2/Boundary/分布用例结构保留**（分布测试 k=40/N=100k 符合 §9.5，无需改）。
  - `r4_ref.py`：**未改动**（Stage 1 已更新 golden_hash/_u32ref/_batch + golden_gumbel_sample 用 hash；与备份 byte-identical；golden_r4 V01-a 完好）。
  - 备份：`history_version/{r4_fused_sampling,r4_ref,test_r4_fused_sampling}_impl_s2_v01b_rev1_attempt1.py`（改前基线，byte-identical 核验）。
- 精度结果（全量 --level all，**exit 0**）: **62 [PRECISION_PASS] / 0 [PRECISION_FAIL]**，**38 [BOUNDARY_PASS] / 0 [BOUNDARY_WARN]**
  - V01-a 回归：L0 8/8 + L1 18/18 + L2 9 + Boundary 13 **全 PASS**（m/s/tau/valid/w_out ratio=1.0，max≤2.4e-7；**K1/K2(w_out) 未被破坏**）。
  - V01-b **bitwise 全过**：L0-A 9/9（hash int32 mism=0，golden==u32ref canonical lowbias32）、L0-B 11/11（tok_out int32 **matched_ratio=1.0/max_abs=0** 逐位，valid 全 1，tau 回归 max≤1.9e-6）、L1 16/16（tok 逐位，含单块/多块/尾块/质数 V=4409/B=1/V=K_MAX=256/BV=2048/K_MAX=128/多种子/生产 V=152064/valrange S·L·ASYM）。
  - **分布一致性（rev0 唯一失败项，本次修复）**：`chi2=22.98/dof=39 p=0.9807`（门槛>0.01，**优于 Stage 1 CPU 预验证 0.935**）、`KL=7.82e-04`（<1e-3）、`coverage=1.000000`（>1-1e-4）、`H_relerr=2.02e-03`（<1%）、`valid=100000/100000` ⟹ **全门禁达标**。RNG 独立性缺陷（rev0 仿射 LCG χ²=5058/p≈0）已由非线性 avalanche hash 修复。
- V-b0~V-b6 结论：V-b0 UB（BV=4096 over-alloc 编译 COMPILE_OK + 运行，memory_planning 别名 h_ub/tmp_ub 与 g_ub ⟹ ≤184KB，**无需降 BV=2048**）；V-b1 hash int32 仿真链 bitwise（M∈{256,4096}×seed∈{0,42,1234} 全 mism=0 == golden_hash == canonical u32ref）；V-b2/b3/b4/b5/b6 沿用 rev0 已验证（Gumbel/cast/topk/compare-select/端到端未改）+ 本次 smoke（B=2/V=4096、B=8/V=16384、B=4/V=152064 tok bitwise）复证。
- error_summary: none（L0 一次收敛通过；全量 exit 0）
- design_error_reason: none（**非 [DESIGN_ERROR]**）。两个根因均为**实现层**：① int32 And halving（AscendC And/Or int16-native）→ over-allocation 修复；② design §3.3 的 HASH_M2 十进制转写笔误（算法 M2=0x846CA68B 明确且 CPU 已验证，golden 用 hex 正确）→ kernel 用 hex 推导的正确 int32。design 的算法/公式/API 选型/tiling/同步均正确可实现，无设计层根本缺陷。
- rollback: **no**（精度通过，保留修改）
- backup_path: examples/R4/history_version/r4_fused_sampling_impl_s2_v01b_rev1_attempt1.py（+ r4_ref/test 同批）
- instrumentation_cleaned: **yes**（20+ 分级隔离探针脚本均在 `/tmp/opencode/r4_*.py`，**未进入交付目录**；交付 kernel `r4_gumbel_sample` 与 `r4_ref.py` 经 grep 核验无 T.printf/T.dump_tensor/print/pdb；kernel 内 `T.tile.fill(state_ub,0)` 是 over-alloc 链的必需初始化，非调试插桩）
- next_hint:
  1. **V01-b rev1 精度通过，可交付 / 进 Stage 3 性能调优（可选）**。Stage 3 关注点：over-alloc 使 hash buffer 翻倍（BV=4096 时 state/h/tmp 各 int32[8192]=32KB）+ 每块 fill(state_ub,0)（2·BV 写）+ arith_progression(count 2·BV) 多算了一半（[BV:2BV) 未读）——可优化：若能用 count=BV 且不 fill（variant3 实测 round1 正确，但需复验全链），或用 memory_planning 更激进别名，可省 hash 相位开销；topk(K=1) 的 64KB arena 是 UB 主占用。
  2. **建议 Stage 1 修订 DESIGN_v01b.md §3.3 的 HASH_M2 十进制注释**（-2073151861 → -2073254261，即 0x846CA68B 的正确 int32）+ §0.0.3 补充「int32 buffer And/Or 运行时 halving（int16-native），需 over-allocation」的运行级发现（compile-only 探针不足以捕获）。非阻塞（golden 用 hex 已正确，kernel 已修正）。
  3. 分布门禁：当前 k=40/N=100k 达标（p=0.98）；若未来增大 keep 集（k≥256）测试，按 §9.5 用 N≥400k 或 KL-over-observed 避免空-bin 有限样本假失败。
  4. 运行约束沿用：wheel、不 source set_env.sh、ACL_OP_INIT_MODE=1、持久 TILELANG_CACHE_DIR、不 disable/clear_cache、detect_vec_core_num 运行时探测（实测 40）。

---

## Attempt 1 (V01-b rev2) — 2026-10-08T20:58:00Z

- mode: **precision_fix**（design_revision_count=1，cycle=V01-b rev2，attempt_index=1/5）
- classification: **precision_pass**
- fail_category: none
- test_level: **all**（L0 收敛 → L1/L2/Boundary/分布 → 新增 stability 层 → 全量 `--level all`；另跑可选 `--level stress`）
- coverage: L0 = V01-a 8 + V01-b L0-A 9 + L0-B 10 | L1 = V01-a 18 + V01-b 16 | 分布 1（N=100k, k=40）| **stability 8 配置×20 reps + 4 组尾块对抗扫描（1342 个尾块下标）** | L2 = V01-a 9 + V01-b 10 | Boundary = V01-a 13 + V01-b 6
- boundary_warnings: **none**（38 [BOUNDARY_PASS] / 0 [BOUNDARY_WARN]）

### 🔴 是否复现了原始症状：**是，100% 确定性复现**（不需要等 bursty 窗口）

复现手段不是靠加大 N 碰运气，而是**主动构造 UB 投毒**：UB 是片上 SRAM，**跨 kernel launch 不清零**，
所以先 launch 一个把整块 UB（[0,163840) B）填成已知字面量的 "poisoner" kernel，再 launch K1。
任何"读在写之前"的位置都会确定性地把毒值读出来。

- 脚本 `/tmp/opencode/r4_fix/poison_ub.py`（毒值 1e30），case-5 原始输入（harness 自己的
  `DataGenerator` + `sha256(case_id_str)` seeding，与评测逐字节同源）。
- **修复前**：`tau_out` **7/7 行全部 = 1.0000000150474662e+30（毒值原样）**，`tok_out` **7/7 全 0**，
  `valid` **仍全 1**，`m_out`/`s_out` **完全正确** —— 与缺陷档案描述的症状（垃圾大有限浮点、
  `tau ≤ row max` 不变式被破坏、守卫不触发、tok 退化为 0、m/s 始终正确）**逐项吻合**。
  12 次投毒 launch 中 3 次触发（后续稳定触发）；k1_diag 版 4/4 触发。
- **修复后**：同样 12 次投毒 launch **全部干净**，`tau` 逐位等于 golden。

### 🎯 根因（代码级证据，两个独立缺陷，同一处尾块 top-k 源）

定位链条（每步都有实证，非推测）：

1. **投毒证明是 stale UB 读**（不是数值误差、不是缓冲轮转）：`tau_out` 逐位等于毒值 1e30。
2. **分级隔离到 `run_v`**：`/tmp/opencode/r4_fix/k1_diag2.py`（关闭 memory_planning 以排除别名干扰）
   把 K1 全部 epilogue 标量 + 整条 `run_v[0:256]` 引出。结果：
   `run_v[0] = 1e30（毒值）`，`run_v[1..255] = host_topk[0..254]` —— **整条归并列表右移一位**
   （`shift1_matches = 255/255`，7 行全部一致）；`d_tau_p = 1e30`、`d_prefix = inf`、`d_tau_k = 9.87028`
   （= 第 21 大，golden 要第 22 大 9.86933）、`d_tau_u = -inf`（正常）。
   ⟹ 毒从 `run_v[0]` 进入 → `q2[0]=exp(1e30-m)/s=+inf` → `prefix` 立刻 inf → `tau_p = run_v[0] = 毒值`
   → `tau = max(tau_k, tau_p, tau_u) = 毒值`；且 `tau(毒) > tau_k(真值)` ⟹ `tieok=1` ⟹ **`valid` 仍为 1**；
   K2 侧 `mask=(z≥tau)` 全假 ⟹ 候选全 -inf ⟹ **argmax=0**。整条传播链与观测完全一致。
   同时解释了 `m_out/s_out` 为什么始终正确：它们来自 `reduce_max/reduce_sum(zt/et)`，**不经过 top-k 归并链**。
3. **最小复现 + 变量分离**：`/tmp/opencode/r4_fix/probe_merge.py` / `probe_sentinel.py` 把 K1 的分层
   top-k 归并链单独抽出来，四组合对照（`memory_planning` × 尾块源构造方式）：

   | memplan | 尾块源 | clean | poisoned |
   |---|---|---|---|
   | 1 | `zt[1,T]`→`ztt[T]` UB-UB copy（**现状**） | **1789/1792 错，非确定** | 错 |
   | 1 | GM 直载 `ztt[T]` | 0 错，确定 | 0 错 |
   | 0 | `zt[1,T]`→`ztt[T]` UB-UB copy | 0 错 | **2/4 次错** |
   | 0 | GM 直载 | 0 错，确定 | 0 错 |

   ⟹ 触发条件 = **尾块 top-k 源用非 32B 整数倍长度的 UB→UB copy 构造**（`TAIL=313` → 1252 B）。
   孤立原语对照（`probe_gather.py`）证明 `gather_mask` 与 `topk` **本身无罪**：
   N∈{32,64,128,256,512} 的 `GatherMask(P0101)` 在投毒下 0 错；`TopK+GatherMask` 在
   **M=313/K=256（正是尾块形状）** 投毒下也 0 错。
4. **机制（sentinel 实证，`probe_fix.py`）**：`Sort32` 每次消费 `ceil(count/32)*32` 个元素，
   `tl::ascend::Sort()`（wheel 内 `src/tl_templates/ascend/common.h:1663-1706`）对余数用
   **带掩码的 `AscendC::Duplicate(src[alignedActual], -inf, masks={0xFE,0}, 1, 1, 0)`** 自行补 -inf。
   **该掩码补位在 dav_c220 上不生效**：S4 变体（对齐 count 但不自己补位）的 pad dump 直接读出
   `ztt[313..319] = 1e+30`（毒值原样）；S3 变体（自己 `fill(-777)`）的 pad dump 读出 `-777 ×8`。
   ⟹ `Sort32` 把 7 个 stale UB 元素当真实候选排序，巨大 stale 浮点直接抢到 rank 0。

   **缺陷 A（档案报告的间歇性垃圾 tau）** = 上述 stale UB 被排入 top-k。
   bursty/周期 5/bit-identical 全部由此解释：UB 残留内容取决于之前跑过什么
   （K2g 的 lowbias32 哈希链把 int32[2·BV] 随机位模式刷满 UB [16576,114880)，
   随机 int32 按 fp32 重解释正好落在 **1e19~1e38** 量级 = 档案实测的垃圾值量级）。
5. **缺陷 B（顺带查出的既存确定性 bug，非本轮引入）**：`AscendC::DataCopy` 对**非 32B 整数倍**的
   count **向下截断到 8 个 fp32 的整数倍**，即 `T.copy(zt[0,:TAIL], ztt)` 只搬 `(TAIL//8)*8` 个元素，
   **静默丢掉尾部 `TAIL%8` 个真实元素**。对抗性实证（`probe_trunc.py`，把哨兵最大值逐个放到每个尾块下标）：

   | shape | TAIL | 被丢下标 | 数量 = TAIL%8 |
   |---|---|---|---|
   | V=4409 | 313 | **仅 j=312** | 1 ✓ |
   | V=261 | 261 | **j=256..260** | 5 ✓ |

   丢元素 ⟹ top-k 候选集缺项 ⟹ `run_v` 秩位整体偏移 ⟹ `tau` 差一个秩（V=261/p=1.0 时
   POST-A 变体 `tau = z_sort[k]` 而 golden `= z_sort[k-1]`，max|diff|=0.426）。
   这是**确定性**的，所以单次执行的 rev1 套件永远看不见它，投毒也看不见它（要靠对抗哨兵）。
   注：`copy_gm_to_ub` 带 `realCount`+`pad_value`，GM→UB 方向**不**截断（`m_out/s_out` 一直正确即证）。

- **为什么只有 case 5 失败、case 2（B=1）不失败**：全 28 个用例（两任务各 14）中
  **只有 case 5 的 `TAIL = V % BV = 313` 满足 `TAIL % 32 != 0`**（其余 TAIL ∈ {0,256,512,1024,2048}，
  `padCount == 0` 且 `TAIL%8 == 0`，两条缺陷路径都不进）。case 2 是 `V=4096 = BV` ⟹ **`TAIL=0`，
  尾块分支根本不执行** —— 与"奇数 B"无关。投毒实证亦印证：`stab_B1_V4096_*` **修复前即 PASS**，
  `stab_case5_*` / `stab_B3_V4353` / `stab_B7_V4409_bv2048` 修复前 FAIL。
  ⟹ **档案里"奇数 B 越界行槽"的怀疑面被否证**：`row < B` 守卫覆盖了全部副作用面，
  真凶是尾块 top-k 源的 32B 对齐/截断，与 B 的奇偶无关（B=2 偶数、TAIL=313 同样可复现缺陷 B）。
- **"周期 5"的来源**：不是缓冲池深度，而是 **UB 残留内容的循环**——同一进程内重复 launch 时
  UB 里 leftover 的是前几轮 K1/K2g 自己的输出（K2g 哈希链的随机 int32），launch 序列周期性地
  把同一批残留值送回 `ztt[313..319]`，所以垃圾值 bit-identical 且成周期。

### ✅ 修复（最小必要改动，只动 K1 `r4_topk_stats` 的尾块 top-k 源构造）

`r4_fused_sampling.py`（sha256 `2b3fcd3a…` → **`bbe47fa4…`**）：
1. L18-26 模块头注释：补记尾块 top-k 源的 32 元素对齐要求与两个机制（Sort 掩码补位失效 / DataCopy 截断）。
2. L97-104 `r4_topk_stats` 外层：新增编译期常量 `TAIL_PAD = ((TAIL + 31) // 32) * 32`（+7 行注释说明根因）。
3. L210-238 尾块分支：
   - `ztt = T.alloc_shared([TAIL], "float32")` → 新增 `ztp = T.alloc_shared([1, TAIL_PAD], "float32")`
     且 `ztt = T.alloc_shared([TAIL_PAD], "float32")`；
   - 删除 `T.copy(zt[0, :TAIL], ztt)`（会截断）→ 改为
     `T.copy(logits[row, n_blk_full * BV], ztp, pad_value=-T.infinity("float32"))` +
     `T.copy(ztp[0, :TAIL_PAD], ztt)`；
   - `T.tile.topk(l_dst, ztt, K_MAX, TAIL)` → `T.tile.topk(l_dst, ztt, K_MAX, TAIL_PAD)`。
   - **`zt/et/mt` 的 softmax/reduce 路径逐字未动**（`m_out/s_out` 一直是正确的，不碰已验证语义）。
   - 复用 K2g 已上板验证的 `pad_value` 尾块模式（AGENTS.md「优先复用」）。
- 生成码核验：`copy_gm_to_ub<float, 320, 1>(ztp, logits+4096, 30863, 1, **313**, **-CUDART_INF_F**)`
  （blockSize 对齐、realCount 按行钳位、pad = -inf）；`copy_ub_to_ub<float,float,**320**>`（1280 B = 32B 整数倍，
  **不可能截断**）；`TopK<float>(l_dst, ztt, tmp, 256, 10, **320**)` ⟹ `alignedCount = 10*32 = 320 = actualCount`
  ⟹ **`padCount == 0`，Sort() 的掩码补位分支永不进入**。
- UB 预算：151364 / 196352 B（修复前 150084，+1280 B = `ztp`）；≤184KB 可用 ⟹ **无需降 BV**。
- pad 内容直读核验（`validate_variant.py` 内建 pad 探针）：TAIL=313→`pad(7) = -inf OK`、
  257→`pad(31) = -inf OK`、511→`pad(1) = -inf OK`、261→`pad(27) = -inf OK`、308→`pad(12) = -inf OK`。
- **未改** `r4_ref.py`（golden，sha256 `879afbc8…` 与备份一致，mtime 未变）；**未改** `proto.yaml`（冻结规格）。
- 未放宽任何阈值、未删用例、未把 `valid` 强置 0。

### 📌 新增「重复执行门槛」（本次逃逸的根因）

`test_r4_fused_sampling.py`（sha256 `e1e4a227…` → **`02a44b79…`**，1160 → 1400 行）：
- 新增 **stability 层**（`test_r4_stability`，**阻塞**，已并入 `--level all`）：
  - `STAB_CASES` 8 个高风险配置（奇数 B、`TAIL%32!=0`、1 元素标量 UB 归并 epilogue、K1→K2 多 kernel 串联；
    两条路径都覆盖），**每配置重复 `STAB_REPS=20` 次**，断言**逐次 bit-identical** + 逐次 golden 达标
    + 逐次不变式；任一次不一致即 `[PRECISION_FAIL]`。
    必含 **B=7/V=4409（原缺陷用例，gumbel + mask 两条路径）** 与 **B=1/V=4096**；
    另含 B=3/V=4353（TAIL=257）、B=9/V=4607（TAIL=511）、B=7/V=4409/BV=2048、B=5/V=10240（对齐对照）。
  - `_tau_invariant()`：便宜的免 golden 不变式 —— **I1 `tau ≤ row_max(logits)` 或 `tau == -inf`**、
    I2 非 NaN、I3 `valid==1` 时 `tok` 对应列满足 `z ≥ tau`。1e19 级垃圾值立即被 I1 抓住。
  - `_run_tail_adversary()`：把哨兵最大值（1e4，远超 LOGIT_SCALE=8 值域）**逐个**放到每个尾块下标，
    要求 `tau(k=1)` 必须反映它 —— **确定性**检出尾块元素被丢（缺陷 B），不靠概率。4 组共 1342 个下标。
- 新增可选 **stress 层**（`--level stress`，非阻塞于 `all`，默认 200 reps）：内建 UB poisoner kernel
  （`_make_ub_poisoner`，把 [0,163840) B UB 填成 1e30）+ 每次 launch 前投毒，把间歇性 stale-UB 变成确定性可检。
- `main()`：`--level` 增加 `stability`/`stress`，新增 `--reps`/`--stress-reps`；
  `--level all` 强制 `reps` 下限 = `STAB_REPS`(20)；模块 docstring 补层级说明。
- 归类理由：单独成 **stability 层**而非塞进 L1 —— 它检验的是"同一输入的重复执行一致性/内存生命周期"，
  与 L1 的"功能/形状/参数覆盖"是正交维度；且它必须阻塞（是本次逃逸的直接缺口），
  而大 N 投毒版必须可选（耗时）。

### 🔬 新门槛的检出能力已实证（对**修复前**内核跑同一份测试文件）

`/tmp/opencode/r4_fix/prefix_check/`（备份内核 + 新测试文件）：
- `--level stability` → **exit 1**：
  - `stab_B7_V4409_bv2048_gumbel` **[PRECISION_FAIL]** `selfdiff=2 goldenfail=2 invariant=1`，
    first_bad = `it0:tau_out not bit-identical to launch 0` ⟹ **重复执行门槛抓到了间歇性 stale-UB**；
  - 4 组对抗扫描全部 **[PRECISION_FAIL]**，被丢下标恰为 `[TAIL-TAIL%8, TAIL)`：
    `TAIL=313→{312}`、`257→{256}`、`511→{504..510}`、`261→{256..260}` ⟹ **缺陷 B 被确定性检出**；
  - `stab_case5_B7_V4409_*`（BV=4096）与 `stab_B1_V4096_*` 修复前 PASS ⟹ 印证 bursty 静默期与 TAIL=0 免疫。
- `--level stress --stress-reps 8` → **exit 1**：`stab_case5_B7_V4409_gumbel` selfdiff=14 goldenfail=16 invariant=8、
  `stab_case5_B7_V4409_mask` goldenfail=16 invariant=8、`stab_B3_V4353_gumbel` goldenfail=16 invariant=8，
  first_bad = `it0:tau_out golden ratio=0.0000 **max=1.000e+30**` ⟹ **投毒门槛复现出档案原始症状**；
  `stab_B1_V4096_*` PASS（TAIL=0 免疫，与档案"case 2 不失败"一致）。
- 修复后同一门槛：`--level stability` **全 12 项 PASS / exit 0**；`--level stress --stress-reps 60`
  （8 配置 × 60 投毒 launch = **480 次**）**全 PASS / exit 0**。

### 精度结果（全量 `--level all`，**exit code = 0**，`Test Passed!`）

- **74 [PRECISION_PASS] / 0 [PRECISION_FAIL]**；**38 [BOUNDARY_PASS] / 0 [BOUNDARY_WARN]**
  （rev1 为 62 PASS，+12 = 新增 stability 层 8 配置 + 4 对抗扫描；其余逐层计数与 rev1 一致，无回归）
- 分层：L0 8/8 · L0-A 9/9 · L0-B 10/10（`tok[ratio=1.0000,max=0]` 逐位）· L1 18/18 · L1-B 16/16 ·
  DIST 1/1 · **STABILITY 12/12** · L2 9 · L2-B 10 · Boundary 19
- **V01-a 回归（K1 共用，必须全过）**：L0 8/8 + L1 18/18 + L2 9 + Boundary 13 全 PASS；
  `m/s/tau/valid/w_out` ratio=1.0，`tau` **逐位相同（MERE=0）**，`s_out` max ≤ 2.4e-7 级别（与 rev1 同量级）。
- **V01-b**：L0-A 9/9（hash int32 bitwise）、L0-B 10/10（`tok_out` int32 bitwise）、L1 16/16、L2 10、Boundary 6。
- **分布门禁**：`chi2=22.98/dof=39 p=0.9807`(>0.01) · `KL=7.82e-04`(<1e-3) · `coverage=1.000000`(>1-1e-4) ·
  `H_relerr=2.02e-03`(<1%) · `valid=100000/100000` ⟹ **与 rev1 实测完全一致，无回归**。
- **覆盖门禁**：`coverage_check.py examples/R4/test_r4_fused_sampling.py --proto examples/R4/proto.yaml`
  → **25 PASS / 0 MISS / 0 N/A，exit 0**（与 rev1 相同；`proto.yaml` 未改动）。

### 交叉验证（cann-bench 任务侧 golden，harness 自己的输入/metric）

- `/tmp/opencode/r4_fix/crossval_cases.py`：**全 28 个用例（r4_fused_sampling 14 + r4_mask_sample 14）× 每例 20 次重复**。
  输入用 harness `DataGenerator` + `sha256(case_id_str)[:8] BE % 2**31 + eval_seed=0`；
  判定用 harness `RelativeErrorChecker` + `get_threshold("float32")`（float MERE/MARE，int32 阈值 0 = 逐位）。
  结果：**passed=28 failed=0**，每例 `chk=0 self=0 inv=0`（20 次 launch 互相 bit-identical），
  `tau_out` **worst MERE=0.00e+00（逐位）**，`golden_valid_all1=True` 全部成立。
  用例矩阵确认：**只有 case 5 的 `TAIL%32 != 0`（313%32=25）**，与根因预测一致。
- **harness 端到端**（`run_evaluation.sh`，含 stage-3 性能阶段——即原先抓到缺陷的那一阶段）：
  `R4FusedSampling × 2 次` + `R4MaskSample × 2 次` = **4 次运行全部 14/14 通过、rc=0**；
  每次运行的 **stage-2 精度检查与 stage-3 精度复检双双通过**（原先正是 stage2 过 / stage3 挂）；
  `r4_fused_sampling_5` 与 `r4_mask_sample_5` 的 `tau_out MERE=0.000000`、`tok_out/valid exact match`。
  另跑 1 次写入标准 `examples/R4/bench/reports` 目录，同样 14/14、rc=0。
- ⚠️ **环境异常（非本次改动引起，如实记录）**：上述 harness 运行中 profiler 采集失败
  （未产出 `kernel_details.csv`），所有用例 `耗时/加速比 = N/A`、性能项按 0 计入
  （harness 明示"不触发反作弊"）。该现象对**两个任务的全部 14 个用例一致出现**，
  包括 `TAIL=0`、我的改动完全不可能触及的用例；且写入 `/tmp` 与写入标准 reports 目录表现相同
  ⟹ 与 `--reports-dir` 无关、与本次 kernel 改动无关，属 profiler/环境问题。
  `npu-smi` 两卡 Health=OK，无残留 msprof 进程。**留给 Stage 3 排查**（18:29 的历史报告曾正常产出 `elapsed_us`）。

### 其他字段

- error_summary: none（全量 exit 0；投毒/对抗/重复三类门槛修复后全绿）
- design_error_reason: **none（非 [DESIGN_ERROR]）**。根因是**实现层**：尾块 top-k 源缓冲的长度对齐
  与搬运方式（`TAIL` 非 32 元素整数倍时，依赖 wheel 内 `Sort()` 的掩码补位 + 非 32B 整数倍 `DataCopy`）。
  design 的算法（定理 T1/T2/T3）、API 选型（`T.tile.topk`/`gather_mask`/`reduce_*`）、tiling（BV/K_MAX/分层归并）、
  内存层级路径（GM→UB→topk→归并）、同步策略（Developer 自动同步）**全部正确且在 NPU 上可实现**；
  修复只在同一设计框架内把尾块 top-k 源换成 32 元素对齐 + 显式 -inf 填充，**未改任何设计选择**。
  与 rev1 判定一致（两个根因均为实现层）。
- rollback: **no**（精度通过，保留修改）
- backup_path: `examples/R4/history_version/{r4_fused_sampling,r4_ref,test_r4_fused_sampling}_impl_s2_v01b_rev2_attempt1.py`
  （改前基线，sha256 三份逐一核验一致：`2b3fcd3a…` / `879afbc8…` / `e1e4a227…`）
- instrumentation_cleaned: **yes**。所有探针/复现脚本（`poison_ub.py`/`k1_diag.py`/`k1_diag2.py`/`probe_gather.py`/
  `probe_merge.py`/`probe_sentinel.py`/`probe_fix.py`/`probe_trunc.py`/`probe_pad.py`/`sweep_postfix.py`/
  `validate_variant.py`/`crossval_cases.py`/`stress_repro.py`/`dump_src.py`/`regress_v261.py`/`variantB.py`/`variantC.py`）
  均在 `/tmp/opencode/r4_fix/`，**未进入交付目录**。交付三件套 grep 核验无
  `T.printf`/`T.dump_tensor`/`print(`/`pdb`/`breakpoint(`（`r4_fused_sampling.py` 与 `r4_ref.py` 内 `print(` 计数 = 0）。
  测试文件内的 `print` 是分层标记输出（`[PRECISION_PASS]` 等），属既有约定，非插桩。
- 提交工程已同步：`bash examples/R4/cann_bench_submission/build.sh` 重新 vendor + 出 wheel；
  4 份副本 sha256 **全等 = `bbe47fa4dd75b2c8aed614446a5ac2b3023045899f227bd2b10ece7eb22bdf9f`**
  （`examples/R4/r4_fused_sampling.py` ≡ `cann_bench_submission/cann_bench/_r4_kernels.py` ≡
  `build/lib/cann_bench/_r4_kernels.py` ≡ wheel 内 ≡ 已安装 site-packages 副本；
  已安装副本用 `pip install --force-reinstall` 同步）。未手改 `_r4_kernels.py`（构建产物）。
- ruff：`ruff check r4_fused_sampling.py test_r4_fused_sampling.py` → **All checks passed!**

### next_hint

1. **建议 Stage 1 在 DESIGN.md / DESIGN_v01b.md 的尾块小节补一条硬约束**（非阻塞，实现已自洽）：
   *尾块 top-k 源的长度必须是 **32 个元素**的整数倍且其 `[TAIL, TAIL_PAD)` 区段必须由调用方显式置 -inf*；
   并记两条 wheel 级发现：① `tl::ascend::Sort()` 对 `actualCount % 32 != 0` 的掩码 `AscendC::Duplicate`
   补位在 dav_c220 上**不生效**（stale UB 会被当真实候选排序）；② `AscendC::DataCopy` 对非 32B 整数倍 count
   **向下截断到 8 个 fp32 的整数倍**（静默丢元素）。任何用 `T.tile.topk` 且 count 非 32 整数倍的算子都会中招。
2. **建议向 tilelang 上游报 wheel 缺陷**：`src/tl_templates/ascend/common.h` 的 `Sort()` 掩码补位分支
   （`Duplicate(src[alignedActual], negInf, masks={0xFE,0}, 1, 1, 0)`）在 dav_c220 上无效；
   `copy_ub_to_ub` 对非 32B 整数倍长度静默截断（应报错或走 tail-masked 路径）。本次已在算子侧规避。
3. **Stage 3 注意**：① profiler 采集当前失败（见上），需先修环境才能拿性能数据；
   ② 本修复使尾块多一次 GM→UB 读（`ztp`，≤1280 B/行）+ UB 多 1280 B（151364/196352 B，仍有余量）；
   若要省掉这次读，可把 `zt/et/mt` 一并改成 `[1, TAIL_PAD]` + `pad_value=-inf`（-inf 对 max/exp/sum 语义中性），
   但这会改动**已验证正确**的 softmax/reduce 路径，属可选优化，需重新验证 `m_out/s_out`。
4. 交付套件新增的 stability 层（N=20，阻塞）+ stress 层（投毒，可选）建议作为本仓所有含
   `T.tile.topk` 尾块路径算子的标准门槛；单次执行的用例对间歇性内存生命周期缺陷检出能力为零。
5. 运行约束沿用：wheel、不 source set_env.sh、`ACL_OP_INIT_MODE=1`、持久 `TILELANG_CACHE_DIR`、
   不 disable/clear_cache、`detect_vec_core_num()` 运行时探测（实测 40）。

### 补充：官方 API 文档已记载该约束，且文档给出的写法在本芯片上仍不安全（skill 反馈）

- `tilelang-api-best-practices/references/api-compute.md:695-716`（`T.tile.topk(dst, src, K, actual_num)`）
  **已明确写着**：「**注意事项：`src` 的大小需要满足 32 或 32 的整数倍**」，并给出示例
  「对 41 个数进行排序…需要使 41 向上对齐至 32*2=64，`actual_num = 41`，`sort_result.shape = (1, 64)`」。
  ⟹ 修复前的 `T.tile.topk(l_dst, ztt, K_MAX, TAIL=313)`（`src` 尺寸 313，既非 32 也非 32 的整数倍）
  **违反了已文档化的 API 契约**。这进一步坐实根因属**实现层**（误用 API），而非设计层。
  本次修复（`src` 尺寸与 `actual_num` 同时取 `TAIL_PAD = ceil(TAIL/32)*32`，并把 pad 显式置 -inf）
  正是该契约要求的用法。
- ⚠️ **但文档给出的写法在 dav_c220 上实测仍不安全**（`/tmp/opencode/r4_fix/probe_varE.py`）：
  按文档原样「`src` 缓冲 = 320（32 的整数倍）+ `actual_num = 313` + 不自己补位」，
  结果 **clean 就 512/512 全错、4/4 投毒 launch 全错**（整条列表右移一位）。
  ⟹ 只把**缓冲尺寸**对齐到 32 的整数倍并不够：**`actual_num` 本身也必须是 32 的整数倍**，
  否则仍会落进 `tl::ascend::Sort()` 那条在 dav_c220 上失效的掩码 `Duplicate` 补位分支。
- **建议 skill 修订（非阻塞，供 skill-journal 汇总）**：把 `api-compute.md` 的 `T.tile.topk` 注意事项改为
  「**`src` 缓冲尺寸与 `actual_num` 都必须是 32 的整数倍**；若真实元素数不是 32 的整数倍，
  调用方必须自行把缓冲补齐到 `ceil(n/32)*32`、**显式把 `[n, padded)` 区段填 -inf**，
  并以补齐后的长度作为 `actual_num` 传入。不要依赖 `Sort()` 的内部掩码补位——它在 dav_c220 上不生效，
  会静默把 stale UB 当真实候选排序（不报错、不崩，表现为结果整体偏移一位 + 间歇性垃圾大浮点）」。
  同时建议在 `tilelang-error-fixer` 增补一条故障模式：「**stale UB 读**——UB 跨 launch 不清零，
  用"先把整块 UB 填成已知毒值再 launch 目标 kernel"的手法可把间歇性缺陷变成确定性可复现」
  （本次即靠此法 100% 复现，现有 error-fixer 文档 grep `stale|未初始化|残留` 无命中）。

---

# R4 融合采样链 —— Stage 2 调试日志（rev3 split-V 并行架构）

## Attempt 1 (rev3, design_revision_count=2) — 2026-10-11T06:07:29Z

- **mode**: `first_impl`（trigger = `USER_DIRECTED_ENHANCEMENT`，非 `[DESIGN_ERROR]` 回退）
- **classification**: `precision_pass`
- **fail_category**: `none`
- **test_level**: `all`（另单独跑 `static` / `boundary` / `stress`）
- **design**: `DESIGN.md` rev3，sha256 `ca664d58ea70a82d4454c44f1e6e5c23374bf1ad8154d185cf628327ae84ab4e`（2941 行，已核对与调度指令一致）
- **交付基准**: `r4_fused_sampling.py` sha256 `19e51e8d8ee126b920cf48c915d863a9be3a349e2c68cf728d35d482b48bcc57`（改前，已核对）
  → 改后 `a764e117ce150d8677096c2e00bde3db3ebc3ec16aba63e4c613d8dffdfbd117`（620 → 1240 行）
- **测试文件**: `02a44b79624e477c570c3c632fcdcef01dcf60041de3372a177aa947d53efb50`（改前）
  → `cea988cbd117f23bb0bf50dbfe0c043dcb46cf4c0489d656b8073cc115265cb7`（1481 → 2410 行）
- **契约**: `proto.yaml` sha256 `41ab797002568e89fcc9a9eaa58be4f1cdfce20b6406abd0fdbf2ad11f7f8949` —— **未变**（§15.1 要求值一致）；`cases.yaml/csv` 未触碰
- **备份**: `history_version/r4_fused_sampling_impl_s2r3_attempt1_base.py`、`history_version/test_r4_fused_sampling_impl_s2r3_attempt1_base.py`（sha256 与改前逐一相同）
- **Stage 1 探针已固化**（`/tmp/opencode` 易失，先拷贝再实现）: `perf_tuning/rev3_probes/{r4_splitv_ub_probe.py, refine.py, refine3.py, perf_model.py}`
- **rollback**: no
- **instrumentation_cleaned**: n/a（本轮未插入任何 `T.printf` / `T.dump_tensor`；调试全部靠 D2H 读取 `pv`/输出与生成源正则解析）

### changes（本轮修改的文件与关键变更）

| 文件 | 变更 |
|---|---|
| `r4_fused_sampling.py` | 新增 4 个 kernel（`r4_topk_stats_splitA/splitB`、`r4_gumbel_sample_splitA/splitB`）+ 2 个 host 辅助（`detect_cube_core_num()`、`_r4_split_plan()`）+ 4 个 guard 常量；host wrapper `r4_fused_sampling()` 体内新增双路分派（**签名与位置参数顺序不变**，前置校验 C6/C7/C3/dtype 逐字保留且分派在其**之后**）。`@tilelang.jit` 由 3 → **7**。**现行 3 个 kernel 一字未改**（见下） |
| `test_r4_fused_sampling.py` | 新增 rev3 层：`test_r4_rev3_static()`（§9.2.5 四项静态断言）、`test_r4_rev3_l0()`（§9.2.3 的 15 例 × 2 任务，Tier-1 + Tier-2）、`test_r4_rev3_l1()`（13 例 split-path 功能）、`test_r4_rev3_l2()`（split-path 负向）、`test_r4_rev3_boundary()`（V-S4 整 split 全 -inf）；`STAB_CASES` 追加 4 个 split 配置、`ADV_CASES` 追加 4 个 split-path shape；`_run_tail_adversary` 增打印实际走的 path；`test_r4_stability`/`test_r4_stress` 增「split 覆盖 ≥1」硬断言（防 §16.8 失明点复发）；`COVERAGE_MANIFEST` 增 10 个 rev3 维度计数；`main` 增 `--level static` 并把新层接入既有 `l0/l1/l2/boundary/all` |
| `perf_tuning/round2_item4/k1_split.py` | **有意识更新** `Block Num` sanity gate（§15.3.1 的 4 步方案，见下）—— 未删除、未静默失效 |
| `bench/hap_thw_model/README.md` | 新增 §10：gate 更新说明 + 18-case 完整 `Block Num` 期望表 + harness 侧偏离的书面记录 + `t_hw` 口径 3 条注记 |
| `cann_bench_submission/README.md` | 新增 rev3 段：7-kernel 清单、`r4_ops.py` 无需改的理由、bitwise 姿态、harness gate 预期偏离、2 趟读的计时口径注记 |
| `cann_bench_submission/cann_bench/_r4_kernels.py` + `dist/*.whl` | 由 `build.sh` **重新 vendor + 重打包**（`_r4_kernels.py` 与主 kernel `cmp` 逐字节相同，sha256 `a764e117…`；wheel `f3a67770…` → `d9f31ae1…`） |

### ⭐ U1（`topk` 硬件确定性）实测结论 —— rev3 单点最高风险，**已排除**

§14.6 把 U1 列为单点最高风险：rev3 的归并树形状与现行不同（现行 38 次 `topk(512)`；rev3 是每 split
`per[s]` 次 + K1B `S` 次），若 `Sort32`/`MrgSort` 对相等值或 `-inf` 的排序依赖输入顺序，bitwise 就会被击穿；
且 §16.4 指出 K2g 的 bitwise **传递性依赖** K1 的 `tau`，U1 失败会同时击穿 4 个输出。

**按调度指令 §2 的要求，实现出第一个能跑的 K1A+K1B 后立刻打 U1，未等全部实现完**（探针
`/tmp/opencode/r4_s2r3/u1_attack.py`，按 §10.3 的调试阶梯 E1 → A3 → D1）：

| 用例 | 形态 | Tier-2 bitwise | Tier-1 golden |
|---|---|---|---|
| **E1** (V-S2) | `B=2,V=16384`，**白盒强制 `S=1, NBS=4`** | **10/10 PASS** | 10/10 PASS |
| **A3** | `B=8,V=16384`，`S=4, NBS=1`（每 split 恰 1 块） | **10/10 PASS** | 10/10 PASS |
| **D1** (V-S1) | `B=4,V=152064`，`S=10, NBS=4, S_TAIL=9, grid_A=20`，负载不均 `[4×9, 2]` | **10/10 PASS** | 10/10 PASS |

- **结论：U1 CONFIRMED SAFE。`tl::ascend::TopK<float>` 对相同值多重集在归并树形状改变后仍产出逐位相同的降序值序列。**
- **未动用 §14.6 的退路**（「K1B 把 S 份 `run_v_s` 拼成 `[S*K_MAX]` 后单次 `topk(K_MAX, S*K_MAX)`」），
  也不需要返回 `[DESIGN_ERROR]`。K1B 保持设计原样的 `for s in T.serial(S)` 逐 split 归并。
- D1 的 `w_out` 是 **4×152064 = 608256 个元素全部逐位相同**（`torch.equal` 于 `view(int32)` 位模式）。
- E1 通过 ⟹ **分解本身正确**（K1A 只做 topk / K1B 只做 softmax+归并+epilogue / `pv` 往返无损）；
  A3 通过 ⟹ 跨 split 归并（顺序、`run_v` 的 -inf 初值、`pv` 索引）正确；D1 通过 ⟹ 大 S + 负载不均 + grid 映射正确。
  三者构成 §10.3 阶梯的完整逐级排除，**无需插桩**。

### 静态自检（§11.4 步骤 1，`get_kernel_source()`，无需 NPU）

- **编译**：24 个 shape × 5 个 job = **120 份生成源全部成功**，0 `LOG(FATAL)` / 0 `ValueError` / 0 `InternalError`
  （`/tmp/opencode/r4_s2r3/static_gate.py`，日志 `static_gate_all.log`，JSON `static_gate_all.json`）
- **§9.2.5 断言①**：4 个新 kernel 的 `@tilelang.jit` 装饰器 **AST 层**核验 `pass_configs == R4_PASS_CONFIGS`，
  且 `R4_PASS_CONFIGS['tl.ascend_memory_planning'] is True`（R-MEM：`ascend_memory_planning.cc:598` 的兄弟路径
  是 `check_overflow=false` 即静默溢出，只有开这项才走 `LOG(FATAL)`）→ **PASS**
- **§9.2.5 断言②**：3 个冻结 kernel 的源码区段 sha256 **逐一相同**（见下节）→ **PASS**
- **§9.2.5 断言③（UB 实测，非模型）**：解析 `ascend_ub.GetWithOffset<T>(n,off)` 取 `max(off+n*sizeof(T))`，
  与生成源 `InitBuffer(ascend_ub, 196352)` 比较 → **全部 ≤ 196352，PASS**

  | kernel | 本轮实测最差 UB | 占比 | 余量 | 最差 shape | §4.5 设计值 | 一致性 |
  |---|---|---|---|---|---|---|
  | K1A | **106496** | 54.24% | 89856 | case6 `(8,10240,TAIL=2048,S=3)` | 106496 | ✅ 精确吻合 |
  | K1B | **104100** | 53.02% | 92252 | case6（`TAIL=2048` 使 `[1,TAIL]` softmax 缓冲变大） | 89444（case17） | ⚠️ **设计值偏低**，见下 |
  | K2gA | **148420** | 75.59% | 47932 | 所有 `BV=4096` case | 148420 | ✅ 精确吻合 |
  | K2gB | **132** | 0.07% | 196220 | — | 132 | ✅ 精确吻合 |

  ⚠️ **对 §4.5 的一处如实修正**：设计写「K1B ≤ 89444 B（case17, TAIL_PAD=832）」，但 Stage 1 只对 3 个 shape
  探过 K1B。本轮 24-shape 全扫发现 **K1B 的真正最差是 case6（`V=10240 → TAIL=2048`）= 104100 B（53.02%）**，
  因为 K1B 保留了尾块 softmax 的 `zt/et/mt = [1,TAIL]×3`，`TAIL=2048` 时这 3 项就是 24576 B。
  我自己在 case17 上实测 K1B = **89444（NEED_MIN_P=False）/ 89508（=True）**，与设计值吻合 ⟹ 设计的公式没错，
  只是**取的 worst-case shape 不是全局最差**。**104100 仍远小于 196352（余量 92252 B），不构成问题**，
  且 case6 被 guard 否决、实际不走 split 路径（但 K1A/K1B 仍必须能编译它，L0-E1 类白盒会构造）。
- **§9.2.5 断言④（精度红线的编译期证明）**：
  - 全部 TopK 调用点的 `(K, repeatTimes, actual_num)` 并集 =
    `{(128,8,256), (128,64,2048), (256,9,288), (256,10,320), (256,16,512), (256,26,832), (256,64,2048), (256,128,4096)}`
    ⟹ `repeatTimes*32 - actual_num` = **全 0** ⟹ **`padCount == 0`，缺陷 A 结构性不可触发**
  - `actual_num` 并集 = `{256, 288, 320, 512, 832, 2048, 4096}` = `{2·K_MAX} ∪ {TAIL_PAD: 288(=257/261→288), 320(313), 512, 832(825), 2048} ∪ {BV: 2048, 4096}`
    ⟹ **与定理 P1 的预测完全一致：rev3 未引入任何新的 `actual_num` 取值**
  - `copy_ub_to_ub` 元素计数并集 = `{128, 256, 288, 320, 512, 832, 2048}` → 字节 `{512,1024,1152,1280,2048,3328,8192}`
    ⟹ **全部 `%32 == 0`，缺陷 B 结构性不可触发**
  - **零动态循环边界**：全部 `for (int32_t …; … < N; …)` 的 `N` 都是字面常量（观测到 `{1,2,3,4,5,6,8,10,19,37,38,128,256}`）
- **`_r4_split_plan` 复现 §5.8.4 / §13.4 全表**：18 case **0 失配**（`S`/`NBS`/`S_TAIL`/`use_split` 逐项相同），
  split 路径 = **6/18 = {3,4,7,8,12,17}** ✅；定理 P3 生效（F1: 朴素 `S=4` → 收缩 `S=3`，per `[2,2,1]`，**无空 split**）
- **L0 15 例的 `S/NBS/S_TAIL/per` 与 §9.2.3 表格逐一吻合**（A2/D1 `S=10,grid=20` 恰 1 满波；B1-B4 `S=6,S_TAIL=5`；
  B5 `S=2,NBS=19,S_TAIL=1`；C1 `b_num=10→split`；**C2 `b_num=11→现行`**（§0.2 判据收紧的唯一可观测点）；
  C3 `b_num=20→现行`；E1 白盒 `S=1,NBS=4`；F1 `S=3`）

### ⭐ 现行 3 个 kernel 逐字节未变的证明（用户硬约束 3）

AST 提取每个函数的完整源码区段（含装饰器行 → `return main`），改前/改后 sha256 对比
（工具 `/tmp/opencode/r4_s2r3/verify_frozen.py`，已内化为测试的**静态断言②**，每次跑测都复验）：

| kernel | 行号（改前 = 改后） | 字节数 | sha256（改前 = 改后） |
|---|---|---|---|
| `r4_topk_stats` | L91-364 | 16387 | `d39090631598f07d1007917b310216b2f650a4ab4c3e7dd6b60b3e63a3a94831` |
| `r4_mask_sample` | L367-412 | 2221 | `f787d1ad05a9fc26913b98de34f3a48ee67f730530ad6add8761082ef187a3d1` |
| `r4_gumbel_sample` | L415-555 | 10306 | `ca90de8d0f6f4f69f366f380009f43915aa3c9f5b90430d683be239916683889` |

⟹ **RESULT: ALL 3 FROZEN KERNELS BYTE-IDENTICAL**（行号都未漂移，因为新代码全部插在 L555 之后）。
round2 #4 已删除的「L211 UB→UB hop」**未复活**（K1A 沿用 `z_ub` rank-1 双用形态，K1B 不含该 hop）。

### 分层测试结果

| 层 | 用例数 | 结果 |
|---|---|---|
| **static**（§9.2.5 ①②③④） | 4 断言 × 30 份生成源 | **38 STATIC_PASS / 0 FAIL** |
| **L0 legacy**（rev2 的 8 例，未删减） | 8 | 全 PASS |
| **L0-A / L0-B**（V01-b hash + tok bitwise，未删减） | 3 seed×2 + 8(+3 var) | 全 PASS |
| ⭐ **L0-REV3**（§9.2.3 的 15 例 × 2 任务 = 30 次运行） | 15（split-path 13 / current-path 2） | **Tier-1 150/150，Tier-2 bitwise 130/130** |
| **L1 legacy**（18 例，未删减） | 18 | 全 PASS |
| **L1 gumbel + distribution**（未删减） | — | 全 PASS |
| ⭐ **L1-REV3**（split-path 功能，Tier-1 + Tier-2） | 13 × 2 任务 | 全 PASS |
| **L2 legacy + gumbel**（未删减） | — | 全 BOUNDARY_PASS |
| ⭐ **L2-REV3**（split-path 负向：dtype/C3/C7 拒绝 + 守卫在 split 路上同样触发） | 10 | 全 BOUNDARY_PASS |
| **Boundary legacy + gumbel**（未删减） | — | 全 BOUNDARY_PASS |
| ⭐ **Boundary-REV3（V-S4）** | 4 × 2 任务 | 见下「BOUNDARY_WARN 归因」 |
| ⭐ **stability**（12 configs × 20 reps，含 **4 个新 split 配置**） | 12 + 8 ADV sweep | **全 `selfdiff=0 goldenfail=0 invariant=0`，ADV `dropped=0`** |
| ⭐ **stress**（UB 投毒 1e30 + 200 reps，**投毒施加到 4 个新 kernel**） | 12 | **全 `selfdiff=0 goldenfail=0 invariant=0`** |

### BOUNDARY_WARN 归因（0 → 6）—— 既有缺陷，非 rev3 回归，已实测定位根因

6 条 WARN **全部**来自本轮新增的 V-S4「整 split 全 -inf」用例（3 shape × 2 任务），且**每条都 `tier2_bitwise=5/5`**。
根因用独立探针实测确认（`/tmp/opencode/r4_s2r3/vs4_rootcause.py`），**不是推断**：

- **机制**：当**首个整块**全为 `-inf` 时，`m_run = -inf` 且 `m_blk = -inf` ⟹ `m_old = max(m_run,m_blk) = -inf`
  ⟹ `tmp = m_run - m_old = -inf − (-inf) = NaN` ⟹ `s_run *= exp(NaN)` 永久污染 `s`；
  继而 `q2 = exp(run_v−m)/NaN = NaN` → `prefix = NaN` → `tau_p` 冻结在 `run_v[0] (= m)`、`okp = 0` → **`valid = 0`**。
- **三重证据证明与 rev3 无关**：① 每条 `tier2_bitwise = 5/5`；② **对照组 `B=64,V=16384`（`use_split=False, S=1`，
  走逐字节不变的现行路径）复现完全相同的 NaN**；③ **直接调用冻结的 `r4_topk_stats`（完全绕过 `_r4_split_plan` 分派）
  与 wrapper 输出 bitwise 相同**。
- **安全降级**：4 个场景 `valid` **全 0** ⟹ D-GUARD 触发 ⟹ host 回退 torch 路径，**不会静默返回错误 token/概率**。
  `m_out`/`tau_out` 有限；只有 `s_out` 是 NaN。
- **稀疏 `-inf` 对照组完全干净**：`bnd_sparse_neginf_split`（split 路径，`S=6,S_TAIL=5`）5 输出 tier1 `r=1.0000`
  （`m_out`/`tau_out`/`valid` 误差恰 0）+ tier2 `5/5` ⟹ `[BOUNDARY_PASS]`。
- **⭐ 对 §9.3 / §14.3(iii) 先验的实测修正**：设计先验「跨 split 归并遇 `-inf` 安全」**成立**（`topk` 只比较搬运，
  `run_v` 的 -inf 初值挤出等价性经 E1/A3/D1 + 稀疏对照组验证）；但设计**未预料到** NaN 来自
  **online-softmax 的重标定**（`s_run *= exp(m_prev − m_new)`）而非归并 —— 该代码 rev3 逐条照抄，故两路完全一致。
  **V-S4 的实测答案是「归并安全、softmax 重标定在整块 -inf 下不安全，且该不安全是 rev2 既有的」。**
- **处置**：**不修**。修它会同时违反用户硬约束 3（3 个 kernel 逐字节保留）与硬约束 2
  （K1B 的 softmax 扫描必须与现行逐条相同 —— 这正是 `m/s` bitwise 的构造性依据）。
  ⟹ 属 **rev3 范围外的既有缺陷，如实上报，交 Orchestrator/用户决定是否另开一轮**。
  测试侧把语义精确化：Tier-1 的 `[BOUNDARY_WARN]` 如实保留（对 golden 确有偏差），
  另**新增一条配对的 `[BOUNDARY_PASS] rev3-invariants`** 断言 rev3 真正负责的两条性质
  （双路 bitwise 相同 + 双路都安全降级 `valid` 全 0），并打印 `s_out_is_nan=True (both paths)`。
  **6/6 该不变量 PASS，0 条 "rev3 invariant broke"** ⟹ WARN 不再可能被误读成 rev3 回归。

### §16.8 门禁失明点 —— 已修掉，且加了防复发的硬断言

- 实测确认设计的判断：8 个 legacy `STAB_CASES` 的 path 全是 `CURRENT`（新打印字段直接可见，例如
  `adv B=2 V=4409 … path=CURRENT(S=2)`）、4 个 legacy `ADV_CASES` 同样 `CURRENT` ⟹ **改前的 stability/stress
  跑再多 reps 也碰不到 split 路径一行代码**。
- **新增 split-path STAB 配置 4 个**（≥ 设计要求的 A2/B5/F1）：`stab_split_A2_B4_V152064_gumbel`（`S=10,grid_A=20`）、
  `stab_split_A2_B4_V152064_mask`、`stab_split_B5_B16_V152377_gumbel`（`S=2,NBS=19,S_TAIL=1`）、
  `stab_split_F1_B10_V20480_gumbel`（`S=3`，定理 P3 收缩形）。**× 20 reps（stability）与 × 200 reps + UB 投毒（stress）全绿。**
  A2 的 mask+fused 双配置确保 K1A/K1B/K2gA/K2gB **四个新 kernel 都被投毒覆盖**（V-S5）。
- **新增 split-path tail-adversary 4 个**（B1-B4 的 V）：`(4,20737) TAIL=257`、`(4,20793) TAIL=313`、
  `(4,20991) TAIL=511`、`(4,20741) TAIL=261`，全部 `path=SPLIT(S=6,NBS=1,S_TAIL=5)`，`dropped=0`
  ⟹ **证明尾块被正确的 split（`S_TAIL=5`）处理且未丢元素**（V-S6）。
  ⚠️ **未纳入 B5 的 `V=152377`（`TAIL=825`）作为 adversary**：哨兵要逐个尾块 index 扫一遍（825 次 launch），
  成本约为已加 4 个之和（1342 次）的 61%；B5 的尾块正确性由 L0-B5（Tier-1+Tier-2）+ `stab_split_B5_…`（20 reps）
  + 静态断言④（`TAIL_PAD=832`，`(256,26,832)` 的 `padCount=0`）三重覆盖。**这是有意识的范围取舍，如实披露。**
- **防复发**：`_rev3_count_split_configs()` + `test_r4_stability` / `test_r4_stress` 各加一条**阻塞**断言
  「split-path 的 STAB 配置数 ≥ 1（stress 同样）；启用 adversary 时 ADV 的 split 配置数 ≥ 1」，
  失败打 `[PRECISION_FAIL]`。当前实测 `STAB=4/12, ADV=4/8` ⟹ 若将来 guard 一改导致这些配置全部退回现行路径，
  门禁会**立刻显式失败**而不是继续静默零覆盖。

### 四门禁 exit code 与计数（改前 → 改后，**无回退**）

改前基线用 **pristine 副本**（`history_version/*_impl_s2r3_attempt1_base.py`，sha256 `19e51e8d…` / `02a44b79…`）
在隔离目录实跑取得，**不是引用设计文档的数字**：

| 门禁 | 命令 | 改前（实测） | 改后（实测） | exit | 判定 |
|---|---|---|---|---|---|
| all | `--level all` | 74 PRECISION_PASS / 0 FAIL，38 BOUNDARY_PASS / 0 WARN，`Test Passed!` | **143+ PRECISION_PASS / 0 FAIL**，**50+ BOUNDARY_PASS / 6 WARN**，131 BITWISE_PASS / 0 FAIL，38 STATIC_PASS / 0 FAIL，`Test Passed!` | **0** | ✅ 无回退（PASS 计数只增不减；6 WARN 已归因为既有缺陷） |
| stability | `--level stability` / `all` 内 | 12 项 | **12 configs × 20 reps + 8 ADV sweep 全 PASS**（含 4+4 个新 split 配置） | **0** | ✅ |
| stress | `--level stress` | 8/8 | **12/12**（`selfdiff=0 goldenfail=0 invariant=0` × 200 reps，投毒覆盖 4 个新 kernel） | **0** | ✅ 计数增加 |
| coverage | `coverage_check.py test_r4_fused_sampling.py --proto proto.yaml` | 25 PASS / 0 MISS / 0 N/A | **25 PASS / 0 MISS / 0 N/A** | **0** | ✅ 精确保持（`proto.yaml` 未变 ⟹ 应覆盖维度不变） |
| static（新） | `--level static` | — | 38 STATIC_PASS / 0 STATIC_FAIL | **0** | ✅ 新增门禁 |

- ruff：`test_r4_fused_sampling.py` + `k1_split.py` → **All checks passed!**；
  `r4_fused_sampling.py` → **仅 1 个既有错误** `F841 AT1 assigned but never used`（`:442`，位于**冻结的**
  `r4_gumbel_sample` 内，是 round1 [#2] 把 `topk(K=1)` 改成 `reduce_max/reduce_min` 后遗留的死变量）。
  **按硬约束 3 不能改它** ⟹ 本轮**新增代码零 ruff 错误**，该既有项如实披露。

### `Block Num` sanity gate（§15.3.1）—— 已按 4 步方案有意识更新（**改动披露**）

- **定位**（§15.3.1 步骤 3 要求先定位）：R4 侧唯一实现是 `perf_tuning/round2_item4/k1_split.py`
  （`:9` 文档、`:32` 读 `int(r["Block Num"])`、`:42-46` 比 `ceil(B/2)`）。
  **`bench/hap_thw_model/gates_report.py` 确实不含 Block Num**（`grep -n "Block Num\|block_num\|blk"` 无命中）
  ⟹ **印证 Stage 1 记录的信息缺口**：GATE 1-4 全是 `t_hw` 自洽性检查，与该 gate 无关。
  harness 侧的 gate 属**只读、另一 agent 所有**，**未修改**，改为在两个 README 里书面记录偏离 + 附完整期望表。
- **旧代码会静默失效**：`if len(seq) != 2: bad += 1; continue` 会把 split-path 的 step（mask 3 行 / fused 4 行）
  **整步丢弃**（6 case × 8 step = 48/144 step，占 1/3 数据）而报告看起来仍正常 —— 正是红线禁止的失败模式。
- **本轮改动**（步骤 1/2/4）：
  1. 期望值改为**由 `_r4_split_plan` 派生**（从交付 kernel `import`，**单一真相源**；判据一改 gate 自动跟随，
     结构上不可能与实现脱节）；case 的 `(B,V,K_MAX,BV)` 改为从 `r4_thw_common.load_cases()` 取，
     **删掉重复硬编码的 `B_BY_CASE`**；
  2. 每 case 的 `path`/`S`/`NBS`/`S_TAIL`/`rows`/期望-vs-实际 `Block Num` 序列成为**一等打印输出**；
     行数不符时**报出 case id 与期望/实际值**，绝不 `continue` 跳过；
  3. **新增 gate**：每 step 的 `Block Num` **之和**必须等于计划值之和；且每个 **split-并行** kernel 的
     `Block Num ≤ CUBE_CORE_NUM`（**定理 P2 的运行时验证**，R-SGUARD 第三重防护）；异常或 P2 违反则 **exit 非 0**；
  4. 新增第 3 个 CLI 参数 `task`（`mask|fused`，缺省从 label 推断并**总是打印**推断值）——
     因为 split 路径 mask 是 3 行、fused 是 4 行，必须知道 task 才能派生期望。
- **干跑验证**（无需 CSV，直接对 18 case × 2 task 派生）：mask 6/18 split、fused 6/18 split；
  期望序列如 case7 fused = `[20, 2, 20, 2]`（4 行）、case17 mask = `[16, 8, 8]`（3 行）；
  **P2 全 OK，最紧点是 case7 的 split-并行 `Block Num = 20 == CUBE_CORE_NUM`（恰好 1 满波）**。
  完整 18-case 表已写入 `bench/hap_thw_model/README.md §10.3`。

### 未验证项 / 信息缺口（如实列出，对应 §14.6 的 U1-U6 与 §10.2 的 V-S1..V-S8）

| 编号 | 内容 | 本轮状态 |
|---|---|---|
| **U1 / V-S1** | `topk` 硬件确定性（归并树形状改变后仍逐位相同） | ✅ **已实测排除**（E1+A3+D1，Tier-2 30/30；D1 含 `w_out` 608256 元素全逐位） |
| **V-S2** | `S=1` 退化等价（分解正确性） | ✅ **已实测通过**（E1 白盒强制 `S=1,NBS=4`，10/10 bitwise） |
| **V-S4** | 「某 split 全为 -inf」的 Boundary 行为 | ✅ **已实测**，结论见上「BOUNDARY_WARN 归因」：归并安全；**但发现 rev2 既有的 softmax 重标定 NaN（整块 -inf 时），安全降级 `valid=0`，rev3 逐位保留未修** |
| **V-S5** | UB 投毒施加到 4 个新 kernel | ✅ **已实测**（`--level stress` 12 configs × 200 reps，4 个 split 配置全 `selfdiff=0 goldenfail=0 invariant=0`） |
| **V-S6** | split 路径的 tail-adversary 哨兵 | ✅ **已实测**（4 个 `path=SPLIT(S=6,NBS=1,S_TAIL=5)` shape，`dropped=0`）。⚠️ B5 的 `TAIL=825` 未做 adversary（成本取舍，已披露，另有三重覆盖） |
| **U2 / V-S3** | `a_launch` 实测值 | 🔴 **仍悬空**。本轮**未做性能测量**（Stage 2 只做正确性；且 `D-SPLIT-GUARD` 仍用保守默认 `a_launch_est=8.0us`，未放宽）。⟹ **guard 的三重收益判据仍是模型值，不是实测值** |
| **U3 / V-S8** | `F(K_MAX=128)` 与 `p(2048)` 真值；case8 的 split 收益 | 🔴 **仍悬空**。`l0_A4_split_k128_bv2048` 与 `l1r3_split_k128bv2048_keq` 证明该特化**正确**（Tier-1+Tier-2 全过、UB 45056/40036），但**未测收益**；若 Stage 3 实测 case8 回退 >3%，退路是 guard 加 `and K_MAX == 256`（一行） |
| **U4 / V-S7** | `grid = b_num*S` 的真实调度是否 1 波 | 🟡 **部分**：静态侧已证生成码地址算式与静态边界正确、`_r4_split_plan` 派生的 `grid_A` 与 §5.8.4 全表吻合（case7 = 20）；**msprof 的 `Block Num` 运行时读数未采**（本轮未跑 profiling）。已把该断言做进更新后的 `k1_split.py`（P2 gate），Stage 3 一跑 msprof 即自动验证 |
| **U5** | K1A 的 4 个 TopK 调用点共享单一 `tmp_ub` arena 是否有复用竞态 | 🟢 **间接强证据**：UB 投毒 × 200 reps × 4 个 split 配置全 `selfdiff=0`（若有 arena 竞态，投毒下必然出现 launch 间差异）。**未做独立的 arena 隔离实验** |
| **U6** | `torch.empty` 的 workspace 是否可能返回毒值从而掩盖 K1A 漏写 | 🟢 **已缓解**：F1 用例（专测空 split）+ stress 的 UB 投毒 + 定理 P3 的结构性保证（`_rev3_run_l0_case` 对每个 split 用例断言 `min(per) >= 1`，实测无一为空）。交付版按 §5.6 用 `torch.empty`（未用 `torch.zeros`） |
| cann-bench 36/36 | `mismatch_count == 0` 端到端复跑（跑的是**提交副本** `_r4_kernels.py`） | 🔴 **本轮未跑**。已完成其前置：`_r4_kernels.py` 与主 kernel `cmp` 逐字节相同、wheel 已重打包、`Block Num` gate 已更新、README 已记录 harness 侧偏离。**留给 Stage 3**（§11.4 步骤 11） |
| 性能数字 | §12.4 的 1.15×–1.29× / case7 3.5×–4.2× 预测 | 🔴 **本轮一个性能数字都没测**，全部仍是模型估计。日志里出现的 `(0.0s)/(5.7s)` 是**含 JIT 编译的墙钟**，不是 kernel 时长，**不得当性能数据引用** |

### next_hint

1. **Stage 3 优先级**：① 跑 cann-bench eval 复验 **36/36 `mismatch_count=0`**（这是比 Tier-2 更强的端到端验证，
   因为它跑提交副本）；② 用更新后的 `k1_split.py <csv> <label> [mask|fused]` 采 msprof，
   **实测 `a_launch`（V-S3）** 并回填 `_SPLIT_A_LAUNCH_US`（当前保守 8.0us），再据 §12.4 敏感性表决定是否放宽
   `D-SPLIT-GUARD` 以纳入 case 6/11；③ **单独实测 case8（V-S8）**，若回退 >3% 就在 guard 里加 `and K_MAX == 256`；
   ④ 跑 `gates_report.py` 确认 GATE 1-4 仍 PASS（`t_hw` 变化 <0.2%）。
2. **若 Stage 3 采 msprof**：注意 `Block Num` 现在是 `[b_num*S, b_num, …]` 而非恒定 `ceil(B/2)`；
   harness 侧那条只读 gate 会对 case {3,4,7,8,12,17} 报错，**这是已记录的预期偏离，不是 kernel 缺陷**
   （两个 README 都写了，期望表在 `bench/hap_thw_model/README.md §10.3`）。
3. **既有缺陷（建议单独立项，不属 rev3）**：整块前导 `-inf` ⟹ `s_run` NaN ⟹ `valid=0` 安全降级。
   修法很小（`m_old = max(m_run, m_blk)` 后把 `tmp = m_run - m_old` 的 `-inf − (-inf)` 情形夹到 0，
   或首轮跳过重标定），但**会改变 `m/s` 的位模式** ⟹ 必须同时放宽/重定义 Tier-2 的适用范围，
   且要动 3 个冻结 kernel。**需用户/Orchestrator 明确授权后才能做**，Stage 2 无权自行变更契约语义。
4. **§4.5 建议 Stage 1 修订**：K1B 的 worst-case 不是 case17（89444）而是 case6（**104100**，因 `TAIL=2048` 的
   `[1,TAIL]×3` softmax 缓冲）。数值仍安全（余量 92252 B），但设计表的「最差 case」标注应更正，
   以免将来有人按 89444 去核算「增大 BV/TAIL 的余量」。
5. **运行约束沿用**：wheel `0.1.4+d75300ed…`、**不 source `set_env.sh`**、cwd 在 repo 根之外、
   `ACL_OP_INIT_MODE=1`、`LD_LIBRARY_PATH` 含 `/usr/local/sqlite3/lib`、持久 `TILELANG_CACHE_DIR`、
   不 `disable_cache()/clear_cache()`、`detect_vec_core_num()` 运行时探测（实测 40 → `CUBE_CORE_NUM=20`）。
   ⚠️ 后台长任务必须 `setsid` 脱离会话，否则会被 shell 回收（本轮踩过一次：首次基线跑被杀、日志为空）。
6. **新增 JIT 特化**：模块内 `@tilelang.jit` 3 → 7；特化键新增 `S/NBS/S_TAIL`（由 `(B,V,BV,CUBE_CORE_NUM)`
   函数依赖确定，同设备不增加独立维度）。本轮实际编译：**静态门禁 120 份生成源** + 上板 JIT（L0 23 shape、
   L1 31 shape、stability/stress 12 config）；单 shape 首次编译约 **5s（`get_kernel_source`）/ 5-22s（含 .so 上板）**，
   缓存命中后 **≈0.0-0.3s**。**这些编译耗时单列，不计入任何性能数字。**

### 本轮证据工件已固化到仓库（`/tmp/opencode` 易失，不可作为交付证据）

`examples/R4/perf_tuning/rev3_probes/stage2/`：

| 文件 | 内容 |
|---|---|
| `static_gate.py` + `static_gate_all.log` + `static_gate_all.json` | §11.4 步骤 1 的静态门禁（24 shape × 5 job = 120 份生成源；UB / padCount / UB→UB 32B / 动态边界 / pass_configs） |
| `u1_attack.py` + `u1_d1.log` | **U1 攻坚**（E1 → A3 → D1 阶梯，Tier-2 位模式对照） |
| `vs4_rootcause.py` | **V-S4 根因探针**（证明整块 -inf 的 `s_run` NaN 与路径无关、且安全降级 `valid=0`） |
| `verify_frozen.py` | 3 个冻结 kernel 的 AST 区段 sha256 对比（已内化为测试静态断言②） |
| `precheck_l1.py` | rev3 L1/Boundary 候选用例的 CPU 预验证（§9.2.1 强制：`valid_ref` 全 1 + 确为 split 路径 + V-S4 配方无 NaN） |
| `submission_smoke.py` | 已安装 wheel 与开发模块的端到端逐位对照（5 case × 2 任务 × 5 输出 = 50 项） |
| `baseline_all.log` | **改前基线**（pristine 副本 `19e51e8d…`/`02a44b79…` 隔离目录实跑）：74 PRECISION_PASS / 0 FAIL + 38 BOUNDARY_PASS / 0 WARN |
| `new_l0.log` / `new_all.log` / `final_all.log` / `new_stress.log`(在 all 内) / `new_boundary.log` / `new_static.log` | 改后各门禁的完整输出 |

（上文正文中出现的 `/tmp/opencode/r4_s2r3/...` 路径均以此目录下的同名文件为准。）
