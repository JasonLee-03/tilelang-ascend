# GQA debug log — eval fix attempt 1 (cann_final_eval_20260909_102400, 29/80 failures)

## Attempt 2 — 2026-09-10 (mode: precision_fix, eval-fix task: cann_final_eval_20260910_071341, 25/100 failures)
- mode: precision_fix (CANN-Bench online eval failure repair, round 2)
- classification: precision_pass (all local verification suites + the local
  cann-bench 51-case variant eval pass; see per-fix status below)
- fail_category: none (targets: G1 Fill / G2 special values / G3 compile / G4 crash)
- test_level: all (eval_fix_verify/ suites + local cann-bench eval)
- coverage: g2_special_values 24/24 · g123_smoke 14/14 · g1_fill_guards 8/8 ·
  g4_stress_100 5/5 (100 repeats each) · g3c_fallback_selftest 4/4 · F4 6/6 ·
  F3 9/9 · F12 10/10 · F6 accuracy 132/132 · F6 perf spotcheck PASS (kernel
  <= 0.5%) · local cann-bench eval 51/51 (20 public + 31 variants)
- boundary_warnings: bf16 dtype-TRUE full range ([-3.4e38, 3.4e38]) is out of
  the eval variant set (which uses the fp16 bound pair [-65504, 65504] for
  both dtypes) and remains NaN-mismatched — documented, not counted
- changes: examples/gqa/gqa.py only (G1-G4 below); verification scripts under
  examples/gqa/eval_fix_verify/ (g*/dbg_* new)
- rollback: no
- backup_path: examples/gqa/history_version/gqa_impl_eval_fix_attempt2.py
- instrumentation_cleaned: n/a (all diagnostics were host-side scripts in
  eval_fix_verify/; no kernel-side debug instrumentation)

### Root causes → fixes

**[G4+G2-case67 — THE BIG ONE: non-aligned d2d→launch ordering race]**
- Symptom: repeated gqa() calls on ANY padded (non-aligned) shape produce
  nondeterministic, partially-NaN outputs (up to 100% of elements differ;
  NaN counts vary per call) — reproduced on BOTH the current code and the
  round-2 eval baseline (gqa_impl_eval_fix_attempt2.py).  Aligned shapes
  are deterministic.  Kernel inputs verified BIT-IDENTICAL across calls
  (Spy-instrumented, dbg_nondet10) yet outputs differ; driving the raw
  compiled kernel with fixed tensors + torch.npu.synchronize() before the
  launch = 6/6 bit-stable (dbg_nondet13).
- Root cause: the tilelang kernel launch is NOT reliably stream-ordered
  against preceding torch device-to-device copy ops — the padding
  slice-assigns (_pad_along ViewCopy) and Mask_full construction run right
  before the launch on the padded path; the kernel can read the pad
  targets before the copies complete.  Aligned paths insert NO torch ops
  between the transposes and the kernel (mask H2D is a blocking copy) and
  never race.  This explains eval case 67 ("NaN位置不匹配" — a raced
  correctness call on a special-value non-aligned case) and very likely
  the 84-100 rc=-11 cascade (repeated raced calls during warmup/perf on
  non-aligned cases; case 84 the trigger, 85-100 the casualties).
- Fix: _npu_sync_tensor() barrier at the three torch-op→launch edges on
  the PADDED path only: (a) _prefill_run_padded after pad/Mask_full build,
  (b) decode _run when padding happened without a blocking mask H2D,
  (c) transpose_out's non-contiguous (crop materialization) branch.
  Aligned bench paths are byte-identical (no sync inserted).
- Verification: dbg_nondet2 (non-aligned causal/non-causal 5-call
  bit-stable), g4_stress_100 5/5 (B2S124 fullrange/inf/nan, D256 S252,
  B60S2Skv2044 decode-causal fullrange — 100 repeats each, bitwise-stable,
  zero exceptions), all accuracy suites re-run clean.

**[G1 / cases 71/72/73/81/82 — fp16/bf16 Fill binary]**
- Root cause: F2's _pad_along used torch.full(0, fp16/bf16, device=npu) →
  aclnnInplaceFillScalar (FillAiCore) fp16/bf16 binary missing on the eval
  SoC (561103); fp32 Fill is available (Mask_full passed both rounds —
  the binary dispatches by dtype).
- Fix: _ZERO_POOL resident 1D zero slab per (dtype, device) — ONE CPU
  zeros + single .to(device) H2D at first use (warmup; anti-cheat guards
  D2H egress, not H2D) — + _pad_along rewritten as torch.empty (pure
  alloc, no kernel) + valid-region slice assign (d2d ViewCopy) +
  pad-region slice assign from the zero-slab view (NPU slice copy).
  Developer-legacy _run padding converted to _pad_along;
  _build_causal_mask S==1 branch moved off NPU ZerosLike (dead path,
  hygiene).
- Verification: g1_fill_guards 8/8 (guards torch.full/zeros/zero_/fill_
  for fp16/bf16 on NPU; padding shapes of all 5 failure classes + D512;
  outputs match golden).

**[G2 / case 67 — special-value semantics (independent, real bugs beyond
the race)]**  (repro: g2_special_values.py, eval-faithful: fp64-CPU golden
+ native CPU reference + the bench's own compare_tensors)
- (a) fp16 full-range CAUSAL value corruption (mere=3.7 pre-fix): with the
  mask constant -(2^30), a full-range dot product (|s| up to ~5.5e11)
  lets a MASKED column win the W5 (max-after-mask) running max
  (s_j - 2^30 > max(visible)) → the mask no longer underflows exp() →
  causal rows attend to future tokens.  FIX: _BIG_NEG -(2^30) → -1e30
  (dominates every finite fp32 score; bit-identical for normal ranges).
- (b) decode-causal fullrange rows output 0 (mare=1.0, 3 rows of 3840,
  dbg_decode_causal.py): the max-over-RAW-scores kernels (per-head, AUTO
  dense, dense_expert x2, developer) let a masked column's huge RAW score
  lift the running max so every VISIBLE weight underflows → output 0
  while the golden (masked_fill before softmax) attends normally.  FIX:
  W5 reorder — the additive mask is now applied BEFORE the running max in
  ALL kernels (per-head, AUTO dense, dense_expert, dense_expert_d256,
  developer; g2/prefill_expert/g2stack were already W5), with
  irresistible -1e30 masks the masked column can never win the max, and
  the m_init/neg_sm sentinels (below) keep fully-masked rows at P==0.
- (c) running-max init sentinels: the historical -(2^30)/2^30 inits act as
  CLAMPS for full-range rows (all-negative rows / few-visible causal
  rows zero every P → output 0 vs golden one-hot).  New inits:
  max-form kernels m_i init = sm_scale*_BIG_NEG (pure sentinel below any
  finite fp32 scaled score); W5 neg_sm init/cap = -sm_scale*_BIG_NEG/2
  (sentinel + fully-masked-row max floor at mask/2 so the mask cannot
  cancel in the shift); g2's [F4b] clamp-at-0 REMOVED (breaks full-range
  rows below 0) and replaced by the floor _BIG_NEG*1e-3 (g2 adds the
  unscaled mask to scaled scores).
- (d) decode mask negative-slice: mask_start_j < 0 (decode S > S_kv)
  wrapped around Python slice semantics — clamped to 0.
- (e) NaN/Inf propagation: verified to match the golden's fp64 semantics
  (this torch build's softmax([inf,1]) = NaN row, matching the kernel's
  exp(inf-inf)=NaN; NaN rows propagate identically) — 24/24 PASS.
- Boundary (documented, out of eval scope): dtype-TRUE bf16 full range
  ([-3.4e38,3.4e38]) overflows fp32 scores → kernel NaN vs golden finite.
  Matching would require fp64 dot products (no Ascend cube support); the
  eval variant set uses [-65504,65504] for both dtypes, which passes.

**[G3 / case 56 — L0C overflow; case 68 — bisheng compile]**
- case 56 ROOT CAUSE CONFIRMED LOCALLY (not a version difference): the
  per-head kernel at D=512, BM=64 needs acc_o_l0c [2,64,512] f32 = 262144B
  — the eval's EXACT error ("Memory allocation failed for: acc_o_l0c
  required: 262144") reproduces verbatim (eval_fix_verify/g3c_d512_probe).
  Also reproduced: D=256 non-causal pipelined (threads=1 single-scope
  planning) exhausts UB ("acc_o_ub required: 65536, available: 0",
  g3b_crash_iso.py).
- Fixes:
  (1) per-head use_pipeline now requires D <= 128 (the serial threads=2
  form splits C/V scopes and compiles at D=256);
  (2) per-head block_M is L0C-budget-gated: _perhead_bm_ladder picks the
  largest BM in {64,32,16} with (2*BM*BN + 2*BM*D)*4 <= 192KB — the
  tightest envelope empirically accepted by BOTH the local pip tilelang
  and the eval build (base6 = D256@BM64 = exactly 192KB passes both;
  D512@BM64 = 320KB fails everywhere → BM=32 primary, verified correct);
  (3) dispatch compile-fallback chain: every dispatch builds an ordered
  candidate list — prefill [g2stack?, prefill_expert?, per-head ladder,
  g2 (re-enabled, LAST fallback only)], decode [dense_expert?/
  dense_expert_d256?, AUTO dense] — and takes the first factory that
  COMPILES.  Compile failures raise at factory time (JITKernel.__init__
  compiles eagerly), so the chain can never mask runtime/precision
  defects.  Resolution cached per shape (_KERNEL_BUILD_CACHE).  Covers
  case 68's eval-build bisheng failure (structurally different fallback
  codes) and any planner-difference variants.
- Verification: g3c_fallback_selftest 4/4 (forced-failure monkey-patches:
  g2stack→prefill_expert, both-experts→per_head_bm64, +per-head-BM64→BM32
  with correct S_padded, dense_expert→dense_auto — all outputs match
  golden); D512 causal/non-causal + D256-noncausal in g123_smoke PASS.

**[G4 / cases 84-100 — rc=-11 cascade]**
- Root cause: the non-aligned d2d→launch race above (repeated raced calls;
  see G4+G2-case67).  Additionally observed locally: two long-lived
  background verification processes died silently during the session
  (smoke suite after 4/14 cases; fallback selftest chain) — not
  reproducible on rerun (same suites then completed 14/14 and 4/4), no
  OOM/cgroup kill traces; consistent with the environmental-instability
  diagnosis from Attempt 1.  Within the cann-bench harness itself: zero
  crashes across two full 51-case runs.
- Verification: g4_stress_100 5/5 (100x repeat on the ~364us-class
  non-aligned + special-value combos — bitwise-stable post-fix).

### Local cann-bench evaluation (the core deliverable)

Submission: /tmp/opencode/gqa_submission (cann_bench wheel, built from the
fixed gqa.py).  Variant task set: /tmp/opencode/gqa_variant_tasks/level4/gqa
(20 public + 31 variants: non-aligned S/S_kv (base-4 / odd / 2^n±1),
D=512, value_range [.nan,.nan] / [-.inf,.inf] / [-65504,65504] / [0,0] /
[-0.1,0.1] / [-1,2], causal S>S_kv incl. mixed-block).

Run (repo $BENCH = .agents/skills/cann-bench):
  ./scripts/run_evaluation.sh --bench-name cann \
      --task-dir /tmp/opencode/gqa_variant_tasks --operator GQA \
      --source-dir /tmp/opencode/gqa_submission --no-perf --device-id 0

Result: **51/51 PASS, 0 failed, 0 crashed, EVAL_EXIT=0** (definitive run,
all fixes incl. the G4 sync; report:
.agents/skills/cann-bench/reports/cann_correctness_eval_20260910_190213.json
+ .md — summary: pass_rate 1.0, genuine_pass_rate 1.0, cascade_cases 0;
an earlier 51/51 run at 18:27 pre-sync-fix confirms single-call accuracy).
mere range 0 .. 8.4e-4 (thresholds 9.8e-4 fp16 / 7.8e-3 bf16); special-value
cases largely mere=0 (NaN-row/one-hot exact matches).  Note: the report
HTML generation crashes on a task-dir layout cosmetic bug (rel_path '.'
when the operator dir is passed directly as --task-dir — pass the PARENT
dir; JSON/MD reports are written before that stage) and variant cases
have no baseline perf anchors (expected, accuracy-only run).

### Performance spotcheck (3 runs, baseline = round-2 eval code)

main_kernel (factory _run): case8-fp16 1.003/1.001/1.005, case10-bf16
1.002/1.000/1.003 — no regression (the W5 reorder is op-count neutral;
the G4 syncs are on the padded path only, aligned paths byte-identical).
e2e oscillates +/-7% bidirectionally (host noise, same band as Attempt 1).

- error_summary: none (all local suites pass)
- design_error_reason: none
- next_hint: submit for eval re-run.  Remaining known boundary: dtype-TRUE
  bf16 full-range (out of the eval variant set).  If the eval still shows
  rc=-11 cascades, compare the eval harness call pattern against the
  local stress (the race fix covers the d2d→launch edge; a harness-side
  stream switch would need the same treatment).

## Attempt 1 — 2026-09-10 (mode: precision_fix, eval-fix task)
- mode: precision_fix (CANN-Bench online eval failure repair, not a fresh impl)
- classification: precision_pass (all local verification suites pass; see per-fix status)
- fail_category: none (target: 4 failure classes A1/A2/B/C from eval 20260909_102400)
- test_level: all (F1-F6 verification suites in eval_fix_verify/)
- coverage: F4 6/6 · F3 9/9 · F1/F2 10/10 · F6 accuracy 132/132 · F5 compile enum 656 g2 + 58 per-head · F6 seq-80 (below) · perf spotcheck (below)
- boundary_warnings: none
- changes: examples/gqa/gqa.py only (F1-F5 below); test scripts under examples/gqa/eval_fix_verify/
- rollback: no
- backup_path: examples/gqa/history_version/gqa_impl_eval_fix_attempt1.py
- instrumentation_cleaned: n/a (no kernel-side debug instrumentation used; all
  diagnostics were host-side monkey-patches in eval_fix_verify/ scripts)

### Root causes → fixes

**[F4 / class B, case 67 — causal S>S_kv fully-masked rows → NaN]**
- Root cause: causal-skip blocks keep sumexp == 0 → final normalize 0/0 = NaN;
  additionally the g2 kernel's W5 max-after-mask CANCELS the additive -2^30
  mask in the max shift for fully-masked rows inside non-skipped (mixed)
  blocks → those rows output an unmasked softmax (golden F217 requires 0).
- Fix: (a) `T.tile.add(sumexp, sumexp, 1e-30)` before every final normalize
  broadcast (10 sites: _gqa_kernel_expert, _gqa_kernel_ggroup,
  _gqa_kernel_prefill_g2 x2, _gqa_kernel_dense, _gqa_kernel_dense_expert,
  _gqa_kernel_dense_expert_d256, _gqa_kernel_prefill_expert,
  _gqa_kernel_prefill_g2stack, _gqa_kernel developer-legacy).  0/1e-30 == 0;
  normal rows have sumexp >= exp(0) = 1 (unchanged in fp32).
  (b) [F4b] g2 kernel: clamp running max at 0 (`T.tile.max(m_i, m_i, 0.0)`
  after the merged max, both segments) — shift-invariance keeps the softmax
  identical for normal rows while masked columns exp(-2^30 - 0) underflow to
  exactly 0, so mixed-block fully-masked rows also converge to sumexp == 0.
- Verification (f4_causal_fullmask.py): repro B2 S256 Skv128 causal fp16/bf16
  (full-skip blocks), B2 S132 Skv128 (mixed block, clamp path), per-head G=1,
  non-causal sanity — 6/6 PASS; masked rows exactly 0.0, no NaN/Inf, visible
  regions within tolerance. f4b_clamp_isolation.py: visible-region diffs are
  bit-identical to baseline (7.3e-4 pre-existing fp16 rounding), clamp effect
  on visible rows = 1 fp16 ULP.

**[F1 / class A2 cases 71/72 — wrapper mask NPU-zeros]**
- Root cause: gqa() wrapper built the non-causal non-aligned mask with
  torch.zeros(..., device=npu) → ZerosLike binary missing on eval SoC (561103).
- Fix: CPU torch.zeros + single .to(q.device) H2D (same pattern as the
  production-proven _build_causal_mask).
- Verification: f12_op_guards.py guards torch.zeros(device=npu) — no trip.

**[F2 / class A2 cases 73/81 + decode variants — padding NPU-zeros/cat]**
- Root cause: _prefill_run_padded (Q/K/V padding) and decode _run (K/V
  padding) used torch.cat([t, torch.zeros(..., device=npu)]) → ZerosLike +
  ConcatD / Cat_1_SliceAiCore binaries missing on eval SoC.
- Fix: new `_pad_along(t, dim, target_len)` helper — torch.full (FillScalar)
  + slice assignment (ViewCopy), the eval-verified op pattern. Replaced at
  all 5 sites (prefill Q/K/V + decode K/V).
- Verification: f12_op_guards.py guards torch.cat on NPU tensors — no trip;
  a2-73/a2-81/decode-variant shapes match golden.

**[F3 / class A1 cases 31/50/56/57 — aclnn Transpose fallback]**
- Root cause: transpose_qkv returned None when S / S_kv are not divisible by
  a shared whole-block bm → wrapper fell back to q.transpose(1,2).contiguous()
  → aclnn Transpose binary missing on eval SoC (561103).
- Fix: tail mode in _transpose_qkv (new `tail=False` param; part of the host
  cache key): per-row guarded copies — the exact _transpose_v1 pattern
  (scalar runtime row read + runtime-base [1, R] write) — through a new
  unconditional [1, R] ub_row staging tile; 3D (B*rows, heads, R) input views
  in tail mode. Dispatch: aligned shapes keep the whole-block path
  byte-for-byte; non-divisible shapes route to tail with fixed bm=128
  (universal q_v1=False fallback; plus a q_v1=True + KV-tail path for
  S<256/Nq>64 shapes whose Skv is non-divisible).
- Verification (f3_transpose_tail.py): 9/9 bit-exact vs
  .transpose(1,2).contiguous() — V2 tail (250/246, 1020/1020, 124/252),
  V1+KV tail (128/400 D256), Nq-nondiv (Nq=192), aligned unchanged (V2, V1,
  V1+KV). transpose_qkv no longer returns None for fp16/bf16, D*2%32==0, S>4.

**[F5 / class C, case 68 — g2 bisheng compile failure]**
- Reproduction attempt: f5_g2_compile_enum.py — 656 g2 parameter combinations
  on the LOCAL CANN 9.2 toolchain (phases 1-5: N_kv 1/2/4/8 + G-odd 24/48,
  S 5..128 + 132/260/516/1028, S_kv 128..1028 including odd kv_iters 3/5/9/17,
  odd q_blocks 3/5/9/17, causal S>S_kv incl. tiny instances, fp16/bf16,
  causal/non-causal) — ZERO compile failures. The eval failure (CANN 9.1.0 /
  Ascend910_9362) does NOT reproduce locally (CANN version difference
  suspected). An earlier phase-4 log showing 123 "failures" was an artifact
  of editing gqa.py while the enumeration process was running
  (inspect.getsource re-reads the file at trace time); the clean re-run is
  192/192.
- Fix per contingency plan: dispatch hardening — the g2 branch in gqa_expert
  is disabled via module constant `_ENABLE_G2_PREFILL = False`; every shape
  it served (odd kv_iters, causal S_kv < S, tiny instance counts — all
  non-bench-verified non-aligned variants; the aligned bench cases never
  reach it) now routes to the per-head _gqa_kernel_expert, which compiles on
  ALL enumerated combinations (phase 6: 58/58) and is the longest-production
  kernel. Cost: ~1.5-2x KV GM traffic on those non-aligned shapes only.
- Note: g2's W5 mask-cancellation defect was independently fixed by [F4b],
  so the kernel is retained (dormant) for future investigation.

**[Class D, cases 82-100 — cascading SIGSEGV]**: environmental per the
orchestrator's analysis (same shape passed in another eval process); not
addressed in code. The sequential-80 simulation below checks local
process-level robustness.

### F6 verification summary (all in eval_fix_verify/, pip tilelang, no set_env.sh)

| suite | result |
|---|---|
| f4_causal_fullmask.py (class B repro + mixed-block + per-head + non-causal) | 6/6 PASS |
| f3_transpose_tail.py (class A1 shapes, bit-exact) | 9/9 PASS |
| f12_op_guards.py (classes A1/A2 shapes; zeros/cat/permuted-contiguous guards) | 10/10 PASS |
| f6_accuracy_regression.py (20 base x fp16/bf16 x {base, S-4, Skv-4, both-4} vs golden) | 132/132 PASS |
| f5_g2_compile_enum.py phases 1-5 (g2 compile) | 656/656 OK |
| f5_g2_compile_enum.py phase 6 (per-head fallback compile) | 58/58 OK |
| f6_bf16_ab.py (baseline-vs-fixed on the 5 marginal bf16 cases) | see notes |

f6_bf16_ab.py notes: c3/c10/c19 (causal S>S_kv variants) were catastrophically
wrong in the baseline (max_diff 0.18-0.19 / NaN; 123k-247k tolerance
violations) and are fixed (max 5.9e-3 = 1-2 bf16 ULP, 0 violations at the
task bf16 tolerance atol=2e-2). c5/c6 are bit-identical to baseline (1-2 ULP
bf16 rounding, pre-existing) — the initial f6 tolerance (atol=4e-3) was
tighter than the task spec (2e-2) and was adjusted.

Guard-scope note (f12): `.contiguous()` on a crop-only view dispatches
aclnnInplaceCopy → the ViewCopy device kernel — the SAME binary the
eval-verified slice-assignment pattern uses (verified in libtorch_npu.so
symbols; only PERMUTED layouts route to the missing aclnnTranspose). The
f12 guard therefore flags only permuted materializations.

(sequential-80 and perf spotcheck results appended below when complete)

### F6 sequential-80 simulation (f6_sequential_80.py)

80/80 clean (56s): 20 base shapes x {fp16, bf16} x {aligned, non-aligned
(S-4/S_kv-4)} executed sequentially in ONE process — no exceptions, no
NaN/Inf, causal fully-masked rows exactly 0. The eval's D-class cascading
SIGSEGV does not reproduce locally (environmental, as diagnosed).

### F6 performance spotcheck (f6_perf_spotcheck.py, 3 runs)

main_kernel (factory _run, BHSD inputs — the metric that maps to the eval's
kernel timing):

| case | run1 | run2 | run3 | verdict |
|---|---|---|---|---|
| case8-fp16 (B4 S1024 causal, g2stack)  | 0.995 | 0.998 | 0.995 | no regression |
| case10-bf16 (B4 S512 Nq128 causal, g2stack) | 1.002 | 1.001 | 0.998 | no regression |

Absolute main_kernel us (run1/run2/run3): case8 baseline 756.2/753.1/749.4,
fixed 752.1/751.5/745.9; case10 baseline 984.2/991.2/997.6, fixed
986.4/991.8/995.9.  Kernel-level delta <= 0.2% — the [F4] ops (one
T.tile.max per KV iter in g2-class kernels is dormant here; g2stack adds
one [half,1] T.tile.add per head at normalize) are negligible.

End-to-end gqa() ratios oscillate +/-5% bidirectionally across runs
(case8 e2e: 0.951/1.007/1.055; case10 e2e: 1.048/0.961/0.916) — host-side
launch/clock noise, not a systematic regression; the aligned e2e code path
is byte-identical to baseline except the dead [1, R] ub_row alloc in the
transpose kernel (no data dependence).

- error_summary: none (all local suites pass)
- design_error_reason: none
- next_hint: submit for eval re-run; if case 68 still fails to compile on the
  eval SoC, the per-head reroute already avoids _gqa_kernel_prefill_g2
  entirely — next suspect would be the transpose tail kernel (also new),
  which case 68's shape class does not exercise (its transpose is aligned).

## Attempt 3 — 2026-09-11 (mode: precision_fix, eval-fix task: cann_final_eval_20260910_173445, 56/80)
- mode: precision_fix (CANN-Bench online eval failure repair, round 3 — continuation
  of an interrupted session; H1 already landed by the prior session)
- classification: in_progress (updated per-task below as H2-H5 complete)
- fail_category: none yet (targets: H1 reduce clear / H2 padding-in-kernel /
  H3 case67 matrix / H4 crash stress / H5 local full eval)

### Progress log (appended incrementally to survive interruption)
- [x] H1 DONE (prior session, verified by orchestrator): all 20+2 T.reduce_max/
      T.reduce_sum sites now pass clear=False (semantics: reduce into pre-zeroed
      dst without re-clearing — avoids the eval-build clear=True CUDART_INF
      undefined-binary issue on gqa_68). Orchestrator independently re-ran
      f4_causal_fullmask.py: 6/6 PASS, no numeric regression.
- [x] Backup DONE (prior session): examples/gqa/history_version/gqa_impl_eval_fix_attempt3.py
      (pre-H2 state; H2 changes will be diffed against this for bit-exact checks).
- [ ] H2 padding-in-kernel: IN PROGRESS
- [ ] H3 case67 matrix: TODO
- [ ] H4 crash stress: TODO
- [ ] H5 local cann-bench full eval: TODO
- [x] H2 CODE DONE (gqa.py): padding moved into self-written kernels.
      Design points:
      * `_transpose_qkv` gains Sq_out/Skv_out (>= real S/S_kv): outputs come
        out BLOCK-ALIGNED PADDED BHSD; pad rows zero-filled in-kernel
        (T.tile.fill on the staging tile — branch-safe, existing pattern).
        Pad code guarded by Python-level pad_q/pad_kv constants so NO-PAD
        traces are byte-identical to pre-H2.  Whole-block segments cannot
        write pad rows -> host dispatch routes any pad-needing segment to a
        per-row (tail / q_v1) mode.  Cache keys carry (Sq_out, Skv_out).
      * NEW `_pad_kv` 2-in-1 decode dim-1 pad kernel ([B,S_kv,Nkv,D] ->
        [B,S_kv_padded,Nkv,D] for K and V in ONE launch, valid rows staged
        through a [N,R] ub + pad rows T.tile.fill) replaces the two
        _pad_along slice-assigns in decode _run (3526/3527); pad kernel ->
        main kernel keeps a _npu_sync_tensor (H2d/G4 belt-and-braces).
      * OUTPUT path too (guard spec bans .contiguous() on the non-aligned
        fp16/bf16 path): `_transpose_v1` gains n_pad (reads the padded
        tensor with padded strides, writes only real-S rows);
        `transpose_out(s_real=...)`; `_prefill_run_padded`/prefill _run gain
        crop=True (factory callers unchanged); the bench wrapper passes
        crop=False + transpose_out(s_real=S) on padded shapes — the crop
        materialization (fp16/bf16 ViewCopy d2d, 561103) is gone.
      * per-head ladder entries now compile with the UNIFIED S_padded
        (ceil(S/64)*64) instead of per-bm ceil(S/bm)*bm — must equal the
        transpose's pad length; extra rows are zeros, cropped away.
      * _prefill_run_padded _pad_along calls KEPT as factory-interface
        no-op-checked fallbacks (bench path: transpose already padded ->
        `target_len <= orig` returns t unchanged; zero device ops —
        deviation from "delete" instruction, documented: deleting would
        break factory callers passing real-S BHSD tensors).
      * developer legacy path (4384-4387) keeps _pad_along: not part of the
        eval harness surface, zero-risk choice.
      * fp32 mask torch.full + slice-assign retained (verified available on
        the eval SoC both rounds).
- [x] H2 VERIFY (eval_fix_verify/h2_kernel_pad.py, log h2_kernel_pad.log):
      * (i) dispatch-hook guard 11/11 PASS — non-aligned fp16/bf16 bench
        path dispatches ZERO torch device-side copy/fill ops (slice
        __setitem__ / .contiguous() / full / zeros / cat / copy_ d2d all
        guarded; fp32 mask + CPU ops exempt). Covers the 5 round-3 failure
        classes + orchestrator H2 shapes (prefill/decode/nq128/d512).
      * (ii) numeric 11/11 PASS vs fp64 golden via the bench checker
        (compare_tensors): B2S124Skv252 fp16/bf16 x causal/non, row-pad-only,
        nq128 (q_v1+KV-tail), D512 (ladder unification), decode
        B60S1Skv2044 fp16 causal/non + bf16.
      * (iii) aligned bit-exact vs pre-H2 backup: IN PROGRESS (g2stack,
        q_v1+whole-KV, universal-tail aligned: BITEXACT so far).
- [~] H3 matrix (h3_matrix.py) launched: seed-0 sweep 180 cells
      ({nan,inf,fullrange} x {fp16,bf16} x {G1,G4,G16} x {causal,non} x
      {prefill S128/S124+Skv252, decode Skv2048/2044} + 36 small-shape
      probes), bench data_generator + bench compare_tensors, device 1.
      * (iii) aligned bit-exact 7/7 PASS vs the pre-H2 backup module
        (history_version/gqa_impl_eval_fix_attempt3.py — pre-H1; H1's
        reduce_max clear=False is provably bit-neutral since the pre-fill
        sentinel sits below every reachable max, so any diff isolates H2):
        whole-block V2 (c8-class g2stack), q_v1+whole-KV, universal-tail
        aligned, decode causal + dense_expert, D256 per-head, D512
        per-head-bm32 — ALL bit-identical (int16 bit views).
      => H2 DONE: the non-aligned fp16/bf16 bench path now dispatches ZERO
      torch device-side copy/fill ops (padding AND output-crop both moved
      into tilelang kernels); aligned paths bit-identical.
      NOTE (env): two concurrent long-running verify processes died
      silently mid-run (same environmental instability as Attempt 2's G4
      note) — rerun sequentially, one heavy process at a time, completed
      clean. Verification scripts are restartable per-part.
- [x] H3 seed-0 DONE (h3_matrix.log): 180/180 PASS, 0 FAIL, 0 RUNTIME-FAIL.
      Matrix: {nan, inf, fullrange} x {fp16, bf16} x {G1(Nkv32), G4(Nkv8),
      G16(Nkv1)} x {causal, non} x {prefill S128/Skv128 aligned +
      S124/Skv252 non-aligned; decode S2 Skv2048/2044, B16} = 144 cells
      + 36 small-shape probes (B2 S8/S32/S5, decode B2 S1 Skv2044 /
      S2 Skv132, causal S128>Skv96).  Inputs via the bench's OWN
      DataGenerator (imported; value semantics identical to the eval:
      nan=50% deterministic-random positions, inf=flat ends n=numel//20,
      fullrange=[-65504,65504]); comparison via the bench's compare_tensors
      (NaN-position equality is its first gate).  NO case-67 signature
      (NaN-position mismatch) reproduced.  Per the H3 protocol, the x10
      seeds rescan (seeds 1-10, same matrix) is now running.
- [x] H3 RESCAN DONE: seeds 1-10 (same 144-cell matrix, deterministic
      cell-indexed seeds) — 0 fail, 0 runtime-fail across all 10 seeds.
      TOTAL: 1620 cells (seed-0 180 incl. small-shape probes + 10x144)
      all PASS; the case-67 form (partial NaN-position mismatch) is NOT
      reproducible locally at any seed.
      Conclusion: consistent with the round-2 root-cause (the G4
      d2d->launch race on a special-value non-aligned case — an
      eval-machine manifestation).  H2 additionally REMOVES the race's
      trigger surface on that path: the fp16/bf16 pad slice-assigns and
      the output-crop .contiguous() (the raced torch d2d ops) are gone —
      the only pre-launch torch op left is the fp32 mask build, still
      covered by _npu_sync_tensor.  Recorded as "not reproducible
      locally; root cause addressed structurally by H2".
- [x] H4 DONE (h4_stress.log): 9/9 PASS.  Degraded fallback forms (per-head
      BM32 / BM16 via forced primary-factory compile failures — the
      g3c monkey-patch technique; decode AUTO dense via forced dense_expert
      failure) x non-aligned + special-value shapes (D128 fullrange/nan/inf,
      D256 fullrange, D512 fullrange, decode B60S2Skv2044 causal
      fullrange/nan), 100 repeats each: zero exceptions/crashes, iteration-0
      numerics vs fp64 golden OK (NaN-position match), bit-stable across
      all 100 repeats (int16 bit views).  The round-3 rc=-11 cascade form
      does not reproduce in any degraded configuration.
- [x] H5 PERF ISSUE FOUND + FIXED ([H1-R] clear=True restore): the first
      h5_perf_spotcheck run showed case8 main_kernel x1.798 / case10 x1.649
      vs the attempt2 baseline.  Isolation (alternating A/B in one process,
      fresh single-module processes, kernel-source md5 comparison):
      the slowdown came from H1's clear=False reduce_max — the LOCAL
      tilelang clear=false lowering (src/tl_templates/ascend/common.h)
      wraps the hardware AscendC::ReduceMax in SCALAR GetValue/SetValue
      backup+merge loops (~2x on the W5-heavy kernels), while clear=true
      is the single fast instruction.  The prior session's f6_perf_attempt3
      log ("no regression") predates H1.
      FIX [H1-R]: clear=True restored in ALL kernels (exact att3 kernel
      sources — g2stack kernel-source md5 verified identical to the
      pre-H1 backup) EXCEPT the dormant g2 fallback kernel (its 2 sites
      keep clear=False + the sentinel-safe m_init_floor): g2 is the kernel
      whose eval-build compile hit the undefined-CUDART_INF error
      (round-3 case 68), so keeping clear=False there preserves the
      case-68 coverage while every dispatched-on-the-eval kernel keeps the
      round-3-proven clear=True form.  Round-3 evidence: 56/80 cases
      compiled and ran with clear=True everywhere; ONLY case 68's chain
      failed (g2's CUDART_INF) — with g2 clear=False, case 68 can now
      compile through its last-resort candidate.
      VERIFIED: h5_perf_spotcheck rerun — case8 kernel 1.003, case10
      kernel 0.997 (<3% gate PASS); non-aligned e2e 0.966 (H2 in-kernel
      padding is FASTER than the old torch d2d path).
- [x] H4 RERUN on final code (h4_stress_final.log): 9/9 PASS (same combos,
      100x repeat each — zero crashes, numerics OK, bit-stable).
- [x] H2 RERUN on final code (h2_rerun.log): guard 11/11 + numeric 11/11 +
      bitexact 7/7 — ALL PASS.
- [x] FULL REGRESSION on final code (h5_regress_final.log): f4 6/6,
      f3 9/9, f12 10/10, g1 8/8, g3c 4/4, g2 24/24, g123 14/14.
- [x] g2 compile-check on final code: the dormant clear=False kernel
      compiles + runs correctly (B2S96Skv300 noncausal, B2S128Skv96 causal;
      nan=0).
- [x] H5 FINAL EVAL: 68/68 PASS, 0 failed, 0 crashed, cascade_cases=0,
      pass_rate=1.0, genuine_pass_rate=1.0.  Report:
      .agents/skills/cann-bench/reports/cann_correctness_eval_20260911_143057.json
      (+ .md).  Task set: 20 public + 31 round-2 variants + 17 H3-matrix
      additions (special values x dtypes x NON-ALIGNED small shapes:
      prefill S124/Skv252 nan/inf/fullrange fp16+bf16, G1-MHA nan, G16 inf,
      D512 nan, decode Skv2044 nan/inf/fullrange fp16+bf16, rowpad-only
      fullrange) = 68 cases.  mere range 0 .. 1.84e-3 (thresholds 9.8e-4
      fp16 / 7.8e-3 bf16 with the checker's value-region logic; the
      special-value cases are largely mere=0 — exact NaN-row/one-hot
      matches).
- [x] H5 PERF SPOTCHECK (final): case8 main_kernel 753.4us vs attempt2
      751.0us = 1.003; case10 983.1 vs 986.3 = 0.997 (<3% gate PASS);
      non-aligned B2S124Skv252 e2e 1364.5 vs 1413.2us = 0.966 (the H2
      in-kernel padding path is FASTER than the old torch-d2d padding).
- H3 note: the 1620-cell matrix + x10-seed rescan ran on the pre-H1-R
      code; the final code's kernel set is bit-identical to the pre-H1
      (round-3-evaluated) sources for every dispatched kernel except the
      dormant g2, and the H5 eval's 17 H3-matrix representative cases +
      g2_special_values 24/24 + h2 numeric 11/11 all re-verified the
      special-value x non-aligned space on the FINAL code.

### Attempt 3 final classification
- classification: precision_pass (all local gates green: H2 guard/numeric/
  bitexact, H3 matrix 1620 cells no-repro + eval representatives, H4 stress
  9/9, H5 68/68 zero-crash + perf <3% + full regression, f4 re-verified)
- fail_category: none
- rollback: no (the [H1-R] perf fix was a targeted in-place refinement of
  H1, not a rollback: clear=True restored everywhere except the dormant g2
  fallback — kernel sources md5-verified identical to the round-3-evaluated
  pre-H1 state for every dispatched kernel)
- next_hint: submit for the round-4 eval re-run.  Expected outcome vs the
  round-3 24 failures: 71/72/73/81/82 fixed by H2 (zero torch d2d ops on
  the non-aligned fp16/bf16 path); 84-100 crash cascade: trigger surface
  removed (the raced pad/crop d2d ops are gone) + H4 stress clean;
  67: not reproducible locally (1620 cells), race trigger removed; 68: g2
  clear=False compiles past the eval CUDART_INF.  Known boundary (out of
  eval scope): dtype-TRUE bf16 full range beyond [-65504,65504].

## Attempt 3 — 2026-09-12 (mode: test_supplement, directory-structure maintenance)
- mode: test_supplement (non-standard Stage-2 task: bring examples/gqa/ to the
  examples/ test-file convention; no kernel logic change)
- classification: precision_pass (test_gqa.py 19/19: L0 9/9 + L1 6/6 blocking
  PASS, Boundary 4/4 non-blocking PASS, "Test Passed!" + exit 0 on --level all)
- fail_category: none
- test_level: all (l0, l1, boundary each run standalone, then --level all)
- coverage: L0 9 (cases.csv c1/c2/c3/c4/c5/c7/c11/c16/c20 — both dtypes,
  prefill+decode dispatch, causal/non, D256, S<S_kv, explicit scale) ·
  L1 6 (f6 S-4/S_kv-4/both-4 tails incl. causal S>S_kv 4 masked rows,
  decode S_kv-4, MQA c18, MHA c19) · Boundary 4 (±8 large value,
  fp16 fullrange ±65504, sparse NaN/+Inf/-Inf propagation, causal
  S=128>S_kv=64 fully-masked rows)
- boundary_warnings: none
- changes:
  - NEW examples/gqa/test_gqa.py (480 lines): golden = replica of the
    cann-bench eval golden; L0/L1 comparison/tolerances/seeds replicated
    from eval_fix_verify/f6_accuracy_regression.py (max_diff values match
    the f6 log exactly, e.g. c1 fp16 4.883e-04, c20 bf16 3.906e-03);
    Boundary special-value semantics from g2_special_values.py (fp64 CPU
    oracle + structural NaN/Inf compare; fullrange bit-exact 0.000e+00).
  - examples/gqa/gqa.py: docstring line 79 ONLY — stale
    "See DESIGN.md / PERFORMANCE_OPTIMIZATION.md" (files do not exist)
    -> "See debug_log.md ... history_version/ ...". diff vs pre-edit
    backup (/tmp/opencode/gqa_docstring_before.py) = 79c79, 1 line;
    `python -c "import gqa"` ok. No other change to the eval deliverable.
- error_summary: none
- rollback: no
- instrumentation_cleaned: n/a (no kernel-side instrumentation; kernel file
  untouched except the docstring reference fix)
- next_hint: run `unset PYTHONPATH; export ACL_OP_INIT_MODE=1; python
  examples/gqa/test_gqa.py --level all` for regression after any future
  gqa.py change. NOTE: the in-repo tilelang is not built (no TVM) — the
  repo root must stay off PYTHONPATH so the installed site-packages
  tilelang is used (same invocation as the f6/g2 suites).

## Attempt 4 — 2026-09-16 (mode: precision_fix / expert-kernel scatter deadlock debug)
- mode: precision_fix (attempt_index 4; actual task: fix the 4 `KNOWN DEAD BRANCH`
  output-scatter sites in _gqa_kernel_dense_expert{,_d256} so segs>2 decode
  shapes compile, launch without deadlock, and truly dispatch the tuned expert
  kernel instead of silently falling back to dense_auto)
- classification: precision_pass (deadlock FIXED via direction A — trace-time
  static fixed-slot unroll; fallback plan D NOT used)
- fail_category: none
- test_level: all (test_gqa.py --level all x3 + dedicated scatter matrices)
- backup_path: examples/gqa/history_version/gqa_impl_s2_attempt4.py
  (md5 ea3281e13f2b93a8d53ef06633833f8d == pre-fix verified baseline)
- rollback: no
- instrumentation_cleaned: yes (no kernel instrumentation was ever added;
  grep T.printf/T.dump_tensor/pdb in gqa.py = 0)

### Root cause (established from TVMScript parser source, not guesswork)
- tvm/script/parser/tir/parser.py `visit_if`: `isinstance(condition, bool)` ->
  Python-bool `if` conditions are folded AT TRACE TIME (only the taken branch
  is traced; no IfThenElse emitted).  This is why `if len(segs) >= k:` frames
  work and why c16's empty vid1 plan never IndexErrors on `segs[0]`.
- Same file `visit_for`: the iterator MUST evaluate to a `T.frame.ForFrame`
  (range/T.serial/T.grid/T.unroll/...).  `for s in range(S)` is therefore a
  TIR RUNTIME loop (s = Var) — it can never unroll at trace time, and a raw
  Python-list iterator is a parse error.  Consequences:
  * compile error (attempts 1/3): `s_f` is bound in the sibling `>=1` frame —
    parser var_table frames pop at frame exit -> "Undefined variable: s_f".
  * attempt-2 error: `segs[si]` indexes a Python list with a TIR Var.
  * attempt-3 HANG: the runtime `for s` + `if s > s_f / s < s_l` made the MTE3
    output scatters runtime-conditional inside the vid branch of the manual
    flag pipeline (SIG_S_HALF); launch deadlocked.  The exact device-side
    mechanism was NOT isolated to instruction level — the fix AVOIDS the
    pattern entirely instead (see below).

### Fix (direction A: fully static trace-time scatter)
- Dispatch guard is 1<=S<=4 (decode) => len(segs) <= S <= 4 => at most TWO
  middle segments (segs[1:-1]) per vid.  Replaced the dead `for s in range(S)`
  branch with two fixed trace-time slots per site:
    `if len(segs) > 2:` unpack segs[1] + unconditional static-slice T.copy;
    `if len(segs) > 3:` unpack segs[2] + unconditional static-slice T.copy.
  Python-bool conditions fold at parse time, so the emitted TIR contains only
  flat unconditional UB->GM MTE3 copies — structurally IDENTICAL to the
  already-proven segs[0]/segs[-1] copies (no new runtime control flow, flag
  handshake untouched).  Exhaustive enumeration of all 217 expert-eligible
  (G,S,D) combos: max len(segs)=4, slot-coverage violations=0 (25 triggering
  (D,S,G,G_inner,M_tile) tuples; note the trigger surface is wider than
  "G_inner<=2" — e.g. G_inner=3/5/13 partial-tail plans — the plan-driven
  slots cover them all by construction).
- Safety net: factory-level `assert S <= 4` in BOTH expert kernels (fires at
  compile time -> _resolve_kernel_chain falls back to dense_auto loudly;
  never silently drops rows if the decode guard is ever relaxed).

### Changes (gqa.py only; 140 diff lines; md5 ea3281e1 -> ec21d36f)
- 4 scatter sites: dense_expert vid==0/vid==1 (acc_s_half), dense_expert_d256
  vid==0/vid==1 (acc_o_half) — dead branch -> static slots + updated notes.
- 2 factory blocks: scatter-plan comment refreshed (stale "runtime s-loop"
  text) + `assert S <= 4`.
- NO function-signature changes; NO other kernel bodies touched; NO host
  dispatch changes needed (chain already tries dense_expert first).
- ruff 0.16.7: `format --check` exit 0, `check` exit 0 (no noqa needed —
  SIM102 does not fire on the new frames; the old noqa:SIM102 sites are gone).

### Verification (all real command outputs; every launch timeout-guarded;
### device Health=OK / AICore=0% / no stray procs checked between phases)
- STATIC (compile-only, zero launches):
  * 4 triggering shapes (D128 S4/G2, D128 S3/G2 causal bf16, D128 S4/G1 MHA
    1-row segs, D256 S4/G2) compile OK on the new code; the BASELINE fails
    all of them with `Undefined variable: s_f` (defect reproduced, stderr).
  * segs<=2 byte-identity: kernel sources old==new md5-identical for
    S2_G4_segs2 (bd8b9c15...), S4_G16_segs2/c17-like (3c966752...),
    D256_S1_segs1 (ab5d536b...) => identical binaries for every working
    shape class.
  * Generated-code inspection (att4_static/ks_*.cc): scatters are flat
    unconditional `copy_ub_to_gm` inside the pre-existing `if (vid==n)`
    branch only; ONE WaitFlag<MTE3_V>(0) ... SetFlag<MTE3_V>(0) sandwich per
    task iter + init set + destroy wait — pairing identical to baseline.
- LAUNCH:
  * Smoke (attempt-3 hang family, B4 S4 Skv512 Nq16 Nkv8 D128): returned,
    dense_expert dispatched, max_diff=6.104e-05, viol=0.
  * Trigger matrix v2 (13 segs>2 shapes incl. partial-tail G13, 1-row MHA,
    unaligned Skv, both dtypes, causal/non): 13/13 OK, EXIT_CODE=0 — all
    dispatch dense_expert/dense_expert_d256; golden viol=0, or bitwise==
    baseline for the unaligned seed (see pre-existing note below).
  * verify_scatter (official, OLD corrected to gqa_ci_clean_baseline.py):
    SUMMARY 19/19 OK, EXIT_CODE=0.  A-group 11/11 (expert dispatched +
    accurate; D128 A-cases bitwise_old_new=True vs dense_auto).  B-group
    8/8 bitwise + same dispatch names.
  * A/B bitexact (14 shapes, baseline vs patched, dispatch names asserted
    via _KERNEL_BUILD_CACHE[key][2]): 14/14 BITEXACT max_diff=0.000e+00,
    names identical — covers g2stack / prefill_expert / per-head /
    dense_expert (segs=1, segs=2, 1-row) / dense_expert_d256 / dense_auto /
    MQA (prefill+decode) / MHA (prefill+decode) / non-aligned pad (prefill
    QKV-pad + decode pad_kv).  EXIT_CODE=0.
  * FULL REGRESSION test_gqa.py --level all: run 3x (2 background + 1
    synchronous).  Final synchronous run: FINAL_EXIT_CODE=0, "Test Passed!",
    15 [PRECISION_PASS] + 0 [PRECISION_FAIL] + 4 [BOUNDARY_PASS] +
    0 [BOUNDARY_WARN], 0 fallback lines (no failed compiles anywhere).
- PRE-EXISTING, OUT OF SCOPE (documented, NOT introduced by this fix):
  unaligned-Skv decode fp16 with certain seeds shows a golden tail of
  ~2.7-2.8e-3 (> strict atol=1e-3+rtol).  Proof it is pre-existing: the
  UNTOUCHED baseline dense_expert path shows it on a segs<=2 unaligned shape
  (probe P1: viol=202, byte-identical codegen old==new), and on the
  triggering unaligned shape old(dense_auto) and new(dense_expert) outputs
  are BITWISE IDENTICAL (probe P2).  It is a golden-fp16 (torch fp16 matmul
  rounding) vs kernel-fp32-accumulate data-dependent artifact; no official
  suite case (bench 20 / test_gqa 19) hits such a seed.

### Residual risks
1. The attempt-3 hang mechanism (runtime-conditional MTE3 inside the manual
   flag pipeline) was avoided, not root-caused to instruction level.  DO NOT
   reintroduce runtime-conditional MTE3 statements in the expert kernels'
   flag regions; keep scatters unconditional/static-slice.
2. The two-slot static pattern is tied to S<=4; protected by the factory
   assert (loud compile-time fallback to dense_auto).  Relaxing the decode
   guard requires adding slots (or a generic mechanism) first.
3. Trigger shapes were verified for accuracy/dispatch/no-hang only; their
   PERF gain over dense_auto was not measured here (Stage-3 scope).
4. New-module md5 ec21d36f1af4d81ca32aad9bd8ab1d26; evidence scripts/logs:
   /tmp/opencode/att4_{static_check,trigger_matrix2,verify_scatter,
   ab_bitexact,probe_unaligned,followup_bitwise}.py and matching *.log;
   generated sources: /tmp/opencode/att4_static/ks_*.cc.
- next_hint: orchestrator may re-run the cann-bench eval; expect the 50
  segs>2 (D,S,G) shapes to dispatch dense_expert/dense_expert_d256 with no
  first-shape failed-compile penalty.  For perf validation of the newly
  dispatched shapes, use the Stage-3 flow.

## Attempt 5 — 2026-09-16 (mode: precision_fix / decode non-causal column-pad mask defect)
- mode: precision_fix (correctness defect, orchestrator-located; user-approved
  immediate fix).  attempt_index=5, design_revision_count=0.
- classification: precision_pass
- fail_category: none (host-side mask-construction defect; NOT a kernel bug —
  no `_gqa_kernel_*` body touched)
- backup_path: examples/gqa/history_version/gqa_impl_s2_attempt5.py
  (md5 ec21d36f1af4d81ca32aad9bd8ab1d26 == the pre-fix delivered file; A/B
  baseline; orchestrator copies /tmp/opencode/gqa_padmask_baseline.py +
  test_gqa_padmask_baseline.py, same md5)
- instrumentation_cleaned: n/a (no T.printf/T.dump_tensor inserted — the defect
  is host-side Python mask construction, localized by reading + A/B, not by
  in-kernel instrumentation)
- rollback: no (fix kept; A/B/regression all green)

### Root cause
`gqa_expert` decode `_run`, the NON-CAUSAL column-padding branch (gqa.py:3404
`else:`, mask_cpu build at ~3407) built an ALL-ZEROS replicated mask
`[B, M_tile, S_kv_padded]` and never masked the zero-filled K/V pad columns
`[S_kv, S_kv_padded)`.  Unmasked pad columns score 0 (K=0 rows), so their
`exp(0 - shift)` weight is added to the softmax denominator -> every output
element is systematically shrunk by ~ pad_ratio.  The CAUSAL branch
(gqa.py:3398-3399) does mask them (`mask_cpu[:, row_idx, S_kv:] = _BIG_NEG`),
which is why causal decode was always correct.

Pre-existing (delivered 20/20 file == CI baseline == attempt-4 are bitwise
identical on all these shapes; independent of the last few attempts).  Never
exposed because: all 20 bench decode cases use 128-aligned S_kv (2048/512/
1024 -> pad=0); test_gqa.py L1's only decode tail case was Skv=2044 (pad 0.2%
-> 9.2e-5, inside tolerance); the deviation scales with pad ratio, so small-pad
cases hid it.

### Fix (gqa.py, decode non-causal branch, 1 logical change / +2 code lines)
Added, mirroring the causal branch's pad-column handling:
```
if S_kv < S_kv_padded:
    mask_cpu[:, :, S_kv:] = _BIG_NEG
```
Applied to ALL M_tile rows (single vectorized CPU slice; pad rows included is
harmless — they are discarded by the scatter plan, see audit item 6).
Constraints preserved: mask still built on CPU (torch.zeros + CPU slice-assign,
host allocator — no ZerosLike / no NPU binary, avoids 561103) then ONE
`.to(Q.device)` H2D (still exactly 1 aclrtMemcpy on the masked path — anti-cheat
budget intact).  No NPU-side torch copy/fill introduced.  Kernel bodies, the
attempt-4 static scatter slots, the `assert S <= 4`, and the 4 `# noqa: SIM102`
directives all untouched.

### Mask-construction-point audit (req 3 — independent re-review)
1. gqa.py:3079 `_prefill_run_padded` Mask_full = torch.full(..., _BIG_NEG) then
   `Mask_full[:, :S, :S_kv] = Mask` — CORRECT.  Whole padded mask defaults to
   _BIG_NEG; only the real [0:S, 0:S_kv] region is overwritten, so every pad
   row/col is masked.  Built only when use_mask=True.
2. gqa.py:3382-3399 decode CAUSAL branch — CORRECT.  Sets causal upper triangle
   AND pad columns (`mask_cpu[:, row_idx, S_kv:] = _BIG_NEG` when S_kv<padded).
   This is the reference the fix mirrors.
3. gqa.py:3404-3413 decode NON-CAUSAL branch — WAS THE DEFECT, now FIXED.
4. gqa.py:4300 `_build_causal_mask` — CORRECT.  Builds [B,S,S_kv] over the REAL
   S_kv (no padding here); prefill pad columns are masked downstream by
   Mask_full's _BIG_NEG fill (item 1).  Decode never calls it (decode rebuilds
   the replicated mask inside `_run`; wrapper passes _dummy_mask for decode).
5. gqa.py:4442 wrapper non-causal non-aligned PREFILL `torch.zeros(B,S,S_kv)` —
   CORRECT.  Built at the REAL S_kv (all-zero = visible), then `_prefill_run_padded`
   copies it into Mask_full[:, :S, :S_kv] with the pad region left _BIG_NEG
   (item 1).  Only reached on the prefill (not decode_direct) path.
6. decode ROW padding (M_tile > S*G_inner) needs NO mask — CONFIRMED at source:
   the dense_auto output scatter (gqa.py:961-989) writes `Output[bz, s, ...]`
   ONLY for `s in range(S)`, and the expert kernel's factory-level scatter_plan
   (gqa.py:1078-1102) enumerates only real s rows; pad accumulator rows
   [S*G_inner, M_tile) are never scattered back.  Empirically corroborated by
   A7 (S=3, 4 pad rows) and A12 (MHA S=1 G=1, 15 pad rows) both passing at
   tolviol=0.

Conclusion: item 3 was the ONLY defect; items 1/2/4/5 correct as-is; row pad
(item 6) is mask-free by construction.

### Verification (gates A-G, all real runs, unset PYTHONPATH; ACL_OP_INIT_MODE=1)
Harness: /tmp/opencode/att5_smoke.py + att5_verify.py (--group B|A|F); base =
gqa_padmask_baseline.py, new = examples/gqa/gqa.py.  Kernel names read from
_KERNEL_BUILD_CACHE[key][2] (authoritative), not stdout.

- SMOKE (Skv=100 non-causal, seed 4242+Skv == orchestrator probe):
  base maxdiff=5.518e-02 tolviol=7702 -> new maxdiff=1.221e-04 tolviol=0;
  base!=new bitwise (fix took effect); kernel dense_auto==dense_auto.  Reproduces
  the orchestrator's table exactly and clears it.

- GATE A (group A, 12/12 OK) — expected-to-change shapes, non-causal unaligned
  decode, base -> new (maxdiff / tolviol):
    A1 S1 Skv100 pad21.9% : 5.249e-02/7638  -> 1.221e-04/0   (dense_auto)
    A2 S1 Skv132 pad48.4% : 1.301e-01/7931  -> 1.221e-04/0   (dense_auto)
    A3 S1 Skv300 pad21.9% : 2.673e-02/7270  -> 6.104e-05/0   (dense_auto)
    A4 S1 Skv380 pad1.0%  : 1.099e-03/0     -> 6.104e-05/0   (dense_auto)
    A5 S1 Skv2044 pad0.2% : 9.155e-05/0     -> 3.052e-05/0   (dense_auto)
    A6 S2 Skv300 bf16     : 2.930e-02/159   -> 9.766e-04/0   (dense_auto)
    A7 S3 Skv300 rowpad   : 2.692e-02/21784 -> 6.104e-05/0   (dense_auto)
    A8 S4 Skv300          : 3.406e-02/58226 -> 1.221e-04/0   (dense_auto)
    A9 D256 S1 Skv300     : 3.357e-02/29138 -> 1.221e-04/0   (dense_expert_d256)
    A10 expert S1 Skv400  : 2.966e-02/56821 -> 6.104e-05/0   (dense_expert)
    A11 MQA S1 Skv132     : 7.977e-02/7938  -> 1.221e-04/0   (dense_auto)
    A12 MHA S1 Skv132     : 1.116e-01/1988  -> 1.221e-04/0   (dense_auto)
  All pad>=20% cases: base tolviol>0 -> new tolviol=0; fp16 new maxdiff ~1e-4
  (<= cap 2e-3), bf16 new 9.766e-4 (<= cap 2e-2).  Fix effective across
  dense_auto / dense_expert / dense_expert_d256 (host-side mask is shared by
  the whole decode dispatch chain).  All changed=True (base!=new bitwise).

- GATE B + C (group B, 24/24 OK) — must be BITWISE identical + same kernel name
  (torch.equal on int16 view; kernel name from _KERNEL_BUILD_CACHE):
    aligned decode: dense_expert (c11 S1/2048 nc, c16 S2/512 c, l0 S2/2048 c
      bf16, G12 S1/2048 nc bf16, S4/512 nc + c, MHA S1/512 nc), dense_expert_d256
      (D256 S1/512 nc, D256 S1/1024 c), dense_auto (MQA S2/1024 c)
    causal NON-aligned decode: dense_auto (Skv 100 / 300 / 2044, S4/380 bf16,
      D256 Skv300)
    prefill (all forms): g2stack (c1 128, long 1024, tail 124x252 nc, MQA 128,
      both-tail 252 bf16, cross 128x256 nc bf16), per_head_bm64 (tail 128x124 c,
      D256 128 c), prefill_expert (MHA 128 c bf16)
  Every case: bitwise=True AND base kernel name == new kernel name AND viol_new=0.
  Gate C satisfied for ALL 6 kernel variants; A-group names also unchanged
  (host-side-only fix).  No silent downgrade to dense_auto anywhere.

- GATE D (ruff 0.16.7, ABSOLUTE path /home/developer/.local/bin/ruff):
  `ruff format --check gqa.py test_gqa.py` -> "2 files already formatted", exit 0;
  `ruff check gqa.py test_gqa.py` -> "All checks passed!", exit 0.  noqa:SIM102
  count still 4.

- GATE E (full regression): `unset PYTHONPATH; export ACL_OP_INIT_MODE=1;
  python examples/gqa/test_gqa.py --level all` (tilelang.disable_cache()) ->
  exit code 0 (captured to /tmp/opencode/att5_exit2.txt), "Test Passed!",
  18 [PRECISION_PASS] (9 L0 + 9 L1 incl. the 3 new decode pad cases) +
  6 [BOUNDARY_PASS] (incl. 2 new decode non-causal special-value cases) +
  0 [PRECISION_FAIL] + 0 [BOUNDARY_WARN] + 0 [gqa][fallback].
  New L1: l1_decode_skv100_noncausal_fp16 max_diff=2.441e-04,
  l1_decode_skv132_noncausal_fp16 max_diff=1.221e-04,
  l1_decode_skv132_noncausal_bf16 max_diff=9.766e-04; existing
  l1_decode_skv_tail_fp16 (Skv=2044) retained, max_diff=3.052e-05.

- GATE F (group F, 3/3 OK — special values, decode non-causal unaligned Skv=300):
  _BIG_NEG pad mask does NOT fight NaN/Inf/full-range:
    F1 fullrange (+-65504): vs fp64 oracle max_diff=0.000e+00, problems=[],
       nan_pos_eq=True, inf_pos_eq=True (base/new NaN+Inf positions identical)
    F2 sparse NaN(q)+ +Inf(v col50) + -Inf(v LAST real col 299): max_diff=
       1.221e-04, problems=[], nan=128 inf=8, positions match oracle + base
    F3 k-NaN col on 48%-pad shape (Skv=132): max_diff=1.221e-04, problems=[],
       nan=512, positions match oracle + base
  Also mirrored into test_gqa.py boundary level (cases 5): both PASS in gate E.

- GATE G (device discipline): every launch under `timeout`; group B/A/F ran
  62.7s/67.6s/1.8s, full regression ~4min — no hang.  No in-kernel
  instrumentation added (host-side fix), so no MTE3/MTE2 in expert flag regions
  touched (attempt-3 hang mechanism not re-triggered).  npu-smi Health OK
  throughout; no residual python/timeout procs after runs (verified via ps).
  Both cards idle at start (AICore 0%).

### test_gqa.py changes (LOCAL ONLY — PR submits gqa.py only; must still pass)
- L1: +3 decode non-causal large-pad cases (Skv=100 pad21.9% fp16, Skv=132
  pad48.4% fp16, Skv=132 pad48.4% bf16 S=2); existing Skv=2044 small-pad case
  retained; +docstring note explaining the blind spot.
- Boundary: +2 decode non-causal unaligned special-value cases (fullrange +
  sparse NaN/Inf), fp64 CPU oracle + structural compare (gate F mirrored in-suite).
- ruff-formatted (magic-trailing-comma expansion); format --check + check exit 0.

### Residual risks
1. The attempt-4 residual note (a data-dependent ~2.7-2.8e-3 golden-fp16 tail on
   SOME unaligned-Skv decode seeds, attributed to torch-fp16-matmul vs
   kernel-fp32-accumulate rounding) is a SEPARATE, still-open artifact and is
   NOT what this attempt fixed.  This attempt fixed the systematic pad-column
   mask defect (5e-2..1.3e-1, pad-ratio-proportional, thousands of tolviol).
   The new L1 seeds (Skv=100/132) do not hit the rounding tail (max_diff ~1-2e-4).
2. Fix is host-side only; if a future decode path ever feeds the kernel a mask
   built somewhere OTHER than `_run`'s use_mask branch, it must replicate the
   pad-column _BIG_NEG masking.  The 5 audited construction points are the
   complete current set.
3. Non-causal decode with pad now masks pad rows' pad columns too (all-rows
   slice) — unobservable (pad rows discarded), but differs cosmetically from the
   causal branch (which leaves pad rows all-zero).  No functional impact.
4. New-module md5 (post-fix gqa.py): see `md5sum examples/gqa/gqa.py`; backup of
   the pre-fix state preserved at history_version/gqa_impl_s2_attempt5.py
   (md5 ec21d36f...).  Evidence: /tmp/opencode/att5_{smoke,verify}.py,
   att5_test_all.log / att5_test_all2.log / att5_exit2.txt.
- next_hint: PR should carry ONLY gqa.py (the 2-line + comment fix).  Orchestrator
  may re-run the cann-bench eval; non-causal unaligned-Skv decode shapes that
  previously would have shrunk toward zero now match the golden.  For perf of the
  decode/prefill kernels, use the Stage-3 flow (this attempt made no perf change).

## Attempt 6 — 2026-09-17T01:11:41+0800 (mode: comment_cleanup, pre-PR P0 comment/docstring cleanup of gqa.py)

- classification: precision_pass (comment-only cleanup; full regression re-verified)
- fail_category: none
- scope: gqa.py ONLY — comments + docstrings; ZERO executable-statement changes
  (no code, no assert messages, no string constants, no ordering).  P0 22 items
  approved by user; P1 dedup explicitly NOT done; P2 red-line comments preserved
  (numbers + technical content intact; only process words removed in place).
- backup: history_version/gqa_impl_s2_attempt6.py
  (md5 fc7a36b906c4ae1a70403d7bcce29f1b == pre-edit gqa.py, verified after cp).
- change stats: 54 diff hunks; ~35 `#` comment sites + 17 docstrings touched;
  +174/-136 lines (net +38, 4486 -> 4524).
  md5 before fc7a36b906c4ae1a70403d7bcce29f1b -> after
  c04e788883a0839e2d9082e73f9ab33c.

### P0 items completed (22/22, full list in the attempt-6 task report)

1. stale 2^30-init claims (2004-2005, 2515 -> neg_sm_cap facts) — done
2. line 212 `-2^30 additive mask` -> `_BIG_NEG (=-1e30)` — done
3-6. 3042 / 3193 / 3336 / 4314 `-2^30` -> `_BIG_NEG` — done
7. dense_expert dispatch-guard docstring: kv_iters>=4 (was >=2) + added
   `G % G_inner == 0` and `block_num >= NUM_CORES` gates — done
8. `_gqa_kernel_expert` block_N docstring: factory passes 128 (sig default 64
   is direct-construction only) — done
9. gqa_expert "Dispatch on S": real g2stack condition (use_expert_prefill AND
   G%2==0 AND stacked instances >= NUM_CORES); non-eligible -> per-head
   expert / AUTO BM ladder / g2 last fallback — done
10. out_plan/transpose_out "caller keeps original tensor" -> actual
    `out_bhsd.transpose(1, 2).contiguous()` — done
11. "all bench shapes ... 128-byte-aligned D" -> real gate `D*2 % 32 == 0`
    (32-byte vector-load alignment) — done
12. `dormant g2` -> `last-resort g2` (253, 547; `_ENABLE_G2_FALLBACK=True`
    and g2 IS in the chain) — done
13. `196,6KB` -> `196.6KB`; added clarifications at BOTH the g2 fused-form
    note (196,352B shared.ub pool ceiling) and `_UB_BYTES=196608` (host-side
    transpose tiling budget) that they are distinct quantities — done,
    flagged "pending author confirmation" in the report (values NOT unified)
14. 3056 `test_gqa.py / perf scripts` dangling ref -> neutral "factory
    callers supplying real-S BHSD tensors" — done
15. attempt-numbering (1085, 1448, 1479, 1938, 1958) removed; the three P2
    technical constraints kept as present-tense facts: (a) runtime-conditional
    MTE3 scatter breaks SIG_S_HALF -> launch HANG; (b) `for s in range(S)` in
    prim_func is a TIR loop (s is a Var), Python-bool `if`s fold at trace
    time; (c) cross-frame `s_f`/`h_f` reads -> `Undefined variable` — done
16. anti-cheat / aclrtMemcpy / H2D-budget / D2H-egress narrative (2966, 3372,
    3402, 4436, 4440-4441) -> pure technical rationale (single host->device
    transfer; avoid a second transfer + fill kernel for a never-read buffer)
    — done, grep-zeroed
17. cann_bench recursion notes 4 copies (30-32, 4276-4277, 4374-4377, 4396)
    -> merged to 1 neutral Note in the module docstring — done
18. `cann-bench interface` naming (20, 4275, 4361, 4371, 4395) -> `Tensor
    (BSND) interface`; bench/harness context grep-zeroed (incl. 365-366, 384,
    391, 792, 3052, 3088, 3100, 4033, 4320 shape-condition rewrites) — done
19. verification-activity records (1532, 2035, 2484, 2519, 2969, 3057, 3072,
    3078, 3466, 4221): deleted or rewritten as design facts; 2484 kept
    "zero margin by design" + 65536B/131072B numbers; 4221 -> neutral
    "tilelang-compiled, no aclnn binary, same layout as aclnn Transpose";
    3466 "5x repeat bit-identical" sentence deleted — done.  Bonus (P2
    red-line process-word rewrite allowance): 2613-2615 "(zero margin,
    verified)" -> "(zero margin by design)", numbers intact.
20. bench-context process phrasing (1016, 3210-3212, 3223-3226, 3440-3445,
    3464-3466, 3472-3477, 4235, 4245-4246, 3622-3624) -> objective
    shape-condition descriptions; quantitative conclusions RETAINED
    (~4 KV iters / ~20 instances net-negative threshold, noise-band note,
    ci=1 over-sync / ci=4 starves-V untouched) — done
21. 2968-2969 "ViewCopy, the verified op" unconditional assertion ->
    conditional statement consistent with the other 5 sites (fp16/bf16
    slice-copy binary missing on SOME CANN 9.x SoC packages (561103),
    present on others e.g. CANN 9.2.0) — done
22. `_pad_along` docstring: ONE risk-note sentence added (fallback depends
    on device-side slice copy; unavailable where its fp16/bf16 binary is
    missing; gqa() main path pads in-kernel and is unaffected; serves only
    direct factory callers + non-contiguous inputs).  Other 5 561103 sites
    untouched — done

### Gate evidence (all measured this attempt)

- A. AST equivalence (hard proof of comment-only change): both files
  `ast.parse` -> strip ALL docstrings (Module/FunctionDef/ClassDef first
  str-Constant) -> `ast.dump`:
  `AST_DUMP_IDENTICAL=True (len=512492)` exit 0
  (tool: /tmp/opencode/ast_equiv.py; run twice — after edits and after all
  gates, same result).
- B. docstring change list: 17 docstrings (module header, _gqa_kernel_expert,
  _gqa_kernel_prefill_g2, _gqa_kernel_dense_expert,
  _gqa_kernel_dense_expert_d256, _gqa_kernel_prefill_expert,
  _gqa_kernel_prefill_g2stack, _zero_block, _pad_along, _prefill_run_padded,
  gqa_expert + its two nested _run docstrings, _pad_kv_npu, out_plan,
  transpose_out, gqa).  No technical information lost: all envelope numbers
  (65536B/131072B/196608/196,352B/196.6KB/524,032B/442KB/737KB/-1e30/~5e10/
  ~5.5e11/D*2%32), guard conditions and causal-defect rationales retained or
  corrected to match code (corrections were the P0-1 stale items).
- C. ruff 0.16.7 (/home/developer/.local/bin/ruff):
  `format --check` -> "1 file already formatted"; `check` -> "All checks
  passed!" — both exit 0 (baseline pre-edit also passed).
- D. grep-zero (case-insensitive counts on final gqa.py): anti-cheat=0,
  aclrtMemcpy=0, attempt-4=0, "attempt 3"=0, cann-bench=0, cann_bench=0,
  test_gqa=0, "eval harness"=0, "D2H egress"=0, "H2D budget"=0, bench=0,
  harness=0, dormant=0, "196,6"=0.  `2^30`: exactly 2 hits, both inside the
  whitelisted _BIG_NEG=-1e30 derivation block (now lines 4323/4325, the
  legitimate historical magnitude comparison).  debug_log|eval_fix_verify|
  history_version refs in gqa.py: 0.
- E. full regression (`unset PYTHONPATH; export ACL_OP_INIT_MODE=1; timeout
  3600 python examples/gqa/test_gqa.py --level all`): 18x [PRECISION_PASS],
  0x [PRECISION_FAIL], 6x [BOUNDARY_PASS], 0x [BOUNDARY_WARN], 0x
  "[gqa][fallback]", "Test Passed!" x1, GATE_E_EXIT=0 (captured via setsid
  wrapper; run twice — first run same markers, second run captured the exit
  code; log /tmp/opencode/gate_e_{test_all,rerun,exit}.log).
- F. A/B bitwise vs backup (14 shapes >= required 12;
  /tmp/opencode/gate_f_ab.py, log gate_f_ab.log, GATE_F_EXIT=0):
  ALL_BITWISE_IDENTICAL=True; dispatch names from each module's
  _KERNEL_BUILD_CACHE identical old vs new on every shape:
  g2stack x3 (incl. MQA N_kv=1), prefill_expert x1 (MHA N_kv=32 bf16),
  per_head_bm64 x4 (D256 causal + D256 noncausal bf16 + odd-kv + unaligned
  S_kv=124), dense_expert x3 (decode noncausal fp16 + decode causal bf16 +
  unaligned S_kv=2044 pad-mask), dense_expert_d256 x1, dense_auto x2
  (instance-short S_kv=2048 + S_kv=100 48.4%-pad), decode causal covered,
  unaligned prefill + unaligned decode covered, D256 covered.
- G. device discipline: every launch under `timeout`; post-run `npu-smi info`
  Health=OK on both chips, "No running processes found", 0 leftover
  test_gqa/gate_f processes.

### Not done / explicitly out of scope this attempt

- P1 (user-approved exclusion): 17 duplicate 561103 explanations NOT merged;
  SEM/SIG trailing-comment duplication untouched; 15 Chinese decorative
  section titles untouched; separator styles untouched; restating short
  comments NOT deleted.
- P0-3-adjacent but NOT in the approved 22-item list (left verbatim, flagged
  for separate approval): line 1536 assert message `"A64: D=256, BN=64"`
  (internal variant numbering — inside an EXECUTABLE statement, forbidden by
  hard constraint 1); `_dummy_mask` docstring "device-kernel-sum elapsed
  metric" phrasing (4296 area); non-listed verification-word sites kept:
  355/742/764/769/776/2474/2500-2501/688 (measured/verified wording outside
  the enumerated 10; 764/769 are P2 red-line regions where the allowance is
  optional); "runs in production" at ~359; "failed-compile attempts" at
  ~3144 (cache design fact, no gate word).
- UB pool numbers 196,352B vs _UB_BYTES=196608 intentionally NOT unified
  (per instruction); clarifying sentences added at both sites; values need
  author confirmation.

- rollback: no
- instrumentation_cleaned: n/a (no instrumentation added; comment-only edits)
- next_hint: file is PR-ready as a single-file example subject to the
  separately-approved items above.  Any future merge of the P1 561103
  duplicates should keep the authoritative missing-binary list (the
  BroadcastToAiCore/GatherV3/Cat_1_SliceAiCore/ConcatD/ZerosLike note in the
  decode _run) as the single full copy.

## Attempt 7 — 2026-09-17T08:09:17+08:00 (mode: comment_cleanup, Orchestrator-recheck residual 3 items / 7 sites)

- classification: precision_pass (cleanup-only attempt; no functional change intended,
  verified bitwise)
- fail_category: none
- test_level: all (L0+L1+Boundary full regression, Gate D)
- scope: exactly the 3 user-approved residual items from the attempt-6 recheck,
  7 edit sites total; nothing else touched.
- backup: examples/gqa/history_version/gqa_impl_s2_attempt7.py
  md5(backup)==md5(pre-edit gqa.py)==c04e788883a0839e2d9082e73f9ab33c (verified)
- post-edit md5(gqa.py) = 9d77b310be490f8c04b68ab9d6efec69; line count 4524 (unchanged)

### The 7 sites (diff vs backup: exactly 7 single-line hunks, verified by `diff -u`)

1. Item 1 (evaluation jargon, comment-only), line 4323 inside the _BIG_NEG=-1e30
   derivation block:
   `-# not just value_range [-1,1] scores.  With -(2^30) a full-range fp16 dot`
   `+# not just unit-range scores.  With -(2^30) a full-range fp16 dot`
   The whitelisted historical magnitude contrast `-(2^30)` (line 4323) and
   `s_j - 2^30` (line 4325) kept verbatim — grep '2\^30' count == 2 before and after.
2. Item 2 (dangling `golden` references -> `the reference`, comment-only), 5 sites:
   - L144:  `(output 0 vs golden's argmax one-hot)` -> `(output 0 vs the reference's argmax one-hot)`
   - L287:  `(golden outputs 0 for fully` -> `(the reference outputs 0 for fully`
   - L534:  `(max-after-mask — the golden semantics): masked` -> `(max-after-mask — the reference semantics): masked`
   - L654:  `(golden outputs 0).  With the` -> `(the reference outputs 0).  With the`
   - L2386: `(the golden semantics; matches the g2/per-head` -> `(the reference semantics; matches the g2/per-head`
   Only the referent word replaced; sentence remainder, line-wrap structure and
   non-ASCII em-dashes untouched; all resulting lines <= 140 cols (ruff format clean).
3. Item 3 (executable-statement string constant, SEPARATELY user-approved), line 1551
   in _gqa_kernel_dense_expert_d256 (attempt-6 log cited ~1536; actual anchor line 1551):
   `-    assert D == 256 and block_N == 64, "A64: D=256, BN=64"`
   `+    assert D == 256 and block_N == 64, "dense_expert_d256 requires D=256, block_N=64"`
   Only the message literal changed; the assert condition expression is provably
   identical (Gate A [A3]: ast.dump(BoolOp(And, [D==256, block_N==64])) equal).

### Gates (all measured, scripts /tmp/opencode/gate_a_ast_att7.py, orch_verify_att7.py)

- A. AST: docstring-stripped ast.dump old-vs-new — full-tree walk finds EXACTLY 1
  differing node: path Module.body[11].body[0].msg.value, Constant at lineno 1551
  col 39, OLD='A64: D=256, BN=64' -> NEW='dense_expert_d256 requires D=256, block_N=64'.
  Per-factory: 6/7 `_gqa_kernel_*` (expert / prefill_g2 / dense / dense_expert /
  prefill_expert / prefill_g2stack) ast.dump identical; `_gqa_kernel_dense_expert_d256`
  differs in exactly that 1 Constant value, nothing else.
- B. Zeroing self-proof (grep -c on new file): golden(ci)=0, value_range=0, "A64=0,
  '2\^30'=2 (whitelist intact), 'noqa: SIM102'=4, 561103=19.
  P2 redline token counts new==backup for: 65536(2), 131072(3), 196608(3),
  196,352(2), 524,032(1), 262144(2), AscendVidReduction(2), 2.2-7.9x(1),
  SIG_S_HALF(49), 'Undefined variable'(1), 1e-30(10), neg_sm_cap(10), kv_useful(15).
- C. ruff 0.16.7 (/home/developer/.local/bin/ruff, NOT the 0.15.14 on PATH):
  `format --check` -> "1 file already formatted", exit 0; `check` -> "All checks
  passed!", exit 0.
- D. Full regression (`unset PYTHONPATH; export ACL_OP_INIT_MODE=1;
  timeout 7200 python examples/gqa/test_gqa.py --level all`, log
  /tmp/opencode/gate_d_att7.log): 18 [PRECISION_PASS] + 6 [BOUNDARY_PASS] +
  `Test Passed!`; PRECISION_FAIL=0, BOUNDARY_WARN=0, `[gqa][fallback]` lines=0,
  error/traceback lines=0. Exit 0 evidenced by harness code path: test_gqa.py
  L542-545 prints "Test Passed!" immediately followed by sys.exit(0) with no
  intervening statement; log ends at that print with clean process termination.
- E. A/B bitwise vs backup (/tmp/opencode/orch_verify_att7.py, att6 paradigm with
  BASE=attempt7 backup; log /tmp/opencode/gate_e_att7.log, GATE_E_EXIT=0):
  16/16 shapes bitwise=True AND identical _KERNEL_BUILD_CACHE dispatch names:
  dense_expert x5 (S1 fp16/bf16-G12, S2 causal, S4 causal G16, MHA G1),
  dense_expert_d256 x1, dense_auto x3 (unaligned S_kv=300/100, causal unaligned 132),
  g2stack x5 (incl. nonaligned pad 124/252, nonaligned causal 124/132, MQA G32,
  long causal 512), prefill_expert x1 (MHA G1), per_head_bm64 x1 (D256 causal).
  Unaligned decode AND unaligned prefill both covered; 16 >= required 10.
- F. Device discipline: both Gate D/E launches wrapped in `timeout` (7200/5400);
  post-run npu-smi: Health=OK on both chips, "No running processes found in NPU",
  0 leftover test_gqa/orch_verify processes; HBM 3126/2870MB is driver baseline
  with zero attached processes.

### Out of scope / untouched (per instruction)

- P1 dedup excluded by user; P2 redlines and attempt-5/6 whitelist regions not
  touched (verified by token counts in Gate B); `.orchestrator_state.json` neither
  read nor written.

- rollback: no
- instrumentation_cleaned: n/a (no instrumentation; comment/string-literal-only edits)
- next_hint: all Orchestrator-recheck residuals cleared; file satisfies the same
  PR-ready state as attempt 6 with the previously-flagged "A64" assert message now
  self-explanatory. No known outstanding cleanup items.

## Attempt 8 — 2026-09-17T10:54:07+08:00 (mode: ci_fix, `python gqa.py` produces no CI completion marker)

- classification: precision_pass (append-only CI fix; no kernel/dispatch change, A/B bitwise-verified)
- fail_category: none
- test_level: all (Gate E full L0+L1+Boundary regression) + the CI harness replica (Gate A)

### Root cause (Orchestrator-supplied diagnosis, re-confirmed here)

`examples/bench_test.sh:485-500` accepts a script only when the exit code is 0 **and** the whole
captured output matches `KERNEL OUTPUT MATCH` or `TEST PASSED!` (case-insensitive):

```bash
if (( exit_code == 0 )) && { [[ "$output" =~ [Kk][Ee][Rr][Nn][Ee][Ll]...[Mm][Aa][Tt][Cc][Hh] ]] ||
                            [[ "$output" =~ [Tt][Ee][Ss][Tt][[:space:]][Pp][Aa][Ss][Ss][Ee][Dd][!] ]] }; then
```

`gqa.py` was a pure kernel-library module with **no `__main__` block**, so `python gqa.py`
(harness: `output=$(cd "$script_dir" && python "$script_name" 2>&1)`, bench_test.sh:477) exited 0
silently with no marker → `[FAILED] ./gqa/gqa.py (Exit: 0)`. Repo convention confirmed:
`examples/gqa_fwd_varlen/gqa_fwd_varlen.py` ends with kernel run → golden compare → assert →
`print("Test Passed!")`; 26 example files print that marker. Also confirmed
`examples/gqa/gqa.py` is NOT claimed by `ci/operator_test_manifest.yaml`
(`resolve_operator_tests.py list --format tsv | grep -c '^examples/gqa/gqa.py'` = 0), so the legacy
runner really does execute it — the CI log is reproducible in kind.

### Change (append-only, 129 lines at EOF; lines 4524 → 4653)

| lines | content |
|---|---|
| 4525-4530 | section banner comment (`Self-verification entry point (\`python gqa.py\`)`) |
| 4532-4586 | `def _ref_gqa(query, key, value, scaleValue=-1.0, is_causal=False)` — module-private host oracle; semantics copied from the authoritative golden (`.agents/skills/cann-bench/tasks/level4/gqa/golden.py:49-80`): GQA group folding (q head `n_kv*G+g` ↔ kv head `n_kv`, G folded into the matmul M dim, K/V never materialized to N_q heads), `scaleValue<=0 → 1/sqrt(D)`, right-bottom-aligned causal (`j > i + (S_kv-S)` → -inf), fully-masked rows (causal, S>S_kv) softmax NaN → output exactly 0, NaN/Inf unsanitized |
| 4588-4653 | `if __name__ == "__main__":` self-check — 2 bench shapes, mixed tolerance, `Test Passed!` / `sys.exit(1)` |

md5: `9d77b310be490f8c04b68ab9d6efec69` → `6dab75d373ae71671b53b661b65ee674`
backup: `examples/gqa/history_version/gqa_impl_s2_attempt8.py`, md5 `9d77b310be490f8c04b68ab9d6efec69` (== pre-edit file, verified)

Self-check cases (both bench-verified shapes, one per dispatch path):

| case | shape | path | seed |
|---|---|---|---|
| `prefill_causal` | B=2 S=128 S_kv=128 N_q=32 N_kv=8 D=128 causal=True fp16 (bench c1) | transpose_qkv + g2stack + transpose_out | `torch.manual_seed(42+B+S+S_kv)` = 272, uniform[-1,1] |
| `decode_noncausal` | B=16 S=1 S_kv=2048 N_q=32 N_kv=8 D=128 causal=False fp16 (bench c11) | dense_expert, BSND direct (no helper kernel) | = 2065, uniform[-1,1] |

Per-case checks: output shape `(B,S,N_q,D)`, dtype == input dtype, contiguity, no NaN/Inf, and the
fp16 mixed tolerance `atol=1e-3, rtol=1e-2` with **zero** allowed violations; `max_diff`, `viol` and
launch time are printed for CI-log triage. `torch.npu.synchronize()` before and after the launch;
device reached via `.npu()` only (no hardcoded device id). `tilelang.disable_cache()` is called
inside the guard (repo convention, e.g. `gqa_fwd_varlen.py`; also used by `test_gqa.py:531`) so CI
genuinely recompiles instead of reusing a container-cached `.so`. All imports (`sys`, `time`,
`torch_npu`) live inside the guard → importing the module performs no device work.

### Gates (all measured; scripts `/tmp/opencode/att8_gate_b_ast.py`, `att8_gate_f_ab.py`, `att8_probe_golden.py`)

- **A. CI harness judgement — PASS, measured two ways.**
  1. Replica of bench_test.sh:475-500 (`output=$(cd examples/gqa && python gqa.py 2>&1); exit_code=$?`
     + the harness's own `if`): `EXIT_CODE=0`, printed `[PASSED] ./gqa/gqa.py`; wall 32 s.
     Output tail (log `/tmp/opencode/att8_gateA.log`, `fallback=0 traceback=0 error=0`):
     ```
     [gqa] prefill_causal B=2 S=128 S_kv=128 N_q=32 N_kv=8 D=128 causal=True fp16 max_diff=3.663e-04 viol=0 launch=15.0s -> PASS
     [gqa] decode_noncausal B=16 S=1 S_kv=2048 N_q=32 N_kv=8 D=128 causal=False fp16 max_diff=2.281e-05 viol=0 launch=6.1s -> PASS
     Test Passed!
     ```
  2. **The real, unmodified `bench_test.sh`** in a sandbox holding exactly the PR content
     (`/tmp/opencode/att8_cisim/examples/gqa/gqa.py`, md5 identical) with `--skip-pytest --dirs gqa`
     (log `/tmp/opencode/att8_gateA_harness.log`): `Collected scripts from gqa: 1 files` →
     `[PASSED] ./gqa/gqa.py` → `Total: 1 | Passed: 1 | Failed: 0`, `Pass rate: 100%`,
     `BENCH_TEST_EXIT=0`, wall 33 s. (Sandbox omits `scripts/ci/resolve_operator_tests.py`, which only
     feeds the migrated-operator skip lists; gqa.py is not in the manifest, so CI behaves identically.)
  3. Failure path proven non-silent: a throwaway copy with the tolerance forced to `1e-9` exits **1**,
     prints `[gqa] self-check FAILED for: prefill_causal, decode_noncausal`, prints no `Test Passed!`,
     and the harness judges `[FAILED]`. Its `max_diff` values (3.663e-04 / 2.281e-05) match the passing
     run exactly → the self-check is deterministic.
- **B. Existing code untouched — PASS.** Byte level: `head -4524 gqa.py | md5sum` ==
  `9d77b310be490f8c04b68ab9d6efec69` == backup md5, `diff <(head -4524 gqa.py) backup` empty;
  `diff -u backup new` = 129 `+` lines / 0 `-` lines (pure append). AST level
  (`/tmp/opencode/att8_gate_b.log`): 52 pre-existing top-level nodes → **52/52 `ast.dump` identical,
  0 differing**; `REMOVED=[]`; `ADDED=['FunctionDef:_ref_gqa', 'If@53']`. All 30 pre-existing
  top-level defs individually `SAME`: `_gqa_kernel_expert`, `_gqa_kernel_prefill_g2`,
  `_dense_tile_params`, `_gqa_kernel_dense`, `_gqa_kernel_dense_expert`, `_gqa_kernel_dense_expert_d256`,
  `_gqa_kernel_prefill_expert`, `_gqa_kernel_prefill_g2stack`, `_zero_block`, `_npu_sync_tensor`,
  `_pad_along`, `_prefill_run_padded`, `_perhead_bm_ladder`, `_resolve_kernel_chain`, `gqa_expert`,
  `_ceildiv`, `_ub_rows`, `_transpose_qkv`, `_transpose_v1`, `_pad_kv`, `_pad_kv_npu`, `_dtype_str`,
  `_shared_bm`, `_tail_bm`, `transpose_qkv`, `out_plan`, `transpose_out`, `_dummy_mask`,
  `_build_causal_mask`, `gqa`; plus all 22 non-def statements `SAME` (module docstring, `import tilelang`,
  `from tilelang import language as T`, `from tilelang.intrinsics import make_zn_layout, make_nz_layout`,
  `import torch`, `P28_NUM_CORES`, `P28_EXPERT_PASS_CONFIGS`, `_ZERO_POOL`, `_ZERO_POOL_INITIAL_NUMEL`,
  `_KERNEL_BUILD_CACHE`, `_ENABLE_G2_FALLBACK`, `_PERHEAD_L0C_BUDGET`, `_TRANSPOSE_PASS_CONFIGS`,
  `_VEC_NUM`, `_UB_BYTES`, `_transpose_kernel_cache`, `_pad_kv_kernel_cache`, `_BIG_NEG`,
  `_KERNEL_BLOCK_N`, `_COMPILED_CACHE`, `_DUMMY_MASK`, `__all__`). `__all__` unchanged =
  `['gqa', 'gqa_expert']`; exactly 1 module-level `__main__` guard (line 4588); the only new
  module-level function is `_ref_gqa` (private, not exported).
- **C. Import safety — PASS.** `cd examples/gqa && python -c "import gqa; print(gqa.__all__)"` →
  stdout exactly `['gqa', 'gqa_expert']`, exit 0; stderr carries only the pre-existing torch_npu
  `Permission mismatch ... libop_plugin_atb.so` UserWarning, byte-identical to the same command run
  against the backup → no new output pollution, no main execution. Wheel simulation
  (`/tmp/opencode/att8_wheelcheck/cann_bench/__init__.py: from .gqa import gqa` over the new file):
  import ok, `callable(cann_bench.gqa)` = True, `torch.npu.is_initialized()` = **False**,
  allocated bytes = 0 → the guard keeps packaging import device-free.
- **D. ruff 0.16.7 (`/home/developer/.local/bin/ruff`) — PASS.** `format --check` →
  `1 file already formatted`, exit 0; `check` → `All checks passed!`, exit 0. Longest new line 111 cols
  (file max 140, pre-existing) — line-length 140 respected.
- **E. Full regression — PASS, exit code captured directly.** `unset PYTHONPATH; export
  ACL_OP_INIT_MODE=1; timeout 7200 python examples/gqa/test_gqa.py --level all`
  (log `/tmp/opencode/att8_gate_e.log`): `PRECISION_PASS=18`, `PRECISION_FAIL=0`, `BOUNDARY_PASS=6`,
  `BOUNDARY_WARN=0`, `Test Passed!`, **`GATE_E_EXIT=0`**, `[gqa][fallback]` lines = **0**,
  traceback lines = 0. Per-case max_diff identical to attempt 7 (e.g. `l0_prefill_causal_fp16`
  4.883e-04, `l0_decode_noncausal_fp16` 3.052e-05, `l1_mha_causal_bf16` 3.906e-03).
- **F. A/B bitwise vs backup — PASS 11/11** (`/tmp/opencode/att8_gate_f_ab.py`, att7 paradigm,
  BASE = attempt8 backup; log `/tmp/opencode/att8_gate_f.log`, `GATE_F_EXIT=0`). Every shape
  `bitwise=True` **and** identical `_KERNEL_BUILD_CACHE` dispatch name:
  `g2stack` ×4 (c1 bench 2/128/128/32/8/128 causal fp16; non-aligned prefill S=124/S_kv=132; MQA G=32;
  long causal S=512), `prefill_expert` ×1 (MHA G=1 bf16), `per_head_bm64` ×1 (D=256 causal),
  `dense_expert` ×1 (c11 bench decode 16/1/2048), `dense_auto` ×3 (S=2 causal non-aligned S_kv=132,
  decode S_kv=300, decode S_kv=100), `dense_expert_d256` ×1 (4/1/512 D=256). All 8 required families
  covered; 11 ≥ 8.
- **G. Device discipline — PASS.** Every device launch wrapped in `timeout` (1800 s for the harness
  runs, 7200 s Gate E, 5400 s Gate F, 300 s import checks). Post-run `npu-smi info`: Health = **OK**
  on both chips, `No running processes found in NPU`, HBM 3125/2870 MB = driver baseline (same as
  attempt 7), 0 residual `gqa.py` / `test_gqa.py` / `att8_*` processes.

### Golden-device decision (and its measurement)

The inlined oracle is evaluated on **host float32**, not in the input dtype on NPU as `test_gqa.py`'s
f6 golden does. Reason: an NPU-side golden needs `permute().contiguous()` / `zeros_like` / `masked_fill`
aclnn binaries, i.e. exactly the Transpose / ViewCopy / ZerosLike class this module is engineered
around because they are missing on some CANN 9.x SoC packages (error 561103, documented in the module
docstring and at 19 call sites) — a golden-side 561103 on the untested CI toolchain would re-break the
job for a reason unrelated to the kernel. Both variants were measured on the same kernel outputs
(`/tmp/opencode/att8_probe_golden.log`, viol counted at atol=1e-3/rtol=1e-2):

| case | NPU fp16 golden (f6) | host fp32 golden (chosen) | host fp64 golden |
|---|---|---|---|
| c1 prefill causal | max_diff 4.883e-04, viol 0 | **max_diff 3.663e-04, viol 0** | 3.663e-04, viol 0 |
| c11 decode | max_diff 3.052e-05, viol 0 | **max_diff 2.281e-05, viol 0** | 2.281e-05, viol 0 |

The host oracle is the *stricter* reference (smaller deviation, fp32≈fp64 here). Tolerance margin: for
c1 the mixed tolerance at |ref|=1.0 is 1.1e-2 vs 3.66e-4 observed (~30×); for c11 ~70×. Both bench
shapes have S ≤ S_kv, so no fully-masked row arises in the self-check; that branch of `_ref_gqa` is
still implemented per contract and is exercised by `test_gqa.py`'s `boundary_full_masked_rows`.

### Unverified risks on the CI environment (honest gaps)

1. **Toolchain difference (main risk).** Local: aarch64, Python 3.12.9, CANN 9.2.0
   (`ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.2.0`), torch 2.7.1+cpu, torch_npu 2.7.1.post4,
   Ascend910_9362, pip tilelang. CI: x86_64, Python 3.11, **CANN 9.0.0-beta.2**, torch_npu
   2.7.1.post2, docker `tilelang_x1`, repo-built tilelang via `install_ascend.sh` + `source set_env.sh`.
   Compilation of the c1/c11 chains on beta.2 **cannot be verified locally**. Mitigation: both shapes
   are bench shapes already validated on two CANN versions, and `gqa_expert` keeps its compile-fallback
   chain — a candidate that fails to compile on beta.2 is skipped (prints `[gqa][fallback]`) in favour
   of the next one; only if *every* candidate fails does the self-check exit non-zero (which would be a
   genuine CI-worthy break).
2. **Environment invocation differs from CI.** Locally `source set_env.sh` is unusable (the in-repo
   tilelang has no `tvm`), so all runs used `unset PYTHONPATH; export ACL_OP_INIT_MODE=1` with the pip
   tilelang. The *invocation form* CI uses (`cd examples/gqa && python gqa.py`, module name `__main__`,
   `sys.path[0]=examples/gqa`) was reproduced exactly, including through the real `bench_test.sh`, but
   the tilelang build backing it was not the repo-built one.
3. **Compile-time budget.** `disable_cache()` forces a fresh compile each CI run: 21.4 s of kernel
   compile locally (15.0 s prefill chain incl. the transpose kernels + 6.1 s decode), 32-33 s total
   wall. If beta.2/x86_64 compiles several times slower it still fits the PR job budget
   (`timeout-minutes: 60`, `MAX_JOBS=8`), but this was measured only locally.
4. **Coverage of the self-check is intentionally 2 shapes** (instruction 4). `prefill_expert`,
   `per_head_bm64`, `dense_expert_d256` and `dense_auto` are not exercised by `python gqa.py`; they stay
   covered by `test_gqa.py` (Gate E), which is not part of the PR. Trade-off: ~33 s CI runtime and no
   extra kernel families compiled on an unverified toolchain, at the cost of a shallower smoke test.

### Out of scope / untouched

`ci/operator_test_manifest.yaml`, `examples/bench_test.sh`, all workflows and
`.orchestrator_state.json` neither read-modified nor written (`git status --porcelain` shows no `M`
entry for any tracked file). No new file is added to the PR: the only PR change is
`examples/gqa/gqa.py` (append-only). `history_version/gqa_impl_s2_attempt8.py` and this log are
workspace artifacts and must not be staged.

- changes: `examples/gqa/gqa.py` (+129 lines at EOF: banner 4525-4530, `_ref_gqa` 4532-4586,
  `__main__` self-check 4588-4653); no existing line modified or deleted.
- error_summary: none (CI failure was the missing completion marker, now emitted).
- design_error_reason: none
- rollback: no
- backup_path: examples/gqa/history_version/gqa_impl_s2_attempt8.py
- instrumentation_cleaned: n/a (no instrumentation added; the failure-path probe ran on a throwaway
  copy under /tmp/opencode/att8_failpath/, the delivered file was never modified for it)
- next_hint: re-trigger the CI `test` job; expect `[PASSED] ./gqa/gqa.py` and `Total: 1 | Passed: 1 |
  Failed: 0`. If beta.2 cannot compile the c1 prefill chain, the log will show `[gqa][fallback]` lines
  followed by a pass (chain fallback) or a non-zero exit (all candidates failed) — the printed
  per-case `max_diff`/`launch` values identify which case and how long compilation took.
