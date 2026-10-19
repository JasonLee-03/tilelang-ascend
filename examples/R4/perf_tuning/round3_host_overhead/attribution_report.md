# R4 Stage 3 — Host-Overhead Attribution Report

**Operator:** `R4 fused sampling` (`examples/R4/r4_fused_sampling.py`, sha256 `a764e117…`)
**Scope:** read-only root-cause investigation of the ~370 us/launch host gap that round 3
(`final_report_round3.md` §7) exposed but did **not** root-cause. **No kernel / test / proto / case
file was modified.** This report synthesises the seven probes already measured in
`round3_host_overhead/` — no expensive measurement was re-run.
**Date:** 2026-10-11 · **HW:** Ascend910_9362 (910b2) · tilelang `0.1.4+d75300ed` · torch_npu `2.7.1.post4` · py 3.12.9

> **Measurement boundary (probe_common.py docstring):** every number below is a HOST-side
> `perf_counter_ns` wall-clock figure or a model estimate built on msprof device Durations.
> **None is a Python-timed kernel duration and none claims a device-time speed-up.** The
> authoritative device times come from round-3's msprof `kernel_details.csv`
> (`../round3_splitv_perf/`), never re-derived here from a Python timer.

---

## 0. Verdict

# `[HOST_OVERHEAD_ROOT_CAUSED]`

**One-line root cause.** The ~370–500 us/launch host cost is **not R4-specific and not our
wrapper** — it is dominated by (a) the **cann-bench harness's per-call guard stack** (~233 us/launch,
C2) and (b) **tilelang's framework-level dispatch**, whose single largest primitive is
`tvm.arith.Analyzer()` at **122.6 us** (C4). msprof instrumentation adds ~66 us/launch (C1). Our own
wrapper is **<2 %** (C3). A minimal tilelang kernel with **zero R4 logic** costs **160–180 us/launch**
(`probe2` controls) versus **18 us** for `torch.add` on NPU — proving the cost is framework-level.
**ACL-graph capture collapses the per-launch host cost to 2.8–6.6 us** and is bit-exact, which flips
rev3's split-path SPAN from a regression to a win.

---

## 1. Attribution table C1–C5

**Backbone = probe3's instrumentation ladder** (`probe3_ladder_and_span.json →
ladder_summary_per_launch_us`), medians over the 4 probed cases (1/3/13/7):

| level | per-launch host (us) | source field |
|---|---|---|
| **L0** bare wrapper (no guard, no profiler) | **200.5115** | `ladder_summary_per_launch_us["0"]` → median |
| **L3** +harness guards (TorchFunctionMode/TorchOpGuard + TorchDispatchMode/DeviceResidencyGuard + record_function) | **433.9085** | `["3"]` → median |
| **L5** +msprof activities=[CPU,NPU] == **cann-bench harness** | **499.6380** | `["5"]` → median |

These three medians are re-stated verbatim in `corrected_span.json → _meta.H_bare_us /
H_guards_us / H_guards_msprof_us`. The ladder is **additive**, so the per-launch host cost of the
full harness regime (L5 = 499.6 us) decomposes exactly:

| id | component | **us / launch** | **% of L5 (499.6)** | derivation (source → field) |
|---|---|---|---|---|
| **C1** | msprof instrumentation | **65.73** | **13.2 %** | L5 − L3 = 499.638 − 433.909 (`probe3 ladder_summary_per_launch_us`); cross-check `probe1._c1_isolation.fused_case1`: none 206.663 → cpu_npu 265.51 = **+58.85 us/launch** (`ratio_cpunpu_over_none` 1.2847) |
| **C2** | cann-bench harness per-call guards | **233.40** | **46.7 %** | L3 − L0 = 433.909 − 200.512 (`probe3`); reproduced by the faithful harness `probe7.ladder_table` case1 L0 208.559 → L3 480.449 = **+271.9**, case3 196.663 → 416.192 = **+219.5** |
| **C4** | tilelang framework dispatch (Analyzer + simplify + tvm_ffi convert + torch_npu stream) | **≈135–158** | **27–32 %** | `probe2b.model[*].model_prep_us` (0-alloc K1A 135.403 → 4-alloc K1B 157.923); dominant primitive `probe2b.analyzer_direct.med_us` = **122.603**, `analyzer_plus_simplify.med_us` = 142.128 |
| **C5** | ACL launch itself (residual, by subtraction) | **≈17–38** | **3.4–7.6 %** | `probe2b.model[*].residual_us` = measured_prep − model_prep (K1A 17.479 → K1B 38.461); `probe2b.model_residual_max_us` = **38.461** |
| **C3** | **our** host wrapper (removable) | **≈2–4** (4.01–15.76 us **per call**, amortised over 2–4 launches) | **<2 %** | `probe5.summary[*].C_wrapper_saving_us` (4.01/12.72/4.07/15.76), `C_pct` **0.97 % / 1.60 % / 0.98 % / 1.97 %** |

**Additive check:** C4+C5 ≈ L0 = 200.5 us (bare launch); L0 + C2 + C1 = 200.5 + 233.4 + 65.7 =
**499.6 = L5** ✓. C3 lives inside the wrapper-call overhead (split_plan/jit_lookup) that L0's
per-launch median already amortises; probe5 isolates the *removable* part as <2 %.

**Reconciling with round-3's "~363–386 us/launch gap".** Round 3 measured the *inter-kernel device
gap*, which is `gap = max(0, H − d_prev)` (`probe_common.py` identity; `reconcile_with_round3_csv.json
→ _meta.identity`). The gap is the **exposed** part of H after the previous kernel's device time
overlaps it. The device-gap-implied H is **403.85 us** median over host-bound launches
(`reconcile_with_round3_csv.json → reconciliation.csv_implied_H_host_bound_us.median`, n=501), and
`harness_over_bare_ratio` = **2.0141** (403.85 / 200.51). So the same host cost reads as ~370 us of
*gap* but ~404 us of *implied H* (device-derived) and ~500 us of *directly-measured L5 host wall* —
the three differ only by how much device overlap and msprof accounting is included. All three are the
**same C1–C5 stack**.

**Separation caveats (honest):**
- **C5 is a residual, not a direct measurement.** It is `measured_prep − model_prep`, i.e. whatever the
  primitive model (Analyzer+simplify+stream+n_alloc×`torch.empty`) does not explain. It bounds
  "ACL launch + unmodelled FFI marshalling" from above; **the ACL launch could not be isolated from
  FFI argument-marshalling with the probes run** — no probe measured `lib.call` alone.
- **C1 from probe3 (65.7 us, guards present) vs probe1 (58.85 us, bare wrapper)** differ by ~7 us
  because msprof's `record_function` nesting costs slightly more on top of the guard stack. Both are
  reported; the ladder value (65.7) is used for the additive table.
- **C4 vs C3 overlap.** `model_prep` (C4) includes the per-launch `torch.empty` output allocations;
  the workspace-reuse saving in C3 (`probe5`, 5.5 us) targets a *different* allocation (the GM
  workspace, once per call). No double-count, but the boundary is by construction, not measurement.

---

## 2. Is it R4-specific? — **No. Framework-level.** (`probe2.controls`)

`probe2_launch_anatomy.json → controls` measures three throw-away kernels that contain **no R4 code**:

| control | back-to-back host / call (us) | host_async med (us) | #allocs | source field |
|---|---|---|---|---|
| `torch_native_add_npu` (plain `torch.add` on NPU) | **20.903** | 17.905 | 1 | `controls.torch_native_add_npu` |
| `tilelang_minimal_NO_out_idx_0alloc` (empty tilelang kernel) | **160.579** | 164.298 | 0 | `controls.tilelang_minimal_NO_out_idx_0alloc` |
| `tilelang_minimal_out_idx_1alloc` (empty tilelang kernel, 1 out) | **180.494** | 179.168 | 1 | `controls.tilelang_minimal_out_idx_1alloc` |
| **R4 wrapper L0 (for reference)** | — | **200.5115** | 0–4 | `probe3 ladder_summary_per_launch_us["0"]` |

**Reading.** `torch.add` on NPU costs ~18–21 us/launch (the true ACL-launch + torch-dispatch floor,
≈ C5). An **empty tilelang kernel** costs **160–180 us/launch** — i.e. **+142 to +160 us of pure
tilelang framework dispatch (C4)** on a kernel that does nothing. R4's own L0 (200.5 us) is only
**+20 to +40 us** above the minimal tilelang kernel, and that delta is R4's extra output allocations
(K1B is 4-alloc) plus the small wrapper logic (C3). 

> **Conclusion: ≳90 % of the bare ~200 us/launch is framework-level tilelang dispatch that ANY
> tilelang kernel pays — it is NOT introduced by R4 and NOT by rev3.** The single biggest lever inside
> it is `Analyzer()` (122.6 us, `probe2b.analyzer_direct.med_us`), confirmed by
> `probe5.B_framework_ceiling`: a hypothetical framework that skipped `Analyzer` when the dynamic-shape
> map is empty and cached the stream would drop K1A(0-alloc) from 149.79 us to **6.14 us** (−95.9 %,
> `saving_pct` 95.9) and K1(4-alloc) from 188.91 us to 42.54 us (−77.5 %). That is a **framework**
> change, not a kernel change.

---

## 3. Did rev3 make the host side heavier? — **No. Same per-launch cost; rev3 only raised the launch COUNT.**

`probe3_ladder_and_span.json → base_vs_cand` compares the **base** arm (2 launches: K1 + K2g) against
the **cand**=rev3 arm (4 launches: K1A/K1B/K2gA/K2gB) at L0, on the two split cases that were measured
(case 1 and 13 are `skipped` — forced S=1/S=2, no split):

| case | arm | n_launch | **per_launch_us** | per_launch_incl_jit_us | source field |
|---|---|---|---|---|---|
| **3** | base_2launch | 2 | **195.118** | 197.888 | `base_vs_cand["3"].arms.base_2launch.per_launch_us` |
| **3** | cand_4launch | 4 | **180.531** | 182.999 | `.cand_4launch.per_launch_us` |
| **7** | base_2launch | 2 | **196.168** | 199.018 | `base_vs_cand["7"].arms.base_2launch` |
| **7** | cand_4launch | 4 | **179.056** | 181.434 | `.cand_4launch` |

`per_launch_delta_cand_minus_base_us` = **−14.587** (case3) / **−17.112** (case7);
`per_launch_ratio_cand_over_base` = **0.9252 / 0.9128**. The **per-launch host cost is the SAME — in
fact marginally LOWER for the split kernels** (each split kernel has fewer/cheaper args, so its
`forward_prep` is a touch smaller). And L0's per-launch median is flat across all four probed cases
(202.9 / 198.1 / 205.8 / 196.5, `ladder_summary_per_launch_us["0"]`), 2-launch and 4-launch alike.

> **Conclusion: rev3 did NOT make the host side heavier per launch.** It kept the ~195–200 us/launch
> cost identical and simply went from **2 launches to 4** on the 6 split cases. The §7 SPAN regression
> is therefore purely a **launch-count × fixed-host-cost** effect, not a per-launch cost increase — which
> is exactly why removing the per-launch host cost (ACL graph, §5) fixes it without touching the kernel.

---

## 4. ⭐ Robustness re-evaluation of the SPAN conclusion

Round 3 (`final_report_round3.md` §7) reported SPAN geomeans measured **inside the msprof harness**
(≈ L5): **fused 0.7443× / mask 0.8005×** (all-18), **split-6 0.3831× / 0.4387×**. But probe3 showed
the harness inflates per-launch host by **L5/L0 = 2.34–2.65×** (`probe3 → inflation[*].L5_over_L0`:
2.6482 / 2.3424 / 2.5983 / 2.3643). So round-3's SPAN describes a **profiled, guarded** harness, not a
deployment. `corrected_span.json` re-estimates SPAN under different host assumptions using the model
`span: start_i = max(end_{i-1}, i·H)`, device durations from round-3 msprof CSVs (authoritative).

**Model validation** (`corrected_span.json → _meta.validated`): **V1** (vs round-3 msprof spans,
per-case CSV-H) median abs err **0.058 %** ✅; **V2** (vs probe3 NPU-event spans, H_bare) median abs err
**7.583 %** ✅. ⚠️ V1 anchors only the **base-arm** span; the **cand-arm** split spans (the ratio
denominator) are model estimates, so the split-6 subset carries the larger V2-scale (7.6 %) uncertainty.

### 4.1 Corrected SPAN geomeans (model estimates)

Fixed-H rows recomputed from `corrected_span.json → {fused,mask}.per_case[*].durs_base/durs_cand`
using `analyze_corrected_span.py → model_span(durs,H)`; the H=200.5 and graph rows reproduce the stored
`geomean_span_bare` / `geomean_span_graph` exactly (verified). CSV-H row = stored `geomean_span_harness`.

**FUSED task:**

| host assumption | H / launch | **all-18 SPAN** | **split-6 SPAN** | source |
|---|---|---|---|---|
| msprof harness (**round-3口径**) | 499.638 us | **0.74434** | 0.41214 | recompute @ `_meta.H_guards_msprof_us` |
| guards, no msprof | 433.909 us | 0.75091 | 0.42311 | recompute @ `_meta.H_guards_us` |
| **bare eager wrapper** | 200.512 us | **0.83513** | 0.58159 | `geomean_span_bare` / `_splitonly` |
| (per-case CSV-H, ≈round-3) | ~404 us var | 0.75799 | 0.40544 | `geomean_span_harness` / `_splitonly` |
| **ACL-graph captured** | 15.588 us residual | **1.12421** | **1.40886** | `geomean_span_graph` / `_splitonly` |
| *scored device time (host-independent)* | — | *1.28027* | *(2.0624, round-3)* | `geomean_scored` |

**MASK task:**

| host assumption | H / launch | **all-18 SPAN** | **split-6 SPAN** | source |
|---|---|---|---|---|
| msprof harness (**round-3口径**) | 499.638 us | **0.79803** | 0.50819 | recompute @ `H_guards_msprof_us` |
| guards, no msprof | 433.909 us | 0.79866 | 0.50938 | recompute @ `H_guards_us` |
| **bare eager wrapper** | 200.512 us | **0.86548** | 0.64807 | `geomean_span_bare` / `_splitonly` |
| (per-case CSV-H, ≈round-3) | ~404 us var | 0.80009 | 0.44540 | `geomean_span_harness` / `_splitonly` |
| **ACL-graph captured** | 15.588 us residual | **1.08422** | **1.26760** | `geomean_span_graph` / `_splitonly` |
| *scored device time (host-independent)* | — | *1.18452* | *(1.6435, round-3)* | `geomean_scored` |

**Self-consistency check (the point of the 499.6 us row):** at H = 499.638 (probe3 L5, the regime the
round-3 CSVs were taken in) the model reproduces round-3's published all-18 SPAN almost exactly —
**fused 0.74434 vs 0.7443 (0.005 % !) · mask 0.79803 vs 0.8005 (0.3 %)**. This confirms the corrected
model + measured H is faithful to round 3. The **split-6** subset differs more (fused 0.412 vs 0.383,
+7.6 %; mask 0.508 vs 0.439 at fixed-H, or 0.445 at CSV-H, +1.5 %) — exactly the V2 (7.583 %) band,
because the split-case cand spans are model estimates (§4 caveat). The **qualitative** conclusion is
identical under every H.

### 4.2 ⭐ Direct answer: "Is rev3 a regression in real deployment?"

- **In ANY eager deployment (with harness, without msprof, or bare): rev3's split path IS a SPAN
  regression.** all-18 SPAN stays **< 1.0** at every eager H (fused 0.744 → 0.835; mask 0.798 → 0.865)
  and split-6 is far below 1.0 (fused 0.41 → 0.58; mask 0.45 → 0.65). Mechanism (§3): rev3 keeps
  ~200 us/launch and doubles the launch count on split cases, so 2 extra launches × 200–500 us of
  exposed host time (>400 us) dwarf the ~65–80 us of device time saved. **The bare-wrapper case
  (H=200.5) is the floor: even with no guards and no profiler, split-6 SPAN is 0.58 (fused) / 0.65
  (mask) — still a regression.**
- **Under ACL-graph deployment: rev3 is a WIN on SPAN.** all-18 **1.12× (fused) / 1.08× (mask)** and
  split-6 **1.41× / 1.27×**. Capture removes the ~200–500 us/launch host prep, leaving only the
  ~15.6 us/launch device-timeline residual (`_meta.graph_residual_overhead_per_launch_us`), so the
  device-time win finally becomes the executed win.
- **On the contractual cann-bench SCORED metric (sum of device Durations, gaps NOT scored): rev3 is
  1.28× (fused) / 1.18× (mask) regardless of host regime** (`geomean_scored`, matching round-3's
  published 1.2796/1.1840). This is the delivered win and it is host-independent.

> **Precise verdict:** rev3's regression is **口径-specific to eager wall-clock SPAN**, caused by
> launch-count × fixed-host-cost, **not** by any device slowdown and **not** by a heavier wrapper. It
> vanishes under ACL-graph capture and never affected the scored device-time metric.

---

## 5. ACL-graph capture evaluation

### 5.1 Availability (API evidence)
`probe4_aclgraph.json → api_surface`: **NPUGraph / graph / graph_pool_handle / make_graphed_callables
all = true**. API basis (`_meta.api_evidence`): `torch_npu/npu/__init__.py:111-113`;
`npu/graphs.py:327,342,361,372,464,518,532`; repo precedent
`examples/aclgraph/rms_rope_aclgraph.py:270-275`.

### 5.2 Correctness — bit-exact, two independent probes
- `probe4 → wrapper_capture[*]`: all 4 cases `capture_ok=true`, `bitwise_equal_to_eager=true`,
  `per_output_bitwise=[true]×5`.
- `probe6 → {1,3,13,7}` (the **realistic** deployment: static input buffers + D2D copy-in + replay,
  tested across **5 distinct input VALUE sets**): `correctness_all_value_sets_bitwise=true`, every
  `correctness_detail[*].per_output_bitwise=[true]×5`, `all_ok=true`. Because a captured graph bakes in
  device **addresses**, probe6 correctly uses copy-in so the graph reads fresh values — and it is still
  bit-exact on all 5 value sets × 5 outputs × 4 cases.

### 5.3 Benefit
- **Per-launch host cost** (`probe4 → eager_vs_captured`): case1 L0 **202.919 → 5.551 us** (ratio
  0.0274); case3 **198.104 → 2.815** (0.0142); case13 **205.819 → 5.640** (0.0274); case7
  **196.464 → 2.860** (0.0146). Under L5 the ratios are even better (0.0071–0.0124). So **per-launch
  host drops ~30–70× to 2.8–6.6 us**.
- **Marginal cost of an EXTRA launch under graph** (`probe4 → marginal`): `marginal_host_per_extra_launch_L0_us`
  = **0.09 us** (case3), **−0.51 us** (case7); `HEAD_TO_HEAD.captured_serial_chain4_over_chain2` =
  **0.8112** (case3) / **0.3084** (case7) — a 4-launch captured chain is *cheaper* in serial than a
  2-launch one. **Extra launches are essentially free under graph.**
- **End-to-end** (`probe6 → summary`, realistic copy-in+replay): `speedup_L0` (host-async) case1
  **3.587×**, case3 **6.866×**, case13 **3.463×**, case7 **6.774×**; `serial_speedup_L0` (host+device)
  2.629× / 3.819× / 2.598× / 2.531×. `aclgraph_saving_us` (`probe5.summary`) = 402 / 782 / 405 / 788 us.

### 5.4 Cost & limitations (must travel with the recommendation)
- **Copy-in D2D cost** (`probe6 → summary.copy_in_L0_us`): **80.7–88.9 us** per step (case1 88.912,
  case3 80.681, case13 88.701, case7 85.592). This is a *real* added cost of the address-baking
  limitation and is already netted into `speedup_L0` (which still shows 3.5–6.9×).
- **Graph bakes device addresses** ⟹ cann-bench's `input_pool.py` rotates inputs per step, so **naive
  replay would read stale addresses**; the correct pattern is **copy-in + replay** (probe6's `_meta.why`).
  Any graph integration MUST use copy-in mode.
- **One-time capture cost** (`probe4 → wrapper_capture.capture_wall_ms`): **3.5–5.5 ms** per shape
  (case1 5.539, case3 4.115, case13 3.527, case7 3.851). Amortised over many steps, but non-zero.
- **Dynamic shapes need per-shape capture.** R4's dispatch produces different launch counts / geometries
  per (B,V,K_MAX,BV,S); each distinct shape needs its own captured graph. The 18-case suite would need
  up to 18 captures (only 4 were measured — §8).
- **Graph replay under msprof is still instrumented** (`probe4 → L5_harness.replay_span_events` ≈
  160–273 us vs L0 45–212 us): profiling the graph re-inflates the *span* even though the *host replay*
  stays ~13 us. Measure graphs at L0, not under msprof.

---

## 6. ⭐ Adjudication of DESIGN §16.2 option A (4-pass, theoretical ~6×) vs option D (rev3, ~3.1×)

**The two opposing facts (round-3 §13):** device-side `a_launch < 1.540 us` (`final_report_round3.md`
§5, E1) *supports* A; eager host cost (~200–500 us × 2 extra launches) *opposes* A. Round 3 could not
adjudicate without a graph measurement. **That measurement now exists.**

| lens | effect on option A | evidence |
|---|---|---|
| **Device time** | A theoretically ~6× vs D ~3.1× (case7) | DESIGN §16.2.2 table: A "≈69 us + 4L" vs D "≈135 us + 2L" — **model estimate, never measured**; round-3 §4 **falsified** the model's absolute per-case predictions (`F(256)=21.63` not `b_num`-separable, case-4 +34.6 % err) |
| **Eager host** | **Against A**: +2 launches × 200–500 us = +400–1000 us span for ~65 us device saved ⟹ net loss | §3 (`base_vs_cand`), §4.2 (split-6 SPAN <1 at every eager H) |
| **Graph host** | **For A**: marginal host per extra launch ≈ **0–6.6 us**; +2 launches = **~6–13 us** host (or ~31 us device-timeline residual), not ~400–800 us | `probe4.marginal.marginal_host_per_extra_launch_L0_us` = 0.09 / −0.51; `eager_vs_captured.cap_L0_per_launch` 2.8–6.6 us |
| **Bitwise (hard constraint 2)** | **Against A**: A's `m/s` bitwise depends on Pass2 exact prefix-max restore **+ Pass4 exact scalar replay** (two proofs); D's is **constructive** (byte-identical scan) | DESIGN §16.2.2 option-A row + §16.2.2 reason 1 |
| **Complexity** | Against A: 5 kernels + 2 extra workspaces (`m_b`/`s_b`) + Pass4 scalar-replay logic | DESIGN §16.2.2 reason 3 |

### Recommendation: **DO NOT implement option A now. Keep D (rev3) as delivered.**

Reasoning (evidence-based, not preference):
1. **The regime that makes A cheap (ACL graph) is not the delivery target and is not fully verified.**
   cann-bench scores *device time*, where D already delivers 1.28×/1.18×. Graph is verified on only
   **4 of 18** cases (§8) and carries copy-in (80–89 us), address-baking, per-shape-capture, and
   3.5–5.5 ms one-time costs. Betting a rewrite on an unverified deployment regime is premature.
2. **A's device advantage is an unmeasured model estimate that round-3 §4 showed is unreliable at the
   per-case absolute level.** The "~6×" is not evidence; it is the same `F`-based model that missed
   case-4 by +34.6 %.
3. **A carries real bitwise risk against hard constraint 2** that D avoids *constructively*. Lowering
   proof burden outranks a ~2× device gain that is itself unproven (DESIGN §16.2.2 reason 1).

**What must be measured before A is reconsidered** (preconditions, in order):
1. A **device-time-only msprof A/B** of a 4-pass K1 vs D's K1A+K1B on case7/case3 — does A actually hit
   ~69 us vs D's ~135 us, or is the ~6× a model artifact? (This is the *only* number that can justify A.)
2. A **bitwise prototype** of Pass2/Pass4 exact scalar replay (proves/disproves the constraint-2 risk).
3. A **commitment to graph-captured deployment** as the delivery target + a full-18-case graph
   verification (removes the "extra launches are cheap" precondition's uncertainty).

Only if all three hold does A become a justified **incremental** upgrade — which is exactly how
DESIGN §16.2.2 reason 5 already framed it ("A can be a follow-on increment on D; D is a strict subset
of A"). Until then, **D is the correct delivery and A stays "evaluated, theoretically faster, not
adopted".**

---

## 7. Mitigation list (expected saving · risk · kernel/contract change?)

| # | mitigation | expected saving | risk | change kernel/contract? | verdict |
|---|---|---|---|---|---|
| **M1** | **ACL-graph capture + copy-in** | **30–70× per-launch host** (200–500 → 2.8–6.6 us); end-to-end `speedup_L0` 3.5–6.9× (`probe6.summary`); flips split-6 SPAN to 1.27–1.41× (§4.1) | copy-in D2D 80–89 us/step; address-baking ⟹ must use copy-in for `input_pool.py`; 3.5–5.5 ms one-time capture; per-shape capture; only 4/18 verified | **No kernel change** (dispatch is pure host int arithmetic *precisely* so it survives capture, DESIGN §5.6); **integration/contract change** in the caller | ⭐ **Highest value. The only mitigation that removes the root cost.** Bit-exact (probe4+probe6). |
| **M2** | Framework: lazy `Analyzer` (skip when dyn map empty) + cached stream | up to **−95.9 %** of bare launch (K1A 149.79→6.14 us; K1 188.91→42.54 us) (`probe5.B_framework_ceiling`) | requires **tilelang framework** change, not ours; would benefit every tilelang kernel | **No** (out of scope for this operator) | Largest theoretical lever but **not actionable inside R4**; graph (M1) achieves the same end without a framework patch. |
| **M3** | `_r4_split_plan` result memoisation | 4.0–10.16 us **per call** (`probe5.C_wrapper_savings.*._r4_split_plan`) | very low; cache key = shape tuple | Small **kernel/wrapper** change | Marginal (part of C3, <2 %). Low priority. |
| **M4** | GM workspace pre-allocate + reuse | 5.54–5.57 us per call, but **applies only to split cases** (`applies_to_this_case=false` for cases 1/13) | low; must guarantee no aliasing across steps | Small **kernel/wrapper** change | Marginal (part of C3). Low priority. |
| **M5** | **Reduce launch count** (merge K1A+K1B back into one kernel) | would remove 1–2 × ~200 us/launch under eager | 🔴 **blocked**: merging needs grid-level synchronisation; DESIGN §2.5 verified **Developer mode provides no grid barrier/semaphore** (`T.barrier_all` is PIPE-level; `ascend_auto_cross_core_sync` is AIC↔AIV; `atomic_add` unordered/add-only; self-made atomic-spin risks **inter-wave deadlock**) | Would require kernel rewrite + Expert mode | **Not feasible in Developer mode.** This is *why* rev3 is multi-launch. M1 (graph) is the correct way to neutralise the launch-count cost. |
| **M6** | Wrapper slimming (pre-checks/shape-dtype memo) | ≈0.0–0.03 us (`probe5.…pre-checks`) | negligible | Small wrapper change | **Not worth it** — C3 is <2 %; `probe5.summary.C_pct` 0.97–1.97 %. |

**Priority: M1 (graph) ≫ M2 (framework, out of scope) > M3/M4 (marginal) > M6 (negligible); M5 blocked.**
The quantified evidence is unambiguous: optimising our own wrapper (M3/M4/M6, all of C3) buys <2 %,
while graph (M1) buys 30–70× on the very cost that dominates. **Effort should go to graph integration,
not wrapper micro-optimisation.**

---

## 8. Unverified items & information gaps (as-is)

1. **`corrected_span` spans are MODEL ESTIMATES, not measurements** (`_meta.note`). Only the anchors are
   measured: round-3 msprof spans (V1) and probe3 NPU-event spans (V2). The fixed-H rows (499.6/433.9)
   in §4.1 were recomputed here from the JSON's `durs_base/durs_cand` via the documented `model_span`
   (arithmetic on already-measured device times — **no re-measurement**); they reproduce the stored
   `geomean_span_bare`/`geomean_span_graph` exactly and round-3's published all-18 fused SPAN to 0.005 %.
2. **V2 error 7.583 % is not cosmetic.** It means spans extrapolated to H_bare (and the split-6 subset
   at any H) carry ~±7.6 % uncertainty, because the **cand-arm split spans are model outputs with no
   direct measured anchor** (V1 anchors base spans only). The all-18 geomean is far tighter (fused
   0.005 %) because it is dominated by 12 non-split cases pinned at ~1.0. **The split-6 SPAN numbers
   (0.41/0.51) should be read as "≈0.4–0.5, definitely <1", not to 3 decimals.**
3. **C5 (ACL launch) is a residual, not isolated.** No probe measured `lib.call`/ACL launch alone; C5 is
   `measured_prep − model_prep` and may include unmodelled FFI marshalling (§1 caveat).
4. **`probe5.probe_guard_relax.py` does not launch the harness's `CannBenchCacheClean` kernel per step**
   (carried from round-3 §14 item 8), so its absolute us may read slightly optimistic; only ratios used.
   The M3/M4 savings (C3) come from `probe5.C_wrapper_savings` measured on the delivered wrapper path.
5. **Graph verified on only 4 of 18 cases** (1/3/13/7). The other 14 — including the largest device-bound
   cases (15/16/18, where K1 alone runs 390–860 us) and the guard-marginal cases 6/11 — were **not**
   graph-captured. `corrected_span`'s graph SPAN for those cases is a model extrapolation
   (`scored + n_launch·G`), not a measurement. A full-18-case graph A/B is the natural next measurement.
6. **Graph residual G = 15.588 us/launch is derived from 4 samples** (`_meta.graph_residual_overhead_samples`:
   case3 chain2/chain4, case7 chain2/chain4), median of per-launch `(captured_serial − scored_device)`.
   Small sample; used uniformly across all 18 cases in the graph SPAN model.
7. **`a_launch` is bounded, not pinned** (round-3 §5): E1 `<1.540 us`, E2 intercept ≈0, but no estimator
   isolates a literally-zero-work launch. Downstream uses the conservative `≤3.100 us`. This does not
   affect any conclusion here (the §6 adjudication turns on host cost and bitwise risk, not on `a_launch`).
8. **C2 internals not sub-decomposed.** The 233.4 us/launch of harness guards is attributed to
   TorchFunctionMode/TorchOpGuard + TorchDispatchMode/DeviceResidencyGuard + per-step record_function
   *as a block* (probe3 L0→L3, probe7 reproduction). probe3's per-level deltas show L0→L1 = +165.6 us
   (TorchFunctionMode) and L1→L2 = +89.6 us (TorchDispatchMode) for case1, but the split was not
   confirmed across all cases, and it is harness code we do not own — so no finer attribution was pursued.
9. **`reconcile`'s CSV-implied H (403.85 us) is a device-gap-derived lower bound**, not a host wall
   measurement; it excludes any host cost hidden by device overlap and is truncated at device-bound
   cases (`H_validity_filter.threshold_gap_us` = 50). The directly-measured host wall is L5 = 499.6 us.
   Both are reported; they bracket the true per-launch host cost of the harness regime.
10. **No mitigation was implemented.** Per the read-only charter, M1–M6 are quantified recommendations
    only. In particular M1 (ACL graph) is proven feasible + bit-exact + beneficial on 4 cases, but **no
    graph integration was added to the delivered kernel/caller** and the kernel sha256 is unchanged.

---

## 9. Artefacts & delivery-integrity evidence

| item | value | source |
|---|---|---|
| kernel `r4_fused_sampling.py` sha256 | **`a764e117ce150d8677096c2e00bde3db3ebc3ec16aba63e4c613d8dffdfbd117`** | `sha256_before.txt` == `sha256_after.txt`, re-verified live this session |
| test `test_r4_fused_sampling.py` sha256 | `cea98cbd117f23bb0bf50dbfe0c043dcb46cf4c0489d656b8073cc115265cb7` | same, unchanged |
| `proto.yaml` sha256 | `41ab797002568e89fcc9a9eaa58be4f1cdfce20b6406abd0fdbf2ad11f7f8949` | same, unchanged |
| every probe's `_meta.delivered_kernel_sha256` | `a764e117…` | probe2/2b/3/4/5/6/7 `_meta` — the kernel was already frozen during measurement |

**This report only synthesises the existing probes**; the sole computation performed was re-deriving the
fixed-H (499.6/433.9) SPAN geomeans in §4.1 from `corrected_span.json`'s already-measured device
durations via the documented `model_span` — pure arithmetic, no NPU, no profiler, no re-measurement.
`.orchestrator_state.json` was **never read or written**. No `*.py` deliverable, proto, or case file was
touched.

**Data provenance (all under `round3_host_overhead/`):** `probe1_host_segments.json` (wrapper segment
decomposition × 3 msprof configs, `_c1_isolation`) · `probe2_launch_anatomy.json` (single-launch anatomy
+ C4 `controls`) · `probe2b_confirm_analyzer.json` (Analyzer/simplify/tvm_ffi/torch_npu, n=500) ·
`probe3_ladder_and_span.json` (L0–L5 ladder, `base_vs_cand`, `inflation`) · `probe4_aclgraph.json`
(capture feasibility/correctness/replay/`eager_vs_captured`/`marginal`) · `probe5_mitigations.json`
(`B_framework_ceiling`, `C_wrapper_savings`, `summary`) · `probe6_graph_realistic.json` (copy-in+replay,
5 value-sets bitwise) · `probe7_faithful_harness.json` (faithful cann-bench step loop, self-validation) ·
`corrected_span.json` + `analyze_corrected_span.py` (SPAN model, V1/V2) · `reconcile_with_round3_csv.json`
(CSV-implied H vs probe3). Device times authoritative from `../round3_splitv_perf/` msprof CSVs.