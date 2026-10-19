# R4 Stage 3 — Round 3 Optimization Log

**Round nature: VERIFICATION, not exploration.** rev3's split-V architecture was implemented and
precision-verified in Stage 2 but **had never been timed** — every number in DESIGN.md §12.4
(1.15×–1.29× fused, 1.11×–1.20× mask, case 7 up to 3.5×–4.2×) was a *model estimate*. This round
measures rev3 for the first time and closes the 5 dangling items (U2/V-S3, U3/V-S8, U4/V-S7,
§11.4 step 11, GATE 1–4). **No new optimization point was sought and none was implemented.**

| | |
|---|---|
| baseline kernel (pre-rev3) | `history_version/r4_fused_sampling_impl_s2r3_attempt1_base.py`, sha256 `19e51e8d8ee126b920cf48c915d863a9be3a349e2c68cf728d35d482b48bcc57` |
| candidate kernel (rev3) | `examples/R4/r4_fused_sampling.py`, sha256 `a764e117ce150d8677096c2e00bde3db3ebc3ec16aba63e4c613d8dffdfbd117` |
| kernel code changed this round | **NO** — sha256 identical before and after every activity (`gate_sha256_before.txt` == `gate_sha256_after.txt`) |
| test file / proto.yaml / cases changed | **NO** — `cea988cb…` / `41ab7970…` unchanged |
| hardware | Ascend910_9362 (910b2), `cube_core_num=20` **probed**, `vector_core_num=40` **probed** |
| software | CANN 9.2.0 / torch_npu 2.7.1.post4 / python 3.12.9 / tilelang `0.1.4+d75300ed` |
| programming mode | Developer (pure Vector) + `T.tile.topk` hybrid; `pass_configs=R4_PASS_CONFIGS` (incl. `tl.ascend_memory_planning`) |

`[RESULT-SCOPE]` Round 3 changed **zero** lines of kernel, test, proto or case data. All 6 tasks
are measurement/verification tasks. The pre-authorised case-8 guard retreat was **not** needed
(see §4) and was **not** used.

---

## 1. Measurement setup

### 1.1 口径

All performance numbers come from **msprof** `kernel_details.csv` via cann-bench
`run_evaluation.sh` (`warmup=3 repeat=5`, `elapsed_us` = median over the 5 active steps of the
**sum** of `main_kernel` `Duration(us)` inside the step). **No Python/torch timing was used
anywhere in this round.** JIT wall-clock appearing in gate logs (`(0.0s)`, `(5.7s)`) was never
treated as a kernel duration; compile cost is disclosed separately in §10.

`[RESULT-KOU-JING]` The sum-of-Durations identity was re-verified on the round-3 data:
`ksplit_fine.py` per-case `total_us` reproduces the report's `elapsed_us` **exactly** for all
18 cases × 2 tasks (e.g. cand fused case 7 `42.440+97.100+36.640+2.840 = 179.060` = report
`179.06`; case 15 `859.400+611.980 = 1470.760` = report `1470.76`).

### 1.2 Same-session 3-pass A/B (drift control)

Round 2 #4 measured **0.3%–1.7% session drift** on a byte-identical control, so a cross-session
baseline is unusable. Round 3 therefore ran **three** passes in one session, bracketing the
candidate:

```
pass order:  base1  ->  cand  ->  base2
             19e51e8d   a764e117   19e51e8d   (base2 == base1 code => pure drift control)
```

| pass | task | report | exit | wall | 18-case |
|---|---|---|---|---|---|
| base1 | R4FusedSampling | `cann_performance_eval_20261011_150336.json` | **0** | 123 s | 18/18 pass, 0 fail |
| base1 | R4MaskSample | `cann_performance_eval_20261011_150535.json` | **0** | 119 s | 18/18 pass, 0 fail |
| cand | R4FusedSampling | `cann_performance_eval_20261011_150737.json` | **0** | 124 s | 18/18 pass, 0 fail |
| cand | R4MaskSample | `cann_performance_eval_20261011_150934.json` | **0** | 116 s | 18/18 pass, 0 fail |
| base2 | R4FusedSampling | `cann_performance_eval_20261011_151140.json` | **0** | 125 s | 18/18 pass, 0 fail |
| base2 | R4MaskSample | `cann_performance_eval_20261011_151335.json` | **0** | 116 s | 18/18 pass, 0 fail |

`[RESULT-AB-SETUP]` 6/6 evals `EXIT_CODE=0`, 6/6 reports 18/18 pass 0 fail. Every arm swap was
sha256-verified before the run and the candidate was sha256-verified after the restore
(`ab_all.sh` output: `restore OK`).

### 1.3 Cache symmetry (compile time kept out of the timing window)

`prewarm.py` was run for **both** arms against the default `~/.tilelang/cache` *before* any timing
pass, and it doubles as the compile-cost census:

| arm | specializations | cold (>1 s) | new cache entries | total wall |
|---|---|---|---|---|
| cand `a764e117` | 90 | 21 | 21 | 111.44 s |
| base `19e51e8d` | 72 | **2** | **2** | 15.89 s |

`[RESULT-CACHE-SYMMETRY]` Both arms were fully cache-warm for all 6 timing evals, so **no JIT
compile happened inside a measurement window**.

`[RESULT-CACHE-SHARED-KEY]` The base arm found only **2 of 72** specializations cold, and both are
`r4_topk_stats` for **cases 12 and 17** — exactly the two shapes the candidate compiles as
`K1A/K1B` instead of `K1` because they take the split path. Every other entry hit the **same cache
directory** as the candidate. This is direct evidence that the 3 frozen kernels produce **identical
JIT cache keys in both arms**, i.e. the 12 non-split cases load *literally the same binaries*.
All 21 cold compiles in the candidate arm were also **frozen** kernels (K1/K2g/K2m) whose shapes
the shared cache had never seen; **every one of the 4 new rev3 split kernels was already warm**
from Stage 2 (0.02–0.13 s).

### 1.4 Dispatch actually taken (runtime, not predicted)

`_r4_split_plan()` at `cube_core_num=20` (probed) gives split = **{3, 4, 7, 8, 12, 17}**, 6/18,
matching DESIGN §13.4 exactly. `ksplit_fine.py` confirms the runtime launch counts agree:
`split=6 current=12` in both candidate CSVs, `split=0 current=18` in all four base CSVs.

`[RESULT-DISPATCH]` Runtime dispatch == DESIGN §13.4 == `_r4_split_plan` prediction, 6/18 split,
verified from the msprof launch counts rather than inferred.

---

## 2. Task item (1) — same-session A/B performance

Full table: `round3_splitv_perf/ab_3pass_comparison.txt` / `.json`.

### 2.1 The 6 split cases (per-case measured speedup)

| case | B | V | K_MAX | BV | S | **fused base1→cand (us)** | **fused ×** | model §12.4 | err | **mask base1→cand (us)** | **mask ×** | model | err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 8 | 16384 | 256 | 4096 | 4 | 100.30 → 57.52 | **1.744** | 1.490 | +17.0% | 70.60 → 50.92 | **1.386** | 1.320 | +5.0% |
| 4 | 4 | 16384 | 256 | 4096 | 4 | 85.50 → 41.80 | **2.045** | 1.520 | +34.6% | 55.82 → 34.20 | **1.632** | 1.310 | +24.6% |
| 7 | 4 | 152064 | 256 | 4096 | 10 | 728.42 → 179.06 | **4.068** | 3.890 | +4.6% | 461.14 → 172.86 | **2.668** | 2.570 | +3.8% |
| 8 | 4 | 16384 | 128 | 2048 | 8 | 91.22 → 43.04 | **2.119** | 1.870 | +13.3% | 59.94 → 37.88 | **1.582** | 1.570 | +0.8% |
| 12 | 16 | 32768 | 256 | 4096 | 2 | 160.98 → 108.18 | **1.488** | 1.440 | +3.3% | 103.64 → 73.92 | **1.402** | 1.310 | +7.0% |
| 17 | 16 | 152377 | 256 | 4096 | 2 | 736.42 → 437.94 | **1.682** | 1.720 | −2.2% | 472.78 → 321.32 | **1.471** | 1.540 | −4.5% |
| | | | | | **geomean** | | **2.0624** | 1.855 | +11.2% | | **1.6435** | 1.554 | +5.8% |

`[RESULT-SPLIT6]` All 6 split cases improved on both tasks: fused **−32.80% … −75.42%**
(1.488×–4.068×), mask **−27.88% … −62.51%** (1.386×–2.668×). **0 regressions**, best case
case 7 fused **4.068×**.

### 2.2 The 12 non-split cases — the "must be exactly 1.000×" check

These run **byte-identical device code**, so any deviation beyond drift means unexpected coupling.

| case | fused b1 | fused cand | fused b2 | c/b1 % | b2/b1 % (drift) | mask b1 | mask cand | mask b2 | c/b1 % | b2/b1 % (drift) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 35.30 | 34.38 | 34.44 | −2.61 | **−2.44** | 26.44 | 26.10 | 26.60 | −1.29 | +0.61 |
| 2 | 42.12 | 41.98 | 42.10 | −0.33 | −0.05 | 33.96 | 33.98 | 33.78 | +0.06 | −0.53 |
| 5 | 55.10 | 54.08 | 54.76 | −1.85 | −0.62 | 40.24 | 39.76 | 40.24 | −1.19 | +0.00 |
| 6 | 77.98 | 77.16 | 77.36 | −1.05 | −0.80 | 55.80 | 54.78 | 55.92 | −1.83 | +0.22 |
| 9 | 85.98 | 85.66 | 85.98 | −0.37 | +0.00 | 69.28 | 69.58 | 69.66 | +0.43 | +0.55 |
| 10 | 42.50 | 42.18 | 42.38 | −0.75 | −0.28 | 34.28 | 34.08 | 33.96 | −0.58 | −0.93 |
| 11 | 62.96 | 62.92 | 62.80 | −0.06 | −0.25 | 47.18 | 46.96 | 47.12 | −0.47 | −0.13 |
| 13 | 23.62 | 23.06 | 23.24 | −2.37 | **−1.61** | 14.84 | 14.64 | 14.66 | −1.35 | −1.21 |
| 14 | 37.44 | 37.48 | 37.62 | +0.11 | +0.48 | 29.12 | 29.18 | 29.44 | +0.21 | +1.10 |
| 15 | 1470.62 | 1470.76 | 1472.28 | +0.01 | +0.11 | 933.10 | 933.98 | 932.72 | +0.09 | −0.04 |
| 16 | 695.62 | 695.66 | 695.36 | +0.01 | −0.04 | 466.70 | 466.62 | 467.06 | −0.02 | +0.08 |
| 18 | 650.66 | 650.00 | 650.76 | −0.10 | +0.02 | 423.78 | 423.96 | 424.12 | +0.04 | +0.08 |
| **geomean** | | | | **1.0079×** | **−0.324%** | | | | **1.0050×** | **−0.036%** |

Three-way consistency on identical device code:

| pairwise geomean (12 non-split) | fused | mask |
|---|---|---|
| cand / base1 | 0.992139 (−0.786%) | 0.995069 (−0.493%) |
| cand / base2 | 0.996713 (−0.329%) | 0.995270 (−0.473%) |
| **base2 / base1 (identical-code control)** | **0.995411 (−0.459%)** | **0.999798 (−0.020%)** |

`[RESULT-NONSPLIT-12]` **No unexpected coupling.** Measured 1.0079× (fused) / 1.0050× (mask),
i.e. **0.5–0.8% inside the drift band**, worst single case **+0.433%** (mask case 9), best
**−2.606%** (fused case 1) — and the identical-code control moved **−2.436%** on that very same
case, so the largest non-split "gain" is fully accounted for by drift. **0 of 24** non-split
measurements fall outside ±3%. Structural proof backing the statistics:
(i) shared JIT cache entries ⇒ same loaded binaries (§1.3);
(ii) `ksplit_fine.py` reports **2 launches with `Block Num == [b_num, b_num]`** for all 12
non-split cases in **both** arms, `anomaly_rows=0/144` steps in all 6 CSVs.
The residual is measurement-level, not code-level. *Not* exactly 1.000× to 3 decimals — the honest
statement is "1.000× within a ±0.5% session drift that a byte-identical control reproduces".

### 2.3 Geomeans and score

| metric | fused | mask | 36-case combined |
|---|---|---|---|
| **geomean speedup, all 18** | **1.2796×** | **1.1840×** | **1.2309×** |
| geomean speedup, split 6 | **2.0624×** | **1.6435×** | 1.8411× (12 meas.) |
| geomean speedup, non-split 12 | 1.0079× | 1.0050× | 1.0064× (24 meas.) |
| drift-adjusted (vs base1/base2 midpoint), all 18 | 1.2775× | 1.1838× | — |
| drift control geomean(base2/base1) | −0.324% | −0.036% | −0.180% |
| cann-bench `score` base1 → cand | 59.2116 → **60.7629** (+2.620%) | 58.4226 → **59.2498** (+1.416%) | — |
| worst single-case delta | −75.42% (best) / +0.107% (worst) | −62.51% / +0.433% | +0.433% |
| cases regressing >3% | **0** | **0** | **0** |
| `mismatch_count` total / acc fails | **0 / 0** | **0 / 0** | **0 / 0** |

`[RESULT-GEOMEAN]` 36-case combined geomean **1.2309×**; fused **1.2796×**; mask **1.1840×**.
Zero regressions >3%, zero accuracy failures, zero mismatches over all 36 measurements.

### 2.4 Measured vs DESIGN §12.4 model

| quantity | §12.4 prediction (band over `a_launch ∈ [0,8] us`) | **measured** | verdict |
|---|---|---|---|
| fused 18-case geomean | 1.151× … 1.292× | **1.2796×** | **INSIDE** (upper half) |
| mask 18-case geomean | 1.107× … 1.196× | **1.1840×** | **INSIDE** (upper half) |
| case 7 fused | 3.52× … 4.15× | **4.068×** | **INSIDE** (near top) |
| case 7 mask | 2.43× … 2.66× | **2.668×** | **marginally ABOVE** (+0.3% over the `a_launch=0` bound) |
| split-6 geomean fused @`a_launch=3us` | 1.855× | **2.0624×** | +11.2% (model conservative) |
| split-6 geomean mask @`a_launch=3us` | 1.554× | **1.6435×** | +5.8% (model conservative) |

`[RESULT-MODEL-VERIFY]` **The §12.4 model is validated.** Both task geomeans land **inside** the
predicted band, and both sit in the band's *upper* half — which is exactly what a真 `a_launch`
near 0 rather than the assumed 3.0 us implies (§3). Case 7 mask overshoots the `a_launch=0`
bound by 0.3%, i.e. it is at the model's optimistic edge, not beyond it in any meaningful sense.

**Where the model was wrong (per case, and why):**

| deviation | measured vs model | root cause (evidenced) |
|---|---|---|
| **case 4 over-predicted gain** — fused 2.045× vs 1.520× (+34.6%), mask 1.632× vs 1.310× (+24.6%) | largest single model error | The model charges `F(256)=21.63 us/row` to `K1B`. Measured `K1B` for case 4 is **18.94 us** — *less than F alone* — while case 3's `K1B` (same V/K/BV/S/NBS, only `b_num` 4 vs 2) is **32.94 us**. So `F` is **not** a per-row constant independent of `b_num`; §12.2 calibrated it on the single-kernel K1 where the two effects were confounded. The same 14 us `b_num` gap exists in the **base** arm (`K1` 65.62 vs 51.28), so it is a pre-existing property of the kernel, not a rev3 artefact. |
| **case 3/8/12 gains under-predicted** by +17.0/+13.3/+3.3% (fused) | same `F` mis-attribution, smaller because `n_blk_full` is larger | ditto |
| **case 17 slightly over-predicted** (−2.2% fused, −4.5% mask) | the only case where the model was *pessimistic* in the safe direction | case 17 has `b_num=8` ⇒ `S_cap = 20//8 = 2`, so `S` is capped at 2 and each split still walks `NBS=19` blocks serially: `K1A = 183.84 us ≈ 19 × 9.68 us/block`. The one-wave constraint (theorem P2) genuinely limits this case, and the model captured that correctly. |
| `t(4096,256)=8.902 us/block` | **confirmed** | `K1A` per-block: case 7 `42.44/4 = 10.61`, case 12 `40.70/4 = 10.18`, case 17 `183.84/19 = 9.68` — all slightly above 8.902 because K1A also carries the GM read and the partial merge. Consistent, no anomaly. |

---

## 3. Task item (2) — U2 / V-S3 closed: `a_launch` measured

`a_launch` was the **only unmeasured parameter** in the §12.4 model; the shipped guard uses the
conservative `_SPLIT_A_LAUNCH_US = 8.0`, which Stage 2 did not relax.

Because cann-bench scores the **sum of device Durations** (inter-kernel gaps are *not* in the sum),
the `a_launch` the guard must model is the **device-side duration floor of a launch**, not host
launch latency. Three estimators (`round3_splitv_perf/a_launch_analysis.txt`):

**E1 — model-free floor (primary).** rev3's `K2gB` (`r4_gumbel_sample_splitB(B,S)`) has **UB = 132 B**
and its entire useful work is folding `S ≤ 10` partial `(m,s,idx)` triples per row, so its Duration
is launch + a little real work ⇒ an upper bound on `a_launch`.

| case | b_num | S | K2gB Block Num | **K2gB Duration (us)** |
|---|---|---|---|---|
| 4 | 2 | 4 | 2 | **1.540** ← min |
| 3 | 4 | 4 | 4 | 2.080 |
| 8 | 2 | 8 | 2 | 2.580 |
| 7 | 2 | 10 | 2 | 2.840 ← median |
| 12 | 8 | 2 | 8 | 2.980 |
| 17 | 8 | 2 | 8 | **3.100** ← max |

**E2 — least-squares decomposition** `K2gB = a0 + a1·b_num + a2·S` over those 6 points:
`a0 = −0.050 us`, `a1 = +0.326 us/b_num`, `a2 = +0.231 us/S`, max |residual| `0.128 us`.
The intercept is **≈0**: K2gB's duration is essentially *all* attributable to `b_num` and `S` work
terms, leaving no measurable constant launch floor in the device-Duration口径.

**E3 — §12.4 model residual: reported but rejected.** Solving the piecewise model for `a_launch`
over 24 (case, kernel) residuals gives **+0.206 … +14.702 us with 8 of 24 NEGATIVE** (e.g. case 4
`K1B` implies −13.79 us). E3 is internally inconsistent for the same reason the case-4 prediction
was off (§2.4): `F(256)=21.63` does not transfer from the single-kernel K1 to K1B. **E3 is recorded
as evidence against the model's F/p separability, not as an `a_launch` value.**

`[RESULT-A-LAUNCH]` **`a_launch < 1.540 us`** (E1 minimum; K2gB still does nonzero fold work, so
the true floor is below it), with E2's intercept consistent with **≈0**. A conservative planning
value is **≤ 3.100 us = 38.8% of the shipped 8.0 us default**. The measured range `[0, 3.1] us`
lies **entirely inside** the `[0, 8] us` interval over which DESIGN §12.4 proved zero regression —
and the measurement confirms that proof empirically: **0 of 36 cases regressed >3%**.

### 3.1 Guard relaxation: analysis only (NOT changed — needs Orchestrator approval)

`_SPLIT_A_LAUNCH_US` is read **only inside the cost guard, after** `S/NBS/S_TAIL` are derived, so
relaxing it flips the `use_split` **boolean** and cannot change any geometry. Scan
(`a_launch_analysis.json`, re-implementation self-checked to reproduce the delivered
`_r4_split_plan` **exactly for 18/18 cases** at the shipped default):

| `a_launch` | split set | n | newly added |
|---|---|---|---|
| **8.00 (shipped)** | {3,4,7,8,12,17} | 6 | — |
| 4.00 | {3,4,7,8,12,17} | 6 | — |
| **3.20** | {3,4,7,8,**11**,12,17} | 7 | **case 11** |
| **2.74 / 1.54 / 1.0 / 0.5 / 0.0** | {3,4,**6**,7,8,**11**,12,17} | **8** | **cases 6, 11** |

**The split set saturates at 8 for any `a_launch ≤ 2.74 us`.** The other 10 non-split cases are
blocked **structurally**, not by the cost guard, so *no* relaxation can ever reach them:
`S_cap = 20//b_num = 0` for cases 9/15/16/18 (`b_num=32/32/64/32`) and `=1` for case 14
(`b_num=16`); `n_blk_total = 1` for cases 1/2/10; `n_blk_full < 2` for cases 5/13.

Only 2 cases are guard-marginal, and both barely:

| case | shape | S | `pred_saving` | threshold @8.0 | @3.20 | @1.54 | verdict |
|---|---|---|---|---|---|---|---|
| 6 | B=8 V=10240 K=256 BV=4096 (TAIL=2048) | 3 | 8.902 us | 16.383 ✗ | 9.183 ✗ (margin −0.281) | 6.693 ✓ (+2.209) | flips only below ~3.0 us |
| 11 | B=8 V=8192 K=256 BV=4096 | 2 | 8.902 us | 14.922 ✗ | 7.722 ✓ (+1.180) | 5.232 ✓ (+3.670) | flips below ~4.0 us |

**Expected benefit — MEASURED, not modelled.** `probe_guard_relax.py` monkey-patched
`_SPLIT_A_LAUNCH_US` **in-process only** (delivered file sha256 verified identical before and
after: `kernel_file_unmodified = true`) and profiled both arms with the same msprof recipe:

| case | task | arm A current (us) | arm B split (us) | **speedup** | Block Num (B) | **bitwise A==B** |
|---|---|---|---|---|---|---|
| 6 | fused | 61.380 | 42.500 | **1.4442×** (−30.76%) | [12, 4, 12, 4] | **all 5 outputs EQUAL** |
| 6 | mask | 38.980 | 29.780 | **1.3089×** (−23.60%) | [12, 4, 4] | **all 5 EQUAL** |
| 11 | fused | 46.920 | 37.600 | **1.2479×** (−19.86%) | [8, 4, 8, 4] | **all 5 EQUAL** |
| 11 | mask | 31.520 | 27.320 | **1.1537×** (−13.33%) | [8, 4, 4] | **all 5 EQUAL** |

Projected effect on the published geomeans (probe ratios applied to the harness-measured values):

| metric | shipped guard | **if relaxed to measured `a_launch`** | uplift |
|---|---|---|---|
| fused 18-case geomean | 1.2796× | **1.3222×** | +3.33% |
| mask 18-case geomean | 1.1840× | **1.2114×** | +2.32% |
| 36-case combined | 1.2309× | **1.2656×** | +2.82% |

Supporting evidence that relaxation is low-risk: the two admitted geometries
(`S=3/NBS=1/S_TAIL=2` and `S=2/NBS=1/S_TAIL=-1`) are **already compiled and already statically
asserted clean** in the Stage-2 static gate, which includes `cb6`/`cb11` under `--shapes all`
(`rev3_probes/stage2/static_gate_all.log`: `--- cb6:` K1A UB 106496/196352, K1B 104036/196352,
`padCount_ALL0=True`, `all32B=True`, `dynamic=[]`; `--- cb11:` K1A 90112/196352, K1B 79460/196352,
same; overall `FAILURES: 0`, `STATIC GATE VERDICT: PASS`). Their split-parallel Block Num
(12 and 8) is comfortably ≤ `cube_core_num=20`, so theorem P2 still holds.

`[RESULT-GUARD-RELAX]` **The shipped guard IS over-conservative**: `a_launch` is ≥2.6× smaller than
the 8.0 us default, and relaxing it would add cases 6 and 11 for a measured **1.15×–1.44×** on 4
more measurements (**+2.82%** on the 36-case geomean), with **bitwise-identical outputs**.
**RECOMMENDATION ONLY — not implemented.** Blocking conditions if the Orchestrator wants it:
(i) it is a kernel-code change outside this round's pre-authorisation;
(ii) **4 test-table entries** carry a hardcoded `expect_use_split` that would go stale
(`L1_CASES/l1_tailmid_V`, `L1B_CASES/l1b_tailmid_V`, `STAB_CASES/stab_B7_V4409_bv2048_gumbel`,
`STAB_CASES/stab_B5_V10240_gumbel` — scanned 84 shape-bearing entries, `followups_analysis.json`),
so the test file must change in the same commit; (iii) all 5 gates + the 36-case A/B must be
re-run afterwards.

---

## 4. Task item (3) — U3 / V-S8 closed: case 8 (`K_MAX=128, BV=2048`)

Correctness was proven in Stage 2 (A4/L1 Tier-1+Tier-2 all pass, UB 45056/40036) but the benefit
had never been measured. DESIGN §12.4 warning 3 flagged the case-8 prediction as **optimistic**
because `F(128)` is not separable (R-F128, −15.0% model fit error).

| task | base1 (us) | base2 (us) | cand (us) | **measured speedup** | §12.4 model | model err | `mismatch_count` |
|---|---|---|---|---|---|---|---|
| fused | 91.22 | 91.26 | **43.04** | **2.119×** (−52.82%) | 1.870× | **+13.3%** | 0 |
| mask | 59.94 | 59.76 | **37.88** | **1.582×** (−36.80%) | 1.570× | **+0.8%** | 0 |

Per-kernel decomposition (fused): base `K1=53.980 + K2g=37.280`; cand
`K1A=6.460 + K1B=26.480 + K2gA=7.400 + K2gB=2.580`. Both stages improved: K1 side
`53.980 → 32.940` (1.639×), K2g side `37.280 → 9.980` (3.735×). Block Num `[16, 2, 16, 2]`,
split-parallel 16 ≤ 20 ✓. The two base passes agree to 0.04%, so the effect is far outside drift.

`[RESULT-CASE8]` **KEEP.** Both tasks gain **≫3%** (fused −52.82%, mask −36.80%), so the
pre-authorised retreat (adding `and K_MAX == 256` to the guard) is **not needed and was NOT used**
— `preauthorised_retreat_used = false`, and the kernel sha256 is unchanged. Contrary to §12.4
warning 3, the case-8 prediction was **not** optimistic: mask matches it (+0.8%) and fused
**exceeds** it (+13.3%). R-F128 can be downgraded — on the fused task the `K_MAX=128/BV=2048`
specialization is the **second-best** split case measured (2.119×, behind only case 7's 4.068×),
and on mask it is third (1.582×, behind case 7's 2.668× and case 4's 1.632×).

---

## 5. Task item (4) — U4 / V-S7 closed: runtime `Block Num` + theorem P2

`ksplit_fine.py` gates every step on (i) launch count, (ii) `Block Num` sequence, (iii) `Block Num`
sum, (iv) **theorem P2**: every split-*parallel* kernel's `Block Num ≤ cube_core_num`. Expectations
are **derived from the delivered `_r4_split_plan`** (single source of truth, DESIGN §15.3.1 step 1),
not hardcoded, so the gate cannot drift from the implementation.

| CSV | arm | task | steps attributed | anomaly rows | split / current cases | **P2 gate** | **Block Num gate** |
|---|---|---|---|---|---|---|---|
| `ksplit_base1_fused` | base | fused | 144 | **0** | 0 / 18 | **PASS** | **PASS** |
| `ksplit_base1_mask` | base | mask | 144 | **0** | 0 / 18 | **PASS** | **PASS** |
| `ksplit_cand_fused` | cand | fused | 144 | **0** | **6 / 12** | **PASS** | **PASS** |
| `ksplit_cand_mask` | cand | mask | 144 | **0** | **6 / 12** | **PASS** | **PASS** |
| `ksplit_base2_fused` | base | fused | 144 | **0** | 0 / 18 | **PASS** | **PASS** |
| `ksplit_base2_mask` | base | mask | 144 | **0** | 0 / 18 | **PASS** | **PASS** |

Runtime `Block Num` (candidate arm, median-of-5 steps; full table in `baseline_round3.json`):

| case | S | NBS | S_TAIL | b_num | **fused Block Num** | **mask Block Num** | max split-parallel | ≤ cube=20 |
|---|---|---|---|---|---|---|---|---|
| 3 | 4 | 1 | −1 | 4 | `[16, 4, 16, 4]` | `[16, 4, 4]` | 16 | ✓ |
| 4 | 4 | 1 | −1 | 2 | `[8, 2, 8, 2]` | `[8, 2, 2]` | 8 | ✓ |
| **7** | **10** | **4** | **9** | **2** | **`[20, 2, 20, 2]`** | `[20, 2, 2]` | **20** | ✓ **exactly one full wave** |
| 8 | 8 | 1 | −1 | 2 | `[16, 2, 16, 2]` | `[16, 2, 2]` | 16 | ✓ |
| 12 | 2 | 4 | −1 | 8 | `[16, 8, 16, 8]` | `[16, 8, 8]` | 16 | ✓ |
| 17 | 2 | 19 | 1 | 8 | `[16, 8, 16, 8]` | `[16, 8, 8]` | 16 | ✓ |
| 12 non-split | — | — | — | — | `[b_num, b_num]` | `[b_num, b_num]` | — | ✓ (identical in both arms) |

`[RESULT-BLOCKNUM]` **Runtime `Block Num` matches the plan on 864/864 attributed steps across all
6 CSVs, `anomaly_rows = 0`.** Theorem P2 holds at runtime with **zero violations**; the tightest
point is **case 7 `K1A`/`K2gA` with `Block Num = 20 == cube_core_num = 20`** — exactly one full
wave, as designed.

`[RESULT-CASE7-BLOCKNUM]` **case 7 fused = `[20, 2, 20, 2]`, EXACTLY the 4-row sequence Stage 2's
dry run predicted.** Confirmed from msprof, not from the plan function.

---

## 6. Task item (5) — cann-bench 36/36 end-to-end

The candidate pass of the A/B **is** the DESIGN §11.4 step-11 end-to-end run (both operators, all
18 cases each, through the installed wheel built from the vendored submission copy).

| report | operator | passed | failed | worst `mismatch_count` |
|---|---|---|---|---|
| `cann_correctness_eval_20261011_150635.json` | R4FusedSampling | **18/18** | 0 | **0** |
| `cann_correctness_eval_20261011_150836.json` | R4MaskSample | **18/18** | 0 | **0** |
| `cann_performance_eval_20261011_150737.json` (accuracy section) | R4FusedSampling | 18/18 | 0 | 0 |
| `cann_performance_eval_20261011_150934.json` (accuracy section) | R4MaskSample | 18/18 | 0 | 0 |

`[RESULT-CANNBENCH-36]` **36/36 `mismatch_count = 0`, 36/36 accuracy passed, 0 failed cases.**
Prerequisites confirmed in place: submission copy `cmp`-identical to the delivered kernel, wheel
rebuilt from it (`build.sh` rc=0; the container hash changed `d9f31ae1… → bdfa160b…` **only**
because wheels embed timestamps — the `_r4_kernels.py` *inside* the new wheel was extracted and
verified to be `a764e117…`), `Block Num` gate updated, README deviation recorded.
Both known environment traps were handled: `R4_HTML_SHIM=1` + the sitecustomize shim on
`PYTHONPATH`, and `LD_LIBRARY_PATH` prefixed with `/usr/local/sqlite3/lib`
(all `elapsed_us` values are non-zero, which is the symptom of getting that wrong).

---

## 7. Task item (6) — `gates_report.py` GATE 1–4

`bench/hap_thw_model/gates_report.py`, exit code **0** (`round3_splitv_perf/gates_report_round3.log`):

| GATE | content | result |
|---|---|---|
| **1** model self-consistency | independent re-derivation of published 910b2 `t_hw_us` | **PASS** — softmax 20/20 + arg_max 20/20 |
| **2** bottleneck attribution | 36 (task, case) pairs, `t_vector` vs `t_read`/`t_write`/`t_scalar` | **PASS** — 36/36 reported, all vector-bound |
| **3** ordering sanity | `t_hw ≥ 1.0`; `t_hw == round(max(raw,1.0),4)`; fused ≥ mask per case | **PASS** |
| **4** intensity + achieved fraction | 910b2 ridge point 3.125 elem-op/byte | **PASS** |

`[RESULT-GATES-1-4]` **GATE 1–4 all PASS, exit 0.** `t_hw` was **not** re-derived, per §15.3.2:
the only ops/elem change is the `+S` cross-split merges, worth **+0.15%** on case 7 — far inside
GATE 1's tolerance, and GATE 1 reproduced the published values exactly, confirming the decision.

### 7.1 Roofline attainment change (+ the mandatory 2-pass GM-read口径 note)

`attain% = 100 · t_hw_us / elapsed_us`, same published `t_hw` for both arms
(`followups_analysis.txt`, per-case table in `baseline_round3.json`):

| task | all-18 geomean attainment | split-6 geomean attainment |
|---|---|---|
| fused | base **2.1961%** → cand **2.8101%** (**1.2796×**) | 1.3612% → **2.8073%** (**2.0624×**) |
| mask | base **1.9493%** → cand **2.3080%** (**1.1840×**) | 1.0803% → **1.7754%** (**1.6435×**) |

Headline single case: **fused case 7 attainment 0.672% → 2.733%** (4.068×), i.e. the 36/40 idle
AIV that made case 7 the worst core-starvation case in §12.2(d) is now 40/40 busy on K1A/K2gA.

⚠️ **口径 caveat required by DESIGN §15.3.2 — computed here, not hand-waved.** The rev3 split path
reads `logits` from GM **twice** (K1A for the Sort side, K1B for the softmax side), while the
published `t_hw` charges it **once** (`hap_lib.py:20` unique-bytes / no-reload-factor convention).
Charging the reload (`t_hw_2pass = max(t_cube, 2·t_read, t_write, t_vector)`, same ≥1.0 clip):

* **fused: 0/6 split cases affected.** `2·t_read` stays below `t_vector` everywhere (case 7:
  `2×1.2672 = 2.5344 < 4.8946`), so the published floor is unchanged and all 6 remain
  vector-bound. The fused attainment figures above are **unbiased**.
* **mask: 3/6 split cases affected (7, 12, 17).** The mask task's arithmetic intensity is only
  **~2.57 ops/byte — already BELOW the 3.125 ridge point** (GATE 4), with a vec/read margin as thin
  as 1.062×, so doubling the read flips the attribution to **READ-bound**:

| mask case | `t_hw` → `t_hw_2pass` | bound | attain (published) → attain (2-pass) |
|---|---|---|---|
| 7 | 1.3464 → **2.5345** us (1.882×) | vector → **READ** | 0.779% → **1.466%** |
| 12 | 1.1676 → **2.1847** us (1.871×) | vector → **READ** | 1.579% → **2.955%** |
| 17 | 5.3964 → **10.1587** us (1.882×) | vector → **READ** | 1.679% → **3.162%** |

`[RESULT-ROOFLINE]` Attainment improves by exactly the measured speedup factor (same `t_hw` both
arms): fused **2.1961% → 2.8101%**, mask **1.9493% → 2.3080%**; split-6 fused **1.3612% → 2.8073%**.
**口径 note:** the fused figures are unbiased; the **mask split-case attainment is understated by
~1.87×** if the second GM read is charged, and under that口径 mask cases 7/12/17 become
**READ-bound rather than vector-bound**. Any published mask attainment for a split case must carry
this note. GATE 2's "all vector-bound" verdict is a statement about the *published unique-bytes
model* and remains PASS as-is; it would need re-running if reloads were ever folded into `t_hw`.

---

## 8. Precision gates re-run on the final delivered state

Re-run because the A/B swapped `r4_fused_sampling.py` to BASE twice and back, and rebuilt the
submission wheel three times. sha256 was verified at every swap, but only the gates prove the
*behaviour* is still green (`run_gates_round3.sh`, `gate_exits.txt`):

| gate | command | **exit** | counts measured this round | Stage-2 / Orchestrator reference | match |
|---|---|---|---|---|---|
| static | `--level static` | **0** (10 s) | **38 STATIC_PASS / 0 FAIL** | 38 / 0 | ✓ |
| all | `--level all` | **0** (35 s) | **143 PRECISION_PASS / 0 FAIL**, **56 BOUNDARY_PASS / 6 WARN**, 38 STATIC_PASS | 143 / 56+6WARN / 38 | ✓ |
| stability | `--level stability` | **0** (18 s) | **21 PRECISION_PASS / 0 FAIL**, STAB_split=4 ADV_split=4, **8 ADV all `dropped=0`** | 21 / 0, 8 ADV dropped=0 | ✓ |
| stress | `--level stress` | **0** (23 s) | **12/12**, all `selfdiff=0 goldenfail=0 invariant=0`, 4 split-path configs | 12/12 | ✓ |
| coverage | `coverage_check.py … --proto proto.yaml` | **0** | **25 PASS / 0 MISS / 0 N/A** | 25-0-0 | ✓ |

`grep -cE "_FAIL\]"` = **0** in all four test logs. `sha256_stable_before_after_gates=YES`.

The **6 BOUNDARY_WARN** are the 3 known V-S4 probe names × 2 tasks
(`bnd_split0_all_neginf`, `bnd_split0to4_all_neginf`, `bnd_splitmid_all_neginf`), each carrying
`tier2_bitwise=5/5` and the annotation *"EXPECTED pre-existing degrade, NOT a rev3 regression"*.

`[RESULT-GATES-5]` **5/5 precision gates exit 0 with counts identical to the Stage-2 /
Orchestrator reference (38 · 143+56/6WARN · 21 · 12 · 25-0-0).** No count went down.
Per hard constraint 6 the 6 `-inf` NaN WARNs were **not touched**.

---

## 9. ⚠️ Major honest caveat found while measuring: SCORED口径 vs WALL-CLOCK SPAN

While reading inter-kernel gaps for the `a_launch` analysis (§3) it emerged that the gaps are
**large and asymmetric between the arms**. Since `elapsed_us` is the *sum of Durations*, the gaps
are **not scored** — but omitting them would make rev3 look better end-to-end than it is, so both
口径 are reported (`analyze_span.py`, `span_vs_scored.txt`).

```
scored_us = median over 5 steps of SUM(main_kernel Duration) in the step   == cann-bench elapsed_us
span_us   = median over 5 steps of (last main_kernel end - first main_kernel start)
gap_us    = span_us - scored_us     (host-side idle between launches, inside the step)
```

| task | geomean **SCORED** speedup | geomean **SPAN** speedup | span, launch-count-changed (6 split) | span, unchanged (12) | host gap per launch (base arm) |
|---|---|---|---|---|---|
| fused | **1.2796×** | **0.7443×** | **0.3831×** | 1.0375× | **362.7 us** |
| mask | **1.1840×** | **0.8005×** | **0.4387×** | 1.0814× | **386.2 us** |

Concretely, **case 7 fused**: base `K1=425.04 + gap 3.00 + K2g=303.58` → span **732.00 us**;
cand `K1A=42.44 + 416.00 + K1B=97.10 + 539.50 + K2gA=36.64 + 334.25 + K2gB=2.84 + 304.75` →
span **1360.25 us**. Scored device time **728.42 → 179.06 us (4.068× better)**; span
**1.86× worse**.

**Mechanism (evidenced, not speculated):**
1. The **base arm already pays ~363–386 us of host time per launch** on every case whose first
   kernel is shorter than that overhead — **13 of 18** fused cases show a ~310–392 us gap, and the
   other 5 (7/15/16/17/18) show ~2.8–11.5 us. This is pre-existing and **not caused by rev3**.
2. The gap is nearly **constant** regardless of kernel or shape (the signature of fixed host-side
   Python/ACL launch preparation, not of device work).
3. It **collapses to ~2.8–3.0 us** exactly when a kernel runs long enough for the host to queue the
   next launch asynchronously (base cases 7/15/16/17/18, where K1 alone runs 390–860 us).
4. rev3 replaces **1–2 long kernels with 3–4 short ones**, so on the split cases the overhead that
   used to be hidden becomes exposed.

**No kernel got slower.** Per-stage device time improved on every split case: case 7 fused K1 side
`425.04 → 139.54 us` (3.046×), K2g side `303.58 → 39.48 us` (7.690×). This is a **host-side
software-overhead finding**, and the scored cann-bench metric, the roofline attainment, and the
core-utilisation argument (4/40 → 40/40 AIV) are all device-side and all genuinely improve.

`[RESULT-SPAN-CAVEAT]` **rev3's win is real in device time and in the cann-bench scored metric
(1.2309× over 36 cases, score +2.62% fused / +1.42% mask), but it does NOT carry to eager-mode
end-to-end wall clock under this harness — on the 6 split cases the span geomean is 0.38× (fused)
/ 0.44× (mask), because each extra launch costs ~360 us of host time while the kernels themselves
cost 3–97 us.** The obvious mitigation is ACL-graph capture (the dispatch is pure host Python int
arithmetic precisely so it does not break capture — DESIGN §5.6 audit block 2), under which launch
overhead drops to microseconds and the device-time win would be what actually runs.
**THIS IS UNVERIFIED — no graph-captured measurement was taken this round.** See §12.

---

## 10. JIT compile cost — disclosed separately (hard constraint 7)

Compile time is **excluded from every performance number above** (all are msprof device Durations,
and both arms were cache-warm before timing — §1.3). Disclosed here on its own:

**(a) The 4 NEW rev3 split kernels, cold** (`probe_cold_split_compile.py` against a **fresh empty**
`TILELANG_CACHE_DIR`, so this is what a first-ever run on a cold machine pays):

| | value |
|---|---|
| distinct new specializations | **23** (24 invocations; `K2gB(16,2)` is shared by cases 12 and 17) |
| total cold compile wall | **117.16 s** |
| mean / min / max per specialization | **5.09 s** / 4.49 s / 6.00 s |
| per kernel | K1A **30.05 s** (6 specs) · K1B **34.23 s** (6) · K2gA **30.32 s** (6) · K2gB **22.56 s** (5) |
| delivered file | sha256 unchanged before/after (`kernel_file_unmodified = true`) |

**(b) Warm-cache re-instantiation** (what an already-provisioned machine pays): all 24 split-kernel
instantiations cost **0.02–0.13 s each**; the candidate prewarm's 21 cold compiles (111.44 s total)
were **all frozen kernels** whose shapes the shared cache had never seen, and the base prewarm's
2 cold compiles (15.89 s total) were `K1` for cases 12/17.

**(c) Eval wall clock** (fused, mask) — base1 `123 s, 119 s`; cand `124 s, 116 s`; base2
`125 s, 116 s`. The candidate arm is **not** slower wall-clock than the baseline arm, because both
were warm.

`[RESULT-JIT]` rev3 adds **23 new JIT specializations costing 117.16 s to compile cold**
(mean 5.09 s), disclosed separately and excluded from all performance figures. Warm, they cost
≤0.13 s each.

---

## 11. Adoption decision

| criterion (task §2) | requirement | measured | verdict |
|---|---|---|---|
| split cases | investigate any regression >3% | **0 regressions**; all 12 split measurements improved 27.9%–75.4% | **PASS** |
| case 8 | keep if >3% gain; retreat if >3% loss | **+52.82% / +36.80% faster** | **KEEP**, retreat not used |
| non-split cases | exactly 1.000× inside drift | 1.0079× / 1.0050×, worst +0.433%, best −2.606% vs identical-code control −2.436% | **PASS** (no coupling) |
| cann-bench `mismatch_count` | all 0 | **0 / 36** | **PASS** |
| 5 precision gates | all exit 0, counts ≥ Stage 2 | 0,0,0,0,0 with 38 · 143+56/6W · 21 · 12 · 25-0-0 | **PASS** |
| overall 36-case geomean | any measurable gain | **1.2309×** (fused 1.2796×, mask 1.1840×) | **PASS** |
| performance-antipattern check | none retained silently | see below | **PASS** |

**Performance-antipattern check** (`tilelang-perf-optimization/references/performance-antipatterns.md`):
the round-3 kernel is unchanged, and the antipattern review of the split architecture is
*positive* — no cross-level GM→L0 copy (all transfers are `T.copy` GM↔UB and UB↔UB at 32 B-aligned
counts, re-asserted by static gate assert-④: `padCount_all0=True`, `all32B=True`), no dynamic loop
bounds (`dyn_bounds=[]` on all 6 CSVs' worth of specializations), no serialized scalar chain added
(the `F(256)` `tau_p` epilogue still runs **once**, in K1B, exactly as §12.2 predicted would be the
case). One deliberate retention, recorded here as required: **the split path reads `logits` from GM
twice**, which trades HBM bandwidth for core parallelism. It is retained because the 14 of 18 cases
with `b_num < 20` were core-starved, not bandwidth-limited (roofline attainment 0.67%–5.3%), and
the measurement confirms the trade pays in device time (case 7: 4/40 → 40/40 AIV, 4.068×). Its
cost is quantified in §7.1 rather than hidden.

`[RESULT-ADOPT]` **ADOPT rev3 (`a764e117`) as the new baseline.** Every adoption criterion is met
on measured msprof data; no rollback is warranted. `consecutive_no_improvement` resets to **0**.
The kernel file is **unchanged this round** — adoption here means "rev3 is now *measured* and
confirmed beneficial", not "a new edit was made".

---

## 12. Unverified items and information gaps (as-is)

1. **🔴 Wall-clock / span口径 (§9) is the material open risk.** rev3 improves the scored device-time
   metric 1.2309× but *degrades* the eager-mode step span on the 6 split cases (0.38×/0.44×).
   **Not verified:** whether ACL-graph capture (or any deployment that removes the ~360 us
   per-launch host cost) realises the device-time win end-to-end. No graph-captured measurement was
   taken. If the delivery target is eager-mode wall clock rather than the cann-bench score, this
   must be settled before rev3 is considered a win.
2. **Cause of the ~360 us per-launch host gap not root-caused.** Its *behaviour* is well evidenced
   (constant, pre-existing in the base arm, collapses when a kernel outruns it), but I did not
   profile the host side to attribute it between Python wrapper, tensor allocation, and ACL launch,
   nor establish how much is msprof instrumentation overhead vs. real.
3. **`a_launch` is bounded, not pinned.** E1 gives `< 1.540 us` and E2's intercept is ≈0, but no
   estimator isolates a launch with *literally zero* work. The claim used downstream is the
   conservative `≤ 3.100 us`, which is safe for any guard decision.
4. **§12.4's `F(256) = 21.63 us/row` is falsified as a `b_num`-independent constant** (§2.4, §3 E3).
   Not re-calibrated this round — that needs a dedicated `b_num` sweep. Consequence: the model's
   *absolute* per-case predictions for small-`n_blk` split cases are unreliable (case 4 off by
   +34.6%), though its *geomean bands* held.
5. **Guard relaxation not implemented** (§3.1). The measured +2.82% on the 36-case geomean is
   available but requires Orchestrator approval, a kernel edit, 4 test-table `expect_use_split`
   updates, and a full gate + A/B re-run.
6. **Only 1 drift replicate.** One identical-code control pair (base1/base2) per task. It was
   sufficient to explain every non-split residual, but a tighter bound on session drift would need
   more replicates.
7. **Probe口径 difference.** `probe_guard_relax.py` does not launch the harness's
   `CannBenchCacheClean` kernel per step, so its absolute microseconds may read slightly optimistic;
   both arms were profiled identically, so only the *ratios* (1.15×–1.44×) are used.
8. **Case-6/11 split path lacks runtime precision coverage today.** Bitwise A==B was verified in the
   probe on one input distribution, and the specializations are statically asserted clean, but the
   delivered runtime suite drives those shapes down the *current* path. Relaxing the guard would
   need the tier-1/tier-2/boundary/stress suite to cover them as split shapes.
9. **`t_hw` not re-derived** (deliberate, §15.3.2 + §7). If reloads were ever folded into the model,
   GATE 1's re-derivation and GATE 2's attribution would both need updating — and mask cases 7/12/17
   would flip to READ-bound (§7.1).
10. **No new optimization explored.** By design: this round's charter was verification. Whether
    DESIGN §16.2's option A (4-pass, theoretical ~6× vs rev3's ~3.1×) is worth pursuing as an
    increment is an Orchestrator decision; the measured `a_launch < 1.54 us` **removes the main
    argument against it** (§16.2 reason 2 was "4 launches bet on an unknown `a_launch`"), while
    §9's finding that **launches are expensive in eager mode** is a new argument *against* adding
    two more of them. Those two facts point in opposite directions and cannot both be resolved
    without the graph-capture measurement in item 1.

---

## 13. Stage-3 round-3 item A — `_SPLIT_A_LAUNCH_US` relaxed to the measured value

**Round nature: a single authorised kernel edit + its mandatory test-side配套.** §3.1 recommended the
relaxation but did not implement it (it was outside round 3's pre-authorisation). This item
implements it, with the four配套 items the Orchestrator made mandatory.

| | |
|---|---|
| kernel BEFORE | `a764e117ce150d8677096c2e00bde3db3ebc3ec16aba63e4c613d8dffdfbd117` (1240 lines, `_SPLIT_A_LAUNCH_US = 8.0`) |
| kernel AFTER | `c59a83bcfa5f43365ab921499e2d3a6f9ec54011d861149e50d1beb12079c534` (1280 lines, `_SPLIT_A_LAUNCH_US = 1.540`) |
| backup (rollback baseline) | `history_version/r4_fused_sampling_impl_s3r3_itemA_attempt0.py` == `a764e117…` (verified) |
| test file BEFORE / AFTER | `cea988cb…` (2410 lines) → `dc69ad6fa452dacd548ede486085802520bde8aea7e1fe3311f3d08562732094` (2523 lines). An intermediate state `2f837ab1…` (2522 lines) is what the A/B and the `cand`/`post` gate runs saw; the only difference is a **comment re-wrap** of one 101-char line down to ≤100 (project style), so all three proofs (`verify_no_weakening`, `verify_split_plan_itemA`, `verify_frozen_itemA`) and all 5 gates were **re-run on the final state** (`gate_final_*`) rather than assumed |
| test backup | `history_version/test_r4_fused_sampling_impl_s3r3_itemA_attempt0.py` == `cea988cb…` (verified) |
| proto.yaml | `41ab7970…` **UNCHANGED** (verified) |
| artifacts | `perf_tuning/round3_itemA_guard/` (26 files: probes, A/B raw, gates, proofs, scripts) |
| hardware / software | unchanged from the round-3 header (910b2, `cube_core_num=20` probed) |

`[ORDER-CHECK]` Executed in the mandated order **analyse → tune → re-verify**, nothing skipped:
① baseline 5 gates re-run on `a764e117` **first**, in this session, and reproduced the
Orchestrator's counts exactly (static 38/0 · all 143 PASS/0 FAIL + 56 BOUNDARY_PASS/6 WARN + 38
STATIC · stability 21/0 · stress 12/0 · coverage 25-0-0, all exit 0 — `gate_pre_*`); ② backups
taken and sha-verified **before** any edit; ③ read-only pre-flight (`preflight_itemA.py`) settled
the guard value, the 18-case admission set, the golden recipe for the new cases and the JIT delta
**before** the kernel was touched; ④ the anti-pattern reference was re-checked (§13.6); ⑤ only then
the single-constant edit; ⑥ test配套 in the same change; ⑦ 5 gates re-run on the candidate; ⑧
3-arm same-session device A/B; ⑨ one flagged pair escalated to a dedicated arbitration experiment
rather than argued away; ⑩ gates re-run a **third** time on the final restored state, because the
A/B swapped the kernel file 9 times. No implementation change preceded its analysis, and no
performance claim was made before its measurement.

### 13.1 The chosen value and why

`_SPLIT_A_LAUNCH_US = 1.540` — the **E1 model-free measured floor** from §3 (`K2gB` = the smallest
complete kernel in the rev3 chain, UB = 132 B, min msprof `Duration` over the 6 split fused cases).

| candidate | provenance | admits case 6? | admits case 11? | decision |
|---|---|---|---|---|
| **1.540** | E1 **min** K2gB Duration (measured) | **YES** (thr 3.013) | **YES** (thr 3.987) | **CHOSEN** |
| 2.840 | E1 median K2gB Duration | yes | yes | rejected — a median *duration* still contains K2gB's fold work, so it is not a clean `a_launch` |
| 3.100 | E1 max, §3's "conservative planning value" | **NO** (3.100 > 3.013) | yes | rejected — would not deliver the task objective |
| 2.74 | §3 sweep grid point | yes | yes | rejected — a grid point with no independent measurement meaning |
| 0.0 / −0.050 | E2 least-squares intercept | yes | yes | rejected — §3 itself flags `a0` as a thin-fit (6 pts / 3 params) extrapolation, "a cross-check on E1, not the answer"; 0 would delete launch accounting from the guard |

Reasons 1.540 is the defensible pick, in the code comment as well as here: it is **directly
measured** (not modelled, not a grid point); it is a **strict upper bound** on the true launch floor
because `K2gB` still folds ≤10 partials per row, so `pred_cost` stays *over*-estimated and the guard
keeps erring toward **not** splitting — the safe direction; it is independently corroborated by E2's
near-zero intercept; and it is the **exact value at which the relaxation was A/B measured on device**
(`guard_relax_probe.json`), so the constant matches the evidence instead of being extrapolated from
it. `_SPLIT_SAFETY = 1.5` is **untouched**, so the triple-benefit test still demands
`pred_saving > 1.5 × pred_cost`; case 6's predicted margin is +2.209 us against a *measured* saving
of 19.50 us, i.e. the guard remains conservative relative to reality.

`[IMPL-itemA]` Kernel diff = **1 hunk, +41 / −1 lines**, of which the only non-comment lines are
`-_SPLIT_A_LAUNCH_US = 8.0 …` / `+_SPLIT_A_LAUNCH_US = 1.540` (+40 lines of evidence comment).
Verified by `verify_frozen_itemA.py`: `hunks=1`, `code_added=['_SPLIT_A_LAUNCH_US = 1.540']`,
`code_removed=[the 8.0 line]`, `only_constant_changed=True`. **Zero other kernel lines changed.**

### 13.2 The four配套 items

**(2) The 4 stale split-expectation entries.** Independently re-derived rather than trusted:
`scan_shape_entries.py` walks the test AST, picks up every tuple carrying a plausible
`(B, V, K_MAX, BV)` window (**92** entries found vs round 3's 84 — my window heuristic is more
permissive and also catches the bare `ADV_CASES` 4-tuples), and evaluates the *delivered*
`_r4_split_plan` at both guard values. It reproduces round 3's answer **exactly — the same 4
entries flip**, and 0 flip at `a=3.20`:

| # | table | entry | shape | plan @8.0 → @1.540 | diff |
|---|---|---|---|---|---|
| 1 | `L1_CASES` | `l1_tailmid_V` | B=8 V=10240 K=256 BV=4096 | `(False,3,1,2)` → `(True,3,1,2)` | tags `["D-SHAPE-TAIL-MID"]` → `["D-PATH-SPLIT","D-SPLIT-NBS1","D-SPLIT-TAILOWNER","D-SHAPE-TAIL-MID"]` + a 7-line comment stating the path change, the guard arithmetic (8.902 > 6.693 vs the old 16.383), and where the Tier-2 gate for this shape now lives |
| 2 | `L1B_CASES` | `l1b_tailmid_V` | same shape, `FUSE_GUMBEL=True` | same flip | tags `["D-PARAM-FUSE_GUMBEL","D-SHAPE-TAIL-MID"]` → `["D-PATH-SPLIT","D-SPLIT-NBS1","D-SPLIT-TAILOWNER","D-PARAM-FUSE_GUMBEL","D-SHAPE-TAIL-MID"]` + comment (line also re-wrapped, it was 128 chars) |
| 3 | `STAB_CASES` | `stab_B7_V4409_bv2048_gumbel` | B=7 V=4409 K=256 **BV=2048** | same flip | tags `["D-PARAM-BV","D-PARAM-FUSE_GUMBEL"]` → `["D-PATH-SPLIT","D-SPLIT-NBS1","D-SPLIT-TAILOWNER","D-PARAM-BV","D-PARAM-FUSE_GUMBEL"]` + comment |
| 4 | `STAB_CASES` | `stab_B5_V10240_gumbel` | B=5 V=10240 K=256 BV=4096 | same flip | tags `["D-SHAPE-TAIL-MID","D-PARAM-FUSE_GUMBEL"]` → `["D-PATH-SPLIT","D-SPLIT-NBS1","D-SPLIT-TAILOWNER","D-SHAPE-TAIL-MID","D-PARAM-FUSE_GUMBEL"]` + comment |

Full unified diff: `round3_itemA_guard/test_file.diff` (+138 / −26). **In every case the name and
all shape/parameter fields are byte-identical; only the tag list grew.** Three now-false comment
claims were also corrected in place (module docstring, the rev3-layer §16.8 header, and the
`STAB_CASES` rev3 block, which asserted "every config above … is vetoed" and "the four shapes
below"): split-path STAB coverage is now **6** configs, not 4, and the other 6 legacy configs remain
blocked **structurally** (`n_blk_total == 1` or `n_blk_full < 2`), so no guard value can move them.

⚠️ **One consequence disclosed rather than hidden:** `stab_B5_V10240_gumbel` was documented as an
"aligned-tail **control**" on the current path. It keeps feeding TAIL=2048 through the tail-padding
path and keeps demanding 20 bit-identical launches, but it now does so on the **split** route, so it
is no longer a current-path control. The current-path controls that remain are
`stab_case5_B7_V4409_{gumbel,mask}`, `stab_B1_V4096_{gumbel,mask}`, `stab_B3_V4353_gumbel`,
`stab_B9_V4607_gumbel` (6 of 8). `D-PATH-CURRENT` coverage is unaffected (still 4: C2/C3 + the
D1/E1 baseline sides), and re-shaping an existing entry to restore the control would have been an
unauthorised edit, so it was not done.

**(3) ⭐ Runtime split-path coverage for case 6 and case 11** — the gap §12 item 8 recorded as
pending. Case 11's shape `(8, 8192)` appeared in **no** test table at all; case 6's shape appeared
only as `l1_tailmid_V` / `l1b_tailmid_V`, which judge against the golden (Tier 1) but never against
the frozen kernels (Tier 2) — and Tier 1 alone cannot see a reduction-order bug (E-S4). Added:

| layer | added | what it proves |
|---|---|---|
| `REV3_L1_CASES` | `l1r3_split_cb6_B8_V10240`, `l1r3_split_cb11_B8_V8192` | **Tier-1 tolerance vs `golden_r4`/`golden_gumbel_sample` + Tier-2 dual-path bit-pattern vs the 3 frozen kernels**, both tasks, `expect_split=True` + `blocking=True` so a future guard change that routes them back **fails loudly** |
| `STAB_CASES` | `stab_split_cb6_B8_V10240_{gumbel,mask}`, `stab_split_cb11_B8_V8192_{gumbel,mask}` | 20 bit-identical launches + golden match + the `tau` invariant, per task |
| `--level stress` | *(same 4 configs, iterated automatically)* | the **UB poisoner** (163840 B of the 196352 B region set to 1e30) × 200 reps — the only *runtime* detector of a stale-UB slot being consumed as a candidate |
| `_REV3_STATIC_SHAPES` | `cb6_S3_NBS1_STAIL2` (8,10240,256,4096), `cb11_S2_NBS1_aligned` (8,8192,256,4096) | compile-time UB ≤ region, `padCount==0`, 32 B-aligned `copy_ub_to_ub`, zero dynamic loop bounds — previously only an **out-of-band Stage-2 probe** owned this; now the delivered gate does (38 → **48** `STATIC_PASS`) |

Measured results: all 4 new L1 dual-path runs report `path=SPLIT` with the expected geometry
(cb6 `S=3 NBS=1 S_TAIL=2`, cb11 `S=2 NBS=1 S_TAIL=-1`), **`tier2_bitwise=5/5`** and Tier-1
`r=1.0000` on every output; all 4 new STAB configs `reps=20 selfdiff=0 goldenfail=0 invariant=0`;
all 4 under UB poisoning `reps=200 selfdiff=0 goldenfail=0 invariant=0`. Static UB for the new
geometries: cb6 K1A **106496**/196352 (54.24 %), K1B 104036 & 104100 (53.02 %), K2gA 148420
(75.59 %), K2gB 132; cb11 K1A **90112** (45.89 %), K1B 79460 & 79524, K2gA 148420, K2gB 132 — all
`padCount_all0=True all32B=True dyn_bounds=[]`. The cb6/cb11 K1A/K1B numbers are **identical to the
Stage-2 probe values quoted in §3.1**, an independent cross-check that the same geometries were
compiled.

**(4) `_r4_split_plan` reproducibility self-check** — `verify_split_plan_itemA.py` reads the constant
**as shipped in the file** (no monkey-patching, so it cannot diverge from the delivered logic) and
prints the full 18-case table. Result: admitted set `[3, 4, 6, 7, 8, 11, 12, 17]` == expected,
**SET MATCH = True**; `(S, NBS, S_TAIL)` unchanged for **all 18** cases (the constant is read only
inside the cost guard, *after* steps 1–6 derive the geometry); **theorem P2 holds** for every split
case (`grid_A = b_num·S ≤ cube_core_num = 20`: cb6 12, cb11 8, worst case 7 at 20);
`_SPLIT_SAFETY` still 1.5. Exit 0. The pre-change prediction
(`preflight_itemA.py`, which *does* patch in-process) carries its own self-check: at `a=8.0` the
delivered function returns `[3,4,7,8,12,17]`, **exact** agreement with §3.1. Sweep confirms the
thresholds analytically — case 11 flips below 3.987, case 6 below 3.013, and the set **saturates at
8 for any `a ≤ 3.01`**, so 1.540 is not knife-edge.

`[SELF-CHECK]` **PASS** (exit 0, `split_plan_reproduction.json`). Post-change, reading the constant
exactly as shipped: admitted set `[3,4,6,7,8,11,12,17]` == the expected 8-case set; `SET MATCH=True`;
geometry `(S,NBS,S_TAIL)` unchanged for **18/18** cases; theorem P2 holds for every split case
(`grid_A ≤ cube_core_num=20`); `_SPLIT_SAFETY` still 1.5. Pre-change prediction self-check also
exact: at `a=8.0` the delivered function reproduces §3.1's `[3,4,7,8,12,17]` for **18/18** cases.

**(5) SPAN consequence — quantified and disclosed** (`span_itemA.py`, model verbatim from
`round3_host_overhead/analyze_corrected_span.py`, V1 0.058 % / V2 7.583 % validated; device durations
from this item's own msprof CSVs):

| task | case | launches | **SCORED口径** (contract metric) | span @ bare eager (H=200.51) | span @ harness guards (H=433.91) | span @ guards+msprof (H=499.64) | span @ **ACL graph** (15.588/launch) |
|---|---|---|---|---|---|---|---|
| fused | 6 | 2 → 4 | 78.000 → 58.500 us = **1.3333×** | 227.93 → 604.20 = **0.3772×** | 461.33 → 1304.39 = 0.3537× | 527.06 → 1501.57 = 0.3510× | 109.08 → 121.03 = **0.9012×** |
| fused | 11 | 2 → 4 | 62.740 → 54.160 us = **1.1584×** | 219.09 → 604.29 = **0.3626×** | 452.49 → 1304.49 = 0.3469× | 518.22 → 1501.67 = 0.3451× | 93.98 → 116.37 = **0.8075×** |
| mask | 6 | 2 → 3 | 55.160 → 44.560 us = **1.2379×** | 205.51 → 405.16 = **0.5072×** | 438.91 → 871.96 = 0.5034× | 504.64 → 1003.42 = 0.5029× | 86.34 → 91.36 = **0.9450×** |
| mask | 11 | 2 → 3 | 47.160 → 43.360 us = **1.0876×** | 203.53 → 404.98 = **0.5026×** | 436.93 → 871.78 = 0.5012× | 502.66 → 1003.24 = 0.5010× | 78.34 → 90.34 = **0.8671×** |

⚠️ **This is worse than §9's split-6 picture and must be read carefully.** Under *every* deployment
assumption the SPAN for these two cases gets **longer**, while the SCORED device-time sum — the
contract口径 the user ruled to keep — gets shorter. The reason is arithmetic: case 6/11 have the
**smallest device saving of any split case** (`pred_saving` 8.902 us; measured 19.50 / 8.58 us fused,
10.60 / 3.80 us mask), while the launch-count penalty is the same +2 (fused) / +1 (mask). Under ACL
graph the device-timeline residual is 15.588 us/launch, so +2 launches cost ~31.2 us against a
19.50 us saving — hence 0.90× / 0.81× rather than the >1 that §9 reported for the already-split 6
(whose savings are far larger; case 7 is 4.05×). Note the two "graph" numbers in circulation measure
different things and are both correct: **2.8–6.6 us** is the *host CPU* cost per launch under replay
(`probe4.cap_L0_per_launch`, marginal ≈0.09 / −0.51 us), **15.588 us** is the *device-timeline*
residual the SPAN recursion needs (`attribution_report.md` L268 distinguishes them explicitly).
Per the Orchestrator's ruling this is **not** a rejection reason — rev3 is retained with device time
as the contract口径 — but it does mean cases 6 and 11 are the two split cases **most exposed** to
launch-count cost, and if the delivery target ever becomes eager wall-clock, these are the first two
that M1 (ACL graph capture + D2D copy-in) would need to be re-evaluated against. No M1–M6
mitigation was implemented this item, as instructed.

### 13.3 Performance — 3-arm same-session A/B, plus a 9-arm arbitration

`ab_all_itemA.sh` ran **base1 (`a764e117`, a=8.0) → cand (`c59a83bc`, a=1.540) → base2 (`a764e117`,
drift control)** back to back in one session, both operators, all 18 cases each; every arm
sha-verified at swap time, `prewarm_itemA.py` inside each arm so no JIT compile lands in a timing
window (base1 90 specs / cand 96 / base2 90; **cand cold=0** — the 8 new contract specs were already
cached by the static gate's new cb6/cb11 shapes). All 6 evaluations **exit 0**, 18/18 passed each.
口径 unchanged: cann-bench → msprof `kernel_details.csv`, warmup=3 repeat=5, median of per-step
`Duration` sums. **No Python/torch timing anywhere.**

**Criterion 1 — case 6 and case 11, per task (harness口径, `cmp_itemA.py`):**

| case | task | base1 us | cand us | **speedup** | gain | drift control (base2 vs base1) | drift-corrected gain | >3 %? |
|---|---|---|---|---|---|---|---|---|
| 6 | fused | 78.000 | 58.500 | **1.3333×** | **−25.00 %** | +0.00 % | −25.00 % | **YES** |
| 6 | mask | 55.160 | 44.560 | **1.2379×** | **−19.22 %** | +0.54 % | −19.76 % | **YES** |
| 11 | fused | 62.740 | 54.160 | **1.1584×** | **−13.68 %** | +0.35 % | −14.03 % | **YES** |
| 11 | mask | 47.160 | 43.360 | **1.0876×** | **−8.06 %** | +0.17 % | −8.23 % | **YES** |

Runtime `Block Num` confirms the split really happened (`ksplit_fine_itemA.py`, all 6 CSVs:
`steps_attributed=144`, `anomaly_rows=0`, **Block Num sequence == `_r4_split_plan`-derived
expectation: PASS**, **theorem P2 PASS**, base arms `split=6 current=12` → cand arms `split=8
current=10`): case 6 fused `[4,4]` → `[12,4,12,4]`, case 6 mask `[4,4]` → `[12,4,4]`, case 11 fused
`[4,4]` → `[8,4,8,4]`, case 11 mask `[4,4]` → `[8,4,4]` — matching §3.1's probe exactly.

These agree with §3.1's **independent** probe (different session, different script): case 6 fused
1.4442× vs 1.3333×, case 6 mask 1.3089× vs 1.2379×, case 11 fused 1.2479× vs 1.1584×, case 11 mask
1.1537× vs 1.0876×. The harness reads lower because its per-step sum includes the
`CannBenchCacheClean` kernel the probe does not launch — a documented caveat, identical in both arms.

**Criterion 2 — the other 34 (task, case) pairs.** The single-sample A/B flagged exactly one:
**case 13 mask, 14.620 → 15.240 us (+4.24 %, absolute +0.62 us)**, while the byte-identical drift
control read 14.720 us (+0.68 %). `cmp_itemA.py` therefore reported criterion 2 = **FAIL** and
`OVERALL = DO NOT ADOPT` on that single sample; that raw verdict is preserved unedited as
`cmp_itemA_single_sample_1run.log` rather than being overwritten.

It was **not** argued away — it was measured out. Case 13 is `(B=6, V=2304, K_MAX=128, BV=2048)`:
`n_blk_full = 2304//2048 = 1 < 2`, so `_r4_split_plan` returns `(False, 2, 1, 1)` at **both** guard
values and the cost guard is never reached; `ksplit_fine_itemA.py` independently confirmed the same
launch sequence (`K1`+`K2m`) and Block Num in all three arms. Two experiments then settled it:

*(a) `probe_null_case13.py` — interleaved A/B/A/B null probe, 5 rounds, 9 probes, kernel file never
modified (sha256 verified `c59a83bc` before **and** after).* On 5 **NULL** probes (identical dispatch
in both arms, so any delta is pure noise) the per-round |delta| envelope reached **5.18 %**:

| probe | kind | A med us | B med us | delta % | per-round spread % |
|---|---|---|---|---|---|
| case 13 mask | NULL | 14.180 | 14.120 | **−0.42** | −0.57 … 0.00 |
| case 13 fused | NULL | 22.800 | 22.340 | −2.02 | −2.27 … −0.44 |
| case 1 mask | NULL | 11.140 | 10.580 | **−5.03** | −5.18 … −4.34 |
| case 14 mask | NULL | 14.560 | 14.600 | +0.27 | −0.14 … +0.82 |
| case 2 mask | NULL | 18.940 | 18.660 | −1.48 | −1.58 … −1.27 |
| case 6 mask | FLIP | 39.580 | 30.060 | **−24.05** (1.3167×) | −24.55 … −23.81 |
| case 11 mask | FLIP | 31.720 | 28.140 | **−11.29** (1.1272×) | −11.65 … −11.05 |
| case 6 fused | FLIP | 62.120 | 43.240 | **−30.39** (1.4366×) | −30.44 … −30.32 |
| case 11 fused | FLIP | 47.200 | 38.220 | **−19.03** (1.2350×) | −19.20 … −18.94 |

The flagged **+4.24 % does not reproduce** (case 13 mask reads −0.42 %), and an *identical-dispatch*
shape one step smaller (case 1 mask, 11.14 us) shows **−5.03 %** of pure noise — larger in magnitude
than the flagged value. The four FLIP probes sit at −11 %…−30 % with per-round spread ≤0.74 pp,
far outside that envelope, and reproduce §3.1's numbers a second time.

*(b) `ab_repeat_mask_itemA.sh` + `arbitrate_case13.py` — because the criterion is stated in the
**harness**口径, the null probe (which omits `CannBenchCacheClean`) cannot arbitrate it alone.* So
`R4MaskSample` was re-run alternately 3 more times per arm, giving **5 base + 4 cand** harness
samples (all 6 extra evaluations exit 0; kernel restored to `c59a83bc` and proven; submission copy
`cmp`-identical; wheel rebuilt):

| case 13 mask | base (a=8.0), n=5 | cand (a=1.540), n=4 |
|---|---|---|
| samples us | 14.62, 14.72, 14.58, 14.78, 14.70 | **15.24**, 14.70, 14.96, 14.80 |
| median | **14.700** | **14.880** |
| within-arm spread | 1.36 % | 3.63 % |

Median-to-median delta = **+1.22 %**, not +4.24 %; the distributions **OVERLAP** (cand min 14.70 ≤
base max 14.78). The original +4.24 % was the *worst* cand sample against the *best* base sample —
an artefact of one sample per arm. The pre-stated decision rule (scatter iff the arms overlap **and**
the median delta is inside the pooled within-arm spread; disjoint-with-cand-above ⇒ real regression
⇒ roll back) returns **MEASUREMENT SCATTER**. Across all 18 mask cases the harness's own noise floor
reaches **3.98 %** within-arm spread, i.e. **the 3 % band is at or below the harness's per-case noise
for the smallest cases** — which is precisely why the criterion had to be arbitrated by distribution
rather than by one sample. No other non-flagged mask case regresses >3 % (worst median delta
case 1 **+0.54 %**); case 17 sits 0.44 % *faster* on disjoint distributions, an already-split case
with no path change, immaterial.

`[RESULT-itemA]` **ADOPT.** All six Orchestrator criteria are met on measured data:

| # | criterion | result | verdict |
|---|---|---|---|
| 1 | case 6 & 11 each >3 % on **both** tasks, same session, with drift control | **1.3333× / 1.2379× / 1.1584× / 1.0876×** (−25.00 / −19.22 / −13.68 / −8.06 %); drift control ≤ +0.54 %; reproduced by an independent interleaved probe (1.4366× / 1.3167× / 1.2350× / 1.1272×) | **PASS** |
| 2 | none of the other 34 pairs regresses >3 % | one pair flagged on a single sample (case 13 mask +4.24 %, +0.62 us); arbitrated to **scatter** by 5+4 pooled harness samples (median **+1.22 %**, distributions overlap) and by a 5.18 % null-noise envelope on identical-dispatch shapes; worst other median delta **+0.54 %** | **PASS (arbitrated)** |
| 3 | 5 gates exit 0, counts ≥ current, no new `BOUNDARY_WARN` | static **48**/0 · all **151** PASS/**0** FAIL + 56 BOUNDARY_PASS/**6** WARN + **48** STATIC · stability **25**/0 · stress **16**/0 · coverage **25**-0-0; all exit **0**; re-run on the **final restored state** with identical counts; the 6 WARN are **byte-identically the same 6** pre-existing V-S4 `-inf` probes (`diff` of the WARN sets pre/cand/post = empty) | **PASS** |
| 4 | cann-bench 36/36 `accuracy.passed`, 180/180 `mismatch_count=0` | **36/36**, **180/180** output tensors, worst `mismatch_count` **0**, worst `max_abs_diff` **0.0**; additionally re-verified on all 9 pooled mask arms (**18/18, 90/90, mismatch 0** each) | **PASS** |
| 5 | 3 frozen kernels byte-identical | `r4_topk_stats` L91-364 `d3909063…`, `r4_mask_sample` L367-412 `f787d1ad…`, `r4_gumbel_sample` L415-555 `ca90de8d…` — **identical line ranges and sha256 before and after**, and equal to the rev2 baseline constants the static gate asserts; 7 `@tilelang.jit` kernels, same set and order | **PASS** |
| 6 | no existing assertion weakened | AST proof (`verify_no_weakening.py`): 52 functions before/after, **0 removed**, the only changed body (`test_r4_rev3_l1`) is **docstring-only**; **0 dropped entries, 0 non-tag field changes, 0 shrunk tag lists** across all 9 tables; `check_precision` body **identical**; `REV3_FROZEN_SHA256` / `REV3_UB_LIMIT` / `STAB_REPS` / `COVERAGE_NA` unchanged; no `COVERAGE_MANIFEST` value decreased; **no coverage dimension lost a tag**, 10 grew | **PASS** |

Geomeans (device time, harness口径): cand/base1 **36-case combined 1.0170×**, fused-18 **1.0227×**,
mask-18 **1.0113×**, the 4 flipping measurements **1.2009×**. Composed with §2.3's *measured*
pre-rev3 anchor (that arm was **not** re-run here, so this is composition, not a fresh measurement):
fused 1.2796 × 1.0227 = **1.3086×**, mask 1.1840 × 1.0113 = **1.1974×**, combined 1.2309 × 1.0170
= **1.2518×**, i.e. **+1.70 %** over the shipped 1.2309× — below §3.1's +2.82 % projection because
the projection applied the probe's ratios (no `CannBenchCacheClean` in the step sum) while this is
the harness口径 end to end. cann-bench `score`: fused 60.7416 → **60.8588**, mask 59.2318 →
**59.2894** (drift control 60.7181 / 59.2055).

**JIT compile cost, disclosed separately (hard constraint 7) — never inside a timing window:**
**17** newly required specializations (`preflight.json` P3), **87.15 s** cold total, **5.13 s/spec**
average, against a fresh empty `TILELANG_CACHE_DIR`; warm re-timing of the same 17 specs **0.0 s**.
Breakdown: K1A ×4, K1B ×5, K2gA ×4, K2gB ×4. ⚠️ This is **larger than §3.1 implied**: that section
correctly noted the relaxation "admits no new JIT specialization *geometry*" for the 18 **contract**
cases (confirmed — `(S,NBS,S_TAIL)` unchanged for all 18), but 8 new *compile jobs* still appear
there (the split kernels are different functions with different arg tuples), and the **4 flipping
test-table entries** add 12 more, of which 17 distinct in total. Both arms were pre-warmed before
their profiler windows, and the scored口径 is a sum of device `Duration`s, so none of this touches
any performance number above.

### 13.4 Delivery state

`examples/R4/r4_fused_sampling.py` = `c59a83bc…`; submission copy
`cann_bench_submission/cann_bench/_r4_kernels.py` = `c59a83bc…`, **`cmp`-identical** (rc=0);
wheel rebuilt rc=0 and the `_r4_kernels.py` **inside** the wheel extracted and verified `c59a83bc…`;
`proto.yaml` = `41ab7970…` untouched; `test_r4_fused_sampling.py` = `dc69ad6f…`.

Gates were run **three** times, and the counts are identical on the last two:

| run | state | static | all | stability | stress | coverage | exit codes |
|---|---|---|---|---|---|---|---|
| `gate_pre_*` | `a764e117` (baseline, before any edit) | 38/0 | 143 PASS/0 FAIL · 56/6W · 38 STATIC | 21/0 | 12/0 | 25-0-0 | 0,0,0,0,0 |
| `gate_cand_*` | `c59a83bc` + test `2f837ab1` | **48/0** | **151** PASS/**0** FAIL · 56/**6**W · **48** STATIC | **25/0** | **16/0** | 25-0-0 | 0,0,0,0,0 |
| `gate_post_*` | same, after the A/B swapped the kernel 9× | 48/0 | 151/0 · 56/6W · 48 | 25/0 | 16/0 | 25-0-0 | 0,0,0,0,0 |
| `gate_final_*` | **delivered state**, test `dc69ad6f` after the comment re-wrap | **48/0** | **151/0** · 56/**6**W · **48** | **25/0** | **16/0** | **25-0-0** | **0,0,0,0,0** |

`sha256_stable_before_after_gates=YES` on every run, and the 6 `BOUNDARY_WARN` are **byte-identically
the same 6** pre-existing V-S4 probes in all four runs (`diff` of the WARN sets pre/cand/post/final is
empty — no new WARN). Stability split-path census moved `STAB=4/12 ADV=4/8` → `STAB=10/16 ADV=4/8`
(ADV unchanged: all 8 of its shapes have `n_blk_full ≤ 1`), 8/8 ADV `dropped=0`,
`cube_core_num=20` probed.

**Format compliance:** every line this item added or changed is ≤100 chars (project style); the
whole-file count of >100-char lines in the test file actually went **down** by one (102 → 101)
because the 128-char `l1b_tailmid_V` entry was re-wrapped. `ruff check` reports exactly one finding,
`F841` on `AT1` at `r4_fused_sampling.py:442` — **pre-existing** (it is present in the `a764e117`
backup too) and located **inside the frozen `r4_gumbel_sample` kernel (L415-555)**, so fixing it is
forbidden by constraint 2 and it was deliberately left alone.

### 13.5 Anti-pattern re-check

`tilelang-perf-optimization/references/performance-antipatterns.md` re-checked against the edited
kernel. The edit is **one host-side float constant** — it adds no device code, so it cannot
introduce a device-side anti-pattern: no cross-level GM→L0 copy, no new dynamic loop bound, no new
serialized scalar chain, no unaligned `copy_ub_to_ub`. The static gate re-asserts all of it
compile-time for the two new geometries (`padCount_all0=True`, `all32B=True`, `dyn_bounds=[]`,
UB ≤ 196352). The **one deliberate retention** from §11 is unchanged and still recorded: the split
path reads `logits` from GM **twice**, trading HBM bandwidth for core parallelism — retained because
the guard-admitted cases are core-starved rather than bandwidth-limited (cb6 `grid_A` 4 → 12, cb11
4 → 8, both ≤ 20), and the measurement confirms it pays in device time (1.3333× / 1.1584× fused).
Its cost is the SPAN exposure quantified in §13.2(5), not hidden.

### 13.6 Unverified items and gaps introduced or left by this item

1. **Criterion 2 rests on an arbitration, not on a clean single-sample pass.** The literal
   single-sample reading failed (+4.24 % on case 13 mask). It is resolved by 5+4 pooled harness
   samples (median +1.22 %, overlapping distributions) plus a 5.18 % null-noise envelope on
   identical-dispatch shapes. Anyone re-reading only `cmp_itemA_single_sample_1run.log` will see
   `DO NOT ADOPT`; that file is kept deliberately, and the arbitration is `arbitrate_case13.json`.
2. **The fused task was not re-sampled.** The repeat experiment covered `R4MaskSample` only (that is
   where the flagged pair was). Case 13 **fused** read +0.86 % in the single-sample A/B and −2.02 %
   in the null probe, so no flag exists there — but it has 1 sample, not 4.
3. **The pre-rev3 anchor was not re-measured.** The 1.2518× combined geomean is *composed* from
   §2.3's measured 1.2309× and this item's measured 1.0170×; it is not a same-session measurement
   against `19e51e8d`.
4. **SPAN numbers are model estimates**, not measurements (V1 0.058 % / V2 7.583 % validated; the
   cand-arm split span has no direct anchor, so §9's ±7.6 % uncertainty applies). The graph residual
   15.588 us/launch is derived from **4 samples** and applied uniformly.
5. **`stab_B5_V10240_gumbel` lost its current-path control role** (§13.2(2) ⚠️). Coverage counts are
   unaffected, but the suite no longer has a 32 B-aligned-tail *current-path* stability control.
6. **Two off-contract geometries are now exercised at runtime** that previously were not:
   `(BV=2048, K_MAX=256)` at S=3 via `stab_B7_V4409_bv2048_gumbel` (the contract's two calibrated
   specializations are `(4096,256)` and `(2048,128)`; the guard falls back to `t_sort` default
   8.902 for this pairing), and `(5, 10240, …)` at S=3. Both pass 20 bit-identical reps + 200
   UB-poisoned reps, so they are *verified*, but they are outside the calibrated cost model, meaning
   the guard's `pred_saving` for them is an estimate rather than a calibrated value.
7. **`a_launch` is still bounded, not isolated.** No probe measured a bare ACL launch alone; 1.540 us
   is the min duration of a kernel that does real (if tiny) work, so the true floor is lower. The
   chosen value is therefore conservative in the safe direction, but it is an upper bound.
8. **Not re-run:** `bench/hap_thw_model/gates_report.py` GATE 1–4 (§7) and the roofline attainment
   table (§7.1). Those consume `t_hw`/baseline metadata that item A does not change (`t_hw` is
   deliberately not re-derived, §12 item 9), and cases 6/11 would move from vector-bound
   single-kernel to split-kernel attribution — worth a follow-up if the Orchestrator wants §7's
   tables restated for the new dispatch.
