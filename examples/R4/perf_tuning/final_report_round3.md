# R4 Stage 3 — Round 3 Final Report

**Operator:** `R4 fused sampling` (`examples/R4/r4_fused_sampling.py`)
**Round:** 3 — **verification round** (measure rev3 for the first time + close 5 dangling items)
**Date:** 2026-10-11 · **Hardware:** Ascend910_9362 (910b2), `cube_core_num=20` / `vector_core_num=40` (both probed)
**Software:** CANN 9.2.0 · torch_npu 2.7.1.post4 · python 3.12.9 · tilelang `0.1.4+d75300ed`

---

## 0. Verdict

# `[PERF_DONE]`

rev3's split-V architecture **is confirmed beneficial on measured msprof data**. The 36-case combined
geomean speedup over the pre-rev3 baseline is **1.2309×** (fused **1.2796×**, mask **1.1840×**),
with **zero regressions >3%**, **zero accuracy failures**, and **`mismatch_count = 0` on all 36
measurements**. All 5 dangling items are closed. **No rollback is warranted.**

**No code was changed this round.** The delivered kernel is sha256 `a764e117…` before and after every
activity; `test_r4_fused_sampling.py` (`cea988cb…`), `proto.yaml` (`41ab7970…`) and the case data are
untouched. "Adopted" here means *rev3 is now measured and confirmed*, not that an edit was made.

⚠️ **One material caveat must travel with this verdict** — see §7. rev3's win is real in **device
time** and in the **cann-bench scored metric**, but under this harness it does **not** carry to
eager-mode **end-to-end wall clock**, where the split path's extra launches expose ~360 us of
pre-existing per-launch host overhead. This is a host-side software-overhead finding, not a device
regression, but it is the single most important open question about rev3 and it is **not resolved**.

---

## 1. What was measured, and how

| | |
|---|---|
| baseline (pre-rev3) | `history_version/r4_fused_sampling_impl_s2r3_attempt1_base.py` · sha256 `19e51e8d8ee126b920cf48c915d863a9be3a349e2c68cf728d35d482b48bcc57` |
| candidate (rev3) | `examples/R4/r4_fused_sampling.py` · sha256 `a764e117ce150d8677096c2e00bde3db3ebc3ec16aba63e4c613d8dffdfbd117` |
| frozen-kernel check | `verify_frozen.py` base vs cand: `r4_topk_stats` L91-364 `d3909063…`, `r4_mask_sample` L367-412 `f787d1ad…`, `r4_gumbel_sample` L415-555 `ca90de8d…` — **all 3 BYTE-IDENTICAL** |
| method | **same-session 3-pass A/B**: `base1 → cand → base2`. base2 is the *same code* as base1, so `base2/base1` is a pure **session-drift control** (round 2 #4 measured 0.3%–1.7% drift on a byte-identical arm, so a cross-session baseline would be unusable) |
| 口径 | cann-bench `run_evaluation.sh` → **msprof** `kernel_details.csv`, `warmup=3 repeat=5`, `elapsed_us` = median over the 5 active steps of the **sum** of `main_kernel` `Duration(us)`. **No Python/torch timing anywhere.** |
| evals | **6/6 `EXIT_CODE=0`**, 6/6 reports 18/18 pass · 0 fail. Wall: base1 `123 s, 119 s`; cand `124 s, 116 s`; base2 `125 s, 116 s` (fused, mask) |
| reports | base1 `_150336`/`_150535` · cand `_150737`/`_150934` · base2 `_151140`/`_151335` (`bench/reports/cann_performance_eval_20261011_*.json`) |
| cache symmetry | `prewarm.py` warmed **both** arms before any timing pass ⟹ **no JIT compile inside a measurement window** |
| dispatch | `_r4_split_plan()` at probed `cube=20` ⟹ split = **{3,4,7,8,12,17}** (6/18), matching DESIGN §13.4; confirmed from the msprof **launch counts** (`split=6 current=12` in both cand CSVs, `split=0 current=18` in all four base CSVs) |

**Cache-key evidence that the 12 non-split cases run identical binaries:** the base arm's prewarm
found only **2 of 72** specializations cold — `r4_topk_stats` for **cases 12 and 17**, precisely the
two shapes the candidate compiles as `K1A/K1B` instead of `K1`. Every other entry hit the **same**
cache directory as the candidate. Conversely, all **21** cold compiles in the candidate arm were
**frozen** kernels (K1/K2g/K2m) at shapes the shared cache had never seen; **all 4 new rev3 split
kernels were already warm** from Stage 2 (0.02–0.13 s).

**Drift measured this round:** geomean(base2/base1) = fused **−0.324%**, mask **−0.036%**,
36-case **−0.180%**; single-case extremes **+1.099% / −2.436%**. Consistent with round 2's
0.3%–1.7%, and large enough that every non-split conclusion below is judged against it.

---

## 2. Headline results

| metric | fused (18) | mask (18) | **36 combined** |
|---|---|---|---|
| **geomean speedup, all cases** | **1.2796×** | **1.1840×** | **1.2309×** |
| geomean speedup, **split 6** | **2.0624×** | **1.6435×** | 1.8411× |
| geomean speedup, **non-split 12** | 1.0079× | 1.0050× | 1.0064× |
| drift-adjusted (vs base1/base2 midpoint), all 18 | 1.2775× | 1.1838× | — |
| cann-bench `score` base1 → cand | 59.2116 → **60.7629** (**+2.620%**) | 58.4226 → **59.2498** (**+1.416%**) | — |
| best single case | case 7 **−75.42%** (4.068×) | case 7 **−62.51%** (2.668×) | — |
| worst single case | **+0.107%** (case 14) | **+0.433%** (case 9) | **+0.433%** |
| cases regressing >3% | **0** | **0** | **0** |
| `mismatch_count` total / acc fails | **0 / 0** | **0 / 0** | **0 / 0** |

---

## 3. Per-case comparison

### 3.1 The 6 split cases — where rev3 acts

| case | B | V | K_MAX | BV | S | NBS | S_TAIL | fused base1 → cand (us) | **fused ×** | mask base1 → cand (us) | **mask ×** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 8 | 16384 | 256 | 4096 | 4 | 1 | −1 | 100.30 → **57.52** | **1.744** | 70.60 → **50.92** | **1.386** |
| 4 | 4 | 16384 | 256 | 4096 | 4 | 1 | −1 | 85.50 → **41.80** | **2.045** | 55.82 → **34.20** | **1.632** |
| **7** | 4 | 152064 | 256 | 4096 | **10** | 4 | 9 | 728.42 → **179.06** | ⭐ **4.068** | 461.14 → **172.86** | ⭐ **2.668** |
| 8 | 4 | 16384 | **128** | **2048** | 8 | 1 | −1 | 91.22 → **43.04** | **2.119** | 59.94 → **37.88** | **1.582** |
| 12 | 16 | 32768 | 256 | 4096 | 2 | 4 | −1 | 160.98 → **108.18** | **1.488** | 103.64 → **73.92** | **1.402** |
| 17 | 16 | 152377 | 256 | 4096 | 2 | **19** | 1 | 736.42 → **437.94** | **1.682** | 472.78 → **321.32** | **1.471** |
| | | | | | | | **geomean** | | **2.0624** | | **1.6435** |

**Per-kernel decomposition** (fused, median of 5 active steps) — *no kernel got slower*:

| case | base `K1` + `K2g` | cand `K1A` + `K1B` + `K2gA` + `K2gB` | K1 side | K2g side |
|---|---|---|---|---|
| 3 | 65.620 + 34.780 | 11.300 + 32.940 + 11.080 + 2.080 | 1.483× | 2.643× |
| 4 | 51.280 + 34.240 | 10.740 + 18.940 + 10.480 + 1.540 | 1.728× | 2.849× |
| **7** | 425.040 + 303.580 | 42.440 + 97.100 + 36.640 + 2.840 | **3.046×** | **7.690×** |
| 8 | 53.980 + 37.280 | 6.460 + 26.480 + 7.400 + 2.580 | 1.639× | 3.735× |
| 12 | 94.860 + 66.220 | 40.700 + 27.960 + 36.260 + 2.980 | 1.382× | 1.687× |
| 17 | 429.760 + 306.720 | 183.840 + 93.960 + 156.800 + 3.100 | 1.547× | 1.918× |

Case 17 is the instructive one: `b_num = 8` ⟹ `S_cap = 20//8 = 2`, so `S` is capped at 2 and each
split still walks `NBS = 19` blocks serially (`K1A = 183.84 ≈ 19 × 9.68 us/block`). The one-wave
constraint (theorem P2) genuinely limits this case, and the model captured that correctly.

### 3.2 The 12 non-split cases — the "must be exactly 1.000×" requirement

These run **byte-identical device code**, so any deviation beyond drift would mean unexpected coupling.

| case | fused b1 / cand / b2 (us) | c/b1 % | drift b2/b1 % | mask b1 / cand / b2 (us) | c/b1 % | drift b2/b1 % |
|---|---|---|---|---|---|---|
| 1 | 35.30 / 34.38 / 34.44 | −2.61 | **−2.44** | 26.44 / 26.10 / 26.60 | −1.29 | +0.61 |
| 2 | 42.12 / 41.98 / 42.10 | −0.33 | −0.05 | 33.96 / 33.98 / 33.78 | +0.06 | −0.53 |
| 5 | 55.10 / 54.08 / 54.76 | −1.85 | −0.62 | 40.24 / 39.76 / 40.24 | −1.19 | +0.00 |
| 6 | 77.98 / 77.16 / 77.36 | −1.05 | −0.80 | 55.80 / 54.78 / 55.92 | −1.83 | +0.22 |
| 9 | 85.98 / 85.66 / 85.98 | −0.37 | +0.00 | 69.28 / 69.58 / 69.66 | **+0.43** | +0.55 |
| 10 | 42.50 / 42.18 / 42.38 | −0.75 | −0.28 | 34.28 / 34.08 / 33.96 | −0.58 | −0.93 |
| 11 | 62.96 / 62.92 / 62.80 | −0.06 | −0.25 | 47.18 / 46.96 / 47.12 | −0.47 | −0.13 |
| 13 | 23.62 / 23.06 / 23.24 | −2.37 | **−1.61** | 14.84 / 14.64 / 14.66 | −1.35 | −1.21 |
| 14 | 37.44 / 37.48 / 37.62 | **+0.11** | +0.48 | 29.12 / 29.18 / 29.44 | +0.21 | **+1.10** |
| 15 | 1470.62 / 1470.76 / 1472.28 | +0.01 | +0.11 | 933.10 / 933.98 / 932.72 | +0.09 | −0.04 |
| 16 | 695.62 / 695.66 / 695.36 | +0.01 | −0.04 | 466.70 / 466.62 / 467.06 | −0.02 | +0.08 |
| 18 | 650.66 / 650.00 / 650.76 | −0.10 | +0.02 | 423.78 / 423.96 / 424.12 | +0.04 | +0.08 |
| **geomean** | | **1.0079×** | **−0.324%** | | **1.0050×** | **−0.036%** |

**Conclusion: no unexpected coupling.**

| pairwise geomean over the 12 non-split cases | fused | mask |
|---|---|---|
| cand / base1 | 0.992139 (−0.786%) | 0.995069 (−0.493%) |
| cand / base2 | 0.996713 (−0.329%) | 0.995270 (−0.473%) |
| **base2 / base1 — identical-code control** | **0.995411 (−0.459%)** | **0.999798 (−0.020%)** |

The largest non-split "gain" (fused case 1, −2.606%) is matched by the identical-code control moving
**−2.436%** on that very case. **0 of 24** non-split measurements fall outside ±3%. Two structural
proofs back the statistics: shared JIT cache entries ⟹ the same loaded binaries (§1); and
`ksplit_fine.py` reports **2 launches with `Block Num == [b_num, b_num]`** for all 12 non-split cases
in **both** arms, `anomaly_rows = 0` on all 144 steps of all 6 CSVs.

**Stated precisely:** the result is *1.000× within a ±0.5% session drift that a byte-identical
control reproduces* — not 1.000× to three decimals. Claiming exact equality would overstate the
measurement's resolution.

---

## 4. Model prediction (DESIGN §12.4) vs measurement

§12.4 was explicit that **all** its numbers were model estimates. This is their first test.

| quantity | §12.4 band (over `a_launch ∈ [0, 8] us`) | **measured** | verdict |
|---|---|---|---|
| fused 18-case geomean | 1.151× … 1.292× | **1.2796×** | ✅ **INSIDE** (upper half) |
| mask 18-case geomean | 1.107× … 1.196× | **1.1840×** | ✅ **INSIDE** (upper half) |
| case 7 fused (max) | 3.52× … 4.15× | **4.068×** | ✅ **INSIDE** (near top) |
| case 7 mask (max) | 2.43× … 2.66× | **2.668×** | ⚠️ **marginally ABOVE** (+0.3% over the `a_launch=0` bound) |
| split-6 geomean fused (@`a_launch=3us`) | 1.855× | **2.0624×** | +11.2% (model conservative) |
| split-6 geomean mask (@`a_launch=3us`) | 1.554× | **1.6435×** | +5.8% (model conservative) |
| zero-regression property over `a_launch ∈ [0,8]` | min = 1.000× by construction | **0 of 36 regress >3%** | ✅ **empirically confirmed** |

**The model is validated.** Both task geomeans land inside their predicted bands and in the bands'
*upper* half — exactly what a true `a_launch` near 0 rather than the assumed 3.0 us implies (§5).

Per-case deviations and their evidenced root causes:

| deviation | size | root cause |
|---|---|---|
| **case 4 gain under-predicted** | fused 2.045× vs 1.520× (**+34.6%**), mask 1.632× vs 1.310× (+24.6%) | §12.2's `F(256) = 21.63 us/row` is charged to `K1B`, but measured `K1B` for case 4 is **18.94 us — less than `F` alone** — while case 3 (identical V/K/BV/S/NBS, only `b_num` 4 vs 2) measures **32.94 us**. So `F` is **not** a `b_num`-independent per-row constant; §12.2 calibrated it on the single-kernel K1 where the two effects were confounded. The same ~14 us `b_num` gap exists in the **base** arm (`K1` 65.62 vs 51.28) ⟹ a pre-existing kernel property, not a rev3 artefact. |
| case 3 / 8 / 12 gains under-predicted | +17.0% / +13.3% / +3.3% (fused) | same `F` mis-attribution, shrinking as `n_blk_full` grows |
| **case 17 gain over-predicted** | −2.2% (fused) / −4.5% (mask) | `S_cap = 20//8 = 2` genuinely caps the parallelism (§3.1). The model got this one right. |
| `t(4096,256) = 8.902 us/block` | **confirmed** | `K1A` per block: case 7 `42.44/4 = 10.61`, case 12 `40.70/4 = 10.18`, case 17 `183.84/19 = 9.68` — uniformly slightly above 8.902 because K1A also carries the GM read and the partial merge. No anomaly. |
| §12.4 warning 3: "case 8's 1.57×/1.87× is optimistic" (R-F128) | **falsified** | mask matches (+0.8%), fused **exceeds** (+13.3%) — see §6 |

**Net:** the model's *geomean bands* are usable; its *absolute* per-case predictions for
small-`n_blk` split cases are not, because `F` needs re-calibration against `b_num`.

---

## 5. Dangling item U2 / V-S3 — `a_launch` measured

`a_launch` was the **only unmeasured parameter** of the §12.4 model; the shipped guard uses the
conservative `_SPLIT_A_LAUNCH_US = 8.0`, which Stage 2 did not relax.

Because cann-bench scores the **sum of device Durations** (inter-kernel gaps are *not* in the sum),
the quantity the guard must model is the **device-side duration floor of a launch**, not host launch
latency. Three independent estimators (`analyze_a_launch.py`):

**E1 — model-free floor (primary).** rev3's `K2gB` (`r4_gumbel_sample_splitB(B,S)`) has **UB = 132 B**
and its entire useful work is folding `S ≤ 10` partial `(m,s,idx)` triples per row, so its measured
Duration is launch + a little real work ⟹ an **upper bound** on `a_launch`.

| case | b_num | S | K2gB Block Num | **K2gB Duration (us)** |
|---|---|---|---|---|
| 4 | 2 | 4 | 2 | **1.540** ← min |
| 3 | 4 | 4 | 4 | 2.080 |
| 8 | 2 | 8 | 2 | 2.580 |
| 7 | 2 | 10 | 2 | 2.840 ← median |
| 12 | 8 | 2 | 8 | 2.980 |
| 17 | 8 | 2 | 8 | **3.100** ← max |

**E2 — least-squares decomposition.** Fitting `K2gB = a0 + a1·b_num + a2·S` over those 6 points gives
`a0 = **−0.050 us**`, `a1 = +0.326 us/b_num`, `a2 = +0.231 us/S`, max |residual| `0.128 us`. The
intercept is ≈0: K2gB's duration is essentially *all* work terms, leaving **no measurable constant
launch floor** in the device-Duration口径.

**E3 — §12.4 model residual: computed, then rejected.** Solving the piecewise model for `a_launch`
across 24 (case, kernel) residuals yields **+0.206 … +14.702 us with 8 of 24 NEGATIVE** (case 4 `K1B`
implies −13.79 us). E3 is internally inconsistent for exactly the reason the case-4 prediction was
off (§4): `F(256) = 21.63` does not transfer from the single-kernel K1 to K1B. **E3 is recorded as
evidence against the model's F/p separability, not as an `a_launch` value.**

> **`a_launch < 1.540 us`**, with E2 consistent with **≈0**. Conservative planning value:
> **≤ 3.100 us = 38.8% of the shipped 8.0 us default.**
> The measured range `[0, 3.1] us` lies **entirely inside** the `[0, 8] us` interval over which
> §12.4 proved zero regression — and the measurement confirms that proof: **0 of 36 cases regress >3%**.

### 5.1 Guard relaxation — **recommendation only, NOT implemented**

`_SPLIT_A_LAUNCH_US` is read **only inside the cost guard, after** `S/NBS/S_TAIL` are derived, so
relaxing it flips the `use_split` **boolean** and cannot change any geometry. Scan (the
re-implementation was self-checked to reproduce the delivered `_r4_split_plan` **exactly for 18/18
cases** at the shipped default):

| `a_launch` | split set | n | newly added |
|---|---|---|---|
| **8.00 (shipped)** | {3,4,7,8,12,17} | 6 | — |
| 4.00 | {3,4,7,8,12,17} | 6 | — |
| **3.20** | {3,4,7,8,**11**,12,17} | 7 | **case 11** |
| **≤ 2.74** | {3,4,**6**,7,8,**11**,12,17} | **8** | **cases 6, 11** — and it **saturates here** |

**The other 10 non-split cases are blocked STRUCTURALLY, not by the cost guard**, so no relaxation can
ever reach them: `S_cap = 20//b_num = 0` for cases 9/15/16/18 (`b_num` = 32/32/64/32) and `= 1` for
case 14 (`b_num=16`); `n_blk_total = 1` for cases 1/2/10; `n_blk_full < 2` for cases 5/13. Only
cases 6 and 11 are guard-marginal, and both barely (case 6 `pred_saving = 8.902 us` vs threshold
`9.183` @3.20 → misses by 0.281 us; case 11 vs `7.722` @3.20 → passes by 1.180 us).

**Benefit MEASURED, not modelled.** `probe_guard_relax.py` monkey-patched the constant
**in-process only** (delivered file sha256 verified identical before and after) and profiled both
arms with the same msprof recipe:

| case | task | arm A current (us) | arm B split (us) | **speedup** | Block Num (B) | **bitwise A == B** |
|---|---|---|---|---|---|---|
| 6 | fused | 61.380 | 42.500 | **1.4442×** (−30.76%) | [12, 4, 12, 4] | **all 5 outputs EQUAL** |
| 6 | mask | 38.980 | 29.780 | **1.3089×** (−23.60%) | [12, 4, 4] | **all 5 EQUAL** |
| 11 | fused | 46.920 | 37.600 | **1.2479×** (−19.86%) | [8, 4, 8, 4] | **all 5 EQUAL** |
| 11 | mask | 31.520 | 27.320 | **1.1537×** (−13.33%) | [8, 4, 4] | **all 5 EQUAL** |

| projected published geomean | shipped guard | **if relaxed** | uplift |
|---|---|---|---|
| fused 18 | 1.2796× | **1.3222×** | +3.33% |
| mask 18 | 1.1840× | **1.2114×** | +2.32% |
| **36 combined** | 1.2309× | **1.2656×** | **+2.82%** |

**Why this is low-risk:** the two admitted geometries (`S=3/NBS=1/S_TAIL=2`, `S=2/NBS=1/S_TAIL=-1`)
are **already compiled and already statically asserted clean** in the Stage-2 static gate, which
covers `cb6`/`cb11` under `--shapes all` (`rev3_probes/stage2/static_gate_all.log`: `--- cb6:` K1A UB
106496/196352, K1B 104036/196352; `--- cb11:` K1A 90112/196352, K1B 79460/196352; all with
`padCount_ALL0=True`, `all32B=True`, `dynamic=[]`; overall `FAILURES: 0`, `VERDICT: PASS`). Their
split-parallel Block Num (12 and 8) is comfortably ≤ 20, so theorem P2 still holds.

**Blocking conditions — why this was NOT done:** (i) it is a kernel-code change outside this round's
pre-authorisation; (ii) **4 test-table entries** carry a hardcoded `expect_use_split` that would go
stale (scanning all 84 shape-bearing test entries: `L1_CASES/l1_tailmid_V`, `L1B_CASES/l1b_tailmid_V`,
`STAB_CASES/stab_B7_V4409_bv2048_gumbel`, `STAB_CASES/stab_B5_V10240_gumbel`), so the **test file
must change in the same commit**; (iii) all 5 gates plus the 36-case A/B must be re-run afterwards;
(iv) the probe verified bitwise A==B on **one** input distribution — the delivered tier-1/tier-2/
boundary/stress suite currently drives those two shapes down the *current* path, so it would need to
cover their split form.

---

## 6. Dangling item U3 / V-S8 — case 8 (`K_MAX=128, BV=2048`)

Correctness was proven in Stage 2 (A4/L1 Tier-1+Tier-2 all pass, UB 45056/40036) but the benefit had
never been measured. §12.4 warning 3 flagged the prediction as **optimistic** because `F(128)` is not
separable (R-F128, −15.0% model fit error).

| task | base1 (us) | base2 (us) | cand (us) | **measured** | §12.4 model | model err | `mismatch_count` |
|---|---|---|---|---|---|---|---|
| fused | 91.22 | 91.26 | **43.04** | **2.119×** (−52.82%) | 1.870× | **+13.3%** | 0 |
| mask | 59.94 | 59.76 | **37.88** | **1.582×** (−36.80%) | 1.570× | **+0.8%** | 0 |

Per-kernel (fused): base `K1 = 53.980 + K2g = 37.280` → cand `K1A = 6.460 + K1B = 26.480 +
K2gA = 7.400 + K2gB = 2.580`. K1 side **1.639×**, K2g side **3.735×**. Block Num `[16, 2, 16, 2]`,
split-parallel 16 ≤ 20 ✓. The two base passes agree to **0.04%**, so the effect is far outside drift.

> **KEEP.** Both tasks gain **≫3%**, so the pre-authorised retreat (adding `and K_MAX == 256` to the
> guard) was **not needed and was NOT used** — `preauthorised_retreat_used = false`, kernel sha256
> unchanged. **Contrary to §12.4 warning 3, the case-8 prediction was not optimistic.** R-F128 can be
> downgraded: on the fused task the `K_MAX=128/BV=2048` specialization is the **second-best** split
> case measured (2.119×, behind only case 7's 4.068×); on mask it is third (1.582×).

---

## 7. 🔴 The material caveat: SCORED口径 vs WALL-CLOCK SPAN

While reading inter-kernel gaps for the `a_launch` analysis it emerged that the gaps are **large and
asymmetric between the arms**. Since `elapsed_us` is the *sum of Durations*, gaps are **not scored** —
but omitting them would make rev3 look better end-to-end than it is, so both are reported
(`analyze_span.py`, `span_vs_scored.txt`):

```
scored_us = median over 5 steps of SUM(main_kernel Duration) in the step   == cann-bench elapsed_us
span_us   = median over 5 steps of (last main_kernel end − first main_kernel start)
gap_us    = span_us − scored_us      (host-side idle between launches, inside the step)
```

| task | geomean **SCORED** | geomean **SPAN** | SPAN on the 6 split cases | SPAN on the 12 unchanged | host gap per launch (base arm) |
|---|---|---|---|---|---|
| fused | **1.2796×** | **0.7443×** | **0.3831×** | 1.0375× | **362.7 us** |
| mask | **1.1840×** | **0.8005×** | **0.4387×** | 1.0814× | **386.2 us** |

**Case 7 fused, concretely:**

```
base : K1 425.04 ─ gap   3.00 ─ K2g 303.58                              span  732.00 us
cand : K1A 42.44 ─ 416.00 ─ K1B 97.10 ─ 539.50 ─ K2gA 36.64 ─ 334.25 ─ K2gB 2.84 ─ 304.75
                                                                              span 1360.25 us
scored device time 728.42 → 179.06 us  (4.068× BETTER)      span 732.00 → 1360.25 us (1.86× WORSE)
```

**Mechanism (evidenced, not speculated):**

1. The **base arm already pays ~363–386 us of host time per launch** on every case whose first kernel
   is shorter than that overhead — **13 of 18** fused cases show a 310–392 us gap; the other 5
   (7/15/16/17/18) show 2.8–11.5 us. **Pre-existing, not caused by rev3.**
2. The gap is nearly **constant** across kernels and shapes — the signature of fixed host-side
   Python/ACL launch preparation, not of device work.
3. It **collapses to ~2.8–3.0 us** exactly when a kernel runs long enough for the host to queue the
   next launch asynchronously (base cases 7/15/16/17/18, where K1 alone runs 390–860 us).
4. rev3 replaces **1–2 long kernels with 3–4 short ones**, so on the split cases the overhead that was
   previously hidden becomes exposed.

**No kernel got slower** (§3.1: case 7 fused K1 side 3.046×, K2g side 7.690×). This is a **host-side
software-overhead finding, not a device regression**: the cann-bench score, the roofline attainment
and the core-utilisation argument (4/40 → 40/40 AIV) are all device-side and all genuinely improve.

The obvious mitigation is **ACL-graph capture** — the dispatch is pure host Python int arithmetic
*precisely* so it does not break capture (DESIGN §5.6 audit block 2) — under which launch overhead
drops to microseconds and the device-time win becomes the executed win.
**THIS IS UNVERIFIED: no graph-captured measurement was taken this round.**

---

## 8. Dangling item U4 / V-S7 — runtime `Block Num` and theorem P2

`ksplit_fine.py` gates every step on (i) launch count, (ii) `Block Num` sequence, (iii) `Block Num`
sum, and (iv) **theorem P2** — every split-*parallel* kernel's `Block Num ≤ cube_core_num`.
Expectations are **derived from the delivered `_r4_split_plan`** (single source of truth, DESIGN
§15.3.1 step 1), so the gate cannot drift out of sync with the implementation.

| CSV | arm | task | steps attributed | anomaly rows | split / current | **P2 gate** | **Block Num gate** |
|---|---|---|---|---|---|---|---|
| `ksplit_base1_fused` | base | fused | 144 | **0** | 0 / 18 | **PASS** | **PASS** |
| `ksplit_base1_mask` | base | mask | 144 | **0** | 0 / 18 | **PASS** | **PASS** |
| `ksplit_cand_fused` | cand | fused | 144 | **0** | **6 / 12** | **PASS** | **PASS** |
| `ksplit_cand_mask` | cand | mask | 144 | **0** | **6 / 12** | **PASS** | **PASS** |
| `ksplit_base2_fused` | base | fused | 144 | **0** | 0 / 18 | **PASS** | **PASS** |
| `ksplit_base2_mask` | base | mask | 144 | **0** | 0 / 18 | **PASS** | **PASS** |

Runtime `Block Num` (candidate arm; full table in `baseline_round3.json`):

| case | S | NBS | S_TAIL | b_num | **fused** | **mask** | max split-parallel | ≤ cube=20 |
|---|---|---|---|---|---|---|---|---|
| 3 | 4 | 1 | −1 | 4 | `[16, 4, 16, 4]` | `[16, 4, 4]` | 16 | ✓ |
| 4 | 4 | 1 | −1 | 2 | `[8, 2, 8, 2]` | `[8, 2, 2]` | 8 | ✓ |
| **7** | **10** | **4** | **9** | **2** | **`[20, 2, 20, 2]`** | `[20, 2, 2]` | **20** | ✓ **exactly one full wave** |
| 8 | 8 | 1 | −1 | 2 | `[16, 2, 16, 2]` | `[16, 2, 2]` | 16 | ✓ |
| 12 | 2 | 4 | −1 | 8 | `[16, 8, 16, 8]` | `[16, 8, 8]` | 16 | ✓ |
| 17 | 2 | 19 | 1 | 8 | `[16, 8, 16, 8]` | `[16, 8, 8]` | 16 | ✓ |
| 12 non-split | — | — | — | — | `[b_num, b_num]` | `[b_num, b_num]` | — | ✓ identical in both arms |

> **Runtime `Block Num` matches the plan on 864/864 attributed steps across all 6 CSVs,
> `anomaly_rows = 0`.** Theorem P2 holds at runtime with **zero violations**; the tightest point is
> **case 7 `K1A`/`K2gA` at `Block Num = 20 == cube_core_num = 20`** — exactly one full wave, as designed.
> **case 7 fused = `[20, 2, 20, 2]`, matching Stage 2's dry-run prediction element-for-element**
> (read from msprof, not from the plan function).

---

## 9. Dangling item §11.4 step 11 — cann-bench 36/36 end-to-end

The candidate pass of the A/B **is** this run (both operators, 18 cases each, through the installed
wheel built from the vendored submission copy).

| report | operator | passed | failed | worst `mismatch_count` |
|---|---|---|---|---|
| `cann_correctness_eval_20261011_150635.json` | R4FusedSampling | **18/18** | 0 | **0** |
| `cann_correctness_eval_20261011_150836.json` | R4MaskSample | **18/18** | 0 | **0** |
| `cann_performance_eval_20261011_150737.json` (accuracy section) | R4FusedSampling | 18/18 | 0 | 0 |
| `cann_performance_eval_20261011_150934.json` (accuracy section) | R4MaskSample | 18/18 | 0 | 0 |

> **36/36 `mismatch_count = 0` · 36/36 accuracy passed · 0 failed cases.**

Prerequisites confirmed in place: submission copy `cmp`-identical to the delivered kernel; wheel
rebuilt by `build.sh` rc=0 (the container hash changed `d9f31ae1… → bdfa160b…` **only** because wheels
embed timestamps — the `_r4_kernels.py` *inside* the new wheel was extracted and verified to be
`a764e117…`); `Block Num` gate updated; README deviation recorded. Both known environment traps were
handled (`R4_HTML_SHIM=1` + sitecustomize shim on `PYTHONPATH`; `LD_LIBRARY_PATH` prefixed with
`/usr/local/sqlite3/lib` — confirmed by all `elapsed_us` being non-zero, which is the symptom of
getting that wrong).

---

## 10. Dangling item GATE 1–4

`bench/hap_thw_model/gates_report.py`, exit code **0** (`gates_report_round3.log`):

| GATE | content | result |
|---|---|---|
| **1** model self-consistency | independent re-derivation of published 910b2 `t_hw_us` | **PASS** — softmax 20/20 + arg_max 20/20 |
| **2** bottleneck attribution | 36 (task, case) pairs | **PASS** — 36/36 reported, all vector-bound |
| **3** ordering sanity | `t_hw ≥ 1.0`; `t_hw == round(max(raw,1.0),4)`; fused ≥ mask | **PASS** |
| **4** intensity + achieved fraction | ridge point 3.125 elem-op/byte | **PASS** |

`t_hw` was **not** re-derived, per §15.3.2: the only ops/elem change is the `+S` cross-split merges,
worth **+0.15%** on case 7 — far inside GATE 1's tolerance. GATE 1 reproducing the published values
exactly confirms that decision.

### 10.1 Roofline attainment change (+ the mandatory 2-pass GM-read口径 note)

`attain% = 100 · t_hw_us / elapsed_us`, same published `t_hw` for both arms:

| task | all-18 geomean | split-6 geomean |
|---|---|---|
| fused | **2.1961% → 2.8101%** (1.2796×) | 1.3612% → **2.8073%** (2.0624×) |
| mask | **1.9493% → 2.3080%** (1.1840×) | 1.0803% → **1.7754%** (1.6435×) |

Headline single case: **fused case 7 attainment 0.672% → 2.733%** — the case §12.2(d) identified as
the worst core-starvation in the suite (2/20 cube groups, 36/40 AIV idle) now runs 40/40 on K1A/K2gA.

⚠️ **口径 caveat required by DESIGN §15.3.2 — computed here, not hand-waved.** The rev3 split path
reads `logits` from GM **twice** (K1A for the Sort side, K1B for the softmax side), while the
published `t_hw` charges it **once** (`hap_lib.py:20` unique-bytes / no-reload-factor convention).
Re-deriving with the reload charged (`t_hw_2pass = max(t_cube, 2·t_read, t_write, t_vector)`, same
≥1.0 clip):

* **fused: 0/6 split cases affected.** `2·t_read` stays below `t_vector` everywhere (case 7:
  `2 × 1.2672 = 2.5344 < 4.8946`) ⟹ the published floor is unchanged and all 6 remain vector-bound.
  **The fused attainment figures above are unbiased.**
* **mask: 3/6 split cases affected (7, 12, 17).** The mask task's arithmetic intensity is only
  **~2.57 ops/byte — already BELOW the 3.125 ridge point** (GATE 4), with a vec/read margin as thin
  as 1.062×, so doubling the read flips the attribution to **READ-bound**:

| mask case | `t_hw` → `t_hw_2pass` | bound | attain (published) → attain (2-pass) |
|---|---|---|---|
| 7 | 1.3464 → **2.5345** us (1.882×) | vector → **READ** | 0.779% → **1.466%** |
| 12 | 1.1676 → **2.1847** us (1.871×) | vector → **READ** | 1.579% → **2.955%** |
| 17 | 5.3964 → **10.1587** us (1.882×) | vector → **READ** | 1.679% → **3.162%** |

> **Any published mask attainment for a split case must carry this note.** GATE 2's "all
> vector-bound" verdict is a statement about the *published unique-bytes model* and remains PASS as
> written; it would need re-running if reloads were ever folded into `t_hw`.

---

## 11. Precision gates re-run on the final delivered state

Re-run because the A/B swapped `r4_fused_sampling.py` to BASE twice and back and rebuilt the wheel
three times. sha256 was verified at every swap, but only the gates prove the *behaviour* is still
green (`run_gates_round3.sh`, `gate_exits.txt`):

| gate | command | **exit** | counts measured | reference | match |
|---|---|---|---|---|---|
| static | `--level static` | **0** (10 s) | **38 STATIC_PASS / 0 FAIL** | 38 / 0 | ✓ |
| all | `--level all` | **0** (35 s) | **143 PRECISION_PASS / 0 FAIL**, **56 BOUNDARY_PASS / 6 WARN**, 38 STATIC_PASS | 143 / 56+6W / 38 | ✓ |
| stability | `--level stability` | **0** (18 s) | **21 PRECISION_PASS / 0 FAIL**, STAB_split=4 ADV_split=4, **8 ADV all `dropped=0`** | 21 / 0, 8 ADV | ✓ |
| stress | `--level stress` | **0** (23 s) | **12/12**, all `selfdiff=0 goldenfail=0 invariant=0`, 4 split-path configs | 12/12 | ✓ |
| coverage | `coverage_check.py … --proto proto.yaml` | **0** | **25 PASS / 0 MISS / 0 N/A** | 25-0-0 | ✓ |

`grep -cE "_FAIL\]"` = **0** in all four test logs. `sha256_stable_before_after_gates = YES`.

The **6 BOUNDARY_WARN** are the 3 known V-S4 probe names × 2 tasks (`bnd_split0_all_neginf`,
`bnd_split0to4_all_neginf`, `bnd_splitmid_all_neginf`), each carrying `tier2_bitwise = 5/5` and the
annotation *"EXPECTED pre-existing degrade, NOT a rev3 regression"*. Per hard constraint 6 they were
**not touched**.

---

## 12. JIT compile cost — disclosed separately (never counted as performance)

Compile time is **excluded from every performance number in this report** (all are msprof device
Durations, and both arms were cache-warm before timing — §1).

**(a) The 4 NEW rev3 split kernels, cold** — `probe_cold_split_compile.py` against a **fresh empty**
`TILELANG_CACHE_DIR`, i.e. what a first-ever run on a cold machine pays:

| | value |
|---|---|
| distinct new specializations | **23** (24 invocations; `K2gB(16,2)` is shared by cases 12 and 17) |
| total cold compile wall | **117.16 s** |
| mean / min / max per specialization | **5.09 s** / 4.49 s / 6.00 s |
| per kernel | K1A **30.05 s** (6) · K1B **34.23 s** (6) · K2gA **30.32 s** (6) · K2gB **22.56 s** (5) |
| delivered file | sha256 unchanged before/after (`kernel_file_unmodified = true`) |

**(b) Warm-cache re-instantiation** — all 24 split-kernel instantiations cost **0.02–0.13 s** each.
The candidate prewarm's 21 cold compiles (111.44 s) were **all frozen kernels** at shapes the shared
cache had never seen; the base prewarm's 2 (15.89 s) were `K1` for cases 12/17.

**(c) Eval wall clock** (fused, mask) — base1 `123 s, 119 s`; cand `124 s, 116 s`; base2 `125 s,
116 s`. The candidate arm is **not** slower wall-clock than the baseline arm, because both were warm.

---

## 13. Adoption decision

| criterion | requirement | measured | verdict |
|---|---|---|---|
| split cases | investigate any regression >3% | **0 regressions**; all 12 split measurements improved 27.9%–75.4% | **PASS** |
| case 8 | keep if >3% gain; retreat if >3% loss | **+52.82% / +36.80% faster** | **KEEP** — retreat not used |
| non-split cases | exactly 1.000× inside drift | 1.0079× / 1.0050×; worst +0.433%, best −2.606% vs identical-code control −2.436% | **PASS** — no coupling |
| cann-bench `mismatch_count` | all 0 | **0 / 36** | **PASS** |
| 5 precision gates | exit 0, counts ≥ Stage 2 | 0,0,0,0,0 · 38 · 143+56/6W · 21 · 12 · 25-0-0 | **PASS** |
| overall 36-case geomean | any measurable gain | **1.2309×** (fused 1.2796×, mask 1.1840×) | **PASS** |

**Performance-antipattern review** (`tilelang-perf-optimization/references/performance-antipatterns.md`)
— the kernel is unchanged and the split architecture reviews **positive**: no cross-level GM→L0 copy
(all transfers are `T.copy` GM↔UB and UB↔UB at 32 B-aligned counts, re-asserted by static gate
assert-④: `padCount_all0=True`, `all32B=True`); no dynamic loop bounds (`dyn_bounds=[]`); no new
serialized scalar chain (the `F(256)` `tau_p` epilogue still runs **once**, in K1B, exactly as §12.2
predicted). **One deliberate retention, recorded as required:** the split path **reads `logits` from
GM twice**, trading HBM bandwidth for core parallelism. It is retained because 14 of 18 cases with
`b_num < 20` were **core-starved, not bandwidth-limited** (attainment 0.67%–5.3%), and measurement
confirms the trade pays in device time (case 7: 4/40 → 40/40 AIV, 4.068×). Its cost is quantified in
§7 and §10.1 rather than hidden.

> **ADOPT rev3 (`a764e117`) as the new baseline.** `consecutive_no_improvement` resets to **0**
> (round 2 had accumulated 2). No rollback: the "no measurable gain, or regression" condition does
> not arise. Rollback point remains available if the Orchestrator weighs §7 differently:
> `history_version/r4_fused_sampling_impl_s2r3_attempt1_base.py` = `19e51e8d`.

### On DESIGN §16.2 — option A (4-pass, theoretical ~6×) vs option D (rev3, ~3.1×)

§16.2 declined option A partly because "4 launches bet on an unmeasured `a_launch`". **This round
removes that argument**: `a_launch < 1.540 us`, and A's modelled `≈69 us + 4L` for case 7 K1 would be
`≈75 us` at the measured floor — better than rev3's measured `139.54 us`. But §7 supplies a **new,
opposing** argument: in eager mode each extra launch costs **~360 us of host time**, so 2 more
launches would add ~720 us of span while saving ~65 us of device time. **These two facts point in
opposite directions and cannot be adjudicated without the ACL-graph-capture measurement in §14 item 1.**
Recommendation: settle that first; only then decide whether A is worth an increment.

---

## 14. Unverified items and information gaps

1. **🔴 The SPAN口径 (§7) is the material open risk.** rev3 improves the scored device-time metric
   1.2309× but *degrades* the eager-mode step span on the 6 split cases (0.38× fused / 0.44× mask).
   **Not verified:** whether ACL-graph capture — or any deployment that removes the ~360 us
   per-launch host cost — realises the device-time win end-to-end. **No graph-captured measurement
   was taken.** If the delivery target is eager-mode wall clock rather than the cann-bench score,
   this must be settled before rev3 is called a win. It also gates the §16.2 option-A decision.
2. **Cause of the ~360 us per-launch host gap not root-caused.** Its *behaviour* is well evidenced
   (constant, pre-existing in the base arm, collapses when a kernel outruns it), but the host side was
   not profiled, so the split between Python wrapper / tensor allocation / ACL launch is unknown, as
   is how much of it is msprof instrumentation overhead rather than real cost.
3. **`a_launch` is bounded, not pinned.** E1 gives `< 1.540 us` and E2's intercept is ≈0, but no
   estimator isolates a launch with *literally zero* work. Downstream use is the conservative
   `≤ 3.100 us`, which is safe for any guard decision.
4. **§12.4's `F(256) = 21.63 us/row` is falsified as a `b_num`-independent constant** (§4) but was
   **not re-calibrated** — that needs a dedicated `b_num` sweep. Consequence: the model's *absolute*
   per-case predictions for small-`n_blk` split cases are unreliable (case 4 off by +34.6%), although
   its *geomean bands* held.
5. **Guard relaxation not implemented** (§5.1). A measured **+2.82%** on the 36-case geomean is
   available but requires Orchestrator approval, a kernel edit, 4 test-table `expect_use_split`
   updates, and a full gate + A/B re-run.
6. **Case-6/11 split path lacks runtime precision coverage today.** Bitwise A==B was verified in the
   probe on **one** input distribution and the specializations are statically asserted clean, but the
   delivered runtime suite drives those shapes down the *current* path.
7. **Only 1 drift replicate.** One identical-code control pair (base1/base2) per task. It sufficed to
   explain every non-split residual, but a tighter drift bound needs more replicates.
8. **Probe口径 difference.** `probe_guard_relax.py` does not launch the harness's
   `CannBenchCacheClean` kernel per step, so its absolute microseconds may read slightly optimistic;
   both arms were profiled identically, so only the **ratios** (1.15×–1.44×) are used.
9. **`t_hw` not re-derived** (deliberate — §10). If reloads were ever folded into the model, GATE 1's
   re-derivation and GATE 2's attribution would both need updating, and mask cases 7/12/17 would flip
   to READ-bound (§10.1).
10. **No new optimization was explored** — by design; this round's charter was verification.

---

## 15. Artefacts

| file | content |
|---|---|
| `baseline_round3.json` | structured per-case baseline: shapes, dispatch, per-kernel Durations, Block Num, 3-pass elapsed, model comparison, attainment, a_launch, compile cost |
| `optimization_log_round3.md` | the round record with `[RESULT-…]` conclusion lines |
| `perf_log.md` §round3-VERIFY | cross-round structured index entry |
| `round3_splitv_perf/ab_all.sh`, `ab_run.sh`, `env_r4.sh` | reproducible 3-pass A/B driver + environment recipe |
| `round3_splitv_perf/prewarm.py` + `prewarm_{cand_coldcheck,base_warm}.json` | cache-symmetry proof + compile census |
| `round3_splitv_perf/cmp_round3.py` + `ab_3pass_comparison.{txt,json}` | per-case 3-pass comparison, drift control, model bands |
| `round3_splitv_perf/ksplit_fine.py` + `ksplit_*.{txt,json}` + `csv_manifest.txt` | per-kernel split of all 6 msprof CSVs; Block Num / theorem-P2 gates |
| `round3_splitv_perf/analyze_a_launch.py` + `a_launch_analysis.{txt,json}` | E1/E2/E3 estimators + guard-relaxation scan |
| `round3_splitv_perf/probe_guard_relax.py` + `guard_relax_probe.json` + `guard_relax_projection.json` | measured benefit of relaxing the guard (cases 6, 11) |
| `round3_splitv_perf/probe_cold_split_compile.py` + `.json` | cold JIT compile cost of the 23 new specializations |
| `round3_splitv_perf/analyze_followups.py` + `followups_analysis.{txt,json}` | roofline attainment + 2-pass口径 correction + test-table blast radius |
| `round3_splitv_perf/analyze_span.py` + `span_vs_scored.{txt,json}` | §7 scored-vs-span measurement |
| `round3_splitv_perf/verify_frozen.py` + `verify_frozen_base_vs_cand.txt` | frozen-kernel byte identity base vs cand |
| `round3_splitv_perf/run_gates_round3.sh` + `gate_*.log` + `gate_exits.txt` + `gate_sha256_{before,after}.txt` | 5 precision gates on the final state |
| `round3_splitv_perf/gates_report_round3.log` | GATE 1–4 |
| `round3_splitv_perf/eval_{base1,cand,base2}_{R4FusedSampling,R4MaskSample}.log` | raw harness logs, 6 evals |
| `bench/reports/cann_*_eval_20261011_15*.json` + `bench/reports/prof_data/_batched/**/kernel_details.csv` | raw cann-bench reports and msprof data |

`.orchestrator_state.json` (and its `.bak_*`) was **never read or written** this round.
