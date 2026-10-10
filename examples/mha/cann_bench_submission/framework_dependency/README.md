# Optional historical patch: tilelang-ascend opt-in `mte2_sync`

**Status: NOT a dependency.** The current submission (`cann_bench/mha.py`, md5
`bc930573aa17003de98e6c3fdc18ff4d`) builds and runs on **stock**
tilelang-ascend. Nothing in the package passes `mte2_sync` to `T.gemm_v0` any
more; the `kv_db` decode domain (`S <= 16`, `D == 256`, `S_kv > 256`) now uses
the stock `gemm_v0` default prologue, whose `SetFlag/WaitFlag<MTE2_MTE1>(0)`
self-drain makes every operand enqueued before a call L1-resident before its
MTE1 reads. This folder is kept for two reasons: honest disclosure of the
development history, and an **optional local performance experiment** that is
unavailable on stock evaluation environments.

## History

A previous revision of the submission (md5 `4798647e2f6648924d16be4a8ebd170e`)
passed the opt-in `mte2_sync=` argument in exactly this domain (3 call kwargs +
4 paired user `SetFlag(MTE2->MTE1)` sites). On the online evaluation platform
(stock tilelang in site-packages) that revision failed at TVMScript parse time
with `TypeError: gemm_v0() got an unexpected keyword argument 'mte2_sync'` for
the two published cases in the domain — `(60,1,512,16,256)` and
`(60,2,512,16,256)` — while all 18 other cases, which never enter the domain,
passed. The fix removed the 3 kwargs **and** the 4 companion user flag sites
**together** (removing only the kwargs would leave dangling `MTE2_MTE1` id-0
tokens that the default prologue's self-wait could consume stale, starting the
reads before the copies retire) and folded the GEMM1 parity-unrolled call pair
back into a single runtime-slot call. Prefetch copy positions and the whole
arithmetic sequence are unchanged, so outputs are bit-exact against the
pre-fix build across a 36-shape probe matrix, and the generated device source
for all 19 non-`kv_db` representative shapes is byte-identical (`diff -rq`
exit 0).

## What the patch does (unchanged description)

`tilelang_r12_mte2_sync.diff` is a patch against **tilelang-ascend itself**,
not against this submission. It adds both halves as an **opt-in** extension,
leaving the default path byte-identical:

| File | Change |
| --- | --- |
| `src/tl_templates/ascend/common.h` | `gemm_v0` gains a 9th template parameter `int32_t kMTE2Sync = -1`. `-1` keeps the existing pipe-level drain bit-for-bit; `>= 0` emits `WaitFlag<HardEvent::MTE2_MTE1>(kMTE2Sync)` in place of the drain. A `static_assert` restricts it to `-1` or `0..5`. |
| `tilelang/language/ascend.py` | `gemm_v0(...)` gains `mte2_sync=None`. When it is not `None` the value is validated as a compile-time `int` in `[0, 5]` and appended to the template argument list. |

Because the parameter defaults to "off", applying the patch changes **nothing**
for this submission: the code never opts in, so every `gemm_v0` call generates
exactly the stock 8-template-argument form.

## Performance context (measured, msprof Task Duration medians, n=25)

| Build of the `kv_db` domain | framework | case15 `(60,1,512,16,256)` | case18 `(60,2,512,16,256)` |
| --- | --- | --- | --- |
| pre-`mte2_sync` sequential (historical, bN=256) | stock | 508.11 us | 508.57 us |
| `mte2_sync` r12 state (historical, bN=256) | patched | 463.58 us (−8.76%) | 465.36 us (−8.50%) |
| `mte2_sync` r13 state (historical, bN=128 dual DB) | patched | 421.64 us | 423.56 us |
| `mte2_sync` r13 state (re-measured this session, pre-fix) | patched | 420.71 us | 422.80 us |
| **this submission** (stock default-drain sync, pre-call prefetch) | **stock** | **560.99 us** | **561.89 us** |
| rejected variant (after-return prefetch, r11-style sites) | stock | 550.81 us | 551.09 us |

The patched-framework numbers are **not attainable on the evaluation
platform** (stock site-packages); they are disclosed so the cost of the
stock-framework requirement is visible rather than hidden. Under the stock
default drain, each `gemm_v0` prologue absorbs whatever prefetch is in flight,
which is why this submission lands above the historical patched numbers. The
after-return variant was measured and rejected by the project's ≥3%-improvement
adoption rule (it gained only ~1.8-1.9% over the shipped form).

## How to apply (optional, local experiments only)

From the root of a tilelang-ascend checkout:

```bash
git apply -p1 /path/to/framework_dependency/tilelang_r12_mte2_sync.diff
# or, equivalently:
patch -p1 < /path/to/framework_dependency/tilelang_r12_mte2_sync.diff
```

Then rebuild/reinstall tilelang-ascend so both halves take effect:

* `tilelang/language/ascend.py` is picked up by a normal Python reinstall.
* `src/tl_templates/ascend/common.h` is a **device-side header consumed at JIT
  time**. If tilelang is installed from a wheel, the header travels inside the
  package (`tilelang/src/tl_templates/...`), so the package must be reinstalled —
  patching the checkout alone is not enough unless `TL_TEMPLATE_PATH` points at
  the patched tree.

Note that applying it to this submission's code yields no behavioural change
(no call opts in); it would only matter for a future revision that re-adopts
per-buffer sync for local runs.

## Disclosure

The operator was developed, tested and benchmarked **with** the patch applied
through its r13/r15 revisions; the shipped artifact was then converted to
stock-framework sync and re-verified end-to-end (compile + real-NPU runs under
a stock-signature simulation that rejects any `mte2_sync` kwarg, bit-exact
probe matrix, byte-identical non-`kv_db` device source, L0 10/10 and L1 19/19
precision suites). The submission does not modify the evaluation framework at
runtime and does not monkey-patch anything in production use.
