# R4 Fused-Sampling — CANN Bench Submission

A [CANN Bench](../../../.agents/skills/cann-bench) submission project that wraps the
delivered TileLang operator `examples/R4/r4_fused_sampling.py` so the cann-bench
harness can build, install and evaluate it. It packages two Ascend NPU kernels
(K1 `r4_topk_stats` + a K2 second pass) behind two thin host wrappers exported from
the `cann_bench` Python package.

This project is **pure Python** (no C++/Ascend C source, no `.run` package): the
kernels are TileLang JIT kernels compiled at first call.

## What is exported

Exactly two public callables (the harness interface scan finds nothing else):

```python
def r4_fused_sampling(logits, top_ks, top_ps, min_ps, rand,
                      need_min_p_sampling=False, K_MAX=256, BV=4096):
    """V01-b path (FUSE_GUMBEL=True): Gumbel-max fused sampling.
    -> (m_out, s_out, tau_out, valid, tok_out)
       float32[B], float32[B], float32[B], int32[B], int32[B]"""

def r4_mask_sample(logits, top_ks, top_ps, min_ps,
                   need_min_p_sampling=False, K_MAX=256, BV=4096):
    """V01-a path (FUSE_GUMBEL=False): masked (un-renormalized) probabilities.
    -> (m_out, s_out, tau_out, valid, w_out)
       float32[B], float32[B], float32[B], int32[B], float32[B,V]"""
```

Inputs (all on NPU): `logits [B,V] float32`, `top_ks [B] int32`, `top_ps [B] float32`,
`min_ps [B] float32`, `rand [B] int32` (V01-b only). Parameter names are fixed by the
CANN Bench contract — `param_builder` passes tensors by `proto.inputs[].name` and
appends `case.attrs` as kwargs, so the names must match exactly. All outputs are plain,
contiguous `torch.Tensor` objects on the NPU, returned in the order/count above.

## Directory layout

```
cann_bench_submission/
├── build.sh                 # vendors ../r4_fused_sampling.py -> cann_bench/_r4_kernels.py, then bdist_wheel
├── setup.py                 # name="cann_bench", find_packages(), python_requires
├── README.md                # this file
├── .gitignore               # ignores build/ dist/ *.egg-info cann_bench/_r4_kernels.py _compile.log
└── cann_bench/
    ├── __init__.py          # __version__ + re-export of the two callables (drops the r4_ops submodule attr)
    ├── r4_ops.py            # the two thin wrappers + `from . import _r4_kernels`
    └── _r4_kernels.py       # GENERATED at build time (verbatim copy of ../r4_fused_sampling.py); gitignored
```

`_r4_kernels.py` is **not** hand-maintained. `build.sh` copies the delivered kernel
`examples/R4/r4_fused_sampling.py` **verbatim** (byte-for-byte) into it at build time,
so the single source of truth stays the delivered kernel — rebuild after any kernel
change and the wheel picks it up. The golden (`r4_ref.py`) and the operator's test file
are deliberately **not** copied into the submission.

## How the wrappers behave (anti-cheat posture)

The wrappers do **zero host compute**. Each one only coerces `need_min_p_sampling` to a
Python `bool` (it selects a compile-time kernel specialization) and forwards to the
vendored kernel host wrapper, returning its output tuple unchanged. There is:

- no torch math, no `.item()` / `.cpu()`, no `torch.topk/sort/softmax/multinomial`;
- no input/output layout transform;
- no CPU fallback and no `try/except` that would substitute a torch implementation —
  any kernel error (e.g. `V < K_MAX`, a too-small tail block, non-float32 logits,
  `B == 0`) propagates to the caller verbatim;
- no output caching keyed on shape / `data_ptr`;
- no monkey-patching of profiler / sync / timing APIs;
- no Tensor subclasses / FakeTensor — plain `torch.Tensor` only.

Verified under the harness's real guards (`TorchOpGuard(mode="block")` +
`DeviceResidencyGuard(mode="block")`, wired as in `kernel_eval/eval/op_runner.py`):
no exception, `forbidden_calls == []`, `compute_calls == []`, `total_egress_bytes == 0`
(no NPU→CPU egress). Each call launches exactly **2** NPU kernels (K1 + K2).

## V01-a scoping caveat (SUB-BEH-001) — IMPORTANT

The original V01-a host chain in `r4_fused_sampling.py` finishes with
`torch.multinomial(w_out, num_samples=1)` to draw a token id from the masked
probabilities. That sampling step is **deliberately out of the evaluated contract** for
this submission:

- `r4_mask_sample` returns the kernel-produced masked-probability tensor `w_out`
  (`float32[B,V]`, un-renormalized) and performs **no** sampling.
- Calling `torch.multinomial` (or any other torch/torch_npu builtin) to carry part of
  the operator's semantics would violate **SUB-BEH-001**; `torch.multinomial` is not in
  the guard blocklists but is explicitly forbidden for this submission, so it is not
  used at all.

Consequence: for the **end-to-end V01-a chain** (mask → categorical sample), the final
`multinomial` draw is not implemented by an NPU kernel here. This is a documented,
intentional scoping decision, and the orchestrator records the SUB-BEH-001 deviation for
the end-to-end V01-a chain in the final report. The V01-b path (`r4_fused_sampling`)
has **no** such caveat: its `tok_out` is produced entirely on the NPU by the fused
Gumbel-max kernel (no host sampler), so the whole V01-b chain is kernel-computed.

## Build

```bash
cd examples/R4/cann_bench_submission
bash build.sh            # -> dist/cann_bench-1.0.0-py3-none-any.whl
```

`build.sh` resolves the kernel source relative to its own directory, fails loudly if it
is missing, is idempotent, and cleans `build/ dist/ *.egg-info` before building. The
harness (`kernel_eval/data/package_manager.py`) additionally deletes `build/`, `dist/`
and `*.egg-info` before running `bash build.sh` (cwd = this dir, 600 s timeout,
stdout+stderr → `_compile.log`), then requires `rc == 0` **and** a `dist/cann_bench*.whl`,
which it installs with `pip install --force-reinstall --no-deps`.

## Evaluate

The evaluation tasks live under `bench_lab/tilelang_ascend_bench/` in the cann-bench
repo (`r4_fused_sampling/` for V01-b, `r4_mask_sample/` for V01-a). From the cann-bench
repo root:

```bash
R=<...>/.agents/skills/cann-bench
SUB=<...>/examples/R4/cann_bench_submission

# V01-b (Gumbel-max fused sampling -> tok_out)
ACL_OP_INIT_MODE=1 "$R/scripts/run_evaluation.sh" \
  --bench-name cann \
  --task-dir bench_lab/tilelang_ascend_bench \
  --operator r4_fused_sampling \
  --source-dir "$SUB" \
  --device-id 0

# V01-a (masked probabilities -> w_out)
ACL_OP_INIT_MODE=1 "$R/scripts/run_evaluation.sh" \
  --bench-name cann \
  --task-dir bench_lab/tilelang_ascend_bench \
  --operator r4_mask_sample \
  --source-dir "$SUB" \
  --device-id 0
```

### Run constraints (from `.orchestrator_requirements.json` → `env_facts`)

- Use the **installed** TileLang wheel (`~/.local/lib/python3.12/site-packages/tilelang`);
  do **not** `source set_env.sh`.
- Run from a **neutral cwd**. The repo's `tilelang/` source directory (and this
  submission's own `cann_bench/` when cwd is here) shadow the installed packages and
  break imports (`import tvm`, `import cann_bench`).
- Set `ACL_OP_INIT_MODE=1`.
- Use a **persistent** `TILELANG_CACHE_DIR`; never call `disable_cache()` / `clear_cache()`.

## Prerequisites

TileLang-Ascend (the Ascend-adapted TileLang; the kernels use `T.tile.*` and
`R4_PASS_CONFIGS` that upstream PyPI `tilelang` does not provide), plus `torch` /
`torch_npu` and a reachable Ascend NPU. The `tilelang` dependency is intentionally not
pinned in `setup.py` (it is environment-provided); the harness installs with `--no-deps`.

---

## rev3 (split-V) — what changed in this submission, and one expected harness-gate deviation

> Added by Stage 2 (rev3). DESIGN.md §13 / §15 are the authority.

### Kernel content

`cann_bench/_r4_kernels.py` is still a **verbatim, build-time copy** of
`examples/R4/r4_fused_sampling.py` (`build.sh` does the copy; there is no hand-maintained fork).
rev3 grew that source from 3 to **7 `@tilelang.jit` kernels**:

| kernel | role | grid | rev3 status |
|---|---|---|---|
| `r4_topk_stats` (K1) | softmax + top-k + epilogue | `b_num` | **byte-identical to the previous delivery** |
| `r4_mask_sample` (K2m) | masked probs `w_out` | `b_num` | **byte-identical** |
| `r4_gumbel_sample` (K2g) | Gumbel-max `tok_out` | `b_num` | **byte-identical** |
| `r4_topk_stats_splitA` (K1A) | split-parallel hierarchical top-k | `b_num*S` | new |
| `r4_topk_stats_splitB` (K1B) | serial softmax scan + cross-split merge + epilogue | `b_num` | new |
| `r4_gumbel_sample_splitA` (K2gA) | split-parallel Gumbel argmax | `b_num*S` | new |
| `r4_gumbel_sample_splitB` (K2gB) | serial cross-split strict-GT fold | `b_num` | new |

`r4_ops.py` needed **no change**: it calls the host wrapper `r4_fused_sampling(...)` by keyword,
and rev3 kept that signature and positional order exactly, so the dispatch is invisible to the
contract. `proto.yaml` is unchanged (DESIGN.md §15.1) — `S`/`NBS`/`S_TAIL` are **derived**
quantities (a function of `B, V, K_MAX, BV` and the probed core count), not user-tunable attrs.

Which path runs is decided **entirely on the host, from Python ints** (`_r4_split_plan`): no
tensor element is read, so there is **no new D2H sync and no ACLGraph break**. On 910b2
(`CUBE_CORE_NUM = 20`, probed at runtime — never hardcoded) **6 of the 18 bench cases** take the
split path: **{3, 4, 7, 8, 12, 17}**. The other 12 execute byte-identical code, so their results
and timings are unchanged by construction.

### Correctness posture

The split path is required to be **bit-identical** to the delivered path, not merely within
tolerance. `test_r4_fused_sampling.py` enforces this as a second acceptance tier
(`[BITWISE_PASS]`/`[BITWISE_FAIL]`): the same input is run through both paths and all 5 outputs
are compared with `torch.equal` on the raw bit pattern (fp32 viewed as `int32`, so `±0`, `-inf`
and NaN patterns must match too). The three frozen kernels are also pinned by **sha256 of their
source segments** as a static assertion, so any accidental edit to them fails the test suite.

### ⚠️ Expected harness-gate deviation (read-only gate, NOT modified here)

The cann-bench harness has a sanity gate recorded as `Block Num == ceildiv(B,2)` plus
"exactly 2 NPU kernels per call". Under rev3 both are **wrong for the 6 split-path cases**,
because those cases launch more kernels with a larger grid:

| path | task | kernels per call | `Block Num` in launch order (`b_num = ceil(B/2)`) |
|------|------|------------------|--------------------------------------------------|
| current | mask / fused | 2 | `[b_num, b_num]` |
| split | mask | **3** | `[b_num*S, b_num, b_num]` |
| split | fused | **4** | `[b_num*S, b_num, b_num*S, b_num]` |

Concretely (`S` per case): 3 → `S=4`, 4 → `S=4`, 7 → `S=10`, 8 → `S=8`, 12 → `S=2`, 17 → `S=2`.
E.g. case 7 fused = 4 rows with `Block Num` `[20, 2, 20, 2]` — the `20` is exactly one full wave
of the 20 cube core groups, which is the design intent (theorem P2), not an anomaly.

The harness is **read-only and owned by another agent**, so it was **not** modified. The R4-side
cross-check that we do own (`perf_tuning/round2_item4/k1_split.py`) **was** consciously updated to
derive its expectation from `_r4_split_plan` (single source of truth) instead of the `ceil(B/2)`
constant, and it now also asserts that no split-parallel kernel exceeds `CUBE_CORE_NUM`.
**A harness `Block Num` / kernel-count failure on cases 3, 4, 7, 8, 12, 17 is therefore an
expected, understood deviation — not a kernel defect.** The full 18-case expectation table is in
`bench/hap_thw_model/README.md` §10.3.

### Timing口径 note

On the split path `logits` is read **twice** (K1A and K1B each sweep `V`). The published `t_hw_us`
uses the model's *unique-bytes-only* convention and therefore counts it once, so `t_hw` is
unchanged (< 0.2% delta) and the achieved fraction `t_hw/elapsed` remains directly comparable —
but any **HBM-bytes-based** utilisation figure is not, and must be annotated as 2-pass on the
6 split-path cases. See `bench/hap_thw_model/README.md` §10.5.
