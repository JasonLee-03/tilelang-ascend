# cann_bench — MHA (Multi-Head Attention) forward

A TileLang-Ascend implementation of the multi-head attention **forward** pass for
Ascend NPU, written in TileLang Expert mode (explicit `T.Scope("C"/"V")`,
explicit L1/UB/L0A/L0B/L0C allocation, explicit event-flag and cross-core
semaphore management). All compute runs inside a TileLang JIT-compiled NPU
kernel.

## Interface

```python
import cann_bench

out = cann_bench.mha(query, key, value, scaleValue=-1.0, is_causal=False)
```

```python
def mha(query, key, value, scaleValue=-1.0, is_causal=False) -> torch.Tensor
```

Computes

```
y = softmax(Q @ K^T * s) @ V,     s = scaleValue if scaleValue > 0 else 1/sqrt(D)
```

on **BSHD** tensors, i.e. the layout is `[batch, seq, head, dim]`.

With `is_causal=True` the mask is **bottom-right aligned**:
`scores[..., i, j] = -inf` for `j > i + (S_kv - S)`. This is the alignment used by
autoregressive decode/MTP workloads where the query block is the tail of a longer
key/value sequence.

| Argument | Type | Meaning |
| --- | --- | --- |
| `query` | `torch.Tensor` `[B, S, N, D]` | fp16 or bf16, contiguous, finite |
| `key` | `torch.Tensor` `[B, S_kv, N, D]` | same dtype; `B`, `N`, `D` must match `query` |
| `value` | `torch.Tensor` `[B, S_kv, N, D]` | must equal `key`'s shape |
| `scaleValue` | `float` | `<= 0` selects the automatic `1/sqrt(D)`; `> 0` is used as-is |
| `is_causal` | `bool` | bottom-right aligned causal mask; requires `S <= S_kv` |

**Returns** `[B, S, N, D]`, same dtype as `query`, contiguous, allocated by the
framework (`@tilelang.jit(out_idx=[3])`).

### Supported domain

| Dimension | Range |
| --- | --- |
| `D` (head dim) | `{64, 128, 256}` only |
| dtype | `torch.float16`, `torch.bfloat16` |
| `B` | `1 .. 128` |
| `S` | `1 .. 2048` |
| `S_kv` | `1 .. 4096` |
| `N` (heads) | `1 .. 64` |
| layout | contiguous BSHD; `query`/`key`/`value` dtypes must match |
| values | finite (no NaN/Inf requirement on inputs beyond finiteness) |
| `is_causal=True` | additionally requires `S <= S_kv` |

`S` and `S_kv` need **not** be equal, and neither needs to be a multiple of any
tile size — row and column tails are handled with runtime extents inside the
kernel. GQA-style head sharing is not part of this interface (`key`/`value` carry
the same `N` as `query`).

Anything outside the table above is rejected by an `AssertionError` in the host
wrapper before any kernel is launched.

## Build and install

Reproducible source build (the recommended path):

```bash
bash build.sh
pip install --force-reinstall --no-deps dist/cann_bench-1.0.0-py3-none-any.whl
```

`build.sh` clears `build/`, `dist/` and `cann_bench.egg-info/`, then runs
`python setup.py bdist_wheel`. A pre-built wheel is also checked in under `dist/`
for evaluators that prefer the prebuilt channel; both routes install the same
package.

Verify:

```bash
python -c "import cann_bench, inspect; \
print(cann_bench.__version__, inspect.signature(cann_bench.mha))"
# 1.0.0 (query, key, value, scaleValue=-1.0, is_causal=False)
```

## Runtime environment

* Ascend NPU with `torch` + `torch_npu` available and a device visible.
* **tilelang-ascend — stock, no patch required.** Every code path in
  `cann_bench/mha.py` uses only the stock `T.gemm_v0` / `T.mma` / `T.copy` /
  flag and semaphore APIs. In particular the decode path for
  `S <= 16 and D == 256 and S_kv > 256` relies on the stock `gemm_v0` default
  prologue (a whole-MTE2-pipe drain that makes every operand enqueued before
  the call resident), so it compiles and runs on an unmodified framework.
  [`framework_dependency/`](framework_dependency/README.md) keeps a historical
  **optional** patch (`tilelang_r12_mte2_sync.diff`) that a previous revision
  used for finer-grained sync in that one domain; this submission does **not**
  call it. Applying it changes nothing for this package (it is opt-in and this
  code never opts in); it is retained only to document the development history
  and a local performance experiment that is unavailable on stock builds.
* The kernel is JIT-compiled per distinct `(B, S, S_kv, N, D, dtype, scaleValue,
  is_causal)` tuple, so the first call for a new configuration pays a compile
  cost. TileLang's own cache governs reuse; this package adds no result caching of
  its own.

## Layout of this submission

```text
.
├── build.sh                                    # bdist_wheel build entry point
├── setup.py                                    # package metadata (cann_bench 1.0.0)
├── README.md                                   # this file
├── cann_bench/
│   ├── __init__.py                             # __version__ + `from .mha import mha`
│   └── mha.py                                  # the operator (kernel + host wrapper)
├── framework_dependency/
│   ├── README.md                               # optional historical patch: what/why, not required
│   └── tilelang_r12_mte2_sync.diff             # opt-in mte2_sync for tilelang-ascend (NOT used by this submission)
└── dist/
    └── cann_bench-1.0.0-py3-none-any.whl       # prebuilt wheel
```

`cann_bench/mha.py` (md5 `bc930573aa17003de98e6c3fdc18ff4d`) is derived from
the operator source developed and performance-tuned in the tilelang-ascend tree
(`examples/mha/mha.py`). The delta is confined to the `kv_db` decode domain
(`S <= 16`, `D == 256`, `S_kv > 256`): that domain's calls no longer pass the
opt-in `mte2_sync=` framework extension and its four companion user
`SetFlag(MTE2->MTE1)` sites are removed with them, so the whole package builds
and runs on **stock** tilelang-ascend. All other domains are byte-for-byte
identical to the tuned source, and the copy inside the wheel has the same md5.
`build/`, `*.egg-info/` and `__pycache__/` are not part of the submission.

## Implementation notes

`cann_bench/mha.py` contains a `@tilelang.jit` kernel factory plus a thin host
wrapper. The host derives tile sizes and selects one of three mutually exclusive
C-scope forms purely from `(S, D, S_kv)`; the V scope is shared by all three
through a `@T.macro` emitter. The C (cube) and V (vector) cores exchange data
through three explicit GM workspaces in an 8-deep batched pipeline, and each AIV
owns half of the query rows.

The module docstring of `cann_bench/mha.py` is the authoritative description of
the architecture, the precision invariants and the synchronization model; read it
before changing anything in the kernel.

## Submission behaviour compliance

* **SUB-BEH-001 / SUB-BEH-003** — no PyTorch, `torch_npu` or CANN built-in
  compute API is used for any part of the attention computation. The only
  `torch` calls in the module are dtype/shape/contiguity assertions,
  `torch_npu.npu.get_device_properties(0).cube_core_num` (a host **metadata**
  query used to size the grid; it touches no tensor data), and the framework's
  own output allocation via `out_idx`. There is no `matmul`, `softmax`, `sdpa`
  or equivalent anywhere in the execution path, and nothing is routed to a CANN
  built-in operator of the same name.
* **SUB-BEH-002** — the host wrapper performs **no** tensor transformation: no
  transpose, permute, contiguous-copy, cast, slice, gather or scatter. Inputs are
  required to already be contiguous BSHD and are passed to the kernel as-is; the
  BSHD row-stride addressing, the `K^T` transpose and the tail masking all happen
  inside the kernel.
* **SUB-BEH-004** — there is no CPU fallback path. The wrapper's only action is
  validation followed by a JIT kernel launch; the returned tensor is produced by
  the submitted NPU kernel. This was verified by executing the *installed* wheel
  and confirming the result is an `npu` tensor matching the golden reference.
* **SUB-BEH-005** — no result caching, no memoisation, no constant or
  pre-baked outputs, and no branching on `data_ptr()`, on a fixed shape, dtype or
  attribute set, or on any published case. The host code path is identical for
  every legal input; only the compile-time tile configuration differs, and it is
  derived from the shape, not from a lookup table of cases.
* **SUB-BEH-006** — nothing monkey-patches, replaces or removes profiler,
  synchronization or timing APIs. The package neither imports nor touches the
  evaluation harness.
* **SUB-BEH-007** — the return value is a real, contiguous `torch.Tensor`
  allocated by the framework on the NPU device.

## Reproducibility of this artifact

The operator source in this submission is derived from the version that was
developed and performance-tuned in the tilelang-ascend tree in two mechanical
steps, each machine-checked rather than asserted:

1. **Comments/docstrings cleanup** — stripping docstrings from both revisions
   yielded a character-identical `ast.dump`, and the generated device source for
   20 representative shapes was byte-identical (20/20).
2. **Stock-framework fix of the `kv_db` decode domain** (`S <= 16`, `D == 256`,
   `S_kv > 256`) — the three `T.gemm_v0(..., mte2_sync=<id>)` call kwargs and
   their four paired user `SetFlag(MTE2->MTE1)` sites were removed **together**
   (removing only one half would leave dangling event tokens that the default
   prologue wait could consume stale), and the GEMM1 parity-unrolled call pair
   folded back into a single runtime-slot call. The kernel's arithmetic
   sequence — tiles, order, accumulation, masking — is untouched; only the
   intra-core synchronization granularity changes: each `gemm_v0` default
   prologue now drains the whole MTE2 pipe (covering all operands enqueued
   before the call) instead of waiting on a per-buffer caller flag. Verified:
   * generated device source for the **19 non-`kv_db` representative shapes** is
     **byte-identical** to the pre-fix revision (`diff -rq` exit 0);
   * the three `kv_db` shapes emit stock 8-template-argument `gemm_v0<...>`
     instantiations and contain **zero** user `SetFlag<MTE2_MTE1>` sites;
   * a stock-signature simulation (memory-level monkeypatch rejecting any
     `mte2_sync` kwarg, exactly the platform's failure mode) compiles and runs
     the `kv_db` shapes on real hardware;
   * outputs are **bit-exact against the pre-fix build** on a 34-shape probe
     matrix (22 `kv_db` targets incl. the published bench shapes, cross-round,
     causal, tail, bf16, scale, mixed task-count, minimal and zero-input
     controls + 12 non-`kv_db` guards), each also passing the golden blocking
     caliber (mixed tolerance and MARE gates).

**Performance disclosure.** The pre-fix revision reached `case15`/`case18`
(60,1,512,16,256)/(60,2,512,16,256) wall medians of 421.64/423.56 us **with the
framework patch applied** (the finer-grained sync was worth −8.76%/−8.50% over
its own default-drain predecessor at the time). On stock tilelang-ascend that
patch is unavailable, and this submission's `kv_db` domain runs on the default
full-drain sync; the wall medians measured for this exact artifact are reported
in `framework_dependency/README.md`. Every other domain is byte-identical to the
tuned source, so all other performance numbers carry over unchanged.
