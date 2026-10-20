"""Iter6 hardware-peak microbenchmarks (roofline roofs) — analysis tooling, NOT an operator.

Measured roofs for Ascend910_9362 (20 AIC + 40 AIV, L2 168MB, HBM 64GB):
  gemm        tilelang Expert-mode fp16 GEMM 4096^3 (512 blocks)      -> cube roof
  torch_gemm  torch.matmul fp16 4096^3 (vendor aclnn kernel)          -> cube roof anchor
  copy        tilelang GM->UB->GM ping-pong copy (40 vec tasks)       -> HBM roof (size >> L2)
  copyl2      same kernel, 12MB working set, 16 passes                -> L2-resident roof
  vec         fp16 vadd loop on UB (40 vec tasks, ~100us each)        -> vector FLOPS roof

Each mode performs EXACTLY ONE profiled kernel launch so msprof op captures
it directly:
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:$LD_LIBRARY_PATH
    msprof op --aic-metrics=PipeUtilization,Memory,L2Cache,ArithmeticUtilization \
        --output=<dir> python3 examples/gqa/perf_tuning/microbench_peak_iter6.py --mode copy

Cross-check anchors (documented, CANN 9.2.0 platform_config/Ascend910_9362.ini):
  cube_freq=1500MHz, cube unit 16x16x16 (fp16 MAC/cycle/core => 8192 FLOP/cycle),
  ddr_rate=32 B/cycle/core (read), l2_rate=110 B/cycle/core (read),
  ub_to_ddr_rate=64 B/cycle/core (write), vec_calc_size=128 lanes.
  msprof reports Current/Rated Freq = 1650MHz for AIC in all collected samples.
"""

import argparse
import os
import sys

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T

_HERE = os.path.dirname(os.path.abspath(__file__))


@tilelang.jit(out_idx=[-1])
def gemm_expert(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    """Expert-mode tiled GEMM (examples/gemm/example_gemm.py pattern)."""
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)
            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)
                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                    T.barrier_all()
                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


_PASS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[1], pass_configs=_PASS)
def copy_kernel(n_tiles_per_task, rows_per_task, iters, dtype="float16", R=128):
    """40 vector tasks (20 cid x 2 vid) ping-pong GM->UB->GM copy.

    Each task copies `iters` tiles of [rows_per_task, R] fp16 (two UB buffers
    alternate so MTE2 of tile i+1 can overlap MTE3 of tile i under AUTO sync).

    n_tiles_per_task == iters : streaming mode, every tile distinct (HBM roof
    when total >> L2).  n_tiles_per_task < iters : L2-hot mode, the task
    re-reads/re-writes its private tile ring (total footprint stays L2-
    resident; L2Cache.csv hit rates confirm).
    """

    @T.prim_func
    def main(
        src: T.Tensor((40 * n_tiles_per_task * rows_per_task, R), dtype),
        dst: T.Tensor((40 * n_tiles_per_task * rows_per_task, R), dtype),
    ):
        with T.Kernel(20, is_npu=True) as (cid, vid):
            task = cid * 2 + vid
            ub0 = T.alloc_ub((rows_per_task, R), dtype)
            ub1 = T.alloc_ub((rows_per_task, R), dtype)
            with T.Scope("V"):
                for i in T.serial(iters):
                    base = (task * n_tiles_per_task + i % n_tiles_per_task) * rows_per_task
                    # NOTE(iter6 fix, post-s1/s2): the original `src[base, :]`
                    # 1-D slice form moved only 256B per iter (Memory.csv
                    # GM_to_UB = 24KB/task for 96 iters) — s1/s2 samples are
                    # invalid as bandwidth roofs.  The point-offset form below
                    # (same as gemm_expert's A/B loads, which measured the
                    # full 1.5GB correctly) copies the whole [rows, R] tile.
                    if i % 2 == 0:
                        T.copy(src[base, 0], ub0)
                        T.copy(ub0, dst[base, 0])
                    else:
                        T.copy(src[base, 0], ub1)
                        T.copy(ub1, dst[base, 0])

    return main


@tilelang.jit(out_idx=[0], pass_configs=_PASS)
def vec_peak_kernel(iters, n_elem=4096, dtype="float16"):
    """40 vector tasks: fp16 vadd chain on a UB tile (2 FLOP/elem/iter)."""

    @T.prim_func
    def main(
        out: T.Tensor((40, n_elem), dtype),
    ):
        with T.Kernel(20, is_npu=True) as (cid, vid):
            task = cid * 2 + vid
            a = T.alloc_ub((n_elem,), dtype)
            b = T.alloc_ub((n_elem,), dtype)
            with T.Scope("V"):
                T.tile.fill(a, 1.0)
                T.tile.fill(b, 0.0)
                for _ in T.serial(iters):
                    T.tile.add(b, a, b)  # b = a + b (vector FMA-class work)
                T.copy(b, out[task, :])

    return main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["gemm", "torch_gemm", "copy", "copyl2", "vec"])
    ap.add_argument("--mb", type=int, default=256, help="copy total MB (per direction)")
    args = ap.parse_args()

    torch.manual_seed(0)
    if args.mode == "gemm":
        M = N = K = 4096
        fn = gemm_expert(M, N, K, 128, 256, 64)
        a = torch.randn(M, K).half().npu()
        b = torch.randn(K, N).half().npu()
        print(f"[BENCH] gemm fp16 {M}x{N}x{K} blocks={M//128 * N//256}", flush=True)
        c = fn(a, b)
        torch.npu.synchronize()
        print(f"[BENCH_DONE] gemm out={tuple(c.shape)}", flush=True)
    elif args.mode == "torch_gemm":
        M = N = K = 4096
        a = torch.randn(M, K).half().npu()
        b = torch.randn(K, N).half().npu()
        print(f"[BENCH] torch_gemm fp16 {M}x{N}x{K}", flush=True)
        c = a @ b
        torch.npu.synchronize()
        print(f"[BENCH_DONE] torch_gemm out={tuple(c.shape)}", flush=True)
    elif args.mode in ("copy", "copyl2"):
        rows_per_task = 256  # tile = 256 rows x 128 cols fp16 = 64KB
        if args.mode == "copy":
            # streaming: >> L2 (168MB) per direction; every tile distinct
            total_mb = args.mb
            per_task_bytes = total_mb * 1024 * 1024 // 40
            iters = per_task_bytes // (rows_per_task * 128 * 2)
            iters = max(8, iters // 8 * 8)
            n_tiles = iters
        else:
            # L2-hot: private 4-tile ring per task (40*4*64KB = 10MB per
            # direction, 20MB total << 168MB L2), 64 re-read passes
            iters, n_tiles = 64, 4
        src_rows = 40 * n_tiles * rows_per_task
        fn = copy_kernel(n_tiles_per_task=n_tiles, rows_per_task=rows_per_task, iters=iters)
        src = torch.randn(src_rows, 128).half().npu()
        moved_mb = 40 * iters * rows_per_task * 128 * 2 / 1e6
        print(f"[BENCH] {args.mode} fp16 footprint_rows={src_rows} tile=[{rows_per_task},128] "
              f"iters={iters} n_tiles={n_tiles} moved_per_dir={moved_mb:.1f}MB", flush=True)
        dst = fn(src)
        torch.npu.synchronize()
        print(f"[BENCH_DONE] {args.mode} out={tuple(dst.shape)}", flush=True)
    elif args.mode == "vec":
        iters = 6000
        fn = vec_peak_kernel(iters)
        print(f"[BENCH] vec fp16 vadd iters={iters} n=4096 tasks=40", flush=True)
        out = fn()
        torch.npu.synchronize()
        print(f"[BENCH_DONE] vec out={tuple(out.shape)}", flush=True)


if __name__ == "__main__":
    main()
