"""Scratch probe (NOT a deliverable): #9-lite feasibility.

Question: under threads=1 (cv_1_1) Developer pass_configs, can a MANUAL
double-buffered L1 gemm operand (alloc_shared [2, bN, D], runtime index k%2)
with T.serial + prefetch compile & run? R9 (§10.1) showed the MEMORY_PLANNING
-generated double buffer fails codegen ("Cannot find pre-allocated address
for buffer: k_l1"); the manual 3D form is a different path — must be proven
before touching mha.py.

Also inspects the generated AIC stream for actual overlap: does the K(k+1)
prefetch copy get issued WITHOUT an intervening M_MTE2 wait against gemm(k)?
"""

import tilelang
import torch
from tilelang import language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def db_probe(B, S_kv, N, D, iters, dtype="float16"):
    block_M, block_N = 16, 128
    accum_dtype = "float"

    @T.prim_func
    def main(
        Q: T.Tensor([B, 16, N, D], dtype),  # type: ignore
        K: T.Tensor([B, S_kv, N, D], dtype),  # type: ignore
        O: T.Tensor([B, 16, N, D], dtype),  # type: ignore
    ):
        with T.Kernel(B * N, threads=1, is_npu=True) as (cid):
            by = cid % N
            bz = cid // N
            q_l1 = T.alloc_shared([block_M, D], dtype)
            k_l1 = T.alloc_shared([2, block_N, D], dtype)  # manual double buffer
            acc_l0c = T.alloc_fragment([block_M, D], accum_dtype)
            acc_ub = T.alloc_shared([block_M, D], accum_dtype)
            acc = T.alloc_shared([block_M, D], accum_dtype)
            acc_half = T.alloc_shared([block_M, D], dtype)

            T.tile.fill(acc, 0.0)
            T.copy(Q[bz, 0:16, by, :], q_l1)
            # prologue prefetch K(0)
            T.copy(K[bz, 0:block_N, by, :], k_l1[0, :, :])
            for k in T.serial(0, iters):
                cur = k % 2
                nxt = (k + 1) % 2
                T.copy(K[bz, ((k + 1) % iters) * block_N : ((k + 2) % iters + 1) * block_N, by, :], k_l1[nxt, :, :])
                # consume cur: QK-ish gemm then fold into acc via L0C->UB
                T.gemm_v0(q_l1, k_l1[cur, :, :], acc_l0c, transpose_B=True, init=True)
                T.copy(acc_l0c, acc_ub)
                T.tile.add(acc, acc, acc_ub)
            T.copy(acc, acc_half)
            T.copy(acc_half[0:16, :], O[bz, 0:16, by, :])

    return main


if __name__ == "__main__":
    B, N, D, iters = 1, 32, 128, 16
    S_kv = iters * 128
    f = db_probe(B=B, S_kv=S_kv, N=N, D=D, iters=iters)
    src = f.get_kernel_source()
    open("/tmp/opencode/db_probe_srcB.cc", "w").write(src)
    print("COMPILE OK: manual double-buffered L1 gemm operand under cv_1_1")
    q = torch.randn(B, 16, N, D, dtype=torch.float16, device="npu")
    k = torch.randn(B, S_kv, N, D, dtype=torch.float16, device="npu")
    o = f(q, k)
    torch.npu.synchronize()
    print("RUN OK, finite:", bool(torch.isfinite(o).all()))
    # gemm dims: q[16,128] x k[128,128]^T -> acc[16,128]: shapes fine for probe purpose
    # overlap inspection: print AIC loop stream
    i = src.find("if ASCEND_IS_AIC")
    j = src.find("if ASCEND_IS_AIV", i + 10)
    aic = src[i:j]
    print("--- AIC section (loop) ---")
    print(aic[aic.find("for (") : aic.find("for (") + 1600])
