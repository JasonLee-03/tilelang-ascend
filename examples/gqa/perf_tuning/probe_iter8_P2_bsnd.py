"""Iter8 P2 Gate-0/A probe — BSND strided Output write feasibility + MTE3 rate.

Probe ONLY (task-book hard boundary): artifacts stay in perf_tuning/; gqa.py /
test_gqa.py NOT modified; NO adoption judgement here (that is Gate 2 on the
real variants).

Two questions the P2 design depends on (iter6 §7-P2 card, iter7 next_hint 1):

  Q1 (feasibility / address mapping)  --exp num
     In EXPERT mode (P28_EXPERT_PASS_CONFIGS: AUTO_SYNC + CV_COMBINE +
     MEMORY_PLANNING all OFF) can the V scope emit a BSND-strided box write
     and land every element exactly where transpose_out would have put it
     (BIT-EXACT, all boxes distinctly tagged)?
     Candidate expression forms:
       bsnd4d — Output declared 4-D [B, S, Nq, D], box = (bz, rows, hq, :)
                (unit extent on the head dim, strided on the row dim)
       bsnd2d — Output declared 2-D [B*S, Nq*D], box = (bz*S+row0:+half,
                hq*D:+D)  == the _transpose_v1 production write form
                (gqa.py:5131-5141), i.e. the already-shipped strided MTE3
       bhsd   — reference arm (shipped FIN form): Output[B, Nq, S, D],
                box = (bz, hq, rows, :) — contiguous 8KB
     Tagging: acc_o is seeded once from GM (proper MTE2 flags) and then
     incremented by 1.0 per box, so box k of core cid carries the value
     seed+k; the host reference enumerates the flat-kv1 task decomposition
     (gqa.py:3051-3058 _ids) independently => any address error, overlap or
     missed box shows up as a bit mismatch.

  Q2 (rate / Phase-A cost model input)  --exp rate
     What does the strided form COST relative to the contiguous shipped form
     at the exact FIN box granularity ([half=32, D=128] fp16 = 8KB = 32 x
     256B bursts, row stride N_q*D)?  The rate kernel mirrors the shipped
     FIN beat (per box: UB->UB copy + V/MTE3 flag round trip + MTE3 write),
     fp16 boxes, GM read hoisted out, task stream replayed `rep` times so
     TaskDur is not launch-floor dominated.  Identical V-side work in all
     arms => the wall delta IS the MTE3 addressing delta.

Run (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python3 examples/gqa/perf_tuning/probe_iter8_P2_bsnd.py --exp num --allshapes
    # rate arms under msprof (TaskDur + PipeUtilization):
    export LD_LIBRARY_PATH=/usr/local/sqlite3/lib:$LD_LIBRARY_PATH
    msprof op --kernel-name=main --aic-metrics=PipeUtilization,Memory \
        --output=examples/gqa/perf_tuning/msprof_op_iter8_PR_<shape>_<arm>_s1 \
        python3 examples/gqa/perf_tuning/probe_iter8_P2_bsnd.py --exp rate \
            --arm <arm> --shape <shape> --rep 8
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_DIR = os.path.dirname(HERE)
sys.path.insert(0, GQA_DIR)

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

P28_EXPERT_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}
P28_NUM_CORES = 20


def _plan(B, S_padded, N_q, N_kv, block_M=64):
    G = N_q // N_kv
    G_batches = G // 2
    q_blocks = S_padded // block_M
    half = block_M // 2
    n_tasks = B * N_kv * G_batches * q_blocks
    return G, G_batches, q_blocks, half, n_tasks


def _ids_py(tid, q_blocks, G_batches, N_kv, G):
    bx = tid % q_blocks
    gb = (tid // q_blocks) % G_batches
    kg = (tid // (q_blocks * G_batches)) % N_kv
    bz = tid // (q_blocks * G_batches * N_kv)
    hq0 = kg * G + gb * 2
    return bx, gb, kg, bz, hq0


# ---------------------------------------------------------------------------
# Q1 — numeric / address-mapping probe
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[1], pass_configs=P28_EXPERT_PASS_CONFIGS)
def probe_num_kernel(B, S_padded, N_q, N_kv, D, mode, accum_dtype="float", block_M=64):
    G, G_batches, q_blocks, half, n_tasks = _plan(B, S_padded, N_q, N_kv, block_M)
    q_tasks = n_tasks // P28_NUM_CORES
    r_tasks = n_tasks % P28_NUM_CORES

    def task_range(cid_val):
        start = cid_val * q_tasks + T.if_then_else(cid_val < r_tasks, cid_val, r_tasks)
        count = q_tasks + T.if_then_else(cid_val < r_tasks, 1, 0)
        return start, count

    def _ids(tid):
        bx = tid % q_blocks
        gb = (tid // q_blocks) % G_batches
        kg = (tid // (q_blocks * G_batches)) % N_kv
        bz = tid // (q_blocks * G_batches * N_kv)
        hq0 = kg * G + gb * 2
        return bx, gb, kg, bz, hq0

    if mode == "bhsd":
        out_shape = [B, N_q, S_padded, D]
    elif mode == "bsnd4d":
        out_shape = [B, S_padded, N_q, D]
    elif mode == "bsnd2d":
        out_shape = [B * S_padded, N_q * D]
    else:
        raise ValueError(mode)

    @T.prim_func
    def main(
        Seed: T.Tensor([half, D], accum_dtype),  # type: ignore
        Output: T.Tensor(out_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(P28_NUM_CORES, is_npu=True) as (cid, vid):
            acc_o = T.alloc_ub([half, D], accum_dtype)
            acc_s_half = T.alloc_ub([half, D], accum_dtype)
            my_start, my_count = task_range(cid)
            with T.Scope("V"):
                # init: prime the MTE3->V release token (shipped V-scope init,
                # gqa.py:3394) then one-shot seed load (manual MTE2->V ordering:
                # expert mode has no AUTO_SYNC — emit_ao pattern gqa.py:3243-3249)
                T.set_flag("MTE3", "V", 0)
                T.copy(Seed, acc_o)
                T.set_flag("MTE2", "V", 0)
                T.wait_flag("MTE2", "V", 0)
                for x in T.serial(my_count):
                    bx, gb, kg, bz, hq0 = _ids(my_start + x)
                    for h in T.serial(2):
                        row0 = bx * block_M + vid * half
                        # tag: box k of this (cid,vid) stream carries seed + k
                        T.tile.add(acc_o, acc_o, 1.0)
                        T.wait_flag("MTE3", "V", 0)
                        T.copy(acc_o, acc_s_half)
                        T.set_flag("V", "MTE3", 0)
                        T.wait_flag("V", "MTE3", 0)
                        if mode == "bhsd":
                            T.copy(acc_s_half, Output[bz, hq0 + h, row0 : row0 + half, :])
                        elif mode == "bsnd4d":
                            T.copy(acc_s_half, Output[bz, row0 : row0 + half, hq0 + h, :])
                        else:
                            T.copy(
                                acc_s_half,
                                Output[bz * S_padded + row0 : bz * S_padded + row0 + half,
                                       (hq0 + h) * D : (hq0 + h + 1) * D],
                            )
                        T.set_flag("MTE3", "V", 0)
                # drain the last MTE3 write before kernel exit
                T.wait_flag("MTE3", "V", 0)

    return main


def host_reference(shape, mode, seed_val, dtype):
    """Independent host enumeration of the flat-kv1 FIN box stream."""
    import torch

    B, S, N_q, N_kv, D = shape
    G, G_batches, q_blocks, half, n_tasks = _plan(B, S, N_q, N_kv)
    q_tasks = n_tasks // P28_NUM_CORES
    r_tasks = n_tasks % P28_NUM_CORES
    if mode == "bhsd":
        out = torch.full((B, N_q, S, D), float("nan"), dtype=dtype)
    else:
        out = torch.full((B, S, N_q, D), float("nan"), dtype=dtype)
    for cid in range(P28_NUM_CORES):
        start = cid * q_tasks + (cid if cid < r_tasks else r_tasks)
        count = q_tasks + (1 if cid < r_tasks else 0)
        for vid in range(2):
            k = 0
            for xl in range(count):
                tid = start + xl
                bx, gb, kg, bz, hq0 = _ids_py(tid, q_blocks, G_batches, N_kv, G)
                for h in range(2):
                    k += 1
                    row0 = bx * 64 + vid * half
                    val = seed_val + k
                    if mode == "bhsd":
                        out[bz, hq0 + h, row0 : row0 + half, :] = val
                    else:
                        out[bz, row0 : row0 + half, hq0 + h, :] = val
    return out


def run_num(shape, mode, dump_src=False):
    import torch
    import torch_npu  # noqa: F401

    B, S, N_q, N_kv, D = shape
    dtype = torch.float32
    seed_val = 100.0
    seed = torch.full((32, D), seed_val, dtype=dtype).npu()
    fn = probe_num_kernel(B, S, N_q, N_kv, D, mode)
    if dump_src:
        p = os.path.join(HERE, f"probe_iter8_P2_num_{mode}_source.cu")
        with open(p, "w") as f:
            f.write(fn.get_kernel_source())
        print(f"[NUM:{mode}] source {len(fn.get_kernel_source())} chars -> {p}", flush=True)
    out = fn(seed)
    torch.npu.synchronize()
    exp = host_reference(shape, mode, seed_val, dtype)
    got = out.view(exp.shape) if tuple(out.shape) != tuple(exp.shape) else out
    got_c = got.cpu()
    exp_c = exp.cpu()
    bit = bool(torch.equal(got_c, exp_c))
    n_nan = int(torch.isnan(got_c).sum())
    n_bad = int((got_c != exp_c).sum()) if not bit else 0
    md = float((got_c - exp_c).abs().max()) if not bit else 0.0
    print(
        f"[NUM:{mode}] shape={shape} out.shape={tuple(out.shape)} contig={out.is_contiguous()} "
        f"bit_exact={bit} mismatches={n_bad} nan_positions={n_nan} max_diff={md:.3e} "
        f"-> {'PASS' if bit else 'FAIL'}",
        flush=True,
    )
    return bit


# ---------------------------------------------------------------------------
# Q2 — rate probe (fp16 boxes, no GM read in the stream, `rep` replays)
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[1], pass_configs=P28_EXPERT_PASS_CONFIGS)
def probe_rate_kernel(B, S_padded, N_q, N_kv, D, mode, rep=1, dtype="float16", block_M=64):
    G, G_batches, q_blocks, half, n_tasks = _plan(B, S_padded, N_q, N_kv, block_M)
    q_tasks = n_tasks // P28_NUM_CORES
    r_tasks = n_tasks % P28_NUM_CORES

    def task_range(cid_val):
        start = cid_val * q_tasks + T.if_then_else(cid_val < r_tasks, cid_val, r_tasks)
        count = q_tasks + T.if_then_else(cid_val < r_tasks, 1, 0)
        return start, count

    def _ids(tid):
        bx = tid % q_blocks
        gb = (tid // q_blocks) % G_batches
        kg = (tid // (q_blocks * G_batches)) % N_kv
        bz = tid // (q_blocks * G_batches * N_kv)
        hq0 = kg * G + gb * 2
        return bx, gb, kg, bz, hq0

    if mode == "bhsd":
        out_shape = [B, N_q, S_padded, D]
    elif mode == "bsnd4d":
        out_shape = [B, S_padded, N_q, D]
    else:
        out_shape = [B * S_padded, N_q * D]

    @T.prim_func
    def main(
        Seed: T.Tensor([half, D], dtype),  # type: ignore
        Output: T.Tensor(out_shape, dtype),  # type: ignore
    ):
        with T.Kernel(P28_NUM_CORES, is_npu=True) as (cid, vid):
            acc_o = T.alloc_ub([half, D], "float")
            acc_s_half = T.alloc_ub([half, D], dtype)
            my_start, my_count = task_range(cid)
            with T.Scope("V"):
                T.set_flag("MTE3", "V", 0)
                T.copy(Seed, acc_s_half)
                T.set_flag("MTE2", "V", 0)
                T.wait_flag("MTE2", "V", 0)
                T.copy(acc_s_half, acc_o)
                for rr in T.serial(rep):
                    for x in T.serial(my_count):
                        bx, gb, kg, bz, hq0 = _ids(my_start + x)
                        for h in T.serial(2):
                            row0 = bx * block_M + vid * half
                            T.wait_flag("MTE3", "V", 0)
                            T.copy(acc_o, acc_s_half)
                            T.set_flag("V", "MTE3", 0)
                            T.wait_flag("V", "MTE3", 0)
                            if mode == "bhsd":
                                T.copy(acc_s_half, Output[bz, hq0 + h, row0 : row0 + half, :])
                            elif mode == "bsnd4d":
                                T.copy(acc_s_half, Output[bz, row0 : row0 + half, hq0 + h, :])
                            else:
                                T.copy(
                                    acc_s_half,
                                    Output[bz * S_padded + row0 : bz * S_padded + row0 + half,
                                           (hq0 + h) * D : (hq0 + h + 1) * D],
                                )
                            T.set_flag("MTE3", "V", 0)
                T.wait_flag("MTE3", "V", 0)

    return main


def run_rate(shape, mode, rep, dump_src=True):
    import torch
    import torch_npu  # noqa: F401

    B, S, N_q, N_kv, D = shape
    seed = torch.randn(32, D, dtype=torch.float16).npu()
    fn = probe_rate_kernel(B, S, N_q, N_kv, D, mode, rep=rep)
    if dump_src:
        p = os.path.join(HERE, f"probe_iter8_P2_rate_{shape[0]}_{mode}_source.cu")
        with open(p, "w") as f:
            f.write(fn.get_kernel_source())
    out = fn(seed)
    torch.npu.synchronize()
    G, G_batches, q_blocks, half, n_tasks = _plan(B, S, N_q, N_kv)
    boxes = n_tasks * 2 * 2
    print(
        f"[RATE:{mode}] shape={shape} rep={rep} tasks={n_tasks} boxes_per_rep={boxes} "
        f"MB_per_rep={boxes * 32 * D * 2 / 1e6:.3f} out.shape={tuple(out.shape)} "
        f"[PROF_DONE] arm=rate mode={mode} shape={shape} rep={rep}",
        flush=True,
    )


SHAPES = {
    "r2": (2, 128, 32, 8, 128),
    "r3": (2, 128, 128, 8, 128),
    "r4": (4, 256, 64, 8, 128),
    "r6": (2, 128, 48, 4, 128),
    "c7": (2, 128, 32, 8, 128),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="num", choices=["num", "rate"])
    ap.add_argument("--arm", default="bsnd4d", choices=["bhsd", "bsnd4d", "bsnd2d"])
    ap.add_argument("--shape", default="r2", choices=list(SHAPES))
    ap.add_argument("--rep", type=int, default=8)
    ap.add_argument("--allshapes", action="store_true")
    args = ap.parse_args()

    if args.exp == "num":
        shapes = list(SHAPES.values()) if args.allshapes else [SHAPES[args.shape]]
        arms = ["bhsd", "bsnd4d", "bsnd2d"] if args.allshapes else [args.arm]
        ok = True
        for shp in shapes:
            for m in arms:
                try:
                    ok &= run_num(shp, m, dump_src=(shp == SHAPES["r2"]))
                except Exception as e:  # noqa: BLE001
                    import traceback

                    print(f"[NUM:{m}] shape={shp} EXCEPTION {type(e).__name__}: {e}", flush=True)
                    traceback.print_exc()
                    ok = False
        print(f"[NUM] overall -> {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)
    else:
        run_rate(SHAPES[args.shape], args.arm, args.rep)


if __name__ == "__main__":
    main()
