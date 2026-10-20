"""Iter7 P3 Gate-0 feasibility probe — FIX→UB direct landing in EXPERT mode.

Probe ONLY (task-book hard boundary): artifacts stay in perf_tuning/; NOT
integrated into dispatch, gqa.py NOT modified, NO performance judgement.

Question (iter6 §6.1 root cause (a) / §7 P3): can the expert-mode CV-split
kernels land L0C results DIRECTLY into UB (FIX→UB or equivalent on-chip
path), replacing `L0C →FIX→ GM(ws_s/ws_o) →MTE2→ UB` round-trips?

Experiments:
  E1  AUTO-mode reference: build the shipped per-head kernel (production
      precedent for `T.copy(acc_s_l0c, acc_s_ub)`, gqa.py:234-class) and
      dump the generated source — WHAT does l0c→ub actually lower to under
      AUTO CV-combine (direct FIX→UB, or GM workspace round-trip)?
  E2  EXPERT-mode probe: minimal CV-split expert kernel (gqa flat_kv2 flag
      choreography in miniature, P28_EXPERT_PASS_CONFIGS) whose C scope
      issues `T.copy(l0c, ub_dst)` where ub_dst is a UB buffer read by the
      V scope.  Record: compiles? generated source? single-shot numeric
      smoke vs torch.matmul (bit-level not required — fp32 accum path)?
  E2b CONTROL: identical kernel with the shipped ws pattern (l0c→GM ws,
      V: ws→UB) — proves the harness itself is sound so any E2 failure is
      attributable to the l0c→ub copy, not the scaffolding.

Run (repo root; NPU needed for E2/E2b smoke, compile-only for E1):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python3 examples/gqa/perf_tuning/probe_iter7_P3_fix2ub.py [--exp e1|e2|e2b|all]
"""

import argparse
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_DIR = os.path.dirname(HERE)
sys.path.insert(0, GQA_DIR)

M, N, K = 64, 64, 64


def dump_source(kernel, tag):
    try:
        src = kernel.get_kernel_source()
    except Exception as e:  # noqa: BLE001
        print(f"[{tag}] get_kernel_source failed: {e}", flush=True)
        return ""
    path = os.path.join(HERE, f"probe_iter7_P3_{tag}_source.cu")
    with open(path, "w") as f:
        f.write(src)
    interesting = [
        ln.strip() for ln in src.splitlines()
        if any(k in ln for k in ("copy_l0c", "l0c_to_ub", "workspace", "copy_gm_to_ub",
                                 "SetFlag", "set_flag", "GetSys", "pipe", "Pipe"))
    ]
    print(f"[{tag}] source {len(src)} chars -> {path}", flush=True)
    print(f"[{tag}] interesting lines ({len(interesting)}):", flush=True)
    for ln in interesting[:40]:
        print(f"    {ln}", flush=True)
    return src


def e1_auto_reference():
    """Dump the shipped per-head AUTO kernel source (l0c→ub production precedent)."""
    import gqa as G

    # R1-shape per-head kernel (per_head_bm64 domain): B1 S128 Skv128 Nq8 Nkv2 D128 causal
    B, S, S_kv, N_q, N_kv, D = 1, 128, 128, 8, 2, 128
    sm_scale = 1.0 / (D**0.5)
    G._KERNEL_BUILD_CACHE.clear()
    G._COMPILED_CACHE.clear()
    import torch
    import torch_npu  # noqa: F401

    q = torch.rand(B, S, N_q, D, dtype=torch.float16).npu()
    k = torch.rand(B, S_kv, N_kv, D, dtype=torch.float16).npu()
    v = torch.rand(B, S_kv, N_kv, D, dtype=torch.float16).npu()
    out = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=True)
    torch.npu.synchronize()
    chosen = [val[2] for val in G._KERNEL_BUILD_CACHE.values()]
    print(f"[E1] chosen={chosen} out={tuple(out.shape)}", flush=True)
    assert chosen and chosen[0].startswith("per_head"), chosen
    # the compiled kernel object is cached in the spec chain
    kernel_obj = None
    for val in G._KERNEL_BUILD_CACHE.values():
        kernel_obj = val[0] if isinstance(val, tuple) else val
    # _COMPILED_CACHE holds the adapter-level runner; grab the tilelang kernel
    # from the build cache values (fn, S_padded, spec) — fn is the jit kernel
    fn = None
    for val in G._KERNEL_BUILD_CACHE.values():
        if callable(val[0]):
            fn = val[0]
    if fn is None:
        print("[E1] could not locate the jit kernel object in the build cache", flush=True)
        return False
    dump_source(fn, "e1_perhead_auto")
    return True


# ---------------------------------------------------------------------------
# E2 / E2b: minimal expert CV-split kernels
# ---------------------------------------------------------------------------
def build_expert_probe(mode):
    """mode: 'direct' (T.copy(l0c, ub_dst) on C side) or 'ws' (control:
    l0c -> GM ws -> UB, the shipped round-trip)."""
    import tilelang
    from tilelang import language as T

    sys.path.insert(0, GQA_DIR)
    from gqa import P28_EXPERT_PASS_CONFIGS

    accum = "float"
    dtype = "float16"
    SEM_DATA = 0  # C(FIX or MTE2) -> V: result ready
    SEM_FREE = 1  # V(MTE3) -> C: destination consumed (release, init token from V)
    SIG_A = 0     # MTE2 <-> MTE1 (a_l1)
    SIG_B = 1     # MTE2 <-> MTE1 (b_l1)
    SIG_L0A = 2   # MTE1 <-> M (l0a)
    SIG_L0B = 3   # MTE1 <-> M (l0b)
    SIG_L0C = 4   # M <-> FIX (l0c)
    SIG_IO = 0    # V: MTE2 <-> V (ub_dst ready, ws mode)
    SIG_VOUT = 1  # V: V <-> MTE3 (ub_tmp ready)

    def factory(M_, N_, K_):
        @T.prim_func
        def main_direct(
            A: T.Tensor((M_, K_), dtype),  # type: ignore
            B: T.Tensor((K_, N_), dtype),  # type: ignore
            C: T.Tensor((M_, N_), accum),  # type: ignore
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                a_l1 = T.alloc_L1([M_, K_], dtype)
                b_l1 = T.alloc_L1([K_, N_], dtype)
                l0a = T.alloc_L0A([M_, K_], dtype)
                l0b = T.alloc_L0B([K_, N_], dtype)
                l0c = T.alloc_L0C([M_, N_], accum)
                ub_dst = T.alloc_ub([M_, N_], accum)
                ub_tmp = T.alloc_ub([M_, N_], accum)

                with T.Scope("C"):
                    # init tokens (pretend consumers released everything)
                    T.set_flag("MTE1", "MTE2", SIG_A)
                    T.set_flag("MTE1", "MTE2", SIG_B)
                    T.set_flag("M", "MTE1", SIG_L0A)
                    T.set_flag("M", "MTE1", SIG_L0B)
                    T.set_flag("FIX", "M", SIG_L0C)
                    # MTE2: GM -> L1
                    T.wait_flag("MTE1", "MTE2", SIG_A)
                    T.copy(A, a_l1)
                    T.set_flag("MTE2", "MTE1", SIG_A)
                    T.wait_flag("MTE1", "MTE2", SIG_B)
                    T.copy(B, b_l1)
                    T.set_flag("MTE2", "MTE1", SIG_B)
                    # MTE1: L1 -> L0
                    T.wait_flag("M", "MTE1", SIG_L0A)
                    T.wait_flag("MTE2", "MTE1", SIG_A)
                    T.copy(a_l1, l0a)
                    T.set_flag("MTE1", "M", SIG_L0A)
                    T.wait_flag("M", "MTE1", SIG_L0B)
                    T.wait_flag("MTE2", "MTE1", SIG_B)
                    T.copy(b_l1, l0b)
                    T.set_flag("MTE1", "M", SIG_L0B)
                    # M: mma
                    T.wait_flag("MTE1", "M", SIG_L0A)
                    T.wait_flag("MTE1", "M", SIG_L0B)
                    T.wait_flag("FIX", "M", SIG_L0C)
                    T.mma(l0a, l0b, l0c, init=True)
                    T.set_flag("M", "MTE1", SIG_L0A)
                    T.set_flag("M", "MTE1", SIG_L0B)
                    T.set_flag("M", "FIX", SIG_L0C)
                    # FIX: THE PROBE — direct l0c -> UB (peer vid's buffer)
                    T.wait_cross_flag(SEM_FREE)
                    T.wait_flag("M", "FIX", SIG_L0C)
                    T.copy(l0c, ub_dst)
                    T.set_flag("FIX", "M", SIG_L0C)
                    T.set_cross_flag("FIX", SEM_DATA)

                with T.Scope("V"):
                    T.set_cross_flag("MTE2", SEM_FREE)  # init release token
                    T.wait_cross_flag(SEM_DATA)
                    # producer of ub_dst is C-side (FIX direct or the inserted
                    # MTE2 ws-read); the cross flag ordered it.  Relay through V
                    # with LOCAL flags (expert discipline: no AUTO sync) so the
                    # MTE3 out-copy cannot race the ub_dst write.  The V->MTE3
                    # READY direction needs no init token (producer sets before
                    # consumer waits — the gqa acc_s_half pattern).
                    T.copy(ub_dst, ub_tmp)              # V: UB->UB relay read
                    T.set_flag("V", "MTE3", SIG_VOUT)
                    T.wait_flag("V", "MTE3", SIG_VOUT)
                    T.copy(ub_tmp, C)                   # MTE3: UB -> GM out
                    T.set_cross_flag("MTE3", SEM_FREE)

        @T.prim_func
        def main_ws(
            A: T.Tensor((M_, K_), dtype),  # type: ignore
            B: T.Tensor((K_, N_), dtype),  # type: ignore
            ws: T.Tensor((1, M_, N_), accum),  # type: ignore
            C: T.Tensor((M_, N_), accum),  # type: ignore
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                a_l1 = T.alloc_L1([M_, K_], dtype)
                b_l1 = T.alloc_L1([K_, N_], dtype)
                l0a = T.alloc_L0A([M_, K_], dtype)
                l0b = T.alloc_L0B([K_, N_], dtype)
                l0c = T.alloc_L0C([M_, N_], accum)
                ub_dst = T.alloc_ub([M_, N_], accum)
                ub_tmp = T.alloc_ub([M_, N_], accum)

                with T.Scope("C"):
                    T.set_flag("MTE1", "MTE2", SIG_A)
                    T.set_flag("MTE1", "MTE2", SIG_B)
                    T.set_flag("M", "MTE1", SIG_L0A)
                    T.set_flag("M", "MTE1", SIG_L0B)
                    T.set_flag("FIX", "M", SIG_L0C)
                    T.wait_flag("MTE1", "MTE2", SIG_A)
                    T.copy(A, a_l1)
                    T.set_flag("MTE2", "MTE1", SIG_A)
                    T.wait_flag("MTE1", "MTE2", SIG_B)
                    T.copy(B, b_l1)
                    T.set_flag("MTE2", "MTE1", SIG_B)
                    T.wait_flag("M", "MTE1", SIG_L0A)
                    T.wait_flag("MTE2", "MTE1", SIG_A)
                    T.copy(a_l1, l0a)
                    T.set_flag("MTE1", "M", SIG_L0A)
                    T.wait_flag("M", "MTE1", SIG_L0B)
                    T.wait_flag("MTE2", "MTE1", SIG_B)
                    T.copy(b_l1, l0b)
                    T.set_flag("MTE1", "M", SIG_L0B)
                    T.wait_flag("MTE1", "M", SIG_L0A)
                    T.wait_flag("MTE1", "M", SIG_L0B)
                    T.wait_flag("FIX", "M", SIG_L0C)
                    T.mma(l0a, l0b, l0c, init=True)
                    T.set_flag("M", "MTE1", SIG_L0A)
                    T.set_flag("M", "MTE1", SIG_L0B)
                    T.set_flag("M", "FIX", SIG_L0C)
                    # shipped pattern: FIX -> GM ws
                    T.wait_flag("M", "FIX", SIG_L0C)
                    T.copy(l0c, ws[cid, :, :])
                    T.set_flag("FIX", "M", SIG_L0C)
                    T.set_cross_flag("FIX", SEM_DATA)

                with T.Scope("V"):
                    T.set_cross_flag("MTE2", SEM_FREE)  # unused by C in ws mode
                    T.wait_cross_flag(SEM_DATA)
                    T.copy(ws[cid, :, :], ub_dst)       # MTE2: GM -> UB
                    T.set_flag("MTE2", "V", SIG_IO)
                    T.wait_flag("MTE2", "V", SIG_IO)
                    T.copy(ub_dst, ub_tmp)              # V: relay read
                    T.set_flag("V", "MTE3", SIG_VOUT)
                    T.wait_flag("V", "MTE3", SIG_VOUT)
                    T.copy(ub_tmp, C)                   # MTE3: UB -> GM out

        return main_direct if mode == "direct" else main_ws

    return tilelang.jit(out_idx=[2] if mode == "direct" else [3],
                        pass_configs=P28_EXPERT_PASS_CONFIGS)(factory)


def e2_run(mode, tag):
    import torch
    import torch_npu  # noqa: F401

    fn = build_expert_probe(mode)
    kernel = fn(M, N, K)
    src = dump_source(kernel, tag)
    torch.manual_seed(7)
    a = (torch.rand(M, K) * 2 - 1).half().npu()
    b = (torch.rand(K, N) * 2 - 1).half().npu()
    if mode == "direct":
        c = kernel(a, b)
    else:
        ws = torch.zeros(1, M, N, dtype=torch.float32).npu()
        c = kernel(a, b, ws)
    torch.npu.synchronize()
    ref = (a.float() @ b.float()).cpu()
    got = c.float().cpu()
    max_diff = float((got - ref).abs().max())
    ok = max_diff < 1e-2
    print(f"[{tag}] smoke max_diff={max_diff:.4e} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=["e1", "e2", "e2b", "all"])
    args = ap.parse_args()
    results = {}
    for name, fn in (("e1", e1_auto_reference),
                     ("e2b", lambda: e2_run("ws", "e2b_ws_control")),
                     ("e2", lambda: e2_run("direct", "e2_expert_direct"))):
        if args.exp not in ("all", name):
            continue
        try:
            results[name] = fn()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            results[name] = f"EXCEPTION: {type(e).__name__}: {e}"
        print(f"[PROBE] {name} -> {results[name]}", flush=True)
    print(f"[PROBE_SUMMARY] {results}", flush=True)


if __name__ == "__main__":
    main()
