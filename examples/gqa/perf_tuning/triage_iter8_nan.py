"""Iter8 P2 Gate-0 failure triage — the boundary_sparse NaN-payload question.

Gate 0 reported bit=False (torch.equal) with special=ok (NaN positions equal,
Inf positions+signs equal, finite bits equal) on the two NaN/Inf injection
cases.  This script decides WHICH of three explanations holds, with data:

  (a) the shipped chain is itself non-deterministic on NaN payloads
      => BASE vs BASE re-run also differs  (not a P2 property)
  (b) the ARM is non-deterministic
      => ARM vs ARM re-run differs
  (c) a genuine P2-induced difference
      => both arms self-deterministic but BASE != ARM at specific positions;
         the raw bit patterns (uint16 view) and positions are printed so the
         difference can be classified (NaN payload/sign vs a real value).

Run (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python3 examples/gqa/perf_tuning/triage_iter8_nan.py
"""

import os
import sys

import torch
import torch_npu  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_DIR = os.path.dirname(HERE)
sys.path.insert(0, GQA_DIR)
sys.path.insert(0, HERE)

import gqa as G  # noqa: E402
import gqa_flat_bsnd_iter8_variant as VAR  # noqa: E402
from run_iter8_p2_gate0 import Arm, _clear, _inputs  # noqa: E402

CASES = [
    ("boundary_sparse_flatkv1_fp16", 2, 128, 128, 32, 8, 128, True, torch.float16),
    ("boundary_sparse_flatkv2_bf16", 2, 128, 256, 32, 8, 128, False, torch.bfloat16),
]


def run(q, k, v, causal, S, arm_on):
    _clear()
    arm = Arm()
    arm.install(S, enable_variant=arm_on)
    try:
        out = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
        torch.npu.synchronize()
        chosen = [val[2] for val in G._KERNEL_BUILD_CACHE.values()]
    finally:
        arm.remove()
        _clear()
    return out.cpu().clone(), chosen


def bits(t):
    return t.view(torch.int16) if t.dtype == torch.float16 else t.view(torch.int16)


def rawbits(t):
    """NaN-safe bit-level view: torch.equal uses IEEE == so ANY NaN position
    makes it False even when the payloads are identical (measured artifact on
    the boundary_sparse cases: the differing positions carry the SAME 0x7fff
    bits in both arms).  Raw-int compare is the rigorous bit-identity test."""
    return t.view(torch.int16)


def report(label, a, b, tag):
    eq = bool(torch.equal(rawbits(a), rawbits(b)))
    print(f"  [{tag}] RAWBIT equal={eq} (ieee torch.equal={bool(torch.equal(a, b))})", flush=True)
    if not eq:
        ne = rawbits(a) != rawbits(b)
        # NaN-position equality is a separate question from raw-bit equality
        same_nan = bool((torch.isnan(a) == torch.isnan(b)).all())
        idx = ne.nonzero()
        print(f"    differing elements={int(ne.sum())} nan_positions_equal={same_nan} "
              f"both_nan_at_all_diffs={bool(torch.isnan(a)[ne].all() and torch.isnan(b)[ne].all())}",
              flush=True)
        for i in range(min(6, idx.shape[0])):
            t = tuple(idx[i].tolist())
            print(f"    at {t}: A={a[t].item()} bits=0x{bits(a)[t].item() & 0xFFFF:04x} | "
                  f"B={b[t].item()} bits=0x{bits(b)[t].item() & 0xFFFF:04x}", flush=True)
    return eq


def main():
    for label, B, S, S_kv, N_q, N_kv, D, causal, dtype in CASES:
        print(f"=== {label} B={B} S={S} Skv={S_kv} Nq={N_q} {str(dtype)[6:]} causal={causal} ===",
              flush=True)
        q, k, v = _inputs(label, B, S, S_kv, N_q, N_kv, D, dtype)
        b1, ch_b = run(q, k, v, causal, S, False)
        b2, _ = run(q, k, v, causal, S, False)
        a1, ch_a = run(q, k, v, causal, S, True)
        a2, _ = run(q, k, v, causal, S, True)
        print(f"  chosen BASE={ch_b} ARM={ch_a}", flush=True)
        det_b = report(label, b1, b2, "BASE run1 vs BASE run2 (determinism)")
        det_a = report(label, a1, a2, "ARM  run1 vs ARM  run2 (determinism)")
        cross = report(label, b1, a1, "BASE vs ARM (the P2 criterion)")
        # struct-level equality (the iter5/iter7 special-value criterion)
        from run_iter8_p2_gate0 import _special_struct_equal

        ok, why = _special_struct_equal(a1.float(), b1.float())
        print(f"  special_struct_equal(BASE,ARM)={ok} ({why})", flush=True)
        verdict = ("(a) shipped chain non-deterministic on NaN payloads" if not det_b
                   else "(b) ARM non-deterministic" if not det_a
                   else "(c) deterministic in both arms, BASE!=ARM bits" if not cross
                   else "bit-identical")
        print(f"  VERDICT: {verdict}\n", flush=True)


if __name__ == "__main__":
    main()
