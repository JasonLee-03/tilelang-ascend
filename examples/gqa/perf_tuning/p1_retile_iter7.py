"""Iter7 P1 — transpose_out retiling rule (host-only; gqa.py NOT modified).

Standalone development file (Stage 3 iter7).  Provides `out_plan_p1(B, Nq, S,
D)`, a drop-in replacement for gqa.out_plan with the SAME contract
(('custom', bm, bn) | None) and the same hard constraints:

  * bn divides S exactly (the _transpose_v1 tile loop over N is exact);
  * bm tail-safe by the kernel's `m_idx < M` guard (no divisibility need);
  * ub tile [bn, R] within budget (bn <= legacy bn0 <= 128, unchanged);
  * D*2 % 32 != 0 -> None (caller fallback, unchanged);
  * HEALTHY grids keep the legacy (bm, bn) BYTE-IDENTICAL whenever the
    retile model gain is < 20% (hysteresis) — zero perturbation wherever
    iter6 found no starvation payoff (e.g. R4/c4-like, c8, c9, c10).

== History (this file is the V2 rule; V1 kept below for provenance) ==

V1 (measured 2026-10-19, msprof_op_iter7_P1_*): scored configs by the
critical-path byte factor g = waves/blk ONLY (min g, eligibility band
1.10*g*, max-area tie-break, 0.8 hysteresis).  Result: R1 e2e +24.4%,
R6 +10.6%, R3 +4.3% (ADOPT-worthy primaries) BUT R2 -16.8% REGRESSION —
preregistered guardrail tripped => V1 rule REJECTED (data kept).
Failure isolation (same-session measurements, parse_iter7_ab.py output):
  tout(blk, waves, bytes/block):  R2 v1 (1,32): 128 blk / 7 waves / 32KB
      -> 21.66us  (g-model predicted ~6us: WRONG)
  R3 v1 (1,128): 128 blk / 7 waves / 125KB -> 21.07us
  R6 v1 (1,64):   96 blk / 5 waves /  64KB -> 15.71us
  R1 v1 (1,32):   16 blk / 1 wave  /  32KB ->  5.06us
  base R2 (8,128): 4 blk / 1 wave / 1MB    -> 13.56us
  base R3 (16,128):8 blk / 1 wave / 2MB    -> 24.22us
  base R4 (16,128):16 blk/ 1 wave / 2MB    -> 29.53us
  base R6 (16,128):3 blk / 1 wave / 2MB    -> 22.14us
Every multi-wave small-block config undershot the g-model by ~2.5us PER
WAVE: wall ~= L + waves * max(W_OH, bytes_per_block / rate) with
L ~= 3us (launch floor, iter3), W_OH ~= 2.5us (block-wave dispatch/drain
overhead — NEW calibration, this session), rate ~= 80GB/s per critical
core (iter6 family range 68-105).  V2 scores with THAT calibrated model.

V2 rule (deterministic, exact same mechanism class — host-only plan):
  1. candidates: bm in pow2[1..bm0], bn in pow2[bn_lo..bn0] (bn_lo keeps
     the contiguous read burst >= 2KB: bn*D*2 >= 2048, capped at bn0);
  2. pred_wall(bm,bn) = L + waves * max(W_OH, bpb/rate), bpb = total_bytes
     / blk, total_bytes = B*Nq*S*D*2B*2(r+w) (fp16/bf16 only on this path);
  3. pick min (round(pred,1), waves, -total, -bn, -bm)  [ties: FEWEST WAVES
     first — every measured wave-1 fat-block config in this family fits the
     model (68-105 GB/s) while the V1 regression came from unmodeled
     multi-wave overhead; then more tasks (parallelism robustness), then
     bigger boxes];
  4. HYSTERESIS: keep legacy unless pred_chosen <= 0.8 * pred_legacy.

The kernel body (_transpose_v1) is NOT touched; the task->data mapping is a
pure re-partition of the same byte set, every output element written exactly
once => outputs must be BIT-IDENTICAL (Gate 0 criterion, re-run for V2).

Run the plan predictor (no NPU needed):
    python3 examples/gqa/perf_tuning/p1_retile_iter7.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gqa import P28_NUM_CORES, _ceildiv  # noqa: E402

_SLOTS = 2 * P28_NUM_CORES  # 40 vector task slots (20 cores x 2 vids)
_VEC_NUM = 2                # gqa._VEC_NUM (2 vids per block)

# --- V2 model constants (calibrated on THIS session's tout measurements) ---
_L_US = 3.0          # launch floor (iter3 §4: 2.5-3.8us across six conditions)
_W_OH_US = 2.5       # per-block-wave dispatch/drain overhead (V1 failure fit)
_RATE_B_PER_US = 80e3  # ~80 GB/s critical-path per-core (family range 68-105)


def legacy_plan(B, Nq, S, D):
    """gqa.out_plan verbatim (the shipped rule) — reference arm."""
    bm = 16 if Nq >= 48 else 8
    bn = 128
    while bn > 1 and S % bn != 0:
        bn //= 2
    if D * 2 % 32 != 0:
        return None
    return ("custom", bm, bn)


def _grid(B, Nq, S, bm, bn):
    total = B * _ceildiv(Nq, bm) * (S // bn)  # bn | S by construction
    blk = _ceildiv(total, _VEC_NUM)
    waves = _ceildiv(blk, P28_NUM_CORES)
    return total, blk, waves


def _pred_wall(B, Nq, S, D, bm, bn):
    total, blk, waves = _grid(B, Nq, S, bm, bn)
    total_bytes = B * Nq * S * D * 2 * 2  # r+w, 2B/elem (fp16/bf16 path)
    bpb = total_bytes / max(1, blk)
    return _L_US + waves * max(_W_OH_US, bpb / _RATE_B_PER_US), total, blk, waves


def out_plan_p1(B, Nq, S, D):
    """V2 drop-in out_plan replacement (see module header for derivation)."""
    base = legacy_plan(B, Nq, S, D)
    if base is None:
        return None
    bm0, bn0 = base[1], base[2]
    total0, blk0, waves0 = _grid(B, Nq, S, bm0, bn0)
    if total0 >= _SLOTS:
        return base  # healthy grid: byte-identical legacy plan

    pred_leg, _, _, _ = _pred_wall(B, Nq, S, D, bm0, bn0)

    # read-burst floor: bn*D*2 >= 2048 bytes, never above the legacy bn0
    bn_lo = max(1, min(1024 // D, bn0))

    best = None
    bm = bm0
    while bm >= 1:
        bn = bn0
        while bn >= bn_lo:
            pred, total, blk, waves = _pred_wall(B, Nq, S, D, bm, bn)
            key = (round(pred, 1), waves, -total, -bn, -bm)
            if best is None or key < best[0]:
                best = (key, bm, bn, pred)
            bn //= 2
        bm //= 2

    _, bm_c, bn_c, pred_c = best
    if pred_c > 0.8 * pred_leg:
        return base  # model gain < 20%: not worth perturbing this grid
    return ("custom", bm_c, bn_c)


# ---------------------------------------------------------------------------
# V1 (REJECTED by the preregistered R2 guardrail — kept for provenance)
# ---------------------------------------------------------------------------
def out_plan_p1_v1(B, Nq, S, D):
    from fractions import Fraction

    base = legacy_plan(B, Nq, S, D)
    if base is None:
        return None
    bm0, bn0 = base[1], base[2]
    total0, blk0, waves0 = _grid(B, Nq, S, bm0, bn0)
    if total0 >= _SLOTS:
        return base
    bn_lo = max(1, min(1024 // D, bn0))

    def g_of(bm, bn):
        total, blk, waves = _grid(B, Nq, S, bm, bn)
        return Fraction(waves, blk), total

    g_leg, _ = g_of(bm0, bn0)
    cands = []
    bm = bm0
    while bm >= 1:
        bn = bn0
        while bn >= bn_lo:
            g, total = g_of(bm, bn)
            cands.append((g, -(bm * bn), -bn, -bm, bm, bn))
            bn //= 2
        bm //= 2
    g_star = min(c[0] for c in cands)
    eligible = [c for c in cands if c[0] <= Fraction(11, 10) * g_star]
    chosen = min(eligible, key=lambda c: (c[1], c[2], c[3]))
    if chosen[0] > Fraction(4, 5) * g_leg:
        return base
    return ("custom", chosen[4], chosen[5])


# ---------------------------------------------------------------------------
# plan predictor: legacy vs V1 vs V2 across judgement / bench / suite shapes
# (V2-measured tout anchors from msprof_op_iter7_P1_* in comments)
# ---------------------------------------------------------------------------
SHAPES = [
    # (label, B, Nq, S, D, note)
    ("R1", 1, 8, 128, 128, "v1 (1,32) measured tout 5.06us (base 12.17)"),
    ("R2/c1/c20", 2, 32, 128, 128, "v1 (1,32) REGRESSED 21.66 (base 13.56)"),
    ("R3/c3", 2, 128, 128, 128, "v1 (1,128) 21.07 (base 24.22)"),
    ("R4/c4", 4, 64, 256, 128, "v1 kept legacy; base tout 29.53"),
    ("R6/c2", 2, 48, 128, 128, "v1 (1,64) 15.71 (base 22.14)"),
    ("c5", 2, 32, 128, 256, "D=256"),
    ("c6", 2, 128, 256, 256, "D=256"),
    ("c7", 2, 32, 128, 128, "cross Sq128"),
    ("c8", 4, 32, 1024, 128, "healthy"),
    ("c9", 4, 64, 512, 128, "healthy"),
    ("c10", 4, 128, 512, 128, "healthy"),
    ("l1_s_tail", 2, 32, 124, 128, "bn0=4"),
    ("l1_both_tail", 4, 64, 252, 128, "bn0=4"),
    ("suite_padrow", 2, 32, 100, 128, "bn0=4"),
    ("suite_small", 1, 8, 100, 128, "v1 retiled (1,4); v2 model: launch-bound"),
    ("suite_s24", 1, 8, 24, 128, "v1 retiled (1,8); v2: launch-bound"),
    ("d_align_fail", 2, 32, 128, 88, "None unchanged"),
]


def main():
    print(f"{'shape':<15}{'legacy':<11}{'tot/blk/wv':<13}{'V1(rej)':<10}{'V2':<11}"
          f"{'tot/blk/wv':<13}{'pred_leg':<10}{'pred_v2':<9}note")
    for label, B, Nq, S, D, note in SHAPES:
        leg = legacy_plan(B, Nq, S, D)
        v1 = out_plan_p1_v1(B, Nq, S, D)
        v2 = out_plan_p1(B, Nq, S, D)
        if leg is None:
            print(f"{label:<15}{'None':<11}{'-':<13}{'None':<10}"
                  f"{'None' if v2 is None else str(v2[1:]):<11}{'-':<13}{'-':<10}{'-':<9}{note}")
            continue
        t0, b0, w0 = _grid(B, Nq, S, leg[1], leg[2])
        t2, b2, w2 = _grid(B, Nq, S, v2[1], v2[2])
        pl, _, _, _ = _pred_wall(B, Nq, S, D, leg[1], leg[2])
        p2, _, _, _ = _pred_wall(B, Nq, S, D, v2[1], v2[2])
        print(f"{label:<15}{str(leg[1:]):<11}{f'{t0}/{b0}/{w0}':<13}{str(v1[1:]):<10}"
              f"{str(v2[1:]):<11}{f'{t2}/{b2}/{w2}':<13}{pl:<10.2f}{p2:<9.2f}{note}")


if __name__ == "__main__":
    main()
