"""Iter5 C2 static audit — gqa_flat_kv2_variant.py vs shipped sources.

Three checks (no NPU needed):

1. SOURCE DRIFT GUARD (gen_variants_iter4 pattern): gqa.py md5 must equal the
   shipped baseline and the two母体 blocks must sit at their asserted lines
   (_gqa_kernel_prefill_g2stack 2472-2969, _gqa_kernel_prefill_g2stack_flat
   2981-3423) with boundary content assertions.

2. FLAG-COUNT MIRROR (family's highest-risk bug class): every flag-op line in
   the variant's macros / init / destroy regions must appear in the pinned
   LINE TABLE below (no stray or dropped ops — regex extraction is compared
   against the table both ways), and the table's per-mc balance computation
   (schedule multiplicities x loop multiplicities x kvpar expansion) must come
   out exactly set==wait per (direction, id), with init tokens / destroy waits
   pinned: release sems EXACTLY 1 init token (hard constraint 1 — shipped
   pacing, iter4 finding), data-ready sems 0, Q protocol (init 2, destroy 2),
   every local id residual (init,destroy) == shipped flat kv1's layout.

3. OP-LINE PROVENANCE: semantically critical arithmetic/op lines (SM/AO/FIN
   bodies, G1/G2 skeleton, fills, cross-set tails) must appear in the variant
   verbatim (after the documented index substitutions kvpar<-cur/i,
   tpar<-par) AND the pinned gqa.py source lines must still hold that exact
   text — drift on either side fails the audit.

Run: python3 examples/gqa/perf_tuning/audit_flat_kv2_iter5.py
"""

import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GQA_PY = os.path.join(os.path.dirname(HERE), "gqa.py")
VARIANT_PY = os.path.join(HERE, "gqa_flat_kv2_variant.py")

SHIPPED_MD5 = "cbce74410fee5f615489d0391b3ea620"

failures = []


def fail(msg):
    failures.append(msg)
    print(f"[AUDIT_FAIL] {msg}")


def norm(s):
    s = s.split("#")[0].strip()
    return re.sub(r"\s+", " ", s)


# ---------------------------------------------------------------- check 1
def check_drift():
    data = open(GQA_PY, "rb").read()
    md5 = hashlib.md5(data).hexdigest()
    if md5 != SHIPPED_MD5:
        fail(f"gqa.py md5 {md5} != shipped {SHIPPED_MD5} — audit table invalid")
        return None
    lines = data.decode().splitlines()
    b = [
        (2472, "def _gqa_kernel_prefill_g2stack("),
        (2969, "return main"),
        (2981, "@tilelang.jit(out_idx=[4], workspace_idx=[5, 6, 7], pass_configs=P28_EXPERT_PASS_CONFIGS)"),
        (2982, "def _gqa_kernel_prefill_g2stack_flat("),
        (3423, "return main"),
    ]
    for ln, expect in b:
        got = lines[ln - 1].strip()
        if got != expect:
            fail(f"gqa.py:{ln} boundary drift: expected {expect!r}, got {got!r}")
    print(f"[OK] drift guard: gqa.py md5 == {SHIPPED_MD5}, block boundaries intact")
    return lines


# ---------------------------------------------------------------- regions
def variant_regions():
    src = open(VARIANT_PY).read().splitlines()
    regions = {}
    idx = [i for i, l in enumerate(src) if re.match(r"    @T\.(macro|prim_func)", l)]
    for j, i in enumerate(idx):
        m = re.match(r"\s*def (emit_\w+)\(", src[i + 1])
        if not m:
            continue
        end = idx[j + 1] if j + 1 < len(idx) else next(k for k, l in enumerate(src) if "@T.prim_func" in l)
        regions[m.group(1)] = src[i + 2 : end]
    di = [i for i, l in enumerate(src) if "# destroy:" in l]
    assert len(di) == 2, f"expected 2 destroy blocks, got {len(di)}"
    regions["C_init"] = src[next(i for i, l in enumerate(src) if "# init: pretend consumers" in l):
                            next(i for i, l in enumerate(src) if "# ---- prologue: Qload" in l)]
    regions["C_destroy"] = src[di[0]:next(i for i, l in enumerate(src) if "V scope" in l and i > di[0])]
    regions["V_init"] = src[next(i for i, l in enumerate(src) if "# init: pretend producers" in l):
                          next(i for i, l in enumerate(src) if "# ---- prologue beats" in l)]
    regions["V_destroy"] = src[di[1]:]
    return regions


FLAG_LINE_RE = re.compile(r"T\.(set_flag|wait_flag|set_cross_flag|wait_cross_flag)\(")

# ---------------------------------------------------------------- check 2
# PINNED flag-op line table.  Format per macro/region:
#   normalized line text -> (occurrences in region, per-call multiplicity note)
# per-mc balance is then computed with: schedule multiplicity (calls per mc)
#   emit_g1 2, emit_g2 2, emit_qload 1, emit_sm 2, emit_ao 2, emit_fin 1
# x per-call expansion encoded in CONTRIB below (kvpar in {0,1} exactly once
# per mc for each of G1/G2/SM/AO; h-loop x2 folded into the per-call counts).

LINES = {
    "emit_qload": [
        ('T.wait_flag("MTE1", "MTE2", SIG_Q_L1)', 1),
        ('T.set_flag("MTE2", "MTE1", SIG_Q_L1)', 1),
    ],
    "emit_g1": [
        ("T.wait_cross_flag(SEM_WS1_V2C)", 1),
        ('T.wait_flag("MTE1", "MTE2", SIG_K_L1 + kvpar)', 1),
        ('T.set_flag("MTE2", "MTE1", SIG_K_L1 + kvpar)', 1),
        ('T.wait_flag("MTE1", "MTE2", SIG_V_L1 + kvpar)', 1),
        ('T.set_flag("MTE2", "MTE1", SIG_V_L1 + kvpar)', 1),
        ('T.wait_flag("M", "MTE1", SIG_L0AB + kvpar)', 1),
        ('T.wait_flag("MTE2", "MTE1", SIG_Q_L1)', 1),  # kv0 calls only -> 1/mc
        ('T.set_flag("MTE1", "MTE2", SIG_Q_L1)', 1),
        ('T.wait_flag("MTE2", "MTE1", SIG_K_L1 + kvpar)', 1),
        ('T.set_flag("MTE1", "MTE2", SIG_K_L1 + kvpar)', 1),
        ('T.set_flag("MTE1", "M", SIG_L0AB + kvpar)', 1),
        ('T.wait_flag("MTE1", "M", SIG_L0AB + kvpar)', 1),
        ('T.wait_flag("FIX", "M", SIG_L0C + kvpar)', 1),
        ('T.set_flag("M", "MTE1", SIG_L0AB + kvpar)', 1),
        ('T.set_flag("M", "FIX", SIG_L0C + kvpar)', 1),
        ('T.wait_flag("M", "FIX", SIG_L0C + kvpar)', 1),
        ('T.set_flag("FIX", "M", SIG_L0C + kvpar)', 1),
        ('T.set_cross_flag("FIX", SEM_WS1_C2V)', 1),
    ],
    "emit_g2": [
        ("T.wait_cross_flag(SEM_WS3_V2C)", 1),
        ('T.wait_flag("MTE1", "MTE2", SIG_P_L1)', 1),
        ("T.wait_cross_flag(SEM_WS2_V2C)", 1),
        ('T.set_flag("MTE2", "MTE1", SIG_P_L1)', 1),
        ('T.wait_flag("MTE2", "MTE1", SIG_V_L1 + kvpar)', 1),
        ('T.wait_flag("M", "MTE1", SIG_L0AB + kvpar)', 1),
        ('T.set_flag("MTE1", "MTE2", SIG_V_L1 + kvpar)', 1),
        ('T.wait_flag("MTE2", "MTE1", SIG_P_L1)', 1),
        ('T.set_flag("MTE1", "MTE2", SIG_P_L1)', 1),
        ('T.set_flag("MTE1", "M", SIG_L0AB + kvpar)', 1),
        ('T.wait_flag("MTE1", "M", SIG_L0AB + kvpar)', 1),
        ('T.wait_flag("FIX", "M", SIG_L0C + kvpar)', 1),
        ('T.set_flag("M", "MTE1", SIG_L0AB + kvpar)', 1),
        ('T.set_flag("M", "FIX", SIG_L0C + kvpar)', 1),
        ('T.wait_flag("M", "FIX", SIG_L0C + kvpar)', 1),
        ('T.set_flag("FIX", "M", SIG_L0C + kvpar)', 1),
        ('T.set_cross_flag("FIX", SEM_WS3_C2V)', 1),
        ('T.set_cross_flag("MTE2", SEM_WS2_C2V)', 1),
    ],
    # emit_sm: per-call counts already include the h-loop x2 where applicable
    "emit_sm": [
        ("T.wait_cross_flag(SEM_WS2_C2V)", 1),
        ("T.wait_cross_flag(SEM_WS1_C2V)", 1),
        ('T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)', 1, 2),
        ('T.set_flag("MTE2", "V", SIG_IO_UB + io_side)', 1, 2),
        ('T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)', 1, 2),
        ('T.set_flag("V", "MTE2", SIG_IO_UB + io_side)', 1, 2),
        ('T.set_flag("V", "MTE2", SIG_MASK)', 1, 2),   # use_mask=True trace only
        ('T.wait_flag("V", "MTE2", SIG_MASK)', 1, 2),  # use_mask=True trace only
        ('T.set_flag("MTE2", "V", SIG_MASK)', 1, 2),   # use_mask=True trace only
        ('T.wait_flag("MTE2", "V", SIG_MASK)', 1, 2),  # use_mask=True trace only
        ('T.wait_flag("MTE3", "V", SIG_S_HALF)', 1, 2),
        ('T.set_flag("V", "MTE3", SIG_S_HALF)', 1, 2),
        ('T.wait_flag("V", "MTE3", SIG_S_HALF)', 1, 2),
        ('T.set_flag("MTE3", "V", SIG_S_HALF)', 1, 2),
        ('T.set_cross_flag("MTE3", SEM_WS2_V2C)', 1),
        ('T.set_cross_flag("MTE2", SEM_WS1_V2C)', 1),
    ],
    "emit_ao": [
        ("T.wait_cross_flag(SEM_WS3_C2V)", 1),
        ('T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)', 1, 2),
        ('T.set_flag("MTE2", "V", SIG_IO_UB + io_side)', 1, 2),
        ('T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)', 1, 2),
        ('T.set_flag("V", "MTE2", SIG_IO_UB + io_side)', 1, 2),
        ('T.set_cross_flag("MTE2", SEM_WS3_V2C)', 1),
    ],
    "emit_fin": [
        ('T.wait_flag("MTE3", "V", SIG_S_HALF)', 1, 2),
        ('T.set_flag("V", "MTE3", SIG_S_HALF)', 1, 2),
        ('T.wait_flag("V", "MTE3", SIG_S_HALF)', 1, 2),
        ('T.set_flag("MTE3", "V", SIG_S_HALF)', 1, 2),
    ],
    "C_init": [
        ('T.set_cross_flag("MTE2", SEM_WS2_C2V)', 1),
        ('T.set_flag("MTE1", "MTE2", SIG_K_L1)', 1),
        ('T.set_flag("MTE1", "MTE2", SIG_K_L1 + 1)', 1),
        ('T.set_flag("MTE1", "MTE2", SIG_P_L1)', 1),
        ('T.set_flag("MTE1", "MTE2", SIG_V_L1)', 1),
        ('T.set_flag("MTE1", "MTE2", SIG_V_L1 + 1)', 1),
        ('T.set_flag("MTE1", "MTE2", SIG_Q_L1)', 1),
        ('T.set_flag("M", "MTE1", SIG_L0AB)', 1),
        ('T.set_flag("M", "MTE1", SIG_L0AB + 1)', 1),
        ('T.set_flag("FIX", "M", SIG_L0C)', 1),
        ('T.set_flag("FIX", "M", SIG_L0C + 1)', 1),
    ],
    "C_destroy": [
        ('T.wait_flag("MTE1", "MTE2", SIG_K_L1)', 1),
        ('T.wait_flag("MTE1", "MTE2", SIG_K_L1 + 1)', 1),
        ('T.wait_flag("MTE1", "MTE2", SIG_P_L1)', 1),
        ('T.wait_flag("MTE1", "MTE2", SIG_V_L1)', 1),
        ('T.wait_flag("MTE1", "MTE2", SIG_V_L1 + 1)', 1),
        ('T.wait_flag("MTE1", "MTE2", SIG_Q_L1)', 1),
        ('T.wait_flag("M", "MTE1", SIG_L0AB)', 1),
        ('T.wait_flag("M", "MTE1", SIG_L0AB + 1)', 1),
        ('T.wait_flag("FIX", "M", SIG_L0C)', 1),
        ('T.wait_flag("FIX", "M", SIG_L0C + 1)', 1),
    ],
    "V_init": [
        ('T.set_cross_flag("MTE2", SEM_WS1_V2C)', 1),
        ('T.set_cross_flag("MTE2", SEM_WS3_V2C)', 1),
        ('T.set_flag("V", "MTE2", SIG_IO_UB)', 1),
        ('T.set_flag("V", "MTE2", SIG_IO_UB + 1)', 1),
        ('T.set_flag("MTE3", "V", SIG_S_HALF)', 1),
    ],
    "V_destroy": [
        ('T.wait_flag("V", "MTE2", SIG_IO_UB)', 1),
        ('T.wait_flag("V", "MTE2", SIG_IO_UB + 1)', 1),
        ('T.wait_flag("MTE3", "V", SIG_S_HALF)', 1),
    ],
}

# per-mc (direction, literal-id) contributions: (sets, waits).
# Derived BY HAND from LINES x schedule multiplicity (design §B.3) — the
# independent derivation the balance pins below check against.
CONTRIB_PER_MC = {
    # cross sems (U = 2 per mc for every stream op)
    ("X", "SEM_WS1_C2V"): (2, 2),   # G1 sets x2; SM waits x2
    ("X", "SEM_WS1_V2C"): (2, 2),   # SM sets x2; G1 waits x2  (+1 init)
    ("X", "SEM_WS2_V2C"): (2, 2),   # SM sets x2; G2 waits x2
    ("X", "SEM_WS2_C2V"): (2, 2),   # G2 sets x2; SM waits x2  (+1 init)
    ("X", "SEM_WS3_C2V"): (2, 2),   # G2 sets x2; AO waits x2
    ("X", "SEM_WS3_V2C"): (2, 2),   # AO sets x2; G2 waits x2  (+1 init)
    # local — K (id0/1 via kvpar: G1 kv0 + G1 kv1 = 1 each per mc)
    ("MTE1->MTE2", "SIG_K_L1+0"): (1, 1),
    ("MTE1->MTE2", "SIG_K_L1+1"): (1, 1),
    ("MTE2->MTE1", "SIG_K_L1+0"): (1, 1),
    ("MTE2->MTE1", "SIG_K_L1+1"): (1, 1),
    # local — V slots (G1 prefetch set + G2 release set; G1 wait + G2 wait)
    ("MTE1->MTE2", "SIG_V_L1+0"): (1, 1),
    ("MTE1->MTE2", "SIG_V_L1+1"): (1, 1),
    ("MTE2->MTE1", "SIG_V_L1+0"): (1, 1),
    ("MTE2->MTE1", "SIG_V_L1+1"): (1, 1),
    # local — P (G2 x2 per mc)
    ("MTE1->MTE2", "SIG_P_L1"): (2, 2),
    ("MTE2->MTE1", "SIG_P_L1"): (2, 2),
    # local — Q (single-token protocol: free set only by G1 kv1 — MTE1 in-order
    # drain implies both reads done; full wait only by G1 kv0)
    ("MTE1->MTE2", "SIG_Q_L1"): (1, 1),
    ("MTE2->MTE1", "SIG_Q_L1"): (1, 1),
    # local — L0AB / L0C (G1+G2 per parity)
    ("MTE1->M", "SIG_L0AB+0"): (2, 2),
    ("MTE1->M", "SIG_L0AB+1"): (2, 2),
    ("M->MTE1", "SIG_L0AB+0"): (2, 2),
    ("M->MTE1", "SIG_L0AB+1"): (2, 2),
    ("M->FIX", "SIG_L0C+0"): (2, 2),
    ("M->FIX", "SIG_L0C+1"): (2, 2),
    ("FIX->M", "SIG_L0C+0"): (2, 2),
    ("FIX->M", "SIG_L0C+1"): (2, 2),
    # local — V-scope IO (SM x2 calls x2h /2 per side + AO same): per side per mc:
    #   SM: 2 calls, each h-loop touches side0 once & side1 once -> 2 per side
    #   AO: same -> 2 per side; total 4 sets + 4 waits per side per mc
    ("V->MTE2", "SIG_IO_UB+0"): (4, 4),
    ("V->MTE2", "SIG_IO_UB+1"): (4, 4),
    ("MTE2->V", "SIG_IO_UB+0"): (4, 4),
    ("MTE2->V", "SIG_IO_UB+1"): (4, 4),
    # local — MASK (SM only, 2 calls/mc x 2h; only in use_mask traces)
    ("V->MTE2", "SIG_MASK"): (4, 4),
    ("MTE2->V", "SIG_MASK"): (4, 4),
    # local — S_HALF (SM 2 calls x 2h = 4; FIN 1 call x 2h = 2) -> 6/6
    ("MTE3->V", "SIG_S_HALF"): (6, 6),
    ("V->MTE3", "SIG_S_HALF"): (6, 6),
}

INIT_TOKENS = {  # (dir,id) -> count
    ("X", "SEM_WS2_C2V"): 1,
    ("X", "SEM_WS1_V2C"): 1,
    ("X", "SEM_WS3_V2C"): 1,
    ("MTE1->MTE2", "SIG_K_L1+0"): 1,
    ("MTE1->MTE2", "SIG_K_L1+1"): 1,
    ("MTE1->MTE2", "SIG_P_L1"): 1,
    ("MTE1->MTE2", "SIG_V_L1+0"): 1,
    ("MTE1->MTE2", "SIG_V_L1+1"): 1,
    ("MTE1->MTE2", "SIG_Q_L1"): 1,
    ("M->MTE1", "SIG_L0AB+0"): 1,
    ("M->MTE1", "SIG_L0AB+1"): 1,
    ("FIX->M", "SIG_L0C+0"): 1,
    ("FIX->M", "SIG_L0C+1"): 1,
    ("V->MTE2", "SIG_IO_UB+0"): 1,
    ("V->MTE2", "SIG_IO_UB+1"): 1,
    ("MTE3->V", "SIG_S_HALF"): 1,
}

DESTROY_WAITS = {  # (dir,id) -> count
    ("MTE1->MTE2", "SIG_K_L1+0"): 1,
    ("MTE1->MTE2", "SIG_K_L1+1"): 1,
    ("MTE1->MTE2", "SIG_P_L1"): 1,
    ("MTE1->MTE2", "SIG_V_L1+0"): 1,
    ("MTE1->MTE2", "SIG_V_L1+1"): 1,
    ("MTE1->MTE2", "SIG_Q_L1"): 1,
    ("M->MTE1", "SIG_L0AB+0"): 1,
    ("M->MTE1", "SIG_L0AB+1"): 1,
    ("FIX->M", "SIG_L0C+0"): 1,
    ("FIX->M", "SIG_L0C+1"): 1,
    ("V->MTE2", "SIG_IO_UB+0"): 1,
    ("V->MTE2", "SIG_IO_UB+1"): 1,
    ("MTE3->V", "SIG_S_HALF"): 1,
}


def key_of(line):
    """Map a normalized flag-op line to (direction, literal-id-key) sets."""

    def nid(t):
        t = re.sub(r"\s*\+\s*", "+", t)
        if re.fullmatch(r"SIG_(K_L1|V_L1|L0AB|L0C|IO_UB)", t):
            t += "+0"  # bare form == +0 slot
        return t

    m = re.search(r'T\.set_cross_flag\("(\w+)", (\w+)\)', line)
    if m:
        return [("set", ("X", m.group(2)))]
    m = re.search(r"T\.wait_cross_flag\((\w+)\)", line)
    if m:
        return [("wait", ("X", m.group(1)))]
    m = re.search(r'T\.set_flag\("(\w+)", "(\w+)", (.+)\)', line)
    if m:
        return [("set", (f"{m.group(1)}->{m.group(2)}", nid(m.group(3))))]
    m = re.search(r'T\.wait_flag\("(\w+)", "(\w+)", (.+)\)', line)
    if m:
        return [("wait", (f"{m.group(1)}->{m.group(2)}", nid(m.group(3))))]
    raise AssertionError("unparsed flag line: " + line)


def check_flag_mirror():
    regions = variant_regions()
    print("\n--- flag line extraction vs pinned table ---")
    for region, table in LINES.items():
        if region not in regions:
            fail(f"region {region} missing")
            continue
        got = {}
        for ln in regions[region]:
            n = norm(ln)
            if FLAG_LINE_RE.search(n):
                got[n] = got.get(n, 0) + 1
        want = {norm(e[0]): e[1] for e in table}
        for t, c in want.items():
            if got.get(t, 0) != c:
                fail(f"{region}: line {t!r} expected x{c}, found x{got.get(t, 0)}")
        for t in got:
            if t not in want:
                fail(f"{region}: UNPINNED flag line {t!r} (x{got[t]})")
        if all(got.get(t, 0) == c for t, c in want.items()) and set(got) <= set(want):
            print(f"  [OK] {region}: {sum(want.values())} flag ops match the pinned table")

    # ---- balance from the table (independent re-derivation) ----
    print("\n--- balance (recomputed from LINES table; per-mc) ---")
    MULT = {"emit_g1": 2, "emit_g2": 2, "emit_qload": 1, "emit_sm": 2, "emit_ao": 2, "emit_fin": 1}
    tally = {}
    for region, table in LINES.items():
        if region.startswith("emit_"):
            calls_per_mc = MULT[region]
        else:
            calls_per_mc = None  # constant regions
        for entry in table:
            text, occ = entry[0], entry[1]
            loop = entry[2] if len(entry) > 2 else 1
            if "SIG_MASK" in text and region == "emit_sm":
                pass  # mask lines: use_mask=True trace; balance checked separately as self-paired
            for kind, (d, idt) in key_of(norm(text)):
                # expand kvpar / io_side: each of the 2 calls per mc (or the
                # 2-iteration h loop) instantiates ONE slot value, so each
                # literal id receives occ * eff / 2
                if "kvpar" in idt:
                    if region == "emit_g1" and idt == "SIG_Q_L1":
                        ids = [idt]  # Q lines sit inside static kvpar branches
                    else:
                        ids = [idt.replace("+kvpar", "+0"), idt.replace("+kvpar", "+1")]
                elif "io_side" in idt:
                    ids = [idt.replace("+io_side", "+0"), idt.replace("+io_side", "+1")]
                else:
                    ids = [idt]
                for lit in ids:
                    k = (d, lit)
                    e = tally.setdefault(k, [0, 0, 0, 0])  # sets/mc, waits/mc, init, destroy
                    if calls_per_mc is None:
                        # constant region: occ contributes to init/destroy once
                        if kind == "set":
                            e[2] += occ
                        else:
                            e[3] += occ
                    else:
                        # per-call occurrence x calls-per-mc; the kv0-only Q
                        # full-wait inside emit_g1 fires on 1 of the 2 calls/mc
                        eff = calls_per_mc
                        if region == "emit_g1" and idt == "SIG_Q_L1":
                            eff = 1  # full-wait fires on kv0 call only; free-set on kv1 call only
                        share = occ * loop * eff / len(ids)
                        if kind == "set":
                            e[0] += share
                        else:
                            e[1] += share

    ok = True
    for k in sorted(set(tally) | set(CONTRIB_PER_MC)):
        s, w, ini, des = tally.get(k, [0, 0, 0, 0])
        cs, cw = CONTRIB_PER_MC.get(k, (0, 0))
        ci = INIT_TOKENS.get(k, 0)
        cd = DESTROY_WAITS.get(k, 0)
        problems = []
        if s != w:
            problems.append(f"sets/mc {s} != waits/mc {w}")
        if (s, w) != (cs, cw):
            problems.append(f"table ({s},{w}) != hand-derived ({cs},{cw})")
        if ini != ci:
            problems.append(f"init {ini} != pinned {ci}")
        if des != cd:
            problems.append(f"destroy {des} != pinned {cd}")
        mark = "OK" if not problems else "FAIL"
        if problems:
            ok = False
            fail(f"balance {k}: " + "; ".join(problems))
        print(f"  [{mark}] {str(k):42s} sets/mc={s} waits/mc={w} init={ini} destroy={des}")
    # MASK self-pair check (use_mask traces): sets == waits per direction
    for d in ("V->MTE2", "MTE2->V"):
        s, w, _, _ = tally.get((d, "SIG_MASK"), [0, 0, 0, 0])
        if s != w:
            fail(f"MASK {d}: sets/mc {s} != waits/mc {w}")
    print(f"  balance recomputation: {'CONSISTENT' if ok else 'INCONSISTENT'}")


# ---------------------------------------------------------------- check 3
PROVENANCE = [
    # (variant op line (normalized compare), gqa.py lineno, expected gqa text)
    # --- emit_sm vs g2stack phase-1 (i -> kvpar, cur -> kvpar, prv -> 1-kvpar) ---
    ("T.reduce_max(work_ub, neg_sm[h * 2 + kvpar, :, :], dim=-1)", 2892, "T.reduce_max(work_ub, neg_sm[h * 2 + cur, :, :], dim=-1)"),
    ("T.tile.mul(neg_sm[h * 2 + kvpar, :, :], neg_sm[h * 2 + kvpar, :, :], -sm_scale)", 2893, "T.tile.mul(neg_sm[h * 2 + cur, :, :], neg_sm[h * 2 + cur, :, :], -sm_scale)"),
    ("T.tile.min(neg_sm[h * 2 + kvpar, :, :], neg_sm[h * 2 + kvpar, :, :], neg_sm[h * 2 + (1 - kvpar), :, :])", 2894, "T.tile.min(neg_sm[h * 2 + cur, :, :], neg_sm[h * 2 + cur, :, :], neg_sm[h * 2 + prv, :, :])"),
    ("T.tile.broadcast(buf_2d, neg_sm[h * 2 + kvpar, :, :])", 2896, "T.tile.broadcast(buf_2d, neg_sm[h * 2 + cur, :, :])"),
    ("T.tile.axpy(buf_2d, work_ub, sm_scale)", 2897, "T.tile.axpy(buf_2d, work_ub, sm_scale)"),
    ("T.tile.exp(work_ub, buf_2d)", 2898, "T.tile.exp(work_ub, buf_2d)"),
    ("T.reduce_sum(work_ub, sumexp_is[kvpar * 2 + h, :, :], dim=-1)", 2900, "T.reduce_sum(work_ub, sumexp_is[i * 2 + h, :, :], dim=-1)"),
    ("T.tile.sub(r_factors[kvpar * 2 + h, :, :], neg_sm[h * 2 + kvpar, :, :], neg_sm[h * 2 + (1 - kvpar), :, :])", 2901, "T.tile.sub(r_factors[i * 2 + h, :, :], neg_sm[h * 2 + cur, :, :], neg_sm[h * 2 + prv, :, :])"),
    # --- emit_ao vs g2stack phase-2 (i -> kvpar; row state h -> tpar*2+h) ---
    ("T.tile.exp(r_factors[kvpar * 2 + h, :, :], r_factors[kvpar * 2 + h, :, :])", 2924, "T.tile.exp(r_factors[i * 2 + h, :, :], r_factors[i * 2 + h, :, :])"),
    ("T.tile.mul(sumexp[tpar * 2 + h, :, :], sumexp[tpar * 2 + h, :, :], r_factors[kvpar * 2 + h, :, :])", 2925, "T.tile.mul(sumexp[h, :, :], sumexp[h, :, :], r_factors[i * 2 + h, :, :])"),
    ("T.tile.add(sumexp[tpar * 2 + h, :, :], sumexp[tpar * 2 + h, :, :], sumexp_is[kvpar * 2 + h, :, :])", 2926, "T.tile.add(sumexp[h, :, :], sumexp[h, :, :], sumexp_is[i * 2 + h, :, :])"),
    ("T.tile.broadcast(buf_2d, r_factors[kvpar * 2 + h, :, :])", 2927, "T.tile.broadcast(buf_2d, r_factors[i * 2 + h, :, :])"),
    ("T.tile.mul(acc_o[tpar * 2 + h, :, :], acc_o[tpar * 2 + h, :, :], buf_2d)", 2928, "T.tile.mul(acc_o[h, :, :], acc_o[h, :, :], buf_2d)"),
    ("T.tile.add(acc_o[tpar * 2 + h, :, :], acc_o[tpar * 2 + h, :, :], work_ub)", 2940, "T.tile.add(acc_o[h, :, :], acc_o[h, :, :], work_ub)"),
    # --- emit_fin vs g2stack finalize ---
    ("T.tile.add(sumexp[tpar * 2 + h, :, :], sumexp[tpar * 2 + h, :, :], 1e-30)", 2951, "T.tile.add(sumexp[h, :, :], sumexp[h, :, :], 1e-30)"),
    ("T.tile.broadcast(buf_2d, sumexp[tpar * 2 + h, :, :])", 2952, "T.tile.broadcast(buf_2d, sumexp[h, :, :])"),
    ("T.tile.div(acc_o[tpar * 2 + h, :, :], acc_o[tpar * 2 + h, :, :], buf_2d)", 2953, "T.tile.div(acc_o[h, :, :], acc_o[h, :, :], buf_2d)"),
    # --- per-task fills: acc_o/sumexp from flat kv1 [S3-iter2-B fix], neg_sm cap from g2stack ---
    ("T.tile.fill(acc_o[tpar * 2 + h, :, :], 0.0)", 3234, "T.tile.fill(acc_o[par * 2 + h, :, :], 0.0)"),
    ("T.tile.fill(sumexp[tpar * 2 + h, :, :], 0.0)", 3235, "T.tile.fill(sumexp[par * 2 + h, :, :], 0.0)"),
    ("T.tile.fill(neg_sm, neg_sm_cap)", 2829, "T.tile.fill(neg_sm, neg_sm_cap)"),
    # --- C-side GEMM skeleton vs g2stack G1/G2 (side/i -> kvpar) ---
    ("T.mma(l0a[kvpar, :, :], l0b[kvpar, :, :], l0c[kvpar, :, :], init=True)", 2734, "T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)"),
    ("T.copy(l0c[kvpar, :, :], ws_s[cid, kvpar, :, :])", 2739, "T.copy(l0c[side, :, :], ws_s[cid, i, :, :])"),
    ("T.copy(l0c[kvpar, :, :], ws_o[cid, kvpar, :, :])", 2789, "T.copy(l0c[side, :, :], ws_o[cid, i, :, :])"),
    ("T.copy(ws_p[cid, kvpar, :, :], p_l1)", 2766, "T.copy(ws_p[cid, i, :, :], p_l1)"),
    ("T.copy(q_l1, l0a[kvpar, :, :])", 2722, "T.copy(q_l1, l0a[side, :, :])"),
    # --- V-side ws reads vs g2stack ---
    ("T.copy(ws_s[cid, kvpar, row0 : row0 + half, :], io_buf[io_side, :, :])", 2859, "T.copy(ws_s[cid, i, row0 : row0 + half, :], io_buf[io_side, :, :])"),
    ("T.copy(ws_o[cid, kvpar, row0 : row0 + half, :], io_buf[io_side, :, :])", 2933, "T.copy(ws_o[cid, i, row0 : row0 + half, :], io_buf[io_side, :, :])"),
    ("T.copy(acc_s_half, ws_p[cid, kvpar, row0 : row0 + half, :])", 2909, "T.copy(acc_s_half, ws_p[cid, i, row0 : row0 + half, :])"),
    # --- cross-flag sets at macro tails vs flat kv1 (pipe attributions) ---
    ('T.set_cross_flag("FIX", SEM_WS1_C2V)', 3119, 'T.set_cross_flag("FIX", SEM_WS1_C2V)'),
    ('T.set_cross_flag("MTE3", SEM_WS2_V2C)', 3213, 'T.set_cross_flag("MTE3", SEM_WS2_V2C)'),
    ('T.set_cross_flag("MTE2", SEM_WS1_V2C)', 3214, 'T.set_cross_flag("MTE2", SEM_WS1_V2C)'),
    ('T.set_cross_flag("FIX", SEM_WS3_C2V)', 3150, 'T.set_cross_flag("FIX", SEM_WS3_C2V)'),
    ('T.set_cross_flag("MTE2", SEM_WS2_C2V)', 3151, 'T.set_cross_flag("MTE2", SEM_WS2_C2V)'),
    ('T.set_cross_flag("MTE2", SEM_WS3_V2C)', 3252, 'T.set_cross_flag("MTE2", SEM_WS3_V2C)'),
]


def check_provenance(gqa_lines):
    vset = {}
    for l in open(VARIANT_PY).read().splitlines():
        n = norm(l)
        if n:
            vset[n] = vset.get(n, 0) + 1
    print("\n--- op-line provenance (variant <-> pinned gqa.py line) ---")
    n_ok = 0
    for vline, gln, gexpect in PROVENANCE:
        gn = norm(gqa_lines[gln - 1])
        if gn != norm(gexpect):
            fail(f"gqa.py:{gln} drifted: expected {norm(gexpect)!r} got {gn!r}")
        if norm(vline) not in vset:
            fail(f"variant missing proven op line: {norm(vline)!r}")
        else:
            n_ok += 1
    print(f"  [OK] {n_ok}/{len(PROVENANCE)} provenance pairs verified (both sides)")


def main():
    gqa_lines = check_drift()
    if gqa_lines is None:
        sys.exit(1)
    check_flag_mirror()
    check_provenance(gqa_lines)
    print()
    if failures:
        print(f"[AUDIT] {len(failures)} FAILURES")
        sys.exit(1)
    print("[AUDIT] ALL CHECKS PASSED (drift guard + flag-count mirror + op-line provenance)")
    sys.exit(0)


if __name__ == "__main__":
    main()
