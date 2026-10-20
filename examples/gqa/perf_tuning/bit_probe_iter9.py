"""Iter9 quick bit-identity probe — decisive per-domain go/no-go BEFORE the
full Gate 0.  The codegen audit (codegen_audit_iter9.py) flagged the per-head
AUTO threads=2 strided write as a mis-lowered vid-reduction (blockCount 32->64
with a stale contiguous vid offset); this probe confirms empirically with
raw-bit output comparison, and confirms the two expert-mode domains
(prefill_expert, g2stack) are correct.

Both arms run through the PUBLIC gqa() API in ONE process, caches cleared per
arm (iter5 finding 2).  ARM = variant factories monkeypatched + transpose_out
conditionally skipped (== the integration's capability-flag behavior).  Raw-bit
compare (view(int16), iter8 finding 3 — torch.equal is not bit-identity on NaN).

Run (repo root):
    unset PYTHONPATH; export ACL_OP_INIT_MODE=1
    python3 examples/gqa/perf_tuning/bit_probe_iter9.py
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
import gqa_w2_bsnd_iter9_variant as VAR  # noqa: E402

W2_TARGETS = ("_gqa_kernel_expert", "_gqa_kernel_prefill_expert", "_gqa_kernel_prefill_g2stack")

# (label, B,S,S_kv,N_q,N_kv,D, causal, dtype, expected_chosen_prefix, domain)
CASES = [
    ("r1_perhead_causal_t2", 1, 128, 128, 8, 2, 128, True, torch.float16, "per_head_bm", "per_head"),
    ("c5_perhead_d256_t2", 2, 128, 128, 32, 8, 256, True, torch.float16, "per_head_bm", "per_head"),
    ("perhead_pipe_t1", 1, 128, 256, 8, 2, 128, False, torch.float16, "per_head_bm", "per_head"),
    ("c19_prexpert", 2, 128, 128, 32, 32, 128, True, torch.bfloat16, "prefill_expert", "prefill_expert"),
    ("c9_g2stack_kv4", 4, 512, 512, 64, 8, 128, False, torch.bfloat16, "g2stack", "g2stack"),
]


def _clear():
    G._KERNEL_BUILD_CACHE.clear()
    G._COMPILED_CACHE.clear()


def _rawbit_eq(a, b):
    return bool(torch.equal(a.view(torch.int16).cpu(), b.view(torch.int16).cpu()))


class Arm:
    def __init__(self):
        self.invoked = []
        self._saved = {}
        self._saved_tout = None

    def install(self, aligned):
        self.invoked = []
        self._saved = {}
        if aligned:
            for name in W2_TARGETS:
                self._saved[name] = getattr(G, name)
                var_fn = getattr(VAR, name)

                def wrap(*a, _n=name, _f=var_fn, **kw):
                    self.invoked.append(_n)
                    return _f(*a, **kw)

                setattr(G, name, wrap)
        self._saved_tout = G.transpose_out
        orig_tout = self._saved_tout
        arm = self

        def tout_cond(out, s_real=None):
            if arm.invoked:
                assert s_real is None
                return out
            return orig_tout(out, s_real=s_real)

        G.transpose_out = tout_cond

    def remove(self):
        for name, orig in self._saved.items():
            setattr(G, name, orig)
        if self._saved_tout is not None:
            G.transpose_out = self._saved_tout


def run(label, B, S, S_kv, N_q, N_kv, D, causal, dtype, expect, domain):
    _clear()
    torch.manual_seed(42 + B + S)
    q = (torch.rand(B, S, N_q, D) * 2 - 1).to(dtype).npu()
    k = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()
    v = (torch.rand(B, S_kv, N_kv, D) * 2 - 1).to(dtype).npu()

    # BASE arm (no patch)
    _clear()
    out_b = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
    torch.npu.synchronize()
    chosen_b = [val[2] for val in G._KERNEL_BUILD_CACHE.values()]
    _clear()

    # W2 arm (patch)
    arm = Arm()
    aligned = (S % 64 == 0)
    arm.install(aligned)
    try:
        out_a = G.gqa(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
        torch.npu.synchronize()
        chosen_a = [val[2] for val in G._KERNEL_BUILD_CACHE.values()]
        inv = list(arm.invoked)
    finally:
        arm.remove()
        _clear()

    bit = _rawbit_eq(out_a, out_b)
    maxdiff = float((out_a.float() - out_b.float()).abs().max())
    ok_chosen = chosen_a and chosen_a[0].startswith(expect) and chosen_a == chosen_b
    print(
        f"[{'BITOK' if bit else 'BITFAIL'}] {label:24s} dom={domain:15s} chosen={chosen_a} "
        f"invoked={inv} bit={bit} maxabsdiff={maxdiff:.4e} chosen_ok={ok_chosen}",
        flush=True,
    )
    return bit


def main():
    results = {}
    for c in CASES:
        results[c[0]] = run(*c)
    print("\n[SUMMARY]")
    doms = {}
    for c in CASES:
        doms.setdefault(c[10], []).append((c[0], results[c[0]]))
    for dom, items in doms.items():
        allok = all(r for _, r in items)
        print(f"  {dom:16s}: {'ALL BIT-IDENTICAL' if allok else 'HAS MISMATCH'} — {items}")


if __name__ == "__main__":
    main()
