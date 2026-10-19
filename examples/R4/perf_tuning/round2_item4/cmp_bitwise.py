"""Bitwise comparison of two capture manifests + their raw output bytes.

Usage: cmp_bitwise.py <before_dir> <after_dir>

Reports, per shape and per output tensor, whether the raw bytes are identical
(by sha256 AND by direct byte comparison). Exits non-zero if ANY tensor differs.
"""
import hashlib
import json
import os
import sys

bd, ad = sys.argv[1], sys.argv[2]
mb = json.load(open(os.path.join(bd, "manifest.json")))
ma = json.load(open(os.path.join(ad, "manifest.json")))

assert set(mb) == set(ma), f"shape-set mismatch: {set(mb) ^ set(ma)}"

total = same = diff = 0
diffs = []
print(f"{'shape':22s} {'n_blk_full':>10} {'TAIL':>6} {'tensor':16s} {'nbytes':>10}  result")
for tag in mb:
    for name, eb in mb[tag]["outputs"].items():
        ea = ma[tag]["outputs"][name]
        pb, pa = os.path.join(bd, f"{tag}__{name.replace('.', '__')}.bin"), \
            os.path.join(ad, f"{tag}__{name.replace('.', '__')}.bin")
        rb, ra = open(pb, "rb").read(), open(pa, "rb").read()
        hb, ha = hashlib.sha256(rb).hexdigest(), hashlib.sha256(ra).hexdigest()
        ok = (hb == ha) and (rb == ra) and (eb["shape"] == ea["shape"]) and (eb["dtype"] == ea["dtype"])
        total += 1
        same += ok
        diff += (not ok)
        if not ok:
            diffs.append((tag, name, hb[:16], ha[:16]))
        if not ok or name in ("mask.tau_out", "mask.w_out", "gumbel.tok_out"):
            print(f"{tag:22s} {mb[tag]['n_blk_full']:>10} {mb[tag]['TAIL']:>6} "
                  f"{name:16s} {len(rb):>10}  {'IDENTICAL' if ok else '*** DIFFERS ***'}")

print(f"\n==== BITWISE SUMMARY ====")
print(f"tensors compared : {total}   identical: {same}   differing: {diff}")
if diffs:
    print("DIFFERING TENSORS:")
    for d in diffs:
        print("   ", d)
    sys.exit(1)
print("VERDICT: ALL OUTPUTS BITWISE IDENTICAL (sha256 + raw-byte compare + shape/dtype)")
sys.exit(0)
