"""Extract the FROZEN tail-block region of r4_fused_sampling.py and hash it.

The task freezes "L219-249" (the `if TAIL > 0:` branch, comments included).
Line numbers shift when lines are deleted above, so the region is located by
CONTENT MARKERS, not by fixed numbers:

  start = the line matching  '# ======== tail block:'
  end   = the line just before '# ======== epilogue:'

That marker-bounded span is a SUPERSET of L219-249 (it also covers the blank
separator line), so proving it byte-identical proves L219-249 byte-identical.

Also emits the strict 31-line span that starts at the tail marker, for an
explicit "L219-249" hash.
"""
import hashlib
import sys

path = sys.argv[1]
out_prefix = sys.argv[2]
lines = open(path).read().split("\n")

start = next(i for i, ln in enumerate(lines) if "# ======== tail block:" in ln)
end = next(i for i, ln in enumerate(lines) if "# ======== epilogue:" in ln)

marker_region = "\n".join(lines[start:end])          # tail marker .. epilogue marker-1
strict_region = "\n".join(lines[start:start + 31])   # exactly 31 lines from tail marker

h_marker = hashlib.sha256(marker_region.encode()).hexdigest()
h_strict = hashlib.sha256(strict_region.encode()).hexdigest()

open(f"{out_prefix}_tail_marker_region.txt", "w").write(marker_region)
open(f"{out_prefix}_tail_strict31.txt", "w").write(strict_region)

print(f"file                 : {path}")
print(f"tail marker at line  : {start + 1}  (1-based)")
print(f"epilogue marker line : {end + 1}  (1-based)")
print(f"marker-bounded span  : L{start + 1}..L{end}  ({end - start} lines)")
print(f"  sha256             : {h_marker}")
print(f"strict 31-line span  : L{start + 1}..L{start + 31}")
print(f"  sha256             : {h_strict}")
print(f"  last line          : {lines[start + 30]!r}")
