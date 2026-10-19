"""Item-A adoption criterion 5: the 3 frozen kernels are byte-identical, before vs after.

Extracts each frozen kernel's source segment (decorator line .. `return main`) with the SAME ast
logic the delivered static gate uses (`_rev3_frozen_segments` in test_r4_fused_sampling.py), from
BOTH the pre-change backup and the post-change delivered file, and compares:

  * per-kernel segment sha256  (against the rev2 baseline constants the static gate asserts)
  * segment line ranges        (they must not shift, or the segment would not be the same code)
  * whole-file sha256 before/after, and the diff hunk census proving the ONLY kernel change is the
    _SPLIT_A_LAUNCH_US constant + its comment block.

Also re-checks the two files that must never move: proto.yaml (contract) and the submission copy
`cann_bench_submission/cann_bench/_r4_kernels.py`, which must stay cmp-identical to the delivered
kernel because build.sh vendors it verbatim.

Usage: verify_frozen_itemA.py [out.json]
"""
import ast
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R4 = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "frozen_kernel_proof.json")

BEFORE = os.path.join(R4, "history_version",
                      "r4_fused_sampling_impl_s3r3_itemA_attempt0.py")
AFTER = os.path.join(R4, "r4_fused_sampling.py")
SUBMISSION = os.path.join(R4, "cann_bench_submission", "cann_bench", "_r4_kernels.py")
PROTO = os.path.join(R4, "proto.yaml")
TEST_BEFORE = os.path.join(R4, "history_version",
                           "test_r4_fused_sampling_impl_s3r3_itemA_attempt0.py")
TEST_AFTER = os.path.join(R4, "test_r4_fused_sampling.py")

# the rev2 delivery values asserted by the static gate's assertion 2 (must never change)
FROZEN = {
    "r4_topk_stats": "d39090631598f07d1007917b310216b2f650a4ab4c3e7dd6b60b3e63a3a94831",
    "r4_mask_sample": "f787d1ad05a9fc26913b98de34f3a48ee67f730530ad6add8761082ef187a3d1",
    "r4_gumbel_sample": "ca90de8d0f6f4f69f366f380009f43915aa3c9f5b90430d683be239916683889",
}
EXPECT_BEFORE = "a764e117ce150d8677096c2e00bde3db3ebc3ec16aba63e4c613d8dffdfbd117"
EXPECT_TEST_BEFORE = "cea988cbd117f23bb0bf50dbfe0c043dcb46cf4c0489d656b8073cc115265cb7"
EXPECT_PROTO = "41ab797002568e89fcc9a9eaa58be4f1cdfce20b6406abd0fdbf2ad11f7f8949"


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def segments(path):
    """{name: (start_line, end_line, sha256, n_lines)} using the static gate's own logic."""
    src = open(path, encoding="utf-8").read()
    lines = src.split("\n")
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in FROZEN:
            start = node.lineno - 1
            if node.decorator_list:
                start = min(start, min(d.lineno for d in node.decorator_list) - 1)
                while start > 0 and not lines[start].lstrip().startswith("@"):
                    start -= 1
            txt = "\n".join(lines[start:node.end_lineno])
            out[node.name] = dict(start=start + 1, end=node.end_lineno,
                                  n_lines=node.end_lineno - start,
                                  sha256=hashlib.sha256(txt.encode()).hexdigest())
    return out


def kernel_names(path):
    """Every @tilelang.jit-decorated function name, in file order (must stay 7)."""
    src = open(path, encoding="utf-8").read()
    names = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef):
            for d in node.decorator_list:
                t = d.func if isinstance(d, ast.Call) else d
                if isinstance(t, ast.Attribute) and t.attr == "jit":
                    names.append(node.name)
    return names


print("=" * 108)
print("ITEM-A ADOPTION CRITERION 5 -- the 3 frozen kernels, byte-identical before vs after")
print("=" * 108)
sb, sa = sha(BEFORE), sha(AFTER)
print(f"  whole-file sha256 BEFORE (backup) = {sb}")
print(f"  whole-file sha256 AFTER  (delivered) = {sa}")
print(f"  BEFORE == the Orchestrator-stated a764e117 = {sb == EXPECT_BEFORE}")

seg_b, seg_a = segments(BEFORE), segments(AFTER)
ok = True
print(f"\n  {'kernel':20s} {'lines BEFORE':>14} {'lines AFTER':>13} "
      f"{'sha BEFORE':>18} {'sha AFTER':>18}  verdict")
for kn, want in FROZEN.items():
    b, a = seg_b.get(kn), seg_a.get(kn)
    same = (b is not None and a is not None and b["sha256"] == a["sha256"]
            and b["start"] == a["start"] and b["end"] == a["end"]
            and a["sha256"] == want)
    ok &= same
    print(f"  {kn:20s} L{b['start']}-{b['end']:<9} L{a['start']}-{a['end']:<8} "
          f"{b['sha256'][:16]}.. {a['sha256'][:16]}..  "
          f"{'IDENTICAL + == rev2 baseline' if same else '*** DIFFERS ***'}")
print(f"\n  frozen segments byte-identical AND equal to the rev2 baseline constants = {ok}")

nb, na = kernel_names(BEFORE), kernel_names(AFTER)
print(f"  @tilelang.jit kernels BEFORE ({len(nb)}) = {nb}")
print(f"  @tilelang.jit kernels AFTER  ({len(na)}) = {na}")
same_kernels = nb == na
print(f"  kernel set + order unchanged = {same_kernels}")
ok &= same_kernels

# ---- diff hunk census: prove the ONLY change is the constant + its comment ----
d = subprocess.run(["diff", "-u", BEFORE, AFTER], capture_output=True, text=True)
hunks = [l for l in d.stdout.splitlines() if l.startswith("@@")]
added = [l[1:] for l in d.stdout.splitlines() if l.startswith("+") and not l.startswith("+++")]
removed = [l[1:] for l in d.stdout.splitlines() if l.startswith("-") and not l.startswith("---")]
code_added = [l for l in added if l.strip() and not l.strip().startswith("#")]
code_removed = [l for l in removed if l.strip() and not l.strip().startswith("#")]
print(f"\n  KERNEL DIFF CENSUS: hunks={len(hunks)}  +{len(added)} / -{len(removed)} lines")
print(f"    non-comment lines added   = {len(code_added)} -> {code_added}")
print(f"    non-comment lines removed = {len(code_removed)} -> {code_removed}")
only_const = (len(hunks) == 1 and code_removed == ["_SPLIT_A_LAUNCH_US = 8.0  # one kernel launch, "
                                                    "CONSERVATIVE default (U2/V-S3: not yet "
                                                    "measured)"]
              and code_added == ["_SPLIT_A_LAUNCH_US = 1.540"])
print(f"    the ONLY executable change is the _SPLIT_A_LAUNCH_US assignment = {only_const}")
ok &= only_const

# ---- files that must never move ----
print(f"\n  proto.yaml sha256 = {sha(PROTO)}  == contract {EXPECT_PROTO[:16]}.. : "
      f"{sha(PROTO) == EXPECT_PROTO}")
ok &= sha(PROTO) == EXPECT_PROTO
sub = sha(SUBMISSION)
cmp_rc = subprocess.run(["cmp", "-s", AFTER, SUBMISSION]).returncode
print(f"  submission copy _r4_kernels.py sha256 = {sub}")
print(f"  cmp delivered-kernel vs submission-copy -> rc={cmp_rc} "
      f"({'IDENTICAL' if cmp_rc == 0 else '*** DIFFERS ***'})  and == delivered sha: {sub == sa}")
ok &= (cmp_rc == 0 and sub == sa)

tb, ta = sha(TEST_BEFORE), sha(TEST_AFTER)
print(f"\n  test file sha256 BEFORE = {tb}  == cea988cb.. : {tb == EXPECT_TEST_BEFORE}")
print(f"  test file sha256 AFTER  = {ta}")
dt = subprocess.run(["diff", "-u", TEST_BEFORE, TEST_AFTER], capture_output=True, text=True)
t_add = [l[1:] for l in dt.stdout.splitlines() if l.startswith("+") and not l.startswith("+++")]
t_rem = [l[1:] for l in dt.stdout.splitlines() if l.startswith("-") and not l.startswith("---")]
t_rem_code = [l for l in t_rem if l.strip() and not l.strip().startswith("#")]
print(f"  TEST DIFF CENSUS: +{len(t_add)} / -{len(t_rem)} lines; removed non-comment lines = "
      f"{len(t_rem_code)}")
for l in t_rem_code:
    print(f"      - {l.strip()[:150]}")
ok &= tb == EXPECT_TEST_BEFORE

print(f"\n{'=' * 108}")
print(f"[CRITERION 5] {'PASS -- frozen kernels byte-identical, only the constant changed' if ok else '*** FAIL ***'}")
print("=" * 108)

json.dump(dict(_meta=dict(before_file=BEFORE, after_file=AFTER,
                          sha_before=sb, sha_after=sa,
                          sha_before_matches_orchestrator=(sb == EXPECT_BEFORE),
                          proto_sha256=sha(PROTO), proto_matches_contract=(sha(PROTO) == EXPECT_PROTO),
                          submission_sha256=sub, submission_cmp_identical=(cmp_rc == 0),
                          test_sha_before=tb, test_sha_after=ta,
                          test_sha_before_matches=(tb == EXPECT_TEST_BEFORE)),
               frozen_segments_before=seg_b, frozen_segments_after=seg_a,
               frozen_identical_and_equal_to_rev2_baseline=ok,
               jit_kernels_before=nb, jit_kernels_after=na,
               kernel_diff=dict(hunks=len(hunks), added=len(added), removed=len(removed),
                                code_added=code_added, code_removed=code_removed,
                                only_constant_changed=only_const),
               test_diff=dict(added=len(t_add), removed=len(t_rem),
                              removed_non_comment=[l.strip() for l in t_rem_code]),
               criterion5_pass=bool(ok)), open(OUT, "w"), indent=1)
print(f"wrote {OUT}")
sys.exit(0 if ok else 1)
