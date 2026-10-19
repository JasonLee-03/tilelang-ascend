"""Item-A adoption criterion 6: NO pre-existing test assertion was weakened.

The task authorises exactly two kinds of test change (section 2 items 2 and 3): update the 4 stale
split-expectation entries, and add case-6/case-11 split-path coverage. This script proves the edit
stayed inside that envelope, by AST rather than by eye:

  A. FUNCTION BODIES -- every function that existed before must have an IDENTICAL ast dump after.
     Any change to a comparison, tolerance, assert, or print that carries a verdict would show up
     here. New functions are listed separately (they may only ADD coverage).
  B. CASE TABLES -- every pre-existing entry must still be present with identical non-tag fields
     (name, B, V, K_MAX, BV, k, p, mu, need_min_p, scale, offset, force_S, seed_mode). Only the
     tag list may differ, and only by GROWING (a superset). A dropped case or a loosened parameter
     is a failure.
  C. CONSTANTS -- REV3_FROZEN_SHA256, REV3_UB_LIMIT, STAB_REPS, check_precision's tolerances and
     the COVERAGE_NA empty-set must be unchanged; COVERAGE_MANIFEST values may not DECREASE.
  D. Tag census: no coverage dimension may lose tags.

Usage: verify_no_weakening.py [out.json]
"""
import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R4 = os.path.abspath(os.path.join(HERE, "..", ".."))
BEFORE = os.path.join(R4, "history_version",
                      "test_r4_fused_sampling_impl_s3r3_itemA_attempt0.py")
AFTER = os.path.join(R4, "test_r4_fused_sampling.py")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "no_weakening_proof.json")

TAG_TABLES = ("L0_CASES", "REV3_L0_CASES", "REV3_L1_CASES", "REV3_BND_CASES", "L0B_CASES",
              "L1_CASES", "L1B_CASES", "STAB_CASES", "ADV_CASES")
CONST_TABLES = ("REV3_FROZEN_SHA256", "REV3_UB_LIMIT", "STAB_REPS", "COVERAGE_NA",
                "COVERAGE_MANIFEST", "_REV3_STATIC_SHAPES", "REV3_NEW_KERNELS")


def parse(path):
    src = open(path, encoding="utf-8").read()
    return ast.parse(src), src


def funcs(tree):
    return {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)}


def assigns(tree):
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    try:
                        out[t.id] = ast.literal_eval(n.value)
                    except Exception:
                        out[t.id] = ast.dump(n.value)
    return out


def entries(table):
    """{name_or_tuple_key: (non_tag_fields, tag_set)} for one case table."""
    out = {}
    if not isinstance(table, list):
        return out
    for e in table:
        if not isinstance(e, tuple):
            continue
        tags, nontag = [], []
        for f in e:
            if isinstance(f, list) and all(isinstance(x, str) and x.startswith("D-") for x in f):
                tags = f
            else:
                nontag.append(f)
        key = nontag[0] if nontag and isinstance(nontag[0], str) else repr(tuple(nontag))
        out[key] = (tuple(nontag), set(tags))
    return out


tb, sb = parse(BEFORE)
ta, sa = parse(AFTER)
fb, fa = funcs(tb), funcs(ta)
ab, aa = assigns(tb), assigns(ta)
ok = True

# ---------------------------------------------------------------- A. function bodies
print("=" * 110)
print("A. FUNCTION BODIES -- every pre-existing function must be ast-IDENTICAL")
print("=" * 110)
changed = [n for n in fb if n in fa and fb[n] != fa[n]]
removed = [n for n in fb if n not in fa]
added = [n for n in fa if n not in fb]
print(f"  functions before={len(fb)} after={len(fa)}")
print(f"  CHANGED bodies : {changed}")
print(f"  REMOVED        : {removed}")
print(f"  ADDED          : {added}")
# a changed body is only acceptable if it is a docstring-only change
doc_only = []
for n in changed:
    nb = next(x for x in tb.body if isinstance(x, ast.FunctionDef) and x.name == n)
    na = next(x for x in ta.body if isinstance(x, ast.FunctionDef) and x.name == n)
    strip = lambda f: ast.dump(ast.Module(body=[ast.FunctionDef(  # noqa: E731
        name=f.name, args=f.args, body=f.body[1:] if ast.get_docstring(f) else f.body,
        decorator_list=[], returns=f.returns, type_comment=None)], type_ignores=[]))
    if strip(nb) == strip(na):
        doc_only.append(n)
print(f"  of the CHANGED, docstring-only (executable body identical): {doc_only}")
real_changed = [n for n in changed if n not in doc_only]
print(f"  => functions with a REAL executable change = {real_changed}")
ok &= (not removed) and (not real_changed)

# ---------------------------------------------------------------- B. case tables
print(f"\n{'=' * 110}")
print("B. CASE TABLES -- every pre-existing entry must survive with identical non-tag fields")
print("=" * 110)
table_report = {}
for t in TAG_TABLES:
    eb, ea = entries(ab.get(t)), entries(aa.get(t))
    dropped = [k for k in eb if k not in ea]
    field_changed, tag_grew, tag_shrank = [], [], []
    for k in eb:
        if k not in ea:
            continue
        if eb[k][0] != ea[k][0]:
            field_changed.append((k, eb[k][0], ea[k][0]))
        if not eb[k][1] <= ea[k][1]:
            tag_shrank.append((k, sorted(eb[k][1] - ea[k][1])))
        if eb[k][1] != ea[k][1]:
            tag_grew.append((k, sorted(ea[k][1] - eb[k][1])))
    new_entries = [k for k in ea if k not in eb]
    print(f"  {t:16s} before={len(eb):>3} after={len(ea):>3} | dropped={len(dropped)} "
          f"non-tag-fields-changed={len(field_changed)} tags-shrunk={len(tag_shrank)} "
          f"tags-grew={len(tag_grew)} new={len(new_entries)}")
    for k, added_tags in tag_grew:
        print(f"        tags GREW  {k:34s} +{added_tags}")
    for k in new_entries:
        print(f"        NEW ENTRY  {k}")
    for k, b, a in field_changed:
        print(f"        *** non-tag FIELDS CHANGED {k}: {b} -> {a}")
    for k, lost in tag_shrank:
        print(f"        *** TAGS LOST {k}: {lost}")
    if dropped:
        print(f"        *** DROPPED ENTRIES: {dropped}")
    ok &= (not dropped) and (not field_changed) and (not tag_shrank)
    table_report[t] = dict(before=len(eb), after=len(ea), dropped=dropped,
                           non_tag_fields_changed=[list(map(str, x)) for x in field_changed],
                           tags_shrunk=[list(map(str, x)) for x in tag_shrank],
                           tags_grew=[[k, v] for k, v in tag_grew], new_entries=new_entries)

# ---------------------------------------------------------------- C. constants
print(f"\n{'=' * 110}")
print("C. CONSTANTS / TOLERANCES -- must not loosen")
print("=" * 110)
for k in ("REV3_FROZEN_SHA256", "REV3_UB_LIMIT", "STAB_REPS", "COVERAGE_NA", "REV3_NEW_KERNELS",
          "_REV3_STATIC_SHAPES"):
    same = ab.get(k) == aa.get(k)
    if k == "_REV3_STATIC_SHAPES":
        # may only GROW (more compile-time coverage); every old shape must still be there
        old = set(map(tuple, ab.get(k) or []))
        new = set(map(tuple, aa.get(k) or []))
        grew = old <= new
        print(f"  {k:22s} before={len(old)} after={len(new)}  all old shapes retained={grew}  "
              f"added={sorted(new - old)}")
        ok &= grew
        continue
    print(f"  {k:22s} unchanged = {same}" + ("" if same else f"  *** {ab.get(k)} -> {aa.get(k)} ***"))
    ok &= same

mb, ma = ab.get("COVERAGE_MANIFEST") or {}, aa.get("COVERAGE_NA") is not None and (aa.get("COVERAGE_MANIFEST") or {})
dec = {k: (mb[k], ma.get(k)) for k in mb if isinstance(mb[k], int)
       and isinstance(ma.get(k), int) and ma[k] < mb[k]}
print(f"  COVERAGE_MANIFEST values that DECREASED = {dec}  (none may)")
print(f"  COVERAGE_MANIFEST keys removed = {sorted(set(mb) - set(ma))}")
ok &= (not dec) and (set(mb) <= set(ma))

# check_precision tolerances live in the function body -> already covered by A.
print(f"\n  check_precision body identical = {fb.get('check_precision') == fa.get('check_precision')}")
ok &= fb.get("check_precision") == fa.get("check_precision")

# ---------------------------------------------------------------- D. tag census
print(f"\n{'=' * 110}\nD. TAG CENSUS -- no dimension may lose tags\n{'=' * 110}")


def tagcensus(tree):
    c = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.List):
            for e in n.elts:
                if isinstance(e, ast.Constant) and isinstance(e.value, str) \
                        and e.value.startswith("D-"):
                    c[e.value] = c.get(e.value, 0) + 1
    return c


cb, ca = tagcensus(tb), tagcensus(ta)
lost = {k: (cb[k], ca.get(k, 0)) for k in cb if ca.get(k, 0) < cb[k]}
print(f"  dimensions before={len(cb)} after={len(ca)}")
print(f"  dimensions whose tag count DECREASED = {lost}  (none may)")
grew = {k: (cb[k], ca[k]) for k in cb if ca.get(k, 0) > cb[k]}
print(f"  dimensions whose tag count grew = {grew}")
print(f"  new dimensions = {sorted(set(ca) - set(cb))}")
ok &= not lost

print(f"\n{'=' * 110}")
print(f"[CRITERION 6] {'PASS -- no pre-existing assertion, case, tolerance or tag was weakened; '
                'changes are additions plus tag growth only' if ok else '*** FAIL ***'}")
print("=" * 110)

json.dump(dict(_meta=dict(before=BEFORE, after=AFTER),
               functions=dict(before=len(fb), after=len(fa), changed=changed,
                              docstring_only=doc_only, real_executable_change=real_changed,
                              removed=removed, added=added),
               tables=table_report,
               constants=dict(decreased_manifest=dec, lost_tag_dims=lost),
               criterion6_pass=bool(ok)), open(OUT, "w"), indent=1)
print(f"wrote {OUT}")
sys.exit(0 if ok else 1)
