"""Verify the 3 delivered kernels are BYTE-IDENTICAL between two file versions.

Uses ast to get each function's exact source segment (decorators included), so the
comparison is not fooled by line-number drift caused by the 4 newly inserted kernels.

Usage: verify_frozen.py <old.py> <new.py>
"""
import ast
import hashlib
import sys

FROZEN = ("r4_topk_stats", "r4_mask_sample", "r4_gumbel_sample")


def segments(path):
    src = open(path, encoding="utf-8").read()
    lines = src.split("\n")
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in FROZEN:
            start = node.lineno - 1
            if node.decorator_list:
                start = min(start, min(d.lineno for d in node.decorator_list) - 1)
                # include the '@' line itself (decorator lineno points at the expression)
                while start > 0 and not lines[start].lstrip().startswith("@"):
                    start -= 1
            end = node.end_lineno  # exclusive index == end_lineno (1-based inclusive)
            txt = "\n".join(lines[start:end])
            out[node.name] = (start + 1, end, txt, hashlib.sha256(txt.encode()).hexdigest())
    return out


def main():
    old, new = sys.argv[1], sys.argv[2]
    so, sn = segments(old), segments(new)
    ok_all = True
    print(f"frozen-kernel byte-identity: {old}\n                          vs {new}")
    for name in FROZEN:
        if name not in so or name not in sn:
            print(f"  {name:22s} MISSING (old={name in so} new={name in sn})")
            ok_all = False
            continue
        ao, eo, to, ho = so[name]
        an, en, tn, hn = sn[name]
        same = ho == hn
        ok_all &= same
        print(f"  {name:22s} old L{ao}-{eo} ({len(to)}B) sha256={ho[:24]}…")
        print(f"  {'':22s} new L{an}-{en} ({len(tn)}B) sha256={hn[:24]}…  "
              f"{'BYTE-IDENTICAL' if same else '*** CHANGED ***'}")
        if not same:
            import difflib
            for ln in list(difflib.unified_diff(to.split("\n"), tn.split("\n"),
                                                "old", "new", lineterm=""))[:40]:
                print("      " + ln)
    print(f"RESULT: {'ALL 3 FROZEN KERNELS BYTE-IDENTICAL' if ok_all else 'FROZEN KERNEL(S) MODIFIED'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
