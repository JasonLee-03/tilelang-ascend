#!/usr/bin/python3
# coding=utf-8
"""Static vector-instruction census of the K1 main loop (part (b), corroborating evidence).

Extracts the `for (int32_t b = 0; b < N; ++b) { ... }` body from an emitted Ascend C source
and buckets every instruction into online-softmax (unavoidable) vs topk+merge (prunable).
`tl::ascend::TopK<...>(dst, src, tmp, K, repeatTimes, actualCount)` is EXPANDED using the
helper source (src/tl_templates/ascend/common.h:1685-1898):

    Sort()      : ArithProgression(alignedCount)
                  + Sort32(repeatTimes)
                  + merge tree: numSegs -> ceil(numSegs/4) until 1, one MrgSort per group
                  + final DataCopy(alignedCount*2) when the last round wrote to tmp
    TopK()      : Sort() + PipeBarrier + DataCopy(alignedTopk)   [alignedTopk = ceil(2K/8)*8]

with alignedCount = repeatTimes*32 and padCount = alignedCount - actualCount (must be 0 on
the full-block path -- precision red line 2).

Usage: probe_insn_census.py <cc_dir>
"""
import json
import os
import re
import sys


def expand_topk(K: int, repeatTimes: int, actualCount: int):
    """Return the hardware-instruction multiset a single TopK call expands to."""
    aligned = repeatTimes * 32
    pad = aligned - actualCount
    mrg, numSegs, stages = 0, repeatTimes, []
    while numSegs > 1:
        groups = (numSegs + 3) // 4
        stages.append((numSegs, groups))
        mrg += groups
        numSegs = groups
    return {
        "alignedCount": aligned,
        "padCount": pad,                       # must be 0 (red line 2)
        "ArithProgression": 1,
        "Duplicate_pad": (0 if pad == 0 else 1),
        "Sort32": 1,
        "Sort32_repeats": repeatTimes,
        "MrgSort": mrg,
        "merge_stages": stages,
        "DataCopy": 2,                         # ping-pong flush + top-K extract
        "PipeBarrier": 1,
    }


def main_loop_body(src: str):
    m = re.search(r"for \(int32_t b = 0; b < (\d+); \+\+b\) \{", src)
    if not m:
        return None, None
    n = int(m.group(1))
    i = src.index("{", m.end() - 1)
    depth, j = 0, i
    while j < len(src):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    return n, src[i + 1 : j]


def census(body: str):
    """Bucket instructions. Returns (softmax_counts, topk_counts, raw_lines)."""
    sm, tk, raw = {}, {}, []
    for ln in body.splitlines():
        s = ln.strip()
        if not s or s.startswith("//"):
            continue
        raw.append(s)
        # --- topk+merge bucket --------------------------------------------
        mt = re.search(r"tl::ascend::TopK<float>\(.*?, (\d+), (\d+), (\d+)\)", s)
        if mt:
            K, rt, ac = (int(x) for x in mt.groups())
            e = expand_topk(K, rt, ac)
            for k in ("ArithProgression", "Sort32", "MrgSort", "DataCopy",
                      "PipeBarrier", "Duplicate_pad"):
                tk[f"TopK::{k}"] = tk.get(f"TopK::{k}", 0) + e[k]
            tk.setdefault("TopK::calls", []).append(
                {"K": K, "repeatTimes": rt, "actualCount": ac,
                 "alignedCount": e["alignedCount"], "padCount": e["padCount"],
                 "Sort32_repeats": e["Sort32_repeats"], "MrgSort": e["MrgSort"],
                 "merge_stages": e["merge_stages"]})
            tk["TopK_total_Sort32_repeats"] = tk.get("TopK_total_Sort32_repeats", 0) + rt
            continue
        if "GatherMask" in s:
            tk["GatherMask"] = tk.get("GatherMask", 0) + 1
            continue
        if "copy_ub_to_ub" in s:
            tk["copy_ub_to_ub"] = tk.get("copy_ub_to_ub", 0) + 1
            continue
        # --- online-softmax bucket ----------------------------------------
        if "copy_gm_to_ub" in s:
            sm["copy_gm_to_ub"] = sm.get("copy_gm_to_ub", 0) + 1
            continue
        mr = re.search(r"tl::ascend::reduce_(max|sum)<float,\s*\d+,\s*(\d+),", s)
        if mr:
            sm[f"reduce_{mr.group(1)}"] = sm.get(f"reduce_{mr.group(1)}", 0) + 1
            sm[f"reduce_{mr.group(1)}_elems"] = int(mr.group(2))
            continue
        mb = re.search(r"tl::ascend::Broadcast<float, \d+, (\d+),", s)
        if mb:
            sm["Broadcast"] = sm.get("Broadcast", 0) + 1
            continue
        ma = re.match(r"AscendC::(\w+)\(([^;]*)\);", s)
        if ma:
            op, argstr = ma.group(1), ma.group(2)
            if op in ("PipeBarrier", "SetFlag", "WaitFlag"):
                sm["sync"] = sm.get("sync", 0) + 1
                continue
            tail = [a.strip() for a in argstr.split(",")]
            cnt = None
            for a in reversed(tail):
                if re.fullmatch(r"\d+", a):
                    cnt = int(a)
                    break
            key = f"{op}[{cnt}]" if cnt is not None else op
            if cnt is not None and cnt > 1:
                sm["vec_elemwise_fullwidth"] = sm.get("vec_elemwise_fullwidth", 0) + 1
            else:
                sm["scalar_op"] = sm.get("scalar_op", 0) + 1
            sm[key] = sm.get(key, 0) + 1
            continue
        if s.startswith("if (") or s in ("}", "{"):
            sm["branch"] = sm.get("branch", 0) + 1
            continue
        sm["other:" + s[:60]] = sm.get("other:" + s[:60], 0) + 1
    return sm, tk, raw


def main():
    cc_dir = sys.argv[1]
    out = {}
    for fn in sorted(os.listdir(cc_dir)):
        if not fn.endswith(".cc"):
            continue
        src = open(os.path.join(cc_dir, fn)).read()
        n, body = main_loop_body(src)
        if body is None:
            out[fn] = {"error": "no main block loop (n_blk_full == 0)"}
            continue
        sm, tk, raw = census(body)
        out[fn] = {"n_blk_full": n, "loop_insn_lines": len(raw),
                   "softmax": sm, "topk_merge": tk}
        print(f"\n=== {fn}  (n_blk_full={n}, {len(raw)} instruction lines in the loop) ===")
        print("  online-softmax (UNAVOIDABLE):")
        for k in sorted(sm):
            if not k.startswith("other:") and not re.search(r"\[\d+\]$", k):
                print(f"    {k:<34} {sm[k]}")
        print("    -- per-op detail --")
        for k in sorted(sm):
            if re.search(r"\[\d+\]$", k):
                print(f"    {k:<34} {sm[k]}")
        print("  topk+merge (PRUNABLE):")
        for k in sorted(tk):
            if k != "TopK::calls":
                print(f"    {k:<34} {tk[k]}")
        for c in tk.get("TopK::calls", []):
            print(f"      TopK(K={c['K']}, repeats={c['repeatTimes']}, actual={c['actualCount']})"
                  f" -> aligned={c['alignedCount']} padCount={c['padCount']}"
                  f" Sort32reps={c['Sort32_repeats']} MrgSort={c['MrgSort']}"
                  f" stages={c['merge_stages']}")
    p = os.path.join(cc_dir, "insn_census.json")
    json.dump(out, open(p, "w"), indent=2, default=str)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
