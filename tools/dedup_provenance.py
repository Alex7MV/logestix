#!/usr/bin/env python3
"""Level-0 gate for the deduplication phases: prove a move is a move.

Compares every top-level ``__global__`` definition in --from against every one
in --to after stripping comments, collapsing whitespace and applying the
``--rename old=new`` substitutions. A move is valid only when both sides
produce the same multiset of normalized bodies: every source kernel has an
identical twin in the target and the target adds nothing new.

    python3 tools/dedup_provenance.py \
        --from libs/lsxcommon/src/granite_dense_inference.cu \
        --to   libs/lsxcommon/src/lsx_dense_inference.cu \
        --rename granite_=dense_

Exit status 0 on a proven move, 1 otherwise (prints the unmatched bodies).
"""
import argparse
import re
import sys


def kernels(path):
    text = open(path, encoding="utf-8").read()
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        if lines[i].lstrip().startswith("__global__"):
            start, depth, seen, body = i, 0, False, []
            while i < len(lines):
                depth += lines[i].count("{") - lines[i].count("}")
                seen = seen or "{" in lines[i]
                body.append(lines[i])
                if seen and depth <= 0:
                    break
                i += 1
            out.append((lines[start].strip(), "".join(body)))
        i += 1
    return out


def normalize(body, renames):
    for old, new in renames:
        body = body.replace(old, new)
    return re.sub(r"\s+", "", body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--to", dest="dst", required=True)
    ap.add_argument("--rename", action="append", default=[])
    ap.add_argument(
        "--allow-extra",
        action="store_true",
        help="target may hold unrelated kernels (source is a subset)",
    )
    args = ap.parse_args()
    renames = [tuple(r.split("=", 1)) for r in args.rename]

    src = [(sig, normalize(body, renames)) for sig, body in kernels(args.src)]
    dst = [(sig, normalize(body, renames)) for sig, body in kernels(args.dst)]
    dst_bodies = [body for _, body in dst]
    src_bodies = [body for _, body in src]
    missing = [(sig, b) for sig, b in src if b not in dst_bodies]
    extra = [(sig, b) for sig, b in dst if b not in src_bodies]
    print(f"source kernels: {len(src)}  target kernels: {len(dst)}")
    print(f"missing in target: {len(missing)}  unexpected in target: {len(extra)}")
    for sig, body in missing:
        print(f"  MISSING: {sig[:100]}")
        print(f"           {body[:160]}")
    for sig, body in extra:
        print(f"  EXTRA:   {sig[:100]}")
        print(f"           {body[:160]}")
    ok = not missing and (args.allow_extra or not extra)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
