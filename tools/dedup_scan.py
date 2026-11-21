#!/usr/bin/env python3
"""dedup_scan.py — the duplication-residue instrument (dedup phase 3+).

Read-only scan of libs/, apps/ and tests/: extracts top-level __global__
kernel bodies from .cu/.cuh and top-level host function bodies from .cpp,
normalizes them (comments stripped, whitespace collapsed) and groups
byte-identical bodies. This is a MEASUREMENT, not a gate: exit status is
always 0 on a successful scan; the numbers are compared before/after a
consolidation phase with the SAME instrument.

    python3 tools/dedup_scan.py --root . --json report.json [--min-lines 6]

Companion gate: tools/dedup_provenance.py proves a move is a move.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def strip_comments(text):
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return text


def kernel_defs(text):
    """Top-level __global__ definitions (signature may span lines)."""
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        s = lines[i].lstrip()
        if s.startswith("__global__") or s.startswith("static __global__"):
            start, depth, seen, body = i, 0, False, []
            while i < len(lines):
                depth += lines[i].count("{") - lines[i].count("}")
                seen = seen or "{" in lines[i]
                body.append(lines[i])
                if seen and depth <= 0:
                    break
                i += 1
            sig = lines[start].strip()
            m = re.search(r"([A-Za-z0-9_]+)\s*\(", "\n".join(body[:3]))
            name = m.group(1) if m else sig[:60]
            out.append((name, start + 1, "\n".join(body)))
        i += 1
    return out


def norm(body):
    return re.sub(r"\s+", "", body)


def norm_kernel(body):
    """Normalize a kernel body for duplicate grouping: the kernel NAME is
    stripped (families rename the same body), comments are already gone,
    whitespace collapses. Everything else — signature types, launch bounds,
    arithmetic — stays: a differing parameter list is a differing kernel."""
    body = re.sub(r"(__global__\s+(?:void\s+)?)[A-Za-z0-9_]+", r"\1", body,
                  count=1)
    return norm(body)


def norm_host(body):
    """Normalize a host function body: strip the function NAME (the first
    identifier followed by '('), keep the rest. Same rule as kernels —
    renamed copies of one body must group."""
    body = re.sub(r"\b[A-Za-z_]\w*(?=\s*\()", "", body, count=1)
    return norm(body)


def nlines(body):
    return len([l for l in body.split("\n") if l.strip()])


def scan(root, min_lines):
    files = []
    for d in ("libs", "apps", "tests"):
        base = root / d
        if base.is_dir():
            files += sorted(base.rglob("*.cu")) + sorted(base.rglob("*.cuh"))

    groups = defaultdict(list)
    total_kernels = 0
    for f in files:
        rel = str(f.relative_to(root))
        text = strip_comments(f.read_text(encoding="utf-8", errors="replace"))
        defs = kernel_defs(text)
        total_kernels += len(defs)
        for name, line, body in defs:
            nl = nlines(body)
            if nl < min_lines:
                continue
            groups[norm_kernel(body)].append(
                {"file": rel, "name": name, "line": line, "lines": nl})

    dups = []
    for members in groups.values():
        if len(members) < 2:
            continue
        where = set(m["file"] for m in members)
        dups.append({"lines": members[0]["lines"], "copies": len(members),
                     "distinct_files": len(where), "members": members})
    dups.sort(key=lambda g: -g["lines"] * (g["copies"] - 1))

    # Host-side duplication: top-level function bodies in .cpp (heuristic).
    host_groups = defaultdict(list)
    cpp_files = []
    for d in ("libs", "apps"):
        base = root / d
        if base.is_dir():
            cpp_files += sorted(base.rglob("*.cpp"))
    for f in cpp_files:
        rel = str(f.relative_to(root))
        text = strip_comments(f.read_text(encoding="utf-8", errors="replace"))
        for m in re.finditer(
                r"^[A-Za-z_][\w:<>,\s\*&]*?\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*"
                r"(?:const\s*)?\{", text, flags=re.M):
            start = m.end() - 1
            depth, i = 0, start
            while i < len(text):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            body = text[start:i + 1]
            nl = nlines(body)
            if nl < 12:
                continue
            line = text.count("\n", 0, m.start()) + 1
            host_groups[norm_host(body)].append(
                {"file": rel, "name": m.group(1), "line": line, "lines": nl})

    host_dups = []
    for members in host_groups.values():
        where = set(m["file"] for m in members)
        if len(members) >= 2 and len(where) >= 2:
            host_dups.append({"lines": members[0]["lines"],
                              "copies": len(members), "members": members})
    host_dups.sort(key=lambda g: -g["lines"] * (g["copies"] - 1))

    return {
        "scanned_cu_files": len(files),
        "total_kernels": total_kernels,
        "kernel_dup_groups": len(dups),
        "kernel_dup_groups_cross_file":
            len([g for g in dups if g["distinct_files"] >= 2]),
        "kernel_wasted_lines":
            sum(g["lines"] * (g["copies"] - 1) for g in dups),
        "host_dup_groups": len(host_dups),
        "host_wasted_lines":
            sum(g["lines"] * (g["copies"] - 1) for g in host_dups),
        "kernel_groups": dups,
        "host_groups": host_dups,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", help="repository root")
    ap.add_argument("--json", required=True, help="output JSON report path")
    ap.add_argument("--min-lines", type=int, default=6,
                    help="ignore kernel bodies shorter than this (default 6)")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    report = scan(root, args.min_lines)
    Path(args.json).write_text(json.dumps(report, indent=1))
    print(f"cu/cuh files: {report['scanned_cu_files']}, "
          f"kernels: {report['total_kernels']}")
    print(f"kernel dup groups: {report['kernel_dup_groups']} "
          f"(cross-file {report['kernel_dup_groups_cross_file']}), "
          f"wasted lines: {report['kernel_wasted_lines']}")
    print(f"host dup groups (>=12 lines, cross-file): "
          f"{report['host_dup_groups']}, "
          f"wasted lines: {report['host_wasted_lines']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())