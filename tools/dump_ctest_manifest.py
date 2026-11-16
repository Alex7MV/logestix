#!/usr/bin/env python3
"""Dump every CTest target of a configured preset tree to JSON.

This is the coverage oracle for the model-manifest migration: it is captured
before a single manifest exists, and every later phase diffs against it. A
migration that silently drops a target, a label or a timeout shows up here.

**The committed snapshot is dated, not live.** `tests/fixtures/ctest/
2026-11-06-baseline.json` describes the tree as of 2026-11-06; re-capturing it
today is not the same file (one guard removed, one self-test added), so
overwriting it would destroy the only pre-manifest record. Because the diff also
flags any emitted target the snapshot does not contain, the check starts failing
on the first model added after that date — by design, as growth. It is migration
evidence, not a suite gate (see `docs/OPEN_TASKS.md`, "Model manifest
migration"). Use it as the "before" of a *future* migration: capture before the
first edit.

Reads the CMake-generated `CTestTestfile.cmake` files (the same source ctest
itself parses), so it does not depend on the project's CMake text.

The committed baseline stores commands relative to a `<REPO_ROOT>/` placeholder
instead of the capturing host's absolute checkout path; `--diff-cmake` compares
labels/timeout/skip/environment/resource_lock only, so the placeholder does not
affect the diff. A freshly dumped snapshot carries absolute paths — replace the
checkout prefix with `<REPO_ROOT>/` before committing it.

Usage:
    tools/dump_ctest_manifest.py --preset linux-debug --out tests/fixtures/ctest/base.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

# `add_test(test_core "/path/to/test_core")` — name first, command after.
ADD_TEST = re.compile(r"""^add_test\("?([^"\s)]+)"?\s+(.*)\)\s*$""")
SET_PROPS = re.compile(
    r"""^set_tests_properties\((?P<names>[^)]*?)\s+PROPERTIES\s+(?P<props>.*)\)\s*$""")
SET_PROPERTY = re.compile(
    r"""^set_property\(TEST\s+(?P<names>[^)]*?)\s+PROPERTY\s+(?P<key>\w+)\s+(?P<value>.+)\)\s*$""")

# Properties we carry. `_BACKTRACE_TRIPLES` is a CMake implementation detail.
WANTED = ("LABELS", "TIMEOUT", "SKIP_RETURN_CODE", "ENVIRONMENT",
          "RESOURCE_LOCK", "WORKING_DIRECTORY", "DEPENDS")

INT_PROPS = {"TIMEOUT", "SKIP_RETURN_CODE"}


def unquote(token: str) -> str:
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] == '"':
        return token[1:-1]
    return token


def split_top_level(text: str):
    """Split a CMake argument list on whitespace, honouring double quotes."""
    out, cur, quoted = [], [], False
    for ch in text:
        if ch == '"':
            quoted = not quoted
            cur.append(ch)
        elif ch.isspace() and not quoted:
            if cur:
                out.append("".join(cur))
                cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def scan_file(path: Path, tests: dict):
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
        return

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        m = ADD_TEST.match(line)
        if m:
            name, rest = m.group(1), m.group(2)
            args = split_top_level(rest)
            if args and args[0] == "NAME":       # new-style add_test(NAME x COMMAND y)
                name = unquote(args[1])
                args = args[3:] if len(args) > 3 else []
            tests.setdefault(name, {
                "name": name,
                "command": unquote(args[0]) if args else "",
                # Everything after the executable: a Catch2 name/tag filter for
                # a second test case over one binary (perf rows, [tool]* cases).
                "args": [unquote(a) for a in args[1:]],
                "labels": [],
                "timeout": None,
                "skip_return_code": None,
                "environment": [],
                "resource_lock": None,
                "working_directory": None,
                "depends": [],
            })
            continue

        m = SET_PROPS.match(line)
        if m:
            names = [unquote(n) for n in split_top_level(m.group("names"))]
            props = split_top_level(m.group("props"))
            _apply(tests, names, props)
            continue

        m = SET_PROPERTY.match(line)
        if m:
            names = [unquote(n) for n in split_top_level(m.group("names"))]
            _apply(tests, names, [m.group("key"), m.group("value")])


def _apply(tests: dict, names, props):
    i = 0
    while i < len(props):
        key = props[i]
        if key not in WANTED or i + 1 >= len(props):
            i += 2 if key in WANTED else 1
            continue
        value = unquote(props[i + 1])
        for name in names:
            entry = tests.get(name)
            if entry is None:
                continue
            if key == "LABELS":
                entry["labels"] = value.split(";") if value else []
            elif key in INT_PROPS:
                try:
                    entry[key.lower()] = int(value)
                except ValueError:
                    entry[key.lower()] = None
            elif key == "ENVIRONMENT":
                entry["environment"] = value.split(";") if value else []
            elif key == "RESOURCE_LOCK":
                entry["resource_lock"] = value
            elif key == "WORKING_DIRECTORY":
                entry["working_directory"] = value
            elif key == "DEPENDS":
                entry["depends"] = value.split(";") if value else []
        i += 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="linux-debug",
                    help="build preset whose CMake tree to read")
    ap.add_argument("--build-root", default="build",
                    help="directory holding the preset trees (default: build)")
    ap.add_argument("--out", required=True, help="JSON file to write")
    args = ap.parse_args(argv)

    tree = Path(args.build_root) / args.preset
    if not (tree / "CTestTestfile.cmake").is_file():
        print(f"error: {tree} is not a configured preset tree", file=sys.stderr)
        return 2

    tests: dict = {}
    files = sorted(tree.rglob("CTestTestfile.cmake"))
    for path in files:
        scan_file(path, tests)

    doc = {
        "preset": args.preset,
        "tests": [tests[name] for name in sorted(tests)],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    unlabelled = [t["name"] for t in doc["tests"] if not t["labels"]]
    print(f"dumped {len(doc['tests'])} targets from {len(files)} files -> {out}",
          file=sys.stderr)
    if unlabelled:
        print(f"note: {len(unlabelled)} targets carry no label "
              f"(e.g. {unlabelled[:5]})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
