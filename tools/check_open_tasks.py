#!/usr/bin/env python3
"""Consistency checker for the OPEN_TASKS index and its `live/` detail files.

`docs/OPEN_TASKS.md` routes; the detail lives in `docs/open-tasks/live/**`.
Nothing else checks that the routing is complete, so this script does:

    index     docs/OPEN_TASKS.md exists
    orphan    every live/**/*.md is linked from the index exactly once
    deadlink  every index link into live/ resolves
    title     every live file starts with a `# ` title
    id        every `§ID` cited inside live/ resolves to a live heading or a
              `Retired IDs` row in docs/open-tasks/README.md
    idcode    same for `§ID` cited from tests/, libs/ and apps/ (the comments
              that make the IDs load-bearing)

Run by hand; deliberately not wired into ctest (a docs guard does not belong in
the 140-target fast sweep).

    python3 tools/check_open_tasks.py --root .
    python3 tools/check_open_tasks.py --selftest
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

HEADING_ID = re.compile(r"^#{2,4} +([A-Z]\d{0,2}[a-z]?)\.")
CITATION = re.compile(r"§([A-Z]\d{0,2}[a-z]?)")
LIVE_LINK = re.compile(r"\]\((open-tasks/live/[^)\s]+\.md)\)")
RETIRED_START = re.compile(r"^## +Retired IDs")
RETIRED_ROW = re.compile(r"^\|\s*(.+?)\s*\|")
CODE_SUFFIXES = (".cpp", ".h", ".cu", ".cuh", ".cmake")
CODE_ROOTS = ("tests", "libs", "apps")


def _retired_ids(readme: Path) -> set[str]:
    """IDs listed in the README's `Retired IDs` table (first column only)."""
    if not readme.is_file():
        return set()
    ids, in_section = set(), False
    for line in readme.read_text().splitlines():
        if RETIRED_START.match(line):
            in_section = True
            continue
        if in_section and line.startswith("## "):
            in_section = False
        elif in_section:
            m = RETIRED_ROW.match(line)
            if m:
                # `R17 P5` names a sub-item of the live `R17`: it retires the
                # reference, not the parent ID, so it contributes nothing here.
                for ident, sub in re.findall(r"([A-Z]\d{0,2}[a-z]?)(\s+P\d+)?", m.group(1)):
                    if not sub:
                        ids.add(ident)
    return ids


def _code_files(root: Path) -> list[Path]:
    """Sources that cite item IDs: the tests, the libraries and the apps."""
    out: list[Path] = []
    for top in CODE_ROOTS:
        base = root / top
        if not base.is_dir():
            continue
        out.extend(
            p for p in base.rglob("*")
            if p.is_file() and (p.suffix in CODE_SUFFIXES or p.name == "CMakeLists.txt")
        )
    return sorted(out)


def check_tree(root: Path) -> list[str]:
    """Return one `code detail` string per violation; empty means consistent."""
    docs = root / "docs"
    index = docs / "OPEN_TASKS.md"
    live_dir = docs / "open-tasks" / "live"
    violations: list[str] = []

    if not index.is_file():
        return [f"index {index.relative_to(root)} missing"]

    index_text = index.read_text()
    links = LIVE_LINK.findall(index_text)
    linked_to = {link: links.count(link) for link in links}

    live = sorted(live_dir.rglob("*.md")) if live_dir.is_dir() else []
    heading_ids: set[str] = set()
    for path in live:
        rel = path.relative_to(docs).as_posix()
        count = linked_to.get(rel, 0)
        if count != 1:
            violations.append(f"orphan {rel} linked {count} times")
        first = next((ln for ln in path.read_text().splitlines() if ln.strip()), "")
        if not first.startswith("# "):
            violations.append(f"title {rel} does not start with a `# ` heading")
        for line in path.read_text().splitlines():
            m = HEADING_ID.match(line)
            if m:
                heading_ids.add(m.group(1))

    for link in links:
        if not (docs / link).is_file():
            violations.append(f"deadlink {link}")

    retired = _retired_ids(docs / "open-tasks" / "README.md")
    resolvable = heading_ids | retired
    for ident in sorted(retired & heading_ids):
        violations.append(f"stale {ident} is a live heading and also listed under Retired IDs")
    for path in live:
        rel = path.relative_to(docs).as_posix()
        for ident in sorted(set(CITATION.findall(path.read_text())) - resolvable):
            violations.append(f"id {rel} cites §{ident}")
    for path in _code_files(root):
        rel = path.relative_to(root).as_posix()
        for ident in sorted(set(CITATION.findall(path.read_text(errors="replace"))) - resolvable):
            violations.append(f"idcode {rel} cites §{ident}")

    return violations


# --- selftest -----------------------------------------------------------------

CLEAN = {
    "docs/OPEN_TASKS.md": "# Open tasks\n\n| [a](open-tasks/live/a.md) | state |\n",
    "docs/open-tasks/README.md": "# Open-tasks archive\n",
    "docs/open-tasks/live/a.md": "# A area\n\n### A1. first item\n\nBody cites §A1.\n",
}

def _with(base: dict[str, str], **extra: str) -> dict[str, str]:
    return {**base, **{k.replace("__", "/").replace("_md", ".md"): v for k, v in extra.items()}}

DEADLINK_TREE = {
    **CLEAN,
    "docs/OPEN_TASKS.md": (
        "# Open tasks\n\n| [a](open-tasks/live/a.md) | state |\n"
        "| [gone](open-tasks/live/gone.md) | state |\n"
    ),
}
ORPHAN_TREE = {**CLEAN, "docs/open-tasks/live/b.md": "# B area\n\n### B1. item\n"}
TITLE_TREE = {**CLEAN, "docs/open-tasks/live/a.md": "Body first\n\n### A1. first item\n"}
ID_TREE = {**CLEAN, "docs/open-tasks/live/a.md": "# A area\n\n### A1. first item\n\nSee §Z9.\n"}
IDCODE_TREE = {**CLEAN, "tests/x.cpp": "// see OPEN_TASKS §E1a\n"}
IDCODE_RETIRED_TREE = {
    **IDCODE_TREE,
    "docs/open-tasks/README.md": (
        "# Open-tasks archive\n\n## Retired IDs\n\n| ID | now in |\n|---|---|\n"
        "| `E1a` | `2026-01-01-x.md` |\n"
    ),
}
# Non-vacuity: the heading that resolves §A1 is deleted, so the same citation
# must now be reported. A checker that passes on an empty set passes this tree.
NONVACUOUS_TREE = {**CLEAN, "docs/open-tasks/live/a.md": "# A area\n\nBody cites §A1.\n"}
# A retired row that names an ID which is still live would shadow the live item
# the next time someone reads the map looking for it.
STALE_TREE = {
    **CLEAN,
    "docs/open-tasks/README.md": (
        "# Open-tasks archive\n\n## Retired IDs\n\n| ID | now in |\n|---|---|\n"
        "| `A1` | `2026-01-01-x.md` |\n"
    ),
}

CASES: list[tuple[str, dict[str, str], set[str], str | None]] = [
    ("clean tree has no violations", CLEAN, set(), None),
    ("an unlinked live file is an orphan", ORPHAN_TREE, {"orphan"}, "live/b.md"),
    ("an index link with no file is dead", DEADLINK_TREE, {"deadlink"}, "gone.md"),
    ("a live file without a title is reported", TITLE_TREE, {"title"}, "live/a.md"),
    ("a citation with no heading is unresolved", ID_TREE, {"id"}, "Z9"),
    ("a citation from tests/ with no retired row is unresolved", IDCODE_TREE, {"idcode"}, "E1a"),
    ("a retired row resolves a tests/ citation", IDCODE_RETIRED_TREE, set(), None),
    ("the id check is not vacuous", NONVACUOUS_TREE, {"id"}, "A1"),
    ("a retired row for a live id is stale", STALE_TREE, {"stale"}, "A1"),
]


def selftest() -> int:
    failures = 0
    for name, files, want_codes, want_detail in CASES:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel, text in files.items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            found = check_tree(root)
        codes = {v.split(" ", 1)[0] for v in found}
        detail_ok = want_detail is None or any(want_detail in v for v in found)
        if codes != want_codes or not detail_ok:
            failures += 1
            print(f"FAIL selftest: {name}: want {sorted(want_codes)} "
                  f"(detail {want_detail!r}), got {found}")
    total = len(CASES)
    if failures:
        print(f"selftest: {total - failures}/{total} cases behaved")
        return 1
    print(f"selftest: {total}/{total} cases behaved")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", help="repository root (default: .)")
    ap.add_argument("--selftest", action="store_true", help="run the fixture cases")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    violations = check_tree(Path(args.root))
    for v in violations:
        print(f"FAIL {v}")
    if violations:
        return 1
    print("open_tasks: consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
