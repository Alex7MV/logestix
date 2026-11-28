# `OPEN_TASKS.md` reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. (`AGENTS.md` forbids subagents, so the subagent-driven path is not available in this repository.)

**Goal:** Turn `docs/OPEN_TASKS.md` (2609 lines / 216 KB) into a ≤150-line index whose detail lives in `docs/open-tasks/live/` — four area files plus one file per model — moving every byte verbatim and keeping every `§ID` resolvable for the `tests/` comments that cite it.

**Architecture:** The index keeps only routing (`| file | state |` tables). Content is split by heading into `live/<area>.md` and `live/models/<slug>.md`; closed records (§U, §G5–§G8, §K's rejected-golden evidence) go to one archive snapshot. `docs/open-tasks/README.md` becomes the single archive index (the duplicated `## Archive` table in `OPEN_TASKS.md` is deleted after its unique rows move there) and gains a `Retired IDs` map so an archived `§E1a` still resolves.

**Tech Stack:** Markdown, Python 3 (one checker script, stdlib only), git.

**Spec:** `docs/superpowers/specs/2026-11-20-open-tasks-reorg-design.md`

## Global Constraints

- **Verbatim moves.** No prose is rewritten, summarised or trimmed when it changes file. Only heading level changes (`##`→`###`, `###`→`####`) and the new header/routing text are new words.
- **IDs never renumber.** `A1`, `A2`, `K`, `M`, `Q`, `R`…`R20`, `S1`–`S7`, `T`/`T.1`, `G1`, `G2`/`G2a`/`G2b`, `G8a`, `G9`, `U` are stable; `tests/CMakeLists.txt` and ~12 `tests/*.cpp` comments cite them.
- **Docs only.** No change under `libs/`, `apps/`, `tests/`, `tools/gen_model_manifest.py`, `models/<org>/<model>.json`. `tools/check_open_tasks.py` is the only new code.
- **No ctest wiring.** The checker runs by hand; a docs guard does not belong in the 140-target `fast` sweep.
- **Archive files are never edited.** New facts go to the live file.
- **Commit only on explicit request** (`AGENTS.md`). The plan stops at a verified working tree otherwise.

## Review Focus

- **A section split mid-block.** `## T.` owns `### T.1`; `## G2.` owns `### G2a`/`### G2b`; `## R18.` owns `### R18a`/`### R18b` and its own heading wraps onto a second line. A naive "split at `## `" loses the children — the word-multiset conservation check in Task 3 is what proves it did not.
- **An archived ID that code still cites.** `§E1a`, `§E1d`, `§E1e`, `§G6`, `§G7` are cited from `tests/` and are *already* absent from today's live file. Task 4 must put every one of them in the README `Retired IDs` map, and Task 1's checker must fail while any is missing.
- **The index linking a file that was not created** (or a created file no row links). Both directions are checked; a one-directional check passes on a tree that silently lost a model.
- **`§`-citations that point at a sibling's item.** `§R17 P6` is cited from `tests/CMakeLists.txt` and `§Q`/`§A2` cross-reference each other; they must land in files that exist, not in files whose name changed during the split.
- **A "live" file that is really an archive.** `§G5`–`§G8` all say `Status: fixed`; leaving them in `live/` re-grows the file this change exists to shrink. Only their open residues stay (`§G8a`).

---

### Task 1: `tools/check_open_tasks.py` + `--selftest`

**Files:**
- Create: `tools/check_open_tasks.py`
- Test: the script's own `--selftest` mode (temp trees under `tempfile.mkdtemp()`; no files under `tests/`)

**Interfaces:**
- Consumes: nothing.
- Produces: `python3 tools/check_open_tasks.py --root <dir>` → exit 0 when the tree is consistent, exit 1 with one `FAIL <check>: <detail>` line per violation; `--selftest` → exit 0 when every fixture case behaves, exit 1 otherwise. Later tasks run both.

**Checks the script implements** (over `<root>/docs`):

1. `index`: `<root>/docs/OPEN_TASKS.md` exists.
2. `orphan`: every `docs/open-tasks/live/**/*.md` is the target of exactly one markdown link in `OPEN_TASKS.md`.
3. `deadlink`: every link in `OPEN_TASKS.md` to `open-tasks/live/...` resolves.
4. `title`: every live file's first non-empty line is `# <something>`.
5. `id`: every `§([A-Z]\d{0,2}[a-z]?)` cited **inside** `docs/open-tasks/live/**` resolves to a heading ID in `live/` (`^#{2,4}\s+(ID)\.`) or to an ID in the `Retired IDs` table of `docs/open-tasks/README.md`.
6. `idcode`: every `§ID` cited in `<root>/tests/**/*.{cpp,h,cu,cmake}` and `<root>/tests/CMakeLists.txt` resolves the same way (against the two sources above, not against a third list).

Rules the script must encode: a heading ID is the token before the first `. ` in a `##`/`###`/`####` heading (`### A1. text` → `A1`; `#### G8a. text` → `G8a`; `## R. text` → `R`); `Retired IDs` rows are `| ID | destination |` table lines.

- [ ] **Step 1: Write the failing selftest cases**

In `tools/check_open_tasks.py`, `selftest()` builds a `docs/` tree from a `dict` of relative path → text and asserts `check_tree(root)` returns the expected violation codes:

| case | tree | expected |
|---|---|---|
| clean | index links `live/a.md`; `live/a.md` starts `# A` and cites `§A1`, defined by `### A1. x` | `[]` |
| orphan | as clean, plus `live/b.md` | `["orphan"]` |
| deadlink | index links `live/gone.md` | `["deadlink"]` |
| title | `live/a.md` starts `Some text` | `["title"]` |
| id | `live/a.md` cites `§Z9`, no such heading | `["id"]` |
| idcode | `tests/x.cpp` cites `§E1a`, README has no `Retired IDs` row for it | `["idcode"]` |
| idcode-retired | same, README gains `| E1a | open-tasks/2026-01-01-x.md |` | `[]` |
| non-vacuity | clean tree, then delete the heading `### A1. x` | `["id"]` |

- [ ] **Step 2: Run the selftest to verify it fails**

Run: `python3 tools/check_open_tasks.py --selftest`
Expected: `ModuleNotFoundError`-free but `selftest` unimplemented → `FAIL selftest: not implemented`, exit 1.

- [ ] **Step 3: Implement the checks**

`check_tree(root: Path) -> list[str]` returns violation code strings (one per offending path, e.g. `orphan docs/open-tasks/live/b.md`); `main()` prints them as `FAIL <code>: <detail>` and exits 1. `--selftest` runs the table above and prints `selftest: 8/8 cases behaved`. Regexes: heading `^#{2,4}[ ]+([A-Z]\d{0,2}[a-z]?)\.[ ]`, citation `§([A-Z]\d{0,2}[a-z]?)`.

- [ ] **Step 4: Run the selftest to verify it passes, and run it against the real tree**

Run: `python3 tools/check_open_tasks.py --selftest` → `selftest: 8/8 cases behaved`
Run: `python3 tools/check_open_tasks.py --root .` → exit 1 with `FAIL orphan`/`FAIL index`-class output (there is no `live/` yet, and `tests/` cites archived IDs with no `Retired IDs` map). Save that output to `/tmp/opencode/open-tasks-red.txt` — it is both the RED evidence for Task 2 and the work list for Task 4.

- [ ] **Step 5: Commit** — only on explicit request:
`git add tools/check_open_tasks.py && git commit -m "test: add a checker for the OPEN_TASKS index, live areas and ID references"`

---

### Task 2: split `live/` verbatim and archive the closed records

**Files:**
- Create: `docs/open-tasks/live/engine-defects.md`, `instruments-and-guards.md`, `parity-goldens.md`, `infra-migrations.md`
- Create: `docs/open-tasks/live/models/{kimi-k2.7-code, gemma-4-31b-it, deepseek-v4-pro-0813, deepseek-v4.1-flash, minimax-m3, hy-memory-30b-a3b, mistral-small-4-119b, mistral-large-4-1t-a52b, flint, ling-3.1-flash, beam-1.0-501b-a23b, step-5-preview, granite-switch-4.1-30b, hy-simulmt-30b-a3b, gigachat3.5-432b-a28b, nemotron-3.5-lightning-30b-a3b-nvfp4, nemotron-3.5-lightning-30b-a3b, nemotron-3-ultra-550b-a55b, kimi-k3, kimi-k3.1-pro, dory1.0-4b}.md`
- Create: `docs/open-tasks/2026-11-20-open-tasks-index-split-closed.md`
- Modify: `docs/OPEN_TASKS.md` (content removed only; the index is Task 3)
- Delete after use: `/tmp/opencode/split_open_tasks.py`

**Interfaces:**
- Consumes: today's `docs/OPEN_TASKS.md` at `git show HEAD:docs/OPEN_TASKS.md` (a stable input — Task 3 replaces the file).
- Produces: the 25 live files and one archive file that Task 3's index links by exact relative path.

**Section → destination map** (keys are literal heading prefixes in today's file):

| source heading | destination |
|---|---|
| `## A. Real defects` → `### A1.` block only | `live/engine-defects.md` (keep the section's 4-line preamble as the file intro) |
| `## A.` → `### A2.` block | `live/models/deepseek-v4.1-flash.md` |
| `## Q.` | `live/models/deepseek-v4.1-flash.md` |
| `## R.` | `live/models/gemma-4-31b-it.md` |
| `## R2.` | `live/models/kimi-k2.7-code.md` |
| `## R3.` | `live/models/deepseek-v4-pro-0813.md` |
| `## R4.` | `live/models/minimax-m3.md` |
| `## R5.` | `live/models/hy-memory-30b-a3b.md` |
| `## T.` (with `### T.1.`) | `live/models/mistral-small-4-119b.md` |
| `## R14.` | `live/models/mistral-large-4-1t-a52b.md` |
| `## R6.` / `## R7.` / `## R8.` / `## R9.` / `## R10.` / `## R11.` / `## R12.` / `## R13.` | `live/models/{flint, ling-3.1-flash, beam-1.0-501b-a23b, step-5-preview, granite-switch-4.1-30b, hy-simulmt-30b-a3b, gigachat3.5-432b-a28b, nemotron-3.5-lightning-30b-a3b-nvfp4}.md` |
| `## R15.` / `## R17.` / `## R18.` (with `### R18a`,`### R18b`) / `## R19.` / `## R20.` | `live/models/{kimi-k3, kimi-k3.1-pro, dory1.0-4b, nemotron-3-ultra-550b-a55b, nemotron-3.5-lightning-30b-a3b}.md` |
| `## K.` minus the "rejected golden set" block | `live/parity-goldens.md` |
| `## G1.`, `## G2.` (with `### G2a`,`### G2b`), `## G9.`, `### G8a.` | `live/instruments-and-guards.md` |
| `## G4.` | `live/models/gigachat3.5-432b-a28b.md` (append to that file) |
| `## S.` (with `### S1.`–`### S7.`), `## M.`, `## Model manifest migration (2026-11-06)` (with its two `###`) | `live/infra-migrations.md` |
| `## U.`, `## G5.`, `## G6.`, `## G7.`, `## G8.`, `## K.`'s "rejected golden set" block | `docs/open-tasks/2026-11-20-open-tasks-index-split-closed.md` |
| `## Archive`, `## Maintenance rule` | deleted here — rows move to `docs/open-tasks/README.md` in Task 4, the rule is restated in the index in Task 3 |

- [ ] **Step 1: Write the one-shot splitter**

`/tmp/opencode/split_open_tasks.py` reads `git show HEAD:docs/OPEN_TASKS.md`, splits on `^## ` (treating a wrapped heading's continuation line as part of the heading — `## R18.` is the only one), keeps the byte-identical body of each block, applies the map, demotes `## X.`→`### X.` and `### Y.`→`#### Y.` inside a destination file, and writes each destination with a `# <Title>` first line. Where two sources feed one destination (§A2 + §Q, §G4 appended after §R12), it concatenates in the order listed above. It prints `wrote <path>: <lines> lines` per file and a `<n> source lines unassigned` count.

- [ ] **Step 2: Run it and confirm nothing is unassigned**

Run: `python3 /tmp/opencode/split_open_tasks.py`
Expected: 26 `wrote …` lines, `0 source lines unassigned`. Any non-zero count means a heading is missing from the map — fix the map, not the prose.

- [ ] **Step 3: Verify conservation by word multiset**

```bash
python3 - <<'PY'
import collections, pathlib, re, subprocess
old = subprocess.run(["git","show","HEAD:docs/OPEN_TASKS.md"],capture_output=True,text=True).stdout
newn = "".join(p.read_text() for p in pathlib.Path("docs/open-tasks/live").rglob("*.md"))
newn += pathlib.Path("docs/open-tasks/2026-11-20-open-tasks-index-split-closed.md").read_text()
w = lambda s: collections.Counter(re.findall(r"\w+", s))
missing = w(old) - w(newn)
print("missing tokens:", sum(missing.values()))
for k, v in missing.most_common(20): print(f"  {v:4d}  {k}")
PY
```
Expected: `missing tokens: 0`. Non-zero means text was dropped: the top offenders name the block to re-check (heading wording and the deleted `## Archive`/`## Maintenance rule` text account for a small residue — those two blocks are the *only* allowed ones, and their absence must be explained line by line, not waved through).

- [ ] **Step 4: Verify the archive snapshot is complete**

Run: `grep -c '^### ' docs/open-tasks/2026-11-20-open-tasks-index-split-closed.md`
Expected: `6` (`U`, `G5`, `G6`, `G7`, `G8`, and the rejected-golden evidence block, which gets its own `###`). Every one of them says `Status: closed`/`fixed`/`record` in its first three lines.

- [ ] **Step 5: Commit** — only on explicit request:
`git commit -m "docs: split OPEN_TASKS detail into open-tasks/live by area and by model"`

---

### Task 3: the index

**Files:**
- Modify/rewrite: `docs/OPEN_TASKS.md` (≤150 lines)

**Interfaces:**
- Consumes: the 26 paths from Task 2, exact relative form `open-tasks/live/models/<slug>.md`.
- Produces: the link set that Task 1's `orphan`/`deadlink` checks validate, and the `state` one-liners that the next reader sees.

- [ ] **Step 1: Write the index**

Header (replaces today's 11-line preamble and restates the maintenance rule):

```markdown
# Open tasks

**Rule:** only genuinely open work lives here — this file routes, it does not
explain. Detail is one hop away in `open-tasks/live/`; closed work moves to
`open-tasks/<date>-<slug>.md` (index and `Retired IDs`: [`open-tasks/README.md`](open-tasks/README.md)).

Nothing here is speculative — each live item carries the measured value or error
code that produced it.

IDs (`A1`, `G2a`, `R17`, `S4`, …) are stable across moves: `tests/` and
`tests/CMakeLists.txt` cite them by ID. A new item takes the next free number in
its area.
```

Then two tables. **Every row's `state` cell is one line and names the measurement that keeps it open** (not "see file"). Areas, in this order:

| file | state |
|---|---|
| `open-tasks/live/engine-defects.md` | `A1` `cudaErrorIllegalAddress` on qwen/ornith kernels, needs the 5060 Ti box |
| `open-tasks/live/instruments-and-guards.md` | `G1` lock is per-ctest-process; `G2`/`G2a` sibling-named target; `G8a` mistral-small-4 at 95.9 %/96.3 %; `G9` flaky bandwidth probe |
| `open-tasks/live/parity-goldens.md` | `K` the 39-entry declared-debt mechanism, and the rejected golden set |
| `open-tasks/live/infra-migrations.md` | `M` ~390 host registrations; `S1`–`S7` Arrow residuals; manifest gaps: no stage-9 report for 20 models |

Models, alphabetical by slug, 21 rows, `state` = the open IDs plus the number that keeps them open (copy the value from the split file — e.g. `kimi-k3.1-pro`: `R17 P1/P2/P6 — DSA unverified, parity sweep ~7 h unrun`).

**Deviation from the spec, deliberate:** the spec's index shape had an `open` *count* column. It is dropped: a hand-maintained integer per file is exactly the rot this change removes (a file that gains an item would need the count edited by someone who may never open it), and the checker cannot verify a number it has to be told. The `state` line carries the same information qualitatively, with the measurement that keeps the item open.

Close with the routing table and the archive pointer:

```markdown
## Where a new item goes

| item is about | file |
|---|---|
| a kernel/runtime fault any family can hit | `live/engine-defects.md` |
| a gate, guard, harness or instrument reporting the wrong thing | `live/instruments-and-guards.md` |
| golden provenance, declared debt, capture tools | `live/parity-goldens.md` |
| build wiring, ctest registration, Arrow IPC, manifest gaps | `live/infra-migrations.md` |
| one model's gates, precision, perf, tokenizer, residency | `live/models/<slug>.md` |

Archive (closed work, one file per session): [`open-tasks/README.md`](open-tasks/README.md).
```

- [ ] **Step 2: Check the index size and that every row points at a real file**

Run: `wc -l docs/OPEN_TASKS.md` → ≤150.
Run: `grep -o 'open-tasks/live/[a-z0-9./-]*\.md' docs/OPEN_TASKS.md | sort -u | while read f; do test -f "docs/$f" || echo "MISSING $f"; done` → no output.

- [ ] **Step 3: Commit** — only on explicit request:
`git commit -m "docs: OPEN_TASKS becomes a routing index over open-tasks/live"`

---

### Task 4: `docs/open-tasks/README.md` becomes the single archive index

**Files:**
- Modify: `docs/open-tasks/README.md`

**Interfaces:**
- Consumes: the `## Archive` table of today's `OPEN_TASKS.md` (`git show HEAD:docs/OPEN_TASKS.md`) and the `Retired IDs` needs printed by Task 1's `idcode` failures.
- Produces: the `Retired IDs` table that Task 1's `id`/`idcode` checks read, and the single place a reader looks for closed work.

- [ ] **Step 1: Append the rows that exist only in the old table**

Ten rows have no counterpart in `README.md`: `B1`, `D1`, `D2`, `E1`, `G1`, `Inference-conformance review (9 deviations)`, `"Closed in this session" list`, `D — KV cache layout`, `L/M/N — NVFP4 E4M3FN codec…`, `Ornith-1.5-35B-A3B-MLX — prompt-framing defect`. Copy each row's cells verbatim into the README table (the `Item | Outcome` shape becomes `Date | File | Session`, with the destination file the old row already links, or `—` where it never had one and the full text lives in `2026-09-30-cuda-review-windows-bringup.md` / `2026-10-02-mimo-closed.md`, which the README already names).

Then add one row for Task 2's snapshot: `| 2026-11-20 | 2026-11-20-open-tasks-index-split-closed.md | … |`.

- [ ] **Step 2: Add the `Retired IDs` table**

Section header `## Retired IDs`, then `| ID | now in |`:

```markdown
| ID | now in |
|---|---|
| `A2` | `live/models/deepseek-v4.1-flash.md` (still open) |
| `E1`, `E1a`–`E1e` | `2026-10-17-lifecycle-abort-contract-closed.md`, `2026-10-17-qwen38-instance-state-closed.md`, `2026-10-17-gemma-moe-swallowed-decode-error-closed.md` |
| `C1` | `2026-10-23-residual-closures.md` |
| `G5`, `G6`, `G7`, `G8` | `2026-11-20-open-tasks-index-split-closed.md` |
| `U` | `2026-11-20-open-tasks-index-split-closed.md` |
| `§U` (granite tool round-trip) | `2026-10-23-residual-closures.md` |
| `J`, `P`, `R16`, `V1`, `V2`, `B`, `D1`, `F`, `G`, `E4` | `2026-10-24-gemma4-moe-residual-plane-closed.md`, `2026-10-23-glm-5.3-flash-audit-closed.md`, `2026-10-23-doc-hygiene-and-non-defect-records.md` |
| `R17 P5` | `2026-11-05-kimi-k3-decode-window-and-eviction.md` |
```

State the rule in one line above it: this table exists so a `§ID` cited from code still resolves after its item is archived; an ID that is live must NOT be listed here.

- [ ] **Step 3: Run the checker to see what is still unresolved**

Run: `python3 tools/check_open_tasks.py --root .`
Expected: only `FAIL id`/`FAIL idcode` lines naming IDs missing from the map. Read each cited ID out of the failure and add its row — **read the citation site first** (`grep -rn '§ID' tests/`) so the destination is the file that actually holds the item, not a guess.

- [ ] **Step 4: Delete the old `## Archive` table**

It is gone from `docs/OPEN_TASKS.md` by Task 3's rewrite; confirm no row was dropped by re-running Task 2 Step 3's conservation script and checking the missing-token list contains only `Archive`, `Maintenance`, `rule`, `Item`, `Outcome`, `Full`, `text` and the §-heading words.

- [ ] **Step 5: Commit** — only on explicit request:
`git commit -m "docs: README is the single archive index, with a Retired IDs map"`

---

### Task 5: verify green, end to end

**Files:** none modified (unless a check fails).

- [ ] **Step 1: Structure and citations**

Run: `python3 tools/check_open_tasks.py --selftest` → `selftest: 8/8 cases behaved`
Run: `python3 tools/check_open_tasks.py --root .` → exit 0, no `FAIL` lines.

- [ ] **Step 2: Prove the checker is not vacuous**

Copy the tree, break one thing at a time, and confirm each is caught (this is the §G7 discipline — a scan that passes on an empty set passes everything):
```bash
rm -rf /tmp/opencode/t && cp -r . /tmp/opencode/t
python3 tools/check_open_tasks.py --root /tmp/opencode/t            # 0
rm /tmp/opencode/t/docs/open-tasks/live/models/flint.md             # expect deadlink
python3 tools/check_open_tasks.py --root /tmp/opencode/t            # 1, names the dead link
rm -rf /tmp/opencode/t && cp -r . /tmp/opencode/t                   # fresh copy
sed -i 's/§G9/§G99/' /tmp/opencode/t/docs/open-tasks/live/instruments-and-guards.md  # expect id
python3 tools/check_open_tasks.py --root /tmp/opencode/t            # 1, names §G99
```

- [ ] **Step 3: Code comments still resolve**

Run: `grep -rhoE '§[A-Z][0-9]{0,2}[a-z]?' tests/ apps/ libs/ | sort -u` and confirm each appears either as a live heading or as a `Retired IDs` row. (The checker covers `tests/`; this widens it to `apps/` and `libs/` once, by hand.)

- [ ] **Step 4: Nothing but docs changed**

Run: `git status --porcelain` → only `docs/` and `tools/check_open_tasks.py`.
Run: `python3 tools/gen_model_manifest.py --check --docs docs/MODELS.md --root .` → unchanged/clean (the manifest and catalogue do not read `OPEN_TASKS.md`; this proves it).

- [ ] **Step 5: Report the numbers**

`wc -l docs/OPEN_TASKS.md docs/open-tasks/live/*.md docs/open-tasks/live/models/*.md | tail -1`, the index's line count, and the largest live file's — the before/after the design exists for (2609 → index + ≤N lines per file).

- [ ] **Step 6: Commit** — only on explicit request (single commit for the whole reorganization if the earlier ones were not taken).

---

## Execution handoff note

`AGENTS.md` forbids subagents, so this runs **native**: `superpowers:executing-plans` in this session, with `superpowers:test-driven-development` applied per task (Task 1 is a real red→green cycle: the selftest fixtures and the failing run against the current tree come before the split).
