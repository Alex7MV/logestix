# `OPEN_TASKS.md` reorganization — design (2026-11-20)

## Intent and why

`docs/OPEN_TASKS.md` is 2609 lines / 216 KB (≈55k tokens). Every session that asks
"what is open" pays that cost, and `AGENTS.md` points at the file as the answer to
"what is open / known broken", so the cost is paid often. The file's own rule —
"only genuinely open work lives here" — stopped being applied: roughly two thirds
of it is ingestion narrative where 1–3 bullets are open and the rest is the
post-mortem, and its own `## Archive` (line 1502) + `## Maintenance rule`
(line 1592) are followed by another ~1000 lines, so 38 % of the live file sits
*below* its footer where nobody looks.

## Goals

- One file — the index — answers "what is open, for whom, and where the detail
  is" in ≤150 lines.
- Reading one model's or one subsystem's open items costs one file and ≤150
  lines.
- Nothing is lost: every move is **verbatim**, no prose is rewritten, no
  evidence is summarised away.
- Item IDs (`§A2`, `§G1`, `§R17 P3`, `§S4`, …) never move or get renumbered:
  `tests/CMakeLists.txt` and ~12 `tests/*.cpp` comments cite them by ID.

## Non-goals

- Not touching `models/<org>/<model>.json`, `docs/MODELS.md`, the generators, or any test —
  this is a documentation change inside `docs/`. The 20 missing stage-9 reports
  stay a listed open item, not part of this change.
- Not creating a second source of truth: the archive index stays in
  `docs/open-tasks/README.md`.

## Anti-goals

- A "live" file that duplicates what the index says — that is the current file
  with a table of contents.
- An index with one row per bullet. At ~85 open bullets that is the old file
  again; rows are per *area* (one row per live file), with an open count.

## Approaches considered

| | approach | verdict |
|---|---|---|
| A | Clean the single file down to ~600–700 lines: open items only, narratives archived, `## Archive` table compressed, section order repaired | Cheapest, zero risk — but the file stays one monolith and the 2/3-of-a-file ingestion narrative grows back with the next ingests |
| **B** | **Index + live files: `OPEN_TASKS.md` becomes a ≤150-line index, content moves to `docs/open-tasks/live/` by area and by model** | **Chosen.** Reading cost per question drops by ~20x; each move is verbatim so the trail survives; no schema or code touched |
| C | B, plus open items move into `models/<org>/<model>.json` and narratives become stage-9 reports | Right eventually, but it changes the manifest schema the generator and guards validate. Out of scope for a documentation repair |

## The file map

`docs/open-tasks/live/` (new; archive directory stays what it is — closed
snapshots).

| live file | absorbs (sections of today's file) |
|---|---|
| `engine-defects.md` | §A1 (qwen/ornith `cudaErrorIllegalAddress`) |
| `instruments-and-guards.md` | §G1 (lock is per-process), §G2 + §G2a + §G2b (sibling coverage / the guard), §G5, §G6, §G7, §G8 (instrument lessons, `Status: fixed`), §G9 (flaky bandwidth probe, open) |
| `parity-goldens.md` | §K (declared-debt mechanism + the rejected golden set) |
| `infra-migrations.md` | §M (test-registration migration remainder), §S1–§S7 (Arrow IPC residuals), "Model manifest migration — what it did not cover" incl. *Stage-9 reports missing for 20 models* and *The ctest oracle is a dated snapshot* |
| `models/kimi-k2.7-code.md` | §R2 (K1, K2, K4 open; K3 archived) |
| `models/gemma-4-31b-it.md` | §R |
| `models/deepseek-v4-pro-0813.md` | §R3 (P2 open; residency note) |
| `models/deepseek-v4.1-flash.md` | §Q + §A2 (the cancel-poisons-next-request defect is this family's instance) |
| `models/minimax-m3.md` | §R4 |
| `models/hy-memory-30b-a3b.md` | §R5 |
| `models/mistral-small-4-119b.md` | §T (incl. T.1) |
| `models/mistral-large-4-1t-a52b.md` | §R14 |
| `models/flint.md` | §R6 |
| `models/ling-3.1-flash.md` | §R7 |
| `models/beam-1.0-501b-a23b.md` | §R8 |
| `models/step-5-preview.md` | §R9 |
| `models/granite-switch-4.1-30b.md` | §R10 |
| `models/hy-simulmt-30b-a3b.md` | §R11 |
| `models/gigachat3.5-432b-a28b.md` | §R12 + §G4 (a `/tmp` conversion that vanished is gigachat's instance) |
| `models/nemotron-3.5-lightning-30b-a3b-nvfp4.md` | §R13 (N1, N2) |
| `models/nemotron-3.5-lightning-30b-a3b.md` | §R20 (N1–N3) |
| `models/nemotron-3-ultra-550b-a55b.md` | §R19 (O1–O4) |
| `models/kimi-k3.md` | §R15 (K1–K7) |
| `models/kimi-k3.1-pro.md` | §R17 (P1, P2, P3, P6 open) |
| `models/dory1.0-4b.md` | §R18 (R18a fixed, R18b a recorded deviation) |

Retired to the archive, because they are records rather than work (each becomes
`docs/open-tasks/<date>-<slug>.md`, verbatim, plus a README row):

- §U (CUDA 13.5 adoption — "remains a record, not a work item");
- the `Status: fixed` halves of §G5/§G6/§G7/§G8 that are pure lesson (their live
  residue stays in `instruments-and-guards.md`);
- §K's "rejected golden set" evidence block.

## The index

`docs/OPEN_TASKS.md`, ≈40 rows, shape:

```
# Open tasks

Rule: only genuinely open work. Detail lives in `live/` — one row per file.
Closed work moves to `docs/open-tasks/<date>-<slug>.md` (index:
`docs/open-tasks/README.md`).

## Areas
| file | open | state, one line |
|---|---|---|
| [engine-defects](open-tasks/live/engine-defects.md) | 1 | §A1 Windows/5060 Ti kernel fault, needs that box |
| [instruments-and-guards](open-tasks/live/instruments-and-guards.md) | 5 | §G1 lock scope, §G2a target naming, §G9 flaky probe |
| …

## Models
| model | file | open | state, one line |
|---|---|---|---|
| kimi-k3.1-pro | [models/kimi-k3.1-pro.md](…) | 2 | DSA unverified; parity sweep ~7 h unrun |
…

## Where does a new item go
| item is about | goes to |
|---|---|
| a kernel/runtime fault any family can hit | live/engine-defects.md |
| a gate, guard, harness or instrument | live/instruments-and-guards.md |
| golden provenance, debt, capture tools | live/parity-goldens.md |
| build wiring, CMake/ctest registration, Arrow IPC | live/infra-migrations.md |
| one model's gates, precision, perf, tokenizer | live/models/<slug>.md |

IDs are never renumbered. New items take the next free number in their area.
Archive lookup for retired IDs: `docs/open-tasks/README.md` § *Retired IDs*.
```

## Decisions I am making (correct these)

1. **`models/<slug>.md` per family** rather than one `models.md` or grouping by
   size class: it is what makes "read one model" cost one small file.
2. **The `## Archive` table in `OPEN_TASKS.md` is deleted**, after the rows that
   exist only there (B1, D1, D2, E1, G1, "inference-conformance review", "closed
   in this session", `D` KV layout, `L/M/N` codec, ornith framing) are appended
   verbatim to `docs/open-tasks/README.md`. Two indexes of one archive is pure
   duplication; README is the one with links to the files.
3. **A `Retired IDs` table is added to `docs/open-tasks/README.md`** mapping
   `§E1a`, `§G6`, `§R17 P5`, … to the archive file that now holds them, so the
   code comments that cite an ID still resolve after the move.
4. **`tools/check_open_tasks.py`** (new, ~60 lines): every file under
   `docs/open-tasks/live/` is linked exactly once from the index, every index
   link resolves, every `§ID` cited inside `live/` exists. Run manually; **not**
   wired into ctest — a docs-only guard does not belong in the 140-target `fast`
   sweep, and the cost this change removes is reading time, not CI time.
5. **No `git mv` of old archive files, no renames** — `docs/open-tasks/*.md`
   names are cited from `OPEN_TASKS.md` history and from model manifests.

## Verification (evidence before "done")

1. **Conservation.** Word-multiset diff of today's file against the union of the
   new files: every sentence that leaves the index must appear in exactly one
   destination. The only unmatched text is expected to be the new index/routing
   text (checked by hand).
2. **ID integrity.** Every `§ID` inside `live/` exists in the same or another
   live file, or in the README `Retired IDs` map. Every ID that changed state
   (`§G5` fixed → archive) is in the map.
3. **Links.** `python3 tools/check_open_tasks.py` green.
4. **No code impact.** `grep -rn 'OPEN_TASKS' tests/ tools/` still resolves (IDs
   kept); `python3 tools/gen_model_manifest.py --check --root .` unchanged;
   markdown-only change, so no build.

## What is left to the builder

- The 25→3:1 prose splitting itself (mechanical, verbatim).
- Judging the boundary in the four mixed sections (`§G2`, `§G5`–`§G8`, `§K`,
  `§U`): live residue vs archived lesson.
- Writing the one-line "state" for each index row — the only genuinely new prose
  in this change, and the thing that makes the index worth reading.
