# Kimi-K3.1-Pro report audit and unification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the Kimi-K3.1-Pro model report and the docs around it to the
house shape and to the facts measured after the expert-store eviction fix, and
close the one guard hole the audit found (the variant is absent from the Arrow
audit matrix).

**Architecture:** Documentation-only for Tasks 1–2 (report + the four docs that
restate the variant's numbers or files). Task 3 is a guard/test change: the
variant is added to `kModels` first so the guard fails, then to the audit
matrix so it passes. No library code changes.

**Tech Stack:** Markdown; Catch2 guard test `test_arrow_usage_guard`;
`ctest --preset linux-debug`.

**Spec:** the audit table approved in conversation (findings A1–A7), reproduced
in this plan's task list. The house report shape is
`docs/model-reports/README.md` (mandatory content) plus the 13 existing reports
(`## Publish-ready post` immediately after the verdict, then
`## Internal evidence`).

## Global Constraints

- A gate result is never rewritten: a `SKIP`/`not run` stays a gap, never a pass.
- The publishable section carries no repository-internal paths.
- `tests/perf/perf_baselines.json` is harness-owned; never hand-edited.
- Numbers must be the measured ones: Pro `2.96`…`3.35` s/token as recorded in
  `SELFCHECK_MATRIX` / `OPEN_TASKS` §R17 P5 and the `583e6dd9` verification log.
- Commit prefixes `docs:` / `test:`; commit only on explicit request (the
  operator's request for this work is the explicit request).

## Review Focus

- **A reader takes the green SelfCheck as DSA verification.** Every place that
  reports PASS must keep the "selection is dormant at ≤2048 tokens / arithmetic
  reconstructed" caveat adjacent to it.
- **A cost argument quoted from the pre-fix rate.** The publishable post must
  not repeat `0.12–0.14 t/s` / "over a day"; the shipped rate is 3.2–3.35 s/token
  and the sweep is ~14.7–15.2 h.
- **The remaining-lever paragraph claims the store loop is unchanged.** The
  drain half was removed in `69d898a9`/`583e6dd9`; only the
  overlap-staging-with-compute half is still open.
- **A future session looks for `kimi_moe_pro_load.*`.** It does not exist; the
  indexer plane is pinned in the shared `kimi_moe_k3_load.cpp`.
- **The Arrow guard passes while a model is missing.** `kModels` is a manual
  list and `rows.size() == kModels.size()`, so an unreported model is invisible
  until it is added to both.

---

### Task 1: Rewrite the Kimi-K3.1-Pro report to the house shape

**Files:**
- Modify: `docs/model-reports/2026-11-05-Kimi-K3.1-Pro.md` (full rewrite, content preserved)

**Interfaces:**
- Produces: sections `# … — integration report`, header table,
  `**Verdict at a glance:**`, `## Publish-ready post`,
  `## Internal evidence` with `### Lifecycle stages (Model Ingestion Protocol)`,
  `### What the checkpoint is`, `### What the checkpoint forced into the engine`,
  `### Conversion`, `### Artifact notes for the model authors`,
  `### Measured gates (2026-11-05)`, `### Declared debt`, `### Evidence links`.

- [ ] **Step 1: Rewrite the file** with the approved skeleton. Content moves, it
  is not rewritten: the indexer tensor table, the five "forced into the engine"
  points, the conversion table + verified index facts, the four artifact notes,
  the gates table and the cost table, the seven debt items, and the full
  publish-ready post all survive verbatim except for the three factual fixes
  below.
- [ ] **Step 2: Fix the stale cost argument (A3).** In the publishable
  "Not measured (and why)" paragraph replace
  `at 0.12–0.14 tokens/s one 2048-token prompt is 4–5 hours while the full
  parity-plus-throughput sweep is over a day of exclusive GPU` with
  `at the shipped 3.2–3.35 s/token one 2048-token prompt is ~1.8 h and the
  parity-plus-throughput sweep is ~14.7–15.2 h of exclusive GPU`.
- [ ] **Step 3: Fix the remaining-lever claim (A4).** In `### Declared debt` and
  in the publishable post, state that the **drain** half of the synchronous loop
  is closed (an evicted mapping is retired with a CUDA event instead of a
  `cudaStreamSynchronize`: 20.3 ms → 0.07 ms per eviction, and 6.95 → 6.26
  s/token at a 512-slot window) and that what remains is **overlapping staging
  with compute**, still the shared `ExpertStore` contract.
- [ ] **Step 4: Use the measured range (A5).** Everywhere the report states a
  single throughput, write `3.23–3.35 s/token` and note the two runs it comes
  from (3.42/3.03 → 3.23 before the store change; 3.65/3.04 → 3.35 after). The
  indexer cost becomes `~9–13 %`.
- [ ] **Step 5: Verification step** — `grep -nE "^#{1,3} " docs/model-reports/2026-11-05-Kimi-K3.1-Pro.md`
  must show the header order above; `grep -c "0.12–0.14\|over a day" …` must be 0;
  `grep -n "3.23–3.35" …` must be non-empty.
- [ ] **Step 6: Commit** — `docs(kimi-k31-pro): unify the report with the house shape and the post-fix numbers`.

### Task 2: Align the four docs that restate the variant

**Files:**
- Modify: `AGENTS.md` (Kimi-K3.1-Pro structure entry)
- Modify: `docs/model-reports/README.md` (index rows)
- Modify: `docs/superpowers/plans/2026-11-05-kimi-k31-pro-ingestion.md` (duplicated unticked Task 6 block, loader-file deviation)
- Modify: `docs/SELFCHECK_MATRIX.md`, `docs/OPEN_TASKS.md` §R17 (throughput range)

- [ ] **Step 1 (A2):** In `AGENTS.md` replace the non-existent
  `kimi_moe_pro_load.{cuh,cu}` with the shared loader that actually pins the
  indexer plane (`kimi_moe_k3_load.{h,cpp}`: the `indexer`/`is_pro` branch, five
  FP16 tensors per DSA layer, `indexer_bytes` counted) and say the variant owns
  no loader TU of its own.
- [ ] **Step 2 (A6):** Add the three missing rows to
  `docs/model-reports/README.md`: `Kimi-K3`, `granite-4.2-30b`,
  `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`.
- [ ] **Step 3 (A7):** Delete the duplicated, unticked Task 6 Steps 2–5 block in
  the ingestion plan and record the landed deviation (the loader was folded into
  the shared K3 TU; there is no `kimi_moe_pro_load.*`). Update the report's
  Evidence line so the plan is described as 9 planned tasks plus the two
  follow-up tasks the ledger carries (the expert-window fix and the store
  eviction fix).
- [ ] **Step 4 (A5):** In `docs/SELFCHECK_MATRIX.md` (row + the §1 note) and
  `docs/OPEN_TASKS.md` §R17 P3/P5, state the throughput as
  `3.23–3.35 s/token` and the indexer cost as `~9–13 %`; leave the cost table's
  arithmetic at 3.23 but note it is the lower bound of the range.
- [ ] **Step 5: Verification step** — `grep -rn "kimi_moe_pro_load" AGENTS.md docs/ | grep -v OPEN_TASKS`
  returns only the plan's "as landed" note; the README has one row per report
  file (`ls docs/model-reports/*.md | wc -l` vs the table's row count).
- [ ] **Step 6: Commit** — `docs(kimi-k31-pro): correct the loader file, the report index and the throughput range`.

### Task 3: Close the Arrow-audit hole (A1)

**Files:**
- Test: `tests/test_arrow_usage_guard.cpp` (`kModels`, line ~134)
- Modify: `docs/ARROW_AUDIT_2026-11-01.md` (one row inside `<!-- arrow-matrix:start -->`)

**Interfaces:**
- Consumes: `ReadMatrix()` parses 6 columns and `FileExistsUnderSource()` checks
  every named file exists; the guard asserts `by_model[model] == 1` for every
  `kModels` entry and `rows.size() == kModels.size()`.

- [ ] **Step 1: Add `"Kimi-K3.1-Pro"` to `kModels`** (alphabetically after
  `"Kimi-K3"`). This is the failing-test step: the guard's manual list is the
  only thing that can make the omission visible.
- [ ] **Step 2: Build and run to verify the guard fails.**
  `cmake --build --preset linux-debug --target test_arrow_usage_guard && ./build/linux-debug/tests/test_arrow_usage_guard`
  Expected: FAIL — `by_model[Kimi-K3.1-Pro]` is 0 (and the row count mismatches
  by one).
- [ ] **Step 3: Add the matrix row** inside the markers, mirroring the K3 row:
  `| Kimi-K3.1-Pro | kimi_moe_pro_convert.cpp | kimi_moe_k3_load.cpp | views+ExpertStore (+ transient pinned unpack) | schema.cpp | views+ExpertStore |`
- [ ] **Step 4: Run to verify it passes** —
  `./build/linux-debug/tests/test_arrow_usage_guard` PASS, then
  `ctest --preset linux-debug` 134/134.
- [ ] **Step 5: Commit** — `test(arrow): report Kimi-K3.1-Pro in the audit matrix`.

## Self-Review

- Spec coverage: A1 → Task 3; A2/A6/A7 → Task 2; A3/A4/A5 → Tasks 1–2. No
  finding is left without a task; findings already declared as debt (P1–P4) are
  not "fixed" here, only kept honest.
- Step scan: every step names the file, the exact text to replace or the exact
  command and its expected output.
- Type consistency: the matrix row uses the same 6 columns and the same verdict
  token (`views+ExpertStore`) that `kReadPaths` accepts.
- Review focus: the five lines above each map to a task step (report caveat,
  cost paragraph, remaining-lever paragraph, loader name, matrix + `kModels`).
- Proportion: three tasks, ~120 lines of plan for a doc unification plus one
  guard row.
