# Model Manifest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **This repository forbids subagents** (`AGENTS.md` § Autonomous Mode). Use
> `executing-plans` (native execution) only.

**Goal:** Make `models/<model_type>.json` the single source of truth for every
registered model, and derive the guard registries, ctest registration, parity
gates, model registry and documentation from it so they cannot drift.

**Architecture:** A Python generator (`tools/gen_model_manifest.py`) validates
the manifests at configure time and emits a C++ table
(`${CMAKE_BINARY_DIR}/generated/model_manifest.h`), a CMake fragment
(`${CMAKE_BINARY_DIR}/generated/model_targets.cmake`) and an include list for the
model registry. Guard tests and the gate drivers compile against the generated
table; `tests/CMakeLists.txt` `include()`s the fragment. Two generated markdown
files are committed and verified by the generator's `--check` mode, which runs as
a `fast` ctest.

**Ordering rule:** registration is switched to the manifest (Task 5) **while the
old per-family files still exist**, and only then are those files replaced by
drivers (Tasks 6–7). That keeps every commit green; doing it the other way leaves
the suite guard red between commits.

**Tech Stack:** C++20, Catch2 v3.16.1, CMake 3.25+ (`find_package(Python3)`),
Python 3 (generator only, build-time), git.

**Spec:** `docs/superpowers/specs/2026-11-06-model-manifest-design.md`

## Global Constraints

- No subagents, no parallel agents — every task is executed in the main session.
- C++20. Host logging only through spdlog / `lsxcommon::log` (no `std::cout`,
  no `printf`) — `test_logging_guard` scans `libs/` and `apps/`.
- Host-only C++ prefers `absl::` containers/strings; headers reachable from
  `.cu`/`.cuh` must not include `<absl/container/*>`.
- Every test target carries exactly one label (`fast`/`convert`/`inference`/`perf`)
  in the same commit, and every skip-capable target sets `SKIP_RETURN_CODE 4`
  (`test_ctest_labels_guard`).
- Generated artifacts are **never committed**, except `docs/MODELS.md` and the
  regenerated rows of `docs/SELFCHECK_MATRIX.md`, which are verified by
  `--check`.
- `models/<model_type>.json` filenames sort stably; `model_type` equals the
  model class's `kType` and the `LsxModelFactory` key.
- Windows: the preset must keep working. (The old `qwen|gemma` test restriction
  was REMOVED at the operator's request during execution — see the amendment at
  the end of this plan; do not emit a filter for it.)
- One GPU workload at a time: run the inference/label sweeps sequentially.
- Numeric behaviour of converters, kernels and models is **out of scope**: no
  golden, threshold or fast-math policy changes in this work.
- Existing guard assertions may move data source; they may not be weakened or
  deleted except where the defect class they police is removed with its subject
  (stated explicitly in that task).

## Review Focus

Failure modes this work can introduce that no single task's tests catch by
construction. Each is pinned by the step named in brackets.

1. **A test file that stops running.** A `tests/test_<family>_*.cpp` that no
   manifest claims is silently dropped from the build — the suite goes green
   while losing coverage. [Task 1 validation rule 8 + Task 2 third test case;
   re-checked in Tasks 6–7 after the deletions.]
2. **A manifest that names the wrong vault directory.** Its gate SKIPs forever
   and reads exactly like "golden not captured" — the defect class this design
   removes. [Task 1 validation rule 7; Task 6 Step 3 requires the SKIP message to
   contain both `model_type` and `vault_dir`.]
3. **A stage target registered with the wrong label.** Loaders are `fast` today
   and models are `inference`; getting it backwards either slows every quick
   build or hides a test from the inference sweep. [Task 1 `DEFAULT_LABEL` table
   is asserted by the guard in Task 5 Step 1, not inferred.]
4. **Transcription errors while authoring 35 manifests by hand.** A wrong target
   name or timeout changes suite behaviour with no visible diff. [Task 3 shadow
   mode diffs the parsed target list against the Task 0 oracle.]
5. **The generator itself becoming unchecked code.** A bug in the emitter
   propagates to every model at once. [Task 1 `--self-test` runs the generator
   against fixture manifests and asserts each schema error is reported.]

---

### Task 0: Branch and preflight oracle

**Files:**
- Create: `tools/dump_ctest_manifest.py`
- Create: `tests/fixtures/ctest/2026-11-06-baseline.json`

**Interfaces:**
- Produces: a JSON document `{"preset": str, "tests": [{"name": str,
  "command": str, "labels": [str], "timeout": int|null,
  "skip_return_code": int|null, "environment": [str],
  "resource_lock": str|null, "sources": [str]}]}` sorted by `name`. Every later
  task diffs against it.

- [ ] **Step 1: Create the working branch**

```bash
git checkout -b refactor/model-manifest
```
(The design commit `00a923f4` may already be on this branch; keep it.)

- [ ] **Step 2: Write `tools/dump_ctest_manifest.py`**

Reads `build/<preset>/CTestTestfile.cmake` and every
`build/<preset>/tests/CTestTestfile.cmake`, parses `add_test`,
`set_tests_properties` and `set_property(... PROPERTIES ...)` into the JSON shape
above. CLI: `--preset <name> --out <path>`. stdlib only (`json`, `re`,
`argparse`, `pathlib`); `main()` guard; diagnostics to stderr.

- [ ] **Step 3: Capture the baseline**

Run: `python3 tools/dump_ctest_manifest.py --preset linux-debug --out tests/fixtures/ctest/2026-11-06-baseline.json`
Expected: the file exists, `len(tests) == 510` (the current `ctest -N` total on
`build/linux-debug`), and `labels` is non-empty for every entry.

- [ ] **Step 4: Sanity-check the oracle itself**

Run: `python3 tools/dump_ctest_manifest.py --preset linux-debug --out /tmp/again.json && diff -q tests/fixtures/ctest/2026-11-06-baseline.json /tmp/again.json`
Expected: identical — an oracle that is not deterministic is worthless.

- [ ] **Step 5: Commit**

```bash
git add tools/dump_ctest_manifest.py tests/fixtures/ctest/2026-11-06-baseline.json
git commit -m "test: capture the ctest target oracle before the manifest migration"
```

---

### Task 1: Generator core, schema validation and `--self-test`

**Files:**
- Create: `tools/gen_model_manifest.py`
- Create: `tests/fixtures/manifest/valid/flint.json`
- Create: `tests/fixtures/manifest/invalid/{unknown_stage,missing_passport,ktype_mismatch,orphan_target,undeclared_gap,vault_env}.json`
- Test: the tool's `--self-test` mode (registered as a ctest in Task 5)

**Interfaces:**
- Produces:
  - `load_manifests(root: Path) -> list[Manifest]` and
    `validate(manifests, repo_root) -> list[str]`, raising/reporting
    `ManifestError(path, json_path, message)`.
  - CLI: `--header <out>`, `--cmake <out>`, `--registry <out>`,
    `--docs <out>`, `--check`, `--self-test <dir>`,
    `--diff-cmake <baseline.json>`, `--print-non-model-tests`, `--root <repo>`
    (default cwd).
  - Stage defaults, exactly:
    `convert → (convert, 120, skip=False)`, `load → (fast, 0, True)`,
    `model → (inference, 300, True)`, `kernels → (inference, 300, True)`,
    `selfcheck → (inference, 1800, True)`,
    `fsi|code|det → (inference, 3600, True)`, `perf → (perf, 0, True)`
    (`timeout 0` = emit none).
- Consumes: the schema in spec §8.1 and §8.5.

- [ ] **Step 1: Write the fixture manifests**

`tests/fixtures/manifest/valid/flint.json` is the row from spec §8.1 verbatim.
Each `invalid/*.json` is that object with exactly one mutation; `--self-test`
asserts the reported message contains the stated substring:

| fixture | mutation | message contains |
|---|---|---|
| `unknown_stage` | add `"mystery": ["test_x"]` to `stages` | `unknown stage` |
| `missing_passport` | `passport` → `libs/lsxcommon/src/nope.h` | `passport not found` |
| `ktype_mismatch` | `model_class` → `NotAFlintModel` | `model_class not found in src` |
| `orphan_target` | add `"test_does_not_exist"` to `stages.model` | `no source for target` |
| `undeclared_gap` | `stages.fsi = []`, no `golden_debt.fsi` | `gap without debt` |
| `vault_env` | `vault.env` → `LSX_TEST_WRONG_ROOT` | `vault.env must be LSX_TEST_FLINT_ROOT` |

- [ ] **Step 2: Run the self-test to verify it fails**

Run: `python3 tools/gen_model_manifest.py --self-test tests/fixtures/manifest --root .`
Expected: FAIL — `No such file or directory` (the tool does not exist yet).

- [ ] **Step 3: Implement `tools/gen_model_manifest.py`**

One module, stdlib only. Structure: `Manifest` dataclass (spec §8.1 fields +
`target_overrides`); `load_manifests` (json → dataclass → shape checks);
`validate` (filesystem checks); `emit_header` / `emit_cmake` / `emit_registry` /
`emit_docs`; `check` (emit to memory, compare with the committed docs, print a
unified diff, exit 1); `diff_cmake` (project to the Task 0 shape and diff);
`self_test`. Every message begins with the manifest filename and JSON path so a
35-row authoring pass is debuggable.

Validation rules — each with its own message:

1. exactly the nine stage keys, no others;
2. every stage list non-empty **or** a `golden_debt`/`debt` reason present;
3. `model_type` unique and equal to the filename stem;
4. `passport` exists **and** declares `namespace <ns>` matching `ns`;
5. `model_class` is declared in `libs/lsxcommon/src/<model_header>` and its
   `kType` literal equals `model_type`;
6. every stage target resolves to `tests/<target>.cpp`/`.cu` or is described in
   `target_overrides`;
7. `vault.env == "LSX_TEST_" + upper(model_type with [.-] → _) + "_ROOT"`;
8. every `tests/test_*.cpp|.cu` file is claimed by some manifest target or listed
   in `NON_MODEL_TESTS` (a frozen allowlist in the tool, one entry per
   infrastructure test);
9. `family` identical across rows sharing it; `variant` null or unique.

- [ ] **Step 4: Run the self-test to verify it passes**

Run: `python3 tools/gen_model_manifest.py --self-test tests/fixtures/manifest --root .`
Expected: PASS and it prints `6/6 schema violations reported`.

- [ ] **Step 5: Commit**

```bash
git add tools/gen_model_manifest.py tests/fixtures/manifest
git commit -m "feat: add the model manifest generator with schema self-tests"
```

---

### Task 2: Configure-time wiring and the sync test

**Files:**
- Create: `cmake/LsxModelManifest.cmake`
- Create: `tests/test_manifest_sync.cpp`
- Create: `models/.gitkeep`
- Modify: `CMakeLists.txt` (include the module after `cmake/LsxTestHelpers.cmake`)
- Modify: `tests/CMakeLists.txt` (one `include()` + one `lsx_add_test` block)

**Interfaces:**
- Produces:
  - `lsx_configure_model_manifest()` — runs the generator into
    `${CMAKE_BINARY_DIR}/generated/`, sets `LSX_MANIFEST_MODELS` (list) and
    `LSX_MANIFEST_GENERATED_DIR`, and `message(FATAL_ERROR)`s with the
    generator's stderr on a non-zero exit.
  - CMake variable `LSX_PYTHON3`.
  - `test_manifest_sync` — `fast`-labelled; reads `LSX_GENERATED_DIR` and
    `LSX_SOURCE_DIR`.
- Consumes: Task 1's CLI.

- [ ] **Step 1: Write the failing test**

`tests/test_manifest_sync.cpp`, three cases:

1. `("manifest sync: generated table matches the manifests", "[manifest][fast]")`
   — do **not** re-implement JSON parsing in C++. Ask the generator for its own
   canonical projection (`--print-model-rows`: one tab-separated line per model,
   fields in the `ModelRow` order) and compare it with what the header scanner
   finds in `kModels[]`. One JSON implementation, two views of it — a difference
   means the header is stale or the emitter is wrong.
2. `("manifest sync: --check reports no drift")` — `std::system` the generator
   with `--check` (path from `LSX_PYTHON3`), `INFO` its stderr, `REQUIRE(rc == 0)`.
3. `("manifest sync: no unclaimed model test file")` — list `tests/test_*.cpp`
   and `*.cu`, assert each is claimed by the generated `kTargets` table or listed
   by `--print-non-model-tests`.

- [ ] **Step 2: Run it to verify it fails**

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug && ctest --preset linux-debug -R manifest_sync`
Expected: configure FAILS (the generated directory does not exist yet) — the
failing state for this task.

- [ ] **Step 3: Implement the module and the header emitter**

The header emitter writes:

```cpp
// generated by tools/gen_model_manifest.py -- do not edit
#pragma once
#include <cstddef>
namespace lsxmanifest {
struct ModelRow { const char *model_type; const char *family; const char *variant;
                  const char *model_class; const char *model_header;
                  const char *passport; const char *ns; const char *format_tag;
                  const char *vault_dir; const char *vault_env; int tokenizer_kind;
                  bool tool_roundtrip; bool auto_path_determinism; };
struct TargetRow { const char *model_type; const char *stage; const char *target;
                   const char *source; const char *label; int timeout; bool skip;
                   const char *env; const char *resource_lock; };
struct LifecycleRow { const char *model_type; const char *tu; };
inline constexpr ModelRow kModels[] = { /* one per manifest, sorted */ };
inline constexpr TargetRow kTargets[] = { /* one per stage target, sorted */ };
inline constexpr LifecycleRow kLifecycle[] = { /* one per lifecycle TU */ };
inline constexpr std::size_t kModelCount = /* … */;
}
```

Strings become escaped C++ literals; a null JSON field becomes `nullptr`; an
absent `timeout` becomes `0`.

In `tests/CMakeLists.txt`, after the helper section:

```cmake
include("${CMAKE_BINARY_DIR}/generated/model_targets.cmake")   # filled in Task 5
lsx_add_test(NAME test_manifest_sync
    SOURCES test_manifest_sync.cpp
    LINK lsxcommon
    DEFINE "LSX_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\""
           "LSX_GENERATED_DIR=\"${LSX_MANIFEST_GENERATED_DIR}\""
           "LSX_PYTHON3=\"${LSX_PYTHON3}\""
    LABEL fast)
```

In `cmake/LsxModelManifest.cmake`: `find_package(Python3 3.8 COMPONENTS
Interpreter REQUIRED)` (build-time only), then
`execute_process(COMMAND ${Python3_EXECUTABLE} tools/gen_model_manifest.py
--root ${PROJECT_SOURCE_DIR} --header … --cmake … --registry …)` with
`RESULT_VARIABLE` / `ERROR_VARIABLE` checked. The generated fragment file is
created **empty** in this task so the `include()` resolves; Task 5 fills it.

- [ ] **Step 4: Run it to verify it passes**

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_manifest_sync && ctest --preset linux-debug -R manifest_sync --output-on-failure`
Expected: PASS with three results. `models/` is empty here, so the count
assertions are trivially true until Task 3 — that is intended (scaffolding
first, data next).

- [ ] **Step 5: Commit**

```bash
git add cmake/LsxModelManifest.cmake tests/test_manifest_sync.cpp CMakeLists.txt tests/CMakeLists.txt models/.gitkeep
git commit -m "build: generate the model manifest table at configure time"
```

---

### Task 3: Author the 35 manifests and prove the transcription (shadow mode)

**Files:**
- Create: `models/<model_type>.json` for all 35 registered models (three commits)
- Create: `docs/MODELS.md` (generated, committed)
- Modify: `tools/gen_model_manifest.py` (`--docs` sections + the 15-line budget)

**Interfaces:**
- Consumes: `LsxModelFactory::Register<T>()` in
  `libs/lsxcommon/src/model_registry.cpp`; `kPassports` / `kLifecycle` in
  `tests/test_model_independence_guard.cpp`; `Suites()` / `kGoldenDebt` in
  `tests/test_model_suite_guard.cpp`; the per-target blocks in
  `tests/CMakeLists.txt`; `tests/perf/perf_baselines.json` for `baseline_key`.
- Produces: the complete `models/` tree and `docs/MODELS.md`.

- [ ] **Step 1: Run shadow mode and confirm it fails**

Run: `python3 tools/gen_model_manifest.py --diff-cmake tests/fixtures/ctest/2026-11-06-baseline.json --root .`
Expected: FAIL with `0 of 35 manifests found`.

- [ ] **Step 2: Author cluster 1 (glm, qwen, deepseek)**

`glm-5.2`, `glm-5.3-flash`, `deepseek-v4-flash`, `deepseek-v4.1-flash`,
`deepseek-v4-pro-0813`, `qwen3.5-9b`, `qwen3.5-35b-a3b`, `qwen3.6-35b-a3b`,
`qwen3.8-27b`, `qwen3.8-flash-next`. Copy every value from the four registries
named in Interfaces — invent nothing. Where a `Suites()` row lists several
targets for one stage, list them all. `tool_roundtrip` is `true` for a model
whose selfcheck file calls `RunToolRoundTrip`; `auto_path_determinism` is `true`
for one whose file has the kAuto case (grep the file — do not guess).

- [ ] **Step 3: Verify cluster 1 against the oracle**

Run: `python3 tools/gen_model_manifest.py --diff-cmake tests/fixtures/ctest/2026-11-06-baseline.json --root . 2>&1 | grep -E '^[-+][^-+]' | head -40`
Expected: no differences for this cluster; the other 25 models are reported as
`no manifest`, which is expected here.

- [ ] **Step 4: Author cluster 2 (gemma, kimi, nemotron, gigachat, ornith, granite, phi, mimo)**

`gemma-4-26b-a4b-it`, `gemma-4-31b-it`, `kimi-k2.7-code`, `kimi-k3`,
`kimi-k3.1-pro`, `nemotron-3.5-lightning-30b-a3b`,
`nemotron-3.5-lightning-30b-a3b-nvfp4`, `gigachat3.5-432b-a28b-reasoning`,
`ornith-1.5-35b-a3b`, `granite-4.2-30b`, `granite-switch-4.1-30b`, `phi-4`,
`mimo-v2.6-flash-rl`. `kimi-k3` / `kimi-k3.1-pro` share `"family": "kimi"` with
`kimi-k2.7-code` and differ by `variant`.

- [ ] **Step 5: Verify cluster 2, then author cluster 3**

Same command; then `alice-foundation-80b-a3b`, `xing4.0-29b-a4b`,
`mistral-small-4-119b`, `mistral-large-4-1t`, `hy-memory-30b-a3b`,
`hy-simulmt-30b-a3b`, `flint`, `ling`, `beam`, `step-5-preview`, `minimax-m3`,
`minimax-m3.1-flash`.

- [ ] **Step 6: Verify the whole tree**

Run: `python3 tools/gen_model_manifest.py --diff-cmake tests/fixtures/ctest/2026-11-06-baseline.json --root .`
Expected: `0 differences across 35 models`, exit 0.

- [ ] **Step 7: Generate the docs and run the sync test**

Run: `python3 tools/gen_model_manifest.py --docs docs/MODELS.md --root . && cmake --preset linux-debug && cmake --build --preset linux-debug --target test_manifest_sync && ctest --preset linux-debug -R manifest_sync --output-on-failure`
Expected: PASS. `docs/MODELS.md` has one `## <model_type>` section per model, at
most 15 lines each (enforced by the sync test).

- [ ] **Step 8: Commit (three commits, one per cluster)**

```bash
git add models/ docs/MODELS.md
git commit -m "feat: add manifests for the glm/qwen/deepseek families"
# … gemma/kimi/nemotron/gigachat/ornith/granite/phi/mimo
# … alice/xing/mistral/hy/flint/ling/beam/step/minimax
```

---

### Task 4: Guard accessors and the generated-table port

**Files:**
- Create: `tests/test_manifest_query.h`
- Modify: `tests/test_model_independence_guard.cpp`,
  `tests/test_model_suite_guard.cpp`
- Modify: `tests/CMakeLists.txt` (both guards get the generated dir on their
  include path)

**Interfaces:**
- Produces (host-only header, `namespace manifest_query`):
  - `const lsxmanifest::ModelRow *FindModel(std::string_view type);`
  - `std::vector<const lsxmanifest::TargetRow *> TargetsOf(
     std::string_view model, std::optional<std::string_view> stage = std::nullopt);`
  - `bool HasGolden(const ModelRow &row, std::string_view stage);` — true when
    `tests/fixtures/<stage>/<model_type>.json` exists; used by the ported
    "a golden that is missing and not declared as debt fails" case.
  - `std::string ModelRoot(const ModelRow &row);` — `getenv(row.vault_env)` else
    `lsx_model_models_dir() / row.vault_dir`.
  - `std::string SkipMessage(const ModelRow &row, std::string_view what);` —
    `"<model_type> <what> not present in <vault_dir>"`.
- Consumes: `#include "model_manifest.h"` (Task 2).

- [ ] **Step 1: Write the failing assertions first**

In `test_model_suite_guard.cpp` add two cases:
`("suite registry covers every family's tests")` — for every
`lsxmanifest::ModelRow`, `TargetsOf(type, "selfcheck"|"fsi"|"code"|"det")` is
non-empty or the row declares debt; and
`("every model family is known to the cross-family check")` — the set of
`family` values in `kModels` equals the set the borrowing scan uses (**this is
the test that would have caught `FamilyStems()` missing `beam`, `hy`, `minimax`,
`mistral`, `xing`**).

Run: `ctest --preset linux-debug -R model_suite_guard --output-on-failure`
Expected: FAIL — the generated table is not included yet.

- [ ] **Step 2: Implement the accessors and port both guards**

Mechanical port; keep every existing assertion, replace only the data source.
Delete `kPassports`, `kLifecycle`, `Suites()`, `kGoldenDebt` and the
`FamilyStems()` literal. Where the old code special-cased a model (the
`deepseek-v4-flash` "no convert stage by design" comment), the reason moves into
that manifest's `notes`, and the guard prints `notes` when a stage is empty.

- [ ] **Step 3: Verify, including the five newly covered families**

Run: `ctest --preset linux-debug -R "model_suite_guard|model_independence_guard|code_naming_guard|ctest_labels_guard" --output-on-failure`
Expected: PASS, with the borrowing scan covering 20 families instead of 15. If a
newly covered family reports a real cross-family borrow, **fix the borrow** (do
not add an allowlist entry), or record it in `docs/OPEN_TASKS.md` and say so in
the commit message.

- [ ] **Step 4: Commit**

```bash
git add tests/test_manifest_query.h tests/test_model_independence_guard.cpp tests/test_model_suite_guard.cpp tests/CMakeLists.txt
git commit -m "refactor: read the guard registries from the generated manifest table"
```

---

### Task 5: CTest registration from the manifest

**Files:**
- Modify: `cmake/LsxTestHelpers.cmake` (`RESOURCE_LOCK` option)
- Modify: `cmake/LsxModelManifest.cmake` (emit `model_targets.cmake`)
- Modify: `tests/CMakeLists.txt` (delete hand blocks for manifest-claimed
  targets; keep `include()`)
- Modify: `tests/test_ctest_labels_guard.cpp` (recognize `lsx_add_test`)

**Interfaces:**
- Produces: `${CMAKE_BINARY_DIR}/generated/model_targets.cmake` — one
  `lsx_add_test(...)` call per manifest target plus
  `set_tests_properties(<t> PROPERTIES RESOURCE_LOCK "gpu")`.
- Consumes: the Task 0 oracle and Task 1's `DEFAULT_*` tables.

- [ ] **Step 1: Extend `lsx_add_test`, and teach the labels guard to see it**

First the failing guard case: `test_ctest_labels_guard.cpp` scans CMake text with
regexes for `set_tests_properties(... LABELS x)`; a `lsx_add_test(NAME t … LABEL
fast)` call is invisible to it, so every generated target would look unlabelled.
Add a fixture (the file already has inline CMake fixtures at the bottom) with one
`lsx_add_test(... LABEL convert SKIP TIMEOUT 120)` call and assert the guard
accepts it; run `ctest --preset linux-debug -R ctest_labels_guard` and watch it
FAIL. Then add the recognizer
`lsx_add_test\s*\(([^)]*?)\bLABEL\s+(\w+)` plus extraction of a bare `SKIP`
keyword and of `TIMEOUT <n>` from the same call, feeding the existing checks
("every target has a label", "skip-capable target sets 4", "convert target has a
bounded TIMEOUT"). Extend the guard's file list to include
`${LSX_MANIFEST_GENERATED_DIR}/model_targets.cmake` when present, and make the
guard fail with "run cmake configure first" if the generated fragment is missing
while `models/` is non-empty.

Then add `RESOURCE_LOCK` to `cmake_parse_arguments` (`"NAME;LABEL;TIMEOUT;RESOURCE_LOCK"`)
and `set_tests_properties(${name} PROPERTIES RESOURCE_LOCK ${P_RESOURCE_LOCK})`
when set.

- [ ] **Step 2: Emit the fragment**

Templates per kind, with values from the manifest:

- plain target → `lsx_add_test(NAME … SOURCES … LINK lsxcommon INCLUDE
  "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src" DEFINE
  "LSX_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\"" LABEL … TIMEOUT … [SKIP]
  [ENV "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1"])` plus `RESOURCE_LOCK "gpu"`
  except for `load`;
- `kernels` with a `.cu` source → `lsx_add_cuda_test(OBJECT …_cu SOURCES … LABEL
  inference TIMEOUT 300 SKIP)`;
- `fsi|code|det` → the parity driver sources are **not** switched yet (Task 6);
  emit the current per-family source so this commit changes only *who writes the
  registration*;
- `selfcheck` keeps its current per-family source until Task 7.

A model target that needs a define, env or timeout beyond the defaults is
expressed in `target_overrides`; no other escape path exists.

~~The fragment wraps the model loop in the platform filter~~ — **do not do this.**
The Windows `qwen|gemma` restriction was removed during execution (operator
request), and the mechanism behind it (`test_router.cmake`'s `if(WIN32)` branch)
is deleted. Every platform now runs one label-filtered sweep. See the amendment
at the end of this plan.

- [ ] **Step 3: Delete the hand blocks and diff against the oracle**

Remove every hand-written block for a manifest-claimed target. Run
`python3 tools/dump_ctest_manifest.py --preset linux-debug --out /tmp/p5.json`
and compare with the baseline (same script as Task 0 Step 4, plus a normaliser
that sorts `environment`/`sources`).
Expected: zero differences. Investigate any difference individually — this diff
is the whole point of the task.

- [ ] **Step 4: Run the fast suite**

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug && ctest --preset linux-debug --output-on-failure`
Expected: green, including `manifest_sync`, `model_suite_guard` and
`ctest_labels_guard`.

- [ ] **Step 5: Commit**

```bash
git add -A cmake tests
git commit -m "build: register every model test target from its manifest"
```

---

### Task 6: Parity gate driver (FSI / CODE / DET)

**Files:**
- Create: `tests/test_parity_gate.cpp`
- Delete: 105 files — `tests/test_*_{fsi,code,det}_selfcheck.cpp`
- Delete: `tests/test_parity_selfcheck_naming_guard.cpp` and its registration
- Modify: `tools/gen_model_manifest.py` (gate targets now use the driver)

**Interfaces:**
- Consumes: `lsxcommon::fsi::LoadFsiGoldenFile` / `RunFsiSelfCheck`,
  `code::LoadCodeGoldenFile` / `RunCodeSelfCheck`,
  `det::LoadDetGoldenFile` / `RunDetSelfCheck` (existing signatures, unchanged);
  `manifest_query::FindModel` / `ModelRoot` / `SkipMessage` (Task 4).
- Produces: one executable per (model, kind), registered by Task 5's generator.

- [ ] **Step 1: Write the driver**

`tests/test_parity_gate.cpp` starts with compile-time guards:

```cpp
#ifndef LSX_GATE_MODEL
#error "LSX_GATE_MODEL must be defined"
#endif
#ifndef LSX_GATE_KIND
#error "LSX_GATE_KIND must be defined (fsi|code|det)"
#endif
```

`constexpr std::string_view kModel = LSX_GATE_MODEL; constexpr std::string_view
kKind = LSX_GATE_KIND;` → `FindModel(kModel)` (a null row is a hard
`REQUIRE` failure: a manifest-less target is a bug, not a skip). `ModelRoot(row)`
and the golden path `${LSX_<KIND>_GOLDEN_DIR}/<model_type>.json`. SKIPs use
`SkipMessage(row, "converted model")` and `"<model_type> golden not captured for
<model_type>"` so the message can no longer name a neighbour. Then `if constexpr`
dispatch on `kKind` to the matching `Load*`/`Run*` pair, with every case's
`expected`/`extracted`/`error` in an `INFO`, and `REQUIRE(result.all_passed)`
followed by `model->Unload()`.

- [ ] **Step 2: Point the generator at the driver for one model, and verify**

Add the gate-source switch behind a `GATE_DRIVER_MODELS` allowlist in the tool
(initially `{"flint"}`), regenerate, rebuild and run:
`ctest --preset linux-debug -R test_flint_moe_fsi_selfcheck -V`
Expected: the same CTest outcome and the same skip/pass text as before (both
skip — the flint golden is declared debt).

- [ ] **Step 3: Verify one model with a real golden**

Run (GPU exclusive): `ctest --preset linux-release-znver4 -R test_granite_dense_fsi_selfcheck -V`
Expected: `3/3` cases pass exactly as the old binary did, or SKIP with a message
containing `granite-4.2-30b` and the manifest's `vault_dir`.

- [ ] **Step 4: Switch every model, delete the 105 files and the naming guard**

Add all models to `GATE_DRIVER_MODELS`, regenerate, then
`git rm tests/test_*_fsi_selfcheck.cpp tests/test_*_code_selfcheck.cpp
tests/test_*_det_selfcheck.cpp tests/test_parity_selfcheck_naming_guard.cpp`.
The naming guard dies with its subject: once one driver serves every model, a
copy cannot carry a neighbour's identity, and `kKnownOffenders` must not outlive
the defect it recorded.

- [ ] **Step 5: Verify target coverage is unchanged**

Run: `python3 tools/dump_ctest_manifest.py --preset linux-debug --out /tmp/p6.json` and diff against the baseline, ignoring the `sources` field only.
Expected: identical target names, labels, timeouts, skip codes and locks; the
only removed entry is the naming guard. Then
`ctest --preset linux-debug -R "manifest_sync|model_suite_guard"`.

- [ ] **Step 6: Commit**

```bash
git add -A tests tools
git commit -m "refactor: one parity-gate driver replaces 105 copied gate files"
```

---

### Task 7: SelfCheck driver with declared hooks

**Files:**
- Create: `tests/test_selfcheck_driver.cpp`
- Delete: the `tests/test_*_selfcheck.cpp` files whose cases are all covered by a
  hook (24 candidates; keep any file with a family-specific case)
- Modify: `models/*.json` (`tool_roundtrip`, `auto_path_determinism`)
- Modify: `tools/gen_model_manifest.py` (selfcheck source switch)

**Interfaces:**
- Consumes: `IModel::Infer` / `IModel::SelfCheck`, `tests/test_tool_roundtrip.h`
  (`RunToolRoundTrip(model, 512)`), the manifest row.
- Produces: hook ids `paris_cycles`, `tool_roundtrip`, `auto_path_determinism`,
  each implemented once in the driver and enabled per model by a manifest field.

- [ ] **Step 1: Record the per-model case inventory before touching anything**

For each of the 24 non-parity selfcheck files, record every `TEST_CASE` name in
the commit message as a table. This is the coverage baseline: the driver plus the
retained family files must reproduce it case for case.

- [ ] **Step 2: Classify, and keep the exceptions as their own targets**

Only three case shapes are uniform and move into the driver: `paris_cycles`
(Paris + >=3 bit-exact `final_logits` cycles, count from
`LSX_TEST_<MODEL>_SELFCHECK_CYCLES` floored at 3), `tool_roundtrip`
(`RunToolRoundTrip(*model, 512)`), `auto_path_determinism` (kAuto generate →
unload → load → compare `gen_tokens` and `final_logits`). Any file with a case not
on that list keeps its target and stays listed in its manifest's `selfcheck`
stage — that is how `test_qwen_moe_exp_eos_selfcheck` and the flash "3 bit-exact
cycles" cases survive.

- [ ] **Step 3: Write the driver**

One file, `TEST_CASE`s selected by `LSX_SELFCHECK_MODEL`; each hook is guarded by
`if (!row->…){ SUCCEED("hook not declared for this model"); return; }` so opting
out needs no new file. Reuse `ModelRoot` / `SkipMessage` from Task 4 — do not
duplicate them.

- [ ] **Step 4: Verify hook by hook on real weights**

Run (GPU exclusive, sequentially): `ctest --preset linux-release-znver4 -R
test_flint_selfcheck -V` then `-R test_granite_selfcheck -V`.
Expected: `paris_cycles` and `tool_roundtrip` pass exactly as the old binaries
did; `auto_path_determinism` passes for flint (which had the case) and reports
"hook not declared" for granite (which did not).

- [ ] **Step 5: Delete covered files and re-verify target coverage**

Repeat Task 6 Step 5, extended to `test_*_selfcheck`. Expected: every removed
file's target still exists (now driven from the manifest); executables drop,
targets do not.

- [ ] **Step 6: Commit**

```bash
git add -A tests models tools
git commit -m "refactor: selfcheck cases become declared hooks on one driver"
```

---

### Task 8: Model registry from the manifest

**Files:**
- Modify: `tools/gen_model_manifest.py` (emit `model_registry.inc`)
- Modify: `libs/lsxcommon/src/model_registry.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt` (generated include dir)
- Modify: `tests/test_manifest_sync.cpp` (registry coverage case)

**Interfaces:**
- Produces: `${CMAKE_BINARY_DIR}/generated/model_registry.inc` —
  `#include "<model_header>"` then `LsxModelFactory::Register<<model_class>>();`
  per manifest, sorted by `model_type`.
- Consumes: `ModelRow.model_class`, `.model_header`.

- [ ] **Step 1: Write the failing case first**

Add `("manifest sync: registry covers every model")` to
`tests/test_manifest_sync.cpp`: for each `ModelRow`, `LsxModelFactory::Create`
with `cfg.model_type = row.model_type` and a non-existent `model_path` returns
non-null (constructing must not need the weights). Run it — it fails for the
alias entry that exists only in the hand-written `model_registry.cpp` today.
Decide that alias's home (a `aliases` field on the owner's manifest) before
implementing, and note the choice in the commit message.

- [ ] **Step 2: Emit and include**

Include the `.inc` inside `EnsureModelRegistry()`'s lambda; delete the 35 hand
`Register<T>` lines and the 35 hand `#include` lines.

- [ ] **Step 3: Verify every registered type resolves**

Run: `cmake --build --preset linux-debug && ctest --preset linux-debug -R manifest_sync --output-on-failure`
Expected: PASS — 35 types plus the alias resolve.

- [ ] **Step 4: Commit**

```bash
git add tools/gen_model_manifest.py libs/lsxcommon/src/model_registry.cpp libs/lsxcommon/CMakeLists.txt tests/test_manifest_sync.cpp
git commit -m "refactor: generate the model registry from the manifests"
```

---

### Task 9: Documentation diet

**Files:**
- Modify: `AGENTS.md` (1109 → ≤ 200 lines)
- Modify: `README.md` (§ Structure → one pointer)
- Modify: `docs/SELFCHECK_MATRIX.md` (table rows generated; narrative kept)
- Modify: `tools/gen_model_manifest.py` (emit matrix rows; `--check` covers them)
- Create: `docs/superpowers/plans/2026-11-06-model-manifest-routing.md`

**Interfaces:**
- Consumes: spec §8.4 (what survives in `AGENTS.md`) and §8.6 (where everything
  else goes).
- Produces: the routing table — one row per paragraph of the old `## Structure`,
  with its destination, committed as evidence.

- [ ] **Step 1: Build the routing table before deleting anything**

Classify every paragraph of the current `## Structure` into exactly one
destination: invariant → `AGENTS.md`; per-model fact → `docs/MODELS.md`;
measured result → stays in `perf_baselines.json` / `SELFCHECK_MATRIX.md` and is
linked; rationale/history → `docs/model-reports/` or `docs/open-tasks/` plus a
pointer. Any paragraph you cannot classify is a real gap — resolve it first.

- [ ] **Step 2: Move the invariants out**

Extract every rule from `## Structure` that a future agent could violate —
examples: the only GEMM path is `lsx_gemm`; no CLI speed flags; MMAP-only Arrow
reads; `FastMathMode::kOff` where a family pins it because the TC path overflows;
the `kLifecycle` contract each family must wire. These are not generated; they
are prose rules and they go into the numbered sections.

- [ ] **Step 3: Write the new `AGENTS.md`**

≤ 200 lines: Project + invariants, Dependencies (table unchanged), a pointers
table (build → `building-the-project`; tests → `test-lifecycle`; models →
`docs/MODELS.md`; code map → `logestix-repo-map`; open work →
`docs/OPEN_TASKS.md`; gates → `docs/SELFCHECK_MATRIX.md`), the existing skill
map, and the cross-cutting rules (absl, logging, autonomous mode, git, NVIDIA
docs). Replace the stale "`fast` is green (86/86)" with the number measured by
`ctest --preset linux-debug -N`.

- [ ] **Step 4: Verify nothing was lost**

Run: `grep -c "" AGENTS.md` (≤ 200). For every routing-table row, confirm the
destination exists and holds the content; spot-check the five riskiest: the
driver/runtime skew paragraph, the `kLifecycle` contract, the expert-store
staging-ring note, the `--bit-exact` rule and the Windows test restriction.
Run: `ctest --preset linux-debug -R "manifest_sync|logging_guard|absl_format_guard"`
Expected: PASS (`--check` regenerates `docs/MODELS.md` and the matrix rows and
finds no drift).

- [ ] **Step 5: Commit**

```bash
git add AGENTS.md README.md docs/ tools/
git commit -m "docs: turn AGENTS.md into a router and generate the model catalogue"
```

---

### Task 10: Final verification sweep

**Files:**
- Modify: `.opencode/skills/ingesting-a-model/SKILL.md` (stage 8: the registries
  are generated — the manifest is the only hand step)
- Modify: `.opencode/skills/logestix-repo-map/SKILL.md` (point at `docs/MODELS.md`)
- Modify: `docs/OPEN_TASKS.md` (record what this migration does not cover)

- [ ] **Step 1: Update the two skills this migration invalidates**

`ingesting-a-model` currently tells the reader to extend the guard registries by
hand, and `logestix-repo-map` calls them the source of truth. Both must name the
manifest instead, in the same commit — otherwise the next ingest is done the old
way and the migration decays.

- [ ] **Step 2: Negative test on the finished system**

Temporarily rename one stage target in a real manifest to a non-existent name and
run `ctest --preset linux-debug -R manifest_sync`.
Expected: FAIL naming the manifest file and JSON path. Revert and confirm PASS.
If a real mutation cannot fail the suite, the registry is decorative.

- [ ] **Step 3: Sweep, in order (one GPU workload at a time)**

```bash
ctest --preset linux-debug                                  # fast
ctest --preset linux-debug-convert                          # convert
ctest --preset linux-debug-inference -R "flint|granite"     # smoke
```
Expected: fast and convert green; the two smoke models match their pre-migration
outcomes from the Task 0 baseline.

- [ ] **Step 4: Commit and record the gaps**

```bash
git add .opencode/skills docs/OPEN_TASKS.md
git commit -m "docs: point the skills at the manifest; record the migration's gaps"
```

---

## Out of scope (deliberately)

- Converter, kernel, loader, pipeline and tool-codec code per model.
- Golden capture and perf seeding (the manifests only *name* the artifacts).
- Approach C (family scaffolder): decided after this plan lands, against the
  measure "how many lines a fresh family needs that are not converter/kernel
  logic".

---

## Amendment (2026-11-06, during execution)

**The Windows `qwen|gemma` test restriction was removed at the operator's
request.** Consequences for this plan:

- Task 5 Step 2 no longer emits a `WIN32` name filter into the generated
  fragment (struck out above). No platform conditional exists in
  `model_targets.cmake`.
- `test_router.cmake` lost its `if(WIN32)` branch. That branch was the only
  MECHANISM: on Windows it narrowed the `convert` and `inference` sweeps with
  `-R "^(test_qwen|test_gemma)"`, so those sweeps covered two families on that OS
  while Linux covered all of them.
- `AGENTS.md` and `.opencode/skills/test-lifecycle/SKILL.md` no longer state the
  restriction.

Other amendments made during execution (each is also in the commits):

- **Task 5 scope:** the generator emits *registration* (`add_test` +
  `set_tests_properties`) only. Build wiring (`add_executable`, the CUDA OBJECT
  libraries, per-target includes/defines) stays hand-written, because only 10 of
  421 targets follow a convention and the rest is per-target compile reality.
- **Task 6/7:** the parity-gate driver and the selfcheck driver also own the
  `selfcheck` stage, not just FSI/CODE/DET, because the vault directory and the
  env var were the last per-family hardcodings.
- **Task 3:** `format_tag` is filled only where a machine-readable source exists
  (8 of 35); the rest print "not declared" in the catalogue rather than a guess.
- **Task 9:** `docs/SELFCHECK_MATRIX.md` is checked, not generated — its cells are
  measured verdicts.
- **Task 9 / Task 3 (catalogue contents):** `docs/MODELS.md` does **not** restate
  per-model geometry. The design's "geometry one-liner" was withdrawn by the
  operator after measuring it: four architecture classes need different fields
  (GQA+dense, GQA+MoE, MLA+MoE, KDA/GDN hybrids), the naming styles differ across
  passports, and a wide synonym table maps a latent rank (`kKvLora = 512`) to a
  head count. A plausible wrong number in a catalogue is worse than no line, and
  the passport is already the machine-checked source (constexpr under
  `static_assert`). The catalogue links the passport; numbers live there or in
  the model report.
- **Task 1:** `--self-test` mutates a LIVE manifest (a checked-in fixture went
  stale the moment the per-family gate files were deleted) and is registered as
  `test_manifest_generator_selftest`.
