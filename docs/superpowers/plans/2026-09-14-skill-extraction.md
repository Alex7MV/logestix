# Skill extraction from AGENTS.md — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **NOTE (repo policy):** `AGENTS.md` forbids subagents and requires an explicit
> operator request before any `git commit`. Execute inline; every commit step is
> gated on operator approval.

**Goal:** Extract `test-lifecycle`, `building-the-project` and `coding-conventions` skills from AGENTS.md, fold the host-resource policy into `running-a-model`, reduce AGENTS.md to anchors, and move the orphaned Qwen3.8-Flash-Next bullet into the repo map.

**Architecture:** Documentation-only. Three new `SKILL.md` files under `.opencode/skills/`, edits to `running-a-model`, `logestix-repo-map` and `AGENTS.md`. Each new skill owns its topic; specialized slices stay in their existing skill and are cross-referenced.

**Tech Stack:** Markdown skill files; the logestix tree / `CMakePresets.json` / `tools/test_router.cmake` as the source of truth.

## Global Constraints

- Docs only: no code, no build, no GPU/inference run.
- No subagents; do not commit unless the operator explicitly asks.
- Every documented command/path/rule MUST exist in the tree.
- No `TBD`/`TODO`/`FIXME` in any skill.
- YAML frontmatter `name:` + trigger-bearing `description:` on every new skill.
- Cross-reference, never duplicate: converter test rules stay in `ingesting-a-model`; CMake diagnosis in `cuda-cmake-build-gotchas`; absl policy in `absl-in-cuda-cpp`; kernel rules in `nvfp4-blackwell-conventions`.

**Verified facts for all tasks:**

- Presets (`CMakePresets.json`): configure `linux-debug`, `linux-release`, `linux-release-znver4`, `windows-debug`, `windows-release`; test presets add `-fast`/`-convert`/`-inference`/`-perf` (e.g. `linux-debug-convert`, `linux-release-znver4-perf`); `linux-debug` test preset filters `^fast$`. The AOCC tree is a hand-configured `build/linux-release-znver4-aocc`.
- Router rules (`tools/test_router.cmake`): `converters→convert`, `quant_engine→convert inference`, `cuda→inference`, `lsxcommon_host→fast`, `lsxcommon→inference`, `network→fast`, `apps→fast`, `tests→convert inference`, `tooling→fast`, `default→fast`; FIRST match wins.
- Labels (AGENTS snapshot): `fast` 45, `convert` 13, `inference` 85, `perf` 15.
- Windows: `cmake`/`ctest`/`ninja` under VS 2026 `18\Community`; MSVC `14.51.36231`; NVCC 13.3; `windows-debug` uses the WIN32 generator; `arrow.dll` copied via POST_BUILD.
- Toolchain minimums: CMake 3.25+, GCC 15.2, NVCC 13.4.59 (Linux) / 13.3 (Windows), CUDA ≥13.2, C++20.
- Binaries: `build/<preset>/apps/<app>/<app>`.

---

### Task 1: Create `.opencode/skills/test-lifecycle/SKILL.md`

**Files:**
- Create: `.opencode/skills/test-lifecycle/SKILL.md`

**Interfaces:**
- Produces: skill `name: test-lifecycle`, referenced by AGENTS.md (§Test pointer) in Task 4 and by `ingesting-a-model`/`perf-tuning-a-model` where relevant.

- [ ] **Step 1: Write frontmatter + overview + categories**

Frontmatter (verbatim):
```yaml
---
name: test-lifecycle
description: Use when adding or running logestix tests — the fast/convert/inference/perf CTest labels, the test_router.cmake path rules and LSX_PRESET/LSX_LIST/LSX_DIFF, label discipline and the guard-test registry, preset commands, SKIP_RETURN_CODE 4 and converter TIMEOUTs. Triggers - add test, label, ctest, test router, guard test, convert timeout.
---
```
`## Overview` (why labels exist), `## Categories` table with the four labels, when each runs, and the measured counts (label them "Linux snapshot 2026-09").

- [ ] **Step 2: Add label discipline + guard registry**

`## Label discipline` — a new test MUST get its label in the same commit, else `test_ctest_labels_guard` is red. `## Guard tests` table: `test_code_naming_guard` (naming scheme), `test_model_independence_guard` (`kPassports`, consumers), `test_model_suite_guard` (`Suites()` lifecycle stages), `test_ctest_labels_guard`, `test_absl_format_guard`, `test_cuda_toolkit_guard`.

- [ ] **Step 3: Add the preset command table**

`## Running the suites` — table: `linux-debug` (fast), `linux-debug-convert`, `linux-debug-inference`, `linux-release-znver4-perf` (Release-only; Debug baselines invalid), `ctest -L "^(fast|convert|inference|perf)$"`; Windows `ctest --preset windows-debug` and the `ctest -R "qwen|gemma"` restriction.

- [ ] **Step 4: Add the router section**

`## Test router` — `tools/test_router.cmake` is the source of truth; `lsx_router_rule(<id> <labels> <regex>)`, FIRST match wins; the current rule table (from the verified facts); autodetect: `cmake -DLSX_PRESET=linux-debug -P tools/test_router.cmake` (and `-DLSX_LIST=ON` dry-run); `LSX_DIFF` override. Note the rule table in this skill is a snapshot.

- [ ] **Step 5: Add converter rules pointer + verification + mistakes**

`## Converter tests` (one paragraph): label `convert`, bounded TIMEOUT ≤600 s (canonically 120 s), synthetic checkpoint, `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY` runtime env — the full rules live in `ingesting-a-model` "Converter test rules". `## Common mistakes`: missing label; skipping `SKIP_RETURN_CODE 4` (skip reports as failure); running `-L perf` in Debug; quoting Debug baselines.

- [ ] **Step 6: Verify**

Run: `rg -n 'TBD|TODO|FIXME' .opencode/skills/test-lifecycle/SKILL.md; for t in test_ctest_labels_guard test_model_suite_guard tools/test_router.cmake lsx_router_rule LSX_PRESET linux-release-znver4-perf 'SKIP_RETURN_CODE 4'; do rg -q -- "$t" .opencode/skills/test-lifecycle/SKILL.md || echo "MISSING $t"; done`
Expected: no placeholder hits, no `MISSING`.

- [ ] **Step 7: Commit (gated)**

```bash
git add .opencode/skills/test-lifecycle/SKILL.md
git commit -m "docs: add test-lifecycle skill (labels, router, guards)"
```

---

### Task 2: Create `.opencode/skills/building-the-project/SKILL.md`

**Files:**
- Create: `.opencode/skills/building-the-project/SKILL.md`

- [ ] **Step 1: Write frontmatter + preset matrix**

Frontmatter (verbatim):
```yaml
---
name: building-the-project
description: Use when configuring, building or locating binaries in logestix — CMake presets (linux-debug/release-znver4/windows-debug), the AOCC host-compiler build, the Windows VS 2026 toolchain paths, toolchain minimums, and where the executables land. Triggers - build, preset, AOCC, windows toolchain, cmake not found.
---
```
`## Presets` — configure presets and the test presets that select labels; which one for what (fast iteration vs perf reference).

- [ ] **Step 2: Add Linux + AOCC sections**

`## Linux` — `git submodule update --init --recursive`, `cmake --preset linux-debug && cmake --build --preset linux-debug`, release-znver4 (LTO). `## Linux release — AOCC` — the full configure command verbatim from AGENTS.md (`AOCC=$HOME/aocc-compiler-5.2.0/bin`, `-DCMAKE_CUDA_HOST_COMPILER=$AOCC/clang++`), then `-j 160` build + `ctest --test-dir build/linux-release-znver4-aocc`.

- [ ] **Step 3: Add the Windows toolchain section**

`## Windows — VS 2026 toolchain` — exact paths for `cmake.exe`, `ctest.exe`, `ninja.exe`, `cl.exe` (`14.51.36231`), `nvcc.exe` (13.3, already in PATH); PowerShell invocation sample; WIN32 generator; `Get-ChildItem "$vs\VC\Tools\MSVC"` when VS updates; binaries under `build\<preset>\apps\<app>\<app>`.

- [ ] **Step 4: Add minimums, outputs, pointer**

`## Toolchain minimums` (CMake 3.25+, GCC 15.2, NVCC 13.4.59/13.3, CUDA ≥13.2, C++20) and `## Output` (`build/<preset>/apps/<app>/<app>`; `arrow.dll` POST_BUILD). Pointer: failures/patches/flags → `cuda-cmake-build-gotchas`.

- [ ] **Step 5: Verify**

Run: `rg -n 'TBD|TODO|FIXME' .opencode/skills/building-the-project/SKILL.md; for t in linux-release-znver4-aocc CMAKE_CUDA_HOST_COMPILER '14.51.36231' nvcc.exe windows-debug build/linux-debug; do rg -q -- "$t" .opencode/skills/building-the-project/SKILL.md || echo "MISSING $t"; done`
Expected: no placeholders, no `MISSING`.

- [ ] **Step 6: Commit (gated)**

```bash
git add .opencode/skills/building-the-project/SKILL.md
git commit -m "docs: add building-the-project skill (presets, AOCC, Windows toolchain)"
```

---

### Task 3: Create `.opencode/skills/coding-conventions/SKILL.md`

**Files:**
- Create: `.opencode/skills/coding-conventions/SKILL.md`

- [ ] **Step 1: Write frontmatter + language/logging/include rules**

Frontmatter (verbatim):
```yaml
---
name: coding-conventions
description: Use when writing or reviewing logestix host code — spdlog-only logging, #include and target_include_directories conventions, where new .cpp files go, Doxygen preservation, and the git commit-message convention. Triggers - code style, logging, include, where to add cpp, commit message, commit.
---
```
`## Language and logging` — C++20/C23; all logging through spdlog, no `std::cout`/`printf`. `## Includes and CMake layout` — include paths without comments; `target_include_directories` with `$<BUILD_INTERFACE:...>`; new `.cpp` into `apps/*/CMakeLists.txt` `add_executable`, logic into `libs/lsxcommon/`; preserve Doxygen/JSDoc on refactor.

- [ ] **Step 2: Add git conventions + pointers**

`## Git` — English messages; prefixes `build:`/`feat:`/`fix:`/`refactor:`/`test:`/`docs:`; commit only when explicitly asked. `## Related` — absl container/string policy → `absl-in-cuda-cpp`; kernel rules → `nvfp4-blackwell-conventions`; CMake wiring → `cuda-cmake-build-gotchas`.

- [ ] **Step 3: Verify**

Run: `rg -n 'TBD|TODO|FIXME' .opencode/skills/coding-conventions/SKILL.md; for t in spdlog target_include_directories BUILD_INTERFACE Doxygen 'docs:'; do rg -q -- "$t" .opencode/skills/coding-conventions/SKILL.md || echo "MISSING $t"; done`
Expected: no placeholders, no `MISSING`.

- [ ] **Step 4: Commit (gated)**

```bash
git add .opencode/skills/coding-conventions/SKILL.md
git commit -m "docs: add coding-conventions skill (style, includes, git)"
```

---

### Task 4: Fold host-resource policy into `running-a-model`; reduce AGENTS.md to anchors

**Files:**
- Modify: `.opencode/skills/running-a-model/SKILL.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Add the section to `running-a-model`**

Insert before `## Common mistakes` a new `## Host resource policy and GPU scheduling`: CPU/RAM are never the constraint (~10% / ~35%); the exception is inference and inference tests, where memory/GPU are critical; one GPU workload at a time; `lsxbenchmark` is strict GPU-exclusive; perf runs come last, after every other GPU task.

- [ ] **Step 2: Replace AGENTS §Build with an anchor**

Replace the §Build body (preset commands, AOCC, Windows paths, minimums, gotchas paragraph) with a short anchor + pointer to `building-the-project` (and keep the existing `cuda-cmake-build-gotchas` pointer).

- [ ] **Step 3: Replace AGENTS §Test with an anchor**

Replace the §Test body (label table, commands, router, converter/perf pointers, Windows restriction) with a short anchor + pointer to `test-lifecycle`; keep one line that converter rules live in `ingesting-a-model` and perf procedure in `perf-tuning-a-model`.

- [ ] **Step 4: Replace AGENTS §Conventions, §Git, §Resources with anchors**

§Conventions → pointer to `coding-conventions` (+ keep the one-line absl/kernel pointers already present). §Git → pointer to `coding-conventions`. §Resources → pointer to `running-a-model` (host resource policy section).

- [ ] **Step 5: Verify pointers resolve**

Run: `for s in test-lifecycle building-the-project coding-conventions running-a-model; do test -f .opencode/skills/$s/SKILL.md && echo "ok $s" || echo "MISSING $s"; done; rg -n 'TBD|TODO|FIXME' AGENTS.md .opencode/skills/running-a-model/SKILL.md`
Expected: all `ok`; no placeholder hits. Then `git diff --stat AGENTS.md` should be a net reduction.

- [ ] **Step 6: Commit (gated)**

```bash
git add AGENTS.md .opencode/skills/running-a-model/SKILL.md
git commit -m "docs: extract build/test/conventions to skills; add resource policy to running-a-model"
```

---

### Task 5: Move the orphaned Qwen3.8-Flash-Next bullet; final coverage re-run

**Files:**
- Modify: `AGENTS.md` (delete line ~339)
- Modify: `.opencode/skills/logestix-repo-map/SKILL.md`

- [ ] **Step 1: Delete the orphan from AGENTS.md**

Remove the trailing `- \`tests/test_qwen_moe_exp_{...}\` ...` bullet currently nested under `## Autonomous Mode`.

- [ ] **Step 2: Add it to the repo map**

Insert the same description as a `tests/` bullet in `logestix-repo-map/SKILL.md` next to `test_qwen_moe_36_*` (after the qwen test entries). Keep the existing `test_qwen_moe_exp_*` source entries untouched.

- [ ] **Step 3: Verify the move**

Run: `rg -n 'test_qwen_moe_exp' AGENTS.md` (expect no hits) and `rg -n 'test_qwen_moe_exp' .opencode/skills/logestix-repo-map/SKILL.md` (expect the suite bullet).

- [ ] **Step 4: Final coverage re-run (fresh)**

Run the mechanical sweep against all skills:
```bash
SK="$(cat .opencode/skills/*/SKILL.md)"
for d in apps/*/; do a=$(basename "$d"); grep -q -- "$a" <<<"$SK" || echo "APP MISSING $a"; done
for h in libs/lsxcommon/include/lsxcommon/*.h; do s=$(basename "$h" .h); grep -q -- "$s" <<<"$SK" || echo "HDR MISSING $s"; done
rg -n 'TBD|TODO|FIXME' .opencode/skills/ AGENTS.md
git status --short
```
Expected: no `APP MISSING`, no `HDR MISSING`, no placeholder hits, only intended files changed.

- [ ] **Step 5: Commit (gated)**

```bash
git add AGENTS.md .opencode/skills/logestix-repo-map/SKILL.md
git commit -m "docs: move qwen3.8-flash-next suite bullet from AGENTS to repo map"
```

---

## Self-Review

- **Spec coverage:** spec §1 → Task 1; §2 → Task 2; §3 → Task 3; §4 → Task 4 Step 1; §5 → Task 4 Steps 2–4 + Task 5; verification → each task + Task 5 Step 4. No gaps.
- **Placeholders:** none — each task names exact files, commands and the verified fact set.
- **Type/name consistency:** skill names `test-lifecycle`, `building-the-project`, `coding-conventions` are used identically in the spec, the tasks and the AGENTS pointers.
