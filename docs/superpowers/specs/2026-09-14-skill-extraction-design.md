# Skill extraction from AGENTS.md — Design

Date: 2026-09-14
Status: approved (design dialogue 2026-09-14)

## Problem

`AGENTS.md` (339 lines) still carries procedural knowledge that has no skill
home. The existing six skills cover the engineering + operational model
lifecycle, but three cross-cutting areas remain inline only:

1. **Test lifecycle** — the four CTest labels and their presets, the label
   discipline guard, the path router (`tools/test_router.cmake`,
   `lsx_router_rule`), and the Windows test restriction. Scattered between
   `ingesting-a-model` (verification + converter test rules), `perf-tuning-a-model`
   (perf suite) and `logestix-repo-map` (guard registries); the router rule table
   is in AGENTS and the tool source only.
2. **Building the project** — the preset matrix, the full AOCC invocation and
   the exact Windows VS 2026 toolchain paths. `cuda-cmake-build-gotchas` is
   symptom-driven diagnosis, not a build manual.
3. **Coding conventions + git** — spdlog-only logging, include style,
   `target_include_directories` generator expressions, new-`.cpp` placement,
   Doxygen preservation, and the commit-message convention.

Separately, the host-resource policy section (CPU/RAM budget, GPU exclusivity)
is duplicated in intent between AGENTS, `perf-tuning-a-model` and
`running-a-model`; and an orphaned Qwen3.8-Flash-Next test-suite bullet sits
under `Autonomous Mode` in AGENTS.md (line 339).

## Non-goals

- No code, no build, no GPU/inference run. Documentation only.
- `Autonomous Mode` is NOT extracted: it is always-on agent behavior, not a
  trigger-loaded skill.
- No duplication: the converter test rules stay in `ingesting-a-model`; CMake
  failure diagnosis stays in `cuda-cmake-build-gotchas`; absl string/container
  policy stays in `absl-in-cuda-cpp`; kernel rules stay in
  `nvfp4-blackwell-conventions`.

## Decisions (design dialogue 2026-09-14)

1. Create three skills: `test-lifecycle`, `building-the-project`,
   `coding-conventions`.
2. Each new skill OWNS its topic; specialized slices stay in their existing
   skill and are cross-referenced, not copied.
3. AGENTS.md sections become anchor + pointer (the pattern already used for
   cmake/ingesting/absl/perf), shrinking AGENTS.
4. Host-resource policy is folded into `running-a-model` as a new section.
5. The orphaned line 339 moves to `logestix-repo-map`.

## 1. `.opencode/skills/test-lifecycle/SKILL.md`

Triggers: add a test, ctest, labels, test router, converter TIMEOUT, guard test.

- Four categories (`fast`, `convert`, `inference`, `perf`) and when each runs;
  measured Linux counts as a snapshot.
- Label discipline: a new test MUST carry a label in the same commit
  (`test_ctest_labels_guard`); guard-test registry table (naming, independence,
  suite, ctest-labels, absl-format, toolkit).
- Preset-command table (`linux-debug`, `-convert`, `-inference`,
  `linux-release-znver4-perf`, `-L "^(fast|convert|inference|perf)$"`).
- Router: `tools/test_router.cmake`, `lsx_router_rule(<id> <labels> <regex>)`
  FIRST-match-wins, the current rule table, `LSX_PRESET`/`LSX_LIST`/`LSX_DIFF`,
  git-diff autodetection.
- `SKIP_RETURN_CODE 4` and bounded `TIMEOUT` (converter rule pointer).
- Windows restriction: `ctest -R "qwen|gemma"` only.
- Pointer: converter test rules live in `ingesting-a-model`; perf suite in
  `perf-tuning-a-model`.

## 2. `.opencode/skills/building-the-project/SKILL.md`

Triggers: build, preset, AOCC, Windows toolchain, CMake not found.

- Preset matrix and which preset answers which question; release = znver4 + LTO.
- Generic Linux configure/build/test; `git submodule update --init --recursive`.
- AOCC full invocation (`-ccbin` via `CMAKE_CUDA_HOST_COMPILER`).
- Windows VS 2026: `cmake`/`ctest`/`ninja` not in PATH, exact paths, MSVC `cl`
  and NVCC 13.3, PowerShell invocations, WIN32 generator, `arrow.dll` POST_BUILD.
- Toolchain minimums (CMake 3.25+, GCC 15.2, NVCC 13.4.59 Linux / 13.3 Windows,
  CUDA ≥13.2, C++20 pinned); binaries at `build/<preset>/apps/<app>/<app>`.
- Pointer: build failures/patches → `cuda-cmake-build-gotchas`.

## 3. `.opencode/skills/coding-conventions/SKILL.md`

Triggers: code style, logging, includes, where to add a .cpp, commit format.

- C++20 / C23; all logging through spdlog (no `std::cout`/`printf`).
- `#include` paths without comments; `target_include_directories` with
  `$<BUILD_INTERFACE:...>`.
- New `.cpp` goes into `apps/*/CMakeLists.txt` `add_executable`; logic into
  `libs/lsxcommon/`. Preserve Doxygen/JSDoc comments on refactor.
- Git: English messages; prefixes `build:`/`feat:`/`fix:`/`refactor:`/`test:`/
  `docs:`; commit only when explicitly asked.
- Pointers: absl container/string policy → `absl-in-cuda-cpp`; kernel rules →
  `nvfp4-blackwell-conventions`.

## 4. `running-a-model` — new section

"Host resource policy & GPU scheduling": CPU/RAM are never the constraint
(~10% / ~35%); the exception is inference and inference tests, where memory and
GPU are critical; one GPU workload at a time; `lsxbenchmark` is strict
GPU-exclusive; run perf last, after every other GPU task.

## 5. AGENTS.md reductions + orphan cleanup

- §Build, §Test, §Conventions, §Git, §Resources shrink to a short anchor plus a
  pointer to the owning skill. §Project/§CMake/§Model Protocol/absl/NVIDIA/GPU
  Perf are already pointers and stay.
- Line 339 (Qwen3.8-Flash-Next `test_qwen_moe_exp_*` suite) moves into
  `logestix-repo-map` next to the other qwen test suites; the orphan line is
  deleted from AGENTS.md.

## Verification

Documentation only.

- Frontmatter valid on all three new skills; `description` carries triggers.
- No `TBD`/`TODO`/`FIXME` in any skill.
- Every AGENTS.md pointer references an existing `.opencode/skills/<name>/SKILL.md`.
- Re-run the mechanical coverage sweep (apps / public headers / env vars /
  `libs/lsxcommon/src`) — no regressions.
- `git status` shows only intended files.

## Risks

- **Over-fragmentation.** Three more skills is a lot; mitigated by strict
  ownership + cross-reference and by keeping each skill to its topic.
- **AGENTS drift.** Removing sections could drop a rule; mitigated by moving,
  not deleting, and by pointing each anchor at the owning skill.
- **Router table staleness.** The documented `lsx_router_rule` table is a
  snapshot; the skill states the source of truth is `tools/test_router.cmake`.
