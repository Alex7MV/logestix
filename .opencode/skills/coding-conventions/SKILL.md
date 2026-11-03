---
name: coding-conventions
description: Use when writing or reviewing logestix host code — spdlog-only logging, #include and target_include_directories conventions, where new .cpp files go, Doxygen preservation, and the git commit-message convention. Triggers - code style, logging, include, where to add cpp, commit message, commit.
---

# Coding conventions

## Overview

Repo-wide style and layout rules for logestix host code. They are small but
enforced by review, and the logging rule has a mechanical check
(`test_logging_guard`, label `fast`). Language/container policy and
kernel-level rules live in dedicated skills (see **Related**).

## Language and logging

- **C++20 / C23.**
- All logging goes through **spdlog** (`lsxcommon/log.h`, `spdlog::…`). No
  `std::cout`, no `printf`, no `fprintf` in production code.
  Machine-checked by `tests/test_logging_guard.cpp` (static scan of `libs/`
  and `apps/`): a stray stdio/stdout call fails, and every declared exception
  (device printf, env-gated perf probes, logger teardown, header-only
  invariants) is an exact trimmed line in its two-way `Allowlist()` — an
  undeclared call and a stale entry both fail.
- `absl::StrFormat`/`StrCat` are the string-formatting tools; `absl::StrFormat`
  takes printf-style conversions only (`%s/%d/%u/%f`) — a `{}` prints literally.
  The guard `test_absl_format_guard` scans `libs/` and `apps/`.

## Includes and CMake layout

- `#include` paths carry no explanatory comments.
- `target_include_directories` uses generator expressions:
  `$<BUILD_INTERFACE:...>` (and the matching `$<INSTALL_INTERFACE:...>` where
  relevant).
- A new `.cpp` goes into the owning app's `apps/*/CMakeLists.txt`
  `add_executable(...)`; shared logic goes into `libs/lsxcommon/` (and its
  `CMakeLists.txt`), not into the app.
- CUDA `.cu` files are isolated in their own OBJECT libraries — see
  `cuda-cmake-build-gotchas`.
- On refactor, preserve all Doxygen/JSDoc comments.

## Git

- Commit messages in **English**.
- Prefixes: `build:`, `feat:`, `fix:`, `refactor:`, `test:`, `docs:`.
- Commit only when explicitly asked by the operator; never amend, force-push or
  commit secrets.

## Related

- Container/string policy (`absl::` vs `std::`, the NVCC reachability rule) →
  `absl-in-cuda-cpp`.
- Kernel/Blackwell rules (one stream, shared-mem sandbox, NVFP4 layer) →
  `nvfp4-blackwell-conventions`.
- CMake wiring and the flag/patches that must not leak into NVCC →
  `cuda-cmake-build-gotchas`.

## Common mistakes

- Adding a `std::cout` or a `printf`/`fprintf(stderr, …)` probe for a quick
  debug and leaving it in — convert it to `log::debug/info/warn/error`, or
  declare the exact line in `test_logging_guard`'s allowlist with a category.
- Putting new logic in `apps/*/src/` instead of `libs/lsxcommon/`.
- Replacing `#include` lines with annotated ones.
- Using `{}` in `absl::StrFormat` and shipping a literal `{}` (see
  `absl-in-cuda-cpp`).
- Stripping Doxygen comments while moving code.
