---
name: coding-conventions
description: Use when writing or reviewing logestix host code — spdlog-only logging, #include and target_include_directories conventions, where new .cpp files go, Doxygen preservation, and the git commit-message convention. Triggers - code style, logging, include, куда добавить cpp, commit message, коммит.
---

# Coding conventions

## Overview

Repo-wide style and layout rules for logestix host code. They are small but
enforced by review, and the logging rule has a mechanical check
(`test_absl_format_guard` plus the `rg` scan). Language/container policy and
kernel-level rules live in dedicated skills (see **Related**).

## Language and logging

- **C++20 / C23.**
- All logging goes through **spdlog** (`lsxcommon/log.h`, `spdlog::…`). No
  `std::cout`, no `printf`, no `fprintf` in production code.
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

- Adding a `std::cout` for a quick debug and leaving it in.
- Putting new logic in `apps/*/src/` instead of `libs/lsxcommon/`.
- Replacing `#include` lines with annotated ones.
- Using `{}` in `absl::StrFormat` and shipping a literal `{}` (see
  `absl-in-cuda-cpp`).
- Stripping Doxygen comments while moving code.
