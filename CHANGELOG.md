# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `LSX_MODEL_ROOT` environment override for the model vault root
  (`libs/lsxcommon/src/path_registry.cpp`).
- `THIRD_PARTY_NOTICES`, `DEPENDENCIES.md` and a first-party license guard test.
- This changelog.
- Between-target GPU cooldown for `test_perf_baseline`: `perf::CoolDownGpu`
  (`libs/lsxcommon/src/gpu_power.cpp`, NVML read + a plateau wait) runs before
  each perf target, so a heat-soaked card no longer reads ~4% low and
  false-DEGRADEs against the references (`docs/OPEN_TASKS.md` §W).

### Changed

- Bumped Apache Arrow 25.0.1 -> 26.0.0 (FetchContent pin `apache-arrow-26.0.0`;
  Arrow now bundles Boost 1.92.0 and xsimd 14.2.0).
- Bumped simdjson v4.6.11 -> v5.0.5 (submodule `third_party/simdjson`; the
  `test_core.cpp` version guard now asserts 5.0.5).
- Bumped cpp-httplib v0.58.0 -> v0.61.0 (FetchContent pin; the `test_core.cpp`
  `CPPHTTPLIB_VERSION` guard now asserts 0.61.0).
- Bumped Catch2 v3.16.0 -> v3.16.1 (FetchContent pin; the `test_core.cpp`
  `CATCH_VERSION_PATCH` guard now asserts 1).
- NVIDIA CCCL is no longer a submodule: it is taken from the CUDA toolkit
  bundle (`<include>/cccl`) and **band-guarded** (configure-time in the root
  `CMakeLists.txt` + `test_core.cpp`). This removes the `-I` shadowing that made
  every `.cu` — including the toolkit's own transitively-included `<nv/target>` —
  compile against our copy, which could silently invert on a toolkit bump. The
  brief 3.4.2 -> 3.5.0 submodule bump is superseded by this.
- Translated `AGENTS.md`, `.opencode/skills/*` and documentation to English.
- Removed vendored NVIDIA documentation dumps; they are now referenced from the
  `nvidia-cuda-docs` MCP (`docs/NVIDIA_DOCS.md`).
- Added `abseil-cpp` to the README dependency table.

### Removed

- Tracked CTest artifact `Testing/Temporary/CTestCostData.txt` (now gitignored).
- `third_party/cccl` submodule — CCCL now comes from the CUDA toolkit bundle
  (band-guarded at configure time).

### Added

- `LOGESTIX_BUILD_APPS` / `LOGESTIX_BUILD_TOOLS` / `LOGESTIX_BUILD_TESTS`
  switches so logestix can be consumed as a dependency without building the
  sibling applications, tools or the engine test suite.

## [0.1.0] - 2026-09-19

Initial public release of the multi-app inference monorepo:

- `libs/lsxcommon` — shared inference core (NVFP4 quantization, tokenizer,
  model factory, device-independent GEMV, model families glm / qwen / deepseek /
  gemma / kimi / nemotron).
- `libs/lsxnetwork` — zero-alloc async network layer.
- Apps: `lsxfabric`, `lsxforge`, `lsxbenchmark`, `taskmosaic`, `lsxcli`.
- Catch2 test suite with `fast` / `convert` / `inference` / `perf` labels.
- Dual licensing: AGPL-3.0 for open source, commercial license on request.
