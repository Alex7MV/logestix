# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Full duplication audit across all 41 models (branch
  `refactor/dedup-audit-2026-11`): provable body moves into neutral layers
  inside `libs/lsxcommon`, bit-identical behaviour criterion. Net −4.3k LOC.
  Phase 3 closed (AttnRes depth mixer shared alice/kimi-k3; one grid-stride
  cast kernel per direction in `lsx_kernels_common.cuh`); tool-format JSON
  helpers shared via `lsxcommon::tools`; qwen/ornith MoE loader + generation
  driver templated on the passport (`lsx_moe_load`, `lsx_moe_inference_pipeline`);
  nemotron lightning/ultra kernels + expert cache shared. New instrument
  `tools/dedup_scan.py` (kernel dup groups 52→21, host 26→8). Reports:
  `docs/open-tasks/2026-11-20-kernel-primitives-closed.md`,
  `docs/open-tasks/2026-11-20-dedup-full-audit-closed.md`.

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

- Bumped simdjson v5.0.6 -> v5.2.0 (submodule `third_party/simdjson`, commit
  `a0d5ad97…`; the `test_core.cpp` version guard now asserts 5.2.0). On-demand
  correctness fixes upstream (malformed-value escapes, incomplete-JSON key
  cuts, `at_pointer` result handling); license texts unchanged.
- Bumped cpp-httplib v0.63.0 -> v0.66.0 (FetchContent pin `v0.66.0`; the
  `test_core.cpp` `CPPHTTPLIB_VERSION` guard now asserts 0.66.0).
- Consolidated the duplicated dense-decoder kernels: a new model-neutral
  `lsxcommon::dense` layer (`lsx_dense_inference.{h,cu}`) now owns RMSNorm,
  full-dim RoPE, causal GQA attention (with the split-KV path), fused split-QKV
  and fused SwiGLU; granite, granite-switch and phi keep their layer drivers
  and delegate through their existing `RunGranite*`/`RunGraniteSwitch*`/`RunPhi*`
  entry points. Net -729 lines in `libs/`, -206 in `tests/`, zero behaviour
  change (`tools/dedup_provenance.py` proves the kernels moved verbatim;
  `logits_hash` unchanged on all three models).
- Bumped Apache Arrow 26.0.0 -> 26.0.1 (FetchContent pin `apache-arrow-26.0.1`;
  the IPC reader now returns `Invalid` for a malformed map array instead of
  crashing — GH-54936. No new APIs in our IPC/io/CUDA surface, so no call site
  changed).
- Bumped simdjson v5.0.5 -> v5.0.6 (submodule `third_party/simdjson`; the
  `test_core.cpp` version guard now asserts 5.0.6).
- Bumped cpp-httplib v0.61.0 -> v0.63.0 (FetchContent pin; the `test_core.cpp`
  `CPPHTTPLIB_VERSION` guard now asserts 0.63.0).
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
