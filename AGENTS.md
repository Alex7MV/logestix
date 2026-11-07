# logestix — AGENTS.md

Multi-app monorepo: fabric, forge, taskmosaic, lsxcli. C++20 / C23.
CMake 3.25+ (CMakePresets). Target architecture: NVIDIA Blackwell
(CC 12.0 / 12.1), 5th-gen Tensor Cores and TMA. Max warps per SM on Blackwell:
48 (factor into occupancy calculations).

**Forbidden architectures:** Hopper (90), Ada (89), Ampere (80). No backward
compatibility — any code targeting pre-Blackwell is rejected.

**Toolkit/driver skew.** CUDA Toolkit 13.5 (nvcc V13.5.37; minimum 13.2+). The
Linux driver is 13.4 (`615.71.09`, driver API 13040), so the runtime runs in
**minor-version compatibility** mode: APIs introduced after 13.4 are in the
headers and absent from `libcuda.so` at runtime
(`cuTensorMapSetAllocationClass`, `cuGraphAddMemcpyBatchNode`,
`cuMemsetBatchAsync`, `cuFabricCpyBatchAsync`). `test_cuda_toolkit_guard` prints
that skew and fails only on a major mismatch.

## Where things are

| question | answer |
|---|---|
| how do I build it / where are the binaries | skill **`building-the-project`** |
| build broke, or I am wiring CMake | skill **`cuda-cmake-build-gotchas`** |
| which file holds model X's converter / kernels / loader | skill **`logestix-repo-map`** |
| what a model IS: geometry, format tag, vault dir, gates, perf, debt | `docs/MODELS.md` (generated) |
| which model passes which parity gate | `docs/SELFCHECK_MATRIX.md` |
| why is throughput what it is | `tests/perf/perf_baselines.json` + the model's report |
| what is open / known broken | `docs/OPEN_TASKS.md` (resolved work: `docs/open-tasks/`) |
| why a green suite is a sample, not a proof | `docs/OPEN_ENDEDNESS.md` |
| how do I run a model | skill **`running-a-model`** |
| how do I add one | skill **`ingesting-a-model`** |
| how do I make it faster | skill **`perf-tuning-a-model`** |
| labels, presets, test router | skill **`test-lifecycle`** |
| FSI / CODE / DET checkpoint gates | skills **`fsi-selfcheck`**, **`code-selfcheck`**, **`det-selfcheck`** |
| host C++ style, logging, git | skill **`coding-conventions`** |
| containers/strings in host C++ | skill **`absl-in-cuda-cpp`** |
| NVFP4 / Blackwell kernel rules | skill **`nvfp4-blackwell-conventions`** |
| an ingested model's report | `docs/model-reports/YYYY-MM-DD-<name>.md` |

A number in prose rots; the registries do not. When this file and a
machine-checked registry disagree, the registry wins.

## The model registry

`models/<model_type>.json` is the single source of truth for one ingested model.
Everything derived from it is generated at configure time into
`${CMAKE_BINARY_DIR}/generated/` and must never be hand-edited:

| generated | replaces |
|---|---|
| `model_manifest.h` | the guard registries (`kModels`/`kTargets`/`kLifecycle`/`kDebt`) |
| `model_targets.cmake` | every model target's `add_test` + label/timeout/skip/lock/env |
| `model_registry.inc` | the `LsxModelFactory` registration list and its aliases |

`test_manifest_sync` (label `fast`) proves the compiled table equals the
manifests, the committed `docs/MODELS.md` is current (`--check`) and no
`tests/test_*.{cpp,cu}` file is unclaimed (`--strict`). Adding a model means
adding a manifest row, a converter and its kernels — no registry, no CMake
registration, no prose.

Regenerate the catalogue after editing a manifest:

```bash
python3 tools/gen_model_manifest.py --docs docs/MODELS.md --root .
```

## The lifecycle gate (+ the direction to grow)

Every model must satisfy: SelfCheck ("Paris", >= 3 bit-exact cycles), FSI, CODE,
DET (`docs/SELFCHECK_MATRIX.md`), a native tool round-trip, and a seeded perf
row at the canonical 2048-token protocol. A gate that SKIPs is **not** a pass;
a gate whose golden was never captured is declared debt in the manifest, and
`test_model_suite_guard` fails an undeclared gap.

Shared kernel layers exist so families do not copy each other: `lsx_nvfp4`
(the only NVFP4 path), `lsx_gemv` (**the only GEMM path** — cuBLAS is not linked),
`lsxcommon::attn` / `dense` / `moe`, `lsx_device_math.h`, `lsx_reduce.cuh`.
`test_model_independence_guard` enforces unique frozen geometry per model and
forbids cross-family borrowing; `test_model_suite_guard` enforces that each
family's suite is complete and self-contained.

## Invariants that are easy to break

- **Logging** goes through spdlog / `lsxcommon::log` only. No `std::cout`,
  no `printf` in `libs/` or `apps/` (`test_logging_guard`, two-way allowlist).
- **`absl::StrFormat` is printf-style** — `%s`/`%d`, never `{}`
  (`test_absl_format_guard`).
- **Host-only C++ prefers `absl::`** (flat/btree containers, `StrCat`).
  Headers reachable from `.cu`/`.cuh` must NOT include `<absl/container/*>`
  (NVCC fails in raw_hash_map with C2794/C2938); there it stays `std::`
  (`test_absl_container_guard`).
- **Arrow reads are mmap-only** and writes atomic (`test_arrow_usage_guard`).
- **No CLI speed flags.** The engine always runs a per-GPU auto policy; the only
  numeric run flag is `--bit-exact` (`FastMathMode::kOff`). A family that pins
  `kOff` does it because the NVFP4 tensor-core path overflows on its
  activations, and says so in its manifest/report.
- **One GPU workload at a time.** Every target whose label implies device work
  (`convert|load|model|kernels|selfcheck|fsi|code|det|perf|inference`) declares
  `RESOURCE_LOCK "gpu"`, and `test_ctest_labels_guard` fails a target that does
  not; the single host-only exception carries a reason in that guard. `fast`
  targets stay unlocked so the 140-target fast sweep still parallelises.
  **The lock is per ctest process, not per machine** — two concurrent `ctest`
  invocations both acquire it and share the GPU, which is how a gate came back
  red on 2026-11-07 while a background sweep held 30 GB; never start a
  GPU-labelled run while another is in flight, check
  `nvidia-smi --query-compute-apps` first.
- **Dependency include paths live only in `lsx_dep_includes`.** Never spell
  `${CMAKE_BINARY_DIR}/_deps/...` in a target (`test_infrastructure_guard`).
- **Device codegen comes from `LSX_CUDA_ARCH`** via
  `lsx_cuda_gencode_flags()`; every device target sets `CUDA_ARCHITECTURES OFF`.
  Never hand `LSX_CUDA_ARCH` to `CUDA_ARCHITECTURES`: CMake 4.x drops the family
  suffix and silently substitutes `compute_75`.
- **Test registration:** use `cmake/LsxTestHelpers.cmake`
  (`lsx_add_test`/`lsx_add_cuda_test`); every target carries one label
  (`fast|convert|inference|perf`) and every skip-capable target sets
  `SKIP_RETURN_CODE 4` (`test_ctest_labels_guard`).
- **NVIDIA documentation is not vendored** — load current docs from the
  `nvidia-cuda-docs` MCP before editing NVFP4/CUDA kernels
  (`docs/NVIDIA_DOCS.md`); if it is unavailable, stop and ask.

## Dependencies

| Dependency | Version | Source | Type |
|---|---|---|---|
| Apache Arrow | 26.0.1 | FetchContent (BUNDLED) | Shared (`arrow.dll`) |
| spdlog | 1.17.0 | FetchContent | Header-only |
| FlatBuffers | v25.12.19 | submodule | Static |
| libzmq | v4.3.5 | submodule | Static |
| simdjson | v5.0.6 | submodule | Static (`.lib`) |
| NVIDIA CCCL | toolkit bundle, band 3.2.x–3.5.x | `<include>/cccl` | Header-only |
| abseil-cpp | 20260817.0 | FetchContent (LTS) | Header-only + static |
| Catch2 | v3.16.1 | FetchContent (test only) | Header-only |
| cpp-httplib | v0.63.0 | FetchContent (host HTTP) | Header-only |

Pins are enforced by `tests/test_core.cpp` → `TEST_CASE("pinned dependency
versions")` (no guard for flatbuffers/libzmq). CCCL is deliberately not pinned:
it is band-guarded at configure time and in `test_core.cpp`, so a toolkit bump
outside the validated window fails configure. The bump procedure (pin + guard +
tables + re-check the after-fetch patches) is skill
`cuda-cmake-build-gotchas`.

## Skill map — the model lifecycle, phase by phase

| phase | skill | what it decides |
|---|---|---|
| 0. intake / triage | **`ingesting-a-model`** § Stage 0 | new family or sibling? geometry, tokenizer verdict, ballast, does it fit |
| 1. ingestion | **`ingesting-a-model`** stages 1–8 | manifest → converter → NVFP4 → tokenizer bridge → 4b tool calling → kernels → sandbox → tests (stage 7, the token-translation passport, is retired) |
| 2. build | **`building-the-project`**, `cuda-cmake-build-gotchas` | presets, toolchain, where binaries land, why the build breaks |
| 3. correctness gates | **`test-lifecycle`**, `fsi-selfcheck`, `code-selfcheck`, `det-selfcheck` | labels, targets, the three parity/determinism goldens |
| 4. throughput | **`perf-tuning-a-model`** | attribute wall time, budget tests, baselines |
| 5. launch + serve | **`running-a-model`** §2a/§2b | the ordered fast-launch runbook and the acceptance checklist |
| 6. locating code / rules | **`logestix-repo-map`**, `nvfp4-blackwell-conventions`, `absl-in-cuda-cpp`, `coding-conventions` | which file owns what; kernel and host-C++ rules |
| 7. report artifact | **`ingesting-a-model`** § Stage 9 | the publishable report in `docs/model-reports/` + its evidence |
| host resources / GPU | **`running-a-model`** §8 | one GPU workload at a time, inference exemption |

## Autonomous Mode

- The AI agent is the sole creator and editor of code files (`.cu`, `.cpp`, `.h`,
  `CMakeLists.txt`). The operator drives through prompts and instruction files.
- If manual code changes appear outside AI sessions, audit the full diff and
  reconcile with the operator before proceeding.
- **No parallel/subagents**: Task subagents (explore/general/parallel agents) are
  forbidden — they make many mistakes. All research, audits and edits happen in
  the main session directly.

## Git

English commit messages; prefixes `build:`/`feat:`/`fix:`/`refactor:`/`test:`/
`docs:`. Commit only on explicit request; never amend, force-push or commit
secrets. Full rules: skill `coding-conventions`.
