# logestix — AGENTS.md

## Project

Multi-app monorepo: fabric, forge, taskmosaic, lsxcli. C++20 / C23. CMake 3.25+ (CMakePresets).
Target architecture: NVIDIA Blackwell (CC 12.0 / 12.1). Active use of 5th-gen Tensor Cores and TMA.
Max warps per SM on Blackwell: 48 (factor into occupancy calculations).
Forbidden architectures: Hopper (90), Ada (89), Ampere (80).
No backward compatibility — any code targeting pre-Blackwell (sm_90, sm_80) is rejected.
CUDA Toolkit 13.4 (nvcc V13.4.59; minimum 13.2+), full native C++23 support in NVCC.

## Dependencies

| Dependency | Version | Source | Type |
|---|---|---|---|
| Apache Arrow | 25.0.1 | FetchContent (BUNDLED) | Shared (`arrow.dll`) |
| spdlog | 1.17.0 | FetchContent | Header-only |
| FlatBuffers | v25.12.19 | submodule (`third_party/flatbuffers`) | Static |
| libzmq | v4.3.5 | submodule (`third_party/libzmq`) | Static |
| simdjson | v4.6.11 | submodule (`third_party/simdjson`) | Static (`.lib`) |
| NVIDIA CCCL | 3.4.2 | submodule (`third_party/cccl`) | Header-only |
| abseil-cpp | 20260817.0 | FetchContent (`20260817.0`, LTS August 2026) | Header-only + static libs |
| Catch2 | v3.16.0 | FetchContent (test only) | Header-only |
| cpp-httplib | v0.18.3 | FetchContent (host-only HTTP) | Header-only |

Versions are enforced by code: `tests/test_core.cpp` → `TEST_CASE("pinned dependency
versions")` (`STATIC_REQUIRE` for Arrow, spdlog, simdjson, Catch2, CCCL, abseil —
the latter via `ABSL_LTS_RELEASE_VERSION`, so a non-LTS pin will not build); there is
no guard for `flatbuffers`/`libzmq`. The dependency bump procedure
(pin the submodule/FetchContent `GIT_TAG` + guard + AGENTS/README tables
+ re-check after-fetch patches) — skill `cuda-cmake-build-gotchas`.

## Build

Presets, the AOCC build, the Windows toolchain (VS 2026) with exact paths, toolchain
minimums and binary locations — skill **`building-the-project`**
(`.opencode/skills/building-the-project/SKILL.md`).

Quick check:

```bash
cmake --preset linux-debug && cmake --build --preset linux-debug
ctest --preset linux-debug                     # fast only
```

Non-obvious build pitfalls — skill **`cuda-cmake-build-gotchas`**
(`.opencode/skills/cuda-cmake-build-gotchas/SKILL.md`): CMake 4.x and
`enable_language(CUDA)` in a nested directory, host flags leaking into NVCC,
`/Zc:preprocessor` vs CCCL/thrust, `CMAKE_POLICY_VERSION_MINIMUM` for xsimd,
Arrow/Boost/abseil patches, `lsx_tune`/LTO gating, the `cc12x`-only arch set, host-only
CUDA surfaces and the `_putenv` guard, dependency bumps. It also has the
"symptom → cause" table.

## Resources (host)

Host-machine resource policy, the inference exemption and the
"one GPU load at a time" rule (including strict GPU exclusivity for `lsxbenchmark` and
running perf last) — the "Host resource policy and GPU scheduling" section of skill
**`running-a-model`**
(`.opencode/skills/running-a-model/SKILL.md`).

## Test

`BUILD_TESTING=ON` is enabled in all presets. Test categories (labels
`fast`/`convert`/`inference`/`perf`), presets, the path router
(`tools/test_router.cmake`, `LSX_PRESET`/`LSX_LIST`/`LSX_DIFF`), label discipline
in the same commit, the guard-test registry and the WINDOWS restriction
(`ctest -R "qwen|gemma"`) — skill **`test-lifecycle`**
(`.opencode/skills/test-lifecycle/SKILL.md`).

Converter test rules (label `convert`, bounded TIMEOUT, synthetic
checkpoint, `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY` as a runtime env) — the
"Converter test rules" section in `ingesting-a-model`; the measurement/speed-up
procedure — `perf-tuning-a-model`.

## Structure

- `libs/lsxcommon/` — shared STATIC library; namespace `lsxcommon`. Core:
  `core.cpp` (logging/init), `schema.cpp` (ExpertSchema/PackedExpert),
  `converter.cpp` (GPU NVFP4 quantization), `loader.cpp`, `tokenizer.cpp` +
  `pretokenizer.cpp`/`unicode.cpp`, `model_factory.cpp`/`model_registry.cpp`,
  `model_runtime.cpp` (lsxbenchmark --bench), `gpu_perf.cpp`, the model-neutral
  layer `lsx_inference_nvfp4.cu` + device-independent GEMV `lsx_gemv.cu`
  (`RunLxGemmVec`/`RunLxGemmVecWT`; the only GEMM path), `expert_store.h`.
  FSISelfCheck (`fsi_selfcheck.{h,cpp}`, host-only, model-neutral; capture
  `tools/fsi_capture.py` + `tools/fsi_prompts.json`; per-family goldens
  `tests/fixtures/fsi/<type>.json`; separate from SelfCheck — compares the
  converted model's canonicalized answer at `kOff` to a golden captured from the
  source safetensors; `lsxbenchmark --fsi-selfcheck`; per-family
  `test_<family>_fsi_selfcheck.cpp`, label `inference`, SKIP 4; findings
  `docs/FSI_SELFCHECK_FINDINGS.md`). Capture/verify workflow, golden schema and
  per-model quirks — skill **`fsi-selfcheck`**
  (`.opencode/skills/fsi-selfcheck/SKILL.md`).
  CODESelfCheck (`code_selfcheck.{h,cpp}`, host-only, model-neutral — the
  compiler-syntactic-strictness analogue of FSI, comparing the converted model's
  *exactly* normalized answer at `kOff` to a golden captured from the source
  safetensors; capture `tools/code_capture.py` + `tools/code_prompts.json`;
  per-family goldens `tests/fixtures/code/<type>.json`; `lsxbenchmark
  --code-selfcheck`; per-family `test_<family>_code_selfcheck.cpp`, label
  `inference`, SKIP 4; suite-guard `code` stage; findings
  `docs/CODE_SELFCHECK_FINDINGS.md`; design
  `docs/superpowers/specs/2026-09-27-code-selfcheck-design.md`). Capture/verify
  workflow, golden schema (incl. the prompt-echo `body` cut), extraction
  (json/assignment/regex) and per-model quirks — skill **`code-selfcheck`**
  (`.opencode/skills/code-selfcheck/SKILL.md`).
  DETSelfCheck (`det_selfcheck.{h,cpp}`, host-only, model-neutral — the
  determinism + near-tie analogue of FSI/CODE: runs each prompt R times
  (`stress_runs`, default 3, floored at 3, `LSX_DET_STRESS_RUNS` override),
  requires byte-identical `gen_tokens` across runs and the extracted canonical
  token at `kOff` to match a golden captured from the source safetensors;
  capture `tools/det_capture.py` + `tools/det_prompts.json`; per-family goldens
  `tests/fixtures/det/<type>.json`; `lsxbenchmark --det-selfcheck`; per-family
  `test_<family>_det_selfcheck.cpp`, label `inference`, SKIP 4; suite-guard
  `det` stage; findings `docs/DET_SELFCHECK_FINDINGS.md`; design
  `docs/superpowers/specs/2026-09-27-det-selfcheck-design.md`). Capture/verify
  workflow, golden schema, extraction (`token`) and per-model quirks — skill
  **`det-selfcheck`** (`.opencode/skills/det-selfcheck/SKILL.md`).
  Models: glm / qwen / deepseek / gemma / kimi / nemotron / ornith (NemotronH hybrid
  Mamba-2 + MoE + full attention: `nemotron_moe_constants.h` (frozen passport),
  `nemotron_moe_convert.cpp`, `nemotron_moe_load.cpp` (dense-only resident graph),
  `nemotron_moe_expert_cache.{h,cpp}` (host mmap store of 2944 experts + device
  slot-LRU; dense always resident, experts streamed — the model fits in 16 GiB,
  knob `LSX_NEMOTRON_EXPERT_CACHE_SLOTS`), `nemotron_moe_model.*`,
  `nemotron_moe_kernels.{h,cu}` (conv/SSM/gated-RMSNorm/router/relu2),
  `nemotron_moe_inference_attn.cu` (RoPE+GQA), `nemotron_moe_inference_pipeline.cu`,
  `nemotron_moe_device_math.h`; format tag `logestix-nemotron-nvfp4`).
  GigaChat (GigaChat3.5-432B-A28B-Reasoning hybrid MLA + GatedDeltaNet + MoE:
  `gigachat_moe_constants.h` (frozen passport), `gigachat_moe_convert.{h,cpp}`
  (FP8 block-128 -> neutral NVFP4), `gigachat_moe_dequant.cu` (partial-tile
  FP8->BF16),   `gigachat_moe_load.{h,cpp}`, `gigachat_moe_model.{h,cpp}`,
  `gigachat_moe_device_math.h`, `gigachat_moe_norm_kernels.cu` /
  `gigachat_moe_kernels.{h,cu}` (ZeroCenteredGatedNorm, GDN output norm,
  sigmoid router, swiglu_limit, sigmoid gate),
  `gigachat_moe_expert_cache.{h,cpp}` (host mmap store of 9472 experts + device
  slot-LRU, knob `LSX_GIGACHAT_EXPERT_CACHE_SLOTS`);
  format tag `logestix-gigachat-nvfp4`; MLA/GatedDeltaNet attention + decode
  pipeline land in Plan 2b).
  Ornith-1.5-35B-A3B (independent family `ornith`: an MLX-exported
  Qwen3.5-MoE reasoning fine-tune, architecture-identical to Qwen3.5/3.6-35B;
  `ornith_moe_constants.h` (frozen passport), the standalone MLX converter
  `ornith_moe_convert.{h,cpp}` (prefix `language_model.model.*`,
  `mlp.switch_mlp.{gate,up,down}_proj` stacked experts, conv `[C,K,1]`),
  `ornith_moe_load.{h,cpp}` + `ornith_moe_inference.{h,cu}` (duplicated
  qwen_moe runtime layer; reuses the shared generic `qwen_inference.*` kernels
  and routes NVFP4 through the neutral `lsx_nvfp4`), `ornith_moe_model.{h,cpp}`;
  format tag `logestix-ornithmoe`; knob
  `LSX_ORNITH_MOE_EXPERT_CACHE_SLOTS`).
  Granite-4.2-30B (independent family `granite`: dense decoder-only,
  64L / 4096H / 32Q / 8KV / 128HD / 32768I, SwiGLU, full-dim RoPE,
  `attention_multiplier=1/128`; `granite_dense_constants.h` (frozen passport),
  `granite_dense_convert.{h,cpp}` (flat Granite config -> `logestix-granite`),
  `granite_dense_load.{h,cpp}` (NVFP4 pinned weights, fused `gate_up`),
  `granite_dense_inference.{h,cu}` (causal GQA + full RoPE + SwiGLU dense loop
  over the neutral `lsx_nvfp4` layer and the shared `lsx_kernels_*.cuh`
  primitives), `granite_dense_model.{h,cpp}`; converted in place with
  `lsxforge --granite`; full 2048-token pipeline is correct-first
  (KV-cache decode, non-incremental prefill)).
- `libs/lsxnetwork/` — shared STATIC library; namespace `lsxnetwork`; zero-alloc
  async network layer for TaskMosaic CE (FrameQueue SPSC, ZMQ context, I/O worker,
  OpenAI endpoint, affinity).
- `apps/lsxfabric/` — B2B inference engine, always maximum speed
  (`--localmodel`, `--validate`, `--testprompt`); speed measurement and bit-exactness
  moved to lsxbenchmark. `apps/lsxforge/` — data preprocessing utility
  (`--glm`/GLM-5.2 pipeline, `--tokenizer`, repack flags). `apps/taskmosaic/` —
  OpenAI-compatible HTTP/SSE bridge to a running `lsxfabric --serve` over the v2
  streaming fabric protocol (`--config`; `/v1/chat/completions`,
  `/v1/completions`, `/v1/models`, `/health`, `/ready`). `apps/lsxcli/` — CLI client
  (`--model`, `--info`, `--infofull`).
- `apps/lsxbenchmark/` — generation-speed measurement + bit-exactness: `--bench`
  (real generation via `IModel::Infer`), `--selfcheck` (Paris), `--bit-exact`
  (`FastMathMode::kOff`); the only app with a numeric mode.
- `tests/` — Catch2 v3.16.0 suite; labels `fast`/`convert`/`inference`/`perf`
  (see § Test) + model/naming/absl/toolchain guard tests.
- `third_party/cccl/`, `third_party/flatbuffers/`, `third_party/libzmq/`,
  `third_party/simdjson/` — submodules.

The full per-file inventory (by family and subsystem: suite membership and
assertions, selfcheck env vars, model perf details) — skill
**`logestix-repo-map`**
(`.opencode/skills/logestix-repo-map/SKILL.md`). Its machine-checked
equivalents: `test_model_suite_guard.cpp::Suites()` (model → stages →
ctest targets) and `test_model_independence_guard.cpp::kPassports` (model →
passport header → namespace) — when they disagree, they are authoritative.

Running and operating inference — skill **`running-a-model`**
(`.opencode/skills/running-a-model/SKILL.md`): apps and their flags
(`lsxbenchmark --bench/--selfcheck`, `lsxfabric --localmodel --testprompt/--validate`,
`lsxforge`, `lsxcli`), vault paths (`path_registry`; override the default root with
`LSX_MODEL_ROOT`, and the CLI takes the model ROOT, not `logestix/`), the engine
lifecycle (`InferenceEngine::Session`), prompt
framing/sampling, the resource layer (governor/VRAM/LRU planning) and the catalog of
runtime env knobs `LSX_*`/`KIMI_*`.

## Model Ingestion Protocol

Adding/adapting a new model — skill **`ingesting-a-model`**
(`.opencode/skills/ingesting-a-model/SKILL.md`): the 8-step MODEL INGEST INVARIANT
(lsxforge converter → NVFP4 quantization → registration in `LsxModelFactory` +
frozen geometry → tokenizer bridge → CUDA kernels → shared-mem sandbox → passport →
SelfCheck/tests), the model file naming convention
(`<family>_<type>[_<variant>]_<role>[_<subsystem>]`) and the guard-test registries
(`test_code_naming_guard`, `test_model_independence_guard`,
`test_model_suite_guard`), which must be extended in the same commit as the
model files. The single source of truth is SKILL.md; only an anchor is kept here
because the protocol is cited in code as "Model Ingestion Protocol step N".

## CMake Key Points

- `lsxcommon` — STATIC (`libs/lsxcommon/CMakeLists.txt`); PUBLIC links to spdlog,
  flatbuffers, libzmq-static, cudart, lsxnetwork; PRIVATE — arrow_shared,
  arrow_cuda_shared, simdjson. Arrow include paths are added PUBLIC directly to
  `lsxcommon` (`_deps/arrow-src/cpp/src`, `_deps/arrow-build/src`).
- CUDA `.cu` files are split into separate OBJECT libraries (`lsxcommon_cu`,
  `lsxcommon_glm_cu`, `lsxcommon_qwen_moe_cu`, ...), linked into `lsxcommon`
  PUBLIC — this isolates NVCC from MSVC flags. `cudart` is found via
  `find_package(CUDAToolkit)` (Linux and Windows), not hardcoded paths; cuBLAS is not
  linked and not used at all — the only GEMM path is device-independent `lsx_gemv`
  (see `numeric-conformance-audit.md`).
- `lsxnetwork` — STATIC; simdjson, libzmq-static and Threads PUBLIC
  (a STATIC library re-exports link dependencies), spdlog PRIVATE.
- Every app in `apps/*/` links `lsxcommon` PRIVATE; Catch2 v3.16.0
  via FetchContent when `BUILD_TESTING=ON`.
- `CMAKE_RUNTIME_OUTPUT_DIRECTORY` = `${CMAKE_BINARY_DIR}/bin` (before
  `add_subdirectory`); POST_BUILD copies `arrow.dll`/`arrow_cuda.dll` (Win32).
- Flags (MSVC/NVCC), `lsx_tune`/LTO gating, dependency patches and CUDA architectures
  — skill `cuda-cmake-build-gotchas`.

## Conventions

Language C++20/C23, logging only via spdlog (no `std::cout`/`printf`),
`#include` style, `target_include_directories` with `$<BUILD_INTERFACE:...>`, placement
of new `.cpp` files (app `add_executable` vs `libs/lsxcommon/`) and preservation of
Doxygen/JSDoc — skill **`coding-conventions`**
(`.opencode/skills/coding-conventions/SKILL.md`).

Kernel/Blackwell rules — skill **`nvfp4-blackwell-conventions`**
(`.opencode/skills/nvfp4-blackwell-conventions/SKILL.md`): one stream and minimal
`cudaDeviceSynchronize`, shared-mem sandbox (thread 0 + barriers), NVFP4 only via the
neutral `lsx_inference_nvfp4` + `std::mdspan`, CCCL priority (cub, thrust
fallback on MSVC), Tile C++, CompileIQ, alignment/48 warps, zero-copy Arrow+CUDA and
simdjson+libzmq, "if unsure about a signature — check the `.h`/`.cuh`".

## absl:: Priority (Container/String policy)

Host-only C++ (NOT compiled by NVCC) prefers `absl::`: `unordered_map/set` →
`flat_hash_map/set`, `map/set` → `btree_map/set`, simple concatenations →
`absl::StrCat`/`StrAppend`. The priority table, exceptions (what stays `std::`),
the NVCC container incompatibility with the reachability rule and recipe check,
the printf contract of `absl::StrFormat` and the CMake anchors (ABSL_PROPAGATE_CXX_STD,
crc32 and format-checker patches) — skill **`absl-in-cuda-cpp`**
(`.opencode/skills/absl-in-cuda-cpp/SKILL.md`). The inviolable parts stay here:

- Headers reachable from `.cu`/`.cuh` must NOT include `<absl/container/*>`
  (NVCC fails in raw_hash_map: `C2794`/`C2938`); in `.cu`/`.cuh` containers and
  streams stay `std::`. Currently only `lsx_orchestrator.h` and
  `model_integrity.h` (host-only) include a container header; there are no
  violations from `.cu`; the transitive-graph check is in the skill (the guard does NOT
  automate it).
- `absl::StrFormat`/`StrAppendFormat`/`Format` — printf style (`%s/%d/%u/%f`);
  `{}` is printed literally (guard `test_absl_format_guard`).
- These stay `std::`: `string_view`/`optional`/`variant`/`function`/`unique_ptr`,
  stream builders with `setw`/`setfill`, custom hasher functors.

## NVIDIA Docs Reference (mandatory)

- NVIDIA documentation is NOT vendored: dumps from the `nvidia-cuda-docs` MCP
  have been removed from the repository to avoid distributing copyrighted material.
  The pointer and procedure — `docs/NVIDIA_DOCS.md`.
- Before editing NVFP4/CUDA kernels, load the current documentation from the
  `nvidia-cuda-docs` MCP (PTX ISA, CUDA C++ Programming Guide, `cuda_fp4`/`cuda_fp8`,
  CUTLASS, TMA, tcgen05) instead of relying on outdated excerpts. If the MCP is
  unavailable — stop and get access to the current docs.
- Discipline (verify semantics BEFORE editing, record derived
  invariants in code/regression tests) — skill `nvfp4-blackwell-conventions`.

## GPU Perf Auto-Tuning

All speed work — decode-token profiling (nsys), TDD budget tests,
`gpu_perf` knobs, the perf baseline — is described in skill **`perf-tuning-a-model`**
(`.opencode/skills/perf-tuning-a-model/SKILL.md`). The inviolable parts stay here:

- **No CLI speed/perf flags**: the engine always runs a per-GPU auto policy;
  the only numeric run flag is `--bit-exact` (lsxbenchmark), which maps
  to `FastMathMode::kOff`. `LSX_PERF_AUTO=0` restores the legacy bit-exact defaults,
  `LSX_TEST_FAST_MATH` forces TC. Tests that pin the numeric class set `kOff`.
- `gpu_perf` (`include/lsxcommon/gpu_perf.h`) resolves the knobs at `InitializeEngine`
  (TC fast-math, GEMV MiB limit, split-K threshold, % VRAM for the expert cache,
  dual-stream staging); the resolution priority and knob consumers are in the skill.
- `lsxbenchmark` — strict GPU exclusivity (see Resources).

## Git

English commit messages and the prefixes `build:`/`feat:`/`fix:`/`refactor:`/
`test:`/`docs:` (plus the "commit only on explicit request" rule) — skill
**`coding-conventions`** (`.opencode/skills/coding-conventions/SKILL.md`).

## Autonomous Mode

- The AI agent is the sole creator and editor of code files (`.cu`, `.cpp`, `.h`, `CMakeLists.txt`).
- The operator manages the process exclusively through high-level prompts and changes to instruction files.
- If manual code changes are detected outside AI sessions, perform a full diff audit and reconcile with the operator before proceeding.
- **No parallel/subagents**: using Task subagents (explore/general/parallel agents) is forbidden — they make many mistakes. All codebase research, audits and edits are done by the main session directly (Read/Grep/Edit/etc.).
