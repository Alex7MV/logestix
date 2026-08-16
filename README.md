# logestix

Multi-app monorepo: **lsxfabric** (B2B inference engine), **lsxforge** (data preprocessing), **taskmosaic** (task orchestration), **lsxcli** (CLI client), **lsxhome** (Windows-native desktop shell). Shared core library `lsxcommon`.

| Dependency | Version | Source | Type |
|---|---|---|---|
| Apache Arrow | 25.0.0 | FetchContent (BUNDLED) | Shared (`arrow.dll`) |
| spdlog | 1.15.0 | FetchContent | Header-only |
| FlatBuffers | v25.12.19 | submodule (`third_party/flatbuffers`) | Static |
| libzmq | v4.3.5 | submodule (`third_party/libzmq`) | Static |
| simdjson | v4.4.2 | submodule (`third_party/simdjson`) | Static (`.lib`) |
| NVIDIA CCCL | latest | submodule (`third_party/cccl`) | Header-only |
| Catch2 | v3.7.0 | FetchContent (test only) | Header-only |
| Dear ImGui | 1.92.9-docking | FetchContent (lsxhome only) | Static |
| ImPlot | 1.0 | FetchContent (lsxhome only) | Static |
| ImNodes | 0.5 | FetchContent (lsxhome only) | Static |
| FreeType | 2.14.3 | FetchContent (lsxhome only) | Static |

## Prerequisites

- CMake 3.25+ (for CMakePresets)
- C++20 / C23 compiler (GCC 15+ on Ubuntu 26.04, MSVC in Visual Studio 2026)
- Ninja build system (required on Ubuntu; optional on Windows)
- Git
- NVIDIA CUDA Toolkit 13.2+ (optional, for GPU support)

## Clone with submodules

```bash
git clone --recurse-submodules https://github.com/Alex7MV/logestix.git
cd logestix
git submodule update --init --recursive
```

## Build

CMakePresets define ready-made configurations for both supported OSes. Replace
`debug` with `release` in the preset name for an optimization build.

### Windows 11 (Visual Studio 2026)

```powershell
cmake --preset windows-debug
cmake --build --preset windows-debug
```

Binaries land in `build/windows-debug/bin/Debug/`.

### Ubuntu 26.04 (Ninja + GCC)

```bash
cmake --preset linux-debug
cmake --build --preset linux-debug
```

Binaries land in `build/linux-debug/bin/`.

Linux release, tuned for AMD EPYC 9V74 (Zen 4), with LTO:

```bash
cmake --preset linux-release-znver4
cmake --build --preset linux-release-znver4   # Ninja -j160
ctest --preset linux-release-znver4
```

`arrow.dll` and `arrow_cuda.dll` are auto-copied via `POST_BUILD` on Windows.

## Test

`BUILD_TESTING=ON` is already enabled in all presets.

```powershell
ctest --preset windows-debug    # Windows 11
```

```bash
ctest --preset linux-debug      # Ubuntu 26.04
```

Test executables (~100 test cases, ~2.6M assertions). GLM-5.2 and DeepSeek
suites are autonomous and run independently:

```bash
ctest -R glm        # GLM-5.2 suite only (11 executables)
ctest -R deepseek   # DeepSeek suite only (2 executables)
```

Real-model tests SKIP gracefully when the checkpoint is absent
(`LSX_TEST_GLM_MODEL_PATH` for GLM-5.2, `LSX_TEST_MODEL_PATH` for DeepSeek).

## Structure

- `libs/lsxcommon/` — shared STATIC library; namespace `lsxcommon`
  - `core` — logging and initialization (spdlog)
  - `common` — version string, CUDA GPU enumeration
  - `schema` — `ExpertSchema` (9 fields: weights, E4M3 per-block scales, float32 global scales), `PackedExpert`, `ExpertData` structs
  - `converter` — GPU-only NVFP4 quantization (per-block E4M3 + global scale via CUDA), safetensors parsing, Arrow IPC serialization
  - `loader` — Arrow file loading, CUDA pinned memory registration, E4M3 scale decoding
  - `tokenizer` / `pretokenizer` / `unicode` — BPE tokenizer, word-level pre-tokenization, Unicode classification
  - `tokenizer_convert` — tokenizer Arrow files (string `"a b"` and array `["a","b"]` merges, byte_to_token_id, bos/eos/pad)
  - `safetensors_gpu` — GPU-accelerated tensor quantization (3 CUDA kernels: convert, block_maxabs, quantize_pack_perblock; CCCL thrust for global max)
  - `glm_convert` — GLM-5.2 classifier, Topology, `ConvertGlmModel`, `SaveGlmPinnedTensors`, `WriteGlmModelIndex` (extended `logestix-glm` format)
  - `glm_inference_load` — host GLM loader: `LoadGlmPinnedWeights` (BF16→FP16 upload), `GlmLayerWeights`/`GlmPinnedWeights`
  - `glm_v5_model` — `GlmV5Model` (IModel, registry `glm-5.2`); 79-layer Infer (dense 0-2 + IndexShare attention + MoE router + NVFP4 experts 3-77 + MTP 78)
  - `deepseek_v4_flash_model` — `DeepSeekV4FlashModel` (IModel, registry `deepseek-v4-flash`)
  - `model_factory` / `model_registry` — `LsxModelFactory` + IModel registry
  - `models/lsx_orchestrator` — TaskMosaic orchestrator: lock-free SPSC dispatch (GLM-5.2 / DeepSeek / unknown)
  - `inference_*.cu` — DeepSeek-V4-Flash CUDA kernels (load, norm, moe, head, attn, csa, pipeline)
  - `shared_experts.cu` — NVFP4 decode tables (E2M1/E8M0), routed-expert FFN
  - `glm_inference.cu` — GLM-5.2 CUDA kernels (router, IndexShare attention, NVFP4 expert decode)
  - `benchmark_gpu` — `--benchmark` pipeline: NVFP4 ping-pong benchmark
- `libs/lsxnetwork/` — zero-alloc async network layer for TaskMosaic CE (FrameQueue SPSC, ZMQ context, I/O worker, OpenAI endpoint, affinity)
- `apps/lsxfabric/` — B2B inference engine (`--localmodel`, `--validate`, `--benchmark`, `--testprompt`, `--selfcheck`)
  - `apps/lsxfabric/src/validator.cpp` — NVFP4/E4M3/global-scale validator
- `apps/lsxforge/` — data preprocessing utility; `--glm` for the GLM-5.2 pipeline, `--tokenizer`, repack flags
- `apps/taskmosaic/` — task orchestrator (CLI wrapper over `lsxcommon::LsxOrchestrator`)
- `apps/lsxcli/` — CLI client (`--model`, `--info`, `--infofull`)
- `apps/lsxhome/` — Windows-native x64 desktop shell (WIN32, D3D12 + Dear ImGui docking branch). Compute thread → lock-free SPSC `GuiBridge` → UI thread. Phase 1 theme layer (`ApplyBlackwellCoworkTheme` + `BuildWorkspaceSkeleton`) adds a matte Blackwell/Cowork styling profile.
  - `apps/lsxhome/tests/` — `test_lsxhome_gui` (5 test cases: payload alignment, SPSC zero-alloc/backpressure/burst) + `test_lsxhome_font` (3 test cases)
- `tests/` — Catch2 suite; GLM (`test_glm_*.cpp`) and DeepSeek (`test_deepseek_*.cpp`) test sets are autonomous
  - `test_core` — integration tests (Arrow, CCCL, CUDA, FlatBuffers, ZeroMQ, simdjson, lsxcommon)
  - `test_deepseek_loader` — loader round-trip and E4M3 encoding tests
  - `test_deepseek_inference_kernels` — DeepSeek CUDA inference kernel unit tests
  - `test_glm_*` — GLM-5.2 converter/loader/model/compute/inference/full-loop tests
  - `test_mosaic_network` — lsxnetwork (FrameQueue SPSC, zero-copy)
  - `test_integrated_pipeline` — TaskMosaic orchestrator (GLM + DeepSeek burst)
- `third_party/cccl/` — NVIDIA CCCL
- `third_party/flatbuffers/` — Google FlatBuffers v25.12.19
- `third_party/libzmq/` — ZeroMQ libzmq v4.3.5
- `third_party/simdjson/` — simdjson v4.4.2

## Licensing

This project is dual-licensed to accommodate both open-source development and commercial production environments.

### Open Source License
The open-source version of **logestix** is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](./LICENSE).

### Commercial License (Enterprise)
For commercial deployments, enterprise infrastructure, closed-source integration, or hosted services, the AGPL-3.0 limitations can be removed.

To obtain a commercial license or request a private evaluation copy (PoC), please contact us at: **<info@logestix.io>**

---

## Contributing

We welcome community contributions! However, to ensure we can protect our dual-licensing model, all contributors are required to sign our **Contributor License Agreement (CLA)** before any Pull Request can be merged.

The CLA signing process is automated via GitHub as soon as you open a Pull Request.
