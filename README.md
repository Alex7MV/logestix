# logestix

Multi-app monorepo: **lsxfabric** (B2B inference engine), **lsxforge** (data preprocessing), **taskmosaic** (task orchestration), **lsxcli** (CLI client). Shared core library `lsxcommon`.

| Dependency | Version | Source | Type |
|---|---|---|---|
| Apache Arrow | 25.0.0 | FetchContent (BUNDLED) | Shared (`arrow.dll`) |
| spdlog | 1.15.0 | FetchContent | Header-only |
| FlatBuffers | v25.12.19 | submodule (`third_party/flatbuffers`) | Static |
| libzmq | v4.3.5 | submodule (`third_party/libzmq`) | Static |
| simdjson | v4.4.2 | submodule (`third_party/simdjson`) | Static (`.lib`) |
| NVIDIA CCCL | latest | submodule (`third_party/cccl`) | Header-only |
| Catch2 | v3.7.0 | FetchContent (test only) | Header-only |

## Prerequisites

- CMake 3.25+ (for CMakePresets)
- C++20 / C11 compiler (GCC 13+ on Ubuntu 26.04, MSVC in Visual Studio 2026)
- Ninja build system (required on Ubuntu; optional on Windows)
- Git
- NVIDIA CUDA Toolkit 13.3+ (optional, for GPU support)

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

`arrow.dll` and `arrow_cuda.dll` are auto-copied via `POST_BUILD` on Windows.

## Test

`BUILD_TESTING=ON` is already enabled in all presets.

```powershell
ctest --preset windows-debug    # Windows 11
```

```bash
ctest --preset linux-debug      # Ubuntu 26.04
```

29 test cases across 6 test executables (~200 assertions).

## Structure

- `libs/lsxcommon/` — shared STATIC library; namespace `lsxcommon`
  - `core` — logging and initialization (spdlog)
  - `common` — version string, CUDA GPU enumeration
  - `schema` — `ExpertSchema` (9 fields: weights, E4M3 per-block scales, float32 global scales), `PackedExpert`, `ExpertData` structs
  - `converter` — GPU-only NVFP4 quantization (per-block E4M3 + global scale via CUDA), safetensors parsing, Arrow IPC serialization
  - `loader` — Arrow file loading, CUDA pinned memory registration, E4M3 scale decoding
  - `safetensors_gpu` — GPU-accelerated tensor quantization (3 CUDA kernels: convert, block_maxabs, quantize_pack_perblock; CCCL thrust for global max)
- `apps/lsxfabric/` — B2B inference engine (`--localmodel`, `--validate`)
- `apps/lsxforge/` — data preprocessing utility (placeholder)
- `apps/taskmosaic/` — task orchestrator (placeholder)
- `apps/lsxcli/` — CLI client (`--model`)
- `tests/test_core.cpp` — integration tests (Arrow, CCCL, CUDA, FlatBuffers, ZeroMQ, simdjson, lsxcommon)
- `tests/test_deepseek_loader.cpp` — loader round-trip and E4M3 encoding tests
- `tests/test_deepseek_inference_kernels.cu` — DeepSeek CUDA inference kernel unit tests
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
