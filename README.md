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
| CUDA cuBLAS | 13.3+ | system (CUDA Toolkit) | Dynamic (`cublas.dll`) |
| Catch2 | v3.7.0 | FetchContent (test only) | Header-only |

## Prerequisites

- CMake 3.15+
- C++20 / C11 compiler (GCC 15+, Clang 12+, MSVC 2026+)
- Git
- NVIDIA CUDA Toolkit 13.3+ (optional, for GPU support)
- CPU with AVX-512 (the Linux build is tuned for AMD EPYC 9V74 / Zen 4; binaries are not portable to pre-AVX-512 hardware)

## Clone with submodules

```bash
git clone --recurse-submodules https://github.com/Alex7MV/logestix.git
cd logestix
git submodule update --init --recursive
```

## Build

### Linux (GCC/Clang)

```bash
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The Linux build is tuned for AMD EPYC 9V74 (Zen 4): full AVX-512
(`-march=znver4`), explicit `-O3` in every build type, and Link-Time
Optimization on the project's own C++ targets (option `LOGESTIX_LTO`, default
ON). Toggle with `-DLOGESTIX_LTO=OFF` or `-DLOGESTIX_FAST_MATH=ON`. CUDA
`--testmodel` host code compiles with `-march=x86-64-v3` (AVX2) to avoid an
NVCC+GCC15 incompatibility with simdjson's AVX-512 path; compute lives in
device kernels targeting Blackwell (`compute_120/121`). CUDA toolkit is located
cross-platform from the `nvcc` path.

**AOCC mode** (AMD clang / LLVM): configure with
`-DCMAKE_CXX_COMPILER=/path/to/aocc/bin/clang++ -DCMAKE_C_COMPILER=/path/to/aocc/bin/clang`
(with the AOCC `bin` in `PATH` so `ld.lld` is found). Uses LLVM-LTO through
`lld`, Zen4 code alignment, and AMD LLVM knobs. Extra options:
`-DLOGESTIX_AGGRESSIVE_INLINE=ON`, `-DLOGESTIX_NO_STACK_PROTECTOR=ON`.
Detected automatically via the compiler ID; unaffected by Windows builds.

### Windows (Visual Studio 2026)

```powershell
git submodule update --init --recursive
cmake -S . -B build -G "Visual Studio 18 2026"
$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"
cmake --build build --config Debug
```

`arrow.dll` and `arrow_cuda.dll` are auto-copied via `POST_BUILD`. All executables land in `build/bin/Debug/`.

## Test

```bash
cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

On Windows:

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -DBUILD_TESTING=ON
$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"
cmake --build build --config Debug
ctest --test-dir build --output-on-failure -C Debug
```

41 test cases across 7 test executables (~410 assertions): `test_core`, `test_loader`, `test_tokenizer`, plus per-app tests for `lsxfabric`, `lsxforge`, `taskmosaic`, `lsxcli`.

## Structure

- `libs/lsxcommon/` — shared STATIC library; namespace `lsxcommon`
  - `core` — logging and initialization (spdlog)
  - `common` — version string, CUDA GPU enumeration
  - `schema` — `ExpertSchema` (9 fields: weights, E4M3 per-block scales, float32 global scales), `PackedExpert`, `ExpertData` structs
  - `converter` — GPU-only NVFP4 quantization (per-block E4M3 + global scale via CUDA), safetensors parsing, Arrow IPC serialization
  - `loader` — Arrow file loading, CUDA pinned memory registration, E4M3 scale decoding
  - `safetensors_gpu` — GPU-accelerated tensor quantization (3 CUDA kernels: convert, block_maxabs, quantize_pack_perblock; CCCL thrust for global max)
  - `testmodel_gpu` — `--testmodel` NVFP4 ping-pong benchmark (cuBLAS GEMM, dual-stream overlap, pinned staging)
  - `inference_*`, `norm_kernels`, `shared_experts` — inference pipeline (load/norm/moe/head/pipeline, layer-norm, NVFP4 shared experts)
- `apps/lsxfabric/` — B2B inference engine (`--localmodel`, `--validate`, `--testmodel`)
- `apps/lsxforge/` — data preprocessing utility (placeholder)
- `apps/taskmosaic/` — task orchestrator (placeholder)
- `apps/lsxcli/` — CLI client (`--model`)
- `tests/test_core.cpp` — integration tests (Arrow, CCCL, CUDA, FlatBuffers, ZeroMQ, simdjson, lsxcommon)
- `tests/test_loader.cpp` — loader round-trip and E4M3 encoding tests
- `tests/test_tokenizer.cpp` — tokenizer/pretokenizer/unicode tests
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
