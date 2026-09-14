---
name: cuda-cmake-build-gotchas
description: Use when a logestix build breaks or when setting up/wiring the CMake build — CUDA language enablement under CMake 4.x, host flags leaking into NVCC, MSVC/CCCL incompatibilities (/Zc:preprocessor, thrust), Arrow/Boost/xsimd patch-after-fetch failures, lsx_tune/LTO leaking into CUDA, or an AOCC/Clang host compiler. Triggers - CMake error, NVCC flag, C2794, C2938, unresolved policy, enable_language(CUDA), lsx_tune, LTO, сборка падает.
---

# CUDA C++ build gotchas

## Overview

One tree, two compilers: the host compiler and NVCC. Nearly every "mysterious"
build failure here is one of three things — a flag leaking from the host
compiler into NVCC, a CMake 4.x policy change, or a bundled dependency that
predates C++20. This skill collects the non-obvious ones so they are not
rediscovered.

Abseil-specific build patches (crc32 SIMD, format checker) live in the
`absl-in-cuda-cpp` skill — cross-referenced, not duplicated here.

## CMake

- Enable CUDA at the **top level**: `project(... LANGUAGES C CXX CUDA)` plus
  `find_package(CUDAToolkit REQUIRED)`. Letting a *nested* directory be the
  first to enable CUDA is what breaks under CMake 4.x, which is why the root
  does it. (The `enable_language(CUDA)` that `libs/lsxcommon/CMakeLists.txt`
  still calls is a harmless no-op precisely because the top level already
  enabled the language — not a licence to enable CUDA from a subdirectory.)
- Discover the toolkit with `find_package(CUDAToolkit)`; it works on Linux and
  Windows. Never hardcode `C:/Program Files/...` paths.
- Pre-C++-policy sub-dependencies need
  `CMAKE_POLICY_VERSION_MINIMUM=3.5` (Arrow's xsimd). Set it in
  `CMakePresets.json` `cacheVariables`; no manual env var.
- Set `CMAKE_RUNTIME_OUTPUT_DIRECTORY` to `${CMAKE_BINARY_DIR}/bin` **before**
  `add_subdirectory` — the `POST_BUILD` `arrow.dll` copies target that dir.
- **Isolate NVCC by putting every `.cu` into its own OBJECT library** and
  linking those into the static library. Host-only flags then cannot reach
  NVCC, and a `.cu` group can carry its own options (e.g. `-fmad=false` for
  bit-exact math).

## Flags that must never reach NVCC

- MSVC flags go into `CMAKE_CXX_FLAGS`, **never** `add_compile_options` (which
  NVCC also consumes): `/W4 /utf-8 /MP /arch:AVX512`, plus
  `WIN32_LEAN_AND_MEAN NOMINMAX` (the last two stop Windows macros colliding
  with CCCL). `/arch:AVX512` enables AVX-512F/BW/CD/DQ/VL.
- **Never `/Zc:preprocessor`** — it breaks CCCL/thrust. Use
  `CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING` +
  `THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP` for the MSVC/CCCL
  combination.
- Host tuning belongs in an **INTERFACE library** (`lsx_tune`), not global
  flags: `-march=znver4 -mtune=znver4 -O3 -fno-math-errno -falign-loops=16
  -funroll-loops` + LTO, enabled by `LSX_ARCH_ZNVER4` / `LSX_LTO`. It is wrapped
  in `$<$<PLATFORM_ID:Linux>:...$<COMPILE_LANGUAGE:C,CXX>>`, and LTO
  additionally in `$<$<LINK_LANGUAGE:CXX>:$<CONFIG:Release>>`, so nothing leaks
  into NVCC or MSVC. The generic `linux-debug`/`linux-release` presets stay
  `-march`-free.

## Dependency patches (idempotent, applied after FetchContent)

- **Arrow 25 + its Boost 1.88 on MSVC/C++20**: `literals.hpp` fails with
  `operator""_cppi` / `_cppui` sized-literal redefinition. Remove the four
  `BOOST_MP_DEFINE_SIZED_CPP_INT_LITERAL(...)` invocations after the Arrow
  download.
- **Arrow warnings on GCC 15**: set `BUILD_WARNING_LEVEL=PRODUCTION`. GCC 15
  promotes previously-ignored warnings to errors (`-Werror=unused-parameter`
  in Arrow 25 under the default CHECKIN debug level); PRODUCTION never adds
  `-Werror`.
- Pattern for every patch: search for a marker string and rewrite at configure
  time, because the pinned dependency tag re-fetches on a fresh configure (so
  the patch must be idempotent and re-applied each time). Same pattern is used
  by the abseil crc32/format-checker patches.

## CUDA codegen

- Architectures: `120-real;121-real` (Blackwell only). Hopper/Ada/Ampere are
  deliberately unsupported — there is no backward compatibility.
- `--extended-lambda` is required for `__host__ __device__` thrust lambdas.
- `cub::BlockReduce` / `cub::DeviceReduce` need CUDA 13.3+ on Linux; on MSVC
  use thrust as the fallback (thrust uses CUB internally).

## Host-only TUs that touch CUDA-surface headers

- A header that exposes CUDA types to both host and device TUs must guard the
  CUDA surface with `#if defined(LOGESTIX_HAS_CUDA) || defined(__CUDACC__)`;
  otherwise a host-only test TU (which has no CUDA include paths) fails to
  compile. Reference: `kimi_moe_expert_cache.h` / `test_kimi_moe_expert_cache`.
- In `.cu` tests never use the Windows-only `_putenv` unguarded — use
  `_putenv_s` / `setenv` under `#ifdef _WIN32` (reference:
  `test_qwen_nvfp4_gemv.cu`).
- CUDA-aware tests are built as `.cu` OBJECT libraries linked into an
  executable with `LINKER_LANGUAGE CXX`, so the test links like a host binary.

## AOCC / Clang as the CUDA host compiler

- Configure with `-DCMAKE_CUDA_HOST_COMPILER=$AOCC/clang++` alongside
  `-DCMAKE_C_COMPILER` / `-DCMAKE_CXX_COMPILER` (see AGENTS.md for the AOCC
  path and the full command).
- `-flto=thin` is **not** supported by AOCC clang 17: it warns
  `Unsupported lto mode -flto=thin, falling back to -flto=full` and builds
  correctly — benign, do not chase it.
- `__has_attribute(enable_if) == 1` on Clang/AOCC (0 on GCC) activates
  abseil's compile-time format checker, whose `ValidFormatImpl` understands
  only printf `%` conversions and rejects the fmt-style `{}` the project uses.
  The root CMakeLists patches it to GCC behaviour — **do not remove that patch
  for AOCC/Clang builds** (details: `absl-in-cuda-cpp`).

## Diagnosing by symptom

| Symptom | Cause |
|---|---|
| `C2794` / `C2938` in `raw_hash_map` while compiling a `.cu` | an abseil container became reachable from a device TU → `absl-in-cuda-cpp` |
| "cannot parse `__asm__`" / VPCLMULQDQ in abseil CRC on MSVC | `__AVX__` from `/arch:AVX512` (crc32 patch) |
| Host flag applied to device code | flag came from `add_compile_options` or an ungated INTERFACE library instead of `CMAKE_CXX_FLAGS` / generator expressions |
| Nested `enable_language(CUDA)` error under CMake 4.x | CUDA must be enabled at the top level |
| `cmake_minimum_required`/policy error from a fetched dependency | `CMAKE_POLICY_VERSION_MINIMUM=3.5` |
| Sized-literal redefinition in Boost `literals.hpp` under C++20 | Arrow/Boost patch not applied after fetch |
| `-Werror` in a third-party build under GCC 15 | missing `BUILD_WARNING_LEVEL=PRODUCTION` |
| Host-only test TU cannot find CUDA headers | the included header's CUDA surface is not `#if defined(LOGESTIX_HAS_CUDA) || defined(__CUDACC__)`-guarded |

## Verification

```bash
cmake --preset linux-debug && cmake --build --preset linux-debug   # generic
cmake --preset windows-debug && cmake --build --preset windows-debug
ctest --preset linux-debug -L fast
```

`build/<preset>` caches are persistent; a fresh configure re-fetches the pinned
dependencies and therefore re-applies every patch.
