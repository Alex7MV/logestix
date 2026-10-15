---
name: cuda-cmake-build-gotchas
description: Use when a logestix build breaks or when setting up/wiring the CMake build — CUDA language enablement under CMake 4.x, host flags leaking into NVCC, MSVC/CCCL incompatibilities (/Zc:preprocessor, thrust), Arrow/Boost/xsimd patch-after-fetch failures, lsx_tune/LTO leaking into CUDA, or an AOCC/Clang host compiler. Triggers - CMake error, NVCC flag, C2794, C2938, unresolved policy, enable_language(CUDA), lsx_tune, LTO, build fails.
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
- Pre-C++-policy sub-dependencies need `CMAKE_POLICY_VERSION_MINIMUM=3.5`
  (CMake ≥4.0). The root `CMakeLists.txt` sets it as a normal variable before
  `FetchContent` (inherited by in-process nested projects); the `windows-*`
  presets also carry it in `cacheVariables`; no manual env var.
- **Never inject `cmake_minimum_required(VERSION 3.5)`** to get CMake-4
  compatibility (e.g. via `CMAKE_PROJECT_INCLUDE`). Such a file is included as
  the last step of *every* `project()` call and resets that scope's policy
  version to 3.5, flipping CMP0076/CMP0091 — and all other post-3.5 policies —
  back to OLD. Symptom (VS generator only): Arrow's bundled uriparser aborts at
  its own `cmake_policy(GET CMP0091)` check ("behavior is '' rather than 'NEW',
  mis-compilation detected") and Boost fails to find its relative
  `extra/*.natvis` sources (CMP0076). The fix is to delete the injection and
  keep only `CMAKE_POLICY_VERSION_MINIMUM`.
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

- **Arrow's bundled Boost (1.92.0 in Arrow 26) on MSVC/C++20**: `literals.hpp` fails with
  `operator""_cppi` / `_cppui` sized-literal redefinition. Remove the four
  `BOOST_MP_DEFINE_SIZED_CPP_INT_LITERAL(...)` invocations after the Arrow
  download.
- **Arrow warnings on GCC 15**: set `BUILD_WARNING_LEVEL=PRODUCTION`. GCC 15
  promotes previously-ignored warnings to errors (`-Werror=unused-parameter`
  in Arrow under the default CHECKIN debug level); PRODUCTION never adds
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

## Bumping a pinned dependency

A version pin lives in three places that must change in the **same commit**, and
two of them are machine-enforced.

1. **The pin.**
   - FetchContent (root `CMakeLists.txt`) `GIT_TAG`: Arrow
     (`apache-arrow-26.0.0`), spdlog (`v1.17.0`), abseil (`20260817.0`), Catch2
     (`v3.16.1`).
   - Submodules under `third_party/`: flatbuffers, libzmq, simdjson — move the
     gitlink (`git -C third_party/<dep> checkout <tag>`) and commit the updated
     submodule pointer.
   - CCCL is **not** pinned: it comes from the CUDA Toolkit bundle
     (`<include>/cccl`) and is **band-guarded** (configure-time in the root
     `CMakeLists.txt` + the same band in `test_core.cpp`). A toolkit bump that
     moves CCCL outside the band fails configure; widen the band only after
     re-validating (fast suite + a device selfcheck).
2. **The guard** — `tests/test_core.cpp` → `TEST_CASE("pinned dependency
   versions")` asserts Arrow / spdlog / simdjson / Catch2 / abseil with
   `STATIC_REQUIRE` (abseil through `ABSL_LTS_RELEASE_VERSION`, which exists
   only on LTS tags — a non-LTS pin fails to compile by design), so a bump
   without updating it either fails to compile or fails the test. `flatbuffers`,
   `libzmq` and the system cuBLAS are **not** guarded — check those by hand.
3. **The documentation tables** — both `AGENTS.md` and `README.md` carry a
   dependency table; update them here too. They have drifted before: the CCCL row
   read "3.3 (latest)" while the pin was already at 3.4.2 (CCCL is now
   toolkit-provided and band-guarded, but the tables still track it).

### Then re-verify the patch-after-fetch blocks

Each patch is keyed to an **exact source string** and only rewrites when its
marker is absent — so a new upstream version can change the string and the patch
becomes a **silent no-op**:

| patch | target | what it does |
|---|---|---|
| `CMAKE_PROJECT_INCLUDE` policy file | Arrow / xsimd | forces `cmake_minimum_required(3.5)` so CMake 4.x does not reject the nested sub-project |
| Boost 1.92 multiprecision `literals.hpp` | MSVC + C++20 | removes the `operator""_cppi` sized-literal redefinition (a no-op with this tree's Arrow flags — Boost is never fetched; keep it for builds that turn Arrow's tests/testing on) |
| bundled fmt (in spdlog) | nvcc 13.x on Windows | replaces the `U"…"` `fractional_part_rounding_thresholds` literal (code points > U+10FFFF, rejected by the nvcc front end) with a `uint32_t` array; marker `gNvccRoundingThresholds` |
| abseil crc32 SIMD | MSVC + `/arch:AVX512` | disables the GCC/Clang-only VPCLMULQDQ inline asm |
| abseil str_format checker | Clang/AOCC | `#if 0 &&` so a fmt-style `{}` is not a hard compile error |

After a bump, confirm each rewritten file actually contains the patch (or that
the build no longer needs it). The spdlog/fmt patch's own comment still says
"against v1.15.0" — version-sensitive comments like that are the tell.

Finally, rebuild on **both** toolchains (MSVC and GCC/AOCC) because most of these
patches fire on only one of them, then run `ctest -L fast` for the guard.

## Diagnosing by symptom

| Symptom | Cause |
|---|---|
| `C2794` / `C2938` in `raw_hash_map` while compiling a `.cu` | an abseil container became reachable from a device TU → `absl-in-cuda-cpp` |
| "cannot parse `__asm__`" / VPCLMULQDQ in abseil CRC on MSVC | `__AVX__` from `/arch:AVX512` (crc32 patch) |
| Host flag applied to device code | flag came from `add_compile_options` or an ungated INTERFACE library instead of `CMAKE_CXX_FLAGS` / generator expressions |
| Nested `enable_language(CUDA)` error under CMake 4.x | CUDA must be enabled at the top level |
| `cmake_minimum_required`/policy error from a fetched dependency | `CMAKE_POLICY_VERSION_MINIMUM=3.5` |
| uriparser `SEND_ERROR "Policy CMP0091 behavior is '' rather than 'NEW'"`, or Boost "Cannot find source file … `.natvis`" (VS generator) | a `cmake_minimum_required(VERSION 3.5)` injected via `CMAKE_PROJECT_INCLUDE` lowered the nested sub-project's policies; delete the injection, keep only `CMAKE_POLICY_VERSION_MINIMUM` |
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
