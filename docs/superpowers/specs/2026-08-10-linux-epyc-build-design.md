# Linux Build Tuning for AMD EPYC 9V74 (Zen 4) — Design

## Purpose

Optimize the C/C++ host build for the development box's CPU — AMD EPYC 9V74
(Zen 4, 80 cores / 160 threads), while keeping the CUDA (NVCC) compilation
path and the existing Windows build untouched and functional.

The optimization only covers *build configuration* (compiler flags, LTO,
parallelism). No runtime/performance work is in scope.

## Decisions (approved)

1. **Fixed `-march=znver4`** — deterministic build, independent of the host;
   enables AVX-512F/BW/DQ/VL/IFMA/VNNI/BF16/VAES/GFNI in one flag.
2. **Thin LTO, release-only** — `-flto=thin`, gated by `LSX_LTO` cache var;
   never in debug, never on CUDA objects.
3. **Flags applied only to our targets** — via an INTERFACE library
   `lsx_tune`; Arrow/deps and `lsxcommon_cu` excluded.
4. **New preset `linux-release-znver4`** added; existing `linux-debug` /
   `linux-release` presets left working (they lose the old global
   `-march=native` but remain valid); Windows presets untouched.
5. **Ninja parallelism fixed to 160** via buildPreset `jobs`.

## Architecture

Root `CMakeLists.txt` introduces an INTERFACE library `lsx_tune` that carries
the tuning flags. It is linked PUBLIC into `lsxcommon`, so apps and tests
inherit it transitively. `lsxcommon_cu` (the CUDA OBJECT library) never links
it.

```cmake
# root CMakeLists.txt
add_library(lsx_tune INTERFACE)
target_compile_options(lsx_tune INTERFACE
    $<$<PLATFORM_ID:Linux>:$<$<COMPILE_LANGUAGE:C,CXX>:...flag set...>>
)
target_link_options(lsx_tune INTERFACE ...)
```

`lsxcommon`:

```cmake
target_link_libraries(lsxcommon PUBLIC lsx_tune)
```

`lsxcommon_cu` is untouched (keeps only `--extended-lambda`,
`-Xcompiler=/utf-8` on Windows).

## Compiler Flags (GCC 13+, Clang 16+ / AOCC)

Applied only when `LSX_ARCH_ZNVER4=ON` and only to C/C++ compilation on Linux:

| Flag | Rationale |
|---|---|
| `-march=znver4` | Zen 4 ISA: AVX-512F/BW/DQ/VL/IFMA/VNNI/BF16/VAES/GFNI; also brings VNNI + BF16 support in one flag |
| `-mtune=znver4` | explicit scheduling target for clarity |
| `-O3` | aggressive auto-vectorization (kept `-Og` in debug builds) |
| `-fno-math-errno` | relax FP errno contract for better vectorized FP code |
| `-falign-loops=16` | loop alignment (explicit user requirement) |
| `-funroll-loops` | unrolling for hot GEMM/norm kernels (enabled) |

A `LSX_LTO=ON` Release build additionally adds `-flto=thin` as both compile
and link option (see LTO section).

Explicitly NOT enabled: `-ffast-math` / `-funsafe-math-optimizations`
(risk of silently changing FP precision in the FP32 conversion kernels).

## CUDA Isolation (three layers)

1. **Remove the global `-march=native`** from `CMAKE_CXX_FLAGS`
   (currently `CMakeLists.txt:35`). The tuning flags now live only in
   `lsx_tune`.
2. **CUDA targets never link `lsx_tune`** — `lsxcommon_cu` keeps only its own
   options.
3. **Guard**: every tuning flag is wrapped in
   `$<$<COMPILE_LANGUAGE:C,CXX>:...>`. Even if `lsx_tune` were attached to a
   CUDA target by mistake, the generator expression would suppress the flags
   for `COMPILE_LANGUAGE:CUDA`.

LTO flags are additionally guarded with `$<$<LINK_LANGUAGE:CXX>:...>`, so
CUDA-linked executables never see `-flto`.

Invariant: no `-march=znver4`, `-flto*`, `-fno-math-errno`, or `-funroll-loops`
ever reaches the NVCC compile step or a CUDA device-link step.

## LTO

- Thin LTO (`-flto=thin`) only in the `linux-release-znver4` preset
  (`LSX_LTO=ON`).
- Applied as compile + link option, C/CXX-guarded, via `lsx_tune`.
- CUDA objects stay non-LTO; their fatbin is linked unchanged.
- AOCC / Clang: native `-flto=thin` support (identical LLVM IR) — no extra
  config.
- Windows: `/GL` + `/LTCG` NOT introduced (Windows build left as-is).

## Ninja Parallelism

- buildPreset sets `"jobs": 160` (EPYC 9V74: 80 physical cores / 160 threads;
  the user requested full 160-thread utilization).
- Ninja receives `-j160` from CMake. SMT2: 160 works well for the build; Ninja
  balances internally.
- Alternative without editing the preset:
  `export CMAKE_BUILD_PARALLEL_LEVEL=160` then
  `cmake --build --preset <name>` (environment variable overrides preset jobs
  only when the preset does not set `jobs`; keep in mind `jobs` here wins).

## CMakePresets.json v6 — new block

```json
{
  "name": "linux-release-znver4",
  "displayName": "Ubuntu (Ninja) - Release, EPYC 9V74 (Zen 4, 160t)",
  "generator": "Ninja",
  "binaryDir": "${sourceDir}/build/linux-release-znver4",
  "cacheVariables": {
    "CMAKE_BUILD_TYPE": "Release",
    "BUILD_TESTING": "ON",
    "LSX_ARCH_ZNVER4": "ON",
    "LSX_LTO": "ON"
  },
  "jobs": 160
}
```

Plus matching `buildPresets` and `testPresets` entries. Existing Linux presets
lose only the global `-march=native` (still valid / generic build). Windows
presets unchanged.

## Cache Vars

- `LSX_ARCH_ZNVER4` (BOOL, default OFF) — when ON, `lsx_tune` enables the
  `-march=znver4` flag set.
- `LSX_LTO` (BOOL, default OFF) — when ON (with a Release build), enable thin
  LTO. Intended for release presets only.

## Verification Plan

1. **Clean build** of `linux-release-znver4` preset; confirm:
   - `-march=znver4` on C/C++ objects,
   - NO `-march`/LTO on CUDA objects (`lsxcommon_cu`),
   - `-flto=thin` only on C/C++ objects and the final exe link.
2. **Flag inspection** via `compile_commands.json` (generated by default in
   Ninja / CMake 4): `.cu` rows must not contain `znver4`/`lto`; `.cpp` rows
   must.
3. **Tests**: `ctest --preset linux-release-znver4` — expect 11/12 (sole
   failure is the pre-existing `test_model_factory`, which needs a real DeepSeek
   checkpoint at a Windows path; unrelated).
4. **CUDA+LTO critical check**: `test_inference_kernels` (links CUDA objects)
   must build and pass with LTO on. If the fatbin link breaks, fallback: drop
   the LTO link flag for executables that link `lsxcommon_cu`.
5. **Windows regression**: flags guarded by `$<$<PLATFORM_ID:Linux>...>`;
   with `LSX_ARCH_ZNVER4=OFF` the build keeps prior behavior — verified by diff
   review (no VS available on this box).
6. **Tuning off switch**: build with `-DLSX_ARCH_ZNVER4=OFF` succeeds without
   `-march` → portability across CPUs proven.

## Files Touched

- `CMakeLists.txt` — remove global `-march=native`; add `lsx_tune` INTERFACE
  library + `LSX_ARCH_ZNVER4` / `LSX_LTO` logic.
- `CMakePresets.json` — add `linux-release-znver4` configure/build/test
  presets with `jobs: 160`.
- `AGENTS.md` — document the tuning flags / presets.
- `docs/superpowers/specs/2026-08-10-linux-epyc-build-design.md` — this spec.

## Out of Scope

- Runtine performance work (kernels, benchmark tuning).
- Windows build changes.
- Wrapper scripts (`build-aocc.sh`) — not requested.