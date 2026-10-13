---
name: building-the-project
description: Use when configuring, building or locating binaries in logestix — CMake presets (linux-debug/release-znver4/windows-debug), the AOCC host-compiler build, the Windows VS 2026 toolchain paths, toolchain minimums, and where the executables land. Triggers - build, preset, AOCC, windows toolchain, cmake not found.
---

# Building the project

## Overview

Every supported configuration is a CMake preset; the ADR-style CMake machinery
is already wired (`CMakePresets.json`). This skill is the build **manual**: which
preset answers which question, how to build with an alternative host compiler,
where the Windows toolchain lives, and where binaries land. Failure diagnosis
(flags leaking into NVCC, CMake 4.x policies, dependency patches) is the
`cuda-cmake-build-gotchas` skill.

## Presets

Configure presets: `linux-debug`, `linux-release`, `linux-release-znver4`
(Zen 4 + LTO), `windows-debug`, `windows-release`. Every configure preset has a
matching **test** preset; the base test preset filters the `fast` label, and a
`-fast` / `-convert` / `-inference` / `-perf` suffix selects one label category
(e.g. `linux-debug-convert`, `linux-release-znver4-perf`). See `test-lifecycle`.

| need | preset |
|---|---|
| quick host-only iteration / fast tests | `linux-debug` |
| perf reference numbers | `linux-release-znver4` (or the AOCC tree) |
| Windows build | `windows-debug` / `windows-release` |

## Linux

```bash
git submodule update --init --recursive
cmake --preset linux-debug && cmake --build --preset linux-debug
ctest --preset linux-debug                     # fast only

# release, tuned for AMD EPYC 9V74 (Zen 4), with LTO:
cmake --preset linux-release-znver4 && cmake --build --preset linux-release-znver4
ctest --preset linux-release-znver4
```

`LSX_ARCH_ZNVER4=ON` + `LSX_LTO=ON` are baked into the znver4 preset. The
generic `linux-debug`/`linux-release` presets stay `-march`-free.

## Linux release — AOCC

Alternative host compiler (clang 17.0.6) used as NVCC's `-ccbin`:

```bash
AOCC=$HOME/aocc-compiler-5.2.0/bin   # point at your AOCC install
cmake -S . -B build/linux-release-znver4-aocc -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DLSX_ARCH_ZNVER4=ON -DLSX_LTO=ON \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_C_COMPILER=$AOCC/clang -DCMAKE_CXX_COMPILER=$AOCC/clang++ \
  -DCMAKE_CUDA_HOST_COMPILER=$AOCC/clang++
cmake --build build/linux-release-znver4-aocc -j 160
ctest --test-dir build/linux-release-znver4-aocc
```

### AOCC: two flags the tree needs on CMake 4.2 + Arrow 26.0.0

Configuring this tree with the command above **fails at the Generate step** on
CMake 4.2:

```
The install of the arrow_shared target requires changing an RPATH from the build
tree, but this is not supported with the Ninja generator unless on an ELF-based
or XCOFF-based platform.
```

This is not caused by any project change — it reproduces on a pristine
`tests/CMakeLists.txt`. Two additions make the tree generate, build and run:

```bash
AR=$PWD/build/linux-release-znver4-aocc/_deps/arrow-build/release
cmake ... -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH="$AR" \
          -DCMAKE_CXX_FLAGS_RELEASE="-DNDEBUG"
```

Both are load-bearing, and dropping either fails differently:

- **`BUILD_WITH_INSTALL_RPATH=ON` alone is a trap.** It fixes the Generate step but
  the binaries then cannot load `libarrow.so.2600` / `libarrow_cuda.so.2600`
  (`error while loading shared libraries`) — the build-tree RPATH that used to point
  at `_deps/arrow-build/release` is gone. Hence the explicit `CMAKE_INSTALL_RPATH`
  pointing at that same directory. Verify with
  `ldd <binary> | grep -c "not found"` → must be `0`; a green build and a green
  ctest are not proof, because the failure is a loader error at exec time.
- **`CMAKE_CXX_FLAGS_RELEASE` is empty in this tree's cache**, not the CMake default
  `-O3 -DNDEBUG`. Left empty, `test_perf_baseline` compiles without `NDEBUG` and its
  own Release guard makes every perf test `SKIP` with *"perf baseline requires a
  Release build (NDEBUG)"* — which reads like "no weights in the vault" and is not.
  Pass `-DNDEBUG` only (not `-O3 -DNDEBUG`): `-O3` and `-flto=thin` are appended
  separately by the `lsx_tune` path, and overwriting the whole variable fights it.
  Note AOCC clang 17 warns `Unsupported lto mode -flto=thin, falling back to
  -flto=full`; that is expected.

Also: `-DCMAKE_C_COMPILER` must be **omitted** for this tree. Passing a bad path
poisons the cache with `CMAKE_C_COMPILER:UNINITIALIZED` and CMake then loses the ELF
platform detection that the Arrow RPATH check depends on, which surfaces as the same
confusing RPATH error one step later. Point only `CMAKE_CXX_COMPILER` and
`CMAKE_CUDA_HOST_COMPILER` at AOCC; `AOCC` lives in `~/projects/amd/`, **not** in
`$HOME` — a `$HOME/aocc-compiler-5.2.0` path does not exist.

This tree (`build/linux-release-znver4-aocc`) is the perf reference. AOCC/Clang
specifics (`-flto=thin` fallback, the abseil format-checker patch) are in
`cuda-cmake-build-gotchas`.

## Windows — VS 2026 toolchain

`cmake`/`ctest`/`ninja` are **not in PATH**; Visual Studio 2026 provides them,
and CUDA Toolkit 13.4 sits separately. The `build\` cache is pre-configured with
the `windows-debug` preset (WIN32 generator).

- CMake 4.3: `"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"`
- CTest: `"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe"`
- Ninja: `"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"`
- MSVC `cl.exe` (Hostx64): `"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\cl.exe"`
- NVCC 13.4 (V13.4.92): `"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe"` (already in PATH; CMake finds it via `find_package(CUDAToolkit)`)

PowerShell without touching PATH:

```powershell
$cmake = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
& $cmake --build build --config Debug --target <target>
$cmake --preset windows-debug && $cmake --build --preset windows-debug
```

When VS updates, re-check the MSVC directory number with
`Get-ChildItem "$vs\VC\Tools\MSVC"`. `arrow.dll` / `arrow_cuda.dll` are copied to
the executable directory by a `POST_BUILD` step on Win32.

## Configuring from scratch on Windows (PowerShell)

Two traps, both hit while re-pointing an existing `build\` tree at a new CUDA
Toolkit:

1. **Quote every `-D`.** An unquoted `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` is
   mangled by PowerShell and reaches CMake as `3`, which 4.3 rejects with
   "Invalid CMAKE_POLICY_VERSION_MINIMUM value" against every nested
   `cmake_minimum_required`. Use `"–DCMAKE_POLICY_VERSION_MINIMUM=3.5"`.
2. **Do not hand-edit `CMakeCache.txt`.** Removing a toolkit-path entry leaves
   its `//Path to a …` header orphaned, and CMake then fails with a parse
   error. Pin the toolkit properly instead and re-configure:
   ```powershell
   & $cmake -S . -B build -G "Visual Studio 18 2026" `
       "-DBUILD_TESTING=ON" "-DCMAKE_POLICY_VERSION_MINIMUM=3.5" `
       "-DCUDAToolkit_ROOT=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4"
   ```
   A cache written while an older Toolkit was installed keeps its old
   `CUDAToolkit_*` paths (they are valid cache entries, so a new
   `CUDAToolkit_ROOT` does not re-derive them). The symptom is silent and
   nasty: headers from the old Toolkit against the new runtime, caught by
   `test_cuda_toolkit_guard` as `CUDART_VERSION=13030 runtime=13040`. If the
   cache is already inconsistent, deleting `build/` is cheaper than repairing
   it — it is gitignored, and the full re-configure takes ~4 min.

Do not pass `-A x64`: the VS generator picks the platform itself, and setting
it makes CMake fail with a bare "Error: generator platform: x64".

## Toolchain minimums

Tested versions; newer also work: CMake 3.25+ (tree used 4.2.3), GCC 15.2
(Ubuntu 26.04), NVCC 13.4.92 (Linux, `/usr/local/cuda`) / 13.4.92 (Windows), CUDA
Toolkit ≥13.2 (guard `test_cuda_toolkit_guard`, which pins only the 13.2 floor
and the 13.x major -- patch levels pass through it). The project is deliberately
pinned to C++20 (NVCC also ships full C++23).

## Output

`CMAKE_RUNTIME_OUTPUT_DIRECTORY` is set to `${CMAKE_BINARY_DIR}/bin` **only on
Win32** (before `add_subdirectory`), so:

- Linux: `build/<preset>/apps/<app>/<app>` (e.g.
  `build/linux-debug/apps/lsxbenchmark/lsxbenchmark`).
- Windows: `build/windows-debug/bin/Debug/<app>.exe`.

`arrow.dll` / `arrow_cuda.dll` are copied next to the executables by a
`POST_BUILD` step on Win32.

## Common mistakes

- Assuming Linux binaries are in `build/linux-debug/bin/` — they are under
  `build/<preset>/apps/<app>/`; only Win32 redirects to `bin/`.
- Building the perf tree in Debug and comparing numbers.
- Configuring AOCC without `-DCMAKE_CUDA_HOST_COMPILER` — NVCC then uses a
  different host compiler than the `.cpp` objects.
- Expecting `cmake` on PATH under Windows (it is not).
- Treating this skill as the place to debug a CMake error — that is
  `cuda-cmake-build-gotchas`.
