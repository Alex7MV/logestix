# Ubuntu 26.04 Linux Build Max-Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt the CMake build for Ubuntu 26.04 by pinning to `znver4` with aggressive optimization flags, adding LTO to our C++ targets, and fixing the Linux-broken CUDA host flags — without touching the MSVC/Windows path.

**Architecture:** All target CPU/optimization flags are GCC/Clang-specific and live only in the non-MSVC `else()` branch of the root `CMakeLists.txt`. LTO is applied narrowly via CMake's `INTERPROCEDURAL_OPTIMIZATION` property (per-target) so CUDA/third-party code is excluded. The NVCC host-code flags in `libs/lsxcommon/CMakeLists.txt` get an MSVC guard and Linux-specific `-Xcompiler` flags.

**Tech Stack:** CMake 4.2.3, GCC 15.2.0, CUDA 13.2 / NVCC, `CheckIPOSupported`, EPYC 9V74 (`znver4`).

## Global Constraints

- Only the non-MSVC (`else()`) branch of `CMakeLists.txt` may change for optimizations. The MSVC/Windows flag set stays byte-for-byte identical.
- CPU pin: `-march=znver4 -mtune=znver4` (EPYC 9V74, Zen 4). Replaces `-march=native`.
- All optimization flags set via `CMAKE_CXX_FLAGS` (global, applies to bundled Arrow/simdjson too).
- `-ffast-math` off by default (`LOGESTIX_FAST_MATH=OFF`).
- LTO (`INTERPROCEDURAL_OPTIMIZATION`) only on `lsxcommon`, `lsxcli`, `lsxfabric`, `lsxforge`, `taskmosaic`. Never on `lsxcommon_cu` (NVCC) or third-party.
- `LOGESTIX_LTO` option, default ON, with graceful fallback if IPO unsupported.
- NVCC host flags: no LTO, only `-march=znver4 -mtune=znver4 -O3` via `-Xcompiler`.
- `-Xcompiler=/utf-8` is MSVC-only; must be guarded so GCC builds work.

---

### Task 1: Global Linux CPU/optimization flags in root CMakeLists

**Files:**
- Modify: `CMakeLists.txt:28-31` (the `else()` branch of the MSVC check)

**Interfaces:**
- Consumes: nothing.
- Produces: global `CMAKE_CXX_FLAGS` for the non-MSVC build — `-march=znver4 -mtune=znver4 -O3 -funroll-loops -fno-plt -fno-semantic-interposition`, plus the `LOGESTIX_FAST_MATH` option (default OFF).

- [ ] **Step 1: Replace the Linux flag block**

Replace lines 28-31 (the entire `else()` branch):

```cmake
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()
```

with:

```cmake
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    # Ubuntu 26.04 / AMD EPYC 9V74 (Zen 4): pin to znver4 for full AVX-512
    # (bf16/vnni/vbmi2) and tune for this microarchitecture. Aggressive
    # optimization for maximum throughput on the target machine.
    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -march=znver4 -mtune=znver4 -O3 -funroll-loops -fno-plt -fno-semantic-interposition")
    option(LOGESTIX_FAST_MATH "Enable -ffast-math (risky for NVFP4 dequant accuracy)" OFF)
    if(LOGESTIX_FAST_MATH)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
    endif()
endif()
```

- [ ] **Step 2: Verify the MSVC branch is untouched and flags apply**

Run: `git diff CMakeLists.txt`
Expected: only the `else()` branch changed; the `MSVC` block (lines 20-27) is untouched. The `-march=native` string no longer appears anywhere in `CMakeLists.txt`.

- [ ] **Step 3: Configure to confirm flags are accepted**

Run: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DLOGESTIX_LTO=OFF 2>&1 | tail -5`
Expected: configuration succeeds (may take minutes due to Arrow FetchContent), no `unrecognized command-line option` errors, no `unknown value for -march` errors.

- [ ] **Step 4: Verify flag propagation in the cache**

Run: `grep -F "CMAKE_CXX_FLAGS:" build/CMakeCache.txt`
Expected: the cache line shows `... -march=znver4 -mtune=znver4 -O3 -funroll-loops -fno-plt -fno-semantic-interposition`.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt
git commit -m "build: pin Linux CPU flags to znver4 for EPYC 9V74"
```

---

### Task 2: LTO on our C++ targets via INTERPROCEDURAL_OPTIMIZATION

**Files:**
- Modify: `CMakeLists.txt` — insert after the `add_subdirectory(apps)` line (currently line 172)

**Interfaces:**
- Consumes: targets `lsxcommon`, `lsxcli`, `lsxfabric`, `lsxforge`, `taskmosaic` defined by `add_subdirectory(apps)`.
- Produces: `LOGESTIX_LTO` option (default ON); `INTERPROCEDURAL_OPTIMIZATION TRUE` set on the 5 targets above when IPO is supported and not MSVC.

- [ ] **Step 1: Add the LTO block**

Insert immediately after `add_subdirectory(apps)`:

```cmake
add_subdirectory(apps)

# ── LTO для наших C++-целей (Linux/GCC) ──────────────────────────────────
# LTO точечно через свойство INTERPROCEDURAL_OPTIMIZATION: применяется только
# к нашему C++ коду и не задевает CUDA-объекты (lsxcommon_cu) и third-party.
include(CheckIPOSupported)
option(LOGESTIX_LTO "Enable LTO for our C++ targets" ON)
if(NOT MSVC AND LOGESTIX_LTO)
    check_ipo_supported(RESULT _lto_ok OUTPUT _lto_err)
    if(_lto_ok)
        foreach(_t IN ITEMS lsxcommon lsxcli lsxfabric lsxforge taskmosaic)
            set_property(TARGET ${_t} APPEND PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        endforeach()
    else()
        message(WARNING "LOGESTIX_LTO enabled but IPO unsupported: ${_lto_err}")
    endif()
endif()
```

- [ ] **Step 2: Re-configure with LTO on and confirm it's accepted**

Run: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DLOGESTIX_LTO=ON 2>&1 | tail -5`
Expected: configuration succeeds, no `INTERPROCEDURAL_OPTIMIZATION` property errors, no warning about unsupported IPO.

- [ ] **Step 3: Confirm the CMake machinery picks LTO**

Run: `grep -rn "flto" build/CMakeFiles/lsxcommon.dir/flags.make 2>/dev/null | head -3`
Expected: a `-flto` (or `-flto=auto`) flag appears in the compile flags of `lsxcommon` (LTO objects are LTO-compiled). If the path differs, locate `flags.make` under `build/CMakeFiles/` for any of the 5 targets.

- [ ] **Step 4: Commit**

```bash
git add CMakeLists.txt
git commit -m "build: enable LTO for our C++ targets via INTERPROCEDURAL_OPTIMIZATION"
```

---

### Task 3: Fix and optimize NVCC host-code flags

**Files:**
- Modify: `libs/lsxcommon/CMakeLists.txt:49`

**Interfaces:**
- Consumes: nothing.
- Produces: on Linux, `lsxcommon_cu` compiles `.cu` host code with `-Xcompiler=-march=znver4 -Xcompiler=-mtune=znver4 -Xcompiler=-O3`; on MSVC the original `--extended-lambda;-Xcompiler=/utf-8` behavior is preserved unchanged.

- [ ] **Step 1: Guard the MSVC flag and add Linux host flags**

Replace line 49:

```cmake
target_compile_options(lsxcommon_cu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-Xcompiler=/utf-8>)
```

with:

```cmake
if(MSVC)
    target_compile_options(lsxcommon_cu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-Xcompiler=/utf-8>)
else()
    target_compile_options(lsxcommon_cu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-Xcompiler=-march=znver4;-Xcompiler=-mtune=znver4;-Xcompiler=-O3>)
endif()
```

- [ ] **Step 2: Re-configure**

Run: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF 2>&1 | tail -5`
Expected: configuration succeeds; NVCC accepts the `-Xcompiler=-march=znver4` flags (GCC host compiler understands them).

- [ ] **Step 3: Commit**

```bash
git add libs/lsxcommon/CMakeLists.txt
git commit -m "build: add znver4 host flags for NVCC and guard MSVC /utf-8"
```

---

### Task 4: Full build + test verification

**Files:**
- No source changes. Verification only.

**Interfaces:**
- Consumes: the full configured build tree from Tasks 1-3.

- [ ] **Step 1: Clean re-configure with testing enabled**

Run: `rm -rf build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON 2>&1 | tail -5`
Expected: clean configuration succeeds with `LOGESTIX_LTO=ON` default and fast-math off.

- [ ] **Step 2: Build the whole project**

Run: `cmake --build build -j"$(nproc)" 2>&1 | tail -20`
Expected: all 4 apps, `lsxcommon`, `lsxcommon_cu`, and the 5 test targets compile and link without errors (no `/utf-8`-style GCC errors, no LTO link failures with the non-LTO NVCC objects). This step may take a long time (Arrow rebuild).

- [ ] **Step 3: Run the test suite**

Run: `ctest --test-dir build --output-on-failure`
Expected: all tests pass.

- [ ] **Step 4: Confirm the CUDA objects were linked LTO-free (mixed-link sanity)**

Run: `nm -C build/bin/lsxfabric 2>/dev/null | grep -m1 "testmodel_gpu\|safetensors_gpu" || true`
Expected: symbols from the CUDA objects resolve (no link-time "LTO object" errors occurred, since the build succeeded in Step 2).

- [ ] **Step 5: Final commit for any incidental fixes**

If any of the steps above revealed an incidental build fix, commit it now with an appropriate `build:`/`fix:` message. If nothing else changed, no commit.

---