# Linux EPYC (Zen 4) Build Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Linux host build target the AMD EPYC 9V74 (Zen 4, 160 threads) via a fixed `-march=znver4` + LTO flag set that only ever touches C/C++ targets, keeping the CUDA (NVCC) compile path and the entire Windows build byte-for-byte untouched.

**Architecture:** A new INTERFACE library `lsx_tune` in the root `CMakeLists.txt` carries the tuning flags guarded by `$<$<AND:$<PLATFORM_ID:Linux>,$<COMPILE_LANGUAGE:C,CXX>>:...>` (LTO additionally by `$<$<LINK_LANGUAGE:CXX>:...>` and `$<CONFIG:Release>`). It is linked `PUBLIC` into `lsxcommon`, so apps/tests inherit it transitively, but the CUDA OBJECT library `lsxcommon_cu` and all `add_subdirectory` dependencies (Arrow, spdlog, simdjson, FlatBuffers, libzmq) never see it. The old global `-march=native` in `CMAKE_CXX_FLAGS` is removed. A new preset `linux-release-znver4` wires `LSX_ARCH_ZNVER4=ON`+`LSX_LTO=ON` with Ninja `jobs: 160`.

**Tech Stack:** CMake 4.2.3 / CMake ≥ 3.25 (CMakePresets v6), Ninja, GCC 15.2 (bis 13+), Clang 16+ / AOCC, NVCC 13.2+ (Blackwell CC 12.0/12.1 only).

## Global Constraints

- C++20 baseline (project fixed); CUDA `CUDA_ARCHITECTURES "120-real;121-real"` only — Hopper/Ada/Ampere rejected.
- All tuning flags go ONLY through the INTERFACE library `lsx_tune`; never through `CMAKE_CXX_FLAGS` or `add_compile_options` at root.
- Tuning flags apply only to C/C++ compilation on Linux: guarded by `$<$<AND:$<PLATFORM_ID:Linux>,$<COMPILE_LANGUAGE:C,CXX>>:...>`.
- LTO flags additionally guarded by `$<$<LINK_LANGUAGE:CXX>:...>` and `$<CONFIG:Release>` — never on CUDA compile, never in Debug.
- CUDA targets (`lsxcommon_cu`) and FetchContent/submodule dependencies (Arrow, spdlog, simdjson, FlatBuffers, libzmq) must NOT receive `-march`, `-m*`, `-fno-math-errno`, `-falign-loops`, `-funroll-loops`, `-O3`, `-flto*`, or `-mtune`.
- Windows (MSVC) build must remain untouched: every flag is behind `$<PLATFORM_ID:Linux>`; no changes to windows presets or MSVC-specific flags.
- Flag mapping: GCC → `-flto` (no `=thin`); Clang/AOCC → `-flto=thin`. `-march=znver4` supported by GCC ≥ 12 and Clang/AOCC (LLVM) — both present here.
- `LSX_ARCH_ZNVER4`/`LSX_LTO` default OFF; only the `linux-release-znver4` preset enables them.
- Test expectation on this box: ctest `11/12` — the only failure is the pre-existing `test_model_factory` (`Create + Load + Inspect on real model dir` needs the real DeepSeek-V4-Flash checkpoint at a Windows path `D:\Programs\lsx_model\...`; unrelated to this work).

---

### Task 1: Add `linux-release-znver4` preset (configure/build/test) to CMakePresets.json

**Files:**
- Modify: `CMakePresets.json`

**Interfaces:**
- Consumes: none.
- Produces: preset names `linux-release-znver4` (configure/build/test). Configures the build into `${sourceDir}/build/linux-release-znver4` with `CMAKE_BUILD_TYPE=Release`, `BUILD_TESTING=ON`, `CMAKE_EXPORT_COMPILE_COMMANDS=ON`, `LSX_ARCH_ZNVER4=ON`, `LSX_LTO=ON`; build preset pins Ninja `"jobs": 160`.
- The cache vars `LSX_ARCH_ZNVER4`/`LSX_LTO` are set here before the root `CMakeLists.txt` defines them via `option()` — `option()` never overrides an existing cache entry, so preset-configured values win. Task 2 adds the actual `option()` calls.

- [ ] **Step 1: Edit `configurePresets`**

In `CMakePresets.json`, insert this entry after the `linux-release` configure preset (after line 47):

```json
    {
      "name": "linux-release-znver4",
      "displayName": "Ubuntu 26.04 (Ninja) - Release, EPYC 9V74 (Zen 4, 160t)",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/linux-release-znver4",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "BUILD_TESTING": "ON",
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
        "LSX_ARCH_ZNVER4": "ON",
        "LSX_LTO": "ON"
      }
    }
```

Ensure the previous array element (`linux-release`) gets a trailing comma.

- [ ] **Step 2: Edit `buildPresets`**

Add after the `linux-release` build preset (after line 54):

```json
    { "name": "linux-release-znver4", "configurePreset": "linux-release-znver4", "jobs": 160 },
```

- [ ] **Step 3: Edit `testPresets`**

Add after the `linux-release` test preset (after line 61):

```json
    { "name": "linux-release-znver4", "configurePreset": "linux-release-znver4", "output": { "outputOnFailure": true } }
```

- [ ] **Step 4: Verify preset is visible and configures (fetches Arrow once here)**

Run:
```bash
cmake --list-presets | grep linux
```
Expected: `linux-debug`, `linux-release`, `linux-release-znver4` listed.

Run:
```bash
cmake --preset linux-release-znver4 2>&1 | tail -5
```
Expected: configure succeeds; the last lines reference generation of build files. Arrow/spdlog/Catch2 FetchContent clones happen at this step.
Note: if the output instead ends with an error about `LSX_ARCH_ZNVER4` or `LSX_LTO` not being recognized, that is fine and expected only if `cmake --fail-on-warnings` is involved — CMake tolerates unknown/unused directory-scoped variables; the `option()` declarations land in Task 2.

- [ ] **Step 5: Commit**

```bash
git add CMakePresets.json
git commit -m "build: add linux-release-znver4 preset (EPYC 9V74, jobs 160)"
```

---

### Task 2: Root CMakeLists.txt — remove global `-march=native`, add `lsx_tune` INTERFACE library (arch flags)

**Files:**
- Modify: `CMakeLists.txt:23-36` (compiler-flags block) and insert the `lsx_tune` block + `option()` calls right after it (before the Arrow FetchContent section at line 38).

**Interfaces:**
- Consumes: cache vars `LSX_ARCH_ZNVER4`, `LSX_LTO` (set by the preset in Task 1).
- Produces: target `lsx_tune` (INTERFACE) carrying `-march=znver4 -mtune=znver4 -fno-math-errno -falign-loops=16 -funroll-loops` (+`-O3` in Release) when `LSX_ARCH_ZNVER4=ON` on Linux. Later tasks link it into `lsxcommon`.
- Precondition: `lsx_tune` must be defined before `add_subdirectory(libs)` (line 180) — it is, since it sits in the flags section near the top.

- [ ] **Step 1: Replace the compiler-flags `else()` branch**

Current (lines 33-36):

```cmake
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()
```

Replace with:

```cmake
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()
```

The global `-march=native` is gone. Default (non-znver4) Linux builds now carry no `-march` at all — generic build; the tuned build opts in via `LSX_ARCH_ZNVER4`.

- [ ] **Step 2: Insert the `lsx_tune` block after the flags `endif()`**

Immediately after the truncated `else()...endif()` above (i.e. before `# ── DEPENDENCIES: Apache Arrow...` at line 38), insert:

```cmake
# ── Linux CPU-tuned build: AMD EPYC 9V74 (Zen 4) ─────────────────────────────
# `lsx_tune` — an INTERFACE library carrying aggressive C/C++ optimization flags
# for the EPYC 9V74. It is linked PUBLIC into `lsxcommon` (and through it —
# transitively into all apps and tests). CUDA objects (`lsxcommon_cu`), FetchContent
# and submodule dependencies (Arrow, spdlog, simdjson, FlatBuffers, libzmq) do NOT
# get these flags: everything is wrapped in $<$<COMPILE_LANGUAGE:C,CXX>:...> plus
# $<$<PLATFORM_ID:Linux>:...>. MSVC and Apple are unaffected.
# Enabled via LSX_ARCH_ZNVER4 / LSX_LTO (see the linux-release-znver4 preset).
option(LSX_ARCH_ZNVER4 "Tune Linux C/C++ build for AMD EPYC 9V74 (Zen 4)" OFF)
option(LSX_LTO "Enable LTO for Linux Release C/C++ (GCC: -flto; Clang/AOCC: -flto=thin)" OFF)

add_library(lsx_tune INTERFACE)

if(LSX_ARCH_ZNVER4 AND UNIX AND NOT APPLE)
    target_compile_options(lsx_tune INTERFACE
        $<$<AND:$<PLATFORM_ID:Linux>,$<COMPILE_LANGUAGE:C,CXX>>:-march=znver4>
        $<$<AND:$<PLATFORM_ID:Linux>,$<COMPILE_LANGUAGE:C,CXX>>:-mtune=znver4>
        $<$<AND:$<PLATFORM_ID:Linux>,$<COMPILE_LANGUAGE:C,CXX>>:-fno-math-errno>
        $<$<AND:$<PLATFORM_ID:Linux>,$<COMPILE_LANGUAGE:C,CXX>>:-falign-loops=16>
        $<$<AND:$<PLATFORM_ID:Linux>,$<COMPILE_LANGUAGE:C,CXX>>:-funroll-loops>
        $<$<AND:$<PLATFORM_ID:Linux>,$<COMPILE_LANGUAGE:C,CXX>,$<CONFIG:Release>>:-O3>
    )
endif()
```

- [ ] **Step 3: Verify the gateless-off build still configures**

Run (reconfigures the existing default dir; generic flags, no tuning):
```bash
cmake --preset linux-debug 2>&1 | tail -3
```
Expected: configure succeeds, no `znver4`, no `-march`.

- [ ] **Step 4: Verify CUDA isolation — reconfigure the znver4 preset**

At this task `lsx_tune` is only *defined*, not yet linked into `lsxcommon`
(that happens in Task 3), so our own `.cpp` rows do NOT carry the flags yet —
that is expected at this stage. What IS verifiable now: the nvcc lines must be
clean. Run:

```bash
cmake --preset linux-release-znver4 2>&1 | tail -3

python3 - <<'EOF'
import json
db = json.load(open("build/linux-release-znver4/compile_commands.json"))
cu = [e for e in db if e["file"].endswith(".cu")]
leak = [e["file"] for e in cu if any(t.startswith(("-march=", "-mtune=")) for t in e["command"].split())]
print("cu rows:", len(cu), "CUDA -march/-mtune flag leaks:", leak)
assert not leak
print("OK: nvcc lines carry no -march/-mtune flags (only --generate-code=arch/sm_120,121)")
EOF
```
Expected: `OK`. This confirms isolation of the flags from CUDA even before the
link step. The positive check (`-march=znver4` present on our C/C++ targets)
runs in Task 3 Step 3, after the PUBLIC link into `lsxcommon`.

- [ ] **Step 5: Smoke-compile `lsxcommon` (C++ + CUDA objects) with the tuning on**

Run:
```bash
cmake --build build/linux-release-znver4 --target lsxcommon --parallel 160 2>&1 | tail -5
```
Expected: build finishes (this compiles both the C++ `lsxcommon` sources under `-march=znver4` and all 11 `.cu` files via NVCC untouched); no `-flto` involved yet.

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt
git commit -m "build: add lsx_tune INTERFACE lib, drop global -march=native (EPYC znver4)"
```

---

### Task 3: Link `lsx_tune` PUBLIC into `lsxcommon` + add LTO logic

**Files:**
- Modify: `CMakeLists.txt` (add `LSX_LTO` block inside the `lsx_tune` section)
- Modify: `libs/lsxcommon/CMakeLists.txt:87`

**Interfaces:**
- Consumes: `lsx_tune` target (Task 2).
- Produces: `lsxcommon` PUBLIC linking `lsx_tune` → apps and test executables get tuning flags transitively at compile time and `-flto` at their final CXX link. `lsxcommon_cu` remains a standalone OBJECT lib with its own options only.

- [ ] **Step 1: Add the LTO block to the `lsx_tune` section**

Inside `CMakeLists.txt`, after the `if(LSX_ARCH_ZNVER4 ...)` `target_compile_options` block, append:

```cmake
if(LSX_LTO AND UNIX AND NOT APPLE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(LSX_LTO_FLAG "-flto=thin")
    else()
        set(LSX_LTO_FLAG "-flto")
    endif()
    target_compile_options(lsx_tune INTERFACE
        $<$<AND:$<PLATFORM_ID:Linux>,$<COMPILE_LANGUAGE:C,CXX>,$<CONFIG:Release>>:${LSX_LTO_FLAG}>
    )
    target_link_options(lsx_tune INTERFACE
        $<$<AND:$<PLATFORM_ID:Linux>,$<LINK_LANGUAGE:CXX>,$<CONFIG:Release>>:${LSX_LTO_FLAG}>
    )
endif()
```

Note: `$<LINK_LANGUAGE:CXX>` in `LINK_OPTIONS` is supported since CMake 3.18 — we are on 4.2.3. The `LINK_LANGUAGE` guard means `-flto` never reaches a device-link step or a CUDA-link-language target such as `test_inference_kernels`.

- [ ] **Step 2: Link `lsx_tune` PUBLIC into `lsxcommon`**

In `libs/lsxcommon/CMakeLists.txt`, change line 87:

```cmake
target_link_libraries(lsxcommon PUBLIC lsxcommon_cu)
```

to:

```cmake
target_link_libraries(lsxcommon PUBLIC lsxcommon_cu lsx_tune)
```

This is what lets `-march=znver4`/`-mtune=znver4`/`-flto` reach every app and test executable (they link `lsxcommon` PRIVATE, and PUBLIC usage requirements propagate transitively).

- [ ] **Step 3: Reconfigure and verify compile + link flags**

Run:
```bash
cmake --preset linux-release-znver4 2>&1 | tail -3
python3 - <<'EOF'
import json
db = json.load(open("build/linux-release-znver4/compile_commands.json"))
cpp = [e for e in db if e["file"].endswith((".cpp", ".c")) and "third_party" not in e["file"]]
cu  = [e for e in db if e["file"].endswith(".cu")]
assert all("-march=znver4" in e["command"] and ("-flto" in e["command"]) for e in cpp), "cpp missing znver4/flto"
assert all("-flto" not in e["command"] and "znver4" not in e["command"] for e in cu), "flto/znver4 on CUDA"
print("compile OK: cpp have -march=znver4 + -flto; cu clean")
EOF
```

Then check the final link line of an app target picks up `-flto`:
```bash
cat build/linux-release-znver4/apps/lsxcli/CMakeFiles/lsxcli.dir/link.txt | tr ' ' '\n' | grep -x -- '-flto'
```
Expected: prints `-flto` (GCC) — proving the link option propagated through `lsxcommon`.

- [ ] **Step 4: Configure the Debug preset again to prove LTO/none of this leaks into Debug**

Run:
```bash
cmake --preset linux-debug 2>&1 | tail -3
grep -r -- '-flto' build/linux-debug/compile_commands.json | head -3
```
Expected: configure succeeds; grep prints nothing (`linux-debug` keeps `LSX_LTO` unset → no `-flto` anywhere).

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt libs/lsxcommon/CMakeLists.txt
git commit -m "build: PUBLIC-link lsx_tune into lsxcommon, add Release LTO"
```

---

### Task 4: Full clean build + ctest on the `linux-release-znver4` preset

**Files:**
- None (build/verification only).

**Interfaces:**
- Consumes: everything from Tasks 1-3.

- [ ] **Step 1: Full clean build at 160 threads**

Run:
```bash
cmake --build --preset linux-release-znver4 2>&1 | tail -15
```
Expected: builds all 4 apps + 12 test executables with `-march=znver4` (and `-flto`), including the CUDA-heavy `test_inference_kernels` — this is the critical "CUDA fatbin + LTO link" check.

- [ ] **Step 2: Run the whole test suite**

Run:
```bash
ctest --preset linux-release-znver4 2>&1 | tail -20
```
Expected: 12 tests, 11 PASS, 1 FAIL.

The single failure is `test_model_factory` — pre-existing and unrelated (`Create + Load + Inspect on real model dir` requires the real DeepSeek-V4-Flash checkpoint at `D:\Programs\lsx_model\DeepSeek-V4-Flash`; not available on Linux). Confirm it is the ONLY failure:
```bash
ctest --preset linux-release-znver4 2>&1 | grep -E "tests passed|Failed"
```
Expected: `11 tests passed out of 12 tests failed (71%)`-style summary where Failed = 1, plus no other failure lines.

- [ ] **Step 3: (Fallback, only if step 2 shows CUDA+LTO link breakage)**

If `test_inference_kernels` fails at build time with LTO errors around fatbin/`sm_120`, roll the executables that link CUDA objects off LTO:

```cmake
# in libs/lsxcommon/CMakeLists.txt, adjust Task 3 Step 2 to:
target_link_libraries(lsxcommon PUBLIC lsxcommon_cu)
target_link_libraries(lsxcommon PUBLIC lsx_tune)
# and strip only the LINK options (keep compile LTO) by replacing the
# target_link_options(...) call with a no-op comment; CUDA-linked exes then
# link non-LTO objects, which GCC/GNU ld accepts.
```
Then rerun Step 1 and Step 2. Do NOT reach for this unless LTO actually fails.

- [ ] **Step 4: Commit any fix produced by Step 3 (otherwise nothing to commit here)**

```bash
# if Step 3 applied:
git add libs/lsxcommon/CMakeLists.txt CMakeLists.txt
git commit -m "fix: keep -flto OFF at link for CUDA-linked executables (LTO/fatbin)"
```

---

### Task 5: Document in AGENTS.md

**Files:**
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: naming from Tasks 1-3 (`linux-release-znver4`, `lsx_tune`, `LSX_ARCH_ZNVER4`, `LSX_LTO`).

- [ ] **Step 1: Add the EPYC release preset to the Build section**

In `## Build`, after the `### Linux (Ubuntu 26.04, Ninja + GCC)` code block (current lines 41-47), insert:

```markdown
### Linux release (tuned for AMD EPYC 9V74, Zen 4)

```bash
cmake --preset linux-release-znver4
cmake --build --preset linux-release-znver4   # Ninja -j160
ctest --preset linux-release-znver4
```

Enables `LSX_ARCH_ZNVER4=ON` + `LSX_LTO=ON`: `lsx_tune` INTERFACE library
applies `-march=znver4` (-mtune=znver4, -O3, -fno-math-errno,
-falign-loops=16, -funroll-loops) and LTO (`-flto` GCC / `-flto=thin`
Clang-AOCC) to C/C++ targets only. CUDA (`lsxcommon_cu`), dependencies and
the Windows/MSVC build are never affected.
```

(Keep the outer code fences balanced — the example above contains one nested
fence; when pasting, the bullet block lives inside the first-level fence of
the plan, but in the real AGENTS.md it is plain markdown.)

- [ ] **Step 2: Replace the stale GCC/Clang flags bullet**

Replace current line 120:

```markdown
- **GCC/Clang flags**: `-march=native` in `CMAKE_CXX_FLAGS` (not `add_compile_options`, so it does not leak into NVCC). Enables all ISAs + host CPU tuning.
```

with:

```markdown
- **GCC/Clang Linux flags**: NOT global. The EPYC 9V74 (Zen 4) tuning lives in the
  INTERFACE library `lsx_tune` (root `CMakeLists.txt`) and is enabled via
  `LSX_ARCH_ZNVER4`/`LSX_LTO` (see the `linux-release-znver4` preset): `-march=znver4`,
  `-mtune=znver4`, `-O3` (Release), `-fno-math-errno`, `-falign-loops=16`,
  `-funroll-loops`, LTO `-flto` (GCC) / `-flto=thin` (Clang/AOCC). Everything is wrapped in
  `$<$<PLATFORM_ID:Linux>...$<COMPILE_LANGUAGE:C,CXX>>`, and LTO additionally in
  `$<$<LINK_LANGUAGE:CXX> $<CONFIG:Release>>` — it does not leak into NVCC or MSVC.
  The regular `linux-debug`/`linux-release` presets remain generic builds without
  `-march`.
```

- [ ] **Step 3: Verify docs render / no stale references**

Run:
```bash
grep -n "march=native" AGENTS.md CMakeLists.txt
```
Expected: no matches (both are clean now).
Also confirm the new preset is referenced:
```bash
grep -n "linux-release-znver4" AGENTS.md
```
Expected: at least one matching line.

- [ ] **Step 4: Commit**

```bash
git add AGENTS.md
git commit -m "docs: document EPYC znver4 tuned build (lsx_tune/linux-release-znver4)"
```

---

## Self-Review

**Spec coverage:**
- Decision 1 (fixed `-march=znver4`) → Task 2 `option(LSX_ARCH_ZNVER4)` + flags.
- Decision 2 (thin LTO, release-only) → Task 3 `LSX_LTO` block, `$<CONFIG:Release>` + `LINK_LANGUAGE` guards.
- Decision 3 (flags only on our targets via INTERFACE) → Task 2/3 `lsx_tune`, PUBLIC link in Task 3.
- Decision 4 (new preset; existing Linux presets stay generic after `-march=native` removal) → Task 1 + Task 2 Step 1/Verification in Task 2 Step 3.
- Decision 5 (Ninja 160) → Task 1 buildPreset `"jobs": 160`.
- Spec flags table — all five flags mapped (`-march`, `-mtune`, `-O3`, `-fno-math-errno`, `-falign-loops=16`) + `-funroll-loops`; LTO mapped.
- Spec CUDA isolation (3 layers) → `lsxcommon_cu` never links `lsx_tune`; `$<COMPILE_LANGUAGE:C,CXX>` guard; `$<LINK_LANGUAGE:CXX>` for LTO.
- Spec verification plan items 1-6 → Task 2 Step 4, Task 3 Step 3-4, Task 4 Steps 1-3.
- Spec Files Touched → Task 1 (CMakePresets.json), Task 2/3 (CMakeLists.txt, libs/lsxcommon/CMakeLists.txt), Task 4 (build), Task 5 (AGENTS.md). Spec doc itself already committed.

**Placeholder scan:** every step has exact code or exact commands with expected output; fallback Step 3 in Task 4 is conditional code, not a placeholder. No `TBD`.

**Type/name consistency:** preset/build/test name `linux-release-znver4`, target `lsx_tune`, options `LSX_ARCH_ZNVER4`/`LSX_LTO`, flag var `LSX_LTO_FLAG` — all consistent across tasks. `LSX_LTO_FLAG` is `set()` inside the `if(LSX_LTO...)` guard so it is always defined before use. CMakePresets `"jobs"` correctly placed in `buildPresets` (not `configurePresets`, which has no such key in v6).

**Gap check:** spec mentions "explicit `-mtune=znver4` for clarity" — included. Spec mentions keeping `-Og` in debug — debug builds simply don't take `lsx_tune`'s `-O3` (only Release does) and keep CMake's default flags, satisfying the intent; noted in Task 2 Step 1 that generic Debug builds carry no `-march`.