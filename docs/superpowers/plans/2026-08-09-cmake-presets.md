# CMakePresets for Windows 11 & Ubuntu 26.04 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `CMakePresets.json` so development on Windows 11 (VS 2026) and Ubuntu 26.04 (Ninja/GCC) is reduced to `cmake --preset <name>` / `cmake --build --preset <name>` / `ctest --preset <name>`.

**Architecture:** A root `CMakePresets.json` (version 6) with 4 configure presets (windows/linux × debug/release), 4 derived build presets, and 4 derived test presets. `CMAKE_POLICY_VERSION_MINIMUM=3.5` moves into cacheVariables (no more `$env:`). README.md and AGENTS.md are updated to the new workflow.

**Tech Stack:** CMake ≥ 3.25 (presets), Visual Studio 18 2026 (Windows), Ninja + GCC (Linux).

**Spec:** `docs/superpowers/specs/2026-08-09-cmake-presets-design.md`

---

### Task 1: Create CMakePresets.json

**Files:**
- Create: `CMakePresets.json`

- [ ] **Step 1: Create the file**

```json
{
  "version": 6,
  "cmakeMinimumRequired": { "major": 3, "minor": 25, "patch": 0 },

  "configurePresets": [
    {
      "name": "windows-debug",
      "displayName": "Windows 11 (VS 2026) - Debug",
      "generator": "Visual Studio 18 2026",
      "architecture": "x64",
      "binaryDir": "${sourceDir}/build/windows-debug",
      "cacheVariables": {
        "CMAKE_POLICY_VERSION_MINIMUM": "3.5",
        "BUILD_TESTING": "ON"
      }
    },
    {
      "name": "windows-release",
      "displayName": "Windows 11 (VS 2026) - Release",
      "generator": "Visual Studio 18 2026",
      "architecture": "x64",
      "binaryDir": "${sourceDir}/build/windows-release",
      "cacheVariables": {
        "CMAKE_POLICY_VERSION_MINIMUM": "3.5",
        "BUILD_TESTING": "ON"
      }
    },
    {
      "name": "linux-debug",
      "displayName": "Ubuntu 26.04 (Ninja/GCC) - Debug",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/linux-debug",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "BUILD_TESTING": "ON"
      }
    },
    {
      "name": "linux-release",
      "displayName": "Ubuntu 26.04 (Ninja/GCC) - Release",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/linux-release",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "BUILD_TESTING": "ON"
      }
    }
  ],

  "buildPresets": [
    { "name": "windows-debug",   "configurePreset": "windows-debug" },
    { "name": "windows-release", "configurePreset": "windows-release" },
    { "name": "linux-debug",     "configurePreset": "linux-debug" },
    { "name": "linux-release",   "configurePreset": "linux-release" }
  ],

  "testPresets": [
    { "name": "windows-debug",   "configurePreset": "windows-debug",   "output": { "outputOnFailure": true } },
    { "name": "windows-release", "configurePreset": "windows-release", "output": { "outputOnFailure": true } },
    { "name": "linux-debug",     "configurePreset": "linux-debug",     "output": { "outputOnFailure": true } },
    { "name": "linux-release",   "configurePreset": "linux-release",   "output": { "outputOnFailure": true } }
  ]
}
```

- [ ] **Step 2: Verify JSON validity + preset listing**

Windows (use the bundled CMake):

```powershell
$cmake="C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
& $cmake --list-presets
```

Expected: output lists all 12 presets (`Available configure presets: * "windows-debug" ...`) and CMake reports no schema errors. If the installed CMake is < 3.25, upgrade or set `"version": 4` and lower `cmakeMinimumRequired` — but with VS 2026 the bundled CMake is ≥ 4.x, so version 6 is fine.

- [ ] **Step 3: Commit**

```bash
git add CMakePresets.json
git commit -m "build: add CMakePresets for Windows 11 & Ubuntu 26.04"
```

---

### Task 2: Verify Windows preset full flow

**Files:**
- (no source changes — verification only)

- [ ] **Step 1: Fresh configure via preset**

```powershell
$cmake="C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
& $cmake --preset windows-debug
```

Expected: configures into `build/windows-debug`, generates VS solution, Arrow/spdlog/Catch2 deps fetched, exits 0. No `$env:CMAKE_POLICY_VERSION_MINIMUM` needed.

- [ ] **Step 2: Build via preset**

```powershell
$cmake="C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
& $cmake --build --preset windows-debug
```

Expected: builds all targets, `lsxfabric.exe`/`lsxcli.exe`/`test_*.exe` appear under `build/windows-debug/bin/Debug/`, `arrow.dll` + `arrow_cuda.dll` auto-copied, exits 0.

- [ ] **Step 3: Test via preset**

```powershell
$ctest="C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe"
& $ctest --preset windows-debug
```

Expected: runs all 9 registered tests (`test_model_factory`, `test_core`, ...), 100% pass.

---

### Task 3: Update README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace Prerequisites version line**

Replace lines 17–18:

```markdown
- CMake 3.15+
- C++20 / C11 compiler (GCC 9+, Clang 12+, MSVC 2026+)
```

with:

```markdown
- CMake 3.25+ (for CMakePresets)
- C++20 / C11 compiler (GCC 13+ on Ubuntu 26.04, MSVC in Visual Studio 2026)
- Ninja build system (required on Ubuntu; optional on Windows)
- Git
```

- [ ] **Step 2: Replace the Build section**

Replace the entire `## Build` section (current lines 30–48) with:

```markdown
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
```

- [ ] **Step 3: Replace the Test section**

Replace the entire `## Test` section (current lines 50–67) with:

```markdown
## Test

`BUILD_TESTING=ON` is already enabled in all presets.

```powershell
ctest --preset windows-debug    # Windows 11
```

```bash
ctest --preset linux-debug      # Ubuntu 26.04
```
```

- [ ] **Step 4: Verify the file reads coherently**

```powershell
Get-Content README.md | Select-Object -First 70
```

Confirm no remaining `CMAKE_POLICY_VERSION_MINIMUM`, `-G "Visual Studio 18 2026"`, or `-DCMAKE_BUILD_TYPE` references in the Build/Test sections.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document CMakePresets workflow in README"
```

---

### Task 4: Update AGENTS.md

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Replace the Build section (Windows + Linux)**

Replace the current `## Build` block (both `### Linux (GCC/Clang)` and `### Windows (Visual Studio 2026)` subsections) with:

```markdown
## Build

Use CMakePresets — ready-made for both supported OSes. Replace `debug` with
`release` in the preset name for an optimization build.

### Linux (Ubuntu 26.04, Ninja + GCC)

```bash
git submodule update --init --recursive
cmake --preset linux-debug
cmake --build --preset linux-debug
```

### Windows (Visual Studio 2026)

```powershell
git submodule update --init --recursive
cmake --preset windows-debug
cmake --build --preset windows-debug
```

`arrow.dll` and `arrow_cuda.dll` are auto-copied via `POST_BUILD`.
```

Note: Arrow's xsimd sub-dependency hooks into `CMAKE_POLICY_VERSION_MINIMUM`; the
variable is now applied automatically through `CMakePresets.json` — no manual
`$env:CMAKE_POLICY_VERSION_MINIMUM` needed.

- [ ] **Step 2: Replace the Test section**

Replace the current `## Test` block with:

```markdown
## Test

```bash
ctest --preset linux-debug      # Ubuntu 26.04
```

```powershell
ctest --preset windows-debug    # Windows 11
```

`BUILD_TESTING=ON` is included in every preset.
```

- [ ] **Step 3: Update the CMake Key Points bullet**

In the `## CMake Key Points` list, find the bullet:

> - **MSVC flags**: ... `CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING` + `THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP` for CCCL/MSVC compat.

Append a new bullet after it:

> - **CMAKE_POLICY_VERSION_MINIMUM**: задаётся через `CMakePresets.json` (`cacheVariables`), ручная установка env-переменной не требуется. Arrow's xsimd требует значения 3.5.

Also confirm the existing Windows Build example that sets `$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"` is gone (it was replaced in Step 1).

- [ ] **Step 4: Commit**

```bash
git add AGENTS.md
git commit -m "docs: align AGENTS.md with CMakePresets workflow"
```

---

### Task 5: Final verification

- [ ] **Step 1: Confirm presets listed**

```powershell
$cmake="C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
& $cmake --list-presets
```

Expected: 4 configure + 4 build + 4 test presets shown.

- [ ] **Step 2: Confirm no stale manual-flow references**

```bash
rg -n "CMAKE_POLICY_VERSION_MINIMUM|Visual Studio 18 2026|DCMAKE_BUILD_TYPE" README.md AGENTS.md
```

Expected: the only remaining hits are the AGENTS.md CMake Key Points bullet (new wording) and policy references inside `CMakeLists.txt` (out of scope). README must have zero hits.

- [ ] **Step 3: Confirm git status clean**

```bash
git status
```

Expected: no untracked/modified files beyond the intended 3 changed docs + 1 new preset file (all already committed).

---

## Self-Review

**Spec coverage:**
- §3 CMakePresets.json — Task 1 ✓
- §4 build/test presets — Task 1 ✓
- §5 developer commands — verified in Task 2 ✓
- §6 README/AGENTS updates — Tasks 3 + 4 ✓
- §8 verification — Tasks 2 + 5 ✓

**Placeholder scan:** No TBD/TODO. All JSON is inline; all doc edits show exact before/after content.

**Type consistency:** Preset names `windows-debug`, `windows-release`, `linux-debug`, `linux-release` used identically across configure/build/test presets, commands, and docs.

**Note for Ubuntu verifier:** Linux-based verification (Tasks 2/5 Ubuntu half) runs on the Ubuntu 26.04 machine via the same commands with `linux-debug`/`linux-release`; Windows-side verification covers what's runnable in this environment.