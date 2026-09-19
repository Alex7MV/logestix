# CMakePresets for Windows 11 & Ubuntu 26.04 — Design Spec

## 1. Overview

Default build settings for the two target OSes — Windows 11 (Visual Studio
2026, multi-config) and Ubuntu 26.04 (Ninja + GCC) — via a single `CMakePresets.json`
at the repository root. Goal: remove manual flags/env variables from the
daily development flow.

Today the build requires manual steps (README/AGENTS.md):

- Windows: `-G "Visual Studio 18 2026"` + `$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"` (the env var is lost after configure, set before every build) + `--config Debug` + `-C Debug`.
- Ubuntu: `-DCMAKE_BUILD_TYPE=Debug` + `-march=native` (internal flag, unchanged).

Project: two presets (Debug/Release) per OS, commands reduced to
`cmake --preset <name>` / `cmake --build --preset <name>` / `ctest --preset <name>`.

## 2. Constraints

- **No new code** — configuration and documentation files only.
- `CMakePresets.json` goes into git; personal overrides — `CMakeUserPresets.json` (already in `.gitignore`, the rule stays).
- `.gitignore` covers `/build/` — the new `build/windows-debug` etc. directories are ignored automatically, no extra rule needed.
- Internal build flags (`/arch:AVX512`, `-march=native`, Boost patch, CCCL defines) are unchanged — edits only in the outer invocation layer.
- `CMAKE_POLICY_VERSION_MINIMUM=3.5` is handled via a cache variable in the preset, not an env var; setting the env manually is no longer required.

## 3. CMakePresets.json

New file `CMakePresets.json` at the root, `version: 6` (CMake 3.25+).

### Configure presets

| name | displayName | generator | arch | binaryDir | cacheVariables |
|---|---|---|---|---|---|
| `windows-debug` | Windows 11 (VS 2026) - Debug | `Visual Studio 18 2026` | `x64` | `${sourceDir}/build/windows-debug` | `CMAKE_POLICY_VERSION_MINIMUM=3.5`, `BUILD_TESTING=ON` |
| `windows-release` | Windows 11 (VS 2026) - Release | `Visual Studio 18 2026` | `x64` | `${sourceDir}/build/windows-release` | `CMAKE_POLICY_VERSION_MINIMUM=3.5`, `BUILD_TESTING=ON` |
| `linux-debug` | Ubuntu 26.04 (Ninja/GCC) - Debug | `Ninja` | — | `${sourceDir}/build/linux-debug` | `CMAKE_BUILD_TYPE=Debug`, `BUILD_TESTING=ON` |
| `linux-release` | Ubuntu 26.04 (Ninja/GCC) - Release | `Ninja` | — | `${sourceDir}/build/linux-release` | `CMAKE_BUILD_TYPE=Release`, `BUILD_TESTING=ON` |

JSON skeleton (see the "Implementation" section for the full file):

```json
{
  "version": 6,
  "cmakeMinimumRequired": { "major": 3, "minor": 25, "patch": 0 },
  "configurePresets": [ ...4 entries... ],
  "buildPresets": [ ...4 entries... ],
  "testPresets": [ ...4 entries... ]
}
```

## 4. Build and Test presets

`buildPresets` — derivatives of the configure presets (they use their binaryDir via `configurePreset`):

```json
{ "name": "windows-debug", "configurePreset": "windows-debug" },
{ "name": "windows-release", "configurePreset": "windows-release" },
{ "name": "linux-debug", "configurePreset": "linux-debug" },
{ "name": "linux-release", "configurePreset": "linux-release" }
```

`testPresets` — derivatives of the build presets (inherit buildDir) with `output.outputOnFailure: true`:

```json
{ "name": "windows-debug", "configurePreset": "windows-debug",
  "output": { "outputOnFailure": true } }
// ... and the other 3
```

`jobCount` is not set in buildPresets — CMake uses the default threads.

## 5. Developer commands

| Action | Windows 11 | Ubuntu 26.04 |
|---|---|---|
| Configure | `cmake --preset windows-debug` | `cmake --preset linux-debug` |
| Build | `cmake --build --preset windows-debug` | `cmake --build --preset linux-debug` |
| Test | `ctest --preset windows-debug` | `ctest --preset linux-debug` |

- `$env:CMAKE_POLICY_VERSION_MINIMUM` is no longer needed on Windows — the policy is set by a cache variable in the preset.
- `--config` / `-C` are not needed: the multi-config generator takes the config from the buildPreset name; for Release replace `debug` → `release` in the preset name.
- Executables on Windows: `build/windows-debug/bin/<config>/`, on Linux: `build/linux-debug/bin/`.

## 6. Documentation update

### README.md

The "Build" and "Test" sections are replaced with the preset flow:

- Windows: `cmake --preset windows-debug` + `cmake --build --preset windows-debug`.
- Ubuntu: `cmake --preset linux-debug` + `cmake --build --preset linux-debug`.
- Tests: `ctest --preset windows-debug` / `ctest --preset linux-debug`.
- Note: replace `debug` → `release` for a Release build; `BUILD_TESTING=ON` is already in the presets.
- All mention of `$env:CMAKE_POLICY_VERSION_MINIMUM` and `-G "Visual Studio 18 2026"` is removed from the commands.
- "Prerequisites" section: explicitly state CMake ≥ 3.25 (for presets), GCC 13+ (Ubuntu 26.04) and the presence of Ninja (Linux) / VS 2026 (Windows).

### AGENTS.md

- The Build/Test sections for Windows and Linux are reduced to the preset commands.
- The `CMAKE_POLICY_VERSION_MINIMUM` bullet in "CMake Key Points" is reworded: the variable is set via `CMakePresets.json` (cacheVariables); setting the env manually is not required.
- Everything else (compiler flags, Boost patch, CUDA requirements) is unchanged.

## 7. Non-goals

- Build scripts (`build.ps1` / `build.sh`), CI (.github/workflows), .editorconfig / .clang-format / VS Code settings.
- Changing the internal compilation flags and the CMakeLists structure.
- macOS / ARM / non-NVIDIA GPU support.

## 8. Implementation (plan reference)

`CMakePresets.json` (full file) — 4 configure, 4 build, 4 test presets per the skeletons above. Edits only in `README.md` and `AGENTS.md`. Verification:

- Windows: `cmake --preset windows-debug` (new `build/windows-debug` directory), `cmake --build --preset windows-debug`, `ctest --preset windows-debug` — all tests pass.
- Ubuntu: the same commands with `linux-debug`.
- `cmake --list-presets` shows 12 presets.
