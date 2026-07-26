# logestix — AGENTS.md

## Project

Multi-app monorepo: fabric, forge, taskmosaic, lsxcli. C++20 / C11. CMake 3.15+.

## Dependencies

| Dependency | Version | Source | Type |
|---|---|---|---|
| Apache Arrow | 25.0.0 | FetchContent (BUNDLED) | Shared (`arrow.dll`) |
| spdlog | 1.15.0 | FetchContent | Header-only |
| FlatBuffers | v25.12.19 | submodule (`third_party/flatbuffers`) | Static |
| libzmq | v4.3.5 | submodule (`third_party/libzmq`) | Static |
| simdjson | v4.4.2 | submodule (`third_party/simdjson`) | Static (`.lib`) |
| NVIDIA CCCL | latest | submodule (`third_party/cccl`) | Header-only |
| Catch2 | v3.7.0 | FetchContent (test only) | Header-only |

## Build

### Linux (GCC/Clang)

```bash
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

### Windows (Visual Studio 2026)

Arrow's xsimd sub-dependency requires `CMAKE_POLICY_VERSION_MINIMUM=3.5`
for CMake 4.x. Set the env var before `cmake --build` (it's lost after configure).

```powershell
git submodule update --init --recursive
cmake -S . -B build -G "Visual Studio 18 2026"
$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"
cmake --build build --config Debug
```

`arrow.dll` and `arrow_cuda.dll` are auto-copied via `POST_BUILD`.

## Test

```bash
cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

On Windows без `-DCMAKE_BUILD_TYPE` (используется `--config Debug`):

```powershell
$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"
cmake --build build --config Debug
ctest --test-dir build --output-on-failure -C Debug
```

## Structure

- `libs/lsxcommon/` — shared STATIC library; namespace `lsxcommon`
  - `libs/lsxcommon/src/core.cpp` — ядро (логирование, инициализация)
  - `libs/lsxcommon/include/lsxcommon/core.h` — публичный заголовок
- `apps/lsxfabric/` — B2B inference engine
  - `apps/lsxfabric/tests/` — 1 test case, 1 assertion
- `apps/lsxforge/` — data preprocessing utility (placeholder)
  - `apps/lsxforge/tests/` — 1 test case, 1 assertion
- `apps/taskmosaic/` — task orchestrator (placeholder)
  - `apps/taskmosaic/tests/` — 1 test case, 1 assertion
- `apps/lsxcli/` — CLI client (placeholder)
  - `apps/lsxcli/tests/` — 1 test case, 1 assertion
- `tests/test_core.cpp` — интеграционные тесты; 10 test cases, 101 assertions
- `third_party/cccl/` — NVIDIA CCCL (сабмодуль)
- `third_party/flatbuffers/` — Google FlatBuffers v25.12.19 (сабмодуль)
- `third_party/libzmq/` — ZeroMQ libzmq v4.3.5 (сабмодуль)
- `third_party/simdjson/` — simdjson v4.4.2 (сабмодуль)

## CMake Key Points

- `lsxcommon` — STATIC library в `libs/lsxcommon/CMakeLists.txt`; линкуется PUBLIC с `arrow_shared`, `arrow_cuda_shared`, `spdlog::spdlog`, `flatbuffers`, `libzmq-static`, `simdjson::simdjson`
- Each app in `apps/*/` links `lsxcommon` PRIVATE
- Catch2 v3.7.0 fetched at root level (via FetchContent) when `BUILD_TESTING=ON`
- `CMAKE_RUNTIME_OUTPUT_DIRECTORY` установлен в `${CMAKE_BINARY_DIR}/bin` (до `add_subdirectory`)
- `add_custom_command(TARGET ... POST_BUILD)` копирует `arrow.dll` + `arrow_cuda.dll` в директорию exe (только Win32); applies to all 4 app targets, all 5 test targets
- **MSVC flags**: `WIN32_LEAN_AND_MEAN NOMINMAX` (suppress Windows macro conflicts with CCCL). DO NOT use `/Zc:preprocessor` (breaks CCCL/thrust). Use `CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING` + `THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP` for CCCL/MSVC compat.
- **Arrow 25.0.0 + Boost patch**: On MSVC, Boost 1.88.0's `literals.hpp` fails with C++20 due to `operator""_cppi` / `_cppui` sized literal redefinition. Workaround: remove the 4 `BOOST_MP_DEFINE_SIZED_CPP_INT_LITERAL(...)` invocations after Arrow download (see `CMakeLists.txt` lines 85-98).
- Arrow include paths добавлены PUBLIC напрямую к `lsxcommon` (`_deps/arrow-src/cpp/src`, `_deps/arrow-build/src`)

## Conventions

- C++20, C11
- `#include` paths без лишних комментариев
- `target_include_directories` с генераторными выражениями `$<BUILD_INTERFACE:...>`
- Новые `.cpp` добавлять в `apps/*/CMakeLists.txt` в `add_executable`, логику — в `libs/lsxcommon/`

## Git

- Английский язык сообщений коммитов
- Префиксы: `build:`, `feat:`, `fix:`, `refactor:`, `test:`, `docs:`
