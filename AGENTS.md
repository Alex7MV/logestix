# logestix — AGENTS.md

## Project

B2B inference engine. C++17 / C11. CMake 3.15+. Executable: `lsxfabric`.

## Dependencies

| Dependency | Version | Source | Type |
|---|---|---|---|
| Apache Arrow | 16.1.0 | FetchContent (BUNDLED) | Shared (`arrow.dll`) |
| spdlog | 1.15.0 | FetchContent | Header-only |
| FlatBuffers | v25.12.19 | submodule (`third_party/flatbuffers`) | Static |
| libzmq | v4.3.5 | submodule (`third_party/libzmq`) | Static |
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

`arrow.dll` is auto-copied to the output directory via `POST_BUILD`.

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

- `src/core.cpp` — ядро; объектная библиотека `lsxfabric_objects`
- `src/main.cpp` — точка входа; исполняемый файл `lsxfabric`
- `tests/test_core.cpp` — тесты Catch2 v3; 7 test cases, 76 assertions
- `third_party/cccl/` — NVIDIA CCCL (сабмодуль)
- `third_party/flatbuffers/` — Google FlatBuffers v25.12.19 (сабмодуль)
- `third_party/libzmq/` — ZeroMQ libzmq v4.3.5 (сабмодуль)
- `include/lsxfabric/` — публичные заголовки

## CMake Key Points

- `lsxfabric_objects` — OBJECT library в `src/CMakeLists.txt`; линкуется PUBLIC с `arrow_shared`, `spdlog::spdlog`, `flatbuffers`, `libzmq-static`
- `CMAKE_RUNTIME_OUTPUT_DIRECTORY` установлен в `${CMAKE_BINARY_DIR}/bin` (до `add_subdirectory`)
- `add_custom_command(TARGET ... POST_BUILD)` копирует `arrow.dll` в директорию exe (только Win32)
- `/Zc:preprocessor` + `THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP` + `CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING` для CCCL/MSVC
- Arrow include paths добавлены PUBLIC напрямую к `lsxfabric_objects` (`_deps/arrow-src/cpp/src`, `_deps/arrow-build/src`)

## Conventions

- C++17, C11
- `#include` paths без лишних комментариев
- `target_include_directories` с генераторными выражениями `$<BUILD_INTERFACE:...>`
- Новые `.cpp` добавлять в `src/CMakeLists.txt` в объектную библиотеку или в `add_executable` в зависимости от принадлежности к ядру

## Git

- Английский язык сообщений коммитов
- Префиксы: `build:`, `feat:`, `fix:`, `refactor:`, `test:`, `docs:`
