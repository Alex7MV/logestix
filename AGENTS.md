# logestix — AGENTS.md

## Project

B2B inference engine. C++17 / C11. CMake 3.15+. Executable: `lsxfabric`.

## Build

### Linux (GCC/Clang)

```bash
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

### Windows (Visual Studio 2026)

```powershell
git submodule update --init --recursive
cmake -S . -B build -G "Visual Studio 18 2026"
cmake --build build --config Debug
```

## Test

```bash
cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

На Windows без `-DCMAKE_BUILD_TYPE` (используется `--config Debug`).

## Structure

- `src/core.cpp` — ядро; объектная библиотека `lsxfabric_objects`
- `src/main.cpp` — точка входа; исполняемый файл `lsxfabric`
- `tests/test_core.cpp` — тесты Catch2 v3; исполняемый файл `test_core`
- `third_party/cccl/` — NVIDIA CCCL (сабмодуль)
- `include/lsxfabric/` — публичные заголовки

## Conventions

- C++17, C11
- `#include` paths без лишних комментариев
- `target_include_directories` с генераторными выражениями `$<BUILD_INTERFACE:...>`
- Новые `.cpp` добавлять в `src/CMakeLists.txt` в объектную библиотеку или в `add_executable` в зависимости от принадлежности к ядру

## Git

- Английский язык сообщений коммитов
- Префиксы: `build:`, `feat:`, `fix:`, `refactor:`, `test:`, `docs:`
