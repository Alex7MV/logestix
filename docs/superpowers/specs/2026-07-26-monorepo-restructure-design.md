# Monorepo Restructure: Multi-App with Shared lsxcommon

## Goal

Restructure the single-application repository (`lsxfabric`) into a monorepo with
four applications (`lsxfabric`, `lsxforge`, `taskmosaic`, `lsxc`) sharing a
common STATIC library (`lsxcommon`).

## Directory Layout

```
logestix/
├── CMakeLists.txt                 # project(logestix ...), deps (Arrow, spdlog, ...)
├── apps/
│   ├── CMakeLists.txt             # add_subdirectory for each app
│   ├── lsxfabric/src/main.cpp     # existing entry point
│   ├── lsxforge/src/main.cpp      # placeholder
│   ├── taskmosaic/src/main.cpp    # placeholder
│   └── lsxc/src/main.cpp          # placeholder
├── libs/
│   ├── CMakeLists.txt             # add_subdirectory(libs/lsxcommon)
│   └── lsxcommon/
│       ├── CMakeLists.txt         # STATIC library target
│       ├── include/lsxcommon/
│       │   ├── core.h             # migrated from include/lsxfabric/core.h
│       │   └── version.h.in       # migrated from include/lsxfabric/version.h.in
│       └── src/
│           ├── core.cpp           # migrated from src/core.cpp
│           └── CMakeLists.txt
├── tests/
│   ├── CMakeLists.txt
│   └── test_core.cpp              # #include path updated to "lsxcommon/core.h"
├── third_party/                   # unchanged
│   ├── cccl/
│   ├── flatbuffers/
│   └── libzmq/
├── docs/
│   └── superpowers/specs/
└── AGENTS.md / README.md          # updated
```

Old `src/` and `include/` directories are removed after migration.

## CMake Architecture

### Root `CMakeLists.txt`

- Renamed project from `lsxfabric` to `logestix`
- Dependencies (Arrow, spdlog, simdjson, FlatBuffers, libzmq, CCCL, Catch2)
  unchanged
- `CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin` (Win32) stays
- `CMAKE_PROJECT_INCLUDE` xsimd workaround stays
- Subdirectories: `libs/`, `apps/`, conditionally `tests/`

### `libs/CMakeLists.txt`

```cmake
add_subdirectory(lsxcommon)
```

### `libs/lsxcommon/CMakeLists.txt`

- STATIC library target `lsxcommon` from `src/core.cpp`
- PUBLIC include directories:
  - `$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>`
  - `$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>` (for generated `version.h`)
- PUBLIC link: `arrow_shared`, `arrow_cuda_shared`, `spdlog::spdlog`,
  `flatbuffers`, `libzmq-static`, `simdjson::simdjson`
- Arrow include paths (from `_deps/arrow-src/cpp/src`,
  `_deps/arrow-build/src`) added PUBLICLY
- CCCL defines (`THRUST_DEVICE_SYSTEM`, `CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING`,
  `/Zc:preprocessor`) on the target
- `version.h` generation via `configure_file` from `version.h.in`

### `apps/lsxfabric/CMakeLists.txt`

```cmake
add_executable(lsxfabric src/main.cpp)
target_link_libraries(lsxfabric PRIVATE lsxcommon)
# POST_BUILD copy arrow.dll + arrow_cuda.dll (Win32)
```

### `apps/lsxforge/CMakeLists.txt` (and taskmosaic, lsxc)

Identical pattern with placeholder `main.cpp`. DLL copies included.

### `tests/CMakeLists.txt`

- Unchanged structure, links `lsxcommon` PRIVATE
- POST_BUILD DLL copy for `test_core`

## Pipeline

1. Create directory tree (`libs/lsxcommon/`, `apps/*/`)
2. Migrate `core.cpp`, `core.h`, `version.h.in` into `libs/lsxcommon/`
3. Write `libs/lsxcommon/CMakeLists.txt` with STATIC target
4. Write `libs/CMakeLists.txt`
5. Create placeholder `main.cpp` for each new app
6. Write `apps/*/CMakeLists.txt` for each app
7. Write `apps/CMakeLists.txt`
8. Update root `CMakeLists.txt`: project name → `logestix`, subdirs
9. Update `tests/test_core.cpp` include path
10. Remove old `src/` and `include/lsxfabric/`
11. Update `AGENTS.md`, `README.md`
12. Build & run tests

## Removed

- `src/` (migrated to `libs/lsxcommon/src/` and `apps/*/src/`)
- `include/lsxfabric/` (migrated to `libs/lsxcommon/include/lsxcommon/`)
