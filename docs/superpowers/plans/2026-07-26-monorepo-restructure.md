# Monorepo Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure single-application repo into monorepo with 4 apps (lsxfabric, lsxforge, taskmosaic, lsxc) sharing a STATIC library (lsxcommon).

**Architecture:** Vertical layout — `libs/lsxcommon/` (STATIC lib with all deps), `apps/*/` (executables linking PRIVATE to lsxcommon), old `src/` and `include/` removed.

**Tech Stack:** CMake 3.15+, C++17, Apache Arrow 16.1.0, spdlog, FlatBuffers, libzmq, simdjson, CCCL, Catch2

---

### Task 1: Create directory tree

**Files:**
- Create: `libs/lsxcommon/src/`
- Create: `libs/lsxcommon/include/lsxcommon/`
- Create: `apps/lsxfabric/src/`
- Create: `apps/lsxforge/src/`
- Create: `apps/taskmosaic/src/`
- Create: `apps/lsxc/src/`

- [ ] **Step 1: Create all directories**

```powershell
mkdir -p libs/lsxcommon/src libs/lsxcommon/include/lsxcommon apps/lsxfabric/src apps/lsxforge/src apps/taskmosaic/src apps/lsxc/src
```

- [ ] **Step 2: Commit**

```bash
git add libs/ apps/
git commit -m "build: create monorepo directory tree"
```

---

### Task 2: Migrate core library into libs/lsxcommon/

**Files:**
- Copy+modify: `include/lsxfabric/core.h` → `libs/lsxcommon/include/lsxcommon/core.h`
- Copy+modify: `include/lsxfabric/version.h.in` → `libs/lsxcommon/include/lsxcommon/version.h.in`
- Copy+modify: `src/core.cpp` → `libs/lsxcommon/src/core.cpp`

- [ ] **Step 1: Migrate and rename `core.h`**

```cpp
#ifndef LSXCOMMON_CORE_H
#define LSXCOMMON_CORE_H

#include <string>

namespace lsxcommon {

class Core {
public:
    static void Initialize(const std::string& log_file_path = "logestix.log");
    static void Shutdown();
    static void ExecuteDummyInference();
};

} // namespace lsxcommon

#endif // LSXCOMMON_CORE_H
```

Write to `libs/lsxcommon/include/lsxcommon/core.h`.

- [ ] **Step 2: Migrate `version.h.in` with new prefix**

```cpp
#ifndef LSXCOMMON_VERSION_H
#define LSXCOMMON_VERSION_H

#define LSXCOMMON_VERSION_MAJOR @PROJECT_VERSION_MAJOR@
#define LSXCOMMON_VERSION_MINOR @PROJECT_VERSION_MINOR@
#define LSXCOMMON_VERSION_PATCH @PROJECT_VERSION_PATCH@
#define LSXCOMMON_VERSION_STRING "@PROJECT_VERSION@"

#endif
```

Write to `libs/lsxcommon/include/lsxcommon/version.h.in`.

- [ ] **Step 3: Migrate and rename `core.cpp`**

```cpp
#include "lsxcommon/core.h"

#include <spdlog/spdlog.h>
#include <spdlog/async.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>

#include <memory>
#include <vector>

namespace lsxcommon {

void Core::Initialize(const std::string& log_file_path) {
    try {
        spdlog::init_thread_pool(8192, 1);

        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            log_file_path, 1024 * 1024 * 5, 3
        );

        std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};

        auto logger = std::make_shared<spdlog::async_logger>(
            "lsx_logger", sinks.begin(), sinks.end(),
            spdlog::thread_pool(), spdlog::async_overflow_policy::block
        );

        logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [thread %t] [%^%l%$] %v");
        logger->set_level(spdlog::level::info);
        spdlog::set_default_logger(logger);

        spdlog::info("Logestix common library core initialized.");
    } catch (const spdlog::spdlog_ex& ex) {
        fprintf(stderr, "Critical: Log initialization failed: %s\n", ex.what());
    }
}

void Core::Shutdown() {
    spdlog::info("Shutting down Logestix common library core...");
    spdlog::shutdown();
}

void Core::ExecuteDummyInference() {
    spdlog::debug("Starting execution pipeline...");
    spdlog::info("Processing target inference batch...");
    spdlog::warn("Hardware allocation high utilization detected, falling back to stream pooling.");
}

} // namespace lsxcommon
```

Write to `libs/lsxcommon/src/core.cpp`.

- [ ] **Step 4: Commit**

```bash
git add libs/lsxcommon/
git commit -m "refactor: migrate core library into libs/lsxcommon with 'lsxcommon' namespace"
```

---

### Task 3: Write CMakeLists for libs/lsxcommon and libs/

**Files:**
- Create: `libs/lsxcommon/CMakeLists.txt`
- Create: `libs/CMakeLists.txt`

- [ ] **Step 1: Write `libs/lsxcommon/CMakeLists.txt`**

```cmake
# ── Shared common library (STATIC) ──────────────────────────────────────
# All public dependencies are linked PUBLICly so apps inherit them.
#
add_library(lsxcommon STATIC src/core.cpp)

target_compile_definitions(lsxcommon PUBLIC
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
)

target_include_directories(lsxcommon
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/cccl/thrust>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/cccl/cub>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/cccl/libcudacxx/include>"
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src>"
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/_deps/arrow-build/src>"
)

# Generate version.h from template
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/lsxcommon/version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/lsxcommon/version.h"
    @ONLY
)

target_link_libraries(lsxcommon PUBLIC
    arrow_shared
    arrow_cuda_shared
    spdlog::spdlog
    flatbuffers
    libzmq-static
    simdjson::simdjson
)
```

- [ ] **Step 2: Write `libs/CMakeLists.txt`**

```cmake
add_subdirectory(lsxcommon)
```

- [ ] **Step 3: Commit**

```bash
git add libs/
git commit -m "build: add lsxcommon STATIC library target with all deps"
```

---

### Task 4: Write CMakeLists for apps

**Files:**
- Create: `apps/lsxfabric/CMakeLists.txt`
- Create: `apps/lsxforge/CMakeLists.txt`
- Create: `apps/taskmosaic/CMakeLists.txt`
- Create: `apps/lsxc/CMakeLists.txt`
- Create: `apps/CMakeLists.txt`

- [ ] **Step 1: Write `apps/lsxfabric/CMakeLists.txt`**

```cmake
add_executable(lsxfabric src/main.cpp)
target_link_libraries(lsxfabric PRIVATE lsxcommon ${CMAKE_DL_LIBS})

if(WIN32)
    add_custom_command(TARGET lsxfabric POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:lsxfabric>"
    )
    add_custom_command(TARGET lsxfabric POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:lsxfabric>"
    )
endif()

install(TARGETS lsxfabric RUNTIME DESTINATION bin)
```

- [ ] **Step 2: Write `apps/lsxforge/CMakeLists.txt`**

```cmake
add_executable(lsxforge src/main.cpp)
target_link_libraries(lsxforge PRIVATE lsxcommon ${CMAKE_DL_LIBS})

if(WIN32)
    add_custom_command(TARGET lsxforge POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:lsxforge>"
    )
    add_custom_command(TARGET lsxforge POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:lsxforge>"
    )
endif()

install(TARGETS lsxforge RUNTIME DESTINATION bin)
```

- [ ] **Step 3: Write `apps/taskmosaic/CMakeLists.txt`**

```cmake
add_executable(taskmosaic src/main.cpp)
target_link_libraries(taskmosaic PRIVATE lsxcommon ${CMAKE_DL_LIBS})

if(WIN32)
    add_custom_command(TARGET taskmosaic POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:taskmosaic>"
    )
    add_custom_command(TARGET taskmosaic POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:taskmosaic>"
    )
endif()

install(TARGETS taskmosaic RUNTIME DESTINATION bin)
```

- [ ] **Step 4: Write `apps/lsxc/CMakeLists.txt`**

```cmake
add_executable(lsxc src/main.cpp)
target_link_libraries(lsxc PRIVATE lsxcommon ${CMAKE_DL_LIBS})

if(WIN32)
    add_custom_command(TARGET lsxc POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:lsxc>"
    )
    add_custom_command(TARGET lsxc POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:lsxc>"
    )
endif()

install(TARGETS lsxc RUNTIME DESTINATION bin)
```

- [ ] **Step 5: Write `apps/CMakeLists.txt`**

```cmake
add_subdirectory(lsxfabric)
add_subdirectory(lsxforge)
add_subdirectory(taskmosaic)
add_subdirectory(lsxc)
```

- [ ] **Step 6: Commit**

```bash
git add apps/
git commit -m "build: add all 4 app targets with DLL copies"
```

---

### Task 5: Migrate and update lsxfabric main.cpp; create placeholder apps

**Files:**
- Copy+modify: `src/main.cpp` → `apps/lsxfabric/src/main.cpp`
- Create: `apps/lsxforge/src/main.cpp`
- Create: `apps/taskmosaic/src/main.cpp`
- Create: `apps/lsxc/src/main.cpp`

- [ ] **Step 1: Migrate `main.cpp` with updated include**

```cpp
#include "lsxcommon/core.h"
#include <iostream>

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    lsxcommon::Core::Initialize("logestix_runtime.log");
    lsxcommon::Core::ExecuteDummyInference();
    lsxcommon::Core::Shutdown();
    return 0;
}
```

Write to `apps/lsxfabric/src/main.cpp`.

- [ ] **Step 2: Create placeholder `lsxforge/src/main.cpp`**

```cpp
#include "lsxcommon/core.h"
#include <iostream>

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    lsxcommon::Core::Initialize("lsxforge_runtime.log");
    lsxcommon::Core::ExecuteDummyInference();
    lsxcommon::Core::Shutdown();
    return 0;
}
```

Write to `apps/lsxforge/src/main.cpp`.

- [ ] **Step 3: Create placeholder `taskmosaic/src/main.cpp`**

```cpp
#include "lsxcommon/core.h"
#include <iostream>

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    lsxcommon::Core::Initialize("taskmosaic_runtime.log");
    lsxcommon::Core::ExecuteDummyInference();
    lsxcommon::Core::Shutdown();
    return 0;
}
```

Write to `apps/taskmosaic/src/main.cpp`.

- [ ] **Step 4: Create placeholder `lsxc/src/main.cpp`**

```cpp
#include "lsxcommon/core.h"
#include <iostream>

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    lsxcommon::Core::Initialize("lsxc_runtime.log");
    lsxcommon::Core::ExecuteDummyInference();
    lsxcommon::Core::Shutdown();
    return 0;
}
```

Write to `apps/lsxc/src/main.cpp`.

- [ ] **Step 5: Commit**

```bash
git add apps/
git commit -m "feat: migrate lsxfabric main, add placeholder apps for lsxforge/taskmosaic/lsxc"
```

---

### Task 6: Update root CMakeLists.txt

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Rename project and update subdirectories**

Change project name from `lsxfabric` to `logestix`:
```cmake
project(logestix
    VERSION 0.1.0
    DESCRIPTION "Logestix — multi-app monorepo: fabric, forge, taskmosaic, c"
    LANGUAGES C CXX
)
```

Move `CMAKE_RUNTIME_OUTPUT_DIRECTORY` block to before `add_subdirectory` calls (keep it where it is, already before the `add_subdirectory(src)` call):

Replace the subdirectory section at the bottom:
```cmake
# ── Поддиректории ───────────────────────────────────────────────────────
#
add_subdirectory(libs)
add_subdirectory(apps)

# Тесты собираются только для корневого проекта
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
```

Remove the old `add_subdirectory(src)` line.

- [ ] **Step 2: Commit**

```bash
git add CMakeLists.txt
git commit -m "build: rename project to logestix, point subdirs to libs/ and apps/"
```

---

### Task 7: Update test target to link lsxcommon

**Files:**
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Update link from `lsxfabric_objects` to `lsxcommon`**

```cmake
add_executable(test_core test_core.cpp)
target_link_libraries(test_core PRIVATE lsxcommon Catch2::Catch2WithMain)
```

- [ ] **Step 2: Commit**

```bash
git add tests/CMakeLists.txt
git commit -m "build: link test_core to lsxcommon instead of lsxfabric_objects"
```

---

### Task 8: Remove old src/ and include/lsxfabric/

**Files:**
- Delete: `src/` (entire directory)
- Delete: `include/lsxfabric/` (entire directory)

- [ ] **Step 1: Remove old directories**

```powershell
Remove-Item -Recurse -Force src
Remove-Item -Recurse -Force include/lsxfabric
```

- [ ] **Step 2: Commit**

```bash
git add -A
git commit -m "refactor: remove old src/ and include/lsxfabric/ directories"
```

---

### Task 9: Update AGENTS.md and README.md

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`

- [ ] **Step 1: Update AGENTS.md structure section**

Replace with:

```markdown
## Structure

- `libs/lsxcommon/` — shared STATIC library; namespace `lsxcommon`
  - `libs/lsxcommon/src/core.cpp` — ядро (логирование, инициализация)
  - `libs/lsxcommon/include/lsxcommon/core.h` — публичный заголовок
- `apps/lsxfabric/` — B2B inference engine
- `apps/lsxforge/` — data preprocessing utility (placeholder)
- `apps/taskmosaic/` — task orchestrator (placeholder)
- `apps/lsxc/` — CLI client (placeholder)
- `tests/test_core.cpp` — тесты Catch2 v3; 10 test cases, 101 assertions
```

- [ ] **Step 2: Commit**

```bash
git add AGENTS.md README.md
git commit -m "docs: update AGENTS.md and README.md for monorepo structure"
```

---

### Task 10: Build and run tests

**Files:**
- No file changes

- [ ] **Step 1: Clean build directory and reconfigure**

```powershell
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
cmake -S . -B build -G "Visual Studio 18 2026" -DBUILD_TESTING=ON
```

Expected: Configuration done, all dependencies found.

- [ ] **Step 2: Build all targets**

```powershell
$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"
cmake --build build --config Debug
```

Expected: Build succeeds with no errors.

- [ ] **Step 3: Run tests**

```powershell
ctest --test-dir build --output-on-failure -C Debug
```

Expected: All tests pass.

- [ ] **Step 4: Verify all 4 executables exist**

```powershell
Get-ChildItem build/bin/Debug/lsxfabric.exe, build/bin/Debug/lsxforge.exe,
    build/bin/Debug/taskmosaic.exe, build/bin/Debug/lsxc.exe
```

Expected: 4 exe files present.

- [ ] **Step 5: Commit final state**

```bash
git add -A
git commit -m "refactor: complete monorepo restructure, all tests pass"
```
