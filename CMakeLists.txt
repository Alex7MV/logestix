cmake_minimum_required(VERSION 3.15)

project(lsxfabric
    VERSION 0.1.0
    DESCRIPTION "Logestix Fabric — B2B inference engine"
    LANGUAGES C CXX
)

# ── Стандарты ────────────────────────────────────────────────────────────────

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Объектная библиотека в src/ будет переиспользована между основной программой
# и тестами; PIC гарантирует, что объектные файлы пригодны для линковки в .exe.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Флаги компилятора ────────────────────────────────────────────────────────

if(MSVC)
    add_compile_options(/W4 /utf-8 /MP /Zc:preprocessor)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── Генерация version.h из шаблона ──────────────────────────────────────────

configure_file(
    "${CMAKE_SOURCE_DIR}/include/lsxfabric/version.h.in"
    "${CMAKE_BINARY_DIR}/include/lsxfabric/version.h"
    @ONLY
)

# ── ЗАВИСИМОСТИ: Apache Arrow через FetchContent (Минимальная сборка) ────────

include(FetchContent)

# ХАК ДЛЯ СОВМЕСТИМОСТИ С CMAKE 4.x (Для xsimd/Arrow):
# Гарантирует, что CMake 4.x не заблокирует зависимости, требующие политики ниже 3.5
# Используем CMAKE_PROJECT_INCLUDE, чтобы файл применялся к project() любого вложенного
# ExternalProject (включая xsimd_ep), а не только к текущему проекту.
set(CMAKE_PROJECT_INCLUDE "${CMAKE_BINARY_DIR}/cmake_policy.cmake")
file(WRITE "${CMAKE_BINARY_DIR}/cmake_policy.cmake"
    "cmake_minimum_required(VERSION 3.5)\n"
    "set(CMAKE_POLICY_VERSION_MINIMUM 3.5)\n"
)

# Жестко отключаем тяжелые и ненужные модули для ускорения сборки
set(ARROW_BUILD_TESTS OFF CACHE INTERNAL "Disable Arrow tests")
set(ARROW_BUILD_BENCHMARKS OFF CACHE INTERNAL "Disable Arrow benchmarks")
set(ARROW_BUILD_SHARED ON CACHE INTERNAL "Build Arrow as shared library")
set(ARROW_BUILD_STATIC OFF CACHE INTERNAL "Disable Arrow static library")
set(ARROW_DATASET OFF CACHE INTERNAL "Disable Arrow Dataset module")
set(ARROW_FLIGHT OFF CACHE INTERNAL "Disable Arrow Flight RPC")
set(ARROW_GANDIVA OFF CACHE INTERNAL "Disable Gandiva LLVM compiler")
set(ARROW_HDFS OFF CACHE INTERNAL "Disable HDFS support")
set(ARROW_PARQUET OFF CACHE INTERNAL "Disable Parquet support")

# Включаем только то, что критично для инференса и работы с данными
set(ARROW_DEPENDENCY_SOURCE "BUNDLED" CACHE STRING "Use bundled dependencies")
set(ARROW_COMPUTE ON CACHE INTERNAL "Enable Arrow Compute Engine (requires Boost)")
set(ARROW_IPC ON CACHE INTERNAL "Enable Arrow IPC")
set(ARROW_CSV ON CACHE INTERNAL "Enable CSV parser")
set(ARROW_FILESYSTEM ON CACHE INTERNAL "Enable local filesystem support")
set(ARROW_JEMALLOC OFF CACHE INTERNAL "Use system allocator instead of jemalloc")
set(ARROW_WITH_SIMD OFF CACHE INTERNAL "Disable SIMD (CMake 4.x compat)")

FetchContent_Declare(
  Arrow
  GIT_REPOSITORY "https://github.com/apache/arrow.git"
  GIT_TAG        "apache-arrow-16.1.0"
  SOURCE_SUBDIR  "cpp"
  GIT_SHALLOW    TRUE
)

# Arrow 16.1.0 на Windows не определяет ARROW_SIMD_LEVEL (пустая строка),
# из-за чего if(NOT (${VAR} STREQUAL "NONE") ...) ломается в CMake 4.x.
# Явно задаём уровень, чтобы условие в src/arrow/io/CMakeLists.txt
# корректно вычислялось как FALSE (бенчмарк memory_benchmark не соберётся).
set(ARROW_SIMD_LEVEL "NONE" CACHE STRING "Arrow SIMD level")
set(ARROW_RUNTIME_SIMD_LEVEL "NONE" CACHE STRING "Arrow runtime SIMD level")

FetchContent_MakeAvailable(Arrow)

# ── ЗАВИСИМОСТИ: spdlog через FetchContent ────────────────────────────────────

# Отключаем сборку тестов и бенчмарков самой библиотеки spdlog
set(SPDLOG_BUILD_TESTS OFF CACHE INTERNAL "")
set(SPDLOG_BUILD_BENCHMARKS OFF CACHE INTERNAL "")

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY "https://github.com/gabime/spdlog.git"
  GIT_TAG        "v1.15.0" # Используем стабильный релиз
)
FetchContent_MakeAvailable(spdlog)

# ── ЗАВИСИМОСТИ: FlatBuffers через submodule (статическая сборка) ────────────

set(FLATBUFFERS_BUILD_FLATLIB ON CACHE INTERNAL "")
set(FLATBUFFERS_BUILD_FLATC OFF CACHE INTERNAL "")
set(FLATBUFFERS_BUILD_TESTS OFF CACHE INTERNAL "")
set(FLATBUFFERS_BUILD_BENCHMARKS OFF CACHE INTERNAL "")
set(FLATBUFFERS_INSTALL OFF CACHE INTERNAL "")

add_subdirectory(third_party/flatbuffers)

# ── ЗАВИСИМОСТИ: libzmq через submodule (статическая сборка) ──────────────────

set(BUILD_SHARED OFF CACHE INTERNAL "")
set(BUILD_STATIC ON CACHE INTERNAL "")
set(BUILD_TESTS OFF CACHE INTERNAL "")
set(WITH_PERF_TOOL OFF CACHE INTERNAL "")
set(WITH_LIBSODIUM OFF CACHE INTERNAL "")
set(ENABLE_DRAFTS OFF CACHE INTERNAL "")

add_subdirectory(third_party/libzmq)

# ── Рантайм окружение для Windows ───────────────────────────────────────────

# Должно быть до add_subdirectory, чтобы exe и dll попали в bin/
if(WIN32)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

# ── Поддиректории ────────────────────────────────────────────────────────────

add_subdirectory(src)

# Тесты собираются только для корневого проекта (не при add_subdirectory
# из внешнего проекта) и только если BUILD_TESTING не выключен.
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()


