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
    add_compile_options(/W4 /utf-8)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── Генерация version.h из шаблона ──────────────────────────────────────────

configure_file(
    "${CMAKE_SOURCE_DIR}/include/lsxfabric/version.h.in"
    "${CMAKE_BINARY_DIR}/include/lsxfabric/version.h"
    @ONLY
)

# ── Поддиректории ────────────────────────────────────────────────────────────

add_subdirectory(src)

# Тесты собираются только для корневого проекта (не при add_subdirectory
# из внешнего проекта) и только если BUILD_TESTING не выключен.
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
