cmake_minimum_required(VERSION 3.15)

project(logestix
    VERSION 0.1.0
    DESCRIPTION "Logestix — multi-app monorepo: fabric, forge, taskmosaic, c"
    LANGUAGES C CXX CUDA
)

find_package(CUDAToolkit REQUIRED)

# ── Стандарты ────────────────────────────────────────────────────────────────

# C++20 используется как baseline для всего кода (включая CUDA через
# CUDA_STANDARD 20). Компиляторы поддерживают более высокие стандарты
# (GCC 15 / nvcc 13.2+ умеют C++23), но проект осознанно зафиксирован на C++20.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Флаги компилятора ────────────────────────────────────────────────────────

if(MSVC)
    # CACHE FORCE so the value propagates into single-file try_compile sub-builds
    # (abseil's check_cxx_source_compiles probes). A plain set() only overrides the
    # in-memory variable and the probe sub-build falls back to the toolchain default
    # C++14, missing the /std:c++20 gate. /std:c++20 aligns with the C++20 pin below.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /utf-8 /MP /arch:AVX512 /std:c++20" CACHE STRING "Flags used by the C++ compiler." FORCE)
    # WIN32_LEAN_AND_MEAN / NOMINMAX: suppress Windows header bloat and min/max macros
    # that conflict with std::numeric_limits::max() in CCCL/Thrust.
    add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX)
    # CCCL (CUDA C++ Core Libraries) doesn't work with /Zc:preprocessor on MSVC.
    # Use CXX standard defines instead to suppress its traditional-preprocessor warning.
    add_compile_definitions(CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── Linux CPU-tuned build: AMD EPYC 9V74 (Zen 4) ─────────────────────────────
# `lsx_tune` — INTERFACE-библиотека, несущая агрессивные флаги оптимизации C/C++
# под EPYC 9V74. Линкуется PUBLIC в `lsxcommon` (а через него — транзитивно во
# все приложения и тесты). CUDA-объекты (`lsxcommon_cu`), FetchContent- и
# submodule-зависимости (Arrow, spdlog, simdjson, FlatBuffers, libzmq) эти флаги
# НЕ получают: всё обёрнуто в $<$<COMPILE_LANGUAGE:C,CXX>:...> плюс
# $<$<PLATFORM_ID:Linux>:...>. MSVC и Apple не затрагиваются.
# Включается через LSX_ARCH_ZNVER4 / LSX_LTO (см. пресет linux-release-znver4).
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
# Новые компиляторы (GCC 15+) строже; CHECKIN добавляет -Werror в debug и
# Arrow не собирается на CHECKIN (-Werror=unused-parameter). Используем
# PRODUCTION, где -Werror не добавляется вовсе.
set(BUILD_WARNING_LEVEL "PRODUCTION" CACHE STRING "Arrow: disable -Werror" FORCE)

# Включаем только то, что критично для инференса и работы с данными
set(ARROW_DEPENDENCY_SOURCE "BUNDLED" CACHE STRING "Use bundled dependencies")
set(ARROW_CUDA ON CACHE INTERNAL "Enable Arrow CUDA GPU support (requires CUDA Toolkit)")
set(ARROW_COMPUTE ON CACHE INTERNAL "Enable Arrow Compute Engine (requires Boost)")
set(ARROW_IPC ON CACHE INTERNAL "Enable Arrow IPC")
set(ARROW_CSV ON CACHE INTERNAL "Enable CSV parser")
set(ARROW_FILESYSTEM ON CACHE INTERNAL "Enable local filesystem support")
set(ARROW_JEMALLOC OFF CACHE INTERNAL "Use system allocator instead of jemalloc")
set(ARROW_WITH_SIMD OFF CACHE INTERNAL "Disable SIMD (CMake 4.x compat)")
set(ARROW_ENABLE_THREADING ON CACHE INTERNAL "Enable threading in Arrow core")

FetchContent_Declare(
  Arrow
  GIT_REPOSITORY "https://github.com/apache/arrow.git"
  GIT_TAG        "apache-arrow-25.0.1"
  SOURCE_SUBDIR  "cpp"
  GIT_SHALLOW    TRUE
)

# Arrow на Windows не определяет ARROW_SIMD_LEVEL (пустая строка),
# из-за чего if(NOT (${VAR} STREQUAL "NONE") ...) ломается в CMake 4.x.
# Явно задаём уровень, чтобы условие в src/arrow/io/CMakeLists.txt
# корректно вычислялось как FALSE (бенчмарк memory_benchmark не соберётся).
set(ARROW_SIMD_LEVEL "NONE" CACHE STRING "Arrow SIMD level")
set(ARROW_RUNTIME_SIMD_LEVEL "NONE" CACHE STRING "Arrow runtime SIMD level")

FetchContent_MakeAvailable(Arrow)

# Patch Boost 1.88.0 multiprecision literals.hpp for MSVC C++20 compat.
# The `operator""_cppi` / `_cppui` variadic UDL templates fail with MSVC 2026
# due to redefinition errors between the unsized and sized (_cppi128 etc)
# literal operators. Arrow doesn't use these literal suffixes, so we wrap
# them in an #ifndef guard.
set(_boost_literals "${arrow_SOURCE_DIR}/../boost-src/libs/multiprecision/include/boost/multiprecision/cpp_int/literals.hpp")
if(EXISTS "${_boost_literals}")
  file(READ "${_boost_literals}" _content)
  string(REGEX REPLACE
    "BOOST_MP_DEFINE_SIZED_CPP_INT_LITERAL\\(128\\)\nBOOST_MP_DEFINE_SIZED_CPP_INT_LITERAL\\(256\\)\nBOOST_MP_DEFINE_SIZED_CPP_INT_LITERAL\\(512\\)\nBOOST_MP_DEFINE_SIZED_CPP_INT_LITERAL\\(1024\\)"
    ""
    _content "${_content}")
  file(WRITE "${_boost_literals}" "${_content}")
endif()

# ── ЗАВИСИМОСТИ: spdlog через FetchContent ────────────────────────────────────

# Отключаем сборку тестов и бенчмарков самой библиотеки spdlog
set(SPDLOG_BUILD_TESTS OFF CACHE INTERNAL "")
set(SPDLOG_BUILD_BENCHMARKS OFF CACHE INTERNAL "")

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY "https://github.com/gabime/spdlog.git"
  GIT_TAG        "v1.17.0" # Используем стабильный релиз
)
FetchContent_MakeAvailable(spdlog)

# ── Patch bundled fmt for nvcc 13.x (Windows) ─────────────────────────────────
# spdlog bundles fmt, whose `fractional_part_rounding_thresholds` uses a
# char32 string literal with code points above U+10FFFF (0x9999999a, ...).
# GCC/Clang/MSVC accept those, but the nvcc front end rejects them with
# "character value is out of range" when a .cu pulls in <spdlog/spdlog.h>
# (glm_moe_load.cu). Replace the literal with a plain uint32_t array —
# idempotent: both replaces no-op once already applied against v1.15.0.
set(_spdlog_fmt "${spdlog_SOURCE_DIR}/include/spdlog/fmt/bundled/format.h")
if(EXISTS "${_spdlog_fmt}")
    file(READ "${_spdlog_fmt}" _spdlog_fmt_content)
    # Idempotence guard: the array declaration is the marker — never re-insert it.
    string(FIND "${_spdlog_fmt_content}" "gNvccRoundingThresholds" _spdlog_fmt_marker)
    if(_spdlog_fmt_marker EQUAL -1)
        string(REPLACE
            "fractional_part_rounding_thresholds(int index) -> uint32_t {"
            "fractional_part_rounding_thresholds(int index) -> uint32_t {\n  constexpr uint32_t gNvccRoundingThresholds[] = {0x9999999a, 0x828f5c29, 0x80418938, 0x80068db9, 0x8000a7c6, 0x800010c7, 0x800001ae, 0x8000002b};"
            _spdlog_fmt_content "${_spdlog_fmt_content}")
        string(REGEX REPLACE
            "return U\"[^\n]*\n[^\]]*\\[index\\];"
            "return gNvccRoundingThresholds[index];"
            _spdlog_fmt_content "${_spdlog_fmt_content}")
        file(WRITE "${_spdlog_fmt}" "${_spdlog_fmt_content}")
    endif()
    unset(_spdlog_fmt_marker)
    unset(_spdlog_fmt_content)
endif()
unset(_spdlog_fmt)

# ── ЗАВИСИМОСТИ: abseil-cpp через FetchContent (статические flat-структуры) ─
# Стабильный LTS-тег (проверен 2026-09: "Abseil LTS branch, August 2026",
# 20260817.0, коммит 2065f4d). Версия принуждается guard'ом
# tests/test_core.cpp "pinned dependency versions" (ABSL_LTS_RELEASE_VERSION).
# Статика обязана учитывать верхнеуровневый C++ стандарт проекта:
# ABSL_PROPAGATE_CXX_STD=ON заставляет каждую abseil-цель наследовать
# CMAKE_CXX_STANDARD родителя (проект осознанно зафиксирован на C++20), а не
# дефолтный C++14, иначе ABI-расхождение при линковке в lsxcommon/lsxhome.
set(ABSL_BUILD_TESTING OFF CACHE INTERNAL "Disable abseil tests")
set(ABSL_BUILD_TEST_HELPERS OFF CACHE INTERNAL "Disable abseil test helpers")
set(ABSL_BUILD_EXAMPLES OFF CACHE INTERNAL "Disable abseil examples")
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "Propagate project C++ standard to abseil" FORCE)
# Abseil's ABSL_INTERNAL_AT_LEAST_CXX17/CXX20 probes run as a nested
# check_cxx_source_compiles (try_compile) in a fresh sub-build. The top-level
# set(CMAKE_CXX_STANDARD 20) above is a NORMAL variable and is invisible there
# (MSVC would fall back to its default C++14, failing the >=C++17 gate). Promote
# it to a CACHE variable so the probe inherits /std:c++20.
set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard (CACHE for abseil try_compile sub-build)" FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "C++ standard required (CACHE for abseil)" FORCE)

FetchContent_Declare(
  abseil-cpp
  GIT_REPOSITORY "https://github.com/abseil/abseil-cpp.git"
  GIT_TAG        "20260817.0" # Abseil LTS, August 2026
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(abseil-cpp)

# Patch abseil CRC for MSVC + project-wide /arch:AVX512 (idempotent, same
# patch-after-fetch pattern as Boost/spdlog/imnodes above). The global
# /arch:AVX512 (root CMAKE_CXX_FLAGS) defines __AVX__ on MSVC, which activates
# abseil's GCC/Clang-only VPCLMULQDQ inline-asm path (crc_x86_arm_combined.cc's
# `__asm__`), and MSVC cannot parse those statements. Disabling the MSVC SIMD
# branch keeps CRC on the generic intrinsic fallback. Rebuilds cleanly anyway,
# since 20260817.0 re-fetches; the guard is applied each configure.
set(_absx86simd "${abseil-cpp_SOURCE_DIR}/absl/crc/internal/crc32_x86_arm_combined_simd.h")
if(EXISTS "${_absx86simd}")
    file(READ "${_absx86simd}" _absxsimd_content)
    string(FIND "${_absxsimd_content}" "defined(_M_AMD64) && 0" _absxsimd_marker)
    if(_absxsimd_marker EQUAL -1)
        string(REPLACE
            "#elif defined(_MSC_VER) && !defined(__clang__) && defined(__AVX__) && \\\n    defined(_M_AMD64)"
            "#elif defined(_MSC_VER) && !defined(__clang__) && defined(__AVX__) && \\\n    defined(_M_AMD64) && 0"
            _absxsimd_content "${_absxsimd_content}")
        file(WRITE "${_absx86simd}" "${_absxsimd_content}")
    endif()
    unset(_absxsimd_marker)
    unset(_absxsimd_content)
endif()
unset(_absx86simd)

# Patch abseil str_format compile-time checker for Clang/AOCC (idempotent,
# same patch-after-fetch pattern as crc32 above). GCC has no
# `__attribute__((enable_if))` (`__has_attribute(enable_if)` == 0), so
# ABSL_INTERNAL_ENABLE_FORMAT_CHECKER stays undefined and the checker is inert;
# Clang/AOCC DO enable it, but its ValidFormatImpl only parses printf-style
# '%' conversions. The repo is printf-style everywhere (enforced by
# `tests/test_absl_format_guard.cpp` — fmt-style "{}" placeholders print
# literally and are forbidden); the `#if 0` guard keeps a future stray
# "{}" typo from breaking the whole AOCC build and mirrors GCC behaviour.
set(_absfmtchk "${abseil-cpp_SOURCE_DIR}/absl/strings/internal/str_format/checker.h")
if(EXISTS "${_absfmtchk}")
    file(READ "${_absfmtchk}" _absfmtchk_content)
    string(FIND "${_absfmtchk_content}" "#if 0 && ABSL_HAVE_ATTRIBUTE(enable_if)" _absfmtchk_marker)
    if(_absfmtchk_marker EQUAL -1)
        string(REPLACE
            "#if ABSL_HAVE_ATTRIBUTE(enable_if) && !defined(__INTELLISENSE__)"
            "#if 0 && ABSL_HAVE_ATTRIBUTE(enable_if) && !defined(__INTELLISENSE__)"
            _absfmtchk_content "${_absfmtchk_content}")
        file(WRITE "${_absfmtchk}" "${_absfmtchk_content}")
    endif()
    unset(_absfmtchk_marker)
    unset(_absfmtchk_content)
endif()
unset(_absfmtchk)

# ── ЗАВИСИМОСТИ: simdjson через submodule (быстрый JSON-парсер) ───────────

set(SIMDJSON_DEVELOPER_MODE OFF CACHE INTERNAL "")
set(SIMDJSON_ENABLE_DEPRECATED_API OFF CACHE INTERNAL "")

add_subdirectory(third_party/simdjson)

# ── ЗАВИСИМОСТИ: Catch2 через FetchContent (только тесты) ─────────────────
if(BUILD_TESTING)
    set(CATCH_BUILD_TESTING OFF CACHE INTERNAL "")
    set(CATCH_BUILD_EXAMPLES OFF CACHE INTERNAL "")
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY "https://github.com/catchorg/Catch2.git"
        GIT_TAG        "v3.16.0"
    )
    FetchContent_MakeAvailable(Catch2)
endif()

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

# THIRD_PARTY_NOTICES обязан сопровождать каждый распространяемый бинарь
# (Apache-2.0 §4(d), MPL-2.0, OFL, Unicode). Helper определён до
# add_subdirectory(apps), чтобы приложения могли его вызвать сразу после
# add_executable.
function(lsx_copy_third_party_notices target)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/THIRD_PARTY_NOTICES"
            "$<TARGET_FILE_DIR:${target}>/THIRD_PARTY_NOTICES"
        COMMENT "Copy THIRD_PARTY_NOTICES next to ${target}"
    )
endfunction()

install(FILES "${CMAKE_SOURCE_DIR}/THIRD_PARTY_NOTICES" DESTINATION bin)

# ── Поддиректории ────────────────────────────────────────────────────────────

# enable_testing должен быть вызван до add_subdirectory(apps), чтобы
# add_test() внутри каждого приложения работал корректно.
# Guard на CMAKE_PROJECT_NAME предотвращает повторный вызов при add_subdirectory
# из внешнего проекта.
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    enable_testing()
endif()

add_subdirectory(libs)
add_subdirectory(apps)
add_subdirectory(tools)

# Общие тесты, не привязанные к конкретному приложению (например, интеграционные
# проверки зависимостей: Arrow, CCCL, FlatBuffers, ZeroMQ, simdjson).
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()


