cmake_minimum_required(VERSION 3.15)

project(logestix
    VERSION 0.1.0
    DESCRIPTION "Logestix — multi-app monorepo: fabric, forge, taskmosaic, c"
    LANGUAGES C CXX CUDA
)

find_package(CUDAToolkit REQUIRED)

# ── Toolkit consistency guard (docs/open-tasks/2026-09-30-..., item G1) ──────────
# Upgrading CUDA leaves an existing build/ pinned to the previous toolkit: the
# cached `CUDAToolkit_*` entries stay valid, so a new CUDAToolkit_ROOT is not
# re-derived and the header version can disagree with the runtime the compiler
# targets. `test_cuda_toolkit_guard` only catches that after a full build; fail
# here instead, before compiling. Re-run configure with `-U CUDAToolkit_*` (or
# wipe the build directory) after a toolkit upgrade.
if(DEFINED CUDAToolkit_VERSION AND DEFINED CMAKE_CUDA_COMPILER_VERSION)
    string(REGEX MATCH "^[0-9]+\\.[0-9]+" _lsx_toolkit_mm
           "${CUDAToolkit_VERSION}")
    string(REGEX MATCH "^[0-9]+\\.[0-9]+" _lsx_nvcc_mm
           "${CMAKE_CUDA_COMPILER_VERSION}")
    if(NOT _lsx_toolkit_mm STREQUAL _lsx_nvcc_mm)
        message(FATAL_ERROR
            "CUDA toolkit/compiler version mismatch: find_package(CUDAToolkit) "
            "resolved ${CUDAToolkit_VERSION} but the CUDA compiler is "
            "${CMAKE_CUDA_COMPILER_VERSION}. This usually means a stale build/ "
            "cache after a toolkit upgrade (docs/open-tasks/2026-09-30-...). Reconfigure with "
            "`cmake -U CUDAToolkit_* ...` or wipe the build directory.")
    endif()
endif()

# ── CCCL provenance + band guard ──────────────────────────────────────────────
# CCCL is not vendored: it comes from the CUDA toolkit bundle at
# <toolkit-include>/cccl (cub/cuda/nv/thrust). A submodule used to shadow it via
# -I order, so every .cu — including the toolkit's own transitively-included
# <nv/target> — compiled against our copy; a toolkit bump could then silently
# invert that (an older submodule shadowing a newer toolkit bundle). Delegating
# to the toolkit removes the shadow. This guard keeps the discipline: a toolkit
# whose CCCL moves outside the validated band fails here, at configure time,
# rather than silently changing the libcu++ under the whole tree. Widen the band
# only after re-validating (fast suite + a device selfcheck).
if(NOT DEFINED CUDAToolkit_INCLUDE_DIRS)
    message(FATAL_ERROR "find_package(CUDAToolkit) provided no include directories")
endif()
list(GET CUDAToolkit_INCLUDE_DIRS 0 _lsx_cuda_inc)
# Band = the CCCL minors the repository accepts from the toolkit. Floor tracks
# the CTK floor (13.2 -> CCCL 3.2.0, see test_cuda_toolkit_guard); ceiling is the
# first unvalidated minor — a forward jump past it forces a deliberate
# re-validation (fast suite + a device selfcheck) before the band is widened.
set(LSX_CCCL_BAND_MIN 3002000)
set(LSX_CCCL_BAND_MAX 3006000)
# CUDA 13 bundles CCCL under <include>/cccl (cub/cuda/nv/thrust); other layouts
# (and some Windows installs) place it at the include root. Accept either so the
# guard does not false-fail on a valid toolkit.
set(LSX_CCCL_INCLUDE_DIR "")
foreach(_lsx_cccl_candidate "${_lsx_cuda_inc}/cccl" "${_lsx_cuda_inc}")
    if(EXISTS "${_lsx_cccl_candidate}/cuda/std/__cccl/version.h")
        set(LSX_CCCL_INCLUDE_DIR "${_lsx_cccl_candidate}")
        break()
    endif()
endforeach()
if(NOT LSX_CCCL_INCLUDE_DIR)
    message(FATAL_ERROR
        "CCCL not found under ${_lsx_cuda_inc} or ${_lsx_cuda_inc}/cccl. CUDA Toolkit "
        "${CUDAToolkit_VERSION} is expected to bundle CCCL (cub/cuda/nv/thrust).")
endif()
file(STRINGS "${LSX_CCCL_INCLUDE_DIR}/cuda/std/__cccl/version.h" _lsx_cccl_line
     REGEX "#define CCCL_VERSION ")
string(REGEX MATCH "[0-9]+" _lsx_cccl_version "${_lsx_cccl_line}")
if(_lsx_cccl_version LESS ${LSX_CCCL_BAND_MIN}
   OR NOT _lsx_cccl_version LESS ${LSX_CCCL_BAND_MAX})
    message(FATAL_ERROR
        "Bundled CCCL ${_lsx_cccl_version} (CUDA Toolkit ${CUDAToolkit_VERSION}) is "
        "outside the validated band [${LSX_CCCL_BAND_MIN}, ${LSX_CCCL_BAND_MAX}). "
        "Re-validate on this toolkit (fast suite + a device selfcheck), then widen "
        "the band in CMakeLists.txt and tests/CMakeLists.txt.")
endif()
message(STATUS
    "CCCL from CUDA toolkit: ${_lsx_cccl_version} (band ${LSX_CCCL_BAND_MIN}..${LSX_CCCL_BAND_MAX})")

# ── Standards ────────────────────────────────────────────────────────────────

# C++20 is the baseline for all code (including CUDA via
# CUDA_STANDARD 20). The compilers support higher standards
# (GCC 15 / nvcc 13.2+ understand C++23), but the project is deliberately pinned to C++20.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Compiler flags ────────────────────────────────────────────────────────

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
# `lsx_tune` — an INTERFACE library carrying aggressive C/C++ optimization flags
# for the EPYC 9V74. It is linked PUBLIC into `lsxcommon` (and through it —
# transitively into all apps and tests). CUDA objects (`lsxcommon_cu`), FetchContent
# and submodule dependencies (Arrow, spdlog, simdjson, FlatBuffers, libzmq) do NOT
# get these flags: everything is wrapped in $<$<COMPILE_LANGUAGE:C,CXX>:...> plus
# $<$<PLATFORM_ID:Linux>:...>. MSVC and Apple are unaffected.
# Enabled via LSX_ARCH_ZNVER4 / LSX_LTO (see the linux-release-znver4 preset).
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

# ── DEPENDENCIES: Apache Arrow via FetchContent (minimal build) ────────

include(FetchContent)

# CMake 4.x compatibility for old bundled sub-projects (Arrow's uriparser /
# Boost / xsimd may still call cmake_minimum_required(VERSION <3.5), which
# CMake 4 rejects). Use the documented mechanism: CMAKE_POLICY_VERSION_MINIMUM
# raises the *effective* minimum only for projects that ask below the floor; it
# does NOT change the policy version of projects that already ask at or above
# it. A normal variable set here is inherited by the in-process nested projects
# (Arrow and its add_subdirectory/FetchContent deps).
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
# Do NOT reintroduce an injected `cmake_minimum_required(VERSION 3.5)` (e.g. via
# CMAKE_PROJECT_INCLUDE): such a file is included as the last step of *every*
# project() call and resets that scope's policy version to 3.5, flipping
# CMP0076/CMP0091 — and every other post-3.5 policy — back to OLD. Under the VS
# generator that mis-compiles Arrow's bundled uriparser (CMP0091, MSVC runtime
# library) and breaks Boost's relative extra/*.natvis sources (CMP0076).

# Hard-disable heavy and unneeded modules to speed up the build
set(ARROW_BUILD_TESTS OFF CACHE INTERNAL "Disable Arrow tests")
set(ARROW_BUILD_BENCHMARKS OFF CACHE INTERNAL "Disable Arrow benchmarks")
set(ARROW_BUILD_SHARED ON CACHE INTERNAL "Build Arrow as shared library")
set(ARROW_BUILD_STATIC OFF CACHE INTERNAL "Disable Arrow static library")
set(ARROW_DATASET OFF CACHE INTERNAL "Disable Arrow Dataset module")
set(ARROW_FLIGHT OFF CACHE INTERNAL "Disable Arrow Flight RPC")
set(ARROW_GANDIVA OFF CACHE INTERNAL "Disable Gandiva LLVM compiler")
set(ARROW_HDFS OFF CACHE INTERNAL "Disable HDFS support")
set(ARROW_PARQUET OFF CACHE INTERNAL "Disable Parquet support")
# Newer compilers (GCC 15+) are stricter; CHECKIN adds -Werror in debug and
# Arrow does not build under CHECKIN (-Werror=unused-parameter). We use
# PRODUCTION, where -Werror is not added at all.
set(BUILD_WARNING_LEVEL "PRODUCTION" CACHE STRING "Arrow: disable -Werror" FORCE)

# Enable only what is critical for inference and data handling
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
  GIT_TAG        "apache-arrow-26.0.0"
  SOURCE_SUBDIR  "cpp"
  GIT_SHALLOW    TRUE
)

# On Windows Arrow does not define ARROW_SIMD_LEVEL (empty string),
# which breaks if(NOT (${VAR} STREQUAL "NONE") ...) under CMake 4.x.
# We set the level explicitly so the condition in src/arrow/io/CMakeLists.txt
# evaluates to FALSE correctly (the memory_benchmark benchmark will not build).
set(ARROW_SIMD_LEVEL "NONE" CACHE STRING "Arrow SIMD level")
set(ARROW_RUNTIME_SIMD_LEVEL "NONE" CACHE STRING "Arrow runtime SIMD level")

FetchContent_MakeAvailable(Arrow)

# Patch Boost 1.92.0 multiprecision literals.hpp for MSVC C++20 compat (Arrow
# 26.0.0 bundles Boost 1.92.0; the same four macro invocations are still there in
# the same shape, so this string-keyed patch still matches).
# The `operator""_cppi` / `_cppui` variadic UDL templates fail with MSVC 2026
# due to redefinition errors between the unsized and sized (_cppi128 etc)
# literal operators. Arrow doesn't use these literal suffixes, so we drop the
# four sized-literal definitions.
# Arrow only fetches Boost when ARROW_BUILD_TESTS/INTEGRATION/TESTING/FLIGHT/
# GANDIVA/THRIFT need it; with this tree's flags (all off) it is never
# downloaded and the patch is a no-op. It is kept because MSVC builds that
# enable any of those flags would otherwise fail on the redefinition.
set(_boost_literals "${arrow_SOURCE_DIR}/../boost-src/libs/multiprecision/include/boost/multiprecision/cpp_int/literals.hpp")
if(EXISTS "${_boost_literals}")
  file(READ "${_boost_literals}" _content)
  string(REGEX REPLACE
    "BOOST_MP_DEFINE_SIZED_CPP_INT_LITERAL\\(128\\)\nBOOST_MP_DEFINE_SIZED_CPP_INT_LITERAL\\(256\\)\nBOOST_MP_DEFINE_SIZED_CPP_INT_LITERAL\\(512\\)\nBOOST_MP_DEFINE_SIZED_CPP_INT_LITERAL\\(1024\\)"
    ""
    _content "${_content}")
  file(WRITE "${_boost_literals}" "${_content}")
endif()

# ── DEPENDENCIES: spdlog via FetchContent ────────────────────────────────────

# Disable building spdlog's own tests and benchmarks
set(SPDLOG_BUILD_TESTS OFF CACHE INTERNAL "")
set(SPDLOG_BUILD_BENCHMARKS OFF CACHE INTERNAL "")

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY "https://github.com/gabime/spdlog.git"
  GIT_TAG        "v1.17.0" # Use the stable release
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

# ── DEPENDENCIES: abseil-cpp via FetchContent (static flat structures) ─
# Stable LTS tag (verified 2026-09: "Abseil LTS branch, August 2026",
# 20260817.0, commit 2065f4d). The version is enforced by the guard in
# tests/test_core.cpp "pinned dependency versions" (ABSL_LTS_RELEASE_VERSION).
# The static libs must honor the project's top-level C++ standard:
# ABSL_PROPAGATE_CXX_STD=ON makes every abseil target inherit the parent's
# CMAKE_CXX_STANDARD (the project is deliberately pinned to C++20) instead of
# the default C++14, otherwise there is an ABI mismatch when linking into
# lsxcommon.
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
# patch-after-fetch pattern as Boost/spdlog above). The global
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

# ── DEPENDENCIES: simdjson via submodule (fast JSON parser) ───────────

set(SIMDJSON_DEVELOPER_MODE OFF CACHE INTERNAL "")
set(SIMDJSON_ENABLE_DEPRECATED_API OFF CACHE INTERNAL "")

add_subdirectory(third_party/simdjson)

# ── DEPENDENCIES: Catch2 via FetchContent (tests only) ─────────────────
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

# ── DEPENDENCIES: cpp-httplib via FetchContent (host-only HTTP server) ────────
# Header-only. OpenSSL/Zlib/Brotli integrations are OFF so no extra system deps
# leak in (the bridge is plaintext HTTP + SSE).
set(HTTPLIB_REQUIRE_OPENSSL OFF CACHE INTERNAL "")
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE INTERNAL "")
set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF CACHE INTERNAL "")
set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE INTERNAL "")
FetchContent_Declare(
    httplib
    GIT_REPOSITORY "https://github.com/yhirose/cpp-httplib.git"
    GIT_TAG        "v0.58.0"
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(httplib)

# ── DEPENDENCIES: FlatBuffers via submodule (static build) ────────────

set(FLATBUFFERS_BUILD_FLATLIB ON CACHE INTERNAL "")
set(FLATBUFFERS_BUILD_FLATC OFF CACHE INTERNAL "")
set(FLATBUFFERS_BUILD_TESTS OFF CACHE INTERNAL "")
set(FLATBUFFERS_BUILD_BENCHMARKS OFF CACHE INTERNAL "")
set(FLATBUFFERS_INSTALL OFF CACHE INTERNAL "")

add_subdirectory(third_party/flatbuffers)

# ── DEPENDENCIES: libzmq via submodule (static build) ──────────────────

set(BUILD_SHARED OFF CACHE INTERNAL "")
set(BUILD_STATIC ON CACHE INTERNAL "")
set(BUILD_TESTS OFF CACHE INTERNAL "")
set(WITH_PERF_TOOL OFF CACHE INTERNAL "")
set(WITH_LIBSODIUM OFF CACHE INTERNAL "")
set(ENABLE_DRAFTS OFF CACHE INTERNAL "")

add_subdirectory(third_party/libzmq)

# ── libzmq Corresponding Source for release builds (MPL-2.0 §3.2) ─────────────
# libzmq is MPL-2.0 (file-level copyleft). Shipping an Executable Form that
# links it requires the Source Code Form to be made available. Release builds
# therefore package the pinned submodule sources into an archive that is copied
# next to every application binary (see lsx_copy_third_party_notices) and into
# install/bin. Debug builds are unaffected. The tree is unmodified, so the
# archive is the exact upstream libzmq source (see DEPENDENCIES.md).
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libzmq/include/zmq.h" _lsx_zmq_h)
string(REGEX MATCH "#define ZMQ_VERSION_MAJOR ([0-9]+)" _ "${_lsx_zmq_h}")
set(_lsx_zmq_major "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define ZMQ_VERSION_MINOR ([0-9]+)" _ "${_lsx_zmq_h}")
set(_lsx_zmq_minor "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define ZMQ_VERSION_PATCH ([0-9]+)" _ "${_lsx_zmq_h}")
set(_lsx_zmq_patch "${CMAKE_MATCH_1}")
unset(_lsx_zmq_h)
if(NOT _lsx_zmq_major OR NOT _lsx_zmq_minor OR NOT _lsx_zmq_patch)
    message(FATAL_ERROR "Could not read the libzmq version from third_party/libzmq/include/zmq.h")
endif()
set(LSX_ZMQ_VERSION "${_lsx_zmq_major}.${_lsx_zmq_minor}.${_lsx_zmq_patch}")
set(LSX_ZMQ_SOURCE_ARCHIVE
    "${CMAKE_BINARY_DIR}/libzmq-${LSX_ZMQ_VERSION}-source.tar.gz")

# A release build is a single-config Release-like CMAKE_BUILD_TYPE, or a
# multi-config generator that offers a Release configuration.
set(_lsx_is_release OFF)
if(CMAKE_BUILD_TYPE MATCHES "^(Release|RelWithDebInfo|MinSizeRel)$")
    set(_lsx_is_release ON)
endif()
if(CMAKE_CONFIGURATION_TYPES AND "Release" IN_LIST CMAKE_CONFIGURATION_TYPES)
    set(_lsx_is_release ON)
endif()

if(_lsx_is_release)
    find_package(Git REQUIRED)
    add_custom_command(
        OUTPUT "${LSX_ZMQ_SOURCE_ARCHIVE}"
        COMMAND "${GIT_EXECUTABLE}" archive
                --format=tar.gz
                --prefix=libzmq-${LSX_ZMQ_VERSION}/
                -o "${LSX_ZMQ_SOURCE_ARCHIVE}"
                HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libzmq"
        COMMENT "Packaging libzmq ${LSX_ZMQ_VERSION} source for redistribution (MPL-2.0 §3.2)"
        VERBATIM
    )
    add_custom_target(lsx_mpl_sources ALL DEPENDS "${LSX_ZMQ_SOURCE_ARCHIVE}")
    install(FILES "${LSX_ZMQ_SOURCE_ARCHIVE}" DESTINATION bin)
endif()

# ── Runtime environment for Windows ───────────────────────────────────────────

# Must come before add_subdirectory so the exe and dll land in bin/
if(WIN32)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

# THIRD_PARTY_NOTICES must accompany every distributed binary
# (Apache-2.0 §4(d), MPL-2.0, OFL, Unicode). The helper is defined before
# add_subdirectory(apps) so apps can call it right after
# add_executable.
function(lsx_copy_third_party_notices target)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${PROJECT_SOURCE_DIR}/THIRD_PARTY_NOTICES"
            "$<TARGET_FILE_DIR:${target}>/THIRD_PARTY_NOTICES"
        COMMENT "Copy THIRD_PARTY_NOTICES next to ${target}"
    )
    # MPL-2.0 §3.2: release binaries must ship the libzmq Corresponding Source.
    # The helper no-ops for non-release configurations (checked via $<CONFIG>).
    if(_lsx_is_release)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                "-DLSX_CONFIG=$<CONFIG>"
                "-DLSX_ZMQ_ARCHIVE=${LSX_ZMQ_SOURCE_ARCHIVE}"
                "-DLSX_DEST=$<TARGET_FILE_DIR:${target}>"
                -P "${PROJECT_SOURCE_DIR}/cmake/copy_mpl_sources.cmake"
            COMMENT "Ship the libzmq source archive next to ${target} (MPL-2.0 §3.2)"
            VERBATIM
        )
    endif()
endfunction()

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/THIRD_PARTY_NOTICES" DESTINATION bin)

# ── Consumer switches ─────────────────────────────────────────────────────────
# When logestix is consumed as a dependency (via FetchContent or
# add_subdirectory), the embedding project builds only the engine library
# (libs/). These options keep the sibling applications,
# the offline tools and the engine's own test suite out of the consumer's build
# tree. Defaults preserve the standalone monorepo build.
option(LOGESTIX_BUILD_APPS "Build the logestix applications" ON)
option(LOGESTIX_BUILD_TOOLS "Build the logestix offline tools" ON)
option(LOGESTIX_BUILD_TESTS "Build the logestix test suite" ON)

# ── Subdirectories ────────────────────────────────────────────────────────────

# enable_testing must be called before add_subdirectory(apps) so that
# add_test() inside each app works correctly.
# The CMAKE_PROJECT_NAME guard prevents a second call when add_subdirectory
# is used from an external project.
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    enable_testing()
endif()

add_subdirectory(libs)

if(LOGESTIX_BUILD_APPS)
    add_subdirectory(apps)
endif()

if(LOGESTIX_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

# Shared tests not tied to a specific app (e.g. dependency integration
# checks: Arrow, CCCL, FlatBuffers, ZeroMQ, simdjson).
if(BUILD_TESTING AND LOGESTIX_BUILD_TESTS)
    add_subdirectory(tests)
endif()


