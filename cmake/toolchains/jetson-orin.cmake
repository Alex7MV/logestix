# Jetson Orin (JetPack 6.2.1 / CUDA 12.6) toolchain.
#
# Why a toolchain FILE and not just cache variables in the preset: three things
# a configure preset cannot express portably.
#
#   1. The compiler is not on PATH. JetPack installs CUDA at /usr/local/cuda-12.6
#      and `nvcc` there is the only one that knows sm_87 + the JetPack driver. A
#      preset that relied on find_package(CUDAToolkit) would pick up whatever
#      /usr/local/cuda points at, which on a machine that also has a desktop
#      toolkit is a configure that silently targets the wrong toolkit.
#   2. The HOST is aarch64, so the x86 conveniences must be off — not merely
#      unset. LSX_ARCH_ZNVER4 would put -march=znver4 in front of an ARM
#      compiler, and LSX_LTO on aarch64 nvcc is a link-time budget this board
#      does not have.
#   3. The RPATH layout differs: an aarch64 build tree is not the x86 one, so the
#      $ORIGIN ladder the AOCC preset sets does not apply here.
#
# Deliberately NOT set here: LSX_CUDA_ARCH. The preset owns the arch list (86;87
# — the Orin is 8.7, and the fatbin carries 8.6 too so one artifact serves the
# 3090 host), and a toolchain file that also pinned it would be a second source of
# truth for the one variable the whole build hangs off.
#
# Usage:  cmake --preset jetson-orin-release
# Override the toolkit root on a machine that installs it elsewhere:
#   -DLSX_JETSON_CUDA_ROOT=/usr/local/cuda-12.6
#
# WHY THIS FILE WARNS INSTEAD OF REFUSING
#
# A toolchain file is re-included in EVERY nested configure, and a CUDA configure
# performs one for compiler-ABI detection (try_compile) with a FRESH, EMPTY cache
# — so no -D value and no cache entry is visible there. A `message(FATAL_ERROR)`
# on a missing toolkit therefore aborts the ABI probe, and the operator sees
# "CMAKE_C_COMPILER not set, after EnableLanguage" instead of the real reason.
# Measured on 2026-11-28 while adding this file. So: resolve loudly, refuse
# quietly, and let find_package(CUDAToolkit REQUIRED) — which runs in the real
# configure with the real cache — be the thing that fails, naming the toolkit.

set(LSX_JETSON_CUDA_ROOT "/usr/local/cuda-12.6" CACHE PATH
    "CUDA 12.6 toolkit root from JetPack 6.2.1")

# A usable root is one whose nvcc exists. When it does not, this pass is almost
# always a nested probe (see above) and must not change what the parent resolved.
if(EXISTS "${LSX_JETSON_CUDA_ROOT}/bin/nvcc")
    set(_lsx_jetson_nvcc "${LSX_JETSON_CUDA_ROOT}/bin/nvcc")
elseif(CMAKE_CUDA_COMPILER AND EXISTS "${CMAKE_CUDA_COMPILER}")
    # Already resolved (a parent configure, or a re-include): keep it.
    set(_lsx_jetson_nvcc "${CMAKE_CUDA_COMPILER}")
else()
    set(_lsx_jetson_nvcc "")
endif()

if(_lsx_jetson_nvcc)
    # The compiler, explicitly. Both the CUDA compiler and the CUDAToolkit root,
    # so find_package(CUDAToolkit) resolves to the JetPack headers rather than to
    # whatever /usr/local/cuda happens to point at.
    set(CMAKE_CUDA_COMPILER "${_lsx_jetson_nvcc}" CACHE FILEPATH "" FORCE)
    set(CUDAToolkit_ROOT "${LSX_JETSON_CUDA_ROOT}" CACHE PATH "" FORCE)
    message(STATUS "Jetson toolchain: CUDA from ${LSX_JETSON_CUDA_ROOT}")
else()
    message(WARNING
        "Jetson toolchain: no nvcc at ${LSX_JETSON_CUDA_ROOT}/bin/nvcc. "
        "JetPack 6.2.1 installs CUDA 12.6 at /usr/local/cuda-12.6; point "
        "-DLSX_JETSON_CUDA_ROOT at it. Do NOT use a desktop toolkit: this board "
        "needs the JetPack build. Proceeding with the toolkit CMake finds, which "
        "find_package(CUDAToolkit REQUIRED) will validate.")
endif()
unset(_lsx_jetson_nvcc)

# Host architecture: aarch64. An absolute `-march=native` would be wrong twice
# over — it is an x86 spelling, and this toolchain is used to BUILD the binaries,
# not to discover the host. An empty processor is a pre-project() probe, so only
# a real answer is worth reporting (and only once).
if(CMAKE_SYSTEM_PROCESSOR AND NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    message(STATUS
        "Jetson toolchain: host reports '${CMAKE_SYSTEM_PROCESSOR}'. This file "
        "assumes aarch64; cross-compiling from x86 needs a fuller toolchain "
        "file than this one.")
endif()

# The two x86 conveniences, off by construction rather than by convention.
set(LSX_ARCH_ZNVER4 OFF CACHE BOOL "" FORCE)
set(LSX_LTO OFF CACHE BOOL "" FORCE)

# No RPATH ladder: an aarch64 build finds its shared libraries through the
# normal loader paths and the JetPack sysroot. Inheriting the AOCC preset's
# $ORIGIN/../_deps/... entries here would point at directories that do not exist
# in this tree.
set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF CACHE BOOL "" FORCE)
