# Probe the REAL arch validator (cmake/LsxArch.cmake), not a copy of its regex.
#
# Run by tests/test_infrastructure_guard.cpp as
#   cmake -DPROBE_ARCH=<arch> -DLsxArch=<abs path> -P probe_arch.cmake
#
# Prints the emitted --generate-code flags on success; a rejected arch makes
# lsx_cuda_gencode_flags() call message(FATAL_ERROR), which in script mode exits
# non-zero with the reason on stderr. The caller reads the exit code and the
# combined output, so a rejection is distinguishable from a crash and the
# operator-facing wording is checkable.
#
# Why a script and not a text scan: an earlier version of this guard checked the
# root CMakeLists.txt for the *shape* of the pattern. It passed while the
# pattern still rejected `86;87`, because "contains 12[01]" is not "accepts 86".
# Only running the function answers the question the guard asks.

if(NOT DEFINED LsxArch)
    message(FATAL_ERROR "probe_arch.cmake needs -DLsxArch=<path to LsxArch.cmake>")
endif()
if(NOT DEFINED PROBE_ARCH)
    message(FATAL_ERROR "probe_arch.cmake needs -DPROBE_ARCH=<arch[;arch...]>")
endif()

include("${LsxArch}")

set(LSX_CUDA_ARCH "${PROBE_ARCH}")
lsx_cuda_gencode_flags(_probe_flags)
message(STATUS "GENCODE=${_probe_flags}")
