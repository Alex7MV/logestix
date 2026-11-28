# Probe the REAL arch module (cmake/LsxArch.cmake), not a copy of its rules.
#
# Two questions, one script, because both are about the same module:
#
#   -DPROBE_ARCH=<arch[;arch...]>   what does the validator emit / refuse?
#   -DPROBE_HAS12X=<arch[;arch...]> does this arch list target a 12x arch?
#
# Run by tests/test_infrastructure_guard.cpp as
#   cmake -DLsxArch=<abs> -DPROBE_ARCH=86;87 -P tests/cmake/probe_arch.cmake
#
# A rejected arch makes lsx_arch_problem()/lsx_cuda_gencode_flags() call
# message(FATAL_ERROR), which in script mode exits non-zero with the reason on
# stderr — so a rejection is distinguishable from a crash and the
# operator-facing wording is checkable.
#
# Why a script and not a text scan: an earlier version of this guard checked the
# root CMakeLists.txt for the *shape* of the pattern. It passed while the pattern
# still rejected `86;87`, because "contains 12[01]" is not "accepts 86". Only
# running the function answers the question the guard asks.

if(NOT DEFINED LsxArch)
    message(FATAL_ERROR "probe_arch.cmake needs -DLsxArch=<path to LsxArch.cmake>")
endif()

include("${LsxArch}")

if(DEFINED PROBE_HAS12X)
    set(LSX_CUDA_ARCH "${PROBE_HAS12X}")
    lsx_arch_has_12x(_probe_has12x)
    message(STATUS "HAS12X=${_probe_has12x}")

    # The object-lib name this configure would build. The tests assert the
    # Ampere-only answer is the STUB and never the arch-'a' object lib, so the
    # decision is a value the guard reads rather than a branch it re-derives.
    lsx_tc_object_lib(_probe_tc_lib)
    message(STATUS "TC_LIB=${_probe_tc_lib}")
else()
    if(NOT DEFINED PROBE_ARCH)
        message(FATAL_ERROR
            "probe_arch.cmake needs -DPROBE_ARCH= or -DPROBE_HAS12X=")
    endif()
    set(LSX_CUDA_ARCH "${PROBE_ARCH}")
    lsx_cuda_gencode_flags(_probe_flags)
    message(STATUS "GENCODE=${_probe_flags}")
endif()
