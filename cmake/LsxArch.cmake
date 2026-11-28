# CUDA architecture policy: which CCs this tree builds for, and how they reach nvcc.
#
# ONE source of truth, in one file, because it is read by three audiences that
# used to disagree:
#   * the build, through lsx_cuda_gencode_flags() (every device target);
#   * tests/test_infrastructure_guard.cpp, which runs this module through
#     `cmake -P` to prove what the validator accepts (a text scan of the pattern
#     cannot answer that — it passed for two years while every Ampere arch was
#     rejected);
#   * whoever adds the next arch, who needs the suffix rule stated once.
#
# Included from the root CMakeLists.txt before any target is created. It defines
# functions and variables only: no targets, no cache entries (the default value
# of LSX_CUDA_ARCH stays in the root file, where a reader looks for it).

include_guard(GLOBAL)

# Accepted spellings.
#
#   12[01]            Blackwell consumer (sm_120, sm_121) — the shipped default
#   12[01]f / 12[01]a family-specific / architecture-specific Blackwell
#   86                 GA102 (RTX 3090)
#   87                 GA10B (Jetson Orin)
#
# Deliberately absent: 89 (Ada: FP8 but not a target), 80 and 90 (Hopper and the
# Ampere datacenter sibling — AGENTS.md forbids them), everything older. A row
# exists here only for silicon this tree builds for; a broader pattern would let
# a configure produce a binary nothing tested.
set(LSX_ARCH_SUPPORTED "120, 121, 120f, 121f (Blackwell consumer); 86, 87 (Ampere GA10x)")

# lsx_arch_problem(<arch> <out_var>)
#
# Sets <out_var> to a human-readable reason when <arch> is not a supported entry,
# or to "" when it is. Split from the FATAL so the probe script and the build can
# share the rule while reporting it differently.
function(lsx_arch_problem arch out_var)
    set(_why "")
    if(NOT arch MATCHES "^(12[01]|8[67])[af]?$")
        set(_why
            "LSX_CUDA_ARCH entry '${arch}' is not a supported target. "
            "Supported: ${LSX_ARCH_SUPPORTED}.")
    elseif(arch MATCHES "^8[67][af]$")
        # There is no sm_86a and no sm_86f: the family-specific and
        # architecture-specific targets are Blackwell-only (tcgen05 lives there,
        # GA10x has no such instruction). Accepting the suffix would send nvcc an
        # arch it rejects with a toolkit-shaped message instead of a spelling one.
        set(_why
            "LSX_CUDA_ARCH entry '${arch}' uses a suffix that only exists on "
            "Blackwell: sm_86/sm_87 have no family-specific (f) or "
            "architecture-specific (a) target. Supported: ${LSX_ARCH_SUPPORTED}.")
    endif()
    set(${out_var} "${_why}" PARENT_SCOPE)
endfunction()

# lsx_arch_has_12x(<out_var>)
#
# TRUE iff LSX_CUDA_ARCH targets at least one Blackwell consumer arch (12x).
#
# This is the switch the arch-'a' object libraries hang off, and the reason is
# hard, not cosmetic: `src/lsx_tc_gemm.cu` uses the `kind::mxf4nvf4` block-scaled
# MMA, whose PTX assembles ONLY for the architecture-specific targets
# (compute_120a/compute_121a). It cannot be compiled for sm_86 at any spelling, so
# in an Ampere-only build the object lib must not be created at all — otherwise
# nvcc is asked for an arch the configure does not target (a wasted compile on a
# 13.5 toolkit, a hard failure on CUDA 12.6, which is the Orin half of this
# branch) and the resulting object could never be launched anyway.
#
# One predicate, read by the build AND by the guard test through `cmake -P`, so
# the two cannot disagree about which configures get the kernels.
function(lsx_arch_has_12x out_var)
    set(_arches ${LSX_CUDA_ARCH})
    set(_has FALSE)
    foreach(arch IN LISTS _arches)
        if(arch MATCHES "^12[01]")
            set(_has TRUE)
        endif()
    endforeach()
    set(${out_var} ${_has} PARENT_SCOPE)
endfunction()

# lsx_tc_object_lib(<out_var>)
#
# The object library that provides the `lsxcommon::lsx_tc::*` entry points in THIS
# configure: the real arch-'a' kernels when a 12x arch is targeted, otherwise the
# stub that reports `cudaErrorNotSupported` for every device call.
#
# The stub exists because the entry points are part of a PUBLIC header used from
# nine host/device files (the NVFP4 dispatcher, the Kimi and MoE batched chains,
# the perf knobs, ModelBase's weight epoch). Compiling those out with #ifdefs
# would put an arch condition in nine numerics-adjacent files and make the built
# code depend on the arch list; one translation unit that answers "not supported"
# is the same contract the real kernels already give on a non-12x device — they
# check `prop.major` and return exactly that.
function(lsx_tc_object_lib out_var)
    lsx_arch_has_12x(_has12x)
    if(_has12x)
        set(${out_var} "lsxcommon_tc_cu" PARENT_SCOPE)
    else()
        set(${out_var} "lsxcommon_tc_stub" PARENT_SCOPE)
    endif()
endfunction()

# lsx_cuda_gencode_flags(<out_var> [ARCH_LIST <arch>...])
#
# Expands LSX_CUDA_ARCH (or ARCH_LIST) into nvcc's own --generate-code spelling,
# which is the only form CMake passes through unharmed.
#
# Why not `set_target_properties(... CUDA_ARCHITECTURES ...)` directly: 30 test
# targets used to spell `--generate-code=arch=compute_120,code=sm_120` by hand, so
# the arch lived in 31 places; and CMake 4.x does not know the family suffix and,
# measured on 4.2.3, silently replaces the whole list with `compute_75` for
# "120f" — a binary that still runs through PTX JIT and is slower than the plain
# sm_120 it replaced, warning about nothing.
#
# The emitted shape is the same one CMake uses for `<N>-real`: PTX for the virtual
# arch plus SASS for the real one, so a device runs the SASS it was built for and
# falls back to the PTX only on a newer arch.
function(lsx_cuda_gencode_flags out_var)
    cmake_parse_arguments(P "" "" "ARCH_LIST" ${ARGN})
    set(_arches ${LSX_CUDA_ARCH})
    if(P_ARCH_LIST)
        set(_arches ${P_ARCH_LIST})
    endif()
    if(NOT _arches)
        message(FATAL_ERROR "LSX_CUDA_ARCH is empty: at least one target is required")
    endif()

    set(flags "")
    foreach(arch IN LISTS _arches)
        lsx_arch_problem("${arch}" _why)
        if(_why)
            message(FATAL_ERROR "${_why}")
        endif()
        list(APPEND flags
            "--generate-code=arch=compute_${arch},code=[compute_${arch},sm_${arch}]")
    endforeach()
    set(${out_var} "${flags}" PARENT_SCOPE)
endfunction()
