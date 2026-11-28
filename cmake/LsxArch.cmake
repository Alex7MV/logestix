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
