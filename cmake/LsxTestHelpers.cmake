# Shared CTest registration helpers.
#
# The test suite registers ~450 executables; before this module every one of
# them repeated the same six blocks (add_executable, link, includes, defines,
# add_test, set_tests_properties) plus the two-command Windows Arrow-DLL copy.
# The helpers below keep the per-test call to one statement, and — more
# importantly — put the dependency include propagators (`lsx_dep_includes`) and
# the CUDA codegen arch (`LSX_CUDA_ARCH`) in one place, so a change of
# FetchContent directory layout or a Blackwell arch bump does not touch
# hundreds of call sites.
#
# Included from the root CMakeLists.txt before add_subdirectory(), so the
# functions are visible in libs/, apps/, tools/ and tests/.
include_guard(GLOBAL)

# lsx_copy_arrow_dlls(<target> [ARROW_ONLY])
#
# Windows-only: copy the runtime DLLs a test needs next to its executable.
# No-op elsewhere (on Linux shared libraries are found through RPATH).
function(lsx_copy_arrow_dlls target)
    if(NOT WIN32)
        return()
    endif()
    set(_lsx_dlls arrow_shared arrow_cuda_shared)
    if("ARROW_ONLY" IN_LIST ARGN)
        set(_lsx_dlls arrow_shared)
    endif()
    foreach(_lsx_dll IN LISTS _lsx_dlls)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:${_lsx_dll}>"
                "$<TARGET_FILE_DIR:${target}>")
    endforeach()
endfunction()

# lsx_add_test(NAME <name>
#              [SOURCES <src>...]          # default: <name>.cpp
#              [LINK <target>...]          # Catch2 is added automatically
#              [INCLUDE <dir>...]
#              [DEFINE <def>...]
#              [LABEL <label>]            # omit when the grouped label list owns it
#              [TIMEOUT <seconds>]
#              [SKIP]                     # SKIP_RETURN_CODE 4
#              [ENV "<K=V[;K=V]...>"]
#              [NO_DLLS]                  # suppress the Arrow DLL copy
#              [ARROW_DLLS_ONLY])
#
# The Arrow DLL copy is automatic when LINK contains a target that pulls the
# Arrow runtime (lsxcommon / lsxnetwork / lsx_dep_includes / arrow_shared /
# arrow_cuda_shared); pass NO_DLLS to opt out.
function(lsx_add_test)
    cmake_parse_arguments(P
        "SKIP;NO_DLLS;ARROW_DLLS_ONLY"
        "NAME;LABEL;TIMEOUT;LINKER_LANGUAGE"
        "SOURCES;LINK;INCLUDE;DEFINE;ENV"
        ${ARGN})
    if(NOT P_NAME)
        message(FATAL_ERROR "lsx_add_test: NAME is required")
    endif()
    set(name "${P_NAME}")

    if(NOT P_SOURCES)
        set(P_SOURCES "${name}.cpp")
    endif()

    add_executable(${name} ${P_SOURCES})
    if(P_LINKER_LANGUAGE)
        set_target_properties(${name} PROPERTIES
            LINKER_LANGUAGE ${P_LINKER_LANGUAGE})
    endif()
    target_link_libraries(${name} PRIVATE Catch2::Catch2WithMain ${P_LINK})
    if(P_INCLUDE)
        target_include_directories(${name} PRIVATE ${P_INCLUDE})
    endif()
    if(P_DEFINE)
        target_compile_definitions(${name} PRIVATE ${P_DEFINE})
    endif()

    add_test(NAME ${name} COMMAND ${name})
    if(P_LABEL)
        set_tests_properties(${name} PROPERTIES LABELS ${P_LABEL})
    endif()
    if(P_TIMEOUT)
        set_tests_properties(${name} PROPERTIES TIMEOUT ${P_TIMEOUT})
    endif()
    if(P_SKIP)
        set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 4)
    endif()
    if(P_ENV)
        set_tests_properties(${name} PROPERTIES ENVIRONMENT "${P_ENV}")
    endif()

    if(P_NO_DLLS)
        return()
    endif()
    set(_lsx_needs_dlls FALSE)
    foreach(_lsx_link IN LISTS P_LINK)
        if(_lsx_link MATCHES "^(lsxcommon|lsxnetwork|lsx_dep_includes|arrow_shared|arrow_cuda_shared)$")
            set(_lsx_needs_dlls TRUE)
        endif()
    endforeach()
    if(_lsx_needs_dlls)
        if(P_ARROW_DLLS_ONLY)
            lsx_copy_arrow_dlls(${name} ARROW_ONLY)
        else()
            lsx_copy_arrow_dlls(${name})
        endif()
    endif()
endfunction()

# lsx_add_cuda_test(NAME <name>
#                   [OBJECT <objlib>]        # default: <name>_cu
#                   [SOURCES <src>...]       # default: <name>.cu
#                   [OBJECT_LINK <target>...]
#                   [LINK <target>...]       # executable-only extras
#                   [INCLUDE <dir>...]
#                   [EXE_INCLUDE <dir>...]
#                   [DEFINE <def>...]
#                   [FMAD_OFF]               # add -fmad=false (bit-exact kernels)
#                   [LABEL ...] [TIMEOUT ...] [SKIP] [ENV ...] [NO_DLLS])
#
# Kernel-parity shape: a .cu OBJECT library for the device TUs (isolated from
# host flags, see cuda-cmake-build-gotchas) linked into a CXX executable. The
# OBJECT library gets `lsx_dep_includes` (Arrow/spdlog/simdjson/absl usage
# requirements) and the standard CCCL/MSVC defines; the executable links
# lsxcommon and Catch2. Arch comes from LSX_CUDA_ARCH.
function(lsx_add_cuda_test)
    cmake_parse_arguments(P
        "SKIP;FMAD_OFF;NO_DLLS;ARROW_DLLS_ONLY"
        "NAME;OBJECT;LABEL;TIMEOUT"
        "SOURCES;OBJECT_LINK;LINK;INCLUDE;EXE_INCLUDE;DEFINE;ENV"
        ${ARGN})
    if(NOT P_NAME)
        message(FATAL_ERROR "lsx_add_cuda_test: NAME is required")
    endif()
    set(name "${P_NAME}")

    if(NOT P_OBJECT)
        set(P_OBJECT "${name}_cu")
    endif()
    if(NOT P_SOURCES)
        set(P_SOURCES "${name}.cu")
    endif()

    add_library(${P_OBJECT} OBJECT ${P_SOURCES})
    set_target_properties(${P_OBJECT} PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES "${LSX_CUDA_ARCH}"
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON)
    target_compile_options(${P_OBJECT} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>)
    if(P_FMAD_OFF)
        target_compile_options(${P_OBJECT} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>)
    endif()
    target_include_directories(${P_OBJECT} PRIVATE
        "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
        "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
        "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
        ${P_INCLUDE})
    target_compile_definitions(${P_OBJECT} PRIVATE
        THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
        CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
        LOGESTIX_HAS_CUDA
        LSX_KERNEL_BUDGETS_FILE="${LSX_KERNEL_BUDGETS}"
        ${P_DEFINE})
    target_link_libraries(${P_OBJECT} PRIVATE
        lsx_dep_includes Catch2::Catch2WithMain ${P_OBJECT_LINK})

    add_executable(${name})
    set_target_properties(${name} PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(${name} PRIVATE
        ${P_OBJECT} lsxcommon Catch2::Catch2WithMain ${P_LINK})
    if(P_EXE_INCLUDE)
        target_include_directories(${name} PRIVATE ${P_EXE_INCLUDE})
    endif()

    add_test(NAME ${name} COMMAND ${name})
    if(P_LABEL)
        set_tests_properties(${name} PROPERTIES LABELS ${P_LABEL})
    endif()
    if(P_TIMEOUT)
        set_tests_properties(${name} PROPERTIES TIMEOUT ${P_TIMEOUT})
    endif()
    if(P_SKIP)
        set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 4)
    endif()
    if(P_ENV)
        set_tests_properties(${name} PROPERTIES ENVIRONMENT "${P_ENV}")
    endif()

    if(NOT P_NO_DLLS)
        if(P_ARROW_DLLS_ONLY)
            lsx_copy_arrow_dlls(${name} ARROW_ONLY)
        else()
            lsx_copy_arrow_dlls(${name})
        endif()
    endif()
endfunction()
