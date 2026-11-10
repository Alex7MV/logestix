lsx_warning_policy()

# GPU-keyed kernel budgets: tests read tests/perf/kernel_budgets.json, and
# LSX_KERNEL_BUDGET_WRITE=1 turns a sweep into a recalibration. Defined before
# any target that consumes it (CMake set() is order-dependent).
set(LSX_KERNEL_BUDGETS "${CMAKE_CURRENT_SOURCE_DIR}/perf/kernel_budgets.json")

# ── Model test registration (generated from models/*.json) ────────────────
# The add_test/set_tests_properties calls for every manifest-claimed target are
# emitted at configure time into ${CMAKE_BINARY_DIR}/generated/model_targets.cmake.
# Build wiring for those targets (add_executable, the CUDA OBJECT libraries,
# per-target includes and defines) stays in this file: it is per-target compile
# reality, not a projection of the model registry. What is generated is the part
# that used to drift - labels, timeouts, skip codes, resource locks, environments.
# Included near the TOP so later grouped statements can address these tests.
if(EXISTS "${LSX_MANIFEST_TARGETS}")
    include("${LSX_MANIFEST_TARGETS}")
else()
    message(FATAL_ERROR "model_targets.cmake is missing - run cmake configure "
                        "from a clean build tree (${LSX_MANIFEST_TARGETS})")
endif()

# ── Repository test executable ─────────────────────────────────
add_executable(test_core test_core.cpp)
target_link_libraries(test_core PRIVATE lsxcommon arrow_shared arrow_cuda_shared simdjson::simdjson httplib::httplib Catch2::Catch2WithMain)
# CCCL now comes from the CUDA toolkit; the band (defined in the root
# CMakeLists.txt) is the single source of truth shared with the configure-time
# guard and the STATIC_REQUIREs in test_core.cpp.
target_compile_definitions(test_core PRIVATE
    LSX_CCCL_VERSION_MIN=${LSX_CCCL_BAND_MIN}
    LSX_CCCL_VERSION_MAX=${LSX_CCCL_BAND_MAX})

# Copy arrow.dll and arrow_cuda.dll into the test executable directory
lsx_copy_arrow_dlls(test_core)

add_test(NAME test_core COMMAND test_core)

# ── Shared fastload helper test (host-only, no CUDA allocations) ────────
add_executable(test_fast_load test_fast_load.cpp)
target_link_libraries(test_fast_load PRIVATE lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_fast_load PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
add_test(NAME test_fast_load COMMAND test_fast_load)

# ── Safetensors descriptor layer (TensorInfo validation + reader) ──────
# Host-only: synthetic shards under the temp dir, no GPU. Owns the validated
# reader every converter calls.
lsx_add_test(NAME test_safetensors LINK lsxcommon)

# ── Shared converter write helpers (host-only, synthetic batches) ──────
lsx_add_test(NAME test_converter_helpers LINK lsxcommon)

# Opt-in libFuzzer harness for the safetensors header parser (tests/fuzz/ is
# outside the naming/label guards because it is not a ctest). Off by default:
# a fuzzer needs a corpus, a time budget and an instrumented library build —
# the deterministic mutation tests in test_safetensors run on every ctest.
option(LSX_BUILD_FUZZERS "Build the libFuzzer harnesses (Clang only)" OFF)
if(LSX_BUILD_FUZZERS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR
            "LSX_BUILD_FUZZERS requires Clang (-fsanitize=fuzzer); see "
            "tests/fuzz/safetensors_fuzz.cpp for the full command line")
    endif()
    add_executable(fuzz_safetensors fuzz/safetensors_fuzz.cpp)
    target_link_libraries(fuzz_safetensors PRIVATE lsxcommon)
    target_compile_options(fuzz_safetensors PRIVATE
        -fsanitize=fuzzer,address -fno-omit-frame-pointer)
    target_link_options(fuzz_safetensors PRIVATE -fsanitize=fuzzer,address)
endif()

# ── AliceAI-Foundation-80B-A3B family (new family; stages land progressively) ──
add_executable(test_alice_moe_model test_alice_moe_model.cpp)
target_link_libraries(test_alice_moe_model PRIVATE lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_alice_moe_model PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)

# ── Kimi-K3 synthetic converter E2E (label convert, bounded) ────────────────
add_executable(test_kimi_k3_convert test_kimi_k3_convert.cpp)
target_link_libraries(test_kimi_k3_convert PRIVATE lsxcommon
    simdjson::simdjson Catch2::Catch2WithMain)

# ── Kimi-K3.1-Pro synthetic converter (Gated DSA indexer, variant gate) ─────
# Same synthetic-checkpoint discipline as the K3 converter above; the gate is
# the wrapper-level model_type + dsa_config/indexer-tensor agreement.
add_executable(test_kimi_pro_convert test_kimi_pro_convert.cpp)
target_link_libraries(test_kimi_pro_convert PRIVATE lsxcommon
    simdjson::simdjson Catch2::Catch2WithMain)

add_executable(test_alice_moe_convert test_alice_moe_convert.cpp)
target_link_libraries(test_alice_moe_convert PRIVATE lsxcommon
    simdjson::simdjson Catch2::Catch2WithMain)

add_executable(test_alice_moe_loader test_alice_moe_loader.cpp)
target_link_libraries(test_alice_moe_loader PRIVATE lsxcommon
    simdjson::simdjson Catch2::Catch2WithMain)

# MiniMax-M3 kernel parity (CUDA OBJECT lib + CXX-linked executable).
add_library(test_minimax_cu OBJECT
    test_minimax_moe_kernels.cu
)
set_target_properties(test_minimax_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_minimax_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(test_minimax_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
# Arrow headers are reachable from lsx_moe_inference.h (expert_store.h), so
# this OBJECT lib needs the same include set the library CUDA targets use.
target_include_directories(test_minimax_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    )
target_link_libraries(test_minimax_cu PRIVATE Catch2::Catch2WithMain
        lsx_dep_includes
    arrow_shared)

add_executable(test_minimax_moe_kernels)
set_target_properties(test_minimax_moe_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_minimax_moe_kernels PRIVATE test_minimax_cu lsxcommon
    Catch2::Catch2WithMain)

# Xing4.0-29B-A4B kernel parity (CUDA OBJECT lib + CXX-linked executable).
add_library(test_xing_cu OBJECT
    test_xing_moe_kernels.cu
)
set_target_properties(test_xing_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_xing_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(test_xing_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_include_directories(test_xing_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_xing_cu PRIVATE Catch2::Catch2WithMain)
target_compile_definitions(test_xing_cu PRIVATE
    LSX_KERNEL_BUDGETS_FILE="${LSX_KERNEL_BUDGETS}")

add_executable(test_xing_moe_kernels)
set_target_properties(test_xing_moe_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_xing_moe_kernels PRIVATE test_xing_cu lsxcommon
    Catch2::Catch2WithMain)

# Alice kernel parity (CUDA OBJECT lib + CXX-linked executable).
add_library(test_alice_cu OBJECT
    test_alice_moe_kernels.cu
)
set_target_properties(test_alice_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_alice_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(test_alice_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_include_directories(test_alice_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_alice_cu PRIVATE Catch2::Catch2WithMain)

add_executable(test_alice_moe_kernels)
set_target_properties(test_alice_moe_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_alice_moe_kernels PRIVATE test_alice_cu lsxcommon Catch2::Catch2WithMain)

add_executable(test_alice_tool_framing test_alice_tool_framing.cpp)
target_link_libraries(test_alice_tool_framing PRIVATE lsxcommon Catch2::Catch2WithMain)


# ── ModelBase shared geometry-validator test (host-only) ────────────────
add_executable(test_geometry_guard test_geometry_guard.cpp)
target_link_libraries(test_geometry_guard PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_geometry_guard COMMAND test_geometry_guard)


# ── GPU perf auto-tuning policy (host-only) ─────────────────────────────
add_executable(test_gpu_perf test_gpu_perf.cpp)
target_link_libraries(test_gpu_perf PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_gpu_perf COMMAND test_gpu_perf)
set_tests_properties(test_gpu_perf PROPERTIES SKIP_RETURN_CODE 4)

# ── Perf baseline reference logic (host-only) ───────────────────────────
add_executable(test_perf_baseline_logic test_perf_baseline_logic.cpp)
target_link_libraries(test_perf_baseline_logic PRIVATE lsxcommon
                      Catch2::Catch2WithMain)
# The registry-pairing case reads tests/perf/perf_baselines.json and the perf
# test TU from the source tree, so it needs the source root (ctest runs the
# binary with the build dir as CWD).
target_compile_definitions(test_perf_baseline_logic PRIVATE
    LSX_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    LSX_MANIFEST_GENERATED_DIR="${LSX_MANIFEST_GENERATED_DIR}")
add_test(NAME test_perf_baseline_logic COMMAND test_perf_baseline_logic)

# ── GPU board-power sampling (host-only; NVML cases SKIP when unavailable) ──
add_executable(test_gpu_power test_gpu_power.cpp)
target_link_libraries(test_gpu_power PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_gpu_power COMMAND test_gpu_power)
set_tests_properties(test_gpu_power PROPERTIES SKIP_RETURN_CODE 4)


# ── Zero-copy Arrow IPC view load test (the loaders' fast path) ──────
add_executable(test_schema_views test_schema_views.cpp)
target_link_libraries(test_schema_views PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_schema_views)

add_test(NAME test_schema_views COMMAND test_schema_views)

# ── Arrow IPC validation: malformed batches must fail, not over-read ──
add_executable(test_arrow_validation test_arrow_validation.cpp)
target_link_libraries(test_arrow_validation PRIVATE lsxcommon
    Catch2::Catch2WithMain)
add_test(NAME test_arrow_validation COMMAND test_arrow_validation)

# Guard: Arrow IPC usage — mmap-only reads, recorded UnpackBatch verdicts,
# atomic writes, and an audit matrix that covers every adapted model.
lsx_add_test(NAME test_arrow_usage_guard
    DEFINE "LSX_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\"")

# Environment-gated validator over the CONVERTED vault trees (26 models);
# SKIP when no tree is present. Footer pass for every .arrow plus a bounded
# payload sample (LSX_ARROW_AUDIT_{ROOT,FULL,MAX_BYTES}).
add_executable(test_arrow_tree_integrity test_arrow_tree_integrity.cpp)
target_link_libraries(test_arrow_tree_integrity PRIVATE lsxcommon
    Catch2::Catch2WithMain)
add_test(NAME test_arrow_tree_integrity COMMAND test_arrow_tree_integrity)
set_tests_properties(test_arrow_tree_integrity PROPERTIES LABELS convert)
set_tests_properties(test_arrow_tree_integrity PROPERTIES TIMEOUT 600)
set_tests_properties(test_arrow_tree_integrity PROPERTIES SKIP_RETURN_CODE 4)


# ── DeepSeek loader round-trip test ─────────────────────────────────
add_executable(test_deepseek_moe_loader test_deepseek_moe_loader.cpp)
target_link_libraries(test_deepseek_moe_loader PRIVATE lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_deepseek_moe_loader PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")

lsx_copy_arrow_dlls(test_deepseek_moe_loader)


# ── Canonical NVFP4 full-E4M3 block-scale codec (model-neutral, host) ──
add_executable(test_nvfp4_scale test_nvfp4_scale.cpp)
target_link_libraries(test_nvfp4_scale PRIVATE lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_nvfp4_scale PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
# The single-source guard below scans tests/ for local re-implementations of
# the block-scale codec, so it needs the repo root.
file(TO_CMAKE_PATH "${PROJECT_SOURCE_DIR}" LSX_SRC_DIR_FWD_NVFP4)
target_compile_definitions(test_nvfp4_scale
    PRIVATE "LSX_SOURCE_DIR=\"${LSX_SRC_DIR_FWD_NVFP4}\"")

lsx_copy_arrow_dlls(test_nvfp4_scale)

add_test(NAME test_nvfp4_scale COMMAND test_nvfp4_scale)
set_tests_properties(test_nvfp4_scale PROPERTIES LABELS fast)

# ── DeepSeek-V4-Flash model lifecycle-stage test (host-only):
#    classification of the converted layout (detect_model_type / factory) +
#    topology contract (load_topology) on a synthetic logestix fixture.
#    DeepSeek has no converter in the repo — there is no convert stage by design.
add_executable(test_deepseek_moe_model test_deepseek_moe_model.cpp)
target_link_libraries(test_deepseek_moe_model PRIVATE lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_deepseek_moe_model PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")

lsx_copy_arrow_dlls(test_deepseek_moe_model)


# ── DeepSeek-V4.1-Flash: frozen passport (model stage, host-only) ────────
add_executable(test_deepseek_moe_v41_model test_deepseek_moe_v41_model.cpp)
target_link_libraries(test_deepseek_moe_v41_model PRIVATE lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_deepseek_moe_v41_model PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")

lsx_copy_arrow_dlls(test_deepseek_moe_v41_model)


# ── DeepSeek-V4.1-Flash: topology contract (load stage, host-only) ───────
add_executable(test_deepseek_moe_v41_loader test_deepseek_moe_v41_loader.cpp)
target_link_libraries(test_deepseek_moe_v41_loader PRIVATE lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_deepseek_moe_v41_loader PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")

lsx_copy_arrow_dlls(test_deepseek_moe_v41_loader)


# ── DeepSeek-V4.1-Flash: converter (synthetic e2e, host-only) ────────────
add_executable(test_deepseek_moe_v41_convert test_deepseek_moe_v41_convert.cpp)
target_compile_definitions(test_deepseek_moe_v41_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_deepseek_moe_v41_convert PRIVATE lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_deepseek_moe_v41_convert PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")

# ── DeepSeek-V4-Pro-0813: converter (synthetic e2e, host-only) ───────────
add_executable(test_deepseek_moe_pro_convert test_deepseek_moe_pro_convert.cpp)
target_compile_definitions(test_deepseek_moe_pro_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_deepseek_moe_pro_convert PRIVATE lsxcommon
                      Catch2::Catch2WithMain)
target_include_directories(test_deepseek_moe_pro_convert PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")

# ── DeepSeek-V4-Pro-0813: load stage (host + CUDA, synthetic fixture) ────
add_executable(test_deepseek_moe_pro_loader test_deepseek_moe_pro_loader.cpp)
target_link_libraries(test_deepseek_moe_pro_loader PRIVATE lsxcommon
                      Catch2::Catch2WithMain)
target_include_directories(test_deepseek_moe_pro_loader PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
lsx_copy_arrow_dlls(test_deepseek_moe_pro_loader)

# ── DeepSeek-V4-Pro-0813: model stage (host-only) ────────────────────────
add_executable(test_deepseek_moe_pro_model test_deepseek_moe_pro_model.cpp)
target_link_libraries(test_deepseek_moe_pro_model PRIVATE lsxcommon
                      Catch2::Catch2WithMain)
target_include_directories(test_deepseek_moe_pro_model PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
lsx_copy_arrow_dlls(test_deepseek_moe_pro_model)

# ── DeepSeek-V4-Pro-0813: fast-load primitives (CUDA OBJECT lib) ─────────
add_library(test_deepseek_moe_pro_load_fast_cu OBJECT
            test_deepseek_moe_pro_load_fast.cu)
set_target_properties(test_deepseek_moe_pro_load_fast_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_deepseek_moe_pro_load_fast_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_deepseek_moe_pro_load_fast_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
)
target_compile_definitions(test_deepseek_moe_pro_load_fast_cu PRIVATE
    LOGESTIX_HAS_CUDA
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_deepseek_moe_pro_load_fast_cu PRIVATE
        lsx_dep_includes
    Catch2::Catch2WithMain)
add_executable(test_deepseek_moe_pro_load_fast)
set_target_properties(test_deepseek_moe_pro_load_fast PROPERTIES
    LINKER_LANGUAGE CXX)
target_link_libraries(test_deepseek_moe_pro_load_fast PRIVATE
    test_deepseek_moe_pro_load_fast_cu lsxcommon Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_deepseek_moe_pro_load_fast)

# ── DeepSeek-V4-Pro-0813: kernel stage (family rmsnorm) ──────────────────
add_library(test_deepseek_moe_pro_kernels_cu OBJECT
            test_deepseek_moe_pro_kernels.cu)
set_target_properties(test_deepseek_moe_pro_kernels_cu PROPERTIES
    CUDA_ARCHITECTURES "${LSX_CUDA_ARCH}")
target_compile_options(test_deepseek_moe_pro_kernels_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_include_directories(test_deepseek_moe_pro_kernels_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    )
target_compile_definitions(test_deepseek_moe_pro_kernels_cu PRIVATE
    LOGESTIX_HAS_CUDA)
target_link_libraries(test_deepseek_moe_pro_kernels_cu PRIVATE
        lsx_dep_includes
    Catch2::Catch2WithMain)
add_executable(test_deepseek_moe_pro_kernels)
set_target_properties(test_deepseek_moe_pro_kernels PROPERTIES
    LINKER_LANGUAGE CXX)
target_link_libraries(test_deepseek_moe_pro_kernels PRIVATE
    test_deepseek_moe_pro_kernels_cu lsxcommon Catch2::Catch2WithMain)

# ── DeepSeek-V4-Pro-0813: SelfCheck (real model, SKIP 4) ─────────────────

# ── DeepSeek-V4-Pro-0813: FSI/CODE/DET parity (goldens, SKIP 4) ──────────


# ── DeepSeek-V4.1-Flash: dense dequant kernels (CUDA) ────────────────────
add_library(test_deepseek_moe_v41_dequant_cu OBJECT
            test_deepseek_moe_v41_dequant.cu)
set_target_properties(test_deepseek_moe_v41_dequant_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_deepseek_moe_v41_dequant_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_deepseek_moe_v41_dequant_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
)
target_compile_definitions(test_deepseek_moe_v41_dequant_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
)
target_link_libraries(test_deepseek_moe_v41_dequant_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)

add_executable(test_deepseek_moe_v41_dequant)
set_target_properties(test_deepseek_moe_v41_dequant PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_deepseek_moe_v41_dequant PRIVATE
    test_deepseek_moe_v41_dequant_cu lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_deepseek_moe_v41_dequant)


# ── DeepSeek-V4.1-Flash: pinned load + expert store (CUDA) ───────────────
add_library(test_deepseek_moe_v41_load_fast_cu OBJECT
            test_deepseek_moe_v41_load_fast.cu)
set_target_properties(test_deepseek_moe_v41_load_fast_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_deepseek_moe_v41_load_fast_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_deepseek_moe_v41_load_fast_cu PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
)
target_compile_definitions(test_deepseek_moe_v41_load_fast_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
)
target_link_libraries(test_deepseek_moe_v41_load_fast_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)

add_executable(test_deepseek_moe_v41_load_fast)
set_target_properties(test_deepseek_moe_v41_load_fast PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_deepseek_moe_v41_load_fast PRIVATE
    test_deepseek_moe_v41_load_fast_cu lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_deepseek_moe_v41_load_fast)


# ── DeepSeek-V4.1-Flash: device-math kernels (CUDA) ─────────────────────
add_library(test_deepseek_moe_v41_kernels_cu OBJECT
            test_deepseek_moe_v41_kernels.cu)
set_target_properties(test_deepseek_moe_v41_kernels_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_deepseek_moe_v41_kernels_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_deepseek_moe_v41_kernels_cu PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
)
target_compile_definitions(test_deepseek_moe_v41_kernels_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
)
target_link_libraries(test_deepseek_moe_v41_kernels_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)

add_executable(test_deepseek_moe_v41_kernels)
set_target_properties(test_deepseek_moe_v41_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_deepseek_moe_v41_kernels PRIVATE
    test_deepseek_moe_v41_kernels_cu lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_deepseek_moe_v41_kernels)


# ── DeepSeek-V4.1-Flash: pipeline smoke (synthetic full geometry) ────────
add_library(test_deepseek_moe_v41_pipeline_cu OBJECT
            test_deepseek_moe_v41_pipeline.cu)
set_target_properties(test_deepseek_moe_v41_pipeline_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_deepseek_moe_v41_pipeline_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_deepseek_moe_v41_pipeline_cu PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
)
target_compile_definitions(test_deepseek_moe_v41_pipeline_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
)
target_link_libraries(test_deepseek_moe_v41_pipeline_cu PRIVATE
        lsx_dep_includes
    Catch2::Catch2WithMain absl::strings)

add_executable(test_deepseek_moe_v41_pipeline)
set_target_properties(test_deepseek_moe_v41_pipeline PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_deepseek_moe_v41_pipeline PRIVATE
    test_deepseek_moe_v41_pipeline_cu lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_deepseek_moe_v41_pipeline)


# ── DeepSeek-V4.1-Flash: FP4 activation quant (CUDA) ────────────────────
add_library(test_deepseek_moe_v41_actquant_cu OBJECT
            test_deepseek_moe_v41_actquant.cu)
set_target_properties(test_deepseek_moe_v41_actquant_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_deepseek_moe_v41_actquant_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_deepseek_moe_v41_actquant_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
)
target_compile_definitions(test_deepseek_moe_v41_actquant_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
)
target_link_libraries(test_deepseek_moe_v41_actquant_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)

add_executable(test_deepseek_moe_v41_actquant)
set_target_properties(test_deepseek_moe_v41_actquant PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_deepseek_moe_v41_actquant PRIVATE
    test_deepseek_moe_v41_actquant_cu lsxcommon Catch2::Catch2WithMain)


# ── DeepSeek-V4.1-Flash: real-model selfcheck (Paris, env opt-in) ────────

# ── Shared pread pool (pinned expert staging, host-only, no GPU) ───────
lsx_add_test(NAME test_pread_pool LINK lsxcommon)
# The label is set on the next line in the literal properties form on purpose:
# test_ctest_labels_guard resolves labels by scanning for that call and does not
# read lsx_add_test's LABEL argument — the same limitation that made
# test_cuda_toolkit_guard's move to fast need a literal call. Weakening the
# guard is not the fix, and a comment that quotes the scanned pattern counts as
# a registration to it.
set_tests_properties(test_pread_pool PROPERTIES LABELS fast)

# ── Staging-ring bandwidth floor (pinned pread + DMA) ─────────────────
# A budget test: it SKIPs without a record for this GPU and takes the shared gpu
# resource lock, so it is registered in LSX_KERNEL_BUDGET_TESTS below as well.
lsx_add_cuda_test(NAME test_expert_store_stage)
set_tests_properties(test_expert_store_stage PROPERTIES LABELS fast)

# ── Shared ExpertStore test (LRU loader for MoE experts) ──────────────
add_executable(test_expert_store test_expert_store.cpp)
target_link_libraries(test_expert_store PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_expert_store)

add_test(NAME test_expert_store COMMAND test_expert_store)
set_tests_properties(test_expert_store PROPERTIES SKIP_RETURN_CODE 4)

# ── Qwen3.5-9B converter test (name parser + index + round-trip) ────────
add_executable(test_qwen_dense_convert test_qwen_dense_convert.cpp)
target_compile_definitions(test_qwen_dense_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_dense_convert PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_qwen_dense_convert)


# ── Granite-4.2-30B converter test (classifier + index + arrow layout) ──
add_executable(test_granite_dense_convert test_granite_dense_convert.cpp)
target_compile_definitions(test_granite_dense_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_granite_dense_convert PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_granite_dense_convert)


# ── Granite-Switch-4.1-30B converter test (classifier + index + arrow layout) ──
add_executable(test_granite_switch_convert test_granite_switch_convert.cpp)
target_compile_definitions(test_granite_switch_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
# The residual cases decode the written NVFP4 records with the canonical
# block-scale codec (nvfp4_scale.h), like test_phi_dense_convert.
target_include_directories(test_granite_switch_convert PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_granite_switch_convert PRIVATE lsxcommon Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_granite_switch_convert)

# ── Granite-Switch-4.1-30B loader test ──
add_executable(test_granite_switch_loader test_granite_switch_loader.cpp)
target_compile_definitions(test_granite_switch_loader PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_granite_switch_loader PRIVATE lsxcommon Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_granite_switch_loader)

# ── Granite-Switch request preparation (host-only) ──
add_executable(test_granite_switch_prepare test_granite_switch_prepare.cpp)
target_link_libraries(test_granite_switch_prepare PRIVATE lsxcommon Catch2::Catch2WithMain)

# ── Granite-Switch-4.1-30B model test ──
add_executable(test_granite_switch_model test_granite_switch_model.cpp)
target_compile_definitions(test_granite_switch_model PRIVATE
    # Absolute, because ctest runs with CWD = the build dir and the source's own
    # fallback is the relative "tests/fixtures/granite_switch": without this the
    # real-weight tool round-trip reads a path that does not exist from there and
    # SKIPs as "golden not captured" while the golden sits in the source tree.
    LSX_GRANITE_SWITCH_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/granite_switch"
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_granite_switch_model PRIVATE lsxcommon Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_granite_switch_model)

# ── Granite-Switch-4.1-30B chat / aLoRA / tool framing (host-only) ──
add_executable(test_granite_switch_tool_framing test_granite_switch_tool_framing.cpp)
target_link_libraries(test_granite_switch_tool_framing PRIVATE
    lsxcommon Catch2::Catch2WithMain)

add_executable(test_granite_switch_tokenizer_parity test_granite_switch_tokenizer_parity.cpp)
target_compile_definitions(test_granite_switch_tokenizer_parity PRIVATE
    LSX_GRANITE_SWITCH_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/granite_switch"
    LSX_GRANITE_SWITCH_TOKENIZER_PATH="/mnt/fast_storage/huggingface/granite-switch-4.1-30b/tokenizer.json")
target_link_libraries(test_granite_switch_tokenizer_parity PRIVATE lsxcommon Catch2::Catch2WithMain)

add_executable(test_granite_switch_adapter_parity test_granite_switch_adapter_parity.cpp)
target_compile_definitions(test_granite_switch_adapter_parity PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
    LSX_ADAPTER_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/adapters")
target_include_directories(test_granite_switch_adapter_parity PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_granite_switch_adapter_parity PRIVATE lsxcommon Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_granite_switch_adapter_parity)

# The 8B sibling of the two stage-model checks that name a model explicitly.
# Both binaries are the family source compiled for the other checkpoint, because
# a hard-coded model name there would make this target verify the 30B while the
# manifest believes the 8B is covered.
add_executable(test_granite_switch_8b_adapter_parity test_granite_switch_adapter_parity.cpp)
target_compile_definitions(test_granite_switch_8b_adapter_parity PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
    LSX_ADAPTER_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/adapters"
    "LSX_GSW_MODEL=\"granite-switch-4.1-8b\""
    "LSX_GSW_ROOT_ENV=\"LSX_TEST_GRANITE_SWITCH_4_1_8B_ROOT\"")
target_include_directories(test_granite_switch_8b_adapter_parity PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_granite_switch_8b_adapter_parity PRIVATE lsxcommon Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_granite_switch_8b_adapter_parity)

add_executable(test_granite_switch_8b_tokenizer_parity test_granite_switch_tokenizer_parity.cpp)
target_compile_definitions(test_granite_switch_8b_tokenizer_parity PRIVATE
    LSX_GRANITE_SWITCH_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/granite_switch"
    "LSX_GSW_MODEL=\"granite-switch-4.1-8b\""
    "LSX_GSW_ROOT_ENV=\"LSX_TEST_GRANITE_SWITCH_4_1_8B_ROOT\"")
target_link_libraries(test_granite_switch_8b_tokenizer_parity PRIVATE lsxcommon Catch2::Catch2WithMain)

# ── phi-4 converter test (classifier + index + arrow layout) ──
add_executable(test_phi_dense_convert test_phi_dense_convert.cpp)
target_compile_definitions(test_phi_dense_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
# The residual cases decode the written NVFP4 records with the canonical
# block-scale codec (nvfp4_scale.h), like test_nvfp4_scale does.
target_include_directories(test_phi_dense_convert PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_phi_dense_convert PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_phi_dense_convert)


# ── Granite-4.2-30B loader test (pinned upload, topology, fused gate_up) ──
add_executable(test_granite_dense_loader test_granite_dense_loader.cpp)
target_compile_definitions(test_granite_dense_loader PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_granite_dense_loader PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_granite_dense_loader)


# ── phi-4 loader test (pinned upload, topology, fused qkv/gate_up) ──
add_executable(test_phi_dense_loader test_phi_dense_loader.cpp)
target_compile_definitions(test_phi_dense_loader PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_phi_dense_loader PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_phi_dense_loader)


# ── Granite-4.2-30B model test (registration + synthetic Load/Infer) ────
add_executable(test_granite_dense_model test_granite_dense_model.cpp)
target_compile_definitions(test_granite_dense_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_granite_dense_model PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_granite_dense_model)


# ── phi-4 model test (registration + synthetic Load/Infer) ────
add_executable(test_phi_dense_model test_phi_dense_model.cpp)
target_compile_definitions(test_phi_dense_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_phi_dense_model PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_phi_dense_model)


# ── Neutral dense-decoder kernel parity tests (.cu) ──

# ── Granite kernel parity tests (.cu, stand-alone OBJECT lib) ──
add_library(test_granite_cu OBJECT
    test_granite_dense_kernels.cu
)
set_target_properties(test_granite_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_granite_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(test_granite_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_granite_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_granite_cu PRIVATE Catch2::Catch2WithMain)

add_executable(test_granite_dense_kernels)
set_target_properties(test_granite_dense_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_granite_dense_kernels PRIVATE test_granite_cu lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_granite_cu PRIVATE
    LSX_KERNEL_BUDGETS_FILE="${LSX_KERNEL_BUDGETS}")
lsx_copy_arrow_dlls(test_granite_dense_kernels)

# ── Granite-Switch kernel parity tests (.cu, stand-alone OBJECT lib) ──
add_library(test_granite_switch_cu OBJECT
    test_granite_switch_kernels.cu
)
set_target_properties(test_granite_switch_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_granite_switch_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(test_granite_switch_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_granite_switch_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_granite_switch_cu PRIVATE Catch2::Catch2WithMain)
add_executable(test_granite_switch_kernels)
set_target_properties(test_granite_switch_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_granite_switch_kernels PRIVATE test_granite_switch_cu lsxcommon Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_granite_switch_kernels)

# ── phi kernel parity tests (.cu, stand-alone OBJECT lib) ──
add_library(test_phi_cu OBJECT
    test_phi_dense_kernels.cu
)
set_target_properties(test_phi_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_phi_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(test_phi_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_phi_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_phi_cu PRIVATE Catch2::Catch2WithMain)

add_executable(test_phi_dense_kernels)
set_target_properties(test_phi_dense_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_phi_dense_kernels PRIVATE test_phi_cu lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_phi_cu PRIVATE
    LSX_KERNEL_BUDGETS_FILE="${LSX_KERNEL_BUDGETS}")
lsx_copy_arrow_dlls(test_phi_dense_kernels)

# ── Dory1.0-4B converter test (classifier + index + fused/residual layout) ──
add_executable(test_dory_dense_convert test_dory_dense_convert.cpp)
target_compile_definitions(test_dory_dense_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_include_directories(test_dory_dense_convert PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_dory_dense_convert PRIVATE lsxcommon Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_dory_dense_convert)

# ── Dory1.0-4B loader test (pinned upload, edges, nGPT vector pre-scaling) ──
add_executable(test_dory_dense_loader test_dory_dense_loader.cpp)
target_compile_definitions(test_dory_dense_loader PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_dory_dense_loader PRIVATE lsxcommon Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_dory_dense_loader)

# ── Dory1.0-4B model test (registration + synthetic Load/Infer) ──
add_executable(test_dory_dense_model test_dory_dense_model.cpp)
target_compile_definitions(test_dory_dense_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_dory_dense_model PRIVATE lsxcommon Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_dory_dense_model)

# ── Dory nGPT kernel parity tests (.cu, stand-alone OBJECT lib) ──
add_library(test_dory_cu OBJECT
    test_dory_dense_kernels.cu
)
set_target_properties(test_dory_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_dory_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(test_dory_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_dory_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_dory_cu PRIVATE Catch2::Catch2WithMain)

add_executable(test_dory_dense_kernels)
set_target_properties(test_dory_dense_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_dory_dense_kernels PRIVATE test_dory_cu lsxcommon Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_dory_dense_kernels)

# ── Dory1.0-4B ChatML/tool framing (host-only, synthetic tokenizer) ──
add_executable(test_dory_tool_framing test_dory_tool_framing.cpp)
target_include_directories(test_dory_tool_framing PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_compile_definitions(test_dory_tool_framing PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_dory_tool_framing PRIVATE lsxcommon Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_dory_tool_framing ARROW_ONLY)

# ── Granite-4.2-30B SelfCheck (real weights; SKIP when absent) ──

# ── Granite-Switch-4.1-30B SelfCheck (real weights; SKIP when absent) ──

# ── phi-4 SelfCheck (real weights; SKIP when absent) ──

# ── Qwen3.5-9B loader test (pinned upload, topology, layer dispatch) ────
add_executable(test_qwen_dense_loader test_qwen_dense_loader.cpp)
target_compile_definitions(test_qwen_dense_loader PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_dense_loader PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_qwen_dense_loader)


# ── CUDA kernel tests (link lsxcommon; kernels live in lsxcommon_qwen_cu) ──


# ── Qwen CUDA kernel test executables (.cu files) ─────────────────────────
# One object library per test file, defined by lsx_add_qwen_cu_test() below
# (just before the foreach that calls it).

# ── Blackwell tensor-core probe tests: kind::mxf4nvf4 assembles ONLY on the
#    arch-'a' targets (sm_120a/sm_121a), so this OBJECT lib carries its own
#    --generate-code set (PTX ISA 8.7+, CUDA 13.3 ships 9.3).
add_library(test_blackwell_tc_cu OBJECT
    test_blackwell_tc_mma.cu
)
set_target_properties(test_blackwell_tc_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_blackwell_tc_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;
    --generate-code=arch=compute_120a,code=sm_120a;
    --generate-code=arch=compute_121a,code=sm_121a;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_blackwell_tc_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
)
target_link_libraries(test_blackwell_tc_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)

foreach(qt test_blackwell_tc_mma)
    add_executable(${qt})
    set_target_properties(${qt} PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(${qt} PRIVATE test_blackwell_tc_cu lsxcommon Catch2::Catch2WithMain)
    lsx_copy_arrow_dlls(${qt})
    add_test(NAME ${qt} COMMAND ${qt})
    set_tests_properties(${qt} PROPERTIES SKIP_RETURN_CODE 4)
endforeach()

# Shared lsx_kernels_* equivalence harness (Phase A). Generic kernels: plain
# 120/121 code, no arch-'a' PTX required.
add_library(test_lsx_kernels_cu OBJECT
    test_lsx_kernels.cu
)
set_target_properties(test_lsx_kernels_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_lsx_kernels_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(test_lsx_kernels_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_lsx_kernels_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_lsx_kernels_cu PRIVATE Catch2::Catch2WithMain)

foreach(qt test_lsx_kernels)
    add_executable(${qt})
    set_target_properties(${qt} PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(${qt} PRIVATE test_lsx_kernels_cu lsxcommon Catch2::Catch2WithMain)
    lsx_copy_arrow_dlls(${qt})
    add_test(NAME ${qt} COMMAND ${qt})
endforeach()

add_library(test_cuda_toolkit_guard_cu OBJECT
    test_cuda_toolkit_guard.cu
)
set_target_properties(test_cuda_toolkit_guard_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_cuda_toolkit_guard_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
# The stamp Task 1 writes at configure time. The case that reads it proves the
# OBJECTS were built with the headers this TU sees — the one thing configure
# cannot check after the fact.
target_compile_definitions(test_cuda_toolkit_guard_cu PRIVATE
    LSX_CUDA_TOOLKIT_STAMP="${CMAKE_BINARY_DIR}/lsx_cuda_toolkit_stamp.txt")
target_compile_options(test_cuda_toolkit_guard_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_link_libraries(test_cuda_toolkit_guard_cu PRIVATE
    Catch2::Catch2WithMain CUDA::cudart)

foreach(qt test_cuda_toolkit_guard)
    add_executable(${qt})
    set_target_properties(${qt} PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(${qt} PRIVATE test_cuda_toolkit_guard_cu Catch2::Catch2WithMain)
    lsx_copy_arrow_dlls(${qt})
    add_test(NAME ${qt} COMMAND ${qt})
    set_tests_properties(${qt} PROPERTIES SKIP_RETURN_CODE 4)
endforeach()

# `fast`, not `inference`: this is the ONLY test that can observe a build tree
# whose objects came from a different toolkit (a stale `CUDART_VERSION` in one TU
# against the linked runtime), so it must run in the default suite. It allocates
# no device memory — it reads versions and device properties, and its device case
# SKIPs (4) where no GPU exists. Spelled literally: the label guard resolves
# static names only, not the loop variable.
set_tests_properties(test_cuda_toolkit_guard PROPERTIES LABELS fast)

# One OBJECT lib + executable per .cu test file. The previous shared
# `test_qwen_cu` object library linked ALL 14 files into EVERY executable:
# the binaries were byte-identical, each ran the full ~187k assertions, and a
# single failing assertion marked all 14 ctest targets failed (docs/open-tasks/2026-09-30-..., item E1).
# Per-file objects keep the target names/labels but run only their own cases.
#
# The shared helper lives in cmake/LsxTestHelpers.cmake; each entry keeps its
# own per-file OBJECT library (the target name/label is unchanged) so one
# failing case no longer marks the whole group failed.
foreach(qt
    test_lsx_nvfp4_dispatch
    test_lsx_gemv
    test_perf_knobs_apply
    test_blackwell_micro_gemm)
    lsx_add_cuda_test(NAME ${qt})
endforeach()

# ── Qwen3.5-9B model test (factory + IModel + generation) ──────────────
add_executable(test_qwen_dense_model test_qwen_dense_model.cpp)
target_compile_definitions(test_qwen_dense_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_dense_model PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_qwen_dense_model)

# Synthetic tiny-geometry fixture: opt out of the frozen-geometry hard gate.

# ── GLM expert device-cache: host-side LRU bookkeeping (no CUDA) ────────
add_executable(test_glm_moe_expert_cache test_glm_moe_expert_cache.cpp)
target_link_libraries(test_glm_moe_expert_cache PRIVATE lsxcommon Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_glm_moe_expert_cache ARROW_ONLY)

# ── Qwen incremental KV cache (Task 7) ─────────────────────────────────
add_executable(test_qwen_dense_kv_cache test_qwen_dense_kv_cache.cpp)
target_compile_definitions(test_qwen_dense_kv_cache PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_dense_kv_cache PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_qwen_dense_kv_cache)


# ── Bit-exactness gate: incremental KV vs full recompute (synthetic) ────────
add_executable(test_qwen_dense_kv_parity test_qwen_dense_kv_parity.cpp)
target_compile_definitions(test_qwen_dense_kv_parity PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_dense_kv_parity PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_qwen_dense_kv_parity)


# ── Qwen3.8-27B model test ──────────────────────────────────────────────
add_executable(test_qwen_dense_27b_model test_qwen_dense_27b_model.cpp)
target_compile_definitions(test_qwen_dense_27b_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_dense_27b_model PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_qwen_dense_27b_model)

# Synthetic tiny-geometry fixture: opt out of the frozen-geometry hard gate.

# ── Shared integrity-check model test (integrity:: helpers) ─────────
add_executable(test_model_integrity test_model_integrity.cpp)
target_compile_definitions(test_model_integrity PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_model_integrity PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_model_integrity)

add_test(NAME test_model_integrity COMMAND test_model_integrity)

# ── Qwen converter integrity integration test ───────────────────────
add_executable(test_qwen_dense_converter_integrity test_qwen_dense_converter_integrity.cpp)
target_compile_definitions(test_qwen_dense_converter_integrity PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_dense_converter_integrity PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_qwen_dense_converter_integrity)


# ── GLM-5.2 converter test (tensor name parser) ────────────────────────
add_executable(test_glm_moe_convert test_glm_moe_convert.cpp)
target_compile_definitions(test_glm_moe_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_moe_convert PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_glm_moe_convert)


# ── GLM-5.3-Flash converter test (geometry, classifier, requant) ─────
add_executable(test_glm_moe_flash_convert test_glm_moe_flash_convert.cpp)
target_compile_definitions(test_glm_moe_flash_convert PRIVATE
    LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_glm_moe_flash_convert PRIVATE
    lsxcommon simdjson::simdjson Catch2::Catch2WithMain)
target_include_directories(test_glm_moe_flash_convert PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)

lsx_copy_arrow_dlls(test_glm_moe_flash_convert ARROW_ONLY)


# ── Xing4.0-29B-A4B converter test (mHC / MLA / router classification) ──
add_executable(test_xing_moe_convert test_xing_moe_convert.cpp)
target_include_directories(test_xing_moe_convert PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_link_libraries(test_xing_moe_convert PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Mistral-Small-4-119B converter test (MLA / 128-expert MoE classifier) ──
add_executable(test_mistral_moe_convert test_mistral_moe_convert.cpp)
target_include_directories(test_mistral_moe_convert PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_link_libraries(test_mistral_moe_convert PRIVATE lsxcommon
    Catch2::Catch2WithMain)


# ── Mistral-Small-4 parity-oracle tool guard (host-only source scan) ─────
# Pins that tools/mistral_small4_oracle.py refuses a partially-random model and
# keeps the two load-bearing rope keys. Fast: no torch/transformers needed.

# ── Mistral-Large-4 tokenizer parity (source-captured golden) ─────────────
add_executable(test_mistral_moe_large4_tokenizer_parity
    test_mistral_moe_large4_tokenizer_parity.cpp)
target_compile_definitions(test_mistral_moe_large4_tokenizer_parity PRIVATE
    "LSX_MISTRAL_LARGE4_FIXTURE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/mistral_large4\"")
target_link_libraries(test_mistral_moe_large4_tokenizer_parity PRIVATE lsxcommon
    Catch2::Catch2WithMain arrow_shared)

# ── Mistral-Large-4 family kernels (LeRoPE / ssmax / router / indexer) ────
add_executable(test_mistral_moe_large4_kernels test_mistral_moe_large4_kernels.cu)
target_include_directories(test_mistral_moe_large4_kernels PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
set_target_properties(test_mistral_moe_large4_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_mistral_moe_large4_kernels PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Mistral-Large-4 model (synthetic converted tree) ──────────────────────
add_executable(test_mistral_moe_large4_model test_mistral_moe_large4_model.cpp)
target_include_directories(test_mistral_moe_large4_model PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_link_libraries(test_mistral_moe_large4_model PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Mistral-Large-4 loader ────────────────────────────────────────────────
add_executable(test_mistral_moe_large4_loader test_mistral_moe_large4_loader.cpp)
target_include_directories(test_mistral_moe_large4_loader PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_link_libraries(test_mistral_moe_large4_loader PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Mistral-Large-4 selfcheck + FSI/CODE/DET gates (SKIP 4 without vault) ──


# ── Mistral-Large-4 tool framing (host-only, step 4b) ─────────────────────
add_executable(test_mistral_moe_large4_tool_framing
    test_mistral_moe_large4_tool_framing.cpp)
target_compile_definitions(test_mistral_moe_large4_tool_framing PRIVATE
    "LSX_LARGE4_TOK_FIXTURE=\"${CMAKE_CURRENT_SOURCE_DIR}/data/synthetic_mistral_large4_tok.json\"")
target_link_libraries(test_mistral_moe_large4_tool_framing PRIVATE lsxcommon
    Catch2::Catch2WithMain arrow_shared)

# ── Mistral-Large-4 pre-tokenizer (kind 12) ───────────────────────────────
add_executable(test_mistral_moe_large4_pretokenizer test_mistral_moe_large4_pretokenizer.cpp)
target_link_libraries(test_mistral_moe_large4_pretokenizer PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Mistral-Large-4 converter test (MLA + DSA indexer + KV sharing) ───────
add_executable(test_mistral_moe_large4_convert test_mistral_moe_large4_convert.cpp)
target_include_directories(test_mistral_moe_large4_convert PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_link_libraries(test_mistral_moe_large4_convert PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── MiniMax-M3 converter test (MSA indexer / dense-prefix MoE classifier) ──
add_executable(test_minimax_moe_convert test_minimax_moe_convert.cpp)
target_include_directories(test_minimax_moe_convert PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_compile_definitions(test_minimax_moe_convert PRIVATE
    LSX_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(test_minimax_moe_convert PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── MiniMax-M3.1-Flash converter test (MXFP8 dense + NVFP4 experts) ──
add_executable(test_minimax_moe_flash_convert test_minimax_moe_flash_convert.cpp)
target_include_directories(test_minimax_moe_flash_convert PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_compile_definitions(test_minimax_moe_flash_convert PRIVATE
    LSX_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(test_minimax_moe_flash_convert PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── MiniMax-M3.1-Flash loader stage (all-layer MSA indexer + dispatch) ──
add_executable(test_minimax_moe_flash_loader test_minimax_moe_flash_loader.cpp)
target_include_directories(test_minimax_moe_flash_loader PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_compile_definitions(test_minimax_moe_flash_loader PRIVATE
    LSX_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(test_minimax_moe_flash_loader PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── MiniMax-M3 loader stage (pinned graph + neutral expert store) ──
add_executable(test_minimax_moe_loader test_minimax_moe_loader.cpp)
target_include_directories(test_minimax_moe_loader PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_compile_definitions(test_minimax_moe_loader PRIVATE
    LSX_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(test_minimax_moe_loader PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── MiniMax-M3 model stage (lifecycle + determinism) ──
add_executable(test_minimax_moe_model test_minimax_moe_model.cpp)
target_include_directories(test_minimax_moe_model PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_compile_definitions(test_minimax_moe_model PRIVATE
    LSX_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(test_minimax_moe_model PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── MiniMax-M3.1-Flash model stage (lifecycle + determinism) ──
add_executable(test_minimax_moe_flash_model test_minimax_moe_flash_model.cpp)
target_include_directories(test_minimax_moe_flash_model PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_compile_definitions(test_minimax_moe_flash_model PRIVATE
    LSX_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(test_minimax_moe_flash_model PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── MiniMax-M3 native tool framing (host-only) ──
add_executable(test_minimax_moe_tool_framing test_minimax_moe_tool_framing.cpp)
target_compile_definitions(test_minimax_moe_tool_framing PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_minimax_moe_tool_framing PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── MiniMax-M3.1-Flash native tool framing (host-only) ──
add_executable(test_minimax_moe_flash_tool_framing
    test_minimax_moe_flash_tool_framing.cpp)
target_include_directories(test_minimax_moe_flash_tool_framing PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_compile_definitions(test_minimax_moe_flash_tool_framing PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_minimax_moe_flash_tool_framing PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Mistral-Small-4-119B loader test (dense BF16 graph upload) ──
add_executable(test_mistral_moe_loader test_mistral_moe_loader.cpp)
target_include_directories(test_mistral_moe_loader PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_link_libraries(test_mistral_moe_loader PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Mistral-Small-4-119B native tool framing (host-only) ──
add_executable(test_mistral_moe_tool_framing test_mistral_moe_tool_framing.cpp)
target_link_libraries(test_mistral_moe_tool_framing PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Mistral-Small-4-119B model stage (decode pipeline) ──
add_executable(test_mistral_moe_model test_mistral_moe_model.cpp)
target_include_directories(test_mistral_moe_model PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_link_libraries(test_mistral_moe_model PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Mistral-Small-4-119B kernel-level test (MLA/RoPE/attention) ──
add_executable(test_mistral_moe_kernels test_mistral_moe_kernels.cpp)
target_include_directories(test_mistral_moe_kernels PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_link_libraries(test_mistral_moe_kernels PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── MiniMax-M3.1-Flash selfcheck + FSI/CODE/DET gates (SKIP 4 w/o vault) ──


# ── MiniMax-M3 selfcheck + FSI/CODE/DET golden parity gates (SKIP 4 w/o vault) ──

add_executable(test_minimax_moe_tokenizer_parity
    test_minimax_moe_tokenizer_parity.cpp)
target_compile_definitions(test_minimax_moe_tokenizer_parity PRIVATE
    "LSX_MINIMAX_FIXTURE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/minimax\"")
target_link_libraries(test_minimax_moe_tokenizer_parity PRIVATE lsxcommon
    Catch2::Catch2WithMain arrow_shared)


# ── Mistral-Small-4-119B selfcheck (real weights; SKIP 4 without vault) ──

# ── Mistral-Small-4-119B FSI/CODE/DET golden gates (SKIP 4 without vault) ──


# ── Xing4.0-29B-A4B selfcheck + FSI/CODE/DET golden parity gates ──

# Bit-exact parity across KV layouts. Registered here rather than folded into
# test_xing_moe_selfcheck so a parity failure names the layout as the suspect.
add_executable(test_xing_kv_layout_parity test_xing_kv_layout_parity.cpp)
target_include_directories(test_xing_kv_layout_parity PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_link_libraries(test_xing_kv_layout_parity PRIVATE lsxcommon Catch2::Catch2WithMain)
# The fixture is a synthetic converted checkpoint: no 16 GiB resident model, and it
# needs the synthetic-geometry escape hatch (a RUNTIME env var, per
# synthetic_geometry.h - not a compile definition).
# Warm-up ramp measurement. Reports the SHAPE of per-token throughput over a
# session; asserts nothing about its value, because a warm-up shape is a property
# of the box (clock policy, driver, power state) and a budget taken from one box
# would flake on another.
add_executable(test_xing_warmup_ramp test_xing_warmup_ramp.cpp)
target_link_libraries(test_xing_warmup_ramp PRIVATE lsxcommon Catch2::Catch2WithMain)


# Written out rather than in a foreach(): test_model_suite_guard's
# RegisteredTestNames() resolves add_test(NAME <literal>) and foreach argument
# lists, but cannot expand a variable inside a target name.


# ── Xing4.0-29B-A4B tool-calling codec (host-only, synthetic tokenizer) ──
add_executable(test_xing_tool_framing test_xing_tool_framing.cpp)
target_compile_definitions(test_xing_tool_framing PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_xing_tool_framing PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Xing4.0-29B-A4B model test (decode step / determinism / KV growth) ──
add_executable(test_xing_moe_model test_xing_moe_model.cpp)
target_include_directories(test_xing_moe_model PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_compile_definitions(test_xing_moe_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_xing_moe_model PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Xing4.0-29B-A4B loader test (mHC / MLA / all-resident experts) ──
add_executable(test_xing_moe_loader test_xing_moe_loader.cpp)
target_include_directories(test_xing_moe_loader PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_link_libraries(test_xing_moe_loader PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Kimi-K2.7-Code converter test (geometry, classifier, topology) ──
add_executable(test_kimi_moe_convert test_kimi_moe_convert.cpp)
target_compile_definitions(test_kimi_moe_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_kimi_moe_convert PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_kimi_moe_convert)


# ── GLM-5.2 loader test (pinned BF16→FP16 upload) ─────────────────────
add_executable(test_glm_moe_loader test_glm_moe_loader.cpp)
target_compile_definitions(test_glm_moe_loader PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_moe_loader PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_glm_moe_loader)


# ── GLM-5.2 model test (factory + IModel) ─────────────────────────────────
add_executable(test_glm_moe_model test_glm_moe_model.cpp)
target_compile_definitions(test_glm_moe_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_moe_model PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_glm_moe_model)


# ── GLM-5.2 SelfCheck test (protocol step 8): incremental 1->2-token
#    selfcheck on the real model, at least 3 stress cycles (3 runs are enough
#    to apply the load; overridable with LSX_TEST_GLM_SELFCHECK_CYCLES),
#    bit-exact determinism of final_logits, exit 0. SKIP without a checkpoint.


# ── Blackwell audit Phase 0: GLM-5.2 perf baseline harness (reporting test,
#    SKIP without a checkpoint). The number of generated tokens — LSX_TEST_PERF_GEN_TOKENS.
add_executable(test_blackwell_perf_glm test_blackwell_perf_glm.cpp)
target_link_libraries(test_blackwell_perf_glm PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_blackwell_perf_glm)

add_test(NAME test_blackwell_perf_glm COMMAND test_blackwell_perf_glm)
set_tests_properties(test_blackwell_perf_glm PROPERTIES SKIP_RETURN_CODE 4)
set_tests_properties(test_blackwell_perf_glm PROPERTIES RESOURCE_LOCK "gpu")

# ── Blackwell audit Phase 0: dense perf baseline harness (Qwen3.5-9B).
add_executable(test_blackwell_perf_dense test_blackwell_perf_dense.cpp)
target_link_libraries(test_blackwell_perf_dense PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_blackwell_perf_dense)

add_test(NAME test_blackwell_perf_dense COMMAND test_blackwell_perf_dense)
set_tests_properties(test_blackwell_perf_dense PROPERTIES SKIP_RETURN_CODE 4)
set_tests_properties(test_blackwell_perf_dense PROPERTIES RESOURCE_LOCK "gpu")

# ── MoE perf baseline harnesses: Qwen3.5-35B-A3B, gemma-4-26B-A4B-it.
add_executable(test_blackwell_perf_moe test_blackwell_perf_moe.cpp)
target_link_libraries(test_blackwell_perf_moe PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_blackwell_perf_moe)

# One binary, two ctest entries (Catch2 tag filters) so each MoE model is
# individually selectable and named in ctest output.
add_test(NAME test_blackwell_perf_moe_qwen
         COMMAND test_blackwell_perf_moe "[qwen35moe]")
add_test(NAME test_blackwell_perf_moe_gemma
         COMMAND test_blackwell_perf_moe "[gemma26moe]")
set_tests_properties(test_blackwell_perf_moe_qwen PROPERTIES SKIP_RETURN_CODE 4)
set_tests_properties(test_blackwell_perf_moe_qwen PROPERTIES RESOURCE_LOCK "gpu")
set_tests_properties(test_blackwell_perf_moe_gemma PROPERTIES SKIP_RETURN_CODE 4)
set_tests_properties(test_blackwell_perf_moe_gemma PROPERTIES RESOURCE_LOCK "gpu")

# ── Blackwell audit Phase 0: DeepSeek-V4-Flash perf baseline harness.
add_executable(test_blackwell_perf_deepseek test_blackwell_perf_deepseek.cpp)
target_link_libraries(test_blackwell_perf_deepseek PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_blackwell_perf_deepseek)

add_test(NAME test_blackwell_perf_deepseek COMMAND test_blackwell_perf_deepseek)
set_tests_properties(test_blackwell_perf_deepseek PROPERTIES SKIP_RETURN_CODE 4)
set_tests_properties(test_blackwell_perf_deepseek PROPERTIES RESOURCE_LOCK "gpu")

# ── GLM-5.2 seq-P prefill parity test: MoE layers 3..77 must keep
#    block_rows = P (the full prompt length), each prompt token occupies one
#    IndexShare KV slot. FAIL on row collapse (kv_slots_used != P + gen),
#    on a mismatch between the argmax of final_logits and the first generated
#    token, or on loss of bit-exactness between two identical runs. SKIP
#    without a checkpoint.
add_executable(test_glm_moe_prefill_seqp test_glm_moe_prefill_seqp.cpp)
target_compile_definitions(test_glm_moe_prefill_seqp PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_moe_prefill_seqp PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_glm_moe_prefill_seqp)


# ── GLM-5.2 meaningful-output test via --run (Infer output_text) ───────────
add_executable(test_glm_moe_prompt_output test_glm_moe_prompt_output.cpp)
target_compile_definitions(test_glm_moe_prompt_output PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_moe_prompt_output PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_glm_moe_prompt_output)


# ── DeepSeek-V4-Flash meaningful-output test (Infer output_text) ────────
add_executable(test_deepseek_moe_prompt_output test_deepseek_moe_prompt_output.cpp)
target_link_libraries(test_deepseek_moe_prompt_output PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_deepseek_moe_prompt_output)


# ── DeepSeek-V4-Flash SelfCheck test (protocol step 8): incremental
#    1->2-token selfcheck on the real model, at least 3 cycles, deterministic
#    final_logits. SKIP without a checkpoint.


# ── Qwen3.5-9B SelfCheck test (protocol step 8): incremental 1→2-token
#    selfcheck on the real model, at least 3 cycles, deterministic final_logits.
#    SKIP without a checkpoint.


# ── Qwen3.8-27B SelfCheck (real model) ─────────────────────────────
#    Selfcheck on the real model, at least 3 cycles, deterministic final_logits.
#    SKIP without a checkpoint.


# ── Model factory test ────────────────────────────────────────────────
add_executable(test_model_factory test_model_factory.cpp)
target_link_libraries(test_model_factory PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_model_factory)

add_test(NAME test_model_factory COMMAND test_model_factory)
set_tests_properties(test_model_factory PROPERTIES SKIP_RETURN_CODE 4)

# ── Generation-path CLI test (--run / -r normalization + default prompt) ─────
add_executable(test_generation_cli test_generation_cli.cpp)
target_link_libraries(test_generation_cli PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_generation_cli)

add_test(NAME test_generation_cli COMMAND test_generation_cli)

# ── GLM-5.2 prompt framing helper test (host-only, synthetic tokenizer) ────
add_executable(test_glm_moe_prompt_framing test_glm_moe_prompt_framing.cpp)
target_compile_definitions(test_glm_moe_prompt_framing PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_moe_prompt_framing PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_glm_moe_prompt_framing)


# ── Gemma-4 prompt framing helper test (host-only, tokenizer.json sentinels) ─────
add_executable(test_gemma_prompt_framing test_gemma_prompt_framing.cpp)
target_compile_definitions(test_gemma_prompt_framing PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_gemma_prompt_framing PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_gemma_prompt_framing ARROW_ONLY)


# ── Qwen ChatML + native tool framing test (host-only, synthetic tokenizer) ──
add_executable(test_qwen_tool_framing test_qwen_tool_framing.cpp)
target_include_directories(test_qwen_tool_framing PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_compile_definitions(test_qwen_tool_framing PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_tool_framing PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_qwen_tool_framing ARROW_ONLY)


# ── Kimi-K3 kernels (CUDA, host references) ────────────────────────────────
add_executable(test_kimi_k3_kernels test_kimi_k3_kernels.cu)
target_include_directories(test_kimi_k3_kernels PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
set_target_properties(test_kimi_k3_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_kimi_k3_kernels PRIVATE lsxcommon Catch2::Catch2WithMain)

# ── Kimi-K3 loader (synthetic converted tree) ──────────────────────────────
add_executable(test_kimi_k3_loader test_kimi_k3_loader.cpp)
target_include_directories(test_kimi_k3_loader PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_kimi_k3_loader PRIVATE lsxcommon Catch2::Catch2WithMain)

# ── Kimi-K3.1-Pro loader: DSA topology + pinned indexer plane ──────────────
add_executable(test_kimi_pro_loader test_kimi_pro_loader.cpp)
target_include_directories(test_kimi_pro_loader PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_kimi_pro_loader PRIVATE lsxcommon Catch2::Catch2WithMain)

# ── Kimi-K3.1-Pro Lightning-Indexer kernel parity (CUDA OBJECT lib + exe) ──
add_library(test_kimi_pro_indexer_cu OBJECT test_kimi_pro_indexer.cu)
set_target_properties(test_kimi_pro_indexer_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_kimi_pro_indexer_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(test_kimi_pro_indexer_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_include_directories(test_kimi_pro_indexer_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_kimi_pro_indexer_cu PRIVATE Catch2::Catch2WithMain
    lsx_dep_includes)

add_executable(test_kimi_pro_indexer)
set_target_properties(test_kimi_pro_indexer PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_kimi_pro_indexer PRIVATE test_kimi_pro_indexer_cu
    lsxcommon Catch2::Catch2WithMain)

# ── Kimi-K3.1-Pro pipeline / model (synthetic converted tree) ───────────────
add_executable(test_kimi_pro_model test_kimi_pro_model.cpp)
target_include_directories(test_kimi_pro_model PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_kimi_pro_model PRIVATE lsxcommon Catch2::Catch2WithMain)

# ── Kimi-K3 model + lifecycle (synthetic converted tree) ───────────────────
add_executable(test_kimi_k3_model test_kimi_k3_model.cpp)
target_include_directories(test_kimi_k3_model PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_kimi_k3_model PRIVATE lsxcommon Catch2::Catch2WithMain)

# ── Kimi-K3 FSI / CODE / DET parity gates (template-derived goldens) ───────


# ── Kimi-K3.1-Pro FSI / CODE / DET parity gates (template-derived goldens) ──
# The budgets are MEASURED, not guessed: this variant decodes at ~0.125 t/s on
# the reference host (92 MoE layers x top-16 experts streamed from a 1.4 TB
# host arrow set; the pinned graph alone is 23 GB and takes ~10 min to load).
# One 2048-token generation is therefore ~4.5 h, which is why the three gates
# below are seeded but not yet measured — `docs/OPEN_TASKS.md` §R17 records the
# debt. A budget that cannot fit a real completion would turn the entry into a
# guaranteed timeout-kill, i.e. a failure that is not a defect.
# 3 prompts x 2048 tokens ~= 13.7 h + load = ~14.2 h.

# 3 prompts x 512 tokens ~= 3.4 h + load = ~3.6 h.

# 3 prompts x 3 stress runs x 512 tokens ~= 10.2 h + load = ~10.4 h.

# ── Kimi-K3 SelfCheck (real weights; SKIP without a model) ─────────────────

# ── Kimi-K3.1-Pro SelfCheck (real weights; SKIP without a model) ───────────
# Two cases: Paris x3 cycles and the real-weight XTML tool round-trip. The
# round-trip generates up to 256 tokens through the thinking channel, so the
# budget covers both on a host-store-bound 2.8T model.

# ── Kimi-K3 XTML chat/tool codec (host-only, synthetic tokenizer) ───────────
add_executable(test_kimi_k3_tool_framing test_kimi_k3_tool_framing.cpp)
target_include_directories(test_kimi_k3_tool_framing PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_compile_definitions(test_kimi_k3_tool_framing PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_kimi_k3_tool_framing PRIVATE lsxcommon Catch2::Catch2WithMain)

# ── Kimi-K3 tiktoken id parity (golden from the checkpoint tokenizer) ───────
add_executable(test_kimi_k3_tokenizer_parity test_kimi_k3_tokenizer_parity.cpp)
target_compile_definitions(test_kimi_k3_tokenizer_parity PRIVATE
    LSX_KIMI_K3_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/kimi_k3")
target_link_libraries(test_kimi_k3_tokenizer_parity PRIVATE lsxcommon Catch2::Catch2WithMain)

# ── Kimi-K3 frozen passport (host-only, no CUDA) ─────────────────────────────
# The Kimi-K3.1-Pro passport (kimi_moe_pro_constants.h) is pinned by the same
# constants gate: the sibling variant's geometry is a strict restatement of the
# K3 tower plus the DSA/indexer block, so one family constants target owns both
# passports (the Suites() rows for kimi-k3 and kimi-k3.1-pro both name it).
add_executable(test_kimi_k3_constants test_kimi_k3_constants.cpp
                                      test_kimi_pro_constants.cpp)
target_link_libraries(test_kimi_k3_constants PRIVATE lsxcommon Catch2::Catch2WithMain)

# ── phi-4 ChatML token bridge test (host-only, synthetic tokenizer) ──
add_executable(test_phi_tool_framing test_phi_tool_framing.cpp)
target_include_directories(test_phi_tool_framing PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_compile_definitions(test_phi_tool_framing PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_phi_tool_framing PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_phi_tool_framing ARROW_ONLY)


# ── Gemma-4 native tool codec test (host-only; render/parse, no tokenizer) ──
add_executable(test_gemma_tool_framing test_gemma_tool_framing.cpp)
target_link_libraries(test_gemma_tool_framing PRIVATE lsxcommon Catch2::Catch2WithMain)

# ── GigaChat-3.5 chatml-v5 GCML tool codec test (host-only) ──
add_executable(test_gigachat_tool_framing test_gigachat_tool_framing.cpp)
target_link_libraries(test_gigachat_tool_framing PRIVATE lsxcommon Catch2::Catch2WithMain)

# ── JSON escape contract + drift guard over the family-local copies ──
add_executable(test_json_escape test_json_escape.cpp)
target_include_directories(test_json_escape PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_compile_definitions(test_json_escape PRIVATE
    LSX_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(test_json_escape PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_json_escape COMMAND test_json_escape)

# ── GLM-5.3-Flash native tool framing test (host-only, synthetic tokenizer) ──
add_executable(test_glm_flash_tool_framing test_glm_flash_tool_framing.cpp)
target_include_directories(test_glm_flash_tool_framing PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_compile_definitions(test_glm_flash_tool_framing PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_flash_tool_framing PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_glm_flash_tool_framing ARROW_ONLY)


# ── DeepSeek-V4 / V4.1 native DSML tool framing test (host-only) ─────────────
add_executable(test_deepseek_tool_framing test_deepseek_tool_framing.cpp)
target_include_directories(test_deepseek_tool_framing PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_compile_definitions(test_deepseek_tool_framing PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_deepseek_tool_framing PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_deepseek_tool_framing ARROW_ONLY)


# ── Gemma-4 converter test (name classifier + NVFP4 policy) ──────────
add_executable(test_gemma_moe_convert test_gemma_moe_convert.cpp)
target_compile_definitions(test_gemma_moe_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
# The residual cases decode the written NVFP4 records with the canonical
# block-scale codec (nvfp4_scale.h), like test_phi_dense_convert does.
target_include_directories(test_gemma_moe_convert PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_gemma_moe_convert PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_gemma_moe_convert)


# ── Gemma-4-31B-it (dense) kernel tests ───────────────────────────────────
add_executable(test_gemma_dense_kernels test_gemma_dense_kernels.cpp)
target_link_libraries(test_gemma_dense_kernels PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_gemma_dense_kernels)


# ── Gemma-4-31B-it (dense) loader test ────────────────────────────────────
add_executable(test_gemma_dense_loader test_gemma_dense_loader.cpp)
target_compile_definitions(test_gemma_dense_loader PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_gemma_dense_loader PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_gemma_dense_loader)


# ── Gemma-4-31B-it (dense) model test ─────────────────────────────────────
add_executable(test_gemma_dense_model test_gemma_dense_model.cpp)
target_include_directories(test_gemma_dense_model PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_definitions(test_gemma_dense_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_gemma_dense_model PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_gemma_dense_model)


# ── Gemma-4-31B-it (dense) selfcheck test ─────────────────────────────────


# ── Gemma-4-31B-it (dense) fsi selfcheck test ──────────────────────────


# ── Gemma-4-31B-it (dense) code selfcheck test ──────────────────────────


# ── Gemma-4-31B-it (dense) det selfcheck test ──────────────────────────


# ── Gemma-4-31B-it (dense) converter test ─────────────────────────────────
add_executable(test_gemma_dense_convert test_gemma_dense_convert.cpp)
target_compile_definitions(test_gemma_dense_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_gemma_dense_convert PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_gemma_dense_convert)


# ── Gemma-4 loader test (BF16→FP16 in-place, arena, expert LRU) ─────────
add_executable(test_gemma_moe_loader test_gemma_moe_loader.cpp)
target_compile_definitions(test_gemma_moe_loader PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_gemma_moe_loader PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_gemma_moe_loader)


# ── Gemma-4 model test (factory + Load + Infer, synthetic geometry) ────
add_executable(test_gemma_moe_model test_gemma_moe_model.cpp)
target_include_directories(test_gemma_moe_model PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_compile_definitions(test_gemma_moe_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_gemma_moe_model PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_gemma_moe_model)


# ── Gemma-4 SelfCheck test (protocol step 8): incremental 1→2-token
#    selfcheck on the real model, at least 3 stress cycles (overridable with
#    LSX_TEST_GEMMA_SELFCHECK_CYCLES), bit-exact determinism of
#    final_logits, exit 0. SKIP without a checkpoint.


# ── Gemma-4 kernel reference tests (norm / rotary) ──────────────────────────


# ── ModelBase test (tokenizer-db load + infer guards, host-only) ─────────
add_executable(test_model_runtime test_model_runtime.cpp)
target_compile_definitions(test_model_runtime PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_include_directories(test_model_runtime PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_link_libraries(test_model_runtime PRIVATE lsxcommon Catch2::Catch2WithMain)

# ── InferenceEngine singleton + TokenChunk streaming/abort test (host-only) ──
add_executable(test_model_engine test_model_engine.cpp)
target_link_libraries(test_model_engine PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_model_engine)

add_test(NAME test_model_engine COMMAND test_model_engine)

lsx_copy_arrow_dlls(test_model_runtime)

add_test(NAME test_model_runtime COMMAND test_model_runtime)

# ── Perf baseline guard (real weights, label perf — manual command) ──────
file(TO_CMAKE_PATH "${PROJECT_SOURCE_DIR}" LSX_SRC_DIR_FWD_PERF)
add_executable(test_perf_baseline test_perf_baseline.cpp)
target_compile_definitions(test_perf_baseline PRIVATE
    LSX_SOURCE_DIR="${LSX_SRC_DIR_FWD_PERF}")
target_link_libraries(test_perf_baseline PRIVATE lsxcommon
                      Catch2::Catch2WithMain)


# GigaChat3.5 is expert-streamed on a single RTX 5090 (~3.3 t/s at 2048
# tokens), and the baseline test measures both numeric modes per invocation.


# Host-store bound (~1 t/s): two 2048-token modes plus the probe do not fit the
# 3000 s budget the dense families use.

# Host-store bound (~0.3 t/s: 27 GiB routed + 9.5 GiB streamed per token): the
# two 2048-token modes plus the probe need a long budget.


# ── E8M0 -> pow2-E4M3 scale-pair test (DeepSeek NVFP4 repack, Task 1) ─────
add_executable(test_deepseek_nvfp4_repack test_deepseek_nvfp4_repack.cpp)
target_include_directories(test_deepseek_nvfp4_repack PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_deepseek_nvfp4_repack PRIVATE lsxcommon
                      Catch2::Catch2WithMain)
# Scale-primitive contract (fast) and the repack-tool e2e (convert, [tool]
# tag filter — one binary, two ctest entries per the label discipline).

# ── DeepSeek converter: synthetic e2e (logestix-v1-nvfp4) ─────────────
add_executable(test_deepseek_moe_convert test_deepseek_moe_convert.cpp)
target_compile_definitions(test_deepseek_moe_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_deepseek_moe_convert PRIVATE lsxcommon
                      Catch2::Catch2WithMain)

# ── Zero-alloc tokenizer test ─────────────────────────────────────────
add_executable(test_tokenizer test_tokenizer.cpp)
target_compile_definitions(test_tokenizer PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_tokenizer PRIVATE lsxcommon Catch2::Catch2WithMain)

# Device-side agreement for the KV geometry. Its own OBJECT library with explicit
# include dirs, NOT lsxcommon at compile time: lsx_kv_cache.h is header-only and
# constexpr, so nothing from the library is needed to compile, and linking
# lsxcommon here would drag its private "-Xcompiler=/utf-8" CUDA option into this
# target's own compile step (the duplicated-bare-/utf-8 nvcc misparse documented
# above test_deepseek_moe_inference_kernels).
#
# It deliberately does NOT pass --expt-relaxed-constexpr: the header's helpers
# are __host__ __device__, so the device pass must compile them on its own. If
# someone drops the explicit device annotation and leans on the flag, this target
# stops compiling - that is the guard.
add_library(test_kv_cache_layout_cu OBJECT test_kv_cache_layout_device.cu)
set_target_properties(test_kv_cache_layout_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_kv_cache_layout_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_kv_cache_layout_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
)
target_link_libraries(test_kv_cache_layout_cu PRIVATE Catch2::Catch2WithMain)

add_executable(test_kv_cache_layout test_kv_cache_layout.cpp $<TARGET_OBJECTS:test_kv_cache_layout_cu>)
set_target_properties(test_kv_cache_layout PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_kv_cache_layout PRIVATE lsxcommon Catch2::Catch2WithMain)

target_link_libraries(test_kv_cache_layout PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_kv_cache_layout COMMAND test_kv_cache_layout)
set_tests_properties(test_kv_cache_layout PROPERTIES LABELS fast)

lsx_copy_arrow_dlls(test_tokenizer)

add_test(NAME test_tokenizer COMMAND test_tokenizer)

# ── AliceAI SentencePiece-BPE tokenizer backend (model-neutral) ────────────
add_executable(test_alice_tokenizer test_alice_tokenizer.cpp)
target_compile_definitions(test_alice_tokenizer PRIVATE
    LSX_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(test_alice_tokenizer PRIVATE lsxcommon
    simdjson::simdjson Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_alice_tokenizer)

# ── GLM-5.2 Dense MLP forward test (layers 0-2) ─────────────────────────────
add_executable(test_glm_moe_layer_compute test_glm_moe_layer_compute.cpp)
target_link_libraries(test_glm_moe_layer_compute PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_glm_moe_layer_compute)


# ── GLM-5.2 Dense MLP forward test (layers 0-2, Phase 4.5 hash fixture) ───────
# Runs the specific C++/CUDA execution loop RunGlmDenseMLPStack (SwiGLU
# w2(silu(w1)x*w3)) over an explicit async stream on a synthetic hash fixture
# hidden=6144 and compares element-by-element against an independent CPU FP64
# reference (RED/GREEN), plus checks the integrity of layer 0-2 pointers/strides.
add_executable(test_glm_moe_dense_compute test_glm_moe_dense_compute.cpp)
target_link_libraries(test_glm_moe_dense_compute PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_glm_moe_dense_compute)


# ── GLM-5.2 MoE Layer-3 router test (top_k=8 warp selection) ───────────────
add_executable(test_glm_moe_moe_router test_glm_moe_moe_router.cpp)
target_link_libraries(test_glm_moe_moe_router PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_glm_moe_moe_router)


# ── GLM-5.2 expert-stack test (NVFP4 dequant + SwiGLU, layers 3-77) ─────────
add_executable(test_glm_moe_experts test_glm_moe_experts.cpp)
target_link_libraries(test_glm_moe_experts PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_glm_moe_experts)


# ── TDD Cycle 9: GLM-5.2 MoE router + NVFP4 expert stacks (layers 3-77) ───────
# Runs RunGlmMoEStack (layers 3-77) through a single contiguous MoE arena
# (256 virtual expert-activation slices) on a synthetic E2M1/E4M3/E8M0 hash
# fixture and compares element-by-element against an independent CPU FP64
# reference (RED/GREEN) + checks the integrity of arena pointers/bounds.
add_executable(test_glm_moe_moe_compute test_glm_moe_moe_compute.cpp)
target_link_libraries(test_glm_moe_moe_compute PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_glm_moe_moe_compute)


# ── GLM-5.2 IndexShare attention test (mod-4 refresh, layers 3-77) ────────────
add_executable(test_glm_moe_attention test_glm_moe_attention.cpp)
target_link_libraries(test_glm_moe_attention PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_glm_moe_attention)


# ── GLM-5.2 full 79-layer loop + MTP epilogue test (generation) ───────
add_executable(test_glm_moe_inference test_glm_moe_inference.cpp)
target_link_libraries(test_glm_moe_inference PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_glm_moe_inference)


# GLM real-weight tests skip (exit 4 = Catch2 "all test cases skipped") when no
# GLM-5.2 checkpoint is present; that is not a failure per the project SKIP
# invariant, so tell CTest to report it as SKIPPED instead.

# ── TDD Cycle 10: GLM-5.2 full 79-layer SKWOS integration regression (unified
#    arena: IndexShare K-V cache + Layer-78 MTP spaces in one contiguous block)
#    Driving the complete execution loop on real weights with multi-turn
#    prompts; asserts no memory leak / no index-boundary truncation, and that
#    the final speculative Layer-78 MTP logits row converges element-for-element
#    against the independent CPU FP64 reference (RED/GREEN).
add_executable(test_glm_moe_full_compute test_glm_moe_full_compute.cpp)
target_link_libraries(test_glm_moe_full_compute PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_glm_moe_full_compute)


# ── TaskMosaic CE network-layer test (lsxnetwork) ───────────────────────────
add_executable(test_mosaic_network test_mosaic_network.cpp)
target_link_libraries(test_mosaic_network PRIVATE lsxnetwork Catch2::Catch2WithMain)

add_test(NAME test_mosaic_network COMMAND test_mosaic_network)

# ── TDD Phase 5: TaskMosaic CE static SPA frontend route harness ──────────────
# Verifies the in-memory static router serves the compiled www/ bundle: asset
# bytes are 64-byte cache-line aligned, GET returns exactly 200 OK with
# byte-for-byte content, a simultaneous multi-threaded hit-rate surge holds 200
# with zero corruption/alignment drift, and the /v1/chat/completions SSE stream
# frames a 200 event-stream. Links lsxnetwork only (no Arrow/CUDA).
add_executable(test_mosaic_frontend_route test_mosaic_frontend_route.cpp)
target_link_libraries(test_mosaic_frontend_route PRIVATE lsxnetwork Catch2::Catch2WithMain)

add_test(NAME test_mosaic_frontend_route COMMAND test_mosaic_frontend_route)

# ── TaskMosaic integration test: the orchestrator (lsx_orchestrator) ──
# Runs a simultaneous burst of OpenAI requests on both branches ("glm-5.2" and
# "deepseek-v4-flash") through a long lock-free SPSC pipeline and checks
# zero-copy retirement without token drift. Links lsxcommon (orchestrator + IModel)
# and lsxnetwork (FrameQueue).
add_executable(test_integrated_pipeline test_integrated_pipeline.cpp)
target_link_libraries(test_integrated_pipeline PRIVATE lsxcommon lsxnetwork Catch2::Catch2WithMain)
target_compile_definitions(test_integrated_pipeline PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

lsx_copy_arrow_dlls(test_integrated_pipeline)

add_test(NAME test_integrated_pipeline COMMAND test_integrated_pipeline)

# ── lsxfabric binary-token wire codec ────────────────────────────────────────
add_executable(test_fabric_wire test_fabric_wire.cpp)
target_link_libraries(test_fabric_wire PRIVATE lsxnetwork Catch2::Catch2WithMain)
add_test(NAME test_fabric_wire COMMAND test_fabric_wire)

# ── lsxfabric binary-token client ────────────────────────────────────────────
add_executable(test_fabric_client test_fabric_client.cpp)
target_link_libraries(test_fabric_client PRIVATE lsxnetwork Catch2::Catch2WithMain)
add_test(NAME test_fabric_client COMMAND test_fabric_client)

# ── incremental UTF-8-safe detokenizer ───────────────────────────────────────
add_executable(test_incremental_detokenizer test_incremental_detokenizer.cpp)
target_link_libraries(test_incremental_detokenizer PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_incremental_detokenizer COMMAND test_incremental_detokenizer)

# ── per-family vocabulary decode contract ────────────────────────────────────
# Walks every converted model present on the machine and asserts that what the
# decoder emits can be encoded again, so chat framing cannot break on a
# vocabulary that maps raw bytes to one-byte tokens. SKIPs without weights.
add_executable(test_tokenizer_vocab_contract test_tokenizer_vocab_contract.cpp)
target_link_libraries(test_tokenizer_vocab_contract PRIVATE lsxcommon
    Catch2::Catch2WithMain)
add_test(NAME test_tokenizer_vocab_contract COMMAND test_tokenizer_vocab_contract)
set_tests_properties(test_tokenizer_vocab_contract PROPERTIES LABELS inference RESOURCE_LOCK "gpu")
# Host-only (vocabulary contract), but it asserts on the tokenizer that the
# GPU stages also use, and it is ~0.1 s: locking it costs nothing and keeps
# the rule uniform. Listed here rather than allowlisted because the cost of
# being wrong in the other direction (a device call added later) is a
# concurrent-GPU flake nobody would attribute to this file.
set_tests_properties(test_tokenizer_vocab_contract PROPERTIES TIMEOUT 300)
set_tests_properties(test_tokenizer_vocab_contract PROPERTIES SKIP_RETURN_CODE 4)

# ── generation streaming handle ──────────────────────────────────────────────
add_executable(test_generation_stream test_generation_stream.cpp)
target_link_libraries(test_generation_stream PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_generation_stream COMMAND test_generation_stream)

# ── generic Hermes tool format ───────────────────────────────────────────────
add_executable(test_tool_format test_tool_format.cpp)
target_link_libraries(test_tool_format PRIVATE lsxcommon simdjson::simdjson Catch2::Catch2WithMain)
add_test(NAME test_tool_format COMMAND test_tool_format)

# ── Anthropic Messages codec ─────────────────────────────────────────────────
add_executable(test_anthropic_endpoint test_anthropic_endpoint.cpp)
target_link_libraries(test_anthropic_endpoint PRIVATE lsxnetwork simdjson::simdjson Catch2::Catch2WithMain)
add_test(NAME test_anthropic_endpoint COMMAND test_anthropic_endpoint)

# ── generic HTTP server wrapper (cpp-httplib) ────────────────────────────────
add_executable(test_http_server test_http_server.cpp)
target_link_libraries(test_http_server PRIVATE lsxnetwork httplib::httplib Catch2::Catch2WithMain)
add_test(NAME test_http_server COMMAND test_http_server)

# ── CUDA inference kernels test ─────────────────────────────────────────
# Model-neutral streaming-read ceiling probe. Its own OBJECT library with explicit
# include dirs and no lsxcommon at compile time: it needs Catch2 and
# test_kernel_budget.h only, and linking lsxcommon here would drag that target's
# private "-Xcompiler=/utf-8" CUDA option into this compile step (the nvcc misparse
# documented below test_deepseek_moe_inference_kernels). lsxcommon IS linked at the
# final executable's link stage, where test_kernel_budget.h's ProbeGpuPerfProfile
# symbol is actually needed.
add_library(test_stream_read_ceiling_cu OBJECT test_stream_read_ceiling.cu)
set_target_properties(test_stream_read_ceiling_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_stream_read_ceiling_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_stream_read_ceiling_cu PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
)
target_link_libraries(test_stream_read_ceiling_cu PRIVATE Catch2::Catch2WithMain)

add_executable(test_stream_read_ceiling)
set_target_properties(test_stream_read_ceiling PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_stream_read_ceiling PRIVATE
    test_stream_read_ceiling_cu lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_stream_read_ceiling COMMAND test_stream_read_ceiling)
set_tests_properties(test_stream_read_ceiling PROPERTIES LABELS perf RESOURCE_LOCK "gpu")
set_tests_properties(test_stream_read_ceiling PROPERTIES TIMEOUT 600)
# Skip-capable (no CUDA device -> SKIP in the test), so it needs the code, per
# test_ctest_labels_guard.
set_tests_properties(test_stream_read_ceiling PROPERTIES SKIP_RETURN_CODE 4)

# Compiled as an OBJECT library (same pattern as lsxcommon_cu) rather than
# passing the .cu file directly to add_executable(). Root cause found by
# direct experimentation: when a CUDA-compiling target links (directly or
# transitively) another target that itself carries a private
# "-Xcompiler=/utf-8" CUDA compile option (as lsxcommon_cu does), CMake's
# Visual Studio generator emits a duplicated, unwrapped bare "/utf-8" on the
# nvcc command line for the *consuming* target's own CUDA compile step —
# which nvcc 13.3.73 misparses as a second input file ("nvcc fatal: A single
# input file is required..."). Confirmed by removing the lsxcommon link from
# the compile step entirely, which eliminates the bare /utf-8.
# Fix: this OBJECT library gets its own explicit include dirs (mirroring
# lsxcommon_cu's) instead of linking lsxcommon at compile time. lsxcommon is
# linked only at the final executable's link stage below, where its actual
# symbols (not its CUDA compile options) are needed.
add_library(test_deepseek_moe_inference_kernels_cu OBJECT test_deepseek_moe_inference_kernels.cu)
set_target_properties(test_deepseek_moe_inference_kernels_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_deepseek_moe_inference_kernels_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_deepseek_moe_inference_kernels_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
)
target_compile_definitions(test_deepseek_moe_inference_kernels_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
)
target_link_libraries(test_deepseek_moe_inference_kernels_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)

add_executable(test_deepseek_moe_inference_kernels)
set_target_properties(test_deepseek_moe_inference_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_deepseek_moe_inference_kernels PRIVATE
    test_deepseek_moe_inference_kernels_cu lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_deepseek_moe_inference_kernels)


# test_deepseek_moe_moe — MoE orchestration tests (route_token, shared FFN, routed
# experts, layer mix). Same OBJECT-lib pattern as the kernel test above: the
# .cu must not inherit lsxcommon's C++ compile flags, so it compiles standalone
# and links against lsxcommon only at the executable's link stage.
add_library(test_deepseek_moe_moe_cu OBJECT test_deepseek_moe_moe.cu)
set_target_properties(test_deepseek_moe_moe_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_deepseek_moe_moe_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_deepseek_moe_moe_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
)
target_compile_definitions(test_deepseek_moe_moe_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
)
target_link_libraries(test_deepseek_moe_moe_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)

# ── Nemotron-3.5 device session (allocation + layer-slot bookkeeping) ──
add_library(test_nemotron_session_cu OBJECT test_nemotron_session.cu)
set_target_properties(test_nemotron_session_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "${LSX_CUDA_ARCH}")
target_compile_options(test_nemotron_session_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_include_directories(test_nemotron_session_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_nemotron_session_cu PRIVATE Catch2::Catch2WithMain)

add_executable(test_nemotron_session)
set_target_properties(test_nemotron_session PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_nemotron_session PRIVATE
    test_nemotron_session_cu lsxcommon Catch2::Catch2WithMain)

# ── DeepSeek-V4-Flash RMSNorm kernel test (GPU parity vs FP64 host ref) ──
add_library(test_deepseek_moe_norm_kernels_cu OBJECT
    test_deepseek_moe_norm_kernels.cu)
set_target_properties(test_deepseek_moe_norm_kernels_cu PROPERTIES
    CUDA_ARCHITECTURES "${LSX_CUDA_ARCH}")
target_compile_options(test_deepseek_moe_norm_kernels_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_include_directories(test_deepseek_moe_norm_kernels_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    )
target_compile_definitions(test_deepseek_moe_norm_kernels_cu PRIVATE
    LOGESTIX_HAS_CUDA)
target_link_libraries(test_deepseek_moe_norm_kernels_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)

add_executable(test_deepseek_moe_norm_kernels)
set_target_properties(test_deepseek_moe_norm_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_deepseek_moe_norm_kernels PRIVATE
    test_deepseek_moe_norm_kernels_cu lsxcommon Catch2::Catch2WithMain)

add_executable(test_deepseek_moe_moe)
set_target_properties(test_deepseek_moe_moe PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_deepseek_moe_moe PRIVATE
    test_deepseek_moe_moe_cu lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_deepseek_moe_moe)


# test_deepseek_moe_arena — single-allocation per-run arena + zero-alloc route_token.
# Same OBJECT-lib pattern as test_deepseek_moe_moe_cu: the .cu must not inherit
# lsxcommon's C++ compile flags, so it compiles standalone and links against
# lsxcommon only at the executable's link stage.
add_library(test_deepseek_moe_arena_cu OBJECT test_deepseek_moe_arena.cu)
set_target_properties(test_deepseek_moe_arena_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_deepseek_moe_arena_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_deepseek_moe_arena_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
)
target_compile_definitions(test_deepseek_moe_arena_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
)
target_link_libraries(test_deepseek_moe_arena_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)

add_executable(test_deepseek_moe_arena)
set_target_properties(test_deepseek_moe_arena PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_deepseek_moe_arena PRIVATE
    test_deepseek_moe_arena_cu lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_deepseek_moe_arena)


# ── Isolated sampling seam test (lsxcommon::sample, GreedySampler) ──────────
# Same OBJECT-lib pattern as test_deepseek_moe_moe_cu: the .cu must not inherit
# lsxcommon's C++ compile flags, so it compiles standalone and links lsxcommon
# only at the executable's link stage.
add_library(test_sampling_cu OBJECT test_sampling.cu)
set_target_properties(test_sampling_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_sampling_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_sampling_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
)
target_compile_definitions(test_sampling_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
)
target_link_libraries(test_sampling_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)

add_executable(test_sampling)
set_target_properties(test_sampling PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_sampling PRIVATE
    test_sampling_cu lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_sampling)

add_test(NAME test_sampling COMMAND test_sampling)

# ── GLM fast loader test (in-place BF16→FP16 kernel + pinned ring + arena) ──
add_library(test_glm_moe_load_fast_cu OBJECT test_glm_moe_load_fast.cu)
set_target_properties(test_glm_moe_load_fast_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_glm_moe_load_fast_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_glm_moe_load_fast_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
)
target_compile_definitions(test_glm_moe_load_fast_cu PRIVATE
    LOGESTIX_HAS_CUDA
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_moe_load_fast_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)

add_executable(test_glm_moe_load_fast)
set_target_properties(test_glm_moe_load_fast PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_glm_moe_load_fast PRIVATE
    test_glm_moe_load_fast_cu lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_glm_moe_load_fast)


# ── GLM-5.2 lifecycle test (ResourceGovernor load-once + leases) ────
add_library(test_glm_moe_model_lifecycle_cu OBJECT test_glm_moe_model_lifecycle.cu)
set_target_properties(test_glm_moe_model_lifecycle_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_glm_moe_model_lifecycle_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_glm_moe_model_lifecycle_cu PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
)
target_compile_definitions(test_glm_moe_model_lifecycle_cu PRIVATE
    LOGESTIX_HAS_CUDA
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_moe_model_lifecycle_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)

add_executable(test_glm_moe_model_lifecycle)
set_target_properties(test_glm_moe_model_lifecycle PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_glm_moe_model_lifecycle PRIVATE
    test_glm_moe_model_lifecycle_cu lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_glm_moe_model_lifecycle)


# ── Shared fast-load primitives test (GPU BF16→FP16 + pinned ring) ───────────
# Same OBJECT-lib pattern as test_glm_moe_load_fast_cu.
add_library(test_deepseek_moe_load_fast_cu OBJECT test_deepseek_moe_load_fast.cu)
set_target_properties(test_deepseek_moe_load_fast_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_deepseek_moe_load_fast_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_deepseek_moe_load_fast_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
)
target_compile_definitions(test_deepseek_moe_load_fast_cu PRIVATE
    LOGESTIX_HAS_CUDA
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_deepseek_moe_load_fast_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)

add_executable(test_deepseek_moe_load_fast)
set_target_properties(test_deepseek_moe_load_fast PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_deepseek_moe_load_fast PRIVATE
    test_deepseek_moe_load_fast_cu lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_deepseek_moe_load_fast)


# ── path registry (vault root + per-user cache dir) ────────────────────────
add_executable(test_path_registry test_path_registry.cpp)
target_link_libraries(test_path_registry PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_path_registry)

add_test(NAME test_path_registry COMMAND test_path_registry)

# ── VRAM-profile estimator test (conversion-time) ───────────────────
add_executable(test_vram_estimate test_vram_estimate.cpp)
target_link_libraries(test_vram_estimate PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_vram_estimate)

add_test(NAME test_vram_estimate COMMAND test_vram_estimate)

add_executable(test_resource_governor test_resource_governor.cpp)
target_link_libraries(test_resource_governor PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_resource_governor)

add_test(NAME test_resource_governor COMMAND test_resource_governor)

add_executable(test_hardware_profile test_hardware_profile.cpp)
target_link_libraries(test_hardware_profile PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_hardware_profile)

add_test(NAME test_hardware_profile COMMAND test_hardware_profile)

add_executable(test_lifecycle_plan test_lifecycle_plan.cpp)
target_link_libraries(test_lifecycle_plan PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_lifecycle_plan)

add_test(NAME test_lifecycle_plan COMMAND test_lifecycle_plan)

add_executable(test_runtime_tuner test_runtime_tuner.cpp)
target_link_libraries(test_runtime_tuner PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_runtime_tuner)

add_test(NAME test_runtime_tuner COMMAND test_runtime_tuner)

# ── Kimi INT4→BF16 dequant kernel test ──────────────────────────────────────
add_library(test_kimi_moe_int4_dequant_cu OBJECT test_kimi_moe_int4_dequant.cu)
set_target_properties(test_kimi_moe_int4_dequant_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_kimi_moe_int4_dequant_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_kimi_moe_int4_dequant_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
)
target_compile_definitions(test_kimi_moe_int4_dequant_cu PRIVATE LOGESTIX_HAS_CUDA)
target_link_libraries(test_kimi_moe_int4_dequant_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)
target_link_libraries(test_kimi_moe_int4_dequant_cu PRIVATE lsxcommon_kimi_cu)

add_executable(test_kimi_moe_int4_dequant)
set_target_properties(test_kimi_moe_int4_dequant PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_kimi_moe_int4_dequant PRIVATE
    test_kimi_moe_int4_dequant_cu lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_kimi_moe_int4_dequant ARROW_ONLY)


# ── Kimi tiktoken tokenizer test ───────────────────────────────────────
add_executable(test_kimi_moe_tokenizer test_kimi_moe_tokenizer.cpp)
target_link_libraries(test_kimi_moe_tokenizer PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_kimi_moe_tokenizer ARROW_ONLY)


# ── Kimi pinned loader test ───────────────────────────────────────────
add_executable(test_kimi_moe_loader test_kimi_moe_loader.cpp)
target_compile_definitions(test_kimi_moe_loader PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_kimi_moe_loader PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_kimi_moe_loader)

# Converts a synthetic checkpoint: the frozen gates (incl. the token ids) are
# relaxed at runtime, never by a compile define.

# ── Kimi MoE router test ─────────────────────────────────────────────────
add_library(test_kimi_moe_router_cu OBJECT test_kimi_moe_router.cu)
set_target_properties(test_kimi_moe_router_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_kimi_moe_router_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_kimi_moe_router_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
)
target_compile_definitions(test_kimi_moe_router_cu PRIVATE LOGESTIX_HAS_CUDA)
target_link_libraries(test_kimi_moe_router_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)
target_link_libraries(test_kimi_moe_router_cu PRIVATE lsxcommon_kimi_cu)

add_executable(test_kimi_moe_router)
set_target_properties(test_kimi_moe_router PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_kimi_moe_router PRIVATE
    test_kimi_moe_router_cu lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_kimi_moe_router ARROW_ONLY)


# ── Kimi MLA attention test ───────────────────────────────────────────────
add_library(test_kimi_moe_attention_cu OBJECT test_kimi_moe_attention.cu)
set_target_properties(test_kimi_moe_attention_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_kimi_moe_attention_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_kimi_moe_attention_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
)
target_compile_definitions(test_kimi_moe_attention_cu PRIVATE LOGESTIX_HAS_CUDA)
target_link_libraries(test_kimi_moe_attention_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)
target_link_libraries(test_kimi_moe_attention_cu PRIVATE lsxcommon_kimi_cu)

add_executable(test_kimi_moe_attention)
set_target_properties(test_kimi_moe_attention PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_kimi_moe_attention PRIVATE
    test_kimi_moe_attention_cu lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_kimi_moe_attention ARROW_ONLY)


# ── Kimi MLA rope/scale parity (host-only golden from the vendored code) ──
add_executable(test_kimi_moe_rope_parity test_kimi_moe_rope_parity.cpp)
target_compile_definitions(test_kimi_moe_rope_parity PRIVATE
    LSX_KIMI_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/kimi")
target_link_libraries(test_kimi_moe_rope_parity PRIVATE lsxcommon Catch2::Catch2WithMain)

# ── Kimi tiktoken id parity (golden from the checkpoint tokenizer) ────────
add_executable(test_kimi_moe_tokenizer_parity test_kimi_moe_tokenizer_parity.cpp)
target_compile_definitions(test_kimi_moe_tokenizer_parity PRIVATE
    LSX_KIMI_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/kimi")
target_link_libraries(test_kimi_moe_tokenizer_parity PRIVATE lsxcommon Catch2::Catch2WithMain)

# ── Kimi native conversation/tool codec (host-only) ───────────────────────
add_executable(test_kimi_tool_framing test_kimi_tool_framing.cpp)
target_compile_definitions(test_kimi_tool_framing PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_include_directories(test_kimi_tool_framing PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_kimi_tool_framing PRIVATE lsxcommon Catch2::Catch2WithMain)

# ── Kimi pinned MLA projection GEMV vs host decoder (real weights) ────────
add_library(test_kimi_moe_mla_proj_real_cu OBJECT test_kimi_moe_mla_proj_real.cu)
set_target_properties(test_kimi_moe_mla_proj_real_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_kimi_moe_mla_proj_real_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_kimi_moe_mla_proj_real_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
)
target_compile_definitions(test_kimi_moe_mla_proj_real_cu PRIVATE LOGESTIX_HAS_CUDA)
target_link_libraries(test_kimi_moe_mla_proj_real_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)
target_link_libraries(test_kimi_moe_mla_proj_real_cu PRIVATE lsxcommon_kimi_cu)

add_executable(test_kimi_moe_mla_proj_real)
set_target_properties(test_kimi_moe_mla_proj_real PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_kimi_moe_mla_proj_real PRIVATE
    test_kimi_moe_mla_proj_real_cu lsxcommon Catch2::Catch2WithMain)

# ── Kimi routed-expert FFN test ───────────────────────────────────────────
add_library(test_kimi_moe_experts_cu OBJECT test_kimi_moe_experts.cu)
set_target_properties(test_kimi_moe_experts_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_kimi_moe_experts_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_kimi_moe_experts_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
)
target_compile_definitions(test_kimi_moe_experts_cu PRIVATE
    LOGESTIX_HAS_CUDA
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_kimi_moe_experts_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)
target_link_libraries(test_kimi_moe_experts_cu PRIVATE lsxcommon_kimi_cu)

add_executable(test_kimi_moe_experts)
set_target_properties(test_kimi_moe_experts PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_kimi_moe_experts PRIVATE
    test_kimi_moe_experts_cu lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_kimi_moe_experts ARROW_ONLY)


# ── Kimi batched TC routed chain test (fast-math; SKIP on non-sm_12x) ──────
add_library(test_kimi_moe_tc_batch_cu OBJECT test_kimi_moe_tc_batch.cu)
set_target_properties(test_kimi_moe_tc_batch_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_kimi_moe_tc_batch_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_kimi_moe_tc_batch_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
)
target_compile_definitions(test_kimi_moe_tc_batch_cu PRIVATE
    LOGESTIX_HAS_CUDA
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_kimi_moe_tc_batch_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)
target_link_libraries(test_kimi_moe_tc_batch_cu PRIVATE lsxcommon_kimi_cu)

add_executable(test_kimi_moe_tc_batch)
set_target_properties(test_kimi_moe_tc_batch PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_kimi_moe_tc_batch PRIVATE
    test_kimi_moe_tc_batch_cu lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_kimi_moe_tc_batch ARROW_ONLY)


add_library(test_kimi_moe_tc_diag_cu OBJECT test_kimi_moe_tc_diag.cu)
set_target_properties(test_kimi_moe_tc_diag_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_kimi_moe_tc_diag_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_kimi_moe_tc_diag_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
)
target_compile_definitions(test_kimi_moe_tc_diag_cu PRIVATE
    LOGESTIX_HAS_CUDA
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_kimi_moe_tc_diag_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)
target_link_libraries(test_kimi_moe_tc_diag_cu PRIVATE lsxcommon_kimi_cu)

add_executable(test_kimi_moe_tc_diag)
set_target_properties(test_kimi_moe_tc_diag PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_kimi_moe_tc_diag PRIVATE
    test_kimi_moe_tc_diag_cu lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_kimi_moe_tc_diag ARROW_ONLY)


# ── Kimi model test (factory + lifecycle) ────────────────────────────────
add_executable(test_kimi_moe_model test_kimi_moe_model.cpp)
target_compile_definitions(test_kimi_moe_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_kimi_moe_model PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_kimi_moe_model ARROW_ONLY)


# ── Kimi SelfCheck test (real weights; SKIP without a model) ──────────────────


# ── Kimi perf baseline test (real weights; SKIP without a model) ──────────────
add_executable(test_blackwell_perf_kimi test_blackwell_perf_kimi.cpp)
target_compile_definitions(test_blackwell_perf_kimi PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_blackwell_perf_kimi PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_blackwell_perf_kimi ARROW_ONLY)

add_test(NAME test_blackwell_perf_kimi COMMAND test_blackwell_perf_kimi)
set_tests_properties(test_blackwell_perf_kimi PROPERTIES SKIP_RETURN_CODE 4)
set_tests_properties(test_blackwell_perf_kimi PROPERTIES RESOURCE_LOCK "gpu")

# ── Kimi full inference test (real weights; SKIP without a model) ─────────────
add_executable(test_kimi_moe_inference test_kimi_moe_inference.cpp)
target_compile_definitions(test_kimi_moe_inference PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_include_directories(test_kimi_moe_inference PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
)
target_link_libraries(test_kimi_moe_inference PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_kimi_moe_inference ARROW_ONLY)


# ── Kimi conversion test on the real checkpoint (SKIP without a model) ─────────
add_library(test_kimi_moe_int4_real_cu OBJECT test_kimi_moe_int4_real.cu)
set_target_properties(test_kimi_moe_int4_real_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_kimi_moe_int4_real_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_kimi_moe_int4_real_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/third_party/simdjson/include"
)
target_compile_definitions(test_kimi_moe_int4_real_cu PRIVATE LOGESTIX_HAS_CUDA)
target_link_libraries(test_kimi_moe_int4_real_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)
target_link_libraries(test_kimi_moe_int4_real_cu PRIVATE lsxcommon_kimi_cu)

add_executable(test_kimi_moe_int4_real)
set_target_properties(test_kimi_moe_int4_real PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_kimi_moe_int4_real PRIVATE
    test_kimi_moe_int4_real_cu lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_kimi_moe_int4_real ARROW_ONLY)


# ── Qwen3.5-35B-A3B tests (MoE) ───────────────────────────────────────────
add_executable(test_qwen_moe_convert test_qwen_moe_convert.cpp)
target_compile_definitions(test_qwen_moe_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_moe_convert PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_qwen_moe_convert)

# Synthetic tiny-geometry fixture: opt out of the frozen-geometry hard gate.

# ── Qwen3.6-35B-A3B tests (MoE, variant 36) ───────────────────────────────
add_executable(test_qwen_moe_36_convert test_qwen_moe_36_convert.cpp)
target_compile_definitions(test_qwen_moe_36_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_moe_36_convert PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_qwen_moe_36_convert)


# ── Ornith-1.5-35B-A3B tests (MoE, MLX source layout) ─────────────────────
add_executable(test_ornith_moe_convert test_ornith_moe_convert.cpp)
target_compile_definitions(test_ornith_moe_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_ornith_moe_convert PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_ornith_moe_convert)


add_executable(test_ornith_moe_loader test_ornith_moe_loader.cpp)
target_compile_definitions(test_ornith_moe_loader PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_ornith_moe_loader PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_ornith_moe_loader)


add_executable(test_ornith_moe_model test_ornith_moe_model.cpp)
target_include_directories(test_ornith_moe_model PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_compile_definitions(test_ornith_moe_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_ornith_moe_model PRIVATE lsxcommon Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_ornith_moe_model ARROW_ONLY)


# test_qwen_moe_kernels — MoE kernel tests (router top-k, routed FFN slot
# chain, shared-expert combine, generation driver). Same OBJECT-lib pattern as
# test_deepseek_moe_inference_kernels_cu / test_deepseek_moe_moe_cu: passing the .cu
# to add_executable() and linking lsxcommon (which carries a private
# "-Xcompiler=/utf-8" CUDA option) makes CMake's Visual Studio generator emit a
# duplicated, unwrapped bare "/utf-8" on the test's own nvcc line — which
# nvcc misparses as a second input file ("nvcc fatal: A single input file is
# required..."). The .cu compiles standalone and links lsxcommon only at the
# executable's link stage below.
add_library(test_qwen_moe_kernels_cu OBJECT test_qwen_moe_kernels.cu)
set_target_properties(test_qwen_moe_kernels_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_qwen_moe_kernels_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_qwen_moe_kernels_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/third_party/flatbuffers/include"
    "${PROJECT_SOURCE_DIR}/third_party/libzmq/include"
    "${CMAKE_BINARY_DIR}/third_party/libzmq"
    "${PROJECT_SOURCE_DIR}/libs/lsxnetwork/include"
    "${CMAKE_BINARY_DIR}/libs/lsxnetwork/generated"
    "${PROJECT_SOURCE_DIR}/third_party/simdjson/include"
)
target_compile_definitions(test_qwen_moe_kernels_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
    SPDLOG_COMPILED_LIB
    ZMQ_STATIC
    SIMDJSON_THREADS_ENABLED=1
)
target_link_libraries(test_qwen_moe_kernels_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)
target_compile_definitions(test_qwen_moe_kernels_cu PRIVATE
    LSX_KERNEL_BUDGETS_FILE="${LSX_KERNEL_BUDGETS}")

add_executable(test_qwen_moe_kernels)
set_target_properties(test_qwen_moe_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_qwen_moe_kernels PRIVATE
    test_qwen_moe_kernels_cu lsxcommon_moe_cu lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_qwen_moe_kernels)


# Ornith MoE kernel tests (duplicated qwen_moe layer, ornith namespace).
add_library(test_ornith_moe_kernels_cu OBJECT test_ornith_moe_kernels.cu)
set_target_properties(test_ornith_moe_kernels_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_ornith_moe_kernels_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_ornith_moe_kernels_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/third_party/flatbuffers/include"
    "${PROJECT_SOURCE_DIR}/third_party/libzmq/include"
    "${CMAKE_BINARY_DIR}/third_party/libzmq"
    "${PROJECT_SOURCE_DIR}/libs/lsxnetwork/include"
    "${CMAKE_BINARY_DIR}/libs/lsxnetwork/generated"
    "${PROJECT_SOURCE_DIR}/third_party/simdjson/include"
)
target_compile_definitions(test_ornith_moe_kernels_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
    SPDLOG_COMPILED_LIB
    ZMQ_STATIC
    SIMDJSON_THREADS_ENABLED=1
)
target_link_libraries(test_ornith_moe_kernels_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)

add_executable(test_ornith_moe_kernels)
set_target_properties(test_ornith_moe_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_ornith_moe_kernels PRIVATE
    test_ornith_moe_kernels_cu lsxcommon_ornith_cu lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_ornith_moe_kernels_cu PRIVATE
    LSX_KERNEL_BUDGETS_FILE="${LSX_KERNEL_BUDGETS}")

lsx_copy_arrow_dlls(test_ornith_moe_kernels)


add_executable(test_qwen_moe_loader test_qwen_moe_loader.cpp)
target_compile_definitions(test_qwen_moe_loader PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_moe_loader PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_qwen_moe_loader)


add_executable(test_qwen_moe_model test_qwen_moe_model.cpp)
target_include_directories(test_qwen_moe_model PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_compile_definitions(test_qwen_moe_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_qwen_moe_model PRIVATE lsxcommon Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_qwen_moe_model ARROW_ONLY)

add_executable(test_qwen_moe_36_model test_qwen_moe_36_model.cpp)
target_include_directories(test_qwen_moe_36_model PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_compile_definitions(test_qwen_moe_36_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_qwen_moe_36_model PRIVATE lsxcommon Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_qwen_moe_36_model ARROW_ONLY)


# ── FSISelfCheck (separate from SelfCheck): golden parity per family ──────


# ── CODESelfCheck (separate from SelfCheck/FSI): golden parity per family ──


# ── DETSelfCheck (separate from SelfCheck/FSI/CODE): determinism golden parity per family ──


# ── Expert NVFP4 packing test (passthrough + GPU quantize contract) ───
add_executable(test_nvfp4_pack test_nvfp4_pack.cpp)
target_link_libraries(test_nvfp4_pack PRIVATE lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_nvfp4_pack ARROW_ONLY)

add_test(NAME test_nvfp4_pack COMMAND test_nvfp4_pack)
set_tests_properties(test_nvfp4_pack PROPERTIES SKIP_RETURN_CODE 4)
set_tests_properties(test_nvfp4_pack PROPERTIES RESOURCE_LOCK "gpu")

# MiMo-V2.6-Flash-RL frozen geometry passport (host-only).
add_executable(test_mimo_moe_geometry test_mimo_moe_geometry.cpp)
target_include_directories(test_mimo_moe_geometry PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_mimo_moe_geometry PRIVATE Catch2::Catch2WithMain)

# MiMo-V2.6-Flash-RL ChatML token bridge + tool codec (host-only).
add_executable(test_mimo_tool_framing test_mimo_tool_framing.cpp)
target_compile_definitions(test_mimo_tool_framing PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_mimo_tool_framing PRIVATE lsxcommon Catch2::Catch2WithMain)

# MiMo-V2.6-Flash-RL converter (synthetic E2E, label convert, bounded timeout).
add_executable(test_mimo_moe_convert test_mimo_moe_convert.cpp)
target_compile_definitions(test_mimo_moe_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_mimo_moe_convert PRIVATE lsxcommon Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_mimo_moe_convert)

# MiMo-V2.6-Flash-RL layout loader (synthetic converted fixture).
add_executable(test_mimo_moe_loader test_mimo_moe_loader.cpp)
target_include_directories(test_mimo_moe_loader PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_compile_definitions(test_mimo_moe_loader PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_mimo_moe_loader PRIVATE lsxcommon Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_mimo_moe_loader)

# MiMo routed-expert host store + device slot LRU (host logic + device staging).
add_executable(test_mimo_moe_expert_cache test_mimo_moe_expert_cache.cpp)
target_link_libraries(test_mimo_moe_expert_cache PRIVATE lsxcommon Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_mimo_moe_expert_cache ARROW_ONLY)

# MiMo non-GEMM kernel parity (RMSNorm / router / SwiGLU) — CUDA OBJECT lib.
add_library(test_mimo_cu OBJECT
    test_mimo_moe_kernels.cu
    test_mimo_moe_attn.cu
)
set_target_properties(test_mimo_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_mimo_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(test_mimo_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_include_directories(test_mimo_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_mimo_cu PRIVATE Catch2::Catch2WithMain)

add_executable(test_mimo_moe_kernels)
set_target_properties(test_mimo_moe_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_mimo_moe_kernels PRIVATE test_mimo_cu lsxcommon Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_mimo_moe_kernels)

# MiMo FP8/MXFP4 dequant parity (host scalar reference).
add_executable(test_mimo_moe_dequant test_mimo_moe_dequant.cpp)
target_link_libraries(test_mimo_moe_dequant PRIVATE lsxcommon Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_mimo_moe_dequant)

# MiMo model registration + synthetic Load/Infer.
add_executable(test_mimo_moe_model test_mimo_moe_model.cpp)
target_compile_definitions(test_mimo_moe_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_mimo_moe_model PRIVATE lsxcommon Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_mimo_moe_model)

# MiMo SelfCheck (real weights; SKIP when absent).

# MiMo FSI / CODE / DET golden-parity gates (real weights + golden; SKIP).


# Guard: model code file naming convention (host-only, no lsxcommon deps).
lsx_add_test(NAME test_code_naming_guard
    DEFINE "LSX_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\"")

# Guard: absl::StrFormat & friends are printf-style; fmt-style "{}" in their
# format strings prints literally (checker inert on GCC). Host-only static scan.
lsx_add_test(NAME test_absl_format_guard
    LINK absl::str_format
    DEFINE "LSX_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\"")

# Guard: abseil containers do not compile under NVCC, so no .cu/.cuh may
# transitively include <absl/container/...> (raw_hash_map C2794/C2938).
# Host-only transitive-closure scan; the machine check for absl-in-cuda-cpp §2.
lsx_add_test(NAME test_absl_container_guard
    LINK absl::flat_hash_map
    DEFINE "LSX_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\"")

# Guard: host logging goes through spdlog / lsxcommon::log only; no std::cout
# and no stdio logging (printf/fprintf/...) in production code. Host-only
# static scan of libs/ and apps/ with a two-way allowlist (device printf,
# env-gated perf probes, logger teardown).
lsx_add_test(NAME test_logging_guard
    DEFINE "LSX_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\"")

# Guard: the E1 FastMathScope depth must be process-global across TUs. The
# header used to keep it in an anonymous-namespace static (one copy per TU),
# which leaked the counter and printed false VIOLATION lines. Two TUs on
# purpose: the second one enters, the first one observes.
add_executable(test_fast_math_guard_cross_tu
    test_fast_math_guard_cross_tu.cpp
    test_fast_math_guard_other_tu.cpp)
target_link_libraries(test_fast_math_guard_cross_tu PRIVATE lsxcommon
    Catch2::Catch2WithMain)
add_test(NAME test_fast_math_guard_cross_tu COMMAND test_fast_math_guard_cross_tu)

add_executable(test_cuda_event_wait test_cuda_event_wait.cpp)
target_link_libraries(test_cuda_event_wait PRIVATE lsxcommon
    Catch2::Catch2WithMain)
add_test(NAME test_cuda_event_wait COMMAND test_cuda_event_wait)

add_executable(test_stream_scratch test_stream_scratch.cpp)
target_link_libraries(test_stream_scratch PRIVATE lsxcommon
    Catch2::Catch2WithMain)
add_test(NAME test_stream_scratch COMMAND test_stream_scratch)

add_executable(test_granite_tool_framing test_granite_tool_framing.cpp)
target_link_libraries(test_granite_tool_framing PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# Guard: routed-expert host storage — no family-private OWNING expert
# container; every entry view must alias the neutral ExpertStore; the legacy
# all-experts-in-pinned-slabs store (MoEExpertStore) must not spread. Host-only
# static scan; the machine check for ingesting-a-model R1/R2.
lsx_add_test(NAME test_expert_store_guard
    DEFINE "LSX_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\"")

# Guard: license compliance — no GPL/LGPL/AGPL/copyleft headers or SPDX license
# tags in first-party source (would block the commercial Enterprise license).
# Host-only static scan of libs/, apps/, tests/ and tools/.
lsx_add_test(NAME test_license_guard
    DEFINE "LSX_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\"")

# Kimi MoE expert cache bookkeeping (host-only, pure logic, no CUDA).
add_executable(test_kimi_moe_expert_cache test_kimi_moe_expert_cache.cpp)
target_link_libraries(test_kimi_moe_expert_cache PRIVATE lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_kimi_moe_expert_cache PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include")

# ── Nemotron-3.5 host reference math (header-only, no CUDA) ───────────────
add_executable(test_nemotron_moe_math test_nemotron_moe_math.cpp)
target_link_libraries(test_nemotron_moe_math PRIVATE Catch2::Catch2WithMain)
target_include_directories(test_nemotron_moe_math PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")

# ── Nemotron-3-Ultra-550B converter (synthetic checkpoint only) ───────────
add_executable(test_nemotron_moe_ultra_convert
    test_nemotron_moe_ultra_convert.cpp)
target_compile_definitions(test_nemotron_moe_ultra_convert PRIVATE
    LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1)
target_link_libraries(test_nemotron_moe_ultra_convert PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_nemotron_moe_ultra_convert PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
lsx_copy_arrow_dlls(test_nemotron_moe_ultra_convert)

# ── Nemotron-3-Ultra-550B frozen passport (host-only, no CUDA) ──────────
# Links lsxcommon only so the shared synthetic fixture (which pulls the
# neutral safetensors/Arrow primitives) can be included; the passport itself
# is header-only.
add_executable(test_nemotron_moe_ultra_constants
    test_nemotron_moe_ultra_constants.cpp)
target_link_libraries(test_nemotron_moe_ultra_constants
    PRIVATE lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_nemotron_moe_ultra_constants PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include")

# ── GigaChat3.5 host reference math (header-only, no CUDA) ────────────────
add_executable(test_gigachat_moe_math test_gigachat_moe_math.cpp)
target_link_libraries(test_gigachat_moe_math PRIVATE Catch2::Catch2WithMain)
target_include_directories(test_gigachat_moe_math PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")

# ── GigaChat3.5-432B-A28B-Reasoning tests (hybrid MLA+GDN+MoE) ────────────
add_executable(test_gigachat_moe_convert test_gigachat_moe_convert.cpp)
target_compile_definitions(test_gigachat_moe_convert PRIVATE
    LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1)
target_link_libraries(test_gigachat_moe_convert PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_gigachat_moe_convert PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
lsx_copy_arrow_dlls(test_gigachat_moe_convert ARROW_ONLY)

add_executable(test_gigachat_moe_kernels test_gigachat_moe_kernels.cpp)
target_link_libraries(test_gigachat_moe_kernels PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_gigachat_moe_kernels PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")

add_executable(test_gigachat_moe_loader test_gigachat_moe_loader.cpp)
target_link_libraries(test_gigachat_moe_loader PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_gigachat_moe_loader PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")

add_executable(test_gigachat_moe_tokenizer_parity
    test_gigachat_moe_tokenizer_parity.cpp)
target_include_directories(test_gigachat_moe_tokenizer_parity PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(test_gigachat_moe_tokenizer_parity PRIVATE
    "LSX_GIGACHAT_FIXTURE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/gigachat\"")
target_link_libraries(test_gigachat_moe_tokenizer_parity PRIVATE lsxcommon
    Catch2::Catch2WithMain)

add_executable(test_gigachat_moe_model test_gigachat_moe_model.cpp)
target_link_libraries(test_gigachat_moe_model PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_gigachat_moe_model PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")


add_executable(test_gigachat_moe_expert_cache test_gigachat_moe_expert_cache.cpp)
target_link_libraries(test_gigachat_moe_expert_cache PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_gigachat_moe_expert_cache PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")

add_executable(test_gigachat_moe_load_fast test_gigachat_moe_load_fast.cpp)
target_link_libraries(test_gigachat_moe_load_fast PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_gigachat_moe_load_fast PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
lsx_copy_arrow_dlls(test_gigachat_moe_load_fast ARROW_ONLY)

# test_gigachat_moe_devkernels — device-kernel parity (standalone .cu object +
# link lsxcommon at the exe; same pattern as test_nemotron_moe_kernels).
add_library(test_gigachat_moe_devkernels_cu OBJECT test_gigachat_moe_devkernels.cu)
set_target_properties(test_gigachat_moe_devkernels_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_gigachat_moe_devkernels_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_gigachat_moe_devkernels_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/third_party/flatbuffers/include"
    "${PROJECT_SOURCE_DIR}/third_party/libzmq/include"
    "${CMAKE_BINARY_DIR}/third_party/libzmq"
    "${PROJECT_SOURCE_DIR}/libs/lsxnetwork/include"
    "${CMAKE_BINARY_DIR}/libs/lsxnetwork/generated"
    "${PROJECT_SOURCE_DIR}/third_party/simdjson/include"
)
target_compile_definitions(test_gigachat_moe_devkernels_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
    SPDLOG_COMPILED_LIB
    ZMQ_STATIC
    SIMDJSON_THREADS_ENABLED=1
)
target_link_libraries(test_gigachat_moe_devkernels_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)
target_compile_definitions(test_gigachat_moe_devkernels_cu PRIVATE
    LSX_KERNEL_BUDGETS_FILE="${LSX_KERNEL_BUDGETS}")

add_executable(test_gigachat_moe_devkernels)
set_target_properties(test_gigachat_moe_devkernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_gigachat_moe_devkernels PRIVATE
    test_gigachat_moe_devkernels_cu lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_gigachat_moe_devkernels ARROW_ONLY)

# The two latency-budget cases SKIP in a Debug build (they need an optimized
# binary to be judged against the budget), so the target must tolerate exit 4.

# test_gigachat_moe_attn — MLA decode parity + bit-exact rope. Same standalone
# .cu object pattern; -ffp-contract=off keeps the host rope reference aligned
# with the -fmad=false device kernel.
add_library(test_gigachat_moe_attn_cu OBJECT test_gigachat_moe_attn.cu)
set_target_properties(test_gigachat_moe_attn_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_gigachat_moe_attn_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    -fmad=false;-Xcompiler=-ffp-contract=off;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_gigachat_moe_attn_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/third_party/flatbuffers/include"
    "${PROJECT_SOURCE_DIR}/third_party/libzmq/include"
    "${CMAKE_BINARY_DIR}/third_party/libzmq"
    "${PROJECT_SOURCE_DIR}/libs/lsxnetwork/include"
    "${CMAKE_BINARY_DIR}/libs/lsxnetwork/generated"
    "${PROJECT_SOURCE_DIR}/third_party/simdjson/include"
)
target_compile_definitions(test_gigachat_moe_attn_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
    SPDLOG_COMPILED_LIB
    ZMQ_STATIC
    SIMDJSON_THREADS_ENABLED=1
)
target_link_libraries(test_gigachat_moe_attn_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)

add_executable(test_gigachat_moe_attn)
set_target_properties(test_gigachat_moe_attn PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_gigachat_moe_attn PRIVATE
    test_gigachat_moe_attn_cu lsxcommon Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_gigachat_moe_attn ARROW_ONLY)

# test_gigachat_moe_parity — whole-stack synthetic parity vs a host-F32
# reference. Same standalone .cu object pattern.
add_library(test_gigachat_moe_parity_cu OBJECT test_gigachat_moe_parity.cu)
set_target_properties(test_gigachat_moe_parity_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_gigachat_moe_parity_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    -fmad=false;-Xcompiler=-ffp-contract=off;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_gigachat_moe_parity_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/third_party/flatbuffers/include"
    "${PROJECT_SOURCE_DIR}/third_party/libzmq/include"
    "${CMAKE_BINARY_DIR}/third_party/libzmq"
    "${PROJECT_SOURCE_DIR}/libs/lsxnetwork/include"
    "${CMAKE_BINARY_DIR}/libs/lsxnetwork/generated"
    "${PROJECT_SOURCE_DIR}/third_party/simdjson/include"
)
target_compile_definitions(test_gigachat_moe_parity_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
    SPDLOG_COMPILED_LIB
    ZMQ_STATIC
    SIMDJSON_THREADS_ENABLED=1
)
target_link_libraries(test_gigachat_moe_parity_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)

add_executable(test_gigachat_moe_parity)
set_target_properties(test_gigachat_moe_parity PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_gigachat_moe_parity PRIVATE
    test_gigachat_moe_parity_cu lsxcommon Catch2::Catch2WithMain)
lsx_copy_arrow_dlls(test_gigachat_moe_parity ARROW_ONLY)

# ── Nemotron-3.5-Lightning-30B-A3B tests (Mamba-2/MoE/attention hybrid) ───
add_executable(test_nemotron_moe_convert test_nemotron_moe_convert.cpp)
target_compile_definitions(test_nemotron_moe_convert PRIVATE
    LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1)
target_link_libraries(test_nemotron_moe_convert PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_nemotron_moe_convert PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)

lsx_copy_arrow_dlls(test_nemotron_moe_convert)


# test_nemotron_moe_nvfp4_convert — pre-quantized (ModelOpt) source layout:
# NVFP4 routed/shared experts + lm_head, per-tensor FP8 mamba in/out_proj,
# BF16 attention with FP8 KV-cache scales, F32 router, MTP ballast.
add_executable(test_nemotron_moe_nvfp4_convert
               test_nemotron_moe_nvfp4_convert.cpp)
target_link_libraries(test_nemotron_moe_nvfp4_convert PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_nemotron_moe_nvfp4_convert PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
lsx_copy_arrow_dlls(test_nemotron_moe_nvfp4_convert)

# test_nemotron_moe_nvfp4_loader — the pre-quantized distribution's own frozen
# passport: geometry, the stop SET, and the model-name cross-load guard.
add_executable(test_nemotron_moe_nvfp4_loader
               test_nemotron_moe_nvfp4_loader.cpp)
target_link_libraries(test_nemotron_moe_nvfp4_loader PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_nemotron_moe_nvfp4_loader PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)

# test_nemotron_moe_nvfp4_model — factory/Load/Infer/streaming for the
# pre-quantized distribution's own model class (synthetic converted fixture).
add_executable(test_nemotron_moe_nvfp4_model
               test_nemotron_moe_nvfp4_model.cpp)
target_compile_definitions(test_nemotron_moe_nvfp4_model PRIVATE
    LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1)
target_link_libraries(test_nemotron_moe_nvfp4_model PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_nemotron_moe_nvfp4_model PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
lsx_copy_arrow_dlls(test_nemotron_moe_nvfp4_model)

# test_nemotron_moe_nvfp4_selfcheck — real-weight Paris SelfCheck, SKIP 4 when
# the converted tree is absent.

# test_nemotron_moe_nvfp4_fsi_selfcheck — the FSI parity gate (SKIP 4).

# test_nemotron_moe_nvfp4_code_selfcheck — the CODE parity gate (SKIP 4).

# test_nemotron_moe_nvfp4_det_selfcheck — the DET parity gate (SKIP 4).

# test_nemotron_moe_tokenizer_parity — real-vault tokenizer gate (SKIP 4).
# Shared by both sibling variants: they ship one byte-identical tokenizer.json,
# so one target polices the pair.
add_executable(test_nemotron_moe_tokenizer_parity
               test_nemotron_moe_tokenizer_parity.cpp)
target_compile_definitions(test_nemotron_moe_tokenizer_parity PRIVATE
    LSX_NEMOTRON_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/nemotron")
target_link_libraries(test_nemotron_moe_tokenizer_parity PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_nemotron_moe_tokenizer_parity PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)

add_executable(test_nemotron_moe_loader test_nemotron_moe_loader.cpp)
target_link_libraries(test_nemotron_moe_loader PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_nemotron_moe_loader PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)

# test_nemotron_moe_expert_cache — routed-expert host store + device LRU
# bookkeeping (slot layout/policy host-only, allocator/staging on device).
add_executable(test_nemotron_moe_expert_cache
    test_nemotron_moe_expert_cache.cpp)
target_link_libraries(test_nemotron_moe_expert_cache PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_nemotron_moe_expert_cache PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)

add_executable(test_nemotron_moe_model test_nemotron_moe_model.cpp)
target_compile_definitions(test_nemotron_moe_model PRIVATE
    LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1)
target_link_libraries(test_nemotron_moe_model PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_nemotron_moe_model PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_compile_definitions(test_nemotron_moe_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

# test_nemotron_moe_kernels — Mamba-2 mixer kernels. Same OBJECT-lib pattern as
# test_qwen_moe_kernels (standalone .cu object + link lsxcommon at the exe).
add_library(test_nemotron_moe_kernels_cu OBJECT test_nemotron_moe_kernels.cu)
set_target_properties(test_nemotron_moe_kernels_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_nemotron_moe_kernels_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    ${LSX_CUDA_GENCODE_FLAGS};
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_nemotron_moe_kernels_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/third_party/flatbuffers/include"
    "${PROJECT_SOURCE_DIR}/third_party/libzmq/include"
    "${CMAKE_BINARY_DIR}/third_party/libzmq"
    "${PROJECT_SOURCE_DIR}/libs/lsxnetwork/include"
    "${CMAKE_BINARY_DIR}/libs/lsxnetwork/generated"
    "${PROJECT_SOURCE_DIR}/third_party/simdjson/include"
)
target_compile_definitions(test_nemotron_moe_kernels_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
    SPDLOG_COMPILED_LIB
    ZMQ_STATIC
    SIMDJSON_THREADS_ENABLED=1
)
target_link_libraries(test_nemotron_moe_kernels_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)
target_compile_definitions(test_nemotron_moe_kernels_cu PRIVATE
    LSX_KERNEL_BUDGETS_FILE="${LSX_KERNEL_BUDGETS}")

add_executable(test_nemotron_moe_kernels)
set_target_properties(test_nemotron_moe_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_nemotron_moe_kernels PRIVATE
    test_nemotron_moe_kernels_cu lsxcommon Catch2::Catch2WithMain)

lsx_copy_arrow_dlls(test_nemotron_moe_kernels)


# ── Qwen3.8-Flash-Next tests (qwen4_exp MoE) ──────────────────────────────
add_executable(test_qwen_moe_exp_convert test_qwen_moe_exp_convert.cpp)
target_compile_definitions(test_qwen_moe_exp_convert PRIVATE
    LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1)
target_link_libraries(test_qwen_moe_exp_convert PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_qwen_moe_exp_convert PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)

lsx_copy_arrow_dlls(test_qwen_moe_exp_convert)

# Synthetic tiny-geometry fixture: opt out of the frozen-geometry hard gate.

add_executable(test_qwen_moe_exp_loader test_qwen_moe_exp_loader.cpp)
target_compile_definitions(test_qwen_moe_exp_loader PRIVATE
    LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1)
target_link_libraries(test_qwen_moe_exp_loader PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_qwen_moe_exp_loader PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)

lsx_copy_arrow_dlls(test_qwen_moe_exp_loader)


# test_qwen_moe_exp_kernels — qwen4_exp kernel tests (hc GatedResidual, QSA
# indexer, PLE inject) against host-replica oracles.
add_library(test_qwen_moe_exp_kernels_cu OBJECT test_qwen_moe_exp_kernels.cu)
set_target_properties(test_qwen_moe_exp_kernels_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(test_qwen_moe_exp_kernels_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(test_qwen_moe_exp_kernels_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(test_qwen_moe_exp_kernels_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${PROJECT_SOURCE_DIR}/third_party/simdjson/include"
)
target_link_libraries(test_qwen_moe_exp_kernels_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)
target_compile_definitions(test_qwen_moe_exp_kernels_cu PRIVATE
    LSX_KERNEL_BUDGETS_FILE="${LSX_KERNEL_BUDGETS}")

add_executable(test_qwen_moe_exp_kernels)
set_target_properties(test_qwen_moe_exp_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_qwen_moe_exp_kernels PRIVATE
    test_qwen_moe_exp_kernels_cu lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_qwen_moe_exp_kernels PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
lsx_copy_arrow_dlls(test_qwen_moe_exp_kernels ARROW_ONLY)

# test_glm_moe_flash_kernels — GLM-5.3-Flash kernel tests (conv1d, KDA
# per-channel-decay gated delta, router) against host references.
add_library(test_glm_moe_flash_kernels_cu OBJECT test_glm_moe_flash_kernels.cu)
set_target_properties(test_glm_moe_flash_kernels_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(test_glm_moe_flash_kernels_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(test_glm_moe_flash_kernels_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(test_glm_moe_flash_kernels_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${PROJECT_SOURCE_DIR}/third_party/simdjson/include"
)
target_link_libraries(test_glm_moe_flash_kernels_cu PRIVATE Catch2::Catch2WithMain lsx_dep_includes)
target_compile_definitions(test_glm_moe_flash_kernels_cu PRIVATE
    LSX_KERNEL_BUDGETS_FILE="${LSX_KERNEL_BUDGETS}")

add_executable(test_glm_moe_flash_kernels)
set_target_properties(test_glm_moe_flash_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_glm_moe_flash_kernels PRIVATE
    test_glm_moe_flash_kernels_cu lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_glm_moe_flash_kernels PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
lsx_copy_arrow_dlls(test_glm_moe_flash_kernels ARROW_ONLY)

add_executable(test_glm_moe_flash_loader test_glm_moe_flash_loader.cpp)
target_compile_definitions(test_glm_moe_flash_loader PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_glm_moe_flash_loader PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_glm_moe_flash_loader PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
lsx_copy_arrow_dlls(test_glm_moe_flash_loader ARROW_ONLY)
# Synthetic-converted fixture: the load-side frozen-geometry gate is open
# only under this env (loader-test discipline: no vault loads).

add_executable(test_glm_moe_flash_model test_glm_moe_flash_model.cpp)
target_compile_definitions(test_glm_moe_flash_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_glm_moe_flash_model PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_glm_moe_flash_model PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
lsx_copy_arrow_dlls(test_glm_moe_flash_model ARROW_ONLY)
# Synthetic-converted fixture: load-side gates open only under this env.


add_executable(test_qwen_moe_exp_model test_qwen_moe_exp_model.cpp)
target_compile_definitions(test_qwen_moe_exp_model PRIVATE
    LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1)
target_link_libraries(test_qwen_moe_exp_model PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_qwen_moe_exp_model PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
lsx_copy_arrow_dlls(test_qwen_moe_exp_model)


# §E1c early-EOS gate. Its own executable on purpose: the gate is only a pure
# function of its inputs in a fresh process (OPEN_TASKS §E1d).
add_executable(test_qwen_moe_exp_eos_selfcheck test_qwen_moe_exp_eos_selfcheck.cpp)
target_link_libraries(test_qwen_moe_exp_eos_selfcheck PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_qwen_moe_exp_eos_selfcheck PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
lsx_copy_arrow_dlls(test_qwen_moe_exp_eos_selfcheck)

# Guard: infrastructure centralization (dependency paths, shared helpers,
# CUDA arch). Host-only static scan, see spec 2026-11-03.
lsx_add_test(NAME test_infrastructure_guard
    DEFINE "LSX_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\"")
# Guard: model independence — unique frozen passports, no cross-model
# geometry borrowing (host-only, no lsxcommon deps).
lsx_add_test(NAME test_model_independence_guard
    LINK absl::btree
    DEFINE "LSX_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\""
    INCLUDE "${LSX_MANIFEST_GENERATED_DIR}")

# Guard: ctest category-label discipline (fast | inference | perf).
# Host-only static scan of tests/CMakeLists.txt (see spec 2026-09-10).
lsx_add_test(NAME test_ctest_labels_guard
    DEFINE "LSX_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\""
           "LSX_MANIFEST_GENERATED_DIR=\"${LSX_MANIFEST_GENERATED_DIR}\"")

# Guard: model test-suite autonomy + full-lifecycle coverage. Host-only static
# scan: per-model lifecycle registry (convert -> load -> model -> kernels ->
# selfcheck), SKIP_RETURN_CODE 4 on selfcheck targets, and a cross-family
# fixture-borrowing ban (test-side mirror of model independence).
lsx_add_test(NAME test_model_suite_guard
    DEFINE "LSX_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\""
           "LSX_MANIFEST_GENERATED_DIR=\"${LSX_MANIFEST_GENERATED_DIR}\""
    INCLUDE "${LSX_MANIFEST_GENERATED_DIR}")

# ── FSISelfCheck: host-only extraction/golden/harness unit tests (no GPU) ──
add_executable(test_fsi_selfcheck test_fsi_selfcheck.cpp)
target_link_libraries(test_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_fsi_selfcheck COMMAND test_fsi_selfcheck)

# ── CODESelfCheck: host-only extraction/golden/harness unit tests (no GPU) ──
add_executable(test_code_selfcheck test_code_selfcheck.cpp)
target_link_libraries(test_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_code_selfcheck COMMAND test_code_selfcheck)

# ── DETSelfCheck: host-only extraction/golden/harness unit tests (no GPU) ──
add_executable(test_det_selfcheck test_det_selfcheck.cpp)
target_link_libraries(test_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_det_selfcheck COMMAND test_det_selfcheck)

# ── Test category labels (fast | convert | inference | perf) ──────────────
# fast     — host-only: runs on every default ctest invocation.
# convert  — model-converter tests: run when a converter changes
#            (tools/test_router.cmake) or explicitly (ctest -L convert).
# inference — CUDA/model contour: runs when the inference contour changes
#             (tools/test_router.cmake).
# perf     — real-weight perf baselines: manual command only (ctest -L perf).
# Discipline guard: tests/test_ctest_labels_guard.cpp
set_tests_properties(
    test_core
    test_fast_load
    test_safetensors
    test_converter_helpers
    test_geometry_guard
    test_gpu_perf
    test_gpu_power
    test_perf_baseline_logic
    test_schema_views
    test_arrow_validation
    test_arrow_usage_guard
    test_expert_store
    test_model_integrity
    test_model_factory
    test_generation_cli
    test_json_escape
    test_model_runtime
    test_fsi_selfcheck
    test_code_selfcheck
    test_det_selfcheck
    test_model_engine
    test_tokenizer
    test_mosaic_network
    test_mosaic_frontend_route
    test_integrated_pipeline
    test_fabric_wire
    test_fabric_client
    test_path_registry
    test_vram_estimate
    test_resource_governor
    test_hardware_profile
    test_lifecycle_plan
    test_runtime_tuner
    test_code_naming_guard
    test_absl_format_guard
    test_absl_container_guard
    test_logging_guard
    test_fast_math_guard_cross_tu
    test_cuda_event_wait
    test_stream_scratch
    test_expert_store_guard
    test_license_guard
    test_model_independence_guard
    test_ctest_labels_guard
    test_infrastructure_guard
    test_model_suite_guard
    test_incremental_detokenizer
    test_generation_stream
    test_tool_format
    test_anthropic_endpoint
    test_http_server
    PROPERTIES LABELS fast)

set_tests_properties(
    test_nvfp4_pack
    PROPERTIES LABELS convert)

# Every one of these executes device kernels, so the group takes the GPU lock
# with the label. Registering them in a group is also why the label guard could
# not see them: it reads literal `set_tests_properties(<name> ...)` calls, and a
# name inside a group list is invisible to that scan (see OPEN_TASKS §G1).
set_tests_properties(
    test_sampling
    test_lsx_nvfp4_dispatch
    test_lsx_gemv
    test_perf_knobs_apply
    test_blackwell_tc_mma
    test_lsx_kernels
    test_blackwell_micro_gemm
    PROPERTIES LABELS inference RESOURCE_LOCK "gpu")

set_tests_properties(
    test_blackwell_perf_glm
    test_blackwell_perf_dense
    test_blackwell_perf_moe_qwen
    test_blackwell_perf_moe_gemma
    test_blackwell_perf_deepseek
    test_blackwell_perf_kimi
    PROPERTIES LABELS perf)

# Converter-test discipline: synthetic fixtures only, bounded runtime.
# A convert test must never convert a real model (guard:
# test_ctest_labels_guard / CheckConvertTimeout). 120 s is generous
# headroom over the ~4 s full convert sweep, tight enough to kill any
# accidental full-checkpoint conversion.
set_tests_properties(
    test_nvfp4_pack
    PROPERTIES TIMEOUT 120)

# ── tokenizer arrows loader (host-only, index-only fixture) ──
add_executable(test_tokenizer_arrows test_tokenizer_arrows.cpp)
target_compile_definitions(test_tokenizer_arrows PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_tokenizer_arrows PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_tokenizer_arrows COMMAND test_tokenizer_arrows)
set_tests_properties(test_tokenizer_arrows PROPERTIES LABELS fast)

# ── prompt_bridge framing bridge (host-only, index-only fixture) ──
add_executable(test_prompt_bridge test_prompt_bridge.cpp)
target_compile_definitions(test_prompt_bridge PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_prompt_bridge PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_prompt_bridge COMMAND test_prompt_bridge)
set_tests_properties(test_prompt_bridge PROPERTIES LABELS fast)

# ── Arrow memory/device diagnostics (host-only) ──
add_executable(test_diagnostics test_diagnostics.cpp)
target_link_libraries(test_diagnostics PRIVATE lsxcommon arrow_shared
    arrow_cuda_shared simdjson::simdjson Catch2::Catch2WithMain)
add_test(NAME test_diagnostics COMMAND test_diagnostics)
set_tests_properties(test_diagnostics PROPERTIES LABELS fast)

# ── GPU-keyed kernel budgets ────────────────────────────────────────────────
# Every suite whose budgets now come from perf/kernel_budgets.json SKIPs when
# no record exists for the running GPU, so all of them need exit code 4
# tolerated. LSX_KERNEL_BUDGET_WRITE=1 recalibrates instead of asserting.
set(LSX_KERNEL_BUDGET_TESTS
    test_gemma_moe_norm test_gemma_moe_moe test_glm_moe_flash_kernels
    test_qwen_moe_kernels test_qwen_nvfp4_gemv test_qwen_moe_exp_kernels
    test_nemotron_moe_kernels test_gigachat_moe_devkernels
    test_ornith_moe_kernels test_expert_store_stage)

foreach(_bt IN LISTS LSX_KERNEL_BUDGET_TESTS)
    set_tests_properties(${_bt} PROPERTIES SKIP_RETURN_CODE 4)
    # These measure latency/bandwidth budgets. A concurrent GPU workload makes
    # the sample a measurement of the box, not the kernel (OPEN_TASKS C1), so
    # ctest -j must serialize them behind one GPU lock.
    set_tests_properties(${_bt} PROPERTIES RESOURCE_LOCK "gpu")
endforeach()


# ── Hy-Memory-30B-A3B converter test (plain-RMSNorm GQA + sigmoid-router MoE) ──
add_executable(test_hy_moe_convert test_hy_moe_convert.cpp)
target_include_directories(test_hy_moe_convert PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_compile_definitions(test_hy_moe_convert PRIVATE
    LSX_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(test_hy_moe_convert PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Hy-SimulMT-30B-A3B converter test (same hy_v3 text backbone; the speech
#    encoder is stripped and counted) ──
add_executable(test_hy_moe_simulmt_convert test_hy_moe_simulmt_convert.cpp)
target_include_directories(test_hy_moe_simulmt_convert PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_compile_definitions(test_hy_moe_simulmt_convert PRIVATE
    LSX_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(test_hy_moe_simulmt_convert PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Hy-Memory-30B-A3B loader stage (pinned graph + resident expert slabs) ──
add_executable(test_hy_moe_loader test_hy_moe_loader.cpp)
target_include_directories(test_hy_moe_loader PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_compile_definitions(test_hy_moe_loader PRIVATE
    LSX_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(test_hy_moe_loader PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Hy-SimulMT-30B-A3B loader stage (shared loader, own passport) ──
add_executable(test_hy_moe_simulmt_loader test_hy_moe_simulmt_loader.cpp)
target_include_directories(test_hy_moe_simulmt_loader PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_compile_definitions(test_hy_moe_simulmt_loader PRIVATE
    LSX_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(test_hy_moe_simulmt_loader PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Hy-Memory-30B-A3B model stage (lifecycle + determinism) ──
add_executable(test_hy_moe_model test_hy_moe_model.cpp)
target_include_directories(test_hy_moe_model PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_compile_definitions(test_hy_moe_model PRIVATE
    LSX_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(test_hy_moe_model PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Hy-SimulMT-30B-A3B model stage (lifecycle + determinism) ──
add_executable(test_hy_moe_simulmt_model test_hy_moe_simulmt_model.cpp)
target_include_directories(test_hy_moe_simulmt_model PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_compile_definitions(test_hy_moe_simulmt_model PRIVATE
    LSX_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(test_hy_moe_simulmt_model PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Hy-Memory-30B-A3B numeric kernels (rope/router/accumulate/norm) ──
add_library(test_hy_cu OBJECT
    test_hy_moe_kernels.cu
)
set_target_properties(test_hy_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_hy_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(test_hy_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_include_directories(test_hy_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    )
target_link_libraries(test_hy_cu PRIVATE Catch2::Catch2WithMain
        lsx_dep_includes
    arrow_shared)

add_executable(test_hy_moe_kernels)
set_target_properties(test_hy_moe_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_hy_moe_kernels PRIVATE test_hy_cu lsxcommon
    Catch2::Catch2WithMain)

# ── Hy-Memory-30B-A3B native tool framing (host-only) ──
add_executable(test_hy_moe_tool_framing test_hy_moe_tool_framing.cpp)
target_compile_definitions(test_hy_moe_tool_framing PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_hy_moe_tool_framing PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Hy-Memory-30B-A3B tokenizer parity (real vault tree; SKIPs when absent) ──
# ── Flint pre-tokenizer split (host-only; no vault needed) ──
add_executable(test_flint_moe_pretokenizer test_flint_moe_pretokenizer.cpp)
target_link_libraries(test_flint_moe_pretokenizer PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Beam pre-tokenizer split (host-only; no vault needed) ──
add_executable(test_beam_tokenizer_split test_beam_tokenizer_split.cpp)
target_link_libraries(test_beam_tokenizer_split PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Ling selfcheck (real weights; SKIPs when the vault is absent) ──

# ── Ling FSI parity gate (declared golden debt) ──

# ── Ling CODE parity gate (declared golden debt) ──

# ── Ling DET parity gate (declared golden debt) ──

# ── Ling perf baseline (seeded through the harness) ──

# ── Ling native chat/tool codec (synthetic tokenizer fixture) ──
add_executable(test_ling_tool_framing test_ling_tool_framing.cpp)
target_compile_definitions(test_ling_tool_framing PRIVATE
    "TOKENIZER_TEST_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/data\"")
target_link_libraries(test_ling_tool_framing PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Ling o200k tokenizer parity (source tokenizer; SKIPs when absent) ──
add_executable(test_ling_tokenizer_parity test_ling_tokenizer_parity.cpp)
target_include_directories(test_ling_tokenizer_parity PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(test_ling_tokenizer_parity PRIVATE
    "LSX_LING_FIXTURE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/ling\"")
target_link_libraries(test_ling_tokenizer_parity PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Ling-3.1-flash model stage (synthetic converted tree; lifecycle) ──
add_executable(test_ling_moe_model test_ling_moe_model.cpp)
target_include_directories(test_ling_moe_model PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/lsxcommon/src)
target_compile_definitions(test_ling_moe_model PRIVATE
    "LSX_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"")
target_link_libraries(test_ling_moe_model PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Ling-3.1-flash numeric kernels (host-reference parity) ──
add_executable(test_ling_moe_kernels test_ling_moe_kernels.cpp)
target_include_directories(test_ling_moe_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/lsxcommon/src)
target_link_libraries(test_ling_moe_kernels PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Ling-3.1-flash loader (synthetic converted tree; no vault) ──
add_executable(test_ling_moe_loader test_ling_moe_loader.cpp)
target_include_directories(test_ling_moe_loader PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/lsxcommon/src)
target_compile_definitions(test_ling_moe_loader PRIVATE
    "LSX_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"")
target_link_libraries(test_ling_moe_loader PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Ling-3.1-flash converter (synthetic checkpoint; label convert) ──
add_executable(test_ling_moe_convert test_ling_moe_convert.cpp)
target_include_directories(test_ling_moe_convert PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/lsxcommon/src)
target_compile_definitions(test_ling_moe_convert PRIVATE
    "LSX_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"")
target_link_libraries(test_ling_moe_convert PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Flint converter (synthetic checkpoint; label convert, bounded TIMEOUT) ──
add_executable(test_flint_moe_convert test_flint_moe_convert.cpp)
target_include_directories(test_flint_moe_convert PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/lsxcommon/src)
target_compile_definitions(test_flint_moe_convert PRIVATE
    "LSX_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"")
target_link_libraries(test_flint_moe_convert PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Beam converter (synthetic checkpoint; label convert, bounded TIMEOUT) ──
add_executable(test_beam_moe_convert test_beam_moe_convert.cpp)
target_include_directories(test_beam_moe_convert PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/lsxcommon/src)
target_compile_definitions(test_beam_moe_convert PRIVATE
    "LSX_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"")
target_link_libraries(test_beam_moe_convert PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Beam kernels (CUDA object library + host-reference parity) ──
set(test_beam_cu_src
    test_beam_moe_kernels.cu
)
add_library(test_beam_cu OBJECT ${test_beam_cu_src})
set_target_properties(test_beam_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_beam_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(test_beam_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_include_directories(test_beam_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(test_beam_cu PRIVATE Catch2::Catch2WithMain)
add_executable(test_beam_moe_kernels)
set_target_properties(test_beam_moe_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_beam_moe_kernels PRIVATE test_beam_cu lsxcommon
    Catch2::Catch2WithMain)

# ── Beam loader stage (synthetic converted tree; no vault) ──
add_executable(test_beam_moe_loader test_beam_moe_loader.cpp)
target_include_directories(test_beam_moe_loader PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_link_libraries(test_beam_moe_loader PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Flint loader stage (synthetic converted tree; no vault) ──
add_executable(test_flint_moe_loader test_flint_moe_loader.cpp)
target_include_directories(test_flint_moe_loader PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_compile_definitions(test_flint_moe_loader PRIVATE
    LSX_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(test_flint_moe_loader PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Flint model stage (synthetic converted tree; lifecycle contract) ──
add_executable(test_flint_moe_model test_flint_moe_model.cpp)
target_include_directories(test_flint_moe_model PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_compile_definitions(test_flint_moe_model PRIVATE
    LSX_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(test_flint_moe_model PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Flint numeric kernels (sink softmax / window / router / rope / norm) ──
add_library(test_flint_cu OBJECT
    test_flint_moe_kernels.cu
)
set_target_properties(test_flint_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_flint_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(test_flint_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_include_directories(test_flint_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
    )
target_link_libraries(test_flint_cu PRIVATE Catch2::Catch2WithMain
        lsx_dep_includes
    arrow_shared)

add_executable(test_flint_moe_kernels)
set_target_properties(test_flint_moe_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_flint_moe_kernels PRIVATE test_flint_cu lsxcommon
    Catch2::Catch2WithMain)

# ── Flint SelfCheck (real weights; SKIPs when the vault is absent) ──

# ── Flint FSI parity gate ──

# ── Flint CODE parity gate ──

# ── Flint DET parity gate ──


# ── Beam model stage (synthetic converted tree) ──
add_executable(test_beam_moe_model test_beam_moe_model.cpp)
target_include_directories(test_beam_moe_model PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_compile_definitions(test_beam_moe_model PRIVATE
    LSX_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(test_beam_moe_model PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Beam selfcheck (real weights; SKIPs when the vault is absent) ──

# ── Beam FSI/CODE/DET parity gates (golden debt; SKIP when absent) ──


# ── Beam native tool framing (host-only; synthetic tokenizer) ──
add_executable(test_beam_tool_framing test_beam_tool_framing.cpp)
target_compile_definitions(test_beam_tool_framing PRIVATE
    LSX_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(test_beam_tool_framing PRIVATE lsxcommon
    Catch2::Catch2WithMain arrow_shared)

# ── Flint native tool framing (host-only; template-captured golden) ──
add_executable(test_flint_moe_tool_framing test_flint_moe_tool_framing.cpp)
target_compile_definitions(test_flint_moe_tool_framing PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
    "LSX_FLINT_FIXTURE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/flint\"")
target_link_libraries(test_flint_moe_tool_framing PRIVATE lsxcommon
    Catch2::Catch2WithMain arrow_shared)

# ── Flint tokenizer parity (source-captured golden; SKIPs without the vault) ──
add_executable(test_flint_moe_tokenizer_parity
    test_flint_moe_tokenizer_parity.cpp)
target_compile_definitions(test_flint_moe_tokenizer_parity PRIVATE
    "LSX_FLINT_FIXTURE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/flint\"")
target_link_libraries(test_flint_moe_tokenizer_parity PRIVATE lsxcommon
    Catch2::Catch2WithMain arrow_shared)

# ── Beam tokenizer parity (source-captured golden; SKIPs without the vault) ──
add_executable(test_beam_tokenizer_parity test_beam_tokenizer_parity.cpp)
target_compile_definitions(test_beam_tokenizer_parity PRIVATE
    "LSX_BEAM_FIXTURE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/beam\"")
target_link_libraries(test_beam_tokenizer_parity PRIVATE lsxcommon
    Catch2::Catch2WithMain arrow_shared)

add_executable(test_hy_moe_tokenizer_parity test_hy_moe_tokenizer_parity.cpp)
target_compile_definitions(test_hy_moe_tokenizer_parity PRIVATE
    "LSX_HY_FIXTURE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/hy\"")
target_link_libraries(test_hy_moe_tokenizer_parity PRIVATE lsxcommon
    Catch2::Catch2WithMain arrow_shared)

# ── Hy-Memory-30B-A3B SelfCheck (real weights; SKIPs when the vault is absent) ──

# ── Hy-Memory-30B-A3B FSI parity gate ──

# ── Hy-Memory-30B-A3B CODE parity gate ──

# ── Hy-Memory-30B-A3B DET determinism gate ──

# ── Hy-SimulMT-30B-A3B selfcheck + parity gates (same text backbone) ──


# ── Hy-Memory-30B-A3B perf baseline (canonical 2048-token protocol) ──

# ── Hy-SimulMT-30B-A3B perf baseline (canonical 2048-token protocol) ──


# ── Hy-Memory-30B-A3B chat framing parity (real tokenizer + template) ──
add_executable(test_hy_moe_chat_framing test_hy_moe_chat_framing.cpp)
target_compile_definitions(test_hy_moe_chat_framing PRIVATE
    "LSX_HY_FIXTURE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/hy\"")
target_link_libraries(test_hy_moe_chat_framing PRIVATE lsxcommon
    Catch2::Catch2WithMain arrow_shared)

# ── Hy-SimulMT chat-framing parity (shared renderer, SimulMT template) ──
add_executable(test_hy_moe_simulmt_tool_framing
    test_hy_moe_simulmt_tool_framing.cpp)
target_compile_definitions(test_hy_moe_simulmt_tool_framing PRIVATE
    "LSX_HY_SIMULMT_FIXTURE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/hy\"")
target_link_libraries(test_hy_moe_simulmt_tool_framing PRIVATE lsxcommon
    Catch2::Catch2WithMain arrow_shared)

# ── Step-5-Preview converter stage (frozen passport + fused FP8 split) ──
add_executable(test_step_moe_convert test_step_moe_convert.cpp)
target_include_directories(test_step_moe_convert PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_compile_definitions(test_step_moe_convert PRIVATE
    LSX_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_step_moe_convert PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Step-5-Preview loader stage (pinned graph + neutral ExpertStore) ──
add_executable(test_step_moe_loader test_step_moe_loader.cpp)
target_include_directories(test_step_moe_loader PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_compile_definitions(test_step_moe_loader PRIVATE
    LSX_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(test_step_moe_loader PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Step-5-Preview numeric kernels vs host references ──
add_library(test_step_cu OBJECT
    test_step_moe_kernels.cu
)
set_target_properties(test_step_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_step_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(test_step_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_include_directories(test_step_cu PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/include"
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src"
)
target_link_libraries(test_step_cu PRIVATE Catch2::Catch2WithMain
        lsx_dep_includes
    arrow_shared)

add_executable(test_step_moe_kernels)
set_target_properties(test_step_moe_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_step_moe_kernels PRIVATE test_step_cu lsxcommon
    Catch2::Catch2WithMain)

# ── Step-5-Preview model stage (lifecycle + determinism) ──
add_executable(test_step_moe_model test_step_moe_model.cpp)
target_include_directories(test_step_moe_model PRIVATE
    ${PROJECT_SOURCE_DIR}/libs/lsxcommon/src)
target_compile_definitions(test_step_moe_model PRIVATE
    LSX_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(test_step_moe_model PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Step-5-Preview native tool framing (host-only) ──
add_executable(test_step_moe_tool_framing test_step_moe_tool_framing.cpp)
target_compile_definitions(test_step_moe_tool_framing PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_step_moe_tool_framing PRIVATE lsxcommon
    Catch2::Catch2WithMain)

# ── Step-5-Preview selfcheck (Paris + determinism on the real vault tree) ──

# ── Step-5-Preview FSI/CODE/DET parity gates (goldens = declared debt) ──


# ── Step-5-Preview perf baseline (canonical 2048-token protocol) ──

# ── Step-5-Preview tokenizer parity (real checkpoint tokenizer; SKIPs when absent) ──
add_executable(test_step_moe_tokenizer test_step_moe_tokenizer.cpp)
target_link_libraries(test_step_moe_tokenizer PRIVATE lsxcommon
    Catch2::Catch2WithMain arrow_shared)

# ── Parity gates (FSI / CODE / DET) — one driver, one target per model ─────
# The 105 copied per-family files became one TU parameterised by two
# compile-time constants; the model row and the SKIP text come from its
# manifest. Build wiring is per target (it is per-target compile reality),
# so it is emitted here by hand once instead of 105 times.
add_executable(test_alice_moe_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_alice_moe_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_alice_moe_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_alice_moe_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"alice-foundation-80b-a3b\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_alice_moe_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_alice_moe_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_alice_moe_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_alice_moe_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"alice-foundation-80b-a3b\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_alice_moe_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_alice_moe_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_alice_moe_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_alice_moe_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"alice-foundation-80b-a3b\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_beam_moe_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_beam_moe_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_beam_moe_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_beam_moe_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"beam\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_beam_moe_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_beam_moe_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_beam_moe_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_beam_moe_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"beam\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_beam_moe_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_beam_moe_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_beam_moe_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_beam_moe_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"beam\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_deepseek_moe_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_deepseek_moe_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_deepseek_moe_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_deepseek_moe_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"deepseek-v4-flash\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_deepseek_moe_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_deepseek_moe_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_deepseek_moe_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_deepseek_moe_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"deepseek-v4-flash\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_deepseek_moe_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_deepseek_moe_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_deepseek_moe_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_deepseek_moe_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"deepseek-v4-flash\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_deepseek_moe_pro_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_deepseek_moe_pro_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_deepseek_moe_pro_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_deepseek_moe_pro_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"deepseek-v4-pro-0813\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_deepseek_moe_pro_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_deepseek_moe_pro_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_deepseek_moe_pro_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_deepseek_moe_pro_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"deepseek-v4-pro-0813\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_deepseek_moe_pro_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_deepseek_moe_pro_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_deepseek_moe_pro_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_deepseek_moe_pro_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"deepseek-v4-pro-0813\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_deepseek_moe_v41_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_deepseek_moe_v41_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_deepseek_moe_v41_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_deepseek_moe_v41_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"deepseek-v4.1-flash\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_deepseek_moe_v41_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_deepseek_moe_v41_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_deepseek_moe_v41_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_deepseek_moe_v41_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"deepseek-v4.1-flash\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_deepseek_moe_v41_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_deepseek_moe_v41_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_deepseek_moe_v41_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_deepseek_moe_v41_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"deepseek-v4.1-flash\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_flint_moe_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_flint_moe_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_flint_moe_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_flint_moe_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"flint\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_flint_moe_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_flint_moe_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_flint_moe_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_flint_moe_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"flint\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_flint_moe_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_flint_moe_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_flint_moe_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_flint_moe_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"flint\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_gemma_moe_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_gemma_moe_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_gemma_moe_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_gemma_moe_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"gemma-4-26b-a4b-it\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_gemma_moe_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_gemma_moe_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_gemma_moe_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_gemma_moe_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"gemma-4-26b-a4b-it\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_gemma_moe_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_gemma_moe_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_gemma_moe_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_gemma_moe_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"gemma-4-26b-a4b-it\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_gemma_dense_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_gemma_dense_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_gemma_dense_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_gemma_dense_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"gemma-4-31b-it\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_gemma_dense_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_gemma_dense_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_gemma_dense_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_gemma_dense_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"gemma-4-31b-it\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_gemma_dense_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_gemma_dense_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_gemma_dense_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_gemma_dense_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"gemma-4-31b-it\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_gigachat_moe_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_gigachat_moe_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_gigachat_moe_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_gigachat_moe_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"gigachat3.5-432b-a28b-reasoning\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_gigachat_moe_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_gigachat_moe_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_gigachat_moe_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_gigachat_moe_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"gigachat3.5-432b-a28b-reasoning\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_gigachat_moe_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_gigachat_moe_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_gigachat_moe_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_gigachat_moe_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"gigachat3.5-432b-a28b-reasoning\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_glm_moe_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_glm_moe_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_glm_moe_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_glm_moe_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"glm-5.2\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_glm_moe_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_glm_moe_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_glm_moe_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_glm_moe_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"glm-5.2\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_glm_moe_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_glm_moe_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_glm_moe_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_glm_moe_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"glm-5.2\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_glm_moe_flash_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_glm_moe_flash_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_glm_moe_flash_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_glm_moe_flash_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"glm-5.3-flash\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_glm_moe_flash_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_glm_moe_flash_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_glm_moe_flash_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_glm_moe_flash_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"glm-5.3-flash\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_glm_moe_flash_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_glm_moe_flash_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_glm_moe_flash_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_glm_moe_flash_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"glm-5.3-flash\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_granite_dense_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_granite_dense_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_granite_dense_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_granite_dense_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"granite-4.2-30b\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_granite_dense_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_granite_dense_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_granite_dense_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_granite_dense_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"granite-4.2-30b\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_granite_dense_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_granite_dense_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_granite_dense_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_granite_dense_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"granite-4.2-30b\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_granite_switch_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_granite_switch_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_granite_switch_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_granite_switch_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"granite-switch-4.1-30b\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_granite_switch_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_granite_switch_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_granite_switch_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_granite_switch_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"granite-switch-4.1-30b\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_granite_switch_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_granite_switch_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_granite_switch_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_granite_switch_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"granite-switch-4.1-30b\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
# The 8B is a sibling of the same architecture with its own frozen passport
# (40 layers, 12800 FFN, theta 1e7, residual 0.22). Same driver, own model in
# LSX_GATE_MODEL, own goldens captured from its own source checkpoint.
add_executable(test_granite_switch_8b_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_granite_switch_8b_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_granite_switch_8b_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_granite_switch_8b_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"granite-switch-4.1-8b\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_granite_switch_8b_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_granite_switch_8b_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_granite_switch_8b_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_granite_switch_8b_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"granite-switch-4.1-8b\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_granite_switch_8b_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_granite_switch_8b_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_granite_switch_8b_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_granite_switch_8b_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"granite-switch-4.1-8b\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_hy_moe_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_hy_moe_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_hy_moe_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_hy_moe_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"hy-memory-30b-a3b\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_hy_moe_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_hy_moe_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_hy_moe_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_hy_moe_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"hy-memory-30b-a3b\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_hy_moe_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_hy_moe_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_hy_moe_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_hy_moe_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"hy-memory-30b-a3b\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_hy_moe_simulmt_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_hy_moe_simulmt_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_hy_moe_simulmt_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_hy_moe_simulmt_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"hy-simulmt-30b-a3b\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_hy_moe_simulmt_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_hy_moe_simulmt_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_hy_moe_simulmt_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_hy_moe_simulmt_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"hy-simulmt-30b-a3b\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_hy_moe_simulmt_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_hy_moe_simulmt_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_hy_moe_simulmt_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_hy_moe_simulmt_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"hy-simulmt-30b-a3b\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_kimi_moe_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_kimi_moe_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_kimi_moe_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_kimi_moe_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"kimi-k2.7-code\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_kimi_moe_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_kimi_moe_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_kimi_moe_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_kimi_moe_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"kimi-k2.7-code\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_kimi_moe_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_kimi_moe_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_kimi_moe_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_kimi_moe_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"kimi-k2.7-code\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_kimi_pro_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_kimi_pro_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_kimi_pro_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_kimi_pro_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"kimi-k3.1-pro\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_kimi_pro_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_kimi_pro_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_kimi_pro_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_kimi_pro_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"kimi-k3.1-pro\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_kimi_pro_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_kimi_pro_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_kimi_pro_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_kimi_pro_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"kimi-k3.1-pro\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_kimi_k3_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_kimi_k3_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_kimi_k3_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_kimi_k3_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"kimi-k3\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_kimi_k3_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_kimi_k3_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_kimi_k3_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_kimi_k3_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"kimi-k3\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_kimi_k3_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_kimi_k3_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_kimi_k3_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_kimi_k3_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"kimi-k3\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_ling_moe_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_ling_moe_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_ling_moe_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_ling_moe_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"ling\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_ling_moe_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_ling_moe_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_ling_moe_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_ling_moe_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"ling\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_ling_moe_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_ling_moe_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_ling_moe_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_ling_moe_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"ling\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_mimo_moe_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_mimo_moe_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_mimo_moe_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_mimo_moe_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"mimo-v2.6-flash-rl\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_mimo_moe_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_mimo_moe_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_mimo_moe_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_mimo_moe_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"mimo-v2.6-flash-rl\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_mimo_moe_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_mimo_moe_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_mimo_moe_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_mimo_moe_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"mimo-v2.6-flash-rl\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_minimax_moe_flash_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_minimax_moe_flash_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_minimax_moe_flash_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_minimax_moe_flash_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"minimax-m3.1-flash\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_minimax_moe_flash_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_minimax_moe_flash_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_minimax_moe_flash_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_minimax_moe_flash_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"minimax-m3.1-flash\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_minimax_moe_flash_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_minimax_moe_flash_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_minimax_moe_flash_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_minimax_moe_flash_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"minimax-m3.1-flash\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_minimax_moe_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_minimax_moe_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_minimax_moe_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_minimax_moe_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"minimax-m3\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_minimax_moe_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_minimax_moe_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_minimax_moe_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_minimax_moe_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"minimax-m3\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_minimax_moe_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_minimax_moe_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_minimax_moe_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_minimax_moe_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"minimax-m3\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_mistral_moe_large4_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_mistral_moe_large4_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_mistral_moe_large4_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_mistral_moe_large4_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"mistral-large-4-1t\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_mistral_moe_large4_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_mistral_moe_large4_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_mistral_moe_large4_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_mistral_moe_large4_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"mistral-large-4-1t\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_mistral_moe_large4_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_mistral_moe_large4_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_mistral_moe_large4_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_mistral_moe_large4_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"mistral-large-4-1t\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_mistral_moe_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_mistral_moe_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_mistral_moe_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_mistral_moe_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"mistral-small-4-119b\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_mistral_moe_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_mistral_moe_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_mistral_moe_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_mistral_moe_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"mistral-small-4-119b\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_mistral_moe_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_mistral_moe_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_mistral_moe_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_mistral_moe_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"mistral-small-4-119b\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_nemotron_moe_nvfp4_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_nemotron_moe_nvfp4_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_nemotron_moe_nvfp4_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_nemotron_moe_nvfp4_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"nemotron-3.5-lightning-30b-a3b-nvfp4\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_nemotron_moe_nvfp4_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_nemotron_moe_nvfp4_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_nemotron_moe_nvfp4_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_nemotron_moe_nvfp4_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"nemotron-3.5-lightning-30b-a3b-nvfp4\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_nemotron_moe_nvfp4_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_nemotron_moe_nvfp4_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_nemotron_moe_nvfp4_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_nemotron_moe_nvfp4_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"nemotron-3.5-lightning-30b-a3b-nvfp4\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_nemotron_moe_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_nemotron_moe_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_nemotron_moe_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_nemotron_moe_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"nemotron-3.5-lightning-30b-a3b\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_nemotron_moe_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_nemotron_moe_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_nemotron_moe_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_nemotron_moe_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"nemotron-3.5-lightning-30b-a3b\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_nemotron_moe_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_nemotron_moe_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_nemotron_moe_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_nemotron_moe_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"nemotron-3.5-lightning-30b-a3b\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_ornith_moe_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_ornith_moe_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_ornith_moe_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_ornith_moe_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"ornith-1.5-35b-a3b\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_ornith_moe_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_ornith_moe_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_ornith_moe_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_ornith_moe_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"ornith-1.5-35b-a3b\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_ornith_moe_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_ornith_moe_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_ornith_moe_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_ornith_moe_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"ornith-1.5-35b-a3b\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_phi_dense_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_phi_dense_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_phi_dense_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_phi_dense_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"phi-4\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_phi_dense_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_phi_dense_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_phi_dense_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_phi_dense_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"phi-4\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_phi_dense_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_phi_dense_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_phi_dense_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_phi_dense_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"phi-4\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_dory_dense_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_dory_dense_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_dory_dense_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_dory_dense_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"dory-1.0-4b\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_dory_dense_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_dory_dense_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_dory_dense_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_dory_dense_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"dory-1.0-4b\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_dory_dense_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_dory_dense_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_dory_dense_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_dory_dense_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"dory-1.0-4b\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_qwen_moe_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_qwen_moe_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_qwen_moe_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_qwen_moe_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"qwen3.5-35b-a3b\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_qwen_moe_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_qwen_moe_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_qwen_moe_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_qwen_moe_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"qwen3.5-35b-a3b\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_qwen_moe_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_qwen_moe_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_qwen_moe_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_qwen_moe_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"qwen3.5-35b-a3b\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_qwen_dense_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_qwen_dense_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_qwen_dense_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_qwen_dense_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"qwen3.5-9b\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_qwen_dense_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_qwen_dense_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_qwen_dense_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_qwen_dense_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"qwen3.5-9b\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_qwen_dense_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_qwen_dense_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_qwen_dense_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_qwen_dense_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"qwen3.5-9b\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_qwen_moe_36_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_qwen_moe_36_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_qwen_moe_36_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_qwen_moe_36_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"qwen3.6-35b-a3b\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_qwen_moe_36_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_qwen_moe_36_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_qwen_moe_36_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_qwen_moe_36_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"qwen3.6-35b-a3b\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_qwen_moe_36_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_qwen_moe_36_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_qwen_moe_36_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_qwen_moe_36_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"qwen3.6-35b-a3b\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_qwen_dense_27b_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_qwen_dense_27b_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_qwen_dense_27b_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_qwen_dense_27b_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"qwen3.8-27b\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_qwen_dense_27b_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_qwen_dense_27b_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_qwen_dense_27b_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_qwen_dense_27b_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"qwen3.8-27b\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_qwen_dense_27b_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_qwen_dense_27b_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_qwen_dense_27b_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_qwen_dense_27b_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"qwen3.8-27b\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_qwen_moe_exp_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_qwen_moe_exp_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_qwen_moe_exp_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_qwen_moe_exp_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"qwen3.8-flash-next\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_qwen_moe_exp_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_qwen_moe_exp_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_qwen_moe_exp_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_qwen_moe_exp_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"qwen3.8-flash-next\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_qwen_moe_exp_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_qwen_moe_exp_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_qwen_moe_exp_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_qwen_moe_exp_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"qwen3.8-flash-next\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_step_moe_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_step_moe_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_step_moe_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_step_moe_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"step-5-preview\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_step_moe_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_step_moe_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_step_moe_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_step_moe_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"step-5-preview\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_step_moe_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_step_moe_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_step_moe_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_step_moe_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"step-5-preview\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")
add_executable(test_xing_moe_fsi_selfcheck test_parity_gate.cpp)
target_include_directories(test_xing_moe_fsi_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_xing_moe_fsi_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_xing_moe_fsi_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"xing4.0-29b-a4b\"" "LSX_GATE_KIND=\"fsi\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fsi\"")
add_executable(test_xing_moe_code_selfcheck test_parity_gate.cpp)
target_include_directories(test_xing_moe_code_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_xing_moe_code_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_xing_moe_code_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"xing4.0-29b-a4b\"" "LSX_GATE_KIND=\"code\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/code\"")
add_executable(test_xing_moe_det_selfcheck test_parity_gate.cpp)
target_include_directories(test_xing_moe_det_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_xing_moe_det_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_xing_moe_det_selfcheck PRIVATE
    "LSX_GATE_MODEL=\"xing4.0-29b-a4b\"" "LSX_GATE_KIND=\"det\""
    "LSX_GOLDEN_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/det\"")


# ── SelfCheck (the "Paris" gate) — one driver, declared per-model hooks ──
# paris_cycles always; tool_roundtrip / auto_path_determinism come from the
# manifest row. A family case that is not a hook keeps its own file.
add_executable(test_alice_moe_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_alice_moe_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_alice_moe_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_alice_moe_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"alice-foundation-80b-a3b\"")
add_executable(test_beam_moe_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_beam_moe_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_beam_moe_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_beam_moe_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"beam\"")
add_executable(test_deepseek_moe_pro_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_deepseek_moe_pro_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_deepseek_moe_pro_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_deepseek_moe_pro_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"deepseek-v4-pro-0813\"")
add_executable(test_deepseek_moe_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_deepseek_moe_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_deepseek_moe_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_deepseek_moe_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"deepseek-v4-flash\"")
add_executable(test_deepseek_moe_v41_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_deepseek_moe_v41_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_deepseek_moe_v41_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_deepseek_moe_v41_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"deepseek-v4.1-flash\"")
add_executable(test_flint_moe_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_flint_moe_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_flint_moe_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_flint_moe_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"flint\"")
add_executable(test_gemma_dense_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_gemma_dense_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_gemma_dense_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_gemma_dense_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"gemma-4-31b-it\"")
add_executable(test_gemma_moe_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_gemma_moe_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_gemma_moe_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_gemma_moe_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"gemma-4-26b-a4b-it\"")
add_executable(test_gigachat_moe_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_gigachat_moe_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_gigachat_moe_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_gigachat_moe_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"gigachat3.5-432b-a28b-reasoning\"")
add_executable(test_glm_moe_flash_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_glm_moe_flash_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_glm_moe_flash_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_glm_moe_flash_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"glm-5.3-flash\"")
add_executable(test_glm_moe_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_glm_moe_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_glm_moe_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_glm_moe_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"glm-5.2\"")
add_executable(test_granite_dense_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_granite_dense_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_granite_dense_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_granite_dense_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"granite-4.2-30b\"")
add_executable(test_granite_switch_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_granite_switch_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_granite_switch_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_granite_switch_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"granite-switch-4.1-30b\"")
# Same source, the 8B checkpoint and passport (see LSX_GSW_MODEL in the file).
add_executable(test_granite_switch_8b_model test_granite_switch_model.cpp)
target_include_directories(test_granite_switch_8b_model PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_granite_switch_8b_model PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_granite_switch_8b_model PRIVATE
    # Same absolute fixture dir as the 30B: the sibling's real-weight tool
    # round-trip was SKIPping for the identical reason (see the 30B above).
    LSX_GRANITE_SWITCH_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/granite_switch"
    "LSX_GSW_MODEL=\"granite-switch-4.1-8b\""
    "LSX_GSW_ROOT_ENV=\"LSX_TEST_GRANITE_SWITCH_4_1_8B_ROOT\""
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
add_executable(test_granite_switch_8b_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_granite_switch_8b_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_granite_switch_8b_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_granite_switch_8b_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"granite-switch-4.1-8b\"")
add_executable(test_hy_moe_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_hy_moe_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_hy_moe_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_hy_moe_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"hy-memory-30b-a3b\"")
add_executable(test_hy_moe_simulmt_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_hy_moe_simulmt_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_hy_moe_simulmt_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_hy_moe_simulmt_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"hy-simulmt-30b-a3b\"")
add_executable(test_kimi_k3_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_kimi_k3_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_kimi_k3_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_kimi_k3_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"kimi-k3\"")
add_executable(test_kimi_moe_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_kimi_moe_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_kimi_moe_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_kimi_moe_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"kimi-k2.7-code\"")
add_executable(test_kimi_pro_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_kimi_pro_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_kimi_pro_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_kimi_pro_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"kimi-k3.1-pro\"")
add_executable(test_ling_moe_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_ling_moe_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_ling_moe_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_ling_moe_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"ling\"")
add_executable(test_mimo_moe_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_mimo_moe_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_mimo_moe_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_mimo_moe_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"mimo-v2.6-flash-rl\"")
add_executable(test_minimax_moe_flash_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_minimax_moe_flash_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_minimax_moe_flash_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_minimax_moe_flash_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"minimax-m3.1-flash\"")
add_executable(test_minimax_moe_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_minimax_moe_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_minimax_moe_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_minimax_moe_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"minimax-m3\"")
add_executable(test_mistral_moe_large4_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_mistral_moe_large4_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_mistral_moe_large4_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_mistral_moe_large4_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"mistral-large-4-1t\"")
add_executable(test_mistral_moe_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_mistral_moe_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_mistral_moe_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_mistral_moe_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"mistral-small-4-119b\"")
add_executable(test_nemotron_moe_nvfp4_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_nemotron_moe_nvfp4_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_nemotron_moe_nvfp4_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_nemotron_moe_nvfp4_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"nemotron-3.5-lightning-30b-a3b-nvfp4\"")
add_executable(test_nemotron_moe_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_nemotron_moe_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_nemotron_moe_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_nemotron_moe_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"nemotron-3.5-lightning-30b-a3b\"")
add_executable(test_ornith_moe_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_ornith_moe_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_ornith_moe_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_ornith_moe_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"ornith-1.5-35b-a3b\"")
add_executable(test_phi_dense_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_phi_dense_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_phi_dense_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_phi_dense_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"phi-4\"")
add_executable(test_dory_dense_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_dory_dense_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_dory_dense_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_dory_dense_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"dory-1.0-4b\"")
add_executable(test_qwen_dense_27b_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_qwen_dense_27b_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_qwen_dense_27b_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_qwen_dense_27b_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"qwen3.8-27b\"")
add_executable(test_qwen_dense_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_qwen_dense_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_qwen_dense_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_qwen_dense_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"qwen3.5-9b\"")
add_executable(test_qwen_moe_36_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_qwen_moe_36_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_qwen_moe_36_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_qwen_moe_36_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"qwen3.6-35b-a3b\"")
add_executable(test_qwen_moe_exp_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_qwen_moe_exp_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_qwen_moe_exp_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_qwen_moe_exp_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"qwen3.8-flash-next\"")
add_executable(test_qwen_moe_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_qwen_moe_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_qwen_moe_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_qwen_moe_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"qwen3.5-35b-a3b\"")
add_executable(test_step_moe_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_step_moe_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_step_moe_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_step_moe_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"step-5-preview\"")
add_executable(test_xing_moe_selfcheck test_selfcheck_driver.cpp)
target_include_directories(test_xing_moe_selfcheck PRIVATE "${LSX_MANIFEST_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_xing_moe_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
target_compile_definitions(test_xing_moe_selfcheck PRIVATE "LSX_SELFCHECK_MODEL=\"xing4.0-29b-a4b\"")

# ── Model manifest generator schema self-test (host-only, label fast) ─────
# Mutates a live manifest and asserts each of the seven schema rules is
# reported. Without it a validator that silently stops rejecting anything would
# stay green.
add_test(NAME test_manifest_generator_selftest
         COMMAND "${LSX_PYTHON3}" "${PROJECT_SOURCE_DIR}/tools/gen_model_manifest.py"
                 --self-test --root "${PROJECT_SOURCE_DIR}")
set_tests_properties(test_manifest_generator_selftest PROPERTIES LABELS fast)
set_tests_properties(test_manifest_generator_selftest PROPERTIES TIMEOUT 120)

# ── Model manifest sync (the generated registries cannot drift) ───────────
# models/<model_type>.json is the single source of truth; this target proves the
# compiled table, the committed catalogue and the test-file ownership all agree
# with it. It regenerates docs/MODELS.md in memory and fails on any difference.
lsx_add_test(NAME test_manifest_sync
    SOURCES test_manifest_sync.cpp
    LINK lsxcommon
    INCLUDE "${LSX_MANIFEST_GENERATED_DIR}"
    DEFINE "LSX_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\""
           "LSX_GENERATED_DIR=\"${LSX_MANIFEST_GENERATED_DIR}\""
           "LSX_PYTHON3=\"${LSX_PYTHON3}\""
    LABEL fast)
