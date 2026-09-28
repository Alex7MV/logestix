find_package(CUDAToolkit REQUIRED)
set(CUDA_INCLUDE_DIR     "${CUDAToolkit_INCLUDE_DIRS}")
set(CUDA_CUDART_LIBRARY  "${CUDAToolkit_CUDART_LIBRARY}")
add_library(lsxcommon STATIC
    src/converter.cpp
    src/core.cpp
    src/common.cpp
    src/deepseek_moe_model.cpp
    src/deepseek_moe_v41_convert.cpp
    src/deepseek_moe_v41_load.cpp
    src/deepseek_moe_v41_model.cpp
    src/generation_cli.cpp
    src/glm_moe_convert.cpp
    src/glm_moe_flash_convert.cpp
    src/glm_moe_flash_fp8.cpp
    src/glm_moe_flash_load.cpp
    src/glm_moe_flash_model.cpp
    src/kimi_moe_convert.cpp
    src/kimi_moe_load.cpp
    src/kimi_moe_model.cpp
    src/glm_moe_load.cpp
    src/glm_moe_model.cpp
    src/gemma_moe_convert.cpp
    src/gemma_moe_load.cpp
    src/granite_dense_convert.cpp
    src/granite_dense_load.cpp
    src/granite_dense_model.cpp
    src/gemma_base_model.cpp
    src/gemma_moe_model.cpp
    src/nemotron_moe_convert.cpp
    src/nemotron_moe_load.cpp
    src/nemotron_moe_expert_cache.cpp
    src/nemotron_moe_model.cpp
    src/gigachat_moe_convert.cpp
    src/gigachat_moe_load.cpp
    src/gigachat_moe_model.cpp
    src/gigachat_moe_expert_cache.cpp
    src/lifecycle_strategy.cpp
    src/log.cpp
    src/model_base.cpp
    src/model_engine.cpp
    src/model_factory.cpp
    src/model_integrity.cpp
    src/model_registry.cpp
    src/model_runtime.cpp
    src/incremental_detokenizer.cpp
    src/tool_format.cpp
    src/qwen_tool_format.cpp
    src/deepseek_moe_tool_format.cpp
    src/gemma_tool_format.cpp
    src/fsi_selfcheck.cpp
    src/code_selfcheck.cpp
    src/det_selfcheck.cpp
    src/path_registry.cpp
    src/prompt_framing.cpp
    src/models/lsx_orchestrator.cpp
    src/pretokenizer.cpp
    src/qwen_dense_27b_model.cpp
    src/qwen_moe_model.cpp
    src/qwen_moe_36_model.cpp
    src/ornith_moe_model.cpp
    src/qwen_dense_convert.cpp
    src/qwen_moe_convert.cpp
    src/qwen_moe_36_convert.cpp
    src/ornith_moe_convert.cpp
    src/qwen_moe_exp_convert.cpp
    src/qwen_moe_exp_load.cpp
    src/qwen_moe_exp_model.cpp
    src/qwen_moe_load.cpp
    src/ornith_moe_load.cpp
    src/qwen_inference_load.cpp
    src/qwen_dense_model.cpp
    src/qwen_base_model.cpp
    src/gpu_perf.cpp
    src/gpu_power.cpp
    src/perf_baseline.cpp
    src/resource_governor.cpp
    src/runtime_tuner.cpp
    src/schema.cpp
    src/tokenizer.cpp
    src/tokenizer_arrows.cpp
    src/prompt_bridge.cpp
    src/tokenizer_convert.cpp
    src/unicode.cpp
    src/vram_estimate.cpp
)
set_target_properties(lsxcommon PROPERTIES
    CUDA_ARCHITECTURES "120-real;121-real"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)

enable_language(CUDA)

# Opt-in per-kernel resource report (registers / smem / local memory) printed
# by nvcc at compile time. Diagnostic only — zero effect on generated code.
option(LSX_NVCC_RES_USAGE
    "Pass -res-usage to nvcc (print register/smem usage per kernel)" OFF)

# Shared CUDA-target plumbing for the two OBJECT libraries below (NVCC include
# paths + CCCL/MSVC compile definitions). lsxcommon_cu carries the general
# kernels (NVFP4 quantizer, benchmark, DeepSeek inference); lsxcommon_glm_cu
# carries ONLY the GLM-5.2 bit-exact parity kernels, which must compile with
# -fmad=false to reproduce the non-contracted IEEE sequences of the -O0 host
# reference (see GlmExpfGlibc / fp64 GEMM / sequential RMSNorm). Scoping the
# flag to the GLM object library keeps the quantizer/benchmark/DeepSeek kernels
# on their default FMA contraction (their numeric contracts do not depend on
# -fmad=false and their performance must not regress).
set(LSXCOMMON_CU_INCLUDES
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_BINARY_DIR}/include"
    "${CUDA_INCLUDE_DIR}"
    "${PROJECT_SOURCE_DIR}/third_party/cccl/thrust"
    "${PROJECT_SOURCE_DIR}/third_party/cccl/cub"
    "${PROJECT_SOURCE_DIR}/third_party/cccl/libcudacxx/include"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
    "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include"
    "${PROJECT_SOURCE_DIR}/third_party/simdjson/singleheader"
)
set(LSXCOMMON_CU_DEFS
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
    FMT_UNICODE=0
)

add_library(lsxcommon_cu OBJECT
    src/lsx_inference_nvfp4.cu
    src/lsx_gemv.cu
    src/perf_calibration.cu
    src/safetensors_gpu.cu
    src/deepseek_moe_load.cu
    src/deepseek_moe_repack.cpp
    src/deepseek_moe_inference_moe.cu
    src/deepseek_moe_inference_head.cu
    src/deepseek_moe_inference_attn.cu
    src/deepseek_moe_inference_csa.cu
    src/deepseek_moe_inference_pipeline.cu
    src/bf16_convert.cu
    src/deepseek_moe_norm_kernels.cu
    src/shared_experts.cu
    src/sampling.cu
)
set_target_properties(lsxcommon_cu PROPERTIES
    CUDA_ARCHITECTURES "120-real;121-real"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${LSX_NVCC_RES_USAGE}>>:-res-usage>
)
target_include_directories(lsxcommon_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_cu
    "${CUDA_CUDART_LIBRARY}"
    arrow_shared
    arrow_cuda_shared
    absl::strings
)

add_library(lsxcommon_glm_cu OBJECT
    src/glm_moe_inference.cu
    src/glm_moe_load.cu
    src/glm_moe_flash_inference.cu
)
set_target_properties(lsxcommon_glm_cu PROPERTIES
    CUDA_ARCHITECTURES "120-real;121-real"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_glm_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${LSX_NVCC_RES_USAGE}>>:-res-usage>
)
target_include_directories(lsxcommon_glm_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_glm_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_glm_cu
    "${CUDA_CUDART_LIBRARY}"
    arrow_shared
    arrow_cuda_shared
    absl::strings
)
# FMA policy: object-lib -fmad flags are historical (qwen_moe/glm/kimi off,
# qwen dense on). Shared kernels in lsx_kernels_*.cuh use explicit *_rn
# intrinsics and are therefore flag-independent; DO NOT rely on contraction
# in any new kernel.
add_library(lsxcommon_qwen_cu OBJECT
    src/qwen_inference.cu
)
set_target_properties(lsxcommon_qwen_cu PROPERTIES
    CUDA_ARCHITECTURES "120-real;121-real"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_qwen_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${LSX_NVCC_RES_USAGE}>>:-res-usage>
)
target_include_directories(lsxcommon_qwen_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_qwen_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_qwen_cu
    "${CUDA_CUDART_LIBRARY}"
    arrow_shared
    arrow_cuda_shared
    absl::strings
)
add_library(lsxcommon_qwen_moe_cu OBJECT
    src/qwen_moe_inference.cu
)
set_target_properties(lsxcommon_qwen_moe_cu PROPERTIES
    CUDA_ARCHITECTURES "120-real;121-real"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
# -fmad=false: bit-exact router/expert math (protocol invariant).
target_compile_options(lsxcommon_qwen_moe_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(lsxcommon_qwen_moe_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_qwen_moe_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_qwen_moe_cu
    "${CUDA_CUDART_LIBRARY}"
    arrow_shared
    arrow_cuda_shared
    absl::strings
)

# Tensor-core NVFP4 fast path (audit Phase 3, F-01): kind::mxf4nvf4 PTX
# assembles ONLY on the arch-'a' targets, so this object lib carries its own
# --generate-code set (sm_120a / sm_121a). Plain sm_120/121 SASS for the
# scalar path is unaffected.
add_library(lsxcommon_tc_cu OBJECT
    src/lsx_tc_gemm.cu
)
set_target_properties(lsxcommon_tc_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_tc_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr;
    -lineinfo;
    --generate-code=arch=compute_120a,code=sm_120a;
    --generate-code=arch=compute_121a,code=sm_121a>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${LSX_NVCC_RES_USAGE}>>:-res-usage>
)
target_include_directories(lsxcommon_tc_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_tc_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_tc_cu
    "${CUDA_CUDART_LIBRARY}"
    "${CUDA_DRIVER_LIBRARY}"
    absl::strings
)

add_library(lsxcommon_kimi_cu OBJECT
    src/kimi_moe_inference.cu
    src/kimi_moe_expert_cache.cu
)
set_target_properties(lsxcommon_kimi_cu PROPERTIES
    CUDA_ARCHITECTURES "120-real;121-real"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_kimi_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(lsxcommon_kimi_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_kimi_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_kimi_cu
    "${CUDA_CUDART_LIBRARY}"
    arrow_shared
    arrow_cuda_shared
    absl::strings
)

add_library(lsxcommon_gemma_cu OBJECT
    src/gemma_moe_inference.cu
)
set_target_properties(lsxcommon_gemma_cu PROPERTIES
    CUDA_ARCHITECTURES "120-real;121-real"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_gemma_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(lsxcommon_gemma_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_gemma_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_gemma_cu
    "${CUDA_CUDART_LIBRARY}"
    arrow_shared
    arrow_cuda_shared
    absl::strings
)
# Ornith-1.5-35B-A3B MoE kernels (duplicated qwen_moe layer).
# -fmad=false: bit-exact router/expert math (protocol invariant).
add_library(lsxcommon_ornith_cu OBJECT
    src/ornith_moe_inference.cu
)
set_target_properties(lsxcommon_ornith_cu PROPERTIES
    CUDA_ARCHITECTURES "120-real;121-real"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_ornith_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(lsxcommon_ornith_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_ornith_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_ornith_cu
    "${CUDA_CUDART_LIBRARY}"
    arrow_shared
    arrow_cuda_shared
    absl::strings
)

# Granite-4.2-30B dense runtime: family-local RoPE + causal GQA attention,
# shared lsx_kernels_*.cuh primitives, neutral NVFP4 GEMM. No -fmad=false is
# required (all shared arithmetic uses explicit *_rn intrinsics).
add_library(lsxcommon_granite_cu OBJECT
    src/granite_dense_inference.cu
)
set_target_properties(lsxcommon_granite_cu PROPERTIES
    CUDA_ARCHITECTURES "120-real;121-real"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_granite_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${LSX_NVCC_RES_USAGE}>>:-res-usage>
)
target_include_directories(lsxcommon_granite_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_granite_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_granite_cu
    "${CUDA_CUDART_LIBRARY}"
    arrow_shared
    arrow_cuda_shared
    absl::strings
)

target_link_libraries(lsxcommon PUBLIC lsxcommon_cu lsxcommon_glm_cu
    lsxcommon_qwen_cu lsxcommon_qwen_moe_cu lsxcommon_tc_cu
    lsxcommon_kimi_cu lsxcommon_gemma_cu lsxcommon_qwen_moe_exp_cu
    lsxcommon_deepseek_v41_cu
    lsxcommon_nemotron_cu
    lsxcommon_gigachat_cu
    lsxcommon_ornith_cu
    lsxcommon_granite_cu
    lsx_tune)

# Nemotron-3.5 (NemotronH hybrid) Mamba-2 / gated-RMSNorm kernels.
# -fmad=false: bit-exact against the host reference (protocol invariant).
add_library(lsxcommon_nemotron_cu OBJECT
    src/nemotron_moe_kernels.cu
    src/nemotron_moe_inference_attn.cu
    src/nemotron_moe_inference_pipeline.cu
)
set_target_properties(lsxcommon_nemotron_cu PROPERTIES
    CUDA_ARCHITECTURES "120-real;121-real"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_nemotron_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(lsxcommon_nemotron_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_nemotron_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_nemotron_cu
    "${CUDA_CUDART_LIBRARY}"
    absl::strings
)

# GigaChat3.5 (hybrid MLA+GatedDeltaNet+MoE) kernels. Plan 1 ships the FP8
# block-128 -> BF16 dequant; Plan 2 adds the inference kernels.
add_library(lsxcommon_gigachat_cu OBJECT
    src/gigachat_moe_dequant.cu
    src/gigachat_moe_norm_kernels.cu
    src/gigachat_moe_kernels.cu
    src/gigachat_moe_inference_attn.cu
    src/gigachat_moe_inference_pipeline.cu
)
set_target_properties(lsxcommon_gigachat_cu PROPERTIES
    CUDA_ARCHITECTURES "120-real;121-real"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_gigachat_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(lsxcommon_gigachat_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_gigachat_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_gigachat_cu
    "${CUDA_CUDART_LIBRARY}"
    absl::strings
)

# qwen4_exp (Qwen3.8-Flash-Next) bit-exact kernels: hc GatedResidual, QSA
# indexer, PLE inject. -fmad=false contract (protocol invariant).
add_library(lsxcommon_qwen_moe_exp_cu OBJECT
    src/qwen_moe_exp_inference.cu
)
set_target_properties(lsxcommon_qwen_moe_exp_cu PROPERTIES
    CUDA_ARCHITECTURES "120-real;121-real"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_qwen_moe_exp_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
    # GCC LTO value-propagation miscompiles the multi-exit PLE inject wrapper
    # into an early-return clone that skips its kernel launch (observed
    # 2026-09-03: cudaLaunchKernel silently skipped on the AOCC/znver4 build).
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<PLATFORM_ID:Windows>>>:-Xcompiler=-fno-inline>
)
target_include_directories(lsxcommon_qwen_moe_exp_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_qwen_moe_exp_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_qwen_moe_exp_cu
    "${CUDA_CUDART_LIBRARY}"
    arrow_shared
    absl::strings
)

# DeepSeek-V4.1-Flash load-time dense dequant (F8_E4M3 x UE8M0 32x32 -> FP16)
# and BF16->FP16 upload. -fmad=false keeps the load path bit-exact.
add_library(lsxcommon_deepseek_v41_cu OBJECT
    src/deepseek_moe_v41_dequant.cu
    src/deepseek_moe_v41_load.cu
    src/deepseek_moe_v41_inference.cu
    src/deepseek_moe_v41_inference_attn.cu
    src/deepseek_moe_v41_inference_pipeline.cu
)
set_target_properties(lsxcommon_deepseek_v41_cu PROPERTIES
    CUDA_ARCHITECTURES "120-real;121-real"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_deepseek_v41_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(lsxcommon_deepseek_v41_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_deepseek_v41_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_deepseek_v41_cu
    "${CUDA_CUDART_LIBRARY}"
    arrow_shared
    absl::strings
)

target_compile_definitions(lsxcommon PUBLIC
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
)

target_include_directories(lsxcommon
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
        "$<BUILD_INTERFACE:${CUDA_INCLUDE_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/cccl/thrust>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/cccl/cub>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/cccl/libcudacxx/include>"
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src>"
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/_deps/arrow-build/src>"
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/lsxcommon/version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/lsxcommon/version.h"
    @ONLY
)

target_link_libraries(lsxcommon PUBLIC
    spdlog::spdlog
    flatbuffers
    libzmq-static
    "${CUDA_CUDART_LIBRARY}"
    absl::flat_hash_map
    absl::span
    absl::strings
)
target_link_libraries(lsxcommon PRIVATE
    arrow_shared
    arrow_cuda_shared
    simdjson::simdjson
    ${CMAKE_DL_LIBS}
)
# lsx_orchestrator.h lives in the PUBLIC include tree and exposes
# lsxnetwork::FrameQueue in ServeTaskMosaic's signature -> the dependency is
# part of lsxcommon's public interface, so it must be PUBLIC (not PRIVATE) or
# consumers including the orchestrator header lack lsxnetwork's include dirs.
target_link_libraries(lsxcommon PUBLIC lsxnetwork)
