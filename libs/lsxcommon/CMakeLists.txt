find_package(CUDAToolkit REQUIRED)
set(CUDA_INCLUDE_DIR     "${CUDAToolkit_INCLUDE_DIRS}")
set(CUDA_CUDART_LIBRARY  "${CUDAToolkit_CUDART_LIBRARY}")
add_library(lsxcommon STATIC
    src/converter.cpp
    src/core.cpp
    src/common.cpp
    src/deepseek_moe_model.cpp
    src/deepseek_moe_pro_convert.cpp
    src/deepseek_moe_pro_model.cpp
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
    src/kimi_moe_tool_format.cpp
    src/kimi_moe_load.cpp
    src/kimi_moe_model.cpp
    src/glm_moe_load.cpp
    src/glm_moe_model.cpp
    src/gemma_moe_convert.cpp
    src/gemma_moe_load.cpp
    src/gemma_dense_convert.cpp
    src/gemma_dense_load.cpp
    src/granite_dense_convert.cpp
    src/granite_dense_load.cpp
    src/granite_dense_model.cpp
    src/granite_dense_switch_convert.cpp
    src/granite_dense_switch_load.cpp
    src/granite_dense_switch_model.cpp
    src/granite_tool_format.cpp
    src/granite_switch_tool_format.cpp
    src/phi_dense_convert.cpp
    src/phi_dense_load.cpp
    src/phi_dense_model.cpp
    src/dory_dense_convert.cpp
    src/dory_dense_load.cpp
    src/dory_dense_model.cpp
    src/dory_tool_format.cpp
    src/gemma_base_model.cpp
    src/gemma_moe_model.cpp
    src/gemma_dense_model.cpp
    src/nemotron_moe_convert.cpp
    src/nemotron_moe_nvfp4_convert.cpp
    src/nemotron_moe_nvfp4_load.cpp
    src/nemotron_moe_load.cpp
    src/nemotron_moe_expert_cache.cpp
    src/nemotron_moe_model.cpp
    src/nemotron_moe_nvfp4_model.cpp
    src/nemotron_moe_ultra_convert.cpp
    src/gigachat_moe_convert.cpp
    src/gigachat_moe_load.cpp
    src/gigachat_moe_model.cpp
    src/gigachat_moe_expert_cache.cpp
    src/mimo_moe_convert.cpp
    src/mimo_moe_load.cpp
    src/mimo_moe_expert_cache.cpp
    src/mimo_moe_model.cpp
    src/alice_moe_convert.cpp
    src/kimi_moe_k3_convert.cpp
    src/kimi_moe_k3_load.cpp
    src/kimi_moe_k3_model.cpp
    src/kimi_moe_k3_tool_format.cpp
    src/kimi_moe_pro_convert.cpp
    src/kimi_moe_pro_model.cpp
    src/alice_moe_load.cpp
    src/alice_moe_model.cpp
    src/alice_tool_format.cpp
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
    src/gigachat_tool_format.cpp
    src/mimo_tool_format.cpp
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
    src/xing_moe_convert.cpp
    src/xing_moe_load.cpp
    src/xing_moe_model.cpp
    src/xing_moe_tool_format.cpp
    src/mistral_moe_convert.cpp
    src/mistral_moe_large4_convert.cpp
    src/minimax_moe_convert.cpp
    src/minimax_moe_flash_convert.cpp
    src/minimax_moe_flash_model.cpp
    src/minimax_moe_load.cpp
    src/minimax_moe_model.cpp
    src/minimax_moe_tool_format.cpp
    src/hy_moe_convert.cpp
    src/hy_moe_simulmt_convert.cpp
    src/hy_moe_load.cpp
    src/step_moe_convert.cpp
    src/step_moe_load.cpp
    src/flint_moe_convert.cpp
    src/flint_moe_load.cpp
    src/flint_moe_model.cpp
    src/flint_moe_tool_format.cpp
    src/ling_moe_convert.cpp
    src/ling_moe_load.cpp
    src/ling_moe_model.cpp
    src/ling_moe_tool_format.cpp
    src/beam_moe_convert.cpp
    src/beam_moe_load.cpp
    src/beam_moe_model.cpp
    src/beam_moe_tool_format.cpp
    src/hy_moe_model.cpp
    src/hy_moe_simulmt_model.cpp
    src/hy_moe_tool_format.cpp
    src/step_moe_model.cpp
    src/step_moe_tool_format.cpp
    src/mistral_moe_load.cpp
    src/mistral_moe_large4_load.cpp
    src/mistral_moe_tool_format.cpp
    src/mistral_moe_model.cpp
    src/mistral_moe_large4_model.cpp
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
    src/pread_pool.cpp
    src/arrow_tree_verify.cpp
    src/tokenizer.cpp
    src/tokenizer_arrows.cpp
    src/prompt_bridge.cpp
    src/tokenizer_convert.cpp
    src/unicode.cpp
    src/unicode_nfc.cpp
    src/vram_estimate.cpp
)
set_target_properties(lsxcommon PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)

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
    "${LSX_CCCL_INCLUDE_DIR}"
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
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${LSX_NVCC_RES_USAGE}>>:-res-usage>
)
target_include_directories(lsxcommon_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
)

add_library(lsxcommon_glm_cu OBJECT
    src/glm_moe_inference.cu
    src/glm_moe_load.cu
    src/glm_moe_flash_inference.cu
)
set_target_properties(lsxcommon_glm_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_glm_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_glm_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${LSX_NVCC_RES_USAGE}>>:-res-usage>
)
target_include_directories(lsxcommon_glm_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_glm_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_glm_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
)
# FMA policy: object-lib -fmad flags are historical (qwen_moe/glm/kimi off,
# qwen dense on). Shared kernels in lsx_kernels_*.cuh use explicit *_rn
# intrinsics and are therefore flag-independent; DO NOT rely on contraction
# in any new kernel.
#
# Concretely, for any kernel on an -fmad=TRUE object library: bit-identity of a
# vectorised rewrite is EMPIRICAL, not structural. Rewriting `dot += a[i]*b[i]` as a
# float4/__half2 loop that keeps the ascending order still relies on the compiler
# contracting both forms the same way - nothing forbids otherwise. Either use explicit
# __fadd_rn/__fmul_rn (flag-independent, like lsx_kernels_*.cuh and Xing's float4 dot),
# or treat the real-weight golden gates as the arbiter and say so in the commit. Do NOT
# write "this TU is -fmad=false" unless you have checked THIS target's options: the
# comment here used to be the only place that said which way each library goes, and
# two helpers claimed a flag their object libraries do not set.
add_library(lsxcommon_qwen_cu OBJECT
    src/lsx_attn_inference.cu
    src/qwen_dense_inference_pipeline.cu
)
set_target_properties(lsxcommon_qwen_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_qwen_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_qwen_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${LSX_NVCC_RES_USAGE}>>:-res-usage>
)
target_include_directories(lsxcommon_qwen_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_qwen_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_qwen_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
)
add_library(lsxcommon_moe_cu OBJECT
    src/lsx_moe_inference.cu
    src/qwen_moe_inference_pipeline.cu
)
set_target_properties(lsxcommon_moe_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_moe_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
# -fmad=false: bit-exact router/expert math (protocol invariant).
target_compile_options(lsxcommon_moe_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(lsxcommon_moe_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_moe_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_moe_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
)

# Xing4.0-29B-A4B kernels + inference (mHC four-stream residual, MLA,
# interleaved YaRN RoPE). -fmad=false like the other bit-exact families: the
# mHC gates pick the expert set, so their arithmetic must not be reassociated.
add_library(lsxcommon_xing_cu OBJECT
    src/xing_moe_kernels.cu
    src/xing_moe_inference.cu
)
set_target_properties(lsxcommon_xing_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_xing_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_xing_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(lsxcommon_xing_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_xing_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_xing_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
)

# MiniMax-M3 kernels + generation pipeline (MSA indexer, SwiGLU-OAI, MoE).
# -fmad=false like the other bit-exact families: the router and the activation
# clamps decide the expert set / the routed sum, so their arithmetic must not
# be reassociated.
add_library(lsxcommon_minimax_cu OBJECT
    src/minimax_moe_kernels.cu
    src/minimax_moe_inference_pipeline.cu
)
set_target_properties(lsxcommon_minimax_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_minimax_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_minimax_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(lsxcommon_minimax_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_minimax_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_minimax_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
)

# Hy-Memory-30B-A3B kernels + generation pipeline (full RoPE, plain per-head
# QK-norm, sigmoid+expert-bias router, resident expert slabs).
# -fmad=false like the other bit-exact families: the router and the routed
# accumulation decide the expert set / the routed sum, so their arithmetic must
# not be reassociated.
add_library(lsxcommon_hy_cu OBJECT
    src/hy_moe_kernels.cu
    src/hy_moe_inference_pipeline.cu
)
set_target_properties(lsxcommon_hy_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_hy_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_hy_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(lsxcommon_hy_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_hy_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_hy_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
)

# Recursive Flint kernels + generation pipeline (learned sink column, hybrid
# sliding/full attention with no RoPE on the full layers, sigmoid top-k router
# and an ungated shared expert).
add_library(lsxcommon_flint_cu OBJECT
    src/flint_moe_kernels.cu
    src/flint_moe_inference_attn.cu
    src/flint_moe_inference_pipeline.cu
)
set_target_properties(lsxcommon_flint_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_flint_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_flint_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(lsxcommon_flint_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_flint_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_flint_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
)

# Beam-1.0 kernels + generation pipeline (gated GQA, SWA/NoPE-SSMax hybrid,
# sandwich norms + depth-scaled residual, group-limited sigmoid-router MoE).
add_library(lsxcommon_beam_cu OBJECT
    src/beam_moe_kernels.cu
    src/beam_moe_inference.cu
)
set_target_properties(lsxcommon_beam_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_beam_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_beam_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(lsxcommon_beam_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_beam_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_beam_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
)

# Ling-3.1-flash (BailingMoeV3): FP8 dequant + KDA/MLA/router/MoE decode.
add_library(lsxcommon_ling_cu OBJECT
    src/ling_moe_dequant.cu
    src/ling_moe_kernels.cu
    src/ling_moe_inference.cu
)
set_target_properties(lsxcommon_ling_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_ling_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_ling_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(lsxcommon_ling_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_ling_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_ling_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
)

# Step-5-Preview: FP8 expert dequant (converter), partial-RoPE/windowed GQA
# attention, clamped SwiGLU and the sigmoid+bias router. -fmad=false: the
# router coefficients and the routed accumulation decide the expert set / the
# routed sum, so their arithmetic must not be reassociated.
add_library(lsxcommon_step_cu OBJECT
    src/step_moe_dequant.cu
    src/step_moe_kernels.cu
    src/step_moe_inference_attn.cu
    src/step_moe_inference_pipeline.cu
)
set_target_properties(lsxcommon_step_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_step_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_step_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(lsxcommon_step_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_step_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_step_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
)

# Mistral-Small-4-119B MLA + MoE decode kernels.
add_library(lsxcommon_mistral_cu OBJECT
    src/mistral_moe_inference.cu
    src/mistral_moe_large4_kernels.cu
    src/mistral_moe_large4_inference_attn.cu
    src/mistral_moe_large4_inference_pipeline.cu
)
set_target_properties(lsxcommon_mistral_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_mistral_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_mistral_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(lsxcommon_mistral_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_mistral_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_mistral_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
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
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;
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
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_kimi_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_kimi_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(lsxcommon_kimi_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_kimi_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_kimi_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
)

add_library(lsxcommon_gemma_cu OBJECT
    src/gemma_moe_inference.cu
    src/gemma_dense_inference.cu
)
set_target_properties(lsxcommon_gemma_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_gemma_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_gemma_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(lsxcommon_gemma_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_gemma_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_gemma_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
)
# Ornith-1.5-35B-A3B MoE kernels (duplicated qwen_moe layer).
# -fmad=false: bit-exact router/expert math (protocol invariant).
add_library(lsxcommon_ornith_cu OBJECT
    src/ornith_moe_inference_pipeline.cu
)
set_target_properties(lsxcommon_ornith_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_ornith_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_ornith_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(lsxcommon_ornith_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_ornith_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_ornith_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
)

# Model-neutral dense-decoder kernels: RMSNorm / RoPE / causal GQA attention /
# fused split-QKV / fused SwiGLU. Algorithm frozen as the Granite-4.2 runtime
# always ran it; consumers delegate here.
add_library(lsxcommon_dense_cu OBJECT
    src/lsx_dense_inference.cu
)
set_target_properties(lsxcommon_dense_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_dense_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_dense_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${LSX_NVCC_RES_USAGE}>>:-res-usage>
)
target_include_directories(lsxcommon_dense_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_dense_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_dense_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
)

# Granite-4.2-30B dense runtime: family-local RoPE + causal GQA attention,
# shared lsx_kernels_*.cuh primitives, neutral NVFP4 GEMM. No -fmad=false is
# required (all shared arithmetic uses explicit *_rn intrinsics).
add_library(lsxcommon_granite_cu OBJECT
    src/granite_dense_inference.cu
    src/granite_dense_switch_inference.cu
)
set_target_properties(lsxcommon_granite_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_granite_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_granite_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${LSX_NVCC_RES_USAGE}>>:-res-usage>
)
target_include_directories(lsxcommon_granite_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_granite_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_granite_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
)

# Dory1.0-4B (looped nGPT) runtime: family-local nGPT normalizations
# (justnorm / slerp-alpha / qk-norm / gated-attention-norm / Suv-SwiGLU-norm)
# and the sliding-window causal GQA attention the neutral dense layer cannot
# express. No -fmad=false: every arithmetic op is an explicit *_rn intrinsic or
# a shuffle-tree reduction, so the kernels are fmad-flag independent.
add_library(lsxcommon_dory_cu OBJECT
    src/dory_dense_inference.cu
)
set_target_properties(lsxcommon_dory_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_dory_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_dory_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${LSX_NVCC_RES_USAGE}>>:-res-usage>
)
target_include_directories(lsxcommon_dory_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_dory_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_dory_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
)

# phi-4 dense runtime: family-local RoPE + causal GQA attention,
# shared lsx_kernels_*.cuh primitives, neutral NVFP4 GEMM. No -fmad=false is
# required (all shared arithmetic uses explicit *_rn intrinsics).
add_library(lsxcommon_phi_cu OBJECT
    src/phi_dense_inference.cu
)
set_target_properties(lsxcommon_phi_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_phi_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_phi_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${LSX_NVCC_RES_USAGE}>>:-res-usage>
)
target_include_directories(lsxcommon_phi_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_phi_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_phi_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
)

# Kimi-K3 runtime: KDA + gated NoPE MLA + AttnRes + Stable LatentMoE.
# -fmad=false: bit-exact against the host references (protocol invariant).
add_library(lsxcommon_kimi_k3_cu OBJECT
    src/kimi_moe_k3_kernels.cu
    src/kimi_moe_k3_inference_pipeline.cu
)
set_target_properties(lsxcommon_kimi_k3_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_kimi_k3_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_kimi_k3_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${LSX_NVCC_RES_USAGE}>>:-res-usage>
)
target_include_directories(lsxcommon_kimi_k3_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_kimi_k3_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_kimi_k3_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
)

# AliceAI runtime: KDA + gated attention + AttnRes + sigmoid router MoE.
# -fmad=false: bit-exact against the host reference (protocol invariant).
add_library(lsxcommon_alice_cu OBJECT
    src/alice_moe_kernels.cu
    src/alice_moe_inference.cu
    src/alice_moe_inference_pipeline.cu
)
set_target_properties(lsxcommon_alice_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_alice_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_alice_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${LSX_NVCC_RES_USAGE}>>:-res-usage>
)
target_include_directories(lsxcommon_alice_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_alice_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_alice_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
)

# NVCC options a translation unit that includes the PUBLIC headers in
# include/lsxcommon needs. Exported as an INTERFACE requirement via lsxcommon so
# headers declare their own needs instead of every consumer rediscovering them
# from a build error. The KV geometry (lsx_kv_cache.h) used to be the reason for
# --expt-relaxed-constexpr here; its helpers are now explicitly __host__
# __device__, so the flag is gone - keeping it only fed nvcc's device-debug
# relaxation path ("Invalid instruction with no BB"). --extended-lambda stays.
add_library(lsxcommon_cuda_reqs INTERFACE)
target_compile_options(lsxcommon_cuda_reqs INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_link_libraries(lsxcommon PUBLIC lsxcommon_cuda_reqs)

target_link_libraries(lsxcommon PUBLIC lsxcommon_cu lsxcommon_glm_cu
    lsxcommon_qwen_cu lsxcommon_moe_cu lsxcommon_tc_cu
    lsxcommon_kimi_cu lsxcommon_gemma_cu lsxcommon_qwen_moe_exp_cu
    lsxcommon_deepseek_v41_cu
    lsxcommon_deepseek_pro_cu
    lsxcommon_nemotron_cu
    lsxcommon_gigachat_cu
    lsxcommon_mimo_cu
    lsxcommon_ornith_cu
    lsxcommon_dense_cu
    lsxcommon_granite_cu
    lsxcommon_phi_cu
    lsxcommon_dory_cu
    lsxcommon_alice_cu
    lsxcommon_kimi_k3_cu
    lsxcommon_xing_cu
    lsxcommon_mistral_cu
    lsxcommon_minimax_cu
    lsxcommon_hy_cu
    lsxcommon_flint_cu
    lsxcommon_beam_cu
    lsxcommon_ling_cu
    lsxcommon_step_cu
    lsx_tune)

# Nemotron-3.5 (NemotronH hybrid) Mamba-2 / gated-RMSNorm kernels.
# -fmad=false: bit-exact against the host reference (protocol invariant).
add_library(lsxcommon_nemotron_cu OBJECT
    src/nemotron_moe_kernels.cu
    src/nemotron_moe_nvfp4_dequant.cu
    src/nemotron_moe_inference_attn.cu
    src/nemotron_moe_inference_pipeline.cu
)
set_target_properties(lsxcommon_nemotron_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_nemotron_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_nemotron_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(lsxcommon_nemotron_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_nemotron_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_nemotron_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
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
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_gigachat_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_gigachat_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(lsxcommon_gigachat_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_gigachat_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_gigachat_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
)

# MiMo-V2.6-Flash-RL: family-local FP8/MXFP4 -> BF16 dequant (converter). The
# inference kernels land in later commits of the ingestion plan.
add_library(lsxcommon_mimo_cu OBJECT
    src/mimo_moe_dequant.cu
    src/mimo_moe_kernels.cu
    src/mimo_moe_inference_attn.cu
)
set_target_properties(lsxcommon_mimo_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_mimo_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_mimo_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(lsxcommon_mimo_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_mimo_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_mimo_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
)

# qwen4_exp (Qwen3.8-Flash-Next) bit-exact kernels: hc GatedResidual, QSA
# indexer, PLE inject. -fmad=false contract (protocol invariant).
add_library(lsxcommon_qwen_moe_exp_cu OBJECT
    src/qwen_moe_exp_inference.cu
)
set_target_properties(lsxcommon_qwen_moe_exp_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_qwen_moe_exp_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_qwen_moe_exp_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-fmad=false>
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
    lsx_dep_includes
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
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_deepseek_v41_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_deepseek_v41_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(lsxcommon_deepseek_v41_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_deepseek_v41_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_deepseek_v41_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
)

# DeepSeek-V4-Pro-0813 inference contour (MLA + HC + CSA/indexer + hash-router
# MoE). Same runtime fast-math contract as DeepSeek-V4-Flash: no -fmad=false;
# the per-call LsxFastMathScope owns the numeric class.
add_library(lsxcommon_deepseek_pro_cu OBJECT
    src/deepseek_moe_pro_load.cu
    src/deepseek_moe_pro_inference_moe.cu
    src/deepseek_moe_pro_inference_attn.cu
    src/deepseek_moe_pro_inference_csa.cu
    src/deepseek_moe_pro_inference_pipeline.cu
)
set_target_properties(lsxcommon_deepseek_pro_cu PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_deepseek_pro_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${LSX_CUDA_GENCODE_FLAGS}>)
target_compile_options(lsxcommon_deepseek_pro_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${LSX_NVCC_RES_USAGE}>>:-res-usage>
)
target_include_directories(lsxcommon_deepseek_pro_cu PRIVATE ${LSXCOMMON_CU_INCLUDES})
target_compile_definitions(lsxcommon_deepseek_pro_cu PRIVATE ${LSXCOMMON_CU_DEFS})
target_link_libraries(lsxcommon_deepseek_pro_cu
    "${CUDA_CUDART_LIBRARY}"
    lsx_dep_includes
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
        "$<BUILD_INTERFACE:${LSX_CCCL_INCLUDE_DIR}>"
    PRIVATE
        # The generated model_registry.inc lives in the build tree and spells its
        # includes the way the old hand-written list did (bare header names), so
        # src/ must be on this target's include path as well.
        "${LSX_MANIFEST_GENERATED_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/lsxcommon/version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/lsxcommon/version.h"
    @ONLY
)

target_link_libraries(lsxcommon PUBLIC
    lsx_dep_includes
    spdlog::spdlog
    flatbuffers
    libzmq-static
    "${CUDA_CUDART_LIBRARY}"
    absl::flat_hash_map
    absl::span
    absl::strings
)
target_link_libraries(lsxcommon PRIVATE
    ${CMAKE_DL_LIBS}
)
# lsx_orchestrator.h lives in the PUBLIC include tree and exposes
# lsxnetwork::FrameQueue in ServeTaskMosaic's signature -> the dependency is
# part of lsxcommon's public interface, so it must be PUBLIC (not PRIVATE) or
# consumers including the orchestrator header lack lsxnetwork's include dirs.
target_link_libraries(lsxcommon PUBLIC lsxnetwork)
