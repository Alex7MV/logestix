# ── Исполняемый файл репозиторных тестов ─────────────────────────────────
add_executable(test_core test_core.cpp)
target_link_libraries(test_core PRIVATE lsxcommon arrow_shared arrow_cuda_shared simdjson::simdjson Catch2::Catch2WithMain)

# Копируем arrow.dll и arrow_cuda.dll в директорию с тестовым исполняемым файлом
if(WIN32)
    add_custom_command(TARGET test_core POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_core>"
    )
    add_custom_command(TARGET test_core POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_core>"
    )
endif()

add_test(NAME test_core COMMAND test_core)

# ── Тест shared fastload-хелперов (host-only, без CUDA-аллокаций) ────────
add_executable(test_fast_load test_fast_load.cpp)
target_link_libraries(test_fast_load PRIVATE lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_fast_load PRIVATE
    ${CMAKE_SOURCE_DIR}/libs/lsxcommon/src)
add_test(NAME test_fast_load COMMAND test_fast_load)

# ── Тест общего geometry-валидатора ModelBase (host-only) ────────────────
add_executable(test_geometry_guard test_geometry_guard.cpp)
target_link_libraries(test_geometry_guard PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_geometry_guard COMMAND test_geometry_guard)


# ── GPU perf auto-tuning policy (host-only) ─────────────────────────────
add_executable(test_gpu_perf test_gpu_perf.cpp)
target_link_libraries(test_gpu_perf PRIVATE lsxcommon Catch2::Catch2WithMain)
add_test(NAME test_gpu_perf COMMAND test_gpu_perf)

# ── Perf baseline reference logic (host-only) ───────────────────────────
add_executable(test_perf_baseline_logic test_perf_baseline_logic.cpp)
target_link_libraries(test_perf_baseline_logic PRIVATE lsxcommon
                      Catch2::Catch2WithMain)
add_test(NAME test_perf_baseline_logic COMMAND test_perf_baseline_logic)


# ── Тест zero-copy Arrow IPC view-загрузки (быстрый путь лоадеров) ──────
add_executable(test_schema_views test_schema_views.cpp)
target_link_libraries(test_schema_views PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_schema_views POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_schema_views>"
    )
    add_custom_command(TARGET test_schema_views POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_schema_views>"
    )
endif()

add_test(NAME test_schema_views COMMAND test_schema_views)


# ── Тест DeepSeek round-trip загрузчика ─────────────────────────────────
add_executable(test_deepseek_moe_loader test_deepseek_moe_loader.cpp)
target_link_libraries(test_deepseek_moe_loader PRIVATE lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_deepseek_moe_loader PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src")

if(WIN32)
    add_custom_command(TARGET test_deepseek_moe_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_moe_loader>"
    )
    add_custom_command(TARGET test_deepseek_moe_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_moe_loader>"
    )
endif()

add_test(NAME test_deepseek_moe_loader COMMAND test_deepseek_moe_loader)

# ── Тест DeepSeek-V4-Flash model-стадии жизненного цикла (host-only):
#    классификация конвертированного layout (detect_model_type / фабрика) +
#    topology-контракт (load_topology) на синтетической logestix-фикстуре.
#    У DeepSeek нет конвертера в репо — convert-стадии by design нет.
add_executable(test_deepseek_moe_model test_deepseek_moe_model.cpp)
target_link_libraries(test_deepseek_moe_model PRIVATE lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_deepseek_moe_model PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src")

if(WIN32)
    add_custom_command(TARGET test_deepseek_moe_model POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_moe_model>"
    )
    add_custom_command(TARGET test_deepseek_moe_model POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_moe_model>"
    )
endif()

add_test(NAME test_deepseek_moe_model COMMAND test_deepseek_moe_model)

# ── DeepSeek-V4.1-Flash: frozen passport (model stage, host-only) ────────
add_executable(test_deepseek_moe_v41_model test_deepseek_moe_v41_model.cpp)
target_link_libraries(test_deepseek_moe_v41_model PRIVATE lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_deepseek_moe_v41_model PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src")

if(WIN32)
    add_custom_command(TARGET test_deepseek_moe_v41_model POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_moe_v41_model>"
    )
    add_custom_command(TARGET test_deepseek_moe_v41_model POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_moe_v41_model>"
    )
endif()

add_test(NAME test_deepseek_moe_v41_model COMMAND test_deepseek_moe_v41_model)

# ── DeepSeek-V4.1-Flash: topology contract (load stage, host-only) ───────
add_executable(test_deepseek_moe_v41_loader test_deepseek_moe_v41_loader.cpp)
target_link_libraries(test_deepseek_moe_v41_loader PRIVATE lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_deepseek_moe_v41_loader PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src")

if(WIN32)
    add_custom_command(TARGET test_deepseek_moe_v41_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_moe_v41_loader>"
    )
    add_custom_command(TARGET test_deepseek_moe_v41_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_moe_v41_loader>"
    )
endif()

add_test(NAME test_deepseek_moe_v41_loader COMMAND test_deepseek_moe_v41_loader)

# ── DeepSeek-V4.1-Flash: converter (synthetic e2e, host-only) ────────────
add_executable(test_deepseek_moe_v41_convert test_deepseek_moe_v41_convert.cpp)
target_compile_definitions(test_deepseek_moe_v41_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_deepseek_moe_v41_convert PRIVATE lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_deepseek_moe_v41_convert PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src")
add_test(NAME test_deepseek_moe_v41_convert COMMAND test_deepseek_moe_v41_convert)
set_tests_properties(test_deepseek_moe_v41_convert PROPERTIES
    TIMEOUT 120
    ENVIRONMENT "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1")
set_tests_properties(test_deepseek_moe_v41_convert PROPERTIES LABELS convert)

# ── Тест общего ExpertStore (LRU-погрузчика MoE-экспертов) ──────────────
add_executable(test_expert_store test_expert_store.cpp)
target_link_libraries(test_expert_store PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_expert_store POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_expert_store>"
    )
    add_custom_command(TARGET test_expert_store POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_expert_store>"
    )
endif()

add_test(NAME test_expert_store COMMAND test_expert_store)
set_tests_properties(test_expert_store PROPERTIES SKIP_RETURN_CODE 4)

# ── Тест Qwen3.5-9B конвертера (парсер имён + индекс + round-trip) ────────
add_executable(test_qwen_dense_convert test_qwen_dense_convert.cpp)
target_compile_definitions(test_qwen_dense_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_dense_convert PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_qwen_dense_convert POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_dense_convert>"
    )
    add_custom_command(TARGET test_qwen_dense_convert POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_dense_convert>"
    )
endif()

add_test(NAME test_qwen_dense_convert COMMAND test_qwen_dense_convert)

# ── Тест Qwen3.5-9B загрузчика (pinned upload, topology, layer dispatch) ────
add_executable(test_qwen_dense_loader test_qwen_dense_loader.cpp)
target_compile_definitions(test_qwen_dense_loader PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_dense_loader PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_qwen_dense_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_dense_loader>"
    )
    add_custom_command(TARGET test_qwen_dense_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_dense_loader>"
    )
endif()

add_test(NAME test_qwen_dense_loader COMMAND test_qwen_dense_loader)

# ── CUDA kernel tests (link lsxcommon; kernels live in lsxcommon_qwen_cu) ──
foreach(qt test_qwen_dense_norm test_qwen_dense_rotary)
    add_executable(${qt} ${qt}.cpp)
    target_link_libraries(${qt} PRIVATE lsxcommon Catch2::Catch2WithMain)
    if(WIN32)
        add_custom_command(TARGET ${qt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:arrow_shared>"
                "$<TARGET_FILE_DIR:${qt}>"
        )
        add_custom_command(TARGET ${qt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:arrow_cuda_shared>"
                "$<TARGET_FILE_DIR:${qt}>"
        )
    endif()
    add_test(NAME ${qt} COMMAND ${qt})
endforeach()

# ── Qwen CUDA kernel test executables (.cu files, stand-alone OBJECT lib) ──
add_library(test_qwen_cu OBJECT
    test_qwen_dense_shuffle_reduce.cu
    test_qwen_dense_mlp.cu
    test_qwen_dense_nvfp4.cu
    test_qwen_nvfp4_tile_gemm.cu
    test_qwen_nvfp4_gemv.cu
    test_lsx_nvfp4_dispatch.cu
    test_perf_knobs_apply.cu
    test_qwen_dense_full_attn.cu
    test_qwen_dense_full_attn_parallel.cu
    test_qwen_dense_linear_attn.cu
    test_qwen_dense_fused_mlp.cu
    test_qwen_dense_fused_linear_attn.cu
    test_blackwell_micro_gemm.cu
)

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
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr;
    --generate-code=arch=compute_120a,code=sm_120a;
    --generate-code=arch=compute_121a,code=sm_121a;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_blackwell_tc_cu PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
)
target_link_libraries(test_blackwell_tc_cu PRIVATE Catch2::Catch2WithMain)

foreach(qt test_blackwell_tc_mma)
    add_executable(${qt})
    set_target_properties(${qt} PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(${qt} PRIVATE test_blackwell_tc_cu lsxcommon Catch2::Catch2WithMain)
    if(WIN32)
        add_custom_command(TARGET ${qt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:arrow_shared>"
                "$<TARGET_FILE_DIR:${qt}>"
        )
        add_custom_command(TARGET ${qt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:arrow_cuda_shared>"
                "$<TARGET_FILE_DIR:${qt}>"
        )
    endif()
    add_test(NAME ${qt} COMMAND ${qt})
endforeach()

# Shared lsx_kernels_* equivalence harness (Phase A). Generic kernels: plain
# 120/121 code, no arch-'a' PTX required.
add_library(test_lsx_kernels_cu OBJECT
    test_lsx_kernels.cu
)
set_target_properties(test_lsx_kernels_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "120-real;121-real"
)
target_compile_options(test_lsx_kernels_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_lsx_kernels_cu PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_lsx_kernels_cu PRIVATE Catch2::Catch2WithMain)

foreach(qt test_lsx_kernels)
    add_executable(${qt})
    set_target_properties(${qt} PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(${qt} PRIVATE test_lsx_kernels_cu lsxcommon Catch2::Catch2WithMain)
    if(WIN32)
        add_custom_command(TARGET ${qt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:arrow_shared>"
                "$<TARGET_FILE_DIR:${qt}>"
        )
        add_custom_command(TARGET ${qt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:arrow_cuda_shared>"
                "$<TARGET_FILE_DIR:${qt}>"
        )
    endif()
    add_test(NAME ${qt} COMMAND ${qt})
endforeach()

add_library(test_cuda_toolkit_guard_cu OBJECT
    test_cuda_toolkit_guard.cu
)
set_target_properties(test_cuda_toolkit_guard_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "120-real;121-real"
)
target_compile_options(test_cuda_toolkit_guard_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_link_libraries(test_cuda_toolkit_guard_cu PRIVATE
    Catch2::Catch2WithMain CUDA::cudart)

foreach(qt test_cuda_toolkit_guard)
    add_executable(${qt})
    set_target_properties(${qt} PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(${qt} PRIVATE test_cuda_toolkit_guard_cu Catch2::Catch2WithMain)
    if(WIN32)
        add_custom_command(TARGET ${qt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:arrow_shared>"
                "$<TARGET_FILE_DIR:${qt}>"
        )
        add_custom_command(TARGET ${qt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:arrow_cuda_shared>"
                "$<TARGET_FILE_DIR:${qt}>"
        )
    endif()
    add_test(NAME ${qt} COMMAND ${qt})
endforeach()

set_target_properties(test_qwen_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_qwen_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr;
    --generate-code=arch=compute_120,code=sm_120;
    --generate-code=arch=compute_121,code=sm_121;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_qwen_cu PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/thrust"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/cub"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/libcudacxx/include"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
)
target_compile_definitions(test_qwen_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
)
target_link_libraries(test_qwen_cu PRIVATE Catch2::Catch2WithMain)

foreach(qt test_qwen_dense_shuffle_reduce test_qwen_dense_mlp test_qwen_dense_nvfp4 test_qwen_nvfp4_tile_gemm test_qwen_nvfp4_gemv test_lsx_nvfp4_dispatch test_perf_knobs_apply test_qwen_dense_full_attn test_qwen_dense_full_attn_parallel test_qwen_dense_linear_attn test_qwen_dense_fused_mlp test_qwen_dense_fused_linear_attn test_blackwell_micro_gemm)
    add_executable(${qt})
    set_target_properties(${qt} PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(${qt} PRIVATE test_qwen_cu lsxcommon Catch2::Catch2WithMain)
    if(WIN32)
        add_custom_command(TARGET ${qt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:arrow_shared>"
                "$<TARGET_FILE_DIR:${qt}>"
        )
        add_custom_command(TARGET ${qt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:arrow_cuda_shared>"
                "$<TARGET_FILE_DIR:${qt}>"
        )
    endif()
    add_test(NAME ${qt} COMMAND ${qt})
endforeach()

# ── Тест Qwen3.5-9B модели (фабрика + IModel + генерация) ──────────────
add_executable(test_qwen_dense_model test_qwen_dense_model.cpp)
target_compile_definitions(test_qwen_dense_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_dense_model PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_qwen_dense_model POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_dense_model>"
    )
    add_custom_command(TARGET test_qwen_dense_model POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_dense_model>"
    )
endif()

add_test(NAME test_qwen_dense_model COMMAND test_qwen_dense_model)
# Synthetic tiny-geometry fixture: opt out of the frozen-geometry hard gate.
set_tests_properties(test_qwen_dense_model PROPERTIES
    ENVIRONMENT "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1")

# ── GLM expert device-cache: host-side LRU bookkeeping (no CUDA) ────────
add_executable(test_glm_moe_expert_cache test_glm_moe_expert_cache.cpp)
target_link_libraries(test_glm_moe_expert_cache PRIVATE lsxcommon Catch2::Catch2WithMain)
if(WIN32)
    add_custom_command(TARGET test_glm_moe_expert_cache POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_expert_cache>"
    )
endif()
add_test(NAME test_glm_moe_expert_cache COMMAND test_glm_moe_expert_cache)

# ── Инкрементальный KV-кэш Qwen (Task 7) ─────────────────────────────────
add_executable(test_qwen_dense_kv_cache test_qwen_dense_kv_cache.cpp)
target_compile_definitions(test_qwen_dense_kv_cache PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_dense_kv_cache PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_qwen_dense_kv_cache POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_dense_kv_cache>"
    )
    add_custom_command(TARGET test_qwen_dense_kv_cache POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_dense_kv_cache>"
    )
endif()

add_test(NAME test_qwen_dense_kv_cache COMMAND test_qwen_dense_kv_cache)
set_tests_properties(test_qwen_dense_kv_cache PROPERTIES SKIP_RETURN_CODE 4)

# ── Bit-exactness gate: incremental KV vs full recompute (synthetic) ────────
add_executable(test_qwen_dense_kv_parity test_qwen_dense_kv_parity.cpp)
target_compile_definitions(test_qwen_dense_kv_parity PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_dense_kv_parity PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_qwen_dense_kv_parity POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_dense_kv_parity>"
    )
    add_custom_command(TARGET test_qwen_dense_kv_parity POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_dense_kv_parity>"
    )
endif()

add_test(NAME test_qwen_dense_kv_parity COMMAND test_qwen_dense_kv_parity)

# ── Тест Qwen3.8-27B модели ──────────────────────────────────────────────
add_executable(test_qwen_dense_27b_model test_qwen_dense_27b_model.cpp)
target_compile_definitions(test_qwen_dense_27b_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_dense_27b_model PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_qwen_dense_27b_model POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_dense_27b_model>"
    )
    add_custom_command(TARGET test_qwen_dense_27b_model POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_dense_27b_model>"
    )
endif()

add_test(NAME test_qwen_dense_27b_model COMMAND test_qwen_dense_27b_model)
# Synthetic tiny-geometry fixture: opt out of the frozen-geometry hard gate.
set_tests_properties(test_qwen_dense_27b_model PROPERTIES
    ENVIRONMENT "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1")

# ── Тест общей модели проверки целостности (integrity:: helpers) ─────────
add_executable(test_model_integrity test_model_integrity.cpp)
target_compile_definitions(test_model_integrity PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_model_integrity PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_model_integrity POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_model_integrity>"
    )
    add_custom_command(TARGET test_model_integrity POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_model_integrity>"
    )
endif()

add_test(NAME test_model_integrity COMMAND test_model_integrity)

# ── Интеграционный тест целостности Qwen-конвертера ───────────────────────
add_executable(test_qwen_dense_converter_integrity test_qwen_dense_converter_integrity.cpp)
target_compile_definitions(test_qwen_dense_converter_integrity PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_dense_converter_integrity PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_qwen_dense_converter_integrity POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_dense_converter_integrity>"
    )
    add_custom_command(TARGET test_qwen_dense_converter_integrity POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_dense_converter_integrity>"
    )
endif()

add_test(NAME test_qwen_dense_converter_integrity COMMAND test_qwen_dense_converter_integrity)

# ── Тест GLM-5.2 конвертера (парсер имён тензоров) ────────────────────────
add_executable(test_glm_moe_convert test_glm_moe_convert.cpp)
target_compile_definitions(test_glm_moe_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_moe_convert PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_moe_convert POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_convert>"
    )
    add_custom_command(TARGET test_glm_moe_convert POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_convert>"
    )
endif()

add_test(NAME test_glm_moe_convert COMMAND test_glm_moe_convert)

# ── Тест GLM-5.3-Flash конвертера (геометрия, классификатор, requant) ─────
add_executable(test_glm_moe_flash_convert test_glm_moe_flash_convert.cpp)
target_compile_definitions(test_glm_moe_flash_convert PRIVATE
    LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_glm_moe_flash_convert PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_glm_moe_flash_convert PRIVATE
    ${CMAKE_SOURCE_DIR}/libs/lsxcommon/src)

if(WIN32)
    add_custom_command(TARGET test_glm_moe_flash_convert POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_flash_convert>"
    )
endif()

add_test(NAME test_glm_moe_flash_convert COMMAND test_glm_moe_flash_convert)
set_tests_properties(test_glm_moe_flash_convert PROPERTIES
    ENVIRONMENT "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1")

# ── Тест Kimi-K2.7-Code конвертера (геометрия, классификатор, топология) ──
add_executable(test_kimi_moe_convert test_kimi_moe_convert.cpp)
target_compile_definitions(test_kimi_moe_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_kimi_moe_convert PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_kimi_moe_convert POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_kimi_moe_convert>"
    )
    add_custom_command(TARGET test_kimi_moe_convert POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_kimi_moe_convert>"
    )
endif()

add_test(NAME test_kimi_moe_convert COMMAND test_kimi_moe_convert)

# ── Тест GLM-5.2 загрузчика (pinned BF16→FP16 upload) ─────────────────────
add_executable(test_glm_moe_loader test_glm_moe_loader.cpp)
target_compile_definitions(test_glm_moe_loader PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_moe_loader PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_moe_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_loader>"
    )
    add_custom_command(TARGET test_glm_moe_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_loader>"
    )
endif()

add_test(NAME test_glm_moe_loader COMMAND test_glm_moe_loader)

# ── Тест GLM-5.2 модели (фабрика + IModel) ─────────────────────────────────
add_executable(test_glm_moe_model test_glm_moe_model.cpp)
target_compile_definitions(test_glm_moe_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_moe_model PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_moe_model POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_model>"
    )
    add_custom_command(TARGET test_glm_moe_model POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_model>"
    )
endif()

add_test(NAME test_glm_moe_model COMMAND test_glm_moe_model)

# ── Тест GLM-5.2 SelfCheck (шаг 8 протокола): инкрементальный 1->2-токенный
#    селфчек на реальной модели, минимум 3 цикла стресса (для подачи нагрузки
#    достаточно 3-х раз; переопределяется LSX_TEST_GLM_SELFCHECK_CYCLES),
#    бит-точная детерминированность final_logits, exit 0. SKIP без чекпоинта.
add_executable(test_glm_moe_selfcheck test_glm_moe_selfcheck.cpp)
target_compile_definitions(test_glm_moe_selfcheck PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_moe_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_moe_selfcheck POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_selfcheck>"
    )
    add_custom_command(TARGET test_glm_moe_selfcheck POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_selfcheck>"
    )
endif()

add_test(NAME test_glm_moe_selfcheck COMMAND test_glm_moe_selfcheck)
set_tests_properties(test_glm_moe_selfcheck PROPERTIES SKIP_RETURN_CODE 4)

# ── Blackwell audit Phase 0: GLM-5.2 perf baseline harness (reporting test,
#    SKIP без чекпоинта). Число генерируемых токенов — LSX_TEST_PERF_GEN_TOKENS.
add_executable(test_blackwell_perf_glm test_blackwell_perf_glm.cpp)
target_link_libraries(test_blackwell_perf_glm PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_blackwell_perf_glm POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_blackwell_perf_glm>"
    )
    add_custom_command(TARGET test_blackwell_perf_glm POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_blackwell_perf_glm>"
    )
endif()

add_test(NAME test_blackwell_perf_glm COMMAND test_blackwell_perf_glm)
set_tests_properties(test_blackwell_perf_glm PROPERTIES SKIP_RETURN_CODE 4)

# ── Blackwell audit Phase 0: dense perf baseline harness (Qwen3.5-9B).
add_executable(test_blackwell_perf_dense test_blackwell_perf_dense.cpp)
target_link_libraries(test_blackwell_perf_dense PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_blackwell_perf_dense POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_blackwell_perf_dense>"
    )
    add_custom_command(TARGET test_blackwell_perf_dense POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_blackwell_perf_dense>"
    )
endif()

add_test(NAME test_blackwell_perf_dense COMMAND test_blackwell_perf_dense)
set_tests_properties(test_blackwell_perf_dense PROPERTIES SKIP_RETURN_CODE 4)

# ── MoE perf baseline harnesses: Qwen3.5-35B-A3B, gemma-4-26B-A4B-it.
add_executable(test_blackwell_perf_moe test_blackwell_perf_moe.cpp)
target_link_libraries(test_blackwell_perf_moe PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_blackwell_perf_moe POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_blackwell_perf_moe>"
    )
    add_custom_command(TARGET test_blackwell_perf_moe POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_blackwell_perf_moe>"
    )
endif()

# One binary, two ctest entries (Catch2 tag filters) so each MoE model is
# individually selectable and named in ctest output.
add_test(NAME test_blackwell_perf_moe_qwen
         COMMAND test_blackwell_perf_moe "[qwen35moe]")
add_test(NAME test_blackwell_perf_moe_gemma
         COMMAND test_blackwell_perf_moe "[gemma26moe]")
set_tests_properties(test_blackwell_perf_moe_qwen PROPERTIES SKIP_RETURN_CODE 4)
set_tests_properties(test_blackwell_perf_moe_gemma PROPERTIES SKIP_RETURN_CODE 4)

# ── Blackwell audit Phase 0: DeepSeek-V4-Flash perf baseline harness.
add_executable(test_blackwell_perf_deepseek test_blackwell_perf_deepseek.cpp)
target_link_libraries(test_blackwell_perf_deepseek PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_blackwell_perf_deepseek POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_blackwell_perf_deepseek>"
    )
    add_custom_command(TARGET test_blackwell_perf_deepseek POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_blackwell_perf_deepseek>"
    )
endif()

add_test(NAME test_blackwell_perf_deepseek COMMAND test_blackwell_perf_deepseek)
set_tests_properties(test_blackwell_perf_deepseek PROPERTIES SKIP_RETURN_CODE 4)

# ── Тест GLM-5.2 seq-P prefill parity: MoE-слои 3..77 обязаны держать
#    block_rows = P (полная длина промпта), каждый промпт-токен занимает один
#    IndexShare KV-слот. FAIL при коллапсе строк (kv_slots_used != P + gen),
#    при расхождении argmax final_logits с первым сгенерированным токеном или
#    при потере бит-точности между двумя идентичными прогонами. SKIP без
#    чекпоинта.
add_executable(test_glm_moe_prefill_seqp test_glm_moe_prefill_seqp.cpp)
target_compile_definitions(test_glm_moe_prefill_seqp PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_moe_prefill_seqp PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_moe_prefill_seqp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_prefill_seqp>"
    )
    add_custom_command(TARGET test_glm_moe_prefill_seqp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_prefill_seqp>"
    )
endif()

add_test(NAME test_glm_moe_prefill_seqp COMMAND test_glm_moe_prefill_seqp)
set_tests_properties(test_glm_moe_prefill_seqp PROPERTIES SKIP_RETURN_CODE 4)

# ── Тест GLM-5.2 осмысленного вывода по --run (Infer output_text) ───────────
add_executable(test_glm_moe_prompt_output test_glm_moe_prompt_output.cpp)
target_compile_definitions(test_glm_moe_prompt_output PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_moe_prompt_output PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_moe_prompt_output POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_prompt_output>"
    )
    add_custom_command(TARGET test_glm_moe_prompt_output POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_prompt_output>"
    )
endif()

add_test(NAME test_glm_moe_prompt_output COMMAND test_glm_moe_prompt_output)
set_tests_properties(test_glm_moe_prompt_output PROPERTIES SKIP_RETURN_CODE 4)

# ── Тест DeepSeek-V4-Flash осмысленного вывода (Infer output_text) ────────
add_executable(test_deepseek_moe_prompt_output test_deepseek_moe_prompt_output.cpp)
target_link_libraries(test_deepseek_moe_prompt_output PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_deepseek_moe_prompt_output POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_moe_prompt_output>"
    )
    add_custom_command(TARGET test_deepseek_moe_prompt_output POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_moe_prompt_output>"
    )
endif()

add_test(NAME test_deepseek_moe_prompt_output COMMAND test_deepseek_moe_prompt_output)
set_tests_properties(test_deepseek_moe_prompt_output PROPERTIES SKIP_RETURN_CODE 4)

# ── Тест DeepSeek-V4-Flash SelfCheck (шаг 8 протокола): инкрементальный
#    1->2-токенный селфчек на реальной модели, минимум 3 цикла, детерминированный
#    final_logits. SKIP без чекпоинта.
add_executable(test_deepseek_moe_selfcheck test_deepseek_moe_selfcheck.cpp)
target_link_libraries(test_deepseek_moe_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_deepseek_moe_selfcheck POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_moe_selfcheck>"
    )
    add_custom_command(TARGET test_deepseek_moe_selfcheck POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_moe_selfcheck>"
    )
endif()

add_test(NAME test_deepseek_moe_selfcheck COMMAND test_deepseek_moe_selfcheck)
set_tests_properties(test_deepseek_moe_selfcheck PROPERTIES SKIP_RETURN_CODE 4)

# ── Тест Qwen3.5-9B SelfCheck (шаг 8 протокола): инкрементальный 1→2-токенный
#    селфчек на реальной модели, минимум 3 цикла, детерминированный final_logits.
#    SKIP без чекпоинта.
add_executable(test_qwen_dense_selfcheck test_qwen_dense_selfcheck.cpp)
target_compile_definitions(test_qwen_dense_selfcheck PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_dense_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_qwen_dense_selfcheck POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_dense_selfcheck>"
    )
    add_custom_command(TARGET test_qwen_dense_selfcheck POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_dense_selfcheck>"
    )
endif()

add_test(NAME test_qwen_dense_selfcheck COMMAND test_qwen_dense_selfcheck)
set_tests_properties(test_qwen_dense_selfcheck PROPERTIES SKIP_RETURN_CODE 4)

# ── Qwen3.8-27B SelfCheck (реальная модель) ─────────────────────────────
#    Селфчек на реальной модели, минимум 3 цикла, детерминированный final_logits.
#    SKIP без чекпоинта.
add_executable(test_qwen_dense_27b_selfcheck test_qwen_dense_27b_selfcheck.cpp)
target_compile_definitions(test_qwen_dense_27b_selfcheck PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_dense_27b_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_qwen_dense_27b_selfcheck POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_dense_27b_selfcheck>"
    )
    add_custom_command(TARGET test_qwen_dense_27b_selfcheck POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_dense_27b_selfcheck>"
    )
endif()

add_test(NAME test_qwen_dense_27b_selfcheck COMMAND test_qwen_dense_27b_selfcheck)
set_tests_properties(test_qwen_dense_27b_selfcheck PROPERTIES SKIP_RETURN_CODE 4)

# ── Тест фабрики моделей ────────────────────────────────────────────────
add_executable(test_model_factory test_model_factory.cpp)
target_link_libraries(test_model_factory PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_model_factory POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_model_factory>"
    )
    add_custom_command(TARGET test_model_factory POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_model_factory>"
    )
endif()

add_test(NAME test_model_factory COMMAND test_model_factory)

# ── Тест generation-path CLI (--run / -r нормализация + default prompt) ─────
add_executable(test_generation_cli test_generation_cli.cpp)
target_link_libraries(test_generation_cli PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_generation_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_generation_cli>"
    )
    add_custom_command(TARGET test_generation_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_generation_cli>"
    )
endif()

add_test(NAME test_generation_cli COMMAND test_generation_cli)

# ── Тест GLM-5.2 prompt framing helper (host-only, синтетический токенизатор) ────
add_executable(test_glm_moe_prompt_framing test_glm_moe_prompt_framing.cpp)
target_compile_definitions(test_glm_moe_prompt_framing PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_moe_prompt_framing PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_moe_prompt_framing POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_prompt_framing>"
    )
    add_custom_command(TARGET test_glm_moe_prompt_framing POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_prompt_framing>"
    )
endif()

add_test(NAME test_glm_moe_prompt_framing COMMAND test_glm_moe_prompt_framing)

# ── Тест Gemma-4 prompt framing helper (host-only, сентинелы tokenizer.json) ─────
add_executable(test_gemma_prompt_framing test_gemma_prompt_framing.cpp)
target_compile_definitions(test_gemma_prompt_framing PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_gemma_prompt_framing PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_gemma_prompt_framing POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_gemma_prompt_framing>"
    )
endif()

add_test(NAME test_gemma_prompt_framing COMMAND test_gemma_prompt_framing)

# ── Тест Gemma-4 конвертера (классификатор имён + NVFP4-политика) ──────────
add_executable(test_gemma_moe_convert test_gemma_moe_convert.cpp)
target_compile_definitions(test_gemma_moe_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_gemma_moe_convert PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_gemma_moe_convert POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_gemma_moe_convert>"
    )
    add_custom_command(TARGET test_gemma_moe_convert POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_gemma_moe_convert>"
    )
endif()

add_test(NAME test_gemma_moe_convert COMMAND test_gemma_moe_convert)
set_tests_properties(test_gemma_moe_convert PROPERTIES
    TIMEOUT 120)

# ── Тест Gemma-4 загрузчика (BF16→FP16 in-place, арена, expert LRU) ─────────
add_executable(test_gemma_moe_loader test_gemma_moe_loader.cpp)
target_compile_definitions(test_gemma_moe_loader PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_gemma_moe_loader PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_gemma_moe_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_gemma_moe_loader>"
    )
    add_custom_command(TARGET test_gemma_moe_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_gemma_moe_loader>"
    )
endif()

add_test(NAME test_gemma_moe_loader COMMAND test_gemma_moe_loader)

# ── Тест Gemma-4 модели (фабрика + Load + Infer, синтетическая геометрия) ────
add_executable(test_gemma_moe_model test_gemma_moe_model.cpp)
target_include_directories(test_gemma_moe_model PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src")
target_compile_definitions(test_gemma_moe_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_gemma_moe_model PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_gemma_moe_model POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_gemma_moe_model>"
    )
    add_custom_command(TARGET test_gemma_moe_model POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_gemma_moe_model>"
    )
endif()

add_test(NAME test_gemma_moe_model COMMAND test_gemma_moe_model)
set_tests_properties(test_gemma_moe_model PROPERTIES
    ENVIRONMENT "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1")

# ── Тест Gemma-4 SelfCheck (шаг 8 протокола): инкрементальный 1→2-токенный
#    селфчек на реальной модели, минимум 3 цикла стресса (переопределяется
#    LSX_TEST_GEMMA_SELFCHECK_CYCLES), бит-точная детерминированность
#    final_logits, exit 0. SKIP без чекпоинта.
add_executable(test_gemma_moe_selfcheck test_gemma_moe_selfcheck.cpp)
target_compile_definitions(test_gemma_moe_selfcheck PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_gemma_moe_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_gemma_moe_selfcheck POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_gemma_moe_selfcheck>"
    )
    add_custom_command(TARGET test_gemma_moe_selfcheck POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_gemma_moe_selfcheck>"
    )
endif()

add_test(NAME test_gemma_moe_selfcheck COMMAND test_gemma_moe_selfcheck)
set_tests_properties(test_gemma_moe_selfcheck PROPERTIES SKIP_RETURN_CODE 4)

# ── Gemma-4 kernel reference tests (norm / rotary) ──────────────────────────
foreach(qt test_gemma_moe_norm test_gemma_moe_rotary test_gemma_moe_attn test_gemma_moe_moe)
    add_executable(${qt} ${qt}.cpp)
    target_link_libraries(${qt} PRIVATE lsxcommon Catch2::Catch2WithMain)
    if(WIN32)
        add_custom_command(TARGET ${qt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:arrow_shared>"
                "$<TARGET_FILE_DIR:${qt}>"
        )
        add_custom_command(TARGET ${qt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:arrow_cuda_shared>"
                "$<TARGET_FILE_DIR:${qt}>"
        )
    endif()
    add_test(NAME ${qt} COMMAND ${qt})
endforeach()

# ── Тест ModelBase (tokenizer-db load + infer guards, host-only) ─────────
add_executable(test_model_runtime test_model_runtime.cpp)
target_compile_definitions(test_model_runtime PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_include_directories(test_model_runtime PRIVATE
    ${CMAKE_SOURCE_DIR}/libs/lsxcommon/src)
target_link_libraries(test_model_runtime PRIVATE lsxcommon Catch2::Catch2WithMain)

# ── Тест InferenceEngine singleton + TokenChunk streaming/abort (host-only) ──
add_executable(test_model_engine test_model_engine.cpp)
target_link_libraries(test_model_engine PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_model_engine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_model_engine>"
    )
    add_custom_command(TARGET test_model_engine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_model_engine>"
    )
endif()

add_test(NAME test_model_engine COMMAND test_model_engine)

if(WIN32)
    add_custom_command(TARGET test_model_runtime POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_model_runtime>"
    )
    add_custom_command(TARGET test_model_runtime POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_model_runtime>"
    )
endif()

add_test(NAME test_model_runtime COMMAND test_model_runtime)

# ── Perf baseline guard (real weights, label perf — manual command) ──────
file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}" LSX_SRC_DIR_FWD_PERF)
add_executable(test_perf_baseline test_perf_baseline.cpp)
target_compile_definitions(test_perf_baseline PRIVATE
    LSX_SOURCE_DIR="${LSX_SRC_DIR_FWD_PERF}")
target_link_libraries(test_perf_baseline PRIVATE lsxcommon
                      Catch2::Catch2WithMain)
# One ctest entry per model: each model must measure in its own process —
# models sharing one process contend for VRAM (expert caches of the previous
# model starve the next one: ~2x measured spread). Process exit frees VRAM,
# giving every model the lsxbenchmark-isolated measurement it baselines.
# Literal names (no foreach): test_ctest_labels_guard statically parses
# add_test/set_tests_properties and requires literal, quoted-label-free calls.
add_test(NAME test_perf_baseline_gemma_4_26b_a4b_it
         COMMAND test_perf_baseline "perf baseline: gemma-4-26b-a4b-it")
set_tests_properties(test_perf_baseline_gemma_4_26b_a4b_it PROPERTIES
    TIMEOUT 1200
    SKIP_RETURN_CODE 4
    ENVIRONMENT "LSX_PERF_VAULT_SUBDIR=gemma-4-26B-A4B-it")
set_tests_properties(test_perf_baseline_gemma_4_26b_a4b_it
    PROPERTIES LABELS perf)
add_test(NAME test_perf_baseline_glm_5_3_flash
         COMMAND test_perf_baseline "perf baseline: glm-5.3-flash")
set_tests_properties(test_perf_baseline_glm_5_3_flash PROPERTIES
    TIMEOUT 1200
    SKIP_RETURN_CODE 4
    ENVIRONMENT "LSX_PERF_VAULT_SUBDIR=GLM-5.3-Flash")
set_tests_properties(test_perf_baseline_glm_5_3_flash
    PROPERTIES LABELS perf)
add_test(NAME test_perf_baseline_deepseek_v4_flash
         COMMAND test_perf_baseline "perf baseline: deepseek-v4-flash")
set_tests_properties(test_perf_baseline_deepseek_v4_flash PROPERTIES
    TIMEOUT 1200
    SKIP_RETURN_CODE 4
    ENVIRONMENT "LSX_PERF_VAULT_SUBDIR=DeepSeek-V4-Flash")
set_tests_properties(test_perf_baseline_deepseek_v4_flash
    PROPERTIES LABELS perf)
add_test(NAME test_perf_baseline_qwen3_5_35b_a3b
         COMMAND test_perf_baseline "perf baseline: qwen3.5-35b-a3b")
set_tests_properties(test_perf_baseline_qwen3_5_35b_a3b PROPERTIES
    TIMEOUT 1200
    SKIP_RETURN_CODE 4
    ENVIRONMENT "LSX_PERF_VAULT_SUBDIR=Qwen3.5-35B-A3B")
set_tests_properties(test_perf_baseline_qwen3_5_35b_a3b
    PROPERTIES LABELS perf)
add_test(NAME test_perf_baseline_qwen3_6_35b_a3b
         COMMAND test_perf_baseline "perf baseline: qwen3.6-35b-a3b")
set_tests_properties(test_perf_baseline_qwen3_6_35b_a3b PROPERTIES
    TIMEOUT 1200
    SKIP_RETURN_CODE 4
    ENVIRONMENT "LSX_PERF_VAULT_SUBDIR=Qwen3.6-35B-A3B")
set_tests_properties(test_perf_baseline_qwen3_6_35b_a3b
    PROPERTIES LABELS perf)
add_test(NAME test_perf_baseline_qwen3_8_27b
         COMMAND test_perf_baseline "perf baseline: qwen3.8-27b")
set_tests_properties(test_perf_baseline_qwen3_8_27b PROPERTIES
    TIMEOUT 1200
    SKIP_RETURN_CODE 4
    ENVIRONMENT "LSX_PERF_VAULT_SUBDIR=Qwen3.8-27B")
set_tests_properties(test_perf_baseline_qwen3_8_27b
    PROPERTIES LABELS perf)

# ── Тест E8M0 -> pow2-E4M3 scale-пары (DeepSeek NVFP4 repack, Task 1) ─────
add_executable(test_deepseek_nvfp4_repack test_deepseek_nvfp4_repack.cpp)
target_include_directories(test_deepseek_nvfp4_repack PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src")
target_link_libraries(test_deepseek_nvfp4_repack PRIVATE lsxcommon
                      Catch2::Catch2WithMain)
# Scale-primitive contract (fast) and the repack-tool e2e (convert, [tool]
# tag filter — one binary, two ctest entries per the label discipline).
add_test(NAME test_deepseek_nvfp4_repack COMMAND test_deepseek_nvfp4_repack)
set_tests_properties(test_deepseek_nvfp4_repack
    PROPERTIES LABELS fast)
add_test(NAME test_deepseek_nvfp4_repack_tool
         COMMAND test_deepseek_nvfp4_repack "[tool]*")
set_tests_properties(test_deepseek_nvfp4_repack_tool PROPERTIES
    TIMEOUT 120)
set_tests_properties(test_deepseek_nvfp4_repack_tool
    PROPERTIES LABELS convert)

# ── DeepSeek конвертер: синтетический e2e (logestix-v1-nvfp4) ─────────────
add_executable(test_deepseek_moe_convert test_deepseek_moe_convert.cpp)
target_compile_definitions(test_deepseek_moe_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_deepseek_moe_convert PRIVATE lsxcommon
                      Catch2::Catch2WithMain)
add_test(NAME test_deepseek_moe_convert COMMAND test_deepseek_moe_convert)
set_tests_properties(test_deepseek_moe_convert PROPERTIES
    TIMEOUT 120
    ENVIRONMENT "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1")
set_tests_properties(test_deepseek_moe_convert
    PROPERTIES LABELS convert)

# ── Тест zero-alloc токенизатора ─────────────────────────────────────────
add_executable(test_tokenizer test_tokenizer.cpp)
target_compile_definitions(test_tokenizer PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_tokenizer PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_tokenizer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_tokenizer>"
    )
    add_custom_command(TARGET test_tokenizer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_tokenizer>"
    )
endif()

add_test(NAME test_tokenizer COMMAND test_tokenizer)

# ── Тест GLM-5.2 Dense MLP forward (слои 0-2) ─────────────────────────────
add_executable(test_glm_moe_layer_compute test_glm_moe_layer_compute.cpp)
target_link_libraries(test_glm_moe_layer_compute PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_moe_layer_compute POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_layer_compute>"
    )
    add_custom_command(TARGET test_glm_moe_layer_compute POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_layer_compute>"
    )
endif()

add_test(NAME test_glm_moe_layer_compute COMMAND test_glm_moe_layer_compute)

# ── Тест GLM-5.2 Dense MLP forward (слои 0-2, Phase 4.5 hash-фикстура) ───────
# Прогоняет конкретный C++/CUDA execution loop RunGlmDenseMLPStack (SwiGLU
# w2(silu(w1)x*w3)) через explicit async stream по синтетической hash-фикстуре
# hidden=6144 и сверяет поэлементно против независимого CPU FP64-референса
# (RED/GREEN), плюс проверяет целостность указателей/стридов слоёв 0-2.
add_executable(test_glm_moe_dense_compute test_glm_moe_dense_compute.cpp)
target_link_libraries(test_glm_moe_dense_compute PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_moe_dense_compute POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_dense_compute>"
    )
    add_custom_command(TARGET test_glm_moe_dense_compute POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_dense_compute>"
    )
endif()

add_test(NAME test_glm_moe_dense_compute COMMAND test_glm_moe_dense_compute)

# ── Тест GLM-5.2 MoE Layer-3 router (top_k=8 warp selection) ───────────────
add_executable(test_glm_moe_moe_router test_glm_moe_moe_router.cpp)
target_link_libraries(test_glm_moe_moe_router PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_moe_moe_router POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_moe_router>"
    )
    add_custom_command(TARGET test_glm_moe_moe_router POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_moe_router>"
    )
endif()

add_test(NAME test_glm_moe_moe_router COMMAND test_glm_moe_moe_router)

# ── Тест GLM-5.2 эксперт-стека (NVFP4 dequant + SwiGLU, слои 3-77) ─────────
add_executable(test_glm_moe_experts test_glm_moe_experts.cpp)
target_link_libraries(test_glm_moe_experts PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_moe_experts POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_experts>"
    )
    add_custom_command(TARGET test_glm_moe_experts POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_experts>"
    )
endif()

add_test(NAME test_glm_moe_experts COMMAND test_glm_moe_experts)

# ── TDD Cycle 9: GLM-5.2 MoE router + NVFP4 expert stacks (layers 3-77) ───────
# Прогоняет RunGlmMoEStack (слои 3-77) через одиночный contiguous MoE-арену
# (256 виртуальных срезов активаций экспертов) по синтетической hash-фикстуре
# E2M1/E4M3/E8M0 и сверяет поэлементно против независимого CPU FP64-референса
# (RED/GREEN) + проверяет целостность указателей/границ аорены.
add_executable(test_glm_moe_moe_compute test_glm_moe_moe_compute.cpp)
target_link_libraries(test_glm_moe_moe_compute PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_moe_moe_compute POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_moe_compute>"
    )
    add_custom_command(TARGET test_glm_moe_moe_compute POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_moe_compute>"
    )
endif()

add_test(NAME test_glm_moe_moe_compute COMMAND test_glm_moe_moe_compute)

# ── Тест GLM-5.2 IndexShare attention (mod-4 refresh, слои 3-77) ────────────
add_executable(test_glm_moe_attention test_glm_moe_attention.cpp)
target_link_libraries(test_glm_moe_attention PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_moe_attention POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_attention>"
    )
    add_custom_command(TARGET test_glm_moe_attention POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_attention>"
    )
endif()

add_test(NAME test_glm_moe_attention COMMAND test_glm_moe_attention)

# ── Тест GLM-5.2 полного 79-слойного цикла + MTP epilogue (генерация) ───────
add_executable(test_glm_moe_inference test_glm_moe_inference.cpp)
target_link_libraries(test_glm_moe_inference PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_moe_inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_inference>"
    )
    add_custom_command(TARGET test_glm_moe_inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_inference>"
    )
endif()

add_test(NAME test_glm_moe_inference COMMAND test_glm_moe_inference)

# GLM real-weight tests skip (exit 4 = Catch2 "all test cases skipped") when no
# GLM-5.2 checkpoint is present; that is not a failure per the project SKIP
# invariant, so tell CTest to report it as SKIPPED instead.
set_tests_properties(test_glm_moe_inference PROPERTIES SKIP_RETURN_CODE 4)

# ── TDD Cycle 10: GLM-5.2 full 79-layer SKWOS integration regression (unified
#    arena: IndexShare K-V cache + Layer-78 MTP spaces in one contiguous block)
#    Driving the complete execution loop on real weights with multi-turn
#    prompts; asserts no memory leak / no index-boundary truncation, and that
#    the final speculative Layer-78 MTP logits row converges element-for-element
#    against the independent CPU FP64 reference (RED/GREEN).
add_executable(test_glm_moe_full_compute test_glm_moe_full_compute.cpp)
target_link_libraries(test_glm_moe_full_compute PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_moe_full_compute POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_full_compute>"
    )
    add_custom_command(TARGET test_glm_moe_full_compute POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_full_compute>"
    )
endif()

add_test(NAME test_glm_moe_full_compute COMMAND test_glm_moe_full_compute)
set_tests_properties(test_glm_moe_full_compute PROPERTIES SKIP_RETURN_CODE 4)

# ── Тест TaskMosaic CE сетевого слоя (lsxnetwork) ───────────────────────────
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

# ── TDD Phase 4: LsxTokenTranslator (cross-model binary token translator) ─────
# Exercises the offline passport builder (BuildPassport/WritePassport) plus the
# runtime zero-copy mapper (CreateFileMapping/MapViewOfFile) over a synthetic
# GLM-5.2 / DeepSeek-V4-Flash vocabulary pair. Asserts 100% bidirectional index
# mapping accuracy, 64-byte cache-line alignment of the access pointers, and a
# zero-runtime-allocation translation burst. Links lsxnetwork only (no Arrow;
# the global replacement operator-new counter is unit-local).
add_executable(test_token_translator test_token_translator.cpp)
target_link_libraries(test_token_translator PRIVATE lsxnetwork Catch2::Catch2WithMain)

add_test(NAME test_token_translator COMMAND test_token_translator)

# ── TDD Phase 4.5: network translator weld ──────────────────────────────────
# Welds a privately-owned, per-thread LsxTokenTranslator into the pinned
# IoWorker thread context. Concurrency burst (4 workers over cores 0..3) asserts
# per-thread isolation, 64-byte cache-line alignment of every embedded mapping,
# zero cross-thread lock contention, and zero runtime heap allocations via the
# unit-local global replacement operator-new counter. Links lsxnetwork only.
add_executable(test_network_translator_weld test_network_translator_weld.cpp)
target_link_libraries(test_network_translator_weld PRIVATE lsxnetwork Catch2::Catch2WithMain)

add_test(NAME test_network_translator_weld COMMAND test_network_translator_weld)

# ── Интеграционный тест TaskMosaic: мультимодальный оркестратор (lsx_orchestrator) ──
# Прогоняет одновременный burst OpenAI-запросов на обе ветки ("glm-5.2" и
# "deepseek-v4-flash") через длинный lock-free SPSC конвейер и проверяет
# zero-copy retirement без token-drift. Линкует lsxcommon (оркестратор + IModel)
# и lsxnetwork (FrameQueue).
add_executable(test_integrated_pipeline test_integrated_pipeline.cpp)
target_link_libraries(test_integrated_pipeline PRIVATE lsxcommon lsxnetwork Catch2::Catch2WithMain)
target_compile_definitions(test_integrated_pipeline PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

if(WIN32)
    add_custom_command(TARGET test_integrated_pipeline POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_integrated_pipeline>"
    )
    add_custom_command(TARGET test_integrated_pipeline POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_integrated_pipeline>"
    )
endif()

add_test(NAME test_integrated_pipeline COMMAND test_integrated_pipeline)

# ── Тест CUDA inference kernels ─────────────────────────────────────────
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
    --generate-code=arch=compute_120,code=sm_120;
    --generate-code=arch=compute_121,code=sm_121;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_deepseek_moe_inference_kernels_cu PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/thrust"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/cub"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/libcudacxx/include"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
    "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include"
)
target_compile_definitions(test_deepseek_moe_inference_kernels_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
)
target_link_libraries(test_deepseek_moe_inference_kernels_cu PRIVATE Catch2::Catch2WithMain)

add_executable(test_deepseek_moe_inference_kernels)
set_target_properties(test_deepseek_moe_inference_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_deepseek_moe_inference_kernels PRIVATE
    test_deepseek_moe_inference_kernels_cu lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_deepseek_moe_inference_kernels POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_moe_inference_kernels>"
    )
    add_custom_command(TARGET test_deepseek_moe_inference_kernels POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_moe_inference_kernels>"
    )
endif()

add_test(NAME test_deepseek_moe_inference_kernels COMMAND test_deepseek_moe_inference_kernels)

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
    --generate-code=arch=compute_120,code=sm_120;
    --generate-code=arch=compute_121,code=sm_121;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_deepseek_moe_moe_cu PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/thrust"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/cub"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/libcudacxx/include"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
    "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include"
)
target_compile_definitions(test_deepseek_moe_moe_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
)
target_link_libraries(test_deepseek_moe_moe_cu PRIVATE Catch2::Catch2WithMain)

add_executable(test_deepseek_moe_moe)
set_target_properties(test_deepseek_moe_moe PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_deepseek_moe_moe PRIVATE
    test_deepseek_moe_moe_cu lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_deepseek_moe_moe POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_moe_moe>"
    )
    add_custom_command(TARGET test_deepseek_moe_moe POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_moe_moe>"
    )
endif()

add_test(NAME test_deepseek_moe_moe COMMAND test_deepseek_moe_moe)

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
    --generate-code=arch=compute_120,code=sm_120;
    --generate-code=arch=compute_121,code=sm_121;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_deepseek_moe_arena_cu PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/thrust"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/cub"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/libcudacxx/include"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
    "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include"
)
target_compile_definitions(test_deepseek_moe_arena_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
)
target_link_libraries(test_deepseek_moe_arena_cu PRIVATE Catch2::Catch2WithMain)

add_executable(test_deepseek_moe_arena)
set_target_properties(test_deepseek_moe_arena PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_deepseek_moe_arena PRIVATE
    test_deepseek_moe_arena_cu lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_deepseek_moe_arena POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_moe_arena>"
    )
    add_custom_command(TARGET test_deepseek_moe_arena POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_moe_arena>"
    )
endif()

add_test(NAME test_deepseek_moe_arena COMMAND test_deepseek_moe_arena)

# ── Тест isolated sampling seam (lsxcommon::sample, GreedySampler) ──────────
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
    --generate-code=arch=compute_120,code=sm_120;
    --generate-code=arch=compute_121,code=sm_121;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_sampling_cu PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/thrust"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/cub"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/libcudacxx/include"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
    "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include"
)
target_compile_definitions(test_sampling_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
)
target_link_libraries(test_sampling_cu PRIVATE Catch2::Catch2WithMain)

add_executable(test_sampling)
set_target_properties(test_sampling PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_sampling PRIVATE
    test_sampling_cu lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_sampling POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_sampling>"
    )
    add_custom_command(TARGET test_sampling POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_sampling>"
    )
endif()

add_test(NAME test_sampling COMMAND test_sampling)

# ── Тест GLM fast loader (in-place BF16→FP16 kernel + pinned ring + арена) ──
add_library(test_glm_moe_load_fast_cu OBJECT test_glm_moe_load_fast.cu)
set_target_properties(test_glm_moe_load_fast_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_glm_moe_load_fast_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    --generate-code=arch=compute_120,code=sm_120;
    --generate-code=arch=compute_121,code=sm_121;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_glm_moe_load_fast_cu PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
    "${CMAKE_BINARY_DIR}/_deps/abseil-cpp-src"
)
target_compile_definitions(test_glm_moe_load_fast_cu PRIVATE
    LOGESTIX_HAS_CUDA
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_moe_load_fast_cu PRIVATE Catch2::Catch2WithMain)

add_executable(test_glm_moe_load_fast)
set_target_properties(test_glm_moe_load_fast PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_glm_moe_load_fast PRIVATE
    test_glm_moe_load_fast_cu lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_moe_load_fast POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_load_fast>"
    )
    add_custom_command(TARGET test_glm_moe_load_fast POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_load_fast>"
    )
endif()

add_test(NAME test_glm_moe_load_fast COMMAND test_glm_moe_load_fast)

# ── Тест жизненного цикла GLM-5.2 (ResourceGovernor load-once + leases) ────
add_library(test_glm_moe_model_lifecycle_cu OBJECT test_glm_moe_model_lifecycle.cu)
set_target_properties(test_glm_moe_model_lifecycle_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_glm_moe_model_lifecycle_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    --generate-code=arch=compute_120,code=sm_120;
    --generate-code=arch=compute_121,code=sm_121;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_glm_moe_model_lifecycle_cu PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
    "${CMAKE_BINARY_DIR}/_deps/abseil-cpp-src"
)
target_compile_definitions(test_glm_moe_model_lifecycle_cu PRIVATE
    LOGESTIX_HAS_CUDA
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_moe_model_lifecycle_cu PRIVATE Catch2::Catch2WithMain)

add_executable(test_glm_moe_model_lifecycle)
set_target_properties(test_glm_moe_model_lifecycle PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_glm_moe_model_lifecycle PRIVATE
    test_glm_moe_model_lifecycle_cu lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_moe_model_lifecycle POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_model_lifecycle>"
    )
    add_custom_command(TARGET test_glm_moe_model_lifecycle POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_model_lifecycle>"
    )
endif()

add_test(NAME test_glm_moe_model_lifecycle COMMAND test_glm_moe_model_lifecycle)

# ── Тест общих fast-load примитивов (GPU BF16→FP16 + pinned ring) ───────────
# Same OBJECT-lib pattern as test_glm_moe_load_fast_cu.
add_library(test_deepseek_moe_load_fast_cu OBJECT test_deepseek_moe_load_fast.cu)
set_target_properties(test_deepseek_moe_load_fast_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_deepseek_moe_load_fast_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    --generate-code=arch=compute_120,code=sm_120;
    --generate-code=arch=compute_121,code=sm_121;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_deepseek_moe_load_fast_cu PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
    "${CMAKE_BINARY_DIR}/_deps/abseil-cpp-src"
)
target_compile_definitions(test_deepseek_moe_load_fast_cu PRIVATE
    LOGESTIX_HAS_CUDA
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_deepseek_moe_load_fast_cu PRIVATE Catch2::Catch2WithMain)

add_executable(test_deepseek_moe_load_fast)
set_target_properties(test_deepseek_moe_load_fast PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_deepseek_moe_load_fast PRIVATE
    test_deepseek_moe_load_fast_cu lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_deepseek_moe_load_fast POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_moe_load_fast>"
    )
    add_custom_command(TARGET test_deepseek_moe_load_fast POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_moe_load_fast>"
    )
endif()

add_test(NAME test_deepseek_moe_load_fast COMMAND test_deepseek_moe_load_fast)

# �� ���� model-metadata vault path registry �����������������������������
add_executable(test_path_registry test_path_registry.cpp)
target_link_libraries(test_path_registry PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_path_registry POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_path_registry>"
    )
    add_custom_command(TARGET test_path_registry POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_path_registry>"
    )
endif()

add_test(NAME test_path_registry COMMAND test_path_registry)

# ── Тест модуля оценки VRAM-профиля (conversion-time) ───────────────────
add_executable(test_vram_estimate test_vram_estimate.cpp)
target_link_libraries(test_vram_estimate PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_vram_estimate POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_vram_estimate>"
    )
    add_custom_command(TARGET test_vram_estimate POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_vram_estimate>"
    )
endif()

add_test(NAME test_vram_estimate COMMAND test_vram_estimate)

add_executable(test_resource_governor test_resource_governor.cpp)
target_link_libraries(test_resource_governor PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_resource_governor POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_resource_governor>"
    )
    add_custom_command(TARGET test_resource_governor POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_resource_governor>"
    )
endif()

add_test(NAME test_resource_governor COMMAND test_resource_governor)

add_executable(test_hardware_profile test_hardware_profile.cpp)
target_link_libraries(test_hardware_profile PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_hardware_profile POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_hardware_profile>"
    )
    add_custom_command(TARGET test_hardware_profile POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_hardware_profile>"
    )
endif()

add_test(NAME test_hardware_profile COMMAND test_hardware_profile)

add_executable(test_lifecycle_plan test_lifecycle_plan.cpp)
target_link_libraries(test_lifecycle_plan PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_lifecycle_plan POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_lifecycle_plan>"
    )
    add_custom_command(TARGET test_lifecycle_plan POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_lifecycle_plan>"
    )
endif()

add_test(NAME test_lifecycle_plan COMMAND test_lifecycle_plan)

add_executable(test_runtime_tuner test_runtime_tuner.cpp)
target_link_libraries(test_runtime_tuner PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_runtime_tuner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_runtime_tuner>"
    )
    add_custom_command(TARGET test_runtime_tuner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_runtime_tuner>"
    )
endif()

add_test(NAME test_runtime_tuner COMMAND test_runtime_tuner)

# ── Тест Kimi INT4→BF16 деquant-ядра ──────────────────────────────────────
add_library(test_kimi_moe_int4_dequant_cu OBJECT test_kimi_moe_int4_dequant.cu)
set_target_properties(test_kimi_moe_int4_dequant_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_kimi_moe_int4_dequant_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    --generate-code=arch=compute_120,code=sm_120;
    --generate-code=arch=compute_121,code=sm_121;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_kimi_moe_int4_dequant_cu PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
)
target_compile_definitions(test_kimi_moe_int4_dequant_cu PRIVATE LOGESTIX_HAS_CUDA)
target_link_libraries(test_kimi_moe_int4_dequant_cu PRIVATE Catch2::Catch2WithMain)
target_link_libraries(test_kimi_moe_int4_dequant_cu PRIVATE lsxcommon_kimi_cu)

add_executable(test_kimi_moe_int4_dequant)
set_target_properties(test_kimi_moe_int4_dequant PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_kimi_moe_int4_dequant PRIVATE
    test_kimi_moe_int4_dequant_cu lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_kimi_moe_int4_dequant POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_kimi_moe_int4_dequant>"
    )
endif()

add_test(NAME test_kimi_moe_int4_dequant COMMAND test_kimi_moe_int4_dequant)

# ── Тест Kimi tiktoken-токенизатора ───────────────────────────────────────
add_executable(test_kimi_moe_tokenizer test_kimi_moe_tokenizer.cpp)
target_link_libraries(test_kimi_moe_tokenizer PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_kimi_moe_tokenizer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_kimi_moe_tokenizer>"
    )
endif()

add_test(NAME test_kimi_moe_tokenizer COMMAND test_kimi_moe_tokenizer)

# ── Тест Kimi pinned-загрузчика ───────────────────────────────────────────
add_executable(test_kimi_moe_loader test_kimi_moe_loader.cpp)
target_compile_definitions(test_kimi_moe_loader PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_kimi_moe_loader PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_kimi_moe_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_kimi_moe_loader>"
    )
    add_custom_command(TARGET test_kimi_moe_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_kimi_moe_loader>"
    )
endif()

add_test(NAME test_kimi_moe_loader COMMAND test_kimi_moe_loader)

# ── Тест Kimi MoE-роутера ─────────────────────────────────────────────────
add_library(test_kimi_moe_router_cu OBJECT test_kimi_moe_router.cu)
set_target_properties(test_kimi_moe_router_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_kimi_moe_router_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    --generate-code=arch=compute_120,code=sm_120;
    --generate-code=arch=compute_121,code=sm_121;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_kimi_moe_router_cu PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
)
target_compile_definitions(test_kimi_moe_router_cu PRIVATE LOGESTIX_HAS_CUDA)
target_link_libraries(test_kimi_moe_router_cu PRIVATE Catch2::Catch2WithMain)
target_link_libraries(test_kimi_moe_router_cu PRIVATE lsxcommon_kimi_cu)

add_executable(test_kimi_moe_router)
set_target_properties(test_kimi_moe_router PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_kimi_moe_router PRIVATE
    test_kimi_moe_router_cu lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_kimi_moe_router POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_kimi_moe_router>"
    )
endif()

add_test(NAME test_kimi_moe_router COMMAND test_kimi_moe_router)

# ── Тест Kimi MLA attention ───────────────────────────────────────────────
add_library(test_kimi_moe_attention_cu OBJECT test_kimi_moe_attention.cu)
set_target_properties(test_kimi_moe_attention_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_kimi_moe_attention_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    --generate-code=arch=compute_120,code=sm_120;
    --generate-code=arch=compute_121,code=sm_121;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_kimi_moe_attention_cu PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
)
target_compile_definitions(test_kimi_moe_attention_cu PRIVATE LOGESTIX_HAS_CUDA)
target_link_libraries(test_kimi_moe_attention_cu PRIVATE Catch2::Catch2WithMain)
target_link_libraries(test_kimi_moe_attention_cu PRIVATE lsxcommon_kimi_cu)

add_executable(test_kimi_moe_attention)
set_target_properties(test_kimi_moe_attention PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_kimi_moe_attention PRIVATE
    test_kimi_moe_attention_cu lsxcommon Catch2::Catch2WithMain CUDA::cublas)

if(WIN32)
    add_custom_command(TARGET test_kimi_moe_attention POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_kimi_moe_attention>"
    )
endif()

add_test(NAME test_kimi_moe_attention COMMAND test_kimi_moe_attention)

# ── Тест Kimi routed-expert FFN ───────────────────────────────────────────
add_library(test_kimi_moe_experts_cu OBJECT test_kimi_moe_experts.cu)
set_target_properties(test_kimi_moe_experts_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_kimi_moe_experts_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    --generate-code=arch=compute_120,code=sm_120;
    --generate-code=arch=compute_121,code=sm_121;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_kimi_moe_experts_cu PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
    "${CMAKE_BINARY_DIR}/_deps/abseil-cpp-src"
)
target_compile_definitions(test_kimi_moe_experts_cu PRIVATE
    LOGESTIX_HAS_CUDA
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_kimi_moe_experts_cu PRIVATE Catch2::Catch2WithMain)
target_link_libraries(test_kimi_moe_experts_cu PRIVATE lsxcommon_kimi_cu)

add_executable(test_kimi_moe_experts)
set_target_properties(test_kimi_moe_experts PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_kimi_moe_experts PRIVATE
    test_kimi_moe_experts_cu lsxcommon Catch2::Catch2WithMain CUDA::cublas)

if(WIN32)
    add_custom_command(TARGET test_kimi_moe_experts POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_kimi_moe_experts>"
    )
endif()

add_test(NAME test_kimi_moe_experts COMMAND test_kimi_moe_experts)

# ── Тест Kimi batched TC routed chain (fast-math; SKIP на не-sm_12x) ──────
add_library(test_kimi_moe_tc_batch_cu OBJECT test_kimi_moe_tc_batch.cu)
set_target_properties(test_kimi_moe_tc_batch_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_kimi_moe_tc_batch_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    --generate-code=arch=compute_120,code=sm_120;
    --generate-code=arch=compute_121,code=sm_121;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_kimi_moe_tc_batch_cu PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
    "${CMAKE_BINARY_DIR}/_deps/abseil-cpp-src"
)
target_compile_definitions(test_kimi_moe_tc_batch_cu PRIVATE
    LOGESTIX_HAS_CUDA
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_kimi_moe_tc_batch_cu PRIVATE Catch2::Catch2WithMain)
target_link_libraries(test_kimi_moe_tc_batch_cu PRIVATE lsxcommon_kimi_cu)

add_executable(test_kimi_moe_tc_batch)
set_target_properties(test_kimi_moe_tc_batch PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_kimi_moe_tc_batch PRIVATE
    test_kimi_moe_tc_batch_cu lsxcommon Catch2::Catch2WithMain CUDA::cublas)

if(WIN32)
    add_custom_command(TARGET test_kimi_moe_tc_batch POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_kimi_moe_tc_batch>"
    )
endif()

add_test(NAME test_kimi_moe_tc_batch COMMAND test_kimi_moe_tc_batch)

add_library(test_kimi_moe_tc_diag_cu OBJECT test_kimi_moe_tc_diag.cu)
set_target_properties(test_kimi_moe_tc_diag_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_kimi_moe_tc_diag_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    --generate-code=arch=compute_120,code=sm_120;
    --generate-code=arch=compute_121,code=sm_121;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_kimi_moe_tc_diag_cu PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
    "${CMAKE_BINARY_DIR}/_deps/abseil-cpp-src"
)
target_compile_definitions(test_kimi_moe_tc_diag_cu PRIVATE
    LOGESTIX_HAS_CUDA
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_kimi_moe_tc_diag_cu PRIVATE Catch2::Catch2WithMain)
target_link_libraries(test_kimi_moe_tc_diag_cu PRIVATE lsxcommon_kimi_cu)

add_executable(test_kimi_moe_tc_diag)
set_target_properties(test_kimi_moe_tc_diag PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_kimi_moe_tc_diag PRIVATE
    test_kimi_moe_tc_diag_cu lsxcommon Catch2::Catch2WithMain CUDA::cublas)

if(WIN32)
    add_custom_command(TARGET test_kimi_moe_tc_diag POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_kimi_moe_tc_diag>"
    )
endif()

add_test(NAME test_kimi_moe_tc_diag COMMAND test_kimi_moe_tc_diag)

# ── Тест Kimi модели (фабрика + lifecycle) ────────────────────────────────
add_executable(test_kimi_moe_model test_kimi_moe_model.cpp)
target_compile_definitions(test_kimi_moe_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_kimi_moe_model PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_kimi_moe_model POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_kimi_moe_model>"
    )
endif()

add_test(NAME test_kimi_moe_model COMMAND test_kimi_moe_model)

# ── Тест Kimi SelfCheck (реальные веса; SKIP без модели) ──────────────────
add_executable(test_kimi_moe_selfcheck test_kimi_moe_selfcheck.cpp)
target_compile_definitions(test_kimi_moe_selfcheck PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_kimi_moe_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_kimi_moe_selfcheck POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_kimi_moe_selfcheck>"
    )
    add_custom_command(TARGET test_kimi_moe_selfcheck POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_kimi_moe_selfcheck>"
    )
endif()

add_test(NAME test_kimi_moe_selfcheck COMMAND test_kimi_moe_selfcheck)
set_tests_properties(test_kimi_moe_selfcheck PROPERTIES SKIP_RETURN_CODE 4)

# ── Тест Kimi perf baseline (реальные веса; SKIP без модели) ──────────────
add_executable(test_blackwell_perf_kimi test_blackwell_perf_kimi.cpp)
target_compile_definitions(test_blackwell_perf_kimi PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_blackwell_perf_kimi PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_blackwell_perf_kimi POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_blackwell_perf_kimi>"
    )
endif()

add_test(NAME test_blackwell_perf_kimi COMMAND test_blackwell_perf_kimi)
set_tests_properties(test_blackwell_perf_kimi PROPERTIES SKIP_RETURN_CODE 4)

# ── Тест Kimi full inference (реальные веса; SKIP без модели) ─────────────
add_executable(test_kimi_moe_inference test_kimi_moe_inference.cpp)
target_compile_definitions(test_kimi_moe_inference PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_include_directories(test_kimi_moe_inference PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src"
)
target_link_libraries(test_kimi_moe_inference PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_kimi_moe_inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_kimi_moe_inference>"
    )
endif()

add_test(NAME test_kimi_moe_inference COMMAND test_kimi_moe_inference)
set_tests_properties(test_kimi_moe_inference PROPERTIES SKIP_RETURN_CODE 4)

# ── Тест Kimi конвертации на реальном чекпойнте (SKIP без модели) ─────────
add_library(test_kimi_moe_int4_real_cu OBJECT test_kimi_moe_int4_real.cu)
set_target_properties(test_kimi_moe_int4_real_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_kimi_moe_int4_real_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    --generate-code=arch=compute_120,code=sm_120;
    --generate-code=arch=compute_121,code=sm_121;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_kimi_moe_int4_real_cu PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
    "${CMAKE_BINARY_DIR}/_deps/abseil-cpp-src"
    "${CMAKE_SOURCE_DIR}/third_party/simdjson/include"
)
target_compile_definitions(test_kimi_moe_int4_real_cu PRIVATE LOGESTIX_HAS_CUDA)
target_link_libraries(test_kimi_moe_int4_real_cu PRIVATE Catch2::Catch2WithMain)
target_link_libraries(test_kimi_moe_int4_real_cu PRIVATE lsxcommon_kimi_cu)

add_executable(test_kimi_moe_int4_real)
set_target_properties(test_kimi_moe_int4_real PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_kimi_moe_int4_real PRIVATE
    test_kimi_moe_int4_real_cu lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_kimi_moe_int4_real POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_kimi_moe_int4_real>"
    )
endif()

add_test(NAME test_kimi_moe_int4_real COMMAND test_kimi_moe_int4_real)
set_tests_properties(test_kimi_moe_int4_real PROPERTIES SKIP_RETURN_CODE 4)

# ── Тесты Qwen3.5-35B-A3B (MoE) ───────────────────────────────────────────
add_executable(test_qwen_moe_convert test_qwen_moe_convert.cpp)
target_compile_definitions(test_qwen_moe_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_moe_convert PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_qwen_moe_convert POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_moe_convert>"
    )
    add_custom_command(TARGET test_qwen_moe_convert POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_moe_convert>"
    )
endif()

add_test(NAME test_qwen_moe_convert COMMAND test_qwen_moe_convert)
# Synthetic tiny-geometry fixture: opt out of the frozen-geometry hard gate.
set_tests_properties(test_qwen_moe_convert PROPERTIES
    ENVIRONMENT "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1")

# ── Тесты Qwen3.6-35B-A3B (MoE, variant 36) ───────────────────────────────
add_executable(test_qwen_moe_36_convert test_qwen_moe_36_convert.cpp)
target_compile_definitions(test_qwen_moe_36_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_moe_36_convert PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_qwen_moe_36_convert POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_moe_36_convert>"
    )
    add_custom_command(TARGET test_qwen_moe_36_convert POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_moe_36_convert>"
    )
endif()

add_test(NAME test_qwen_moe_36_convert COMMAND test_qwen_moe_36_convert)
set_tests_properties(test_qwen_moe_36_convert PROPERTIES
    ENVIRONMENT "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1")

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
    --generate-code=arch=compute_120,code=sm_120;
    --generate-code=arch=compute_121,code=sm_121;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_qwen_moe_kernels_cu PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/thrust"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/cub"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/libcudacxx/include"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
    "${CMAKE_BINARY_DIR}/_deps/abseil-cpp-src"
    "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include"
    "${CMAKE_SOURCE_DIR}/third_party/flatbuffers/include"
    "${CMAKE_SOURCE_DIR}/third_party/libzmq/include"
    "${CMAKE_BINARY_DIR}/third_party/libzmq"
    "${CMAKE_SOURCE_DIR}/libs/lsxnetwork/include"
    "${CMAKE_BINARY_DIR}/libs/lsxnetwork/generated"
    "${CMAKE_SOURCE_DIR}/third_party/simdjson/include"
)
target_compile_definitions(test_qwen_moe_kernels_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
    SPDLOG_COMPILED_LIB
    ZMQ_STATIC
    SIMDJSON_THREADS_ENABLED=1
)
target_link_libraries(test_qwen_moe_kernels_cu PRIVATE Catch2::Catch2WithMain)

add_executable(test_qwen_moe_kernels)
set_target_properties(test_qwen_moe_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_qwen_moe_kernels PRIVATE
    test_qwen_moe_kernels_cu lsxcommon_qwen_moe_cu lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_qwen_moe_kernels POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_moe_kernels>"
    )
    add_custom_command(TARGET test_qwen_moe_kernels POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_moe_kernels>"
    )
endif()

add_test(NAME test_qwen_moe_kernels COMMAND test_qwen_moe_kernels)

add_executable(test_qwen_moe_loader test_qwen_moe_loader.cpp)
target_compile_definitions(test_qwen_moe_loader PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_moe_loader PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_qwen_moe_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_moe_loader>"
    )
    add_custom_command(TARGET test_qwen_moe_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_moe_loader>"
    )
endif()

add_test(NAME test_qwen_moe_loader COMMAND test_qwen_moe_loader)
set_tests_properties(test_qwen_moe_loader PROPERTIES
    ENVIRONMENT "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1")

add_executable(test_qwen_moe_model test_qwen_moe_model.cpp)
target_include_directories(test_qwen_moe_model PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src")
target_compile_definitions(test_qwen_moe_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_qwen_moe_model PRIVATE lsxcommon Catch2::Catch2WithMain)
if(WIN32)
    add_custom_command(TARGET test_qwen_moe_model POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_moe_model>"
    )
endif()
add_test(NAME test_qwen_moe_model COMMAND test_qwen_moe_model)
set_tests_properties(test_qwen_moe_model PROPERTIES
    ENVIRONMENT "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1")

add_executable(test_qwen_moe_36_model test_qwen_moe_36_model.cpp)
target_include_directories(test_qwen_moe_36_model PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src")
target_compile_definitions(test_qwen_moe_36_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_qwen_moe_36_model PRIVATE lsxcommon Catch2::Catch2WithMain)
if(WIN32)
    add_custom_command(TARGET test_qwen_moe_36_model POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_moe_36_model>"
    )
endif()
add_test(NAME test_qwen_moe_36_model COMMAND test_qwen_moe_36_model)
set_tests_properties(test_qwen_moe_36_model PROPERTIES
    ENVIRONMENT "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1")

add_executable(test_qwen_moe_36_selfcheck test_qwen_moe_36_selfcheck.cpp)
target_include_directories(test_qwen_moe_36_selfcheck PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src")
target_compile_definitions(test_qwen_moe_36_selfcheck PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_moe_36_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
if(WIN32)
    add_custom_command(TARGET test_qwen_moe_36_selfcheck POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_moe_36_selfcheck>"
    )
endif()
add_test(NAME test_qwen_moe_36_selfcheck COMMAND test_qwen_moe_36_selfcheck)
set_tests_properties(test_qwen_moe_36_selfcheck PROPERTIES SKIP_RETURN_CODE 4)

add_executable(test_qwen_moe_selfcheck test_qwen_moe_selfcheck.cpp)
target_include_directories(test_qwen_moe_selfcheck PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src")
target_compile_definitions(test_qwen_moe_selfcheck PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_moe_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)
if(WIN32)
    add_custom_command(TARGET test_qwen_moe_selfcheck POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_moe_selfcheck>"
    )
endif()
add_test(NAME test_qwen_moe_selfcheck COMMAND test_qwen_moe_selfcheck)
set_tests_properties(test_qwen_moe_selfcheck PROPERTIES SKIP_RETURN_CODE 4)

# ── Тест NVFP4-упаковки экспертов (passthrough + GPU quantize contract) ───
add_executable(test_nvfp4_pack test_nvfp4_pack.cpp)
target_link_libraries(test_nvfp4_pack PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_nvfp4_pack POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_nvfp4_pack>"
    )
endif()

add_test(NAME test_nvfp4_pack COMMAND test_nvfp4_pack)
set_tests_properties(test_nvfp4_pack PROPERTIES SKIP_RETURN_CODE 4)

# Guard: model code file naming convention (host-only, no lsxcommon deps).
add_executable(test_code_naming_guard test_code_naming_guard.cpp)
target_link_libraries(test_code_naming_guard PRIVATE Catch2::Catch2WithMain)
file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}" LSX_SRC_DIR_FWD)
target_compile_definitions(test_code_naming_guard
    PRIVATE "LSX_SOURCE_DIR=\"${LSX_SRC_DIR_FWD}\"")
add_test(NAME test_code_naming_guard COMMAND test_code_naming_guard)

# Guard: absl::StrFormat & friends are printf-style; fmt-style "{}" in their
# format strings prints literally (checker inert on GCC). Host-only static scan.
add_executable(test_absl_format_guard test_absl_format_guard.cpp)
target_link_libraries(test_absl_format_guard PRIVATE Catch2::Catch2WithMain
    absl::str_format)
file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}" LSX_SRC_DIR_FWD_FMT)
target_compile_definitions(test_absl_format_guard
    PRIVATE "LSX_SOURCE_DIR=\"${LSX_SRC_DIR_FWD_FMT}\"")
add_test(NAME test_absl_format_guard COMMAND test_absl_format_guard)

# Kimi MoE expert cache bookkeeping (host-only, pure logic, no CUDA).
add_executable(test_kimi_moe_expert_cache test_kimi_moe_expert_cache.cpp)
target_link_libraries(test_kimi_moe_expert_cache PRIVATE lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_kimi_moe_expert_cache PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include")
add_test(NAME test_kimi_moe_expert_cache COMMAND test_kimi_moe_expert_cache)

# ── Тесты Qwen3.8-Flash-Next (qwen4_exp MoE) ──────────────────────────────
add_executable(test_qwen_moe_exp_convert test_qwen_moe_exp_convert.cpp)
target_compile_definitions(test_qwen_moe_exp_convert PRIVATE
    LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1)
target_link_libraries(test_qwen_moe_exp_convert PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_qwen_moe_exp_convert PRIVATE
    ${CMAKE_SOURCE_DIR}/libs/lsxcommon/src)

if(WIN32)
    add_custom_command(TARGET test_qwen_moe_exp_convert POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_moe_exp_convert>"
    )
    add_custom_command(TARGET test_qwen_moe_exp_convert POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_moe_exp_convert>"
    )
endif()

add_test(NAME test_qwen_moe_exp_convert COMMAND test_qwen_moe_exp_convert)
# Synthetic tiny-geometry fixture: opt out of the frozen-geometry hard gate.
set_tests_properties(test_qwen_moe_exp_convert PROPERTIES
    ENVIRONMENT "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1")

add_executable(test_qwen_moe_exp_loader test_qwen_moe_exp_loader.cpp)
target_compile_definitions(test_qwen_moe_exp_loader PRIVATE
    LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1)
target_link_libraries(test_qwen_moe_exp_loader PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_qwen_moe_exp_loader PRIVATE
    ${CMAKE_SOURCE_DIR}/libs/lsxcommon/src)

if(WIN32)
    add_custom_command(TARGET test_qwen_moe_exp_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_moe_exp_loader>"
    )
    add_custom_command(TARGET test_qwen_moe_exp_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_moe_exp_loader>"
    )
endif()

add_test(NAME test_qwen_moe_exp_loader COMMAND test_qwen_moe_exp_loader)
set_tests_properties(test_qwen_moe_exp_loader PROPERTIES
    ENVIRONMENT "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1")

# test_qwen_moe_exp_kernels — qwen4_exp kernel tests (hc GatedResidual, QSA
# indexer, PLE inject) against host-replica oracles.
add_library(test_qwen_moe_exp_kernels_cu OBJECT test_qwen_moe_exp_kernels.cu)
set_target_properties(test_qwen_moe_exp_kernels_cu PROPERTIES
    CUDA_ARCHITECTURES "120-real;121-real"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(test_qwen_moe_exp_kernels_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(test_qwen_moe_exp_kernels_cu PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/thrust"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/cub"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/libcudacxx/include"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
    "${CMAKE_BINARY_DIR}/_deps/abseil-cpp-src"
    "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include"
    "${CMAKE_SOURCE_DIR}/third_party/simdjson/include"
)
target_link_libraries(test_qwen_moe_exp_kernels_cu PRIVATE
    Catch2::Catch2WithMain)

add_executable(test_qwen_moe_exp_kernels)
set_target_properties(test_qwen_moe_exp_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_qwen_moe_exp_kernels PRIVATE
    test_qwen_moe_exp_kernels_cu lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_qwen_moe_exp_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/libs/lsxcommon/src)
if(WIN32)
    add_custom_command(TARGET test_qwen_moe_exp_kernels POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_moe_exp_kernels>"
    )
endif()
add_test(NAME test_qwen_moe_exp_kernels COMMAND test_qwen_moe_exp_kernels)

# test_glm_moe_flash_kernels — GLM-5.3-Flash kernel tests (conv1d, KDA
# per-channel-decay gated delta, router) against host references.
add_library(test_glm_moe_flash_kernels_cu OBJECT test_glm_moe_flash_kernels.cu)
set_target_properties(test_glm_moe_flash_kernels_cu PROPERTIES
    CUDA_ARCHITECTURES "120-real;121-real"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(test_glm_moe_flash_kernels_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr;-fmad=false>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/utf-8>
)
target_include_directories(test_glm_moe_flash_kernels_cu PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/thrust"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/cub"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/libcudacxx/include"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
    "${CMAKE_BINARY_DIR}/_deps/abseil-cpp-src"
    "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include"
    "${CMAKE_SOURCE_DIR}/third_party/simdjson/include"
)
target_link_libraries(test_glm_moe_flash_kernels_cu PRIVATE
    Catch2::Catch2WithMain)

add_executable(test_glm_moe_flash_kernels)
set_target_properties(test_glm_moe_flash_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_glm_moe_flash_kernels PRIVATE
    test_glm_moe_flash_kernels_cu lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_glm_moe_flash_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/libs/lsxcommon/src)
if(WIN32)
    add_custom_command(TARGET test_glm_moe_flash_kernels POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_flash_kernels>"
    )
endif()
add_test(NAME test_glm_moe_flash_kernels COMMAND test_glm_moe_flash_kernels)

add_executable(test_glm_moe_flash_loader test_glm_moe_flash_loader.cpp)
target_compile_definitions(test_glm_moe_flash_loader PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_glm_moe_flash_loader PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_glm_moe_flash_loader PRIVATE
    ${CMAKE_SOURCE_DIR}/libs/lsxcommon/src)
if(WIN32)
    add_custom_command(TARGET test_glm_moe_flash_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_flash_loader>"
    )
endif()
add_test(NAME test_glm_moe_flash_loader COMMAND test_glm_moe_flash_loader)
# Synthetic-converted fixture: the load-side frozen-geometry gate is open
# only under this env (loader-test discipline: no vault loads).
set_tests_properties(test_glm_moe_flash_loader PROPERTIES
    ENVIRONMENT "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1")

add_executable(test_glm_moe_flash_model test_glm_moe_flash_model.cpp)
target_compile_definitions(test_glm_moe_flash_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_glm_moe_flash_model PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_glm_moe_flash_model PRIVATE
    ${CMAKE_SOURCE_DIR}/libs/lsxcommon/src)
if(WIN32)
    add_custom_command(TARGET test_glm_moe_flash_model POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_flash_model>"
    )
endif()
add_test(NAME test_glm_moe_flash_model COMMAND test_glm_moe_flash_model)
# Synthetic-converted fixture: load-side gates open only under this env.
set_tests_properties(test_glm_moe_flash_model PROPERTIES
    ENVIRONMENT "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1")

add_executable(test_glm_moe_flash_selfcheck test_glm_moe_flash_selfcheck.cpp)
target_link_libraries(test_glm_moe_flash_selfcheck PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_glm_moe_flash_selfcheck PRIVATE
    ${CMAKE_SOURCE_DIR}/libs/lsxcommon/src)
if(WIN32)
    add_custom_command(TARGET test_glm_moe_flash_selfcheck POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_flash_selfcheck>"
    )
endif()
add_test(NAME test_glm_moe_flash_selfcheck COMMAND test_glm_moe_flash_selfcheck)
set_tests_properties(test_glm_moe_flash_selfcheck PROPERTIES SKIP_RETURN_CODE 4)

add_executable(test_qwen_moe_exp_model test_qwen_moe_exp_model.cpp)
target_compile_definitions(test_qwen_moe_exp_model PRIVATE
    LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1)
target_link_libraries(test_qwen_moe_exp_model PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_qwen_moe_exp_model PRIVATE
    ${CMAKE_SOURCE_DIR}/libs/lsxcommon/src)
if(WIN32)
    add_custom_command(TARGET test_qwen_moe_exp_model POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_moe_exp_model>"
    )
    add_custom_command(TARGET test_qwen_moe_exp_model POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_moe_exp_model>"
    )
endif()
add_test(NAME test_qwen_moe_exp_model COMMAND test_qwen_moe_exp_model)
set_tests_properties(test_qwen_moe_exp_model PROPERTIES
    ENVIRONMENT "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1")

add_executable(test_qwen_moe_exp_selfcheck test_qwen_moe_exp_selfcheck.cpp)
target_link_libraries(test_qwen_moe_exp_selfcheck PRIVATE
    lsxcommon Catch2::Catch2WithMain)
target_include_directories(test_qwen_moe_exp_selfcheck PRIVATE
    ${CMAKE_SOURCE_DIR}/libs/lsxcommon/src)
if(WIN32)
    add_custom_command(TARGET test_qwen_moe_exp_selfcheck POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_moe_exp_selfcheck>"
    )
    add_custom_command(TARGET test_qwen_moe_exp_selfcheck POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_moe_exp_selfcheck>"
    )
endif()
add_test(NAME test_qwen_moe_exp_selfcheck COMMAND test_qwen_moe_exp_selfcheck)
set_tests_properties(test_qwen_moe_exp_selfcheck PROPERTIES SKIP_RETURN_CODE 4)

# Guard: model independence — unique frozen passports, no cross-model
# geometry borrowing (host-only, no lsxcommon deps).
add_executable(test_model_independence_guard test_model_independence_guard.cpp)
target_link_libraries(test_model_independence_guard PRIVATE Catch2::Catch2WithMain
    absl::btree)
file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}" LSX_SRC_DIR_FWD2)
target_compile_definitions(test_model_independence_guard
    PRIVATE "LSX_SOURCE_DIR=\"${LSX_SRC_DIR_FWD2}\"")
add_test(NAME test_model_independence_guard COMMAND test_model_independence_guard)

# Guard: ctest category-label discipline (fast | inference | perf).
# Host-only static scan of tests/CMakeLists.txt (see spec 2026-09-10).
add_executable(test_ctest_labels_guard test_ctest_labels_guard.cpp)
target_link_libraries(test_ctest_labels_guard PRIVATE Catch2::Catch2WithMain)
file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}" LSX_SRC_DIR_FWD_LABELS)
target_compile_definitions(test_ctest_labels_guard
    PRIVATE "LSX_SOURCE_DIR=\"${LSX_SRC_DIR_FWD_LABELS}\"")
add_test(NAME test_ctest_labels_guard COMMAND test_ctest_labels_guard)

# Guard: model test-suite autonomy + full-lifecycle coverage. Host-only static
# scan: per-model lifecycle registry (convert -> load -> model -> kernels ->
# selfcheck), SKIP_RETURN_CODE 4 on selfcheck targets, and a cross-family
# fixture-borrowing ban (test-side mirror of model independence).
add_executable(test_model_suite_guard test_model_suite_guard.cpp)
target_link_libraries(test_model_suite_guard PRIVATE Catch2::Catch2WithMain)
file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}" LSX_SRC_DIR_FWD_SUITE)
target_compile_definitions(test_model_suite_guard
    PRIVATE "LSX_SOURCE_DIR=\"${LSX_SRC_DIR_FWD_SUITE}\"")
add_test(NAME test_model_suite_guard COMMAND test_model_suite_guard)

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
    test_geometry_guard
    test_gpu_perf
    test_perf_baseline_logic
    test_schema_views
    test_deepseek_moe_loader
    test_deepseek_moe_model
    test_deepseek_moe_v41_model
    test_deepseek_moe_v41_loader
    test_expert_store
    test_glm_moe_expert_cache
    test_model_integrity
    test_model_factory
    test_generation_cli
    test_glm_moe_prompt_framing
    test_gemma_prompt_framing
    test_model_runtime
    test_model_engine
    test_tokenizer
    test_mosaic_network
    test_mosaic_frontend_route
    test_token_translator
    test_network_translator_weld
    test_integrated_pipeline
    test_path_registry
    test_vram_estimate
    test_resource_governor
    test_hardware_profile
    test_lifecycle_plan
    test_runtime_tuner
    test_kimi_moe_tokenizer
    test_code_naming_guard
    test_absl_format_guard
    test_kimi_moe_expert_cache
    test_model_independence_guard
    test_ctest_labels_guard
    test_model_suite_guard
    PROPERTIES LABELS fast)

set_tests_properties(
    test_qwen_dense_convert
    test_qwen_dense_converter_integrity
    test_glm_moe_convert
    test_glm_moe_flash_convert
    test_kimi_moe_convert
    test_gemma_moe_convert
    test_qwen_moe_convert
    test_qwen_moe_36_convert
    test_qwen_moe_exp_convert
    test_nvfp4_pack
    PROPERTIES LABELS convert)

set_tests_properties(
    test_sampling
    test_lsx_nvfp4_dispatch
    test_perf_knobs_apply
    test_blackwell_tc_mma
    test_lsx_kernels
    test_blackwell_micro_gemm
    test_deepseek_moe_inference_kernels
    test_deepseek_moe_moe
    test_deepseek_moe_arena
    test_qwen_moe_kernels
    test_qwen_moe_exp_kernels
    test_glm_moe_flash_kernels
    test_kimi_moe_int4_dequant
    test_kimi_moe_int4_real
    test_kimi_moe_router
    test_kimi_moe_attention
    test_kimi_moe_experts
    test_kimi_moe_tc_batch
    test_kimi_moe_tc_diag
    test_glm_moe_load_fast
    test_deepseek_moe_load_fast
    test_glm_moe_model_lifecycle
    test_qwen_dense_loader
    test_qwen_dense_model
    test_qwen_dense_27b_model
    test_qwen_dense_kv_cache
    test_qwen_dense_kv_parity
    test_qwen_dense_selfcheck
    test_qwen_dense_27b_selfcheck
    test_qwen_dense_norm
    test_qwen_dense_rotary
    test_qwen_dense_shuffle_reduce
    test_qwen_dense_mlp
    test_qwen_dense_nvfp4
    test_qwen_nvfp4_tile_gemm
    test_qwen_nvfp4_gemv
    test_qwen_dense_full_attn
    test_qwen_dense_full_attn_parallel
    test_qwen_dense_linear_attn
    test_qwen_dense_fused_mlp
    test_qwen_dense_fused_linear_attn
    test_qwen_moe_loader
    test_qwen_moe_model
    test_qwen_moe_selfcheck
    test_qwen_moe_36_model
    test_qwen_moe_36_selfcheck
    test_qwen_moe_exp_loader
    test_qwen_moe_exp_model
    test_qwen_moe_exp_selfcheck
    test_glm_moe_loader
    test_glm_moe_model
    test_glm_moe_selfcheck
    test_glm_moe_inference
    test_glm_moe_full_compute
    test_glm_moe_prefill_seqp
    test_glm_moe_prompt_output
    test_glm_moe_attention
    test_glm_moe_dense_compute
    test_glm_moe_layer_compute
    test_glm_moe_experts
    test_glm_moe_moe_router
    test_glm_moe_moe_compute
    test_glm_moe_flash_loader
    test_glm_moe_flash_model
    test_glm_moe_flash_selfcheck
    test_deepseek_moe_prompt_output
    test_deepseek_moe_selfcheck
    test_gemma_moe_loader
    test_gemma_moe_model
    test_gemma_moe_selfcheck
    test_gemma_moe_norm
    test_gemma_moe_rotary
    test_gemma_moe_attn
    test_gemma_moe_moe
    test_kimi_moe_loader
    test_kimi_moe_model
    test_kimi_moe_selfcheck
    test_kimi_moe_inference
    test_cuda_toolkit_guard
    PROPERTIES LABELS inference)

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
    test_qwen_dense_convert
    test_qwen_dense_converter_integrity
    test_glm_moe_convert
    test_glm_moe_flash_convert
    test_kimi_moe_convert
    test_gemma_moe_convert
    test_qwen_moe_convert
    test_qwen_moe_36_convert
    test_qwen_moe_exp_convert
    test_nvfp4_pack
    PROPERTIES TIMEOUT 120)
