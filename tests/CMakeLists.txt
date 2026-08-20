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

# ── Тест DeepSeek round-trip загрузчика ─────────────────────────────────
add_executable(test_deepseek_loader test_deepseek_loader.cpp)
target_link_libraries(test_deepseek_loader PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_deepseek_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_loader>"
    )
    add_custom_command(TARGET test_deepseek_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_loader>"
    )
endif()

add_test(NAME test_deepseek_loader COMMAND test_deepseek_loader)

# ── Тест Qwen3.5-9B конвертера (парсер имён + индекс + round-trip) ────────
add_executable(test_qwen_convert test_qwen_convert.cpp)
target_compile_definitions(test_qwen_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_convert PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_qwen_convert POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_convert>"
    )
    add_custom_command(TARGET test_qwen_convert POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_convert>"
    )
endif()

add_test(NAME test_qwen_convert COMMAND test_qwen_convert)

# ── Тест Qwen3.5-9B загрузчика (pinned upload, topology, layer dispatch) ────
add_executable(test_qwen_loader test_qwen_loader.cpp)
target_compile_definitions(test_qwen_loader PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_loader PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_qwen_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_loader>"
    )
    add_custom_command(TARGET test_qwen_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_loader>"
    )
endif()

add_test(NAME test_qwen_loader COMMAND test_qwen_loader)

# ── CUDA kernel tests (link lsxcommon; kernels live in lsxcommon_qwen_cu) ──
foreach(qt test_qwen_norm test_qwen_rotary)
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
    test_qwen_mlp.cu
    test_qwen_nvfp4_dense.cu
    test_qwen_full_attn.cu
    test_qwen_linear_attn.cu
)
set_target_properties(test_qwen_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_qwen_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    --generate-code=arch=compute_120,code=sm_120;
    --generate-code=arch=compute_121,code=sm_121;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_qwen_cu PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/thrust"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/cub"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/libcudacxx/include"
)
target_compile_definitions(test_qwen_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
)
target_link_libraries(test_qwen_cu PRIVATE Catch2::Catch2WithMain)

foreach(qt test_qwen_mlp test_qwen_nvfp4_dense test_qwen_full_attn test_qwen_linear_attn)
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
add_executable(test_qwen_model test_qwen_model.cpp)
target_compile_definitions(test_qwen_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_model PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_qwen_model POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_model>"
    )
    add_custom_command(TARGET test_qwen_model POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_model>"
    )
endif()

add_test(NAME test_qwen_model COMMAND test_qwen_model)

# ── Инкрементальный KV-кэш Qwen (Task 7) ─────────────────────────────────
add_executable(test_qwen_kv_cache test_qwen_kv_cache.cpp)
target_compile_definitions(test_qwen_kv_cache PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_kv_cache PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_qwen_kv_cache POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_kv_cache>"
    )
    add_custom_command(TARGET test_qwen_kv_cache POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_kv_cache>"
    )
endif()

add_test(NAME test_qwen_kv_cache COMMAND test_qwen_kv_cache)
set_tests_properties(test_qwen_kv_cache PROPERTIES SKIP_RETURN_CODE 4)

# ── Bit-exactness gate: incremental KV vs full recompute (synthetic) ────────
add_executable(test_qwen_kv_parity test_qwen_kv_parity.cpp)
target_compile_definitions(test_qwen_kv_parity PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_kv_parity PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_qwen_kv_parity POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_kv_parity>"
    )
    add_custom_command(TARGET test_qwen_kv_parity POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_kv_parity>"
    )
endif()

add_test(NAME test_qwen_kv_parity COMMAND test_qwen_kv_parity)

# ── Тест Qwen3.8-27B модели ──────────────────────────────────────────────
add_executable(test_qwen_3_8_27b_model test_qwen_3_8_27b_model.cpp)
target_compile_definitions(test_qwen_3_8_27b_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_3_8_27b_model PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_qwen_3_8_27b_model POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_3_8_27b_model>"
    )
    add_custom_command(TARGET test_qwen_3_8_27b_model POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_3_8_27b_model>"
    )
endif()

add_test(NAME test_qwen_3_8_27b_model COMMAND test_qwen_3_8_27b_model)

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
add_executable(test_qwen_converter_integrity test_qwen_converter_integrity.cpp)
target_compile_definitions(test_qwen_converter_integrity PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_qwen_converter_integrity PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_qwen_converter_integrity POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_qwen_converter_integrity>"
    )
    add_custom_command(TARGET test_qwen_converter_integrity POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_qwen_converter_integrity>"
    )
endif()

add_test(NAME test_qwen_converter_integrity COMMAND test_qwen_converter_integrity)

# ── Тест GLM-5.2 конвертера (парсер имён тензоров) ────────────────────────
add_executable(test_glm_convert test_glm_convert.cpp)
target_compile_definitions(test_glm_convert PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_convert PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_convert POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_convert>"
    )
    add_custom_command(TARGET test_glm_convert POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_convert>"
    )
endif()

add_test(NAME test_glm_convert COMMAND test_glm_convert)

# ── Тест GLM-5.2 загрузчика (pinned BF16→FP16 upload) ─────────────────────
add_executable(test_glm_loader test_glm_loader.cpp)
target_compile_definitions(test_glm_loader PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_loader PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_loader>"
    )
    add_custom_command(TARGET test_glm_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_loader>"
    )
endif()

add_test(NAME test_glm_loader COMMAND test_glm_loader)

# ── Тест GLM-5.2 модели (фабрика + IModel) ─────────────────────────────────
add_executable(test_glm_model test_glm_model.cpp)
target_compile_definitions(test_glm_model PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_model PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_model POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_model>"
    )
    add_custom_command(TARGET test_glm_model POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_model>"
    )
endif()

add_test(NAME test_glm_model COMMAND test_glm_model)

# ── Тест GLM-5.2 SelfCheck (шаг 8 протокола): инкрементальный 1->2-токенный
#    селфчек на реальной модели, минимум 50 циклов стресса, бит-точная
#    детерминированность final_logits, exit 0. SKIP без чекпоинта.
add_executable(test_glm_selfcheck test_glm_selfcheck.cpp)
target_compile_definitions(test_glm_selfcheck PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_selfcheck PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_selfcheck POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_selfcheck>"
    )
    add_custom_command(TARGET test_glm_selfcheck POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_selfcheck>"
    )
endif()

add_test(NAME test_glm_selfcheck COMMAND test_glm_selfcheck)
set_tests_properties(test_glm_selfcheck PROPERTIES SKIP_RETURN_CODE 4)

# ── Тест GLM-5.2 seq-P prefill parity: MoE-слои 3..77 обязаны держать
#    block_rows = P (полная длина промпта), каждый промпт-токен занимает один
#    IndexShare KV-слот. FAIL при коллапсе строк (kv_slots_used != P + gen),
#    при расхождении argmax final_logits с первым сгенерированным токеном или
#    при потере бит-точности между двумя идентичными прогонами. SKIP без
#    чекпоинта.
add_executable(test_glm_prefill_seqp test_glm_prefill_seqp.cpp)
target_compile_definitions(test_glm_prefill_seqp PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_prefill_seqp PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_prefill_seqp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_prefill_seqp>"
    )
    add_custom_command(TARGET test_glm_prefill_seqp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_prefill_seqp>"
    )
endif()

add_test(NAME test_glm_prefill_seqp COMMAND test_glm_prefill_seqp)
set_tests_properties(test_glm_prefill_seqp PROPERTIES SKIP_RETURN_CODE 4)

# ── Тест GLM-5.2 осмысленного вывода по --run (Infer output_text) ───────────
add_executable(test_glm_prompt_output test_glm_prompt_output.cpp)
target_compile_definitions(test_glm_prompt_output PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_prompt_output PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_prompt_output POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_prompt_output>"
    )
    add_custom_command(TARGET test_glm_prompt_output POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_prompt_output>"
    )
endif()

add_test(NAME test_glm_prompt_output COMMAND test_glm_prompt_output)
set_tests_properties(test_glm_prompt_output PROPERTIES SKIP_RETURN_CODE 4)

# ── Тест DeepSeek-V4-Flash осмысленного вывода (Infer output_text) ────────
add_executable(test_deepseek_prompt_output test_deepseek_prompt_output.cpp)
target_link_libraries(test_deepseek_prompt_output PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_deepseek_prompt_output POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_prompt_output>"
    )
    add_custom_command(TARGET test_deepseek_prompt_output POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_prompt_output>"
    )
endif()

add_test(NAME test_deepseek_prompt_output COMMAND test_deepseek_prompt_output)
set_tests_properties(test_deepseek_prompt_output PROPERTIES SKIP_RETURN_CODE 4)

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
add_executable(test_glm_prompt_framing test_glm_prompt_framing.cpp)
target_compile_definitions(test_glm_prompt_framing PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_glm_prompt_framing PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_prompt_framing POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_prompt_framing>"
    )
    add_custom_command(TARGET test_glm_prompt_framing POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_prompt_framing>"
    )
endif()

add_test(NAME test_glm_prompt_framing COMMAND test_glm_prompt_framing)

# ── Тест ModelBase (tokenizer-db load + infer guards, host-only) ─────────
add_executable(test_model_runtime test_model_runtime.cpp)
target_compile_definitions(test_model_runtime PRIVATE
    TOKENIZER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_link_libraries(test_model_runtime PRIVATE lsxcommon Catch2::Catch2WithMain)

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
add_executable(test_glm_compute test_glm_compute.cpp)
target_link_libraries(test_glm_compute PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_compute POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_compute>"
    )
    add_custom_command(TARGET test_glm_compute POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_compute>"
    )
endif()

add_test(NAME test_glm_compute COMMAND test_glm_compute)

# ── Тест GLM-5.2 Dense MLP forward (слои 0-2, Phase 4.5 hash-фикстура) ───────
# Прогоняет конкретный C++/CUDA execution loop RunGlmDenseMLPStack (SwiGLU
# w2(silu(w1)x*w3)) через explicit async stream по синтетической hash-фикстуре
# hidden=6144 и сверяет поэлементно против независимого CPU FP64-референса
# (RED/GREEN), плюс проверяет целостность указателей/стридов слоёв 0-2.
add_executable(test_glm_dense_compute test_glm_dense_compute.cpp)
target_link_libraries(test_glm_dense_compute PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_dense_compute POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_dense_compute>"
    )
    add_custom_command(TARGET test_glm_dense_compute POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_dense_compute>"
    )
endif()

add_test(NAME test_glm_dense_compute COMMAND test_glm_dense_compute)

# ── Тест GLM-5.2 MoE Layer-3 router (top_k=8 warp selection) ───────────────
add_executable(test_glm_moe_router test_glm_moe_router.cpp)
target_link_libraries(test_glm_moe_router PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_moe_router POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_router>"
    )
    add_custom_command(TARGET test_glm_moe_router POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_router>"
    )
endif()

add_test(NAME test_glm_moe_router COMMAND test_glm_moe_router)

# ── Тест GLM-5.2 эксперт-стека (NVFP4 dequant + SwiGLU, слои 3-77) ─────────
add_executable(test_glm_experts test_glm_experts.cpp)
target_link_libraries(test_glm_experts PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_experts POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_experts>"
    )
    add_custom_command(TARGET test_glm_experts POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_experts>"
    )
endif()

add_test(NAME test_glm_experts COMMAND test_glm_experts)

# ── TDD Cycle 9: GLM-5.2 MoE router + NVFP4 expert stacks (layers 3-77) ───────
# Прогоняет RunGlmMoEStack (слои 3-77) через одиночный contiguous MoE-арену
# (256 виртуальных срезов активаций экспертов) по синтетической hash-фикстуре
# E2M1/E4M3/E8M0 и сверяет поэлементно против независимого CPU FP64-референса
# (RED/GREEN) + проверяет целостность указателей/границ аорены.
add_executable(test_glm_moe_compute test_glm_moe_compute.cpp)
target_link_libraries(test_glm_moe_compute PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_moe_compute POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_compute>"
    )
    add_custom_command(TARGET test_glm_moe_compute POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_moe_compute>"
    )
endif()

add_test(NAME test_glm_moe_compute COMMAND test_glm_moe_compute)

# ── Тест GLM-5.2 IndexShare attention (mod-4 refresh, слои 3-77) ────────────
add_executable(test_glm_attention test_glm_attention.cpp)
target_link_libraries(test_glm_attention PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_attention POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_attention>"
    )
    add_custom_command(TARGET test_glm_attention POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_attention>"
    )
endif()

add_test(NAME test_glm_attention COMMAND test_glm_attention)

# ── Тест GLM-5.2 полного 79-слойного цикла + MTP epilogue (генерация) ───────
add_executable(test_glm_inference test_glm_inference.cpp)
target_link_libraries(test_glm_inference PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_inference>"
    )
    add_custom_command(TARGET test_glm_inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_inference>"
    )
endif()

add_test(NAME test_glm_inference COMMAND test_glm_inference)

# GLM real-weight tests skip (exit 4 = Catch2 "all test cases skipped") when no
# GLM-5.2 checkpoint is present; that is not a failure per the project SKIP
# invariant, so tell CTest to report it as SKIPPED instead.
set_tests_properties(test_glm_inference PROPERTIES SKIP_RETURN_CODE 4)

# ── TDD Cycle 10: GLM-5.2 full 79-layer SKWOS integration regression (unified
#    arena: IndexShare K-V cache + Layer-78 MTP spaces in one contiguous block)
#    Driving the complete execution loop on real weights with multi-turn
#    prompts; asserts no memory leak / no index-boundary truncation, and that
#    the final speculative Layer-78 MTP logits row converges element-for-element
#    against the independent CPU FP64 reference (RED/GREEN).
add_executable(test_glm_full_compute test_glm_full_compute.cpp)
target_link_libraries(test_glm_full_compute PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_full_compute POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_full_compute>"
    )
    add_custom_command(TARGET test_glm_full_compute POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_full_compute>"
    )
endif()

add_test(NAME test_glm_full_compute COMMAND test_glm_full_compute)
set_tests_properties(test_glm_full_compute PROPERTIES SKIP_RETURN_CODE 4)

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
add_library(test_deepseek_inference_kernels_cu OBJECT test_deepseek_inference_kernels.cu)
set_target_properties(test_deepseek_inference_kernels_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_deepseek_inference_kernels_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    --generate-code=arch=compute_120,code=sm_120;
    --generate-code=arch=compute_121,code=sm_121;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_deepseek_inference_kernels_cu PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/thrust"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/cub"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/libcudacxx/include"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
    "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include"
)
target_compile_definitions(test_deepseek_inference_kernels_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
)
target_link_libraries(test_deepseek_inference_kernels_cu PRIVATE Catch2::Catch2WithMain)

add_executable(test_deepseek_inference_kernels)
set_target_properties(test_deepseek_inference_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_deepseek_inference_kernels PRIVATE
    test_deepseek_inference_kernels_cu lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_deepseek_inference_kernels POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_inference_kernels>"
    )
    add_custom_command(TARGET test_deepseek_inference_kernels POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_inference_kernels>"
    )
endif()

add_test(NAME test_deepseek_inference_kernels COMMAND test_deepseek_inference_kernels)

# test_deepseek_moe — MoE orchestration tests (route_token, shared FFN, routed
# experts, layer mix). Same OBJECT-lib pattern as the kernel test above: the
# .cu must not inherit lsxcommon's C++ compile flags, so it compiles standalone
# and links against lsxcommon only at the executable's link stage.
add_library(test_deepseek_moe_cu OBJECT test_deepseek_moe.cu)
set_target_properties(test_deepseek_moe_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_deepseek_moe_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    --generate-code=arch=compute_120,code=sm_120;
    --generate-code=arch=compute_121,code=sm_121;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_deepseek_moe_cu PRIVATE
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
target_compile_definitions(test_deepseek_moe_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
)
target_link_libraries(test_deepseek_moe_cu PRIVATE Catch2::Catch2WithMain)

add_executable(test_deepseek_moe)
set_target_properties(test_deepseek_moe PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_deepseek_moe PRIVATE
    test_deepseek_moe_cu lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_deepseek_moe POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_moe>"
    )
    add_custom_command(TARGET test_deepseek_moe POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_moe>"
    )
endif()

add_test(NAME test_deepseek_moe COMMAND test_deepseek_moe)

# test_deepseek_arena — single-allocation per-run arena + zero-alloc route_token.
# Same OBJECT-lib pattern as test_deepseek_moe_cu: the .cu must not inherit
# lsxcommon's C++ compile flags, so it compiles standalone and links against
# lsxcommon only at the executable's link stage.
add_library(test_deepseek_arena_cu OBJECT test_deepseek_arena.cu)
set_target_properties(test_deepseek_arena_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_deepseek_arena_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    --generate-code=arch=compute_120,code=sm_120;
    --generate-code=arch=compute_121,code=sm_121;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_deepseek_arena_cu PRIVATE
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
target_compile_definitions(test_deepseek_arena_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
)
target_link_libraries(test_deepseek_arena_cu PRIVATE Catch2::Catch2WithMain)

add_executable(test_deepseek_arena)
set_target_properties(test_deepseek_arena PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_deepseek_arena PRIVATE
    test_deepseek_arena_cu lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_deepseek_arena POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_arena>"
    )
    add_custom_command(TARGET test_deepseek_arena POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_arena>"
    )
endif()

add_test(NAME test_deepseek_arena COMMAND test_deepseek_arena)

# ── Тест isolated sampling seam (lsxcommon::sample, GreedySampler) ──────────
# Same OBJECT-lib pattern as test_deepseek_moe_cu: the .cu must not inherit
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
