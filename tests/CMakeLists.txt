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

# ── Тест round-trip загрузчика ──────────────────────────────────────────
add_executable(test_loader test_loader.cpp)
target_link_libraries(test_loader PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_loader>"
    )
    add_custom_command(TARGET test_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_loader>"
    )
endif()

add_test(NAME test_loader COMMAND test_loader)

# ── Тест GLM-5.2 конвертера (парсер имён тензоров) ────────────────────────
add_executable(test_glm_convert test_glm_convert.cpp)
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

# ── Тест TaskMosaic CE сетевого слоя (lsxnetwork) ───────────────────────────
add_executable(test_mosaic_network test_mosaic_network.cpp)
target_link_libraries(test_mosaic_network PRIVATE lsxnetwork Catch2::Catch2WithMain)

add_test(NAME test_mosaic_network COMMAND test_mosaic_network)

# ── Интеграционный тест TaskMosaic: мультимодальный оркестратор (lsx_orchestrator) ──
# Прогоняет одновременный burst OpenAI-запросов на обе ветки ("glm-5.2" и
# "deepseek-v4-flash") через длинный lock-free SPSC конвейер и проверяет
# zero-copy retirement без token-drift. Линкует lsxcommon (оркестратор + IModel)
# и lsxnetwork (FrameQueue).
add_executable(test_integrated_pipeline test_integrated_pipeline.cpp)
target_link_libraries(test_integrated_pipeline PRIVATE lsxcommon lsxnetwork Catch2::Catch2WithMain)

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
add_library(test_inference_kernels_cu OBJECT test_inference_kernels.cu)
set_target_properties(test_inference_kernels_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_inference_kernels_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    --generate-code=arch=compute_120,code=sm_120;
    --generate-code=arch=compute_121,code=sm_121;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_inference_kernels_cu PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/thrust"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/cub"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/libcudacxx/include"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
    "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include"
)
target_compile_definitions(test_inference_kernels_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
)
target_link_libraries(test_inference_kernels_cu PRIVATE Catch2::Catch2WithMain)

add_executable(test_inference_kernels)
set_target_properties(test_inference_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_inference_kernels PRIVATE
    test_inference_kernels_cu lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_inference_kernels POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_inference_kernels>"
    )
    add_custom_command(TARGET test_inference_kernels POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_inference_kernels>"
    )
endif()

add_test(NAME test_inference_kernels COMMAND test_inference_kernels)
