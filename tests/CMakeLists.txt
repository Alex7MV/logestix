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

# ── Тест CUDA inference kernels ─────────────────────────────────────────
add_executable(test_inference_kernels test_inference_kernels.cu)
set_target_properties(test_inference_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "120-real;121-real"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
# CUDA 13.3 + MSVC: TypeInfo generates separate .ptx causing nvcc -o conflict
set_property(TARGET test_inference_kernels PROPERTY CudaCompile_TypeInfo false)
target_compile_options(test_inference_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
target_include_directories(test_inference_kernels PRIVATE
    "${CMAKE_SOURCE_DIR}/third_party/cccl/thrust"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/cub"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/libcudacxx/include"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
    "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include"
)
target_link_libraries(test_inference_kernels PRIVATE lsxcommon Catch2::Catch2WithMain)

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
