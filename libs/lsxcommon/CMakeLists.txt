find_path(CUDA_INCLUDE_DIR cuda_runtime.h
    PATHS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/include"
    REQUIRED
)
find_library(CUDA_CUDART_LIBRARY cudart
    PATHS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/lib/x64"
    REQUIRED
)
find_library(CUDA_CUBLAS_LIBRARY cublas
    PATHS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/lib/x64"
    REQUIRED
)

add_library(lsxcommon STATIC
    src/converter.cpp
    src/core.cpp
    src/common.cpp
    src/loader.cpp
    src/pretokenizer.cpp
    src/schema.cpp
    src/tokenizer.cpp
    src/tokenizer_convert.cpp
    src/unicode.cpp
)
set_target_properties(lsxcommon PROPERTIES
    CUDA_ARCHITECTURES "120-real;121-real"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)

enable_language(CUDA)
add_library(lsxcommon_cu OBJECT
    src/safetensors_gpu.cu
    src/testmodel_gpu.cu
    src/inference_load.cu
    src/inference_norm.cu
    src/inference_moe.cu
    src/inference_head.cu
    src/inference_pipeline.cu
    src/norm_kernels.cu
    src/shared_experts.cu
)
set_target_properties(lsxcommon_cu PROPERTIES
    CUDA_ARCHITECTURES "120-real;121-real"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(lsxcommon_cu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-Xcompiler=/utf-8>)
target_include_directories(lsxcommon_cu PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_BINARY_DIR}/include"
    "${CUDA_INCLUDE_DIR}"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/thrust"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/cub"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/libcudacxx/include"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
    "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include"
    "${CMAKE_SOURCE_DIR}/third_party/simdjson/singleheader"
)
target_compile_definitions(lsxcommon_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
    FMT_UNICODE=0
)
target_link_libraries(lsxcommon_cu
    "${CUDA_CUDART_LIBRARY}"
    "${CUDA_CUBLAS_LIBRARY}"
    arrow_shared
    arrow_cuda_shared
)
target_link_libraries(lsxcommon PUBLIC lsxcommon_cu)

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
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/cccl/thrust>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/cccl/cub>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/cccl/libcudacxx/include>"
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
)
target_link_libraries(lsxcommon PRIVATE
    arrow_shared
    arrow_cuda_shared
    simdjson::simdjson
)
