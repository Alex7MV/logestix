#include "lsxcommon/benchmark_gpu.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/converter.h"
#include "deepseek_inference_load.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <arrow/api.h>
#include <arrow/ipc/reader.h>
#include <arrow/io/file.h>
#include "lsxcommon/log.h"

#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <simdjson.h>

namespace lsxcommon {
namespace gpu {

// The legacy ping-pong expert benchmark was superseded by IModel::Benchmark
// (see benchmark_gpu.cpp shim). What survives here is the NVFP4 dequant kernel,
// which deepseek_inference_pipeline.cu / deepseek_inference_moe.cu reference
// via extern __global__ declarations and therefore must remain linkable.

__device__ float nvfp4_to_float(uint8_t nibble)
{
    int sign = (nibble >> 3) & 1;
    int exp  = (nibble >> 1) & 3;
    int mant = nibble & 1;
    float v = (exp == 0)
        ? static_cast<float>(mant) * 0.5f
        : (1.0f + static_cast<float>(mant) * 0.5f) * static_cast<float>(1 << (exp - 1));
    return sign ? -v : v;
}

__device__ float e4m3_scale_to_float(uint8_t e4m3)
{
    int e4 = (e4m3 >> 3) & 0x0F;
    if (e4 == 0) return 0.0f;
    return __uint_as_float(static_cast<uint32_t>(e4 - 7 + 127) << 23);
}

__global__ void dequant_nvfp4_to_half_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales,
    float global_scale,
    __half* __restrict__ output,
    int num_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    int byte_idx = i >> 1;
    uint8_t nibble = (i & 1)
        ? (packed[byte_idx] >> 4)
        : (packed[byte_idx] & 0x0F);

    float bs = e4m3_scale_to_float(scales[i >> 4]);
    float f32 = nvfp4_to_float(nibble) * bs * global_scale;
    output[i] = __float2half(f32);
}

} // namespace gpu
} // namespace lsxcommon