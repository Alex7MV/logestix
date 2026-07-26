#include "lsxcommon/safetensors_gpu.h"

#include <arrow/gpu/cuda_api.h>

#include <cuda_runtime.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include <cstring>
#include <cstdint>
#include <mutex>

namespace lsxcommon {
namespace gpu {

namespace {

// ── Singleton Arrow CUDA context ─────────────────────────────────────────────

arrow::Result<std::shared_ptr<arrow::cuda::CudaContext>> get_cuda_ctx()
{
    static std::shared_ptr<arrow::cuda::CudaContext> ctx;
    static std::once_flag init;
    static arrow::Status status;

    std::call_once(init, [] {
        auto mgr = arrow::cuda::CudaDeviceManager::Instance();
        if (!mgr.ok()) { status = mgr.status(); return; }
        auto result = (*mgr)->GetContext(0);
        if (!result.ok()) { status = result.status(); return; }
        ctx = *result;
    });

    if (!ctx) return status;
    return ctx;
}

// ── Device helpers for dtype conversion ──────────────────────────────────────

__device__ float dev_bf16_to_float(const uint8_t *p)
{
    uint16_t u;
    memcpy(&u, p, sizeof(u));
    uint32_t bits = static_cast<uint32_t>(u) << 16;
    float v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

__device__ float dev_f8e4m3_to_float(uint8_t b)
{
    if (!b || b == 0x80) return 0.0f;
    int s = (b >> 7) & 1;
    int e = (b >> 3) & 0x0F;
    int m = b & 0x07;
    uint32_t f32 = (static_cast<uint32_t>(s) << 31)
                 | (static_cast<uint32_t>(e + 127 - 7) << 23)
                 | (static_cast<uint32_t>(m) << 20);
    float v;
    memcpy(&v, &f32, sizeof(v));
    return v;
}

__device__ float dev_f8e8m0_to_float(uint8_t b)
{
    if (!b) return 0.0f;
    uint32_t bits = static_cast<uint32_t>(b) << 23;
    float v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

__device__ float dev_i8_to_float(uint8_t b)
{
    return static_cast<float>(static_cast<int8_t>(b));
}

// ── NVFP4 quantizer ──────────────────────────────────────────────────────────

__device__ uint8_t dev_quantize_nvfp4(float val, float scale)
{
    float scaled = val / scale;
    if (scaled == 0.0f) return 0;
    int sign = scaled < 0 ? 1 : 0;
    float abs_val = fabsf(scaled);
    if (abs_val >= 8.0f) abs_val = 7.999f;

    static const float levels[8] = {
        0.0625f, 0.125f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f
    };
    int best = 0;
    float best_diff = fabsf(abs_val - levels[0]);
    for (int i = 1; i < 8; i++) {
        float d = fabsf(abs_val - levels[i]);
        if (d < best_diff) { best_diff = d; best = i; }
    }
    return static_cast<uint8_t>((sign << 3) | best);
}

// ── Kernels ──────────────────────────────────────────────────────────────────

__global__ void convert_kernel(const uint8_t *input, float *output,
                               int64_t n, int dtype_code)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (; i < n; i += stride) {
        float v;
        switch (dtype_code) {
        case 0: v = dev_bf16_to_float(input + i * 2); break;
        case 1: v = dev_f8e4m3_to_float(input[i]);    break;
        case 2: v = dev_f8e8m0_to_float(input[i]);    break;
        case 3: memcpy(&v, input + i * 4, sizeof(v)); break;
        case 4: v = dev_i8_to_float(input[i]);         break;
        default: v = 0.0f;
        }
        output[i] = v;
    }
}

__global__ void quantize_pack_kernel(const float *input, uint8_t *output,
                                     int64_t n, float scale)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t pair = i; pair < n / 2; pair += stride) {
        uint8_t lo = dev_quantize_nvfp4(input[pair * 2], scale);
        uint8_t hi = dev_quantize_nvfp4(input[pair * 2 + 1], scale);
        output[pair] = (hi << 4) | lo;
    }
    if (i == 0 && (n & 1)) {
        output[n / 2] = dev_quantize_nvfp4(input[n - 1], scale);
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

int dtype_code(const std::string &dtype)
{
    if (dtype == "BF16") return 0;
    if (dtype == "F8_E4M3") return 1;
    if (dtype == "F8_E8M0") return 2;
    if (dtype == "F32") return 3;
    if (dtype == "I8" || dtype == "U8") return 4;
    return -1;
}

int64_t dtype_byte_size(int dc)
{
    switch (dc) {
    case 0: return 2;
    case 1: return 1;
    case 2: return 1;
    case 3: return 4;
    case 4: return 1;
    default: return 0;
    }
}

} // anonymous namespace

bool quantize_tensor_gpu(const uint8_t *raw_data, int64_t numel,
                         const std::string &dtype,
                         std::vector<uint8_t> &packed, float &scale)
{
    int dc = dtype_code(dtype);
    if (dc < 0) return false;

    auto maybe_ctx = get_cuda_ctx();
    if (!maybe_ctx.ok()) return false;
    auto ctx = *maybe_ctx;

    int64_t raw_size = numel * dtype_byte_size(dc);
    int64_t npacked = (numel + 1) / 2;

    // Allocate via Arrow CUDA context
    auto maybe_raw = ctx->Allocate(raw_size);
    auto maybe_float = ctx->Allocate(numel * sizeof(float));
    auto maybe_packed = ctx->Allocate(npacked);

    if (!maybe_raw.ok() || !maybe_float.ok() || !maybe_packed.ok())
        return false;

    auto raw_buf = std::move(*maybe_raw);
    auto float_buf = std::move(*maybe_float);
    auto packed_buf = std::move(*maybe_packed);

    float *d_float = reinterpret_cast<float *>(float_buf->mutable_data());
    uint8_t *d_packed = packed_buf->mutable_data();

    // Copy raw input host→device via Arrow Buffers
    auto copy_status = raw_buf->CopyFromHost(0, raw_data, raw_size);
    if (!copy_status.ok()) return false;

    uint8_t *d_raw = raw_buf->mutable_data();

    // Launch convert kernel
    int block_size = 256;
    unsigned int grid_size = 512;

    convert_kernel<<<grid_size, (unsigned int)block_size>>>(
        d_raw, d_float, numel, dc);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return false;

    // Find max_abs via Thrust transform_reduce
    float max_abs = thrust::transform_reduce(
        thrust::device,
        d_float, d_float + numel,
        [] __host__ __device__ (float x) { return fabsf(x); },
        0.0f,
        thrust::maximum<float>());

    // raw_buf goes out of scope here → GPU memory freed

    if (!isfinite(max_abs)) max_abs = 1.0f;
    if (max_abs == 0.0f) max_abs = 1.0f;
    scale = max_abs / 8.0f;

    // Launch quantize+pack kernel
    quantize_pack_kernel<<<grid_size, (unsigned int)block_size>>>(
        d_float, d_packed, numel, scale);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return false;

    // Copy packed data device→host via Arrow buffer
    packed.resize(npacked);
    copy_status = packed_buf->CopyToHost(0, npacked, packed.data());
    if (!copy_status.ok()) return false;

    return true;
}

} // namespace gpu
} // namespace lsxcommon
