#include "lsxcommon/safetensors_gpu.h"

#include <arrow/gpu/cuda_api.h>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include <cstring>
#include <cstdint>
#include <mutex>
#include <algorithm>
#include <vector>
#include <cmath>

namespace lsxcommon {
namespace gpu {

namespace {

constexpr int64_t kBlockSize = 16;

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

// ── NVFP4 quantizer (already-scaled value) ────────────────────────────────────

__device__ uint8_t dev_quantize_nvfp4(float val)
{
    if (val == 0.0f) return 0;
    int sign = val < 0 ? 1 : 0;
    float abs_val = fabsf(val);
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

__global__ void block_maxabs_kernel(const float *input, float *block_max,
                                    int64_t nblocks)
{
    __shared__ float sdata[32];

    int64_t blk = blockIdx.x; // NVFP4 block index
    int64_t base = blk * kBlockSize;

    float val = 0.0f;
    if (threadIdx.x < kBlockSize && base + threadIdx.x < nblocks * kBlockSize)
        val = fabsf(input[base + threadIdx.x]);
    sdata[threadIdx.x] = val;
    __syncthreads();

    for (int s = kBlockSize / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x],
                                        sdata[threadIdx.x + s]);
        __syncthreads();
    }

    if (threadIdx.x == 0)
        block_max[blk] = sdata[0];
}

__global__ void quantize_pack_perblock_kernel(const float *input,
                                               uint8_t *output,
                                               const float *block_scales,
                                               int64_t n)
{
    int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
    if (i >= n) return;

    int64_t blk0 = i / kBlockSize;
    int64_t blk1 = (i + 1) / kBlockSize;

    float s0 = block_scales[blk0];
    float s1 = block_scales[blk1];

    uint8_t lo = dev_quantize_nvfp4(input[i] / s0);
    uint8_t hi = (i + 1 < n) ? dev_quantize_nvfp4(input[i + 1] / s1) : 0;
    output[i / 2] = (hi << 4) | lo;
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

float e4m3_p2_to_float(uint8_t b)
{
    if (b == 0) return 0.0f;
    int e_stored = (b >> 3) & 0x0F;
    int exp_unb = e_stored - 7;
    return std::ldexp(1.0f, exp_unb);
}

uint8_t float_to_e4m3_p2(float v)
{
    if (v == 0.0f) return 0;
    int e;
    float m = std::frexp(v, &e);
    int e4 = e - 1 + 7;
    if (e4 < 1) e4 = 1;
    if (e4 > 14) e4 = 14;
    return static_cast<uint8_t>(e4 << 3);
}

} // anonymous namespace

bool quantize_tensor_gpu(const uint8_t *raw_data, int64_t numel,
                         const std::string &dtype,
                         std::vector<uint8_t> &packed,
                         std::vector<uint8_t> &scales_e4m3,
                         float &global_scale)
{
    int dc = dtype_code(dtype);
    if (dc < 0) return false;

    auto maybe_ctx = get_cuda_ctx();
    if (!maybe_ctx.ok()) return false;
    auto ctx = *maybe_ctx;

    int64_t raw_size = numel * dtype_byte_size(dc);
    int64_t npacked = (numel + 1) / 2;
    int64_t nblocks = (numel + kBlockSize - 1) / kBlockSize;

    // Allocate via Arrow CUDA context
    auto maybe_raw = ctx->Allocate(raw_size);
    auto maybe_float = ctx->Allocate(numel * sizeof(float));
    auto maybe_packed = ctx->Allocate(npacked);
    auto maybe_block_max = ctx->Allocate(nblocks * sizeof(float));
    auto maybe_block_scales = ctx->Allocate(nblocks * sizeof(float));

    if (!maybe_raw.ok() || !maybe_float.ok() || !maybe_packed.ok()
        || !maybe_block_max.ok() || !maybe_block_scales.ok())
        return false;

    auto raw_buf = std::move(*maybe_raw);
    auto float_buf = std::move(*maybe_float);
    auto packed_buf = std::move(*maybe_packed);
    auto block_max_buf = std::move(*maybe_block_max);
    auto block_scales_buf = std::move(*maybe_block_scales);

    float *d_float = reinterpret_cast<float *>(float_buf->mutable_data());
    uint8_t *d_packed = packed_buf->mutable_data();
    float *d_block_max = reinterpret_cast<float *>(block_max_buf->mutable_data());
    float *d_block_scales = reinterpret_cast<float *>(block_scales_buf->mutable_data());

    // 1. Copy raw input host → device
    auto st = raw_buf->CopyFromHost(0, raw_data, raw_size);
    if (!st.ok()) return false;
    uint8_t *d_raw = raw_buf->mutable_data();

    // 2. Convert raw → float32
    int block_size = 256;
    unsigned int grid_size = 512;

    convert_kernel<<<grid_size, static_cast<unsigned int>(block_size)>>>(
        d_raw, d_float, numel, dc);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return false;

    // 3. Per-block max_abs
    int block_max_grid = static_cast<int>(nblocks);
    int block_max_block = 32;

    block_maxabs_kernel<<<block_max_grid, block_max_block>>>(
        d_float, d_block_max, nblocks);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return false;

    // 4. Global max_abs via Thrust
    float global_max_abs = thrust::transform_reduce(
        thrust::device,
        thrust::device_pointer_cast(d_float),
        thrust::device_pointer_cast(d_float) + numel,
        [] __host__ __device__ (float x) { return fabsf(x); },
        0.0f,
        thrust::maximum<float>());

    // raw_buf goes out of scope → GPU memory freed

    if (!isfinite(global_max_abs) || global_max_abs == 0.0f)
        global_max_abs = 1.0f / 128.0f;
    global_scale = global_max_abs / 8.0f;

    // 5. Copy block max_abs → host, compute E4M3 scales + float scales
    std::vector<float> host_block_max(nblocks);
    st = block_max_buf->CopyToHost(0, nblocks * sizeof(float),
                                    reinterpret_cast<uint8_t*>(host_block_max.data()));
    if (!st.ok()) return false;

    scales_e4m3.assign(nblocks, 0);
    std::vector<float> host_block_scales(nblocks);
    for (int64_t i = 0; i < nblocks; i++) {
        float max_val = host_block_max[i];
        float s = (max_val == 0.0f) ? (1.0f / 128.0f) : (max_val / 8.0f);
        uint8_t e4 = float_to_e4m3_p2(s);
        scales_e4m3[i] = e4;
        host_block_scales[i] = e4m3_p2_to_float(e4);
    }

    // 6. Upload float scales → device
    st = block_scales_buf->CopyFromHost(0,
         reinterpret_cast<const uint8_t*>(host_block_scales.data()),
         nblocks * sizeof(float));
    if (!st.ok()) return false;

    // 7. Per-block quantize + pack
    quantize_pack_perblock_kernel<<<grid_size,
                                     static_cast<unsigned int>(block_size)>>>(
        d_float, d_packed, d_block_scales, numel);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return false;

    // 8. Copy packed data device → host
    packed.resize(npacked);
    st = packed_buf->CopyToHost(0, npacked, packed.data());
    if (!st.ok()) return false;

    return true;
}

} // namespace gpu
} // namespace lsxcommon
