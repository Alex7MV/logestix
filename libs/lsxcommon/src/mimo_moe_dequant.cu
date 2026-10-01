// MiMo-V2.6-Flash-RL offline dequant (converter path).
//   * FP8 E4M3 + F32 weight_scale_inv block-128 -> BF16 (tolerating a scale
//     grid with spare row blocks, the GA qkv_proj quirk).
//   * MXFP4 packed E2M1 nibbles + F8_E8M0 per-32 -> BF16 (repack fallback).
#include "lsxcommon/mimo_moe_convert.h"

#include "nvfp4_codec.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace lsxcommon {
namespace mimo {
namespace {

__device__ __forceinline__ float MimoE8m0ToFloat(uint8_t b)
{
    if (!b) return 0.0f;
    const uint32_t bits = static_cast<uint32_t>(b) << 23;
    float v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

__global__ void fp8_tile128_to_bf16_kernel(const uint8_t *__restrict__ fp8,
                                           const float *__restrict__ scale,
                                           __nv_bfloat16 *__restrict__ out,
                                           long long numel, int cols,
                                           int scale_cols)
{
    const long long i = blockIdx.x * static_cast<long long>(blockDim.x) +
                        threadIdx.x;
    if (i >= numel) return;
    const int r = static_cast<int>(i / cols);
    const int c = static_cast<int>(i % cols);
    const int tile = (r >> 7) * scale_cols + (c >> 7);
    const __half h = __half(__nv_cvt_fp8_to_halfraw(fp8[i], __NV_E4M3));
    out[i] = __float2bfloat16(__half2float(h) * scale[tile]);
}

__global__ void mxfp4_to_bf16_kernel(const uint8_t *__restrict__ packed,
                                     const uint8_t *__restrict__ e8m0,
                                     __nv_bfloat16 *__restrict__ out,
                                     long long numel, int logical_cols,
                                     int blocks_per_row)
{
    const long long i = blockIdx.x * static_cast<long long>(blockDim.x) +
                        threadIdx.x;
    if (i >= numel) return;
    const int r = static_cast<int>(i / logical_cols);
    const int c = static_cast<int>(i % logical_cols);
    const uint8_t byte = packed[i >> 1];
    const int nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
    const float s = MimoE8m0ToFloat(e8m0[r * blocks_per_row + (c >> 5)]);
    out[i] = __float2bfloat16(gpu::nvfp4::E2M1Value(nib) * s);
}

struct DevGuard {
    uint8_t *d_a = nullptr;
    uint8_t *d_b = nullptr;
    uint8_t *d_out = nullptr;
    ~DevGuard()
    {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_out) cudaFree(d_out);
    }
};

int LaunchBlocks(long long numel, int threads)
{
    const long long grid64 = (numel + threads - 1) / threads;
    return static_cast<int>(std::min<long long>(grid64, 65535LL * 16));
}

} // namespace

arrow::Status DequantFp8Block128ToBf16Gpu(const std::vector<uint8_t> &fp8,
                                          const std::vector<int64_t> &shape,
                                          const std::vector<uint8_t> &scale_f32,
                                          const std::vector<int64_t> &scale_shape,
                                          std::vector<uint8_t> &bf16)
{
    if (shape.size() != 2 || scale_shape.size() != 2)
        return arrow::Status::Invalid("FP8 tensor and scale must be 2-D");
    const int64_t rows = shape[0], cols = shape[1];
    const int64_t sr = scale_shape[0], sc = scale_shape[1];
    if (sr < 1 || sr * 128 < rows || sc < 1 || sc * 128 < cols)
        return arrow::Status::Invalid("scale grid does not cover the tensor");
    const int64_t numel = rows * cols;
    if (static_cast<int64_t>(fp8.size()) != numel)
        return arrow::Status::Invalid("fp8 byte size != rows*cols");
    if (static_cast<int64_t>(scale_f32.size()) != sr * sc * 4)
        return arrow::Status::Invalid("scale byte size != SR*SC*4");

    DevGuard g;
    if (cudaMalloc(&g.d_a, fp8.size()) != cudaSuccess ||
        cudaMalloc(&g.d_b, scale_f32.size()) != cudaSuccess ||
        cudaMalloc(&g.d_out, static_cast<size_t>(numel) * 2) != cudaSuccess)
        return arrow::Status::IOError("cudaMalloc failed: ",
                                      cudaGetErrorString(cudaGetLastError()));
    cudaStream_t stream = nullptr;
    if (cudaMemcpyAsync(g.d_a, fp8.data(), fp8.size(), cudaMemcpyHostToDevice,
                        stream) != cudaSuccess ||
        cudaMemcpyAsync(g.d_b, scale_f32.data(), scale_f32.size(),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess)
        return arrow::Status::IOError("H2D copy failed: ",
                                      cudaGetErrorString(cudaGetLastError()));

    const int threads = 256;
    fp8_tile128_to_bf16_kernel<<<LaunchBlocks(numel, threads), threads, 0, stream>>>(
        g.d_a, reinterpret_cast<const float *>(g.d_b),
        reinterpret_cast<__nv_bfloat16 *>(g.d_out), numel, static_cast<int>(cols),
        static_cast<int>(sc));
    if (cudaGetLastError() != cudaSuccess)
        return arrow::Status::IOError("kernel launch failed: ",
                                      cudaGetErrorString(cudaGetLastError()));

    bf16.resize(static_cast<size_t>(numel) * 2);
    if (cudaMemcpyAsync(bf16.data(), g.d_out, bf16.size(),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        return arrow::Status::IOError("D2H copy failed: ",
                                      cudaGetErrorString(cudaGetLastError()));
    if (cudaStreamSynchronize(stream) != cudaSuccess)
        return arrow::Status::IOError("stream sync failed: ",
                                      cudaGetErrorString(cudaGetLastError()));
    return arrow::Status::OK();
}

arrow::Status DequantMxfp4ToBf16Gpu(const std::vector<uint8_t> &packed,
                                    const std::vector<int64_t> &shape,
                                    const std::vector<uint8_t> &e8m0,
                                    const std::vector<int64_t> &scale_shape,
                                    std::vector<uint8_t> &bf16)
{
    if (shape.size() != 2 || scale_shape.size() != 2)
        return arrow::Status::Invalid("MXFP4 tensor and scale must be 2-D");
    const int64_t rows = shape[0];
    const int64_t packed_cols = shape[1];
    const int64_t logical_cols = packed_cols * 2;
    const int64_t blocks = scale_shape[1];
    if (blocks * 32 != logical_cols)
        return arrow::Status::Invalid("MXFP4 scale block count mismatch");
    if (scale_shape[0] != rows)
        return arrow::Status::Invalid("MXFP4 scale row count mismatch");
    const int64_t numel = rows * logical_cols;
    if (static_cast<int64_t>(packed.size()) != rows * packed_cols)
        return arrow::Status::Invalid("MXFP4 packed byte size mismatch");
    if (static_cast<int64_t>(e8m0.size()) != rows * blocks)
        return arrow::Status::Invalid("MXFP4 E8M0 byte size mismatch");

    DevGuard g;
    if (cudaMalloc(&g.d_a, packed.size()) != cudaSuccess ||
        cudaMalloc(&g.d_b, e8m0.size()) != cudaSuccess ||
        cudaMalloc(&g.d_out, static_cast<size_t>(numel) * 2) != cudaSuccess)
        return arrow::Status::IOError("cudaMalloc failed: ",
                                      cudaGetErrorString(cudaGetLastError()));
    cudaStream_t stream = nullptr;
    if (cudaMemcpyAsync(g.d_a, packed.data(), packed.size(),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(g.d_b, e8m0.data(), e8m0.size(), cudaMemcpyHostToDevice,
                        stream) != cudaSuccess)
        return arrow::Status::IOError("H2D copy failed: ",
                                      cudaGetErrorString(cudaGetLastError()));

    const int threads = 256;
    mxfp4_to_bf16_kernel<<<LaunchBlocks(numel, threads), threads, 0, stream>>>(
        g.d_a, g.d_b, reinterpret_cast<__nv_bfloat16 *>(g.d_out), numel,
        static_cast<int>(logical_cols), static_cast<int>(blocks));
    if (cudaGetLastError() != cudaSuccess)
        return arrow::Status::IOError("kernel launch failed: ",
                                      cudaGetErrorString(cudaGetLastError()));

    bf16.resize(static_cast<size_t>(numel) * 2);
    if (cudaMemcpyAsync(bf16.data(), g.d_out, bf16.size(),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        return arrow::Status::IOError("D2H copy failed: ",
                                      cudaGetErrorString(cudaGetLastError()));
    if (cudaStreamSynchronize(stream) != cudaSuccess)
        return arrow::Status::IOError("stream sync failed: ",
                                      cudaGetErrorString(cudaGetLastError()));
    return arrow::Status::OK();
}

} // namespace mimo
} // namespace lsxcommon
