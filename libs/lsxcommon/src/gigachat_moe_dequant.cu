// GigaChat3.5 FP8 -> BF16 dequant (offline converter path).
//
// The source checkpoint stores every linear as F8_E4M3 with a paired F32
// weight_scale_inv of [ceil(R/128), ceil(C/128)] tiles. Unlike the
// GLM-5.3-Flash helper this one clamps the partial last tile (e.g.
// kv_a_proj_with_mqa [576,7168] with scale [5,56]).
#include "lsxcommon/gigachat_moe_convert.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace lsxcommon::gigachat {
namespace {

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

    const __half h =
        __half(__nv_cvt_fp8_to_halfraw(fp8[i], __NV_E4M3));
    out[i] = __float2bfloat16(__half2float(h) * scale[tile]);
}

struct DevGuard {
    uint8_t *d_fp8 = nullptr;
    float *d_scale = nullptr;
    uint8_t *d_out = nullptr;
    ~DevGuard()
    {
        if (d_fp8) cudaFree(d_fp8);
        if (d_scale) cudaFree(d_scale);
        if (d_out) cudaFree(d_out);
    }
};

} // namespace

arrow::Status DequantFp8Block128ToBf16Gpu(const std::vector<uint8_t> &fp8,
                                          const std::vector<int64_t> &shape,
                                          const std::vector<uint8_t> &scale_f32,
                                          const std::vector<int64_t> &scale_shape,
                                          std::vector<uint8_t> &bf16)
{
    if (shape.size() != 2)
        return arrow::Status::Invalid("FP8 tensor must be 2-D");
    if (scale_shape.size() != 2)
        return arrow::Status::Invalid("scale_inv must be 2-D");
    const int64_t rows = shape[0], cols = shape[1];
    const int64_t sr = scale_shape[0], sc = scale_shape[1];
    if (sr * 128 < rows || (sr - 1) * 128 >= rows || sr < 1)
        return arrow::Status::Invalid("scale row grid mismatch");
    if (sc * 128 < cols || (sc - 1) * 128 >= cols || sc < 1)
        return arrow::Status::Invalid("scale col grid mismatch");
    const int64_t numel = rows * cols;
    if (static_cast<int64_t>(fp8.size()) != numel)
        return arrow::Status::Invalid("fp8 byte size != rows*cols");
    if (static_cast<int64_t>(scale_f32.size()) != sr * sc * 4)
        return arrow::Status::Invalid("scale byte size != SR*SC*4");

    DevGuard g;
    if (cudaMalloc(&g.d_fp8, fp8.size()) != cudaSuccess ||
        cudaMalloc(&g.d_scale, scale_f32.size()) != cudaSuccess ||
        cudaMalloc(&g.d_out, static_cast<size_t>(numel) * 2) != cudaSuccess)
        return arrow::Status::IOError("cudaMalloc failed: ",
                                      cudaGetErrorString(cudaGetLastError()));
    cudaStream_t stream = nullptr;  // offline path: legacy default stream
    if (cudaMemcpyAsync(g.d_fp8, fp8.data(), fp8.size(),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(g.d_scale, scale_f32.data(), scale_f32.size(),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess)
        return arrow::Status::IOError("H2D copy failed: ",
                                      cudaGetErrorString(cudaGetLastError()));

    const int threads = 256;
    const long long grid64 = (numel + threads - 1) / threads;
    const int blocks =
        static_cast<int>(std::min<long long>(grid64, 65535LL * 16));
    fp8_tile128_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
        g.d_fp8, g.d_scale, reinterpret_cast<__nv_bfloat16 *>(g.d_out), numel,
        static_cast<int>(cols), static_cast<int>(sc));
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

} // namespace lsxcommon::gigachat
