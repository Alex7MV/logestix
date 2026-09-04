// GLM-5.3-Flash converter GPU helpers: F8_E4M3 + F32 tile-scale dequant to
// BF16 (offline conversion path). The generic quantize_tensor_gpu e8m0 hook
// only accepts byte scales; this checkpoint stores F32 weight_scale_inv.
#include "lsxcommon/glm_moe_flash_convert.h"

#include "lsx_kernels_quant.cuh"

#include <cuda_runtime.h>

namespace lsxcommon {
namespace glm_moe_flash {
namespace {

__global__ void fp8_f32tile_to_bf16_kernel(const uint8_t *__restrict__ src,
                                           const float *__restrict__ scale_inv,
                                           uint16_t *__restrict__ dst,
                                           int64_t numel, int64_t cols,
                                           int64_t sc_tiles)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x
                    + threadIdx.x;
    if (i >= numel) return;
    const int64_t r = i / cols;
    const int64_t c = i % cols;
    const float v = kernels::F8E4m3ToFloat(src[i])
                  * scale_inv[(r / 128) * sc_tiles + (c / 128)];
    // BF16 round-to-nearest-even from F32 (drop 16 low mantissa bits).
    uint32_t bits = __float_as_uint(v);
    bits += 0x7FFF + ((bits >> 16) & 1);
    dst[i] = static_cast<uint16_t>(bits >> 16);
}

struct DevGuard {
    uint8_t *d_fp8 = nullptr;
    float *d_scale = nullptr;
    uint16_t *d_out = nullptr;
    ~DevGuard() {
        cudaFree(d_fp8);
        cudaFree(d_scale);
        cudaFree(d_out);
    }
};

} // namespace

arrow::Status DequantFp8TilesToBf16Gpu(const std::vector<uint8_t> &fp8,
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
    if (rows != sr * 128 || cols != sc * 128)
        return arrow::Status::Invalid(
            "scale tile grid mismatch: expect rows/128 x cols/128");
    const int64_t numel = rows * cols;
    if (static_cast<int64_t>(fp8.size()) != numel)
        return arrow::Status::Invalid("fp8 byte size != rows*cols");
    if (static_cast<int64_t>(scale_f32.size()) != sr * sc * 4)
        return arrow::Status::Invalid("scale byte size != SR*SC*4");

    DevGuard g;
    if (cudaMalloc(&g.d_fp8, fp8.size()) != cudaSuccess
        || cudaMalloc(&g.d_scale, scale_f32.size()) != cudaSuccess
        || cudaMalloc(&g.d_out, numel * 2) != cudaSuccess)
        return arrow::Status::IOError("cudaMalloc failed: ",
                                      cudaGetErrorString(cudaGetLastError()));
    cudaStream_t stream = nullptr;  // legacy default stream (offline path)
    if (cudaMemcpyAsync(g.d_fp8, fp8.data(), fp8.size(),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess
        || cudaMemcpyAsync(g.d_scale, scale_f32.data(), scale_f32.size(),
                           cudaMemcpyHostToDevice, stream) != cudaSuccess)
        return arrow::Status::IOError("H2D copy failed: ",
                                      cudaGetErrorString(cudaGetLastError()));

    const int threads = 256;
    const int64_t grid64 = (numel + threads - 1) / threads;
    const int blocks = static_cast<int>(std::min<int64_t>(grid64, 65535LL * 16));
    fp8_f32tile_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
        g.d_fp8, g.d_scale, g.d_out, numel, cols, sc);
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

} // namespace glm_moe_flash
} // namespace lsxcommon
