#include "step_moe_dequant.h"

#include <cuda_fp8.h>

namespace lsxcommon::step {
namespace {

constexpr int kBlock = 128;

__global__ void step_fp8_block_dequant_kernel(const uint8_t *__restrict__ w,
                                              const float *__restrict__ s,
                                              uint16_t *__restrict__ out,
                                              int experts, int rows, int cols)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int exp = blockIdx.z;
    if (col >= cols || row >= rows || exp >= experts) return;
    const int scols = (cols + kBlock - 1) / kBlock;
    const int srow = (row + kBlock - 1) / kBlock - 1;
    const float scale =
        s[(static_cast<size_t>(exp) * srow + (row / kBlock)) * scols +
          (col / kBlock)];
    const size_t idx = (static_cast<size_t>(exp) * rows + row) * cols + col;
    const __half h = __nv_cvt_fp8_to_halfraw(w[idx], __NV_E4M3);
    const float v = __half2float(h) * scale;
    __nv_bfloat16 b = __float2bfloat16_rn(v);
    uint16_t bits;
    memcpy(&bits, &b, sizeof(bits));
    out[idx] = bits;
}

} // namespace

cudaError_t LaunchStepFp8StackedDequantToBf16(const uint8_t *w,
                                              const float *scales, int experts,
                                              int rows, int cols,
                                              uint16_t *out,
                                              cudaStream_t stream)
{
    if (experts <= 0 || rows <= 0 || cols <= 0)
        return cudaErrorInvalidValue;
    const dim3 block(32, 8, 1);
    const dim3 grid((cols + block.x - 1) / block.x,
                    (rows + block.y - 1) / block.y, experts);
    step_fp8_block_dequant_kernel<<<grid, block, 0, stream>>>(
        w, scales, out, experts, rows, cols);
    return cudaGetLastError();
}

bool StepFp8StackedDequantToBf16Host(const uint8_t *w, const float *scales,
                                     int experts, int rows, int cols,
                                     std::vector<uint8_t> &out_bf16)
{
    const int64_t n = static_cast<int64_t>(experts) * rows * cols;
    const int64_t scale_n = static_cast<int64_t>(experts) *
                            ((rows + kBlock - 1) / kBlock) *
                            ((cols + kBlock - 1) / kBlock);
    uint8_t *d_w = nullptr;
    float *d_s = nullptr;
    uint16_t *d_out = nullptr;
    if (cudaMalloc(&d_w, static_cast<size_t>(n)) != cudaSuccess) return false;
    if (cudaMalloc(&d_s, static_cast<size_t>(scale_n) * sizeof(float)) !=
        cudaSuccess) {
        cudaFree(d_w);
        return false;
    }
    if (cudaMalloc(&d_out, static_cast<size_t>(n) * sizeof(uint16_t)) !=
        cudaSuccess) {
        cudaFree(d_w);
        cudaFree(d_s);
        return false;
    }
    bool ok = cudaMemcpy(d_w, w, static_cast<size_t>(n),
                         cudaMemcpyHostToDevice) == cudaSuccess;
    ok = ok && cudaMemcpy(d_s, scales,
                          static_cast<size_t>(scale_n) * sizeof(float),
                          cudaMemcpyHostToDevice) == cudaSuccess;
    ok = ok && LaunchStepFp8StackedDequantToBf16(d_w, d_s, experts, rows, cols,
                                                 d_out, nullptr) == cudaSuccess;
    if (ok) ok = cudaDeviceSynchronize() == cudaSuccess;
    out_bf16.resize(static_cast<size_t>(n) * 2);
    if (ok)
        ok = cudaMemcpy(out_bf16.data(), d_out,
                        static_cast<size_t>(n) * sizeof(uint16_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(d_w);
    cudaFree(d_s);
    cudaFree(d_out);
    return ok;
}

} // namespace lsxcommon::step
