#include "lsxcommon/glm_inference.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cmath>

namespace lsxcommon {
namespace glm {

namespace {

constexpr float kGlmSwigluLimit = 10.0f;
constexpr int kGlmRmsBlock = 256;

__global__ void GlmRmsNormKernel(const __half *__restrict__ x,
                                 const __half *__restrict__ w,
                                 __half *__restrict__ y,
                                 int dim, float eps)
{
    constexpr int kBlock = 256;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float s_red[kBlock / 32];
    float acc = 0.0f;
    for (int i = tid; i < dim; i += kBlock) {
        float v = __half2float(x[row * dim + i]);
        acc += v * v;
    }
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, off);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) s_red[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float v = (tid < (kBlock >> 5)) ? s_red[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffffu, v, off);
        if (lane == 0) s_red[0] = v;
    }
    __syncthreads();
    const float rinv = rsqrtf(s_red[0] / static_cast<float>(dim) + eps);
    for (int i = tid; i < dim; i += kBlock) {
        float v = __half2float(x[row * dim + i]);
        y[row * dim + i] = __float2half(v * rinv * __half2float(w[i]));
    }
}

__global__ void GlmSwigluClampKernel(__half *__restrict__ gate,
                                     const __half *__restrict__ up,
                                     int n, float limit)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(gate[i]);
    float u = __half2float(up[i]);
    if (g > limit) g = limit;
    if (u > limit) u = limit;
    if (u < -limit) u = -limit;
    float s = g / (1.0f + expf(-g));
    gate[i] = __float2half(s * u);
}

__global__ void GlmAddResidualKernel(__half *__restrict__ dst,
                                     const __half *__restrict__ base,
                                     int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(__half2float(dst[i]) + __half2float(base[i]));
}

cudaError_t GemmF16(cublasHandle_t cublas_h, cudaStream_t stream,
                    const __half *d_w, int m, int k, int batch,
                    const __half *d_x, __half *d_out)
{
    cublasSetStream(cublas_h, stream);
    float alpha = 1.0f;
    float beta  = 0.0f;
    cublasStatus_t st = cublasGemmEx(
        cublas_h, CUBLAS_OP_T, CUBLAS_OP_N, m, batch, k,
        &alpha, d_w, CUDA_R_16F, k,
                d_x, CUDA_R_16F, k,
        &beta, d_out, CUDA_R_16F, m,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    return st == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
}

cudaError_t RunGlmRmsNorm(cudaStream_t stream, const __half *x,
                          const __half *w, __half *y, int rows, int dim,
                          float eps)
{
    GlmRmsNormKernel<<<rows, kGlmRmsBlock, 0, stream>>>(x, w, y, dim, eps);
    return cudaGetLastError();
}

cudaError_t RunDenseMLPLayer(cublasHandle_t cublas_h, cudaStream_t stream,
                             const GlmLayerWeights &lw,
                             int seq, int hidden, int inter,
                             const __half *d_x, __half *d_y,
                             __half *d_normed, __half *d_gate, __half *d_up)
{
    if (!lw.d_input_norm || !lw.d_post_attn_norm ||
        !lw.d_mlp_gate || !lw.d_mlp_up || !lw.d_mlp_down)
        return cudaErrorInvalidValue;

    cudaError_t e = RunGlmRmsNorm(stream, d_x, lw.d_input_norm, d_normed,
                                  seq, hidden, kGlmRmsEps);
    if (e != cudaSuccess) return e;
    e = RunGlmRmsNorm(stream, d_x, lw.d_post_attn_norm, d_normed,
                      seq, hidden, kGlmRmsEps);
    if (e != cudaSuccess) return e;
    e = GemmF16(cublas_h, stream, lw.d_mlp_gate, inter, hidden, seq,
                d_normed, d_gate);
    if (e != cudaSuccess) return e;
    e = GemmF16(cublas_h, stream, lw.d_mlp_up, inter, hidden, seq,
                d_normed, d_up);
    if (e != cudaSuccess) return e;
    const int n_act = seq * inter;
    GlmSwigluClampKernel<<<(n_act + 255) / 256, 256, 0, stream>>>(
        d_gate, d_up, n_act, kGlmSwigluLimit);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    e = GemmF16(cublas_h, stream, lw.d_mlp_down, hidden, inter, seq,
                d_gate, d_y);
    if (e != cudaSuccess) return e;
    const int n_out = seq * hidden;
    GlmAddResidualKernel<<<(n_out + 255) / 256, 256, 0, stream>>>(d_y, d_x, n_out);
    return cudaGetLastError();
}

} // namespace

cudaError_t RunGlmDenseMLPStack(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmPinnedWeights &pw, int seq,
    __half *d_act, __half *d_alt, __half *d_result, __half *d_scratch)
{
    const int hidden = kGlmHidden;
    const int inter  = kGlmMoeIntermediate;
    if (pw.layers.size() < 3 || seq <= 0) return cudaErrorInvalidValue;
    __half *d_normed = d_scratch;
    __half *d_gate   = d_scratch + static_cast<size_t>(seq) * hidden;
    __half *d_up     = d_gate  + static_cast<size_t>(seq) * inter;
    __half *cur = d_act;
    for (int l = 0; l < 3; ++l) {
        __half *nxt = (l == 2) ? d_result
                     : (cur == d_act ? d_alt : d_act);
        cudaError_t e = RunDenseMLPLayer(cublas_h, stream, pw.layers[l],
                                         seq, hidden, inter,
                                         cur, nxt, d_normed, d_gate, d_up);
        if (e != cudaSuccess) return e;
        cur = nxt;
    }
    return cudaSuccess;
}

} // namespace glm
} // namespace lsxcommon