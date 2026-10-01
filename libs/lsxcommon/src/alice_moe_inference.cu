// AliceAI inference kernels: gated full attention (E4) and the KDA/full/MoE
// layer pipeline. Projections go through the neutral NVFP4 GEMV layer; the
// RoPE and causal-GQA primitives come from the documented shared kernel layer
// (qwen_inference.h).
#include "lsxcommon/alice_moe_inference.h"

#include "lsxcommon/alice_moe_kernels.h"
#include "lsxcommon/qwen_inference.h"

namespace lsxcommon::alice {
namespace {

// q_proj emits [H, 2*D] per token: query[head h] = qg[h*2D : h*2D+D], gate =
// qg[h*2D+D : (h+1)*2D] (chunk along the last dim), NOT two contiguous halves.
__global__ void split_qg_kernel(const __half *__restrict__ qg,
                                __half *__restrict__ q,
                                __half *__restrict__ gate, int64_t total,
                                int h, int d)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t hd = static_cast<int64_t>(h) * d;
    const int64_t row = i / hd;
    const int64_t rem = i % hd;
    const int64_t hh = rem / d;
    const int64_t dd = rem % d;
    const __half *base = qg + row * 2 * hd;
    q[i] = base[hh * 2 * d + dd];
    gate[i] = base[hh * 2 * d + d + dd];
}

__global__ void gate_mul_kernel(__half *__restrict__ o,
                                const __half *__restrict__ gate, int64_t n)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = 1.0f / (1.0f + __expf(-__half2float(gate[i])));
    o[i] = __float2half(__half2float(o[i]) * g);
}

} // namespace

cudaError_t RunAliceGatedAttentionCore(
    cudaStream_t stream, const __half *qg, const __half *k, const __half *v,
    const __half *q_norm_w, const __half *k_norm_w, const __half *cos,
    const __half *sin, __half *out, int seq, int H, int KVH, int D, int rot,
    float scale, int kv_base, __half *scratch)
{
    (void)scratch;
    const int64_t qn = static_cast<int64_t>(seq) * H * D;
    const int64_t kn = static_cast<int64_t>(seq) * KVH * D;
    __half *d_q = nullptr, *d_gate = nullptr, *d_qr = nullptr, *d_kr = nullptr;
    if (cudaMallocAsync(&d_q, qn * sizeof(__half), stream) != cudaSuccess)
        return cudaErrorMemoryAllocation;
    if (cudaMallocAsync(&d_gate, qn * sizeof(__half), stream) != cudaSuccess) {
        cudaFreeAsync(d_q, stream);
        return cudaErrorMemoryAllocation;
    }
    if (cudaMallocAsync(&d_qr, qn * sizeof(__half), stream) != cudaSuccess) {
        cudaFreeAsync(d_q, stream);
        cudaFreeAsync(d_gate, stream);
        return cudaErrorMemoryAllocation;
    }
    if (cudaMallocAsync(&d_kr, kn * sizeof(__half), stream) != cudaSuccess) {
        cudaFreeAsync(d_q, stream);
        cudaFreeAsync(d_gate, stream);
        cudaFreeAsync(d_qr, stream);
        return cudaErrorMemoryAllocation;
    }

    const int block = 256;
    split_qg_kernel<<<(qn + block - 1) / block, block, 0, stream>>>(
        qg, d_q, d_gate, qn, H, D);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess)
        e = RunAliceZcRmsNorm(stream, d_q, q_norm_w, d_q, seq * H, D, 1e-6f,
                              true);
    // Norm only the newly computed key rows (cache rows are already normed).
    if (e == cudaSuccess)
        e = RunAliceZcRmsNorm(stream, k + static_cast<size_t>(kv_base) * KVH * D,
                              k_norm_w,
                              const_cast<__half *>(k) +
                                  static_cast<size_t>(kv_base) * KVH * D,
                              seq * KVH, D, 1e-6f, true);
    if (e == cudaSuccess)
        e = qwen::RunQwenPartialRope(stream, d_q,
                                     k + static_cast<size_t>(kv_base) * KVH * D,
                                     cos, sin, d_qr, d_kr, seq, H, KVH, D, rot);
    if (e == cudaSuccess)
        e = cudaMemcpyAsync(const_cast<__half *>(k) +
                                static_cast<size_t>(kv_base) * KVH * D,
                            d_kr, kn * sizeof(__half), cudaMemcpyDeviceToDevice,
                            stream);
    if (e == cudaSuccess)
        e = qwen::RunQwenFullAttnKernel(stream, d_qr, k, v, out, seq, H, D, KVH,
                                        scale, kv_base);
    if (e == cudaSuccess)
        gate_mul_kernel<<<(qn + block - 1) / block, block, 0, stream>>>(
            out, d_gate, qn);
    cudaError_t e2 = cudaGetLastError();
    cudaFreeAsync(d_q, stream);
    cudaFreeAsync(d_gate, stream);
    cudaFreeAsync(d_qr, stream);
    cudaFreeAsync(d_kr, stream);
    return e != cudaSuccess ? e : e2;
}

} // namespace lsxcommon::alice
