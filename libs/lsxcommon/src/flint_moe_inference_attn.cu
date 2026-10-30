// Recursive Flint decode attention. One block per query head; the QK scores are
// produced warp-per-key (each warp reduces a full head_dim dot with shfl), then
// the sink joins the softmax denominator before the V mix. No atomicAdd; every
// reduction has a fixed order, so the result is bit-stable run to run.
#include "lsxcommon/flint_moe_inference_attn.h"

#include <algorithm>

namespace lsxcommon::flint {
namespace {

__global__ void flint_attn_decode_kernel(const __half *__restrict__ q, int pos,
                                         const __half *__restrict__ sink,
                                         int heads, int kv_heads, int head_dim,
                                         int window, float scale,
                                         const __half *__restrict__ k_cache,
                                         const __half *__restrict__ v_cache,
                                         __half *__restrict__ out)
{
    extern __shared__ float sm[];
    const int h = blockIdx.x;
    const int kh = h / (heads / kv_heads);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = blockDim.x >> 5;

    const int j0 = window > 0 ? max(0, pos - window + 1) : 0;
    const int nk = pos - j0 + 1;

    float *q_s = sm;                    // head_dim
    float *score = q_s + head_dim;      // nk
    float *acc = score + nk;            // head_dim

    for (int d = tid; d < head_dim; d += blockDim.x)
        q_s[d] = __half2float(q[h * head_dim + d]);
    __syncthreads();

    // Pass 1: QK scores. One warp per key, keys striped across warps.
    for (int j = j0 + warp; j <= pos; j += warps) {
        const __half *kj =
            k_cache + (static_cast<int64_t>(j) * kv_heads + kh) * head_dim;
        float dot = 0.0f;
        for (int d = lane; d < head_dim; d += 32)
            dot = __fadd_rn(dot, __fmul_rn(q_s[d], __half2float(kj[d])));
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            dot = __fadd_rn(dot, __shfl_down_sync(0xffffffffu, dot, off));
        if (lane == 0) score[j - j0] = __fmul_rn(dot, scale);
    }
    __syncthreads();

    // Pass 2: softmax with the learned sink as an extra column.
    __shared__ float s_m, s_inv;
    if (tid == 0) {
        float m = __half2float(sink[h]);
        for (int t = 0; t < nk; ++t) m = fmaxf(m, score[t]);
        float sum = 0.0f;
        for (int t = 0; t < nk; ++t) sum += __expf(score[t] - m);
        sum += __expf(__half2float(sink[h]) - m);
        s_m = m;
        s_inv = 1.0f / sum;
    }
    __syncthreads();
    const float m = s_m;
    const float inv = s_inv;

    for (int d = tid; d < head_dim; d += blockDim.x) acc[d] = 0.0f;
    __syncthreads();
    for (int j = j0; j <= pos; ++j) {
        const float pj = __expf(score[j - j0] - m) * inv;
        const __half *vj =
            v_cache + (static_cast<int64_t>(j) * kv_heads + kh) * head_dim;
        for (int d = tid; d < head_dim; d += blockDim.x)
            acc[d] = __fadd_rn(acc[d], __fmul_rn(pj, __half2float(vj[d])));
    }
    for (int d = tid; d < head_dim; d += blockDim.x)
        out[h * head_dim + d] = __float2half(acc[d]);
}

} // namespace

cudaError_t RunFlintAttnDecode(cudaStream_t stream, const __half *q, int pos,
                               const __half *sink, int heads, int kv_heads,
                               int head_dim, int window, float scale,
                               const __half *k_cache, const __half *v_cache,
                               __half *out)
{
    if (pos < 0 || heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        heads % kv_heads != 0)
        return cudaErrorInvalidValue;
    if (!sink) return cudaErrorInvalidValue;
    const int j0 = window > 0 ? std::max(0, pos - window + 1) : 0;
    const int nk = pos - j0 + 1;
    const size_t smem =
        (2 * static_cast<size_t>(head_dim) + static_cast<size_t>(nk)) *
        sizeof(float);
    flint_attn_decode_kernel<<<heads, 128, smem, stream>>>(
        q, pos, sink, heads, kv_heads, head_dim, window, scale, k_cache,
        v_cache, out);
    return cudaGetLastError();
}

} // namespace lsxcommon::flint
