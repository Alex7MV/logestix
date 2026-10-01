// AliceAI small device kernels: zero-centred RMSNorm, AttnRes depth-softmax
// mixer, sigmoid top-k router, SwiGLU. All FP32 accumulate; fixed reduction
// geometry (deterministic run-to-run). Projections go through the neutral
// NVFP4 GEMV layer, not here.
#include "lsxcommon/alice_moe_kernels.h"

#include <cuda_runtime.h>

namespace lsxcommon::alice {
namespace {

constexpr int kBlock = 256;

__device__ __forceinline__ float silu_f(float x)
{
    return x / (1.0f + __expf(-x));
}

__device__ float block_sum(float v, float *scratch)
{
    scratch[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) scratch[threadIdx.x] += scratch[threadIdx.x + s];
        __syncthreads();
    }
    return scratch[0];
}

__global__ void zc_rmsnorm_kernel(const __half *__restrict__ x,
                                  const __half *__restrict__ w,
                                  __half *__restrict__ out, int hidden,
                                  float eps, int zero_centered)
{
    const int row = blockIdx.x;
    const __half *xr = x + static_cast<size_t>(row) * hidden;
    __half *orr = out + static_cast<size_t>(row) * hidden;
    __shared__ float scratch[kBlock];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float v = __half2float(xr[i]);
        sum += v * v;
    }
    const float ms = block_sum(sum, scratch) / static_cast<float>(hidden);
    const float rstd = rsqrtf(ms + eps);
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float v = __half2float(xr[i]);
        const float wt = __half2float(w[i]);
        const float scale = zero_centered ? (1.0f + wt) : wt;
        orr[i] = __float2half(v * rstd * scale);
    }
}

__global__ void depth_mix_kernel(const __half *__restrict__ sources, int n_src,
                                 const __half *__restrict__ norm_w,
                                 const __half *__restrict__ proj_w,
                                 __half *__restrict__ out, int hidden, float eps)
{
    __shared__ float scratch[kBlock];
    __shared__ float logits[64];
    for (int s = 0; s < n_src; ++s) {
        const __half *p = sources + static_cast<size_t>(s) * hidden;
        float sq = 0.0f;
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
            const float v = __half2float(p[i]);
            sq += v * v;
        }
        const float rstd = rsqrtf(block_sum(sq, scratch) / hidden + eps);
        float dot = 0.0f;
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
            const float v = __half2float(p[i]) * rstd * __half2float(norm_w[i]);
            dot += v * __half2float(proj_w[i]);
        }
        const float total = block_sum(dot, scratch);
        if (threadIdx.x == 0) logits[s] = total;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float mx = logits[0];
        for (int s = 1; s < n_src; ++s) mx = fmaxf(mx, logits[s]);
        float sum = 0.0f;
        for (int s = 0; s < n_src; ++s) {
            logits[s] = __expf(logits[s] - mx);
            sum += logits[s];
        }
        for (int s = 0; s < n_src; ++s) logits[s] /= sum;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_src; ++s)
            acc += logits[s] * __half2float(sources[static_cast<size_t>(s) * hidden + i]);
        out[i] = __float2half(acc);
    }
}

// One block per expert: logits[e] = dot(x, gate[e]); sel[e] = sigmoid + bias
// (top-k selection); raw[e] = sigmoid (the un-biased weight HF gathers).
__global__ void router_score_kernel(const __half *__restrict__ x,
                                    const __half *__restrict__ gate,
                                    const __half *__restrict__ bias,
                                    int hidden, float *__restrict__ logits,
                                    float *__restrict__ sel,
                                    float *__restrict__ raw)
{
    const int e = blockIdx.x;
    const __half *gr = gate + static_cast<size_t>(e) * hidden;
    __shared__ float scratch[kBlock];
    float dot = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        dot += __half2float(x[i]) * __half2float(gr[i]);
    const float d = block_sum(dot, scratch);
    if (threadIdx.x == 0) {
        const float s = 1.0f / (1.0f + __expf(-d));
        logits[e] = d;
        sel[e] = s + __half2float(bias[e]);
        raw[e] = s;
    }
}

__global__ void router_topk_kernel(const float *__restrict__ logits,
                                   const float *__restrict__ sel,
                                   const float *__restrict__ raw, int n_exp,
                                   int top_k, int32_t *__restrict__ out_idx,
                                   float *__restrict__ out_w)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float wsum = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        int best = -1;
        float best_v = -INFINITY;
        for (int e = 0; e < n_exp; ++e) {
            bool used = false;
            for (int j = 0; j < k; ++j)
                if (out_idx[j] == e) { used = true; break; }
            if (used) continue;
            if (sel[e] > best_v) { best_v = sel[e]; best = e; }
        }
        out_idx[k] = best;
        out_w[k] = raw[best];  // un-biased score (HF gathers `scores`)
        wsum += out_w[k];
    }
    if (wsum > 0.0f)
        for (int k = 0; k < top_k; ++k) out_w[k] /= wsum;
    (void)logits;
}

__global__ void swiglu_kernel(const __half *__restrict__ a,
                              const __half *__restrict__ b,
                              __half *__restrict__ out, int64_t n)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(silu_f(__half2float(a[i])) * __half2float(b[i]));
}

} // namespace

cudaError_t RunAliceZcRmsNorm(cudaStream_t stream, const __half *x,
                              const __half *w, __half *out, int rows,
                              int hidden, float eps, bool zero_centered)
{
    zc_rmsnorm_kernel<<<rows, kBlock, 0, stream>>>(
        x, w, out, hidden, eps, zero_centered ? 1 : 0);
    return cudaGetLastError();
}

cudaError_t RunAliceDepthMix(cudaStream_t stream, const __half *sources,
                             int n_sources, const __half *norm_w,
                             const __half *proj_w, __half *out, int hidden,
                             float eps)
{
    if (n_sources > 64) return cudaErrorInvalidValue;
    depth_mix_kernel<<<1, kBlock, 0, stream>>>(sources, n_sources, norm_w,
                                               proj_w, out, hidden, eps);
    return cudaGetLastError();
}

cudaError_t RunAliceRouterTopK(cudaStream_t stream, const __half *x,
                               const __half *gate, const __half *bias,
                               int hidden, int n_experts, int top_k,
                               int32_t *out_idx, float *out_w,
                               float *out_logits)
{
    // Scratch: logits + scores (2*n_experts floats). Caller-visible pointers
    // are the outputs; scratch is allocated per call (prompt/decode-internal).
    float *scratch = nullptr;
    if (cudaMallocAsync(&scratch, sizeof(float) * 3 * n_experts, stream) !=
        cudaSuccess)
        return cudaErrorMemoryAllocation;
    float *sel = scratch + n_experts;
    float *raw = scratch + 2 * n_experts;
    router_score_kernel<<<n_experts, kBlock, 0, stream>>>(
        x, gate, bias, hidden, scratch, sel, raw);
    cudaError_t e1 = cudaGetLastError();
    if (e1 != cudaSuccess) {
        cudaFreeAsync(scratch, stream);
        return e1;
    }
    router_topk_kernel<<<1, 1, 0, stream>>>(scratch, sel, raw, n_experts, top_k,
                                            out_idx, out_w);
    cudaError_t e2 = cudaGetLastError();
    if (out_logits && e2 == cudaSuccess)
        cudaMemcpyAsync(out_logits, scratch, sizeof(float) * n_experts,
                        cudaMemcpyDeviceToDevice, stream);
    cudaFreeAsync(scratch, stream);
    return e2;
}

cudaError_t RunAliceSwiGLU(cudaStream_t stream, const __half *a,
                           const __half *b, __half *out, int64_t n)
{
    const int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);
    swiglu_kernel<<<grid, block, 0, stream>>>(a, b, out, n);
    return cudaGetLastError();
}

// ── KDA linear attention ────────────────────────────────────────────────────

__device__ __forceinline__ float softplus_f(float x)
{
    return fmaxf(x, 0.0f) + log1pf(__expf(-fabsf(x)));
}

__global__ void kda_conv_kernel(const __half *__restrict__ x,
                                const __half *__restrict__ w,
                                float *__restrict__ hist,
                                __half *__restrict__ out, int seq, int C, int k)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float wv[8];
    for (int j = 0; j < k; ++j) wv[j] = __half2float(w[c * k + j]);
    for (int i = 0; i < seq; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < k; ++j) {
            const int pos = i - (k - 1) + j;
            float xv;
            if (pos < 0)
                xv = hist[c * (k - 1) + (pos + (k - 1))];
            else
                xv = __half2float(x[pos * C + c]);
            acc += wv[j] * xv;
        }
        out[i * C + c] = __float2half(acc / (1.0f + __expf(-acc)));
    }
    for (int j = 0; j < k - 1; ++j) {
        const int idx = seq - (k - 1) + j;
        float xv;
        if (idx < 0)
            xv = hist[c * (k - 1) + (idx + (k - 1))];
        else
            xv = __half2float(x[idx * C + c]);
        hist[c * (k - 1) + j] = xv;
    }
}

// One block per (K==V) head; blockDim threads stride over KD for loads and VD
// for the state column update.
__global__ void kda_recur_kernel(const __half *__restrict__ q,
                                 const __half *__restrict__ k,
                                 const __half *__restrict__ v,
                                 const __half *__restrict__ alpha,
                                 const __half *__restrict__ beta,
                                 const __half *__restrict__ a_log,
                                 const __half *__restrict__ dt_bias,
                                 float *__restrict__ state,
                                 __half *__restrict__ out, int seq, int H,
                                 int KD, int VD)
{
    extern __shared__ float sh[];
    float *sq = sh;             // [KD]
    float *sk = sh + KD;        // [KD]
    float *sg = sh + 2 * KD;    // [KD]
    float *red = sh + 3 * KD;   // [blockDim.x]
    const int h = blockIdx.x;
    const float alog = __half2float(a_log[h]);
    const float kdim_scale = rsqrtf(static_cast<float>(KD));

    for (int t = 0; t < seq; ++t) {
        const __half *qt = q + (static_cast<size_t>(t) * H + h) * KD;
        const __half *kt = k + (static_cast<size_t>(t) * H + h) * KD;
        const __half *at = alpha + (static_cast<size_t>(t) * H + h) * KD;
        float qn = 0.0f, kn = 0.0f;
        for (int d = threadIdx.x; d < KD; d += blockDim.x) {
            const float qv = __half2float(qt[d]);
            const float kv = __half2float(kt[d]);
            sq[d] = qv;
            sk[d] = kv;
            qn += qv * qv;
            kn += kv * kv;
        }
        // block reduction of qn and kn.
        red[threadIdx.x] = qn;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        float qsum = red[0];
        __syncthreads();
        red[threadIdx.x] = kn;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        float ksum = red[0];
        const float qscale = rsqrtf(qsum + 1e-12f) * kdim_scale;
        const float kscale = rsqrtf(ksum + 1e-12f);
        for (int d = threadIdx.x; d < KD; d += blockDim.x) {
            sq[d] *= qscale;
            sk[d] *= kscale;
            const float a = __half2float(at[d]) +
                            __half2float(dt_bias[h * KD + d]);
            sg[d] = -__expf(alog) * softplus_f(a);
        }
        __syncthreads();
        const float beta_t =
            1.0f / (1.0f + __expf(-__half2float(beta[static_cast<size_t>(t) * H + h])));
        for (int vd = threadIdx.x; vd < VD; vd += blockDim.x) {
            float *col = state + (static_cast<size_t>(h) * KD) * VD + vd;
            float pred = 0.0f;
            for (int d = 0; d < KD; ++d) {
                const float g = __expf(sg[d]);
                col[static_cast<size_t>(d) * VD] *= g;
                pred += sk[d] * col[static_cast<size_t>(d) * VD];
            }
            const float delta = (__half2float(v[(static_cast<size_t>(t) * H + h) * VD + vd]) - pred) * beta_t;
            float o = 0.0f;
            for (int d = 0; d < KD; ++d) {
                const size_t idx = static_cast<size_t>(d) * VD;
                col[idx] += sk[d] * delta;
                o += sq[d] * col[idx];
            }
            out[(static_cast<size_t>(t) * H + h) * VD + vd] = __float2half(o);
        }
        __syncthreads();
    }
}

cudaError_t RunAliceKdaConv(cudaStream_t stream, const __half *x,
                            const __half *conv_w, float *history, __half *out,
                            int seq, int C, int k)
{
    const int block = 256;
    const int grid = (C + block - 1) / block;
    kda_conv_kernel<<<grid, block, 0, stream>>>(x, conv_w, history, out, seq, C,
                                                k);
    return cudaGetLastError();
}

cudaError_t RunAliceKdaRecurrence(cudaStream_t stream, const __half *q,
                                  const __half *k, const __half *v,
                                  const __half *alpha, const __half *beta,
                                  const __half *a_log, const __half *dt_bias,
                                  float *state, __half *out, int seq, int H,
                                  int KD, int VD)
{
    const int block = 256;
    const size_t smem = static_cast<size_t>(3 * KD + block) * sizeof(float);
    cudaFuncSetAttribute(kda_recur_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem));
    kda_recur_kernel<<<H, block, smem, stream>>>(
        q, k, v, alpha, beta, a_log, dt_bias, state, out, seq, H, KD, VD);
    return cudaGetLastError();
}

} // namespace lsxcommon::alice
