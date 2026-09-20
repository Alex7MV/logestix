// GigaChat3.5 MLA decode (Plan 2b). One-token full-attention block: MLA
// latent projections, interleaved YaRN RoPE, scaled F32 softmax over the
// per-layer KV cache, sigmoid gating and o_proj. All activations are F32; the
// NVFP4 GEMV boundary converts to/from FP16. Mirrors modeling_gigachat3_5.py
// GigaChat35Attention.forward.
#include "lsxcommon/gigachat_moe_inference_attn.h"

#include "gigachat_moe_device_math.h"
#include "gigachat_moe_kernels.h"

#include <cuda_fp16.h>

#include <cmath>
#include <cstddef>

namespace lsxcommon::gigachat {
namespace {

constexpr int kAlign = 256;

inline int64_t AlignUp(int64_t v)
{
    return (v + kAlign - 1) / kAlign * kAlign;
}

__global__ void f32_to_f16_kernel(const float *__restrict__ x,
                                  __half *__restrict__ y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

__global__ void f16_to_f32_kernel(const __half *__restrict__ x,
                                  float *__restrict__ y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half2float(x[i]);
}

__global__ void scale_f32_to_f16_kernel(const float *__restrict__ x,
                                        __half *__restrict__ y, int n,
                                        float s)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i] * s);
}

// Interleaved YaRN RoPE: adjacent pairs of the rope subspace (equivalent to
// the reference view/transpose/rotate_half, which is a per-head permutation on
// both q and k and therefore score-preserving).
__global__ void rope_qk_kernel(float *__restrict__ q, float *__restrict__ k,
                               int heads, int qk, int nope, int half,
                               const float *__restrict__ cos_t,
                               const float *__restrict__ sin_t)
{
    const int h = blockIdx.x;
    for (int j = threadIdx.x; j < half; j += blockDim.x)
    {
        const float c = cos_t[j];
        const float s = sin_t[j];
        const int i0 = h * qk + nope + 2 * j;
        const float a = q[i0];
        const float b = q[i0 + 1];
        q[i0] = a * c - b * s;
        q[i0 + 1] = a * s + b * c;
    }
    if (blockIdx.x == 0)
    {
        for (int j = threadIdx.x; j < half; j += blockDim.x)
        {
            const float c = cos_t[j];
            const float s = sin_t[j];
            const float a = k[2 * j];
            const float b = k[2 * j + 1];
            k[2 * j] = a * c - b * s;
            k[2 * j + 1] = a * s + b * c;
        }
    }
}

__global__ void write_cache_kernel(const float *__restrict__ kvb,
                                   const float *__restrict__ krot,
                                   float *__restrict__ kc,
                                   float *__restrict__ vc, int pos, int heads,
                                   int nope, int rope, int vh)
{
    const int h = blockIdx.x;
    const int kvw = nope + vh;
    float *kb = kc + (static_cast<size_t>(pos) * heads + h) * (nope + rope);
    float *vb = vc + (static_cast<size_t>(pos) * heads + h) * vh;
    for (int i = threadIdx.x; i < nope; i += blockDim.x)
        kb[i] = kvb[h * kvw + i];
    for (int i = threadIdx.x; i < rope; i += blockDim.x)
        kb[nope + i] = krot[i];
    for (int i = threadIdx.x; i < vh; i += blockDim.x)
        vb[i] = kvb[h * kvw + nope + i];
}

__global__ void attn_scores_kernel(const float *__restrict__ q,
                                   const float *__restrict__ kc, int pos,
                                   int heads, int qk, float scale,
                                   float *__restrict__ scores, int capacity)
{
    const int h = blockIdx.x;
    const int t = blockIdx.y * blockDim.x + threadIdx.x;
    if (t > pos) return;
    const float *qt = q + static_cast<size_t>(h) * qk;
    const float *kt = kc + (static_cast<size_t>(t) * heads + h) * qk;
    float acc = 0.0f;
    for (int i = 0; i < qk; ++i) acc += qt[i] * kt[i];
    scores[static_cast<size_t>(h) * capacity + t] = acc * scale;
}

// F32 softmax over t in [0, pos], then out[h][d] = sum_t p[t]*v[t][h][d].
// Deterministic: the max/sum reduction runs on thread 0 behind a barrier.
__global__ void attn_softmax_kernel(const float *__restrict__ scores,
                                    const float *__restrict__ vc, int pos,
                                    int heads, int vh, float *__restrict__ attn,
                                    int capacity)
{
    const int h = blockIdx.x;
    extern __shared__ float sp[];
    const float *sc = scores + static_cast<size_t>(h) * capacity;
    if (threadIdx.x == 0)
    {
        float m = -INFINITY;
        for (int t = 0; t <= pos; ++t) m = fmaxf(m, sc[t]);
        float sum = 0.0f;
        for (int t = 0; t <= pos; ++t)
        {
            const float e = expf(sc[t] - m);
            sp[t] = e;
            sum += e;
        }
        const float inv = 1.0f / sum;
        for (int t = 0; t <= pos; ++t) sp[t] *= inv;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < vh; d += blockDim.x)
    {
        float acc = 0.0f;
        for (int t = 0; t <= pos; ++t)
            acc += sp[t] * vc[(static_cast<size_t>(t) * heads + h) * vh + d];
        attn[static_cast<size_t>(h) * vh + d] = acc;
    }
}

__global__ void gate_attn_kernel(const float *__restrict__ attn,
                                 const __half *__restrict__ gate,
                                 __half *__restrict__ out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __float2half(
            attn[i] * dev::Sigmoid(__half2float(gate[i])));
}

inline cudaError_t LaunchF32ToF16(const float *x, __half *y, int n,
                                  cudaStream_t s)
{
    if (n <= 0) return cudaSuccess;
    f32_to_f16_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, y, n);
    return cudaGetLastError();
}

inline cudaError_t LaunchF16ToF32(const __half *x, float *y, int n,
                                  cudaStream_t s)
{
    if (n <= 0) return cudaSuccess;
    f16_to_f32_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, y, n);
    return cudaGetLastError();
}

inline cudaError_t LaunchScale(const float *x, __half *y, int n, float s,
                               cudaStream_t stream)
{
    if (n <= 0) return cudaSuccess;
    scale_f32_to_f16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, y, n, s);
    return cudaGetLastError();
}

inline cudaError_t LaunchGemm(const lsx_nvfp4::LsxNvfp4Weight &w, const __half *x,
                              __half *y, cudaStream_t s)
{
    return lsx_nvfp4::RunLsxNvfp4Gemm(s, x, w.d_packed, w.d_scales,
                                      w.global_scale, y, 1, w.cols, w.rows);
}

} // namespace

bool GcMlaScratch::Alloc(const GcMlaDims &d, int capacity)
{
    Free();
    const int64_t qk = d.qk_nope + d.rope;
    const int64_t kvw = d.kv_lora + d.rope;
    const int64_t kvb = static_cast<int64_t>(d.heads) * (d.qk_nope + d.v_head);
    const int64_t hv = static_cast<int64_t>(d.heads) * d.v_head;

    int64_t off = 0;
    auto carve = [&](int64_t bytes) -> void * {
        off = AlignUp(off);
        void *p = static_cast<uint8_t *>(d_base) + off;
        off += bytes;
        return p;
    };
    int64_t total = 0;
    auto need = [&](int64_t bytes) { total = AlignUp(total) + bytes; };
    // Pass 1: size.
    need(static_cast<int64_t>(d.hidden) * 2);            // x_h
    need(static_cast<int64_t>(d.q_lora) * 2);            // q_a_h
    need(static_cast<int64_t>(d.q_lora) * 4);            // q_a_f
    need(static_cast<int64_t>(d.q_lora) * 4);            // q_a_n
    need(static_cast<int64_t>(d.q_lora) * 2);            // q_a_s
    need(static_cast<int64_t>(d.heads) * qk * 2);        // q_h
    need(static_cast<int64_t>(d.heads) * qk * 4);        // q_f
    need(kvw * 2);                                       // kv_h
    need(kvw * 4);                                       // kv_f
    need(static_cast<int64_t>(d.kv_lora) * 4);           // kpass_n
    need(static_cast<int64_t>(d.kv_lora) * 2);           // kpass_s
    need(kvb * 2);                                       // kv_b_h
    need(kvb * 4);                                       // kv_b_f
    need(static_cast<int64_t>(d.rope) * 4);              // k_rot
    need(hv * 2);                                        // gate_h
    need(hv * 4);                                        // attn_f
    need(hv * 2);                                        // attn_g_h
    need(static_cast<int64_t>(d.hidden) * 2);            // out_h
    need(static_cast<int64_t>(d.heads) * capacity * 4);  // scores
    if (total <= 0) return false;

    void *base = nullptr;
    if (cudaMalloc(&base, static_cast<size_t>(total)) != cudaSuccess)
        return false;
    d_base = base;
    bytes = total;
    x_h = carve(static_cast<int64_t>(d.hidden) * 2);
    q_a_h = carve(static_cast<int64_t>(d.q_lora) * 2);
    q_a_f = carve(static_cast<int64_t>(d.q_lora) * 4);
    q_a_n = carve(static_cast<int64_t>(d.q_lora) * 4);
    q_a_s = carve(static_cast<int64_t>(d.q_lora) * 2);
    q_h = carve(static_cast<int64_t>(d.heads) * qk * 2);
    q_f = carve(static_cast<int64_t>(d.heads) * qk * 4);
    kv_h = carve(kvw * 2);
    kv_f = carve(kvw * 4);
    kpass_n = carve(static_cast<int64_t>(d.kv_lora) * 4);
    kpass_s = carve(static_cast<int64_t>(d.kv_lora) * 2);
    kv_b_h = carve(kvb * 2);
    kv_b_f = carve(kvb * 4);
    k_rot = carve(static_cast<int64_t>(d.rope) * 4);
    gate_h = carve(hv * 2);
    attn_f = carve(hv * 4);
    attn_g_h = carve(hv * 2);
    out_h = carve(static_cast<int64_t>(d.hidden) * 2);
    scores = carve(static_cast<int64_t>(d.heads) * capacity * 4);
    return true;
}

void GcMlaScratch::Free()
{
    if (d_base)
    {
        cudaFree(d_base);
        d_base = nullptr;
    }
    bytes = 0;
    x_h = q_a_h = q_a_f = q_a_n = q_a_s = q_h = q_f = nullptr;
    kv_h = kv_f = kpass_n = kpass_s = kv_b_h = kv_b_f = k_rot = nullptr;
    gate_h = attn_f = attn_g_h = out_h = scores = nullptr;
}

cudaError_t RunGigaChatRopeQk(float *q, float *k, int heads, int qk, int nope,
                              int rope, const float *cos_t, const float *sin_t,
                              cudaStream_t stream)
{
    if (heads <= 0 || rope <= 0 || rope % 2 != 0) return cudaErrorInvalidValue;
    rope_qk_kernel<<<heads, 64, 0, stream>>>(q, k, heads, qk, nope, rope / 2,
                                             cos_t, sin_t);
    return cudaGetLastError();
}

cudaError_t RunGigaChatMlaDecode(const float *x, const GcLayerWeights &lw,
                                 const GcMlaDims &d, float norm_eps,
                                 float attn_scale, const float *rope_cos,
                                 const float *rope_sin, int pos, int capacity,
                                 float *k_cache, float *v_cache,
                                 GcMlaScratch &sc, float *out,
                                 cudaStream_t stream)
{
    const int qk = d.qk_nope + d.rope;
    const int kvw = d.kv_lora + d.rope;
    const int kvb = d.heads * (d.qk_nope + d.v_head);
    const int hv = d.heads * d.v_head;
    const int half = d.rope / 2;
    if (pos < 0 || pos >= capacity || !sc.d_base || d.rope % 2 != 0)
        return cudaErrorInvalidValue;
    if (lw.q_a.rows != d.q_lora || lw.q_a.cols != d.hidden ||
        lw.q_b.rows != d.heads * qk || lw.kv_b.rows != kvb)
        return cudaErrorInvalidValue;

    __half *x_h = static_cast<__half *>(sc.x_h);
    __half *q_a_h = static_cast<__half *>(sc.q_a_h);
    float *q_a_f = static_cast<float *>(sc.q_a_f);
    float *q_a_n = static_cast<float *>(sc.q_a_n);
    __half *q_a_s = static_cast<__half *>(sc.q_a_s);
    __half *q_h = static_cast<__half *>(sc.q_h);
    float *q_f = static_cast<float *>(sc.q_f);
    __half *kv_h = static_cast<__half *>(sc.kv_h);
    float *kv_f = static_cast<float *>(sc.kv_f);
    float *kpass_n = static_cast<float *>(sc.kpass_n);
    __half *kpass_s = static_cast<__half *>(sc.kpass_s);
    __half *kv_b_h = static_cast<__half *>(sc.kv_b_h);
    float *kv_b_f = static_cast<float *>(sc.kv_b_f);
    float *k_rot = static_cast<float *>(sc.k_rot);
    __half *gate_h = static_cast<__half *>(sc.gate_h);
    float *attn_f = static_cast<float *>(sc.attn_f);
    __half *attn_g_h = static_cast<__half *>(sc.attn_g_h);
    __half *out_h = static_cast<__half *>(sc.out_h);
    float *scores = static_cast<float *>(sc.scores);

    const float alpha_q =
        std::sqrt(static_cast<float>(d.hidden) / static_cast<float>(d.q_lora));
    const float alpha_kv = std::sqrt(static_cast<float>(d.hidden) /
                                     static_cast<float>(d.kv_lora));
    const float *cos_t = rope_cos + static_cast<size_t>(pos) * half;
    const float *sin_t = rope_sin + static_cast<size_t>(pos) * half;

#define GC_TRY(expr)                          \
    do {                                      \
        const cudaError_t _e = (expr);        \
        if (_e != cudaSuccess) return _e;     \
    } while (false)

    GC_TRY(LaunchF32ToF16(x, x_h, d.hidden, stream));

    // Query latent path.
    GC_TRY(LaunchGemm(lw.q_a, x_h, q_a_h, stream));
    GC_TRY(LaunchF16ToF32(q_a_h, q_a_f, d.q_lora, stream));
    GC_TRY(RunZeroCenteredGatedNorm(q_a_f, 1, d.q_lora, lw.q_a_norm.d_weight,
                                    lw.q_a_norm.d_gate_up,
                                    lw.q_a_norm.d_gate_down, norm_eps, q_a_n,
                                    stream));
    GC_TRY(LaunchScale(q_a_n, q_a_s, d.q_lora, alpha_q, stream));
    GC_TRY(LaunchGemm(lw.q_b, q_a_s, q_h, stream));
    GC_TRY(LaunchF16ToF32(q_h, q_f, d.heads * qk, stream));

    // Compressed KV path.
    GC_TRY(LaunchGemm(lw.kv_a, x_h, kv_h, stream));
    GC_TRY(LaunchF16ToF32(kv_h, kv_f, kvw, stream));
    GC_TRY(RunZeroCenteredGatedNorm(kv_f, 1, d.kv_lora, lw.kv_a_norm.d_weight,
                                    lw.kv_a_norm.d_gate_up,
                                    lw.kv_a_norm.d_gate_down, norm_eps, kpass_n,
                                    stream));
    GC_TRY(LaunchScale(kpass_n, kpass_s, d.kv_lora, alpha_kv, stream));
    GC_TRY(LaunchGemm(lw.kv_b, kpass_s, kv_b_h, stream));
    GC_TRY(LaunchF16ToF32(kv_b_h, kv_b_f, kvb, stream));
    GC_TRY(cudaMemcpyAsync(k_rot, kv_f + d.kv_lora,
                           static_cast<size_t>(d.rope) * sizeof(float),
                           cudaMemcpyDeviceToDevice, stream));

    GC_TRY(RunGigaChatRopeQk(q_f, k_rot, d.heads, qk, d.qk_nope, d.rope, cos_t,
                             sin_t, stream));
    write_cache_kernel<<<d.heads, 128, 0, stream>>>(
        kv_b_f, k_rot, k_cache, v_cache, pos, d.heads, d.qk_nope, d.rope,
        d.v_head);
    GC_TRY(cudaGetLastError());

    // Gated attention.
    GC_TRY(LaunchGemm(lw.attn_gate, x_h, gate_h, stream));
    attn_scores_kernel<<<dim3(d.heads, (pos + 1 + 127) / 128), 128, 0, stream>>>(
        q_f, k_cache, pos, d.heads, qk, attn_scale, scores, capacity);
    GC_TRY(cudaGetLastError());
    attn_softmax_kernel<<<d.heads, 256, capacity * sizeof(float), stream>>>(
        scores, v_cache, pos, d.heads, d.v_head, attn_f, capacity);
    GC_TRY(cudaGetLastError());
    {
        const int n = hv;
        gate_attn_kernel<<<(n + 255) / 256, 256, 0, stream>>>(attn_f, gate_h,
                                                              attn_g_h, n);
        GC_TRY(cudaGetLastError());
    }
    GC_TRY(LaunchGemm(lw.o_proj, attn_g_h, out_h, stream));
    GC_TRY(LaunchF16ToF32(out_h, out, d.hidden, stream));
#undef GC_TRY
    return cudaSuccess;
}

} // namespace lsxcommon::gigachat
