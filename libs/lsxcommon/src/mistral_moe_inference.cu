// Mistral-Small-4-119B decode step (Model Ingestion Protocol stage 5).
// Correctness-first: token-by-token prefill, one block per attention head, and
// the neutral MoE store for the routed experts. See the header for the graph.

#include "lsxcommon/mistral_moe_inference.h"

#include "mistral_moe_constants.h"
#include "lsx_device_math.h"

#include "lsxcommon/expert_device_cache.h"
#include "lsxcommon/lsx_attn_inference.h"
#include "lsxcommon/lsx_gemv.h"
#include "lsxcommon/log.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// Variadic so a kernel launch's `<<<grid, block, smem, stream>>>` commas do not
// split the macro arguments. MISTRAL_CK is for calls that RETURN cudaError_t;
// MISTRAL_LAUNCH is for `<<<...>>>` launches (which return void).
#define MISTRAL_CK(...)                              \
    do {                                             \
        const cudaError_t e_ = (__VA_ARGS__);        \
        if (e_ != cudaSuccess) return e_;            \
    } while (0)
#define MISTRAL_LAUNCH(...)                          \
    do {                                             \
        (__VA_ARGS__);                               \
        const cudaError_t e_ = cudaGetLastError();   \
        if (e_ != cudaSuccess) return e_;            \
    } while (0)

namespace lsxcommon {
namespace mistral_moe {
namespace {

namespace F = lsxcommon::mistral_moe::frozen;
using lsxcommon::device::ExpfGlibc;
using lsxcommon::device::RsqrtfGlibc;

__device__ __forceinline__ float ToF32(float v) { return v; }
__device__ __forceinline__ float ToF32(__half v) { return __half2float(v); }
__device__ __forceinline__ float ToF32(__nv_bfloat16 v) { return __bfloat162float(v); }

// ── elementwise ─────────────────────────────────────────────────────────────
__global__ void bf16_to_f16_kernel(const __nv_bfloat16 *src, __half *dst, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        dst[i] = __float2half(__bfloat162float(src[i]));
}
__global__ void f32_to_f16_kernel(const float *src, __half *dst, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        dst[i] = __float2half(src[i]);
}
__global__ void f32_to_bf16_kernel(const float *src, __nv_bfloat16 *dst, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        dst[i] = __float2bfloat16(src[i]);
}
__global__ void f16_to_f32_kernel(const __half *src, float *dst, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        dst[i] = __half2float(src[i]);
}
__global__ void add_f32_inplace_kernel(float *a, const float *b, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        a[i] += b[i];
}
__global__ void acc_add_half_kernel(float *acc, const __half *b, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        acc[i] += __half2float(b[i]);
}

// ── RMSNorm (plain weight multiply, no gating) ──────────────────────────────
// out_kind: 0 = F32, 1 = FP16, 2 = BF16.
template <typename XT>
__global__ void rms_norm_kernel(const XT *x, const __nv_bfloat16 *w,
                                float *out_f32, __half *out_h,
                                __nv_bfloat16 *out_bf16, int n, float eps,
                                int out_kind)
{
    __shared__ float red[256];
    const int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x)
    {
        const float v = ToF32(x[i]);
        ss += v * v;
    }
    red[tid] = ss;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float rstd = RsqrtfGlibc(red[0] / static_cast<float>(n) + eps);
    for (int i = tid; i < n; i += blockDim.x)
    {
        const float v = ToF32(x[i]) * rstd * ToF32(w[i]);
        if (out_kind == 1) out_h[i] = __float2half(v);
        else if (out_kind == 2) out_bf16[i] = __float2bfloat16(v);
        else out_f32[i] = v;
    }
}

// ── split-half RoPE (HF rotate_half pairing) ────────────────────────────────
__global__ void rope_q_kernel(float *q, int heads, int qk_head, int nope,
                              int rope, const float *cos_t, const float *sin_t,
                              int pos)
{
    const int half = rope / 2;
    const int total = heads * half;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x)
    {
        const int h = idx / half;
        const int j = idx % half;
        float *base = q + static_cast<int64_t>(h) * qk_head + nope;
        const float c = cos_t[static_cast<int64_t>(pos) * half + j];
        const float s = sin_t[static_cast<int64_t>(pos) * half + j];
        const float lo = base[j];
        const float hi = base[j + half];
        base[j] = lo * c - hi * s;
        base[j + half] = hi * c + lo * s;
    }
}
__global__ void rope_k_kernel(float *krot, int rope, const float *cos_t,
                              const float *sin_t, int pos)
{
    const int half = rope / 2;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < half;
         j += gridDim.x * blockDim.x)
    {
        const float c = cos_t[static_cast<int64_t>(pos) * half + j];
        const float s = sin_t[static_cast<int64_t>(pos) * half + j];
        const float lo = krot[j];
        const float hi = krot[j + half];
        krot[j] = lo * c - hi * s;
        krot[j + half] = hi * c + lo * s;
    }
}

// ── KV cache write (K = [k_nope | k_rope], V = kv_b's v slice) ──────────────
__global__ void kv_write_kernel(const float *kv_b, const float *k_rope,
                                int heads, int nope, int v_head, int qk_head,
                                float *K, float *V, int layer, int pos,
                                int capacity)
{
    const int h = blockIdx.x;
    for (int d = threadIdx.x; d < qk_head; d += blockDim.x)
    {
        float k = (d < nope) ? kv_b[h * (nope + v_head) + d]
                             : k_rope[d - nope];
        K[((static_cast<int64_t>(layer) * capacity + pos) * heads + h) *
              qk_head +
          d] = k;
    }
    for (int d = threadIdx.x; d < v_head; d += blockDim.x)
        V[((static_cast<int64_t>(layer) * capacity + pos) * heads + h) *
              v_head +
          d] = kv_b[h * (nope + v_head) + nope + d];
}

// ── MLA attention: one block per head, softmax over [0, pos] ────────────────
__global__ void mla_attn_kernel(const float *q, const float *K, const float *V,
                                int heads, int qk_head, int v_head, int layer,
                                int pos, int capacity, float scale, float temp,
                                float *scores, float *out)
{
    __shared__ float red[256];
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    float *sc = scores + static_cast<int64_t>(h) * capacity;
    const float *qh = q + static_cast<int64_t>(h) * qk_head;

    for (int p = t; p <= pos; p += blockDim.x)
    {
        const float *kp =
            K + ((static_cast<int64_t>(layer) * capacity + p) * heads + h) *
                    qk_head;
        float s = 0.0f;
        for (int d = 0; d < qk_head; ++d) s += qh[d] * kp[d];
        sc[p] = s * scale * temp;
    }
    __syncthreads();
    float m = -INFINITY;
    for (int p = t; p <= pos; p += blockDim.x) m = fmaxf(m, sc[p]);
    red[t] = m;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (t < s) red[t] = fmaxf(red[t], red[t + s]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();
    float sum = 0.0f;
    for (int p = t; p <= pos; p += blockDim.x)
    {
        const float e = ExpfGlibc(sc[p] - m);
        sc[p] = e;
        sum += e;
    }
    red[t] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    const float inv = 1.0f / red[0];
    for (int d = t; d < v_head; d += blockDim.x)
    {
        float acc = 0.0f;
        for (int p = 0; p <= pos; ++p)
            acc += sc[p] *
                   V[((static_cast<int64_t>(layer) * capacity + p) * heads + h) *
                         v_head +
                     d];
        out[static_cast<int64_t>(h) * v_head + d] = acc * inv;
    }
}

int EnvInt(const char *name, int fallback)
{
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return fallback;
    return std::max(1, std::atoi(v));
}

} // namespace

MistralDims MistralDimsFromWeights(const MistralPinnedWeights &pw)
{
    MistralDims d;
    const auto pick = [](int got, int frozen) {
        return got > 0 ? got : frozen;
    };
    d.hidden = pick(pw.hidden, static_cast<int>(pw.embed.cols));
    d.vocab = static_cast<int>(pw.embed.rows);
    d.layers = pw.num_layers;
    d.experts = pw.num_experts;
    d.heads = pick(pw.heads, F::kHeads);
    d.q_lora = pick(pw.q_lora, F::kQLoraRank);
    d.kv_lora = pick(pw.kv_lora, F::kKvLoraRank);
    d.qk_nope = pick(pw.qk_nope, F::kQkNopeHeadDim);
    d.qk_rope = pick(pw.qk_rope, F::kQkRopeHeadDim);
    d.qk_head = d.qk_nope + d.qk_rope;
    d.v_head = pick(pw.v_head, F::kVHeadDim);
    d.moe_inter = pick(pw.moe_inter, F::kMoeIntermediate);
    d.top_k = pick(pw.top_k, F::kTopK);
    d.attn_scale = F::kAttnScale;
    d.llama4_beta = static_cast<float>(F::kLlama4Beta);
    d.llama4_orig = F::kLlama4Orig;
    d.rms_eps = F::kRmsEps;
    return d;
}

void MistralSession::Free() { FreeMistralSession(*this); }

void FreeMistralSession(MistralSession &s)
{
    auto drop = [](void *p) {
        if (p) cudaFree(p);
    };
    drop(s.d_k);
    drop(s.d_v);
    drop(s.d_h);
    drop(s.d_norm);
    drop(s.d_q_lat);
    drop(s.d_q_lat16);
    drop(s.d_q);
    drop(s.d_kv);
    drop(s.d_kv_lat16);
    drop(s.d_kv_b);
    drop(s.d_attn);
    drop(s.d_attn16);
    drop(s.d_attn_out);
    drop(s.d_post);
    drop(s.d_moe_acc);
    drop(s.d_shared);
    drop(s.d_mlp_scratch);
    drop(s.d_yg);
    drop(s.d_yu);
    drop(s.d_y2);
    drop(s.d_slots);
    drop(s.h_slots);
    drop(s.d_scores);
    drop(s.d_logits);
    lsxcommon::moe::FreeMoEExpertCache(s.cache);
    s = MistralSession{};
}

cudaError_t AllocateMistralSession(MistralSession &s, const MistralDims &dims,
                                   int capacity, int slots, cudaStream_t stream)
{
    (void)stream;
    s.capacity = capacity;
    s.vocab = dims.vocab;
    const int H = dims.hidden;
    const int V = dims.vocab;
    const int heads = dims.heads;
    auto alloc = [](void **p, size_t bytes) -> cudaError_t {
        return cudaMalloc(p, bytes);
    };
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_k),
                     static_cast<size_t>(dims.layers) * capacity * heads *
                         dims.qk_head * sizeof(float)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_v),
                     static_cast<size_t>(dims.layers) * capacity * heads *
                         dims.v_head * sizeof(float)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_h), H * sizeof(__half)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_norm), H * sizeof(__half)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_q_lat),
                     dims.q_lora * sizeof(float)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_q_lat16),
                     dims.q_lora * sizeof(__half)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_q),
                     heads * dims.qk_head * sizeof(float)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_kv),
                     (dims.kv_lora + dims.qk_rope) * sizeof(float)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_kv_lat16),
                     dims.kv_lora * sizeof(__half)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_kv_b),
                     heads * (dims.qk_nope + dims.v_head) * sizeof(float)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_attn),
                     heads * dims.v_head * sizeof(float)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_attn16),
                     heads * dims.v_head * sizeof(__half)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_attn_out),
                     H * sizeof(float)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_post), H * sizeof(__half)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_moe_acc), H * sizeof(float)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_shared), H * sizeof(__half)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_mlp_scratch),
                     2 * dims.moe_inter * sizeof(__half)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_yg),
                     dims.moe_inter * sizeof(__half)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_yu),
                     dims.moe_inter * sizeof(__half)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_y2), H * sizeof(__half)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_slots),
                     dims.top_k * sizeof(lsxcommon::moe::MoERoutedSlot)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_scores),
                     static_cast<size_t>(heads) * capacity * sizeof(float)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_logits),
                     static_cast<size_t>(V) * sizeof(float)));
    MISTRAL_CK(cudaMallocHost(reinterpret_cast<void **>(&s.h_slots),
                              dims.top_k * sizeof(lsxcommon::moe::MoERoutedSlot)));
    return cudaSuccess;
}

bool BuildMistralYarnTables(int capacity, std::vector<float> &cos_t,
                            std::vector<float> &sin_t)
{
    const int rope = F::kRopeDim;
    if (capacity <= 0 || rope <= 0 || rope % 2 != 0) return false;
    const int half = rope / 2;
    const double base = F::kRopeTheta;
    const double factor = F::kYarnFactor;
    const double orig = F::kYarnOrig;
    const double ln_base = std::log(base);
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    auto corrected = [&](double rotations) {
        return rope * std::log(orig / (rotations * kTwoPi)) / (2.0 * ln_base);
    };
    const int low = std::max(
        static_cast<int>(std::floor(corrected(F::kYarnBetaFast))), 0);
    const int high = std::min(
        static_cast<int>(std::ceil(corrected(F::kYarnBetaSlow))), rope - 1);
    std::vector<double> freqs(static_cast<size_t>(half));
    for (int j = 0; j < half; ++j)
    {
        double f = 1.0 / std::pow(base, static_cast<double>(2 * j) / rope);
        const double span = std::max(high - low, 1);
        const double ramp = std::clamp((j - low) / span, 0.0, 1.0);
        const double smooth = 1.0 - ramp;
        f = f / factor * (1.0 - smooth) + f * smooth;
        freqs[static_cast<size_t>(j)] = f;
    }
    // yarn.apply_scale = false -> tables are NOT multiplied by an mscale.
    cos_t.assign(static_cast<size_t>(capacity) * half, 0.0f);
    sin_t.assign(static_cast<size_t>(capacity) * half, 0.0f);
    for (int p = 0; p < capacity; ++p)
        for (int j = 0; j < half; ++j)
        {
            const double a =
                static_cast<double>(p) * freqs[static_cast<size_t>(j)];
            cos_t[static_cast<size_t>(p) * half + j] =
                static_cast<float>(std::cos(a));
            sin_t[static_cast<size_t>(p) * half + j] =
                static_cast<float>(std::sin(a));
        }
    return true;
}

cudaError_t UploadMistralYarnTables(cudaStream_t stream, int capacity,
                                    const std::vector<float> &cos_t,
                                    const std::vector<float> &sin_t,
                                    float **dst_rc, float **dst_rs)
{
    (void)capacity;
    const size_t n = cos_t.size();
    if (n == 0 || sin_t.size() != n) return cudaErrorInvalidValue;
    MISTRAL_CK(cudaMalloc(reinterpret_cast<void **>(dst_rc), n * sizeof(float)));
    MISTRAL_CK(cudaMalloc(reinterpret_cast<void **>(dst_rs), n * sizeof(float)));
    MISTRAL_CK(cudaMemcpyAsync(*dst_rc, cos_t.data(), n * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    MISTRAL_CK(cudaMemcpyAsync(*dst_rs, sin_t.data(), n * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    return cudaGetLastError();
}

void MistralReadLogits(const MistralSession &s, float *host_logits,
                       cudaStream_t stream)
{
    cudaMemcpyAsync(host_logits, s.d_logits,
                    static_cast<size_t>(s.vocab) * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
}

cudaError_t MistralDecodeStep(MistralPinnedWeights &pw,
                              const MistralDims &dims, MistralSession &s,
                              int token_id, const float *d_rc,
                              const float *d_rs, int pos, cudaStream_t stream)
{
    const int H = dims.hidden;
    const int heads = dims.heads;
    const int qk = dims.qk_head;
    const int nope = dims.qk_nope;
    const int rope = dims.qk_rope;
    const int vh = dims.v_head;
    const int ql = dims.q_lora;
    const int kvl = dims.kv_lora;
    const int E = dims.experts;
    const int K = dims.top_k;
    const int I = dims.moe_inter;
    const float eps = dims.rms_eps;
    const int t256 = 256;
    const auto grid = [&](int n) { return (n + t256 - 1) / t256; };

    // Embedding row (BF16) -> FP16 hidden.
    MISTRAL_LAUNCH(bf16_to_f16_kernel<<<grid(H), t256, 0, stream>>>(
        pw.embed.data + static_cast<int64_t>(token_id) * H, s.d_h, H));

    // llama-4 logit temperature for this query position.
    const float temp =
        pos < dims.llama4_orig
            ? 1.0f
            : static_cast<float>(1.0 + dims.llama4_beta *
                                           std::log(std::floor(
                                                        pos / dims.llama4_orig) +
                                                    1.0));
    // RoPE row offset for the k_rope slice.
    const float *krope_base = s.d_kv + kvl;

    for (int l = 0; l < dims.layers; ++l)
    {
        const auto &lw = pw.layers[static_cast<size_t>(l)];
        // input norm
        MISTRAL_LAUNCH(rms_norm_kernel<__half><<<1, t256, 0, stream>>>(
            s.d_h, lw.input_ln, nullptr, nullptr, s.d_norm, H, eps, 2));
        // q_a
        MISTRAL_CK(lsxcommon::gemv::RunLxGemmVec(
            stream, lw.wq_a.data, s.d_norm, 1, ql, H,
            lsxcommon::gemv::LxDtype::kBF16, lsxcommon::gemv::LxDtype::kBF16,
            1.0f, s.d_q_lat, lsxcommon::gemv::LxDtype::kF32, 0.0f));
        // q_a norm + bf16
        MISTRAL_LAUNCH(rms_norm_kernel<float><<<1, t256, 0, stream>>>(
            s.d_q_lat, lw.q_a_norm, nullptr, nullptr, s.d_q_lat16, ql, eps, 2));
        // q_b
        MISTRAL_CK(lsxcommon::gemv::RunLxGemmVec(
            stream, lw.wq_b.data, s.d_q_lat16, 1, heads * qk, ql,
            lsxcommon::gemv::LxDtype::kBF16, lsxcommon::gemv::LxDtype::kBF16,
            1.0f, s.d_q, lsxcommon::gemv::LxDtype::kF32, 0.0f));
        // kv_a
        MISTRAL_CK(lsxcommon::gemv::RunLxGemmVec(
            stream, lw.wkv_a.data, s.d_norm, 1, kvl + rope, H,
            lsxcommon::gemv::LxDtype::kBF16, lsxcommon::gemv::LxDtype::kBF16,
            1.0f, s.d_kv, lsxcommon::gemv::LxDtype::kF32, 0.0f));
        // kv_a norm (only the latent part) + bf16
        MISTRAL_LAUNCH(rms_norm_kernel<float><<<1, t256, 0, stream>>>(
            s.d_kv, lw.kv_a_norm, nullptr, nullptr, s.d_kv_lat16, kvl, eps, 2));
        // kv_b
        MISTRAL_CK(lsxcommon::gemv::RunLxGemmVec(
            stream, lw.wkv_b.data, s.d_kv_lat16, 1, heads * (nope + vh), kvl,
            lsxcommon::gemv::LxDtype::kBF16, lsxcommon::gemv::LxDtype::kBF16,
            1.0f, s.d_kv_b, lsxcommon::gemv::LxDtype::kF32, 0.0f));
        // RoPE (split-half) on q's rope subspace and on the kv_a rope slice.
        {
            const int half = rope / 2;
            MISTRAL_LAUNCH(rope_q_kernel<<<grid(heads * half), t256, 0, stream>>>(
                s.d_q, heads, qk, nope, rope, d_rc, d_rs, pos));
            MISTRAL_LAUNCH(rope_k_kernel<<<grid(half), t256, 0, stream>>>(
                s.d_kv + kvl, rope, d_rc, d_rs, pos));
            (void)krope_base;
        }
        // KV write + attention.
        MISTRAL_LAUNCH(kv_write_kernel<<<heads, 128, 0, stream>>>(
            s.d_kv_b, s.d_kv + kvl, heads, nope, vh, qk, s.d_k, s.d_v, l, pos,
            s.capacity));
        MISTRAL_LAUNCH(mla_attn_kernel<<<heads, 128, 0, stream>>>(
            s.d_q, s.d_k, s.d_v, heads, qk, vh, l, pos, s.capacity,
            dims.attn_scale, temp, s.d_scores, s.d_attn));
        // o_proj + residual
        MISTRAL_LAUNCH(f32_to_bf16_kernel<<<grid(heads * vh), t256, 0, stream>>>(
            s.d_attn, s.d_attn16, heads * vh));
        MISTRAL_CK(lsxcommon::gemv::RunLxGemmVec(
            stream, lw.wo.data, s.d_attn16, 1, H, heads * vh,
            lsxcommon::gemv::LxDtype::kBF16, lsxcommon::gemv::LxDtype::kBF16,
            1.0f, s.d_attn_out, lsxcommon::gemv::LxDtype::kF32, 0.0f));
        MISTRAL_LAUNCH(f16_to_f32_kernel<<<grid(H), t256, 0, stream>>>(
            s.d_h, s.d_moe_acc, H));
        MISTRAL_LAUNCH(add_f32_inplace_kernel<<<grid(H), t256, 0, stream>>>(
            s.d_moe_acc, s.d_attn_out, H));
        MISTRAL_LAUNCH(f32_to_f16_kernel<<<grid(H), t256, 0, stream>>>(
            s.d_moe_acc, s.d_h, H));

        // post norm (FP16: the MoE router/FFN take __half).
        MISTRAL_LAUNCH(rms_norm_kernel<__half><<<1, t256, 0, stream>>>(
            s.d_h, lw.post_ln, nullptr, s.d_post, nullptr, H, eps, 1));

        // router (neutral: softmax over all experts, top-K, renorm)
        MISTRAL_CK(lsxcommon::moe::RunMoERouterTopK(
            stream, lw.router, s.d_post, 1, E, K, H, s.d_slots));
        // Host needs the ids to stage experts (streaming store).
        MISTRAL_CK(cudaMemcpyAsync(
            s.h_slots, s.d_slots, K * sizeof(lsxcommon::moe::MoERoutedSlot),
            cudaMemcpyDeviceToHost, stream));
        MISTRAL_CK(cudaStreamSynchronize(stream));

        // acc = residual (pre-MoE hidden), then routed experts + shared expert.
        MISTRAL_CK(lsxcommon::moe::RunMoEAccInitForTest(stream, s.d_h,
                                                        s.d_moe_acc, H));
        std::string moe_error;
        MISTRAL_CK(lsxcommon::moe::RunMoERoutedFFN(
            stream, pw.expert_store, s.cache, l, s.h_slots, K, s.d_post, H, I,
            s.d_moe_acc, s.d_yg, s.d_yu, s.d_y2, moe_error));
        // shared expert (UNGATED: added directly, unlike Qwen/Ornith).
        {
            lsxcommon::attn::LayerWeights slw;
            slw.loaded = true;
            slw.mlp_gate = lw.shared_gate;
            slw.mlp_up = lw.shared_up;
            slw.mlp_down = lw.shared_down;
            MISTRAL_CK(lsxcommon::qwen::RunAttnMLP(stream, slw, 1, s.d_post,
                                                   s.d_shared,
                                                   s.d_mlp_scratch));
        }
        MISTRAL_LAUNCH(acc_add_half_kernel<<<grid(H), t256, 0, stream>>>(
            s.d_moe_acc, s.d_shared, H));
        MISTRAL_LAUNCH(f32_to_f16_kernel<<<grid(H), t256, 0, stream>>>(
            s.d_moe_acc, s.d_h, H));
    }

    // final norm + lm_head.
    MISTRAL_LAUNCH(rms_norm_kernel<__half><<<1, t256, 0, stream>>>(
        s.d_h, pw.final_norm, nullptr, nullptr, s.d_norm, H, eps, 2));
    MISTRAL_CK(lsxcommon::gemv::RunLxGemmVec(
        stream, pw.head.data, s.d_norm, 1, dims.vocab, H,
        lsxcommon::gemv::LxDtype::kBF16, lsxcommon::gemv::LxDtype::kBF16, 1.0f,
        s.d_logits, lsxcommon::gemv::LxDtype::kF32, 0.0f));

    s.length = pos + 1;
    return cudaSuccess;
}

} // namespace mistral_moe
} // namespace lsxcommon
