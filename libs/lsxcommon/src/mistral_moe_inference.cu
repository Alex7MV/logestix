// Mistral-Small-4-119B decode step (Model Ingestion Protocol stage 5).
// Correctness-first: token-by-token prefill, one block per attention head, and
// the neutral MoE store for the routed experts. See the header for the graph.

#include "lsxcommon/mistral_moe_inference.h"
#include "lsx_kernels_common.cuh"

#include "mistral_moe_constants.h"
#include "lsx_device_math.h"
#include "lsx_kernels_act.cuh"

#include "lsxcommon/expert_device_cache.h"
#include "lsxcommon/lsx_attn_inference.h"
#include "lsxcommon/lsx_gemv.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/lsx_kv_cache.h"
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
__global__ void f32_copy_kernel(const float *src, float *dst, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        dst[i] = src[i];
}

// Mistral router: softmax over ALL experts, top-K (value desc, index asc),
// renormalized coeffs. Single deterministic thread; E <= 512.
__global__ void mistral_router_kernel(const float *logits, int E, int K,
                                      lsxcommon::moe::MoERoutedSlot *slots)
{
    if (threadIdx.x != 0) return;
    float p[512];
    float m = -INFINITY;
    for (int e = 0; e < E; ++e) m = fmaxf(m, logits[e]);
    float z = 0.0f;
    for (int e = 0; e < E; ++e)
    {
        p[e] = ExpfGlibc(logits[e] - m);
        z += p[e];
    }
    bool used[512];
    for (int e = 0; e < E; ++e) used[e] = false;
    float sm = 0.0f;
    int chosen[16];
    for (int k = 0; k < K; ++k)
    {
        int best = -1;
        float bv = -1.0f;
        for (int e = 0; e < E; ++e)
        {
            if (used[e]) continue;
            if (best < 0 || p[e] > bv)
            {
                best = e;
                bv = p[e];
            }
        }
        used[best] = true;
        chosen[k] = best;
        sm += bv;
    }
    for (int k = 0; k < K; ++k)
    {
        slots[k].eid = chosen[k];
        slots[k].coeff = p[chosen[k]] / sm;
    }
}
__global__ void add_f32_inplace_kernel(float *a, const float *b, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        a[i] += b[i];
}
// acc += y * alpha (y carries the SwiGLU prescale's 1/alpha factor: only the
// activation was scaled, so the down projection returns out/alpha -- one factor,
// not alpha^2).
__global__ void acc_add_half_scaled_kernel(float *acc, const __half *b,
                                           const float *alpha, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        acc[i] += __half2float(b[i]) * (*alpha);
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
                                int nope, lsxcommon::KvCacheGeometry kg,
                                lsxcommon::KvCacheGeometry vg, float *K,
                                float *V, int layer, int pos)
{
    const int heads = static_cast<int>(kg.kv_heads);
    const int qk_head = static_cast<int>(kg.head_dim);
    const int v_head = static_cast<int>(vg.head_dim);
    const int h = blockIdx.x;
    for (int d = threadIdx.x; d < qk_head; d += blockDim.x)
    {
        float k = (d < nope) ? kv_b[h * (nope + v_head) + d]
                             : k_rope[d - nope];
        K[kg.Offset(layer, h, pos, d)] = k;
    }
    for (int d = threadIdx.x; d < v_head; d += blockDim.x)
        V[vg.Offset(layer, h, pos, d)] = kv_b[h * (nope + v_head) + nope + d];
}

// ── MLA attention: one block per head, softmax over [0, pos] ────────────────
__global__ void mla_attn_kernel(const float *q, const float *K, const float *V,
                                lsxcommon::KvCacheGeometry kg,
                                lsxcommon::KvCacheGeometry vg, int layer,
                                int pos, int capacity, float scale, float temp,
                                float *scores, float *out)
{
    __shared__ float red[256];
    const int heads = static_cast<int>(kg.kv_heads);
    const int qk_head = static_cast<int>(kg.head_dim);
    const int v_head = static_cast<int>(vg.head_dim);
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    float *sc = scores + static_cast<int64_t>(h) * capacity;
    const float *qh = q + static_cast<int64_t>(h) * qk_head;

    for (int p = t; p <= pos; p += blockDim.x)
    {
        const float *kp = K + kg.Offset(layer, h, p);
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
            acc += sc[p] * V[vg.Offset(layer, h, p, d)];
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
    drop(s.d_shared_alpha);
    drop(s.d_swi_scratch);
    drop(s.d_mlp_scratch);
    drop(s.d_yg);
    drop(s.d_yu);
    drop(s.d_y2);
    drop(s.d_ygu_batch);
    drop(s.d_qpack_b);
    drop(s.d_qscales_b);
    drop(s.d_sw_batch);
    drop(s.d_alpha_batch);
    drop(s.d_slots);
    // h_slots is page-locked (cudaMallocHost), so it must not go through
    // cudaFree: that returned cudaErrorInvalidValue on every teardown.
    if (s.h_slots) cudaFreeHost(s.h_slots);
    drop(s.d_scores);
    drop(s.d_logits);
    lsxcommon::moe::FreeMoEExpertCache(s.cache);
    // Reset the pointers in place. `s = MistralSession{}` would construct a
    // temporary whose ~MistralSession re-enters this function (infinite
    // recursion) now that the session owns its teardown.
    s.d_k = nullptr;
    s.d_v = nullptr;
    s.d_h = nullptr;
    s.d_norm = nullptr;
    s.d_q_lat = nullptr;
    s.d_q_lat16 = nullptr;
    s.d_q = nullptr;
    s.d_kv = nullptr;
    s.d_kv_lat16 = nullptr;
    s.d_kv_b = nullptr;
    s.d_attn = nullptr;
    s.d_attn16 = nullptr;
    s.d_attn_out = nullptr;
    s.d_post = nullptr;
    s.d_moe_acc = nullptr;
    s.d_shared = nullptr;
    s.d_shared_alpha = nullptr;
    s.d_swi_scratch = nullptr;
    s.d_mlp_scratch = nullptr;
    s.d_yg = nullptr;
    s.d_yu = nullptr;
    s.d_y2 = nullptr;
    s.d_ygu_batch = nullptr;
    s.d_qpack_b = nullptr;
    s.d_qscales_b = nullptr;
    s.d_sw_batch = nullptr;
    s.d_alpha_batch = nullptr;
    s.d_slots = nullptr;
    s.h_slots = nullptr;
    s.d_scores = nullptr;
    s.d_logits = nullptr;
    s.capacity = 0;
    s.vocab = 0;
    s.length = 0;
}

cudaError_t AllocateMistralSession(MistralSession &s, const MistralDims &dims,
                                   int capacity, int /*slots*/,
                                   cudaStream_t stream)
{
    (void)stream;
    s.capacity = capacity;
    s.vocab = dims.vocab;
    const int H = dims.hidden;
    const int V = dims.vocab;
    const int heads = dims.heads;
    // pos-major (the repo default): [layer][pos][head][dim].
    s.kgeom = lsxcommon::KvCacheGeometry::Make(
        lsxcommon::KvCacheLayout::kPosMajor, capacity, heads, dims.qk_head,
        dims.layers);
    s.vgeom = lsxcommon::KvCacheGeometry::Make(
        lsxcommon::KvCacheLayout::kPosMajor, capacity, heads, dims.v_head,
        dims.layers);
    auto alloc = [](void **p, size_t bytes) -> cudaError_t {
        return cudaMalloc(p, bytes);
    };
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_k),
                     static_cast<size_t>(s.kgeom.Elements()) * sizeof(float)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_v),
                     static_cast<size_t>(s.vgeom.Elements()) * sizeof(float)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_h), H * sizeof(float)));
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
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_shared_alpha), sizeof(float)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_swi_scratch),
                     (static_cast<size_t>(dims.moe_inter) + 4) * sizeof(float)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_mlp_scratch),
                     2 * dims.moe_inter * sizeof(__half)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_yg),
                     2 * dims.moe_inter * sizeof(__half)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_yu),
                     dims.moe_inter * sizeof(__half)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_y2), H * sizeof(__half)));
    const int K = dims.top_k;
    const int I = dims.moe_inter;
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_ygu_batch),
                     static_cast<size_t>(K) * 2 * I * sizeof(__half)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_qpack_b),
                     static_cast<size_t>(K) * (I / 2)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_qscales_b),
                     static_cast<size_t>(K) * ((I + 15) / 16)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_sw_batch),
                     static_cast<size_t>(K) * I * sizeof(float)));
    MISTRAL_CK(alloc(reinterpret_cast<void **>(&s.d_alpha_batch),
                     2 * static_cast<size_t>(K) * sizeof(float)));
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
    return cudaSuccess;
}

void MistralReadLogits(const MistralSession &s, float *host_logits,
                       cudaStream_t stream)
{
    cudaMemcpyAsync(host_logits, s.d_logits,
                    static_cast<size_t>(s.vocab) * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
}

cudaError_t MistralSharedExpert(cudaStream_t stream,
                                const lsx_nvfp4::LsxNvfp4Weight &gate,
                                const lsx_nvfp4::LsxNvfp4Weight &up,
                                const lsx_nvfp4::LsxNvfp4Weight &down,
                                const __half *x, __half *y, __half *scratch,
                                float *d_alpha_out, float *d_swi_scratch)
{
    if (!gate.d_packed || !up.d_packed || !down.d_packed || !gate.d_scales ||
        !up.d_scales || !down.d_scales || gate.rows <= 0 || gate.cols <= 0 ||
        up.rows != gate.rows || up.cols != gate.cols || down.rows <= 0 ||
        down.cols != gate.rows)
        return cudaErrorInvalidValue;

    const int I = gate.rows;
    const int t256 = 256;
    // Drain any stale CUDA error from unrelated host-side work so the
    // cudaGetLastError() checks below only ever report this call's own
    // failures (see the same drain in MistralDecodeStep).
    cudaGetLastError();
    __half *g_out = scratch;
    __half *u_out = scratch + I;
    const auto gemv = [&stream](const lsx_nvfp4::LsxNvfp4Weight &w,
                                const __half *in, __half *out) {
        return lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(
            stream, in, w.d_packed, w.d_scales, w.global_scale, out, w.cols,
            w.rows, false);
    };
    cudaError_t e = gemv(gate, x, g_out);
    if (e != cudaSuccess) return e;
    e = gemv(up, x, u_out);
    if (e != cudaSuccess) return e;

    // Overflow-proof SwiGLU, same contract as the neutral scalar routed path:
    // the product is evaluated in FP32 and stored to the FP16 activation
    // buffer under a power-of-two prescale. alpha == 1 whenever the product
    // fits FP16, making this bit-identical to the plain SwigluKernel there.
    // This model carries massive activations, so the FP16 ceiling is reachable.
    float *sw_f32 = nullptr, *d_alpha = nullptr, *d_inv_alpha = nullptr;
    const bool own_swi = (d_swi_scratch == nullptr);
    if (own_swi)
    {
        const size_t sw_bytes = static_cast<size_t>(I) * sizeof(float);
        cudaError_t me = cudaMallocAsync(&sw_f32, sw_bytes, stream);
        if (me == cudaSuccess)
            me = cudaMallocAsync(&d_alpha, 2 * sizeof(float), stream);
        if (me != cudaSuccess)
        {
            if (sw_f32) cudaFreeAsync(sw_f32, stream);
            return cudaErrorInvalidValue;
        }
    }
    else
    {
        sw_f32 = d_swi_scratch;
        d_alpha = d_swi_scratch + I;
    }
    d_inv_alpha = d_alpha + 1;
    ::lsxcommon::kernels::SwigluF32Kernel<<<(I + t256 - 1) / t256, t256, 0, stream>>>(
        g_out, u_out, sw_f32, I);
    ::lsxcommon::kernels::SwigluAbsMaxKernel<<<1, t256, 0, stream>>>(sw_f32, d_alpha, I);
    ::lsxcommon::kernels::SwigluAlphaKernel<<<1, 32, 0, stream>>>(d_alpha, d_alpha, d_inv_alpha);
    ::lsxcommon::kernels::SwigluStoreScaledKernel<<<(I + t256 - 1) / t256, t256, 0, stream>>>(
        sw_f32, d_inv_alpha, g_out, I);
    e = cudaGetLastError();
    if (e == cudaSuccess) e = gemv(down, g_out, y);
    if (e != cudaSuccess)
    {
        cudaFreeAsync(sw_f32, stream);
        cudaFreeAsync(d_alpha, stream);
        return e;
    }
    if (d_alpha_out)
    {
        if (cudaMemcpyAsync(d_alpha_out, d_alpha, sizeof(float),
                           cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
            e = cudaErrorInvalidValue;
    }
    if (own_swi)
    {
        cudaFreeAsync(sw_f32, stream);
        cudaFreeAsync(d_alpha, stream);
    }
    return e;
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
    // Drain any stale CUDA error left by unrelated host-side work (the weight
    // loader's pinned staging leaves one on the synthetic fixture). MISTRAL_CK /
    // MISTRAL_LAUNCH report cudaGetLastError(), which would otherwise fail this
    // step on an error raised before it was entered.
    cudaGetLastError();
    const auto grid = [&](int n) { return (n + t256 - 1) / t256; };

    // Embedding row (BF16) -> FP16 hidden.
    MISTRAL_LAUNCH(lsxcommon::kernels::Bf16ToF32Kernel<<<grid(H), t256, 0, stream>>>(
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
        MISTRAL_LAUNCH(rms_norm_kernel<float><<<1, t256, 0, stream>>>(
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
            s.d_kv_b, s.d_kv + kvl, nope, s.kgeom, s.vgeom, s.d_k, s.d_v, l,
            pos));
        MISTRAL_LAUNCH(mla_attn_kernel<<<heads, 128, 0, stream>>>(
            s.d_q, s.d_k, s.d_v, s.kgeom, s.vgeom, l, pos, s.capacity,
            dims.attn_scale, temp, s.d_scores, s.d_attn));
        // o_proj + residual
        MISTRAL_LAUNCH(lsxcommon::kernels::F32ToBf16Kernel<<<grid(heads * vh), t256, 0, stream>>>(
            s.d_attn, s.d_attn16, heads * vh));
        MISTRAL_CK(lsxcommon::gemv::RunLxGemmVec(
            stream, lw.wo.data, s.d_attn16, 1, H, heads * vh,
            lsxcommon::gemv::LxDtype::kBF16, lsxcommon::gemv::LxDtype::kBF16,
            1.0f, s.d_attn_out, lsxcommon::gemv::LxDtype::kF32, 0.0f));
        MISTRAL_LAUNCH(f32_copy_kernel<<<grid(H), t256, 0, stream>>>(
            s.d_h, s.d_moe_acc, H));
        MISTRAL_LAUNCH(add_f32_inplace_kernel<<<grid(H), t256, 0, stream>>>(
            s.d_moe_acc, s.d_attn_out, H));
        MISTRAL_LAUNCH(f32_copy_kernel<<<grid(H), t256, 0, stream>>>(
            s.d_moe_acc, s.d_h, H));
        // post norm (FP16: the MoE FFN takes __half).
        MISTRAL_LAUNCH(rms_norm_kernel<float><<<1, t256, 0, stream>>>(
            s.d_h, lw.post_ln, nullptr, s.d_post, nullptr, H, eps, 1));

        // Router: the neutral device entry (moe::RunMoERouterTopK) -- softmax
        // over all E logits, top-K, renormalise, HF-parity selection with a
        // lower-index tie-break. The slot table lands on the device, which is
        // where the expert store is keyed, so it is staged back to the host
        // (RunMoERoutedFFN takes host slots) with one K-entry D2H copy.
        MISTRAL_CK(lsxcommon::moe::RunMoERouterTopK(
            stream, lw.router, s.d_post, 1, E, K, H, s.d_slots));
        MISTRAL_CK(cudaMemcpyAsync(s.h_slots, s.d_slots,
                                   K * sizeof(lsxcommon::moe::MoERoutedSlot),
                                   cudaMemcpyDeviceToHost, stream));
        MISTRAL_CK(cudaStreamSynchronize(stream));

        // acc = residual (pre-MoE hidden), then routed experts + shared expert.
        MISTRAL_LAUNCH(f32_copy_kernel<<<grid(H), t256, 0, stream>>>(
            s.d_h, s.d_moe_acc, H));
        std::string moe_error;
        const lsxcommon::moe::MoERoutedTcScratch tc_scratch{
            s.d_qpack_b, s.d_qscales_b, s.d_sw_batch, s.d_alpha_batch};
        MISTRAL_CK(lsxcommon::moe::RunMoERoutedFFN(
            stream, pw.expert_store, s.cache, l, s.h_slots, K, s.d_post, H, I,
            s.d_moe_acc, s.d_yg, s.d_yu, s.d_y2, moe_error, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, s.d_ygu_batch,
            nullptr, nullptr, nullptr, nullptr, &tc_scratch));
        // Shared expert (UNGATED: added directly, unlike Qwen/Ornith).
        MISTRAL_CK(MistralSharedExpert(stream, lw.shared_gate, lw.shared_up,
                                      lw.shared_down, s.d_post, s.d_shared,
                                      s.d_mlp_scratch, s.d_shared_alpha,
                                      s.d_swi_scratch));
        // The shared branch carries the SwiGLU prescale's 1/alpha; fold it back
        // in here, in the FP32 accumulator.
        MISTRAL_LAUNCH(acc_add_half_scaled_kernel<<<grid(H), t256, 0, stream>>>(
            s.d_moe_acc, s.d_shared, s.d_shared_alpha, H));
        MISTRAL_LAUNCH(f32_copy_kernel<<<grid(H), t256, 0, stream>>>(
            s.d_moe_acc, s.d_h, H));
    }

    // final norm + lm_head.
    MISTRAL_LAUNCH(rms_norm_kernel<float><<<1, t256, 0, stream>>>(
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
