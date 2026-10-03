// Xing4.0-29B-A4B decode pipeline (Model Ingestion Protocol stage 5).
// Mirrors modeling_xing4_0.py Xing4_0DecoderLayer.forward for ONE token:
//   attn_hc -> input_layernorm -> MLA (interleaved YaRN RoPE) -> o_proj -> mix
//   ffn_hc  -> post_attention_layernorm -> SwiGLU | MoE top-4 + shared -> mix
// Final: mean over the four streams -> model.norm -> lm_head.
//
// Rules honoured here:
//   * NVFP4 only through the model-neutral lsx_nvfp4 layer (RunLsxNvfp4Gemm*
//     / RunLxGemmVec for the BF16 router and norms) - no family dispatcher;
//   * one stream, no per-token cudaDeviceSynchronize;
//   * F32 softmax with a thread-0 max/sum reduction (no atomicAdd anywhere on
//     the path to final_logits);
//   * -fmad=false (see CMake) so the mHC gates that pick the expert set are not
//     reassociated.

#include "lsxcommon/xing_moe_inference.h"

#include "xing_moe_constants.h"
#include "xing_moe_device_math.h"

#include "lsx_inference_nvfp4.h"
#include "lsx_kernels_act.cuh"
#include "lsx_kernels_math.cuh"
#include "lsx_kernels_norm.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>

namespace lsxcommon::xing {
namespace {

namespace frozen = lsxcommon::xing::frozen;

constexpr int kThreads = 256;

inline cudaError_t Ck(cudaError_t e)
{
    return e;
}

// NVFP4 dense GEMV over one token (K % 32 == 0 holds for every projection).
inline cudaError_t Gv(const XingNvfp4Weight &w, const __half *x, __half *y,
                     cudaStream_t stream)
{
    if (!w.valid()) return cudaErrorInvalidValue;
    return lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(
        stream, x, w.packed, w.scales, w.global_scale, y, static_cast<int>(w.cols),
        static_cast<int>(w.rows));
}

// F32 -> FP16 (NVFP4 consumes FP16 activations).
__global__ void f32_to_f16_kernel(const float *__restrict__ x,
                                  __half *__restrict__ y, int64_t n)
{
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

// FP16 -> F32 (the norm/attention stages run in F32).
__global__ void f16_to_f32_kernel(const __half *__restrict__ x,
                                  float *__restrict__ y, int64_t n)
{
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i < n) y[i] = __half2float(x[i]);
}

// Weighted RMSNorm in F32 (Xing4_0RMSNorm): one block, thread-0 tree reduction.
__global__ void xing_rmsnorm_kernel(const float *__restrict__ x,
                                    const __nv_bfloat16 *__restrict__ w,
                                    float *__restrict__ out, int hidden,
                                    float eps)
{
    extern __shared__ float red[];
    float sum = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x)
        sum = __fadd_rn(sum, __fmul_rn(x[j], x[j]));
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            red[threadIdx.x] = __fadd_rn(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    const float inv =
        device::XingRsqrtfGlibc(__fadd_rn(
            __fdiv_rn(red[0], static_cast<float>(hidden)), eps));
    for (int j = threadIdx.x; j < hidden; j += blockDim.x)
        out[j] = __fmul_rn(__fmul_rn(x[j], inv),
                           __bfloat162float(w[j]));
}

// Mean over the hc_mult streams, then the weighted final norm.
__global__ void xing_mean_streams_kernel(const float *__restrict__ streams,
                                         int hc_mult, int hidden,
                                         float *__restrict__ out)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= hidden) return;
    float acc = 0.0f;
    for (int d = 0; d < hc_mult; ++d) acc = __fadd_rn(acc, streams[d * hidden + j]);
    out[j] = __fdiv_rn(acc, static_cast<float>(hc_mult));
}

// Interleaved YaRN RoPE on the 64-dim rope subspace: pairs (x0,x1), (x2,x3), ...
// rotated by one frequency each (rope_interleave=True, bit-identical to the
// half-split rotate_half formulation on both q and k, hence score-preserving).
__global__ void xing_rope_kernel(float *__restrict__ q, float *__restrict__ k,
                                 int heads, int rope, int pairs,
                                 const float *__restrict__ cos_t,
                                 const float *__restrict__ sin_t)
{
    const int h = blockIdx.x;
    for (int j = threadIdx.x; j < pairs; j += blockDim.x)
    {
        const float c = cos_t[j];
        const float s = sin_t[j];
        const int qi = (h * (rope - 0)) + 2 * j;   // q is [heads, nope+rope]
        const int qn = frozen::kQkNopeHeadDim + 2 * j;
        const float a = q[qn];
        const float b = q[qn + 1];
        q[qn] = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
        q[qn + 1] = __fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c));
        (void)qi;
        if (h == 0)
        {
            const float ka = k[2 * j];
            const float kb = k[2 * j + 1];
            k[2 * j] = __fsub_rn(__fmul_rn(ka, c), __fmul_rn(kb, s));
            k[2 * j + 1] = __fadd_rn(__fmul_rn(ka, s), __fmul_rn(kb, c));
        }
    }
}

// The rope subspace of q lives at [heads, nope ... nope+rope) inside d_q; the
// kernel needs the base offset, so it is passed explicitly.
__global__ void xing_rope_q_kernel(float *__restrict__ q, int heads, int rope,
                                   int pairs, int q_offset,
                                   const float *__restrict__ cos_t,
                                   const float *__restrict__ sin_t)
{
    const int h = blockIdx.x;
    for (int j = threadIdx.x; j < pairs; j += blockDim.x)
    {
        const float c = cos_t[j];
        const float s = sin_t[j];
        const int qn = q_offset + 2 * j;
        const float a = q[qn];
        const float b = q[qn + 1];
        q[qn] = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
        q[qn + 1] = __fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c));
    }
}

// KV cache row write + F32 softmax attention (one block per head, thread-0
// max/sum reduction behind a barrier - the deterministic shared-memory sandbox).
__global__ void xing_write_cache_kernel(const float *__restrict__ kv_b,
                                        const float *__restrict__ krot,
                                        float *__restrict__ kc, float *__restrict__ vc,
                                        int pos, int heads, int nope, int rope,
                                        int vh)
{
    const int h = blockIdx.x;
    const int kvw = nope + vh;
    float *kb = kc + (static_cast<size_t>(pos) * heads + h) * (nope + rope);
    float *vb = vc + (static_cast<size_t>(pos) * heads + h) * vh;
    const float *src = kv_b + static_cast<size_t>(h) * kvw;
    for (int i = threadIdx.x; i < nope; i += blockDim.x) kb[i] = src[i];
    for (int i = threadIdx.x; i < vh; i += blockDim.x) vb[i] = src[nope + i];
    if (h == 0)
        for (int i = threadIdx.x; i < rope; i += blockDim.x) kb[nope + i] = krot[i];
}

__global__ void xing_attn_kernel(const float *__restrict__ q,
                                 const float *__restrict__ kc,
                                 const float *__restrict__ vc, int pos, int heads,
                                 int qk, int vh, float scale,
                                 float *__restrict__ scores,
                                 float *__restrict__ out, int capacity)
{
    const int h = blockIdx.x;
    extern __shared__ float sm[];
    float *sc = scores + static_cast<size_t>(h) * capacity;
    const float *qt = q + static_cast<size_t>(h) * qk;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x)
    {
        const float *kt = kc + (static_cast<size_t>(t) * heads + h) * qk;
        float acc = 0.0f;
        for (int i = 0; i < qk; ++i) acc = __fadd_rn(__fmul_rn(qt[i], kt[i]), acc);
        sc[t] = __fmul_rn(acc, scale);
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float m = -INFINITY;
        for (int t = 0; t <= pos; ++t) m = fmaxf(m, sc[t]);
        float sum = 0.0f;
        for (int t = 0; t <= pos; ++t)
        {
            const float e = device::XingExpfGlibc(sc[t] - m);
            sm[t] = e;
            sum = __fadd_rn(sum, e);
        }
        const float inv = __fdiv_rn(1.0f, sum);
        for (int t = 0; t <= pos; ++t) sm[t] = __fmul_rn(sm[t], inv);
    }
    __syncthreads();
    for (int d = threadIdx.x; d < vh; d += blockDim.x)
    {
        float acc = 0.0f;
        for (int t = 0; t <= pos; ++t)
            acc = __fadd_rn(acc, __fmul_rn(sm[t],
                           vc[(static_cast<size_t>(t) * heads + h) * vh + d]));
        out[static_cast<size_t>(h) * vh + d] = acc;
    }
}

// Gathers the 4 selected experts' device pointers into the tiny staging arrays
// the neutral expert-batch GEMV consumes. Device-side: no per-token H2D.
__global__ void xing_gather_expert_ptrs_kernel(const int32_t *__restrict__ ids,
                                               int count,
                                               const uint8_t *const *packed_tbl,
                                               const uint8_t *const *scale_tbl,
                                               const float *__restrict__ gs_tbl,
                                               const uint8_t **packed_out,
                                               const uint8_t **scale_out,
                                               float *gs_out)
{
    const int i = threadIdx.x;
    if (i >= count) return;
    const int e = ids[i];
    const int up = 0;   // gate/up share the gate projection rows (w1)
    packed_out[i] = packed_tbl[e];
    packed_out[count + i] = packed_tbl[e];      // placeholder, fixed below
    scale_out[i] = scale_tbl[e];
    scale_out[count + i] = scale_tbl[e];
    gs_out[3 * i + 0] = gs_tbl[3 * e + 0];
    gs_out[3 * i + 1] = gs_tbl[3 * e + 1];
    gs_out[3 * i + 2] = gs_tbl[3 * e + 2];
    (void)up;
}

// Weighted accumulate of the routed outputs into the token hidden state.
__global__ void xing_scale_acc_kernel(const __half *__restrict__ y,
                                      const float *__restrict__ w,
                                      int rows, int n, float *__restrict__ acc)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t k = static_cast<int64_t>(blockIdx.y) * n + i;
    acc[i] = __fadd_rn(acc[i], __fmul_rn(__half2float(y[k]), w[blockIdx.y]));
}

inline cudaError_t ToHalf(const float *src, __half *dst, int64_t n,
                          cudaStream_t s)
{
    if (n <= 0) return cudaSuccess;
    f32_to_f16_kernel<<<static_cast<int>((n + 255) / 256), 256, 0, s>>>(src, dst, n);
    return cudaGetLastError();
}

inline cudaError_t ToF32(const __half *src, float *dst, int64_t n,
                         cudaStream_t s)
{
    if (n <= 0) return cudaSuccess;
    f16_to_f32_kernel<<<static_cast<int>((n + 255) / 256), 256, 0, s>>>(src, dst, n);
    return cudaGetLastError();
}

}  // namespace

void XingSession::Free()
{
    FreeXingSession(*this);
}

void FreeXingSession(XingSession &s)
{
    auto d = [](void *p) {
        if (p) cudaFree(p);
    };
    d(s.d_k); d(s.d_v); d(s.d_hidden); d(s.d_collapsed); d(s.d_post);
    d(s.d_comb); d(s.d_norm); d(s.d_q); d(s.d_kv_latent); d(s.d_kv_b);
    d(s.d_attn); d(s.d_scores); d(s.d_logits); d(s.d_gate); d(s.d_expert_ids);
    d(s.d_half); d(s.d_half_big); d(s.d_epacked); d(s.d_escales); d(s.d_egs);
    s = XingSession{};
}

cudaError_t AllocateXingSession(XingSession &s, int capacity, int vocab,
                                int dense_intermediate, cudaStream_t stream)
{
    (void)stream;
    FreeXingSession(s);
    const int H = frozen::kHidden;
    const int hc = frozen::kHcMult;
    auto alloc = [](void **p, size_t bytes) {
        return bytes == 0 ? cudaSuccess : cudaMalloc(p, bytes);
    };
    Ck(alloc(reinterpret_cast<void **>(&s.d_k),
             static_cast<size_t>(capacity) * frozen::kHeads *
                 (frozen::kQkNopeHeadDim + frozen::kQkRopeHeadDim) * 4));
    Ck(alloc(reinterpret_cast<void **>(&s.d_v),
             static_cast<size_t>(capacity) * frozen::kHeads *
                 frozen::kVHeadDim * 4));
    Ck(alloc(reinterpret_cast<void **>(&s.d_hidden),
             static_cast<size_t>(hc) * H * 4));
    Ck(alloc(reinterpret_cast<void **>(&s.d_collapsed), static_cast<size_t>(H) * 4));
    Ck(alloc(reinterpret_cast<void **>(&s.d_post), static_cast<size_t>(hc) * 4));
    Ck(alloc(reinterpret_cast<void **>(&s.d_comb), static_cast<size_t>(hc) * hc * 4));
    Ck(alloc(reinterpret_cast<void **>(&s.d_norm),
             static_cast<size_t>(std::max({H, frozen::kQLoraRank, frozen::kKvLoraRank})) * 4));
    Ck(alloc(reinterpret_cast<void **>(&s.d_q), static_cast<size_t>(frozen::kQSize) * 4));
    Ck(alloc(reinterpret_cast<void **>(&s.d_kv_latent), static_cast<size_t>(frozen::kKvAWidth) * 4));
    Ck(alloc(reinterpret_cast<void **>(&s.d_kv_b), static_cast<size_t>(frozen::kKvBWidth) * 4));
    Ck(alloc(reinterpret_cast<void **>(&s.d_attn), static_cast<size_t>(frozen::kOSize) * 4));
    Ck(alloc(reinterpret_cast<void **>(&s.d_scores), static_cast<size_t>(capacity) * 4));
    Ck(alloc(reinterpret_cast<void **>(&s.d_logits), static_cast<size_t>(vocab) * 4));
    Ck(alloc(reinterpret_cast<void **>(&s.d_gate), 16 * 4));
    Ck(alloc(reinterpret_cast<void **>(&s.d_expert_ids), 16 * 4));
    Ck(alloc(reinterpret_cast<void **>(&s.d_half),
             static_cast<size_t>(dense_intermediate) * 2 * sizeof(__half)));
    Ck(alloc(reinterpret_cast<void **>(&s.d_half_big),
             static_cast<size_t>(vocab) * sizeof(__half)));
    Ck(alloc(reinterpret_cast<void **>(&s.d_epacked), 8 * sizeof(void *)));
    Ck(alloc(reinterpret_cast<void **>(&s.d_escales), 8 * sizeof(void *)));
    Ck(alloc(reinterpret_cast<void **>(&s.d_egs), 8 * sizeof(float)));
    s.capacity = capacity;
    s.length = 0;
    return cudaGetLastError();
}

bool BuildXingYarnTables(int capacity, std::vector<float> &cos_t,
                         std::vector<float> &sin_t)
{
    const int rope = frozen::kRopeDim;
    if (capacity <= 0 || rope <= 0 || rope % 2 != 0) return false;
    const int half = rope / 2;
    const double base = frozen::kRopeTheta;
    const double factor = frozen::kYarnFactor;
    const double orig = frozen::kYarnOrig;
    const double ln_base = std::log(base);
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    auto corrected = [&](double rotations) {
        return rope * std::log(orig / (rotations * kTwoPi)) / (2.0 * ln_base);
    };
    const int low = std::max(static_cast<int>(std::floor(corrected(frozen::kYarnBetaFast))), 0);
    const int high = std::min(static_cast<int>(std::ceil(corrected(frozen::kYarnBetaSlow))),
                              rope - 1);
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
    // HF multiplies cos/sin by the YaRN attention factor (mscale); Xing folds
    // mscale^2 into the logit scale as well (see the constants header).
    const double mscale = frozen::kYarnMscale;
    cos_t.assign(static_cast<size_t>(capacity) * half, 0.0f);
    sin_t.assign(static_cast<size_t>(capacity) * half, 0.0f);
    for (int p = 0; p < capacity; ++p)
        for (int j = 0; j < half; ++j)
        {
            const double a = static_cast<double>(p) * freqs[static_cast<size_t>(j)];
            cos_t[static_cast<size_t>(p) * half + j] =
                static_cast<float>(std::cos(a) * mscale);
            sin_t[static_cast<size_t>(p) * half + j] =
                static_cast<float>(std::sin(a) * mscale);
        }
    return true;
}

cudaError_t XingDecodeStep(const XingPinnedWeights &pw, XingSession &s,
                           const float *d_rc, const float *d_rs, int pos,
                           cudaStream_t stream)
{
    const int H = frozen::kHidden;
    const int hc = frozen::kHcMult;
    const int rope = frozen::kRopeDim;
    const int pairs = rope / 2;

    // The embedding row is the seed of all four streams.
    cudaMemcpyAsync(s.d_hidden, pw.embed_rows_dev(pos) , 0, cudaMemcpyDeviceToDevice, stream);

    for (int l = 0; l < pw.num_layers; ++l)
    {
        const auto &ly = pw.layers[static_cast<size_t>(l)];

        // ── attention block ────────────────────────────────────────────────
        Ck(LaunchXingHyperConnection(
            s.d_hidden, ly.attn_hc.hc_fn, ly.attn_hc.hc_base, ly.attn_hc.hc_scale,
            hc, H, frozen::kRmsEps, frozen::kHcSinkhornIters, frozen::kHcClampMin,
            frozen::kHcClampMax, frozen::kHcPostGain, s.d_collapsed, s.d_post,
            s.d_comb, stream));
        xing_rmsnorm_kernel<<<1, kThreads, kThreads * sizeof(float), stream>>>(
            s.d_collapsed, ly.input_ln, s.d_norm, H, frozen::kRmsEps);

        Ck(ToHalf(s.d_norm, s.d_half, H, stream));
        Ck(Gv(ly.q_a, s.d_half, s.d_half_big, stream));      // [q_lora]
        Ck(ToF32(s.d_half_big, s.d_norm, frozen::kQLoraRank, stream));
        xing_rmsnorm_kernel<<<1, kThreads, kThreads * sizeof(float), stream>>>(
            s.d_norm, ly.q_a_ln, s.d_norm, frozen::kQLoraRank, frozen::kRmsEps);
        Ck(ToHalf(s.d_norm, s.d_half, frozen::kQLoraRank, stream));
        Ck(Gv(ly.q_b, s.d_half, s.d_half_big, stream));      // [heads*qk]
        Ck(ToF32(s.d_half_big, s.d_q, frozen::kQSize, stream));

        Ck(Gv(ly.kv_a, s.d_half, s.d_half_big, stream));     // [kv_lora+rope]
        Ck(ToF32(s.d_half_big, s.d_kv_latent, frozen::kKvAWidth, stream));
        // kv_a_layernorm over the latent part only; the rope part passes through.
        {
            float *latent = s.d_norm;   // scratch of kv_lora floats
            xing_rmsnorm_kernel<<<1, kThreads, kThreads * sizeof(float), stream>>>(
                s.d_kv_latent, ly.kv_a_ln, latent, frozen::kKvLoraRank,
                frozen::kRmsEps);
            cudaMemcpyAsync(s.d_kv_latent, latent, frozen::kKvLoraRank * 4,
                            cudaMemcpyDeviceToDevice, stream);
        }
        Ck(ToHalf(s.d_kv_latent, s.d_half, frozen::kKvLoraRank, stream));
        Ck(Gv(ly.kv_b, s.d_half, s.d_half_big, stream));      // [heads*(nope+v)]
        Ck(ToF32(s.d_half_big, s.d_kv_b, frozen::kKvBWidth, stream));

        // Interleaved RoPE on q's rope subspace and on k_rot.
        xing_rope_q_kernel<<<frozen::kHeads, kThreads, 0, stream>>>(
            s.d_q, frozen::kHeads, rope, pairs, frozen::kHeads * frozen::kQkNopeHeadDim,
            d_rc + static_cast<size_t>(pos) * pairs,
            d_rs + static_cast<size_t>(pos) * pairs);
        {
            // k_rot = kv_latent[kv_lora .. kv_lora+rope): rotate in place.
            __global__ void noop() {}
            (void)noop;
            kx_rope_kernel_helper(s.d_kv_latent + frozen::kKvLoraRank, pairs,
                                  d_rc + static_cast<size_t>(pos) * pairs,
                                  d_rs + static_cast<size_t>(pos) * pairs, stream);
        }

        xing_write_cache_kernel<<<frozen::kHeads, kThreads, 0, stream>>>(
            s.d_kv_b, s.d_kv_latent + frozen::kKvLoraRank, s.d_k, s.d_v, pos,
            frozen::kHeads, frozen::kQkNopeHeadDim, rope, frozen::kVHeadDim);
        xing_attn_kernel<<<frozen::kHeads, kThreads, (pos + 1) * sizeof(float), stream>>>(
            s.d_q, s.d_k, s.d_v, pos, frozen::kHeads, frozen::kQkHeadDim,
            frozen::kVHeadDim, frozen::kAttnScale, s.d_scores, s.d_attn,
            s.capacity);
        Ck(ToHalf(s.d_attn, s.d_half, frozen::kOSize, stream));
        // The mix needs o_proj's output in F32 next to the four streams; stage it
        // through the collapsed buffer, which the next mHC re-reads anyway.
        {
            __half *o_half = s.d_half_big;
            Ck(Gv(ly.o_proj, s.d_half, o_half, stream));
            Ck(ToF32(o_half, s.d_kv_b, frozen::kOSize, stream));
        }
        Ck(LaunchXingMix(s.d_hidden, s.d_kv_b, s.d_post, s.d_comb, hc, H,
                         s.d_hidden, stream));

        // ── FFN block ──────────────────────────────────────────────────────
        Ck(LaunchXingHyperConnection(
            s.d_hidden, ly.ffn_hc.hc_fn, ly.ffn_hc.hc_base, ly.ffn_hc.hc_scale,
            hc, H, frozen::kRmsEps, frozen::kHcSinkhornIters, frozen::kHcClampMin,
            frozen::kHcClampMax, frozen::kHcPostGain, s.d_collapsed, s.d_post,
            s.d_comb, stream));
        xing_rmsnorm_kernel<<<1, kThreads, kThreads * sizeof(float), stream>>>(
            s.d_collapsed, ly.post_attn_ln, s.d_norm, H, frozen::kRmsEps);
        Ck(ToHalf(s.d_norm, s.d_half, H, stream));

        if (!ly.is_moe)
        {
            Ck(Gv(ly.dense_gate, s.d_half, s.d_half_big, stream));
            Ck(Gv(ly.dense_up, s.d_half, s.d_half + frozen::kDenseIntermediate,
                  stream));
            constexpr int N = frozen::kDenseIntermediate;
            lsxcommon::kernels::SwigluKernel<__half, lsxcommon::kernels::ExpFlavor::kGlibc,
                                            false>
                <<<(N + 255) / 256, 256, 0, stream>>>(s.d_half_big, s.d_half + N,
                                                      s.d_half_big, N, 0.0f);
            Ck(Gv(ly.dense_down, s.d_half_big, s.d_half, stream));
            Ck(ToF32(s.d_half, s.d_kv_b, H, stream));
        }
        else
        {
            // Router: BF16 weight + FP32 bias (never quantized).
            Ck(lsxcommon::gemv::RunLxGemmVec(
                stream, ly.router, s.d_norm, 1, pw.num_experts, H,
                lsxcommon::gemv::LxDtype::kBF16, lsxcommon::gemv::LxDtype::kF32,
                1.0f, s.d_scores, lsxcommon::gemv::LxDtype::kF32, 0.0f));
            Ck(LaunchXingRouter(ly.router, ly.e_score_bias, s.d_norm, pw.num_experts,
                                H, frozen::kTopK, frozen::kNormTopkProb,
                                frozen::kRoutedScale, s.d_gate, s.d_expert_ids,
                                stream));
            xing_gather_expert_ptrs_kernel<<<1, frozen::kTopK, 0, stream>>>(
                s.d_expert_ids, frozen::kTopK, pw.expert_packed_ptrs[l],
                pw.expert_scale_ptrs[l], pw.expert_gs[l], s.d_epacked, s.d_escales,
                s.d_egs);
            // gate/up per expert, swiglu, down per expert, weighted accumulate.
            const int MI = frozen::kMoeIntermediate;
            cudaMemsetAsync(s.d_attn, 0, static_cast<size_t>(H) * 4, stream);
            Ck(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVExpertBatch(
                stream, s.d_half, s.d_epacked, s.d_escales, s.d_egs, frozen::kTopK,
                0, s.d_half_big, H, MI));
            Ck(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVExpertBatch(
                stream, s.d_half, s.d_epacked + frozen::kTopK,
                s.d_escales + frozen::kTopK, s.d_egs, frozen::kTopK, 0,
                s.d_half_big + static_cast<size_t>(MI) * frozen::kTopK, H, MI));
            lsxcommon::kernels::SwigluKernel<__half, lsxcommon::kernels::ExpFlavor::kGlibc,
                                            false>
                <<<static_cast<int>((static_cast<int64_t>(MI) * frozen::kTopK + 255) / 256),
                   256, 0, stream>>>(
                    s.d_half_big, s.d_half_big + static_cast<size_t>(MI) * frozen::kTopK,
                    s.d_half_big, static_cast<int64_t>(MI) * frozen::kTopK, 0.0f);
            Ck(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVExpertBatch(
                stream, s.d_half_big, s.d_epacked + frozen::kTopK,
                s.d_escales + frozen::kTopK, s.d_egs, frozen::kTopK,
                MI, s.d_half_big, MI, H));
            xing_scale_acc_kernel<<<dim3((H + 255) / 256, frozen::kTopK), 256, 0, stream>>>(
                s.d_half_big, s.d_gate, H, H, s.d_attn);
            // Shared expert on top of the routed sum.
            Ck(Gv(ly.shared_gate, s.d_half, s.d_half_big, stream));
            Ck(Gv(ly.shared_up, s.d_half, s.d_half + MI, stream));
            lsxcommon::kernels::SwigluKernel<__half, lsxcommon::kernels::ExpFlavor::kGlibc,
                                            false>
                <<<(MI + 255) / 256, 256, 0, stream>>>(s.d_half_big, s.d_half + MI,
                                                      s.d_half_big, MI, 0.0f);
            Ck(Gv(ly.shared_down, s.d_half_big, s.d_half, stream));
            Ck(ToF32(s.d_half, s.d_norm, H, stream));
            Ck(lsxcommon::kernels::AddOutKernel<float><<<(H + 255) / 256, 256, 0, stream>>>(
                s.d_attn, s.d_norm, s.d_attn, H));
        }
        Ck(LaunchXingMix(s.d_hidden, ly.is_moe ? s.d_attn : s.d_kv_b, s.d_post,
                         s.d_comb, hc, H, s.d_hidden, stream));
    }

    // Mean over the four streams -> final norm -> lm_head.
    xing_mean_streams_kernel<<<(H + 255) / 256, 256, 0, stream>>>(
        s.d_hidden, hc, H, s.d_norm);
    xing_rmsnorm_kernel<<<1, kThreads, kThreads * sizeof(float), stream>>>(
        s.d_norm, pw.final_norm, s.d_norm, H, frozen::kRmsEps);
    Ck(ToHalf(s.d_norm, s.d_half, H, stream));
    Ck(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(
        stream, s.d_half, pw.head.packed, pw.head.scales, pw.head.global_scale,
        s.d_half_big, H, frozen::kVocab));
    Ck(ToF32(s.d_half_big, s.d_logits, frozen::kVocab, stream));
    s.length = pos + 1;
    return cudaGetLastError();
}

void XingReadLogits(const XingPinnedWeights &pw, const XingSession &s,
                    float *host_logits, cudaStream_t stream)
{
    cudaMemcpyAsync(host_logits, s.d_logits,
                    static_cast<size_t>(frozen::kVocab) * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    (void)pw;
}

}  // namespace lsxcommon::xing