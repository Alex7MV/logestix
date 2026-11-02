// Mistral-Large-4 MLA attention module (Model Ingestion Protocol stage 5).
// See the header for the absorbed-decode contract and the cache layout.

#include "lsxcommon/mistral_moe_large4_inference_attn.h"

#include "mistral_moe_large4_constants.h"
#include "lsxcommon/lsx_gemv.h"
#include "lsxcommon/lsx_inference_nvfp4.h"

#include <algorithm>
#include <cstring>

namespace lsxcommon {
namespace mistral_large4 {
namespace {

namespace frozen = lsxcommon::mistral_large4::frozen;

cudaError_t Alloc(void **p, size_t n)
{
    return cudaMalloc(p, n == 0 ? 1 : n);
}

// One block per head. Dense path iterates [0, n); masked path iterates the
// ascending index list. Deterministic (block-ordered accumulation).
__global__ void mla_attend_kernel(const float *__restrict__ q_abs,
                                  const float *__restrict__ q,
                                  int qk_head, int kn, int rope_dim,
                                  const __nv_bfloat16 *__restrict__ latent,
                                  const __nv_bfloat16 *__restrict__ kpe,
                                  const int32_t *__restrict__ idx, int n,
                                  int kv_lora, float scale,
                                  float *__restrict__ scores,
                                  float *__restrict__ ctx)
{
    const int h = blockIdx.x;
    const float *qa = q_abs + h * kv_lora;
    const float *qr = q + h * qk_head + kn;
    float *sc = scores + h * n;

    // 1) scores over the selected keys.
    for (int j = threadIdx.x; j < n; j += blockDim.x)
    {
        const int t = idx ? idx[j] : j;
        const __nv_bfloat16 *lat = latent + static_cast<int64_t>(t) * kv_lora;
        float dot = 0.0f;
        for (int c = 0; c < kv_lora; ++c) dot += qa[c] * __bfloat162float(lat[c]);
        if (rope_dim > 0 && kpe)
        {
            const __nv_bfloat16 *kp = kpe + static_cast<int64_t>(t) * rope_dim;
            for (int c = 0; c < rope_dim; ++c)
                dot += qr[c] * __bfloat162float(kp[c]);
        }
        sc[j] = dot * scale;
    }
    __syncthreads();

    // 2) block max.
    __shared__ float s_mx[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < n; j += blockDim.x) mx = fmaxf(mx, sc[j]);
    for (int off = 16; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, off));
    if (lane == 0) s_mx[warp] = mx;
    __syncthreads();
    __shared__ float s_max;
    if (threadIdx.x == 0)
    {
        float t = -INFINITY;
        const int warps = (blockDim.x + 31) / 32;
        for (int i = 0; i < warps; ++i) t = fmaxf(t, s_mx[i]);
        s_max = t;
    }
    __syncthreads();
    const float bmax = s_max;

    // 3) exp + block sum.
    float sum = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x)
    {
        const float e = expf(sc[j] - bmax);
        sc[j] = e;
        sum += e;
    }
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    __shared__ float s_sum[32];
    if (lane == 0) s_sum[warp] = sum;
    __syncthreads();
    __shared__ float s_tot;
    if (threadIdx.x == 0)
    {
        float t = 0.0f;
        const int warps = (blockDim.x + 31) / 32;
        for (int i = 0; i < warps; ++i) t += s_sum[i];
        s_tot = t;
    }
    __syncthreads();
    const float inv = 1.0f / s_tot;

    // 4) ctx[c] = sum_j p_j * latent[t_j][c] (threads over the latent lane).
    float *c_out = ctx + h * kv_lora;
    for (int c = threadIdx.x; c < kv_lora; c += blockDim.x)
    {
        float acc = 0.0f;
        for (int j = 0; j < n; ++j)
        {
            const int t = idx ? idx[j] : j;
            acc += sc[j] * inv *
                   __bfloat162float(latent[static_cast<int64_t>(t) * kv_lora + c]);
        }
        c_out[c] = acc;
    }
}

} // namespace

void FreeLarge4AttnState(Large4AttnState &s)
{
    for (auto *p : s.latent)
        if (p) cudaFree(p);
    for (auto *p : s.k_pe)
        if (p) cudaFree(p);
    for (auto *p : s.indexer_k)
        if (p) cudaFree(p);
    s.latent.clear();
    s.k_pe.clear();
    s.indexer_k.clear();
    for (void **p : {reinterpret_cast<void **>(&s.d_topk),
                     reinterpret_cast<void **>(&s.d_topk_count),
                     reinterpret_cast<void **>(&s.d_q),
                     reinterpret_cast<void **>(&s.d_q_abs),
                     reinterpret_cast<void **>(&s.d_ctx),
                     reinterpret_cast<void **>(&s.d_scores),
                     reinterpret_cast<void **>(&s.d_kv),
                     reinterpret_cast<void **>(&s.d_attn),
                     reinterpret_cast<void **>(&s.d_q16),
                     reinterpret_cast<void **>(&s.d_x16),
                     reinterpret_cast<void **>(&s.d_attn16),
                     reinterpret_cast<void **>(&s.d_out16),
                     reinterpret_cast<void **>(&s.d_index_q),
                     reinterpret_cast<void **>(&s.d_index_w),
                     reinterpret_cast<void **>(&s.d_index_scores),
                     reinterpret_cast<void **>(&s.d_index_q16),
                     reinterpret_cast<void **>(&s.d_index_x16),
                     reinterpret_cast<void **>(&s.d_index_norm),
                     reinterpret_cast<void **>(&s.d_index_norm32),
                     reinterpret_cast<void **>(&s.d_index_kvec),
                     reinterpret_cast<void **>(&s.d_index_k32),
                     reinterpret_cast<void **>(&s.d_norm),
                     reinterpret_cast<void **>(&s.d_proj),
                     reinterpret_cast<void **>(&s.d_cos),
                     reinterpret_cast<void **>(&s.d_sin)})
    {
        if (*p) cudaFree(*p);
        *p = nullptr;
    }
    s.capacity = 0;
    s.length = 0;
}

void Large4AttnState::Free() { FreeLarge4AttnState(*this); }

cudaError_t AllocateLarge4AttnState(Large4AttnState &s, const Large4Dims &dims,
                                    const std::vector<Large4LayerWeights> &layers,
                                    int capacity, const std::vector<float> &cos_t,
                                    const std::vector<float> &sin_t,
                                    cudaStream_t stream)
{
    if (capacity <= 0) return cudaErrorInvalidValue;
    s.capacity = capacity;
    s.length = 0;
    const size_t L = static_cast<size_t>(dims.layers);
    s.latent.assign(L, nullptr);
    s.k_pe.assign(L, nullptr);
    s.indexer_k.assign(L, nullptr);

    for (int l = 0; l < dims.layers; ++l)
    {
        const auto &lw = layers[static_cast<size_t>(l)];
        if (lw.owns_kv)
        {
            cudaError_t e = Alloc(reinterpret_cast<void **>(&s.latent[l]),
                                  static_cast<size_t>(capacity) * dims.kv_lora *
                                      sizeof(__nv_bfloat16));
            if (e != cudaSuccess) return e;
            if (lw.uses_rope)
            {
                e = Alloc(reinterpret_cast<void **>(&s.k_pe[l]),
                          static_cast<size_t>(capacity) * dims.qk_rope *
                              sizeof(__nv_bfloat16));
                if (e != cudaSuccess) return e;
            }
        }
        if (lw.full_indexer)
        {
            const cudaError_t e =
                Alloc(reinterpret_cast<void **>(&s.indexer_k[l]),
                      static_cast<size_t>(capacity) * dims.index_head_dim *
                          sizeof(__nv_bfloat16));
            if (e != cudaSuccess) return e;
        }
    }

    const auto alloc = [](void **p, size_t bytes) {
        return cudaMalloc(p, bytes == 0 ? 1 : bytes);
    };
    cudaError_t e = cudaSuccess;
    e = e == cudaSuccess ? alloc(reinterpret_cast<void **>(&s.d_topk),
                                 static_cast<size_t>(dims.index_topk) * sizeof(int32_t))
                         : e;
    e = e == cudaSuccess ? alloc(reinterpret_cast<void **>(&s.d_topk_count), sizeof(int)) : e;
    e = e == cudaSuccess ? alloc(reinterpret_cast<void **>(&s.d_q),
                                 static_cast<size_t>(dims.heads) * dims.qk_head * 4) : e;
    e = e == cudaSuccess ? alloc(reinterpret_cast<void **>(&s.d_q_abs),
                                 static_cast<size_t>(dims.heads) * dims.kv_lora * 4) : e;
    e = e == cudaSuccess ? alloc(reinterpret_cast<void **>(&s.d_ctx),
                                 static_cast<size_t>(dims.heads) * dims.kv_lora * 4) : e;
    e = e == cudaSuccess ? alloc(reinterpret_cast<void **>(&s.d_scores),
                                 static_cast<size_t>(dims.heads) * capacity * 4) : e;
    e = e == cudaSuccess ? alloc(reinterpret_cast<void **>(&s.d_kv),
                                 (static_cast<size_t>(dims.kv_lora) + dims.qk_rope) * 4) : e;
    e = e == cudaSuccess ? alloc(reinterpret_cast<void **>(&s.d_attn),
                                 static_cast<size_t>(dims.heads) * dims.v_head * 4) : e;
    e = e == cudaSuccess ? alloc(reinterpret_cast<void **>(&s.d_q16),
                                 static_cast<size_t>(dims.heads) * dims.qk_head * 2) : e;
    e = e == cudaSuccess ? alloc(reinterpret_cast<void **>(&s.d_x16),
                                 static_cast<size_t>(dims.hidden) * 2) : e;
    e = e == cudaSuccess ? alloc(reinterpret_cast<void **>(&s.d_attn16),
                                 static_cast<size_t>(dims.heads) * dims.v_head * 2) : e;
    e = e == cudaSuccess ? alloc(reinterpret_cast<void **>(&s.d_out16),
                                 static_cast<size_t>(dims.hidden) * 2) : e;
    e = e == cudaSuccess ? alloc(reinterpret_cast<void **>(&s.d_index_q),
                                 static_cast<size_t>(dims.index_heads) * dims.index_head_dim * 4) : e;
    e = e == cudaSuccess ? alloc(reinterpret_cast<void **>(&s.d_index_w),
                                 static_cast<size_t>(dims.index_heads) * 4) : e;
    e = e == cudaSuccess ? alloc(reinterpret_cast<void **>(&s.d_index_scores),
                                 static_cast<size_t>(capacity) * 4) : e;
    e = e == cudaSuccess ? alloc(reinterpret_cast<void **>(&s.d_index_q16),
                                 static_cast<size_t>(dims.index_heads) * dims.index_head_dim * 2) : e;
    e = e == cudaSuccess ? alloc(reinterpret_cast<void **>(&s.d_index_x16),
                                 static_cast<size_t>(dims.hidden) * 2) : e;
    e = e == cudaSuccess ? alloc(reinterpret_cast<void **>(&s.d_index_norm),
                                 static_cast<size_t>(dims.hidden) * 2) : e;
    e = e == cudaSuccess ? alloc(reinterpret_cast<void **>(&s.d_index_norm32),
                                 static_cast<size_t>(dims.hidden) * 4) : e;
    e = e == cudaSuccess ? alloc(reinterpret_cast<void **>(&s.d_index_kvec),
                                 static_cast<size_t>(dims.index_head_dim) * 2) : e;
    e = e == cudaSuccess ? alloc(reinterpret_cast<void **>(&s.d_index_k32),
                                 static_cast<size_t>(dims.index_head_dim) * 4) : e;
    e = e == cudaSuccess ? alloc(reinterpret_cast<void **>(&s.d_norm),
                                 static_cast<size_t>(dims.hidden) * 4) : e;
    e = e == cudaSuccess ? alloc(reinterpret_cast<void **>(&s.d_proj),
                                 static_cast<size_t>(dims.hidden) * 4) : e;
    if (e != cudaSuccess) return e;
    return UploadLarge4RopeTables(stream, cos_t, sin_t, &s.d_cos, &s.d_sin);
}

cudaError_t Large4AttentionStep(Large4AttnState &s, const Large4Dims &dims,
                                const Large4LayerWeights &lw, int layer,
                                const __nv_bfloat16 *normed,
                                const float *normed32, int pos, float *out,
                                cudaStream_t stream)
{
    const int H = dims.hidden;
    const int heads = dims.heads;
    const int kn = lw.uses_rope ? dims.qk_nope : dims.qk_head;
    const int rope = lw.uses_rope ? dims.qk_rope : 0;
    const int kvl = dims.kv_lora;

    if (s.length <= pos + 1) s.length = pos + 1;

    // ── DSA: this layer's mask, or the previous full layer's ───────────────
    if (lw.full_indexer)
    {
        // q = q_input_norm(x) @ wq_b
        cudaError_t e = LaunchLarge4LayerNorm(
            normed, lw.idx_q_norm, lw.idx_q_norm_bias, H, frozen::kRmsEps,
            s.d_index_norm32, stream);
        if (e != cudaSuccess) return e;
        e = LaunchLarge4F32ToF16(s.d_index_norm32, s.d_index_x16, H, stream);
        if (e != cudaSuccess) return e;
        e = lsx_nvfp4::LaunchNvfp4Gemm(lw.indexer_wq_b, s.d_index_x16, 1,
                                       s.d_index_q16, stream);
        if (e != cudaSuccess) return e;
        e = LaunchLarge4F16ToF32(s.d_index_q16,
                                 s.d_index_q, dims.index_heads * dims.index_head_dim,
                                 stream);
        if (e != cudaSuccess) return e;
        e = LaunchLarge4SsmaxScale(s.d_index_q, dims.index_heads,
                                   dims.index_head_dim, lw.idx_ssmax_s,
                                   lw.idx_ssmax_b, pos, stream);
        if (e != cudaSuccess) return e;
        if (rope > 0)
        {
            e = LaunchLarge4RopeInterleaved(s.d_index_q, dims.index_heads,
                                            dims.index_head_dim, 0, rope,
                                            s.d_cos, s.d_sin, pos, stream);
            if (e != cudaSuccess) return e;
        }

        // k = k_norm(wk @ x), then RoPE on the first `rope` lanes.
        cudaError_t ke = gemv::RunLxGemmVec(
            stream, lw.indexer_wk.data, normed, 1, dims.index_head_dim, H,
            gemv::LxDtype::kBF16, gemv::LxDtype::kBF16, 1.0f, s.d_index_k32,
            gemv::LxDtype::kF32, 0.0f);
        if (ke != cudaSuccess) return ke;
        // LayerNorm over the F32 vector: reuse the BF16 Layernorm path by
        // converting through the index_norm buffers (normed32 is scratch we
        // no longer need).
        {
            const cudaError_t ce = LaunchLarge4F32ToBf16(
                s.d_index_k32, s.d_index_norm, dims.index_head_dim, stream);
            if (ce != cudaSuccess) return ce;
            // k_norm has weight+bias; output must be F32 for the rope step.
            // Reuse d_index_k32 in place through a dedicated call below.
        }
        cudaError_t ne = LaunchLarge4LayerNorm(
            s.d_index_norm, lw.idx_k_norm, lw.idx_k_norm_bias,
            dims.index_head_dim, frozen::kRmsEps, s.d_index_k32, stream);
        if (ne != cudaSuccess) return ne;
        if (rope > 0)
        {
            ne = LaunchLarge4RopeInterleavedVec(s.d_index_k32, rope, s.d_cos,
                                                s.d_sin, pos, stream);
            if (ne != cudaSuccess) return ne;
        }
        ne = LaunchLarge4F32ToBf16(s.d_index_k32, s.d_index_kvec,
                                   dims.index_head_dim, stream);
        if (ne != cudaSuccess) return ne;
        if (s.indexer_k[static_cast<size_t>(layer)] == nullptr)
            return cudaErrorInvalidValue;
        cudaError_t ce = cudaMemcpyAsync(
            s.indexer_k[static_cast<size_t>(layer)] +
                static_cast<int64_t>(pos) * dims.index_head_dim,
            s.d_index_kvec, static_cast<size_t>(dims.index_head_dim) * 2,
            cudaMemcpyDeviceToDevice, stream);
        if (ce != cudaSuccess) return ce;

        // weights = weights_proj(x) * index_heads^-0.5
        ce = gemv::RunLxGemmVec(
            stream, lw.indexer_weights_proj.data, normed, 1, dims.index_heads,
            H, gemv::LxDtype::kBF16, gemv::LxDtype::kBF16,
            frozen::kIndexWeightScale, s.d_index_w, gemv::LxDtype::kF32, 0.0f);
        if (ce != cudaSuccess) return ce;
        ce = LaunchLarge4IndexerScores(
            s.d_index_q, s.indexer_k[static_cast<size_t>(layer)], s.d_index_w,
            dims.index_heads, dims.index_head_dim, frozen::kIndexScoreScale,
            pos + 1, s.d_index_scores, stream);
        if (ce != cudaSuccess) return ce;
        ce = LaunchLarge4TopkSelect(s.d_index_scores, pos + 1, dims.index_topk,
                                    s.d_topk, s.d_topk_count, stream);
        if (ce != cudaSuccess) return ce;
    }

    // ── q ─────────────────────────────────────────────────────────────────
    cudaError_t e = LaunchLarge4Bf16ToF16(normed, s.d_x16, H, stream);
    if (e != cudaSuccess) return e;
    e = lsx_nvfp4::LaunchNvfp4Gemm(lw.wq, s.d_x16, 1, s.d_q16, stream);
    if (e != cudaSuccess) return e;
    e = LaunchLarge4F16ToF32(s.d_q16, s.d_q, heads * dims.qk_head, stream);
    if (e != cudaSuccess) return e;
    e = LaunchLarge4SsmaxScale(s.d_q, heads, dims.qk_head, lw.ssmax_s,
                               lw.ssmax_b, pos, stream);
    if (e != cudaSuccess) return e;
    if (rope > 0)
    {
        e = LaunchLarge4RopeInterleaved(s.d_q, heads, dims.qk_head, kn, rope,
                                        s.d_cos, s.d_sin, pos, stream);
        if (e != cudaSuccess) return e;
    }

    // ── KV latent (own or shared) ─────────────────────────────────────────
    const int src = lw.kv_source;
    if (lw.owns_kv)
    {
        const int kvw = kvl + rope;
        e = gemv::RunLxGemmVec(stream, lw.wkv_a.data, normed, 1, kvw, H,
                               gemv::LxDtype::kBF16, gemv::LxDtype::kBF16,
                               1.0f, s.d_kv, gemv::LxDtype::kF32, 0.0f);
        if (e != cudaSuccess) return e;
        // latent = rmsnorm(kv_a[0:kv_lora]) -> BF16 cache.
        e = LaunchLarge4RmsNorm(s.d_kv, lw.kv_a_norm, kvl,
                                frozen::kRmsEps,
                                s.latent[static_cast<size_t>(layer)] +
                                    static_cast<int64_t>(pos) * kvl,
                                stream);
        if (e != cudaSuccess) return e;
        if (rope > 0)
        {
            e = LaunchLarge4RopeInterleavedVec(s.d_kv + kvl, rope, s.d_cos,
                                               s.d_sin, pos, stream);
            if (e != cudaSuccess) return e;
            e = LaunchLarge4F32ToBf16(
                s.d_kv + kvl,
                s.k_pe[static_cast<size_t>(layer)] +
                    static_cast<int64_t>(pos) * rope,
                rope, stream);
            if (e != cudaSuccess) return e;
        }
    }

    // ── absorbed score/softmax/value accumulation ─────────────────────────
    e = LaunchLarge4AbsorbQuery(lw.wkv_b.data, s.d_q, heads, dims.qk_head, kn,
                                dims.v_head, kvl, s.d_q_abs, stream);
    if (e != cudaSuccess) return e;

    int n_sel = pos + 1;
    const int32_t *mask = nullptr;
    if (n_sel > dims.index_topk)
    {
        // The selector always returns exactly `index_topk` indices when
        // n > k (all strictly-greater entries plus the tie fill), so the
        // count need not be read back on the host.
        n_sel = dims.index_topk;
        mask = s.d_topk;
    }

    const auto *latent = s.latent[static_cast<size_t>(src)];
    const auto *kpe = s.k_pe[static_cast<size_t>(src)];
    if (latent == nullptr) return cudaErrorInvalidValue;
    mla_attend_kernel<<<heads, 256, 0, stream>>>(
        s.d_q_abs, s.d_q, dims.qk_head, kn, rope, latent, kpe, mask, n_sel,
        kvl, frozen::kAttnScale, s.d_scores, s.d_ctx);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;

    e = LaunchLarge4ValueProject(lw.wkv_b.data, s.d_ctx, heads, kn,
                                 dims.v_head, kvl, s.d_attn, stream);
    if (e != cudaSuccess) return e;
    e = LaunchLarge4F32ToF16(s.d_attn, s.d_attn16, heads * dims.v_head, stream);
    if (e != cudaSuccess) return e;
    e = lsx_nvfp4::LaunchNvfp4Gemm(lw.wo, s.d_attn16, 1, s.d_out16, stream);
    if (e != cudaSuccess) return e;
    return LaunchLarge4F16ToF32(s.d_out16, out, H, stream);
}

} // namespace mistral_large4
} // namespace lsxcommon
