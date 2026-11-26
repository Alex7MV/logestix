// GigaChat3.5 MLA decode (Plan 2b). One-token full-attention block: MLA
// latent projections, interleaved YaRN RoPE, scaled F32 softmax over the
// per-layer KV cache, sigmoid gating and o_proj. All activations are F32; the
// NVFP4 GEMV boundary converts to/from FP16. Mirrors modeling_gigachat3_5.py
// GigaChat35Attention.forward.
#include "lsxcommon/gigachat_moe_inference_attn.h"
#include "lsx_kernels_common.cuh"

#include "gigachat_moe_device_math.h"
#include "gigachat_moe_kernels.h"

#include "lsxcommon/lsx_gemv.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace lsxcommon::gigachat {
namespace {

constexpr int kAlign = 256;

inline int64_t AlignUp(int64_t v)
{
    return (v + kAlign - 1) / kAlign * kAlign;
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

// q . k_row over `qk` floats, ASCENDING i with a PLAIN `+=`.
//
// Ascending i with a plain `+=` keeps the per-element order identical to the scalar
// form; that part is guaranteed.
//
// Bit-identity itself is EMPIRICAL, not flag-guaranteed - see the FMA policy note in
// libs/lsxcommon/CMakeLists.txt. attn_scores_kernel is thread-per-position, so a thread
// walked a whole qk-wide row with one row in flight: the latency-bound shape that
// Xing's float4 dot fixed (+6.5%), qwen took (+4.8%) and gemma took (+3.0%). float4 per
// iteration replaces four 4-byte loads with one 16-byte load.
//
// Needs qk % 4 == 0, which also makes the row 16-B aligned (a row offset is a multiple
// of qk floats). Scalar otherwise.
__device__ __forceinline__ float GigachatDotRow(const float *__restrict__ qt,
                                                 const float *__restrict__ kt, int qk)
{
    float acc = 0.0f;
    if ((qk & 3) == 0)
    {
        const float4 *q4 = reinterpret_cast<const float4 *>(qt);
        const float4 *k4 = reinterpret_cast<const float4 *>(kt);
        for (int j = 0; j < (qk >> 2); ++j)
        {
            const float4 b = k4[j];
            const float4 a = q4[j];
            acc += a.x * b.x;
            acc += a.y * b.y;
            acc += a.z * b.z;
            acc += a.w * b.w;
        }
        return acc;
    }
    acc = GigachatDotRow(qt, kt, qk);
    return acc;
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

// DSA selection mask: every position in [0, pos] that is NOT in the ascending
// selection gets -inf, so the softmax attends to the selected keys only.
// Membership is a binary search over the ascending index list.
__global__ void dsa_mask_scores_kernel(float *__restrict__ scores,
                                       const int32_t *__restrict__ sel,
                                       int n_sel, int pos, int heads,
                                       int capacity)
{
    const int h = blockIdx.x;
    for (int t = blockIdx.y * blockDim.x + threadIdx.x; t <= pos;
         t += blockDim.x * gridDim.y)
    {
        int lo = 0, hi = n_sel - 1;
        bool found = false;
        while (lo <= hi)
        {
            const int mid = lo + (hi - lo) / 2;
            const int v = sel[mid];
            if (v == t)
            {
                found = true;
                break;
            }
            if (v < t) lo = mid + 1;
            else hi = mid - 1;
        }
        if (!found) scores[static_cast<size_t>(h) * capacity + t] = -INFINITY;
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
    lsxcommon::kernels::F32ToF16Kernel<<<(n + 255) / 256, 256, 0, s>>>(x, y, n);
    return cudaGetLastError();
}

inline cudaError_t LaunchF16ToF32(const __half *x, float *y, int n,
                                  cudaStream_t s)
{
    if (n <= 0) return cudaSuccess;
    lsxcommon::kernels::F16ToF32Kernel<<<(n + 255) / 256, 256, 0, s>>>(x, y, n);
    return cudaGetLastError();
}

inline cudaError_t LaunchF32ToBf16(const float *x, __nv_bfloat16 *y, int n,
                                   cudaStream_t stream)
{
    if (n <= 0) return cudaSuccess;
    lsxcommon::kernels::F32ToBf16Kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, y, n);
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
    need(static_cast<int64_t>(d.hidden) * 2);            // x_bf
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
    x_bf = carve(static_cast<int64_t>(d.hidden) * 2);
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
    x_bf = nullptr;
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
                                 cudaStream_t stream,
                                 const GcAttnVariant &variant)
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
    GC_TRY(RunGigaChatGatedNorm(q_a_f, 1, d.q_lora, lw.q_a_norm.rank,
                                lw.q_a_norm.d_weight, lw.q_a_norm.d_gate_up,
                                lw.q_a_norm.d_gate_down, norm_eps,
                                variant.norm_w_offset, q_a_n, stream,
                                variant.norm_gate_from_scaled));
    GC_TRY(LaunchScale(q_a_n, q_a_s, d.q_lora, alpha_q, stream));

    // DSA: score the cached keys for this query and select the top-k before the
    // softmax. Uses the unscaled q latent (the indexer applies alpha_q itself)
    // and the BF16 normed hidden, which is what the checkpoint's wk /
    // weights_proj consume.
    const int32_t *sel = variant.dsa_sel;
    int n_sel = variant.dsa_n_sel;
    if (variant.dsa && variant.dsa->run_indexer && variant.dsa->weights &&
        variant.dsa->scratch && variant.dsa->k_cache && variant.dsa->sel &&
        variant.dsa->count)
    {
        GcDsaHook &hook = *variant.dsa;
        GC_TRY(LaunchF32ToBf16(x, static_cast<__nv_bfloat16 *>(sc.x_bf),
                               d.hidden, stream));
        GcDsaIndexerDims idims = hook.dims;
        idims.latent_scale = alpha_q;
        // The indexer selects table row `pos` itself (its documented contract),
        // so it gets the BASE tables: handing it the already-advanced `cos_t`
        // would rotate by 2*pos. That stayed invisible below index_topk, where
        // the selection is the identity, and corrupted the scores above it.
        GC_TRY(RunGigaChatDsaIndexerStep(
            q_a_n, static_cast<const uint8_t *>(sc.x_bf), *hook.weights, idims,
            rope_cos, rope_sin, pos, hook.capacity, hook.k_cache,
            *hook.scratch, hook.sel, hook.count, stream));
        sel = hook.sel;
        n_sel = std::min(idims.topk, pos + 1);
        hook.n_sel = n_sel;
    }

    GC_TRY(LaunchGemm(lw.q_b, q_a_s, q_h, stream));
    GC_TRY(LaunchF16ToF32(q_h, q_f, d.heads * qk, stream));

    // Compressed KV path.
    GC_TRY(LaunchGemm(lw.kv_a, x_h, kv_h, stream));
    GC_TRY(LaunchF16ToF32(kv_h, kv_f, kvw, stream));
    GC_TRY(RunGigaChatGatedNorm(kv_f, 1, d.kv_lora, lw.kv_a_norm.rank,
                                lw.kv_a_norm.d_weight, lw.kv_a_norm.d_gate_up,
                                lw.kv_a_norm.d_gate_down, norm_eps,
                                variant.norm_w_offset, kpass_n, stream,
                                variant.norm_gate_from_scaled));
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
    // DSA: the indexer already dropped the keys this query may not attend to.
    // The selector returns exactly min(topk, pos+1) ascending indices, so the
    // caller's `dsa_n_sel` is authoritative and nothing is read back.
    if (sel && n_sel > 0 && n_sel < pos + 1)
    {
        const int threads = 128;
        const int gy = std::max(1, std::min(32, (pos + threads) / threads));
        dsa_mask_scores_kernel<<<dim3(d.heads, gy), threads, 0, stream>>>(
            static_cast<float *>(sc.scores), sel, n_sel, pos, d.heads,
            capacity);
        const cudaError_t me = cudaGetLastError();
        if (me != cudaSuccess) return me;
    }

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

// ── GatedDeltaNet decode ────────────────────────────────────────────────────

namespace {

// Reorders the interleaved per-k-head qkvz projection into the [q|k|v] conv
// channel order and extracts z, matching fix_query_key_value_ordering.
__global__ void gdn_prepare_kernel(const float *__restrict__ qkvz,
                                   float *__restrict__ mixed,
                                   float *__restrict__ z, int hk, int hv,
                                   int rep, int key_dim)
{
    const int g = blockIdx.x;
    const int vper = hv * rep;
    const int space = 2 * hk + 2 * vper;
    const float *base = qkvz + static_cast<size_t>(g) * space;
    for (int i = threadIdx.x; i < hk; i += blockDim.x)
    {
        mixed[g * hk + i] = base[i];
        mixed[key_dim + g * hk + i] = base[hk + i];
    }
    for (int i = threadIdx.x; i < vper; i += blockDim.x)
    {
        mixed[2 * key_dim + g * vper + i] = base[2 * hk + i];
        z[g * vper + i] = base[2 * hk + vper + i];
    }
}

__global__ void gdn_gate_kernel(const float *__restrict__ ba,
                                const float *__restrict__ A_log,
                                const float *__restrict__ dt_bias,
                                float *__restrict__ beta, float *__restrict__ g,
                                int rep, int v_heads)
{
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= v_heads) return;
    const int kh = h / rep;
    const float b = ba[kh * 2 * rep + (h % rep)];
    const float a = ba[kh * 2 * rep + rep + (h % rep)];
    beta[h] = 1.0f / (1.0f + expf(-b));
    const float t = a + dt_bias[h];
    const float sp = t > 20.0f ? t : log1pf(expf(t));
    g[h] = -expf(A_log[h]) * sp;
}

// Depthwise causal conv (kernel K) over one new token + the K-1 state taps.
__global__ void gdn_conv_kernel(const float *__restrict__ mixed,
                                const float *__restrict__ w,
                                float *__restrict__ state,
                                float *__restrict__ conv_out, int conv_dim,
                                int K)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= conv_dim) return;
    const float *wd = w + static_cast<size_t>(d) * K;
    float *sd = state + static_cast<size_t>(d) * (K - 1);
    float acc = wd[K - 1] * mixed[d];
    for (int j = 0; j < K - 1; ++j) acc += wd[j] * sd[j];
    for (int j = 0; j < K - 2; ++j) sd[j] = sd[j + 1];
    sd[K - 2] = mixed[d];
    conv_out[d] = acc / (1.0f + expf(-acc));
}

// Repeat query/key across the value-head ratio and L2-normalize (q also scaled
// by 1/sqrt(hk)); copies the value slice.
__global__ void gdn_l2_kernel(const float *__restrict__ conv_out,
                              float *__restrict__ q, float *__restrict__ k,
                              float *__restrict__ v, int key_dim, int hk,
                              int hv, int rep, float eps, float scale)
{
    const int h = blockIdx.x;
    const int kh = h / rep;
    __shared__ float invq, invk;
    if (threadIdx.x == 0)
    {
        float s = 0.0f;
        for (int i = 0; i < hk; ++i)
        {
            const float t = conv_out[static_cast<size_t>(kh) * hk + i];
            s += t * t;
        }
        invq = rsqrtf(s + eps);
        s = 0.0f;
        for (int i = 0; i < hk; ++i)
        {
            const float t =
                conv_out[static_cast<size_t>(key_dim) + kh * hk + i];
            s += t * t;
        }
        invk = rsqrtf(s + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < hk; i += blockDim.x)
    {
        q[static_cast<size_t>(h) * hk + i] =
            conv_out[static_cast<size_t>(kh) * hk + i] * invq * scale;
        k[static_cast<size_t>(h) * hk + i] =
            conv_out[static_cast<size_t>(key_dim) + kh * hk + i] * invk;
    }
    for (int j = threadIdx.x; j < hv; j += blockDim.x)
        v[static_cast<size_t>(h) * hv + j] =
            conv_out[2 * static_cast<size_t>(key_dim) +
                     static_cast<size_t>(h) * hv + j];
}

// One recurrent delta-rule step for every value head (state [h, hk, hv]).
__global__ void gdn_recurrent_kernel(const float *__restrict__ q,
                                     const float *__restrict__ k,
                                     const float *__restrict__ v,
                                     const float *__restrict__ beta,
                                     const float *__restrict__ g,
                                     float *__restrict__ state,
                                     float *__restrict__ o, int hk, int hv)
{
    const int h = blockIdx.x;
    const int j = threadIdx.x;
    if (j >= hv) return;
    const size_t sb = static_cast<size_t>(h) * hk * hv;
    const float decay = expf(g[h]);
    for (int i = 0; i < hk; ++i)
        state[sb + static_cast<size_t>(i) * hv + j] *= decay;
    float kvmem = 0.0f;
    for (int i = 0; i < hk; ++i)
        kvmem += state[sb + static_cast<size_t>(i) * hv + j] *
                 k[static_cast<size_t>(h) * hk + i];
    const float delta = (v[static_cast<size_t>(h) * hv + j] - kvmem) * beta[h];
    for (int i = 0; i < hk; ++i)
        state[sb + static_cast<size_t>(i) * hv + j] +=
            k[static_cast<size_t>(h) * hk + i] * delta;
    float acc = 0.0f;
    for (int i = 0; i < hk; ++i)
        acc += state[sb + static_cast<size_t>(i) * hv + j] *
               q[static_cast<size_t>(h) * hk + i];
    o[static_cast<size_t>(h) * hv + j] = acc;
}

} // namespace

bool GcGdnScratch::Alloc(const GcGdnDims &d)
{
    Free();
    const int64_t key_dim = static_cast<int64_t>(d.k_heads) * d.hk;
    const int64_t value_dim = static_cast<int64_t>(d.v_heads) * d.hv;
    const int64_t conv_dim = 2 * key_dim + value_dim;
    const int64_t qkvz_dim = 2 * key_dim + 2 * value_dim;
    const int64_t ba_dim = 2 * static_cast<int64_t>(d.v_heads);
    const int64_t qdim = static_cast<int64_t>(d.v_heads) * d.hk;

    int64_t off = 0;
    auto carve = [&](int64_t bytes) -> void * {
        off = AlignUp(off);
        void *p = static_cast<uint8_t *>(d_base) + off;
        off += bytes;
        return p;
    };
    int64_t total = 0;
    auto need = [&](int64_t bytes) { total = AlignUp(total) + bytes; };
    need(static_cast<int64_t>(d.hidden) * 2);
    need(static_cast<int64_t>(d.hidden) * 2);
    need(qkvz_dim * 2);
    need(qkvz_dim * 4);
    need(ba_dim * 4);
    need(conv_dim * 4);
    need(value_dim * 4);
    need(static_cast<int64_t>(d.v_heads) * 4);
    need(static_cast<int64_t>(d.v_heads) * 4);
    need(conv_dim * 4);
    need(qdim * 4);
    need(qdim * 4);
    need(value_dim * 4);
    need(value_dim * 4);
    need(value_dim * 4);
    need(value_dim * 2);
    need(static_cast<int64_t>(d.hidden) * 2);
    if (total <= 0) return false;
    void *base = nullptr;
    if (cudaMalloc(&base, static_cast<size_t>(total)) != cudaSuccess)
        return false;
    d_base = base;
    bytes = total;
    x_h = carve(static_cast<int64_t>(d.hidden) * 2);
    x_bf16 = carve(static_cast<int64_t>(d.hidden) * 2);
    qkvz_h = carve(qkvz_dim * 2);
    qkvz_f = carve(qkvz_dim * 4);
    ba_f = carve(ba_dim * 4);
    mixed = carve(conv_dim * 4);
    z = carve(value_dim * 4);
    beta = carve(static_cast<int64_t>(d.v_heads) * 4);
    g = carve(static_cast<int64_t>(d.v_heads) * 4);
    conv_out = carve(conv_dim * 4);
    q = carve(qdim * 4);
    k = carve(qdim * 4);
    v = carve(value_dim * 4);
    o = carve(value_dim * 4);
    onorm = carve(value_dim * 4);
    onorm_h = carve(value_dim * 2);
    out_h = carve(static_cast<int64_t>(d.hidden) * 2);
    return true;
}

void GcGdnScratch::Free()
{
    if (d_base)
    {
        cudaFree(d_base);
        d_base = nullptr;
    }
    bytes = 0;
    x_h = x_bf16 = qkvz_h = qkvz_f = ba_f = mixed = z = beta = g = nullptr;
    conv_out = q = k = v = o = onorm = onorm_h = out_h = nullptr;
}

cudaError_t RunGigaChatGdnDecode(const float *x, const GcLayerWeights &lw,
                                 const GcGdnDims &d, float norm_eps,
                                 float o_norm_scale, float *conv_state,
                                 float *rec_state, GcGdnScratch &sc, float *out,
                                 cudaStream_t stream, float w_offset)
{
    const int key_dim = d.k_heads * d.hk;
    const int value_dim = d.v_heads * d.hv;
    const int conv_dim = 2 * key_dim + value_dim;
    const int qkvz_dim = 2 * key_dim + 2 * value_dim;
    const int ba_dim = 2 * d.v_heads;
    const int qdim = d.v_heads * d.hk;
    const int rep = d.v_heads / d.k_heads;
    if (d.k_heads <= 0 || rep <= 0 || d.hk <= 0 || d.hv <= 0 || d.conv_k < 2)
        return cudaErrorInvalidValue;
    if (!sc.d_base || lw.gdn_in_qkvz.rows != qkvz_dim ||
        lw.gdn_in_qkvz.cols != d.hidden || lw.gdn_ba_rows != ba_dim ||
        lw.gdn_out_proj.cols != value_dim)
        return cudaErrorInvalidValue;

    __half *x_h = static_cast<__half *>(sc.x_h);
    __nv_bfloat16 *x_bf16 = static_cast<__nv_bfloat16 *>(sc.x_bf16);
    __half *qkvz_h = static_cast<__half *>(sc.qkvz_h);
    float *qkvz_f = static_cast<float *>(sc.qkvz_f);
    float *ba_f = static_cast<float *>(sc.ba_f);
    float *mixed = static_cast<float *>(sc.mixed);
    float *z = static_cast<float *>(sc.z);
    float *beta = static_cast<float *>(sc.beta);
    float *g = static_cast<float *>(sc.g);
    float *conv_out = static_cast<float *>(sc.conv_out);
    float *q = static_cast<float *>(sc.q);
    float *k = static_cast<float *>(sc.k);
    float *v = static_cast<float *>(sc.v);
    float *o = static_cast<float *>(sc.o);
    float *onorm = static_cast<float *>(sc.onorm);
    __half *onorm_h = static_cast<__half *>(sc.onorm_h);
    __half *out_h = static_cast<__half *>(sc.out_h);
    const float *conv_w = reinterpret_cast<const float *>(lw.d_conv1d);
    const float *A_log = reinterpret_cast<const float *>(lw.d_A_log);
    const float *dt_bias = reinterpret_cast<const float *>(lw.d_dt_bias);

#define GC_TRY(expr)                      \
    do {                                  \
        const cudaError_t _e = (expr);    \
        if (_e != cudaSuccess) return _e; \
    } while (false)

    GC_TRY(LaunchF32ToF16(x, x_h, d.hidden, stream));
    GC_TRY(LaunchF32ToBf16(x, x_bf16, d.hidden, stream));
    GC_TRY(LaunchGemm(lw.gdn_in_qkvz, x_h, qkvz_h, stream));
    GC_TRY(LaunchF16ToF32(qkvz_h, qkvz_f, qkvz_dim, stream));
    GC_TRY(gemv::RunLxGemmVec(stream, lw.d_gdn_in_ba, x_bf16, 1, ba_dim,
                              d.hidden, gemv::LxDtype::kBF16,
                              gemv::LxDtype::kBF16, 1.0f, ba_f,
                              gemv::LxDtype::kF32, 0.0f));
    gdn_prepare_kernel<<<d.k_heads, 256, 0, stream>>>(qkvz_f, mixed, z, d.hk,
                                                      d.hv, rep, key_dim);
    GC_TRY(cudaGetLastError());
    gdn_gate_kernel<<<(d.v_heads + 127) / 128, 128, 0, stream>>>(
        ba_f, A_log, dt_bias, beta, g, rep, d.v_heads);
    GC_TRY(cudaGetLastError());
    gdn_conv_kernel<<<(conv_dim + 255) / 256, 256, 0, stream>>>(
        mixed, conv_w, conv_state, conv_out, conv_dim, d.conv_k);
    GC_TRY(cudaGetLastError());
    gdn_l2_kernel<<<d.v_heads, 128, 0, stream>>>(conv_out, q, k, v, key_dim,
                                                 d.hk, d.hv, rep, 1e-6f,
                                                 rsqrtf(static_cast<float>(d.hk)));
    GC_TRY(cudaGetLastError());
    gdn_recurrent_kernel<<<d.v_heads, 128, 0, stream>>>(q, k, v, beta, g,
                                                        rec_state, o, d.hk,
                                                        d.hv);
    GC_TRY(cudaGetLastError());
    GC_TRY(RunGdnOutputNormOffset(o, z, lw.d_gdn_norm, d.v_heads, d.hv,
                                  norm_eps, o_norm_scale, w_offset, onorm,
                                  stream));
    GC_TRY(LaunchF32ToF16(onorm, onorm_h, value_dim, stream));
    GC_TRY(LaunchGemm(lw.gdn_out_proj, onorm_h, out_h, stream));
    GC_TRY(LaunchF16ToF32(out_h, out, d.hidden, stream));
#undef GC_TRY
    return cudaSuccess;
}

} // namespace lsxcommon::gigachat
