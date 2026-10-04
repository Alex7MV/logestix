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

#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/xing_moe_kernels.h"
#include "lsx_kernels_act.cuh"
#include "lsx_kernels_math.cuh"
#include "lsx_kernels_norm.cuh"
#include "lsxcommon/lsx_gemv.h"
#include "lsxcommon/log.h"
#include "lsxcommon/shared_experts.h"

#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace lsxcommon::xing {
namespace {

// Stage probe (LSX_TEST_XING_LAYER_DEBUG): RMS of one device buffer, printed to
// stderr with the layer/token that produced it, so a device run can be lined up
// against modeling_xing4_0.py's per-module hook dump. Same fprintf diagnostic
// convention as qwen_moe's LSX_TEST_MOE_STAGE_DEBUG.
void XingStageDump(const char *stage, int layer, int token_id, const float *d,
                   int64_t n, cudaStream_t stream)
{
    const char *dir = std::getenv("LSX_TEST_XING_DUMP_DIR");
    if (dir == nullptr) return;
    std::string path;
    static std::vector<float> host;
    if (static_cast<int64_t>(host.size()) < n) host.resize(static_cast<size_t>(n));
    if (cudaMemcpyAsync(host.data(), d, static_cast<size_t>(n) * sizeof(float),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        return;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return;
    absl::StrAppendFormat(&path, "%s/%s_L%d_tok%d.bin", dir, stage, layer, token_id);
    FILE *f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) return;
    std::fwrite(host.data(), sizeof(float), static_cast<size_t>(n), f);
    std::fclose(f);
}

unsigned XingHashF32(const float *host, size_t n)
{
    unsigned h = 2166136261u;
    for (size_t i = 0; i < n; ++i)
    {
        unsigned bits;
        std::memcpy(&bits, &host[i], sizeof(bits));
        h = (h ^ bits) * 16777619u;
    }
    return h;
}

void XingStageRms(const char *stage, int layer, int token_id, const float *d,
                  int64_t n, cudaStream_t stream)
{
    if (!std::getenv("LSX_TEST_XING_LAYER_DEBUG")) return;
    static std::vector<float> host;
    if (static_cast<int64_t>(host.size()) < n) host.resize(static_cast<size_t>(n));
    if (cudaMemcpyAsync(host.data(), d, static_cast<size_t>(n) * sizeof(float),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        return;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return;
    double ss = 0.0;
    int64_t bad = 0;
    for (int64_t j = 0; j < n; ++j)
    {
        const double v = host[static_cast<size_t>(j)];
        if (!std::isfinite(v)) ++bad;
        else ss += v * v;
    }
    std::fprintf(stderr, "[xing-stage] %-22s L=%d tok=%d rms=%.6g nonfinite=%ld\n",
                 stage, layer, token_id,
                 std::sqrt(ss / static_cast<double>(n)), static_cast<long>(bad));
}

namespace frozen = lsxcommon::xing::frozen;

constexpr int kThreads = 256;
constexpr int kMaxHcMultDebug = 8;

// Propagate-on-error: as a plain function call `XING_CK(x)` evaluated and DISCARDED
// the code, so a failed launch silently produced an all-NaN logit row instead of
// stopping the step. Every call site is a statement, which makes this macro safe.
#define XING_CK(expr)                             \
    do {                                          \
        const cudaError_t xing_ck_e_ = (expr);    \
        if (xing_ck_e_ != cudaSuccess) {           \
            xing_fail_line = static_cast<int>(__LINE__); \
            lsxcommon::log::error(absl::StrFormat(   \
                "xing stage failed at line %d: %s", __LINE__,              \
                cudaGetErrorString(xing_ck_e_)));    \
            return xing_ck_e_;                    \
        }                                         \
    } while (0)

// Line of the failing stage, for diagnostics when a step returns an error.
static thread_local int xing_fail_line = 0;

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

// Interleaved YaRN RoPE: the 64-dim rope subspace is laid out as adjacent pairs
// (x0,x1), (x2,x3), ... each rotated by ONE frequency (rope_interleave=True).
// Upstream notes this is bit-identical to the half-split rotate_half form on
// both q and k, hence score-preserving - so either layout gives the same logits.
//
// Two entry points because the two tensors are laid out differently: q's rope
// subspace is strided per head inside the [heads, nope+rope] buffer, while k_rot
// is a standalone [rope] row copied into the cache.
// `nope` is the rope subspace's offset INSIDE one head, so the element address
// is h*qk + nope + 2*j (gigachat's rope_qk_kernel convention). An earlier
// version took a single shared `q_offset` argument, and the decode step passed
// the constant `heads * kQkNopeHeadDim`: all 32 heads then rotated the same 64
// floats while head 0's rope subspace stayed unrotated. Only a multi-position
// prefill can see it - with one cached token the softmax is trivial.
__global__ void xing_rope_q_kernel(float *__restrict__ q, int pairs, int qk,
                                   int nope, const float *__restrict__ cos_t,
                                   const float *__restrict__ sin_t)
{
    const int h = blockIdx.x;
    for (int j = threadIdx.x; j < pairs; j += blockDim.x)
    {
        const float c = cos_t[j];
        const float s = sin_t[j];
        const int qn = h * qk + nope + 2 * j;
        const float a = q[qn];
        const float b = q[qn + 1];
        q[qn] = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
        q[qn + 1] = __fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c));
    }
}

// k_rot is a single [rope] row (MQA-style shared across heads upstream).
__global__ void xing_rope_k_kernel(float *__restrict__ k_rot, int pairs,
                                   const float *__restrict__ cos_t,
                                   const float *__restrict__ sin_t)
{
    for (int j = threadIdx.x; j < pairs; j += blockDim.x)
    {
        const float c = cos_t[j];
        const float s = sin_t[j];
        const float a = k_rot[2 * j];
        const float b = k_rot[2 * j + 1];
        k_rot[2 * j] = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
        k_rot[2 * j + 1] = __fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c));
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
    // EVERY head writes the shared rotated k_rot. Upstream expands k_rot across
    // heads (`k_rot.expand(*k_pass.shape[:-1], -1)`), so writing it from the h==0
    // block alone left heads 1..N-1 reading uninitialized rope dims - garbage
    // that differs run to run and only shows up as non-determinism in the first
    // step of a session (a lone token attends to its own row, where head 0 is
    // the only one that was written).
    for (int i = threadIdx.x; i < rope; i += blockDim.x) kb[nope + i] = krot[i];
}


// Host-side staging for the four selected experts: the model-neutral
// expert-batch GEMV takes its packed/scale pointer arrays BY VALUE (it copies
// them into a by-value struct for the kernel), so the routed ids must reach the
// host. One D2H of 4 ints per layer, exactly the shape nemotron and gigachat
// use — the ids, not the 4 KiB activation.
// One routed expert's three NVFP4 matrices, resolved from the layer's host
// pointer table ([gate | down | up], see XingPinnedWeights).
struct XingRoutedViews {
    std::vector<const uint8_t *> packed_gate, packed_up, packed_down;
    std::vector<const uint8_t *> scales_gate, scales_up, scales_down;
    std::vector<float> gs_gate, gs_up, gs_down;
};

cudaError_t ResolveXingRouted(const XingPinnedWeights &pw, int layer,
                              const int32_t *d_ids, int top_k, cudaStream_t stream,
                              XingRoutedViews &out, XingSession *s)
{
    const size_t slot = static_cast<size_t>(layer);
    if (slot >= pw.expert_packed_ptrs.size() || top_k <= 0) return cudaErrorInvalidValue;
    const size_t n_view = static_cast<size_t>(top_k);
    out.packed_gate.assign(n_view, nullptr);
    out.packed_up.assign(n_view, nullptr);
    out.packed_down.assign(n_view, nullptr);
    out.scales_gate.assign(n_view, nullptr);
    out.scales_up.assign(n_view, nullptr);
    out.scales_down.assign(n_view, nullptr);
    out.gs_gate.assign(n_view, 1.0f);
    out.gs_up.assign(n_view, 1.0f);
    out.gs_down.assign(n_view, 1.0f);
    std::vector<int> ids(static_cast<size_t>(top_k), -1);
    const cudaError_t cp = cudaMemcpyAsync(
        ids.data(), d_ids, sizeof(int) * static_cast<size_t>(top_k),
        cudaMemcpyDeviceToHost, stream);
    if (cp != cudaSuccess) return cp;
    const cudaError_t sy = cudaStreamSynchronize(stream);
    if (sy != cudaSuccess) return sy;
    const int E = pw.num_experts;
    if (s) s->last_routed_ids.assign(ids.begin(), ids.end());
    const auto &pp = pw.expert_packed_ptrs[slot];
    const auto &sp = pw.expert_scale_ptrs[slot];
    const auto &gv = pw.expert_gs[slot];
    for (int i = 0; i < top_k; ++i)
    {
        const int e = ids[static_cast<size_t>(i)];
        if (e < 0 || e >= E)
        {
            log::error(absl::StrFormat(
                "xing routed id %d out of range (experts=%d, layer=%d)", e, E, layer));
            return cudaErrorInvalidValue;
        }
        const size_t k = static_cast<size_t>(i);
        const size_t n = static_cast<size_t>(E);
        const size_t ge = static_cast<size_t>(e);
        out.packed_gate[k] = pp[ge];         // w1: gate rows
        out.scales_gate[k] = sp[ge];
        out.gs_gate[k] = gv[ge];
        out.packed_down[k] = pp[n + ge];     // w2: down rows
        out.scales_down[k] = sp[n + ge];
        out.gs_down[k] = gv[n + ge];
        out.packed_up[k] = pp[2 * n + ge];   // w3: up rows
        out.scales_up[k] = sp[2 * n + ge];
        out.gs_up[k] = gv[2 * n + ge];
        if (!out.packed_gate[k] || !out.scales_gate[k] ||
            !out.packed_up[k] || !out.scales_up[k] ||
            !out.packed_down[k] || !out.scales_down[k])
        {
            log::error(absl::StrCat("xing resolved null expert view, layer ", layer,
                                    " e=", e, " tables=", pp.size(), "/",
                                    sp.size(), "/", gv.size()));
            return cudaErrorInvalidValue;
        }
    }
    return cudaSuccess;
}

// Broadcast one embedding row into all hc_mult streams.
__global__ void xing_embed_gather_kernel(const __half *__restrict__ table,
                                         int id, float *__restrict__ streams,
                                         int hc_mult, int hidden)
{
    for (int j = threadIdx.x; j < hidden; j += blockDim.x)
    {
        const float v = __half2float(table[static_cast<int64_t>(id) * hidden + j]);
        for (int d = 0; d < hc_mult; ++d) streams[d * hidden + j] = v;
    }
}

// Weighted accumulate of the routed outputs into the token hidden state.
// acc[i] += sum_e w[e] * y[e][i], with ONE thread per output element looping
// over the experts. Mapping the experts onto blockIdx.y instead made every
// block a read-modify-write of the SAME acc[i]: the top_k contributions raced
// (dropped updates, and a different winner run to run - the decode step was
// non-deterministic in its FIRST step, before any cache slot was warm).
__global__ void xing_scale_acc_kernel(const __half *__restrict__ y,
                                      const float *__restrict__ w, int top_k,
                                      int n, float *__restrict__ acc)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = acc[i];
    for (int e = 0; e < top_k; ++e)
    {
        const int64_t k = static_cast<int64_t>(e) * n + i;
        a = __fadd_rn(a, __fmul_rn(__half2float(y[k]), w[e]));
    }
    acc[i] = a;
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

XingDims XingDimsFromWeights(const XingPinnedWeights &pw)
{
    XingDims d;
    d.hidden = frozen::kHidden;
    d.hc_mult = frozen::kHcMult;
    d.vocab = frozen::kVocab;
    d.layers = pw.num_layers;
    d.experts = pw.num_experts;
    d.top_k = frozen::kTopK;
    d.first_moe_layer = pw.first_moe_layer;
    d.q_lora = frozen::kQLoraRank;
    d.kv_lora = frozen::kKvLoraRank;
    d.heads = frozen::kHeads;
    d.qk_nope = frozen::kQkNopeHeadDim;
    d.qk_rope = frozen::kQkRopeHeadDim;
    d.qk_head = frozen::kQkHeadDim;
    d.v_head = frozen::kVHeadDim;
    d.dense_inter = frozen::kDenseIntermediate;
    d.moe_inter = frozen::kMoeIntermediate;
    d.sinkhorn_iters = frozen::kHcSinkhornIters;
    d.attn_scale = frozen::kAttnScale;
    d.routed_scale = frozen::kRoutedScale;
    d.norm_topk_prob = frozen::kNormTopkProb;
    d.hc_eps = frozen::kRmsEps;
    d.clamp_min = frozen::kHcClampMin;
    d.clamp_max = frozen::kHcClampMax;
    d.post_gain = frozen::kHcPostGain;
    // A synthetic fixture shrinks the geometry; the weight shapes are the only
    // honest source for it, so every field the step reads comes from them.
    if (!pw.layers.empty())
    {
        const auto &l0 = pw.layers.front();
        if (l0.input_ln != nullptr && l0.q_a.rows > 0)
        {
            // The decode fixture writes FULL frozen widths (only the layer and
            // expert COUNTS shrink), so hidden/heads stay at the passport and
            // only the projection widths the tensors actually describe are
            // taken from them.
            d.q_lora = static_cast<int>(l0.q_a.rows);
            if (d.q_lora <= 0) d.q_lora = frozen::kQLoraRank;
        }
        d.hc_mult = static_cast<int>(l0.attn_hc.hc_fn_cols /
                                     std::max(1, d.hidden));
        d.dense_inter = static_cast<int>(l0.dense_gate.rows);
    }
    // The MoE widths live on the first MoE layer, not on layer 0: reading
    // shared_gate.rows off the dense prefix yields 0 and every routed GEMV then
    // gets N=0 (cudaErrorInvalidValue one layer deep, with no clue where).
    for (int l = pw.first_moe_layer; l < pw.num_layers; ++l)
    {
        const auto &lm = pw.layers[static_cast<size_t>(l)];
        if (lm.shared_gate.valid())
        {
            d.moe_inter = static_cast<int>(lm.shared_gate.rows);
            break;
        }
    }
    if (d.dense_inter <= 0 && pw.num_layers > pw.first_moe_layer)
    {
        // MoE-only fixture: the dense width lives in the passport.
        d.dense_inter = frozen::kDenseIntermediate;
    }
    if (pw.embed.rows > 0) d.vocab = static_cast<int>(pw.embed.rows);
    return d;
}

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
    d(s.d_comb); d(s.d_act); d(s.d_norm); d(s.d_q); d(s.d_kv_latent); d(s.d_kv_b);
    d(s.d_attn); d(s.d_scores); d(s.d_logits); d(s.d_gate); d(s.d_expert_ids);
    d(s.d_half); d(s.d_half_up); d(s.d_half_big); d(s.d_expert_out); d(s.d_epacked); d(s.d_escales); d(s.d_egs);
    d(s.d_embed_table); d(s.d_router_act); d(s.d_router_logits);
    s = XingSession{};
}

cudaError_t AllocateXingSession(XingSession &s, const XingDims &dims,
                                int capacity, cudaStream_t stream)
{
    (void)stream;
    FreeXingSession(s);
    const int H = dims.hidden;
    const int hc = dims.hc_mult;
    const int vocab = dims.vocab;
    const int dense_intermediate = dims.dense_inter;
    auto alloc = [](void **p, size_t bytes) {
        return bytes == 0 ? cudaSuccess : cudaMalloc(p, bytes);
    };
    const int kv_layers = std::max(1, dims.layers);
    s.kv_layers = kv_layers;
    s.kv_layer_k = static_cast<int64_t>(capacity) * frozen::kHeads *
                   (frozen::kQkNopeHeadDim + frozen::kQkRopeHeadDim);
    s.kv_layer_v = static_cast<int64_t>(capacity) * frozen::kHeads *
                   frozen::kVHeadDim;
    XING_CK(alloc(reinterpret_cast<void **>(&s.d_k),
             static_cast<size_t>(s.kv_layer_k) * kv_layers * 4));
    XING_CK(alloc(reinterpret_cast<void **>(&s.d_v),
             static_cast<size_t>(s.kv_layer_v) * kv_layers * 4));
    XING_CK(alloc(reinterpret_cast<void **>(&s.d_hidden),
             static_cast<size_t>(hc) * H * 4));
    XING_CK(alloc(reinterpret_cast<void **>(&s.d_collapsed), static_cast<size_t>(H) * 4));
    XING_CK(alloc(reinterpret_cast<void **>(&s.d_post), static_cast<size_t>(hc) * 4));
    XING_CK(alloc(reinterpret_cast<void **>(&s.d_comb), static_cast<size_t>(hc) * hc * 4));
    XING_CK(alloc(reinterpret_cast<void **>(&s.d_act), static_cast<size_t>(H) * 4));
    XING_CK(alloc(reinterpret_cast<void **>(&s.d_norm),
             static_cast<size_t>(std::max({H, frozen::kQLoraRank, frozen::kKvLoraRank})) * 4));
    XING_CK(alloc(reinterpret_cast<void **>(&s.d_q), static_cast<size_t>(dims.heads * dims.qk_head) * 4));
    XING_CK(alloc(reinterpret_cast<void **>(&s.d_kv_latent), static_cast<size_t>(dims.kv_lora + dims.qk_rope) * 4));
    XING_CK(alloc(reinterpret_cast<void **>(&s.d_kv_b), static_cast<size_t>(dims.heads * (dims.qk_nope + dims.v_head)) * 4));
    XING_CK(alloc(reinterpret_cast<void **>(&s.d_attn), static_cast<size_t>(dims.heads * dims.v_head) * 4));
    // One score row PER HEAD: the attention kernel indexes scores[h*capacity+t],
    // so a single capacity-sized row overruns as soon as heads > 1 (found by the
    // full-geometry decode test; the synthetic kernel test used one head).
    XING_CK(alloc(reinterpret_cast<void **>(&s.d_scores),
             static_cast<size_t>(capacity) * dims.heads * 4));
    XING_CK(alloc(reinterpret_cast<void **>(&s.d_logits), static_cast<size_t>(vocab) * 4));
    s.logit_count = vocab;
    XING_CK(alloc(reinterpret_cast<void **>(&s.d_router_act),
                  static_cast<size_t>(H) * sizeof(__nv_bfloat16)));
    XING_CK(alloc(reinterpret_cast<void **>(&s.d_router_logits),
                  static_cast<size_t>(std::max(1, dims.experts)) *
                      sizeof(__nv_bfloat16)));
    XING_CK(alloc(reinterpret_cast<void **>(&s.d_gate), 16 * 4));
    XING_CK(alloc(reinterpret_cast<void **>(&s.d_expert_ids), 16 * 4));
    XING_CK(alloc(reinterpret_cast<void **>(&s.d_half),
             static_cast<size_t>(dense_intermediate) * 2 * sizeof(__half)));
    XING_CK(alloc(reinterpret_cast<void **>(&s.d_half_up),
             static_cast<size_t>(dense_intermediate) * sizeof(__half)));
    XING_CK(alloc(reinterpret_cast<void **>(&s.d_half_big),
             static_cast<size_t>(vocab) * sizeof(__half)));
    // Routed-expert down-projection output, one row per selected expert. It
    // MUST NOT alias the swiglu input: the batched GEMV writes y + e*H while
    // later experts still read x + e*moe_intermediate, and that overlap races
    // with block order (the step was non-deterministic run to run because of it).
    XING_CK(alloc(reinterpret_cast<void **>(&s.d_expert_out),
             static_cast<size_t>(std::max(1, dims.top_k)) *
                 static_cast<size_t>(H) * sizeof(__half)));
    XING_CK(alloc(reinterpret_cast<void **>(&s.d_epacked), 8 * sizeof(void *)));
    XING_CK(alloc(reinterpret_cast<void **>(&s.d_escales), 8 * sizeof(void *)));
    XING_CK(alloc(reinterpret_cast<void **>(&s.d_egs), 8 * sizeof(float)));
    s.capacity = capacity;
    s.length = 0;
    return cudaGetLastError();
}

cudaError_t LaunchXingRopeQ(float *q, int heads, int qk_head_dim, int pairs,
                            const float *cos_t, const float *sin_t,
                            cudaStream_t stream)
{
    xing_rope_q_kernel<<<heads, kThreads, 0, stream>>>(
        q, pairs, qk_head_dim, qk_head_dim - 2 * pairs, cos_t, sin_t);
    return cudaGetLastError();
}

cudaError_t LaunchXingRopeK(float *krot, int pairs, const float *cos_t,
                            const float *sin_t, cudaStream_t stream)
{
    xing_rope_k_kernel<<<1, kThreads, 0, stream>>>(krot, pairs, cos_t, sin_t);
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

cudaError_t UploadXingYarnTables(cudaStream_t stream, int capacity,
                                 const std::vector<float> &cos_t,
                                 const std::vector<float> &sin_t,
                                 float **dst_rc, float **dst_rs)
{
    const size_t n = cos_t.size();
    if (n == 0 || sin_t.size() != n) return cudaErrorInvalidValue;
    XING_CK(cudaMalloc(reinterpret_cast<void **>(dst_rc), n * sizeof(float)));
    XING_CK(cudaMalloc(reinterpret_cast<void **>(dst_rs), n * sizeof(float)));
    XING_CK(cudaMemcpy(*dst_rc, cos_t.data(), n * sizeof(float),
                  cudaMemcpyHostToDevice));
    XING_CK(cudaMemcpy(*dst_rs, sin_t.data(), n * sizeof(float),
                  cudaMemcpyHostToDevice));
    return cudaGetLastError();
}

cudaError_t XingDecodeStep(const XingPinnedWeights &pw, const XingDims &dims,
                           XingSession &s, int token_id, const float *d_rc,
                           const float *d_rs, int pos, cudaStream_t stream,
                           int layers_override)
{
    const int H = dims.hidden;
    const int hc = dims.hc_mult;
    const int rope = dims.qk_rope;
    const int pairs = rope / 2;
    const int heads = dims.heads;
    xing_fail_line = 0;
    if (std::getenv("LSX_TEST_XING_LAYER_DEBUG"))
    {
        XingStageDump("v_start", 0, token_id, s.d_v,
                      static_cast<int64_t>(pos + 1) * frozen::kHeads *
                          frozen::kVHeadDim,
                      stream);
    }
    XingRoutedViews routed;   // reused per layer; resize() keeps the capacity

    // Seed all four streams with the embedding row. The embedding is stored
    // NVFP4, so it is dequantized ONCE into a persistent FP16 table (the
    // qwen_moe pattern) and gathered per step - a 131072x3584 dequant per token
    // would dominate the step time.
    if (s.d_embed_table == nullptr)
    {
        const int64_t elems = static_cast<int64_t>(dims.vocab) * H;
        const cudaError_t ae = cudaMalloc(
            reinterpret_cast<void **>(&s.d_embed_table),
            static_cast<size_t>(elems) * sizeof(__half));
        if (ae != cudaSuccess) return ae;
        const cudaError_t de = lsxcommon::gpu::dequant_shared_nvfp4(
            stream, pw.embed.packed, pw.embed.packed_bytes, pw.embed.scales,
            pw.embed.scale_bytes, pw.embed.global_scale, s.d_embed_table,
            elems);
        if (de != cudaSuccess) return de;
    }
    xing_embed_gather_kernel<<<1, kThreads, 0, stream>>>(s.d_embed_table,
                                                         token_id, s.d_hidden,
                                                         hc, H);
    s.token_id = token_id;
    if (std::getenv("LSX_TEST_XING_LAYER_DEBUG"))
    {
        static std::vector<float> host;
        host.resize(static_cast<size_t>(hc) * H);
        if (cudaMemcpyAsync(host.data(), s.d_hidden, host.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, stream) == cudaSuccess)
        {
            cudaStreamSynchronize(stream);
            double ss = 0.0, amax = 0.0;
            for (const float v : host)
            {
                ss += static_cast<double>(v) * v;
                amax = std::max(amax, std::fabs(static_cast<double>(v)));
            }
            std::fprintf(stderr, "[xing-embed] tok=%d rms=%.6g max=%.6g\n", token_id,
                         std::sqrt(ss / static_cast<double>(host.size())), amax);
        }
    }

    // layers_override lets a test run a PREFIX of the stack (bisecting where a
    // non-finite value first appears); 0 means the whole model.
    const int layer_limit =
        layers_override < 0 ? pw.num_layers
                            : std::min(layers_override, pw.num_layers);
    for (int l = 0; l < layer_limit; ++l)
    {
        const auto &ly = pw.layers[static_cast<size_t>(l)];
        // Per-layer KV slice: upstream keys layer_idx, so sharing one buffer
        // would silently hand every layer the previous layer's token rows.
        float *const kc = s.d_k + static_cast<int64_t>(l) * s.kv_layer_k;
        float *const vc = s.d_v + static_cast<int64_t>(l) * s.kv_layer_v;

        // ── attention block ────────────────────────────────────────────────
        XING_CK(LaunchXingHyperConnection(
            s.d_hidden, ly.attn_hc.hc_fn, ly.attn_hc.hc_base, ly.attn_hc.hc_scale,
            hc, H, frozen::kRmsEps, frozen::kHcSinkhornIters, frozen::kHcClampMin,
            frozen::kHcClampMax, frozen::kHcPostGain, s.d_collapsed, s.d_post,
            s.d_comb, stream));
        XingStageRms("hc.collapsed", l, token_id, s.d_collapsed, H, stream);
        XingStageRms("hc.post", l, token_id, s.d_post, hc, stream);
        XingStageRms("hc.comb", l, token_id, s.d_comb, hc * hc, stream);
        xing_rmsnorm_kernel<<<1, kThreads, kThreads * sizeof(float), stream>>>(
            s.d_collapsed, ly.input_ln, s.d_act, H, frozen::kRmsEps);
        XingStageRms("input_ln", l, token_id, s.d_act, H, stream);

        // q_a down-projects the activation; its inner norm overwrites d_norm, so
        // the activation stays in d_act for kv_a below.
        XING_CK(ToHalf(s.d_act, s.d_half, H, stream));
        XING_CK(Gv(ly.q_a, s.d_half, s.d_half_big, stream));      // [q_lora]
        XING_CK(ToF32(s.d_half_big, s.d_norm, frozen::kQLoraRank, stream));
        xing_rmsnorm_kernel<<<1, kThreads, kThreads * sizeof(float), stream>>>(
            s.d_norm, ly.q_a_ln, s.d_norm, frozen::kQLoraRank, frozen::kRmsEps);
        XING_CK(ToHalf(s.d_norm, s.d_half, frozen::kQLoraRank, stream));
        XING_CK(Gv(ly.q_b, s.d_half, s.d_half_big, stream));      // [heads*qk]
        XING_CK(ToF32(s.d_half_big, s.d_q, frozen::kQSize, stream));
        XingStageRms("q_a", l, token_id, s.d_norm, frozen::kQLoraRank, stream);
        XingStageRms("q_b", l, token_id, s.d_q, frozen::kQSize, stream);

        XING_CK(ToHalf(s.d_act, s.d_half, H, stream));
        XING_CK(Gv(ly.kv_a, s.d_half, s.d_half_big, stream));     // [kv_lora+rope]
        XING_CK(ToF32(s.d_half_big, s.d_kv_latent, frozen::kKvAWidth, stream));
        XingStageRms("kv_a", l, token_id, s.d_kv_latent, frozen::kKvAWidth,
                     stream);
        // kv_a_layernorm over the latent part only; the rope part passes through.
        {
            float *latent = s.d_norm;   // scratch of kv_lora floats
            xing_rmsnorm_kernel<<<1, kThreads, kThreads * sizeof(float), stream>>>(
                s.d_kv_latent, ly.kv_a_ln, latent, frozen::kKvLoraRank,
                frozen::kRmsEps);
            cudaMemcpyAsync(s.d_kv_latent, latent, frozen::kKvLoraRank * 4,
                            cudaMemcpyDeviceToDevice, stream);
        }
        XING_CK(ToHalf(s.d_kv_latent, s.d_half, frozen::kKvLoraRank, stream));
        XING_CK(Gv(ly.kv_b, s.d_half, s.d_half_big, stream));      // [heads*(nope+v)]
        XING_CK(ToF32(s.d_half_big, s.d_kv_b, frozen::kKvBWidth, stream));
        XingStageRms("kv_b", l, token_id, s.d_kv_b, frozen::kKvBWidth, stream);
        XingStageDump("kv_b", l, token_id, s.d_kv_b, frozen::kKvBWidth, stream);
        XingStageDump("q_b", l, token_id, s.d_q, frozen::kQSize, stream);

        // Interleaved RoPE on q's rope subspace and on k_rot.
        const float *rc = d_rc + static_cast<size_t>(pos) * pairs;
        const float *rs = d_rs + static_cast<size_t>(pos) * pairs;
        XING_CK(LaunchXingRopeQ(s.d_q, frozen::kHeads, frozen::kQkHeadDim, pairs,
                               rc, rs, stream));
        XING_CK(LaunchXingRopeK(s.d_kv_latent + frozen::kKvLoraRank, pairs, rc,
                               rs, stream));

        xing_write_cache_kernel<<<frozen::kHeads, kThreads, 0, stream>>>(
            s.d_kv_b, s.d_kv_latent + frozen::kKvLoraRank, kc, vc, pos,
            frozen::kHeads, frozen::kQkNopeHeadDim, rope, frozen::kVHeadDim);
        if (l == 0 && std::getenv("LSX_TEST_XING_LAYER_DEBUG"))
        {
            std::fprintf(stderr, "[xing-kv] tok=%d pos=%d layer=%d v=%p cap=%d\n",
                         token_id, pos, l, static_cast<void *>(vc), s.capacity);
            XingStageDump("v_pre", l, token_id, vc,
                          static_cast<int64_t>(pos + 1) * frozen::kHeads *
                              frozen::kVHeadDim,
                          stream);
        }
        XING_CK(LaunchXingAttention(
            s.d_q, kc, vc, pos, frozen::kHeads, frozen::kQkHeadDim,
            frozen::kVHeadDim, frozen::kAttnScale, s.d_scores, s.d_attn,
            s.capacity, pos + 1, stream));
        XingStageRms("attn_core", l, token_id, s.d_attn, frozen::kOSize, stream);
        if (l == 0)
        {
            XingStageDump("q_rope", l, token_id, s.d_q, frozen::kQSize, stream);
            XingStageDump("kcache", l, token_id, kc,
                          static_cast<int64_t>(pos + 1) * frozen::kHeads *
                              (frozen::kQkNopeHeadDim + frozen::kQkRopeHeadDim),
                          stream);
            XingStageDump("vcache", l, token_id, vc,
                          static_cast<int64_t>(pos + 1) * frozen::kHeads *
                              frozen::kVHeadDim,
                          stream);
        }
        XingStageDump("attn_core", l, token_id, s.d_attn, frozen::kOSize, stream);
        XING_CK(ToHalf(s.d_attn, s.d_half, frozen::kOSize, stream));
        // The mix needs o_proj's output in F32 next to the four streams; stage it
        // through the collapsed buffer, which the next mHC re-reads anyway.
        {
            __half *o_half = s.d_half_big;
            XING_CK(Gv(ly.o_proj, s.d_half, o_half, stream));
            XING_CK(ToF32(o_half, s.d_kv_b, frozen::kOSize, stream));
            XingStageRms("attn_out", l, token_id, s.d_kv_b, frozen::kOSize,
                         stream);
        }
        XING_CK(LaunchXingMix(s.d_hidden, s.d_kv_b, s.d_post, s.d_comb, hc, H,
                         s.d_hidden, stream));

        // ── FFN block ──────────────────────────────────────────────────────
        XING_CK(LaunchXingHyperConnection(
            s.d_hidden, ly.ffn_hc.hc_fn, ly.ffn_hc.hc_base, ly.ffn_hc.hc_scale,
            hc, H, frozen::kRmsEps, frozen::kHcSinkhornIters, frozen::kHcClampMin,
            frozen::kHcClampMax, frozen::kHcPostGain, s.d_collapsed, s.d_post,
            s.d_comb, stream));
        XingStageRms("ffnhc.collapsed", l, token_id, s.d_collapsed, H, stream);
        XingStageRms("ffnhc.comb", l, token_id, s.d_comb, hc * hc, stream);
        xing_rmsnorm_kernel<<<1, kThreads, kThreads * sizeof(float), stream>>>(
            s.d_collapsed, ly.post_attn_ln, s.d_act, H, frozen::kRmsEps);
        XingStageRms("post_attn_ln", l, token_id, s.d_act, H, stream);
        XING_CK(ToHalf(s.d_act, s.d_half, H, stream));

        if (!ly.is_moe)
        {
            XING_CK(Gv(ly.dense_gate, s.d_half, s.d_half_big, stream));
            constexpr int N = frozen::kDenseIntermediate;
            XING_CK(Gv(ly.dense_up, s.d_half, s.d_half_up, stream));
            lsxcommon::kernels::SwigluKernel<__half, lsxcommon::kernels::ExpFlavor::Glibc,
                                            false>
                <<<(N + 255) / 256, 256, 0, stream>>>(s.d_half_big, s.d_half_up,
                                                      s.d_half_big, N, 0.0f);
            XING_CK(Gv(ly.dense_down, s.d_half_big, s.d_half, stream));
            XING_CK(ToF32(s.d_half, s.d_kv_b, H, stream));
        }
        else
        {
            // Router: BF16 weight + FP32 bias (never quantized). The
            // device-independent GEMV dispatches matching dtypes only, so the
            // activation is converted to BF16 first.
            lsxcommon::kernels::F32toTKernel<__nv_bfloat16>
                <<<(H + 255) / 256, 256, 0, stream>>>(s.d_act, s.d_router_act,
                                                      static_cast<int64_t>(H));
            XING_CK(lsxcommon::gemv::RunLxGemmVec(
                stream, ly.router, s.d_router_act, 1, dims.experts, H,
                lsxcommon::gemv::LxDtype::kBF16, lsxcommon::gemv::LxDtype::kBF16,
                1.0f, s.d_router_logits, lsxcommon::gemv::LxDtype::kBF16, 0.0f));
            XING_CK(LaunchXingRouter(ly.router, ly.e_score_bias, s.d_router_act,
                                     dims.experts, H, dims.top_k,
                                     dims.norm_topk_prob, dims.routed_scale,
                                     s.d_gate, s.d_expert_ids, stream));
            XING_CK(ResolveXingRouted(pw, l, s.d_expert_ids, dims.top_k, stream,
                                 routed, &s));
            // gate/up per expert, swiglu, down per expert, weighted accumulate.
            const int MI = dims.moe_inter;
            cudaMemsetAsync(s.d_attn, 0, static_cast<size_t>(H) * 4, stream);
            XING_CK(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVExpertBatch(
                stream, s.d_half, routed.packed_gate.data(),
                routed.scales_gate.data(), routed.gs_gate.data(), dims.top_k, 0,
                s.d_half_big, H, MI));
            // The SECOND branch of the SwiGLU is up_proj (w3). Passing the
            // gate views here - and skipping the down projection entirely -
            // still yields finite logits, because silu(gate)*gate is a
            // perfectly well-formed MLP; it just isn't this model.
            XING_CK(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVExpertBatch(
                stream, s.d_half, routed.packed_up.data(),
                routed.scales_up.data(), routed.gs_up.data(), dims.top_k, 0,
                s.d_half_big + static_cast<size_t>(MI) * dims.top_k, H, MI));
            lsxcommon::kernels::SwigluKernel<__half, lsxcommon::kernels::ExpFlavor::Glibc,
                                            false>
                <<<static_cast<int>((static_cast<int64_t>(MI) * dims.top_k + 255) / 256),
                   256, 0, stream>>>(
                    s.d_half_big, s.d_half_big + static_cast<size_t>(MI) * dims.top_k,
                    s.d_half_big, static_cast<int64_t>(MI) * dims.top_k, 0.0f);
            // down_proj (w2) over each expert's own SwiGLU slice
            // (x_stride = MI), into the per-expert output rows.
            XING_CK(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVExpertBatch(
                stream, s.d_half_big, routed.packed_down.data(),
                routed.scales_down.data(), routed.gs_down.data(), dims.top_k, MI,
                s.d_expert_out, MI, H));
            // Separate output buffer: aliasing the swiglu input here races
            // (writing y + e*H overwrites the inputs of experts e+1..e+3) and
            // made the whole step non-deterministic run to run.
            xing_scale_acc_kernel<<<(H + 255) / 256, 256, 0, stream>>>(
                s.d_expert_out, s.d_gate, dims.top_k, H, s.d_attn);
            // Shared expert on top of the routed sum.
            XING_CK(Gv(ly.shared_gate, s.d_half, s.d_half_big, stream));
            XING_CK(Gv(ly.shared_up, s.d_half, s.d_half_up, stream));
            lsxcommon::kernels::SwigluKernel<__half, lsxcommon::kernels::ExpFlavor::Glibc,
                                            false>
                <<<(MI + 255) / 256, 256, 0, stream>>>(s.d_half_big, s.d_half_up,
                                                      s.d_half_big, MI, 0.0f);
            XING_CK(Gv(ly.shared_down, s.d_half_big, s.d_half, stream));
            XING_CK(ToF32(s.d_half, s.d_norm, H, stream));
            lsxcommon::kernels::AddOutKernel<float><<<(H + 255) / 256, 256, 0, stream>>>(
                s.d_attn, s.d_norm, s.d_attn, H);
        }
        XingStageRms(ly.is_moe ? "moe_out" : "mlp_out", l, token_id,
                     ly.is_moe ? s.d_attn : s.d_kv_b, H, stream);
        XING_CK(LaunchXingMix(s.d_hidden, ly.is_moe ? s.d_attn : s.d_kv_b, s.d_post,
                         s.d_comb, hc, H, s.d_hidden, stream));
        // Layer-by-layer divergence probe: the mean-stream RMS of every layer is
        // printed so a device run can be compared against modeling_xing4_0.py's
        // output_hidden_states. The first layer whose RMS departs from the
        // reference localises the wiring defect (fprintf to stderr follows the
        // qwen_moe LSX_TEST_MOE_STAGE_DEBUG convention - this is a diagnostic,
        // not application logging).
        if (std::getenv("LSX_TEST_XING_LAYER_DEBUG"))
        {
            static std::vector<float> host;
            host.resize(static_cast<size_t>(hc) * H);
            const cudaError_t ce = cudaMemcpyAsync(
                host.data(), s.d_hidden, host.size() * sizeof(float),
                cudaMemcpyDeviceToHost, stream);
            if (ce == cudaSuccess)
            {
                cudaStreamSynchronize(stream);
                double ss = 0.0, amax = 0.0, bad = 0.0;
                for (size_t j = 0; j < host.size(); ++j)
                {
                    const double v = host[j];
                    if (!std::isfinite(v)) ++bad;
                    ss += v * v;
                    amax = std::max(amax, std::fabs(v));
                }
                std::fprintf(stderr,
                             "[xing-layer] tok=%d %d rms=%.6g max=%.6g nonfinite=%g\n",
                             token_id, l, std::sqrt(ss / static_cast<double>(host.size())),
                             amax, bad);
                double per[kMaxHcMultDebug] = {};
                for (int d = 0; d < hc; ++d)
                    for (int j = 0; j < H; ++j)
                        per[d] += static_cast<double>(host[d * H + j]) *
                                  static_cast<double>(host[d * H + j]);
                std::fprintf(stderr, "[xing-layer] %d %d pos=%d hash=%08x\n", l,
                             token_id, pos,
                             XingHashF32(host.data(), host.size()));
                std::fprintf(stderr, "[xing-layer] tok=%d %d streams", token_id, l);
                for (int d = 0; d < hc; ++d)
                    std::fprintf(stderr, " %.6g", std::sqrt(per[d] / H));
                std::fprintf(stderr, "\n");
            }
        }
        s.last_layer = l;
    }

    // Mean over the four streams -> final norm -> lm_head.
    xing_mean_streams_kernel<<<(H + 255) / 256, 256, 0, stream>>>(
        s.d_hidden, hc, H, s.d_norm);
    xing_rmsnorm_kernel<<<1, kThreads, kThreads * sizeof(float), stream>>>(
        s.d_norm, pw.final_norm, s.d_norm, H, frozen::kRmsEps);
    XING_CK(ToHalf(s.d_norm, s.d_half, H, stream));
    XING_CK(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(
        stream, s.d_half, pw.head.packed, pw.head.scales, pw.head.global_scale,
        s.d_half_big, H, dims.vocab));
    XING_CK(ToF32(s.d_half_big, s.d_logits, dims.vocab, stream));
    s.length = pos + 1;
    s.fail_line = 0;
    return cudaGetLastError();
}

__global__ void xing_count_nonfinite_kernel(const float *__restrict__ x, int64_t n,
                                             int *__restrict__ out)
{
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    if (!isfinite(x[i])) atomicAdd(out, 1);
}

cudaError_t XingCopyLayerKV(const XingSession &s, int layer, int pos,
                            float *host_k, float *host_v, cudaStream_t stream)
{
    if (layer < 0 || layer >= s.kv_layers) return cudaErrorInvalidValue;
    const int64_t base_k = static_cast<int64_t>(layer) * s.kv_layer_k +
                           static_cast<int64_t>(pos) * frozen::kHeads *
                               (frozen::kQkNopeHeadDim + frozen::kQkRopeHeadDim);
    const int64_t base_v = static_cast<int64_t>(layer) * s.kv_layer_v +
                           static_cast<int64_t>(pos) * frozen::kHeads *
                               frozen::kVHeadDim;
    const size_t k_row = static_cast<size_t>(frozen::kHeads) * frozen::kQkHeadDim;
    const size_t v_row = static_cast<size_t>(frozen::kHeads) * frozen::kVHeadDim;
    if (host_k != nullptr && pos < s.capacity &&
        cudaMemcpyAsync(host_k, s.d_k + base_k, k_row * sizeof(float),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        return cudaErrorInvalidValue;
    if (host_v != nullptr && pos < s.capacity &&
        cudaMemcpyAsync(host_v, s.d_v + base_v, v_row * sizeof(float),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        return cudaErrorInvalidValue;
    return cudaSuccess;
}

cudaError_t XingCountNonFinite(const float *d, int64_t n, int *out_count,
                               cudaStream_t stream)
{
    // One counter per block, summed on the host: no cross-block atomics on the
    // decode path itself, and this is a diagnostic entry point anyway.
    constexpr int kBlocks = 256;
    int *d_count = nullptr;
    if (cudaMalloc(&d_count, sizeof(int)) != cudaSuccess) return cudaErrorMemoryAllocation;
    cudaMemsetAsync(d_count, 0, sizeof(int), stream);
    xing_count_nonfinite_kernel<<<kBlocks, 256, 0, stream>>>(d, n, d_count);
    const cudaError_t e = cudaMemcpyAsync(out_count, d_count, sizeof(int),
                                          cudaMemcpyDeviceToHost, stream);
    cudaFree(d_count);
    return e;
}

__global__ void xing_max_abs_kernel(const float *__restrict__ x, int64_t n,
                                    float *__restrict__ out)
{
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    atomicMax(reinterpret_cast<int *>(out), __float_as_int(fabsf(x[i])));
}

cudaError_t XingMaxAbs(const float *d, int64_t n, float *out_max,
                       cudaStream_t stream)
{
    float *d_max = nullptr;
    if (cudaMalloc(&d_max, sizeof(float)) != cudaSuccess) return cudaErrorMemoryAllocation;
    const unsigned zero = 0x00000000u;
    cudaMemcpyAsync(d_max, &zero, sizeof(float), cudaMemcpyHostToDevice, stream);
    const int blocks = static_cast<int>(std::min<int64_t>((n + 255) / 256, 65536));
    xing_max_abs_kernel<<<blocks, 256, 0, stream>>>(d, n, d_max);
    const cudaError_t e = cudaMemcpyAsync(out_max, d_max, sizeof(float),
                                          cudaMemcpyDeviceToHost, stream);
    cudaFree(d_max);
    return e;
}

int XingLastFailureLine()
{
    return xing_fail_line;
}

void XingReadLogits(const XingSession &s, float *host_logits,
                    cudaStream_t stream)
{
    // vocab is implicit in the buffer the step allocated; the caller sizes the
    // host array from XingDims.
    cudaMemcpyAsync(host_logits, s.d_logits, s.logit_count * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
}

}  // namespace lsxcommon::xing