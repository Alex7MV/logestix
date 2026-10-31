// Ling-3.1-flash generation driver: token-by-token hybrid KDA/MLA prefill and
// decode over the neutral NVFP4 GEMV, the neutral mmap expert store and the
// device slot LRU.
#include "lsxcommon/ling_moe_inference.h"

#include "ling_moe_constants.h"

#include "lsxcommon/ling_moe_kernels.h"
#include "lsxcommon/log.h"
#include "lsxcommon/lsx_attn_inference.h"
#include "lsxcommon/lsx_gemv.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/model_runtime.h"

#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace lsxcommon::ling {
namespace {

constexpr int kThreads = 256;

inline int Blocks(int64_t n)
{
    return static_cast<int>((n + kThreads - 1) / kThreads);
}

cudaError_t Gemm(cudaStream_t stream, const __half *x,
                 const lsx_nvfp4::LsxNvfp4Weight &w, __half *y, int K, int N)
{
    return lsx_nvfp4::RunLsxNvfp4Gemm(stream, x, w.d_packed, w.d_scales,
                                      w.global_scale, y, 1, K, N, false);
}

cudaError_t GemvHalf(cudaStream_t stream, const __half *w, const __half *x,
                     __half *y, int K, int N)
{
    return gemv::RunLxGemmVec(stream, w, x, 1, N, K, gemv::LxDtype::kF16,
                              gemv::LxDtype::kF16, 1.0f, y,
                              gemv::LxDtype::kF16, 0.0f);
}

__global__ void add_half_kernel(const __half *__restrict__ a,
                                const __half *__restrict__ b,
                                __half *__restrict__ out, int n)
{
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

__global__ void store_mla_kv_kernel(const __half *__restrict__ kb,
                                    const __half *__restrict__ k_rot,
                                    __half *__restrict__ kv,
                                    const KvCacheGeometry geom, int layer_slot,
                                    int64_t pos, int heads, int nope,
                                    int rope, int v_head)
{
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    const int kdim = nope + rope;
    if (d < kdim) {
        float v;
        if (d < nope)
            v = __half2float(kb[h * (nope + v_head) + d]);
        else
            v = __half2float(k_rot[d - nope]);
        kv[geom.Offset(layer_slot, h, pos, d)] = __float2half(v);
    } else if (d < kdim + v_head) {
        const int i = d - kdim;
        kv[geom.Offset(layer_slot, h, pos, d)] = __float2half(
            __half2float(kb[h * (nope + v_head) + nope + i]));
    }
}

bool IsMla(const LingIndex &top, int l)
{
    return std::find(top.mla_layers.begin(), top.mla_layers.end(), l) !=
           top.mla_layers.end();
}

int KdaOrdinal(const LingIndex &top, int l)
{
    int n = 0;
    for (int x : top.kda_layers) {
        if (x == l) return n;
        ++n;
    }
    return -1;
}

int MlaOrdinal(const LingIndex &top, int l)
{
    int n = 0;
    for (int x : top.mla_layers) {
        if (x == l) return n;
        ++n;
    }
    return -1;
}

} // namespace

std::int64_t LingSession::Bytes() const
{
    int64_t b = 0;
    if (d_kv)
        b += static_cast<int64_t>(geom.Elements()) * static_cast<int64_t>(sizeof(__half));
    if (d_kda_state)
        b += static_cast<int64_t>(frozen::kKdaLayerCount) * 64 * 128 * 128 *
             static_cast<int64_t>(sizeof(float));
    if (d_conv_hist)
        b += static_cast<int64_t>(frozen::kKdaLayerCount) *
             (frozen::kConvKernel - 1) * 3 * 8192 * sizeof(__half);
    b += static_cast<int64_t>(gen_tail.capacity()) * 4;
    return b;
}

void LingScratch::Free()
{
    if (arena) cudaFree(arena);
    arena = nullptr;
    arena_bytes = 0;
    if (d_logits) cudaFree(d_logits);
    d_logits = nullptr;
    logits_elems = 0;
    if (d_argmax) cudaFree(d_argmax);
    d_argmax = nullptr;
    if (d_maxv) cudaFree(d_maxv);
    d_maxv = nullptr;
    if (d_slots) cudaFree(d_slots);
    d_slots = nullptr;
    if (h_slots) cudaFreeHost(h_slots);
    h_slots = nullptr;
    if (d_scores) cudaFree(d_scores);
    d_scores = nullptr;
    scores_elems = 0;
    if (d_moe_acc) cudaFree(d_moe_acc);
    d_moe_acc = nullptr;
    if (cache_ready) moe::FreeMoEExpertCache(cache);
    cache_ready = false;
}

void LingSession::Free()
{
    if (d_kv) cudaFree(d_kv);
    if (d_kda_state) cudaFree(d_kda_state);
    if (d_conv_hist) cudaFree(d_conv_hist);
    d_kv = nullptr;
    d_kda_state = nullptr;
    d_conv_hist = nullptr;
    geom = {};
    capacity = 0;
    kv_rows = 0;
    last_tok = -1;
    gen_tail.clear();
}

void LingSession::Reset(const LingIndex &top, cudaStream_t stream)
{
    kv_rows = 0;
    last_tok = -1;
    gen_tail.clear();
    const int64_t kd = static_cast<int64_t>(top.heads) * top.head_dim;
    if (d_kda_state)
        cudaMemsetAsync(
            d_kda_state, 0,
            static_cast<size_t>(top.kda_layers.size()) * top.heads *
                top.head_dim * top.head_dim * sizeof(float),
            stream);
    if (d_conv_hist)
        cudaMemsetAsync(
            d_conv_hist, 0,
            static_cast<size_t>(top.kda_layers.size()) *
                (top.conv_kernel - 1) * 3 * kd * sizeof(__half),
            stream);
}

cudaError_t LingSession::EnsureCapacity(const LingIndex &top, int64_t want,
                                         cudaStream_t stream)
{
    if (want <= capacity && d_kv) return cudaSuccess;
    const int64_t cap = std::max<int64_t>(want, capacity * 2);
    const KvCacheGeometry g = LingKvGeometry(top, cap);
    __half *nk = nullptr;
    if (cudaMalloc(&nk, static_cast<size_t>(g.Elements()) * sizeof(__half)) !=
        cudaSuccess)
        return cudaGetLastError();
    const int64_t kd = static_cast<int64_t>(top.heads) * top.head_dim;
    if (!d_kda_state) {
        const size_t bytes = static_cast<size_t>(top.kda_layers.size()) *
                             top.heads * top.head_dim * top.head_dim *
                             sizeof(float);
        if (cudaMalloc(&d_kda_state, bytes) != cudaSuccess) {
            cudaFree(nk);
            return cudaGetLastError();
        }
        cudaMemsetAsync(d_kda_state, 0, bytes, stream);
    }
    if (!d_conv_hist) {
        const size_t bytes = static_cast<size_t>(top.kda_layers.size()) *
                             (top.conv_kernel - 1) * 3 * kd * sizeof(__half);
        if (cudaMalloc(&d_conv_hist, bytes) != cudaSuccess) {
            cudaFree(nk);
            return cudaGetLastError();
        }
        cudaMemsetAsync(d_conv_hist, 0, bytes, stream);
    }
    if (d_kv && kv_rows > 0) {
        for (int64_t l = 0; l < g.layers; ++l) {
            const size_t bytes = static_cast<size_t>(kv_rows) *
                                 static_cast<size_t>(g.pos_stride) *
                                 sizeof(__half);
            const cudaError_t ce = cudaMemcpyAsync(
                nk + l * g.LayerStride(), d_kv + l * geom.LayerStride(),
                bytes, cudaMemcpyDeviceToDevice, stream);
            if (ce != cudaSuccess) {
                std::fprintf(stderr,
                             "[ling-dbg] kv copy fail l=%lld bytes=%zu src_off=%lld "
                             "dst_off=%lld: %s\n",
                             static_cast<long long>(l), bytes,
                             static_cast<long long>(l * geom.LayerStride()),
                             static_cast<long long>(l * g.LayerStride()),
                             cudaGetErrorString(ce));
                cudaFree(nk);
                return ce;
            }
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            cudaFree(nk);
            return cudaGetLastError();
        }
    }
    if (d_kv) cudaFree(d_kv);
    d_kv = nk;
    geom = g;
    capacity = cap;
    return cudaSuccess;
}

bool LingPrepareExpertCache(LingPinnedWeights &pw, LingScratch &scratch,
                            std::string &error)
{
    if (scratch.cache_ready) return true;
    if (pw.expert_store.records.empty() ||
        pw.expert_store.records[static_cast<size_t>(pw.top.moe_layers.front())]
                               .empty()) {
        error = "expert store geometry missing";
        return false;
    }
    const int l0 = pw.top.moe_layers.front();
    const auto &rec = pw.expert_store.records[static_cast<size_t>(l0)][0];
    const int64_t slot_bytes = rec.w1_bytes + rec.w2_bytes + rec.w3_bytes +
                               rec.s1_bytes + rec.s2_bytes + rec.s3_bytes;
    if (slot_bytes <= 0) {
        error = "expert slot geometry is empty";
        return false;
    }
    size_t free_bytes = 0, total = 0;
    if (cudaMemGetInfo(&free_bytes, &total) != cudaSuccess) {
        error = "cudaMemGetInfo failed";
        return false;
    }
    const int64_t reserve = int64_t{2} << 30;
    int slots = static_cast<int>(
        (static_cast<int64_t>(free_bytes) - reserve) / slot_bytes);
    const int total_experts =
        static_cast<int>(pw.top.moe_layers.size()) * pw.top.num_experts;
    const int lo = std::min(64, total_experts);
    slots = std::clamp(slots, lo, total_experts);
    if (const char *env = std::getenv("LSX_LING_EXPERT_CACHE_SLOTS")) {
        const long v = std::atol(env);
        if (v > 0) slots = static_cast<int>(v);
    }
    if (!moe::AllocateMoEExpertCache(pw.expert_store, slots, scratch.cache,
                                     error))
        return false;
    scratch.cache_ready = true;
    log::info(absl::StrFormat(
        "[ling] expert cache: %d slots x %s = %s", slots,
        lsxcommon::FormatBytesHuman(static_cast<uint64_t>(slot_bytes)),
        lsxcommon::FormatBytesHuman(
            static_cast<uint64_t>(scratch.cache.DeviceBytes()))));
    return true;
}

namespace {

struct StepCtx {
    cudaStream_t stream;
    LingPinnedWeights &pw;
    LingSession &session;
    LingScratch &scratch;
    const LingIndex &top;
    int H, KD, NH, HD, QL, KVL, QKN, QKR, VH, QKD, KVD, CONV, E, K, G,
        TG, I, SI, DI, MLACOUNT, eos;
};

bool Step(StepCtx &c, int32_t token, int64_t pos, __half *d_x, __half *sc,
          float *f32, std::string &error)
{
    const LingIndex &top = c.top;
    cudaStream_t stream = c.stream;
    static const bool kDump = std::getenv("LSX_LING_DUMP") != nullptr;
    if (cudaError_t ee =
            RunLingEmbedToken(stream, c.pw.d_embed, token, d_x, c.H);
        ee != cudaSuccess) {
        error = absl::StrCat("embed token=", token, " err=",
                             cudaGetErrorString(ee));
        return false;
    }
    __half *d_normed = sc;
    __half *d_attn = sc + c.H;
    __half *d_post = sc + 2 * c.H;
    __half *d_q = sc + 3 * c.H;            // KD
    __half *d_k = d_q + c.KD;
    __half *d_v = d_k + c.KD;
    __half *d_qc = d_v + c.KD;
    __half *d_kc = d_qc + c.KD;
    __half *d_vc = d_kc + c.KD;
    __half *d_f = d_vc + c.KD;
    __half *d_g = d_f + c.KD;
    __half *d_kout = d_g + c.KD;
    __half *d_qa = d_kout + c.KD;          // QL
    __half *d_qb = d_qa + c.QL;            // NH*QKD
    __half *d_ca = d_qb + c.NH * c.QKD;    // KVL+QKR
    __half *d_cn = d_ca + c.KVL + c.QKR;   // KVL
    __half *d_kb = d_cn + c.KVL;            // NH*(QKN+VH)
    __half *d_mattn = d_kb + c.NH * (c.QKN + c.VH);  // NH*VH
    __half *d_mgate = d_mattn + c.NH * c.VH;         // NH
    const int maxi = std::max(std::max(c.I, c.SI), c.DI);
    const int yg_elems = std::max(2 * c.I, maxi);
    __half *d_yg = d_mgate + c.NH;
    __half *d_yu = d_yg + yg_elems;
    __half *d_y2 = d_yu + maxi;
    __half *d_shared = d_y2 + c.H;
    float *d_beta = f32;
    float *d_decay = d_beta + c.NH;

    if (kDump && pos == 0) {
        std::vector<__half> hx(static_cast<size_t>(c.H));
        cudaMemcpyAsync(hx.data(), d_x,
                        static_cast<size_t>(c.H) * sizeof(__half),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        std::fprintf(stderr,
                     "[ling-emb] tok=%d x0=%.6f x1=%.6f x2=%.6f x3=%.6f\n",
                     token, __half2float(hx[0]), __half2float(hx[1]),
                     __half2float(hx[2]), __half2float(hx[3]));
    }
    for (int l = 0; l < top.num_layers; ++l) {
        const LingLayerWeights &lw = c.pw.layers[static_cast<size_t>(l)];
        if (RunLingRmsNorm(stream, d_x, lw.d_input_norm, d_normed, 1, c.H,
                           static_cast<float>(top.rms_eps)) != cudaSuccess) {
            error = "input norm";
            return false;
        }
        const bool mla = IsMla(top, l);
        if (!mla) {
            const int ord = KdaOrdinal(top, l);
            cudaError_t e = Gemm(stream, d_normed, lw.kda_q, d_q, c.H, c.KD);
            if (e == cudaSuccess)
                e = Gemm(stream, d_normed, lw.kda_k, d_k, c.H, c.KD);
            if (e == cudaSuccess)
                e = Gemm(stream, d_normed, lw.kda_v, d_v, c.H, c.KD);
            if (e == cudaSuccess)
                e = Gemm(stream, d_normed, lw.kda_f, d_f, c.H, c.KD);
            if (e == cudaSuccess)
                e = Gemm(stream, d_normed, lw.kda_g, d_g, c.H, c.KD);
            if (e == cudaSuccess)
                e = GemvHalf(stream, lw.d_kda_b, d_normed, d_kout, c.H, c.NH);
            if (e != cudaSuccess) {
                error = "kda projections";
                return false;
            }
            __half *hist = c.session.d_conv_hist +
                           static_cast<int64_t>(ord) * (c.CONV - 1) * 3 * c.KD;
            const int64_t plane = static_cast<int64_t>(c.CONV - 1) * c.KD;
            e = RunLingConvDecodeSilu(stream, hist, d_q, lw.d_conv_q, d_qc,
                                      c.KD, c.CONV, true);
            if (e == cudaSuccess)
                e = RunLingConvDecodeSilu(stream, hist + plane, d_k, lw.d_conv_k,
                                          d_kc, c.KD, c.CONV, true);
            if (e == cudaSuccess)
                e = RunLingConvDecodeSilu(stream, hist + 2 * plane, d_v,
                                          lw.d_conv_v, d_vc, c.KD, c.CONV,
                                          true);
            if (e == cudaSuccess)
                e = RunLingL2NormQK(stream, d_qc, d_kc, c.NH, c.HD,
                                    1.0f / std::sqrt(static_cast<float>(c.HD)));
            if (e == cudaSuccess)
                e = RunLingKdaGating(stream, d_kout, d_f, lw.d_a_log,
                                     lw.d_dt_bias, d_beta, d_decay, c.NH,
                                     c.HD, -5.0f);
            if (e == cudaSuccess)
                e = RunLingKdaStep(
                    stream, d_qc, d_kc, d_vc, d_beta, d_decay,
                    c.session.d_kda_state + static_cast<int64_t>(ord) * c.NH *
                                                c.HD * c.HD,
                    d_kout, c.NH, c.HD);
            if (e == cudaSuccess)
                e = RunLingRmsNormGated(stream, d_kout, d_g, lw.d_kda_o_norm,
                                        d_qc, c.NH, c.HD,
                                        static_cast<float>(top.rms_eps));
            if (e == cudaSuccess)
                e = Gemm(stream, d_qc, lw.kda_o, d_attn, c.KD, c.H);
            if (e != cudaSuccess) {
                error = absl::StrCat("kda layer ", l);
                return false;
            }
        } else {
            const int ord = MlaOrdinal(top, l);
            cudaError_t e = Gemm(stream, d_normed, lw.mla_q_a, d_qa, c.H, c.QL);
            if (e == cudaSuccess)
                e = RunLingRmsNorm(stream, d_qa, lw.d_q_a_norm, d_qa, 1, c.QL,
                                   static_cast<float>(top.rms_eps));
            if (e == cudaSuccess)
                e = Gemm(stream, d_qa, lw.mla_q_b, d_qb, c.QL, c.NH * c.QKD);
            if (e == cudaSuccess)
                e = Gemm(stream, d_normed, lw.mla_kv_a, d_ca, c.H,
                         c.KVL + c.QKR);
            if (e == cudaSuccess)
                e = RunLingRmsNorm(stream, d_ca, lw.d_kv_a_norm, d_cn, 1,
                                   c.KVL, static_cast<float>(top.rms_eps));
            if (e == cudaSuccess)
                e = Gemm(stream, d_cn, lw.mla_kv_b, d_kb, c.KVL,
                         c.NH * (c.QKN + c.VH));
            if (e == cudaSuccess)
                e = RunLingRopeInterleaved(stream, d_qb, d_ca + c.KVL, c.NH,
                                           c.QKD, c.QKR, static_cast<int>(pos),
                                           top.rope_theta);
            if (e == cudaSuccess) {
                store_mla_kv_kernel<<<c.NH, 512, 0, stream>>>(
                    d_kb, d_ca + c.KVL, c.session.d_kv, c.session.geom, ord,
                    pos, c.NH, c.QKN, c.QKR, c.VH);
                e = cudaGetLastError();
            }
            if (e == cudaSuccess)
                e = RunLingMlaAttention(
                    stream, d_qb, c.session.d_kv,
                    c.scratch.d_scores + static_cast<int64_t>(ord) * c.NH *
                                             c.session.capacity,
                    c.session.geom, ord, pos + 1, c.NH, c.QKD, c.VH,
                    1.0f / std::sqrt(static_cast<float>(c.QKD)), d_mattn);
            if (e == cudaSuccess)
                e = GemvHalf(stream, lw.d_mla_g, d_normed, d_mgate, c.H,
                             c.NH);
            if (e == cudaSuccess)
                e = RunLingMlaOutGate(stream, d_mattn, d_mgate, c.NH, c.VH);
            if (e == cudaSuccess)
                e = Gemm(stream, d_mattn, lw.mla_dense, d_attn, c.NH * c.VH,
                         c.H);
            if (e != cudaSuccess) {
                error = absl::StrCat("mla layer ", l);
                return false;
            }
        }
        add_half_kernel<<<1, kThreads, 0, stream>>>(d_x, d_attn, d_x, c.H);
        if (cudaGetLastError() != cudaSuccess) {
            error = "residual add";
            return false;
        }

        if (kDump && pos == 0) {
            std::vector<__half> hx(static_cast<size_t>(c.H));
            cudaMemcpyAsync(hx.data(), d_x,
                            static_cast<size_t>(c.H) * sizeof(__half),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            double ss = 0;
            for (int i = 0; i < c.H; ++i) {
                const double v = __half2float(hx[static_cast<size_t>(i)]);
                ss += v * v;
            }
            std::fprintf(stderr,
                         "[ling-x] l=%d x0=%.6f x1=%.6f x2=%.6f x3=%.6f ss=%.6f\n",
                         l, __half2float(hx[0]), __half2float(hx[1]),
                         __half2float(hx[2]), __half2float(hx[3]), ss);
        }
        if (RunLingRmsNorm(stream, d_x, lw.d_post_norm, d_post, 1, c.H,
                           static_cast<float>(top.rms_eps)) != cudaSuccess) {
            error = "post norm";
            return false;
        }
        if (lw.is_moe) {
            cudaMemsetAsync(c.scratch.d_slots, 0,
                            sizeof(moe::MoERoutedSlot) * c.K, stream);
            cudaError_t e = RunLingRouterTopK(
                stream, reinterpret_cast<const uint8_t *>(lw.d_router),
                lw.d_router_bias, d_post, c.E, c.K, c.G, c.TG, c.H,
                static_cast<float>(top.routed_scale), c.scratch.d_slots);
            if (e != cudaSuccess) {
                error = absl::StrCat("router layer ", l);
                return false;
            }
            cudaMemcpyAsync(c.scratch.h_slots, c.scratch.d_slots,
                            sizeof(moe::MoERoutedSlot) * c.K,
                            cudaMemcpyDeviceToHost, stream);
            if (cudaStreamSynchronize(stream) != cudaSuccess) {
                error = "router readback";
                return false;
            }
            if (std::getenv("LSX_LING_DEBUG")) {
                std::vector<__half> hx(static_cast<size_t>(c.H));
                cudaMemcpyAsync(hx.data(), d_post,
                                static_cast<size_t>(c.H) * sizeof(__half),
                                cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                float mn = __half2float(hx[0]), mx = mn;
                bool bad = false;
                for (int i = 0; i < c.H; ++i) {
                    const float v = __half2float(hx[static_cast<size_t>(i)]);
                    if (!std::isfinite(v)) bad = true;
                    mn = std::min(mn, v);
                    mx = std::max(mx, v);
                }
                std::fprintf(stderr,
                             "[ling-debug] l=%d x0=%.5f min=%.5f max=%.5f "
                             "bad=%d eids=%d,%d\n",
                             l, __half2float(hx[0]), mn, mx, bad ? 1 : 0,
                             c.scratch.h_slots[0].eid,
                             c.K > 1 ? c.scratch.h_slots[1].eid : -2);
            }
            e = Gemm(stream, d_post, lw.shared_gate, d_yg, c.H, c.SI);
            if (e == cudaSuccess)
                e = Gemm(stream, d_post, lw.shared_up, d_yu, c.H, c.SI);
            if (e == cudaSuccess)
                e = RunLingSwiGLU(stream, d_yg, d_yu, d_yg, c.SI,
                              lw.shared_swiglu_limit);
            if (e == cudaSuccess)
                e = Gemm(stream, d_yg, lw.shared_down, d_shared, c.SI, c.H);
            if (e != cudaSuccess) {
                error = absl::StrCat("shared expert layer ", l);
                return false;
            }
            cudaMemsetAsync(c.scratch.d_moe_acc, 0,
                            static_cast<size_t>(c.H) * sizeof(float), stream);
            e = moe::RunMoERoutedFFN(
                stream, c.pw.expert_store, c.scratch.cache, l,
                c.scratch.h_slots, c.K, d_post, c.H, c.I, c.scratch.d_moe_acc,
                d_yg, d_yu, d_y2, error, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, lw.expert_swiglu_limit);
            if (e != cudaSuccess) {
                if (error.empty())
                    error = absl::StrCat("routed expert layer ", l);
                return false;
            }
            if (std::getenv("LSX_LING_DEBUG")) {
                std::vector<__half> hs(static_cast<size_t>(c.H));
                std::vector<float> ha(static_cast<size_t>(c.H));
                cudaMemcpyAsync(hs.data(), d_shared,
                                static_cast<size_t>(c.H) * sizeof(__half),
                                cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(ha.data(), c.scratch.d_moe_acc,
                                static_cast<size_t>(c.H) * sizeof(float),
                                cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                float sm = 0, am = 0;
                for (int i = 0; i < c.H; ++i) {
                    sm = std::max(sm, std::fabs(__half2float(hs[i])));
                    am = std::max(am, std::fabs(ha[i]));
                }
                std::fprintf(stderr,
                             "[ling-debug] l=%d shared_max=%.5f routed_max=%.5f\n",
                             l, sm, am);
            }
            if (RunLingAddRoutedShared(stream, d_x, d_shared,
                                       c.scratch.d_moe_acc, d_x,
                                       c.H) != cudaSuccess) {
                error = absl::StrCat("combine layer ", l);
                return false;
            }
        } else {
            cudaError_t e = Gemm(stream, d_post, lw.dense_gate, d_yg, c.H,
                                 c.DI);
            if (e == cudaSuccess)
                e = Gemm(stream, d_post, lw.dense_up, d_yu, c.H, c.DI);
            if (e == cudaSuccess)
                e = RunLingSwiGLU(stream, d_yg, d_yu, d_yg, c.DI);
            if (e == cudaSuccess)
                e = Gemm(stream, d_yg, lw.dense_down, d_attn, c.DI, c.H);
            if (e != cudaSuccess) {
                error = absl::StrCat("dense mlp layer ", l);
                return false;
            }
            if (kDump && pos == 0 && l == 0) {
                std::vector<__half> hm(static_cast<size_t>(c.H));
                std::vector<__half> hp(static_cast<size_t>(c.H));
                cudaMemcpyAsync(hm.data(), d_attn,
                                static_cast<size_t>(c.H) * sizeof(__half),
                                cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(hp.data(), d_post,
                                static_cast<size_t>(c.H) * sizeof(__half),
                                cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                double ss = 0, sp = 0;
                for (int i = 0; i < c.H; ++i) {
                    const double v = __half2float(hm[static_cast<size_t>(i)]);
                    const double u = __half2float(hp[static_cast<size_t>(i)]);
                    ss += v * v;
                    sp += u * u;
                }
                std::fprintf(stderr,
                             "[ling-mlp] y0=%.6f y1=%.6f y2=%.6f y3=%.6f ss=%.6f "
                             "post0=%.6f post1=%.6f postss=%.6f\n",
                             __half2float(hm[0]), __half2float(hm[1]),
                             __half2float(hm[2]), __half2float(hm[3]), ss,
                             __half2float(hp[0]), __half2float(hp[1]), sp);
            }
            add_half_kernel<<<1, kThreads, 0, stream>>>(d_x, d_attn, d_x,
                                                        c.H);
            if (cudaGetLastError() != cudaSuccess) {
                error = absl::StrCat("dense combine layer ", l);
                return false;
            }
        }
        if (kDump && pos == 0) {
            std::vector<__half> hx(static_cast<size_t>(c.H));
            cudaMemcpyAsync(hx.data(), d_x,
                            static_cast<size_t>(c.H) * sizeof(__half),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            double ss = 0;
            for (int i = 0; i < c.H; ++i) {
                const double v = __half2float(hx[static_cast<size_t>(i)]);
                ss += v * v;
            }
            std::fprintf(stderr,
                         "[ling-xf] l=%d x0=%.6f x1=%.6f x2=%.6f x3=%.6f ss=%.6f\n",
                         l, __half2float(hx[0]), __half2float(hx[1]),
                         __half2float(hx[2]), __half2float(hx[3]), ss);
        }
    }
    (void)pos;
    return true;
}

} // namespace

gpu::InferenceResult RunLingInference(
    const std::string &model_path, const std::vector<int32_t> &input_ids,
    int max_gen_tokens, const std::vector<int32_t> &eos_ids,
    LingPinnedWeights &pw, LingSession &session, LingScratch &scratch,
    sample::ISampler &sampler, cudaStream_t stream, bool ignore_eos,
    const GenerationStream *gs)
{
    gpu::InferenceResult out;
    out.num_layers = pw.top.num_layers;
    (void)model_path;
    // Previous calls report their own failures; a code left in the per-thread
    // slot by a benign wrapper must not be attributed to the first launch of
    // this request.
    (void)cudaGetLastError();

    const LingIndex &top = pw.top;
    const int H = top.hidden;
    const int V = top.vocab;
    const int KD = top.heads * top.head_dim;
    const int QKD = top.qk_nope + top.qk_rope;
    const int KVD = top.heads * (top.qk_nope + top.v_head);
    const int I = top.moe_intermediate;
    const int SI = top.shared_intermediate;
    const int DI = top.dense_intermediate;
    const float eps = static_cast<float>(top.rms_eps);

    if (H <= 0 || V <= 0 || top.num_layers <= 0 || top.num_experts <= 0 ||
        input_ids.empty() || max_gen_tokens <= 0) {
        out.error_msg = "invalid ling topology or request";
        return out;
    }
    std::string error;
    if (!LingPrepareExpertCache(pw, scratch, error)) {
        out.error_msg = error;
        return out;
    }

    const qwen::SessionDecision decision = qwen::DecideSession(
        session.allocated(), session.last_tok, session.gen_tail, input_ids);
    const std::vector<int32_t> prefill = decision.gen_input;
    if (!decision.is_continuation) session.Reset(top, stream);

    const int prompt_steps = static_cast<int>(prefill.size());
    if (std::getenv("LSX_LING_DUMP")) {
        std::fprintf(stderr, "[ling-prompt]");
        for (int32_t id : prefill) std::fprintf(stderr, " %d", id);
        std::fprintf(stderr, "\n");
    }
    const int64_t need_cap =
        session.kv_rows + prompt_steps + max_gen_tokens + 8;
    if (cudaError_t ce = session.EnsureCapacity(top, need_cap, stream);
        ce != cudaSuccess) {
        out.error_msg = absl::StrCat("kv/state alloc failed: ",
                                     cudaGetErrorString(ce));
        return out;
    }

    const size_t maxi = static_cast<size_t>(std::max(std::max(I, SI), DI));
    const size_t sc_halves =
        3 * static_cast<size_t>(H) + 9 * static_cast<size_t>(KD) +
        static_cast<size_t>(top.q_lora) + static_cast<size_t>(top.heads) * QKD +
        static_cast<size_t>(top.kv_lora + top.qk_rope) +
        static_cast<size_t>(top.kv_lora) +
        static_cast<size_t>(top.heads) * (top.qk_nope + top.v_head) +
        static_cast<size_t>(top.heads) * top.v_head + top.heads +
        static_cast<size_t>(std::max(2 * I, static_cast<int>(maxi))) + maxi +
        2 * static_cast<size_t>(H);
    const size_t halves = static_cast<size_t>(H) + sc_halves;
    const size_t arena_need = halves * sizeof(__half);
    const size_t f32_need =
        (static_cast<size_t>(top.heads) * (1 + top.head_dim)) *
        sizeof(float);
    if (!scratch.arena || scratch.arena_bytes < arena_need + f32_need) {
        if (scratch.arena) cudaFree(scratch.arena);
        scratch.arena = nullptr;
        if (cudaMalloc(&scratch.arena, arena_need + f32_need) != cudaSuccess) {
            out.error_msg = "scratch alloc failed";
            return out;
        }
        scratch.arena_bytes = arena_need + f32_need;
    }
    if (!scratch.d_argmax) {
        cudaMalloc(&scratch.d_argmax, sizeof(int32_t));
        cudaMalloc(&scratch.d_maxv, sizeof(float));
        cudaMalloc(&scratch.d_slots,
                   sizeof(moe::MoERoutedSlot) * static_cast<size_t>(top.top_k));
        cudaMallocHost(&scratch.h_slots,
                       sizeof(moe::MoERoutedSlot) * top.top_k);
    }
    if (!scratch.d_moe_acc &&
        cudaMalloc(&scratch.d_moe_acc,
                   static_cast<size_t>(H) * sizeof(float)) != cudaSuccess) {
        out.error_msg = "moe accumulator alloc failed";
        return out;
    }
    const int64_t need_scores =
        static_cast<int64_t>(top.mla_layers.size()) * top.heads *
        session.capacity;
    if (scratch.scores_elems < need_scores) {
        if (scratch.d_scores) cudaFree(scratch.d_scores);
        scratch.d_scores = nullptr;
        if (cudaMalloc(&scratch.d_scores,
                       static_cast<size_t>(need_scores) * sizeof(float)) !=
            cudaSuccess) {
            out.error_msg = "attention score scratch alloc failed";
            return out;
        }
        scratch.scores_elems = need_scores;
    }
    if (scratch.logits_elems < V) {
        if (scratch.d_logits) cudaFree(scratch.d_logits);
        scratch.d_logits = nullptr;
        if (cudaMalloc(&scratch.d_logits,
                       static_cast<size_t>(V) * sizeof(float)) != cudaSuccess) {
            out.error_msg = "logits alloc failed";
            return out;
        }
        scratch.logits_elems = V;
    }

    // The arena is reused across requests and cudaMalloc does not zero it; a
    // single unwritten tail would otherwise inherit a previous request's bits.
    cudaMemsetAsync(scratch.arena, 0, scratch.arena_bytes, stream);
    __half *d_x = static_cast<__half *>(scratch.arena);
    __half *sc = d_x + H;
    float *f32 = reinterpret_cast<float *>(
        static_cast<uint8_t *>(scratch.arena) + arena_need);

    StepCtx ctx{stream,
                pw,
                session,
                scratch,
                top,
                H,
                KD,
                top.heads,
                top.head_dim,
                top.q_lora,
                top.kv_lora,
                top.qk_nope,
                top.qk_rope,
                top.v_head,
                QKD,
                KVD,
                top.conv_kernel,
                top.num_experts,
                top.top_k,
                top.n_group,
                top.topk_group,
                I,
                SI,
                DI,
                static_cast<int>(top.mla_layers.size()),
                top.eos_token_id};

    cudaEvent_t t_start = nullptr, t_ttft = nullptr, t_stop = nullptr,
                tok_ready = nullptr;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_ttft);
    cudaEventCreate(&t_stop);
    cudaEventCreateWithFlags(&tok_ready, cudaEventDisableTiming);
    cudaEventRecord(t_start, stream);

    std::vector<int32_t> tokens;
    tokens.reserve(static_cast<size_t>(max_gen_tokens));
    float first_maxv = 0.0f;
    bool eos_terminated = false;
    int generated = 0;

    auto cleanup = [&]() {
        if (t_start) cudaEventDestroy(t_start);
        if (t_ttft) cudaEventDestroy(t_ttft);
        if (t_stop) cudaEventDestroy(t_stop);
        if (tok_ready) cudaEventDestroy(tok_ready);
    };

    int32_t cur = prefill[0];
    for (int step = 0;; ++step) {
        const int64_t pos = session.kv_rows;
        if (!Step(ctx, cur, pos, d_x, sc, f32, error)) {
            if (std::getenv("LSX_LING_DEBUG"))
                std::fprintf(stderr, "[ling-debug] step fail: '%s' token=%d\n",
                             error.c_str(), cur);
            out.error_msg = absl::StrCat(error, " @pos=", pos);
            cleanup();
            return out;
        }
        if (RunLingRmsNorm(stream, d_x, pw.d_final_norm, sc, 1, H, eps) !=
            cudaSuccess) {
            out.error_msg = "final norm";
            cleanup();
            return out;
        }
        if (gemv::RunLxGemmVec(stream, pw.d_head, sc, 1, V, H,
                               gemv::LxDtype::kF16, gemv::LxDtype::kF16, 1.0f,
                               scratch.d_logits, gemv::LxDtype::kF32, 0.0f) !=
            cudaSuccess) {
            out.error_msg = "head";
            cleanup();
            return out;
        }
        session.kv_rows += 1;
        if (std::getenv("LSX_LING_DUMP")) {
            std::vector<float> hl(static_cast<size_t>(V));
            cudaMemcpyAsync(hl.data(), scratch.d_logits,
                            static_cast<size_t>(V) * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            std::vector<int> top(5, -1);
            std::vector<float> tv(5, -1e30f);
            for (int i = 0; i < V; ++i) {
                const float v = hl[static_cast<size_t>(i)];
                for (int k = 0; k < 5; ++k)
                    if (v > tv[static_cast<size_t>(k)]) {
                        for (int j = 4; j > k; --j) {
                            tv[static_cast<size_t>(j)] =
                                tv[static_cast<size_t>(j - 1)];
                            top[static_cast<size_t>(j)] =
                                top[static_cast<size_t>(j - 1)];
                        }
                        tv[static_cast<size_t>(k)] = v;
                        top[static_cast<size_t>(k)] = i;
                        break;
                    }
            }
            std::fprintf(stderr, "[ling-top5 step=%d]", step);
            for (int k = 0; k < 5; ++k)
                std::fprintf(stderr, " %d:%.4f", top[static_cast<size_t>(k)],
                             tv[static_cast<size_t>(k)]);
            std::fprintf(stderr, "\n");
        }

        if (step < prompt_steps - 1) {
            cur = prefill[static_cast<size_t>(step) + 1];
            continue;
        }
        if (gs && gs->Aborted()) break;

        cudaError_t e = sampler.Sample(stream, scratch.d_logits, V,
                                       sample::LogitType::kF32,
                                       scratch.d_argmax, scratch.d_maxv);
        if (e != cudaSuccess) {
            out.error_msg = "sampler";
            cleanup();
            return out;
        }
        int32_t tok = -1;
        float maxv = 0.0f;
        cudaMemcpyAsync(&tok, scratch.d_argmax, sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&maxv, scratch.d_maxv, sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(tok_ready, stream);
        cudaEventSynchronize(tok_ready);
        if (generated == 0) {
            first_maxv = maxv;
            cudaEventRecord(t_ttft, stream);
        }
        tokens.push_back(tok);
        if (gs) {
            gs->Emit(static_cast<std::uint32_t>(tok),
                     static_cast<std::uint32_t>(generated), maxv);
            if (gs->Aborted()) break;
        }
        ++generated;
        if (!ignore_eos) {
            bool stop = false;
            for (int32_t id : eos_ids)
                if (tok == id) stop = true;
            if (tok == top.eos_token_id) stop = true;
            if (stop) {
                eos_terminated = true;
                break;
            }
        }
        if (generated >= max_gen_tokens) break;
        cur = tok;
    }

    if (!tokens.empty()) session.last_tok = tokens.back();
    session.gen_tail = tokens;
    {
        std::vector<float> hlast(static_cast<size_t>(V));
        cudaMemcpyAsync(hlast.data(), scratch.d_logits,
                        static_cast<size_t>(V) * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(t_stop, stream);
        if (cudaEventSynchronize(t_stop) != cudaSuccess) {
            out.error_msg = "logits fence";
            cleanup();
            return out;
        }
        out.final_logits = std::move(hlast);
    }
    float ms_total = 0.0f, ms_ttft = 0.0f;
    cudaEventElapsedTime(&ms_total, t_start, t_stop);
    ms_ttft = ms_total;
    if (!tokens.empty()) (void)cudaEventElapsedTime(&ms_ttft, t_start, t_ttft);
    cleanup();

    out.gen_tokens = tokens;
    out.ttft_ms = static_cast<double>(ms_ttft);
    out.total_time_ms = static_cast<double>(ms_total);
    out.first_gen_tok_id = tokens.empty() ? -1 : tokens.front();
    out.first_gen_maxv = first_maxv;
    out.eos_terminated = eos_terminated;
    out.ok = true;
    out.weight_resident_bytes =
        pw.resident_bytes + scratch.arena_bytes + session.Bytes();
    out.num_tokens = static_cast<int>(tokens.size());
    if (out.total_time_ms > 0.0)
        out.tokens_per_sec =
            static_cast<double>(out.num_tokens) / (out.total_time_ms / 1000.0);
    out.kv_slots_used = static_cast<int>(session.kv_rows);
    return out;
}

} // namespace lsxcommon::ling
