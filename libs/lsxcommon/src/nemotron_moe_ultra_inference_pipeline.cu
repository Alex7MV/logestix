// Nemotron-3-Ultra-550B decode pipeline (Model Ingestion Protocol steps 5-6):
// the 108-layer Mamba-2 / MoE / full-attention walk for one token, plus the
// session allocator the walk needs. Single stream; the recurrent state and the
// KV cache persist in UltraSession across tokens.
//
// Differences from the 30B sibling that are NOT cosmetic:
//   * MoE runs through the LATENT bridge: fc1 (H->L), the routed experts at L,
//     the weighted sum in latent space, fc2 (L->H), then + shared (at H).
//   * the router is GROUPED (n_group / topk_group from the index), not n_group==1.
//   * the routed scale comes from the index (5.0 here), not a frozen constant.
//   * experts are indexed at the latent width, so the batched GEMV's K/L are L.
//
// Prefill uses the same sequential recurrent path as decode (mathematically
// identical to the reference chunk scan). Router expert selection is read back to
// the host per MoE layer to drive the per-expert NVFP4 GEMVs.
#include "lsxcommon/nemotron_moe_ultra_inference.h"

#include "nemotron_moe_ultra_kernels.h"

#include "lsxcommon/log.h"

#include <absl/strings/str_format.h>

#include "lsxcommon/lsx_gemv.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/nemotron_moe_ultra_expert_cache.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

namespace lsxcommon::nemotron_ultra {

namespace {

__global__ void embed_gather_kernel(const __half *__restrict__ e, int H, int token,
                                    __half *__restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;
    out[i] = e[static_cast<size_t>(token) * H + i];
}

__global__ void rmsnorm_half_kernel(const __half *__restrict__ x,
                                    const __half *__restrict__ w, int n, float eps,
                                    __half *__restrict__ out)
{
    __shared__ float red[256];
    const int t = threadIdx.x;
    float s = 0.0f;
    for (int i = t; i < n; i += blockDim.x) {
        const float v = __half2float(x[i]);
        s += v * v;
    }
    red[t] = s;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (t < st) red[t] += red[t + st];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / static_cast<float>(n) + eps);
    for (int i = t; i < n; i += blockDim.x)
        out[i] = __float2half(__half2float(x[i]) * inv * __half2float(w[i]));
}

__global__ void half_to_float_kernel(const __half *__restrict__ src, int n,
                                     float *__restrict__ dst)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __half2float(src[i]);
}

__global__ void float_to_half_kernel(const float *__restrict__ src, int n,
                                     __half *__restrict__ dst)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(src[i]);
}

__global__ void residual_add_kernel(__half *__restrict__ h,
                                    const __half *__restrict__ sub, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    h[i] = __float2half(__half2float(h[i]) + __half2float(sub[i]));
}

__global__ void kv_store_kernel(float *__restrict__ cache,
                                const float *__restrict__ src, int seq_pos,
                                int max_seq, int head_dim)
{
    // cache [kv_heads, max_seq, head_dim]; copy one row per head.
    const int row = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= head_dim) return;
    cache[(static_cast<size_t>(row) * max_seq + seq_pos) * head_dim + d] =
        src[static_cast<size_t>(row) * head_dim + d];
}

inline int Grid1D(int n, int threads) { return (n + threads - 1) / threads; }

// Diagnostic (LSX_NEMOTRON_ULTRA_DUMP_LAYERS=<file>, optional
// LSX_NEMOTRON_ULTRA_DUMP_POS=<position>): append one line per layer holding the
// residual stream's L2 norm and its leading dims. Without DUMP_POS every position
// is recorded, which is how a norm profile is read for a whole prompt.
//
// This is the engine side of the slice-oracle gate
// (tests/test_nemotron_ultra_layers.cpp): the oracle records the same summary
// from the reference `NemotronHBlock` output, so the two can be compared
// directly. Off unless both env vars are set, and it synchronizes, so it is a
// measurement instrument and never a run-path feature.
void DumpLayerRow(const __half *h, int n, int layer, int pos, cudaStream_t stream)
{
    const char *path = std::getenv("LSX_NEMOTRON_ULTRA_DUMP_LAYERS");
    if (!path || !*path) return;
    const char *pos_env = std::getenv("LSX_NEMOTRON_ULTRA_DUMP_POS");
    if (pos_env && *pos_env && pos != std::atoi(pos_env)) return;

    // 64 leading dims: the gate's cosine runs over these. 8 is too few to be a
    // meaningful direction estimate (0.98 over 8 dims can still be a 3e-4 L2
    // match, which is what the first version of this gate showed).
    const int kDims = 64;
    std::vector<__half> host(static_cast<std::size_t>(n));
    if (cudaMemcpyAsync(host.data(), h, host.size() * sizeof(__half),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        return;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return;
    double ss = 0.0;
    for (int i = 0; i < n; ++i) {
        const double v = __half2float(host[static_cast<std::size_t>(i)]);
        ss += v * v;
    }
    const double l2 = std::sqrt(ss);
    FILE *f = std::fopen(path, "a");
    if (!f) return;
    std::fprintf(f, "layer %d pos %d l2 %.6f", layer, pos, l2);
    for (int i = 0; i < kDims && i < n; ++i)
        std::fprintf(f, " %.6f", __half2float(host[static_cast<std::size_t>(i)]));
    std::fprintf(f, "\n");
    std::fclose(f);
}

void HalfToFloat(const __half *src, int n, float *dst, cudaStream_t s)
{
    half_to_float_kernel<<<Grid1D(n, 256), 256, 0, s>>>(src, n, dst);
}

void FloatToHalf(const float *src, int n, __half *dst, cudaStream_t s)
{
    float_to_half_kernel<<<Grid1D(n, 256), 256, 0, s>>>(src, n, dst);
}

} // namespace

void UltraSession::Free()
{
    auto free_f = [](float *&p) { cudaFree(p); p = nullptr; };
    auto free_h = [](__half *&p) { cudaFree(p); p = nullptr; };
    for (auto &p : mamba_state) free_f(p);
    for (auto &p : conv_state) free_f(p);
    for (auto &p : k_cache) free_f(p);
    for (auto &p : v_cache) free_f(p);
    mamba_state.clear();
    conv_state.clear();
    k_cache.clear();
    v_cache.clear();
    mamba_slot.clear();
    attn_slot.clear();
    free_h(h); free_h(normed); free_h(sub);
    free_f(gate); free_f(xbc); free_f(dt); free_f(ssm_out); free_f(gn_out);
    free_f(q); free_f(k); free_f(v); free_f(attn);
    free_f(router_logits); free_f(topk_w);
    if (topk_i) { cudaFree(topk_i); topk_i = nullptr; }
    free_f(moe_acc); free_f(head_logits);
    free_h(latent16); free_h(proj16); free_h(out16);
}

bool AllocUltraSession(const UltraIndex &idx, int max_seq, UltraSession &s,
                       std::string &error)
{
    s.H = idx.hidden_size;
    s.num_layers = idx.num_layers;
    s.num_experts = idx.num_experts;
    s.topk = idx.num_experts_per_tok;
    s.d_inner = idx.mamba_num_heads * idx.mamba_head_dim;
    s.conv_dim = s.d_inner + 2 * idx.n_groups * idx.ssm_state_size;
    s.heads = idx.mamba_num_heads;
    s.head_dim = idx.mamba_head_dim;
    s.ssm_groups = idx.n_groups;
    s.ssm_state_size = idx.ssm_state_size;
    s.kv_heads = idx.num_key_value_heads;
    s.attn_head_dim = idx.head_dim;
    s.max_seq = max_seq;
    s.moe_latent = idx.moe_latent_size;
    s.moe_intermediate = idx.moe_intermediate_size;
    s.shared_intermediate = idx.shared_expert_intermediate_size;
    s.n_group = idx.n_group;
    s.topk_group = idx.topk_group;
    s.routed_scale = idx.routed_scaling_factor;
    const int in_proj = s.d_inner + s.conv_dim + s.heads;
    const int latent = s.moe_latent > 0 ? s.moe_latent : s.H;

    auto alloc_f = [&](float *&p, size_t n) {
        return cudaMalloc(reinterpret_cast<void **>(&p), n * sizeof(float)) ==
               cudaSuccess;
    };
    auto alloc_h = [&](__half *&p, size_t n) {
        return cudaMalloc(reinterpret_cast<void **>(&p), n * sizeof(__half)) ==
               cudaSuccess;
    };
    const int K = idx.conv_kernel;
    const int attn_hd_total = idx.num_attention_heads * idx.head_dim;
    const int kv_total = s.kv_heads * idx.head_dim;

    s.mamba_slot.assign(idx.num_layers, -1);
    s.attn_slot.assign(idx.num_layers, -1);
    for (int l = 0; l < idx.num_layers; ++l) {
        if (idx.layer_types[l] == "mamba") s.mamba_slot[l] = s.mamba_layers++;
        else if (idx.layer_types[l] == "attention")
            s.attn_slot[l] = s.attn_layers++;
    }
    s.mamba_state.resize(s.mamba_layers);
    s.conv_state.resize(s.mamba_layers);
    for (int i = 0; i < s.mamba_layers; ++i) {
        if (!alloc_f(s.mamba_state[i],
                     static_cast<size_t>(s.d_inner) * s.ssm_state_size) ||
            !alloc_f(s.conv_state[i],
                     static_cast<size_t>(s.conv_dim) * (K - 1))) {
            error = "mamba state alloc failed";
            s.Free();
            return false;
        }
    }
    s.k_cache.resize(s.attn_layers);
    s.v_cache.resize(s.attn_layers);
    for (int i = 0; i < s.attn_layers; ++i) {
        if (!alloc_f(s.k_cache[i],
                     static_cast<size_t>(s.kv_heads) * max_seq * s.attn_head_dim) ||
            !alloc_f(s.v_cache[i],
                     static_cast<size_t>(s.kv_heads) * max_seq * s.attn_head_dim)) {
            error = "kv cache alloc failed";
            s.Free();
            return false;
        }
    }

    if (!alloc_h(s.h, s.H) || !alloc_h(s.normed, s.H) || !alloc_h(s.sub, s.H) ||
        !alloc_f(s.gate, s.d_inner) || !alloc_f(s.xbc, s.conv_dim) ||
        !alloc_f(s.dt, s.heads) || !alloc_f(s.ssm_out, s.d_inner) ||
        !alloc_f(s.gn_out, s.d_inner) || !alloc_f(s.q, attn_hd_total) ||
        !alloc_f(s.k, kv_total) || !alloc_f(s.v, kv_total) ||
        !alloc_f(s.attn, attn_hd_total) ||
        !alloc_f(s.router_logits, s.num_experts) || !alloc_f(s.topk_w, s.topk) ||
        !alloc_f(s.moe_acc, std::max(s.H, latent)) ||
        !alloc_f(s.head_logits, idx.vocab_size) ||
        !alloc_h(s.proj16, std::max({in_proj, s.topk * idx.moe_intermediate_size,
                                    s.shared_intermediate, latent, s.H})) ||
        !alloc_h(s.out16, std::max({s.H, s.topk * latent, attn_hd_total,
                                   s.kv_heads * s.attn_head_dim}))) {
        error = "session scratch alloc failed";
        s.Free();
        return false;
    }
    if (s.moe_latent > 0 && !alloc_h(s.latent16, s.moe_latent)) {
        error = "session latent scratch alloc failed";
        s.Free();
        return false;
    }
    if (cudaMalloc(reinterpret_cast<void **>(&s.topk_i), sizeof(int) * s.topk) !=
        cudaSuccess) {
        error = "session topk index alloc failed";
        s.Free();
        return false;
    }
    return true;
}

void FreeUltraSession(UltraSession &s) { s.Free(); }

void ResetUltraSession(UltraSession &s, cudaStream_t stream)
{
    for (auto &p : s.mamba_state)
        cudaMemsetAsync(p, 0,
                        static_cast<size_t>(s.d_inner) * s.ssm_state_size *
                            sizeof(float),
                        stream);
    for (auto &p : s.conv_state)
        cudaMemsetAsync(p, 0, static_cast<size_t>(s.conv_dim) * 3 * sizeof(float),
                        stream);
    for (size_t i = 0; i < s.k_cache.size(); ++i) {
        const size_t bytes = static_cast<size_t>(s.kv_heads) * s.max_seq *
                             s.attn_head_dim * sizeof(float);
        cudaMemsetAsync(s.k_cache[i], 0, bytes, stream);
        cudaMemsetAsync(s.v_cache[i], 0, bytes, stream);
    }
}

cudaError_t UltraForwardToken(const UltraIndex &idx, const UltraWeights &w,
                              UltraSession &s, UltraExpertCache &experts, int token,
                              int pos, bool prefill, cudaStream_t stream)
{
    using namespace lsx_nvfp4;
    const int H = s.H;
    const int d_inner = s.d_inner;
    const int conv_dim = s.conv_dim;
    const int heads = s.heads;
    const int groups = s.ssm_groups;
    const int ss = s.ssm_state_size;
    const int K = idx.conv_kernel;
    const int attn_heads = idx.num_attention_heads;
    const int attn_hd = idx.head_dim;
    const int attn_out = attn_heads * attn_hd;
    const int kv_total = s.kv_heads * attn_hd;
    const float eps = static_cast<float>(idx.rms_norm_eps);
    const float attn_scale = 1.0f / sqrtf(static_cast<float>(attn_hd));
    const int LAT = s.moe_latent > 0 ? s.moe_latent : H;
    const int I = idx.moe_intermediate_size;

    embed_gather_kernel<<<Grid1D(H, 256), 256, 0, stream>>>(w.d_embed, H, token,
                                                            s.h);

    for (int l = 0; l < idx.num_layers; ++l) {
        const UltraLayerWeights &lw = w.layers[static_cast<std::size_t>(l)];
        rmsnorm_half_kernel<<<1, 256, 0, stream>>>(s.h, lw.d_norm, H, eps,
                                                  s.normed);

        if (idx.layer_types[static_cast<std::size_t>(l)] == "mamba") {
            const int mi = s.mamba_slot[static_cast<std::size_t>(l)];
            cudaError_t e = RunLsxNvfp4Gemm(
                stream, s.normed, lw.in_proj.d_packed, lw.in_proj.d_scales,
                lw.in_proj.global_scale, s.proj16, 1, H, lw.in_proj.rows);
            if (e != cudaSuccess) return e;
            // gate | xBC | dt
            HalfToFloat(s.proj16, d_inner, s.gate, stream);
            HalfToFloat(s.proj16 + d_inner, conv_dim, s.xbc, stream);
            HalfToFloat(s.proj16 + d_inner + conv_dim, heads, s.dt, stream);
            RunUltraConvStep(s.xbc, lw.d_conv_w, lw.d_conv_b, s.conv_state[mi],
                             s.conv_state[mi], s.xbc, conv_dim, K, stream);
            float *xs = s.xbc;
            float *B = s.xbc + d_inner;
            float *C = s.xbc + d_inner + groups * ss;
            RunUltraSelectiveStateStep(s.mamba_state[mi], xs, B, C, lw.d_D, s.dt,
                                       lw.d_dt_bias, lw.d_A_log, heads, s.head_dim,
                                       groups, ss, prefill ? 1 : 0,
                                       s.mamba_state[mi], s.ssm_out, stream);
            RunUltraGatedRmsNorm(s.ssm_out, s.gate, lw.d_mamba_norm, d_inner,
                                 d_inner / groups, eps, s.gn_out, stream);
            FloatToHalf(s.gn_out, d_inner, s.proj16, stream);
            e = RunLsxNvfp4Gemm(stream, s.proj16, lw.out_proj.d_packed,
                                lw.out_proj.d_scales, lw.out_proj.global_scale,
                                s.out16, 1, d_inner, H);
            if (e != cudaSuccess) return e;
            residual_add_kernel<<<Grid1D(H, 256), 256, 0, stream>>>(s.h, s.out16, H);
        } else if (idx.layer_types[static_cast<std::size_t>(l)] == "attention") {
            const int ai = s.attn_slot[static_cast<std::size_t>(l)];
            cudaError_t e = RunLsxNvfp4Gemm(stream, s.normed, lw.q.d_packed,
                                            lw.q.d_scales, lw.q.global_scale,
                                            s.proj16, 1, H, lw.q.rows);
            if (e == cudaSuccess)
                e = RunLsxNvfp4Gemm(stream, s.normed, lw.k.d_packed,
                                    lw.k.d_scales, lw.k.global_scale, s.out16, 1, H,
                                    lw.k.rows);
            if (e != cudaSuccess) return e;
            HalfToFloat(s.proj16, attn_out, s.q, stream);
            HalfToFloat(s.out16, kv_total, s.k, stream);
            e = RunLsxNvfp4Gemm(stream, s.normed, lw.v.d_packed, lw.v.d_scales,
                                lw.v.global_scale, s.out16, 1, H, lw.v.rows);
            if (e != cudaSuccess) return e;
            HalfToFloat(s.out16, kv_total, s.v, stream);
            // NO positional encoding in the full-attention layers. The
            // checkpoint's own `NemotronHAttention` (transformers 5.15.1) has no
            // rotary module and ignores `position_ids`: this hybrid is NoPE
            // outside Mamba-2, and applying rotate_half RoPE here is the exact
            // defect the 30B sibling shipped and fixed on 2026-11-01 (per-layer
            // relL2 0.09 -> 1.20 by position 284 against a BF16 reference).
            kv_store_kernel<<<s.kv_heads, attn_hd, 0, stream>>>(
                s.k_cache[ai], s.k, pos, s.max_seq, attn_hd);
            kv_store_kernel<<<s.kv_heads, attn_hd, 0, stream>>>(
                s.v_cache[ai], s.v, pos, s.max_seq, attn_hd);
            RunUltraAttnDecode(s.q, s.k_cache[ai], s.v_cache[ai], attn_heads,
                               s.kv_heads, attn_hd, pos + 1, s.max_seq, attn_scale,
                               s.attn, stream);
            FloatToHalf(s.attn, attn_out, s.proj16, stream);
            e = RunLsxNvfp4Gemm(stream, s.proj16, lw.o.d_packed, lw.o.d_scales,
                                lw.o.global_scale, s.out16, 1, attn_out, H);
            if (e != cudaSuccess) return e;
            residual_add_kernel<<<Grid1D(H, 256), 256, 0, stream>>>(s.h, s.out16, H);
        } else {  // moe
            cudaError_t e = gemv::RunLxGemmVec(stream, lw.d_router, s.normed, 1,
                                               s.num_experts, H, gemv::LxDtype::kF16,
                                               gemv::LxDtype::kF16, 1.0f,
                                               s.router_logits, gemv::LxDtype::kF32,
                                               0.0f);
            if (e != cudaSuccess) return e;
            RunUltraRouter(s.router_logits, lw.d_router_bias, s.num_experts, s.topk,
                           s.n_group, s.topk_group, s.routed_scale, s.topk_w,
                           s.topk_i, stream);

            std::vector<int> ti(static_cast<std::size_t>(s.topk));
            if (cudaMemcpyAsync(ti.data(), s.topk_i, sizeof(int) * s.topk,
                                cudaMemcpyDeviceToHost, stream) != cudaSuccess)
                return cudaErrorUnknown;
            if (cudaStreamSynchronize(stream) != cudaSuccess) return cudaErrorUnknown;

            // Materialize the routed top-k in the device slot LRU (staging on a
            // miss, in the same stream) and read back the slot views.
            std::vector<UltraExpertView> ev(static_cast<std::size_t>(s.topk));
            std::string res_err;
            e = ResolveUltraExperts(experts, l, ti.data(), s.topk, stream, ev.data(),
                                    res_err);
            if (e != cudaSuccess) {
                // A silent return here reads as "the forward failed" with no way
                // to tell a wrong slot id from an unreadable expert file.
                lsxcommon::log::error(absl::StrFormat(
                    "[nemotron-ultra] expert resolve failed at layer %d: %s", l,
                    res_err));
                return e;
            }
            std::vector<const uint8_t *> up_p(static_cast<std::size_t>(s.topk)),
                up_s(static_cast<std::size_t>(s.topk)),
                dn_p(static_cast<std::size_t>(s.topk)),
                dn_s(static_cast<std::size_t>(s.topk));
            std::vector<float> up_g(static_cast<std::size_t>(s.topk)),
                dn_g(static_cast<std::size_t>(s.topk));
            for (int t = 0; t < s.topk; ++t) {
                const auto k = static_cast<std::size_t>(t);
                up_p[k] = ev[k].up_packed;
                up_s[k] = ev[k].up_scales;
                up_g[k] = ev[k].up_global_scale;
                dn_p[k] = ev[k].down_packed;
                dn_s[k] = ev[k].down_scales;
                dn_g[k] = ev[k].down_global_scale;
            }

            // Latent bridge: project the activation down to L first; the routed
            // experts (and therefore the batched GEMV's K) run at L, not at H.
            const __half *expert_in = s.normed;
            if (s.moe_latent > 0) {
                // Own buffer, NOT proj16: see UltraSession::latent16.
                e = RunLsxNvfp4Gemm(stream, s.normed, lw.latent_up.d_packed,
                                    lw.latent_up.d_scales,
                                    lw.latent_up.global_scale, s.latent16, 1, H,
                                    lw.latent_up.rows);
                if (e != cudaSuccess) return e;
                expert_in = s.latent16;
            }

            // Routed experts, batched per projection. The batch entry point
            // accepts at most kMaxExpertsPerBatch experts per launch (its own
            // contract), and THIS model routes to top-22 — the 30B sibling's
            // top-6 fits in one launch, so a straight copy of its call fails
            // with cudaErrorInvalidValue here. Chunking keeps the [count, N]
            // layout: chunk c writes rows [c*kChunk, ...) of the same buffer.
            constexpr int kMaxExpertsPerBatch = 8;
            for (int c = 0; c < s.topk; c += kMaxExpertsPerBatch) {
                const int cnt = std::min(kMaxExpertsPerBatch, s.topk - c);
                e = RunLsxNvfp4GemmSeq1GemmVExpertBatch(
                    stream, expert_in, up_p.data() + c, up_s.data() + c,
                    up_g.data() + c, cnt, /*x_stride=*/0,
                    s.proj16 + static_cast<std::size_t>(c) * I, LAT, I, false);
                if (e != cudaSuccess) return e;
            }
            RunUltraRelu2Half(s.proj16, s.topk * I, stream);
            for (int c = 0; c < s.topk; c += kMaxExpertsPerBatch) {
                const int cnt = std::min(kMaxExpertsPerBatch, s.topk - c);
                e = RunLsxNvfp4GemmSeq1GemmVExpertBatch(
                    stream, s.proj16 + static_cast<std::size_t>(c) * I,
                    dn_p.data() + c, dn_s.data() + c, dn_g.data() + c, cnt,
                    /*x_stride=*/I, s.out16 + static_cast<std::size_t>(c) * LAT, I,
                    LAT, false);
                if (e != cudaSuccess) return e;
            }
            RunUltraMoeWeightedSum(s.out16, s.topk_w, s.topk, LAT, s.moe_acc,
                                   /*add=*/false, stream);

            // Latent -> hidden, then the shared expert at the hidden width.
            if (s.moe_latent > 0) {
                FloatToHalf(s.moe_acc, LAT, s.proj16, stream);
                e = RunLsxNvfp4Gemm(stream, s.proj16, lw.latent_down.d_packed,
                                    lw.latent_down.d_scales,
                                    lw.latent_down.global_scale, s.out16, 1, LAT, H);
                if (e != cudaSuccess) return e;
                // s.out16 now holds the routed result at H; move it into moe_acc
                // so the shared expert can be added, then reuse out16 for it.
                cudaMemsetAsync(s.moe_acc, 0, sizeof(float) * H, stream);
                RunUltraAddHalfToF32(s.out16, H, s.moe_acc, stream);
            }

            const int SI = lw.shared_up.rows;
            e = RunLsxNvfp4Gemm(stream, s.normed, lw.shared_up.d_packed,
                                lw.shared_up.d_scales, lw.shared_up.global_scale,
                                s.proj16, 1, H, SI);
            if (e != cudaSuccess) return e;
            RunUltraRelu2Half(s.proj16, SI, stream);
            e = RunLsxNvfp4Gemm(stream, s.proj16, lw.shared_down.d_packed,
                                lw.shared_down.d_scales,
                                lw.shared_down.global_scale, s.out16, 1, SI, H);
            if (e != cudaSuccess) return e;
            RunUltraAddHalfToF32(s.out16, H, s.moe_acc, stream);
            FloatToHalf(s.moe_acc, H, s.sub, stream);
            residual_add_kernel<<<Grid1D(H, 256), 256, 0, stream>>>(s.h, s.sub, H);
        }
        DumpLayerRow(s.h, H, l, pos, stream);
    }

    // Diagnostic: the final un-normalised residual, so the head can be checked
    // independently of the layer walk (write once per run, last token wins).
    if (const char *hp = std::getenv("LSX_NEMOTRON_ULTRA_DUMP_HIDDEN"); hp && *hp) {
        std::vector<__half> hh(static_cast<std::size_t>(H));
        if (cudaMemcpyAsync(hh.data(), s.h, hh.size() * sizeof(__half),
                            cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
            cudaStreamSynchronize(stream) == cudaSuccess) {
            if (FILE *f = std::fopen(hp, "wb")) {
                std::fwrite(hh.data(), sizeof(__half), hh.size(), f);
                std::fclose(f);
            }
        }
    }
    rmsnorm_half_kernel<<<1, 256, 0, stream>>>(s.h, w.d_final_norm, H, eps,
                                              s.normed);
    return gemv::RunLxGemmVec(stream, w.d_head, s.normed, 1, idx.vocab_size, H,
                              gemv::LxDtype::kF16, gemv::LxDtype::kF16, 1.0f,
                              s.head_logits, gemv::LxDtype::kF32, 0.0f);
}

} // namespace lsxcommon::nemotron_ultra
