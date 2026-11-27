// Nemotron-3.5 decode pipeline (Model Ingestion Protocol steps 5-6): the
// 52-layer Mamba-2 / MoE / full-attention walk for one token. Single stream;
// recurrent state and KV cache persist in NmSession across tokens.
//
// Prefill uses the same sequential recurrent path as decode (mathematically
// identical to the reference chunk scan). Router expert selection is read back
// to the host per MoE layer to drive the per-expert NVFP4 GEMVs — a correctness
// milestone; device-side expert dispatch is a perf follow-up.
#include "lsxcommon/nemotron_moe_inference.h"
#include "nemotron_moe_shared_kernels.cuh"
#include "lsx_kernels_common.cuh"
#include "nemotron_moe_kernels.h"

#include "lsxcommon/lsx_gemv.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/nemotron_moe_expert_cache.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <string>
#include <vector>

namespace lsxcommon::nemotron_moe {

namespace {

__global__ void axpy_f32_kernel(float *__restrict__ acc,
                                const float *__restrict__ x, float a, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    acc[i] += a * x[i];
}

inline int Grid1D(int n, int threads) { return (n + threads - 1) / threads; }

void HalfToFloat(const __half *src, int n, float *dst, cudaStream_t s)
{
    lsxcommon::kernels::F16ToF32Kernel<<<Grid1D(n, 256), 256, 0, s>>>(src, dst, n);
}

void FloatToHalf(const float *src, int n, __half *dst, cudaStream_t s)
{
    lsxcommon::kernels::F32ToF16Kernel<<<Grid1D(n, 256), 256, 0, s>>>(src, dst, n);
}

} // namespace

void NmSession::Free()
{
    auto free_f = [](float *&p) { cudaFree(p); p = nullptr; };
    auto free_h = [](__half *&p) { cudaFree(p); p = nullptr; };
    for (auto &p : mamba_state) free_f(p);
    for (auto &p : conv_state) free_f(p);
    for (auto &p : k_cache) free_f(p);
    for (auto &p : v_cache) free_f(p);
    mamba_state.clear(); conv_state.clear(); k_cache.clear(); v_cache.clear();
    free_h(h); free_h(normed); free_h(sub);
    free_f(gate); free_f(xbc); free_f(dt); free_f(ssm_out); free_f(gn_out);
    free_f(q); free_f(k); free_f(v); free_f(attn);
    free_f(router_logits); free_f(topk_w); free_f(expert_acc);
    free_f(head_logits);
    cudaFree(topk_i); topk_i = nullptr;
    free_h(proj16); free_h(out16);
}

bool AllocNemotronSession(const NemotronIndex &idx, int max_seq, NmSession &s,
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
    const int in_proj = s.d_inner + s.conv_dim + s.heads;

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
                     static_cast<size_t>(s.kv_heads) * max_seq *
                         s.attn_head_dim) ||
            !alloc_f(s.v_cache[i],
                     static_cast<size_t>(s.kv_heads) * max_seq *
                         s.attn_head_dim)) {
            error = "kv cache alloc failed";
            s.Free();
            return false;
        }
    }

    if (!alloc_h(s.h, s.H) || !alloc_h(s.normed, s.H) ||
        !alloc_h(s.sub, s.H) || !alloc_f(s.gate, s.d_inner) ||
        !alloc_f(s.xbc, s.conv_dim) || !alloc_f(s.dt, s.heads) ||
        !alloc_f(s.ssm_out, s.d_inner) || !alloc_f(s.gn_out, s.d_inner) ||
        !alloc_f(s.q, attn_hd_total) || !alloc_f(s.k, kv_total) ||
        !alloc_f(s.v, kv_total) || !alloc_f(s.attn, attn_hd_total) ||
        !alloc_f(s.router_logits, s.num_experts) || !alloc_f(s.topk_w, s.topk) ||
        !alloc_f(s.expert_acc, s.H) || !alloc_f(s.head_logits, idx.vocab_size) ||
        !alloc_h(s.proj16, std::max(in_proj, s.topk * idx.moe_intermediate_size)) ||
        !alloc_h(s.out16, std::max(s.H, s.topk * s.H))) {
        error = "session scratch alloc failed";
        s.Free();
        return false;
    }
    if (cudaMalloc(reinterpret_cast<void **>(&s.topk_i),
                   sizeof(int) * s.topk) != cudaSuccess) {
        error = "session topk index alloc failed";
        s.Free();
        return false;
    }
    return true;
}

void FreeNemotronSession(NmSession &s) { s.Free(); }

void ResetNemotronSession(NmSession &s, cudaStream_t stream)
{
    const int K = 4;  // conv kernel (state length K-1)
    (void)K;
    for (auto &p : s.mamba_state)
        cudaMemsetAsync(p, 0,
                        static_cast<size_t>(s.d_inner) * s.ssm_state_size *
                            sizeof(float),
                        stream);
    for (auto &p : s.conv_state)
        cudaMemsetAsync(p, 0,
                        static_cast<size_t>(s.conv_dim) * 3 * sizeof(float),
                        stream);
    for (size_t i = 0; i < s.k_cache.size(); ++i) {
        const size_t bytes = static_cast<size_t>(s.kv_heads) * s.max_seq *
                             s.attn_head_dim * sizeof(float);
        cudaMemsetAsync(s.k_cache[i], 0, bytes, stream);
        cudaMemsetAsync(s.v_cache[i], 0, bytes, stream);
    }
}

cudaError_t NemotronForwardToken(const NemotronIndex &idx, const NmWeights &w,
                                 NmSession &s, NmExpertCache &experts, int token,
                                 int pos, bool prefill, cudaStream_t stream)
{
    using namespace lsx_nvfp4;
    const int H = s.H;
    const int d_inner = s.d_inner;
    const int conv_dim = s.conv_dim;
    const int heads = s.heads;
    const int head_dim = s.head_dim;
    const int groups = s.ssm_groups;
    const int ss = s.ssm_state_size;
    const int K = idx.conv_kernel;
    const int attn_heads = idx.num_attention_heads;
    const int attn_hd = idx.head_dim;
    const int attn_out = attn_heads * attn_hd;
    const int kv_total = s.kv_heads * attn_hd;
    const float eps = static_cast<float>(idx.rms_norm_eps);
    const float attn_scale = 1.0f / sqrtf(static_cast<float>(attn_hd));

    nemotron_shared::embed_gather_kernel<> <<<Grid1D(H, 256), 256, 0, stream>>>(w.d_embed, H, token,
                                                            s.h);

    for (int l = 0; l < idx.num_layers; ++l) {
        const NmLayerWeights &lw = w.layers[l];
        nemotron_shared::rmsnorm_half_kernel<> <<<1, 256, 0, stream>>>(s.h, lw.d_norm, H, eps,
                                                   s.normed);

        if (idx.layer_types[l] == "mamba") {
            const int mi = s.mamba_slot[l];
            cudaError_t e = RunLsxNvfp4Gemm(
                stream, s.normed, lw.in_proj.d_packed, lw.in_proj.d_scales,
                lw.in_proj.global_scale, s.proj16, 1, H, lw.in_proj.rows);
            if (e != cudaSuccess) return e;
            // gate | xBC | dt
            HalfToFloat(s.proj16, d_inner, s.gate, stream);
            HalfToFloat(s.proj16 + d_inner, conv_dim, s.xbc, stream);
            HalfToFloat(s.proj16 + d_inner + conv_dim, heads, s.dt, stream);
            RunNemotronConvStep(s.xbc, lw.d_conv_w, lw.d_conv_b, s.conv_state[mi],
                                s.conv_state[mi], s.xbc, conv_dim, K, stream);
            // xBC (silu) -> xs | B | C
            float *xs = s.xbc;
            float *B = s.xbc + d_inner;
            float *C = s.xbc + d_inner + groups * ss;
            RunNemotronSelectiveStateStep(
                s.mamba_state[mi], xs, B, C, lw.d_D, s.dt, lw.d_dt_bias,
                lw.d_A_log, heads, head_dim, groups, ss, prefill ? 1 : 0,
                s.mamba_state[mi], s.ssm_out, stream);
            RunNemotronGatedRmsNorm(s.ssm_out, s.gate, lw.d_mamba_norm, d_inner,
                                    d_inner / groups, eps, s.gn_out, stream);
            FloatToHalf(s.gn_out, d_inner, s.proj16, stream);
            e = RunLsxNvfp4Gemm(stream, s.proj16, lw.out_proj.d_packed,
                                lw.out_proj.d_scales, lw.out_proj.global_scale,
                                s.out16, 1, d_inner, H);
            if (e != cudaSuccess) return e;
            nemotron_shared::residual_add_kernel<> <<<Grid1D(H, 256), 256, 0, stream>>>(s.h, s.out16,
                                                                    H);
        } else if (idx.layer_types[l] == "attention") {
            const int ai = s.attn_slot[l];
            cudaError_t e = RunLsxNvfp4Gemm(
                stream, s.normed, lw.q.d_packed, lw.q.d_scales,
                lw.q.global_scale, s.proj16, 1, H, lw.q.rows);
            if (e == cudaSuccess)
                e = RunLsxNvfp4Gemm(stream, s.normed, lw.k.d_packed,
                                    lw.k.d_scales, lw.k.global_scale, s.out16, 1,
                                    H, lw.k.rows);
            if (e != cudaSuccess) return e;
            HalfToFloat(s.proj16, attn_out, s.q, stream);
            HalfToFloat(s.out16, kv_total, s.k, stream);
            e = RunLsxNvfp4Gemm(stream, s.normed, lw.v.d_packed,
                                lw.v.d_scales, lw.v.global_scale, s.out16, 1, H,
                                lw.v.rows);
            if (e != cudaSuccess) return e;
            HalfToFloat(s.out16, kv_total, s.v, stream);
            // NO positional encoding in the full-attention layers. The
            // checkpoint's own `NemotronHAttention` (transformers 5.15.1) has
            // no rotary module and ignores `position_ids`: this hybrid model
            // is NoPE outside Mamba-2. Applying rotate_half RoPE here — as the
            // engine did until 2026-11-01 — poisons q.k with a position-
            // dependent rotation the weights were never trained with, and the
            // corruption grows with context length (measured: per-layer relL2
            // 0.09 at pos 0 -> 1.20 at pos 284 against a BF16 reference built
            // from this same converted tree, with the Tool-calling prompt
            // degenerating into HTML-tag soup). `rope_theta` stays in the
            // passport because it is a checkpoint config field, not because it
            // is applied.
            nemotron_shared::kv_store_kernel<> <<<s.kv_heads, attn_hd, 0, stream>>>(
                s.k_cache[ai], s.k, pos, s.max_seq, attn_hd);
            nemotron_shared::kv_store_kernel<> <<<s.kv_heads, attn_hd, 0, stream>>>(
                s.v_cache[ai], s.v, pos, s.max_seq, attn_hd);
            RunNemotronAttnDecode(s.q, s.k_cache[ai], s.v_cache[ai], attn_heads,
                                  s.kv_heads, attn_hd, pos + 1, s.max_seq,
                                  attn_scale, s.attn, stream);
            FloatToHalf(s.attn, attn_out, s.proj16, stream);
            e = RunLsxNvfp4Gemm(stream, s.proj16, lw.o.d_packed,
                                lw.o.d_scales, lw.o.global_scale, s.out16, 1,
                                attn_out, H);
            if (e != cudaSuccess) return e;
            nemotron_shared::residual_add_kernel<> <<<Grid1D(H, 256), 256, 0, stream>>>(s.h, s.out16,
                                                                    H);
        } else {  // moe
            cudaError_t e =
                gemv::RunLxGemmVec(stream, lw.d_router, s.normed, 1, s.num_experts,
                                   H, gemv::LxDtype::kF16, gemv::LxDtype::kF16,
                                   1.0f, s.router_logits, gemv::LxDtype::kF32,
                                   0.0f);
            if (e != cudaSuccess) return e;
            RunNemotronRouter(s.router_logits, lw.d_router_bias, s.num_experts,
                              s.topk, frozen::kRoutedScale, s.topk_w, s.topk_i,
                              stream);
            std::vector<int> ti(s.topk);
            if (cudaMemcpyAsync(ti.data(), s.topk_i, sizeof(int) * s.topk,
                                cudaMemcpyDeviceToHost, stream) != cudaSuccess)
                return cudaErrorUnknown;
            if (cudaStreamSynchronize(stream) != cudaSuccess)
                return cudaErrorUnknown;

            cudaMemsetAsync(s.expert_acc, 0, sizeof(float) * H, stream);
            const int I = idx.moe_intermediate_size;
            const int SI = lw.shared_up.rows;
            // Materialize the routed top-k in the device slot LRU (staging on
            // a miss, in the same stream) and read back the slot views.
            std::vector<NmExpertView> ev(s.topk);
            std::string res_err;
            e = ResolveNmExperts(experts, l, ti.data(), s.topk, stream, ev.data(),
                                 res_err);
            if (e != cudaSuccess) return e;
            std::vector<const uint8_t *> up_p(s.topk), up_s(s.topk);
            std::vector<const uint8_t *> dn_p(s.topk), dn_s(s.topk);
            std::vector<float> up_g(s.topk), dn_g(s.topk);
            for (int t = 0; t < s.topk; ++t) {
                up_p[t] = ev[t].up_packed;
                up_s[t] = ev[t].up_scales;
                up_g[t] = ev[t].up_global_scale;
                dn_p[t] = ev[t].down_packed;
                dn_s[t] = ev[t].down_scales;
                dn_g[t] = ev[t].down_global_scale;
            }
            // Routed experts in two batched launches (one per projection).
            e = RunLsxNvfp4GemmSeq1GemmVExpertBatch(
                stream, s.normed, up_p.data(), up_s.data(), up_g.data(), s.topk,
                /*x_stride=*/0, s.proj16, H, I, false);
            if (e != cudaSuccess) return e;
            RunNemotronRelu2Half(s.proj16, s.topk * I, stream);
            e = RunLsxNvfp4GemmSeq1GemmVExpertBatch(
                stream, s.proj16, dn_p.data(), dn_s.data(), dn_g.data(), s.topk,
                /*x_stride=*/I, s.out16, I, H, false);
            if (e != cudaSuccess) return e;
            RunNemotronMoeWeightedSum(s.out16, s.topk_w, s.topk, H,
                                      s.expert_acc, /*add=*/false, stream);
            // Shared expert (ungated relu2 MLP, every token).
            e = RunLsxNvfp4Gemm(stream, s.normed, lw.shared_up.d_packed,
                                lw.shared_up.d_scales, lw.shared_up.global_scale,
                                s.proj16, 1, H, SI);
            if (e != cudaSuccess) return e;
            RunNemotronRelu2Half(s.proj16, SI, stream);
            e = RunLsxNvfp4Gemm(stream, s.proj16, lw.shared_down.d_packed,
                                lw.shared_down.d_scales,
                                lw.shared_down.global_scale, s.out16, 1, SI, H);
            if (e != cudaSuccess) return e;
            RunNemotronAddHalfToF32(s.out16, H, s.expert_acc, stream);
            FloatToHalf(s.expert_acc, H, s.sub, stream);
            nemotron_shared::residual_add_kernel<> <<<Grid1D(H, 256), 256, 0, stream>>>(s.h, s.sub,
                                                                    H);
        }
    }

    nemotron_shared::rmsnorm_half_kernel<> <<<1, 256, 0, stream>>>(s.h, w.d_final_norm, H, eps,
                                               s.normed);
    return gemv::RunLxGemmVec(stream, w.d_head, s.normed, 1, idx.vocab_size, H,
                              gemv::LxDtype::kF16, gemv::LxDtype::kF16, 1.0f,
                              s.head_logits, gemv::LxDtype::kF32, 0.0f);
}

} // namespace lsxcommon::nemotron_moe
