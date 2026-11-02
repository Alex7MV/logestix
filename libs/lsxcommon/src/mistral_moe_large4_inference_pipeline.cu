// Mistral-Large-4-1T-A52B decode pipeline (Model Ingestion Protocol stage 5).
// Layer graph: pre-norm -> MLA attention -> residual -> post-norm ->
// (FP32 router + routed experts) + gated shared expert (or the dense MLP on
// layer 0) -> residual, then final norm -> BF16 lm_head -> tanh softcap.

#include "lsxcommon/mistral_moe_large4_inference_pipeline.h"

#include "mistral_moe_large4_constants.h"
#include "lsx_kernels_act.cuh"
#include "lsxcommon/lsx_gemv.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/log.h"

#include <absl/strings/str_cat.h>

#include <algorithm>
#include <string>

namespace lsxcommon {
namespace mistral_large4 {
namespace {

namespace frozen = lsxcommon::mistral_large4::frozen;

cudaError_t Alloc(void **p, size_t n)
{
    return cudaMalloc(p, n == 0 ? 1 : n);
}

// Overflow-proof SwiGLU over NVFP4 weights (same contract as the neutral
// routed path): the gate*up product is formed in F32 and stored to FP16 under
// a power-of-two prescale; `d_alpha` receives that scale, which the caller
// folds into its FP32 accumulator (the down projection returns y / alpha).
cudaError_t SwiGLU(cudaStream_t stream,
                   const lsx_nvfp4::LsxNvfp4Weight &gate,
                   const lsx_nvfp4::LsxNvfp4Weight &up,
                   const lsx_nvfp4::LsxNvfp4Weight &down, const __half *x,
                   __half *y, __half *g_out, __half *u_out, __half *scratch,
                   float *sw_f32, float *d_alpha, float *d_inv_alpha)
{
    const int I = gate.rows;
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
    kernels::SwigluF32Kernel<<<(I + 255) / 256, 256, 0, stream>>>(g_out, u_out,
                                                                 sw_f32, I);
    kernels::SwigluAbsMaxKernel<<<1, 256, 0, stream>>>(sw_f32, d_alpha, I);
    kernels::SwigluAlphaKernel<<<1, 32, 0, stream>>>(d_alpha, d_alpha,
                                                     d_inv_alpha);
    kernels::SwigluStoreScaledKernel<<<(I + 255) / 256, 256, 0, stream>>>(
        sw_f32, d_inv_alpha, scratch, I);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    return gemv(down, scratch, y);
}

} // namespace

void FreeLarge4Session(Large4Session &s)
{
    FreeLarge4AttnState(s.attn);
    for (void **p : {reinterpret_cast<void **>(&s.d_h),
                     reinterpret_cast<void **>(&s.d_norm),
                     reinterpret_cast<void **>(&s.d_norm16),
                     reinterpret_cast<void **>(&s.d_norm32),
                     reinterpret_cast<void **>(&s.d_attn_out),
                     reinterpret_cast<void **>(&s.d_moe_acc),
                     reinterpret_cast<void **>(&s.d_mlp_g),
                     reinterpret_cast<void **>(&s.d_mlp_u),
                     reinterpret_cast<void **>(&s.d_mlp_y),
                     reinterpret_cast<void **>(&s.d_sw_f32),
                     reinterpret_cast<void **>(&s.d_alpha),
                     reinterpret_cast<void **>(&s.d_yg),
                     reinterpret_cast<void **>(&s.d_yu),
                     reinterpret_cast<void **>(&s.d_y2),
                     reinterpret_cast<void **>(&s.d_logits),
                     reinterpret_cast<void **>(&s.d_embed_row),
                     reinterpret_cast<void **>(&s.d_slots)})
    {
        if (*p) cudaFree(*p);
        *p = nullptr;
    }
    if (s.h_slots) cudaFreeHost(s.h_slots);
    s.h_slots = nullptr;
    moe::FreeMoEExpertCache(s.cache);
    s.capacity = 0;
    s.length = 0;
}

void Large4Session::Free() { FreeLarge4Session(*this); }

cudaError_t AllocateLarge4Session(Large4Session &s, const Large4Dims &dims,
                                  const std::vector<Large4LayerWeights> &layers,
                                  int capacity, int slots,
                                  const std::vector<float> &cos_t,
                                  const std::vector<float> &sin_t,
                                  cudaStream_t stream)
{
    (void)slots;
    s.capacity = capacity;
    s.length = 0;
    s.vocab = dims.vocab;
    const int I = std::max(dims.moe_inter, dims.dense_inter);
    const size_t H = static_cast<size_t>(dims.hidden);
    cudaError_t e = cudaSuccess;
    e = e == cudaSuccess ? Alloc(reinterpret_cast<void **>(&s.d_h), H * 4) : e;
    e = e == cudaSuccess ? Alloc(reinterpret_cast<void **>(&s.d_norm), H * 2) : e;
    e = e == cudaSuccess ? Alloc(reinterpret_cast<void **>(&s.d_norm16), H * 2) : e;
    e = e == cudaSuccess ? Alloc(reinterpret_cast<void **>(&s.d_norm32), H * 4) : e;
    e = e == cudaSuccess ? Alloc(reinterpret_cast<void **>(&s.d_attn_out), H * 4) : e;
    e = e == cudaSuccess ? Alloc(reinterpret_cast<void **>(&s.d_moe_acc), H * 4) : e;
    e = e == cudaSuccess ? Alloc(reinterpret_cast<void **>(&s.d_mlp_g), static_cast<size_t>(I) * 2) : e;
    e = e == cudaSuccess ? Alloc(reinterpret_cast<void **>(&s.d_mlp_u), static_cast<size_t>(I) * 2) : e;
    e = e == cudaSuccess ? Alloc(reinterpret_cast<void **>(&s.d_mlp_y), H * 2) : e;
    e = e == cudaSuccess ? Alloc(reinterpret_cast<void **>(&s.d_sw_f32), static_cast<size_t>(I) * 4) : e;
    e = e == cudaSuccess ? Alloc(reinterpret_cast<void **>(&s.d_alpha), 2 * 4) : e;
    e = e == cudaSuccess ? Alloc(reinterpret_cast<void **>(&s.d_yg), static_cast<size_t>(2 * dims.moe_inter) * 2) : e;
    e = e == cudaSuccess ? Alloc(reinterpret_cast<void **>(&s.d_yu), static_cast<size_t>(dims.moe_inter) * 2) : e;
    e = e == cudaSuccess ? Alloc(reinterpret_cast<void **>(&s.d_y2), H * 2) : e;
    e = e == cudaSuccess ? Alloc(reinterpret_cast<void **>(&s.d_logits), static_cast<size_t>(dims.vocab) * 4) : e;
    e = e == cudaSuccess ? Alloc(reinterpret_cast<void **>(&s.d_embed_row), H * 4) : e;
    e = e == cudaSuccess ? Alloc(reinterpret_cast<void **>(&s.d_slots),
                                 static_cast<size_t>(dims.top_k) * sizeof(moe::MoERoutedSlot)) : e;
    if (e != cudaSuccess) return e;
    if (cudaMallocHost(reinterpret_cast<void **>(&s.h_slots),
                       static_cast<size_t>(dims.top_k) *
                           sizeof(moe::MoERoutedSlot)) != cudaSuccess)
        return cudaErrorMemoryAllocation;

    return AllocateLarge4AttnState(s.attn, dims, layers, capacity, cos_t,
                                   sin_t, stream);
}

cudaError_t Large4DecodeStep(Large4PinnedWeights &pw, Large4Session &s,
                             int token_id, int pos, cudaStream_t stream)
{
    const Large4Dims &dims = pw.dims;
    const int H = dims.hidden;
    const int E = dims.experts;
    const int K = dims.top_k;
    const int I = dims.moe_inter;
    const float eps = dims.rms_eps;
    cudaGetLastError();  // drain unrelated host-side errors

    // ── embedding ─────────────────────────────────────────────────────────
    if (token_id < 0 || token_id >= dims.vocab) return cudaErrorInvalidValue;
    cudaError_t e = cudaMemcpyAsync(
        s.d_norm,
        reinterpret_cast<const uint8_t *>(pw.embed.data) +
            static_cast<int64_t>(token_id) * H * 2,
        static_cast<size_t>(H) * 2, cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return e;
    e = LaunchLarge4Bf16ToF32(s.d_norm, s.d_h, H, stream);
    if (e != cudaSuccess) return e;

    for (int l = 0; l < dims.layers; ++l)
    {
        const Large4LayerWeights &lw = pw.layers[static_cast<size_t>(l)];
        e = LaunchLarge4RmsNorm(s.d_h, lw.input_ln, H, eps, s.d_norm, stream);
        if (e != cudaSuccess) return e;
        e = LaunchLarge4Bf16ToF32(s.d_norm, s.d_norm32, H, stream);
        if (e != cudaSuccess) return e;
        e = Large4AttentionStep(s.attn, dims, lw, l, s.d_norm, s.d_norm32, pos,
                                s.d_attn_out, stream);
        if (e != cudaSuccess) return e;
        e = LaunchLarge4AddInPlace(s.d_h, s.d_attn_out, H, stream);
        if (e != cudaSuccess) return e;

        e = LaunchLarge4RmsNorm(s.d_h, lw.post_ln, H, eps, s.d_norm, stream);
        if (e != cudaSuccess) return e;
        e = LaunchLarge4Bf16ToF32(s.d_norm, s.d_norm32, H, stream);
        if (e != cudaSuccess) return e;
        e = LaunchLarge4Bf16ToF16(s.d_norm, s.d_norm16, H, stream);
        if (e != cudaSuccess) return e;

        e = cudaMemsetAsync(s.d_moe_acc, 0, static_cast<size_t>(H) * 4, stream);
        if (e != cudaSuccess) return e;

        if (lw.is_moe)
        {
            e = LaunchLarge4Router(lw.router, s.d_norm32, E, K, H, s.d_slots,
                                   stream);
            if (e != cudaSuccess) return e;
            // The neutral routed-expert chain takes the slot table on the host
            // (it drives the prefetch / LRU decisions). One D2H per layer, the
            // same contract as the Mistral-Small-4 sibling (perf debt).
            e = cudaMemcpyAsync(s.h_slots, s.d_slots,
                                static_cast<size_t>(K) *
                                    sizeof(moe::MoERoutedSlot),
                                cudaMemcpyDeviceToHost, stream);
            if (e != cudaSuccess) return e;
            if (cudaStreamSynchronize(stream) != cudaSuccess)
                return cudaErrorUnknown;
            std::string moe_error;
            e = moe::RunMoERoutedFFN(stream, pw.expert_store, s.cache, l,
                                     s.h_slots, K, s.d_norm16, H, I,
                                     s.d_moe_acc, s.d_yg, s.d_yu, s.d_y2,
                                     moe_error);
            if (e != cudaSuccess)
            {
                log::error("[mistral-large4] routed MoE failed: " + moe_error);
                return e;
            }
            e = SwiGLU(stream, lw.shared_gate, lw.shared_up, lw.shared_down,
                       s.d_norm16, s.d_mlp_y, s.d_mlp_g, s.d_mlp_u, s.d_mlp_g,
                       s.d_sw_f32, s.d_alpha, s.d_alpha + 1);
            if (e != cudaSuccess) return e;
            e = LaunchLarge4AccAddScaledF16ToF32(s.d_moe_acc, s.d_mlp_y,
                                                 s.d_alpha, H, stream);
            if (e != cudaSuccess) return e;
        }
        else
        {
            e = SwiGLU(stream, lw.dense_gate, lw.dense_up, lw.dense_down,
                       s.d_norm16, s.d_mlp_y, s.d_mlp_g, s.d_mlp_u, s.d_mlp_g,
                       s.d_sw_f32, s.d_alpha, s.d_alpha + 1);
            if (e != cudaSuccess) return e;
            e = LaunchLarge4AccAddScaledF16ToF32(s.d_moe_acc, s.d_mlp_y,
                                                 s.d_alpha, H, stream);
            if (e != cudaSuccess) return e;
        }
        e = LaunchLarge4AddInPlace(s.d_h, s.d_moe_acc, H, stream);
        if (e != cudaSuccess) return e;
    }

    // ── final norm + lm_head + softcap ────────────────────────────────────
    e = LaunchLarge4RmsNorm(s.d_h, pw.final_norm, H, eps, s.d_norm, stream);
    if (e != cudaSuccess) return e;
    e = gemv::RunLxGemmVec(stream, pw.head.data, s.d_norm, 1, dims.vocab, H,
                           gemv::LxDtype::kBF16, gemv::LxDtype::kBF16, 1.0f,
                           s.d_logits, gemv::LxDtype::kF32, 0.0f);
    if (e != cudaSuccess) return e;
    e = LaunchLarge4LogitSoftcap(s.d_logits, dims.vocab, dims.logit_softcap,
                                 stream);
    if (e != cudaSuccess) return e;

    s.length = pos + 1;
    return cudaSuccess;
}

void Large4ReadLogits(const Large4Session &s, float *host_logits,
                      cudaStream_t stream)
{
    if (s.d_logits == nullptr || s.vocab <= 0) return;
    cudaMemcpyAsync(host_logits, s.d_logits,
                    static_cast<size_t>(s.vocab) * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
}

} // namespace mistral_large4
} // namespace lsxcommon
