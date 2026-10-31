#include "step_moe_inference_attn.h"

#include "step_moe_constants.h"

#include "lsxcommon/lsx_attn_inference.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/step_moe_kernels.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace lsxcommon::step {
namespace {

bool FineDump() { return std::getenv("LSX_STEP_DUMP_FINE") != nullptr; }

void DumpHalf(cudaStream_t stream, const char *tag, int layer, int pos,
              const __half *d, int n)
{
    if (!FineDump()) return;
    std::vector<__half> h(static_cast<size_t>(n));
    cudaMemcpyAsync(h.data(), d, static_cast<size_t>(n) * sizeof(__half),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    uint64_t hash = 1469598103934665603ull;
    const uint8_t *b = reinterpret_cast<const uint8_t *>(h.data());
    for (size_t i = 0; i < h.size() * sizeof(__half); ++i) {
        hash ^= b[i];
        hash *= 1099511628211ull;
    }
    std::fprintf(stderr, "[fine] %s l=%d pos=%d hash=%016llx v=%g", tag, layer,
                 pos, static_cast<unsigned long long>(hash),
                 static_cast<double>(__half2float(h[0])));
    if (n <= 64)
        for (int i = 0; i < n; ++i)
            std::fprintf(stderr, " %.6g", static_cast<double>(__half2float(h[static_cast<size_t>(i)])));
    std::fprintf(stderr, "\n");
}

} // namespace

cudaError_t RunStepAttentionBlock(cudaStream_t stream,
                                  const StepLayerWeights &lw,
                                  const StepTopology &top, int layer,
                                  const float *d_resid,
                                  const StepAttnScratch &s, __half *k_base,
                                  __half *v_base, int pos)
{
    const int H = top.hidden_size;
    const int NH = top.num_attention_heads;
    const int NKV = top.num_key_value_heads;
    const int HD = top.head_dim;
    const float eps = static_cast<float>(top.rms_norm_eps);
    // The frozen spans are production geometry; clamp to the actual head so a
    // synthetic fixture (smaller head_dim) can never rotate past the head into
    // neighbouring arena buffers.
    const int span = std::min(frozen::RotarySpan(layer), HD);
    const float theta = frozen::RopeTheta(layer);
    const int window = frozen::IsFullAttention(layer) ? 0 : top.sliding_window;

    cudaError_t e = RunStepNormF32ToHalf(stream, d_resid, lw.d_input_norm,
                                         s.normed, H, eps);
    if (e != cudaSuccess) return e;

    e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, s.normed, lw.attn.q.d_packed,
                                   lw.attn.q.d_scales, lw.attn.q.global_scale,
                                   s.q, 1, H, NH * HD, false);
    if (e != cudaSuccess) return e;
    e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, s.normed, lw.attn.k.d_packed,
                                   lw.attn.k.d_scales, lw.attn.k.global_scale,
                                   s.k, 1, H, NKV * HD, false);
    if (e != cudaSuccess) return e;
    e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, s.normed, lw.attn.v.d_packed,
                                   lw.attn.v.d_scales, lw.attn.v.global_scale,
                                   s.v, 1, H, NKV * HD, false);
    if (e != cudaSuccess) return e;
    e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, s.normed, lw.attn.g.d_packed,
                                   lw.attn.g.d_scales, lw.attn.g.global_scale,
                                   s.gate, 1, H, NH, false);
    if (e != cudaSuccess) return e;

    // Per-head QK norm (F32 weights, FP16 activation boundary), then partial
    // NEOX RoPE on the leading `span` dims.
    e = RunStepHeadNorm(stream, s.q, lw.attn.d_q_norm, s.q, NH, HD, eps);
    if (e != cudaSuccess) return e;
    e = RunStepHeadNorm(stream, s.k, lw.attn.d_k_norm, s.k, NKV, HD, eps);
    if (e != cudaSuccess) return e;
    e = RunStepPartialRope(stream, s.q, NH, s.k, NKV, HD, span, theta, pos);
    if (e != cudaSuccess) return e;

    // Append K/V at the current position (pos-major cache rows).
    const size_t row = static_cast<size_t>(NKV) * HD * sizeof(__half);
    e = cudaMemcpyAsync(k_base + static_cast<size_t>(pos) * NKV * HD, s.k, row,
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(v_base + static_cast<size_t>(pos) * NKV * HD, s.v, row,
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return e;

    e = RunStepWindowedAttnDecode(stream, s.q, k_base, v_base, s.attn, pos, NH,
                                  NKV, HD, frozen::kAttnScale, window);
    if (e != cudaSuccess) return e;

    DumpHalf(stream, "normed", layer, pos, s.normed, H);
    DumpHalf(stream, "q", layer, pos, s.q, NH * HD);
    DumpHalf(stream, "k", layer, pos, s.k, NKV * HD);
    DumpHalf(stream, "v", layer, pos, s.v, NKV * HD);
    DumpHalf(stream, "gate", layer, pos, s.gate, NH);
    DumpHalf(stream, "attn", layer, pos, s.attn, NH * HD);

    e = RunStepApplyHeadGate(stream, s.attn, s.gate, NH, HD);
    if (e != cudaSuccess) return e;

    return lsx_nvfp4::RunLsxNvfp4Gemm(stream, s.attn, lw.attn.o.d_packed,
                                      lw.attn.o.d_scales,
                                      lw.attn.o.global_scale, s.out, 1, NH * HD,
                                      H, false);
}

} // namespace lsxcommon::step
