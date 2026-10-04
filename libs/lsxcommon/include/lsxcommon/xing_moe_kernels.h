#pragma once

// Xing4.0-29B-A4B elementwise/device-math kernels (stages 5-6).
//
// The family-specific kernels here are the ones no other family needs:
//   * XingUnweightedRmsNorm4Kernel — Xing4_0UnweightedRMSNorm over the
//     CONCATENATED hc_mult streams (one scalar per token, no weight vector);
//   * XingHyperConnKernel — the mHC block: pre/post/Comb logits from hc_fn,
//     sigmoid gates, clamped + Sinkhorn-normalised 4x4 mixing matrix, and the
//     per-stream collapse that the norms consume;
//   * XingMixKernel — post ⊗ out + comb @ hidden (the four-stream residual);
//   * XingRouterKernel — sigmoid noaux_tc routing (bias-corrected top-k,
//     norm_topk_prob, routed_scaling_factor) for one token.
//
// Everything else (weighted RMSNorm, SwiGLU) is the shared model-neutral pair
// in lsx_kernels_norm.cuh / lsx_kernels_act.cuh.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace lsxcommon::xing {

// out[j] = x[j] * rsqrt(mean(x^2 over hc_mult*hidden) + eps). One block,
// thread-0 reduction behind a barrier (deterministic, no atomics — it feeds
// final_logits through 40 layers).
cudaError_t LaunchXingUnweightedRmsNorm4(const float *x, float *out,
                                          int hc_mult, int hidden, float eps,
                                          cudaStream_t stream);

// Xing4_0HyperConnection.forward for one token.
//   streams      [hc_mult * hidden]  the four hidden states, stream-major
//   hc_fn        [mix, hc_mult * hidden]  BF16, mix = (2+hc_mult)*hc_mult
//   hc_base      [mix]                    BF16
//   hc_scale     [3]                      F32 (pre, post, comb gains)
//   collapsed    [hidden]  out = sum_d sigmoid(pre) * hidden_d
//   post         [hc_mult] out = post_gain * sigmoid(post)
//   comb         [hc_mult * hc_mult]     Sinkhorn-normalised mixing matrix
cudaError_t LaunchXingHyperConnection(
    const float *streams, const __nv_bfloat16 *hc_fn,
    const __nv_bfloat16 *hc_base, const float *hc_scale, int hc_mult,
    int hidden, float eps, int sinkhorn_iters, float clamp_min, float clamp_max,
    float post_gain, float *collapsed, float *post, float *comb,
    cudaStream_t stream);

// hidden_new[d][c] = post[d] * out[c] + sum_k comb[d][k] * streams[k][c]
// `streams` MAY alias `hidden_new` (the decode loop mixes in place): the kernel
// buffers one column in registers before writing it.
cudaError_t LaunchXingMix(const float *streams, const float *out,
                          const float *post, const float *comb, int hc_mult,
                          int hidden, float *hidden_new, cudaStream_t stream);

// Sigmoid noaux_tc router for ONE token: scores = sigmoid(h . w_e),
// top-k by (score + bias), weights renormalised (norm_topk_prob) and scaled
// by routed_scaling_factor. Writes top-k*4 values per output lane and returns
// the expert ids; the caller feeds them straight into the expert chain.
//   weight  [num_experts, hidden] BF16
//   bias    [num_experts] F32 (e_score_correction_bias)
//   x       [hidden] BF16 - the device-independent neutral GEMV only dispatches
//           matching dtypes, so the router's activation is BF16 like its weight
//           (the kernel converts per element, keeping the dot product in F32).
// Causal MHA decode over one layer's KV cache slice.
//   q      [heads, qk]                        this token's query, rope applied
//   kc/vc  [n_pos, heads, qk] / [n_pos, heads, vh]   rows 0..pos are valid
//   scores [capacity * heads]                 scratch, one n_pos row per head
//   out    [heads, vh]
// `n_pos` (= pos + 1) sizes the shared staging; it is passed separately so the
// caller never has to round the context up to a block size.
cudaError_t LaunchXingAttention(const float *q, const float *kc, const float *vc,
                                int pos, int heads, int qk, int vh, float scale,
                                float *scores, float *out, int capacity,
                                int n_pos, cudaStream_t stream);

cudaError_t LaunchXingRouter(const __nv_bfloat16 *weight, const float *bias,
                             const __nv_bfloat16 *x, int num_experts, int hidden,
                             int top_k, bool norm_topk_prob,
                             float routed_scale, float *out_weights,
                             int32_t *out_experts, cudaStream_t stream);

}  // namespace lsxcommon::xing