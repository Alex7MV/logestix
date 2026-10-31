#pragma once

// Ling-3.1-flash low-level kernels: KDA (short conv + gated delta), gated MLA
// (interleaved partial RoPE + causal attention + head-wise sigmoid gate),
// group-limited sigmoid router and the SwiGLU/combine epilogues. One stream,
// no host synchronization; the generation driver owns the shape contract.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/lsx_moe_inference.h"

namespace lsxcommon::ling {

// out = x * rsqrt(mean(x^2)+eps) * w, FP16 in/out, FP32 math. rows x hidden.
cudaError_t RunLingRmsNorm(cudaStream_t stream, const __half *x,
                           const __half *w, __half *out, int rows, int hidden,
                           float eps);

// Depthwise causal conv (kernel taps, window = hist rows oldest..newest + cur)
// followed by SiLU (optional) and a history shift that appends `cur`.
// hist: [(kernel-1)][channels] FP16 (persistent); weight row-major [channels,
// kernel].
cudaError_t RunLingConvDecodeSilu(cudaStream_t stream, __half *hist,
                                  const __half *cur, const __half *weight,
                                  __half *out, int channels, int kernel,
                                  bool apply_silu);

// L2-normalize q/k rows with eps=1e-6; q additionally scaled by q_scale.
cudaError_t RunLingL2NormQK(cudaStream_t stream, __half *q, __half *k,
                            int heads, int dim, float q_scale);

// beta[h] = sigmoid(b[h]); decay[h*D+l] = exp(lower*sigmoid(exp(A_log[h]) *
// (f[h*D+l] + dt_bias[h*D+l]))).
cudaError_t RunLingKdaGating(cudaStream_t stream, const __half *b,
                             const __half *f, const float *a_log,
                             const float *dt_bias, float *beta, float *decay,
                             int heads, int dim, float lower_bound);

// One-token per-channel-decay gated delta step. state[l][d] FP32, out FP16.
cudaError_t RunLingKdaStep(cudaStream_t stream, const __half *q,
                           const __half *k, const __half *v,
                           const float *beta, const float *decay,
                           float *state, __half *out, int heads, int dim);

// out[h][l] = rmsnorm(core[h])[l] * w[l] * sigmoid(gate[h][l]).
cudaError_t RunLingRmsNormGated(cudaStream_t stream, const __half *core,
                                const __half *gate, const __half *w,
                                __half *out, int heads, int dim, float eps);

// Interleaved (pairwise) RoPE over the LAST rope_dim dims of each q head and
// over the single k row.
cudaError_t RunLingRopeInterleaved(cudaStream_t stream, __half *q,
                                   __half *k_rot, int heads, int head_dim,
                                   int rope_dim, int pos, double theta);

// Group-limited sigmoid router (sigmoid + bias for selection, raw sigmoid
// renormalized * routed_scale for the coefficients).
cudaError_t RunLingRouterTopK(cudaStream_t stream, const uint8_t *gate_bf16,
                              const float *bias, const __half *x,
                              int experts, int top_k, int n_group,
                              int topk_group, int hidden, float routed_scale,
                              moe::MoERoutedSlot *slots);

// out = silu(gate) * up, FP16.
cudaError_t RunLingSwiGLU(cudaStream_t stream, const __half *gate,
                          const __half *up, __half *out, int n,
                          float limit = 0.0f);

// Causal MLA attention for one layer over `count` cached positions.
// kv layout per (head, pos): [k (head_dim) | v (v_head)]; addressed through
// `geom` with layer slot `layer_slot`. `scores` is a [heads * capacity] FP32
// scratch.
cudaError_t RunLingMlaAttention(cudaStream_t stream, const __half *q,
                                const __half *kv, float *scores,
                                const KvCacheGeometry &geom, int layer_slot,
                                int64_t count, int heads, int head_dim,
                                int v_head, float scale, __half *out);

// out[h][d] *= sigmoid(gate[h]).
cudaError_t RunLingMlaOutGate(cudaStream_t stream, __half *out,
                              const __half *gate, int heads, int v_head);

// out[hidden] = embed[token_id * hidden + i].
cudaError_t RunLingEmbedToken(cudaStream_t stream, const __half *embed,
                              int32_t token_id, __half *out, int hidden);

// out = resid + shared + routed (FP32 routed accumulator).
cudaError_t RunLingAddRoutedShared(cudaStream_t stream, const __half *resid,
                                   const __half *shared, const float *routed,
                                   __half *out, int hidden);

} // namespace lsxcommon::ling
