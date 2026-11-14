#pragma once

// Inkling-2-Small numeric kernels.
//
// Only the arithmetic that exists in NO other ingested family lives here. The
// model-neutral primitives (NVFP4 GEMV/GEMM, RMSNorm, SwiGLU, expert store,
// KV geometry) come from `lsx_nvfp4`, `lsx_gemm`, `lsx_kernels_*` and
// `lsx_kv_cache.h`; a family-owned GEMM or NVFP4 dispatcher is forbidden.
//
// The five deltas this file owns, each pinned by a host-reference case in
// tests/test_inkling_moe_kernels.cu:
//
//  1. AttnRes depth mixing: a single-head scaled-dot-product mixture over a
//     stack of block prefix sums, with an UNWEIGHTED RMSNorm on every source row
//     and a learned fixed query (1 query for sliding/MLP inputs, 3 for the full
//     layers' separate q/k/v inputs).
//  2. Learned per-head L2 clipping of q / relative states (`scaled`: an
//     element-wise learned scale first) and of k (weight-free). This replaces
//     QK-RMSNorm, which this family does not have.
//  3. A learned relative-position bias: `bias[h][d] = r_h . proj[:,d]` gathered
//     at the backward distance `d`, zero outside `[0, rel_extent)`. There is NO
//     RoPE anywhere in this model.
//  4. Gated attention with learned per-head SINKS: one extra logit per query
//     with a zero value, which only absorbs probability mass.
//  5. Four short causal convolutions per layer with two DIFFERENT tap orders
//     (see RunInklingShortConv), each blended with its own input and normalised
//     by `1/sqrt(k+1)` (static) or `1/sqrt(2k+1)` (static+dynamic taps).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "lsxcommon/lsx_kv_cache.h"

#include <cstdint>

namespace lsxcommon::inkling {

// ── routed slot (router output) ────────────────────────────────────────────
struct InklingRoutedSlot {
    int eid;
    float coeff;
};

// ── 1. AttnRes depth mixture ───────────────────────────────────────────────
// sources: [n_src, hidden] F32, the stack of block prefix sums plus the live
//          partial state (caller keeps the order: blocks first, live last).
// queries: [num_queries, hidden] F32, the learned fixed query bank.
// out:     [num_queries, hidden] F32.
// Each source row is first normalised by an unweighted RMSNorm (no weight, eps
// default 1e-6), then out[i] = sum_j softmax_j(q_i . s_j * hidden**-0.5) * s_j
// with the NORMALISED s_j on both sides.
cudaError_t RunInklingDepthMix(cudaStream_t stream, const float *sources,
                               int n_src, const float *queries, int num_queries,
                               float *out, int hidden, float eps);

// ── 2. L2 clipping ─────────────────────────────────────────────────────────
// In place, per head row: v = x * scale (when `scale` is non-null, one value per
// (head, channel)), then v *= min(threshold / (||v|| + 1e-6), 1). FP32 in and
// out: the reference upcasts, clips in FP32 and casts back, so the caller keeps
// the vector in FP32 between the projection and the cache store.
cudaError_t RunInklingL2Clip(cudaStream_t stream, float *x, const float *scale,
                             int rows, int dim, float threshold);

// Unweighted RMSNorm in place, FP32: x *= rsqrt(mean(x^2) + eps).
cudaError_t RunInklingUnweightedNorm(cudaStream_t stream, float *x, int rows,
                                     int dim, float eps);

// ── 3. relative-position bias ──────────────────────────────────────────────
// For ONE query position: rel_logits[h][d] = sum_c rel_states[h][c] * proj[c][d]
// for d in [0, rel_extent); bias[h][j] = rel_logits[h][q_pos - j] when
// 0 <= q_pos - j < rel_extent, else 0 (j indexes cached key positions).
// bias: [heads, kv_len] F32.
cudaError_t RunInklingRelBias(cudaStream_t stream, const float *rel_states,
                              const float *proj, float *bias, int heads,
                              int d_rel, int rel_extent, int q_pos,
                              int kv_len);

// ── 4. gated attention with sinks ──────────────────────────────────────────
// Appends the current key/value to the layer's cache block at `pos`.
cudaError_t RunInklingKvAppend(cudaStream_t stream, const __half *k_new,
                               const __half *v_new, __half *d_k, __half *d_v,
                               const KvCacheGeometry &geom, int pos);

// out[h] = sum_j p_j v_j with
//   logit_j  = scale * (q_h . k_j) + bias[h][j]        (bias may be null)
//   p        = softmax over [logits..., sink[h]]        (sink may be null)
// and the sliding `window` (0 = full attention) applied as `j > pos - window`.
// q: [heads, head_dim]; out: [heads, head_dim], FP16 in and out. `heads` is the
// query-head count; the kernel maps each query head onto `geom.kv_heads` (GQA
// repeat = heads / kv_heads, which is 6 in this checkpoint).
cudaError_t RunInklingAttention(cudaStream_t stream, const __half *q,
                                const __half *d_k, const __half *d_v,
                                const KvCacheGeometry &geom, int heads, int pos,
                                const float *sink, const float *bias, int window,
                                float scale, __half *out);

// out = x * sigmoid(gate), FP16, n elements (the attention output gate).
cudaError_t RunInklingGateMul(cudaStream_t stream, const __half *x,
                              const __half *gate, __half *out, int64_t n);

// ── 5. short convolutions ──────────────────────────────────────────────────
// TWO tap orders exist in this checkpoint and they are NOT the same:
//   * `conv1d` taps (the post-attention / post-MLP / sliding k-v convs):
//     out[c] = sum_j w[c][j] * state[j][c],  state[j] = x[t-(k-1)+j]
//     so tap k-1 weights the CURRENT token and tap 0 the oldest.
//   * `rel` taps (the full layers' dynamic k-v conv): lag 0 (weight
//     `static[c][0]` plus the per-token projection) weights the CURRENT token,
//     so out[c] = sum_lag taps * state[k-1-lag].
// In both cases out = (conv + x_t) * inv_scale with `inv_scale` either
// 1/sqrt(k+1) (static) or 1/sqrt(2k+1) (static + dynamic).
//
// `state` is the [k, channels] FP32 ring in chronological order (index k-1 is
// the current token). The kernel pushes `x_t` into the ring itself, so the
// caller never shifts the window.
// The taps are FP32 on purpose: the reference keeps every `*_convolution`
// module in FP32 (`_keep_in_fp32_modules_strict`), unlike the projections.
cudaError_t RunInklingShortConv(cudaStream_t stream, const __half *x_t,
                                float *state, const float *w,
                                const float *dyn_taps, __half *out, int channels,
                                int k, float inv_scale);

// Full-attention log scaling: q *= tau and rel *= tau in place, FP32. tau
// itself is 1 + alpha*ln(max((pos+1)/floor, 1)), computed on the host (one
// scalar per token) and applied to the CLIPPED q and the CLIPPED relative
// states, which is the order the reference uses.
cudaError_t RunInklingLogScale(cudaStream_t stream, float *q, float *rel,
                              int64_t nq, int64_t nrel, float tau);

// Assembles the full layers' dynamic convolution taps:
//   taps[ch*k + lag] = static_w[lag*channels + ch] + proj_out[head(ch)*k + lag]
// where head(ch) = ch / head_dim. `proj_out` is the per-token projection of the
// normalized key/value stream ([kv_heads * k] values); note that it depends on
// the head and the lag but NOT on the channel inside the head, which is why one
// projection serves a whole head's taps.
cudaError_t RunInklingDynamicTaps(cudaStream_t stream, const float *static_w,
                                 const float *proj_out, float *taps,
                                 int channels, int head_dim, int k);

// ── 6. softcapped SwiGLU (the routed expert activation) ────────────────────
// out = tanh(g/cap_g)*cap_g * sigmoid(g) * tanh(u/cap_u)*cap_u, computed in FP32
// from the raw gate/up halves and cast back to FP16 on store.
cudaError_t RunInklingSoftcapSwiGLU(cudaStream_t stream, const __half *gate,
                                    const __half *up, __half *out, int64_t n,
                                    float gate_cap, float up_cap);

// ── 7. router ──────────────────────────────────────────────────────────────
// scores = sigmoid(router_w @ x) in FP32; the top-k SELECTION uses
// `scores + bias` (bias affects the choice only); the weights are the selected
// scores renormalised to 1 and then multiplied by `route_scale`. Ties resolve to
// the lower expert id. `d_slots` is [K] and `d_ids` (may be null) receives the
// ids for the expert-batch GEMV.
cudaError_t RunInklingRouter(cudaStream_t stream, const float *router_w,
                             const float *router_bias, const __half *x, int E,
                             int K, int H, float route_scale,
                             InklingRoutedSlot *d_slots, int32_t *d_ids);

// ── 8. routed accumulation and residual plumbing ───────────────────────────
// acc[i] = sum_{k<count} slot[k].coeff * y[k*rows + i]  (FP32 accumulate, the
// reference sums the top-k expert outputs in FP32 and casts once at the end).
cudaError_t RunInklingExpertAccumulate(cudaStream_t stream, float *acc,
                                       const __half *y,
                                       const InklingRoutedSlot *slots, int count,
                                       int rows);

// out[i] = __float2half(acc[i]) (the single FP32 -> FP16 cast of the MoE block).
cudaError_t RunInklingCastToHalf(cudaStream_t stream, const float *acc,
                                 __half *out, int64_t n);

// dst[i] = a[i] + b[i] (FP32, residual accumulation).
cudaError_t RunInklingAddF32(cudaStream_t stream, float *acc, const float *add,
                             int64_t n);

// dst[i] = __half2float(src[i]).
cudaError_t RunInklingHalfToF32(cudaStream_t stream, const __half *src,
                                float *dst, int64_t n);

// dst[i] = __float2half(scale * src[i]).
cudaError_t RunInklingScaleToHalf(cudaStream_t stream, const float *src,
                                  __half *dst, int64_t n, float scale);

// out[i] = tanh(x[i] / cap) * cap, FP32 (the final logit softcap). Only the
// first `n` entries (the unpadded vocabulary) are produced.
cudaError_t RunInklingLogitSoftcap(cudaStream_t stream, const float *logits,
                                   float *out, int n, float cap);

// ── 9. plumbing shared by every layer ──────────────────────────────────────
// Weighted RMSNorm with an FP32 input and an FP16 output: the layer mixes, the
// residual stream's FP32 accumulator and the AttnRes outputs are FP32, while
// every projection consumes FP16.
cudaError_t RunInklingNormF32In(cudaStream_t stream, const float *x,
                                const __half *w, __half *out, int rows,
                                int hidden, float eps);

// y[r] = sum_c w[r*cols + c] * x[c], FP16 weight and input, FP32 output. Used
// for the two tiny non-quantized projections (`attn_sink_proj` [heads, hidden]
// and nothing else); a 4-bit rounding there would perturb the softmax
// denominator of every head, which is why they stay BF16 in the checkpoint.
cudaError_t RunInklingMatVecF16(cudaStream_t stream, const __half *w,
                                const __half *x, float *y, int rows, int cols);

// Dequantizes ONE embedding row of the NVFP4 table into FP16. `d_id` is the
// token id ON THE DEVICE, so a decode step never syncs to fetch it.
cudaError_t RunInklingEmbedRow(cudaStream_t stream, const uint8_t *packed,
                               const uint8_t *scales, float global_scale,
                               const int32_t *d_id, __half *out, int hidden);

}  // namespace lsxcommon::inkling
