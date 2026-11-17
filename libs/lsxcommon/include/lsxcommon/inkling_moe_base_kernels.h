#pragma once

// Inkling (base) numeric kernels (Model Ingestion Protocol stages 5/6).
//
// Numeric contract per kernel (host references: tests/test_inkling_moe_base_kernels.cu):
//   * every reduction is a fixed-order shared-memory tree, thread-0 publication,
//     no atomicAdd anywhere (R3 bit-exact decode);
//   * attention scale is 1/head_dim (q/k are per-head RMS-normed), NOT
//     1/sqrt(head_dim) — the inkling2 sibling uses the other one;
//   * no attention sinks, no output gate, no L2 clips (those are inkling2's);
//   * short conv is `conv(x) + x` with NO normalizer and STATIC taps only;
//   * the router renormalises jointly over the top-k routed + all shared
//     logits (softmax of logsigmoid), then x route_scale x global_scale; the
//     selection bias steers only WHICH routed experts run;
//   * SwiGLU is plain silu(g)*u — no softcaps.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "lsxcommon/lsx_kv_cache.h"

namespace lsxcommon::inkling::base {

// One selected routed expert: id + renormalised weight (joint softmax over
// the k routed + S shared logits, x route_scale x gate global_scale).
struct RoutedSlot {
    int32_t eid;
    float coeff;
};

// Weighted per-head RMSNorm over rows of `dim` elements (q/k after the short
// conv): out[r,i] = x[r,i] * rsqrt(mean(x[r]^2)+eps) * w[i]. FP32 in/out.
cudaError_t RunHeadNorm(cudaStream_t stream, const float *x, const __half *w,
                        float *out, int rows, int dim, float eps);

// Learned relative-position bias: bias[h,j] = sum_c rel[h,c]*proj[c,d] for
// d = q_pos - j in [0, extent), else 0. rel is [heads, d_rel] FP32, proj is
// [d_rel, extent] FP32, bias is [heads, kv_len] FP32.
cudaError_t RunRelBias(cudaStream_t stream, const float *rel_states,
                       const float *proj, float *bias, int heads, int d_rel,
                       int rel_extent, int q_pos, int kv_len);

// Multiply q (FP32, nq elements) and the rel bias (FP32, nrel elements) by
// tau — the full-attention log-scaling, applied BEFORE the softmax.
cudaError_t RunLogScale(cudaStream_t stream, float *q, int64_t nq, float *rel,
                        int64_t nrel, float tau);

// Append one position of K/V (FP16, [kv_heads, head_dim]) into the cache.
cudaError_t RunKvAppend(cudaStream_t stream, const __half *k_new,
                        const __half *v_new, __half *d_k, __half *d_v,
                        const KvCacheGeometry &geom, int pos);

// Single-query attention over the cached positions [0, pos]. NO sinks. The
// window (0 = full) keeps only the last `window` positions visible; `bias`
// ([heads, pos+1] FP32, may be null) is added to the scaled QK dot before the
// softmax. scale = 1/head_dim. Output FP16 [heads, head_dim].
cudaError_t RunAttention(cudaStream_t stream, const __half *q,
                         const __half *d_k, const __half *d_v,
                         const KvCacheGeometry &geom, int heads, int pos,
                         const float *bias, int window, float scale,
                         __half *out);

// Static-tap depthwise causal short conv over one token: push x into the
// [k][channels] ring, out[c] = sum_j w[c,j]*state[j,c] + x[c]. NO normalizer
// (inv_scale is always 1 here; the parameter exists only to keep the launch
// shape explicit). FP16 in/out, FP32 ring state and taps.
cudaError_t RunShortConv(cudaStream_t stream, const __half *x_t, float *state,
                         const float *w, __half *out, int channels, int k);

// Router: w is [E+S, H] FP32, bias [E] FP32 (selection only), x [H] FP16.
// Scores = sigmoid(w·x); the top-K of the first E scores+bias are selected;
// weights = softmax over logsigmoid of the K selected + S shared logits,
// x route_scale x global_scale. Writes K RoutedSlots (+ ids) and S shared
// gammas. Deterministic thread-0 selection, ties to the lower id.
cudaError_t RunRouter(cudaStream_t stream, const float *router_w,
                      const float *router_bias, const __half *x, int E, int S,
                      int K, int H, float route_scale, float global_scale,
                      RoutedSlot *d_slots, int32_t *d_ids, float *d_gammas);

// Plain SwiGLU: out = silu(gate) * up (FP16, n elements). No softcaps.
cudaError_t RunSwiGLU(cudaStream_t stream, const __half *gate,
                      const __half *up, __half *out, int64_t n);

// acc[i] = sum_k slots[k].coeff * y[k*rows+i] (FP32 acc, FP16 y).
cudaError_t RunExpertAccumulate(cudaStream_t stream, float *acc,
                                const __half *y, const RoutedSlot *slots,
                                int count, int rows);

// acc[i] += gamma * y[i] (FP32 acc, FP16 y) — one shared expert's contribution.
cudaError_t RunSharedAccumulate(cudaStream_t stream, float *acc,
                                const __half *y, float gamma, int rows);

// ── plumbing ────────────────────────────────────────────────────────────────
cudaError_t RunEmbedGather(cudaStream_t stream, const __half *table,
                           const int32_t *d_id, __half *out, int hidden);
// Weighted RMSNorm, FP32 input row(s) -> FP16 output (norm weights FP16).
cudaError_t RunNormF32In(cudaStream_t stream, const float *x, const __half *w,
                         __half *out, int rows, int hidden, float eps);
cudaError_t RunHalfToF32(cudaStream_t stream, const __half *src, float *dst,
                         int64_t n);
cudaError_t RunCastToHalf(cudaStream_t stream, const float *src, __half *dst,
                          int64_t n);
cudaError_t RunScaleToHalf(cudaStream_t stream, const float *src, __half *dst,
                           int64_t n, float scale);
cudaError_t RunAddF32(cudaStream_t stream, float *acc, const float *add,
                      int64_t n);
// FP16 table matvec -> FP32 out (the unembed head; rows may be the full vocab).
cudaError_t RunMatVecF16(cudaStream_t stream, const __half *w, const __half *x,
                         float *y, int rows, int cols);
// out[i] = logits[i] * scale (the muP 1/24 head pre-scale), FP32.
cudaError_t RunScaleF32(cudaStream_t stream, float *x, int64_t n, float scale);

}  // namespace lsxcommon::inkling::base