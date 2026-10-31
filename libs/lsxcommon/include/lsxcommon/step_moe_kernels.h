#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "lsxcommon/lsx_moe_inference.h"

namespace lsxcommon::step {

// One routed slot: selected expert id + renormalized routing coefficient
// (already multiplied by the router scaling factor 3.0).
using StepRoutedSlot = moe::MoERoutedSlot;

// Step5RMSNorm is ZERO-CENTERED: out = x / rms(x) * (1 + w). The checkpoint's
// weights are centred on zero, so reading them as a plain scale (w) multiplies
// every activation by ~0. Every norm below therefore takes the zero-centred
// form; the host-reference tests pin it.
//
// F32 residual -> FP32 RMSNorm -> FP16 activation (one row).
cudaError_t RunStepNormF32ToHalf(cudaStream_t stream, const float *x,
                                     const float *w, __half *out, int n,
                                     float eps);

// F32 residual -> BF16 narrowing -> FP32 RMSNorm -> BF16 activation. The final
// norm's narrowing is part of the reference arithmetic.
cudaError_t RunStepNormF32ToBf16Narrowed(cudaStream_t stream, const float *x,
                                         const float *w, __nv_bfloat16 *out,
                                         int n, float eps);

// Per-head QK RMSNorm: FP16 input, FP32 weight, FP16 output.
cudaError_t RunStepHeadNorm(cudaStream_t stream, const __half *x,
                                const float *w, __half *out, int heads,
                                int head_dim, float eps);

// NEOX-paired partial RoPE: rotates dims [0, span) in pairs (j, j+span/2).
// Trigonometry is computed in FP32 from the position, then rounded to FP16.
cudaError_t RunStepPartialRope(cudaStream_t stream, __half *q, int q_heads,
                                   __half *k, int k_heads, int head_dim,
                                   int span, float theta, int pos);

// Windowed causal GQA decode attention (one query row). `window <= 0` is the
// full causal range; FP16 inputs, FP32 online softmax, FP16 out.
cudaError_t RunStepWindowedAttnDecode(cudaStream_t stream, const __half *q,
                                      const __half *k_cache,
                                      const __half *v_cache, __half *out,
                                      int pos, int heads, int kv_heads,
                                      int head_dim, float scale, int window);

// Per-head sigmoid output gate: attn[h, :] *= sigmoid(g[h]) in FP32.
cudaError_t RunStepApplyHeadGate(cudaStream_t stream, __half *attn,
                                     const __half *g, int heads, int head_dim);

// Sigmoid router with a selection-only bias:
//   logit_e = W[e] . x (F32) ; score_e = sigmoid(logit_e) + bias_e
//   top-k of score (lower index wins ties, emitted descending)
//   coeff_k = (score_k - bias_k) / sum * routed_scale
cudaError_t RunStepRouter(cudaStream_t stream, const __nv_bfloat16 *gate,
                              const float *bias, const __half *x, int E, int K,
                              int H, float routed_scale, StepRoutedSlot *d_slots,
                              int32_t *d_ids);

// Clamped SwiGLU: min(silu(gate), limit) * clamp(up, -limit, limit), F32 math.
cudaError_t RunStepSwigluClamp(cudaStream_t stream, const __half *gate,
                                   const __half *up, __half *out, int64_t n,
                                   float limit);

// residual_f32 += bf16(routed_acc_f32) + shared_f16.
cudaError_t RunStepCombineF32(cudaStream_t stream, const float *routed_acc,
                                  const __half *shared, float *residual, int n);

// residual_f32 += attn_out_f16.
cudaError_t RunStepAddHalfToF32(cudaStream_t stream, const __half *y,
                                    float *residual, int n);

// residual_f32 = f32(embed[id]).
cudaError_t RunStepEmbedBf16(cudaStream_t stream,
                                 const __nv_bfloat16 *table, const int32_t *ids,
                                 float *out, int n);

} // namespace lsxcommon::step
