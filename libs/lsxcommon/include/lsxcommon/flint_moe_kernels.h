#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace lsxcommon::flint {

// One routed slot: selected expert id + routing coefficient (the renormalized
// sigmoid score; Flint has no routed scale factor). 8 bytes.
struct alignas(8) FlintRoutedSlot {
    int32_t eid;
    float coeff;
};

// ── Numeric kernels (FP16 activations, FP32 accumulation) ───────────────────
//
// Every kernel here is stream-ordered on the caller's stream, deterministic
// run to run, and uses no atomicAdd in anything feeding final_logits.

// Flint router (the vendor's `sigmoid_routing`):
//   logits_e = W[e] . x                (F32 weights, F32 accumulation)
//   score_e  = sigmoid(logits_e)
//   select   = top-K of score_e        (lower expert id wins ties)
//   coeff_k  = score_e / (sum(selected) + 1e-20)
// The checkpoint carries no expert bias, so selection is plain sigmoid top-k.
// Writes K slots per token ordered by ascending expert id (deterministic
// accumulation order for the caller).
// `d_ids` (optional, [K]) receives the selected ids for the device-table GEMV.
cudaError_t RunFlintRouter(cudaStream_t stream,
                           const float *router_w,   // [E,H]
                           const __half *x,         // [seq,H]
                           int seq, int E, int K, int H,
                           FlintRoutedSlot *d_slots,   // [seq*K]
                           int32_t *d_ids = nullptr);

// Full rotary over every head dim (HF rotate_half pairing j <-> j + head_dim/2),
// FP16 in place. `q` may be null (k-only). Position of row t is pos_base + t.
// Flint applies this ONLY on sliding-window layers; the full-attention layers
// carry no positional encoding at all.
cudaError_t RunFlintRope(cudaStream_t stream, __half *q, __half *k, int seq,
                         int q_heads, int k_heads, int head_dim, float theta,
                         int pos_base);

// Plain RMSNorm over full rows: out = rmsnorm(x) * w (weight form, NOT Gemma's
// 1+w). x: [rows, hidden].
cudaError_t RunFlintNorm(cudaStream_t stream, const __half *x, const __half *w,
                         __half *out, int rows, int hidden, float eps);

// ── FP32 residual support ───────────────────────────────────────────────────
//
// Flint's residual stream carries MASSIVE activations (the source model reaches
// ~1.4e4 at the last layers, and one pre-final layer amplifies a 29-magnitude
// input by ~300x). BF16 represents that range, FP16 does not: the checkpoint
// must keep its residual stream in FP32 (the DeepSeek-Pro precedent, where the
// HC-combined vector is FP32 before the scale-invariant RMSNorm). Everything
// the projections consume is scale-normalized, so the GEMM/attention legs stay
// FP16.

// RMSNorm with an FP32 input row and an FP16 (normalized) output row. The
// variance is computed in FP32 from the FP32 residual, so the 1e4 magnitudes
// never leave the accumulator.
cudaError_t RunFlintNormF32In(cudaStream_t stream, const float *x,
                              const __half *w, __half *out, int rows,
                              int hidden, float eps);

// Token embedding lookup from an FP16 table into an FP32 residual row.
cudaError_t RunFlintEmbedF32(cudaStream_t stream, const __half *table,
                             const int32_t *ids, float *out, int seq,
                             int hidden);

// acc[i] += float(add[i])  (the attention/MLP branch added to the FP32
// residual, one rounding at the very end).
cudaError_t RunFlintAddHalfToF32(cudaStream_t stream, float *acc,
                                 const __half *add, int64_t n,
                                 float scale = 1.0f);

// acc[i] += add[i]  (FP32 branch into the FP32 residual).
cudaError_t RunFlintAddF32(cudaStream_t stream, float *acc, const float *add,
                           int64_t n);

// Plain per-head RMSNorm over [rows, heads, head_dim]; out may alias x.
cudaError_t RunFlintHeadNorm(cudaStream_t stream, const __half *x,
                             const __half *w, __half *out, int rows, int heads,
                             int head_dim, float eps);

// Token embedding lookup from an FP16 table: out[t][h] = table[ids[t]][h].
cudaError_t RunFlintEmbed(cudaStream_t stream, const __half *table,
                          const int32_t *ids, __half *out, int seq, int hidden);

// ── Overflow-proof FFN scaling (power-of-two, exact) ───────────────────────
//
// The same massive activations that force the FP32 residual also blow the FP16
// range INSIDE the expert FFN: at layer 46 of the tool prompt the gate
// projection reaches 5.3e4 and the SwiGLU product ~1e8, both legal in the
// checkpoint's BF16. Scaling by powers of two keeps every intermediate in range
// without changing the arithmetic:
//   gate/up GEMV input      x/64        (kFfnInScale)
//   SwiGLU output           silu(64a)*(64b) / 8192 = silu(a)*b / 8192
//   down GEMV output        out / 8192  (linear in the scaled input)
//   consumer                x8192       (kFfnOutScale)
// The pair is chosen so the OBSERVED extremes keep a margin on both sides
// (gate 5.3e4 -> 833, SwiGLU 1.5e8 -> 1.8e4) while the smallest down outputs
// stay on the normal FP16 grid: an 8192 restore divides the effective floor to
// 5e-4, i.e. finer than the source's own BF16 rounding of the same tensor.
inline constexpr float kFfnInScale = 64.0f;
inline constexpr float kFfnOutScale = 8192.0f;

// dst[i] = src[i] * s  (FP16 -> FP16; `s` is a power of two, so exact).
cudaError_t RunFlintScaleHalf(cudaStream_t stream, const __half *src,
                              __half *dst, int64_t n, float s);

// SwiGLU in FP32 over the scaled gate/up pair: out = silu(kFfnInScale*g) *
// (kFfnInScale*u) / kFfnOutScale, stored FP16.
cudaError_t RunFlintSwiGLU( cudaStream_t stream, const __half *gate,
                           const __half *up, __half *out, int64_t n);

// acc[i] = out_scale * sum_{k<count} slots[k].coeff * y[k*H + i].
cudaError_t RunFlintExpertAccumulate(cudaStream_t stream, float *acc,
                                     const __half *y,
                                     const FlintRoutedSlot *slots, int count,
                                     int hidden, float out_scale = 1.0f);

// out[t][h] = float(shared[t][h]) + acc[t][h]  (the routed sum and the ungated
// shared expert combine in FP32, then round once).
// Writes the FP32 combined branch: the routed sum plus the shared expert can
// reach 1.6e5 on the massive-activation layers, which FP16 cannot hold.
cudaError_t RunFlintMoECombine(cudaStream_t stream, const float *acc,
                               const __half *shared, float *out, int rows,
                               int hidden, float shared_scale = 1.0f);

} // namespace lsxcommon::flint
