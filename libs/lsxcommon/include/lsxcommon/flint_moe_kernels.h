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

// Plain per-head RMSNorm over [rows, heads, head_dim]; out may alias x.
cudaError_t RunFlintHeadNorm(cudaStream_t stream, const __half *x,
                             const __half *w, __half *out, int rows, int heads,
                             int head_dim, float eps);

// Token embedding lookup from an FP16 table: out[t][h] = table[ids[t]][h].
cudaError_t RunFlintEmbed(cudaStream_t stream, const __half *table,
                          const int32_t *ids, __half *out, int seq, int hidden);

// acc[i] = sum_{k<count} slots[k].coeff * y[k*H + i]  (FP32 accumulator).
cudaError_t RunFlintExpertAccumulate(cudaStream_t stream, float *acc,
                                     const __half *y,
                                     const FlintRoutedSlot *slots, int count,
                                     int hidden);

// out[t][h] = float(shared[t][h]) + acc[t][h]  (the routed sum and the ungated
// shared expert combine in FP32, then round once).
cudaError_t RunFlintMoECombine(cudaStream_t stream, const float *acc,
                               const __half *shared, __half *out, int rows,
                               int hidden);

} // namespace lsxcommon::flint
