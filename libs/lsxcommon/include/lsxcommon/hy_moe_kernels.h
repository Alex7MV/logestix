#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace lsxcommon::hy {

// One routed slot: selected expert id + routing coefficient (already multiplied
// by the passport's router_scaling_factor). 8 bytes.
struct alignas(8) HyRoutedSlot {
    int32_t eid;
    float coeff;
};

// ── Numeric kernels (FP16 activations, FP32 accumulation) ───────────────────
//
// Every kernel here is stream-ordered on the caller's stream, deterministic
// run to run, and uses no atomicAdd in anything feeding final_logits.

// Hy-Memory router (modeling_hy_v3.HYV3TopKRouter):
//   logits_e = W[e] . x                  (F32 weights, F32 accumulation)
//   score_e  = sigmoid(logits_e)
//   select   = top-K of (score_e + bias_e)      (lower expert id wins ties)
//   coeff_k  = (score_e / sum(selected)) * router_scaling_factor
// Writes K slots per token, ordered by ascending expert id (deterministic
// accumulation order for the caller).
// Launch: <<<seq, 256, shared, stream>>>.
// `d_ids` (optional, [seq*K]) receives the selected expert ids so the caller
// can feed the device-table expert GEMV without a D2H round trip.
cudaError_t RunHyRouter(cudaStream_t stream,
                        const float *router_w,     // [E,H]
                        const float *router_bias,  // [E]
                        const __half *x,           // [seq,H]
                        int seq, int E, int K, int H, float routed_scale,
                        HyRoutedSlot *d_slots,     // [seq*K]
                        int32_t *d_ids = nullptr);

// Full rotary over every head dim (HF rotate_half pairing j <-> j + head_dim/2),
// FP16 in place. `q` may be null (k-only). Position of row t is pos_base + t.
cudaError_t RunHyRope(cudaStream_t stream, __half *q, __half *k, int seq,
                      int q_heads, int k_heads, int head_dim, float theta,
                      int pos_base);

// Plain RMSNorm over full rows: out = rmsnorm(x) * w (weight form, NOT Gemma's
// 1+w). x: [rows, hidden].
cudaError_t RunHyNorm(cudaStream_t stream, const __half *x, const __half *w,
                      __half *out, int rows, int hidden, float eps);

// Plain per-head RMSNorm over [rows, heads, head_dim]; out may alias x.
cudaError_t RunHyHeadNorm(cudaStream_t stream, const __half *x,
                          const __half *w, __half *out, int rows, int heads,
                          int head_dim, float eps);

// Token embedding lookup from an FP16 table: out[t][h] = table[ids[t]][h].
cudaError_t RunHyEmbed(cudaStream_t stream, const __half *table,
                       const int32_t *ids, __half *out, int seq, int hidden);

// acc[i] = sum_{k<count} slots[k].coeff * y[k*H + i]  (FP32 accumulator).
// y is the per-expert down-projection output, [count][H] contiguous.
cudaError_t RunHyExpertAccumulate(cudaStream_t stream, float *acc,
                                  const __half *y, const HyRoutedSlot *slots,
                                  int count, int hidden);

// out[t][h] = float(shared[t][h]) + acc[t][h]  (the checkpoint's
// enable_moe_fp32_combine: the shared expert joins the routed sum in FP32).
cudaError_t RunHyMoECombine(cudaStream_t stream, const float *acc,
                            const __half *shared, __half *out, int rows,
                            int hidden);

} // namespace lsxcommon::hy
