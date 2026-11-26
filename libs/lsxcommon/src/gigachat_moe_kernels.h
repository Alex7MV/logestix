#pragma once

// Device-kernel entry points for GigaChat3.5 elementwise gating math
// (Plan 2a). Compiled into lsxcommon_gigachat_cu; declared here for the .cu
// test/pipeline translation units.
#include <cstdint>

#include <cuda_runtime.h>

namespace lsxcommon::gigachat {

// ZeroCenteredGatedNorm over `rows` rows of `dim`:
//   xn[r]     = x[r] * rsqrt(mean(x[r]^2) + eps)
//   gate(r)   = Wdown @ silu(Wup @ xn[r])
//   out[r][i] = ZeroCenteredGatedNormElement(xn[r][i], w[i], gate(r)[i])
// `w` BF16 [dim]; `w_up` BF16 [rank, dim]; `w_down` BF16 [dim, rank]; F32
// compute. `rank` is the checkpoint's gate rank (GcGatedNorm::rank), not the
// frozen constant: the kernel must index by the actual rank or it reads past
// the gate_up/gate_down buffers.
// `w_offset` selects the norm variant: 1.0 for GigaChat3.5's zero-centred
// scale (`1+w`), 0.0 for GigaChat4's GatedNorm (scale `w`).
cudaError_t RunGigaChatGatedNorm(const float *x, int rows, int dim, int rank,
                                 const uint8_t *w, const uint8_t *w_up,
                                 const uint8_t *w_down, float eps,
                                 float w_offset, float *out,
                                 cudaStream_t stream);

// Legacy 3.5 spelling: zero-centred (`1+w`), bit-identical to
// RunGigaChatGatedNorm(..., w_offset = 1.0f).
cudaError_t RunZeroCenteredGatedNorm(const float *x, int rows, int dim,
                                     int rank, const uint8_t *w,
                                     const uint8_t *w_up,
                                     const uint8_t *w_down, float eps,
                                     float *out, cudaStream_t stream);

// GatedDeltaNet output norm (zero-centered sigmoid gated RMSNorm), one block
// per head:
//   xn = rmsnorm(o[h]) ; out[h][d] = GdnOutputNormElement(xn[d], w[d],
//   z[h][d], eps, scale)
// `w` BF16 [head_dim]; `z` is the per-head gate [heads, head_dim].
// `w_offset` as for RunGigaChatGatedNorm: 1.0 zero-centred (3.5),
// 0.0 plain scale (GigaChat4).
cudaError_t RunGdnOutputNormOffset(const float *o, const float *z,
                                   const uint8_t *w, int heads, int head_dim,
                                   float eps, float scale, float w_offset,
                                   float *out, cudaStream_t stream);

// Legacy 3.5 spelling: zero-centred (`1+w`).
cudaError_t RunGdnOutputNorm(const float *o, const float *z, const uint8_t *w,
                             int heads, int head_dim, float eps, float scale,
                             float *out, cudaStream_t stream);

// Sigmoid router for one token:
//   logits = x . W^T (F32); scores = sigmoid(logits);
//   choice = scores + e_score_correction_bias;
//   idx   = top-k(choice) (lowest index wins ties);
//   w     = scores[idx]; w /= sum + 1e-20; w *= routed_scaling_factor.
// `w` BF16 [num_experts, hidden]; `bias` F32 [num_experts] (may be null).
// Deterministic: one block, thread-0 sequential selection, no atomics.
cudaError_t RunGigaChatRouterTopK(const float *x, const uint8_t *w,
                                  const float *bias, int num_experts, int hidden,
                                  int top_k, float routed_scaling_factor,
                                  int *out_idx, float *out_weights,
                                  cudaStream_t stream);

// Final logit soft cap in place: `x = cap * tanh(x / cap)` (F32, `cap <= 0`
// is the identity). Applied before sampling and before final_logits is read
// back, so every gate compares the quantity the served model produces.
cudaError_t RunGigaChatSoftcap(float *logits, int n, float cap,
                               cudaStream_t stream);

// SwiGLU with hard limit: `mid = silu(clamp(g,max=L)) * clamp(u,-L,L)`.
cudaError_t RunSwigluLimit(const float *gate, const float *up, float limit,
                           int n, float *mid, cudaStream_t stream);

// Gated attention: `y[i] *= sigmoid(gate[i])`.
cudaError_t RunSigmoidGate(float *y, const float *gate, int n,
                           cudaStream_t stream);

} // namespace lsxcommon::gigachat
