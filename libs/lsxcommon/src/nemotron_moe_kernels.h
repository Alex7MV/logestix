#pragma once

// Device kernels for the Nemotron-3.5 Mamba-2 mixer. Host-callable launchers;
// math pinned by src/nemotron_moe_device_math.h (host reference) and
// transformers 5.15.1 modeling_nemotron_h.py.

#include <cuda_runtime.h>

namespace lsxcommon::nemotron_moe {

// Single-token depthwise causal conv + silu (groups == conv_dim, kernel K).
//   xBC[c]               current conv input              [conv_dim]
//   w[c*K + k]           depthwise weight                [conv_dim*K]
//   bias[c]              optional depthwise bias
//   state_in[c*(K-1)+k]  previous K-1 inputs (oldest first)
//   state_out            shifted state after the step
//   out[c]               silu(conv)                       [conv_dim]
void RunNemotronConvStep(const float *xBC, const float *w, const float *bias,
                         const float *state_in, float *state_out, float *out,
                         int conv_dim, int K, cudaStream_t stream);

// One decode step of the Mamba-2 selective state-space recurrence.
// Layouts as in ref::SelectiveStateUpdate (state [heads, head_dim, state_size]).
void RunNemotronSelectiveStateStep(const float *state_in, const float *xs,
                                   const float *B, const float *C,
                                   const float *D, const float *dt,
                                   const float *dt_bias, const float *A_log,
                                   int heads, int head_dim, int groups,
                                   int state_size, float *state_out, float *out,
                                   cudaStream_t stream);

// Gated RMSNorm: h = x * silu(gate); per-group rmsnorm; * weight.
void RunNemotronGatedRmsNorm(const float *x, const float *gate,
                             const float *weight, int n, int group_size,
                             float eps, float *out, cudaStream_t stream);

// Full-dimension RoPE (HF rotate_half convention) applied in place to
// x[n_heads, head_dim] at position `pos`, base `theta`.
void RunNemotronRope(float *x, int n_heads, int head_dim, int pos, float theta,
                     cudaStream_t stream);

// Single-token GQA decode attention.
//   q       [heads, head_dim]           (already RoPE-rotated)
//   k_cache [kv_heads, seq_len, head_dim]
//   v_cache [kv_heads, seq_len, head_dim]
//   out     [heads, head_dim]
void RunNemotronAttnDecode(const float *q, const float *k_cache,
                           const float *v_cache, int heads, int kv_heads,
                           int head_dim, int seq_len, float scale, float *out,
                           cudaStream_t stream);

} // namespace lsxcommon::nemotron_moe
