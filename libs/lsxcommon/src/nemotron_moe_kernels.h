#pragma once

// Device kernels for the Nemotron-3.5 Mamba-2 mixer. Host-callable launchers;
// math pinned by src/nemotron_moe_device_math.h (host reference) and
// transformers 5.15.1 modeling_nemotron_h.py.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "nemotron_moe_constants.h"

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
// `cache_stride` is the KV cache row stride (max_seq).
void RunNemotronAttnDecode(const float *q, const float *k_cache,
                           const float *v_cache, int heads, int kv_heads,
                           int head_dim, int seq_len, int cache_stride,
                           float scale, float *out, cudaStream_t stream);

// Sigmoid top-k router with e_score_correction_bias (n_group == 1 in this
// checkpoint, so the group stage is the identity). Bias selects only.
//   logits [E] F32; weights [k]; indices [k].
void RunNemotronRouter(const float *logits, const float *e_score_bias, int E,
                       int k, float scale, float *weights, int *indices,
                       cudaStream_t stream);

// relu2 activation: out[i] = max(x[i],0)^2.
void RunNemotronRelu2(const float *x, int n, float *out, cudaStream_t stream);

// In-place FP16 relu2 (applied to NVFP4 GEMV outputs to skip the F32
// round-trip). Bit-identical to float relu2 then round-to-half.
void RunNemotronRelu2Half(__half *x, int n, cudaStream_t stream);

// Routed-expert accumulation over a batched down-GEMV output [count, n]:
//   acc[j] = (add ? acc[j] : 0) + Σ_e w[e] * float(y[e*n + j])
void RunNemotronMoeWeightedSum(const __half *y, const float *weights, int count,
                               int n, float *acc, bool add,
                               cudaStream_t stream);

// acc[j] += float(x[j]); used to add the ungated shared expert.
void RunNemotronAddHalfToF32(const __half *x, int n, float *acc,
                             cudaStream_t stream);

} // namespace lsxcommon::nemotron_moe
