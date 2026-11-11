#pragma once

// Device kernels for the Nemotron-3-Ultra-550B-A55B decode path.
//
// Host-callable launchers; the math is pinned by
// src/nemotron_moe_ultra_device_math.h (host reference) and by the checkpoint's
// own reference implementation (transformers 5.15.1 modeling_nemotron_h.py +
// modeling_zamba2.py). Every width is an argument: the same launchers serve the
// 8192-hidden / 256-head / 512-expert production geometry and the tiny
// synthetic fixtures, and no kernel reads a passport constant at device scope.
//
// Shared-memory sandbox: reductions use a shared array behind __syncthreads();
// nothing here uses atomics, and nothing allocates more than one block's worth
// of shared memory (Blackwell CC 12.x: 48 warps/SM, 100 KB smem/SM).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lsxcommon::nemotron_ultra {

// Single-token depthwise causal conv + silu (groups == conv_dim, kernel K).
//   xBC[c]               current conv input              [conv_dim]
//   w[c*K + k]           depthwise weight                [conv_dim*K]
//   bias[c]              optional depthwise bias (may be null)
//   state_in[c*(K-1)+k]  previous K-1 inputs (oldest first)
//   state_out            shifted state after the step
//   out[c]               silu(conv)                       [conv_dim]
void RunUltraConvStep(const float *xBC, const float *w, const float *bias,
                      const float *state_in, float *state_out, float *out,
                      int conv_dim, int K, cudaStream_t stream);

// One step of the Mamba-2 selective state-space recurrence.
// `clamp_dt` = 1 for the prefill form (chunk scan clamps dt to time_step_min),
// 0 for the decode form (selective update: softplus only) — exactly the split in
// the reference (modeling_zamba2.py L478-534 vs L540-562).
void RunUltraSelectiveStateStep(const float *state_in, const float *xs,
                                const float *B, const float *C, const float *D,
                                const float *dt, const float *dt_bias,
                                const float *A_log, int heads, int head_dim,
                                int groups, int state_size, int clamp_dt,
                                float *state_out, float *out, cudaStream_t stream);

// Gated RMSNorm: h = x * silu(gate); per-group rmsnorm; * weight.
// Supports group_size > 1024 via a strided loop (production group is 2048).
void RunUltraGatedRmsNorm(const float *x, const float *gate, const float *weight,
                          int n, int group_size, float eps, float *out,
                          cudaStream_t stream);

// Single-token GQA decode attention over the FP32 KV cache. The full-attention
// layers are NoPE: no rotary is applied and no rotary kernel exists.
void RunUltraAttnDecode(const float *q, const float *k_cache,
                        const float *v_cache, int heads, int kv_heads,
                        int head_dim, int seq_len, int cache_stride, float scale,
                        float *out, cudaStream_t stream);

// Grouped sigmoid top-k router:
//   scores = sigmoid(logits); choice = scores + e_score_correction_bias;
//   group_scores = sum of the top-2 choice per group; keep `topk_group` groups;
//   top-k inside the survivors; normalize (norm_topk_prob) and scale.
// Bias selects, never scales. Ties resolve to the lowest expert index.
//   logits [E] F32; weights [k]; indices [k].
void RunUltraRouter(const float *logits, const float *e_score_bias, int E, int k,
                    int n_group, int topk_group, float scale, float *weights,
                    int *indices, cudaStream_t stream);

// NOTE: there is no F32 relu2 launcher here on purpose. The routed and shared
// experts are relu2'd in place on their FP16 GEMV output (RunUltraRelu2Half),
// so an F32 variant would be dead surface — the 30B sibling carries one and only
// its kernel test ever calls it.

// In-place FP16 relu2 (applied to NVFP4 GEMV outputs to skip the F32 round-trip).
// Bit-identical to float relu2 then round-to-half.
void RunUltraRelu2Half(__half *x, int n, cudaStream_t stream);

// Routed-expert accumulation over a batched down-GEMV output [count, n]:
//   acc[j] = (add ? acc[j] : 0) + Σ_e w[e] * float(y[e*n + j])
void RunUltraMoeWeightedSum(const __half *y, const float *weights, int count, int n,
                            float *acc, bool add, cudaStream_t stream);

// acc[j] += float(x[j]); used to add the ungated shared expert.
void RunUltraAddHalfToF32(const __half *x, int n, float *acc, cudaStream_t stream);

} // namespace lsxcommon::nemotron_ultra
