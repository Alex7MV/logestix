#pragma once

// Kimi-K3 device kernels: SiTU-GLU, the sigmoid+bias 896-way router, the
// AttnRes depth mixer, KDA (per-channel decay, lower-bound gate, conv-4),
// absorbed gated NoPE MLA decode, and the small elementwise glue.
// Projections go through the neutral NVFP4 layer, not here.
//
// Deterministic: fixed reduction geometry, thread-0 only RMW behind barriers,
// no atomicAdd anywhere on the logits path.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace lsxcommon::kimi_k3 {

// row-wise RMSNorm (weight-form, half weight), rows x n.
cudaError_t RunKimiK3RmsNorm(cudaStream_t stream, const __half *x,
                             const __half *w, __half *out, int rows, int n,
                             float eps);

// Gated RMSNorm over the last `n` dims (KDA o_norm): out = rms(x) * w * sigmoid(gate).
cudaError_t RunKimiK3GatedRmsNorm(cudaStream_t stream, const __half *x,
                                  const float *w, const __half *gate,
                                  __half *out, int rows, int n, float eps);

// SiTU-GLU: beta*tanh(g/beta)*sigmoid(g) * (linear_beta*tanh(u/linear_beta)).
cudaError_t RunKimiK3Situ(cudaStream_t stream, const __half *gate,
                          const __half *up, __half *out, int64_t n,
                          float beta, float linear_beta);

// Router scores: sel[e] = sigmoid(x.w[e]) + bias[e], raw[e] = sigmoid(...).
cudaError_t RunKimiK3RouterScores(cudaStream_t stream, const __half *x,
                                  const __half *w, const float *bias, int H,
                                  int E, float *sel, float *raw);

// Deterministic top-K (by sel desc, index asc) + renormalized weights.
cudaError_t RunKimiK3RouterTopK(cudaStream_t stream, const float *sel,
                                const float *raw, int E, int K, float scale,
                                int renorm, int32_t *idx, float *wts);

// AttnRes mixer over n_src sources (each `hidden` wide).
cudaError_t RunKimiK3DepthMix(cudaStream_t stream, const __half *sources,
                              int n_src, const __half *norm_w,
                              const __half *proj_w, __half *out, int hidden,
                              float eps);

// KDA short conv (seq==1): out = silu(depthwise_conv(hist + x)); hist updated.
cudaError_t RunKimiK3KdaConv(cudaStream_t stream, const __half *x,
                             const float *w, float *hist, __half *out, int C,
                             int k);

// One KDA recurrence step. state: [H][KD][VD] FP32, carried across steps.
cudaError_t RunKimiK3KdaRecur(cudaStream_t stream, const __half *q,
                              const __half *k, const __half *v,
                              const __half *g, const __half *beta,
                              const float *a_log, const float *dt_bias,
                              float *state, __half *out, int H, int KL,
                              int KD, int VD, float lower_bound);

// Absorbed MLA: q_abs[h] = W_k[h]^T q_nope[h] (W_k = kv_b rows [h*2D, h*2D+D)).
cudaError_t RunKimiK3MlaAbsorbQ(cudaStream_t stream, const __half *q_nope,
                                const uint8_t *wk_packed,
                                const uint8_t *wk_scales, float wk_gs,
                                __half *q_abs, int heads, int nope, int latent,
                                int rows_per_head, int row_offset,
                                int q_head_stride, int wk_row_stride_packed,
                                int wk_row_stride_scales);

// Latent-cache decode attention + absorbed value: writes out[h][v_head] then
// the caller applies the output gate and o_proj.
//   latent[t] = [norm_latent(kv_lora) | k_rot(rope)] contiguous, stride 576.
cudaError_t RunKimiK3MlaAttention(cudaStream_t stream, const __half *latent,
                                  int n_pos, const __half *q_abs,
                                  const __half *q_rot,
                                  const uint8_t *wv_packed,
                                  const uint8_t *wv_scales, float wv_gs,
                                  __half *out, int heads, int latent_dim,
                                  int rope, int v_head, int rows_per_head,
                                  int row_offset, int q_rot_stride, float scale,
                                  int wv_row_stride_packed,
                                  int wv_row_stride_scales);

// out[i] = x[i] * sigmoid(gate[i]).
cudaError_t RunKimiK3SigmoidMul(cudaStream_t stream, __half *x,
                                const __half *gate, int64_t n);

// acc[i] += scale * add[i].
cudaError_t RunKimiK3Axpy(cudaStream_t stream, __half *acc, const __half *add,
                          float scale, int64_t n);

// Dequantizes one NVFP4 row (embedding lookup) into `out` (cols halves).
cudaError_t RunKimiK3DequantRow(cudaStream_t stream, const uint8_t *packed,
                                const uint8_t *scales, float global_scale,
                                __half *out, int cols);

}  // namespace lsxcommon::kimi_k3
