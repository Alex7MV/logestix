#pragma once

// Mistral-Large-4-1T-A52B family kernels (Model Ingestion Protocol stage 5):
// LeRoPE table construction and application, per-head position-dependent query
// scaling (ssmax), the FP32 router, the DSA indexer scoring + exact top-k, and
// the MLA absorption helpers. Model-neutral primitives (NVFP4 GEMV, RMSNorm,
// the routed-expert FFN) are reused from lsxcommon.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "lsxcommon/lsx_moe_inference.h"

namespace lsxcommon {
namespace mistral_large4 {

// ── RoPE / LeRoPE ─────────────────────────────────────────────────────────
// YaRN (NTK-by-parts, no magnitude scaling) over the original window, with the
// learned per-frequency LeRoPE scale: inv_freq[j] *= exp(alpha[j]).
// Tables are [capacity][rope_dim/2] F32, cos then sin.
bool BuildLarge4RopeTables(int capacity, const std::vector<float> &alpha,
                           int rope_dim, double theta, double factor,
                           double orig, double beta_fast, double beta_slow,
                           std::vector<float> &cos_t, std::vector<float> &sin_t);

cudaError_t UploadLarge4RopeTables(cudaStream_t stream,
                                   const std::vector<float> &cos_t,
                                   const std::vector<float> &sin_t,
                                   float **d_cos, float **d_sin);

// Interleaved-pair rotation of the LAST `rope_dim` lanes of every head of
// @p vec ([heads][head_dim] F32, lane offset `nope`).
cudaError_t LaunchLarge4RopeInterleaved(float *vec, int heads, int head_dim,
                                        int nope, int rope_dim,
                                        const float *d_cos, const float *d_sin,
                                        int pos, cudaStream_t stream);

// Single-vector variant (the decoupled k_pe / indexer key): [rope_dim] F32.
cudaError_t LaunchLarge4RopeInterleavedVec(float *vec, int rope_dim,
                                           const float *d_cos,
                                           const float *d_sin, int pos,
                                           cudaStream_t stream);

// q[head][d] *= bf16(s[head]) * bf16(log(pos+1)) + bf16(b[head]).
cudaError_t LaunchLarge4SsmaxScale(float *q, int heads, int head_dim,
                                   const __nv_bfloat16 *s,
                                   const __nv_bfloat16 *b, int pos,
                                   cudaStream_t stream);

// Plain RMSNorm (weight multiply, no 1+w offset) over an F32 input:
// out = bf16(x/rms * w).
cudaError_t LaunchLarge4RmsNorm(const float *x, const __nv_bfloat16 *w, int n,
                                float eps, __nv_bfloat16 *out,
                                cudaStream_t stream);

// acc += scale * src (scale read on device; src FP16).
cudaError_t LaunchLarge4AccAddScaledF16ToF32(float *acc, const __half *src,
                                             const float *d_scale, int n,
                                             cudaStream_t stream);

// z = tanh(z / cap) * cap (final logit softcapping).
cudaError_t LaunchLarge4LogitSoftcap(float *logits, int n, float cap,
                                     cudaStream_t stream);

// ── LayerNorm (indexer q_input_norm / k_norm: weight + bias) ──────────────
cudaError_t LaunchLarge4LayerNorm(const __nv_bfloat16 *x,
                                  const __nv_bfloat16 *w,
                                  const __nv_bfloat16 *b, int n, float eps,
                                  float *out, cudaStream_t stream);

// ── Router (softmax over all experts, top-k, renormalize) ─────────────────
cudaError_t LaunchLarge4Router(const __nv_bfloat16 *router_w, const float *x,
                               int experts, int top_k, int hidden,
                               moe::MoERoutedSlot *d_slots,
                               cudaStream_t stream);

// ── Indexer ───────────────────────────────────────────────────────────────
// index_scores[t] = sum_h w[h] * relu(dot(q[h], k[t]) * scale) for t in [0,n).
// q: [heads][head_dim] F32, k: [n][head_dim] BF16, w: [heads] F32.
cudaError_t LaunchLarge4IndexerScores(const float *q, const __nv_bfloat16 *k,
                                      const float *weights, int heads,
                                      int head_dim, float scale, int n,
                                      float *scores, cudaStream_t stream);

// Exact top-k over @p n scores into @p out_idx (ascending), writing the count
// to @p d_count. When n <= k every index is selected (identity mask).
cudaError_t LaunchLarge4TopkSelect(const float *scores, int n, int k,
                                   int32_t *out_idx, int *d_count,
                                   cudaStream_t stream);

// ── MLA absorption ────────────────────────────────────────────────────────
// q_abs[h][k] = sum_d q_nope[h][d] * w_k[h][d][k], where w_k[h][d][:] is row
// h*(kn+v)+d of the [heads*(kn+v), kv_lora] wkv_b. q_nope is [heads][qk_head]
// F32 read at lane offset 0 (the nope slice is the first `kn` lanes).
cudaError_t LaunchLarge4AbsorbQuery(const __nv_bfloat16 *wkv_b,
                                    const float *q, int heads, int qk_head,
                                    int kn, int v_head, int kv_lora,
                                    float *q_abs, cudaStream_t stream);

// out[h][:] = w_v[h] (v_head x kv_lora) @ ctx[h]; rows h*(kn+v)+kn .. +v of
// wkv_b. out is [heads][v_head] F32.
cudaError_t LaunchLarge4ValueProject(const __nv_bfloat16 *wkv_b,
                                     const float *ctx, int heads, int kn,
                                     int v_head, int kv_lora, float *out,
                                     cudaStream_t stream);

// ── elementwise helpers ───────────────────────────────────────────────────
cudaError_t LaunchLarge4Bf16ToF32(const __nv_bfloat16 *src, float *dst,
                                  int n, cudaStream_t stream);
cudaError_t LaunchLarge4Bf16ToF16(const __nv_bfloat16 *src, __half *dst,
                                  int n, cudaStream_t stream);
cudaError_t LaunchLarge4F16ToF32(const __half *src, float *dst, int n,
                                 cudaStream_t stream);
cudaError_t LaunchLarge4F32ToF16(const float *src, __half *dst, int n,
                                 cudaStream_t stream);
cudaError_t LaunchLarge4F32ToBf16(const float *src, __nv_bfloat16 *dst, int n,
                                  cudaStream_t stream);
cudaError_t LaunchLarge4AddInPlace(float *a, const float *b, int n,
                                   cudaStream_t stream);

} // namespace mistral_large4
} // namespace lsxcommon
