// hy_moe_flash_kernels.h — Hy4-flash family kernels.
//
// Everything here is a one-token (decode) primitive. Matrix products route
// through the model-neutral NVFP4 layer (`lsx_nvfp4`); the kernels below are the
// operations that layer does not provide:
//
//   * iHC (identity Hyper-Connections): the 4-stream mix / pre-mixdown / post
//     update / head collapse, all fp32 like the reference;
//   * the two RoPE regimes (theta 1e7 on sparse layers, 1e4 on sliding ones);
//   * the router (fp32 sigmoid scores, selection-only bias, top-k, renorm,
//     routed scaling);
//   * MLA attention: the transposed qk projection (score = (kv_b_nope^T q) .
//     latent — the only way to read kv_b once per token instead of once per
//     visible position), the sink softmax, and the value projection against the
//     softmax-weighted latent sum;
//   * IndexPool: token keys, gated 4-token pooling, group scoring and selection;
//   * the small glue (Swiglu with clamps, scaled accumulate, casts, embed
//     lookup, fp16 head logits).
//
// Determinism: every reduction has a fixed iteration order (no atomics on any
// path that feeds final_logits), so kOff and kAuto stay bit-reproducible.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "lsxcommon/lsx_inference_nvfp4.h"

namespace lsxcommon {
namespace hy4 {
namespace gpu {

// ── iHC ────────────────────────────────────────────────────────────────────
// Streams layout: x[streams][hidden], fp32.
//   rsqrt   = rsqrt(mean(flat(x)^2) + eps)
//   mix[r]  = (fn[r] . flat(x)) * rsqrt                                  r < 2S
//   pre[j]  = sigmoid(mix[j] * scale[0] + base[j]) + eps
//   post[j] = magnitude * sigmoid(mix[S+j] * scale[1] + base[S+j]) + eps
//   agg[i]  = sum_j pre[j] * x[j][i]
cudaError_t RunIhcMix(const float *x, const float *fn, const float *base,
                      const float *scale, int streams, int hidden, float eps,
                      float magnitude, float *agg, float *pre, float *post,
                      float *mix, float *rsqrt, cudaStream_t stream);

// streams[j][i] = post[j] * agg[i] + streams[j][i]   (in place over streams)
cudaError_t RunIhcPost(const float *agg, const float *post, int n_streams,
                       int hidden, float *stream_buf, cudaStream_t stream);

// Head collapse: agg[i] = sum_j pre[j] * x[j][i] with the head's own fn/base/
// scale (fn is [streams, streams*hidden], scale is a single value).
cudaError_t RunIhcHead(const float *x, const float *fn, const float *base,
                       const float *scale, int streams, int hidden, float eps,
                       float *agg, float *pre, float *mix, float *rsqrt,
                       cudaStream_t stream);

// ── Casts ──────────────────────────────────────────────────────────────────
cudaError_t RunCastF32ToHalf(const float *src, __half *dst, int n,
                             cudaStream_t stream);
cudaError_t RunCastHalfToF32(const __half *src, float *dst, int n,
                             cudaStream_t stream);

// ── RoPE (rotate-half over the LAST rope_dim lanes of each row) ────────────
cudaError_t RunRopeHalf(__half *x, int rows, int row_len, int rope_dim,
                        const float *cos_row, const float *sin_row,
                        bool inverse, cudaStream_t stream);

// ── Router ─────────────────────────────────────────────────────────────────
// scores[j] = sigmoid(gate[j] . x)                 (fp32, gate is BF16->FP16)
// choice[j] = scores[j] + bias[j]                  (bias selects, never scales)
// top-k of choice, ties -> lower index; weights = scores[idx] renormalized,
// then multiplied by route_scale.
cudaError_t RunRouterLogits(const __half *x, const __half *gate, int n_routed,
                            int hidden, float *scores, cudaStream_t stream);
cudaError_t RunRouterTopK(const float *scores, const float *bias, int n_routed,
                          int topk, float route_scale, int32_t *out_idx,
                          float *out_w, cudaStream_t stream);

// ── Expert glue ────────────────────────────────────────────────────────────
cudaError_t RunSwigluClamp(const __half *gate, const __half *up, __half *out,
                           int n, float limit, cudaStream_t stream);
// dst[i] += scale * src[i]   (fp32 accumulate into the routed sum)
cudaError_t RunAccumulateScaled(const __half *src, float *dst, int n,
                                float scale, cudaStream_t stream);
cudaError_t RunGatedScale(__half *x, const __half *gate, int n,
                          cudaStream_t stream);

// ── MLA attention ──────────────────────────────────────────────────────────
// qk[h][d] = sum_r kv_b_nope[h*nope + r][d] * q[h*q_stride + r]
// (the transposed projection: kv_lora outputs per head from the nope rows).
// `q_stride` is the FULL query row width (nope + rope), because the query buffer
// is [heads][qk_head] with the nope lanes first — passing the nope count here
// reads every head after the first from the wrong offset.
cudaError_t RunMlaQkProject(const __half *q, const lsx_nvfp4::LsxNvfp4Weight &w,
                            int heads, int nope, int q_stride, float *qk,
                            cudaStream_t stream);

// Sink softmax over the visible latent rows. For each head h:
//   s[t] = scale * (qk[h] . latent[t][0..kv_lora) + q_rope[h] . latent[t][kv_lora..))
//   m    = max(s), denom = sum_t exp(s[t]-m) + exp(sink[h]-m)
//   acc[h][d] = sum_t exp(s[t]-m)/denom * latent[t][d]      (d < kv_lora)
// `latent` is the MLA cache: [capacity][kv_lora + rope] fp16.
cudaError_t RunMlaSinkSoftmax(const float *qk, const __half *q_all,
                              int q_stride, int q_rope_off,
                              const __half *latent, int64_t latent_stride,
                              const int32_t *pos, int npos, int heads,
                              int kv_lora, int rope, const float *sink,
                              float scale, float *scores, float *acc,
                              cudaStream_t stream);

// o[h][i] = sum_d w[h*v_head + i][d] * acc[h][d]        (NVFP4 value rows)
cudaError_t RunMlaValueProject(const float *acc,
                               const lsx_nvfp4::LsxNvfp4Weight &w, int heads,
                               int v_head, int kv_lora, __half *out,
                               cudaStream_t stream);

// Visible-position list for a sliding-window layer: [max(0,p-window+1), p].
cudaError_t RunWindowPositions(int32_t *pos, int window, int p,
                               cudaStream_t stream);

// ── IndexPool (sparse layers) ──────────────────────────────────────────────
// Weighted RMSNorm of one row, in place (token_key_norm / key_norm).
cudaError_t RunRmsNormHalf(__half *x, const __half *w, int dim, float eps,
                           cudaStream_t stream);

// Gated pooling of one COMPLETE group of `ratio` token keys:
//   u_i = k_i + ape[i];  v_i = W_kv u_i;  s_i = sigmoid(W_gate u_i)
//   p   = key_norm(sum_i s_i * v_i)
// W_kv and W_gate are fp32 [dim][dim], ape is fp32 [ratio][dim].
cudaError_t RunIndexPoolGroup(const __half *keys, const float *ape,
                              const float *w_kv, const float *w_gate,
                              const float *norm, int ratio, int dim,
                              __half *out, cudaStream_t stream);

// score[g] = sum_h w[h] * relu((qk[h] . p[g]) * scale_hd * scale_heads)
// where w[h] = (head_weight_projection . x)[h] * n_heads^-0.5 * head_dim^-0.5
cudaError_t RunIndexGroupScores(const float *qk, const float *w_heads,
                                const __half *pooled, int n_groups, int heads,
                                int dim, float head_scale, float *scores,
                                cudaStream_t stream);

// Keeps the `topk` best groups (ties -> lower index) and expands them to token
// positions in ascending order; the tail is padded with -1.
cudaError_t RunIndexSelectGroups(const float *scores, int n_complete, int ratio,
                                 int topk, int tail_start, int n_tail,
                                 int32_t *out, int n_pad, unsigned char *used,
                                 cudaStream_t stream);

// ── Vocabulary ─────────────────────────────────────────────────────────────
cudaError_t RunEmbedLookup(const __half *embed, int vocab, int hidden,
                           const int32_t *tok, int n, __half *out,
                           cudaStream_t stream);
// logits[v] = head[v] . x   (fp32 accumulate; the head stays fp16)
cudaError_t RunHeadLogits(const __half *head, const __half *x, int vocab,
                          int hidden, float *logits, cudaStream_t stream);
// argmax + max value of a fp32 vector, one block (deterministic tie -> lower).
cudaError_t RunArgmax(const float *x, int n, int32_t *out_idx, float *out_val,
                      cudaStream_t stream);

} // namespace gpu
} // namespace hy4
} // namespace lsxcommon
