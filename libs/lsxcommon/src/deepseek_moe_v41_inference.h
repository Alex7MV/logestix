// deepseek_moe_v41_inference.h — DeepSeek-V4.1-Flash device kernel launchers.
// FP16 paths (uint16_t bit patterns to keep host TUs cuda_fp16-free) and FP32
// Hyper-Connection paths (the hc residual stream is carried in F32; norm/rope
// bridges cast at the boundary).
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lsxcommon {
namespace deepseek_v41 {

// RMSNorm (F32 compute, eps = 1e-20 for V4.1): out = gamma*x*rsqrt(mean(x^2)+eps).
cudaError_t LaunchRmsNormHalf(const uint16_t *x, const uint16_t *gamma,
                              uint16_t *out, int n, float eps,
                              cudaStream_t stream);

// Rotates the last `rope_dim` elements of each of `rows` row-vectors in place.
cudaError_t LaunchRopeHalf(uint16_t *x, int rows, int row_len, int rope_dim,
                           const float *cos_row, const float *sin_row,
                           bool inverse, cudaStream_t stream);

// Hyper-Connections: mixes = (hc_fn @ x) * rsqrt(mean(x^2)+eps), then
// split+Sinkhorn into pre/post/comb. x: [hc*dim] F32.
cudaError_t LaunchHcMixes(const float *x, const float *hc_fn,
                          const float *hc_scale, const float *hc_base, int hc,
                          int dim, int sinkhorn_iters, float eps, float *pre,
                          float *post, float *comb, cudaStream_t stream);

// out[d] = sum_c pre[c] * x[c*dim + d]. x: [hc*dim] F32.
cudaError_t LaunchHcPre(const float *x, const float *pre_mix, float *out,
                        int hc, int dim, cudaStream_t stream);

// out[c][d] = post[c]*x[d] + sum_cp comb[cp*hc+c]*residual[cp][d].
cudaError_t LaunchHcPost(const float *x, const float *residual,
                         const float *post, const float *comb, float *out,
                         int hc, int dim, cudaStream_t stream);

// Sparse attention over gathered KV indices with a learnable attn_sink.
// q: [q_rows, heads, d] FP16; kv: [kv_rows, d] FP16; sink: [heads] F32 (may be
// null); idxs: [q_rows, topk] INT32 (-1 = invalid); out: [q_rows, heads, d].
cudaError_t LaunchSparseAttnHalf(const uint16_t *q, int q_rows, int heads,
                                 int d, const uint16_t *kv, int kv_rows,
                                 const float *sink, const int32_t *idxs,
                                 int topk, uint16_t *out, float softmax_scale,
                                 cudaStream_t stream);

// Softmax-pools `n_groups` groups of `ratio` consecutive tokens starting at
// `first_token` from kv/score [tokens, d] F32 into out [n_groups, d] F32.
cudaError_t LaunchCompressorPool(const float *kv, const float *score,
                                 int n_groups, int ratio, int d,
                                 int first_token, float *out,
                                 cudaStream_t stream);

// indexer score[t] = sum_h relu(q[h] . k[t]) * w[h]. q: [heads, dim] FP16;
// k: [n_pos, dim] FP16; w: [heads] F32; out: [n_pos] F32.
cudaError_t LaunchIndexerScores(const uint16_t *q, int heads, int dim,
                                const uint16_t *k, int n_pos, const float *w,
                                float *out, cudaStream_t stream);

} // namespace deepseek_v41
} // namespace lsxcommon
