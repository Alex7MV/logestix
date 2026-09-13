// deepseek_moe_v41_inference.h — DeepSeek-V4.1-Flash device kernel launchers.
// FP16 paths (uint16_t bit patterns to keep host TUs cuda_fp16-free) and FP32
// Hyper-Connection paths (the hc residual stream is carried in F32; norm/rope
// bridges cast at the boundary).
#pragma once

#include <cstdint>
#include <cublas_v2.h>
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

// out[i] = silu(min(gate[i], limit)) * clamp(up[i], -limit, limit). limit<=0
// disables the clamps. F32 vectors of length n.
cudaError_t LaunchExpertSwiglu(const float *gate, const float *up, float *out,
                               int n, float limit, cudaStream_t stream);

// Row-major FP16 dense GEMM: C[M,N] = A[M,K] * B[K,N]. F32 accumulation.
cudaError_t LaunchGemmFp16(cublasHandle_t handle, cudaStream_t stream,
                           const uint16_t *A, const uint16_t *B, uint16_t *C,
                           int M, int N, int K);

// Head logits (F32): logits[V] = W[V,H] * x[H], FP16 inputs, F32 accumulation.
cudaError_t LaunchHeadLogits(cublasHandle_t handle, cudaStream_t stream,
                             const uint16_t *w, const uint16_t *x, float *logits,
                             int vocab, int hidden);

// Gathers `n` embedding rows (FP16) by token id into out [n, hidden].
cudaError_t LaunchEmbedLookup(const uint16_t *embed, int vocab, int hidden,
                              const int32_t *ids, int n, uint16_t *out,
                              cudaStream_t stream);

// Expands one hidden vector (FP16) into `hc` identical F32 copies
// (model.py h.unsqueeze(2).repeat): out[c*dim + d] = x[d].
cudaError_t LaunchHcExpand(const uint16_t *x, int dim, int hc, float *out,
                           cudaStream_t stream);

// One decode step of DeepSeek-V4.1 MLA attention for a pure-SWA layer
// (compress_ratio == 0): q_lora -> q -> RoPE; wkv -> norm -> RoPE -> FP8
// quant; window ring + sparse attn + attn_sink; un-RoPE; grouped wo_a -> wo_b.
// All weights/shapes are explicit so callers can test with synthetic tensors.
// rope_cos/rope_sin point at the single position's [rope_dim/2] row.
cudaError_t AttentionForwardDecode(
    cublasHandle_t handle, cudaStream_t stream, const uint16_t *x, int hidden,
    const uint16_t *wq_a, int q_lora, const uint16_t *q_norm,
    const uint16_t *wq_b, int heads, int head_dim, int rope_dim,
    const uint16_t *wkv, const uint16_t *kv_norm, const float *attn_sink,
    const uint16_t *wo_a, int o_groups, int o_lora_rank, const uint16_t *wo_b,
    const float *rope_cos, const float *rope_sin, uint16_t *kv_cache,
    int window, int start_pos, uint16_t *out);

} // namespace deepseek_v41
} // namespace lsxcommon
