// deepseek_moe_v41_inference.h — DeepSeek-V4.1-Flash device kernel launchers.
// FP16 paths (uint16_t bit patterns to keep host TUs cuda_fp16-free) and FP32
// Hyper-Connection paths (the hc residual stream is carried in F32; norm/rope
// bridges cast at the boundary).
#pragma once

#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "deepseek_moe_v41_inference_csa.h"

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

// Matrix-vector with F32 output: y[M] = W[M,K] * x[K] (FP16 inputs).
cudaError_t LaunchGemvFp16ToF32(cublasHandle_t handle, cudaStream_t stream,
                                const uint16_t *w, const uint16_t *x, float *y,
                                int M, int K);

// Elementwise F32 -> FP16 (RNE).
cudaError_t LaunchCastF32ToHalf(const float *src, uint16_t *dst, int n,
                                cudaStream_t stream);

// Elementwise FP16 -> F32.
cudaError_t LaunchCastHalfToF32(const uint16_t *src, float *dst, int n,
                                cudaStream_t stream);

// fill dst[0..n) with `value`.
cudaError_t LaunchFillF32(float *dst, int n, float value,
                          cudaStream_t stream);

// acc[i] += half_to_float(src[i]).
cudaError_t LaunchAccumulateHalfToF32(const uint16_t *src, float *acc, int n,
                                      cudaStream_t stream);

// Shared expert SwiGLU FFN: gate=w1@x, up=w3@x (F32), silu(clamped)*clamped,
// then out=w2@mid (FP16). Scratch is allocated internally.
cudaError_t SharedExpertForward(cublasHandle_t handle, cudaStream_t stream,
                                const uint16_t *x, int hidden,
                                const uint16_t *w1, const uint16_t *w3,
                                const uint16_t *w2, int inter, float limit,
                                uint16_t *out);

// Head logits (F32): logits[V] = W[V,H] * x[H], FP16 inputs, F32 accumulation.
cudaError_t LaunchHeadLogits(cublasHandle_t handle, cudaStream_t stream,
                             const uint16_t *w, const uint16_t *x, float *logits,
                             int vocab, int hidden);

// ── MoE router + routed expert FFN (model.py:792,830) ──────────────────────
// One routed NVFP4 expert's three projections ([N,K] row-major nibbles +
// per-16 E4M3 scales + per-tensor global scale), as served by the neutral
// lsx_nvfp4 layer.
struct DeviceExpertWeight {
    const uint8_t *packed = nullptr;
    const uint8_t *scales = nullptr;
    float global_scale = 1.0f;
};

// sqrtsoftplus top-k router. scores = gate_w @ x / temp; experts are selected
// by scores + bias (bias steers selection only), the returned weights are the
// selected unbiased scores, renormalized when topk > 1 and scaled by
// route_scale. x/gate_w are FP16; bias F32.
cudaError_t LaunchRouterTopK(cublasHandle_t handle, cudaStream_t stream,
                             const uint16_t *x, const uint16_t *gate_w,
                             const float *bias, int n_routed, int hidden,
                             int topk, float temp, float route_scale, bool norm,
                             int32_t *out_idx, float *out_w);

// One routed expert on a single token:
//   accum[d] += weight * w2( silu(clamp(w1@x)) * clamp(w3@x) ).
// accum is F32 [hidden]; weight is applied to the expert output.
cudaError_t LaunchExpertFfn(cublasHandle_t handle, cudaStream_t stream,
                            const uint16_t *x, int hidden,
                            const DeviceExpertWeight &w1,
                            const DeviceExpertWeight &w3,
                            const DeviceExpertWeight &w2, int inter,
                            float swiglu_limit, float weight,
                            float *accum_f32);

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

// Indexer top-k over compressed positions (model.py:579): keeps the `topk`
// best of `n_pos` scores (ties -> lower position), re-sorts ascending, adds
// `offset`, and fills the tail (up to n_pos_pad) with -1.
cudaError_t LaunchIndexerTopk(const float *score, int n_pos, int topk,
                              int offset, int32_t *out_idx, int n_pos_pad,
                              cudaStream_t stream);

// Level-one candidate mask (model.py:583): keep the `topk_blocks` best blocks
// of `block_size`; the block holding position `num_valid-1` is pinned. mask has
// `n_pos` entries (1 = keep).
cudaError_t LaunchSelectCandidateBlocks(const float *score, int n_pos,
                                        int num_valid, int topk_blocks,
                                        int block_size, uint8_t *mask,
                                        cudaStream_t stream);

// Masks scores outside the candidate mask (consumers, model.py:575).
cudaError_t LaunchMaskIndexerScores(float *score, int n_pos,
                                    const uint8_t *mask, cudaStream_t stream);

// ── CED attention decode (compress_ratio > 0, model.py:613) ────────────────
// Layer weights/geometry for one CED decode step. All pointers are device.
struct CedLayerParams {
    int compress_ratio = 0;
    bool is_kv_source = false;
    bool is_index_source = false;
    bool is_candidate_source = false;
    bool owns_k = false;
    bool uses_candidates = false;

    const uint16_t *wq_a = nullptr;
    int q_lora = 0;
    const uint16_t *q_norm = nullptr;
    const uint16_t *wq_b = nullptr;
    int heads = 0;
    int head_dim = 0;
    int rope_dim = 0;
    const uint16_t *wkv = nullptr;
    const uint16_t *kv_norm = nullptr;
    const float *attn_sink = nullptr;
    const uint16_t *wo_a = nullptr;
    int o_groups = 0;
    int o_lora_rank = 0;
    const uint16_t *wo_b = nullptr;

    const uint16_t *comp_wkv = nullptr;
    const uint16_t *comp_wgate = nullptr;
    const uint16_t *comp_norm = nullptr;

    const uint16_t *idx_wq_b = nullptr;
    const uint16_t *idx_wk = nullptr;
    const uint16_t *idx_k_norm = nullptr;
    const uint16_t *idx_wproj = nullptr;
    int idx_n_heads = 0;
    int idx_head_dim = 0;
    int idx_topk = 0;
    int candidate_topk_blocks = 0;
    int candidate_block_size = 0;
};

// Per-layer persistent + scratch device buffers. Persistent fields live across
// decode steps; scratch fields may be shared with the attention scratch.
struct CedLayerBuffers {
    // persistent
    uint16_t *window_kv = nullptr;    // [window * head_dim]
    float *kv_state = nullptr;        // [ratio * head_dim] (ratio > 1)
    float *score_state = nullptr;     // [ratio * head_dim]
    uint16_t *compress_kv = nullptr;  // [compress_cap * head_dim] (kv_source)
    uint16_t *index_k = nullptr;      // [compress_cap * idx_head_dim] (owns_k)
    int compress_cap = 0;
    // scratch
    uint16_t *qr = nullptr;
    uint16_t *q = nullptr;
    uint16_t *kv = nullptr;
    uint16_t *latent = nullptr;
    uint16_t *o = nullptr;
    uint16_t *olora = nullptr;
    uint16_t *full_kv = nullptr;
    uint16_t *idx_q = nullptr;
    uint16_t *idx_knew = nullptr;
    float *idx_score = nullptr;
    float *idx_w = nullptr;
    float *latent_f = nullptr;   // [head_dim] ratio > 1 pooling scratch
    int32_t *win_idx = nullptr;
    int32_t *comp_idx = nullptr;
    int32_t *full_idx = nullptr;
    uint8_t *candidates = nullptr;
};

// One decode step of CED attention at position `start_pos` (seqlen == 1).
// `rope_cos`/`rope_sin` are the full [max_seq * rope_dim/2] tables for this
// layer's theta; they are indexed by position. On return `csa` holds the
// resources this layer published for its consumers.
cudaError_t CedAttentionForwardDecode(
    cublasHandle_t handle, cudaStream_t stream, const uint16_t *x, int hidden,
    int window, int layer_id, const CedLayerParams &p, CedLayerBuffers &b,
    const float *rope_cos, const float *rope_sin, int start_pos,
    CsaRuntime &csa, uint16_t *out);

} // namespace deepseek_v41
} // namespace lsxcommon
