#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>


namespace lsxcommon::minimax {

// One routed slot: selected expert id + renormalized routing weight (8 bytes,
// the same layout the shared expert chain uses, but owned here so this family
// does not pull the legacy MoE layer into its TUs).
struct alignas(8) MinimaxRoutedSlot {
    int32_t eid;
    float coeff;
};

// ── Numeric kernels (FP16 activations, FP32 accumulation) ───────────────────
//
// Every kernel here is stream-ordered on the caller's stream, deterministic
// run to run, and uses no atomicAdd in anything feeding final_logits.

// HF-parity router (modeling_minimax_m3_vl.MiniMaxM3VLTopKRouter):
//   logits_e = W[e] . x             (F32 weights, F32 accumulation)
//   score_e  = sigmoid(logits_e)
//   select   = top-K of (score_e + bias_e)   (lower expert id wins ties)
//   coeff_k  = score_e / sum(score of selected)
// Writes K slots per token, ordered by ascending expert id (deterministic
// accumulation order for the caller).
// Launch: <<<seq, 256, 2*E floats shared, stream>>>.
cudaError_t RunMinimaxRouter(cudaStream_t stream,
                             const float *router_w,   // [E,H]
                             const float *router_bias,  // [E]
                             const __half *x,           // [seq,H]
                             int seq, int E, int K, int H,
                             MinimaxRoutedSlot *d_slots);  // [seq*K]

// GPT-OSS-style SwiGLU-OAI (config hidden_act "swigluoai"):
//   g = min(gate, limit); u = clamp(up, -limit, limit)
//   out = (u + 1) * g * sigmoid(alpha * g)
// in-place capable (out may alias gate). n elements.
cudaError_t RunMinimaxSwiGLUOai(cudaStream_t stream, const __half *gate,
                                const __half *up, __half *out, int64_t n,
                                float alpha, float limit);

// Gemma-form RMSNorm over full rows: out = rmsnorm(x) * (1 + w).
cudaError_t RunMinimaxNorm(cudaStream_t stream, const __half *x,
                           const __half *w, __half *out, int rows, int hidden,
                           float eps);

// Gemma-form per-head RMSNorm over [rows, heads, head_dim]; out may alias x.
cudaError_t RunMinimaxHeadNorm(cudaStream_t stream, const __half *x,
                               const __half *w, __half *out, int rows,
                               int heads, int head_dim, float eps);

// Partial rotary (HF rotate_half pairing: dims j and j + rotary_dim/2),
// applied to the FIRST rotary_dim channels of each head; the tail passes
// through. q: [seq, q_heads, head_dim], k: [seq, k_heads, head_dim]
// (k_heads = 0 skips k). Position of row t is pos_base + t.
cudaError_t RunMinimaxPartialRope(cudaStream_t stream, __half *q, __half *k,
                                  int seq, int q_heads, int k_heads,
                                  int head_dim, int rotary_dim, float theta,
                                  int pos_base);

// Token embedding lookup from an FP16 table: out[t][h] = table[ids[t]][h].
cudaError_t RunMinimaxEmbed(cudaStream_t stream, const __half *table,
                            const int32_t *ids, __half *out, int seq,
                            int hidden);

// Token embedding lookup straight out of the NVFP4 embed table
// (packed [V,H] row-major E2M1 nibbles, E4M3 scales [V, H/16], per-tensor
// global scale): dequantizes ONLY the requested row, so the model never keeps
// a dequantized 200k x 6144 FP16 copy (2.4 GiB) resident.
cudaError_t RunMinimaxEmbedNvfp4(cudaStream_t stream, const uint8_t *packed,
                                 const uint8_t *scales, float global_scale,
                                 const int32_t *ids, __half *out, int seq,
                                 int hidden);

// MiniMax Sparse Attention block selection (Lightning Indexer):
//   score[h][k] = idx_q[h] . idx_k[k]      (UNSCALED, FP32)
//   causal: keys past `pos` are -inf; per-block max-pool (block_size keys);
//   the `local_blocks` blocks ending at the query's own block are forced to
//   +inf; top-`topk_blocks` blocks per index head are kept.
// idx_q: [index_heads, index_head_dim] (one query token), idx_k: [kv, dim]
// (already appended for this position). Writes d_mask[h*num_blocks + b] = 1/0.
cudaError_t RunMinimaxIndexerSelect(cudaStream_t stream,
                                    const __half *idx_q, const __half *idx_k,
                                    int kv_len, int pos, int index_heads,
                                    int index_head_dim, int block_size,
                                    int topk_blocks, int local_blocks,
                                    uint8_t *d_mask);

// Causal GQA decode attention with an OPTIONAL block mask (the dense path
// passes block_mask == nullptr). q: [1, heads, head_dim]; k/v row for
// attended index s is at byte offset s*pos_stride (+ head*kv_head_stride [+
// dim]); the caller supplies the already-offset per-layer base pointers and
// the two strides, so the cache layout stays a KvCacheGeometry decision.
// `group_heads` = heads / index_heads maps a query head to its selection row.
cudaError_t RunMinimaxMaskedAttn(cudaStream_t stream, const __half *q,
                                 const __half *k_base, const __half *v_base,
                                 int kv_len, int heads, int kv_heads,
                                 int head_dim, float scale, int64_t pos_stride,
                                 int64_t head_stride, const uint8_t *block_mask,
                                 int num_blocks, int block_size,
                                 int group_heads, int64_t score_stride,
                                 float *d_scores, __half *out);

// acc[i] += coeff * y[i] (FP32 accumulator, one routed slot).
cudaError_t RunMinimaxRoutedAccumulate(cudaStream_t stream, float *acc,
                                       const __half *y, float coeff, int64_t n);

// out[t][h] = shared[t][h] + routed_scale * acc[t][h] (+ residual folded in by
// the caller). acc is FP32, shared/out FP16.
cudaError_t RunMinimaxMoECombine(cudaStream_t stream, const float *acc,
                                 const __half *shared, __half *out, int rows,
                                 int hidden, float routed_scale);

} // namespace lsxcommon::minimax
