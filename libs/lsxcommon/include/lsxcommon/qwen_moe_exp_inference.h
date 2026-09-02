#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "lsxcommon/qwen_inference.h"  // QwenNvfp4Weight

namespace lsxcommon::qwen_moe_exp {

// GatedResidual hyper-connection: mixed[H] + injection weights[hc] per token.
// d_hyper_input: [seq, hc*H] __half. d_hc_norm: BF16 bit-cast weight [hc*H]
// (1 + w centering applied on device). d_normed_scratch: [seq, hc*H] __half
// workspace (caller-owned). Math: fp32 intermediates, -fmad=false contract.
cudaError_t RunExpHcGatedResidual(
    cudaStream_t stream, const __half *d_hyper_input,
    const qwen::QwenNvfp4Weight &down, const qwen::QwenNvfp4Weight &up,
    const uint8_t *d_hc_norm, int seq, int H, int hc, int lowrank,
    float *d_mixed, float *d_inject_w, __half *d_normed_scratch);

// Test-only raw-float twin (identical op sequence, no NVFP4 decode, no BF16
// norm cast) for bit-parity against the host replica in
// qwen_moe_exp_hostref.h. down/up/inject passed as raw fp32 row-major.
cudaError_t RunExpHcGatedResidualF32(
    cudaStream_t stream, const float *d_hyper_input, const float *d_down,
    const float *d_up, const float *d_inject_w, int seq, int H, int hc,
    int lowrank, float *d_mixed, float *d_inject_w_out);

// hyper_input[c*H+i] += branch_out[i] * inject_w[c] for each stream c.
cudaError_t RunExpHcInjectScatter(cudaStream_t stream,
                                  const __half *d_hyper_input,
                                  const __half *d_branch_out,
                                  const float *d_inject_w, int seq, int H,
                                  int hc, __half *d_out);

// Final global mixer (GatedResidual without block_inject) — out [seq, H].
cudaError_t RunExpHcMixer(cudaStream_t stream, const __half *d_hidden,
                          const qwen::QwenNvfp4Weight &down,
                          const qwen::QwenNvfp4Weight &up,
                          const uint8_t *d_mixer_norm, int seq, int H, int hc,
                          int lowrank, __half *d_out, __half *d_normed_scratch);

// ── QSA indexer state + kernels (Task 5) ─────────────────────────────────

struct ExpQsaIndexerState {
    __half *d_raw_keys = nullptr;    // [cap, HD] MQA key cache (incremental)
    float *d_pooled_keys = nullptr;  // [cap/compress, HD] append-only pooled
    int *d_block_starts = nullptr;   // [cap/compress] first token of block
    int64_t capacity_ctx = 0;
    int64_t host_pooled_count = 0;   // test seam: valid pooled rows
    int head_dim = 0;
    int compress = 0;
};

cudaError_t ExpQsaIndexerAlloc(ExpQsaIndexerState &st, int64_t max_ctx,
                               int head_dim);
void ExpQsaIndexerFree(ExpQsaIndexerState &st);

// Selects micro-blocks for the current query window and writes the selected
// mask as 0/1 int32 [seq, past_ctx + seq]. Degenerate fast path:
// ctx <= block_topk * compress -> all-visible mask, no scoring.
// d_indexer_q: [seq, index_heads, HD] post-norm post-rope queries.
// d_new_raw_key: [seq, HD] raw (unpooled) keys appended to the cache.
cudaError_t RunExpQsaIndexer(cudaStream_t stream, ExpQsaIndexerState &st,
                             const __half *d_indexer_q,
                             const __half *d_new_raw_key, int past_ctx,
                             int seq, int head_dim, int compress,
                             int block_topk, int *d_selected_mask);

// Masked GQA attention (causal ∧ indexer-selected), online softmax fp32.
// d_q: [seq, heads, head_dim]; caches [past+seq, kv_heads, head_dim] laid out
// position-major; d_out: [seq, heads * head_dim].
cudaError_t RunExpQsaAttention(cudaStream_t stream, const __half *d_q,
                               const __half *d_k_cache,
                               const __half *d_v_cache,
                               const int *d_selected_mask, int seq,
                               int past_ctx, int heads, int kv_heads,
                               int head_dim, float scale, __half *d_out);

} // namespace lsxcommon::qwen_moe_exp
