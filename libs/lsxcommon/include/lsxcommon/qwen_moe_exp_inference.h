#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include <cuda_bf16.h>

#include "lsxcommon/qwen_inference.h"  // QwenNvfp4Weight
#include "lsxcommon/qwen_moe_inference.h"  // QwenMoeRoutedSlot + router kernel
#include "qwen_moe_exp_load.h"
#include "lsxcommon/sampling.h"

namespace lsxcommon::qwen_moe_exp {

// GatedResidual hyper-connection: mixed[H] + injection weights[hc] per token.
// d_hyper_input: [seq, hc*H] __half. d_hc_norm: BF16 bit-cast weight [hc*H]
// (1 + w centering applied on device). d_normed_scratch: [seq, hc*H] __half
// workspace (caller-owned). Math: fp32 intermediates, -fmad=false contract.
cudaError_t RunExpHcGatedResidual(
    cudaStream_t stream, const __half *d_hyper_input,
    const qwen::QwenNvfp4Weight &down, const qwen::QwenNvfp4Weight &up,
    const qwen::QwenNvfp4Weight &inject_w, const uint8_t *d_hc_norm,
    int seq, int H, int hc, int lowrank, float *d_mixed, float *d_inject_out,
    __half *d_mixed_h);

// Test-only raw-float twin (identical op sequence, no NVFP4 decode, no BF16
// norm cast) for bit-parity against the host replica in
// qwen_moe_exp_device_math.h. down/up/inject passed as raw fp32 row-major.
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


// ── PLE n-gram inject state + kernel (Task 6) ────────────────────────────

// Incremental state for the dilated depthwise short-conv over the gated
// PLE values: rolling window of the last (k-1)*dilation gated rows.
struct ExpPleState {
    float *d_window = nullptr;   // [state_len, hc*H] fp32 gated+normed rows
    float *d_scratch = nullptr;  // [3*hc*H + H] fp32 kernel workspace (global
                                 // memory: the 5*hc*H-float footprint exceeds
                                 // the 99KB/block shared-memory opt-in limit
                                 // on sm_120)
    int32_t *d_ctx = nullptr;    // [state_len] rolling token ids (host mirror)
    int state_len = 0;
    int H = 0;
    int hc = 0;
};

cudaError_t ExpPleStateAlloc(ExpPleState &st, int H, int hc, int state_len);
void ExpPleStateFree(ExpPleState &st);

// Single-token PLE inject (decode path):
//   key   = ple_key_proj @ emb_row            [hc*H]
//   query = group-RMS(hidden) with norm_query
//   gate  = sign(q·k / sqrt(H)) * sqrt(|q·k| clamp 1e-6)
//   value = ple_value_proj @ emb_row          [H]
//   gated = sigmoid(gate) * value broadcast to hc streams; norm_conv (1+w);
//   out   = gated + dilated_conv(window); hidden += out (in-place).
// d_emb_row: [heads*head_dim] fp32 gathered n-gram embedding rows (host
// pre-gathered from the mmap shards). d_token: current token id.
// d_ple_conv_w: depthwise conv weights [hc*H, k] fp32.
cudaError_t RunExpPleInject(cudaStream_t stream, ExpPleState &st,
                            const qwen::QwenNvfp4Weight &key_proj,
                            const qwen::QwenNvfp4Weight &value_proj,
                            const uint8_t *d_norm_key,
                            const uint8_t *d_norm_query,
                            const uint8_t *d_norm_conv,
                            const uint8_t *d_ple_conv_bf16,
                            const int32_t *d_token, const float *d_emb_row,
                            int heads, int head_dim, float eps,
                            __half *d_hidden /*[hc*H] in/out*/);

// LM head GEMM: logits[v] = head_w[v,:] . hidden[:] (FP16 weights in,
// FP32 accumulate + output). Uses a process-wide cuBLAS handle.
cudaError_t RunExpHeadGemm(cudaStream_t stream, const __half *d_head_f16,
                           const __half *d_hidden, float *d_logits, int H,
                           int V);

// Router top-10 (thin wrapper over the parameterized qwen router kernel).
cudaError_t RunExpRouterTop10(cudaStream_t stream, const uint8_t *d_router_bf16,
                              const __half *d_x, int seq, int H, int E, int K,
                              qwen_moe::QwenMoeRoutedSlot *d_slots);


// ── Generation driver (Task 8) ───────────────────────────────────────────
cudaError_t RunExpGeneration(
    cudaStream_t stream, const ExpPinnedWeights &pw,
    ExpExpertStore &store, gpu::ExpertDeviceCache &cache, ExpPleState &ple,
    std::vector<ExpQsaIndexerState> &qsa,
    const std::vector<ExpPleShardView> &ple_shards,
    const std::vector<int32_t> &prompt, int max_new, int eos_id,
    qwen::QwenGenerationResult &result, lsxcommon::sample::ISampler &sampler,
    qwen::QwenKvCache *kv);

} // namespace lsxcommon::qwen_moe_exp
