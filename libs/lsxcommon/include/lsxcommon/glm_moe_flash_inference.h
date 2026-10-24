#pragma once

// GLM-5.3-Flash inference kernels + generation driver.
// KDA (linear_attention) layers: causal conv1d + gated delta rule with
// per-channel decay; DSA (deepseek_sparse_attention) layers: MLA + indexer
// (ported as NEW functions from the GLM-5.2 kernel semantics); MoE: sigmoid
// noaux_tc top-8 router + NVFP4 routed/shared experts; hyper-connections
// via the shared cached GatedResidual primitives.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "lsxcommon/glm_moe_flash_load.h"
#include "lsxcommon/qwen_dense_inference_pipeline.h"
#include "lsxcommon/lsx_attn_inference.h"     // lsx_nvfp4::LsxNvfp4Weight, GenerationResult
#include "lsxcommon/lsx_moe_inference.h"  // MoERoutedSlot
#include "lsxcommon/sampling.h"
#include "lsxcommon/token_stream.h"

#ifndef LSXCOMMON_GLM_FLASH_INFERENCE_H
#define LSXCOMMON_GLM_FLASH_INFERENCE_H

namespace lsxcommon {
namespace glm_moe_flash {

// ── Test twins (raw F32 buffers, no FP16, no NVFP4) for bit-parity ────────

// Depthwise causal conv window: x [K, C] rows oldest..newest, w [C, K],
// y [C]. y[c] = sum_k w[c*K+k] * x[k*C+c].
cudaError_t RunFlashConv1dF32(cudaStream_t stream, const float *x,
                              const float *w, float *y, int channels,
                              int kernel);

// Gated delta rule with per-channel decay (test twin). q/k/v: [seq, H, D]
// FP32; beta: [seq, H]; gdecay: [seq, H*D]; state_out: [H, D, D];
// out: [seq, H, D]. Mirrors FlashGatedDeltaHost exactly.
cudaError_t RunFlashGatedDeltaF32(cudaStream_t stream, const float *q,
                                  const float *k, const float *v,
                                  const float *beta, const float *gdecay,
                                  int seq, int heads, int dim,
                                  float *state_out, float *out);

// Router top-8: gate BF16 bits [E, H], bias F32 [E], x BF16 bits [tok, H].
// Slots: [tok*8] expert + coeff (renormed, scaled by 2.5).
cudaError_t RunFlashRouterTop8(cudaStream_t stream, const uint8_t *d_gate,
                               const float *d_bias, const uint8_t *d_x,
                               int tok, int hidden, int experts,
                               moe::MoERoutedSlot *d_slots);

// ── DSA indexer (sparse attention) ───────────────────────────────────────
// transformers `Glm5NextTextIndexer` with k-pool compression: the indexer is
// its own lightweight attention. Per position it stores
//   k_i   = LayerNorm(wk @ x_i)            (eps 1e-6, bias included)
//   gate_i= compress_gate @ x_i            (one logit per key channel)
// and every `index_kpool` positions it builds a pooled key
//   pool_q[c] = sum_i softmax_i(gate_i[c] + ape[i][c]) * k_i[c]
// A query scores the *pools* (relu(dot) * 128^-0.5 summed over the 32 indexer
// heads, weighted by weights_proj(x) * 32^-0.5), keeps the top
// `index_topk / index_kpool` pools, expands them back to token indices in
// score order, and appends the visible tail (the current incomplete pool).
// The main MLA then attends only to those positions. Below `index_topk` the
// selection is the whole context, so the driver keeps the dense kernel and
// everything stays bit-identical (pinned by the kernels test).
struct FlashIndexerState {
    __half *d_k = nullptr;      // [cap * kIndexerHeadDim] token keys
    __half *d_gate = nullptr;   // [cap * kIndexerHeadDim] per-channel gates
    float *d_pool = nullptr;    // [n_pool_cap * kIndexerHeadDim]
    float *d_q = nullptr;       // [kIndexerHeads * kIndexerHeadDim]
    float *d_w = nullptr;       // [kIndexerHeads]
    float *d_scores = nullptr;  // [n_pool_cap]
    int32_t *d_sel = nullptr;   // [kIndexerTopk + kIndexerKpool - 1]
    int32_t *d_sel_count = nullptr;
    __half *d_kraw = nullptr;   // [kIndexerHeadDim]       wk staging
    __half *d_qraw = nullptr;   // [kIndexerHeads*dim]     wq_b staging
    __half *d_wraw = nullptr;   // [kIndexerHeads]         weights_proj staging
    int64_t capacity_ctx = 0;
    int64_t pool_cap = 0;
    int64_t n_pools = 0;        // completed pools
};

cudaError_t FlashIndexerStateAlloc(FlashIndexerState &st, int64_t max_ctx);
void FlashIndexerStateFree(FlashIndexerState &st);

// Stores k_i and gate_i for the token at `pos` (indexer key cache append).
cudaError_t RunFlashIndexerToken(cudaStream_t stream, FlashIndexerState &st,
                                 const FlashLayerWeights &lw, int hidden,
                                 const __half *d_x, int64_t pos);

// Builds the pooled key for the pool that just completed (the last `kpool`
// positions) and advances n_pools.
cudaError_t RunFlashIndexerPool(cudaStream_t stream, FlashIndexerState &st,
                                const FlashLayerWeights &lw,
                                int64_t first_tok);

// Scores every completed pool and writes the selected token indices for the
// query at `pos` (score-descending pools, expanded 1:kpool, plus the visible
// tail, padded with -1).
cudaError_t RunFlashIndexerSelect(cudaStream_t stream, FlashIndexerState &st,
                                  const FlashLayerWeights &lw, int hidden,
                                  const __half *d_x, const __half *d_q_resid,
                                  int64_t pos);

// Sparse MLA over the selected positions; same numerics as the dense kernel
// (FP32 scores/softmax, one FP16 rounding at the end). Entries < 0 are skipped.
cudaError_t RunFlashMlaAttnSelected(cudaStream_t stream, const __half *d_q,
                                    const __half *k_cache,
                                    const __half *v_cache,
                                    const int32_t *d_sel,
                                    const int32_t *d_count, __half *d_out,
                                    int n_sel, int heads, int dim,
                                    float scale, size_t smem);

// Test twins (raw FP32/FP16 buffers, no NVFP4) for host parity.
cudaError_t RunFlashIndexerPoolTest(cudaStream_t stream, const __half *keys,
                                    const __half *gates, const uint16_t *ape,
                                    float *pool, int dim, int kpool);
cudaError_t RunFlashIndexerScoresTest(cudaStream_t stream, const float *q,
                                      const float *w, const float *pools,
                                      int heads, int dim, int n_pools,
                                      float *out_scores);
// Runs the PRODUCTION selection kernel on raw scores.
cudaError_t RunFlashIndexerSelectKTest(cudaStream_t stream,
                                       const float *scores, int n_pools,
                                       int select_k, int kpool,
                                       int64_t visible_count, int topk,
                                       int32_t *out_sel, int32_t *out_count);

// Device bytes of one layer's indexer state at `max_ctx` (keys/gates scale with
// the context, the selection scratch does not). The allocator carves exactly
// this block, so the session-scratch plan cannot drift from it.
int64_t FlashIndexerStateBytes(int64_t max_ctx);

// ── KDA decode-path state + kernels ──────────────────────────────────────
// Per-layer recurrent state: S[heads, dim, dim] FP32 + short-conv history
// window [K-1 rows, 3*8192] FP16 (q/k/v channels interleaved by projection).
struct FlashKdaState {
    float *d_state = nullptr;    // [heads*dim*dim] (heads=64, dim=128)
    __half *d_conv_hist = nullptr;  // [(K-1), 3*8192] oldest..newest
    int64_t capacity_ctx = 0;
};

cudaError_t FlashKdaStateAlloc(FlashKdaState &st, int64_t max_ctx);
void FlashKdaStateFree(FlashKdaState &st);

// Device bytes of one KDA layer's state: S[64,128,128] FP32 + the fixed
// (conv_kernel - 1)-row history. Constant - there is no context buffer.
inline constexpr int64_t FlashKdaLayerBytes()
{
    return static_cast<int64_t>(64) * 128 * 128 * 4 +
           static_cast<int64_t>(3) * 3 * 8192 * 2;
}

// Process-lifetime FP16 embedding table (BF16 source -> FP16 widening), keyed
// by BOTH the source tensor and the model weight epoch. A size-only key served
// the previous model's embeddings when two GLM instances shared V*H (audit X9).
struct FlashEmbedCache {
    __half *data = nullptr;
    int64_t elems = 0;
    unsigned long long epoch = 0;
    const __nv_bfloat16 *source = nullptr;

    cudaError_t Ensure(cudaStream_t stream, const __nv_bfloat16 *src,
                       int64_t n, unsigned long long weight_epoch);
    void Free();
};

// Single-token KDA decode for one layer (seq == 1):
//   d_x: [H] current token hidden (post hc mixing, FP16).
//   d_out: [H] FP16 layer attention output (pre o_norm? no: post o_proj).
// Pipeline: q/k/v GEMM (own FP16 GEMV path) -> depthwise causal
// conv1d (window = history + current) -> l2norm q/k per head -> gating prep
// (beta = sigmoid(b_proj@x), decay = exp(-exp(A_log)*softplus(dt+dt_bias))
// per (h,l), dt = f_b(f_a(x))) -> per-channel-decay delta step (state seed
// from d_state, write-back) -> RMSNormGated(core, gate, o_norm) -> o GEMM.
// All on `stream`, no host sync.
cudaError_t RunFlashKdaDecode(cudaStream_t stream, FlashKdaState &st,
                              const FlashLayerWeights &lw, int hidden,
                              int heads, int dim, int conv_kernel,
                              const __half *d_x, __half *d_out,
                              void *d_scratch /* caller workspace */);

// Router top-8 with FP16 activations (gate stays BF16 bits on device).
cudaError_t RunFlashRouterTop8F16(cudaStream_t stream, const uint8_t *d_gate,
                                  const float *d_bias, const __half *d_x,
                                  int tok, int hidden, int experts,
                                  moe::MoERoutedSlot *d_slots);

// Test probe: flash_expf_glibc(0/1).
cudaError_t RunFlashExp0Probe(cudaStream_t stream, float *d_out);

// Test wrapper: MLA attention kernel over a single-token cache.
cudaError_t RunFlashMlaAttnTest(cudaStream_t stream, const __half *d_q,
                                const __half *d_k, const __half *d_v,
                                __half *d_out, int heads, int dim,
                                float scale, size_t smem);

// ── Fused decode test twins (FP16 bit-parity harness) ────────────────────
// Depthwise causal conv decode with optional fused post-conv SiLU (the
// production decode path). hist [K-1, channels], x [channels], w [channels,
// K], y [channels], all FP16.
cudaError_t RunFlashConvDecodeTest(cudaStream_t stream, const __half *hist,
                                   const __half *x, const __half *w, __half *y,
                                   int channels, int kernel, bool apply_silu);

// In-place SiLU over n FP16 elements (reference for the conv+silu parity).
cudaError_t RunFlashSiluTest(cudaStream_t stream, __half *x, int n);

// In-place gate activation (sigmoid default, silu opt-in) over n FP16 elems.
cudaError_t RunFlashGateActTest(cudaStream_t stream, __half *g, int n,
                                bool silu);

// RMSNormGated per head with optional fused on-the-fly gate activation.
cudaError_t RunFlashRmsnormGatedTest(cudaStream_t stream, const float *core,
                                     const __half *gate, const __half *norm_w,
                                     __half *out, int heads, int dim, float eps,
                                     bool fuse_gate_act, bool gate_silu);

// L2-normalize q rows in place (single-plane reference path).
cudaError_t RunFlashL2normTest(cudaStream_t stream, __half *row, int dim,
                               float extra_scale, int heads);

// Fused q/k L2-normalize (one launch; q rows use q_scale, k rows use 1.0).
cudaError_t RunFlashL2normQKTest(cudaStream_t stream, __half *q, __half *k,
                                 int dim, float q_scale, int heads);

// ── MoE: clamped-swiglu routed chain (flash-local; GLM semantics) ────────
// Per-slot chain over LRU-resident experts (staging via
// moe::EnsureMoEExpertResident): y_g = w1@x, y_u = w3@x (NVFP4 GEMM),
// inter = clamp10(silu(y_g)) * clamp10(y_u), y2 = w2@inter, acc += c*y2.
//
// Perf path: when `batch_ygu`/`batch_hbuf`/`batch_ypart`/`d_gemv_tbl` are
// supplied, the K slots run through the shared batched routed-GEMV chain
// (moe::RunMoERoutedGemmVBatchCore) -- 4 launches per layer instead
// of K*4, same slot-major weight streaming. Nulls fall back to the legacy
// per-slot chain (also the test/ad-hoc path).
cudaError_t RunFlashRoutedFFN(
    cudaStream_t stream, lsxcommon::moe::MoEExpertStore &store,
    gpu::ExpertDeviceCache &cache, int layer, int moe_layer,
    const moe::MoERoutedSlot *h_slots, int k_top,
    const __half *d_x, int hidden, int moe_inter, float *d_acc,
    __half *d_yg, __half *d_yu, __half *d_y2, std::string &error,
    cudaStream_t h2d_stream, cudaEvent_t *copy_ev, cudaEvent_t *slot_free_ev,
    __half *batch_ygu, __half *batch_hbuf, __half *batch_ypart,
    moe::MoEGvSlotDev *d_gemv_tbl);

// Shared expert: same clamped swiglu semantics; out += result.
cudaError_t RunFlashSharedFFN(cudaStream_t stream,
                              const FlashLayerWeights &lw,
                              const __half *d_x, int hidden, int shared_inter,
                              __half *d_yg, __half *d_yu, __half *d_y2,
                              __half *d_out);

// ── Generation driver ─────────────────────────────────────────────────────
// Single-stream incremental decode over the GLM-5.3-Flash layer stack:
// hc GatedResidual (cached) -> KDA/DSA attention branch -> hc inject scatter
// -> hc FFN branch (MoE: router top-8 + LRU-resident NVFP4 routed experts +
// shared expert; dense layers 0..2: NVFP4 MLP) -> hc inject scatter.
cudaError_t RunGlmFlashGeneration(
    cudaStream_t stream, const FlashPinnedWeights &pw,
    lsxcommon::moe::MoEExpertStore &store, gpu::ExpertDeviceCache &cache,
    const std::vector<int32_t> &prompt, int max_new,
    const std::vector<int> &eos_ids, qwen::GenerationResult &result,
    sample::ISampler &sampler,
    const lsxcommon::GenerationStream *gs = nullptr);


// ── Session scratch budget ───────────────────────────────────────────────
// Everything the generation session allocates ON TOP of the resident graph,
// which is why the expert cache cannot size itself from the free memory alone:
// the DSA KV caches scale with the context (0.69 MiB per position across the 11
// DSA layers), while the embed table, the KDA state and the arena are constant.
// Callers reserve this before sizing the cache (moe::EnsureMoEExpertCacheHeadroom)
// so a long prompt cannot starve the session.
struct FlashSessionScratch {
    int64_t embed = 0;    // FP16 embedding table, vocab x hidden
    int64_t kda = 0;      // 34 x FlashKdaLayerBytes()
    int64_t dsa_kv = 0;   // 11 layers x 2 x cap x heads x qk_dim x 2 bytes
    int64_t indexer = 0;  // 11 x FlashIndexerStateBytes(cap)
    int64_t arena = 0;    // decode scratch (constant, see GlmFlashArenaBytes)
    int64_t Total() const { return embed + kda + dsa_kv + indexer + arena; }
};
FlashSessionScratch GlmFlashSessionScratchBytes(int64_t max_ctx);

} // namespace glm_moe_flash
} // namespace lsxcommon

#endif // LSXCOMMON_GLM_FLASH_INFERENCE_H
