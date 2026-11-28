#pragma once

// Gemma-4-31B-it (dense) runtime — Model Ingestion Protocol step 5.
//
// One stream, no per-token cudaDeviceSynchronize. Every dense projection goes
// through the model-neutral NVFP4 layer (lsx_nvfp4::RunLsxNvfp4Gemm); the
// family-local kernels are the four per-layer norms, the HF-faithful RoPE
// (which differs between the sliding and the "proportional" global layers),
// the causal GQA attention with the sliding band, GeGLU, the NVFP4 embed
// gather and the softcapped logits/argmax.
//
// Activations are FP16 with FP32 accumulation (the repo's newer dense-family
// convention); the KV cache is addressed exclusively through KvCacheGeometry,
// because the two layer kinds store DIFFERENT per-layer geometries
// (sliding: 16 heads x 256; global: 4 heads x 512).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/gemma_dense_constants.h"
#include "lsxcommon/gemma_dense_load.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/token_stream.h"
#include "lsxcommon/sampling.h"

namespace lsxcommon {
namespace gemma_dense {

// ── Norms ───────────────────────────────────────────────────────────────────
// rows x hidden RMSNorm, out = rms(x) * w (Gemma4RMSNorm has NO 1+ offset).
// @p w == nullptr selects the plain form used by v_norm.
cudaError_t RunGemmaDenseRMSNorm(cudaStream_t stream, const __half *x,
                                 const __half *w, __half *out, int rows,
                                 int hidden, float eps);

// Per-head variant: grid (rows, heads), one RMSNorm over head_dim each — the
// q_norm / k_norm (weighted) and v_norm (plain) form.
cudaError_t RunGemmaDenseHeadNorm(cudaStream_t stream, const __half *x,
                                  const __half *w, __half *out, int rows,
                                  int heads, int head_dim, float eps);

// ── RoPE ────────────────────────────────────────────────────────────────────
// HF `apply_rotary_pos_emb` over the WHOLE head dim: dims (i, i + head_dim/2)
// form a pair for i < @p rot_angles, the frequency is theta^(-2i/head_dim), and
// the remaining pairs carry cos=1/sin=0 (the "proportional" encoding of the
// global layers, rot_angles = head_dim/8; a sliding layer is the
// rot_angles == head_dim/2 case of the same form).
cudaError_t RunGemmaDenseRope(cudaStream_t stream, const int32_t *d_pos,
                              __half *q, __half *k, int seq, int nh, int nkv,
                              int hd, int rot_angles, double theta);

// ── Attention ───────────────────────────────────────────────────────────────
// Causal attention with an optional sliding band (window > 0 attends
// [pos - window + 1, pos]; window <= 0 attends the whole prefix). One kernel
// serves both layer kinds: the global geometry is just different kv_heads /
// head_dim parameters plus a window of 0. @p scale is Gemma-4's fixed 1.0.
// @p pos_base is the first query row's absolute position.
cudaError_t RunGemmaDenseAttention(cudaStream_t stream, const __half *q,
                                   const __half *k, const __half *v, __half *out,
                                   int seq, int nh, int nkv, int hd, int window,
                                   float scale, int pos_base,
                                   KvCacheGeometry kgeom, KvCacheGeometry vgeom);

// Split-KV (flash-decode) variant for the seq==1 long-context regime; the
// partial slab is caller-owned and sized [kGemmaDenseAttnSplitMax][seq][nh]
// x (hd + 2). Fixed split boundaries + fixed merge order => run-to-run
// deterministic.
inline constexpr int kGemmaDenseAttnSplitMax = 32;
cudaError_t RunGemmaDenseAttentionSplit(cudaStream_t stream, const __half *q,
                                        const __half *k, const __half *v,
                                        __half *out, float *partials, int seq,
                                        int nh, int nkv, int hd, int window,
                                        float scale, int pos_base, int splits,
                                        KvCacheGeometry kgeom,
                                        KvCacheGeometry vgeom);

// ── MLP ─────────────────────────────────────────────────────────────────────
// gelu_pytorch_tanh elementwise over two [seq, I] projection outputs:
// out = gelu_tanh(gate) * up.
cudaError_t RunGemmaDenseGeGLU(cudaStream_t stream, const __half *gate,
                               const __half *up, __half *out, std::int64_t n);

// ── Embedding / head ────────────────────────────────────────────────────────
// Gathers @p seq rows out of the tied NVFP4 [vocab, hidden] table and scales
// them by sqrt(hidden_size) as upstream does (the constant is cast to the
// weight dtype first, so it is bf16(sqrt(hidden)) — 62.0 for 3840, 73.5 for
// 5376 — and the loader hands it in as data, never as a sibling's constant).
//
cudaError_t RunGemmaDenseEmbedGatherNvfp4(cudaStream_t stream,
                                          const std::uint8_t *packed,
                                          const std::uint8_t *scales,
                                          float global_scale,
                                          const std::int32_t *ids, __half *out,
                                          int seq, int vocab, int hidden,
                                          float embed_scale);

// Softcapped tied-head logits (logits = tanh(l / cap) * cap) plus a
// deterministic argmax (lowest index wins a tie) and its logit value. The
// projections run through the neutral NVFP4 GEMV (the warp-per-row kernel,
// ~2x the hand-rolled row kernel's bandwidth); @p d_logits_h is a caller-owned
// half[vocab] scratch, and @p d_logits is caller-owned FP32 and must hold
// vocab floats PLUS ceil(vocab / 128) * 2 trailing words for the argmax
// winners.
cudaError_t RunGemmaDenseLogitsArgmax(cudaStream_t stream,
                                      const std::uint8_t *packed,
                                      const std::uint8_t *scales,
                                      float global_scale, const __half *x,
                                      int vocab, int hidden, float cap,
                                      std::int32_t *d_argmax, float *d_maxlogit,
                                      float *d_logits, __half *d_logits_h);

// ── Generation ──────────────────────────────────────────────────────────────
struct GemmaDenseGenerationResult {
    bool ok = false;
    std::string error_msg;
    std::vector<std::int32_t> tokens;
    std::vector<float> final_logits;  // last decode step (softcapped)
    double ttft_ms = 0.0;
    double total_ms = 0.0;
    bool eos_terminated = false;
    std::uint64_t resident_bytes = 0;
};

// Persistent per-session generation context (capacity-grown workspace). Owned
// by the model; single-threaded (one consumer).
struct GemmaDenseGenContext {
    struct Workspace {
        __half *x = nullptr;        // [S][H]  residual stream
        __half *normed = nullptr;   // [S][H]
        __half *q = nullptr;        // [S][q_rows_max]
        __half *k = nullptr;        // [S][kv_rows_max]
        __half *v = nullptr;        // [S][kv_rows_max]
        __half *attn = nullptr;     // [S][q_rows_max]
        __half *proj = nullptr;     // [S][H]
        __half *gate_up = nullptr;  // [S][2I]
        __half *act = nullptr;      // [S][I]
        __half *ffn = nullptr;      // [S][H]
        __half *logits_h = nullptr; // [V] half GEMV output of the tied head
        float *logits = nullptr;    // [V] fp32 + the argmax winner tail
        int32_t *ids = nullptr;     // [S]
        int32_t *pos = nullptr;     // [S]
        int32_t *argmax = nullptr;  // [1]
        float *maxv = nullptr;      // [1]
        float *attn_part = nullptr; // split-KV partials
        size_t seq_cap = 0, q_rows_cap = 0, kv_rows_cap = 0;

        void Free();
        cudaError_t Alloc(std::int64_t H, std::int64_t I, std::int64_t V,
                          size_t S, size_t q_rows_max, size_t kv_rows_max,
                          size_t nh, size_t hd);
    };

    Workspace ws;
    bool ws_ready = false;
    int eos_count = 0;

    ~GemmaDenseGenContext() { ws.Free(); }
};

GemmaDenseGenContext *CreateGemmaDenseGenContext();
void DestroyGemmaDenseGenContext(GemmaDenseGenContext *ctx);

// Full generation loop: prefill(input) + up to max_gen_tokens decode steps on
// ONE stream. @p kv must be pre-allocated for prompt + generated capacity.
cudaError_t RunGemmaDenseGeneration(
    cudaStream_t stream, const GemmaDensePinnedWeights &pw,
    const std::vector<std::int32_t> &input_ids, int max_gen_tokens,
    const std::vector<int> &eos_ids, GemmaDenseGenerationResult &out,
    GemmaDenseKvCache &kv, GemmaDenseGenContext *ctx = nullptr,
    lsxcommon::sample::ISampler &sampler = lsxcommon::sample::DefaultSampler(),
    const lsxcommon::GenerationStream *gs = nullptr);

} // namespace gemma_dense
} // namespace lsxcommon
