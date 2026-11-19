#pragma once

// Muse-Glimmer-30B runtime — Model Ingestion Protocol step 5.
//
// One stream, no per-token cudaDeviceSynchronize. Every dense projection goes
// through the model-neutral NVFP4 layer (lsx_nvfp4::RunLsxNvfp4Gemm); the
// family-local kernels are the centered norms, the weightless per-head QK
// norm + Q scale, the sliding-band / NoPE causal GQA attention, the sigmoid
// attention output gate, SwiGLU (silu), the FP16 embed gather + weightless
// embed norm and the softcapped FP16-head logits/argmax.
//
// Activations are FP16 with FP32 accumulation; the KV cache is addressed
// exclusively through KvCacheGeometry (uniform 2-head x 128 per layer,
// pos-major).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/muse_dense_constants.h"
#include "lsxcommon/muse_dense_load.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/token_stream.h"
#include "lsxcommon/sampling.h"

namespace lsxcommon {
namespace muse_dense {

// ── Norms ───────────────────────────────────────────────────────────────────
// Centered (Gemma2-style) row norm: out = rms(x) * (1 + w). The four
// per-layer norms use this form (pre eps rms_norm_eps, post eps post_norm_eps).
cudaError_t RunMuseDenseCenteredNorm(cudaStream_t stream, const __half *x,
                                     const __half *w, __half *out, int rows,
                                     int hidden, float eps);

// Plain (Gemma4-style) row norm: out = rms(x) * w; @p w == nullptr selects the
// weightless form (the final norm uses w, the embed norm is weightless).
cudaError_t RunMuseDensePlainNorm(cudaStream_t stream, const __half *x,
                                  const __half *w, __half *out, int rows,
                                  int hidden, float eps);

// Weightless per-head QK norm: q and k are RMS-normalized over head_dim, then
// Q is multiplied by @p q_scale (3.87). k keeps scale 1. In-place on q/k.
cudaError_t RunMuseDenseQkNormScale(cudaStream_t stream, __half *q, __half *k,
                                    int seq, int nh, int nkv, int hd, float eps,
                                    float q_scale);

// ── RoPE ────────────────────────────────────────────────────────────────────
// HF rotate_half over the WHOLE head dim, frequency theta^(-2i/hd). The caller
// SKIPS this on NoPE (full-attention) layers. q/k rotated in place.
cudaError_t RunMuseDenseRope(cudaStream_t stream, const int32_t *d_pos,
                             __half *q, __half *k, int seq, int nh, int nkv,
                             int hd, double theta);

// ── Attention ───────────────────────────────────────────────────────────────
// Causal GQA attention with an optional sliding band (window > 0 attends
// [pos - window + 1, pos]; window <= 0 attends the whole prefix — the NoPE
// full layers). KV addressed exclusively through KvCacheGeometry. @p scale is
// the standard 1/sqrt(head_dim). Deterministic shared-memory reductions.
cudaError_t RunMuseDenseAttention(cudaStream_t stream, const __half *q,
                                  const __half *k, const __half *v, __half *out,
                                  int seq, int nh, int nkv, int hd, int window,
                                  float scale, int pos_base,
                                  KvCacheGeometry kgeom, KvCacheGeometry vgeom);

// Split-KV (flash-decode) variant for seq==1 long context; caller-owned
// partial slab. Fixed split boundaries + fixed merge order => deterministic.
inline constexpr int kMuseDenseAttnSplitMax = 32;
cudaError_t RunMuseDenseAttentionSplit(cudaStream_t stream, const __half *q,
                                       const __half *k, const __half *v,
                                       __half *out, float *partials, int seq,
                                       int nh, int nkv, int hd, int window,
                                       float scale, int pos_base, int splits,
                                       KvCacheGeometry kgeom,
                                       KvCacheGeometry vgeom);

// ── Gates / MLP ─────────────────────────────────────────────────────────────
// Attention OUTPUT gate: out = attn * sigmoid(gate), elementwise over n dims,
// applied BEFORE o_proj (n = seq * q_rows).
cudaError_t RunMuseDenseAttnGate(cudaStream_t stream, const __half *attn,
                                 const __half *gate, __half *out,
                                 std::int64_t n);

// SwiGLU (silu): out = silu(gate) * up, elementwise over n.
cudaError_t RunMuseDenseSwiGLU(cudaStream_t stream, const __half *gate,
                               const __half *up, __half *out, std::int64_t n);

// ── Embedding / head ────────────────────────────────────────────────────────
// Gathers @p seq FP16 rows out of the [vocab, hidden] embed table (the
// weightless embed norm is applied separately via RunMuseDensePlainNorm).
cudaError_t RunMuseDenseEmbedGatherFp16(cudaStream_t stream, const __half *embed,
                                        const std::int32_t *ids, __half *out,
                                        int seq, int hidden);

// FP16-head logits + deterministic argmax: raw = lm_head @ x (neutral FP16
// GEMV), logits = cap * tanh(raw * multiplier / cap). @p d_logits is caller-
// owned FP32, vocab floats PLUS ceil(vocab/128)*2 trailing words for the argmax
// winners. Lowest index wins a tie; no atomicAdd feeds final_logits.
cudaError_t RunMuseDenseLogitsArgmax(cudaStream_t stream, const __half *lm_head,
                                     const __half *x, int vocab, int hidden,
                                     float multiplier, float cap,
                                     std::int32_t *d_argmax, float *d_maxlogit,
                                     float *d_logits);

// ── Generation ──────────────────────────────────────────────────────────────
struct MuseDenseGenerationResult {
    bool ok = false;
    std::string error_msg;
    std::vector<std::int32_t> tokens;
    std::vector<float> final_logits;  // last decode step (softcapped)
    double ttft_ms = 0.0;
    double total_ms = 0.0;
    bool eos_terminated = false;
    std::uint64_t resident_bytes = 0;
};

// Full generation loop: prefill(input) + up to max_gen_tokens decode steps on
// ONE stream. @p kv must be pre-allocated for prompt + generated capacity.
// The driver body lands with the kernel suite (Task 7); the signature is the
// contract the model TU compiles against.
cudaError_t RunMuseDenseGeneration(
    cudaStream_t stream, const MuseDensePinnedWeights &pw,
    const std::vector<std::int32_t> &input_ids, int max_gen_tokens,
    const std::vector<int> &eos_ids, MuseDenseGenerationResult &out,
    MuseDenseKvCache &kv,
    lsxcommon::sample::ISampler &sampler = lsxcommon::sample::DefaultSampler(),
    const lsxcommon::GenerationStream *gs = nullptr);

} // namespace muse_dense
} // namespace lsxcommon