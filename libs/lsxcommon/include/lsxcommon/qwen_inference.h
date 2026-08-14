#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/qwen_inference_load.h"

namespace lsxcommon {
namespace qwen {

// ── Numeric kernels (TDD-testable, FP16 in / FP16 or FP32 out) ──────────────

// RMSNorm, offset-weight form: out = rmsnorm(x) * (1 + w), eps=1e-6.
// x: [rows, hidden], w: [hidden], out: [rows, hidden].
cudaError_t RunQwenRMSNorm(
    cudaStream_t stream, const __half *x, const __half *w,
    __half *out, int rows, int hidden, double eps);

// Computes cos/sin for partial rotary on the first `rotary_dim` dims.
// positions: [seq] int32, cos/sin: [seq, rotary_dim] FP16.
cudaError_t RunQwenRopeCosSin(
    cudaStream_t stream, const int32_t *positions, int seq, int rotary_dim,
    double theta, __half *cos, __half *sin);

// Partial rotary: rotates the first `rotary_dim` of `head_dim` dims per
// (seq,head) pair, pass-through the rest. q,k: [seq, heads, head_dim].
cudaError_t RunQwenPartialRope(
    cudaStream_t stream, const __half *q, const __half *k,
    const __half *cos, const __half *sin,
    __half *q_out, __half *k_out,
    int seq, int heads, int head_dim, int rotary_dim);

// ── Inference entry points (operate on QwenPinnedWeights) ──────────────────

// Token embedding: out[t][h] = embed[ids[t]][h].
cudaError_t RunQwenEmbed(cudaStream_t stream, const __half *d_embed,
                         const int32_t *d_ids, __half *d_out,
                         int seq, int hidden);

// Full attention layer: RMSNorm input -> qkvo (NVFP4 dequant + cuBLAS) ->
// q/k per-head norm -> rope -> GQA scaled dot-product causal -> output gate ->
// o_proj. x/residual: [seq, hidden].
// Internal buffers d_q, d_k, d_v, d_o, d_attn are caller-provided scratch.
cudaError_t RunQwenFullAttention(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const QwenLayerWeights &lw, const QwenTopology &top, int seq,
    const __half *d_normed, __half *d_out,
    __half *d_scratch);

// Linear (GatedDeltaNet) attention layer: in_proj qkv -> causal conv1d ->
// gated delta rule (recurrent) -> RMSNormGated -> out_proj.
cudaError_t RunQwenLinearAttention(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const QwenLayerWeights &lw, const QwenTopology &top, int seq,
    const __half *d_normed, __half *d_out,
    __half *d_scratch);

// Dense SwiGLU MLP: gate=silu(x@Wg), up=x@Wu, inter=gate*up, out=inter@Wd.
cudaError_t RunQwenMLP(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const QwenLayerWeights &lw, int seq,
    const __half *d_in, __half *d_out, __half *d_scratch);

// ── Generation ──────────────────────────────────────────────────────────────

struct QwenGenerationResult {
    bool ok = false;
    std::string error_msg;
    std::vector<int32_t> tokens;  // greedy-argmax token ids, EOS-terminated
    float first_maxv = 0.0f;
    std::vector<float> final_logits;  // last-row logits over vocab (when computable)
};

// Full greedy multi-token generation over the frozen Qwen3.5-9B hybrid loop:
// per decoder layer, RMSNorm input -> (linear or full attention) -> residual ->
// RMSNorm -> MLP -> residual; final norm -> lm_head -> greedy argmax.
cudaError_t RunQwenGeneration(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const QwenPinnedWeights &pw,
    const std::vector<int32_t> &prompt, int max_new,
    QwenGenerationResult &result);

} // namespace qwen
} // namespace lsxcommon