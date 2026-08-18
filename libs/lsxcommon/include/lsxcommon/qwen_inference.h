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

// Partial rotary (HF rotate_half pairing: dims j and j+rot/2 form a pair):
// rotates the first `rotary_dim` of `head_dim` dims per (seq,head) pair,
// pass-through the rest. q: [seq, q_heads, head_dim],
// k: [seq, k_heads, head_dim] (GQA: q_heads=NH, k_heads=NKV).
cudaError_t RunQwenPartialRope(
    cudaStream_t stream, const __half *q, const __half *k,
    const __half *cos, const __half *sin,
    __half *q_out, __half *k_out,
    int seq, int q_heads, int k_heads, int head_dim, int rotary_dim);

// Same-heads convenience overload (q_heads = k_heads = heads).
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

// Full attention layer: input RMSNorm -> qkv projections (NVFP4 dequant GEMM)
// -> q/k per-head norm -> partial rope -> GQA causal attention -> output gate
// -> o_proj. All buffers caller-provided (see RunQwenGeneration arena layout).
cudaError_t RunQwenFullAttention(
    cudaStream_t stream,
    const QwenLayerWeights &lw, const QwenTopology &top, int seq,
    const __half *d_normed, __half *d_out,
    __half *d_qc, __half *d_q, __half *d_gate,
    __half *d_k, __half *d_v, __half *d_attn,
    __half *d_cos, __half *d_sin, int32_t *d_pos);

// Linear (GatedDeltaNet) attention layer: in_proj qkv (NVFP4 dequant GEMM) ->
// causal conv1d -> gated delta rule (recurrent) -> RMSNormGated -> out_proj.
// Buffers caller-provided (halves unless noted):
//   d_qkv: [seq, KD*2+VD], d_z: [seq,VD], d_ab: [seq*nv*2] (a then b),
//   d_g: [seq*nv] float decay, d_beta: [seq*nv] float beta,
//   d_conv: [seq, KD*2+VD], d_core: [seq,VD], d_gated: [seq,VD].
cudaError_t RunQwenLinearAttention(
    cudaStream_t stream,
    const QwenLayerWeights &lw, const QwenTopology &top, int seq,
    const __half *d_normed, __half *d_out,
    __half *d_qkv, __half *d_z, __half *d_ab,
    float *d_g, float *d_beta,
    __half *d_conv, __half *d_core, __half *d_gated);

// Dense SwiGLU MLP: gate=silu(x@Wg), up=x@Wu, inter=gate*up, out=inter@Wd.
cudaError_t RunQwenMLP(
    cudaStream_t stream,
    const QwenLayerWeights &lw, int seq,
    const __half *d_in, __half *d_out, __half *d_scratch);

// ── Generation ──────────────────────────────────────────────────────────────

struct QwenGenerationResult {
    bool ok = false;
    std::string error_msg;
    std::vector<int32_t> tokens;
    int32_t first_token = -1;
    std::vector<float> final_logits;
    double ttft_ms = 0.0;
    double total_ms = 0.0;
    std::uint64_t resident_bytes = 0;
    bool eos_terminated = false;
};

// Full greedy multi-token generation over the frozen Qwen3.5-9B hybrid loop:
// per decoder layer, RMSNorm input -> (linear or full attention) -> residual ->
// RMSNorm -> MLP -> residual; final norm -> lm_head -> greedy argmax.
cudaError_t RunQwenGeneration(
    cudaStream_t stream,
    const QwenPinnedWeights &pw,
    const std::vector<int32_t> &prompt, int max_new,
    int eos_id,
    QwenGenerationResult &result);

} // namespace qwen
} // namespace lsxcommon