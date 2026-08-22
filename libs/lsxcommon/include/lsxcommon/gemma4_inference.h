#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/gemma4_inference_load.h"

namespace lsxcommon {
namespace gemma {

// ── Norms ───────────────────────────────────────────────────────────────────
// out[r] = rmsnorm(x[r]) * (1 + w), rows x hidden, FP16 storage / FP32 math.
cudaError_t RunGemmaRMSNorm(const __half *d_x, const __half *d_w, __half *d_out,
                            int rows, int hidden, float eps, cudaStream_t stream);

// ── RoPE ────────────────────────────────────────────────────────────────────
// cos/sin tables [seq][rotary_dim/2] for positions pos[t], exponent base theta.
cudaError_t RunGemmaRopeCosSin(const int32_t *d_pos, __half *d_cos,
                               __half *d_sin, int seq, int rotary_dim,
                               double theta, cudaStream_t stream);

// Partial rotary embedding over packed [seq, heads, head_dim]; rotates the
// first rotary_dim dims of every head with HF rotate_half pairing.
cudaError_t RunGemmaPartialRope(const __half *d_x, const __half *d_cos,
                                const __half *d_sin, __half *d_out, int seq,
                                int heads, int head_dim, int rotary_dim,
                                cudaStream_t stream);

// ── Attention ───────────────────────────────────────────────────────────────
// Causal attention over a KV cache. One kernel serves both layer kinds:
//   window > 0  -> sliding band mask (slot >= pos_base+t-window+1)
//   window <= 0 -> full causal context
// "Global" KV geometry (full-attention layers) is just kv_heads/global_head_dim
// parameters — the layout is identical.
//
// Layouts (stride invariant):
//   q/out row (t,h):  [(t*q_heads + h) * head_dim]
//   k/v  row (s,kvh): [(s*kv_heads + kvh) * head_dim], kvh = h/(q_heads/kv_heads)
//
// Determinism: fixed slot iteration order; block reductions are tree-shaped
// with __syncthreads() barriers and no read-modify-write races.
cudaError_t RunGemmaAttention(const __half *d_q, const __half *d_k,
                              const __half *d_v, __half *d_out, int seq,
                              int total_slots, int pos_base, int q_heads,
                              int kv_heads, int head_dim, int window,
                              float scale, cudaStream_t stream);

// ── Dense MLP + MoE ─────────────────────────────────────────────────────────
// out = down @ (gelu_tanh(gate @ x) * (up @ x)); weights [rows, hidden]
// row-major; x [seq, hidden]; out [seq, hidden].
cudaError_t RunGemmaDenseMLP(const __half *d_gate, const __half *d_up,
                             const __half *d_down, int64_t I, int64_t H,
                             const __half *d_x, __half *d_out, int seq,
                             cudaStream_t stream);

// Router: logits[e] = router_proj[e] . x; top-k selection with deterministic
// lower-index tie-break; weights = softmax(selected logits). When
// d_per_expert_scale is non-null each selected weight is additionally
// multiplied by per_expert_scale[e]. Outputs d_sel_experts [seq, top_k] and
// d_sel_weights [seq, top_k].
cudaError_t RunGemmaRouterTopK(const __half *d_router_proj,
                               const float *d_per_expert_scale,
                               const __half *d_x, int num_experts,
                               int hidden, int top_k, int32_t *d_sel_experts,
                               float *d_sel_weights, int seq,
                               cudaStream_t stream);

// Weighted expert-output combine: out[h] = router_scale[h] *
// sum_k w_k * expert_out_k[h]. Null d_router_scale skips the channel scale.
cudaError_t RunGemmaMoECombine(const __half *const *d_expert_outs,
                               const float *d_weights, const float *d_router_scale,
                               __half *d_out, int hidden, int k, cudaStream_t stream);

// ── Generation ──────────────────────────────────────────────────────────────
struct GemmaGenerationResult {
    bool ok = false;
    std::string error_msg;
    std::vector<int32_t> tokens;
    std::vector<float> final_logits;   // last decode step (softcapped)
    double ttft_ms = 0.0;
    double total_ms = 0.0;
    bool eos_terminated = false;
};

// Full generation loop: prefill(input) + up to max_gen_tokens decode steps on
// ONE stream (no cudaDeviceSynchronize in the loop; per-token D2H argmax via
// pinned handshake). kv must be pre-allocated for prompt+gen capacity.
// Applies final RMSNorm, tied-head logits and softcap tanh(l/cap)*cap.
cudaError_t RunGemmaGeneration(cudaStream_t stream, const GemmaPinnedWeights &pw,
                               GemmaExpertDb &db, const std::vector<int32_t> &input_ids,
                               int max_gen_tokens, int eos_id, float softcap,
                               GemmaGenerationResult &out, GemmaKvCache &kv);

} // namespace gemma
} // namespace lsxcommon
