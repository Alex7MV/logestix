#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>

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

} // namespace gemma
} // namespace lsxcommon
