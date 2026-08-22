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

} // namespace gemma
} // namespace lsxcommon
