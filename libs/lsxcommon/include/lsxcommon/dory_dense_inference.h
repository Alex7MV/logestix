#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "lsxcommon/dory_dense_constants.h"
#include "lsxcommon/dory_dense_load.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/token_stream.h"

namespace lsxcommon {
namespace dory {

// ── Testable low-level kernels (Model Ingestion Protocol stage 5) ──────────
// Exposed for kernel-level parity tests; the generation loop uses them too.

// out[r][i] = x[r][i] * rsqrt(sum_i x[r][i]^2 + eps)      (nGPT justnorm)
cudaError_t RunDoryJustNormKernel(cudaStream_t stream, __half *out,
                                  const __half *x, int rows, int n, float eps);

// out[r] = justnorm(a[r] + b[r])  (the loop's reinjection combine).
cudaError_t RunDoryJustNormSumKernel(cudaStream_t stream, __half *out,
                                     const __half *a, const __half *b, int rows,
                                     int n, float eps);

// out = justnorm( justnorm(a) + |alpha|*(justnorm(b) - justnorm(a)) ), the nGPT
// spherical residual. `alpha` is the pre-scaled per-dimension vector.
cudaError_t RunDorySlerpAlphaKernel(cudaStream_t stream, __half *out,
                                    const __half *a, const __half *b,
                                    const float *alpha, int rows, int n,
                                    float eps);

// q: per-head justnorm then multiply by sqk2 (pre-squared nGPT q scale);
// k: per-head justnorm only. `q`/`k` are [rows, nh*hd] / [rows, nkv*hd].
cudaError_t RunDoryQkNormKernel(cudaStream_t stream, __half *q, __half *k,
                                const float *sqk2, int rows, int nh, int nkv,
                                int hd, float eps);

// out = justnorm(attn * sigmoid(gate * sgate2)) over the full head stack.
cudaError_t RunDoryAttnGateNormKernel(cudaStream_t stream, __half *out,
                                      const __half *attn, const __half *gate,
                                      const float *sgate2, int rows, int q_size,
                                      float eps);

// out[r][i] = justnorm_i( silu(gu[r][i]*suv_gate[i]) * gu[r][I+i]*suv_up[i] ).
cudaError_t RunDorySuvSwiGLUNormKernel(cudaStream_t stream, __half *out,
                                       const __half *gu, const float *suv_gate,
                                       const float *suv_up, int rows, int I,
                                       float eps);

// Causal GQA attention over a persistent KV cache at `kv_base`, with an optional
// sliding window (`window <= 0` = full attention). The neutral dense attention
// layer has no window predicate, which is why Dory keeps its own.
cudaError_t RunDoryAttentionKernel(cudaStream_t stream, const __half *q,
                                   const __half *k, const __half *v,
                                   __half *out, float *partials, int seq,
                                   int nh, int nkv, int hd, float scale,
                                   int kv_base, int window);

// logits[v] = half2float(y[v]) * logit_scale[v]   (nGPT per-vocab scale).
cudaError_t RunDoryLogitScaleKernel(cudaStream_t stream, float *logits,
                                    const __half *y, const float *logit_scale,
                                    int vocab);

struct DoryGenerationResult {
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

// Persistent per-session KV cache: one row of [num_kv_heads*head_dim] per slot
// per position. `per_loop` caching gives a recurrent attention layer one slot
// per loop (DoryTopology::recurrent_loops), so the cache count is
// DoryPinnedWeights::kv_slots — not the attention-layer count.
struct DoryKvCache {
    __half *d_k = nullptr;   // [slots][max_total][kv_size]
    __half *d_v = nullptr;
    int32_t last_tok = -1;
    int kv_rows = 0;
    int max_total = 0;
    int64_t cache_bytes = 0;

    bool allocated() const { return d_k != nullptr; }
    std::uint64_t Bytes() const { return static_cast<std::uint64_t>(cache_bytes); }

    cudaError_t Allocate(const DoryPinnedWeights &pw, int max_total,
                         cudaStream_t stream);
    cudaError_t EnsureCapacity(const DoryPinnedWeights &pw, int needed,
                               cudaStream_t stream);
    void Free();
};

// Greedy (or sampled) multi-token generation over the frozen Dory graph:
//   input block -> {justnorm(state + input_h), recurrent block, justnorm} x loops
//   -> output block -> lm head -> per-vocab logit scale -> sample.
// A non-null `kv` selects the incremental path (one token per step appended at
// the persistent kv_rows); nullptr runs a full recompute per step.
cudaError_t RunDoryGeneration(
    cudaStream_t stream, const DoryPinnedWeights &pw,
    const std::vector<int32_t> &prompt, int max_new, int eos_id,
    DoryGenerationResult &result,
    lsxcommon::sample::ISampler &sampler = lsxcommon::sample::DefaultSampler(),
    DoryKvCache *kv = nullptr,
    const lsxcommon::GenerationStream *gs = nullptr);

} // namespace dory
} // namespace lsxcommon
