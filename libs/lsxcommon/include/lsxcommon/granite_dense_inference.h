#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "lsxcommon/granite_dense_constants.h"
#include "lsxcommon/granite_dense_load.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/token_stream.h"

namespace lsxcommon {
namespace granite {

// ── Testable low-level kernels (Model Ingestion Protocol step 5) ───────────
// Exposed for kernel-level parity tests; the generation loop uses them too.
cudaError_t RunGraniteRopeKernel(cudaStream_t stream, __half *q, __half *k,
                                 const int32_t *pos, int seq, int nh, int nkv,
                                 int hd, double theta);
cudaError_t RunGraniteAttentionKernel(cudaStream_t stream, const __half *q,
                                      const __half *k, const __half *v,
                                      __half *out, int seq, int nh, int nkv,
                                      int hd, float scale, int kv_base);
// Forced split-KV (flash-decoding) variant: always routes through the
// multi-block + combine path regardless of shape. Used by the parity test to
// exercise the split boundaries; production dispatch is in GraniteAttention.
cudaError_t RunGraniteAttentionSplitKernel(cudaStream_t stream, const __half *q,
                                           const __half *k, const __half *v,
                                           __half *out, int seq, int nh, int nkv,
                                           int hd, float scale, int kv_base);
cudaError_t RunGraniteRmsNormKernel(cudaStream_t stream, __half *out,
                                    const __half *x, const __half *w, int rows,
                                    int hidden, float eps);

struct GraniteGenerationResult {
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

// Persistent per-session GQA KV cache. Full-attention k/v rows live in
// [num_layers][max_total][NKV*HD] buffers; the decode path appends one row per
// step at kv_rows. Fresh sessions allocate (kv_rows == 0); continuation calls
// resume at kv->kv_rows with prompt = {kv->last_tok}.
struct GraniteKvCache {
    __half *d_k = nullptr;  // [num_layers][max_total][NKV*HD]
    __half *d_v = nullptr;
    int32_t last_tok = -1;
    int kv_rows = 0;
    int max_total = 0;
    int64_t cache_bytes = 0;

    bool allocated() const { return d_k != nullptr; }
    std::uint64_t Bytes() const {
        return static_cast<std::uint64_t>(cache_bytes);
    }

    cudaError_t Allocate(const GranitePinnedWeights &pw, int max_total,
                         cudaStream_t stream);
    cudaError_t EnsureCapacity(const GranitePinnedWeights &pw, int needed,
                               cudaStream_t stream);
    void Free();
};

// Greedy multi-token generation over the frozen Granite dense loop:
//   per layer: RMSNorm -> q/k/v (NVFP4) -> full RoPE -> causal GQA attention
//   (scaled by attention_multiplier) -> o (NVFP4) -> residual -> RMSNorm ->
//   SwiGLU SwiGLU (NVFP4) -> residual; final norm -> lm_head -> argmax.
// A non-null `kv` selects the incremental path (one token per step at the
// persistent kv_rows); nullptr runs a full recompute per step.
cudaError_t RunGraniteGeneration(
    cudaStream_t stream, const GranitePinnedWeights &pw,
    const std::vector<int32_t> &prompt, int max_new, int eos_id,
    GraniteGenerationResult &result,
    lsxcommon::sample::ISampler &sampler = lsxcommon::sample::DefaultSampler(),
    GraniteKvCache *kv = nullptr,
    const lsxcommon::GenerationStream *gs = nullptr);

} // namespace granite
} // namespace lsxcommon
