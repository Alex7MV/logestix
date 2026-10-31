#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "lsxcommon/granite_dense_switch_constants.h"
#include "lsxcommon/granite_dense_switch_load.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/token_stream.h"

namespace lsxcommon {
namespace granite_switch {

// Host-side per-request preparation (Model Ingestion Protocol step 4).
//
// The reference switch (counting head + Kerdock memory) resolves to a prefix
// scan over control tokens: every position takes the adapter of the most recent
// control token at or before it, or `carry_adapter` when none. Token exchange
// rewrites control ids to their substitute ids before embedding.
struct GraniteSwitchPrepared {
    std::vector<int32_t> embed_ids;   // control ids replaced by substitutes
    std::vector<int32_t> adapter_ids; // 0 = base, 1..N = adapter
    int last_adapter = 0;             // carry for the next request / decode step
};

GraniteSwitchPrepared PrepareGraniteSwitchTokens(
    const GraniteSwitchTopology &top, const std::vector<int32_t> &ids,
    int carry_adapter);

// Returns the substitute id for a control token, or the id itself.
int32_t GraniteSwitchSubstitute(const GraniteSwitchTopology &top, int32_t id);

struct GraniteSwitchGenerationResult {
    bool ok = false;
    std::string error_msg;
    std::vector<int32_t> tokens;
    int32_t first_token = -1;
    std::vector<float> final_logits;
    double ttft_ms = 0.0;
    double total_ms = 0.0;
    std::uint64_t resident_bytes = 0;
    bool eos_terminated = false;
    int last_adapter = 0;
};

// Persistent per-session GQA KV cache (one row per decoder layer). The switch
// adapter carried by the session is stored beside `last_tok`: a continuation
// request replays only {last_tok}, so the adapter state must survive the call.
struct GraniteSwitchKvCache {
    __half *d_k = nullptr;
    __half *d_v = nullptr;
    int32_t last_tok = -1;
    int last_adapter = 0;
    int kv_rows = 0;
    int max_total = 0;
    int64_t cache_bytes = 0;

    bool allocated() const { return d_k != nullptr; }
    std::uint64_t Bytes() const {
        return static_cast<std::uint64_t>(cache_bytes);
    }
    cudaError_t Allocate(const GraniteSwitchPinnedWeights &pw, int max_total,
                         cudaStream_t stream);
    cudaError_t EnsureCapacity(const GraniteSwitchPinnedWeights &pw,
                               int needed, cudaStream_t stream);
    void Free();
};

// Greedy/sampled generation over the Granite-Switch dense loop. `kv == nullptr`
// runs a full recompute per step; a non-null cache selects the incremental
// path and carries the adapter state between calls.
cudaError_t RunGraniteSwitchGeneration(
    cudaStream_t stream, const GraniteSwitchPinnedWeights &pw,
    const std::vector<int32_t> &prompt, int max_new, int eos_id,
    GraniteSwitchGenerationResult &result,
    lsxcommon::sample::ISampler &sampler = lsxcommon::sample::DefaultSampler(),
    GraniteSwitchKvCache *kv = nullptr,
    const lsxcommon::GenerationStream *gs = nullptr);

// ── Testable low-level kernels (Model Ingestion Protocol step 5) ──────────
cudaError_t RunGraniteSwitchRopeKernel(cudaStream_t stream, __half *q,
                                       __half *k, const int32_t *pos, int seq,
                                       int nh, int nkv, int hd, double theta);
cudaError_t RunGraniteSwitchAttentionKernel(
    cudaStream_t stream, const __half *q, const __half *k, const __half *v,
    __half *out, int seq, int nh, int nkv, int hd, float scale, int kv_base);
cudaError_t RunGraniteSwitchAttentionSplitKernel(
    cudaStream_t stream, const __half *q, const __half *k, const __half *v,
    __half *out, int seq, int nh, int nkv, int hd, float scale, int kv_base);
cudaError_t RunGraniteSwitchRmsNormKernel(cudaStream_t stream, __half *out,
                                          const __half *x, const __half *w,
                                          int rows, int hidden, float eps);
cudaError_t RunGraniteSwitchSplitQKV(cudaStream_t stream,
                                     const __half *qkv, __half *q, __half *k,
                                     __half *v, int seq, int qn, int kn);
cudaError_t RunGraniteSwitchSwiGLU(cudaStream_t stream, const __half *gu,
                                   __half *out, int I, int64_t n);
cudaError_t RunGraniteSwitchScale(cudaStream_t stream, __half *x, int64_t n,
                                  float s);
cudaError_t RunGraniteSwitchScaleAdd(cudaStream_t stream, __half *dst,
                                     const __half *src, int64_t n, float s);
cudaError_t RunGraniteSwitchLoraShrink(cudaStream_t stream, const __half *x,
                                       const __half *A,
                                       const int32_t *d_adapter, __half *y,
                                       int seq, int K, int rank, int y_stride,
                                       int target = -1);
cudaError_t RunGraniteSwitchLoraExpandAdd(
    cudaStream_t stream, const __half *y, const __half *B,
    const int32_t *d_adapter, __half *out, int seq, int rank, int N,
    int out_stride, int out_off, int y_stride, int target = -1);

} // namespace granite_switch
} // namespace lsxcommon
