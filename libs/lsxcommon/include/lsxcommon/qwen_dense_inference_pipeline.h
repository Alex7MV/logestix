#pragma once
// Qwen-3.x generation driver + persistent KV session. Model-owned:
// RunQwenGeneration, QwenKvCache and QwenGenerationResult all take
// QwenPinnedWeights, so they stay family-named even though the attention/MLP
// kernels they call are the neutral layer (see lsx_attn_inference.h).
//
// Named "dense" to satisfy the repo file-name scheme (test_code_naming_guard),
// not because it is dense-specific: the hybrid linear+full-attention QwenKvCache
// is also used by the MoE families, which include this header.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "lsxcommon/lsx_attn_inference.h"        // kernel entry points
#include "lsxcommon/qwen_inference_load.h"   // QwenPinnedWeights
#include "lsxcommon/sampling.h"
#include "lsxcommon/token_stream.h"

namespace lsxcommon {
namespace qwen {

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

// Persistent per-session incremental KV cache (spec item 15). A non-null
// `kv` handed to RunQwenGeneration turns the generation loop into a uniform
// one-token-per-step stream: fresh sessions allocate (kv_rows == 0), later
// calls continue at kv->kv_rows with prompt = {kv->last_tok}. Full-attention
// k/v rows live in [max_total][NKV*HD] buffers (one per full-attn layer);
// linear-attention layers keep their raw qkv history [max_total][KD*2+VD]
// (for the causal conv1d) plus the persistent GatedDeltaNet col accumulator
// [nv*vd*kd] floats. Growth is handled by EnsureCapacity.
struct QwenKvCache {
    __half *d_kv_k = nullptr;        // [num_full][max_total][NKV*HD]
    __half *d_kv_v = nullptr;        // [num_full][max_total][NKV*HD]
    __half *d_qkv = nullptr;         // [num_linear][max_total][KD*2+VD]
    float  *d_col = nullptr;         // [num_linear][nv*vd*kd]  (zero-initialized)
    int32_t last_tok = -1;
    int kv_rows = 0;                 // rows written so far (== global pos base)
    int max_total = 0;
    int num_linear_layers = 0;
    int num_full_layers = 0;
    int64_t cache_bytes = 0;

    bool allocated() const { return d_kv_k != nullptr; }
    std::uint64_t Bytes() const { return static_cast<std::uint64_t>(cache_bytes); }

    cudaError_t Allocate(const QwenPinnedWeights &pw, int max_total,
                         cudaStream_t stream);
    cudaError_t EnsureCapacity(const QwenPinnedWeights &pw, int needed,
                               cudaStream_t stream);
    void Free();
};

// Full greedy multi-token generation over the frozen Qwen3.5-9B hybrid loop:
// per decoder layer, RMSNorm input -> (linear or full attention) -> residual ->
// RMSNorm -> MLP -> residual; final norm -> lm_head -> greedy argmax.
// A non-null `kv` selects the incremental path (spec item 15): each step runs
// exactly one token at the persistent kv_base and the recurrent GatedDeltaNet
// state is carried across calls; `nullptr` keeps today's full-recompute path.
cudaError_t RunQwenGeneration(
    cudaStream_t stream,
    const QwenPinnedWeights &pw,
    const std::vector<int32_t> &prompt, int max_new,
    int eos_id,
    QwenGenerationResult &result,
    lsxcommon::sample::ISampler &sampler =
        lsxcommon::sample::DefaultSampler(),
    QwenKvCache *kv = nullptr,
    const lsxcommon::GenerationStream *gs = nullptr);

} // namespace qwen
} // namespace lsxcommon
