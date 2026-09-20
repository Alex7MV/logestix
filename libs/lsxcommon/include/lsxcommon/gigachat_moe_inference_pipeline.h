#pragma once

// GigaChat3.5 40-layer hybrid decode pipeline (Plan 2b). Walks the layers with
// the `pre_post` residual structure, dispatching MLA or GatedDeltaNet per layer
// and MoE / dense MLP per layer, then final norm + lm_head (F32 logits).
//
// Host-owned session state: MLA KV caches (full-attention layers), GDN
// conv/recurrent state (linear layers), the YaRN tables and the step scratch.
// Routed experts stream through a caller-provided GcExpertCache.

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "lsxcommon/gigachat_moe_expert_cache.h"
#include "lsxcommon/gigachat_moe_inference_attn.h"
#include "lsxcommon/gigachat_moe_load.h"

namespace lsxcommon::gigachat {

struct GcPipelineDims {
    GcMlaDims mla;
    GcGdnDims gdn;
    int vocab = 0;
    int num_experts = 0;
    int top_k = 0;
    int moe_intermediate = 0;
    int first_k_dense = 0;
    float rms_eps = 1e-6f;
    float routed_scale = 2.5f;
    float swiglu_limit = 10.0f;
};

// Builds the interleaved YaRN cos/sin tables [capacity, rope_dim/2] from the
// frozen passport (theta 1e5, factor 8, original 32768, beta 32/1).
bool GcBuildYarnTables(int capacity, int rope_dim, std::vector<float> &cos_t,
                       std::vector<float> &sin_t);

class GcPipeline {
public:
    GcPipeline();
    ~GcPipeline();
    GcPipeline(const GcPipeline &) = delete;
    GcPipeline &operator=(const GcPipeline &) = delete;

    bool Init(const GcWeights &w, const GcPipelineDims &dims, int capacity,
              std::string &error);
    void Free();
    bool ok() const;

    // Zeroes every KV / conv / recurrent state for a new sequence.
    cudaError_t Reset(cudaStream_t stream);

    // Runs one decode token at position `pos` and writes F32 logits [vocab].
    cudaError_t Step(const GcWeights &w, GcExpertCache &ec, int token_id,
                     int pos, float *logits, cudaStream_t stream);

private:
    struct Impl;
    Impl *impl_ = nullptr;
};

} // namespace lsxcommon::gigachat
