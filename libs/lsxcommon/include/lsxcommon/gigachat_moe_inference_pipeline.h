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

#include "lsxcommon/gigachat_moe_37b_inference_indexer.h"
#include "lsxcommon/gigachat_moe_expert_cache.h"
#include "lsxcommon/gigachat_moe_inference_attn.h"
#include "lsxcommon/gigachat_moe_load.h"

namespace lsxcommon::gigachat {

// RoPE / YaRN parameters of one variant. The defaults are GigaChat3.5's
// frozen values, so a caller that does not set them keeps the sibling's exact
// tables; GigaChat4 sets its own (theta 1e5, factor 32, orig 32768).
struct GcYarnParams {
    double theta = 100000.0;
    double factor = 8.0;
    double orig = 32768.0;
    double beta_fast = 32.0;
    double beta_slow = 1.0;
};

// ── Variant knobs ──────────────────────────────────────────────────────────
// One pipeline serves both GigaChat checkpoints. Everything a model does
// differently is data here, never a compile-time branch: the norm scale offset
// (3.5 `1+w`, 4.0 `w`), the GDN output-norm offset, the logit soft cap, the
// YaRN parameters and the attention scale, and whether the full-attention
// layers run the DSA lightning indexer.
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

    // Variant knobs (defaults == GigaChat3.5).
    float norm_w_offset = 1.0f;
    float gdn_w_offset = 1.0f;
    // Which state the GatedNorm's rank-16 gate consumes: false = the normalized
    // row (3.5), true = the row already scaled by (offset+w) (4.0).
    bool norm_gate_from_scaled = false;
    float logit_softcap = 0.0f;  // <= 0 disables
    float attn_scale = 0.0f;     // <= 0 derives the 3.5 YaRN value
    GcYarnParams yarn;

    // DSA lightning indexer (0 = every full-attention layer is dense).
    int index_heads = 0;
    int index_head_dim = 0;
    int index_topk = 0;
    int index_rope = 0;
    // Passport values the DSA step consumes; they live here so this TU never
    // reads another member's constants header. The defaults are the values both
    // shipped members happen to share (128^-0.5 and eps 1e-6).
    float index_onorm_eps = 1e-6f;
    float index_score_scale = 0.08838834764831845f;
    // GDN output-norm sigmoid gate scale (passport: linear_sigmoid_gate_scale).
    float gdn_sigmoid_gate_scale = 2.0f;
};

// Builds the interleaved YaRN cos/sin tables [capacity, rope_dim/2] from the
// frozen passport (theta 1e5, factor 8, original 32768, beta 32/1).
bool GcBuildYarnTables(int capacity, int rope_dim, std::vector<float> &cos_t,
                       std::vector<float> &sin_t);

// Same, for the parameters of one variant.
bool GcBuildYarnTables(const GcYarnParams &p, int capacity, int rope_dim,
                       std::vector<float> &cos_t, std::vector<float> &sin_t);

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
