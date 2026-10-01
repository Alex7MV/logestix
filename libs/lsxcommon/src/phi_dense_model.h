#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "lsxcommon/phi_dense_constants.h"
#include "lsxcommon/phi_dense_inference.h"
#include "lsxcommon/phi_dense_load.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

// phi-4 (Microsoft): dense decoder-only transformer, 40 layers / 5120
// hidden / 40 Q / 10 KV heads / 17920 intermediate, SwiGLU MLP, full-dim
// RoPE, attention scaled by 1/sqrt(head_dim). Frozen geometry lives in
// phi_dense_constants.h (passport, model-independence invariant).
class Phi4Model final : public ModelBase {
public:
    static constexpr int kHiddenSize = phi::frozen::kHiddenSize;
    static constexpr int kNumLayers = phi::frozen::kNumLayers;
    static constexpr int kNumHeads = phi::frozen::kNumHeads;
    static constexpr int kNumKVHeads = phi::frozen::kNumKVHeads;
    static constexpr int kHeadDim = phi::frozen::kHeadDim;
    static constexpr int kIntermediateSize =
        phi::frozen::kIntermediateSize;
    static constexpr int kVocabSize = phi::frozen::kVocabSize;

    static_assert(kHiddenSize == kNumHeads * kHeadDim,
                  "hidden_size must equal num_attention_heads * head_dim");
    static_assert(kNumKVHeads > 0 && kNumKVHeads <= kNumHeads,
                  "num_key_value_heads must be in (0, num_attention_heads]");
    static_assert(kNumLayers > 0, "num_layers must be positive");

    explicit Phi4Model(ModelInitConfig cfg);
    ~Phi4Model() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }
    static const std::string kType;

protected:
    bool ValidateGeometry(const phi::PhiTopology &top) const;
    void TeardownWeights() override;

    phi::PhiPinnedWeights pw_;
    bool weights_loaded_ = false;
    std::unique_ptr<phi::PhiKvCache> kv_;
    std::vector<int32_t> session_tail_;
};

} // namespace lsxcommon
