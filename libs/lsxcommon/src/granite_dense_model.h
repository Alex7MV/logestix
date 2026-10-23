#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "lsxcommon/granite_dense_constants.h"
#include "lsxcommon/granite_dense_inference.h"
#include "lsxcommon/granite_dense_load.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

// Granite-4.2-30B (IBM): dense decoder-only transformer, 64 layers / 4096
// hidden / 32 Q / 8 KV heads / 32768 intermediate, SwiGLU MLP, full-dim RoPE,
// fixed attention_multiplier. Frozen geometry lives in
// granite_dense_constants.h (passport, model-independence invariant).
class Granite42_30BModel final : public ModelBase {
public:
    static constexpr int kHiddenSize = granite::frozen::kHiddenSize;
    static constexpr int kNumLayers = granite::frozen::kNumLayers;
    static constexpr int kNumHeads = granite::frozen::kNumHeads;
    static constexpr int kNumKVHeads = granite::frozen::kNumKVHeads;
    static constexpr int kHeadDim = granite::frozen::kHeadDim;
    static constexpr int kIntermediateSize =
        granite::frozen::kIntermediateSize;
    static constexpr int kVocabSize = granite::frozen::kVocabSize;

    static_assert(kHiddenSize == kNumHeads * kHeadDim,
                  "hidden_size must equal num_attention_heads * head_dim");
    static_assert(kNumKVHeads > 0 && kNumKVHeads <= kNumHeads,
                  "num_key_value_heads must be in (0, num_attention_heads]");
    static_assert(kNumLayers > 0, "num_layers must be positive");

    explicit Granite42_30BModel(ModelInitConfig cfg);
    ~Granite42_30BModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

    const std::string &Type() const override { return kType; }
    static const std::string kType;

protected:
    bool ValidateGeometry(const granite::GraniteTopology &top) const;
    void TeardownWeights() override;

    granite::GranitePinnedWeights pw_;
    bool weights_loaded_ = false;
    std::unique_ptr<granite::GraniteKvCache> kv_;
    std::vector<int32_t> session_tail_;
};

} // namespace lsxcommon
