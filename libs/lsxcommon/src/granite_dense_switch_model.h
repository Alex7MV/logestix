#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "lsxcommon/granite_dense_switch_constants.h"
#include "lsxcommon/granite_dense_switch_inference.h"
#include "lsxcommon/granite_dense_switch_load.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

// Granite-Switch 4.1 30B (IBM): Granite 4.1 30B dense decoder with two
// embedded aLoRA adapters, control-token routing and token exchange. Frozen
// geometry lives in granite_dense_switch_constants.h.
class GraniteSwitch41_30BModel final : public ModelBase {
public:
    static constexpr int kHiddenSize =
        granite_switch::frozen::kHiddenSize;
    static constexpr int kNumLayers = granite_switch::frozen::kNumLayers;
    static constexpr int kNumHeads = granite_switch::frozen::kNumHeads;
    static constexpr int kNumKVHeads = granite_switch::frozen::kNumKVHeads;
    static constexpr int kHeadDim = granite_switch::frozen::kHeadDim;
    static constexpr int kIntermediateSize =
        granite_switch::frozen::kIntermediateSize;
    static constexpr int kVocabSize = granite_switch::frozen::kVocabSize;

    static_assert(kHiddenSize == kNumHeads * kHeadDim,
                  "hidden_size must equal num_attention_heads * head_dim");
    static_assert(kNumKVHeads > 0 && kNumKVHeads <= kNumHeads,
                  "num_key_value_heads must be in (0, num_attention_heads]");
    static_assert(kNumLayers > 0, "num_layers must be positive");

    explicit GraniteSwitch41_30BModel(ModelInitConfig cfg);
    ~GraniteSwitch41_30BModel() override;

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

    // Test/debug hook: adapter carried by the current incremental session.
    int LastAdapter() const { return kv_ ? kv_->last_adapter : 0; }

protected:
    bool ValidateGeometry(const granite_switch::GraniteSwitchTopology &top) const;
    void TeardownWeights() override;

    granite_switch::GraniteSwitchPinnedWeights pw_;
    bool weights_loaded_ = false;
    std::unique_ptr<granite_switch::GraniteSwitchKvCache> kv_;
    std::vector<int32_t> session_tail_;
};

} // namespace lsxcommon
