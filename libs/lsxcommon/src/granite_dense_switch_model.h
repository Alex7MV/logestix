#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "lsxcommon/granite_dense_switch_8b_constants.h"
#include "lsxcommon/granite_dense_switch_constants.h"
#include "lsxcommon/granite_dense_switch_inference.h"
#include "lsxcommon/granite_dense_switch_load.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

// Granite-Switch 4.1 (IBM): a Granite 4.1 dense decoder with two embedded aLoRA
// adapters, control-token routing and token exchange.
//
// ONE converter, ONE set of kernels and ONE lifecycle body serve two
// checkpoints (30B and 8B). They differ in depth, FFN width, rope theta and the
// residual multiplier — four fields, three of which are arithmetic. So the
// implementation takes its frozen expectations as data (`GraniteSwitchFrozen`)
// and each concrete class passes its own passport:
//
//   GraniteSwitch41_30BModel -> granite_switch::frozen
//   GraniteSwitch41_8BModel  -> granite_switch_8b::frozen
//
// A shared copy of those numbers is what would make the sibling decode a
// plausible but wrong model, so `ValidateGeometry` compares the converted tree
// against the instance's passport (see GraniteSwitchGeometryMismatch) instead of
// against a namespace.
class GraniteSwitchDenseModel : public ModelBase {
public:
    explicit GraniteSwitchDenseModel(ModelInitConfig cfg,
                                     granite_switch::GraniteSwitchFrozen frozen);

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

    // Test/debug hook: adapter carried by the current incremental session.
    int LastAdapter() const { return kv_ ? kv_->last_adapter : 0; }

protected:
    bool ValidateGeometry(const granite_switch::GraniteSwitchTopology &top) const;
    void TeardownWeights() override;

    granite_switch::GraniteSwitchFrozen frozen_;
    granite_switch::GraniteSwitchPinnedWeights pw_;
    bool weights_loaded_ = false;
    std::unique_ptr<granite_switch::GraniteSwitchKvCache> kv_;
    std::vector<int32_t> session_tail_;
};

// The 30B: 64 decoder layers, intermediate 32768, theta 5e7, residual 0.175.
class GraniteSwitch41_30BModel final : public GraniteSwitchDenseModel {
public:
    static constexpr int kHiddenSize = granite_switch::frozen::kHiddenSize;
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

    const std::string &Type() const override { return kType; }
    static const std::string kType;
};

// The 8B: 40 decoder layers, intermediate 12800, theta 1e7, residual 0.22.
class GraniteSwitch41_8BModel final : public GraniteSwitchDenseModel {
public:
    static constexpr int kHiddenSize = granite_switch_8b::frozen::kHiddenSize;
    static constexpr int kNumLayers = granite_switch_8b::frozen::kNumLayers;
    static constexpr int kNumHeads = granite_switch_8b::frozen::kNumHeads;
    static constexpr int kNumKVHeads = granite_switch_8b::frozen::kNumKVHeads;
    static constexpr int kHeadDim = granite_switch_8b::frozen::kHeadDim;
    static constexpr int kIntermediateSize =
        granite_switch_8b::frozen::kIntermediateSize;
    static constexpr int kVocabSize = granite_switch_8b::frozen::kVocabSize;

    static_assert(kHiddenSize == kNumHeads * kHeadDim,
                  "hidden_size must equal num_attention_heads * head_dim");
    static_assert(kNumKVHeads > 0 && kNumKVHeads <= kNumHeads,
                  "num_key_value_heads must be in (0, num_attention_heads]");
    static_assert(kNumLayers > 0, "num_layers must be positive");

    explicit GraniteSwitch41_8BModel(ModelInitConfig cfg);
    ~GraniteSwitch41_8BModel() override;

    const std::string &Type() const override { return kType; }
    static const std::string kType;
};

} // namespace lsxcommon
