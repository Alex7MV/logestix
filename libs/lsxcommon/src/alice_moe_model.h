#pragma once

#include <memory>
#include <string>
#include <vector>

#include "lsxcommon/alice_moe_load.h"
#include "lsxcommon/alice_moe_inference_pipeline.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

// AliceAI-Foundation-80B-A3B-Base (kType "alice-foundation-80b-a3b"): hybrid
// KDA + gated-attention sparse MoE with block attention-residual mixing.
// Frozen geometry lives in src/alice_moe_constants.h (passport,
// model-independence invariant); Load cross-checks the converted index.
class AliceFoundation80BModel final : public ModelBase {
public:
    static const std::string kType;  // "alice-foundation-80b-a3b"

    explicit AliceFoundation80BModel(ModelInitConfig cfg);
    ~AliceFoundation80BModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    // Model Ingestion Protocol step 4b: native AliceAI conversation/tool codec.
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

    const std::string &Type() const override { return kType; }

protected:
    void TeardownWeights() override;

    alice::AliceMoeTopology top_;
    alice::AlicePinnedWeights pw_;
    std::unique_ptr<alice::AliceExpertHost> experts_;
    bool weights_loaded_ = false;
};

} // namespace lsxcommon
