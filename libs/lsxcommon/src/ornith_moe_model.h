#pragma once

// Ornith-1.5-35B-A3B MoE model (kType "ornith-1.5-35b-a3b"). Own family: the
// MoE runtime layer is a duplicate of the qwen_moe family; the generic
// qwen_inference kernel layer is shared (option 1b). Own frozen-geometry
// passport (src/ornith_moe_constants.h) and its own Infer fast-math scope
// (model independence invariant).

#include <memory>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/model_base.h"
#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/qwen_inference_load.h"
#include "lsxcommon/expert_device_cache.h"
#include "lsxcommon/lsx_moe_inference.h"
#include "lsxcommon/ornith_moe_inference_pipeline.h"
#include "lsxcommon/ornith_moe_load.h"

namespace lsxcommon {

class OrnithMoeModelBase : public ModelBase {
public:
    // Test/diagnostics access to persistent generation scratch.
    const ornith::OrnithMoeGenScratch &GenScratchForTest() const
    {
        return scratch_;
    }

    bool Load() override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    // Tool/function-calling bridge (step 4b): Ornith is a Qwen3.5-MoE fine-tune
    // and uses the shared Qwen ChatML/XML codec (qwen_tool_format.h).
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

protected:
    explicit OrnithMoeModelBase(ModelInitConfig cfg);

    void TeardownWeights() override;

    // Runtime geometry gate: parsed index vs the leaf model's frozen passport.
    virtual bool ValidateGeometry(
        const ornith::OrnithMoeTopology &top) const = 0;

    // Shared Infer body. The caller MUST have opened its own
    // LsxFastMathScope(EffectiveFastMathMode(cfg_)) first (per-model policy).
    gpu::InferenceResult RunInferShared(const ModelRequest &req);

    ornith::OrnithMoePinnedWeights pw_;
    moe::MoEExpertStore experts_;
    ornith::OrnithMoeExpertDeviceCache cache_{};
    ornith::OrnithMoeGenScratch scratch_{};
    std::unique_ptr<qwen::QwenKvCache> kv_;
    std::vector<int32_t> session_tail_;
};

class OrnithMoeModel final : public OrnithMoeModelBase {
public:
    static const std::string kType;  // "ornith-1.5-35b-a3b"

    explicit OrnithMoeModel(ModelInitConfig cfg);
    ~OrnithMoeModel() override;

    gpu::InferenceResult Infer(const ModelRequest &req) override;

    // Ornith is a reasoning model: the control SelfCheck frames the prompt with
    // its own ChatML template and a prefilled empty think block so the answer
    // is produced directly (Model Ingestion Protocol step 8; recorded as a
    // per-model framing deviation).
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    // Own chat framing: ChatML single turn + prefilled `<think>...</think>`
    // (enable_thinking=false semantics of the checkpoint's chat template).
    bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                             std::string_view raw_prompt,
                             std::vector<int32_t> &out_ids) const override;

    const std::string &Type() const override { return kType; }

private:
    bool ValidateGeometry(const ornith::OrnithMoeTopology &top) const override;
};

} // namespace lsxcommon
