#pragma once

// GigaChat4-37B-A4B model (kType "gigachat4-37b-a4b"). Hybrid GatedDeltaNet +
// DSA MLA + MoE, standalone ModelBase subclass. The frozen geometry passport is
// compile-time locked (src/gigachat_moe_37b_constants.h); Load cross-checks the
// converted index against it and fails on mismatch.
//
// The decode pipeline (DSA indexer + logit softcap) is wired in the shared
// family TUs gigachat_moe_inference_pipeline.cu / _attn.cu; this class owns the
// session state, the expert slot cache and the Infer loop.

#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/gigachat_moe_expert_cache.h"
#include "lsxcommon/gigachat_moe_inference_pipeline.h"
#include "lsxcommon/gigachat_moe_37b_load.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

class GigaChat4MoeModel final : public ModelBase {
public:
    static const std::string kType;  // "gigachat4-37b-a4b"

    explicit GigaChat4MoeModel(ModelInitConfig cfg);
    ~GigaChat4MoeModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }

    // Prompt framing (stage 4): chatml-v5 single turn with a prefilled empty
    // <think></think>, ids resolved from the checkpoint tokenizer.
    bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                             std::string_view raw_prompt,
                             std::vector<int32_t> &out_ids) const override;

    // Tool/function-calling bridge (stage 4b): GigaChat chatml-v5 GCML calls.
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

    // Test/diagnostics access to the parsed passport.
    const gigachat4::Gc4Index &IndexForTest() const { return idx_; }

private:
    void TeardownWeights() override;
    bool BuildPipelineDims(gigachat::GcPipelineDims &dims) const;

    gigachat4::Gc4Index idx_;
    gigachat::GcWeights weights_;
    gigachat::GcExpertCache expert_cache_;
    gigachat::GcPipeline pipeline_;
    float *d_logits_ = nullptr;
    int max_seq_ = 0;  // pipeline KV/state capacity (prompt + generated)
};

} // namespace lsxcommon
