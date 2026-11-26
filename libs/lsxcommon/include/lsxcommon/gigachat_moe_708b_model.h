#pragma once

// GigaChat4-708B-A41B model (kType "gigachat4-708b-a41b"). Hybrid
// GatedDeltaNet + DSA MLA + MoE at 64 layers / 61 MoE layers / 256 top-8
// experts, standalone ModelBase subclass. The frozen geometry passport is
// compile-time locked (src/gigachat_moe_708b_constants.h); Load cross-checks
// the converted index against it and fails on mismatch.
//
// The decode pipeline (DSA indexer + logit softcap) is the SHARED family TUs
// gigachat_moe_inference_pipeline.cu / _attn.cu / _37b_inference_indexer.cu;
// this class owns the member's passport, the session state, the expert slot
// cache and the Infer loop.

#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/gigachat_moe_expert_cache.h"
#include "lsxcommon/gigachat_moe_inference_pipeline.h"
#include "lsxcommon/gigachat_moe_37b_load.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

namespace gigachat4_708b {

// The frozen expectation of this member (format tag, geometry, layer map and
// gate rank), built from gigachat_moe_708b_constants.h. Consumed by the shared
// loader so it never reads another member's passport.
gigachat4::Gc4FrozenExpect Gc708BExpect();

} // namespace gigachat4_708b

class GigaChat4Moe708BModel final : public ModelBase {
public:
    static const std::string kType;  // "gigachat4-708b-a41b"

    explicit GigaChat4Moe708BModel(ModelInitConfig cfg);
    ~GigaChat4Moe708BModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }

    // Prompt framing (stage 4): chatml-v5 single turn with a prefilled empty
    // <think></think>, ids resolved from the checkpoint tokenizer. The template
    // is byte-identical to the 37B's.
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
