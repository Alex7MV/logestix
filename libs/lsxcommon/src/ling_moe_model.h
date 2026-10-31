#pragma once

#include <string>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/ling_moe_inference.h"
#include "lsxcommon/ling_moe_load.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

// Ling-3.1-flash (BailingMoeV3 hybrid: 63 KDA + 9 gated MLA layers, 68 MoE
// layers with 512 routed experts top-8 and one ungated shared expert).
class LingMoeModel final : public ModelBase {
public:
    static const std::string kType;
    explicit LingMoeModel(ModelInitConfig cfg);
    ~LingMoeModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }
    KvCacheLayout KvLayout() const override { return KvCacheLayout::kPosMajor; }

    bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                             std::string_view raw_prompt,
                             std::vector<int32_t> &out_ids) const override;
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

private:
    void TeardownWeights() override;

    ling::LingPinnedWeights pw_;
    ling::LingSession session_;
    ling::LingScratch scratch_;
    std::vector<int32_t> eos_ids_;
};

} // namespace lsxcommon
