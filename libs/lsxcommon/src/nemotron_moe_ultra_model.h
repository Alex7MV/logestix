#pragma once

// NVIDIA-Nemotron-3-Ultra-550B-A55B leaf model (kType
// "nemotron-ultra-550b-a55b").
//
// A SEPARATE LINEAGE from Nemotron-3.5-Lightning-30B-A3B: own frozen passport,
// own index/weights/expert-cache/kernel/pipeline TUs, own tool codec. Only the
// model-neutral layers are shared (ModelBase, lsx_nvfp4, lsx_gemv, the neutral
// expert store, ArrowTokenizerDb). That is deliberate - the 30B is a
// perf-frontier row with seeded baselines and four gates, and this model shares
// no width, layer count or routing geometry with it.

#include <string>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/model_base.h"
#include "lsxcommon/nemotron_moe_ultra_expert_cache.h"
#include "lsxcommon/nemotron_moe_ultra_inference.h"
#include "lsxcommon/nemotron_moe_ultra_load.h"

namespace lsxcommon {

class NemotronUltra550BModel final : public ModelBase {
public:
    static const std::string kType;  // "nemotron-ultra-550b-a55b"

    explicit NemotronUltra550BModel(ModelInitConfig cfg);
    ~NemotronUltra550BModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }

    bool BuildPromptInputIds(const ArrowTokenizerDb &tok, std::string_view raw_prompt,
                             std::vector<int32_t> &out_ids) const override;

    // Tool/function-calling bridge: Ultra's own chat template (a `<|im_start|>`
    // ChatML frame with a `<tools>` XML preamble and `<function=…><parameter=…>`
    // calls), rendered by nemotron_ultra_tool_format and parsed with the shared
    // XML parser.
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

    // Mirror of the 30B family: the attention kernel indexes the KV cache
    // head-major by construction, so this is not a choice the caller can flip.
    KvCacheLayout KvLayout() const override { return KvCacheLayout::kHeadMajor; }

    // Test/diagnostics access.
    const nemotron_ultra::UltraIndex &IndexForTest() const { return idx_; }
    const nemotron_ultra::UltraExpertCache &ExpertCacheForTest() const
    {
        return expert_cache_;
    }

protected:
    void TeardownWeights() override;

private:
    bool LoadAndValidateIndex(std::string &error);

    nemotron_ultra::UltraIndex idx_;
    nemotron_ultra::UltraWeights weights_;
    nemotron_ultra::UltraSession session_;
    nemotron_ultra::UltraExpertCache expert_cache_;
};

} // namespace lsxcommon
