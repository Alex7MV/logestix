#pragma once

#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/hy_moe_inference_pipeline.h"
#include "lsxcommon/hy_moe_load.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

// Hy-SimulMT-30B-A3B (registry "hy-simulmt-30b-a3b") — a sibling of
// Hy-Memory-30B-A3B: the same hy_v3 text backbone (GQA + per-head plain
// QK-norm + full RoPE + sigmoid router with an expert bias + one gated shared
// expert, every routed expert resident in a per-layer device slab). The
// checkpoint's speech encoder is stripped; this class serves the text decoder.
class HySimulMTModel final : public ModelBase {
public:
    static const std::string kType;
    explicit HySimulMTModel(ModelInitConfig cfg);
    ~HySimulMTModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }
    KvCacheLayout KvLayout() const override { return KvCacheLayout::kPosMajor; }

    // Single-prompt framing through the checkpoint's own template (one user
    // turn, thinking disabled).
    bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                             std::string_view raw_prompt,
                             std::vector<int32_t> &out_ids) const override;

    // Tool/function-calling bridge: the verified Hy-native
    // `<|User|>`/`<|Assistant|>` framing plus the `<tool_calls>` /
    // `<arg_key>` XML codec (the SimulMT template's text branch is identical
    // to Hy-Memory's, and it fixes the toolcalls_end spelling).
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

private:
    void TeardownWeights() override;

    hy::HyPinnedWeights pw_;
    hy::HySession session_;
    hy::HyScratch scratch_;
};

} // namespace lsxcommon
