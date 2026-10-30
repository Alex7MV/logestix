#pragma once

#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/flint_moe_inference_pipeline.h"
#include "lsxcommon/flint_moe_load.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

// Recursive Flint (registry "flint").
// Hybrid dense-prefix + routed-MoE decoder: GQA + per-head QK-norm + learned
// per-head sinks + sliding-window layers with RoPE and full-attention layers
// with NO positional encoding + sigmoid top-k router + one ungated shared
// expert. Every routed expert is resident in a per-layer device slab.
class FlintMoeModel final : public ModelBase {
public:
    static const std::string kType;
    explicit FlintMoeModel(ModelInitConfig cfg);
    ~FlintMoeModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }
    // The cache this engine actually builds (pos-major, see FlintSession).
    KvCacheLayout KvLayout() const override { return KvCacheLayout::kPosMajor; }

    // Single-prompt framing: a chat model must never see an unframed prompt, so
    // the raw control prompt goes through the checkpoint's own template as one
    // user turn.
    bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                             std::string_view raw_prompt,
                             std::vector<int32_t> &out_ids) const override;

    // Tool/function-calling bridge (Model Ingestion Protocol step 4b): native
    // inline `<|user|>`/`<|assistant|>` framing plus the
    // `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`
    // codec of chat_template.jinja.
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

private:
    void TeardownWeights() override;

    flint::FlintPinnedWeights pw_;
    flint::FlintSession session_;
    flint::FlintScratch scratch_;
    std::vector<int32_t> eos_ids_;
};

} // namespace lsxcommon
