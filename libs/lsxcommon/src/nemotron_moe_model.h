#pragma once

// Nemotron-3.5-Lightning-30B-A3B family: the BF16-distribution model
// (kType "nemotron-3.5-lightning-30b-a3b") and the pre-quantized NVFP4
// sibling (kType "nemotron-3.5-lightning-30b-a3b-nvfp4"), which reuse this
// shared lifecycle + generation contour. Hybrid Mamba-2 + MoE + full-attention,
// standalone ModelBase subclass (GlmV5Model / QwenMoeModelBase pattern). The
// frozen geometry passport is compile-time locked per model (each leaf owns its
// own constants header); the leaf's LoadAndValidateIndex cross-checks the
// converted index against its OWN passport and fails on mismatch.
//
// The two distributions emit the SAME `logestix-nemotron-nvfp4` converted
// layout, so the loader, expert cache, kernels, pipeline and tool codec are
// shared verbatim; only the source-layout converter and the frozen passport
// differ. `NemotronModelBase` owns everything shared; each leaf supplies its
// kType, its index loader/validator and its per-call FastMathScope.

#include <string>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/model_base.h"
#include "lsxcommon/nemotron_moe_expert_cache.h"
#include "lsxcommon/nemotron_moe_inference.h"
#include "lsxcommon/nemotron_moe_load.h"

namespace lsxcommon {

// Shared Nemotron lifecycle + generation plumbing. Leaf models provide kType,
// their own frozen-geometry gate (LoadAndValidateIndex) and an Infer that opens
// LsxFastMathScope with the instance's own config before RunInferShared (model
// independence invariant).
class NemotronModelBase : public ModelBase {
public:
    // Head-major by construction: its attention indexes
    // (hk * cache_stride + t) * head_dim, so cfg.kv_cache_layout has nothing to
    // switch here. Reported rather than assumed, so a test asking "what layout is
    // this model using" gets the truth instead of the request echoed back - and
    // so this stays visible as the one family where the two layouts are not a
    // choice.
    KvCacheLayout KvLayout() const override
    {
        return KvCacheLayout::kHeadMajor;
    }

    bool Load() override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                             std::string_view raw_prompt,
                             std::vector<int32_t> &out_ids) const override;

    // Tool/function-calling bridge (step 4b): Nemotron-3.5 uses the Qwen-style
    // ChatML XML call syntax (`<tool_call><function=…>`), but its own tool
    // declaration block; parse is shared with the Qwen codec.
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

    static bool RenderNemotronConversation(const ChatConversation &conv,
                                           std::string &out);

    // Test/diagnostics access to the parsed passport.
    const nemotron_moe::NemotronIndex &IndexForTest() const { return idx_; }

    // Test/diagnostics access to the routed-expert slot LRU.
    const nemotron_moe::NmExpertCache &ExpertCacheForTest() const
    {
        return expert_cache_;
    }

protected:
    explicit NemotronModelBase(ModelInitConfig cfg);

    void TeardownWeights() override;

    // Loads <model_path>/logestix/model.index.json into `idx_` and validates it
    // against the leaf's OWN frozen passport. Returns false and fills `error` on
    // failure (the base logs it and tears the load down).
    virtual bool LoadAndValidateIndex(std::string &error) = 0;

    // Shared Infer body. The caller MUST have opened its own
    // LsxFastMathScope(EffectiveFastMathMode(cfg_)) first (per-model policy).
    gpu::InferenceResult RunInferShared(const ModelRequest &req);

    nemotron_moe::NemotronIndex idx_;
    nemotron_moe::NmWeights weights_;
    nemotron_moe::NmSession session_;
    nemotron_moe::NmExpertCache expert_cache_;
};

// BF16-distribution Nemotron-3.5-Lightning-30B-A3B (kType
// "nemotron-3.5-lightning-30b-a3b").
class NemotronHModel final : public NemotronModelBase {
public:
    static const std::string kType;  // "nemotron-3.5-lightning-30b-a3b"

    explicit NemotronHModel(ModelInitConfig cfg);
    ~NemotronHModel() override;

    gpu::InferenceResult Infer(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }

protected:
    bool LoadAndValidateIndex(std::string &error) override;
};

} // namespace lsxcommon
