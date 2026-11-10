#pragma once

#include <memory>
#include <string>
#include <vector>

#include "lsxcommon/dory_dense_constants.h"
#include "lsxcommon/dory_dense_inference.h"
#include "lsxcommon/dory_dense_load.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

// NVIDIA-Nemotron-Labs-Dory1.0-4B: looped (latent-recurrent) normalized-GPT.
// Registry id `dory-1.0-4b`; frozen passport in dory_dense_constants.h.
class DoryDenseModel final : public ModelBase {
public:
    static const std::string kType;

    explicit DoryDenseModel(ModelInitConfig cfg);
    ~DoryDenseModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;
    const std::string &Type() const override { return kType; }
    // The KV caches are [slot][pos][kv-dim] (pos-major).
    KvCacheLayout KvLayout() const override { return KvCacheLayout::kPosMajor; }

    // ── Token bridge (Model Ingestion Protocol stage 4 + 4b) ──────────────
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
    bool ValidateGeometry(const dory::DoryTopology &top) const;

    dory::DoryPinnedWeights pw_;
    bool weights_loaded_ = false;
    std::unique_ptr<dory::DoryKvCache> kv_;
    std::vector<int32_t> session_tail_;
};

} // namespace lsxcommon
