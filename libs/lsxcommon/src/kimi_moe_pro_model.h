#pragma once

#include <memory>
#include <string>
#include <vector>

#include "lsxcommon/kimi_moe_k3_inference_pipeline.h"
#include "lsxcommon/kimi_moe_k3_load.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

// Kimi-K3.1-Pro (kType "kimi-k3.1-pro"): the Kimi-K3 KimiLinear tower (69 KDA +
// 24 gated NoPE MLA layers, AttnRes block 12, Stable LatentMoE, SiTU-GLU) with
// Gated DSA on every full-attention layer — a Lightning Indexer that selects
// `kIndexTopK` keys per query. The whole generation path is the Kimi-K3 path;
// the only delta is the indexer, dormant at `n_pos <= index_topk`.
//
// The variant is the FROZEN PASSPORT plus the tree: Load requires the DSA form
// of the topology cross-check (src/kimi_moe_pro_constants.h) and the indexer's
// own geometry, which the structural cross-check deliberately leaves to the
// model. A Kimi-K3 tree is rejected, and so is a Kimi-K3.1-Pro tree whose
// indexer geometry is not this passport's.
class KimiK31ProModel final : public ModelBase {
public:
    static const std::string kType;  // "kimi-k3.1-pro"

    explicit KimiK31ProModel(ModelInitConfig cfg);
    ~KimiK31ProModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    // Same wire as the rest of the family: the variant changes what the layers
    // attend to, never how a turn is framed, so the XTML codec is shared.
    bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                             std::string_view raw_prompt,
                             std::vector<int32_t> &out_ids) const override;
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

    const std::string &Type() const override { return kType; }

protected:
    void TeardownWeights() override;

    kimi_k3::KimiK3Topology top_;
    kimi_k3::KimiK3PinnedWeights pw_;
    std::unique_ptr<kimi_k3::KimiK3ExpertDb> experts_;
    bool weights_loaded_ = false;
};

}  // namespace lsxcommon
