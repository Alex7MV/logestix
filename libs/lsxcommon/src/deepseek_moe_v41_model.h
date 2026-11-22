#pragma once

#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/model_base.h"

#include "deepseek_moe_v41_1017_constants.h"
#include "deepseek_moe_v41_constants.h"
#include "deepseek_moe_v41_load_internal.h"

namespace lsxcommon {

// DeepSeek-V4.1 text backbone — ONE implementation body, TWO registered
// checkpoints.
//
//   DeepSeekV41FlashModel       -> "deepseek-v4.1-flash"        (the 09-10 tree)
//   DeepSeekV41Flash1017Model   -> "deepseek-v4.1-flash-1017"   (the 10-17 tree)
//
// Stage 0 triage (2026-11-22) proved the two source checkpoints share a
// byte-identical config.json / tokenizer.json / model.safetensors.index.json and
// differ only in payload bytes, so they are a SIBLING VARIANT: same geometry,
// same CED layer classification, same emitted `logestix-v41-nvfp4` layout. An
// autonomous contour (own pinned weights, own expert store, no other family's
// headers) serves both; only the frozen passport and the kType are per model,
// because the independence guard requires one namespace and one registry id per
// REGISTERED model. The numbers themselves are pinned against each other here,
// so a source drift is a compile error rather than a silent cross-model load.
static_assert(deepseek_v41_flash1017::kHidden == deepseek_v41::kHidden,
              "sibling hidden width");
static_assert(deepseek_v41_flash1017::kLayers == deepseek_v41::kLayers,
              "sibling layer count");
static_assert(deepseek_v41_flash1017::kMoeIntermediate ==
                  deepseek_v41::kMoeIntermediate,
              "sibling moe intermediate");
static_assert(deepseek_v41_flash1017::kSharedIntermediate ==
                  deepseek_v41::kSharedIntermediate,
              "sibling shared-expert intermediate");
static_assert(deepseek_v41_flash1017::kNumRouted == deepseek_v41::kNumRouted,
              "sibling routed-expert count");
static_assert(deepseek_v41_flash1017::kTopK == deepseek_v41::kTopK,
              "sibling top-k");
static_assert(deepseek_v41_flash1017::kHeads == deepseek_v41::kHeads,
              "sibling attention head count");
static_assert(deepseek_v41_flash1017::kHeadDim == deepseek_v41::kHeadDim,
              "sibling head dim");
static_assert(deepseek_v41_flash1017::kRopeHeadDim ==
                  deepseek_v41::kRopeHeadDim,
              "sibling rope head dim");
static_assert(deepseek_v41_flash1017::kQLoraRank == deepseek_v41::kQLoraRank,
              "sibling q_lora rank");
static_assert(deepseek_v41_flash1017::kOLoraRank == deepseek_v41::kOLoraRank,
              "sibling o_lora rank");
static_assert(deepseek_v41_flash1017::kOGroups == deepseek_v41::kOGroups,
              "sibling o groups");
static_assert(deepseek_v41_flash1017::kIndexNHeads ==
                  deepseek_v41::kIndexNHeads,
              "sibling indexer heads");
static_assert(deepseek_v41_flash1017::kIndexHeadDim ==
                  deepseek_v41::kIndexHeadDim,
              "sibling indexer head dim");
static_assert(deepseek_v41_flash1017::kIndexTopk == deepseek_v41::kIndexTopk,
              "sibling indexer top-k");
static_assert(deepseek_v41_flash1017::kWindow == deepseek_v41::kWindow,
              "sibling sliding window");
static_assert(deepseek_v41_flash1017::kVocab == deepseek_v41::kVocab,
              "sibling vocab");
static_assert(deepseek_v41_flash1017::kHcMult == deepseek_v41::kHcMult,
              "sibling hc multiplier");
static_assert(deepseek_v41_flash1017::kSinkhornIters ==
                  deepseek_v41::kSinkhornIters,
              "sibling sinkhorn iterations");
static_assert(deepseek_v41_flash1017::kRmsEps == deepseek_v41::kRmsEps,
              "sibling rms eps");
static_assert(deepseek_v41_flash1017::kRouteScale == deepseek_v41::kRouteScale,
              "sibling routed scaling factor");
static_assert(deepseek_v41_flash1017::kCompressRopeTheta ==
                  deepseek_v41::kCompressRopeTheta,
              "sibling compressed-layer rope theta");
static_assert(deepseek_v41_flash1017::kRopeTheta == deepseek_v41::kRopeTheta,
              "sibling pure-SWA rope theta");
static_assert(deepseek_v41_flash1017::kCandidateBlockSize ==
                  deepseek_v41::kCandidateBlockSize,
              "sibling candidate block size");
static_assert(deepseek_v41_flash1017::kCandidateTopkBlocks ==
                  deepseek_v41::kCandidateTopkBlocks,
              "sibling candidate top-k blocks");
static_assert(deepseek_v41_flash1017::kCandidateSourceLayer ==
                  deepseek_v41::kCandidateSourceLayer,
              "sibling candidate source layer");

// The shared body. Owns the pinned weights and the mmap routed-expert store; no
// DeepSeek-V4-Flash headers (autonomy invariant).
class DeepSeekV41BackboneModel : public ModelBase {
public:
    explicit DeepSeekV41BackboneModel(ModelInitConfig cfg,
                                      const std::string &type);
    ~DeepSeekV41BackboneModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return type_; }

    // Tool/function-calling bridge (Model Ingestion Protocol step 4b).
    // DeepSeek-V4.1 native DSML template (encoding.py, leading-space tags;
    // identical chat-mode rendering in the 1017 tree's chat_template.jinja).
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

protected:
    void TeardownWeights() override;

private:
    // Bound to the concrete class's `static const std::string kType`, which has
    // static storage duration (see the constructors in the .cpp).
    const std::string &type_;
    deepseek_v41::gpu::PinnedWeights pw_;
    deepseek_v41::gpu::ExpertDb expert_db_;
    bool expert_db_ready_ = false;
};

// The original V4.1-Flash checkpoint (registry "deepseek-v4.1-flash").
class DeepSeekV41FlashModel final : public DeepSeekV41BackboneModel {
public:
    static const std::string kType;
    explicit DeepSeekV41FlashModel(ModelInitConfig cfg);
};

// The 1017 update of the same checkpoint (registry "deepseek-v4.1-flash-1017"):
// identical geometry and layout, re-trained weights, new card name.
class DeepSeekV41Flash1017Model final : public DeepSeekV41BackboneModel {
public:
    static const std::string kType;
    explicit DeepSeekV41Flash1017Model(ModelInitConfig cfg);
};

} // namespace lsxcommon
