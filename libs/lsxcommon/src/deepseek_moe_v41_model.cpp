#include "deepseek_moe_v41_model.h"

#include <vector>

#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/tokenizer.h"
#include "lsxcommon/deepseek_moe_tool_format.h"

#include <cstring>

#include "deepseek_moe_v41_constants.h"
#include "deepseek_moe_v41_load.h"
#include "deepseek_moe_v41_inference_pipeline.h"

namespace lsxcommon {

const std::string DeepSeekV41FlashModel::kType = "deepseek-v4.1-flash";
const std::string DeepSeekV41Flash1017Model::kType =
    "deepseek-v4.1-flash-1017";

DeepSeekV41BackboneModel::DeepSeekV41BackboneModel(ModelInitConfig cfg,
                                                   const std::string &type)
    : ModelBase{std::move(cfg)}, type_(type)
{
}

DeepSeekV41FlashModel::DeepSeekV41FlashModel(ModelInitConfig cfg)
    : DeepSeekV41BackboneModel{std::move(cfg), kType}
{
}

DeepSeekV41Flash1017Model::DeepSeekV41Flash1017Model(ModelInitConfig cfg)
    : DeepSeekV41BackboneModel{std::move(cfg), kType}
{
}

DeepSeekV41BackboneModel::~DeepSeekV41BackboneModel() { Unload(); }

void DeepSeekV41BackboneModel::TeardownWeights()
{
    if (expert_db_ready_) {
        deepseek_v41::gpu::free_expert_index(expert_db_);
        expert_db_ready_ = false;
    }
    pw_.Free();
    TeardownContext();
}

bool DeepSeekV41BackboneModel::Load()
{
    Unload();  // idempotent re-load support
    if (!BeginLoad(cfg_.model_path)) return false;

    const auto topo = deepseek_v41::LoadTopology(cfg_.model_path);
    if (!topo.error_msg.empty()) {
        log::error("[v41-model] topology error: " + topo.error_msg);
        return AbortLoad();
    }
    if (!deepseek_v41::IsSupportedFormat(topo)) {
        log::error(
            "[v41-model] unsupported index format (expected logestix-v41-nvfp4)");
        return AbortLoad();
    }
    std::string geo_err;
    if (!deepseek_v41::ValidateGeometry(topo, &geo_err) &&
        !SyntheticGeometryAllowed()) {
        log::error("[v41-model] frozen geometry mismatch: " + geo_err);
        return AbortLoad();
    }
    // The index stamps the CARD name of the tree it was converted from, which
    // is what path auto-detection resolves to this object's kType. A mismatch
    // means the tree belongs to a sibling checkpoint (or was hand-edited); the
    // weights would load, and the model would then answer as a different
    // model.
    if (!SyntheticGeometryAllowed() &&
        !deepseek_v41::MatchesModelType(topo, type_)) {
        log::error("[v41-model] converted tree is '" + topo.model +
                   "', this model is '" + type_ + "'");
        return AbortLoad();
    }
    vram_ = topo.vram;

    if (!EnsureContext()) {
        log::error("[v41-model] context create failed");
        return AbortLoad();
    }
    if (deepseek_v41::gpu::load_pinned_weights(cfg_.model_path, pw_, stream()) !=
        cudaSuccess) {
        log::error("[v41-model] pinned weight load failed");
        return AbortLoad();
    }
    expert_db_ = deepseek_v41::gpu::load_expert_index(
        cfg_.model_path, topo.num_layers, topo.experts_per_layer);
    expert_db_ready_ = true;

    resident_bytes_ = pw_.resident_bytes;
    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();
    FinishLoad();
    return true;
}

gpu::InferenceResult DeepSeekV41BackboneModel::Infer(const ModelRequest &req)
{
    // Model independence: assert THIS model's fast-math policy for the call.
    const lsx_nvfp4::LsxFastMathScope fm_scope(EffectiveFastMathMode(cfg_));
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;
    if (!loaded_) {
        r.error_msg = "model not loaded";
        return r;
    }
    lsxcommon::GenerationStream gs;
    gs.on_token = req.token_callback ? &req.token_callback : nullptr;
    gs.abort_flag = req.abort_flag;
    return RunUnderResidencyLock([&] {
        return gpu::RunDeepSeekV41Inference(
            cfg_.model_path, req.input_ids, req.max_gen_tokens, tok_db_.get(),
            pw_,
            expert_db_, lsxcommon::sample::SamplerFor(req.sampling), stream(),
            req.ignore_eos, &gs);
    });
}

gpu::InferenceResult DeepSeekV41BackboneModel::SelfCheck(
    const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

// ── DeepSeek-V4.1 native DSML tool format (encoding.py) ────────────────────
namespace {
constexpr DsmlTags kV41DsmlTags{" calls", " invoke", " parameter",
                                "<｜System｜>"};
constexpr const char *kDsmlUserToken = "<｜User｜>";
} // namespace

bool DeepSeekV41BackboneModel::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    std::string text;
    if (!RenderDsmlConversation(conv, kV41DsmlTags, text)) return false;
    out_ids.clear();
    if (arrow_tokenizer_lookup(tok, kDsmlUserToken,
                               std::strlen(kDsmlUserToken)) < 0)
        return BuildPromptInputIds(tok, text, out_ids);
    return AppendDeepSeekIds(tok, text, kV41DsmlTags, out_ids);
}

bool DeepSeekV41BackboneModel::ParseToolCalls(
    const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
    std::vector<ToolCall> &out_calls) const
{
    return ParseDeepSeekToolCalls(tok, gen_tokens, kV41DsmlTags, out_calls);
}

} // namespace lsxcommon
