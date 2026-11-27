#include "ling_moe_model.h"

#include "ling_moe_constants.h"
#include "lsxcommon/ling_moe_tool_format.h"
#include "lsxcommon/index_layout.h"

#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/synthetic_geometry.h"
#include "lsxcommon/tokenizer.h"

#include <absl/strings/str_cat.h>

#include <algorithm>

namespace lsxcommon {

const std::string LingMoeModel::kType = "ling";

LingMoeModel::LingMoeModel(ModelInitConfig cfg) : ModelBase{std::move(cfg)} {}

LingMoeModel::~LingMoeModel() { Unload(); }

void LingMoeModel::TeardownWeights()
{
    scratch_.Free();
    session_.Free();
    pw_.Free();
    TeardownContext();
}

bool LingMoeModel::Load()
{
    Unload();
    tokenizer_optional_ = SyntheticGeometryAllowed();
    if (!BeginLoad(cfg_.model_path)) return false;

    ling::LingIndex top = ling::LoadLingIndex(cfg_.model_path);
    if (!top.error_msg.empty()) {
        log::error(absl::StrCat("[ling-model] topology error: ",
                                top.error_msg));
        return AbortLoad();
    }
    if (!lsxcommon::IndexLayoutSupported(top.layout)) {
        log::error(absl::StrCat(
            "[ling-model] unsupported index format '", std::to_string(top.layout),
            "' — convert via: lsxforge --ling --input <model_dir> --output "
            "<model_dir>"));
        return AbortLoad();
    }
    using namespace ling::frozen;
    if (!SyntheticGeometryAllowed()) {
        const bool ok =
            top.num_layers == kLayers && top.hidden == kHidden &&
            top.vocab == kVocab && top.heads == kHeads &&
            top.kv_heads == kKvHeads && top.head_dim == kHeadDim &&
            top.qk_head_dim == kQkHeadDim && top.qk_nope == kQkNope &&
            top.qk_rope == kQkRope && top.v_head == kVHead &&
            top.q_lora == kQLora && top.kv_lora == kKvLora &&
            top.conv_kernel == kConvKernel && top.num_experts == kNumExperts &&
            top.top_k == kTopK && top.n_group == kGroupCount &&
            top.topk_group == kTopkGroup &&
            top.moe_intermediate == kMoeIntermediate &&
            top.shared_intermediate == kSharedIntermediate &&
            top.dense_intermediate == kDenseIntermediate &&
            top.first_moe_layer == kMoeFirstLayer &&
            static_cast<int>(top.mla_layers.size()) == kMlaLayerCount;
        if (ok) {
            for (int i = 0; i < kMlaLayerCount; ++i)
                if (top.mla_layers[static_cast<size_t>(i)] != kMlaLayerIds[i]) {
                    log::error("[ling-model] MLA layer table mismatch");
                    return AbortLoad();
                }
        } else {
            log::error(absl::StrCat(
                "[ling-model] topology mismatches frozen passport (layers ",
                top.num_layers, "/", static_cast<int>(kLayers), ", hidden ",
                top.hidden, "/", static_cast<int>(kHidden), ", experts ",
                top.num_experts, "/", static_cast<int>(kNumExperts), ")"));
            return AbortLoad();
        }
    }

    if (!EnsureContext()) {
        log::error("[ling-model] context create failed");
        return AbortLoad();
    }
    std::string error;
    if (!ling::LoadLingDenseWeights(cfg_.model_path, pw_, stream(), error)) {
        log::error(
            absl::StrCat("[ling-model] pinned weight load failed: ", error));
        return AbortLoad();
    }
    if (!ling::LingPrepareExpertCache(pw_, scratch_, error)) {
        log::error(
            absl::StrCat("[ling-model] expert cache allocation failed: ",
                         error));
        return AbortLoad();
    }
    resident_bytes_ = pw_.resident_bytes + scratch_.cache.DeviceBytes();
    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();
    const uint64_t cache_bytes = scratch_.cache.DeviceBytes();
    if (!AcquireExpertCacheLease(cache_bytes, cache_bytes / 4, cache_bytes))
        log::warn("[ling-model] expert cache lease not granted; continuing");

    eos_ids_ = {top.eos_token_id};
    log::info(absl::StrCat(
        "[ling-model] loaded: ", pw_.layers.size(), " layers, ",
        pw_.resident_bytes >> 20, " MiB pinned + ",
        scratch_.cache.DeviceBytes() >> 20, " MiB expert cache"));
    FinishLoad();
    return true;
}

gpu::InferenceResult LingMoeModel::Infer(const ModelRequest &req)
{
    const lsx_nvfp4::LsxFastMathScope fm_scope(EffectiveFastMathMode(cfg_));
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;
    if (!loaded_) {
        r.error_msg = "model not loaded";
        return r;
    }
    GenerationStream gs;
    gs.on_token = req.token_callback ? &req.token_callback : nullptr;
    gs.abort_flag = req.abort_flag;
    r = RunUnderResidencyLock([&] {
        return ling::RunLingInference(
            cfg_.model_path, req.input_ids, req.max_gen_tokens, eos_ids_, pw_,
            session_, scratch_, lsxcommon::sample::SamplerFor(req.sampling),
            stream(), req.ignore_eos, &gs);
    });
    if (r.ok) {
        if (tok_db_ && tok_db_->valid())
            r.output_text = runtime::DecodeGeneratedText(
                *tok_db_, r.gen_tokens.data(), r.gen_tokens.size());
        r.num_layers = pw_.top.num_layers;
        r.weight_resident_bytes =
            resident_bytes_ + scratch_.arena_bytes + session_.Bytes();
    }
    return r;
}

gpu::InferenceResult LingMoeModel::SelfCheck(const ModelRequest &req)
{
    const runtime::ParisSelfCheckOptions opts{
        .frame_prompt = true,
        .min_tokens = 256,
        .answer_window = 96,
        .substring_match = true,
    };
    return runtime::RunParisSelfCheck(*this, req, opts);
}

bool LingMoeModel::BuildPromptInputIds(
    const ArrowTokenizerDb &tok, std::string_view raw_prompt,
    std::vector<int32_t> &out_ids) const
{
    ChatConversation conv;
    conv.messages.push_back({"user", std::string(raw_prompt), "", {}});
    if (ling::RenderLingConversation(tok, conv, out_ids)) return true;
    out_ids.clear();
    return prompt::EncodeRawPrompt(tok, raw_prompt, out_ids);
}

bool LingMoeModel::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    if (ling::RenderLingConversation(tok, conv, out_ids)) return true;
    const std::string text =
        tools::RenderConversation(conv, /*include_tools=*/true);
    if (text.empty()) return false;
    out_ids.clear();
    return BuildPromptInputIds(tok, text, out_ids);
}

bool LingMoeModel::ParseToolCalls(
    const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
    std::vector<ToolCall> &out_calls) const
{
    return ling::ParseLingToolCalls(tok, gen_tokens, out_calls);
}

} // namespace lsxcommon
