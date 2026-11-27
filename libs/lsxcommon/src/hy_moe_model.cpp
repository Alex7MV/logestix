#include "hy_moe_model.h"

#include "hy_moe_constants.h"
#include "lsxcommon/hy_moe_tool_format.h"

#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/tokenizer.h"

#include <absl/strings/str_cat.h>

namespace lsxcommon {

const std::string HyMemoryModel::kType = "hy-memory-30b-a3b";

HyMemoryModel::HyMemoryModel(ModelInitConfig cfg) : ModelBase{std::move(cfg)} {}

HyMemoryModel::~HyMemoryModel() { Unload(); }

void HyMemoryModel::TeardownWeights()
{
    scratch_.Free();
    session_.Free();
    pw_.Free();
    TeardownContext();
}

bool HyMemoryModel::Load()
{
    Unload();  // idempotent re-load support
    if (!BeginLoad(cfg_.model_path)) return false;

    hy::HyTopology top = hy::LoadHyTopology(cfg_.model_path);
    if (top.num_layers <= 0) {
        lsxcommon::log::error(absl::StrCat(
            "[hy-model] topology error: ",
            "model.index.json unreadable or empty"));
        return AbortLoad();
    }
    // Format contract: no backward compatibility (AGENTS.md).
    // The schema generation is checked by the loader that parses the
    // index (lsxcommon::IndexLayoutSupported); this used to compare a
    // per-model tag that travelled here through error_msg.

    // Frozen-geometry passport cross-check (Model Ingestion Protocol step 3).
    using namespace hy::frozen;
    if (!SyntheticGeometryAllowed() &&
        (top.num_layers != kLayers || top.hidden_size != kHidden ||
         top.num_experts != kNumExperts || top.num_attention_heads != kHeads ||
         top.num_key_value_heads != kKvHeads || top.head_dim != kHeadDim ||
         top.vocab_size != kVocab ||
         top.moe_intermediate_size != kMoeIntermediate)) {
        lsxcommon::log::error(absl::StrCat(
            "[hy-model] topology mismatches frozen passport (layers ",
            top.num_layers, "/", static_cast<int>(kLayers), ", hidden ",
            top.hidden_size, "/", static_cast<int>(kHidden), ", experts ",
            top.num_experts, "/", static_cast<int>(kNumExperts), ")"));
        return AbortLoad();
    }
    vram_ = top.vram;

    if (!EnsureContext()) {
        lsxcommon::log::error("[hy-model] context create failed");
        return AbortLoad();
    }
    std::string error;
    if (!hy::LoadHyDenseWeights(cfg_.model_path, pw_, stream(), error)) {
        lsxcommon::log::error(
            absl::StrCat("[hy-model] pinned weight load failed: ", error));
        return AbortLoad();
    }

    resident_bytes_ = pw_.resident_bytes;
    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();

    lsxcommon::log::info(absl::StrCat(
        "[hy-model] loaded: ", pw_.layers.size(), " layers, ",
        pw_.resident_bytes >> 20, " MiB pinned (all routed experts resident)"));
    FinishLoad();
    return true;
}

gpu::InferenceResult HyMemoryModel::Infer(const ModelRequest &req)
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
    r = RunUnderResidencyLock([&] {
        return hy::RunHyInference(
            cfg_.model_path, req.input_ids, req.max_gen_tokens, pw_, session_,
            scratch_, lsxcommon::sample::SamplerFor(req.sampling), stream(),
            req.ignore_eos, &gs);
    });
    if (r.ok) {
        r.output_text = runtime::DecodeGeneratedText(
            *tok_db_, r.gen_tokens.data(), r.gen_tokens.size());
        r.num_layers = pw_.top.num_layers;
        r.weight_resident_bytes =
            resident_bytes_ + scratch_.arena_bytes + session_.Bytes();
    }
    return r;
}

gpu::InferenceResult HyMemoryModel::SelfCheck(const ModelRequest &req)
{
    // Chat model with a no-think generation prefix: frame the raw control
    // prompt with the checkpoint's own template and accept "Paris" within a
    // short window (same deviation as MiniMax / ornith).
    const runtime::ParisSelfCheckOptions opts{
        .frame_prompt = true,
        .min_tokens = 16,
        .answer_window = 16,
        .substring_match = true,
    };
    return runtime::RunParisSelfCheck(*this, req, opts);
}

bool HyMemoryModel::BuildPromptInputIds(
    const ArrowTokenizerDb &tok, std::string_view raw_prompt,
    std::vector<int32_t> &out_ids) const
{
    ChatConversation conv;
    conv.messages.push_back({"user", std::string(raw_prompt), "", {}});
    if (hy::RenderHyConversation(tok, conv, out_ids)) return true;
    out_ids.clear();
    return prompt::EncodeRawPrompt(tok, raw_prompt, out_ids);
}

bool HyMemoryModel::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    if (hy::RenderHyConversation(tok, conv, out_ids)) return true;
    // Fallback: the shared Hermes renderer (a floor, not a pass).
    const std::string text = tools::RenderConversation(conv, /*include_tools=*/true);
    if (text.empty()) return false;
    out_ids.clear();
    return BuildPromptInputIds(tok, text, out_ids);
}

bool HyMemoryModel::ParseToolCalls(
    const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
    std::vector<ToolCall> &out_calls) const
{
    return hy::ParseHyToolCalls(tok, gen_tokens, out_calls);
}

} // namespace lsxcommon
