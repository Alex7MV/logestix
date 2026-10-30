#include "flint_moe_model.h"

#include "flint_moe_constants.h"
#include "lsxcommon/flint_moe_tool_format.h"

#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/tokenizer.h"

#include <absl/strings/str_cat.h>

namespace lsxcommon {

const std::string FlintMoeModel::kType = "flint";

FlintMoeModel::FlintMoeModel(ModelInitConfig cfg) : ModelBase{std::move(cfg)} {}

FlintMoeModel::~FlintMoeModel() { Unload(); }

void FlintMoeModel::TeardownWeights()
{
    scratch_.Free();
    session_.Free();
    pw_.Free();
    TeardownContext();
}

bool FlintMoeModel::Load()
{
    Unload();  // idempotent re-load support
    if (!BeginLoad(cfg_.model_path)) return false;

    flint::FlintTopology top = flint::LoadFlintTopology(cfg_.model_path);
    const std::string fmt = top.error_msg;
    top.error_msg.clear();
    if (top.num_layers <= 0) {
        lsxcommon::log::error(absl::StrCat(
            "[flint-model] topology error: ",
            fmt.empty() ? "model.index.json unreadable" : fmt));
        return AbortLoad();
    }
    // Format contract: no backward compatibility (AGENTS.md).
    if (!fmt.empty() && fmt != flint::frozen::kFormatTag) {
        lsxcommon::log::error(absl::StrCat(
            "[flint-model] unsupported index format '", fmt,
            "' — convert via: lsxforge --flint --input <model_dir> --output "
            "<model_dir>"));
        return AbortLoad();
    }
    // Frozen-geometry passport cross-check (Model Ingestion Protocol step 3).
    using namespace flint::frozen;
    if (!SyntheticGeometryAllowed() &&
        (top.num_layers != kLayers || top.hidden_size != kHidden ||
         top.num_experts != kNumExperts || top.num_attention_heads != kHeads ||
         top.num_key_value_heads != kKvHeads || top.head_dim != kHeadDim ||
         top.vocab_size != kVocab ||
         top.moe_intermediate_size != kMoeIntermediate ||
         top.dense_intermediate_size != kDenseIntermediate ||
         top.shared_intermediate_size != kSharedIntermediate ||
         top.num_experts_per_tok != kTopK ||
         top.sliding_window != kWindow)) {
        lsxcommon::log::error(absl::StrCat(
            "[flint-model] topology mismatches frozen passport (layers ",
            top.num_layers, "/", static_cast<int>(kLayers), ", hidden ",
            top.hidden_size, "/", static_cast<int>(kHidden), ", experts ",
            top.num_experts, "/", static_cast<int>(kNumExperts), ", window ",
            top.sliding_window, "/", static_cast<int>(kWindow), ")"));
        return AbortLoad();
    }
    // No-RoPE and the learned sink are load-bearing: an index without them
    // would be a converter from a different recipe.
    if (top.no_rope_layers.empty()) {
        lsxcommon::log::error("[flint-model] index carries no no_rope_layers");
        return AbortLoad();
    }
    vram_ = top.vram;
    eos_ids_ = top.eos_ids;
    if (eos_ids_.empty() && top.eos_token_id >= 0)
        eos_ids_.push_back(top.eos_token_id);

    if (!EnsureContext()) {
        lsxcommon::log::error("[flint-model] context create failed");
        return AbortLoad();
    }
    std::string error;
    if (!flint::LoadFlintDenseWeights(cfg_.model_path, pw_, stream(), error)) {
        lsxcommon::log::error(
            absl::StrCat("[flint-model] pinned weight load failed: ", error));
        return AbortLoad();
    }

    resident_bytes_ = pw_.resident_bytes;
    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();

    lsxcommon::log::info(absl::StrCat(
        "[flint-model] loaded: ", pw_.layers.size(), " layers, ",
        pw_.resident_bytes >> 20, " MiB pinned (all routed experts resident)"));
    FinishLoad();
    return true;
}

gpu::InferenceResult FlintMoeModel::Infer(const ModelRequest &req)
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
        return flint::RunFlintInference(
            cfg_.model_path, req.input_ids, req.max_gen_tokens, eos_ids_, pw_,
            session_, scratch_, lsxcommon::sample::SamplerFor(req.sampling),
            stream(), req.ignore_eos, &gs);
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

gpu::InferenceResult FlintMoeModel::SelfCheck(const ModelRequest &req)
{
    // Chat model that decides on its own whether to reason: frame the raw
    // control prompt with the checkpoint's template and accept "Paris" within a
    // short window (same deviation as Hy-Memory / MiniMax / ornith).
    // Flint reasons before it answers, so the control answer lives well past
    // the first few tokens: the probe gets a real reasoning budget and searches
    // a wide window for "Paris".
    const runtime::ParisSelfCheckOptions opts{
        .frame_prompt = true,
        .min_tokens = 256,
        .answer_window = 96,
        .substring_match = true,
    };
    return runtime::RunParisSelfCheck(*this, req, opts);
}

bool FlintMoeModel::BuildPromptInputIds(
    const ArrowTokenizerDb &tok, std::string_view raw_prompt,
    std::vector<int32_t> &out_ids) const
{
    ChatConversation conv;
    conv.messages.push_back({"user", std::string(raw_prompt), "", {}});
    if (flint::RenderFlintConversation(tok, conv, out_ids)) return true;
    out_ids.clear();
    return prompt::EncodeRawPrompt(tok, raw_prompt, out_ids);
}

bool FlintMoeModel::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    if (flint::RenderFlintConversation(tok, conv, out_ids)) return true;
    // Fallback: the shared Hermes renderer (a floor, not a pass).
    const std::string text =
        tools::RenderConversation(conv, /*include_tools=*/true);
    if (text.empty()) return false;
    out_ids.clear();
    return BuildPromptInputIds(tok, text, out_ids);
}

bool FlintMoeModel::ParseToolCalls(
    const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
    std::vector<ToolCall> &out_calls) const
{
    return flint::ParseFlintToolCalls(tok, gen_tokens, out_calls);
}

} // namespace lsxcommon
