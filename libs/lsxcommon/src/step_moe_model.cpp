#include "step_moe_model.h"

#include "step_moe_constants.h"
#include "lsxcommon/step_moe_tool_format.h"

#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/tokenizer.h"

#include <absl/strings/str_cat.h>

namespace lsxcommon {

const std::string Step5PreviewModel::kType = "step-5-preview";

Step5PreviewModel::Step5PreviewModel(ModelInitConfig cfg)
    : ModelBase{std::move(cfg)}
{
}

Step5PreviewModel::~Step5PreviewModel() { Unload(); }

void Step5PreviewModel::TeardownWeights()
{
    scratch_.Free();
    session_.Free();
    pw_.Free();
    TeardownContext();
}

bool Step5PreviewModel::Load()
{
    Unload();  // idempotent re-load support
    if (!BeginLoad(cfg_.model_path)) return false;

    step::StepTopology top = step::LoadStepTopology(cfg_.model_path);
    if (top.num_layers <= 0) {
        lsxcommon::log::error(absl::StrCat(
            "[step-model] topology error: ",
            "model.index.json unreadable or empty"));
        return AbortLoad();
    }
    // Format contract: no backward compatibility (AGENTS.md).
    // The schema generation is checked by the loader that parses the
    // index (lsxcommon::IndexLayoutSupported); this used to compare a
    // per-model tag that travelled here through error_msg.

    // Frozen-geometry passport cross-check (Model Ingestion Protocol step 3).
    using namespace step::frozen;
    bool dispatch_ok = true;
    if (!SyntheticGeometryAllowed())
        for (int l = 0; l < top.num_layers; ++l)
            if ((top.mlp_types[static_cast<size_t>(l)] == "sparse") !=
                IsMoeLayer(l))
                dispatch_ok = false;
    if (!SyntheticGeometryAllowed() &&
        (!dispatch_ok || top.num_layers != kLayers || top.hidden_size != kHidden ||
         top.num_experts != kNumExperts ||
         top.num_attention_heads != kHeads ||
         top.num_key_value_heads != kKvHeads || top.head_dim != kHeadDim ||
         top.vocab_size != kVocab ||
         top.moe_intermediate_size != kMoeIntermediate ||
         top.first_moe_layer != kFirstMoeLayer)) {
        lsxcommon::log::error(absl::StrCat(
            "[step-model] topology mismatches frozen passport (layers ",
            top.num_layers, "/", static_cast<int>(kLayers), ", hidden ",
            top.hidden_size, "/", static_cast<int>(kHidden), ", experts ",
            top.num_experts, "/", static_cast<int>(kNumExperts), ")"));
        return AbortLoad();
    }
    vram_ = top.vram;

    if (!EnsureContext()) {
        lsxcommon::log::error("[step-model] context create failed");
        return AbortLoad();
    }
    std::string error;
    if (!step::LoadStepDenseWeights(cfg_.model_path, pw_, stream(), error)) {
        lsxcommon::log::error(
            absl::StrCat("[step-model] weight load failed: ", error));
        return AbortLoad();
    }

    resident_bytes_ = pw_.resident_bytes +
                      static_cast<std::uint64_t>(pw_.expert_cache_bytes);
    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();

    lsxcommon::log::info(absl::StrCat(
        "[step-model] loaded: ", pw_.layers.size(), " layers, ",
        pw_.resident_bytes >> 20, " MiB pinned + ",
        static_cast<std::uint64_t>(pw_.expert_cache_bytes) >> 20,
        " MiB expert cache (routed experts streamed; DSA indexer stored, "
        "dense-equivalent context <= ",
        static_cast<int>(step::frozen::kMaxContextTokens), " tokens)"));
    FinishLoad();
    return true;
}

gpu::InferenceResult Step5PreviewModel::Infer(const ModelRequest &req)
{
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
        return step::RunStepInference(
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

gpu::InferenceResult Step5PreviewModel::SelfCheck(const ModelRequest &req)
{
    // Chat model whose template opens the assistant turn with <think>: frame
    // the raw control prompt with the checkpoint's own template and accept
    // "Paris" within a reasoning-tolerant window.
    const runtime::ParisSelfCheckOptions opts{
        .frame_prompt = true,
        .min_tokens = 24,
        .answer_window = 96,
        .substring_match = true,
    };
    return runtime::RunParisSelfCheck(*this, req, opts);
}

bool Step5PreviewModel::BuildPromptInputIds(const ArrowTokenizerDb &tok,
                                            std::string_view raw_prompt,
                                            std::vector<int32_t> &out_ids) const
{
    ChatConversation conv;
    conv.messages.push_back({"user", std::string(raw_prompt), "", {}});
    if (step::RenderStepConversation(tok, conv, out_ids)) return true;
    out_ids.clear();
    return prompt::EncodeRawPrompt(tok, raw_prompt, out_ids);
}

bool Step5PreviewModel::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    if (step::RenderStepConversation(tok, conv, out_ids)) return true;
    const std::string text = tools::RenderConversation(conv, /*include_tools=*/true);
    if (text.empty()) return false;
    out_ids.clear();
    return BuildPromptInputIds(tok, text, out_ids);
}

bool Step5PreviewModel::ParseToolCalls(
    const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
    std::vector<ToolCall> &out_calls) const
{
    out_calls = step::ParseStepToolCalls(tok, gen_tokens);
    return !out_calls.empty();
}

} // namespace lsxcommon
