#include "beam_moe_model.h"

#include "beam_moe_constants.h"
#include "lsxcommon/beam_moe_tool_format.h"

#include "lsxcommon/log.h"
#include "lsxcommon/lsx_moe_inference.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/tokenizer.h"

#include <absl/strings/str_cat.h>

#include <cstdlib>

namespace lsxcommon {

const std::string BeamMoeModel::kType = "beam";

BeamMoeModel::BeamMoeModel(ModelInitConfig cfg) : ModelBase{std::move(cfg)} {}

BeamMoeModel::~BeamMoeModel() { Unload(); }

void BeamMoeModel::TeardownWeights()
{
    scratch_.Free();
    session_.Free();
    pw_.Free();
    TeardownContext();
}

bool BeamMoeModel::Load()
{
    Unload();
    if (!BeginLoad(cfg_.model_path)) return false;

    beam::BeamTopology top = beam::LoadBeamTopology(cfg_.model_path);
    const std::string fmt = top.error_msg;
    top.error_msg.clear();
    if (top.num_layers <= 0) {
        lsxcommon::log::error(absl::StrCat(
            "[beam-model] topology error: ",
            fmt.empty() ? "model.index.json unreadable" : fmt));
        return AbortLoad();
    }
    if (!fmt.empty() && fmt != beam::frozen::kFormatTag) {
        lsxcommon::log::error(absl::StrCat(
            "[beam-model] unsupported index format '", fmt,
            "' — convert via: lsxforge --beam --input <model_dir> --output "
            "<model_dir>"));
        return AbortLoad();
    }
    using namespace beam::frozen;
    if (!SyntheticGeometryAllowed() &&
        (top.num_layers != kLayers || top.hidden_size != kHidden ||
         top.num_experts != kNumExperts || top.num_attention_heads != kHeads ||
         top.num_key_value_heads != kKvHeads || top.head_dim != kHeadDim ||
         top.vocab_size != kVocab ||
         top.moe_intermediate_size != kMoeIntermediate ||
         top.dense_intermediate_size != kDenseIntermediate ||
         top.num_experts_per_tok != kTopK || top.n_group != kNGroups ||
         top.topk_group != kTopkGroup || top.sliding_window != kWindow)) {
        lsxcommon::log::error(absl::StrCat(
            "[beam-model] topology mismatches frozen passport (layers ",
            top.num_layers, "/", static_cast<int>(kLayers), ", hidden ",
            top.hidden_size, "/", static_cast<int>(kHidden), ", experts ",
            top.num_experts, "/", static_cast<int>(kNumExperts), ", window ",
            top.sliding_window, "/", static_cast<int>(kWindow), ")"));
        return AbortLoad();
    }
    vram_ = top.vram;
    eos_ids_ = top.eos_ids;
    if (eos_ids_.empty() && top.eos_token_id >= 0)
        eos_ids_.push_back(top.eos_token_id);

    if (!EnsureContext()) {
        lsxcommon::log::error("[beam-model] context create failed");
        return AbortLoad();
    }
    std::string error;
    if (!beam::LoadBeamWeights(cfg_.model_path, pw_, stream(), error)) {
        lsxcommon::log::error(
            absl::StrCat("[beam-model] pinned weight load failed: ", error));
        return AbortLoad();
    }

    // Device slot LRU for the streamed experts. Sized from the free VRAM the
    // engine sees now (the dense graph is already resident).
    int slots = 512;
    if (const char *env = std::getenv("LSX_BEAM_EXPERT_CACHE_SLOTS")) {
        const long v = std::atol(env);
        if (v > 0) slots = static_cast<int>(v);
    }
    if (!moe::AllocateMoEExpertCache(pw_.expert_store, slots, scratch_.cache,
                                     error)) {
        lsxcommon::log::error(
            absl::StrCat("[beam-model] expert cache alloc failed: ", error));
        return AbortLoad();
    }
    pw_.resident_bytes += scratch_.cache.DeviceBytes();

    resident_bytes_ = pw_.resident_bytes;
    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();

    lsxcommon::log::info(absl::StrCat(
        "[beam-model] loaded: ", pw_.layers.size(), " layers, ",
        pw_.resident_bytes >> 20,
        " MiB pinned + expert cache; routed experts streamed from ",
        pw_.expert_store.num_layers * pw_.expert_store.num_experts,
        " files"));
    FinishLoad();
    return true;
}

gpu::InferenceResult BeamMoeModel::Infer(const ModelRequest &req)
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
        return beam::RunBeamInference(
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
        // Keep the generated tail for session continuation (an aborted call
        // publishes no tokens and must not clobber the previous tail).
        if (!r.gen_tokens.empty()) session_.gen_tail = r.gen_tokens;
    }
    return r;
}

gpu::InferenceResult BeamMoeModel::SelfCheck(const ModelRequest &req)
{
    // Harmony reasons in the analysis channel before the final answer, so the
    // probe is framed and searches a wide window for "Paris".
    const runtime::ParisSelfCheckOptions opts{
        .frame_prompt = true,
        .min_tokens = 256,
        .answer_window = 96,
        .substring_match = true,
    };
    return runtime::RunParisSelfCheck(*this, req, opts);
}

bool BeamMoeModel::BuildPromptInputIds(
    const ArrowTokenizerDb &tok, std::string_view raw_prompt,
    std::vector<int32_t> &out_ids) const
{
    ChatConversation conv;
    conv.messages.push_back({"user", std::string(raw_prompt), "", {}});
    if (beam::RenderBeamConversation(tok, conv, out_ids)) return true;
    out_ids.clear();
    return prompt::EncodeRawPrompt(tok, raw_prompt, out_ids);
}

bool BeamMoeModel::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    if (beam::RenderBeamConversation(tok, conv, out_ids)) return true;
    const std::string text =
        tools::RenderConversation(conv, /*include_tools=*/true);
    if (text.empty()) return false;
    out_ids.clear();
    return BuildPromptInputIds(tok, text, out_ids);
}

bool BeamMoeModel::ParseToolCalls(
    const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
    std::vector<ToolCall> &out_calls) const
{
    if (beam::ParseBeamToolCalls(tok, gen_tokens, out_calls)) return true;
    return ModelBase::ParseToolCalls(tok, gen_tokens, out_calls);
}

} // namespace lsxcommon
