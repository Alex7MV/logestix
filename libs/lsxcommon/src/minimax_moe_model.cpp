#include "minimax_moe_model.h"

#include "minimax_moe_constants.h"
#include "lsxcommon/minimax_moe_tool_format.h"

#include "lsxcommon/log.h"
#include "lsxcommon/minimax_moe_kernels.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/tokenizer.h"

#include <absl/strings/str_cat.h>

#include <cmath>
#include <cstdlib>

namespace lsxcommon {

const std::string MinimaxM3Model::kType = "minimax-m3";

MinimaxM3Model::MinimaxM3Model(ModelInitConfig cfg) : ModelBase{std::move(cfg)} {}

MinimaxM3Model::~MinimaxM3Model() { Unload(); }

void MinimaxM3Model::TeardownWeights()
{
    scratch_.Free();
    session_.Free();
    if (expert_cache_ready_) {
        minimax::FreeMinimaxExpertCache(cache_);
        expert_cache_ready_ = false;
    }
    experts_.Free();
    pw_.Free();
    TeardownContext();
}

bool MinimaxM3Model::Load()
{
    Unload();  // idempotent re-load support
    if (!BeginLoad(cfg_.model_path)) return false;

    minimax::MinimaxTopology top = minimax::LoadMinimaxTopology(cfg_.model_path);
    if (top.num_layers <= 0) {
        lsxcommon::log::error(absl::StrCat(
            "[minimax-model] topology error: ",
            "model.index.json unreadable or empty"));
        return AbortLoad();
    }
    // Format contract: no backward compatibility (AGENTS.md).
    // The schema generation is checked by the loader that parses the
    // index (lsxcommon::IndexLayoutSupported); this used to compare a
    // per-model tag that travelled here through error_msg.

    // Frozen-geometry passport cross-check (Model Ingestion Protocol step 3).
    using namespace minimax::frozen;
    if (!SyntheticGeometryAllowed() &&
        (top.num_layers != kLayers || top.hidden_size != kHidden ||
         top.num_experts != kNumExperts || top.num_attention_heads != kHeads ||
         top.num_key_value_heads != kKvHeads || top.head_dim != kHeadDim ||
         top.vocab_size != kVocab ||
         top.first_moe_layer != kFirstMoeLayer ||
         top.first_sparse_layer != kFirstSparseLayer ||
         std::fabs(top.rope_theta - kRopeTheta) > kRopeTheta * 1e-6)) {
        lsxcommon::log::error(absl::StrCat(
            "[minimax-model] topology mismatches frozen passport (layers ",
            top.num_layers, "/", kLayers, ", hidden ", top.hidden_size, "/",
            kHidden, ", experts ", top.num_experts, "/", kNumExperts,
            ", first sparse ", top.first_sparse_layer, "/", kFirstSparseLayer,
            ", theta ", top.rope_theta, "/", kRopeTheta, ")"));
        return AbortLoad();
    }
    vram_ = top.vram;

    if (!EnsureContext()) {
        lsxcommon::log::error("[minimax-model] context create failed");
        return AbortLoad();
    }
    std::string error;
    if (!minimax::LoadMinimaxDenseWeights(cfg_.model_path, pw_, stream(),
                                          error)) {
        lsxcommon::log::error(
            absl::StrCat("[minimax-model] pinned weight load failed: ", error));
        return AbortLoad();
    }

    // Routed-expert host store (mmap views + LRU) — never pinned.
    if (!minimax::LoadMinimaxExpertStore(cfg_.model_path, experts_, error)) {
        lsxcommon::log::error(
            absl::StrCat("[minimax-model] expert store load failed: ", error));
        return AbortLoad();
    }

    // Device slot LRU: sized from the VRAM still free after the pinned graph.
    if (!minimax::AllocateMinimaxExpertCache(experts_, cache_, 0,
                                          top.num_experts_per_tok, error)) {
        lsxcommon::log::error(
            absl::StrCat("[minimax-model] expert cache alloc failed: ", error));
        return AbortLoad();
    }
    expert_cache_ready_ = true;

    resident_bytes_ = pw_.resident_bytes;
    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();
    // The expert cache is the adaptive slice: a refusal is non-fatal.
    if (!AcquireExpertCacheLease(cache_.DeviceBytes(), 0, cache_.DeviceBytes()))
        lsxcommon::log::warn(absl::StrCat(
            "[minimax-model] governor kept a smaller expert-cache lease than ",
            cache_.DeviceBytes() >> 20, " MiB (continuing)"));

    lsxcommon::log::info(absl::StrCat(
        "[minimax-model] loaded: ", pw_.layers.size(), " layers, ",
        pw_.resident_bytes >> 20, " MiB pinned + ", cache_.slots,
        " expert slots x ", (cache_.pack_stride + cache_.scale_stride) >> 20,
        " MiB"));
    FinishLoad();
    return true;
}

gpu::InferenceResult MinimaxM3Model::Infer(const ModelRequest &req)
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
        return minimax::RunMinimaxInference(
            cfg_.model_path, req.input_ids, req.max_gen_tokens, pw_, experts_,
            cache_, session_, scratch_,
            lsxcommon::sample::SamplerFor(req.sampling), stream(),
            req.ignore_eos, &gs);
    });
    if (r.ok) {
        r.output_text = runtime::DecodeGeneratedText(
            *tok_db_, r.gen_tokens.data(), r.gen_tokens.size());
        r.num_layers = pw_.top.num_layers;
        r.weight_resident_bytes = resident_bytes_ + scratch_.arena_bytes +
                                  session_.Bytes() + cache_.DeviceBytes();
    }
    return r;
}

gpu::InferenceResult MinimaxM3Model::SelfCheck(const ModelRequest &req)
{
    // Chat model with a thinking switch: frame the raw control prompt with the
    // checkpoint's own template (thinking disabled => the generation prompt
    // starts with the think-END marker) and accept "Paris" within a short
    // window. Same deviation as ornith (Model Ingestion Protocol step 8).
    const runtime::ParisSelfCheckOptions opts{
        .frame_prompt = true,
        .min_tokens = 16,
        .answer_window = 16,
        .substring_match = true,
    };
    return runtime::RunParisSelfCheck(*this, req, opts);
}

bool MinimaxM3Model::BuildPromptInputIds(
    const ArrowTokenizerDb &tok, std::string_view raw_prompt,
    std::vector<int32_t> &out_ids) const
{
    ChatConversation conv;
    conv.messages.push_back({"user", std::string(raw_prompt), "", {}});
    if (minimax::RenderMinimaxConversation(tok, conv, out_ids)) return true;
    out_ids.clear();
    return prompt::EncodeRawPrompt(tok, raw_prompt, out_ids);
}

bool MinimaxM3Model::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    if (minimax::RenderMinimaxConversation(tok, conv, out_ids)) return true;
    // Fallback: the shared Hermes renderer (kept as a floor, not a pass).
    const std::string text = tools::RenderConversation(conv, /*include_tools=*/true);
    if (text.empty()) return false;
    out_ids.clear();
    return BuildPromptInputIds(tok, text, out_ids);
}

bool MinimaxM3Model::ParseToolCalls(
    const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
    std::vector<ToolCall> &out_calls) const
{
    return minimax::ParseMinimaxToolCalls(tok, gen_tokens, out_calls);
}

} // namespace lsxcommon
