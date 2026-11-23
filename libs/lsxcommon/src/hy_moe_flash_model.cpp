#include "hy_moe_flash_model.h"

#include "hy_moe_flash_constants.h"

#include "lsxcommon/hy_moe_flash_tool_format.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/tokenizer.h"

#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>

namespace lsxcommon {

const std::string Hy4FlashModel::kType = hy4::kType;

Hy4FlashModel::Hy4FlashModel(ModelInitConfig cfg) : ModelBase{std::move(cfg)} {}

Hy4FlashModel::~Hy4FlashModel() { Unload(); }

void Hy4FlashModel::TeardownWeights()
{
    scratch_.Free();
    session_.Free();
    pw_.Free();
    if (expert_db_ready_) {
        hy4::gpu::FreeExpertIndex(expert_db_);
        expert_db_ready_ = false;
    }
    TeardownContext();
}

bool Hy4FlashModel::Load()
{
    Unload();  // idempotent re-load support
    if (!BeginLoad(cfg_.model_path)) return false;

    hy4::gpu::Topology top = hy4::gpu::LoadTopology(cfg_.model_path);
    const std::string err = top.error_msg;
    top.error_msg.clear();
    if (top.num_layers <= 0) {
        log::error(absl::StrCat("[hy4-model] topology error: ",
                                err.empty() ? "model.index.json unreadable"
                                            : err));
        return AbortLoad();
    }
    // Format contract: no backward compatibility (AGENTS.md).
    if (!hy4::gpu::IsSupportedFormat(top)) {
        log::error(absl::StrCat("[hy4-model] unsupported index format '",
                                top.format,
                                "' — convert via: lsxforge --hy4_flash --input "
                                "<model_dir> --output <model_dir>"));
        return AbortLoad();
    }
    // One card name per artifact: the registry id and the index must agree, or
    // a tree converted from a different recipe would decode as this model.
    if (!hy4::gpu::MatchesModelType(top, kType)) {
        log::error(absl::StrFormat(
            "[hy4-model] index names card '%s' which folds to '%s', not '%s'",
            top.model, hy4::gpu::NormalizeCardName(top.model), kType));
        return AbortLoad();
    }
    // Frozen-geometry passport cross-check (Model Ingestion Protocol step 3).
    if (!SyntheticGeometryAllowed()) {
        std::string why;
        if (!hy4::gpu::ValidateGeometry(top, &why)) {
            log::error(absl::StrCat("[hy4-model] topology mismatches frozen "
                                    "passport: ", why));
            return AbortLoad();
        }
        // The sparse layers must have been counted: an index that silently lost
        // its Engram cut is a converter from a different recipe.
        if (top.deferred_engram <= 0) {
            log::error("[hy4-model] index carries no Engram cut record");
            return AbortLoad();
        }
    }
    vram_ = top.vram;
    if (top.eos_token_id >= 0) eos_ids_.push_back(top.eos_token_id);

    if (!EnsureContext()) {
        log::error("[hy4-model] context create failed");
        return AbortLoad();
    }
    std::string error;
    if (!hy4::gpu::LoadHy4Weights(cfg_.model_path, top, pw_, stream(), error)) {
        log::error(absl::StrCat("[hy4-model] weight load failed: ", error));
        return AbortLoad();
    }

    resident_bytes_ = pw_.resident_bytes;
    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();
    // Soft expert-cache lease: the device slot cache is sized from free VRAM at
    // the first Infer, so the lease is advisory.
    AcquireExpertCacheLease(0, 0, 0);

    expert_db_ = hy4::gpu::LoadExpertIndex(cfg_.model_path, top.num_layers,
                                          top.experts_per_layer,
                                          /*max_open=*/4096);
    expert_db_ready_ = true;

    log::info(absl::StrFormat(
        "[hy4-model] loaded: %d layers, %d experts/layer, %llu MiB pinned, "
        "Engram/ViT/DFly2 deferred (engram=%lld vision=%lld dfly=%lld)",
        top.num_layers, top.experts_per_layer,
        static_cast<unsigned long long>(pw_.resident_bytes >> 20),
        static_cast<long long>(top.deferred_engram),
        static_cast<long long>(top.deferred_vision),
        static_cast<long long>(top.deferred_dfly)));
    FinishLoad();
    return true;
}

gpu::InferenceResult Hy4FlashModel::Infer(const ModelRequest &req)
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
        return hy4::gpu::RunHy4Inference(
            cfg_.model_path, req.input_ids, req.max_gen_tokens, eos_ids_, pw_,
            expert_db_, session_, scratch_,
            lsxcommon::sample::SamplerFor(req.sampling), stream(),
            req.ignore_eos, &gs);
    });
    if (r.ok) {
        r.output_text = runtime::DecodeGeneratedText(
            *tok_db_, r.gen_tokens.data(), r.gen_tokens.size());
        r.num_layers = pw_.num_layers;
        r.weight_resident_bytes =
            resident_bytes_ + scratch_.arena_bytes + scratch_.slot_bytes +
            static_cast<std::uint64_t>(session_.Bytes());
        r.expert_cache_slots = scratch_.slot_count;
        r.num_tokens = static_cast<int>(r.gen_tokens.size());
    }
    return r;
}

gpu::InferenceResult Hy4FlashModel::SelfCheck(const ModelRequest &req)
{
    // The checkpoint reasons before it answers when the template asks it to, but
    // the engine serves `no_think` by default, so the control answer lands in
    // the first few tokens.
    const runtime::ParisSelfCheckOptions opts{
        .frame_prompt = true,
        .min_tokens = 4,
        .answer_window = 24,
        .substring_match = true,
    };
    return runtime::RunParisSelfCheck(*this, req, opts);
}

bool Hy4FlashModel::BuildPromptInputIds(const ArrowTokenizerDb &tok,
                                        std::string_view raw_prompt,
                                        std::vector<int32_t> &out_ids) const
{
    ChatConversation conv;
    ChatMessage m;
    m.role = "user";
    m.content = std::string(raw_prompt);
    conv.messages.push_back(std::move(m));
    if (hy4::RenderHy4Conversation(tok, conv, out_ids)) return true;
    out_ids.clear();
    return prompt::EncodeRawPrompt(tok, raw_prompt, out_ids);
}

bool Hy4FlashModel::BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                              const ChatConversation &conv,
                                              std::vector<int32_t> &out_ids) const
{
    if (hy4::RenderHy4Conversation(tok, conv, out_ids)) return true;
    // Fallback: the shared Hermes renderer (a floor, not a pass).
    const std::string text =
        tools::RenderConversation(conv, /*include_tools=*/true);
    if (text.empty()) return false;
    out_ids.clear();
    return BuildPromptInputIds(tok, text, out_ids);
}

bool Hy4FlashModel::ParseToolCalls(const ArrowTokenizerDb &tok,
                                  const std::vector<int32_t> &gen_tokens,
                                  std::vector<ToolCall> &out_calls) const
{
    return hy4::ParseHy4ToolCalls(tok, gen_tokens, out_calls);
}

} // namespace lsxcommon
