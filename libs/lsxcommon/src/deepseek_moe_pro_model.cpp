#include "deepseek_moe_pro_model.h"

#include "lsxcommon/log.h"
#include "lsxcommon/index_layout.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/tokenizer.h"
#include "lsxcommon/deepseek_moe_tool_format.h"

#include <absl/strings/str_cat.h>
#include <cstring>

#include "deepseek_moe_pro_load.cuh"

namespace lsxcommon {

const std::string DeepSeekV4ProModel::kType = "deepseek-v4-pro-0813";

namespace deepseek_pro {
namespace gpu {
// Internal engine entry point (deepseek_moe_pro_inference_pipeline.cu).
lsxcommon::gpu::InferenceResult RunInferenceModel(
    const std::string &model_path,
    const std::vector<int32_t> &input_ids,
    int max_gen_tokens,
    ::lsxcommon::ArrowTokenizerDb &tok_db,
    deepseek_pro::gpu::PinnedWeights &pw,
    lsxcommon::sample::ISampler &sampler =
        lsxcommon::sample::DefaultSampler(),
    cudaStream_t ext_stream = nullptr,
    bool ignore_eos = false,
    const lsxcommon::GenerationStream *gs = nullptr);
} // namespace gpu
} // namespace deepseek_pro

DeepSeekV4ProModel::DeepSeekV4ProModel(ModelInitConfig cfg)
    : ModelBase{std::move(cfg)}
{
}

DeepSeekV4ProModel::~DeepSeekV4ProModel() { Unload(); }

void DeepSeekV4ProModel::TeardownWeights()
{
    pw_.Free();
    TeardownContext();
}

bool DeepSeekV4ProModel::Load()
{
    Unload();  // idempotent re-load support (GLM/Kimi/Qwen parity)
    if (!BeginLoad(cfg_.model_path)) return false;
    auto topo = deepseek_pro::gpu::load_topology(cfg_.model_path);
    if (!topo.error_msg.empty()) {
        lsxcommon::log::error(absl::StrCat(
            "[deepseek-pro-model] topology error: ", topo.error_msg));
        return AbortLoad();
    }
    // Format contract: no backward compatibility (AGENTS.md) — only the
    // NVFP4 tree this converter writes loads.
    if (!lsxcommon::IndexLayoutSupported(topo.layout)) {
        lsxcommon::log::error(absl::StrCat(
            "[deepseek-pro-model] unsupported index format '",
            std::to_string(topo.layout),
            "' — convert via: lsxforge --deepseek_pro --input <model_dir>"
            " --output <model_dir>"));
        return AbortLoad();
    }
    // Frozen-geometry passport cross-check (MODEL INGEST INVARIANT step 3).
    // Synthetic fixtures bypass through the shared runtime env var.
    if (!SyntheticGeometryAllowed() &&
        (topo.num_layers != deepseek_pro::kLayers ||
         topo.experts_per_layer != deepseek_pro::kNumRouted)) {
        lsxcommon::log::error(absl::StrCat(
            "[deepseek-pro-model] topology mismatches frozen passport: layers ",
            topo.num_layers, " != ", deepseek_pro::kLayers,
            " or experts ", topo.experts_per_layer, " != ",
            deepseek_pro::kNumRouted));
        return AbortLoad();
    }
    vram_ = topo.vram;

    // Persistent execution context (stream created ONCE and reused across
    // Infer calls; GLM/Qwen-parity ownership model).
    if (!EnsureContext()) {
        lsxcommon::log::error("[deepseek-pro-model] context create failed");
        return AbortLoad();
    }
    if (deepseek_pro::gpu::load_pinned_weights(cfg_.model_path, pw_, stream())
        != cudaSuccess) {
        lsxcommon::log::error(
            "[deepseek-pro-model] pinned weight load failed");
        return AbortLoad();
    }
    resident_bytes_ = pw_.resident_bytes;
    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();
    FinishLoad();
    return true;
}

gpu::InferenceResult DeepSeekV4ProModel::Infer(const ModelRequest &req)
{
    // Model independence: assert THIS model's fast-math policy for the
    // duration of the call. The routed FFN runs neutral lsx_nvfp4 GEMVs whose
    // dispatch reads the process-global toggles.
    const lsx_nvfp4::LsxFastMathScope fm_scope(EffectiveFastMathMode(cfg_));
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;
    if (!loaded_) { r.error_msg = "model not loaded"; return r; }
    lsxcommon::GenerationStream gs;
    gs.on_token = req.token_callback ? &req.token_callback : nullptr;
    gs.abort_flag = req.abort_flag;
    return RunUnderResidencyLock([&] {
        return deepseek_pro::gpu::RunInferenceModel(
            cfg_.model_path, req.input_ids, req.max_gen_tokens, *tok_db_, pw_,
            lsxcommon::sample::SamplerFor(req.sampling), stream(),
            req.ignore_eos, &gs);
    });
}

gpu::InferenceResult DeepSeekV4ProModel::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

// ── DeepSeek-V4-Pro native DSML tool format (encoding/encoding_dsv4.py) ────
namespace {
constexpr DsmlTags kProDsmlTags{"tool_calls", "invoke", "parameter", ""};
constexpr const char *kDsmlUserToken = "<｜User｜>";
} // namespace

bool DeepSeekV4ProModel::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    std::string text;
    if (!RenderDsmlConversation(conv, kProDsmlTags, text)) return false;
    out_ids.clear();
    if (arrow_tokenizer_lookup(tok, kDsmlUserToken,
                               std::strlen(kDsmlUserToken)) < 0)
        return BuildPromptInputIds(tok, text, out_ids);
    return AppendDeepSeekIds(tok, text, kProDsmlTags, out_ids);
}

bool DeepSeekV4ProModel::ParseToolCalls(
    const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
    std::vector<ToolCall> &out_calls) const
{
    return ParseDeepSeekToolCalls(tok, gen_tokens, kProDsmlTags, out_calls);
}

} // namespace lsxcommon
