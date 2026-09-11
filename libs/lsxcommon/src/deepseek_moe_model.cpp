#include "deepseek_moe_model.h"

#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/sampling.h"

#include <absl/strings/str_cat.h>

#include "deepseek_moe_load.cuh"

namespace lsxcommon {

const std::string DeepSeekV4FlashModel::kType = "deepseek-v4-flash";

namespace gpu {
// Internal engine entry point (deepseek_moe_inference_pipeline.cu).
InferenceResult RunInferenceModel(
    const std::string &model_path,
    const std::vector<int32_t> &input_ids,
    int max_gen_tokens,
    ::lsxcommon::ArrowTokenizerDb &tok_db,
    gpu::PinnedWeights &pw,
    lsxcommon::sample::ISampler &sampler =
        lsxcommon::sample::DefaultSampler(),
    cudaStream_t ext_stream = nullptr,
    cublasHandle_t ext_cublas = nullptr);
} // namespace gpu

DeepSeekV4FlashModel::DeepSeekV4FlashModel(ModelInitConfig cfg)
    : ModelBase{std::move(cfg)}
{
}

DeepSeekV4FlashModel::~DeepSeekV4FlashModel() { Unload(); }

void DeepSeekV4FlashModel::TeardownWeights()
{
    pw_.Free();
    TeardownContext();
}

bool DeepSeekV4FlashModel::Load()
{
    Unload();  // idempotent re-load support (GLM/Kimi/Qwen parity)
    if (!BeginLoad(cfg_.model_path)) return false;
    auto topo = gpu::load_topology(cfg_.model_path);
    if (!topo.error_msg.empty()) {
        lsxcommon::log::error(absl::StrCat(
            "[deepseek-model] topology error: ", topo.error_msg));
        return AbortLoad();
    }
    // Format contract: only the NVFP4 repacked index loads. The pre-repack
    // logestix-v1 layout (E8M0 per-32 expert scales, global_scale ignored)
    // is retired — no backward compatibility (AGENTS.md).
    if (topo.format != gpu::kIndexFormatNvfp4) {
        lsxcommon::log::error(absl::StrCat(
            "[deepseek-model] unsupported index format '",
            topo.format.empty() ? "<missing>" : topo.format,
            "' — repack via: lsxforge --deepseek_nvfp4 --input <model_dir>"
            " --output <model_dir>"));
        return AbortLoad();
    }
    // Frozen-geometry passport cross-check (MODEL INGEST INVARIANT step 3):
    // the converted index must match the compile-time locked architecture.
    // Synthetic test fixtures bypass via LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY
    // (same policy as every other family).
    if (!SyntheticGeometryAllowed() &&
        (topo.num_layers != gpu::kNumLayers ||
         topo.experts_per_layer != gpu::kNumRouted)) {
        lsxcommon::log::error(absl::StrCat(
            "[deepseek-model] topology mismatches frozen passport: layers ",
            topo.num_layers, " != ", gpu::kNumLayers, " or experts ",
            topo.experts_per_layer, " != ", gpu::kNumRouted));
        return AbortLoad();
    }
    vram_ = topo.vram;

    // Persistent execution context (stream + cuBLAS handle created ONCE and
    // reused across Infer calls; GLM/Qwen-parity ownership model).
    if (!EnsureContext(/*with_cublas=*/true)) {
        lsxcommon::log::error("[deepseek-model] context create failed");
        return AbortLoad();
    }
    if (gpu::load_pinned_weights(cfg_.model_path, pw_, stream())
        != cudaSuccess) {
        lsxcommon::log::error("[deepseek-model] pinned weight load failed");
        return AbortLoad();
    }
    // Dense residency disclosure (GLM/Kimi/Qwen parity): publish the pinned
    // weight set through ResidentWeightBytes() and account it with the
    // governor permanent-weight lease.
    resident_bytes_ = pw_.resident_bytes;
    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();
    FinishLoad();
    return true;
}

gpu::InferenceResult DeepSeekV4FlashModel::Infer(const ModelRequest &req)
{
    // Model independence: assert THIS model's fast-math policy for the
    // duration of the call (constructor-time flags would leak the
    // last-constructed model's config across the registry).
    // Neutral-layer policy: the routed FFN runs lsx_nvfp4 GEMVs, whose
    // dispatch reads the process-global toggles — assert THIS model's
    // policy for the duration of the call (model-independence contract).
    const lsx_nvfp4::LsxFastMathScope fm_scope(EffectiveFastMathMode(cfg_));
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;
    if (!loaded_) { r.error_msg = "model not loaded"; return r; }
    return RunUnderResidencyLock([&] {
        return gpu::RunInferenceModel(cfg_.model_path, req.input_ids,
                                      req.max_gen_tokens, *tok_db_, pw_,
                                      lsxcommon::sample::DefaultSampler(),
                                      stream(), cublas());
    });
}

// DeepSeekV4FlashModel::Unload — canonical ModelBase template method
// (TeardownWeights override above); no per-model override.

gpu::InferenceResult DeepSeekV4FlashModel::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

} // namespace lsxcommon