#include "deepseek_v4_flash_model.h"

#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/sampling.h"

#include <absl/strings/str_cat.h>

#include "deepseek_inference_load.cuh"

namespace lsxcommon {

const std::string DeepSeekV4FlashModel::kType = "deepseek-v4-flash";

namespace gpu {
// Internal engine entry point (deepseek_inference_pipeline.cu).
InferenceResult RunInferenceModel(
    const std::string &model_path,
    const std::vector<int32_t> &input_ids,
    int max_gen_tokens,
    ::lsxcommon::ArrowTokenizerDb &tok_db,
    gpu::PinnedWeights &pw,
    lsxcommon::sample::ISampler &sampler =
        lsxcommon::sample::DefaultSampler());
} // namespace gpu

DeepSeekV4FlashModel::DeepSeekV4FlashModel(ModelInitConfig cfg)
    : ModelBase{std::move(cfg)} {}

DeepSeekV4FlashModel::~DeepSeekV4FlashModel() { Unload(); }

bool DeepSeekV4FlashModel::Load()
{
    if (!BeginLoad(cfg_.model_path)) return false;
    auto topo = gpu::load_topology(cfg_.model_path);
    if (!topo.error_msg.empty()) {
        lsxcommon::log::error(absl::StrCat(
            "[deepseek-model] topology error: ", topo.error_msg));
        return false;
    }
    vram_ = topo.vram;

    // Fast-load the pinned weights ONCE (GLM-parity ownership model):
    // Infer/Benchmark reuse them instead of paying full load cost per call.
    if (!EnsureContext(/*with_cublas=*/false)) {
        lsxcommon::log::error("[deepseek-model] stream create failed");
        return false;
    }
    if (gpu::load_pinned_weights(cfg_.model_path, pw_, stream())
        != cudaSuccess) {
        lsxcommon::log::error("[deepseek-model] pinned weight load failed");
        pw_.Free();
        return false;
    }
    FinishLoad();
    return true;
}

gpu::InferenceResult DeepSeekV4FlashModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;
    if (!loaded_) { r.error_msg = "model not loaded"; return r; }
    return RunUnderResidencyLock([&] {
        return gpu::RunInferenceModel(cfg_.model_path, req.input_ids,
                                      req.max_gen_tokens, *tok_db_, pw_);
    });
}

void DeepSeekV4FlashModel::Unload()
{
    pw_.Free();
    TeardownContext();
    tok_db_.reset();
    loaded_ = false;
}

gpu::InferenceResult DeepSeekV4FlashModel::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

gpu::BenchmarkResult DeepSeekV4FlashModel::Benchmark(int num_tokens)
{
    gpu::BenchmarkResult r;
    if (!loaded_) { r.error_msg = "model not loaded"; return r; }
    const std::vector<int32_t> prompt = {0};
    auto res = gpu::RunInferenceModel(cfg_.model_path, prompt, num_tokens,
                                      *tok_db_, pw_);
    if (!res.ok) { r.error_msg = res.error_msg; return r; }
    runtime::FillBenchmarkResult(r, res.total_time_ms, res.num_tokens,
                                 res.num_layers,
                                 res.weight_resident_bytes);
    return r;
}

} // namespace lsxcommon