#include "deepseek_v4_flash_model.h"

#include <spdlog/spdlog.h>

#include "lsxcommon/model_runtime.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/sampling.h"

namespace lsxcommon {

const std::string DeepSeekV4FlashModel::kType = "deepseek-v4-flash";

namespace gpu {
// Internal engine entry point (deepseek_inference_pipeline.cu).
InferenceResult RunInferenceModel(
    const std::string &model_path,
    const std::vector<int32_t> &input_ids,
    int max_gen_tokens,
    ::lsxcommon::ArrowTokenizerDb &tok_db,
    lsxcommon::sample::ISampler &sampler =
        lsxcommon::sample::DefaultSampler());
} // namespace gpu

DeepSeekV4FlashModel::DeepSeekV4FlashModel(ModelInitConfig cfg)
    : ModelBase{std::move(cfg)} {}

DeepSeekV4FlashModel::~DeepSeekV4FlashModel() { Unload(); }

bool DeepSeekV4FlashModel::Load()
{
    if (!BeginLoad(cfg_.model_path)) return false;
    FinishLoad();
    return true;
}

gpu::InferenceResult DeepSeekV4FlashModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;
    return gpu::RunInferenceModel(cfg_.model_path, req.input_ids,
                                  req.max_gen_tokens, *tok_db_);
}

void DeepSeekV4FlashModel::Unload()
{
    // RunInferenceModel loads pinned weights per-call and frees them on return.
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
                                      *tok_db_);
    if (!res.ok) { r.error_msg = res.error_msg; return r; }
    r.ok = true;
    r.num_tokens = res.num_tokens;
    r.num_layers = res.num_layers;
    r.total_time_ms = res.total_time_ms;
    r.tokens_per_sec = res.tokens_per_sec;
    r.resident_bytes = res.weight_resident_bytes;
    return r;
}

} // namespace lsxcommon