#include "deepseek_v4_flash_model.h"

#include <spdlog/spdlog.h>

#include "lsxcommon/model_runtime.h"
#include "lsxcommon/safetensors.h"

namespace lsxcommon {

const std::string DeepSeekV4FlashModel::kType = "deepseek-v4-flash";

namespace gpu {
// Internal engine entry point (deepseek_inference_pipeline.cu).
InferenceResult RunInferenceModel(
    const std::string &model_path,
    const std::vector<int32_t> &input_ids,
    int max_gen_tokens,
    ::lsxcommon::ArrowTokenizerDb &tok_db);
} // namespace gpu

DeepSeekV4FlashModel::DeepSeekV4FlashModel(ModelInitConfig cfg)
    : ModelBase{std::move(cfg)} {}

DeepSeekV4FlashModel::~DeepSeekV4FlashModel() { Unload(); }

bool DeepSeekV4FlashModel::Load()
{
    if (loaded_) return true;
    if (cfg_.model_path.empty()) {
        spdlog::error("[model] empty model_path");
        return false;
    }
    if (!LoadTokenizerDb(cfg_.model_path)) return false;
    loaded_ = true;
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

SafetensorsModel DeepSeekV4FlashModel::Inspect()
{
    return lsxcommon::scan_directory(cfg_.model_path);
}

gpu::InferenceResult DeepSeekV4FlashModel::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

} // namespace lsxcommon