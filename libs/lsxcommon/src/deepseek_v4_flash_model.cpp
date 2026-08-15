#include "deepseek_v4_flash_model.h"

#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <filesystem>
#include <spdlog/spdlog.h>

#include "lsxcommon/safetensors.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

const std::string DeepSeekV4FlashModel::kType = "deepseek-v4-flash";

namespace gpu {
// Internal engine entry points (deepseek_inference_pipeline.cu).
InferenceResult RunInferenceModel(
    const std::string &model_path,
    const std::vector<int32_t> &input_ids,
    int max_gen_tokens,
    ::lsxcommon::ArrowTokenizerDb &tok_db);
InferenceResult RunSelfCheckModel(
    const std::string &model_path,
    const std::vector<int32_t> &input_ids,
    int max_gen_tokens,
    ::lsxcommon::ArrowTokenizerDb &tok_db);
} // namespace gpu

DeepSeekV4FlashModel::DeepSeekV4FlashModel(ModelInitConfig cfg)
    : cfg_(std::move(cfg)) {}

DeepSeekV4FlashModel::~DeepSeekV4FlashModel() { Unload(); }

bool DeepSeekV4FlashModel::Load()
{
    if (loaded_) return true;
    if (cfg_.model_path.empty()) {
        spdlog::error("[model] empty model_path");
        return false;
    }
    tok_db_ = std::make_unique<ArrowTokenizerDb>();
    auto pinned = std::filesystem::path(cfg_.model_path) / "logestix" / "pinned";
    auto v = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    auto m = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    if (!v.ok() || !m.ok()) {
        spdlog::error("[model] cannot open tokenizer arrows in {}", pinned.string());
        return false;
    }
    auto st = load_tokenizer_metadata(*v, *m, *tok_db_);
    if (!st.ok()) {
        spdlog::error("[model] load_tokenizer_metadata: {}", st.ToString());
        return false;
    }
    loaded_ = true;
    return true;
}

gpu::InferenceResult DeepSeekV4FlashModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!loaded_) {
        r.error_msg = "model not loaded";
        return r;
    }
    if (req.input_ids.empty()) {
        r.error_msg = "empty input_ids";
        return r;
    }
    if (req.max_gen_tokens <= 0) {
        r.error_msg = "max_gen_tokens must be positive";
        return r;
    }
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
    gpu::InferenceResult r;
    if (!loaded_) {
        r.error_msg = "model not loaded";
        return r;
    }
    if (req.input_ids.empty()) {
        r.error_msg = "empty input_ids";
        return r;
    }
    if (req.max_gen_tokens <= 0) {
        r.error_msg = "max_gen_tokens must be positive";
        return r;
    }
    return gpu::RunSelfCheckModel(cfg_.model_path, req.input_ids,
                                  req.max_gen_tokens, *tok_db_);
}

ArrowTokenizerDb &DeepSeekV4FlashModel::Tokenizer()
{
    return *tok_db_;
}

} // namespace lsxcommon