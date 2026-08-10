#include "glm_v5_model.h"

#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <filesystem>
#include <spdlog/spdlog.h>

#include "lsxcommon/safetensors.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

const std::string GlmV5Model::kType = "glm-5.2";

GlmV5Model::GlmV5Model(ModelInitConfig cfg)
    : cfg_(std::move(cfg)) {}

GlmV5Model::~GlmV5Model() { Unload(); }

bool GlmV5Model::Load()
{
    if (loaded_) return true;
    if (cfg_.model_path.empty()) {
        spdlog::error("[glm-model] empty model_path");
        return false;
    }
    tok_db_ = std::make_unique<ArrowTokenizerDb>();
    auto pinned = std::filesystem::path(cfg_.model_path) / "logestix" / "pinned";
    auto v = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    auto m = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    if (!v.ok() || !m.ok()) {
        spdlog::error("[glm-model] cannot open tokenizer arrows in {}",
                      pinned.string());
        return false;
    }
    auto st = load_tokenizer_metadata(*v, *m, *tok_db_);
    if (!st.ok()) {
        spdlog::error("[glm-model] load_tokenizer_metadata: {}", st.ToString());
        return false;
    }
    loaded_ = true;
    return true;
}

gpu::InferenceResult GlmV5Model::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!loaded_) {
        r.error_msg = "model not loaded";
        return r;
    }
    r.error_msg = "GLM inference pipeline not implemented (glm-5.2)";
    return r;
}

void GlmV5Model::Unload()
{
    tok_db_.reset();
    loaded_ = false;
}

SafetensorsModel GlmV5Model::Inspect()
{
    return lsxcommon::scan_directory(cfg_.model_path);
}

gpu::InferenceResult GlmV5Model::SelfCheck(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!loaded_) {
        r.error_msg = "model not loaded";
        return r;
    }
    r.error_msg = "GLM self-check not implemented (glm-5.2)";
    return r;
}

ArrowTokenizerDb &GlmV5Model::Tokenizer()
{
    return *tok_db_;
}

} // namespace lsxcommon