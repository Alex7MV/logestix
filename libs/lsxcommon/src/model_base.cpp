#include "lsxcommon/model_base.h"

#include <arrow/io/file.h>
#include <filesystem>
#include <spdlog/spdlog.h>

namespace lsxcommon {

bool ModelBase::LoadTokenizerDb(const std::string &model_path)
{
    if (tok_db_) return true;
    tok_db_ = std::make_unique<ArrowTokenizerDb>();
    auto pinned = std::filesystem::path(model_path) / "logestix" / "pinned";
    auto v = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    auto m = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    if (!v.ok() || !m.ok()) {
        if (tokenizer_optional_) {
            spdlog::warn("[model] tokenizer arrows not found (OK for synthetic test)");
            return true;
        }
        spdlog::error("[model] cannot open tokenizer arrows in {}", pinned.string());
        tok_db_.reset();
        return false;
    }
    auto st = load_tokenizer_metadata(*v, *m, *tok_db_);
    if (!st.ok()) {
        if (tokenizer_optional_) {
            spdlog::warn("[model] tokenizer load failed: {}", st.ToString());
            return true;
        }
        spdlog::error("[model] load_tokenizer_metadata: {}", st.ToString());
        tok_db_.reset();
        return false;
    }
    return true;
}

bool ModelBase::ValidateInferRequest(const ModelRequest &req, gpu::InferenceResult &r)
{
    if (!loaded_) {
        r.error_msg = "model not loaded";
        return false;
    }
    if (req.input_ids.empty()) {
        r.error_msg = "empty input_ids";
        return false;
    }
    if (req.max_gen_tokens <= 0) {
        r.error_msg = "max_gen_tokens must be positive";
        return false;
    }
    return true;
}

} // namespace lsxcommon