#include "lsxcommon/tokenizer_arrows.h"

#include <arrow/io/file.h>

#include <filesystem>

namespace lsxcommon {
namespace {

std::filesystem::path ResolvePinnedDir(const std::string& model_path) {
    auto pinned = std::filesystem::path(model_path) / "pinned";
    if (!std::filesystem::exists(pinned)) {
        for (const char* alt : {"logestix-qwen35moe", "logestix-glm",
                                "logestix-kimi", "logestix-gemma"}) {
            auto cand = std::filesystem::path(model_path) / alt / "pinned";
            if (std::filesystem::exists(cand)) return cand;
        }
    }
    return pinned;
}

}  // namespace

bool LoadTokenizerArrows(const std::string& model_path, ArrowTokenizerDb& out) {
    const std::filesystem::path pinned = ResolvePinnedDir(model_path);
    auto vocab = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    auto merges = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    if (!vocab.ok() || !merges.ok()) {
        out.error_msg = "tokenizer arrows not found under " + pinned.string();
        return false;
    }
    auto st = load_tokenizer_metadata(*vocab, *merges, out);
    if (!st.ok()) {
        out.error_msg = st.ToString();
        return false;
    }
    return true;
}

}  // namespace lsxcommon
