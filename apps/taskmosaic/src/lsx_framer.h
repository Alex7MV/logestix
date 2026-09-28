#pragma once

#include <absl/container/flat_hash_map.h>

#include <memory>
#include <mutex>
#include <string>

#include "framer.h"
#include "lsxcommon/tokenizer.h"

namespace taskmosaic {

/// Real framer: per-family chat framing via lsxcommon::FramePromptForModel,
/// full decode via runtime::DecodeGeneratedText, and the Arrow tokenizer-backed
/// incremental decoder (tokenizer DB cached per model path).
class LsxPromptFramer : public PromptFramer {
public:
    bool Encode(std::string_view model_path, std::string_view raw_prompt,
                std::vector<std::int32_t>& out_ids) override;
    std::string Decode(std::string_view model_path,
                       const std::vector<std::int32_t>& ids) override;
    std::unique_ptr<StreamingDecoder> MakeDecoder(
        std::string_view model_path) override;

private:
    lsxcommon::ArrowTokenizerDb* TokenizerFor(std::string_view model_path);
    std::mutex mtx_;
    absl::flat_hash_map<std::string,
                        std::unique_ptr<lsxcommon::ArrowTokenizerDb>>
        toks_;
};

}  // namespace taskmosaic
