#include "lsx_framer.h"

#include "lsxcommon/incremental_detokenizer.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/prompt_bridge.h"
#include "lsxcommon/tokenizer_arrows.h"

#include <cstdint>
#include <string>
#include <utility>

namespace taskmosaic {
namespace {

class ArrowStreamingDecoder : public StreamingDecoder {
public:
    explicit ArrowStreamingDecoder(lsxcommon::IncrementalDetokenizer d)
        : d_(std::move(d)) {}
    std::string Push(std::int32_t id) override { return d_.Push(id); }
    std::string Flush() override { return d_.Flush(); }

private:
    lsxcommon::IncrementalDetokenizer d_;
};

}  // namespace

bool LsxPromptFramer::Encode(std::string_view model_path,
                             std::string_view raw_prompt,
                             std::vector<std::int32_t>& out_ids) {
    return lsxcommon::FramePromptForModel(std::string(model_path), raw_prompt,
                                          out_ids);
}

lsxcommon::ArrowTokenizerDb* LsxPromptFramer::TokenizerFor(
    std::string_view model_path) {
    std::lock_guard<std::mutex> lock(mtx_);
    const std::string key(model_path);
    auto it = toks_.find(key);
    if (it != toks_.end()) return it->second.get();
    auto db = std::make_unique<lsxcommon::ArrowTokenizerDb>();
    if (!lsxcommon::LoadTokenizerArrows(key, *db)) return nullptr;
    lsxcommon::ArrowTokenizerDb* raw = db.get();
    toks_.emplace(key, std::move(db));
    return raw;
}

std::string LsxPromptFramer::Decode(std::string_view model_path,
                                    const std::vector<std::int32_t>& ids) {
    lsxcommon::ArrowTokenizerDb* db = TokenizerFor(model_path);
    if (db == nullptr) return {};
    return lsxcommon::runtime::DecodeGeneratedText(*db, ids);
}

std::unique_ptr<StreamingDecoder> LsxPromptFramer::MakeDecoder(
    std::string_view model_path) {
    lsxcommon::ArrowTokenizerDb* db = TokenizerFor(model_path);
    if (db == nullptr) return nullptr;
    return std::make_unique<ArrowStreamingDecoder>(
        lsxcommon::IncrementalDetokenizer(
            lsxcommon::IncrementalDetokenizer::ArrowDecoder(*db)));
}

bool LsxPromptFramer::EncodeConversation(std::string_view model_path,
                                         const lsxcommon::ChatConversation& conv,
                                         std::vector<std::int32_t>& out_ids) {
    lsxcommon::ArrowTokenizerDb* db = TokenizerFor(model_path);
    if (db == nullptr) return false;
    return lsxcommon::FrameConversationForModel(std::string(model_path), *db,
                                                conv, out_ids);
}

bool LsxPromptFramer::ParseToolCalls(std::string_view model_path,
                                     const std::vector<std::int32_t>& ids,
                                     std::vector<lsxcommon::ToolCall>& out) {
    lsxcommon::ArrowTokenizerDb* db = TokenizerFor(model_path);
    if (db == nullptr) return false;
    return lsxcommon::ParseToolCallsForModel(std::string(model_path), *db, ids,
                                             out);
}

}  // namespace taskmosaic
