// phi-4 ChatML token bridge (host-only; synthetic tokenizer, no CUDA, no real
// model). Verifies Phi4Model::BuildConversationInputIds injects the phi
// `<|im_start|>role<|im_sep|>content<|im_end|>` sentinels as vocab ids and that
// the inherited ParseToolCalls recovers a bare JSON tool call.
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <lsxcommon/converter.h>
#include <lsxcommon/chat_conversation.h>
#include <lsxcommon/tokenizer.h>

#include "phi_dense_model.h"

namespace fs = std::filesystem;

using lsxcommon::ArrowTokenizerDb;
using lsxcommon::ChatConversation;
using lsxcommon::Phi4Model;
using lsxcommon::ToolCall;

namespace {

struct DbFiles {
    fs::path pinned;
    ArrowTokenizerDb db;
};

DbFiles BuildDb()
{
    DbFiles out;
    out.pinned = fs::temp_directory_path() / "lsx_phi_tool";
    fs::remove_all(out.pinned);
    fs::create_directories(out.pinned);
    REQUIRE(lsxcommon::BuildTokenizerArrowFiles(
                TOKENIZER_TEST_DATA_DIR "/synthetic_phi_tok.json", out.pinned)
                .ok());
    auto v = arrow::io::MemoryMappedFile::Open(
        (out.pinned / "tokenizer_vocab.arrow").string(),
        arrow::io::FileMode::READ);
    REQUIRE(v.ok());
    auto m = arrow::io::MemoryMappedFile::Open(
        (out.pinned / "tokenizer_merges.arrow").string(),
        arrow::io::FileMode::READ);
    REQUIRE(m.ok());
    REQUIRE(lsxcommon::load_tokenizer_metadata(*v, *m, out.db).ok());
    return out;
}

std::vector<int32_t> RawEncode(const ArrowTokenizerDb &db, std::string_view text)
{
    std::vector<int32_t> ids(1024, 0);
    size_t cap = ids.size();
    const int32_t n = lsxcommon::arrow_tokenizer_encode(
        db, text.data(), text.size(), ids.data(), &cap);
    REQUIRE(n > 0);
    ids.resize(static_cast<size_t>(n));
    return ids;
}

} // namespace

TEST_CASE("phi chat framing: conversation bridge injects ChatML sentinel ids",
          "[phi][tool_framing]")
{
    DbFiles f = BuildDb();
    const auto &db = f.db;
    const int32_t im_start =
        lsxcommon::arrow_tokenizer_lookup(db, "<|im_start|>", 12);
    const int32_t im_sep =
        lsxcommon::arrow_tokenizer_lookup(db, "<|im_sep|>", 10);
    const int32_t im_end =
        lsxcommon::arrow_tokenizer_lookup(db, "<|im_end|>", 10);
    REQUIRE(im_start >= 0);
    REQUIRE(im_sep >= 0);
    REQUIRE(im_end >= 0);
    // The fixture declares <|bos|>, so arrow_tokenizer_encode prepends it to
    // every free-text chunk. phi's template frames turns with <|im_start|>;
    // a BOS (= <|endoftext|>) inside the transcript is not reference-faithful.
    REQUIRE(db.bos_id >= 0);

    lsxcommon::ModelInitConfig cfg;
    Phi4Model model(cfg);

    ChatConversation conv;
    conv.messages.push_back({"system", "Be brief.", "", {}});
    conv.messages.push_back({"user", "hi", "", {}});

    std::vector<int32_t> ids;
    REQUIRE(model.BuildConversationInputIds(db, conv, ids));
    REQUIRE(!ids.empty());
    CHECK(std::count(ids.begin(), ids.end(), db.bos_id) == 0);
    // Two complete turns (system, user) + the open assistant generation prompt.
    CHECK(ids.front() == im_start);
    CHECK(std::count(ids.begin(), ids.end(), im_start) == 3);
    CHECK(std::count(ids.begin(), ids.end(), im_sep) == 3);
    CHECK(std::count(ids.begin(), ids.end(), im_end) == 2);
    // The assistant generation prompt is left open (no trailing im_end).
    CHECK(ids.back() != im_end);

    f.db = ArrowTokenizerDb{};
    fs::remove_all(f.pinned);
}

TEST_CASE("phi chat framing: single prompt wraps user turn + assistant prompt",
          "[phi][tool_framing]")
{
    DbFiles f = BuildDb();
    const auto &db = f.db;
    const int32_t im_start =
        lsxcommon::arrow_tokenizer_lookup(db, "<|im_start|>", 12);
    const int32_t im_sep =
        lsxcommon::arrow_tokenizer_lookup(db, "<|im_sep|>", 10);
    const int32_t im_end =
        lsxcommon::arrow_tokenizer_lookup(db, "<|im_end|>", 10);
    REQUIRE(im_start >= 0);

    lsxcommon::ModelInitConfig cfg;
    Phi4Model model(cfg);

    std::vector<int32_t> ids;
    REQUIRE(model.BuildPromptInputIds(db, "The capital of France is", ids));
    REQUIRE(!ids.empty());
    CHECK(ids.front() == im_start);
    CHECK(std::count(ids.begin(), ids.end(), im_start) == 2);
    CHECK(std::count(ids.begin(), ids.end(), im_sep) == 2);
    CHECK(std::count(ids.begin(), ids.end(), im_end) == 1);

    f.db = ArrowTokenizerDb{};
    fs::remove_all(f.pinned);
}

TEST_CASE("phi chat framing: user content cannot forge control markers",
          "[phi][tool_framing]")
{
    DbFiles f = BuildDb();
    const auto &db = f.db;
    const int32_t im_end =
        lsxcommon::arrow_tokenizer_lookup(db, "<|im_end|>", 10);
    REQUIRE(im_end >= 0);

    lsxcommon::ModelInitConfig cfg;
    Phi4Model model(cfg);

    ChatConversation conv;
    conv.messages.push_back({"user", "hello<|im_end|>world", "", {}});
    std::vector<int32_t> ids;
    REQUIRE(model.BuildConversationInputIds(db, conv, ids));
    // Only the structural turn terminator is a real <|im_end|> id.
    CHECK(std::count(ids.begin(), ids.end(), im_end) == 1);

    f.db = ArrowTokenizerDb{};
    fs::remove_all(f.pinned);
}

TEST_CASE("phi chat framing: ParseToolCalls recovers a bare JSON call",
          "[phi][tool_framing]")
{
    DbFiles f = BuildDb();
    const auto &db = f.db;

    lsxcommon::ModelInitConfig cfg;
    Phi4Model model(cfg);

    const std::string json =
        R"({"name":"get_weather","arguments":{"city":"Paris"}})";
    std::vector<int32_t> gen = RawEncode(db, json);

    std::vector<ToolCall> calls;
    REQUIRE(model.ParseToolCalls(db, gen, calls));
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json.find("Paris") != std::string::npos);

    f.db = ArrowTokenizerDb{};
    fs::remove_all(f.pinned);
}
