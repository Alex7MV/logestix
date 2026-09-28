// GLM-5.3-Flash native tool framing (host-only; synthetic tokenizer, no CUDA,
// no real model). Verifies Glm5NextFlashModel::RenderGlmFlashConversation,
// ParseGlmFlashToolCallText and the tokenizer-backed conversation/parse
// bridges against the checked-in synthetic GLM-Flash tokenizer fixture.
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <lsxcommon/converter.h>
#include <lsxcommon/tokenizer.h>

#include "glm_moe_flash_model.h"

namespace fs = std::filesystem;

using lsxcommon::ArrowTokenizerDb;
using lsxcommon::ChatConversation;
using lsxcommon::ChatMessage;
using lsxcommon::Glm5NextFlashModel;
using lsxcommon::ToolCall;
using lsxcommon::ToolSpec;

namespace {

struct DbFiles {
    fs::path pinned;
    ArrowTokenizerDb db;
};

DbFiles BuildDb(const std::string &fixture_name)
{
    DbFiles out;
    out.pinned = fs::temp_directory_path() / ("lsx_glm_flash_" + fixture_name);
    fs::remove_all(out.pinned);
    fs::create_directories(out.pinned);
    REQUIRE(lsxcommon::BuildTokenizerArrowFiles(
                TOKENIZER_TEST_DATA_DIR "/" + fixture_name, out.pinned)
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

bool Contains(const std::string &hay, std::string_view needle)
{
    return hay.find(needle) != std::string::npos;
}

ChatConversation ToolConversation()
{
    ChatConversation conv;
    conv.messages.push_back({"system", "You are a helper.", "", {}});
    conv.messages.push_back({"user", "Weather?", "", {}});
    ToolSpec t;
    t.name = "get_weather";
    t.description = "Get the weather";
    t.parameters_json = R"({"type":"object"})";
    conv.tools.push_back(t);
    return conv;
}

} // namespace

TEST_CASE("glm-flash tool framing: render places tools, calls and responses",
          "[glm-flash][tool_framing]")
{
    ChatConversation conv = ToolConversation();
    ChatMessage asst;
    asst.role = "assistant";
    asst.content = "Sure.";
    ToolCall call;
    call.name = "get_weather";
    call.arguments_json = R"({"city":"Paris","days":3})";
    asst.tool_calls.push_back(call);
    conv.messages.push_back(asst);
    conv.messages.push_back({"tool", "sunny", "call_0", {}});

    std::string out;
    REQUIRE(Glm5NextFlashModel::RenderGlmFlashConversation(conv, out));

    REQUIRE(out.rfind("[gMASK]<sop><|system|>Reasoning Effort: Max", 0) == 0);
    CHECK(Contains(out, "<|system|>\n# Tools\n"));
    CHECK(Contains(out,
                   "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key>"));
    CHECK(Contains(out, "<|user|>Weather?\n"));
    // Native call: strings raw, numbers as JSON.
    CHECK(Contains(out,
                   "<|assistant|>\n<think></think>Sure.<tool_call>get_weather"
                   "<arg_key>city</arg_key><arg_value>Paris</arg_value>"
                   "<arg_key>days</arg_key><arg_value>3</arg_value>"
                   "</tool_call>\n"));
    CHECK(Contains(out, "<|observation|><tool_response>sunny</tool_response>"));
    const std::string gen = "<|assistant|><think>";
    REQUIRE(out.size() >= gen.size());
    CHECK(out.substr(out.size() - gen.size()) == gen);
}

TEST_CASE("glm-flash tool framing: render rejects unknown roles",
          "[glm-flash][tool_framing]")
{
    ChatConversation conv;
    conv.messages.push_back({"developer", "nope", "", {}});
    std::string out;
    CHECK_FALSE(Glm5NextFlashModel::RenderGlmFlashConversation(conv, out));
}

TEST_CASE("glm-flash tool framing: parse native calls", "[glm-flash][tool_framing]")
{
    const std::string text =
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
        "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>"
        "<tool_call>ping<arg_key>host</arg_key><arg_value>\"a.b\"</arg_value>"
        "</tool_call>";
    const std::vector<ToolCall> calls =
        Glm5NextFlashModel::ParseGlmFlashToolCallText(text);
    REQUIRE(calls.size() == 2);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json == R"({"city":"Paris","days":3})");
    CHECK(calls[0].id == "call_0");
    CHECK(calls[1].name == "ping");
    CHECK(calls[1].arguments_json == R"({"host":"a.b"})");
    CHECK(calls[1].id == "call_1");
}

TEST_CASE("glm-flash tool framing: conversation bridge injects role sentinels",
          "[glm-flash][tool_framing]")
{
    DbFiles f = BuildDb("synthetic_glm_flash_tok.json");
    const auto &db = f.db;
    const int32_t gmask = lsxcommon::arrow_tokenizer_lookup(db, "[gMASK]", 7);
    const int32_t user = lsxcommon::arrow_tokenizer_lookup(db, "<|user|>", 8);
    const int32_t asst =
        lsxcommon::arrow_tokenizer_lookup(db, "<|assistant|>", 13);
    REQUIRE(gmask >= 0);
    REQUIRE(user >= 0);
    REQUIRE(asst >= 0);

    lsxcommon::ModelInitConfig cfg;
    Glm5NextFlashModel model(cfg);

    std::vector<int32_t> ids;
    REQUIRE(model.BuildConversationInputIds(db, ToolConversation(), ids));
    REQUIRE(!ids.empty());
    CHECK(ids.front() == gmask);
    // The generation prompt emits the assistant sentinel exactly once.
    CHECK(std::count(ids.begin(), ids.end(), asst) == 1);
    CHECK(std::count(ids.begin(), ids.end(), user) >= 1);

    f.db = ArrowTokenizerDb{};
    fs::remove_all(f.pinned);
}

TEST_CASE("glm-flash tool framing: ParseToolCalls round-trips generated ids",
          "[glm-flash][tool_framing]")
{
    DbFiles f = BuildDb("synthetic_glm_flash_tok.json");
    const auto &db = f.db;

    lsxcommon::ModelInitConfig cfg;
    Glm5NextFlashModel model(cfg);

    const std::string answer =
        "<tool_call>get_weather<arg_key>city</arg_key>"
        "<arg_value>Paris</arg_value><arg_key>days</arg_key>"
        "<arg_value>3</arg_value></tool_call>";
    std::vector<int32_t> gen = RawEncode(db, answer);

    std::vector<ToolCall> calls;
    REQUIRE(model.ParseToolCalls(db, gen, calls));
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json == R"({"city":"Paris","days":3})");

    f.db = ArrowTokenizerDb{};
    fs::remove_all(f.pinned);
}
