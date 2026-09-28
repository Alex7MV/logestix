// DeepSeek-V4 / V4.1 native DSML tool framing (host-only; synthetic tokenizer,
// no CUDA, no real model). Verifies the shared deepseek_moe_tool_format codec
// and both model overrides against the checked-in synthetic DeepSeek fixture.
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <lsxcommon/converter.h>
#include <lsxcommon/deepseek_moe_tool_format.h>
#include <lsxcommon/tokenizer.h>

#include "deepseek_moe_model.h"
#include "deepseek_moe_v41_model.h"

namespace fs = std::filesystem;

using lsxcommon::ArrowTokenizerDb;
using lsxcommon::ChatConversation;
using lsxcommon::ChatMessage;
using lsxcommon::DeepSeekV41FlashModel;
using lsxcommon::DeepSeekV4FlashModel;
using lsxcommon::DsmlTags;
using lsxcommon::ToolCall;
using lsxcommon::ToolSpec;

namespace {

constexpr DsmlTags kV4{"tool_calls", "invoke", "parameter"};
constexpr DsmlTags kV41{" calls", " invoke", " parameter"};

struct DbFiles {
    fs::path pinned;
    ArrowTokenizerDb db;
};

DbFiles BuildDb(const std::string &fixture_name)
{
    DbFiles out;
    out.pinned = fs::temp_directory_path() / ("lsx_deepseek_" + fixture_name);
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

TEST_CASE("deepseek tool framing: v4.1 render uses leading-space DSML tags",
          "[deepseek][tool_framing]")
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
    REQUIRE(lsxcommon::RenderDsmlConversation(conv, kV41, out));

    REQUIRE(out.rfind("<｜begin▁of▁sentence｜><｜System｜>You are a helper.", 0) ==
            0);
    CHECK(Contains(out, "<｜DSML｜ calls>"));
    CHECK(Contains(out, "<｜DSML｜ invoke name=\"get_weather\">"));
    CHECK(Contains(out,
                   "<｜DSML｜ parameter name=\"city\" string=\"true\">Paris"
                   "</｜DSML｜ parameter>"));
    CHECK(Contains(out,
                   "<｜DSML｜ parameter name=\"days\" string=\"false\">3"
                   "</｜DSML｜ parameter>"));
    CHECK(Contains(out, "<｜User｜>Weather?"));
    CHECK(Contains(out, "<｜Assistant｜></think>Sure."));
    CHECK(Contains(out, "<｜end▁of▁sentence｜>"));
    CHECK(Contains(out, "<tool_result>sunny</tool_result>"));

    // V4 uses the same shape with unspaced tag names.
    std::string v4;
    REQUIRE(lsxcommon::RenderDsmlConversation(conv, kV4, v4));
    CHECK(Contains(v4, "<｜DSML｜tool_calls>"));
    CHECK(Contains(v4, "<｜DSML｜invoke name=\"get_weather\">"));
    CHECK(Contains(v4, "<｜DSML｜parameter name=\"city\" string=\"true\">Paris"));
}

TEST_CASE("deepseek tool framing: parse extracts both DSML dialects",
          "[deepseek][tool_framing]")
{
    const std::string v41 =
        "<｜Assistant｜></think>ok\n\n<｜DSML｜ calls>\n"
        "<｜DSML｜ invoke name=\"get_weather\">\n"
        "<｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n"
        "<｜DSML｜ parameter name=\"days\" string=\"false\">3</｜DSML｜ parameter>\n"
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    const std::vector<ToolCall> c41 =
        lsxcommon::ParseDsmlToolCalls(v41, kV41);
    REQUIRE(c41.size() == 1);
    CHECK(c41[0].name == "get_weather");
    CHECK(c41[0].arguments_json == R"({"city":"Paris","days":3})");
    CHECK(c41[0].id == "call_0");

    const std::string v4 =
        "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"ping\">\n"
        "<｜DSML｜parameter name=\"host\" string=\"true\">a.b</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    const std::vector<ToolCall> c4 = lsxcommon::ParseDsmlToolCalls(v4, kV4);
    REQUIRE(c4.size() == 1);
    CHECK(c4[0].name == "ping");
    CHECK(c4[0].arguments_json == R"({"host":"a.b"})");
}

TEST_CASE("deepseek tool framing: string flag overrides JSON-looking values",
          "[deepseek][tool_framing]")
{
    const std::string text =
        "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n"
        "<｜DSML｜ parameter name=\"zip\" string=\"true\">01234</｜DSML｜ parameter>\n"
        "<｜DSML｜ parameter name=\"count\" string=\"false\">3</｜DSML｜ parameter>\n"
        "<｜DSML｜ parameter name=\"flag\" string=\"true\">true</｜DSML｜ parameter>\n"
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    const std::vector<ToolCall> calls = lsxcommon::ParseDsmlToolCalls(text, kV41);
    REQUIRE(calls.size() == 1);
    // string="true" must stay a string even when it looks like a number/bool.
    CHECK(calls[0].arguments_json ==
          R"({"zip":"01234","count":3,"flag":"true"})");
}

TEST_CASE("deepseek tool framing: model overrides round-trip through the tokenizer",
          "[deepseek][tool_framing]")
{
    DbFiles f = BuildDb("synthetic_deepseek_tok.json");
    const auto &db = f.db;
    const int32_t bos = lsxcommon::arrow_tokenizer_lookup(
        db, "<｜begin▁of▁sentence｜>", std::strlen("<｜begin▁of▁sentence｜>"));
    REQUIRE(bos >= 0);

    lsxcommon::ModelInitConfig cfg;

    // V4.1 override: render injects the bos/role sentinels.
    {
        DeepSeekV41FlashModel model(cfg);
        std::vector<int32_t> ids;
        REQUIRE(model.BuildConversationInputIds(db, ToolConversation(), ids));
        REQUIRE(!ids.empty());
        CHECK(ids.front() == bos);
    }

    // V4 override: parse a rendered DSML call back to a ToolCall.
    {
        DeepSeekV4FlashModel model(cfg);
        const std::string answer =
            "\n\n<｜DSML｜tool_calls>\n"
            "<｜DSML｜invoke name=\"get_weather\">\n"
            "<｜DSML｜parameter name=\"city\" string=\"true\">Paris"
            "</｜DSML｜parameter>\n"
            "<｜DSML｜parameter name=\"days\" string=\"false\">3"
            "</｜DSML｜parameter>\n"
            "</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        std::vector<int32_t> gen;
        REQUIRE(lsxcommon::AppendDeepSeekIds(db, answer, gen));
        std::vector<ToolCall> calls;
        REQUIRE(model.ParseToolCalls(db, gen, calls));
        REQUIRE(calls.size() == 1);
        CHECK(calls[0].name == "get_weather");
        CHECK(calls[0].arguments_json == R"({"city":"Paris","days":3})");
    }

    f.db = ArrowTokenizerDb{};
    fs::remove_all(f.pinned);
}
