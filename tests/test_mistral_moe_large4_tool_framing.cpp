// Mistral-Large-4 native tool framing + token-bridge round-trip (host-only,
// Model Ingestion Protocol step 4b). The frame is the checkpoint's own
// chat_template.jinja: [INST]/[TOOL_CALLS]name[ARGS]{json} with an ALWAYS
// emitted [MODEL_SETTINGS]{"reasoning_effort": "high"} block before the first
// user turn. The id-injection path runs on a synthetic tokenizer fixture
// (tests/data/synthetic_mistral_large4_tok.json) whose special ids match the
// frozen passport.
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/mistral_moe_tool_format.h"

#include <arrow/io/file.h>

#include <cstdlib>
#include <filesystem>
#include <algorithm>
#include <string>
#include <vector>

#include <lsxcommon/converter.h>
#include <lsxcommon/tokenizer.h>

#ifndef LSX_LARGE4_TOK_FIXTURE
#define LSX_LARGE4_TOK_FIXTURE "tests/data/synthetic_mistral_large4_tok.json"
#endif

using lsxcommon::ChatConversation;
using lsxcommon::ChatMessage;
using lsxcommon::ToolCall;
using lsxcommon::ToolSpec;
namespace tf = lsxcommon::mistral_moe::tool_format;

TEST_CASE("Mistral-Large-4 RenderText emits the reasoning_effort block",
          "[mistral_large4][tool]")
{
    ChatConversation conv;
    conv.messages.push_back({"system", "You are helpful.", "", {}});
    conv.messages.push_back({"user", "Weather?", "", {}});

    tf::RenderOptions opts;
    opts.format = tf::MistralFormat::kLarge4;
    opts.reasoning_effort = "high";
    std::string text;
    REQUIRE(tf::RenderText(conv, text, true, opts));
    // The block precedes the first user turn, even without tools.
    const auto settings = text.find(
        "[MODEL_SETTINGS]{\"reasoning_effort\": \"high\"}[/MODEL_SETTINGS]");
    REQUIRE(settings != std::string::npos);
    REQUIRE(settings < text.find("[INST]Weather?[/INST]"));
    CHECK(text.rfind("<s>[SYSTEM_PROMPT]You are helpful.[/SYSTEM_PROMPT]",
                     0) == 0);

    opts.reasoning_effort = "low";
    REQUIRE(tf::RenderText(conv, text, true, opts));
    CHECK(text.find("\"reasoning_effort\": \"low\"") != std::string::npos);

    // The Small-4 default is unchanged (no block without tools).
    tf::RenderOptions small;
    REQUIRE(tf::RenderText(conv, text, true, small));
    CHECK(text.find("[MODEL_SETTINGS]") == std::string::npos);
}

TEST_CASE("Mistral-Large-4 token bridge injects markers and parses tool calls",
          "[mistral_large4][tool]")
{
    const std::filesystem::path tok_path = LSX_LARGE4_TOK_FIXTURE;
    if (!std::filesystem::exists(tok_path)) SKIP("synthetic tokenizer absent");
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "lsx_large4_tool_tok";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);

    lsxcommon::TokenizerSpecialPolicy policy;
    policy.pretokenizer = 10;
    policy.bos_id = 1;
    policy.eos_id = 2;
    policy.pad_id = 11;
    auto st = lsxcommon::BuildTokenizerArrowFiles(tok_path.string(), dir,
                                                  policy);
    INFO(st.ToString());
    REQUIRE(st.ok());

    auto vf = arrow::io::MemoryMappedFile::Open(
        (dir / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    auto mf = arrow::io::MemoryMappedFile::Open(
        (dir / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    REQUIRE(vf.ok());
    REQUIRE(mf.ok());
    lsxcommon::ArrowTokenizerDb db;
    auto meta = lsxcommon::load_tokenizer_metadata(*vf, *mf, db);
    INFO(meta.ToString());
    REQUIRE(meta.ok());
    REQUIRE(db.tokenizer_kind == 12);
    REQUIRE(tf::SentinelsPresent(db));

    ChatConversation conv;
    ToolSpec tool;
    tool.name = "get_weather";
    tool.description = "Get the weather";
    tool.parameters_json = "{\"type\":\"object\"}";
    conv.tools.push_back(tool);
    conv.messages.push_back({"user", "Weather?", "", {}});

    tf::RenderOptions opts;
    opts.format = tf::MistralFormat::kLarge4;
    opts.reasoning_effort = "medium";
    std::vector<int32_t> ids;
    REQUIRE(tf::Render(db, conv, ids, true, opts));
    REQUIRE(!ids.empty());
    CHECK(ids[0] == 1);  // <s>
    CHECK(ids[1] == 5);  // [AVAILABLE_TOOLS]
    CHECK(ids[2] != 5);
    CHECK(std::find(ids.begin(), ids.end(), 3) != ids.end());  // [INST]
    CHECK(std::find(ids.begin(), ids.end(), 36) != ids.end());    // settings
    CHECK(std::find(ids.begin(), ids.end(), 37) != ids.end());

    // Parse a native tool call from generated tokens: render the assistant
    // turn, then parse it back (never the full prompt: its instructions embed
    // an example call).
    ChatConversation call;
    ChatMessage a;
    a.role = "assistant";
    a.tool_calls.push_back({"", "get_weather", "{\"city\":\"Paris\"}"});
    call.messages.push_back(a);
    std::vector<int32_t> call_ids;
    REQUIRE(tf::Render(db, call, call_ids, false, opts));
    std::vector<ToolCall> parsed;
    REQUIRE(tf::Parse(db, call_ids, parsed));
    REQUIRE(parsed.size() == 1);
    CHECK(parsed[0].name == "get_weather");
    CHECK(parsed[0].arguments_json == "{\"city\":\"Paris\"}");
}
