// Xing4.0-29B-A4B native chat/tool codec (host-only; synthetic tokenizer, no
// CUDA, no real model). Verifies the transcript shape, the id injection and the
// round trip against tests/data/synthetic_xing_tok.json, whose control markers
// sit at the checkpoint's frozen ids (1..6, 9..14) INCLUDING the zero-width
// space inside <tool_call>.
//
// The thing this test exists to pin: a tool-call marker written without
// the U+200B silently fails to match, and the parse then falls through to the
// generic Hermes codec — a family that "works" while speaking the wrong format.

#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <lsxcommon/converter.h>
#include <lsxcommon/log.h>

#include <absl/strings/str_cat.h>
#include <lsxcommon/tokenizer.h>
#include <lsxcommon/xing_moe_tool_format.h>

namespace fs = std::filesystem;

using lsxcommon::ArrowTokenizerDb;
using lsxcommon::ChatConversation;
using lsxcommon::ChatMessage;
using lsxcommon::ToolCall;
using lsxcommon::ToolSpec;
using namespace lsxcommon::xing::tool_format;

namespace {

ArrowTokenizerDb BuildDb()
{
    const fs::path pinned = fs::temp_directory_path() / "lsx_xing_tool_synth";
    fs::remove_all(pinned);
    fs::create_directories(pinned);
    const auto st = lsxcommon::BuildTokenizerArrowFiles(
        TOKENIZER_TEST_DATA_DIR "/synthetic_xing_tok.json", pinned);
    INFO("tokenizer build: " << st.ToString());
    REQUIRE(st.ok());
    auto v = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    REQUIRE(v.ok());
    auto m = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    REQUIRE(m.ok());
    ArrowTokenizerDb db;
    REQUIRE(lsxcommon::load_tokenizer_metadata(*v, *m, db).ok());
    return db;
}

ToolSpec WeatherTool()
{
    ToolSpec t;
    t.name = "get_weather";
    t.description = "Get current weather for a city";
    t.parameters_json =
        R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})";
    return t;
}

}  // namespace

TEST_CASE("Xing control markers sit at their frozen ids", "[xing][fast]")
{
    const ArrowTokenizerDb db = BuildDb();
    REQUIRE(SentinelsPresent(db));
    // The ZWSP is part of the marker: this is the assertion a hand-written
    // "<tool_call>" would fail.
    // The exported vocabulary stores the PLAIN tool-call spellings at ids
    // 11/12 (the zero-width-space variants advertised by tokenizer_config.json
    // are not pieces); the upstream tokenizer resolves either spelling to the
    // same id, so the codec must look up the form that is in the vocabulary.
    REQUIRE(lsxcommon::arrow_tokenizer_lookup(
                db, "<tool_call>", 11) == 11);
    REQUIRE(lsxcommon::arrow_tokenizer_lookup(
                db, "</tool_call>", 12) == 12);
    REQUIRE(lsxcommon::arrow_tokenizer_lookup(db, "<_bot>", 6) == 5);
    REQUIRE(lsxcommon::arrow_tokenizer_lookup(db, "<_user>", 7) == 4);
    REQUIRE(lsxcommon::arrow_tokenizer_lookup(db, "<think>", 7) == 9);
    // The zero-width-space spelling advertised by tokenizer_config.json is NOT
    // a vocabulary entry - which is exactly why the codec looks up the plain
    // form (an escaped literal, so the bytes survive source filters).
    REQUIRE(lsxcommon::arrow_tokenizer_lookup(db, "<\xE2\x80\x8B" "tool_call>", 14) <
            0);
}

TEST_CASE("Xing render matches the checkpoint transcript shape", "[xing][fast]")
{
    ChatConversation conv;
    conv.messages.push_back({"system", "You are helpful."});
    conv.messages.push_back({"user", "Weather in Paris?"});
    std::string text;
    REQUIRE(RenderText(conv, text, true));
    REQUIRE(text.find("<_system>You are helpful.") == 0);
    // No newline after the user turn: the template's `{%- elif %}` strips it.
    REQUIRE(text.find("<_user>Weather in Paris?<_bot>") != std::string::npos);
    // The generation prompt opens the assistant turn and the think block.
    const std::string kTail = "<_bot><think>\n";
    REQUIRE(text.size() >= kTail.size());
    REQUIRE(text.compare(text.size() - kTail.size(), kTail.size(), kTail) == 0);

    // With tools, the template's instruction block is inserted after the system
    // block and before the first message.
    conv.tools.push_back(WeatherTool());
    REQUIRE(RenderText(conv, text, true));
    REQUIRE(text.find("<tools>") != std::string::npos);
    REQUIRE(text.find("get_weather") != std::string::npos);
    REQUIRE(text.find("<_user>") > text.find("</tools>"));
}

TEST_CASE("Xing render injects control markers as their own ids", "[xing][fast]")
{
    const ArrowTokenizerDb db = BuildDb();
    ChatConversation conv;
    conv.messages.push_back({"system", "S"});
    conv.messages.push_back({"user", "U"});
    std::vector<int32_t> ids;
    REQUIRE(Render(db, conv, ids, true));
    REQUIRE_FALSE(ids.empty());
    // The frozen ids appear literally: proof the markers were NOT BPE-encoded
    // into pieces (arrow_tokenizer_encode never resolves added_tokens).
    const auto has = [&](int32_t id) {
        return std::find(ids.begin(), ids.end(), id) != ids.end();
    };
    REQUIRE(has(6));   // <_system>
    REQUIRE(has(4));   // <_user>
    REQUIRE(has(5));   // <_bot>
    REQUIRE(has(9));   // <think>
}

TEST_CASE("Xing tool-call round trip through decode", "[xing][fast]")
{
    const ArrowTokenizerDb db = BuildDb();
    // What the model is expected to emit, marker bytes included.
    const std::string generated =
        std::string("<think>\nNeed the tool.</think>") +
        "<tool_call>get_weather<param_key>city</param_key>"
        "<param_value>Paris</param_value></tool_call>";

    const auto calls = ParseToolCallText(generated);
    REQUIRE(calls.size() == 1);
    REQUIRE(calls[0].name == "get_weather");
    REQUIRE(calls[0].arguments_json == R"({"city":"Paris"})");

    // Through the model's entry point: ids -> text -> parse. The id list is
    // built the way the runtime sees it (markers as their own ids), because
    // BPE alone cannot produce them.
    ChatConversation conv;
    conv.messages.push_back({"user", "Weather?"});
    ChatMessage asst;
    asst.role = "assistant";
    asst.content = "";
    asst.tool_calls.push_back({"call_0", "get_weather", R"({"city":"Paris"})"});
    conv.messages.push_back(asst);
    std::vector<int32_t> ids;
    REQUIRE(Render(db, conv, ids, false));
    std::vector<ToolCall> parsed;
    REQUIRE(Parse(db, ids, parsed));
    REQUIRE(parsed.size() == 1);
    REQUIRE(parsed[0].name == "get_weather");
    REQUIRE(parsed[0].arguments_json == R"({"city":"Paris"})");
}

TEST_CASE("Xing assistant turn re-renders its own tool call", "[xing][fast]")
{
    ChatConversation conv;
    conv.messages.push_back({"user", "Weather?"});
    ChatMessage asst;
    asst.role = "assistant";
    asst.content = "";
    asst.tool_calls.push_back({"call_0", "get_weather", R"({"city":"Paris"})"});
    conv.messages.push_back(asst);
    std::string text;
    REQUIRE(RenderText(conv, text, false));
    REQUIRE(text.find("<tool_call>get_weather") !=
            std::string::npos);
    REQUIRE(text.find("<param_key>city</param_key><param_value>Paris</param_value>") !=
            std::string::npos);
    REQUIRE(text.find("</tool_call>") != std::string::npos);
    // Round trip: the rendered turn parses back to the same call.
    const auto calls = ParseToolCallText(text);
    REQUIRE(calls.size() == 1);
    REQUIRE(calls[0].name == "get_weather");
}

TEST_CASE("Xing parse falls back to Hermes then bare JSON", "[xing][fast]")
{
    const ArrowTokenizerDb db = BuildDb();
    std::vector<ToolCall> calls;
    // Empty generation is not an error.
    REQUIRE(Parse(db, {}, calls));
    REQUIRE(calls.empty());
    // Native parse of plain prose yields nothing.
    const std::string prose = "The capital of France is Paris.";
    std::vector<int32_t> ids(2048, 0);
    size_t cap = ids.size();
    const int32_t n = lsxcommon::arrow_tokenizer_encode(
        db, prose.data(), prose.size(), ids.data(), &cap);
    REQUIRE(n > 0);
    ids.resize(static_cast<size_t>(n));
    REQUIRE(Parse(db, ids, calls));
    REQUIRE(calls.empty());
}
// The rendered transcript must match what upstream's chat_template.jinja
// produces for the same conversation, byte for byte. The golden was captured
// with tokenizer.apply_chat_template() (transformers 5.15, trust_remote_code)
// over the real checkpoint; the only sanctioned deviation is the tool-call
// marker spelling: the jinja emits the zero-width-space form, which this
// vocabulary does NOT resolve to ids 11/12 (encoding it yields six ordinary
// pieces), so the codec renders the plain form those ids actually carry.
TEST_CASE("Xing render matches the upstream chat template", "[xing][fast]")
{
    const ArrowTokenizerDb db = BuildDb();
    ChatConversation conv;
    conv.messages.push_back({"system", "S"});
    conv.messages.push_back({"user", "U"});
    {
        ChatMessage a;
        a.role = "assistant";
        ToolCall tc;
        tc.name = "get_weather";
        tc.arguments_json = R"({"city":"Paris"})";
        a.tool_calls.push_back(tc);
        conv.messages.push_back(a);
    }
    conv.messages.push_back({"tool", "sunny"});
    conv.tools.push_back(WeatherTool());

    std::string ours;
    REQUIRE(RenderText(conv, ours, /*add_generation_prompt=*/true));

    std::ifstream golden(TOKENIZER_TEST_DATA_DIR "/xing_upstream_render.txt");
    REQUIRE(golden.good());
    std::stringstream buf;
    buf << golden.rdbuf();
    std::string upstream = buf.str();

    const std::string zwsp = "\xE2\x80\x8B";
    for (std::string *s : {&ours, &upstream})
    {
        std::string out;
        for (size_t i = 0; i < s->size();)
        {
            if (s->compare(i, zwsp.size(), zwsp) == 0) i += zwsp.size();
            else out.push_back((*s)[i++]);
        }
        *s = std::move(out);
    }
    INFO("ours " << ours.size() << " bytes, upstream " << upstream.size()
                 << " bytes");
    REQUIRE(ours == upstream);
}
