// Hy-Memory-30B-A3B native chat/tool framing (host-only; synthetic tokenizer,
// no real model). Pins the checkpoint's chat_template.jinja frame — the
// `<|User|>`/`<|Assistant|>` turn markers, the `no_think` generation prefix and
// the `<tool_calls>` / `<tool_sep>` / `<arg_key>` XML tool codec.
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/converter.h>
#include <lsxcommon/hy_moe_tool_format.h>
#include <lsxcommon/tokenizer.h>

namespace fs = std::filesystem;

using lsxcommon::ArrowTokenizerDb;
using lsxcommon::ChatConversation;
using lsxcommon::ChatMessage;
using lsxcommon::ToolCall;
using lsxcommon::ToolSpec;

namespace {

struct DbFiles {
    fs::path pinned;
    ArrowTokenizerDb db;
};

DbFiles BuildDb()
{
    DbFiles out;
    out.pinned = fs::temp_directory_path() / "lsx_hy_tool";
    fs::remove_all(out.pinned);
    fs::create_directories(out.pinned);
    auto st = lsxcommon::BuildTokenizerArrowFiles(
        TOKENIZER_TEST_DATA_DIR "/synthetic_hy_tok.json", out.pinned);
    if (!st.ok()) WARN(st.ToString());
    REQUIRE(st.ok());
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

int32_t IdOf(const ArrowTokenizerDb &db, const char *text)
{
    return lsxcommon::arrow_tokenizer_lookup(db, text, std::strlen(text));
}

std::string Decode(const ArrowTokenizerDb &db, const std::vector<int32_t> &ids)
{
    std::vector<char> buf(ids.size() * 16 + 64);
    size_t cap = buf.size();
    REQUIRE(lsxcommon::arrow_tokenizer_decode(db, ids.data(), ids.size(),
                                              buf.data(), &cap) == 0);
    return std::string(buf.data(), cap);
}

} // namespace

TEST_CASE("Hy chat template markers are present in the tokenizer", "[hy][tool]")
{
    auto f = BuildDb();
    for (const char *m : {"<｜hy_begin▁of▁sentence｜>", "<｜hy_User｜>",
                          "<｜hy_Assistant｜>", "<think>", "</think>",
                          "<｜reasoning_mode｜>", "<tool_calls>",
                          "</tool_calls>", "<tool_call>", "</tool_call>",
                          "<tool_sep>", "<arg_key>", "</arg_key>",
                          "<arg_value>", "</arg_value>", "<tool_responses>",
                          "</tool_responses>", "<tool_response>",
                          "</tool_response>", "<eos:6124c78e>"})
        CHECK(IdOf(f.db, m) >= 0);
    CHECK(lsxcommon::hy::HyChatSentinelsPresent(f.db));
}

TEST_CASE("Hy conversation frames the no-think generation prefix", "[hy][tool]")
{
    auto f = BuildDb();
    ChatConversation conv;
    conv.messages.push_back({"system", "Be terse.", "", {}});
    conv.messages.push_back({"user", "Hi", "", {}});
    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::hy::RenderHyConversation(f.db, conv, ids));
    REQUIRE(ids.size() > 6);
    CHECK(ids[0] == IdOf(f.db, "<｜hy_begin▁of▁sentence｜>"));
    // System text, then the reasoning-mode switch, then the user turn.
    CHECK(Decode(f.db, ids).find("Be terse.<｜reasoning_mode｜>reasoning_effort:") !=
          std::string::npos);
    // The generation prompt is `<|Assistant|><think></think>`.
    CHECK(ids[ids.size() - 1] == IdOf(f.db, "</think>"));
    CHECK(ids[ids.size() - 2] == IdOf(f.db, "<think>"));
    CHECK(ids[ids.size() - 3] == IdOf(f.db, "<｜hy_Assistant｜>"));
    CHECK(std::count(ids.begin(), ids.end(), IdOf(f.db, "<｜hy_User｜>")) == 1);
}

TEST_CASE("Hy conversation renders the tools block and tool responses",
          "[hy][tool]")
{
    auto f = BuildDb();
    ChatConversation conv;
    conv.messages.push_back({"user", "Weather in Paris?", "", {}});
    ToolSpec t;
    t.name = "get_weather";
    t.description = "Get the weather";
    t.parameters_json = "{\"type\":\"object\"}";
    conv.tools.push_back(t);
    ChatMessage a;
    a.role = "assistant";
    a.content = "On it.";
    a.tool_calls.push_back({"", "get_weather", "{\"city\":\"Paris\"}"});
    conv.messages.push_back(a);
    conv.messages.push_back({"tool", "sunny", "call_0", {}});

    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::hy::RenderHyConversation(f.db, conv, ids));
    const std::string text = Decode(f.db, ids);
    // The tool schema is emitted in the template's Python-tojson shape.
    CHECK(text.find("{\"type\": \"function\", \"function\": {\"name\": "
                    "\"get_weather\", \"description\": \"Get the weather\", "
                    "\"parameters\": {\"type\": \"object\"}}}") !=
          std::string::npos);
    // The turn-closing <eos:6124c78e> is a SPECIAL token, so decode strips it;
    // the turn structure itself must be byte-exact.
    CHECK(text.find("<tool_calls>\n<tool_call>get_weather<tool_sep>\n"
                    "<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n"
                    "</tool_call>\n</tool_calls>") != std::string::npos);
    CHECK(std::count(ids.begin(), ids.end(), IdOf(f.db, "<eos:6124c78e>")) == 1);
    CHECK(text.find("<tool_responses>\n<tool_response>\nsunny\n</tool_response>"
                    "\n</tool_responses>") != std::string::npos);
    // One real call plus the template's own example block.
    CHECK(std::count(ids.begin(), ids.end(), IdOf(f.db, "<tool_call>")) >= 2);
    CHECK(ids.back() == IdOf(f.db, "</think>"));
}

TEST_CASE("Hy tool call render/parse round-trip", "[hy][tool]")
{
    ToolCall in;
    in.name = "get_weather";
    in.arguments_json = "{\"city\":\"Paris\",\"days\":2}";
    const std::string xml = lsxcommon::hy::RenderHyToolCallText(in);
    CHECK(xml.find("<tool_call>get_weather<tool_sep>\n") == 0);
    CHECK(xml.find("<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n") !=
          std::string::npos);
    const auto calls = lsxcommon::hy::ParseHyToolCallText(xml);
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json == "{\"city\":\"Paris\",\"days\":2}");
}

TEST_CASE("Hy parse extracts a native tool call from a token stream", "[hy][tool]")
{
    auto f = BuildDb();
    std::vector<int32_t> gen;
    auto push_text = [&](const std::string &t) {
        std::vector<int32_t> buf(512, 0);
        size_t cap = buf.size();
        const int32_t n = lsxcommon::arrow_tokenizer_encode(
            f.db, t.data(), t.size(), buf.data(), &cap);
        REQUIRE(n > 0);
        gen.insert(gen.end(), buf.begin(), buf.begin() + n);
    };
    push_text("Calling. ");
    gen.push_back(IdOf(f.db, "<tool_calls>"));
    push_text("\n");
    gen.push_back(IdOf(f.db, "<tool_call>"));
    push_text("get_weather");
    gen.push_back(IdOf(f.db, "<tool_sep>"));
    push_text("\n");
    gen.push_back(IdOf(f.db, "<arg_key>"));
    push_text("city");
    gen.push_back(IdOf(f.db, "</arg_key>"));
    push_text("\n");
    gen.push_back(IdOf(f.db, "<arg_value>"));
    push_text("Paris");
    gen.push_back(IdOf(f.db, "</arg_value>"));
    push_text("\n");
    gen.push_back(IdOf(f.db, "</tool_call>"));
    push_text("\n");
    gen.push_back(IdOf(f.db, "</tool_calls>"));

    std::vector<ToolCall> calls;
    REQUIRE(lsxcommon::hy::ParseHyToolCalls(f.db, gen, calls));
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json == "{\"city\":\"Paris\"}");
}

TEST_CASE("Hy parse falls back to the bare JSON object", "[hy][tool]")
{
    auto f = BuildDb();
    std::vector<int32_t> gen;
    std::vector<int32_t> buf(512, 0);
    size_t cap = buf.size();
    const std::string text =
        "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}";
    const int32_t n = lsxcommon::arrow_tokenizer_encode(
        f.db, text.data(), text.size(), buf.data(), &cap);
    REQUIRE(n > 0);
    gen.insert(gen.end(), buf.begin(), buf.begin() + n);
    std::vector<ToolCall> calls;
    REQUIRE(lsxcommon::hy::ParseHyToolCalls(f.db, gen, calls));
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "get_weather");
}
