// MiniMax-M3 native chat/tool framing (host-only; synthetic tokenizer, no
// real model). Pins the checkpoint's chat_template.jinja frame — the
// `]~!b[]~b]system` / developer / user / ai turn markers, the think-END
// generation prefix, and the `]<]minimax[>[<invoke name="f">` XML tool codec.
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>

#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/converter.h>
#include <lsxcommon/minimax_moe_tool_format.h>
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
    out.pinned = fs::temp_directory_path() / "lsx_minimax_tool";
    fs::remove_all(out.pinned);
    fs::create_directories(out.pinned);
    auto st = lsxcommon::BuildTokenizerArrowFiles(
        TOKENIZER_TEST_DATA_DIR "/synthetic_minimax_tok.json", out.pinned);
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

ChatConversation ToolConversation()
{
    ChatConversation conv;
    conv.messages.push_back({"system", "You are a helper.", "", {}});
    conv.messages.push_back({"user", "Weather in Paris?", "", {}});
    ToolSpec t;
    t.name = "get_weather";
    t.description = "Get the weather";
    t.parameters_json = "{\"type\":\"object\"}";
    conv.tools.push_back(t);
    return conv;
}

} // namespace

TEST_CASE("MiniMax chat template markers are present in the tokenizer",
          "[minimax][tool]")
{
    auto f = BuildDb();
    for (const char *m : {"]~!b[", "]~b]", "[e~[", "]<]minimax[>[",
                          "<tool_call>", "</tool_call>", "<mm:think>",
                          "</mm:think>"})
        CHECK(IdOf(f.db, m) >= 0);
    CHECK(lsxcommon::minimax::MinimaxChatSentinelsPresent(f.db));
}

TEST_CASE("MiniMax conversation frames the system/developer/user/ai turns",
          "[minimax][tool]")
{
    auto f = BuildDb();
    ChatConversation conv;
    conv.messages.push_back({"system", "SYS", "", {}});
    conv.messages.push_back({"user", "Hi", "", {}});
    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::minimax::RenderMinimaxConversation(f.db, conv, ids));
    REQUIRE(ids.size() > 8);
    // Bod + bos + encoded "system\n" at the front.
    CHECK(ids[0] == IdOf(f.db, "]~!b["));
    CHECK(ids[1] == IdOf(f.db, "]~b]"));
    // The template ends with the generation prompt `]~b]ai\n</mm:think>`.
    CHECK(ids[ids.size() - 1] == IdOf(f.db, "</mm:think>"));
}

TEST_CASE("MiniMax conversation renders the tools block and assistant calls",
          "[minimax][tool]")
{
    auto f = BuildDb();
    ChatConversation conv = ToolConversation();
    ChatMessage a;
    a.role = "assistant";
    a.content = "On it.";
    a.tool_calls.push_back({"", "get_weather", "{\"city\":\"Paris\"}"});
    conv.messages.push_back(a);
    conv.messages.push_back({"tool", "sunny", "call_0", {}});
    conv.messages.push_back({"user", "Thanks", "", {}});

    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::minimax::RenderMinimaxConversation(f.db, conv, ids));
    // The ns marker must appear (tool definitions / invokes inject it).
    CHECK(std::count(ids.begin(), ids.end(), IdOf(f.db, "]<]minimax[>[") ) >
          0);
    // One real call plus the template's own example block (both wrap in the
    // native markers), which is why the count is not 1.
    CHECK(std::count(ids.begin(), ids.end(), IdOf(f.db, "<tool_call>")) >= 2);
    CHECK(std::count(ids.begin(), ids.end(), IdOf(f.db, "</tool_call>")) >= 2);
    CHECK(std::count(ids.begin(), ids.end(), IdOf(f.db, "</mm:think>")) >= 1);
}

TEST_CASE("MiniMax invoke render/parse round-trip", "[minimax][tool]")
{
    ToolCall in;
    in.name = "get_weather";
    in.arguments_json = "{\"city\":\"Paris\",\"days\":2}";
    const std::string xml = lsxcommon::minimax::RenderMinimaxInvoke(in);
    CHECK(xml.find("]<]minimax[>[<invoke name=\"get_weather\">") == 0);
    CHECK(xml.find("]<]minimax[>[<city>Paris]<]minimax[>[</city>") !=
          std::string::npos);

    const auto calls = lsxcommon::minimax::ParseMinimaxInvokeText(xml);
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json == "{\"city\":\"Paris\",\"days\":2}");
}

TEST_CASE("MiniMax parse extracts a native tool call from a token stream",
          "[minimax][tool]")
{
    auto f = BuildDb();
    // Build the generated stream the model would emit: text ids plus the
    // special markers as their own ids.
    std::vector<int32_t> gen;
    auto push_text = [&](const std::string &t) {
        std::vector<int32_t> buf(512, 0);
        size_t cap = buf.size();
        const int32_t n = lsxcommon::arrow_tokenizer_encode(
            f.db, t.data(), t.size(), buf.data(), &cap);
        REQUIRE(n > 0);
        gen.insert(gen.end(), buf.begin(), buf.begin() + n);
    };
    gen.push_back(IdOf(f.db, "</mm:think>"));
    push_text("Calling. ");
    gen.push_back(IdOf(f.db, "<tool_call>"));
    push_text("\n");
    push_text("]<]minimax[>[<invoke name=\"get_weather\">");
    push_text("]<]minimax[>[<city>Paris]<]minimax[>[</city>");
    push_text("]<]minimax[>[</invoke>");
    gen.push_back(IdOf(f.db, "</tool_call>"));

    std::vector<ToolCall> calls;
    REQUIRE(lsxcommon::minimax::ParseMinimaxToolCalls(f.db, gen, calls));
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json == "{\"city\":\"Paris\"}");
}

TEST_CASE("MiniMax parse falls back to the bare JSON object", "[minimax][tool]")
{
    auto f = BuildDb();
    std::vector<int32_t> gen;
    std::vector<int32_t> buf(512, 0);
    size_t cap = buf.size();
    const std::string text = "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}";
    const int32_t n = lsxcommon::arrow_tokenizer_encode(
        f.db, text.data(), text.size(), buf.data(), &cap);
    REQUIRE(n > 0);
    gen.insert(gen.end(), buf.begin(), buf.begin() + n);
    std::vector<ToolCall> calls;
    REQUIRE(lsxcommon::minimax::ParseMinimaxToolCalls(f.db, gen, calls));
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "get_weather");
}
