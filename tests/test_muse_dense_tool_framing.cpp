// Muse-Glimmer-30B native ATEM chat/tool framing (host-only; synthetic
// tokenizer, no real model). Pins the checkpoint's chat_template.jinja frame:
// the <|start|>/<|message|>/<|eot|>/<|eom|> channel markers, the default
// system turn, the tool-defs block and the <atem:function_calls> XML codec.
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <simdjson.h>

#include <lsxcommon/converter.h>
#include <lsxcommon/tokenizer.h>

#include "test_muse_dense_fixture.h"
#include "muse_tool_format.h"

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
    out.pinned = fs::temp_directory_path() / "lsx_muse_tool";
    fs::remove_all(out.pinned);
    fs::create_directories(out.pinned);
    // Build through the kind-3 (Llama-3/o200k) policy the converter uses —
    // the framing must ride the real pre-tokenizer path, not the legacy kind-0
    // default (which also leaves bos_id unset).
    lsxcommon::TokenizerSpecialPolicy policy;
    policy.pretokenizer = 1;
    policy.bos_id = 259;
    policy.eos_id = 260;
    policy.pad_id = 263;
    auto st = lsxcommon::BuildTokenizerArrowFiles(
        TOKENIZER_TEST_DATA_DIR "/synthetic_muse_tok.json", out.pinned, policy);
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

std::vector<int32_t> Encode(const ArrowTokenizerDb &db, const std::string &t)
{
    std::vector<int32_t> buf(4096, 0);
    size_t cap = buf.size();
    const int32_t n = lsxcommon::arrow_tokenizer_encode(
        db, t.data(), t.size(), buf.data(), &cap);
    REQUIRE(n > 0);
    // kind 3 does not auto-prepend BOS; strip defensively if it ever does.
    size_t begin = (n > 0 && buf[0] == db.bos_id) ? 1 : 0;
    return std::vector<int32_t>(buf.begin() + begin, buf.begin() + n);
}

} // namespace

TEST_CASE("muse tokenizer kind-3 parity against HF goldens", "[muse][tool]")
{
    // Goldens captured from the REAL checkpoint tokenizer.json via
    // transformers AutoTokenizer (add_special_tokens=False).
    std::ifstream f(TOKENIZER_TEST_DATA_DIR "/muse/tokenizer_goldens.json");
    REQUIRE(f.good());
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    REQUIRE(parser.parse(json).get(doc) == simdjson::SUCCESS);

    auto dbf = BuildDb();
    // The synthetic fixture shares the pre-tokenizer + byte table but not the
    // 200k merge table, so exact ids cannot match; what MUST match is the
    // pre-tokenizer's word split. Re-encode each golden's decoded text and
    // compare the round-trip, then pin the split on ASCII cases where the
    // fixture's merges are a subset.
    for (auto c : doc["cases"].get_array()) {
        std::string_view text;
        REQUIRE(c["text"].get_string().get(text) == simdjson::SUCCESS);
        const auto ids = Encode(dbf.db, std::string(text));
        REQUIRE_FALSE(ids.empty());
        CHECK(Decode(dbf.db, ids) == std::string(text));
    }
}

TEST_CASE("muse ATEM markers are present in the tokenizer", "[muse][tool]")
{
    auto f = BuildDb();
    for (const char *m : {"<|begin_of_text|>", "<|end_of_text|>", "<|eot|>",
                          "<|eom|>", "<|finetune_right_pad|>", "<|start|>",
                          "<|message|>"})
        CHECK(IdOf(f.db, m) >= 0);
    CHECK(lsxcommon::muse::MuseChatSentinelsPresent(f.db));
}

TEST_CASE("muse conversation frames the default system turn", "[muse][tool]")
{
    auto f = BuildDb();
    ChatConversation conv;
    conv.messages.push_back({"user", "Hi", "", {}});
    std::vector<int32_t> ids;
    lsxcommon::muse::MuseRenderPolicy policy;
    policy.current_date = "2026-11-19";
    REQUIRE(lsxcommon::muse::RenderMuseConversation(f.db, conv, ids, policy));
    REQUIRE(ids.size() > 8);
    CHECK(ids[0] == IdOf(f.db, "<|begin_of_text|>"));
    CHECK(ids[1] == IdOf(f.db, "<|start|>"));
    // "system" BPE-encodes to 6 byte tokens (the fixture has no such merges),
    // then <|message|>. Find the marker rather than hard-coding the offset.
    const auto msg_it = std::find(ids.begin(), ids.end(),
                                  IdOf(f.db, "<|message|>"));
    REQUIRE(msg_it != ids.end());
    CHECK(msg_it - ids.begin() == 8);   // bos + start + 6 "system" bytes
    const std::string text = Decode(f.db, ids);
    CHECK(text.find("system") == 0);
    CHECK(text.find("You are a helpful AI assistant.\n"
                    "Knowledge cutoff: 2026-01-04.\n"
                    "Current date: 2026-11-19.\n\n"
                    "Reasoning strength: high.\n\n"
                    "# Valid recipients: \"self\", \"user\".") !=
          std::string::npos);
    // user turn, then the generation prompt <|start|>assistant. The trailing
    // "assistant" is TEXT after the last <|start|> marker (no <|message|>).
    CHECK(text.find("user") != std::string::npos);
    const auto last_start = std::find(ids.rbegin(), ids.rend(),
                                      IdOf(f.db, "<|start|>"));
    REQUIRE(last_start != ids.rend());
    const std::string after = Decode(f.db,
        std::vector<int32_t>(last_start.base(), ids.end()));
    CHECK(after == "assistant");
    CHECK(std::count(ids.begin(), ids.end(), IdOf(f.db, "<|eot|>")) == 2);
}

TEST_CASE("muse conversation omits the date line when unset", "[muse][tool]")
{
    auto f = BuildDb();
    ChatConversation conv;
    conv.messages.push_back({"user", "Hi", "", {}});
    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::muse::RenderMuseConversation(f.db, conv, ids));
    const std::string text = Decode(f.db, ids);
    CHECK(text.find("Current date:") == std::string::npos);
    CHECK(text.find("Knowledge cutoff: 2026-01-04.\n\nReasoning strength: high.") !=
          std::string::npos);
}

TEST_CASE("muse conversation renders the tool-defs block byte-faithful",
          "[muse][tool]")
{
    auto f = BuildDb();
    ChatConversation conv;
    conv.messages.push_back({"user", "Weather in Paris?", "", {}});
    ToolSpec t;
    t.name = "get_weather";
    t.description = "Get the weather";
    t.parameters_json =
        "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},"
        "\"required\":[\"city\"]}";
    conv.tools.push_back(t);

    std::vector<int32_t> ids;
    lsxcommon::muse::MuseRenderPolicy policy;
    policy.current_date = "2026-11-19";
    REQUIRE(lsxcommon::muse::RenderMuseConversation(f.db, conv, ids, policy));
    const std::string text = Decode(f.db, ids);
    // The template's tool instructions (verbatim literals).
    CHECK(text.find("In this environment you have access to a set of tools "
                    "you can use to answer the user's question.") !=
          std::string::npos);
    CHECK(text.find("<atem:function_calls>\n<atem:invoke name=\"$FUNCTION_NAME\">\n"
                    "<atem:parameter name=\"$PARAMETER_NAME\">$PARAMETER_VALUE"
                    "</atem:parameter>\n...\n</atem:invoke>\n</atem:function_calls>") !=
          std::string::npos);
    // Namespace metadata + function schema in Python-tojson shape.
    CHECK(text.find("// Tool metadata\n{\"name\": \"get_weather\", "
                    "\"description\": \"\"}\n// Function schemas\n"
                    "{\"name\": \"get_weather\", \"description\": "
                    "\"Get the weather\", \"parameters\": {\"type\": \"object\", "
                    "\"properties\": {\"city\": {\"type\": \"string\"}}, "
                    "\"required\": [\"city\"]}}") != std::string::npos);
    // The example call block.
    CHECK(text.find("to=example_tool_name.example_function_name\n\n"
                    "<atem:function_calls>\n<atem:invoke "
                    "name=\"example_tool_name.example_function_name\">\n"
                    "<atem:parameter name=\"example_parameter_1\">value_1"
                    "</atem:parameter>\n") != std::string::npos);
    // Valid recipients include the tool namespace wildcard.
    CHECK(text.find("# Valid recipients: \"self\", \"get_weather.*\", \"user\".") !=
          std::string::npos);
}

TEST_CASE("muse conversation renders a tool call and a tool result",
          "[muse][tool]")
{
    auto f = BuildDb();
    ChatConversation conv;
    conv.messages.push_back({"user", "Weather in Paris?", "", {}});
    ChatMessage a;
    a.role = "assistant";
    a.tool_calls.push_back({"call_0", "get_weather",
                            "{\"city\":\"Paris\",\"days\":2}"});
    conv.messages.push_back(a);
    conv.messages.push_back({"tool", "sunny", "call_0", {}});

    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::muse::RenderMuseConversation(f.db, conv, ids));
    const std::string text = Decode(f.db, ids);
    // The assistant tool channel: to=get_weather, then the ATEM block.
    CHECK(text.find("assistant to=get_weather") != std::string::npos);
    CHECK(text.find("<atem:function_calls>\n<atem:invoke name=\"get_weather\">\n"
                    "<atem:parameter name=\"city\">Paris</atem:parameter>\n"
                    "<atem:parameter name=\"days\">2</atem:parameter>\n"
                    "</atem:invoke>\n</atem:function_calls>") !=
          std::string::npos);
    // The tool result channel carries the tool name twice (header + wrapper).
    CHECK(text.find("tool get_weather") != std::string::npos);
    CHECK(text.find("<tool_output name=\"get_weather\">\nsunny\n</tool_output>") !=
          std::string::npos);
    // Generation prompt at the end: <|start|> then the text "assistant".
    const auto last_start = std::find(ids.rbegin(), ids.rend(),
                                      IdOf(f.db, "<|start|>"));
    REQUIRE(last_start != ids.rend());
    CHECK(Decode(f.db, std::vector<int32_t>(last_start.base(), ids.end())) ==
          "assistant");
}

TEST_CASE("muse ATEM tool call render/parse round-trip", "[muse][tool]")
{
    ToolCall in;
    in.name = "get_weather";
    in.arguments_json = "{\"city\":\"Paris\",\"days\":2,\"verbose\":true,"
                        "\"tags\":[\"a\",\"b\"]}";
    const std::string xml = lsxcommon::muse::RenderMuseToolCallText(in);
    CHECK(xml.find("<atem:function_calls>\n<atem:invoke name=\"get_weather\">\n") == 0);
    CHECK(xml.find("<atem:parameter name=\"city\">Paris</atem:parameter>\n") !=
          std::string::npos);
    CHECK(xml.find("<atem:parameter name=\"days\">2</atem:parameter>\n") !=
          std::string::npos);
    CHECK(xml.find("<atem:parameter name=\"verbose\">true</atem:parameter>\n") !=
          std::string::npos);
    CHECK(xml.find("<atem:parameter name=\"tags\">[\"a\", \"b\"]</atem:parameter>\n") !=
          std::string::npos);
    const auto calls = lsxcommon::muse::ParseMuseAtemText(xml);
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "get_weather");
    // Scalars keep their type; the list stays a JSON array.
    CHECK(calls[0].arguments_json.find("\"city\":\"Paris\"") != std::string::npos);
    CHECK(calls[0].arguments_json.find("\"days\":2") != std::string::npos);
    CHECK(calls[0].arguments_json.find("\"verbose\":true") != std::string::npos);
    CHECK(calls[0].arguments_json.find("\"tags\":[\"a\", \"b\"]") != std::string::npos);
}

TEST_CASE("muse parse extracts a native call from a token stream",
          "[muse][tool]")
{
    auto f = BuildDb();
    // The generated tail: assistant tool channel + ATEM block + <|eot|>.
    std::vector<int32_t> gen;
    auto push = [&](const std::string &t) {
        const auto ids = Encode(f.db, t);
        gen.insert(gen.end(), ids.begin(), ids.end());
    };
    gen.push_back(IdOf(f.db, "<|start|>"));
    push("assistant to=get_weather");
    gen.push_back(IdOf(f.db, "<|message|>"));
    push("<atem:function_calls>\n<atem:invoke name=\"get_weather\">\n"
         "<atem:parameter name=\"city\">Paris</atem:parameter>\n"
         "</atem:invoke>\n</atem:function_calls>");
    gen.push_back(IdOf(f.db, "<|eot|>"));

    std::vector<ToolCall> calls;
    REQUIRE(lsxcommon::muse::ParseMuseToolCalls(f.db, gen, calls));
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json == "{\"city\":\"Paris\"}");
}

TEST_CASE("muse parse handles two calls and a no-call answer", "[muse][tool]")
{
    auto f = BuildDb();
    auto push = [&](std::vector<int32_t> &gen, const std::string &t) {
        const auto ids = Encode(f.db, t);
        gen.insert(gen.end(), ids.begin(), ids.end());
    };
    std::vector<int32_t> gen;
    gen.push_back(IdOf(f.db, "<|start|>"));
    push(gen, "assistant to=f");
    gen.push_back(IdOf(f.db, "<|message|>"));
    push(gen, "<atem:function_calls>\n<atem:invoke name=\"f\">\n"
              "<atem:parameter name=\"x\">1</atem:parameter>\n"
              "</atem:invoke>\n</atem:function_calls>");
    gen.push_back(IdOf(f.db, "<|eom|>"));
    gen.push_back(IdOf(f.db, "<|start|>"));
    push(gen, "assistant to=g");
    gen.push_back(IdOf(f.db, "<|message|>"));
    push(gen, "<atem:function_calls>\n<atem:invoke name=\"g\">\n"
              "<atem:parameter name=\"y\">two</atem:parameter>\n"
              "</atem:invoke>\n</atem:function_calls>");
    gen.push_back(IdOf(f.db, "<|eot|>"));

    std::vector<ToolCall> calls;
    REQUIRE(lsxcommon::muse::ParseMuseToolCalls(f.db, gen, calls));
    REQUIRE(calls.size() == 2);
    CHECK(calls[0].name == "f");
    CHECK(calls[0].arguments_json == "{\"x\":1}");
    CHECK(calls[1].name == "g");
    CHECK(calls[1].arguments_json == "{\"y\":\"two\"}");

    // A plain answer channel produces no calls.
    std::vector<int32_t> answer;
    answer.push_back(IdOf(f.db, "<|start|>"));
    push(answer, "assistant to=user");
    answer.push_back(IdOf(f.db, "<|message|>"));
    push(answer, "The weather is sunny.");
    answer.push_back(IdOf(f.db, "<|eot|>"));
    std::vector<ToolCall> none;
    REQUIRE(lsxcommon::muse::ParseMuseToolCalls(f.db, answer, none));
    CHECK(none.empty());
}

TEST_CASE("muse parse falls back to the bare JSON object", "[muse][tool]")
{
    auto f = BuildDb();
    const auto gen = Encode(
        f.db, "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}");
    std::vector<ToolCall> calls;
    REQUIRE(lsxcommon::muse::ParseMuseToolCalls(f.db, gen, calls));
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "get_weather");
}

TEST_CASE("muse prompt framing wraps a raw prompt as one user turn",
          "[muse][tool]")
{
    auto f = BuildDb();
    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::muse::MusePromptInputIds(f.db, "The capital of France is",
                                                ids));
    REQUIRE(ids.size() > 8);
    CHECK(ids[0] == IdOf(f.db, "<|begin_of_text|>"));
    const std::string text = Decode(f.db, ids);
    CHECK(text.find("The capital of France is") != std::string::npos);
    const auto last_start = std::find(ids.rbegin(), ids.rend(),
                                      IdOf(f.db, "<|start|>"));
    REQUIRE(last_start != ids.rend());
    CHECK(Decode(f.db, std::vector<int32_t>(last_start.base(), ids.end())) ==
          "assistant");
}
// Byte-faithfulness against the checkpoint's own chat_template.jinja: the
// rendered text (specials re-inserted as their literal pieces) must equal
// HF AutoTokenizer.apply_chat_template byte-for-byte. Goldens captured with
// transformers 5.15.1, current_date="2026-11-19".
namespace {

std::string RenderTextWithSpecials(const ArrowTokenizerDb &db,
                                   const std::vector<int32_t> &ids)
{
    std::string out;
    for (int32_t id : ids) {
        if (db.is_special[id]) {
            const int64_t b = db.token_offsets[id];
            const int64_t e = db.token_offsets[id + 1];
            out.append(reinterpret_cast<const char *>(db.token_bytes + b),
                       static_cast<size_t>(e - b));
        } else {
            std::vector<char> buf(64);
            size_t cap = buf.size();
            REQUIRE(lsxcommon::arrow_tokenizer_decode(db, &id, 1, buf.data(),
                                                      &cap) == 0);
            out.append(buf.data(), cap);
        }
    }
    return out;
}

std::string ReadGolden(const char *name)
{
    std::ifstream f(std::string(TOKENIZER_TEST_DATA_DIR "/muse/") + name);
    REQUIRE(f.good());
    return std::string((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
}

} // namespace

TEST_CASE("muse render is byte-identical to the HF chat template (tools)",
          "[muse][tool]")
{
    auto f = BuildDb();
    ChatConversation conv;
    conv.messages.push_back({"user", "Weather in Paris?", "", {}});
    ToolSpec t;
    t.name = "get_weather";
    t.description = "Get the weather";
    t.parameters_json =
        "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},"
        "\"required\":[\"city\"]}";
    conv.tools.push_back(t);
    std::vector<int32_t> ids;
    lsxcommon::muse::MuseRenderPolicy policy;
    policy.current_date = "2026-11-19";
    REQUIRE(lsxcommon::muse::RenderMuseConversation(f.db, conv, ids, policy));
    CHECK(RenderTextWithSpecials(f.db, ids) == ReadGolden("hf_render_tools.txt"));
}

TEST_CASE("muse render is byte-identical to the HF chat template (full turn)",
          "[muse][tool]")
{
    auto f = BuildDb();
    ChatConversation conv;
    conv.messages.push_back({"user", "Weather in Paris?", "", {}});
    ChatMessage a;
    a.role = "assistant";
    a.tool_calls.push_back({"call_0", "get_weather",
                            "{\"city\":\"Paris\",\"days\":2}"});
    conv.messages.push_back(a);
    conv.messages.push_back({"tool", "sunny", "call_0", {}});
    ToolSpec t;
    t.name = "get_weather";
    t.description = "Get the weather";
    t.parameters_json =
        "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},"
        "\"required\":[\"city\"]}";
    conv.tools.push_back(t);
    std::vector<int32_t> ids;
    lsxcommon::muse::MuseRenderPolicy policy;
    policy.current_date = "2026-11-19";
    REQUIRE(lsxcommon::muse::RenderMuseConversation(f.db, conv, ids, policy));
    CHECK(RenderTextWithSpecials(f.db, ids) == ReadGolden("hf_render_full.txt"));
}
