// Recursive Flint native chat/tool codec (Model Ingestion Protocol step 4b).
// The golden transcripts are captured from the checkpoint's own
// chat_template.jinja by tools/flint_chat_framing_capture.py; the engine's
// renderer must reproduce them byte-for-byte, and the native parser must
// recover the calls (including nested JSON arguments and parallel calls).
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <simdjson.h>

#include <lsxcommon/converter.h>
#include <lsxcommon/flint_moe_tool_format.h>
#include <lsxcommon/tokenizer.h>

#ifndef TOKENIZER_TEST_DATA_DIR
#define TOKENIZER_TEST_DATA_DIR "tests/data"
#endif
#ifndef LSX_FLINT_FIXTURE_DIR
#define LSX_FLINT_FIXTURE_DIR "tests/fixtures/flint"
#endif

using namespace lsxcommon;

namespace {

struct GoldenCase {
    std::string name;
    ChatConversation conv;
    std::string text;
};

std::string CompactJson(const simdjson::dom::element &v)
{
    switch (v.type()) {
    case simdjson::dom::element_type::OBJECT: {
        std::string out = "{";
        bool first = true;
        for (auto [k, val] : v.get_object()) {
            if (!first) out += ",";
            first = false;
            out += "\"" + std::string(k) + "\":" + CompactJson(val);
        }
        return out + "}";
    }
    case simdjson::dom::element_type::ARRAY: {
        std::string out = "[";
        bool first = true;
        for (auto e : v.get_array()) {
            if (!first) out += ",";
            first = false;
            out += CompactJson(e);
        }
        return out + "]";
    }
    case simdjson::dom::element_type::STRING: {
        std::string_view s;
        if (v.get_string().get(s) != simdjson::SUCCESS) s = {};
        std::string out = "\"";
        for (char c : s) {
            if (c == '"' || c == '\\') out += '\\';
            out += c;
        }
        return out + "\"";
    }
    case simdjson::dom::element_type::INT64: {
        int64_t x = 0;
        const auto i64_err = v.get_int64().get(x);
        (void)i64_err;
        return std::to_string(x);
    }
    case simdjson::dom::element_type::UINT64: {
        uint64_t x = 0;
        const auto u64_err = v.get_uint64().get(x);
        (void)u64_err;
        return std::to_string(x);
    }
    case simdjson::dom::element_type::DOUBLE: {
        double d = 0.0;
        const auto f64_err = v.get_double().get(d);
        (void)f64_err;
        return std::to_string(d);
    }
    case simdjson::dom::element_type::BOOL: {
        bool b = false;
        const auto bool_err = v.get_bool().get(b);
        (void)bool_err;
        return b ? "true" : "false";
    }
    default:
        return "null";
    }
}

std::vector<GoldenCase> LoadCases()
{
    std::ifstream f(LSX_FLINT_FIXTURE_DIR "/chat_framing.jsonl", std::ios::binary);
    REQUIRE(f.good());
    std::vector<GoldenCase> cases;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        simdjson::dom::parser parser;
        simdjson::dom::element doc;
        REQUIRE(parser.parse(line).get(doc) == simdjson::SUCCESS);
        GoldenCase c;
        std::string_view sv;
        REQUIRE(doc["name"].get_string().get(sv) == simdjson::SUCCESS);
        c.name = std::string(sv);
        REQUIRE(doc["text"].get_string().get(sv) == simdjson::SUCCESS);
        c.text = std::string(sv);
        for (auto t : doc["tools"].get_array()) {
            ToolSpec spec;
            auto fn = t["function"];
            REQUIRE(fn["name"].get_string().get(sv) == simdjson::SUCCESS);
            spec.name = std::string(sv);
            if (fn["description"].get_string().get(sv) == simdjson::SUCCESS)
                spec.description = std::string(sv);
            spec.parameters_json = CompactJson(fn["parameters"]);
            c.conv.tools.push_back(std::move(spec));
        }
        for (auto m : doc["messages"].get_array()) {
            ChatMessage msg;
            REQUIRE(m["role"].get_string().get(sv) == simdjson::SUCCESS);
            msg.role = std::string(sv);
            std::string_view content;
            if (m["content"].get_string().get(content) == simdjson::SUCCESS)
                msg.content = std::string(content);
            if (m["tool_calls"].get_array().error() == simdjson::SUCCESS) {
                for (auto tc : m["tool_calls"].get_array()) {
                    auto fn = tc["function"];
                    ToolCall call;
                    REQUIRE(fn["name"].get_string().get(sv) ==
                            simdjson::SUCCESS);
                    call.name = std::string(sv);
                    call.arguments_json = CompactJson(fn["arguments"]);
                    msg.tool_calls.push_back(std::move(call));
                }
            }
            c.conv.messages.push_back(std::move(msg));
        }
        cases.push_back(std::move(c));
    }
    return cases;
}

bool LoadSyntheticTokenizer(ArrowTokenizerDb &db)
{
    const std::filesystem::path pinned =
        std::filesystem::temp_directory_path() / "lsx_flint_tool";
    std::filesystem::remove_all(pinned);
    std::filesystem::create_directories(pinned);
    // The fixture's markers are added tokens matched literally, so the policy
    // only has to say "no auto-BOS" (the template spells it out).
    TokenizerSpecialPolicy policy;
    policy.bos_id = -1;
    const auto st = BuildTokenizerArrowFiles(
        std::string(TOKENIZER_TEST_DATA_DIR) + "/synthetic_flint_tok.json",
        pinned, policy);
    if (!st.ok()) return false;
    auto vf = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(),
        arrow::io::FileMode::READ);
    auto mf = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(),
        arrow::io::FileMode::READ);
    if (!vf.ok() || !mf.ok()) return false;
    return load_tokenizer_metadata(*vf, *mf, db).ok();
}

} // namespace

TEST_CASE("flint renders the checkpoint's chat template byte-for-byte",
          "[flint][framing]")
{
    const auto cases = LoadCases();
    REQUIRE(cases.size() >= 4);
    for (const auto &c : cases) {
        INFO("case " << c.name);
        CHECK(flint::RenderFlintConversationText(c.conv,
                                                 /*add_generation_prompt=*/true) ==
              c.text);
    }
}

TEST_CASE("flint parses its native tool-call format", "[flint][framing]")
{
    // Nested JSON arguments, a numeric and a boolean value, and two parallel
    // calls in one turn.
    const std::string text =
        "I will book it.\n"
        "<tool_call>book_meeting<arg_key>title</arg_key><arg_value>Sync</arg_value>"
        "<arg_key>attendees</arg_key><arg_value>[{\"name\": \"Ada\", \"email\": "
        "\"ada@example.com\"}]</arg_value><arg_key>duration</arg_key>"
        "<arg_value>30</arg_value><arg_key>remote</arg_key>"
        "<arg_value>true</arg_value></tool_call>"
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
        "</tool_call>";
    const auto calls = flint::ParseFlintToolCallText(text);
    REQUIRE(calls.size() == 2);
    CHECK(calls[0].name == "book_meeting");
    CHECK(calls[0].arguments_json.find("\"title\":\"Sync\"") != std::string::npos);
    CHECK(calls[0].arguments_json.find("\"duration\":30") != std::string::npos);
    CHECK(calls[0].arguments_json.find("\"remote\":true") != std::string::npos);
    CHECK(calls[0].arguments_json.find("\"attendees\":[") != std::string::npos);
    CHECK(calls[0].arguments_json.find("\"Ada\"") != std::string::npos);
    CHECK(calls[1].name == "get_weather");
    CHECK(calls[1].arguments_json == "{\"city\":\"Paris\"}");

    // A call with no arguments at all.
    const auto bare = flint::ParseFlintToolCallText("<tool_call>ping</tool_call>");
    REQUIRE(bare.size() == 1);
    CHECK(bare[0].name == "ping");
    CHECK(bare[0].arguments_json == "{}");
}

TEST_CASE("flint framing injects control markers as their own ids",
          "[flint][framing]")
{
    ArrowTokenizerDb db;
    if (!LoadSyntheticTokenizer(db))
        SKIP("synthetic flint tokenizer arrows not built");
    ChatConversation conv;
    conv.messages.push_back({"user", "hi", "", {}});
    std::vector<int32_t> ids;
    REQUIRE(flint::RenderFlintConversation(db, conv, ids));
    REQUIRE(ids.size() >= 3);
    // <|user|> hi <|assistant|> — markers are single ids, never BPE'd.
    REQUIRE(ids.front() == arrow_tokenizer_lookup(db, "<|user|>", 8));
    REQUIRE(ids.back() == arrow_tokenizer_lookup(db, "<|assistant|>", 13));

    // The parser reads a decoded generation stream.
    std::vector<int32_t> gen;
    const std::string call = "<tool_call>get_weather<arg_key>city</arg_key>"
                             "<arg_value>Paris</arg_value></tool_call>";
    std::vector<int32_t> buf(call.size() * 4 + 16, 0);
    size_t cap = buf.size();
    const int32_t n = arrow_tokenizer_encode(db, call.data(), call.size(),
                                             buf.data(), &cap);
    REQUIRE(n >= 0);
    gen.assign(buf.begin(), buf.begin() + n);
    std::vector<ToolCall> parsed;
    REQUIRE(flint::ParseFlintToolCalls(db, gen, parsed));
    REQUIRE(parsed.size() == 1);
    CHECK(parsed[0].name == "get_weather");
    CHECK(parsed[0].arguments_json == "{\"city\":\"Paris\"}");
}
