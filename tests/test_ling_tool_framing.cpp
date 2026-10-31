// Ling-3.1-flash native chat/tool codec (Model Ingestion Protocol step 4b):
// '<role>SYSTEM</role>...<|role_end|>' framing plus the
// '<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>'
// codec, on the synthetic tokenizer fixture (host-only, label fast).
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/converter.h>
#include <lsxcommon/ling_moe_tool_format.h>
#include <lsxcommon/tokenizer.h>

#ifndef TOKENIZER_TEST_DATA_DIR
#define TOKENIZER_TEST_DATA_DIR "tests/data"
#endif

using namespace lsxcommon;

namespace {

struct Db {
    std::shared_ptr<arrow::io::MemoryMappedFile> v, m;
    ArrowTokenizerDb db;
};

bool Build(Db &out)
{
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "lsx_ling_tok_framing";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    TokenizerSpecialPolicy policy;
    policy.bos_id = -1;
    policy.pretokenizer = 3;
    policy.eos_id = 270;
    policy.pad_id = 281;
    auto st = BuildTokenizerArrowFiles(
        std::string(TOKENIZER_TEST_DATA_DIR) + "/synthetic_ling_tok.json", dir,
        policy);
    if (!st.ok()) return false;
    auto v = arrow::io::MemoryMappedFile::Open(
        (dir / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    auto m = arrow::io::MemoryMappedFile::Open(
        (dir / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    if (!v.ok() || !m.ok()) return false;
    out.v = *v;
    out.m = *m;
    return load_tokenizer_metadata(out.v, out.m, out.db).ok();
}

std::string Decode(const ArrowTokenizerDb &db, const std::vector<int32_t> &ids)
{
    std::string s(ids.size() * 16 + 64, '\0');
    size_t cap = s.size();
    if (arrow_tokenizer_decode(db, ids.data(), ids.size(), s.data(), &cap) != 0)
        return {};
    s.resize(cap);
    return s;
}

int32_t Id(const ArrowTokenizerDb &db, const char *m)
{
    return arrow_tokenizer_lookup(db, m, std::strlen(m));
}

bool Has(const std::string &hay, const std::string &needle)
{
    return hay.find(needle) != std::string::npos;
}

} // namespace

TEST_CASE("Ling renderer emits the native role framing and tool block",
          "[ling][tool]")
{
    Db t;
    REQUIRE(Build(t));
    CHECK(t.db.tokenizer_kind == 5);
    CHECK(t.db.bos_id == -1);
    CHECK(t.db.eos_id == 270);
    CHECK(t.db.pad_id == 281);
    REQUIRE(Id(t.db, "<role>") >= 0);
    REQUIRE(Id(t.db, "</role>") >= 0);
    REQUIRE(Id(t.db, "<|role_end|>") >= 0);
    REQUIRE(Id(t.db, "<tool_call>") >= 0);

    ChatConversation conv;
    conv.tools.push_back(
        {"get_weather", "Look up the weather",
         R"({"type":"object","properties":{"city":{"type":"string"}}})"});
    conv.messages.push_back({"system", "You are helpful.", "", {}});
    conv.messages.push_back({"user", "Weather in Paris?", "", {}});
    ChatMessage a;
    a.role = "assistant";
    a.content = "Let me check.";
    a.tool_calls.push_back(
        {"", "get_weather", R"({"city":"Paris","units":"c"})"});
    conv.messages.push_back(a);
    conv.messages.push_back({"tool", "sunny", "call", {}});

    std::vector<int32_t> ids;
    REQUIRE(ling::RenderLingConversation(t.db, conv, ids));
    CHECK_FALSE(ids.empty());
    const std::string text = Decode(t.db, ids);
    CHECK(Has(text, "You are helpful."));
    CHECK(Has(text, "detailed thinking on"));
    CHECK(Has(text, "# Tools"));
    CHECK(Has(text, "<tools>"));
    CHECK(Has(text, "get_weather"));
    CHECK(Has(text, "HUMAN"));
    CHECK(Has(text, "Weather in Paris?"));
    CHECK(Has(text, "ASSISTANT"));
    CHECK(Has(text, "Let me check."));
    CHECK(Has(text, "<tool_call>get_weather<arg_key>city</arg_key>"));
    CHECK(Has(text, "<arg_value>Paris</arg_value>"));
    CHECK(Has(text, "<arg_key>units</arg_key>"));
    CHECK(Has(text, "<arg_value>c</arg_value>"));
    CHECK(Has(text, "</tool_call>"));
    CHECK(Has(text, "OBSERVATION"));
    CHECK(Has(text, "<tool_response>"));
    CHECK(Has(text, "sunny"));
    CHECK(Has(text, "</tool_response>"));
    // Generation prompt: the assistant turn ends inside an open think block.
    REQUIRE(text.size() > 16);
    CHECK(text.rfind("ASSISTANT\n<think>") != std::string::npos);
}

TEST_CASE("Ling single-prompt framing never leaves the prompt unframed",
          "[ling][tool]")
{
    Db t;
    REQUIRE(Build(t));
    ChatConversation conv;
    conv.messages.push_back({"user", "The capital of France is", "", {}});
    std::vector<int32_t> ids;
    REQUIRE(ling::RenderLingConversation(t.db, conv, ids));
    const std::string text = Decode(t.db, ids);
    CHECK(Has(text, "detailed thinking on"));
    CHECK(Has(text, "HUMAN"));
    CHECK(Has(text, "The capital of France is"));
    CHECK(text.rfind("ASSISTANT\n<think>") != std::string::npos);
}

TEST_CASE("Ling native tool parser recovers names and arguments",
          "[ling][tool]")
{
    Db t;
    REQUIRE(Build(t));
    SECTION("flat arguments")
    {
        std::vector<int32_t> ids;
        REQUIRE(ling::EncodeLingWithControls(
            t.db,
            "<tool_call>get_weather<arg_key>city</arg_key>\n"
            "<arg_value>Paris</arg_value>\n</tool_call>",
            ids));
        std::vector<ToolCall> calls;
        REQUIRE(ling::ParseLingToolCalls(t.db, ids, calls));
        REQUIRE(calls.size() == 1);
        CHECK(calls[0].name == "get_weather");
        CHECK(calls[0].arguments_json == R"({"city":"Paris"})");
    }
    SECTION("nested JSON and a parallel call")
    {
        std::vector<int32_t> ids;
        REQUIRE(ling::EncodeLingWithControls(
            t.db,
            "<tool_call>search<arg_key>filter</arg_key>\n"
            "<arg_value>{\"a\":1,\"b\":[1,2]}</arg_value>\n</tool_call>"
            "<tool_call>lookup<arg_key>id</arg_key>\n"
            "<arg_value>42</arg_value>\n</tool_call>",
            ids));
        std::vector<ToolCall> calls;
        REQUIRE(ling::ParseLingToolCalls(t.db, ids, calls));
        REQUIRE(calls.size() == 2);
        CHECK(calls[0].name == "search");
        CHECK(calls[0].arguments_json == R"({"filter":{"a":1,"b":[1,2]}})");
        CHECK(calls[1].name == "lookup");
        CHECK(calls[1].arguments_json == R"({"id":42})");
    }
}
