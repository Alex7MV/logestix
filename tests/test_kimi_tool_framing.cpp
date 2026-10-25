// Kimi-K2.7-Code native conversation/tool codec (host-only; synthetic
// tokenizer, no CUDA). The expected frames are transcribed from the
// checkpoint's chat_template.jinja — the source of truth — not from the
// implementation.
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/converter.h>
#include <lsxcommon/kimi_moe_tool_format.h>
#include <lsxcommon/tokenizer.h>

#include "kimi_moe_model.h"

using namespace lsxcommon;

namespace {

ChatConversation Sample()
{
    ChatConversation conv;
    conv.messages.push_back({"system", "Be brief.", "", {}});
    conv.messages.push_back({"user", "Weather in Paris?", "", {}});
    ToolSpec t;
    t.name = "get_weather";
    t.description = "Get the weather";
    t.parameters_json =
        R"({"type":"object","properties":{"city":{"type":"string"}}})";
    conv.tools.push_back(t);
    ChatMessage asst;
    asst.role = "assistant";
    asst.content = "Checking.";
    ToolCall call;
    call.id = "call_0";
    call.name = "get_weather";
    call.arguments_json = R"({"city":"Paris"})";
    asst.tool_calls.push_back(call);
    conv.messages.push_back(asst);
    conv.messages.push_back({"tool", "sunny, 20C", "call_0", {}});
    return conv;
}

}  // namespace

TEST_CASE("kimi tool framing: render matches the checkpoint template",
          "[kimi][tool_framing][fast]")
{
    std::string out;
    REQUIRE(RenderKimiConversation(Sample(), out));
    INFO(out);

    // Tools preamble (compact JSON, OpenAI tools array shape).
    CHECK(out.rfind(
              "<|im_system|>tool_declare<|im_middle|>[{\"type\":\"function\","
              "\"function\":{\"name\":\"get_weather\",\"description\":"
              "\"Get the weather\",\"parameters\":",
              0) == 0);
    // System + user frames.
    CHECK(out.find("<|im_system|>system<|im_middle|>Be brief.<|im_end|>") !=
          std::string::npos);
    CHECK(out.find("<|im_user|>user<|im_middle|>Weather in Paris?<|im_end|>") !=
          std::string::npos);
    // Assistant: thinking block, content, native call with the wire id, close.
    CHECK(out.find(
              "<|im_assistant|>assistant<|im_middle|><think></think>Checking."
              "<|tool_calls_section_begin|><|tool_call_begin|>"
              "functions.get_weather:0<|tool_call_argument_begin|>"
              "{\"city\":\"Paris\"}<|tool_call_end|>"
              "<|tool_calls_section_end|><|im_end|>") != std::string::npos);
    // Tool result echoes the synthesized wire id (not the client id).
    CHECK(out.find("<|im_system|>tool<|im_middle|>## Return of "
                   "functions.get_weather:0\nsunny, 20C<|im_end|>") !=
          std::string::npos);
    // Thinking-only generation prompt.
    const std::string gen = "<|im_assistant|>assistant<|im_middle|><think>";
    REQUIRE(out.size() >= gen.size());
    CHECK(out.substr(out.size() - gen.size()) == gen);
}

TEST_CASE("kimi tool framing: unknown role is rejected",
          "[kimi][tool_framing][fast]")
{
    ChatConversation conv;
    conv.messages.push_back({"developer", "nope", "", {}});
    std::string out;
    CHECK_FALSE(RenderKimiConversation(conv, out));
}

TEST_CASE("kimi tool framing: parse native calls with and without fallback",
          "[kimi][tool_framing][fast]")
{
    const std::string text =
        "Let me check.\n"
        "<|tool_calls_section_begin|><|tool_call_begin|>"
        "functions.get_weather:0<|tool_call_argument_begin|>"
        "{\"city\":\"Paris\",\"days\":3}<|tool_call_end|>"
        "<|tool_call_begin|>functions.get_time:1<|tool_call_argument_begin|>"
        "{\"zone\":\"UTC\"}<|tool_call_end|>"
        "<|tool_calls_section_end|>";
    const std::vector<ToolCall> calls = ParseKimiToolCalls(text);
    REQUIRE(calls.size() == 2);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json == R"({"city":"Paris","days":3})");
    CHECK(calls[0].id == "functions.get_weather:0");
    CHECK(calls[1].name == "get_time");
    CHECK(calls[1].arguments_json == R"({"zone":"UTC"})");

    // No native markers -> the generic Hermes parser still works.
    const std::vector<ToolCall> hermes = ParseKimiToolCalls(
        "<tool_call>{\"name\":\"get_time\",\"arguments\":{\"zone\":\"UTC\"}}"
        "</tool_call>");
    REQUIRE(hermes.size() == 1);
    CHECK(hermes[0].name == "get_time");
}

TEST_CASE("kimi tool framing: conversation bridge emits Kimi sentinel ids",
          "[kimi][tool_framing][fast]")
{
    namespace fs = std::filesystem;
    const auto pinned = fs::temp_directory_path() / "lsx_kimi_tool_tok";
    fs::remove_all(pinned);
    fs::create_directories(pinned);
    REQUIRE(BuildTokenizerArrowFiles(
                TOKENIZER_TEST_DATA_DIR "/synthetic_kimi_tok.json", pinned)
                .ok());

    ArrowTokenizerDb db;
    auto v = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(),
        arrow::io::FileMode::READ);
    auto m = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(),
        arrow::io::FileMode::READ);
    REQUIRE(v.ok());
    REQUIRE(m.ok());
    REQUIRE(load_tokenizer_metadata(*v, *m, db).ok());

    const int32_t im_user = arrow_tokenizer_lookup(db, "<|im_user|>", 11);
    const int32_t im_assistant =
        arrow_tokenizer_lookup(db, "<|im_assistant|>", 16);
    const int32_t im_end = arrow_tokenizer_lookup(db, "<|im_end|>", 10);
    const int32_t call_begin =
        arrow_tokenizer_lookup(db, "<|tool_call_begin|>", 19);
    const int32_t think = arrow_tokenizer_lookup(db, "<think>", 7);
    REQUIRE(im_user >= 0);
    REQUIRE(im_assistant >= 0);
    REQUIRE(im_end >= 0);
    REQUIRE(call_begin >= 0);
    REQUIRE(think >= 0);

    ModelInitConfig cfg;
    KimiK27CodeModel model(cfg);

    ChatConversation conv;
    conv.messages.push_back({"system", "Be brief.", "", {}});
    conv.messages.push_back({"user", "hi", "", {}});
    std::vector<int32_t> ids;
    REQUIRE(model.BuildConversationInputIds(db, conv, ids));
    REQUIRE(!ids.empty());
    CHECK(ids.front() == arrow_tokenizer_lookup(db, "<|im_system|>", 13));
    CHECK(std::count(ids.begin(), ids.end(), im_user) == 1);
    CHECK(std::count(ids.begin(), ids.end(), im_assistant) == 1);
    CHECK(std::count(ids.begin(), ids.end(), im_end) == 2);
    CHECK(ids.back() == think);   // generation prompt ends with <think>

    // With tools and a tool call the sentinel ids are present.
    std::vector<int32_t> tids;
    REQUIRE(model.BuildConversationInputIds(db, Sample(), tids));
    CHECK(std::count(tids.begin(), tids.end(), call_begin) == 1);

    // Parse bridge: native markers survive decode (special=false markers) and
    // are parsed back into a tool call.
    const std::string wire =
        "<|tool_calls_section_begin|><|tool_call_begin|>"
        "functions.get_weather:0<|tool_call_argument_begin|>"
        "{\"city\":\"Paris\"}<|tool_call_end|><|tool_calls_section_end|>";
    std::vector<int32_t> gen;
    REQUIRE(AppendKimiIds(db, wire, gen));
    std::vector<ToolCall> parsed;
    REQUIRE(model.ParseToolCalls(db, gen, parsed));
    REQUIRE(parsed.size() == 1);
    CHECK(parsed[0].name == "get_weather");
    CHECK(parsed[0].arguments_json == R"({"city":"Paris"})");
}
