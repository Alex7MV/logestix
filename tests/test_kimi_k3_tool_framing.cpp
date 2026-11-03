// Kimi-K3 native XTML conversation/tool codec (host-only; synthetic tokenizer,
// no CUDA). The expected frames are transcribed from the checkpoint's
// encoding_k3.py (build_chat_segments), which is its source of truth — the
// checkpoint ships no chat_template.jinja.
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/converter.h>
#include <lsxcommon/kimi_moe_k3_tool_format.h>
#include <lsxcommon/tokenizer.h>

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
    call.arguments_json = R"({"city":"Paris","days":3})";
    asst.tool_calls.push_back(call);
    conv.messages.push_back(asst);
    conv.messages.push_back({"tool", "sunny, 20C", "call_0", {}});
    return conv;
}

}  // namespace

TEST_CASE("kimi-k3 tool framing: render matches encoding_k3.py",
          "[kimi-k3][tool_framing][fast]")
{
    std::vector<KimiK3Segment> segs;
    REQUIRE(RenderKimiK3Conversation(Sample(), segs));
    const std::string out = KimiK3SegmentsText(segs);
    INFO(out);

    // Tool declaration: XTML system message, "# Tools" preamble, canonical
    // compact OpenAI tools array.
    CHECK(out.rfind(
              "<|open|>message role=\"system\" type=\"tool-declare\"<|sep|>"
              "# Tools\nHere are the available tools, described in JSONSchema."
              "\n\n```json\n[{\"function\":{\"description\":\"Get the weather\","
              "\"name\":\"get_weather\",\"parameters\":{\"properties\":"
              "{\"city\":{\"type\":\"string\"}},\"type\":\"object\"}},"
              "\"type\":\"function\"}]\n```<|close|>message<|sep|>"
              "<|end_of_msg|>",
              0) == 0);

    // System + user frames.
    CHECK(out.find("<|open|>message role=\"system\"<|sep|>Be brief."
                   "<|close|>message<|sep|><|end_of_msg|>") != std::string::npos);
    CHECK(out.find("<|open|>message role=\"user\"<|sep|>Weather in Paris?"
                   "<|close|>message<|sep|><|end_of_msg|>") != std::string::npos);

    // Assistant: structural empty think channel, response, native call.
    CHECK(out.find(
              "<|open|>message role=\"assistant\"<|sep|>"
              "<|open|>think<|sep|><|close|>think<|sep|>"
              "<|open|>response<|sep|>Checking.<|close|>response<|sep|>"
              "<|open|>tools<|sep|>"
              "<|open|>call tool=\"get_weather\" index=\"1\"<|sep|>"
              "<|open|>argument key=\"city\" type=\"string\"<|sep|>Paris"
              "<|close|>argument<|sep|>"
              "<|open|>argument key=\"days\" type=\"number\"<|sep|>3"
              "<|close|>argument<|sep|>"
              "<|close|>call<|sep|><|close|>tools<|sep|>"
              "<|close|>message<|sep|><|end_of_msg|>") != std::string::npos);

    // Tool result echoes the name and the 1-based index.
    CHECK(out.find("<|open|>message role=\"tool\" tool=\"get_weather\" "
                   "index=\"1\"<|sep|>sunny, 20C<|close|>message<|sep|>"
                   "<|end_of_msg|>") != std::string::npos);

    // Thinking generation prompt.
    const std::string gen_prompt =
        "<|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>";
    REQUIRE(out.size() >= gen_prompt.size());
    CHECK(out.compare(out.size() - gen_prompt.size(), gen_prompt.size(),
                      gen_prompt) == 0);
}

TEST_CASE("kimi-k3 tool framing: marker text inside user content stays text",
          "[kimi-k3][tool_framing][fast]")
{
    ChatConversation conv;
    conv.messages.push_back({"user", "literal <|open|> token", "", {}});
    std::vector<KimiK3Segment> segs;
    REQUIRE(RenderKimiK3Conversation(conv, segs));
    // Only the five structural markers are control segments (4 message tags,
    // 1 generation prompt open) plus end_of_msg; the user text is one text
    // segment containing the literal spelling.
    int control = 0;
    bool user_text_is_plain = false;
    for (const auto &s : segs)
        if (!s.is_control && s.text == "literal <|open|> token")
            user_text_is_plain = true;
        else if (s.is_control)
            ++control;
    CHECK(user_text_is_plain);
    CHECK(control > 0);
}

TEST_CASE("kimi-k3 tool framing: parse a native XTML tool call",
          "[kimi-k3][tool_framing][fast]")
{
    const std::string wire =
        "<|open|>response<|sep|>Sure.<|close|>response<|sep|>"
        "<|open|>tools<|sep|>"
        "<|open|>call tool=\"get_weather\" index=\"1\"<|sep|>"
        "<|open|>argument key=\"city\" type=\"string\"<|sep|>Paris"
        "<|close|>argument<|sep|>"
        "<|open|>argument key=\"days\" type=\"number\"<|sep|>3"
        "<|close|>argument<|sep|>"
        "<|close|>call<|sep|>"
        "<|open|>call tool=\"get_time\" index=\"2\"<|sep|>"
        "<|open|>json type=\"object\"<|sep|>{\"zone\":\"UTC\"}"
        "<|close|>json<|sep|>"
        "<|close|>call<|sep|>"
        "<|close|>tools<|sep|>";

    const std::vector<ToolCall> calls = ParseKimiK3ToolCalls(wire);
    REQUIRE(calls.size() == 2);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json == R"({"city":"Paris","days":3})");
    CHECK(calls[1].name == "get_time");
    CHECK(calls[1].arguments_json == R"({"zone":"UTC"})");

    // No native markers -> the generic Hermes parser still works.
    const std::vector<ToolCall> hermes = ParseKimiK3ToolCalls(
        "<tool_call>{\"name\":\"get_time\",\"arguments\":{\"zone\":\"UTC\"}}"
        "</tool_call>");
    REQUIRE(hermes.size() == 1);
    CHECK(hermes[0].name == "get_time");
}

TEST_CASE("kimi-k3 tool framing: bridge injects marker ids",
          "[kimi-k3][tool_framing][fast]")
{
    namespace fs = std::filesystem;
    const auto pinned = fs::temp_directory_path() / "lsx_kimi_k3_tool_tok";
    fs::remove_all(pinned);
    fs::create_directories(pinned);
    REQUIRE(BuildTokenizerArrowFiles(
                TOKENIZER_TEST_DATA_DIR "/synthetic_kimi_k3_tok.json", pinned)
                .ok());

    ArrowTokenizerDb db;
    auto v = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    auto m = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    REQUIRE(v.ok());
    REQUIRE(m.ok());
    REQUIRE(load_tokenizer_metadata(*v, *m, db).ok());

    const int32_t open =
        arrow_tokenizer_lookup(db, "<|open|>", std::strlen("<|open|>"));
    const int32_t sep =
        arrow_tokenizer_lookup(db, "<|sep|>", std::strlen("<|sep|>"));
    const int32_t eom = arrow_tokenizer_lookup(
        db, "<|end_of_msg|>", std::strlen("<|end_of_msg|>"));
    REQUIRE(open >= 0);
    REQUIRE(sep >= 0);
    REQUIRE(eom >= 0);

    std::vector<KimiK3Segment> segs;
    REQUIRE(RenderKimiK3Conversation(Sample(), segs));
    std::vector<int32_t> ids;
    REQUIRE(AppendKimiK3Ids(db, segs, ids));
    REQUIRE(!ids.empty());
    CHECK(ids.front() == open);
    CHECK(ids.back() == sep);  // generation prompt: <|open|>think<|sep|>
    CHECK(std::count(ids.begin(), ids.end(), eom) == 5);
    CHECK(std::count(ids.begin(), ids.end(), open) == 13);

    // Parse bridge: the non-special markers survive decode and parse back.
    const std::vector<KimiK3Segment> wsegs = {
        {"<|open|>", true}, {"tools", false}, {"<|sep|>", true},
        {"<|open|>", true}, {"call tool=\"get_weather\" index=\"1\"", false},
        {"<|sep|>", true}, {"<|open|>", true},
        {"argument key=\"city\" type=\"string\"", false}, {"<|sep|>", true},
        {"Paris", false}, {"<|close|>", true}, {"argument", false},
        {"<|sep|>", true}, {"<|close|>", true}, {"call", false},
        {"<|sep|>", true}, {"<|close|>", true}, {"tools", false},
        {"<|sep|>", true}};
    std::vector<int32_t> gen;
    REQUIRE(AppendKimiK3Ids(db, wsegs, gen));
    std::string text(8192, '\0');
    size_t cap = text.size();
    REQUIRE(arrow_tokenizer_decode(db, gen.data(), gen.size(), text.data(),
                                   &cap) == 0);
    text.resize(cap);
    const std::vector<ToolCall> parsed = ParseKimiK3ToolCalls(text);
    REQUIRE(parsed.size() == 1);
    CHECK(parsed[0].name == "get_weather");
    CHECK(parsed[0].arguments_json == R"({"city":"Paris"})");
}
