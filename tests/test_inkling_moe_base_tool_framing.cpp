// Inkling (base) native conversation/tool codec (host-only, synthetic
// tokenizer, no CUDA). The expected frames are the BASE checkpoint's own
// `chat_template.jinja` output, rendered and transcribed literally on
// 2026-11-17 — the template is the source of truth for this family.
//
// The verified difference from the inkling2 sibling: the tool-result header
// carries the tool name (`<|message_tool|>get_weather<|content_text|>…`).
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/converter.h>
#include <lsxcommon/inkling_moe_base_tool_format.h>
#include <lsxcommon/tokenizer.h>

using namespace lsxcommon;
using namespace lsxcommon::inkling::base;

namespace {

bool HaveFixture() { return std::filesystem::exists(TOKENIZER_TEST_DATA_DIR); }

ChatConversation Sample()
{
    ChatConversation conv;
    conv.messages.push_back({"system", "Be brief.", "", {}});
    conv.messages.push_back({"user", "Weather in Paris?", "", {}});
    ToolSpec t;
    t.name = "get_weather";
    t.description = "Get the weather";
    t.parameters_json =
        R"({"properties":{"city":{"type":"string"}},"type":"object"})";
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

TEST_CASE("inkling base tool framing: render matches chat_template.jinja",
          "[inkling][tool_framing][fast]")
{
    std::vector<Segment> segs;
    REQUIRE(RenderConversation(Sample(), segs));
    const std::string out = SegmentsText(segs);
    INFO(out);

    // The template's literal output for this conversation (rendered
    // 2026-11-17), byte for byte.
    CHECK(out ==
          "<|message_system|>tool_declare<|content_xml|>[{\"description\":"
          "\"Get the weather\",\"name\":\"get_weather\",\"parameters\":"
          "{\"properties\":{\"city\":{\"type\":\"string\"}},"
          "\"type\":\"object\"},\"type\":\"function\"}]<|end_message|>"
          "<|message_system|><|content_text|>Be brief.<|end_message|>"
          "<|message_system|><|content_text|>Thinking effort level: 0.9"
          "<|end_message|>"
          "<|message_user|><|content_text|>Weather in Paris?<|end_message|>"
          "<|message_model|><|content_text|>Checking.<|end_message|>"
          "<|message_model|>get_weather<|content_invoke_tool_json|>"
          "{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\"}}"
          "<|end_message|>"
          "<|content_model_end_sampling|>"
          // The base template appends the tool NAME to the tool role header
          // (resolved through the assistant call's id) — inkling2's does not.
          "<|message_tool|>get_weather<|content_text|>sunny, 20C"
          "<|end_message|>");
}

TEST_CASE("inkling base tool framing: single prompt + generation prompt",
          "[inkling][tool_framing][fast]")
{
    ChatConversation conv;
    conv.messages.push_back({"user", "The capital of France is", "", {}});
    std::vector<Segment> segs;
    REQUIRE(RenderConversation(conv, segs, /*generation_prompt=*/true));
    CHECK(SegmentsText(segs) ==
          "<|message_system|><|content_text|>Thinking effort level: 0.9"
          "<|end_message|>"
          "<|message_user|><|content_text|>The capital of France is"
          "<|end_message|><|message_model|>");
}

TEST_CASE("inkling base tool framing: marker text inside user content stays text",
          "[inkling][tool_framing][fast]")
{
    ChatConversation conv;
    conv.messages.push_back({"user", "literal <|message_model|> token", "", {}});
    std::vector<Segment> segs;
    REQUIRE(RenderConversation(conv, segs));
    int control = 0;
    int text_segments = 0;
    bool saw_literal = false;
    for (const auto &s : segs) {
        if (s.is_control) {
            ++control;
            continue;
        }
        ++text_segments;
        if (s.text == "literal <|message_model|> token") saw_literal = true;
    }
    CHECK(saw_literal);
    // effort (system + text + end) then the user turn (role + text + end):
    // six control markers, two free-text segments.
    CHECK(control == 6);
    CHECK(text_segments == 2);
}

TEST_CASE("inkling base tool framing: assistant-only effort line and empty turn",
          "[inkling][tool_framing][fast]")
{
    ChatConversation sys;
    sys.messages.push_back({"system", "only system", "", {}});
    std::vector<Segment> segs;
    REQUIRE(RenderConversation(sys, segs));
    CHECK(SegmentsText(segs) ==
          "<|message_system|><|content_text|>only system<|end_message|>"
          "<|message_system|><|content_text|>Thinking effort level: 0.9"
          "<|end_message|>");
    const std::string once = SegmentsText(segs);
    CHECK(once.find("Thinking effort level") ==
          once.rfind("Thinking effort level"));

    ChatConversation empty;
    empty.messages.push_back({"assistant", "", "", {}});
    REQUIRE(RenderConversation(empty, segs));
    CHECK(SegmentsText(segs) ==
          "<|message_system|><|content_text|>Thinking effort level: 0.9"
          "<|end_message|><|message_model|><|content_text|><|end_message|>"
          "<|content_model_end_sampling|>");
}

TEST_CASE("inkling base tool framing: unknown role is rejected",
          "[inkling][tool_framing][fast]")
{
    ChatConversation conv;
    conv.messages.push_back({"wizard", "hi", "", {}});
    std::vector<Segment> segs;
    CHECK_FALSE(RenderConversation(conv, segs));
}

TEST_CASE("inkling base tool parse: native JSON payloads",
          "[inkling][tool_framing][fast]")
{
    std::vector<ToolCall> calls;
    const std::string text =
        "Sure.<|content_invoke_tool_json|>"
        "{\"name\":\"lookup\",\"args\":{\"q\":\"x\",\"nested\":{\"a\":1}}}"
        "<|end_message|>";
    REQUIRE(ParseToolCalls(text, calls) == 1);
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "lookup");
    CHECK(calls[0].arguments_json == R"({"q":"x","nested":{"a":1}})");

    const std::string two =
        "<|content_invoke_tool_json|>{\"name\":\"a\",\"args\":{}}<|end_message|>"
        "<|content_invoke_tool_json|>{\"name\":\"b\",\"args\":{\"k\":2}}"
        "<|end_message|>";
    REQUIRE(ParseToolCalls(two, calls) == 2);
    CHECK(calls[0].name == "a");
    CHECK(calls[0].arguments_json == "{}");
    CHECK(calls[1].name == "b");
    CHECK(calls[1].arguments_json == R"({"k":2})");

    REQUIRE(ParseToolCalls("just prose", calls) == 0);
    CHECK(calls.empty());
}

TEST_CASE("inkling base tool framing: ids round-trip through the marker table",
          "[inkling][tool_framing][fast]")
{
    namespace fs = std::filesystem;
    REQUIRE(HaveFixture());
    const auto pinned = fs::temp_directory_path() / "lsx_inkling_base_tool_tok";
    fs::remove_all(pinned);
    fs::create_directories(pinned);
    REQUIRE(BuildTokenizerArrowFiles(
                TOKENIZER_TEST_DATA_DIR "/synthetic_inkling_base_tok.json",
                pinned)
                .ok());
    ArrowTokenizerDb db;
    auto v = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    auto m = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    REQUIRE(v.ok());
    REQUIRE(m.ok());
    REQUIRE(load_tokenizer_metadata(*v, *m, db).ok());
    for (const char *mk : {"<|message_user|>", "<|message_model|>",
                           "<|message_system|>", "<|message_tool|>",
                           "<|content_text|>", "<|content_thinking|>",
                           "<|end_message|>", "<|content_model_end_sampling|>",
                           "<|content_xml|>", "<|content_invoke_tool_json|>"}) {
        CAPTURE(mk);
        CHECK(MarkerId(db, mk) >= 0);
    }
    ChatConversation conv;
    conv.messages.push_back({"user", "hello", "", {}});
    std::vector<Segment> segs;
    REQUIRE(RenderConversation(conv, segs, /*generation_prompt=*/true));
    std::vector<int32_t> ids;
    REQUIRE(AppendIds(db, segs, ids));
    REQUIRE_FALSE(ids.empty());
    CHECK(ids.front() == MarkerId(db, "<|message_system|>"));
    CHECK(ids.back() == MarkerId(db, "<|message_model|>"));
}