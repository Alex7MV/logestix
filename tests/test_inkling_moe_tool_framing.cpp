// Inkling-2-Small native conversation/tool codec (host-only, synthetic
// tokenizer, no CUDA). The expected frames are the checkpoint's own
// `chat_template.jinja` output, transcribed literally — the template is the
// source of truth for this family.
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/converter.h>
#include <lsxcommon/inkling_moe_tool_format.h>
#include <lsxcommon/tokenizer.h>

using namespace lsxcommon;

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

TEST_CASE("inkling tool framing: render matches chat_template.jinja",
          "[inkling][tool_framing][fast]")
{
    std::vector<InklingSegment> segs;
    REQUIRE(RenderInklingConversation(Sample(), segs));
    const std::string out = InklingSegmentsText(segs);
    INFO(out);

    // Tool declaration first: a system message whose text wraps the compact
    // tools array in the XML content channel.
    CHECK(out.rfind("<|message_system|>tool_declare<|content_xml|>[{\"description\":"
                    "\"Get the weather\",\"name\":\"get_weather\",\"parameters\":"
                    "{\"properties\":{\"city\":{\"type\":\"string\"}},"
                    "\"type\":\"object\"},\"type\":\"function\"}]"
                    "<|end_message|>",
                    0) == 0);

    // A leading SYSTEM message does NOT consume the effort line: the template
    // emits it before the first non-system turn.
    CHECK(out.find("<|message_system|><|content_text|>Be brief.<|end_message|>") !=
          std::string::npos);
    CHECK(out.find("<|message_system|><|content_text|>Be brief.<|end_message|>"
                   "<|message_system|><|content_text|>Thinking effort level: "
                   "0.9<|end_message|><|message_user|>") != std::string::npos);

    // User turn.
    CHECK(out.find("<|message_user|><|content_text|>Weather in Paris?"
                   "<|end_message|>") != std::string::npos);

    // Assistant: text, then the native call with the name in the header AND in
    // the JSON payload, then the end-of-sampling marker.
    CHECK(out.find("<|message_model|><|content_text|>Checking.<|end_message|>"
                   "<|message_model|>get_weather<|content_invoke_tool_json|>"
                   "{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\"}}"
                   "<|end_message|>"
                   "<|content_model_end_sampling|>") != std::string::npos);

    // Tool result.
    CHECK(out.find("<|message_tool|><|content_text|>sunny, 20C<|end_message|>") !=
          std::string::npos);

    // No trailing generation prompt is appended after a full turn: the model
    // continues from the last <|message_tool|> turn.
    const std::string end_marker = "<|end_message|>";
    REQUIRE(out.size() >= end_marker.size());
    CHECK(out.compare(out.size() - end_marker.size(), end_marker.size(),
                      end_marker) == 0);
}

TEST_CASE("inkling tool framing: marker text inside user content stays text",
          "[inkling][tool_framing][fast]")
{
    ChatConversation conv;
    conv.messages.push_back({"user", "literal <|message_model|> token", "", {}});
    std::vector<InklingSegment> segs;
    REQUIRE(RenderInklingConversation(conv, segs));
    // The three structural markers (effort system+text+end, user role, text,
    // end) are control segments; the user's own spelling is one text segment.
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

TEST_CASE("inkling tool framing: assistant-only effort line and empty turn",
          "[inkling][tool_framing][fast]")
{
    // A conversation that is ALL system turns still gets one effort line.
    ChatConversation sys;
    sys.messages.push_back({"system", "only system", "", {}});
    std::vector<InklingSegment> segs;
    REQUIRE(RenderInklingConversation(sys, segs));
    CHECK(InklingSegmentsText(segs) ==
          "<|message_system|><|content_text|>only system<|end_message|>"
          "<|message_system|><|content_text|>Thinking effort level: 0.9"
          "<|end_message|>");
    // ... and exactly once.
    const std::string once = InklingSegmentsText(segs);
    CHECK(once.find("Thinking effort level") ==
          once.rfind("Thinking effort level"));

    // An assistant turn with nothing at all still carries its frame and the
    // end-of-sampling marker.
    ChatConversation empty;
    empty.messages.push_back({"assistant", "", "", {}});
    REQUIRE(RenderInklingConversation(empty, segs));
    CHECK(InklingSegmentsText(segs) ==
          "<|message_system|><|content_text|>Thinking effort level: 0.9"
          "<|end_message|><|message_model|><|content_text|><|end_message|>"
          "<|content_model_end_sampling|>");
}

TEST_CASE("inkling tool framing: unknown role is rejected",
          "[inkling][tool_framing][fast]")
{
    ChatConversation conv;
    conv.messages.push_back({"wizard", "hi", "", {}});
    std::vector<InklingSegment> segs;
    CHECK_FALSE(RenderInklingConversation(conv, segs));
}

TEST_CASE("inkling tool parse: native JSON payloads", "[inkling][tool_framing][fast]")
{
    // The parser runs on decoded text with the marker re-attached; a nested
    // args object must survive the brace scan.
    std::vector<ToolCall> calls;
    const std::string text =
        "Sure.<|content_invoke_tool_json|>"
        "{\"name\":\"lookup\",\"args\":{\"q\":\"x\",\"nested\":{\"a\":1}}}"
        "<|end_message|>";
    REQUIRE(ParseInklingToolCalls(text, calls) == 1);
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "lookup");
    CHECK(calls[0].arguments_json == R"({"q":"x","nested":{"a":1}})");

    // Two calls in one turn.
    const std::string two =
        "<|content_invoke_tool_json|>{\"name\":\"a\",\"args\":{}}<|end_message|>"
        "<|content_invoke_tool_json|>{\"name\":\"b\",\"args\":{\"k\":2}}"
        "<|end_message|>";
    REQUIRE(ParseInklingToolCalls(two, calls) == 2);
    CHECK(calls[0].name == "a");
    CHECK(calls[0].arguments_json == "{}");
    CHECK(calls[1].name == "b");
    CHECK(calls[1].arguments_json == R"({"k":2})");

    // No marker at all -> no calls, and no crash.
    REQUIRE(ParseInklingToolCalls("just prose", calls) == 0);
    CHECK(calls.empty());
}

TEST_CASE("inkling tool framing: ids round-trip through the marker table",
          "[inkling][tool_framing][fast]")
{
    namespace fs = std::filesystem;
    REQUIRE(HaveFixture());
    const auto pinned = fs::temp_directory_path() / "lsx_inkling_tool_tok";
    fs::remove_all(pinned);
    fs::create_directories(pinned);
    REQUIRE(BuildTokenizerArrowFiles(
                TOKENIZER_TEST_DATA_DIR "/synthetic_inkling_tok.json", pinned)
                .ok());
    ArrowTokenizerDb db;
    auto v = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    auto m = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    REQUIRE(v.ok());
    REQUIRE(m.ok());
    REQUIRE(load_tokenizer_metadata(*v, *m, db).ok());
    const ArrowTokenizerDb *tok = &db;
    // Every marker the renderer emits must resolve to a vocabulary id, or the
    // framed prompt silently degrades to raw text.
    for (const char *m : {"<|message_user|>", "<|message_model|>",
                          "<|message_system|>", "<|message_tool|>",
                          "<|content_text|>", "<|content_thinking|>",
                          "<|end_message|>", "<|content_model_end_sampling|>",
                          "<|content_xml|>", "<|content_invoke_tool_json|>"}) {
        CAPTURE(m);
        CHECK(InklingMarkerId(db, m) >= 0);
    }
    ChatConversation conv;
    conv.messages.push_back({"user", "hello", "", {}});
    std::vector<InklingSegment> segs;
    REQUIRE(RenderInklingConversation(conv, segs));
    std::vector<int32_t> ids;
    REQUIRE(AppendInklingIds(db, segs, ids));
    REQUIRE_FALSE(ids.empty());
    // The first id is the effort line's system marker, not a BPE byte.
    CHECK(ids.front() == InklingMarkerId(db, "<|message_system|>"));
}
