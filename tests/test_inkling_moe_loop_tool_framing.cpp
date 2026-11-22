// Inkling-2-Mini native conversation/tool codec (host-only, synthetic
// tokenizer, no CUDA).
//
// The expected frames are NOT transcribed from the sibling family: they are
// this checkpoint's own `chat_template.jinja` rendered by jinja2 over the same
// conversation (`{% generation %}` markers stripped, `tojson(sort_keys=true,
// separators=(",",":"))` supplied), with the template's `raise_exception`
// wired to a Python exception. Mini's template differs from Small's only by
// those generation markers, and the render below is the whole output — the
// first case compares it byte for byte.
//
// One place Mini's codec is REQUIRED to be stricter than Small's: the template
// appends `message.name` to the tool-result role token
// (`<|message_tool|>get_weather<|content_text|>`), and ChatMessage carries no
// `name` field, so the name is resolved from the `tool_call_id` of an earlier
// assistant turn (the inkling-base / muse / beam precedent). The last case pins
// both the resolved and the unresolvable form.
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/converter.h>
#include <lsxcommon/inkling_moe_loop_tool_format.h>
#include <lsxcommon/tokenizer.h>

using namespace lsxcommon;
namespace lt = lsxcommon::inkling::loop::tools;

namespace {

constexpr const char *kTokFixture =
    TOKENIZER_TEST_DATA_DIR "/synthetic_inkling_loop_tok.json";

bool HaveFixture() { return std::filesystem::exists(kTokFixture); }

ChatConversation Sample()
{
    ChatConversation conv;
    conv.messages.push_back({"system", "Be brief.", "", {}});
    conv.messages.push_back({"user", "Weather in Paris?", "", {}});
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
    ToolSpec t;
    t.name = "get_weather";
    t.description = "Get the weather";
    t.parameters_json =
        R"({"properties":{"city":{"type":"string"}},"type":"object"})";
    conv.tools.push_back(t);
    return conv;
}

// `chat_template.jinja` output for `Sample()` (see the file header).
const char *kSampleRender =
    "<|message_system|>tool_declare<|content_xml|>"
    "[{\"description\":\"Get the weather\",\"name\":\"get_weather\","
    "\"parameters\":{\"properties\":{\"city\":{\"type\":\"string\"}},"
    "\"type\":\"object\"},\"type\":\"function\"}]"
    "<|end_message|>"
    "<|message_system|><|content_text|>Be brief.<|end_message|>"
    "<|message_system|><|content_text|>Thinking effort level: 0.9"
    "<|end_message|>"
    "<|message_user|><|content_text|>Weather in Paris?<|end_message|>"
    "<|message_model|><|content_text|>Checking.<|end_message|>"
    "<|message_model|>get_weather<|content_invoke_tool_json|>"
    "{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\"}}"
    "<|end_message|>"
    "<|content_model_end_sampling|>"
    "<|message_tool|>get_weather<|content_text|>sunny, 20C<|end_message|>";

}  // namespace

TEST_CASE("inkling loop tool framing: render matches chat_template.jinja",
          "[inkling][tool_framing][fast]")
{
    std::vector<lt::Segment> segs;
    REQUIRE(lt::RenderConversation(Sample(), segs));
    const std::string out = lt::SegmentsText(segs);
    INFO(out);
    // Byte-for-byte: the whole conversation, in the template's own order.
    CHECK(out == kSampleRender);

    // The structural rules spelled out separately, so a failure says WHICH one
    // broke instead of only that some byte moved.
    //
    // 1. Tool declarations come first, as a system turn whose text wraps the
    //    compact, key-sorted tools array in the XML content channel.
    CHECK(out.rfind("<|message_system|>tool_declare<|content_xml|>[", 0) == 0);
    // 2. A leading SYSTEM message does NOT consume the effort line: the
    //    template emits it before the first non-system turn.
    CHECK(out.find("<|message_system|><|content_text|>Be brief.<|end_message|>"
                   "<|message_system|><|content_text|>Thinking effort level: "
                   "0.9<|end_message|><|message_user|>") != std::string::npos);
    // 3. An assistant turn ends with the end-of-sampling marker, which is also
    //    the configured stop token.
    CHECK(out.find("<|content_model_end_sampling|>") != std::string::npos);
    // 4. No trailing generation prompt: the model continues from the last turn.
    const std::string end_marker = "<|end_message|>";
    REQUIRE(out.size() >= end_marker.size());
    CHECK(out.compare(out.size() - end_marker.size(), end_marker.size(),
                      end_marker) == 0);
}

TEST_CASE("inkling loop tool framing: tool-result header carries the name",
          "[inkling][tool_framing][fast]")
{
    // The name is NOT in ChatMessage: it is resolved through the tool_call_id
    // of the earlier assistant turn, as the template's `message.name` does.
    std::vector<lt::Segment> segs;
    REQUIRE(lt::RenderConversation(Sample(), segs));
    const std::string out = lt::SegmentsText(segs);
    CHECK(out.find("<|message_tool|>get_weather<|content_text|>sunny, 20C"
                   "<|end_message|>") != std::string::npos);

    // An unknown id degrades to the bare role token (no invented name).
    ChatConversation conv;
    conv.messages.push_back({"user", "hi", "", {}});
    conv.messages.push_back({"tool", "42", "call_unknown", {}});
    REQUIRE(lt::RenderConversation(conv, segs));
    const std::string bare = lt::SegmentsText(segs);
    CHECK(bare.find("<|message_tool|><|content_text|>42<|end_message|>") !=
          std::string::npos);
}

TEST_CASE("inkling loop tool framing: marker text inside content stays text",
          "[inkling][tool_framing][fast]")
{
    ChatConversation conv;
    conv.messages.push_back({"user", "literal <|message_model|> token", "", {}});
    std::vector<lt::Segment> segs;
    REQUIRE(lt::RenderConversation(conv, segs));
    // effort (system + text + end) then the user turn (role + text + end):
    // six control markers, two free-text segments — and the user's own
    // spelling stays ONE text segment, so the encoder never sees a marker.
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
    CHECK(control == 6);
    CHECK(text_segments == 2);
}

TEST_CASE("inkling loop tool framing: effort line once, empty turn framed",
          "[inkling][tool_framing][fast]")
{
    // A conversation that is ALL system turns still gets exactly one effort
    // line, at the END (there is no non-system message to precede).
    ChatConversation sys;
    sys.messages.push_back({"system", "only system", "", {}});
    std::vector<lt::Segment> segs;
    REQUIRE(lt::RenderConversation(sys, segs));
    CHECK(lt::SegmentsText(segs) ==
          "<|message_system|><|content_text|>only system<|end_message|>"
          "<|message_system|><|content_text|>Thinking effort level: 0.9"
          "<|end_message|>");

    // An assistant turn with nothing at all still carries its frame and the
    // end-of-sampling marker.
    ChatConversation empty;
    empty.messages.push_back({"assistant", "", "", {}});
    REQUIRE(lt::RenderConversation(empty, segs));
    CHECK(lt::SegmentsText(segs) ==
          "<|message_system|><|content_text|>Thinking effort level: 0.9"
          "<|end_message|><|message_model|><|content_text|><|end_message|>"
          "<|content_model_end_sampling|>");

    // An assistant turn that is ONLY tool calls skips the text wrapper.
    ChatConversation calls;
    ChatMessage m;
    m.role = "assistant";
    ToolCall c;
    c.name = "lookup";
    c.arguments_json = "{}";
    m.tool_calls.push_back(c);
    calls.messages.push_back(m);
    REQUIRE(lt::RenderConversation(calls, segs));
    CHECK(lt::SegmentsText(segs) ==
          "<|message_system|><|content_text|>Thinking effort level: 0.9"
          "<|end_message|><|message_model|>lookup<|content_invoke_tool_json|>"
          "{\"name\":\"lookup\",\"args\":{}}<|end_message|>"
          "<|content_model_end_sampling|>");
}

TEST_CASE("inkling loop tool framing: unknown role and empty name rejected",
          "[inkling][tool_framing][fast]")
{
    ChatConversation conv;
    conv.messages.push_back({"wizard", "hi", "", {}});
    std::vector<lt::Segment> segs;
    CHECK_FALSE(lt::RenderConversation(conv, segs));

    ChatConversation unnamed;
    ChatMessage m;
    m.role = "assistant";
    m.tool_calls.push_back(ToolCall{"", "", "{}"});
    unnamed.messages.push_back(m);
    CHECK_FALSE(lt::RenderConversation(unnamed, segs));
}

TEST_CASE("inkling loop tool parse: native JSON payloads",
          "[inkling][tool_framing][fast]")
{
    // The parser runs on decoded text with the marker re-attached; a nested
    // args object must survive the brace scan.
    std::vector<ToolCall> calls;
    const std::string text =
        "Sure.<|content_invoke_tool_json|>"
        "{\"name\":\"lookup\",\"args\":{\"q\":\"x\",\"nested\":{\"a\":1}}}"
        "<|end_message|>";
    REQUIRE(lt::ParseToolCalls(text, calls) == 1);
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "lookup");
    CHECK(calls[0].arguments_json == R"({"q":"x","nested":{"a":1}})");

    // Two calls in one turn.
    const std::string two =
        "<|content_invoke_tool_json|>{\"name\":\"a\",\"args\":{}}"
        "<|end_message|>"
        "<|content_invoke_tool_json|>{\"name\":\"b\",\"args\":{\"k\":2}}"
        "<|end_message|>";
    REQUIRE(lt::ParseToolCalls(two, calls) == 2);
    CHECK(calls[0].name == "a");
    CHECK(calls[0].arguments_json == "{}");
    CHECK(calls[1].name == "b");
    CHECK(calls[1].arguments_json == R"({"k":2})");

    // A brace inside a string does not end the payload early.
    const std::string quoted =
        "<|content_invoke_tool_json|>{\"name\":\"c\",\"args\":{\"s\":\"a}b\"}}"
        "<|end_message|>";
    REQUIRE(lt::ParseToolCalls(quoted, calls) == 1);
    CHECK(calls[0].arguments_json == R"({"s":"a}b"})");

    // No marker at all -> no calls, and no crash.
    REQUIRE(lt::ParseToolCalls("just prose", calls) == 0);
    CHECK(calls.empty());
}

TEST_CASE("inkling loop tool framing: ids round-trip through the marker table",
          "[inkling][tool_framing][fast]")
{
    namespace fs = std::filesystem;
    REQUIRE(HaveFixture());
    const auto pinned = fs::temp_directory_path() / "lsx_inkling_loop_tool_tok";
    fs::remove_all(pinned);
    fs::create_directories(pinned);
    REQUIRE(BuildTokenizerArrowFiles(kTokFixture, pinned).ok());
    ArrowTokenizerDb db;
    auto v = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    auto m = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    REQUIRE(v.ok());
    REQUIRE(m.ok());
    REQUIRE(load_tokenizer_metadata(*v, *m, db).ok());

    // Every marker the renderer emits must resolve to a vocabulary id, or the
    // framed prompt silently degrades to raw text.
    for (const char *spelling :
         {"<|message_user|>", "<|message_model|>", "<|message_system|>",
          "<|message_tool|>", "<|content_text|>", "<|content_thinking|>",
          "<|end_message|>", "<|content_model_end_sampling|>",
          "<|content_xml|>", "<|content_invoke_tool_json|>"}) {
        CAPTURE(spelling);
        CHECK(lt::MarkerId(db, spelling) >= 0);
    }

    std::vector<lt::Segment> segs;
    REQUIRE(lt::RenderConversation(Sample(), segs));
    std::vector<int32_t> ids;
    REQUIRE(lt::AppendIds(db, segs, ids));
    REQUIRE_FALSE(ids.empty());
    // The framed stream opens with the tool_declare system marker, exactly as
    // the captured source probe does.
    CHECK(ids.front() == lt::MarkerId(db, "<|message_system|>"));
    // ... and every control segment went in as a single id.
    std::size_t control = 0;
    for (const auto &s : segs)
        if (s.is_control) ++control;
    CHECK(control < ids.size());

    fs::remove_all(pinned);
}
