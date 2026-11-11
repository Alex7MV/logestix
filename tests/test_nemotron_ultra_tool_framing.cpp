// TDD — Nemotron-3-Ultra-550B conversation codec, verified against the
// checkpoint's own chat_template.jinja (sha fa104c9a…), NOT against the 30B
// sibling's (a2445a9f…). The two templates differ in one semantic place — the
// `medium_effort` marker appended to the last user turn — and in nothing else
// that reaches the wire, which is why the call syntax is shared and this test
// pins the parts Ultra owns.
#include <catch2/catch_test_macros.hpp>

#include <string>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/nemotron_ultra_tool_format.h"

using namespace lsxcommon;
using namespace lsxcommon::nemotron_ultra;

namespace {

ChatMessage Msg(const std::string &role, const std::string &content)
{
    ChatMessage m;
    m.role = role;
    m.content = content;
    return m;
}

ToolSpec WeatherTool()
{
    ToolSpec t;
    t.name = "get_weather";
    t.description = "Get the weather for a city";
    t.parameters_json =
        R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})";
    return t;
}

} // namespace

TEST_CASE("ultra framing: a bare user turn opens an unclosed think block",
          "[ultra][framing]")
{
    // enable_thinking defaults TRUE in this checkpoint's template (the qwen
    // families default the other way), so the generation prompt is
    // "<|im_start|>assistant\n<think>\n" with no </think>.
    ChatConversation conv;
    conv.messages.push_back(Msg("user", "The capital of France is"));
    std::string out;
    REQUIRE(RenderUltraConversation(conv, out));
    CHECK(out ==
          "<|im_start|>user\nThe capital of France is<|im_end|>\n"
          "<|im_start|>assistant\n<think>\n");
}

TEST_CASE("ultra framing: thinking closed puts an empty block on the prompt",
          "[ultra][framing]")
{
    ChatConversation conv;
    conv.messages.push_back(Msg("user", "hi"));
    UltraRenderPolicy p;
    p.enable_thinking = false;
    std::string out;
    REQUIRE(RenderUltraConversation(conv, out, p));
    CHECK(out == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think></think>");
}

TEST_CASE("ultra framing: tools render the Ultra preamble with JSON parameters",
          "[ultra][framing]")
{
    ChatConversation conv;
    conv.messages.push_back(Msg("user", "weather in Paris?"));
    conv.tools.push_back(WeatherTool());
    std::string out;
    REQUIRE(RenderUltraConversation(conv, out));

    // The preamble is Ultra's own wording, not Qwen's and not the 30B's.
    CHECK(out.rfind("<|im_start|>system\n# Tools\n\nYou have access to the "
                    "following functions:\n\n<tools>",
                    0) == 0);
    CHECK(out.find("<function>\n<name>get_weather</name>") != std::string::npos);
    CHECK(out.find("<description>Get the weather for a city</description>") !=
          std::string::npos);
    // jinja `tojson` spacing: ", " and ": " inside the schema.
    CHECK(out.find("\"type\": \"object\"") != std::string::npos);
    CHECK(out.find("\"required\": [\"city\"]") != std::string::npos);
    CHECK(out.find("</tools>") != std::string::npos);
    // The reminder block belongs to this template.
    CHECK(out.find("<IMPORTANT>\nReminder:") != std::string::npos);
    CHECK(out.find("If you choose to call a function ONLY reply in the following "
                   "format with NO suffix") != std::string::npos);
    // The tool system turn is closed before the user turn.
    CHECK(out.find("</IMPORTANT><|im_end|>\n<|im_start|>user\n") !=
          std::string::npos);
}

TEST_CASE("ultra framing: a system turn comes first and tools follow it",
          "[ultra][framing]")
{
    ChatConversation conv;
    conv.messages.push_back(Msg("system", "You are helpful."));
    conv.messages.push_back(Msg("user", "hi"));
    conv.tools.push_back(WeatherTool());
    std::string out;
    REQUIRE(RenderUltraConversation(conv, out));
    CHECK(out.rfind("<|im_start|>system\nYou are helpful.\n\n# Tools", 0) == 0);
}

TEST_CASE("ultra framing: an assistant tool call renders the XML call form",
          "[ultra][framing]")
{
    ChatConversation conv;
    conv.messages.push_back(Msg("user", "weather in Paris?"));
    ChatMessage a = Msg("assistant", "");
    ToolCall c;
    c.name = "get_weather";
    c.arguments_json = R"({"city":"Paris"})";
    a.tool_calls.push_back(c);
    conv.messages.push_back(std::move(a));
    std::string out;
    REQUIRE(RenderUltraConversation(conv, out));
    CHECK(out.find("<|im_start|>assistant\n<think></think><tool_call>\n"
                   "<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
                   "</function>\n</tool_call>\n<|im_end|>\n") != std::string::npos);
}

TEST_CASE("ultra framing: consecutive tool results share one user turn",
          "[ultra][framing]")
{
    ChatConversation conv;
    conv.messages.push_back(Msg("user", "q"));
    ChatMessage a = Msg("assistant", "");
    ToolCall c1, c2;
    c1.name = "get_weather";
    c1.arguments_json = R"({"city":"Paris"})";
    c2.name = "get_weather";
    c2.arguments_json = R"({"city":"Lyon"})";
    a.tool_calls = {c1, c2};
    conv.messages.push_back(std::move(a));
    ChatMessage t1 = Msg("tool", "sunny");
    t1.tool_call_id = "1";
    ChatMessage t2 = Msg("tool", "rain");
    t2.tool_call_id = "2";
    conv.messages.push_back(std::move(t1));
    conv.messages.push_back(std::move(t2));
    std::string out;
    REQUIRE(RenderUltraConversation(conv, out));

    // ONE user turn, two <tool_response> blocks, closed once at the end.
    const auto first = out.find("<|im_start|>user\n<tool_response>");
    REQUIRE(first != std::string::npos);
    CHECK(out.find("<|im_start|>user\n<tool_response>", first + 1) == std::string::npos);
    CHECK(out.find("<tool_response>\nsunny\n</tool_response>\n<tool_response>\nrain\n"
                   "</tool_response>\n<|im_end|>\n") != std::string::npos);
}

TEST_CASE("ultra framing: medium_effort annotates only the last user turn",
          "[ultra][framing]")
{
    ChatConversation conv;
    conv.messages.push_back(Msg("user", "first"));
    conv.messages.push_back(Msg("assistant", "answer"));
    conv.messages.push_back(Msg("user", "second"));
    UltraRenderPolicy p;
    p.medium_effort = true;
    std::string out;
    REQUIRE(RenderUltraConversation(conv, out, p));
    // The marker reaches the LAST user turn only.
    CHECK(out.find("<|im_start|>user\nfirst<|im_end|>") != std::string::npos);
    CHECK(out.find("<|im_start|>user\nsecond\n\n{reasoning effort: efficient}"
                   "<|im_end|>") != std::string::npos);
}

TEST_CASE("ultra framing: history reasoning is truncated before the last user turn",
          "[ultra][framing]")
{
    ChatConversation conv;
    conv.messages.push_back(Msg("user", "q1"));
    conv.messages.push_back(Msg("assistant", "<think>\nlong chain\n</think>short"));
    conv.messages.push_back(Msg("user", "q2"));
    std::string out;
    REQUIRE(RenderUltraConversation(conv, out));
    // The tool-less history turn keeps only the post-think content, prefixed by
    // an EMPTY closed block (template's truncate_history_thinking default).
    CHECK(out.find("<|im_start|>assistant\n<think></think>short<|im_end|>") !=
          std::string::npos);
    CHECK(out.find("long chain") == std::string::npos);
}

TEST_CASE("ultra framing: an empty conversation renders nothing", "[ultra][framing]")
{
    ChatConversation conv;
    std::string out = "sentinel";
    CHECK_FALSE(RenderUltraConversation(conv, out));
}
