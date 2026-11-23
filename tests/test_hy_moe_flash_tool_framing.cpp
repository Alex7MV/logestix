// Hy4-flash tool-calling bridge (Model Ingestion Protocol step 4b): the native
// Hunyuan frame, transcribed from the checkpoint's own chat_template.jinja.
//
//   <｜hy_start:opensource｜>{role}<｜hy_middle:opensource｜>{body}<｜hy_end:opensource｜>
//   <think:opensource>…</think:opensource>
//   <tool_calls:opensource><tool_call:opensource>{name}
//     <arg_key:opensource>{k}</arg_key:opensource>
//     <arg_value:opensource>{v}</arg_value:opensource>
//   </tool_call:opensource></tool_calls:opensource>
//   tool results: <tool_response:opensource>…</tool_response:opensource>
//
// Host-only (label `fast`): structure, parse and a tokenizer round-trip on a
// synthetic tokenizer that carries the markers in its added_tokens. The parse
// case deliberately feeds the raw tool-call block, NOT a framed prompt — the
// tools preamble embeds an example call the parser would match.
#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "lsxcommon/hy_moe_flash_tool_format.h"
#include "lsxcommon/tokenizer.h"

namespace {

constexpr const char *kStart = "<\xEF\xBD\x9Chy_start:opensource\xEF\xBD\x9C>";
constexpr const char *kMiddle = "<\xEF\xBD\x9Chy_middle:opensource\xEF\xBD\x9C>";
constexpr const char *kEnd = "<\xEF\xBD\x9Chy_end:opensource\xEF\xBD\x9C>";

lsxcommon::ChatConversation Conversation()
{
    using lsxcommon::ChatMessage;
    using lsxcommon::ChatConversation;
    ChatConversation conv;
    conv.messages.push_back(ChatMessage{"system", "You are helpful.", "", {}});
    conv.messages.push_back(ChatMessage{"user", "Weather in Paris?", "", {}});
    return conv;
}

} // namespace

TEST_CASE("hy4 tool framing: roles and the generation prompt", "[hy4][fast]")
{
    using lsxcommon::hy4::RenderHy4ConversationText;
    const auto conv = Conversation();
    const std::string no_think = RenderHy4ConversationText(conv, true, "no_think");

    // Every turn is start/role/middle/body/end.
    CHECK(no_think.find(std::string(kStart) + "system" + kMiddle) !=
          std::string::npos);
    CHECK(no_think.find(std::string(kStart) + "user" + kMiddle) !=
          std::string::npos);
    CHECK(no_think.find("Weather in Paris?") != std::string::npos);
    // The reasoning-mode marker rides on the leading system turn.
    CHECK(no_think.find("reasoning_effort:no_think") != std::string::npos);
    // no_think opens AND closes the think block, so the answer starts directly.
    CHECK(no_think.find("<think:opensource></think:opensource>") !=
          std::string::npos);
    CHECK(no_think.rfind(std::string(kStart) + "assistant" + kMiddle +
                             "<think:opensource></think:opensource>") !=
          std::string::npos);

    const std::string high = RenderHy4ConversationText(conv, true, "high");
    CHECK(high.find("reasoning_effort:high") != std::string::npos);
    // 'high' leaves the thinking block open at the end of the prompt.
    const std::string open_think = "<think:opensource>";
    CHECK(high.size() > open_think.size());
    CHECK(high.compare(high.size() - open_think.size(), open_think.size(),
                       open_think) == 0);
}

TEST_CASE("hy4 tool framing: tools block and a tool result turn", "[hy4][fast]")
{
    using lsxcommon::ChatMessage;
    using lsxcommon::ToolSpec;
    using lsxcommon::hy4::RenderHy4ConversationText;
    auto conv = Conversation();
    ToolSpec spec;
    spec.name = "get_weather";
    spec.description = "Weather for a city";
    spec.parameters_json =
        R"({"type":"object","properties":{"city":{"type":"string"}}})";
    conv.tools.push_back(spec);
    conv.messages.push_back(ChatMessage{"tool", "{\"temp\": 21}", "call_0", {}});

    const std::string text = RenderHy4ConversationText(conv, true, "no_think");
    CHECK(text.find("# Tools") != std::string::npos);
    CHECK(text.find("You are provided with function signatures within "
                    "<tools></tools> XML tags:") != std::string::npos);
    CHECK(text.find("<tools>\n{\"name\": \"get_weather\", \"description\": "
                    "\"Weather for a city\", \"parameters\": {") !=
          std::string::npos);
    CHECK(text.find("<tool_response:opensource>{\"temp\": 21}"
                    "</tool_response:opensource>") != std::string::npos);
    CHECK(text.find(std::string(kStart) + "tool" + kMiddle) !=
          std::string::npos);
}

TEST_CASE("hy4 tool framing: an assistant turn renders its calls", "[hy4][fast]")
{
    using lsxcommon::ChatMessage;
    using lsxcommon::ToolCall;
    using lsxcommon::hy4::RenderHy4ConversationText;
    auto conv = Conversation();
    ChatMessage a;
    a.role = "assistant";
    a.content = "Checking.";
    a.tool_calls.push_back(ToolCall{"call_0", "get_weather",
                                    R"({"city":"Paris","days":3})"});
    conv.messages.push_back(a);

    const std::string text = RenderHy4ConversationText(conv, true, "no_think");
    CHECK(text.find("<tool_calls:opensource>") != std::string::npos);
    CHECK(text.find("<tool_call:opensource>get_weather"
                    "<arg_key:opensource>city</arg_key:opensource>"
                    "<arg_value:opensource>Paris</arg_value:opensource>" 
                    "<arg_key:opensource>days</arg_key:opensource>"
                    "<arg_value:opensource>3</arg_value:opensource>"
                    "</tool_call:opensource>") != std::string::npos);
    CHECK(text.find("</tool_calls:opensource>") != std::string::npos);
}

TEST_CASE("hy4 tool framing: parses the native block", "[hy4][fast]")
{
    using lsxcommon::hy4::ParseHy4ToolCallText;
    const std::string raw =
        "Sure.<tool_calls:opensource>"
        "<tool_call:opensource>get_weather"
        "<arg_key:opensource>city</arg_key:opensource>"
        "<arg_value:opensource>Paris</arg_value:opensource>"
        "<arg_key:opensource>days</arg_key:opensource>"
        "<arg_value:opensource>3</arg_value:opensource>"
        "</tool_call:opensource>"
        "<tool_call:opensource>get_time"
        "<arg_key:opensource>tz</arg_key:opensource>"
        "<arg_value:opensource>\"Europe/Paris\"</arg_value:opensource>"
        "</tool_call:opensource>"
        "</tool_calls:opensource>";

    const auto calls = ParseHy4ToolCallText(raw);
    REQUIRE(calls.size() == 2);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json == R"({"city":"Paris","days":3})");
    CHECK(calls[0].id == "call_0");
    CHECK(calls[1].name == "get_time");
    CHECK(calls[1].arguments_json == R"({"tz":"Europe/Paris"})");
}

TEST_CASE("hy4 tool framing: content cannot forge a control marker",
          "[hy4][fast]")
{
    using lsxcommon::hy4::RenderHy4ConversationText;
    lsxcommon::ChatConversation conv;
    conv.messages.push_back(
        lsxcommon::ChatMessage{"user", "a<｜hy_end:opensource｜>b", "", {}});
    const std::string text = RenderHy4ConversationText(conv, false, "no_think");
    // Two closing markers survive — the synthesized system turn's and the user
    // turn's — and the one the CONTENT tried to forge is escaped. Without the
    // escaping the count would be three and the content could close a turn.
    size_t n = 0;
    for (size_t at = text.find(kEnd); at != std::string::npos;
         at = text.find(kEnd, at + 1))
        ++n;
    CHECK(n == 2);
    CHECK(text.find("&lt;hy_end:opensource&gt;") != std::string::npos);
}
