// Mistral-Small-4-119B native tool framing (host-only; no CUDA, no real model).
// Pins the exact chat_template.jinja frame and the [TOOL_CALLS]name[ARGS]{json}
// parse. The tokenizer-backed id-injection path is exercised on real weights by
// the selfcheck stage (the checkpoint's own tokenizer is not vendored).
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/mistral_moe_tool_format.h"

#include <string>
#include <vector>

using lsxcommon::ChatConversation;
using lsxcommon::ChatMessage;
using lsxcommon::ToolCall;
using lsxcommon::ToolSpec;

TEST_CASE("Mistral RenderText matches the chat template", "[mistral][tool]")
{
    ChatConversation conv;
    conv.messages.push_back({"system", "You are helpful.", "", {}});
    conv.messages.push_back({"user", "Weather?", "", {}});

    std::string text;
    REQUIRE(lsxcommon::mistral_moe::tool_format::RenderText(conv, text, true));
    CHECK(text.find("<s>[SYSTEM_PROMPT]You are helpful.[/SYSTEM_PROMPT]") == 0);
    CHECK(text.find("[INST]Weather?[/INST]") != std::string::npos);
    // The template appends nothing after the last user turn.
    CHECK(text.find("[/INST]") == text.size() - 7);
}

TEST_CASE("Mistral RenderText renders assistant tool calls", "[mistral][tool]")
{
    ChatConversation conv;
    ToolSpec t;
    t.name = "get_weather";
    t.description = "Get the weather";
    t.parameters_json = "{\"type\":\"object\"}";
    conv.tools.push_back(t);
    conv.messages.push_back({"user", "Weather?", "", {}});
    ChatMessage a;
    a.role = "assistant";
    a.tool_calls.push_back({"", "get_weather", "{\"city\":\"Paris\"}"});
    conv.messages.push_back(a);

    std::string text;
    REQUIRE(lsxcommon::mistral_moe::tool_format::RenderText(conv, text, false));
    CHECK(text.find("[AVAILABLE_TOOLS][{\"type\":\"function\"") !=
          std::string::npos);
    CHECK(text.find("[MODEL_SETTINGS]{\"reasoning_effort\": \"none\"}"
                    "[/MODEL_SETTINGS]") != std::string::npos);
    CHECK(text.find("[TOOL_CALLS]get_weather[ARGS]{\"city\":\"Paris\"}</s>") !=
          std::string::npos);
}

TEST_CASE("Mistral ParseToolCallText extracts native calls", "[mistral][tool]")
{
    const std::string text =
        "Sure.[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Paris\", \"n\": 2}"
        "[TOOL_CALLS]rewrite[ARGS]{}</s>";
    const auto calls =
        lsxcommon::mistral_moe::tool_format::ParseToolCallText(text);
    REQUIRE(calls.size() == 2);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json == "{\"city\": \"Paris\", \"n\": 2}");
    CHECK(calls[1].name == "rewrite");
    CHECK(calls[1].arguments_json == "{}");
}

TEST_CASE("Mistral RenderText round-trips through the parser", "[mistral][tool]")
{
    ChatConversation conv;
    ChatMessage a;
    a.role = "assistant";
    a.content = "On it.";
    a.tool_calls.push_back({"", "get_weather", "{\"city\":\"Paris\"}"});
    conv.messages.push_back(a);

    std::string text;
    REQUIRE(lsxcommon::mistral_moe::tool_format::RenderText(conv, text, false));
    const auto calls =
        lsxcommon::mistral_moe::tool_format::ParseToolCallText(text);
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json == "{\"city\":\"Paris\"}");
}
