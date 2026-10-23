// Granite-4.2 native tool framing (audit: the generic Hermes prompt made the
// real model invent `get_weather.x`). Host-only codec test; the real-weight
// round-trip lives in test_granite_dense_selfcheck.

#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/granite_tool_format.h"

using namespace lsxcommon;

namespace {

ChatConversation MakeConv()
{
    ChatConversation conv;
    conv.messages.push_back({"system",
                             "You are a helpful assistant. Use the provided "
                             "tools when needed.",
                             "", {}});
    conv.messages.push_back(
        {"user", "What is the current weather in Paris? Use the get_weather "
                 "tool.",
         "", {}});
    ToolSpec t;
    t.name = "get_weather";
    t.description = "Get the current weather for a city";
    t.parameters_json =
        R"({"type":"object","properties":{"city":{"type":"string","description":"City name"}},"required":["city"]})";
    conv.tools.push_back(t);
    return conv;
}

} // namespace

TEST_CASE("granite native renderer uses the ChatML tool template", "[fast]")
{
    std::string text;
    REQUIRE(RenderGraniteConversation(MakeConv(), text));

    CHECK(text.rfind("<|im_start|>system\n", 0) == 0);
    CHECK(text.find("# Tools") != std::string::npos);
    CHECK(text.find("<tools>") != std::string::npos);
    CHECK(text.find("\"name\": \"get_weather\"") != std::string::npos);
    CHECK(text.find("ONLY reply in the following format with NO suffix") !=
          std::string::npos);
    CHECK(text.find("<|im_start|>user\n") != std::string::npos);
    CHECK(text.ends_with("<|im_start|>assistant\n<think></think>\n"));
}

TEST_CASE("granite native parser reads function/parameter XML", "[fast]")
{
    const std::string text =
        "Some reasoning.\n"
        "<tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=city>\n"
        "Paris\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n";
    const auto calls = ParseGraniteToolCallText(text);
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json.find("Paris") != std::string::npos);
    CHECK(calls[0].arguments_json.find("\"city\"") != std::string::npos);
}
