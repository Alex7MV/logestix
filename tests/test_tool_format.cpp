#include <catch2/catch_test_macros.hpp>

#include <lsxcommon/tool_format.h>

#include <string>
#include <vector>

using namespace lsxcommon;

TEST_CASE("tool_format: tools block is valid JSON in <tools>",
          "[tools][fast]") {
    std::vector<ToolSpec> ts{
        {"get_weather", "Get weather",
         R"({"type":"object","properties":{"city":{"type":"string"}}})"}};
    const std::string b = tools::RenderToolsBlock(ts);
    CHECK(b.rfind("<tools>", 0) == 0u);
    CHECK(b.find("get_weather") != std::string::npos);
    CHECK(b.rfind("</tools>") != std::string::npos);
    CHECK(b.find("\"type\":\"function\"") != std::string::npos);
}

TEST_CASE("tool_format: parses a Hermes tool call", "[tools][fast]") {
    const std::string text =
        "sure<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>";
    const auto calls = tools::ParseToolCallsText(text);
    REQUIRE(calls.size() == 1u);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json == R"({"city":"Paris"})");
    CHECK(calls[0].id == "call_0");
}

TEST_CASE("tool_format: parses multiple calls and string arguments",
          "[tools][fast]") {
    const std::string text =
        "<tool_call>{\"name\":\"a\",\"arguments\":\"{}\"}</tool_call>"
        "<tool_call>{\"name\":\"b\",\"arguments\":{\"x\":1}}</tool_call>";
    const auto calls = tools::ParseToolCallsText(text);
    REQUIRE(calls.size() == 2u);
    CHECK(calls[0].name == "a");
    CHECK(calls[0].arguments_json == "{}");
    CHECK(calls[1].name == "b");
    CHECK(calls[1].arguments_json == R"({"x":1})");
    CHECK(calls[1].id == "call_1");
}

TEST_CASE("tool_format: no marker yields no calls", "[tools][fast]") {
    CHECK(tools::ParseToolCallsText("plain answer").empty());
}

TEST_CASE("tool_format: renders conversation roles and assistant calls",
          "[tools][fast]") {
    ChatConversation conv;
    conv.messages.push_back({"user", "hi", "", {}});
    conv.messages.push_back(
        {"assistant", "", "", {ToolCall{"call_0", "f", R"({"a":1})"}}});
    conv.messages.push_back({"tool", "42", "call_0", {}});
    const std::string text = tools::RenderConversation(conv, false);
    CHECK(text.find("user: hi") != std::string::npos);
    CHECK(text.find("<tool_call>") != std::string::npos);
    CHECK(text.find("tool: 42") != std::string::npos);
}

TEST_CASE("tool_format: generic fallback renders the tools block",
          "[tools][fast]") {
    // IModel::BuildConversationInputIds' default passes !tools.empty() as
    // include_tools, so the generic Hermes fallback emits <tools> when specs
    // are present and omits it otherwise.
    ChatConversation conv;
    conv.messages.push_back({"user", "hi", "", {}});
    ToolSpec t;
    t.name = "f";
    t.description = "d";
    t.parameters_json = R"({"type":"object"})";
    conv.tools.push_back(t);
    const std::string text = tools::RenderConversation(conv, !conv.tools.empty());
    CHECK(text.find("<tools>") != std::string::npos);
    CHECK(text.find("\"name\":\"f\"") != std::string::npos);
    CHECK(text.find("</tools>") != std::string::npos);

    ChatConversation plain;
    plain.messages.push_back({"user", "hi", "", {}});
    CHECK(tools::RenderConversation(plain, !plain.tools.empty())
              .find("<tools>") == std::string::npos);
}
