// Granite-Switch 4.1 native chat / aLoRA / tool framing (host-only).
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/granite_switch_tool_format.h"

using namespace lsxcommon;

TEST_CASE("granite-switch renderer uses role markers and generation prompt",
          "[granite_switch][fast]")
{
    ChatConversation conv;
    conv.messages.push_back({"user", "The capital of France is", "", {}});
    std::string text;
    REQUIRE(RenderGraniteSwitchConversation(conv, text));
    CHECK(text ==
          "<|start_of_role|>user<|end_of_role|>The capital of France is"
          "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>");

    ChatConversation sys;
    sys.messages.push_back({"system", "You are helpful.", "", {}});
    sys.messages.push_back({"user", "Hi", "", {}});
    REQUIRE(RenderGraniteSwitchConversation(sys, text));
    CHECK(text ==
          "<|start_of_role|>system<|end_of_role|>You are helpful."
          "<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Hi"
          "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>");
}

TEST_CASE("granite-switch aLoRA activation follows the checkpoint template",
          "[granite_switch][fast]")
{
    std::string text;
    ChatConversation dec;
    dec.adapter_name = "decision";
    dec.messages.push_back({"user", "### DECISION\nQuestion:\nWhich?", "", {}});
    REQUIRE(RenderGraniteSwitchConversation(dec, text));
    CHECK(text.find("<|decision|> DECISION\nQuestion:\nWhich?") !=
          std::string::npos);
    CHECK(text.ends_with("<|start_of_role|>assistant<|end_of_role|>"));

    ChatConversation guard;
    guard.adapter_name = "guardian-core";
    guard.messages.push_back({"user", "<guardian>As a judge agent", "", {}});
    REQUIRE(RenderGraniteSwitchConversation(guard, text));
    CHECK(text.find("<|guardian-core|>guardian>As a judge agent") !=
          std::string::npos);

    // Invocation text auto-selects the adapter when adapter_name is absent.
    ChatConversation auto_dec;
    auto_dec.messages.push_back({"user", "### DECISION\nA or B?", "", {}});
    REQUIRE(RenderGraniteSwitchConversation(auto_dec, text));
    CHECK(text.find("<|decision|> DECISION") != std::string::npos);

    // An explicit control token is not duplicated.
    ChatConversation explicit_tok;
    explicit_tok.messages.push_back({"user", "<|decision|> A or B?", "", {}});
    REQUIRE(RenderGraniteSwitchConversation(explicit_tok, text));
    CHECK(text.find("<|decision|> A or B?") != std::string::npos);
    CHECK(text.find("<|decision|><|start_of_role|>") == std::string::npos);
}

TEST_CASE("granite-switch renderer emits the tools system block",
          "[granite_switch][fast]")
{
    ChatConversation conv;
    conv.messages.push_back({"user", "Weather in Paris?", "", {}});
    ToolSpec t;
    t.name = "get_weather";
    t.description = "Get the current weather for a city";
    t.parameters_json =
        R"({"type": "object", "properties": {"city": {"type": "string"}}})";
    conv.tools.push_back(t);
    std::string text;
    REQUIRE(RenderGraniteSwitchConversation(conv, text));
    CHECK(text.rfind("<|start_of_role|>system<|end_of_role|>", 0) == 0);
    CHECK(text.find("<tools>") != std::string::npos);
    CHECK(text.find("\"name\": \"get_weather\"") != std::string::npos);
    CHECK(text.find("function signatures within <tools></tools>") !=
          std::string::npos);
}

TEST_CASE("granite-switch parses bare JSON inside tool_call",
          "[granite_switch][fast]")
{
    const std::string text =
        "reasoning\n<tool_call>\n"
        "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n"
        "</tool_call>\n";
    auto calls = ParseGraniteSwitchToolCallText(text);
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json.find("Paris") != std::string::npos);
    CHECK(calls[0].id == "call_0");
}
