#include <catch2/catch_test_macros.hpp>

#include <lsxnetwork/anthropic_endpoint.h>

#include <string>
#include <vector>

using namespace lsxnetwork;

TEST_CASE("anthropic: parses request with tools and blocks",
          "[anthropic][fast]") {
    const std::string body = R"({
      "model":"m1","max_tokens":64,"stream":false,
      "system":"be nice",
      "tools":[{"name":"get_weather","description":"d",
                "input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}],
      "tool_choice":{"type":"auto"},
      "messages":[
        {"role":"user","content":[{"type":"text","text":"weather?"}]},
        {"role":"assistant","content":[{"type":"tool_use","id":"tu_1",
          "name":"get_weather","input":{"city":"Paris"}}]},
        {"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1",
          "content":"sunny"}]}]})";
    AnthropicRequest req;
    simdjson::padded_string padded(body);
    REQUIRE(ParseAnthropicRequest(padded.data(), padded.size(),
                                  padded.size() + simdjson::SIMDJSON_PADDING,
                                  req) == simdjson::SUCCESS);
    CHECK(req.model == "m1");
    CHECK(req.max_tokens == 64);
    CHECK(req.system == "be nice");
    REQUIRE(req.tools.size() == 1u);
    CHECK(req.tools[0].name == "get_weather");
    CHECK(req.tools[0].parameters_json.find("\"city\"") != std::string::npos);
    CHECK(req.tool_choice == "auto");
    REQUIRE(req.messages.size() == 3u);
    CHECK(req.messages[0].role == "user");
    CHECK(req.messages[0].content == "weather?");
    REQUIRE(req.messages[1].tool_calls.size() == 1u);
    CHECK(req.messages[1].tool_calls[0].name == "get_weather");
    CHECK(req.messages[1].tool_calls[0].arguments_json == R"({"city":"Paris"})");
    CHECK(req.messages[2].role == "tool");
    CHECK(req.messages[2].tool_call_id == "tu_1");
    CHECK(req.messages[2].content == "sunny");
}

TEST_CASE("anthropic: event framing and message builders",
          "[anthropic][fast]") {
    CHECK(AnthropicEvent("x", "{}") == "event: x\ndata: {}\n\n");
    CHECK(AnthropicMessageStart("msg_1", "m1", 3).find("message_start") !=
          std::string::npos);
    CHECK(AnthropicContentBlockDeltaText(0, "hi").find("text_delta") !=
          std::string::npos);
    CHECK(AnthropicMessageStop().find("message_stop") != std::string::npos);

    const std::string msg = BuildAnthropicMessage("msg_1", "m1", "hello",
                                                  "end_turn", 3, 2);
    CHECK(msg.find("\"type\":\"text\"") != std::string::npos);
    CHECK(msg.find("\"stop_reason\":\"end_turn\"") != std::string::npos);
    CHECK(msg.find("\"input_tokens\":3") != std::string::npos);

    const std::vector<ChatToolCall> calls{{"tu_1", "get_weather",
                                           R"({"city":"Paris"})"}};
    const std::string tool = BuildAnthropicToolMessage("msg_1", "m1", calls,
                                                       "tool_use", 3, 1);
    CHECK(tool.find("\"type\":\"tool_use\"") != std::string::npos);
    CHECK(tool.find("\"input\":{\"city\":\"Paris\"}") != std::string::npos);

    CHECK(BuildAnthropicError("nope", "invalid_request_error").find(
              "\"type\":\"error\"") != std::string::npos);
    CHECK(BuildAnthropicModelsList({"m1"}).find("\"display_name\":\"m1\"") !=
          std::string::npos);
}
