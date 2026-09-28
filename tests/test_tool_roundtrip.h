#pragma once

// Shared real-weight tool round-trip helper (Model Ingestion Protocol step 4b
// verification). Host header, included from inference-labelled selfcheck
// tests; family-neutral so any suite may use it.
//
// Frames a one-tool conversation through the model's own conversation bridge,
// generates, and parses the tool call back out of the generated tokens.

#include <catch2/catch_test_macros.hpp>

#include <spdlog/spdlog.h>

#include <lsxcommon/model.h>

#include <string>
#include <vector>

inline std::vector<lsxcommon::ToolCall> RunToolRoundTrip(
    lsxcommon::IModel &model, int max_gen_tokens = 256)
{
    using namespace lsxcommon;

    ChatConversation conv;
    conv.messages.push_back(
        {"system",
         "You are a helpful assistant. Use the provided tools when needed.",
         "", {}});
    conv.messages.push_back(
        {"user",
         "What is the current weather in Paris? Use the get_weather tool.", "",
         {}});
    ToolSpec t;
    t.name = "get_weather";
    t.description = "Get the current weather for a city";
    t.parameters_json =
        R"({"type":"object","properties":{"city":{"type":"string","description":"City name"}},"required":["city"]})";
    conv.tools.push_back(t);

    std::vector<int32_t> ids;
    REQUIRE(model.BuildConversationInputIds(model.Tokenizer(), conv, ids));
    REQUIRE(!ids.empty());

    ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = max_gen_tokens;
    auto r = model.Infer(req);
    spdlog::info("[tool-roundtrip] output='{}' gen_tokens={}", r.output_text,
                 r.gen_tokens.size());
    REQUIRE(r.ok);

    std::vector<ToolCall> calls;
    REQUIRE(model.ParseToolCalls(model.Tokenizer(), r.gen_tokens, calls));
    for (const auto &c : calls)
        spdlog::info("[tool-roundtrip] call name='{}' args={}", c.name,
                     c.arguments_json);
    if (calls.empty()) {
        std::string ids;
        for (size_t i = 0; i < r.gen_tokens.size() && i < 64; ++i) {
            if (i) ids += ",";
            ids += std::to_string(r.gen_tokens[i]);
        }
        spdlog::info("[tool-roundtrip] no calls; gen_tokens[:64]={}", ids);
    }
    return calls;
}
