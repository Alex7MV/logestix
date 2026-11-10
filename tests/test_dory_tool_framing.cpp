// Dory1.0-4B ChatML + native XML tool framing (host-only; synthetic tokenizer,
// no CUDA, no real model). The expected transcripts are the checkpoint's own
// `chat_template.jinja` rendered by transformers 5.15.1 (captured 2026-11-10),
// so a drift in the C++ renderer fails here instead of silently changing what
// the model sees.
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <lsxcommon/converter.h>
#include <lsxcommon/dory_tool_format.h>
#include <lsxcommon/tokenizer.h>

#include "dory_dense_model.h"

namespace fs = std::filesystem;

using lsxcommon::ArrowTokenizerDb;
using lsxcommon::ChatConversation;
using lsxcommon::ChatMessage;
using lsxcommon::DoryDenseModel;
using lsxcommon::ToolCall;
using lsxcommon::ToolSpec;

namespace {

struct DbFiles {
    fs::path pinned;
    ArrowTokenizerDb db;
};

DbFiles BuildDb()
{
    DbFiles out;
    out.pinned = fs::temp_directory_path() / "lsx_dory_tool";
    fs::remove_all(out.pinned);
    fs::create_directories(out.pinned);
    REQUIRE(lsxcommon::BuildTokenizerArrowFiles(
                TOKENIZER_TEST_DATA_DIR "/synthetic_dory_tok.json", out.pinned)
                .ok());
    auto v = arrow::io::MemoryMappedFile::Open(
        (out.pinned / "tokenizer_vocab.arrow").string(),
        arrow::io::FileMode::READ);
    REQUIRE(v.ok());
    auto m = arrow::io::MemoryMappedFile::Open(
        (out.pinned / "tokenizer_merges.arrow").string(),
        arrow::io::FileMode::READ);
    REQUIRE(m.ok());
    REQUIRE(lsxcommon::load_tokenizer_metadata(*v, *m, out.db).ok());
    return out;
}

ToolSpec WeatherTool()
{
    ToolSpec t;
    t.name = "get_weather";
    t.description = "Get the current weather for a city";
    t.parameters_json =
        R"({"type":"object","properties":{"city":{"type":"string","description":"City name"}},"required":["city"]})";
    return t;
}

ChatConversation ToolConversation()
{
    ChatConversation conv;
    conv.messages.push_back({"system", "You are a helper.", "", {}});
    conv.messages.push_back({"user", "Weather in Paris?", "", {}});
    conv.tools.push_back(WeatherTool());
    return conv;
}

} // namespace

TEST_CASE("dory renderer reproduces the checkpoint chat template",
          "[dory][tool]")
{
    // The renderer's default policy keeps the think block OPEN (the model is
    // served with reasoning on). These exact-match cases pin the closed form,
    // which is what the FSI/CODE goldens are captured in.
    const lsxcommon::DoryRenderPolicy closed{.enable_thinking = false};

    SECTION("system + tools + user, thinking closed")
    {
        std::string text;
        REQUIRE(lsxcommon::RenderDoryConversation(ToolConversation(), text,
                                                 closed));
        const char *want =
            R"TXT(<|im_start|>system
You are a helper.

# Tools

You have access to the following functions:

<tools>
<function>
<name>get_weather</name>
<description>Get the current weather for a city</description>
<parameters>
<parameter>
<name>city</name>
<type>string</type>
<description>City name</description>
</parameter>
<required>["city"]</required>
</parameters>
</function>
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT><|im_end|>
<|im_start|>user
Weather in Paris?<|im_end|>
<|im_start|>assistant
<think></think>)TXT";
        CHECK(text == want);
    }

    SECTION("no system message still emits the empty system turn")
    {
        ChatConversation conv;
        conv.messages.push_back({"user", "hi", "", {}});
        std::string text;
        REQUIRE(lsxcommon::RenderDoryConversation(conv, text, closed));
        CHECK(text ==
              "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nhi"
              "<|im_end|>\n<|im_start|>assistant\n<think></think>");
    }

    SECTION("assistant tool call + tool response turn")
    {
        ChatConversation conv;
        conv.messages.push_back({"user", "Weather in Paris?", "", {}});
        ChatMessage assistant;
        assistant.role = "assistant";
        assistant.tool_calls.push_back(
            {"call_0", "get_weather", R"({"city":"Paris"})"});
        conv.messages.push_back(assistant);
        conv.messages.push_back({"tool", "18C sunny", "", {}});
        conv.messages.push_back({"user", "thanks", "", {}});
        std::string text;
        REQUIRE(lsxcommon::RenderDoryConversation(conv, text, closed));
        const char *want =
            R"TXT(<|im_start|>system
<|im_end|>
<|im_start|>user
Weather in Paris?<|im_end|>
<|im_start|>assistant
<think></think>
<tool_call>
<function=get_weather>
<parameter=city>
Paris
</parameter>
</function>
</tool_call>
<|im_end|>
<|im_start|>user
<tool_response>
18C sunny
</tool_response>
<|im_end|>
<|im_start|>user
thanks<|im_end|>
<|im_start|>assistant
<think></think>)TXT";
        CHECK(text == want);
    }

    SECTION("thinking mode leaves the think block open")
    {
        ChatConversation conv;
        conv.messages.push_back({"user", "hi", "", {}});
        std::string text;
        // The DEFAULT policy is the served mode (reasoning on): an open
        // <think> block the model fills itself.
        REQUIRE(lsxcommon::RenderDoryConversation(conv, text));
        const std::string want_tail = "<|im_start|>assistant\n<think>\n";
        REQUIRE(text.size() >= want_tail.size());
        CHECK(text.compare(text.size() - want_tail.size(), want_tail.size(),
                           want_tail) == 0);
    }
}

TEST_CASE("dory control markers survive only as their own vocab ids",
          "[dory][tool]")
{
    auto fx = BuildDb();
    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::AppendDoryChatIds(
        fx.db, "<|im_start|>system\nx<|im_end|>\n<think></think>", ids));
    REQUIRE(ids.size() >= 6);
    CHECK(ids.front() == lsxcommon::arrow_tokenizer_lookup(
                             fx.db, "<|im_start|>", 12));
    // The `special: false` markers survive id injection AND re-encoding of the
    // same text yields the same id (they are added tokens, not BPE merges).
    const int32_t think_id =
        lsxcommon::arrow_tokenizer_lookup(fx.db, "<think>", 7);
    REQUIRE(think_id >= 0);
    CHECK(std::count(ids.begin(), ids.end(), think_id) == 1);
    fs::remove_all(fx.pinned);
}

TEST_CASE("dory parses a native XML tool call out of the token stream",
          "[dory][tool]")
{
    auto fx = BuildDb();
    const char *gen_text =
        "<think></think>\n<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::AppendDoryChatIds(fx.db, gen_text, ids));

    // No Load(): the token bridges are pure host code.
    DoryDenseModel model{lsxcommon::ModelInitConfig{}};
    std::vector<ToolCall> calls;
    REQUIRE(model.ParseToolCalls(fx.db, ids, calls));
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json == R"({"city":"Paris"})");

    // A marker-free stream parses to no calls (and does not fail).
    std::vector<int32_t> plain;
    REQUIRE(lsxcommon::AppendDoryChatIds(fx.db, "just an answer", plain));
    calls.clear();
    REQUIRE(model.ParseToolCalls(fx.db, plain, calls));
    CHECK(calls.empty());
    fs::remove_all(fx.pinned);
}

TEST_CASE("dory conversation ids start with the system turn sentinel",
          "[dory][tool]")
{
    auto fx = BuildDb();
    // No Load(): the token bridges are pure host code.
    DoryDenseModel model{lsxcommon::ModelInitConfig{}};
    std::vector<int32_t> ids;
    REQUIRE(model.BuildConversationInputIds(fx.db, ToolConversation(), ids));
    REQUIRE(ids.size() > 8);
    const int32_t im_start =
        lsxcommon::arrow_tokenizer_lookup(fx.db, "<|im_start|>", 12);
    const int32_t im_end =
        lsxcommon::arrow_tokenizer_lookup(fx.db, "<|im_end|>", 10);
    CHECK(ids[0] == im_start);
    CHECK(std::count(ids.begin(), ids.end(), im_start) >= 3);
    CHECK(std::count(ids.begin(), ids.end(), im_end) >= 2);
    fs::remove_all(fx.pinned);
}
