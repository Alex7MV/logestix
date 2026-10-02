// prompt_bridge — the model-path framing bridge (host-only; no CUDA, no
// weights, no vault).
//
// FramePromptForModel / FrameConversationForModel / ParseToolCallsForModel
// create a throwaway IModel through the factory and delegate to the family's
// own BuildPromptInputIds / BuildConversationInputIds / ParseToolCalls. The
// model is never Load()ed, so an index-only fixture drives the whole path.
//
// Before this suite the bridge had no test at all: it was reached only from
// taskmosaic's LsxPromptFramer against a real model in the vault.
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/chat_conversation.h>
#include <lsxcommon/prompt_bridge.h>
#include <lsxcommon/tokenizer.h>
#include <lsxcommon/tokenizer_arrows.h>

#include "test_lightweight_model_fixture.h"

namespace fs = std::filesystem;
using lsxcommon::ArrowTokenizerDb;
using lsxcommon::ChatConversation;

namespace {

// Index + tokenizer arrows, tokenizer already loaded.
struct Bridge {
    synthmodel::LightweightModel m;
    ArrowTokenizerDb db;
};

Bridge MakeBridge(const char *dir)
{
    Bridge b;
    b.m = synthmodel::MakeLightweightModel(
        TOKENIZER_TEST_DATA_DIR "/synthetic_qwen_tok.json", "Qwen3.5-9B", dir);
    if (b.m.db.error_msg.empty())
        REQUIRE(lsxcommon::LoadTokenizerArrows(b.m.path(), b.db));
    return b;
}

} // namespace

TEST_CASE("prompt bridge: frames a raw prompt into ids", "[prompt_bridge]")
{
    Bridge b = MakeBridge("lsx_bridge_prompt");
    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::FramePromptForModel(b.m.path(), "hello", ids));
    REQUIRE(!ids.empty());
    // The bridge must reproduce the family's own BuildPromptInputIds. Qwen does
    // not override it (IModel::model.h falls back to prompt::EncodeRawPrompt),
    // so the frame is the raw BOS-led encoding — which is exactly the contract
    // worth pinning, since a bridge that framed nothing would look identical
    // to a bridge that framed correctly here.
    const auto raw = synthmodel::RawEncode(b.db, "hello");
    REQUIRE(!raw.empty());
    CHECK(ids == raw);
    CHECK(ids.front() == b.db.bos_id);
}

TEST_CASE("prompt bridge: framing is deterministic across calls",
          "[prompt_bridge]")
{
    Bridge b = MakeBridge("lsx_bridge_determinism");
    std::vector<int32_t> a, c;
    REQUIRE(lsxcommon::FramePromptForModel(b.m.path(), "hi there", a));
    REQUIRE(lsxcommon::FramePromptForModel(b.m.path(), "hi there", c));
    CHECK(a == c);
}

TEST_CASE("prompt bridge: frames a multi-turn conversation", "[prompt_bridge]")
{
    Bridge b = MakeBridge("lsx_bridge_conv");
    ChatConversation conv;
    conv.messages.push_back({"system", "Be brief.", "", {}});
    conv.messages.push_back({"user", "hi", "", {}});

    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::FrameConversationForModel(b.m.path(), b.db, conv, ids));
    CHECK(!ids.empty());
    // A system + user turn must be longer than a single raw user prompt.
    std::vector<int32_t> prompt_ids;
    REQUIRE(lsxcommon::FramePromptForModel(b.m.path(), "hi", prompt_ids));
    CHECK(ids.size() > prompt_ids.size());
}

TEST_CASE("prompt bridge: carries a tool schema through the conversation bridge",
          "[prompt_bridge]")
{
    Bridge b = MakeBridge("lsx_bridge_tools");
    ChatConversation conv;
    conv.messages.push_back({"user", "Weather in Paris?", "", {}});
    lsxcommon::ToolSpec t;
    t.name = "get_weather";
    t.description = "Get the weather";
    t.parameters_json = R"({"type":"object","properties":{}})";
    conv.tools.push_back(t);

    std::vector<int32_t> without, with;
    ChatConversation bare;
    bare.messages.push_back({"user", "Weather in Paris?", "", {}});
    REQUIRE(lsxcommon::FrameConversationForModel(b.m.path(), b.db, bare, without));
    REQUIRE(lsxcommon::FrameConversationForModel(b.m.path(), b.db, conv, with));
    // The tool block adds ids; if it did not, the schema never reached the
    // model and the model could not call anything.
    CHECK(with.size() > without.size());
}

TEST_CASE("prompt bridge: parses a tool call from generated tokens",
          "[prompt_bridge]")
{
    Bridge b = MakeBridge("lsx_bridge_parse");
    const std::string json =
        R"({"name":"get_weather","arguments":{"city":"Paris"}})";
    const auto gen = synthmodel::RawEncode(b.db, json);

    std::vector<lsxcommon::ToolCall> calls;
    REQUIRE(lsxcommon::ParseToolCallsForModel(b.m.path(), b.db, gen, calls));
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json.find("Paris") != std::string::npos);
}

TEST_CASE("prompt bridge: an undetectable model path fails cleanly",
          "[prompt_bridge]")
{
    // No index -> the factory cannot pick a type -> the bridge must return
    // false, not throw and not read a partial frame.
    const fs::path absent = fs::temp_directory_path() / "lsx_bridge_absent";
    std::error_code ec;
    fs::remove_all(absent, ec);

    std::vector<int32_t> ids;
    CHECK_FALSE(lsxcommon::FramePromptForModel(absent.string(), "hi", ids));
    CHECK(ids.empty());
}

TEST_CASE("prompt bridge: a model dir without tokenizer arrows fails framing",
          "[prompt_bridge]")
{
    // The index is present (so the model is created) but LoadTokenizerArrows
    // cannot succeed: FramePromptForModel must report failure, not return an
    // empty frame that a caller would mistake for a valid prompt.
    const fs::path base = fs::temp_directory_path() / "lsx_bridge_no_tok";
    std::error_code ec;
    fs::remove_all(base, ec);
    synthmodel::WriteIndex(base, "Qwen3.5-9B");

    std::vector<int32_t> ids;
    CHECK_FALSE(lsxcommon::FramePromptForModel(base.string(), "hi", ids));
    CHECK(ids.empty());

    fs::remove_all(base, ec);
}

TEST_CASE("prompt bridge: parsing an empty generation yields no calls",
          "[prompt_bridge]")
{
    Bridge b = MakeBridge("lsx_bridge_empty");
    std::vector<lsxcommon::ToolCall> calls;
    const std::vector<int32_t> gen;
    REQUIRE(lsxcommon::ParseToolCallsForModel(b.m.path(), b.db, gen, calls));
    CHECK(calls.empty());
}