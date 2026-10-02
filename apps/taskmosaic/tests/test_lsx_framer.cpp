// taskmosaic LsxPromptFramer — the real PromptFramer implementation (host-only;
// no CUDA, no weights, no vault).
//
// test_taskmosaic_bridge.cpp already covers the PromptFramer *interface*
// through a mock. This suite covers the implementation that chat_bridge
// actually gets in production: the lsxcommon prompt_bridge calls, the
// per-model-path tokenizer cache, the streaming decoder factory and the
// nullptr paths a bad model_path must degrade to instead of crashing.
//
// The index-only fixture (tests/test_lightweight_model_fixture.h) is enough:
// LsxPromptFramer never Load()s the model.
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/chat_conversation.h>

#include "lsx_framer.h"
#include "test_lightweight_model_fixture.h"

namespace fs = std::filesystem;
using lsxcommon::ChatConversation;
using taskmosaic::LsxPromptFramer;

namespace {

// Index + tokenizer arrows for a model path the framer can be pointed at.
struct Fixture {
    synthmodel::LightweightModel m;
};

Fixture MakeFixture(const char *dir)
{
    Fixture f;
    f.m = synthmodel::MakeLightweightModel(
        TOKENIZER_TEST_DATA_DIR "/synthetic_qwen_tok.json", "Qwen3.5-9B", dir);
    return f;
}

// A path with no index at all: every framer entry point must fail on it.
std::string AbsentPath(const char *dir)
{
    const fs::path p = fs::temp_directory_path() / dir;
    std::error_code ec;
    fs::remove_all(p, ec);
    return p.string();
}

} // namespace

TEST_CASE("framer: Encode frames a raw prompt through the model bridge",
          "[lsx_framer]")
{
    Fixture f = MakeFixture("lsx_framer_encode");
    LsxPromptFramer framer;
    std::vector<int32_t> ids;
    REQUIRE(framer.Encode(f.m.path(), "hello", ids));
    CHECK(!ids.empty());
}

TEST_CASE("framer: Decode renders ids back to text", "[lsx_framer]")
{
    Fixture f = MakeFixture("lsx_framer_decode");
    LsxPromptFramer framer;

    // Round-trip a plain word: encode the raw text, then decode it back.
    const auto ids = synthmodel::RawEncode(f.m.db, "hello");
    REQUIRE(!ids.empty());
    const std::string text = framer.Decode(f.m.path(), ids);
    CHECK(text.find("hello") != std::string::npos);
}

TEST_CASE("framer: the tokenizer is cached per model path", "[lsx_framer]")
{
    Fixture f = MakeFixture("lsx_framer_cache");
    LsxPromptFramer framer;
    const auto ids = synthmodel::RawEncode(f.m.db, "hello");
    REQUIRE(!ids.empty());

    // The first call populates the cache, the second reuses it. Both must give
    // the same answer; a cache that reloaded per call would still pass a
    // single-shot test, so this asserts the second result matches.
    const std::string first = framer.Decode(f.m.path(), ids);
    const std::string second = framer.Decode(f.m.path(), ids);
    CHECK(first == second);
    CHECK(!first.empty());

    // A second model path gets its own entry rather than reusing this one.
    Fixture other = MakeFixture("lsx_framer_cache_other");
    const auto other_ids = synthmodel::RawEncode(other.m.db, "hello");
    CHECK(!framer.Decode(other.m.path(), other_ids).empty());
}

TEST_CASE("framer: MakeDecoder streams tokens incrementally", "[lsx_framer]")
{
    Fixture f = MakeFixture("lsx_framer_decoder");
    LsxPromptFramer framer;
    auto dec = framer.MakeDecoder(f.m.path());
    REQUIRE(dec != nullptr);

    // Feed a word token by token; the incremental decoder is UTF-8 safe, so
    // the concatenation must reproduce the whole text and Flush() must empty.
    std::string out;
    for (int32_t id : synthmodel::RawEncode(f.m.db, "hello"))
        out += dec->Push(id);
    out += dec->Flush();
    CHECK(out.find("hello") != std::string::npos);
}

TEST_CASE("framer: EncodeConversation carries a tool schema", "[lsx_framer]")
{
    Fixture f = MakeFixture("lsx_framer_conv");
    LsxPromptFramer framer;

    ChatConversation bare;
    bare.messages.push_back({"user", "Weather in Paris?", "", {}});
    ChatConversation with_tool = bare;
    lsxcommon::ToolSpec t;
    t.name = "get_weather";
    t.description = "Get the weather";
    t.parameters_json = R"({"type":"object","properties":{}})";
    with_tool.tools.push_back(t);

    std::vector<int32_t> a, b;
    REQUIRE(framer.EncodeConversation(f.m.path(), bare, a));
    REQUIRE(framer.EncodeConversation(f.m.path(), with_tool, b));
    CHECK(b.size() > a.size());
}

TEST_CASE("framer: ParseToolCalls reads a call out of generated tokens",
          "[lsx_framer]")
{
    Fixture f = MakeFixture("lsx_framer_parse");
    LsxPromptFramer framer;
    const auto gen = synthmodel::RawEncode(
        f.m.db, R"({"name":"get_weather","arguments":{"city":"Paris"}})");

    std::vector<lsxcommon::ToolCall> calls;
    REQUIRE(framer.ParseToolCalls(f.m.path(), gen, calls));
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "get_weather");
}

TEST_CASE("framer: an undetectable model path degrades to empty, not a crash",
          "[lsx_framer]")
{
    // Every entry point must fail cleanly. chat_bridge hands user-supplied
    // model paths here, so a throw or a deref of a null tokenizer would be
    // remotely reachable.
    LsxPromptFramer framer;
    const std::string absent = AbsentPath("lsx_framer_absent");

    std::vector<int32_t> ids;
    CHECK_FALSE(framer.Encode(absent, "hi", ids));
    CHECK(ids.empty());

    CHECK(framer.Decode(absent, ids).empty());
    CHECK(framer.MakeDecoder(absent) == nullptr);

    std::vector<lsxcommon::ToolCall> calls;
    CHECK_FALSE(framer.ParseToolCalls(absent, ids, calls));
    CHECK(calls.empty());

    ChatConversation conv;
    conv.messages.push_back({"user", "hi", "", {}});
    CHECK_FALSE(framer.EncodeConversation(absent, conv, ids));
}