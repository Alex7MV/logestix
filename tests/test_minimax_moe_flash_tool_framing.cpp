// MiniMax-M3.1-Flash native chat/tool framing (host-only, synthetic
// tokenizer). Pins the M3.1 chat_template.jinja deltas against M3:
//   - system prompt names MiniMax-M3.1 and appends <model_id>8bb8149fe</model_id>
//   - thinking is fixed to ADAPTIVE (no think-END generation prefix)
//   - mid-conversation system/developer turns render as `developer` blocks
//   - consecutive tool results share ONE `]~b]tool` block
// and that the model class actually routes through that policy.
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/converter.h>
#include "minimax_moe_flash_model.h"
#include <lsxcommon/minimax_moe_tool_format.h>
#include <lsxcommon/tokenizer.h>

namespace fs = std::filesystem;

using lsxcommon::ArrowTokenizerDb;
using lsxcommon::ChatConversation;
using lsxcommon::ChatMessage;
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
    out.pinned = fs::temp_directory_path() / "lsx_minimax_flash_tool";
    fs::remove_all(out.pinned);
    fs::create_directories(out.pinned);
    auto st = lsxcommon::BuildTokenizerArrowFiles(
        TOKENIZER_TEST_DATA_DIR "/synthetic_minimax_tok.json", out.pinned);
    if (!st.ok()) WARN(st.ToString());
    REQUIRE(st.ok());
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

int32_t IdOf(const ArrowTokenizerDb &db, const char *text)
{
    return lsxcommon::arrow_tokenizer_lookup(db, text, std::strlen(text));
}

size_t CountId(const std::vector<int32_t> &ids, int32_t id)
{
    size_t n = 0;
    for (int32_t v : ids)
        if (v == id) ++n;
    return n;
}

ChatConversation ToolConversation()
{
    ChatConversation conv;
    conv.messages.push_back({"system", "You are a helper.", "", {}});
    conv.messages.push_back({"user", "Weather in Paris?", "", {}});
    ToolSpec t;
    t.name = "get_weather";
    t.description = "Get the weather";
    t.parameters_json = "{\"type\":\"object\"}";
    conv.tools.push_back(t);
    return conv;
}

} // namespace

TEST_CASE("M3.1 policy renders its own system prompt and adaptive mode",
          "[minimax-flash][tool]")
{
    auto f = BuildDb();
    ChatConversation conv;
    conv.messages.push_back({"user", "Hi", "", {}});
    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::minimax::RenderMinimaxConversation(
        f.db, conv, lsxcommon::minimax::MinimaxFlashRenderPolicy(), ids));
    REQUIRE(!ids.empty());
    // The system prompt (default text + model id) is BPE text, not markers:
    // assert it on the rendered string source instead (the policy).
    const auto policy = lsxcommon::minimax::MinimaxFlashRenderPolicy();
    REQUIRE(policy.default_system != nullptr);
    CHECK(std::string(policy.default_system).find("MiniMax-M3.1") !=
          std::string::npos);
    CHECK(std::string(policy.system_suffix).find("8bb8149fe") !=
          std::string::npos);
    CHECK(std::string(policy.thinking_mode_line).find("adaptive") !=
          std::string::npos);
    // Fixed-adaptive: the generation prompt carries NO think-END prefix, so
    // the M3.1 stream has exactly one fewer than M3 (whose prompt adds it).
    std::vector<int32_t> m3_ids;
    REQUIRE(lsxcommon::minimax::RenderMinimaxConversation(f.db, conv, m3_ids));
    CHECK(ids.back() != IdOf(f.db, "</mm:think>"));
    CHECK(m3_ids.back() == IdOf(f.db, "</mm:think>"));
    CHECK(CountId(ids, IdOf(f.db, "</mm:think>")) + 1 ==
          CountId(m3_ids, IdOf(f.db, "</mm:think>")));
    CHECK(CountId(ids, IdOf(f.db, "]~b]")) >= 3);  // system, developer, ai
}

TEST_CASE("M3.1 policy batches tool results and renders developer blocks",
          "[minimax-flash][tool]")
{
    auto f = BuildDb();
    ChatConversation conv = ToolConversation();
    ChatMessage a;
    a.role = "assistant";
    a.content = "On it.";
    a.tool_calls.push_back({"", "get_weather", "{\"city\":\"Paris\"}"});
    conv.messages.push_back(a);
    conv.messages.push_back({"tool", "sunny", "call_0", {}});
    conv.messages.push_back({"tool", "22C", "call_0", {}});
    conv.messages.push_back({"system", "Late instruction.", "", {}});
    conv.messages.push_back({"user", "Thanks", "", {}});

    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::minimax::RenderMinimaxConversation(
        f.db, conv, lsxcommon::minimax::MinimaxFlashRenderPolicy(), ids));
    // One `]~b]` per rendered turn: system, developer, user, ai, assistant,
    // ONE tool block (two results batched), late developer, user = 8.
    // The M3 policy renders each tool message separately (9).
    std::vector<int32_t> m3_ids;
    REQUIRE(lsxcommon::minimax::RenderMinimaxConversation(f.db, conv, m3_ids));
    CHECK(CountId(ids, IdOf(f.db, "]~b]")) + 1 ==
          CountId(m3_ids, IdOf(f.db, "]~b]")));
    CHECK(std::count(ids.begin(), ids.end(), IdOf(f.db, "<tool_call>")) >= 2);
    // The assistant turn contributes one think-END prefix; the generation
    // prompt contributes none (M3 adds one more).
    CHECK(CountId(ids, IdOf(f.db, "</mm:think>")) + 1 ==
          CountId(m3_ids, IdOf(f.db, "</mm:think>")));
}

TEST_CASE("M3.1 model routes conversation framing through its policy",
          "[minimax-flash][tool]")
{
    auto f = BuildDb();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = "/nonexistent";
    lsxcommon::MiniMaxM31FlashModel model(cfg);

    ChatConversation conv;
    conv.messages.push_back({"user", "Hello", "", {}});
    std::vector<int32_t> via_model, via_policy;
    REQUIRE(model.BuildConversationInputIds(f.db, conv, via_model));
    REQUIRE(lsxcommon::minimax::RenderMinimaxConversation(
        f.db, conv, lsxcommon::minimax::MinimaxFlashRenderPolicy(), via_policy));
    CHECK(via_model == via_policy);
}

TEST_CASE("M3.1 parse extracts a native tool call from a token stream",
          "[minimax-flash][tool]")
{
    auto f = BuildDb();
    std::vector<int32_t> gen;
    auto push_text = [&](const std::string &t) {
        std::vector<int32_t> buf(512, 0);
        size_t cap = buf.size();
        const int32_t n = lsxcommon::arrow_tokenizer_encode(
            f.db, t.data(), t.size(), buf.data(), &cap);
        REQUIRE(n > 0);
        gen.insert(gen.end(), buf.begin(), buf.begin() + n);
    };
    push_text("Calling. ");
    gen.push_back(IdOf(f.db, "<tool_call>"));
    push_text("\n");
    push_text("]<]minimax[>[<invoke name=\"get_weather\">");
    push_text("]<]minimax[>[<city>Paris]<]minimax[>[</city>");
    push_text("]<]minimax[>[</invoke>");
    gen.push_back(IdOf(f.db, "</tool_call>"));

    std::vector<ToolCall> calls;
    REQUIRE(lsxcommon::minimax::ParseMinimaxToolCalls(f.db, gen, calls));
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json == "{\"city\":\"Paris\"}");
}
