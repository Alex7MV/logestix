// Step-5-Preview native chat/tool framing (host-only; synthetic tokenizer, no
// real model). Pins the checkpoint's chat_template.jinja frame — the
// `<|im_start|>role` turns, the `<|im_start|>assistant\n<think>` generation
// prefix, the `<tools>` schema block and the
// `<tool_call>\n<function=…>\n<parameter=…>` XML codec.
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/converter.h>
#include <lsxcommon/step_moe_tool_format.h>
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
    out.pinned = fs::temp_directory_path() / "lsx_step_tool";
    fs::remove_all(out.pinned);
    fs::create_directories(out.pinned);
    lsxcommon::TokenizerSpecialPolicy policy;
    policy.bos_id = -1;
    policy.pretokenizer = 2;
    auto st = lsxcommon::BuildTokenizerArrowFiles(
        TOKENIZER_TEST_DATA_DIR "/synthetic_step_tok.json", out.pinned, policy);
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

std::string Decode(const ArrowTokenizerDb &db, const std::vector<int32_t> &ids)
{
    std::vector<char> buf(ids.size() * 16 + 256);
    size_t cap = buf.size();
    REQUIRE(lsxcommon::arrow_tokenizer_decode(db, ids.data(), ids.size(),
                                              buf.data(), &cap) == 0);
    return std::string(buf.data(), cap);
}

void AppendText(const ArrowTokenizerDb &db, const std::string &text,
                std::vector<int32_t> &ids)
{
    std::vector<int32_t> buf(text.size() * 4 + 64);
    size_t cap = buf.size();
    const int32_t n = lsxcommon::arrow_tokenizer_encode(
        db, text.data(), text.size(), buf.data(), &cap);
    REQUIRE(n >= 0);
    ids.insert(ids.end(), buf.begin(), buf.begin() + n);
}

} // namespace

TEST_CASE("Step-5 chat template markers are present in the tokenizer",
          "[step][tool]")
{
    auto f = BuildDb();
    for (const char *m : {"<|im_start|>", "<|im_end|>", "<think>", "</think>",
                          "<tool_call>", "</tool_call>", "<tool_response>",
                          "</tool_response>", "<｜begin▁of▁sentence｜>"}) {
        INFO(m);
        CHECK(IdOf(f.db, m) >= 0);
    }
    CHECK(lsxcommon::step::StepChatSentinelsPresent(f.db));
}

TEST_CASE("Step-5 conversation render emits the native frame", "[step][tool]")
{
    auto f = BuildDb();
    ChatConversation conv;
    conv.messages.push_back({"system", "be brief", "", {}});
    conv.messages.push_back({"user", "hello", "", {}});
    ToolSpec t;
    t.name = "get_weather";
    t.description = "weather";
    t.parameters_json = R"({"type":"object","properties":{"city":{"type":"string"}}})";
    conv.tools.push_back(t);

    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::step::RenderStepConversation(f.db, conv, ids));
    REQUIRE(!ids.empty());
    CHECK(ids.front() == IdOf(f.db, "<｜begin▁of▁sentence｜>"));
    // The turn markers are `special: true`, so the decoder drops them; the
    // frame is pinned through the ids plus the decoded free text.
    const int32_t im_start = IdOf(f.db, "<|im_start|>");
    const int32_t im_end = IdOf(f.db, "<|im_end|>");
    CHECK(std::count(ids.begin(), ids.end(), im_start) >= 3);  // system+user+gen
    CHECK(std::count(ids.begin(), ids.end(), im_end) >= 2);    // system+user
    const std::string text = Decode(f.db, ids);
    INFO(text);
    CHECK(text.find("system\nbe brief") != std::string::npos);
    CHECK(text.find("# Tools") != std::string::npos);
    CHECK(text.find("<tools>") != std::string::npos);
    CHECK(text.find("\"get_weather\"") != std::string::npos);
    CHECK(text.find("user\nhello") != std::string::npos);
    // The generation prompt always opens the assistant turn with <think>.
    CHECK(ids.back() == IdOf(f.db, "<think>"));
    CHECK(text.find("assistant\n<think>") != std::string::npos);
}

TEST_CASE("Step-5 tool call round-trips the native XML", "[step][tool]")
{
    auto f = BuildDb();
    std::vector<int32_t> ids;
    ids.push_back(IdOf(f.db, "<tool_call>"));
    AppendText(f.db, "\n<function=get_weather>\n<parameter=city>\nParis\n"
                     "</parameter>\n<parameter=days>\n2\n</parameter>\n"
                     "</function>\n",
                ids);
    ids.push_back(IdOf(f.db, "</tool_call>"));

    const std::string text = Decode(f.db, ids);
    INFO(text);
    const auto calls = lsxcommon::step::ParseStepToolCalls(f.db, ids);
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json.find("\"city\":\"Paris\"") != std::string::npos);
    // A JSON scalar keeps its type; `days` was written as the number 2.
    CHECK(calls[0].arguments_json.find("\"days\":2") != std::string::npos);
}

TEST_CASE("Step-5 render of an assistant tool call reproduces the template",
          "[step][tool]")
{
    auto f = BuildDb();
    ChatConversation conv;
    conv.messages.push_back({"user", "weather?", "", {}});
    ChatMessage asst;
    asst.role = "assistant";
    asst.content = "";
    ToolCall call;
    call.id = "call_0";
    call.name = "get_weather";
    call.arguments_json = R"({"city":"Paris"})";
    asst.tool_calls.push_back(call);
    conv.messages.push_back(asst);

    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::step::RenderStepConversation(f.db, conv, ids));
    const std::string text = Decode(f.db, ids);
    INFO(text);
    CHECK(text.find("assistant\n<think></think><tool_call>") !=
          std::string::npos);
    CHECK(text.find("<tool_call>\n<function=get_weather>\n<parameter=city>\n"
                    "Paris\n</parameter>\n</function>\n</tool_call>") !=
          std::string::npos);
}
