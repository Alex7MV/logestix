// AliceAI native chat/tool codec test (host-only). Uses the real SentencePiece
// tokenizer.model from the checkpoint (LSX_TEST_ALICE_ROOT if set, else the
// vault); SKIPs when absent.
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/alice_tool_format.h>
#include <lsxcommon/converter.h>
#include <lsxcommon/path_registry.h>
#include <lsxcommon/tokenizer.h>

namespace fs = std::filesystem;
using lsxcommon::ArrowTokenizerDb;
using lsxcommon::ChatConversation;
using lsxcommon::ChatMessage;
using lsxcommon::ToolCall;
using lsxcommon::ToolSpec;

namespace {

// Same resolution as the sibling inference gates (selfcheck/fsi/code/det): the
// env var wins, otherwise the checkpoint is looked up in the vault. Resolving
// only from the env made this gate SKIP silently after conversion.
fs::path AliceTokenModelDir()
{
    if (const char *env = std::getenv("LSX_TEST_ALICE_ROOT"))
        return fs::path(env);
    return fs::path(lsxcommon::lsx_model_models_dir()) /
           "AliceAI-Foundation-80B-A3B-Base";
}

bool BuildDb(ArrowTokenizerDb &db, fs::path &out)
{
    const fs::path spm = AliceTokenModelDir() / "tokenizer.model";
    if (!fs::exists(spm)) return false;
    out = fs::temp_directory_path() / "alice_tool_tok";
    fs::remove_all(out);
    fs::create_directories(out);
    if (!lsxcommon::BuildTokenizerArrowFilesSentencePiece(spm.string(),
                                                          out.string())
             .ok())
        return false;
    auto v = arrow::io::MemoryMappedFile::Open(
        (out / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    auto m = arrow::io::MemoryMappedFile::Open(
        (out / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    if (!v.ok() || !m.ok()) return false;
    return lsxcommon::load_tokenizer_metadata(*v, *m, db).ok();
}

std::string Decode(const ArrowTokenizerDb &db,
                   const std::vector<int32_t> &ids)
{
    std::size_t cap = ids.size() * 16 + 256;
    std::string s(cap, '\0');
    REQUIRE(lsxcommon::arrow_tokenizer_decode(db, ids.data(), ids.size(),
                                              s.data(), &cap) == 0);
    s.resize(cap);
    return s;
}

} // namespace

TEST_CASE("alice native tool frame renders the template markers", "[alice][fast]")
{
    ArrowTokenizerDb db;
    fs::path out;
    if (!BuildDb(db, out)) SKIP("alice tokenizer.model not present");

    ChatConversation conv;
    conv.messages.push_back({"system", "You are helpful.", "", {}});
    conv.messages.push_back({"user", "Weather in Paris?", "", {}});
    ToolSpec t;
    t.name = "get_weather";
    t.description = "Get the weather";
    t.parameters_json =
        R"({"type":"object","properties":{"city":{"type":"string"}}})";
    conv.tools.push_back(t);

    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::alice::RenderConversation(db, conv, true, ids));
    const std::string text = Decode(db, ids);
    INFO(text);
    REQUIRE(text.find("user:") != std::string::npos);
    REQUIRE(text.find("assistant:") != std::string::npos);
    REQUIRE(text.find("get_weather") != std::string::npos);
    // The generation prompt ends with the assistant role marker.
    const auto tail = text.find_last_not_of(" \t\n\r");
    REQUIRE(tail != std::string::npos);
    REQUIRE(text.rfind("assistant:", tail) != std::string::npos);

    // Parse a native tool call back.
    std::vector<int32_t> call_ids;
    {
        std::vector<int32_t> tmp(256, 0);
        size_t cap = tmp.size();
        const std::string call =
            "[TOOL_CALL_START]get_weather\n{\"city\":\"Paris\"}";
        const int32_t n = lsxcommon::arrow_tokenizer_encode(
            db, call.data(), call.size(), tmp.data(), &cap);
        REQUIRE(n > 0);
        tmp.resize(static_cast<size_t>(n));
        call_ids = std::move(tmp);
    }
    std::vector<ToolCall> calls;
    REQUIRE(lsxcommon::alice::ParseToolCallsAlice(db, call_ids, calls));
    REQUIRE(calls.size() == 1);
    REQUIRE(calls[0].name == "get_weather");
    REQUIRE(calls[0].arguments_json.find("Paris") != std::string::npos);
}
