// Beam (Harmony-XML) tool bridge (Model Ingestion Protocol step 4b): the
// renderer must reproduce the checkpoint template's wire structure, the parser
// must round-trip a native `to={name}` / `<rfl:parameter>` call, and the
// tokenizer must resolve the plain Harmony markers literally.
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "lsxcommon/beam_moe_tool_format.h"
#include "lsxcommon/converter.h"
#include "lsxcommon/tokenizer.h"

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

namespace fs = std::filesystem;

namespace {

struct Fixture {
    fs::path dir;
    lsxcommon::ArrowTokenizerDb db;
};

Fixture MakeTokenizer()
{
    Fixture f;
    f.dir = fs::temp_directory_path() / "lsx_beam_tool_tok";
    std::error_code ec;
    fs::remove_all(f.dir, ec);
    fs::create_directories(f.dir, ec);
    const fs::path tok = fs::path(LSX_SOURCE_DIR) / "tests" / "data" /
                         "synthetic_beam_tok.json";
    lsxcommon::TokenizerSpecialPolicy policy;
    policy.bos_id = -1;
    policy.pretokenizer = 4;
    policy.eos_id = 61;
    policy.pad_id = 63;
    auto st = lsxcommon::BuildTokenizerArrowFiles(tok.string(), f.dir, policy);
    REQUIRE(st.ok());
    auto vf = arrow::io::MemoryMappedFile::Open(
        (f.dir / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    auto mf = arrow::io::MemoryMappedFile::Open(
        (f.dir / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    REQUIRE(vf.ok());
    REQUIRE(mf.ok());
    auto meta = lsxcommon::load_tokenizer_metadata(*vf, *mf, f.db);
    REQUIRE(meta.ok());
    return f;
}

std::string Decode(const lsxcommon::ArrowTokenizerDb &db,
                   const std::vector<int32_t> &ids)
{
    std::vector<char> buf(ids.size() * 8 + 64);
    size_t cap = buf.size();
    const int32_t rc = lsxcommon::arrow_tokenizer_decode(
        db, ids.data(), ids.size(), buf.data(), &cap);
    REQUIRE(rc >= 0);
    return std::string(buf.data(), cap);
}

} // namespace

TEST_CASE("beam renders the Harmony system/developer/user framing",
          "[beam][tool]")
{
    Fixture f = MakeTokenizer();
    lsxcommon::ChatConversation conv;
    conv.messages.push_back({"system", "Be terse.", "", {}});
    conv.messages.push_back({"user", "Hello", "", {}});

    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::beam::RenderBeamConversation(f.db, conv, ids));
    INFO("ids=" << ids.size() << " vocab=" << f.db.vocab_size
                << " base=" << f.db.base_vocab_size
                << " kind=" << f.db.tokenizer_kind
                << " literals=" << f.db.literal_ids.size());
    const std::string text = Decode(f.db, ids);
    INFO("rendered: [" << text << "]");

    CHECK(text.find("<|start|>system<|message|>You are Beam, an AI assistant "
                    "trained by Reflection.") == 0);
    CHECK(text.find("Reasoning effort: 512") != std::string::npos);
    CHECK(text.find("# Valid channels: analysis, final.") != std::string::npos);
    CHECK(text.find("<|end|><|start|>developer<|message|># Instructions\n\nBe "
                    "terse.<|end|>") != std::string::npos);
    CHECK(text.find("<|start|>user<|message|>Hello<|end|>") != std::string::npos);
    CHECK(text.rfind("<|start|>assistant") == text.size() - 18);
}

TEST_CASE("beam renders a native to={name} rfl:parameter tool call",
          "[beam][tool]")
{
    Fixture f = MakeTokenizer();
    lsxcommon::ChatConversation conv;
    conv.messages.push_back({"user", "Weather in Paris?", "", {}});
    lsxcommon::ChatMessage a;
    a.role = "assistant";
    lsxcommon::ToolCall call;
    call.id = "call_1";
    call.name = "get_weather";
    call.arguments_json = "{\"city\":\"Paris\",\"days\":3}";
    a.tool_calls.push_back(call);
    conv.messages.push_back(a);

    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::beam::RenderBeamConversation(f.db, conv, ids));
    const std::string text = Decode(f.db, ids);
    CHECK(text.find("<|start|>assistant to=get_weather<|channel|>analysis "
                    "<|constrain|>xml<|message|><rfl:parameter name=\"city\">"
                    "\nParis\n</rfl:parameter>") != std::string::npos);
    CHECK(text.find("<rfl:parameter name=\"days\">\n3\n</rfl:parameter>") !=
          std::string::npos);
    CHECK(text.find("<|call|>") != std::string::npos);
}

TEST_CASE("beam parses the native tool call back to JSON arguments",
          "[beam][tool]")
{
    Fixture f = MakeTokenizer();
    const std::string wire =
        "<|start|>assistant<|channel|>analysis<|message|>thinking<|end|>"
        "<|start|>assistant to=get_weather<|channel|>analysis "
        "<|constrain|>xml<|message|><rfl:parameter name=\"city\">\nParis\n"
        "</rfl:parameter>\n<rfl:parameter name=\"days\">\n3\n"
        "</rfl:parameter><|call|>";
    std::vector<int32_t> ids(1024);
    size_t cap = ids.size();
    const int32_t n = lsxcommon::arrow_tokenizer_encode(
        f.db, wire.data(), wire.size(), ids.data(), &cap);
    REQUIRE(n > 0);
    ids.resize(static_cast<size_t>(n));

    std::vector<lsxcommon::ToolCall> calls;
    REQUIRE(lsxcommon::beam::ParseBeamToolCalls(f.db, ids, calls));
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json == "{\"city\":\"Paris\",\"days\":3}");

    // A plain answer is not a tool call.
    const std::string plain =
        "<|start|>assistant<|channel|>final<|message|>Paris<|return|>";
    cap = ids.size();
    const int32_t m = lsxcommon::arrow_tokenizer_encode(
        f.db, plain.data(), plain.size(), ids.data(), &cap);
    REQUIRE(m > 0);
    ids.resize(static_cast<size_t>(m));
    std::vector<lsxcommon::ToolCall> none;
    CHECK_FALSE(lsxcommon::beam::ParseBeamToolCalls(f.db, ids, none));
}

TEST_CASE("beam tool parameters that are not strings stay JSON", "[beam][tool]")
{
    Fixture f = MakeTokenizer();
    const std::string wire =
        "<|start|>assistant to=set_flags<|channel|>analysis "
        "<|constrain|>xml<|message|><rfl:parameter name=\"n\">\n42\n"
        "</rfl:parameter>\n<rfl:parameter name=\"ok\">\ntrue\n</rfl:parameter>"
        "\n<rfl:parameter name=\"label\">\nhello\n</rfl:parameter><|call|>";
    std::vector<int32_t> ids(1024);
    size_t cap = ids.size();
    const int32_t n = lsxcommon::arrow_tokenizer_encode(
        f.db, wire.data(), wire.size(), ids.data(), &cap);
    REQUIRE(n > 0);
    ids.resize(static_cast<size_t>(n));
    std::vector<lsxcommon::ToolCall> calls;
    REQUIRE(lsxcommon::beam::ParseBeamToolCalls(f.db, ids, calls));
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].arguments_json == "{\"n\":42,\"ok\":true,\"label\":\"hello\"}");
}

TEST_CASE("beam parses the real-weight native call token stream",
          "[beam][tool][real-tokenizer]")
{
    const fs::path real =
        "/mnt/fast_storage/huggingface/Beam-1.0-501B-A23B/tokenizer.json";
    if (!fs::exists(real)) SKIP("Beam source tokenizer.json absent");
    fs::path dir = fs::temp_directory_path() / "lsx_beam_tool_real";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    lsxcommon::TokenizerSpecialPolicy policy;
    policy.bos_id = -1;
    policy.pretokenizer = 4;
    policy.eos_id = 151674;
    policy.pad_id = 151643;
    REQUIRE(lsxcommon::BuildTokenizerArrowFiles(real.string(), dir, policy).ok());
    auto vf = arrow::io::MemoryMappedFile::Open(
        (dir / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    auto mf = arrow::io::MemoryMappedFile::Open(
        (dir / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    REQUIRE(vf.ok());
    REQUIRE(mf.ok());
    lsxcommon::ArrowTokenizerDb db;
    REQUIRE(lsxcommon::load_tokenizer_metadata(*vf, *mf, db).ok());

    const std::vector<int32_t> ids = {
        151672, 34484, 151671, 785, 1196, 6801, 311, 1414, 279, 1482, 9104,
        304, 12095, 13, 358, 1265, 990, 279, 633, 69364, 5392, 448, 3283,
        330, 59604, 3263, 151670, 151669, 77091, 311, 28280, 69364, 151672,
        34484, 220, 151673, 6455, 151671, 151679, 8926, 881, 59604, 151680,
        151675};
    std::vector<lsxcommon::ToolCall> calls;
    const bool ok = lsxcommon::beam::ParseBeamToolCalls(db, ids, calls);
    INFO("decoded: [" << Decode(db, ids) << "]");
    REQUIRE(ok);
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json == "{\"city\":\"Paris\"}");
}
