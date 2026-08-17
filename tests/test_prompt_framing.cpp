// TDD — GLM-5.2 prompt framing helper (host-only, no CUDA, no real model).
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/converter.h>
#include <lsxcommon/prompt_framing.h>
#include <lsxcommon/tokenizer.h>

namespace fs = std::filesystem;

namespace {

struct DbFiles {
    fs::path pinned;
    lsxcommon::ArrowTokenizerDb db;
};

DbFiles BuildDb(const std::string &fixture_name)
{
    DbFiles out;
    out.pinned = fs::temp_directory_path() / ("logestix_pf_" + fixture_name);
    fs::remove_all(out.pinned);
    fs::create_directories(out.pinned);
    REQUIRE(lsxcommon::BuildTokenizerArrowFiles(
                TOKENIZER_TEST_DATA_DIR "/" + fixture_name, out.pinned)
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

std::vector<int32_t> RawEncode(const lsxcommon::ArrowTokenizerDb &db,
                               const char *text)
{
    std::vector<int32_t> ids(256, 0);
    size_t cap = ids.size();
    const int32_t n = lsxcommon::arrow_tokenizer_encode(
        db, text, std::strlen(text), ids.data(), &cap);
    REQUIRE(n > 0);
    ids.resize(static_cast<size_t>(n));
    return ids;
}

} // namespace

TEST_CASE("prompt_framing: GLM chat-template framing order", "[prompt_framing]")
{
    DbFiles f = BuildDb("synthetic_glm_tok.json");
    const auto &db = f.db;
    REQUIRE(db.vocab_size == 274);

    const int32_t gmask = lsxcommon::arrow_tokenizer_lookup(db, "[gMASK]", 7);
    const int32_t sop   = lsxcommon::arrow_tokenizer_lookup(db, "<sop>", 5);
    const int32_t usr   = lsxcommon::arrow_tokenizer_lookup(db, "<|user|>", 8);
    const int32_t asst  = lsxcommon::arrow_tokenizer_lookup(db, "<|assistant|>", 13);
    const int32_t think = lsxcommon::arrow_tokenizer_lookup(db, " thinking", 7);
    const int32_t ethink = lsxcommon::arrow_tokenizer_lookup(db, " response", 8);
    REQUIRE(gmask >= 0 && sop >= 0 && usr >= 0 && asst >= 0 &&
            think >= 0 && ethink >= 0);

    const char *prompt = "The capital of France is";
    const std::vector<int32_t> raw = RawEncode(db, prompt);

    std::vector<int32_t> expected = {gmask, sop, usr};
    expected.insert(expected.end(), raw.begin(), raw.end());
    expected.push_back(asst);
    expected.push_back(think);
    expected.push_back(ethink);

    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::prompt::BuildPromptInputIds("glm-5.2", db, prompt, ids));
    REQUIRE(ids == expected);
    REQUIRE(ids.size() == expected.size());

    f.db = lsxcommon::ArrowTokenizerDb{};
    fs::remove_all(f.pinned);
}

TEST_CASE("prompt_framing: non-GLM model type is encoded raw", "[prompt_framing]")
{
    DbFiles f = BuildDb("synthetic_tok.json");
    const char *prompt = "hello world";
    const std::vector<int32_t> expected = RawEncode(f.db, prompt);

    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::prompt::BuildPromptInputIds(
        "deepseek-v4-flash", f.db, prompt, ids));
    REQUIRE(ids == expected);

    f.db = lsxcommon::ArrowTokenizerDb{};
    fs::remove_all(f.pinned);
}

TEST_CASE("prompt_framing: GLM falls back to raw when specials missing", "[prompt_framing]")
{
    DbFiles f = BuildDb("synthetic_tok.json");
    const char *prompt = "hello world";
    const std::vector<int32_t> expected = RawEncode(f.db, prompt);

    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::prompt::BuildPromptInputIds("glm-5.2", f.db, prompt, ids));
    REQUIRE(ids == expected);

    f.db = lsxcommon::ArrowTokenizerDb{};
    fs::remove_all(f.pinned);
}

TEST_CASE("prompt_framing: empty prompt is rejected", "[prompt_framing]")
{
    DbFiles f = BuildDb("synthetic_tok.json");
    std::vector<int32_t> ids(4, 0);
    REQUIRE_FALSE(lsxcommon::prompt::BuildPromptInputIds("glm-5.2", f.db, "", ids));
    f.db = lsxcommon::ArrowTokenizerDb{};
    fs::remove_all(f.pinned);
}

TEST_CASE("prompt_framing: long prompt grows past the initial capacity", "[prompt_framing]")
{
    DbFiles f = BuildDb("synthetic_glm_tok.json");
    std::string long_text;
    for (int i = 0; i < 1200; ++i) long_text += "ab ";

    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::prompt::BuildPromptInputIds(
        "glm-5.2", f.db, long_text, ids));
    REQUIRE(ids.size() > 256);
    REQUIRE(ids[0] == lsxcommon::arrow_tokenizer_lookup(f.db, "[gMASK]", 7));
    REQUIRE(ids[1] == lsxcommon::arrow_tokenizer_lookup(f.db, "<sop>", 5));
    REQUIRE(ids[2] == lsxcommon::arrow_tokenizer_lookup(f.db, "<|user|>", 8));

    f.db = lsxcommon::ArrowTokenizerDb{};
    fs::remove_all(f.pinned);
}