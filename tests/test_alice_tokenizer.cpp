// Model-neutral SentencePiece-BPE tokenizer backend parity (AliceAI).
// Builds the arrow artifacts from the checkpoint's tokenizer.model and checks
// them against goldens captured from the real sentencepiece processor.
// The checkpoint is located via LSX_TEST_ALICE_ROOT (if set), else the vault;
// SKIPs when the model or its tokenizer.model is absent.
#include <catch2/catch_test_macros.hpp>
#include <arrow/io/file.h>
#include <lsxcommon/converter.h>
#include <lsxcommon/path_registry.h>
#include <lsxcommon/tokenizer.h>
#include <simdjson.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

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

bool LoadAliceDb(lsxcommon::ArrowTokenizerDb &db, const fs::path &out)
{
    auto vmm = arrow::io::MemoryMappedFile::Open(
                   (out / "tokenizer_vocab.arrow").string(),
                   arrow::io::FileMode::READ)
                   .ValueOrDie();
    auto mmm = arrow::io::MemoryMappedFile::Open(
                   (out / "tokenizer_merges.arrow").string(),
                   arrow::io::FileMode::READ)
                   .ValueOrDie();
    return lsxcommon::load_tokenizer_metadata(vmm, mmm, db).ok();
}

std::vector<int32_t> Encode(const lsxcommon::ArrowTokenizerDb &db,
                            const std::string &text)
{
    std::vector<int32_t> out(text.size() * 4 + 16);
    size_t cap = out.size();
    int32_t n = lsxcommon::arrow_tokenizer_encode(db, text.data(), text.size(),
                                                  out.data(), &cap);
    REQUIRE(n >= 0);
    out.resize(static_cast<size_t>(n));
    return out;
}

std::string Decode(const lsxcommon::ArrowTokenizerDb &db,
                   const std::vector<int32_t> &ids)
{
    std::vector<char> out(ids.size() * 8 + 16);
    size_t cap = out.size();
    int32_t rc = lsxcommon::arrow_tokenizer_decode(db, ids.data(), ids.size(),
                                                   out.data(), &cap);
    if (rc == -2) {
        out.resize(cap);
        rc = lsxcommon::arrow_tokenizer_decode(db, ids.data(), ids.size(),
                                               out.data(), &cap);
    }
    REQUIRE(rc == 0);
    return std::string(out.data(), cap);
}

} // namespace

TEST_CASE("sp-bpe builder parses the real tokenizer.model", "[alice][fast]")
{
    const fs::path src = AliceTokenModelDir() / "tokenizer.model";
    if (!fs::exists(src)) SKIP("alice tokenizer.model not present");

    const fs::path out = fs::temp_directory_path() / "alice_tok_meta";
    fs::remove_all(out);
    fs::create_directories(out);
    REQUIRE(lsxcommon::BuildTokenizerArrowFilesSentencePiece(src.string(),
                                                             out.string())
                .ok());

    lsxcommon::ArrowTokenizerDb db;
    REQUIRE(LoadAliceDb(db, out));
    REQUIRE(db.tokenizer_kind == 1);
    REQUIRE(db.vocab_size == 129024);
    REQUIRE(db.bos_id == 1);
    REQUIRE(db.eos_id == 2);
    REQUIRE(lsxcommon::arrow_tokenizer_lookup(db, "<s>", 3) == 1);
    REQUIRE(lsxcommon::arrow_tokenizer_lookup(db, "</s>", 4) == 2);
    REQUIRE(lsxcommon::arrow_tokenizer_lookup(db, "<0x00>", 6) == 7);
    REQUIRE(lsxcommon::arrow_tokenizer_lookup(db, "<0xFF>", 6) == 262);
}

TEST_CASE("sp-bpe encode/decode matches sentencepiece goldens", "[alice][fast]")
{
    const fs::path src = AliceTokenModelDir() / "tokenizer.model";
    if (!fs::exists(src)) SKIP("alice tokenizer.model not present");

    const fs::path out = fs::temp_directory_path() / "alice_tok_meta";
    fs::create_directories(out);
    REQUIRE(lsxcommon::BuildTokenizerArrowFilesSentencePiece(src.string(),
                                                             out.string())
                .ok());
    lsxcommon::ArrowTokenizerDb db;
    REQUIRE(LoadAliceDb(db, out));

    std::ifstream f(std::string(LSX_SOURCE_DIR) +
                    "/tests/fixtures/tokenizer/alice.json");
    if (!f) SKIP("golden missing");
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    simdjson::padded_string pad(json);
    simdjson::ondemand::parser parser;
    auto doc = parser.iterate(pad);
    simdjson::ondemand::array cases;
    REQUIRE(doc["cases"].get_array().get(cases) == simdjson::SUCCESS);

    int checked = 0;
    for (auto c : cases) {
        std::string_view text;
        REQUIRE(c["text"].get_string().get(text) == simdjson::SUCCESS);
        std::vector<int32_t> want;
        for (auto id : c["ids"].get_array()) {
            int64_t v;
            REQUIRE(id.get_int64().get(v) == simdjson::SUCCESS);
            want.push_back(static_cast<int32_t>(v));
        }
        const auto got = Encode(db, std::string(text));
        INFO("text=" << text);
        REQUIRE(got == want);

        std::string_view decoded;
        REQUIRE(c["decoded"].get_string().get(decoded) == simdjson::SUCCESS);
        // Decode the content tokens (drop the leading bos).
        std::vector<int32_t> body(got.begin() + (db.bos_id >= 0 ? 1 : 0),
                                  got.end());
        REQUIRE(Decode(db, body) == std::string(decoded));
        ++checked;
    }
    REQUIRE(checked == 8);
}
