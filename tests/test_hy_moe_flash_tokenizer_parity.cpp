// Tokenizer parity for the Hy4-flash byte-level BPE bridge (tokenizer_kind 13,
// HunyuanWordStream): the engine's encoder must reproduce the checkpoint
// tokenizer's ids exactly on a corpus that is adversarial for this checkpoint's
// own three-split Sequence — 1-3 digit isolation, CJK isolation, the
// punctuation+ASCII-letter alternative that replaces a contraction rule, the
// single-codepoint prefix of the letter alternative, the optional space before a
// punctuation run, CRLF runs, and the Hunyuan chat/tool markers that the
// reference matches literally.
//
// The golden (tests/fixtures/hy4/tokenizer_parity.jsonl) is captured by
// tools/hy4_tokenizer_capture.py from the checkpoint's own tokenizer, never from
// this bridge. SKIPs (4) when the converted vault entry is absent.
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <simdjson.h>

#include <lsxcommon/path_registry.h>
#include <lsxcommon/tokenizer.h>

#ifndef LSX_HY4_FIXTURE_DIR
#define LSX_HY4_FIXTURE_DIR "tests/fixtures/hy4"
#endif

namespace {

struct Case {
    std::string text;
    std::vector<int32_t> ids;
};

std::string Join(const std::vector<int32_t> &v)
{
    std::string s;
    for (auto x : v) s += std::to_string(x) + " ";
    return s;
}

std::vector<Case> LoadCases()
{
    std::ifstream f(LSX_HY4_FIXTURE_DIR "/tokenizer_parity.jsonl",
                    std::ios::binary);
    REQUIRE(f.good());
    std::string line;
    std::vector<Case> cases;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        simdjson::dom::parser parser;
        simdjson::dom::element doc;
        REQUIRE(parser.parse(line).get(doc) == simdjson::SUCCESS);
        Case c;
        std::string_view text;
        REQUIRE(doc["text"].get_string().get(text) == simdjson::SUCCESS);
        c.text = std::string(text);
        simdjson::dom::array arr;
        REQUIRE(doc["ids"].get_array().get(arr) == simdjson::SUCCESS);
        for (auto v : arr) {
            int64_t id = 0;
            REQUIRE(v.get_int64().get(id) == simdjson::SUCCESS);
            c.ids.push_back(static_cast<int32_t>(id));
        }
        cases.push_back(std::move(c));
    }
    return cases;
}

std::filesystem::path VaultRoot()
{
    if (const char *env = std::getenv("LSX_TEST_HY4_FLASH_ROOT"))
        return std::filesystem::path(env);
    return std::filesystem::path("/mnt/fast_storage/lsx_model/tencent/Hy4-flash");
}

} // namespace

TEST_CASE("hy4 tokenizer: the bridge reproduces the reference ids",
          "[hy4][fast]")
{
    const auto pinned = VaultRoot() / "pinned";
    if (!std::filesystem::exists(pinned / "tokenizer_vocab.arrow"))
        SKIP("converted Hy4-flash tokenizer absent at " + pinned.string());

    const auto cases = LoadCases();
    REQUIRE(cases.size() >= 30);

    lsxcommon::ArrowTokenizerDb db;
    {
        auto vf = arrow::io::MemoryMappedFile::Open(
            (pinned / "tokenizer_vocab.arrow").string(),
            arrow::io::FileMode::READ);
        auto mf = arrow::io::MemoryMappedFile::Open(
            (pinned / "tokenizer_merges.arrow").string(),
            arrow::io::FileMode::READ);
        REQUIRE(vf.ok());
        REQUIRE(mf.ok());
        const auto loaded = lsxcommon::load_tokenizer_metadata(*vf, *mf, db);
        REQUIRE(loaded.ok());
    }
    CHECK(db.tokenizer_kind == 13);

    int failures = 0;
    for (const auto &c : cases) {
        std::vector<int32_t> got(c.text.size() + 8);
        size_t cap = got.size();
        // The convenience overload returns the TOKEN COUNT (negative on error).
        const int32_t n = lsxcommon::arrow_tokenizer_encode(
            db, c.text.data(), c.text.size(), got.data(), &cap);
        REQUIRE(n >= 0);
        got.resize(static_cast<size_t>(n));
        if (got != c.ids) {
            ++failures;
            INFO("text: " << c.text);
            INFO("want: " << Join(c.ids));
            INFO("got:  " << Join(got));
            CHECK(got == c.ids);
        }
    }
    CHECK(failures == 0);
}
