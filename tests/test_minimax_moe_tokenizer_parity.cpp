// Tokenizer parity for the MiniMax-M3 byte-level BPE bridge: the engine's
// encoder must reproduce the checkpoint tokenizer's ids exactly on a corpus
// that is deliberately adversarial for the PRE-tokenizer (Llama-3/o200k-style
// case runs, <=3-digit groups, multi-space lookahead, contractions, CJK).
//
// The golden (tests/fixtures/minimax/tokenizer_parity.txt) is captured by
// tools/minimax_tokenizer_capture.py from the checkpoint's own tokenizer,
// never from this bridge. SKIPs (4) when the converted vault entry is absent.
#include <catch2/catch_test_macros.hpp>

#include "test_manifest_query_root.h"

#include <arrow/io/file.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <simdjson.h>

#include <lsxcommon/tokenizer.h>

#ifndef LSX_MINIMAX_FIXTURE_DIR
#define LSX_MINIMAX_FIXTURE_DIR "tests/fixtures/minimax"
#endif

// JSON Lines fixture: {"text": "...", "ids": [...]} per case.

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
    std::ifstream f(LSX_MINIMAX_FIXTURE_DIR "/tokenizer_parity.jsonl",
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
        for (auto v : doc["ids"].get_array()) {
            int64_t id = 0;
            REQUIRE(v.get_int64().get(id) == simdjson::SUCCESS);
            c.ids.push_back(static_cast<int32_t>(id));
        }
        REQUIRE(!c.ids.empty());
        cases.push_back(std::move(c));
    }
    return cases;
}

std::filesystem::path ModelRoot()
{
    if (const char *env = std::getenv("LSX_TEST_MINIMAX_ROOT")) return env;
    return manifest_query::ConvertedDirFor("minimax-m3");
}

} // namespace

TEST_CASE("minimax tokenizer parity against the checkpoint tokenizer",
          "[minimax][tokenizer][real-model]")
{
    const std::filesystem::path pinned = ModelRoot() / "pinned";
    if (!std::filesystem::exists(pinned / "tokenizer_vocab.arrow"))
        SKIP(std::string("MiniMax-M3 converted tokenizer not present: ") + pinned.string());

    auto v = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(),
        arrow::io::FileMode::READ);
    REQUIRE(v.ok());
    auto m = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(),
        arrow::io::FileMode::READ);
    REQUIRE(m.ok());
    lsxcommon::ArrowTokenizerDb db;
    REQUIRE(lsxcommon::load_tokenizer_metadata(*v, *m, db).ok());
    REQUIRE(db.valid());

    const auto cases = LoadCases();
    REQUIRE(!cases.empty());
    for (const auto &c : cases) {
        std::vector<int32_t> got(c.text.size() + 16);
        size_t cap = got.size();
        const int32_t n = lsxcommon::arrow_tokenizer_encode(
            db, c.text.data(), c.text.size(), got.data(), &cap);
        INFO("text: " << c.text);
        INFO("want: " << Join(c.ids));
        INFO("got:  " << (n > 0 ? Join(std::vector<int32_t>(
                                      got.begin(), got.begin() + n))
                                : std::string("<encode failed>")));
        REQUIRE(n == static_cast<int32_t>(c.ids.size()));
        got.resize(static_cast<size_t>(n));
        CHECK(got == c.ids);
    }
}
