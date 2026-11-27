// Tokenizer parity for the Hy-Memory-30B-A3B byte-level BPE bridge: the
// engine's encoder must reproduce the checkpoint tokenizer's ids exactly on a
// corpus that is deliberately adversarial for the PRE-tokenizer (<=3-digit
// groups, CJK isolation, multi-space lookahead, contractions) and for the two
// vocabulary quirks this checkpoint has (no CR byte token, which the reference
// DROPS; no single-byte token for the 13 UTF-8-illegal lead bytes).
//
// The golden (tests/fixtures/hy/tokenizer_parity.jsonl) is captured by
// tools/hy_tokenizer_capture.py from the checkpoint's own tokenizer, never from
// this bridge. SKIPs (4) when the converted vault entry is absent.
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <simdjson.h>

#include <lsxcommon/path_registry.h>
#include <lsxcommon/tokenizer.h>

#ifndef LSX_HY_FIXTURE_DIR
#define LSX_HY_FIXTURE_DIR "tests/fixtures/hy"
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
    std::ifstream f(LSX_HY_FIXTURE_DIR "/tokenizer_parity.jsonl",
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
        for (auto e : doc["ids"].get_array()) {
            int64_t v = 0;
            REQUIRE(e.get_int64().get(v) == simdjson::SUCCESS);
            c.ids.push_back(static_cast<int32_t>(v));
        }
        cases.push_back(std::move(c));
    }
    return cases;
}

bool LoadVaultDb(lsxcommon::ArrowTokenizerDb &db)
{
    const std::filesystem::path root =
        std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
        "Hy-Memory-30B-A3B" / "pinned";
    const std::filesystem::path v = root / "tokenizer_vocab.arrow";
    const std::filesystem::path m = root / "tokenizer_merges.arrow";
    if (!std::filesystem::exists(v) || !std::filesystem::exists(m)) return false;
    auto vf = arrow::io::MemoryMappedFile::Open(v.string(),
                                                arrow::io::FileMode::READ);
    auto mf = arrow::io::MemoryMappedFile::Open(m.string(),
                                                arrow::io::FileMode::READ);
    if (!vf.ok() || !mf.ok()) return false;
    return lsxcommon::load_tokenizer_metadata(*vf, *mf, db).ok();
}

} // namespace

TEST_CASE("Hy tokenizer reproduces the checkpoint ids exactly",
          "[hy][tokenizer]")
{
    lsxcommon::ArrowTokenizerDb db;
    if (!LoadVaultDb(db)) SKIP("converted Hy-Memory vault entry absent");
    const auto cases = LoadCases();
    REQUIRE(cases.size() >= 10);
    for (const auto &c : cases) {
        std::vector<int32_t> out(c.text.size() * 4 + 16, 0);
        size_t cap = out.size();
        const int32_t n = lsxcommon::arrow_tokenizer_encode(
            db, c.text.data(), c.text.size(), out.data(), &cap);
        INFO("text: " << c.text);
        REQUIRE(n >= 0);
        out.resize(static_cast<size_t>(n));
        INFO("want: " << Join(c.ids) << "\n got: " << Join(out));
        CHECK(out == c.ids);
    }
}
