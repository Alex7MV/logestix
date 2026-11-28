// Ling-3.1-flash tokenizer parity for the o200k byte-level BPE bridge
// (tokenizer_kind 3). The golden (tests/fixtures/ling/tokenizer_parity.jsonl)
// is captured by tools/ling_tokenizer_capture.py from the checkpoint's own
// tokenizer, never from this bridge. SKIPs when the source is absent.
#include <catch2/catch_test_macros.hpp>

#include "test_manifest_query_root.h"

#include <arrow/io/file.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <simdjson.h>

#include <lsxcommon/converter.h>
#include <lsxcommon/tokenizer.h>

#ifndef LSX_LING_FIXTURE_DIR
#define LSX_LING_FIXTURE_DIR "tests/fixtures/ling"
#endif
// No literal: the checkpoint comes from the registry
// (`<source root>/<org>/<model>`) — the store's org is not knowable here.
#define LSX_LING_TOKENIZER_PATH \
    (manifest_query::SourceDirFor("ling") / "tokenizer.json")

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
    std::ifstream f(LSX_LING_FIXTURE_DIR "/tokenizer_parity.jsonl",
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

} // namespace

TEST_CASE("Ling o200k tokenizer reproduces the checkpoint ids exactly",
          "[ling][tokenizer]")
{
    const char *env = std::getenv("LSX_LING_TOKENIZER");
    const std::filesystem::path tok =
        env ? std::filesystem::path(env) : std::filesystem::path(
                                                LSX_LING_TOKENIZER_PATH);
    if (!std::filesystem::exists(tok))
        SKIP(std::string("Ling source tokenizer.json absent: ") + tok.string());
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "lsx_ling_tok";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);

    lsxcommon::TokenizerSpecialPolicy policy;
    policy.bos_id = -1;
    policy.pretokenizer = 3;  // Ling letter-run pattern -> tokenizer_kind 5
    policy.eos_id = 156895;
    policy.pad_id = 156892;
    auto st = lsxcommon::BuildTokenizerArrowFiles(tok.string(), dir, policy);
    INFO(st.ToString());
    REQUIRE(st.ok());

    auto vf = arrow::io::MemoryMappedFile::Open(
        (dir / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    auto mf = arrow::io::MemoryMappedFile::Open(
        (dir / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    REQUIRE(vf.ok());
    REQUIRE(mf.ok());
    lsxcommon::ArrowTokenizerDb db;
    auto meta = lsxcommon::load_tokenizer_metadata(*vf, *mf, db);
    INFO(meta.ToString());
    REQUIRE(meta.ok());
    CHECK(db.tokenizer_kind == 5);
    CHECK(db.bos_id == -1);
    CHECK(db.eos_id == 156895);
    CHECK(db.pad_id == 156892);

    const auto cases = LoadCases();
    REQUIRE(cases.size() >= 20);
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
    std::filesystem::remove_all(dir, ec);
}
