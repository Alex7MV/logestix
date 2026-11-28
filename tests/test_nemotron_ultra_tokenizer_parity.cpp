// Token-level parity for the Nemotron-3-Ultra-550B tokenizer bridge
// (tokenizer_kind 9, the same MistralWordStream kind the 30B siblings use).
//
// The checkpoint ships the BYTE-IDENTICAL `tokenizer.json`
// (md5 e7f0d67fb6f0824ce61e6b85b7d0ba57 — the same file the 30B BF16/NVFP4 pair
// and Dory1.0 ship), so this target reuses the family golden captured by
// tools/nemotron_tokenizer_capture.py from that file instead of re-deriving one.
// The sharing is deliberate: the source of truth is the checkpoint file, and it
// is the same file.
//
// SKIPs (exit 4) when the Ultra vault directory is absent.
#include <catch2/catch_test_macros.hpp>

#include "test_manifest_query_root.h"

#include <arrow/io/file.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <lsxcommon/model_integrity.h>
#include <lsxcommon/tokenizer.h>

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

// Same reader as the 30B target: `CASE <len> <count> <ids...>` then `len` bytes
// of raw prompt text on the following bytes of the stream.
std::vector<Case> LoadCases()
{
    std::ifstream f(LSX_NEMOTRON_FIXTURE_DIR "/tokenizer_parity.txt", std::ios::binary);
    REQUIRE(f.good());
    std::vector<Case> cases;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("CASE ", 0) != 0) continue;
        long len = 0, count = 0;
        int consumed = 0;
        REQUIRE(std::sscanf(line.c_str(), "CASE %ld %ld %n", &len, &count, &consumed) == 2);
        Case c;
        c.ids.reserve(static_cast<size_t>(count));
        const char *p = line.c_str() + consumed;
        while (static_cast<long>(c.ids.size()) < count) {
            char *end = nullptr;
            const long v = std::strtol(p, &end, 10);
            REQUIRE(end != p);
            c.ids.push_back(static_cast<int32_t>(v));
            p = end;
        }
        c.text.resize(static_cast<size_t>(len));
        f.read(c.text.data(), len);
        char nl = 0;
        f.get(nl);
        cases.push_back(std::move(c));
    }
    return cases;
}

} // namespace

TEST_CASE("ultra tokenizer bridge reproduces the reference ids", "[ultra][tokenizer]")
{
    namespace fs = std::filesystem;
    const fs::path model = manifest_query::SourceDirFor("nemotron-ultra-550b-a55b");
    std::error_code ec;
    if (!fs::exists(model / "tokenizer.json", ec))
        SKIP("Ultra vault tokenizer not present: " + model.string());

    const auto pinned = fs::temp_directory_path() / "nemotron_ultra_tok_parity";
    fs::remove_all(pinned);
    fs::create_directories(pinned);

    // Mistral/Nemotron WordStream -> tokenizer_kind 9, exactly as the converter
    // builds it.
    lsxcommon::TokenizerSpecialPolicy pol;
    pol.pretokenizer = 7;
    const auto built = lsxcommon::integrity::BuildAndVerifyTokenizer(
        (model / "tokenizer.json").string(), pinned,
        lsxcommon::integrity::TokenizerSpecialIds{.bos_required = false,
                                                  .eos_required = true},
        pol);
    INFO(built.ToString());
    REQUIRE(built.ok());

    lsxcommon::ArrowTokenizerDb db;
    auto v = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    auto m = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    REQUIRE(v.ok());
    REQUIRE(m.ok());
    const auto ld = lsxcommon::load_tokenizer_metadata(*v, *m, db);
    INFO(ld.ToString());
    REQUIRE(ld.ok());
    CHECK(db.tokenizer_kind == 9);

    const std::vector<Case> cases = LoadCases();
    REQUIRE(!cases.empty());
    int failures = 0;
    for (const Case &c : cases) {
        std::vector<int32_t> ids(c.text.size() * 4 + 64, 0);
        size_t cap = ids.size();
        const int32_t n = lsxcommon::arrow_tokenizer_encode(
            db, c.text.data(), c.text.size(), ids.data(), &cap);
        REQUIRE(n >= 0);
        ids.resize(static_cast<size_t>(n));
        if (ids != c.ids) {
            ++failures;
            INFO("text: [" << c.text << "]\n engine: " << Join(ids)
                           << "\n ref   : " << Join(c.ids));
            CHECK(ids == c.ids);
        }
    }
    CHECK(failures == 0);
    fs::remove_all(pinned);
}
