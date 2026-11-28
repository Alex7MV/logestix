// Step-5-Preview tokenizer parity (host-only; the real checkpoint tokenizer).
// The pre-tokenizer is the Flint 3-split (numeric runs + CJK isolated) over the
// o200k case-run pattern with [\p{P}\p{S}]+[\r\n]*, which the bridge writes as
// `tokenizer_kind = 4` (policy.pretokenizer = 2). These ids were captured from
// the checkpoint's own tokenizer.json (`tokenizers.Tokenizer.from_file`, BOS
// stripped — the chat template spells BOS out and the bridge runs with
// bos_id = -1).
#include <catch2/catch_test_macros.hpp>

#include "test_manifest_query_root.h"

#include <arrow/io/file.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/converter.h>
#include <lsxcommon/tokenizer.h>

namespace fs = std::filesystem;

namespace {

struct Case {
    const char *text;
    std::vector<int32_t> ids;
};

const std::vector<Case> &Cases()
{
    static const std::vector<Case> v = {
        {"The capital of France is", {671, 6102, 294, 8760, 344}},
        {"Hello, world! 12345", {19923, 14, 2058, 3, 223, 6895, 1883}},
        {"def f(x): return x+1", {3465, 285, 4042, 2605, 1354, 1527, 13, 19}},
        {"\xD0\x9A\xD0\xB8\xD1\x80\xD0\xB8\xD0\xBB\xD0\xBB\xD0\xB8\xD1\x86\xD0\xB0"
         " \xD0\xB8 \xE4\xB8\xAD\xE6\x96\x87\xE6\xB5\x8B\xE8\xAF\x95",
         {7725, 393, 1558, 31700, 9667, 1130, 223, 21134, 10251}},
        {"a\nb\tc", {67, 201, 68, 28544}},
        {"{ \"k\": [1, 2.5, true] }",
         {93, 582, 77, 3362, 764, 19, 14, 223, 20, 16, 23, 14, 3292, 63, 837}},
        {"\xF0\x9F\x98\x80 emoji", {28927, 225, 980, 82644}},
    };
    return v;
}

fs::path ModelDir()
{
    if (const char *env = std::getenv("LSX_TEST_STEP_ROOT"))
        return fs::path(env);
    return manifest_query::SourceDirFor("step-5-preview");
}

} // namespace

TEST_CASE("Step-5 tokenizer bridge matches the checkpoint tokenizer",
          "[step][tokenizer]")
{
    manifest_query::RequireSourceRoot();
    const fs::path tok = ModelDir() / "tokenizer.json";
    if (!fs::exists(tok)) SKIP(std::string("Step-5-Preview tokenizer.json not present: ") + tok.string());

    fs::path pinned = fs::temp_directory_path() / "lsx_step_tok";
    fs::remove_all(pinned);
    fs::create_directories(pinned);
    lsxcommon::TokenizerSpecialPolicy policy;
    policy.bos_id = -1;
    policy.pretokenizer = 2;
    auto st = lsxcommon::BuildTokenizerArrowFiles(tok.string(), pinned, policy);
    INFO(st.ToString());
    REQUIRE(st.ok());

    auto v = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    REQUIRE(v.ok());
    auto m = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    REQUIRE(m.ok());
    lsxcommon::ArrowTokenizerDb db;
    REQUIRE(lsxcommon::load_tokenizer_metadata(*v, *m, db).ok());
    CHECK(db.bos_id == -1);

    for (const auto &c : Cases()) {
        std::vector<int32_t> buf(std::strlen(c.text) * 4 + 64);
        size_t cap = buf.size();
        const int32_t n = lsxcommon::arrow_tokenizer_encode(
            db, c.text, std::strlen(c.text), buf.data(), &cap);
        INFO("text: " << c.text);
        REQUIRE(n >= 0);
        std::vector<int32_t> got(buf.begin(), buf.begin() + n);
        CHECK(got == c.ids);

        // Decode round-trip: the bridge must reproduce the input bytes.
        std::vector<char> out(std::strlen(c.text) * 8 + 256);
        size_t ocap = out.size();
        REQUIRE(lsxcommon::arrow_tokenizer_decode(db, got.data(), got.size(),
                                                  out.data(), &ocap) == 0);
        CHECK(std::string(out.data(), ocap) == c.text);
    }
    std::error_code ec;
    fs::remove_all(pinned, ec);
}
