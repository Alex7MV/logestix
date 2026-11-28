// Token-level parity for the Nemotron-3.5-Lightning tokenizer bridge
// (tokenizer_kind 9, MistralWordStream). The golden is captured by
// tools/nemotron_tokenizer_capture.py from the checkpoint's own tokenizer.json
// (tokenizers library, add_special_tokens=False) — never from this bridge.
//
// The bridge is exercised as the converter builds it: tokenizer.json from the
// source directory through integrity::BuildAndVerifyTokenizer with the
// Nemotron policy (pretokenizer 7, no required BOS, required EOS).
//
// Real-vault target: SKIPs when the checkpoint is absent (SKIP_RETURN_CODE 4).
// Shared by both nemotron suites — the sibling variants ship one byte-identical
// tokenizer.json, so one target polices the pair.
#include <catch2/catch_test_macros.hpp>

#include "test_manifest_query_root.h"

#include <arrow/io/file.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <lsxcommon/converter.h>
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

std::vector<Case> LoadCases()
{
    std::ifstream f(LSX_NEMOTRON_FIXTURE_DIR "/tokenizer_parity.txt",
                    std::ios::binary);
    REQUIRE(f.good());
    std::vector<Case> cases;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("CASE ", 0) != 0) continue;
        long len = 0, count = 0;
        int consumed = 0;
        REQUIRE(std::sscanf(line.c_str(), "CASE %ld %ld %n", &len, &count,
                            &consumed) == 2);
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

// The BF16 sibling is the canonical source when both are in the vault; the
// NVFP4 distribution ships the same tokenizer.json.
std::filesystem::path SourceModelDir()
{
    if (const char *env = std::getenv("LSX_TEST_NEMOTRON_TOKENIZER_SOURCE"))
        return std::filesystem::path(env);
    // Both siblings are registry rows, so both spellings come from the registry;
    // the store directories are `nvidia/NVIDIA-Nemotron-3.5-...-{BF16,NVFP4}`.
    const auto bf16 =
        manifest_query::SourceDirFor("nemotron-3.5-lightning-30b-a3b");
    if (std::filesystem::exists(bf16 / "tokenizer.json")) return bf16;
    return manifest_query::SourceDirFor(
        "nemotron-3.5-lightning-30b-a3b-nvfp4");
}

} // namespace

TEST_CASE("nemotron tokenizer bridge reproduces the reference ids",
          "[nemotron][tokenizer][inference]")
{
    const std::filesystem::path model = SourceModelDir();
    if (!std::filesystem::exists(model / "tokenizer.json"))
        SKIP(std::string("vault Nemotron tokenizer not present: ") + model.string());

    const auto pinned =
        std::filesystem::temp_directory_path() / "nemotron_tok_parity";
    std::filesystem::remove_all(pinned);
    std::filesystem::create_directories(pinned);

    lsxcommon::TokenizerSpecialPolicy pol;
    pol.pretokenizer = 7;  // Mistral/Nemotron WordStream -> tokenizer_kind 9
    REQUIRE(lsxcommon::integrity::BuildAndVerifyTokenizer(
                (model / "tokenizer.json").string(), pinned,
                lsxcommon::integrity::TokenizerSpecialIds{
                    .bos_required = false, .eos_required = true},
                pol)
                .ok());

    lsxcommon::ArrowTokenizerDb db;
    auto v = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(),
        arrow::io::FileMode::READ);
    auto m = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(),
        arrow::io::FileMode::READ);
    REQUIRE(v.ok());
    REQUIRE(m.ok());
    const auto ld = lsxcommon::load_tokenizer_metadata(*v, *m, db);
    INFO(ld.ToString());
    REQUIRE(ld.ok());
    CHECK(db.tokenizer_kind == 9);
    // The chat template spells the ChatML markers out as text, so the encoder
    // must not prepend the checkpoint's `<s>` even though the vocabulary has
    // one (the ids below match the reference encode exactly, which is what
    // pins that). The metadata id is informational only.
    INFO("bos_id: " << db.bos_id);

    const std::vector<Case> cases = LoadCases();
    REQUIRE(!cases.empty());
    for (const Case &c : cases) {
        std::vector<int32_t> ids(c.text.size() * 4 + 64, 0);
        size_t cap = ids.size();
        const int32_t n = lsxcommon::arrow_tokenizer_encode(
            db, c.text.data(), c.text.size(), ids.data(), &cap);
        INFO("text: " << c.text);
        REQUIRE(n >= 0);
        ids.resize(static_cast<size_t>(n));
        INFO("engine: " << Join(ids));
        INFO("ref   : " << Join(c.ids));
        CHECK(ids == c.ids);
    }

    std::filesystem::remove_all(pinned);
}
