// Token-level parity for the tiktoken bridge: the engine's BPE must reproduce
// the checkpoint tokenizer's ids exactly. The golden is captured by
// tools/kimi_tokenizer_capture.py from the checkpoint's own tokenizer
// (tiktoken + Kimi's pat_str), never from this bridge.
//
// Real-vault target: SKIPs when the checkpoint is absent (SKIP_RETURN_CODE 4).
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
    std::ifstream f(LSX_KIMI_K3_FIXTURE_DIR "/tokenizer_parity.txt", std::ios::binary);
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

} // namespace

TEST_CASE("kimi tiktoken bridge reproduces the reference ids",
          "[kimi-k3][tokenizer][fast]")
{
    const std::filesystem::path model =
        manifest_query::SourceDirFor("kimi-k3");
    if (!std::filesystem::exists(model / "tiktoken.model"))
        SKIP(std::string("vault Kimi tokenizer not present: ") + model.string());

    const auto pinned =
        std::filesystem::temp_directory_path() / "kimi_k3_tok_parity";
    std::filesystem::remove_all(pinned);
    std::filesystem::create_directories(pinned);

    lsxcommon::TikTokenSpecialPolicy pol;
    pol.bos_id = -1;      // reference emits no BOS
    pol.eos_id = 163586;  // <|end_of_msg|>
    pol.pad_id = 163839;
    REQUIRE(lsxcommon::BuildTokenizerArrowFilesTikToken(
                (model / "tiktoken.model").string(),
                (model / "tokenizer_config.json").string(), pinned, pol)
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

    for (const Case &c : LoadCases()) {
        std::vector<int32_t> ids(c.text.size() + 16, 0);
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
}
