// TDD — GigaChat4-708B-A41B tokenizer bridge (Model Ingestion Protocol stage 4).
// Runs against the REAL vault tokenizer and SKIPs when the converted tree is
// absent, so it never fails a machine without the model.
//
// The point of the ALIGNMENT INVARIANT is that the framing markers resolve to
// the checkpoint's own ids. A silent fallback is the failure mode: if
// `arrow_tokenizer_lookup` cannot find `<|role_sep|>\n`, BuildPromptInputIds
// degrades to a raw BPE encode, the model is served an unframed prompt and the
// answer becomes plausible-looking junk instead of an error. That is invisible
// on a "Paris" probe (a raw "The capital of France is" completes to " Paris"
// too), so it needs its own assertion.
#include <catch2/catch_test_macros.hpp>

#include "test_gigachat4_fixture.h"

#include "lsxcommon/gigachat_moe_708b_model.h"
#include "lsxcommon/tokenizer.h"

#include <arrow/io/file.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace lsxcommon;

namespace {

std::filesystem::path VaultRoot()
{
    if (const char *env = std::getenv("LSX_TEST_GIGACHAT4_708B_A41B_ROOT"))
        return env;
    return "/mnt/fast_storage/lsx_model/ai-sage/GigaChat4-708B-A41B";
}

ArrowTokenizerDb LoadVaultTokenizer(const std::filesystem::path &root)
{
    const auto pinned = root / "logestix" / "pinned";
    auto v = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(),
        arrow::io::FileMode::READ);
    auto m = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(),
        arrow::io::FileMode::READ);
    REQUIRE(v.ok());
    REQUIRE(m.ok());
    ArrowTokenizerDb db;
    auto st = load_tokenizer_metadata(*v, *m, db);
    INFO("load_tokenizer_metadata: " << st.ToString());
    REQUIRE(st.ok());
    return db;
}

int CountId(const std::vector<int32_t> &ids, int32_t want)
{
    return static_cast<int>(std::count(ids.begin(), ids.end(), want));
}

} // namespace

TEST_CASE("gigachat4-708b tokenizer bridge resolves the checkpoint's own markers",
          "[gigachat708b][tokenizer]")
{
    const auto root = VaultRoot();
    if (!std::filesystem::exists(root / "logestix" / "pinned" /
                                 "tokenizer_vocab.arrow"))
        SKIP("converted model absent");

    const auto db = LoadVaultTokenizer(root);
    // Same byte-level BPE and marker ids as the 37B sibling: tokenizer.json is
    // byte-identical, and the loader takes bos/eos/pad from config.json (1/2/2).
    CHECK(db.tokenizer_kind == 3);
    CHECK(db.bos_id == 1);
    CHECK(db.eos_id == 2);
    CHECK(db.pad_id == 2);

    // Stage 4 alignment invariant: the structural markers must resolve to the
    // ids the checkpoint's chat template uses. A -1 here means the framing
    // silently degrades to a raw encode.
    CHECK(arrow_tokenizer_lookup(db, "<|role_sep|>\n", 13) == 128000);
    CHECK(arrow_tokenizer_lookup(db, "<|message_sep|>\n\n", 17) == 128001);
    CHECK(arrow_tokenizer_lookup(db, "<think>", 7) == 128014);
    CHECK(arrow_tokenizer_lookup(db, "</think>", 8) == 128015);
}

TEST_CASE("gigachat4-708b prompt framing is chatml-v5, not a raw encode",
          "[gigachat708b][tokenizer]")
{
    const auto root = VaultRoot();
    if (!std::filesystem::exists(root / "logestix" / "pinned" /
                                 "tokenizer_vocab.arrow"))
        SKIP("converted model absent");

    const auto db = LoadVaultTokenizer(root);
    ModelInitConfig cfg;
    cfg.model_type = "gigachat4-708b-a41b";
    GigaChat4Moe708BModel model(cfg);

    std::vector<int32_t> ids;
    REQUIRE(model.BuildPromptInputIds(db, "What is the capital of France?", ids));
    REQUIRE(ids.size() > 8);

    // One BOS, then the chatml-v5 skeleton for a single turn: three role
    // separators (system / user / assistant) and two message separators (after
    // each payload), then the closed think block at the generation prompt.
    CHECK(ids.front() == 1);
    CHECK(CountId(ids, 1) == 1);
    CHECK(CountId(ids, 128000) == 3);
    CHECK(CountId(ids, 128001) == 2);
    REQUIRE(ids.size() >= 2);
    CHECK(ids[ids.size() - 2] == 128014);  // <think>
    CHECK(ids.back() == 128015);           // </think>
}

TEST_CASE("gigachat4-708b tokenizer round-trips a mixed-case identifier",
          "[gigachat708b][tokenizer]")
{
    const auto root = VaultRoot();
    if (!std::filesystem::exists(root / "logestix" / "pinned" /
                                 "tokenizer_vocab.arrow"))
        SKIP("converted model absent");

    const auto db = LoadVaultTokenizer(root);
    // Kind 3 (the Llama-3 case-run split) must keep the sub-word boundaries:
    // `get,User,Name` rather than the legacy kind-0 `get,UserName`.
    for (const char *text : {"getUserName", "The capital of France is",
                             "DET-Validation-Core-02"})
    {
        INFO("text: " << text);
        std::vector<int32_t> ids(64, 0);
        size_t cap = ids.size();
        const int32_t n = arrow_tokenizer_encode(db, text, std::strlen(text),
                                                 ids.data(), &cap);
        REQUIRE(n > 0);
        ids.resize(static_cast<size_t>(n));
        std::vector<char> back(256, '\0');
        size_t back_len = back.size();
        REQUIRE(arrow_tokenizer_decode(db, ids.data(), ids.size(), back.data(),
                                       &back_len) == 0);
        CHECK(std::string(back.data(), back_len) == text);
    }
}
