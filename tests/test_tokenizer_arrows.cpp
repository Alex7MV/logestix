// LoadTokenizerArrows (host-only; no CUDA, no weights, no vault).
//
// tokenizer_arrows.cpp mmap's a converted model's tokenizer arrows and falls
// back to the legacy MoE layout roots when <model>/logestix/pinned is absent.
// Before this suite the loader was reached only through taskmosaic's
// LsxPromptFramer and prompt_bridge, i.e. never on a synthetic tree — so the
// alternate-root fallback and the error contract were untested.
//
// The index-only fixture (test_lightweight_model_fixture.h) is enough: the
// tokenizer loader reads nothing else.
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>

#include <lsxcommon/tokenizer_arrows.h>

#include "test_lightweight_model_fixture.h"

namespace fs = std::filesystem;
using lsxcommon::ArrowTokenizerDb;
using lsxcommon::LoadTokenizerArrows;

namespace {

// A pinned dir holding the tokenizer arrows written from a synthetic json.
void WritePinnedArrows(const fs::path &pinned,
                       const std::string &tokenizer_json)
{
    fs::create_directories(pinned);
    REQUIRE(lsxcommon::BuildTokenizerArrowFiles(tokenizer_json, pinned.string())
                .ok());
}

} // namespace

TEST_CASE("tokenizer arrows: loads vocab and merges from logestix/pinned",
          "[tokenizer_arrows]")
{
    synthmodel::LightweightModel m = synthmodel::MakeLightweightModel(
        TOKENIZER_TEST_DATA_DIR "/synthetic_qwen_tok.json", "Qwen3.5-9B",
        "lsx_tok_arrows_main");
    REQUIRE(m.db.error_msg.empty());

    ArrowTokenizerDb db;
    REQUIRE(LoadTokenizerArrows(m.path(), db));
    CHECK(db.error_msg.empty());
    CHECK(db.vocab_size > 0);
    CHECK(db.bos_id == 0);
    // The synthetic vocab carries the ChatML tool sentinels, so a lookup
    // proves the vocab/merges pair was actually parsed rather than just
    // mmapped (an empty mmapped file would also "load" without erroring).
    CHECK(lsxcommon::arrow_tokenizer_lookup(db, "<|bos|>", 7) >= 0);
    CHECK(lsxcommon::arrow_tokenizer_lookup(db, "<tool_call>", 11) >= 0);
}

TEST_CASE("tokenizer arrows: falls back to the legacy MoE layout roots",
          "[tokenizer_arrows]")
{
    // ResolvePinnedDir probes logestix-{qwen35moe,glm,kimi,gemma}/pinned when
    // the canonical logestix/pinned is missing. Each root must work on its
    // own, because a converted model only ever carries one of them.
    for (const char *root : {"logestix-qwen35moe", "logestix-glm",
                             "logestix-kimi", "logestix-gemma"}) {
        CAPTURE(root);
        const fs::path base =
            fs::temp_directory_path() / (std::string("lsx_tok_alt_") + root);
        std::error_code ec;
        fs::remove_all(base, ec);
        synthmodel::WriteIndex(base, "Qwen3.5-9B");
        WritePinnedArrows(base / root / "pinned",
                          TOKENIZER_TEST_DATA_DIR "/synthetic_qwen_tok.json");

        ArrowTokenizerDb db;
        REQUIRE(LoadTokenizerArrows(base.string(), db));
        CHECK(db.error_msg.empty());
        CHECK(db.vocab_size > 0);
        db = ArrowTokenizerDb{};
        fs::remove_all(base, ec);
    }
}

TEST_CASE("tokenizer arrows: the canonical root wins over a legacy one",
          "[tokenizer_arrows]")
{
    const fs::path base = fs::temp_directory_path() / "lsx_tok_precedence";
    std::error_code ec;
    fs::remove_all(base, ec);
    synthmodel::WriteIndex(base, "Qwen3.5-9B");
    // A legacy root with a broken vocab, plus a valid canonical one: if the
    // fallback were consulted first, the load would fail.
    fs::create_directories(base / "logestix-glm" / "pinned");
    {
        std::ofstream bad(base / "logestix-glm" / "pinned" /
                          "tokenizer_vocab.arrow");
        bad << "not an arrow file";
    }
    WritePinnedArrows(base / "pinned",
                      TOKENIZER_TEST_DATA_DIR "/synthetic_qwen_tok.json");

    ArrowTokenizerDb db;
    REQUIRE(LoadTokenizerArrows(base.string(), db));
    CHECK(db.vocab_size > 0);
    db = ArrowTokenizerDb{};
    fs::remove_all(base, ec);
}

TEST_CASE("tokenizer arrows: a missing model dir fails with the resolved path",
          "[tokenizer_arrows]")
{
    const fs::path missing =
        fs::temp_directory_path() / "lsx_tok_definitely_absent";
    std::error_code ec;
    fs::remove_all(missing, ec);

    ArrowTokenizerDb db;
    CHECK_FALSE(LoadTokenizerArrows(missing.string(), db));
    CHECK_FALSE(db.error_msg.empty());
    CHECK(db.error_msg.find("tokenizer arrows not found") != std::string::npos);
    // The message must name the directory that was probed, otherwise a user
    // with a MoE-layout model cannot tell which path was searched.
    CHECK(db.error_msg.find("pinned") != std::string::npos);
}

TEST_CASE("tokenizer arrows: a corrupt vocab file fails cleanly", "[tokenizer_arrows]")
{
    const fs::path base = fs::temp_directory_path() / "lsx_tok_corrupt";
    std::error_code ec;
    fs::remove_all(base, ec);
    synthmodel::WriteIndex(base, "Qwen3.5-9B");
    fs::create_directories(base / "pinned");
    {
        std::ofstream bad(base / "pinned" / "tokenizer_vocab.arrow");
        bad << "not an arrow file";
        std::ofstream ok(base / "pinned" / "tokenizer_merges.arrow");
        ok << "not an arrow file either";
    }

    ArrowTokenizerDb db;
    CHECK_FALSE(LoadTokenizerArrows(base.string(), db));
    CHECK_FALSE(db.error_msg.empty());
    db = ArrowTokenizerDb{};
    fs::remove_all(base, ec);
}

TEST_CASE("tokenizer arrows: a missing merges file fails the pair", "[tokenizer_arrows]")
{
    // Vocab alone is not a tokenizer: the pair must both load.
    const fs::path base = fs::temp_directory_path() / "lsx_tok_halfwritten";
    std::error_code ec;
    fs::remove_all(base, ec);
    synthmodel::WriteIndex(base, "Qwen3.5-9B");
    fs::create_directories(base / "pinned");
    REQUIRE(lsxcommon::BuildTokenizerArrowFiles(
                TOKENIZER_TEST_DATA_DIR "/synthetic_qwen_tok.json",
                (base / "pinned").string())
                .ok());
    fs::remove(base / "pinned" / "tokenizer_merges.arrow", ec);

    ArrowTokenizerDb db;
    CHECK_FALSE(LoadTokenizerArrows(base.string(), db));
    CHECK_FALSE(db.error_msg.empty());
    db = ArrowTokenizerDb{};
    fs::remove_all(base, ec);
}