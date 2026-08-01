#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/unicode.h>

TEST_CASE("unicode classification spot checks", "[tokenizer][unicode]")
{
    using lsxcommon::unicode::CharClass;
    CHECK(lsxcommon::unicode::classify(0x41) == CharClass::Letter);       // 'A'
    CHECK(lsxcommon::unicode::classify(0x61) == CharClass::Letter);       // 'a'
    CHECK(lsxcommon::unicode::classify(0x4E2D) == CharClass::Letter);     // 中
    CHECK(lsxcommon::unicode::classify(0x9FA6) == CharClass::Letter);     // 龦 (L, outside CJK split range)
    CHECK(lsxcommon::unicode::classify(0x0301) == CharClass::Mark);       // combining acute
    CHECK(lsxcommon::unicode::classify(0x30) == CharClass::Number);       // '0'
    CHECK(lsxcommon::unicode::classify(0x00B2) == CharClass::Number);     // ² (No)
    CHECK(lsxcommon::unicode::classify(0x21) == CharClass::Punct);        // '!'
    CHECK(lsxcommon::unicode::classify(0x20AC) == CharClass::Symbol);     // €
    CHECK(lsxcommon::unicode::classify(0x20) == CharClass::Whitespace);   // space
    CHECK(lsxcommon::unicode::classify(0x00A0) == CharClass::Whitespace); // NBSP
    CHECK(lsxcommon::unicode::classify(0x0A) == CharClass::Whitespace);   // \n
    CHECK(lsxcommon::unicode::classify(0x09) == CharClass::Whitespace);   // \t
    CHECK(lsxcommon::unicode::classify(0x0001) == CharClass::Other);      // control char
}

#include <lsxcommon/schema.h>
#include <arrow/api.h>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

TEST_CASE("TokenizerVocabSchema round-trip", "[tokenizer][schema]")
{
    std::vector<lsxcommon::VocabRow> rows;
    rows.push_back({0, {'<', '|', 'b', 'o', 's', '|', '>'}, 1});
    rows.push_back({1, {}, 1});
    rows.push_back({2, {}, 1});
    for (int b = 3; b < 259; ++b)
        rows.push_back({b, {static_cast<uint8_t>(b - 3)}, 0});

    lsxcommon::TokenizerVocabMeta meta;
    meta.bos_id = 0; meta.eos_id = 1; meta.pad_id = 2; meta.vocab_size = 259;
    for (int i = 0; i < 256; ++i) meta.byte_to_token_id[i] = i + 3;

    auto batch_res = lsxcommon::TokenizerVocabSchema::MakeBatch(rows, meta);
    REQUIRE(batch_res.ok());
    auto batch = *batch_res;
    REQUIRE(batch->ValidateFull().ok());

    auto unpack_res = lsxcommon::TokenizerVocabSchema::UnpackBatch(batch);
    REQUIRE(unpack_res.ok());
    auto &out = *unpack_res;
    REQUIRE(out.size() == 259);
    CHECK(out[0].is_special == 1);
    CHECK(out[35].token_id == 35);
    CHECK(out[35].is_special == 0);
    REQUIRE(out[35].token_bytes.size() == 1);
    CHECK(out[35].token_bytes[0] == 0x20);

    auto path = fs::temp_directory_path() / "logestix_tok_vocab_test.arrow";
    REQUIRE(lsxcommon::WriteRecordBatchToFile(batch, path).ok());
    auto read_res = lsxcommon::ReadRecordBatchFromFile(path);
    REQUIRE(read_res.ok());
    auto meta2 = (*read_res)->schema()->metadata();
    REQUIRE(meta2 != nullptr);
    auto bos_res = meta2->Get("bos_id");
    REQUIRE(bos_res.ok());
    CHECK(*bos_res == "0");
    auto btt_res = meta2->Get("byte_to_token_id");
    REQUIRE(btt_res.ok());
    CHECK(btt_res->rfind("3,", 0) == 0);
    fs::remove(path);
}

#include <lsxcommon/converter.h>

TEST_CASE("BuildTokenizerArrowFiles from synthetic fixture", "[tokenizer][convert]")
{
    auto pinned = fs::temp_directory_path() / "logestix_tok_synth";
    fs::remove_all(pinned);
    fs::create_directories(pinned);

    auto st = lsxcommon::BuildTokenizerArrowFiles(
        TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json", pinned);
    REQUIRE(st.ok());
    REQUIRE(fs::exists(pinned / "tokenizer_vocab.arrow"));
    REQUIRE(fs::exists(pinned / "tokenizer_merges.arrow"));

    auto v = lsxcommon::ReadRecordBatchFromFile(pinned / "tokenizer_vocab.arrow");
    REQUIRE(v.ok());
    CHECK((*v)->num_rows() == 268);

    auto m = lsxcommon::ReadRecordBatchFromFile(pinned / "tokenizer_merges.arrow");
    REQUIRE(m.ok());
    CHECK((*m)->num_rows() == 9);

    auto meta = (*v)->schema()->metadata();
    REQUIRE(meta != nullptr);
    REQUIRE(meta->Get("vocab_size").ok());
    CHECK(*meta->Get("vocab_size") == "268");
    REQUIRE(meta->Get("bos_id").ok());
    CHECK(*meta->Get("bos_id") == "0");
    REQUIRE(meta->Get("byte_to_token_id").ok());
    CHECK(meta->Get("byte_to_token_id")->rfind("3,", 0) == 0);

    auto merges = lsxcommon::TokenizerMergesSchema::UnpackBatch(*m);
    REQUIRE(merges.ok());
    REQUIRE(merges->size() == 9);
    for (size_t i = 1; i < merges->size(); ++i) {
        auto &a = (*merges)[i - 1];
        auto &b = (*merges)[i];
        CHECK((a.merge_left < b.merge_left ||
               (a.merge_left == b.merge_left && a.merge_right < b.merge_right)));
    }
    // "h e" (ids 107/104) -> "he" = 259, ranks follow input order
    bool found_he = false;
    for (auto &r : *merges) {
        if (r.merge_left == 107 && r.merge_right == 104) {
            CHECK(r.merge_token_id == 259);
            CHECK(r.merge_rank == 0);
            found_he = true;
        }
    }
    CHECK(found_he);
    fs::remove_all(pinned);
}

#include <lsxcommon/tokenizer.h>
#include <arrow/io/file.h>

static fs::path BuildSyntheticDb(lsxcommon::ArrowTokenizerDb &db)
{
    auto pinned = fs::temp_directory_path() / "logestix_tok_synth_db";
    fs::remove_all(pinned);
    fs::create_directories(pinned);
    auto st = lsxcommon::BuildTokenizerArrowFiles(
        TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json", pinned);
    REQUIRE(st.ok());
    auto v = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    REQUIRE(v.ok());
    auto m = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    REQUIRE(m.ok());
    auto lst = lsxcommon::load_tokenizer_metadata(*v, *m, db);
    REQUIRE(lst.ok());
    return pinned;
}

TEST_CASE("load_tokenizer_metadata from synthetic files", "[tokenizer][load]")
{
    lsxcommon::ArrowTokenizerDb db;
    auto pinned = BuildSyntheticDb(db);

    CHECK(db.valid());
    CHECK(db.vocab_size == 268);
    CHECK(db.merge_count == 9);
    CHECK(db.bos_id == 0);
    CHECK(db.eos_id == 1);
    CHECK(db.pad_id == 2);

    CHECK(db.byte_to_token_id[0x20] == 35);
    CHECK(db.byte_to_token_id[0x78] == 123);
    CHECK(db.byte_to_token_id[0x68] == 107);

    CHECK(db.token_offsets != nullptr);
    CHECK(db.token_bytes != nullptr);
    CHECK(db.token_offsets[35 + 1] - db.token_offsets[35] == 1);
    CHECK(db.token_bytes[db.token_offsets[35]] == 0x20);

    CHECK(db.is_special[0] == 1);
    CHECK(db.is_special[1] == 1);
    CHECK(db.is_special[2] == 1);
    CHECK(db.is_special[35] == 0);

    CHECK(db.merge_left != nullptr);
    CHECK(db.merge_right != nullptr);
    CHECK(db.merge_rank != nullptr);
    CHECK(db.merge_token_id != nullptr);
    // merges sorted by (left,right): row 0 is "\u0120 w" = (35,122), rank 4
    // (fixture merge #5); "h e" = (107,104) is rank 0 (fixture merge #1)
    CHECK(db.merge_left[0] == 35);
    CHECK(db.merge_right[0] == 122);
    CHECK(db.merge_rank[0] == 4);
    CHECK(db.merge_token_id[0] == 263);
    int he_idx = -1;
    for (int i = 0; i < db.merge_count; ++i) {
        if (db.merge_left[i] == 107 && db.merge_right[i] == 104) { he_idx = i; break; }
    }
    REQUIRE(he_idx >= 0);
    CHECK(db.merge_rank[he_idx] == 0);
    CHECK(db.merge_token_id[he_idx] == 259);

    lsxcommon::ArrowTokenizerDb bad;
    CHECK_FALSE(bad.valid());

    db = lsxcommon::ArrowTokenizerDb{};   // release mmap handles before deleting files
    fs::remove_all(pinned);
}

#include <lsxcommon/pretokenizer.h>
#include <vector>
#include <string>

namespace {

std::vector<std::string> SplitWords(const std::string &text)
{
    lsxcommon::WordStream ws(text.data(), text.size());
    std::vector<std::string> out;
    lsxcommon::WordSpan w;
    while (ws.next(w))
        out.emplace_back(w.begin, w.length);
    CHECK(ws.error() == 0);
    return out;
}

} // namespace

TEST_CASE("WordStream GPT-2 word splits", "[tokenizer][pretokenizer]")
{
    using V = std::vector<std::string>;
    CHECK(SplitWords("hello world") == V{"hello", " world"});
    CHECK(SplitWords("  x") == V{" ", " x"});
    CHECK(SplitWords("a  b") == V{"a", " ", " b"});
    CHECK(SplitWords(" a") == V{" a"});
    CHECK(SplitWords("a ") == V{"a", " "});
    CHECK(SplitWords("trailing  ") == V{"trailing", "  "});
    CHECK(SplitWords("x  y  z") == V{"x", " ", " y", " ", " z"});
    CHECK(SplitWords("  leading spaces") == V{" ", " leading", " spaces"});
    CHECK(SplitWords("a b") == V{"a", " b"});
}

TEST_CASE("WordStream CRLF, numbers, CJK, gaps", "[tokenizer][pretokenizer]")
{
    using V = std::vector<std::string>;
    CHECK(SplitWords("\r\n") == V{"\r\n"});
    CHECK(SplitWords("\n\n") == V{"\n\n"});
    CHECK(SplitWords("1abc") == V{"1", "abc"});
    CHECK(SplitWords("a中b") == V{"a", "中", "b"});
    CHECK(SplitWords("12345") == V{"123", "45"});
    CHECK(SplitWords("a\x01") == V{"a", "\x01"});
    CHECK(SplitWords("a\x01\x02b") == V{"a", "\x01", "\x02b"});
    CHECK(SplitWords("!abc") == V{"!abc"});
    CHECK(SplitWords("a?x") == V{"a", "?x"});
    CHECK(SplitWords("a !") == V{"a", " !"});
}

TEST_CASE("WordStream rejects malformed UTF-8", "[tokenizer][pretokenizer]")
{
    lsxcommon::WordStream ws("a\xffz", 3);
    lsxcommon::WordSpan w;
    REQUIRE(ws.next(w));                     // "a"
    CHECK(std::string(w.begin, w.length) == "a");
    CHECK_FALSE(ws.next(w));                 // stops at bad byte
    CHECK(ws.error() != 0);
}

TEST_CASE("arrow_tokenizer_encode synthetic goldens", "[tokenizer][encode]")
{
    lsxcommon::ArrowTokenizerDb db;
    auto pinned = BuildSyntheticDb(db);

    struct Case { const char *text; std::vector<int32_t> ids; };
    const Case cases[] = {
        {"hello world", {0, 262, 267}},
        {"hello", {0, 262}},
        {"  x", {0, 35, 35, 123}},
        {" a", {0, 35, 100}},
        {"a ", {0, 100, 35}},
        {"a b", {0, 100, 35, 101}},
        {"a  b", {0, 100, 35, 35, 101}},
        {"x  y  z", {0, 123, 35, 35, 124, 35, 35, 125}},
        {"trailing  ", {0, 119, 117, 100, 108, 111, 108, 113, 106, 35, 35}},
        {"  leading spaces",
         {0, 35, 35, 111, 104, 100, 103, 108, 113, 106, 35, 118, 115, 100, 102, 104, 118}},
    };
    for (const auto &c : cases) {
        std::vector<int32_t> out(64, -7);
        size_t cap = out.size();
        int32_t n = lsxcommon::arrow_tokenizer_encode(
            db, c.text, std::strlen(c.text), out.data(), &cap);
        REQUIRE(n == static_cast<int32_t>(c.ids.size()));
        CHECK(cap == c.ids.size());
        CHECK(std::equal(out.begin(), out.begin() + n, c.ids.begin()));
    }

    db = lsxcommon::ArrowTokenizerDb{};   // release mmap handles before deleting files
    fs::remove_all(pinned);
}

TEST_CASE("arrow_tokenizer_decode round-trip and specials", "[tokenizer][decode]")
{
    lsxcommon::ArrowTokenizerDb db;
    auto pinned = BuildSyntheticDb(db);

    // round-trip: bos (special) is skipped, raw bytes come back
    {
        std::vector<int32_t> ids(64, -7);
        size_t cap = ids.size();
        int32_t n = lsxcommon::arrow_tokenizer_encode(
            db, "hello world", 11, ids.data(), &cap);
        REQUIRE(n == 3);
        std::string buf(64, '\0');
        size_t ocap = buf.size();
        int32_t r = lsxcommon::arrow_tokenizer_decode(
            db, ids.data(), static_cast<size_t>(n), buf.data(), &ocap);
        CHECK(r == 0);
        CHECK(ocap == 11);
        CHECK(std::string(buf.data(), ocap) == "hello world");
    }

    // special token in the middle is skipped
    {
        const int32_t ids[] = {100, 0, 101};
        std::string buf(64, '\0');
        size_t ocap = buf.size();
        int32_t r = lsxcommon::arrow_tokenizer_decode(db, ids, 3, buf.data(), &ocap);
        CHECK(r == 0);
        CHECK(ocap == 2);
        CHECK(std::string(buf.data(), ocap) == "ab");
    }

    // two spaces decode back to "  x"
    {
        const int32_t ids[] = {0, 35, 35, 123};
        std::string buf(64, '\0');
        size_t ocap = buf.size();
        int32_t r = lsxcommon::arrow_tokenizer_decode(db, ids, 4, buf.data(), &ocap);
        CHECK(r == 0);
        CHECK(ocap == 3);
        CHECK(std::string(buf.data(), ocap) == "  x");
    }

    db = lsxcommon::ArrowTokenizerDb{};   // release mmap handles before deleting files
    fs::remove_all(pinned);
}

TEST_CASE("arrow_tokenizer_decode error codes", "[tokenizer][decode]")
{
    lsxcommon::ArrowTokenizerDb db;
    auto pinned = BuildSyntheticDb(db);

    // capacity too small -> -2, reports required count
    {
        const int32_t ids[] = {0, 262, 267};
        std::string buf(2, '\0');
        size_t ocap = buf.size();
        int32_t r = lsxcommon::arrow_tokenizer_decode(db, ids, 3, buf.data(), &ocap);
        CHECK(r == -2);
        CHECK(ocap == 11);
    }

    // out-of-range id -> -1
    {
        const int32_t ids[] = {999};
        char buf[16];
        size_t ocap = 16;
        CHECK(lsxcommon::arrow_tokenizer_decode(db, ids, 1, buf, &ocap) == -1);
    }

    // default-constructed db -> -1
    {
        lsxcommon::ArrowTokenizerDb bad;
        const int32_t ids[] = {0};
        char buf[16];
        size_t ocap = 16;
        CHECK(lsxcommon::arrow_tokenizer_decode(bad, ids, 1, buf, &ocap) == -1);
    }

    db = lsxcommon::ArrowTokenizerDb{};   // release mmap handles before deleting files
    fs::remove_all(pinned);
}

TEST_CASE("arrow_tokenizer_encode error codes", "[tokenizer][encode]")
{
    lsxcommon::ArrowTokenizerDb db;
    auto pinned = BuildSyntheticDb(db);

    // capacity too small -> -2, *max_out_len reports the required count
    std::vector<int32_t> small(2, -7);
    size_t cap = small.size();
    int32_t r = lsxcommon::arrow_tokenizer_encode(db, "hello world", 11, small.data(), &cap);
    CHECK(r == -2);
    CHECK(cap == 3);

    // a word longer than kMaxWordBytes -> -3 (single L-run word of 600 'a')
    std::string long_word(600, 'a');
    std::vector<int32_t> big(700, -7);
    cap = big.size();
    r = lsxcommon::arrow_tokenizer_encode(db, long_word.data(), long_word.size(), big.data(), &cap);
    CHECK(r == -3);

    // default-constructed db -> -1
    lsxcommon::ArrowTokenizerDb bad;
    std::vector<int32_t> out(16, -7);
    cap = out.size();
    r = lsxcommon::arrow_tokenizer_encode(bad, "x", 1, out.data(), &cap);
    CHECK(r == -1);

    // active-db overload
    lsxcommon::set_active_tokenizer_db(&db);
    std::vector<int32_t> out2(16, -7);
    cap = out2.size();
    r = lsxcommon::arrow_tokenizer_encode("hello", 5, out2.data(), &cap);
    CHECK(r == 2);
    CHECK(out2[0] == 0);
    CHECK(out2[1] == 262);
    lsxcommon::set_active_tokenizer_db(nullptr);
    cap = out2.size();
    r = lsxcommon::arrow_tokenizer_encode("hello", 5, out2.data(), &cap);
    CHECK(r == -1);

    db = lsxcommon::ArrowTokenizerDb{};   // release mmap handles before deleting files
    fs::remove_all(pinned);
}
