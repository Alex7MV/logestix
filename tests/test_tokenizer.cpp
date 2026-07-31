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
