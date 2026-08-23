#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/converter.h"
#include "lsxcommon/schema.h"

#include <arrow/io/file.h>
#include <arrow/record_batch.h>

#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

const char *kB64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string B64Encode(const uint8_t *p, size_t n)
{
    std::string out;
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = static_cast<uint32_t>(p[i]) << 16;
        out += kB64[(v >> 18) & 63];
        if (i + 1 < n) {
            v |= static_cast<uint32_t>(p[i + 1]) << 8;
            out += kB64[(v >> 12) & 63];
            if (i + 2 < n) {
                v |= p[i + 2];
                out += kB64[(v >> 6) & 63];
                out += kB64[v & 63];
            } else {
                out += kB64[(v >> 6) & 63];
                out += '=';
            }
        } else {
            out += kB64[(v >> 12) & 63];
            out += "==";
        }
    }
    return out;
}

// Writes a tiktoken file with the complete single-byte alphabet (ids 0..255)
// plus one merged token "!\"" (id 256). Returns its path.
inline std::string WriteByteAlphabetTikToken(const std::filesystem::path &dir)
{
    auto path = dir / "tiktoken.model";
    std::ofstream f(path);
    for (int b = 0; b < 256; ++b) {
        uint8_t byte = static_cast<uint8_t>(b);
        f << B64Encode(&byte, 1) << " " << b << "\n";
    }
    const uint8_t pair[2] = {0x21, 0x22};   // '!"'
    f << B64Encode(pair, 2) << " 256\n";
    f.close();
    return path.string();
}

} // namespace

using namespace lsxcommon;

TEST_CASE("BuildTokenizerArrowFilesTikToken builds vocab+merges",
          "[kimi][tokenizer]")
{
    namespace fs = std::filesystem;
    auto dir = fs::temp_directory_path() / "kimi_tok_fixture";
    auto pinned = dir / "pinned";
    fs::remove_all(dir);
    fs::create_directories(pinned);

    auto model_path = WriteByteAlphabetTikToken(dir);

    // Specials: [BOS]=257, [EOS]=258, [PAD]=259.
    {
        std::ofstream c(dir / "tokenizer_config.json");
        c << "{\"bos_token\":\"[BOS]\",\"eos_token\":\"[EOS]\","
             "\"pad_token\":\"[PAD]\",\"added_tokens_decoder\":{"
             "\"257\":{\"content\":\"[BOS]\",\"special\":true},"
             "\"258\":{\"content\":\"[EOS]\",\"special\":true},"
             "\"259\":{\"content\":\"[PAD]\",\"special\":true}}}";
    }

    auto st = BuildTokenizerArrowFilesTikToken(
        model_path, (dir / "tokenizer_config.json").string(), pinned);
    if (!st.ok()) FAIL(st.ToString());
    REQUIRE(st.ok());

    // Vocab: ids 0..259 (base 0..256 + specials 257..259), no gaps.
    auto vres = ReadRecordBatchFromFile(pinned / "tokenizer_vocab.arrow");
    REQUIRE(vres.ok());
    auto vrows = TokenizerVocabSchema::UnpackBatch(*vres);
    REQUIRE(vrows.ok());
    REQUIRE(vrows->size() == 260);

    const auto &rows = *vrows;
    REQUIRE(rows[0x21].token_bytes == std::vector<uint8_t>{0x21});
    REQUIRE(rows[0x22].token_bytes == std::vector<uint8_t>{0x22});
    REQUIRE(rows[256].token_bytes ==
            std::vector<uint8_t>{0x21, 0x22});
    REQUIRE(rows[257].is_special == 1);   // [BOS]
    REQUIRE(rows[258].is_special == 1);   // [EOS]
    REQUIRE(rows[259].is_special == 1);   // [PAD]

    // Meta: specials wired from tokenizer_config.json; byte table complete.
    {
        auto meta = (*vres)->schema()->metadata();
        REQUIRE(meta != nullptr);
        auto get_i32 = [&](const char *key) -> int32_t {
            auto s = meta->Get(key);
            int32_t v = -999;
            if (!s.ok()) return v;
            (void)absl::SimpleAtoi(*s, &v);
            return v;
        };
        REQUIRE(get_i32("bos_id") == 257);
        REQUIRE(get_i32("eos_id") == 258);
        REQUIRE(get_i32("pad_id") == 259);
        REQUIRE(get_i32("vocab_size") == 260);

        // byte_to_token_id metadata: comma-joined 256 i32 values.
        auto b2t = meta->Get("byte_to_token_id");
        REQUIRE(b2t.ok());
        if (b2t.ok()) {
            std::vector<int32_t> table;
            size_t start = 0;
            const std::string &raw = *b2t;
            while (start <= raw.size()) {
                size_t comma = raw.find(',', start);
                if (comma == std::string::npos) comma = raw.size();
                int32_t v = -1;
                REQUIRE(absl::SimpleAtoi(raw.substr(start, comma - start),
                                         &v));
                table.push_back(v);
                start = comma + 1;
            }
            REQUIRE(table.size() == 256);
            REQUIRE(table[0x21] == 0x21);
            REQUIRE(table[0x22] == 0x22);
        }
    }

    // Merges: exactly one reconstructed merge '!' + '"' -> '!"'.
    auto mres = ReadRecordBatchFromFile(pinned / "tokenizer_merges.arrow");
    REQUIRE(mres.ok());
    auto mrows = TokenizerMergesSchema::UnpackBatch(*mres);
    REQUIRE(mrows.ok());
    REQUIRE(mrows->size() == 1);
    REQUIRE((*mrows)[0].merge_left == 0x21);
    REQUIRE((*mrows)[0].merge_right == 0x22);
    REQUIRE((*mrows)[0].merge_rank == 0);
    REQUIRE((*mrows)[0].merge_token_id == 256);
}

TEST_CASE("BuildTokenizerArrowFilesTikToken fills sparse id gaps",
          "[kimi][tokenizer]")
{
    namespace fs = std::filesystem;
    auto dir = fs::temp_directory_path() / "kimi_tok_gap";
    auto pinned = dir / "pinned";
    fs::remove_all(dir);
    fs::create_directories(pinned);

    // Base ids 0..256 (full alphabet + merge), special at id 300 -> gap
    // 257..299 must be filled with placeholder rows instead of erroring
    // (real Kimi vocab is sparse between base ranks and high special ids).
    (void)WriteByteAlphabetTikToken(dir);
    {
        std::ofstream c(dir / "tokenizer_config.json");
        c << "{\"bos_token\":\"[B]\",\"added_tokens_decoder\":{"
             "\"300\":{\"content\":\"[B]\",\"special\":true}}}";
    }

    auto st = BuildTokenizerArrowFilesTikToken(
        (dir / "tiktoken.model").string(),
        (dir / "tokenizer_config.json").string(), pinned);
    if (!st.ok()) FAIL(st.ToString());
    REQUIRE(st.ok());

    auto vres = ReadRecordBatchFromFile(pinned / "tokenizer_vocab.arrow");
    REQUIRE(vres.ok());
    auto vrows = TokenizerVocabSchema::UnpackBatch(*vres);
    REQUIRE(vrows.ok());
    REQUIRE(vrows->size() == 301);   // ids 0..300
}
