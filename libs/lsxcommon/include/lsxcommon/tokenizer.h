#pragma once

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lsxcommon {

constexpr size_t kMaxWordBytes = 512;

struct ArrowTokenizerDb {
    std::shared_ptr<arrow::io::MemoryMappedFile> vocab_mmap;
    std::shared_ptr<arrow::io::MemoryMappedFile> merges_mmap;
    const uint8_t *vocab_file_base = nullptr;   // ptr − file_base offset math
    int64_t vocab_file_size = 0;

    const int64_t *token_offsets = nullptr;     // LargeBinary offsets, n = vocab_size + 1
    const uint8_t *token_bytes = nullptr;       // value buffer base
    const uint8_t *is_special = nullptr;        // Int8 column
    int32_t vocab_size = 0;

    const int32_t *merge_left = nullptr;
    const int32_t *merge_right = nullptr;
    const int32_t *merge_rank = nullptr;
    const int32_t *merge_token_id = nullptr;
    int32_t merge_count = 0;

    int32_t bos_id = -1, eos_id = -1, pad_id = -1;
    int32_t byte_to_token_id[256];
    std::string error_msg;

    bool valid() const;
};

arrow::Status load_tokenizer_metadata(
    const std::shared_ptr<arrow::io::MemoryMappedFile> &vocab_mmap,
    const std::shared_ptr<arrow::io::MemoryMappedFile> &merges_mmap,
    ArrowTokenizerDb &db);

void set_active_tokenizer_db(const ArrowTokenizerDb *db);

// Returns token count on success. Negative on error:
//   -1 invalid db / no active db, -2 output capacity too small,
//   -3 a word exceeds kMaxWordBytes. *max_out_len is capacity in, count out.
int32_t arrow_tokenizer_encode(const char *text, size_t text_len,
                               int32_t *out_tokens, size_t *max_out_len);

int32_t arrow_tokenizer_encode(const ArrowTokenizerDb &db, const char *text,
                               size_t text_len, int32_t *out_tokens,
                               size_t *max_out_len);

// Returns 0 on success (specials skipped, raw bytes appended),
// -1 invalid db, -2 output capacity too small. *max_out_len is in/out.
int32_t arrow_tokenizer_decode(const ArrowTokenizerDb &db, const int32_t *ids,
                               size_t n_ids, char *out_text,
                               size_t *max_out_len);

} // namespace lsxcommon
