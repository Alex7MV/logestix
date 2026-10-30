#pragma once

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
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
    // 0 = gpt2 byte-level BPE (default), 1 = sentencepiece BPE,
    // 2 = tiktoken byte-level BPE (legacy GPT-2/cl100k pre-tokenizer),
    // 3 = byte-level BPE with the Llama-3/o200k pre-tokenizer (same BPE table,
    //     different split: case runs instead of a single letter run, no CJK
    //     split). MiniMax-M3, Qwen2/3 and Llama-3 checkpoints use this one.
    int32_t tokenizer_kind = 0;
    // tokenizer_kind == 2 only. Ids >= base_vocab_size are added tokens that
    // the reference encoder matches literally (tiktoken allowed_special="all");
    // literal_ids holds them longest-text-first, and rank_by_bytes is the
    // vocabulary itself acting as the merge table (pair bytes -> id).
    // std:: containers, not absl: this header is reachable from .cu TUs.
    int32_t base_vocab_size = 0;
    std::vector<int32_t> literal_ids;
    std::unordered_map<std::string_view, int32_t> rank_by_bytes;
    int32_t unk_id = 0;
    int32_t byte_fallback = 0;
    int32_t add_dummy_prefix = 1;
    int32_t escape_whitespaces = 1;
    // 1 = bytes whose byte_to_token_id is -1 are skipped by the encoder
    // (reference-faithful for checkpoints that never emit them; Hy-Memory).
    int32_t drop_unmapped_bytes = 0;
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

// Returns the token id whose raw bytes equal `content[0..len)`, or -1 when not
// found (or the db is invalid). Linear scan over the vocab; use for a handful
// of lookups at init time (e.g. resolving chat-template special tokens).
int32_t arrow_tokenizer_lookup(const ArrowTokenizerDb &db, const char *content,
                               size_t len);

} // namespace lsxcommon
