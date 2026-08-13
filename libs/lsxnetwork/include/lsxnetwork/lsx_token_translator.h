#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lsxnetwork {

namespace token_passport {

/// Current passport format revision.
constexpr std::uint32_t kVersion = 1;
/// Byte signature "LSXTPASS" read as a little-endian uint64.
constexpr std::uint64_t kMagic = 0x535341505458534CULL;

/// Fixed 64-byte (one cache line) header. All translation index arrays live in
/// the file at 64-byte-offset boundaries, so their mapped access pointers are
/// always cache-line aligned regardless of the backing page base.
struct alignas(64) PassportHeader {
    std::uint64_t magic = kMagic;
    std::uint32_t version = kVersion;
    std::uint32_t glm_vocab_size = 0;       // rows of glm_to_deepseek[]
    std::uint32_t deepseek_vocab_size = 0;  // rows of deepseek_to_glm[]
    std::uint32_t glm_unk_id = 0;
    std::uint32_t deepseek_unk_id = 0;
    std::uint32_t reserved[9] = {};         // pad header to a full cache line
};
static_assert(sizeof(PassportHeader) == 64,
              "PassportHeader must occupy exactly one cache line");
static_assert(alignof(PassportHeader) == 64,
              "PassportHeader must be cache-line aligned");

}  // namespace token_passport

/// Inputs for offline passport generation. Each vector slot is one token whose
/// index equals its tokenizer id: vector[i] is the byte string of token id i.
struct BuildPassportOptions {
    std::vector<std::string> glm_tokens;
    std::vector<std::string> deepseek_tokens;
    std::uint32_t glm_unk_id = 0;
    std::uint32_t deepseek_unk_id = 0;
    std::uint16_t lmcs_min_overlap = 2;  // substring hits below this -> unk
};

/// Builds the flat, contiguous, 64-byte cache-line aligned binary passport in
/// host memory. Cleanup responsibility of the caller.
std::vector<std::byte> BuildPassport(const BuildPassportOptions& opts);

/// The passport is written/read from the unified vault by default; callers
/// resolve the location via lsxcommon::lsx_model_metadata("passport/passport.bin").
/// Writes the generated passport to @p path. Returns 0 on success, -1 on I/O
/// failure.
int WritePassport(const char* path, const BuildPassportOptions& opts);

/// Runtime zero-copy translator. Maps a generated `.bin` passport file into
/// virtual memory via Win32 CreateFileMapping/MapViewOfFile (native mmap on
/// Linux). The translation fast path performs zero heap allocations.
class LsxTokenTranslator {
public:
    LsxTokenTranslator() noexcept = default;
    ~LsxTokenTranslator() noexcept { Unmap(); }

    LsxTokenTranslator(LsxTokenTranslator&& o) noexcept;
    LsxTokenTranslator& operator=(LsxTokenTranslator&& o) noexcept;
    LsxTokenTranslator(const LsxTokenTranslator&) = delete;
    LsxTokenTranslator& operator=(const LsxTokenTranslator&) = delete;

    /// Maps @p path into virtual memory and validates the passport header.
    /// Returns 0 on success, -1 on any I/O or format error.
    int Load(const char* path) noexcept;
    void Reset() noexcept { Unmap(); }

    /// True once a valid passport is mapped and validated.
    bool Valid() const noexcept { return glm_to_ds_ != nullptr; }

    std::uint32_t GlmVocabSize() const noexcept { return glm_vocab_; }
    std::uint32_t DeepSeekVocabSize() const noexcept { return ds_vocab_; }
    std::uint32_t GlmUnkId() const noexcept { return glm_unk_; }
    std::uint32_t DeepSeekUnkId() const noexcept { return ds_unk_; }

    /// Cache-line aligned, read-only views into the mapped index arrays.
    /// The tight 64-byte alignment lives on the backing member pointers, so the
    /// returned pointers are always a cache-line multiple in a live mapping.
    const std::uint32_t* GlmToDeepSeek() const noexcept { return glm_to_ds_; }
    const std::uint32_t* DeepSeekToGlm() const noexcept { return ds_to_glm_; }

    /// O(1) zero-copy single-token lookup. Out-of-range ids fall back to the
    /// remote model's unknown-token id. noexcept and heap-allocation-free.
    std::uint32_t TranslateGlmToDeepSeek(std::uint32_t token_id) const noexcept {
        return (token_id < glm_vocab_ && glm_to_ds_) ? glm_to_ds_[token_id] : ds_unk_;
    }
    std::uint32_t TranslateDeepSeekToGlm(std::uint32_t token_id) const noexcept {
        return (token_id < ds_vocab_ && ds_to_glm_) ? ds_to_glm_[token_id] : glm_unk_;
    }

private:
    void Unmap() noexcept;

    alignas(64) const std::uint32_t* glm_to_ds_ = nullptr;
    alignas(64) const std::uint32_t* ds_to_glm_ = nullptr;
    std::uint32_t glm_vocab_ = 0;
    std::uint32_t ds_vocab_ = 0;
    std::uint32_t glm_unk_ = 0;
    std::uint32_t ds_unk_ = 0;

    /// Win: CreateFileMapping HANDLE (also the mmap region length carrier on
    /// Linux when nonzero). base_: MapViewOfFile/mmap base. len_: mapped length.
    void* handle_ = nullptr;
    void* base_ = nullptr;
    std::size_t len_ = 0;
};

}  // namespace lsxnetwork