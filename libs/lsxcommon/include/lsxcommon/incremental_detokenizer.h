#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

/// Converts a token-id stream into UTF-8 text deltas without ever splitting a
/// multibyte codepoint across emitted chunks, and without ever emitting a
/// byte sequence the tokenizer could not encode again.
///
/// The decoder is injected (default `ArrowDecoder`) so the boundary logic is
/// unit-testable without a real vocabulary. Each Push re-decodes the full id
/// list (matching `runtime::DecodeGeneratedText`) and emits only the newly
/// decided, UTF-8-valid suffix; an incomplete trailing sequence is withheld
/// until later tokens complete it. Bytes that cannot form a well-formed
/// sequence even when complete (overlong forms, surrogates, a lone byte from a
/// byte-fallback token) are emitted as U+FFFD, so the caller always receives
/// text that `arrow_tokenizer_encode` accepts — otherwise framing the next turn
/// of a chat fails and the question is silently dropped.
class IncrementalDetokenizer {
public:
    using DecodeFn = std::function<std::string(const std::vector<std::int32_t>&)>;

    explicit IncrementalDetokenizer(DecodeFn decode);

    /// Decoder backed by the real Arrow tokenizer (specials skipped, raw bytes
    /// appended) with capacity retry — same semantics as DecodeGeneratedText.
    static DecodeFn ArrowDecoder(const ArrowTokenizerDb& db);

    /// Appends @p id and returns the new UTF-8-safe text suffix (possibly empty).
    std::string Push(std::int32_t id);

    /// Resolves and releases any withheld bytes at end of stream.
    std::string Flush();

private:
    DecodeFn decode_;
    std::vector<std::int32_t> ids_;
    std::string emitted_;
    /// Bytes of the decoder output already turned into emitted_ text. Tracked
    /// separately because substitution makes emitted_ longer than the input.
    std::size_t consumed_ = 0;
};

}  // namespace lsxcommon
