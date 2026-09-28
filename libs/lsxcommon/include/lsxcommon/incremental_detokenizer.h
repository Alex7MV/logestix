#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

/// Converts a token-id stream into UTF-8 text deltas without ever splitting a
/// multibyte codepoint across emitted chunks.
///
/// The decoder is injected (default `ArrowDecoder`) so the boundary logic is
/// unit-testable without a real vocabulary. Each Push re-decodes the full id
/// list (matching `runtime::DecodeGeneratedText`) and emits only the newly
/// completed, UTF-8-valid suffix; incomplete trailing bytes are withheld until
/// later tokens complete them (or Flush releases them).
class IncrementalDetokenizer {
public:
    using DecodeFn = std::function<std::string(const std::vector<std::int32_t>&)>;

    explicit IncrementalDetokenizer(DecodeFn decode);

    /// Decoder backed by the real Arrow tokenizer (specials skipped, raw bytes
    /// appended) with capacity retry — same semantics as DecodeGeneratedText.
    static DecodeFn ArrowDecoder(const ArrowTokenizerDb& db);

    /// Appends @p id and returns the new UTF-8-safe text suffix (possibly empty).
    std::string Push(std::int32_t id);

    /// Releases any withheld bytes at end of stream.
    std::string Flush();

private:
    DecodeFn decode_;
    std::vector<std::int32_t> ids_;
    std::string emitted_;
};

}  // namespace lsxcommon
