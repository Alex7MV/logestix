#include "lsxcommon/incremental_detokenizer.h"

#include "lsxcommon/pretokenizer.h"

#include <cstddef>
#include <string_view>
#include <utility>

namespace lsxcommon {
namespace {

// U+FFFD REPLACEMENT CHARACTER. Anything the decoder could not turn into a
// well-formed sequence becomes one of these, so text handed to a caller is
// always something arrow_tokenizer_encode will accept again. Dropping the
// bytes instead would silently truncate an answer; passing them through makes
// BuildConversationInputIds fail and drops the next turn.
constexpr char kReplacement[] = "\xEF\xBF\xBD";

// Length of the prefix of `s` that is fully decided: it ends on a codepoint
// boundary and every maximal invalid subpart inside it can already be resolved.
// The remainder is a trailing sequence that is merely incomplete, so more bytes
// may still complete it and it must be withheld.
std::size_t DecidedPrefixLen(std::string_view s) {
    std::size_t i = 0;
    while (i < s.size()) {
        const auto b0 = static_cast<unsigned char>(s[i]);
        if (b0 < 0x80u) { ++i; continue; }
        const std::size_t need = utf8_expected_len(b0);
        if (need == 0) { ++i; continue; }              // cannot start a sequence
        if (i + need > s.size()) {
            // Truncated by the end of the buffer. Decided only when one of the
            // bytes we do have already breaks the sequence; otherwise more
            // bytes may still complete it, so it stays withheld.
            bool broken = false;
            for (std::size_t k = 1; k < s.size() - i; ++k) {
                if ((static_cast<unsigned char>(s[i + k]) & 0xC0u) != 0x80u) {
                    broken = true;
                    break;
                }
            }
            if (!broken) return i;
            ++i;                                  // broken subpart: decided
            continue;
        }
        std::size_t len = 0;
        if (decode_utf8(s.data() + i, need, &len) == kInvalidUtf8) { ++i; continue; }
        i += len;
    }
    return i;
}

// Replaces every maximal invalid subpart of a decided region with U+FFFD.
// Must agree with DecidedPrefixLen on where sequences start and end.
std::string SanitizeUtf8(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    std::size_t i = 0;
    while (i < s.size()) {
        const auto b0 = static_cast<unsigned char>(s[i]);
        if (b0 < 0x80u) { out.push_back(static_cast<char>(b0)); ++i; continue; }
        const std::size_t need = utf8_expected_len(b0);
        std::size_t len = 0;
        if (need != 0 && i + need <= s.size() &&
            decode_utf8(s.data() + i, need, &len) != kInvalidUtf8) {
            out.append(s, i, len);
            i += len;
            continue;
        }
        // The byte that follows the maximal invalid subpart is reprocessed by
        // the next iteration, per the Unicode "maximal subpart" rule.
        out.append(kReplacement, 3);
        ++i;
    }
    return out;
}

}  // namespace

IncrementalDetokenizer::IncrementalDetokenizer(DecodeFn decode)
    : decode_(std::move(decode)) {}

IncrementalDetokenizer::DecodeFn IncrementalDetokenizer::ArrowDecoder(
    const ArrowTokenizerDb& db) {
    return [&db](const std::vector<std::int32_t>& ids) -> std::string {
        if (ids.empty()) return {};
        std::size_t cap = 4096;
        std::string out(cap, '\0');
        for (int attempt = 0; attempt < 8; ++attempt) {
            const int rc = arrow_tokenizer_decode(db, ids.data(), ids.size(),
                                                  out.data(), &cap);
            if (rc == 0) {
                out.resize(cap);
                return out;
            }
            if (rc == -2 && cap > out.size() && cap <= (1u << 20)) {
                out.resize(cap);
                continue;
            }
            break;
        }
        return {};
    };
}

std::string IncrementalDetokenizer::Push(std::int32_t id) {
    ids_.push_back(id);
    const std::string full = decode_ ? decode_(ids_) : std::string{};
    const std::size_t decided = DecidedPrefixLen(full);
    if (decided < consumed_) {
        // Decoder output is expected monotonic; resync from scratch.
        consumed_ = 0;
        emitted_.clear();
    }
    if (decided <= consumed_) return {};
    const std::string delta =
        SanitizeUtf8(std::string_view(full).substr(consumed_, decided - consumed_));
    consumed_ = decided;
    emitted_ += delta;
    return delta;
}

std::string IncrementalDetokenizer::Flush() {
    const std::string full = decode_ ? decode_(ids_) : std::string{};
    // Nothing more will arrive, so the trailing sequence — incomplete or not —
    // is final: resolve it rather than emit raw bytes.
    const std::string delta =
        SanitizeUtf8(std::string_view(full).substr(std::min(consumed_, full.size())));
    consumed_ = full.size();
    emitted_ += delta;
    return delta;
}

}  // namespace lsxcommon
