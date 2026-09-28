#include "lsxcommon/incremental_detokenizer.h"

#include <cstddef>
#include <utility>

namespace lsxcommon {
namespace {

// Length of the longest prefix of `s` that is a sequence of complete UTF-8
// codepoints. Returns s.size() when the whole string is valid.
std::size_t ValidUtf8Prefix(const std::string& s) {
    std::size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        std::size_t need = 0;
        if (c < 0x80u) {
            need = 1;
        } else if ((c & 0xE0u) == 0xC0u) {
            need = 2;
        } else if ((c & 0xF0u) == 0xE0u) {
            need = 3;
        } else if ((c & 0xF8u) == 0xF0u) {
            need = 4;
        } else {
            return i;  // invalid lead byte: stop here
        }
        if (i + need > s.size()) return i;  // incomplete: withhold
        for (std::size_t k = 1; k < need; ++k) {
            if ((static_cast<unsigned char>(s[i + k]) & 0xC0u) != 0x80u)
                return i;  // malformed continuation
        }
        i += need;
    }
    return i;
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
    const std::size_t valid_len = ValidUtf8Prefix(full);
    if (valid_len <= emitted_.size()) return {};
    if (full.compare(0, emitted_.size(), emitted_) != 0) {
        // Decoder output is expected monotonic; resync by emitting the valid
        // prefix from scratch.
        emitted_ = full.substr(0, valid_len);
        return emitted_;
    }
    std::string delta = full.substr(emitted_.size(), valid_len - emitted_.size());
    emitted_ = full.substr(0, valid_len);
    return delta;
}

std::string IncrementalDetokenizer::Flush() {
    const std::string full = decode_ ? decode_(ids_) : std::string{};
    if (full.size() <= emitted_.size()) return {};
    std::string delta = full.substr(emitted_.size());
    emitted_ = full;
    return delta;
}

}  // namespace lsxcommon
