// tensor_index_parse.h — non-throwing parse of a tensor-name numeric index.
// Shared by the per-family converters: a malformed checkpoint name (empty or
// non-numeric layer/expert index) must classify as Unknown and fail cleanly,
// never throw std::invalid_argument/std::out_of_range out of the converter.
#pragma once

#include <string_view>

namespace lsxcommon {
namespace detail {

// Parses a non-empty run of ASCII digits into a bounded non-negative int.
// Returns false on empty input, a non-digit byte, or an implausible magnitude
// (defensive cap against a corrupted name overflowing int). Never throws.
inline bool ParseTensorIndex(std::string_view digits, int &out)
{
    if (digits.empty()) return false;
    int v = 0;
    for (char c : digits) {
        if (c < '0' || c > '9') return false;
        v = v * 10 + (c - '0');
        if (v > 1000000) return false;
    }
    out = v;
    return true;
}

} // namespace detail
} // namespace lsxcommon
