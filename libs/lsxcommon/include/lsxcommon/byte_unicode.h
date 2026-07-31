#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace lsxcommon {
namespace byte_unicode {

// GPT-2 byte→Unicode table (standard construction algorithm):
// bytes 33..126, 161..172, 174..255 self-map; the remaining 68 bytes map to
// U+0100..U+0143 in order of first appearance.
inline const std::array<char32_t, 256> &gpt2_byte_table()
{
    static const std::array<char32_t, 256> t = [] {
        std::array<char32_t, 256> m{};
        for (int i = 33; i <= 126; ++i)  m[static_cast<size_t>(i)] = static_cast<char32_t>(i);
        for (int i = 161; i <= 172; ++i) m[static_cast<size_t>(i)] = static_cast<char32_t>(i);
        for (int i = 174; i <= 255; ++i) m[static_cast<size_t>(i)] = static_cast<char32_t>(i);
        int n = 0;
        for (int b = 0; b < 256; ++b)
            if (m[static_cast<size_t>(b)] == 0)
                m[static_cast<size_t>(b)] = static_cast<char32_t>(0x100 + n++);
        return m;
    }();
    return t;
}

inline char32_t byte_to_unicode(int b)
{
    return gpt2_byte_table()[static_cast<size_t>(b) & 0xFF];
}

// Returns 0..255, or -1 if `cp` is not a byte-level char.
inline int unicode_to_byte(char32_t cp)
{
    const auto &t = gpt2_byte_table();
    for (int b = 0; b < 256; ++b)
        if (t[static_cast<size_t>(b)] == cp) return b;
    return -1;
}

// Encode raw bytes into the byte-unicode key string (UTF-8).
inline std::string bytes_to_key(const uint8_t *data, size_t n)
{
    std::string s;
    s.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        char32_t c = byte_to_unicode(data[i]);
        if (c < 0x80) {
            s.push_back(static_cast<char>(c));
        } else if (c < 0x800) {
            s.push_back(static_cast<char>(0xC0 | (c >> 6)));
            s.push_back(static_cast<char>(0x80 | (c & 0x3F)));
        } else {
            s.push_back(static_cast<char>(0xE0 | (c >> 12)));
            s.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | (c & 0x3F)));
        }
    }
    return s;
}

// Decode a byte-unicode key string back to raw bytes. Returns false if any
// codepoint is unmappable (i.e. the key is a special token, not byte-level).
inline bool key_to_bytes(const std::string &key, std::vector<uint8_t> &out)
{
    out.clear();
    out.reserve(key.size());
    for (size_t i = 0; i < key.size();) {
        char32_t cp = 0;
        size_t len = 0;
        uint8_t b0 = static_cast<uint8_t>(key[i]);
        if (b0 < 0x80) { cp = b0; len = 1; }
        else if ((b0 & 0xE0) == 0xC0) { cp = b0 & 0x1F; len = 2; }
        else if ((b0 & 0xF0) == 0xE0) { cp = b0 & 0x0F; len = 3; }
        else { cp = b0 & 0x07; len = 4; }
        if (i + len > key.size()) return false;
        for (size_t j = 1; j < len; ++j)
            cp = (cp << 6) | (static_cast<uint8_t>(key[i + j]) & 0x3F);
        int b = unicode_to_byte(cp);
        if (b < 0) return false;
        out.push_back(static_cast<uint8_t>(b));
        i += len;
    }
    return true;
}

} // namespace byte_unicode
} // namespace lsxcommon
