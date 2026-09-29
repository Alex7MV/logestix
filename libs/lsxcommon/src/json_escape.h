// json_escape.h — the single JSON string-escaping implementation for
// lsxcommon (response bodies, tool-call arguments, perf and governor logs).
//
// Do NOT substitute absl::CEscape here: it is a C escape, not a JSON one.
// absl::CEscape renders non-printable bytes as OCTAL ("\001", see
// absl/strings/escaping.cc kCEscapedSequence) and escapes '\'' as "\'".
// Neither is a valid JSON string escape, so a control character in a model
// response would produce a body no JSON parser accepts.
//
// Contract (JSON RFC 8259 §7): the quote, the backslash and the three
// mandatory escapes are short forms; every other character below U+0020 must
// become a \u00XX escape. Bytes >= 0x80 pass through unchanged so valid UTF-8
// (which JSON requires to be emitted as-is) is preserved.
#pragma once

#include <string>
#include <string_view>

namespace lsxcommon::json {

inline std::string JsonEscape(std::string_view in)
{
    std::string out;
    out.reserve(in.size() + 8);
    static const char kHex[] = "0123456789abcdef";
    for (const char ch : in) {
        switch (ch) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(ch) < 0x20) {
                out += "\\u00";
                out += kHex[(static_cast<unsigned char>(ch) >> 4) & 0xF];
                out += kHex[static_cast<unsigned char>(ch) & 0xF];
            } else {
                out += ch;
            }
        }
    }
    return out;
}

} // namespace lsxcommon::json
