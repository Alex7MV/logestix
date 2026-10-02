// json_escape.h — the single JSON string-escaping implementation (response
// bodies, tool-call arguments, perf and governor logs).
//
// It lives in lsxnetwork, not lsxcommon, because the dependency runs
// lsxcommon -> lsxnetwork: lsxnetwork must not include lsxcommon or the two
// libraries would depend on each other. Since lsxcommon links lsxnetwork
// PUBLIC, this header is visible to both — the SSE serializers in
// openai_endpoint.cpp / anthropic_endpoint.cpp and the model tool codecs share
// one implementation instead of each keeping a copy.
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

namespace lsxnetwork::json {

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

} // namespace lsxnetwork::json
