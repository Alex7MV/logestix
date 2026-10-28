#include "lsxcommon/pretokenizer.h"
#include "lsxcommon/unicode.h"
#include <cstring>

namespace lsxcommon {

namespace {

constexpr uint32_t kInvalid = kInvalidUtf8;

inline bool is_ascii_punct(uint32_t cp)
{
    if (cp > 0x7E) return false;
    static const char kPunct[] = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    return std::strchr(kPunct, static_cast<char>(cp)) != nullptr;
}

inline bool is_ascii_letter(uint32_t cp)
{
    return (cp >= 0x41 && cp <= 0x5A) || (cp >= 0x61 && cp <= 0x7A);
}

inline bool is_cjk_run(uint32_t cp)
{
    return (cp >= 0x4E00 && cp <= 0x9FA5) ||
           (cp >= 0x3040 && cp <= 0x309F) ||
           (cp >= 0x30A0 && cp <= 0x30FF);
}

} // anonymous namespace

// Decodes one UTF-8 codepoint at p (0 < n bytes available). Returns kInvalid
// and sets *len = 1 on any malformed sequence (overlong, surrogate, out of
// range, bad continuation). All other returns are valid Unicode.
// Declared in pretokenizer.h: arrow_tokenizer_decode sanitizes with the same
// decoder, which is what keeps decode output re-encodable.
uint32_t decode_utf8(const char *p, size_t n, size_t *len)
{
    const uint8_t *s = reinterpret_cast<const uint8_t *>(p);
    uint8_t b0 = s[0];
    if (b0 < 0x80) { *len = 1; return b0; }
    if ((b0 & 0xE0) == 0xC0) {
        if (n < 2 || (s[1] & 0xC0) != 0x80 || b0 < 0xC2) goto bad;
        *len = 2;
        return (uint32_t)((b0 & 0x1F) << 6) | (s[1] & 0x3F);
    }
    if ((b0 & 0xF0) == 0xE0) {
        if (n < 3 || (s[1] & 0xC0) != 0x80 || (s[2] & 0xC0) != 0x80) goto bad;
        uint32_t cp = (uint32_t)((b0 & 0x0F) << 12) | ((uint32_t)(s[1] & 0x3F) << 6) | (s[2] & 0x3F);
        if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) goto bad;
        *len = 3;
        return cp;
    }
    if ((b0 & 0xF8) == 0xF0) {
        if (n < 4 || (s[1] & 0xC0) != 0x80 || (s[2] & 0xC0) != 0x80 || (s[3] & 0xC0) != 0x80) goto bad;
        uint32_t cp = (uint32_t)((b0 & 0x07) << 18) | ((uint32_t)(s[1] & 0x3F) << 12) |
                      ((uint32_t)(s[2] & 0x3F) << 6) | (s[3] & 0x3F);
        if (cp < 0x10000 || cp > 0x10FFFF) goto bad;
        *len = 4;
        return cp;
    }
bad:
    *len = 1;
    return kInvalid;
}

size_t utf8_expected_len(uint8_t b)
{
    if (b < 0x80u) return 1;
    if ((b & 0xE0u) == 0xC0u) return 2;
    if ((b & 0xF0u) == 0xE0u) return 3;
    if ((b & 0xF8u) == 0xF0u) return 4;
    return 0;
}

WordStream::WordStream(const char *text, size_t length)
    : begin_(text), end_(text + length)
{
}

bool WordStream::next_number(WordSpan &word)
{
    size_t p = pos_;
    size_t count = 0;
    while (p < static_cast<size_t>(end_ - begin_) && count < 3) {
        size_t cl = 0;
        uint32_t c2 = decode_utf8(begin_ + p, static_cast<size_t>(end_ - (begin_ + p)), &cl);
        if (c2 == kInvalid || unicode::classify(c2) != unicode::CharClass::Number) break;
        p += cl;
        ++count;
    }
    word = WordSpan{begin_ + pos_, p - pos_};
    pos_ = p;
    return true;
}

bool WordStream::next_cjk(WordSpan &word)
{
    size_t p = pos_;
    while (p < static_cast<size_t>(end_ - begin_)) {
        size_t cl = 0;
        uint32_t c2 = decode_utf8(begin_ + p, static_cast<size_t>(end_ - (begin_ + p)), &cl);
        if (c2 == kInvalid || !is_cjk_run(c2)) break;
        p += cl;
    }
    word = WordSpan{begin_ + pos_, p - pos_};
    pos_ = p;
    return true;
}

// ---- main 6-alternative matcher ----
namespace {

// L/M run in bytes starting at a known-L/M first codepoint at 'start'.
size_t consume_lm(const char *begin, const char *end, size_t start, size_t first_len)
{
    size_t run = first_len;
    size_t p = start + first_len;
    while (p < static_cast<size_t>(end - begin)) {
        size_t cl = 0;
        uint32_t c2 = decode_utf8(begin + p, static_cast<size_t>(end - (begin + p)), &cl);
        if (c2 == kInvalid) break;
        // The CJK split runs before the main regex (Isolated), so a CJK
        // codepoint is a hard word boundary even though it is \p{L}.
        if (is_cjk_run(c2)) break;
        if (unicode::classify(c2) != unicode::CharClass::Letter &&
            unicode::classify(c2) != unicode::CharClass::Mark) break;
        run += cl;
        p += cl;
    }
    // Optional English contraction suffix, case-insensitive:
    //   (?i:'s|'t|'re|'ve|'m|'ll|'d)
    // Longest alternatives first ('re/'ve/'ll before 's/'t/'m/'d). The
    // apostrophe is the ASCII one only, like the reference pattern.
    if (p < static_cast<size_t>(end - begin) && begin[p] == '\'') {
        static const char *const kSuffix[] = {"re", "ve", "ll", "s", "t", "m", "d"};
        for (const char *sfx : kSuffix) {
            const size_t l = std::strlen(sfx);
            if (p + 1 + l > static_cast<size_t>(end - begin)) continue;
            bool ok = true;
            for (size_t i = 0; i < l; ++i) {
                char c = begin[p + 1 + i];
                if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
                if (c != sfx[i]) { ok = false; break; }
            }
            if (!ok) continue;
            // A contraction must not be followed by another letter
            // ("'s" inside "'something" stays with the letter run).
            const size_t after = p + 1 + l;
            if (after < static_cast<size_t>(end - begin)) {
                size_t cl2 = 0;
                const uint32_t c3 = decode_utf8(begin + after,
                    static_cast<size_t>(end - (begin + after)), &cl2);
                if (c3 != kInvalid &&
                    (unicode::classify(c3) == unicode::CharClass::Letter ||
                     unicode::classify(c3) == unicode::CharClass::Mark))
                    continue;
            }
            run = (p + 1 + l) - start;
            break;
        }
    }
    return run;
}

// Maximal Unicode whitespace run; on return *k is the byte offset after the
// last whitespace codepoint and *prev_start the start of that codepoint.
// *saw_invalid is set if decoding failed mid-run (run stops there).
void ws_run(const char *begin, const char *end, size_t pos, size_t *k, size_t *prev_start)
{
    size_t p = pos;
    size_t prev = pos;
    while (p < static_cast<size_t>(end - begin)) {
        size_t cl = 0;
        uint32_t c2 = decode_utf8(begin + p, static_cast<size_t>(end - (begin + p)), &cl);
        if (c2 == kInvalid || unicode::classify(c2) != unicode::CharClass::Whitespace) break;
        prev = p;
        p += cl;
    }
    *k = p;
    *prev_start = prev;
}

} // anonymous namespace

// Tries alt1..alt6 in priority order at 'pos'. On success sets *match_len
// (bytes) and returns true. Returns false when no alternative matches (the
// caller then emits a gap word). Malformed UTF-8 -> returns false with the
// stream error_ set by the caller on the next top-level decode; the matcher
// itself stops runs at kInvalid but never consumes beyond it.
static bool match_main(const char *begin, const char *end, size_t pos, size_t *match_len)
{
    size_t n = static_cast<size_t>(end - begin);
    if (pos >= n) return false;
    size_t cp_len = 0;
    uint32_t cp = decode_utf8(begin + pos, n - pos, &cp_len);
    if (cp == kInvalid) return false;

    // alt1: [ascii punct][ascii letter]+
    if (is_ascii_punct(cp)) {
        size_t p = pos + cp_len;
        size_t run = 0;
        while (p < n) {
            size_t cl = 0;
            uint32_t c2 = decode_utf8(begin + p, n - p, &cl);
            if (c2 == kInvalid || !is_ascii_letter(c2)) break;
            run += cl;
            p += cl;
        }
        if (run > 0) { *match_len = cp_len + run; return true; }
    }

    // alt2: [not \r\n \p{L} \p{P} \p{S}]? [\p{L}\p{M}]+
    //
    // The reference's kimi pat_str excludes \p{Han} from these letter classes
    // (a Han run is its own alternative), and the CJK split below is applied
    // before the regex. A Han codepoint must therefore never start (or be
    // pulled in as) this alternative — otherwise " 中" becomes one pretoken and
    // the Han run is cut after its first character, which changes the ids.
    {
        bool cp_lm = (unicode::classify(cp) == unicode::CharClass::Letter ||
                      unicode::classify(cp) == unicode::CharClass::Mark) &&
                     !is_cjk_run(cp);
        bool cp_other = cp != 0x0D && cp != 0x0A &&
                        unicode::classify(cp) != unicode::CharClass::Letter &&
                        unicode::classify(cp) != unicode::CharClass::Punct &&
                        unicode::classify(cp) != unicode::CharClass::Symbol;
        if (cp_lm) {
            *match_len = consume_lm(begin, end, pos, cp_len);
            return true;
        }
        if (cp_other) {
            size_t p = pos + cp_len;
            if (p < n) {
                size_t cl = 0;
                uint32_t c2 = decode_utf8(begin + p, n - p, &cl);
                if (c2 != kInvalid && (!is_cjk_run(c2)) &&
                    (unicode::classify(c2) == unicode::CharClass::Letter ||
                     unicode::classify(c2) == unicode::CharClass::Mark)) {
                    *match_len = cp_len + consume_lm(begin, end, p, cl);
                    return true;
                }
            }
            // prefix matched but no L/M follows -> alt2 fails, keep trying
        }
    }

    // alt3: ' '? [\p{P}\p{S}]+ [\r\n]*
    {
        size_t start = pos;
        if (cp == 0x20) start = pos + cp_len;
        size_t p = start;
        size_t ps = 0;
        bool has_ps = false;
        while (p < n) {
            size_t cl = 0;
            uint32_t c2 = decode_utf8(begin + p, n - p, &cl);
            if (c2 == kInvalid) break;
            if (unicode::classify(c2) != unicode::CharClass::Punct &&
                unicode::classify(c2) != unicode::CharClass::Symbol) break;
            ps += cl;
            p += cl;
            has_ps = true;
        }
        if (has_ps) {
            size_t crlf = 0;
            while (p < n) {
                uint8_t b = static_cast<uint8_t>(begin[p]);
                if (b != 0x0D && b != 0x0A) break;
                crlf += 1;
                p += 1;
            }
            *match_len = (start - pos) + ps + crlf;
            return true;
        }
    }

    // alt4: \s*[\r\n]+  (whitespace run ending in >= 1 \r or \n)
    if (unicode::classify(cp) == unicode::CharClass::Whitespace) {
        size_t k = 0, prev_start = 0;
        ws_run(begin, end, pos, &k, &prev_start);
        size_t last_crlf_end = static_cast<size_t>(-1);
        size_t p = pos;
        while (p < k) {
            size_t cl = 0;
            uint32_t c2 = decode_utf8(begin + p, n - p, &cl);
            if (c2 == kInvalid) break;
            if (c2 == 0x0D || c2 == 0x0A) last_crlf_end = p + cl;
            p += cl;
        }
        if (last_crlf_end != static_cast<size_t>(-1)) {
            *match_len = last_crlf_end - pos;
            return true;
        }
    }

    // alt5: \s+(?!\S)  (backtrack one codepoint when followed by non-ws)
    if (unicode::classify(cp) == unicode::CharClass::Whitespace) {
        size_t k = 0, prev_start = 0;
        ws_run(begin, end, pos, &k, &prev_start);
        if (k >= n) { *match_len = k - pos; return true; }   // lookahead sees EOF
        if (prev_start > pos) { *match_len = prev_start - pos; return true; }
        // single whitespace codepoint followed by non-whitespace -> alt5 fails
    }

    // alt6: \s+  (only reached when alt5 failed)
    if (unicode::classify(cp) == unicode::CharClass::Whitespace) {
        size_t k = 0, prev_start = 0;
        ws_run(begin, end, pos, &k, &prev_start);
        *match_len = k - pos;
        return true;
    }

    return false;
}

bool WordStream::next_main(WordSpan &word)
{
    size_t n = static_cast<size_t>(end_ - begin_);
    size_t len = 0;
    if (match_main(begin_, end_, pos_, &len)) {
        word = WordSpan{begin_ + pos_, len};
        pos_ += len;
        return true;
    }

    // Gap: extend until the next number / CJK-run / main-regex match.
    size_t gap_start = pos_;
    size_t p = pos_;
    while (p < n) {
        size_t cl = 0;
        uint32_t c2 = decode_utf8(begin_ + p, n - p, &cl);
        if (c2 == kInvalid) { error_ = 1; break; }
        if (unicode::classify(c2) == unicode::CharClass::Number || is_cjk_run(c2)) break;
        size_t mlen = 0;
        if (match_main(begin_, end_, p, &mlen)) break;
        p += cl;
    }
    word = WordSpan{begin_ + gap_start, p - gap_start};
    pos_ = p;
    return true;
}

bool WordStream::next(WordSpan &word)
{
    if (error_) return false;
    size_t n = static_cast<size_t>(end_ - begin_);
    if (pos_ >= n) return false;

    size_t cp_len = 0;
    uint32_t cp = decode_utf8(begin_ + pos_, n - pos_, &cp_len);
    if (cp == kInvalid) { error_ = 1; return false; }

    if (unicode::classify(cp) == unicode::CharClass::Number) return next_number(word);
    if (is_cjk_run(cp)) return next_cjk(word);
    return next_main(word);
}

// ── Llama-3 / o200k-style pre-tokenizer ────────────────────────────────────
namespace {

using unicode::CharClass;

// [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}] — marks plus every letter that is not
// lowercase (Lm/Lo belong to both classes).
inline bool cp_upper_class(uint32_t cp)
{
    if (unicode::classify(cp) == CharClass::Mark) return true;
    if (unicode::classify(cp) != CharClass::Letter) return false;
    return !unicode::is_lowercase_letter(cp);
}
// [\p{Ll}\p{Lm}\p{Lo}\p{M}] — marks plus every letter that is neither
// uppercase nor titlecase.
inline bool cp_lower_class(uint32_t cp)
{
    if (unicode::classify(cp) == CharClass::Mark) return true;
    if (unicode::classify(cp) != CharClass::Letter) return false;
    return !unicode::is_uppercase_letter(cp) &&
           !unicode::is_titlecase_letter(cp);
}
inline bool cp_letter_or_mark(uint32_t cp)
{
    const CharClass k = unicode::classify(cp);
    return k == CharClass::Letter || k == CharClass::Mark;
}
template <typename Pred>
size_t consume_run(const char *begin, const char *end, size_t pos, Pred pred)
{
    const size_t n = static_cast<size_t>(end - begin);
    size_t p = pos;
    while (p < n) {
        size_t cl = 0;
        const uint32_t cp = decode_utf8(begin + p, n - p, &cl);
        if (cp == kInvalidUtf8 || !pred(cp)) break;
        p += cl;
    }
    return p;
}

} // namespace

Llama3WordStream::Llama3WordStream(const char *text, size_t length)
    : begin_(text), end_(text + length)
{
}

size_t Llama3WordStream::consume_case_run(size_t pos) const
{
    // alt1/alt2 collapse to one greedy scan: the case-run alternatives match
    // exactly the maximal [Upper*][Lower+] region (an all-upper run is alt2's
    // [Upper+][Lower*] with an empty tail, which ends at the same byte).
    const size_t up = consume_run(begin_, end_, pos, cp_upper_class);
    return consume_run(begin_, end_, up, cp_lower_class);
}

size_t Llama3WordStream::consume_contraction(size_t pos) const
{
    const size_t n = static_cast<size_t>(end_ - begin_);
    if (pos >= n || begin_[pos] != '\'') return pos;
    static const char *const kSuffix[] = {"re", "ve", "ll", "s", "t", "m", "d"};
    for (const char *sfx : kSuffix) {
        const size_t l = std::strlen(sfx);
        if (pos + 1 + l > n) continue;
        bool ok = true;
        for (size_t i = 0; i < l; ++i) {
            char c = begin_[pos + 1 + i];
            if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            if (c != sfx[i]) { ok = false; break; }
        }
        if (!ok) continue;
        const size_t after = pos + 1 + l;
        if (after < n) {
            size_t cl = 0;
            const uint32_t c3 = decode_utf8(begin_ + after, n - after, &cl);
            if (c3 != kInvalidUtf8 && cp_letter_or_mark(c3)) continue;
        }
        return after;
    }
    return pos;
}

bool Llama3WordStream::match(size_t pos, size_t &len) const
{
    const size_t n = static_cast<size_t>(end_ - begin_);
    if (pos >= n) return false;
    size_t cl = 0;
    const uint32_t cp = decode_utf8(begin_ + pos, n - pos, &cl);
    if (cp == kInvalidUtf8) return false;

    const bool is_prefix =
        cp != 0x0D && cp != 0x0A &&
        unicode::classify(cp) != CharClass::Letter &&
        unicode::classify(cp) != CharClass::Number;

    // alt1/alt2 with the optional `[^\r\n\p{L}\p{N}]?` prefix. The prefix is
    // optional-greedy, so it is tried first and dropped when the rest fails.
    for (int with_prefix = is_prefix ? 1 : 0; with_prefix >= 0; --with_prefix) {
        const size_t p = pos + (with_prefix ? cl : 0);
        if (p >= n) continue;
        size_t q = 0;
        const uint32_t c2 = decode_utf8(begin_ + p, n - p, &q);
        if (c2 == kInvalidUtf8) continue;
        if (!cp_upper_class(c2) && !cp_lower_class(c2)) continue;
        const size_t run_end = consume_case_run(p);
        if (run_end == p) continue;
        len = consume_contraction(run_end) - pos;
        return true;
    }

    // alt3: \p{N}{1,3}
    if (unicode::classify(cp) == CharClass::Number) {
        size_t p = pos;
        int count = 0;
        while (p < n && count < 3) {
            size_t q = 0;
            const uint32_t c2 = decode_utf8(begin_ + p, n - p, &q);
            if (c2 == kInvalidUtf8 || unicode::classify(c2) != CharClass::Number)
                break;
            p += q;
            ++count;
        }
        if (p > pos) { len = p - pos; return true; }
    }

    // alt4: ' '? [^\s\p{L}\p{N}]+ [\r\n/]*
    {
        const size_t p0 = (cp == 0x20) ? pos + cl : pos;
        const size_t run_begin = p0;
        size_t p = p0;
        while (p < n) {
            size_t q = 0;
            const uint32_t c2 = decode_utf8(begin_ + p, n - p, &q);
            if (c2 == kInvalidUtf8) break;
            const CharClass k = unicode::classify(c2);
            if (k == CharClass::Whitespace || k == CharClass::Letter ||
                k == CharClass::Number)
                break;
            p += q;
        }
        if (p > run_begin) {
            while (p < n) {
                const char b = begin_[p];
                if (b != '\r' && b != '\n' && b != '/') break;
                ++p;
            }
            len = p - pos;
            return true;
        }
    }

    // alt5/alt6/alt7: whitespace runs. alt5 keeps up to the last \r/\n; alt6
    // drops one trailing codepoint when a non-space follows (its (?!\S)
    // lookahead); alt7 takes the whole run.
    if (unicode::classify(cp) == CharClass::Whitespace) {
        size_t p = pos;
        size_t last_crlf = static_cast<size_t>(-1);
        size_t prev_cp = pos;
        while (p < n) {
            size_t q = 0;
            const uint32_t c2 = decode_utf8(begin_ + p, n - p, &q);
            if (c2 == kInvalidUtf8 || unicode::classify(c2) != CharClass::Whitespace)
                break;
            if (c2 == 0x0D || c2 == 0x0A) last_crlf = p + q;
            prev_cp = p;
            p += q;
        }
        if (last_crlf != static_cast<size_t>(-1)) {
            len = last_crlf - pos;
            return true;
        }
        if (p >= n) {
            len = p - pos;   // alt5/. alt6 with the lookahead at EOF
            return true;
        }
        if (prev_cp > pos) {
            len = prev_cp - pos;   // alt6: all but the last codepoint
            return true;
        }
        len = p - pos;             // alt7: a single whitespace codepoint
        return true;
    }

    return false;
}

bool Llama3WordStream::next(WordSpan &word)
{
    if (error_) return false;
    const size_t n = static_cast<size_t>(end_ - begin_);
    if (pos_ >= n) return false;
    size_t cl = 0;
    const uint32_t cp = decode_utf8(begin_ + pos_, n - pos_, &cl);
    if (cp == kInvalidUtf8) { error_ = 1; return false; }
    size_t len = 0;
    if (!match(pos_, len) || len == 0) len = cl;  // unmatched text: its own piece
    word = WordSpan{begin_ + pos_, len};
    pos_ += len;
    return true;
}

} // namespace lsxcommon
