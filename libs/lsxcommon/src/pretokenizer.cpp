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

WordStream::WordStream(const char *text, size_t length, bool three_stage_split)
    : begin_(text), end_(text + length), three_stage_(three_stage_split)
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

// End of the gap the main regex is allowed to see.
//
// The reference DeepSeek pre-tokenizer is a Sequence of three Split stages:
// \p{N}{1,3} Isolated, the CJK run Isolated, and only then the main pattern.
// The main pattern therefore runs on the text BETWEEN isolated runs, and its
// whitespace lookahead hits EOF at that boundary. Running it against the whole
// remaining buffer instead splits "  " before a digit into two pieces; the
// reference keeps the run whole ("a  1" -> ["a", "  ", "1"]).
//
// `pos` is never itself a digit or CJK codepoint (next() dispatches those to
// next_number/next_cjk first), so the returned offset is > pos unless the rest
// of the buffer is a single isolated run.
size_t StageGapEnd(const char *begin, const char *end, size_t pos)
{
    const size_t n = static_cast<size_t>(end - begin);
    size_t p = pos;
    while (p < n) {
        size_t cl = 0;
        const uint32_t cp = decode_utf8(begin + p, n - p, &cl);
        if (cp == kInvalid) return n;
        if (unicode::classify(cp) == unicode::CharClass::Number ||
            is_cjk_run(cp))
            return p;
        p += cl;
    }
    return n;
}

bool WordStream::next_main(WordSpan &word)
{
    if (!three_stage_) {
        const char *gap_end = end_;
        size_t len = 0;
        if (match_main(begin_, gap_end, pos_, &len)) {
            word = WordSpan{begin_ + pos_, len};
            pos_ += len;
            return true;
        }
        size_t gap_start = pos_;
        size_t p = pos_;
        const size_t n = static_cast<size_t>(end_ - begin_);
        while (p < n) {
            size_t cl = 0;
            uint32_t c2 = decode_utf8(begin_ + p, n - p, &cl);
            if (c2 == kInvalid) { error_ = 1; break; }
            if (unicode::classify(c2) == unicode::CharClass::Number ||
                is_cjk_run(c2))
                break;
            size_t mlen = 0;
            if (match_main(begin_, gap_end, p, &mlen)) break;
            p += cl;
        }
        word = WordSpan{begin_ + gap_start, p - gap_start};
        pos_ = p;
        return true;
    }
    if (!stage_valid_ || stage_next_ < pos_) {
        stage_next_ = StageGapEnd(begin_, end_, pos_);
        stage_valid_ = true;
    }
    const char *gap_end = begin_ + stage_next_;
    const size_t gap_n = static_cast<size_t>(gap_end - begin_);
    size_t len = 0;
    if (match_main(begin_, gap_end, pos_, &len)) {
        word = WordSpan{begin_ + pos_, len};
        pos_ += len;
        return true;
    }

    // Gap: extend until the next number / CJK-run / main-regex match, never
    // past the isolated-run boundary.
    size_t gap_start = pos_;
    size_t p = pos_;
    while (p < gap_n) {
        size_t cl = 0;
        uint32_t c2 = decode_utf8(begin_ + p, gap_n - p, &cl);
        if (c2 == kInvalid) { error_ = 1; break; }
        size_t mlen = 0;
        if (match_main(begin_, gap_end, p, &mlen)) break;
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
template <typename Pred>
size_t consume_run(const char *begin, const char *end, size_t pos, Pred pred,
                   bool stop_at_cjk = false)
{
    const size_t n = static_cast<size_t>(end - begin);
    size_t p = pos;
    while (p < n) {
        size_t cl = 0;
        const uint32_t cp = decode_utf8(begin + p, n - p, &cl);
        if (cp == kInvalidUtf8 || !pred(cp)) break;
        if (stop_at_cjk && is_cjk_run(cp)) break;
        p += cl;
    }
    return p;
}

} // namespace

Llama3WordStream::Llama3WordStream(const char *text, size_t length)
    : begin_(text), end_(text + length)
{
}

size_t CaseRunEnd(const char *begin, const char *end, size_t pos,
                  bool stop_at_cjk)
{
    // alt1/alt2 collapse to one greedy scan: the case-run alternatives match
    // exactly the maximal [Upper*][Lower+] region (an all-upper run is alt2's
    // [Upper+][Lower*] with an empty tail, which ends at the same byte).
    const size_t up = consume_run(begin, end, pos, cp_upper_class, stop_at_cjk);
    return consume_run(begin, end, up, cp_lower_class, stop_at_cjk);
}

size_t ContractionEnd(const char *begin, const char *end, size_t pos)
{
    const size_t n = static_cast<size_t>(end - begin);
    if (pos >= n || begin[pos] != '\'') return pos;
    static const char *const kSuffix[] = {"re", "ve", "ll", "s", "t", "m", "d"};
    for (const char *sfx : kSuffix) {
        const size_t l = std::strlen(sfx);
        if (pos + 1 + l > n) continue;
        bool ok = true;
        for (size_t i = 0; i < l; ++i) {
            char c = begin[pos + 1 + i];
            if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            if (c != sfx[i]) { ok = false; break; }
        }
        if (!ok) continue;
        // The o200k/Llama-3 pattern's optional contraction is greedy: it is
        // taken even when more letters follow ("majnf'mpRKP" ->
        // ["majnf'm", "p", "RKP"]). Do not add a "next codepoint is not a
        // letter" guard here — that is the DeepSeek alt1 behaviour, not this
        // pattern's, and it changes the ids of mixed-case identifiers.
        return pos + 1 + l;
    }
    return pos;
}

// Matches the case-run pattern at `pos`. `flint_punct` switches alternative 4
// between o200k's `' '? [^\s\p{L}\p{N}]+ [\r\n/]*` (Llama-3 / Qwen) and
// Flint's `' '? [\p{P}\p{S}]+ [\r\n]*`.
bool MatchCaseRunPattern(const char *begin, const char *end, size_t pos,
                         bool flint_punct, size_t &len, int digit_max = 3,
                         bool contraction = true, bool stop_at_cjk = false,
                         bool slash_after_punct = true,
                         bool grouped_digits = false)
{
    const size_t n = static_cast<size_t>(end - begin);
    if (pos >= n) return false;
    size_t cl = 0;
    const uint32_t cp = decode_utf8(begin + pos, n - pos, &cl);
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
        const uint32_t c2 = decode_utf8(begin + p, n - p, &q);
        if (c2 == kInvalidUtf8) continue;
        if (!cp_upper_class(c2) && !cp_lower_class(c2)) continue;
        const size_t run_end = CaseRunEnd(begin, end, p, stop_at_cjk);
        if (run_end == p) continue;
        len = (contraction ? ContractionEnd(begin, end, run_end) : run_end) -
              pos;
        return true;
    }

    // alt3: \p{N}{1,digit_max}; with `grouped_digits` the count is the one
    // that makes the remainder a multiple of three (right-grouped runs).
    if (unicode::classify(cp) == CharClass::Number) {
        size_t p = pos;
        int count = 0;
        int limit = digit_max;
        if (grouped_digits) {
            int run = 0;
            while (p < n) {
                size_t q = 0;
                const uint32_t c2 = decode_utf8(begin + p, n - p, &q);
                if (c2 == kInvalidUtf8 ||
                    unicode::classify(c2) != CharClass::Number)
                    break;
                p += q;
                ++run;
            }
            limit = (run % 3 == 0) ? 3 : (run % 3 == 1 ? 1 : 2);
            p = pos;
        }
        while (p < n && count < limit) {
            size_t q = 0;
            const uint32_t c2 = decode_utf8(begin + p, n - p, &q);
            if (c2 == kInvalidUtf8 || unicode::classify(c2) != CharClass::Number)
                break;
            p += q;
            ++count;
        }
        if (p > pos) { len = p - pos; return true; }
    }

    // alt4: ' '? (o200k: [^\s\p{L}\p{N}]+ [\r\n/]* / Flint: [\p{P}\p{S}]+ [\r\n]*)
    {
        const size_t p0 = (cp == 0x20) ? pos + cl : pos;
        const size_t run_begin = p0;
        size_t p = p0;
        while (p < n) {
            size_t q = 0;
            const uint32_t c2 = decode_utf8(begin + p, n - p, &q);
            if (c2 == kInvalidUtf8) break;
            const CharClass k = unicode::classify(c2);
            if (flint_punct) {
                if (k != CharClass::Punct && k != CharClass::Symbol) break;
            } else if (k == CharClass::Whitespace ||
                       k == CharClass::Letter || k == CharClass::Number) {
                break;
            }
            p += q;
        }
        if (p > run_begin) {
            while (p < n) {
                const char b = begin[p];
                if (b != '\r' && b != '\n' &&
                    !(slash_after_punct && b == '/'))
                    break;
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
            const uint32_t c2 = decode_utf8(begin + p, n - p, &q);
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

bool Llama3WordStream::match(size_t pos, size_t &len) const
{
    return MatchCaseRunPattern(begin_, end_, pos, /*flint_punct=*/false, len);
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

// ── Ling single-run pattern (tokenizer_kind 5) ─────────────────────────────

LingWordStream::LingWordStream(const char *text, size_t length)
    : begin_(text), end_(text + length)
{
}

bool LingWordStream::match(size_t pos, size_t &len) const
{
    const size_t n = static_cast<size_t>(end_ - begin_);
    if (pos >= n) return false;
    size_t cl = 0;
    const uint32_t cp = decode_utf8(begin_ + pos, n - pos, &cl);
    if (cp == kInvalid) return false;

    // alt1: '(?i:[sdmt]|ll|ve|re)
    if (cp == '\'') {
        static const char *const kSuffix[] = {"s", "t", "d", "m", "ll", "ve",
                                              "re"};
        for (const char *sfx : kSuffix) {
            const size_t l = std::strlen(sfx);
            if (pos + 1 + l > n) continue;
            bool ok = true;
            for (size_t i = 0; i < l; ++i) {
                char ch = begin_[pos + 1 + i];
                if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
                if (ch != sfx[i]) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                len = 1 + l;
                return true;
            }
        }
    }

    // alt2: [^\r\n\p{L}\p{N}]?+ \p{L}+   (possessive prefix)
    {
        const bool is_prefix =
            cp != 0x0D && cp != 0x0A &&
            unicode::classify(cp) != unicode::CharClass::Letter &&
            unicode::classify(cp) != unicode::CharClass::Number;
        const size_t p = pos + (is_prefix ? cl : 0);
        if (p < n) {
            size_t q = 0;
            const uint32_t c2 = decode_utf8(begin_ + p, n - p, &q);
            if (c2 != kInvalid &&
                unicode::classify(c2) == unicode::CharClass::Letter) {
                size_t e = p;
                while (e < n) {
                    size_t r = 0;
                    const uint32_t c3 = decode_utf8(begin_ + e, n - e, &r);
                    if (c3 == kInvalid ||
                        unicode::classify(c3) != unicode::CharClass::Letter)
                        break;
                    e += r;
                }
                len = e - pos;
                return true;
            }
        }
    }

    // alt3: \p{N}
    if (unicode::classify(cp) == unicode::CharClass::Number) {
        len = cl;
        return true;
    }

    // alt4: ' '? [^\s\p{L}\p{N}]++ [\r\n]*
    {
        const size_t p0 = (cp == 0x20) ? pos + cl : pos;
        size_t p = p0;
        while (p < n) {
            size_t q = 0;
            const uint32_t c2 = decode_utf8(begin_ + p, n - p, &q);
            if (c2 == kInvalid) break;
            const CharClass k = unicode::classify(c2);
            if (k == CharClass::Whitespace || k == CharClass::Letter ||
                k == CharClass::Number)
                break;
            p += q;
        }
        if (p > p0) {
            while (p < n && (begin_[p] == '\r' || begin_[p] == '\n')) ++p;
            len = p - pos;
            return true;
        }
    }

    // alt5-7: whitespace runs (same semantics as the GPT-2 stream).
    if (unicode::classify(cp) == unicode::CharClass::Whitespace) {
        size_t p = pos;
        size_t last_crlf = static_cast<size_t>(-1);
        size_t prev_cp = pos;
        while (p < n) {
            size_t q = 0;
            const uint32_t c2 = decode_utf8(begin_ + p, n - p, &q);
            if (c2 == kInvalid ||
                unicode::classify(c2) != unicode::CharClass::Whitespace)
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
            len = p - pos;
            return true;
        }
        if (prev_cp > pos) {
            len = prev_cp - pos;
            return true;
        }
        len = p - pos;
        return true;
    }
    return false;
}

bool LingWordStream::next(WordSpan &word)
{
    if (error_) return false;
    const size_t n = static_cast<size_t>(end_ - begin_);
    if (pos_ >= n) return false;
    size_t cl = 0;
    const uint32_t cp = decode_utf8(begin_ + pos_, n - pos_, &cl);
    if (cp == kInvalid) {
        error_ = 1;
        return false;
    }
    size_t len = 0;
    if (!match(pos_, len) || len == 0) len = cl;
    word = WordSpan{begin_ + pos_, len};
    pos_ += len;
    return true;
}

// ── Beam single-run pattern (tokenizer_kind 6) ─────────────────────────────

BeamWordStream::BeamWordStream(const char *text, size_t length)
    : begin_(text), end_(text + length)
{
}

bool BeamWordStream::match(size_t pos, size_t &len) const
{
    const size_t n = static_cast<size_t>(end_ - begin_);
    if (pos >= n) return false;
    size_t cl = 0;
    const uint32_t cp = decode_utf8(begin_ + pos, n - pos, &cl);
    if (cp == kInvalid) return false;

    // alt1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (cp == '\'') {
        static const char *const kSuffix[] = {"s", "t", "re", "ve", "m", "ll",
                                              "d"};
        for (const char *sfx : kSuffix) {
            const size_t l = std::strlen(sfx);
            if (pos + 1 + l > n) continue;
            bool ok = true;
            for (size_t i = 0; i < l; ++i) {
                char ch = begin_[pos + 1 + i];
                if (ch >= 'A' && ch <= 'Z')
                    ch = static_cast<char>(ch - 'A' + 'a');
                if (ch != sfx[i]) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                len = 1 + l;
                return true;
            }
        }
    }

    // alt2: [^\r\n\p{L}\p{N}]? \p{L}+
    {
        const bool is_prefix =
            cp != 0x0D && cp != 0x0A &&
            unicode::classify(cp) != unicode::CharClass::Letter &&
            unicode::classify(cp) != unicode::CharClass::Number;
        const size_t p = pos + (is_prefix ? cl : 0);
        if (p < n) {
            size_t q = 0;
            const uint32_t c2 = decode_utf8(begin_ + p, n - p, &q);
            if (c2 != kInvalid &&
                unicode::classify(c2) == unicode::CharClass::Letter) {
                size_t e = p;
                while (e < n) {
                    size_t r = 0;
                    const uint32_t c3 = decode_utf8(begin_ + e, n - e, &r);
                    if (c3 == kInvalid ||
                        unicode::classify(c3) != unicode::CharClass::Letter)
                        break;
                    e += r;
                }
                len = e - pos;
                return true;
            }
        }
    }

    // alt3: \p{N} (one digit/codepoint per word)
    if (unicode::classify(cp) == unicode::CharClass::Number) {
        len = cl;
        return true;
    }

    // alt4: ' '? [^\s\p{L}\p{N}]+ [\r\n]*
    {
        const size_t p0 = (cp == 0x20) ? pos + cl : pos;
        size_t p = p0;
        while (p < n) {
            size_t q = 0;
            const uint32_t c2 = decode_utf8(begin_ + p, n - p, &q);
            if (c2 == kInvalid) break;
            const CharClass k = unicode::classify(c2);
            if (k == CharClass::Whitespace || k == CharClass::Letter ||
                k == CharClass::Number)
                break;
            p += q;
        }
        if (p > p0) {
            while (p < n && (begin_[p] == '\r' || begin_[p] == '\n')) ++p;
            len = p - pos;
            return true;
        }
    }

    // alt5-7: \s*[\r\n]+ | \s+(?!\S) | \s+
    if (unicode::classify(cp) == unicode::CharClass::Whitespace) {
        size_t p = pos;
        size_t last_crlf = static_cast<size_t>(-1);
        size_t prev_cp = pos;
        while (p < n) {
            size_t q = 0;
            const uint32_t c2 = decode_utf8(begin_ + p, n - p, &q);
            if (c2 == kInvalid ||
                unicode::classify(c2) != unicode::CharClass::Whitespace)
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
            len = p - pos;
            return true;
        }
        if (prev_cp > pos) {
            len = prev_cp - pos;
            return true;
        }
        len = p - pos;
        return true;
    }
    return false;
}

bool BeamWordStream::next(WordSpan &word)
{
    if (error_) return false;
    const size_t n = static_cast<size_t>(end_ - begin_);
    if (pos_ >= n) return false;
    size_t cl = 0;
    const uint32_t cp = decode_utf8(begin_ + pos_, n - pos_, &cl);
    if (cp == kInvalid) {
        error_ = 1;
        return false;
    }
    size_t len = 0;
    if (!match(pos_, len) || len == 0) len = cl;
    word = WordSpan{begin_ + pos_, len};
    pos_ += len;
    return true;
}

// ── Flint 3-split (tokenizer_kind 4) ───────────────────────────────────────

FlintWordStream::FlintWordStream(const char *text, size_t length)
    : begin_(text), end_(text + length)
{
}

bool FlintWordStream::next_number(WordSpan &word)
{
    const size_t n = static_cast<size_t>(end_ - begin_);
    size_t p = pos_;
    int count = 0;
    while (p < n && count < 3) {
        size_t q = 0;
        const uint32_t cp = decode_utf8(begin_ + p, n - p, &q);
        if (cp == kInvalidUtf8 || unicode::classify(cp) != unicode::CharClass::Number)
            break;
        p += q;
        ++count;
    }
    word = WordSpan{begin_ + pos_, p - pos_};
    pos_ = p;
    return true;
}

bool FlintWordStream::next_cjk(WordSpan &word)
{
    const size_t n = static_cast<size_t>(end_ - begin_);
    size_t p = pos_;
    while (p < n) {
        size_t q = 0;
        const uint32_t cp = decode_utf8(begin_ + p, n - p, &q);
        if (cp == kInvalidUtf8 || !is_cjk_run(cp)) break;
        p += q;
    }
    word = WordSpan{begin_ + pos_, p - pos_};
    pos_ = p;
    return true;
}

bool FlintWordStream::next_main(WordSpan &word)
{
    if (!stage_valid_ || stage_next_ < pos_) {
        stage_next_ = StageGapEnd(begin_, end_, pos_);
        stage_valid_ = true;
    }
    const char *gap_end = begin_ + stage_next_;
    const size_t gap_n = static_cast<size_t>(gap_end - begin_);
    size_t len = 0;
    if (!MatchCaseRunPattern(begin_, gap_end, pos_, /*flint_punct=*/true,
                             len, /*digit_max=*/3, /*contraction=*/true,
                             /*stop_at_cjk=*/true,
                             /*slash_after_punct=*/false) ||
        len == 0) {
        size_t cl = 0;
        const uint32_t cp = decode_utf8(begin_ + pos_, gap_n - pos_, &cl);
        len = (cp == kInvalidUtf8) ? 1 : cl;   // unmatched text: its own piece
    }
    word = WordSpan{begin_ + pos_, len};
    pos_ += len;
    return true;
}

bool FlintWordStream::next(WordSpan &word)
{
    if (error_) return false;
    const size_t n = static_cast<size_t>(end_ - begin_);
    if (pos_ >= n) return false;
    size_t cl = 0;
    const uint32_t cp = decode_utf8(begin_ + pos_, n - pos_, &cl);
    if (cp == kInvalidUtf8) { error_ = 1; return false; }
    if (unicode::classify(cp) == unicode::CharClass::Number)
        return next_number(word);
    if (is_cjk_run(cp)) return next_cjk(word);
    return next_main(word);
}

// ── single-stage GPT-2-family patterns (tokenizer_kind 7-10) ───────────────
//
// The reference `pre_tokenizer` of these checkpoints is ONE Split stage, not
// the three-stage Sequence WordStream models. In particular a run of spaces
// before a digit is NOT kept whole (`\s+(?!\S)` backtracks one codepoint and
// `\s+` takes the last), while a space IS attached to a following Han/Kana
// letter run by the optional prefix.
namespace {

bool MatchContractionAlt(const char *begin, const char *end, size_t pos,
                         size_t &len, bool case_insensitive)
{
    const size_t n = static_cast<size_t>(end - begin);
    if (pos >= n || begin[pos] != '\'') return false;
    static const char *const kSuffix[] = {"s", "t", "re", "ve", "m", "ll", "d"};
    for (const char *sfx : kSuffix) {
        const size_t l = std::strlen(sfx);
        if (pos + 1 + l > n) continue;
        bool ok = true;
        for (size_t i = 0; i < l; ++i) {
            char a = begin[pos + 1 + i];
            if (case_insensitive && a >= 'A' && a <= 'Z')
                a = static_cast<char>(a - 'A' + 'a');
            if (a != sfx[i]) { ok = false; break; }
        }
        if (ok) {
            len = 1 + l;
            return true;
        }
    }
    return false;
}

// alt5-7: \s*[\r\n]+ | \s+(?!\S) | \s+. `crlf_alt = false` drops the first
// alternative (the original GPT-2 pattern has only the two lookahead forms).
bool MatchWhitespaceAlts(const char *begin, const char *end, size_t pos,
                         size_t &len, bool crlf_alt = true)
{
    const size_t n = static_cast<size_t>(end - begin);
    size_t cl = 0;
    const uint32_t cp = decode_utf8(begin + pos, n - pos, &cl);
    if (cp == kInvalidUtf8 ||
        unicode::classify(cp) != unicode::CharClass::Whitespace)
        return false;
    size_t p = pos;
    size_t last_crlf = static_cast<size_t>(-1);
    size_t prev_cp = pos;
    while (p < n) {
        size_t q = 0;
        const uint32_t c2 = decode_utf8(begin + p, n - p, &q);
        if (c2 == kInvalidUtf8 ||
            unicode::classify(c2) != unicode::CharClass::Whitespace)
            break;
        if (c2 == 0x0D || c2 == 0x0A) last_crlf = p + q;
        prev_cp = p;
        p += q;
    }
    if (crlf_alt && last_crlf != static_cast<size_t>(-1)) {
        len = last_crlf - pos;
        return true;
    }
    if (p >= n) {
        len = p - pos;
        return true;
    }
    if (prev_cp > pos) {
        len = prev_cp - pos;
        return true;
    }
    len = p - pos;
    return true;
}

size_t ConsumeLetters(const char *begin, const char *end, size_t pos,
                      bool include_marks)
{
    const size_t n = static_cast<size_t>(end - begin);
    size_t p = pos;
    while (p < n) {
        size_t cl = 0;
        const uint32_t cp = decode_utf8(begin + p, n - p, &cl);
        if (cp == kInvalidUtf8) break;
        const unicode::CharClass k = unicode::classify(cp);
        if (k != unicode::CharClass::Letter &&
            !(include_marks && k == unicode::CharClass::Mark))
            break;
        p += cl;
    }
    return p;
}

size_t ConsumeDigits(const char *begin, const char *end, size_t pos,
                     int max_digits)
{
    const size_t n = static_cast<size_t>(end - begin);
    size_t p = pos;
    int count = 0;
    while (p < n && (max_digits <= 0 || count < max_digits)) {
        size_t cl = 0;
        const uint32_t cp = decode_utf8(begin + p, n - p, &cl);
        if (cp == kInvalidUtf8 ||
            unicode::classify(cp) != unicode::CharClass::Number)
            break;
        p += cl;
        ++count;
    }
    return p;
}

} // namespace

// ── Hunyuan 3-split (tokenizer_kind 13) ────────────────────────────────────

HunyuanWordStream::HunyuanWordStream(const char *text, size_t length)
    : begin_(text), end_(text + length)
{
}

bool HunyuanWordStream::next_number(WordSpan &word)
{
    const size_t n = static_cast<size_t>(end_ - begin_);
    size_t p = pos_;
    int count = 0;
    while (p < n && count < 3) {
        size_t q = 0;
        const uint32_t cp = decode_utf8(begin_ + p, n - p, &q);
        if (cp == kInvalid || unicode::classify(cp) != unicode::CharClass::Number)
            break;
        p += q;
        ++count;
    }
    word = WordSpan{begin_ + pos_, p - pos_};
    pos_ = p;
    return true;
}

bool HunyuanWordStream::next_cjk(WordSpan &word)
{
    const size_t n = static_cast<size_t>(end_ - begin_);
    size_t p = pos_;
    while (p < n) {
        size_t q = 0;
        const uint32_t cp = decode_utf8(begin_ + p, n - p, &q);
        if (cp == kInvalid || !is_cjk_run(cp)) break;
        p += q;
    }
    word = WordSpan{begin_ + pos_, p - pos_};
    pos_ = p;
    return true;
}

namespace {

// The explicit ASCII punctuation set of the reference pattern's first
// alternative: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
bool IsAsciiPunctLead(uint32_t cp)
{
    if (cp < 0x21 || cp > 0x7E) return false;
    const char c = static_cast<char>(cp);
    return std::strchr("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", c) != nullptr;
}

// [A-Za-z]+
size_t ConsumeAsciiLetters(const char *begin, const char *end, size_t pos)
{
    const size_t n = static_cast<size_t>(end - begin);
    size_t p = pos;
    while (p < n) {
        const char c = begin[p];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))) break;
        ++p;
    }
    return p;
}

// [^\r\n\p{L}\p{P}\p{S}]?  — one codepoint that may prefix a letter run
// (a space is the case that matters: " hello" stays one piece, "  中" does not
// because stage 2 already isolated the Han run).
bool IsLetterRunPrefix(uint32_t cp)
{
    if (cp == 0x0D || cp == 0x0A) return false;
    const unicode::CharClass k = unicode::classify(cp);
    return k != unicode::CharClass::Letter && k != unicode::CharClass::Mark &&
           k != unicode::CharClass::Punct && k != unicode::CharClass::Symbol;
}

} // namespace

bool HunyuanWordStream::match(size_t pos, const char *gap_end,
                             size_t &len) const
{
    const size_t n = static_cast<size_t>(gap_end - begin_);
    if (pos >= n) return false;

    // alt1: [punct][A-Za-z]+  — this is where contractions land ("'s", "'ll"),
    // because there is no contraction alternative in this pattern.
    size_t cl = 0;
    const uint32_t cp = decode_utf8(begin_ + pos, n - pos, &cl);
    if (cp == kInvalid) return false;
    if (IsAsciiPunctLead(cp)) {
        const size_t letters = ConsumeAsciiLetters(begin_, gap_end, pos + cl);
        if (letters > pos + cl) {
            len = letters - pos;
            return true;
        }
    }

    // alt2: [prefix]?[\p{L}\p{M}]+
    //
    // A failing prefix check must NOT abort the alternation: " _call" is alt3's
    // (space + punctuation run) and "   m" is alt6's, and returning early here
    // is what made both split into single characters.
    {
        size_t p = pos;
        const unicode::CharClass k0 = unicode::classify(cp);
        bool prefix_ok = true;
        if (k0 != unicode::CharClass::Letter && k0 != unicode::CharClass::Mark) {
            if (!IsLetterRunPrefix(cp) || pos + cl >= n) {
                prefix_ok = false;
            } else {
                size_t cl2 = 0;
                const uint32_t cp2 =
                    decode_utf8(begin_ + pos + cl, n - pos - cl, &cl2);
                const unicode::CharClass k1 =
                    cp2 == kInvalid ? unicode::CharClass::Other
                                    : unicode::classify(cp2);
                if (k1 == unicode::CharClass::Letter ||
                    k1 == unicode::CharClass::Mark)
                    p = pos + cl;
                else
                    prefix_ok = false;
            }
        }
        if (prefix_ok) {
            size_t q = p;
            while (q < n) {
                size_t cq = 0;
                const uint32_t c2 = decode_utf8(begin_ + q, n - q, &cq);
                if (c2 == kInvalid) break;
                const unicode::CharClass k = unicode::classify(c2);
                if (k != unicode::CharClass::Letter &&
                    k != unicode::CharClass::Mark)
                    break;
                q += cq;
            }
            if (q > p) {
                len = q - pos;
                return true;
            }
        }
    }

    // alt3: ' '?[\p{P}\p{S}]+[\r\n]*
    {
        size_t p = pos;
        if (cp == 0x20) p += cl;
        size_t q = p;
        while (q < n) {
            size_t cq = 0;
            const uint32_t c2 = decode_utf8(begin_ + q, n - q, &cq);
            if (c2 == kInvalid) break;
            const unicode::CharClass k = unicode::classify(c2);
            if (k != unicode::CharClass::Punct &&
                k != unicode::CharClass::Symbol)
                break;
            q += cq;
        }
        if (q > p) {
            while (q < n && (begin_[q] == '\r' || begin_[q] == '\n')) ++q;
            len = q - pos;
            return true;
        }
    }

    // alt4-6: \s*[\r\n]+ | \s+(?!\S) | \s+
    return MatchWhitespaceAlts(begin_, gap_end, pos, len);
}

bool HunyuanWordStream::next_main(WordSpan &word)
{
    if (!stage_valid_ || stage_next_ < pos_) {
        stage_next_ = StageGapEnd(begin_, end_, pos_);
        stage_valid_ = true;
    }
    const size_t gap_n = stage_next_;
    size_t len = 0;
    // The main pattern runs on the GAP text only: the digit and CJK splits have
    // already claimed their runs, and a match that crossed `gap_n` would have to
    // be discarded (that is what split "abc" into a, b, c before "あいうえお").
    const char *const gap_end = begin_ + gap_n;
    if (!match(pos_, gap_end, len) || len == 0) {
        size_t cl = 0;
        const uint32_t cp = decode_utf8(begin_ + pos_, gap_n - pos_, &cl);
        len = (cp == kInvalid) ? 1 : cl;   // unmatched text: its own piece
    }
    word = WordSpan{begin_ + pos_, len};
    pos_ += len;
    return true;
}

bool HunyuanWordStream::next(WordSpan &word)
{
    if (error_) return false;
    const size_t n = static_cast<size_t>(end_ - begin_);
    if (pos_ >= n) return false;
    size_t cl = 0;
    const uint32_t cp = decode_utf8(begin_ + pos_, n - pos_, &cl);
    if (cp == kInvalid) { error_ = 1; return false; }
    if (unicode::classify(cp) == unicode::CharClass::Number)
        return next_number(word);
    if (is_cjk_run(cp)) return next_cjk(word);
    return next_main(word);
}

QwenWordStream::QwenWordStream(const char *text, size_t length)
    : begin_(text), end_(text + length)
{
}

bool QwenWordStream::match(size_t pos, size_t &len) const
{
    const size_t n = static_cast<size_t>(end_ - begin_);
    if (pos >= n) return false;
    size_t cl = 0;
    const uint32_t cp = decode_utf8(begin_ + pos, n - pos, &cl);
    if (cp == kInvalidUtf8) return false;

    // alt1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (MatchContractionAlt(begin_, end_, pos, len, /*case_insensitive=*/true))
        return true;

    // alt2: [^\r\n\p{L}\p{N}]? [\p{L}\p{M}]+  (Han/Kana are NOT isolated)
    const unicode::CharClass k0 = unicode::classify(cp);
    const bool is_prefix =
        cp != 0x0D && cp != 0x0A && k0 != unicode::CharClass::Letter &&
        k0 != unicode::CharClass::Number;
    {
        const size_t p = pos + (is_prefix ? cl : 0);
        if (p < n) {
            size_t q = 0;
            const uint32_t c2 = decode_utf8(begin_ + p, n - p, &q);
            const unicode::CharClass k2 = unicode::classify(c2);
            if (c2 != kInvalidUtf8 &&
                (k2 == unicode::CharClass::Letter ||
                 k2 == unicode::CharClass::Mark)) {
                len = ConsumeLetters(begin_, end_, p, /*include_marks=*/true) -
                      pos;
                return true;
            }
        }
    }

    // alt3: \p{N}
    if (k0 == unicode::CharClass::Number) {
        len = cl;
        return true;
    }

    // alt4: ' '? [^\s\p{L}\p{M}\p{N}]+ [\r\n]*
    {
        const size_t p0 = (cp == 0x20) ? pos + cl : pos;
        size_t p = p0;
        while (p < n) {
            size_t q = 0;
            const uint32_t c2 = decode_utf8(begin_ + p, n - p, &q);
            if (c2 == kInvalidUtf8) break;
            const unicode::CharClass k = unicode::classify(c2);
            if (k == unicode::CharClass::Whitespace ||
                k == unicode::CharClass::Letter ||
                k == unicode::CharClass::Mark ||
                k == unicode::CharClass::Number)
                break;
            p += q;
        }
        if (p > p0) {
            while (p < n && (begin_[p] == '\r' || begin_[p] == '\n')) ++p;
            len = p - pos;
            return true;
        }
    }

    return MatchWhitespaceAlts(begin_, end_, pos, len);
}

bool QwenWordStream::next(WordSpan &word)
{
    if (error_) return false;
    const size_t n = static_cast<size_t>(end_ - begin_);
    if (pos_ >= n) return false;
    size_t cl = 0;
    const uint32_t cp = decode_utf8(begin_ + pos_, n - pos_, &cl);
    if (cp == kInvalidUtf8) { error_ = 1; return false; }
    size_t len = 0;
    if (!match(pos_, len) || len == 0) len = cl;
    word = WordSpan{begin_ + pos_, len};
    pos_ += len;
    return true;
}

Gpt2WordStream::Gpt2WordStream(const char *text, size_t length)
    : begin_(text), end_(text + length)
{
}

bool Gpt2WordStream::match(size_t pos, size_t &len) const
{
    const size_t n = static_cast<size_t>(end_ - begin_);
    if (pos >= n) return false;
    size_t cl = 0;
    const uint32_t cp = decode_utf8(begin_ + pos, n - pos, &cl);
    if (cp == kInvalidUtf8) return false;

    // alt1: (?i:contraction)
    if (MatchContractionAlt(begin_, end_, pos, len, /*case_insensitive=*/true))
        return true;

    // alt2: [^\r\n\p{L}\p{N}]? \p{L}+
    const unicode::CharClass k0 = unicode::classify(cp);
    const bool is_prefix =
        cp != 0x0D && cp != 0x0A && k0 != unicode::CharClass::Letter &&
        k0 != unicode::CharClass::Number;
    {
        const size_t p = pos + (is_prefix ? cl : 0);
        if (p < n) {
            size_t q = 0;
            const uint32_t c2 = decode_utf8(begin_ + p, n - p, &q);
            if (c2 != kInvalidUtf8 &&
                unicode::classify(c2) == unicode::CharClass::Letter) {
                len = ConsumeLetters(begin_, end_, p, /*include_marks=*/false) -
                      pos;
                return true;
            }
        }
    }

    // alt3: \p{N}{1,3}
    if (k0 == unicode::CharClass::Number) {
        len = ConsumeDigits(begin_, end_, pos, 3) - pos;
        return true;
    }

    // alt4: ' '? [^\s\p{L}\p{N}]+ [\r\n]*
    {
        const size_t p0 = (cp == 0x20) ? pos + cl : pos;
        size_t p = p0;
        while (p < n) {
            size_t q = 0;
            const uint32_t c2 = decode_utf8(begin_ + p, n - p, &q);
            if (c2 == kInvalidUtf8) break;
            const unicode::CharClass k = unicode::classify(c2);
            if (k == unicode::CharClass::Whitespace ||
                k == unicode::CharClass::Letter ||
                k == unicode::CharClass::Number)
                break;
            p += q;
        }
        if (p > p0) {
            while (p < n && (begin_[p] == '\r' || begin_[p] == '\n')) ++p;
            len = p - pos;
            return true;
        }
    }

    return MatchWhitespaceAlts(begin_, end_, pos, len);
}

bool Gpt2WordStream::next(WordSpan &word)
{
    if (error_) return false;
    const size_t n = static_cast<size_t>(end_ - begin_);
    if (pos_ >= n) return false;
    size_t cl = 0;
    const uint32_t cp = decode_utf8(begin_ + pos_, n - pos_, &cl);
    if (cp == kInvalidUtf8) { error_ = 1; return false; }
    size_t len = 0;
    if (!match(pos_, len) || len == 0) len = cl;
    word = WordSpan{begin_ + pos_, len};
    pos_ += len;
    return true;
}

MistralGroupedWordStream::MistralGroupedWordStream(const char *text,
                                                   size_t length)
    : begin_(text), end_(text + length)
{
}

bool MistralGroupedWordStream::match(size_t pos, size_t &len) const
{
    // Mistral case-run pattern with o200k's RIGHT-grouped number alternative.
    return MatchCaseRunPattern(begin_, end_, pos, /*flint_punct=*/false, len,
                               /*digit_max=*/3, /*contraction=*/false,
                               /*stop_at_cjk=*/false,
                               /*slash_after_punct=*/true,
                               /*grouped_digits=*/true);
}

bool MistralGroupedWordStream::next(WordSpan &word)
{
    if (error_) return false;
    const size_t n = static_cast<size_t>(end_ - begin_);
    if (pos_ >= n) return false;
    size_t cl = 0;
    const uint32_t cp = decode_utf8(begin_ + pos_, n - pos_, &cl);
    if (cp == kInvalidUtf8) { error_ = 1; return false; }
    size_t len = 0;
    if (!match(pos_, len) || len == 0) len = cl;
    word = WordSpan{begin_ + pos_, len};
    pos_ += len;
    return true;
}

MistralWordStream::MistralWordStream(const char *text, size_t length)
    : begin_(text), end_(text + length)
{
}

bool MistralWordStream::match(size_t pos, size_t &len) const
{
    // The Llama-3 case-run alternatives WITHOUT a contraction suffix, ONE digit
    // per number word, and `[\r\n/]*` after the punctuation run.
    return MatchCaseRunPattern(begin_, end_, pos, /*flint_punct=*/false, len,
                               /*digit_max=*/1, /*contraction=*/false);
}

bool MistralWordStream::next(WordSpan &word)
{
    if (error_) return false;
    const size_t n = static_cast<size_t>(end_ - begin_);
    if (pos_ >= n) return false;
    size_t cl = 0;
    const uint32_t cp = decode_utf8(begin_ + pos_, n - pos_, &cl);
    if (cp == kInvalidUtf8) { error_ = 1; return false; }
    size_t len = 0;
    if (!match(pos_, len) || len == 0) len = cl;
    word = WordSpan{begin_ + pos_, len};
    pos_ += len;
    return true;
}

Gpt2OriginalWordStream::Gpt2OriginalWordStream(const char *text,
                                               size_t length)
    : begin_(text), end_(text + length)
{
}

bool Gpt2OriginalWordStream::match(size_t pos, size_t &len) const
{
    const size_t n = static_cast<size_t>(end_ - begin_);
    if (pos >= n) return false;
    size_t cl = 0;
    const uint32_t cp = decode_utf8(begin_ + pos, n - pos, &cl);
    if (cp == kInvalidUtf8) return false;

    // alt1: 's|'t|'re|'ve|'m|'ll|'d  (case-SENSITIVE in the GPT-2 pattern)
    if (MatchContractionAlt(begin_, end_, pos, len, /*case_insensitive=*/false))
        return true;

    const size_t p = (cp == 0x20) ? pos + cl : pos;   // '?': one ASCII space
    if (p < n) {
        size_t q = 0;
        const uint32_t c2 = decode_utf8(begin_ + p, n - p, &q);
        const unicode::CharClass k2 = unicode::classify(c2);
        // alt2: ' '? \p{L}+
        if (c2 != kInvalidUtf8 && k2 == unicode::CharClass::Letter) {
            len = ConsumeLetters(begin_, end_, p, /*include_marks=*/false) -
                  pos;
            return true;
        }
        // alt3: ' '? \p{N}+  (unbounded)
        if (c2 != kInvalidUtf8 && k2 == unicode::CharClass::Number) {
            len = ConsumeDigits(begin_, end_, p, 0) - pos;
            return true;
        }
        // alt4: ' '? [^\s\p{L}\p{N}]+
        size_t e = p;
        while (e < n) {
            size_t r = 0;
            const uint32_t c3 = decode_utf8(begin_ + e, n - e, &r);
            if (c3 == kInvalidUtf8) break;
            const unicode::CharClass k = unicode::classify(c3);
            if (k == unicode::CharClass::Whitespace ||
                k == unicode::CharClass::Letter ||
                k == unicode::CharClass::Number)
                break;
            e += r;
        }
        if (e > p) {
            len = e - pos;
            return true;
        }
    }

    // alt5-6: \s+(?!\S) | \s+   (the original pattern has no \s*[\r\n]+ form)
    return MatchWhitespaceAlts(begin_, end_, pos, len, /*crlf_alt=*/false);
}

bool Gpt2OriginalWordStream::next(WordSpan &word)
{
    if (error_) return false;
    const size_t n = static_cast<size_t>(end_ - begin_);
    if (pos_ >= n) return false;
    size_t cl = 0;
    const uint32_t cp = decode_utf8(begin_ + pos_, n - pos_, &cl);
    if (cp == kInvalidUtf8) { error_ = 1; return false; }
    size_t len = 0;
    if (!match(pos_, len) || len == 0) len = cl;
    word = WordSpan{begin_ + pos_, len};
    pos_ += len;
    return true;
}

// ── Kimi tiktoken pattern (tokenizer_kind 2) ───────────────────────────────
//
// `[\\p{Han}]+` is the first alternative in the reference pat_str, so a Han run
// is isolated before the case-run alternatives (which exclude Han); the
// punctuation tail is [\\r\\n]* (no slash). Everything else is the Llama-3
// case-run pattern, contraction included.
KimiWordStream::KimiWordStream(const char *text, size_t length)
    : begin_(text), end_(text + length)
{
}

bool KimiWordStream::match(size_t pos, size_t &len) const
{
    return MatchCaseRunPattern(begin_, end_, pos, /*flint_punct=*/false, len,
                               /*digit_max=*/3, /*contraction=*/true,
                               /*stop_at_cjk=*/true,
                               /*slash_after_punct=*/false);
}

bool KimiWordStream::next(WordSpan &word)
{
    if (error_) return false;
    const size_t n = static_cast<size_t>(end_ - begin_);
    if (pos_ >= n) return false;
    size_t cl = 0;
    const uint32_t cp = decode_utf8(begin_ + pos_, n - pos_, &cl);
    if (cp == kInvalidUtf8) { error_ = 1; return false; }
    if (is_cjk_run(cp)) {
        size_t p = pos_;
        while (p < n) {
            size_t q = 0;
            const uint32_t c2 = decode_utf8(begin_ + p, n - p, &q);
            if (c2 == kInvalidUtf8 || !is_cjk_run(c2)) break;
            p += q;
        }
        word = WordSpan{begin_ + pos_, p - pos_};
        pos_ = p;
        return true;
    }
    size_t len = 0;
    if (!match(pos_, len) || len == 0) len = cl;
    word = WordSpan{begin_ + pos_, len};
    pos_ += len;
    return true;
}

} // namespace lsxcommon
