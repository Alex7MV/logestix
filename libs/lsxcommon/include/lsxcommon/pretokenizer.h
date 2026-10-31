#pragma once
#include <cstddef>
#include <cstdint>

namespace lsxcommon {

// Strict UTF-8 codepoint decoder, shared with arrow_tokenizer_decode so the
// decoder and the encoder can never disagree on what "valid text" means:
// overlong forms, surrogates and codepoints above U+10FFFF are rejected.
// On any malformed sequence returns kInvalidUtf8 and sets *len = 1.
constexpr uint32_t kInvalidUtf8 = 0xFFFFFFFFu;
uint32_t decode_utf8(const char *p, size_t n, size_t *len);

// Length of the sequence the lead byte @p b starts, or 0 when it cannot start
// one. Shape only — decode_utf8 decides validity, so the overlong / surrogate /
// out-of-range rules stay in one place.
size_t utf8_expected_len(uint8_t b);

// Zero-copy byte span referencing the input text passed to WordStream.
struct WordSpan {
    const char *begin = nullptr;
    size_t length = 0;
};

// Single-pass streaming pre-tokenizer. Reproduces the GPT-2 3-split Sequence:
//   1. \p{N}{1,3} Isolated               (number runs, chunks of <= 3 codepoints)
//   2. [Han|Hiragana|Katakana]+ Isolated (U+4E00-U+9FA5, U+3040-U+309F, U+30A0-U+30FF)
//   3. main 6-alternative regex + Isolated gap words
// next() returns WordSpans pointing into the input buffer - no allocation,
// no copies, valid UTF-8 only. On a malformed UTF-8 sequence next() returns
// false and error() is set to 1.
class WordStream {
public:
    WordStream(const char *text, size_t length);

    bool next(WordSpan &word);
    uint32_t error() const { return error_; }

private:
    bool next_number(WordSpan &word);   // split 1 (<= 3 \p{N} codepoints)
    bool next_cjk(WordSpan &word);      // split 2 (one CJK-run word)
    bool next_main(WordSpan &word);     // split 3 (one alt match or one gap word)

    const char *const begin_;
    const char *const end_;
    size_t pos_ = 0;
    uint32_t error_ = 0;
};

// Llama-3 / o200k-style pre-tokenizer (Qwen2/3, MiniMax-M3, Llama-3). Same
// streaming contract as WordStream, but a DIFFERENT pattern: no CJK split, and
// the first two alternatives split on case runs, which is why it needs the
// Unicode Lu/Ll/Lt tables. The alternation is the reference one:
//
//   1. [^\r\n\p{L}\p{N}]? [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*
//      [\p{Ll}\p{Lm}\p{Lo}\p{M}]+ (?i:'s|'t|'re|'ve|'m|'ll|'d)?
//   2. [^\r\n\p{L}\p{N}]? [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+
//      [\p{Ll}\p{Lm}\p{Lo}\p{M}]* (?i:'s|'t|'re|'ve|'m|'ll|'d)?
//   3. \p{N}{1,3}
//   4. ' '? [^\s\p{L}\p{N}]+ [\r\n/]*
//   5. \s*[\r\n]+
//   6. \s+(?!\S)
//   7. \s+
//
// Alternation is leftmost-first, so the case-run alternatives win over the
// generic letter run ("getUserName" -> "get", "User", "Name").
class Llama3WordStream {
public:
    Llama3WordStream(const char *text, size_t length);

    bool next(WordSpan &word);
    uint32_t error() const { return error_; }

private:
    bool match(size_t pos, size_t &len) const;
    size_t consume_letter_run(size_t pos) const;
    size_t consume_case_run(size_t pos) const;   // Lu/Lt/Lm/Lo/M* Ll/Lm/Lo/M+
    size_t consume_upper_run(size_t pos) const;  // Lu/Lt/Lm/Lo/M+ Ll/Lm/Lo/M*
    size_t consume_contraction(size_t pos) const;

    const char *const begin_;
    const char *const end_;
    size_t pos_ = 0;
    uint32_t error_ = 0;
};

// Flint pre-tokenizer (tokenizer_kind 4). A three-split Sequence of the o200k
// case-run pattern over a differently-shaped tail:
//
//   1. \p{N}{1,3} Isolated                    (as in WordStream)
//   2. [Han|Hiragana|Katakana]+ Isolated      (as in WordStream)
//   3. the Llama3WordStream alternatives 1, 2, 5, 6, 7 with Flint's
//      punctuation alternative ' '? [\p{P}\p{S}]+ [\r\n]*
//      (o200k uses ' '? [^\s\p{L}\p{N}]+ [\r\n/]* - different for marks,
//      control characters and a slash before a newline)
//
// Numbers are consumed by split 1, so the pattern's own \p{N}{1,3} alternative
// can never fire here. Same streaming contract as the other two streams.
class FlintWordStream {
public:
    FlintWordStream(const char *text, size_t length);

    bool next(WordSpan &word);
    uint32_t error() const { return error_; }

private:
    bool next_number(WordSpan &word);   // split 1 (<= 3 \p{N} codepoints)
    bool next_cjk(WordSpan &word);      // split 2 (one CJK-run word)
    bool next_main(WordSpan &word);     // split 3 (one case-run match)

    const char *const begin_;
    const char *const end_;
    size_t pos_ = 0;
    uint32_t error_ = 0;
};

// Beam-1.0 pre-tokenizer (tokenizer_kind 6). A SINGLE-run Qwen2/3-style
// pattern:
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]? \p{L}+ | \p{N} |
//   ' '? [^\s\p{L}\p{N}]+ [\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
//
// One letter run (no case split, no CJK isolation), ONE digit per number word,
// and no `\p{M}` in the letter/punctuation classes. Same streaming contract as
// the other streams.
class BeamWordStream {
public:
    BeamWordStream(const char *text, size_t length);

    bool next(WordSpan &word);
    uint32_t error() const { return error_; }

private:
    bool match(size_t pos, size_t &len) const;

    const char *const begin_;
    const char *const end_;
    size_t pos_ = 0;
    uint32_t error_ = 0;
};

// Ling-3.1-flash pre-tokenizer (tokenizer_kind 5). One split over the
// reference pattern
//
//   '(?i:[sdmt]|ll|ve|re) | [^\r\n\p{L}\p{N}]?+ \p{L}+ | \p{N} |
//   ' '? [^\s\p{L}\p{N}]++ [\r\n]* | \s*[\r\n] | \s+(?!\S) | \s+
//
// A SINGLE letter run (no case split, no CJK isolation) and ONE digit per
// number word. Same streaming contract as the other streams.
class LingWordStream {
public:
    LingWordStream(const char *text, size_t length);

    bool next(WordSpan &word);
    uint32_t error() const { return error_; }

private:
    bool match(size_t pos, size_t &len) const;

    const char *const begin_;
    const char *const end_;
    size_t pos_ = 0;
    uint32_t error_ = 0;
};

} // namespace lsxcommon
