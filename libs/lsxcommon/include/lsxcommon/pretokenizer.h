#pragma once
#include <cstddef>
#include <cstdint>

namespace lsxcommon {

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

} // namespace lsxcommon
