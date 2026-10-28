#include "lsxcommon/unicode.h"
#include "unicode_tables.inc"

namespace lsxcommon {
namespace unicode {
namespace {

bool in_ranges(uint32_t cp, const CodeRange *ranges, size_t count)
{
    size_t lo = 0, hi = count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (ranges[mid].last < cp) lo = mid + 1;
        else if (ranges[mid].first > cp) hi = mid;
        else return true;
    }
    return false;
}

} // namespace

bool is_letter(uint32_t cp)     { return in_ranges(cp, kLetterRanges, kLetterRangesCount); }
bool is_mark(uint32_t cp)       { return in_ranges(cp, kMarkRanges, kMarkRangesCount); }
bool is_number(uint32_t cp)     { return in_ranges(cp, kNumberRanges, kNumberRangesCount); }
bool is_punct(uint32_t cp)      { return in_ranges(cp, kPunctRanges, kPunctRangesCount); }
bool is_symbol(uint32_t cp)     { return in_ranges(cp, kSymbolRanges, kSymbolRangesCount); }
bool is_whitespace(uint32_t cp) { return in_ranges(cp, kWhitespaceRanges, kWhitespaceRangesCount); }
bool is_lowercase_letter(uint32_t cp) { return in_ranges(cp, kLowercaseLetterRanges, kLowercaseLetterRangesCount); }
bool is_uppercase_letter(uint32_t cp) { return in_ranges(cp, kUppercaseLetterRanges, kUppercaseLetterRangesCount); }
bool is_titlecase_letter(uint32_t cp) { return in_ranges(cp, kTitlecaseLetterRanges, kTitlecaseLetterRangesCount); }

CharClass classify(uint32_t cp)
{
    if (is_letter(cp))   return CharClass::Letter;
    if (is_mark(cp))     return CharClass::Mark;
    if (is_number(cp))   return CharClass::Number;
    if (is_punct(cp))    return CharClass::Punct;
    if (is_symbol(cp))   return CharClass::Symbol;
    if (is_whitespace(cp)) return CharClass::Whitespace;
    return CharClass::Other;
}

} // namespace unicode
} // namespace lsxcommon
