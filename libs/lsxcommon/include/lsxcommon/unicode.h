#pragma once

#include <cstdint>

namespace lsxcommon {
namespace unicode {

enum class CharClass { Letter, Mark, Number, Punct, Symbol, Whitespace, Other };

CharClass classify(uint32_t cp);
bool is_letter(uint32_t cp);
bool is_mark(uint32_t cp);
bool is_number(uint32_t cp);
bool is_punct(uint32_t cp);
bool is_symbol(uint32_t cp);
bool is_whitespace(uint32_t cp);

} // namespace unicode
} // namespace lsxcommon
