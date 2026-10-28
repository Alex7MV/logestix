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
// General-category case properties (Ll, Lu, Lt) — needed by the Llama-3 /
// o200k-style pre-tokenizer, whose first two alternatives split on case runs.
bool is_lowercase_letter(uint32_t cp);
bool is_uppercase_letter(uint32_t cp);
bool is_titlecase_letter(uint32_t cp);

} // namespace unicode
} // namespace lsxcommon
