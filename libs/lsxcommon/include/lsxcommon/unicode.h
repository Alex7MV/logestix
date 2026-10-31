#pragma once

#include <cstdint>
#include <string>
#include <string_view>

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

// Canonical NFC normalization (UAX #15). The converted tokenizers whose
// tokenizer.json declares an NFC normalizer must normalize the text BEFORE the
// pre-tokenizer runs, exactly like the reference pipeline; otherwise a
// decomposed accent ("e" + U+0301) tokenizes differently from its composed
// form. Invalid UTF-8 is returned unchanged (the encoder rejects it later).
std::string NfcNormalize(std::string_view input);

} // namespace unicode
} // namespace lsxcommon
