#pragma once

#include <string>
#include <vector>

namespace lsxcommon {
namespace cli {

// Blueprint prompt injected when the generation-path flag (--run / -r) is
// supplied without an explicit value.
constexpr const char* kDefaultGenerationPrompt =
    "Explain the phrase: That's one small step for a man, one giant leap for mankind";

// Returns the effective generation prompt: an empty run value falls back to
// the default blueprint prompt.
std::string ResolveGenerationPrompt(const std::string& run_value);

// Canonicalizes argv before absl::ParseCommandLine so the generation flag is
// always spelled as --run=<value>:
//   -r Hello       -> --run Hello          (short alias, space-separated value)
//   -r=Hello       -> --run=Hello
//   -r             -> --run=<default>      (last token)
//   --run Hello    -> --run Hello          (absl consumes the next value)
//   --run=Hello    -> --run=Hello
//   --run          -> --run=<default>      (last token / followed by a flag)
//   --run=         -> --run=<default>
std::vector<std::string> NormalizeGenerationArgv(int argc, char* argv[]);

} // namespace cli
} // namespace lsxcommon
