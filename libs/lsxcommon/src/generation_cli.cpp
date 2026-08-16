#include "lsxcommon/generation_cli.h"

#include <cctype>
#include <cstring>
#include <string>
#include <vector>

namespace lsxcommon {
namespace cli {

std::string ResolveGenerationPrompt(const std::string& run_value)
{
    bool blank = true;
    for (char c : run_value) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            blank = false;
            break;
        }
    }
    if (blank) return kDefaultGenerationPrompt;
    return run_value;
}

namespace {

constexpr const char* kRunLongFlag  = "--run";
constexpr const char* kRunShortFlag = "-r";

bool HasInlineValue(const char* arg)
{
    return arg && std::strncmp(arg, "--run=", 6) == 0;
}

} // namespace

std::vector<std::string> NormalizeGenerationArgv(int argc, char* argv[])
{
    std::vector<std::string> out;
    out.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        const char* arg = argv[i] ? argv[i] : "";

        if (std::strcmp(arg, kRunShortFlag) == 0 ||
            std::strcmp(arg, kRunLongFlag) == 0) {
            // Bare -r/--run: absl consumes the next argv token as the value
            // only when it does not look like another flag.
            const bool has_value =
                i + 1 < argc && argv[i + 1] && argv[i + 1][0] != '\0' &&
                argv[i + 1][0] != '-';
            if (has_value) {
                out.emplace_back(kRunLongFlag);
            } else {
                out.emplace_back(std::string(kRunLongFlag) + "=" +
                                 kDefaultGenerationPrompt);
            }
            continue;
        }

        if (std::strncmp(arg, "-r=", 3) == 0) {
            const char* value = arg + 3;
            if (*value == '\0') {
                out.emplace_back(std::string(kRunLongFlag) + "=" +
                                 kDefaultGenerationPrompt);
            } else {
                out.emplace_back(std::string(kRunLongFlag) + "=" + value);
            }
            continue;
        }

        if (HasInlineValue(arg)) {
            const char* value = arg + 6;
            if (*value == '\0') {
                out.emplace_back(std::string(kRunLongFlag) + "=" +
                                 kDefaultGenerationPrompt);
            } else {
                out.emplace_back(arg);
            }
            continue;
        }

        out.emplace_back(arg);
    }
    return out;
}

} // namespace cli
} // namespace lsxcommon
