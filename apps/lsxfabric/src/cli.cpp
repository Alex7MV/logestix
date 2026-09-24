#include "cli.h"

#include "lsxcommon/path_registry.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace lsxfabric {

namespace {

bool IsKnownFlag(const char *s) {
    return std::strcmp(s, "--localmodel") == 0 ||
           std::strcmp(s, "--validate") == 0 ||
           std::strcmp(s, "--testprompt") == 0 ||
           std::strcmp(s, "--max-gen") == 0 ||
           std::strcmp(s, "--prompt") == 0;
}

bool ParsePositiveInt(const char *s, int &out) {
    errno = 0;
    char *end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < 1 || v > INT_MAX)
        return false;
    out = static_cast<int>(v);
    return true;
}

} // namespace

FabricCliArgs ParseFabricCliArgs(int argc, char *argv[]) {
    FabricCliArgs a;
    a.model_path = (std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                    "DeepSeek-V4-Flash")
                       .string();
    for (int i = 1; i < argc; ++i) {
        const bool has_value = (i + 1 < argc) && !IsKnownFlag(argv[i + 1]);
        if (std::strcmp(argv[i], "--localmodel") == 0) {
            if (has_value) a.model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--validate") == 0) {
            a.validate = true;
        } else if (std::strcmp(argv[i], "--testprompt") == 0) {
            a.testprompt = true;
        } else if (std::strcmp(argv[i], "--max-gen") == 0) {
            int v = 0;
            if (has_value && ParsePositiveInt(argv[i + 1], v)) {
                a.max_gen = v;
                ++i;
            }
        } else if (std::strcmp(argv[i], "--prompt") == 0) {
            if (has_value) a.prompt = argv[++i];
        }
    }
    return a;
}

} // namespace lsxfabric
