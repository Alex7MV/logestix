#include "cli.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>

namespace taskmosaic {
namespace {

bool HasValue(int argc, char** argv, int i) {
    return (i + 1 < argc) && (argv[i + 1][0] != '-');
}

bool ParsePositiveInt(const char* s, int& out) {
    errno = 0;
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < 1 || v > INT_MAX)
        return false;
    out = static_cast<int>(v);
    return true;
}

}  // namespace

TaskMosaicCliArgs ParseTaskMosaicCliArgs(int argc, char** argv) {
    TaskMosaicCliArgs a;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--server") == 0) {
            if (HasValue(argc, argv, i)) a.server = argv[++i];
        } else if (std::strcmp(argv[i], "--model") == 0) {
            if (HasValue(argc, argv, i)) a.model = argv[++i];
        } else if (std::strcmp(argv[i], "--prompt") == 0) {
            if (HasValue(argc, argv, i)) a.prompt = argv[++i];
        } else if (std::strcmp(argv[i], "--max-gen") == 0) {
            int v = 0;
            if (HasValue(argc, argv, i) && ParsePositiveInt(argv[i + 1], v)) {
                a.max_gen = v;
                ++i;
            }
        }
    }
    return a;
}

}  // namespace taskmosaic
