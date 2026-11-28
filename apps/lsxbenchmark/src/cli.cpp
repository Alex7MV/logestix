#include "cli.h"

#include "lsxcommon/path_registry.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace lsxbenchmark {

BenchmarkCliArgs ParseBenchmarkCliArgs(int argc, char *argv[]) {
    BenchmarkCliArgs a;
    // A model IDENTITY, not a path. `lsx::storage::Resolve` searches the
    // artifact store for it (and reports a source-only checkpoint as the
    // fixable thing it is). The old default built
    // `<source root>/DeepSeek-V4-Flash` — without the org the store nests it
    // under, and pointing at the read-only checkpoint rather than the
    // converted tree — so the default could never load.
    a.model_path = "deepseek-v4-flash";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            a.model_path = argv[++i];
        else if (std::strcmp(argv[i], "--bench") == 0)
            a.bench = true;
        else if (std::strcmp(argv[i], "--selfcheck") == 0) {
            a.selfcheck = true;
            a.bench = false;
        } else if (std::strcmp(argv[i], "--fsi-selfcheck") == 0) {
            a.fsi_selfcheck = true;
            a.bench = false;
        } else if (std::strcmp(argv[i], "--code-selfcheck") == 0) {
            a.code_selfcheck = true;
            a.bench = false;
        } else if (std::strcmp(argv[i], "--det-selfcheck") == 0) {
            a.det_selfcheck = true;
            a.bench = false;
        } else if (std::strcmp(argv[i], "--mcp-selfcheck") == 0) {
            a.mcp_selfcheck = true;
            a.bench = false;
        } else if (std::strcmp(argv[i], "--bit-exact") == 0)
            a.bit_exact = true;
        else if (std::strcmp(argv[i], "--max-gen") == 0 && i + 1 < argc)
            a.max_gen = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc)
            a.prompt = argv[++i];
    }
    return a;
}

} // namespace lsxbenchmark
