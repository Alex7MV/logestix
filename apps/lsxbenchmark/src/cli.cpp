#include "cli.h"

#include "lsxcommon/path_registry.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace lsxbenchmark {

BenchmarkCliArgs ParseBenchmarkCliArgs(int argc, char *argv[]) {
    BenchmarkCliArgs a;
    a.model_path = (std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                    "DeepSeek-V4-Flash")
                       .string();
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
