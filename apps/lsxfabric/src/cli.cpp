#include "cli.h"

#include "lsxcommon/path_registry.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace lsxfabric {

FabricCliArgs ParseFabricCliArgs(int argc, char *argv[]) {
    FabricCliArgs a;
    a.model_path = (std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                    "DeepSeek-V4-Flash")
                       .string();
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--localmodel") == 0 && i + 1 < argc)
            a.model_path = argv[++i];
        else if (std::strcmp(argv[i], "--validate") == 0)
            a.validate = true;
        else if (std::strcmp(argv[i], "--testprompt") == 0)
            a.testprompt = true;
        else if (std::strcmp(argv[i], "--max-gen") == 0 && i + 1 < argc)
            a.max_gen = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc)
            a.prompt = argv[++i];
    }
    return a;
}

} // namespace lsxfabric
