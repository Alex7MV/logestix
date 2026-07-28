#include "lsxcommon/core.h"
#include "lsxcommon/common.h"
#include <cstring>
#include <spdlog/spdlog.h>
#include <string>
#include "validator.h"

int main(int argc, char* argv[]) {
    bool validate = false;
    std::string model_path = R"(D:\Programs\huggingface.co\DeepSeek-V4-Flash)";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--localmodel") == 0 && i + 1 < argc)
            model_path = argv[++i];
        else if (std::strcmp(argv[i], "--validate") == 0)
            validate = true;
    }

    lsxcommon::Core::Initialize("logestix_runtime.log");
    lsxcommon::version();
    lsxcommon::log_gpu_info();
    spdlog::info("Model path: {}", model_path);
    if (validate) {
        spdlog::info("Running validation...");
        auto result = lsxfabric::RunValidator(model_path);
        lsxcommon::Core::Shutdown();
        return result.ok ? 0 : 1;
    }
    lsxcommon::Core::Shutdown();
    return 0;
}
