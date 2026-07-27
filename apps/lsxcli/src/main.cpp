#include "lsxcommon/core.h"
#include "lsxcommon/common.h"
#include "lsxcommon/safetensors.h"
#include <cstring>
#include <spdlog/spdlog.h>
#include <string>

int main(int argc, char* argv[]) {
    std::string model_path = R"(D:\Programs\huggingface.co\DeepSeek-V4-Flash)";
    bool info = false;
    bool infofull = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--info") == 0) {
            info = true;
        } else if (std::strcmp(argv[i], "--infofull") == 0) {
            infofull = true;
        }
    }

    lsxcommon::Core::Initialize("lsxcli_runtime.log");
    lsxcommon::version();
    lsxcommon::log_gpu_info();
    spdlog::info("Model path: {}", model_path);

    if (info || infofull) {
        auto model = lsxcommon::scan_directory(model_path);
        if (infofull) {
            for (auto &shard : model.shards)
                lsxcommon::print_shard(shard);
        }
        lsxcommon::print_model(model);
    }

    lsxcommon::Core::Shutdown();
    return 0;
}
