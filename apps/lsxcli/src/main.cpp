#include "lsxcommon/core.h"
#include "lsxcommon/common.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/model.h"
#include "lsxcommon/model_factory.h"
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
        lsxcommon::ModelInitConfig cfg;
        cfg.model_path = model_path;
        auto model = lsxcommon::LsxModelFactory::Create(cfg);
        if (!model || !model->Load()) {
            spdlog::error("Failed to create/load model: {}", model_path);
            lsxcommon::Core::Shutdown();
            return 1;
        }
        auto mi = model->Inspect();
        if (infofull) {
            for (auto &shard : mi.shards)
                lsxcommon::print_shard(shard);
        }
        lsxcommon::print_model(mi);
        model->Unload();
    }

    lsxcommon::Core::Shutdown();
    return 0;
}
