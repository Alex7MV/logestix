#include "lsxcommon/core.h"
#include "lsxcommon/common.h"
#include "lsxcommon/testmodel_gpu.h"
#include <cstring>
#include <spdlog/spdlog.h>
#include <string>
#include "validator.h"

int main(int argc, char* argv[]) {
    bool validate = false;
    bool testmodel = false;
    std::string model_path = R"(D:\Programs\huggingface.co\DeepSeek-V4-Flash)";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--localmodel") == 0 && i + 1 < argc)
            model_path = argv[++i];
        else if (std::strcmp(argv[i], "--validate") == 0)
            validate = true;
        else if (std::strcmp(argv[i], "--testmodel") == 0)
            testmodel = true;
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
    if (testmodel) {
        spdlog::info("Running model benchmark...");
        auto result = lsxcommon::gpu::RunTestModel(model_path);
        if (!result.ok) {
            spdlog::error("TestModel failed: {}", result.error_msg);
        } else {
            spdlog::info("=== --testmodel Results ===");
            spdlog::info("Tokens generated : {}", result.num_tokens);
            spdlog::info("Layers processed : {}",
                         result.num_layers * result.num_tokens);
            spdlog::info("Total time       : {:0.1f} ms", result.total_time_ms);
            spdlog::info("Avg per layer    : {:0.3f} ms",
                         result.avg_bus_latency_ms);
            spdlog::info("Tokens/sec       : {:0.1f} t/s",
                         result.tokens_per_sec);
            spdlog::info("DMA volume       : {} MiB",
                         result.total_bytes_transferred / (1024 * 1024));
        }
        lsxcommon::Core::Shutdown();
        return result.ok ? 0 : 1;
    }
    lsxcommon::Core::Shutdown();
    return 0;
}
