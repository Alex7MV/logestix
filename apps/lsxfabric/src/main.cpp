#include "lsxcommon/core.h"
#include "lsxcommon/common.h"
#include "lsxcommon/benchmark_gpu.h"
#include "lsxcommon/model.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/tokenizer.h"
#include "lsxcommon/path_registry.h"
#include <cstring>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>
#include "validator.h"

int main(int argc, char* argv[]) {
    bool validate = false;
    bool benchmark = false;
    bool testprompt = false;
    bool selfcheck = false;
    auto model_path = (std::filesystem::path(lsxcommon::lsx_model_models_dir())
                       / "DeepSeek-V4-Flash").string();
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--localmodel") == 0 && i + 1 < argc)
            model_path = argv[++i];
        else if (std::strcmp(argv[i], "--validate") == 0)
            validate = true;
        else if (std::strcmp(argv[i], "--benchmark") == 0)
            benchmark = true;
        else if (std::strcmp(argv[i], "--testprompt") == 0)
            testprompt = true;
        else if (std::strcmp(argv[i], "--selfcheck") == 0)
            selfcheck = true;
    }

    lsxcommon::Core::Initialize("logestix_runtime.log");
    lsxcommon::version();
    lsxcommon::log_gpu_info();
    spdlog::info("Model path: {}", model_path);
    if (testprompt) {
        lsxcommon::runtime::GenerationCliArgs args;
        args.model_path = model_path;
        args.prompt = "The capital of France is";
        args.max_gen_tokens = 64;
        int rc = lsxcommon::runtime::RunCliGeneration(args);
        lsxcommon::Core::Shutdown();
        return rc;
    }
    if (selfcheck) {
        lsxcommon::runtime::GenerationCliArgs args;
        args.model_path = model_path;
        args.prompt = "The capital of France is";
        args.max_gen_tokens = 1;
        int rc = lsxcommon::runtime::RunCliSelfCheck(args);
        lsxcommon::Core::Shutdown();
        return rc;
    }
    if (validate) {
        spdlog::info("Running validation...");
        auto result = lsxfabric::RunValidator(model_path);
        lsxcommon::Core::Shutdown();
        return result.ok ? 0 : 1;
    }
    if (benchmark) {
        spdlog::info("Running model benchmark...");
        auto result = lsxcommon::gpu::RunBenchmark(model_path);
        if (!result.ok) {
            spdlog::error("Benchmark failed: {}", result.error_msg);
        } else {
            spdlog::info("=== --benchmark Results ===");
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
