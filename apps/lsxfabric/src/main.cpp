#include "lsxcommon/core.h"
#include <absl/strings/str_format.h>
#include "lsxcommon/common.h"
#include "lsxcommon/benchmark_gpu.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/log.h"
#include "cli.h"
#include "validator.h"

int main(int argc, char *argv[]) {
    const lsxfabric::FabricCliArgs cli =
        lsxfabric::ParseFabricCliArgs(argc, argv);

    lsxcommon::Core::Initialize("logestix_runtime.log");
    lsxcommon::version();
    lsxcommon::log_gpu_info();
    lsxcommon::log::info(absl::StrFormat("Model path: %s", cli.model_path));
    const lsxcommon::FastMathMode fm = cli.bit_exact
                                           ? lsxcommon::FastMathMode::kOff
                                           : lsxcommon::FastMathMode::kAuto;
    if (cli.testprompt) {
        lsxcommon::runtime::GenerationCliArgs args;
        args.model_path = cli.model_path;
        args.prompt = cli.prompt;
        args.max_gen_tokens = cli.max_gen;
        args.fast_math_mode = fm;
        int rc = lsxcommon::runtime::RunCliGeneration(args);
        lsxcommon::Core::Shutdown();
        return rc;
    }
    if (cli.selfcheck) {
        lsxcommon::runtime::GenerationCliArgs args;
        args.model_path = cli.model_path;
        args.prompt = cli.prompt;
        args.max_gen_tokens = 1;
        args.fast_math_mode = fm;
        int rc = lsxcommon::runtime::RunCliSelfCheck(args);
        lsxcommon::Core::Shutdown();
        return rc;
    }
    if (cli.validate) {
        lsxcommon::log::info("Running validation...");
        auto result = lsxfabric::RunValidator(cli.model_path);
        lsxcommon::Core::Shutdown();
        return result.ok ? 0 : 1;
    }
    if (cli.benchmark) {
        lsxcommon::log::info("Running model benchmark...");
        auto result = lsxcommon::gpu::RunBenchmark(cli.model_path, 2048);
        if (!result.ok) {
            lsxcommon::log::error(absl::StrFormat("Benchmark failed: %s", result.error_msg));
        } else {
            lsxcommon::log::info("=== --benchmark Results ===");
            lsxcommon::log::info(absl::StrFormat("Tokens generated : %d", result.num_tokens));
            lsxcommon::log::info(absl::StrFormat("Layers processed : %d",
                         result.num_layers * result.num_tokens));
            lsxcommon::log::info(absl::StrFormat("Total time       : %.1f ms", result.total_time_ms));
            lsxcommon::log::info(absl::StrFormat("Avg per layer    : %.3f ms",
                         result.avg_bus_latency_ms));
            lsxcommon::log::info(absl::StrFormat("Tokens/sec       : %.1f t/s",
                         result.tokens_per_sec));
            lsxcommon::log::info(absl::StrFormat("DMA volume       : %d MiB",
                         static_cast<int>(result.total_bytes_transferred / (1024 * 1024))));
        }
        lsxcommon::Core::Shutdown();
        return result.ok ? 0 : 1;
    }
    lsxcommon::Core::Shutdown();
    return 0;
}
