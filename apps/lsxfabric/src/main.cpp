#include "lsxcommon/core.h"
#include <absl/strings/str_format.h>
#include "lsxcommon/common.h"
#include "lsxcommon/model_engine.h"
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
    // Engine auto policy: resolve the per-GPU perf knobs (TC fast-math,
    // expert-cache sizing, dual-stream staging) for every launch mode.
    lsxcommon::InferenceEngine::Session engine_session;
    lsxcommon::log::info(absl::StrFormat("Model path: %s", cli.model_path));
    if (cli.testprompt) {
        lsxcommon::runtime::GenerationCliArgs args;
        args.model_path = cli.model_path;
        args.prompt = cli.prompt;
        args.max_gen_tokens = cli.max_gen;
        args.fast_math_mode = lsxcommon::FastMathMode::kAuto;
        int rc = lsxcommon::runtime::RunCliGeneration(args);
        lsxcommon::Core::Shutdown();
        return rc;
    }
    if (cli.validate) {
        lsxcommon::log::info("Running validation...");
        auto result = lsxfabric::RunValidator(cli.model_path);
        lsxcommon::Core::Shutdown();
        return result.ok ? 0 : 1;
    }
    lsxcommon::Core::Shutdown();
    return 0;
}
