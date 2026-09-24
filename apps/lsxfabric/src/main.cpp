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

    int rc = 0;
    // Scoped engine session: the per-GPU auto perf policy is resolved for
    // every launch mode, and ShutdownEngine (which logs) runs while the
    // spdlog default logger installed by Core::Initialize is still alive.
    {
        lsxcommon::InferenceEngine::Session engine_session;
        lsxcommon::log::info(absl::StrFormat("Model path: %s", cli.model_path));
        if (cli.testprompt) {
            lsxcommon::runtime::GenerationCliArgs args;
            args.model_path = cli.model_path;
            args.prompt = cli.prompt;
            args.max_gen_tokens = cli.max_gen;
            args.fast_math_mode = lsxcommon::FastMathMode::kAuto;
            rc = lsxcommon::runtime::RunCliGeneration(args);
        } else if (cli.validate) {
            lsxcommon::log::info("Running validation...");
            rc = lsxfabric::RunModelValidation(cli.model_path) ? 0 : 1;
        } else {
            lsxcommon::log::error(
                "No run mode selected: pass --validate or --testprompt");
            rc = 1;
        }
    }
    lsxcommon::Core::Shutdown();
    return rc;
}
