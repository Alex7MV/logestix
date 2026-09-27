#include "lsxcommon/core.h"
#include <absl/strings/str_format.h>
#include "lsxcommon/common.h"
#include "lsxcommon/model_engine.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/log.h"
#include "cli.h"

int main(int argc, char *argv[]) {
    const lsxbenchmark::BenchmarkCliArgs cli =
        lsxbenchmark::ParseBenchmarkCliArgs(argc, argv);

    lsxcommon::Core::Initialize("lsxbenchmark_runtime.log");
    lsxcommon::version();
    lsxcommon::log_gpu_info();

    int rc = 0;
    // Scoped engine session: the per-GPU auto perf policy is resolved for
    // every launch mode, and ShutdownEngine (which logs) runs while the
    // spdlog default logger installed by Core::Initialize is still alive.
    {
        lsxcommon::InferenceEngine::Session engine_session;
        lsxcommon::log::info(absl::StrFormat("Model path: %s", cli.model_path));

        lsxcommon::runtime::GenerationCliArgs args;
        args.model_path = cli.model_path;
        args.prompt = cli.prompt;
        // Paris selfcheck contract: 1-token request (the harness forces the
        // 1->2-token minimum internally, legacy lsxfabric --selfcheck parity);
        // bench honors --max-gen (default 2048).
        args.max_gen_tokens =
            (cli.selfcheck || cli.fsi_selfcheck || cli.code_selfcheck ||
             cli.det_selfcheck)
                ? 1
                : cli.max_gen;
        // Numeric policy: kAuto = per-GPU maximum-speed perf policy;
        // kOff = bit-exact deterministic scalar path (--bit-exact).
        args.fast_math_mode = cli.bit_exact ? lsxcommon::FastMathMode::kOff
                                            : lsxcommon::FastMathMode::kAuto;

        if (cli.det_selfcheck)
            rc = lsxcommon::runtime::RunCliDetSelfCheck(args);
        else if (cli.code_selfcheck)
            rc = lsxcommon::runtime::RunCliCodeSelfCheck(args);
        else if (cli.fsi_selfcheck)
            rc = lsxcommon::runtime::RunCliFsiSelfCheck(args);
        else if (cli.selfcheck)
            rc = lsxcommon::runtime::RunCliSelfCheck(args);
        else
            rc = lsxcommon::runtime::RunCliBenchmark(args);
    }
    lsxcommon::Core::Shutdown();
    return rc;
}
