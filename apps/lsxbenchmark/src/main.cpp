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
    // Engine auto policy: resolve the per-GPU perf knobs (TC fast-math,
    // expert-cache sizing, dual-stream staging) before any run — kAuto
    // scopes read the globals this session applies.
    lsxcommon::InferenceEngine::Session engine_session;
    lsxcommon::log::info(absl::StrFormat("Model path: %s", cli.model_path));

    lsxcommon::runtime::GenerationCliArgs args;
    args.model_path = cli.model_path;
    args.prompt = cli.prompt;
    // Paris selfcheck contract: 1-token request (the harness forces the
    // 1->2-token minimum internally, legacy lsxfabric --selfcheck parity);
    // bench honors --max-gen (default 2048).
    args.max_gen_tokens = cli.selfcheck ? 1 : cli.max_gen;
    // Numeric policy: kAuto = per-GPU maximum-speed perf policy;
    // kOff = bit-exact deterministic scalar path (--bit-exact).
    args.fast_math_mode = cli.bit_exact ? lsxcommon::FastMathMode::kOff
                                        : lsxcommon::FastMathMode::kAuto;

    const int rc = cli.selfcheck
                       ? lsxcommon::runtime::RunCliSelfCheck(args)
                       : lsxcommon::runtime::RunCliBenchmark(args);
    lsxcommon::Core::Shutdown();
    return rc;
}
