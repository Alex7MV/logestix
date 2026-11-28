#pragma once

#include <string>

namespace lsxbenchmark {

// lsxbenchmark launch parameters. This app owns generation-speed measurement
// (--bench) and bit-exactness verification (--selfcheck [--bit-exact]);
// every other app always runs the per-GPU maximum-speed auto perf policy.
struct BenchmarkCliArgs {
    std::string model_path;
    bool bench = true;      // default mode when no mode flag is given
    bool selfcheck = false; // --selfcheck: Paris verification harness
    bool fsi_selfcheck = false; // --fsi-selfcheck: FSI golden-parity gate
    bool code_selfcheck = false; // --code-selfcheck: CODE golden-parity gate
    bool det_selfcheck = false; // --det-selfcheck: DET determinism gate
    bool mcp_selfcheck = false; // --mcp-selfcheck: MCP protocol-payload gate
    bool bit_exact = false; // --bit-exact: FastMathMode::kOff for the active mode
    std::string prompt = "The capital of France is";
    int max_gen = 2048;
};

// Parses argv (argv[0] is skipped). Unknown tokens are silently ignored.
// Mode rule: --selfcheck / --fsi-selfcheck / --code-selfcheck /
// --det-selfcheck / --mcp-selfcheck select verification mode; anything else
// (including no mode flag) is bench mode.
BenchmarkCliArgs ParseBenchmarkCliArgs(int argc, char *argv[]);

} // namespace lsxbenchmark
