#pragma once

#include <string>

namespace lsxfabric {

// lsxfabric launch parameters. The engine always runs the per-GPU
// maximum-speed auto perf policy; the single numeric-mode switch is
// bit_exact (deterministic scalar path for 100% answer verification).
struct FabricCliArgs {
    std::string model_path;
    std::string prompt = "The capital of France is";
    int max_gen = 64;
    bool validate = false;
    bool benchmark = false;
    bool testprompt = false;
    bool selfcheck = false;
    bool bit_exact = false;
};

// Parses argv (argv[0] is skipped). Unknown tokens — including the removed
// --fast-math/--no-fast-math/--calibrate — are silently ignored.
FabricCliArgs ParseFabricCliArgs(int argc, char *argv[]);

} // namespace lsxfabric
