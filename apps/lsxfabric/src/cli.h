#pragma once

#include <string>

namespace lsxfabric {

// lsxfabric launch parameters. The engine always runs the per-GPU
// maximum-speed auto perf policy. Speed measurement and bit-exactness
// verification live in lsxbenchmark; this surface is validate + generate.
struct FabricCliArgs {
    std::string model_path;
    std::string prompt = "The capital of France is";
    int max_gen = 64;
    bool validate = false;
    bool testprompt = false;
    bool serve = false;
    std::string listen = "tcp://*:5555";
};

// Parses argv (argv[0] is skipped). Unknown tokens — including the
// benchmark-era --benchmark/--selfcheck/--bit-exact and the removed
// --fast-math/--no-fast-math/--calibrate — are silently ignored.
FabricCliArgs ParseFabricCliArgs(int argc, char *argv[]);

} // namespace lsxfabric
