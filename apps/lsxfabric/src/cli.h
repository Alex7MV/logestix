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
    // --explain-artifact: print why this model would load the profile it would
    // load, and exit. The only diagnostic flag added by the codec contract
    // (design §2.7); it never loads a model, so it is safe beside a running one.
    bool explain_artifact = false;
    bool serve = false;
    std::string listen = "tcp://*:5555";
    std::string listen_cancel = "tcp://*:5556";
};

// Parses argv (argv[0] is skipped). Unknown tokens — including the
// benchmark-era --benchmark/--selfcheck/--bit-exact and the removed
// --fast-math/--no-fast-math/--calibrate — are silently ignored.
FabricCliArgs ParseFabricCliArgs(int argc, char *argv[]);

} // namespace lsxfabric
