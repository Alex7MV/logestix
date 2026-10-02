#pragma once

// lsxforge pipeline dispatch — the flag-to-pipeline decision, separated from
// main() so it is unit-testable (apps/lsxfabric/cli.h and
// apps/lsxbenchmark/cli.h follow the same split).
//
// main() keeps only the execution of whatever ResolvePipeline() selects; the
// precedence below therefore IS the tool's dispatch order and is pinned by
// tests/test_lsxforge_cli.cpp.

#include <string>

namespace lsxforge {

// Every executable path of lsxforge. kGeneric is the default safetensors
// converter; kError means the flag combination is not a valid invocation.
enum class Pipeline {
    kVersion,
    kGenerate,
    kTokenizer,
    kVramRecompute,
    kRepackShared,
    kRepackRouted,
    kDeepseekNvfp4,
    kDeepseekV41,
    kGlm,
    kKimi,
    kQwenDense,
    kQwen35Moe,
    kQwen36Moe,
    kOrnith,
    kQwen38FlashNext,
    kGemma4,
    kNemotron,
    kGigachat,
    kGlmFlash,
    kGranite,
    kPhi,
    kMimo,
    kAlice,
    kGeneric,
    kError,
};

// The absl flags of lsxforge, copied out of the global flag registry so the
// dispatcher stays a pure function of its input.
struct Flags {
    std::string input;
    std::string output;
    std::string tokenizer;
    bool repack_shared = false;
    bool repack_experts = false;
    bool vram_recompute = false;
    bool deepseek_nvfp4 = false;
    bool deepseek_v41 = false;
    bool run = false;
    bool glm = false;
    bool kimi = false;
    bool qwen = false;
    bool qwen35moe = false;
    bool qwen36moe = false;
    bool ornith = false;
    bool qwen38flashnext = false;
    bool gemma4 = false;
    bool nemotron = false;
    bool gigachat = false;
    bool glmflash = false;
    bool granite = false;
    bool phi = false;
    bool mimo = false;
    bool alice = false;
};

struct Resolution {
    Pipeline pipeline = Pipeline::kGeneric;
    // Empty unless pipeline == kError; a human-readable reason for the CLI.
    std::string error;
};

// Precedence (first match wins):
//   version banner (no --input) -> --run -> --tokenizer -> --vram-recompute
//   -> --repack_shared -> --repack_experts -> --deepseek_nvfp4
//   -> --deepseek_v41 -> family flags -> generic converter.
// Selecting two different families is an error rather than a silent
// first-flag-wins, because running the wrong converter is unrecoverable.
Resolution ResolvePipeline(const Flags &flags);

}  // namespace lsxforge