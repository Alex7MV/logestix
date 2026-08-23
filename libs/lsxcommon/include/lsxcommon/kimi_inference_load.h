#pragma once

#include <cstdint>

namespace lsxcommon {
namespace kimi {

// Kimi-K2.7-Code geometry (config.json text_config, verified against the
// vault checkpoint). Compile-time locked per MODEL INGEST INVARIANT; config
// values are cross-checked at Load() and any mismatch is a hard error.
constexpr int kKimiHidden = 7168;
constexpr int kKimiNumLayers = 61;
constexpr int kKimiNumHeads = 64;
constexpr int kKimiKvLoraRank = 512;   // compressed KV latent
constexpr int kKimiQLoraRank = 1536;   // q_lora_rank
constexpr int kKimiQkNope = 128;       // per-head non-rope q/k dim
constexpr int kKimiQkRope = 64;        // shared rope dim
constexpr int kKimiHeadV = 128;        // v_head_dim
constexpr int kKimiMoeIntermediate = 2048;
constexpr int kKimiDenseIntermediate = 18432;
constexpr int kKimiNumRouted = 384;
constexpr int kKimiNumShared = 1;
constexpr int kKimiTopK = 8;
constexpr int kKimiVocab = 163840;
constexpr int kKimiFirstKDenseReplace = 1;
constexpr float kKimiRoutedScalingFactor = 2.827f;
constexpr float kKimiRmsEps = 1e-5f;
constexpr float kKimiRopeTheta = 50000.0f;
constexpr float kKimiYarnFactor = 64.0f;
constexpr int kKimiYarnOriginalMaxPos = 4096;

// Derived MLA projection row counts (rows of the checkpoint weight matrices).
constexpr int kKimiQaRows = kKimiQLoraRank;                               // 1536
constexpr int kKimiQbRows = kKimiNumHeads * (kKimiQkNope + kKimiQkRope);  // 12288
constexpr int kKimiKvaRows = kKimiKvLoraRank + kKimiQkRope;               // 576
constexpr int kKimiKvbRows = kKimiNumHeads * (kKimiQkNope + kKimiHeadV);  // 16384
constexpr int kKimiORows = kKimiHidden;                                   // 7168

} // namespace kimi
} // namespace lsxcommon
