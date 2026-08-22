#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "lsxcommon/resource_governor.h"
#include "lsxcommon/vram_estimate.h"

namespace lsxcommon {
namespace lifecycle {

// Model execution archetype, resolved from the manifest topology
// (non-empty moe_layers[] -> MoE, else dense).
enum class Archetype { General, Dense, MoE };

// Runtime-tunable parameter ranges (spec: the tuner may only move inside
// these bounds; the plan fixes the start value).
struct TuningBounds {
    std::size_t expert_cache_slots_min = 4;
    std::size_t expert_cache_slots_max = 64;
};

// Deterministic residency plan for one model on one machine (spec 3).
struct LifecyclePlan {
    // Permanent VRAM residency
    bool pin_globals_in_vram = true;
    bool pin_dense_layers = true;
    std::uint64_t permanent_weight_bytes = 0;

    // Expert subsystem (MoE)
    std::size_t expert_cache_slots = 0;  // start value within tuner_bounds
    std::uint64_t expert_cache_min_bytes = 0;
    std::uint64_t expert_cache_max_bytes = 0;
    std::size_t expert_prefetch_batch = 1;

    // Compute pipeline
    int copy_streams = 2;
    std::size_t staging_ring_bytes = 8ull << 20;
    std::size_t host_upload_threads = 1;

    TuningBounds tuner_bounds{};
};

// Strategy inputs: everything except the hardware snapshot itself.
struct LifecycleInput {
    VramProfile vram;
    Archetype archetype = Archetype::General;
    std::string model_type;
    std::uint64_t expert_mean_bytes = 0;  // mean routed-expert payload bytes
};

class ILifecycleStrategy {
public:
    virtual ~ILifecycleStrategy() = default;
    // Deterministic: identical inputs -> identical plan. Never allocates GPU
    // memory and never throws.
    virtual LifecyclePlan BuildPlan(const LifecycleInput &in,
                                    const HardwareProfile &hw,
                                    std::uint64_t vram_keep_out_bytes) const = 0;
};

class GeneralStrategy : public ILifecycleStrategy {
public:
    LifecyclePlan BuildPlan(const LifecycleInput &in, const HardwareProfile &hw,
                            std::uint64_t vram_keep_out_bytes) const override;
};

// Extends the general plan with the expert-cache sizing + prefetch policy.
class MoEStrategy : public GeneralStrategy {
public:
    LifecyclePlan BuildPlan(const LifecycleInput &in, const HardwareProfile &hw,
                            std::uint64_t vram_keep_out_bytes) const override;
};

// Per-model anchor for GLM-5.2 (registered in ResolveStrategy). Inherits the
// MoE plan; the MTP epilogue stays permanently resident (spec amendment).
class Glm52Strategy final : public MoEStrategy {};

// Registry resolution: per-model override ("glm-5.2") -> archetype -> General.
const ILifecycleStrategy &ResolveStrategy(Archetype archetype,
                                          const std::string &model_type);

// Builds (once) and returns the plan for a model type. The plan is cached per
// model_type: repeated calls return the same object.
const LifecyclePlan &GetOrBuildPlan(const LifecycleInput &in);

void ResetPlanCache();  // tests only

} // namespace lifecycle
} // namespace lsxcommon
