#pragma once

#include <cuda_fp16.h>

#include <string>
#include <vector>

#include "lsxcommon/qwen_dense_convert.h"
#include "lsxcommon/qwen_inference_load.h"
#include "lsxcommon/qwen_moe_convert.h"
#include "lsxcommon/lsx_moe_load.h"
#include "lsxcommon/qwen_moe_expert_cache.h"
#include "lsxcommon/lsx_moe_inference.h"  // moe::MoEExpertStore, lsxcommon::moe::EnsureMoEExpertResident

namespace lsxcommon::qwen_moe {

// Dedup Ф6: the weights structs and the loader are the neutral
// lsxcommon::moe template (lsx_moe_load.h); the public family names stay.
using QwenMoeLayerWeights = lsxcommon::moe::MoeLayerWeights;
using QwenMoePinnedWeights = lsxcommon::moe::MoePinnedWeights<QwenMoeTopology>;

// Loads dense parts (globals, attention, norms, router, shared expert) from
// <model_path>/logestix/ (legacy logestix-qwen35moe/ still read). Routed experts are NOT resident; they go
// through lsxcommon::moe::MoEExpertStore separately. Strict: returns false and fills
// `error` on any missing/malformed input.
inline bool LoadQwenMoeDenseWeights(const std::string &model_path,
                                    QwenMoePinnedWeights &pw,
                                    cudaStream_t stream, std::string &error)
{
    return lsxcommon::moe::LoadMoeDenseWeights(model_path, pw, stream, error);
}

inline void FreeQwenMoePinnedWeights(QwenMoePinnedWeights &pw) { pw.Free(); }


// Ensures expert `eid` of `layer` is resident in the device cache; stages its
// payload H2D from the pinned slab on miss. Returns slot index, -1 on error
// (error text goes to `error` when non-null). When `h2d_stream` is provided,
// miss staging is enqueued on that (copy) stream instead of `stream`; set
// *missed so the caller can fence the upload before consuming the slot.
// *evicted_slot receives the recycled slot index on eviction (-1 otherwise):
// the caller must fence in-flight readers of that slot (event wait) before
// the staged upload overwrites it.

} // namespace lsxcommon::qwen_moe
