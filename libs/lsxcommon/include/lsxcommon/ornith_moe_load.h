#pragma once

#include <cuda_fp16.h>

#include <string>
#include <vector>

#include "lsxcommon/qwen_dense_convert.h"
#include "lsxcommon/qwen_inference_load.h"
#include "lsxcommon/ornith_moe_convert.h"
#include "lsxcommon/lsx_moe_load.h"
#include "lsxcommon/expert_device_cache.h"
#include "lsxcommon/lsx_moe_inference.h"  // MoEExpertStore, LoadMoEExpertStore

namespace lsxcommon::ornith {

// Unified expert device cache (shared neutral LRU; local alias keeps the
// ornith call sites stable).
using OrnithMoeExpertDeviceCache = gpu::ExpertDeviceCache;

// Dedup Ф6: the weights structs and the loader are the neutral
// lsxcommon::moe template (lsx_moe_load.h); the public family names stay.
using OrnithMoeLayerWeights = lsxcommon::moe::MoeLayerWeights;
using OrnithMoePinnedWeights = lsxcommon::moe::MoePinnedWeights<OrnithMoeTopology>;

// Loads dense parts (globals, attention, norms, router, shared expert) from
// <model_path>/logestix/ (legacy logestix-qwen35moe/ still read). Routed experts are NOT resident; they go
// through moe::MoEExpertStore separately. Strict: returns false and fills
// `error` on any missing/malformed input.
inline bool LoadOrnithMoeDenseWeights(const std::string &model_path,
                                      OrnithMoePinnedWeights &pw,
                                      cudaStream_t stream, std::string &error)
{
    return lsxcommon::moe::LoadMoeDenseWeights(model_path, pw, stream, error);
}

inline void FreeOrnithMoePinnedWeights(OrnithMoePinnedWeights &pw) { pw.Free(); }



} // namespace lsxcommon::ornith
