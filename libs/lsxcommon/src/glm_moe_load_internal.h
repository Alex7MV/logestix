#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstring>

#include <lsxcommon/glm_moe_load.h>
#include "fast_load_internal.h"

namespace lsxcommon {
namespace glm {

// GLM now shares the model-neutral fast-load primitives; the glm:: name is
// kept so all existing call sites (loader + tests) compile unchanged.
using PinnedStagingRing = lsxcommon::fastload::PinnedStagingRing;

} // namespace glm
} // namespace lsxcommon
