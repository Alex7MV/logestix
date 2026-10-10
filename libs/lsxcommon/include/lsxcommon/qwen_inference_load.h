#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "lsxcommon/lsx_inference_nvfp4.h"
#include <cuda_fp16.h>

#include "lsxcommon/lsx_inference_load.h"  // attn::LayerWeights
#include "lsxcommon/qwen_dense_convert.h"

namespace lsxcommon::qwen {

// One NVFP4-quantized dense weight on device: E2M1 nibbles + E4M3 per-16 block
// scales + global float scale + the ORIGINAL weight shape (rows x cols).
// Device NVFP4 weight container moved to the model-neutral NVFP4 layer
// (lsxcommon/lsx_inference_nvfp4.h); the qwen-named alias keeps the
// qwen-family call sites unchanged.
using QwenNvfp4Weight = lsx_nvfp4::LsxNvfp4Weight;

// Per decoder layer weights. `is_linear` selects linear_attn vs self_attn.

struct QwenPinnedWeights {
    QwenTopology top;
    QwenNvfp4Weight embed;        // vocab x hidden
    QwenNvfp4Weight head;         // vocab x hidden
    __half *d_final_norm = nullptr;
    int final_norm_size = 0;
    std::vector<attn::LayerWeights> layers;
    std::uint64_t resident_bytes = 0;   // total device bytes held on load

    void Free();
};

// Loads all Qwen3.5-9B weights from output _root (either <root>/logestix/pinned
// or <root>/pinned) into device memory on `stream`. Returns false on failure.
bool LoadQwenPinnedWeights(const std::string &output_root,
                           QwenPinnedWeights &pw, cudaStream_t stream);

std::vector<uint8_t> ReadFileBytes(const std::string &path);

} // namespace lsxcommon::qwen