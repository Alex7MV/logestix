#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "lsxcommon/granite_dense_constants.h"
#include "lsxcommon/lsx_inference_nvfp4.h"

namespace lsxcommon {
namespace granite {

// Per-decoder-layer device weights of a dense Granite model.
struct GraniteLayerWeights {
    bool loaded = false;

    // RMSNorm weights (raw BF16/F32 -> FP16), standard w * x_hat form.
    __half *d_input_norm = nullptr;
    __half *d_post_norm = nullptr;
    int norm_size = 0;

    // Attention projections (NVFP4).
    lsx_nvfp4::LsxNvfp4Weight q;  // hidden -> NH*HD
    lsx_nvfp4::LsxNvfp4Weight k;  // hidden -> NKV*HD
    lsx_nvfp4::LsxNvfp4Weight v;  // hidden -> NKV*HD
    lsx_nvfp4::LsxNvfp4Weight o;  // NH*HD -> hidden

    // SwiGLU MLP (NVFP4). `mlp_gate_up` is the fused [2I, K] record with a
    // single global scale; `mlp_gate`/`mlp_up` are non-owning row-slices.
    lsx_nvfp4::LsxNvfp4Weight mlp_gate;
    lsx_nvfp4::LsxNvfp4Weight mlp_up;
    lsx_nvfp4::LsxNvfp4Weight mlp_down;
    lsx_nvfp4::LsxNvfp4Weight mlp_gate_up;

    void *d_arena = nullptr;
    size_t arena_bytes = 0;

    void Free();
};

struct GranitePinnedWeights {
    GraniteTopology top;
    lsx_nvfp4::LsxNvfp4Weight embed;  // vocab x hidden
    lsx_nvfp4::LsxNvfp4Weight head;   // vocab x hidden
    __half *d_final_norm = nullptr;
    int final_norm_size = 0;
    std::vector<GraniteLayerWeights> layers;
    std::uint64_t resident_bytes = 0;

    void Free();
};

// Loads all Granite weights from output_root (either <root>/logestix/pinned or
// <root>/pinned) into device memory on `stream`. Returns false on failure.
bool LoadGranitePinnedWeights(const std::string &output_root,
                              GranitePinnedWeights &pw, cudaStream_t stream);

} // namespace granite
} // namespace lsxcommon
