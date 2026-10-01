#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "lsxcommon/phi_dense_constants.h"
#include "lsxcommon/lsx_inference_nvfp4.h"

namespace lsxcommon {
namespace phi {

// Per-decoder-layer device weights of a dense phi model.
struct PhiLayerWeights {
    bool loaded = false;

    // RMSNorm weights (raw BF16/F32 -> FP16), standard w * x_hat form.
    __half *d_input_norm = nullptr;
    __half *d_post_norm = nullptr;
    int norm_size = 0;

    // Attention projections (NVFP4). phi stores q/k/v pre-fused in the source
    // as `qkv_proj` ([NH*HD + 2*NKV*HD, K], one global scale) so decode
    // streams them in a single GEMV; q/k/v below are non-owning row-slices.
    lsx_nvfp4::LsxNvfp4Weight qkv;
    lsx_nvfp4::LsxNvfp4Weight q;  // hidden -> NH*HD
    lsx_nvfp4::LsxNvfp4Weight k;  // hidden -> NKV*HD
    lsx_nvfp4::LsxNvfp4Weight v;  // hidden -> NKV*HD
    lsx_nvfp4::LsxNvfp4Weight o;  // NH*HD -> hidden

    // SwiGLU MLP (NVFP4). phi stores gate/up pre-fused in the source as
    // `gate_up_proj` ([2I, K], single global scale); `mlp_gate`/`mlp_up` are
    // non-owning row-slices.
    lsx_nvfp4::LsxNvfp4Weight mlp_gate;
    lsx_nvfp4::LsxNvfp4Weight mlp_up;
    lsx_nvfp4::LsxNvfp4Weight mlp_down;
    lsx_nvfp4::LsxNvfp4Weight mlp_gate_up;

    void *d_arena = nullptr;
    size_t arena_bytes = 0;

    void Free();
};

struct PhiPinnedWeights {
    PhiTopology top;
    lsx_nvfp4::LsxNvfp4Weight embed;  // vocab x hidden
    lsx_nvfp4::LsxNvfp4Weight head;   // vocab x hidden
    __half *d_final_norm = nullptr;
    int final_norm_size = 0;
    std::vector<PhiLayerWeights> layers;
    std::uint64_t resident_bytes = 0;

    void Free();
};

// Loads all phi weights from output_root (either <root>/logestix/pinned or
// <root>/pinned) into device memory on `stream`. Returns false on failure.
bool LoadPhiPinnedWeights(const std::string &output_root,
                          PhiPinnedWeights &pw, cudaStream_t stream);

} // namespace phi
} // namespace lsxcommon
