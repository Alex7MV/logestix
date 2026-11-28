#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "lsxcommon/phi_dense_constants.h"
#include "lsxcommon/lsx_gemv.h"
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

    // Second NVFP4 plane per projection (the residual w - dequant(base)),
    // stored as "<name>.res". The decode issues one more GEMV with acc=true,
    // which lifts the effective weight precision from 4 to ~8 bits at 2x the
    // bytes: a 4-bit-only body cannot express every near-tie decision and the
    // quantization error is spread over all layers, so this is applied to every
    // projection rather than to a chosen subset. Null on a legacy tree.
    lsx_nvfp4::LsxNvfp4Weight qkv_res;
    lsx_nvfp4::LsxNvfp4Weight o_res;
    lsx_nvfp4::LsxNvfp4Weight mlp_gate_up_res;
    lsx_nvfp4::LsxNvfp4Weight mlp_down_res;

    // The SAME four projections in the high-precision encoding (the `bf16`
    // codec profile): device-resident BF16 planes, consumed by
    // gemv::RunLxGemmVec. Exactly one encoding is present per tree — a loader
    // fills whichever the artifact holds, and the pipeline dispatches on
    // `valid()` / `d_packed`. Why both: an NVFP4 plane cannot be executed on
    // Ampere, and a bf16 tree has no nibbles to upload (defect A3.13).
    gemv::DenseWeight qkv_bf16;
    gemv::DenseWeight o_bf16;
    gemv::DenseWeight mlp_gate_up_bf16;
    gemv::DenseWeight mlp_down_bf16;

    void *d_arena = nullptr;
    size_t arena_bytes = 0;
    // Separate arena for the dense planes: RunUploadJobs() allocates one arena
    // per call and the two encodings are uploaded by different calls.
    void *d_arena_bf16 = nullptr;
    size_t arena_bytes_bf16 = 0;

    void Free();
};

struct PhiPinnedWeights {
    PhiTopology top;
    // Edges (Model Ingestion Protocol step 2, measured 2026-10-24): the
    // embedding and the LM head stay FP16. Quantizing them to NVFP4 cost
    // 0.12 perplexity and two of nine near-tie answers against the same
    // 4-bit body, and no standard 4-bit recipe (NF4/AWQ/GPTQ) quantizes them.
    // A legacy tree that still stores them as NVFP4 loads through
    // `embed`/`head` instead; exactly one of the two pairs is populated.
    __half *d_embed_f16 = nullptr;  // [embed_rows][embed_cols]
    __half *d_head_f16 = nullptr;   // [head_rows][head_cols]
    int embed_rows = 0, embed_cols = 0;
    int head_rows = 0, head_cols = 0;
    lsx_nvfp4::LsxNvfp4Weight embed;  // legacy NVFP4 edge (vocab x hidden)
    lsx_nvfp4::LsxNvfp4Weight head;   // legacy NVFP4 edge (vocab x hidden)
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
