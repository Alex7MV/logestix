#pragma once

#include <arrow/api.h>
#include <cstdint>
#include <string>
#include <vector>

namespace lsxcommon::glm_moe_flash {

// Converts a GLM-5.3-Flash (glm5_next) checkpoint into the unified
// <output_root>/logestix layout. Format tag: logestix-glm5next.
// Layout:
//   model.index.json                    — extended index (format tag +
//                                       kda/dsa/moe/dense layer lists)
//   pinned/global.arrow                 — embed/lm_head/final_norm (raw BF16)
//   pinned/global_nvfp4.arrow           — (none today; reserved)
//   pinned/layer_NNN.arrow              — per-layer raw tensors (norms,
//                                       KDA BF16 weights, hc, router, indexer)
//   pinned/layer_NNN_nvfp4.arrow        — per-layer NVFP4 tensors (DSA
//                                       projections, shared experts, dense MLP)
//   experts/layer_NNN/exp_EEE.arrow     — NVFP4 routed experts (ExpertSchema)
// Per-layer expert shards keep peak host RAM bounded (306 GB source): each
// MoE layer's experts are quantized and flushed in their own pass.
arrow::Status ConvertGlm5NextModel(const std::string &input_dir,
                                   const std::string &output_root);

// GPU-dequantizes an F8_E4M3 tensor with per-128x128 F32 scale_inv tiles
// into a BF16 byte buffer (row-major, [rows, cols]). Feeds
// gpu::quantize_tensor_gpu("BF16") for the NVFP4 requant path (the
// existing e8m0-tile hook only accepts byte scales; the GLM-5.3-Flash
// checkpoint stores F32 weight_scale_inv). Output sizing:
// bf16.size() == rows*cols*2.
arrow::Status DequantFp8TilesToBf16Gpu(const std::vector<uint8_t> &fp8,
                                       const std::vector<int64_t> &shape,
                                       const std::vector<uint8_t> &scale_f32,
                                       const std::vector<int64_t> &scale_shape,
                                       std::vector<uint8_t> &bf16);

} // namespace lsxcommon::glm_moe_flash
