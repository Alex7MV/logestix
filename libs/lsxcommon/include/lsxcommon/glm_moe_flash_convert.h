#pragma once

#include <arrow/api.h>
#include <cstdint>
#include <filesystem>
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

// Builds the step-7 network passport token tables from a converted flash
// tokenizer: first = glm_tokens from @p pinned_dir/tokenizer_vocab.arrow,
// second = deepseek_tokens from the DeepSeek-V4-Flash vault vocabulary
// (<models>/DeepSeek-V4-Flash/logestix/pinned). Unknown-token anchors are
// each side's byte-0 token (byte-level BPE has no <unk>; a NUL byte is the
// loss-free sink). Returns (glm_tokens, ds_tokens, glm_unk, ds_unk) so the
// caller can feed lsxnetwork::BuildPassportOptions without lsxnetwork
// headers in this public header (OBJECT CUDA targets include it). When the
// DeepSeek vocabulary is absent the deepseek side stays empty (header
// ds_vocab == 0) and the caller warns.
struct FlashPassportTokens {
    std::vector<std::string> glm_tokens;
    std::vector<std::string> deepseek_tokens;
    std::uint32_t glm_unk_id = 0;
    std::uint32_t deepseek_unk_id = 0;
};
FlashPassportTokens BuildFlashPassportTokens(
    const std::filesystem::path &pinned_dir);

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

// Parsed text_config of a GLM-5.3-Flash checkpoint.
struct FlashTopology {
    int hidden = 0, layers = 0, vocab = 0, heads = 0;
    int qk_head_dim = 0, q_lora = 0, kv_lora = 0;
    int num_experts = 0, top_k = 0;
    int moe_intermediate = 0, dense_intermediate = 0;
    int hc_count = 0, gate_lowrank = 0, conv_kernel = 0;
    int indexer_heads = 0, indexer_head_dim = 0, indexer_topk = 0;
    double routed_scaling = 0.0;
    std::vector<int> eos_ids;
    int pad_id = -1;
    std::string error_msg;  // non-empty when config is invalid
};

// Parses text_config from <input_dir>/config.json via simdjson. Falls back
// to flat root keys when text_config is absent. On any structural problem
// the returned struct carries a non-empty error_msg.
FlashTopology ParseFlashConfig(const std::string &input_dir);


} // namespace lsxcommon::glm_moe_flash
