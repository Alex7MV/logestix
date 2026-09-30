#pragma once

// MiMo-V2.6-Flash-RL converter (Model Ingestion Protocol steps 1-2). Produces
// the `logestix-mimo-nvfp4` layout under <output_root>/logestix.
//
// Geometry sources: config.json is authoritative; model.layers.N.self_attn is
// a fused qkv_proj (FP8 E4M3 + F32 weight_scale_inv block-128); routed experts
// are already MXFP4 (U8 E2M1 nibbles + U8 F8_E8M0 per-32 `.weight_scale`).
// Multimodal payload (visual/audio/speech) is stripped; MTP is stored+counted.
#include <arrow/api.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace lsxcommon {
namespace mimo {

enum class MiMoTensorKind {
    Unknown,
    Embed,
    LmHead,
    FinalNorm,
    InputNorm,
    PostNorm,
    QkvProj,       // self_attn.qkv_proj (FP8, fused)
    OProj,         // self_attn.o_proj (BF16)
    SinkBias,      // self_attn.attention_sink_bias (BF16, SWA)
    RouterWeight,  // mlp.gate.weight (BF16)
    RouterBias,    // mlp.gate.e_score_correction_bias (F32)
    DenseGate,     // mlp.gate_proj (FP8, layer 0)
    DenseUp,       // mlp.up_proj   (FP8)
    DenseDown,     // mlp.down_proj (FP8)
    ExpertGate,    // mlp.experts.E.gate_proj (MXFP4)
    ExpertUp,      // mlp.experts.E.up_proj
    ExpertDown,    // mlp.experts.E.down_proj
    Mtp,           // model.mtp.*  (stored + counted)
    Multimodal,    // visual/audio/speech (stripped)
};

struct ParsedMimoTensor {
    MiMoTensorKind kind = MiMoTensorKind::Unknown;
    int layer = -1;
    int expert = -1;
};

struct MiMoTopology {
    int hidden = 4096;
    int layers = 48;
    int vocab = 152576;
    int dense_intermediate = 16384;
    int moe_intermediate = 2048;
    int num_experts = 256;
    int top_k = 8;
    int heads = 64;
    int kv_heads_ga = 4;
    int kv_heads_swa = 8;
    int head_dim = 192;
    int v_head_dim = 128;
    double rms_eps = 1e-6;
    double rope_theta_ga = 1e7;
    double rope_theta_swa = 1e4;
    double value_scale = 0.707;
    int sliding_window = 128;
    std::vector<int> full_attn_layers;  // hybrid_layer_pattern == 0
    std::vector<int> moe_layers;        // moe_layer_freq != 0
    std::string error_msg;
};

// Classifies an HF MiMo-V2.6 tensor name. Unknown kinds are errors in the
// converter; multimodal names classify as Multimodal (stripped).
ParsedMimoTensor classify_mimo_tensor(const std::string &name);

// True when the tensor is quantized to NVFP4 (attn qkv, dense MLP, experts).
bool mimo_tensor_is_nvfp4(MiMoTensorKind kind);
bool mimo_tensor_is_expert(MiMoTensorKind kind);

MiMoTopology ParseMiMoConfig(const std::string &config_json_text);

// FP8 E4M3 + F32 weight_scale_inv [ceil(R/128), ceil(C/128)] → BF16. Indexes
// the scale by r/128; a `scale_rows*128 > R` tail is never read (the GA
// qkv_proj quirk).
arrow::Status DequantFp8Block128ToBf16Gpu(const std::vector<uint8_t> &fp8,
                                          const std::vector<int64_t> &shape,
                                          const std::vector<uint8_t> &scale,
                                          const std::vector<int64_t> &scale_shape,
                                          std::vector<uint8_t> &out_bf16);

// MXFP4 (packed E2M1 nibbles + F8_E8M0 per-32) → BF16 (logical columns =
// packed columns * 2). Fallback path when the exact NVFP4 repack cannot fold a
// single power-of-two global scale.
arrow::Status DequantMxfp4ToBf16Gpu(const std::vector<uint8_t> &packed,
                                    const std::vector<int64_t> &shape,
                                    const std::vector<uint8_t> &e8m0,
                                    const std::vector<int64_t> &scale_shape,
                                    std::vector<uint8_t> &out_bf16);

arrow::Status ConvertMiMoModel(const std::string &input_dir,
                               const std::string &output_root);

} // namespace mimo
} // namespace lsxcommon
