#pragma once

// GigaChat3.5-432B-A28B-Reasoning ingestion converter (Model Ingestion
// Protocol step 1). Produces the `logestix-gigachat-nvfp4` layout under
// <output>/logestix/: model.index.json, pinned/global.arrow,
// pinned/layer_NNN.arrow, pinned/shared_experts.arrow,
// experts/layer_N/exp_E.arrow and mtp/layer_NNN.arrow.
//
// Source weights are FP8 E4M3 with per-128x128 F32 weight_scale_inv tiles; the
// converter dequantizes them to BF16 and re-quantizes to the neutral NVFP4
// contour (E2M1 nibbles + per-16 E4M3 scales + per-tensor global_scale).

#include <arrow/status.h>

#include <cstdint>
#include <string>
#include <vector>

namespace lsxcommon::gigachat {

enum class GigaChatTensorKind {
    Unknown,
    Embed,
    LmHead,
    FinalNormWeight,
    FinalNormGateUp,
    FinalNormGateDown,
    LayerNormWeight,
    LayerNormGateUp,
    LayerNormGateDown,
    QaProj,
    QbProj,
    KvaProj,
    KvbProj,
    OProj,
    AttnGateProj,
    MlpGate,
    MlpUp,
    MlpDown,
    GdnInProjQkvz,
    GdnOutProj,
    GdnConv1d,
    GdnALog,
    GdnDtBias,
    GdnInProjBa,
    GdnNormWeight,
    RouterWeight,
    RouterBias,
    SharedGate,
    SharedUp,
    SharedDown,
    ExpertGate,
    ExpertUp,
    ExpertDown,
    Mtp,
};

struct ParsedGigaChatTensor {
    GigaChatTensorKind kind = GigaChatTensorKind::Unknown;
    int layer = -1;
    int expert = -1;  // only for ExpertGate / ExpertUp / ExpertDown
};

// Classifies an HF GigaChat3.5 tensor name. `weight_scale_inv` names classify
// to the same kind as their `weight` counterpart; the converter distinguishes
// the payload by the name suffix.
ParsedGigaChatTensor classify_gigachat_tensor(const std::string &name);

// True when the kind is an FP8 linear matrix converted to NVFP4; false for the
// high-precision pass-through set (norms, biases, embeddings, gates, A_log,
// dt_bias, conv1d, in_proj_ba).
bool gigachat_tensor_is_nvfp4(GigaChatTensorKind kind);

// True when the kind belongs to a routed expert (stored per expert).
bool gigachat_tensor_is_expert(GigaChatTensorKind kind);

struct GigaChatTopology {
    int hidden = 0;
    int layers = 0;
    int vocab = 0;
    int heads = 0;
    int kv_heads = 0;
    int num_experts = 0;
    int top_k = 0;
    int moe_intermediate = 0;
    int first_k_dense = 0;
    double rms_eps = 0.0;
    double routed_scale = 0.0;
    double swiglu_limit = 0.0;
    int bos_id = -1;
    int eos_id = -1;
    int pad_id = -1;
    std::vector<int> full_attention_layers;
    std::string error_msg;  // non-empty when config is invalid
};

GigaChatTopology ParseGigaChatConfig(const std::string &config_json_text);

// GPU-dequantizes an F8_E4M3 tensor with per-128x128 F32 scale_inv tiles into
// a BF16 byte buffer (row-major [rows, cols]). The last tile may be partial
// (`scale_shape = [ceil(rows/128), ceil(cols/128)]`). Output sizing:
// bf16.size() == rows*cols*2.
arrow::Status DequantFp8Block128ToBf16Gpu(const std::vector<uint8_t> &fp8,
                                          const std::vector<int64_t> &shape,
                                          const std::vector<uint8_t> &scale_f32,
                                          const std::vector<int64_t> &scale_shape,
                                          std::vector<uint8_t> &bf16);

arrow::Status ConvertGigaChatModel(const std::string &input_path,
                                   const std::string &output_path);

} // namespace lsxcommon::gigachat
