#pragma once

// GigaChat4-37B-A4B ingestion converter (Model Ingestion Protocol step 1).
// Produces the `logestix-gigachat4-nvfp4` layout under <output>/logestix/:
// model.index.json, pinned/global.arrow, pinned/layer_NNN.arrow,
// pinned/layer_NNN_nvfp4.arrow, experts/layer_N/exp_E.arrow and
// mtp/layer_NNN.arrow.
//
// The checkpoint is mixed precision: attention/expert/GDN-input matrices are
// FP8 E4M3 with per-128x128 F32 weight_scale_inv tiles (dequantized to BF16 and
// re-quantized to the neutral NVFP4 contour), while the attention gate, the
// shared experts, the dense MLP and the indexer's wk/weights_proj are BF16 and
// are quantized to NVFP4 directly. Norms, routers, biases, conv1d, A_log,
// dt_bias, in_proj_ba and the indexer k-norm stay at source precision.

#include <arrow/status.h>

#include <cstdint>
#include <string>
#include <vector>

namespace lsxcommon::gigachat4 {

enum class Gc4TensorKind {
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
    IndexWqB,
    IndexWk,
    IndexWeightsProj,
    IndexKNorm,
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

struct ParsedGc4Tensor {
    Gc4TensorKind kind = Gc4TensorKind::Unknown;
    int layer = -1;
    int expert = -1;  // only for ExpertGate / ExpertUp / ExpertDown
};

// Classifies an HF GigaChat4 tensor name. `weight_scale_inv` names classify to
// the same kind as their `weight` counterpart; the converter distinguishes the
// payload by the name suffix and by the presence of the scale tensor.
// `num_layers` is the decoder depth: every layer index at or above it belongs to
// the MTP stack (stored, never applied). The one-argument overload uses the
// frozen passport depth, which is what the real checkpoint has.
ParsedGc4Tensor classify_gigachat4_tensor(const std::string &name,
                                         int num_layers);
ParsedGc4Tensor classify_gigachat4_tensor(const std::string &name);

// True when the kind is a linear matrix that becomes NVFP4 (from FP8 or from
// BF16, decided per tensor by the presence of a `_scale_inv` sibling).
bool gigachat4_tensor_is_nvfp4(Gc4TensorKind kind);

// True when the kind belongs to a routed expert (stored per expert).
bool gigachat4_tensor_is_expert(Gc4TensorKind kind);

struct Gc4Topology {
    int hidden = 0;
    int layers = 0;
    int vocab = 0;
    int heads = 0;
    int kv_heads = 0;
    int num_experts = 0;
    int top_k = 0;
    int moe_intermediate = 0;
    int dense_intermediate = 0;
    int first_k_dense = 0;
    int q_lora = 0;
    int kv_lora = 0;
    int qk_nope = 0;
    int qk_rope = 0;
    int v_head = 0;
    int index_heads = 0;
    int index_head_dim = 0;
    int index_topk = 0;
    double rms_eps = 0.0;
    double routed_scale = 0.0;
    double swiglu_limit = 0.0;
    double logit_softcap = 0.0;
    int bos_id = -1;
    int eos_id = -1;
    int pad_id = -1;
    std::vector<int> full_attention_layers;
    std::string error_msg;  // non-empty when config is invalid
};

Gc4Topology ParseGigaChat4Config(const std::string &config_json_text);

arrow::Status ConvertGigaChat4Model(const std::string &input_dir,
                                    const std::string &output_root);

} // namespace lsxcommon::gigachat4
