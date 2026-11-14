#pragma once

// Inkling-2-Small converter (Model Ingestion Protocol stages 1-2).
//
// The vault checkpoint is `thinkingmachines/Inkling-2-Small`: a modelopt
// release whose 594B-parameter text tower is already quantized in TWO different
// codecs (routed experts of layers 0..39 NVFP4, layers 40..47 MXFP8, everything
// else BF16/F32), plus an unused vision tower and a separate MTP file.
//
// The artifact this converter writes is uniform NVFP4:
//   * layers 0..39 experts are passed through BYTE-VERBATIM (the source block
//     scales are full E4M3FN and the repo's scale decode is mantissa-aware, so
//     no requantization is needed): per-expert global scale = amax_e / 2688;
//   * layers 40..47 experts are dequantized from MXFP8 (ue8m0/32) and
//     requantized through the neutral `quantize_tensor_gpu`;
//   * attention/MLP dense weights are BF16 -> NVFP4 through the same neutral
//     quantizer; norms, biases, the router, the relative-logit bank, the L2-clip
//     scales, the short-convolution taps and the AttnRes queries stay
//     BF16/F32 (never quantized);
//   * the vision tower and the MTP head are stripped and counted.
#include <arrow/api.h>

#include <string>
#include <vector>

#include "lsxcommon/vram_estimate.h"

namespace lsxcommon::inkling {

// Tensor kinds of the released checkpoint. Names are matched after the leading
// `model.llm.` scope; `model.visual.*` is ballast and `mtp.safetensors` is a
// separate file that the classifier never sees.
enum class InklingTensorKind {
    Unknown,
    // globals
    Embed,            // embed.weight            [vocab, hidden] NVFP4
    Unembed,          // unembed.weight          [vocab, hidden] NVFP4
    EmbedNorm,        // embed_norm.weight       [hidden]        raw BF16
    FinalNorm,        // norm.weight             [hidden]        raw BF16
    OutputQuery,      // attn_res.output_query   [hidden]        raw F32
    AttnQuery,        // attn_res.attn_queries.L [hidden] | [3,hidden] raw F32
    MlpQuery,         // attn_res.mlp_queries.L  [hidden]        raw F32
    // per-layer attention
    AttnNorm,         // attn_norm.weight        [hidden]  (sliding layers only)
    QInputNorm,       // q_input_norm.weight     [hidden]  (full layers only)
    KInputNorm,       // k_input_norm.weight     [hidden]  (full layers only)
    VInputNorm,       // v_input_norm.weight     [hidden]  (full layers only)
    MlpNorm,          // mlp_norm.weight         [hidden]
    Wq,               // attn.wq_du.weight       [heads*hd, hidden]   NVFP4
    Wk,               // attn.wk_dv.weight       [kv*hd, hidden]      NVFP4
    Wv,               // attn.wv_dv.weight       [kv*hd, hidden]      NVFP4
    Wo,               // attn.wo_ud.weight       [hidden, heads*hd]   NVFP4
    Wr,               // attn.wr_du.weight       [heads*d_rel, hidden] NVFP4
    AttnGate,         // attn.gate_attn.weight   [heads*hd, hidden]   NVFP4
    AttnSink,         // attn.attn_sink_proj.weight [heads, hidden]   raw
    RelProj,          // attn.rel_logits_proj.proj  [d_rel, rel_extent] raw
    QClipScale,       // attn.q_l2_clip.scale    [heads, hd]      raw
    RelClipScale,     // attn.rel_l2_clip.scale  [heads, d_rel]   raw
    KSConv,           // attn.k_sconv.weight     [kv*hd, 1, k]    raw (sliding)
    VSConv,           // attn.v_sconv.weight     [kv*hd, 1, k]    raw (sliding)
    KSConvStatic,     // attn.k_sconv.static_w   [k, kv*hd]       raw (full)
    VSConvStatic,     // attn.v_sconv.static_w   [k, kv*hd]       raw (full)
    KSConvProj,       // attn.k_sconv.weight_proj [kv*k, hidden]   NVFP4 (full)
    VSConvProj,       // attn.v_sconv.weight_proj [kv*k, hidden]   NVFP4 (full)
    AttnSConv,        // attn_sconv.weight       [hidden, 1, k]   raw
    MlpSConv,         // mlp_sconv.weight        [hidden, 1, k]   raw
    // per-layer MoE
    RouterGate,       // mlp.gate.weight         [experts, hidden]  raw BF16
    RouterBias,       // mlp.gate.bias           [experts]          raw F32
    ExpertW13,        // mlp.experts.w13_weight  [E, 2I, H/2] | [E, 2I, H]
    ExpertW13Scale,   // mlp.experts.w13_weight_scale
    ExpertW13Amax,    // mlp.experts.w13_weight_amax        [E]
    ExpertW13Denom,   // mlp.experts.w13_weight_outer_scale_denominator [1]
    ExpertW2,         // mlp.experts.w2_weight   [E, H, I/2] | [E, H, I]
    ExpertW2Scale,    // mlp.experts.w2_weight_scale
    ExpertW2Amax,     // mlp.experts.w2_weight_amax         [E]
    ExpertW2Denom,    // mlp.experts.w2_weight_outer_scale_denominator [1]
    // ballast (stripped + counted)
    Vision,           // model.visual.*
};

struct ParsedInklingTensor {
    InklingTensorKind kind = InklingTensorKind::Unknown;
    int layer = -1;
};

// config.json (+ hf_quant_config.json for the codec split) parsed into the
// topology the converter writes into model.index.json. The loader re-reads the
// index and cross-checks it against the frozen passport.
struct InklingConvertTopology {
    std::string model_type;
    int hidden_size = 0;
    int num_layers = 0;
    int vocab_size = 0;
    int unpadded_vocab_size = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int d_rel = 0;
    int rel_extent = 0;
    int sliding_window_size = 0;
    int conv_kernel = 0;
    int attn_res_block = 0;
    int num_experts = 0;
    int num_experts_per_tok = 0;
    int moe_intermediate_size = 0;
    int num_shared_experts = 0;
    bool moe_renormalize = true;
    double route_scale = 0.0;
    double rms_norm_eps = 0.0;
    double attn_scale_denom = 0.0;
    double qk_l2_clip = 0.0;
    double rel_l2_clip = 0.0;
    int log_scaling_n_floor = 0;
    double log_scaling_alpha = 0.0;
    double mlp_gate_softcap = 0.0;
    double mlp_up_softcap = 0.0;
    double final_logit_softcap = 0.0;
    int bos_id = -1;
    int eos_id = -1;
    int stop_id_alt = -1;
    int pad_id = -1;
    // 0-based; full-attention layers derived from `local_layer_ids`.
    std::vector<int> full_layers;
    std::vector<int> sliding_layers;
    // 0-based; true where the source expert plane is NVFP4 (else MXFP8).
    std::vector<bool> expert_nvfp4;
    // counters
    int skipped_visual_tensors = 0;
    int total_pinned_tensors = 0;
    std::string error_msg;
};

InklingConvertTopology ParseInklingConfig(const std::string &input_dir);
ParsedInklingTensor ClassifyInklingTensor(const std::string &name);

// Dense weights uploaded as NVFP4 for the whole session.
bool InklingTensorIsPinnedNvfp4(InklingTensorKind kind);

// Converts the checkpoint into the logestix-inkling-2-small-nvfp4 layout under
// <output_root>/logestix.
arrow::Status ConvertInklingModel(const std::string &input_dir,
                                  const std::string &output_root);

}  // namespace lsxcommon::inkling
