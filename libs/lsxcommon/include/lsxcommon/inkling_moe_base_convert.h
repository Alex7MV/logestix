#pragma once

// Inkling (base) converter (Model Ingestion Protocol stages 1-2).
//
// The vault checkpoint is `thinkingmachines/Inkling` (architecture
// `inkling_mm_model`, text tower `inkling`): a 952B-parameter pure-BF16
// release — no hf_quant_config.json, every weight unquantized. The artifact
// this converter writes is uniform NVFP4:
//   * every dense linear matrix (attention q/k/v/o/r, the dense-MLP prefix of
//     layers 0..1, the 2 shared experts of every MoE layer) goes through the
//     neutral `QuantizeSlice`;
//   * the 256 routed experts of layers 2..65 go through `PackExpertNVFP4`
//     (one Arrow record per (layer, expert), streamed by the neutral
//     ExpertStore at inference);
//   * norms, the per-head q/k norm weights, the relative-logit bank, the four
//     short-convolution tap tables, the router weight/bias/global_scale, the
//     dense MLP's global_scale and the embedding tables stay raw (never
//     quantized);
//   * the vision tower (hmlp), the audio tower and the MTP head are stripped
//     and counted, never applied.
//
// ROW LAYOUTS of the three fused gate/up slabs (transformers'
// conversion_mapping.py "inkling_mm_model", verified 2026-11-17):
//   * routed `mlp.experts.w13_weight [E, 2I, H]` is HALVES — rows [0,I) gate,
//     rows [I,2I) up (the mapping's Interleave(dim=1) DE-interleaves the
//     source into the target's interleaved gate_up_proj, whose forward chunks
//     in halves; the source itself is halves);
//   * `shared_experts.shared_w13_weight [S, 2I, H]` is ROW-INTERLEAVED
//     (Interleave(dim=1) + Chunk(dim=1): source rows alternate gate, up);
//   * dense `mlp.w13_dn.weight [2*denseI, H]` is ROW-INTERLEAVED
//     (Interleave(dim=0) + Chunk(dim=0)).
// Confusing the two layouts keeps every byte and every plane size while
// pairing gate rows with the wrong up rows — the defect class that cost
// Inkling-2-Small perplexity 12881 vs 30.4. tests/test_inkling_moe_base_convert
// pins all three pairings by reconstruction.
#include <arrow/api.h>

#include <string>
#include <vector>

namespace lsxcommon::inkling::base {

// Parsed text-tower topology (config.json). The frozen cross-check against
// src/inkling_moe_base_constants.h runs unless synthetic geometry is allowed.
struct ConvertTopology {
    std::string model_type;
    int hidden_size = 0;
    int num_layers = 0;
    int vocab_size = 0;
    int unpadded_vocab_size = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;        // full layers
    int swa_num_key_value_heads = 0;    // sliding layers
    int head_dim = 0;
    int d_rel = 0;
    int rel_extent = 0;
    int sliding_window_size = 0;
    int conv_kernel = 0;
    int dense_mlp_layers = 0;           // config `dense_mlp_idx`
    int dense_intermediate_size = 0;
    int num_experts = 0;
    int num_experts_per_tok = 0;
    int moe_intermediate_size = 0;
    int num_shared_experts = 0;
    double route_scale = 1.0;
    double rms_norm_eps = 1e-6;
    int log_scaling_n_floor = 0;
    double log_scaling_alpha = 0.0;
    double logits_mup_div = 1.0;
    int bos_id = -1;
    int eos_id = -1;
    int pad_id = -1;
    int stop_id_alt = -1;
    std::vector<int> full_layers;
    std::vector<int> sliding_layers;
    int skipped_visual_tensors = 0;
    int skipped_audio_tensors = 0;
    int skipped_mtp_tensors = 0;
    int total_pinned_tensors = 0;
    std::string error_msg;
};

ConvertTopology ParseConfig(const std::string &input_dir);

// Tensor kinds of the released checkpoint. Names are matched after the leading
// `model.llm.` scope; `model.visual.*`, `model.audio.*` and `model.mtp.*` are
// ballast (counted, never applied).
enum class TensorKind {
    Unknown,
    Vision,
    Audio,
    Mtp,
    // globals
    Embed,        // embed.weight        [vocab, hidden] raw BF16
    Unembed,      // unembed.weight      [vocab, hidden] raw BF16
    EmbedNorm,    // embed_norm.weight   [hidden]        raw BF16
    FinalNorm,    // norm.weight         [hidden]        raw BF16
    // per-layer attention
    Wq,           // attn.wq_du.weight   [heads*hd, hidden]    NVFP4
    Wk,           // attn.wk_dv.weight   [kv*hd, hidden]       NVFP4
    Wv,           // attn.wv_dv.weight   [kv*hd, hidden]       NVFP4
    Wo,           // attn.wo_ud.weight   [hidden, heads*hd]    NVFP4
    Wr,           // attn.wr_du.weight   [heads*d_rel, hidden] NVFP4
    QNorm,        // attn.q_norm.weight  [hd]   raw BF16 (per-head RMSNorm)
    KNorm,        // attn.k_norm.weight  [hd]   raw BF16
    RelProj,      // attn.rel_logits_proj.proj [d_rel, extent] raw BF16
    KSconv,       // attn.k_sconv.weight [kv*hd, 1, k] raw BF16
    VSconv,       // attn.v_sconv.weight [kv*hd, 1, k] raw BF16
    AttnSconv,    // attn_sconv.weight   [hidden, 1, k] raw BF16
    MlpSconv,     // mlp_sconv.weight    [hidden, 1, k] raw BF16
    AttnNorm,     // attn_norm.weight    [hidden] raw BF16
    MlpNorm,      // mlp_norm.weight     [hidden] raw BF16
    // dense MLP prefix (layers < dense_mlp_idx)
    DenseW13,     // mlp.w13_dn.weight   [2*denseI, hidden] NVFP4 (interleaved)
    DenseW2,      // mlp.w2_md.weight    [hidden, denseI]     NVFP4
    DenseScale,   // mlp.global_scale    [1] raw BF16
    // routed MoE (layers >= dense_mlp_idx)
    ExpertW13,    // mlp.experts.w13_weight [E, 2I, hidden] NVFP4 (halves)
    ExpertW2,     // mlp.experts.w2_weight  [E, hidden, I]   NVFP4
    SharedW13,    // mlp.shared_experts.shared_w13_weight [S, 2I, hidden]
    SharedW2,     // mlp.shared_experts.shared_w2_weight  [S, hidden, I]
    RouterGate,   // mlp.gate.weight     [E+S, hidden] raw BF16
    RouterBias,   // mlp.gate.bias       [E] raw F32 (selection only)
    RouterScale,  // mlp.gate.global_scale [1] raw F32
};

struct ParsedTensor {
    TensorKind kind = TensorKind::Unknown;
    int layer = -1;
};

ParsedTensor ClassifyTensor(const std::string &name);

// True when the kind's tensor is quantized to NVFP4 in the artifact.
bool TensorIsNvfp4(TensorKind kind);

arrow::Status ConvertInklingBaseModel(const std::string &input_dir,
                                      const std::string &output_root);

}  // namespace lsxcommon::inkling::base