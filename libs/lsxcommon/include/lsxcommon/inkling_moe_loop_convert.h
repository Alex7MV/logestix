#pragma once

// Inkling-2-Mini converter (Model Ingestion Protocol stages 1-2).
//
// The vault checkpoint is `thinkingmachines/Inkling-2-Mini`: a modelopt
// release of the LOOPED inkling2_loop architecture (24 physical layers, the
// middle 12 run 3 passes). Its routed experts ship in TWO codecs (layers
// 0..19 NVFP4, layers 20..23 MXFP8); everything else is BF16/F32, plus an
// unused vision tower and a separate MTP file the index does not list.
//
// The artifact keeps the release's own precision (nothing is requantized):
//   * NVFP4 experts pass through BYTE-VERBATIM, de-interleaved (the fused w13
//     slab alternates gate/up rows); per-expert global scale = amax_e / 2688;
//   * MXFP8 experts keep their E4M3 bytes; each ue8m0 per-32 scale byte is
//     duplicated to the neutral schema's per-16 count;
//   * dense weights stay BF16 (widened to FP16 on device by the loader);
//     norms, the router, the rel bank, the clip scales, the conv taps, the
//     AttnRes queries, the cross-pass message queries and the loop
//     controllers stay raw (never quantized);
//   * the two cross-pass query tensors are merged into one
//     [loop_layers, 4, hidden] FP32 `message_queries` record, order
//     [q, k, v, mlp] per loop layer (the release's
//     Inkling2LoopMergeMessageQueries, verified on the real tensors);
//   * the vision tower is stripped and counted; mtp.safetensors is never
//     read (the converter follows the index / the shard list, and the MTP
//     file is not part of either when an index exists).
#include <arrow/api.h>

#include <string>
#include <vector>

namespace lsxcommon::inkling {

// Tensor kinds of the released inkling2_loop checkpoint. Names are matched
// after the leading `model.llm.` scope; `model.visual.*` is ballast.
enum class InklingLoopTensorKind {
    Unknown,
    // globals
    Embed,            // embed.weight            [vocab, hidden] BF16
    Unembed,          // unembed.weight          [vocab, hidden] BF16
    EmbedNorm,        // embed_norm.weight       [hidden]        BF16
    FinalNorm,        // norm.weight             [hidden]        BF16
    OutputQuery,      // attn_res.output_query   [hidden]        F32
    AttnQuery,        // attn_res.attn_queries.L [hidden] | [3,hidden] F32
    MlpQuery,         // attn_res.mlp_queries.L  [hidden]        F32
    CrossPassQueries,    // middle_cycle_cross_pass_queries    [2*loop, hidden]
    CrossPassKvQueries,  // middle_cycle_cross_pass_kv_queries [2*loop, hidden]
    // per-layer attention
    AttnNorm,         // attn_norm.weight        [hidden]  (sliding layers only)
    QInputNorm,       // q_input_norm.weight     [hidden]  (full layers only)
    KInputNorm,       // k_input_norm.weight     [hidden]  (full layers only)
    VInputNorm,       // v_input_norm.weight     [hidden]  (full layers only)
    MlpNorm,          // mlp_norm.weight         [hidden]
    Wq,               // attn.wq_du.weight       [heads*hd, hidden]
    Wk,               // attn.wk_dv.weight       [kv*hd, hidden]
    Wv,               // attn.wv_dv.weight       [kv*hd, hidden]
    Wo,               // attn.wo_ud.weight       [hidden, heads*hd]
    Wr,               // attn.wr_du.weight       [heads*d_rel, hidden]
    AttnGate,         // attn.gate_attn.weight   [heads*hd, hidden]
    AttnSink,         // attn.attn_sink_proj.weight [heads, hidden]
    RelProj,          // attn.rel_logits_proj.proj  [d_rel, rel_extent]
    QClipScale,       // attn.q_l2_clip.scale    [heads, hd]
    RelClipScale,     // attn.rel_l2_clip.scale  [heads, d_rel]
    KSConv,           // attn.k_sconv.weight     [kv*hd, 1, k]  (sliding)
    VSConv,           // attn.v_sconv.weight     [kv*hd, 1, k]  (sliding)
    KSConvStatic,     // attn.k_sconv.static_w   [k, kv*hd]     (full)
    VSConvStatic,     // attn.v_sconv.static_w   [k, kv*hd]     (full)
    KSConvProj,       // attn.k_sconv.weight_proj [kv*k, hidden] (full)
    VSConvProj,       // attn.v_sconv.weight_proj [kv*k, hidden] (full)
    AttnSConv,        // attn_sconv.weight       [hidden, 1, k]
    MlpSConv,         // mlp_sconv.weight        [hidden, 1, k]
    // per-layer MoE
    RouterGate,       // mlp.gate.weight         [experts, hidden]  BF16
    RouterBias,       // mlp.gate.bias           [experts]          F32
    ExpertW13,        // mlp.experts.w13_weight  [E, 2I, H/2] | [E, 2I, H]
    ExpertW13Scale,
    ExpertW13Amax,
    ExpertW13Denom,
    ExpertW2,         // mlp.experts.w2_weight   [E, H, I/2] | [E, H, I]
    ExpertW2Scale,
    ExpertW2Amax,
    ExpertW2Denom,
    // loop controllers (loop layers only): four MessageGates per layer
    ControllerDown,   // controllers.<m>.down_weight   [rank, 2*hidden] BF16
    ControllerForget, // controllers.<m>.forget_weight [hidden, rank]   BF16
    ControllerGain,   // controllers.<m>.gain_weight   [hidden, rank]   BF16
    // ballast (stripped + counted)
    Vision,           // model.visual.*
};

struct ParsedInklingLoopTensor {
    InklingLoopTensorKind kind = InklingLoopTensorKind::Unknown;
    int layer = -1;
    // Controller message index: 0=q(attn), 1=k, 2=v, 3=mlp; -1 otherwise.
    int message = -1;
};

// The frozen numbers a conversion validates the source against. Each model row
// supplies its OWN passport's values (an identity header per model is the
// independence rule), so a sibling with identical geometry repeats them rather
// than sharing a header. Only the fields this architecture cannot vary are
// here: everything else (softcaps, scales, eps) is read from the checkpoint and
// recorded, not gated.
struct InklingLoopFrozenGeometry {
    int hidden = 0;
    int layers = 0;
    int vocab = 0;
    int base_vocab = 0;
    int heads = 0;
    int kv_heads = 0;
    int head_dim = 0;
    int d_rel = 0;
    int rel_extent = 0;
    int sliding_window = 0;
    int conv_kernel = 0;
    int attn_res_block = 0;
    int experts = 0;
    int top_k = 0;
    int moe_intermediate = 0;
    int shared_experts = 0;
    int loop_start = 0;
    int loop_end = 0;
    // The checkpoint's own declared pass count (the model's default).
    int loop_passes = 0;
    int loop_gate_rank = 0;
    int log_scaling_floor = 0;
    int nvfp4_expert_layers = 0;
    std::vector<int> full_layers;
};

// One model row's conversion contract: the card name (`model` in the index is
// what `gpu::detect_model_type` resolves into a factory key), the format tag,
// the geometry to gate on, and — for a varloop tree — the trained pass ceiling
// that the checkpoint's config does not carry.
struct InklingLoopConvertSpec {
    std::string model_name;
    std::string format_tag;
    InklingLoopFrozenGeometry frozen;
    // `is_varloop` in the checkpoint must equal this. A mismatch is an error in
    // BOTH directions: the fixed-pass weights were trained at exactly
    // `num_passes`, and a varloop tree converted here would publish a range it
    // does not have.
    bool varloop = false;
    // Legal MAXIMUM pass count for a varloop tree (the model card's range).
    // Ignored when `varloop` is false.
    int max_passes = 1;
};

// The fixed-pass row's spec (Inkling-2-Mini): its own frozen passport.
InklingLoopConvertSpec MiniConvertSpec();

// config.json (+ hf_quant_config.json for the codec split) parsed into the
// topology the converter writes into model.index.json. The loader re-reads the
// index and cross-checks it against the frozen passport.
struct InklingLoopConvertTopology {
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
    // the loop core
    int loop_start = 0;
    int loop_end = 0;
    int loop_passes = 1;
    int loop_gate_rank = 0;
    // `loop_config.is_varloop` as declared by the checkpoint, and the ceiling
    // this pipeline is allowed to advertise (== loop_passes for a fixed tree).
    bool varloop = false;
    int loop_passes_max = 1;
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

InklingLoopConvertTopology ParseInklingLoopConfig(
    const std::string &input_dir, const InklingLoopConvertSpec &spec);
InklingLoopConvertTopology ParseInklingLoopConfig(const std::string &input_dir);
ParsedInklingLoopTensor ClassifyInklingLoopTensor(const std::string &name);

// Converts the checkpoint into the logestix-inkling-2-mini layout under
// <output_root>/logestix.
arrow::Status ConvertInklingLoopModel(const std::string &input_dir,
                                      const std::string &output_root);

// The same conversion, parameterised by the model row's spec. A varloop tree
// records `varloop: true` and the pass ceiling in its index; everything else in
// the artifact is identical, because the architecture is.
arrow::Status ConvertInklingLoopModelWithSpec(const std::string &input_dir,
                                              const std::string &output_root,
                                              const InklingLoopConvertSpec &spec);

}  // namespace lsxcommon::inkling