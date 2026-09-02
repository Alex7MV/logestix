#pragma once

#include <arrow/api.h>
#include <string>
#include <vector>

#include "lsxcommon/vram_estimate.h"

namespace lsxcommon::qwen_moe_exp {

// Tensor kinds of the Qwen3.8-Flash-Next (qwen4_exp_text) checkpoint.
// QSA = Qwen Sparse Attention (full-attention layers with micro-block indexer);
// GDN = Gated DeltaNet (linear-attention layers);
// hc = GatedResidual hyper-connections (attn + mlp per layer, global mixer);
// PLE = Per-Layer n-gram Embedding (layer 2 / HF layer_idx 1).
enum class ExpTensorKind {
    Unknown,
    Embed,               // model.language_model.embed_tokens.weight
    LmHead,              // lm_head.weight
    FinalNorm,           // model.language_model.norm.weight (absent in real
                         //  qwen4_exp inventory; classifier keeps the slot)
    GlobalMixerNorm,     // model.language_model.hyper_connection_mixer.hc_norm
    GlobalMixerDown,     // .../input_mix_weight_down
    GlobalMixerUp,       // .../input_mix_weight_up
    GdnQkv,              // layers.N.linear_attn.in_proj_qkv.weight
    GdnZ,                // layers.N.linear_attn.in_proj_z.weight
    GdnB,                // layers.N.linear_attn.in_proj_b.weight
    GdnA,                // layers.N.linear_attn.in_proj_a.weight
    GdnOut,              // layers.N.linear_attn.out_proj.weight
    GdnConv,             // layers.N.linear_attn.conv1d.weight
    GdnNorm,             // layers.N.linear_attn.norm.weight
    GdnALog,             // layers.N.linear_attn.A_log
    GdnDt,               // layers.N.linear_attn.dt_bias
    QsaQ,                // layers.N.self_attn.q_proj.weight (gate doubled)
    QsaK,                // layers.N.self_attn.k_proj.weight
    QsaV,                // layers.N.self_attn.v_proj.weight
    QsaO,                // layers.N.self_attn.o_proj.weight
    QsaQNorm,            // layers.N.self_attn.q_norm.weight
    QsaKNorm,            // layers.N.self_attn.k_norm.weight
    IndexerQkProj,       // layers.N.self_attn.indexer.index_qk_proj.weight
    IndexerQNorm,        // layers.N.self_attn.indexer.q_layernorm.weight
    IndexerKNorm,        // layers.N.self_attn.indexer.k_layernorm.weight
    AttnHcNorm,          // layers.N.attn_hyper_connection.hc_norm.weight
    AttnHcDown,          // layers.N.attn_hyper_connection.input_mix_weight_down
    AttnHcUp,            // layers.N.attn_hyper_connection.input_mix_weight_up
    AttnHcBlockInject,   // layers.N.attn_hyper_connection.block_inject_weight
    MlpHcNorm,           // layers.N.mlp_hyper_connection.hc_norm.weight
    MlpHcDown,           // layers.N.mlp_hyper_connection.input_mix_weight_down
    MlpHcUp,             // layers.N.mlp_hyper_connection.input_mix_weight_up
    MlpHcBlockInject,    // layers.N.mlp_hyper_connection.block_inject_weight
    RouterGate,          // layers.N.mlp.gate.weight [E,H] — stays BF16
    FusedExpertsGateUp,  // layers.N.mlp.experts.gate_up_proj [E,2I,H]
    FusedExpertsDown,    // layers.N.mlp.experts.down_proj [E,H,I]
    SharedGateProj,      // layers.N.mlp.shared_expert.gate_proj.weight
    SharedUpProj,        // layers.N.mlp.shared_expert.up_proj.weight
    SharedDownProj,      // layers.N.mlp.shared_expert.down_proj.weight
    SharedGate,          // layers.N.mlp.shared_expert_gate.weight [1,H]
    PleKeyProj,          // layers.N.ple.key_proj.weight
    PleValueProj,        // layers.N.ple.value_proj.weight
    PleNormKey,          // layers.N.ple.norm_key.weight
    PleNormQuery,        // layers.N.ple.norm_query.weight
    PleNormConv,         // layers.N.ple.norm_conv.weight
    PleConv,             // layers.N.ple.conv1d.weight
    PleNgramShard,       // layers.N.ple.ple_embedding.ngram_embedding.shard_S.weight
    PleMeta,             // layer_multipliers / ngram_heads_offsets / sizes (I64)
    Mtp,                 // mtp.* — converted+stored, never applied at inference
    Visual,              // visual/model.visual.* — skipped and counted
};

struct ParsedExpTensor {
    ExpTensorKind kind = ExpTensorKind::Unknown;
    int layer = -1;
    int shard = -1;
};

// Classifies an HF qwen4_exp tensor name.
ParsedExpTensor classify_qwen4_exp_tensor(const std::string &name);

// True when `kind` is NVFP4-quantized by the converter. Norms, router gate
// matrix, shared-expert scalar gate, conv weights, A_log/dt_bias, PLE norms
// and n-gram shards always stay high precision.
bool qwen4_exp_tensor_is_nvfp4(ExpTensorKind kind);

struct Qwen4ExpTopology {
    int num_layers = 0, hidden_size = 0, vocab_size = 0;
    int num_attention_heads = 0, num_key_value_heads = 0, head_dim = 0;
    int linear_num_key_heads = 0, linear_num_value_heads = 0;
    int linear_key_head_dim = 0, linear_value_head_dim = 0;
    int linear_conv_kernel_dim = 4;
    int num_experts = 0, num_experts_per_tok = 0;
    int moe_intermediate_size = 0, shared_expert_intermediate_size = 0;
    int hc_count = 4, hc_lowrank = 320;
    int indexer_n_heads = 4, indexer_kv_heads = 1, indexer_head_dim = 128;
    int indexer_budget = 2048, indexer_compress_ratio = 4;
    int ngram_size = 3, heads_per_ngram = 8, ple_embed_dim = 0;
    int ple_conv_kernel_size = 4, split_ngram_parts = 128;
    int eos_token_id = -1;
    int64_t ngram_vocab_size_base = 20000000;
    std::vector<std::string> layer_types;
    std::vector<int> ple_layer_ids;
    double rms_norm_eps = 1e-6, partial_rotary_factor = 0.25;
    double rope_theta = 10000000.0;
    int total_pinned_tensors = 0;
    VramProfile vram;       // conversion-time VRAM requirements
    std::string error_msg;  // non-empty when config is invalid
};

// Parses text_config from <input_dir>/config.json via simdjson. Falls back to
// flat root keys when text_config is absent. On any structural problem the
// returned struct carries a non-empty error_msg.
Qwen4ExpTopology ParseQwen4ExpConfig(const std::string &input_dir);

// Bit-exact port of HF _build_layer_multipliers (splitmix64 hash chain).
// Verified 2026-09-02 against the checkpoint's stored layer_multipliers:
// seed=1234, ple_layer_index=0, vocab=248320 ->
// {23703573157769, 20109073645365, 8052911324071}.
void Splitmix64Multipliers(int64_t seed, int ple_layer_index,
                           int64_t unigram_vocab, int ngram_size,
                           int64_t out[3]);

// Port of HF _find_nth_prime_after: count-th prime strictly greater than
// `start`. Verified against the checkpoint's ngram_heads_vocab_sizes.
int64_t FindNthPrimeAfter(int64_t start, int64_t count);

// Converts a Qwen3.8-Flash-Next model directory (safetensors shards +
// config.json) into the unified <output_root>/logestix arrow layout (vault
// convention). Format tag: logestix-qwen38flashnext.
arrow::Status ConvertQwen38FlashNextModel(const std::string &input_dir,
                                          const std::string &output_root);

} // namespace lsxcommon::qwen_moe_exp
