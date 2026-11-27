#pragma once

// GigaChat4-37B-A4B synthetic checkpoint fixture. Family-owned: re-exports the
// family-neutral primitives of test_synth_checkpoint.h under `gigachat4test::`
// and builds a tiny hybrid checkpoint (GatedDeltaNet + DSA MLA + MoE + MTP,
// mixed FP8/BF16) that exercises every converter and loader path without
// touching the real vault.
//
// Frozen gates are bypassed in converter/loader tests via
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 (runtime ENV, never a compile define).

#include "test_gigachat_moe_fixture.h"
#include "lsxcommon/index_layout.h"
#include "test_synth_checkpoint.h"

#include "lsxcommon/tokenizer.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace gigachat4test {

using synthtest::FakeTensor;
using synthtest::WriteSafetensors;

// Tiny synthetic geometry (all dims that feed an FP8 tensor are 128-multiples,
// which is what the block-128 dequantize kernel requires).
inline constexpr int kTH = 128;         // hidden
inline constexpr int kTVocab = 128;
inline constexpr int kTLayers = 2;      // layer 0 GDN + dense, layer 1 DSA MLA + MoE
inline constexpr int kTFullAttn = 1;    // layer 1
inline constexpr int kTExperts = 2;
inline constexpr int kTMoeI = 256;
inline constexpr int kTDenseI = 256;
inline constexpr int kTGateRank = 16;

// GatedDeltaNet: 1 key head, 2 value heads, hk = hv = 128.
inline constexpr int kTGdnKeyDim = 128;
inline constexpr int kTGdnValueDim = 256;
inline constexpr int kTGdnQkvz = 2 * kTGdnKeyDim + 2 * kTGdnValueDim;  // 768
inline constexpr int kTGdnConv = 2 * kTGdnKeyDim + kTGdnValueDim;      // 512
inline constexpr int kTGdnBa = 4;

// MLA: 2 heads, qk 128 nope + 64 rope, q_lora = kv_lora = 128.
inline constexpr int kTHeads = 2;
inline constexpr int kTQk = 192;
inline constexpr int kTQkOut = kTHeads * kTQk;                    // 384
inline constexpr int kTQlora = 128;
inline constexpr int kTKvLora = 128;
inline constexpr int kTKvA = kTKvLora + 64;                       // 192
inline constexpr int kTKvB = kTHeads * (128 + 128);               // 512
inline constexpr int kTVOut = kTHeads * 128;                      // 256

// DSA indexer: 2 index heads x 128, rope 64, top-k 2 (real: 64 x 128, 2048).
inline constexpr int kTIndexHeads = 2;
inline constexpr int kTIndexDim = 128;
inline constexpr int kTIndexRope = 64;
inline constexpr int kTIndexTopk = 2;
inline constexpr int kTIndexQ = kTIndexHeads * kTIndexDim;        // 256

inline std::filesystem::path MakeTempDir(const std::string &tag)
{
    return gigachattest::MakeTempDir(tag);
}

inline std::vector<uint8_t> MakeFp8(int64_t n) { return gigachattest::MakeFp8(n); }
inline std::vector<uint8_t> MakeOnesF32(int64_t n) { return gigachattest::MakeOnesF32(n); }

inline std::vector<uint8_t> Bf16Data(int64_t n, bool verified)
{
    return gigachattest::Bf16Data(n, verified);
}

inline void AddFp8(std::vector<FakeTensor> &ts, const std::string &base,
                   int64_t rows, int64_t cols, bool scale_inv = true)
{
    gigachattest::AddFp8(ts, base, rows, cols, scale_inv);
}

inline void AddBf16(std::vector<FakeTensor> &ts, const std::string &name,
                    std::vector<int64_t> shape, bool verified = false)
{
    gigachattest::AddBf16(ts, name, std::move(shape), verified);
}

// GatedNorm triple (weight + gate_up + gate_down).
inline void AddGatedNorm(std::vector<FakeTensor> &ts, const std::string &base,
                         int64_t dim, bool verified = false,
                         int64_t gate_rank = kTGateRank)
{
    gigachattest::AddGatedNorm(ts, base, dim, verified, gate_rank);
}

// Byte-level tokenizer arrows covering ALL 256 byte values (the family fixture
// covers ASCII only, which cannot represent the GCML markers' U+FF5C bytes).
inline void WriteByteLevelTokenizerArrows(const std::filesystem::path &pinned)
{
    namespace fs = std::filesystem;
    fs::create_directories(pinned);
    constexpr int32_t kTkVocab = 128016;
    std::vector<lsxcommon::VocabRow> rows(static_cast<size_t>(kTkVocab));
    for (int32_t id = 0; id < kTkVocab; ++id) rows[id].token_id = id;
    auto set = [&](int32_t id, const std::string &s, int8_t sp) {
        rows[id].token_bytes.assign(s.begin(), s.end());
        rows[id].is_special = sp;
    };
    set(1, "<s>", 1);
    set(2, "</s>", 1);
    for (int b = 0; b < 256; ++b)
        rows[3 + b].token_bytes.assign(1, static_cast<char>(b));
    set(128000, "<|role_sep|>\n", 1);
    set(128001, "<|message_sep|>\n\n", 1);
    set(128014, "<think>", 1);
    set(128015, "</think>", 1);

    lsxcommon::TokenizerVocabMeta meta;
    meta.bos_id = 1;
    meta.eos_id = 2;
    meta.pad_id = 2;
    meta.vocab_size = kTkVocab;
    for (int b = 0; b < 256; ++b) meta.byte_to_token_id[b] = 3 + b;
    auto vb = lsxcommon::TokenizerVocabSchema::MakeBatch(rows, meta);
    REQUIRE(vb.ok());
    REQUIRE(lsxcommon::WriteRecordBatchToFile(*vb,
                                             pinned / "tokenizer_vocab.arrow")
                .ok());
    std::vector<lsxcommon::MergeRow> merges(1);
    merges[0].merge_left = 1000000;
    merges[0].merge_right = 1000000;
    merges[0].merge_rank = 0;
    merges[0].merge_token_id = 0;
    auto mb = lsxcommon::TokenizerMergesSchema::MakeBatch(merges, 1);
    REQUIRE(mb.ok());
    REQUIRE(lsxcommon::WriteRecordBatchToFile(*mb,
                                             pinned / "tokenizer_merges.arrow")
                .ok());
}

// Byte-level BPE encode with a retry on the capacity error.
inline bool EncodeText(const lsxcommon::ArrowTokenizerDb &db,
                       const std::string &text, std::vector<int32_t> &out)
{
    for (size_t cap = 64; cap <= (1u << 20); cap *= 2)
    {
        std::vector<int32_t> ids(cap, 0);
        size_t n = cap;
        const int32_t r = lsxcommon::arrow_tokenizer_encode(
            db, text.data(), text.size(), ids.data(), &n);
        if (r == -2) continue;
        if (r <= 0) return false;
        out.assign(ids.begin(), ids.begin() + r);
        return true;
    }
    return false;
}

inline void WriteConfig(const std::filesystem::path &dir)
{
    std::ofstream f(dir / "config.json");
    f << R"({
  "architectures": ["GigaChat4ForCausalLM"],
  "model_type": "gigachat4",
  "hidden_size": 128,
  "num_hidden_layers": 2,
  "vocab_size": 128,
  "num_attention_heads": 2,
  "num_key_value_heads": 2,
  "q_lora_rank": 128,
  "kv_lora_rank": 128,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "qk_head_dim": 192,
  "v_head_dim": 128,
  "n_routed_experts": 2,
  "num_experts_per_tok": 1,
  "moe_intermediate_size": 256,
  "intermediate_size": 256,
  "first_k_dense_replace": 1,
  "rms_norm_eps": 1e-06,
  "routed_scaling_factor": 2.5,
  "swiglu_limit": 10,
  "final_logit_softcapping": 30,
  "index_n_heads": 2,
  "index_head_dim": 128,
  "index_topk": 2,
  "linear_attention_type": "GigaChat4GatedDeltaNet",
  "full_attention_layers": [1],
  "num_nextn_predict_layers": 1,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 2
})";
}

inline void WriteGigaChat4Checkpoint(const std::filesystem::path &dir,
                                     bool verified = false,
                                     bool extra_unknown = false)
{
    WriteConfig(dir);

    std::vector<FakeTensor> ts;

    // Globals.
    AddBf16(ts, "model.embed_tokens.weight", {kTVocab, kTH}, verified);
    AddBf16(ts, "lm_head.weight", {kTVocab, kTH}, verified);
    AddGatedNorm(ts, "model.norm", kTH, verified);

    // Layer 0: GatedDeltaNet + dense MLP (dense projections are BF16 here,
    // unlike GigaChat3.5, and still become NVFP4).
    for (const char *n : {"input_layernorm", "post_attention_layernorm",
                          "post_self_attn_layernorm",
                          "post_feedforward_layernorm"})
        AddGatedNorm(ts, std::string("model.layers.0.") + n, kTH, verified);
    AddFp8(ts, "model.layers.0.self_attn.in_proj_qkvz", kTGdnQkvz, kTH);
    AddFp8(ts, "model.layers.0.self_attn.out_proj", kTH, kTGdnValueDim);
    AddBf16(ts, "model.layers.0.self_attn.conv1d.weight", {kTGdnConv, 1, 4},
            verified);
    AddBf16(ts, "model.layers.0.self_attn.A_log", {2}, verified);
    AddBf16(ts, "model.layers.0.self_attn.dt_bias", {2}, verified);
    AddBf16(ts, "model.layers.0.self_attn.in_proj_ba.weight", {kTGdnBa, kTH},
            verified);
    AddBf16(ts, "model.layers.0.self_attn.norm.weight", {128}, verified);
    AddBf16(ts, "model.layers.0.mlp.gate_proj.weight", {kTDenseI, kTH}, verified);
    AddBf16(ts, "model.layers.0.mlp.up_proj.weight", {kTDenseI, kTH}, verified);
    AddBf16(ts, "model.layers.0.mlp.down_proj.weight", {kTH, kTDenseI}, verified);

    // Layer 1: gated MLA with DSA + MoE (BF16 shared experts and attention gate).
    for (const char *n : {"input_layernorm", "post_attention_layernorm",
                          "post_self_attn_layernorm",
                          "post_feedforward_layernorm"})
        AddGatedNorm(ts, std::string("model.layers.1.") + n, kTH, verified);
    AddFp8(ts, "model.layers.1.self_attn.q_a_proj", kTQlora, kTH);
    AddGatedNorm(ts, "model.layers.1.self_attn.q_a_layernorm", kTQlora, false);
    AddFp8(ts, "model.layers.1.self_attn.q_b_proj", kTQkOut, kTQlora);
    AddFp8(ts, "model.layers.1.self_attn.kv_a_proj_with_mqa", kTKvA, kTH);
    AddGatedNorm(ts, "model.layers.1.self_attn.kv_a_layernorm", kTKvLora, false);
    AddFp8(ts, "model.layers.1.self_attn.kv_b_proj", kTKvB, kTKvLora);
    AddFp8(ts, "model.layers.1.self_attn.o_proj", kTH, kTVOut);
    AddBf16(ts, "model.layers.1.self_attn.gate_proj.weight", {kTVOut, kTH},
            verified);
    // DSA indexer: wq_b is FP8 in the checkpoint, the rest BF16.
    AddFp8(ts, "model.layers.1.self_attn.indexer.wq_b", kTIndexQ, kTQlora);
    AddBf16(ts, "model.layers.1.self_attn.indexer.wk.weight", {kTIndexDim, kTH},
            verified);
    AddBf16(ts, "model.layers.1.self_attn.indexer.weights_proj.weight",
            {kTIndexHeads, kTH}, verified);
    AddBf16(ts, "model.layers.1.self_attn.indexer.k_norm.weight", {kTIndexDim},
            verified);
    AddBf16(ts, "model.layers.1.mlp.gate.weight", {kTExperts, kTH}, verified);
    AddBf16(ts, "model.layers.1.mlp.gate.e_score_correction_bias", {kTExperts},
            verified);
    AddBf16(ts, "model.layers.1.mlp.shared_experts.gate_proj.weight",
            {kTMoeI, kTH}, verified);
    AddBf16(ts, "model.layers.1.mlp.shared_experts.up_proj.weight",
            {kTMoeI, kTH}, verified);
    AddBf16(ts, "model.layers.1.mlp.shared_experts.down_proj.weight",
            {kTH, kTMoeI}, verified);
    for (int e = 0; e < kTExperts; ++e)
    {
        const std::string p =
            "model.layers.1.mlp.experts." + std::to_string(e);
        AddFp8(ts, p + ".gate_proj", kTMoeI, kTH);
        AddFp8(ts, p + ".up_proj", kTMoeI, kTH);
        AddFp8(ts, p + ".down_proj", kTH, kTMoeI);
    }

    // MTP layer 2: stored, never applied.
    AddBf16(ts, "model.layers.2.eh_proj.weight", {kTH, 2 * kTH}, verified);
    AddBf16(ts, "model.layers.2.embed_tokens.weight", {kTVocab, kTH}, verified);
    AddGatedNorm(ts, "model.layers.2.enorm", kTH, verified);
    AddGatedNorm(ts, "model.layers.2.hnorm", kTH, verified);
    AddGatedNorm(ts, "model.layers.2.shared_head.norm", kTH, verified);
    AddBf16(ts, "model.layers.2.shared_head.head.weight", {kTVocab, kTH},
            verified);

    if (extra_unknown)
        AddBf16(ts, "model.layers.0.mystery_proj.weight", {kTH, kTH}, verified);

    WriteSafetensors(dir, "model.safetensors", ts);
}

// Writes a real-geometry model.index.json (no weights) so the loader's frozen
// passport can be exercised without the synthetic-geometry bypass.
inline void WriteGigaChat4Index(
    const std::filesystem::path &model_root, int hidden_override = 2560,
    int64_t layout = lsxcommon::kIndexLayout)
{
    std::filesystem::create_directories(model_root);
    std::ofstream f(model_root / "model.index.json");
    f << "{\n  \"layout\": " << layout << ",\n"
      << "  \"model\": \"GigaChat4-37B-A4B\",\n"
      << "  \"num_layers\": 32,\n  \"full_attention_layers\": "
         "[3,7,11,15,19,23,27,31],\n"
      << "  \"hidden_size\": " << hidden_override
      << ",\n  \"vocab_size\": 128256,\n  \"num_attention_heads\": 32,\n"
      << "  \"num_key_value_heads\": 32,\n  \"num_experts\": 192,\n"
      << "  \"num_experts_per_tok\": 8,\n  \"moe_intermediate_size\": 768,\n"
      << "  \"first_k_dense_replace\": 1,\n  \"rms_norm_eps\": 1e-06,\n"
      << "  \"router_scaling_factor\": 2.5,\n  \"swiglu_limit\": 10,\n"
      << "  \"logit_softcap\": 30,\n  \"attn_scale\": 0.13086079996295008,\n"
      << "  \"index_heads\": 64,\n  \"index_head_dim\": 128,\n"
      << "  \"index_topk\": 2048,\n  \"index_rope\": 64,\n"
      << "  \"bos_token_id\": 1,\n  \"eos_token_id\": 2,\n  \"pad_token_id\": 2,\n"
      << "  \"deferred\": {\"mtp\": \"stored\", \"mtp_layer\": 32, "
         "\"mtp_tensors\": 129},\n  \"total_pinned_tensors\": 0\n}\n";
}

// The 708B row: same tiny synthetic weights, but an index whose identity and
// geometry are the real 708B numbers. Used by the loader/frozen-gate tests,
// which run with the synthetic-geometry bypass OFF.
inline void WriteGigaChat4_708BIndex(
    const std::filesystem::path &model_root, int hidden_override = 7168,
    int64_t layout_override = lsxcommon::kIndexLayout)
{
    std::filesystem::create_directories(model_root);
    std::ofstream f(model_root / "model.index.json");
    f << "{\n  \"layout\": " << layout_override << ",\n"
      << "  \"model\": \"GigaChat4-708B-A41B\",\n"
      << "  \"num_layers\": 64,\n  \"full_attention_layers\": "
         "[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63],\n"
      << "  \"hidden_size\": " << hidden_override
      << ",\n  \"vocab_size\": 128256,\n  \"num_attention_heads\": 64,\n"
      << "  \"num_key_value_heads\": 64,\n  \"num_experts\": 256,\n"
      << "  \"num_experts_per_tok\": 8,\n  \"moe_intermediate_size\": 2048,\n"
      << "  \"dense_intermediate_size\": 18432,\n"
      << "  \"first_k_dense_replace\": 3,\n  \"rms_norm_eps\": 1e-06,\n"
      << "  \"router_scaling_factor\": 2.5,\n  \"swiglu_limit\": 10,\n"
      << "  \"logit_softcap\": 30,\n  \"attn_scale\": 0.13086079996295008,\n"
      << "  \"index_heads\": 64,\n  \"index_head_dim\": 128,\n"
      << "  \"index_topk\": 2048,\n  \"index_rope\": 64,\n"
      << "  \"bos_token_id\": 1,\n  \"eos_token_id\": 2,\n  \"pad_token_id\": 2,\n"
      << "  \"deferred\": {\"mtp\": \"stored\", \"mtp_layer\": 64, "
         "\"mtp_tensors\": 11},\n  \"total_pinned_tensors\": 0\n}\n";
}

} // namespace gigachat4test
