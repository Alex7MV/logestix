#pragma once

// Synthetic Muse-Glimmer-30B checkpoint fixture.
//
// Family-specific geometry only: the byte-pattern / safetensors primitives are
// the model-neutral ones from test_synth_checkpoint.h (test-suite autonomy
// invariant - a family fixture never borrows another family's fixture).
//
// Shapes mirror the real checkpoint proportionally:
//   real: hidden 6656, q/gate 4096x6656, kv 256x6656, o 6656x4096,
//         mlp 19968x6656, vocab 202048 (200000 BPE + 2048 added)
//   here: hidden 64,   q/gate 64x64,    kv 32x64,    o 64x64,
//         mlp 32x64,   vocab 266 (259 BPE + 7 added)
// Every layer has ALL FIVE attention projections (q/k/v/o/gate — the gate is
// the attention OUTPUT gate, present on sliding AND full layers), four
// centered norms, and a SwiGLU MLP. No q_norm/k_norm weights (weightless QK
// norm), no layer_scalar, no experts, no router.

#include <absl/strings/str_cat.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "lsxcommon/byte_unicode.h"
#include "test_synth_checkpoint.h"

namespace muse_dense_test {

struct MuseDenseTestConfig {
    int hidden_size = 64;
    int intermediate_size = 32;   // 16-multiple: NVFP4 block size
    int vocab_size = 266;         // 259 BPE + 7 added specials
    int base_vocab_size = 259;
    int num_layers = 4;
    std::vector<std::string> layer_types = {
        "sliding_attention", "sliding_attention", "sliding_attention",
        "full_attention"};

    int num_attention_heads = 4;
    int num_key_value_heads = 2;
    int head_dim = 16;
    int sliding_window = 8;

    // Special ids in the synthetic added-token range (259..265).
    int bos_id = 259;         // <|begin_of_text|>
    int eos_text_id = 260;    // <|end_of_text|>
    int eot_id = 261;         // <|eot|>
    int eom_id = 262;         // <|eom|>
    int pad_id = 263;         // <|finetune_right_pad|>
    int start_id = 264;       // <|start|>
    int message_id = 265;     // <|message|>
};

inline const MuseDenseTestConfig &TestCfg()
{
    static MuseDenseTestConfig cfg;
    return cfg;
}

inline int QRows()
{
    return TestCfg().num_attention_heads * TestCfg().head_dim;
}
inline int KvRows()
{
    return TestCfg().num_key_value_heads * TestCfg().head_dim;
}

// Every text tensor of the synthetic checkpoint, in the real safetensors
// layout (single shard; the converter is shard-agnostic).
inline std::vector<synthtest::FakeTensor> BuildSyntheticMuseDenseModel()
{
    const auto &c = TestCfg();
    const int H = c.hidden_size;
    const int I = c.intermediate_size;
    const int V = c.vocab_size;
    const int q_rows = QRows();
    const int kv_rows = KvRows();

    std::vector<synthtest::FakeTensor> t;

    t.push_back({"model.language_model.embed_tokens.weight", "BF16", {V, H},
                 synthtest::MakeBf16(V * H)});
    t.push_back({"lm_head.weight", "BF16", {V, H},
                 synthtest::MakeBf16(V * H)});
    t.push_back({"model.language_model.norm.weight", "BF16", {H},
                 synthtest::MakeBf16(H)});

    for (int L = 0; L < c.num_layers; ++L) {
        const std::string lp =
            "model.language_model.layers." + std::to_string(L) + ".";

        t.push_back({lp + "input_layernorm.weight", "BF16", {H},
                     synthtest::MakeBf16(H)});
        t.push_back({lp + "post_attention_layernorm.weight", "BF16", {H},
                     synthtest::MakeBf16(H)});
        t.push_back({lp + "pre_feedforward_layernorm.weight", "BF16", {H},
                     synthtest::MakeBf16(H)});
        t.push_back({lp + "post_feedforward_layernorm.weight", "BF16", {H},
                     synthtest::MakeBf16(H)});

        t.push_back({lp + "mlp.gate_proj.weight", "BF16", {I, H},
                     synthtest::MakeBf16(I * H)});
        t.push_back({lp + "mlp.up_proj.weight", "BF16", {I, H},
                     synthtest::MakeBf16(I * H)});
        t.push_back({lp + "mlp.down_proj.weight", "BF16", {H, I},
                     synthtest::MakeBf16(H * I)});

        const std::string sa = lp + "self_attn.";
        t.push_back({sa + "q_proj.weight", "BF16", {q_rows, H},
                     synthtest::MakeBf16(q_rows * H)});
        t.push_back({sa + "k_proj.weight", "BF16", {kv_rows, H},
                     synthtest::MakeBf16(kv_rows * H)});
        t.push_back({sa + "v_proj.weight", "BF16", {kv_rows, H},
                     synthtest::MakeBf16(kv_rows * H)});
        t.push_back({sa + "o_proj.weight", "BF16", {H, q_rows},
                     synthtest::MakeBf16(H * q_rows)});
        // The attention OUTPUT gate: present on EVERY layer (sliding + full).
        t.push_back({sa + "gate_proj.weight", "BF16", {q_rows, H},
                     synthtest::MakeBf16(q_rows * H)});
    }

    // Vision ballast - counted, never applied.
    t.push_back({"model.vision_tower.layers.0.attn.q_proj.weight", "BF16",
                 {H, H}, synthtest::MakeBf16(H * H)});
    t.push_back({"model.vision_tower.ln_post.weight", "BF16", {H},
                 synthtest::MakeBf16(H)});
    t.push_back({"model.vision_adapter.fc1.weight", "BF16", {H, H},
                 synthtest::MakeBf16(H * H)});
    t.push_back({"model.vision_projection.weight", "BF16", {H, H},
                 synthtest::MakeBf16(H * H)});

    return t;
}

// config.json in the real HF nesting (model_type muse_glimmer + text_config).
inline void WriteConfig(const std::filesystem::path &dir)
{
    const auto &c = TestCfg();
    std::string layer_types = "[";
    std::string rope_thetas = "[";
    for (size_t i = 0; i < c.layer_types.size(); ++i) {
        if (i) { layer_types += ","; rope_thetas += ","; }
        layer_types += "\"" + c.layer_types[i] + "\"";
        rope_thetas += (c.layer_types[i] == "full_attention") ? "0" : "500000.0";
    }
    layer_types += "]";
    rope_thetas += "]";

    const std::string json =
        "{\"model_type\":\"muse_glimmer\","
        "\"image_token_id\":200092,\"video_token_id\":200091,"
        "\"text_config\":{"
        "\"model_type\":\"muse_glimmer_text\","
        "\"hidden_size\":" + std::to_string(c.hidden_size) +
        ",\"intermediate_size\":" + std::to_string(c.intermediate_size) +
        ",\"num_hidden_layers\":" + std::to_string(c.num_layers) +
        ",\"vocab_size\":" + std::to_string(c.vocab_size) +
        ",\"num_attention_heads\":" + std::to_string(c.num_attention_heads) +
        ",\"num_key_value_heads\":" + std::to_string(c.num_key_value_heads) +
        ",\"head_dim\":" + std::to_string(c.head_dim) +
        ",\"sliding_window\":" + std::to_string(c.sliding_window) +
        ",\"attention_bias\":false"
        ",\"rms_norm_eps\":1e-05"
        ",\"post_norm_eps\":1e-08"
        ",\"qk_scale_factor\":3.87"
        ",\"output_multiplier\":0.19611613513818404"
        ",\"final_logit_softcapping\":20.0"
        ",\"hidden_activation\":\"silu\""
        ",\"tie_word_embeddings\":false"
        ",\"bos_token_id\":" + std::to_string(c.bos_id) +
        ",\"eos_token_id\":" + std::to_string(c.eos_text_id) +
        ",\"pad_token_id\":null"
        ",\"layer_types\":" + layer_types +
        ",\"layer_rope_theta\":" + rope_thetas +
        ",\"rope_parameters\":{\"rope_theta\":500000.0,"
        "\"rope_type\":\"default\"}"
        "}}";
    std::ofstream f(dir / "config.json");
    f << json;

    // The real generation_config adds the turn-end <|eot|> and the pad id.
    std::ofstream g(dir / "generation_config.json");
    g << "{\"bos_token_id\":" << c.bos_id
      << ",\"eos_token_id\":[" << c.eos_text_id << "," << c.eot_id << "]"
      << ",\"pad_token_id\":" << c.pad_id << "}";
}

// Minimal HF tokenizer.json (+ tokenizer_config.json): the GPT-2 byte table
// (ids 0..255), three merges (256..258) and the ATEM control markers
// (259..265, ALL special:true) — the shape BuildTokenizerArrowFiles expects
// for a kind-3 (Llama-3/o200k) byte-level BPE checkpoint.
inline void WriteTokenizerJson(const std::filesystem::path &dir)
{
    const auto &c = TestCfg();
    auto escape = [](const std::string &s) {
        std::string out;
        for (char ch : s) {
            switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04X",
                                  static_cast<unsigned char>(ch));
                    out += buf;
                } else {
                    out.push_back(ch);
                }
            }
        }
        return out;
    };
    auto key_of = [](const std::string &raw) {
        return lsxcommon::byte_unicode::bytes_to_key(
            reinterpret_cast<const uint8_t *>(raw.data()), raw.size());
    };

    std::ostringstream j;
    // added_tokens must precede `model` (simdjson on-demand is forward-only).
    j << "{\"added_tokens\":[";
    const std::vector<std::pair<int32_t, std::string>> added = {
        {c.bos_id, "<|begin_of_text|>"},
        {c.eos_text_id, "<|end_of_text|>"},
        {c.eot_id, "<|eot|>"},
        {c.eom_id, "<|eom|>"},
        {c.pad_id, "<|finetune_right_pad|>"},
        {c.start_id, "<|start|>"},
        {c.message_id, "<|message|>"},
    };
    for (size_t i = 0; i < added.size(); ++i) {
        if (i) j << ",";
        j << "{\"id\":" << added[i].first << ",\"content\":\""
          << escape(added[i].second) << "\",\"single_word\":false,"
          << "\"lstrip\":false,\"rstrip\":false,\"normalized\":false,"
          << "\"special\":true}";
    }
    // The EXACT o200k/Llama-3 split pattern the real checkpoint ships.
    j << "],\"normalizer\":null,\"pre_tokenizer\":{\"type\":\"Sequence\","
         "\"pretokenizers\":[{\"type\":\"Split\",\"pattern\":{\"Regex\":\""
         "[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?[\\\\p{Lu}\\\\p{Lt}\\\\p{Lm}\\\\p{Lo}"
         "\\\\p{M}]*[\\\\p{Ll}\\\\p{Lm}\\\\p{Lo}\\\\p{M}]+"
         "(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
         "[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?[\\\\p{Lu}\\\\p{Lt}\\\\p{Lm}\\\\p{Lo}"
         "\\\\p{M}]+[\\\\p{Ll}\\\\p{Lm}\\\\p{Lo}\\\\p{M}]*"
         "(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
         "\\\\p{N}{1,3}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n/]*|"
         "\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+\"},"
         "\"behavior\":\"Isolated\",\"invert\":false},"
         "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,"
         "\"trim_offsets\":true,\"use_regex\":false}]},"
         "\"model\":{\"type\":\"BPE\",\"dropout\":null,\"unk_token\":null,"
         "\"continuing_subword_prefix\":null,\"end_of_word_suffix\":null,"
         "\"fuse_unk\":false,\"vocab\":{";
    bool first = true;
    auto emit = [&](int32_t id, const std::string &key) {
        if (!first) j << ",";
        first = false;
        j << "\"" << escape(key) << "\":" << id;
    };
    for (int b = 0; b < 256; ++b) {
        const char raw = static_cast<char>(b);
        emit(b, key_of(std::string(1, raw)));
    }
    emit(256, key_of("th"));
    emit(257, key_of("he"));
    emit(258, key_of("in"));
    j << "},\"merges\":[[\"";
    j << escape(key_of("t")) << "\",\"" << escape(key_of("h")) << "\"],[\""
      << escape(key_of("h")) << "\",\"" << escape(key_of("e")) << "\"],[\""
      << escape(key_of("i")) << "\",\"" << escape(key_of("n")) << "\"]]},"
         "\"post_processor\":null,\"decoder\":{\"type\":\"ByteLevel\","
         "\"add_prefix_space\":true,\"trim_offsets\":true,"
         "\"use_regex\":true}}";

    std::ofstream f(dir / "tokenizer.json");
    f << j.str();

    std::ofstream cfg(dir / "tokenizer_config.json");
    cfg << R"({"bos_token":"<|begin_of_text|>","eos_token":"<|end_of_text|>",)"
        << R"("pad_token":"<|finetune_right_pad|>"})";
}

} // namespace muse_dense_test