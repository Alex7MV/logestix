#pragma once

// Shared fixture builder for GLM-5.2 converter + loader tests.
// Builds a synthetic 79-layer safetensors model directory on disk and reads
// logged logestix layout files back, so both the converter and the loader
// tests exercise the same realistic layout.

#include <lsxcommon/schema.h>
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/writer.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace glmtest {

inline std::vector<uint8_t> MakeBf16(int64_t n, float base = 0.0f)
{
    std::vector<uint8_t> out(static_cast<size_t>(n) * 2);
    for (int64_t i = 0; i < n; ++i) {
        uint16_t bits = static_cast<uint16_t>(i % 65536);
        out[static_cast<size_t>(i) * 2 + 0] = static_cast<uint8_t>(bits & 0xff);
        out[static_cast<size_t>(i) * 2 + 1] = static_cast<uint8_t>(bits >> 8);
        (void)base;
    }
    return out;
}

// Round-to-nearest-even float → bf16 bits (host-side, no CUDA dependency).
inline uint16_t FloatToBf16(float v)
{
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    if ((u & 0x7f800000U) == 0x7f800000U)  // NaN/Inf: keep as-is (top 16 bits)
        return static_cast<uint16_t>(u >> 16);
    uint32_t rounding_bias = 0x7fffU + ((u >> 16) & 1U);
    u += rounding_bias;
    return static_cast<uint16_t>(u >> 16);
}

// Verified-value BF16 fixture: element i holds FloatToBf16(pattern(i)).
inline std::vector<uint8_t> MakeBf16Values(
    int64_t n, const std::function<float(int64_t)> &pattern)
{
    std::vector<uint8_t> out(static_cast<size_t>(n) * 2);
    for (int64_t i = 0; i < n; ++i) {
        uint16_t b = FloatToBf16(pattern(i));
        out[static_cast<size_t>(i) * 2 + 0] = static_cast<uint8_t>(b & 0xff);
        out[static_cast<size_t>(i) * 2 + 1] = static_cast<uint8_t>(b >> 8);
    }
    return out;
}

// Verified-value FP32 fixture: element i holds pattern(i) as a float.
inline std::vector<uint8_t> MakeF32Values(
    int64_t n, const std::function<float(int64_t)> &pattern)
{
    std::vector<uint8_t> out(static_cast<size_t>(n) * 4);
    for (int64_t i = 0; i < n; ++i) {
        float v = pattern(i);
        std::memcpy(out.data() + static_cast<size_t>(i) * 4, &v, 4);
    }
    return out;
}

inline std::vector<uint8_t> MakeF32(int64_t n)
{
    std::vector<uint8_t> out(static_cast<size_t>(n) * 4);
    for (int64_t i = 0; i < n; ++i) {
        uint32_t bits = static_cast<uint32_t>(i * 1000);
        std::memcpy(out.data() + static_cast<size_t>(i) * 4, &bits, 4);
    }
    return out;
}

struct FakeTensor {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
};

// Builds a single safetensors shard file on disk from in-memory tensors.
inline void WriteSafetensors(const std::filesystem::path &dir,
                             const std::string &fname,
                             const std::vector<FakeTensor> &tensors)
{
    // Header JSON    : {"__metadata__":{},"name":{"dtype":"...","shape":[...],"data_offsets":[b,e]},...}
    // File layout    : [8-byte BE header_len][header JSON padded][raw data]
    std::string header = R"({"__metadata__":{})";
    int64_t data_pos = 0;
    std::vector<std::pair<int64_t, int64_t>> offsets;  // begin,end of each tensor
    for (const auto &t : tensors) {
        std::string shape_s = absl::StrCat("[", absl::StrJoin(t.shape, ","), "]");
        int64_t begin = data_pos;
        int64_t end = data_pos + static_cast<int64_t>(t.data.size());
        absl::StrAppend(&header, ",\"", t.name, "\":{\"dtype\":\"", t.dtype,
                        "\",\"shape\":", shape_s, ",\"data_offsets\":[", begin,
                        ",", end, "]}");
        data_pos = end;
        offsets.push_back({begin, end});
    }
    header += "}";

    // Header must be padded to a multiple of 8 so the data section is aligned.
    while ((header.size() + 8) % 8 != 0)
        header += " ";

    std::ofstream f(dir / fname, std::ios::binary);
    uint64_t hlen = static_cast<uint64_t>(header.size());
    uint8_t lenbuf[8];
    for (int i = 0; i < 8; ++i) lenbuf[i] = static_cast<uint8_t>((hlen >> (8 * i)) & 0xff);
    f.write(reinterpret_cast<char*>(lenbuf), 8);
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const auto &t : tensors)
        f.write(reinterpret_cast<const char*>(t.data.data()),
                static_cast<std::streamsize>(t.data.size()));
}

inline std::vector<FakeTensor> BuildSyntheticModel()
{
    // Layer 0: dense MLP + attention (GLM layers 0-2 are dense).
    // Layer 3: MoE with 2 routed experts + 1 shared expert + router.
    // Layer 6: indexer layer.
    // MTP layer 78: eh_proj etc.
    // Global: embed / lm_head / model.norm.
    std::vector<FakeTensor> t;
    t.push_back({"model.embed_tokens.weight", "BF16", {64, 16},  MakeBf16(64 * 16)});
    t.push_back({"lm_head.weight",            "BF16", {64, 16},  MakeBf16(64 * 16)});
    t.push_back({"model.norm.weight",         "BF16", {16},      MakeBf16(16)});

    // Layer 0 (dense)
    t.push_back({"model.layers.0.input_layernorm.weight",  "BF16", {16}, MakeBf16(16)});
    t.push_back({"model.layers.0.post_attention_layernorm.weight", "BF16", {16}, MakeBf16(16)});
    t.push_back({"model.layers.0.self_attn.q_a_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.0.self_attn.q_a_layernorm.weight", "BF16", {16}, MakeBf16(16)});
    t.push_back({"model.layers.0.self_attn.q_b_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.0.self_attn.kv_a_proj_with_mqa.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.0.self_attn.kv_a_layernorm.weight", "BF16", {16}, MakeBf16(16)});
    t.push_back({"model.layers.0.self_attn.kv_b_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.0.self_attn.o_proj.weight",   "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.0.mlp.gate_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.0.mlp.down_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.0.mlp.up_proj.weight",   "BF16", {16, 16}, MakeBf16(256)});

    // Layer 3 (MoE): attention + norm
    t.push_back({"model.layers.3.input_layernorm.weight",  "BF16", {16}, MakeBf16(16)});
    t.push_back({"model.layers.3.post_attention_layernorm.weight", "BF16", {16}, MakeBf16(16)});
    t.push_back({"model.layers.3.self_attn.q_a_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.3.self_attn.q_a_layernorm.weight", "BF16", {16}, MakeBf16(16)});
    t.push_back({"model.layers.3.self_attn.q_b_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.3.self_attn.kv_a_proj_with_mqa.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.3.self_attn.kv_a_layernorm.weight", "BF16", {16}, MakeBf16(16)});
    t.push_back({"model.layers.3.self_attn.kv_b_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.3.self_attn.o_proj.weight",    "BF16", {16, 16}, MakeBf16(256)});
    // Router
    t.push_back({"model.layers.3.mlp.gate.weight",              "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.3.mlp.gate.e_score_correction_bias", "F32", {16}, MakeF32(16)});
    // Routed experts
    for (int e = 0; e < 2; ++e) {
        auto base = absl::StrCat("model.layers.3.mlp.experts.", e, ".");
        t.push_back({base + "gate_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
        t.push_back({base + "down_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
        t.push_back({base + "up_proj.weight",   "BF16", {16, 16}, MakeBf16(256)});
    }
    // Shared expert
    t.push_back({"model.layers.3.mlp.shared_experts.gate_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.3.mlp.shared_experts.down_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.3.mlp.shared_experts.up_proj.weight",   "BF16", {16, 16}, MakeBf16(256)});

    // Layer 6 (indexer)
    t.push_back({"model.layers.6.input_layernorm.weight",  "BF16", {16}, MakeBf16(16)});
    t.push_back({"model.layers.6.post_attention_layernorm.weight", "BF16", {16}, MakeBf16(16)});
    t.push_back({"model.layers.6.self_attn.q_a_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.6.self_attn.q_a_layernorm.weight", "BF16", {16}, MakeBf16(16)});
    t.push_back({"model.layers.6.self_attn.q_b_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.6.self_attn.kv_a_proj_with_mqa.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.6.self_attn.kv_a_layernorm.weight", "BF16", {16}, MakeBf16(16)});
    t.push_back({"model.layers.6.self_attn.kv_b_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.6.self_attn.o_proj.weight",    "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.6.self_attn.indexer.wq_b.weight",      "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.6.self_attn.indexer.wk.weight",        "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.6.self_attn.indexer.weights_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.6.self_attn.indexer.k_norm.weight",    "BF16", {16}, MakeBf16(16)});
    t.push_back({"model.layers.6.self_attn.indexer.k_norm.bias",      "F32",  {16}, MakeF32(16)});
    // Bare layers have MLP too (3-78 MoE)
    t.push_back({"model.layers.6.mlp.gate.weight",    "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.6.mlp.gate.e_score_correction_bias", "F32", {16}, MakeF32(16)});
    for (int e = 0; e < 2; ++e) {
        auto base = absl::StrCat("model.layers.6.mlp.experts.", e, ".");
        t.push_back({base + "gate_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
        t.push_back({base + "down_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
        t.push_back({base + "up_proj.weight",   "BF16", {16, 16}, MakeBf16(256)});
    }
    t.push_back({"model.layers.6.mlp.shared_experts.gate_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.6.mlp.shared_experts.down_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.6.mlp.shared_experts.up_proj.weight",   "BF16", {16, 16}, MakeBf16(256)});

    // MTP layer 78 (MoE, has own routed + shared experts per real model)
    t.push_back({"model.layers.78.enorm.weight", "BF16", {16}, MakeBf16(16)});
    t.push_back({"model.layers.78.hnorm.weight", "BF16", {16}, MakeBf16(16)});
    t.push_back({"model.layers.78.eh_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.78.shared_head.norm.weight", "BF16", {16}, MakeBf16(16)});
    t.push_back({"model.layers.78.mlp.gate.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.78.mlp.gate.e_score_correction_bias", "F32", {16}, MakeF32(16)});
    for (int e = 0; e < 2; ++e) {
        auto base = absl::StrCat("model.layers.78.mlp.experts.", e, ".");
        t.push_back({base + "gate_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
        t.push_back({base + "down_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
        t.push_back({base + "up_proj.weight",   "BF16", {16, 16}, MakeBf16(256)});
    }
    t.push_back({"model.layers.78.mlp.shared_experts.gate_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.78.mlp.shared_experts.down_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.78.mlp.shared_experts.up_proj.weight",   "BF16", {16, 16}, MakeBf16(256)});

    return t;
}

// Reads the logged arrow file and returns the tensor names it stores.
inline std::vector<std::string> TensorNamesIn(const std::filesystem::path &arrow_path)
{
    auto batch_res = lsxcommon::ReadRecordBatchFromFile(arrow_path);
    if (!batch_res.ok()) return {};
    auto t_res = lsxcommon::PinnedSchema::UnpackBatch(*batch_res);
    if (!t_res.ok()) return {};
    std::vector<std::string> names;
    for (auto &t : *t_res) names.push_back(t.name);
    return names;
}

// Writes the config.json the converter's integrity checks expect. Defaults
// match BuildSyntheticModel(): 79 layers, hidden 16, vocab 64, 2 local experts.
inline void WriteConfig(const std::filesystem::path &dir,
                        int num_layers = 79, int hidden = 16,
                        int vocab = 64, int experts = 2)
{
    std::string json = absl::StrCat(
        "{\"model_type\":\"glm\",",
        "\"num_layers\":", num_layers, ",",
        "\"hidden_size\":", hidden, ",",
        "\"vocab_size\":", vocab, ",",
        "\"num_local_experts\":", experts, ",",
        "\"num_experts\":", experts,
        "}");
    std::ofstream f(dir / "config.json");
    f << json;
}

// Copies the shared synthetic BPE tokenizer.json into `dir` so converters
// that make the tokenizer mandatory can build+verify it.
inline void WriteTokenizer(const std::filesystem::path &dir)
{
    std::filesystem::copy_file(
        TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
        dir / "tokenizer.json");
}

} // namespace glmtest