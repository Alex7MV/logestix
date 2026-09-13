#pragma once

// Synthetic DeepSeek-V4.1-Flash index fixture (host-only). P1 needs only the
// index JSON; P2/P3 extend this header with pinned/ and expert Arrow writers.
// Family-neutral helpers (FakeTensor/WriteSafetensors/MakeBf16*) come from
// test_synth_checkpoint.h when P2 lands — never from another family's fixture.
#include <filesystem>
#include <fstream>
#include <cstring>
#include <string>
#include <vector>

#include <arrow/status.h>

#include <lsxcommon/schema.h>

namespace v41fixtest {

inline void WriteIndex(const std::filesystem::path &model_dir,
                       const std::string &json)
{
    std::filesystem::create_directories(model_dir / "logestix");
    std::ofstream f(model_dir / "logestix" / "model.index.json");
    f << json;
}

inline std::string DefaultIndexJson(int num_layers, int experts,
                                    const std::string &format)
{
    return "{\n"
           "  \"format\": \"" + format + "\",\n"
           "  \"model\": \"DeepSeek-V4.1-Flash\",\n"
           "  \"num_layers\": " + std::to_string(num_layers) + ",\n"
           "  \"experts_per_layer\": " + std::to_string(experts) + ",\n"
           "  \"total_experts\": " +
           std::to_string(static_cast<long long>(num_layers) * experts) +
           ",\n"
           "  \"vram\": {\n"
           "    \"min_bytes\": 1024,\n"
           "    \"optimal_bytes\": 2048,\n"
           "    \"max_bytes\": 4096,\n"
           "    \"context_tokens\": 1048576\n"
           "  }\n"
           "}\n";
}

inline arrow::Status BuildSyntheticIndex(
    const std::filesystem::path &model_dir, int num_layers = 40,
    int experts = 384, const std::string &format = "logestix-v41-nvfp4")
{
    WriteIndex(model_dir, DefaultIndexJson(num_layers, experts, format));
    return arrow::Status::OK();
}

inline arrow::Status BuildSyntheticIndexRaw(
    const std::filesystem::path &model_dir, const std::string &json)
{
    WriteIndex(model_dir, json);
    return arrow::Status::OK();
}

// ── Synthetic converted layout (pinned + expert arrows) ─────────────────────
// Used by the CUDA load-stage test; a single 64-wide layer with the full
// required dense tensor set for a compress_ratio-0 (pure window) layer.
inline lsxcommon::TensorRecord Bf16Rec(const std::string &name,
                                       std::vector<int64_t> shape,
                                       uint16_t bits)
{
    lsxcommon::TensorRecord t;
    t.name = name;
    t.dtype = "BF16";
    t.shape = std::move(shape);
    int64_t n = 1;
    for (int64_t d : t.shape) n *= d;
    t.data.resize(static_cast<size_t>(n) * 2);
    for (int64_t i = 0; i < n; ++i) {
        t.data[static_cast<size_t>(i) * 2] = static_cast<uint8_t>(bits & 0xff);
        t.data[static_cast<size_t>(i) * 2 + 1] =
            static_cast<uint8_t>(bits >> 8);
    }
    return t;
}

inline lsxcommon::TensorRecord F32Rec(const std::string &name,
                                      std::vector<int64_t> shape, float fill)
{
    lsxcommon::TensorRecord t;
    t.name = name;
    t.dtype = "F32";
    t.shape = std::move(shape);
    int64_t n = 1;
    for (int64_t d : t.shape) n *= d;
    t.data.resize(static_cast<size_t>(n) * 4);
    for (size_t off = 0; off < t.data.size(); off += 4)
        std::memcpy(t.data.data() + off, &fill, 4);
    return t;
}

inline lsxcommon::TensorRecord F8Rec(const std::string &name,
                                     std::vector<int64_t> shape, uint8_t fill)
{
    lsxcommon::TensorRecord t;
    t.name = name;
    t.dtype = "F8_E4M3";
    t.shape = std::move(shape);
    int64_t n = 1;
    for (int64_t d : t.shape) n *= d;
    t.data.assign(static_cast<size_t>(n), fill);
    return t;
}

inline lsxcommon::TensorRecord E8m0Rec(const std::string &name,
                                       std::vector<int64_t> shape,
                                       uint8_t byte)
{
    lsxcommon::TensorRecord t;
    t.name = name;
    t.dtype = "F8_E8M0";
    t.shape = std::move(shape);
    int64_t n = 1;
    for (int64_t d : t.shape) n *= d;
    t.data.assign(static_cast<size_t>(n), byte);
    return t;
}

inline arrow::Status WritePinned(const std::filesystem::path &path,
                                 const std::vector<lsxcommon::TensorRecord> &t)
{
    auto batch = lsxcommon::PinnedSchema::MakeBatch(t);
    if (!batch.ok()) return batch.status();
    return lsxcommon::WriteRecordBatchToFile(*batch, path);
}

inline arrow::Status BuildSyntheticConverted(const std::filesystem::path &dir,
                                             int hidden = 64, int vocab = 128,
                                             int experts = 2)
{
    std::filesystem::create_directories(dir / "logestix" / "pinned");
    std::filesystem::create_directories(dir / "logestix" / "experts" /
                                        "layer_0");
    if (auto st = BuildSyntheticIndex(dir, /*num_layers=*/1, experts,
                                      "logestix-v41-nvfp4");
        !st.ok())
        return st;

    std::vector<lsxcommon::TensorRecord> g;
    g.push_back(Bf16Rec("embed.weight", {vocab, hidden}, 0x4248));
    g.push_back(Bf16Rec("head.weight", {vocab, hidden}, 0x3555));
    g.push_back(Bf16Rec("norm.weight", {hidden}, 0x3C00));
    if (auto st = WritePinned(dir / "logestix" / "pinned" / "global.arrow",
                              g);
        !st.ok())
        return st;

    const int s = hidden / 32;  // 32x32 tile side
    std::vector<lsxcommon::TensorRecord> v;
    v.push_back(Bf16Rec("layers.0.attn_norm.weight", {hidden}, 0x3C00));
    v.push_back(Bf16Rec("layers.0.ffn_norm.weight", {hidden}, 0x3C00));
    v.push_back(Bf16Rec("layers.0.attn.q_norm.weight", {hidden}, 0x3C00));
    v.push_back(Bf16Rec("layers.0.attn.kv_norm.weight", {hidden}, 0x3C00));
    v.push_back(F32Rec("layers.0.attn.attn_sink", {hidden}, 1.0f));
    for (const char *w : {"attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_a",
                          "attn.wo_b"}) {
        v.push_back(F8Rec(std::string("layers.0.") + w + ".weight",
                          {hidden, hidden}, 0x38));  // E4M3 1.0
        v.push_back(E8m0Rec(std::string("layers.0.") + w + ".scale", {s, s},
                            127));  // 2^0
    }
    v.push_back(Bf16Rec("layers.0.ffn.gate.weight", {experts, hidden},
                        0x3C00));
    v.push_back(F32Rec("layers.0.ffn.gate.bias", {experts}, 0.0f));
    v.push_back(F32Rec("layers.0.ffn.gate.bias_vl", {experts}, 0.0f));
    for (const char *w : {"w1", "w2", "w3"}) {
        v.push_back(F8Rec(std::string("layers.0.ffn.shared_experts.") + w +
                              ".weight",
                          {hidden, hidden}, 0x38));
        v.push_back(E8m0Rec(std::string("layers.0.ffn.shared_experts.") + w +
                                ".scale",
                            {s, s}, 127));
    }
    v.push_back(F32Rec("layers.0.hc_attn_fn", {24, 4 * hidden}, 1.0f));
    v.push_back(F32Rec("layers.0.hc_attn_base", {24}, 0.5f));
    v.push_back(F32Rec("layers.0.hc_attn_scale", {3}, 2.0f));
    v.push_back(F32Rec("layers.0.hc_ffn_fn", {24, 4 * hidden}, 1.0f));
    v.push_back(F32Rec("layers.0.hc_ffn_base", {24}, 0.5f));
    v.push_back(F32Rec("layers.0.hc_ffn_scale", {3}, 2.0f));
    if (auto st = WritePinned(dir / "logestix" / "pinned" / "layer_000.arrow",
                              v);
        !st.ok())
        return st;

    for (int e = 0; e < experts; ++e) {
        lsxcommon::PackedExpert pe;
        pe.w1_packed.assign(64, static_cast<uint8_t>(0xB0 + e));
        pe.w2_packed.assign(64, 0xC1);
        pe.w3_packed.assign(64, 0xC2);
        pe.w1_scales.assign(8, 0x20);
        pe.w2_scales.assign(8, 0x20);
        pe.w3_scales.assign(8, 0x20);
        pe.w1_global_scale = 1.0f;
        pe.w2_global_scale = 1.0f;
        pe.w3_global_scale = 1.0f;
        auto batch = lsxcommon::ExpertSchema::MakeBatch(pe);
        if (!batch.ok()) return batch.status();
        if (auto st = lsxcommon::WriteRecordBatchToFile(
                *batch, dir / "logestix" / "experts" / "layer_0" /
                            ("exp_" + std::to_string(e) + ".arrow"));
            !st.ok())
            return st;
    }
    return arrow::Status::OK();
}

} // namespace v41fixtest
