#pragma once

#include <arrow/api.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <lsxcommon/schema.h>

// Synthetic minimal DeepSeek-V4-Flash converted-model layout for loader tests:
//   <model_dir>/logestix/model.index.json     (topology + vram block)
//   <model_dir>/logestix/pinned/global.arrow  (embed/head/norm + HC head F32)
//   <model_dir>/logestix/pinned/layer_<n>.arrow (pure-MLA layer tensor set)
// No shared_experts.arrow: that file stays optional-if-absent by design.
namespace dsfixtest {

constexpr int kVocab = 128;
constexpr int kHidden = 256;
constexpr int kNumLayers = 3;
constexpr int kNumRouted = 8;

inline void AppendU16(std::vector<uint8_t> &data, uint16_t v)
{
    const auto off = data.size();
    data.resize(off + 2);
    std::memcpy(data.data() + off, &v, 2);
}

inline lsxcommon::TensorRecord Bf16Record(const std::string &name,
                                          std::vector<int64_t> shape,
                                          uint16_t fill_bits)
{
    lsxcommon::TensorRecord t;
    t.name = name;
    t.dtype = "BF16";
    t.shape = std::move(shape);
    int64_t numel = 1;
    for (int64_t d : t.shape) numel *= d;
    t.data.reserve(static_cast<size_t>(numel) * 2);
    for (int64_t i = 0; i < numel; ++i) AppendU16(t.data, fill_bits);
    return t;
}

inline lsxcommon::TensorRecord F32Record(const std::string &name,
                                         std::vector<int64_t> shape,
                                         float fill)
{
    lsxcommon::TensorRecord t;
    t.name = name;
    t.dtype = "F32";
    t.shape = std::move(shape);
    int64_t numel = 1;
    for (int64_t d : t.shape) numel *= d;
    t.data.resize(static_cast<size_t>(numel) * 4);
    for (size_t off = 0; off < t.data.size(); off += 4)
        std::memcpy(t.data.data() + off, &fill, 4);
    return t;
}

inline lsxcommon::TensorRecord ByteRecord(const std::string &name,
                                          std::vector<int64_t> shape,
                                          uint8_t fill,
                                          const char *dtype = "F8_E4M3")
{
    lsxcommon::TensorRecord t;
    t.name = name;
    t.dtype = dtype;
    t.shape = std::move(shape);
    int64_t numel = 1;
    for (int64_t d : t.shape) numel *= d;
    t.data.resize(static_cast<size_t>(numel), fill);
    return t;
}

inline void WriteIndexJson(const std::filesystem::path &model_dir)
{
    std::filesystem::create_directories(model_dir / "logestix");
    std::ofstream f(model_dir / "logestix" / "model.index.json");
    f << "{\n"
      << "  \"model\": \"DeepSeek-V4-Flash\",\n"
      << "  \"num_layers\": " << kNumLayers << ",\n"
      << "  \"experts_per_layer\": 2,\n"
      << "  \"total_experts\": " << (kNumLayers * 2) << ",\n"
      << "  \"vram\": {\n"
      << "    \"min_bytes\": 1024,\n"
      << "    \"optimal_bytes\": 2048,\n"
      << "    \"max_bytes\": 4096,\n"
      << "    \"context_tokens\": 2048\n"
      << "  }\n"
      << "}\n";
}

inline arrow::Status WritePinnedFile(
    const std::filesystem::path &path,
    const std::vector<lsxcommon::TensorRecord> &tensors)
{
    auto batch = lsxcommon::PinnedSchema::MakeBatch(tensors);
    if (!batch.ok()) return batch.status();
    return lsxcommon::WriteRecordBatchToFile(*batch, path);
}

inline std::vector<lsxcommon::TensorRecord> GlobalRecords()
{
    std::vector<lsxcommon::TensorRecord> g;
    g.push_back(Bf16Record("model.embed.weight", {kVocab, kHidden}, 0x4248));
    g.push_back(Bf16Record("lm_head.weight", {kVocab, kHidden}, 0x3555));
    g.push_back(Bf16Record("model.norm.weight", {kHidden}, 0x3C00));
    g.push_back(F32Record("hc_head_fn", {4, kHidden * 4}, 1.0f));
    g.push_back(F32Record("hc_head_base", {4}, 0.5f));
    g.push_back(F32Record("hc_head_scale", {1}, 2.0f));
    return g;
}

inline std::vector<lsxcommon::TensorRecord> LayerRecords(int l)
{
    const std::string p = "layers." + std::to_string(l) + ".";
    std::vector<lsxcommon::TensorRecord> v;
    v.push_back(Bf16Record(p + "ffn_norm.weight", {kHidden}, 0x3C00));
    v.push_back(Bf16Record(p + "ffn.gate.weight", {kNumRouted, kHidden},
                           0x4248));
    v.push_back(F32Record(p + "ffn.gate.bias", {kNumRouted}, 0.25f));
    v.push_back(ByteRecord(p + "attn.wq_a.weight", {512, kHidden}, 0xAA));
    v.push_back(ByteRecord(p + "attn.wq_a.scale", {4, 4}, 0x38,
                           "F8_E8M0"));
    v.push_back(Bf16Record(p + "attn.q_norm.weight", {64}, 0x3C00));
    v.push_back(Bf16Record(p + "attn.kv_norm.weight", {32}, 0x3C00));
    v.push_back(Bf16Record(p + "attn_norm.weight", {kHidden}, 0x3C00));
    v.push_back(F32Record(p + "attn.attn_sink", {8}, 1.0f));
    v.push_back(F32Record(p + "hc_attn_fn", {24, kHidden * 4}, 1.0f));
    v.push_back(F32Record(p + "hc_attn_base", {24}, 0.5f));
    v.push_back(F32Record(p + "hc_attn_scale", {3}, 2.0f));
    v.push_back(F32Record(p + "hc_ffn_fn", {24, kHidden * 4}, 1.0f));
    v.push_back(F32Record(p + "hc_ffn_base", {24}, 0.5f));
    v.push_back(F32Record(p + "hc_ffn_scale", {3}, 2.0f));
    return v;
}

inline arrow::Status WriteGlobalArrow(
    const std::filesystem::path &model_dir,
    const std::vector<lsxcommon::TensorRecord> &records)
{
    return WritePinnedFile(model_dir / "logestix" / "pinned" /
                               "global.arrow",
                           records);
}

inline arrow::Status BuildSyntheticModel(const std::filesystem::path &model_dir)
{
    const auto pinned = model_dir / "logestix" / "pinned";
    std::filesystem::create_directories(pinned);
    WriteIndexJson(model_dir);
    if (auto st = WritePinnedFile(pinned / "global.arrow", GlobalRecords());
        !st.ok())
        return st;
    for (int l = 0; l < kNumLayers; ++l) {
        auto fn = pinned / ("layer_" + std::to_string(l) + ".arrow");
        if (auto st = WritePinnedFile(fn, LayerRecords(l)); !st.ok())
            return st;
    }
    return arrow::Status::OK();
}

} // namespace dsfixtest
