#pragma once

#include <arrow/api.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <lsxcommon/schema.h>

#include "deepseek_moe_pro_constants.h"

// Synthetic minimal DeepSeek-V4-Pro-0813 converted-model layout for loader tests:
//   <model_dir>/logestix/model.index.json     (topology + vram block)
//   <model_dir>/logestix/pinned/global.arrow  (embed/head/norm + HC head F32)
//   <model_dir>/logestix/pinned/layer_<n>.arrow (pure-MLA layer tensor set)
// No shared_experts.arrow: that file stays optional-if-absent by design.
namespace dsprofixtest {

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
    std::filesystem::create_directories(model_dir);
    std::ofstream f(model_dir / "model.index.json");
    f << "{\n"
      << "  \"format\": \"logestix-v1-nvfp4\",\n"
      << "  \"model\": \"DeepSeek-V4-Pro-0813\",\n"
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
    if (l < lsxcommon::deepseek_pro::kHashLayers) {
        lsxcommon::TensorRecord tid;
        tid.name = p + "ffn.gate.tid2eid";
        tid.dtype = "I64";
        tid.shape = {kVocab, 6};
        tid.data.assign(static_cast<size_t>(kVocab) * 6 * sizeof(int64_t), 0);
        v.push_back(std::move(tid));
    } else {
        v.push_back(F32Record(p + "ffn.gate.bias", {kNumRouted}, 0.25f));
    }
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

    const int ratio = lsxcommon::deepseek_pro::CompressRatio(l);
    const int coff = lsxcommon::deepseek_pro::CompressorCoff(l);
    const int coff_dim = coff * 512;
    v.push_back(Bf16Record(p + "attn.compressor.wkv.weight",
                           {coff_dim, kHidden}, 0x3C00));
    v.push_back(Bf16Record(p + "attn.compressor.wgate.weight",
                           {coff_dim, kHidden}, 0x3C00));
    v.push_back(F32Record(p + "attn.compressor.ape", {ratio, coff_dim}, 0.5f));
    v.push_back(Bf16Record(p + "attn.compressor.norm.weight", {512}, 0x3C00));

    if (lsxcommon::deepseek_pro::HasIndexer(l)) {
        v.push_back(ByteRecord(p + "attn.indexer.wq_b.weight", {128, 64},
                               0xAA));
        v.push_back(ByteRecord(p + "attn.indexer.wq_b.scale", {1, 1}, 0x38,
                               "F8_E8M0"));
        v.push_back(Bf16Record(p + "attn.indexer.weights_proj.weight",
                               {64, kHidden}, 0x3C00));
        v.push_back(Bf16Record(p + "attn.indexer.compressor.wkv.weight",
                               {256, kHidden}, 0x3C00));
        v.push_back(Bf16Record(p + "attn.indexer.compressor.wgate.weight",
                               {256, kHidden}, 0x3C00));
        v.push_back(F32Record(p + "attn.indexer.compressor.ape", {4, 256},
                              0.5f));
        v.push_back(Bf16Record(p + "attn.indexer.compressor.norm.weight",
                               {128}, 0x3C00));
    }
    return v;
}

inline arrow::Status WriteGlobalArrow(
    const std::filesystem::path &model_dir,
    const std::vector<lsxcommon::TensorRecord> &records)
{
    return WritePinnedFile(model_dir / "pinned" /
                               "global.arrow",
                           records);
}

inline lsxcommon::PackedExpert SyntheticExpert(uint8_t seed)
{
    lsxcommon::PackedExpert e;
    e.w1_packed.assign(64, seed);
    e.w2_packed.assign(64, static_cast<uint8_t>(seed + 1));
    e.w3_packed.assign(64, static_cast<uint8_t>(seed + 2));
    e.w1_scales = {0x38};
    e.w2_scales = {0x38};
    e.w3_scales = {0x38};
    e.w1_global_scale = 1.0f;
    e.w2_global_scale = 1.0f;
    e.w3_global_scale = 1.0f;
    return e;
}

inline arrow::Status BuildSyntheticModel(const std::filesystem::path &model_dir)
{
    const auto pinned = model_dir / "pinned";
    std::filesystem::create_directories(pinned);
    WriteIndexJson(model_dir);
    if (auto st = WritePinnedFile(pinned / "global.arrow", GlobalRecords());
        !st.ok())
        return st;
    for (int l = 0; l < kNumLayers; ++l) {
        auto fn = pinned / ("layer_" + std::to_string(l) + ".arrow");
        if (auto st = WritePinnedFile(fn, LayerRecords(l)); !st.ok())
            return st;
        const auto edir = model_dir / "experts" /
                          ("layer_" + std::to_string(l));
        std::filesystem::create_directories(edir);
        for (int e = 0; e < 2; ++e) {
            auto batch = lsxcommon::ExpertSchema::MakeBatch(
                SyntheticExpert(static_cast<uint8_t>(l * 2 + e)));
            if (!batch.ok()) return batch.status();
            if (auto st = lsxcommon::WriteRecordBatchToFile(
                    *batch, edir / ("exp_" + std::to_string(e) + ".arrow"));
                !st.ok())
                return st;
        }
    }
    return arrow::Status::OK();
}

} // namespace dsprofixtest
