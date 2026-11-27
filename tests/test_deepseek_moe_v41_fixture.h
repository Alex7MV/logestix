#pragma once

// Synthetic DeepSeek-V4.1-Flash index fixture (host-only). P1 needs only the
// index JSON; P2/P3 extend this header with pinned/ and expert Arrow writers.
// Family-neutral helpers (FakeTensor/WriteSafetensors/MakeBf16*) come from
// test_synth_checkpoint.h when P2 lands — never from another family's fixture.
#include <filesystem>
#include <fstream>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include <arrow/status.h>

#include <lsxcommon/schema.h>

namespace v41fixtest {

inline void WriteIndex(const std::filesystem::path &model_dir,
                       const std::string &json)
{
    std::filesystem::create_directories(model_dir);
    std::ofstream f(model_dir / "model.index.json");
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
    std::filesystem::create_directories(dir / "pinned");
    std::filesystem::create_directories(dir / "experts" /
                                        "layer_0");
    if (auto st = BuildSyntheticIndex(dir, /*num_layers=*/1, experts,
                                      "logestix-v41-nvfp4");
        !st.ok())
        return st;

    std::vector<lsxcommon::TensorRecord> g;
    g.push_back(Bf16Rec("embed.weight", {vocab, hidden}, 0x4248));
    g.push_back(Bf16Rec("head.weight", {vocab, hidden}, 0x3555));
    g.push_back(Bf16Rec("norm.weight", {hidden}, 0x3C00));
    if (auto st = WritePinned(dir / "pinned" / "global.arrow",
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
    if (auto st = WritePinned(dir / "pinned" / "layer_000.arrow",
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
        const float gs = std::ldexp(1.0f, -14);
        pe.w1_global_scale = gs;
        pe.w2_global_scale = gs;
        pe.w3_global_scale = gs;
        auto batch = lsxcommon::ExpertSchema::MakeBatch(pe);
        if (!batch.ok()) return batch.status();
        if (auto st = lsxcommon::WriteRecordBatchToFile(
                *batch, dir / "experts" / "layer_0" /
                            ("exp_" + std::to_string(e) + ".arrow"));
            !st.ok())
            return st;
    }
    return arrow::Status::OK();
}

// ── Full-geometry synthetic converted model (one ratio-0 layer) ─────────────
// The frozen V4.1 shapes (hidden 5120, 64 heads x 512, q_lora 1280, o_groups 8
// x 1024, MoE inter 2304) so the production pipeline can actually run on the
// synthetic path. All weights are constant fills (finite, non-degenerate);
// this validates the execution graph, not the model numerics.
inline arrow::Status BuildSyntheticFullGeometry(
    const std::filesystem::path &dir, int vocab = 64, int experts = 2)
{
    const int hidden = 5120, q_lora = 1280, head_dim = 512;
    const int heads = 64, o_groups = 8, o_lora = 1024, inter = 2304;
    const int q_dim = heads * head_dim, oa_rows = o_groups * o_lora,
              oa_cols = q_dim / o_groups;

    std::filesystem::create_directories(dir / "pinned");
    std::filesystem::create_directories(dir / "experts" /
                                        "layer_0");

    std::string json = "{";
    json += "\"format\": \"logestix-v41-nvfp4\",";
    json += "\"model\": \"DeepSeek-V4.1-Flash\",";
    json += "\"num_layers\": 1,";
    json += "\"experts_per_layer\": " + std::to_string(experts) + ",";
    json += "\"total_experts\": " + std::to_string(experts) + ",";
    json += "\"vram\": {\"min_bytes\": 1, \"optimal_bytes\": 2, "
            "\"max_bytes\": 3, \"context_tokens\": 4096},";
    json += "\"layers\": [{\"compress_ratio\": 0, \"is_kv_source\": false, "
            "\"is_index_source\": false, \"is_candidate_source\": false}]";
    json += "}";
    WriteIndex(dir, json);

    std::vector<lsxcommon::TensorRecord> g;
    g.push_back(Bf16Rec("embed.weight", {vocab, hidden}, 0x3C00));
    g.push_back(Bf16Rec("head.weight", {vocab, hidden}, 0x3C00));
    g.push_back(Bf16Rec("norm.weight", {hidden}, 0x3C00));
    if (auto st = WritePinned(dir / "pinned" / "global.arrow", g);
        !st.ok())
        return st;

    auto fp8v = [&](std::vector<lsxcommon::TensorRecord> &v, const char *name,
                    int rows, int cols) {
        v.push_back(F8Rec(std::string("layers.0.") + name + ".weight",
                          {rows, cols}, 0x03));
        v.push_back(E8m0Rec(std::string("layers.0.") + name + ".scale",
                            {(rows + 31) / 32, (cols + 31) / 32}, 127));
    };
    std::vector<lsxcommon::TensorRecord> v;
    v.push_back(Bf16Rec("layers.0.attn_norm.weight", {hidden}, 0x3C00));
    v.push_back(Bf16Rec("layers.0.ffn_norm.weight", {hidden}, 0x3C00));
    v.push_back(Bf16Rec("layers.0.attn.q_norm.weight", {q_lora}, 0x3C00));
    v.push_back(Bf16Rec("layers.0.attn.kv_norm.weight", {head_dim}, 0x3C00));
    v.push_back(F32Rec("layers.0.attn.attn_sink", {heads}, 0.0f));
    fp8v(v, "attn.wq_a", q_lora, hidden);
    fp8v(v, "attn.wq_b", q_dim, q_lora);
    fp8v(v, "attn.wkv", head_dim, hidden);
    fp8v(v, "attn.wo_a", oa_rows, oa_cols);
    fp8v(v, "attn.wo_b", hidden, oa_rows);
    v.push_back(Bf16Rec("layers.0.ffn.gate.weight", {experts, hidden},
                        0x3C00));
    v.push_back(F32Rec("layers.0.ffn.gate.bias", {experts}, 0.0f));
    v.push_back(F32Rec("layers.0.ffn.gate.bias_vl", {experts}, 0.0f));
    fp8v(v, "ffn.shared_experts.w1", inter, hidden);
    fp8v(v, "ffn.shared_experts.w2", hidden, inter);
    fp8v(v, "ffn.shared_experts.w3", inter, hidden);
    v.push_back(F32Rec("layers.0.hc_attn_fn", {24, 4 * hidden}, 0.0f));
    v.push_back(F32Rec("layers.0.hc_attn_base", {24}, 0.0f));
    v.push_back(F32Rec("layers.0.hc_attn_scale", {3}, 1.0f));
    v.push_back(F32Rec("layers.0.hc_ffn_fn", {24, 4 * hidden}, 0.0f));
    v.push_back(F32Rec("layers.0.hc_ffn_base", {24}, 0.0f));
    v.push_back(F32Rec("layers.0.hc_ffn_scale", {3}, 1.0f));
    if (auto st = WritePinned(dir / "pinned" / "layer_000.arrow",
                              v);
        !st.ok())
        return st;

    const size_t packed = static_cast<size_t>(inter) * hidden / 2;
    const size_t scales = static_cast<size_t>(inter) * hidden / 16;
    for (int e = 0; e < experts; ++e) {
        lsxcommon::PackedExpert pe;
        pe.w1_packed.assign(packed, 0x11);
        pe.w2_packed.assign(packed, 0x22);
        pe.w3_packed.assign(packed, 0x33);
        pe.w1_scales.assign(scales, 0x38);
        pe.w2_scales.assign(scales, 0x38);
        pe.w3_scales.assign(scales, 0x38);
        const float gs = std::ldexp(1.0f, -14);
        pe.w1_global_scale = gs;
        pe.w2_global_scale = gs;
        pe.w3_global_scale = gs;
        auto batch = lsxcommon::ExpertSchema::MakeBatch(pe);
        if (!batch.ok()) return batch.status();
        if (auto st = lsxcommon::WriteRecordBatchToFile(
                *batch, dir / "experts" / "layer_0" /
                            ("exp_" + std::to_string(e) + ".arrow"));
            !st.ok())
            return st;
    }
    return arrow::Status::OK();
}

// Two-layer geometry: layer 0 is pure SWA, layer 1 is a ratio-1 kv/index
// source (exercises the CED compressor + indexer + top-k + concat path).
inline arrow::Status BuildSyntheticCedGeometry(
    const std::filesystem::path &dir, int vocab = 64, int experts = 2)
{
    const int hidden = 5120, q_lora = 1280, head_dim = 512;
    const int heads = 64, o_groups = 8, o_lora = 1024, inter = 2304;
    const int q_dim = heads * head_dim, oa_rows = o_groups * o_lora,
              oa_cols = q_dim / o_groups;
    const int idx_heads = 32, idx_dim = 128, idx_qdim = idx_heads * idx_dim;
    const int idx_wkrow = idx_dim;

    std::filesystem::create_directories(dir / "pinned");
    for (int l = 0; l < 2; ++l)
        std::filesystem::create_directories(
            dir / "experts" / ("layer_" + std::to_string(l)));

    std::string json = "{";
    json += "\"format\": \"logestix-v41-nvfp4\",";
    json += "\"model\": \"DeepSeek-V4.1-Flash\",";
    json += "\"num_layers\": 2,";
    json += "\"experts_per_layer\": " + std::to_string(experts) + ",";
    json += "\"total_experts\": " + std::to_string(2 * experts) + ",";
    json += "\"vram\": {\"min_bytes\": 1, \"optimal_bytes\": 2, "
            "\"max_bytes\": 3, \"context_tokens\": 4096},";
    json += "\"layers\": [";
    json += "{\"compress_ratio\": 0, \"is_kv_source\": false, "
            "\"is_index_source\": false, \"is_candidate_source\": false},";
    json += "{\"compress_ratio\": 1, \"is_kv_source\": true, "
            "\"is_index_source\": true, \"is_candidate_source\": false}";
    json += "]}";
    WriteIndex(dir, json);

    std::vector<lsxcommon::TensorRecord> g;
    g.push_back(Bf16Rec("embed.weight", {vocab, hidden}, 0x3C00));
    g.push_back(Bf16Rec("head.weight", {vocab, hidden}, 0x3C00));
    g.push_back(Bf16Rec("norm.weight", {hidden}, 0x3C00));
    if (auto st = WritePinned(dir / "pinned" / "global.arrow", g);
        !st.ok())
        return st;

    auto fp8v = [&](std::vector<lsxcommon::TensorRecord> &v,
                    const std::string &pfx, const char *name, int rows,
                    int cols) {
        v.push_back(F8Rec(pfx + name + ".weight", {rows, cols}, 0x03));
        v.push_back(E8m0Rec(pfx + name + ".scale",
                            {(rows + 31) / 32, (cols + 31) / 32}, 127));
    };

    for (int l = 0; l < 2; ++l) {
        const bool ced = (l == 1);
        const std::string pfx = "layers." + std::to_string(l) + ".";
        std::vector<lsxcommon::TensorRecord> v;
        v.push_back(Bf16Rec(pfx + "attn_norm.weight", {hidden}, 0x3C00));
        v.push_back(Bf16Rec(pfx + "ffn_norm.weight", {hidden}, 0x3C00));
        v.push_back(Bf16Rec(pfx + "attn.q_norm.weight", {q_lora}, 0x3C00));
        v.push_back(Bf16Rec(pfx + "attn.kv_norm.weight", {head_dim}, 0x3C00));
        v.push_back(F32Rec(pfx + "attn.attn_sink", {heads}, 0.0f));
        fp8v(v, pfx, "attn.wq_a", q_lora, hidden);
        fp8v(v, pfx, "attn.wq_b", q_dim, q_lora);
        fp8v(v, pfx, "attn.wkv", head_dim, hidden);
        fp8v(v, pfx, "attn.wo_a", oa_rows, oa_cols);
        fp8v(v, pfx, "attn.wo_b", hidden, oa_rows);
        v.push_back(Bf16Rec(pfx + "ffn.gate.weight", {experts, hidden},
                            0x3C00));
        v.push_back(F32Rec(pfx + "ffn.gate.bias", {experts}, 0.0f));
        v.push_back(F32Rec(pfx + "ffn.gate.bias_vl", {experts}, 0.0f));
        fp8v(v, pfx, "ffn.shared_experts.w1", inter, hidden);
        fp8v(v, pfx, "ffn.shared_experts.w2", hidden, inter);
        fp8v(v, pfx, "ffn.shared_experts.w3", inter, hidden);
        v.push_back(F32Rec(pfx + "hc_attn_fn", {24, 4 * hidden}, 0.0f));
        v.push_back(F32Rec(pfx + "hc_attn_base", {24}, 0.0f));
        v.push_back(F32Rec(pfx + "hc_attn_scale", {3}, 1.0f));
        v.push_back(F32Rec(pfx + "hc_ffn_fn", {24, 4 * hidden}, 0.0f));
        v.push_back(F32Rec(pfx + "hc_ffn_base", {24}, 0.0f));
        v.push_back(F32Rec(pfx + "hc_ffn_scale", {3}, 1.0f));
        if (ced) {
            v.push_back(Bf16Rec(pfx + "attn.compressor.wkv.weight",
                                {head_dim, hidden}, 0x3C00));
            v.push_back(Bf16Rec(pfx + "attn.compressor.norm.weight",
                                {head_dim}, 0x3C00));
            fp8v(v, pfx, "attn.indexer.wq_b", idx_qdim, q_lora);
            v.push_back(Bf16Rec(pfx + "attn.indexer.weights_proj.weight",
                                {idx_heads, hidden}, 0x3C00));
            v.push_back(Bf16Rec(pfx + "attn.indexer.wk.weight",
                                {idx_wkrow, head_dim}, 0x3C00));
            v.push_back(Bf16Rec(pfx + "attn.indexer.k_norm.weight", {idx_dim},
                                0x3C00));
        }
        const std::string lname = "layer_00" + std::to_string(l);
        if (auto st = WritePinned(dir / "pinned" /
                                      (lname + ".arrow"),
                                  v);
            !st.ok())
            return st;

        const size_t packed = static_cast<size_t>(inter) * hidden / 2;
        const size_t scales = static_cast<size_t>(inter) * hidden / 16;
        const float gs = std::ldexp(1.0f, -14);
        for (int e = 0; e < experts; ++e) {
            lsxcommon::PackedExpert pe;
            pe.w1_packed.assign(packed, 0x11);
            pe.w2_packed.assign(packed, 0x22);
            pe.w3_packed.assign(packed, 0x33);
            pe.w1_scales.assign(scales, 0x38);
            pe.w2_scales.assign(scales, 0x38);
            pe.w3_scales.assign(scales, 0x38);
            pe.w1_global_scale = gs;
            pe.w2_global_scale = gs;
            pe.w3_global_scale = gs;
            auto batch = lsxcommon::ExpertSchema::MakeBatch(pe);
            if (!batch.ok()) return batch.status();
            if (auto st = lsxcommon::WriteRecordBatchToFile(
                    *batch, dir / "experts" /
                                ("layer_" + std::to_string(l)) /
                                ("exp_" + std::to_string(e) + ".arrow"));
                !st.ok())
                return st;
        }
    }
    return arrow::Status::OK();
}

} // namespace v41fixtest
