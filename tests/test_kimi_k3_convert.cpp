// Kimi-K3 synthetic end-to-end converter test. Runs the full converter on a
// tiny legal fixture (LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1) and asserts the
// logestix-kimi-k3-nvfp4 layout: hybrid layer tables, the pin/stream split,
// the stripped+counted vision ballast, and the exact MXFP4 -> NVFP4 expert
// repack (nibbles verbatim, E8M0/32 -> two identical pow2-E4M3/16 scales with
// a per-tensor power-of-two global scale).
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/kimi_moe_k3_convert.h>
#include <lsxcommon/schema.h>
#include <simdjson.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "test_kimi_k3_fixture.h"

namespace fs = std::filesystem;
using namespace kimik3test;

namespace {

std::string ReadFile(const fs::path &p)
{
    std::ifstream f(p);
    return std::string((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
}

bool PinnedHasName(const fs::path &path, const std::string &name)
{
    if (!fs::exists(path)) return false;
    auto batch = lsxcommon::ReadRecordBatchFromFile(path);
    if (!batch.ok()) return false;
    auto rows = lsxcommon::PinnedSchema::UnpackBatch(*batch);
    if (!rows.ok()) return false;
    for (const auto &r : *rows)
        if (r.name == name) return true;
    return false;
}

bool DenseHasName(const fs::path &path, const std::string &name)
{
    if (!fs::exists(path)) return false;
    auto batch = lsxcommon::ReadRecordBatchFromFile(path);
    if (!batch.ok()) return false;
    auto rows = lsxcommon::DenseNvfp4Schema::UnpackBatch(*batch);
    if (!rows.ok()) return false;
    for (const auto &r : *rows)
        if (r.name == name) return true;
    return false;
}

// Host-side E2M1 nibble table (matches the neutral codec).
float E2m1(uint8_t nib)
{
    static const float kTab[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = kTab[nib & 0x7];
    return (nib & 0x8) ? -v : v;
}

// E4M3FN block-scale decode (bit-compatible with nvfp4_scale.h).
float E4m3(uint8_t b)
{
    const int e = (b >> 3) & 0xF;
    const int m = b & 0x7;
    if (e == 0) return static_cast<float>(m) * (1.0f / 512.0f);
    return std::ldexp(1.0f + static_cast<float>(m) / 8.0f, e - 7);
}

}  // namespace

TEST_CASE("kimi-k3 synthetic convert emits logestix-kimi-k3-nvfp4",
          "[kimi-k3][convert]")
{
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    const fs::path in = fs::temp_directory_path() / "kimi_k3_synth_in";
    const fs::path out = fs::temp_directory_path() / "kimi_k3_synth_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    WriteCheckpoint(in);

    auto st = lsxcommon::kimi_k3::ConvertKimiK3Model(in.string(), out.string());
    if (!st.ok()) FAIL(st.ToString());
    REQUIRE(st.ok());

    const fs::path base = out;
    const fs::path idx = base / "model.index.json";
    REQUIRE(fs::exists(idx));

    const std::string json = ReadFile(idx);
    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json);
    auto doc = parser.iterate(pad);

    std::string_view fmt;
    REQUIRE(doc["format"].get_string().get(fmt) == simdjson::SUCCESS);
    REQUIRE(fmt == "logestix-kimi-k3-nvfp4");

    int64_t layers = 0;
    REQUIRE(doc["num_layers"].get_int64().get(layers) == simdjson::SUCCESS);
    REQUIRE(layers == kLayers);

    std::vector<int64_t> mla_layers, kda_layers;
    for (auto v : doc["mla_layers"].get_array())
        mla_layers.push_back(v.get_int64().value());
    for (auto v : doc["kda_layers"].get_array())
        kda_layers.push_back(v.get_int64().value());
    REQUIRE(mla_layers == std::vector<int64_t>{2, 3});
    REQUIRE(kda_layers == std::vector<int64_t>{0, 1});

    int64_t skipped_vis = -1, skipped_proj = -1, experts_per_layer = -1;
    REQUIRE(doc["skipped_visual_tensors"].get_int64().get(skipped_vis) ==
            simdjson::SUCCESS);
    REQUIRE(doc["skipped_projector_tensors"].get_int64().get(skipped_proj) ==
            simdjson::SUCCESS);
    REQUIRE(doc["experts_per_layer"].get_int64().get(experts_per_layer) ==
            simdjson::SUCCESS);
    REQUIRE(skipped_vis == 1);
    REQUIRE(skipped_proj == 1);
    REQUIRE(experts_per_layer == kExperts);

    // KDA layer 0: attention projections are pinned NVFP4, gate/beta/conv raw.
    REQUIRE(DenseHasName(base / "pinned" / "layer_0_nvfp4.arrow",
                         "language_model.model.layers.0.self_attn.q_proj.weight"));
    REQUIRE(DenseHasName(base / "pinned" / "layer_0_nvfp4.arrow",
                         "language_model.model.layers.0.mlp.gate_proj.weight"));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_0.arrow",
                          "language_model.model.layers.0.self_attn.q_conv1d.weight"));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_0.arrow",
                          "language_model.model.layers.0.self_attn.A_log"));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_0.arrow",
                          "language_model.model.layers.0.self_attn.b_proj.weight"));

    // MLA layer 2: q/kv projections pinned; the latent/shared MoE plane is
    // streamed, never pinned.
    REQUIRE(DenseHasName(base / "pinned" / "layer_2_nvfp4.arrow",
                         "language_model.model.layers.2.self_attn.q_a_proj.weight"));
    REQUIRE(DenseHasName(base / "pinned" / "layer_2_nvfp4.arrow",
                         "language_model.model.layers.2.self_attn.kv_b_proj.weight"));
    REQUIRE(DenseHasName(base / "streamed" / "layer_2.arrow",
                         "language_model.model.layers.2.block_sparse_moe."
                         "routed_expert_down_proj.weight"));
    REQUIRE(DenseHasName(base / "streamed" / "layer_2.arrow",
                         "language_model.model.layers.2.block_sparse_moe."
                         "shared_experts.gate_proj.weight"));
    REQUIRE_FALSE(DenseHasName(base / "pinned" / "layer_2_nvfp4.arrow",
                               "language_model.model.layers.2.block_sparse_moe."
                               "shared_experts.gate_proj.weight"));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_2.arrow",
                          "language_model.model.layers.2.block_sparse_moe.gate.weight"));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_2.arrow",
                          "language_model.model.layers.2.block_sparse_moe."
                          "routed_expert_norm.weight"));

    // Embed / lm_head are pinned NVFP4; the final norm stays raw.
    REQUIRE(DenseHasName(base / "pinned" / "global_nvfp4.arrow",
                         "language_model.model.embed_tokens.weight"));
    REQUIRE(DenseHasName(base / "pinned" / "global_nvfp4.arrow",
                         "language_model.lm_head.weight"));
    REQUIRE(PinnedHasName(base / "pinned" / "global.arrow",
                          "language_model.model.norm.weight"));

    // Ballast is stripped: no vision/mm tensor anywhere in the tree.
    REQUIRE_FALSE(PinnedHasName(base / "pinned" / "global.arrow",
                                "vision_tower.encoder.blocks.0.wo.weight"));
    REQUIRE_FALSE(DenseHasName(base / "pinned" / "global_nvfp4.arrow",
                               "mm_projector.proj.0.weight"));

    // Expert files exist for every MoE layer and every expert; layer 0 (dense)
    // has none.
    for (int l = 0; l < kLayers; ++l) {
        const fs::path dir = base / "experts" / ("layer_" + std::to_string(l));
        if (l < kDenseLayers) {
            REQUIRE_FALSE(fs::exists(dir / "exp_0.arrow"));
            continue;
        }
        for (int e = 0; e < kExperts; ++e)
            REQUIRE(fs::exists(dir / ("exp_" + std::to_string(e) + ".arrow")));
    }

    // ── Exact MXFP4 -> NVFP4 expert repack ──
    auto batch = lsxcommon::ReadRecordBatchFromFile(
        base / "experts" / "layer_1" / "exp_2.arrow");
    REQUIRE(batch.ok());
    auto pe_res = lsxcommon::ExpertSchema::UnpackBatch(*batch);
    if (!pe_res.ok()) FAIL(pe_res.status().ToString());
    const lsxcommon::PackedExpert &pe = *pe_res;

    // Nibbles are verbatim; the per-16 scale count is exactly 2x the source
    // per-32 count.
    REQUIRE(pe.w1_packed.size() ==
            static_cast<size_t>(kMoeInter * (kLatent / 2)));
    REQUIRE(pe.w1_scales.size() ==
            static_cast<size_t>(kMoeInter * (kLatent / 32) * 2));
    // (The neutral ExpertSchema stores no shapes; the loader derives them from
    // the frozen passport.)

    // Decode both sides at a few offsets: E2M1(nib) * 2^(e8m0-127) must equal
    // E2M1(nib) * E4M3(scale/2) * global_scale.
    const auto src_packed = MakeMxfp4Packed(kMoeInter, kLatent);
    const auto src_scales = MakeE8m0Scales(kMoeInter, kLatent / 32);
    REQUIRE(pe.w1_packed == src_packed);
    int checked = 0;
    for (int64_t r = 0; r < kMoeInter; r += 7) {
        for (int64_t c = 0; c < kLatent; c += 6) {
            const uint8_t byte = pe.w1_packed[static_cast<size_t>(r * (kLatent / 2) + c / 2)];
            const uint8_t nib = (c % 2) ? static_cast<uint8_t>(byte >> 4)
                                        : static_cast<uint8_t>(byte & 0xF);
            const float src = E2m1(nib) *
                std::ldexp(1.0f, int(src_scales[static_cast<size_t>(
                                      r * (kLatent / 32) + c / 32)]) - 127);
            const float dst = E2m1(nib) *
                E4m3(pe.w1_scales[static_cast<size_t>(r * (kLatent / 16) + c / 16)]) *
                pe.w1_global_scale;
            INFO("r=" << r << " c=" << c << " src=" << src << " dst=" << dst);
            REQUIRE(std::fabs(src - dst) <= 1e-6f * std::fabs(src) + 1e-9f);
            ++checked;
        }
    }
    REQUIRE(checked > 0);
}
