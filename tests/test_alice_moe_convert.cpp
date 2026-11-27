// AliceAI synthetic end-to-end converter test. Runs the full converter on a
// tiny legal fixture (LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1) and asserts the
// logestix-alice-nvfp4 layout, expert count, and the layer-0 AttnRes absence.
#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/index_layout.h"
#include <lsxcommon/alice_moe_convert.h>
#include <lsxcommon/schema.h>
#include <simdjson.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "test_alice_moe_fixture.h"

namespace fs = std::filesystem;
using namespace alicetest;

namespace {

std::string ReadFile(const fs::path &p)
{
    std::ifstream f(p);
    return std::string((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
}

bool PinnedHasName(const fs::path &path, const std::string &name)
{
    auto batch = lsxcommon::ReadRecordBatchFromFile(path);
    if (!batch.ok()) return false;
    auto rows = lsxcommon::PinnedSchema::UnpackBatch(*batch);
    if (!rows.ok()) return false;
    for (const auto &r : *rows)
        if (r.name == name) return true;
    return false;
}

fs::path ConvertSyntheticAlice(const char *tag)
{
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    const fs::path in = fs::temp_directory_path() /
                        (std::string("alice_synth_in_") + tag);
    const fs::path out = fs::temp_directory_path() /
                         (std::string("alice_synth_out_") + tag);
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    WriteAliceConfigJson(in);
    WriteAliceShard(in);
    auto st = lsxcommon::alice::ConvertAliceModel(in.string(), out.string());
    if (!st.ok()) FAIL(st.ToString());
    REQUIRE(st.ok());
    return out;
}

bool DenseNvfp4HasName(const fs::path &path, const std::string &name)
{
    auto batch = lsxcommon::ReadRecordBatchFromFile(path);
    if (!batch.ok()) return false;
    auto rows = lsxcommon::DenseNvfp4Schema::UnpackBatch(*batch);
    if (!rows.ok()) return false;
    for (const auto &r : *rows)
        if (r.name == name) return true;
    return false;
}

} // namespace

TEST_CASE("alice synthetic convert emits logestix-alice-nvfp4", "[alice][convert]")
{
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    const fs::path in = fs::temp_directory_path() / "alice_synth_in";
    const fs::path out = fs::temp_directory_path() / "alice_synth_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    WriteAliceConfigJson(in);
    WriteAliceShard(in);

    auto st = lsxcommon::alice::ConvertAliceModel(in.string(), out.string());
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
    REQUIRE(fmt == std::to_string(lsxcommon::kIndexLayout));
    int64_t block = 0;
    REQUIRE(doc["block_attn_res_block_size"].get_int64().get(block) ==
            simdjson::SUCCESS);
    REQUIRE(block == kBlockAttnRes);
    int64_t layers = 0;
    REQUIRE(doc["num_layers"].get_int64().get(layers) == simdjson::SUCCESS);
    REQUIRE(layers == kLayers);

    // Every (layer, expert) file exists.
    for (int l = 0; l < kLayers; ++l)
        for (int e = 0; e < kExperts; ++e)
            REQUIRE(fs::exists(base / "experts" /
                               ("layer_" + std::to_string(l)) /
                               ("exp_" + std::to_string(e) + ".arrow")));

    // Layer 0 has no AttnRes tensors; layer 1 does.
    REQUIRE(PinnedHasName(base / "pinned" / "layer_0.arrow",
                          "model.layers.0.input_layernorm.weight"));
    REQUIRE_FALSE(PinnedHasName(base / "pinned" / "layer_0.arrow",
                                "model.layers.0.attn_res_proj.weight"));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_1.arrow",
                          "model.layers.1.attn_res_proj.weight"));

    // Full-attention layer 3: q_proj/q_norm stay FP16 (precision policy),
    // the shared expert stays NVFP4.
    REQUIRE(PinnedHasName(base / "pinned" / "layer_3.arrow",
                          "model.layers.3.self_attn.q_proj.weight"));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_3.arrow",
                          "model.layers.3.self_attn.q_norm.weight"));
    REQUIRE(DenseNvfp4HasName(base / "pinned" / "layer_3_nvfp4.arrow",
                              "model.layers.3.mlp.shared_expert.gate_proj.weight"));

    // Embedding stays high precision.
    REQUIRE(PinnedHasName(base / "pinned" / "global.arrow",
                          "model.embed_tokens.weight"));
}

TEST_CASE("alice vram profile streams routed experts as NVFP4",
          "[alice][convert]")
{
    // The routed-expert set is the memory bulk and is streamed from the host
    // store, so it is neither device-resident (min_bytes) nor stored at BF16 in
    // the profile. Before this was pinned, the converter's NVFP4 predicate
    // omitted the fused expert tensors and the estimator was given no
    // memory-class rule at all: the emitted index advertised
    // min_bytes == the whole BF16 source (162 GB on the real checkpoint) and
    // max_bytes == optimal_bytes (no expert pool), which is exactly backwards.
    using K = lsxcommon::alice::AliceTensorKind;
    REQUIRE(lsxcommon::alice::alice_tensor_is_nvfp4(K::FusedExpertsGateUp));
    REQUIRE(lsxcommon::alice::alice_tensor_is_nvfp4(K::FusedExpertsDown));
    REQUIRE(lsxcommon::alice::alice_tensor_is_routed_expert(K::FusedExpertsGateUp));
    REQUIRE(lsxcommon::alice::alice_tensor_is_routed_expert(K::FusedExpertsDown));
    REQUIRE_FALSE(lsxcommon::alice::alice_tensor_is_routed_expert(K::Embed));
    REQUIRE_FALSE(lsxcommon::alice::alice_tensor_is_routed_expert(K::SharedUpProj));

    const fs::path out = ConvertSyntheticAlice("vram");
    const std::string json = ReadFile(out / "model.index.json");
    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json);
    auto doc = parser.iterate(pad);
    int64_t min_bytes = 0, optimal_bytes = 0, max_bytes = 0;
    REQUIRE(doc["vram"]["min_bytes"].get_int64().get(min_bytes) ==
            simdjson::SUCCESS);
    REQUIRE(doc["vram"]["optimal_bytes"].get_int64().get(optimal_bytes) ==
            simdjson::SUCCESS);
    REQUIRE(doc["vram"]["max_bytes"].get_int64().get(max_bytes) ==
            simdjson::SUCCESS);

    const int64_t expert_numel =
        static_cast<int64_t>(kLayers) * kExperts * 3 * kMoeInter * kHidden;
    const int64_t expert_nvfp4_bytes =
        expert_numel / 2 + (expert_numel + 15) / 16;
    INFO("min=" << min_bytes << " optimal=" << optimal_bytes
                << " max=" << max_bytes << " expert_nvfp4=" << expert_nvfp4_bytes);
    REQUIRE(min_bytes > 0);
    REQUIRE(optimal_bytes > min_bytes);          // KV cache + workspace
    REQUIRE(max_bytes > optimal_bytes);          // routed-expert pool
    REQUIRE(max_bytes - optimal_bytes == expert_nvfp4_bytes);
}
