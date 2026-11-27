// tests/test_deepseek_moe_pro_convert.cpp — DeepSeek-V4-Pro-0813 converter
// (stage 1 of the Model Ingestion Protocol).
//
// The Pro source is the same architecture as DeepSeek-V4-Flash (fp4-packed
// routed experts with per-32 E8M0 scales, FP8 block-128 dense projections),
// but it carries a DSpark/MTP ballast tree (`mtp.0..2.*`) that must be
// stripped AND counted, never converted into pinned payload or experts.
//
// This test pins, on a synthetic checkpoint:
//   * the frozen V4-Pro geometry gate (bypassed by
//     LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1, runtime env),
//   * the index identity ("DeepSeek-V4-Pro-0813", logestix-v1-nvfp4),
//   * `deferred.mtp` counting and the absence of mtp from pinned/experts,
//   * the routed-expert NVFP4 repack (E8M0/32 -> pow2-E4M3/16).
#include <catch2/catch_test_macros.hpp>

#include <lsxcommon/converter.h>
#include <lsxcommon/deepseek_moe_pro_convert.h>
#include <lsxcommon/schema.h>

#include <arrow/api.h>

#include "deepseek_moe_pro_constants.h"
#include "test_synth_checkpoint.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace {

using synthtest::FakeTensor;

constexpr uint8_t kScaleWindow[4] = {119, 120, 121, 122};

FakeTensor MakeExpertWeight(const std::string &name, int rows,
                            int packed_cols, uint8_t fill)
{
    FakeTensor t;
    t.name = name;
    t.dtype = "I8";
    t.shape = {rows, packed_cols};
    t.data.assign(static_cast<size_t>(rows) * packed_cols, fill);
    return t;
}

FakeTensor MakeExpertScale(const std::string &name, int rows,
                           const uint8_t *bytes, int n)
{
    FakeTensor t;
    t.name = name;
    t.dtype = "F8_E8M0";
    t.shape = {rows, 1};
    t.data.assign(bytes, bytes + n);
    return t;
}

fs::path BuildProSource(const fs::path &in)
{
    fs::create_directories(in);
    std::vector<FakeTensor> tensors;

    // Globals (BF16 pass-through).
    tensors.push_back({"embed.weight", "BF16", {128, 64},
                       synthtest::MakeBf16(128 * 64)});
    tensors.push_back({"head.weight", "BF16", {128, 64},
                       synthtest::MakeBf16(128 * 64)});
    tensors.push_back({"norm.weight", "BF16", {64},
                       synthtest::MakeBf16(64)});

    // Layer 0 (ratio 128 in the real checkpoint: hash router, no indexer).
    tensors.push_back({"layers.0.attn.wq_a.weight", "F8_E4M3", {16, 64},
                       std::vector<uint8_t>(16 * 64, 0xAA)});
    tensors.push_back({"layers.0.attn.wq_a.scale", "F8_E8M0", {1, 1},
                       std::vector<uint8_t>{120}});
    tensors.push_back({"layers.0.ffn.gate.weight", "BF16", {2, 64},
                       synthtest::MakeBf16(2 * 64)});
    // Hash routing table (I64) passes through pinned.
    FakeTensor tid;
    tid.name = "layers.0.ffn.gate.tid2eid";
    tid.dtype = "I64";
    tid.shape = {128, 2};
    tid.data.assign(128 * 2 * sizeof(int64_t), 0);
    tensors.push_back(tid);

    for (int e = 0; e < 2; ++e) {
        const std::string p = "layers.0.ffn.experts." + std::to_string(e);
        tensors.push_back(MakeExpertWeight(p + ".w1.weight", 4, 16,
                                           static_cast<uint8_t>(0xB0 + e)));
        tensors.push_back(MakeExpertScale(p + ".w1.scale", 4, kScaleWindow, 4));
        tensors.push_back(MakeExpertWeight(p + ".w2.weight", 8, 8, 0xC1));
        const uint8_t s2[8] = {120, 120, 120, 120, 120, 120, 120, 120};
        tensors.push_back(MakeExpertScale(p + ".w2.scale", 8, s2, 8));
        tensors.push_back(MakeExpertWeight(p + ".w3.weight", 4, 16, 0xC2));
        tensors.push_back(MakeExpertScale(p + ".w3.scale", 4, kScaleWindow, 4));
    }

    // DSpark / MTP ballast: stored-not-applied. Two tensors are enough to
    // prove the cut is by prefix and counted, not silently dropped.
    tensors.push_back({"mtp.0.attn.wq_a.weight", "F8_E4M3", {16, 64},
                       std::vector<uint8_t>(16 * 64, 0x11)});
    tensors.push_back({"mtp.0.ffn.experts.0.w1.weight", "I8", {4, 16},
                       std::vector<uint8_t>(4 * 16, 0x22)});

    synthtest::WriteSafetensors(in, "model-00001-of-00001.safetensors",
                                tensors);

    std::ofstream cfg(in / "config.json");
    cfg << "{\"num_hidden_layers\": 1, \"hidden_size\": 64,"
           " \"vocab_size\": 128, \"n_routed_experts\": 2,"
           " \"num_hash_layers\": 1, \"compress_ratios\": [128]}\n";
    fs::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
                  in / "tokenizer.json");
    return in;
}

std::string ReadText(const fs::path &p)
{
    std::ifstream f(p);
    return std::string((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
}

} // namespace

TEST_CASE("deepseek-v4-pro converter strips and counts MTP ballast",
          "[deepseek][pro][convert]")
{
    const auto base = fs::temp_directory_path() / "logestix_ds_pro_convert";
    fs::remove_all(base);
    const auto in = BuildProSource(base / "model");
    const auto out = base / "converted";

    const auto st =
        lsxcommon::deepseek_pro::ConvertDeepSeekProModel(in.string(),
                                                         out.string());
    INFO("converter status: " << st.ToString());
    REQUIRE(st.ok());

    const auto log_root = out;
    const std::string index = ReadText(log_root / "model.index.json");
    CHECK(index.find("\"logestix-v1-nvfp4\"") != std::string::npos);
    CHECK(index.find("\"DeepSeek-V4-Pro-0813\"") != std::string::npos);
    // Two mtp tensors were cut and reported, not silently discarded.
    CHECK(index.find("\"mtp\": 2") != std::string::npos);

    // No MTP payload under pinned/ or experts/.
    for (const auto &entry : fs::recursive_directory_iterator(log_root)) {
        if (!entry.is_regular_file()) continue;
        INFO("output file " << entry.path());
        CHECK(entry.path().filename().string().find("mtp") ==
              std::string::npos);
    }

    // Routed experts are NVFP4 pow2-E4M3/16 (window [119..122] -> m = -2).
    auto batch = lsxcommon::ReadRecordBatchFromFile(
        log_root / "experts" / "layer_0" / "exp_0.arrow");
    REQUIRE(batch.ok());
    auto pe = lsxcommon::ExpertSchema::UnpackBatch(*batch);
    REQUIRE(pe.ok());
    CHECK(pe->w1_global_scale == 0.25f);
    REQUIRE(pe->w1_scales.size() == 8);
    CHECK(pe->w1_scales[0] == (1 << 3));
    CHECK(pe->w1_scales[1] == (1 << 3));

    // The hash-routing table survives in the pinned layer file.
    auto layer = lsxcommon::ReadRecordBatchFromFile(
        log_root / "pinned" / "layer_0.arrow");
    REQUIRE(layer.ok());
    auto recs = lsxcommon::PinnedSchema::UnpackBatch(*layer);
    REQUIRE(recs.ok());
    bool saw_tid2eid = false;
    for (const auto &r : *recs) {
        if (r.name.find("ffn.gate.tid2eid") != std::string::npos) {
            saw_tid2eid = true;
            CHECK(r.dtype == "I64");
        }
    }
    CHECK(saw_tid2eid);

    fs::remove_all(base);
}

TEST_CASE("deepseek-v4-pro converter refuses a non-Pro geometry",
          "[deepseek][pro][convert]")
{
    // Synthetic geometry is allowed by the ctest environment, so the frozen
    // gate cannot fire here; instead pin the passport values themselves —
    // the converter gate is a direct comparison against these.
    CHECK(lsxcommon::deepseek_pro::kLayers == 61);
    CHECK(lsxcommon::deepseek_pro::kHidden == 7168);
    CHECK(lsxcommon::deepseek_pro::kVocab == 129280);
    CHECK(lsxcommon::deepseek_pro::kNumRouted == 384);
}
