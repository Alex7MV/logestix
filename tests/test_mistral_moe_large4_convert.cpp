// Mistral-Large-4-1T-A52B passport + converter tests (Model Ingestion Protocol
// stages 0/1). The passport case is host-only; the synthetic E2E converter case
// runs the real pipeline on a fixture (label `convert`, bounded TIMEOUT,
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY as a RUNTIME env var).

#include <catch2/catch_test_macros.hpp>

#include "mistral_moe_large4_constants.h"

using namespace lsxcommon::mistral_large4::frozen;

TEST_CASE("Mistral-Large-4 geometry is compile-time locked", "[mistral_large4][convert]")
{
    STATIC_REQUIRE(kHidden == 6144);
    STATIC_REQUIRE(kLayers == 72);
    STATIC_REQUIRE(kVocab == 204800);
    STATIC_REQUIRE(kHeads == 64);
    STATIC_REQUIRE(kKvHeads == 64);
    STATIC_REQUIRE(kKvLora == 512);
    STATIC_REQUIRE(kQkNope == 64);
    STATIC_REQUIRE(kQkRope == 64);
    STATIC_REQUIRE(kQkHead == 128);
    STATIC_REQUIRE(kVHead == 128);
    STATIC_REQUIRE(kQSize == 8192);
    STATIC_REQUIRE(kKvAWidthRope == 576);
    STATIC_REQUIRE(kKvAWidthNope == 512);
    STATIC_REQUIRE(kKvBRowsRope == 12288);
    STATIC_REQUIRE(kKvBRowsNope == 16384);
    STATIC_REQUIRE(kOSize == 8192);
    STATIC_REQUIRE(kDenseIntermediate == 24576);
    STATIC_REQUIRE(kMoeIntermediate == 6144);
    STATIC_REQUIRE(kNumRouted == 128);
    STATIC_REQUIRE(kTopK == 4);
    STATIC_REQUIRE(kIndexHeads == 32);
    STATIC_REQUIRE(kIndexHeadDim == 128);
    STATIC_REQUIRE(kIndexTopk == 2048);
    STATIC_REQUIRE(kLeRopeDims == 32);
    STATIC_REQUIRE(kEosId == 2);
    STATIC_REQUIRE(kPadId == 11);
}

TEST_CASE("Mistral-Large-4 layer tables match the checkpoint config",
          "[mistral_large4][convert]")
{
    int rope = 0, nope = 0, shared = 0, full = 0, moe = 0, dense = 0;
    for (int i = 0; i < kLayers; ++i)
    {
        if (kRopeLayer[static_cast<size_t>(i)]) ++rope; else ++nope;
        if (kKvSource[static_cast<size_t>(i)] != i) ++shared;
        if (kIndexerFull[static_cast<size_t>(i)]) ++full;
        if (kMoeLayer[static_cast<size_t>(i)]) ++moe; else ++dense;
    }
    REQUIRE(rope == kNumRopeLayers);
    REQUIRE(nope == kNumNopeLayers);
    REQUIRE(shared == kNumKvShared);
    REQUIRE(full == kNumFullIndexers);
    REQUIRE(moe == 71);
    REQUIRE(dense == 1);
    // Every NoPE layer is index i % 4 == 3 and every KV sharer is a NoPE layer
    // reading layer 35.
    for (int i = 0; i < kLayers; ++i)
    {
        const bool is_nope = kRopeLayer[static_cast<size_t>(i)] == 0;
        REQUIRE(is_nope == (i % 4 == 3));
        if (kKvSource[static_cast<size_t>(i)] != i)
        {
            REQUIRE(is_nope);
            REQUIRE(kKvSource[static_cast<size_t>(i)] == kKvSourceLayer);
        }
    }
    // The dense MLP prefix is exactly layer 0.
    REQUIRE(kMoeLayer[0] == 0);
    REQUIRE(kFirstKDense == 1);
}

// ── synthetic end-to-end converter ─────────────────────────────────────────
#include <lsxcommon/mistral_moe_large4_convert.h>
#include <lsxcommon/schema.h>
#include "nvfp4_codec.h"
#include "test_mistral_moe_large4_fixture.h"

#include <absl/strings/match.h>
#include <absl/strings/str_cat.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace fs = std::filesystem;
using K4 = lsxcommon::mistral_large4::Large4TensorKind;

static std::string ReadAll(const fs::path &p)
{
    std::ifstream f(p);
    return std::string(std::istreambuf_iterator<char>(f),
                       std::istreambuf_iterator<char>());
}

TEST_CASE("Mistral-Large-4 classifier covers every tensor family",
          "[mistral_large4][convert]")
{
    using lsxcommon::mistral_large4::mistral_large4_classify;
    auto t = mistral_large4_classify("layers.7.attention.wq.weight");
    REQUIRE(t.kind == K4::AttnNvfp4);
    REQUIRE(t.layer == 7);
    REQUIRE(t.base == "attention.wq");
    REQUIRE(t.companion.empty());
    REQUIRE(mistral_large4_classify("layers.7.attention.wq.qscale_weight")
                .companion == "scale");
    REQUIRE(mistral_large4_classify("layers.7.attention.wo.weight").kind ==
            K4::AttnNvfp4);
    REQUIRE(mistral_large4_classify("layers.7.attention.indexer.wq_b.weight")
                .kind == K4::IndexerNvfp4);
    REQUIRE(mistral_large4_classify("layers.7.attention.wkv_b.weight").kind ==
            K4::AttnRaw);
    REQUIRE(mistral_large4_classify("layers.7.attention.wkv_a_with_mqa.weight")
                .kind == K4::AttnRaw);
    REQUIRE(mistral_large4_classify("layers.7.attention.kv_a_norm.weight").kind ==
            K4::LayerRaw);
    REQUIRE(mistral_large4_classify("layers.7.attention.ssmax_scaling.s").kind ==
            K4::LayerRaw);
    REQUIRE(mistral_large4_classify("layers.7.attention.indexer.k_norm.bias")
                .kind == K4::LayerRaw);
    REQUIRE(mistral_large4_classify("layers.2.gate.weight").kind == K4::Router);
    REQUIRE(mistral_large4_classify("layers.2.experts.3.w2.weight").kind ==
            K4::RoutedExpert);
    REQUIRE(mistral_large4_classify("layers.2.experts.3.w2.weight").expert == 3);
    REQUIRE(mistral_large4_classify("layers.2.experts.3.w2.weight").weight_id ==
            2);
    REQUIRE(mistral_large4_classify("layers.2.shared_experts.w3.weight").kind ==
            K4::SharedExpert);
    REQUIRE(mistral_large4_classify("layers.0.feed_forward.w1.weight").kind ==
            K4::DenseMlp);
    REQUIRE(mistral_large4_classify("tok_embeddings.weight").kind == K4::Global);
    REQUIRE(mistral_large4_classify("lerope.alpha").kind == K4::Global);
    REQUIRE(mistral_large4_classify("vision_encoder.ln_pre.weight").kind ==
            K4::Ballast);
    REQUIRE(mistral_large4_classify("mtp_dspark.backbone.final_norm.weight").kind ==
            K4::Ballast);
    REQUIRE(mistral_large4_classify("layers.7.unknown.weight").kind ==
            K4::Unknown);
}

TEST_CASE("Mistral-Large-4 synthetic convert emits the full tree",
          "[mistral_large4][convert]")
{
    const auto in = fs::temp_directory_path() / "mistral_large4_conv_in";
    const auto out = fs::temp_directory_path() / "mistral_large4_conv_out";
    mistrallarge4test::Large4SynthGeom g;
    mistrallarge4test::WriteCheckpoint(in, g);
    fs::remove_all(out);

    const auto st =
        lsxcommon::mistral_large4::ConvertMistralLarge4Model(in.string(), out);
    INFO(st.ToString());
    REQUIRE(st.ok());

    const fs::path base = out;
    const std::string index = ReadAll(base / "model.index.json");
    REQUIRE(absl::StrContains(index, "\"format\": \"logestix-mistral-large4-nvfp4\""));
    REQUIRE(absl::StrContains(index, "\"model\": \"Mistral-Large-4-1T\""));
    REQUIRE(absl::StrContains(index, "\"num_layers\": 4"));
    REQUIRE(absl::StrContains(index, "\"experts_per_layer\": 2"));
    REQUIRE(absl::StrContains(index, "\"rope_layers\": [1,1,1,0]"));
    REQUIRE(absl::StrContains(index, "\"kv_source_layers\": [0,1,2,2]"));
    REQUIRE(absl::StrContains(index, "\"indexer_layers\": [1,0,0,1]"));
    REQUIRE(absl::StrContains(index, "\"moe_layers\": [0,1,1,1]"));
    REQUIRE(absl::StrContains(index, "\"dense_layers\": [0]"));
    REQUIRE(absl::StrContains(index, "\"mtp_tensors\": 2"));
    REQUIRE(absl::StrContains(index, "\"total_experts\": 6"));

    // Globals stay BF16 verbatim.
    const auto gnames = synthtest::TensorNamesIn(base / "pinned" / "global.arrow");
    REQUIRE(gnames.size() == 4);

    // Layer 3 is the NoPE KV-sharing layer: no wkv_a/kv_a_norm on disk.
    const auto l3 = synthtest::TensorNamesIn(base / "pinned" / "layer_3.arrow");
    REQUIRE(!l3.empty());
    for (const auto &n : l3)
        REQUIRE(!absl::StrContains(n, "wkv_a_with_mqa"));
    const auto l0 = synthtest::TensorNamesIn(base / "pinned" / "layer_0.arrow");
    bool has_kv_a = false;
    for (const auto &n : l0)
        if (absl::StrContains(n, "wkv_a_with_mqa")) has_kv_a = true;
    REQUIRE(has_kv_a);

    // Per-layer NVFP4 projections: wq/wo everywhere, indexer on full layers,
    // dense MLP on layer 0 and shared expert on the MoE layers.
    for (int l = 0; l < g.layers; ++l)
    {
        auto b = lsxcommon::ReadRecordBatchFromFile(
            base / "pinned" / absl::StrCat("layer_", l, "_nvfp4.arrow"));
        REQUIRE(b.ok());
        auto recs = lsxcommon::DenseNvfp4Schema::UnpackBatch(*b);
        REQUIRE(recs.ok());
        bool wq = false, wo = false, idx = false, ff = false, sh = false;
        for (const auto &r : *recs)
        {
            wq |= absl::StrContains(r.name, ".attention.wq.weight");
            wo |= absl::StrContains(r.name, ".attention.wo.weight");
            idx |= absl::StrContains(r.name, ".indexer.wq_b.weight");
            ff |= absl::StrContains(r.name, ".feed_forward.w1.weight");
            sh |= absl::StrContains(r.name, ".shared_experts.w1.weight");
            REQUIRE(r.global_scale > 0.0f);
            REQUIRE(!r.packed.empty());
            REQUIRE(!r.scales.empty());
        }
        REQUIRE(wq);
        REQUIRE(wo);
        REQUIRE(idx == g.full_indexer(l));
        REQUIRE(ff == !g.moe(l));
        REQUIRE(sh == g.moe(l));
    }

    // Routed experts: 3 MoE layers x 2 experts.
    int expert_files = 0;
    for (int l = 1; l < g.layers; ++l)
        for (int e = 0; e < g.experts; ++e)
            if (fs::exists(base / "experts" / absl::StrCat("layer_", l) /
                           absl::StrCat("exp_", e, ".arrow")))
                ++expert_files;
    REQUIRE(expert_files == 6);

    // ── numeric oracle: decode the base plane on the host and compare with
    // the source FP8 dequant. `base relL2 < 0.15` is the NVFP4 noise floor.
    auto b = lsxcommon::ReadRecordBatchFromFile(base / "pinned" /
                                                "layer_0_nvfp4.arrow");
    REQUIRE(b.ok());
    auto recs = lsxcommon::DenseNvfp4Schema::UnpackBatch(*b);
    REQUIRE(recs.ok());
    const lsxcommon::DenseNvfp4Record *wq = nullptr;
    for (const auto &r : *recs)
        if (absl::StrContains(r.name, "attention.wq.weight")) wq = &r;
    REQUIRE(wq != nullptr);
    const int64_t rows = wq->shape[0], cols = wq->shape[1];
    const int64_t numel = rows * cols;
    REQUIRE(wq->packed.size() == static_cast<size_t>(numel / 2));

    // Source FP8 dequant (same fixture pattern) for the same tensor.
    const int64_t gsc = (cols + 127) / 128;
    auto src_scales = mistrallarge4test::TileScales(rows, cols);
    auto fp8 = mistrallarge4test::Fp8Bytes(numel);
    double num = 0, den = 0;
    for (int64_t i = 0; i < numel; ++i)
    {
        const uint8_t byte = wq->packed[static_cast<size_t>(i / 2)];
        const int nib = (i % 2 == 0) ? (byte & 0xF) : (byte >> 4);
        const int64_t blk = i / 16;
        const float s = lsxcommon::gpu::nvfp4::E4M3ScaleToFloat(
            wq->scales[static_cast<size_t>(blk)]);
        const float got = lsxcommon::gpu::nvfp4::E2M1Value(nib) * s *
                          wq->global_scale;
        const float tile_s = [&] {
            float v;
            std::memcpy(&v, src_scales.data() + 4 * ((i / cols) / 128 * gsc +
                                                     (i % cols) / 128), 4);
            return v;
        }();
        const float want =
            lsxcommon::e4m3_to_float(fp8[static_cast<size_t>(i)]) * tile_s;
        num += static_cast<double>(got - want) * (got - want);
        den += static_cast<double>(want) * want;
    }
    REQUIRE(den > 0.0);
    // Quantization noise: the reconstruction must track the source far better
    // than zero (relL2 well below 1).
    REQUIRE(std::sqrt(num / den) < 0.15);
}
