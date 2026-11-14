// Inkling-2-Small loader-stage test.
//
// Two halves:
//   * a synthetic half that pins the topology reader against a converted tree
//     written by the converter under LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1;
//   * a real-weight half that reads ONE light file (the converted
//     `logestix/model.index.json`) plus the byte sizes of two expert records,
//     and SKIPs when the vault is absent. No weight is loaded into VRAM here.
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/inkling_moe_convert.h>
#include <lsxcommon/inkling_moe_load.h>
#include <lsxcommon/schema.h>

#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "test_inkling_fixture.h"

#include "../libs/lsxcommon/src/inkling_moe_constants.h"

namespace fs = std::filesystem;
using namespace inklingtest;

namespace {

std::string VaultRoot()
{
    if (const char *env = std::getenv("LSX_TEST_INKLING_2_SMALL_ROOT"))
        return env;
    return "/mnt/fast_storage/huggingface/Inkling-2-Small";
}

}  // namespace

TEST_CASE("inkling topology reader reads a converted tree",
          "[inkling][load]")
{
    if (std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") == nullptr)
        SKIP("synthetic-geometry fixture not enabled");
    const fs::path in = fs::temp_directory_path() / "inkling_load_in";
    const fs::path out = fs::temp_directory_path() / "inkling_load_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    WriteCheckpoint(in);

    auto st = lsxcommon::inkling::ConvertInklingModel(in.string(), out.string());
    if (!st.ok()) FAIL(st.ToString());

    lsxcommon::inkling::InklingTopology top;
    auto ts = lsxcommon::inkling::LoadInklingTopology(out.string(), top);
    if (!ts.ok()) FAIL(ts.ToString());
    REQUIRE(top.num_layers == kLayers);
    REQUIRE(top.hidden == kHidden);
    REQUIRE(top.num_heads == kHeads);
    REQUIRE(top.num_kv_heads == kKvHeads);
    REQUIRE(top.head_dim == kHeadDim);
    REQUIRE(top.num_experts == kExperts);
    REQUIRE(top.top_k == 2);
    REQUIRE(top.moe_intermediate == kMoeInter);
    REQUIRE(top.d_rel == kDRel);
    REQUIRE(top.rel_extent == kRelExtent);
    REQUIRE(top.sliding_window == kWindow);
    REQUIRE(top.conv_kernel == kConvK);
    REQUIRE(top.attn_res_block == kAttnResBlock);
    REQUIRE(top.attn_scale_denom == static_cast<double>(kHeadDim));
    REQUIRE(top.eos_id == 200006);
    REQUIRE(top.stop_id_alt == 199999);
    REQUIRE(top.bos_id == -1);
    REQUIRE(top.pad_id == -1);
    REQUIRE(top.full_layers.size() == 1);
    REQUIRE(top.full_layers[0] == kFullLayer);
    REQUIRE(top.sliding_layers.size() == 5);
    REQUIRE(top.IsFull(kFullLayer));
    REQUIRE_FALSE(top.IsFull(4));
    REQUIRE(top.IsBlockBoundary(0));
    REQUIRE_FALSE(top.IsBlockBoundary(1));
    REQUIRE(top.FullLayerIndex(kFullLayer) == 0);
    REQUIRE(top.FullLayerIndex(0) == -1);
    REQUIRE(top.expert_nvfp4.size() == static_cast<size_t>(kLayers));
    for (int l = 0; l < kLayers; ++l)
        REQUIRE(top.expert_nvfp4[static_cast<size_t>(l)] == (l < kNovfp4Layers));

    // Expert records carry the uniform NVFP4 layout regardless of the source
    // codec: [I, H] gate/up and [H, I] down, per-16 scales.
    for (int l : {0, kLayers - 1}) {
        auto batch = lsxcommon::ReadRecordBatchFromFile(
            out / "logestix" / "experts" / ("layer_" + std::to_string(l)) /
            "exp_0.arrow");
        REQUIRE(batch.ok());
        auto pe = lsxcommon::ExpertSchema::UnpackBatch(*batch, 16);
        if (!pe.ok()) FAIL(pe.status().ToString());
        REQUIRE(pe->w1_packed.size() ==
                static_cast<size_t>(kMoeInter) * (kHidden / 2));
        REQUIRE(pe->w3_packed.size() ==
                static_cast<size_t>(kMoeInter) * (kHidden / 2));
        REQUIRE(pe->w2_packed.size() ==
                static_cast<size_t>(kHidden) * (kMoeInter / 2));
        REQUIRE(pe->w1_scales.size() ==
                static_cast<size_t>(kMoeInter) * (kHidden / 16));
        REQUIRE(pe->w2_scales.size() ==
                static_cast<size_t>(kHidden) * (kMoeInter / 16));
    }
}

TEST_CASE("inkling real vault index matches the frozen passport",
          "[inkling][load]")
{
    const fs::path root = VaultRoot();
    if (!fs::exists(root / "logestix" / "model.index.json"))
        SKIP("converted Inkling-2-Small tree absent: " + root.string());

    lsxcommon::inkling::InklingTopology top;
    auto st = lsxcommon::inkling::LoadInklingTopology(root.string(), top);
    if (!st.ok()) FAIL(st.ToString());

    REQUIRE(top.format == "logestix-inkling-2-small-nvfp4");
    REQUIRE(top.source_model_type == "inkling2");
    REQUIRE(top.tokenizer_kind == 2);

    // The whole point of the passport: the real tree must match it exactly.
    REQUIRE(lsxcommon::inkling::InklingTopologyMatchesFrozen(top));

    namespace frozen = lsxcommon::inkling::frozen;
    REQUIRE(top.num_layers == frozen::kLayers);
    REQUIRE(top.hidden == frozen::kHidden);
    REQUIRE(top.vocab == frozen::kVocab);
    REQUIRE(top.unpadded_vocab == frozen::kBaseVocab);
    REQUIRE(top.num_heads == frozen::kHeads);
    REQUIRE(top.num_kv_heads == frozen::kKvHeads);
    REQUIRE(top.head_dim == frozen::kHeadDim);
    REQUIRE(top.d_rel == frozen::kDRel);
    REQUIRE(top.rel_extent == frozen::kRelExtent);
    REQUIRE(top.sliding_window == frozen::kSlidingWindow);
    REQUIRE(top.conv_kernel == frozen::kConvKernel);
    REQUIRE(top.attn_res_block == frozen::kAttnResBlock);
    REQUIRE(top.num_experts == frozen::kExperts);
    REQUIRE(top.top_k == frozen::kTopK);
    REQUIRE(top.moe_intermediate == frozen::kMoeIntermediate);
    REQUIRE(top.shared_experts == 0);
    REQUIRE(top.attn_scale_denom == frozen::kAttnScaleDenom);
    REQUIRE(top.qk_l2_clip == frozen::kQkL2Clip);
    REQUIRE(top.rel_l2_clip == frozen::kRelL2Clip);
    REQUIRE(top.log_scaling_n_floor == frozen::kLogScaleFloor);
    REQUIRE(top.eos_id == frozen::kEosId);
    REQUIRE(top.bos_id == -1);
    REQUIRE(top.full_layers.size() == static_cast<size_t>(frozen::kFullLayerCount));
    for (size_t i = 0; i < frozen::kFullLayerCount; ++i)
        REQUIRE(top.full_layers[i] == frozen::kFullLayers[i]);
    REQUIRE(top.expert_nvfp4.size() == static_cast<size_t>(frozen::kLayers));
    int nvfp4_layers = 0;
    for (bool b : top.expert_nvfp4) nvfp4_layers += b ? 1 : 0;
    REQUIRE(nvfp4_layers == frozen::kNovfp4ExpertLayerCount);

    // One expert record of an NVFP4-source layer: the pass-through must keep
    // the checkpoint's own byte count (I*H/2 packed, I*H/16 scales per plane).
    const fs::path dir = root / "logestix" / "experts" / "layer_0" / "exp_0.arrow";
    REQUIRE(fs::exists(dir));
    auto batch = lsxcommon::ReadRecordBatchFromFile(dir);
    if (!batch.ok()) FAIL(batch.status().ToString());
    auto pe = lsxcommon::ExpertSchema::UnpackBatch(*batch, lsxcommon::kNVFP4BlockSize);
    if (!pe.ok()) FAIL(pe.status().ToString());
    REQUIRE(pe->w1_packed.size() ==
            static_cast<size_t>(frozen::kMoeIntermediate) * (frozen::kHidden / 2));
    REQUIRE(pe->w2_packed.size() ==
            static_cast<size_t>(frozen::kHidden) * (frozen::kMoeIntermediate / 2));
    REQUIRE(pe->w1_global_scale > 0.0f);
    REQUIRE(pe->w2_global_scale > 0.0f);
    // The per-expert global scale is amax/2688, i.e. ~1e-5 for these weights.
    REQUIRE(pe->w1_global_scale < 1.0f);
    REQUIRE(pe->w2_global_scale < 1.0f);
}
