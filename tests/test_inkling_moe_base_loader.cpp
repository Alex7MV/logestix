// Inkling (base) loader-stage test.
//
// Synthetic half: pins the topology reader and the pinned-plane upload against
// a converted tree written by the converter under
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1. Real-weight half: reads ONE light file
// (the converted index) and SKIPs when the vault is absent — no weight is
// loaded into VRAM here.
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/inkling_moe_base_convert.h>
#include <lsxcommon/inkling_moe_base_load.h>
#include <lsxcommon/schema.h>

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "test_inkling_base_fixture.h"

#include "../libs/lsxcommon/src/inkling_moe_base_constants.h"

namespace fs = std::filesystem;
using namespace inklingbasetest;
namespace base = lsxcommon::inkling::base;

namespace {

std::string VaultRoot()
{
    if (const char *env = std::getenv("LSX_TEST_INKLING_ROOT"))
        return env;
    return "/mnt/fast_storage/lsx_model/thinkingmachines/Inkling";
}

fs::path ConvertedFixture()
{
    const fs::path in = fs::temp_directory_path() / "inkling_base_load_in";
    const fs::path out = fs::temp_directory_path() / "inkling_base_load_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    WriteCheckpoint(in);
    auto st = base::ConvertInklingBaseModel(in.string(), out.string());
    REQUIRE(st.ok());
    return out;
}

}  // namespace

TEST_CASE("inkling base topology reader reads a converted tree",
          "[inkling][load]")
{
    if (std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") == nullptr)
        SKIP("synthetic-geometry fixture not enabled");
    const fs::path out = ConvertedFixture();

    base::Topology top;
    auto ts = base::LoadTopology(out.string(), top);
    if (!ts.ok()) FAIL(ts.ToString());
    REQUIRE(top.num_layers == kLayers);
    REQUIRE(top.hidden == kHidden);
    REQUIRE(top.num_heads == kHeads);
    REQUIRE(top.num_kv_heads == kKvHeadsFull);
    REQUIRE(top.swa_kv_heads == kKvHeadsSliding);
    REQUIRE(top.head_dim == kHeadDim);
    REQUIRE(top.num_experts == kExperts);
    REQUIRE(top.top_k == kTopK);
    REQUIRE(top.moe_intermediate == kMoeInter);
    REQUIRE(top.shared_experts == kShared);
    REQUIRE(top.dense_mlp_layers == kDenseLayers);
    REQUIRE(top.dense_intermediate == kDenseInter);
    REQUIRE(top.d_rel == kDRel);
    REQUIRE(top.rel_extent == kRelExtent);
    REQUIRE(top.sliding_window == kWindow);
    REQUIRE(top.conv_kernel == kConvK);
    REQUIRE(top.attn_scale_denom == static_cast<double>(kHeadDim));
    REQUIRE(top.logits_mup_div == 24.0);
    REQUIRE(top.eos_id == 200006);
    REQUIRE(top.stop_id_alt == 199999);
    REQUIRE(top.bos_id == -1);
    REQUIRE(top.pad_id == -1);
    REQUIRE(top.full_layers == std::vector<int>{kFullLayer});
    REQUIRE(top.sliding_layers.size() == 5);
    REQUIRE(top.IsFull(kFullLayer));
    REQUIRE_FALSE(top.IsFull(4));
    REQUIRE(top.IsDenseMlp(0));
    REQUIRE(top.IsDenseMlp(1));
    REQUIRE_FALSE(top.IsDenseMlp(2));
    REQUIRE(top.KvHeadsOf(0) == kKvHeadsSliding);
    REQUIRE(top.KvHeadsOf(kFullLayer) == kKvHeadsFull);
    // Synthetic geometry relaxes the frozen cross-check by design.
    REQUIRE(base::TopologyMatchesFrozen(top));
}

TEST_CASE("inkling base pinned upload covers every plane and Free releases",
          "[inkling][load]")
{
    if (std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") == nullptr)
        SKIP("synthetic-geometry fixture not enabled");
    const fs::path out = ConvertedFixture();

    base::Topology top;
    REQUIRE(base::LoadTopology(out.string(), top).ok());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    size_t free_before = 0, total = 0;
    REQUIRE(cudaMemGetInfo(&free_before, &total) == cudaSuccess);

    base::PinnedWeights pw;
    pw.top = top;
    REQUIRE(base::LoadPinnedWeights(out.string(), pw, stream));
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    REQUIRE(pw.resident_bytes > 0);
    REQUIRE(pw.embed != nullptr);
    REQUIRE(pw.head != nullptr);
    REQUIRE(pw.embed_rows == kVocab);
    REQUIRE(pw.embed_cols == kHidden);
    REQUIRE(pw.embed_norm != nullptr);
    REQUIRE(pw.final_norm != nullptr);
    REQUIRE(pw.layers.size() == static_cast<size_t>(kLayers));

    for (int l = 0; l < kLayers; ++l) {
        const auto &lw = pw.layers[static_cast<size_t>(l)];
        CAPTURE(l);
        REQUIRE(lw.full == IsFull(l));
        REQUIRE(lw.wq.d_packed != nullptr);
        REQUIRE(lw.wq.rows == kQDim);
        REQUIRE(lw.wq.cols == kHidden);
        REQUIRE(lw.wk.rows == KvDim(l));
        REQUIRE(lw.wv.rows == KvDim(l));
        REQUIRE(lw.wo.rows == kHidden);
        REQUIRE(lw.wo.cols == kQDim);
        REQUIRE(lw.wr.rows == kRelDim);
        REQUIRE(lw.attn_norm != nullptr);
        REQUIRE(lw.mlp_norm != nullptr);
        REQUIRE(lw.q_norm != nullptr);
        REQUIRE(lw.k_norm != nullptr);
        REQUIRE(lw.rel_proj != nullptr);
        REQUIRE(lw.kconv_w != nullptr);
        REQUIRE(lw.vconv_w != nullptr);
        REQUIRE(lw.attn_conv_w != nullptr);
        REQUIRE(lw.mlp_conv_w != nullptr);
        REQUIRE(lw.conv_state != nullptr);
        const int kv = KvDim(l);
        REQUIRE(lw.conv_state_bytes ==
                int64_t(2 * kConvK * kv + 2 * kConvK * kHidden) * 4);
        if (IsDense(l)) {
            REQUIRE(lw.dense_gate.rows == kDenseInter);
            REQUIRE(lw.dense_up.rows == kDenseInter);
            REQUIRE(lw.dense_down.rows == kHidden);
            REQUIRE(lw.dense_down.cols == kDenseInter);
            REQUIRE(lw.mlp_gs != nullptr);
            REQUIRE(lw.router_w == nullptr);
        } else {
            REQUIRE(lw.router_w != nullptr);
            REQUIRE(lw.router_bias != nullptr);
            REQUIRE(lw.router_gs != nullptr);
            for (int s = 0; s < kShared; ++s) {
                REQUIRE(lw.shared[static_cast<size_t>(s)].gate.rows == kMoeInter);
                REQUIRE(lw.shared[static_cast<size_t>(s)].up.rows == kMoeInter);
                REQUIRE(lw.shared[static_cast<size_t>(s)].down.rows == kHidden);
            }
        }
    }

    // Expert store: MoE layers only, NVFP4 plane sizes.
    auto db = base::LoadExpertIndex(out.string(), top.num_layers,
                                    top.dense_mlp_layers, top.num_experts, 64);
    REQUIRE(db.error_msg.empty());
    for (int l = top.dense_mlp_layers; l < top.num_layers; ++l) {
        db.EnsureResident(l, 1, nullptr);
        const auto &e = db.entries[static_cast<size_t>(l) *
                                       static_cast<size_t>(top.num_experts) + 1];
        REQUIRE(e.host_w1 != nullptr);
        REQUIRE(e.host_w2 != nullptr);
        REQUIRE(e.host_w3 != nullptr);
        REQUIRE(e.w1_bytes == int64_t(kMoeInter) * kHidden / 2);
        REQUIRE(e.w2_bytes == int64_t(kHidden) * kMoeInter / 2);
        REQUIRE(e.w1_scales_bytes == int64_t(kMoeInter) * (kHidden / 16));
        REQUIRE(e.w2_scales_bytes == int64_t(kHidden) * (kMoeInter / 16));
        REQUIRE(e.w1_global_scale > 0.0f);
    }
    base::FreeExpertIndex(db);

    base::FreePinnedWeights(pw);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    size_t free_after = 0;
    REQUIRE(cudaMemGetInfo(&free_after, &total) == cudaSuccess);
    // Every allocation the loader added must be released (2 MiB slack for
    // context bookkeeping).
    INFO("free before " << free_before << " after " << free_after);
    REQUIRE(free_after + (2u << 20) >= free_before);
    REQUIRE(pw.resident_bytes == 0);
    REQUIRE(pw.layers.empty());
    cudaStreamDestroy(stream);
}

TEST_CASE("inkling base real vault index matches the frozen passport",
          "[inkling][load]")
{
    const fs::path root = VaultRoot();
    if (!fs::exists(root / "logestix" / "model.index.json"))
        SKIP("converted Inkling tree absent: " + root.string());

    base::Topology top;
    auto st = base::LoadTopology(root.string(), top);
    if (!st.ok()) FAIL(st.ToString());
    REQUIRE(top.format == "logestix-inkling-nvfp4");
    REQUIRE(top.source_model_type == "inkling_mm_model");
    REQUIRE(top.tokenizer_kind == 2);
    REQUIRE(base::TopologyMatchesFrozen(top));

    namespace frozen = lsxcommon::inkling::base::frozen;
    REQUIRE(top.num_layers == frozen::kLayers);
    REQUIRE(top.hidden == frozen::kHidden);
    REQUIRE(top.vocab == frozen::kVocab);
    REQUIRE(top.unpadded_vocab == frozen::kBaseVocab);
    REQUIRE(top.num_heads == frozen::kHeads);
    REQUIRE(top.num_kv_heads == frozen::kKvHeadsFull);
    REQUIRE(top.swa_kv_heads == frozen::kKvHeadsSliding);
    REQUIRE(top.head_dim == frozen::kHeadDim);
    REQUIRE(top.num_experts == frozen::kExperts);
    REQUIRE(top.top_k == frozen::kTopK);
    REQUIRE(top.moe_intermediate == frozen::kMoeIntermediate);
    REQUIRE(top.shared_experts == frozen::kSharedExperts);
    REQUIRE(top.dense_mlp_layers == frozen::kDenseMlpLayers);
    REQUIRE(top.dense_intermediate == frozen::kDenseIntermediate);
    REQUIRE(top.logits_mup_div == frozen::kLogitsMupDiv);
    REQUIRE(top.full_layers.size() == static_cast<size_t>(frozen::kFullLayerCount));
    for (size_t i = 0; i < frozen::kFullLayerCount; ++i)
        REQUIRE(top.full_layers[i] == frozen::kFullLayers[i]);

    // One expert record: the BF16 source quantizes to I*H/2 packed bytes and
    // I*H/16 scale bytes per plane.
    const fs::path rec = root / "logestix" / "experts" / "layer_2" / "exp_0.arrow";
    REQUIRE(fs::exists(rec));
    auto batch = lsxcommon::ReadRecordBatchFromFile(rec);
    if (!batch.ok()) FAIL(batch.status().ToString());
    auto pe = lsxcommon::ExpertSchema::UnpackBatch(*batch, 16);
    if (!pe.ok()) FAIL(pe.status().ToString());
    REQUIRE(pe->w1_packed.size() ==
            static_cast<size_t>(frozen::kMoeIntermediate) * (frozen::kHidden / 2));
    REQUIRE(pe->w2_packed.size() ==
            static_cast<size_t>(frozen::kHidden) * (frozen::kMoeIntermediate / 2));
    REQUIRE(pe->w1_global_scale > 0.0f);
    REQUIRE(pe->w2_global_scale > 0.0f);
}