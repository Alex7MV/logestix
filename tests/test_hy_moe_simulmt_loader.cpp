// Hy-SimulMT-30B-A3B loader stage (Model Ingestion Protocol stage 3): the
// shared hy loader with THIS model's passport (format tag + geometry), the
// resident routed-expert slabs and the KV geometry. Synthetic converted tree;
// no vault, no real weights.
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "hy_moe_simulmt_constants.h"
#include "lsxcommon/hy_moe_convert.h"
#include "lsxcommon/hy_moe_load.h"
#include "lsxcommon/hy_moe_simulmt_convert.h"
#include "test_hy_moe_simulmt_fixture.h"

namespace fs = std::filesystem;
using namespace lsxcommon;

namespace {

struct Converted {
    fs::path root;
    hy::HyTopology top;
};

hy::HyPassportGeometry Geometry()
{
    using namespace hy_simulmt::frozen;
    return hy::HyPassportGeometry{
        .layers = kLayers,
        .hidden = kHidden,
        .heads = kHeads,
        .kv_heads = kKvHeads,
        .head_dim = kHeadDim,
        .rotary_dim = kRotaryDim,
        .vocab = kVocab,
        .dense_intermediate = kDenseIntermediate,
        .moe_intermediate = kMoeIntermediate,
        .shared_intermediate = kSharedIntermediate,
        .num_experts = kNumExperts,
        .top_k = kTopK,
        .first_moe_layer = kFirstMoeLayer,
    };
}

Converted BuildConverted()
{
    Converted c;
    fs::path in = fs::temp_directory_path() / "lsx_hy_simulmt_load_in";
    c.root = fs::temp_directory_path() / "lsx_hy_simulmt_load_out";
    fs::remove_all(in);
    fs::remove_all(c.root);
    fs::create_directories(in);
    REQUIRE(hy_simulmt_test::BuildSyntheticCheckpoint(in) > 0);
    auto st = hy_simulmt::ConvertHySimulMTModel(in.string(), c.root.string());
    if (!st.ok()) WARN(st.ToString());
    REQUIRE(st.ok());
    c.top = hy::LoadHyTopology(c.root.string());
    c.top.error_msg.clear();
    return c;
}

} // namespace

TEST_CASE("Hy-SimulMT loader uploads the pinned graph with its passport",
          "[hy][simulmt][loader]")
{
    const Converted c = BuildConverted();
    const auto &cfg = hy_simulmt_test::TestCfg();
    REQUIRE(c.top.num_layers == cfg.layers);
    CHECK(c.top.moe_intermediate_size == cfg.moe_inter);

    hy::HyPinnedWeights pw;
    std::string error;
    REQUIRE(hy::LoadHyDenseWeights(c.root.string(), Geometry(), pw, nullptr,
                                   error));
    CHECK(error.empty());
    CHECK(pw.resident_bytes > 0);
    CHECK(pw.d_embed != nullptr);
    CHECK(pw.d_head != nullptr);
    CHECK(pw.d_final_norm != nullptr);
    REQUIRE(pw.layers.size() == static_cast<size_t>(cfg.layers));

    for (int l = 0; l < cfg.layers; ++l) {
        const auto &lw = pw.layers[static_cast<size_t>(l)];
        CHECK(lw.attn.loaded);
        CHECK(lw.attn.q.d_packed != nullptr);
        CHECK(lw.attn.d_q_norm != nullptr);
        const bool moe = l >= cfg.first_moe_layer;
        CHECK(lw.is_moe == moe);
        if (moe) {
            CHECK(lw.d_router != nullptr);
            CHECK(lw.d_router_bias != nullptr);
            CHECK(lw.shared_gate.d_packed != nullptr);
            REQUIRE(lw.experts.size() == static_cast<size_t>(cfg.experts));
            CHECK(lw.expert_slab != nullptr);
            CHECK(lw.d_row_table != nullptr);
        } else {
            CHECK(lw.mlp_gate.d_packed != nullptr);
            CHECK(lw.experts.empty());
        }
    }
    pw.Free();
    pw.Free();
    CHECK(pw.layers.empty());
    CHECK(pw.resident_bytes == 0);
}

TEST_CASE("the Hy loaders refuse an index generation they cannot read",
          "[hy][simulmt][loader]")
{
    // What a LOADER can own after the `format` tag retired (design §2.4.3): it
    // parses a tree, so it owns the schema GENERATION check. It cannot own the
    // parent-vs-sibling identity check any more — both trees are layout 2 and
    // have the same tower, so no geometry or generation test separates them.
    // That refusal moved to ModelBase::IdentityAllows, against artifact.fb's
    // model_type; test_load_codec_gate.cpp pins it.
    Converted c = BuildConverted();
    const fs::path idx = c.root / "model.index.json";
    std::string json;
    {
        std::ifstream f(idx);
        json.assign(std::istreambuf_iterator<char>(f),
                    std::istreambuf_iterator<char>());
    }
    const std::string from = "\"layout\": 2";
    const size_t at = json.find(from);
    REQUIRE(at != std::string::npos);
    json.replace(at, from.size(), "\"layout\": 1");
    {
        std::ofstream f(idx);
        f << json;
    }

    // Both overloads: the sibling's passport and the parent's default.
    {
        hy::HyPinnedWeights pw;
        std::string error;
        CHECK_FALSE(hy::LoadHyDenseWeights(c.root.string(), Geometry(), pw,
                                           nullptr, error));
        // The message is the SHARED refusal, naming both generations and the
        // fix. It used to read "unsupported index format '<the same sentence>'"
        // because the layout error travelled through the retired format carrier.
        CHECK(error.find("layout 1") != std::string::npos);
        CHECK(error.find("reads layout 2") != std::string::npos);
        CHECK(error.find("reconvert") != std::string::npos);
    }
    {
        hy::HyPinnedWeights pw;
        std::string error;
        CHECK_FALSE(hy::LoadHyDenseWeights(c.root.string(), pw, nullptr, error));
        CHECK(error.find("layout 1") != std::string::npos);
    }
}

TEST_CASE("Hy-SimulMT KV geometry is the parent's pos-major layout",
          "[hy][simulmt][loader]")
{
    hy::HyTopology top;
    top.num_layers = 4;
    top.num_key_value_heads = 2;
    top.head_dim = 8;
    const KvCacheGeometry g = hy::HyKvGeometry(top, 16);
    CHECK(g.pos_stride == 2 * 8);
    CHECK(g.head_stride == 8);
    CHECK(g.LayerStride() == 2 * 16 * 8);
    CHECK(g.Offset(1, 0, 3, 0) == g.LayerStride() + 3 * 16);
}
