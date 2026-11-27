// Beam loader stage (Model Ingestion Protocol stage 3): the pinned dense graph,
// the F32 router, and the streamed neutral ExpertStore geometry. Runs on a
// synthetic converted tree; no vault, no real weights.
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "lsxcommon/beam_moe_convert.h"
#include "lsxcommon/beam_moe_load.h"
#include "test_beam_moe_fixture.h"

namespace fs = std::filesystem;
using namespace lsxcommon::beam;

namespace {

struct Converted {
    fs::path root;
    BeamTopology top;
};

Converted BuildConverted()
{
    Converted c;
    fs::path in = fs::temp_directory_path() / "lsx_beam_load_in";
    c.root = fs::temp_directory_path() / "lsx_beam_load_out";
    fs::remove_all(in);
    fs::remove_all(c.root);
    REQUIRE(beam_test::BuildSyntheticCheckpoint(in) > 0);
    auto st = ConvertBeamModel(in.string(), c.root.string());
    if (!st.ok()) WARN(st.ToString());
    REQUIRE(st.ok());
    c.top = LoadBeamTopology(c.root.string());
    c.top.error_msg.clear();
    return c;
}

} // namespace

TEST_CASE("beam loader uploads the pinned graph and opens the expert store",
          "[beam][loader]")
{
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
    const Converted c = BuildConverted();
    const auto &cfg = beam_test::TestCfg();
    REQUIRE(c.top.num_layers == cfg.layers);
    CHECK(c.top.moe_intermediate_size == cfg.moe_inter);
    CHECK(c.top.n_group == cfg.n_group);

    BeamPinnedWeights pw;
    std::string error;
    REQUIRE(LoadBeamWeights(c.root.string(), pw, nullptr, error));
    CHECK(error.empty());
    CHECK(pw.resident_bytes > 0);
    CHECK(pw.d_embed != nullptr);
    CHECK(pw.d_head != nullptr);
    CHECK(pw.d_final_norm != nullptr);
    CHECK(pw.embed_elems == static_cast<int64_t>(cfg.vocab) * cfg.hidden);
    CHECK(pw.head_elems == static_cast<int64_t>(cfg.vocab) * cfg.hidden);
    REQUIRE(pw.layers.size() == static_cast<size_t>(cfg.layers));

    for (int l = 0; l < cfg.layers; ++l) {
        const auto &lw = pw.layers[static_cast<size_t>(l)];
        CHECK(lw.loaded);
        CHECK(lw.attn.loaded);
        CHECK(lw.attn.q.d_packed != nullptr);
        CHECK(lw.attn.k.d_packed != nullptr);
        CHECK(lw.attn.v.d_packed != nullptr);
        CHECK(lw.attn.o.d_packed != nullptr);
        CHECK(lw.attn.gate.d_packed != nullptr);
        CHECK(lw.attn.d_input_norm != nullptr);
        CHECK(lw.attn.d_post_attn_norm != nullptr);
        CHECK(lw.attn.d_pre_ffn_norm != nullptr);
        CHECK(lw.attn.d_post_ffn_norm != nullptr);
        CHECK(lw.attn.d_q_norm != nullptr);
        CHECK(lw.attn.d_k_norm != nullptr);
        CHECK(lw.attn.d_gate_bias != nullptr);
        CHECK(lw.attn.q.rows == cfg.heads * cfg.head_dim);
        CHECK(lw.attn.q.cols == cfg.hidden);
        CHECK(lw.attn.k.rows == cfg.kv_heads * cfg.head_dim);
        CHECK(lw.attn.o.rows == cfg.hidden);
        CHECK(lw.attn.gate.rows == cfg.heads * cfg.head_dim);
        CHECK(lw.is_moe == (l >= cfg.first_moe_layer));
        if (lw.is_moe) {
            CHECK(lw.d_router != nullptr);
            CHECK(lw.d_router_bias != nullptr);
            CHECK(lw.router_bytes == static_cast<int64_t>(cfg.experts) *
                                         cfg.hidden * 4 +
                                         static_cast<int64_t>(cfg.experts) * 4);
            CHECK(lw.mlp_gate.d_packed == nullptr);
        } else {
            CHECK(lw.mlp_gate.d_packed != nullptr);
            CHECK(lw.mlp_up.d_packed != nullptr);
            CHECK(lw.mlp_down.d_packed != nullptr);
            CHECK(lw.mlp_gate.rows == cfg.dense_inter);
            CHECK(lw.mlp_down.rows == cfg.hidden);
            CHECK(lw.d_router == nullptr);
        }
    }

    // Streamed expert store: dense prefix slots are empty, every MoE slot maps.
    const auto &db = pw.expert_store.experts;
    CHECK(db.num_layers == cfg.layers);
    CHECK(db.experts_per_layer == cfg.experts);
    CHECK(db.first_moe_layer == cfg.first_moe_layer);
    for (int l = cfg.first_moe_layer; l < cfg.layers; ++l) {
        REQUIRE(pw.expert_store.records[static_cast<size_t>(l)].size() ==
                static_cast<size_t>(cfg.experts));
        const auto &r = pw.expert_store.records[static_cast<size_t>(l)][0];
        CHECK(r.w1_bytes > 0);
        CHECK(r.w2_bytes > 0);
        CHECK(r.w3_bytes > 0);
        CHECK(r.gs1 > 0.0f);
    }

    pw.Free();
    CHECK(pw.resident_bytes == 0);
    CHECK(pw.d_embed == nullptr);
    CHECK(pw.expert_store.records.empty());
}

TEST_CASE("beam loader rejects a foreign index format", "[beam][loader]")
{
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
    Converted c = BuildConverted();
    const fs::path idx = c.root / "model.index.json";
    std::string json;
    {
        std::ifstream f(idx);
        json.assign(std::istreambuf_iterator<char>(f),
                    std::istreambuf_iterator<char>());
    }
    const std::string from = "logestix-beam-nvfp4";
    const size_t at = json.find(from);
    REQUIRE(at != std::string::npos);
    json.replace(at, from.size(), "logestix-other-nvfp");
    {
        std::ofstream f(idx);
        f << json;
    }
    BeamPinnedWeights pw;
    std::string error;
    CHECK_FALSE(LoadBeamWeights(c.root.string(), pw, nullptr, error));
    CHECK(error.find("unsupported index format") != std::string::npos);
}

TEST_CASE("beam KV geometry is pos-major and size-neutral", "[beam][loader]")
{
    BeamTopology top;
    top.num_layers = 4;
    top.num_key_value_heads = 2;
    top.head_dim = 8;
    const lsxcommon::KvCacheGeometry g = BeamKvGeometry(top, 16);
    CHECK(g.pos_stride == 2 * 8);
    CHECK(g.head_stride == 8);
    CHECK(g.LayerStride() == 2 * 16 * 8);
    CHECK(g.Offset(1, 0, 3, 0) == g.LayerStride() + 3 * 16);
}
