// RecursiveSI/flint loader stage (Model Ingestion Protocol stage 3): the
// pinned dense graph, the resident routed-expert slabs and the pointer tables
// the batched expert GEMV consumes. Runs on a synthetic converted tree; no
// vault, no real weights.
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "lsxcommon/flint_moe_convert.h"
#include "lsxcommon/flint_moe_load.h"
#include "test_flint_moe_fixture.h"

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

namespace fs = std::filesystem;
using namespace lsxcommon::flint;

namespace {

struct Converted {
    fs::path root;
    FlintTopology top;
};

Converted BuildConverted()
{
    Converted c;
    fs::path in = fs::temp_directory_path() / "lsx_flint_load_out";
    c.root = fs::temp_directory_path() / "lsx_flint_load_out";
    fs::remove_all(in);
    fs::remove_all(c.root);
    const fs::path tok = fs::path(LSX_SOURCE_DIR) / "tests" / "data" /
                         "synthetic_flint_tok.json";
    REQUIRE(flint_test::BuildSyntheticCheckpoint(in, true, tok) > 0);
    auto st = ConvertFlintModel(in.string(), c.root.string());
    if (!st.ok()) WARN(st.ToString());
    REQUIRE(st.ok());
    c.top = LoadFlintTopology(c.root.string());
    c.top.error_msg.clear();
    return c;
}

} // namespace

TEST_CASE("flint loader uploads the pinned graph", "[flint][loader]")
{
    const Converted c = BuildConverted();
    const auto &cfg = flint_test::TestCfg();
    REQUIRE(c.top.num_layers == cfg.layers);
    CHECK(c.top.moe_intermediate_size == cfg.moe_inter);

    FlintPinnedWeights pw;
    std::string error;
    REQUIRE(LoadFlintDenseWeights(c.root.string(), pw, nullptr, error));
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
        CHECK(lw.attn.loaded);
        CHECK(lw.attn.q.d_packed != nullptr);
        CHECK(lw.attn.k.d_packed != nullptr);
        CHECK(lw.attn.v.d_packed != nullptr);
        CHECK(lw.attn.o.d_packed != nullptr);
        CHECK(lw.attn.d_input_norm != nullptr);
        CHECK(lw.attn.d_post_norm != nullptr);
        CHECK(lw.attn.d_q_norm != nullptr);
        CHECK(lw.attn.d_k_norm != nullptr);
        CHECK(lw.attn.d_sink != nullptr);
        CHECK(lw.attn.q.rows == cfg.heads * cfg.head_dim);
        CHECK(lw.attn.q.cols == cfg.hidden);
        CHECK(lw.attn.k.rows == cfg.kv_heads * cfg.head_dim);
        CHECK(lw.attn.o.rows == cfg.hidden);
        // Second NVFP4 plane (w - dequant(base)) on the dense side: uploaded
        // with the base shape, and never on the routed experts.
        CHECK(lw.attn.q_res.d_packed != nullptr);
        CHECK(lw.attn.o_res.d_packed != nullptr);
        CHECK(lw.attn.q_res.rows == cfg.heads * cfg.head_dim);
        CHECK(lw.attn.q_res.cols == cfg.hidden);
        const bool moe = l >= cfg.first_moe_layer;
        CHECK(lw.is_moe == moe);
        if (moe) {
            CHECK(lw.d_router != nullptr);
            CHECK(lw.shared_gate.d_packed != nullptr);
            CHECK(lw.shared_up.d_packed != nullptr);
            CHECK(lw.shared_down.d_packed != nullptr);
            CHECK(lw.shared_down_res.d_packed != nullptr);
            CHECK(lw.shared_gate.rows == cfg.shared_inter);
            REQUIRE(lw.experts.size() == static_cast<size_t>(cfg.experts));
            CHECK(lw.expert_slab != nullptr);
            CHECK(lw.slab_bytes > 0);
            // The device-resident pointer table the ids-streaming GEMV reads
            // ([gate E][down E][up E] rows).
            CHECK(lw.d_row_table != nullptr);
            // Pointer tables index into the slab and every expert is distinct.
            REQUIRE(lw.packed_gate.size() == static_cast<size_t>(cfg.experts));
            for (int e = 0; e < cfg.experts; ++e) {
                const size_t ui = static_cast<size_t>(e);
                CHECK(lw.packed_gate[ui] >= lw.expert_slab);
                CHECK(lw.packed_gate[ui] <
                      lw.expert_slab + lw.slab_bytes);
                CHECK(lw.packed_down[ui] != lw.packed_gate[ui]);
                CHECK(lw.packed_up[ui] != lw.packed_gate[ui]);
                CHECK(lw.scales_gate[ui] >= lw.expert_slab);
                CHECK(lw.scales_gate[ui] < lw.expert_slab + lw.slab_bytes);
            }
            // gate != up in the source fixture, so the packed tables differ.
            CHECK(lw.packed_gate[0] != lw.packed_up[0]);
        } else {
            CHECK(lw.mlp_gate.d_packed != nullptr);
            CHECK(lw.mlp_up.d_packed != nullptr);
            CHECK(lw.mlp_down.d_packed != nullptr);
            CHECK(lw.mlp_gate.rows == cfg.dense_inter);
            CHECK(lw.experts.empty());
            CHECK(lw.expert_slab == nullptr);
            CHECK(lw.d_row_table == nullptr);
        }
    }

    // Idempotent teardown (no leak on a second Free).
    pw.Free();
    pw.Free();
    CHECK(pw.layers.empty());
    CHECK(pw.resident_bytes == 0);
}

TEST_CASE("flint loader rejects a foreign index format", "[flint][loader]")
{
    Converted c = BuildConverted();
    // Corrupt the format tag: the loader must refuse rather than decode.
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
    FlintPinnedWeights pw;
    std::string error;
    CHECK_FALSE(LoadFlintDenseWeights(c.root.string(), pw, nullptr, error));
    CHECK(error.find("unsupported index format") != std::string::npos);
}

TEST_CASE("flint KV geometry is pos-major and size-neutral", "[flint][loader]")
{
    FlintTopology top;
    top.num_layers = 4;
    top.num_key_value_heads = 2;
    top.head_dim = 8;
    const lsxcommon::KvCacheGeometry g = FlintKvGeometry(top, 16);
    CHECK(g.pos_stride == 2 * 8);
    CHECK(g.head_stride == 8);
    CHECK(g.LayerStride() == 2 * 16 * 8);
    CHECK(g.Offset(1, 0, 3, 0) == g.LayerStride() + 3 * 16);
}
