// MiniMax-M3.1-Flash loader stage (Model Ingestion Protocol stage 3). The
// sibling reuses the parent loader verbatim: the emitted artifact layout is
// identical, so what this suite pins is that (a) the MSA indexer weights are
// uploaded on EVERY layer (the M3 tree leaves layers 0-2 without an indexer),
// (b) the topology reports the derived dispatch boundaries the sibling's
// passport cross-checks, and (c) the routed-expert store still aliases the
// neutral mmap ExpertStore with the verbatim NVFP4 reciprocal global scale.
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <cmath>
#include <filesystem>
#include <string>

#include "lsxcommon/minimax_moe_flash_convert.h"
#include "lsxcommon/minimax_moe_inference_pipeline.h"
#include "lsxcommon/minimax_moe_load.h"
#include "test_minimax_moe_flash_fixture.h"

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

namespace fs = std::filesystem;
using namespace lsxcommon::minimax;

namespace {

struct Converted {
    fs::path root;
    MinimaxTopology top;
};

Converted BuildConverted()
{
    Converted c;
    fs::path in = fs::temp_directory_path() / "lsx_minimax_flash_load_in";
    c.root = fs::temp_directory_path() / "lsx_minimax_flash_load_out";
    fs::remove_all(in);
    fs::remove_all(c.root);
    REQUIRE(minimax_flash_test::BuildSyntheticCheckpoint(in) > 0);
    auto st = ConvertMinimaxFlashModel(in.string(), c.root.string());
    if (!st.ok()) WARN(st.ToString());
    REQUIRE(st.ok());
    c.top = LoadMinimaxTopology(c.root.string());
    c.top.error_msg.clear();
    return c;
}

} // namespace

TEST_CASE("M3.1 topology derives the dispatch boundaries the passport checks",
          "[minimax-flash][loader]")
{
    const Converted c = BuildConverted();
    const auto &cfg = minimax_flash_test::TestCfg();
    REQUIRE(c.top.num_layers == cfg.layers);
    CHECK(c.top.first_moe_layer == cfg.first_moe_layer);
    CHECK(c.top.first_sparse_layer == 0);
    CHECK(c.top.rope_theta == 10000000.0);
    CHECK(c.top.rms_norm_eps == 1e-6);
    for (int l = 0; l < cfg.layers; ++l)
        CHECK(c.top.layer_types[static_cast<size_t>(l)] ==
              "sparse_attention");
}

TEST_CASE("M3.1 loader uploads the indexer on every layer", "[minimax-flash][loader]")
{
    const Converted c = BuildConverted();
    const auto &cfg = minimax_flash_test::TestCfg();

    MinimaxPinnedWeights pw;
    std::string error;
    REQUIRE(LoadMinimaxDenseWeights(c.root.string(), pw, nullptr, error));
    CHECK(error.empty());
    CHECK(pw.resident_bytes > 0);
    REQUIRE(pw.layers.size() == static_cast<size_t>(cfg.layers));
    for (int l = 0; l < cfg.layers; ++l) {
        const auto &lw = pw.layers[static_cast<size_t>(l)];
        CHECK(lw.attn.loaded);
        CHECK(lw.attn.q.d_packed != nullptr);
        // Sparse attention throughout: layer 0 carries the MSA indexer too.
        CHECK(lw.attn.index_q.d_packed != nullptr);
        CHECK(lw.attn.index_k.d_packed != nullptr);
        CHECK(lw.attn.d_index_q_norm != nullptr);
        CHECK(lw.attn.d_index_k_norm != nullptr);
        const bool moe = l >= cfg.first_moe_layer;
        CHECK(lw.is_moe == moe);
        if (moe) {
            CHECK(lw.d_router != nullptr);
            CHECK(lw.d_router_bias != nullptr);
            CHECK(lw.shared_gate.d_packed != nullptr);
        } else {
            CHECK(lw.d_router == nullptr);
            CHECK(lw.mlp_gate.d_packed != nullptr);
        }
    }
    CHECK(pw.d_embed != nullptr);
    CHECK(pw.d_head != nullptr);
    pw.Free();
    pw.Free();
    CHECK(pw.layers.empty());
}

TEST_CASE("M3.1 routed experts stay the neutral mmap store",
          "[minimax-flash][loader]")
{
    const Converted c = BuildConverted();
    const auto &cfg = minimax_flash_test::TestCfg();

    MinimaxExpertStore experts;
    std::string error;
    REQUIRE(LoadMinimaxExpertStore(c.root.string(), experts, error));
    CHECK(experts.loaded);
    CHECK(experts.first_moe_layer == cfg.first_moe_layer);
    CHECK(experts.db.first_moe_layer == cfg.first_moe_layer);
    const std::int64_t per =
        static_cast<std::int64_t>(cfg.moe_inter) * cfg.hidden;
    for (int l = cfg.first_moe_layer; l < cfg.layers; ++l) {
        const auto &rec = experts.records[static_cast<size_t>(l)];
        REQUIRE(rec.size() == static_cast<size_t>(cfg.experts));
        CHECK(rec[0].w1_bytes == per / 2);
        CHECK(rec[0].s1_bytes == per / 16);
        // Verbatim ModelOpt record: the divisor global scale was inverted.
        CHECK(rec[0].gs1 > 0.0f);
        CHECK(rec[0].gs1 < 1.0f);
    }

    lsxcommon::gpu::ExpertDeviceCache cache;
    REQUIRE(AllocateMinimaxExpertCache(experts, cache, 4, cfg.top_k, error));
    const int slot = EnsureMinimaxExpertResident(
        experts, cache, cfg.first_moe_layer, 3, nullptr, error, nullptr,
        nullptr);
    INFO(error);
    REQUIRE(slot >= 0);
    REQUIRE(cudaStreamSynchronize(nullptr) == cudaSuccess);
    FreeMinimaxExpertCache(cache);
    experts.Free();
}
