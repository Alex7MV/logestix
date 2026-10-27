// MiniMax-M3 loader stage (Model Ingestion Protocol stage 3): the pinned
// dense graph, the mmap routed-expert store (R1/R2: the neutral
// MoEExpertStore, never a family container) and the device slot LRU.
// Runs on a synthetic converted tree; no vault, no real weights.
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "lsxcommon/minimax_moe_convert.h"
#include "lsxcommon/minimax_moe_inference_pipeline.h"
#include "lsxcommon/minimax_moe_load.h"
#include "test_minimax_moe_fixture.h"

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
    fs::path in = fs::temp_directory_path() / "lsx_minimax_load_in";
    c.root = fs::temp_directory_path() / "lsx_minimax_load_out";
    fs::remove_all(in);
    fs::remove_all(c.root);
    const fs::path tok = fs::path(LSX_SOURCE_DIR) / "tests" / "data" /
                         "synthetic_minimax_tok.json";
    REQUIRE(minimax_test::BuildSyntheticCheckpoint(in, true, tok) > 0);
    auto st = ConvertMinimaxMoeModel(in.string(), c.root.string());
    if (!st.ok()) WARN(st.ToString());
    REQUIRE(st.ok());
    c.top = LoadMinimaxTopology(c.root.string());
    c.top.error_msg.clear();
    return c;
}

} // namespace

TEST_CASE("MiniMax loader uploads the pinned graph", "[minimax][loader]")
{
    const Converted c = BuildConverted();
    const auto &cfg = minimax_test::TestCfg();
    REQUIRE(c.top.num_layers == cfg.layers);

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
        CHECK(lw.attn.k.d_packed != nullptr);
        CHECK(lw.attn.v.d_packed != nullptr);
        CHECK(lw.attn.o.d_packed != nullptr);
        CHECK(lw.attn.d_input_norm != nullptr);
        CHECK(lw.attn.d_post_norm != nullptr);
        CHECK(lw.attn.d_q_norm != nullptr);
        CHECK(lw.attn.d_k_norm != nullptr);
        CHECK(lw.attn.q.rows == cfg.heads * cfg.head_dim);
        CHECK(lw.attn.q.cols == cfg.hidden);
        CHECK(lw.attn.k.rows == cfg.kv_heads * cfg.head_dim);
        const bool moe = l >= cfg.first_moe_layer;
        CHECK(lw.is_moe == moe);
        if (moe) {
            CHECK(lw.attn.index_q.d_packed != nullptr);
            CHECK(lw.attn.index_k.d_packed != nullptr);
            CHECK(lw.attn.d_index_q_norm != nullptr);
            CHECK(lw.d_router != nullptr);
            CHECK(lw.router_bytes ==
                  static_cast<std::int64_t>(cfg.experts) * cfg.hidden * 4);
            CHECK(lw.d_router_bias != nullptr);
            CHECK(lw.shared_gate.d_packed != nullptr);
            CHECK(lw.shared_up.d_packed != nullptr);
            CHECK(lw.shared_down.d_packed != nullptr);
            CHECK(lw.shared_gate.rows == cfg.shared_inter);
            CHECK(lw.shared_gate.cols == cfg.hidden);
        } else {
            CHECK(lw.attn.index_q.d_packed == nullptr);
            CHECK(lw.d_router == nullptr);
            CHECK(lw.mlp_gate.d_packed != nullptr);
            CHECK(lw.mlp_up.d_packed != nullptr);
            CHECK(lw.mlp_down.d_packed != nullptr);
            CHECK(lw.mlp_gate.rows == cfg.dense_inter);
        }
    }
    CHECK(pw.embed.d_packed != nullptr);
    CHECK(pw.head.d_packed != nullptr);
    CHECK(pw.d_final_norm != nullptr);
    CHECK(pw.head.rows == cfg.vocab);

    // Teardown must be complete and idempotent (a leak shows up as OOM on the
    // next model load, not here).
    pw.Free();
    pw.Free();
    CHECK(pw.layers.empty());
    CHECK(pw.resident_bytes == 0);
    CHECK(pw.embed.d_packed == nullptr);
}

TEST_CASE("MiniMax routed-expert store is the neutral mmap store",
          "[minimax][loader]")
{
    const Converted c = BuildConverted();
    const auto &cfg = minimax_test::TestCfg();

    MinimaxExpertStore experts;
    std::string error;
    REQUIRE(LoadMinimaxExpertStore(c.root.string(), experts, error));
    CHECK(error.empty());
    CHECK(experts.loaded);
    CHECK(experts.num_layers == cfg.layers);
    CHECK(experts.num_experts == cfg.experts);
    CHECK(experts.first_moe_layer == cfg.first_moe_layer);
    CHECK(experts.db.first_moe_layer == cfg.first_moe_layer);
    REQUIRE(experts.records.size() == static_cast<size_t>(cfg.layers));
    for (int l = 0; l < cfg.layers; ++l) {
        const auto &rec = experts.records[static_cast<size_t>(l)];
        REQUIRE(rec.size() == static_cast<size_t>(cfg.experts));
        if (l < cfg.first_moe_layer) continue;
        const std::int64_t per = static_cast<std::int64_t>(cfg.moe_inter) *
                                 cfg.hidden;
        CHECK(rec[0].w1_bytes == per / 2);
        CHECK(rec[0].w2_bytes == per / 2);
        CHECK(rec[0].w3_bytes == per / 2);
        CHECK(rec[0].s1_bytes == per / 16);
        CHECK(rec[0].gs1 > 0.0f);
    }

    // Device slot cache sized for the fixture, then one staged expert.
    lsxcommon::gpu::ExpertDeviceCache cache;
    REQUIRE(AllocateMinimaxExpertCache(experts, cache, 4, cfg.top_k, error));
    // The decode working-set floor (4 x top-k) may raise the request.
    CHECK(cache.slots >= 4);
    CHECK(cache.d_pack != nullptr);
    CHECK(cache.d_scale != nullptr);
    CHECK(cache.pack_stride == 3 * (static_cast<std::int64_t>(cfg.moe_inter) *
                                    cfg.hidden / 2));
    const int slot = EnsureMinimaxExpertResident(
        experts, cache, cfg.first_moe_layer, 3, nullptr, error, nullptr,
        nullptr);
    INFO(error);
    REQUIRE(slot >= 0);
    CHECK(cache.h_slot_owner[slot] == cfg.first_moe_layer * cfg.experts + 3);
    // Staging is async on the stream it was given (nullptr = default).
    REQUIRE(cudaStreamSynchronize(nullptr) == cudaSuccess);

    FreeMinimaxExpertCache(cache);
    experts.Free();
}

TEST_CASE("MiniMax loader rejects a foreign index format", "[minimax][loader]")
{
    const Converted c = BuildConverted();
    const fs::path idx = c.root / "logestix" / "model.index.json";
    {
        std::ofstream f(idx);
        f << "{\"format\":\"logestix-qwen35moe\",\"num_layers\":4,"
             "\"num_experts\":8,\"layer_types\":[\"dense_attention\","
             "\"sparse_attention\",\"sparse_attention\",\"sparse_attention\"],"
             "\"mlp_layer_types\":[\"dense\",\"sparse\",\"sparse\",\"sparse\"]}";
    }
    MinimaxPinnedWeights pw;
    std::string error;
    CHECK_FALSE(LoadMinimaxDenseWeights(c.root.string(), pw, nullptr, error));
    CHECK(error.find("unsupported index format") != std::string::npos);
}
