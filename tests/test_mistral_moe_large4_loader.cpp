// Mistral-Large-4 loader test (Model Ingestion Protocol stage 3). Converts a
// synthetic checkpoint, uploads it through the real loader and checks the
// device-resident dense graph, the per-layer topology flags and the neutral
// expert store (dense layer 0 => first_moe_layer = 1). No vault access.

#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/mistral_moe_large4_convert.h"
#include "lsxcommon/mistral_moe_large4_load.h"
#include "test_mistral_moe_large4_fixture.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

namespace {
bool GpuAvailable()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
} // namespace

TEST_CASE("Mistral-Large-4 loader uploads the dense graph and expert store",
          "[mistral_large4][load]")
{
    if (!GpuAvailable()) std::exit(4);

    auto in = fs::temp_directory_path() / "mistral_large4_load_in";
    auto out = fs::temp_directory_path() / "mistral_large4_load_out";
    fs::remove_all(out);
    mistrallarge4test::Large4SynthGeom g;
    mistrallarge4test::WriteCheckpoint(in, g);
    REQUIRE(lsxcommon::mistral_large4::ConvertMistralLarge4Model(in.string(), out)
                .ok());

    lsxcommon::mistral_large4::Large4PinnedWeights pw;
    std::string err;
    const cudaError_t e = lsxcommon::mistral_large4::LoadMistralLarge4Weights(
        out.string(), pw, nullptr, err);
    INFO("load error: " << err);
    REQUIRE(e == cudaSuccess);

    REQUIRE(pw.dims.layers == g.layers);
    REQUIRE(pw.dims.experts == g.experts);
    REQUIRE(pw.dims.hidden == g.hidden);
    REQUIRE(pw.dims.vocab == g.vocab);
    REQUIRE(pw.dims.qk_head == g.qk_head());
    REQUIRE(pw.embed.rows == g.vocab);
    REQUIRE(pw.embed.cols == g.hidden);
    REQUIRE(pw.final_norm != nullptr);
    REQUIRE(pw.lerope_alpha != nullptr);
    REQUIRE(pw.resident_bytes > 0);

    for (int l = 0; l < g.layers; ++l)
    {
        const auto &lw = pw.layers[static_cast<size_t>(l)];
        INFO("layer " << l);
        REQUIRE(lw.uses_rope == g.rope(l));
        REQUIRE(lw.owns_kv == (g.kv_source(l) == l));
        REQUIRE(lw.full_indexer == g.full_indexer(l));
        REQUIRE(lw.is_moe == g.moe(l));
        REQUIRE(lw.kv_source == g.kv_source(l));
        REQUIRE(lw.wq.d_packed != nullptr);
        REQUIRE(lw.wq.rows == g.heads * g.qk_head());
        REQUIRE(lw.wq.cols == g.hidden);
        REQUIRE(lw.wo.rows == g.hidden);
        REQUIRE(lw.wo.cols == g.heads * g.v_head);
        REQUIRE(lw.wkv_b.data != nullptr);
        REQUIRE(lw.wkv_b.rows ==
                g.heads * ((g.rope(l) ? g.qk_nope : g.qk_head()) + g.v_head));
        REQUIRE(lw.wkv_b.cols == g.kv_lora);
        REQUIRE(lw.input_ln != nullptr);
        REQUIRE(lw.post_ln != nullptr);
        REQUIRE(lw.ssmax_s != nullptr);
        REQUIRE(lw.ssmax_b != nullptr);
        if (lw.owns_kv)
        {
            REQUIRE(lw.wkv_a.data != nullptr);
            REQUIRE(lw.wkv_a.rows ==
                    g.kv_lora + (g.rope(l) ? g.qk_rope : 0));
            REQUIRE(lw.kv_a_norm != nullptr);
        }
        else
        {
            REQUIRE(lw.wkv_a.data == nullptr);
            REQUIRE(lw.kv_a_norm == nullptr);
        }
        if (lw.full_indexer)
        {
            REQUIRE(lw.indexer_wq_b.d_packed != nullptr);
            REQUIRE(lw.indexer_wq_b.rows == g.index_heads * g.index_head_dim);
            REQUIRE(lw.indexer_wk.data != nullptr);
            REQUIRE(lw.indexer_wk.rows == g.index_head_dim);
            REQUIRE(lw.indexer_weights_proj.rows == g.index_heads);
            REQUIRE(lw.idx_q_norm != nullptr);
            REQUIRE(lw.idx_k_norm_bias != nullptr);
            REQUIRE(lw.idx_ssmax_s != nullptr);
        }
        else
        {
            REQUIRE(lw.indexer_wq_b.d_packed == nullptr);
            REQUIRE(lw.indexer_wk.data == nullptr);
        }
        if (lw.is_moe)
        {
            REQUIRE(lw.router != nullptr);
            REQUIRE(lw.shared_gate.d_packed != nullptr);
            REQUIRE(lw.shared_gate.rows == g.moe_inter);
            REQUIRE(lw.shared_down.rows == g.hidden);
            REQUIRE(lw.dense_gate.d_packed == nullptr);
        }
        else
        {
            REQUIRE(lw.router == nullptr);
            REQUIRE(lw.shared_gate.d_packed == nullptr);
            REQUIRE(lw.dense_gate.d_packed != nullptr);
            REQUIRE(lw.dense_gate.rows == g.dense_inter);
        }
    }

    // Neutral store geometry: the dense prefix is skipped, the 3 MoE layers
    // are present with 2 experts each.
    REQUIRE(pw.expert_store.num_layers == g.layers);
    REQUIRE(pw.expert_store.num_experts == g.experts);
    REQUIRE(pw.expert_store.experts.first_moe_layer == 1);
    // Layer 0 is dense: its records stay zero-sized, the MoE layers carry the
    // real geometry.
    for (const auto &r : pw.expert_store.records[0]) REQUIRE(r.w1_bytes == 0);
    REQUIRE(pw.expert_store.records[1][0].w1_bytes > 0);

    pw.Free();
    REQUIRE(pw.layers.empty());
}
