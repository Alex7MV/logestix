// Gemma-4-31B-it (dense) loader test: pinned NVFP4 projections + FP16 norms +
// the per-layer KV cache geometry. Synthetic converted fixture only — the real
// vault belongs to the selfcheck/perf suites.
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <vector>

#include "test_gemma_dense_fixture.h"

#include "lsxcommon/gemma_dense_convert.h"
#include "lsxcommon/gemma_dense_load.h"
#include "lsxcommon/schema.h"

namespace {

std::filesystem::path MakeConvertedSynthetic()
{
    namespace fs = std::filesystem;
    auto tmp = fs::temp_directory_path() / "gemma4dense_loader_test";
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    synthtest::WriteSafetensors(tmp, "model.safetensors",
                                gemma_dense_test::BuildSyntheticGemmaDenseModel());
    gemma_dense_test::WriteConfig(tmp);
    fs::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_gemma_tok.json",
                  tmp / "tokenizer.json");
    auto out = tmp / "out";
    const auto st =
        lsxcommon::gemma_dense::ConvertGemmaDenseModel(tmp.string(), out.string());
    REQUIRE(st.ok());
    return out;
}

} // namespace

TEST_CASE("LoadGemmaDensePinnedWeights layout", "[gemma_dense][loader]") {
    auto out = MakeConvertedSynthetic();

    lsxcommon::gemma_dense::GemmaDensePinnedWeights pw;
    REQUIRE(lsxcommon::gemma_dense::LoadGemmaDensePinnedWeights(out.string(), pw,
                                                               nullptr) ==
            cudaSuccess);
    CHECK(pw.top.error_msg.empty());
    CHECK(pw.layers.size() == 4);
    CHECK(pw.d_final_norm != nullptr);
    CHECK(pw.vocab_size == 64);
    CHECK(pw.hidden == 64);
    CHECK(pw.resident_bytes > 0);

    // Tied embed/lm_head is ONE NVFP4 table.
    CHECK(pw.embed.d_packed != nullptr);
    CHECK(pw.embed.d_scales != nullptr);
    CHECK(pw.embed.rows == 64);
    CHECK(pw.embed.cols == 64);
    CHECK(pw.embed.packed_bytes == 64 * 64 / 2);
    CHECK(pw.embed.scale_bytes == 64 * 64 / 16);

    for (size_t l = 0; l < pw.layers.size(); ++l) {
        const auto &lw = pw.layers[l];
        INFO("layer " << l);
        CHECK(lw.d_arena != nullptr);
        CHECK(lw.arena_bytes > 0);
        const auto *base = static_cast<const std::uint8_t *>(lw.d_arena);
        const auto *end = base + lw.arena_bytes;
        // Every norm is an arena view; none of them is quantized.
        for (auto *p : {lw.d_input_norm, lw.d_post_norm, lw.d_pre_ffn,
                        lw.d_post_ffn, lw.d_q_norm, lw.d_k_norm}) {
            REQUIRE(p != nullptr);
            CHECK(reinterpret_cast<const std::uint8_t *>(p) >= base);
            CHECK(reinterpret_cast<const std::uint8_t *>(p) < end);
        }
        // layer_scalar: BF16 in the checkpoint, uploaded as F32 with a host
        // mirror for the hot loop.
        CHECK(lw.d_layer_scalar != nullptr);
        REQUIRE(l < pw.h_layer_scalar.size());
        CHECK(pw.h_layer_scalar[l] == 1.0f);

        // All seven projections are NVFP4 triples.
        CHECK(lw.attn_q.d_packed != nullptr);
        CHECK(lw.attn_k.d_packed != nullptr);
        CHECK(lw.attn_o.d_packed != nullptr);
        CHECK(lw.mlp_gate.d_packed != nullptr);
        CHECK(lw.mlp_up.d_packed != nullptr);
        CHECK(lw.mlp_down.d_packed != nullptr);

        const bool full = pw.top.IsFull(static_cast<int>(l));
        if (full) {
            // attention_k_eq_v: no v projection, and the KV rows are the
            // GLOBAL geometry.
            CHECK(lw.attn_v.d_packed == nullptr);
            CHECK(lw.HasV() == false);
            CHECK(lw.attn_q.rows == pw.top.QRows(static_cast<int>(l)));
            CHECK(lw.attn_k.rows == pw.top.KvRows(static_cast<int>(l)));
        } else {
            CHECK(lw.attn_v.d_packed != nullptr);
            CHECK(lw.HasV() == true);
        }
        CHECK(lw.attn_q.cols == pw.top.hidden_size);
        CHECK(lw.mlp_gate.rows == pw.top.intermediate_size);
        CHECK(lw.mlp_down.rows == pw.top.hidden_size);
        // Packed + scales footprint of an NVFP4 [N, K] weight.
        const int64_t numel =
            static_cast<int64_t>(lw.mlp_gate.rows) * lw.mlp_gate.cols;
        CHECK(lw.mlp_gate.packed_bytes == numel / 2);
        CHECK(lw.mlp_gate.scale_bytes == numel / 16);
    }

    pw.Free();
    CHECK(pw.embed.d_packed == nullptr);
    CHECK(pw.layers.empty());
    CHECK(pw.resident_bytes == 0);
}

TEST_CASE("KV cache carries the per-layer geometry of both layer kinds",
          "[gemma_dense][loader]") {
    auto out = MakeConvertedSynthetic();
    lsxcommon::gemma_dense::GemmaDensePinnedWeights pw;
    REQUIRE(lsxcommon::gemma_dense::LoadGemmaDensePinnedWeights(out.string(), pw,
                                                               nullptr) ==
            cudaSuccess);

    lsxcommon::gemma_dense::GemmaDenseKvCache kv;
    const int slots = 32;
    REQUIRE(kv.Allocate(pw, slots) == cudaSuccess);
    CHECK(kv.max_total == slots);
    CHECK(kv.cache_bytes > 0);
    REQUIRE(kv.geo.size() == 4);

    for (size_t l = 0; l < kv.geo.size(); ++l) {
        const int layer = static_cast<int>(l);
        const auto &g = kv.geo[l];
        INFO("layer " << l);
        CHECK(g.capacity == slots);
        CHECK(g.kv_heads == pw.top.KvHeads(layer));
        CHECK(g.head_dim == pw.top.HeadDim(layer));
        CHECK(g.pos_stride == pw.top.KvRows(layer));   // pos-major
        CHECK(g.head_stride == pw.top.HeadDim(layer));
        CHECK(g.LayerStride() ==
              pw.top.KvRows(layer) * slots);          // layout-independent size
        // The two kinds MUST differ, or the sliding/global distinction is not
        // represented in the cache at all.
        CHECK(g.kv_heads * g.head_dim == g.pos_stride);
        if (pw.top.IsFull(layer))
            CHECK(g.kv_heads == pw.top.num_global_key_value_heads);
        else
            CHECK(g.kv_heads == pw.top.num_key_value_heads);
    }
    CHECK(kv.geo[0].LayerStride() != kv.geo[3].LayerStride());

    kv.Free();
    CHECK(kv.dk.empty());
    CHECK(kv.max_total == 0);
    pw.Free();
}

TEST_CASE("loader refuses a checkpoint whose layer is missing a projection",
          "[gemma_dense][loader]") {
    namespace fs = std::filesystem;
    auto out = MakeConvertedSynthetic();
    // Drop the o_proj triple from layer 1: the loader must fail loudly instead
    // of running a layer with a null weight pointer.
    const auto path = out / "pinned" / "layer_1.arrow";
    {
        auto batch_res = lsxcommon::LoadTensorBatchViews(path);
        REQUIRE(batch_res.ok());
        std::vector<lsxcommon::TensorRecord> kept;
        for (const auto &t : batch_res->tensors) {
            if (t.name.find("self_attn.o_proj.weight") != std::string::npos)
                continue;
            lsxcommon::TensorRecord r;
            r.name = t.name;
            r.dtype = "U8";
            r.shape.assign(t.shape, t.shape + t.rank);
            r.data.assign(t.data, t.data + t.bytes);
            kept.push_back(std::move(r));
        }
        batch_res->keepalive.reset();
        auto batch = lsxcommon::PinnedSchema::MakeBatch(kept);
        REQUIRE(batch.ok());
        const auto st = lsxcommon::WriteRecordBatchToFile(*batch, path);
        REQUIRE(st.ok());
    }
    lsxcommon::gemma_dense::GemmaDensePinnedWeights pw;
    CHECK(lsxcommon::gemma_dense::LoadGemmaDensePinnedWeights(out.string(), pw,
                                                             nullptr) !=
          cudaSuccess);
    pw.Free();
    fs::remove_all(out.parent_path());
}
