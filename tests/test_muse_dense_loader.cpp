// Muse-Glimmer-30B loader test: pinned NVFP4 projections + FP16 embed/head/
// norms + the uniform per-layer KV cache geometry. Synthetic converted
// fixture only — the real vault belongs to the selfcheck/perf suites.
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <vector>

#include "test_muse_dense_fixture.h"

#include "lsxcommon/muse_dense_convert.h"
#include "lsxcommon/muse_dense_load.h"
#include "lsxcommon/schema.h"

namespace {

std::filesystem::path MakeConvertedSynthetic()
{
    namespace fs = std::filesystem;
    auto tmp = fs::temp_directory_path() / "muse_dense_loader_test";
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    synthtest::WriteSafetensors(tmp, "model.safetensors",
                                muse_dense_test::BuildSyntheticMuseDenseModel());
    muse_dense_test::WriteConfig(tmp);
    muse_dense_test::WriteTokenizerJson(tmp);
    auto out = tmp / "out";
    const auto st =
        lsxcommon::muse_dense::ConvertMuseDenseModel(tmp.string(), out.string());
    REQUIRE(st.ok());
    return out;
}

} // namespace

TEST_CASE("LoadMuseDensePinnedWeights layout", "[muse_dense][loader]") {
    auto out = MakeConvertedSynthetic();

    lsxcommon::muse_dense::MuseDensePinnedWeights pw;
    REQUIRE(lsxcommon::muse_dense::LoadMuseDensePinnedWeights(out.string(), pw,
                                                              nullptr) ==
            cudaSuccess);
    CHECK(pw.top.error_msg.empty());
    CHECK(pw.layers.size() == 4);
    CHECK(pw.vocab_size == 266);
    CHECK(pw.hidden == 64);
    CHECK(pw.resident_bytes > 0);

    // FP16 globals: embed + UNTIED lm_head + final norm (never quantized).
    CHECK(pw.d_embed != nullptr);
    CHECK(pw.d_lm_head != nullptr);
    CHECK(pw.d_embed != pw.d_lm_head);
    CHECK(pw.d_final_norm != nullptr);

    for (size_t l = 0; l < pw.layers.size(); ++l) {
        const auto &lw = pw.layers[l];
        INFO("layer " << l);
        CHECK(lw.d_arena != nullptr);
        CHECK(lw.arena_bytes > 0);
        const auto *base = static_cast<const std::uint8_t *>(lw.d_arena);
        const auto *end = base + lw.arena_bytes;
        // Every norm is an arena view; none of them is quantized.
        for (auto *p : {lw.d_input_norm, lw.d_post_attn_norm, lw.d_pre_ffn,
                        lw.d_post_ffn}) {
            REQUIRE(p != nullptr);
            CHECK(reinterpret_cast<const std::uint8_t *>(p) >= base);
            CHECK(reinterpret_cast<const std::uint8_t *>(p) < end);
        }

        // ALL EIGHT projections are NVFP4 triples — including the attention
        // gate, on EVERY layer (sliding AND full; no V-tying in this family).
        for (const auto *w : {&lw.attn_q, &lw.attn_k, &lw.attn_v, &lw.attn_o,
                              &lw.attn_gate, &lw.mlp_gate, &lw.mlp_up,
                              &lw.mlp_down}) {
            REQUIRE(w->d_packed != nullptr);
            CHECK(w->d_scales != nullptr);
        }
        CHECK(lw.attn_q.rows == pw.top.QRows());
        CHECK(lw.attn_q.cols == pw.top.hidden_size);
        CHECK(lw.attn_k.rows == pw.top.KvRows());
        CHECK(lw.attn_v.rows == pw.top.KvRows());
        CHECK(lw.attn_gate.rows == pw.top.QRows());
        CHECK(lw.attn_o.rows == pw.top.hidden_size);
        CHECK(lw.attn_o.cols == pw.top.QRows());
        CHECK(lw.mlp_gate.rows == pw.top.intermediate_size);
        CHECK(lw.mlp_down.rows == pw.top.hidden_size);
        // Packed + scales footprint of an NVFP4 [N, K] weight.
        const int64_t numel =
            static_cast<int64_t>(lw.mlp_gate.rows) * lw.mlp_gate.cols;
        CHECK(lw.mlp_gate.packed_bytes == numel / 2);
        CHECK(lw.mlp_gate.scale_bytes == numel / 16);
    }

    pw.Free();
    CHECK(pw.d_embed == nullptr);
    CHECK(pw.d_lm_head == nullptr);
    CHECK(pw.layers.empty());
    CHECK(pw.resident_bytes == 0);
    // Free() is idempotent.
    pw.Free();
}

TEST_CASE("muse dense KV cache: uniform pos-major geometry per layer",
          "[muse_dense][loader]") {
    auto out = MakeConvertedSynthetic();
    lsxcommon::muse_dense::MuseDensePinnedWeights pw;
    REQUIRE(lsxcommon::muse_dense::LoadMuseDensePinnedWeights(out.string(), pw,
                                                              nullptr) ==
            cudaSuccess);

    lsxcommon::muse_dense::MuseDenseKvCache kv;
    REQUIRE(kv.Allocate(pw, 32) == cudaSuccess);
    REQUIRE(kv.dk.size() == 4);
    REQUIRE(kv.geo.size() == 4);
    CHECK(kv.cache_bytes > 0);
    for (size_t l = 0; l < kv.geo.size(); ++l) {
        INFO("layer " << l);
        // Unlike gemma, EVERY layer stores the same 2-head x 16-dim geometry;
        // the sliding/full difference is the attention WINDOW, not the cache.
        CHECK(kv.geo[l].kv_heads == 2);
        CHECK(kv.geo[l].head_dim == 16);
        CHECK(kv.dk[l] != nullptr);
        CHECK(kv.dv[l] != nullptr);
    }
    kv.Free();
    CHECK(kv.dk.empty());
    CHECK(kv.cache_bytes == 0);
    pw.Free();
    std::filesystem::remove_all(out.parent_path());
}

TEST_CASE("muse dense loader refuses a missing layer file",
          "[muse_dense][loader]") {
    auto out = MakeConvertedSynthetic();
    std::filesystem::remove(out / "pinned" / "layer_2.arrow");
    lsxcommon::muse_dense::MuseDensePinnedWeights pw;
    CHECK(lsxcommon::muse_dense::LoadMuseDensePinnedWeights(out.string(), pw,
                                                            nullptr) !=
          cudaSuccess);
    pw.Free();
    std::filesystem::remove_all(out.parent_path());
}

TEST_CASE("muse dense loader refuses a missing global table",
          "[muse_dense][loader]") {
    auto out = MakeConvertedSynthetic();
    std::filesystem::remove(out / "pinned" / "global.arrow");
    lsxcommon::muse_dense::MuseDensePinnedWeights pw;
    CHECK(lsxcommon::muse_dense::LoadMuseDensePinnedWeights(out.string(), pw,
                                                            nullptr) !=
          cudaSuccess);
    pw.Free();
    std::filesystem::remove_all(out.parent_path());
}