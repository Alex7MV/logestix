// Gemma-4 loader test: BF16 raw passthrough (no conversion) + arena + LRU.
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <vector>

#include "test_gemma_moe_fixture.h"

#include "lsxcommon/gemma_moe_convert.h"
#include "lsxcommon/gemma_moe_load.h"

namespace {

std::filesystem::path MakeConvertedSynthetic()
{
    namespace fs = std::filesystem;
    auto tmp = fs::temp_directory_path() / "gemma4_loader_test";
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    auto tensors = gemmatest::BuildSyntheticGemmaModel();
    gemmatest::WriteSafetensors(tmp, "model.safetensors", tensors);
    gemmatest::WriteConfig(tmp);
    std::filesystem::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_gemma_tok.json",
                               tmp / "tokenizer.json");
    auto out = tmp / "out";
    REQUIRE(lsxcommon::gemma::ConvertGemmaModel(tmp.string(), out.string())
                .ok());
    return out;
}

} // namespace

TEST_CASE("LoadGemmaPinnedWeights arena layout", "[gemma][loader]") {
    auto out = MakeConvertedSynthetic();

    lsxcommon::gemma::GemmaPinnedWeights pw;
    REQUIRE(lsxcommon::gemma::LoadGemmaPinnedWeights(out.string(), pw, nullptr) ==
            cudaSuccess);
    CHECK(pw.layers.size() == 4);
    CHECK((pw.d_embed != nullptr || pw.embed_nvfp4.d_packed != nullptr));
    CHECK(pw.d_final_norm != nullptr);
    CHECK(pw.vocab_size == 64);
    CHECK(pw.hidden == 64);
    CHECK(pw.resident_bytes > 0);

    for (size_t l = 0; l < pw.layers.size(); ++l) {
        const auto &lw = pw.layers[l];
        INFO("layer " << l);
        CHECK(lw.d_arena != nullptr);
        CHECK(lw.arena_bytes > 0);
        // Arena contract: every half pointer lives inside the allocation.
        const void *base = lw.d_arena;
        const void *end = static_cast<const uint8_t *>(base) + lw.arena_bytes;
        // Attention projections are NVFP4 now (not arena-backed); norms and
        // the router projection remain arena halves.
        for (auto *p : {lw.d_input_norm, lw.d_post_norm,
                        lw.d_router_proj}) {
            REQUIRE(p != nullptr);
            CHECK(static_cast<const void *>(p) >= base);
            CHECK(static_cast<const void *>(p) < end);
        }
        // F32 side allocations are separate from the arena. BF16-stored
        // scalars must be converted and uploaded (regression: BF16
        // layer_scalar / per_expert_scale used to be silently skipped).
        CHECK(lw.d_layer_scalar != nullptr);
        CHECK(lw.d_per_expert_scale != nullptr);
        CHECK(lw.d_router_scale != nullptr);
        if (lw.d_layer_scalar && lw.d_per_expert_scale) {
            float scalar = 0.f;
            REQUIRE(cudaMemcpy(&scalar, lw.d_layer_scalar, sizeof(float),
                               cudaMemcpyDeviceToHost) == cudaSuccess);
            CHECK(scalar == 1.0f);
            float pes0 = 0.f;
            REQUIRE(cudaMemcpy(&pes0, lw.d_per_expert_scale, sizeof(float),
                               cudaMemcpyDeviceToHost) == cudaSuccess);
            CHECK(pes0 == 0.5f);   // fixture pattern: 0.5 + 0.01 * e
        }
        // Full-attention layer (3) carries global geometry.
        bool full = (l == 3);
        CHECK(lw.q_rows == (full ? 4 * 32 : 4 * 16));
        CHECK(lw.kv_rows == (full ? 1 * 32 : 2 * 16));
    }
    pw.Free();
    CHECK(pw.layers.empty());
    std::filesystem::remove_all(out.parent_path());
}

TEST_CASE("gemma expert store resolves all synthetic experts",
          "[gemma][loader]") {
    auto out = MakeConvertedSynthetic();

    auto db = lsxcommon::gemma::load_gemma_expert_index(
        out.string(), /*num_layers=*/4, /*experts_per_layer=*/8);
    REQUIRE(db.error_msg.empty());
    for (int l = 0; l < 4; ++l)
        for (int e = 0; e < 8; ++e) {
            lsxcommon::gemma::ensure_gemma_expert_resident(db, l, e, nullptr);
            const auto &ent =
                db.entries[static_cast<size_t>(l) * 8 +
                           static_cast<size_t>(e)];
            INFO("layer " << l << " expert " << e);
            // w1=gate [MI,H] -> packed bytes MI*H/2
            CHECK(ent.host_w1 != nullptr);
            CHECK(ent.w1_bytes == static_cast<int64_t>(32 * 64 / 2));
            CHECK(ent.host_w3 != nullptr);
            CHECK(ent.w3_bytes == static_cast<int64_t>(32 * 64 / 2));
            CHECK(ent.host_w2 != nullptr);
            CHECK(ent.w2_bytes == static_cast<int64_t>(64 * 32 / 2));
            CHECK(ent.w1_scales_bytes > 0);
            CHECK(ent.w1_global_scale > 0.f);
            CHECK(ent.w2_global_scale > 0.f);
        }
    lsxcommon::gemma::free_gemma_expert_index(db);

    std::filesystem::remove_all(out.parent_path());
}

TEST_CASE("gemma expert store parse counters", "[gemma][loader]") {
    auto out = MakeConvertedSynthetic();

    auto db = lsxcommon::gemma::load_gemma_expert_index(
        out.string(), /*num_layers=*/4, /*experts_per_layer=*/8);
    REQUIRE(db.error_msg.empty());
    // Warm-up parses every file exactly once.
    CHECK(db.parse_calls == 32);
    CHECK(db.parse_ok == 32);

    // A resident entry must not trigger a re-parse.
    const int64_t after_load = db.parse_calls;
    lsxcommon::gemma::ensure_gemma_expert_resident(db, 2, 5, nullptr);
    CHECK(db.parse_calls == after_load);

    lsxcommon::gemma::free_gemma_expert_index(db);
    std::filesystem::remove_all(out.parent_path());
}


// Residency budget (qwen 17348a1 analogy): all-experts target bounded by
// free VRAM with the unified 4 GiB headroom reserve
// (lifecycle::kExpertCacheVramReserveBytes, resource_governor.h); explicit
// env override wins.
TEST_CASE("gemma resident slot budget: VRAM-bounded, clamped, env-override",
          "[gemma][loader][cache]")
{
    using lsxcommon::gemma::ComputeGemmaResidentSlots;
    const size_t MiB = 1024ull * 1024ull;
    const size_t per_expert = 5 * MiB;
    const size_t reserve = lsxcommon::lifecycle::kExpertCacheVramReserveBytes;

    // Ample VRAM: every expert resident.
    CHECK(ComputeGemmaResidentSlots(32ull * 1024 * MiB, per_expert, 3840)
          == 3840);
    // Exactly the pool fits: boundary inclusive.
    CHECK(ComputeGemmaResidentSlots(reserve + 3840ull * per_expert,
                                    per_expert, 3840) == 3840);
    // Tight: (free - unified reserve) / per_expert slots.
    CHECK(ComputeGemmaResidentSlots(reserve + 1000ull * per_expert,
                                    per_expert, 3840) == 1000);
    // Under-reserve floor: nothing fits -> caller applies its own fallback.
    CHECK(ComputeGemmaResidentSlots(2ull * 1024 * MiB, per_expert, 3840) == 0);
    // Env override wins and is clamped to the pool.
    CHECK(ComputeGemmaResidentSlots(32ull * 1024 * MiB, per_expert, 3840,
                                    512) == 512);
    CHECK(ComputeGemmaResidentSlots(32ull * 1024 * MiB, per_expert, 3840,
                                    9999) == 3840);
}

TEST_CASE("gemma loader uploads NVFP4 dense MLP when present",
          "[gemma][loader][nvfp4]") {
    namespace fs = std::filesystem;
    auto tmp = fs::temp_directory_path() / "gemma4_loader_mlp_nvfp4";
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    auto tensors = gemmatest::BuildSyntheticGemmaModel();
    gemmatest::WriteSafetensors(tmp, "model.safetensors", tensors);
    gemmatest::WriteConfig(tmp);
    std::filesystem::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_gemma_tok.json",
                               tmp / "tokenizer.json");
    auto out = tmp / "out";
    REQUIRE(lsxcommon::gemma::ConvertGemmaModel(tmp.string(), out.string())
                .ok());

    lsxcommon::gemma::GemmaPinnedWeights pw;
    auto rc = lsxcommon::gemma::LoadGemmaPinnedWeights(out.string(), pw,
                                                       nullptr);
    if (rc != cudaSuccess) {
        INFO("LoadGemmaPinnedWeights rc=" << cudaGetErrorString(rc));
        REQUIRE(rc == cudaSuccess);
    }
    REQUIRE(!pw.layers.empty());
    const auto &lw = pw.layers[0];
    REQUIRE(lw.mlp_nvfp4);
    REQUIRE(lw.mlp_gate.d_packed != nullptr);
    CHECK(lw.mlp_gate.rows == gemmatest::TestCfg().intermediate_size);
    CHECK(lw.mlp_gate.cols == gemmatest::TestCfg().hidden_size);
    CHECK(lw.mlp_down.rows == gemmatest::TestCfg().hidden_size);
    CHECK(lw.mlp_down.cols == gemmatest::TestCfg().intermediate_size);
    CHECK(lw.d_mlp_gate == nullptr);  // BF16 path not populated
    CHECK(lw.d_mlp_up == nullptr);
    CHECK(lw.d_mlp_down == nullptr);
    pw.Free();
    fs::remove_all(tmp);
}

TEST_CASE("gemma loader uploads the attn + dense-MLP residual planes",
          "[gemma][loader][nvfp4][residual]") {
    namespace fs = std::filesystem;
    auto tmp = fs::temp_directory_path() / "gemma4_loader_residual";
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    auto tensors = gemmatest::BuildSyntheticGemmaModel();
    gemmatest::WriteSafetensors(tmp, "model.safetensors", tensors);
    gemmatest::WriteConfig(tmp);
    std::filesystem::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_gemma_tok.json",
                               tmp / "tokenizer.json");
    auto out = tmp / "out";
    REQUIRE(lsxcommon::gemma::ConvertGemmaModel(tmp.string(), out.string())
                .ok());

    lsxcommon::gemma::GemmaPinnedWeights pw;
    auto rc = lsxcommon::gemma::LoadGemmaPinnedWeights(out.string(), pw,
                                                       nullptr);
    if (rc != cudaSuccess) {
        INFO("LoadGemmaPinnedWeights rc=" << cudaGetErrorString(rc));
        REQUIRE(rc == cudaSuccess);
    }
    REQUIRE(pw.layers.size() ==
            static_cast<size_t>(gemmatest::TestCfg().num_layers));
    const auto &lw = pw.layers[0];
    REQUIRE(lw.attn_nvfp4);
    REQUIRE(lw.mlp_nvfp4);
    // Same shape as the base plane: the decode accumulates plane two into the
    // same output buffer with acc=true.
    CHECK(lw.attn_q_res.d_packed != nullptr);
    CHECK(lw.attn_q_res.rows == lw.attn_q.rows);
    CHECK(lw.attn_q_res.cols == lw.attn_q.cols);
    CHECK(lw.attn_k_res.rows == lw.attn_k.rows);
    CHECK(lw.attn_o_res.rows == lw.attn_o.rows);
    CHECK(lw.attn_o_res.cols == lw.attn_o.cols);
    CHECK(lw.mlp_gate_res.rows == lw.mlp_gate.rows);
    CHECK(lw.mlp_up_res.rows == lw.mlp_up.rows);
    CHECK(lw.mlp_down_res.rows == lw.mlp_down.rows);
    CHECK(lw.mlp_down_res.cols == lw.mlp_down.cols);
    // Sliding layers ship a v_proj; full layers tie V to K and have none.
    CHECK(lw.attn_v_res.d_packed != nullptr);
    CHECK(pw.layers[3].attn_v_res.d_packed == nullptr);
    pw.Free();
    fs::remove_all(tmp);
}

TEST_CASE("gemma loader uploads NVFP4 attention projections",
          "[gemma][loader][nvfp4]") {
    namespace fs = std::filesystem;
    auto tmp = fs::temp_directory_path() / "gemma4_loader_attn_nvfp4";
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    auto tensors = gemmatest::BuildSyntheticGemmaModel();
    gemmatest::WriteSafetensors(tmp, "model.safetensors", tensors);
    gemmatest::WriteConfig(tmp);
    std::filesystem::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_gemma_tok.json",
                               tmp / "tokenizer.json");
    auto out = tmp / "out";
    REQUIRE(lsxcommon::gemma::ConvertGemmaModel(tmp.string(), out.string())
                .ok());

    lsxcommon::gemma::GemmaPinnedWeights pw;
    auto rc = lsxcommon::gemma::LoadGemmaPinnedWeights(out.string(), pw,
                                                       nullptr);
    if (rc != cudaSuccess) {
        INFO("LoadGemmaPinnedWeights rc=" << cudaGetErrorString(rc));
        REQUIRE(rc == cudaSuccess);
    }
    REQUIRE(!pw.layers.empty());
    const auto &lw = pw.layers[0];   // sliding layer
    REQUIRE(lw.attn_nvfp4);
    REQUIRE(lw.attn_q.d_packed != nullptr);
    // sliding: q 4 heads x 16 dim (fixture) x H=64
    CHECK(lw.attn_q.rows ==
          gemmatest::TestCfg().num_attention_heads *
              gemmatest::TestCfg().head_dim);
    CHECK(lw.attn_q.cols == gemmatest::TestCfg().hidden_size);
    CHECK(lw.attn_o.cols == lw.attn_q.rows);
    CHECK(lw.d_q == nullptr);  // BF16 projections not uploaded
    // full-attention layer: global geometry
    const auto &fw = pw.layers[3];
    CHECK(fw.attn_q.rows ==
          gemmatest::TestCfg().num_attention_heads *
              gemmatest::TestCfg().global_head_dim);
    // full layers tie V to K (attention_k_eq_v): no v record
    CHECK(fw.attn_v.d_packed == nullptr);
    pw.Free();
    fs::remove_all(tmp);
}

TEST_CASE("gemma loader uploads NVFP4 tied embed/lm_head",
          "[gemma][loader][nvfp4]") {
    namespace fs = std::filesystem;
    auto tmp = fs::temp_directory_path() / "gemma4_loader_head_nvfp4";
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    auto tensors = gemmatest::BuildSyntheticGemmaModel();
    gemmatest::WriteSafetensors(tmp, "model.safetensors", tensors);
    gemmatest::WriteConfig(tmp);
    std::filesystem::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_gemma_tok.json",
                               tmp / "tokenizer.json");
    auto out = tmp / "out";
    REQUIRE(lsxcommon::gemma::ConvertGemmaModel(tmp.string(), out.string())
                .ok());
    CHECK(fs::exists(out / "pinned" / "global_nvfp4.arrow"));

    lsxcommon::gemma::GemmaPinnedWeights pw;
    auto rc = lsxcommon::gemma::LoadGemmaPinnedWeights(out.string(), pw,
                                                       nullptr);
    if (rc != cudaSuccess) {
        INFO("LoadGemmaPinnedWeights rc=" << cudaGetErrorString(rc));
        REQUIRE(rc == cudaSuccess);
    }
    REQUIRE(pw.embed_nvfp4.d_packed != nullptr);
    CHECK(pw.embed_nvfp4.rows == gemmatest::TestCfg().vocab_size);
    CHECK(pw.embed_nvfp4.cols == gemmatest::TestCfg().hidden_size);
    CHECK(pw.d_embed == nullptr);  // BF16 table not uploaded
    // head aliases the same packed buffer (tied embeddings)
    CHECK(pw.head_nvfp4.d_packed == pw.embed_nvfp4.d_packed);
    pw.Free();
    fs::remove_all(tmp);
}
