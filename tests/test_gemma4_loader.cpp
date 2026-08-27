// Тест Gemma-4 загрузчика: BF16 raw-passthrough (без конверсии) + арена + LRU.
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <vector>

#include "gemma4_test_fixture.h"

#include "lsxcommon/gemma4_convert.h"
#include "lsxcommon/gemma4_inference_load.h"

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
    std::filesystem::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
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
    CHECK(pw.d_embed != nullptr);
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
        for (auto *p : {lw.d_input_norm, lw.d_post_norm, lw.d_q, lw.d_o,
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
            CHECK(ent.w1_bytes == static_cast<int64_t>(16 * 64 / 2));
            CHECK(ent.host_w3 != nullptr);
            CHECK(ent.w3_bytes == static_cast<int64_t>(16 * 64 / 2));
            CHECK(ent.host_w2 != nullptr);
            CHECK(ent.w2_bytes == static_cast<int64_t>(64 * 16 / 2));
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
// free VRAM with a 6 GiB headroom reserve; explicit env override wins.
TEST_CASE("gemma resident slot budget: VRAM-bounded, clamped, env-override",
          "[gemma][loader][cache]")
{
    using lsxcommon::gemma::ComputeGemmaResidentSlots;
    const size_t MiB = 1024ull * 1024ull;
    const size_t per_expert = 5 * MiB;

    // Ample VRAM: every expert resident.
    CHECK(ComputeGemmaResidentSlots(32ull * 1024 * MiB, per_expert, 3840)
          == 3840);
    // Exactly the pool fits: boundary inclusive.
    CHECK(ComputeGemmaResidentSlots(6144ull * MiB + 3840ull * per_expert,
                                    per_expert, 3840) == 3840);
    // Tight: (free - 6 GiB) / per_expert slots.
    CHECK(ComputeGemmaResidentSlots(6144ull * MiB + 1000ull * per_expert,
                                    per_expert, 3840) == 1000);
    // Under-reserve floor: nothing fits -> caller applies its own fallback.
    CHECK(ComputeGemmaResidentSlots(2ull * 1024 * MiB, per_expert, 3840) == 0);
    // Env override wins and is clamped to the pool.
    CHECK(ComputeGemmaResidentSlots(32ull * 1024 * MiB, per_expert, 3840,
                                    512) == 512);
    CHECK(ComputeGemmaResidentSlots(32ull * 1024 * MiB, per_expert, 3840,
                                    9999) == 3840);
}
