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
        // F32 side allocations are separate from the arena.
        CHECK(lw.d_layer_scalar != nullptr);
        CHECK(lw.d_per_expert_scale != nullptr);
        CHECK(lw.d_router_scale != nullptr);
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

