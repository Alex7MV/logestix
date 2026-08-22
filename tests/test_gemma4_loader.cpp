// Тест Gemma-4 загрузчика: бит-точность GPU BF16→FP16 in-place конверсии.
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <vector>

#include "gemma4_test_fixture.h"

#include "lsxcommon/gemma4_convert.h"
#include "lsxcommon/gemma4_inference_load.h"

TEST_CASE("GemmaBf16ToHalfInPlace bit-exact vs host", "[gemma][loader]") {
    // BF16 bit patterns: zeros, normals, subnormals, inf, nan, sign.
    const std::vector<uint16_t> bits = {
        0x0000, 0x8000, 0x3F80, 0xBF80, 0x4049, 0xC049,
        0x0080, 0x007F, 0x7F80, 0xFF80, 0x7FC0, 0x3C00, 0x42C8,
    };
    const int n = static_cast<int>(bits.size());

    uint16_t *d_buf = nullptr;
    REQUIRE(cudaMalloc(&d_buf, n * sizeof(uint16_t)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_buf, bits.data(), n * sizeof(uint16_t),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    auto rc = lsxcommon::gemma::GemmaBf16ToHalfInPlace(
        reinterpret_cast<uint16_t *>(d_buf), n, nullptr);
    REQUIRE(rc == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(nullptr) == cudaSuccess);

    std::vector<uint16_t> got(n);
    REQUIRE(cudaMemcpy(got.data(), d_buf, n * sizeof(uint16_t),
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    for (int i = 0; i < n; ++i) {
        // Host reference: nearest-even via float round-trip — the same math
        // GlmBf16ToHalfInPlace is held to.
        uint32_t wide = static_cast<uint32_t>(bits[i]) << 16;
        float f;
        std::memcpy(&f, &wide, 4);
        __half h = __float2half(f);
        uint16_t want;
        std::memcpy(&want, &h, 2);
        INFO("bf16 0x" << std::hex << bits[i] << " got 0x" << got[i]);
        CHECK(got[i] == want);
    }
    REQUIRE(cudaFree(d_buf) == cudaSuccess);
}

TEST_CASE("GemmaBf16ToHalfInPlace rejects null/misaligned input",
          "[gemma][loader]") {
    CHECK(lsxcommon::gemma::GemmaBf16ToHalfInPlace(nullptr, 8, nullptr) ==
          cudaErrorInvalidValue);

    uint16_t *p = nullptr;
    REQUIRE(cudaMalloc(&p, 64 * sizeof(uint16_t)) == cudaSuccess);
    // 2-byte-aligned but not 16-byte-aligned buffer must be rejected.
    uintptr_t addr = reinterpret_cast<uintptr_t>(p);
    uintptr_t mis = (addr % 16 == 0) ? addr + 2 : addr;
    CHECK(lsxcommon::gemma::GemmaBf16ToHalfInPlace(
              reinterpret_cast<uint16_t *>(mis), 8, nullptr) ==
          cudaErrorInvalidValue);
    REQUIRE(cudaFree(p) == cudaSuccess);
}

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
