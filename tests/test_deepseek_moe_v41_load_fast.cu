// tests/test_deepseek_moe_v41_load_fast.cu — DeepSeek-V4.1-Flash load stage
// (CUDA): pinned dense upload (BF16/F32/FP8->FP16 dequant), expert-store LRU,
// and strictness on missing tensors. Synthetic converted fixture only.
#include <catch2/catch_test_macros.hpp>

#include "deepseek_moe_v41_load_internal.h"

#include "test_deepseek_moe_v41_fixture.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

TEST_CASE("deepseek-v41 pinned dense load uploads the synthetic layer",
          "[deepseek-v41][loader][fastload]")
{
    const auto base = fs::temp_directory_path() / "lsx_v41_load";
    fs::remove_all(base);
    const auto model = base / "model";
    REQUIRE(v41fixtest::BuildSyntheticConverted(model).ok());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    lsxcommon::deepseek_v41::gpu::PinnedWeights pw;
    REQUIRE(lsxcommon::deepseek_v41::gpu::load_pinned_weights(
                model.string(), pw, stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    CHECK(pw.vocab_size == 128);
    CHECK(pw.hidden == 64);
    CHECK(pw.layers.size() == 1);
    CHECK(pw.d_embed != nullptr);
    CHECK(pw.d_head != nullptr);
    CHECK(pw.d_final_norm != nullptr);
    CHECK(pw.resident_bytes > 0);

    auto &lw = pw.layers[0];
    CHECK(lw.d_wq_a != nullptr);
    CHECK(lw.d_wq_b != nullptr);
    CHECK(lw.d_wkv != nullptr);
    CHECK(lw.d_wo_a != nullptr);
    CHECK(lw.d_wo_b != nullptr);
    CHECK(lw.d_shared_w1 != nullptr);
    CHECK(lw.d_gate_weight != nullptr);
    CHECK(lw.d_attn_sink != nullptr);
    CHECK(lw.wq_a_rows == 64);
    CHECK(lw.wq_a_cols == 64);

    // MLA dense q_a stays FP8 on device (e4954d3): raw E4M3 0x38 (1.0) plus the
    // UE8M0 2^0 scale byte 127; the GEMV dequantizes on the fly.
    uint8_t f8 = 0, scale = 0;
    REQUIRE(cudaMemcpy(&f8, lw.d_wq_a, 1, cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(&scale, lw.d_wq_a_scale, 1, cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    CHECK(f8 == 0x38);
    CHECK(scale == 127);

    // Idempotent reload frees the previous set (no leak).
    REQUIRE(lsxcommon::deepseek_v41::gpu::load_pinned_weights(
                model.string(), pw, stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(pw.layers.size() == 1);

    pw.Free();
    cudaStreamDestroy(stream);
    fs::remove_all(base);
}

TEST_CASE("deepseek-v41 expert store decodes the synthetic experts",
          "[deepseek-v41][loader][fastload]")
{
    const auto base = fs::temp_directory_path() / "lsx_v41_experts";
    fs::remove_all(base);
    const auto model = base / "model";
    REQUIRE(v41fixtest::BuildSyntheticConverted(model).ok());

    auto db = lsxcommon::deepseek_v41::gpu::load_expert_index(
        model.string(), /*num_layers=*/1, /*experts_per_layer=*/2);
    REQUIRE(db.entries.size() == 2);

    lsxcommon::deepseek_v41::gpu::ensure_expert_resident(db, 0, 0, nullptr);
    CHECK(db.entries[0].host_w1 != nullptr);
    CHECK(db.entries[0].w1_bytes == 64);
    CHECK(db.entries[1].host_w1 != nullptr);

    lsxcommon::deepseek_v41::gpu::free_expert_index(db);
    fs::remove_all(base);
}

TEST_CASE("deepseek-v41 loader errors on a missing layer file",
          "[deepseek-v41][loader][strict]")
{
    const auto base = fs::temp_directory_path() / "lsx_v41_load_missing";
    fs::remove_all(base);
    const auto model = base / "model";
    REQUIRE(v41fixtest::BuildSyntheticConverted(model).ok());
    fs::remove(model / "pinned" / "layer_000.arrow");

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    lsxcommon::deepseek_v41::gpu::PinnedWeights pw;
    CHECK(lsxcommon::deepseek_v41::gpu::load_pinned_weights(
              model.string(), pw, stream) == cudaErrorInvalidValue);
    pw.Free();

    cudaStreamDestroy(stream);
    fs::remove_all(base);
}
